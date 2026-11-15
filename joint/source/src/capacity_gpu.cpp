#include "hbflab/capacity_gpu.hpp"
#include <cuda.h>
#include <atomic>
#include <chrono>
#include <map>
#include <thread>
#include <cstring>
#include <algorithm>

namespace hbflab {
namespace {
std::atomic<std::uint64_t> next_gpu{1};
using Clock=std::chrono::steady_clock;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void driver(CUresult r){if(r!=CUDA_SUCCESS)throw std::runtime_error("capacity CUDA context/stream error");}
}
struct CapacityGpu::Impl {
    struct Held {PageLease host;PageInfo info;cudaStream_t stream;bool mirror_current;
        bool asynchronous=false,speculative=false;std::size_t slot=0;
        Allocation allocation{};std::size_t page=0;GpuLeaseState state=GpuLeaseState::Preparing;};
    CapacityStore store;
    unsigned char* device=nullptr;
    CUcontext context=nullptr;
    const std::thread::id thread=std::this_thread::get_id();
    const std::uint64_t identity=next_gpu.fetch_add(1);
    std::uint64_t serial=0;
    std::map<std::uint64_t,Held> held;
    struct Mirror {std::uint64_t generation=0,revision=0;};
    std::vector<Mirror> mirrors;
    CapacityGpuStats stats;
    struct Slot {bool busy=false;cudaEvent_t ready=nullptr,consumer=nullptr,complete=nullptr;};
    std::vector<Slot> slots;unsigned char* staging=nullptr;cudaStream_t upload_stream=nullptr,download_stream=nullptr;
    CapacityGpuAsyncOptions async;
    bool poisoned=false,shared=false,closed=false;
    explicit Impl(CapacityOptions o,CapacityMedia* media,CapacityGpuAsyncOptions a):store(o,media),mirrors(store.frame_count()),async(a),shared(media!=nullptr) {
        if(a.staging_pages>o.frames || (a.staging_pages && (!a.demand_reserve || a.demand_reserve>a.staging_pages)))
            throw std::invalid_argument("invalid bounded async staging/reserve");
        check(cudaFree(nullptr));driver(cuCtxGetCurrent(&context));
        try{
            check(cudaMalloc(reinterpret_cast<void**>(&device),store.frame_count()*4096));
            if(a.staging_pages){
                slots.resize(a.staging_pages);check(cudaMallocHost(reinterpret_cast<void**>(&staging),slots.size()*4096));
                check(cudaStreamCreateWithFlags(&upload_stream,cudaStreamNonBlocking));check(cudaStreamCreateWithFlags(&download_stream,cudaStreamNonBlocking));
                for(auto& slot:slots)for(auto* event:{&slot.ready,&slot.consumer,&slot.complete})check(cudaEventCreateWithFlags(event,cudaEventDisableTiming));
            }
        }catch(...){cleanup();throw;}
    }
    void cleanup() noexcept {
        for(auto& slot:slots)for(auto event:{slot.ready,slot.consumer,slot.complete})if(event)cudaEventDestroy(event);
        if(upload_stream)cudaStreamDestroy(upload_stream);if(download_stream)cudaStreamDestroy(download_stream);
        if(staging)cudaFreeHost(staging);if(device)cudaFree(device);device=nullptr;staging=nullptr;
    }
    ~Impl() {
        if(!device)return;
        CUcontext old=nullptr;cuCtxGetCurrent(&old);
        if(cuCtxSetCurrent(context)==CUDA_SUCCESS){cudaDeviceSynchronize();cleanup();cuCtxSetCurrent(old);}
    }
    void current() const {
        if(thread!=std::this_thread::get_id())throw std::logic_error("GPU capacity host owner changed");
        CUcontext now=nullptr;driver(cuCtxGetCurrent(&now));
        if(now!=context)throw std::logic_error("GPU capacity context changed");
        if(poisoned)throw std::runtime_error("GPU capacity session poisoned");
    }
    Held& get(GpuPageLease l) {
        current();auto it=held.find(l.value);
        if(l.owner!=identity || it==held.end())throw std::invalid_argument("foreign/stale GPU page lease");return it->second;
    }
    void open() const {current();if(closed)throw std::logic_error("GPU capacity closed");}
    void stream_check(cudaStream_t stream) const {
        CUcontext c=context;if(stream)driver(cuStreamGetCtx(reinterpret_cast<CUstream>(stream),&c));
        if(c!=context)throw std::invalid_argument("capacity stream context mismatch");
        cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(stream,&capture));
        if(capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("capacity graph capture unsupported");
    }
    void fatal(){poisoned=true;store.abort();}
    void upload(PageLease host,const PageInfo& info,cudaStream_t stream,bool speculative=false) {
        auto& mirror=mirrors[info.frame];
        if(mirror.generation==info.generation && mirror.revision==info.revision){++stats.mirror_hits;return;}
        std::array<unsigned char,4096> staging{};store.read_pinned(host,0,staging.data(),info.bytes);
        const auto copy_start=Clock::now();
        check(cudaMemcpyAsync(device+info.frame*4096,staging.data(),4096,cudaMemcpyHostToDevice,stream));
        check(cudaStreamSynchronize(stream));
        stats.copy_wall_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-copy_start).count();
        ++stats.uploads;stats.h2d_bytes+=4096;if(speculative)++stats.prefetch_uploads;
        mirror={info.generation,info.revision};
    }
    void pace(Clock::time_point start,std::uint64_t before) {
        if(shared)return; // Shared owner already waits this request's completion;
                          // unrelated GPU work must not be charged a second time.
        const auto now=store.model_time_ps();
        const auto ns=(now-before)/1000+((now-before)%1000!=0);
        if(ns>static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()))throw std::overflow_error("capacity wait overflow");
        std::this_thread::sleep_until(start+std::chrono::nanoseconds(ns));
    }
};
CapacityGpu::CapacityGpu(CapacityOptions o,CapacityMedia* media,CapacityGpuAsyncOptions a):impl_(std::make_unique<Impl>(o,media,a)){}
CapacityGpu::~CapacityGpu(){if(impl_->async.staging_pages && !impl_->closed)try{drain();}catch(...){} }
CapacityStore& CapacityGpu::store(){impl_->current();return impl_->store;}
GpuPageLease CapacityGpu::acquire(Allocation a,std::size_t page,PageAccess access,cudaStream_t stream) {
    auto& s=*impl_;s.open();s.stream_check(stream);
    if(async_enabled())return acquire_async(a,page,access,stream);
    return acquire_impl(a,page,access,stream,false);
}
GpuPageLease CapacityGpu::acquire_async(Allocation a,std::size_t page,PageAccess access,cudaStream_t stream,const void* source,cudaEvent_t source_ready) {
        auto& s=*impl_;
        poll();
        for(auto& item:s.held){auto& h=item.second;
            if(h.asynchronous && h.speculative && h.allocation.store==a.store && h.allocation.value==a.value && h.page==page && access==PageAccess::Read){
                if(source && source_ready)check(cudaStreamWaitEvent(stream,source_ready,0));
                h.speculative=false;++s.stats.prefetch_used;GpuPageLease l{s.identity,item.first};use(l,stream);return l;
            }
        }
        for(;;){
            try{auto l=prepare_impl(a,page,access,false,source,source_ready);use(l,stream);return l;}
            catch(const CapacityBusy&){
                GpuPageLease candidate{};bool discard=false;
                for(const auto& item:s.held)if(item.second.asynchronous && item.second.speculative){candidate={s.identity,item.first};discard=true;break;}
                if(!candidate.value)for(const auto& item:s.held)if(item.second.asynchronous &&
                    (item.second.state==GpuLeaseState::Writeback || item.second.state==GpuLeaseState::Retiring)){candidate={s.identity,item.first};break;}
                if(!candidate.value)throw;
                ++s.stats.pressure_waits;if(discard)cancel(candidate);else wait(candidate);
            }
        }
}
GpuPageLease CapacityGpu::acquire_read_device_async(Allocation a,std::size_t page,const void* source,cudaEvent_t source_ready,cudaStream_t stream) {
    auto& s=*impl_;s.open();s.stream_check(stream);
    if(!async_enabled() || !source)throw std::invalid_argument("async replica requires staging and a source");
    return acquire_async(a,page,PageAccess::Read,stream,source,source_ready);
}
GpuPageLease CapacityGpu::acquire_read_device(Allocation a,std::size_t page,const void* source,cudaStream_t stream) {
    if(!source)throw std::invalid_argument("null GPU read replica");
    // Async mode reads the authoritative host page; the optional short-lived
    // replica cannot be borrowed beyond this call's original lifetime contract.
    if(async_enabled())return acquire(a,page,PageAccess::Read,stream);
    return acquire_impl(a,page,PageAccess::Read,stream,false,source);
}
GpuPageLease CapacityGpu::acquire_impl(Allocation a,std::size_t page,PageAccess access,cudaStream_t stream,bool overwrite,const void* read_source) {
    auto& s=*impl_;s.open();
    // Legacy replica/write helpers may coexist with asynchronous leases, but
    // cannot overwrite a frame still consumed by any asynchronous reader.
    if(async_enabled()){
        for(const auto& item:s.held)if(item.second.asynchronous && !item.second.speculative &&
            (item.second.state==GpuLeaseState::InUse || item.second.state==GpuLeaseState::Preparing || item.second.state==GpuLeaseState::Ready))
            throw CapacityBusy("synchronous helper requires finished async consumers");
        drain();
    }
    CUcontext context=s.context;
    if(stream)driver(cuStreamGetCtx(reinterpret_cast<CUstream>(stream),&context));
    if(context!=s.context)throw std::invalid_argument("capacity stream context mismatch");
    cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("capacity graph capture unsupported");
    check(cudaStreamSynchronize(stream));
    const auto start=Clock::now();const auto before=s.store.model_time_ps();
    const auto host=s.store.pin(a,page,access);const auto info=s.store.info(host);
    try {
        auto& mirror=s.mirrors[info.frame];
        bool current=mirror.generation==info.generation && mirror.revision==info.revision;
        if(overwrite) ++s.stats.overwrite_uploads_skipped;
        else if(read_source && !current) {
            const auto copy_start=Clock::now();
            check(cudaMemcpyAsync(s.device+info.frame*4096,read_source,info.bytes,cudaMemcpyDeviceToDevice,stream));
            if(info.bytes<4096)check(cudaMemsetAsync(s.device+info.frame*4096+info.bytes,0,4096-info.bytes,stream));
            check(cudaStreamSynchronize(stream));
            s.stats.copy_wall_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-copy_start).count();
            s.stats.read_source_d2d_bytes+=info.bytes;++s.stats.read_source_fills;
            mirror={info.generation,info.revision};current=true;
        }
        else s.upload(host,info,stream);
        s.pace(start,before);const auto id=s.serial=hbf_eventlab::checked_add(s.serial,1);s.held.emplace(id,Impl::Held{host,info,stream,current || !overwrite});return {s.identity,id};
    } catch(...) {cudaStreamSynchronize(stream);s.store.unpin(host);s.poisoned=true;s.store.abort();throw;}
}
const void* CapacityGpu::data(GpuPageLease l) const {auto& s=*impl_;const auto& h=s.get(l);
    if(h.asynchronous && h.state!=GpuLeaseState::InUse)throw std::logic_error("GPU data requires use dependency");
    return s.device+h.info.frame*4096;}
void* CapacityGpu::writable_data(GpuPageLease l) const {
    auto& s=*impl_;const auto& h=s.get(l);if(h.info.access!=PageAccess::Write)throw std::invalid_argument("read-only GPU lease");
    return const_cast<void*>(data(l));
}
std::size_t CapacityGpu::bytes(GpuPageLease l) const {return impl_->get(l).info.bytes;}
void CapacityGpu::finish(GpuPageLease l) {
    if(impl_->get(l).asynchronous){finish_async(l);wait(l);return;}
    finish_range(l,0,impl_->get(l).info.bytes);
}
void CapacityGpu::finish_range(GpuPageLease l,std::size_t offset,std::size_t bytes) {
    auto& s=*impl_;const auto h=s.get(l);
    if(offset>h.info.bytes || bytes>h.info.bytes-offset)throw std::invalid_argument("GPU finish range");
    try {
        check(cudaStreamSynchronize(h.stream));
        if(h.info.access==PageAccess::Write) {
            std::array<unsigned char,4096> staging{};
            const auto copy_start=Clock::now();
            check(cudaMemcpy(staging.data(),s.device+h.info.frame*4096+offset,bytes,cudaMemcpyDeviceToHost));
            s.stats.copy_wall_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-copy_start).count();
            ++s.stats.downloads;s.stats.d2h_bytes+=bytes;
            s.store.write_pinned(h.host,offset,staging.data(),bytes);
            const auto latest=s.store.info(h.host);
            if(h.mirror_current || (offset==0 && bytes==h.info.bytes))s.mirrors[h.info.frame]={latest.generation,latest.revision};
            else s.mirrors[h.info.frame]={};
        }
        s.store.unpin(h.host);s.held.erase(l.value);
    } catch(...){s.poisoned=true;s.store.abort();throw;}
}
void CapacityGpu::write_device(Allocation allocation, std::size_t offset, const void* source,
                               std::size_t bytes, cudaStream_t stream) {
    auto& s = *impl_;
    s.current();
    const auto size = s.store.size(allocation);
    if ((!source && bytes) || offset > size || bytes > size - offset)
        throw std::invalid_argument("GPU write source/range");
    for (std::size_t at = 0; at < bytes;) {
        const auto position = offset + at;
        const auto page = position / 4096;
        const auto within = position % 4096;
        const auto valid = (std::min)(std::size_t{4096}, size - page * 4096);
        const auto count = (std::min)(bytes - at, valid - within);
        const auto lease = acquire_impl(allocation,page,PageAccess::Write,stream,true);
        try {
            check(cudaMemcpyAsync(static_cast<unsigned char*>(writable_data(lease)) + within,
                                  static_cast<const unsigned char*>(source) + at,count,cudaMemcpyDeviceToDevice,stream));
            s.stats.d2d_bytes += count;
            finish_range(lease,within,count);
        } catch (...) {
            cudaStreamSynchronize(stream);
            // Failure is terminal; release the borrowed page without publishing
            // a partially copied GPU value to the host authority.
            auto held = s.held.find(lease.value);
            if (held != s.held.end()) { s.store.unpin(held->second.host); s.held.erase(held); }
            s.poisoned = true;
            s.store.abort();
            throw;
        }
        at += count;
    }
}
void CapacityGpu::flush(){auto& s=*impl_;s.current();if(async_enabled())drain();const auto start=Clock::now();const auto before=s.store.model_time_ps();s.store.flush_all();s.pace(start,before);}
std::size_t CapacityGpu::prefetch(Allocation a,std::size_t first,std::size_t count){
    if(async_enabled()){std::size_t n=0;for(std::size_t i=0;i<count;++i)n+=try_prefetch(a,first+i);return n;}
    auto& s=*impl_;s.current();
    cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(nullptr,&capture));
    if(capture!=cudaStreamCaptureStatusNone)throw std::invalid_argument("prefetch graph capture unsupported");
    const auto start=Clock::now();const auto before=s.store.model_time_ps();const auto n=s.store.prefetch(a,first,count);
    for(std::size_t page=first;page<first+count;++page)if(const auto host=s.store.pin_prefetched(a,page)) {
        try{s.upload(*host,s.store.info(*host),nullptr,true);s.store.unpin(*host);}
        catch(...){cudaStreamSynchronize(nullptr);s.store.unpin(*host);s.poisoned=true;s.store.abort();throw;}
    }
    s.pace(start,before);return n;
}
std::size_t CapacityGpu::maintain(double watts){auto& s=*impl_;s.current();const auto start=Clock::now();const auto before=s.store.model_time_ps();const auto n=s.store.maintain(watts);s.pace(start,before);return n;}
std::size_t CapacityGpu::device_bytes() const {impl_->current();return impl_->store.frame_count()*4096;}
CapacityGpuStats CapacityGpu::statistics() const {impl_->current();auto stats=impl_->stats;stats.device_bytes=device_bytes();stats.active_leases=impl_->held.size();stats.pinned_staging_bytes=impl_->staging?impl_->slots.size()*4096:0;return stats;}
bool CapacityGpu::async_enabled() const{return impl_->async.staging_pages!=0;}
GpuPageLease CapacityGpu::prepare(Allocation a,std::size_t page,PageAccess access){return prepare_impl(a,page,access,false);}
GpuPageLease CapacityGpu::prepare_impl(Allocation a,std::size_t page,PageAccess access,bool speculative,const void* source,cudaEvent_t source_ready){
    auto& s=*impl_;s.open();if(!async_enabled())throw std::logic_error("async staging disabled");
    poll();std::size_t slot=0;while(slot<s.slots.size() && s.slots[slot].busy)++slot;
    if(slot==s.slots.size())throw CapacityBusy("async staging budget exhausted");
    const auto start=Clock::now();const auto before=s.store.model_time_ps();
    const auto host=s.store.pin(a,page,access);const auto info=s.store.info(host);
    for(const auto& item:s.held)if(item.second.info.frame==info.frame){s.store.unpin(host);throw CapacityBusy("frame already leased to GPU");}
    // Insert metadata before enqueueing DMA so allocation failure cannot orphan work.
    const auto id=s.serial=hbf_eventlab::checked_add(s.serial,1);
    try{s.held.emplace(id,Impl::Held{host,info,nullptr,true,true,speculative,slot,a,page,GpuLeaseState::Preparing});}
    catch(...){s.store.unpin(host);throw;}
    s.slots[slot].busy=true;
    try{
        auto& mirror=s.mirrors[info.frame];auto& events=s.slots[slot];
        s.pace(start,before); // Media completed before DMA readiness may be published.
        // Also honor source readiness on a mirror hit: retirement is then a
        // safe lifetime boundary for the supplied producer/source pair.
        if(source && source_ready)check(cudaStreamWaitEvent(s.upload_stream,source_ready,0));
        if(mirror.generation==info.generation && mirror.revision==info.revision)++s.stats.mirror_hits;
        else if(source){
            check(cudaMemcpyAsync(s.device+info.frame*4096,source,info.bytes,cudaMemcpyDeviceToDevice,s.upload_stream));
            if(info.bytes<4096)check(cudaMemsetAsync(s.device+info.frame*4096+info.bytes,0,4096-info.bytes,s.upload_stream));
            s.stats.read_source_d2d_bytes+=info.bytes;++s.stats.read_source_fills;
        }else{
            auto* buffer=s.staging+slot*4096;std::memset(buffer,0,4096);s.store.read_pinned(host,0,buffer,info.bytes);
            check(cudaMemcpyAsync(s.device+info.frame*4096,buffer,4096,cudaMemcpyHostToDevice,s.upload_stream));
            ++s.stats.uploads;s.stats.h2d_bytes+=4096;
        }
        check(cudaEventRecord(events.ready,s.upload_stream));++s.stats.async_prepared;
        s.stats.peak_inflight=(std::max)(s.stats.peak_inflight,s.held.size());return {s.identity,id};
    }catch(...){cudaStreamSynchronize(s.upload_stream);s.store.unpin(host);s.held.erase(id);s.slots[slot].busy=false;s.fatal();throw;}
}
GpuLeaseState CapacityGpu::lease_state(GpuPageLease l){
    auto& s=*impl_;auto& h=s.get(l);if(!h.asynchronous)throw std::logic_error("synchronous lease state");
    if(h.state==GpuLeaseState::Preparing){auto r=cudaEventQuery(s.slots[h.slot].ready);if(r==cudaSuccess)h.state=GpuLeaseState::Ready;else if(r!=cudaErrorNotReady){s.fatal();check(r);}}
    return h.state;
}
void CapacityGpu::use(GpuPageLease l,cudaStream_t stream){
    auto& s=*impl_;auto& h=s.get(l);s.stream_check(stream);
    if(!h.asynchronous || (h.state!=GpuLeaseState::Preparing && h.state!=GpuLeaseState::Ready))throw std::logic_error("lease already used or retired");
    try{check(cudaStreamWaitEvent(stream,s.slots[h.slot].ready,0));h.stream=stream;h.state=GpuLeaseState::InUse;}
    catch(...){s.fatal();throw;}
}
void CapacityGpu::finish_async(GpuPageLease l){
    auto& s=*impl_;auto& h=s.get(l);
    if(!h.asynchronous || h.state!=GpuLeaseState::InUse)throw std::logic_error("finish requires in-use async lease");
    try{
        auto& slot=s.slots[h.slot];check(cudaEventRecord(slot.consumer,h.stream));
        if(h.info.access==PageAccess::Write){
            check(cudaStreamWaitEvent(s.download_stream,slot.consumer,0));
            check(cudaMemcpyAsync(s.staging+h.slot*4096,s.device+h.info.frame*4096,h.info.bytes,cudaMemcpyDeviceToHost,s.download_stream));
            check(cudaEventRecord(slot.complete,s.download_stream));h.state=GpuLeaseState::Writeback;
        }else h.state=GpuLeaseState::Retiring;
    }catch(...){s.fatal();throw;}
}
void CapacityGpu::retire(GpuPageLease l){
    auto& s=*impl_;const auto h=s.get(l);
    try{
        if(h.state==GpuLeaseState::Writeback){s.store.write_pinned(h.host,0,s.staging+h.slot*4096,h.info.bytes);++s.stats.downloads;s.stats.d2h_bytes+=h.info.bytes;}
        const auto latest=s.store.info(h.host);s.mirrors[h.info.frame]={latest.generation,latest.revision};
        s.store.unpin(h.host);s.slots[h.slot].busy=false;s.held.erase(l.value);++s.stats.async_retired;
    }catch(...){s.fatal();throw;}
}
void CapacityGpu::wait(GpuPageLease l){
    auto& s=*impl_;const auto& h=s.get(l);
    if(!h.asynchronous || (h.state!=GpuLeaseState::Writeback && h.state!=GpuLeaseState::Retiring))throw std::logic_error("wait requires submitted completion");
    const auto& slot=s.slots[h.slot];try{check(cudaEventSynchronize(h.state==GpuLeaseState::Writeback?slot.complete:slot.consumer));retire(l);}catch(...){s.fatal();throw;}
}
std::size_t CapacityGpu::poll(){
    auto& s=*impl_;s.current();std::size_t n=0;
    for(auto it=s.held.begin();it!=s.held.end();){const auto id=it->first;const auto& h=(it++)->second;
        if(!h.asynchronous || (h.state!=GpuLeaseState::Writeback && h.state!=GpuLeaseState::Retiring))continue;
        const auto& slot=s.slots[h.slot];auto r=cudaEventQuery(h.state==GpuLeaseState::Writeback?slot.complete:slot.consumer);
        if(r==cudaSuccess){retire({s.identity,id});++n;}else if(r!=cudaErrorNotReady){s.fatal();check(r);}
    }return n;
}
void CapacityGpu::cancel(GpuPageLease l){
    auto& s=*impl_;auto& h=s.get(l);
    if(!h.asynchronous || (h.state!=GpuLeaseState::Preparing && h.state!=GpuLeaseState::Ready))throw std::logic_error("cannot cancel a consumer; finish it first");
    try{check(cudaEventSynchronize(s.slots[h.slot].ready));if(h.speculative)++s.stats.prefetch_cancelled;retire(l);}catch(...){s.fatal();throw;}
}
bool CapacityGpu::try_prefetch(Allocation a,std::size_t page){
    return try_prefetch_impl(a,page,nullptr,nullptr);
}
bool CapacityGpu::try_prefetch_device(Allocation a,std::size_t page,const void* source,cudaEvent_t source_ready){
    if(!source)throw std::invalid_argument("null async prefetch replica");
    return try_prefetch_impl(a,page,source,source_ready);
}
bool CapacityGpu::try_prefetch_impl(Allocation a,std::size_t page,const void* source,cudaEvent_t source_ready){
    auto& s=*impl_;s.open();if(!async_enabled())return false;poll();
    for(const auto& item:s.held){const auto& h=item.second;if(h.allocation.store==a.store && h.allocation.value==a.value && h.page==page)return false;}
    const auto free=std::count_if(s.slots.begin(),s.slots.end(),[](const auto& slot){return !slot.busy;});
    if(static_cast<std::size_t>(free)<=s.async.demand_reserve)return false;
    try{prepare_impl(a,page,PageAccess::Read,true,source,source_ready);++s.stats.prefetch_issued;return true;}catch(const CapacityBusy&){return false;}
}
void CapacityGpu::drain(){
    auto& s=*impl_;s.current();while(!s.held.empty()){
        auto it=s.held.begin();GpuPageLease l{s.identity,it->first};const auto h=it->second;
        if(!h.asynchronous)finish(l);
        else if(h.state==GpuLeaseState::Preparing || h.state==GpuLeaseState::Ready)cancel(l);
        else{if(h.state==GpuLeaseState::InUse)finish_async(l);wait(l);}
    }
}
void CapacityGpu::close(){auto& s=*impl_;s.current();if(s.closed)return;drain();flush();if(s.staging){check(cudaFreeHost(s.staging));s.staging=nullptr;}s.closed=true;}
}
