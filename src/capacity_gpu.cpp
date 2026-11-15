#include "hbflab/capacity_gpu.hpp"
#include <cuda.h>
#include <atomic>
#include <chrono>
#include <map>
#include <thread>

namespace hbflab {
namespace {
std::atomic<std::uint64_t> next_gpu{1};
using Clock=std::chrono::steady_clock;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void driver(CUresult r){if(r!=CUDA_SUCCESS)throw std::runtime_error("capacity CUDA context/stream error");}
}
struct CapacityGpu::Impl {
    struct Held {PageLease host;PageInfo info;cudaStream_t stream;bool mirror_current;};
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
    bool poisoned=false,shared=false;
    explicit Impl(CapacityOptions o,CapacityMedia* media):store(o,media),mirrors(store.frame_count()),shared(media!=nullptr) {
        check(cudaFree(nullptr));driver(cuCtxGetCurrent(&context));
        check(cudaMalloc(reinterpret_cast<void**>(&device),store.frame_count()*4096));
    }
    ~Impl() {
        if(!device)return;
        CUcontext old=nullptr;cuCtxGetCurrent(&old);
        if(cuCtxSetCurrent(context)==CUDA_SUCCESS){cudaDeviceSynchronize();cudaFree(device);cuCtxSetCurrent(old);}
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
CapacityGpu::CapacityGpu(CapacityOptions o,CapacityMedia* media):impl_(std::make_unique<Impl>(o,media)){}
CapacityGpu::~CapacityGpu()=default;
CapacityStore& CapacityGpu::store(){impl_->current();return impl_->store;}
GpuPageLease CapacityGpu::acquire(Allocation a,std::size_t page,PageAccess access,cudaStream_t stream) {
    return acquire_impl(a,page,access,stream,false);
}
GpuPageLease CapacityGpu::acquire_read_device(Allocation a,std::size_t page,const void* source,cudaStream_t stream) {
    if(!source)throw std::invalid_argument("null GPU read replica");
    return acquire_impl(a,page,PageAccess::Read,stream,false,source);
}
GpuPageLease CapacityGpu::acquire_impl(Allocation a,std::size_t page,PageAccess access,cudaStream_t stream,bool overwrite,const void* read_source) {
    auto& s=*impl_;s.current();
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
const void* CapacityGpu::data(GpuPageLease l) const {auto& s=*impl_;return s.device+s.get(l).info.frame*4096;}
void* CapacityGpu::writable_data(GpuPageLease l) const {
    auto& s=*impl_;const auto& h=s.get(l);if(h.info.access!=PageAccess::Write)throw std::invalid_argument("read-only GPU lease");
    return s.device+h.info.frame*4096;
}
std::size_t CapacityGpu::bytes(GpuPageLease l) const {return impl_->get(l).info.bytes;}
void CapacityGpu::finish(GpuPageLease l) {
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
void CapacityGpu::flush(){auto& s=*impl_;s.current();const auto start=Clock::now();const auto before=s.store.model_time_ps();s.store.flush_all();s.pace(start,before);}
std::size_t CapacityGpu::prefetch(Allocation a,std::size_t first,std::size_t count){
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
CapacityGpuStats CapacityGpu::statistics() const {impl_->current();auto stats=impl_->stats;stats.device_bytes=device_bytes();stats.active_leases=impl_->held.size();return stats;}
}
