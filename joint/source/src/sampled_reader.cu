#include "hbflab/sampled_reader.hpp"
#include "hbflab/async_device.cuh"
#include "hbflab/async_module.hpp"
#include "hbflab/ptx.hpp"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

using json=nlohmann::ordered_json;
namespace hbflab {
namespace {
constexpr unsigned chunk_capacity=262144;
void need(bool b,const char *message){if(!b)throw std::runtime_error(message);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<bool Payload>
__global__ void sampled_window(AsyncDeviceView view,const unsigned char *input,unsigned bytes,unsigned unit,
                               unsigned char *mirror,unsigned long long *times){
    const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,warp=tid/32,lane=tid%32,warps=gridDim.x*blockDim.x/32;
    const unsigned groups=(bytes+32*unit-1)/(32*unit);
    for(unsigned q=warp;q<groups;q+=warps){
        const unsigned offset=q*32*unit+lane*unit;
        const bool valid=offset<bytes;
        const unsigned mask=__ballot_sync(0xffffffffu,valid);
        const auto before=gpu_now_ns();
        Status status=Status::Ok;
        if(valid)status=AsyncDeviceClient(view).gate_load_adaptive(input+offset,unit,mask);
        if(__any_sync(0xffffffffu,status!=Status::Ok))return;
        if(valid){
            if constexpr(Payload){
                if(unit==16){
                    const auto *from=reinterpret_cast<const volatile unsigned *>(input+offset);
                    const uint4 value=make_uint4(from[0],from[1],from[2],from[3]);
                    *reinterpret_cast<uint4 *>(mirror+offset)=value;
                }else{
                    const auto value=*reinterpret_cast<const volatile unsigned short *>(input+offset);
                    *reinterpret_cast<unsigned short *>(mirror+offset)=value;
                }
            }
            times[2*(offset/unit)]=before;times[2*(offset/unit)+1]=gpu_now_ns();
        }
        __syncwarp();
    }
}
__global__ void sampled_stamp(unsigned long long *output){*output=gpu_now_ns();}
std::string ptx_copy(unsigned unit){
    std::ostringstream text;
    text<<R"(.version 8.0
.target sm_75
.address_size 64
.visible .entry sampled_copy(
 .param .u64 input, .param .u64 output, .param .u64 times, .param .u32 bytes
) {
 .reg .pred %valid;
 .reg .b32 %thread_index, %block_index, %thread_count, %index, %offset, %n, %a, %b, %c, %d;
 .reg .b64 %src, %dst, %timer, %wide, %timeoff, %before, %after;
 ld.param.u64 %src, [input];
 ld.param.u64 %dst, [output];
 ld.param.u64 %timer, [times];
 ld.param.u32 %n, [bytes];
 mov.u32 %thread_index, %tid.x;
 mov.u32 %block_index, %ctaid.x;
 mov.u32 %thread_count, %ntid.x;
 mad.lo.u32 %index, %block_index, %thread_count, %thread_index;
)";
    text<<" mul.lo.u32 %offset, %index, "<<unit<<";\n";
    text<<R"( cvt.u64.u32 %wide, %offset;
 add.u64 %src, %src, %wide;
 add.u64 %dst, %dst, %wide;
 setp.lt.u32 %valid, %offset, %n;
 mov.u64 %before, %globaltimer;
)";
    if(unit==16)text<<" @%valid ld.global.v4.u32 {%a,%b,%c,%d}, [%src];\n";
    else text<<" @%valid ld.global.u16 %a, [%src];\n";
    text<<" mov.u64 %after, %globaltimer;\n";
    if(unit==16)text<<" @%valid st.global.v4.u32 [%dst], {%a,%b,%c,%d};\n";
    else text<<" @%valid st.global.u16 [%dst], %a;\n";
    text<<R"( mul.wide.u32 %timeoff, %index, 16;
 add.u64 %timer, %timer, %timeoff;
 @%valid st.global.v2.u64 [%timer], {%before,%after};
 ret;
}
)";
    return text.str();
}
struct RequestRecord {
    std::uint64_t launch,ticket,offset,logical,arrival_ns,arrival_ps,end_ps,first_ns,last_ns,time_file_offset,grant_floor_ns;
    std::uint32_t range,bytes,producer,unit;
};
static_assert(sizeof(RequestRecord)==104,"request record ABI");
std::string hash_bytes(const std::vector<unsigned char> &bytes,unsigned count){
    return sha256(std::string_view(reinterpret_cast<const char *>(bytes.data()),count));
}
}
struct SampledReader::Impl {
    SampledRegistration &registration;LiveTimingService &service;SampledRoute route;
    std::filesystem::path output;std::thread::id owner=std::this_thread::get_id();
    unsigned char *slab=nullptr;unsigned long long *times=nullptr;
    std::vector<unsigned char> reference=std::vector<unsigned char>(chunk_capacity),returned=std::vector<unsigned char>(chunk_capacity);
    std::vector<unsigned long long> timing=std::vector<unsigned long long>(chunk_capacity/8);
    std::vector<uint64_t> tickets=std::vector<uint64_t>(2048);
    std::ofstream requests,raw_times,chunks;
    std::unique_ptr<AsyncModuleInterceptor> loader;
    AsyncModuleHandle modules[2]{};AsyncFunctionHandle functions[2]{};
    SampledReadStats stats;uint64_t time_offset=0;bool finished=false,failed=false;
    Impl(SampledRegistration &r,SampledRoute kind,const std::filesystem::path &path):registration(r),service(r.service()),route(kind),output(path){
        sampled_route_name(kind);need(!std::filesystem::exists(path),"reader output exists");std::filesystem::create_directories(path);
        requests.open(path/"requests.bin",std::ios::binary);raw_times.open(path/"lane-times.bin",std::ios::binary);chunks.open(path/"chunks.jsonl");
        need(bool(requests)&&bool(raw_times)&&bool(chunks),"reader logs open failed");
        ck(cudaMalloc(reinterpret_cast<void **>(&slab),(route==SampledRoute::Symbolic?1:2)*chunk_capacity));
        times=reinterpret_cast<unsigned long long *>(slab+(route==SampledRoute::Symbolic?0:chunk_capacity));
        try {
        if(route==SampledRoute::Ptx){
            ModuleOptions options;options.uniform_warp_reads=true;loader=std::make_unique<AsyncModuleInterceptor>(service,options);
            json identities=json::array();
            for(unsigned i=0;i<2;++i){
                const unsigned unit=i?2:16;const auto source=ptx_copy(unit);
                std::ofstream(path/("copy-"+std::to_string(unit)+".ptx"),std::ios::binary)<<source;
                std::ofstream(path/("copy-"+std::to_string(unit)+".rewritten.ptx"),std::ios::binary)<<rewrite_async_ptx(source,true).text;
                modules[i]=loader->load_ptx(source);functions[i]=loader->function(modules[i],"sampled_copy");
                const auto id=loader->identity(modules[i]);
                identities.push_back({{"unit",unit},{"source_sha256",id.source_sha256},{"rewritten_sha256",id.rewritten_sha256},
                    {"helper_sha256",id.helper_sha256},{"artifact_sha256",id.artifact_sha256},{"cubin_sha256",id.cubin_sha256},
                    {"instance",id.instance},{"context",id.context_id},{"target_sm",id.target_sm},{"driver",id.driver_version},
                    {"frontend",id.frontend},{"protocol",id.protocol},
                    {"kernel","sampled_copy"},{"control_flow","straight line with predicated loads/stores"}});
            }
            std::ofstream(path/"ptx-identities.json")<<identities.dump(2)<<'\n';
        }
        } catch(...) {loader.reset();cudaFree(slab);slab=nullptr;throw;}
        std::ofstream(path/"request-schema.json")<<json({{"format","little-endian <11Q4I"},{"record_bytes",104},
            {"fields",{"launch","ticket","offset","logical","arrival_ns","arrival_ps","end_ps","first_ns","last_ns","time_file_offset","grant_floor_ns","range","bytes","producer","unit"}},
            {"lane_times","consecutive uint64 before/after pairs; per-chunk mapping in chunks.jsonl"}}).dump(2)<<'\n';
    }
    ~Impl(){loader.reset();if(slab)cudaFree(slab);}
    void current() const {need(owner==std::this_thread::get_id() && !finished && !failed,"reader owner or lifecycle");}
    void one(unsigned index,uint64_t offset,unsigned count,unsigned unit){
        const auto &range=registration.ranges().at(index);
        auto *input=static_cast<const unsigned char *>(range.address)+offset;
        const bool payload=route!=SampledRoute::Symbolic;
        if(payload){ck(cudaMemcpy(reference.data(),input,count,cudaMemcpyDeviceToHost));stats.validation_d2h_bytes+=count;}
        if(payload)ck(cudaMemset(slab,0xa5,chunk_capacity));ck(cudaMemset(times,0,chunk_capacity));
        stats.memset_bytes+=(payload?2:1)*chunk_capacity;
        uint64_t launch=0;
        if(route==SampledRoute::Ptx){
            std::vector<void *> args{&input,&slab,&times,&count};
            launch=loader->launch(functions[unit==2],{8,256},args,{slab,2*chunk_capacity});
        }else{
            std::vector<void *> args{&input,&count,&unit,&slab,&times};
            const void *kernel=payload?(const void *)sampled_window<true>:(const void *)sampled_window<false>;
            launch=service.launch({8,256},kernel,args,nullptr);
        }
        service.synchronize();auto audit=service.take_audit();
        const unsigned groups=(count+32*unit-1)/(32*unit),timer_bytes=(count/unit)*16;
        need(audit.size()==groups && timer_bytes<=chunk_capacity,"reader request/timing coverage");
        ck(cudaMemcpy(timing.data(),times,timer_bytes,cudaMemcpyDeviceToHost));stats.validation_d2h_bytes+=timer_bytes;
        raw_times.write(reinterpret_cast<const char *>(timing.data()),timer_bytes);
        std::set<unsigned> seen;
        uint64_t latest=0;
        for(const auto &a:audit){
            need(a.first_address>=uintptr_t(input),"reader lower bound");const auto relative=a.first_address-uintptr_t(input);
            need(relative<count && relative%(32*unit)==0,"reader address layout");
            const auto q=unsigned(relative/(32*unit));const auto span=std::min(32*unit,count-unsigned(relative));
            need(seen.insert(q).second && a.launch==launch && a.first_bytes==span && a.span_count==1 &&
                a.producer<2048 && a.producer%32==0 && relative%(2048*unit)==a.producer*unit &&
                a.ticket==tickets[a.producer]+1 && a.arrival_ns>=a.grant_floor_ns && a.end_ps>=a.arrival_ps,"reader request/ticket mismatch");
            tickets[a.producer]=a.ticket;
            const auto deadline=a.arrival_ns+(a.end_ps-a.arrival_ps+999)/1000;
            uint64_t first=UINT64_MAX,last=0;
            for(unsigned i=unsigned(relative)/unit;i<(unsigned(relative)+span)/unit;++i){
                need(timing[2*i]<=a.arrival_ns && timing[2*i+1]>=deadline,"reader early data use");
                first=std::min<uint64_t>(first,timing[2*i]);last=std::max<uint64_t>(last,timing[2*i+1]);++stats.lane_checks;
            }
            const auto logical=range.logical+offset+relative;
            need(a.sectors==((logical%64+span+63)/64),"reader sector coverage");
            latest=std::max(latest,last);
            RequestRecord record{a.launch,a.ticket,offset+relative,logical,a.arrival_ns,a.arrival_ps,a.end_ps,first,last,time_offset,a.grant_floor_ns,index,span,a.producer,unit};
            requests.write(reinterpret_cast<const char *>(&record),sizeof(record));
        }
        json hashes=nullptr;
        if(payload){
            ck(cudaMemcpy(returned.data(),slab,count,cudaMemcpyDeviceToHost));
            need(std::memcmp(reference.data(),returned.data(),count)==0,"reader returned bytes differ");
            const auto reference_hash=hash_bytes(reference,count),returned_hash=hash_bytes(returned,count);
            ck(cudaMemcpy(const_cast<unsigned char *>(input),slab,count,cudaMemcpyDeviceToDevice));ck(cudaDeviceSynchronize());
            ck(cudaMemcpy(returned.data(),input,count,cudaMemcpyDeviceToHost));
            need(std::memcmp(reference.data(),returned.data(),count)==0,"reader published bytes differ");
            hashes={{"reference",reference_hash},{"returned",returned_hash},{"published",hash_bytes(returned,count)}};
            stats.payload_bytes+=count;stats.publication_bytes+=count;stats.byte_checks+=2*count;stats.validation_d2h_bytes+=2*count;
        }
        sampled_stamp<<<1,1>>>(times);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
        ck(cudaMemcpy(&stats.last_publication_ns,times,8,cudaMemcpyDeviceToHost));
        stats.validation_d2h_bytes+=8;need(stats.last_publication_ns>=latest,"publication before reader completion");
        chunks<<json({{"range",index},{"tensor",range.name},{"offset",offset},{"bytes",count},{"unit",unit},{"launch",launch},
            {"requests",audit.size()},{"time_file_offset",time_offset},{"time_bytes",timer_bytes},{"hashes",hashes},
            {"release_stamp_ns",stats.last_publication_ns},{"payload_published",payload}}).dump()<<'\n';
        need(bool(requests)&&bool(raw_times)&&bool(chunks),"reader log write failed");
        time_offset+=timer_bytes;stats.requests+=audit.size();stats.requested_bytes+=count;++stats.launches;
    }
};
SampledReader::SampledReader(SampledRegistration &r,SampledRoute kind,const std::filesystem::path &output):impl_(std::make_unique<Impl>(r,kind,output)){}
SampledReader::~SampledReader()=default;
void SampledReader::read(std::size_t index,uint64_t offset,uint64_t bytes){
    auto &s=*impl_;s.current();const auto &range=s.registration.ranges().at(index);
    need(bytes && bytes%2==0 && offset%2==0 && offset<=range.bytes && bytes<=range.bytes-offset,"reader range or element alignment");
    const auto start=std::chrono::steady_clock::now();
    try{
        const auto address=reinterpret_cast<uintptr_t>(range.address)+offset;
        const unsigned unit=address%16==0 && bytes%16==0?16:2;
        const unsigned limit=s.route==SampledRoute::Ptx?(unit==16?32768:4096):(unit==16?chunk_capacity:32768);
        for(uint64_t done=0;done<bytes;done+=std::min<uint64_t>(limit,bytes-done))
            s.one(unsigned(index),offset+done,unsigned(std::min<uint64_t>(limit,bytes-done)),unit);
        s.stats.read_wall_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }catch(...){s.failed=true;throw;}
}
SampledReadStats SampledReader::statistics() const {return impl_->stats;}
void SampledReader::finish(){
    auto &s=*impl_;s.current();
    if(s.loader){for(auto module:s.modules)s.loader->unload(module);s.loader.reset();}
    s.requests.close();s.raw_times.close();s.chunks.close();need(bool(s.requests)&&bool(s.raw_times)&&bool(s.chunks),"reader close logs");
    const auto r=s.service.statistics();const auto t=s.stats;
    std::ofstream summary(s.output/"summary.json");summary<<json({{"status","pass"},{"route",sampled_route_name(s.route)},
        {"requests",t.requests},{"requested_bytes",t.requested_bytes},{"payload_bytes",t.payload_bytes},{"publication_bytes",t.publication_bytes},
        {"launches",t.launches},{"lane_checks",t.lane_checks},{"byte_checks",t.byte_checks},{"validation_d2h_bytes",t.validation_d2h_bytes},
        {"memset_bytes",t.memset_bytes},{"request_log_bytes",t.requests*sizeof(RequestRecord)},{"lane_time_log_bytes",s.time_offset},
        {"read_wall_seconds",t.read_wall_seconds},{"mirror_capacity_bytes",s.route==SampledRoute::Symbolic?0:chunk_capacity},{"timing_capacity_bytes",chunk_capacity},
        {"backend_model_now_ps",r.model_now_ps},{"common_clock_qualified",false}}).dump(2)<<'\n';summary.close();need(bool(summary),"reader summary write");s.finished=true;
}
}
