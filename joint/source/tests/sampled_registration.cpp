#include "hbflab/sampled_readonly.hpp"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>

using namespace hbflab;
using json=nlohmann::ordered_json;
void need(bool b,const char *m){if(!b)throw std::runtime_error(m);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class F> void rejects(F action){bool caught=false;try{action();}catch(const std::exception&){caught=true;}need(caught,"negative accepted");}
struct Memory {void *pointer=nullptr;~Memory(){if(pointer)cudaFree(pointer);}};
int main(int argc,char **argv)try{
    need(argc==4,"usage: sampled_registration inventory.json route report.json");
    ck(cudaFree(nullptr));
    std::ifstream input(argv[1]);json inventory;input>>inventory;
    SampledOptions options;options.route=sampled_route(argv[2]);
    options.layout_bytes=inventory.at("logical_weight_layout_bytes");options.kv_bytes=inventory.at("kv_reserved_bytes");
    unsigned negatives=0;
    for(const char *route:{"", "PTX", "native", "automatic"}){rejects([&]{sampled_route(route);});++negatives;}
    {auto bad=options;bad.geometry.blocks_per_bank=64;rejects([&]{SampledRegistration r(bad);});++negatives;}
    {auto bad=options;bad.layout_bytes=1;rejects([&]{SampledRegistration r(bad);});++negatives;}
    {auto bad=options;bad.route=static_cast<SampledRoute>(99);rejects([&]{SampledRegistration r(bad);});++negatives;}
    for(const uint64_t alignment:{0ull,6144ull,12288ull}){
        AsyncRuntimeOptions bad;bad.readonly_reservation_alignment=alignment;
        rejects([&]{AsyncRuntime runtime(bad);});++negatives;
    }
    {
        SampledRegistration r(options);const void *a=reinterpret_cast<void *>(0x100000);
        r.add({"a",a,512,0});
        rejects([&]{r.add({"b",a,512,32768});});++negatives;
        rejects([&]{r.add({"b",reinterpret_cast<void *>(0x200000),512,0});});++negatives;
        rejects([&]{r.add({"a",reinterpret_cast<void *>(0x200000),512,32768});});++negatives;
        rejects([&]{r.add({"b",reinterpret_cast<void *>(0x200000),0,32768});});++negatives;
        rejects([&]{r.add({"b",reinterpret_cast<void *>(0x200000),512,4096});});++negatives;
        rejects([&]{r.add({"b",reinterpret_cast<void *>(0x200000),512,options.layout_bytes});});++negatives;
        rejects([&]{r.add({"b",reinterpret_cast<void *>(UINTPTR_MAX-16),512,32768});});++negatives;
        bool caught=false;std::thread other([&]{try{r.add({"x",reinterpret_cast<void *>(0x200000),512,32768});}catch(...){caught=true;}});other.join();need(caught,"owner not enforced");++negatives;
        rejects([&]{r.start();});++negatives;
    }
    {
        SampledRegistration r(options);
        for(unsigned i=0;i<max_ranges;++i)r.add({std::to_string(i),reinterpret_cast<void *>(uintptr_t(0x100000)+i*65536),512,uint64_t(i)*32768});
        rejects([&]{r.add({"overflow",reinterpret_cast<void *>(0x80000000),512,uint64_t(max_ranges)*32768});});++negatives;
    }
    {
        SampledRegistration r(options);int host=0;r.add({"host",&host,sizeof(host),0});
        rejects([&]{r.prepare();});++negatives;
        rejects([&]{r.add({"after-failure",reinterpret_cast<void *>(0x80000000),512,32768});});++negatives;
    }
    const auto start=std::chrono::steady_clock::now();
    Memory allocation;const auto selected_bytes=inventory.at("selected_bytes").get<size_t>();ck(cudaMalloc(&allocation.pointer,selected_bytes));
    SampledRegistration registration(options);size_t offset=0;
    for(const auto &t:inventory.at("all_tensors"))if(t.at("selected").get<bool>()){
        const auto bytes=t.at("bytes").get<uint64_t>();
        registration.add({t.at("name"),static_cast<unsigned char *>(allocation.pointer)+offset,bytes,t.at("logical_base")});offset+=bytes;
    }
    need(offset==selected_bytes && registration.ranges().size()==14,"actual inventory binding count");
    registration.prepare();
    rejects([&]{registration.add({"late",allocation.pointer,1,0});});++negatives;
    rejects([&]{registration.start();});++negatives;
    uint64_t kv_base=0;
    registration.prepare_kv([&](LiveTimingService *service){kv_base=service->reserve_media(options.kv_bytes);need(kv_base==options.layout_bytes,"KV base differs from complete layout");});
    rejects([&]{registration.prepare_kv([](LiveTimingService*){});});++negatives;
    registration.start();
    auto &service=registration.service();
    need(service.read_media(kv_base,4096).status==Status::Ok,"shared initial KV read");
    std::array<uint64_t,64> tokens{};for(unsigned i=0;i<64;++i)tokens[i]=401+i;
    need(service.write_media(kv_base,tokens).status==Status::Ok,"shared KV write");
    need(service.read_media(kv_base,4096).status==Status::Ok,"shared updated KV read");
    const auto media=service.take_media_audit();need(media.size()==3 && service.take_audit().empty(),"shared audit coverage");
    for(const auto &r:media)need(r.logical==kv_base && r.end_ps>=r.arrival_ps,"shared media range/time");
    const auto stats=service.statistics();
    const auto initialized=registration.initialized_bytes();registration.close();
    const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    json result={{"status","pass"},{"route",argv[2]},{"scope","registration and shared service only"},
        {"selected_ranges",14},{"selected_bytes",selected_bytes},{"weight_initialized_bytes",initialized},
        {"kv_base",kv_base},{"kv_bytes",options.kv_bytes},{"media_capacity_bytes",options.geometry.channel_bytes()},
        {"programs",stats.backend.device.programs},{"media_requests",stats.media_submitted},
        {"peak_resident_fragments",stats.backend.peak_resident_fragments},{"negative_checks",negatives},
        {"drained",true},{"wall_seconds",seconds},{"readonly_alignment",SampledRegistration::alignment}};
    std::ofstream report(argv[3]);report<<result.dump(2)<<'\n';report.close();need(bool(report),"report failed");
    std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
