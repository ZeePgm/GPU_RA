#include "hbflab/sampled_reader.hpp"
#include "hbflab/ptx.hpp"
#include "sampled_trace.hpp"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

using namespace hbflab;
using json=nlohmann::ordered_json;
namespace fs=std::filesystem;
void need(bool value,const char *message){if(!value)throw std::runtime_error(message);}
void ck(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
template<class F>void rejects(F action){bool caught=false;try{action();}catch(const std::exception &){caught=true;}need(caught,"negative reader operation accepted");}
struct Memory{void *pointer=nullptr;~Memory(){if(pointer)cudaFree(pointer);}};
int main(int argc,char **argv)try{
    need(argc==6 || argc==7,"usage: sampled_reader_fixture inventory.json model.gguf request-plan.json output route");
    const fs::path output=argv[4];need(!fs::exists(output),"fixture output exists");fs::create_directories(output);
    const bool summary=argc==7 && std::string(argv[6])=="summary";
    if(argc==7)need(summary || std::string(argv[6])=="full","fixture audit mode");
    std::unique_ptr<SampledGpuTrace> trace;if(!summary)trace=std::make_unique<SampledGpuTrace>(output);
    std::ifstream inv_file(argv[1]),plan_file(argv[3]);json inventory,plan;inv_file>>inventory;plan_file>>plan;
    const auto route=sampled_route(argv[5]);
    need(fs::file_size(argv[2])==inventory.at("model").at("bytes").get<uint64_t>(),"GGUF file size mismatch");
    ck(cudaFree(nullptr));const auto begin=std::chrono::steady_clock::now();
    Memory allocation;const auto selected_bytes=inventory.at("selected_bytes").get<size_t>();ck(cudaMalloc(&allocation.pointer,selected_bytes));
    SampledOptions options;options.route=route;options.layout_bytes=inventory.at("logical_weight_layout_bytes");
    SampledRegistration registration(options);
    std::vector<unsigned char> buffer(262144);std::ifstream model(argv[2],std::ios::binary);need(bool(model),"GGUF read open failed");
    uint64_t packed=0;
    for(const auto &t:inventory.at("all_tensors"))if(t.at("selected").get<bool>()){
        const auto bytes=t.at("bytes").get<uint64_t>(),file=t.at("file_offset").get<uint64_t>();
        need(file<=fs::file_size(argv[2]) && bytes<=fs::file_size(argv[2])-file,"GGUF tensor file bounds");
        auto *pointer=static_cast<unsigned char *>(allocation.pointer)+packed;
        model.seekg(file);
        for(uint64_t offset=0;offset<bytes;offset+=std::min<uint64_t>(buffer.size(),bytes-offset)){
            const auto count=std::min<uint64_t>(buffer.size(),bytes-offset);
            model.read(reinterpret_cast<char *>(buffer.data()),count);need(model.gcount()==std::streamsize(count),"GGUF tensor truncated");
            ck(cudaMemcpy(pointer+offset,buffer.data(),count,cudaMemcpyHostToDevice));
        }
        registration.add({t.at("name"),pointer,bytes,t.at("logical_base")});packed+=bytes;
    }
    need(packed==selected_bytes,"fixture selected byte budget");registration.prepare();registration.start();
    const auto initialized=std::chrono::steady_clock::now();
    std::map<std::string,size_t> indices;
    json bindings=json::array();
    for(size_t i=0;i<registration.ranges().size();++i){const auto &range=registration.ranges()[i];indices.emplace(range.name,i);
        bindings.push_back({{"name",range.name},{"address",uintptr_t(range.address)},{"bytes",range.bytes},{"logical",range.logical}});}
    std::ofstream(output/"bindings.json")<<bindings.dump(2)<<'\n';
    SampledReader reader(registration,route,output/"reader",summary?SampledAuditMode::Summary:SampledAuditMode::Full);
    unsigned negatives=0;
    const auto first=registration.ranges().front();
    for(const auto span:std::vector<std::pair<uint64_t,uint64_t>>{{0,0},{1,2},{0,1},{first.bytes,2},{UINT64_MAX,2},{0,UINT64_MAX}}){
        rejects([&]{reader.read(0,span.first,span.second);});++negatives;
    }
    rejects([&]{reader.read(99,0,2);});++negatives;
    need(reader.statistics().requests==0,"invalid ranges issued requests");
    const std::string unsupported=R"(.version 8.0
.target sm_75
.address_size 64
.visible .entry unsupported(.param .u64 source) {
.reg .b32 %value;
.reg .b64 %pointer;
ld.param.u64 %pointer, [source];
bra L;
L: ld.global.u32 %value, [%pointer];
ret;
}
)";
    std::ofstream(output/"unsupported-control-flow.ptx")<<unsupported;
    rejects([&]{rewrite_async_ptx(unsupported,true);});++negatives;
    json calls=json::array();
    if(trace)trace->phase(0);
    for(const auto &span:plan.at("spans")){
        const std::string name=span.at("tensor");need(indices.count(name),"fixture plan tensor not registered");
        const auto offset=span.at("offset").get<uint64_t>(),bytes=span.at("bytes").get<uint64_t>();
        const auto before=reader.statistics();const auto call_start=std::chrono::steady_clock::now();
        reader.read(indices.at(name),offset,bytes);const auto after=reader.statistics();
        calls.push_back({{"tensor",name},{"offset",offset},{"bytes",bytes},{"kind",span.at("kind")},
            {"requests",after.requests-before.requests},{"launches",after.launches-before.launches},
            {"wall_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-call_start).count()}});
    }
    const auto stats=reader.statistics();reader.finish();const auto backend=registration.service().statistics();registration.close();
    if(trace)trace->finish();const auto end=std::chrono::steady_clock::now();
    json result={{"status","pass"},{"route",argv[5]},{"calls",calls},{"negative_checks",negatives},
        {"selected_ranges",registration.ranges().size()},{"audit_mode",summary?"summary":"full"},{"selected_bytes",selected_bytes},{"requested_bytes",stats.requested_bytes},{"requests",stats.requests},
        {"payload_bytes",stats.payload_bytes},{"publication_bytes",stats.publication_bytes},{"lane_checks",stats.lane_checks},
        {"initialized_seconds",std::chrono::duration<double>(initialized-begin).count()},
        {"total_seconds",std::chrono::duration<double>(end-begin).count()},
        {"drained",true},{"peak_resident_fragments",backend.backend.peak_resident_fragments},
        {"scope",plan.at("scope")}};
    std::ofstream report(output/"summary.json");report<<result.dump(2)<<'\n';report.close();need(bool(report),"fixture report write failed");
    std::cout<<result.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
