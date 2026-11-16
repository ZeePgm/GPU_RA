#include "sampled_model.hpp"
#include "integrated_profile.hpp"
#include "integrated_geometry.hpp"
#include "instrumented_bridge.hpp"
#include "paged_bridge.hpp"
#include "ggml-backend.h"
#include "hbflab/sampled_reader.hpp"
#include "hbflab/async_device.cuh"
#include "hbflab/ptx.hpp"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <vector>

using namespace hbflab;
using json=nlohmann::ordered_json;
namespace fs=std::filesystem;
namespace {
void need(bool b,const char *message){if(!b)throw std::runtime_error(message);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::string env(const char *key){const char *value=std::getenv(key);need(value && *value,"required sampled environment absent");return value;}
__global__ void sampled_consumer_release(unsigned long long *out){*out=gpu_now_ns();}
__global__ void sampled_consumer_complete(unsigned long long *out){*out=gpu_now_ns();}
struct State {
    bool bound=false,finished=false,paged=false,reference=false;
    int step=-1;unsigned prefill=0;
    std::string route;fs::path output;json inventory;
    std::map<const ggml_tensor *,json> weights;
    std::map<const ggml_tensor *,std::vector<std::string>> initial_hashes;
    std::map<const ggml_tensor *,size_t> indices;
    std::map<const ggml_tensor *,uint64_t> pending;
    std::set<const ggml_tensor *> seen;
    std::unique_ptr<SampledRegistration> registration;
    std::unique_ptr<LiveTimingService> reference_service;
    std::unique_ptr<SampledReader> reader;
    LiveTimingService *service=nullptr;
    unsigned long long *stamp=nullptr;
    std::ofstream gates,consumers;
    uint64_t calls=0,post_calls=0,hash_validation_bytes=0;
    json kv_versions=json::array();
};
std::unique_ptr<State> state;
bool configured_once=false;
std::vector<std::string> hashes(State &s,const ggml_tensor *tensor){
    ck(cudaDeviceSynchronize());std::string block(262144,'\0');std::vector<std::string> result;
    const auto bytes=ggml_nbytes(tensor);
    for(size_t offset=0;offset<bytes;offset+=std::min(block.size(),bytes-offset)){
        const auto count=std::min(block.size(),bytes-offset);
        ck(cudaMemcpy(block.data(),static_cast<const unsigned char *>(tensor->data)+offset,count,cudaMemcpyDeviceToHost));
        result.push_back(sha256(std::string_view(block.data(),count)));s.hash_validation_bytes+=count;
    }
    return result;
}
uint64_t stamp(State &s,bool complete=false){
    if(complete)sampled_consumer_complete<<<1,1>>>(s.stamp);else sampled_consumer_release<<<1,1>>>(s.stamp);
    ck(cudaGetLastError());ck(cudaDeviceSynchronize());uint64_t value=0;
    ck(cudaMemcpy(&value,s.stamp,8,cudaMemcpyDeviceToHost));return value;
}
}
bool sampled_model_active(){return bool(state);}
void sampled_model_configure(const char *mode,const char *output){
    need(!configured_once,"one sampled model session per process");configured_once=true;state=std::make_unique<State>();auto &s=*state;
    s.route=env("HBFLAB_SAMPLED_ROUTE");s.reference=s.route=="native";
    if(!s.reference)sampled_route(s.route);
    const auto kv=env("HBFLAB_JOINT_KV");need(kv=="native" || kv=="paged","sampled KV selection");s.paged=kv=="paged";
    need(std::string(mode)==(s.reference && !s.paged?"native":"modeled"),"sampled execution mode must be explicit");
    need(env("HBFLAB_SAMPLED_TOKEN_ONLY")=="1","sampled token-only graph required");
    const auto tokens=env("HBFLAB_SAMPLED_PREFILL");size_t consumed=0;s.prefill=unsigned(std::stoul(tokens,&consumed));
    need(consumed==tokens.size() && s.prefill>0 && s.prefill<=256,"sampled prefill scope");
    s.output=output;std::ifstream input(env("HBFLAB_SAMPLED_INVENTORY"));input>>s.inventory;
    m6_load_profile(env("HBFLAB_SAMPLED_PROFILE").c_str(),output);
    need(integrated_profile().blocks_per_bank==65536 && integrated_profile().pages_per_block==4 && integrated_profile().frames==4,"sampled profile geometry/frames");
    s.gates.open(s.output/"sampled-gates.jsonl");s.consumers.open(s.output/"sampled-consumers.jsonl");
    need(bool(s.gates)&&bool(s.consumers),"sampled logs open");ck(cudaMalloc(reinterpret_cast<void **>(&s.stamp),8));
}
void sampled_model_bind(const ggml_tensor *const *tensors,unsigned count){
    auto &s=*state;need(!s.bound && count==14,"sampled bind scope");json identities=json::array();
    for(unsigned i=0;i<count;++i){
        const auto *t=tensors[i];need(t && ggml_is_contiguous(t),"sampled noncontiguous layout unsupported");
        bool found=false;
        for(const auto &entry:s.inventory.at("all_tensors"))if(entry.at("selected").get<bool>() && entry.at("name")==t->name){
            const auto type=entry.at("dtype").get<std::string>();
            need((type=="F16" && t->type==GGML_TYPE_F16)||(type=="F32" && t->type==GGML_TYPE_F32),"sampled weight dtype unsupported");
            need(ggml_nbytes(t)==entry.at("bytes").get<size_t>(),"sampled weight size mismatch");
            const auto shape=entry.at("shape");for(unsigned d=0;d<4;++d)need(t->ne[d]==(d<shape.size()?shape[d].get<int64_t>():1),"sampled shape mismatch");
            need(s.weights.emplace(t,entry).second,"sampled duplicate tensor identity");found=true;break;
        }
        need(found,"unexpected sampled weight");s.initial_hashes[t]=hashes(s,t);
        identities.push_back({{"tensor",t->name},{"bytes",ggml_nbytes(t)},{"chunk_bytes",262144},{"chunk_sha256",s.initial_hashes[t]}});
    }
    std::ofstream(s.output/"weight-chunk-hashes.json")<<identities.dump(2)<<'\n';
    if(!s.reference){
        SampledOptions options;options.route=sampled_route(s.route);options.layout_bytes=s.inventory.at("logical_weight_layout_bytes");
        options.kv_bytes=s.paged?s.inventory.at("kv_reserved_bytes").get<uint64_t>():0;options.geometry=m6_geometry();
        s.registration=std::make_unique<SampledRegistration>(options);
        for(const auto &[t,entry]:s.weights)s.registration->add({t->name,t->data,ggml_nbytes(t),entry.at("logical_base")});
        s.registration->prepare();s.service=&s.registration->service();
        if(s.paged)s.registration->prepare_kv([&](LiveTimingService *service){m4_begin(2,service,s.output.string().c_str());});
        s.registration->start();
        for(size_t i=0;i<s.registration->ranges().size();++i)for(const auto &[t,entry]:s.weights)
            if(s.registration->ranges()[i].name==t->name)s.indices.emplace(t,i);
        s.reader=std::make_unique<SampledReader>(*s.registration,sampled_route(s.route),s.output/"readonly");
    }else if(s.paged){
        AsyncRuntimeOptions options;options.backend.geometry=m6_geometry();options.backend.max_events_per_batch=2000000000ull;
        options.backend.coalesce_capacity_feedback=true;options.backend.direct_device_transactions=true;
        options.readonly_layout_bytes=s.inventory.at("logical_weight_layout_bytes");options.audit_limit=8193;
        options.timeout_ns=1000000000ull;options.service_timeout_ns=60000000000ull;
        s.reference_service=std::make_unique<LiveTimingService>(options,2048,1);s.service=s.reference_service.get();
        m4_begin(2,s.service,s.output.string().c_str());s.service->start();
    }
    s.bound=true;
}
void sampled_model_step_begin(int step){
    auto &s=*state;need(s.bound && !s.finished && step==s.step+1 && s.pending.empty(),"sampled step lifecycle");s.step=step;s.seen.clear();if(s.paged)m4_step_begin(step);
}
void sampled_model_graph_node(const ggml_tensor *node){
    auto &s=*state;if(!s.bound || s.step<0)return;
    for(unsigned slot=0;slot<GGML_MAX_SRC;++slot){
        const auto *weight=node->src[slot];if(!weight || !s.weights.count(weight))continue;
        const auto &entry=s.weights.at(weight);
        need(entry.at("expected_op")==ggml_op_name(node->op) && entry.at("expected_source_slot")==slot,"sampled consumer op/slot unsupported");
        need(s.seen.insert(weight).second,"duplicate sampled graph consumer");
        json ranges=json::array();
        if(node->op==GGML_OP_GET_ROWS){
            const auto *ids=node->src[1];need(ids && ids->type==GGML_TYPE_I32 && ggml_is_contiguous(ids),"sampled gather index layout");
            std::vector<int32_t> tokens(ggml_nelements(ids));need(tokens.size()==(s.step?1:s.prefill),"sampled gather row count");
            ggml_backend_tensor_get(ids,tokens.data(),0,tokens.size()*sizeof(int32_t));
            for(size_t i=0;i<tokens.size();++i){need(tokens[i]>=0 && tokens[i]<weight->ne[1],"sampled gather row bounds");
                ranges.push_back({{"row",tokens[i]},{"occurrence",i},{"offset",uint64_t(tokens[i])*weight->nb[1]},{"bytes",weight->nb[1]}});}
        }else ranges.push_back({{"offset",0},{"bytes",ggml_nbytes(weight)}});
        ck(cudaDeviceSynchronize());const auto before=s.reader?s.reader->statistics():SampledReadStats{};
        if(s.reader)for(const auto &range:ranges)s.reader->read(s.indices.at(weight),range.at("offset"),range.at("bytes"));
        const auto after=s.reader?s.reader->statistics():SampledReadStats{};
        const auto release=stamp(s);need(!s.reader || release>=after.last_publication_ns,"consumer release precedes publication");
        need(s.pending.emplace(node,release).second,"sampled consumer already pending");
        s.gates<<json({{"step",s.step},{"tensor",weight->name},{"node",node->name},{"op",ggml_op_name(node->op)},
            {"ranges",ranges},{"requests",after.requests-before.requests},{"requested_bytes",after.requested_bytes-before.requested_bytes},
            {"payload_bytes",after.payload_bytes-before.payload_bytes},{"publication_bytes",after.publication_bytes-before.publication_bytes},
            {"gate_release_ns",release},{"route",s.route},{"dependency","publication synchronized before framework dispatch"}}).dump()<<'\n';
        need(bool(s.gates),"sampled gate log write");++s.calls;
    }
}
extern "C" HBFLAB_M2_API void hbflab_sampled_after_node(const ggml_tensor *node){
    if(!state || !state->pending.count(node))return;auto &s=*state;
    const auto finished=stamp(s,true),release=s.pending.at(node);need(finished>=release,"sampled early consumer completion");
    s.consumers<<json({{"step",s.step},{"node",node->name},{"gate_release_ns",release},{"consumer_finished_ns",finished},
        {"start_lower_bound","framework dispatch follows completed gate callback"}}).dump()<<'\n';
    need(bool(s.consumers),"sampled consumer post log");s.pending.erase(node);++s.post_calls;
}
void sampled_model_step_end(){auto &s=*state;need(s.seen.size()==14 && s.pending.empty(),"sampled consumer coverage/post dependency");if(s.paged){m4_step_end();const auto version=m4_sequence_version(0);need(version==uint64_t(s.step+1),"KV version publication mismatch");s.kv_versions.push_back({{"step",s.step},{"version",version}});}}
void sampled_model_finish(){
    auto &s=*state;need(!s.finished && s.step==7 && s.calls==112 && s.post_calls==112 && s.pending.empty(),"sampled model final scope");
    using Clock=std::chrono::steady_clock;
    const auto audit_start=Clock::now();
    if(s.paged){const auto version=m4_sequence_version(0);m4_sequence_audit((s.output/"kv-content.json").string().c_str(),s.prefill);need(m4_sequence_version(0)==version,"KV audit changed version");std::ofstream(s.output/"kv-versions.json")<<s.kv_versions.dump(2)<<'\n';}
    const double kv_audit_ms=std::chrono::duration<double,std::milli>(Clock::now()-audit_start).count();
    const auto drain_start=Clock::now();
    if(s.paged)m4_finish();
    const auto reads=s.reader?s.reader->statistics():SampledReadStats{};
    if(s.reader)s.reader->finish();
    json runtime=nullptr;
    if(s.service){const auto stats=s.service->statistics();
        need(stats.submitted==reads.requests && stats.completed==reads.requests && !stats.audit_dropped &&
            stats.media_submitted==stats.media_completed,"sampled final request accounting");
        if(s.registration)s.registration->close();else{
            s.service->finish();const auto final=s.service->statistics();
            need(s.service->drained() && final.backend.registered_globals==final.backend.completed_globals &&
                 !final.backend.host.logical_failed && final.backend.peak_resident_fragments<=256,"reference KV not drained");
        }
        runtime={{"drained",true},{"requests",stats.submitted},{"media_requests",stats.media_submitted},
            {"peak_resident_fragments",stats.backend.peak_resident_fragments},{"backend_model_now_ps",stats.model_now_ps}};
    }
    const double drain_ms=std::chrono::duration<double,std::milli>(Clock::now()-drain_start).count();
    const auto hash_start=Clock::now();
    for(const auto &[t,original]:s.initial_hashes)need(hashes(s,t)==original,"borrowed sampled weight changed");
    const double hash_ms=std::chrono::duration<double,std::milli>(Clock::now()-hash_start).count();
    s.gates.close();s.consumers.close();need(bool(s.gates)&&bool(s.consumers),"sampled model logs close");
    std::ofstream summary(s.output/"sampled-summary.json");summary<<json({{"status","pass"},{"route",s.route},{"kv",s.paged?"modeled":"native"},
        {"steps",8},{"consumers",s.calls},{"post_consumers",s.post_calls},{"borrowed_unchanged",true},
        {"requests",reads.requests},{"requested_bytes",reads.requested_bytes},{"payload_bytes",reads.payload_bytes},
        {"publication_bytes",reads.publication_bytes},{"lane_checks",reads.lane_checks},{"byte_checks",reads.byte_checks},
        {"hash_validation_bytes",s.hash_validation_bytes},{"runtime",runtime},{"common_clock_qualified",false},
        {"kv_audit_wall_ms",kv_audit_ms},{"drain_wall_ms",drain_ms},{"final_weight_hash_wall_ms",hash_ms}}).dump(2)<<'\n';
    summary.close();need(bool(summary),"sampled model summary write");s.finished=true;
}
void sampled_model_abort(){if(!state)return;auto &s=*state;m4_abort();s.reader.reset();s.registration.reset();s.reference_service.reset();s.service=nullptr;if(s.stamp){cudaFree(s.stamp);s.stamp=nullptr;}}
