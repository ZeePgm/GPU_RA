#include "integrated_profile.hpp"
#include "integrated_geometry.hpp"
#include "instrumented_bridge.hpp"
#include "paged_bridge.hpp"
#include "hbflab/async_device.cuh"
#include "hbflab/ptx.hpp"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <cstring>
#include <memory>
#include <algorithm>

using namespace hbflab;
using json=nlohmann::ordered_json;
const void* b2_kernel(bool,bool);
namespace {
void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
constexpr unsigned K=262144,W=32768,MATRIX=6291456;
constexpr uint64_t LAYOUT=uint64_t(28)*3*MATRIX;
struct State {
    bool configured=false,bound=false,finished=false,read_weights=false,paged=false;
    int mode=0,step=-1;
    std::filesystem::path out;
    std::map<const ggml_tensor*,uint64_t> weights;
    std::map<const ggml_tensor*,std::string> original_hash;
    std::set<const ggml_tensor*> seen;
    std::vector<uint64_t> tickets=std::vector<uint64_t>(2048);
    std::unique_ptr<LiveTimingService> service;
    unsigned char* mirror=nullptr;
    unsigned long long* times=nullptr;
    uint64_t requests=0,covered=0,checks=0,launches=0,publications=0;
    std::ofstream calls,request_log;
} s;
void save(const char* name,const json& j){std::ofstream f(s.out/name);f<<j.dump(2)<<'\n';f.close();need(bool(f),"joint report write");}
std::string bytes_of(const ggml_tensor* t){std::string b(ggml_nbytes(t),'\0');ck(cudaMemcpy(b.data(),t->data,b.size(),cudaMemcpyDeviceToHost));return b;}
void gate(const ggml_tensor* w) {
    ck(cudaDeviceSynchronize());
    auto* input=static_cast<unsigned char*>(w->data);
    const auto reference=bytes_of(w);
    need(sha256(reference)==s.original_hash.at(w),"weight changed before read");
    std::vector<unsigned char> got(K);
    std::vector<unsigned long long> times(K/8);
    uint64_t local=0;
    for(unsigned base=0;base<MATRIX;base+=K){
        unsigned count=K,slow=0;unsigned long long* phases=nullptr;
        std::vector<void*> args{&input,&base,&count,&s.mirror,&s.times,&phases,&slow};
        uint64_t id=0;
        if(s.mode==2) {
            id=s.service->launch({W/4096,256},b2_kernel(true,false),args,nullptr);
            s.service->synchronize();
        } else {
            AsyncDeviceView view{};std::vector<void*> actual{&view};actual.insert(actual.end(),args.begin(),args.end());
            ck(cudaLaunchKernel(b2_kernel(true,false),dim3(W/4096),dim3(256),actual.data(),0,nullptr));
            ck(cudaDeviceSynchronize());
        }
        ck(cudaMemcpy(got.data(),s.mirror+base,count,cudaMemcpyDeviceToHost));
        need(!memcmp(got.data(),reference.data()+base,count),"B2 mirror byte mismatch");
        if(s.mode==2){
            auto audit=s.service->take_audit();need(audit.size()==count/512,"B2 audit count");
            ck(cudaMemcpy(times.data(),s.times,count,cudaMemcpyDeviceToHost));
            auto previous=s.tickets;std::set<unsigned> seen;
            for(const auto& a:audit){
                need(a.first_address>=uintptr_t(input)+base,"B2 lower address bound");
                auto relative=a.first_address-uintptr_t(input)-base;
                unsigned q=unsigned(relative/512),batch=q/(W/512);
                need(relative<count && relative%512==0 && seen.insert(q).second && a.launch==id &&
                    a.producer<2048 && a.producer%32==0 && relative%W==a.producer*16 &&
                    a.ticket==previous[a.producer]+batch+1 && a.first_bytes==512 && a.span_count==1 && a.sectors==8,"B2 coverage/ticket");
                s.tickets[a.producer]=std::max(s.tickets[a.producer],a.ticket);
                need(a.end_ps>=a.arrival_ps,"B2 time order");
                const auto deadline=a.arrival_ns+(a.end_ps-a.arrival_ps+999)/1000;
                uint64_t consumed=0;
                for(unsigned i=q*32;i<q*32+32;++i){need(times[2*i]<=a.arrival_ns && times[2*i+1]>=deadline,"B2 early consumer");consumed=std::max<uint64_t>(consumed,times[2*i+1]);++s.checks;}
                s.request_log<<json({{"step",s.step},{"tensor",w->name},{"offset",base+relative},{"logical",s.weights.at(w)+base+relative},{"launch",id},{"producer",a.producer},{"ticket",a.ticket},{"arrival_ns",a.arrival_ns},{"arrival_ps",a.arrival_ps},{"end_ps",a.end_ps},{"consumed_ns",consumed}}).dump()<<'\n';
            }
            local+=audit.size();
        }
        ++s.launches;
    }
    // Complete publication before the original MUL_MAT consumes these bytes.
    ck(cudaMemcpy(input,s.mirror,MATRIX,cudaMemcpyDeviceToDevice));
    need(bytes_of(w)==reference,"B2 publication byte mismatch");
    ++s.publications;s.requests+=local;s.covered+=MATRIX;
    s.calls<<json({{"step",s.step},{"tensor",w->name},{"logical_base",s.weights.at(w)},{"bytes",MATRIX},{"requests",local},{"kernels",MATRIX/K},{"publication_bytes",MATRIX},{"mirror_verified",true}}).dump()<<'\n';
    need(bool(s.calls)&&bool(s.request_log),"B2 log write");
}
}
extern "C" void hbflab_m2_configure(const char* mode,const char* output){
    need(!s.configured,"one joint session per process");std::string m=mode;
    need(m=="native"||m=="bypass"||m=="modeled","joint mode");s.mode=m=="native"?0:m=="bypass"?1:2;
    const char* w=std::getenv("HBFLAB_JOINT_WEIGHTS");const char* k=std::getenv("HBFLAB_JOINT_KV");
    need(w && (std::string(w)=="native"||std::string(w)=="b2"),"joint weights mode");
    need(k && (std::string(k)=="native"||std::string(k)=="paged"),"joint KV mode");
    s.read_weights=std::string(w)=="b2";s.paged=std::string(k)=="paged";
    need(s.mode!=0 || (!s.read_weights&&!s.paged),"native must have no adapters");
    s.out=output;s.configured=true;s.calls.open(s.out/"weight-calls.jsonl");s.request_log.open(s.out/"weight-requests.jsonl");
}
extern "C" void hbflab_joint_bind(const ggml_tensor* const* tensors,unsigned count){
    need(s.configured&&!s.bound&&(count==3||count==12),"joint bind lifecycle");
    std::vector<std::pair<uint64_t,const ggml_tensor*>> ordered;json ranges=json::array();
    for(unsigned i=0;i<count;++i){auto* t=tensors[i];need(t&&ggml_is_contiguous(t)&&t->type==GGML_TYPE_F16&&ggml_nbytes(t)==MATRIX,"FFN tensor layout");
        std::string name=t->name;int layer=std::stoi(name.substr(4));unsigned role=name.find("gate")!=std::string::npos?0:name.find("up")!=std::string::npos?1:2;
        need(layer>=0&&layer<28,"FFN layer");ordered.push_back({(uint64_t(layer)*3+role)*MATRIX,t});
    }
    std::sort(ordered.begin(),ordered.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    for(auto [base,t]:ordered){need(s.weights.emplace(t,base).second,"duplicate weight");s.original_hash[t]=sha256(bytes_of(t));ranges.push_back({{"tensor",t->name},{"base",base},{"bytes",MATRIX},{"sha256",s.original_hash[t]}});}
    if(s.read_weights){ck(cudaMalloc((void**)&s.mirror,MATRIX));ck(cudaMalloc((void**)&s.times,K));}
    if(s.mode==2){
        AsyncRuntimeOptions o;o.backend.geometry=m6_geometry();o.backend.max_events_per_batch=2000000000ull;
        o.backend.coalesce_capacity_feedback=true;o.backend.direct_device_transactions=true;
        o.audit_limit=513;o.timeout_ns=1000000000ull;o.service_timeout_ns=60000000000ull;o.readonly_layout_bytes=LAYOUT;
        if(s.read_weights)for(auto [base,t]:ordered)o.readonly_bases.push_back(base);
        s.service=std::make_unique<LiveTimingService>(o,2048,1);
        if(s.read_weights)for(auto [base,t]:ordered)s.service->register_readonly(t->data,MATRIX);
    }
    if(s.paged)m4_begin(s.mode,s.service.get(),s.out.string().c_str());
    if(s.service){const auto st=s.service->statistics();save("initialization.json",{{"programs",st.backend.device.programs},{"erases",st.backend.device.erases}});s.service->start();}
    save("address-layout.json",{{"weight_layout_bytes",LAYOUT},{"selected",ranges},{"kv_base",LAYOUT},{"kv_bytes",uint64_t(integrated_profile().logical_pages)*4096},{"pages_per_block",integrated_profile().pages_per_block},{"blocks_per_bank",integrated_profile().blocks_per_bank},{"gap_initialization","none"},{"one_shared_service",bool(s.service)}});
    s.bound=true;
}
extern "C" void hbflab_m2_bind(const ggml_tensor*,const ggml_tensor*){throw std::runtime_error("use joint bind");}
extern "C" void hbflab_m2_step_begin(int step){need(s.bound&&!s.finished&&step==s.step+1,"joint step order");s.step=step;s.seen.clear();if(s.paged)m4_step_begin(step);}
extern "C" void hbflab_m2_graph_node(const ggml_tensor* t){
    if(t->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(const_cast<ggml_tensor*>(t),GGML_PREC_F32);
    if(!s.bound||s.step<0||t->op!=GGML_OP_MUL_MAT)return;
    if(s.weights.count(t->src[0])){need(s.seen.insert(t->src[0]).second,"duplicate FFN consumer");if(s.read_weights)gate(t->src[0]);}
}
extern "C" void hbflab_m2_step_end(){need(s.seen.size()==s.weights.size(),"FFN consumer missing");if(s.paged)m4_step_end();}
extern "C" void hbflab_m2_finish(){
    need(s.bound&&!s.finished&&s.step==7,"joint finish lifecycle");
    if(s.paged){m4_sequence_audit((s.out/"kv-content.json").string().c_str(),44);m4_finish();}
    json report={{"status","pass"},{"steps",s.step+1},{"weight_mode",s.read_weights?"b2":"native"},{"kv_mode",s.paged?"paged":"native"},{"execution_mode",s.mode},{"requests",s.requests},{"covered_bytes",s.covered},{"dependency_checks",s.checks},{"publications",s.publications},{"read_kernels",s.launches},{"K",K},{"W",W},{"mailbox_slots",2048},{"launch_slots",1},{"audit_limit",513},{"mirror_bytes",s.read_weights?MATRIX:0},{"timing_bytes",s.read_weights?K:0}};
    if(s.service){s.service->finish();auto st=s.service->statistics();auto b=st.backend;
        need(s.service->drained()&&st.submitted==s.requests&&st.completed==s.requests&&st.media_submitted==st.media_completed&&!st.audit_dropped&&s.service->take_audit().empty()&&b.peak_resident_fragments<=256,"joint accounting/drain");
        report["runtime"]={{"drained",true},{"submitted",st.submitted},{"completed",st.completed},{"media_submitted",st.media_submitted},{"media_completed",st.media_completed},{"audit_dropped",st.audit_dropped},{"peak_resident_fragments",b.peak_resident_fragments},{"registered_globals",b.registered_globals},{"completed_globals",b.completed_globals},{"host_failed",b.host.logical_failed},{"programs",b.device.programs},{"senses",b.device.senses},{"writebacks",b.writebacks},{"replayed_pages",b.replayed_pages}};
        need(b.registered_globals==b.completed_globals&&!b.host.logical_failed,"joint Host completion");s.service.reset();
    }
    for(const auto& [t,h]:s.original_hash)need(sha256(bytes_of(t))==h,"borrowed weight bytes changed");
    report["borrowed_unchanged"]=true;s.calls.close();s.request_log.close();need(bool(s.calls)&&bool(s.request_log),"joint close logs");save("joint-summary.json",report);s.finished=true;
}
extern "C" void hbflab_m2_abort(){m4_abort();s.service.reset();if(s.mirror){cudaFree(s.mirror);s.mirror=nullptr;}if(s.times){cudaFree(s.times);s.times=nullptr;}}
bool hbflab_m2_precision(const ggml_tensor*,ggml_tensor*){return false;}
bool hbflab_m2_selected_mat(const ggml_tensor*){return false;}
bool hbflab_m2_mul_mat(const ggml_tensor*,const ggml_tensor*,ggml_tensor*,cudaStream_t){return false;}
bool hbflab_m2_rms_norm(const ggml_tensor*,const ggml_tensor*,const ggml_tensor*,float*,int,int,int,int,float,cudaStream_t){return false;}
bool hbflab_m5_mul(ggml_tensor*,cudaStream_t){return false;}
