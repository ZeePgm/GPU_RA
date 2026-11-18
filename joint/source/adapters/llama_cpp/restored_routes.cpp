#include "restored_routes.hpp"
#include "integrated_profile.hpp"
#include "integrated_geometry.hpp"
#include "weights_bridge.hpp"
#include "hbflab/readonly_access.hpp"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <limits>
namespace fs=std::filesystem;
using json=nlohmann::ordered_json;
#include "../../experiments/gpu_ffn/gate.hpp"

namespace {
using namespace hbflab;
enum class Route { Native, Symbolic, Ptx, Cuda };
struct Session {
    Route route=Route::Native;
    fs::path output;
    std::map<const ggml_tensor*,unsigned> indices;
    std::set<const ggml_tensor*> seen;
    std::unique_ptr<S3Gate> symbolic;
    std::unique_ptr<LiveTimingService> ptx_service;
    ReadonlyAccess norm;
    int step=-1;
    bool active=false;
    uint64_t calls=0, parents=0, requested_bytes=0;
};
std::unique_ptr<Session> session;
void check(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
Route choose(const std::string& route){
    if(route=="symbolic")return Route::Symbolic;
    if(route=="ptx")return Route::Ptx;
    if(route=="cuda")return Route::Cuda;
    if(route=="native")return Route::Native;
    throw std::invalid_argument("unknown selected route");
}
uint64_t clock_ns(){
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}
void restored_bind(const char* route,const ggml_tensor* const* weights,const uint64_t* bases,
                   unsigned count,uint64_t layout,const char* profile,const char* config,
                   const char* output,bool full){
    check(!session && count && weights && bases && profile,"invalid restored binding");
    session=std::make_unique<Session>();auto& s=*session;
    s.route=choose(route);s.output=output;
    m6_load_profile(profile,output);
    json regions=json::array();
    std::set<std::string> names;
    for(unsigned i=0;i<count;++i){
        const auto* w=weights[i];
        check(w && w->data && ggml_is_contiguous(w) && names.emplace(w->name).second &&
              s.indices.emplace(w,i).second,"invalid or duplicate selected weight");
        auto bytes=ggml_nbytes(w);
        check(bytes && bytes<=std::numeric_limits<unsigned>::max() && bytes%64==0 &&
              bases[i]%4096==0 && bases[i]<=layout && bytes<=layout-bases[i],"unsupported selected range");
        const auto padded=(bytes+4095)/4096*4096;
        regions.push_back({{"region_id",w->name},{"tensor",w->name},{"global_base",bases[i]},
            {"valid_bytes",bytes},{"registered_bytes",padded},{"source_offset",0},{"source_bytes",bytes}});
    }
    if(s.route==Route::Symbolic){
        check(config && *config,"symbolic requires the second F configuration");
        std::ifstream file(config);
        check(bool(file),"missing second F configuration");
        const auto geometry=hbf_eventlab::read_configuration(file).device.geometry;
        // Enlarge only the number of blocks to fit the declared GGUF address space.
        check(layout && layout<=(uint64_t(1)<<32),"selected layout exceeds supported model size");
        auto blocks=geometry.blocks_per_bank;
        while(geometry.channel_bytes()*(blocks/geometry.blocks_per_bank)<layout)blocks*=2;
        json cfg={{"regions",regions},{"effective_config_path",config},{"blocks_per_bank",blocks}};
        s.symbolic=std::make_unique<S3Gate>(s.output,cfg,clock_ns);
        s.symbolic->initialize();s.symbolic->request_begin();
    } else if(s.route==Route::Cuda){
        restored_b2_bind(weights,bases,count,layout,s.output,full);
    } else if(s.route==Route::Ptx){
        const auto* w=weights[0];const auto& p=integrated_profile();
        check(count==1 && w->type==GGML_TYPE_F32 && w->ne[0]==p.hidden &&
              ggml_nelements(w)==p.hidden && std::string(w->name).find("norm.weight")!=std::string::npos,
              "R2 model PTX supports one contiguous F32 norm multiply; down row 0 uses r2_down");
        s.norm.add(w->name,ggml_nbytes(w),ReadonlyStrategy::PtxScalar);
        s.norm.bind(0,w->data);s.norm.start();
        AsyncRuntimeOptions o;o.timeout_ns=1000000000ull;o.service_timeout_ns=10000000000ull;
        o.backend.geometry=m6_geometry();o.backend.max_events_per_batch=p.max_events;
        o.audit_limit=p.audit_records;o.readonly_layout_bytes=layout;o.readonly_bases={bases[0]};
        s.ptx_service=std::make_unique<LiveTimingService>(o,512,2);
        s.ptx_service->register_readonly(w->data,ggml_nbytes(w));s.ptx_service->start();
        m5_begin(s.ptx_service.get(),2,output,&s.norm);
    }
}
void restored_step_begin(int step){
    auto& s=*session;check(!s.active && step==s.step+1,"restored step order");
    s.step=step;s.active=true;s.seen.clear();
    if(s.ptx_service){s.norm.begin_step(step);m5_step();}
}
void restored_before(const ggml_tensor* node){
    auto& s=*session;check(s.active,"restored graph outside step");
    if(node->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(const_cast<ggml_tensor*>(node),GGML_PREC_F32);
    for(unsigned slot=0;slot<GGML_MAX_SRC;++slot){
        const auto* w=node->src[slot];auto it=s.indices.find(w);
        if(it==s.indices.end())continue;
        check(s.seen.insert(w).second,"duplicate selected consumer");
        const bool gather=std::string(w->name)=="token_embd.weight";
        const bool norm=std::string(w->name).find("norm.weight")!=std::string::npos;
        check(node->op==(gather?GGML_OP_GET_ROWS:norm?GGML_OP_MUL:GGML_OP_MUL_MAT) &&
              slot==(norm?1u:0u),"selected consumer layout/op unsupported");
        ++s.calls;
        if(s.ptx_service){s.norm.expect(0);continue;}
        if(s.route==Route::Native)continue;
        auto access=[&](unsigned offset,unsigned bytes){
            if(s.symbolic)s.symbolic->before_range(it->second,offset,bytes,s.step);
            else restored_b2_read(w,offset,bytes,s.step);
            ++s.parents;s.requested_bytes+=bytes;
        };
        if(gather){
            const auto* ids=node->src[1];
            check(ids && ids->type==GGML_TYPE_I32 && ggml_is_contiguous(ids),"invalid embedding indices");
            std::vector<int32_t> rows(ggml_nelements(ids));
            ggml_backend_tensor_get(ids,rows.data(),0,rows.size()*sizeof(int32_t));
            for(auto row:rows){
                check(row>=0 && row<w->ne[1],"embedding row out of bounds");
                access(unsigned(row*w->nb[1]),unsigned(w->nb[1]));
            }
        }else access(0,unsigned(ggml_nbytes(w)));
    }
}
bool restored_ptx_mul(ggml_tensor* node,cudaStream_t stream){
    if(!session || !session->ptx_service || node->op!=GGML_OP_MUL ||
       !session->indices.count(node->src[1]))return false;
    auto& s=*session;const auto* x=node->src[0];const auto* w=node->src[1];
    check(s.active && x->type==GGML_TYPE_F32 && node->type==GGML_TYPE_F32 &&
          ggml_is_contiguous(x) && ggml_is_contiguous(node) && x->ne[0]==w->ne[0] &&
          x->ne[1]>=1 && x->ne[1]<=integrated_profile().ubatch && x->ne[2]==1 && x->ne[3]==1 &&
          ggml_nelements(node)==ggml_nelements(x),"R2 norm multiply shape");
    s.norm.handled(0);
    m5_mul(static_cast<const float*>(x->data),static_cast<const float*>(w->data),
           static_cast<float*>(node->data),unsigned(x->ne[1]),stream);
    return true;
}
void restored_step_end(){
    auto& s=*session;check(s.active && s.seen.size()==s.indices.size(),"selected consumer coverage");
    if(s.ptx_service){
        s.ptx_service->synchronize();
        const auto records=s.ptx_service->take_audit();
        const auto* w=s.indices.begin()->first;
        const auto base=reinterpret_cast<uintptr_t>(w->data);
        check(records.size()==m5_step_requests(),"R2 scalar request count");
        std::map<uint64_t,std::set<uintptr_t>> addresses;
        for(const auto& r:records){
            check(r.first_bytes==4 && r.span_count==1 && r.first_address>=base &&
                  r.first_address-base<ggml_nbytes(w) && (r.first_address-base)%4==0 &&
                  r.end_ps>=r.arrival_ps && addresses[r.launch].insert(r.first_address).second,
                  "R2 scalar address/identity");
        }
        for(const auto& row:addresses)check(row.second.size()==size_t(w->ne[0]),"R2 row coverage");
        s.parents+=records.size();s.requested_bytes+=4*records.size();s.norm.end_step(true);
    }
    s.active=false;
}
void restored_finish(){
    auto& s=*session;check(!s.active && s.step>=0,"unfinished restored step");
    if(s.symbolic){
        const auto& r=s.symbolic->finish();
        check(r.at("workload_success") && r.at("host_drained") && r.at("device_drained") &&
              r.at("online_audit").at("read_fragments").get<uint64_t>()*64==s.requested_bytes,
              "symbolic completion/online coverage");
    }
    if(s.route==Route::Cuda)restored_b2_finish();
    if(s.ptx_service){
        m5_finish();s.ptx_service->finish();const auto stats=s.ptx_service->statistics();
        check(s.ptx_service->drained() && !stats.audit_dropped && stats.submitted==s.parents &&
              stats.completed==s.parents && !stats.backend.host.logical_failed,"R2 accounting/drain");
        s.norm.finish();
    }
    std::ofstream f(s.output/"route-summary.json");
    f<<json({{"steps",s.step+1},{"consumers",s.calls},{"range_or_scalar_calls",s.parents},
             {"requested_bytes",s.requested_bytes},{"drained",true}}).dump(2);f.close();
    check(bool(f),"restored summary write");
}
void restored_abort(){
    if(session && session->ptx_service)m5_abort();
    if(session && session->route==Route::Cuda)hbflab_m2_abort();
    session.reset();
}
