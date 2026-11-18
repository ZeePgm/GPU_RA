#include "restored_routes.hpp"
#include "integrated_profile.hpp"
#include "integrated_geometry.hpp"
#include "weights_bridge.hpp"
#include "hbflab/async_module.hpp"
#include <cuda.h>
#include "down_embedded.hpp"
#include "r2_half_to_float_embedded.hpp"
#include <algorithm>
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
    const ggml_tensor* norm_weight=nullptr;
    std::unique_ptr<AsyncModuleInterceptor> loader;
    AsyncFunctionHandle dot, unpack;
    std::set<const ggml_tensor*> handled;
    std::map<const ggml_tensor*,size_t> converted;
    std::map<std::string,uint64_t> scalar_totals;
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
    const auto geometry=m6_geometry();
    const uint64_t group=uint64_t(geometry.block_group_dlus())*4096;
    json regions=json::array();
    std::set<std::string> names;
    for(unsigned i=0;i<count;++i){
        const auto* w=weights[i];
        check(w && w->data && ggml_is_contiguous(w) && names.emplace(w->name).second &&
              s.indices.emplace(w,i).second,"invalid or duplicate selected weight");
        auto bytes=ggml_nbytes(w);
        check(bytes && bytes<=std::numeric_limits<unsigned>::max() && bytes%64==0 &&
              bases[i]%4096==0 && bases[i]<=layout && bytes<=layout-bases[i],"unsupported selected range");
        const auto padded=(bytes+group-1)/group*group;
        check(bases[i]%group==0 && padded<=layout-bases[i],"selected group reservation bounds");
        regions.push_back({{"region_id",w->name},{"tensor",w->name},{"global_base",bases[i]},
            {"valid_bytes",bytes},{"registered_bytes",padded},{"source_offset",0},{"source_bytes",bytes}});
    }
    if(s.route==Route::Symbolic){
        check(config && *config,"symbolic requires the second F configuration");
        // The GGUF inventory reserves whole 32 KiB V3 block groups. Program those
        // groups at registration; before_range still reads only the demanded bytes.
        json cfg={{"regions",regions},{"effective_config_path",config},
                  {"pages_per_block",geometry.pages_per_block},{"blocks_per_bank",geometry.blocks_per_bank}};
        s.symbolic=std::make_unique<S3Gate>(s.output,cfg,clock_ns);
        s.symbolic->initialize();s.symbolic->request_begin();
    } else if(s.route==Route::Cuda){
        restored_b2_bind(weights,bases,count,layout,s.output,full);
    } else if(s.route==Route::Ptx){
        const auto& p=integrated_profile();
        AsyncRuntimeOptions o;
        o.timeout_ns=1000000000ull;o.service_timeout_ns=10000000000ull;
        o.backend.geometry=geometry;o.backend.max_events_per_batch=p.max_events;
        o.audit_limit=p.audit_records;o.readonly_layout_bytes=layout;
        o.readonly_reservation_alignment=group;
        o.readonly_bases.assign(bases,bases+count);
        s.ptx_service=std::make_unique<LiveTimingService>(o,512,2);
        for(unsigned i=0;i<count;++i){
            const auto* w=weights[i];
            if(w->type==GGML_TYPE_F32){
                check(!s.norm_weight && ggml_nelements(w)==p.hidden && w->ne[0]==p.hidden &&
                      std::string(w->name).find("norm.weight")!=std::string::npos,"R2 norm layout");
                s.norm_weight=w;
                s.norm.add(w->name,ggml_nbytes(w),ReadonlyStrategy::PtxScalar);
                s.norm.bind(0,w->data);s.norm.start();
            }else{
                check(w->type==GGML_TYPE_F16 && (w->ne[0]==p.hidden || w->ne[0]==p.ffn) &&
                      w->ne[2]==1 && w->ne[3]==1,"R2 F16 matrix layout");
            }
            s.ptx_service->register_readonly(w->data,ggml_nbytes(w));
        }
        s.ptx_service->start();
        if(s.norm_weight)m5_begin(s.ptx_service.get(),2,output,&s.norm);
        if(count>unsigned(s.norm_weight!=nullptr)){
            s.loader=std::make_unique<AsyncModuleInterceptor>(*s.ptx_service);
            json modules=json::object();
            auto load=[&](const std::string& source,const char* entry){
                const auto module=s.loader->load_ptx(source);
                const auto id=s.loader->identity(module);
                modules[entry]={{"source_sha256",id.source_sha256},{"rewritten_sha256",id.rewritten_sha256},
                                {"helper_sha256",id.helper_sha256},{"cubin_sha256",id.cubin_sha256}};
                return s.loader->function(module,entry);
            };
            s.dot=load(embedded_down,"r2_down_raw");
            s.unpack=load(embedded_r2_half_to_float,"r2_half_to_float");
            std::ofstream f(s.output/"r2-modules.json");f<<modules.dump(2);f.close();check(bool(f),"R2 module identity write");
        }
    }
}
void restored_step_begin(int step){
    auto& s=*session;check(!s.active && step==s.step+1,"restored step order");
    s.step=step;s.active=true;s.seen.clear();
    s.handled.clear();s.converted.clear();
    if(s.norm_weight){s.norm.begin_step(step);m5_step();}
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
        if(s.ptx_service){if(w==s.norm_weight)s.norm.expect(0);continue;}
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
namespace {
void audit_scalar(Session& s,const ggml_tensor* w,const void* start,size_t bytes,uint64_t launch){
    s.ptx_service->synchronize();
    const auto records=s.ptx_service->take_audit();
    const auto base=reinterpret_cast<uintptr_t>(start);
    const auto storage=reinterpret_cast<uintptr_t>(w->data);
    check(base>=storage && bytes<=ggml_nbytes(w) && base-storage<=ggml_nbytes(w)-bytes,"R2 launch outside weight");
    check(records.size()==bytes/4,"R2 missing scalar loads");
    std::vector<bool> seen(bytes/4);
    for(const auto& r:records){
        check(r.launch==launch && r.first_bytes==4 && r.span_count==1 && r.end_ps>=r.arrival_ps &&
              r.first_address>=base && r.first_address-base<bytes && (r.first_address-base)%4==0,"R2 scalar role/address");
        const auto index=(r.first_address-base)/4;
        check(!seen[index],"duplicate R2 scalar load");seen[index]=true;
    }
    s.parents+=records.size();s.requested_bytes+=bytes;
}
void convert_pairs(Session& s,const ggml_tensor* w,const char* src,float* dst,size_t bytes,cudaStream_t stream,
                   AsyncOutputMemory memory=AsyncOutputMemory::ContextAllocation){
    // Bound live audit memory independently of the matrix size.
    constexpr size_t batch_bytes=4096;
    while(bytes){
        const size_t n=std::min(bytes,batch_bytes);
        unsigned pairs=unsigned(n/4);
        auto launch=s.loader->launch(s.unpack,{1,256},{&src,&dst,&pairs},{dst,n*2},stream,{},memory);
        audit_scalar(s,w,src,n,launch);
        src+=n;dst+=2*pairs;bytes-=n;
    }
}
}
// Replaces only the existing F16 -> F32 conversion on the native cuBLAS path.
// The allocation and SGEMM belong to ggml; there is no extra weight buffer.
bool restored_ptx_convert(const ggml_tensor* part,float* dst,cudaStream_t stream){
    if(!session || !session->ptx_service || !session->active || part->type!=GGML_TYPE_F16)return false;
    auto& s=*session;
    const auto address=reinterpret_cast<uintptr_t>(part->data);
    for(const auto& item:s.indices){
        const auto* w=item.first;
        const auto base=reinterpret_cast<uintptr_t>(w->data);
        if(address<base || address-base>=ggml_nbytes(w))continue;
        check(s.seen.count(w) && std::string(w->name)==part->name && part->nb[1]==part->ne[0]*sizeof(uint16_t) &&
              ggml_is_matrix(part),"R2 cuBLAS weight identity");
        const size_t offset=address-base,bytes=ggml_nelements(part)*sizeof(uint16_t);
        check(offset==s.converted[w] && bytes<=ggml_nbytes(w)-offset,"R2 cuBLAS weight chunk order");
        CUcontext owner=nullptr;
        check(cuPointerGetAttribute(&owner,CU_POINTER_ATTRIBUTE_CONTEXT,reinterpret_cast<CUdeviceptr>(dst))==CUDA_SUCCESS,
              "cuBLAS workspace context query");
        convert_pairs(s,w,static_cast<const char*>(part->data),dst,bytes,stream,
                      owner?AsyncOutputMemory::ContextAllocation:AsyncOutputMemory::GgmlMappedWorkspace);
        s.scalar_totals["cublas_f16_to_f32"]+=bytes/4;
        s.converted[w]+=bytes;
        if(s.converted[w]==ggml_nbytes(w))check(s.handled.insert(w).second,"duplicate converted matrix");
        return true;
    }
    return false;
}
bool restored_ptx_mul(ggml_tensor* node,cudaStream_t stream){
    if(!session || !session->ptx_service)return false;
    auto& rt=*session;
    const auto* matrix=node->src[0];
    if((node->op==GGML_OP_GET_ROWS || node->op==GGML_OP_MUL_MAT) && rt.indices.count(matrix)){
        const auto* other=node->src[1];
        check(rt.active && rt.seen.count(matrix) && matrix->type==GGML_TYPE_F16 && node->type==GGML_TYPE_F32 &&
              ggml_is_contiguous(node) && ggml_is_contiguous(other),"R2 consumer layout");
        unsigned width=unsigned(matrix->ne[0]),height=unsigned(matrix->ne[1]);
        auto* dst=static_cast<float*>(node->data);
        const auto* weight=static_cast<const char*>(matrix->data);
        if(node->op==GGML_OP_GET_ROWS){
            check(other->type==GGML_TYPE_I32 && ggml_nelements(node)==width*ggml_nelements(other),"R2 gather shape");
            std::vector<int32_t> ids(ggml_nelements(other));
            check(cudaMemcpyAsync(ids.data(),other->data,ids.size()*4,cudaMemcpyDeviceToHost,stream)==cudaSuccess &&
                  cudaStreamSynchronize(stream)==cudaSuccess,"R2 gather indices");
            for(auto row:ids){
                check(row>=0 && unsigned(row)<height,"R2 gather row out of bounds");
                convert_pairs(rt,matrix,weight+row*matrix->nb[1],dst,width*2,stream);
                dst+=width;rt.scalar_totals["get_rows"]+=width/2;
            }
        }else{
            check(other->type==GGML_TYPE_F32 && other->ne[0]==width && ggml_is_matrix(other) &&
                  ggml_nelements(node)==height*other->ne[1],"R2 dot shape");
            if(other->ne[1]>16)return false; // Native cuBLAS consumes the registered F16 weight via restored_ptx_convert.
            for(int64_t token=0;token<other->ne[1];++token){
                const auto* x=static_cast<const float*>(other->data)+token*width;
                for(unsigned row=0;row<height;++row){
                    const auto* src=weight+row*matrix->nb[1];
                    auto* y=dst+token*height+row;
                    auto launch=rt.loader->launch(rt.dot,{1,256},{&src,&x,&y,&height,&width},{y,4},stream,{x,width*4});
                    audit_scalar(rt,matrix,src,width*2,launch);
                    rt.scalar_totals["r2_dot"]+=width/2;
                }
            }
        }
        check(rt.handled.insert(matrix).second,"duplicate R2 selected consumer");
        return true;
    }
    if(node->op!=GGML_OP_MUL || !rt.norm_weight || node->src[1]!=rt.norm_weight)return false;
    auto& s=*session;const auto* x=node->src[0];const auto* w=node->src[1];
    check(s.active && x->type==GGML_TYPE_F32 && node->type==GGML_TYPE_F32 &&
          ggml_is_contiguous(x) && ggml_is_contiguous(node) && x->ne[0]==w->ne[0] &&
          x->ne[1]>=1 && x->ne[1]<=integrated_profile().ubatch && x->ne[2]==1 && x->ne[3]==1 &&
          ggml_nelements(node)==ggml_nelements(x),"R2 norm multiply shape");
    s.norm.handled(0);
    m5_mul(static_cast<const float*>(x->data),static_cast<const float*>(w->data),
           static_cast<float*>(node->data),unsigned(x->ne[1]),stream);
    check(s.handled.insert(w).second,"duplicate R2 norm consumer");
    s.ptx_service->synchronize();
    const auto records=s.ptx_service->take_audit();

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
    s.scalar_totals["r2_norm"]+=records.size();
    return true;
}
void restored_step_end(){
    auto& s=*session;check(s.active && s.seen.size()==s.indices.size(),"selected consumer coverage");
    if(s.ptx_service)check(s.handled==s.seen,"R2 selected compute hook coverage");
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
        if(s.norm_weight)m5_finish();
        s.loader.reset();s.ptx_service->finish();const auto stats=s.ptx_service->statistics();
        check(s.ptx_service->drained() && !stats.audit_dropped && stats.submitted==s.parents &&
              stats.backend.registered_globals==stats.backend.completed_globals &&
              stats.completed==s.parents && !stats.backend.host.logical_failed,"R2 accounting/drain");
        if(s.norm_weight)s.norm.finish();
    }
    std::ofstream f(s.output/"route-summary.json");
    f<<json({{"steps",s.step+1},{"consumers",s.calls},{"range_or_scalar_calls",s.parents},
             {"scalar_requests_by_consumer",s.scalar_totals},{"requested_bytes",s.requested_bytes},{"drained",true}}).dump(2);f.close();
    check(bool(f),"restored summary write");
}
void restored_abort(){
    if(session && session->norm_weight)m5_abort();
    if(session && session->route==Route::Cuda)hbflab_m2_abort();
    session.reset();
}
