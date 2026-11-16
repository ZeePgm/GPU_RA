#include "integrated_profile.hpp"
#include "integrated_geometry.hpp"
#include "paged_bridge.hpp"
#include "hbflab/capacity_gpu.hpp"
#include "hbflab/kv.hpp"
#include "hbflab/ptx.hpp"
#include <cuda_fp16.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <map>
using namespace hbflab;
using json=nlohmann::ordered_json;
namespace {
void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
// No timing model in bypass; identical real page/lease/attention data path.
struct Bypass:CapacityMedia {
    std::uint64_t reserve_media(std::uint64_t) override{return 0;}
    ReadResult read_media(std::uint64_t,std::uint32_t n) override{return {Status::Ok,0,0,n/64};}
    WriteResult write_media(std::uint64_t,const std::array<std::uint64_t,64>&) override{return {Status::Ok,0,0,0};}
    BackendStats media_statistics() override{return {};}
    std::uint64_t media_time_ps() override{return 0;}
    void abort_media() override{}
};
CapacityOptions capacity_options(){CapacityOptions o;o.logical_pages=integrated_profile().logical_pages;o.frames=integrated_profile().frames;o.backend.geometry=m6_geometry();return o;}
struct StorageConfig {
    CapacityOptions options=capacity_options();bool active=false,prefetch=false;double gpu_watts=0;
    json record;
    StorageConfig(const std::filesystem::path& output,bool modeled) {
        const char* path=std::getenv("HBFLAB_STORAGE_CONFIG");if(!path || !*path)return;
        std::ifstream input(std::filesystem::u8path(path));need(bool(input),"storage config open failed");json config;input>>config;
        need(config.is_object() && config.value("version",0)==1,"storage config version");
        for(const auto& item:config.items())need(item.key()=="version" || item.key()=="file_backing" || item.key()=="prefetch" || item.key()=="gpu_watts" || item.key()=="environment","unknown storage config field");
        active=true;prefetch=config.value("prefetch",false);gpu_watts=config.value("gpu_watts",0.0);need(std::isfinite(gpu_watts)&&gpu_watts>=0,"invalid supplied GPU watts");
        if(config.value("file_backing",false)){options.backing_mode=BackingMode::CreateFile;options.backing_path=output/"capacity.pages";}
        options.prefetch_evict_clean=prefetch;
        auto& e=options.environment;
        if(config.contains("environment")) {
            const auto& env=config["environment"];need(env.is_object(),"environment config object required");
            for(const auto& item:env.items())need(item.key()=="enabled" || item.key()=="ambient_c" || item.key()=="initial_c" || item.key()=="resistance_k_per_w" || item.key()=="capacitance_j_per_k" || item.key()=="gpu_coupling" || item.key()=="idle_w" || item.key()=="read_j_per_byte" || item.key()=="program_j_per_byte" || item.key()=="reference_c" || item.key()=="retention_seconds" || item.key()=="half_life_degrees" || item.key()=="refresh_fraction" || item.key()=="refresh_budget","unknown environment field");
            e.enabled=env.value("enabled",false);
#define HBFLAB_ENV_VALUE(name) e.name=env.value(#name,e.name)
            HBFLAB_ENV_VALUE(ambient_c);HBFLAB_ENV_VALUE(initial_c);HBFLAB_ENV_VALUE(resistance_k_per_w);HBFLAB_ENV_VALUE(capacitance_j_per_k);
            HBFLAB_ENV_VALUE(gpu_coupling);HBFLAB_ENV_VALUE(idle_w);HBFLAB_ENV_VALUE(read_j_per_byte);HBFLAB_ENV_VALUE(program_j_per_byte);
            HBFLAB_ENV_VALUE(reference_c);HBFLAB_ENV_VALUE(retention_seconds);HBFLAB_ENV_VALUE(half_life_degrees);HBFLAB_ENV_VALUE(refresh_fraction);HBFLAB_ENV_VALUE(refresh_budget);
#undef HBFLAB_ENV_VALUE
        }
        EnvironmentModel validate(e); // Reject invalid profiles even in bypass.
        record={{"version",1},{"requested",config},{"file_backing",options.backing_mode!=BackingMode::Memory},{"prefetch",prefetch},
            {"gpu_watts",gpu_watts},{"power_source","explicit configuration; not measured telemetry"},{"calibrated",false},
            {"environment_active",modeled&&e.enabled},{"sampling","model media completion and layer boundary"},
            {"environment",{{"enabled",e.enabled},{"ambient_c",e.ambient_c},{"initial_c",e.initial_c},{"resistance_k_per_w",e.resistance_k_per_w},
              {"capacitance_j_per_k",e.capacitance_j_per_k},{"gpu_coupling",e.gpu_coupling},{"idle_w",e.idle_w},{"read_j_per_byte",e.read_j_per_byte},
              {"program_j_per_byte",e.program_j_per_byte},{"reference_c",e.reference_c},{"retention_seconds",e.retention_seconds},
              {"half_life_degrees",e.half_life_degrees},{"refresh_fraction",e.refresh_fraction},{"refresh_budget",e.refresh_budget}}}};
        e.enabled=e.enabled&&modeled;
        std::ofstream saved(output/"storage-config.json");saved<<record.dump(2)<<'\n';saved.close();need(bool(saved),"storage config write failed");
    }
};
KvOptions kv_options(){KvOptions o;const auto& p=integrated_profile();o.layers=p.layers;o.heads=p.kv_heads;o.head_dimension=p.head_dim;o.element_bytes=2;o.tokens_per_block=p.block_tokens;o.max_tokens=p.context;o.max_sequences=16;
    const char* setting=std::getenv("HBFLAB_KV_CANCELLABLE");need(!setting || std::string(setting)=="0" || std::string(setting)=="1","invalid cancellable KV setting");
    o.cancellable_append=setting && std::string(setting)=="1";return o;}
__global__ void pack(const float* k,const float* v,half* hk,half* hv,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){hk[i]=__float2half_rn(k[i]);hv[i]=__float2half_rn(v[i]);}}
__global__ void init_attention(float* out,float* ml,unsigned n,unsigned dim){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*dim)out[i]=0;if(i<n){ml[i*2]=-INFINITY;ml[i*2+1]=0;}}
struct PageRow {
    const half* pages[3]{};unsigned offset=0;
    __device__ float get(unsigned index) const {index+=offset;return __half2float(pages[index/2048][index%2048]);}
};
// A full-row group uses one page; a split row pins every fragment until the kernel completes.
__global__ void page_attention(const float* q,PageRow k,PageRow v,float* out,float* ml,
                               unsigned past,unsigned first,unsigned count,unsigned heads,unsigned kv_heads,unsigned dim,float scale) {
    const unsigned h=blockIdx.x%heads,r=blockIdx.x/heads,d=threadIdx.x,idx=r*heads+h;
    __shared__ float partial[8],a,b;
    float acc=out[idx*dim+d],m=ml[idx*2],l=ml[idx*2+1];
    for(unsigned j=0;j<count;++j) {
        if(first+j>past+r)break;
        const unsigned address=j*kv_heads*dim+(h/(heads/kv_heads))*dim+d;
        float dot=q[idx*dim+d]*k.get(address);
        for(int shift=16;shift;shift/=2)dot+=__shfl_down_sync(0xffffffff,dot,shift);
        if((d&31)==0)partial[d/32]=dot;
        __syncthreads();
        if(d==0){float score=partial[0];for(unsigned w=1;w<dim/32;++w)score+=partial[w];score*=scale;
            const float next=fmaxf(m,score);a=expf(m-next);b=expf(score-next);ml[idx*2]=next;ml[idx*2+1]=l*a+b;}
        __syncthreads();m=ml[idx*2];l=ml[idx*2+1];acc=acc*a+b*v.get(address);
        __syncthreads();
    }
    out[idx*dim+d]=acc;
}
__global__ void normalize(float* out,const float* ml,unsigned n,unsigned dim){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]/=ml[(i/dim)*2+1];}
struct SequenceState {KvSequence sequence{};KvAppend tx{};unsigned layer=0,batch=0,past=0;};
struct State {
    int paging_mode=0; // 0 sync, 1 asynchronous transfer, 2 bounded lookahead
    std::map<int,SequenceState> sequences;int sequence_id=0;std::size_t peak_sequences=0,peak_group_scratch=0,group_count=0,group_tokens=0;
    Bypass bypass;LiveTimingService* service=nullptr;
    std::unique_ptr<CapacityGpu> gpu;std::unique_ptr<KvCache> kv;KvSequence sequence{};KvAppend tx{};
    std::unique_ptr<StorageConfig> storage;
    half* packed=nullptr;float* ml=nullptr;
    std::vector<half> keys,values;std::filesystem::path output;std::ofstream calls,media;
    unsigned step=0,layer=0,batch=0,past=0;bool active=false;std::uint64_t page_pairs=0,appended=0,media_count=0,copy_bytes=0,cross_page_rows=0,transactions=0;
    ~State(){kv.reset();gpu.reset();if(packed)cudaFree(packed);if(ml)cudaFree(ml);}
    void drain(){if(service)for(const auto& t:service->take_media_audit()){
        media<<json({{"step",step},{"layer",layer},{"ticket",t.ticket},{"dispatch",t.dispatch},{"producer",t.producer},
          {"write",t.write},{"logical",t.logical},{"bytes",t.bytes},{"floor_ns",t.floor_ns},{"arrival_ns",t.arrival_ns},
          {"arrival_ps",t.arrival_ps},{"end_ps",t.end_ps},{"replayed_pages",t.replayed_pages},
          {"ready_wall_ns",t.ready_wall_ns},{"completion_wall_ns",t.completion_wall_ns},{"resumed_wall_ns",t.resumed_wall_ns},{"delay_ns",t.delay_ns}}).dump()<<'\n';++media_count;}
        need(bool(media),"M4 media audit failed");}
};
std::unique_ptr<State> state;
}
bool m4_is_attention(const ggml_tensor* t){int tag=0;std::memcpy(&tag,reinterpret_cast<const char*>(t->op_params)+60,4);return t->op==GGML_OP_CUSTOM && tag==0x4d34564b;}
void m4_begin(int mode,LiveTimingService* service,const char* output){
    if(mode==0)return;need(!state,"M4 already initialized");state=std::make_unique<State>();auto& s=*state;s.service=service;s.output=output;
    s.calls.open(s.output/"kv-attention.jsonl");s.media.open(s.output/"kv-media.jsonl");need(s.calls && s.media,"M4 audit open failed");
    s.storage=std::make_unique<StorageConfig>(s.output,service!=nullptr);
    if(const char* setting=std::getenv("HBFLAB_KV_PAGING")){
        const std::string value=setting;need(value=="sync" || value=="async" || value=="prefetch","invalid KV paging mode");s.paging_mode=value=="prefetch"?2:value=="async"?1:0;
    }
    CapacityGpuAsyncOptions async;if(s.paging_mode){
        async.staging_pages=s.storage->options.frames;
        if(const char* setting=std::getenv("HBFLAB_KV_STAGING_PAGES")){
            const std::string value=setting;std::size_t end=0;
            const auto slots=std::stoull(value,&end);
            need(end==value.size() && slots>0 && slots<=s.storage->options.frames,"invalid staging page budget");
            async.staging_pages=static_cast<std::size_t>(slots);
        }
    }
    s.gpu=std::make_unique<CapacityGpu>(s.storage->options,service?static_cast<CapacityMedia*>(service):&s.bypass,async);
    s.kv=std::make_unique<KvCache>(s.gpu->store(),kv_options());s.sequence=s.kv->create();s.sequences.emplace(0,SequenceState{s.sequence});s.peak_sequences=1;
    ck(cudaMalloc(reinterpret_cast<void**>(&s.packed),2*integrated_profile().ubatch*integrated_profile().kv_elements()*sizeof(half)));ck(cudaMalloc(reinterpret_cast<void**>(&s.ml),integrated_profile().ubatch*integrated_profile().q_heads*2*sizeof(float)));
}
void m4_step_begin(int step){if(!state)return;auto& s=*state;need(!s.active && step==s.step,"M4 step order");s.active=true;s.layer=0;s.batch=0;for(auto& item:s.sequences){need(item.second.layer==0,"unfinished sequence transaction");item.second.batch=0;}}
void attention_one(ggml_tensor* dst,cudaStream_t stream){
    need(state && state->active && m4_is_attention(dst),"M6 attention lifecycle");auto& s=*state;const auto& p=integrated_profile();
    const unsigned ke=p.kv_elements(),qe=p.q_elements(),row_bytes=ke*2;
    int layer;std::memcpy(&layer,reinterpret_cast<const char*>(dst->op_params)+56,4);
    auto q=dst->src[0],k=dst->src[1],v=dst->src[2];const unsigned n=static_cast<unsigned>(q->ne[2]);
    need(layer==s.layer && layer<p.layers && n>=1 && n<=p.ubatch && q->type==GGML_TYPE_F32 && k->type==GGML_TYPE_F32 && v->type==GGML_TYPE_F32 &&
      ggml_is_contiguous(q)&&ggml_is_contiguous(k)&&ggml_is_contiguous(v) && q->ne[0]==p.head_dim && q->ne[1]==p.q_heads && q->ne[3]==1 &&
      k->ne[0]==p.head_dim && v->ne[0]==p.head_dim && k->ne[1]==p.kv_heads && v->ne[1]==p.kv_heads && k->ne[2]==n && v->ne[2]==n && k->ne[3]==1 && v->ne[3]==1 &&
      ggml_nelements(dst)==n*qe,"M6 unsupported attention shape/order");
    need(dst->src[3] && dst->src[3]->type==GGML_TYPE_I32 && ggml_nelements(dst->src[3])==n,"M6 position tensor required");
    if(!layer){s.past=static_cast<unsigned>(s.kv->length(s.sequence));std::vector<int> positions(n);ck(cudaMemcpyAsync(positions.data(),dst->src[3]->data,n*4,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
        for(unsigned i=0;i<n;++i)need(positions[i]==s.past+i,"M6 requires contiguous append positions");
        s.batch=n;s.tx=s.kv->begin_append(s.sequence,n);
    }else need(n==s.batch,"M6 inconsistent layer batch");
    pack<<<(n*ke+255)/256,256,0,stream>>>(static_cast<const float*>(k->data),static_cast<const float*>(v->data),s.packed,s.packed+p.ubatch*ke,n*ke);ck(cudaGetLastError());
    const char* write_mode=std::getenv("HBFLAB_KV_WRITE_MODE");
    need(!write_mode || std::string(write_mode)=="host" || std::string(write_mode)=="device","invalid KV write mode");
    const bool device_write=write_mode && std::string(write_mode)=="device";
    const char* reuse_setting=std::getenv("HBFLAB_KV_REUSE_PACKED");
    need(!reuse_setting || std::string(reuse_setting)=="0" || std::string(reuse_setting)=="1","invalid packed reuse setting");
    const bool reuse_packed=device_write && (!reuse_setting || std::string(reuse_setting)=="1");
    const char* async_replica_setting=std::getenv("HBFLAB_KV_ASYNC_REPLICA");
    need(!async_replica_setting || std::string(async_replica_setting)=="0" || std::string(async_replica_setting)=="1","invalid async replica setting");
    const bool async_replica=!async_replica_setting || std::string(async_replica_setting)=="1";
    if(device_write) {
        s.kv->write_layer_with(s.tx,layer,n*row_bytes,[&](KvKind kind,KvSlice slice,std::size_t offset) {
            const auto* source=reinterpret_cast<const unsigned char*>(s.packed+(kind==KvKind::Key?0:p.ubatch*ke));
            s.gpu->write_device(slice.allocation,slice.offset,source+offset,slice.bytes,stream);
        });
        s.drain();
    } else {
    s.keys.resize(n*ke);s.values.resize(n*ke);
    ck(cudaMemcpyAsync(s.keys.data(),s.packed,n*row_bytes,cudaMemcpyDeviceToHost,stream));ck(cudaMemcpyAsync(s.values.data(),s.packed+p.ubatch*ke,n*row_bytes,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
    s.copy_bytes+=n*row_bytes*2;s.kv->write_layer(s.tx,layer,s.keys.data(),s.values.data(),n*row_bytes);s.drain();
    }
    auto out=static_cast<float*>(dst->data);init_attention<<<(n*qe+255)/256,256,0,stream>>>(out,s.ml,n*p.q_heads,p.head_dim);ck(cudaGetLastError());
    const auto before=s.gpu->store().statistics();unsigned pairs=0,cross=0;
    for(unsigned first=0;first<s.past+n;){
        auto ks=s.kv->locate_layer(s.tx,layer,first,KvKind::Key),vs=s.kv->locate_layer(s.tx,layer,first,KvKind::Value);
        const unsigned offset=static_cast<unsigned>(ks.offset%4096);
        const unsigned pages=(offset+row_bytes+4095)/4096;
        unsigned count=pages>1?1:(4096-offset)/row_bytes;
        count=std::min(count,std::min(s.past+n-first,p.block_tokens-first%p.block_tokens));
        need(count>0&&pages<=3&&ks.offset==vs.offset,"M6 invalid page plan");
        PageRow kr{},vr{};kr.offset=vr.offset=offset/2;std::vector<GpuPageLease> leases;
        // The pack buffer already exists in both paths and lives through this
        // layer's attention. Only wholly covered current-batch pages qualify;
        // old-prefix/mixed/tail pages use the normal host-authority upload.
        auto replica=[&](KvSlice slice,std::size_t index,KvKind kind,unsigned token)->const void* {
            const auto valid=std::min(std::size_t{4096},s.gpu->store().size(slice.allocation)-index*4096);
            const auto origin=(std::int64_t(token-token%p.block_tokens)-s.past)*row_bytes+std::int64_t(index*4096);
            if(reuse_packed && (!s.paging_mode || async_replica) && origin>=0 && std::uint64_t(origin)+valid<=std::uint64_t(n)*row_bytes) {
                const auto* source=reinterpret_cast<const unsigned char*>(s.packed+(kind==KvKind::Key?0:p.ubatch*ke));
                return source+origin;
            }
            return nullptr;
        };
        auto acquire_read=[&](KvSlice slice,unsigned page,KvKind kind) {
            const auto index=slice.offset/4096+page;
            if(const auto* source=replica(slice,index,kind,first)) {
                // write_device has completed pack/source writes. packed remains
                // immutable until this layer's drain, including speculative uses.
                if(s.paging_mode)return s.gpu->acquire_read_device_async(slice.allocation,index,source,nullptr,stream);
                return s.gpu->acquire_read_device(slice.allocation,index,source,stream);
            }
            return s.gpu->acquire(slice.allocation,index,PageAccess::Read,stream);
        };
        try {
            for(unsigned page=0;page<pages;++page){
                if(s.storage->prefetch){s.gpu->prefetch(ks.allocation,ks.offset/4096+page,1);s.gpu->prefetch(vs.allocation,vs.offset/4096+page,1);}
                auto kl=acquire_read(ks,page,KvKind::Key);leases.push_back(kl);
                auto vl=acquire_read(vs,page,KvKind::Value);leases.push_back(vl);
                kr.pages[page]=static_cast<const half*>(s.gpu->data(kl));vr.pages[page]=static_cast<const half*>(s.gpu->data(vl));
            }
            page_attention<<<n*p.q_heads,p.head_dim,0,stream>>>(static_cast<const float*>(q->data),kr,vr,out,s.ml,s.past,first,count,p.q_heads,p.kv_heads,p.head_dim,1.0f/std::sqrt(float(p.head_dim)));ck(cudaGetLastError());
            while(!leases.empty()){if(s.paging_mode==2)s.gpu->finish_async(leases.back());else s.gpu->finish(leases.back());leases.pop_back();}
            if(s.paging_mode==2 && first+count<s.past+n){
                // The next attention span is known, so this is exact lookahead,
                // not a guess across requests or unrelated sequence IDs.
                auto nk=s.kv->locate_layer(s.tx,layer,first+count,KvKind::Key),nv=s.kv->locate_layer(s.tx,layer,first+count,KvKind::Value);
                auto prefetch=[&](KvSlice slice,KvKind kind){
                    const auto index=slice.offset/4096;
                    if(const auto* source=replica(slice,index,kind,first+count))s.gpu->try_prefetch_device(slice.allocation,index,source,nullptr);
                    else s.gpu->try_prefetch(slice.allocation,index);
                };
                prefetch(nk,KvKind::Key);prefetch(nv,KvKind::Value);
            }
        }catch(...){for(auto lease:leases){try{s.gpu->finish(lease);}catch(...){}}throw;}
        ++pairs;if(pages>1)++cross;s.drain();first+=count;
    }
    normalize<<<(n*qe+255)/256,256,0,stream>>>(out,s.ml,n*qe,p.head_dim);ck(cudaGetLastError());
    if(s.paging_mode)s.gpu->drain();
    const auto after=s.gpu->store().statistics();
    s.calls<<json({{"source_groups",s.group_count},{"source_tokens",s.group_tokens},{"sequence",s.sequence_id},{"step",s.step},{"transaction",s.transactions},{"layer",layer},{"past",s.past},{"batch",n},{"visible",s.past+n},{"committed",s.kv->length(s.sequence)},
      {"page_pairs",pairs},{"cross_page_rows",cross},{"read_faults",after.faults-before.faults},{"frame_bytes",s.gpu->device_bytes()}}).dump()<<'\n';need(bool(s.calls),"M6 attention audit failed");
    s.page_pairs+=pairs;s.cross_page_rows+=cross;s.appended+=n;++s.layer;
    if(s.layer==p.layers){ck(cudaStreamSynchronize(stream));s.kv->commit(s.tx);need(s.kv->length(s.sequence)==s.past+s.batch,"M6 publish length");s.layer=0;++s.transactions;}
    if(s.storage->options.environment.enabled){s.gpu->maintain(s.storage->gpu_watts);s.drain();}
}
void m4_attention(ggml_tensor* dst,cudaStream_t stream){
    need(state && state->active,"inactive sequence attention");auto& s=*state;
    const auto n=static_cast<std::size_t>(dst->src[0]->ne[2]);need(n && n<=integrated_profile().ubatch,"sequence batch bound");
    std::vector<int> ids(n,0);
    if(dst->src[4]){need(dst->src[4]->type==GGML_TYPE_I32 && ggml_nelements(dst->src[4])==n,"sequence tensor shape");ck(cudaMemcpyAsync(ids.data(),dst->src[4]->data,n*4,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));}
    std::map<int,std::vector<std::size_t>> groups;
    for(std::size_t i=0;i<n;++i){need(ids[i]>=0 && ids[i]<16,"sequence identity outside budget");groups[ids[i]].push_back(i);}
    s.group_count=groups.size();s.group_tokens=n;
    auto invoke=[&](int id,ggml_tensor* t){
        auto it=s.sequences.find(id);if(it==s.sequences.end()){auto seq=s.kv->create();it=s.sequences.emplace(id,SequenceState{seq}).first;s.peak_sequences=std::max(s.peak_sequences,s.sequences.size());}
        auto& q=it->second;s.sequence_id=id;s.sequence=q.sequence;s.tx=q.tx;s.layer=q.layer;s.batch=q.batch;s.past=q.past;
        attention_one(t,stream);q={s.sequence,s.tx,s.layer,s.batch,s.past};
    };
    if(groups.size()==1){invoke(groups.begin()->first,dst);return;}
    // Stable per-sequence gather/scatter retains each sequence's token order.
    // Scratch is released synchronously and bounded by this ubatch's tensors.
    for(const auto& group:groups){
        ggml_tensor qt=*dst->src[0],kt=*dst->src[1],vt=*dst->src[2],pt=*dst->src[3],out=*dst;
        const auto count=group.second.size();const auto& profile=integrated_profile();s.peak_group_scratch=std::max(s.peak_group_scratch,count*((2*profile.q_elements()+2*profile.kv_elements())*4+4));std::vector<void*> memory;
        auto allocate=[&](std::size_t bytes){void* p=nullptr;ck(cudaMalloc(&p,bytes));memory.push_back(p);return p;};
        try {
            ggml_tensor* inputs[]={&qt,&kt,&vt};
            for(int k=0;k<3;++k){auto* t=inputs[k];const auto row=dst->src[k]->ne[0]*dst->src[k]->ne[1]*4;t->data=allocate(count*row);t->ne[2]=count;t->nb[3]=t->nb[2]*count;
                for(std::size_t j=0;j<count;++j)ck(cudaMemcpyAsync(static_cast<char*>(t->data)+j*row,static_cast<char*>(dst->src[k]->data)+group.second[j]*row,row,cudaMemcpyDeviceToDevice,stream));}
            pt.data=allocate(count*4);pt.ne[0]=count;pt.ne[1]=pt.ne[2]=pt.ne[3]=1;
            for(std::size_t j=0;j<count;++j)ck(cudaMemcpyAsync(static_cast<char*>(pt.data)+j*4,static_cast<char*>(dst->src[3]->data)+group.second[j]*4,4,cudaMemcpyDeviceToDevice,stream));
            const auto row=qt.ne[0]*qt.ne[1]*4;out.data=allocate(count*row);out.ne[1]=count;out.nb[2]=out.nb[3]=count*row;
            out.src[0]=&qt;out.src[1]=&kt;out.src[2]=&vt;out.src[3]=&pt;out.src[4]=nullptr;
            invoke(group.first,&out);
            for(std::size_t j=0;j<count;++j)ck(cudaMemcpyAsync(static_cast<char*>(dst->data)+group.second[j]*row,static_cast<char*>(out.data)+j*row,row,cudaMemcpyDeviceToDevice,stream));
            ck(cudaStreamSynchronize(stream));
        }catch(...){cudaStreamSynchronize(stream);for(auto p:memory)cudaFree(p);throw;}
        for(auto p:memory)ck(cudaFree(p));
    }
}
void m4_sequence_truncate(int id,std::size_t length){need(state && !state->active,"sequence mutation during step");auto& s=*state;auto& q=s.sequences.at(id);s.kv->truncate(q.sequence,length);q.layer=q.batch=0;}
void m4_sequence_release(int id){need(state && !state->active,"sequence mutation during step");auto& s=*state;auto it=s.sequences.find(id);need(it!=s.sequences.end(),"unknown sequence");s.kv->release(it->second.sequence);s.sequences.erase(it);}
void m4_step_end(){if(!state)return;auto& s=*state;need(s.active,"inactive step");bool work=false;for(const auto& item:s.sequences){need(item.second.layer==0,"missing sequence layer");work|=item.second.batch>0;}need(work,"empty step");s.drain();s.active=false;++s.step;}
std::uint64_t m4_sequence_version(int id){need(state && !state->active && state->sequences.count(id),"KV version requires idle sequence");return state->kv->version(state->sequences.at(id).sequence);}
void m4_finish(){if(!state)return;auto& s=*state;need(!s.active,"M4 unfinished batch");for(const auto& item:s.sequences)s.kv->flush(item.second.sequence);s.drain();
    if(s.storage->active){s.gpu->flush();s.drain();}
    const auto c=s.gpu->store().statistics();const auto g=s.gpu->statistics();need(!c.pins&&!c.dirty_frames&&!g.active_leases,"M4 leases/dirty data not drained");
    const auto txstats=s.kv->transaction_statistics();
    std::ofstream f(s.output/"kv-summary.json");f<<json({{"cancellable_append",kv_options().cancellable_append},{"staging_pages",txstats.staging_pages},{"peak_staging_pages",txstats.peak_staging_pages},{"staging_budget_pages",txstats.staging_budget_pages},{"copied_tail_bytes",txstats.copied_tail_bytes},{"cancellations",txstats.cancellations},{"steps",s.step},{"length",s.sequences.count(0)?s.kv->length(s.sequences.at(0).sequence):0},{"peak_sequences",s.peak_sequences},{"peak_group_scratch_bytes",s.peak_group_scratch},{"profile_sha256",integrated_profile().hash},{"layers",integrated_profile().layers},{"transactions",s.transactions},{"cross_page_rows",s.cross_page_rows},{"appended_layer_tokens",s.appended},
      {"page_pairs",s.page_pairs},{"frame_bytes",g.device_bytes},{"logical_pages",integrated_profile().logical_pages},{"pages_per_block",m6_geometry().pages_per_block},{"blocks_per_bank",m6_geometry().blocks_per_bank},{"allocated_pages",c.allocated_pages},{"faults",c.faults},{"hits",c.hits},
      {"evictions",c.evictions},{"writebacks",c.writebacks},{"dirty_frames",c.dirty_frames},{"pins",c.pins},{"media_records",s.media_count},
      {"kv_write_mode",std::getenv("HBFLAB_KV_WRITE_MODE")?std::getenv("HBFLAB_KV_WRITE_MODE"):"host"},{"kv_commit_d2h_bytes",s.copy_bytes},{"h2d_bytes",g.h2d_bytes},{"capacity_d2h_bytes",g.d2h_bytes},{"capacity_d2d_bytes",g.d2d_bytes},{"overwrite_uploads_skipped",g.overwrite_uploads_skipped},
      {"paging_mode",s.paging_mode},{"pinned_staging_bytes",g.pinned_staging_bytes},{"peak_inflight",g.peak_inflight},{"async_prepared",g.async_prepared},{"async_retired",g.async_retired},{"prefetch_issued",g.prefetch_issued},{"prefetch_used",g.prefetch_used},{"prefetch_cancelled",g.prefetch_cancelled},
      {"kv_transfer_version",2},{"read_source_d2d_bytes",g.read_source_d2d_bytes},{"read_source_fills",g.read_source_fills},{"packed_buffer_bytes",2*integrated_profile().ubatch*integrated_profile().kv_elements()*sizeof(half)},
      {"kv_reuse_packed",std::getenv("HBFLAB_KV_REUSE_PACKED")?std::getenv("HBFLAB_KV_REUSE_PACKED"):"1"},{"native_gpu_kv_bytes",0}}).dump(2)<<'\n';f.close();need(bool(f),"M4 summary failed");
    if(s.storage->active){std::ofstream storage(s.output/"storage-summary.json");storage<<json({
        {"file_read_bytes",c.file_read_bytes},{"file_write_bytes",c.file_write_bytes},{"checkpoints",c.checkpoints},
        {"prefetch_fills",c.prefetch_fills},{"prefetch_useful",c.prefetch_useful},{"prefetch_wasted",c.prefetch_wasted},{"prefetch_skipped",c.prefetch_skipped},
        {"prefetch_uploads",g.prefetch_uploads},
        {"refreshes",c.refreshes},{"refresh_read_bytes",c.refresh_read_bytes},{"refresh_pending",c.refresh_pending},{"refresh_blocked",c.refresh_blocked},
        {"retention_expirations",c.retention_expirations},{"temperature_c",c.environment.temperature_c},{"peak_c",c.environment.peak_c},
        {"media_energy_j",c.environment.media_energy_j},{"exposure",c.environment.exposure},{"samples",c.environment.samples},
        {"calibrated",false},{"model_resume_supported",false}}).dump(2)<<'\n';storage.close();need(bool(storage),"storage summary failed");}
    for(const auto& item:s.sequences)s.kv->release(item.second.sequence);need(!s.gpu->store().statistics().allocated_pages,"M4 allocation leak");
    if(s.storage->active)s.gpu->flush(); // Persist the final empty allocation catalog.
    s.gpu->close();
    std::ofstream release(s.output/"kv-release.json");release<<json({{"allocated_pages",s.gpu->store().statistics().allocated_pages},{"pins",s.gpu->store().statistics().pins},{"dirty_frames",s.gpu->store().statistics().dirty_frames},{"active_leases",s.gpu->statistics().active_leases},{"pinned_staging_bytes",s.gpu->statistics().pinned_staging_bytes}}).dump()<<'\n';release.close();need(bool(release),"M6 release evidence failed");
    state.reset();}
void m4_abort(){state.reset();}

void m4_sequence_audit(const char* path,std::size_t prefix){
    need(state && !state->active,"audit requires idle step");auto& s=*state;json sequences=json::object();
    const auto row=s.kv->row_bytes();std::vector<char> bytes(row);
    for(const auto& entry:s.sequences){const auto seq=entry.second.sequence;const auto length=s.kv->length(seq);std::string content;content.reserve(length*integrated_profile().layers*2*row);
        for(std::size_t token=0;token<length;++token)for(unsigned layer=0;layer<integrated_profile().layers;++layer)for(auto kind:{KvKind::Key,KvKind::Value}){s.kv->read(seq,layer,token,kind,bytes.data(),row);s.drain();content.append(bytes.data(),row);}
        const auto prefix_bytes=std::min(prefix,length)*integrated_profile().layers*2*row;
        sequences[std::to_string(entry.first)]={{"identity",seq.value},{"length",length},{"sha256",hbflab::sha256(content)},{"prefix_sha256",hbflab::sha256(std::string_view(content.data(),prefix_bytes))}};
    }
    s.drain();const auto c=s.gpu->store().statistics();const auto g=s.gpu->statistics();
    std::ofstream file(path);file<<json({{"sequences",sequences},{"allocated_pages",c.allocated_pages},{"pins",c.pins},{"active_leases",g.active_leases},{"frame_bytes",g.device_bytes}}).dump(2)<<'\n';file.close();need(bool(file),"sequence audit write failed");
}
