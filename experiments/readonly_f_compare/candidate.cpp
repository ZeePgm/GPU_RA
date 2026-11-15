#define NOMINMAX
#include "llama.h"
#include "capture.hpp"
#include "ggml-backend.h"
#include "hbflab/async_module.hpp"
#include "plan_ptx.hpp"
#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <chrono>
#include <cstring>
using namespace hbflab;using json=nlohmann::ordered_json;using Clock=std::chrono::steady_clock;
const void* r3_kernel(unsigned,bool,bool);void r3_scatter(const void*,void*,unsigned,unsigned,unsigned);
void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}void ck(cudaError_t e){need(e==cudaSuccess,cudaGetErrorString(e));}
std::map<std::string,ggml_tensor*> weights;std::set<int> layers;bool discovery=true,failed=false,merge=true,verify=true; bool bypass=false;std::string route;int step=0;json calls=json::array();
std::unique_ptr<LiveTimingService> service;std::unique_ptr<AsyncModuleInterceptor> loader;AsyncModuleHandle module;AsyncFunctionHandle fn;
unsigned* plan;unsigned char *tile,*mirror;uint64_t requests=0,covered=0,checks=0;std::string error;
bool selected(const ggml_tensor*t){if(!t)return false;for(auto l:layers)for(auto k:{"gate","up","down"})if(std::string(t->name)=="blk."+std::to_string(l)+".ffn_"+k+".weight")return true;return false;}
void gate(ggml_tensor*w){
 auto begin=Clock::now();const unsigned bytes=unsigned(ggml_nbytes(w));auto* input=static_cast<unsigned char*>(w->data);unsigned offsets[256];std::vector<unsigned char> reference(verify?bytes:0),got(verify?8192:0);if(verify)ck(cudaMemcpy(reference.data(),input,bytes,cudaMemcpyDeviceToHost));
 uint64_t local=0;for(unsigned base=0;base<bytes;base+=4096){if(bypass)capture_mark(w->name,base);for(unsigned i=0;i<256;++i)offsets[i]=base+i*16;ck(cudaMemcpy(plan,offsets,sizeof offsets,cudaMemcpyHostToDevice));std::vector<void*> args{&input,&plan,&tile};uint64_t id;
 if(bypass){AsyncDeviceView view{};std::vector<void*> a{&view,&input,&plan,&tile};ck(cudaLaunchKernel(r3_kernel(16,false,true),dim3(1),dim3(256),a.data(),0,nullptr));ck(cudaDeviceSynchronize());id=0;}else if(route=="ptx")id=loader->launch(fn,{1,256},args,{tile,8192},nullptr,{plan,1024});else id=service->launch({1,256},r3_kernel(16,verify,merge),args,nullptr);
 service->synchronize();auto audit=service->take_audit();need(audit.size()==(bypass?0:merge?8:256),"request count");std::set<unsigned> seen;
 if(verify)ck(cudaMemcpy(got.data(),tile,8192,cudaMemcpyDeviceToHost));
 for(const auto&a:audit){unsigned width=merge?512:16;need(a.launch==id && seen.insert(a.producer).second && a.first_address==uintptr_t(input)+base+a.producer*16 && a.first_bytes==width && a.span_count==1 && a.sectors==(merge?8:1),"coverage identity");if(verify)for(unsigned i=a.producer;i<a.producer+(merge?32:1);++i){uint64_t b,e;memcpy(&b,got.data()+i*32+16,8);memcpy(&e,got.data()+i*32+24,8);need(b<=a.arrival_ns && e>=a.arrival_ns+(a.end_ps-a.arrival_ps+999)/1000,"early consumer");need(!memcmp(got.data()+i*32,reference.data()+base+i*16,16),"weight bytes");++checks;}}
 local+=audit.size();r3_scatter(tile,mirror,256,16,base);ck(cudaDeviceSynchronize());}
 // Publish bytes actually obtained by the adapter to the original framework allocation.
 ck(cudaMemcpy(input,mirror,bytes,cudaMemcpyDeviceToDevice));if(verify){std::vector<unsigned char> back(bytes);ck(cudaMemcpy(back.data(),input,bytes,cudaMemcpyDeviceToHost));need(back==reference,"published bytes");}
 requests+=local;covered+=bytes;calls.push_back({{"step",step},{"tensor",w->name},{"bytes",bytes},{"requests",local},{"gate_ms",std::chrono::duration<double,std::milli>(Clock::now()-begin).count()}});
}
bool callback(ggml_tensor*t,bool ask,void*){if(!ask)return true; if(t->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(t,GGML_PREC_F32);if(!selected(t->src[0]))return false;
 try{if(discovery)weights[t->src[0]->name]=t->src[0];else if(route!="native")gate(t->src[0]);}catch(const std::exception&e){error=e.what();failed=true;throw;}return false;}
int main(int argc,char**argv)try{
 need(argc==3,"config output");std::ifstream cf(argv[1]);json c;cf>>c;std::filesystem::path out=argv[2];need(!std::filesystem::exists(out),"output exists");std::filesystem::create_directories(out);route=c.at("route");bypass=c.value("bypass",false);merge=c.value("merge",true);verify=c.value("verify",true);for(int l:c.at("layers"))layers.insert(l);
 ck(cudaSetDevice(0));llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=999;mp.split_mode=LLAMA_SPLIT_MODE_NONE;auto* model=llama_model_load_from_file(c.at("model").get<std::string>().c_str(),mp);need(model,"model");auto* vocab=llama_model_get_vocab(model);
 std::ifstream pf(c.at("prompt").get<std::string>(),std::ios::binary);need(bool(pf),"prompt open");std::string prompt((std::istreambuf_iterator<char>(pf)),{});int n=-llama_tokenize(vocab,prompt.data(),int(prompt.size()),nullptr,0,true,true);std::vector<llama_token> input(n);llama_tokenize(vocab,prompt.data(),int(prompt.size()),input.data(),n,true,true);std::ofstream(out/"input.json")<<json({{"prompt_bytes",prompt.size()},{"tokens",input}}).dump(2);
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=256;cp.n_ubatch=256;cp.n_threads=8;cp.n_threads_batch=8;cp.n_seq_max=1;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.cb_eval=callback;
 auto process=[&](llama_context*ctx,const std::vector<llama_token>&tokens,int position){auto*b=llama_batch_ext_init(ctx);for(auto tok:tokens){int i=llama_batch_ext_add_token(b,0,tok);llama_pos p=position++;need(i>=0 && llama_batch_ext_set_pos(b,i,&p),"batch");}llama_batch_ext_set_output_logits(b,int(tokens.size()-1),true);need(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,b)==0 && !failed,"decode");llama_synchronize(ctx);llama_batch_ext_free(b);};
 auto* ctx=llama_init_from_model(model,cp);need(ctx,"context");process(ctx,input,0);llama_free(ctx);need(weights.size()==layers.size()*3,"discovery coverage");discovery=false;
 auto setup=Clock::now();if(route!="native"){ck(cudaMalloc((void**)&plan,1024));ck(cudaMalloc((void**)&tile,8192));ck(cudaMalloc((void**)&mirror,6*1024*1024));AsyncRuntimeOptions o;o.backend.geometry={1,1,2,16,4096,1};o.backend.max_events_per_batch=2000000000ull;o.audit_limit=257;o.timeout_ns=1000000000ull;o.service_timeout_ns=60000000000ull;std::vector<std::pair<uint64_t,ggml_tensor*>> ordered;for(auto&[name,w]:weights){auto l=std::stoi(name.substr(4));unsigned k=name.find("gate")!=std::string::npos?0:name.find("up")!=std::string::npos?1:2;ordered.push_back({(uint64_t(l)*3+k)*6291456,w});}std::sort(ordered.begin(),ordered.end());for(auto& item:ordered)o.comparison_bases.push_back(item.first);service=std::make_unique<LiveTimingService>(o,256,1);for(auto& item:ordered)service->register_readonly(item.second->data,ggml_nbytes(item.second));service->start();if(route=="ptx"){ModuleOptions mo;mo.uniform_warp_reads=merge;loader=std::make_unique<AsyncModuleInterceptor>(*service,mo);module=loader->load_ptx(plan_ptx(16,verify));fn=loader->function(module,"r3_read");}}
 double setup_ms=std::chrono::duration<double,std::milli>(Clock::now()-setup).count();ctx=llama_init_from_model(model,cp);need(ctx,"measured context");int position=0;json times=json::array(),tokens=json::array();std::ofstream logits(out/"logits.f32",std::ios::binary);const unsigned outputs=c.value("outputs",2);if(bypass)capture_begin(out);auto request_begin=Clock::now();json token_wall_ms=json::array();for(step=0;step<int(outputs);++step){if(bypass && step)cuptiActivityFlushAll(0);capture_step=step;auto begin=Clock::now();process(ctx,input,position);position+=int(input.size());float* l=llama_get_logits_ith(ctx,-1);need(l,"logits");int best=0;for(int i=0;i<llama_vocab_n_tokens(vocab);++i){need(std::isfinite(l[i]),"nonfinite");if(l[i]>l[best])best=i;}times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());token_wall_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-request_begin).count());logits.write((char*)l,llama_vocab_n_tokens(vocab)*4);tokens.push_back(best);input={best};}
 if(bypass)capture_end();bool drained=true;if(service){if(loader)loader->unload(module);service->finish();auto st=service->statistics();drained=service->drained();need(drained && st.submitted==requests && st.completed==requests && !st.audit_dropped,"drain");}
 std::ofstream(out/"result.json")<<json({{"status","pass"},{"config",c},{"prompt_tokens",n},{"step_ms",times},{"token_wall_ms",token_wall_ms},{"tokens",tokens},{"setup_ms",setup_ms},{"calls",calls},{"requests",requests},{"covered_bytes",covered},{"dependency_checks",checks},{"drained",drained},{"kv","native F16 llama; no capacity/KV adapter hooks"}}).dump(2);
 loader.reset();service.reset();llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<" "<<error<<'\n';return 1;}
