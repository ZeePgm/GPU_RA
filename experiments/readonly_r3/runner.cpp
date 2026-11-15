#define NOMINMAX
#include "hbflab/async_module.hpp"
#include "plan_ptx.hpp"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <tuple>
#include <vector>
using namespace hbflab;
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
const void* r3_kernel(unsigned,bool,bool);
void r3_scatter(const void*,void*,unsigned,unsigned,unsigned);
void r3_consume(const void*,const float*,float*,unsigned,unsigned,unsigned,bool);
void need(bool v,const char* s){if(!v)throw std::runtime_error(s);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void dr(CUresult e){if(e!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorName(e,&s);throw std::runtime_error(s?s:"CUDA driver error");}}
std::string read(const std::string& p){std::ifstream f(p,std::ios::binary);need(bool(f),"file missing");return {std::istreambuf_iterator<char>(f),{}};}
double ns(Clock::time_point t){return std::chrono::duration<double,std::nano>(Clock::now()-t).count();}
struct Expected {unsigned leader,offset,bytes;std::vector<unsigned> lanes;};
int main(int argc,char** argv)try {
 need(argc==3,"config.json result.json");const auto cfg=json::parse(read(argv[1]));
 const std::string requested=cfg.at("strategy"),mode=cfg.value("mode","modeled");std::string strategy=requested;
 const bool merge=cfg.at("merge"),timed=cfg.value("timed",false),modeled=mode!="bypass";
 const unsigned width=cfg.value("width",4),repeats=cfg.value("repeats",3),scale=timed?100:1;
 need((strategy=="ptx"||strategy=="cuda"||strategy=="hybrid") && (mode=="modeled"||mode=="bypass"||mode=="fault"||mode=="range") && (width==4||width==16) && repeats && repeats<=100,"invalid options");
 const bool whole=cfg.contains("input_file");
 std::vector<unsigned char> data;
 if(whole){const auto s=read(cfg.at("input_file"));data.assign(s.begin(),s.end());need(data.size()==cfg.at("bytes").get<size_t>(),"manifest size mismatch");}
 else {data.resize(65536);for(unsigned i=0;i<data.size();++i)data[i]=(i*13+i/31+19)%251;}
 need(data.size()%width==0 && data.size()<=64*1024*1024,"input width/budget");
 const unsigned split=cfg.value("split",0);
 need(!split || (split<data.size() && split%64==0),"invalid registered split");
 const unsigned batch=whole?256:cfg.value("lanes",256);
 need(batch && batch<=2048 && batch%32==0,"lane budget");
 const unsigned padded=(batch+255)/256*256;
 ck(cudaSetDevice(0));ck(cudaFree(nullptr));
 unsigned char *input=nullptr,*output=nullptr,*mirror=nullptr;unsigned* plan=nullptr;
 ck(cudaMalloc(reinterpret_cast<void**>(&input),data.size()));ck(cudaMemcpy(input,data.data(),data.size(),cudaMemcpyHostToDevice));
 ck(cudaMalloc(reinterpret_cast<void**>(&output),padded*32));ck(cudaMalloc(reinterpret_cast<void**>(&plan),padded*4));
 if(whole)ck(cudaMalloc(reinterpret_cast<void**>(&mirror),data.size()));
 auto source=plan_ptx(width,timed);
 if(cfg.value("control_flow_fixture",false))source.insert(source.find("ld.param.u64"),"bra R3_join; R3_join:\n");
 if(strategy=="hybrid") {
  try{rewrite_async_ptx(source,merge);strategy="ptx";}
  catch(const std::invalid_argument&){strategy="cuda";}
 }
 const uint64_t event_budget=std::max<uint64_t>(10000000,static_cast<uint64_t>(data.size()/width)*128+static_cast<uint64_t>((data.size()+4095)/4096)*128);
 auto start=Clock::now();std::unique_ptr<LiveTimingService> service;
 if(modeled){AsyncRuntimeOptions o;o.time_scale=scale;o.audit_limit=padded+1;o.timeout_ns=1000000000ull;o.service_timeout_ns=60000000000ull;o.backend.max_events_per_batch=mode=="fault"?100:event_budget;o.backend.geometry={1,1,2,16,512,1};
 service=std::make_unique<LiveTimingService>(o,padded,1);
 if(split){service->register_readonly(input,split);service->register_readonly(input+split,data.size()-split);}else service->register_readonly(input,data.size()-(mode=="range"?width:0));service->start();}
 json result={{"config",cfg},{"input_sha256",sha256(std::string_view(reinterpret_cast<char*>(data.data()),data.size()))},{"service_setup_ns",ns(start)}};
 result["selected_strategy"]=strategy;result["event_budget"]=event_budget;result["tile_lanes"]=batch;
 std::unique_ptr<AsyncModuleInterceptor> loader;AsyncModuleHandle module{};AsyncFunctionHandle fn{};CUmodule raw=nullptr;CUfunction rawfn=nullptr;
 start=Clock::now();
 if(strategy=="ptx"){
  if(modeled){ModuleOptions options;options.uniform_warp_reads=merge;loader=std::make_unique<AsyncModuleInterceptor>(*service,options);module=loader->load_ptx(source);fn=loader->function(module,"r3_read");auto id=loader->identity(module);result["rewritten_sha256"]=id.rewritten_sha256;result["helper_sha256"]=id.helper_sha256;}
  else{dr(cuModuleLoadData(&raw,source.c_str()));dr(cuModuleGetFunction(&rawfn,raw,"r3_read"));}
 }
 result["module_setup_ns"]=ns(start);json samples=json::array();
 uint64_t total_requests=0,total_sectors=0,logical_accesses=0,dependency_checks=0;
 const unsigned elements=whole?static_cast<unsigned>(data.size()/width):batch;
 const unsigned iterations=whole?1:repeats+1;
 std::vector<unsigned> offsets(padded);std::vector<unsigned char> got(padded*32);
 std::vector<unsigned char> covered(whole?data.size():0);
 for(unsigned iteration=0;iteration<iterations;++iteration){double wall=0,submit=0;uint64_t iteration_requests=0,iteration_sectors=0;
 for(unsigned base=0;base<elements;base+=batch){
  std::fill(offsets.begin(),offsets.end(),0xffffffffu);const unsigned count=std::min(batch,elements-base);
  const unsigned mask=cfg.value("mask",0xffffffffu),shift=cfg.value("shift",0),stride=cfg.value("stride",1);
  const std::string layout=cfg.value("layout","contiguous");
  for(unsigned i=0;i<count;++i){const unsigned lane=i%32;if(!whole && !(mask&(1u<<lane)))continue;
   unsigned index=whole?base+i:(i/32)*32+__popcnt(mask&((1u<<lane)-1));
   if(layout=="duplicate")index=(i/32)*32;
   if(layout=="reverse")index=(i/32)*32+31-lane;
   if(layout=="hole" && lane==5)++index;
   offsets[i]=whole?index*width:shift+index*stride*width;
   if(mode=="range" && i==count-1)offsets[i]=static_cast<unsigned>(data.size()-width);
   need(offsets[i]+width<=data.size(),"plan outside allocated data");
  }
  // Independent host oracle from lane offsets and registered interval boundaries.
  std::vector<Expected> expected;
  for(unsigned first=0;first<padded;first+=32){std::vector<unsigned> lanes;for(unsigned i=first;i<first+32;++i)if(offsets[i]!=0xffffffffu)lanes.push_back(i);
   if(lanes.empty())continue;bool contiguous=merge;
   for(unsigned rank=0;rank<lanes.size();++rank)contiguous&=offsets[lanes[rank]]==offsets[lanes[0]]+rank*width;
   const unsigned low=offsets[lanes[0]],bytes=static_cast<unsigned>(lanes.size())*width;
   if(split && low<split && low+bytes>split)contiguous=false;
   if(contiguous)expected.push_back({lanes[0],low,bytes,lanes});
   else for(auto lane:lanes)expected.push_back({lane,offsets[lane],width,{lane}});
  }
  ck(cudaMemcpy(plan,offsets.data(),padded*4,cudaMemcpyHostToDevice));ck(cudaMemset(output,0xcd,padded*32));
  std::vector<void*> args{&input,&plan,&output};uint64_t launch=0;start=Clock::now();
  if(strategy=="ptx"){
   if(modeled)launch=loader->launch(fn,{padded/256,256},args,{output,padded*32},nullptr,{plan,padded*4});
   else dr(cuLaunchKernel(rawfn,padded/256,1,1,256,1,1,0,nullptr,args.data(),nullptr));
  }else{const auto k=r3_kernel(width,timed,merge);if(modeled)launch=service->launch({padded/256,256},k,args,nullptr);else{AsyncDeviceView v{};std::vector<void*> all{&v,&input,&plan,&output};ck(cudaLaunchKernel(k,dim3(padded/256),dim3(256),all.data(),0,nullptr));}}
  submit+=ns(start);
  if(mode=="fault"||mode=="range"){
   bool failed=false;try{service->synchronize();}catch(const std::exception& e){result["error"]=e.what();failed=true;}
   need(failed,"fault accepted");loader.reset();service.reset();ck(cudaDeviceSynchronize());result["status"]="expected_failure";result["cuda_drained"]=true;std::ofstream(argv[2])<<result.dump(2);return 0;
  }
  if(service)service->synchronize();else ck(cudaDeviceSynchronize());wall+=ns(start);
  ck(cudaMemcpy(got.data(),output,padded*32,cudaMemcpyDeviceToHost));
  for(unsigned i=0;i<padded;++i){if(offsets[i]==0xffffffffu){for(unsigned b=0;b<32;++b)need(got[i*32+b]==0xcd,"false predicate wrote output");continue;}
   need(!std::memcmp(got.data()+i*32,data.data()+offsets[i],width),"consumed bytes mismatch");++logical_accesses;
   if(whole)for(unsigned b=0;b<width;++b){need(!covered[offsets[i]+b],"duplicate tensor coverage");covered[offsets[i]+b]=1;}
  }
  if(service){const auto audit=service->take_audit();need(audit.size()==expected.size(),"parent request count mismatch");std::map<unsigned,const AsyncServiceTrace*> by_producer;
   for(const auto& a:audit){need(a.launch==launch && by_producer.emplace(a.producer,&a).second,"request identity mismatch");}
   for(const auto& e:expected){need(by_producer.count(e.leader),"leader missing");const auto& a=*by_producer.at(e.leader);
    const unsigned local=e.offset-(split && e.offset>=split?split:0);const unsigned sectors=(local+e.bytes-1)/64-local/64+1;
    need(a.first_address==reinterpret_cast<uintptr_t>(input)+e.offset && a.first_bytes==e.bytes && a.span_count==1 && a.sectors==sectors,"parent span/sector mismatch");
    iteration_sectors+=sectors;
    if(!whole && iteration==0)result["parents"].push_back({{"leader",e.leader},{"offset",e.offset},{"bytes",e.bytes},{"sectors",a.sectors},{"arrival_ns",a.arrival_ns},{"arrival_ps",a.arrival_ps},{"end_ps",a.end_ps}});
    if(timed)for(auto lane:e.lanes){uint64_t begin,end;std::memcpy(&begin,got.data()+lane*32+16,8);std::memcpy(&end,got.data()+lane*32+24,8);
     const auto delay=((a.end_ps-a.arrival_ps)*scale+999)/1000;
     need(begin<=a.arrival_ns && end>=a.arrival_ns && end-a.arrival_ns>=delay,"consumer before parent completion");++dependency_checks;}
   }
   iteration_requests+=audit.size();
  }
  if(whole){if((base/batch)%64==0)std::cout<<"coverage "<<base*width<<"/"<<data.size()<<std::endl;r3_scatter(output,mirror,count,width,base*width);ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
 }
 if(iteration==0)result["first_iteration_wall_ns"]=wall;
 if(whole||iteration)samples.push_back({{"wall_ns",wall},{"submit_ns",submit},{"requests",iteration_requests},{"sectors",iteration_sectors}});
 total_requests+=iteration_requests;total_sectors+=iteration_sectors;
 if(whole){std::vector<unsigned char> back(data.size());ck(cudaMemcpy(back.data(),mirror,data.size(),cudaMemcpyDeviceToHost));need(back==data && std::all_of(covered.begin(),covered.end(),[](unsigned char v){return v==1;}),"full tensor coverage mismatch");
  result["covered_bytes"]=data.size();result["mirror_sha256"]=sha256(std::string_view(reinterpret_cast<char*>(back.data()),back.size()));
  const unsigned rows=cfg.at("rows"),columns=cfg.at("columns"),tokens=cfg.value("tokens",1);const bool f32=cfg.value("dtype",std::string("F16"))=="F32";
  need(data.size()==size_t(rows)*columns*(f32?4:2) && columns%2==0 && tokens && tokens<=32 && (!f32||rows==1),"consumer shape invalid");
  std::vector<float> x(columns*tokens),a((f32?columns:rows)*tokens),b(a.size());for(unsigned i=0;i<x.size();++i)x[i]=float(int(i%29)-14)/17.0f;
  float *dx,*da,*db;ck(cudaMalloc(reinterpret_cast<void**>(&dx),x.size()*4));ck(cudaMalloc(reinterpret_cast<void**>(&da),a.size()*4));ck(cudaMalloc(reinterpret_cast<void**>(&db),a.size()*4));ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
  r3_consume(input,dx,da,rows,columns,tokens,f32);r3_consume(mirror,dx,db,rows,columns,tokens,f32);ck(cudaDeviceSynchronize());ck(cudaMemcpy(a.data(),da,a.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),db,b.size()*4,cudaMemcpyDeviceToHost));need(!std::memcmp(a.data(),b.data(),a.size()*4),"matrix consumer mismatch");
  // CPU qualification uses the frozen scalar FMA and xor reduction order.
  // Check every F32 output and first/middle/last F16 rows for each token;
  // full native-vs-adapted GPU arrays above are compared byte-for-byte.
  uint64_t cpu_checks=0;
  for(unsigned token=0;token<tokens;++token) {
   if(f32){for(unsigned i=0;i<columns;++i){float w;std::memcpy(&w,data.data()+i*4,4);const float ref=w*x[token*columns+i];need(!std::memcmp(&ref,&b[token*columns+i],4),"CPU norm oracle mismatch");++cpu_checks;}}
   else for(unsigned row:std::set<unsigned>{0,rows/2,rows-1}) {
    float lane[256]{};
    for(unsigned t=0;t<256;++t)for(unsigned v=t;v<columns/2;v+=256){__half h0,h1;std::memcpy(&h0,data.data()+(size_t(row)*columns+2*v)*2,2);std::memcpy(&h1,data.data()+(size_t(row)*columns+2*v+1)*2,2);
     lane[t]=std::fma(__half2float(h0),x[token*columns+2*v],lane[t]);lane[t]=std::fma(__half2float(h1),x[token*columns+2*v+1],lane[t]);}
    for(unsigned step=16;step;step/=2){float old[256];std::memcpy(old,lane,sizeof lane);for(unsigned t=0;t<256;++t)lane[t]=old[t]+old[t^step];}
    float sums[32]{};for(unsigned t=0;t<8;++t)sums[t]=lane[t*32];
    for(unsigned step=16;step;step/=2){float old[32];std::memcpy(old,sums,sizeof sums);for(unsigned t=0;t<32;++t)sums[t]=old[t]+old[t^step];}
    need(!std::memcmp(&sums[0],&b[token*rows+row],4),"CPU matrix oracle mismatch");++cpu_checks;
   }
  }
  result["cpu_consumer_checks"]=cpu_checks;
  result["consumer_output_sha256"]=sha256(std::string_view(reinterpret_cast<char*>(b.data()),b.size()*4));result["consumer_values"]=b.size();ck(cudaFree(dx));ck(cudaFree(da));ck(cudaFree(db));
 }
 }
 if(loader)loader->unload(module);if(raw)dr(cuModuleUnload(raw));
 if(service){service->finish();const auto st=service->statistics();need(service->drained() && st.submitted==total_requests && st.completed==total_requests && !st.audit_dropped,"final drain/accounting");result["registered_globals"]=st.backend.registered_globals;result["completed_globals"]=st.backend.completed_globals;result["input_closed"]=st.backend.input_closed;}
 result["status"]="pass";result["drained"]=true;result["samples"]=samples;result["requests"]=total_requests;result["sectors"]=total_sectors;result["logical_accesses"]=logical_accesses;result["dependency_checks"]=dependency_checks;
 ck(cudaFree(input));ck(cudaFree(output));ck(cudaFree(plan));if(mirror)ck(cudaFree(mirror));
 std::ofstream(argv[2])<<result.dump(2);std::cout<<strategy<<" merge="<<merge<<" requests="<<total_requests<<" pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
