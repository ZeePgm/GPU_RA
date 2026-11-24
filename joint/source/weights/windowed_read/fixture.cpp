#define NOMINMAX
#include "hbflab/live_runtime.hpp"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <set>
using namespace hbflab;using json=nlohmann::json;
const void* b2_kernel(bool,bool);
void need(bool x,const char*s){if(!x)throw std::runtime_error(s);}
void ck(cudaError_t x){need(x==cudaSuccess,cudaGetErrorString(x));}
int main(int argc,char**argv)try{
 need(argc==3,"output mode");bool negative=std::string(argv[2])=="negative";
 constexpr unsigned size=1024*1024;std::vector<unsigned char> expected(size),back(size);
 for(unsigned i=0;i<size;++i)expected[i]=(i*37+i/251)%256;
 unsigned char *input,*mirror;unsigned long long *times,*phases=nullptr;unsigned slow_ns=0;
 ck(cudaMalloc((void**)&input,size));ck(cudaMalloc((void**)&mirror,size));ck(cudaMalloc((void**)&times,262144));
 ck(cudaMemcpy(input,expected.data(),size,cudaMemcpyHostToDevice));
 AsyncRuntimeOptions o;o.backend.geometry={1,1,2,4,4096,1};o.backend.direct_device_transactions=true;o.backend.coalesce_capacity_feedback=true;o.audit_limit=513;o.timeout_ns=1000000000;o.service_timeout_ns=60000000000ull;
 LiveTimingService service(o,2048,1);service.register_readonly(input,size);service.start();
 if(negative){unsigned base=size,count=512;std::vector<void*> a{&input,&base,&count,&mirror,&times,&phases,&slow_ns};service.launch({1,256},b2_kernel(true,false),a,nullptr);service.synchronize();throw std::runtime_error("negative unexpectedly succeeded");}
 json cases=json::array();uint64_t refill_overlap=0;std::vector<uint64_t> tickets(2048,0);uint64_t total=0,checks=0;
 for(unsigned w:{4096u,8192u,16384u,32768u})for(unsigned bytes:{512u,3584u,4096u,4608u,262144u-512,262144u,262144u+512}){unsigned k=262144;slow_ns=bytes==262144?100000:0;
  ck(cudaMemset(mirror,0xa5,size));uint64_t count_requests=0;
  for(unsigned start=0;start<bytes;start+=k){unsigned base=512+start,count=std::min(k,bytes-start);
   std::vector<void*> a{&input,&base,&count,&mirror,&times,&phases,&slow_ns};auto id=service.launch({w/4096,256},b2_kernel(true,false),a,nullptr);service.synchronize();
   auto audit=service.take_audit();need(audit.size()==count/512,"fixture requests");std::set<unsigned> addresses;
   std::vector<unsigned long long> timing(count/8);ck(cudaMemcpy(timing.data(),times,count,cudaMemcpyDeviceToHost));
   auto old=tickets;
   for(auto&r:audit){auto off=unsigned(r.first_address-uintptr_t(input)-base);unsigned batch=off/w;
    need(r.launch==id&&off<count&&off%512==0&&addresses.insert(off).second&&r.producer==off%w/16&&r.first_bytes==512&&r.ticket==old[r.producer]+batch+1,"fixture ownership");
    tickets[r.producer]=std::max(tickets[r.producer],r.ticket);
    for(unsigned i=off/16;i<off/16+32;++i){need(timing[2*i]<=r.arrival_ns&&timing[2*i+1]>=r.arrival_ns+(r.end_ps-r.arrival_ps+999)/1000,"fixture dependency");++checks;}
   }
   for(const auto& r:audit)for(const auto& other:audit)if(r.producer!=other.producer && r.ticket>old[r.producer]+1 && r.arrival_ns<other.arrival_ns+(other.end_ps-other.arrival_ps+999)/1000 && r.arrival_ns>=other.arrival_ns){++refill_overlap;break;}
   count_requests+=audit.size();
  }
  ck(cudaMemcpy(back.data(),mirror,size,cudaMemcpyDeviceToHost));
  for(unsigned i=0;i<size;++i)need(back[i]==(i>=512&&i<512+bytes?expected[i]:0xa5),"tail/guard bytes");
  total+=count_requests;cases.push_back({{"K",k},{"W",w},{"slow_ns",slow_ns},{"bytes",bytes},{"requests",count_requests}});
 }
 need(refill_overlap>0,"no independent refill observed");service.finish();auto stats=service.statistics();need(service.drained()&&stats.completed==total&&!stats.audit_dropped&&stats.backend.peak_resident_fragments<=256,"fixture drain/budget");
 std::ofstream(argv[1])<<json({{"status","pass"},{"refill_overlap",refill_overlap},{"cases",cases},{"checks",checks},{"requests",total},{"peak_resident_fragments",stats.backend.peak_resident_fragments},{"drained",true}}).dump(2);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 23;}
