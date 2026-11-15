#include "hbflab/live_runtime.hpp"
#include <iostream>
using namespace hbflab;
void check(bool b,const char*s){if(!b)throw std::runtime_error(s);}
int main()try{
    check(cudaFree(nullptr)==cudaSuccess,"CUDA context initialization");
    AsyncRuntimeOptions o;o.backend.geometry={1,1,2,4,64,1};o.readonly_layout_bytes=32768;o.audit_limit=32;
    o.timeout_ns=1000000000ull;o.service_timeout_ns=60000000000ull;
    LiveTimingService service(o,2048,1);auto base=service.reserve_media(32768);
    check(base==32768 && service.statistics().backend.device.programs==8,"media-only address/initialization");
    service.start();std::array<uint64_t,64> tokens{};for(unsigned i=0;i<64;++i)tokens[i]=101+i;
    check(service.read_media(base,4096).status==Status::Ok,"initial read");
    check(service.write_media(base,tokens).status==Status::Ok,"write");
    check(service.read_media(base,4096).status==Status::Ok,"updated read");
    auto audit=service.take_media_audit();check(audit.size()==3,"media audit count");
    for(auto a:audit)check(a.logical==base && a.end_ps>=a.arrival_ps,"media trace");
    service.finish();auto s=service.statistics();check(service.drained()&&s.submitted==0&&s.media_submitted==3&&s.media_completed==3&&!s.audit_dropped,"media-only drain");
    std::cout<<"PASS media-only shared runtime, sparse base, read/write/read, final drain\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
