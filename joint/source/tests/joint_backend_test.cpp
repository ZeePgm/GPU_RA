#include "hbflab/backend.hpp"
#include <iostream>
#include <stdexcept>
#include <tuple>
using namespace hbflab;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class F>void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected,"expected invalid reservation/access rejection");}
std::tuple<uint64_t,uint64_t,uint64_t> trace(bool direct,unsigned resident){
    BackendOptions o;o.geometry={1,1,2,4,256,1};o.direct_device_transactions=direct;o.coalesce_capacity_feedback=direct;o.resident_fragments=resident;
    TimingBackend b(o);b.reserve(8*4096);b.reserve_gap(8*4096);auto base=b.reserve(8*4096);
    check(base==16*4096,"sparse base");auto init=b.statistics().device.programs;check(init==16,"gap initialized unexpectedly");
    check(b.read(8*4096,64).status==Status::InvalidRange,"gap read accepted");
    std::array<uint64_t,64> t{};for(unsigned i=0;i<64;++i)t[i]=100000+i;
    rejects([&]{b.write_page(8*4096,t);});
    b.begin_stream(4);auto at=b.now_ps()+100;
    b.submit_read(1,{at,{{0,512}}});b.submit_write(2,at,base,t);b.submit_read(3,{at,{{base,512}}});
    while(b.advance_before(UINT64_MAX)){}
    auto r=b.take_completed();auto w=b.take_completed_writes();check(r.size()==2&&w.size()==1,"completion count");
    for(auto x:r)check(x.result.status==Status::Ok,"read result");check(w[0].result.status==Status::Ok,"write result");
    b.end_stream();b.finish();check(b.drained(),"drain");auto st=b.statistics();check(st.peak_resident_fragments<=resident,"resident budget");
    return {b.now_ps(),st.device.programs,st.device.senses};
}
int main()try{
    for(unsigned budget:{64,128,256})check(trace(false,budget)==trace(true,budget),"overlay changed deterministic semantics");
    BackendOptions o;o.geometry={1,1,2,4,16,1};TimingBackend b(o);
    rejects([&]{b.reserve_gap(4096);});rejects([&]{b.reserve_gap(o.geometry.channel_bytes()+32768);});
    b.reserve_gap(32768);check(b.statistics().device.programs==0,"gap program");
    check(b.reserve(32768)==32768,"valid allocation after rejected gap");b.finish();check(b.drained(),"final drain");
    std::cout<<"PASS sparse gaps, invalid reads/writes, capacity bounds, 3 deterministic legacy/overlay pairs\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
