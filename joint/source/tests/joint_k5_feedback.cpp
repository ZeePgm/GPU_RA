#include "hbflab/backend.hpp"
#include <iostream>
#include <tuple>
#include <map>
using namespace hbflab;
using Row=std::tuple<unsigned,std::uint64_t,std::uint64_t,std::uint64_t,std::uint64_t>;
struct Result {std::map<std::uint64_t,Row> rows;BackendStats stats;std::uint64_t now;};
void check(bool v,const char* s){if(!v)throw std::runtime_error(s);}
Result run(bool coalesce,unsigned resident,bool pressure,bool finite){
    std::cerr<<"case "<<coalesce<<' '<<resident<<' '<<pressure<<' '<<finite<<'\n';
    BackendOptions options;options.coalesce_capacity_feedback=coalesce;options.direct_device_transactions=coalesce;options.resident_fragments=resident;
    if(pressure){options.outstanding=1;options.timing.buffers.ingress={1,4096};options.timing.buffers.egress={1,64};}
    TimingBackend b(options);b.reserve(16*4096);b.begin_stream(32);auto at=b.now_ps()+10;
    std::array<std::uint64_t,64> a{},c{};for(unsigned i=0;i<64;++i){a[i]=10000+i;c[i]=20000+i;}
    b.submit_read(9,{at,{{0,4096}}});b.submit_write_batch(2,at,{{0,a},{8192,c}});
    b.submit_read(1,{at,{{0,4096}}});b.submit_write_batch(4,at,{{0,c},{8192,a}});
    for(unsigned i=0;i<12;++i)b.submit_read(10+i,{at+1,{{std::uint64_t(i)*4096,4096}}});
    Result result;
    auto collect=[&]{for(const auto& r:b.take_completed()){
        check(r.result.status==Status::Ok,"read content");check(result.rows.emplace(r.id,Row{0,r.result.begin_ps,r.result.end_ps,r.result.sectors,0}).second,"duplicate read");}
        for(const auto& r:b.take_completed_writes()){
        check(r.result.status==Status::Ok,"write content");check(result.rows.emplace(r.id,Row{1,r.result.begin_ps,r.result.end_ps,0,r.result.replayed_pages}).second,"duplicate write");}};
    if(finite){
        check(!b.advance_before(at),"exclusive watermark");
        while(b.advance_before(at+1))collect();collect();
        check(result.rows.empty(),"early completion");
    }
    do{collect();}while(b.advance_before(UINT64_MAX));collect();
    check(result.rows.size()==16,"lost completion");b.end_stream();b.finish();check(b.drained(),"final drain");
    result.stats=b.statistics();result.now=b.now_ps();return result;
}
Result sequential(bool coalesce,unsigned resident){
    BackendOptions options;options.coalesce_capacity_feedback=coalesce;options.direct_device_transactions=coalesce;options.resident_fragments=resident;
    TimingBackend b(options);b.reserve(4*4096);b.begin_stream(4);Result result;
    // Real demand paging repeatedly drains a parent before the next request.
    // A final capacity handoff must remain resumable at the entered tick.
    for(unsigned i=0;i<8;++i){
        const auto at=b.now_ps()+1+(i%2);b.submit_read(i+1,{at,{{(i%4)*4096ull,4096}}});
        for(;;){
            const auto done=b.take_completed();
            if(!done.empty()){
                check(done.size()==1 && done[0].id==i+1 && done[0].result.status==Status::Ok,"sequential completion");
                const auto& r=done[0].result;result.rows.emplace(i,Row{0,r.begin_ps,r.end_ps,r.sectors,0});break;
            }
            check(b.advance_before(UINT64_MAX),"sequential progress");
        }
    }
    while(b.advance_before(UINT64_MAX)){};
    b.end_stream();b.finish();check(b.drained(),"sequential drain");
    result.stats=b.statistics();result.now=b.now_ps();return result;
}
int main()try{
    unsigned count=0;for(unsigned resident:{64,128,256})for(bool pressure:{false,true})for(bool finite:{false,true}){
        const auto a=run(false,resident,pressure,finite),b=run(true,resident,pressure,finite);
        check(a.rows==b.rows && a.now==b.now,"integer event result mismatch");
        const auto& x=a.stats;const auto& y=b.stats;
        check(x.requests==y.requests && x.sectors==y.sectors && x.registered_globals==y.registered_globals &&
            x.completed_globals==y.completed_globals && x.registered_fragments==y.registered_fragments &&
            x.writebacks==y.writebacks && x.replayed_pages==y.replayed_pages &&
            x.device.accepted==y.device.accepted && x.device.delivered==y.device.delivered &&
            x.device.senses==y.device.senses && x.device.cache_hits==y.device.cache_hits &&
            x.device.programs==y.device.programs && x.device.erases==y.device.erases,"work accounting changed");
        check(!x.capacity_feedback_elided && y.capacity_feedback_elided &&
            x.capacity_feedback_calls==y.capacity_feedback_calls,"feedback activation/accounting");
        check(y.peak_resident_fragments<=resident,"Host bound");++count;
    }
    for(unsigned resident:{64,128,256}){
        const auto a=sequential(false,resident),b=sequential(true,resident);
        check(a.rows==b.rows && a.now==b.now,"sequential timing mismatch");
        check(a.stats.requests==b.stats.requests && a.stats.sectors==b.stats.sectors &&
              a.stats.device.senses==b.stats.device.senses && a.stats.device.cache_hits==b.stats.device.cache_hits,
              "sequential work mismatch");
        check(b.stats.capacity_feedback_elided>0,"sequential optimization activation");++count;
    }
    std::cout<<count<<" paired scenarios: exact integer completion times, symbolic content, requests/sectors/media work and drain passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
