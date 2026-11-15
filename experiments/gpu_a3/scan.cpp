#include "hbf_eventlab/adapters/tensor_access_bridge.hpp"
#include "hbf_eventlab/statistics/collector.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>
using namespace hbf_eventlab;using json=nlohmann::ordered_json;namespace fs=std::filesystem;
static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
static json rec(const TensorAccessResult& r){return {{"id",r.id},{"offered_ps",r.offered.value},{"completed_ps",r.completed?json(r.completed->value):json(nullptr)},{"success",r.success},{"complete",r.complete},{"registered_globals",r.registered_globals},{"finished_globals",r.finished_globals},{"capacity_rejections",r.capacity_rejections},{"unsubmitted_bytes",r.unsubmitted_bytes}};}
int main(int argc,char** argv){try{
 need(argc==4,"usage: a3_scan H0.ini plan.json output");fs::path out=argv[3];need(!fs::exists(out),"output exists");fs::create_directories(out);
 std::ifstream pf(argv[2]);json plan;pf>>plan;
 for(const auto& c:plan.at("cases"))for(int rep=0;rep<3;++rep){
  const auto d=out/(c.at("name").get<std::string>()+"-"+std::to_string(rep));fs::create_directories(d);
  std::ifstream ini(argv[1]);need(bool(ini),"missing H0");auto resolved=resolve_configuration(read_configuration(ini,"A3 frozen S8 H0"));
  need(resolved.values().device.geometry.host_channels==1,"single channel required");std::ofstream eff(d/"effective.ini");write_effective_configuration(eff,resolved);eff.close();
  StatisticsCollector stats;const auto& r=plan.at("region");TensorAccessBridge bridge(resolved,{{"R",GlobalByteAddress{r.at("global_base").get<std::uint64_t>()},r.at("registered_bytes"),1}}, {},&stats);
  std::ofstream trace(d/"bridge.jsonl");bridge.set_trace([&](const TensorBridgeTrace& e){trace<<json({{"action",e.action},{"access_id",e.access},{"global_id",e.global.value},{"tick_ps",e.at.value}}).dump()<<'\n';});
  auto init=bridge.initialize("R");need(init.success&&init.complete,"initialization failed");const auto t0=bridge.session().now().value;
  auto lo=std::max(c.at("source_offset").get<std::uint64_t>(),r.at("source_offset").get<std::uint64_t>());
  auto hi=std::min(c.at("source_offset").get<std::uint64_t>()+c.at("source_bytes").get<std::uint64_t>(),r.at("source_offset").get<std::uint64_t>()+r.at("valid_bytes").get<std::uint64_t>());need(hi>lo,"empty intersection");
  const auto offset=lo-r.at("source_offset").get<std::uint64_t>();auto read=bridge.access("R",offset,hi-lo);need(read.success&&read.complete,"read failed");auto final=bridge.finish();
  need(final.run.status==RunStatus::Completed&&final.run.workload_success&&final.run.host.drained()&&final.run.device.drained(),"not successful/drained");
  json s={{"case",c},{"repeat",rep},{"initialization",rec(init)},{"request_t0_ps",t0},{"read",rec(read)},{"mapped_offset",offset},{"mapped_bytes",hi-lo},{"status",to_string(final.run.status)},{"workload_success",final.run.workload_success},{"input_closed",final.input_closed},{"host_drained",final.run.host.drained()},{"device_drained",final.run.device.drained()}};
  std::ofstream(d/"session.json")<<s.dump(2);std::ofstream f(d/"statistics.json");stats.write_json(f,bridge.session().now(),to_string(final.run.status));need(bool(f)&&bool(trace),"write failure");
 }
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
