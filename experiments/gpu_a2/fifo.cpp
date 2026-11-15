// Mechanism adapter: real production EventQueue, experimental FIFO server.
// No HostDriver/DeviceDriver/H0 policy is instantiated or validated here.
#include "hbf_eventlab/infrastructure/event_queue.hpp"
#include <nlohmann/json.hpp>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
using namespace hbf_eventlab;
using json=nlohmann::ordered_json;
static void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
static std::uint64_t integer(const json& x){require(x.is_number_unsigned(),"expected unsigned integer");return x.get<std::uint64_t>();}
static json simulate(const json& input){
 EventQueue q;q.set_limits({1000000,1000000,1000000});
 struct Job {std::uint64_t id,arrival,service;};std::vector<Job> jobs;std::set<std::uint64_t> ids;
 for(const auto& r:input.at("requests")){Job j{integer(r.at("id")),integer(r.at("arrival_ps")),integer(r.at("service_ps"))};require(ids.insert(j.id).second,"duplicate request id");jobs.push_back(j);}
 require(jobs.size()<=100000,"too many requests");
 std::deque<std::size_t> ready;bool busy=false;json events=json::array(),completed=json::array();
 auto emit=[&](const char* kind,const Job& j){events.push_back({{"kind",kind},{"id",j.id},{"tick_ps",q.now().value},{"round",q.round()}});};
 std::function<void()> dispatch;
 dispatch=[&]{if(busy||ready.empty())return;const auto j=jobs.at(ready.front());ready.pop_front();busy=true;const auto start=q.now().value;emit("start",j);
  q.schedule_at(Tick{checked_add(start,j.service)},EventPhase::Completion,[&,j,start]{
   require(busy,"completion without busy server");emit("complete",j);
   completed.push_back({{"id",j.id},{"arrival_ps",j.arrival},{"service_ps",j.service},{"start_ps",start},{"completion_ps",q.now().value},{"wait_ps",start-j.arrival}});
   busy=false;dispatch();
  });
 };
 for(std::size_t i=0;i<jobs.size();++i)q.schedule_at(Tick{jobs[i].arrival},EventPhase::Arrival,[&,i]{emit("arrival",jobs[i]);ready.push_back(i);dispatch();});
 q.run();require(!busy&&ready.empty()&&q.pending()==0&&completed.size()==jobs.size(),"not drained");
 return {{"name",input.at("name")},{"completed",completed},{"events",events},{"final_tick_ps",q.now().value},{"executed_events",q.executed()},{"pending",q.pending()}};
}
static json probes(){
 json result;
 {EventQueue q;json seen=json::array();q.schedule_at(Tick{7},EventPhase::Arrival,[&]{seen.push_back("arrival");});q.schedule_at(Tick{7},EventPhase::Completion,[&]{seen.push_back("completion");});q.schedule_at(Tick{7},EventPhase::Deadline,[&]{seen.push_back("deadline");});q.run();result["same_tick_phases"]=seen;}
 {EventQueue q;json seen=json::array();q.schedule_at(Tick{0},EventPhase::Arrival,[&]{seen.push_back("arrival1");q.schedule_at(q.now(),EventPhase::Completion,[&]{seen.push_back("dynamic_completion");});});q.schedule_at(Tick{0},EventPhase::Arrival,[&]{seen.push_back("arrival2");});q.run();result["same_tick_dynamic_round"]=seen;result["dynamic_final_tick"]=q.now().value;}
 {EventQueue q;bool called=false;auto id=q.schedule_at(Tick{3},EventPhase::Arrival,[&]{called=true;});bool first=q.cancel(id),second=q.cancel(id);q.run();result["cancel"]={{"first",first},{"second",second},{"called",called},{"executed",q.executed()},{"pending",q.pending()}};}
 {EventQueue q;q.schedule_at(Tick{5},EventPhase::Arrival,[]{});bool skip=false,back=false,past=false;try{q.advance_idle_to(Tick{6});}catch(const std::invalid_argument&){skip=true;}q.advance_idle_to(Tick{5});result["idle_endpoint_unexecuted"]=q.executed()==0;q.run();try{q.advance_idle_to(Tick{4});}catch(const std::invalid_argument&){back=true;}try{q.schedule_at(Tick{4},EventPhase::Arrival,[]{});}catch(const std::invalid_argument&){past=true;}result["invalid_time_rejected"]={skip,back,past};}
 {EventQueue q;q.set_limits({1,100,100});q.schedule_at(Tick{1},EventPhase::Arrival,[]{});q.schedule_at(Tick{2},EventPhase::Arrival,[]{});bool rejected=false;try{q.run();}catch(const EventLimitReached&){rejected=true;}result["limit_before_resume"]={{"rejected",rejected},{"pending",q.pending()},{"executed",q.executed()},{"tick",q.now().value}};q.set_limits({2,100,100});q.run();result["limit_after_resume"]={{"pending",q.pending()},{"executed",q.executed()},{"tick",q.now().value}};}
 return result;
}
int main(int argc,char** argv){try{require(argc==3,"usage: a2_fifo input.json output.json");require(!std::filesystem::exists(argv[2]),"output exists");std::ifstream f(argv[1]);require(bool(f),"missing input");json in;f>>in;json out={{"mechanism","single-server FIFO on production EventQueue; not H0"},{"cases",json::array()}};for(const auto& c:in.at("cases"))out["cases"].push_back(simulate(c));out["queue_probes"]=probes();std::ofstream file(argv[2]);file<<out.dump();require(bool(file),"output failure");return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
