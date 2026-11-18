#define NOMINMAX
#include "hbflab/async_module.hpp"
#include "hbflab/readonly_access.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <chrono>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <windows.h>

using namespace hbflab;
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
const void *r2_down_explicit_kernel();
const void *r2_down_explicit_timed_kernel();
namespace {
void need(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void ck(cudaError_t code) {
  if (code != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(code));
}
void driver(CUresult code) {
  if (code != CUDA_SUCCESS)
    throw std::runtime_error("CUDA Driver error " + std::to_string(code));
}
std::string read(const std::filesystem::path &path) {
  std::ifstream f(path);
  need(bool(f), "input missing");
  return {std::istreambuf_iterator<char>(f), {}};
}
double ns(Clock::time_point start) {
  return std::chrono::duration<double, std::nano>(Clock::now() - start).count();
}
std::uint64_t process_cpu_ns() {
  FILETIME creation, exit, kernel, user;
  need(GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user) !=
           0,
       "process CPU clock");
  auto value = [](FILETIME time) {
    return (std::uint64_t(time.dwHighDateTime) << 32) | time.dwLowDateTime;
  };
  return (value(kernel) + value(user)) * 100;
}
// Instrument immediately before/after each packed weight load. Offset is
// derived from its address, so compiler unrolling cannot lose an observation.
std::string timed(std::string source) {
  auto at=source.find('{');
  source.insert(at+1,"\n.reg .b64 %r2_begin, %r2_end, %r2_base, %r2_weight, %r2_delta, %r2_ptr, %r2_offset, %r2_token_offset;\n.reg .b32 %r2_token, %r2_ffn;\n");
  std::istringstream in(source); std::string line,out;
  const std::regex parameter(R"(ld\.param\.u64\s+(%\w+),\s*\[r2_down_raw_param_([02])\];)");
  const std::regex load(R"(ld\.global\.u32\s+%\w+,\s*\[(%\w+)\];)");
  unsigned count=0;
  while(std::getline(in,line)) {
    std::smatch m;
    if(std::regex_search(line,m,load)) {
      out+="mov.u32 %r2_token, %ctaid.x;\nld.param.u32 %r2_ffn, [r2_down_raw_param_4];\nmul.wide.u32 %r2_token_offset, %r2_token, %r2_ffn;\nshl.b64 %r2_token_offset, %r2_token_offset, 2;\nmov.u64 %r2_begin, %globaltimer;\n"+line+"\nmov.u64 %r2_end, %globaltimer;\nsub.u64 %r2_delta, %r2_end, %r2_begin;\nsub.u64 %r2_offset, "+m[1].str()+", %r2_weight;\nshl.b64 %r2_offset, %r2_offset, 1;\nadd.u64 %r2_offset, %r2_offset, %r2_token_offset;\nadd.u64 %r2_ptr, %r2_base, %r2_offset;\nadd.u64 %r2_ptr, %r2_ptr, 32768;\nst.global.u64 [%r2_ptr], %r2_delta;\n";
      ++count; continue;
    }
    out+=line+'\n';
    if(std::regex_search(line,m,parameter))out+="mov.u64 "+std::string(m[2]=="0"?"%r2_weight":"%r2_base")+", "+m[1].str()+";\n";
  }
  need(count>0,"weight timer missing"); return out;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc == 5 && std::string(argv[1]) == "replay") {
    const auto a = json::parse(read(argv[2])), b = json::parse(read(argv[3]));
    need(a["parents"] == b["parents"] && a["width"] == b["width"] && a["tokens"]==b["tokens"] && a["weight_sha256"]==b["weight_sha256"],
         "replay input mismatch");
    TimingBackend left, right;
    const auto bytes = a["width"].get<unsigned>() * 2;
    left.reserve((bytes + 4095) / 4096 * 4096);
    right.reserve((bytes + 4095) / 4096 * 4096);
    for (const auto &parent : a["parents"]) {
      const auto old = left.read(parent[0].get<std::uint64_t>(),
                                 parent[1].get<unsigned>(), 123000);
      const auto updated = right.read(parent[0].get<std::uint64_t>(),
                                      parent[1].get<unsigned>(), 123000);
      need(old.status == Status::Ok && old.status == updated.status &&
               old.begin_ps == updated.begin_ps &&
               old.end_ps == updated.end_ps && old.sectors == updated.sectors,
           "fixed arrival result/time mismatch");
    }
    left.finish();
    right.finish();
    need(left.drained() && right.drained(), "replay drain");
    std::ofstream(argv[4]) << json({{"status", "pass"},
                                    {"requests", a["parents"].size()},
                                    {"integer_time_residual_ps", 0}})
                                  .dump(2);
    return 0;
  }
  need(argc == 9 || argc==10, "strategy mode ffn repeats timed down.ptx tokens "
                  "output.json required");
  const std::string strategy = argv[1], mode = argv[2];
  const unsigned width = std::stoul(argv[3]), repeats = std::stoul(argv[4]);
  const bool timers = std::stoi(argv[5]) != 0;
  need((strategy == "ptx" || strategy == "cuda") &&
           (mode == "native" || mode == "bypass" || mode == "modeled" || mode == "fault" || mode == "invalid_range"),
       "invalid strategy/mode");
  need(width >= 2 && width <= 16384 && width%2==0 && repeats && repeats <= 100 &&
           (!timers || mode == "modeled" || mode == "fault" || mode == "invalid_range"),
       "invalid dimensions/timers");
  ck(cudaSetDevice(0));
  ck(cudaFree(nullptr));
  auto source = read(argv[6]);
  json result = {{"strategy", strategy},
                 {"mode", mode},
                 {"width", width},
                 {"timed", timers},
                 {"repeats", repeats}};
  const unsigned count=width/2, hidden=8, tokens=std::stoul(argv[7]);
  need(tokens>=1 && tokens<=2,"only one/two CTA tokens supported");
  result["tokens"]=tokens;
  const bool fault=mode=="fault" || mode=="invalid_range";
  std::vector<float> x(width*tokens), y(hidden*tokens), expected(hidden*tokens);
  std::vector<__half> w(width);
  const unsigned pattern=argc==10?std::stoul(argv[9]):0;
  need(pattern<=1,"unknown data pattern");result["pattern"]=pattern;
  for(unsigned i=0;i<width;++i) {
    x[i]=float(int(i%23)-11)/8;
    w[i]=__float2half(float(int(i%19)-9)/16);
    if(pattern) {x[i]=std::sin(float(i)*1.2345f)*2.37f; w[i]=__float2half(std::cos(float(i)*0.8731f)*0.91f);}
  }
  // CPU reference reconstructs the frozen 256-thread FMA / xor-reduction
  // order, without calling a GPU implementation or the request planner.
  for(unsigned t=1;t<tokens;++t)for(unsigned i=0;i<width;++i)x[t*width+i]=x[i]*0.73f+float(int(i%7)-3)*0.17f;
  for(unsigned token=0;token<tokens;++token) {
  float lane[256]{};
  for(unsigned t=0;t<256;++t)for(unsigned v=t;v<count;v+=256) {
    lane[t]=std::fma(__half2float(w[2*v]),x[token*width+2*v],lane[t]);
    lane[t]=std::fma(__half2float(w[2*v+1]),x[token*width+2*v+1],lane[t]);
  }
  for(unsigned step=16;step;step/=2) {
    float old[256]; std::memcpy(old,lane,sizeof lane);
    for(unsigned t=0;t<256;++t)lane[t]=old[t]+old[t^step];
  }
  float sums[32]{};for(unsigned t=0;t<8;++t)sums[t]=lane[32*t];
  for(unsigned step=16;step;step/=2) {
    float old[32];std::memcpy(old,sums,sizeof sums);
    for(unsigned t=0;t<32;++t)sums[t]=old[t]+old[t^step];
  }
  expected[token*hidden]=sums[0];
  }
  float *dx,*dy; __half* dw;
  ck(cudaMalloc(reinterpret_cast<void**>(&dx),width*4*tokens));
  ck(cudaMalloc(reinterpret_cast<void**>(&dw),width*2));
  ck(cudaMalloc(reinterpret_cast<void**>(&dy),32768+count*8*tokens));
  ck(cudaMemcpy(dx,x.data(),width*4*tokens,cudaMemcpyHostToDevice));
  ck(cudaMemcpy(dw,w.data(),width*2,cudaMemcpyHostToDevice));
  const bool modeled = mode == "modeled" || fault,
             explicit_path = strategy == "cuda" && mode != "native";
  const auto policy = explicit_path ? ReadonlyStrategy::CudaScalar
                                    : ReadonlyStrategy::PtxScalar;
  ReadonlyAccess registry;
  registry.add("down", width * 2, policy);
  registry.bind(0, dw);
  registry.start();
  std::unique_ptr<LiveTimingService> service;
  auto start = Clock::now();
  if (modeled) {
    AsyncRuntimeOptions options;
    options.audit_limit = width + 1;
    options.time_scale = timers ? 100 : 1;
    options.timeout_ns = 1000000000ull;
    options.service_timeout_ns = 30000000000ull;
    options.backend.max_events_per_batch = mode=="fault"?100:10000000;
    service = std::make_unique<LiveTimingService>(options, 512, 2);
    service->register_readonly(dw, width * 2-(mode=="invalid_range"?4:0));
    service->start();
  }
  result["service_setup_ns"] = ns(start);
  std::unique_ptr<AsyncModuleInterceptor> loader;
  AsyncModuleHandle module{};
  AsyncFunctionHandle function{};
  CUmodule raw = nullptr;
  CUfunction raw_function = nullptr;
  start = Clock::now();
  if (!explicit_path) {
    if (modeled) {
      loader = std::make_unique<AsyncModuleInterceptor>(*service);
      module = loader->load_ptx(timers ? timed(source) : source);
      function = loader->function(module, "r2_down_raw");
      const auto identity = loader->identity(module);
      result["source_sha256"] = identity.source_sha256;
      result["rewritten_sha256"] = identity.rewritten_sha256;
    } else {
      driver(cuModuleLoadData(&raw, source.c_str()));
      driver(cuModuleGetFunction(&raw_function, raw, "r2_down_raw"));
    }
  }
  result["module_setup_ns"] = ns(start);
  cudaEvent_t begin, end;
  ck(cudaEventCreate(&begin));
  ck(cudaEventCreate(&end));
  json samples = json::array(), audits = json::array();
  for (unsigned iteration = 0; iteration < repeats + 1; ++iteration) {
    registry.begin_step(iteration);
    registry.expect(0);
    ck(cudaMemset(dy, 0, 32768 + count * 8*tokens));
    std::vector<void *> args{&dw, &dx, &dy, const_cast<unsigned *>(&hidden), const_cast<unsigned *>(&width)};
    ck(cudaEventRecord(begin));
    const auto cpu_before = process_cpu_ns();
    start = Clock::now();
    const auto launch = registry.dispatch(
        0, policy, dw, width * 2, modeled ? count*tokens : 0, [&]() -> std::uint64_t {
          if (explicit_path) {
            const void *kernel =
                timers ? r2_down_explicit_timed_kernel() : r2_down_explicit_kernel();
            if (modeled)
              return service->launch({tokens, 256}, kernel, args, nullptr);
            AsyncDeviceView view{};
            std::vector<void *> actual{&view};
            actual.insert(actual.end(), args.begin(), args.end());
            ck(cudaLaunchKernel(kernel, dim3(tokens), dim3(256), actual.data(), 0,
                                nullptr));
            return 0;
          }
          if (modeled)
            return loader->launch(function, {tokens, 256}, args,
                                  {dy, 32768 + count * 8*tokens}, nullptr,
                                  {dx, width * 4*tokens});
          driver(cuLaunchKernel(raw_function, tokens, 1, 1, 256, 1, 1, 0, nullptr,
                                args.data(), nullptr));
          return 0;
        });
    const auto submit_ns = ns(start);
    if(fault) {
      bool rejected=false; std::string message;
      try {service->synchronize();}catch(const std::exception& e){rejected=true;message=e.what();}
      need(rejected,"injected failure accepted");
      loader.reset(); service.reset(); ck(cudaDeviceSynchronize());
      registry.abort();
      result["status"]="expected_failure"; result["error"]=message;
      result["cuda_drained"]=true;
      ck(cudaFree(dx));ck(cudaFree(dw));ck(cudaFree(dy));
      ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
      std::ofstream(argv[8])<<result.dump(2);return 0;
    }
    if (service)
      service->synchronize();
    else
      ck(cudaDeviceSynchronize());
    const auto wall_ns = ns(start);
    const auto cpu_used = process_cpu_ns() - cpu_before;
    ck(cudaEventRecord(end));
    ck(cudaEventSynchronize(end));
    float milliseconds;
    ck(cudaEventElapsedTime(&milliseconds, begin, end));
    ck(cudaMemcpy(y.data(), dy, 4*tokens*hidden, cudaMemcpyDeviceToHost));
    need(std::memcmp(y.data(), expected.data(), 4*tokens*hidden) == 0,
         "down arithmetic changed");
    registry.handled(0);
    registry.end_step(true);
    json signature = json::array();
    if (service) {
      auto audit = service->take_audit();
      need(audit.size() == count*tokens, "request count");
      std::vector<bool> seen(count*tokens);
      std::vector<std::uint64_t> elapsed(count*tokens);
      if (timers)
        ck(cudaMemcpy(elapsed.data(), reinterpret_cast<char *>(dy) + 32768,
                      count * 8*tokens, cudaMemcpyDeviceToHost));
      for (const auto &a : audit) {
        const auto offset =
            a.first_address - reinterpret_cast<std::uintptr_t>(dw);
        const auto index=(a.producer/256)*count+offset/4;
        need(a.launch == launch && offset % 4 == 0 && offset / 4 < count &&
                 index<count*tokens && !seen[index],
             "missing/duplicate/foreign access");
        seen[index] = true;
        need(a.first_bytes == 4 && a.sectors == 1 && a.span_count == 1 &&
                 a.end_ps >= a.arrival_ps,
             "scalar access semantics");
        if (timers)
          need(elapsed[index] >=
                   ((a.end_ps - a.arrival_ps) * 100 + 999) / 1000,
               "consumer preceded completion");
        signature.push_back({offset, a.first_bytes, a.sectors, a.span_count});
        if(iteration==1)result["arrivals"].push_back({{"producer",a.producer},{"offset",offset},{"arrival_ps",a.arrival_ps},{"end_ps",a.end_ps},{"dispatch",a.dispatch},{"launch",a.launch},{"ticket",a.ticket}});
      }
      std::sort(signature.begin(), signature.end());
    }
    if (iteration == 0)
      result["first_launch_ns"] = wall_ns;
    else {
      samples.push_back({{"wall_ns", wall_ns},
                         {"submit_ns", submit_ns},
                         {"stream_interval_ns", milliseconds * 1e6},
                         {"process_cpu_ns", cpu_used}});
      if (iteration == 1)
        audits = signature;
    }
  }
  registry.finish();
  result["samples"] = samples;
  result["weight_sha256"]=sha256(std::string_view(reinterpret_cast<const char*>(w.data()),width*2));
  result["parents"] = audits;
  result["output_sha256"] = sha256(
      std::string_view(reinterpret_cast<const char *>(y.data()), 4*tokens*hidden));
  if (loader)
    loader->unload(module);
  if (raw)
    driver(cuModuleUnload(raw));
  if (service) {
    service->finish();
    need(service->drained(), "service not drained");
    auto stats = service->statistics();
    need(stats.submitted == std::uint64_t(count)*tokens * (repeats + 1) &&
             stats.completed == stats.submitted && !stats.audit_dropped,
         "final accounting");
    result["submitted"] = stats.submitted;
    result["input_closed"] = stats.backend.input_closed;
    result["registered_globals"] = stats.backend.registered_globals;
    result["completed_globals"] = stats.backend.completed_globals;
  }
  result["status"] = "pass";
  result["drained"] = true;
  ck(cudaEventDestroy(begin));
  ck(cudaEventDestroy(end));
  ck(cudaFree(dx));
  ck(cudaFree(dw));
  ck(cudaFree(dy));
  std::ofstream(argv[8]) << result.dump(2) << '\n';
  std::cout << strategy << ' ' << mode << ' ' << width << " passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
