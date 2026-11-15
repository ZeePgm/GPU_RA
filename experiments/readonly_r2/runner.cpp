#define NOMINMAX
#include "hbflab/async_module.hpp"
#include "hbflab/readonly_access.hpp"
#include <algorithm>
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
const void *r2_explicit_kernel();
const void *r2_explicit_timed_kernel();
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
// Same GPU timer placement used by the existing norm PTX dependency test.
std::string timed(std::string source) {
  auto start = source.find('{');
  need(start != std::string::npos, "entry missing");
  source.insert(
      start + 1,
      "\n.reg .b64 %r2_begin, %r2_end, %r2_base, %r2_delta, %r2_ptr;\n");
  std::smatch parameter;
  need(std::regex_search(
           source, parameter,
           std::regex(
               R"(ld\.param\.u64\s+(%\w+),\s*\[m6_weight_mul_param_2\];)")),
       "output parameter missing");
  source.insert(parameter.position() + parameter.length(),
                "\nmov.u64 %r2_base, " + parameter[1].str() + ";\n");
  std::istringstream input(source);
  std::string line, result;
  bool active = false;
  unsigned stores = 0;
  const std::regex store(R"(st\.global\.f32\s+\[(%\w+)\],)");
  while (std::getline(input, line)) {
    if (line.find("ld.global.f32") != std::string::npos && !active) {
      result += "mov.u64 %r2_begin, %globaltimer;\n";
      active = true;
    }
    std::smatch match;
    if (std::regex_search(line, match, store)) {
      need(active, "store without interval");
      result += "mov.u64 %r2_end, %globaltimer;\nsub.u64 %r2_delta, %r2_end, "
                "%r2_begin;\nsub.u64 %r2_ptr, " +
                match[1].str() +
                ", %r2_base;\nshl.b64 %r2_ptr, %r2_ptr, 1;\nadd.u64 %r2_ptr, "
                "%r2_ptr, %r2_base;\nadd.u64 %r2_ptr, %r2_ptr, "
                "32768;\nst.global.u64 [%r2_ptr], %r2_delta;\n";
      active = false;
      ++stores;
    }
    result += line + '\n';
  }
  need(stores == 2, "NVCC timer layout changed");
  return result;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc == 5 && std::string(argv[1]) == "replay") {
    const auto a = json::parse(read(argv[2])), b = json::parse(read(argv[3]));
    need(a["parents"] == b["parents"] && a["width"] == b["width"],
         "replay input mismatch");
    TimingBackend left, right;
    const auto bytes = a["width"].get<unsigned>() * 4;
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
  need(argc == 9, "strategy mode width repeats timed norm.ptx down.ptx "
                  "output.json required");
  const std::string strategy = argv[1], mode = argv[2];
  const unsigned width = std::stoul(argv[3]), repeats = std::stoul(argv[4]);
  const bool timers = std::stoi(argv[5]) != 0;
  need((strategy == "ptx" || strategy == "cuda") &&
           (mode == "native" || mode == "bypass" || mode == "modeled"),
       "invalid strategy/mode");
  need(width >= 32 && width <= 8192 && repeats && repeats <= 100 &&
           (!timers || mode == "modeled"),
       "invalid dimensions/timers");
  ck(cudaSetDevice(0));
  ck(cudaFree(nullptr));
  auto source = read(argv[6]);
  json result = {{"strategy", strategy},
                 {"mode", mode},
                 {"width", width},
                 {"timed", timers},
                 {"repeats", repeats}};
  const auto down_source=read(argv[7]);
  CUmodule down_original=nullptr;
  driver(cuModuleLoadData(&down_original,down_source.c_str()));
  driver(cuModuleUnload(down_original));
  result["down_original_driver_load"]=true;
  try {
    rewrite_async_ptx(down_source);
    result["down_supported"] = true;
  } catch (const std::exception &e) {
    result["down_supported"] = false;
    result["down_rejection"] = e.what();
  }
  std::vector<float> x(width), w(width), y(width), expected(width);
  for (unsigned i = 0; i < width; ++i) {
    x[i] = float(int(i % 23) - 11) / 8;
    w[i] = float(int(i % 19) - 9) / 16;
    expected[i] = x[i] * w[i];
  }
  float *dx, *dw, *dy;
  ck(cudaMalloc(reinterpret_cast<void **>(&dx), width * 4));
  ck(cudaMalloc(reinterpret_cast<void **>(&dw), width * 4));
  ck(cudaMalloc(reinterpret_cast<void **>(&dy), 32768 + width * 8));
  ck(cudaMemcpy(dx, x.data(), width * 4, cudaMemcpyHostToDevice));
  ck(cudaMemcpy(dw, w.data(), width * 4, cudaMemcpyHostToDevice));
  const bool modeled = mode == "modeled",
             explicit_path = strategy == "cuda" && mode != "native";
  const auto policy = explicit_path ? ReadonlyStrategy::CudaScalar
                                    : ReadonlyStrategy::PtxScalar;
  ReadonlyAccess registry;
  registry.add("norm", width * 4, policy);
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
    options.backend.max_events_per_batch = 10000000;
    service = std::make_unique<LiveTimingService>(options, 512, 2);
    service->register_readonly(dw, width * 4);
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
      function = loader->function(module, "m6_weight_mul");
      const auto identity = loader->identity(module);
      result["source_sha256"] = identity.source_sha256;
      result["rewritten_sha256"] = identity.rewritten_sha256;
    } else {
      driver(cuModuleLoadData(&raw, source.c_str()));
      driver(cuModuleGetFunction(&raw_function, raw, "m6_weight_mul"));
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
    ck(cudaMemset(dy, 0, 32768 + width * 8));
    std::vector<void *> args{&dx, &dw, &dy, const_cast<unsigned *>(&width)};
    ck(cudaEventRecord(begin));
    const auto cpu_before = process_cpu_ns();
    start = Clock::now();
    const auto launch = registry.dispatch(
        0, policy, dw, width * 4, modeled ? width : 0, [&]() -> std::uint64_t {
          if (explicit_path) {
            const void *kernel =
                timers ? r2_explicit_timed_kernel() : r2_explicit_kernel();
            if (modeled)
              return service->launch({2, 128}, kernel, args, nullptr);
            AsyncDeviceView view{};
            std::vector<void *> actual{&view};
            actual.insert(actual.end(), args.begin(), args.end());
            ck(cudaLaunchKernel(kernel, dim3(2), dim3(128), actual.data(), 0,
                                nullptr));
            return 0;
          }
          if (modeled)
            return loader->launch(function, {2, 128}, args,
                                  {dy, 32768 + width * 8}, nullptr,
                                  {dx, width * 4});
          driver(cuLaunchKernel(raw_function, 2, 1, 1, 128, 1, 1, 0, nullptr,
                                args.data(), nullptr));
          return 0;
        });
    const auto submit_ns = ns(start);
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
    ck(cudaMemcpy(y.data(), dy, width * 4, cudaMemcpyDeviceToHost));
    need(std::memcmp(y.data(), expected.data(), width * 4) == 0,
         "norm arithmetic changed");
    registry.handled(0);
    registry.end_step(true);
    json signature = json::array();
    if (service) {
      auto audit = service->take_audit();
      need(audit.size() == width, "request count");
      std::vector<bool> seen(width);
      std::vector<std::uint64_t> elapsed(width);
      if (timers)
        ck(cudaMemcpy(elapsed.data(), reinterpret_cast<char *>(dy) + 32768,
                      width * 8, cudaMemcpyDeviceToHost));
      for (const auto &a : audit) {
        const auto offset =
            a.first_address - reinterpret_cast<std::uintptr_t>(dw);
        need(a.launch == launch && offset % 4 == 0 && offset / 4 < width &&
                 !seen[offset / 4],
             "missing/duplicate/foreign access");
        seen[offset / 4] = true;
        need(a.first_bytes == 4 && a.sectors == 1 && a.span_count == 1 &&
                 a.end_ps >= a.arrival_ps,
             "scalar access semantics");
        if (timers)
          need(elapsed[offset / 4] >=
                   ((a.end_ps - a.arrival_ps) * 100 + 999) / 1000,
               "consumer preceded completion");
        signature.push_back({offset, a.first_bytes, a.sectors, a.span_count});
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
  result["parents"] = audits;
  result["output_sha256"] = sha256(
      std::string_view(reinterpret_cast<const char *>(y.data()), width * 4));
  if (loader)
    loader->unload(module);
  if (raw)
    driver(cuModuleUnload(raw));
  if (service) {
    service->finish();
    need(service->drained(), "service not drained");
    auto stats = service->statistics();
    need(stats.submitted == std::uint64_t(width) * (repeats + 1) &&
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
