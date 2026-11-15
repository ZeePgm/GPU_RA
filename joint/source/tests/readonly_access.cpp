#include "hbflab/readonly_access.hpp"
#include "hbflab/async_module.hpp"
#include "hbflab/backend.hpp"
#include <array>
#include <cuda.h>
#include <iostream>
#include <thread>

using namespace hbflab;
namespace {
void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
template <class F> void rejects(F operation) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::exception &) {
    rejected = true;
  }
  check(rejected, "operation should reject");
}
void exercise(ReadonlyStrategy strategy) {
  std::array<unsigned char, 8192> bytes{};
  ReadonlyAccess access;
  access.add("weight", 4096, strategy);
  rejects([&] { access.add("weight", 64, strategy); });
  rejects([&] { access.start(); });
  rejects([&] { access.bind(0, nullptr); });
  access.bind(0, bytes.data());
  rejects([&] { access.bind(0, bytes.data()); });
  access.add("overlap", 64, strategy);
  rejects([&] { access.bind(1, bytes.data() + 4095); });
  access.bind(1, bytes.data() + 4096);
  access.start();
  rejects([&] { access.add("late", 64, strategy); });
  rejects([&] { access.begin_step(1); });
  check(access.find("alias", bytes.data() + 10, 4) == 0,
        "overlap alias detection");
  check(access.find("unselected", bytes.data() + 5000, 4) == -1,
        "native range");
  bool wrong_owner_rejected = false;
  std::thread other([&] {
    try {
      access.begin_step(0);
    } catch (...) {
      wrong_owner_rejected = true;
    }
  });
  other.join();
  check(wrong_owner_rejected, "thread ownership");
  access.begin_step(0);
  access.expect(0);
  access.expect(1);
  rejects([&] { access.end_step(true); });
  rejects([&] { access.finish(); });
  const auto wrong = strategy == ReadonlyStrategy::PtxScalar
                         ? ReadonlyStrategy::CudaWarp
                         : ReadonlyStrategy::PtxScalar;
  rejects(
      [&] { access.dispatch(0, wrong, bytes.data(), 4, 1, [] { return 0; }); });
  rejects([&] {
    access.dispatch(0, strategy, bytes.data() + 4090, 16, 1, [] { return 0; });
  });
  rejects([&] {
    access.dispatch(0, strategy, bytes.data(), 0, 1, [] { return 0; });
  });
  check(access.dispatch(0, strategy, bytes.data() + 4092, 4, 1,
                        [] { return 7; }) == 7,
        "valid boundary launch");
  access.handled(0);
  access.handled(1);
  access.end_step(true);
  check(access.entries()[0].launches == 1 && access.entries()[0].requests == 1,
        "rejected calls not counted");
  access.finish();
  rejects([&] { access.begin_step(1); });
  rejects([&] { access.finish(); });

  ReadonlyAccess failed;
  failed.add("weight", 4096, strategy);
  failed.bind(0, bytes.data());
  failed.start();
  failed.begin_step(0);
  rejects([&] {
    failed.dispatch(0, strategy, bytes.data(), 4, 1, [] {
      throw std::runtime_error("load/launch failure");
      return 0;
    });
  });
  rejects([&] {
    failed.dispatch(0, strategy, bytes.data(), 4, 1, [] { return 0; });
  });
  rejects([&] { failed.finish(); });
  failed.abort();
  failed.abort();
}
void replay(ReadonlyStrategy strategy) {
  // Fixed byte plan supplied independently of registry bookkeeping.
  std::array<unsigned char, 4096> bytes{};
  const std::array<unsigned, 6> offsets{0, 64, 4032, 128, 64, 256};
  TimingBackend old_backend, new_backend;
  old_backend.reserve(4096);
  new_backend.reserve(4096);
  ReadonlyAccess access;
  access.add("weight", 4096, strategy);
  access.bind(0, bytes.data());
  access.start();
  access.begin_step(0);
  access.expect(0);
  for (const auto offset : offsets) {
    const auto old = old_backend.read(offset, 64, 123000);
    const auto updated =
        access.dispatch(0, strategy, bytes.data() + offset, 64, 1,
                        [&] { return new_backend.read(offset, 64, 123000); });
    check(old.status == updated.status && old.begin_ps == updated.begin_ps &&
              old.end_ps == updated.end_ps && old.sectors == updated.sectors,
          "fixed trace final result/time");
  }
  access.handled(0);
  access.end_step(true);
  access.finish();
  old_backend.finish();
  new_backend.finish();
  const auto a = old_backend.statistics(), b = new_backend.statistics();
  check(a.registered_globals == b.registered_globals &&
            a.completed_globals == b.completed_globals &&
            a.sectors == b.sectors,
        "fixed trace accounting");
  check(old_backend.drained() && new_backend.drained() && b.input_closed,
        "fixed trace drain");
}
void module_failure() {
  check(cuInit(0) == CUDA_SUCCESS, "CUDA init");
  CUcontext context;
  check(cuDevicePrimaryCtxRetain(&context, 0) == CUDA_SUCCESS,
        "primary context");
  check(cuCtxSetCurrent(context) == CUDA_SUCCESS, "current context");
  CUdeviceptr weight;
  check(cuMemAlloc(&weight, 4096) == CUDA_SUCCESS, "GPU weight allocation");
  {
    LiveTimingService service;
    service.register_readonly(reinterpret_cast<void *>(weight), 4096);
    service.start();
    AsyncModuleInterceptor loader(service);
    ReadonlyAccess access;
    access.add("weight", 4096, ReadonlyStrategy::PtxScalar);
    access.bind(0, reinterpret_cast<void *>(weight));
    access.start();
    access.begin_step(0);
    rejects([&] {
      access.dispatch(0, ReadonlyStrategy::PtxScalar,
                      reinterpret_cast<void *>(weight), 4096, 1,
                      [&] { return loader.load_ptx("invalid PTX module"); });
    });
    rejects([&] { access.finish(); });
    service.finish();
    check(service.drained(), "load failure leaves service drainable");
  }
  check(cuMemFree(weight) == CUDA_SUCCESS, "GPU weight release");
  check(cuDevicePrimaryCtxRelease(0) == CUDA_SUCCESS, "context release");
}
} // namespace
int main() try {
  exercise(ReadonlyStrategy::CudaScalar);
  exercise(ReadonlyStrategy::PtxScalar);
  exercise(ReadonlyStrategy::CudaWarp);
  replay(ReadonlyStrategy::CudaScalar);
  replay(ReadonlyStrategy::PtxScalar);
  replay(ReadonlyStrategy::CudaWarp);
  module_failure();
  std::cout << "readonly lifecycle/negative cases: three strategies passed; "
               "fixed backend replay: 18 exact results passed; real "
               "module-load failure drained\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
