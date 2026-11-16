#pragma once
#include "hbflab/live_runtime.hpp"
#include <functional>
#include <string>
#include <thread>

namespace hbflab {
enum class SampledRoute { Symbolic, Ptx, Cuda };
SampledRoute sampled_route(const std::string &);
const char *sampled_route_name(SampledRoute);
struct SampledRange {
    std::string name;
    const void *address;
    std::uint64_t bytes, logical;
};
struct SampledOptions {
    SampledRoute route=SampledRoute::Cuda;
    std::uint64_t layout_bytes=0, kv_bytes=0;
    hbf_eventlab::Geometry geometry{1,1,2,4,65536,1};
};
// The caller keeps borrowed CUDA allocations alive until close completes.
class SampledRegistration {
public:
    explicit SampledRegistration(SampledOptions);
    void add(SampledRange);
    void prepare();
    void prepare_kv(const std::function<void(LiveTimingService *)> &);
    void start();
    void close();
    LiveTimingService &service();
    const std::vector<SampledRange> &ranges() const { return ranges_; }
    std::uint64_t registered_bytes() const { return registered_bytes_; }
    std::uint64_t initialized_bytes() const { return initialized_bytes_; }
    static constexpr std::uint64_t alignment=32768;
private:
    void current() const;
    SampledOptions options_;
    std::vector<SampledRange> ranges_;
    std::unique_ptr<LiveTimingService> service_;
    std::thread::id owner_=std::this_thread::get_id();
    std::uint64_t registered_bytes_=0,initialized_bytes_=0;
    bool prepared_=false,started_=false,closed_=false,kv_prepared_=false;
};
}
