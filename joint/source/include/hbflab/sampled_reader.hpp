#pragma once
#include "hbflab/sampled_readonly.hpp"
#include <filesystem>

namespace hbflab {
enum class SampledAuditMode { Full, Summary };
struct SampledReadStats {
    std::uint64_t requests=0,requested_bytes=0,payload_bytes=0,publication_bytes=0;
    std::uint64_t launches=0,lane_checks=0,byte_checks=0,validation_d2h_bytes=0;
    std::uint64_t memset_bytes=0;
    std::uint64_t last_publication_ns=0;
    double read_wall_seconds=0;
};
class SampledReader {
public:
    SampledReader(SampledRegistration &,SampledRoute,const std::filesystem::path &, SampledAuditMode=SampledAuditMode::Full);
    ~SampledReader();
    SampledReader(const SampledReader &)=delete;
    SampledReader &operator=(const SampledReader &)=delete;
    void read(std::size_t range,std::uint64_t offset,std::uint64_t bytes);
    SampledReadStats statistics() const;
    void finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
