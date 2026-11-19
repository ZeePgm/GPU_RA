#include "q_auto_device_api.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

extern "C" cudaError_t hbflab_q_auto_bind_convert(void*);
extern "C" cudaError_t hbflab_q_auto_bind_mmvf(void*);

namespace {

constexpr std::size_t summary_words = 32;
constexpr std::size_t site_count = 32768;

struct ProbeState {
    const void* base;
    std::uint64_t bytes;
    std::uint32_t* counts;
    std::uint64_t* summary;
    std::uint64_t* sites;
};
static_assert(offsetof(ProbeState, bytes) == 8);
static_assert(offsetof(ProbeState, counts) == 16);
static_assert(offsetof(ProbeState, summary) == 24);
static_assert(offsetof(ProbeState, sites) == 32);

ProbeState state{};
ProbeState* device_state = nullptr;

cudaError_t destroy() {
    const auto convert = hbflab_q_auto_bind_convert(nullptr);
    const auto mmvf = hbflab_q_auto_bind_mmvf(nullptr);
    const auto device = cudaFree(device_state);
    const auto counts = cudaFree(state.counts);
    const auto summary = cudaFree(state.summary);
    const auto sites = cudaFree(state.sites);
    state = {};
    device_state = nullptr;
    for (auto result : {convert, mmvf, device, counts, summary, sites}) {
        if (result != cudaSuccess) return result;
    }
    return cudaSuccess;
}

}

extern "C" int hbflab_q_auto_device_begin(const void* base, std::size_t bytes) {
    if (device_state || !base || bytes < 32 || bytes % 2) return cudaErrorInvalidValue;
    state.base = base;
    state.bytes = bytes;
    auto result = cudaMalloc(&state.counts, bytes / 2 * sizeof(*state.counts));
    if (result == cudaSuccess) result = cudaMalloc(&state.summary, summary_words * sizeof(*state.summary));
    if (result == cudaSuccess) result = cudaMalloc(&state.sites, site_count * sizeof(*state.sites));
    if (result == cudaSuccess) result = cudaMalloc(&device_state, sizeof(state));
    if (result == cudaSuccess) result = cudaMemcpy(device_state, &state, sizeof(state), cudaMemcpyHostToDevice);
    if (result == cudaSuccess) result = hbflab_q_auto_bind_convert(device_state);
    if (result == cudaSuccess) result = hbflab_q_auto_bind_mmvf(device_state);
    if (result != cudaSuccess) destroy();
    return result;
}

extern "C" int hbflab_q_auto_device_step_begin() {
    if (!device_state) return cudaErrorInvalidValue;
    auto result = cudaMemset(state.counts, 0, state.bytes / 2 * sizeof(*state.counts));
    if (result == cudaSuccess) result = cudaMemset(state.summary, 0, summary_words * sizeof(*state.summary));
    if (result == cudaSuccess) result = cudaMemset(state.sites, 0, site_count * sizeof(*state.sites));
    if (result == cudaSuccess) result = cudaDeviceSynchronize();
    return result;
}

extern "C" int hbflab_q_auto_device_step_end(std::uint32_t* counts, std::size_t count,
                                              std::uint64_t* summary, std::size_t words) {
    if (!device_state || !counts || !summary || count != state.bytes / 2 || words != summary_words) {
        return cudaErrorInvalidValue;
    }
    auto result = cudaDeviceSynchronize();
    if (result == cudaSuccess) result = cudaMemcpy(counts, state.counts, count * sizeof(*counts), cudaMemcpyDeviceToHost);
    if (result == cudaSuccess) result = cudaMemcpy(summary, state.summary, words * sizeof(*summary), cudaMemcpyDeviceToHost);
    return result;
}

extern "C" int hbflab_q_auto_device_sites(std::uint64_t* counts, std::size_t count) {
    if (!device_state || !counts || count != site_count) return cudaErrorInvalidValue;
    return cudaMemcpy(counts, state.sites, count * sizeof(*counts), cudaMemcpyDeviceToHost);
}

extern "C" int hbflab_q_auto_device_finish() {
    if (!device_state) return cudaErrorInvalidValue;
    const auto result = cudaDeviceSynchronize();
    const auto cleanup = destroy();
    return result == cudaSuccess ? cleanup : result;
}
