#pragma once
#include "instrumented_bridge.hpp"
#include <filesystem>
#include <cstdint>

extern "C" {
HBFLAB_M2_API void restored_bind(const char* route, const ggml_tensor* const* weights,
    const std::uint64_t* logical_bases, unsigned count, std::uint64_t layout_bytes,
    const char* profile, const char* symbolic_config, const char* output, bool full_audit);
HBFLAB_M2_API void restored_step_begin(int step);
HBFLAB_M2_API void restored_before(const ggml_tensor* node);
HBFLAB_M2_API void restored_step_end();
HBFLAB_M2_API void restored_finish();
HBFLAB_M2_API void restored_abort();
}

bool restored_ptx_mul(ggml_tensor* node, cudaStream_t stream);
void restored_b2_bind(const ggml_tensor* const*, const std::uint64_t*, unsigned,
                      std::uint64_t, const std::filesystem::path&, bool);
void restored_b2_read(const ggml_tensor*, unsigned offset, unsigned bytes, int step);
void restored_b2_finish();
