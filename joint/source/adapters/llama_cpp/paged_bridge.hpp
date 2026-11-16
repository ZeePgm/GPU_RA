#pragma once
#include "instrumented_bridge.hpp"
#include "hbflab/live_runtime.hpp"
void m4_begin(int mode,hbflab::LiveTimingService*,const char* output);
void m4_step_begin(int step);
void m4_step_end();
HBFLAB_M2_API void m4_sequence_truncate(int,std::size_t);
HBFLAB_M2_API void m4_sequence_release(int);
void m4_finish();
void m4_abort();
bool m4_is_attention(const ggml_tensor*);
void m4_attention(ggml_tensor*,cudaStream_t);

inline hbf_eventlab::Geometry m4_geometry(){
    auto g=hbflab::BackendOptions{}.geometry;
    g.pages_per_block=4;g.blocks_per_bank=2048;
    return g;
}

// Diagnostic snapshot at an idle boundary: SHA256 of true KV bytes and prefix.
HBFLAB_M2_API void m4_sequence_audit(const char* path,std::size_t prefix);
HBFLAB_M2_API std::uint64_t m4_sequence_version(int);
