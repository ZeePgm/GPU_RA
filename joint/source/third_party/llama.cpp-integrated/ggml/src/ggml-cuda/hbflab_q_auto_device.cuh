#pragma once

#include <cuda_runtime.h>

// A distinct symbol per translation unit points at the same registered span and counters.
#define HBFLAB_Q_AUTO_JOIN_IMPL(prefix, suffix) prefix##suffix
#define HBFLAB_Q_AUTO_JOIN(prefix, suffix) HBFLAB_Q_AUTO_JOIN_IMPL(prefix, suffix)
#define HBFLAB_Q_AUTO_STATE HBFLAB_Q_AUTO_JOIN(hbflab_q_auto_state_, HBFLAB_Q_AUTO_TU)
#define HBFLAB_Q_AUTO_BIND HBFLAB_Q_AUTO_JOIN(hbflab_q_auto_bind_, HBFLAB_Q_AUTO_TU)

extern "C" {
__device__ void* HBFLAB_Q_AUTO_STATE = nullptr;

cudaError_t HBFLAB_Q_AUTO_BIND(void* state) {
    return cudaMemcpyToSymbol(HBFLAB_Q_AUTO_STATE, &state, sizeof(state));
}
}

#undef HBFLAB_Q_AUTO_BIND
#undef HBFLAB_Q_AUTO_STATE
#undef HBFLAB_Q_AUTO_JOIN
#undef HBFLAB_Q_AUTO_JOIN_IMPL
