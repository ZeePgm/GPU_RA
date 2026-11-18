#include <cuda_fp16.h>

// Each half2 load is a single R2 request. The converted values are the
// actual GET_ROWS result or ggml's existing F32 input to cuBLAS.
extern "C" __global__ void r2_half_to_float(const half2 *weight, float2 *output, unsigned pairs) {
    for (unsigned i = threadIdx.x; i < pairs; i += 256) {
        output[i] = __half22float2(weight[i]);
    }
}
