#include "hbflab/async_device.cuh"
#include <cstring>
#include <cuda_fp16.h>
using namespace hbflab;
__device__ __forceinline__ void ggml_cuda_mad(float &a, float v, float u) {
  a = __fmaf_rn(v, u, a);
}
template <int W> __device__ __forceinline__ float warp_reduce_sum(float a) {
#pragma unroll
  for (int offset = W / 2; offset > 0; offset >>= 1)
    a += __shfl_xor_sync(0xffffffff, a, offset, W);
  return a;
}
#ifndef R2_DOWN_ENTRY
#define R2_DOWN_ENTRY r2_down_raw
#endif
extern "C" __global__ void R2_DOWN_ENTRY(
#ifdef R2_DOWN_EXPLICIT
AsyncDeviceView view,
#endif
const half *w, const float *x,
                                       float *dst, unsigned hidden,
                                       unsigned ffn) {
  const unsigned token = blockIdx.x;
  const unsigned row = 0;
  float sum = 0;
  __shared__ float sums[32];
  if (threadIdx.x < 32)
    sums[threadIdx.x] = 0;
  __syncthreads();
  for (unsigned v = threadIdx.x; v < (ffn / 2); v += 256) {
    half2 weight{};
    bool ok = true;
    const auto *address = reinterpret_cast<const half2 *>(w + row * ffn) + v;
#ifdef R2_DOWN_TIMED
    const auto begin = gpu_now_ns();
#endif
#ifdef R2_DOWN_EXPLICIT
    if(view.mailboxes && AsyncDeviceClient(view).gate_load(address,4,4)!=Status::Ok) return;
#endif
    weight = *address;
#ifdef R2_DOWN_TIMED
    reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(dst)+32768)[token*(ffn/2)+v]=gpu_now_ns()-begin;
#endif
    if (ok) {
      const float2 values = __half22float2(weight);
      const float2 input = reinterpret_cast<const float2 *>(x + token * ffn)[v];
      ggml_cuda_mad(sum, values.x, input.x);
      ggml_cuda_mad(sum, values.y, input.y);
    }
  }
  sum = warp_reduce_sum<32>(sum);
  if ((threadIdx.x & 31) == 0)
    sums[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32)
    sum = warp_reduce_sum<32>(sums[threadIdx.x]);
  if (threadIdx.x == 0)
    dst[token * hidden + row] = sum;
}
