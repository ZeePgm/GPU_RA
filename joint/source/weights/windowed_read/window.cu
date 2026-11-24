#include "hbflab/async_device.cuh"
#include <cuda_runtime.h>
using namespace hbflab;

template<bool Timed, bool Phases>
__global__ void b2_read_window(AsyncDeviceView view, const unsigned char* input,
                             unsigned base, unsigned bytes, unsigned char* mirror,
                             unsigned long long* times, unsigned long long* phases,
                             unsigned slow_ns) {
    const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned warp=tid/32, lane=tid%32, warps=gridDim.x*blockDim.x/32;
    // A producer owns one request credit until all of its lanes finish storing.
    // No cross-warp/CTA barrier participates in replenishment.
    for(unsigned q=warp;q<bytes/512;q+=warps) {
        const unsigned local=q*512+lane*16;
        if constexpr(Phases) if(lane==0) phases[2*q]=gpu_now_ns();
        if(slow_ns && warp==0) {
            const auto begin=gpu_now_ns();
            while(gpu_now_ns()-begin<slow_ns) __nanosleep(128);
        }
        unsigned long long before=0;
        if constexpr(Timed) before=gpu_now_ns();
        Status status=Status::Ok;
        if(view.mailboxes)
            status=AsyncDeviceClient(view).gate_load_adaptive(input+base+local,16,0xffffffffu);
        if(status!=Status::Ok) return;
        const uint4 value=*reinterpret_cast<const uint4*>(input+base+local);
        *reinterpret_cast<uint4*>(mirror+base+local)=value;
        __syncwarp();
        if constexpr(Timed) {
            times[2*(local/16)]=before;
            times[2*(local/16)+1]=gpu_now_ns();
        }
        if constexpr(Phases) if(lane==0) phases[2*q+1]=gpu_now_ns();
        __syncwarp();
    }
}
const void* b2_kernel(bool timed,bool phases) {
    if(timed)return phases?(const void*)b2_read_window<true,true>:(const void*)b2_read_window<true,false>;
    return phases?(const void*)b2_read_window<false,true>:(const void*)b2_read_window<false,false>;
}
