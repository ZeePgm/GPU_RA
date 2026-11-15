#include "hbflab/async_device.cuh"
#include <cuda_fp16.h>
using namespace hbflab;
template<unsigned W,bool Timed,bool Merge>
__global__ void read_plan(AsyncDeviceView view,const unsigned char* input,const unsigned* offsets,unsigned char* output) {
 const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned offset=offsets[i];const bool active=offset!=0xffffffffu;
 const unsigned mask=__ballot_sync(0xffffffffu,active);
 unsigned long long before=0,after=0;
 if constexpr(Timed)before=gpu_now_ns();
 Status status=Status::Ok;
 if(active && view.mailboxes) {
  if constexpr(Merge)status=AsyncDeviceClient(view).gate_load_adaptive(input+offset,W,mask);
  else status=AsyncDeviceClient(view).gate_load(input+offset,W,W);
 }
 if(active && status==Status::Ok) {
  // Copy actual consumed data. Output slots are disjoint even for duplicates.
  if constexpr(W==16) {
   const auto v=*reinterpret_cast<const uint4*>(input+offset);
   if constexpr(Timed)after=gpu_now_ns();
   *reinterpret_cast<uint4*>(output+i*32)=v;
  }else {
   const auto v=*reinterpret_cast<const unsigned*>(input+offset);
   if constexpr(Timed)after=gpu_now_ns();
   *reinterpret_cast<unsigned*>(output+i*32)=v;
  }
  if constexpr(Timed){auto* t=reinterpret_cast<unsigned long long*>(output+i*32+16);t[0]=before;t[1]=after;}
 }
}
const void* r3_kernel(unsigned width,bool timed,bool merge) {
#define CHOICE(W,T) if(width==W && timed==T)return merge?reinterpret_cast<const void*>(read_plan<W,T,true>):reinterpret_cast<const void*>(read_plan<W,T,false>);
 CHOICE(4,false) CHOICE(4,true) CHOICE(16,false) CHOICE(16,true)
 return nullptr;
}
__global__ void scatter(const unsigned char* tiles,unsigned char* mirror,unsigned count,unsigned width,unsigned base) {
 const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count)for(unsigned b=0;b<width;++b)mirror[base+i*width+b]=tiles[i*32+b];
}
void r3_scatter(const void* tiles,void* mirror,unsigned count,unsigned width,unsigned base){
 scatter<<<(count+255)/256,256>>>(static_cast<const unsigned char*>(tiles),static_cast<unsigned char*>(mirror),count,width,base);
}
__device__ float reduce(float s) {for(int i=16;i;i>>=1)s+=__shfl_xor_sync(0xffffffff,s,i);return s;}
__global__ void matrix(const half* w,const float* x,float* y,unsigned rows,unsigned columns) {
 const unsigned row=blockIdx.x%rows,token=blockIdx.x/rows;
 float s=0;__shared__ float sums[32];
 if(threadIdx.x<32)sums[threadIdx.x]=0;__syncthreads();
 for(unsigned i=threadIdx.x;i<columns/2;i+=256) {
  const float2 v=__half22float2(reinterpret_cast<const half2*>(w+row*columns)[i]);
  const float2 a=reinterpret_cast<const float2*>(x+token*columns)[i];
  s=__fmaf_rn(v.x,a.x,s);s=__fmaf_rn(v.y,a.y,s);
 }
 s=reduce(s);if((threadIdx.x&31)==0)sums[threadIdx.x/32]=s;__syncthreads();
 if(threadIdx.x<32)s=reduce(sums[threadIdx.x]);if(threadIdx.x==0)y[token*rows+row]=s;
}
__global__ void norm(const float* w,const float* x,float* y,unsigned columns,unsigned tokens) {
 const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<columns*tokens)y[i]=w[i%columns]*x[i];
}
void r3_consume(const void* w,const float* x,float* y,unsigned rows,unsigned columns,unsigned tokens,bool f32) {
 if(f32)norm<<<(columns*tokens+255)/256,256>>>(static_cast<const float*>(w),x,y,columns,tokens);
 else matrix<<<rows*tokens,256>>>(static_cast<const half*>(w),x,y,rows,columns);
}
