#define HBFLAB_R2_EXPLICIT
#define HBFLAB_R2_TIMED
#define R2_ENTRY m6_weight_mul_explicit_timed
#include "../../adapters/llama_cpp/integrated_weight_mul.cu"
const void* r2_explicit_timed_kernel(){return reinterpret_cast<const void*>(m6_weight_mul_explicit_timed);}
