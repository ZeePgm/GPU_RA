#define R2_DOWN_EXPLICIT
#define R2_DOWN_ENTRY r2_down_explicit
#include "../../adapters/llama_cpp/integrated_down_scalar.cu"
const void* r2_down_explicit_kernel() {return reinterpret_cast<const void*>(r2_down_explicit);}
