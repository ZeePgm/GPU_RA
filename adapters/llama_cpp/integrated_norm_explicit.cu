// Same norm arithmetic and launch shape as the PTX entry, with an explicit gate.
#define HBFLAB_R2_EXPLICIT
#define R2_ENTRY m6_weight_mul_explicit
#include "integrated_weight_mul.cu"

const void* r2_explicit_kernel() {
    return reinterpret_cast<const void*>(m6_weight_mul_explicit);
}
