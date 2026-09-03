#include <hip/hip_runtime.h>

using bf16 = __bf16;

struct MlaMaterializePackArgs {
    bf16* k;
    bf16* v;
    const bf16* kv;
    const bf16* k_rope;
    unsigned t;
    unsigned heads;
    unsigned qk_nope;
    unsigned qk_rope;
    unsigned v_head;
};

extern "C" __global__ __launch_bounds__(256, 4)
void plow_mla_materialize_pack_gfx950(MlaMaterializePackArgs a) {
    const size_t rows = static_cast<size_t>(a.t) * a.heads;
    const size_t kdim = a.qk_nope + a.qk_rope;
    const size_t kvdim = a.qk_nope + a.v_head;
    const size_t total = rows * (kdim + a.v_head);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (i < rows * kdim) {
            const size_t row = i / kdim;
            const size_t d = i - row * kdim;
            a.k[i] = d < a.qk_nope
                         ? a.kv[row * kvdim + d]
                         : a.k_rope[(row / a.heads) * a.qk_rope + d - a.qk_nope];
        } else {
            const size_t j = i - rows * kdim;
            const size_t row = j / a.v_head;
            const size_t d = j - row * a.v_head;
            a.v[j] = a.kv[row * kvdim + a.qk_nope + d];
        }
    }
}

extern "C" __device__ __attribute__((used)) const unsigned char
    plow_mla_materialize_pack_abi_1 = 1;
extern "C" __device__ __attribute__((used)) const unsigned char
    plow_mla_materialize_pack_hd192_v128_1 = 1;
extern "C" __device__ __attribute__((used)) const unsigned char
    plow_mla_materialize_pack_nospill_1 = 1;
