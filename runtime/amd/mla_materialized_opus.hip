#include <hip/hip_runtime.h>

#include "third_party/aiter_opus/fmha_fwd_hd192_v128_bf16_opus_defs.h"
#include "third_party/aiter_opus/fmha_fwd_hd192_v128_bf16_opus_kernel.hpp"

using PlowMlaMaterializedTraits =
    opus_gqa_d192_traits<32, 64, 8, true, false, false, false>;

extern "C" __global__ __launch_bounds__(PlowMlaMaterializedTraits::BLOCK_SIZE, 2)
void plow_mla_materialized_hd192_v128_gfx950(opus_gqa_d192_kargs args) {
    __shared__ char smem[PlowMlaMaterializedTraits::smem_size_bytes()];
    const int wave =
        __builtin_amdgcn_readfirstlane(opus::thread_id_x() /
                                      PlowMlaMaterializedTraits::WARP_SIZE);
    __builtin_amdgcn_sched_barrier(0);
    if (wave / 4)
        gqa_d192_v128::gqa_d192_v128_impl<PlowMlaMaterializedTraits, true>(args,
                                                                          smem);
    __builtin_amdgcn_sched_barrier(0);
    if (!(wave / 4))
        gqa_d192_v128::gqa_d192_v128_impl<PlowMlaMaterializedTraits, false>(args,
                                                                           smem);
}

extern "C" __device__ __attribute__((used)) const unsigned char
    plow_mla_materialized_opus_abi_1 = 1;
extern "C" __device__ __attribute__((used)) const unsigned char
    plow_mla_materialized_hd192_v128_1 = 1;
extern "C" __device__ __attribute__((used)) const unsigned char
    plow_mla_materialized_wave64_nospill_1 = 1;
