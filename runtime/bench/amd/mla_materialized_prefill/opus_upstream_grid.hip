#include <hip/hip_runtime.h>

#include "fmha_fwd_hd192_v128_bf16_opus_defs.h"
#include "fmha_fwd_hd192_v128_bf16_opus_kernel.hpp"

using Traits = opus_gqa_d192_traits<32, 64, 8, true, false, false, false>;

extern "C" __global__ __launch_bounds__(Traits::BLOCK_SIZE, 2)
void oracle_mla_materialized_upstream_grid(opus_gqa_d192_kargs args) {
    __shared__ char smem[Traits::smem_size_bytes()];
    const int wave = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / Traits::WARP_SIZE);
    __builtin_amdgcn_sched_barrier(0);
    if (wave / 4) gqa_d192_v128::gqa_d192_v128_impl<Traits, true>(args, smem);
    __builtin_amdgcn_sched_barrier(0);
    if (!(wave / 4)) gqa_d192_v128::gqa_d192_v128_impl<Traits, false>(args, smem);
}

extern "C" __device__ __attribute__((used)) const unsigned char
    plow_mla_materialized_upstream_grid_oracle_1 = 1;
