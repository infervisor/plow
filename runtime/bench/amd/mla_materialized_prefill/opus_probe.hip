#include <hip/hip_runtime.h>

#include "fmha_fwd_hd192_v128_bf16_opus_defs.h"
#include "fmha_fwd_hd192_v128_bf16_opus_kernel.hpp"

using ProbeTraits = opus_gqa_d192_traits<32, 64, 8, true, false, false, false>;
template __global__ void gqa_d192_v128_kernel<ProbeTraits>(opus_gqa_d192_kargs);
