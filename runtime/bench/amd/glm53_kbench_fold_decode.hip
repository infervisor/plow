/* glm53_kbench_fold_decode.hip — the interpreter's MlaMergeFold body at the GLM-5.3 TP8 decode
 * rung-20 shape, for the tier-2 comparison against the native FP32 fold
 * (crates/plowrt/src/exec/amd_mla_fold.rs, `native_fold_decode_rung20_timing`).
 *
 * The production header verbatim, compiled with the decode object's geometry (PLOW_WG_WAVES=8):
 * at rows 20 x 8 heads the emitter hands the packet 160 workgroups and `exec_mla_merge_fold`
 * takes `d_mla_merge_fold<512, 256>` (bh*8 > nblk). `nblk` is an argument rather than
 * `gridDim.x` because the harness launches through the runtime's HSA path, which passes no
 * hidden kernel arguments.
 *
 *   hipcc --genco --offload-arch=gfx942 -O3 -std=c++17 -Iruntime/amd -Iruntime/common \
 *     -DPLOW_ARCH_SUFFIX=gfx942 -DPLOW_BUCKET_DECODE=1 -DPLOW_WG_WAVES=8 -DPLOW_FP8=1 \
 *     -DPLOW_FP8_KV=1 runtime/bench/amd/glm53_kbench_fold_decode.hip -o fold_decode.co
 *   (then unbundle to an ELF as scripts/build_glm_fold_lt.sh does)
 */
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_attention.h"

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void plow_kb_fold_decode(
    bf16* O, const float* Op, const float* mlp, const bf16* Wuv, unsigned n_batch,
    unsigned n_head, unsigned V, unsigned nsplit, unsigned nblk) {
    __shared__ float olds[512 + PLOW_WAVES * 256];
    d_mla_merge_fold<512, 256>(O, Op, mlp, Wuv, n_batch, n_head, V, nsplit, blockIdx.x, nblk,
                               olds);
}
