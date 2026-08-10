/* Does ONE flash-MLA-decode launch serve rows with DIFFERENT sequence lengths?
 *
 * This is the prerequisite for fusing decode rows into a prefill chunk's M dimension: that
 * design needs one attention launch to handle per-row KV lengths. `d_flash_mla_decode` reads
 * `kv_len[b]` per row (op_attention.h:2003), and `decode_step_batched` documents ragged pos/kvlen
 * as supported -- but neither is a measurement, and the campaign's own history says an unexercised
 * contract is a claim, not a fact.
 *
 * METHOD. For each row r, compare:
 *    A) ONE launch, n_batch=B, kv_len = the RAGGED vector
 *    B) ONE launch, n_batch=B, kv_len = uniform L_r on every row
 * and check row r's output slice is BIT-IDENTICAL between them. Memory layout is identical in
 * both; only the kv_len vector differs. So a mismatch means either the kernel does not honour
 * kv_len per row, or rows contaminate each other.
 *
 * The uniform arm is the control rather than a CPU oracle on purpose: it exercises the SAME
 * kernel and the same launch shape, so it isolates the one variable and cannot fail for an
 * unrelated numerics reason.
 */
#include <hip/hip_runtime.h>
/* Same preamble order as runtime/amd/test_kernels.hip -- op_attention.h depends on the
 * address-space and MoE macros amd_common.h/op_norm.h establish first, and including it
 * standalone fails with `cannot pass pointer to address space '1'`. */
#include "amd/amd_common.h"
#include "amd/op_elementwise.h"
#include "amd/op_norm.h"
#ifndef PLOW_MOE_GEMMA
#define PLOW_MOE_GEMMA 1
#endif
#ifndef PLOW_MOE_GEMMA_PF
#define PLOW_MOE_GEMMA_PF 1
#endif
#include "amd/op_attention.h"

#ifndef GLM_MLA_GF
#define GLM_MLA_GF 8
#endif

extern "C" __global__ __attribute__((flatten))
__launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void mla_dec(
    float* Opart, float* mlpart, const bf16* Qabs, const bf16* Qrope, const bf16* Ckv,
    const bf16* Krope, const int* kv_len, unsigned n_batch, unsigned n_head, unsigned kv_stride,
    unsigned window, float scale, unsigned nsplit) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    d_flash_mla_decode<512, 64, GLM_MLA_GF>(Opart, mlpart, Qabs, Qrope, Ckv, Krope, kv_len,
                                            n_batch, n_head, kv_stride, window, scale, nsplit,
                                            PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}
