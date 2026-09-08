/* kx experiment: quant_scale — is the power-of-two quantization scale worth building differently?
 *
 * THE QUESTION. `fast_round_scale` is `exp2(ceil(log2(amax / top)))`, and every fp8/fp4 quantizer
 * in this tree transliterates it, then divides each element by the resulting scale.
 * `PLOW_QUANT_SCALE_EXP=1` builds the same exponent out of `frexpf` and two `ldexpf` — and gets
 * the exact reciprocal for free, turning the per-element IEEE divide into a multiply that is
 * BIT-IDENTICAL to it (the scale is an exact power of two and its reciprocal is representable
 * across the whole reachable range).
 *
 * WHAT THE STATIC ISA ALREADY SAYS, and why it does not settle it. The gfx942 disassembly of a
 * four-kernel probe shows arm 1 removing exactly one `v_log_f32`, one `v_exp_f32` and one divide
 * expansion per site — but the whole-kernel total goes UP at three of four sites (k_q_quant
 * 371 -> 383, k_pool_compress 569 -> 581, k_compress_fp8 957 -> 968) and down only for fp4
 * (1453 -> 1417). The audit's cost table put `log2f` at 35 instructions and `exp2f` at 28; those
 * are whole-probe numbers, and in these bodies the compiler lowers each to a single
 * `v_log_f32_e32` / `v_exp_f32_e32` plus a short range fixup. So the scale CONSTRUCTION is
 * roughly a wash and the static count cannot see the part that is not a wash:
 *
 *     `cmp_fake_quant_block` divides `qblk` elements by ONE loop-invariant scale, on ONE thread.
 *
 * At the shipped shapes that is 64 (or 32) divides per block, 7 blocks per pool, on 7 of 512
 * threads — a serial tail the static count shows once because the loop is rolled. That dynamic
 * divide is the only thing here that can pay, and it is what this experiment measures.
 *
 * THE TWO SHAPES are the reference's own, from `compress_pool_gfx942_test.hip`'s table:
 *   csa      ratio 4, coff 2, d 512, rd 64, qblk 64 — the attention compressor, fp8/e4m3
 *   indexer  ratio 4, coff 2, d 128, rd 64, qblk 32 — the indexer compressor, ROTATE, fp4/e2m1
 * `d == 128` selects ROTATE, which is why the template choice is a runtime branch on a
 * workgroup-uniform dim rather than two arms: two template instantiations would be two symbols
 * and the correctness gate needs one reference per shape.
 *
 * WHAT IT DOES NOT ANSWER. No blob on this box emits a compress op — GLM-5.3 `tp4-long` is MLA +
 * MoE — so there is no model measurement behind this file. Arm 1 also CHANGES VALUES at
 * power-of-two boundaries where `log2f`'s own rounding hides a fraction; that is a compatibility
 * question about the vendor kernel, it is enumerated in
 * `runtime/tests/quant_scale_gfx942_test.hip`, and it is why arm 1 is default-off regardless of
 * what this file measures.
 */
#include "kx.h"

#ifndef KX_HOST
#include "op_compress.h"

#define QS_RATIO 4u
#define QS_COFF 2u
#define QS_RD 64u

KX_BODY(b_qs) {
    const unsigned n_pools = kx_d0, d = kx_d1, qblk = kx_d2;
    __shared__ float lds[PLOW_THREADS > 512 ? PLOW_THREADS : 512];
    __shared__ float part[PLOW_WAVES];

    bf16* const out = (bf16*)kx_out;
    const bf16* const gamma = (const bf16*)kx_x;
    /* ape is [ratio][coff*d] f32 and cos/sin are [*][rd/2] f32; both are read out of the bf16
     * side buffer, so they are uniformly tiny. They are additive bias and a rotation, not a
     * scale, so tiny is harmless — and this body's cost is not data dependent. */
    const float* const ape = (const float*)kx_aux;
    const float* const cosb = ape + 65536u;
    const float* const sinb = ape + 65536u + 262144u;

    const size_t rows = (size_t)n_pools * QS_RATIO;
    const size_t row_elems = rows * QS_COFF * d;

    for (unsigned e = 0; e < kx_nrep; e++) {
        const bf16* const kv = (const bf16*)kx_w + (size_t)e * 2u * row_elems;
        const bf16* const score = kv + row_elems;
        if (d == 128u)
            d_compress_pool<true>(out, kv, score, ape, gamma, cosb, sinb, n_pools, QS_RATIO,
                                  QS_COFF, d, QS_RD, qblk, 1e-5f, 0u, blockIdx.x, kx_nblk, lds,
                                  part);
        else
            d_compress_pool<false>(out, kv, score, ape, gamma, cosb, sinb, n_pools, QS_RATIO,
                                   QS_COFF, d, QS_RD, qblk, 1e-5f, 0u, blockIdx.x, kx_nblk, lds,
                                   part);
    }
}
KX_ARM_OF(k_qs, b_qs)
KX_ARM_OF(k_qs_aa, b_qs)

#else /* ------------------------------------------------------- host descriptor */

static const kx_variant VAR[] = {
    {"log2", "-DPLOW_QUANT_SCALE_EXP=0"},
    {"frexp", "-DPLOW_QUANT_SCALE_EXP=1"},
};

static const kx_arm ARM[] = {
    {"k_qs", 0, KX_BASE | KX_REF, "the shipped scale: exp2f(ceilf(log2f(v))), one divide per element"},
    {"k_qs_aa", 0, KX_AA, "byte-identical device code to k_qs"},
    {"k_qs", 1, 0, "frexpf exponent + two ldexpf; the per-element divide becomes a multiply"},
};

/* grid: no packet emits this op, so there is no emitter count to copy. 304 = one workgroup per
 * pool at n_pools = n_cu, which is the shape the body's `slice/nblk` grid stride is written for.
 * `inst` stays 0 — a per-token total would be invented. */
static const kx_shape SHP[] = {
    /* name, d0=n_pools, d1=d, d2=qblk, grid, inst, slab_bytes, reps, out_elems */
    {"csa", 304, 512, 64, 304, 0, 304.0 * 4 * 2 * 512 * 2 * 2, 0, 304 * 512},
    {"indexer", 304, 128, 32, 304, 0, 304.0 * 4 * 2 * 128 * 2 * 2, 0, 304 * 128},
};

static const kx_exp EXP = {
    "quant_scale",
    "does building the power-of-two quant scale from the exponent beat log2/ceil/exp2 + divide?",
    "runtime/bench/amd/kx/exp_quant_scale.hip",
    "-DPLOW_BUCKET_DECODE=1 -DPLOW_WG_WAVES=8",
    "gfx942", 512, KX_DT_BF16,
    VAR, 2, ARM, 3, SHP, 2,
    0.98, 1.02, /* A/A tolerance */
    /* Arm 1 can pick a scale one binade apart from arm 0 at a boundary, and the fake quant then
     * round-trips through a different fp8/fp4 grid, so the tolerance is a bf16 magnitude and not
     * a ULP. The BOUNDARY behaviour is not tested here — quant_scale_gfx942_test.hip does that. */
    2e-2,
    "arm 0 is ISA-identical to HEAD; arm 1 removes one v_log_f32, one v_exp_f32 and one divide "
    "per site statically, and one divide per quantized element dynamically",
};
extern "C" const kx_exp* kx_experiment(void) { return &EXP; }
#endif
