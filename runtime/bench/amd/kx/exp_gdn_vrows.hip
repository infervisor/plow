/* kx experiment: gdn_vrows — does hoisting DeltaNet's per-head work off the V rows pay?
 *
 * THE QUESTION. In `d_qwen_gdn_step` a wave owns ONE V row of the state, and everything above
 * the state loop is a property of the (slot, head) pair, not of the row:
 *
 *     2*PL bf16 loads of q and k          two `wave_sum` reductions over them
 *     2x sqrtf + 2x divide                expf(-expf(a_log) * softplus(dt))   sigmoid(b)
 *
 * At vdim = 128 that is recomputed 128 times per head. `PLOW_QWEN_GDN_VROWS` gives a wave a strip
 * of consecutive V rows so it happens once per strip — the AMD port of the NVIDIA arm's
 * `PLOW_NV_GDN_STEP_VROWS8`. It is arithmetic-preserving by construction — each row's arithmetic
 * is unchanged and independent, and only the row-to-wave mapping moves — so every arm is
 * `KX_EXACT` and the harness DEMANDS bit-identity rather than assuming it. It gets it: 6144/6144
 * and 24576/24576 exact.
 *
 * WHAT THAT GATE DOES AND DOES NOT COVER. It compares the bf16 `out`. The f32 `state` the
 * recurrence carries is not compared bitwise by any harness here; the f64 oracle in
 * `runtime/tests/qwen_gdn_gfx942_test.hip` scores it, and VROWS 1/2/4/8 all land inside the
 * SHIPPED arm's own run-to-run band (max rel 1.01e-7 .. 1.17e-7 over five runs of VROWS=1 — that
 * test is not bit-reproducible across runs, which is a property of the test, not of these arms).
 * So: bit-identical output, and a state that is indistinguishable from the shipped arm's at the
 * resolution anything on this box can measure.
 *
 * AND THE COST, which is the whole reason this is a measurement and not a patch: VROWS divides
 * the launch's wave count by VROWS. At the shipped Qwen3-Next decode shape — hv=48, vdim=128,
 * kdim=128 — batch 1 is 6144 rows over 304 workgroups x 8 waves = 2432 waves, i.e. 2.5 rows per
 * wave BEFORE any tiling. VROWS=4 leaves 1536 tiles for 2432 waves (37% idle) and VROWS=8 leaves
 * 768 (68% idle). Fewer instructions against fewer waves is exactly the trade this branch has
 * lost before; the `batch4` shape exists so the answer is read at two occupancies, not one.
 *
 * WHERE THE STATE LIVES, and why it is in the OUTPUT buffer. The recurrence is read-modify-write
 * on `state`. Putting it in the read-only arena would let arm A's mutation change what arm B
 * reads and quietly destroy the correctness gate. kx zeroes the output buffer before EVERY
 * correctness launch, so `state` carved out of it starts at 0 for every arm — identical initial
 * conditions, and a bit-identity claim that means something.
 *
 * WHAT IT DOES NOT ANSWER. `PLOW_QWEN_GDN` is 0 in `scripts/build_gfx942.sh`, so op 137 is
 * compiled out of every shipped gfx942 object and no served token executes it. There is no model
 * measurement behind this file and there cannot be one on this box. Numerical validation is the
 * f64 oracle in `runtime/tests/qwen_gdn_gfx942_test.hip`, not this harness.
 */
#include "kx.h"

#ifndef KX_HOST
#include "op_qwen_gdn.h"

/* Shape constants are Qwen3-Next's, the same ones `qwen_gdn_gfx942_test.hip` uses:
 * hk=16, hv=48, kdim=128, vdim=128. `kx_d0` carries the decode batch. */
#define GDN_HK 16u
#define GDN_HV 48u
#define GDN_DK 128u
#define GDN_DV 128u

KX_BODY(b_gdn) {
    const unsigned batch = kx_d0;
    (void)kx_d1;
    (void)kx_d2;
    const unsigned packed = 2u * GDN_HK * GDN_DK + GDN_HV * GDN_DV;

    bf16* const out = (bf16*)kx_out;
    /* 1 MiB past `out`, which is 12 KiB at batch 1 — clear of it and 4-byte aligned. */
    float* const state = (float*)((char*)kx_out + (1u << 20));

    const bf16* const qkv = (const bf16*)kx_x;
    const bf16* const a = (const bf16*)kx_aux;
    const bf16* const b = a + 4096u;
    const bf16* const a_log = a + 8192u;
    const bf16* const dt_bias = a + 8192u + 256u;

    for (unsigned e = 0; e < kx_nrep; e++)
        d_qwen_gdn_step(out, qkv, a, b, (const void*)a_log, dt_bias, state, nullptr, GDN_HK,
                        GDN_HV, GDN_DK, GDN_DV, batch, 0.08838834764831845f, 1e-6f, 0u,
                        blockIdx.x, kx_nblk);
    (void)packed;
}
KX_ARM_OF(k_gdn, b_gdn)
KX_ARM_OF(k_gdn_aa, b_gdn)

#else /* ------------------------------------------------------- host descriptor */

static const kx_variant VAR[] = {
    {"v1", "-DPLOW_QWEN_GDN_VROWS=1"},
    {"v2", "-DPLOW_QWEN_GDN_VROWS=2"},
    {"v4", "-DPLOW_QWEN_GDN_VROWS=4"},
    {"v8", "-DPLOW_QWEN_GDN_VROWS=8"},
};

static const kx_arm ARM[] = {
    {"k_gdn", 0, KX_BASE | KX_REF, "the shipped mapping: one V row per wave"},
    {"k_gdn_aa", 0, KX_AA, "byte-identical device code to k_gdn"},
    {"k_gdn", 1, KX_EXACT, "2 V rows per wave; head-level q/k, sqrt/div and gates hoisted"},
    {"k_gdn", 2, KX_EXACT, "4 V rows per wave"},
    {"k_gdn", 3, KX_EXACT, "8 V rows per wave — the NVIDIA arm's VROWS8, ported"},
};

/* grid is `b.all()` at `DevOp::QwenGdnStep` (crates/devgen/src/qwen35.rs) = n_cu = 304.
 * `reps` rather than `slab_bytes`: this body streams no weights, so an arena walk would measure
 * a memory system that op 137 does not have. */
static const kx_shape SHP[] = {
    /* name, d0=batch, d1, d2, grid, inst, slab_bytes, reps, out_elems */
    {"batch1", 1, 0, 0, 304, 0, 0.0, 16, 1 * 48 * 128},
    {"batch4", 4, 0, 0, 304, 0, 0.0, 16, 4 * 48 * 128},
};

static const kx_exp EXP = {
    "gdn_vrows",
    "does giving a wave a strip of V rows pay for the waves it takes away?",
    "runtime/bench/amd/kx/exp_gdn_vrows.hip",
    /* The decode object's defines plus the family gate, which is 0 in the shipped recipe: this
     * body is NOT in any shipped gfx942 object and the bench has to turn it on to see it. */
    "-DPLOW_BUCKET_DECODE=1 -DPLOW_WG_WAVES=8 -DPLOW_QWEN_GDN=1",
    "gfx942", 512, KX_DT_BF16,
    VAR, 4, ARM, 5, SHP, 2,
    0.98, 1.02, /* A/A tolerance */
    0.0,        /* every arm is KX_EXACT; no tolerance is offered */
    "VROWS=1 is ISA-identical to HEAD; static instruction totals barely move (2832 -> 2934) "
    "because the saving is dynamic, and the loss is the wave count",
};
extern "C" const kx_exp* kx_experiment(void) { return &EXP; }
#endif
