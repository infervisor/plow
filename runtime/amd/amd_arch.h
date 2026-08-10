/* amd_arch.h — the CDNA3 (gfx942) / CDNA4 (gfx950) instruction divergence, in ONE place.
 *
 * Op bodies call the wrappers here; they never name a builtin that only one arch has.
 * Before this file every CDNA4-only builtin was called unconditionally, so the whole
 * interpreter compiled for gfx950 and nothing else -- clang names each one exactly:
 *
 *   __builtin_amdgcn_mfma_f32_32x32x16_bf16   needs target feature gfx950-insts
 *   __builtin_amdgcn_mfma_f32_16x16x32_bf16   needs target feature gfx950-insts
 *   __builtin_amdgcn_fdot2_f32_bf16           needs target feature dot12-insts
 *   __builtin_amdgcn_mfma_scale_*_f8f6f4      needs target feature gfx950-insts
 *   __builtin_amdgcn_cvt_scalef32_pk_bf16_*   needs target feature fp{4,8}-cvt-scale-insts
 *   __builtin_amdgcn_global_load_lds @ 16 B   "invalid size value" (gfx942 caps at 4 B/lane)
 *
 * PLOW_CDNA4 is 0 in the HOST pass as well as on gfx942. That is deliberate and harmless:
 * the gfx942 arms below parse on the host, and no device code runs there.
 */
#ifndef PLOW_AMD_ARCH_H
#define PLOW_AMD_ARCH_H

#if defined(__gfx950__)
#define PLOW_CDNA4 1
#else
#define PLOW_CDNA4 0
#endif

/* MX (microscaling) block-scale converts and the scaled f8f6f4 matrix core are CDNA4 silicon.
 * CDNA3 has no fp4 datatype at all -- see crates/hwspec/src/isa.rs, which already models this
 * as Gfx942 { mx_scale_cvt: false, no MmaDtype::Fp4 }. The mxfp4 / A4W4 op arms are gated on
 * this and are absent from a gfx942 object rather than emulated. */
#define PLOW_HAS_MX_CVT PLOW_CDNA4

/* The SCALED f8f6f4 matrix core (v_mfma_scale_f32_32x32x64_f8f6f4). Separate from the converts
 * above because it is a matrix instruction with no CDNA3 analogue at all -- CDNA3's widest fp8
 * MFMA is K=16 and unscaled -- so an A4W4 body cannot be lowered to CDNA3, only replaced. */
#define PLOW_HAS_MX_MMA PLOW_CDNA4

/* Per-workgroup LDS ceiling. A CDNA4 workgroup may take 160 KiB; CDNA3 caps at 64 KiB, and the
 * cap is enforced by the COMPILER ("local memory (147464) exceeds limit (65536)"), not at launch.
 * Exposed so tile tables can be filtered per arch instead of duplicated. */
#ifndef PLOW_LDS_MAX_BYTES
#if PLOW_CDNA4
#define PLOW_LDS_MAX_BYTES 163840
#else
#define PLOW_LDS_MAX_BYTES 65536
#endif
#endif

/* --------------------------------------------------------------------------
 * bf16 matrix core.
 *
 * CDNA4 doubled the K of the bf16 MFMA: 32x32x16 and 16x16x32, at 512 MACs/cycle/core.
 * CDNA3 tops out at 32x32x8 / 16x16x16 and 256 MACs/cycle/core (crates/hwspec/src/amd/
 * mi300.rs vs mi350.rs). So one CDNA4 issue is exactly two CDNA3 issues over the two halves
 * of the same lane fragment -- same operand layout, same accumulator, same f32 accumulation
 * order per k-step. The fragment map in amd_common.h (MFMA_M/N/K = 32/32/16) is therefore
 * unchanged on both arches and no call site re-indexes.
 *
 * MEASURED (ROCm 7.14, 4-wave flash object): the split costs ZERO extra registers --
 * gfx942 lands at 256 VGPR / 256 AGPR / occ 1, against gfx950's 256 / 256 / occ 1.
 * It costs THROUGHPUT (two issues where CDNA4 needs one), which is the hardware's rate
 * difference and not something a different lowering recovers.
 * ------------------------------------------------------------------------- */
typedef bf16_t plow_bf16x4 __attribute__((ext_vector_type(4)));
typedef bf16_t bf16x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ f32x16 plow_mfma_bf16_32x32(bf16x8 a, bf16x8 b, f32x16 acc) {
#if PLOW_CDNA4
    return __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, acc, 0, 0, 0);
#else
    const plow_bf16x4 a0 = {a[0], a[1], a[2], a[3]}, a1 = {a[4], a[5], a[6], a[7]};
    const plow_bf16x4 b0 = {b[0], b[1], b[2], b[3]}, b1 = {b[4], b[5], b[6], b[7]};
    acc = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b0, acc, 0, 0, 0);
    return __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b1, acc, 0, 0, 0);
#endif
}

__device__ __forceinline__ f32x4 plow_mfma_bf16_16x16(bf16x8 a, bf16x8 b, f32x4 acc) {
#if PLOW_CDNA4
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0);
#else
    const plow_bf16x4 a0 = {a[0], a[1], a[2], a[3]}, a1 = {a[4], a[5], a[6], a[7]};
    const plow_bf16x4 b0 = {b[0], b[1], b[2], b[3]}, b1 = {b[4], b[5], b[6], b[7]};
    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a0, b0, acc, 0, 0, 0);
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a1, b1, acc, 0, 0, 0);
#endif
}

/* --------------------------------------------------------------------------
 * fp8 matrix core, UNSCALED.
 *
 * The w8a8 prefill GEMM (d_gemm_fp8_t) passes cbsz/blgp/opsel/scale all as compile-time 0, so
 * the backend selects the UNSCALED v_mfma_f32_32x32x64_f8f6f4 and drops the scale operands --
 * amd_common.h documents that trap, and scripts/asm_expect_gfx950.json asserts exactly this
 * mnemonic. Unscaled means the K=64 issue decomposes: CDNA3's widest fp8 MFMA is K=16, so four
 * of them over the four 8-byte quarters of the 32-byte fragment cover the same K.
 *
 * WHY THE QUARTERING IS EXACT, and it is the same argument as the bf16 split above. A 32x32xK
 * MFMA gives lane l row (l%32) and k-slots 8*(l/32)+[0..7] of the issue. Taking elements
 * [8q..8q+7] therefore feeds issue q the k values {8q..8q+7} from the low lanes and
 * {32+8q..32+8q+7} from the high ones -- a PERMUTATION of k, not a subset. A and B are permuted
 * IDENTICALLY, so every product still pairs A[m][k] with B[k][n], and the four issues partition
 * K=64 exactly once. Only the f32 accumulation GROUPING differs from CDNA4.
 *
 * RATE: this is where CDNA3 gives up the most. op_gemm.h measured the scaled/unscaled K=64 form
 * at 4532 TF/s -- exactly 2x bf16 -- while CDNA3's K=16 fp8 runs at the SAME rate as its bf16
 * (hwspec mi300.rs: fp8 512 vs bf16 256 MACs/cycle/core, i.e. 2x per instruction but a quarter
 * of the K per issue). So on gfx942 fp8 prefill buys memory footprint, not throughput.
 *
 * FORMAT, and it is the whole reason PLOW_FP8_MMA_FIX exists. The CDNA3 matrix core reads its
 * fp8 operands as e4m3FNUZ while plow's bytes are OCP e4m3 -- see the fp8 section below, where
 * the identity OCP(b) == 2 * FNUZ(b) (all 256 bytes but 0x80) is established and measured. The
 * matrix core cannot convert on the way in, but it does not have to: feed it the OCP bytes and
 * every operand is exactly HALVED, so the product is quartered and one scalar multiply in the
 * epilogue restores it. The single exception, 0x80, is masked to 0x00 as the tile is staged
 * (plow_fp8_mask_neg0) -- same value in OCP, and it keeps the core off its NaN encoding.
 * ------------------------------------------------------------------------- */
/* Epilogue correction for the halved CDNA3 fp8 operands: 2 (A) * 2 (B). Exactly 1.0 on CDNA4,
 * so the multiply folds away there. */
#if PLOW_CDNA4
#define PLOW_FP8_MMA_FIX 1.0f
#else
#define PLOW_FP8_MMA_FIX 4.0f
#endif
/* Duplicate of amd_common.h's typedef (legal, identical): that one is declared below this
 * include, and the wrapper needs the type here. */
typedef int mfma_f8f6f4_operand __attribute__((ext_vector_type(8)));

__device__ __forceinline__ f32x16 plow_mfma_fp8_32x32(mfma_f8f6f4_operand a,
                                                      mfma_f8f6f4_operand b, f32x16 acc) {
#if PLOW_CDNA4
    return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, acc, 0, 0, 0, 0, 0, 0);
#else
    union { mfma_f8f6f4_operand v; long q[4]; } ua{a}, ub{b};
#pragma unroll
    for (int q = 0; q < 4; q++)
        acc = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(ua.q[q], ub.q[q], acc, 0, 0, 0);
    return acc;
#endif
}

/* --------------------------------------------------------------------------
 * Packed bf16 dot.
 *
 * v_dot2c_f32_bf16 is CDNA4. CDNA3 carries the fp16 dot2 but NOT the bf16 one (clang:
 * "needs target feature dot12-insts"), and bf16->fp16 would lose exponent range on weights,
 * so the CDNA3 arm widens to f32 and issues two FMAs. That is the 24-VALU-per-16-bytes cost
 * the dot8 comment warns about -- but decode is bandwidth-bound (costmodel puts a 4.6 us
 * dispatch floor under it), so the arithmetic is expected to hide. VERIFY on the GEMV bench
 * before treating that as settled.
 * ------------------------------------------------------------------------- */
/* PROBE ONLY, AND IT COMPUTES THE WRONG ANSWER. Collapses the CDNA3 fallback from 6 VALU ops
 * (2 shifts, 2 masks, 2 dependent FMAs per bf16 PAIR) to 1, to price the emulated dot against
 * the memory stream. gfx950 has v_dot2c_f32_bf16 and does this in ONE instruction; gfx942 does
 * not, so the decode GEMV pays 24 VALU ops per 16 bytes of weight and every FMA chains on the
 * same accumulator. Whether that is the binding resource is the whole question, and this is how
 * to ask it -- the loads, the addressing and the LDS reads are untouched, only the arithmetic
 * is deleted. Never ship it. */
#ifndef GV_HACK_CHEAPDOT
#define GV_HACK_CHEAPDOT 0
#endif
__device__ __forceinline__ float plow_dot2_bf16(bf16x2 a, bf16x2 b, float acc) {
#if PLOW_CDNA4
    return __builtin_amdgcn_fdot2_f32_bf16(a, b, acc, false);
#elif GV_HACK_CHEAPDOT
    const unsigned pa = __builtin_bit_cast(unsigned, a), pb = __builtin_bit_cast(unsigned, b);
    float fa, fb;
    __builtin_memcpy(&fa, &pa, 4);
    __builtin_memcpy(&fb, &pb, 4);
    return __builtin_fmaf(fa, fb, acc);
#else
    /* BIT-cast, not a numeric conversion. `a[0]` is a __bf16 VALUE: (unsigned short)a[0] rounds
     * it to an integer, so every weight below 1.0 becomes 0 and the whole GEMV returns zero.
     * Measured -- gemm_gfx950_test's decode GEMV block failed at worst rel 0.9996 on all four
     * shapes while every MFMA GEMM passed, which is the signature of exactly this mistake.
     * bf16 -> f32 is a 16-bit left shift, the same widening bf2f() does below. */
    const unsigned pa = __builtin_bit_cast(unsigned, a), pb = __builtin_bit_cast(unsigned, b);
    float fa0, fb0, fa1, fb1;
    unsigned t;
    t = pa << 16;
    __builtin_memcpy(&fa0, &t, 4);
    t = pa & 0xffff0000u;
    __builtin_memcpy(&fa1, &t, 4);
    t = pb << 16;
    __builtin_memcpy(&fb0, &t, 4);
    t = pb & 0xffff0000u;
    __builtin_memcpy(&fb1, &t, 4);
    return __builtin_fmaf(fa0, fb0, __builtin_fmaf(fa1, fb1, acc));
#endif
}

/* --------------------------------------------------------------------------
 * MX dequant scaffolding for the CDNA3 arms of fp8_to_bf16v8 / fp4_to_bf16v8*.
 *
 * CDNA4 decodes an fp8 or fp4 pair straight to bf16 with a block scale in ONE instruction.
 * CDNA3 has v_cvt_pk_f32_fp8 (so fp8 costs a convert plus a pack) and NOTHING for fp4 (so fp4
 * is a table lookup). Both land in bf16 with no accuracy loss versus CDNA4:
 *
 *   fp8 e4m3 -> bf16 : <=3 mantissa bits, exact in f32 and exact in bf16. Truncating the
 *                      high half IS the round -- this is the two-step path CDNA4's single
 *                      instruction replaced, and the header above records it as bit-exact.
 *   fp4 e2m1 -> bf16 : 8 magnitudes, all exactly representable. The E8M0 block scale is a
 *                      power of two, so folding it is exact too.
 * ------------------------------------------------------------------------- */
typedef float f32x2 __attribute__((ext_vector_type(2)));

/* Two f32 -> one u32 holding a bf16 pair (lo in bits 0..15). Truncation, not RNE: every caller
 * feeds it values that are already exact in bf16 (see above), so there is nothing to round. */
__device__ __forceinline__ unsigned plow_pack_bf16x2(float lo, float hi) {
    unsigned a, b;
    __builtin_memcpy(&a, &lo, 4);
    __builtin_memcpy(&b, &hi, 4);
    return (a >> 16) | (b & 0xffff0000u);
}

/* ==========================================================================================
 * fp8 IS A DIFFERENT FORMAT ON CDNA3. This is the sharpest trap in the whole port.
 *
 * CDNA4's v_cvt_*_fp8 decode OCP **e4m3** (exponent bias 7, 0x80 = -0, 0x7f/0xff = NaN).
 * CDNA3's decode **e4m3fnuz** (bias 8, 0x80 = NaN, no signed zero, max 240 not 448).
 * MEASURED on this MI300X with v_cvt_pk_f32_fp8 over all 256 byte values:
 *
 *     byte 0x38 -> 0.5   (OCP says 1.0)      byte 0x40 -> 1.0   (OCP says 2.0)
 *     byte 0x80 -> NaN   (OCP says -0.0)     byte 0x7f -> 240.0 (OCP says NaN)
 *
 * So the SAME weight bytes mean different numbers on the two parts. Using the hardware convert
 * on gfx942 halves every value AND turns every negative-zero weight into a NaN that poisons the
 * whole dot product -- which is exactly what it did: the block-fp8 GEMV returned NaN for every
 * shape, and the test harness reported "PASS rel 0.0000" for most of them because its
 * `if (rel > worst)` comparison is false for NaN.
 *
 * plow's packets carry OCP e4m3 (that is what the compiler quantizes to and what gfx950 runs),
 * so CDNA3 must decode OCP in software. It is exact: e4m3 has 3 mantissa bits and bf16 has 7,
 * so every finite value lands on a bf16 with no rounding.
 * ========================================================================================== */
__device__ __forceinline__ unsigned short plow_fp8_ocp_to_bf16(unsigned b) {
    const unsigned sgn = (b & 0x80u) << 8; /* fp8 bit 7 -> bf16 bit 15 */
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    /* Subnormals are m * 2^-9; all 8 are exact bf16 constants, so a tiny table beats renormalising. */
    const unsigned short sub[8] = {0x0000, 0x3B00, 0x3B80, 0x3BC0, 0x3C00, 0x3C20, 0x3C40, 0x3C60};
    /* Normals: bf16 exp = (e - 7) + 127 = e + 120, and the 3 mantissa bits sit at bf16 bit 4. */
    unsigned mag = e ? (((e + 120u) << 7) | (m << 4)) : (unsigned)sub[m];
    if (e == 15u && m == 7u) mag = 0x7FC0u; /* the one OCP NaN encoding */
    return (unsigned short)(sgn | mag);
}

/* Zero every byte equal to 0x80 in a word. SWAR: xor makes the target byte 0x00, an EXACT
 * per-byte zero test marks it, and the marker widens to 0xff. ~7 VALU for 4 bytes.
 * NOT the classic `(t-0x01..) & ~t & 0x80..` — that one only answers "contains a zero": a real
 * zero byte borrows into the byte above it, falsely flagging an adjacent 0x01 (= weight byte
 * 0x81). Verified exhaustively over all 2^32 words on host. */
__device__ __forceinline__ unsigned plow_fp8_mask_neg0(unsigned w) {
    const unsigned t = w ^ 0x80808080u;
    const unsigned z = ~((t & 0x7f7f7f7fu) + 0x7f7f7f7fu) & ~t & 0x80808080u;
    return w & ~(z | (z - (z >> 7))); /* widen 0x80 -> 0xff, then clear */
}

/* Four OCP e4m3 bytes (one u32) -> two packed bf16 pairs.
 *
 * THE FAST PATH, and it rests on an identity VERIFIED EXHAUSTIVELY on this MI300X over all 256
 * byte values: the CDNA3 hardware decoder's e4m3fnuz output is EXACTLY half the OCP value for
 * every byte except 0x80. FNUZ and OCP differ only in exponent bias (8 vs 7), so one hardware
 * convert plus a doubling reproduces OCP exactly -- and the doubling is free in f32, and exact,
 * because e4m3 has 3 mantissa bits and lands on a bf16 with nothing to round.
 *
 * 0x80 is the sole exception (OCP -0, FNUZ NaN) and is masked to +0 first, which is the same
 * VALUE in OCP and keeps the decoder off its NaN encoding. This matters in practice, not in
 * theory: any weight that quantises to negative zero would otherwise turn the whole dot product
 * into a NaN, which is exactly what the block-fp8 GEMV did before the mask.
 *
 * DIVERGENCE, documented: OCP's NaN encodings (0x7f / 0xff) decode to 480 here rather than NaN.
 * Reaching them means a NaN weight in the packet, which is an upstream bug either way; the exact
 * scalar decoder below is kept for anything that must be bit-faithful on those two bytes. */
__device__ __forceinline__ void plow_fp8x4_ocp_to_bf16(unsigned w, unsigned& lo, unsigned& hi) {
    const unsigned m = plow_fp8_mask_neg0(w);
    const f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8(m, false); /* bytes 0,1 */
    const f32x2 c = __builtin_amdgcn_cvt_pk_f32_fp8(m, true);  /* bytes 2,3 */
    lo = plow_pack_bf16x2(a[0] * 2.0f, a[1] * 2.0f);
    hi = plow_pack_bf16x2(c[0] * 2.0f, c[1] * 2.0f);
}

/* Sixteen OCP e4m3 bytes (one b128 load) -> 16 RAW f32 — no bf16 pack, no x2.
 *
 * The fp8 GEMV fast path on THIS ISA. The packed-bf16 route costs CDNA3 twice:
 * plow_fp8x4_ocp_to_bf16 PACKS the cvt results into bf16 pairs, and `dot2` (no bf16 fdot2 here)
 * immediately UNPACKS them back to f32 for the FMA. Skipping the round trip — raw f32 from the
 * cvt, f32 FMA against an x row widened ONCE per chunk and shared across every column/stream
 * consuming it — measured -8..-11% on the standalone loop (fp8gemv3 probe: down 25.67 -> 23.62 us,
 * q_proj 10.01 -> 8.89 at 304/152 WGs). gfx950 keeps the packed path: its dequant is one native
 * cvt_scalef32_pk_bf16_fp8 per pair and its fdot2 consumes bf16 pairs directly.
 *
 * THE OCP x2 IS NOT APPLIED HERE — the caller folds it into the widened x (one mul per x element,
 * amortized across all consuming columns) instead of per weight element per column. Same 0x80
 * mask, same FNUZ-halving identity as the bf16 form above. VALUES equal the packed path's; only
 * the FMA accumulation ORDER differs (serial vs dot8's pair nesting).
 *
 * TRIED IN THE MEGAKERNEL AND REVERTED — the standalone win does NOT transfer. Wired into all
 * three fp8 GEMV bodies (pair loop, odd tail, GLU) and measured on the full model (Gemma-4-12B
 * fp8 occ4, 48 steps x 3 interleaved reps): 11.32 -> 11.42 ms/token @4k, 11.43 -> 11.52 @8k —
 * a 1% REGRESSION at identical resource stats (104 VGPR / 0 spills). The in-model GEMV is
 * memory-LATENCY-bound at 2 waves/SIMD, so the packed path's extra VALU was hiding under the
 * weight-load latency for free (the same verdict amd_common.h's cvt_scalef32 note records for
 * gfx950), while this path's 32 live f32 (wf+xf) per iteration cost scheduling room the
 * 104-register budget did not have. The standalone probe won BECAUSE it ran at a 256-register
 * budget. Kept as the documented probe; do not re-wire without changing the register story. */
__device__ __forceinline__ void plow_fp8x16_to_f32(unsigned w0, unsigned w1, unsigned w2,
                                                   unsigned w3, float* o) {
    const unsigned w[4] = {w0, w1, w2, w3};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned m = plow_fp8_mask_neg0(w[i]);
        const f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8(m, false);
        const f32x2 c = __builtin_amdgcn_cvt_pk_f32_fp8(m, true);
        o[i * 4 + 0] = a[0];
        o[i * 4 + 1] = a[1];
        o[i * 4 + 2] = c[0];
        o[i * 4 + 3] = c[1];
    }
}

/* f32 -> one OCP e4m3 byte, RNE, saturating at 448. The ENCODE half of the same divergence:
 * CDNA3's v_cvt_pk_fp8_f32 emits e4m3fnuz, so the hardware encoder cannot be used to produce the
 * OCP bytes the rest of plow (and every gfx950 object) expects. Software, and exact.
 *
 * Everything below 2^-9 rounds into the subnormal ladder m*2^-9; above that the value is
 * (1+m/8)*2^(e-7). Done by biasing the f32 exponent and rounding the mantissa RNE. */
__device__ __forceinline__ unsigned char plow_f32_to_fp8_ocp(float v) {
    unsigned u;
    __builtin_memcpy(&u, &v, 4);
    const unsigned sgn = (u >> 24) & 0x80u;
    u &= 0x7fffffffu;
    if (u >= 0x7f800000u) return (unsigned char)(sgn | 0x7fu); /* NaN/Inf -> the OCP NaN */
    const int exp = (int)(u >> 23) - 127;                      /* unbiased f32 exponent */
    const unsigned man = u & 0x7fffffu;
    unsigned mag;
    if (exp >= 9) {
        mag = 0x7eu; /* saturate at 448 = OCP e4m3 max finite */
    } else if (exp >= -6) {
        /* normal: keep 3 mantissa bits, round-to-nearest-even on bit 19. exp==8 lands in the
         * TOP binade (e=15: 256..448) -- it is NOT overflow; only what rounds past 448 is. */
        unsigned m = man >> 20, r = man & 0xfffffu;
        unsigned e = (unsigned)(exp + 7);
        if (r > 0x80000u || (r == 0x80000u && (m & 1u))) {
            if (++m == 8u) { m = 0u; ++e; }
        }
        if (e >= 16u || (e == 15u && m == 7u)) { e = 15u; m = 6u; } /* (15,7) is the NaN byte */
        mag = (e << 3) | m;
    } else {
        /* subnormal: value/2^-9, rounded RNE; a carry to 8 is the min normal 2^-6, not 7*2^-9 */
        const float q = v < 0.0f ? -v : v;
        const unsigned m = (unsigned)__builtin_roundevenf(q * 512.0f);
        mag = (m > 7u) ? 0x08u : m;
    }
    /* NEVER EMIT 0x80 (OCP -0): a negative value whose magnitude rounds to zero encodes as +0,
     * the SAME VALUE. This makes every CDNA3 device-produced e4m3 stream (the fp8 KV cache, the
     * fused w8a8 activation quant) 0x80-free BY CONSTRUCTION, so their decoders can skip the
     * neg-0 SWAR mask (~8 of the chain's 14 VALU per 4 bytes — see
     * perf-data/plow-gfx942/fp8-dequant-valu-audit.md). CDNA3-only guarantee: gfx950 encodes
     * with the native cvt, which does emit -0. */
    return (unsigned char)(mag ? (sgn | mag) : 0u);
}

/* ==========================================================================================
 * MXFP4 ON CDNA3 — w4a16, in software, and EXACT.
 *
 * This used to be PLOW_MX_PAUSED: gfx942 has no fp4 datatype, no cvt_scalef32 and no scaled
 * matrix core, so the decode returned a bf16 NaN pair rather than a plausible-looking number.
 * That is still the right verdict for A4W4 (PLOW_HAS_MX_MMA, fp4 on BOTH operands through the
 * matrix core) — there is no CDNA3 instruction to lower it to. It is NOT the right verdict for
 * w4a16, where fp4 is only a WEIGHT ENCODING: the weights are dequantized to bf16 before they
 * ever reach a matrix core or a dot, so the only thing CDNA3 lacks is the convert itself, and a
 * convert is arithmetic. Kimi-K3's routed experts are mxfp4, so on gfx942 this is the difference
 * between serving the model and not.
 *
 * THE IDENTITY THAT MAKES IT ONE SHIFT, NOT AN 8-WAY SELECT.
 *
 * e2m1 is `s eeee=ee m`: a 2-bit exponent (bias 1) and a 1-bit mantissa, so its 8 magnitudes are
 * {0, .5, 1, 1.5, 2, 3, 4, 6} — and that ladder is NOT linear in the code, because code 0 and
 * code 1 are SUBNORMALS (value m/2, no implicit leading 1) while codes 2..7 are normals
 * (value 2^(e-1)(1 + m/2)). Every naive decoder pays for that discontinuity: a memory LUT (the
 * previous body — a scratch load in the middle of a GEMV inner loop), or a compare-and-select
 * chain, or SWAR predicates. All of them cost more than the dot product they feed.
 *
 * But IEEE already knows how to evaluate that exact discontinuity — it is the normal/subnormal
 * boundary of any float format. So place e2m1's field bits at the BOTTOM of a wider format's
 * fields and let the hardware decide which side of the boundary they are on:
 *
 *     fp16 (e5m10, bias 15): exp field := e (0..3), mantissa field := m << 9
 *
 *     e == 0 -> exp field 0 -> SUBNORMAL -> (m*2^9)/2^10 * 2^-14 = (m/2) * 2^-14
 *     e >= 1 -> NORMAL                   -> 2^(e-15) * (1 + m/2) = 2^(e-1)(1+m/2) * 2^-14
 *
 * Both arms land on the true e2m1 value times the SAME constant 2^-14 — the subnormal case falls
 * out of the format instead of being special-cased. So the decode is `(code & 7) << 9` for the
 * magnitude and `(code & 8) << 12` for the sign, and the 2^14 is folded into the block scale,
 * which every caller already applies once per 32 elements. Nothing is left over.
 *
 * EXACTNESS. The fp16 encoding above is the value times 2^-14 exactly (it is a bit-construction,
 * not a rounding), v_cvt_f32_f16 of it is exact (fp16 -> f32 always is, subnormals included),
 * the E8M0 block scale is a power of two, and 2^14 is a power of two — so the f32 product carries
 * e2m1's <= 3 significant bits with nothing to round, and the bf16 narrowing at the end is exact
 * for the same reason. This matches CDNA4's native cvt_scalef32_pk_bf16_fp4 VALUE FOR VALUE, and
 * that is the bar: an object built here and an object built for gfx950 must agree bit for bit, or
 * the nibble-order oracle (runtime/tests/k3_mxfp4_nibble_*) cannot arbitrate between them.
 *
 * ONE DIVERGENCE, and it is in the scale rather than the element: `scale * 2^14` overflows f32
 * for E8M0 bytes above 240, i.e. a block scale past 2^113. CDNA4 feeds the byte to the hardware
 * and does not. No quantizer emits it (it would need a weight block with magnitude ~2^113), and
 * the surrounding code already treats a NaN/Inf scale as an upstream bug, so it is recorded here
 * rather than branched on in the inner loop.
 * ========================================================================================== */
typedef _Float16 plow_f16x2 __attribute__((ext_vector_type(2)));

/* 8 fp4 nibbles (one u32) -> 4 PACKED fp16 PAIRS, each element carrying its true value * 2^-14.
 * `h[j]` holds elements 2j (low half) and 2j+1 (high half) — the op_sel order of CDNA4's
 * cvt_scalef32_pk_bf16_fp4, so the callers' lane order is unchanged.
 *
 * WHY THE TWO SHUFFLE WORDS. A packed pair wants its two elements SIXTEEN bits apart; adjacent
 * nibbles are four. Extracting them separately costs two masks and two DIFFERENT shifts per pair
 * (the halves sit at different offsets), i.e. ten VALU per pair. Instead deinterleave once —
 * `lo` takes the even elements into bytes, `hi` the odd ones — and reassemble so that each pair's
 * members land exactly 16 bits apart. Then ONE mask and ONE shift serve BOTH halves, and the
 * per-pair cost is four VALU. The two setup words are amortized over all four pairs. */
__device__ __forceinline__ void plow_fp4x8_to_f16x4(unsigned w, unsigned h[4]) {
    const unsigned lo = w & 0x0F0F0F0Fu;        /* elements 0,2,4,6 -> bytes 0..3 */
    const unsigned hi = (w >> 4) & 0x0F0F0F0Fu; /* elements 1,3,5,7 -> bytes 0..3 */
    /* z0: e0@0 e2@8 e1@16 e3@24   z1: e4@0 e6@8 e5@16 e7@24  — every pair is 16 bits apart. */
    const unsigned z0 = (lo & 0x0000FFFFu) | (hi << 16);
    const unsigned z1 = (lo >> 16) | (hi & 0xFFFF0000u);
    /* mag -> fp16 bits [11:9] (exp field := e, mantissa top := m); sign bit 3 -> bit 15. */
    h[0] = ((z0 & 0x00070007u) << 9) | ((z0 & 0x00080008u) << 12);
    h[1] = ((z0 & 0x07000700u) << 1) | ((z0 & 0x08000800u) << 4);
    h[2] = ((z1 & 0x00070007u) << 9) | ((z1 & 0x00080008u) << 12);
    h[3] = ((z1 & 0x07000700u) << 1) | ((z1 & 0x08000800u) << 4);
}

/* One u32 of 8 fp4 -> 4 packed bf16 pairs (u32 each), scale folded. `out` gets 4 words. */
__device__ __forceinline__ void plow_fp4x8_to_bf16x8(unsigned w, float scale, unsigned out[4]) {
    unsigned h[4];
    plow_fp4x8_to_f16x4(w, h);
    const float s = scale * 16384.0f; /* the 2^-14 the fp16 encoding above carries */
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const plow_f16x2 p = __builtin_bit_cast(plow_f16x2, h[j]);
        out[j] = plow_pack_bf16x2((float)p[0] * s, (float)p[1] * s);
    }
}

#endif /* PLOW_AMD_ARCH_H */
