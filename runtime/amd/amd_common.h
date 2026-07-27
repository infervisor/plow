/* amd_common.h — CDNA (gfx942/gfx950) device primitives shared by every op.
 *
 * bf16 conversion, wave64 reductions, and the MFMA operand types. Nothing here
 * is model-specific and nothing here touches the HIP runtime.
 */
#ifndef PLOW_AMD_COMMON_H
#define PLOW_AMD_COMMON_H

#include <hip/hip_runtime.h>

/* Storage type. We keep bf16 as a bare u16 in memory and convert explicitly, so
 * no ROCm math header is needed and the rounding is ours (round-to-nearest-even,
 * matching torch). __bf16 is used only as an MFMA operand type. */
typedef unsigned short bf16;

typedef __bf16 bf16_t;
typedef bf16_t bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x16 __attribute__((ext_vector_type(16)));

/* ---------------------------------------------------------------------------
 * GLOBAL, AND SAY SO.
 *
 * The interpreter gets its operands out of a POINTER TABLE (`tensors[handle]`), so they
 * arrive as plain `void*` with no provenance whatsoever. The compiler cannot prove such a
 * pointer is in the global aperture -- it might be LDS, it might be scratch -- so every
 * access through it compiles to `flat_*` rather than `global_*`. Measured on the decode
 * interpreter: 141 flat_load_ushort, 31 flat_load_dwordx4, and ZERO global_load_dwordx4.
 * The entire weight stream -- 57 GiB per token -- was going through the generic path.
 *
 * flat is not merely a different encoding. It costs:
 *   - a full 64-bit address per lane in VGPRs (hence the v_lshl_add_u64 storm), instead of
 *     global_load's saddr form (a scalar base + a 32-bit vector offset),
 *   - conservative lgkmcnt waits, because a flat address MIGHT be LDS -- so the weight
 *     stream serialises against LDS traffic that has nothing to do with it,
 *   - and it defeats the 16-byte vectoriser, which is how 16-byte loads decayed into
 *     `flat_load_ushort`: TWO BYTES at a time, in the hottest loop in the model.
 *
 * Address space 1 IS the global aperture on AMDGPU. Tensor pointers come from
 * hsa_memory_allocate, so they are global by construction; the compiler just has no way to
 * know it. as_glob() is where we tell it. */
#define PLOW_GLOB __attribute__((address_space(1)))
template <typename T>
__device__ __forceinline__ PLOW_GLOB T* as_glob(T* p) {
    return (PLOW_GLOB T*)(PLOW_GLOB void*)p;
}
template <typename T>
__device__ __forceinline__ const PLOW_GLOB T* as_glob(const T* p) {
    return (const PLOW_GLOB T*)(const PLOW_GLOB void*)p;
}

/* ...AND ALIGNED, AND SAY THAT TOO.
 *
 * `bf16` is `unsigned short`, so a `const bf16*` has alignment 2 -- and
 * `__builtin_memcpy(dst, src, 16)` off an align-2 pointer is compiled EXACTLY as written:
 * the compiler may not assume more than it is told, so it emits eight 2-byte loads. That is
 * the other half of the flat_load_ushort story, and it is why address space alone does not
 * fix it. A 16-byte vector type carries both facts in its own type.
 *
 * The 16-byte alignment is real, not a hope: every tensor base comes from
 * hsa_memory_allocate (4 KiB), and every K in the model is a multiple of 8 -- which the
 * 8-wide loads here already require anyway, since the tail reads a full 16 bytes at the
 * last k. */
typedef bf16 bf16v8 __attribute__((ext_vector_type(8))); /* 16 B, align 16 */

__device__ __forceinline__ bf16v8 ld_glob8(const bf16* p) {
    return *(const PLOW_GLOB bf16v8*)(const PLOW_GLOB void*)p;
}
__device__ __forceinline__ bf16v8 ld_glob8(const PLOW_GLOB bf16* p) {
    return *(const PLOW_GLOB bf16v8*)(const PLOW_GLOB void*)p;
}
__device__ __forceinline__ bf16v8 ld_lds8(const bf16* p) { return *(const bf16v8*)p; }
/* NON-TEMPORAL. The weight stream has ZERO reuse -- 57 GiB read exactly once per token --
 * yet every line of it allocates in L2 and evicts the activations that ARE reused. `nt` tells
 * the cache not to keep it. */
__device__ __forceinline__ bf16v8 ld_glob8_nt(const PLOW_GLOB bf16* p) {
    return __builtin_nontemporal_load((const PLOW_GLOB bf16v8*)(const PLOW_GLOB void*)p);
}

__device__ __forceinline__ void st_glob8(bf16* p, bf16v8 v) {
    *(PLOW_GLOB bf16v8*)(PLOW_GLOB void*)p = v;
}
__device__ __forceinline__ void st_glob8(PLOW_GLOB bf16* p, bf16v8 v) {
    *(PLOW_GLOB bf16v8*)(PLOW_GLOB void*)p = v;
}
__device__ __forceinline__ bf16v8 bf16v8_zero(void) {
    bf16v8 z;
#pragma unroll
    for (int j = 0; j < 8; j++) z[j] = 0;
    return z;
}

/* ---------------------------------------------------------------------------
 * ASYNC HBM -> LDS. This is CDNA's cp.async, and it is what lets a gate stop the CU without
 * stopping the memory system.
 *
 * A packet's operands split in two: ACTIVATIONS, which a predecessor must produce, and
 * CONSTANTS -- weights, RoPE tables, all but the newest row of the KV cache -- which have no
 * producer at all and have been sitting unchanged in HBM since the model loaded. There is
 * nothing to wait for. So the interpreter issues the constant loads BEFORE the gate and lets
 * them fly while the CU spins: measured, every GEMV gate has ~11 us of idle behind it and the
 * memory system does nothing for all of it.
 *
 * global_load_lds writes straight into LDS with NO VGPRs, which is the only reason this is
 * affordable -- decode is at 223 of 256 registers, so a register-held prefetch is not on the
 * table. It is tracked on vmcnt, so the loads survive s_barrier and we wait for them once, on
 * the far side of the gate.
 *
 * CONTRACT, and it is sharp: the LDS destination comes from M0, which is UNIFORM. The
 * compiler emits `v_readfirstlane` on whatever LDS pointer you pass and the hardware then
 * lays the 64 lanes down CONTIGUOUSLY from there (lane l -> M0 + l*16). The global source may
 * be per-lane; the LDS destination may NOT. Hand it a non-contiguous LDS mapping and it will
 * quietly write somewhere else. */
/* The __HIP_DEVICE_COMPILE__ guard is not cosmetic: hipcc parses every __device__ body in the
 * HOST pass too, and this builtin does not exist there -- it fails with "invalid size value",
 * which reads like a bad argument and is nothing of the sort. */
__device__ __forceinline__ void cp_async16(const PLOW_GLOB bf16* src,
                                           bf16* dst_lane_contiguous) {
#ifdef __HIP_DEVICE_COMPILE__
    __builtin_amdgcn_global_load_lds(
        (const PLOW_GLOB unsigned*)(const PLOW_GLOB void*)src,
        (__attribute__((address_space(3))) unsigned*)(__attribute__((address_space(3))) void*)
            dst_lane_contiguous,
        16 /* bytes per lane */, 0 /* offset */, 0 /* aux */);
#else
    (void)src;
    (void)dst_lane_contiguous;
#endif
}
__device__ __forceinline__ void cp_async_wait(void) {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

/* ---------------------------------------------------------------------------
 * BUFFER LOADS: a hardware bounds check that is FREE, and the only clean way to unroll a loop
 * whose trip count is ragged.
 *
 * A buffer load addresses memory through a 128-bit resource descriptor carrying `num_records`.
 * A load past the end returns ZERO and — this is the part that matters — **issues no memory
 * request at all**. Overshoot costs an instruction slot and NOTHING in bandwidth.
 *
 * That is what makes it different from predication, which this file already tried and reverted:
 * clamping an out-of-range offset to 0 still FETCHES (16.9 -> 20.6 ms/token, gemv 52 -> 70
 * us/inst). The hardware guard fetches nothing.
 *
 * The payoff is the GEMV's ragged tail. The k-loop advances by GV_UNROLL*512 halves, and whatever
 * K leaves over used to run in a SCALAR loop — one load, wait, consume, repeat. At Gemma's
 * K = 5376 that was 24% of every row with no memory-level parallelism, and it also BROKE the
 * compiler's software pipeline: it pipelines ~20 loads deep across the main loop and then hits a
 * dependent scalar loop. With a buffer load there is no tail at all — the row is one uniform
 * unrolled loop and the overshoot is free.
 *
 * Measured standalone on the real projections (scalar tail -> buffer_load, no tail):
 *     gate/up 21504x5376   3510 -> 4090 GB/s   (+16.5%)
 *     q_proj   8192x5376   3248 -> 3564        ( +9.7%)
 *     o_proj   5376x8192   5342 -> 5448        ( +2.0%)   <- K already divided the unroll
 *     down    5376x21504   6012 -> 6119        ( +1.8%)   <- ditto
 * The gain tracks the tail fraction exactly.
 *
 * THE DESCRIPTOR'S WORD 3 IS NOT OPTIONAL. On gfx9/CDNA it must be 0x00020000 (this is
 * CK's CK_BUFFER_RESOURCE_3RD_DWORD). Passing 0 compiles, runs, faults nothing — and returns
 * ZERO FOR EVERY LANE, in range or not. It fails as silently as anything in this codebase. */
#define PLOW_BUF_RSRC3 0x00020000u

/* `aux` = 3 is glc|slc: slc is the streaming / non-temporal hint. The weight stream has zero
 * reuse, so it must not allocate in L2 — the same reason GV_LDW uses ld_glob8_nt. */
__device__ __forceinline__ bf16v8 buf_ld8(__amdgpu_buffer_rsrc_t r, unsigned byte_off) {
    return __builtin_bit_cast(bf16v8,
                              __builtin_amdgcn_raw_buffer_load_b128(r, byte_off, 0, /*nt*/ 3));
}
__device__ __forceinline__ __amdgpu_buffer_rsrc_t buf_rsrc(const bf16* base, unsigned n_halves) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, /*stride*/ (short)0,
                                             /*num_records, BYTES*/ n_halves * 2u, PLOW_BUF_RSRC3);
}

/* ---------------------------------------------------------------------------
 * FP8 (OCP e4m3) WEIGHT LOADS — the decode w8a16 path.
 *
 * The fp8 weight row is uint8[K], so num_records is K BYTES (not K*2). The b128 buffer load then
 * pulls 16 fp8 per lane instead of 8 bf16 — HALF the bytes per output element, which is the whole
 * point: decode is weight-bandwidth-bound, so fp8 weights ~= 2x the roofline.
 *
 * gfx950 (CDNA4) has NATIVE OCP e4m3. cvt_pk_f32_fp8(word, sel) converts a packed pair of fp8 to
 * two f32 — sel=false picks bytes [0,1] of the u32, sel=true bytes [2,3] (verified: 0x38 -> 1.0,
 * 0x40 -> 2.0, 0x30 -> 0.5, 0xB8 -> -1.0, matching torch.float8_e4m3fn). The decoded fp8 is exact
 * in bf16 (fp8 e4m3 has 3 mantissa bits, bf16 has 7), so f2bf() then feeds the SAME fdot2 `dot8`
 * and the same wave reduction as the bf16 GEMV; the per-channel dequant scale is applied ONCE in
 * the epilogue. (cvt_scalef32_pk_bf16_fp8 was tried and its word_sel does NOT pick adjacent byte
 * pairs — it broadcasts one byte — so it is unusable here.) */
typedef unsigned fp8v16 __attribute__((ext_vector_type(4))); /* 16 fp8 == 4 u32, 16 B, align 16 */
typedef int fp8v32 __attribute__((ext_vector_type(8)));      /* 32 fp8 == 8 i32, 32 B — the K64 MX-fp8 MFMA operand fragment */

__device__ __forceinline__ __amdgpu_buffer_rsrc_t buf_rsrc_fp8(const unsigned char* base,
                                                               unsigned n_bytes) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, /*stride*/ (short)0,
                                             /*num_records, BYTES*/ n_bytes, PLOW_BUF_RSRC3);
}
/* PROBE: force a wave-uniform base pointer into SGPRs so make_buffer_rsrc emits a SCALAR resource
 * (no per-load readfirstlane waterfall). Valid ONLY when `base` is genuinely uniform across the wave
 * (it is in the GEMV: base = W + n*K, n = wave index, uniform per wave). */
__device__ __forceinline__ __amdgpu_buffer_rsrc_t buf_rsrc_fp8_u(const unsigned char* base,
                                                                 unsigned n_bytes) {
    unsigned long long u = (unsigned long long)(size_t)base;
    unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)u);
    unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(u >> 32));
    const unsigned char* ub = (const unsigned char*)(size_t)(((unsigned long long)hi << 32) | lo);
    return __builtin_amdgcn_make_buffer_rsrc((void*)ub, (short)0, n_bytes, PLOW_BUF_RSRC3);
}
/* `aux` = 3 (glc|slc): streaming/non-temporal, exactly as the bf16 weight stream — zero reuse. */
__device__ __forceinline__ fp8v16 buf_ld_fp8(__amdgpu_buffer_rsrc_t r, unsigned byte_off) {
    return __builtin_bit_cast(fp8v16,
                              __builtin_amdgcn_raw_buffer_load_b128(r, byte_off, 0, /*nt*/ 3));
}

/* Wait until at most D vector-memory ops are still outstanding.
 *
 * This is the THROTTLE, and it is the whole difference between a prefetch that helps and one
 * that hurts. The window a mover runs in is genuinely idle -- the two norms in front of a
 * q_proj move 40 KB in 6 us, i.e. 7 GB/s of a 6200 GB/s machine -- but those norms are
 * LATENCY-bound: one CU, one row, one HBM round trip. Issuing the whole prefetch as a burst
 * puts 16 MB (256 CUs x 64 KB) into the memory controller ahead of their 10 KB and their round
 * trip goes from ~1.3 us to ~2 (measured: rmsnorm +18%, norm_residual +41%).
 *
 * Capping outstanding requests caps the queue the critical op sits behind. A CU sustains
 * roughly D KB / 1.3 us, so D is a direct rate knob: D=14 is ~10.7 GB/s per CU, enough to move
 * 64 KB across a 6 us window. */
template <int D>
__device__ __forceinline__ void cp_async_depth(void) {
    asm volatile("s_waitcnt vmcnt(%0)" ::"i"(D) : "memory");
}

/* An 8-wide bf16 dot product in FOUR instructions.
 *
 * gfx950 has v_dot2c_f32_bf16: two bf16 lanes multiplied and accumulated into an f32, as one
 * VALU op, with NO conversion. The obvious C -- `acc += bf2f(w[j]) * bf2f(x[j])` -- instead
 * costs a shift per operand plus an FMA, i.e. 24 VALU ops per 16 bytes rather than 4.
 *
 * This matters because the GEMV moves 57 GiB per token and every byte passes through here. */
typedef bf16_t bf16x2 __attribute__((ext_vector_type(2)));
union bf16v8_pairs {
    bf16v8 v;
    bf16x2 p[4];
};

__device__ __forceinline__ float dot8(bf16v8 a, bf16v8 b, float acc) {
    bf16v8_pairs x{a}, y{b};
#pragma unroll
    for (int j = 0; j < 4; j++) acc = __builtin_amdgcn_fdot2_f32_bf16(x.p[j], y.p[j], acc, false);
    return acc;
}

__device__ __forceinline__ float bf2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

__device__ __forceinline__ bf16 f2bf(float f) {
    unsigned u;
    __builtin_memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u); /* qNaN */
    u += 0x7fffu + ((u >> 16) & 1u);                                          /* RNE */
    return (bf16)(u >> 16);
}

/* 16 fp8 (a b128 load) -> two bf16v8, in fdot2-ready lane order. Each u32 word holds 4 fp8.
 *
 * ONE INSTRUCTION per fp8 PAIR: gfx950's v_cvt_scalef32_pk_bf16_fp8 decodes a packed fp8 pair
 * straight to a bf16 pair (with an f32 scale; passed 1.0 — the per-channel w_scale stays a single
 * epilogue multiply). sel=false picks bytes [0,1] of the u32, sel=true bytes [2,3] — verified on
 * gfx950 (word 0xB8304038 -> 1.0, 2.0, 0.5, -1.0), the SAME byte order as cvt_pk_f32_fp8. This
 * replaces the old two-step decode (8x cvt_pk_f32_fp8 to f32 + 16 high-half truncations = ~24 VALU
 * per fp8v16) with 8 packed converts — ~3x fewer dequant VALU. NOTE: this is a cleaner/cheaper
 * kernel but PERF-NEUTRAL at bs=1 decode: measured the fp8 GEMV op-work is unchanged (gemv_fp8
 * 253 vs 251 us, gemv_glu_fp8 152 vs 152), because the dequant already hides fully under the fp8
 * weight-load latency — the decode GEMV op is memory-latency-bound (per-wave outstanding-load
 * limited), not dequant-bound. Kept because it is the correct native instruction and frees VALU
 * headroom. Bit-exact to the old truncation: fp8 e4m3 -> bf16 is lossless (<=3 mantissa bits) and
 * the instruction is the native round. Words 0,1 fill `lo` (elems 0..7), words 2,3 fill `hi` —
 * matching x[kx..kx+8] and x[kx+8..kx+16] in the fp8 GEMV.
 *
 * (The earlier note that this builtin "broadcasts one byte" was wrong — see the byte-order check
 * above; it is a drop-in for the two-step path.) */
__device__ __forceinline__ void fp8_to_bf16v8(fp8v16 w, bf16v8& lo, bf16v8& hi) {
    typedef bf16_t bf16_2 __attribute__((ext_vector_type(2)));
    union { bf16v8 v; unsigned u[4]; } ol, oh; /* pack pairs as u32 to avoid per-lane extraction */
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const bf16_2 a = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w[i], 1.0f, false); /* bytes 0,1 */
        const bf16_2 c = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w[i], 1.0f, true);  /* bytes 2,3 */
        auto& o = (i < 2) ? ol : oh;
        const int b = (i & 1) * 2; /* two u32 (= 4 bf16) per word */
        o.u[b + 0] = __builtin_bit_cast(unsigned, a); /* a0 in low16, a1 in high16 */
        o.u[b + 1] = __builtin_bit_cast(unsigned, c);
    }
    lo = ol.v;
    hi = oh.v;
}

/* WHY scale=1.0 above and NOT the block scale: v_cvt_scalef32_pk_bf16_fp8's scalef32 operand is an
 * E8M0 (power-of-2) MICROSCALING scale — the hardware uses ONLY the f32 exponent (2^floor(log2 s))
 * and DISCARDS the mantissa. Probed on gfx950 (2026-07-17): powers of two (0.5, 2, 2^-6) fold
 * exactly, but 0.01 folds as 2^-7 = 0.0078 (~22% low). That is correct for MX/mxfp8 (E8M0 block
 * scales, which is why AITER always feeds it e8m0_to_f32(byte)=2^(byte-127)), but DeepSeek/GLM
 * block-fp8 (weight_block_size [128,128]) carries ARBITRARY f32 weight_scale_inv. So the block scale
 * must stay a SEPARATE f32 multiply after the exact fp8->bf16 decode (gemv_rows_fp8_blk /
 * wave_dot_fp8_blk apply it per 128-K block); it cannot be folded into this cvt. */

/* ---------------------------------------------------------------------------
 * MXFP4 (OCP microscaling: e2m1 element + one shared E8M0 scale per 32) — weight decode.
 *
 * This is the case the fp8 block-scale comment above rules OUT for GLM/DeepSeek and IN for MX: an
 * MX scale IS a power of two by construction (E8M0 is a bare 8-bit exponent), so folding it into
 * the cvt's scalef32 operand is EXACT, not an approximation. The mantissa the hardware discards is
 * a mantissa the format does not have. So mxfp4 dequant costs the same VALU as fp8 dequant and the
 * scale is free — no separate multiply, no per-block FMA, nothing in the epilogue.
 *
 * THE ALIGNMENT THAT MAKES THIS CHEAP: a lane's b128 weight load is 16 bytes = 32 fp4 = EXACTLY one
 * MX block. So one load consumes one scale byte, the scale never varies within a lane's fragment,
 * and there is no cross-lane scale reshuffle (contrast gemv_rows_fp8_blk, which has to reason about
 * 16-element partials landing inside a 128-K block). Pick any other fragment width and this
 * property is lost. */
typedef unsigned fp4v32 __attribute__((ext_vector_type(4))); /* 32 fp4 == 4 u32, 16 B, align 16 */

/* E8M0 byte -> f32 2^(b-127). Bit-construct rather than exp2f: byte b placed in the f32 exponent
 * field with a zero mantissa IS 2^(b-127), so this is one shift, no transcendental. b=0xFF is the
 * MX NaN encoding; it is not special-cased here because the cvt consumes only the exponent and a
 * NaN scale would poison the block anyway (quantisers do not emit it for weights). */
__device__ __forceinline__ float e8m0_to_f32(unsigned char b) {
    return __builtin_bit_cast(float, (unsigned)b << 23);
}

/* 32 fp4 + one E8M0 scale -> four bf16v8, in the SAME fdot2-ready lane order as fp8_to_bf16v8, so
 * the GEMV reduction, `dot8` and the wave reduction are shared verbatim with the fp8 path.
 * Each u32 holds 8 fp4 = 4 pairs; op_sel (the third operand, 0..3) picks the pair, so one word
 * becomes 8 bf16 in 4 packed converts — 16 converts per 32-element fragment. */
__device__ __forceinline__ void fp4_to_bf16v8x4(fp4v32 w, float scale, bf16v8& a, bf16v8& b,
                                                bf16v8& c, bf16v8& d) {
    typedef bf16_t bf16_2 __attribute__((ext_vector_type(2)));
    union { bf16v8 v; unsigned u[4]; } o[4];
    /* op_sel must be a literal (clang rejects a loop induction variable even under #pragma unroll,
     * the check runs before unrolling), so the pair select is spelled out. */
#define MXFP4_CVT(i)                                                                          \
    o[i].u[0] = __builtin_bit_cast(                                                            \
        unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w[i], scale, 0));             \
    o[i].u[1] = __builtin_bit_cast(                                                            \
        unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w[i], scale, 1));             \
    o[i].u[2] = __builtin_bit_cast(                                                            \
        unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w[i], scale, 2));             \
    o[i].u[3] = __builtin_bit_cast(                                                            \
        unsigned, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w[i], scale, 3));
    MXFP4_CVT(0)
    MXFP4_CVT(1)
    MXFP4_CVT(2)
    MXFP4_CVT(3)
#undef MXFP4_CVT
    a = o[0].v;
    b = o[1].v;
    c = o[2].v;
    d = o[3].v;
}

/* One u32 (8 fp4) + one E8M0 scale -> bf16v8, the per-word slice of fp4_to_bf16v8x4. Used by the
 * w4a16 prefill GEMM's dequant-on-load B-fetch, where the 8-half load granularity wants exactly 8
 * bf16 at a time (an 8-element load never crosses a 32-element MX block, so one scale byte covers
 * it). Same 4 packed op_sel converts, same fdot2-ready order — the scale fold stays EXACT. */
__device__ __forceinline__ bf16v8 fp4_to_bf16v8(unsigned w, float scale) {
    union { bf16v8 v; unsigned u[4]; } o;
    o.u[0] = __builtin_bit_cast(unsigned,
                                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w, scale, 0));
    o.u[1] = __builtin_bit_cast(unsigned,
                                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w, scale, 1));
    o.u[2] = __builtin_bit_cast(unsigned,
                                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w, scale, 2));
    o.u[3] = __builtin_bit_cast(unsigned,
                                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4((int)w, scale, 3));
    return o.v;
}

/* ---------------------------------------------------------------------------
 * FP8 (OCP e4m3) KV-CACHE loads/stores — the decode flash-attention path.
 *
 * The KV cache is HBM-bound in decode (op_attention.h: "1.91 ms @ 2.0 TB/s"), so storing K/V as
 * e4m3 halves the KV stream — the same 2x roofline the fp8 GEMV gets. These mirror the fp8 weight
 * helpers above: the DECODE (fp8 -> bf16) side reuses fp8_to_bf16v8; this block adds the two things
 * the KV path needs that the weight path does not — an 8-wide (b64) decode for the V phase, which
 * reads only 8 head-dims per lane, and an ENCODE (f32 -> e4m3) for the HeadNormRope write. */

/* A whole K row is a b128 (16 fp8) load, exactly like the fp8 weight stream. */
__device__ __forceinline__ fp8v16 ld_glob_fp8v16(const unsigned char* p) {
    return *(const PLOW_GLOB fp8v16*)(const PLOW_GLOB void*)p;
}

/* 8 fp8 == 2 u32 == 8 B: the V phase reads only its 8 owned head-dims per lane, so it wants HALF
 * the b128 width — a b64 load, then the same cvt_pk_f32_fp8 truncation as fp8_to_bf16v8. */
typedef unsigned fp8v8 __attribute__((ext_vector_type(2))); /* 8 fp8, 8 B, align 8 */
__device__ __forceinline__ fp8v8 ld_glob_fp8v8(const unsigned char* p) {
    return *(const PLOW_GLOB fp8v8*)(const PLOW_GLOB void*)p;
}
__device__ __forceinline__ bf16v8 fp8v8_to_bf16v8(fp8v8 w) {
    typedef float f32x2 __attribute__((ext_vector_type(2)));
    auto trunc = [](float f) -> bf16 {
        unsigned u;
        __builtin_memcpy(&u, &f, 4);
        return (bf16)(u >> 16); /* e4m3 -> f32 has zero low 16 mantissa bits, so this is EXACT */
    };
    bf16v8 d;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8(w[i], false); /* bytes 0,1 */
        const f32x2 c = __builtin_amdgcn_cvt_pk_f32_fp8(w[i], true);  /* bytes 2,3 */
        d[i * 4 + 0] = trunc(a[0]);
        d[i * 4 + 1] = trunc(a[1]);
        d[i * 4 + 2] = trunc(c[0]);
        d[i * 4 + 3] = trunc(c[1]);
    }
    return d;
}

/* f32 -> one e4m3 byte. gfx950 native: cvt_pk_fp8_f32 packs (a,b) into two bytes of `old`; we use
 * one and keep byte 0. RNE + saturation are in hardware. */
__device__ __forceinline__ unsigned char quant_fp8(float v) {
    const unsigned packed = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false);
    return (unsigned char)(packed & 0xffu);
}

/* Reduce a MAX across the 64 lanes of a wave (the HeadNormRope per-row amax for the KV scale). */
__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 64));
    return v;
}

/* e4m3 max finite magnitude (torch.float8_e4m3fn): 448. A row's scale maps its amax to 448 so the
 * largest element uses the full e4m3 range; the reciprocal is what the write multiplies by. */
#define PLOW_FP8_E4M3_MAX 448.0f

/* Wave is 64 lanes on CDNA. __shfl_sync/warpSize-32 assumptions from CUDA code
 * do not port — this is the reason the NVIDIA RoPE reference cannot be
 * transliterated (it pairs lane i with lane i+32). */
#include "../common/dev_isa.h"

#define PLOW_WAVE 64

/* Workgroup shape for the persistent interpreter: ONE workgroup per CU, 4 waves.
 *
 * A persistent interpreter owns the CU (co-residency is what makes the counter
 * spin safe), so it has NO occupancy to hide HBM latency with: when a wave stalls
 * on a global load, a 4-wave workgroup (1 wave/SIMD) has nothing to switch to.
 * That suggested 8 waves (2/SIMD) would be a large win, and in an isolated GEMM
 * it is a real one — but a small one, and it is NOT the main lever:
 *
 *   gate/up_proj (4096 x 21504 x 5376), full GPU, vs the 2308 TF/s MFMA peak:
 *     4 waves, duplicated fetch path (spilling)    230 TF/s
 *     8 waves, duplicated fetch path (spilling)    279
 *     8 waves, single fetch path                   449
 *     4 waves, single fetch path                   405-502   <- current
 *
 * The dominant factor was REGISTER PRESSURE, not wave count. A duplicated
 * "interior tile" / "edge tile" fetch path doubled the live set and spilled ~1 KB
 * per lane into scratch (i.e. HBM) inside the mainloop; collapsing it to one
 * predicated path was worth more than the extra waves.
 *
 * We stay at 4 waves because the interpreter inlines every op, so its register
 * allocation is the worst case over all of them — and flash at head_dim=512 holds
 * Q as MFMA fragments (D/2 halves = 128 VGPRs) plus the O accumulator, pushing the
 * kernel to 384 arch VGPRs. At 512 threads a wave may use at most 256 (two waves
 * must be resident per SIMD), so an 8-wave interpreter is REJECTED AT DISPATCH
 * with HSA_STATUS_ERROR_INVALID_ISA — which reads like a code-object problem and
 * is nothing of the sort. Empirically 256 launches, 384 does not.
 *
 * To unlock 8 waves for the interpreter, flash must stop holding all D/16 Q
 * fragments live: accumulate S over D-chunks so only FA_DC/16 are live at once.
 *
 * NOTE on __launch_bounds__: on AMD the second argument is MIN WAVES PER EU, not
 * min blocks per CU as in CUDA. Passing 1 lets the compiler take all 512
 * registers. Kernels use __launch_bounds__(PLOW_THREADS, PLOW_WAVES_PER_EU).
 *
 * Every op assumes PLOW_THREADS, never a hardcoded 256. */
#define PLOW_WAVES PLOW_WG_WAVES               /* from dev_isa.h: host + device agree */
#define PLOW_THREADS PLOW_WG_THREADS           /* 512: 8 waves, 2 per SIMD          */
#define PLOW_WAVES_PER_EU (PLOW_WAVES / 4)     /* 4 SIMDs per CU */

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, PLOW_WAVE);
    return v;
}

/* Reduce across the 32 lanes of a HALF-wave. The MFMA 32x32 accumulator layout
 * puts one output row entirely inside one half-wave (lanes 0-31 or 32-63), so a
 * row-wise softmax reduction must stop at 32 — going to 64 would fold two
 * different rows together. */
__device__ __forceinline__ float half_wave_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, PLOW_WAVE));
    return v;
}
__device__ __forceinline__ float half_wave_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, PLOW_WAVE);
    return v;
}

/* Block reduction over the PLOW_WAVES waves of the workgroup. */
__device__ __forceinline__ float block_sum(float v, float* part) {
    v = wave_sum(v);
    const unsigned wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    if (lane == 0) part[wave] = v;
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < PLOW_WAVES; i++) t += part[i];
    __syncthreads(); /* `part` is reused by the next op in the interpreter loop */
    return t;
}

/* ---------------------------------------------------------------------------
 * v_mfma_f32_32x32x16_bf16 register layout.
 *
 * CONFIRMED empirically on gfx950 against a CPU reference (0/1024 elements
 * wrong). Do not "correct" these from memory — a wrong lane map yields a GEMM
 * that is almost right, which is the worst possible failure mode.
 *
 *   A[32][16] : lane l holds  m = l%32,  k = 8*(l/32) + j        (j = 0..7)
 *   B[16][32] : lane l holds  n = l%32,  k = 8*(l/32) + j        (j = 0..7)
 *   D[32][32] : lane l holds  n = l%32,  m = 4*(l/32) + (i%4) + 8*(i/4)  (i = 0..15)
 *
 * Both operands want k contiguous, which is why every LDS tile below is stored
 * row-major with K innermost: one 16-byte LDS read fills a fragment.
 * ------------------------------------------------------------------------- */
#define MFMA_M 32
#define MFMA_N 32
#define MFMA_K 16

/* k-octet this lane supplies for an A/B fragment at k-step `kk`. */
__device__ __forceinline__ unsigned mfma_frag_k(unsigned lane, unsigned kk) {
    return kk + 8 * (lane / 32);
}
/* row (m for A, n for B) this lane supplies. */
__device__ __forceinline__ unsigned mfma_frag_row(unsigned lane) { return lane % 32; }
/* m of accumulator element `i` for this lane. */
__device__ __forceinline__ unsigned mfma_acc_m(unsigned lane, unsigned i) {
    return 4 * (lane / 32) + (i % 4) + 8 * (i / 4);
}
/* n of every accumulator element for this lane. */
__device__ __forceinline__ unsigned mfma_acc_n(unsigned lane) { return lane % 32; }

#endif /* PLOW_AMD_COMMON_H */
