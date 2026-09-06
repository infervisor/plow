#ifndef PLOW_NV_PACKED_REQUEST
#define PLOW_NV_PACKED_REQUEST 0
#endif
#include "mixed_step.h"
/* sm_120 (RTX 5090 / GB202) flash DECODE + MERGE.
 *
 * Port of runtime/amd/op_attention.h d_flash_decode / d_flash_merge to a 32-lane warp.
 * The AMD original is MFMA-free, so this is NOT a tensor-core port: the only structural
 * changes are
 *   (a) workgroup 512 threads / 8 waves of 64  ->  256 threads / 8 warps of 32,
 *   (b) every wave64 reduction (__shfl_xor off=32.. , wave_sum) RE-DERIVED for 32 lanes,
 *   (c) FA_DEC_TILE follows the thread count (one KV row per thread), so the tile is 256
 *       rows here instead of 512 and NG (row-groups) halves with it.
 *
 * The (Opart f32, mlpart f32) split-KV contract is UNCHANGED, so d_flash_merge below is the
 * exact partner: decode writes per-split (O_partial, m, l) in log2 domain, merge folds them.
 *
 * NOT ported: the fp8-KV arm (FP8KV) and the FA_DEC_VPIPE A/B path. There is no fp8 code in
 * runtime/nvidia at all; adding it here would be untested surface.
 *
 * OPERAND CONTRACT (read off the EMITTER crates/plowc/src/bin/gemma4.rs:1070-1080 and the
 * consumer runtime/amd/interp.hip, NOT crates/packet/src/dev.rs which is stale):
 *   FLASH_DECODE  t0=Opart(f32)  t1=mlpart(f32)  t2=Q(bf16)  t3=K  t4=V  t5=kv_len(i32)
 *                 t6=decode_slot (bf16 only; TENSOR_NONE means compact b == physical slot)
 *                 t6=k_scale     t7=v_scale                (fp8 only)
 *                 i0=n_batch i1=n_head i2=n_kv_head i3=kv_stride i4=window
 *                 i5=nsplit  i6=head_dim i7=kv_mask        <-- i7 IS kv_mask (dev.rs omits it)
 *                 f0=scale
 *   FLASH_MERGE   t0=O(bf16) t1=Opart t2=mlpart
 *                 i0=n_batch i1=n_head i2=nsplit i3=head_dim
 * Opart is [b][head][split][D] f32; mlpart is [b][head][split][2] f32 = (m, l).
 * K/V are HEAD-MAJOR: ((b*n_kv_head + hkv)*kv_stride + row)*D.  kv_mask is the RING mask
 * (0xFFFFFFFF on a full-attention layer, kv_stride-1 on a sliding one).
 */
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define PLOW_NV_THREADS 256u
#define PLOW_NV_WARPS 8u
#define FA_NEG_INF (-3.0e38f)
/* Scores are carried in LOG2 domain: fold log2(e) into the softmax scale once, then every
 * exp() is a hardware exp2. m/l and therefore mlpart are log2-domain too, so d_flash_merge
 * MUST use the same pair. (Same trick as the AMD FA_EXP/FA_SCALE macros.) */
/* Raw hardware exp2 (one MUFU.EX2), not exp2f: exp2f adds a 3-instr guard (FSETP + 2
 * predicated FMUL) that only produces SUBNORMAL results for x in (-149,-126). A softmax
 * weight below 2^-126 is zero for all practical purposes; ftz flushing it to 0 removes at
 * most a <=1e-38 term, and every normal-range result is the same MUFU.EX2 bits. */
__device__ __forceinline__ float __fa_ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
#define FA_EXP(x) __fa_ex2(x)
#define FA_SCALE(s) ((s) * 1.4426950408889634f)

/* One KV row per thread per pass. */
#define FA_DEC_TILE ((int)PLOW_NV_THREADS)

/* K-stream pre-issue depth for d_flash_decode (H100 campaign round 6). 1 = the original
 * consume-immediately loop, byte-identical. See perf-data/gemma26b-h100-gemv-mlp.md. */
#ifndef PLOW_NV_FA_KUN
#define PLOW_NV_FA_KUN 1
#endif
/* Warp-per-row flash score phase. 0 = the original one-row-per-thread body, byte-identical. */
#ifndef PLOW_NV_FA_WPR
#define PLOW_NV_FA_WPR 0
#endif
/* Skip staging Q into smem and read it from global instead (WPR path only). Q is GF*D bf16 --
 * 1 KiB at GF=2,D=256 -- and L2-resident, while the staging costs a full __syncthreads on
 * EVERY work item. At nsplit=32 an item owns only 32 of the 256 tile rows, so that fixed cost
 * is a large share of it: flash runs at 640 GB/s where the occ-2 ceiling is 3269. 0 = stage. */
#ifndef PLOW_NV_FA_QGLOB
#define PLOW_NV_FA_QGLOB 0
#endif
#ifndef PLOW_NV_FA_QREG
#define PLOW_NV_FA_QREG 0
#endif
/* Rows a warp carries CONCURRENTLY in the warp-per-row score phase. With nsplit=32 a work item
 * owns ~32 of the 256 tile rows, so each of the 8 warps gets ~4 -- and processing them one at a
 * time leaves ~1 load in flight, which is why flash measured 688 GB/s against a 3269 ceiling
 * while the arm is latency-bound rather than bandwidth-bound. Batching them puts RB independent
 * row loads in flight for one warp reduction each. 1 = the original sequential sweep. */
#ifndef PLOW_NV_FA_WPR_RB
#define PLOW_NV_FA_WPR_RB 1
#endif
/* Bound the two softmax reductions to the tile's LIVE rows. Entries past rmax_t are NEG_INF in
 * both score bodies, so scanning them is pure waste -- and at nsplit=32 a work item owns ~32 of
 * the 256 tile rows, so 7 of every 8 iterations reduce padding. 0 = sweep the whole tile. */
#ifndef PLOW_NV_FA_REDBOUND
#define PLOW_NV_FA_REDBOUND 0
#endif
/* Thread map for the V phase / O accumulator: 8 consecutive head-dims per thread, so both K
 * and V move 16 bytes per lane contiguously (one 128-bit global load), and the block's rows
 * are split across NG row-groups whose partial O is folded once at the end. */
#define FA_DEC_NDT(D) ((D) / 8)                          /* threads covering one row: 16 at D=128 */
#define FA_DEC_NG(D) ((int)PLOW_NV_THREADS / FA_DEC_NDT(D)) /* row-groups: 16 at D=128 */
/* smem floats: Ssm[GF][TILE] + hmax[WARPS] + hsum[WARPS] + qsm[GF][D] bf16 + osm[NG][D].
 * D=128,GF=2 -> 512 + 16 + 128 + 2048 floats = 10.6 KiB, well inside the 48 KiB default. */
#ifndef PLOW_NV_FA_GF16_BENCH
#define PLOW_NV_FA_GF16_BENCH 0
#endif
#if PLOW_NV_FA_GF16_BENCH
#define FA_DEC_REDUCTION_HEADS(GF) ((GF) > (int)PLOW_NV_WARPS ? (GF) : (int)PLOW_NV_WARPS)
#else
#define FA_DEC_REDUCTION_HEADS(GF) ((int)PLOW_NV_WARPS)
#endif
#ifndef PLOW_NV_FA_TC_GQA8_HD512
#define PLOW_NV_FA_TC_GQA8_HD512 0
#endif
/* Opt-in HD512/GQA8 decode candidate: 16 padded Q rows and a 64-row K/V staging tile. */
#define FA_DEC_TC_GQA8(HD, GF) (PLOW_NV_FA_TC_GQA8_HD512 && (HD) == 512 && (GF) == 8)
#define FA_DEC_BASE_SMEM_FLOATS(D, GF)                                                         \
    ((GF) * FA_DEC_TILE + 2 * FA_DEC_REDUCTION_HEADS(GF) + (GF) * ((D) / 2) + FA_DEC_NG(D) * (D))
#define FA_DEC_SMEM_FLOATS(D, GF)                                                              \
    (FA_DEC_BASE_SMEM_FLOATS(D, GF) +                                                         \
     (FA_DEC_TC_GQA8(D, GF) ? (16 + 64) * ((D) + 8) / 2 : 0))

/* V rows in flight per thread. A fused row feeds GF accumulators, so arithmetic per load
 * grows with GF and the unroll can shrink before the 255-register cliff. */
#define FA_DEC_VU(GF) ((GF) >= 8 ? 2 : ((GF) >= 4 ? 4 : 8))

struct bf16v8 {
    __nv_bfloat16 x[8];
};

__device__ __forceinline__ bf16v8 ld_glob8(const __nv_bfloat16* p) {
    bf16v8 r;
    *(uint4*)&r = *(const uint4*)p; /* 16B, naturally aligned: D and dbase are multiples of 8 */
    return r;
}
__device__ __forceinline__ bf16v8 ld_smem8(const __nv_bfloat16* p) {
    bf16v8 r;
    *(uint4*)&r = *(const uint4*)p;
    return r;
}
/* Streaming (evict-first) twin of ld_glob8 for ONCE-STREAMED rows — the KV cache and GEMV
 * weight rows, read exactly once per token, which can only displace reused lines (the x row,
 * neighbouring packets' operands) from L1/L2. Identical bytes; cache policy only. */
__device__ __forceinline__ bf16v8 ld_glob8_cs(const __nv_bfloat16* p) {
    bf16v8 r;
    *(uint4*)&r = __ldcs((const uint4*)p);
    return r;
}
__device__ __forceinline__ float dot8(const bf16v8& a, const bf16v8& b, float acc) {
#pragma unroll
    for (int i = 0; i < 8; i++) acc = fmaf(__bfloat162float(a.x[i]), __bfloat162float(b.x[i]), acc);
    return acc;
}

/* ---- KVZIP-SZ12 v1.2 lossless KV row blob (bench-only until KV-3) --
 * Row blob = [D/2 B code plane (4-bit codes, low nibble = even dim) | D B lo plane (sign<<7|mant7)
 * | 32 B tail: u32 hdr = base | nesc<<8, then 9 x 3 B escape slots {exp u8, dim u16 LE}].
 * code c in [0,14]: exp = base + c; c == 15: exp from the slot carrying this dim. Decode is exact
 * bit reassembly of the original bf16 (sign | exp | mant7), so QK/PV below stays BYTE-IDENTICAL
 * to the bf16 cache — same bf16v8, same dot8 — only the load path (1.28x fewer bytes) differs.
 * Escape rows are <0.15% of rows (oracle, 21 GB): the slot scan is real but almost never taken. */
#define FA_SZ_ROWB(D) ((unsigned)((D) / 2 + (D) + 32))
template <int D>
__device__ __forceinline__ bf16v8 fa_sz12_dec8(const unsigned char* __restrict__ row, int d,
                                               unsigned hdr) {
    const unsigned base = hdr & 0xFFu, nesc = (hdr >> 8) & 0xFFu;
    const unsigned codes = __ldcs((const unsigned*)(row + (d >> 1)));
    const uint2 lov = __ldcs((const uint2*)(row + D / 2 + d));
    bf16v8 r;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned c = (codes >> (4 * i)) & 0xFu;
        const unsigned lo = ((i < 4 ? lov.x : lov.y) >> (8 * (i & 3))) & 0xFFu;
        unsigned e = base + c;
        if (nesc && c == 15u) {
            e = 0u;
#pragma unroll 1
            for (unsigned j = 0; j < nesc; j++) {
                const unsigned char* sl = row + D / 2 + D + 4 + 3 * j;
                if ((unsigned)(sl[1] | (sl[2] << 8)) == (unsigned)(d + i)) { e = sl[0]; break; }
            }
        }
        __nv_bfloat16_raw br;
        br.x = (unsigned short)(((lo & 0x80u) << 8) | (e << 7) | (lo & 0x7Fu));
        r.x[i] = __nv_bfloat16(br);
    }
    return r;
}

/* ---- FP8 (OCP e4m3) KV-cache helpers (PLOW_FP8_KV) ------------------------------------------------
 * NVIDIA port of the amd_common.h KV twins. The decode/prefill flash reads the K/V cache as e4m3
 * (1 byte/elem, HALF the HBM bytes) with a PER-ROW f32 dequant scale; d_headnorm_rope_fp8 writes it.
 * e4m3 has 3 mantissa bits, which fit EXACTLY in bf16, so fp8 -> bf16 is lossless and the QK/PV
 * math is byte-identical to a bf16 cache holding the same rounded values — the only loss is the
 * e4m3 quantization at the write, which attention tolerates (vLLM/CK/AITER all ship fp8 KV). */
#define PLOW_FP8_E4M3_MAX 448.0f
struct fp8v8 {
    uint8_t x[8];
};
/* 8 e4m3 == 8 B == one uint2: the V phase reads only its 8 owned head-dims per lane; a K row is
 * streamed 8 at a time. D and dbase are multiples of 8, so this is a naturally aligned 8-byte load. */
__device__ __forceinline__ fp8v8 ld_glob_fp8v8(const unsigned char* p) {
    fp8v8 r;
    *(uint2*)&r = *(const uint2*)p;
    return r;
}
/* Decode 8 e4m3 bytes -> bf16v8 (reuses the exact op_gemm.cuh fp8x2 idiom). fp8 -> half -> bf16 is
 * exact for e4m3 (3 mantissa bits < bf16's 7), so no precision is lost beyond the stored quantization. */
__device__ __forceinline__ bf16v8 fp8v8_to_bf16v8(fp8v8 w) {
    const uint16_t* wp = (const uint16_t*)&w; /* 4 packed fp8x2 pairs */
    bf16v8 d;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
        d.x[2 * j] = __float2bfloat16(f.x);
        d.x[2 * j + 1] = __float2bfloat16(f.y);
    }
    return d;
}
/* ---- PLOW_FP8_FAST dequant path (beat26b-flashdec) ------------------------------------------
 * The shipped fp8 arm converts e4m3 -> bf16 (fp8v8_to_bf16v8) then the dot converts bf16 -> f32,
 * a double rounding + extra ALU per element. The hd512 full-attn decode microbench shows the fp8
 * read is DEQUANT-ALU bound, not bandwidth bound (issued BW plateaus ~1300 GB/s, below the 1535
 * ceiling and far below the L2-boosted bf16 path). Two ALU cuts, fp8-arm only, flag-gated so the
 * shipped fp8 path + all bf16 paths stay byte-identical:
 *   (a) 16-byte K loads (uint4 = 16 e4m3) in the score phase -> half the load instructions.
 *   (b) e4m3 -> f32 DIRECTLY (skip the intermediate __float2bfloat16), then fma against f32(Q). */
struct fp8v16 { uint8_t x[16]; };
__device__ __forceinline__ fp8v16 ld_glob_fp8v16(const unsigned char* p) {
    fp8v16 r;
    *(uint4*)&r = *(const uint4*)p; /* 16 e4m3 == 16 B == one uint4; d steps by 16, base row-aligned */
    return r;
}
/* 16 e4m3 -> 16 f32 (exact), no bf16 intermediate. */
__device__ __forceinline__ void fp8v16_to_f32(const fp8v16& w, float out[16]) {
    const uint16_t* wp = (const uint16_t*)&w; /* 8 packed fp8x2 pairs */
#pragma unroll
    for (int j = 0; j < 8; j++) {
        __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
        out[2 * j] = f.x;
        out[2 * j + 1] = f.y;
    }
}
/* 8 e4m3 -> 8 f32 (exact), no bf16 intermediate — the V phase per-lane 8-dim decode. */
__device__ __forceinline__ void fp8v8_to_f32(fp8v8 w, float out[8]) {
    const uint16_t* wp = (const uint16_t*)&w;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
        out[2 * j] = f.x;
        out[2 * j + 1] = f.y;
    }
}
/* f32 -> one e4m3 byte; hardware round-to-nearest-even + saturation to 448 (same rounding the
 * offline w8a8 activation quant uses in d_quant_fp8). */
__device__ __forceinline__ uint8_t quant_fp8(float v) {
    __nv_fp8_e4m3 q(v);
    return *(const uint8_t*)&q;
}
/* One e4m3 byte -> f32 (exact). The prefill fp8 dequant stages one head-dim per thread. */
__device__ __forceinline__ float fp8_to_f32(uint8_t b) {
    __nv_fp8_e4m3 q;
    *(uint8_t*)&q = b;
    return (float)q;
}
/* 32-LANE reductions. The AMD twins are wave64 (offsets start at 32); transliterating them
 * would leave half of every tile unreduced and silently produce a too-small max / partial sum. */
/* FA_NV_WAVE64_NEGCTRL builds the WRONG (naively transliterated wave64) reduction on purpose,
 * so the test suite can prove it catches exactly this bug. Never define it in a real build. */
#ifdef FA_NV_WAVE64_NEGCTRL
#define FA_RED_OFF0 32
#else
#define FA_RED_OFF0 16
#endif
__device__ __forceinline__ float warp_max32(float v) {
#pragma unroll
    for (int off = FA_RED_OFF0; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off, 32));
    return v;
}
__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int off = FA_RED_OFF0; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off, 32);
    return v;
}

/* ---- mma.sync m16n8k16 primitives for the flash-prefill QK^T (rtx-05 T1) ------------------------
 * Flash-local copies of the ldmatrix/mma atoms. op_attention.cuh is included BEFORE op_gemm.cuh
 * (via sm120_common.cuh), so the pgm_* twins there are not yet in scope here; these fa_* wrappers
 * are byte-identical asm so the QK^T tile reuses the exact operand layout the tiled GEMM validates.
 * A = Q [16 query][16 hd] row-major (ldmatrix.x4); B = K^T [16 hd][8 kv], staged transposed to
 * [hd][kv] and loaded with ldmatrix.x2.trans, EXACTLY as d_gemm loads its B=weight operand. */
__device__ __forceinline__ void fa_ldmatrix_x4(unsigned (&r)[4], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void fa_ldmatrix_x2_trans(unsigned (&r)[2], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void fa_mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
                                       const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
/* m16n8k32 e4m3 mma (2x-rate on sm_120), f32 acc — flash-local twin of op_gemm.cuh's
 * pgm_mma_fp8_k32 (op_attention.cuh is included first, see the fa_* note above). Fragment layout
 * verified in experiments/fp8_verify.cu: the f32 ACCUMULATOR layout is IDENTICAL to the bf16
 * m16n8k16 accumulator (row=(L>>2)+8*(e>>1), col=2*(L&3)+(e&1)); operands are PLAIN u32 smem
 * reads (8-bit has no ldmatrix): A lane L holds rows (L>>2)/(L>>2)+8 at k-bytes 8*(L&3)..+7
 * (a[0]=rlo k..k+3, a[2]=rlo k+4..k+7, a[1]/a[3] same for rhi); B lane L holds col (L>>2) at the
 * same k-bytes (b[0]=k..k+3, b[1]=k+4..k+7). */
__device__ __forceinline__ void fa_mma_fp8_k32(float (&d)[4], const unsigned (&a)[4],
                                               const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
/* m16n8k16 FP16-operand mma, f32 acc — the fp8mma P.V twin (V dequants e4m3 -> half via the
 * single-instruction cvt.f16x2.e4m3x2, so the P.V tile stays fp16; same rate as bf16 mma). */
__device__ __forceinline__ void fa_mma_f16(float (&d)[4], const unsigned (&a)[4],
                                           const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

/* ---- cp.async KV-stream pipeline for flash prefill (rtx-07 T5, PLOW_NV_FA_PIPE) -----------------
 * The T4 flash-prefill stages each K/V tile with plain vectorized loads + a __syncthreads, so the
 * mma waits on the whole tile landing from HBM (the 128k HBM-KV-bound tail). This is the same
 * cp.async ring discipline the tiled GEMM proved in T3 (op_gemm.cuh pgm_*): the K/V rows are
 * contiguous [kv][hd] in the cache, so 16-byte cp.async.cg lines fill smem gmem->smem with no
 * register round-trip and overlap with the mma.
 *
 * KEY REUSE (T3): the QK^T B operand is K^T. T4 staged K TRANSPOSED into KsT[hd][kv] and read it
 * with ldmatrix.x2.trans — a scatter cp.async cannot express. Instead stage K NATURAL [kv][hd]
 * (contiguous, cp.async-friendly, EXACTLY the Vs layout) and read the mma B fragment with
 * ldmatrix.x2 NON-.trans (fa_ldmatrix_x2 below), which T3 proved bit-identical to the transposed
 * path (t3_pipe_probe.cu, 0/2048). V staging is already natural; only its load becomes cp.async.
 *
 * SMEM WALL (measured, sm_120 = 99 KiB opt-in): a full cross-tile double-buffer of BOTH K and V
 * (+33 KiB at hd256) exceeds 99 KiB, so it does NOT fit. This pipeline stays SINGLE-buffered and
 * still overlaps both loads with compute by exploiting the operand lifetimes: K is dead after QK,
 * so K[t+1] is prefetched right after QK[t] (overlaps softmax[t]+P.V[t]); V is needed last, so
 * V[t] is loaded at the tile top (overlaps QK[t]+softmax[t]). Natural K also SHRINKS the arena
 * (KsT[hd][kv] 10240 -> Ks[kv][hd] 8448 bf16 at hd256), so occupancy is unchanged.
 *
 * DEFAULT 1 (rtx-07 T5, proven): bit-identical to the T4 sync path (natural-K is the T3-proven
 * non-.trans equivalent; cp.async moves the same bytes — GPU-verified LOGITS BIT-IDENTICAL vs the
 * T4 build), and −16%@4k / −35%@16k / −46%@32k / −62%@64k / −81%@128k end-to-end prefill (the win
 * grows with ctx as the O(ctx²) HBM-KV-bound full-attention share rises). Build
 * with -DPLOW_NV_FA_PIPE=0 for the T4 synchronous-staging A/B control. Only objects that compile
 * d_flash_prefill (the _pf prefill object + flash oracle) are affected; decode/Qwen objects do not
 * carry flash-prefill and stay byte-identical either way. */
#ifndef PLOW_NV_FA_PIPE
#define PLOW_NV_FA_PIPE 1
#endif
__device__ __forceinline__ void fa_ldmatrix_x2(unsigned (&r)[2], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
/* ---- PX-8: the 8-bit TRANSPOSING ldmatrix (sm_120a) ----------------------------------------
 * mma.m16n8k32 needs both operands with the CONTRACTION dim contiguous per lane. For the P.V the
 * contraction is kv, and V is staged natural Vs8[kv][hd] — so an fp8 P.V needs V transposed. This
 * instruction does it in the load. Only compiled into the PX-8 arm (PLOW_NV_FA_FP8PV), which is
 * sm_120a-only; ptxas rejects it on every other target, and -arch=sm_120a is NOT enough (it also
 * emits a compute_120 PTX image) — build with -gencode arch=compute_120a,code=sm_120a.
 *
 * MEASURED fragment map (perf-data/px8_flash_fp8pv_bench.cu `layout`): lane L supplies the address
 * of source row L (rows 0..15 = matrix 0, 16..31 = matrix 1) and receives, for source column n:
 *   r[0] = T[n = L>>2    ][srcrow 4*(L&3) .. +3]  (matrix 0)
 *   r[1] = T[n = (L>>2)+8][srcrow 4*(L&3) .. +3]  (matrix 0)
 *   r[2], r[3] = the same two columns out of matrix 1
 * The mma's B operand instead wants lane L to hold B[n=L>>2][k = 8*(L&3) .. +7], which differs by a
 * QUAD PERMUTATION of k. FA_PX8_VROW absorbs that permutation into the smem ROW ORDER V is staged
 * in — cp.async copies whole 16B lines, so only the destination row index changes and the whole
 * transpose costs nothing. With it, {r0,r2} and {r1,r3} ARE the two B operands. */
#ifndef PLOW_NV_FA_FP8PV
#define PLOW_NV_FA_FP8PV 0
#endif
#if PLOW_NV_FA_FP8PV
__device__ __forceinline__ void fa_ldmatrix_x2_trans_b8(unsigned (&r)[4], const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
#endif
/* kv row -> smem row for the V tile of the PX-8 arm (BKV=32). Bijection; see above. */
#define FA_PX8_VROW(kv) ((((kv) & 4) ? 16 : 0) + 4 * ((kv) >> 3) + ((kv) & 3))
__device__ __forceinline__ void fa_cp_async_cg16(void* smem, const void* gmem, int src_bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
                 "r"(src_bytes));
}
__device__ __forceinline__ void fa_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void fa_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

#if PLOW_NV_FA_TC_GQA8_HD512
__device__ __forceinline__ unsigned fa_tf32_rn(float value) {
    unsigned bits;
    asm volatile("cvt.rn.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value));
    return bits;
}

__device__ __forceinline__ void fa_mma_tf32(float (&d)[4], const unsigned (&a)[4],
                                             const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <int D, int GF>
__device__ __forceinline__ void fa_decode_qk_tc_gqa8(
    float* scores, const __nv_bfloat16* qtile, __nv_bfloat16* ktile,
    const __nv_bfloat16* kbase, unsigned kv0, unsigned live_rows, unsigned kv_mask,
    float scale) {
    static_assert(D == 512 && GF == 8, "tensor-core decode is HD512/GQA8 only");
    constexpr unsigned STRIDE = D + 8;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    for (unsigned r = live_rows + tid; r < FA_DEC_TILE; r += PLOW_NV_THREADS)
#pragma unroll
        for (int g = 0; g < GF; ++g) scores[g * FA_DEC_TILE + r] = FA_NEG_INF;
    for (unsigned first = 0; first < live_rows; first += 64) {
        const unsigned nr = live_rows - first < 64 ? live_rows - first : 64;
        for (unsigned v = tid; v < 64u * (D / 8); v += PLOW_NV_THREADS) {
            const unsigned r = v / (D / 8), c = (v % (D / 8)) * 8;
            const bool live = r < nr;
            const __nv_bfloat16* in = live
                ? kbase + (size_t)((kv0 + first + r) & kv_mask) * D + c
                : kbase;
            fa_cp_async_cg16(ktile + r * STRIDE + c, in, live ? 16 : 0);
        }
        fa_cp_commit();
        fa_cp_wait<0>();
        __syncthreads();
        float acc[4] = {};
#pragma unroll
        for (unsigned k0 = 0; k0 < D; k0 += 64) {
            float partial[4] = {};
#pragma unroll
            for (unsigned k = k0; k < k0 + 64; k += 16) {
                unsigned a[4], b[2];
                fa_ldmatrix_x4(a, qtile + (lane % 16) * STRIDE + k + (lane / 16) * 8);
                fa_ldmatrix_x2(b, ktile + (warp * 8 + (lane & 7)) * STRIDE + k +
                                           ((lane >> 3) & 1) * 8);
                fa_mma(partial, a, b, partial);
            }
#pragma unroll
            for (unsigned e = 0; e < 4; ++e) acc[e] = __fadd_rn(acc[e], partial[e]);
        }
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            const unsigned g = lane / 4 + (e / 2) * 8;
            const unsigned r = warp * 8 + (lane % 4) * 2 + (e % 2);
            if (g < GF && r < nr)
                scores[g * FA_DEC_TILE + first + r] = acc[e] * FA_SCALE(scale);
        }
        __syncthreads();
    }
}

template <int D, int GF>
__device__ __forceinline__ void fa_decode_pv_tc_gqa8(
    float (&acc)[D / 64][4], const float* scores, __nv_bfloat16* vtile,
    const __nv_bfloat16* vbase, unsigned kv0, unsigned live_rows, unsigned kv_mask,
    const float* corr_shared) {
    static_assert(D == 512 && GF == 8, "tensor-core decode is HD512/GQA8 only");
    constexpr unsigned STRIDE = D + 8, NJ = D / 64;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const float resc[2] = {lane / 4 < GF ? corr_shared[lane / 4] : 0.0f,
                           lane / 4 + 8 < GF ? corr_shared[lane / 4 + 8] : 0.0f};
#pragma unroll
    for (unsigned j = 0; j < NJ; ++j)
#pragma unroll
        for (unsigned e = 0; e < 4; ++e) acc[j][e] *= resc[e / 2];
    for (unsigned first = 0; first < live_rows; first += 64) {
        const unsigned nr = live_rows - first < 64 ? live_rows - first : 64;
        for (unsigned i = tid; i < 64u * (D / 8); i += PLOW_NV_THREADS) {
            const unsigned r = i / (D / 8), c = (i % (D / 8)) * 8;
            const bool live = r < nr;
            const __nv_bfloat16* in = live
                ? vbase + (size_t)((kv0 + first + r) & kv_mask) * D + c
                : vbase;
            fa_cp_async_cg16(vtile + r * STRIDE + c, in, live ? 16 : 0);
        }
        fa_cp_commit();
        fa_cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (unsigned k = 0; k < 64; k += 8) {
            unsigned hi[4], lo[4];
#pragma unroll
            for (unsigned e = 0; e < 4; ++e) {
                const unsigned g = lane / 4 + (e & 1) * 8;
                const unsigned r = k + lane % 4 + (e / 2) * 4;
                const float p = g < GF && r < nr
                    ? scores[g * FA_DEC_TILE + first + r]
                    : 0.0f;
                /* The low term retains the signed residual from the first TF32 rounding. */
                hi[e] = fa_tf32_rn(p);
                lo[e] = fa_tf32_rn(__fsub_rn(p, __uint_as_float(hi[e])));
            }
#pragma unroll
            for (unsigned j = 0; j < NJ; ++j) {
                const unsigned col = (warp * NJ + j) * 8 + lane / 4;
                unsigned b[2] = {
                    __float_as_uint(__bfloat162float(vtile[(k + lane % 4) * STRIDE + col])),
                    __float_as_uint(__bfloat162float(
                        vtile[(k + lane % 4 + 4) * STRIDE + col]))};
                fa_mma_tf32(acc[j], lo, b, acc[j]);
                fa_mma_tf32(acc[j], hi, b, acc[j]);
            }
        }
        __syncthreads();
    }
}
#endif

/* Persistent-grid body: this block runs work items `slice, slice+nblk, ...`.
 *
 * BATCH>1 (serving pending #4): the KV cache is [n_batch][n_kv_head][kv_stride][D] — each
 * sequence owns its own ring — and this body ALREADY indexes it per-batch (kbase/vbase below).
 * The historical blocker was NOT here: declare() sized the KV tensor for ONE sequence, so a
 * b>=1 read walked PAST it into the next arena tensor — fluent wrong text, no crash. `kv_cap`
 * is the DECLARED KV row-capacity (n_batch_alloc*n_kv_head*kv_stride); under PLOW_NV_KVBOUNDS a
 * b>=1 read past an under-sized allocation TRAPS instead of reading neighbouring memory. The
 * emitter passes cap=0 for B=1 (the check is skipped and the B=1 packet stays byte-identical). */
#ifndef PLOW_NV_KVBOUNDS
#define PLOW_NV_KVBOUNDS 0
#endif
template <int D, int GF, bool FP8KV = false, bool SZKV = false, bool SLOTMAP = false>
__device__ void d_flash_decode(float* __restrict__ Opart, float* __restrict__ mlpart,
                               const __nv_bfloat16* __restrict__ Q,
                               const __nv_bfloat16* __restrict__ K,
                               const __nv_bfloat16* __restrict__ V,
                               const int* __restrict__ kv_len, unsigned n_batch, unsigned n_head,
                               unsigned n_kv_head, unsigned kv_stride, unsigned window,
                               float scale, unsigned nsplit, unsigned kv_mask, unsigned slice,
                               unsigned nblk, float* lds, unsigned kv_cap = 0,
                               const float* __restrict__ k_scale = nullptr,
                               const float* __restrict__ v_scale = nullptr,
                               const int* __restrict__ decode_slot = nullptr) {
    /* A work item carries GF CONSECUTIVE query heads sharing one KV head (needs GF | gqa).
     * Indexing by head-GROUP, not by kv_head, is what makes GF < gqa correct. */
    static_assert(GF <= (int)PLOW_NV_WARPS || (PLOW_NV_FA_GF16_BENCH && GF == 16), "unsupported decode GQA grouping");
    const unsigned gqa = n_head / n_kv_head;
    const unsigned n_grp = n_head / GF;
    const unsigned n_work = n_batch * n_grp * nsplit;
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5, lane = tid & 31;

    float* Ssm = lds;
    float* hmax = lds + GF * FA_DEC_TILE;
    float* hsum = hmax + FA_DEC_REDUCTION_HEADS(GF);
    __nv_bfloat16* qsm = (__nv_bfloat16*)(hsum + FA_DEC_REDUCTION_HEADS(GF));
    float* osm = (float*)(qsm + GF * D);

    constexpr int NDT = FA_DEC_NDT(D);
    constexpr int NG = FA_DEC_NG(D);
    const unsigned dbase = (tid % NDT) * 8;
    const unsigned grp = tid / NDT;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned hg = (w / nsplit) % n_grp;
        const unsigned b = w / (nsplit * n_grp);
        unsigned slot = b;
        if constexpr (SLOTMAP) {
            const int physical = decode_slot[b];
            if (physical < 0) __trap();
            slot = (unsigned)physical;
        }
        const unsigned h0 = hg * GF;
        const unsigned hkv = h0 / gqa;

        const unsigned len = (unsigned)kv_len[b];
        const unsigned qpos = len - 1; /* decode: the query token is the newest one */

        /* This split's KV range, already clamped to the window so a sliding layer's work does
         * not grow with context (the AMD comment's O(ctx) fix; kept because it is free). */
        const unsigned first = (window && len > window) ? (len - window) : 0u;
        const unsigned span = len - first;
        const unsigned per = (span + nsplit - 1) / nsplit;
        const unsigned lo = first + sp * per;
        const unsigned hi = (lo + per < len) ? (lo + per) : len;

        /* Per-batch KV base: sequence b's own ring. The trap fires when this base's LAST row
         * (b*n_kv_head+hkv)*kv_stride + (kv_stride-1) exceeds the declared capacity — i.e. when
         * declare() under-sized the KV tensor for the batch this packet was emitted at. */
#if PLOW_NV_KVBOUNDS
        if constexpr (SLOTMAP) {
            if (kv_cap && (((size_t)slot * n_kv_head + hkv) * kv_stride + (kv_stride - 1)) >= kv_cap)
                __trap();
        } else {
            if (kv_cap && (((size_t)b * n_kv_head + hkv) * kv_stride + (kv_stride - 1)) >= kv_cap)
                __trap();
        }
#endif
        const size_t kv_batch = slot;
        const __nv_bfloat16* kbase = K + (kv_batch * n_kv_head + hkv) * (size_t)kv_stride * D;
        const __nv_bfloat16* vbase = V + (kv_batch * n_kv_head + hkv) * (size_t)kv_stride * D;
        /* FP8 KV: the cache is uint8 e4m3 (1 byte/elem, HALF the bytes) + a PER-ROW f32 scale, in
         * the SAME head-major RING layout. These byte bases and the per-(kv_head) scale slice are
         * unused on the bf16 instantiation (FP8KV=false compiles them out). */
        /* SZKV: rows are FA_SZ_ROWB(D)-byte blobs, not D elems — same head-major ring indexing. */
        constexpr size_t RB = SZKV ? (size_t)FA_SZ_ROWB(D) : (size_t)D;
        const unsigned char* kb8 = (const unsigned char*)K + (kv_batch * n_kv_head + hkv) * (size_t)kv_stride * RB;
        const unsigned char* vb8 = (const unsigned char*)V + (kv_batch * n_kv_head + hkv) * (size_t)kv_stride * RB;
        /* Offset the scale slice only on the FP8KV arm — k_scale/v_scale are null on the bf16
         * instantiation, so forming null+offset (even unused) is UB. if constexpr keeps ksc/vsc
         * in scope for the FP8KV code below with zero runtime cost. */
        const float* ksc = nullptr;
        const float* vsc = nullptr;
        if constexpr (FP8KV) {
            ksc = k_scale + (kv_batch * n_kv_head + hkv) * (size_t)kv_stride;
            vsc = v_scale + (kv_batch * n_kv_head + hkv) * (size_t)kv_stride;
        }

        __syncthreads(); /* previous item's osm reads must finish before qsm is rewritten */
#if !(PLOW_NV_FA_WPR && PLOW_NV_FA_QGLOB)
        for (unsigned i = tid; i < GF * D; i += PLOW_NV_THREADS)
            qsm[i] = Q[((size_t)b * n_head + h0 + i / D) * D + i % D];
        __syncthreads();
#endif

#if PLOW_NV_FA_TC_GQA8_HD512
        __nv_bfloat16* tc_q = (__nv_bfloat16*)(lds + FA_DEC_BASE_SMEM_FLOATS(D, GF));
        __nv_bfloat16* tc_kv = tc_q + 16 * (D + 8);
        if constexpr (FA_DEC_TC_GQA8(D, GF) && !FP8KV && !SZKV) {
            for (unsigned i = tid; i < 16u * D; i += PLOW_NV_THREADS) {
                const unsigned g = i / D, c = i % D;
                tc_q[g * (D + 8) + c] =
                    g < GF ? qsm[g * D + c] : __float2bfloat16(0.0f);
            }
            __syncthreads();
        }
#endif

#if PLOW_NV_FA_WPR && PLOW_NV_FA_QREG
        /* Hoist invariant Q fragments across KV rows; GF>4 keeps the lower-register path. */
        bf16v8 qreg[GF][D >= 256 ? D / 256 : 1];
        if constexpr (!SZKV && !FP8KV && D == 512 && GF <= 4) {
#pragma unroll
            for (int g = 0; g < GF; g++)
#pragma unroll
                for (int c = 0; c < D / 256; c++) {
                    const unsigned off = (unsigned)c * 256u + lane * 8u;
#if PLOW_NV_FA_QGLOB
                    qreg[g][c] = ld_glob8(Q + ((size_t)b * n_head + h0 + (unsigned)g) * D + off);
#else
                    qreg[g][c] = ld_smem8(qsm + g * D + off);
#endif
                }
        }
#endif

        float m_st[GF], l_st[GF];
#pragma unroll
        for (int g = 0; g < GF; g++) { m_st[g] = FA_NEG_INF; l_st[g] = 0.0f; }
        float oacc[GF][8];
#pragma unroll
        for (int g = 0; g < GF; g++)
#pragma unroll
            for (int u = 0; u < 8; u++) oacc[g][u] = 0.0f;
#if PLOW_NV_FA_TC_GQA8_HD512
        float tc_oacc[D / 64][4];
        if constexpr (FA_DEC_TC_GQA8(D, GF) && !FP8KV && !SZKV) {
#pragma unroll
            for (int j = 0; j < D / 64; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) tc_oacc[j][e] = 0.0f;
        }
#endif

        for (unsigned kv0 = lo; kv0 < hi; kv0 += FA_DEC_TILE) {
            /* Live rows in this tile. Entries [rmax_t, FA_DEC_TILE) are NEG_INF in BOTH bodies
             * (the per-thread one leaves s[]=NEG_INF for kv>=hi), so the softmax reductions can
             * stop at rmax_t instead of sweeping the dead tail -- at nsplit=8 that tail is half
             * the tile. */
            const unsigned rmax_t = (hi - kv0 < (unsigned)FA_DEC_TILE) ? (hi - kv0)
                                                                       : (unsigned)FA_DEC_TILE;
            float s[GF];
#if PLOW_NV_FA_TC_GQA8_HD512
            if constexpr (FA_DEC_TC_GQA8(D, GF) && !FP8KV && !SZKV) {
                fa_decode_qk_tc_gqa8<D, GF>(Ssm, tc_q, tc_kv, kbase, kv0, rmax_t,
                                             kv_mask, scale);
#pragma unroll
                for (int g = 0; g < GF; ++g) s[g] = Ssm[g * FA_DEC_TILE + tid];
            } else
#endif
#if PLOW_NV_FA_WPR
          /* WARP-PER-ROW SCORE PHASE (H100 round 9). The default body gives each THREAD a whole
           * KV row, so one warp instruction issues 32 requests D*2 bytes apart -- 32 scattered
           * sectors rather than one coalesced burst -- and each thread then walks its row with
           * D/8 dependent 16 B loads. flash_decode measured 237 GB/s, ~14x off roofline, and
           * neither nsplit nor an explicit K pre-issue depth (PLOW_NV_FA_KUN) moved it.
           * Here a WARP owns a row: its 32 lanes cover 32*8 = 256 elements, so a D=256 row is ONE
           * fully coalesced 512 B load (D=512 is two). The dot then costs a warp reduction per
           * query head, which the per-thread form did not need. Threads read their own row's
           * score back out of Ssm so the softmax/PV code below is untouched.
           * Only the plain bf16 KV layout takes this path; SZ/fp8 KV keep the default body. The K
           * reduction order changes, so scores are numerically equivalent, not bit-identical. */
          if constexpr (!SZKV && !FP8KV && D >= 256) {
            constexpr int NC = D / 256; /* 256 = 32 lanes * 8 elems */
            /* Only [0,rmax) of the tile has live rows -- at nsplit=8 a work item owns 128 rows
             * against a 256-row tile, so half the sweep was pure loop+store overhead. Fill the
             * dead tail with NEG_INF in one cheap strided pass instead of iterating it. */
            const unsigned rmax = rmax_t;
            for (unsigned i = tid + rmax; i < (unsigned)FA_DEC_TILE; i += PLOW_NV_THREADS) {
#pragma unroll
                for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + i] = FA_NEG_INF;
            }
            constexpr int WRB = PLOW_NV_FA_WPR_RB;
            for (unsigned rb = warp; rb < rmax; rb += PLOW_NV_WARPS * WRB) {
                /* WRB rows, strided by the warp count so each warp's batch stays disjoint. */
                bf16v8 k8[WRB][NC];
                bool live[WRB];
#pragma unroll
                for (int t = 0; t < WRB; t++) {
                    const unsigned r = rb + (unsigned)t * PLOW_NV_WARPS;
                    const unsigned kvr = kv0 + r;
                    live[t] = (r < rmax) && (kvr <= qpos) &&
                              (!window || (qpos - kvr) < window);
                    /* `kvr & kv_mask` always lands on a real ring row, so the load is
                     * in-bounds even for a masked or past-the-tile r; only the dot and the
                     * store below are gated. Loading unconditionally keeps all WRB loads
                     * issued back-to-back, which is the entire point of the batch. */
                    const __nv_bfloat16* krow = kbase + (size_t)(kvr & kv_mask) * D;
#pragma unroll
                    for (int c = 0; c < NC; c++)
                        k8[t][c] = ld_glob8_cs(krow + (unsigned)c * 256u + lane * 8u);
                }
#pragma unroll
                for (int t = 0; t < WRB; t++) {
                    const unsigned r = rb + (unsigned)t * PLOW_NV_WARPS;
                    float sr[GF];
#pragma unroll
                    for (int g = 0; g < GF; g++) sr[g] = FA_NEG_INF;
                    if (live[t]) {
                        float dt[GF];
#pragma unroll
                        for (int g = 0; g < GF; g++) dt[g] = 0.0f;
#pragma unroll
                        for (int c = 0; c < NC; c++) {
                            const unsigned off = (unsigned)c * 256u + lane * 8u;
#pragma unroll
                            for (int g = 0; g < GF; g++)
#if PLOW_NV_FA_QREG
                                if constexpr (D == 512 && GF <= 4)
                                    dt[g] = dot8(k8[t][c], qreg[g][c], dt[g]);
                                else
#endif
#if PLOW_NV_FA_QGLOB
                                dt[g] = dot8(k8[t][c],
                                             ld_glob8(Q + ((size_t)b * n_head + h0 + (unsigned)g) * D + off),
                                             dt[g]);
#else
                                dt[g] = dot8(k8[t][c], ld_smem8(qsm + g * D + off), dt[g]);
#endif
                        }
#pragma unroll
                        for (int g = 0; g < GF; g++) sr[g] = warp_sum32(dt[g]) * FA_SCALE(scale);
                    }
                    if (lane == 0 && r < rmax) {
#pragma unroll
                        for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + r] = sr[g];
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = Ssm[g * FA_DEC_TILE + tid];
          } else
#endif
          {
            /* SCORES: each thread streams one whole K row and dots it against all GF query rows
             * out of smem — the row crosses HBM once instead of GF times (the GQA fusion). */
            const unsigned kv = kv0 + tid;
#pragma unroll
            for (int g = 0; g < GF; g++) s[g] = FA_NEG_INF;
            if (kv < hi && kv <= qpos && (!window || (qpos - kv) < window)) {
                float dot[GF];
#pragma unroll
                for (int g = 0; g < GF; g++) dot[g] = 0.0f;
                if constexpr (SZKV) {
                    /* sz12: stream the 1.28x-smaller row blob, reassemble exact bf16, same dot8. */
                    const unsigned char* krow = kb8 + (size_t)(kv & kv_mask) * RB;
                    const unsigned khdr = *(const unsigned*)(krow + D / 2 + D);
#pragma unroll
                    for (int d = 0; d < D; d += 8) {
                        const bf16v8 k8 = fa_sz12_dec8<D>(krow, d, khdr);
#pragma unroll
                        for (int g = 0; g < GF; g++) dot[g] = dot8(k8, ld_smem8(qsm + g * D + d), dot[g]);
                    }
#pragma unroll
                    for (int g = 0; g < GF; g++) s[g] = dot[g] * FA_SCALE(scale);
                } else if constexpr (FP8KV) {
                    /* fp8: the K row is e4m3 (HALF the bytes). Decode 8-wide to bf16v8 (exact) and
                     * dot against the bf16 Q rows; the per-row dequant scale multiplies ONCE, after
                     * the dot (it factors out of every term). */
                    const unsigned char* krow = kb8 + (size_t)(kv & kv_mask) * D;
#if defined(PLOW_FP8_LD16)
                    /* 16B loads + e4m3->f32 direct (no bf16 round-trip). D is a multiple of 16. */
#pragma unroll
                    for (int d = 0; d < D; d += 16) {
                        float kf[16];
                        fp8v16_to_f32(ld_glob_fp8v16(krow + d), kf);
#pragma unroll
                        for (int g = 0; g < GF; g++) {
                            const bf16v8 q0 = ld_smem8(qsm + g * D + d);
                            const bf16v8 q1 = ld_smem8(qsm + g * D + d + 8);
#pragma unroll
                            for (int i = 0; i < 8; i++) dot[g] = fmaf(kf[i], __bfloat162float(q0.x[i]), dot[g]);
#pragma unroll
                            for (int i = 0; i < 8; i++) dot[g] = fmaf(kf[8 + i], __bfloat162float(q1.x[i]), dot[g]);
                        }
                    }
#elif defined(PLOW_FP8_FAST)
                    /* 8B loads + e4m3->f32 direct (no bf16 round-trip); no extra register pressure. */
#pragma unroll
                    for (int d = 0; d < D; d += 8) {
                        float kf[8];
                        fp8v8_to_f32(ld_glob_fp8v8(krow + d), kf);
#pragma unroll
                        for (int g = 0; g < GF; g++) {
                            const bf16v8 q0 = ld_smem8(qsm + g * D + d);
#pragma unroll
                            for (int i = 0; i < 8; i++) dot[g] = fmaf(kf[i], __bfloat162float(q0.x[i]), dot[g]);
                        }
                    }
#else
#pragma unroll
                    for (int d = 0; d < D; d += 8) {
                        const bf16v8 k8 = fp8v8_to_bf16v8(ld_glob_fp8v8(krow + d));
#pragma unroll
                        for (int g = 0; g < GF; g++) dot[g] = dot8(k8, ld_smem8(qsm + g * D + d), dot[g]);
                    }
#endif
                    const float ks = ksc[kv & kv_mask];
#pragma unroll
                    for (int g = 0; g < GF; g++) s[g] = dot[g] * FA_SCALE(scale) * ks;
                } else {
                    const __nv_bfloat16* krow = kbase + (size_t)(kv & kv_mask) * D;
#if PLOW_NV_FA_KUN > 1
                    /* Stage KUN K chunks before consuming any (the V/PV loop below already does
                     * this with vv[VU]; the K stream did not). D is constexpr so the loop was
                     * already fully unrolled, but each k8 fed its dot8 immediately and ptxas kept
                     * only ~1 load in flight under the register cap -- flash_decode measured
                     * 237 GB/s, 14x off its roofline. The d order and the g order are unchanged,
                     * so every dot[g] is BIT-IDENTICAL. D % (8*KUN) == 0 for D in {256,512}. */
#pragma unroll
                    for (int d0 = 0; d0 < D; d0 += 8 * PLOW_NV_FA_KUN) {
                        bf16v8 k8[PLOW_NV_FA_KUN];
#pragma unroll
                        for (int u = 0; u < PLOW_NV_FA_KUN; u++)
                            k8[u] = ld_glob8_cs(krow + d0 + u * 8);
#pragma unroll
                        for (int u = 0; u < PLOW_NV_FA_KUN; u++)
#pragma unroll
                            for (int g = 0; g < GF; g++)
                                dot[g] = dot8(k8[u], ld_smem8(qsm + g * D + d0 + u * 8), dot[g]);
                    }
#else
#pragma unroll
                    for (int d = 0; d < D; d += 8) {
                        const bf16v8 k8 = ld_glob8_cs(krow + d); /* KV: read once, evict first */
#pragma unroll
                        for (int g = 0; g < GF; g++) dot[g] = dot8(k8, ld_smem8(qsm + g * D + d), dot[g]);
                    }
#endif
#pragma unroll
                    for (int g = 0; g < GF; g++) s[g] = dot[g] * FA_SCALE(scale);
                }
            }
#pragma unroll
            for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + tid] = s[g];
            __syncthreads();
          }

            /* GF softmax reductions, ONE PER WARP, so they run concurrently: the tile costs 3
             * barriers, not 3*GF. (8 warps, GF <= 8.) */
#if PLOW_NV_FA_GF16_BENCH
            if constexpr (GF > (int)PLOW_NV_WARPS) {
            for (unsigned g = warp; g < GF; g += PLOW_NV_WARPS) {
                float mx = FA_NEG_INF;
#if PLOW_NV_FA_REDBOUND
                for (int i = lane; i < (int)rmax_t; i += 32)
#else
                for (int i = lane; i < FA_DEC_TILE; i += 32)
#endif
                    mx = fmaxf(mx, Ssm[g * FA_DEC_TILE + i]);
                mx = warp_max32(mx);
                if (lane == 0) hmax[g] = mx;
            }
            } else
#endif
            {
            if (warp < GF) {
                float mx = FA_NEG_INF;
#if PLOW_NV_FA_REDBOUND
                for (int i = lane; i < (int)rmax_t; i += 32)
#else
                for (int i = lane; i < FA_DEC_TILE; i += 32)
#endif
                    mx = fmaxf(mx, Ssm[warp * FA_DEC_TILE + i]);
                mx = warp_max32(mx);
                if (lane == 0) hmax[warp] = mx;
            }
            }
            __syncthreads();

            /* NO NEG_INF ternaries: an EXECUTED tile always has a live lane (kv0 < hi, and every
             * kv in [lo,hi) passes the causal+window mask at decode), so hmax[g] > NEG_INF and
             * mnew[g] > NEG_INF from the first tile on. The cases the ternaries handled fall out
             * of the raw exp2: corr = ex2(NEG_INF - mnew) = 0 on the first tile (m_st still
             * NEG_INF), pe = ex2(NEG_INF - mnew) = 0 on masked lanes (s still NEG_INF) — the same
             * zeros, minus the FSETP/predicate soup. ex2.ftz flushes the <=2^-126 tail to 0. */
            float mnew[GF], corr[GF];
#pragma unroll
            for (int g = 0; g < GF; g++) {
                mnew[g] = fmaxf(m_st[g], hmax[g]);
                corr[g] = FA_EXP(m_st[g] - mnew[g]);
            }
            float pe[GF];
#pragma unroll
            for (int g = 0; g < GF; g++) pe[g] = FA_EXP(s[g] - mnew[g]);
            __syncthreads();
#if PLOW_NV_FA_TC_GQA8_HD512
            if constexpr (FA_DEC_TC_GQA8(D, GF) && !FP8KV && !SZKV) {
                if (tid < GF) hmax[tid] = corr[tid];
            }
#endif
#pragma unroll
            for (int g = 0; g < GF; g++) Ssm[g * FA_DEC_TILE + tid] = pe[g];
            __syncthreads();

#if PLOW_NV_FA_GF16_BENCH
            if constexpr (GF > (int)PLOW_NV_WARPS) {
            for (unsigned g = warp; g < GF; g += PLOW_NV_WARPS) {
                float sm = 0.0f;
#if PLOW_NV_FA_REDBOUND
                for (int i = lane; i < (int)rmax_t; i += 32) sm += Ssm[g * FA_DEC_TILE + i];
#else
                for (int i = lane; i < FA_DEC_TILE; i += 32) sm += Ssm[g * FA_DEC_TILE + i];
#endif
                sm = warp_sum32(sm);
                if (lane == 0) hsum[g] = sm;
            }
            } else
#endif
            {
            if (warp < GF) {
                float sm = 0.0f;
#if PLOW_NV_FA_REDBOUND
                for (int i = lane; i < (int)rmax_t; i += 32) sm += Ssm[warp * FA_DEC_TILE + i];
#else
                for (int i = lane; i < FA_DEC_TILE; i += 32) sm += Ssm[warp * FA_DEC_TILE + i];
#endif
                sm = warp_sum32(sm);
                if (lane == 0) hsum[warp] = sm;
            }
            }
            __syncthreads();

#pragma unroll
            for (int g = 0; g < GF; g++) {
                l_st[g] = l_st[g] * corr[g] + hsum[g];
                m_st[g] = mnew[g];
            }
#if PLOW_NV_FA_TC_GQA8_HD512
            if constexpr (FA_DEC_TC_GQA8(D, GF) && !FP8KV && !SZKV) {
                fa_decode_pv_tc_gqa8<D, GF>(tc_oacc, Ssm, tc_kv, vbase, kv0,
                                             rmax_t, kv_mask, hmax);
            } else
#endif
            {
#pragma unroll
            for (int g = 0; g < GF; g++)
#pragma unroll
                for (int u = 0; u < 8; u++) oacc[g][u] *= corr[g];

            /* o[dbase..+8) += sum_r p[r]*V[r][dbase..+8) over this thread's row-group.
             * The V row is loaded ONCE and accumulated into all GF outputs (other half of the
             * fusion). No `if (pw != 0)`: predicating the loads stops them batching. */
            constexpr int VU = FA_DEC_VU(GF);
            const unsigned rmax = (hi - kv0 < (unsigned)FA_DEC_TILE) ? (hi - kv0) : (unsigned)FA_DEC_TILE;
            unsigned r = grp;
            for (; r + (VU - 1) * NG < rmax; r += VU * NG) {
                if constexpr (FP8KV) {
                    /* fp8: this lane reads only its 8 owned head-dims (b64), decodes to f32
                     * (PLOW_FP8_FAST: no bf16 round-trip), and folds v_scale into pw once/row. */
                    float vf[VU][8];
                    float vsf[VU];
#pragma unroll
                    for (int c = 0; c < VU; c++) {
                        const unsigned rr = (kv0 + r + (unsigned)c * NG) & kv_mask;
#ifdef PLOW_FP8_FAST
                        fp8v8_to_f32(ld_glob_fp8v8(vb8 + (size_t)rr * D + dbase), vf[c]);
#else
                        const bf16v8 t = fp8v8_to_bf16v8(ld_glob_fp8v8(vb8 + (size_t)rr * D + dbase));
#pragma unroll
                        for (int u = 0; u < 8; u++) vf[c][u] = __bfloat162float(t.x[u]);
#endif
                        vsf[c] = vsc[rr];
                    }
#pragma unroll
                    for (int c = 0; c < VU; c++)
#pragma unroll
                        for (int g = 0; g < GF; g++) {
                            const float pw = Ssm[g * FA_DEC_TILE + r + (unsigned)c * NG] * vsf[c];
#pragma unroll
                            for (int u = 0; u < 8; u++)
                                oacc[g][u] = fmaf(pw, vf[c][u], oacc[g][u]);
                        }
                } else {
                    bf16v8 vv[VU];
                    float vsf[VU];
#pragma unroll
                    for (int c = 0; c < VU; c++) {
                        const unsigned rr = (kv0 + r + (unsigned)c * NG) & kv_mask;
                        if constexpr (SZKV) {
                            const unsigned char* vrow = vb8 + (size_t)rr * RB;
                            vv[c] = fa_sz12_dec8<D>(vrow, (int)dbase,
                                                    *(const unsigned*)(vrow + D / 2 + D));
                        } else {
                            vv[c] = ld_glob8_cs(vbase + (size_t)rr * D + dbase);
                        }
                        vsf[c] = 1.0f;
                    }
#pragma unroll
                    for (int c = 0; c < VU; c++)
#pragma unroll
                        for (int g = 0; g < GF; g++) {
                            const float pw = Ssm[g * FA_DEC_TILE + r + (unsigned)c * NG] * vsf[c];
#pragma unroll
                            for (int u = 0; u < 8; u++)
                                oacc[g][u] = fmaf(pw, __bfloat162float(vv[c].x[u]), oacc[g][u]);
                        }
                }
            }
            for (; r < rmax; r += NG) {
                const unsigned rr = (kv0 + r) & kv_mask;
                bf16v8 v;
                float vsf;
                if constexpr (SZKV) {
                    const unsigned char* vrow = vb8 + (size_t)rr * RB;
                    v = fa_sz12_dec8<D>(vrow, (int)dbase, *(const unsigned*)(vrow + D / 2 + D));
                    vsf = 1.0f;
                } else if constexpr (FP8KV) {
                    v = fp8v8_to_bf16v8(ld_glob_fp8v8(vb8 + (size_t)rr * D + dbase));
                    vsf = vsc[rr];
                } else {
                    v = ld_glob8_cs(vbase + (size_t)rr * D + dbase);
                    vsf = 1.0f;
                }
#pragma unroll
                for (int g = 0; g < GF; g++) {
                    const float pw = Ssm[g * FA_DEC_TILE + r] * vsf;
#pragma unroll
                    for (int u = 0; u < 8; u++)
                        oacc[g][u] = fmaf(pw, __bfloat162float(v.x[u]), oacc[g][u]);
                }
            }
            __syncthreads();
            }
        }

#if PLOW_NV_FA_TC_GQA8_HD512
        if constexpr (FA_DEC_TC_GQA8(D, GF) && !FP8KV && !SZKV) {
#pragma unroll
            for (unsigned j = 0; j < D / 64; ++j)
#pragma unroll
                for (unsigned e = 0; e < 4; ++e) {
                    const unsigned g = lane / 4 + (e / 2) * 8;
                    const unsigned d =
                        (warp * (D / 64) + j) * 8 + (lane % 4) * 2 + (e % 2);
                    if (g < GF)
                        Opart[((size_t)(b * n_head + h0 + g) * nsplit + sp) * D + d] =
                            tc_oacc[j][e];
                }
#pragma unroll
            for (unsigned g = 0; g < GF; ++g) {
                if (tid == g) {
                    float* ml =
                        mlpart + ((size_t)(b * n_head + h0 + g) * nsplit + sp) * 2;
                    ml[0] = m_st[g];
                    ml[1] = l_st[g];
                }
            }
        } else
#endif
        {
        /* Fold the NG row-group partials, one query head at a time; osm is REUSED across the GF
         * heads rather than sized GF*NG*D. Runs once per work item, so the barriers are free. */
#pragma unroll
        for (int g = 0; g < GF; g++) {
            __syncthreads();
#pragma unroll
            for (int u = 0; u < 8; u++) osm[grp * D + dbase + u] = oacc[g][u];
            __syncthreads();

            const unsigned h = h0 + (unsigned)g;
            float* op = Opart + ((size_t)(b * n_head + h) * nsplit + sp) * D;
            for (unsigned d = tid; d < D; d += PLOW_NV_THREADS) {
                float acc = 0.0f;
#pragma unroll
                for (int gg = 0; gg < NG; gg++) acc += osm[gg * D + d];
                op[d] = acc;
            }
            if (tid == 0) {
                float* ml = mlpart + ((size_t)(b * n_head + h) * nsplit + sp) * 2;
                ml[0] = m_st[g];
                ml[1] = l_st[g];
            }
        }
        }
    }
}

template <int D, int GF>
__device__ __noinline__ void d_flash_decode_slots(
    float* Opart, float* mlpart, const __nv_bfloat16* Q, const __nv_bfloat16* K,
    const __nv_bfloat16* V, const int* kv_len, unsigned n_batch, unsigned n_head,
    unsigned n_kv_head, unsigned kv_stride, unsigned window, float scale, unsigned nsplit,
    unsigned kv_mask, unsigned slice, unsigned nblk, float* lds, unsigned kv_cap,
    const int* decode_slot) {
    d_flash_decode<D, GF, false, false, true>(
        Opart, mlpart, Q, K, V, kv_len, n_batch, n_head, n_kv_head, kv_stride, window, scale,
        nsplit, kv_mask, slice, nblk, lds, kv_cap, nullptr, nullptr, decode_slot);
}

/* Combine the split partials: standard online-softmax merge, work unit = (batch, head).
 * Deliberately NOT split over the feature axis — see the AMD comment: widening the merge
 * widens its consumer's gate and made the TOKEN slower there. Kept identical here. */
template <int D>
__device__ void d_flash_merge(__nv_bfloat16* __restrict__ O, const float* __restrict__ Opart,
                              const float* __restrict__ mlpart, unsigned n_batch, unsigned n_head,
                              unsigned nsplit, unsigned slice, unsigned nblk, const int* req = nullptr) {
    /* BRANCHLESS: the old body computed the exp weight under an `== NEG_INF` guard per (d, s)
     * element — nsplit FSETP+guarded-EX2 chains per output, predicate pressure high enough that
     * ptxas spilled predicates (P2R). The guard is unnecessary: an empty split carries
     * m = NEG_INF, l = 0, so ex2(NEG_INF - gm) = 0 zeroes its term exactly as `continue` did (and
     * when ALL splits are empty, every l is 0 so gl = 0 and inv = 0, output 0 either way). With
     * FA_EXP now a single MUFU.EX2, the recompute is cheaper than any spill-prone weight array.
     * (m,l) is read as one 8-byte float2 — mlpart is [split][2] f32, 8-aligned. */
    const unsigned n_work = n_batch * n_head;
    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned h = w % n_head, b = w / n_head;
#if PLOW_NV_PACKED_REQUEST
        if (req && b >= (unsigned)(req[1 + 4 * (req[0]-1)] + req[2 + 4 * (req[0]-1)])) {
            for (unsigned d=threadIdx.x; d<D; d+=PLOW_NV_THREADS)
                O[((size_t)b*n_head+h)*D+d]=__float2bfloat16(0.0f);
            continue;
        }
#else
        if (req) __trap();
#endif
        const float2* ml2 = (const float2*)(mlpart + (size_t)(b * n_head + h) * nsplit * 2);

        float gm = FA_NEG_INF;
        for (unsigned s = 0; s < nsplit; s++) gm = fmaxf(gm, ml2[s].x);
        float gl = 0.0f;
        for (unsigned s = 0; s < nsplit; s++) {
            const float2 ml = ml2[s];
            gl += ml.y * FA_EXP(ml.x - gm);
        }
        const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

        for (unsigned d = threadIdx.x; d < D; d += PLOW_NV_THREADS) {
            float acc = 0.0f;
            for (unsigned s = 0; s < nsplit; s++)
                acc += Opart[((size_t)(b * n_head + h) * nsplit + s) * D + d] * FA_EXP(ml2[s].x - gm);
            O[((size_t)b * n_head + h) * D + d] = __float2bfloat16(acc * inv);
        }
    }
}

/* ---- PREFILL flash (FA-2 tiling, mma.sync QK^T scores) -------------------------------------------
 * Multi-query-row causal / sliding-window attention for hd 256 (sliding) and 512 (full, k_eq_v).
 * (a) HD 256/512 templated; (b) SPLIT-KV so a short prompt fills the machine — a work item is
 * (q_tile, head, split) and writes the same (Opart f32, mlpart f32) partials d_flash_merge folds,
 * so the merge op above is the exact partner; (c) at nsplit==1 there is nothing to merge, so the
 * epilogue normalises in place and writes the final bf16 straight to `O` (the emitter's `n.at`),
 * and no FLASH_MERGE is emitted; (d) head-major RING KV cache (kv_stride + kv_mask), matching the
 * cache d_headnorm_rope wrote; (e) q_pos0 for chunked prefill (a chunk's rows sit at absolute
 * positions q_pos0 + local).
 *
 * T1 (rtx-05): scores S = Q.K^T are computed with mma.sync m16n8k16 (bf16, f32-acc) TENSOR CORES,
 * the exact operand layout d_gemm validates (A=Q row-major, B=K^T staged transposed, ldmatrix). The
 * whole BQ x BKV score tile lands in smem `Ss` in ONE tensor-core pass — replacing the first-cut
 * kernel's serial per-KV-element warp_sum32 dot (BKV latency-bound shuffle reductions per query row,
 * the measured bottleneck, G5).
 *
 * T4 (rtx-05): the P.V accumulation is now ALSO mma.sync m16n8k16 tensor cores (was FFMA-serial: one
 * warp per BQ/WARPS query rows streaming Vs scalar*vector — the O(ctx^2) long-ctx tail, the 25x@128k).
 * A = P (softmax probs) staged bf16 into `Ps` [query][kv] (ldmatrix.x4, same layout as QK's A=Q); B =
 * V staged NATURAL in Vs[kv][hd] (ldmatrix.x2.trans, EXACTLY KsT's B-load structure, so V needs no
 * extra transpose). O is register-resident in the mma f32 accumulator; online-softmax rescale hits the
 * accumulator fragment directly (per-query-row corr, mapped to the D-fragment (row,col) ownership).
 *
 * HD-SPLIT WARP PARTITION (T4): O[BQ][HD] is too big for one warp, so P.V splits HD across warps —
 * WPV_M = BQ/16 query-warp-rows x WPV_N hd-warp-cols = all 8 warps, each owning a 16(query) x 128(hd)
 * O tile = 64 f32 acc/lane (identical budget to the retired FFMA o[RPW][DE]). The three phases own
 * DIFFERENT warp grids (QK: query x kv; softmax: BQ/WARPS rows/warp; P.V: query x hd), so the online
 * state m/l/corr lives in SMEM, not registers, letting each phase pick its natural partition.
 *
 * QK warp grid: WQK_M = BQ/16 query-warps x WQK_N=2 kv-warps (8 warps at hd256, 4 at hd512; the rest
 * idle in QK, all 8 run softmax + P.V). Each active warp owns a 16(query) x (BKV/2)(kv) score block,
 * contracting the full HD in k16 steps. Ss carries the scaled scores; the softmax phase applies the
 * causal/sliding mask (masked/pad entries get P=0, excluded from the row max).
 *
 * Q is token-major [seq_q][n_head][hd]; K/V head-major [n_kv_head][kv_stride][hd] (RING). Opart is
 * [qrow][head][split][hd], mlpart [qrow][head][split][2] (m,l log2-domain) — b=qrow, EXACTLY the
 * (n_batch:=seq_q) layout d_flash_merge indexes. */
#define FA_PRE_PAD 8
/* smem bf16: Qs[BQ][HD+PAD] + K + Vs[BKV][HD+PAD] + Ps[BQ][BKV+PAD] (softmax probs, mma P.V A — T4).
 *   PIPE=0 (T4): K staged TRANSPOSED KsT[HD][BKV+PAD] (ldmatrix.x2.trans B operand).
 *   PIPE=1 (T5): K staged NATURAL Ks[BKV][HD+PAD] (cp.async-friendly, ldmatrix.x2 non-.trans);
 *               strictly SMALLER, and single-buffered like T4 (the +33 KiB double-buffer does not
 *               fit the 99 KiB opt-in cap — see fa_cp_* header).
 * smem f32: Ss[BQ][BKV] (the scaled QK^T tile) + m[BQ] + l[BQ] + corr[BQ] (per-row online-softmax
 *           state, smem-resident so the QK/softmax/P.V phases can own different warp partitions). */
#if PLOW_NV_FA_PIPE
#define FA_PRE_KBUF(HD, BKV) ((BKV) * ((HD) + FA_PRE_PAD))   /* Ks[kv][hd] natural */
/* rtx-07 T7 L1: TRUE 2-stage V double-buffer on the hd512 FULL-attention arm only. That arm
 * (BQ32/BKV16 = 68.9 KiB) carries the dominant O(ctx^2) 128k cost and is HBM-KV-read-bound (T5).
 * A second V buffer lets V[t+1] stream gmem->smem DURING P.V[t] compute instead of stalling at the
 * tile boundary (the single-buffer scheme could only start V[t+1] AFTER P.V[t] freed Vs). K is NOT
 * doubled: the single-buffer pipeline already issues K[t+1] right after QK[t], so K already overlaps
 * softmax[t]+P.V[t]; the tile-boundary stall is V's. Doubling BOTH K and V (+32.5 KiB -> 101.4 KiB)
 * overruns the 99 KiB opt-in cap; V-only (+16.25 KiB -> 85.1 KiB) fits with margin and captures the
 * boundary stall. hd256 sliding arm stays single-buffered (VBUF=1, T5 byte-identical).
 *
 * MEASURED NEGATIVE (T7 A/B, 12B): the V double-buffer is a WASH at 32k/64k (−0.06% / −0.30%) and
 * +2.2% SLOWER at 128k — the ctx it was meant to help. The 128k full-attention flash is
 * HBM-BANDWIDTH-bound (T5's own finding), not tile-boundary-LATENCY-bound, so a second V buffer hides
 * a latency that is not the wall while its extra smem (85 vs 79.75 KiB) + extra cp.async commits cost.
 * So the DEFAULT is 0 (T5 single-buffer, the winner, byte-identical numerics — first-gen-token
 * identical A/B at 32k/64k/128k). Build -DPLOW_NV_FA_VDBUF=1 to opt into the double-buffer path. */
#ifndef PLOW_NV_FA_VDBUF
#define PLOW_NV_FA_VDBUF 0
#endif
#define FA_PRE_VBUF(HD) ((PLOW_NV_FA_VDBUF && (HD) == 512) ? 2 : 1)
#else
#define FA_PRE_KBUF(HD, BKV) ((HD) * ((BKV) + FA_PRE_PAD))   /* KsT[hd][kv] transposed */
#define FA_PRE_VBUF(HD) 1
#endif
#define FA_PRE_BF16(HD, BQ, BKV)                                                                    \
    ((BQ) * ((HD) + FA_PRE_PAD) + FA_PRE_KBUF(HD, BKV) +                                            \
     FA_PRE_VBUF(HD) * (BKV) * ((HD) + FA_PRE_PAD) + (BQ) * ((BKV) + FA_PRE_PAD))
/* ---- PX-4 (rtx-11): restructured hd512 FULL-layer arm ---------------------------------------
 * Ablation (px4_fa_ablate.cu, RTX PRO 6000, ncu unavailable — RmProfilingAdminOnly=1) attributes
 * the T5 kernel's 2.0 us/tile as: softmax 36% (RPW_S-serial 32-lane shuffle reductions + Ps smem
 * round-trip), QK 21% (only 4 of 8 warps active, 32-deep dependent mma chains), staging exposure
 * 16%, P.V 6%, barriers 3% — NOT DRAM-bound (per-tile time flat 8k->128k). PX4 therefore:
 *   (a) REGISTER softmax fused into the P.V A-fragment: each P.V warp computes P for its own
 *       16 query rows directly in the mma A layout (quad shfl row-reductions, m/l/corr in regs,
 *       Ps buffer + its ldmatrix + one barrier GONE);
 *   (b) 8-warp QK: the HD-512 contraction splits across two warp groups (hd 0..255 / 256..511)
 *       into SsA+SsB, summed at the softmax read — halves the dependent-chain depth and puts the
 *       4 QK-idle warps to work;
 *   (c) optional TMA staging (PLOW_NV_FA_TMA=1): per-row cp.async.bulk + mbarrier instead of
 *       per-thread cp.async.cg lines (measured A/B; see perf-data/px4-flash-streaming.*).
 * hd256 (sliding) keeps the T5 path byte-identical; PLOW_NV_FA_PX4=0 restores T5 everywhere. */
#ifndef PLOW_NV_FA_PX4
#define PLOW_NV_FA_PX4 1
#endif
#ifndef PLOW_NV_FA_TMA
#define PLOW_NV_FA_TMA 0
#endif
#define FA_PX4_ELIGIBLE(HD) (PLOW_NV_FA_PIPE && PLOW_NV_FA_PX4 && (HD) == 512)
/* ---- fp8-KV FAST-prefill staging (beat-fp8-prefill Exp1) ------------------------------------
 * The PIPE=1 cp.async ring cannot dequant e4m3 inline, so the fp8 arm stages RAW e4m3 bytes into
 * an extra pair of uint8 smem tiles (Ks8/Vs8, row stride HD + 16B pad) and dequants stage t to the
 * bf16 Ks/Vs the mma already reads, AFTER its cp_wait, while stage t+1 streams — half the HBM
 * bytes vs the bf16 KV, pipeline structure otherwise unchanged. These extra tiles exist ONLY in
 * the fp8 object (compiled -DPLOW_FP8_KV=1); the bf16 arena (no PLOW_FP8_KV) is byte-identical. */
#if defined(PLOW_FP8_KV)
#define FA_FP8_PAD8 32 /* byte pad per uint8 staging row: multiple of 16 (cp.async line) AND
                        * row stride HD+32 == 32 mod 128 -> the fp8mma u32/uint2 fragment reads
                        * hit banks 8r+2c per half-warp = CONFLICT-FREE (16 was 2-way). */
/* 2 uint8 tiles (K,V), BKV rows x (HD+pad) bytes, rounded up to a float count. */
#define FA_FP8_STAGE_FLOATS(HD, BKV) ((2 * (BKV) * ((HD) + FA_FP8_PAD8) + 3) / 4)
#define FA_FP8_PAD8_K FA_FP8_PAD8
#else
#define FA_FP8_STAGE_FLOATS(HD, BKV) 0
#define FA_FP8_PAD8_K 32
#endif
/* ---- fp8 QK^T mma (beat-fp8-mma Lever A, fp8-KV arm only) -----------------------------------
 * The NO-GO root cause was dequant-to-bf16 ON the critical path. Lever A deletes the K half of
 * it: the QK^T runs as mma.m16n8k32.e4m3 (2x rate, half the dependent-chain depth) consuming the
 * RAW e4m3 Ks8 tile directly — no dequantK pass, no dequant barrier. Q is quantized per-row to a
 * new Qs8 tile ONCE per q-tile (amortized over the whole KV loop); the row scale multiplies the
 * score with the existing per-column k-scale. The bf16 Ks tile is DEAD in this arm and its smem
 * is re-used for Qs8 [BQ][HD+PAD8] + qsc[BQ] f32 (net +96 floats at hd512/BQ32/BKV16). V still
 * dequants to bf16 for the P.V mma (fp8 P.V needs BKV=32 to fill k32 — Lever C/D). Compiled only
 * into fp8 objects (-DPLOW_FP8_KV=1) and gated: -DPLOW_NV_FA_FP8MMA=0 restores the NO-GO arm. */
#ifndef PLOW_NV_FA_FP8MMA
#if defined(PLOW_FP8_KV)
#define PLOW_NV_FA_FP8MMA 1
#else
#define PLOW_NV_FA_FP8MMA 0
#endif
#endif
/* fp8mma V-dequant: OWN-BYTES, barrier-free. stageV ownership is per-thread (thread T stages
 * 16B lines L = T, T+256, ...), so after ITS OWN cp.async wait each thread dequants exactly the
 * bytes it staged — no visibility barrier needed before the dequant, and the post-softmax
 * __syncthreads publishes the bf16 Vs for the P.V ldmatrix. This puts the fp8 arm at the SAME
 * 4 barriers/tile as bf16 (the NO-GO arm paid 5 + a serialized block-wide dequant pass; a
 * per-warp fused variant was also measured SLOWER — 2x duplicated conversions serial in front
 * of each warp's ldmatrix). */
/* DIAGNOSTIC bitmask (microbench-only, default 0 = full kernel; NEVER set on a shipped build —
 * nonzero bits break numerics on purpose to attribute the fp8mma arm's per-tile cost):
 * bit0 skip dequantV+barrier; bit1 skip the per-q-tile Q quant; bit2 skip QK mma+store;
 * bit3 skip scale staging + vsc fold. */
#ifndef PLOW_NV_FA_FP8ABL
#define PLOW_NV_FA_FP8ABL 0
#endif
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
#if PLOW_NV_FA_TMA
#error "PLOW_NV_FA_FP8MMA is cp.async-staging only (the TMA arm stages bf16 Ks)"
#endif
/* Qs8 bytes + qsc f32 + per-tile ksc/vsc staging (BKV f32 each), replacing the dead bf16 Ks
 * tile in the fp8mma arm. */
#define FA_PX4_KS_OR_Q8_FLOATS(HD, BQ, BKV)                                                         \
    (((BQ) * ((HD) + FA_FP8_PAD8) + 3) / 4 + (BQ) + 2 * (BKV))
#else
#define FA_PX4_KS_OR_Q8_FLOATS(HD, BQ, BKV) (((BKV) * ((HD) + FA_PRE_PAD) + 1) / 2)
#endif
/* px4 smem: mbar[2] u64 (4 floats, present in both staging arms so the layout is A/B-stable) +
 * SsA/SsB [BQ][BKV] f32 + Qs[BQ][HD+PAD] + Ks/Vs[BKV][HD+PAD] bf16. No Ps, no m/l/corr arrays.
 * fp8 (PLOW_FP8_KV) additionally reserves the raw-e4m3 Ks8/Vs8 staging pair; the fp8mma arm
 * (PLOW_NV_FA_FP8MMA) swaps the dead bf16 Ks for Qs8+qsc (FA_PX4_KS_OR_Q8_FLOATS). */
#define FA_PX4_SMEM_FLOATS(HD, BQ, BKV)                                                             \
    (4 + 2 * (BQ) * (BKV) +                                                                         \
     ((BQ) * ((HD) + FA_PRE_PAD) + (BKV) * ((HD) + FA_PRE_PAD) + 1) / 2 +                           \
     FA_PX4_KS_OR_Q8_FLOATS(HD, BQ, BKV) + FA_FP8_STAGE_FLOATS(HD, BKV))
/* ---- PX-8: e4m3 P.V at BKV=32 (perf-data/px8-flash-fp8-pv.md) --------------------------------
 * The px4 fp8mma arm runs QK as mma.m16n8k32.e4m3 but dequants V to fp16 for a m16n8k16 P.V, so
 * the P.V costs 4x the tensor-core time of the QK for the same MACs. PX-8 makes it e4m3 too:
 * BKV 16 -> 32 fills the k32, V stays RAW e4m3 in smem (the dequant pass is deleted outright) and
 * the B operand comes out of fa_ldmatrix_x2_trans_b8 off a permuted-row V tile.
 *
 * NOT BQ=64. PX-7 Result 5 bundled BQ=64 into this change; it does not fit. oacc is
 * BQ*HD/(WARPS*32) f32 per lane = 64 at BQ=32 and 128 at BQ=64 (PLOW_NV_THREADS is fixed at 256),
 * measured +64 registers, which puts this arm past the 255 cap and spills accumulators that are
 * live across the whole KV loop. BQ=64 at hd512 needs a 512-thread block.
 *
 * DEFAULT 0: sm_120a-only (8-bit ldmatrix) and it changes numerics (P is quantised to e4m3), so it
 * is opt-in. -DPLOW_NV_FA_FP8PV=1 on an fp8 object with the fp8mma arm selects it. */
#if PLOW_NV_FA_FP8PV && !(PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV) && PLOW_NV_FA_PIPE)
#error "PLOW_NV_FA_FP8PV needs the fp8mma arm: -DPLOW_FP8_KV=1 with PLOW_NV_FA_PIPE=1"
#endif
#define FA_PX8_ELIGIBLE(HD) (PLOW_NV_FA_FP8PV && (HD) == 512)
/* mbar[2] (kept for layout stability) + SsA/SsB + Qs bf16 + qsc + ksc + vsc + Qs8 + Ks8 + Vs8.
 * No bf16 Ks and no bf16 Vs at all — this arm never materialises a dequanted K or V. */
#define FA_PX8_SMEM_FLOATS(HD, BQ, BKV)                                                             \
    (4 + 2 * (BQ) * (BKV) + ((BQ) * ((HD) + FA_PRE_PAD) + 1) / 2 + (BQ) + 2 * (BKV) +                \
     ((BQ) * ((HD) + FA_FP8_PAD8_K) + 3) / 4 + 2 * (((BKV) * ((HD) + FA_FP8_PAD8_K) + 3) / 4))

/* ---- PX-23: the hd256 SLIDING-layer fp8 fast prefill arm ------------------------------------
 * An ALL-LAYER e4m3 packet (what vLLM ships by default) emits hd256 FLASH_PREFILL_FP8, and until
 * this arm existed the PIPE=1 fp8 object trapped on it — so the whole packet fell to the PIPE=0
 * synchronous-staging path at 176 s of prefill per 127k request (PX-20 §4d) against 34.9 s for the
 * same model at bf16. That one missing arm is the whole of PX-20's 7.61x conc-8 gap.
 *
 * It is a RETILE, not an instantiation of px4. Two reasons, both measured before any body was
 * written (perf-data/px23-hd256-fp8-prefill.md):
 *   (1) FA_PX4_SMEM_FLOATS(256,64,32) claims 104,464 B against a 101,376 B optin cap. The naive
 *       retile does not fail at COMPILE time — it fails as a refused module load at serve time.
 *   (2) px4's 8-warp QK grid splits HD in half (2 x 2 x 2) only because at BQ=32/BKV=16 the
 *       query x kv tile is 4 warp-blocks and it needs 8. At BQ=64/BKV=32 that tile is 4 x 4, so
 *       the hd split — and with it SsB and the SsA+SsB add at every softmax read — is dead weight.
 *
 * BQ=64 (not px4's 32) costs exactly what the shipped arm already pays: oacc is BQ*HD/THREADS f32
 * per lane = 64 at BOTH hd512/BQ32 and hd256/BQ64. PX-8 Result 3's BQ=64 register wall (128
 * f32/lane) was an hd512 property, not a BQ property. BQ=64 doubles arithmetic intensity per KV
 * byte (2*BQ FLOP/B), which matters here in a way it did not for px4: the sliding layers re-read a
 * whole `window` of KV per q-tile, so unlike PX-8's full-attention arm this one IS traffic-exposed.
 * BKV=32 halves the tile count, hence half the barriers and half the loop floor (PX-8 Result 10
 * puts loop+barrier at 18% and cp.async exposure at 24.5% of the hd512 fp8 arm).
 *
 * smem: px4 carries a bf16 Qs tile that is DEAD in the fp8mma arm — Q is staged bf16 only to be
 * read straight back by the per-q-tile quant, after which nothing reads it. px23 drops it and
 * quantizes Q out of REGISTERS straight from gmem. Layout: Ss | Vs | qsc | ksc | vsc | Qs8 | Ks8 |
 * Vs8 = 60,416 B at (256,64,32) — 40% under the cap AND under the fp8 object's existing 89,104 B
 * px4 claim, so PLOW_NV_PRE_A does not move and the shipped arena is unchanged.
 *
 * Qs8 is [BQ*HD] bytes in A-FRAGMENT order with NO row pad: it is a pure fragment array indexed
 * ((qm*KSTEPS8 + kf) << 9) + lane*16, never a [row][col] tile. */
#define FA_PX23_ELIGIBLE(HD)                                                                        \
    (PLOW_NV_FA_PIPE && PLOW_NV_FA_PX4 && PLOW_NV_FA_FP8MMA && (HD) == 256)
#define FA_PX23_SMEM_FLOATS(HD, BQ, BKV)                                                            \
    ((BQ) * (BKV) + ((BKV) * ((HD) + FA_PRE_PAD) + 1) / 2 + (BQ) + 2 * (BKV) +                      \
     ((BQ) * (HD) + 3) / 4 + FA_FP8_STAGE_FLOATS(HD, BKV))
/* NOTE: the raw-e4m3 staging pair exists ONLY in the px4 (hd512 PIPE=1) arm. The generic arm
 * never stages e4m3 smem tiles — PIPE=0 dequants inline from gmem, and the PIPE=1 generic
 * kernel has no fp8 arm — so its arena must NOT carry FA_FP8_STAGE_FLOATS: with it, the
 * hd256 (BQ64/BKV32) claim crosses the 99 KiB opt-in cap and the cooperative prefill grid
 * collapses to 0 blocks (FATAL: prefill grid 0). */
/* PX-8 runs the hd512 arm at BKV=32, so its claim is taken at (BQ, 2*BKV) and the arena is the max
 * of the two (the object still carries px4 for the A/B). 94,096 B at BQ32/BKV32 vs px4's 89,104 —
 * both are occ-1 and both fit the 101,376 B cap. */
#define FA_PX8_CLAIM(HD, BQ, BKV)                                                                   \
    (FA_PX8_ELIGIBLE(HD) ? FA_PX8_SMEM_FLOATS(HD, BQ, 2 * (BKV)) : 0)
#define FA_PRE_SMEM_BASE(HD, BQ, BKV)                                                               \
    (FA_PX4_ELIGIBLE(HD)                                                                            \
         ? (FA_PX8_CLAIM(HD, BQ, BKV) > FA_PX4_SMEM_FLOATS(HD, BQ, BKV)                             \
                ? FA_PX8_CLAIM(HD, BQ, BKV)                                                         \
                : FA_PX4_SMEM_FLOATS(HD, BQ, BKV))                                                  \
         : ((BQ) * (BKV) + 3 * (BQ) + (FA_PRE_BF16(HD, BQ, BKV) + 1) / 2))
/* ---- sm_90a FORK: the hd256 prefill arm runs on warpgroup MMA ------------------------------
 * op_attention_sm90.cuh replaces d_flash_prefill's per-tile math for the shapes it claims
 * (FA_SM90_WG_ELIGIBLE — today only <256,64,32>) with the oracle-validated wgmma flash prefill.
 * Its 128B-swizzled operand tiles + a 2-deep K/V pipeline claim more smem than the mma.sync
 * layout, so the arena high-water mark takes the max. Everything else — hd512/px4, the PIPE=0
 * fp8-KV arm, and every sm_120a build — is untouched and byte-identical. */
#if defined(PLOW_NV_HOPPER)
#include "op_attention_sm90.cuh"
#define FA_PRE_SMEM_FLOATS(HD, BQ, BKV)                                                             \
    (FA_SM90_WG_ELIGIBLE(HD, BQ, BKV) && PLOW_NV_FA_PIPE                                            \
         ? (FA_SM90_PRE_FLOATS(HD, BQ, BKV) > FA_PRE_SMEM_BASE(HD, BQ, BKV)                         \
                ? FA_SM90_PRE_FLOATS(HD, BQ, BKV)                                                   \
                : FA_PRE_SMEM_BASE(HD, BQ, BKV))                                                    \
         : FA_PRE_SMEM_BASE(HD, BQ, BKV))
#else
#define FA_PRE_SMEM_FLOATS(HD, BQ, BKV) FA_PRE_SMEM_BASE(HD, BQ, BKV)
#endif

#if !PLOW_NV_FA_PIPE
template <int HD, int BQ, int BKV, bool FP8KV = false>
__device__ void d_flash_prefill(float* __restrict__ Opart, float* __restrict__ mlpart,
                                const __nv_bfloat16* __restrict__ Q,
                                const __nv_bfloat16* __restrict__ K,
                                const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
                                unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head,
                                unsigned q_pos0, unsigned window, unsigned nsplit,
                                unsigned kv_stride, unsigned kv_mask, float scale, unsigned slice,
                                unsigned nblk, float* lds,
                                const float* __restrict__ k_scale = nullptr,
                                const float* __restrict__ v_scale = nullptr) {
    static_assert(HD % 32 == 0, "HD must be a multiple of the warp width");
    static_assert(HD % 16 == 0 && BQ % 16 == 0 && BKV % 16 == 0, "mma m16n8k16 tiling");
    constexpr int PAD = FA_PRE_PAD;
    /* QK^T tensor-core grid. */
    constexpr int WQK_N = 2;                      /* kv warp columns */
    constexpr int WQK_M = BQ / 16;                /* query warp rows (each owns 16 query rows) */
    constexpr int WN = BKV / WQK_N;               /* kv cols per warp (multiple of 8) */
    constexpr int NJ = WN / 8;                    /* n8 sub-tiles per warp */
    constexpr int KSTEPS = HD / 16;               /* k16 contraction steps */
    static_assert(WN % 8 == 0, "kv cols per warp must be a multiple of 8");
    static_assert(WQK_M * WQK_N <= (int)PLOW_NV_WARPS, "QK warp grid exceeds the block");
    /* P.V tensor-core grid (T4): split HD across warps, each owns 16 query x HDW hd (64 f32 acc/lane). */
    constexpr int WPV_M = BQ / 16;                     /* query warp rows */
    constexpr int WPV_N = (int)PLOW_NV_WARPS / WPV_M;  /* hd warp cols */
    constexpr int HDW = HD / WPV_N;                    /* hd slice per warp (128 at both hd256/hd512) */
    constexpr int NJ_PV = HDW / 8;                     /* n8 hd sub-tiles per warp */
    constexpr int KSTEPS_PV = BKV / 16;                /* k16 contraction over kv */
    static_assert(WPV_M * WPV_N == (int)PLOW_NV_WARPS, "P.V grid must use all warps");
    static_assert(HD % WPV_N == 0 && HDW % 8 == 0, "hd slice must be a multiple of 8");
    /* Softmax phase: each warp owns RPW_S query rows. BKV <= 32: one lane per kv column.
     * BKV > 32: each lane owns BKV/32 columns spaced by 32, reduced locally before the warp op. */
    constexpr int RPW_S = BQ / (int)PLOW_NV_WARPS;
    constexpr int SOFT_COLS = BKV > 32 ? BKV / 32 : 1;
    static_assert(BKV <= 64, "softmax reduction supports at most 2 kv cols per lane");
    static_assert(BKV <= 32 || BKV % 32 == 0, "BKV > 32 must be a multiple of 32");

    float* Ss = lds;                                       /* [BQ][BKV] scaled scores */
    float* m_arr = Ss + BQ * BKV;                          /* [BQ] running max (log2 domain) */
    float* l_arr = m_arr + BQ;                             /* [BQ] running denom */
    float* corr_arr = l_arr + BQ;                          /* [BQ] this-tile rescale factor */
    __nv_bfloat16* Qs = (__nv_bfloat16*)(corr_arr + BQ);   /* [BQ][HD+PAD] */
    __nv_bfloat16* KsT = Qs + BQ * (HD + PAD);             /* [HD][BKV+PAD] transposed K */
    __nv_bfloat16* Vs = KsT + HD * (BKV + PAD);            /* [BKV][HD+PAD] */
    __nv_bfloat16* Ps = Vs + BKV * (HD + PAD);             /* [BQ][BKV+PAD] softmax probs (P.V A operand) */

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned gqa = n_head / n_kv_head;
    const unsigned n_qt = (seq_q + BQ - 1) / BQ;
    const unsigned n_work = n_qt * n_head * nsplit;
    const float lscale = FA_SCALE(scale);
    /* This warp's QK tile origin (query row base, kv col base) — only WQK_M*WQK_N warps are active. */
    const int qk_wm = warp / WQK_N, qk_wn = warp % WQK_N;
    const bool qk_active = warp < WQK_M * WQK_N;
    /* This warp's P.V tile origin (query row base, hd col base) — all warps active. */
    const int pv_wm = warp / WPV_N, pv_wn = warp % WPV_N;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned h = (w / nsplit) % n_head;
        const unsigned qt = w / (nsplit * n_head);
        const unsigned q0 = qt * BQ;
        const unsigned hkv = h / gqa;

        /* This split's KV slice [lo, hi). Causal/sliding masks trim it per row. */
        const unsigned per = (seq_kv + nsplit - 1) / nsplit;
        const unsigned lo = sp * per;
        const unsigned hi = (lo + per < seq_kv) ? (lo + per) : seq_kv;

        const __nv_bfloat16* Qh = Q + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const __nv_bfloat16* Kb = K + (size_t)hkv * kv_stride * HD;
        const __nv_bfloat16* Vb = V + (size_t)hkv * kv_stride * HD;
        /* fp8-KV: byte bases + per-row scale slice (unused on the bf16 instantiation). */
        const unsigned char* Kb8 = (const unsigned char*)K + (size_t)hkv * kv_stride * HD;
        const unsigned char* Vb8 = (const unsigned char*)V + (size_t)hkv * kv_stride * HD;
        /* Offset only on FP8KV — null+offset on the bf16 arm is UB (see the decode flash note). */
        const float* ksc = nullptr;
        const float* vsc = nullptr;
        if constexpr (FP8KV) {
            ksc = k_scale + (size_t)hkv * kv_stride;
            vsc = v_scale + (size_t)hkv * kv_stride;
        }

        __syncthreads(); /* previous item's Qs/KsT/Vs reads done before restage */
        /* Stage this q-tile's Q (one head) resident. */
        for (int idx = tid; idx < BQ * HD; idx += (int)PLOW_NV_THREADS) {
            int r = idx / HD, c = idx % HD;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (q0 + r < seq_q) v = Qh[(size_t)r * n_head * HD + c];
            Qs[r * (HD + PAD) + c] = v;
        }
        __syncthreads();

        /* O accumulator: register-resident mma f32 fragment, this warp's 16(query) x HDW(hd) tile. */
        float oacc[NJ_PV][4];
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) oacc[nj][e] = 0.0f;
        /* Online-softmax state is smem-resident (the phases own different warp partitions). */
        for (int r = tid; r < BQ; r += (int)PLOW_NV_THREADS) {
            m_arr[r] = FA_NEG_INF;
            l_arr[r] = 0.0f;
        }
        __syncthreads();

        /* Newest absolute query position in this q-tile (upper bound; pad rows are dropped in the
         * epilogue). Drives whole-tile skips so a sliding layer is O(window), not O(ctx). */
        const int qabs_max = (int)(q_pos0 + q0 + BQ - 1);
        /* Jump straight to the first attended tile on a sliding layer: the OLDEST query in the tile
         * (qabs_min = q_pos0+q0) attends nothing below qabs_min-window+1, so start the loop there.
         * Align down to BKV, never below `lo`. The union of every query's window over the tile is the
         * CONTIGUOUS range [qabs_min-window+1, qabs_max], so from eff_lo up to the causal break every
         * tile is attended by SOME query — per-row masking (softmax phase) drops the rest. */
        unsigned eff_lo = lo;
        if (window) {
            const long wfloor = (long)q_pos0 + q0 - (long)window + 1;
            if (wfloor > (long)lo) eff_lo = ((unsigned)wfloor / BKV) * (unsigned)BKV;
        }
        for (unsigned kv0 = eff_lo; kv0 < hi; kv0 += BKV) {
            /* CAUSAL: once a KV tile starts beyond the newest query, no later tile is attended. */
            if ((int)kv0 > qabs_max) break;
            /* NO per-tile sliding skip here: the earlier "skip a tile below the NEWEST query's window
             * floor (qabs_max - (kv0+BKV-1) >= window)" was WRONG — it dropped tiles the OLDER queries
             * in the tile still attend (the trailing-window edge, ~BQ positions), silently corrupting
             * every sliding layer. It was masked by a degenerate oracle ref; the fixed ref catches it. */
            /* Stage a KV tile out of the RING cache: K TRANSPOSED into KsT[hd][kv] (the mma B
             * operand layout), V natural into Vs[kv][hd] (the FFMA P.V layout). */
            for (int idx = tid; idx < BKV * HD; idx += (int)PLOW_NV_THREADS) {
                int r = idx / HD, c = idx % HD;
                unsigned kv = kv0 + (unsigned)r;
                __nv_bfloat16 kk = __float2bfloat16(0.f), vv = __float2bfloat16(0.f);
                if (kv < hi) {
                    const size_t row = (size_t)(kv & kv_mask);
                    if constexpr (FP8KV) {
                        /* fp8: DEQUANTIZE at the smem stage — e4m3 -> f32 ×per-row scale -> bf16 — so
                         * the mma below reads bf16 exactly as the bf16 path. HBM traffic is halved. */
                        kk = __float2bfloat16(fp8_to_f32(Kb8[row * HD + c]) * ksc[row]);
                        vv = __float2bfloat16(fp8_to_f32(Vb8[row * HD + c]) * vsc[row]);
                    } else {
                        kk = Kb[row * HD + c];
                        vv = Vb[row * HD + c];
                    }
                }
                KsT[c * (BKV + PAD) + r] = kk;
                Vs[r * (HD + PAD) + c] = vv;
            }
            __syncthreads();

            /* S = Q.K^T for this tile via mma.sync tensor cores, scaled, written to Ss[BQ][BKV]. */
            if (qk_active) {
                float acc[NJ][4];
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[nj][e] = 0.f;
#pragma unroll
                for (int kf = 0; kf < KSTEPS; kf++) {
                    unsigned af[4];
                    fa_ldmatrix_x4(af, &Qs[(qk_wm * 16 + (lane % 16)) * (HD + PAD) +
                                          kf * 16 + (lane / 16) * 8]);
                    unsigned bf[NJ][2];
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++)
                        fa_ldmatrix_x2_trans(bf[nj], &KsT[(kf * 16 + (lane % 16)) * (BKV + PAD) +
                                                          qk_wn * WN + nj * 8]);
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++) fa_mma(acc[nj], af, bf[nj], acc[nj]);
                }
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        int qr = qk_wm * 16 + (lane / 4) + (e / 2) * 8;
                        int kc = qk_wn * WN + nj * 8 + (lane % 4) * 2 + (e % 2);
                        Ss[qr * BKV + kc] = acc[nj][e] * lscale;
                    }
            }
            __syncthreads();

            const unsigned rmax = (hi - kv0 < (unsigned)BKV) ? (hi - kv0) : (unsigned)BKV;

            /* SOFTMAX phase: each warp owns RPW_S query rows. When BKV <= 32, lane == kv column
             * (original path). When BKV == 64, each lane owns 2 kv columns (lane and lane+32),
             * reduced locally before the warp max/sum. */
#pragma unroll
            for (int rr = 0; rr < RPW_S; rr++) {
                const int row = warp * RPW_S + rr;
                const int qabs = (int)(q_pos0 + q0 + row);
                float sv[SOFT_COLS];
                bool active[SOFT_COLS];
#pragma unroll
                for (int sc = 0; sc < SOFT_COLS; sc++) {
                    sv[sc] = FA_NEG_INF;
                    active[sc] = false;
                    const unsigned col = lane + sc * 32;
                    if (col < rmax) {
                        const int kv = (int)kv0 + col;
                        bool masked = (kv > qabs);
                        if (window) masked |= ((unsigned)(qabs - kv) >= window);
                        if (!masked) { sv[sc] = Ss[row * BKV + col]; active[sc] = true; }
                    }
                }
                float local_max = sv[0];
#pragma unroll
                for (int sc = 1; sc < SOFT_COLS; sc++) local_max = fmaxf(local_max, sv[sc]);
                const float rowmax = warp_max32(local_max);
                const float m_old = m_arr[row];
                const float m_new = fmaxf(m_old, rowmax);
                const float corr = (m_old == FA_NEG_INF) ? 0.0f : FA_EXP(m_old - m_new);
                float psum = 0.0f;
#pragma unroll
                for (int sc = 0; sc < SOFT_COLS; sc++) {
                    const unsigned col = lane + sc * 32;
                    const float p = (active[sc] && m_new != FA_NEG_INF)
                                      ? FA_EXP(sv[sc] - m_new) : 0.0f;
                    if (col < (unsigned)BKV)
                        Ps[row * (BKV + PAD) + col] = __float2bfloat16(p);
                    psum += p;
                }
                const float rowsum = warp_sum32(psum);
                if (lane == 0) {
                    l_arr[row] = l_arr[row] * corr + rowsum;
                    m_arr[row] = m_new;
                    corr_arr[row] = corr;
                }
            }
            __syncthreads();

            /* P.V phase: O += P.V via mma.sync tensor cores. This warp owns 16 query x HDW hd. First
             * rescale the O accumulator fragment by the per-query-row corr (D-frag row = pv_wm*16 +
             * lane/4 for e in {0,1}, +8 for e in {2,3}). Then A = P (Ps, ldmatrix.x4), B = V (Vs
             * natural, ldmatrix.x2.trans — same operand layout as QK's K^T load). */
            {
                const float c_lo = corr_arr[pv_wm * 16 + (lane >> 2)];
                const float c_hi = corr_arr[pv_wm * 16 + (lane >> 2) + 8];
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    oacc[nj][0] *= c_lo;
                    oacc[nj][1] *= c_lo;
                    oacc[nj][2] *= c_hi;
                    oacc[nj][3] *= c_hi;
                }
#pragma unroll
                for (int kf = 0; kf < KSTEPS_PV; kf++) {
                    unsigned af[4];
                    fa_ldmatrix_x4(af, &Ps[(pv_wm * 16 + (lane % 16)) * (BKV + PAD) +
                                          kf * 16 + (lane / 16) * 8]);
#pragma unroll
                    for (int nj = 0; nj < NJ_PV; nj++) {
                        unsigned bf[2];
                        fa_ldmatrix_x2_trans(bf, &Vs[(kf * 16 + (lane % 16)) * (HD + PAD) +
                                                     pv_wn * HDW + nj * 8]);
                        fa_mma(oacc[nj], af, bf, oacc[nj]);
                    }
                }
            }
            __syncthreads();
        }

        /* Epilogue. The O accumulator is the mma D fragment: element (nj,e) maps to query row
         * pv_wm*16 + lane/4 + (e/2)*8 and hd pv_wn*HDW + nj*8 + (lane%4)*2 + (e%2). nsplit>1: emit
         * UNNORMALISED partials for d_flash_merge. nsplit==1: normalise by l and write the bf16. */
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned qrow = (unsigned)(pv_wm * 16 + (lane >> 2) + (e >> 1) * 8);
                const unsigned qabs_row = q0 + qrow;
                if (qabs_row >= seq_q) continue;
                const int hd = pv_wn * HDW + nj * 8 + (lane & 3) * 2 + (e & 1);
                if (nsplit > 1) {
                    Opart[((size_t)(qabs_row * n_head + h) * nsplit + sp) * HD + hd] = oacc[nj][e];
                } else {
                    const float lv = l_arr[qrow];
                    const float inv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
                    O[(size_t)(qabs_row * n_head + h) * HD + hd] = __float2bfloat16(oacc[nj][e] * inv);
                }
            }
        /* ml partials (nsplit>1): one write per query row, from the smem-resident (m,l). */
        if (nsplit > 1) {
            for (int qrow = tid; qrow < BQ; qrow += (int)PLOW_NV_THREADS) {
                const unsigned qabs_row = q0 + (unsigned)qrow;
                if (qabs_row >= seq_q) continue;
                float* ml = mlpart + ((size_t)(qabs_row * n_head + h) * nsplit + sp) * 2;
                ml[0] = m_arr[qrow];
                ml[1] = l_arr[qrow];
            }
        }
        /* No barrier needed here: the next work item restages behind its own leading __syncthreads,
         * which also fences these epilogue reads of the m/l smem before the next softmax rewrites it. */
    }
}
#else /* PLOW_NV_FA_PIPE — cp.async KV-stream pipeline (T5 L1). See the fa_cp_* header. */

/* ---- PX-4 TMA (cp.async.bulk + mbarrier) staging helpers (sm_120 single-CTA TMA, rtx-01 §1) - */
__device__ __forceinline__ void fa_mbar_init(void* bar, unsigned count) {
    unsigned s = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(s), "r"(count));
}
__device__ __forceinline__ void fa_mbar_expect_tx(void* bar, unsigned bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(s), "r"(bytes));
}
__device__ __forceinline__ void fa_mbar_wait(void* bar, unsigned parity) {
    unsigned s = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("{\n\t.reg .pred P%=;\n"
                 "LAB%=:\n\tmbarrier.try_wait.parity.shared.b64 P%=, [%0], %1;\n"
                 "\t@!P%= bra LAB%=;\n\t}\n" ::"r"(s), "r"(parity));
}
/* One contiguous gmem row -> smem burst; completion counted on `bar` (expect_tx set by caller). */
__device__ __forceinline__ void fa_cp_bulk_g2s(void* smem_dst, const void* gmem_src, unsigned bytes,
                                               void* bar) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    unsigned b = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];\n" ::"r"(d), "l"(gmem_src), "r"(bytes), "r"(b)
                 : "memory");
}

/* PX-4 hd512 FULL-layer body. Same contract as d_flash_prefill (dispatched from it).
 * FP8KV (beat-fp8-prefill Exp1): the K/V cache is e4m3 (1 byte/elem) + a PER-ROW f32 dequant
 * scale. Raw bytes are cp.async-staged into Ks8/Vs8, dequanted (unscaled) to the bf16 Ks/Vs the
 * mma reads; the K-scale post-multiplies the score tile per kv column, the V-scale folds into the
 * P fragment — identical numerics to the PIPE=0 reference arm (d_flash_prefill FP8KV). */
template <int HD, int BQ, int BKV, bool FP8KV = false>
__device__ void d_flash_prefill_px4(float* __restrict__ Opart, float* __restrict__ mlpart,
                                    const __nv_bfloat16* __restrict__ Q,
                                    const __nv_bfloat16* __restrict__ K,
                                    const __nv_bfloat16* __restrict__ V,
                                    __nv_bfloat16* __restrict__ O, unsigned seq_q, unsigned seq_kv,
                                    unsigned n_head, unsigned n_kv_head, unsigned q_pos0,
                                    unsigned window, unsigned nsplit, unsigned kv_stride,
                                    unsigned kv_mask, float scale, unsigned slice, unsigned nblk,
                                    float* lds, const float* __restrict__ k_scale = nullptr,
                                    const float* __restrict__ v_scale = nullptr) {
    static_assert(HD == 512 && BQ == 32 && BKV == 16, "px4 arm is the hd512 FULL-layer tiling");
    static_assert((int)PLOW_NV_WARPS == 8, "px4 warp grids assume 8 warps");
    constexpr int PAD = FA_PRE_PAD;
    /* QK: 8 warps = 2(khalf: hd 0..255 / 256..511) x 2(query 16-row half) x 2(kv 8-col half). */
    constexpr int KSTEPS_H = HD / 2 / 16; /* 16 k16 steps per hd half */
    /* P.V: 2 query-warp-rows x 4 hd-warp-cols (as T4/T5); softmax rides this partition. */
    constexpr int WPV_N = 4;
    constexpr int HDW = HD / WPV_N;  /* 128 */
    constexpr int NJ_PV = HDW / 8;   /* 16 */
    constexpr int HCH = HD / 8;      /* 16B cp.async lines per K/V row */
    (void)HCH;

    unsigned long long* mbar = (unsigned long long*)lds; /* [0]=K, [1]=V (TMA arm only) */
    (void)mbar;
    float* SsA = lds + 4;                                /* [BQ][BKV] scores, hd 0..255 */
    float* SsB = SsA + BQ * BKV;                         /* [BQ][BKV] scores, hd 256..511 */
    __nv_bfloat16* Qs = (__nv_bfloat16*)(SsB + BQ * BKV); /* [BQ][HD+PAD] */
    constexpr int PAD8 = FA_FP8_PAD8_K; /* fp8 staging row pad (32; see FA_FP8_PAD8) */
    __nv_bfloat16* Ks;      /* [BKV][HD+PAD] natural (bf16 arms; DEAD under fp8mma) */
    __nv_bfloat16* Vs;      /* [BKV][HD+PAD] natural */
    float* qsc_s;           /* [BQ] per-query-row Q dequant scale (fp8mma arm only) */
    float* ksc_s;           /* [BKV] this tile's K row scales (fp8mma arm only) */
    float* vsc_s;           /* [BKV] this tile's V row scales (fp8mma arm only) */
    unsigned char* Qs8;     /* [BQ][HD+PAD8] e4m3 Q (fp8mma arm only) */
    unsigned char* Ks8;     /* [BKV][HD+PAD8] raw-e4m3 staging (fp8 arms) */
    unsigned char* Vs8;
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
    if constexpr (FP8KV) {
        /* fp8mma (Lever A): the bf16 Ks tile is DEAD (QK consumes Ks8 raw via mma.e4m3), so its
         * slot holds qsc/ksc/vsc + Qs8. Layout: Qs | Vs | qsc | ksc | vsc | Qs8 | Ks8 | Vs8
         * (== FA_PX4_SMEM_FLOATS). */
        Vs = Qs + BQ * (HD + PAD);
        Ks = nullptr;
        qsc_s = (float*)(Vs + BKV * (HD + PAD));
        ksc_s = qsc_s + BQ;
        vsc_s = ksc_s + BKV;
        Qs8 = (unsigned char*)(vsc_s + BKV);
        Ks8 = Qs8 + BQ * (HD + PAD8);
        Vs8 = Ks8 + BKV * (HD + PAD8);
    } else
#endif
    {
        /* classic layout (bf16 arm, and the NO-GO dequant fp8 arm when PLOW_NV_FA_FP8MMA=0).
         * fp8: raw-e4m3 staging pair, row stride HD + 16B pad, after the bf16 tiles. */
        Ks = Qs + BQ * (HD + PAD);
        Vs = Ks + BKV * (HD + PAD);
        qsc_s = nullptr;
        ksc_s = nullptr;
        vsc_s = nullptr;
        Qs8 = nullptr;
        Ks8 = (unsigned char*)(Vs + BKV * (HD + PAD));
        Vs8 = Ks8 + BKV * (HD + PAD8);
    }
    (void)qsc_s;
    (void)ksc_s;
    (void)vsc_s;
    (void)Qs8;
    (void)Ks8;
    (void)Vs8;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned gqa = n_head / n_kv_head;
    const unsigned n_qt = (seq_q + BQ - 1) / BQ;
    const unsigned n_work = n_qt * n_head * nsplit;
    const float lscale = FA_SCALE(scale);
    const int qk_kh = warp >> 2, qk_wm = (warp >> 1) & 1, qk_wn = warp & 1;
    const int pv_wm = warp >> 2, pv_wn = warp & 3;
    /* This lane's softmax/P.V ownership: rows r0 / r0+8 of the q-tile, kv cols c0,c0+1,c0+8,c0+9. */
    const int r0 = pv_wm * 16 + (lane >> 2);
    const int c0 = (lane & 3) * 2;

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned h = (w / nsplit) % n_head;
        const unsigned qt = w / (nsplit * n_head);
        const unsigned q0 = qt * BQ;
        const unsigned hkv = h / gqa;

        const unsigned per = (seq_kv + nsplit - 1) / nsplit;
        const unsigned lo = sp * per;
        const unsigned hi = (lo + per < seq_kv) ? (lo + per) : seq_kv;

        const __nv_bfloat16* Qh = Q + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const __nv_bfloat16* Kb = K + (size_t)hkv * kv_stride * HD;
        const __nv_bfloat16* Vb = V + (size_t)hkv * kv_stride * HD;
        /* fp8: e4m3 byte bases + per-kv-row scale slices (null on the bf16 instantiation). */
        const unsigned char* Kb8 = (const unsigned char*)K + (size_t)hkv * kv_stride * HD;
        const unsigned char* Vb8 = (const unsigned char*)V + (size_t)hkv * kv_stride * HD;
        const float* ksc = nullptr;
        const float* vsc = nullptr;
        if constexpr (FP8KV) {
            ksc = k_scale + (size_t)hkv * kv_stride;
            vsc = v_scale + (size_t)hkv * kv_stride;
        }

        __syncthreads(); /* previous item's Qs/Ks/Vs/Ss reads done before restage */
        for (int idx = tid; idx < BQ * HD; idx += (int)PLOW_NV_THREADS) {
            int r = idx / HD, c = idx % HD;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (q0 + r < seq_q) v = Qh[(size_t)r * n_head * HD + c];
            Qs[r * (HD + PAD) + c] = v;
        }
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
        /* fp8mma (Lever A): quantize Q -> e4m3 ONCE per q-tile (amortized over the whole KV loop).
         * 8 warps x 4 rows each; 8 lanes per row cover HD/8=64 elems. Per-row amax -> qinv, the
         * inverse scale lands in qsc_s and multiplies the score with the k-scale (both factor out
         * of the e4m3 dot exactly as the decode/GEMM w8a8 scales do). Zero rows (pad) => qinv 0 =>
         * stored bytes 0, scale 0 — masked columns aside, a zero Q row yields zero scores. */
        if constexpr (FP8KV) {
#if !(PLOW_NV_FA_FP8ABL & 2)
            __syncthreads(); /* Qs staged by ALL threads visible before the quant read */
            const int qr = warp * 4 + (lane >> 3);
            const int le = lane & 7;
            const int cb = le * (HD / 8);
            float amax = 0.0f;
            for (int e = 0; e < HD / 8; e++)
                amax = fmaxf(amax, fabsf(__bfloat162float(Qs[qr * (HD + PAD) + cb + e])));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            const float qinv = (amax > 0.0f) ? (PLOW_FP8_E4M3_MAX / amax) : 0.0f;
            if (le == 0) qsc_s[qr] = amax * (1.0f / PLOW_FP8_E4M3_MAX);
            /* Store in A-FRAGMENT ORDER: slot ((qm*2+kh)*8 + kf)*512 + L*16 + hi8 holds the
             * 8 k-bytes lane L reads for its (rlo|rhi, kf) m16n8k32 A operand — the QK loop
             * then reads ONE conflict-free LDS.128 per kf (consecutive lanes 16B apart)
             * instead of two strided LDS.64. */
            {
                const int qm = qr >> 4, r7 = qr & 7, hi8 = ((qr >> 3) & 1) * 8;
                for (int g = 0; g < HD / 64; g++) {
                    const int c = cb + g * 8; /* 8 consecutive k-bytes = one frag half */
                    unsigned u0 = 0, u1 = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        u0 |= (unsigned)quant_fp8(
                                  __bfloat162float(Qs[qr * (HD + PAD) + c + j]) * qinv)
                              << (8 * j);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        u1 |= (unsigned)quant_fp8(
                                  __bfloat162float(Qs[qr * (HD + PAD) + c + 4 + j]) * qinv)
                              << (8 * j);
                    const int kh = c >> 8, kf = (c & 255) >> 5, L = r7 * 4 + ((c >> 3) & 3);
                    *(uint2*)&Qs8[(((qm * 2 + kh) * 8 + kf) << 9) + L * 16 + hi8] =
                        make_uint2(u0, u1);
                }
            }
#endif /* !(PLOW_NV_FA_FP8ABL & 2) */
        }
#endif
#if PLOW_NV_FA_TMA
        if (tid == 0) {
            fa_mbar_init(&mbar[0], 1);
            fa_mbar_init(&mbar[1], 1);
        }
        unsigned kph = 0, vph = 0; /* mbarrier phase counters (one flip per stage) */
#endif

        float oacc[NJ_PV][4];
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) oacc[nj][e] = 0.0f;
        /* Online-softmax state: REGISTER-resident, per lane, rows r0 (j=0) and r0+8 (j=1).
         * Replicated across the WPV_N warps of a query row (identical arithmetic). */
        float m_reg[2] = {FA_NEG_INF, FA_NEG_INF}, l_reg[2] = {0.0f, 0.0f};

        const int qabs_max = (int)(q_pos0 + q0 + BQ - 1);
        unsigned eff_lo = lo;
        if (window) {
            const long wfloor = (long)q_pos0 + q0 - (long)window + 1;
            if (wfloor > (long)lo) eff_lo = ((unsigned)wfloor / BKV) * (unsigned)BKV;
        }
        long cap = (long)hi - 1;
        if ((long)qabs_max < cap) cap = (long)qabs_max;
        const int nt = (cap >= (long)eff_lo) ? (int)((cap - (long)eff_lo) / BKV) + 1 : 0;

#if PLOW_NV_FA_TMA
#if defined(PLOW_FP8_KV) && PLOW_NV_PREFILL
#error "PLOW_NV_FA_TMA staging is bf16-only; build fp8-KV prefill with PLOW_NV_FA_PIPE=0"
#endif
        /* TMA staging: one cp.async.bulk per contiguous [kv][hd] row, counted on mbar[K|V].
         * Ragged tail V rows are ZEROED (stale smem could hold NaN bits; 0*NaN != 0 in the mma).
         * Ragged K rows only feed masked score columns, which the softmax never reads. */
        auto stageK = [&](unsigned kv0) {
            const unsigned nrows = (hi > kv0) ? ((hi - kv0 < (unsigned)BKV) ? hi - kv0 : (unsigned)BKV) : 0;
            if (tid == 0) {
                fa_mbar_expect_tx(&mbar[0], nrows * HD * 2);
                for (unsigned r = 0; r < nrows; r++)
                    fa_cp_bulk_g2s(&Ks[r * (HD + PAD)], Kb + (size_t)((kv0 + r) & kv_mask) * HD,
                                   HD * 2, &mbar[0]);
            }
        };
        auto stageV = [&](unsigned kv0) {
            const unsigned nrows = (hi > kv0) ? ((hi - kv0 < (unsigned)BKV) ? hi - kv0 : (unsigned)BKV) : 0;
            if (tid == 0) {
                fa_mbar_expect_tx(&mbar[1], nrows * HD * 2);
                for (unsigned r = 0; r < nrows; r++)
                    fa_cp_bulk_g2s(&Vs[r * (HD + PAD)], Vb + (size_t)((kv0 + r) & kv_mask) * HD,
                                   HD * 2, &mbar[1]);
            }
            if (nrows < (unsigned)BKV)
                for (int idx = tid; idx < (BKV - (int)nrows) * HD; idx += (int)PLOW_NV_THREADS) {
                    int r = (int)nrows + idx / HD, c = idx % HD;
                    Vs[r * (HD + PAD) + c] = __float2bfloat16(0.f);
                }
        };
#define FA_PX4_WAIT_K() do { fa_mbar_wait(&mbar[0], kph & 1u); kph++; } while (0)
#define FA_PX4_WAIT_V() do { fa_mbar_wait(&mbar[1], vph & 1u); vph++; } while (0)
#else
        /* fp8: HCH8 16B lines per e4m3 row (16 elems/line, HALF the bf16 lines) -> Ks8/Vs8 bytes. */
        constexpr int HCH8 = HD / 16;
        auto stageK = [&](unsigned kv0) {
            if constexpr (FP8KV) {
                for (int L = tid; L < BKV * HCH8; L += (int)PLOW_NV_THREADS) {
                    int r = L / HCH8, c16 = (L % HCH8) * 16;
                    unsigned kv = kv0 + (unsigned)r;
                    bool in = (kv < hi);
                    const unsigned char* g = in ? Kb8 + (size_t)(kv & kv_mask) * HD + c16 : Kb8;
                    fa_cp_async_cg16(&Ks8[r * (HD + PAD8) + c16], g, in ? 16 : 0);
                }
            } else {
                for (int L = tid; L < BKV * HCH; L += (int)PLOW_NV_THREADS) {
                    int r = L / HCH, c8 = (L % HCH) * 8;
                    unsigned kv = kv0 + (unsigned)r;
                    bool in = (kv < hi);
                    const __nv_bfloat16* g = in ? Kb + (size_t)(kv & kv_mask) * HD + c8 : Kb;
                    fa_cp_async_cg16(&Ks[r * (HD + PAD) + c8], g, in ? 16 : 0);
                }
            }
            fa_cp_commit();
        };
        auto stageV = [&](unsigned kv0) {
            if constexpr (FP8KV) {
                for (int L = tid; L < BKV * HCH8; L += (int)PLOW_NV_THREADS) {
                    int r = L / HCH8, c16 = (L % HCH8) * 16;
                    unsigned kv = kv0 + (unsigned)r;
                    bool in = (kv < hi);
                    const unsigned char* g = in ? Vb8 + (size_t)(kv & kv_mask) * HD + c16 : Vb8;
                    fa_cp_async_cg16(&Vs8[r * (HD + PAD8) + c16], g, in ? 16 : 0);
                }
            } else {
                for (int L = tid; L < BKV * HCH; L += (int)PLOW_NV_THREADS) {
                    int r = L / HCH, c8 = (L % HCH) * 8;
                    unsigned kv = kv0 + (unsigned)r;
                    bool in = (kv < hi);
                    const __nv_bfloat16* g = in ? Vb + (size_t)(kv & kv_mask) * HD + c8 : Vb;
                    fa_cp_async_cg16(&Vs[r * (HD + PAD) + c8], g, in ? 16 : 0);
                }
            }
            fa_cp_commit();
        };
        /* fp8: dequant a landed raw-e4m3 tile (Ks8/Vs8) -> bf16 (Ks/Vs) UNSCALED, 8 elems/thread
         * (reuses the exact fp8v8->bf16v8 idiom). Scales apply later (K post-mma, V into P). The
         * caller fences Ks8/Vs8 visible (post-wait __syncthreads) before and Ks/Vs after. */
        auto dequantK = [&]() {
            for (int L = tid; L < BKV * HCH; L += (int)PLOW_NV_THREADS) {
                int r = L / HCH, c8 = (L % HCH) * 8;
                bf16v8 d = fp8v8_to_bf16v8(ld_glob_fp8v8(&Ks8[r * (HD + PAD8) + c8]));
                *(uint4*)&Ks[r * (HD + PAD) + c8] = *(const uint4*)&d;
            }
        };
        auto dequantV = [&]() {
            for (int L = tid; L < BKV * HCH; L += (int)PLOW_NV_THREADS) {
                int r = L / HCH, c8 = (L % HCH) * 8;
                bf16v8 d = fp8v8_to_bf16v8(ld_glob_fp8v8(&Vs8[r * (HD + PAD8) + c8]));
                *(uint4*)&Vs[r * (HD + PAD) + c8] = *(const uint4*)&d;
            }
        };
#define FA_PX4_WAIT_K() fa_cp_wait<1>()
#define FA_PX4_WAIT_V() fa_cp_wait<1>()
#endif

        __syncthreads(); /* Qs published (+ mbar init visible) before the pipeline starts */

#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
        /* fp8mma: prefetch tile 0's K/V row scales into a register (threads 0..2*BKV-1). */
        float sc_pf = 0.0f;
        if constexpr (FP8KV) {
            if (tid < 2 * BKV && nt > 0) {
                const unsigned kvr = eff_lo + (unsigned)(tid & (BKV - 1));
                if (kvr < hi) sc_pf = (tid < BKV) ? ksc[kvr & kv_mask] : vsc[kvr & kv_mask];
            }
        }
        (void)sc_pf;
#endif
        if (nt > 0) stageK(eff_lo); /* prologue: K[0] in flight */

        for (int t = 0; t < nt; t++) {
            const unsigned kv0 = eff_lo + (unsigned)t * BKV;
            stageV(kv0);   /* V[t] streams under QK[t] + softmax[t] */
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
            /* fp8mma: publish the PREFETCHED scales for tile t (STS only — the gmem load was
             * issued a full tile ago, so its latency never touches the critical path), then
             * issue tile t+1's loads. Out-of-range rows carry 0 (their scores are masked /
             * their P is 0; 0 avoids NaN from uninitialized ring rows). The WAIT_K
             * __syncthreads below orders the STS before every consumer. */
            if constexpr (FP8KV) {
#if !(PLOW_NV_FA_FP8ABL & 8)
                if (tid < 2 * BKV) {
                    ((tid < BKV) ? ksc_s : vsc_s)[tid & (BKV - 1)] = sc_pf;
                    const unsigned kvn = kv0 + BKV + (unsigned)(tid & (BKV - 1));
                    sc_pf = 0.0f;
                    if (kvn < hi) sc_pf = (tid < BKV) ? ksc[kvn & kv_mask] : vsc[kvn & kv_mask];
                }
#endif /* !(PLOW_NV_FA_FP8ABL & 8) */
            }
#endif
            FA_PX4_WAIT_K();
            __syncthreads();
#if !(PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)) && !PLOW_NV_FA_TMA
            /* fp8 NO-GO arm: K[t] landed as raw e4m3 in Ks8 — dequant (unscaled) to bf16 Ks
             * before QK. The fp8mma arm (Lever A) consumes Ks8 RAW below: no pass, no barrier.
             * TMA staging has no raw-e4m3 arm (fp8-KV prefill builds use PLOW_NV_FA_PIPE=0),
             * and the lambda only exists in the cp.async branch. */
            if constexpr (FP8KV) {
                dequantK();
                __syncthreads();
            }
#endif

            /* S = Q.K^T, all 8 warps: this warp contracts one HD HALF for a 16(query) x 8(kv)
             * block; the halves land in SsA/SsB and are summed at the softmax read. */
            {
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                const int khoff = qk_kh * (HD / 2);
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
                if constexpr (FP8KV) {
                    /* Lever A: mma.m16n8k32.e4m3 straight off Qs8/Ks8 — 2x per-instruction depth
                     * (8 k32 steps per hd half vs 16 k16), half the dependent-chain latency, and
                     * the operands are PLAIN u32 reads (no ldmatrix for 8-bit; layout per
                     * fa_mma_fp8_k32's note, validated in experiments/fp8_verify.cu). */
                    constexpr int KSTEPS8_H = HD / 2 / 32; /* 8 k32 steps per hd half */
#if !(PLOW_NV_FA_FP8ABL & 4)
                    const int nn = qk_wn * 8 + (lane >> 2);
                    const int kb0 = khoff + 8 * (lane & 3);
                    /* A: ONE conflict-free LDS.128 per kf from the fragment-order Qs8 (see the
                     * quant pass). B: LDS.64 from the conflict-free-strided Ks8. TWO independent
                     * accumulator chains (even/odd kf) halve the dependent-mma latency. */
                    const uint4* QsAf =
                        (const uint4*)&Qs8[(((qk_wm * 2 + qk_kh) * 8) << 9) + lane * 16];
                    /* FOUR independent accumulator chains: the dependent-QMMA latency (~28cyc)
                     * never stacks more than 2 deep over the 8 k32 steps. */
                    float accB[4] = {0.f, 0.f, 0.f, 0.f}, accC[4] = {0.f, 0.f, 0.f, 0.f},
                          accD[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int kf = 0; kf < KSTEPS8_H; kf++) {
                        const int kb = kb0 + kf * 32;
                        unsigned a8[4], b8[2];
                        const uint4 av = QsAf[kf * 32]; /* (kf<<9)/16 u4-slots */
                        const uint2 bb = *(const uint2*)&Ks8[nn * (HD + PAD8) + kb];
                        a8[0] = av.x; a8[2] = av.y;
                        a8[1] = av.z; a8[3] = av.w;
                        b8[0] = bb.x; b8[1] = bb.y;
                        switch (kf & 3) {
                            case 0: fa_mma_fp8_k32(acc, a8, b8, acc); break;
                            case 1: fa_mma_fp8_k32(accB, a8, b8, accB); break;
                            case 2: fa_mma_fp8_k32(accC, a8, b8, accC); break;
                            default: fa_mma_fp8_k32(accD, a8, b8, accD); break;
                        }
                    }
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[e] += accB[e] + (accC[e] + accD[e]);
#endif /* !(PLOW_NV_FA_FP8ABL & 4) */
                } else
#endif
                {
#pragma unroll
                    for (int kf = 0; kf < KSTEPS_H; kf++) {
                        unsigned af[4];
                        fa_ldmatrix_x4(af, &Qs[(qk_wm * 16 + (lane % 16)) * (HD + PAD) + khoff +
                                              kf * 16 + (lane / 16) * 8]);
                        unsigned bf[2];
                        {
                            const int n = qk_wn * 8 + (lane & 7);
                            const int kcol = khoff + kf * 16 + ((lane >> 3) & 1) * 8;
                            fa_ldmatrix_x2(bf, &Ks[n * (HD + PAD) + kcol]);
                        }
                        fa_mma(acc, af, bf, acc);
                    }
                }
                float* Sdst = qk_kh ? SsB : SsA;
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
                if constexpr (FP8KV) {
                    /* fp8mma: e-pairs (0,1)/(2,3) hit adjacent kv columns of the same row —
                     * two STS.64 instead of four STS.32. Scales smem-staged (0 out-of-range). */
                    const int kc0 = qk_wn * 8 + (lane % 4) * 2;
                    const float ks0 = ksc_s[kc0] * lscale, ks1 = ksc_s[kc0 + 1] * lscale;
                    const int qlo = qk_wm * 16 + (lane / 4);
                    const float q0s = qsc_s[qlo], q1s = qsc_s[qlo + 8];
                    *(float2*)&Sdst[qlo * BKV + kc0] =
                        make_float2(acc[0] * ks0 * q0s, acc[1] * ks1 * q0s);
                    *(float2*)&Sdst[(qlo + 8) * BKV + kc0] =
                        make_float2(acc[2] * ks0 * q1s, acc[3] * ks1 * q1s);
                } else
#endif
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int qr = qk_wm * 16 + (lane / 4) + (e / 2) * 8;
                    int kc = qk_wn * 8 + (lane % 4) * 2 + (e % 2);
                    /* fp8: K-scale is per-kv-column -> post-multiply here (factors out of the dot,
                     * identical to decode's scale-after-dot). In-range guard: masked cols aren't
                     * read by softmax, so a stale scale there is harmless. fp8mma additionally
                     * multiplies the per-query-row Q scale (same factoring). */
                    float sc = 1.0f;
                    if constexpr (FP8KV) {
                        const unsigned kvr = kv0 + (unsigned)kc;
                        if (kvr < hi) sc = ksc[kvr & kv_mask];
                    }
                    Sdst[qr * BKV + kc] = acc[e] * lscale * sc;
                }
            }
            __syncthreads(); /* Ss halves published; Ks free for K[t+1] */

            if (t + 1 < nt) stageK(kv0 + BKV);
#if !PLOW_NV_FA_TMA
            else fa_cp_commit(); /* keep group counts symmetric for the V wait below */
#endif

            const unsigned rmax = (hi - kv0 < (unsigned)BKV) ? (hi - kv0) : (unsigned)BKV;

#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
            /* fp8mma: V[t] wait + OWN-BYTES dequant, then softmax, then ONE barrier publishes
             * both. cp_wait<1> here has K[t+1] as the single allowed outstanding group (the
             * stageK/commit above kept counts symmetric), exactly as the old post-softmax wait. */
            if constexpr (FP8KV) {
                FA_PX4_WAIT_V();
#if !(PLOW_NV_FA_FP8ABL & 1)
                /* OWN-BYTES dequant to FP16 (one cvt.f16x2.e4m3x2 per byte-pair — no f32/bf16
                 * round-trip) with the per-row V-scale FOLDED INTO V (|V*vsc| is the actual
                 * activation magnitude, comfortably fp16; P then stays UNSCALED in [0,1] and
                 * the softmax path drops its per-element scale entirely). Vs holds half. */
                constexpr int HCH8 = HD / 16;
                for (int L = tid; L < BKV * HCH8; L += (int)PLOW_NV_THREADS) {
                    const int r = L / HCH8, c16 = (L % HCH8) * 16;
                    const __half2 vs2 = __float2half2_rn(vsc_s[r]);
                    const uint4 raw = *(const uint4*)&Vs8[r * (HD + PAD8) + c16];
                    uint4 out0;
                    __half2 h2;
#define FA_CVT8(dst, w)                                                                             \
    {                                                                                               \
        __half2_raw hr0 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)((w) & 0xffffu),        \
                                                     __NV_E4M3);                                    \
        __half2_raw hr1 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)((w) >> 16), __NV_E4M3);\
        h2 = __hmul2(*(__half2*)&hr0, vs2);                                                         \
        (dst).x = *(unsigned*)&h2;                                                                  \
        h2 = __hmul2(*(__half2*)&hr1, vs2);                                                         \
        (dst).y = *(unsigned*)&h2;                                                                  \
    }
                    uint2 lo, hi2;
                    FA_CVT8(lo, raw.x);
                    FA_CVT8(hi2, raw.y);
                    out0.x = lo.x; out0.y = lo.y; out0.z = hi2.x; out0.w = hi2.y;
                    *(uint4*)&Vs[r * (HD + PAD) + c16] = out0;
                    FA_CVT8(lo, raw.z);
                    FA_CVT8(hi2, raw.w);
                    out0.x = lo.x; out0.y = lo.y; out0.z = hi2.x; out0.w = hi2.y;
                    *(uint4*)&Vs[r * (HD + PAD) + c16 + 8] = out0;
#undef FA_CVT8
                }
#endif
            }
#endif

            /* REGISTER softmax, fused into the P.V A-fragment. Each lane owns rows r0/r0+8 and
             * kv cols c0,c0+1,c0+8,c0+9 of P — exactly the mma m16n8k16 A layout — so the row
             * reductions are two quad shfls and P never touches smem. */
            unsigned af_pv[4];
            {
                float p[2][4];
                float corr[2];
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int row = r0 + j * 8;
                    const int qabs = (int)(q_pos0 + q0 + row);
                    float s[4], mx = FA_NEG_INF;
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
                    if constexpr (FP8KV) {
                        /* col pairs (c0,c0+1) and (c0+8,c0+9) are adjacent: 4 LDS.64. */
                        const float2 a0 = *(const float2*)&SsA[row * BKV + c0];
                        const float2 b0 = *(const float2*)&SsB[row * BKV + c0];
                        const float2 a1 = *(const float2*)&SsA[row * BKV + c0 + 8];
                        const float2 b1 = *(const float2*)&SsB[row * BKV + c0 + 8];
                        s[0] = a0.x + b0.x; s[1] = a0.y + b0.y;
                        s[2] = a1.x + b1.x; s[3] = a1.y + b1.y;
#pragma unroll
                        for (int ci = 0; ci < 4; ci++) {
                            const int col = c0 + (ci & 1) + (ci >> 1) * 8;
                            const int kv = (int)kv0 + col;
                            bool masked = ((unsigned)col >= rmax) || (kv > qabs);
                            if (window) masked |= ((unsigned)(qabs - kv) >= window);
                            if (masked) s[ci] = FA_NEG_INF;
                            mx = fmaxf(mx, s[ci]);
                        }
                    } else
#endif
#pragma unroll
                    for (int ci = 0; ci < 4; ci++) {
                        const int col = c0 + (ci & 1) + (ci >> 1) * 8;
                        const int kv = (int)kv0 + col;
                        bool masked = ((unsigned)col >= rmax) || (kv > qabs);
                        if (window) masked |= ((unsigned)(qabs - kv) >= window);
                        s[ci] = masked ? FA_NEG_INF : SsA[row * BKV + col] + SsB[row * BKV + col];
                        mx = fmaxf(mx, s[ci]);
                    }
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                    const float m_new = fmaxf(m_reg[j], mx);
                    corr[j] = (m_reg[j] == FA_NEG_INF) ? 0.0f : FA_EXP(m_reg[j] - m_new);
                    float lsum = 0.0f;
#pragma unroll
                    for (int ci = 0; ci < 4; ci++) {
                        p[j][ci] = (s[ci] == FA_NEG_INF || m_new == FA_NEG_INF)
                                       ? 0.0f
                                       : FA_EXP(s[ci] - m_new);
                        lsum += p[j][ci];
                    }
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
                    l_reg[j] = l_reg[j] * corr[j] + lsum;
                    m_reg[j] = m_new;
                    /* fp8: V-scale is per-kv-row -> fold into P AFTER the (unscaled) lsum, mirroring
                     * decode's pw = P*v_scale. Masked cols carry p=0 (scale is a no-op there). */
#if !(PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV))
                    if constexpr (FP8KV) {
#pragma unroll
                        for (int ci = 0; ci < 4; ci++) {
                            const int col = c0 + (ci & 1) + (ci >> 1) * 8;
                            const unsigned kvr = kv0 + (unsigned)col;
                            if (kvr < hi) p[j][ci] *= vsc[kvr & kv_mask];
                        }
                    }
#endif /* fp8mma folds vsc into V at the fp16 dequant instead */
                }
                /* A fragment: af[0]=(r0,k-lo) af[1]=(r0+8,k-lo) af[2]=(r0,k-hi) af[3]=(r0+8,k-hi). */
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
                if constexpr (FP8KV) {
                    /* fp8mma: P packs to HALF (the fp16 P.V mma twin; P is unscaled [0,1]). */
                    __half2 hh;
                    hh = __floats2half2_rn(p[0][0], p[0][1]); af_pv[0] = *(unsigned*)&hh;
                    hh = __floats2half2_rn(p[1][0], p[1][1]); af_pv[1] = *(unsigned*)&hh;
                    hh = __floats2half2_rn(p[0][2], p[0][3]); af_pv[2] = *(unsigned*)&hh;
                    hh = __floats2half2_rn(p[1][2], p[1][3]); af_pv[3] = *(unsigned*)&hh;
                } else
#endif
                {
                    __nv_bfloat162 h;
                    h = __floats2bfloat162_rn(p[0][0], p[0][1]); af_pv[0] = *(unsigned*)&h;
                    h = __floats2bfloat162_rn(p[1][0], p[1][1]); af_pv[1] = *(unsigned*)&h;
                    h = __floats2bfloat162_rn(p[0][2], p[0][3]); af_pv[2] = *(unsigned*)&h;
                    h = __floats2bfloat162_rn(p[1][2], p[1][3]); af_pv[3] = *(unsigned*)&h;
                }
                /* fp8mma only — warp-uniform corr==1 skip: after the running max stabilizes,
                 * the rescale is a 64-FMUL no-op; skipping when EVERY lane's corr is exactly
                 * 1.0 (ex2(0)) is bitwise identical. The bf16 arm keeps the unconditional
                 * rescale (its compute is mma-hidden; the ballot only added overhead there,
                 * and the bf16 object must stay byte-identical). */
#ifndef PLOW_NV_FA_CORRSKIP
#define PLOW_NV_FA_CORRSKIP 0 /* measured SLOWER (the ballot defeats the mma scheduler) */
#endif
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV) && PLOW_NV_FA_CORRSKIP
                constexpr bool FA_CORR_SKIP = FP8KV;
#else
                constexpr bool FA_CORR_SKIP = false;
#endif
                if (!FA_CORR_SKIP ||
                    __ballot_sync(0xffffffffu, corr[0] != 1.0f || corr[1] != 1.0f) != 0u) {
#pragma unroll
                    for (int nj = 0; nj < NJ_PV; nj++) {
                        oacc[nj][0] *= corr[0];
                        oacc[nj][1] *= corr[0];
                        oacc[nj][2] *= corr[1];
                        oacc[nj][3] *= corr[1];
                    }
                }
            }

#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
            if constexpr (FP8KV) {
                /* fp8mma: V was waited + own-bytes-dequanted before softmax; this single barrier
                 * publishes the bf16 Vs to every P.V warp (== bf16 arm's barrier count). */
                __syncthreads();
            } else {
                FA_PX4_WAIT_V();
                __syncthreads(); /* all threads' V rows visible */
            }
#else
            FA_PX4_WAIT_V();
            __syncthreads(); /* all threads' V rows visible */
#if !PLOW_NV_FA_TMA
            /* fp8: V[t] landed as raw e4m3 in Vs8 — dequant (unscaled; V-scale is folded into P). */
            if constexpr (FP8KV) {
                dequantV();
                __syncthreads();
            }
#endif
#endif

            /* O += P.V: BKV=16 is one k16 step; B = V natural, ldmatrix.x2.trans (as T5).
             * fp8mma: Vs holds fp16 (vsc folded) and P is half -> the f16 mma twin. */
#pragma unroll
            for (int nj = 0; nj < NJ_PV; nj++) {
                unsigned bf[2];
                fa_ldmatrix_x2_trans(bf, &Vs[(lane % 16) * (HD + PAD) + pv_wn * HDW + nj * 8]);
#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV)
                if constexpr (FP8KV) fa_mma_f16(oacc[nj], af_pv, bf, oacc[nj]);
                else fa_mma(oacc[nj], af_pv, bf, oacc[nj]);
#else
                fa_mma(oacc[nj], af_pv, bf, oacc[nj]);
#endif
            }
            __syncthreads(); /* P.V done reading Vs before V[t+1] restages it */
        }

        /* Epilogue: as T5, but m/l come straight from this lane's registers. */
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned qrow = (unsigned)(r0 + (e >> 1) * 8);
                const unsigned qabs_row = q0 + qrow;
                if (qabs_row >= seq_q) continue;
                const int hd = pv_wn * HDW + nj * 8 + (lane & 3) * 2 + (e & 1);
                if (nsplit > 1) {
                    Opart[((size_t)(qabs_row * n_head + h) * nsplit + sp) * HD + hd] = oacc[nj][e];
                } else {
                    const float lv = l_reg[e >> 1];
                    const float inv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
                    O[(size_t)(qabs_row * n_head + h) * HD + hd] = __float2bfloat16(oacc[nj][e] * inv);
                }
            }
        if (nsplit > 1 && pv_wn == 0 && (lane & 3) == 0) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned qabs_row = q0 + (unsigned)(r0 + j * 8);
                if (qabs_row >= seq_q) continue;
                float* ml = mlpart + ((size_t)(qabs_row * n_head + h) * nsplit + sp) * 2;
                ml[0] = m_reg[j];
                ml[1] = l_reg[j];
            }
        }
    }
#undef FA_PX4_WAIT_K
#undef FA_PX4_WAIT_V
}

#if PLOW_NV_FA_FP8PV
/* ---- PX-8: the px4 fp8mma arm with an e4m3 P.V at BKV=32 ------------------------------------
 * Structurally d_flash_prefill_px4 (cp.async K/V ring, 8-warp hd-split QK into SsA/SsB, register
 * softmax fused into the P.V A fragment, no Ps buffer). Three differences, and only three:
 *
 *  1. BKV 16 -> 32, so the P.V contraction fills one k32. The QK warp grid keeps its 2(hd half) x
 *     2(query 16-rows) split and each warp now owns 16 kv columns (2 n8 sub-tiles) instead of 8.
 *  2. The e4m3 -> fp16 V dequant pass is DELETED. V stays raw in Vs8 and the mma B operand comes
 *     from fa_ldmatrix_x2_trans_b8 off a FA_PX8_VROW-permuted tile. There is no bf16/fp16 V tile
 *     in this arm's smem at all.
 *  3. P is quantised to e4m3. Since v_scale is per-kv-row it can no longer fold into V, and
 *     P*v_scale (~1e-2) is below e4m3's smallest normal — so P is normalised by the tile's max
 *     v_scale and the resulting unit change rides in the online-softmax `corr` multiply that is
 *     already applied to the accumulator. The epilogue multiplies by the final units (gscale).
 *     Nothing extra touches the 64 accumulator registers.
 *
 * Numerics: P carries ~2 decimal digits instead of fp16's ~3. l (the softmax denominator) is
 * accumulated from the UNQUANTISED p, so the quantisation error is confined to the numerator and
 * cannot compound through the rescale. See perf-data/px8-flash-fp8-pv.md. */
template <int HD, int BQ, int BKV, bool FP8KV = false>
__device__ void d_flash_prefill_px8(float* __restrict__ Opart, float* __restrict__ mlpart,
                                    const __nv_bfloat16* __restrict__ Q,
                                    const __nv_bfloat16* __restrict__ K,
                                    const __nv_bfloat16* __restrict__ V,
                                    __nv_bfloat16* __restrict__ O, unsigned seq_q, unsigned seq_kv,
                                    unsigned n_head, unsigned n_kv_head, unsigned q_pos0,
                                    unsigned window, unsigned nsplit, unsigned kv_stride,
                                    unsigned kv_mask, float scale, unsigned slice, unsigned nblk,
                                    float* lds, const float* __restrict__ k_scale = nullptr,
                                    const float* __restrict__ v_scale = nullptr) {
    static_assert(HD == 512 && BQ == 32 && BKV == 32, "px8 arm is the hd512 e4m3-P.V tiling");
    static_assert((int)PLOW_NV_WARPS == 8, "px8 warp grids assume 8 warps");
    static_assert(FP8KV, "px8 is an fp8-KV arm");
    constexpr int PAD = FA_PRE_PAD, PAD8 = FA_FP8_PAD8_K;
    constexpr int WPV_N = 4, HDW = HD / WPV_N, NJ_PV = HDW / 8; /* 128, 16 */
    constexpr int KSTEPS8_H = HD / 2 / 32;                      /* 8 k32 steps per hd half */
    constexpr int HCH8 = HD / 16;                               /* 16B cp.async lines per e4m3 row */
    constexpr float PS = 256.0f;                                /* P headroom inside e4m3 */

    float* SsA = lds + 4;
    float* SsB = SsA + BQ * BKV;
    __nv_bfloat16* Qs = (__nv_bfloat16*)(SsB + BQ * BKV);
    float* qsc_s = (float*)(Qs + BQ * (HD + PAD));
    float* ksc_s = qsc_s + BQ;
    float* vsc_s = ksc_s + BKV;
    unsigned char* Qs8 = (unsigned char*)(vsc_s + BKV);
    unsigned char* Ks8 = Qs8 + BQ * (HD + PAD8);
    unsigned char* Vs8 = Ks8 + BKV * (HD + PAD8);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned gqa = n_head / n_kv_head;
    const unsigned n_qt = (seq_q + BQ - 1) / BQ;
    const unsigned n_work = n_qt * n_head * nsplit;
    const float lscale = FA_SCALE(scale);
    const int qk_kh = warp >> 2, qk_wm = (warp >> 1) & 1, qk_wn = warp & 1;
    const int pv_wm = warp >> 2, pv_wn = warp & 3;
    const int r0 = pv_wm * 16 + (lane >> 2); /* this lane's softmax rows: r0 and r0+8 */
    const int kb0 = (lane & 3) * 8;          /* its 8 CONSECUTIVE kv columns (the k32 A fragment) */

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned h = (w / nsplit) % n_head;
        const unsigned qt = w / (nsplit * n_head);
        const unsigned q0 = qt * BQ;
        const unsigned hkv = h / gqa;

        const unsigned per = (seq_kv + nsplit - 1) / nsplit;
        const unsigned lo = sp * per;
        const unsigned hi = (lo + per < seq_kv) ? (lo + per) : seq_kv;

        const __nv_bfloat16* Qh = Q + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const unsigned char* Kb8 = (const unsigned char*)K + (size_t)hkv * kv_stride * HD;
        const unsigned char* Vb8 = (const unsigned char*)V + (size_t)hkv * kv_stride * HD;
        const float* ksc = k_scale + (size_t)hkv * kv_stride;
        const float* vsc = v_scale + (size_t)hkv * kv_stride;

        __syncthreads(); /* previous item's reads done before restage */
        for (int idx = tid; idx < BQ * HD; idx += (int)PLOW_NV_THREADS) {
            int r = idx / HD, c = idx % HD;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (q0 + r < seq_q) v = Qh[(size_t)r * n_head * HD + c];
            Qs[r * (HD + PAD) + c] = v;
        }
        __syncthreads();
        /* Q -> e4m3 once per q-tile, stored in mma A-fragment order (px4's layout, verbatim). */
        {
            const int qr = warp * (BQ / 8) + (lane >> 3);
            const int le = lane & 7;
            const int cb = le * (HD / 8);
            float amax = 0.0f;
            for (int e = 0; e < HD / 8; e++)
                amax = fmaxf(amax, fabsf(__bfloat162float(Qs[qr * (HD + PAD) + cb + e])));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            const float qinv = (amax > 0.0f) ? (PLOW_FP8_E4M3_MAX / amax) : 0.0f;
            if (le == 0) qsc_s[qr] = amax * (1.0f / PLOW_FP8_E4M3_MAX);
            const int qm = qr >> 4, r7 = qr & 7, hi8 = ((qr >> 3) & 1) * 8;
            for (int g = 0; g < HD / 64; g++) {
                const int c = cb + g * 8;
                unsigned u0 = 0, u1 = 0;
#pragma unroll
                for (int j = 0; j < 4; j++)
                    u0 |= (unsigned)quant_fp8(__bfloat162float(Qs[qr * (HD + PAD) + c + j]) * qinv)
                          << (8 * j);
#pragma unroll
                for (int j = 0; j < 4; j++)
                    u1 |= (unsigned)quant_fp8(
                              __bfloat162float(Qs[qr * (HD + PAD) + c + 4 + j]) * qinv)
                          << (8 * j);
                const int kh = c >> 8, kf = (c & 255) >> 5, L = r7 * 4 + ((c >> 3) & 3);
                *(uint2*)&Qs8[(((qm * 2 + kh) * 8 + kf) << 9) + L * 16 + hi8] = make_uint2(u0, u1);
            }
        }

        float oacc[NJ_PV][4];
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) oacc[nj][e] = 0.0f;
        float m_reg[2] = {FA_NEG_INF, FA_NEG_INF}, l_reg[2] = {0.0f, 0.0f};
        float gscale = 0.0f; /* the accumulator's units; see the P normaliser below */

        const int qabs_max = (int)(q_pos0 + q0 + BQ - 1);
        unsigned eff_lo = lo;
        if (window) {
            const long wfloor = (long)q_pos0 + q0 - (long)window + 1;
            if (wfloor > (long)lo) eff_lo = ((unsigned)wfloor / BKV) * (unsigned)BKV;
        }
        long cap = (long)hi - 1;
        if ((long)qabs_max < cap) cap = (long)qabs_max;
        const int nt = (cap >= (long)eff_lo) ? (int)((cap - (long)eff_lo) / BKV) + 1 : 0;

        auto stageK = [&](unsigned kv0) {
            for (int L = tid; L < BKV * HCH8; L += (int)PLOW_NV_THREADS) {
                int r = L / HCH8, c16 = (L % HCH8) * 16;
                unsigned kv = kv0 + (unsigned)r;
                bool in = (kv < hi);
                const unsigned char* g = in ? Kb8 + (size_t)(kv & kv_mask) * HD + c16 : Kb8;
                fa_cp_async_cg16(&Ks8[r * (HD + PAD8) + c16], g, in ? 16 : 0);
            }
            fa_cp_commit();
        };
        /* kv row r lands in smem row FA_PX8_VROW(r) — the free half of the fp8 P.V. */
        auto stageV = [&](unsigned kv0) {
            for (int L = tid; L < BKV * HCH8; L += (int)PLOW_NV_THREADS) {
                int r = L / HCH8, c16 = (L % HCH8) * 16;
                unsigned kv = kv0 + (unsigned)r;
                bool in = (kv < hi);
                const unsigned char* g = in ? Vb8 + (size_t)(kv & kv_mask) * HD + c16 : Vb8;
                fa_cp_async_cg16(&Vs8[FA_PX8_VROW(r) * (HD + PAD8) + c16], g, in ? 16 : 0);
            }
            fa_cp_commit();
        };

        __syncthreads(); /* Qs8 published before the pipeline starts */
        float sc_pf = 0.0f;
        if (tid < 2 * BKV && nt > 0) {
            const unsigned kvr = eff_lo + (unsigned)(tid & (BKV - 1));
            if (kvr < hi) sc_pf = (tid < BKV) ? ksc[kvr & kv_mask] : vsc[kvr & kv_mask];
        }
        if (nt > 0) stageK(eff_lo);

        for (int t = 0; t < nt; t++) {
            const unsigned kv0 = eff_lo + (unsigned)t * BKV;
            stageV(kv0);
            if (tid < 2 * BKV) {
                ((tid < BKV) ? ksc_s : vsc_s)[tid & (BKV - 1)] = sc_pf;
                const unsigned kvn = kv0 + BKV + (unsigned)(tid & (BKV - 1));
                sc_pf = 0.0f;
                if (kvn < hi) sc_pf = (tid < BKV) ? ksc[kvn & kv_mask] : vsc[kvn & kv_mask];
            }
            fa_cp_wait<1>(); /* K[t] */
            __syncthreads();

            /* S = Q.K^T, e4m3 k32. Each warp: one hd half x 16 query rows x 16 kv (2 n8). */
            {
                float acc[2][4], accB[2][4];
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) { acc[j][e] = 0.f; accB[j][e] = 0.f; }
                const int khoff = qk_kh * (HD / 2);
                const int kbq = khoff + 8 * (lane & 3);
                const uint4* QsAf = (const uint4*)&Qs8[(((qk_wm * 2 + qk_kh) * 8) << 9) + lane * 16];
#pragma unroll
                for (int kf = 0; kf < KSTEPS8_H; kf++) {
                    const int kb = kbq + kf * 32;
                    unsigned a8[4];
                    const uint4 av = QsAf[kf * 32];
                    a8[0] = av.x; a8[2] = av.y; a8[1] = av.z; a8[3] = av.w;
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const int nn = qk_wn * 16 + j * 8 + (lane >> 2);
                        const uint2 bb = *(const uint2*)&Ks8[nn * (HD + PAD8) + kb];
                        unsigned b8[2] = {bb.x, bb.y};
                        if (kf & 1) fa_mma_fp8_k32(accB[j], a8, b8, accB[j]);
                        else        fa_mma_fp8_k32(acc[j], a8, b8, acc[j]);
                    }
                }
                float* Sdst = qk_kh ? SsB : SsA;
                const int qlo = qk_wm * 16 + (lane / 4);
                const float q0s = qsc_s[qlo], q1s = qsc_s[qlo + 8];
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int kc0 = qk_wn * 16 + j * 8 + (lane % 4) * 2;
                    const float ks0 = ksc_s[kc0] * lscale, ks1 = ksc_s[kc0 + 1] * lscale;
                    const float a0 = acc[j][0] + accB[j][0], a1 = acc[j][1] + accB[j][1];
                    const float a2 = acc[j][2] + accB[j][2], a3 = acc[j][3] + accB[j][3];
                    *(float2*)&Sdst[qlo * BKV + kc0] = make_float2(a0 * ks0 * q0s, a1 * ks1 * q0s);
                    *(float2*)&Sdst[(qlo + 8) * BKV + kc0] =
                        make_float2(a2 * ks0 * q1s, a3 * ks1 * q1s);
                }
            }
            __syncthreads(); /* Ss published; Ks8 free for K[t+1] */
            if (t + 1 < nt) stageK(kv0 + BKV);
            else fa_cp_commit(); /* keep group counts symmetric for the V wait */

            const unsigned rmax = (hi - kv0 < (unsigned)BKV) ? (hi - kv0) : (unsigned)BKV;
            fa_cp_wait<1>(); /* V[t] */

            /* Register softmax fused into the e4m3 P.V A fragment. */
            unsigned af_pv[4];
            {
                float vmax = 0.0f;
#pragma unroll
                for (int i = 0; i < BKV; i += 4) {
                    const float4 v4 = *(const float4*)&vsc_s[i];
                    vmax = fmaxf(fmaxf(vmax, v4.x), fmaxf(fmaxf(v4.y, v4.z), v4.w));
                }
                const float vnorm = (vmax > 0.0f) ? (PS / vmax) : 0.0f;
                float p[2][8];
                float corr[2];
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int row = r0 + j * 8;
                    const int qabs = (int)(q_pos0 + q0 + row);
                    float s[8], mx = FA_NEG_INF;
                    const float4 a0 = *(const float4*)&SsA[row * BKV + kb0];
                    const float4 a1 = *(const float4*)&SsA[row * BKV + kb0 + 4];
                    const float4 b0 = *(const float4*)&SsB[row * BKV + kb0];
                    const float4 b1 = *(const float4*)&SsB[row * BKV + kb0 + 4];
                    s[0] = a0.x + b0.x; s[1] = a0.y + b0.y; s[2] = a0.z + b0.z; s[3] = a0.w + b0.w;
                    s[4] = a1.x + b1.x; s[5] = a1.y + b1.y; s[6] = a1.z + b1.z; s[7] = a1.w + b1.w;
#pragma unroll
                    for (int ci = 0; ci < 8; ci++) {
                        const int col = kb0 + ci;
                        const int kv = (int)kv0 + col;
                        bool masked = ((unsigned)col >= rmax) || (kv > qabs);
                        if (window) masked |= ((unsigned)(qabs - kv) >= window);
                        if (masked) s[ci] = FA_NEG_INF;
                        mx = fmaxf(mx, s[ci]);
                    }
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                    const float m_new = fmaxf(m_reg[j], mx);
                    corr[j] = (m_reg[j] == FA_NEG_INF) ? 0.0f : FA_EXP(m_reg[j] - m_new);
                    float lsum = 0.0f;
#pragma unroll
                    for (int ci = 0; ci < 8; ci++) {
                        p[j][ci] = (s[ci] == FA_NEG_INF || m_new == FA_NEG_INF)
                                       ? 0.0f : FA_EXP(s[ci] - m_new);
                        lsum += p[j][ci];
                    }
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
                    l_reg[j] = l_reg[j] * corr[j] + lsum; /* l uses the UNQUANTISED p */
                    m_reg[j] = m_new;
#pragma unroll
                    for (int ci = 0; ci < 8; ci++) p[j][ci] *= vsc_s[kb0 + ci] * vnorm;
                }
                /* A fragment: a0=(r0,k0..3) a1=(r0+8,k0..3) a2=(r0,k4..7) a3=(r0+8,k4..7) */
                unsigned u[4] = {0, 0, 0, 0};
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    u[0] |= (unsigned)quant_fp8(p[0][b]) << (8 * b);
                    u[1] |= (unsigned)quant_fp8(p[1][b]) << (8 * b);
                    u[2] |= (unsigned)quant_fp8(p[0][4 + b]) << (8 * b);
                    u[3] |= (unsigned)quant_fp8(p[1][4 + b]) << (8 * b);
                }
                af_pv[0] = u[0]; af_pv[1] = u[1]; af_pv[2] = u[2]; af_pv[3] = u[3];
                /* The tile's P normaliser must NOT multiply the 64 mma outputs. Carry it in the
                 * accumulator's UNITS instead: oacc == O/gscale, and the unit change folds into
                 * the corr multiply that is applied anyway. */
                const float sc_pv = (vmax > 0.0f) ? (vmax / PS) : 0.0f;
                float cadj = 1.0f;
                if (sc_pv > 0.0f) {
                    cadj = (gscale > 0.0f) ? (gscale / sc_pv) : 0.0f;
                    gscale = sc_pv;
                }
                const float c0 = corr[0] * cadj, c1 = corr[1] * cadj;
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    oacc[nj][0] *= c0; oacc[nj][1] *= c0;
                    oacc[nj][2] *= c1; oacc[nj][3] *= c1;
                }
            }
            __syncthreads(); /* every thread's V[t] bytes visible to every P.V warp */

            /* O += P.V, e4m3 k32, straight off the RAW permuted-row Vs8 tile. */
#pragma unroll
            for (int nj = 0; nj < NJ_PV; nj += 2) {
                unsigned rb[4];
                fa_ldmatrix_x2_trans_b8(rb, &Vs8[lane * (HD + PAD8) + pv_wn * HDW + nj * 8]);
                unsigned b0[2] = {rb[0], rb[2]}, b1[2] = {rb[1], rb[3]};
                fa_mma_fp8_k32(oacc[nj], af_pv, b0, oacc[nj]);
                fa_mma_fp8_k32(oacc[nj + 1], af_pv, b1, oacc[nj + 1]);
            }
            __syncthreads(); /* P.V done reading Vs8 before V[t+1] restages it */
        }

#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned qrow = (unsigned)(r0 + (e >> 1) * 8);
                const unsigned qabs_row = q0 + qrow;
                if (qabs_row >= seq_q) continue;
                const int hd = pv_wn * HDW + nj * 8 + (lane & 3) * 2 + (e & 1);
                if (nsplit > 1) {
                    Opart[((size_t)(qabs_row * n_head + h) * nsplit + sp) * HD + hd] =
                        oacc[nj][e] * gscale;
                } else {
                    const float lv = l_reg[e >> 1];
                    const float inv = (lv > 0.0f) ? (gscale / lv) : 0.0f;
                    O[(size_t)(qabs_row * n_head + h) * HD + hd] =
                        __float2bfloat16(oacc[nj][e] * inv);
                }
            }
        if (nsplit > 1 && pv_wn == 0 && (lane & 3) == 0) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned qabs_row = q0 + (unsigned)(r0 + j * 8);
                if (qabs_row >= seq_q) continue;
                float* ml = mlpart + ((size_t)(qabs_row * n_head + h) * nsplit + sp) * 2;
                ml[0] = m_reg[j];
                ml[1] = l_reg[j];
            }
        }
    }
}
#endif /* PLOW_NV_FA_FP8PV */

#if PLOW_NV_FA_FP8MMA && defined(PLOW_FP8_KV) && !PLOW_NV_FA_TMA
/* ============================ PX-23: hd256 SLIDING-layer fp8 fast prefill ====================
 * The arm PX-20 §5 named as the single most actionable finding in the campaign: an all-layer
 * e4m3 packet emits hd256 FLASH_PREFILL_FP8, the PIPE=1 fp8 object trapped on it, and the whole
 * packet fell to the PIPE=0 synchronous-staging path at 176 s of prefill per 127k request.
 *
 * Same contract as d_flash_prefill (FP8KV=true): e4m3 K/V cache + PER-ROW f32 dequant scales.
 * Same numerics discipline as the px4 fp8mma arm, which is already validated against the PIPE=0
 * reference: QK runs as mma.m16n8k32.e4m3 straight off a per-q-tile-quantized Q and the RAW Ks8
 * tile (the Q row scale and the K column scale both factor out of the dot and post-multiply the
 * score); V dequants e4m3 -> fp16 with its row scale FOLDED IN, so P stays unscaled in [0,1] and
 * the P.V is the fp16 mma twin.
 *
 * Tiling rationale, smem budget and the warp-specialization decision: see FA_PX23_SMEM_FLOATS
 * above and the design notes. In one line: BQ=64/BKV=32 with a 4x2 warp grid and NO hd
 * split, because at hd256 the query x kv tile already fills 8 warps — which deletes px4's second
 * score tile, and with it the naive retile's 104,464 B arena (over the 101,376 B cap). */
template <int HD, int BQ, int BKV>
__device__ void d_flash_prefill_px23(float* __restrict__ Opart, float* __restrict__ mlpart,
                                     const __nv_bfloat16* __restrict__ Q,
                                     const __nv_bfloat16* __restrict__ K,
                                     const __nv_bfloat16* __restrict__ V,
                                     __nv_bfloat16* __restrict__ O, unsigned seq_q, unsigned seq_kv,
                                     unsigned n_head, unsigned n_kv_head, unsigned q_pos0,
                                     unsigned window, unsigned nsplit, unsigned kv_stride,
                                     unsigned kv_mask, float scale, unsigned slice, unsigned nblk,
                                     float* lds, const float* __restrict__ k_scale,
                                     const float* __restrict__ v_scale) {
    static_assert(HD == 256 && BQ == 64 && BKV == 32, "px23 arm is the hd256 SLIDING-layer tiling");
    static_assert((int)PLOW_NV_WARPS == 8 && (int)PLOW_NV_THREADS == 256,
                  "px23 warp grids assume 8 warps / 256 threads");
    /* The arm must never outgrow the arena the host allocated (PLOW_NV_PRE_A). A too-large claim
     * does NOT fail at compile time otherwise — it fails as a refused module load at serve time. */
    static_assert(FA_PX23_SMEM_FLOATS(HD, BQ, BKV) <= FA_PRE_SMEM_FLOATS(HD, BQ, BKV),
                  "px23 smem claim exceeds the prefill arena");
    constexpr int PAD = FA_PRE_PAD;      /* bf16/fp16 row pad, 8 elems */
    constexpr int PAD8 = FA_FP8_PAD8_K;  /* e4m3 row pad, 32 bytes */
    constexpr int KSTEPS8 = HD / 32;     /* 8 k32 QK steps (no hd split) */
    constexpr int NJ_QK = BKV / 16;      /* 2 n8 QK sub-tiles per warp (WQK_N = 2) */
    constexpr int WPV_N = 2;             /* hd warp cols */
    constexpr int HDW = HD / WPV_N;      /* 128 */
    constexpr int NJ_PV = HDW / 8;       /* 16 */
    constexpr int KSTEPS_PV = BKV / 16;  /* 2 k16 P.V steps */
    constexpr int HCH8 = HD / 16;        /* 16B cp.async lines per e4m3 row */
    constexpr int QPL = HD / 8;          /* Q elems per lane in the quant pass (8 lanes/row) */

    /* smem: Ss | Vs | qsc | ksc | vsc | Qs8 | Ks8 | Vs8  (== FA_PX23_SMEM_FLOATS).
     * px4's bf16 Qs tile is GONE: it only ever fed the quant pass, which here reads gmem into
     * registers instead. That is what brings hd256 under the cap. */
    float* Ss = lds;                                      /* [BQ][BKV] f32 scores */
    __half* Vs = (__half*)(Ss + BQ * BKV);                /* [BKV][HD+PAD] fp16, v_scale folded */
    float* qsc_s = (float*)(Vs + BKV * (HD + PAD));       /* [BQ] per-query-row Q scale */
    float* ksc_s = qsc_s + BQ;                            /* [BKV] this tile's K row scales */
    float* vsc_s = ksc_s + BKV;                           /* [BKV] this tile's V row scales */
    unsigned char* Qs8 = (unsigned char*)(vsc_s + BKV);   /* [BQ*HD] e4m3, A-FRAGMENT order */
    unsigned char* Ks8 = Qs8 + BQ * HD;                   /* [BKV][HD+PAD8] raw e4m3 */
    unsigned char* Vs8 = Ks8 + BKV * (HD + PAD8);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    /* ONE warp map for both phases: warp owns query rows [wm*16, +16) throughout.
     * QK  : wm = query warp row (4), wn = kv warp col (2, 16 kv cols each, NJ_QK n8 sub-tiles).
     * P.V : wm = query warp row (4), wn = hd warp col (2, HDW=128 each). */
    const int wm = warp >> 1, wn = warp & 1;
    const int r0 = wm * 16 + (lane >> 2); /* this lane's P rows: r0 and r0+8 */
    const int c0 = (lane & 3) * 2;        /* ... and P cols c0,c0+1,c0+8,c0+9 per k16 step */

    const unsigned gqa = n_head / n_kv_head;
    const unsigned n_qt = (seq_q + BQ - 1) / BQ;
    const unsigned n_work = n_qt * n_head * nsplit;
    const float lscale = FA_SCALE(scale);

    for (unsigned w = slice; w < n_work; w += nblk) {
        const unsigned sp = w % nsplit;
        const unsigned h = (w / nsplit) % n_head;
        const unsigned qt = w / (nsplit * n_head);
        const unsigned q0 = qt * BQ;
        const unsigned hkv = h / gqa;

        const unsigned per = (seq_kv + nsplit - 1) / nsplit;
        const unsigned lo = sp * per;
        const unsigned hi = (lo + per < seq_kv) ? (lo + per) : seq_kv;

        const __nv_bfloat16* Qh = Q + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const unsigned char* Kb8 = (const unsigned char*)K + (size_t)hkv * kv_stride * HD;
        const unsigned char* Vb8 = (const unsigned char*)V + (size_t)hkv * kv_stride * HD;
        const float* ksc = k_scale + (size_t)hkv * kv_stride;
        const float* vsc = v_scale + (size_t)hkv * kv_stride;

        __syncthreads(); /* previous item's Vs/Ss/Qs8 reads done before restage */

        /* ---- Q -> e4m3, ONCE per q-tile, straight from gmem through REGISTERS ----------------
         * 8 lanes per row, each owning QPL = HD/8 contiguous bf16 (one coalesced 512 B burst per
         * 8 lanes); 8 warps x 4 rows = 32 rows per pass, BQ/32 passes. Per-row amax via 3 quad
         * shfls -> qinv; the inverse scale lands in qsc_s and multiplies the score together with
         * the K column scale (both factor out of the e4m3 dot exactly as the w8a8 GEMM scales do).
         * Out-of-range rows: amax 0 => qinv 0 => stored bytes 0 and scale 0.
         * No barrier is needed here at all — px4 needed one only because it round-tripped Q
         * through a shared Qs tile that this arm does not have.
         *
         * Store is in A-FRAGMENT order: slot ((qm*KSTEPS8 + kf) << 9) + L*16 + hi8 holds the 8
         * k-bytes lane L reads for its (row lo|hi, kf) m16n8k32 A operand, so the QK loop is ONE
         * conflict-free LDS.128 per k32 step. */
        {
            const int le = lane & 7;
            const int cb = le * QPL;
#pragma unroll 1
            for (int qr = warp * 4 + (lane >> 3); qr < BQ; qr += 32) {
                __nv_bfloat16 qv[QPL];
                const bool inq = (q0 + (unsigned)qr) < seq_q;
                const __nv_bfloat16* src = Qh + (size_t)qr * n_head * HD + cb;
                float amax = 0.0f;
#pragma unroll
                for (int e = 0; e < QPL; e += 8) {
                    uint4 z = make_uint4(0u, 0u, 0u, 0u);
                    if (inq) z = *(const uint4*)(src + e);
                    *(uint4*)&qv[e] = z;
                }
#pragma unroll
                for (int e = 0; e < QPL; e++) amax = fmaxf(amax, fabsf(__bfloat162float(qv[e])));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
                const float qinv = (amax > 0.0f) ? (PLOW_FP8_E4M3_MAX / amax) : 0.0f;
                if (le == 0) qsc_s[qr] = amax * (1.0f / PLOW_FP8_E4M3_MAX);
                const int qm = qr >> 4, r7 = qr & 7, hi8 = ((qr >> 3) & 1) * 8;
#pragma unroll
                for (int g = 0; g < QPL / 8; g++) {
                    unsigned u0 = 0, u1 = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        u0 |= (unsigned)quant_fp8(__bfloat162float(qv[g * 8 + j]) * qinv) << (8 * j);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        u1 |= (unsigned)quant_fp8(__bfloat162float(qv[g * 8 + 4 + j]) * qinv)
                              << (8 * j);
                    const int c = cb + g * 8;
                    const int kf = c >> 5, L = r7 * 4 + ((c >> 3) & 3);
                    *(uint2*)&Qs8[(((qm * KSTEPS8) + kf) << 9) + L * 16 + hi8] = make_uint2(u0, u1);
                }
            }
        }

        float oacc[NJ_PV][4];
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) oacc[nj][e] = 0.0f;
        /* Online-softmax state: REGISTER-resident, rows r0 (j=0) and r0+8 (j=1). Replicated
         * across the WPV_N warps of a query row (identical arithmetic). */
        float m_reg[2] = {FA_NEG_INF, FA_NEG_INF}, l_reg[2] = {0.0f, 0.0f};

        const int qabs_max = (int)(q_pos0 + q0 + BQ - 1);
        unsigned eff_lo = lo;
        if (window) {
            const long wfloor = (long)q_pos0 + q0 - (long)window + 1;
            if (wfloor > (long)lo) eff_lo = ((unsigned)wfloor / BKV) * (unsigned)BKV;
        }
        long cap = (long)hi - 1;
        if ((long)qabs_max < cap) cap = (long)qabs_max;
        const int nt = (cap >= (long)eff_lo) ? (int)((cap - (long)eff_lo) / BKV) + 1 : 0;

        /* Staging: BKV*HCH8 = 512 16B lines per tile over 256 threads = 2 lines each. Ownership
         * is per-thread, which is what makes the V dequant below barrier-free. */
        auto stageK = [&](unsigned kv0) {
#pragma unroll
            for (int L = tid; L < BKV * HCH8; L += (int)PLOW_NV_THREADS) {
                const int r = L / HCH8, c16 = (L % HCH8) * 16;
                const unsigned kv = kv0 + (unsigned)r;
                const bool in = (kv < hi);
                const unsigned char* g = in ? Kb8 + (size_t)(kv & kv_mask) * HD + c16 : Kb8;
                fa_cp_async_cg16(&Ks8[r * (HD + PAD8) + c16], g, in ? 16 : 0);
            }
            fa_cp_commit();
        };
        auto stageV = [&](unsigned kv0) {
#pragma unroll
            for (int L = tid; L < BKV * HCH8; L += (int)PLOW_NV_THREADS) {
                const int r = L / HCH8, c16 = (L % HCH8) * 16;
                const unsigned kv = kv0 + (unsigned)r;
                const bool in = (kv < hi);
                const unsigned char* g = in ? Vb8 + (size_t)(kv & kv_mask) * HD + c16 : Vb8;
                fa_cp_async_cg16(&Vs8[r * (HD + PAD8) + c16], g, in ? 16 : 0);
            }
            fa_cp_commit();
        };

        __syncthreads(); /* Qs8 published before the QK loop reads it */

        /* Prefetch tile 0's K/V row scales into a register (threads 0..2*BKV-1). */
        float sc_pf = 0.0f;
        if (tid < 2 * BKV && nt > 0) {
            const unsigned kvr = eff_lo + (unsigned)(tid & (BKV - 1));
            if (kvr < hi) sc_pf = (tid < BKV) ? ksc[kvr & kv_mask] : vsc[kvr & kv_mask];
        }
        if (nt > 0) stageK(eff_lo); /* prologue: K[0] in flight */

        for (int t = 0; t < nt; t++) {
            const unsigned kv0 = eff_lo + (unsigned)t * BKV;
            stageV(kv0); /* V[t] streams under QK[t] + softmax[t] */
            /* Publish the PREFETCHED scales for tile t (STS only — the gmem load was issued a
             * full tile ago), then issue tile t+1's loads. Out-of-range rows carry 0. */
            if (tid < 2 * BKV) {
                ((tid < BKV) ? ksc_s : vsc_s)[tid & (BKV - 1)] = sc_pf;
                const unsigned kvn = kv0 + BKV + (unsigned)(tid & (BKV - 1));
                sc_pf = 0.0f;
                if (kvn < hi) sc_pf = (tid < BKV) ? ksc[kvn & kv_mask] : vsc[kvn & kv_mask];
            }
            fa_cp_wait<1>(); /* K[t] landed (V[t] is the one outstanding group) */
            __syncthreads();

            /* ---- S = Q.K^T : mma.m16n8k32.e4m3 off Qs8 (A-fragment order) and RAW Ks8 --------
             * 4 independent accumulator chains (NJ_QK sub-tiles x kf parity) so the ~28cyc
             * dependent-QMMA latency never stacks more than 2 deep over the 8 k32 steps. */
            {
                float acc[NJ_QK][4], accB[NJ_QK][4];
#pragma unroll
                for (int nj = 0; nj < NJ_QK; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) { acc[nj][e] = 0.0f; accB[nj][e] = 0.0f; }
                const int nn0 = wn * 16 + (lane >> 2);
                const int kb0 = 8 * (lane & 3);
                const uint4* QsAf = (const uint4*)&Qs8[((wm * KSTEPS8) << 9) + lane * 16];
#pragma unroll
                for (int kf = 0; kf < KSTEPS8; kf++) {
                    const uint4 av = QsAf[kf * 32]; /* (kf<<9)/16 uint4 slots */
                    unsigned a8[4];
                    a8[0] = av.x; a8[2] = av.y;
                    a8[1] = av.z; a8[3] = av.w;
#pragma unroll
                    for (int nj = 0; nj < NJ_QK; nj++) {
                        const uint2 bb =
                            *(const uint2*)&Ks8[(nn0 + nj * 8) * (HD + PAD8) + kb0 + kf * 32];
                        unsigned b8[2] = {bb.x, bb.y};
                        if (kf & 1) fa_mma_fp8_k32(accB[nj], a8, b8, accB[nj]);
                        else fa_mma_fp8_k32(acc[nj], a8, b8, acc[nj]);
                    }
                }
                /* C layout: acc[0],acc[1] = row (lane>>2), kv cols 2*(lane&3),+1; acc[2],acc[3]
                 * = row +8. Adjacent kv cols -> two STS.64 per sub-tile. */
                const int kc0 = wn * 16 + (lane & 3) * 2;
                const int qlo = wm * 16 + (lane >> 2);
                const float q0s = qsc_s[qlo], q1s = qsc_s[qlo + 8];
#pragma unroll
                for (int nj = 0; nj < NJ_QK; nj++) {
                    const int kc = kc0 + nj * 8;
                    const float ks0 = ksc_s[kc] * lscale, ks1 = ksc_s[kc + 1] * lscale;
                    const float a0 = acc[nj][0] + accB[nj][0], a1 = acc[nj][1] + accB[nj][1],
                                a2 = acc[nj][2] + accB[nj][2], a3 = acc[nj][3] + accB[nj][3];
                    *(float2*)&Ss[qlo * BKV + kc] = make_float2(a0 * ks0 * q0s, a1 * ks1 * q0s);
                    *(float2*)&Ss[(qlo + 8) * BKV + kc] =
                        make_float2(a2 * ks0 * q1s, a3 * ks1 * q1s);
                }
            }
            __syncthreads(); /* Ss published; Ks8 free for K[t+1] */

            if (t + 1 < nt) stageK(kv0 + BKV);
            else fa_cp_commit(); /* keep group counts symmetric for the V wait below */

            const unsigned rmax = (hi - kv0 < (unsigned)BKV) ? (hi - kv0) : (unsigned)BKV;

            /* ---- V[t] wait + OWN-BYTES dequant e4m3 -> fp16, v_scale FOLDED IN ----------------
             * Each thread dequants exactly the lines it staged, so no visibility barrier is
             * needed before it; the post-softmax __syncthreads publishes Vs to every P.V warp.
             * |V*vsc| is the real activation magnitude (comfortably fp16), so P stays unscaled
             * in [0,1] and the softmax path carries no per-element scale. */
            fa_cp_wait<1>();
            {
#pragma unroll
                for (int L = tid; L < BKV * HCH8; L += (int)PLOW_NV_THREADS) {
                    const int r = L / HCH8, c16 = (L % HCH8) * 16;
                    const __half2 vs2 = __float2half2_rn(vsc_s[r]);
                    const uint4 raw = *(const uint4*)&Vs8[r * (HD + PAD8) + c16];
                    __half2 h2;
                    uint2 dlo, dhi;
                    uint4 out0;
#define FA_PX23_CVT8(dst, wrd)                                                                      \
    {                                                                                               \
        __half2_raw hr0 =                                                                           \
            __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)((wrd) & 0xffffu), __NV_E4M3);         \
        __half2_raw hr1 =                                                                           \
            __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)((wrd) >> 16), __NV_E4M3);             \
        h2 = __hmul2(*(__half2*)&hr0, vs2);                                                         \
        (dst).x = *(unsigned*)&h2;                                                                  \
        h2 = __hmul2(*(__half2*)&hr1, vs2);                                                         \
        (dst).y = *(unsigned*)&h2;                                                                  \
    }
                    FA_PX23_CVT8(dlo, raw.x);
                    FA_PX23_CVT8(dhi, raw.y);
                    out0.x = dlo.x; out0.y = dlo.y; out0.z = dhi.x; out0.w = dhi.y;
                    *(uint4*)&Vs[r * (HD + PAD) + c16] = out0;
                    FA_PX23_CVT8(dlo, raw.z);
                    FA_PX23_CVT8(dhi, raw.w);
                    out0.x = dlo.x; out0.y = dlo.y; out0.z = dhi.x; out0.w = dhi.y;
                    *(uint4*)&Vs[r * (HD + PAD) + c16 + 8] = out0;
#undef FA_PX23_CVT8
                }
            }

            /* ---- REGISTER softmax, fused into the P.V A-fragments ----------------------------
             * BKV=32 is TWO k16 P.V steps, so each lane owns 8 P elements: cols
             * {c0,c0+1,c0+8,c0+9} and the same +16. The row reduction is still two quad shfls
             * (4 lanes x 8 elems = 32 cols = BKV) and P never touches smem. */
            unsigned af_pv[KSTEPS_PV][4];
            {
                float p[2][4 * KSTEPS_PV];
                float corr[2];
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int row = r0 + j * 8;
                    const int qabs = (int)(q_pos0 + q0 + row);
                    float s[4 * KSTEPS_PV], mx = FA_NEG_INF;
#pragma unroll
                    for (int ks = 0; ks < KSTEPS_PV; ks++) {
                        const float2 v0 = *(const float2*)&Ss[row * BKV + ks * 16 + c0];
                        const float2 v1 = *(const float2*)&Ss[row * BKV + ks * 16 + c0 + 8];
                        s[ks * 4 + 0] = v0.x; s[ks * 4 + 1] = v0.y;
                        s[ks * 4 + 2] = v1.x; s[ks * 4 + 3] = v1.y;
                    }
#pragma unroll
                    for (int ci = 0; ci < 4 * KSTEPS_PV; ci++) {
                        const int col = (ci >> 2) * 16 + c0 + (ci & 1) + ((ci >> 1) & 1) * 8;
                        const int kv = (int)kv0 + col;
                        bool masked = ((unsigned)col >= rmax) || (kv > qabs);
                        if (window) masked |= ((unsigned)(qabs - kv) >= window);
                        if (masked) s[ci] = FA_NEG_INF;
                        mx = fmaxf(mx, s[ci]);
                    }
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                    const float m_new = fmaxf(m_reg[j], mx);
                    corr[j] = (m_reg[j] == FA_NEG_INF) ? 0.0f : FA_EXP(m_reg[j] - m_new);
                    float lsum = 0.0f;
#pragma unroll
                    for (int ci = 0; ci < 4 * KSTEPS_PV; ci++) {
                        p[j][ci] = (s[ci] == FA_NEG_INF || m_new == FA_NEG_INF)
                                       ? 0.0f
                                       : FA_EXP(s[ci] - m_new);
                        lsum += p[j][ci];
                    }
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
                    l_reg[j] = l_reg[j] * corr[j] + lsum;
                    m_reg[j] = m_new;
                }
                /* A fragment per k16 step: [0]=(r0,klo) [1]=(r0+8,klo) [2]=(r0,khi) [3]=(r0+8,khi). */
#pragma unroll
                for (int ks = 0; ks < KSTEPS_PV; ks++) {
                    __half2 hh;
                    hh = __floats2half2_rn(p[0][ks * 4 + 0], p[0][ks * 4 + 1]);
                    af_pv[ks][0] = *(unsigned*)&hh;
                    hh = __floats2half2_rn(p[1][ks * 4 + 0], p[1][ks * 4 + 1]);
                    af_pv[ks][1] = *(unsigned*)&hh;
                    hh = __floats2half2_rn(p[0][ks * 4 + 2], p[0][ks * 4 + 3]);
                    af_pv[ks][2] = *(unsigned*)&hh;
                    hh = __floats2half2_rn(p[1][ks * 4 + 2], p[1][ks * 4 + 3]);
                    af_pv[ks][3] = *(unsigned*)&hh;
                }
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    oacc[nj][0] *= corr[0];
                    oacc[nj][1] *= corr[0];
                    oacc[nj][2] *= corr[1];
                    oacc[nj][3] *= corr[1];
                }
            }

            __syncthreads(); /* the own-bytes fp16 Vs is now visible to every P.V warp */

            /* O += P.V, fp16 mma twin. ks outer / nj inner: all NJ_PV mmas of a step are
             * independent, so the dependent chain is KSTEPS_PV deep, not NJ_PV*KSTEPS_PV. */
#pragma unroll
            for (int ks = 0; ks < KSTEPS_PV; ks++)
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    unsigned bf[2];
                    fa_ldmatrix_x2_trans(
                        bf, &Vs[(ks * 16 + (lane % 16)) * (HD + PAD) + wn * HDW + nj * 8]);
                    fa_mma_f16(oacc[nj], af_pv[ks], bf, oacc[nj]);
                }
            __syncthreads(); /* P.V done reading Vs before V[t+1] restages it */
        }

        /* Epilogue: m/l come straight from this lane's registers. */
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned qrow = (unsigned)(r0 + (e >> 1) * 8);
                const unsigned qabs_row = q0 + qrow;
                if (qabs_row >= seq_q) continue;
                const int hd = wn * HDW + nj * 8 + (lane & 3) * 2 + (e & 1);
                if (nsplit > 1) {
                    Opart[((size_t)(qabs_row * n_head + h) * nsplit + sp) * HD + hd] = oacc[nj][e];
                } else {
                    const float lv = l_reg[e >> 1];
                    const float inv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
                    O[(size_t)(qabs_row * n_head + h) * HD + hd] =
                        __float2bfloat16(oacc[nj][e] * inv);
                }
            }
        if (nsplit > 1 && wn == 0 && (lane & 3) == 0) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned qabs_row = q0 + (unsigned)(r0 + j * 8);
                if (qabs_row >= seq_q) continue;
                float* ml = mlpart + ((size_t)(qabs_row * n_head + h) * nsplit + sp) * 2;
                ml[0] = m_reg[j];
                ml[1] = l_reg[j];
            }
        }
    }
}
#endif /* PX-23 hd256 fp8 arm */

/* PX-1 STAGE 2 (varlen): `req` (default nullptr = legacy single-request, byte-identical) is the
 * packed chunk's request table [R, {q0, qlen, slot, kvlen} per request]. In varlen mode ONE
 * persistent-grid pass enumerates the UNION of every request's (q_tile, head) work items —
 * request-major, head-fastest — instead of stage 1's R serial full-grid passes (each paying its
 * own partial-wave tail). Query tiles are PER-REQUEST (tile r,qt covers rows [q0 + qt*BQ,
 * q0 + min((qt+1)*BQ, qlen)) and never spans a request boundary; tail rows past qlen stage zero
 * and are dropped by the sq epilogue bound), and each item's K/V base is slot-offset into the
 * batch-major cache — so cross-request attention is impossible by construction (block-diagonal
 * causal on full layers; per-request independent windows on sliding layers), and the per-item
 * math is IDENTICAL to the stage-1 serial per-request call (bit-exact outputs). nsplit is forced
 * to 1 per item (fused epilogue; the host neuters FLASH_MERGE in batched mode). */
template <int HD, int BQ, int BKV>
__device__ void d_flash_prefill(float* __restrict__ Opart, float* __restrict__ mlpart,
                                const __nv_bfloat16* __restrict__ Q,
                                const __nv_bfloat16* __restrict__ K,
                                const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
                                unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head,
                                unsigned q_pos0, unsigned window, unsigned nsplit,
                                unsigned kv_stride, unsigned kv_mask, float scale, unsigned slice,
                                unsigned nblk, float* lds, const int* __restrict__ req = nullptr,
                                const void* __restrict__ mapkv = nullptr) {
#if defined(PLOW_NV_HOPPER)
    /* sm_90a FORK first: when the wgmma arm claims the shape (always for <256,64,32>; for
     * <512,64,16> under PLOW_NV_FA512_WG) it beats the px4 mma.sync arm below. */
    if constexpr (FA_SM90_WG_ELIGIBLE(HD, BQ, BKV)) {
        d_flash_prefill_sm90<HD, BQ, BKV>(Opart, mlpart, Q, K, V, O, seq_q, seq_kv, n_head,
                                          n_kv_head, q_pos0, window, nsplit, kv_stride, kv_mask,
                                          scale, slice, nblk, lds, req, mapkv);
        return;
    }
#endif
    /* px4 must not even INSTANTIATE when the wgmma arm claims the shape: its static_assert
     * pins BQ==32, and the <512,64,16> flagged shape reaches here as compiled-but-dead code. */
#if defined(PLOW_NV_HOPPER)
    constexpr bool fa_px4_takes = FA_PX4_ELIGIBLE(HD) && !FA_SM90_WG_ELIGIBLE(HD, BQ, BKV);
#else
    constexpr bool fa_px4_takes = FA_PX4_ELIGIBLE(HD);
#endif
    if constexpr (fa_px4_takes) {
        /* MERGE (px4 + PX-1 stage-2): the restructured hd512 kernel has NO req/varlen awareness,
         * so route only the legacy single-request path (req==nullptr) through it. Batched varlen
         * (req!=nullptr) falls through to the block-diagonal body below on ALL HD to preserve
         * per-request masking (Gate B); px1-varlen measured full-layer varlen as locality-neutral,
         * so no perf is left on the table here. */
        if (req == nullptr) {
            d_flash_prefill_px4<HD, BQ, BKV>(Opart, mlpart, Q, K, V, O, seq_q, seq_kv, n_head,
                                             n_kv_head, q_pos0, window, nsplit, kv_stride, kv_mask,
                                             scale, slice, nblk, lds);
            return;
        }
    }
#if defined(PLOW_NV_HOPPER) && 0
    /* (folded into the branch above) */
    if constexpr (FA_SM90_WG_ELIGIBLE(HD, BQ, BKV)) {
        d_flash_prefill_sm90<HD, BQ, BKV>(Opart, mlpart, Q, K, V, O, seq_q, seq_kv, n_head,
                                          n_kv_head, q_pos0, window, nsplit, kv_stride, kv_mask,
                                          scale, slice, nblk, lds, req, mapkv);
        return;
    }
#endif
    /* T21: the shared mma.sync body must not INSTANTIATE for wgmma-claimed shapes —
     * <256,64,64>'s BKV breaks its lane-per-kv softmax static_assert. The asserts below are
     * template-dependent, so the discarded branch stays uninstantiated. */
#if defined(PLOW_NV_HOPPER)
    if constexpr (!FA_SM90_WG_ELIGIBLE(HD, BQ, BKV)) {
#else
    if constexpr (true) {
#endif
    static_assert(HD % 32 == 0, "HD must be a multiple of the warp width");
    static_assert(HD % 16 == 0 && BQ % 16 == 0 && BKV % 16 == 0, "mma m16n8k16 tiling");
    constexpr int PAD = FA_PRE_PAD;
    constexpr int WQK_N = 2;
    constexpr int WQK_M = BQ / 16;
    constexpr int WN = BKV / WQK_N;
    constexpr int NJ = WN / 8;
    constexpr int KSTEPS = HD / 16;
    static_assert(WN % 8 == 0, "kv cols per warp must be a multiple of 8");
    static_assert(WQK_M * WQK_N <= (int)PLOW_NV_WARPS, "QK warp grid exceeds the block");
    constexpr int WPV_M = BQ / 16;
    constexpr int WPV_N = (int)PLOW_NV_WARPS / WPV_M;
    constexpr int HDW = HD / WPV_N;
    constexpr int NJ_PV = HDW / 8;
    constexpr int KSTEPS_PV = BKV / 16;
    static_assert(WPV_M * WPV_N == (int)PLOW_NV_WARPS, "P.V grid must use all warps");
    static_assert(HD % WPV_N == 0 && HDW % 8 == 0, "hd slice must be a multiple of 8");
    constexpr int RPW_S = BQ / (int)PLOW_NV_WARPS;
    constexpr int SOFT_COLS = BKV > 32 ? BKV / 32 : 1;
    static_assert(BKV <= 64, "softmax reduction supports at most 2 kv cols per lane");
    static_assert(BKV <= 32 || BKV % 32 == 0, "BKV > 32 must be a multiple of 32");
    constexpr int HCH = HD / 8; /* 16B cp.async lines per K/V row */
    constexpr int VBUF = FA_PRE_VBUF(HD); /* T7 L1: 2 on hd512 (V double-buffer), 1 on hd256 */
    constexpr int VBSZ = BKV * (HD + PAD); /* one V buffer's bf16 count */

    float* Ss = lds;
    float* m_arr = Ss + BQ * BKV;
    float* l_arr = m_arr + BQ;
    float* corr_arr = l_arr + BQ;
    __nv_bfloat16* Qs = (__nv_bfloat16*)(corr_arr + BQ);   /* [BQ][HD+PAD] */
    __nv_bfloat16* Ks = Qs + BQ * (HD + PAD);              /* [BKV][HD+PAD] NATURAL (cp.async) */
    __nv_bfloat16* Vs = Ks + BKV * (HD + PAD);             /* [VBUF][BKV][HD+PAD] natural ring */
    __nv_bfloat16* Ps = Vs + VBUF * VBSZ;                  /* [BQ][BKV+PAD] softmax probs */

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned gqa = n_head / n_kv_head;
    const float lscale = FA_SCALE(scale);
    const int qk_wm = warp / WQK_N, qk_wn = warp % WQK_N;
    const bool qk_active = warp < WQK_M * WQK_N;
    const int pv_wm = warp / WPV_N, pv_wn = warp % WPV_N;

    /* Work-item count. Varlen: Σ_r ceil(qlen_r/BQ) * n_head (nsplit forced 1); qlen<=0 requests
     * contribute nothing. req is block-uniform gmem, so this scan (and the per-item decode below)
     * is a handful of broadcast loads — noise against a tile of tensor-core work. */
    unsigned n_work;
    if (req) {
        n_work = 0;
        for (int r = 0; r < req[0]; r++) {
            const int qlen = req[2 + 4 * r];
            if (qlen > 0) n_work += (unsigned)((qlen + BQ - 1) / BQ) * n_head;
        }
    } else {
        n_work = ((seq_q + BQ - 1) / BQ) * n_head * nsplit;
    }

    for (unsigned w = slice; w < n_work; w += nblk) {
        /* Decode w -> (request, q_tile, head). Per-item locals shadow the single-request params:
         * sq/skv/qp0/ns and the request's Q/O row base + slot KV base. Legacy (req==nullptr) keeps
         * the exact stage-1 decode and zero offsets — byte-identical. */
        unsigned sp, h, q0, sq = seq_q, skv = seq_kv, qp0 = q_pos0, ns = nsplit;
        size_t qoff = 0, kvoff = 0;
        if (req) {
            unsigned rem = w;
            int r = 0, qlen;
            for (;;) {
                qlen = req[2 + 4 * r];
                const unsigned nw_r =
                    (qlen > 0) ? (unsigned)((qlen + BQ - 1) / BQ) * n_head : 0u;
                if (rem < nw_r) break;
                rem -= nw_r;
                r++;
            }
            const int rq0 = req[1 + 4 * r], slot = req[3 + 4 * r], kvlen = req[4 + 4 * r];
            sp = 0;
            ns = 1;
            h = rem % n_head;
            q0 = (rem / n_head) * BQ;
            sq = (unsigned)qlen;
            skv = (unsigned)kvlen;
            qp0 = (unsigned)(kvlen - qlen);
            qoff = (size_t)rq0 * n_head * HD;
            kvoff = (size_t)slot * n_kv_head * (size_t)kv_stride * HD;
        } else {
            sp = w % nsplit;
            h = (w / nsplit) % n_head;
            q0 = (w / (nsplit * n_head)) * BQ;
        }
        const unsigned hkv = h / gqa;

        const unsigned per = (skv + ns - 1) / ns;
        const unsigned lo = sp * per;
        const unsigned hi = (lo + per < skv) ? (lo + per) : skv;

        const __nv_bfloat16* Qh = Q + qoff + (size_t)q0 * n_head * HD + (size_t)h * HD;
        const __nv_bfloat16* Kb = K + kvoff + (size_t)hkv * kv_stride * HD;
        const __nv_bfloat16* Vb = V + kvoff + (size_t)hkv * kv_stride * HD;

        __syncthreads(); /* previous item's Qs/Ks/Vs reads done before restage */
        for (int idx = tid; idx < BQ * HD; idx += (int)PLOW_NV_THREADS) {
            int r = idx / HD, c = idx % HD;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (q0 + r < sq) v = Qh[(size_t)r * n_head * HD + c];
            Qs[r * (HD + PAD) + c] = v;
        }

        float oacc[NJ_PV][4];
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) oacc[nj][e] = 0.0f;
        for (int r = tid; r < BQ; r += (int)PLOW_NV_THREADS) {
            m_arr[r] = FA_NEG_INF;
            l_arr[r] = 0.0f;
        }

        /* Tile enumeration: same union range [eff_lo, causal/hi cap) as T4, but computed up front so
         * the cp.async pipeline can prefetch across the fixed tile count. */
        const int qabs_max = (int)(qp0 + q0 + BQ - 1);
        unsigned eff_lo = lo;
        if (window) {
            const long wfloor = (long)qp0 + q0 - (long)window + 1;
            if (wfloor > (long)lo) eff_lo = ((unsigned)wfloor / BKV) * (unsigned)BKV;
        }
        long cap = (long)hi - 1;                       /* last kv0 < hi */
        if ((long)qabs_max < cap) cap = (long)qabs_max; /* causal: no tile beyond newest query */
        const int nt = (cap >= (long)eff_lo) ? (int)((cap - (long)eff_lo) / BKV) + 1 : 0;

        /* Stage helpers: K/V rows are contiguous [kv][hd]; each thread cp.async's 16B (8 bf16) lines,
         * zero-filling out-of-range kv (src-size 0). K NATURAL Ks[kv][hd] (was KsT[hd][kv]). */
        auto stageK = [&](unsigned kv0) {
            for (int L = tid; L < BKV * HCH; L += (int)PLOW_NV_THREADS) {
                int r = L / HCH, c8 = (L % HCH) * 8;
                unsigned kv = kv0 + (unsigned)r;
                bool in = (kv < hi);
                const __nv_bfloat16* g = in ? Kb + (size_t)(kv & kv_mask) * HD + c8 : Kb;
                fa_cp_async_cg16(&Ks[r * (HD + PAD) + c8], g, in ? 16 : 0);
            }
            fa_cp_commit();
        };
        auto stageV = [&](unsigned kv0, int buf) {
            __nv_bfloat16* dst = Vs + (size_t)buf * VBSZ;
            for (int L = tid; L < BKV * HCH; L += (int)PLOW_NV_THREADS) {
                int r = L / HCH, c8 = (L % HCH) * 8;
                unsigned kv = kv0 + (unsigned)r;
                bool in = (kv < hi);
                const __nv_bfloat16* g = in ? Vb + (size_t)(kv & kv_mask) * HD + c8 : Vb;
                fa_cp_async_cg16(&dst[r * (HD + PAD) + c8], g, in ? 16 : 0);
            }
            fa_cp_commit();
        };

        __syncthreads(); /* Qs + m/l init published before the pipeline reads/writes */

        if (nt > 0) {
            stageK(eff_lo);                        /* prologue: K[0] in flight */
            if (VBUF > 1) stageV(eff_lo, 0);       /* double-buffer: V[0] prefetched too */
        }

        for (int t = 0; t < nt; t++) {
            const unsigned kv0 = eff_lo + (unsigned)t * BKV;
            const int vb = t % VBUF;               /* V[t]'s buffer in the ring */
            /* Single-buffer (hd256): V[t] staged at the tile top, overlaps QK[t]+softmax[t] only.
             * Double-buffer (hd512): V[t] is ALREADY in flight (prologue or the previous iter's
             * prefetch), so nothing is staged here — V[t+1] is prefetched below, during P.V[t]. */
            if (VBUF == 1) stageV(kv0, 0);
            fa_cp_wait<1>();       /* K[t] (older group) landed; V[t] may still stream */
            __syncthreads();

            /* S = Q.K^T for this tile via mma.sync, scaled, written to Ss[BQ][BKV]. K read NON-.trans
             * from the natural Ks[kv][hd] — the T3-proven equivalent of the T4 transposed .trans read. */
            if (qk_active) {
                float acc[NJ][4];
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[nj][e] = 0.f;
#pragma unroll
                for (int kf = 0; kf < KSTEPS; kf++) {
                    unsigned af[4];
                    fa_ldmatrix_x4(af, &Qs[(qk_wm * 16 + (lane % 16)) * (HD + PAD) +
                                          kf * 16 + (lane / 16) * 8]);
                    unsigned bf[NJ][2];
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++) {
                        const int n = qk_wn * WN + nj * 8 + (lane & 7);
                        const int kcol = kf * 16 + ((lane >> 3) & 1) * 8;
                        fa_ldmatrix_x2(bf[nj], &Ks[n * (HD + PAD) + kcol]);
                    }
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++) fa_mma(acc[nj], af, bf[nj], acc[nj]);
                }
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        int qr = qk_wm * 16 + (lane / 4) + (e / 2) * 8;
                        int kc = qk_wn * WN + nj * 8 + (lane % 4) * 2 + (e % 2);
                        Ss[qr * BKV + kc] = acc[nj][e] * lscale;
                    }
            }
            __syncthreads(); /* Ss published; Ks now free to be overwritten by K[t+1] */

            /* K is dead after QK: prefetch K[t+1] now so it streams under softmax[t]+P.V[t]. On the
             * last tile issue an EMPTY commit so the group count stays right and the V wait below
             * (which must complete V[t], an older group) is never a no-op. */
            if (t + 1 < nt) stageK(kv0 + BKV);
            else fa_cp_commit();

            /* Double-buffer only: prefetch V[t+1] into the OTHER ring buffer NOW, so it streams
             * gmem->smem during softmax[t]+P.V[t] instead of stalling at the next tile's top. The
             * target buffer (t+1)%VBUF was last read by P.V[t-1] whose trailing __syncthreads freed
             * it; V[t] lives in the DISTINCT buffer vb, so this write cannot clobber the P.V[t] read.
             * Empty commit on the last tile keeps the group count symmetric with K's. */
            if (VBUF > 1) {
                if (t + 1 < nt) stageV(kv0 + BKV, (t + 1) % VBUF);
                else fa_cp_commit();
            }

            const unsigned rmax = (hi - kv0 < (unsigned)BKV) ? (hi - kv0) : (unsigned)BKV;

#pragma unroll
            for (int rr = 0; rr < RPW_S; rr++) {
                const int row = warp * RPW_S + rr;
                const int qabs = (int)(qp0 + q0 + row);
                float sv[SOFT_COLS];
                bool active[SOFT_COLS];
#pragma unroll
                for (int sc = 0; sc < SOFT_COLS; sc++) {
                    sv[sc] = FA_NEG_INF;
                    active[sc] = false;
                    const unsigned col = lane + sc * 32;
                    if (col < rmax) {
                        const int kv = (int)kv0 + col;
                        bool masked = (kv > qabs);
                        if (window) masked |= ((unsigned)(qabs - kv) >= window);
                        if (!masked) { sv[sc] = Ss[row * BKV + col]; active[sc] = true; }
                    }
                }
                float local_max = sv[0];
#pragma unroll
                for (int sc = 1; sc < SOFT_COLS; sc++) local_max = fmaxf(local_max, sv[sc]);
                const float rowmax = warp_max32(local_max);
                const float m_old = m_arr[row];
                const float m_new = fmaxf(m_old, rowmax);
                const float corr = (m_old == FA_NEG_INF) ? 0.0f : FA_EXP(m_old - m_new);
                float psum = 0.0f;
#pragma unroll
                for (int sc = 0; sc < SOFT_COLS; sc++) {
                    const unsigned col = lane + sc * 32;
                    const float p = (active[sc] && m_new != FA_NEG_INF)
                                      ? FA_EXP(sv[sc] - m_new) : 0.0f;
                    if (col < (unsigned)BKV)
                        Ps[row * (BKV + PAD) + col] = __float2bfloat16(p);
                    psum += p;
                }
                const float rowsum = warp_sum32(psum);
                if (lane == 0) {
                    l_arr[row] = l_arr[row] * corr + rowsum;
                    m_arr[row] = m_new;
                    corr_arr[row] = corr;
                }
            }
            /* Drain V[t]. Single-buffer: {V[t], K[t+1]/empty} in flight, wait<1> completes V[t].
             * Double-buffer: {K[t+1]/empty, V[t+1]/empty} are ALSO in flight, so V[t] is 3rd-oldest
             * of {V[t], K[t+1], V[t+1]} — wait<2> completes it while leaving K[t+1]+V[t+1] streaming. */
            if (VBUF > 1) fa_cp_wait<2>();
            else fa_cp_wait<1>();
            __syncthreads();

            __nv_bfloat16* Vd = Vs + (size_t)vb * VBSZ; /* this tile's V ring buffer */
            {
                const float c_lo = corr_arr[pv_wm * 16 + (lane >> 2)];
                const float c_hi = corr_arr[pv_wm * 16 + (lane >> 2) + 8];
#pragma unroll
                for (int nj = 0; nj < NJ_PV; nj++) {
                    oacc[nj][0] *= c_lo;
                    oacc[nj][1] *= c_lo;
                    oacc[nj][2] *= c_hi;
                    oacc[nj][3] *= c_hi;
                }
#pragma unroll
                for (int kf = 0; kf < KSTEPS_PV; kf++) {
                    unsigned af[4];
                    fa_ldmatrix_x4(af, &Ps[(pv_wm * 16 + (lane % 16)) * (BKV + PAD) +
                                          kf * 16 + (lane / 16) * 8]);
#pragma unroll
                    for (int nj = 0; nj < NJ_PV; nj++) {
                        unsigned bf[2];
                        fa_ldmatrix_x2_trans(bf, &Vd[(kf * 16 + (lane % 16)) * (HD + PAD) +
                                                     pv_wn * HDW + nj * 8]);
                        fa_mma(oacc[nj], af, bf, oacc[nj]);
                    }
                }
            }
            __syncthreads(); /* P.V done reading Vs[vb] before it is reused VBUF tiles later */
        }

        /* Epilogue — identical to T4; varlen (ns==1) writes land at the request's O row base. */
#pragma unroll
        for (int nj = 0; nj < NJ_PV; nj++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned qrow = (unsigned)(pv_wm * 16 + (lane >> 2) + (e >> 1) * 8);
                const unsigned qabs_row = q0 + qrow;
                if (qabs_row >= sq) continue;
                const int hd = pv_wn * HDW + nj * 8 + (lane & 3) * 2 + (e & 1);
                if (ns > 1) {
                    Opart[((size_t)(qabs_row * n_head + h) * ns + sp) * HD + hd] = oacc[nj][e];
                } else {
                    const float lv = l_arr[qrow];
                    const float inv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
                    O[qoff + (size_t)(qabs_row * n_head + h) * HD + hd] =
                        __float2bfloat16(oacc[nj][e] * inv);
                }
            }
        if (ns > 1) {
            for (int qrow = tid; qrow < BQ; qrow += (int)PLOW_NV_THREADS) {
                const unsigned qabs_row = q0 + (unsigned)qrow;
                if (qabs_row >= sq) continue;
                float* ml = mlpart + ((size_t)(qabs_row * n_head + h) * ns + sp) * 2;
                ml[0] = m_arr[qrow];
                ml[1] = l_arr[qrow];
            }
        }
    }
    }
}
#endif /* PLOW_NV_FA_PIPE */

/* ---- PX-1 cross-request batched prefill: block-diagonal varlen flash (STAGE 2) -----------------
 * `req` (host-patched t6, nullptr on every legacy packet) is the packed chunk's request table:
 *   req[0] = R;  req[1+4r..] = { q0, qlen, slot, kvlen } per request r.
 * The packed launch's GEMMs run all requests' rows as one M = Σ qlen matrix (rows are
 * independent), but attention is BLOCK-DIAGONAL: request r's queries (rows [q0, q0+qlen) of the
 * activation tile, absolute positions [kvlen-qlen, kvlen)) attend ONLY request r's KV, which
 * lives in seq-slot `slot` of the batch-major cache (+ slot*n_kv_head*kv_stride*HD elements).
 * STAGE 2 (PIPE=1, the shipped object): for the sliding (hd256) layers d_flash_prefill enumerates
 * all requests' query tiles in ONE persistent-grid pass (the standard varlen/cu_seqlens layout,
 * req as the seqlens table) — no more R serial passes each paying a partial-wave tail. The hd512
 * FULL layers stay serial-per-request (see the merge note in d_flash_prefill_mux: px4 has no varlen
 * arm and fused hd512 varlen is locality-negative). Per-item math is identical to the stage-1
 * serial call, so outputs are bit-exact either way. The PIPE=0 A/B control object keeps the
 * stage-1 serial loop (its d_flash_prefill has no varlen arm). */
template <int HD, int BQ, int BKV>
__device__ void d_flash_prefill_mux(const int* __restrict__ req, float* __restrict__ Opart,
                                    float* __restrict__ mlpart,
                                    const __nv_bfloat16* __restrict__ Q,
                                    const __nv_bfloat16* __restrict__ K,
                                    const __nv_bfloat16* __restrict__ V,
                                    __nv_bfloat16* __restrict__ O, unsigned seq_q, unsigned seq_kv,
                                    unsigned n_head, unsigned n_kv_head, unsigned q_pos0,
                                    unsigned window, unsigned nsplit, unsigned kv_stride,
                                    unsigned kv_mask, float scale, unsigned slice, unsigned nblk,
                                    float* lds, const void* __restrict__ mapkv = nullptr) {
#if PLOW_NV_PACKED_REQUEST
    if (req) {
        const unsigned count=(unsigned)req[0];
        for (unsigned r=0; r<count; ++r) {
            const unsigned q0=req[1+4*r], qlen=req[2+4*r], slot=req[3+4*r], kvlen=req[4+4*r];
            const size_t qoff=(size_t)q0*n_head*HD;
            const size_t kvoff=(size_t)slot*n_kv_head*kv_stride*HD;
            const void* descriptor=mapkv ? (const void*)((const uint64_t*)mapkv)[slot] : nullptr;
            d_flash_prefill<HD,BQ,BKV>(Opart+qoff*nsplit,
                mlpart+(size_t)q0*n_head*nsplit*2, Q+qoff,K+kvoff,V+kvoff,
                O ? O+qoff : nullptr,qlen,kvlen,n_head,n_kv_head,kvlen-qlen,
                window,nsplit,kv_stride,kv_mask,scale,slice,nblk,lds,nullptr,descriptor);
            __syncthreads();
        }
        if (O) {
            const unsigned real=req[1+4*(count-1)]+req[2+4*(count-1)];
            const size_t begin=(size_t)real*n_head*HD, end=(size_t)seq_q*n_head*HD;
            for (size_t i=begin+(size_t)slice*blockDim.x+threadIdx.x;i<end;i+=(size_t)nblk*blockDim.x)
                O[i]=__float2bfloat16(0.0f);
        }
        return;
    }
#endif
    if (req && O == nullptr) __trap(); /* legacy fused request ABI */
    /* MERGE (px4 + PX-1 stage-2) routing. The fused varlen body handles the sliding (hd256) layers
     * in ONE block-diagonal pass — the PX-1 stage-2 win (1.30-2.57x at R=2..8). The hd512 FULL
     * layers instead run each request SERIALLY (offset bases): px4's restructured single-request
     * kernel has no varlen arm, fused hd512 varlen was measured locality-negative (no win over
     * serial), and serial-per-request stays BIT-EXACT with the single-request px4 path (which the
     * non-batched prefill uses). PIPE=0 (the A/B control object) always serializes. FA_PX4_ELIGIBLE
     * already folds in PLOW_NV_FA_PIPE, so OR in !PIPE for the control object. */
    constexpr bool USE_SERIAL = FA_PX4_ELIGIBLE(HD) || !PLOW_NV_FA_PIPE;
    if constexpr (!USE_SERIAL) {
        d_flash_prefill<HD, BQ, BKV>(Opart, mlpart, Q, K, V, O, seq_q, seq_kv, n_head, n_kv_head,
                                     q_pos0, window, nsplit, kv_stride, kv_mask, scale, slice, nblk,
                                     lds, req, mapkv);
    } else {
        const int R = req ? req[0] : 1;
        for (int r = 0; r < R; r++) {
            size_t qoff = 0, kvoff = 0;
            unsigned sq = seq_q, skv = seq_kv, qp0 = q_pos0, ns = nsplit;
            if (req) {
                const int q0 = req[1 + 4 * r], qlen = req[2 + 4 * r], slot = req[3 + 4 * r],
                          kvlen = req[4 + 4 * r];
                if (qlen <= 0) continue; /* uniform: req is identical for every thread */
                qoff = (size_t)q0 * n_head * HD;
                kvoff = (size_t)slot * n_kv_head * kv_stride * HD;
                sq = (unsigned)qlen;
                skv = (unsigned)kvlen;
                qp0 = (unsigned)(kvlen - qlen);
                ns = 1;
            }
            d_flash_prefill<HD, BQ, BKV>(Opart, mlpart, Q + qoff, K + kvoff, V + kvoff,
                                         O ? O + qoff : O, sq, skv, n_head, n_kv_head, qp0, window,
                                         ns, kv_stride, kv_mask, scale, slice, nblk, lds);
        }
    }
}

#if PLOW_MIXED_STEP
template <int HD, int BQ, int BKV>
__device__ void d_flash_prefill_mixed(const PlowProgram* prog, float* __restrict__ Opart,
                                     float* __restrict__ mlpart,
                                     const __nv_bfloat16* __restrict__ Q,
                                     const __nv_bfloat16* __restrict__ K,
                                     const __nv_bfloat16* __restrict__ V,
                                     __nv_bfloat16* __restrict__ O, unsigned seq_q,
                                     unsigned n_head, unsigned n_kv_head, unsigned window,
                                     unsigned nsplit, unsigned kv_stride, unsigned kv_mask,
                                     float scale, unsigned slice, unsigned nblk, float* lds,
                                     const void* __restrict__ mapkv = nullptr) {
    if (!plow_mixed_step_enabled(prog) || seq_q != prog->n_prefill_rows || nsplit != 1u || !O)
        __trap();
    for (unsigned si = 0; si < prog->n_prefill_spans; ++si) {
        const PlowPrefillSpan* span = plow_mixed_prefill_span(prog, si);
        const size_t qoff = (size_t)span->row0 * n_head * HD;
        const size_t kvoff = (size_t)span->slot * n_kv_head * kv_stride * HD;
        const void* descriptor = mapkv ? (const void*)((const uint64_t*)mapkv)[span->slot] : nullptr;
        d_flash_prefill<HD, BQ, BKV>(
            Opart + qoff, mlpart + (size_t)span->row0 * n_head * 2u, Q + qoff, K + kvoff,
            V + kvoff, O + qoff, span->n_rows, span->kv_len, n_head, n_kv_head,
            span->kv_row0, window, 1u, kv_stride, kv_mask, scale, slice, nblk, lds, nullptr,
            descriptor);
        __syncthreads();
    }
}
#endif
