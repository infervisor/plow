/* op_moe_nvfp4_pf.cuh — W4A4 prefill arm of the NVFP4 latent MoE (Nemotron-3.5), sm_120a/sm_121a.
 *
 * The peer's arithmetic (vLLM FLASHINFER_CUTLASS NvFp4 MoE / FlashInferCutlassNvFp4Linear):
 * activations quantized to NVFP4 (e2m1 + one e4m3 scale per 16, global scale = 1 / max of the
 * layer's input_scale), mma.sync m16n8k64 kind::mxf4nvf4 block_scale ue4m3 (f32 accumulate),
 * output scaled by alpha = input_scale_max * weight_scale_2[e]. The up epilogue rounds
 * gs * acc to bf16 (the GEMM output dtype), applies relu^2 and quantizes the result to NVFP4 for
 * the down projection (the second input_scale), so the intermediate never exists in bf16.
 *
 * NVFP4 activation tensor ("q tensor") of R rows x K: [R][K/2] e2m1 (low nibble = even k) |
 * [R][K/16] e4m3 | f32 alpha_a (= the input_scale max the rows were quantized with).
 */
#pragma once
#include "op_moe_nvfp4.cuh"

#define NF4W_BN 128
#define NF4W_BKB 64 /* k bytes per stage (128 e2m1) */
#define NF4W_ST 80  /* smem row stride (bytes): 5 x 16, ldmatrix conflict-free */
#define NF4W_STAGES 3
/* Ring depth of the 64-row tile (routed experts): a stage is 3/4 of the 128-row tile's bytes. */
#ifndef PLOW_NV_W4A4_STAGES64
#define PLOW_NV_W4A4_STAGES64 3
#endif
#define NF4W_RING(BM, S) ((S) * ((BM) + NF4W_BN) * (NF4W_ST + 8))
/* v2 kernel (nf4v_tiled): 3 stages of (128 + BN) rows x (64 B + 8 B scales), then the tile prefix. */
#ifndef PLOW_NV_W4A4_V2
#define PLOW_NV_W4A4_V2 1
#endif
/* Fewest tokens for v2 (v1 below). 0 since nf4v_tiled has its own register allocation: v2 is
 * faster from 128 rows (L1 real text, routed up + down: 128 rows -0.80 ms, 1024 rows -0.91 ms). */
#ifndef PLOW_NV_W4A4_V2_MIN_T
#define PLOW_NV_W4A4_V2_MIN_T 0u
#endif
/* Below this many tokens each v2 item bulk-prefetches its weight tile into L2 at setup (the
 * weight-streaming regime: one item per weight tile; on top of v2, 128 rows -0.15 ms, 1024 rows
 * -0.23 ms). From 4096 rows an expert's row tiles share each weight tile and the prefetches
 * thrash L2 (+4.7 ms at 4096). 0 = off. */
#ifndef PLOW_NV_W4A4_V2_PF_MAXT
#define PLOW_NV_W4A4_V2_PF_MAXT 2048u
#endif
/* N-major item order inside an expert: bit 0 up, bit 1 down. */
#ifndef PLOW_NV_W4A4_V2_NMAJOR
#define PLOW_NV_W4A4_V2_NMAJOR 2
#endif
/* L2 prefetch distance of the weight tiles in waves of nblk items (0 = off). */
#ifndef PLOW_NV_W4A4_V2_PFD
#define PLOW_NV_W4A4_V2_PFD 0u
#endif
/* Up epilogue: q rows staged in the free ring slot and written by bulk async copies under the
 * next item's loads (0 = direct stores). */
#ifndef PLOW_NV_W4A4_V2_BULK
#define PLOW_NV_W4A4_V2_BULK 1
#endif
/* Ring stages. 2 keeps the arena floor at v1's 67.6 KB (3: 92 KB). */
#ifndef PLOW_NV_W4A4_V2_NST
#define PLOW_NV_W4A4_V2_NST 3
#endif
#define NF4V_NST ((unsigned)PLOW_NV_W4A4_V2_NST)
/* Routed up below PP_MAXT tokens: two 4-warp groups per block ping-pong on 128 x 96 items, each
 * with its own 2-stage ring, so one group's epilogue runs under the other's MMAs (L1 block, real
 * text: 128 rows -0.13 ms, 1024 rows -0.11 ms). From 4096 rows the doubled A-tile traffic and the
 * shallower rings cost more than the hidden epilogue (+0.5 ms at 4096 and 8192). 0 = off. */
#ifndef PLOW_NV_W4A4_V2_PP
#define PLOW_NV_W4A4_V2_PP 1
#endif
#ifndef PLOW_NV_W4A4_V2_PP_MAXT
#define PLOW_NV_W4A4_V2_PP_MAXT 2048u
#endif
#define NF4V_STAGE(BN) ((128u + (BN)) * 72u)
#define NF4V_EMAX 512u
/* After the ring (u32): tile prefix [EMAX + 1], 15 scan slots, rowoff [EMAX], cnt [EMAX], then per
 * item parity 128 row ids (up: row_token) or 128 part indices + 128 gates (down). */
#define NF4V_META_U32 (3u * NF4V_EMAX + 16u + 3u * 256u)
#define NF4V_ARENA_BYTES (NF4V_NST * NF4V_STAGE(256u) + 4u * NF4V_META_U32)
/* Ring bytes of the widest tile in use (BM = 128 unless pinned to 64): the interpreter's arena floor. */
#if defined(PLOW_NV_W4A4_BM) && PLOW_NV_W4A4_BM == 64
#define NF4W_ARENA_V1 NF4W_RING(64, PLOW_NV_W4A4_STAGES64)
#else
#define NF4W_ARENA_V1                                                                               \
    (NF4W_RING(128, NF4W_STAGES) > NF4W_RING(64, PLOW_NV_W4A4_STAGES64) ? NF4W_RING(128, NF4W_STAGES) \
                                                                         : NF4W_RING(64, PLOW_NV_W4A4_STAGES64))
#endif
#if PLOW_NV_W4A4_V2
#define NF4W_ARENA_BYTES (NF4W_ARENA_V1 > NF4V_ARENA_BYTES ? NF4W_ARENA_V1 : NF4V_ARENA_BYTES)
#else
#define NF4W_ARENA_BYTES NF4W_ARENA_V1
#endif

__device__ __forceinline__ size_t nf4w_sf_off(unsigned rows, unsigned K) { return (size_t)rows * (K / 2u); }
__device__ __forceinline__ size_t nf4w_alpha_off(unsigned rows, unsigned K) {
    return (size_t)rows * (K / 2u) + (size_t)rows * (K / 16u);
}

__device__ __forceinline__ void nf4w_cp8(void* smem, const void* g, unsigned bytes) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(g), "r"(bytes));
}
template <int N>
__device__ __forceinline__ void nf4w_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void nf4w_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                         unsigned sfa, unsigned sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

/* Eight f32 -> four bytes of e2m1 pairs (element 2i in the low nibble of byte i), round to
 * nearest even, saturating (vLLM's cvt_warp_fp16_to_fp4 sequence). */
__device__ __forceinline__ unsigned nf4w_e2m1x8(float f0, float f1, float f2, float f3, float f4, float f5,
                                                float f6, float f7) {
    unsigned r;
    asm("{\n.reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n}\n"
        : "=r"(r)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3), "f"(f4), "f"(f5), "f"(f6), "f"(f7));
    return r;
}
/* e4m3 scale of a 16-block (vLLM nvfp4 quant): sf = e4m3(gs * amax / 6) (saturating),
 * multiplier = gs / f32(sf) (0 for a zero block). Returns the multiplier; *byte the e4m3 bits. */
__device__ __forceinline__ float nf4w_block_scale(float amax, float gs, unsigned& byte) {
    const __nv_fp8_e4m3 sf(fminf(gs * (amax * (1.0f / 6.0f)), 448.0f));
    byte = (unsigned)sf.__x;
    const float v = (float)sf;
    return v != 0.f ? 1.0f / (v * (1.0f / gs)) : 0.f;
}

/* Block max of n f32 (input_scale vector) broadcast through `red` (8 floats of arena). */
__device__ __forceinline__ float nf4w_max_scale(const float* __restrict__ s, unsigned n, float* red) {
    float m = 0.f;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, s[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    if ((threadIdx.x & 31u) == 0u) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
    for (unsigned w = 1; w < blockDim.x / 32u; w++) m = fmaxf(m, red[w]);
    __syncthreads();
    return m;
}

/* Dense down mode 3: bf16(radd + bf16(y)) per pair, radd bf16 [T][N] passed in the row_partidx slot
 * (the separate Residual op's math: the shared output rounds to bf16 before the add). */
__device__ __forceinline__ unsigned nf4w_radd(const unsigned* radd, size_t i, float y0, float y1) {
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const __nv_bfloat16*>(radd) + i);
    return nf4_f2bf2(__low2float(a) + __bfloat162float(__float2bfloat16(y0)),
                     __high2float(a) + __bfloat162float(__float2bfloat16(y1)));
}

/* MOE_QUANT_NVFP4 (op 239): q tensor of x [R][K] bf16 with gs = 1 / max(in_scale[0..n)).
 * Thread per 16-element block. */
static __device__ __noinline__ void d_moe_quant_nvfp4(uint8_t* __restrict__ q, const __nv_bfloat16* __restrict__ x,
                                                      const float* __restrict__ in_scale, unsigned R,
                                                      unsigned K, unsigned n, unsigned slice,
                                                      unsigned nblk, float* arena) {
    if (K % 32u || n == 0u || blockDim.x != 256u) { __trap(); return; }
    const float amax_in = nf4w_max_scale(in_scale, n, arena);
    const float gs = 1.0f / amax_in;
    uint8_t* sf = q + nf4w_sf_off(R, K);
    if (slice == 0u && threadIdx.x == 0u) *reinterpret_cast<float*>(q + nf4w_alpha_off(R, K)) = amax_in;
    const size_t blocks = (size_t)R * (K / 16u);
    for (size_t b = (size_t)slice * blockDim.x + threadIdx.x; b < blocks; b += (size_t)nblk * blockDim.x) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(x + b * 16u);
        const uint4 v1 = *reinterpret_cast<const uint4*>(x + b * 16u + 8u);
        const __nv_bfloat16* p0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
        const __nv_bfloat16* p1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
        float f[16], m = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            f[j] = __bfloat162float(p0[j]);
            f[8 + j] = __bfloat162float(p1[j]);
            m = fmaxf(m, fmaxf(fabsf(f[j]), fabsf(f[8 + j])));
        }
        unsigned byte;
        const float mul = nf4w_block_scale(m, gs, byte);
        uint2 o;
        o.x = nf4w_e2m1x8(f[0] * mul, f[1] * mul, f[2] * mul, f[3] * mul, f[4] * mul, f[5] * mul,
                          f[6] * mul, f[7] * mul);
        o.y = nf4w_e2m1x8(f[8] * mul, f[9] * mul, f[10] * mul, f[11] * mul, f[12] * mul, f[13] * mul,
                          f[14] * mul, f[15] * mul);
        *reinterpret_cast<uint2*>(q + b * 8u) = o;
        sf[b] = (uint8_t)byte;
    }
}

/* Grouped W4A4 GEMM tile loop: C[BM rows x 128 cols] per item, 128 k (64 B) per stage, 3-stage
 * cp.async ring of raw e2m1 bytes and e4m3 scales (A and B), 8 warps: BM = 64 -> 2 (m) x 4 (n)
 * warps of 32 x 32, BM = 128 -> 4 x 2 warps of 32 x 64 (twice the MMAs per staged byte and per
 * B fragment). ldmatrix of the byte tiles goes straight into the m16n8k64 fragments (a 16 x 32 B
 * tile has the A layout of 16 x 64 e2m1; B rows are the weight rows [n][k/2]).
 * Routed tiles come from MoeAlign's 64-row tile prefix: BM = 128 takes the even local tiles and
 * covers rows [64 lt, 64 lt + 128) of the expert (the odd items are empty).
 * MODE 1 up: out = q tensor of relu(bf16(alpha acc))^2 with gs2 = 1 / max(s2[0..n2)), rows in
 *   the expert-sorted order (routed: T*k rows; dense: T rows).
 * MODE 2 down: part[row_partidx[r]][col] = row_gate[r] alpha acc (dmode 0) or bf16 out (dmode 1). */
template <int MODE, int BM>
static __device__ void nf4w_tiled(void* __restrict__ out, const uint8_t* __restrict__ aq,
                                  const int* __restrict__ meta, const uint8_t* __restrict__ W,
                                  const uint8_t* __restrict__ S, const float* __restrict__ gs,
                                  const unsigned* __restrict__ row_token,
                                  const unsigned* __restrict__ row_partidx,
                                  const float* __restrict__ row_gate, const float* __restrict__ s2,
                                  unsigned n2, unsigned arows, unsigned orows, unsigned T,
                                  unsigned N, unsigned K, unsigned E, unsigned dmode,
                                  unsigned slice, unsigned nblk, uint8_t* arena) {
    constexpr unsigned BN = NF4W_BN, WM = BM / 32u, WN = 8u / WM, WT = BN / WN, NJ = WT / 8u;
    constexpr unsigned STAGE = (BM + BN) * (NF4W_ST + 8u), AR = BM / 64u;
    constexpr unsigned NST = BM == 64u ? PLOW_NV_W4A4_STAGES64 : NF4W_STAGES;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned wm = warp / WN, wn = warp % WN, g = lane >> 2, tq = lane & 3u;
    const unsigned ntn = N / BN, kb = K / 2u, ks16 = K / 16u, nk = kb / NF4W_BKB;
    const unsigned mtiles = meta ? (unsigned)meta[3u * E] : (T + BM - 1u) / BM;
    const unsigned items = mtiles * ntn;
    const uint8_t* asf = aq + nf4w_sf_off(arows, K);
    const float alpha_a = *reinterpret_cast<const float*>(aq + nf4w_alpha_off(arows, K));
    float gs2 = 0.f;
    if (MODE == 1) {
        const float m2 = nf4w_max_scale(s2, n2, reinterpret_cast<float*>(arena));
        gs2 = 1.0f / m2;
        if (slice == 0u && tid == 0u) *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(out) + nf4w_alpha_off(orows, N)) = m2;
    }
    /* Staging roles: A rows tid/4 (+ 64 h), 16 B chunk tid%4; B rows tid/4 and 64 + tid/4;
     * scales: threads [0, BM) the A rows, [BM, BM + 128) the B rows. */
    const unsigned ar = tid >> 2, ac = tid & 3u;
    struct Ctx {
        unsigned e, r0, rc, n0, sfok;
        const uint8_t* asrc[AR];
        bool av[AR];
        const uint8_t* bsrc0;
        const uint8_t* sfsrc;
    };
    /* false: an empty item (the odd 64-row tile of a 128-row pair). Block-uniform. */
    auto setup = [&](unsigned item, Ctx& c) -> bool {
        const unsigned mt = item / ntn;
        c.n0 = (item - mt * ntn) * BN;
        c.e = 0u;
        if (meta) {
            c.e = nf4_expert_of_tile(meta + 2u * E, mt, E);
            const unsigned lt = mt - (unsigned)meta[2u * E + c.e];
            if (BM == 128 && (lt & 1u)) return false;
            c.r0 = (unsigned)meta[c.e] + lt * 64u;
            c.rc = min((unsigned)BM, (unsigned)meta[E + c.e] - lt * 64u);
        } else {
            c.r0 = mt * BM;
            c.rc = min((unsigned)BM, T - c.r0);
        }
#pragma unroll
        for (unsigned h = 0; h < AR; h++) {
            const unsigned rr = ar + 64u * h;
            c.av[h] = rr < c.rc;
            unsigned arow = c.r0 + (c.av[h] ? rr : 0u);
            if (MODE == 1 && meta) arow = row_token[arow];
            c.asrc[h] = aq + (size_t)arow * kb + ac * 16u;
        }
        c.bsrc0 = W + ((size_t)c.e * N + c.n0 + ar) * kb + ac * 16u;
        c.sfsrc = nullptr;
        c.sfok = 0u;
        if (tid < BM) {
            const bool ok = tid < c.rc;
            unsigned sr = c.r0 + (ok ? tid : 0u);
            if (MODE == 1 && meta) sr = row_token[sr];
            c.sfsrc = asf + (size_t)sr * ks16;
            c.sfok = ok ? 8u : 0u;
        } else if (tid < BM + BN) {
            c.sfsrc = S + ((size_t)c.e * N + c.n0 + tid - BM) * ks16;
            c.sfok = 8u;
        }
        return true;
    };
    auto stage = [&](const Ctx& c, unsigned ks, unsigned slot) {
        uint8_t* base = arena + slot * STAGE;
        uint8_t* As = base;
        uint8_t* Bs = base + BM * NF4W_ST;
        uint8_t* Ss = base + (BM + BN) * NF4W_ST; /* [BM A rows | 128 B rows] x 8 B */
        const unsigned ko = ks * NF4W_BKB;
#pragma unroll
        for (unsigned h = 0; h < AR; h++)
            nf4_cp16(As + (ar + 64u * h) * NF4W_ST + ac * 16u, c.asrc[h] + ko, c.av[h] ? 16u : 0u);
        nf4_cp16(Bs + ar * NF4W_ST + ac * 16u, c.bsrc0 + ko, 16u);
        nf4_cp16(Bs + (64u + ar) * NF4W_ST + ac * 16u, c.bsrc0 + (size_t)64u * kb + ko, 16u);
        if (tid < BM + BN) nf4w_cp8(Ss + tid * 8u, c.sfsrc + ks * 8u, c.sfok);
    };
    auto prologue = [&](const Ctx& c) {
#pragma unroll
        for (unsigned s = 0; s < NST - 1u; s++) {
            if (s < nk) stage(c, s, s);
            nf4_cp_commit();
        }
    };
    auto next_item = [&](unsigned item, Ctx& c) {
        while (item < items && !setup(item, c)) item += nblk;
        return item;
    };
    Ctx cur;
    unsigned item = next_item(slice, cur);
    if (item < items) prologue(cur);
    while (item < items) {
        float acc[2][NJ][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < (int)NJ; j++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;
        for (unsigned ks = 0; ks < nk; ks++) {
            nf4w_cp_wait<NST - 2>();
            __syncthreads();
            if (ks + NST - 1u < nk) stage(cur, ks + NST - 1u, (ks + NST - 1u) % NST);
            nf4_cp_commit();
            const uint8_t* base = arena + (ks % NST) * STAGE;
            const uint8_t* As = base;
            const uint8_t* Bs = base + BM * NF4W_ST;
            const uint8_t* Ss = base + (BM + BN) * NF4W_ST;
#pragma unroll
            for (unsigned sub = 0; sub < 2u; sub++) {
                unsigned af[2][4], bf[NJ / 2][4], sfa[2], sfb[NJ];
#pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    nf4_ldsm_x4(af[mi], As + (wm * 32u + mi * 16u + (lane & 15u)) * NF4W_ST + sub * 32u + (lane >> 4) * 16u);
                    sfa[mi] = *reinterpret_cast<const unsigned*>(Ss + (wm * 32u + mi * 16u + g + 8u * (lane & 1u)) * 8u + sub * 4u);
                }
#pragma unroll
                for (int p = 0; p < (int)(NJ / 2); p++)
                    nf4_ldsm_x4(bf[p], Bs + (wn * WT + p * 16u + (lane & 7u) + ((lane >> 4) << 3)) * NF4W_ST +
                                           sub * 32u + ((lane >> 3) & 1u) * 16u);
#pragma unroll
                for (int nj = 0; nj < (int)NJ; nj++)
                    sfb[nj] = *reinterpret_cast<const unsigned*>(Ss + (BM + wn * WT + nj * 8u + g) * 8u + sub * 4u);
#pragma unroll
                for (int mi = 0; mi < 2; mi++)
#pragma unroll
                    for (int nj = 0; nj < (int)NJ; nj++)
                        nf4w_mma(acc[mi][nj], af[mi], bf[nj >> 1][(nj & 1) * 2], bf[nj >> 1][(nj & 1) * 2 + 1],
                                 sfa[mi], sfb[nj]);
            }
        }
        nf4w_cp_wait<0>();
        /* Every warp is done with the ring: the next item's prologue loads now overlap this
         * item's epilogue (registers and global only). */
        __syncthreads();
        Ctx nxt;
        const unsigned item_next = next_item(item + nblk, nxt);
        if (item_next < items) prologue(nxt);
        const unsigned e = cur.e, r0 = cur.r0, rc = cur.rc, n0 = cur.n0;
        const float alpha = alpha_a * gs[e];
        if (MODE == 1) {
            uint8_t* oq = reinterpret_cast<uint8_t*>(out);
            uint8_t* osf = oq + nf4w_sf_off(orows, N);
            const unsigned ob = N / 2u, os = N / 16u;
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const unsigned lr = wm * 32u + mi * 16u + g + 8u * h;
                    const size_t r = r0 + lr;
#pragma unroll
                    for (int b = 0; b < (int)(NJ / 2); b++) {
                        float v[4];
#pragma unroll
                        for (int u = 0; u < 2; u++)
#pragma unroll
                            for (int c = 0; c < 2; c++) {
                                const float y = __bfloat162float(__float2bfloat16(acc[mi][2 * b + u][2 * h + c] * alpha));
                                const float rl = fmaxf(y, 0.f);
                                v[2 * u + c] = rl * rl;
                            }
                        float m = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
                        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
                        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
                        unsigned byte;
                        const float mul = nf4w_block_scale(m, gs2, byte);
                        const unsigned pk = nf4w_e2m1x8(v[0] * mul, v[1] * mul, v[2] * mul, v[3] * mul, 0.f,
                                                        0.f, 0.f, 0.f);
                        unsigned lo = (pk & 0xFFu) << (8u * tq);
                        unsigned hi = ((pk >> 8) & 0xFFu) << (8u * tq);
                        lo |= __shfl_xor_sync(0xFFFFFFFFu, lo, 1);
                        hi |= __shfl_xor_sync(0xFFFFFFFFu, hi, 1);
                        lo |= __shfl_xor_sync(0xFFFFFFFFu, lo, 2);
                        hi |= __shfl_xor_sync(0xFFFFFFFFu, hi, 2);
                        if (tq == 0u && lr < rc) {
                            const unsigned col = n0 + wn * WT + b * 16u;
                            *reinterpret_cast<uint2*>(oq + r * ob + col / 2u) = make_uint2(lo, hi);
                            osf[r * os + col / 16u] = (uint8_t)byte;
                        }
                    }
                }
        } else {
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const unsigned lr = wm * 32u + mi * 16u + g + 8u * h;
                    if (lr >= rc) continue;
                    const unsigned r = r0 + lr;
                    size_t orow = r;
                    float rg = 1.0f;
                    if (dmode == 0u || dmode == 2u) orow = row_partidx[r];
                    if (dmode == 0u) rg = row_gate[r];
#pragma unroll
                    for (int nj = 0; nj < (int)NJ; nj++) {
                        const unsigned col = n0 + wn * WT + nj * 8u + 2u * tq;
                        const float y0 = acc[mi][nj][2 * h] * alpha, y1 = acc[mi][nj][2 * h + 1] * alpha;
                        if (dmode == 0u)
                            *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + orow * N + col) =
                                make_float2(y0 * rg, y1 * rg);
                        else if (dmode == 3u)
                            *reinterpret_cast<unsigned*>(reinterpret_cast<__nv_bfloat16*>(out) + orow * N + col) =
                                nf4w_radd(row_partidx, orow * N + col, y0, y1);
                        else
                            *reinterpret_cast<unsigned*>(reinterpret_cast<__nv_bfloat16*>(out) + orow * N + col) =
                                nf4_f2bf2(y0, y1);
                    }
                }
        }
        cur = nxt;
        item = item_next;
    }
}

/* v2 grouped W4A4 GEMM: C[128 x BN] per item, 8 warps 2 (m) x 4 (n) of 64 x BN/4 (4 m16 x BN/32 n8
 * per k64: 3-4x the MMAs per ldmatrix and scale load of v1's 32 x 32). An expert's rows are cut
 * into ceil(cnt / 128) tiles (prefix scanned per block from meta's cnt); a tile of <= 64 rows runs
 * 32-row warp tiles (2 m16) on the same ring. Items are M-major and an expert's tiles adjacent, so
 * its weights stay in L2 across its row tiles. Ring: 3 stages of 128 k; rows of 64 B XOR-swizzled
 * (16 B chunk ^ (row >> 1 & 3): conflict-free ldmatrix), then 8 B of e4m3 scales per row (A rows
 * then B rows), read once per stage for both k64 halves. Epilogues: v1's math. */
template <unsigned V>
struct nf4v_ic {
    static constexpr unsigned value = V;
};

__device__ __forceinline__ void nf4v_cp4(void* smem, const void* g) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(g));
}

/* Bulk async store smem -> global (16 B aligned, bytes % 16 == 0), its group commit / read wait,
 * and the generic -> async proxy fence the staged smem writes need before it. */
__device__ __forceinline__ void nf4v_bulk_st(void* g, const void* smem, unsigned bytes) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(g), "r"(s), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void nf4v_bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void nf4v_bulk_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}
__device__ __forceinline__ void nf4v_fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

__device__ __forceinline__ unsigned nf4v_swz(unsigned row, unsigned chunk) {
    return row * 64u + ((chunk ^ ((row >> 1) & 3u)) << 4);
}

__device__ __forceinline__ void nf4v_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                         unsigned sfa, unsigned sfb) {
    asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

#ifndef NF4V_DIAG /* test builds: 1 = no k-loop staging, 2 = no epilogue, 4 = no epilogue stores */
#define NF4V_DIAG 0
#endif
#ifdef NF4V_STAMP /* test builds: warp-0 clock64 split per block [mode][slice][wait, mma, epilogue, total] */
__device__ unsigned long long nf4v_stamp[2][64][4];
#define NF4V_T(...) __VA_ARGS__
#else
#define NF4V_T(...)
#endif

/* Own register allocation: inlined into the v1 entry points it took them to 245 registers and a
 * 1.3 KB interpreter stack, slowing the v1 path below MIN_T (1K TTFT +4%). */
template <int MODE, unsigned BN, unsigned G = 1u>
static __device__ __noinline__ void nf4v_tiled(void* __restrict__ out, const uint8_t* __restrict__ aq,
                                  const int* __restrict__ meta, const uint8_t* __restrict__ W,
                                  const uint8_t* __restrict__ S, const float* __restrict__ gs,
                                  const unsigned* __restrict__ row_token,
                                  const unsigned* __restrict__ row_partidx,
                                  const float* __restrict__ row_gate, const float* __restrict__ s2,
                                  unsigned n2, unsigned arows, unsigned orows, unsigned T,
                                  unsigned N, unsigned K, unsigned E, unsigned dmode,
                                  unsigned slice, unsigned nblk, uint8_t* arena) {
    /* G groups of NTH threads; group warps 2 (m) x WN (n), warp tile 64 x WT; loads RP rows per pass. */
    constexpr unsigned NTH = 256u / G, WN = 4u / G, WT = BN / WN, NJ = WT / 8u, RP = NTH / 4u;
    constexpr unsigned AH = 128u / RP, BR = BN / RP, SJ = (128u + BN + NTH - 1u) / NTH;
    constexpr unsigned STAGE = NF4V_STAGE(BN), NST = G == 1u ? NF4V_NST : 2u;
    constexpr unsigned SRU = (BN / 2u + BN / 16u + 15u) / 16u * 16u;
    constexpr bool UPB = MODE == 1 && (PLOW_NV_W4A4_V2_BULK & 1) && !(NF4V_DIAG & 6);
    static_assert(!UPB || 128u * SRU <= STAGE, "up staging tile exceeds a ring slot");
    static_assert(BN % RP == 0u && WT % 16u == 0u && (G == 1u || G == 2u), "tile shape");
    static_assert(G * NST * STAGE + 4u * (3u * NF4V_EMAX + 16u + 512u * G + 256u) <= NF4V_ARENA_BYTES, "arena");
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned grp = G == 1u ? 0u : tid / NTH, vt = tid - grp * NTH, vw = vt >> 5;
    const unsigned wm = vw / WN, wn = vw % WN, g = lane >> 2, tq = lane & 3u;
    uint8_t* const ring = arena + grp * (NST * STAGE);
    auto bar = [&]() {
        if (G == 1u) __syncthreads();
        else asm volatile("bar.sync %0, %1;\n" ::"r"(1u + grp), "n"(NTH) : "memory");
    };
    const unsigned ntn = N / BN, kb = K / 2u, ks16 = K / 16u, nk = kb / 64u;
    const uint8_t* asf = aq + nf4w_sf_off(arows, K);
    const float alpha_a = *reinterpret_cast<const float*>(aq + nf4w_alpha_off(arows, K));
    float gs2 = 0.f;
    unsigned* pre = reinterpret_cast<unsigned*>(arena + G * NST * STAGE);
    /* Up epilogue: nf4w_block_scale's multiplier per e4m3 scale byte (gs2 is fixed per launch), so
     * the per-block scale is a lookup instead of a branch and an IEEE division. */
    float* multab = reinterpret_cast<float*>(pre + 3u * NF4V_EMAX + 16u + 512u * G);
    if (MODE == 1) {
        const float m2 = nf4w_max_scale(s2, n2, reinterpret_cast<float*>(arena));
        gs2 = 1.0f / m2;
        if (slice == 0u && tid == 0u) *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(out) + nf4w_alpha_off(orows, N)) = m2;
        __nv_fp8_e4m3 sf;
        sf.__x = (__nv_fp8_storage_t)tid;
        const float v = (float)sf;
        multab[tid] = v != 0.f ? 1.0f / (v * (1.0f / gs2)) : 0.f;
        __syncthreads();
    }
    unsigned* roff = pre + NF4V_EMAX + 16u;
    unsigned* rcnt = roff + NF4V_EMAX;
    unsigned* rid = rcnt + NF4V_EMAX + 256u * grp; /* [G][2][128] row_token (up) / row_partidx (down) */
    float* rgt = reinterpret_cast<float*>(rcnt + NF4V_EMAX + 256u * (G + grp)); /* [G][2][128] row_gate (down) */
    unsigned ntiles;
    if (meta) {
        unsigned c[4], s = 0u;
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            const unsigned e = 4u * tid + j;
            const unsigned n = e < E ? (unsigned)meta[E + e] : 0u;
            if (e < E) {
                roff[e] = (unsigned)meta[e];
                rcnt[e] = n;
            }
            c[j] = (n + 127u) / 128u;
            s += c[j];
        }
        unsigned x = s;
#pragma unroll
        for (unsigned o = 1; o < 32u; o <<= 1) {
            const unsigned y = __shfl_up_sync(0xFFFFFFFFu, x, o);
            if (lane >= o) x += y;
        }
        unsigned* wsum = pre + NF4V_EMAX + 1u;
        if (lane == 31u) wsum[warp] = x;
        __syncthreads();
        unsigned base = 0u, total = 0u;
#pragma unroll
        for (unsigned w = 0; w < 8u; w++) {
            const unsigned v = wsum[w];
            base += w < warp ? v : 0u;
            total += v;
        }
        unsigned ex = base + x - s;
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            const unsigned e = 4u * tid + j;
            if (e < E) pre[e] = ex;
            ex += c[j];
        }
        if (tid == 0u) pre[E] = total;
        __syncthreads();
        ntiles = total;
    } else {
        ntiles = (T + 127u) / 128u;
    }
    const unsigned items = ntiles * ntn, istep = nblk * G;
    const unsigned ar = vt >> 2, ac = vt & 3u;
    struct Ctx {
        unsigned e, r0, rc, n0, abytes[AH], sbytes[SJ];
        float alpha;
        const uint8_t* asrc[AH];
        const uint8_t* bsrc;
        const uint8_t* ssrc[SJ];
    };
    /* item -> expert, local row tile, n-tile. */
    auto locate = [&](unsigned item, unsigned& e, unsigned& lt, unsigned& nt) {
        const unsigned mt = item / ntn;
        nt = item - mt * ntn;
        e = 0u;
        lt = mt;
        if (meta) {
            unsigned lo = 0u, hi = E;
            while (hi - lo > 1u) {
                const unsigned mid = (lo + hi) >> 1;
                if (pre[mid] <= mt) lo = mid;
                else hi = mid;
            }
            e = lo;
            lt = mt - pre[lo];
            if ((PLOW_NV_W4A4_V2_NMAJOR >> (MODE - 1)) & 1) {
                /* N-major inside the expert: concurrent items share one weight tile. */
                const unsigned te = pre[lo + 1u] - pre[lo], j = item - pre[lo] * ntn;
                nt = j / te;
                lt = j - nt * te;
            }
        }
    };
    /* cp.async of an item's row ids (up: row_token, read by its setup) or epilogue rows (down:
     * row_partidx, row_gate) into parity buffer b. Threads [0, 128). */
    auto rowfetch = [&](unsigned item, unsigned b) {
        if (!meta || vt >= 128u || item >= items) return;
        unsigned e, lt, nt;
        locate(item, e, lt, nt);
        const unsigned rc = min(128u, rcnt[e] - lt * 128u);
        const unsigned r = roff[e] + lt * 128u + (vt < rc ? vt : 0u);
        if (MODE == 1) {
            nf4v_cp4(rid + 128u * b + vt, row_token + r);
        } else {
            nf4v_cp4(rid + 128u * b + vt, row_partidx + r);
            nf4v_cp4(rgt + 128u * b + vt, row_gate + r);
        }
    };
    auto setup = [&](unsigned item, Ctx& c, unsigned b) {
        unsigned lt, nt;
        locate(item, c.e, lt, nt);
        c.n0 = nt * BN;
        c.alpha = alpha_a * gs[c.e];
        if (meta) {
            c.r0 = roff[c.e] + lt * 128u;
            c.rc = min(128u, rcnt[c.e] - lt * 128u);
        } else {
            c.r0 = lt * 128u;
            c.rc = min(128u, T - c.r0);
        }
#if PLOW_NV_W4A4_V2_PFD
        /* L2 prefetch of the weight tile PFD waves ahead: the next experts' weights stream from
         * DRAM under this wave's MMAs instead of all blocks missing on them together. */
        if (vt == 0u && item + PLOW_NV_W4A4_V2_PFD * istep < items) {
            unsigned pe, plt, pnt;
            locate(item + PLOW_NV_W4A4_V2_PFD * istep, pe, plt, pnt);
            const size_t row0 = (size_t)pe * N + pnt * BN;
            nf4_pf_l2(W + row0 * kb, BN * kb);
            nf4_pf_l2(S + row0 * ks16, BN * ks16);
        }
#endif
        if (T < PLOW_NV_W4A4_V2_PF_MAXT && vt == 0u) {
            const size_t row0 = (size_t)c.e * N + c.n0;
            const uint8_t* w = W + row0 * kb;
            for (unsigned o = 0; o < BN * kb; o += 65536u) nf4_pf_l2(w + o, min(65536u, BN * kb - o));
            nf4_pf_l2(S + row0 * ks16, BN * ks16);
        }
#pragma unroll
        for (unsigned h = 0; h < AH; h++) {
            const unsigned rr = ar + RP * h;
            const bool ok = rr < c.rc;
            unsigned arow = c.r0 + (ok ? rr : 0u);
            if (MODE == 1 && meta) arow = rid[128u * b + (ok ? rr : 0u)];
            c.asrc[h] = aq + (size_t)arow * kb + ac * 16u;
            c.abytes[h] = ok ? 16u : 0u;
        }
        c.bsrc = W + ((size_t)c.e * N + c.n0 + ar) * kb + ac * 16u;
#pragma unroll
        for (unsigned j = 0; j < SJ; j++) {
            const unsigned i = vt + NTH * j;
            c.ssrc[j] = nullptr;
            c.sbytes[j] = 8u;
            if (i < 128u) {
                if (c.rc > 64u || i < 64u) {
                    const bool ok = i < c.rc;
                    unsigned sr = c.r0 + (ok ? i : 0u);
                    if (MODE == 1 && meta) sr = rid[128u * b + (ok ? i : 0u)];
                    c.ssrc[j] = asf + (size_t)sr * ks16;
                    c.sbytes[j] = ok ? 8u : 0u;
                }
            } else if (i < 128u + BN) {
                c.ssrc[j] = S + ((size_t)c.e * N + c.n0 + i - 128u) * ks16;
            }
        }
    };
    auto stage = [&](const Ctx& c, unsigned ks, unsigned slot) {
        uint8_t* base = ring + slot * STAGE;
        uint8_t* Bs = base + 128u * 64u;
        uint8_t* Ss = base + (128u + BN) * 64u;
        const unsigned ko = ks * 64u;
#pragma unroll
        for (unsigned h = 0; h < AH; h++)
            if (RP * h < 64u || c.rc > 64u) nf4_cp16(base + nf4v_swz(ar + RP * h, ac), c.asrc[h] + ko, c.abytes[h]);
#pragma unroll
        for (unsigned h = 0; h < BR; h++)
            nf4_cp16(Bs + nf4v_swz(ar + RP * h, ac), c.bsrc + (size_t)h * RP * kb + ko, 16u);
#pragma unroll
        for (unsigned j = 0; j < SJ; j++)
            if (c.ssrc[j]) nf4w_cp8(Ss + (vt + NTH * j) * 8u, c.ssrc[j] + ks * 8u, c.sbytes[j]);
    };
    auto prologue = [&](const Ctx& c) {
#pragma unroll
        for (unsigned s = 0; s < NST - 1u; s++) {
            if (s < nk) stage(c, s, s);
            nf4_cp_commit();
        }
    };
    NF4V_T(unsigned long long st_w = 0, st_k = 0, st_e = 0; const unsigned long long st_0 = clock64();)
    Ctx cur;
    unsigned item = slice * G + grp, par = 0u;
    if (item < items) {
        if (MODE == 1 && meta) {
            rowfetch(item, 0u);
            nf4_cp_commit();
            nf4w_cp_wait<0>();
            bar();
        }
        setup(item, cur, 0u);
        rowfetch(MODE == 1 ? item + istep : item, MODE == 1 ? 1u : 0u);
        prologue(cur);
    }
    while (item < items) {
        auto body = [&](auto mic) {
            constexpr unsigned MI = decltype(mic)::value;
            const unsigned wr = wm * MI * 16u;
            float acc[MI][NJ][4];
#pragma unroll
            for (unsigned i = 0; i < MI; i++)
#pragma unroll
                for (unsigned j = 0; j < NJ; j++)
#pragma unroll
                    for (unsigned q = 0; q < 4u; q++) acc[i][j][q] = 0.f;
            for (unsigned ks = 0; ks < nk; ks++) {
                NF4V_T(const unsigned long long c0 = clock64();)
                nf4w_cp_wait<NST - 2u>();
                if (UPB && ks == 0u && vt < 128u) nf4v_bulk_wait_read();
                bar();
                NF4V_T(const unsigned long long c1 = clock64(); st_w += c1 - c0;)
                if (!(NF4V_DIAG & 1) && ks + NST - 1u < nk) stage(cur, ks + NST - 1u, (ks + NST - 1u) % NST);
                nf4_cp_commit();
                const uint8_t* As = ring + (ks % NST) * STAGE;
                const uint8_t* Bs = As + 128u * 64u;
                const uint8_t* Ss = As + (128u + BN) * 64u;
                uint2 sa[MI], sb[NJ];
#pragma unroll
                for (unsigned mi = 0; mi < MI; mi++)
                    sa[mi] = *reinterpret_cast<const uint2*>(Ss + (wr + mi * 16u + g + 8u * (lane & 1u)) * 8u);
#pragma unroll
                for (unsigned nj = 0; nj < NJ; nj++)
                    sb[nj] = *reinterpret_cast<const uint2*>(Ss + (128u + wn * WT + nj * 8u + g) * 8u);
#pragma unroll
                for (unsigned sub = 0; sub < 2u; sub++) {
                    unsigned af[MI][4], bf[NJ / 2u][4];
#pragma unroll
                    for (unsigned mi = 0; mi < MI; mi++)
                        nf4_ldsm_x4(af[mi], As + nf4v_swz(wr + mi * 16u + (lane & 15u), 2u * sub + (lane >> 4)));
#pragma unroll
                    for (unsigned p = 0; p < NJ / 2u; p++)
                        nf4_ldsm_x4(bf[p], Bs + nf4v_swz(wn * WT + p * 16u + (lane & 7u) + ((lane >> 4) << 3),
                                                         2u * sub + ((lane >> 3) & 1u)));
#pragma unroll
                    for (unsigned mi = 0; mi < MI; mi++)
#pragma unroll
                        for (unsigned nj = 0; nj < NJ; nj++)
                            nf4v_mma(acc[mi][nj], af[mi], bf[nj >> 1][(nj & 1u) * 2u], bf[nj >> 1][(nj & 1u) * 2u + 1u],
                                     sub ? sa[mi].y : sa[mi].x, sub ? sb[nj].y : sb[nj].x);
                }
                NF4V_T(st_k += clock64() - c1;)
            }
            NF4V_T(const unsigned long long c2 = clock64();)
            nf4w_cp_wait<0>();
            bar();
            const unsigned r0 = cur.r0, rc = cur.rc, n0 = cur.n0, pb = par;
            const float alpha = cur.alpha;
            const unsigned item_next = item + istep;
            par ^= 1u;
            if (item_next < items) {
                setup(item_next, cur, par);
                rowfetch(MODE == 1 ? item_next + istep : item_next, MODE == 1 ? pb : par);
                prologue(cur);
            }
            if ((NF4V_DIAG & 2) && rc != 999u) {
            } else if (MODE == 1) {
                uint8_t* oq = reinterpret_cast<uint8_t*>(out);
                uint8_t* osf = oq + nf4w_sf_off(orows, N);
                const unsigned ob = N / 2u, os = N / 16u;
#pragma unroll
                for (unsigned mi = 0; mi < MI; mi++)
#pragma unroll
                    for (unsigned h = 0; h < 2u; h++) {
                        const unsigned lr = wr + mi * 16u + g + 8u * h;
                        const size_t r = r0 + lr;
#pragma unroll
                        for (unsigned b = 0; b < NJ / 2u; b++) {
                            float v[4];
#pragma unroll
                            for (unsigned u = 0; u < 2u; u++)
#pragma unroll
                                for (unsigned c = 0; c < 2u; c++) {
                                    const float y = __bfloat162float(__float2bfloat16(acc[mi][2u * b + u][2u * h + c] * alpha));
                                    const float rl = fmaxf(y, 0.f);
                                    v[2u * u + c] = rl * rl;
                                }
                            float m = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
                            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
                            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
                            const __nv_fp8_e4m3 sf(fminf(gs2 * (m * (1.0f / 6.0f)), 448.0f));
                            const unsigned byte = (unsigned)sf.__x;
                            const float mul = multab[byte];
                            const unsigned pk = nf4w_e2m1x8(v[0] * mul, v[1] * mul, v[2] * mul, v[3] * mul, 0.f,
                                                            0.f, 0.f, 0.f);
                            if (UPB) {
                                /* staging row lr: BN / 2 q bytes then BN / 16 scale bytes, stride SRU;
                                 * each lane's two bytes go straight to their columns */
                                uint8_t* st = ring + (NST - 1u) * STAGE + lr * SRU;
                                const unsigned lc = wn * WT + b * 16u;
                                st[lc / 2u + tq] = (uint8_t)pk;
                                st[lc / 2u + 4u + tq] = (uint8_t)(pk >> 8);
                                if (tq == 0u) st[BN / 2u + lc / 16u] = (uint8_t)byte;
                                continue;
                            }
                            unsigned lo = (pk & 0xFFu) << (8u * tq);
                            unsigned hi = ((pk >> 8) & 0xFFu) << (8u * tq);
                            lo |= __shfl_xor_sync(0xFFFFFFFFu, lo, 1);
                            hi |= __shfl_xor_sync(0xFFFFFFFFu, hi, 1);
                            lo |= __shfl_xor_sync(0xFFFFFFFFu, lo, 2);
                            hi |= __shfl_xor_sync(0xFFFFFFFFu, hi, 2);
                            if (tq == 0u && lr < rc && !(NF4V_DIAG & 4)) {
                                const unsigned col = n0 + wn * WT + b * 16u;
                                *reinterpret_cast<uint2*>(oq + r * ob + col / 2u) = make_uint2(lo, hi);
                                osf[r * os + col / 16u] = (uint8_t)byte;
                            }
                        }
                    }
                if (UPB) {
                    nf4v_fence_async();
                    bar();
                    if (vt < rc) {
                        const uint8_t* st = ring + (NST - 1u) * STAGE + vt * SRU;
                        const size_t r = r0 + vt;
                        nf4v_bulk_st(oq + r * ob + n0 / 2u, st, BN / 2u);
                        nf4v_bulk_commit();
                        if (BN % 64u == 0u) {
#pragma unroll
                            for (unsigned w = 0; w < BN / 64u; w++)
                                *reinterpret_cast<unsigned*>(osf + r * os + n0 / 16u + 4u * w) =
                                    *reinterpret_cast<const unsigned*>(st + BN / 2u + 4u * w);
                        } else {
#pragma unroll
                            for (unsigned w = 0; w < BN / 32u; w++)
                                *reinterpret_cast<unsigned short*>(osf + r * os + n0 / 16u + 2u * w) =
                                    *reinterpret_cast<const unsigned short*>(st + BN / 2u + 2u * w);
                        }
                    }
                }
            } else {
#pragma unroll
                for (unsigned mi = 0; mi < MI; mi++)
#pragma unroll
                    for (unsigned h = 0; h < 2u; h++) {
                        const unsigned lr = wr + mi * 16u + g + 8u * h;
                        if (lr >= rc || (NF4V_DIAG & 4)) continue;
                        const unsigned r = r0 + lr;
                        size_t orow = r;
                        float rg = 1.0f;
                        if (dmode == 0u || dmode == 2u) orow = rid[128u * pb + lr];
                        if (dmode == 0u) rg = rgt[128u * pb + lr];
#pragma unroll
                        for (unsigned nj = 0; nj < NJ; nj++) {
                            const unsigned col = n0 + wn * WT + nj * 8u + 2u * tq;
                            const float y0 = acc[mi][nj][2u * h] * alpha, y1 = acc[mi][nj][2u * h + 1u] * alpha;
                            if (dmode == 0u)
                                *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + orow * N + col) =
                                    make_float2(y0 * rg, y1 * rg);
                            else if (dmode == 3u)
                                *reinterpret_cast<unsigned*>(reinterpret_cast<__nv_bfloat16*>(out) + orow * N + col) =
                                    nf4w_radd(row_partidx, orow * N + col, y0, y1);
                            else
                                *reinterpret_cast<unsigned*>(reinterpret_cast<__nv_bfloat16*>(out) + orow * N + col) =
                                    nf4_f2bf2(y0, y1);
                        }
                    }
            }
            item = item_next;
            NF4V_T(st_e += clock64() - c2;)
        };
        if (cur.rc > 64u) body(nf4v_ic<4u>{});
        else body(nf4v_ic<2u>{});
    }
    /* The next op reads what the bulk stores wrote (after the interpreter's grid barrier). */
    if (UPB && vt < 128u) {
        asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
    }
#ifdef NF4V_STAMP
    if (tid == 0u && slice < 64u) {
        atomicAdd(&nf4v_stamp[MODE - 1][slice][0], st_w);
        atomicAdd(&nf4v_stamp[MODE - 1][slice][1], st_k);
        atomicAdd(&nf4v_stamp[MODE - 1][slice][2], st_e);
        atomicAdd(&nf4v_stamp[MODE - 1][slice][3], clock64() - st_0);
    }
#endif
}

/* v2 tile width for an N-column output: 192 or 256 when N divides, else 0 (v1). */
__device__ __forceinline__ unsigned nf4v_bn(unsigned N) {
#if PLOW_NV_W4A4_V2
    if (N % 256u == 0u) return 256u;
    if (N % 192u == 0u) return 192u;
#endif
    (void)N;
    return 0u;
}

/* Tile rows: 128 (the 32 x 64 warp tile halves the staged bytes per MMA) or 64. Dense (shared
 * expert): 128 at T >= 512. Routed: 64 — a 128-row item over an expert's odd last 64-row tile
 * computes 64 dead rows, and on real-text routing (layer 1, all 512 experts touched from 1024
 * rows) the 128-row routed tile measured +1.9 ms at 4096 rows and within noise at 128/1024/8192.
 * PLOW_NV_W4A4_BM pins both for A/B. */
__device__ __forceinline__ bool nf4w_wide(const int* meta, unsigned rows) {
#if defined(PLOW_NV_W4A4_BM)
    return PLOW_NV_W4A4_BM == 128;
#else
    return !meta && rows >= 512u;
#endif
}

/* MOE_COMBINE_PF with bf16 partials (i5 = 1): out[t][h] = bf16(sum_j w_tj * part[t k + j][h]) in
 * f32, j ascending, w from the routing table (t4, {u32 expert, f32 w}); part = bf16(alpha acc) of
 * the down projection (dmode 2). The finalize order of the peer's fused MoE (TRT-LLM / vLLM):
 * bf16 expert outputs, router weights applied in the f32 sum. 8 columns per thread. */
static __device__ __noinline__ void d_moe_combine_nvfp4_bf16(__nv_bfloat16* __restrict__ out,
                                                             const __nv_bfloat16* __restrict__ part,
                                                             const unsigned char* __restrict__ table,
                                                             unsigned H, unsigned k, unsigned T,
                                                             unsigned slice, unsigned nblk) {
    if (H % 8u || blockDim.x != 256u) { __trap(); return; }
    const unsigned hv = H / 8u;
    const size_t total = (size_t)T * hv;
    for (size_t i = (size_t)slice * 256u + threadIdx.x; i < total; i += (size_t)nblk * 256u) {
        const size_t t = i / hv;
        const unsigned h = (unsigned)(i - t * hv) * 8u;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (unsigned j = 0; j < k; j++) {
            const size_t s = t * k + j;
            const float w = *reinterpret_cast<const float*>(table + s * 8u + 4u);
            const uint4 v = *reinterpret_cast<const uint4*>(part + s * H + h);
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
            for (int c = 0; c < 8; c++) acc[c] = fmaf(w, __bfloat162float(p[c]), acc[c]);
        }
        uint4 o;
        unsigned* op = reinterpret_cast<unsigned*>(&o);
#pragma unroll
        for (int c = 0; c < 4; c++) op[c] = nf4_f2bf2(acc[2 * c], acc[2 * c + 1]);
        *reinterpret_cast<uint4*>(out + t * H + h) = o;
    }
}

/* MOE_UP_NVFP4_PF, W4A4 (i5 = 1): t0 = q tensor of relu^2 for the down projection (T*k sorted
 * rows, or T dense rows), t1 = q tensor of x (T rows), t7 = the down projection's input_scale
 * vector (n2 = i6 entries). */
static __device__ __noinline__ void d_moe_up_nvfp4_pf_w4a4(uint8_t* fq, const uint8_t* xq, const int* meta,
                                                           const uint8_t* W, const uint8_t* S, const float* gs,
                                                           const unsigned* row_token, const float* s2,
                                                           unsigned T, unsigned k, unsigned I, unsigned K,
                                                           unsigned E, unsigned n2, unsigned slice,
                                                           unsigned nblk, float* arena) {
    if (K % 128u || I % NF4W_BN || (!meta && E != 1u) || (meta && !row_token) || !s2 || n2 == 0u ||
        blockDim.x != 256u) {
        __trap();
        return;
    }
    uint8_t* ar = reinterpret_cast<uint8_t*>(arena);
    const unsigned orows = meta ? T * k : T;
    const unsigned bn = T >= PLOW_NV_W4A4_V2_MIN_T ? nf4v_bn(I) : 0u;
    if (bn && (!meta || E <= NF4V_EMAX)) {
#if PLOW_NV_W4A4_V2_PP
        if (meta && bn == 192u && T < PLOW_NV_W4A4_V2_PP_MAXT) {
            nf4v_tiled<1, 96u, 2u>(fq, xq, meta, W, S, gs, row_token, nullptr, nullptr, s2, n2, T, orows, T, I, K, E,
                                   0u, slice, nblk, ar);
            return;
        }
#endif
        if (bn == 256u)
            nf4v_tiled<1, 256u>(fq, xq, meta, W, S, gs, row_token, nullptr, nullptr, s2, n2, T, orows, T, I, K, E,
                                0u, slice, nblk, ar);
        else
            nf4v_tiled<1, 192u>(fq, xq, meta, W, S, gs, row_token, nullptr, nullptr, s2, n2, T, orows, T, I, K, E,
                                0u, slice, nblk, ar);
        return;
    }
    if (nf4w_wide(meta, orows))
        nf4w_tiled<1, 128>(fq, xq, meta, W, S, gs, row_token, nullptr, nullptr, s2, n2, T, orows, T, I, K, E,
                           0u, slice, nblk, ar);
    else
        nf4w_tiled<1, 64>(fq, xq, meta, W, S, gs, row_token, nullptr, nullptr, s2, n2, T, orows, T, I, K, E,
                          0u, slice, nblk, ar);
}

/* MOE_DOWN_NVFP4_PF, W4A4 (i6 = 1): t1 = q tensor of the up projection's output (T*k or T rows). */
static __device__ __noinline__ void d_moe_down_nvfp4_pf_w4a4(void* out, const uint8_t* fq, const int* meta,
                                                             const uint8_t* W, const uint8_t* S, const float* gs,
                                                             const unsigned* row_partidx, const float* row_gate,
                                                             unsigned T, unsigned k, unsigned H, unsigned I,
                                                             unsigned E, unsigned mode, unsigned slice,
                                                             unsigned nblk, float* arena) {
    /* mode: routed 0 (f32 gated part) / 2 (bf16 part); dense 1 (bf16) / 3 (bf16 + t6 residual). */
    if (I % 128u || H % NF4W_BN || (!meta && E != 1u) || (mode == 1u || mode == 3u) != (meta == nullptr) ||
        mode > 3u || (meta && (!row_partidx || !row_gate)) || (mode == 3u && !row_partidx) || blockDim.x != 256u) {
        __trap();
        return;
    }
    uint8_t* ar = reinterpret_cast<uint8_t*>(arena);
    const unsigned arows = meta ? T * k : T;
    const unsigned bn = T >= PLOW_NV_W4A4_V2_MIN_T ? nf4v_bn(H) : 0u;
    if (bn && (!meta || E <= NF4V_EMAX)) {
        if (bn == 256u)
            nf4v_tiled<2, 256u>(out, fq, meta, W, S, gs, nullptr, row_partidx, row_gate, nullptr, 0u, arows, 0u, T,
                                H, I, E, mode, slice, nblk, ar);
        else
            nf4v_tiled<2, 192u>(out, fq, meta, W, S, gs, nullptr, row_partidx, row_gate, nullptr, 0u, arows, 0u, T,
                                H, I, E, mode, slice, nblk, ar);
        return;
    }
    if (nf4w_wide(meta, arows))
        nf4w_tiled<2, 128>(out, fq, meta, W, S, gs, nullptr, row_partidx, row_gate, nullptr, 0u, arows, 0u, T,
                           H, I, E, mode, slice, nblk, ar);
    else
        nf4w_tiled<2, 64>(out, fq, meta, W, S, gs, nullptr, row_partidx, row_gate, nullptr, 0u, arows, 0u, T,
                          H, I, E, mode, slice, nblk, ar);
}

