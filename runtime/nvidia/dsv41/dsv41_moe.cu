// DeepSeek-V4.1 sm_90a: MoE router, expert alignment, grouped FP4 expert GEMM, SwiGLU, combine.
#include "dsv41_common.cuh"

// Router (model.py Gate.forward, sqrtsoftplus + noaux_tc): s = sqrt(softplus(logit)); experts = top-k of
// s + bias; weights = s[experts], normalized by (sum + 1e-20) when norm, times route_scale.
// One warp per token; ties pick the lower expert id.
DSV_EXTERN void dsv_moe_route(int* __restrict__ idx, float* __restrict__ wt, const float* __restrict__ logits,
                              const float* __restrict__ bias, int T, int E, int topk, int norm, float route_scale) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (t >= T) return;
    const int per = (E + 31) / 32;  // 12 for 384
    float sc[16], key[16];
    for (int i = 0; i < per; i++) {
        const int e = lane + i * 32;
        if (e < E) {
            const float x = logits[(long long)t * E + e];
            const float sp = x > 20.f ? x : log1pf(expf(x));
            sc[i] = sqrtf(sp);
            key[i] = sc[i] + bias[e];
        } else {
            sc[i] = 0.f;
            key[i] = -INFINITY;
        }
    }
    float sel_w[8];
    int sel_e[8];
    for (int r = 0; r < topk; r++) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = 0; i < per; i++) {
            const int e = lane + i * 32;
            if (key[i] > bv || (key[i] == bv && e < bi)) {
                bv = key[i];
                bi = e;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        // owner lane knocks it out and broadcasts its score
        float s_owner = 0.f;
        if ((bi & 31) == lane) {
            key[bi >> 5] = -INFINITY;
            s_owner = sc[bi >> 5];
        }
        s_owner = __shfl_sync(0xffffffffu, s_owner, bi & 31);
        sel_w[r] = s_owner;
        sel_e[r] = bi;
    }
    if (lane == 0) {
        float sum = 0.f;
        for (int r = 0; r < topk; r++) sum += sel_w[r];
        for (int r = 0; r < topk; r++) {
            float w = sel_w[r];
            if (norm && topk > 1) w = w / (sum + 1e-20f);
            idx[t * topk + r] = sel_e[r];
            wt[t * topk + r] = w * route_scale;
        }
    }
}

// Alignment. counts[E] (zeroed by the caller) <- histogram; then offsets/tiles; then fill.
DSV_EXTERN void dsv_moe_count(int* __restrict__ counts, const int* __restrict__ idx, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&counts[idx[i]], 1);
}
// One block of >= E threads: offs[e] = exclusive prefix of counts, and the tile list (expert, row0)
// of BM-row tiles in expert order; meta[0] = number of tiles. fill_ctr[e] zeroed for the fill pass.
// Two block-wide exclusive scans (row counts, tile counts), then each thread writes its own tiles.
DSV_EXTERN void dsv_moe_offsets(int* __restrict__ offs, int* __restrict__ tiles, int* __restrict__ meta,
                                int* __restrict__ fill_ctr, const int* __restrict__ counts, int E, int BM) {
    __shared__ int sc[1024], st[1024];
    const int e = threadIdx.x;
    const int c = e < E ? counts[e] : 0;
    const int nt = (c + BM - 1) / BM;
    sc[e] = c;
    st[e] = nt;
    __syncthreads();
    for (int o = 1; o < (int)blockDim.x; o <<= 1) {
        const int a = e >= o ? sc[e - o] : 0, b = e >= o ? st[e - o] : 0;
        __syncthreads();
        sc[e] += a;
        st[e] += b;
        __syncthreads();
    }
    const int row0 = sc[e] - c, tile0 = st[e] - nt;  // exclusive
    if (e < E) {
        offs[e] = row0;
        fill_ctr[e] = 0;
        for (int k = 0; k < nt; k++) {
            tiles[(tile0 + k) * 2] = e;
            tiles[(tile0 + k) * 2 + 1] = row0 + k * BM;
        }
    }
    if (e == E - 1) {
        offs[E] = sc[e];
        meta[0] = st[e];
    }
}

// rows[pos] = token, rowpos[t*topk+s] = pos, row_w[pos] = routing weight.
DSV_EXTERN void dsv_moe_fill(int* __restrict__ rows, int* __restrict__ rowpos, float* __restrict__ row_w,
                             int* __restrict__ fill_ctr, const int* __restrict__ offs, const int* __restrict__ idx,
                             const float* __restrict__ wt, int n, int topk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = idx[i];
    const int p = offs[e] + atomicAdd(&fill_ctr[e], 1);
    rows[p] = i / topk;
    rowpos[i] = p;
    row_w[p] = wt[i];
}

__device__ __forceinline__ void mma_e4m3_m(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// e2m1 -> e4m3 (exact: every e2m1 value is an e4m3 value), four at a time. `x16` holds four nibbles in
// element order (element 2i is byte i's low nibble); a byte-permute looks up the 3-bit magnitudes in
// a register-resident table {00,30,38,3C,40,44,48,4C} and the sign bits move from bit 3 of each
// nibble to bit 7 of each byte.
// Eight packed e2m1 (element 2i in the low nibble of byte i) as two e4m3x4 words, by shifts alone:
// an e2m1 nibble s.ee.m moved to s.0000.ee.m00 is the e4m3 encoding of its value * 2^-6 (the
// subnormal 0.5 included: both formats keep their subnormal step at the same place), so the MMA
// sees w * 2^-6 and the weight scale carries the 2^6. lo = the even elements, hi = the odd ones.
#define FP4_E4M3_UNSCALE 64.f
__device__ __forceinline__ uint32_t fp4x8_lo(uint32_t w) { return ((w << 4) & 0x80808080u) | ((w << 2) & 0x1C1C1C1Cu); }
__device__ __forceinline__ uint32_t fp4x8_hi(uint32_t w) { return (w & 0x80808080u) | ((w >> 2) & 0x1C1C1C1Cu); }

__device__ __forceinline__ uint32_t e2m1x4_to_e4m3x4(uint32_t x16) {
    const uint32_t mag = __byte_perm(0x3C383000u, 0x4C484440u, x16 & 0x7777u);
    const uint32_t s = x16 & 0x8888u;
    return mag | ((s & 0x8u) << 4) | ((s & 0x80u) << 8) | ((s & 0x800u) << 12) | ((s & 0x8000u) << 16);
}

// Grouped expert GEMM (kernel.py fp4_gemm at act block 32): for tile i = (expert e, row0), rows
// row0 .. min(row0+BM, offs[e+1]) of the expert-sorted list: C[p][n] = sum_kb (A8[tok(p)] . W4[e][n]) *
// sa[tok(p)][kb] * sw[e][n][kb]. A is indexed by token (gather through `rows`) unless a_by_row, in which
// case A row p itself (the down projection reads the SwiGLU output, already in sorted order).
// W fp4 packed [N][K/2] per expert (element 2i in the low nibble), sw ue8m0 [N][K/32]. K % 128 == 0.
//
// Tile BM x 128 (BM = 16, 32 or 64: the host sizes it to the rows an expert holds, so prefill tiles
// are not mostly padding), 4 warps side by side along N, warp tile BM x 32: each warp decodes its 32
// weight columns once per K block and reuses them over all BM/16 row fragments. A three-stage
// cp.async ring stages 128 K per stage: the gathered e4m3 A rows and the RAW packed weights; the B
// fragments are decoded from the raw nibbles in registers (fp4x8_lo/hi: a thread's 8 weights of a K
// block, even elements into b0 and odd into b1, and its activations permuted to match). Both scale
// grids for the tile are staged up front with vector copies (the tile's weight scales are one
// contiguous block). Dynamic shared memory: G_SMEM(K, BM) bytes.
#define G_BN 128
#define G_ST 3
#define G_KS 128  // K per stage
#define G_ALD 160 // A row: 128 B + pad (40 words: the 8-byte fragment loads of 4 rows x 4 lanes hit distinct banks)
#define G_WLD 80  // W row: 64 B raw (128 fp4) + pad (20 words)
#define G_SMEM(K, BM) (G_ST * (BM) * G_ALD + G_ST * G_BN * G_WLD + (G_BN + (BM)) * ((K) / 32) + (BM) * 4 + 16)

__device__ __forceinline__ void cp_async16(void* dst, const void* src, int bytes) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}

template <int BM>
__device__ __forceinline__ void moe_gemm_fp4(bf16* __restrict__ C, const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                                             const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw, const int* __restrict__ tiles,
                                             const int* __restrict__ meta, const int* __restrict__ offs, const int* __restrict__ rows,
                                             int a_by_row, int N, int K, long long w_estride, long long sw_estride) {
    constexpr int MI = BM / 16;  // row fragments per warp
    extern __shared__ __align__(16) uint8_t g_smem[];
    const int tile = blockIdx.y;
    if (tile >= meta[0]) return;
    const int KB = K >> 5, KS = K / G_KS;
    uint8_t* As = g_smem;                                  // [ST][BM][160]
    uint8_t* Ws = As + G_ST * BM * G_ALD;                  // [ST][128][80]
    uint8_t* SWs = Ws + G_ST * G_BN * G_WLD;               // [128][KB] weight scales
    uint8_t* SAs = SWs + G_BN * KB;                        // [BM][KB] activation scales
    int* Arow = (int*)(SAs + BM * KB + ((16 - ((G_BN + BM) * KB) % 16) % 16));
    const int kb_per_stage = G_KS / 32;
    const int e = tiles[tile * 2], row0 = tiles[tile * 2 + 1];
    const int rend = offs[e + 1];
    const int n0 = blockIdx.x * G_BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const uint8_t* We = W + (long long)e * w_estride;
    const uint8_t* Se = sw + (long long)e * sw_estride;
    if (tid < BM) {
        const int p = row0 + tid;
        Arow[tid] = p < rend ? (a_by_row ? p : rows[p]) : -1;
    }
    __syncthreads();
    // Scales for the whole tile. The weight scales of rows n0..n0+127 are one contiguous
    // 128*KB-byte block (N is a multiple of 128 for every V4.1 expert GEMM); the activation scales
    // are gathered per row, KB bytes each (KB % 8 == 0), with 8-byte loads.
    {
        const int chunks = G_BN * KB / 16;
        const uint8_t* src = Se + (long long)n0 * KB;
        for (int c = tid; c < chunks; c += 128) cp_async16(&SWs[c * 16], src + c * 16, n0 + (c * 16) / KB < N ? 16 : 0);
        const int per_row = KB / 8;
        for (int c = tid; c < BM * per_row; c += 128) {
            const int r = c / per_row, off = (c % per_row) * 8;
            const int ar = Arow[r];
            *(uint2*)&SAs[r * KB + off] = ar >= 0 ? *(const uint2*)(sa + (long long)ar * KB + off) : make_uint2(0, 0);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    auto load = [&](int slot, int ks) {
#pragma unroll
        for (int i = 0; i < BM / 16; i++) {  // A: BM rows x 8 chunks of 16 B
            const int c = tid + i * 128, r = c >> 3, off = (c & 7) * 16;
            const int ar = Arow[r];
            cp_async16(&As[(slot * BM + r) * G_ALD + off], A + (long long)max(ar, 0) * K + ks * G_KS + off, ar >= 0 ? 16 : 0);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {  // W: 128 rows x 4 chunks of 16 B (raw fp4)
            const int c = tid + i * 128, r = c >> 2, off = (c & 3) * 16;
            const int gn = n0 + r;
            cp_async16(&Ws[(slot * G_BN + r) * G_WLD + off], We + (long long)min(gn, N - 1) * (K / 2) + ks * (G_KS / 2) + off, gn < N ? 16 : 0);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[MI][4][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;
    load(0, 0);
    if (KS > 1) load(1, 1); else asm volatile("cp.async.commit_group;\n" ::);
    for (int ks = 0; ks < KS; ks++) {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        // the slot computed at ks-1 is free now: refill it with stage ks+2
        if (ks + 2 < KS) load((ks + 2) % G_ST, ks + 2); else asm volatile("cp.async.commit_group;\n" ::);
        const int slot = ks % G_ST;
#pragma unroll
        for (int kk = 0; kk < G_KS / 32; kk++) {
            const int kb = ks * kb_per_stage + kk;
            uint32_t bfr[4][2];
            float s0[4], s1[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int cn = warp * 32 + j * 8;
                const uint32_t w8 = *(const uint32_t*)&Ws[(slot * G_BN + cn + g) * G_WLD + kk * 16 + t4 * 4];
                bfr[j][0] = fp4x8_lo(w8);
                bfr[j][1] = fp4x8_hi(w8);
                s0[j] = e8m0_to_f(SWs[(cn + t4 * 2) * KB + kb]) * FP4_E4M3_UNSCALE;
                s1[j] = e8m0_to_f(SWs[(cn + t4 * 2 + 1) * KB + kb]) * FP4_E4M3_UNSCALE;
            }
#pragma unroll
            for (int i = 0; i < MI; i++) {
                uint32_t af[4];
                const uint8_t* base = &As[(slot * BM + i * 16 + g) * G_ALD + kk * 32 + t4 * 8];
                const uint2 a0 = *(const uint2*)(base), a1 = *(const uint2*)(base + 8 * G_ALD);
                af[0] = __byte_perm(a0.x, a0.y, 0x6420);  // even K of the 8, as the weights' low nibbles
                af[2] = __byte_perm(a0.x, a0.y, 0x7531);  // odd K
                af[1] = __byte_perm(a1.x, a1.y, 0x6420);
                af[3] = __byte_perm(a1.x, a1.y, 0x7531);
                const float sa0 = e8m0_to_f(SAs[(i * 16 + g) * KB + kb]);
                const float sa1 = e8m0_to_f(SAs[(i * 16 + g + 8) * KB + kb]);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    float tt[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_e4m3_m(tt, af, bfr[j]);
                    acc[i][j][0] += tt[0] * sa0 * s0[j];
                    acc[i][j][1] += tt[1] * sa0 * s1[j];
                    acc[i][j][2] += tt[2] * sa1 * s0[j];
                    acc[i][j][3] += tt[3] * sa1 * s1[j];
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = i * 16 + g + h * 8;
            const int p = row0 + r;
            if (p >= rend) continue;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = n0 + warp * 32 + j * 8 + t4 * 2;
                if (n >= N) continue;
                __nv_bfloat162 v;
                v.x = f2bf(acc[i][j][h * 2]);
                v.y = f2bf(acc[i][j][h * 2 + 1]);
                *(__nv_bfloat162*)&C[(long long)p * N + n] = v;
            }
        }
}

#define DSV_MOE_GEMM(name, BM)                                                                                          \
    DSV_EXTERN void __launch_bounds__(128)                                                                             \
        name(bf16* __restrict__ C, const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,                      \
             const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw, const int* __restrict__ tiles,             \
             const int* __restrict__ meta, const int* __restrict__ offs, const int* __restrict__ rows, int a_by_row,  \
             int N, int K, long long w_estride, long long sw_estride) {                                                \
        moe_gemm_fp4<BM>(C, A, sa, W, sw, tiles, meta, offs, rows, a_by_row, N, K, w_estride, sw_estride);           \
    }
DSV_MOE_GEMM(dsv_moe_gemm_fp4, 64)
DSV_MOE_GEMM(dsv_moe_gemm_fp4_m32, 32)
DSV_MOE_GEMM(dsv_moe_gemm_fp4_m16, 16)

// Decode form of the grouped fp4 expert GEMM: the same math as dsv_moe_gemm_fp4 for tiles holding a
// handful of rows, as one warp per output row streaming that row's packed weights (bandwidth-bound,
// where the 64-row MMA tile would be almost all padding). Per 32-wide K block: the fp4 x e4m3
// products are exact in fp32 (<= 6 significant bits), summed in fp32, then scaled by sa * sw into
// the accumulator -- the MMA kernel's per-block promotion, in another summation order.
// grid = (ceil(N / 8), tiles), 256 threads.
__device__ __forceinline__ void e4m3x4_to_f32x4(uint32_t v, float* o) {
    const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xffffu), __NV_E4M3);
    const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v >> 16), __NV_E4M3);
    const float2 a = __half22float2(*(const __half2*)&lo), b = __half22float2(*(const __half2*)&hi);
    o[0] = a.x;
    o[1] = a.y;
    o[2] = b.x;
    o[3] = b.y;
}

#define GV_ROWS 4  // output rows per warp: four independent weight streams in flight per lane
DSV_EXTERN void __launch_bounds__(256)
    dsv_moe_gemv_fp4(bf16* __restrict__ C, const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                     const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw, const int* __restrict__ tiles,
                     const int* __restrict__ meta, const int* __restrict__ offs, const int* __restrict__ rows,
                     int a_by_row, int N, int K, long long w_estride, long long sw_estride) {
    const int tile = blockIdx.y;
    if (tile >= meta[0]) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nw = (blockIdx.x * 8 + warp) * GV_ROWS;  // this warp's first output row
    if (nw >= N) return;
    const int e = tiles[tile * 2], row0 = tiles[tile * 2 + 1];
    const int R = min(offs[e + 1], row0 + 64) - row0;
    const int KB = K >> 5;
    const uint8_t* wbase = W + (long long)e * w_estride + (long long)nw * (K / 2);
    const uint8_t* sbase = sw + (long long)e * sw_estride + (long long)nw * KB;
    const int nr = min(GV_ROWS, N - nw);
    for (int tb = 0; tb < R; tb += 8) {
        const int nt = min(8, R - tb);
        int tok[8];
#pragma unroll
        for (int t = 0; t < 8; t++) tok[t] = t < nt ? (a_by_row ? row0 + tb + t : rows[row0 + tb + t]) : -1;
        float acc[GV_ROWS][8];
#pragma unroll
        for (int r = 0; r < GV_ROWS; r++)
#pragma unroll
            for (int t = 0; t < 8; t++) acc[r][t] = 0.f;
        for (int kb = lane; kb < KB; kb += 32) {
            uint4 raw[GV_ROWS];
            float ws[GV_ROWS];
#pragma unroll
            for (int r = 0; r < GV_ROWS; r++) {  // all rows' loads issued before any arithmetic
                raw[r] = r < nr ? *(const uint4*)(wbase + (long long)r * (K / 2) + kb * 16) : make_uint4(0, 0, 0, 0);
                ws[r] = r < nr ? e8m0_to_f(sbase[(long long)r * KB + kb]) : 0.f;
            }
#pragma unroll
            for (int r = 0; r < GV_ROWS; r++) {
                // 32 weights as f32, via the exact fp4 -> e4m3 -> f16 path
                float w[32];
                const uint32_t rw[4] = {raw[r].x, raw[r].y, raw[r].z, raw[r].w};
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    e4m3x4_to_f32x4(e2m1x4_to_e4m3x4(rw[q] & 0xffffu), &w[q * 8]);
                    e4m3x4_to_f32x4(e2m1x4_to_e4m3x4(rw[q] >> 16), &w[q * 8 + 4]);
                }
#pragma unroll
                for (int t = 0; t < 8; t++) {
                    if (tok[t] < 0) continue;
                    const uint8_t* ap = A + (long long)tok[t] * K + kb * 32;
                    const uint4 a0 = *(const uint4*)ap, a1 = *(const uint4*)(ap + 16);
                    const uint32_t ra[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                    float d = 0.f;
#pragma unroll
                    for (int q = 0; q < 8; q++) {
                        float a[4];
                        e4m3x4_to_f32x4(ra[q], a);
                        d = fmaf(w[q * 4 + 0], a[0], d);
                        d = fmaf(w[q * 4 + 1], a[1], d);
                        d = fmaf(w[q * 4 + 2], a[2], d);
                        d = fmaf(w[q * 4 + 3], a[3], d);
                    }
                    acc[r][t] += d * e8m0_to_f(sa[(long long)tok[t] * KB + kb]) * ws[r];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < GV_ROWS; r++)
#pragma unroll
            for (int t = 0; t < 8; t++) {
                const float v = warp_sum(acc[r][t]);
                if (lane == 0 && t < nt && r < nr) C[(long long)(row0 + tb + t) * N + nw + r] = f2bf(v);
            }
    }
}

// SwiGLU (model.py Expert.forward) fused with the down projection's act_quant:
// h = bf16(silu(min(g, lim)) * clamp(u, -lim, lim) * w), then e4m3 per 32 with a ue8m0 scale.
// gate [R][I] / up [R][I] bf16 (separate buffers, or one with up_off), w optional per-row weight.
// One warp per 32-element group.
DSV_EXTERN void dsv_swiglu_quant(uint8_t* __restrict__ q, uint8_t* __restrict__ s, const bf16* __restrict__ gate,
                                 const bf16* __restrict__ up, long long ld, const float* __restrict__ w, int R, int I,
                                 float lim, const int* __restrict__ nrows) {
    const int lane = threadIdx.x & 31;
    const long long gi = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int R_eff = nrows ? *nrows : R;
    const int kg = I >> 5;
    if (gi >= (long long)R_eff * kg) return;
    const int r = (int)(gi / kg), b = (int)(gi % kg);
    float gv = bf2f(gate[(long long)r * ld + b * 32 + lane]);
    float uv = bf2f(up[(long long)r * ld + b * 32 + lane]);
    if (lim > 0.f) {
        uv = fminf(fmaxf(uv, -lim), lim);
        gv = fminf(gv, lim);
    }
    float h = (gv / (1.f + expf(-gv))) * uv;
    if (w) h = w[r] * h;
    const float v = bf2f(f2bf(h));
    float amax = warp_max(fabsf(v));
    amax = fmaxf(amax, 1e-4f);
    const float sc = fast_pow2(fast_log2_ceil(amax * (1.0f / 448.0f)));
    q[(long long)r * I + b * 32 + lane] = f_to_e4m3(fminf(fmaxf(v / sc, -448.f), 448.f));
    if (lane == 0) s[(long long)r * kg + b] = f_to_e8m0_pow2(sc);
}

// y[t] = bf16( sum over the token's topk slots in ascending expert order of f32(down[rowpos]) + f32(shared[t]) ).
DSV_EXTERN void dsv_moe_combine(bf16* __restrict__ y, const bf16* __restrict__ down, const bf16* __restrict__ shared,
                                const int* __restrict__ idx, const int* __restrict__ rowpos, int T, int H, int topk) {
    const int t = blockIdx.x;
    int e[8], p[8];
    for (int s = 0; s < topk; s++) {
        e[s] = idx[t * topk + s];
        p[s] = rowpos[t * topk + s];
    }
    // insertion sort by expert id (topk <= 8)
    for (int i = 1; i < topk; i++)
        for (int j = i; j > 0 && e[j] < e[j - 1]; j--) {
            const int te = e[j], tp = p[j];
            e[j] = e[j - 1];
            p[j] = p[j - 1];
            e[j - 1] = te;
            p[j - 1] = tp;
        }
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < topk; s++) acc += bf2f(down[(long long)p[s] * H + h]);
        if (shared) acc += bf2f(shared[(long long)t * H + h]);
        y[(long long)t * H + h] = f2bf(acc);
    }
}
