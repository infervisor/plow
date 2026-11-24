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

// Decode form of the grouped fp4 expert GEMM: swap-AB tensor-core GEMV. For tile i = (expert e, row0)
// of dsv_moe_offsets at bm = 8 (up to 8 routed rows of one expert): the expert's weights are the
// MMA's M side (16 rows per warp), the tile's rows its N side. A 32-wide K block of an fp4 row is only
// 16 bytes, so lanes load 8 contiguous bytes at t4 * 8 of a PAIR of blocks (whole 32 B sectors per
// instruction): lanes 0, 1 hold block 2j's bytes 0-7 / 8-15, lanes 2, 3 block 2j+1's, and one
// shfl.xor 2 hands each lane the other block's half it needs. Lane t4 thus feeds block 2j with bytes
// at off = (t4 & 1) * 8 + (t4 >> 1) * 4 and block 2j+1 with off = (t4 & 1) * 8 + (1 - (t4 >> 1)) * 4 --
// permutations inside each block, mirrored on the activation side (8 bytes at 2 * off). The shift
// decode gives the even elements (-> a0 / a1) and the odd ones (-> a2 / a3) as e4m3 * 2^-6, the
// activation bytes are permuted to the same order (b0 = even, b1 = odd); each MMA is one scale block,
// promoted with sa[tok][kb] * sw[e][n][kb] * 2^6 as dsv_moe_gemm_fp4. GVM_U block pairs are loaded
// before any MMA; KB must be even. Block: 4 warps = 64 weight rows; grid = (ceil(N / 64),
// max_tiles, ksplit); ksplit > 1 writes part [ksplit][nrows][N] f32 for dsv_splitk_reduce (in split
// order, deterministic), else C bf16 directly.
#ifndef GVM_U
#define GVM_U 8
#endif
DSV_EXTERN void __launch_bounds__(128)
    dsv_moe_gemv_fp4(bf16* __restrict__ C, const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                     const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw, const int* __restrict__ tiles,
                     const int* __restrict__ meta, const int* __restrict__ offs, const int* __restrict__ rows,
                     int a_by_row, int N, int K, long long w_estride, long long sw_estride, float* __restrict__ part, int nrows) {
    const int tile = blockIdx.y;
    if (tile >= meta[0]) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int n0 = (blockIdx.x * 4 + warp) * 16;
    if (n0 >= N) return;
    const int e = tiles[tile * 2], row0 = tiles[tile * 2 + 1];
    const int R = min(offs[e + 1], row0 + 8) - row0;
    const int KB = K >> 5;
    const int split = blockIdx.z, ksplit = gridDim.z;
    const int KG = KB >> 4;  // 16-block granules: splits and batches start 8-byte aligned in the scale rows
    const int kb0 = 16 * (int)((long long)KG * split / ksplit);
    const int kb1 = split + 1 == ksplit ? KB : 16 * (int)((long long)KG * (split + 1) / ksplit);
    const uint8_t* We = W + (long long)e * w_estride;
    const uint8_t* Se = sw + (long long)e * sw_estride;
    const int ra = min(n0 + g, N - 1), rb = min(n0 + g + 8, N - 1);
    const uint8_t* w0 = We + (long long)ra * (K / 2) + t4 * 4;
    const uint8_t* w1 = We + (long long)rb * (K / 2) + t4 * 4;
    const uint8_t* s0p = Se + (long long)ra * KB;
    const uint8_t* s1p = Se + (long long)rb * KB;
    // this lane's B column (row g of the tile) and C columns (rows t4*2, t4*2+1)
    const bool bv = g < R;
    const int tb = bv ? (a_by_row ? row0 + g : rows[row0 + g]) : 0;
    const uint8_t* ab = A + (long long)tb * K + t4 * 8;
    int tc[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int r = t4 * 2 + h;
        tc[h] = r < R ? (a_by_row ? row0 + r : rows[row0 + r]) : -1;
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const bool hi_lane = t4 >= 2;
    const int off_e = (t4 & 1) * 8 + (t4 >> 1) * 4, off_o = (t4 & 1) * 8 + (1 - (t4 >> 1)) * 4;
    // scale bytes: weight rows g / g + 8 (s0 / s1) and the two C tokens (q0 / q1); 0 = the zero scale
    const uint8_t* q0p = tc[0] >= 0 ? sa + (long long)tc[0] * KB : nullptr;
    const uint8_t* q1p = tc[1] >= 0 ? sa + (long long)tc[1] * KB : nullptr;
    auto block = [&](uint32_t wa, uint32_t wb, uint2 x, uint32_t s0, uint32_t s1, uint32_t q0, uint32_t q1) {
        const uint32_t a[4] = {fp4x8_lo(wa), fp4x8_lo(wb), fp4x8_hi(wa), fp4x8_hi(wb)};
        const uint32_t b[2] = {__byte_perm(x.x, x.y, 0x6420), __byte_perm(x.x, x.y, 0x7531)};
        float tt[4] = {0.f, 0.f, 0.f, 0.f};
        mma_e4m3_m(tt, a, b);
        const float swa = e8m0_to_f(s0) * FP4_E4M3_UNSCALE, swb = e8m0_to_f(s1) * FP4_E4M3_UNSCALE;
        const float sa0 = q0p ? e8m0_to_f(q0) : 0.f, sa1 = q1p ? e8m0_to_f(q1) : 0.f;
        acc[0] += tt[0] * swa * sa0;
        acc[1] += tt[1] * swa * sa1;
        acc[2] += tt[2] * swb * sa0;
        acc[3] += tt[3] * swb * sa1;
    };
    auto sbyte = [](uint2 v, int i) { return ((i < 4 ? v.x : v.y) >> (8 * (i & 3))) & 0xffu; };
    // one pair of blocks (kb, kb + 1) from this lane's 8-byte loads of rows g / g + 8
    auto pair = [&](uint2 va, uint2 vb, uint2 xe, uint2 xo, const uint32_t* s0, const uint32_t* s1, const uint32_t* q0,
                    const uint32_t* q1) {
        const uint32_t pa = __shfl_xor_sync(0xffffffffu, va.y, 2), pb = __shfl_xor_sync(0xffffffffu, vb.y, 2);
        block(hi_lane ? pa : va.x, hi_lane ? pb : vb.x, xe, s0[0], s1[0], q0[0], q1[0]);
        block(hi_lane ? va.x : pa, hi_lane ? vb.x : pb, xo, s0[1], s1[1], q0[1], q1[1]);
    };
    const uint8_t* ab_e = ab - t4 * 8 + off_e * 2;  // activation bytes for block 2j / 2j + 1
    const uint8_t* ab_o = ab - t4 * 8 + off_o * 2 + 32;
    const uint8_t* wp0 = w0 - t4 * 4 + t4 * 8;  // 8 bytes at t4 * 8 of each 32-byte block pair
    const uint8_t* wp1 = w1 - t4 * 4 + t4 * 8;
    int kb = kb0;
    // batches of 8 block pairs (16 blocks): weights, activations and all their scale bytes issued first
    static_assert(GVM_U == 8, "a batch is 16 blocks: two 8-byte scale loads per row");
    for (; kb + 16 <= kb1; kb += 16) {
        uint2 va[8], vb[8], xe[8], xo[8], sv0[2], sv1[2], sq0[2], sq1[2];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            const long long kp = (long long)(kb + 2 * u);
            va[u] = __ldg((const uint2*)(wp0 + kp * 16));
            vb[u] = __ldg((const uint2*)(wp1 + kp * 16));
            xe[u] = bv ? *(const uint2*)(ab_e + kp * 32) : make_uint2(0, 0);
            xo[u] = bv ? *(const uint2*)(ab_o + kp * 32) : make_uint2(0, 0);
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            sv0[h] = __ldg((const uint2*)(s0p + kb + 8 * h));
            sv1[h] = __ldg((const uint2*)(s1p + kb + 8 * h));
            sq0[h] = q0p ? *(const uint2*)(q0p + kb + 8 * h) : make_uint2(0, 0);
            sq1[h] = q1p ? *(const uint2*)(q1p + kb + 8 * h) : make_uint2(0, 0);
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            uint32_t s0[2], s1[2], q0[2], q1[2];
#pragma unroll
            for (int d = 0; d < 2; d++) {
                const int bi = 2 * u + d;  // block within the batch
                s0[d] = sbyte(sv0[bi >> 3], bi & 7);
                s1[d] = sbyte(sv1[bi >> 3], bi & 7);
                q0[d] = sbyte(sq0[bi >> 3], bi & 7);
                q1[d] = sbyte(sq1[bi >> 3], bi & 7);
            }
            pair(va[u], vb[u], xe[u], xo[u], s0, s1, q0, q1);
        }
    }
    for (; kb < kb1; kb += 2) {
        const long long kp = kb;
        const uint32_t s0[2] = {s0p[kb], s0p[kb + 1]}, s1[2] = {s1p[kb], s1p[kb + 1]};
        const uint32_t q0[2] = {q0p ? q0p[kb] : 0u, q0p ? q0p[kb + 1] : 0u}, q1[2] = {q1p ? q1p[kb] : 0u, q1p ? q1p[kb + 1] : 0u};
        pair(__ldg((const uint2*)(wp0 + kp * 16)), __ldg((const uint2*)(wp1 + kp * 16)),
             bv ? *(const uint2*)(ab_e + kp * 32) : make_uint2(0, 0), bv ? *(const uint2*)(ab_o + kp * 32) : make_uint2(0, 0), s0, s1,
             q0, q1);
    }
    // c0 (n g, row t4*2), c1 (n g, row t4*2+1), c2 (n g+8, row t4*2), c3 (n g+8, row t4*2+1)
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const int n = n0 + g + (q >> 1) * 8, r = t4 * 2 + (q & 1);
        if (n >= N || r >= R) continue;
        const long long o = (long long)(row0 + r) * N + n;
        if (part) part[(long long)split * nrows * N + o] = acc[q];
        else C[o] = f2bf(acc[q]);
    }
}

// SwiGLU (model.py Expert.forward) fused with the down projection's act_quant:
// h = bf16(silu(min(g, lim)) * clamp(u, -lim, lim) * w), then e4m3 per 32 with a ue8m0 scale; fq
// (optional) takes the dequantized bf16 [R][I] for the wgmma GEMMs.
// gate [R][I] / up [R][I] bf16 (separate buffers, or one with up_off), w optional per-row weight.
// One warp per 32-element group.
DSV_EXTERN void dsv_swiglu_quant(uint8_t* __restrict__ q, uint8_t* __restrict__ s, const bf16* __restrict__ gate,
                                 const bf16* __restrict__ up, long long ld, const float* __restrict__ w, int R, int I,
                                 float lim, const int* __restrict__ nrows, bf16* __restrict__ fq) {
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
    const uint8_t code = f_to_e4m3(fminf(fmaxf(v / sc, -448.f), 448.f));
    q[(long long)r * I + b * 32 + lane] = code;
    if (lane == 0) s[(long long)r * kg + b] = f_to_e8m0_pow2(sc);
    if (fq) fq[(long long)r * I + b * 32 + lane] = f2bf(e4m3_to_f(code) * sc);
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
