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
// Single block: offs[e] = exclusive prefix of counts; tile list (expert, row0) of BM-row tiles.
// meta[0] = number of tiles. fill_ctr[e] zeroed for the fill pass.
DSV_EXTERN void dsv_moe_offsets(int* __restrict__ offs, int* __restrict__ tiles, int* __restrict__ meta,
                                int* __restrict__ fill_ctr, const int* __restrict__ counts, int E, int BM) {
    if (threadIdx.x != 0) return;
    int acc = 0, nt = 0;
    for (int e = 0; e < E; e++) {
        offs[e] = acc;
        fill_ctr[e] = 0;
        const int c = counts[e];
        for (int r = 0; r < c; r += BM) {
            tiles[nt * 2] = e;
            tiles[nt * 2 + 1] = acc + r;
            nt++;
        }
        acc += c;
    }
    offs[E] = acc;
    meta[0] = nt;
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

// e2m1 nibble -> e4m3 byte (exact: every e2m1 value is an e4m3 value).
__constant__ uint8_t kE2M1toE4M3[16] = {0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
                                        0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC};

// Grouped expert GEMM (kernel.py fp4_gemm at act block 32): for tile i = (expert e, row0), rows
// row0 .. min(row0+64, offs[e+1]) of the expert-sorted list: C[p][n] = sum_kb (A8[tok(p)] . W4[e][n]) *
// sa[tok(p)][kb] * sw[e][n][kb]. A is indexed by token (gather through `rows`) unless a_by_row, in which
// case A row p itself (the down projection reads the SwiGLU output, already in sorted order).
// W fp4 packed [N][K/2] per expert (element 2i in the low nibble), sw ue8m0 [N][K/32].
#define G_BM 64
#define G_BN 128
#define G_LD 48
DSV_EXTERN void __launch_bounds__(128)
    dsv_moe_gemm_fp4(bf16* __restrict__ C, const uint8_t* __restrict__ A, const uint8_t* __restrict__ sa,
                     const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw, const int* __restrict__ tiles,
                     const int* __restrict__ meta, const int* __restrict__ offs, const int* __restrict__ rows,
                     int a_by_row, int N, int K, long long w_estride, long long sw_estride) {
    __shared__ __align__(16) uint8_t As[G_BM * G_LD];
    __shared__ __align__(16) uint8_t Ws[G_BN * G_LD];
    __shared__ float Ss[G_BN];
    __shared__ int Arow[G_BM];
    const int tile = blockIdx.y;
    if (tile >= meta[0]) return;
    const int e = tiles[tile * 2], row0 = tiles[tile * 2 + 1];
    const int rend = offs[e + 1];
    const int n0 = blockIdx.x * G_BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int g = lane >> 2, t4 = lane & 3;
    const int KB = K >> 5;
    const uint8_t* We = W + (long long)e * w_estride;
    const uint8_t* Se = sw + (long long)e * sw_estride;
    if (tid < G_BM) {
        const int p = row0 + tid;
        Arow[tid] = p < rend ? (a_by_row ? p : rows[p]) : -1;
    }
    __syncthreads();
    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;
    int arows[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        arows[i][0] = Arow[wm * 32 + i * 16 + g];
        arows[i][1] = Arow[wm * 32 + i * 16 + g + 8];
    }
    for (int kb = 0; kb < KB; kb++) {
        {  // A: 64 rows x 32 B
            const int r = tid >> 1, c = (tid & 1) * 16;
            const int ar = Arow[r];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (ar >= 0) v = *(const uint4*)(A + (long long)ar * K + kb * 32 + c);
            *(uint4*)&As[r * G_LD + c] = v;
        }
        {  // W: 128 rows x 16 B packed -> 32 B e4m3, plus the per-row scale
            const int r = tid;
            const int gn = n0 + r;
            uint8_t o[32];
            float s = 0.f;
            if (gn < N) {
                const uint4 raw = *(const uint4*)(We + (long long)gn * (K / 2) + kb * 16);
                const uint8_t* rb = (const uint8_t*)&raw;
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    o[2 * i] = kE2M1toE4M3[rb[i] & 0xf];
                    o[2 * i + 1] = kE2M1toE4M3[rb[i] >> 4];
                }
                s = e8m0_to_f(Se[(long long)gn * KB + kb]);
            } else {
#pragma unroll
                for (int i = 0; i < 32; i++) o[i] = 0;
            }
            *(uint4*)&Ws[r * G_LD] = *(const uint4*)&o[0];
            *(uint4*)&Ws[r * G_LD + 16] = *(const uint4*)&o[16];
            Ss[r] = s;
        }
        __syncthreads();
        uint32_t af[2][4], bfr[8][2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint8_t* base = &As[(wm * 32 + i * 16 + g) * G_LD + t4 * 4];
            af[i][0] = *(const uint32_t*)(base);
            af[i][1] = *(const uint32_t*)(base + 8 * G_LD);
            af[i][2] = *(const uint32_t*)(base + 16);
            af[i][3] = *(const uint32_t*)(base + 8 * G_LD + 16);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint8_t* base = &Ws[(wn * 64 + j * 8 + g) * G_LD + t4 * 4];
            bfr[j][0] = *(const uint32_t*)(base);
            bfr[j][1] = *(const uint32_t*)(base + 16);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const float sa0 = arows[i][0] >= 0 ? e8m0_to_f(sa[(long long)arows[i][0] * KB + kb]) : 0.f;
            const float sa1 = arows[i][1] >= 0 ? e8m0_to_f(sa[(long long)arows[i][1] * KB + kb]) : 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                float tt[4] = {0.f, 0.f, 0.f, 0.f};
                mma_e4m3_m(tt, af[i], bfr[j]);
                const int cn = wn * 64 + j * 8 + t4 * 2;
                const float s0 = Ss[cn], s1 = Ss[cn + 1];
                acc[i][j][0] += tt[0] * sa0 * s0;
                acc[i][j][1] += tt[1] * sa0 * s1;
                acc[i][j][2] += tt[2] * sa1 * s0;
                acc[i][j][3] += tt[3] * sa1 * s1;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = wm * 32 + i * 16 + g + h * 8;
            const int p = row0 + r;
            if (p >= rend) continue;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int n = n0 + wn * 64 + j * 8 + t4 * 2;
                if (n >= N) continue;
                __nv_bfloat162 v;
                v.x = f2bf(acc[i][j][h * 2]);
                v.y = f2bf(acc[i][j][h * 2 + 1]);
                *(__nv_bfloat162*)&C[(long long)p * N + n] = v;
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
