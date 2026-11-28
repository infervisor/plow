// DeepSeek-V4.1 sm_90a: the indexer (score + top-k) and the candidate-block pre-filter.
#include "dsv41_common.cuh"

__device__ __forceinline__ void mma_bf16_ix(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// index_score[t][s] = bf16( sum_h bf16( relu(bf16(q[t][h] . k[s])) * w[t][h] ) ), -inf for s >= clen[t]
// (model.py Indexer.forward: einsum -> relu_ -> * weights -> sum(dim=2), each step a bf16 tensor).
// q [n_q][32][128], w [n_q][32] bf16 (already scaled), k_ptrs[b] -> [S][128] bf16 for batch row
// b = t / q_per_b. Block: 4 queries x 64 positions, 8 warps; warp w = query w>>1, heads (w&1)*16..+15.
#define IX_H 32
#define IX_D 128
#define IX_LD (IX_D + 8)
#define IX_SMEM ((4 * IX_H * IX_LD + 64 * IX_LD) * 2 + 2 * 4 * 64 * 4)
DSV_EXTERN void __launch_bounds__(256)
    dsv_index_score(bf16* __restrict__ out, long long out_ld, const bf16* __restrict__ q, const bf16* __restrict__ w,
                    const unsigned long long* __restrict__ k_ptrs, int q_per_b, const int* __restrict__ clen, int n_q) {
    // dynamic: Qs [128][136] bf16, Ks [64][136] bf16, part [2][4][64] f32 = IX_SMEM bytes
    extern __shared__ __align__(16) uint8_t ix_smem[];
    bf16* Qs = (bf16*)ix_smem;
    bf16* Ks = Qs + 4 * IX_H * IX_LD;
    float(*part)[4][64] = (float(*)[4][64])(Ks + 64 * IX_LD);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int t0 = blockIdx.y * 4, s0 = blockIdx.x * 64;
    int maxlen = 0;
    for (int i = 0; i < 4; i++)
        if (t0 + i < n_q) maxlen = max(maxlen, clen[t0 + i]);
    const bf16 ninf = __float2bfloat16(-INFINITY);
    if (s0 >= maxlen) {
        for (int e = tid; e < 4 * 64; e += 256) {
            const int qi = e >> 6, s = s0 + (e & 63);
            if (t0 + qi < n_q) out[(long long)(t0 + qi) * out_ld + s] = ninf;
        }
        return;
    }
    // stage Q (4 queries x 32 heads x 128) and K (64 positions x 128); all 4 queries share batch row b
    // unless they straddle one, in which case the per-query K tile below is reloaded (decode: q_per_b 1).
    for (int ch = tid; ch < 4 * IX_H * IX_D / 8; ch += 256) {
        const int r = ch / (IX_D / 8), c = (ch % (IX_D / 8)) * 8;
        const int qi = r / IX_H;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (t0 + qi < n_q) v = *(const uint4*)&q[((long long)t0 * IX_H + r) * IX_D + c];
        *(uint4*)&Qs[r * IX_LD + c] = v;
    }
    const int qi_w = warp >> 1;
    const int hq = (warp & 1) * 16;
    const int t = t0 + qi_w;
    const bool same_b = (t0 / q_per_b) == (min(t0 + 3, n_q - 1) / q_per_b);
    float s[8][4];
    auto load_k = [&](int b) {
        const bf16* kb = (const bf16*)k_ptrs[b];
        for (int ch = tid; ch < 64 * IX_D / 8; ch += 256) {
            const int r = ch / (IX_D / 8), c = (ch % (IX_D / 8)) * 8;
            *(uint4*)&Ks[r * IX_LD + c] = *(const uint4*)&kb[(long long)(s0 + r) * IX_D + c];
        }
    };
    auto compute = [&]() {
#pragma unroll
        for (int j = 0; j < 8; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int k = 0; k < IX_D; k += 16) {
            uint32_t af[4];
            const bf16* qa = &Qs[(qi_w * IX_H + hq + g) * IX_LD + k + t4 * 2];
            af[0] = *(const uint32_t*)qa;
            af[1] = *(const uint32_t*)(qa + 8 * IX_LD);
            af[2] = *(const uint32_t*)(qa + 8);
            af[3] = *(const uint32_t*)(qa + 8 * IX_LD + 8);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t bf[2];
                const bf16* kb = &Ks[(j * 8 + g) * IX_LD + k + t4 * 2];
                bf[0] = *(const uint32_t*)kb;
                bf[1] = *(const uint32_t*)(kb + 8);
                mma_bf16_ix(s[j], af, bf);
            }
        }
    };
    if (same_b) {
        load_k(min(t0, n_q - 1) / q_per_b);
        __syncthreads();
        compute();
    } else {
        // rare: a 4-query tile spans batch rows -- serialize per query
        for (int qi = 0; qi < 4; qi++) {
            __syncthreads();
            if (t0 + qi < n_q) load_k((t0 + qi) / q_per_b);
            __syncthreads();
            if (qi == qi_w) compute();
        }
    }
    // per-thread: heads hq+g and hq+g+8, positions j*8 + t4*2 + {0,1}
    float wh0 = 0.f, wh1 = 0.f;
    if (t < n_q) {
        wh0 = bf2f(w[(long long)t * IX_H + hq + g]);
        wh1 = bf2f(w[(long long)t * IX_H + hq + g + 8]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const float a0 = bf2f(f2bf(fmaxf(bf2f(f2bf(s[j][e])), 0.f) * wh0));
            const float a1 = bf2f(f2bf(fmaxf(bf2f(f2bf(s[j][2 + e])), 0.f) * wh1));
            float v = a0 + a1;
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            if (g == 0) part[warp & 1][qi_w][j * 8 + t4 * 2 + e] = v;
        }
    __syncthreads();
    for (int e = tid; e < 4 * 64; e += 256) {
        const int qi = e >> 6, sp = e & 63;
        const int tt = t0 + qi;
        if (tt >= n_q) continue;
        const int spos = s0 + sp;
        const float v = part[0][qi][sp] + part[1][qi][sp];
        out[(long long)tt * out_ld + spos] = spos < clen[tt] ? f2bf(v) : ninf;
    }
}

// ---------------------------------------------------------------------------------------------------
// Top-k over bf16 scores, emitted in POSITION order (model.py: topk(...).indices.sort()).
// Row t considers positions [0, len[t]) (optionally only those whose candidate block is kept);
// k_eff = min(k, len[t]) are selected; out[t][0..kout) = pos + offset for the selected, then -1.
// Ties at the threshold resolve to the lowest positions. One block (1024 threads) per row.
__device__ __forceinline__ uint32_t bf_key(uint16_t b) { return (b & 0x8000u) ? (uint32_t)(~b & 0xffffu) : (uint32_t)(b | 0x8000u); }

__device__ __forceinline__ int block_excl_scan(int v, int* sh, int* total) {
    // 1024 threads: warp scan then scan of 32 warp totals
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    __syncthreads();
    if (lane == 31) sh[wid] = x;
    __syncthreads();
    if (wid == 0) {
        int w = sh[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        sh[lane] = w;
    }
    __syncthreads();
    const int base = wid ? sh[wid - 1] : 0;
    *total = sh[31];
    return base + x - v;
}

DSV_EXTERN void __launch_bounds__(1024)
    dsv_topk_select(int* __restrict__ out, int kout, const bf16* __restrict__ scores, long long ld,
                    const int* __restrict__ len, int k, int offset, const uint8_t* __restrict__ keep,
                    long long keep_ld, int keep_blk) {
    __shared__ int hist[256];
    __shared__ int sh[32];
    __shared__ int s_thr_hi, s_thr_lo, s_need;
    const int t = blockIdx.x;
    const int n = len[t];
    const uint16_t* row = (const uint16_t*)(scores + (long long)t * ld);
    int* o = out + (long long)t * kout;
    auto valid = [&](int p) { return !keep || keep[(long long)t * keep_ld + p / keep_blk]; };
    // count candidates
    int cnt_local = 0;
    for (int p = threadIdx.x; p < n; p += 1024) cnt_local += valid(p) ? 1 : 0;
    int total_valid;
    (void)block_excl_scan(cnt_local, sh, &total_valid);
    const int k_eff = min(k, total_valid);
    uint32_t thr = 0;  // select key > thr, plus `need` keys == thr in position order
    int need = 0;
    if (k_eff < total_valid) {
        // pass 1: high byte
        for (int i = threadIdx.x; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int p = threadIdx.x; p < n; p += 1024)
            if (valid(p)) atomicAdd(&hist[bf_key(row[p]) >> 8], 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0, b = 255;
            for (; b >= 0; b--) {
                if (acc + hist[b] >= k_eff) break;
                acc += hist[b];
            }
            s_thr_hi = b;
            s_need = k_eff - acc;
        }
        __syncthreads();
        const int hi = s_thr_hi;
        const int need_hi = s_need;
        for (int i = threadIdx.x; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int p = threadIdx.x; p < n; p += 1024) {
            if (!valid(p)) continue;
            const uint32_t key = bf_key(row[p]);
            if ((int)(key >> 8) == hi) atomicAdd(&hist[key & 0xff], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0, b = 255;
            for (; b >= 0; b--) {
                if (acc + hist[b] >= need_hi) break;
                acc += hist[b];
            }
            s_thr_lo = b;
            s_need = need_hi - acc;
        }
        __syncthreads();
        thr = ((uint32_t)hi << 8) | (uint32_t)s_thr_lo;
        need = s_need;
    }
    // compaction in position order
    int written = 0, eq_seen = 0;
    const bool all = k_eff >= total_valid;
    for (int base = 0; base < n; base += 1024) {
        const int p = base + threadIdx.x;
        int sel = 0, eq = 0;
        if (p < n && valid(p)) {
            if (all) {
                sel = 1;
            } else {
                const uint32_t key = bf_key(row[p]);
                if (key > thr) sel = 1;
                else if (key == thr) eq = 1;
            }
        }
        int eq_tot;
        const int eq_rank = block_excl_scan(eq, sh, &eq_tot) + eq_seen;
        if (eq && eq_rank < need) sel = 1;
        int sel_tot;
        const int pos_out = block_excl_scan(sel, sh, &sel_tot) + written;
        if (sel && pos_out < kout) o[pos_out] = p + offset;
        written += sel_tot;
        eq_seen += eq_tot;
    }
    for (int j = written + threadIdx.x; j < kout; j += 1024) o[j] = -1;
}

// Candidate pre-filter, level one (model.py select_candidate_blocks): block score = max over each
// `blk`-wide block of positions < len[t] (-inf beyond), the block holding position len-1 pinned to +inf.
// Writes bf16 block scores [n_q][nblk_ld]; dsv_topk_select over them (len = ceil(len/blk), k = topk
// blocks) then dsv_keep_from_idx turn the picks into the keep mask.
DSV_EXTERN void dsv_cand_block_scores(bf16* __restrict__ bs, long long bs_ld, const bf16* __restrict__ scores,
                                      long long ld, const int* __restrict__ len, int blk) {
    const int t = blockIdx.y;
    const int n = len[t];
    const int nb = (n + blk - 1) / blk;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
        float m = -INFINITY;
        for (int i = 0; i < blk; i++) {
            const int p = b * blk + i;
            if (p < n) m = fmaxf(m, bf2f(scores[(long long)t * ld + p]));
        }
        if (b == (n - 1) / blk) m = INFINITY;
        bs[(long long)t * bs_ld + b] = f2bf(m);
    }
}
DSV_EXTERN void dsv_keep_from_idx(uint8_t* __restrict__ keep, long long keep_ld, int nblk_ld, const int* __restrict__ idx,
                                  int kidx) {
    const int t = blockIdx.x;
    for (int b = threadIdx.x; b < nblk_ld; b += blockDim.x) keep[(long long)t * keep_ld + b] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < kidx; j += blockDim.x) {
        const int b = idx[(long long)t * kidx + j];
        if (b >= 0) keep[(long long)t * keep_ld + b] = 1;
    }
}
