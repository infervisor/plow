// DeepSeek-V4.1 sm_90a: RoPE and the sparse (gathered) attention.
#include "dsv41_common.cuh"

// In-place interleaved rotary on the last `rd` dims of each head row (model.py apply_rotary_emb):
// pairs (x[2i], x[2i+1]) times (cos, sin)[pos][i], conjugated when `inverse`. fp32 math, bf16 store.
// Row r = token t * n_head + head h lives at x + t*tok_stride + h*head_stride; `off` is where the
// rotary tail starts. pos[t] indexes the [max_pos][rd/2] tables; pos_mul/pos_add remap it
// (compressed group j of ratio r sits at position j*r).
DSV_EXTERN void dsv_rope(bf16* __restrict__ x, const int* __restrict__ pos, const float* __restrict__ cosb,
                         const float* __restrict__ sinb, int n_tok, int n_head, long long tok_stride,
                         long long head_stride, int off, int rd, int pos_mul, int pos_add, int inverse) {
    const int half = rd >> 1;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)n_tok * n_head * half;
    if (i >= total) return;
    const int p = (int)(i % half);
    const long long th = i / half;
    const int h = (int)(th % n_head);
    const int t = (int)(th / n_head);
    const int ps = (pos ? pos[t] : t) * pos_mul + pos_add;
    bf16* xp = x + (long long)t * tok_stride + (long long)h * head_stride + off + 2 * p;
    const float a = bf2f(xp[0]), b = bf2f(xp[1]);
    const float c = cosb[(long long)ps * half + p];
    const float s = inverse ? -sinb[(long long)ps * half + p] : sinb[(long long)ps * half + p];
    xp[0] = f2bf(a * c - b * s);
    xp[1] = f2bf(a * s + b * c);
}

// ---------------------------------------------------------------------------------------------------
// Sparse attention (kernel.py sparse_attn), 64 heads x 512 dims, one KV head that is both K and V.
//
// Query row t (of batch row b = t / q_per_b) attends over idx[t][0..n_idx): an index < off reads
// win[b][idx], one >= off reads cmp[b][idx - off], -1 reads nothing. Online softmax over tiles of 64
// gathered rows; the sink joins the denominator once at the end; the running max starts at -1e30 so
// a row with no valid index returns zeros, as the reference does.
//
// Block: 512 threads (16 warps), one query row. Warp w: head tile (w & 3) x 16 heads; for S it owns
// kv columns (w >> 2) * 16, for O it owns dims (w >> 2) * 128.
#define SA_H 64
#define SA_D 512
#define SA_BN 64
#define SA_LDQ (SA_D + 8)
#define SA_LDP (SA_BN + 8)

__device__ __forceinline__ void mma_bf16_sa(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldsm_x2_trans(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

DSV_EXTERN void __launch_bounds__(512)
    dsv_sparse_attn(bf16* __restrict__ o, const bf16* __restrict__ q, const int* __restrict__ idx, int n_idx,
                    const unsigned long long* __restrict__ win_ptrs, const unsigned long long* __restrict__ cmp_ptrs,
                    int off, int q_per_b, const float* __restrict__ sink, float scale) {
    extern __shared__ __align__(16) uint8_t smem[];
    bf16* Qs = (bf16*)smem;                         // [64][520]
    bf16* Ks = Qs + SA_H * SA_LDQ;                  // [64][520]
    bf16* Ps = Ks + SA_BN * SA_LDQ;                 // [64][72]
    float* redmax = (float*)(Ps + SA_H * SA_LDP);   // [4][64]
    float* redsum = redmax + 4 * SA_H;              // [4][64]
    int* sidx = (int*)(redsum + 4 * SA_H);          // [64]

    const int t = blockIdx.x;
    const int b = t / q_per_b;
    const bf16* win = (const bf16*)win_ptrs[b];
    const bf16* cmp = cmp_ptrs ? (const bf16*)cmp_ptrs[b] : nullptr;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int mt = warp & 3, nq = warp >> 2;

    // Q tile
    const bf16* qt = q + (long long)t * SA_H * SA_D;
    for (int ch = tid; ch < SA_H * SA_D / 8; ch += 512) {
        const int r = ch / (SA_D / 8), c = (ch % (SA_D / 8)) * 8;
        *(uint4*)&Qs[r * SA_LDQ + c] = *(const uint4*)&qt[r * SA_D + c];
    }

    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    // running max / sum for rows g and g+8 of this warp's head tile (replicated across the 4 nq warps)
    float mrow[2] = {-1e30f, -1e30f}, lrow[2] = {0.f, 0.f};

    const int n_tiles = (n_idx + SA_BN - 1) / SA_BN;
    for (int tile = 0; tile < n_tiles; tile++) {
        __syncthreads();
        if (tid < SA_BN) {
            const int j = tile * SA_BN + tid;
            sidx[tid] = j < n_idx ? idx[(long long)t * n_idx + j] : -1;
        }
        __syncthreads();
        for (int ch = tid; ch < SA_BN * SA_D / 8; ch += 512) {
            const int r = ch / (SA_D / 8), c = (ch % (SA_D / 8)) * 8;
            const int id = sidx[r];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (id >= 0) {
                const bf16* src = id < off ? win + (long long)id * SA_D : cmp + (long long)(id - off) * SA_D;
                v = *(const uint4*)&src[c];
            }
            *(uint4*)&Ks[r * SA_LDQ + c] = v;
        }
        __syncthreads();

        // S[16 heads][16 kv] for this warp: two n8 tiles, K = 512
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 8
        for (int k = 0; k < SA_D; k += 16) {
            uint32_t af[4];
            const bf16* qa = &Qs[(mt * 16 + g) * SA_LDQ + k + t4 * 2];
            af[0] = *(const uint32_t*)qa;
            af[1] = *(const uint32_t*)(qa + 8 * SA_LDQ);
            af[2] = *(const uint32_t*)(qa + 8);
            af[3] = *(const uint32_t*)(qa + 8 * SA_LDQ + 8);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                uint32_t bf[2];
                const bf16* kb = &Ks[(nq * 16 + j * 8 + g) * SA_LDQ + k + t4 * 2];
                bf[0] = *(const uint32_t*)kb;
                bf[1] = *(const uint32_t*)(kb + 8);
                mma_bf16_sa(s[j], af, bf);
            }
        }
        // scale + mask, partial row max
        float pmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int col = nq * 16 + j * 8 + t4 * 2 + (e & 1);
                float v = sidx[col] >= 0 ? s[j][e] * scale : -INFINITY;
                s[j][e] = v;
                pmax[e >> 1] = fmaxf(pmax[e >> 1], v);
            }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            pmax[h] = fmaxf(pmax[h], __shfl_xor_sync(0xffffffffu, pmax[h], 1));
            pmax[h] = fmaxf(pmax[h], __shfl_xor_sync(0xffffffffu, pmax[h], 2));
        }
        if (t4 == 0) {
            redmax[nq * SA_H + mt * 16 + g] = pmax[0];
            redmax[nq * SA_H + mt * 16 + g + 8] = pmax[1];
        }
        __syncthreads();
        float mnew[2], alpha[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int row = mt * 16 + g + h * 8;
            float mx = fmaxf(fmaxf(redmax[row], redmax[SA_H + row]), fmaxf(redmax[2 * SA_H + row], redmax[3 * SA_H + row]));
            mnew[h] = fmaxf(mrow[h], mx);
            alpha[h] = __expf(mrow[h] - mnew[h]);
        }
        float psum[2] = {0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int h = e >> 1;
                const float p = __expf(s[j][e] - mnew[h]);
                psum[h] += p;
                const int row = mt * 16 + g + h * 8;
                const int col = nq * 16 + j * 8 + t4 * 2 + (e & 1);
                Ps[row * SA_LDP + col] = f2bf(p);
            }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            psum[h] += __shfl_xor_sync(0xffffffffu, psum[h], 1);
            psum[h] += __shfl_xor_sync(0xffffffffu, psum[h], 2);
        }
        if (t4 == 0) {
            redsum[nq * SA_H + mt * 16 + g] = psum[0];
            redsum[nq * SA_H + mt * 16 + g + 8] = psum[1];
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int row = mt * 16 + g + h * 8;
            const float tot = redsum[row] + redsum[SA_H + row] + redsum[2 * SA_H + row] + redsum[3 * SA_H + row];
            lrow[h] = lrow[h] * alpha[h] + tot;
            mrow[h] = mnew[h];
        }
        // O = O * alpha + P . V ; this warp: heads mt*16.., dims nq*128 .. +128 (16 n8 tiles), K = 64 kv
#pragma unroll
        for (int j = 0; j < 16; j++) {
            acc[j][0] *= alpha[0];
            acc[j][1] *= alpha[0];
            acc[j][2] *= alpha[1];
            acc[j][3] *= alpha[1];
        }
#pragma unroll
        for (int k = 0; k < SA_BN; k += 16) {
            uint32_t af[4];
            const bf16* pa = &Ps[(mt * 16 + g) * SA_LDP + k + t4 * 2];
            af[0] = *(const uint32_t*)pa;
            af[1] = *(const uint32_t*)(pa + 8 * SA_LDP);
            af[2] = *(const uint32_t*)(pa + 8);
            af[3] = *(const uint32_t*)(pa + 8 * SA_LDP + 8);
#pragma unroll
            for (int j = 0; j < 16; j++) {
                uint32_t bf[2];
                // rows k..k+15 of V, columns nq*128 + j*8 .. +8; lanes 0-15 give the row addresses
                const bf16* vb = &Ks[(k + (lane & 15)) * SA_LDQ + nq * 128 + j * 8];
                ldsm_x2_trans(bf, vb);
                mma_bf16_sa(acc[j], af, bf);
            }
        }
    }

    // sink + normalize
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int row = mt * 16 + g + h * 8;
        lrow[h] += __expf(sink[row] - mrow[h]);
    }
    bf16* ot = o + (long long)t * SA_H * SA_D;
#pragma unroll
    for (int j = 0; j < 16; j++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int row = mt * 16 + g + h * 8;
            const int col = nq * 128 + j * 8 + t4 * 2;
            __nv_bfloat162 v;
            v.x = f2bf(acc[j][h * 2] / lrow[h]);
            v.y = f2bf(acc[j][h * 2 + 1] / lrow[h]);
            *(__nv_bfloat162*)&ot[row * SA_D + col] = v;
        }
}

// Attention index table for one forward. Row t (batch row b = t / q_per_b) gets n_win window slots then
// the compressed picks shifted by `off`:
//   prefill (win_ring == 0): slots max(0, p-W+1) .. p of the chunk itself, p = t (chunk-relative), -1 pad;
//   decode (win_ring == 1): ring slots listed oldest first, -1 while pos[b] has not reached them.
// cmp_idx[t][0..n_cmp) is copied through (already offset, -1 = none) when present.
DSV_EXTERN void dsv_attn_index(int* __restrict__ out, int n_rows, int q_per_b, int W, int win_ring,
                               const int* __restrict__ pos, const int* __restrict__ cmp_idx, int n_cmp) {
    const int t = blockIdx.x;
    if (t >= n_rows) return;
    const int stride = W + n_cmp;
    for (int j = threadIdx.x; j < stride; j += blockDim.x) {
        int v;
        if (j < W) {
            if (!win_ring) {
                const int start = max(t - W + 1, 0);
                const int id = start + j;
                v = id > t ? -1 : id;
            } else {
                const int p = pos[t / q_per_b];
                const int oldest = p % W + 1;
                const int id = oldest + j < W ? oldest + j : oldest + j - W;
                v = id > p ? -1 : id;
            }
        } else {
            v = cmp_idx[(long long)t * n_cmp + (j - W)];
        }
        out[(long long)t * stride + j] = v;
    }
}
