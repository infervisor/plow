/* sample_sm120.cu — device-side stochastic sampler. ONE 256-thread block per
 * batch row samples that row's next token from its
 * `[V]` bf16 logits and writes it where the next decode step's EMBED reads it (`in.ids[b]`),
 * exactly as ARGMAX_FIN does for greedy — so `temperature > 0` no longer downloads the whole
 * vocabulary row to the host, and the host softmax+full-vocab sort leaves the critical path.
 *
 * SEMANTICS (the device sampler's own contract; the host CPU sampler stays as the reference /
 * debug mode). Given per-row temperature t, top_k, top_p, min_p and a uniform rng01 in [0,1):
 *   e_i    = exp((l_i - lmax) / t)                    (unnormalised softmax weight, e in (0,1])
 *   min_p  keeps e_i >= min_p                          (max weight is 1, so this is p_i >= min_p*p_max)
 *   top_k  keeps the k largest weights                 (threshold = the k-th largest e_i)
 *   top_p  keeps the smallest high-weight set whose mass >= top_p * total
 *   draw   inverse-CDF over the kept set in INDEX order, target = rng01 * sum(kept e_i)
 * The three truncations compose into ONE weight floor; the draw is index-order (no device
 * sort) — deterministic for a fixed rng01 and distributionally exact vs the kept set. The
 * two boundary searches (top_k, top_p) are threshold bisections, each a handful of O(V) block
 * reductions; min_p is a direct floor. t <= 0 is greedy argmax with the ARGMAX tie-break
 * (lowest index wins), byte-identical to d_argmax_fin.
 *
 * NOT handled on device (host keeps these; the engine only routes rows the device can finish):
 *   - repetition/frequency/presence penalties (need per-row token history — DeviceRunState),
 *   - structured-decoding masks,
 *   - the pathological no-truncation config (top_k==0 && top_p>=1 && min_p==0): the kept set is
 *     the full vocab and index-order inverse-CDF over 262k is pointless work — the host path is
 *     used instead. The engine checks the params and only launches this kernel for rows with at
 *     least one active truncation and no penalties/mask.
 */

#include <cuda_bf16.h>

#ifndef PLOW_SMP_THREADS
#define PLOW_SMP_THREADS 1024
#endif
#define PLOW_SMP_WARPS (PLOW_SMP_THREADS / 32)
/* The block width this object was built for; the engine launches with it (absent => 256). */
extern "C" __device__ unsigned plow_sample_threads = PLOW_SMP_THREADS;
/* Candidate capacity of the fast path (top_p/min_p rows, no top_k). */
#ifndef PLOW_SMP_CAND
#define PLOW_SMP_CAND 4096
#endif

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}
__device__ __forceinline__ float warp_max32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float x = __shfl_xor_sync(0xffffffffu, v, o, 32);
        v = x > v ? x : v;
    }
    return v;
}
__device__ __forceinline__ float block_reduce(float v, float* part, bool is_max) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    v = is_max ? warp_max32(v) : warp_sum32(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = part[0];
#pragma unroll
        for (unsigned w = 1; w < PLOW_SMP_WARPS; w++) r = is_max ? (part[w] > r ? part[w] : r) : r + part[w];
        part[0] = r;
    }
    __syncthreads();
    const float r = part[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float block_max(float v, float* part) { return block_reduce(v, part, true); }
__device__ __forceinline__ float block_sum(float v, float* part) { return block_reduce(v, part, false); }

/* Mass (sum of weights) with weight >= floor — the reduction both bisections evaluate. */
__device__ __forceinline__ float mass_ge(const float* e, unsigned V, float floor, float* part) {
    float s = 0.0f;
    for (unsigned i = threadIdx.x; i < V; i += PLOW_SMP_THREADS)
        if (e[i] >= floor) s += e[i];
    return block_sum(s, part);
}
/* Count of weights >= floor (top_k bisection target). */
__device__ __forceinline__ float count_ge(const float* e, unsigned V, float floor, float* part) {
    float c = 0.0f;
    for (unsigned i = threadIdx.x; i < V; i += PLOW_SMP_THREADS)
        if (e[i] >= floor) c += 1.0f;
    return block_sum(c, part);
}

/* FAST PATH for rows with top_p < 1 and no top_k (the TTS/chat default).
 *
 * Exact candidate filter: with c = max(eps, min_p), eps = (1 - top_p) * total / V, every weight
 * below eps sums to at most (1 - top_p) * total, so the top_p floor is >= eps and no weight below
 * c can be kept. For every floor >= c the candidates' mass equals the full vocabulary's, so the
 * bisection over the candidates finds the same floor the full-vocabulary one does. A peaked row
 * leaves a few hundred candidates; three streaming passes over the bf16 logits (max, total,
 * compact) replace ~27 passes over an f32 scratch copy, with contiguous per-thread chunks and
 * independent loads so the passes are not latency-bound on one SM.
 *
 * The draw is inverse-CDF in ascending INDEX order over the kept set (the legacy path's order is
 * thread-strided); for a fixed rng01 the token can differ from the legacy path, the distribution
 * does not. Returns false (nothing written) when the candidates overflow PLOW_SMP_CAND. */
__device__ bool sample_fast(const __nv_bfloat16* __restrict__ row, unsigned V, float t, float tp,
                            float mp, float u, int* __restrict__ out) {
    __shared__ unsigned c_idx[PLOW_SMP_CAND];
    __shared__ float c_e[PLOW_SMP_CAND];
    __shared__ float part[PLOW_SMP_WARPS];
    __shared__ unsigned cnt[PLOW_SMP_THREADS];
    __shared__ float pre[PLOW_SMP_THREADS];
    __shared__ unsigned sh_pick;
    const unsigned T = PLOW_SMP_THREADS, tid = threadIdx.x;
    const unsigned chunk = (V + T - 1u) / T;
    const unsigned lo = min(V, tid * chunk), hi = min(V, lo + chunk);

    float m = -3.4e38f;
    for (unsigned i = lo; i < hi; i += 8u) {
        float v[8];
#pragma unroll
        for (unsigned j = 0; j < 8u; j++) v[j] = i + j < hi ? __bfloat162float(row[i + j]) : -3.4e38f;
#pragma unroll
        for (unsigned j = 0; j < 8u; j++) m = fmaxf(m, v[j]);
    }
    m = block_max(m, part);
    const float inv_t = 1.0f / t;
    float s = 0.0f;
    for (unsigned i = lo; i < hi; i += 8u) {
        float v[8];
#pragma unroll
        for (unsigned j = 0; j < 8u; j++) v[j] = i + j < hi ? __bfloat162float(row[i + j]) : -3.4e38f;
#pragma unroll
        for (unsigned j = 0; j < 8u; j++) s += i + j < hi ? expf((v[j] - m) * inv_t) : 0.0f;
    }
    const float total = block_sum(s, part);
    const float c = fmaxf((1.0f - tp) * total / (float)V, mp);

    /* Compact candidates in index order: count per chunk, exclusive scan, write. */
    unsigned n = 0;
    for (unsigned i = lo; i < hi; i += 8u) {
        float v[8];
#pragma unroll
        for (unsigned j = 0; j < 8u; j++) v[j] = i + j < hi ? __bfloat162float(row[i + j]) : -3.4e38f;
#pragma unroll
        for (unsigned j = 0; j < 8u; j++) n += (i + j < hi && expf((v[j] - m) * inv_t) >= c) ? 1u : 0u;
    }
    cnt[tid] = n;
    __syncthreads();
    for (unsigned d = 1; d < T; d <<= 1) {
        const unsigned add = tid >= d ? cnt[tid - d] : 0u;
        __syncthreads();
        cnt[tid] += add;
        __syncthreads();
    }
    const unsigned N = cnt[T - 1u];
    if (N > PLOW_SMP_CAND || N == 0u) return false; /* uniform across the block */
    unsigned o = cnt[tid] - n;
    for (unsigned i = lo; i < hi && o < cnt[tid]; i++) {
        const float e = expf((__bfloat162float(row[i]) - m) * inv_t);
        if (e >= c) { c_idx[o] = i; c_e[o] = e; o++; }
    }
    __syncthreads();

    /* Largest floor whose kept mass exceeds top_p * total (the legacy bisection, over N). */
    const float want = tp * total;
    float flo = 0.0f, fhi = 1.0f;
#pragma unroll 1
    for (int it = 0; it < 24; it++) {
        const float mid = 0.5f * (flo + fhi);
        float ms = 0.0f;
        for (unsigned k = tid; k < N; k += T) ms += c_e[k] >= mid ? c_e[k] : 0.0f;
        if (block_sum(ms, part) > want) flo = mid; else fhi = mid;
    }
    const float floor = fmaxf(flo, mp);

    /* Inverse CDF in index order: thread tid owns candidates [tid*q, tid*q+q). */
    const unsigned q = (N + T - 1u) / T;
    const unsigned k0 = min(N, tid * q), k1 = min(N, k0 + q);
    float mine = 0.0f;
    for (unsigned k = k0; k < k1; k++) mine += c_e[k] >= floor ? c_e[k] : 0.0f;
    pre[tid] = mine;
    if (tid == 0) sh_pick = 0xFFFFFFFFu;
    __syncthreads();
    for (unsigned d = 1; d < T; d <<= 1) {
        const float add = tid >= d ? pre[tid - d] : 0.0f;
        __syncthreads();
        pre[tid] += add;
        __syncthreads();
    }
    const float target = u * pre[T - 1u];
    const float excl = pre[tid] - mine;
    if (mine > 0.0f && target >= excl && target < excl + mine) {
        float acc = excl;
        for (unsigned k = k0; k < k1; k++) {
            if (c_e[k] >= floor) {
                acc += c_e[k];
                if (acc > target) { sh_pick = c_idx[k]; break; }
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        unsigned p = sh_pick;
        if (p == 0xFFFFFFFFu) /* rng01 rounded onto the mass edge: lowest-index kept token */
            for (unsigned k = 0; k < N; k++) if (c_e[k] >= floor) { p = c_idx[k]; break; }
        *out = (int)p;
    }
    return true;
}

/* Device run-state advance (plan stage 5: bounded multi-step). Between two
 * decode launches on the engine stream, capture each active row's just-written
 * token (in.ids[b], from ARGMAX_FIN or plow_sample) into the token ring and
 * advance that row's device-owned position/kv-length by one — so the next
 * decode launch reads the advanced pos with NO host round trip. One thread per
 * row; idle rows (fed[b]==0) are untouched (their pos must not drift). The
 * decode program must derive its KV write row from in.pos (dynamic-kvrow
 * cubin) for this to be correct at B==1. */
extern "C" __global__ void plow_advance(
    const int* __restrict__ ids, int* __restrict__ pos, int* __restrict__ kvlen,
    int* __restrict__ ring, const int* __restrict__ fed,
    unsigned step, unsigned K, unsigned B) {
    const unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || !fed[b]) return;
    ring[(size_t)b * K + step] = ids[b];
    pos[b] += 1;
    kvlen[b] += 1;
}

extern "C" __global__ void plow_sample(
    const __nv_bfloat16* __restrict__ logits, int* __restrict__ out_ids,
    const float* __restrict__ temp, const int* __restrict__ top_k,
    const float* __restrict__ top_p, const float* __restrict__ min_p,
    const float* __restrict__ rng01, float* __restrict__ escratch, unsigned V, unsigned B) {
    const unsigned b = blockIdx.x;
    if (b >= B) return;
    __shared__ float part[PLOW_SMP_WARPS];
    __shared__ unsigned long long ipart[PLOW_SMP_WARPS];
    __shared__ float sh_floor, sh_target;
    __shared__ unsigned sh_pick;

    const __nv_bfloat16* row = logits + (size_t)b * V;
    const float t = temp[b];

    /* Greedy: argmax with the ARGMAX packed-key tie-break (lowest index wins). */
    if (t <= 1e-6f) {
        unsigned long long best = 0;
        for (unsigned i = threadIdx.x; i < V; i += PLOW_SMP_THREADS) {
            const unsigned short bits = *(const unsigned short*)&row[i];
            const unsigned key = (bits & 0x8000u) ? (unsigned)(unsigned short)~bits : (unsigned)(bits | 0x8000u);
            const unsigned long long p = ((unsigned long long)key << 32) | (unsigned long long)(~i);
            best = p > best ? p : best;
        }
        const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { unsigned long long x = __shfl_xor_sync(0xffffffffu, best, o, 32); best = x > best ? x : best; }
        if (lane == 0) ipart[warp] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long r = ipart[0];
#pragma unroll
            for (unsigned w = 1; w < PLOW_SMP_WARPS; w++) r = ipart[w] > r ? ipart[w] : r;
            out_ids[b] = (int)~(unsigned)(r & 0xFFFFFFFFull);
        }
        return;
    }

    if (top_k[b] <= 0 && top_p[b] < 1.0f &&
        sample_fast(row, V, t, top_p[b], min_p[b], rng01[b], out_ids + b))
        return;

    /* Weights e_i = exp((l_i - lmax)/t), materialised to scratch (reused by every pass). */
    const float inv_t = 1.0f / t;
    float* e = escratch + (size_t)b * V;
    float lmax = -3.4e38f;
    for (unsigned i = threadIdx.x; i < V; i += PLOW_SMP_THREADS) {
        const float l = __bfloat162float(row[i]);
        lmax = l > lmax ? l : lmax;
    }
    lmax = block_max(lmax, part);
    float total = 0.0f;
    for (unsigned i = threadIdx.x; i < V; i += PLOW_SMP_THREADS) {
        /* Accurate expf, not __expf: the top_p/min_p thresholds are ratios over
         * the whole vocab, and the fast intrinsic's ~2^-14 relative error visibly
         * distorts a broad nucleus (measured ~0.12 TVD at top_p=0.95). One expf
         * per token per stochastic row is negligible against the decode step. */
        const float w = expf((__bfloat162float(row[i]) - lmax) * inv_t);
        e[i] = w;
        total += w;
    }
    total = block_sum(total, part);

    /* Compose the weight floor from the three truncations (max of the three). */
    float floor = min_p[b]; /* min_p: e_i >= min_p (weights are already relative to max=1) */

    const int k = top_k[b];
    if (k > 0) {
        /* Largest floor whose count is still >= k: bisect in [0,1]. count is monotone
         * non-increasing in floor, so the k-th largest weight is the target. */
        float lo = 0.0f, hi = 1.0f;
#pragma unroll 1
        for (int it = 0; it < 24; it++) {
            const float mid = 0.5f * (lo + hi);
            const float c = count_ge(e, V, mid, part);
            if (c > (float)k) lo = mid; else hi = mid; /* too many kept -> raise floor */
        }
        floor = floor > lo ? floor : lo;
    }

    const float tp = top_p[b];
    if (tp < 1.0f) {
        /* Largest floor whose kept mass still covers top_p*total: bisect. mass is monotone
         * non-increasing in floor. */
        const float want = tp * total;
        float lo = 0.0f, hi = 1.0f;
#pragma unroll 1
        for (int it = 0; it < 24; it++) {
            const float mid = 0.5f * (lo + hi);
            const float m = mass_ge(e, V, mid, part);
            if (m > want) lo = mid; else hi = mid; /* still enough mass -> raise floor */
        }
        floor = floor > lo ? floor : lo;
    }
    if (threadIdx.x == 0) sh_floor = floor;
    __syncthreads();
    floor = sh_floor;

    /* Kept mass and the inverse-CDF target. */
    const float keptmass = mass_ge(e, V, floor, part);
    if (threadIdx.x == 0) { sh_target = rng01[b] * keptmass; sh_pick = 0xFFFFFFFFu; }
    __syncthreads();

    /* Index-order inverse-CDF via a block scan: each thread sums the kept weights in its
     * strided slice; an exclusive prefix over threads locates the slice holding `target`;
     * that thread walks its slice serially to the exact token. */
    float slice = 0.0f;
    for (unsigned i = threadIdx.x; i < V; i += PLOW_SMP_THREADS)
        if (e[i] >= floor) slice += e[i];
    /* Exclusive prefix of `slice` across the block (Hillis-Steele over 256 via smem). */
    __shared__ float pre[PLOW_SMP_THREADS];
    pre[threadIdx.x] = slice;
    __syncthreads();
    float excl = 0.0f;
    for (unsigned d = 1; d < PLOW_SMP_THREADS; d <<= 1) {
        const float add = (threadIdx.x >= d) ? pre[threadIdx.x - d] : 0.0f;
        __syncthreads();
        pre[threadIdx.x] += add;
        __syncthreads();
    }
    excl = pre[threadIdx.x] - slice; /* exclusive prefix = inclusive - own */
    const float target = sh_target;
    /* The owning thread is the one whose [excl, excl+slice) straddles target. */
    if (target >= excl && target < excl + slice) {
        float acc = excl;
        unsigned pick = 0xFFFFFFFFu;
        for (unsigned i = threadIdx.x; i < V; i += PLOW_SMP_THREADS) {
            if (e[i] >= floor) {
                acc += e[i];
                if (acc > target) { pick = i; break; }
            }
        }
        if (pick != 0xFFFFFFFFu) sh_pick = pick;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned p = sh_pick;
        if (p == 0xFFFFFFFFu) {
            /* rng01 rounded to the mass edge: fall back to the highest-weight kept token. */
            for (unsigned i = 0; i < V; i++) if (e[i] >= floor) { p = i; break; }
        }
        out_ids[b] = (int)p;
    }
}
