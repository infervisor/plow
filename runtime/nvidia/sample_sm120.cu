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
#define PLOW_SMP_THREADS 256
#endif
#define PLOW_SMP_WARPS (PLOW_SMP_THREADS / 32)

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
