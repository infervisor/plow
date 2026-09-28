/* sample_sm120.cu — device-side stochastic sampler. ONE 256-thread block per
 * batch row samples that row's next token from its
 * `[V]` bf16 logits and writes it where the next decode step's EMBED reads it (`in.ids[b]`),
 * exactly as ARGMAX_FIN does for greedy — so `temperature > 0` no longer downloads the whole
 * vocabulary row to the host, and the host softmax+full-vocab sort leaves the critical path.
 *
 * SEMANTICS: the host sampler's (`text::sample::sample_with_scratch`). Given per-row temperature
 * t, top_k, top_p, min_p and a uniform rng01 in [0,1), with p_i = softmax(l_i / t):
 *   top_k  keeps the k largest (ties at the k-th value: lowest index first)
 *   min_p  keeps p_i >= min_p * p_max
 *   top_p  keeps the shortest descending prefix whose mass reaches top_p (crossing token kept)
 *   draw   inverse-CDF over the kept set in descending order, target = rng01 * kept mass
 * `sample_select` implements it for every row with a truncation. Rows it cannot hold (more than
 * PLOW_SMP_CAND candidates) fall back to the legacy threshold path
 * below: one weight floor from bisections, index-order draw — the same distribution up to the
 * top_p boundary token. t <= 0 is greedy argmax with the ARGMAX tie-break (lowest index wins),
 * byte-identical to d_argmax_fin.
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

/* Ordered 16-bit key of a bf16 value: a larger key is a larger value (the ARGMAX key). */
__device__ __forceinline__ unsigned bf16_key(unsigned short bits) {
    return (bits & 0x8000u) ? (unsigned)(unsigned short)~bits : (unsigned)(bits | 0x8000u);
}

/* Row element j of a uint4 (8 bf16) word. */
__device__ __forceinline__ unsigned short bf16_lane(const uint4& q, unsigned j) {
    const unsigned w = j < 2u ? q.x : j < 4u ? q.y : j < 6u ? q.z : q.w;
    return (unsigned short)(j & 1u ? w >> 16 : w & 0xffffu);
}

/* Warp-aggregated shared histogram increment; `bin` >= 256 means no element. */
__device__ __forceinline__ void hist_add(unsigned* hist, unsigned bin) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned peers = __match_any_sync(0xffffffffu, bin);
    if (bin < 256u && lane == (unsigned)(__ffs(peers) - 1)) atomicAdd(&hist[bin], (unsigned)__popc(peers));
}

/* The highest bin whose count, summed from bin 255 downwards, reaches `need` (warp 0 only).
 * Returns (bin, count strictly above it) through `sel`. */
__device__ __forceinline__ void hist_pick(const unsigned* hist, unsigned need, unsigned* sel) {
    const unsigned lane = threadIdx.x & 31u;
    unsigned mine = 0;
#pragma unroll
    for (unsigned j = 0; j < 8u; j++) mine += hist[255u - lane * 8u - j];
    unsigned incl = mine;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned x = __shfl_up_sync(0xffffffffu, incl, o, 32);
        if ((int)lane >= o) incl += x;
    }
    const unsigned hit = __ballot_sync(0xffffffffu, incl >= need);
    const unsigned first = hit ? (unsigned)(__ffs(hit) - 1) : 31u;
    if (lane == first) {
        unsigned above = incl - mine;
        unsigned bin = 255u - lane * 8u;
        for (unsigned j = 0; j < 8u; j++, bin--) {
            if (above + hist[bin] >= need || j == 7u) break;
            above += hist[bin];
        }
        sel[0] = bin;
        sel[1] = above;
    }
}

/* Every element of the row once per thread-uniform trip: f(live, index, bf16 bits). 16-byte
 * vector loads when the row allows them (V % 8 == 0 and an aligned base), else coalesced scalar
 * loads. The trip count is the same for every thread, so f may use warp collectives. */
template <typename F>
__device__ __forceinline__ void row_walk(const __nv_bfloat16* __restrict__ row, unsigned V, F&& f) {
    const unsigned T = PLOW_SMP_THREADS, tid = threadIdx.x;
    if ((V & 7u) == 0u && (reinterpret_cast<size_t>(row) & 15u) == 0u) {
        const uint4* q = reinterpret_cast<const uint4*>(row);
        const unsigned words = V / 8u, iters = (words + T - 1u) / T;
        for (unsigned it = 0; it < iters; it++) {
            const unsigned w = it * T + tid;
            const bool live = w < words;
            const uint4 v = live ? q[w] : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (unsigned j = 0; j < 8u; j++) f(live, w * 8u + j, bf16_lane(v, j));
        }
    } else {
        const unsigned short* r = reinterpret_cast<const unsigned short*>(row);
        const unsigned iters = (V + T - 1u) / T;
        for (unsigned it = 0; it < iters; it++) {
            const unsigned i = it * T + tid;
            const bool live = i < V;
            f(live, i, live ? r[i] : (unsigned short)0u);
        }
    }
}

/* In warp 0: the first j < n whose inclusive prefix of w[0..] reaches `target` (n if none), and
 * the prefix sum of w[0..n) through `sum`. */
__device__ __forceinline__ unsigned warp_first_ge(const float* w, unsigned n, float target, float* sum) {
    const unsigned lane = threadIdx.x & 31u;
    float acc = 0.0f;
    unsigned found = n;
    for (unsigned base = 0; base < n; base += 32u) {
        const unsigned j = base + lane;
        float incl = j < n ? w[j] : 0.0f;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float x = __shfl_up_sync(0xffffffffu, incl, o, 32);
            if ((int)lane >= o) incl += x;
        }
        incl += acc;
        const unsigned hit = __ballot_sync(0xffffffffu, j < n && incl >= target);
        if (hit && found == n) found = base + (unsigned)(__ffs(hit) - 1);
        acc = __shfl_sync(0xffffffffu, incl, 31);
    }
    *sum = acc;
    return found;
}

/* The host sampler's semantics (`text::sample::sample_with_scratch`) on device, for rows with at
 * least one truncation. With p_i = softmax(l / t) over the whole vocabulary:
 *   top_k  keeps the k largest (ties at the k-th value: lowest index first),
 *   min_p  keeps p_i >= min_p * p_max,
 *   top_p  keeps the shortest descending prefix whose mass reaches top_p (the crossing token
 *          included),
 *   draw   inverse CDF over the kept set in DESCENDING order, target = rng01 * kept mass.
 * Candidates are found without a vocabulary-sized scratch: top_k by a radix select on the bf16
 * keys (two 256-bin histograms), otherwise by the exact filter e_i >= max(eps, min_p) with
 * eps = (1 - top_p) * total / V (every weight below eps sums to at most (1 - top_p) * total, so the
 * descending prefix reaches top_p * total before any of them). Three streaming passes over the
 * bf16 row, then a bitonic sort of the <= PLOW_SMP_CAND candidates in shared memory.
 * Returns false (nothing written) when the candidates overflow. */
__device__ bool sample_select(const __nv_bfloat16* __restrict__ row, unsigned V, float t, int k,
                              float tp, float mp, float u, int* __restrict__ out) {
    __shared__ unsigned c_idx[PLOW_SMP_CAND];
    __shared__ float c_w[PLOW_SMP_CAND];
    __shared__ unsigned hist[256];
    __shared__ float part[PLOW_SMP_WARPS];
    __shared__ unsigned sel[2];
    __shared__ unsigned n_cand;
    __shared__ int pick;
    const unsigned T = PLOW_SMP_THREADS, tid = threadIdx.x;
    if (k > PLOW_SMP_CAND) return false;
    const float inv_t = 1.0f / t;
    const bool by_rank = k > 0;
    if (tid < 256u) hist[tid] = 0u;
    if (tid == 0) n_cand = 0u;
    __syncthreads();

    /* Pass 1: max, online softmax mass, and the high-byte histogram of the keys. */
    float m = -3.4e38f, s = 0.0f;
    row_walk(row, V, [&](bool live, unsigned, unsigned short bits) {
        if (live) {
            const float l = __bfloat162float(__ushort_as_bfloat16(bits));
            if (l > m) { s = s * expf((m - l) * inv_t) + 1.0f; m = l; }
            else s += expf((l - m) * inv_t);
        }
        if (by_rank) hist_add(hist, live ? bf16_key(bits) >> 8 : 256u);
    });
    const float mx = block_max(m, part);
    const float total = block_sum(m > -3.4e38f ? s * expf((m - mx) * inv_t) : 0.0f, part);

    /* Candidate bound: the k-th largest key (low byte by a second pass), or a weight floor. */
    unsigned kkey = 0u;
    float wfloor = 0.0f;
    if (by_rank) {
        if (tid < 32u) hist_pick(hist, (unsigned)k, sel);
        __syncthreads();
        const unsigned hi = sel[0], need = (unsigned)k - sel[1];
        __syncthreads();
        if (tid < 256u) hist[tid] = 0u;
        __syncthreads();
        row_walk(row, V, [&](bool live, unsigned, unsigned short bits) {
            const unsigned key = bf16_key(bits);
            hist_add(hist, live && (key >> 8) == hi ? (key & 255u) : 256u);
        });
        __syncthreads();
        if (tid < 32u) hist_pick(hist, need, sel);
        __syncthreads();
        kkey = (hi << 8) | sel[0];
    } else {
        wfloor = fmaxf(tp < 1.0f ? (1.0f - tp) * total / (float)V : 0.0f, mp);
    }

    /* Pass 3: compact the candidates (any order; the sort below fixes it). */
    row_walk(row, V, [&](bool live, unsigned i, unsigned short bits) {
        const float e = expf((__bfloat162float(__ushort_as_bfloat16(bits)) - mx) * inv_t);
        const bool keep = live && (by_rank ? bf16_key(bits) >= kkey : e >= wfloor);
        const unsigned ballot = __ballot_sync(0xffffffffu, keep);
        if (!ballot) return;
        const unsigned lane = tid & 31u;
        unsigned base = 0u;
        if (lane == (unsigned)(__ffs(ballot) - 1)) base = atomicAdd(&n_cand, (unsigned)__popc(ballot));
        base = __shfl_sync(0xffffffffu, base, __ffs(ballot) - 1);
        const unsigned slot = base + __popc(ballot & ((1u << lane) - 1u));
        if (keep && slot < PLOW_SMP_CAND) { c_idx[slot] = i; c_w[slot] = e; }
    });
    __syncthreads();
    unsigned N = n_cand;
    if (N == 0u || N > PLOW_SMP_CAND) return false; /* uniform across the block */

    /* A wide nucleus: narrow the candidates to the top_p/min_p kept set before sorting. The
     * bisected floor sits just below the crossing token's weight (the largest floor whose kept
     * mass still exceeds top_p * total), so the set keeps every token the sorted cut below can
     * keep; the cut then fixes the exact boundary. */
    if (!by_rank && N > 64u && (tp < 1.0f || mp > 0.0f)) {
        float flo = 0.0f;
        if (tp < 1.0f) {
            const float want = tp * total;
            float fhi = 1.0f;
#pragma unroll 1
            for (int it = 0; it < 24; it++) {
                const float mid = 0.5f * (flo + fhi);
                float ms = 0.0f;
                for (unsigned i = tid; i < N; i += T) ms += c_w[i] >= mid ? c_w[i] : 0.0f;
                if (block_sum(ms, part) > want) flo = mid; else fhi = mid;
            }
        }
        const float keep_floor = fmaxf(flo, mp);
        constexpr unsigned PER = (PLOW_SMP_CAND + PLOW_SMP_THREADS - 1) / PLOW_SMP_THREADS;
        float rw[PER];
        unsigned ri[PER];
#pragma unroll
        for (unsigned k = 0; k < PER; k++) {
            const unsigned i = tid + k * T;
            rw[k] = i < N ? c_w[i] : -1.0f;
            ri[k] = i < N ? c_idx[i] : 0u;
        }
        if (tid == 0) n_cand = 0u;
        __syncthreads();
#pragma unroll
        for (unsigned k = 0; k < PER; k++) {
            const bool keep = rw[k] >= keep_floor;
            const unsigned ballot = __ballot_sync(0xffffffffu, keep);
            if (!ballot) continue;
            const unsigned lane = tid & 31u;
            unsigned base = 0u;
            if (lane == (unsigned)(__ffs(ballot) - 1)) base = atomicAdd(&n_cand, (unsigned)__popc(ballot));
            base = __shfl_sync(0xffffffffu, base, __ffs(ballot) - 1);
            if (keep) {
                const unsigned slot = base + __popc(ballot & ((1u << lane) - 1u));
                c_w[slot] = rw[k];
                c_idx[slot] = ri[k];
            }
        }
        __syncthreads();
        N = n_cand;
    }

    /* Sort descending by weight, ascending index on ties (the rank order top_k truncates). */
    unsigned n2 = 1u;
    while (n2 < N) n2 <<= 1;
    for (unsigned i = N + tid; i < n2; i += T) { c_w[i] = -1.0f; c_idx[i] = 0xFFFFFFFFu; }
    __syncthreads();
    for (unsigned size = 2u; size <= n2; size <<= 1) {
        for (unsigned stride = size >> 1; stride > 0u; stride >>= 1) {
            for (unsigned i = tid; i < n2; i += T) {
                const unsigned j = i ^ stride;
                if (j <= i) continue;
                const float wi = c_w[i], wj = c_w[j];
                const unsigned ii = c_idx[i], ij = c_idx[j];
                const bool j_first = wj > wi || (wj == wi && ij < ii);
                const bool i_first = wi > wj || (wi == wj && ii < ij);
                if ((i & size) == 0u ? j_first : i_first) {
                    c_w[i] = wj; c_w[j] = wi; c_idx[i] = ij; c_idx[j] = ii;
                }
            }
            __syncthreads();
        }
    }

    /* Truncate (top_k, min_p on a descending list are prefixes; then top_p), then draw. */
    if (tid < 32u) {
        unsigned n = by_rank ? min(N, (unsigned)k) : N;
        if (mp > 0.0f) {
            unsigned cut = n;
            for (unsigned base = 0; base < n && cut == n; base += 32u) {
                const unsigned j = base + (tid & 31u);
                const unsigned hit = __ballot_sync(0xffffffffu, j < n && c_w[j] < mp);
                if (hit) cut = base + (unsigned)(__ffs(hit) - 1);
            }
            n = max(cut, 1u);
        }
        float kept;
        if (tp < 1.0f) {
            const unsigned j = warp_first_ge(c_w, n, tp * total, &kept);
            n = min(n, j + 1u);
        }
        warp_first_ge(c_w, n, 0.0f, &kept);
        const unsigned j = warp_first_ge(c_w, n, u * kept, &kept);
        if (tid == 0) pick = (int)c_idx[min(j, n - 1u)];
    }
    __syncthreads();
    if (tid == 0) *out = pick;
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

    if ((top_k[b] > 0 || top_p[b] < 1.0f || min_p[b] > 0.0f) &&
        sample_select(row, V, t, top_k[b], top_p[b], min_p[b], rng01[b], out_ids + b))
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

/* Classifier-free guided draw for CFG pairs: conditional row b, unconditional row b + 1, the
 * host guided sampler (`text::sample::sample_cfg`) on device, so a pair decodes with no host
 * round trip (and rides a multi-step quantum).
 *   g_i = c_i + w (c_i - u_i)            (explicit _rn ops: no FMA contraction, the host rounding)
 *   repetition penalty once if i occurs in the row's history (counts[b][i] > 0), as HF's
 *     RepetitionPenaltyLogitsProcessor: g < 0 ? g * p : g / p
 *   e_i = exp((g_i - max g) / t); e_i < min_p -> 0
 *   top_p < 1: the shortest descending (weight, index) prefix of the min_p-kept set whose
 *     mass reaches top_p * kept (the host's sort, as an exact threshold + index-order ties)
 *   draw: the lowest index whose inclusive prefix of kept weights exceeds rng01 * kept
 * t <= 0 is the host's greedy: argmax of g before the penalty, lowest index on ties.
 * The token goes to BOTH members' in.ids and is counted into the owner's history row.
 * prm is [6][B] f32: flag (nonzero = owner), w, penalty, t, top_p, min_p. */
__device__ __forceinline__ unsigned long long argmax_key(float v, unsigned i) {
    const unsigned bits = __float_as_uint(v);
    const unsigned key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return ((unsigned long long)key << 32) | (unsigned long long)(~i);
}

__device__ __forceinline__ unsigned long long block_max_u64(unsigned long long v, unsigned long long* ipart) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long x = __shfl_xor_sync(0xffffffffu, v, o, 32);
        v = x > v ? x : v;
    }
    if (lane == 0) ipart[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long r = ipart[0];
#pragma unroll
        for (unsigned w = 1; w < PLOW_SMP_WARPS; w++) r = ipart[w] > r ? ipart[w] : r;
        ipart[0] = r;
    }
    __syncthreads();
    const unsigned long long r = ipart[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void plow_sample_cfg(
    const __nv_bfloat16* __restrict__ logits, int* __restrict__ ids, const float* __restrict__ prm,
    const float* __restrict__ rng01, unsigned* __restrict__ counts, float* __restrict__ escratch,
    unsigned V, unsigned B) {
    const unsigned b = blockIdx.x;
    if (b + 1u >= B || prm[b] == 0.0f) return;
    __shared__ float part[PLOW_SMP_WARPS];
    __shared__ unsigned long long ipart[PLOW_SMP_WARPS];
    __shared__ float pre[PLOW_SMP_THREADS];
    __shared__ unsigned sh_pick;
    const float w = prm[B + b], pen = prm[2u * B + b], t = prm[3u * B + b];
    const float tp = prm[4u * B + b], mp = prm[5u * B + b];
    const __nv_bfloat16* c = logits + (size_t)b * V;
    const __nv_bfloat16* u = c + V;
    unsigned* cnt = counts + (size_t)b * V;
    float* e = escratch + (size_t)b * V;
    const unsigned T = PLOW_SMP_THREADS, tid = threadIdx.x;
    const unsigned chunk = (V + T - 1u) / T;
    const unsigned lo = min(V, tid * chunk), hi = min(V, lo + chunk);
    auto guided = [&](unsigned i) {
        const float a = __bfloat162float(c[i]), z = __bfloat162float(u[i]);
        return __fadd_rn(a, __fmul_rn(w, __fsub_rn(a, z)));
    };
    if (tid == 0) sh_pick = 0xFFFFFFFFu;

    if (t <= 1e-6f) {
        unsigned long long best = 0;
        for (unsigned i = lo; i < hi; i++) {
            const unsigned long long k = argmax_key(guided(i), i);
            best = k > best ? k : best;
        }
        best = block_max_u64(best, ipart);
        if (tid == 0) sh_pick = ~(unsigned)(best & 0xFFFFFFFFull);
    } else {
        float m = -3.4e38f;
        for (unsigned i = lo; i < hi; i++) {
            float g = guided(i);
            if (cnt[i] > 0u) g = g < 0.0f ? __fmul_rn(g, pen) : __fdiv_rn(g, pen);
            e[i] = g;
            m = fmaxf(m, g);
        }
        m = block_max(m, part);
        const float inv_t = __fdiv_rn(1.0f, t);
        float s = 0.0f;
        for (unsigned i = lo; i < hi; i++) {
            float x = expf(__fmul_rn(__fsub_rn(e[i], m), inv_t));
            x = x < mp ? 0.0f : x;
            e[i] = x;
            s += x;
        }
        const float kept = block_sum(s, part);
        if (tp < 1.0f) {
            /* The host keeps the shortest prefix of the descending (weight, index) order whose
             * mass reaches want: its smallest weight v is the largest float with
             * mass(e >= v) >= want (bisected on the bit pattern, exact), weights above v stay,
             * and of the weights equal to v the lowest-index n do. */
            const float want = tp * kept;
            unsigned blo = 0u, bhi = 0x3F800001u; /* mass(e >= 0) = kept >= want; nothing > 1 */
#pragma unroll 1
            while (bhi - blo > 1u) {
                const unsigned mid = blo + ((bhi - blo) >> 1);
                if (mass_ge(e, V, __uint_as_float(mid), part) >= want) blo = mid; else bhi = mid;
            }
            const float v = __uint_as_float(blo);
            float above = 0.0f, ties = 0.0f;
            for (unsigned i = lo; i < hi; i++) {
                above += e[i] > v ? e[i] : 0.0f;
                ties += e[i] == v ? 1.0f : 0.0f;
            }
            const float need = fmaxf(1.0f, ceilf((want - block_sum(above, part)) / v));
            pre[tid] = ties;
            __syncthreads();
            for (unsigned d = 1; d < T; d <<= 1) {
                const float add = tid >= d ? pre[tid - d] : 0.0f;
                __syncthreads();
                pre[tid] += add;
                __syncthreads();
            }
            float rank = pre[tid] - ties;
            __syncthreads();
            for (unsigned i = lo; i < hi; i++) {
                if (e[i] == v) {
                    if (rank >= need) e[i] = 0.0f;
                    rank += 1.0f;
                } else if (e[i] < v) {
                    e[i] = 0.0f;
                }
            }
        }
        /* Inverse CDF in index order: thread tid owns [lo, hi); exclusive prefix of the chunk
         * sums by warp shuffles plus one pass over the warp totals. */
        float mine = 0.0f;
        for (unsigned i = lo; i < hi; i++) mine += e[i];
        const unsigned lane = tid & 31u, warp = tid >> 5;
        float incl = mine;
#pragma unroll
        for (unsigned d = 1; d < 32u; d <<= 1) {
            const float x = __shfl_up_sync(0xffffffffu, incl, d, 32);
            if (lane >= d) incl += x;
        }
        if (lane == 31u) pre[warp] = incl;
        __syncthreads();
        if (warp == 0) {
            float w_incl = lane < PLOW_SMP_WARPS ? pre[lane] : 0.0f;
#pragma unroll
            for (unsigned d = 1; d < 32u; d <<= 1) {
                const float x = __shfl_up_sync(0xffffffffu, w_incl, d, 32);
                if (lane >= d) w_incl += x;
            }
            if (lane < PLOW_SMP_WARPS) pre[32u + lane] = w_incl;
        }
        __syncthreads();
        const float total = pre[32u + PLOW_SMP_WARPS - 1u];
        const float target = rng01[b] * total;
        const float excl = (warp ? pre[32u + warp - 1u] : 0.0f) + incl - mine;
        if (mine > 0.0f && target >= excl && target < excl + mine) {
            float acc = excl;
            unsigned last = 0xFFFFFFFFu, pick = 0xFFFFFFFFu;
            for (unsigned i = lo; i < hi; i++) {
                if (e[i] > 0.0f) {
                    last = i;
                    acc += e[i];
                    if (acc > target) { pick = i; break; }
                }
            }
            sh_pick = pick != 0xFFFFFFFFu ? pick : last; /* chunk-sum rounding: its last kept */
        }
        __syncthreads();
        if (sh_pick == 0xFFFFFFFFu) { /* rng01 * kept rounded onto the mass edge: the host's argmax */
            unsigned long long best = 0;
            for (unsigned i = lo; i < hi; i++) {
                const unsigned long long k = argmax_key(e[i], i);
                best = k > best ? k : best;
            }
            best = block_max_u64(best, ipart);
            if (tid == 0) sh_pick = ~(unsigned)(best & 0xFFFFFFFFull);
        }
    }
    __syncthreads();
    if (tid == 0) {
        const unsigned p = sh_pick;
        ids[b] = (int)p;
        ids[b + 1u] = (int)p;
        cnt[p] += 1u;
    }
}

/* OpenAI logprobs for one logits row (`text::logprobs::RowStats` on device), so a greedy row that
 * asks for them downloads 43 floats instead of its whole vocabulary row.
 *   out[0] = logsumexp(row)   out[1] = row[tok]   out[2] = 1 when the top-k may be inexact
 *   out[3 .. 3+k] = top-k ids (u32 bits), out[3+k .. 3+2k] = their logits; best first, ties to
 *   the lower id (the host order).
 * Each thread keeps its 4 best entries; the block then pops the global best k times. A thread
 * whose 4 entries all make the top-k might have held a fifth: out[2] = 1 and the caller falls
 * back to the host row. */
#define PLOW_LP_LOCAL 4
__device__ __forceinline__ unsigned long long lp_key(float v, unsigned i) {
    const unsigned u = __float_as_uint(v);
    const unsigned o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)o << 32) | (unsigned long long)(~i);
}
__device__ __forceinline__ void lp_stats_row(
    const __nv_bfloat16* __restrict__ logits, float* __restrict__ out, unsigned row, unsigned tok,
    unsigned V, unsigned k) {
    __shared__ float part[PLOW_SMP_WARPS];
    __shared__ unsigned long long kpart[PLOW_SMP_WARPS];
    __shared__ unsigned long long sh_win;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const __nv_bfloat16* r = logits + (size_t)row * V;
    float lv[PLOW_LP_LOCAL];
    unsigned li[PLOW_LP_LOCAL];
#pragma unroll
    for (int j = 0; j < PLOW_LP_LOCAL; j++) { lv[j] = -INFINITY; li[j] = 0xFFFFFFFFu; }
    float m = -INFINITY;
    for (unsigned i = tid; i < V; i += PLOW_SMP_THREADS) {
        const float x = __bfloat162float(r[i]);
        m = fmaxf(m, x);
        if (x > lv[PLOW_LP_LOCAL - 1]) {
            /* strict >: an equal later index stays behind the earlier one */
            int j = PLOW_LP_LOCAL - 1;
            while (j > 0 && x > lv[j - 1]) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; j--; }
            lv[j] = x; li[j] = i;
        }
    }
    m = block_max(m, part);
    float s = 0.0f;
    for (unsigned i = tid; i < V; i += PLOW_SMP_THREADS) s += expf(__bfloat162float(r[i]) - m);
    s = block_sum(s, part);
    int head = 0;
    bool inexact = false;
    for (unsigned n = 0; n < k; n++) {
        unsigned long long best = head < PLOW_LP_LOCAL && li[head] != 0xFFFFFFFFu ? lp_key(lv[head], li[head]) : 0ull;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const unsigned long long x = __shfl_xor_sync(0xffffffffu, best, o, 32);
            best = x > best ? x : best;
        }
        if (lane == 0) kpart[warp] = best;
        __syncthreads();
        if (tid == 0) {
            unsigned long long w = kpart[0];
            for (unsigned q = 1; q < PLOW_SMP_WARPS; q++) w = kpart[q] > w ? kpart[q] : w;
            sh_win = w;
        }
        __syncthreads();
        const unsigned long long w = sh_win;
        if (w != 0ull && head < PLOW_LP_LOCAL && lp_key(lv[head], li[head]) == w) {
            out[3 + n] = __uint_as_float(li[head]);
            out[3 + k + n] = lv[head];
            head++;
            inexact |= head == PLOW_LP_LOCAL && n + 1 < k;
        } else if (w == 0ull && tid == 0) {
            out[3 + n] = __uint_as_float(0xFFFFFFFFu);
            out[3 + k + n] = -INFINITY;
        }
        __syncthreads();
    }
    const int any_inexact = __syncthreads_or(inexact);
    if (tid == 0) {
        out[0] = m + logf(s);
        out[1] = __bfloat162float(r[tok]);
        out[2] = any_inexact ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void plow_logprob_stats(
    const __nv_bfloat16* __restrict__ logits, float* __restrict__ out, unsigned row, unsigned tok,
    unsigned V, unsigned k) {
    lp_stats_row(logits, out, row, tok, V, k);
}

/* One block per request: req[3i..3i+3] = (logits row, token, k); its stats land at
 * out + i * out_stride. One launch and one readback for every logprobs row of a step. With
 * `ids` the token is the row's sampled id on the device (ids[row]), not req's: a device
 * multi-step quantum can run this after each step's sampler without a host round trip. */
extern "C" __global__ void plow_logprob_stats_rows(
    const __nv_bfloat16* __restrict__ logits, float* __restrict__ out,
    const unsigned* __restrict__ req, const int* __restrict__ ids, unsigned V,
    unsigned out_stride) {
    const unsigned* q = req + 3u * blockIdx.x;
    const unsigned tok = ids ? (unsigned)ids[q[0]] : q[1];
    lp_stats_row(logits, out + (size_t)blockIdx.x * out_stride, q[0], tok, V, q[2]);
}
