p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/runtime/nvidia/sample_sm120.cu'
s = open(p).read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


rep("""#ifndef PLOW_SMP_THREADS
#define PLOW_SMP_THREADS 256
#endif
#define PLOW_SMP_WARPS (PLOW_SMP_THREADS / 32)""", """#ifndef PLOW_SMP_THREADS
#define PLOW_SMP_THREADS 1024
#endif
#define PLOW_SMP_WARPS (PLOW_SMP_THREADS / 32)
/* The block width this object was built for; the engine launches with it (absent => 256). */
extern "C" __device__ unsigned plow_sample_threads = PLOW_SMP_THREADS;
/* Candidate capacity of the fast path (top_p/min_p rows, no top_k). */
#ifndef PLOW_SMP_CAND
#define PLOW_SMP_CAND 4096
#endif""")

rep("""/* Device run-state advance""", """/* FAST PATH for rows with top_p < 1 and no top_k (the TTS/chat default).
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

/* Device run-state advance""")

rep("""    /* Weights e_i = exp((l_i - lmax)/t), materialised to scratch (reused by every pass). */""",
    """    if (top_k[b] <= 0 && top_p[b] < 1.0f &&
        sample_fast(row, V, t, top_p[b], min_p[b], rng01[b], out_ids + b))
        return;

    /* Weights e_i = exp((l_i - lmax)/t), materialised to scratch (reused by every pass). */""")
open(p, 'w').write(s)
print("ok")
