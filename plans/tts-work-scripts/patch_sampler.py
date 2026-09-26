p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/runtime/nvidia/sample_sm120.cu'
s = open(p).read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


rep("""/* Device run-state advance""", """/* Largest weight floor v whose kept mass (sum of e_i >= v) still reaches `want`: an exact
 * radix select over the float bits of e (all e in (0,1], so the bits order like the values).
 * Three 10-bit levels (bits 29:20, 19:10, 9:0), each ONE pass over e with a shared-memory mass
 * histogram, replace the 24 whole-vocabulary reductions of a bisection. */
#define PLOW_SMP_BINS 1024
__device__ float top_p_floor(const float* e, unsigned V, float want) {
    __shared__ float hist[PLOW_SMP_BINS];
    __shared__ unsigned sh_prefix;
    __shared__ float sh_above;
    if (threadIdx.x == 0) { sh_prefix = 0u; sh_above = 0.0f; }
#pragma unroll 1
    for (int level = 0; level < 3; level++) {
        const unsigned shift = 20u - 10u * (unsigned)level;
        for (unsigned j = threadIdx.x; j < PLOW_SMP_BINS; j += PLOW_SMP_THREADS) hist[j] = 0.0f;
        __syncthreads();
        const unsigned prefix = sh_prefix;
        for (unsigned i = threadIdx.x; i < V; i += PLOW_SMP_THREADS) {
            const unsigned bits = __float_as_uint(e[i]);
            if (level == 0 || (bits >> (shift + 10u)) == prefix)
                atomicAdd(&hist[(bits >> shift) & (PLOW_SMP_BINS - 1u)], e[i]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float acc = sh_above;
            unsigned pick = 0u;
            for (int j = PLOW_SMP_BINS - 1; j >= 0; j--) {
                if (acc + hist[j] >= want) { pick = (unsigned)j; break; }
                acc += hist[j];
            }
            sh_above = acc;
            sh_prefix = (prefix << 10) | pick;
        }
        __syncthreads();
    }
    return __uint_as_float(sh_prefix);
}

/* Device run-state advance""")
rep("""        const float want = tp * total;
        float lo = 0.0f, hi = 1.0f;
#pragma unroll 1
        for (int it = 0; it < 24; it++) {
            const float mid = 0.5f * (lo + hi);
            const float m = mass_ge(e, V, mid, part);
            if (m > want) lo = mid; else hi = mid; /* still enough mass -> raise floor */
        }
        floor = floor > lo ? floor : lo;""", """        const float lo = top_p_floor(e, V, tp * total);
        floor = floor > lo ? floor : lo;""")
open(p, 'w').write(s)
print("ok")
