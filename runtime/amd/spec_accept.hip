/* spec_accept.hip — HIP twin of plow_spec_accept (runtime/nvidia/sample_sm120.cu): greedy
 * speculative-decode acceptance, one thread per verify span. Same body, same PlowSpecVerdict layout
 * (crates/plowrt/src/orch/speculative.rs `SpecVerdict`). Not yet wired into an AMD build. */
#include <hip/hip_runtime.h>

/* Contract plow.spec.v1.
 * One thread per verify span. A span is rows row0..row0+k of the verify launch; `inputs[row]` is the
 * token fed to that row (the pending token, then k drafts) and `argmax[row]` the token the target
 * produced there. Draft j survives while inputs[row0+j+1] == argmax[row0+j]; the emission is the
 * accepted drafts plus argmax[row0+n], cut after the first stop token. Mirrors the host golden
 * `orch::speculative::accept_greedy`; the verdict layout is `SpecVerdict` there. */
#define PLOW_SPEC_KMAX 15
typedef struct {
    unsigned req_gen;
    unsigned short n_acc;
    unsigned short n_emit;
    unsigned hrow;
    unsigned flags;
    unsigned tokens[PLOW_SPEC_KMAX + 1];
} PlowSpecVerdict;
typedef struct {
    unsigned row0;
    unsigned k;
    unsigned req_gen;
    unsigned _pad;
} PlowSpecSpan;

extern "C" __global__ void plow_spec_accept(const int* __restrict__ argmax,
                                            const int* __restrict__ inputs,
                                            const PlowSpecSpan* __restrict__ spans,
                                            unsigned n_spans, const int* __restrict__ stop,
                                            unsigned n_stop, PlowSpecVerdict* __restrict__ out) {
    const unsigned s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_spans) return;
    const PlowSpecSpan sp = spans[s];
    const unsigned k = sp.k > PLOW_SPEC_KMAX ? PLOW_SPEC_KMAX : sp.k;
    unsigned n = 0;
    while (n < k && inputs[sp.row0 + n + 1] == argmax[sp.row0 + n]) n++;
    PlowSpecVerdict v;
    v.req_gen = sp.req_gen;
    v.flags = 0;
    unsigned emit = 0;
    for (unsigned j = 0; j <= n; j++) {
        const unsigned t = (unsigned)(j < n ? inputs[sp.row0 + j + 1] : argmax[sp.row0 + n]);
        v.tokens[emit++] = t;
        bool is_stop = false;
        for (unsigned e = 0; e < n_stop; e++) is_stop |= (unsigned)stop[e] == t;
        if (is_stop) { v.flags |= 1u; break; }
    }
    for (unsigned j = emit; j <= PLOW_SPEC_KMAX; j++) v.tokens[j] = 0;
    v.n_emit = (unsigned short)emit;
    v.n_acc = (unsigned short)(emit < n ? emit : n);
    v.hrow = sp.row0 + emit - 1;
    out[s] = v;
}
