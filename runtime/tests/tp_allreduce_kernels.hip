/* tp_allreduce_kernels.hip — device code for the one-shot all-reduce microbench.
 *
 * Validates op_collective.h's d_xreduce_oneshot END-TO-END across two real GPUs:
 * the publish-signal, the SYSTEM-scope xctr gate, and the f32-accumulate reduction,
 * exactly as the persistent megakernel's XREDUCE packet will run it. This is the
 * standalone correctness+latency proof for the runtime collective (P1-B), reusing
 * the transport primitives tp-transport already proved.
 *
 * The kernels call op_collective.h directly, so the bench validates THE op code, not
 * a copy — the same rule every op_*.h golden test follows.
 */
#include "../amd/op_collective.h"

/* Fill this rank's partial slot with a checkable element-varying bf16 pattern.
 * val(r,e) = (r+1) * (1 + (e&7)*0.125) — all exact in bf16, and it varies per
 * element so a mis-indexed peer read or a partial write is caught, not just a
 * uniform 1+2=3. */
extern "C" __global__ void tp_fill_partial(bf16* part, uint32_t n, uint32_t rank) {
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += gridDim.x * blockDim.x)
        part[e] = f2bf((float)(rank + 1) * (1.0f + (float)(e & 7u) * 0.125f));
}

/* One-shot all-reduce, timed. Loops `iters` collectives so the per-collective
 * latency is (cycles / iters); iteration i uses gate_target = i*nranks because each
 * rank's workgroup-0 bumps every peer gate once per iteration (nranks arrivals/iter).
 * All workgroups wait + reduce; only workgroup 0 signals. */
extern "C" __global__ void tp_allreduce(bf16* out, const void* const* peer_scratch,
                                        uint32_t* const* peer_gate, uint32_t nranks,
                                        uint32_t rank, uint32_t n, uint32_t slot_bytes,
                                        uint32_t iters, uint64_t deadline,
                                        uint64_t* cycles, uint32_t* status) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 1; i <= iters; i++) {
        d_xreduce_oneshot(out, peer_scratch, peer_gate, nranks, rank, n, slot_bytes,
                          i * nranks, blockIdx.x == 0, deadline, status,
                          blockIdx.x, gridDim.x);
    }
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    if (blockIdx.x == 0 && threadIdx.x == 0 && cycles) *cycles = t1 - t0;
}
