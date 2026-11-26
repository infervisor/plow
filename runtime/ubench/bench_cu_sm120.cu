/* bench_cu_sm120.cu — Plow μBench device side for sm_12x (GB10 = sm_121a, RTX 50xx = sm_120a).
 *
 * Includes the production interpreter TU, so `ubench_exec` runs the EXACT op bodies through the
 * interpreter's own per-instruction executor (`plow_exec`): same dispatch, same arms, same
 * inlining. Build it with the packet's generated config and the object's axis defines, exactly
 * as the served cubin (runtime/CMakeLists.txt PLOW_SM120_CUBIN rows), e.g. the row-slot decode
 * object:
 *
 *   nvcc -arch=sm_121a -O3 -cubin -I ../nvidia -I ../common -I <assets> \
 *        -DPLOW_NV_EMBED_SMEM=1 -DPLOW_NV_W8A8=1 -DPLOW_NV_W8A16=1 -DPLOW_CONFIG='"plow_config.h"' \
 *        -DPLOW_BUCKET_DECODE=1 -DPLOW_NV_FA_GF_FULL=4 -DPLOW_NV_ROW_SLOT=1 <PLOW_EXTRA_DEFINES> \
 *        bench_cu_sm120.cu
 *
 * (run_ubench_cu.sh --nv derives the list and checks it against the production command.)
 *
 * Instructions come verbatim from the packet (model.pkt), tensor handles index the packet's own
 * tensor table, so neighbouring-instruction fusions (Mamba scan state+pass) see real neighbours.
 */
#include "interp_sm120.cu"

#if PLOW_NV_INST_PRELOAD || PLOW_NV_SEGMENTS || PLOW_NV_SEG_GEMM || PLOW_NV_SEG_WS384
#error "ubench_exec drives plow_exec directly: build a plain decode/prefill object (no preload/segments)"
#endif

/* Runs instructions [0, n_inst) of `insts` against tensor table T.
 *   gridDim.x > 1: block b runs slice b of each instruction (the interpreter's distribution,
 *                  nblk = the instruction's blocks); blocks past it idle.
 *   gridDim.x == 1: the one block runs slices [slice0, slice0 + nslice) back to back, nblk =
 *                  nblk_override or the instruction's blocks: a real per-SM work slice, or (nslice
 *                  = nblk) the whole op on one SM.
 * stamp[4b..4b+3] = clock64 / %globaltimer before and after (block-synchronised). */
extern "C" __global__ void __launch_bounds__(PLOW_NV_THREADS, PLOW_NV_MINBLK)
    ubench_exec(const PlowDevInst* insts, void* const* T, unsigned long long* stamp, unsigned n_inst,
                unsigned slice0, unsigned nslice, unsigned nblk_override) {
    extern __shared__ float arena[];
    __syncthreads();
    unsigned long long c0 = 0, t0 = 0;
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        c0 = clock64();
    }
    __syncthreads();
    for (unsigned i = 0; i < n_inst; i++) {
        const PlowDevInst* in = insts + i;
        const unsigned nblk = nblk_override ? nblk_override : (in->blocks ? in->blocks : gridDim.x);
        const unsigned lo = gridDim.x > 1 ? blockIdx.x : slice0;
        const unsigned hi = gridDim.x > 1 ? blockIdx.x + 1 : slice0 + nslice;
        for (unsigned s = lo; s < hi && s < nblk; s++) {
            plow_exec(in, T, s, nblk, arena
#if PLOW_MIXED_STEP
                      , nullptr
#endif
#if PLOW_NV_TOKEN_BATCH_BF16
                      , nullptr
#endif
            );
            __syncthreads();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned long long c1 = clock64();
        unsigned long long t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        unsigned long long* s = stamp + 4ull * blockIdx.x;
        s[0] = c0;
        s[1] = c1;
        s[2] = t0;
        s[3] = t1;
    }
}
