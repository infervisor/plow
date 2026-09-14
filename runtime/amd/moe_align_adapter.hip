#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "dev_isa.h"
#include "op_moe.h"
extern "C" __device__ unsigned plow_moe_align_abi_1 = 1;
#endif

/* The grouped-MoE prefill align (`d_moe_align_pf` phases 1-4) as a host-launched native kernel, so
 * the AITER MoE segment can sort its own routing table instead of four interpreter packets doing it
 * (PLOW_GLM_MOE_NATIVE_ALIGN). The host launches the phases with the interpreter's geometry --
 * grids 64 / 1 / 64 / 64, npart 64 -- so meta, the partial counts and the row maps are the same
 * bytes the interpreter writes. */
struct MoeAlignArgs {
    int* meta;
    const unsigned char* table;
    unsigned* row_token;
    unsigned* row_partidx;
    float* row_gate;
    unsigned rows;
    unsigned n_exp;
    unsigned topk;
    unsigned phase;
    unsigned npart;
    unsigned pad;
};
static_assert(sizeof(MoeAlignArgs) == 64);

extern "C" __global__ void plow_moe_align(MoeAlignArgs a) {
#if defined(__HIP_DEVICE_COMPILE__)
    /* phase 1: cnt[n_exp]; phase 2: cnt[PLOW_THREADS]. */
    __shared__ unsigned lds[1024];
    d_moe_align_pf(a.meta, a.table, a.row_token, a.row_partidx, a.row_gate, a.rows, a.n_exp, a.topk,
                   blockIdx.x, lds, a.phase, gridDim.x, a.npart);
#endif
}
