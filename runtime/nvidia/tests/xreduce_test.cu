// Test-only wrapper: the interpreter's TP collectives (op_collective.cuh) as a plain kernel, one block
// per packet slice, for scripts/nv_tp/test_xreduce.py. Not part of any shipped object.
#include <cuda_bf16.h>
#include <stdio.h>
#include "../../common/dev_isa.h"
#include "../op_collective.cuh"

extern "C" __global__ void t_xreduce(__nv_bfloat16* out, void* const* peer_table, uint32_t* xctr, unsigned rank, unsigned n_gpu,
                                     unsigned n, unsigned slot, unsigned gate0, unsigned gate1, unsigned two, unsigned reps) {
    PlowProgram prog{};
    prog.xctr = xctr;
    prog.peer_scratch = peer_table;
    prog.rank = rank;
    prog.n_gpu = n_gpu;
    for (unsigned r = 0; r < reps; r++) {
        if (two)
            d_xreduce_twoshot_nv(prog, out, n, slot, gate0 + 2 * r, gate1 + 2 * r, blockIdx.x, gridDim.x);
        else
            d_xreduce_nv(prog, out, n, slot, gate0 + 2 * r, blockIdx.x, gridDim.x);
        __syncthreads();
    }
}
