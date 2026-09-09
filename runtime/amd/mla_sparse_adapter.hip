#include <hip/hip_runtime.h>

struct SparsePackArgs {
    uint4 *q, *kv;
    const uint4 *qa, *qr, *ck, *kr;
    unsigned *qp, *kp, *last, *splits;
    unsigned rows, ctx;
};
static_assert(sizeof(SparsePackArgs) == 88);

extern "C" __global__ void plow_mla_sparse_pack(SparsePackArgs a) {
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    for (unsigned i = first; i < a.rows * 8u * 72u; i += stride) {
        const unsigned row = i / 72u, col = i % 72u;
        a.q[i] = col < 64u ? a.qa[row * 64u + col] : a.qr[row * 8u + col - 64u];
    }
    for (unsigned i = first; i < a.ctx * 72u; i += stride) {
        const unsigned row = i / 72u, col = i % 72u;
        a.kv[i] = col < 64u ? a.ck[row * 64u + col] : a.kr[row * 8u + col - 64u];
    }
    for (unsigned row = first; row <= a.rows; row += stride) {
        a.qp[row] = row;
        a.kp[row] = row * 2048u;
        a.splits[row] = row * 2u;
        if (row < a.rows) a.last[row] = 1;
    }
}

struct SparseReduceArgs {
    float *out, *ml;
    const float *part, *lse;
    unsigned rows, pad;
};
static_assert(sizeof(SparseReduceArgs) == 40);

extern "C" __global__ void plow_mla_sparse_reduce(SparseReduceArgs a) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < a.rows * 8u * 128u;
         i += blockDim.x * gridDim.x) {
        const unsigned hrow = i / 128u, col = i % 128u;
        const unsigned l = (hrow / 8u) * 16u + hrow % 8u;
        const float m = fmaxf(a.lse[l], a.lse[l + 8u]);
        const float wa = expf(a.lse[l] - m), wb = expf(a.lse[l + 8u] - m);
        const float4 pa = ((const float4*)a.part)[l * 128u + col];
        const float4 pb = ((const float4*)a.part)[(l + 8u) * 128u + col];
        ((float4*)a.out)[i] = make_float4(
            (pa.x * wa + pb.x * wb) / (wa + wb),
            (pa.y * wa + pb.y * wb) / (wa + wb),
            (pa.z * wa + pb.z * wb) / (wa + wb),
            (pa.w * wa + pb.w * wb) / (wa + wb));
        if (col == 0) {
            a.ml[hrow * 2u] = 0;
            a.ml[hrow * 2u + 1u] = 1;
        }
    }
}

extern "C" __device__ unsigned plow_mla_sparse_adapter_abi_1 = 1;
