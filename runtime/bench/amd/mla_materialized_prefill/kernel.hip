#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_attention.h"

static constexpr unsigned DK_ABS = 512;
static constexpr unsigned DR = 64;
static constexpr unsigned DK_MAT = 192;
static constexpr unsigned DV_MAT = 128;

extern "C" __global__ __launch_bounds__(256, 1) void k_absorbed(
    float* opart, float* mlpart, const bf16* qabs, const bf16* qrope, const bf16* ckv,
    const bf16* krope, const int* kv_len, unsigned n_tok, unsigned n_head,
    unsigned kv_stride, float scale) {
    __shared__ bf16 lds[FA_MLA_PF2_LDS_BYTES(DK_ABS, DR) / 2];
    d_flash_mla_prefill_v2<DK_ABS, DR>(
        opart, mlpart, qabs, qrope, ckv, krope, kv_len, 1, n_tok, n_head, kv_stride, 0,
        scale, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(256, 1) void k_absorbed_fold(
    bf16* out, const float* opart, const float* mlpart, const bf16* wuv,
    unsigned n_tok, unsigned n_head) {
    __shared__ float lds[DK_ABS + PLOW_WAVES * PLOW_MLA_FOLD_VT];
    d_mla_merge_fold<DK_ABS, PLOW_MLA_FOLD_VT>(
        out, opart, mlpart, wuv, n_tok, n_head, DV_MAT, 1, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ __launch_bounds__(256, 1) void k_materialized(
    float* opart, float* mlpart, bf16* out, const bf16* q, const bf16* k,
    const bf16* v, unsigned n_tok, unsigned n_head, unsigned kv_stride, float scale) {
    __shared__ bf16 lds[FA_LDS_HALVES(DK_MAT)];
    d_flash_prefill<DK_MAT, false, DV_MAT>(
        opart, mlpart, out, q, k, v, n_tok, n_tok, n_head, n_head, 0, 0, scale,
        kv_stride, PLOW_KV_MASK_NONE, 1, blockIdx.x, gridDim.x, lds);
}

extern "C" __global__ void k_make_inputs(
    bf16* qabs, bf16* qrope, bf16* ckv, bf16* krope, bf16* qmat,
    bf16* kmat, bf16* vmat, bf16* wuv, unsigned n_tok, unsigned n_head) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t qrows = (size_t)n_tok * n_head;
    for (size_t e = i; e < qrows * DK_ABS; e += stride) {
        const unsigned d = e % DK_ABS;
        const unsigned x = (unsigned)e * 1664525u + 1013904223u;
        qabs[e] = d < DV_MAT ? (bf16)(0x3b00u | ((x >> 16) & 0xffu)) : (bf16)0;
    }
    for (size_t e = i; e < qrows * DR; e += stride) {
        const unsigned x = (unsigned)e * 22695477u + 1u;
        qrope[e] = (bf16)(0x3a80u | ((x >> 16) & 0x7fu));
    }
    for (size_t e = i; e < (size_t)n_tok * DK_ABS; e += stride) {
        const unsigned x = (unsigned)e * 1103515245u + 12345u;
        ckv[e] = (bf16)(0x3b00u | ((x >> 16) & 0xffu));
    }
    for (size_t e = i; e < (size_t)n_tok * DR; e += stride) {
        const unsigned x = (unsigned)e * 214013u + 2531011u;
        krope[e] = (bf16)(0x3a80u | ((x >> 16) & 0x7fu));
    }
    for (size_t e = i; e < qrows * DK_MAT; e += stride) {
        const size_t row = e / DK_MAT;
        const unsigned d = e % DK_MAT;
        qmat[e] = d < DV_MAT ? qabs[row * DK_ABS + d]
                             : qrope[row * DR + d - DV_MAT];
    }
    for (size_t e = i; e < (size_t)n_head * n_tok * DK_MAT; e += stride) {
        const size_t row = e / DK_MAT;
        const unsigned d = e % DK_MAT;
        const unsigned tok = row % n_tok;
        kmat[e] = d < DV_MAT ? ckv[(size_t)tok * DK_ABS + d]
                             : krope[(size_t)tok * DR + d - DV_MAT];
    }
    for (size_t e = i; e < (size_t)n_head * n_tok * DV_MAT; e += stride) {
        const unsigned d = e % DV_MAT;
        const unsigned tok = (e / DV_MAT) % n_tok;
        vmat[e] = ckv[(size_t)tok * DK_ABS + d];
    }
    for (size_t e = i; e < (size_t)n_head * DK_ABS * DV_MAT; e += stride) {
        const unsigned d = e % DV_MAT;
        const unsigned l = (e / DV_MAT) % DK_ABS;
        wuv[e] = (bf16)(l == d ? 0x3f80u : 0u);
    }
}

extern "C" __global__ void k_cache_flush(const unsigned* in, unsigned* out, size_t n) {
    unsigned v = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        v ^= in[i];
    if (v) atomicXor(out, v);
}
