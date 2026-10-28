// Host-side wrapper of FlashAttention-4's SM90 forward (CuTe DSL, from vLLM's vllm_flash_attn,
// Apache-2.0 / BSD-3 upstream flash-attention) exported AOT for head dim 512 by
// scripts/export_fa4_hd512_sm90.py: causal, varlen Q, paged K/V of 64 rows (the TMA paged path).
// One exported entry per GQA ratio (16 query heads over 1 or 2 KV heads). plowrt dlopens the
// library beside the packet and launches it on the segment stream; no allocation, no sync.
#include "fa4_hd512_h16kv1.h"
#include "fa4_hd512_h16kv2.h"
#include <atomic>
#include <new>
#include <stdint.h>

static std::atomic_flag live_handle = ATOMIC_FLAG_INIT;

struct PlowFa4Handle {
    plow_fa4_hd512_h16kv1_Kernel_Module_t h16kv1{};
    plow_fa4_hd512_h16kv2_Kernel_Module_t h16kv2{};
};

extern "C" unsigned plow_fa4_abi() { return 1; }

extern "C" int plow_fa4_create(int device, PlowFa4Handle** result) {
    if (!result) return static_cast<int>(cudaErrorInvalidValue);
    *result = nullptr;
    if (live_handle.test_and_set(std::memory_order_acquire)) return -1001;
    cudaError_t rc = cudaSetDevice(device);
    auto* h = rc == cudaSuccess ? new (std::nothrow) PlowFa4Handle : nullptr;
    if (!h) {
        live_handle.clear(std::memory_order_release);
        return static_cast<int>(rc == cudaSuccess ? cudaErrorMemoryAllocation : rc);
    }
    struct { cudaLibrary_t** library; cudaError_t* result; } init;
    struct { cudaLibrary_t** library; int32_t* device; cudaError_t* result; } load;
    cudaLibrary_t* lib1 = &h->h16kv1.module;
    cudaLibrary_t* lib2 = &h->h16kv2.module;
    init = {&lib1, &rc};
    _mlir_plow_fa4_hd512_h16kv1_cuda_init(reinterpret_cast<void**>(&init));
    if (rc == cudaSuccess) {
        load = {&lib1, &device, &rc};
        _mlir_plow_fa4_hd512_h16kv1_cuda_load_to_device(reinterpret_cast<void**>(&load));
    }
    if (rc == cudaSuccess) {
        init = {&lib2, &rc};
        _mlir_plow_fa4_hd512_h16kv2_cuda_init(reinterpret_cast<void**>(&init));
    }
    if (rc == cudaSuccess) {
        load = {&lib2, &device, &rc};
        _mlir_plow_fa4_hd512_h16kv2_cuda_load_to_device(reinterpret_cast<void**>(&load));
    }
    if (rc != cudaSuccess) {
        if (h->h16kv1.module) cudaLibraryUnload(h->h16kv1.module);
        if (h->h16kv2.module) cudaLibraryUnload(h->h16kv2.module);
        delete h;
        live_handle.clear(std::memory_order_release);
        return static_cast<int>(rc);
    }
    *result = h;
    return 0;
}

extern "C" int plow_fa4_destroy(PlowFa4Handle* h) {
    if (!h) return 0;
    cudaError_t a = cudaLibraryUnload(h->h16kv1.module);
    cudaError_t b = cudaLibraryUnload(h->h16kv2.module);
    delete h;
    live_handle.clear(std::memory_order_release);
    return static_cast<int>(a != cudaSuccess ? a : b);
}

// q / out [total_q][16][512]; k / v pages [num_pages][64][kv_heads][512] with page stride 64*512
// and row stride 512 elements, head stride `head_stride` elements (a head-major slot's
// kv_stride*512); cu_q [nreq + 1]; used_k [nreq]; page_table [nreq][max_pages] int32.
extern "C" int plow_fa4_hd512_run(PlowFa4Handle* h, unsigned heads, unsigned kv_heads,
                                  void* q, void* k, void* v, void* out, int total_q, int nreq,
                                  void* cu_q, void* used_k, void* page_table, int max_pages,
                                  int num_pages, int64_t head_stride, float scale, void* stream) {
    if (!h || heads != 16 || (kv_heads != 1 && kv_heads != 2) || !q || !k || !v || !out ||
        total_q <= 0 || nreq <= 0 || !cu_q || !used_k || !page_table || max_pages <= 0 ||
        num_pages <= 0 || head_stride <= 0 || head_stride % 8)
        return static_cast<int>(cudaErrorInvalidValue);
    const int32_t qs[3] = {total_q, 16, 512};
    const int32_t ks[4] = {num_pages, 64, static_cast<int32_t>(kv_heads), 512};
    const int32_t one[1] = {nreq + 1};
    const int32_t used[1] = {nreq};
    const int32_t pts[2] = {nreq, max_pages};
    cudaStream_t s = static_cast<cudaStream_t>(stream);
#define PLOW_FA4_CALL(V)                                                                       \
    {                                                                                          \
        plow_fa4_hd512_##V##_Tensor_q_t tq{q, {qs[0], qs[1], qs[2]}};                        \
        plow_fa4_hd512_##V##_Tensor_k_t tk{k, {ks[0], ks[1], ks[2], ks[3]}, {head_stride}};   \
        plow_fa4_hd512_##V##_Tensor_v_t tv{v, {ks[0], ks[1], ks[2], ks[3]}, {head_stride}};   \
        plow_fa4_hd512_##V##_Tensor_o_t to{out, {qs[0], qs[1], qs[2]}};                      \
        plow_fa4_hd512_##V##_Tensor_cuq_t tc{cu_q, {one[0]}};                                 \
        plow_fa4_hd512_##V##_Tensor_usedk_t tu{used_k, {used[0]}};                            \
        plow_fa4_hd512_##V##_Tensor_pt_t tp{page_table, {pts[0], pts[1]}, {max_pages}};       \
        return cute_dsl_plow_fa4_hd512_##V##_wrapper(&h->V, &tq, &tk, &tv, &to, &tc, &tu, &tp, \
                                                     scale, s);                                \
    }
    if (kv_heads == 1) PLOW_FA4_CALL(h16kv1)
    PLOW_FA4_CALL(h16kv2)
#undef PLOW_FA4_CALL
}
