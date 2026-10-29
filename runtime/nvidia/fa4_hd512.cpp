// Host-side wrapper of FlashAttention-4's SM90 forward (CuTe DSL, from vLLM's vllm_flash_attn,
// Apache-2.0 / BSD-3 upstream flash-attention) exported AOT for head dim 512 by
// scripts/export_fa4_hd512_sm90.py: causal, varlen Q, paged K/V of 64 rows (the TMA paged path).
// Per GQA ratio (16 query heads over 1 or 2 KV heads) a whole-KV entry and a split-KV entry whose
// f32 partials the exported combine entry folds. plowrt dlopens the library beside the packet
// and launches it on the segment stream; no allocation, no sync.
#include "fa4_hd512_combine.h"
#include "fa4_hd512_h16kv1.h"
#include "fa4_hd512_h16kv1_split.h"
#include "fa4_hd512_h16kv2.h"
#include "fa4_hd512_h16kv2_split.h"
#include <atomic>
#include <new>
#include <stdint.h>

static std::atomic_flag live_handle = ATOMIC_FLAG_INIT;

struct PlowFa4Handle {
    plow_fa4_hd512_h16kv1_Kernel_Module_t h16kv1{};
    plow_fa4_hd512_h16kv1_split_Kernel_Module_t h16kv1_split{};
    plow_fa4_hd512_h16kv2_Kernel_Module_t h16kv2{};
    plow_fa4_hd512_h16kv2_split_Kernel_Module_t h16kv2_split{};
    plow_fa4_hd512_combine_Kernel_Module_t combine{};
};

extern "C" unsigned plow_fa4_abi() { return 2; }

#define PLOW_FA4_MODULES(X) X(h16kv1) X(h16kv1_split) X(h16kv2) X(h16kv2_split) X(combine)

static void unload_all(PlowFa4Handle* h) {
#define PLOW_FA4_UNLOAD(V) \
    if (h->V.module) cudaLibraryUnload(h->V.module);
    PLOW_FA4_MODULES(PLOW_FA4_UNLOAD)
#undef PLOW_FA4_UNLOAD
}

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
#define PLOW_FA4_LOAD(V)                                                              \
    if (rc == cudaSuccess) {                                                          \
        cudaLibrary_t* lib = &h->V.module;                                            \
        init = {&lib, &rc};                                                           \
        _mlir_plow_fa4_hd512_##V##_cuda_init(reinterpret_cast<void**>(&init));        \
        if (rc == cudaSuccess) {                                                      \
            load = {&lib, &device, &rc};                                              \
            _mlir_plow_fa4_hd512_##V##_cuda_load_to_device(reinterpret_cast<void**>(&load)); \
        }                                                                             \
    }
    PLOW_FA4_MODULES(PLOW_FA4_LOAD)
#undef PLOW_FA4_LOAD
    if (rc != cudaSuccess) {
        unload_all(h);
        delete h;
        live_handle.clear(std::memory_order_release);
        return static_cast<int>(rc);
    }
    *result = h;
    return 0;
}

extern "C" int plow_fa4_destroy(PlowFa4Handle* h) {
    if (!h) return 0;
    unload_all(h);
    delete h;
    live_handle.clear(std::memory_order_release);
    return 0;
}

// Workspace bytes a split launch of `total_q` rows at `splits` splits needs.
extern "C" uint64_t plow_fa4_hd512_workspace(int total_q, int splits) {
    return splits <= 1 ? 0
                       : (uint64_t)splits * total_q * 16 * (512 * 4 + 4);
}

// q / out [total_q][16][512]; k / v pages [num_pages][64][kv_heads][512] with page stride 64*512
// and row stride 512 elements, head stride `head_stride` elements; cu_q [nreq + 1]; used_k
// [nreq]; page_table [nreq][max_pages] int32. `splits` > 1 runs the split-KV entry through
// `workspace` (plow_fa4_hd512_workspace bytes, 16 B aligned) and the combine.
extern "C" int plow_fa4_hd512_run(PlowFa4Handle* h, unsigned heads, unsigned kv_heads,
                                  void* q, void* k, void* v, void* out, int total_q, int nreq,
                                  void* cu_q, void* used_k, void* page_table, int max_pages,
                                  int num_pages, int64_t head_stride, float scale, int splits,
                                  void* workspace, uint64_t workspace_bytes, void* stream) {
    if (!h || heads != 16 || (kv_heads != 1 && kv_heads != 2) || !q || !k || !v || !out ||
        total_q <= 0 || nreq <= 0 || !cu_q || !used_k || !page_table || max_pages <= 0 ||
        num_pages <= 0 || head_stride <= 0 || head_stride % 8 || splits < 1 || splits > 128 ||
        (splits > 1 && (!workspace || reinterpret_cast<uintptr_t>(workspace) % 16 ||
                        workspace_bytes < plow_fa4_hd512_workspace(total_q, splits))))
        return static_cast<int>(cudaErrorInvalidValue);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int32_t kvh = static_cast<int32_t>(kv_heads);
    float* opart = static_cast<float*>(workspace);
    float* lse = splits > 1 ? opart + (size_t)splits * total_q * 16 * 512 : nullptr;
    const int64_t opart_split = (int64_t)total_q * 16 * 512;
    const int64_t lse_split = (int64_t)16 * total_q;
#define PLOW_FA4_TENSORS(V)                                                                   \
    plow_fa4_hd512_##V##_Tensor_q_t tq{q, {total_q, 16, 512}};                               \
    plow_fa4_hd512_##V##_Tensor_k_t tk{k, {num_pages, 64, kvh, 512}, {head_stride}};          \
    plow_fa4_hd512_##V##_Tensor_v_t tv{v, {num_pages, 64, kvh, 512}, {head_stride}};          \
    plow_fa4_hd512_##V##_Tensor_cuq_t tc{cu_q, {nreq + 1}};                                   \
    plow_fa4_hd512_##V##_Tensor_usedk_t tu{used_k, {nreq}};                                   \
    plow_fa4_hd512_##V##_Tensor_pt_t tp{page_table, {nreq, max_pages}, {max_pages}};
#define PLOW_FA4_WHOLE(V)                                                                     \
    {                                                                                         \
        PLOW_FA4_TENSORS(V)                                                                   \
        plow_fa4_hd512_##V##_Tensor_o_t to{out, {total_q, 16, 512}};                         \
        return cute_dsl_plow_fa4_hd512_##V##_wrapper(&h->V, &tq, &tk, &tv, &to, &tc, &tu, &tp, \
                                                     scale, s);                                \
    }
#define PLOW_FA4_SPLIT(V)                                                                     \
    {                                                                                         \
        PLOW_FA4_TENSORS(V)                                                                   \
        plow_fa4_hd512_##V##_Tensor_o_t to{opart, {splits, total_q, 16, 512}, {opart_split}}; \
        plow_fa4_hd512_##V##_Tensor_lse_t tl{lse, {splits, 16, total_q}, {lse_split, total_q}}; \
        rc = cute_dsl_plow_fa4_hd512_##V##_wrapper(&h->V, &tq, &tk, &tv, &to, &tl, &tc, &tu,  \
                                                   &tp, scale, s);                             \
    }
    if (splits == 1) {
        if (kv_heads == 1) PLOW_FA4_WHOLE(h16kv1)
        PLOW_FA4_WHOLE(h16kv2)
    }
    int32_t rc = 0;
    if (kv_heads == 1) PLOW_FA4_SPLIT(h16kv1_split) else PLOW_FA4_SPLIT(h16kv2_split)
    if (rc) return rc;
    plow_fa4_hd512_combine_Tensor_o_part_t cp{opart, {splits, total_q, 16, 512}, {opart_split}};
    // LSE partials [splits][16][total_q] read as [splits][total_q][16].
    plow_fa4_hd512_combine_Tensor_lse_part_t cl{lse, {splits, total_q, 16}, {lse_split, total_q}};
    plow_fa4_hd512_combine_Tensor_o_t co{out, {total_q, 16, 512}};
    plow_fa4_hd512_combine_Tensor_cuq_t cc{cu_q, {nreq + 1}};
    return cute_dsl_plow_fa4_hd512_combine_wrapper(&h->combine, &cp, &cl, &co, &cc, s);
#undef PLOW_FA4_SPLIT
#undef PLOW_FA4_WHOLE
#undef PLOW_FA4_TENSORS
}
