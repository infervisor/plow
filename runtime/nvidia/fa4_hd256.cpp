// Host-side wrapper of FlashAttention-4's SM90 forward (CuTe DSL, from vLLM's vllm_flash_attn,
// Apache-2.0 / BSD-3 upstream flash-attention) exported AOT for head dim 256 by
// scripts/export_fa4_hd256_sm90.py: causal, varlen Q, paged K/V of 64 rows over a row-interleaved
// cache (row stride kv_heads * 256; the non-TMA paged path, row-predicated at seqused_k).
// PLOW_FA4_H query heads over PLOW_FA4_KVH KV heads (the build passes both and -includes the
// exported headers): a whole-KV entry and a split-KV entry whose f32 partials the exported combine
// entry folds. plowrt dlopens the library beside the packet and launches it on the segment
// stream; no allocation, no sync. Same C ABI as fa4_hd512.cpp (plow_fa4 ABI 3).
#include <atomic>
#include <new>
#include <stdint.h>

#if !defined(PLOW_FA4_H) || !defined(PLOW_FA4_KVH)
#error "PLOW_FA4_H and PLOW_FA4_KVH name the exported geometry"
#endif

#define PF_CAT_(a, b) a##b
#define PF_CAT(a, b) PF_CAT_(a, b)
#define PF_STR_(x) #x
#define PF_STR(x) PF_STR_(x)
// plow_fa4_hd256_h<H>kv<KVH>, its split twin, and plow_fa4_hd256_h<H>_combine.
#define PF_W PF_CAT(PF_CAT(PF_CAT(plow_fa4_hd256_h, PLOW_FA4_H), kv), PLOW_FA4_KVH)
#define PF_S PF_CAT(PF_W, _split)
#define PF_C PF_CAT(PF_CAT(plow_fa4_hd256_h, PLOW_FA4_H), _combine)
#define PF_T(m, x) PF_CAT(m, PF_CAT(_, x))
#define PF_WRAP(m) PF_CAT(PF_CAT(cute_dsl_, m), _wrapper)
#define PF_INIT(m) PF_CAT(PF_CAT(_mlir_, m), _cuda_init)
#define PF_LOAD(m) PF_CAT(PF_CAT(_mlir_, m), _cuda_load_to_device)

static std::atomic_flag live_handle = ATOMIC_FLAG_INIT;
static constexpr int32_t H = PLOW_FA4_H, KVH = PLOW_FA4_KVH, D = 256;

struct PlowFa4Handle {
    PF_T(PF_W, Kernel_Module_t) whole{};
    PF_T(PF_S, Kernel_Module_t) split{};
    PF_T(PF_C, Kernel_Module_t) combine{};
};

// ABI 3: the KV adapters (runtime/nvidia/kv_dequant_sm90.cu) are linked in.
extern "C" unsigned plow_fa4_abi() { return 3; }
extern "C" unsigned plow_fa4_dtype() { return 0; }
// The emitter reads the ABI and geometry from this marker without loading the library.
extern "C" __attribute__((used, visibility("default"))) const char plow_fa4_caps[] =
    "plow_fa4_caps:abi=3:dtype=0:hd=256:heads=" PF_STR(PLOW_FA4_H) ":kv_heads=" PF_STR(
        PLOW_FA4_KVH) ":layout=ilv";

static void unload_all(PlowFa4Handle* h) {
    if (h->whole.module) cudaLibraryUnload(h->whole.module);
    if (h->split.module) cudaLibraryUnload(h->split.module);
    if (h->combine.module) cudaLibraryUnload(h->combine.module);
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
#define PF_LOAD_ONE(field, m)                                                  \
    if (rc == cudaSuccess) {                                                   \
        cudaLibrary_t* lib = &h->field.module;                                 \
        init = {&lib, &rc};                                                    \
        PF_INIT(m)(reinterpret_cast<void**>(&init));                           \
        if (rc == cudaSuccess) {                                               \
            load = {&lib, &device, &rc};                                       \
            PF_LOAD(m)(reinterpret_cast<void**>(&load));                       \
        }                                                                      \
    }
    PF_LOAD_ONE(whole, PF_W)
    PF_LOAD_ONE(split, PF_S)
    PF_LOAD_ONE(combine, PF_C)
#undef PF_LOAD_ONE
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

// q / out [total_q][H][256]; k / v pages [num_pages][64][KVH][256] (page stride 64*KVH*256, row
// stride KVH*256: the row-interleaved cache in place); cu_q [nreq + 1]; used_k [nreq]; page_table
// [nreq][max_pages] int32. `splits` > 1 runs the split-KV entry through `workspace` (f32 O + LSE
// per split, row, head; 16 B aligned) and the combine. head_stride is fixed by the layout.
extern "C" int plow_fa4_run(PlowFa4Handle* h, unsigned heads, unsigned kv_heads, void* q, void* k,
                            void* v, void* out, int total_q, int nreq, void* cu_q, void* used_k,
                            void* page_table, int max_pages, int num_pages, int64_t head_stride,
                            float scale, int splits, void* workspace, uint64_t workspace_bytes,
                            void* stream) {
    const uint64_t need = splits <= 1 ? 0 : (uint64_t)splits * total_q * H * (D * 4 + 4);
    if (!h || heads != (unsigned)H || kv_heads != (unsigned)KVH || !q || !k || !v || !out ||
        total_q <= 0 || nreq <= 0 || !cu_q || !used_k || !page_table || max_pages <= 0 ||
        num_pages <= 0 || head_stride != D || splits < 1 || splits > 128 ||
        (splits > 1 && (!workspace || reinterpret_cast<uintptr_t>(workspace) % 16 ||
                        workspace_bytes < need)))
        return static_cast<int>(cudaErrorInvalidValue);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    float* opart = static_cast<float*>(workspace);
    float* lse = splits > 1 ? opart + (size_t)splits * total_q * H * D : nullptr;
    const int64_t opart_split = (int64_t)total_q * H * D;
    const int64_t lse_split = (int64_t)H * total_q;
#define PF_TENSORS(m)                                                          \
    PF_T(m, Tensor_q_t) tq{q, {total_q, H, D}};                                \
    PF_T(m, Tensor_k_t) tk{k, {num_pages, 64, KVH, D}};                        \
    PF_T(m, Tensor_v_t) tv{v, {num_pages, 64, KVH, D}};                        \
    PF_T(m, Tensor_cuq_t) tc{cu_q, {nreq + 1}};                                \
    PF_T(m, Tensor_usedk_t) tu{used_k, {nreq}};                                \
    PF_T(m, Tensor_pt_t) tp{page_table, {nreq, max_pages}, {max_pages}};
    if (splits == 1) {
        PF_TENSORS(PF_W)
        PF_T(PF_W, Tensor_o_t) to{out, {total_q, H, D}};
        return PF_WRAP(PF_W)(&h->whole, &tq, &tk, &tv, &to, &tc, &tu, &tp, scale, s);
    }
    {
        PF_TENSORS(PF_S)
        PF_T(PF_S, Tensor_o_t) to{opart, {splits, total_q, H, D}, {opart_split}};
        PF_T(PF_S, Tensor_lse_t) tl{lse, {splits, H, total_q}, {lse_split, total_q}};
        const int32_t rc = PF_WRAP(PF_S)(&h->split, &tq, &tk, &tv, &to, &tl, &tc, &tu, &tp, scale, s);
        if (rc) return rc;
    }
    PF_T(PF_C, Tensor_o_part_t) cp{opart, {splits, total_q, H, D}, {opart_split}};
    // LSE partials [splits][H][total_q] read as [splits][total_q][H].
    PF_T(PF_C, Tensor_lse_part_t) cl{lse, {splits, total_q, H}, {lse_split, total_q}};
    PF_T(PF_C, Tensor_o_t) co{out, {total_q, H, D}};
    PF_T(PF_C, Tensor_cuq_t) cc{cu_q, {nreq + 1}};
    return PF_WRAP(PF_C)(&h->combine, &cp, &cl, &co, &cc, s);
#undef PF_TENSORS
}
