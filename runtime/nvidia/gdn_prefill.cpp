#include "gdn_sm90.h"
#include <new>
#include <atomic>
#include <stdint.h>

// The generated CuTe host object caches its kernel globally within this shared library.
static std::atomic_flag live_handle = ATOMIC_FLAG_INIT;

struct PlowGdnHandle {
    plow_gdn_sm90_Kernel_Module_t module{};
};

extern "C" int plow_gdn_create(int device, PlowGdnHandle** result) {
    if (!result) return static_cast<int>(cudaErrorInvalidValue);
    *result = nullptr;
    if (live_handle.test_and_set(std::memory_order_acquire)) return -1001;
    cudaError_t rc = cudaSetDevice(device);
    if (rc != cudaSuccess) {
        live_handle.clear(std::memory_order_release);
        return static_cast<int>(rc);
    }
    auto* h = new (std::nothrow) PlowGdnHandle;
    if (!h) {
        live_handle.clear(std::memory_order_release);
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    cudaLibrary_t* library = &h->module.module;
    struct { cudaLibrary_t** library; cudaError_t* result; } init{&library, &rc};
    _mlir_plow_gdn_sm90_cuda_init(reinterpret_cast<void**>(&init));
    if (rc == cudaSuccess) {
        struct { cudaLibrary_t** library; int32_t* device; cudaError_t* result; }
            load{&library, &device, &rc};
        _mlir_plow_gdn_sm90_cuda_load_to_device(reinterpret_cast<void**>(&load));
    }
    if (rc != cudaSuccess) {
        cudaError_t cleanup = h->module.module ? cudaLibraryUnload(h->module.module) : cudaSuccess;
        delete h;
        if (cleanup == cudaSuccess) live_handle.clear(std::memory_order_release);
        return static_cast<int>(rc);
    }
    *result = h;
    return 0;
}

extern "C" int plow_gdn_destroy(PlowGdnHandle* h) {
    if (!h) return 0;
    cudaError_t rc = cudaLibraryUnload(h->module.module);
    delete h;
    if (rc == cudaSuccess) live_handle.clear(std::memory_order_release);
    return static_cast<int>(rc);
}

extern "C" int plow_gdn_run(PlowGdnHandle* h, void* q, void* k, void* v, void* out,
    void* alpha, void* beta, void* outstate, void* initialstate, void* maps,
    void* offsets, int tokens, void* stream) {
    if (!h || tokens <= 0 || tokens > 8192 || !q || !k || !v || !out || !alpha ||
        !beta || !outstate || !initialstate || !maps || !offsets ||
        (reinterpret_cast<uintptr_t>(maps) & 127u))
        return static_cast<int>(cudaErrorInvalidValue);
    constexpr size_t state_bytes = 48u * 128u * 128u * sizeof(float);
    uintptr_t dst = reinterpret_cast<uintptr_t>(initialstate);
    uintptr_t src = reinterpret_cast<uintptr_t>(outstate);
    if ((dst <= src && src - dst < state_bytes) ||
        (src < dst && dst - src < state_bytes))
        return static_cast<int>(cudaErrorInvalidValue);
    plow_gdn_sm90_Tensor_q_t qd{q, {tokens}};
    plow_gdn_sm90_Tensor_k_t kd{k, {tokens}};
    plow_gdn_sm90_Tensor_v_t vd{v, {tokens}};
    plow_gdn_sm90_Tensor_o_t od{out, {tokens}};
    plow_gdn_sm90_Tensor_alpha_t ad{alpha, {tokens * 48}};
    plow_gdn_sm90_Tensor_beta_t bd{beta, {tokens * 48}};
    plow_gdn_sm90_Tensor_state_t sd{outstate, {48 * 128 * 128}};
    plow_gdn_sm90_Tensor_initial_t id{initialstate, {48 * 128 * 128}};
    plow_gdn_sm90_Tensor_maps_t md{maps};
    plow_gdn_sm90_Tensor_offsets_t cd{offsets, {2}};
    int rc = cute_dsl_plow_gdn_sm90_wrapper(&h->module, &qd, &kd, &vd, &od,
        &ad, &bd, &sd, &id, &md, &cd, 0.08838834764831845f,
        static_cast<cudaStream_t>(stream));
    if (rc) return rc;
    cudaError_t launch = cudaPeekAtLastError();
    if (launch != cudaSuccess) return static_cast<int>(launch);
    // Prefill writes a separate final state; update the persistent V-first state in stream order.
    return static_cast<int>(cudaMemcpyAsync(initialstate, outstate, state_bytes,
        cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
}
