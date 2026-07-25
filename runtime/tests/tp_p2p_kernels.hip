/* tp_p2p_kernels.hip — device code for the cross-GPU (XGMI/Infinity-Fabric)
 * transport de-risk microbench. Phase 0 of tensor-parallel decode.
 *
 * These kernels run on peer-mapped VRAM (plow_hsa_alloc_peer): a buffer that
 * lives on one GPU but is visible to a kernel on another over XGMI. They prove
 * three things the TP decode collective needs:
 *
 *   1. tp_fill / tp_verify  — a kernel on GPU_B can READ a buffer a kernel on
 *      GPU_A wrote (peer load over the fabric), byte-exact.
 *   2. tp_ping / tp_pong    — two kernels on two GPUs synchronize with nothing
 *      but a SYSTEM-scope atomic on peer memory (no HSA signal, no host). This
 *      is the cross-GPU counter-gate primitive; the ping side times the
 *      round-trip on its own s_memrealtime clock.
 *   3. tp_reduce_oneshot    — the decode all-reduce inner loop: sum N peer
 *      copies of a vector into the local output. Proves the one-shot reduction
 *      runs as plain device code over peer buffers.
 *
 * Clock: s_memrealtime is the constant-rate ~100 MHz counter (10 ns/tick) that
 * dev_isa.h uses for tracing. The ping side reads it before and after, so no
 * cross-GPU clock comparison is ever made — only a single GPU's delta.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

/* System-scope atomic helpers. AGENT scope (what interp.hip's counter-gate uses
 * intra-GPU) is NOT enough across GPUs: the release/acquire has to reach past
 * this device's L2 out onto XGMI to the peer. SYSTEM scope is the widest and is
 * exactly the extension the cross-GPU gate needs. */
__device__ static inline uint32_t sys_load(const uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ static inline void sys_store(uint32_t* p, uint32_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

/* ---- peer read/write proof ---------------------------------------------- */

/* GPU_A: write a checkable pattern into `buf` (which may be peer VRAM). */
extern "C" __global__ void tp_fill(uint32_t* buf, uint32_t n, uint32_t seed) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        buf[i] = seed + i * 2654435761u; /* Knuth mix — catches partial writes */
}

/* GPU_B: read `buf` (peer VRAM on GPU_A) over XGMI and count mismatches into
 * `errs` (local VRAM on GPU_B, read back by the host). */
extern "C" __global__ void tp_verify(const uint32_t* buf, uint32_t n,
                                     uint32_t seed, uint32_t* errs) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        if (buf[i] != seed + i * 2654435761u)
            atomicAdd(errs, 1u);
}

/* ---- cross-GPU handshake (ping-pong over one system-scope atomic) -------- */

/* Both kernels launch as a single thread (grid=1, block=1): this is a latency
 * probe, not a throughput one. `p2c` and `c2p` are two words in the SAME peer
 * buffer. `status`/`cycles` live in the ping GPU's local VRAM.
 *
 * Round i: ping stores p2c=i and spins on c2p; pong sees p2c=i and stores c2p=i.
 * `cycles` = s_memrealtime delta over `iters` round-trips; one-way handshake =
 * cycles / (2*iters) ticks. A spin deadline (in ticks) guards against a fabric
 * where system-scope atomics do NOT propagate — the kernel bails instead of
 * hanging the queue, and `status` reports it. */
extern "C" __global__ void tp_ping(uint32_t* p2c, uint32_t* c2p,
                                    uint32_t iters, uint64_t deadline_ticks,
                                    uint64_t* cycles, uint32_t* status) {
    const uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    for (uint32_t i = 1; i <= iters; i++) {
        sys_store(p2c, i);
        const uint64_t spin0 = __builtin_amdgcn_s_memrealtime();
        while (sys_load(c2p) < i) {
            if (__builtin_amdgcn_s_memrealtime() - spin0 > deadline_ticks) {
                *status = 0xDEAD0000u | i; /* atomics did not propagate */
                *cycles = 0;
                return;
            }
        }
    }
    const uint64_t t1 = __builtin_amdgcn_s_memrealtime();
    *cycles = t1 - t0;
    *status = 0; /* ok */
}

extern "C" __global__ void tp_pong(uint32_t* p2c, uint32_t* c2p,
                                   uint32_t iters, uint64_t deadline_ticks) {
    for (uint32_t i = 1; i <= iters; i++) {
        const uint64_t spin0 = __builtin_amdgcn_s_memrealtime();
        while (sys_load(p2c) < i) {
            if (__builtin_amdgcn_s_memrealtime() - spin0 > deadline_ticks)
                return; /* ping already gave up; don't hang the queue */
        }
        sys_store(c2p, i);
    }
}

/* ---- one-shot all-reduce reduction step --------------------------------- */

/* The decode all-reduce, per GPU: every rank has already exposed its partial in
 * a peer buffer; `peers[r]` points at rank r's copy (peer VRAM). Sum the N
 * copies element-wise into `out` (local). This is the compute half of one-shot;
 * the data movement is the peer mapping itself plus one system-scope counter
 * bump (tp_ping's primitive) to gate "all N partials are visible". */
extern "C" __global__ void tp_reduce_oneshot(float* out, const float* const* peers,
                                             uint32_t nranks, uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < nranks; r++) acc += peers[r][i];
        out[i] = acc;
    }
}
