/* cpu_dev.h — CPU kernel library for the device ISA (runtime/common/dev_isa.h).
 *
 * The CPU twin of interp.hip's `plow_exec`: one entry per PlowDevInst, computing the
 * `slice`-th of `nblk` shares of the op, exactly the contract every GPU interpreter
 * honours. Gating, counters, streams and threads live in Rust (plowrt::exec::cpu);
 * this library is pure compute over host pointers and is called from persistent
 * worker threads, so nothing here allocates, locks, or spawns.
 *
 * Tiers: every op has a GOLDEN scalar kernel (the oracle, any x86-64). AVX-512 and
 * AMX kernels override table entries at plow_cpu_init() according to cpuid and the
 * caller's cap, so one binary runs everywhere.
 *
 * ABI: plowrt::exec::cpu::ffi mirrors this header; crates/plowrt/tests/cpu_abi.rs
 * compiles it and asserts sizes/offsets. Change a struct here and that test fails.
 */
#ifndef PLOW_CPU_DEV_H
#define PLOW_CPU_DEV_H

#include <stddef.h>
#include <stdint.h>
#include "dev_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel tiers, ordered; plow_cpu_init(cap) never activates above `cap`. */
enum {
    PLOW_CPU_ISA_SCALAR = 0,
    PLOW_CPU_ISA_AVX512 = 1, /* F/BW/VL + BF16 (+FP16/VNNI when present) */
    PLOW_CPU_ISA_AMX    = 2, /* AVX512 tier + AMX-TILE/BF16 (+INT8) */
};

/* Dispatch table extent: PLOW_DOP_* are dense small integers (< 200 today). */
#define PLOW_CPU_DOP_TABLE 256

/* Per-worker-thread context. Owned and zeroed by Rust; passed to every kernel.
 * `scratch` is a 64-byte-aligned per-thread arena of at least plow_cpu_scratch_bytes()
 * (flash softmax rows, GEMM C tiles, dequant staging). Fixed 64 bytes. */
typedef struct {
    void*    scratch;
    uint32_t scratch_bytes;
    uint32_t worker;    /* worker index in the pool */
    uint32_t node;      /* NUMA node the worker is pinned to */
    uint32_t isa;       /* active tier for this thread (== plow_cpu_isa()) */
    uint64_t reserved[5];
} PlowCpuCtx;

/* Kernel entry. `tensors[h]` is the host base pointer for handle h; an absent operand
 * (t[k] == PLOW_TENSOR_NONE, the u16 sentinel from dev_isa.h) reads as NULL via
 * PLOW_CPU_TEN(). */
typedef void (*plow_cpu_kernel_fn)(const PlowDevInst* in, uint32_t slice, uint32_t nblk,
                                   void* const* tensors, PlowCpuCtx* ctx);

#define PLOW_CPU_TEN(in, T, k) \
    ((in)->t[(k)] == PLOW_TENSOR_NONE ? (void*)0 : (T)[(in)->t[(k)]])

/* Process-wide init: cpuid detection, AMX permission (arch_prctl), table fill.
 * `isa_cap` is a PLOW_CPU_ISA_* ceiling. Idempotent. Returns the active tier (>= 0),
 * or a negative errno-style value if even the scalar tier cannot be set up. */
int plow_cpu_init(int isa_cap);

/* Active tier after init (PLOW_CPU_ISA_*), or -1 before init. */
int plow_cpu_isa(void);

/* Per-thread init on a worker: AMX tile config, ctx->isa. Call once per thread after
 * plow_cpu_init(), before the first kernel. Returns 0 or negative errno-style. */
int plow_cpu_thread_init(PlowCpuCtx* ctx);

/* Scratch bytes a worker must provide in ctx->scratch (multiple of 64). */
uint32_t plow_cpu_scratch_bytes(void);

/* Coverage: 1 if `op` has a kernel at the active tier (golden counts), else 0. */
int plow_cpu_has(uint16_t op);

/* Which tier's kernel `op` resolves to (PLOW_CPU_ISA_*), or -1 if none. Lets a caller
 * assert that a fast tier is actually live rather than a silent golden fallback. */
int plow_cpu_tier_of(uint16_t op);

/* Resolve `op` to its kernel, or NULL. Resolve once per program at load; do not
 * look up per packet. */
plow_cpu_kernel_fn plow_cpu_kernel(uint16_t op);

/* Convenience dispatch (lookup + call). 0 on success, -1 if `op` has no kernel. */
int plow_cpu_exec(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* tensors,
                  PlowCpuCtx* ctx);

/* --- Weight prepack (load time, off the hot path) ------------------------------- */

/* AMX/VNNI B layout for a bf16 weight W[n][k] (Linear: [out_features, in_features]):
 * [n/16][k/32][16 rows][32 bf16] with K pairs interleaved as TDPBF16PS consumes them.
 * n, k must be multiples of 16 / 32 (plowc pads). Bytes are the same as the source. */
size_t plow_cpu_prepack_bf16_b_bytes(uint32_t n, uint32_t k);
int    plow_cpu_prepack_bf16_b(void* dst, const void* src, uint32_t n, uint32_t k);

/* --- bf16 helpers shared by every tier (inline, no libm) ------------------------- */

typedef uint16_t plow_bf16;

static inline float plow_bf2f(plow_bf16 h) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

/* Round-to-nearest-even, matching the GPU kernels' __float2bfloat16. NaN keeps a
 * payload bit so it never collapses to Inf. */
static inline plow_bf16 plow_f2bf(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    if ((v.u & 0x7F800000u) == 0x7F800000u)
        return (plow_bf16)((v.u >> 16) | ((v.u & 0xFFFFu) ? 0x40u : 0u));
    uint32_t lsb = (v.u >> 16) & 1u;
    v.u += 0x7FFFu + lsb;
    return (plow_bf16)(v.u >> 16);
}

#ifdef __cplusplus
}
#endif

#endif /* PLOW_CPU_DEV_H */
