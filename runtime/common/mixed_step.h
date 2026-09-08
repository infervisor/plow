#ifndef PLOW_MIXED_STEP_H
#define PLOW_MIXED_STEP_H

#include "dev_isa.h"

#ifndef PLOW_MIXED_STEP
#define PLOW_MIXED_STEP 0
#endif

#if PLOW_MIXED_STEP
#if defined(__HIP_DEVICE_COMPILE__)
#define PLOW_MIXED_INLINE __device__ __attribute__((always_inline)) inline
#else
#define PLOW_MIXED_INLINE __device__ __forceinline__
#endif

PLOW_MIXED_INLINE void plow_mixed_trap(void) {
#if defined(__CUDA_ARCH__)
    asm volatile("trap;");
#else
    __builtin_trap();
#endif
}

typedef struct {
    const PlowPrefillSpan* span;
    uint32_t local_row;
    uint32_t slot;
    uint32_t position;
    uint32_t active;
} PlowMixedRow;

PLOW_MIXED_INLINE bool plow_mixed_step_enabled(const PlowProgram* prog) {
    if (!prog) return false;
    const bool any = prog->prefill_spans || prog->prefill_parked || prog->n_prefill_spans ||
                     prog->n_prefill_rows;
    const bool all = prog->prefill_spans && prog->prefill_parked && prog->n_prefill_spans &&
                     prog->n_prefill_rows;
    if (any != all) plow_mixed_trap();
    return all;
}

PLOW_MIXED_INLINE const PlowPrefillSpan* plow_mixed_prefill_span(
    const PlowProgram* prog, uint32_t index) {
    if (!plow_mixed_step_enabled(prog) || index >= prog->n_prefill_spans) plow_mixed_trap();
    const PlowPrefillSpan* span = prog->prefill_spans + index;
    const uint32_t end = span->row0 + span->n_rows;
    if (!span->n_rows || end < span->row0 || end > prog->n_prefill_rows) plow_mixed_trap();
    if ((index == 0u && span->row0 == 0u) ||
        (index && (span - 1)->row0 + (span - 1)->n_rows != span->row0) ||
        prog->prefill_parked[span->row0] || prog->prefill_parked[end - 1u])
        plow_mixed_trap();
    return span;
}

PLOW_MIXED_INLINE PlowMixedRow plow_mixed_row(const PlowProgram* prog,
                                                        const int* decode_slots,
                                                        const int* positions, uint32_t row) {
    PlowMixedRow out = {nullptr, 0u, 0u, 0u, 0u};
    const PlowPrefillSpan* first = plow_mixed_prefill_span(prog, 0u);
    if (row >= prog->n_prefill_rows) plow_mixed_trap();
    if (prog->prefill_parked[row]) return out;
    if (row < first->row0) {
        if (!decode_slots || !positions || decode_slots[row] < 0) plow_mixed_trap();
        out.slot = (uint32_t)decode_slots[row];
        out.position = (uint32_t)positions[row];
        out.active = 1u;
        return out;
    }

    uint32_t lo = 0u, hi = prog->n_prefill_spans;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (prog->prefill_spans[mid].row0 <= row)
            lo = mid + 1u;
        else
            hi = mid;
    }
    if (!lo) plow_mixed_trap();
    const PlowPrefillSpan* span = plow_mixed_prefill_span(prog, lo - 1u);
    const uint32_t local = row - span->row0;
    if (local >= span->n_rows) plow_mixed_trap();
    const uint32_t position = span->kv_row0 + local;
    if (position < span->kv_row0 || (positions && (uint32_t)positions[row] != position))
        plow_mixed_trap();
    out.span = span;
    out.local_row = local;
    out.slot = span->slot;
    out.position = position;
    out.active = 1u;
    return out;
}
#undef PLOW_MIXED_INLINE
#endif

#endif
