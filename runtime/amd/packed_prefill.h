#ifndef PLOW_PACKED_PREFILL_H
#define PLOW_PACKED_PREFILL_H

#include "dev_isa.h"

#ifndef PLOW_PACKED_PREFILL_CONSUMERS
#define PLOW_PACKED_PREFILL_CONSUMERS 0
#endif
#ifndef PLOW_PACKED_PREFILL_MLA_CONSUMERS
#define PLOW_PACKED_PREFILL_MLA_CONSUMERS PLOW_PACKED_PREFILL_CONSUMERS
#endif
#ifndef PLOW_PACKED_PREFILL_KDA_CONSUMERS
#define PLOW_PACKED_PREFILL_KDA_CONSUMERS PLOW_PACKED_PREFILL_CONSUMERS
#endif
#ifndef PLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS
#define PLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS PLOW_PACKED_PREFILL_MLA_CONSUMERS
#endif
#ifndef PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
#define PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS PLOW_PACKED_PREFILL_MLA_CONSUMERS
#endif
#define PLOW_PACKED_PREFILL_ANY_CONSUMERS                                      \
    (PLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS ||                                \
     PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS || PLOW_PACKED_PREFILL_KDA_CONSUMERS)

typedef struct {
    const PlowPrefillSpan* span;
    uint32_t local_row;
    uint32_t active;
} PlowPackedRow;

__device__ __forceinline__ bool plow_packed_prefill_enabled(const PlowProgram* prog) {
#if !PLOW_PACKED_PREFILL_ANY_CONSUMERS
    (void)prog;
    return false;
#else
    if (!prog) return false;
    if (!prog->prefill_spans) {
        if (prog->prefill_parked || prog->n_prefill_spans || prog->n_prefill_rows)
            __builtin_trap();
        return false;
    }
    const bool any = prog->prefill_spans || prog->prefill_parked || prog->n_prefill_spans ||
                     prog->n_prefill_rows;
    const bool all = prog->prefill_spans && prog->prefill_parked && prog->n_prefill_spans &&
                     prog->n_prefill_rows;
    if (any && !all) __builtin_trap();
    return all;
#endif
}

/* Resolve one dense activation row to its request span. Parked rows are returned inactive;
 * an active row without exactly one containing span is malformed metadata and traps. */
__device__ __forceinline__ PlowPackedRow plow_packed_prefill_row(const PlowProgram* prog,
                                                                 uint32_t row) {
    PlowPackedRow out = {nullptr, 0u, 1u};
    if (!plow_packed_prefill_enabled(prog)) return out;
    if (row >= prog->n_prefill_rows) __builtin_trap();
    if (prog->prefill_parked[row]) {
        out.active = 0u;
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
    if (!lo) __builtin_trap();
    const PlowPrefillSpan* span = prog->prefill_spans + lo - 1u;
    const uint32_t end = span->row0 + span->n_rows;
    if (end < span->row0 || row >= end) __builtin_trap();
    if (lo > 1u) {
        const PlowPrefillSpan* prior = span - 1;
        if (prior->row0 + prior->n_rows > span->row0) __builtin_trap();
    }
    out.span = span;
    out.local_row = row - span->row0;
    return out;
}

__device__ __forceinline__ uint32_t plow_packed_prefill_position(const PlowPackedRow& row,
                                                                  uint32_t fallback) {
    return row.span ? row.span->kv_row0 + row.local_row : fallback;
}

__device__ __forceinline__ size_t plow_packed_prefill_cache_row(const PlowPackedRow& row,
                                                                 uint32_t slot_stride,
                                                                 uint32_t fallback) {
    return row.span ? (size_t)row.span->slot * slot_stride +
                          row.span->kv_row0 + row.local_row
                    : fallback;
}

__device__ __forceinline__ const PlowPrefillSpan* plow_packed_prefill_span(
    const PlowProgram* prog, uint32_t index) {
    if (!plow_packed_prefill_enabled(prog) || index >= prog->n_prefill_spans) __builtin_trap();
    const PlowPrefillSpan* span = prog->prefill_spans + index;
    if (!span->n_rows) __builtin_trap();
    const uint32_t end = span->row0 + span->n_rows;
    if (end < span->row0 || end > prog->n_prefill_rows) __builtin_trap();
    if ((index == 0u && span->row0 != 0u) ||
        (index && (span - 1)->row0 + (span - 1)->n_rows != span->row0))
        __builtin_trap();
    if (prog->prefill_parked[span->row0] || prog->prefill_parked[end - 1u]) __builtin_trap();
    return span;
}

#endif
