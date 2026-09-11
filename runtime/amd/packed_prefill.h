#ifndef PLOW_PACKED_PREFILL_H
#define PLOW_PACKED_PREFILL_H

#include "dev_isa.h"

#ifndef PLOW_PACKED_PREFILL_DENSE_CONSUMERS
#define PLOW_PACKED_PREFILL_DENSE_CONSUMERS 0
#endif
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
/* SLOT BAND (plans/unified-token-batch.md, "AMD TP8 lowering decision"). A token-batch BODY
 * program carries `prog->token_batch` (the shared PlowTokenBatch descriptor, dev_isa.h): rows
 * `[0, token_batch->sample_rows)` are a decode band indexed by KV slot — row t IS slot t — and
 * the spans start at that band. Under this axis the consumers resolve a band row to
 * (slot = row, position = token_batch->positions[row]) and a parked band row to INACTIVE, so a
 * mid-prefill slot's band row cannot clobber the KV row its own span writes in the same launch.
 * Off (the default) keeps every existing packed object's behaviour; the axis is compiled only
 * into the `_tb` family objects. */
#ifndef PLOW_PACKED_PREFILL_BAND
#define PLOW_PACKED_PREFILL_BAND 0
#endif
#ifndef PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS
#define PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS PLOW_PACKED_PREFILL_MLA_CONSUMERS
#endif
#define PLOW_PACKED_PREFILL_ANY_CONSUMERS                                      \
    (PLOW_PACKED_PREFILL_DENSE_CONSUMERS || PLOW_PACKED_PREFILL_MLA_NORM_CONSUMERS ||                                \
     PLOW_PACKED_PREFILL_MLA_FLASH_CONSUMERS || PLOW_PACKED_PREFILL_KDA_CONSUMERS)

typedef struct {
    const PlowPrefillSpan* span;
    uint32_t local_row;
    uint32_t active;
#if PLOW_PACKED_PREFILL_BAND
    uint32_t band;     /* 1: a slot-band row (no span); slot/position below are its address */
    uint32_t slot;
    uint32_t position;
#endif
} PlowPackedRow;
#if PLOW_PACKED_PREFILL_BAND
#define PLOW_PACKED_ROW_INIT {nullptr, 0u, 1u, 0u, 0u, 0u}
#else
#define PLOW_PACKED_ROW_INIT {nullptr, 0u, 1u}
#endif

/* The band width of a token-batch body, 0 for an ordinary packed program. */
__device__ __forceinline__ uint32_t plow_packed_prefill_band(const PlowProgram* prog) {
#if PLOW_PACKED_PREFILL_BAND
    return (prog && prog->token_batch) ? prog->token_batch->sample_rows : 0u;
#else
    (void)prog;
    return 0u;
#endif
}

__device__ __forceinline__ bool plow_packed_prefill_enabled(const PlowProgram* prog) {
#if !PLOW_PACKED_PREFILL_ANY_CONSUMERS
    (void)prog;
    return false;
#else
    if (!prog) return false;
    if (!prog->prefill_spans) {
        if (prog->prefill_parked || prog->n_prefill_spans || prog->n_prefill_rows
#if PLOW_PACKED_PREFILL_BAND
            || prog->token_batch
#endif
        )
            __builtin_trap();
        return false;
    }
#if PLOW_PACKED_PREFILL_BAND
    /* A body may carry NO spans (a band-only step); its descriptor stands in for the count. */
    const bool spans = prog->n_prefill_spans != 0u || prog->token_batch != nullptr;
#else
    const bool spans = prog->n_prefill_spans != 0u;
#endif
    const bool any = prog->prefill_spans || prog->prefill_parked || spans ||
                     prog->n_prefill_rows;
    const bool all = prog->prefill_spans && prog->prefill_parked && spans &&
                     prog->n_prefill_rows;
    if (any && !all) __builtin_trap();
    return all;
#endif
}

/* Resolve one dense activation row to its request span. Parked rows are returned inactive;
 * an active row without exactly one containing span is malformed metadata and traps. */
__device__ __forceinline__ PlowPackedRow plow_packed_prefill_row(const PlowProgram* prog,
                                                                 uint32_t row) {
    PlowPackedRow out = PLOW_PACKED_ROW_INIT;
    if (!plow_packed_prefill_enabled(prog)) return out;
    if (row >= prog->n_prefill_rows) __builtin_trap();
    if (prog->prefill_parked[row]) {
        out.active = 0u;
        return out;
    }
#if PLOW_PACKED_PREFILL_BAND
    if (row < plow_packed_prefill_band(prog)) {
        /* Row t is slot t; its position is the descriptor's. Parked band rows returned above. */
        out.band = 1u;
        out.slot = row;
        out.local_row = row;
        out.position = prog->token_batch->positions[row];
        return out;
    }
#endif
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

/* Whether the row's KV address comes from packed metadata (a span, or a band slot). */
__device__ __forceinline__ bool plow_packed_prefill_addressed(const PlowPackedRow& row) {
#if PLOW_PACKED_PREFILL_BAND
    return row.span != nullptr || row.band != 0u;
#else
    return row.span != nullptr;
#endif
}

__device__ __forceinline__ uint32_t plow_packed_prefill_slot(const PlowPackedRow& row) {
#if PLOW_PACKED_PREFILL_BAND
    return row.span ? row.span->slot : row.slot;
#else
    return row.span->slot;
#endif
}

__device__ __forceinline__ uint32_t plow_packed_prefill_position(const PlowPackedRow& row,
                                                                  uint32_t fallback) {
#if PLOW_PACKED_PREFILL_BAND
    if (row.band) return row.position;
#endif
    return row.span ? row.span->kv_row0 + row.local_row : fallback;
}

__device__ __forceinline__ size_t plow_packed_prefill_cache_row(const PlowPackedRow& row,
                                                                 uint32_t slot_stride,
                                                                 uint32_t fallback) {
#if PLOW_PACKED_PREFILL_BAND
    if (row.band) return (size_t)row.slot * slot_stride + row.position;
#endif
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
    if ((index == 0u && span->row0 != plow_packed_prefill_band(prog)) ||
        (index && (span - 1)->row0 + (span - 1)->n_rows != span->row0))
        __builtin_trap();
    if (prog->prefill_parked[span->row0] || prog->prefill_parked[end - 1u]) __builtin_trap();
    return span;
}

#endif
