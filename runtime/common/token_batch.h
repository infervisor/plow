/* token_batch.h — the ONE row resolver for the unified token batch.
 *
 * row -> (span, local_row, slot, state_slot, position, active)
 *
 * Compiled by HIP and CUDA under the same guards `mixed_step.h` uses, and by the HOST for the
 * CPU interpreter, which is why this is a header and not three binary searches over one span
 * table. Three independent searches is how backends drift; a drifted search does not fault, it
 * hands one request's hidden state to another request's next token.
 *
 * IT IS NOT `plow_mixed_row`. That resolver has a DECODE PREFIX: rows below the first span are
 * decode rows resolved from a separate `decode_slots[]` image. Here every row belongs to a span,
 * decode spans included, and there is no second row source to keep in sync.
 *
 * THE DEFENSIVE CHECKS STAY ON IN RELEASE. Dense cover, monotone row0, position agreement and
 * the park mask are all cheap next to the work of any packet that calls this, and each one turns
 * a mis-planned batch into a trap instead of a plausible wrong answer. On AMD the interpreter's
 * dispatch `default:` neither writes nor traps, so "plausible wrong answer" is this stack's
 * actual failure mode, not a hypothetical.
 *
 * HOST TWIN. Define PLOW_TOKEN_BATCH_HOST before including to get `static inline` host
 * functions. The host build additionally exposes `plow_token_batch_validate`, which performs
 * exactly the checks the device resolver traps on and RETURNS an error code instead — so a host
 * can refuse a malformed plan before launch and a test can compare refusals against the Rust
 * twin without aborting the process.
 */
#ifndef PLOW_TOKEN_BATCH_H
#define PLOW_TOKEN_BATCH_H

#include "dev_isa.h"

#if defined(__HIP_DEVICE_COMPILE__)
#define PLOW_TB_INLINE __device__ __attribute__((always_inline)) inline
#define PLOW_TB_DEVICE 1
#elif defined(__CUDA_ARCH__)
#define PLOW_TB_INLINE __device__ __forceinline__
#define PLOW_TB_DEVICE 1
#elif defined(__HIPCC__) || defined(__CUDACC__)
/* Host pass of a device compile: the arms are compiled but never called. */
#define PLOW_TB_INLINE __host__ __device__ inline
#define PLOW_TB_DEVICE 0
#else
#define PLOW_TB_INLINE static inline
#define PLOW_TB_DEVICE 0
#endif

#if defined(__cplusplus)
#define PLOW_TB_NULL nullptr
#else
#define PLOW_TB_NULL ((void*)0)
#endif

PLOW_TB_INLINE void plow_token_trap(void) {
#if defined(__CUDA_ARCH__)
    asm volatile("trap;");
#elif defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap();
#else
    __builtin_trap();
#endif
}

typedef struct {
    const PlowPrefillSpan* span; /* NULL for a padding row */
    uint32_t local_row;          /* row - span->row0 */
    uint32_t slot;               /* physical KV slot */
    uint32_t state_slot;         /* carried-state slot (D-class families) */
    uint32_t position;           /* absolute position within the owning request */
    uint32_t active;             /* 1 = live row, 0 = padding */
} PlowTokenRow;

/* Refusal codes. Negative so a caller can `if (rc < 0)`; each names ONE fault, because a
 * refusal that does not name what it refused is indistinguishable from a crash. */
enum {
    PLOW_TB_OK = 0,
    PLOW_TB_E_NULL = -1,       /* descriptor or a required array is NULL */
    PLOW_TB_E_VERSION = -2,    /* descriptor version this build does not implement */
    PLOW_TB_E_FLAGS = -3,      /* reserved flags set */
    PLOW_TB_E_CAPACITY = -4,   /* real_rows > row_capacity, or a zero capacity */
    PLOW_TB_E_SPANS = -5,      /* zero spans with live rows, or too many */
    PLOW_TB_E_COVER = -6,      /* gap, overlap, zero length, or a short/long cover of [0, M) */
    PLOW_TB_E_POSITION = -7,   /* positions[] disagrees with the span's arithmetic */
    PLOW_TB_E_ACTIVE = -8,     /* park mask disagrees with the span cover */
    PLOW_TB_E_KVLEN = -9,      /* kv_len != kv_row0 + n_rows */
    PLOW_TB_E_SAMPLE = -10,    /* a sample index is not a live row */
    PLOW_TB_E_ROW = -11,       /* row index outside row_capacity */
};

PLOW_TB_INLINE int plow_token_batch_present(const PlowProgram* prog) {
    return prog && prog->token_batch ? 1 : 0;
}

/* Structural checks a resolver relies on, without touching the row arrays. Cheap enough to run
 * at every entry point; `plow_token_batch_validate` below is the full O(M + R) sweep. */
PLOW_TB_INLINE int plow_token_batch_check(const PlowTokenBatch* tb) {
    if (!tb) return PLOW_TB_E_NULL;
    if (tb->version != PLOW_TOKEN_BATCH_VERSION) return PLOW_TB_E_VERSION;
    if (tb->flags || tb->_pad0 || tb->_pad1) return PLOW_TB_E_FLAGS;
    if (!tb->row_capacity || tb->real_rows > tb->row_capacity) return PLOW_TB_E_CAPACITY;
    if (!tb->input_ids || !tb->positions || !tb->active) return PLOW_TB_E_NULL;
    if (tb->real_rows && (!tb->spans || !tb->n_spans)) return PLOW_TB_E_SPANS;
    if (!tb->real_rows && tb->n_spans) return PLOW_TB_E_SPANS;
    if (tb->n_spans > tb->real_rows) return PLOW_TB_E_COVER;
    if (tb->sample_rows && !tb->sample_rows_idx) return PLOW_TB_E_NULL;
    if (tb->sample_rows > tb->real_rows) return PLOW_TB_E_SAMPLE;
    return PLOW_TB_OK;
}

/* Span `index`, with its own coverage invariants proven before it is returned: nonzero length,
 * no wrap, inside [0, M), starting at 0 or exactly where its predecessor ended, and consistent
 * KV arithmetic. */
PLOW_TB_INLINE const PlowPrefillSpan* plow_token_span(const PlowTokenBatch* tb, uint32_t index) {
    if (plow_token_batch_check(tb) != PLOW_TB_OK || index >= tb->n_spans) plow_token_trap();
    const PlowPrefillSpan* span = tb->spans + index;
    const uint32_t end = span->row0 + span->n_rows;
    if (!span->n_rows || end < span->row0 || end > tb->real_rows) plow_token_trap();
    if (index == 0u ? span->row0 != 0u
                    : (tb->spans[index - 1u].row0 + tb->spans[index - 1u].n_rows) != span->row0)
        plow_token_trap();
    if (span->kv_len != span->kv_row0 + span->n_rows || span->kv_len < span->kv_row0)
        plow_token_trap();
    if (index + 1u == tb->n_spans && end != tb->real_rows) plow_token_trap();
    return span;
}

/* Resolve one packed row. A padding row returns `active == 0` and a NULL span; every other
 * disagreement traps. */
PLOW_TB_INLINE PlowTokenRow plow_token_row(const PlowTokenBatch* tb, uint32_t row) {
    PlowTokenRow out;
    out.span = (const PlowPrefillSpan*)PLOW_TB_NULL;
    out.local_row = 0u;
    out.slot = 0u;
    out.state_slot = 0u;
    out.position = 0u;
    out.active = 0u;
    if (plow_token_batch_check(tb) != PLOW_TB_OK) plow_token_trap();
    if (row >= tb->row_capacity) plow_token_trap();
    if (row >= tb->real_rows) {
        /* Padding. The mask must agree with the span cover; a live mask bit outside [0, M)
         * means the filler and the planner disagree, which is exactly the class of bug that
         * writes KV for a row nobody owns. */
        if (tb->active[row]) plow_token_trap();
        return out;
    }
    if (!tb->active[row]) plow_token_trap();

    uint32_t lo = 0u, hi = tb->n_spans;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (tb->spans[mid].row0 <= row)
            lo = mid + 1u;
        else
            hi = mid;
    }
    if (!lo) plow_token_trap();
    const PlowPrefillSpan* span = plow_token_span(tb, lo - 1u);
    const uint32_t local = row - span->row0;
    if (local >= span->n_rows) plow_token_trap();
    const uint32_t position = span->kv_row0 + local;
    if (position < span->kv_row0 || tb->positions[row] != position) plow_token_trap();
    out.span = span;
    out.local_row = local;
    out.slot = span->slot;
    out.state_slot = span->state_slot;
    out.position = position;
    out.active = 1u;
    return out;
}

/* The sample list is separate from the row map on purpose: it is indices INTO the body's hidden
 * rows, not rows of the batch, and it is what the terminal segment gathers. */
PLOW_TB_INLINE uint32_t plow_token_sample_row(const PlowTokenBatch* tb, uint32_t s) {
    if (plow_token_batch_check(tb) != PLOW_TB_OK || s >= tb->sample_rows) plow_token_trap();
    const uint32_t row = tb->sample_rows_idx[s];
    if (row >= tb->real_rows || !tb->active[row]) plow_token_trap();
    return row;
}

#if !PLOW_TB_DEVICE
/* Full O(M + R) sweep, host only. Same predicates as the device resolver, returned rather than
 * trapped, so a host refuses a malformed plan BEFORE launch and names which invariant failed. */
static inline int plow_token_batch_validate(const PlowTokenBatch* tb) {
    int rc = plow_token_batch_check(tb);
    if (rc != PLOW_TB_OK) return rc;

    uint32_t next = 0u;
    for (uint32_t i = 0; i < tb->n_spans; i++) {
        const PlowPrefillSpan* span = tb->spans + i;
        const uint32_t end = span->row0 + span->n_rows;
        if (!span->n_rows || end < span->row0 || end > tb->real_rows) return PLOW_TB_E_COVER;
        if (span->row0 != next) return PLOW_TB_E_COVER;
        if (span->kv_len != span->kv_row0 + span->n_rows || span->kv_len < span->kv_row0)
            return PLOW_TB_E_KVLEN;
        for (uint32_t j = 0; j < span->n_rows; j++) {
            const uint32_t row = span->row0 + j;
            if (!tb->active[row]) return PLOW_TB_E_ACTIVE;
            if (tb->positions[row] != span->kv_row0 + j) return PLOW_TB_E_POSITION;
        }
        next = end;
    }
    if (next != tb->real_rows) return PLOW_TB_E_COVER;
    for (uint32_t row = tb->real_rows; row < tb->row_capacity; row++)
        if (tb->active[row]) return PLOW_TB_E_ACTIVE;
    for (uint32_t s = 0; s < tb->sample_rows; s++) {
        const uint32_t row = tb->sample_rows_idx[s];
        if (row >= tb->real_rows || !tb->active[row]) return PLOW_TB_E_SAMPLE;
    }
    return PLOW_TB_OK;
}

/* Host-callable resolver that reports instead of trapping: `out` is filled only when the
 * return code is PLOW_TB_OK. Used by the CPU interpreter and by the cross-language row-resolver
 * test, which compares this against the Rust twin over identical span tables. */
static inline int plow_token_row_checked(const PlowTokenBatch* tb, uint32_t row,
                                         PlowTokenRow* out) {
    int rc = plow_token_batch_validate(tb);
    if (rc != PLOW_TB_OK) return rc;
    if (row >= tb->row_capacity) return PLOW_TB_E_ROW;
    *out = plow_token_row(tb, row);
    return PLOW_TB_OK;
}
#endif /* !PLOW_TB_DEVICE */

#undef PLOW_TB_INLINE
#undef PLOW_TB_NULL

#endif /* PLOW_TOKEN_BATCH_H */
