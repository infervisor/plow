#ifndef PLOW_TOKEN_BATCH_H
#define PLOW_TOKEN_BATCH_H

/* Unified token batch — the AMD consumer side (plans/unified-token-batch.md §4.2/§4.3/§5).
 *
 * =====================================================================================
 * ASSUMED DESCRIPTOR LAYOUT. Read this before changing anything here.
 * =====================================================================================
 *
 * The shared contract (`PlowTokenBatch` in runtime/common/dev_isa.h, its `packet` mirror,
 * the ABI-lock test and the shared row resolver beside runtime/common/mixed_step.h) is
 * owned by the Phase 1a/1b/1c work and IS NOT LANDED AT THE TIME THIS FILE WAS WRITTEN.
 * This header is the AMD-local adapter that lets the Phase 2 dense-GQA route be built and
 * tested against it. It assumes:
 *
 *     typedef struct {
 *         uint32_t version;            // 1
 *         uint32_t row_capacity;       // compiled/padded allocation capacity
 *         uint32_t real_rows;          // M: scheduled input tokens this step
 *         uint32_t sample_rows;        // S
 *         uint32_t n_spans;            // R
 *         uint32_t sample_capacity;
 *         const PlowPrefillSpan* spans;             // [n_spans], covering EXACTLY [0, M)
 *         const int32_t*         positions;         // [M]
 *         const int32_t*         active;            // [row_capacity], 1 = live row
 *         const uint32_t*        sample_input_rows; // [S]
 *     } PlowTokenBatch;
 *
 * and that `PlowProgram` gains one appended `const PlowTokenBatch* token_batch;`, NULL
 * meaning "old semantics".
 *
 * When that lands, define PLOW_TOKEN_BATCH_DESC=1 and this header reads it directly. Until
 * then it synthesises the SAME view from the packed-prefill tail that PlowProgram already
 * carries (`prefill_spans` / `prefill_parked` / `n_prefill_spans` / `n_prefill_rows`), which
 * holds every field the dense-GQA family needs:
 *
 *     positions[row] = span->kv_row0 + (row - span->row0)      (§4.4's own invariant)
 *     active[row]    = !prefill_parked[row]
 *     real_rows      = last span's end
 *
 * TWO DIFFERENCES FROM `runtime/common/mixed_step.h`, both deliberate and both from §4.3:
 *
 *   1. THERE IS NO DECODE PREFIX. Every row belongs to a span, decode spans included, and
 *      the spans cover exactly [0, M). `plow_mixed_prefill_span` TRAPS when
 *      `span[0].row0 == 0` — it *requires* a nonempty decode band ahead of the spans. That
 *      is the old mixed-step convention, not this contract, so this header cannot reuse it.
 *      (`runtime/amd/packed_prefill.h` has the opposite convention again: it requires
 *      `span[0].row0 == 0` and has no decode rows at all. Three conventions over one span
 *      table is exactly the drift §4.3 exists to stop.)
 *
 *   2. THE LIVE ROW COUNT IS `real_rows`, NOT the compiled capacity. A row in
 *      [real_rows, row_capacity) is padding: it is parked, it never writes KV, and it never
 *      produces output.
 *
 * The defensive checks below stay ON in release builds, per §4.3: a mis-planned batch must
 * trap, not produce a plausible wrong answer. On AMD in particular the interpreter's
 * dispatch `default:` writes nothing and does not trap, so silence is the failure mode this
 * whole route has to design against.
 */

#include "dev_isa.h"

#ifndef PLOW_TOKEN_BATCH
#define PLOW_TOKEN_BATCH 0
#endif

/* 1 once the shared `PlowTokenBatch` descriptor is appended to PlowProgram. */
#ifndef PLOW_TOKEN_BATCH_DESC
#define PLOW_TOKEN_BATCH_DESC 0
#endif

#if PLOW_TOKEN_BATCH

#if defined(__HIP_DEVICE_COMPILE__)
#define PLOW_TB_INLINE __device__ __attribute__((always_inline)) inline
#else
#define PLOW_TB_INLINE __device__ __forceinline__
#endif

PLOW_TB_INLINE void plow_tb_trap(void) {
#if defined(__CUDA_ARCH__)
    asm volatile("trap;");
#else
    __builtin_trap();
#endif
}

/* One step's row map, resolved once per consumer. Pointer-sized and trivially copyable so a
 * caller can hold it across a work loop without spending registers on the program struct. */
typedef struct {
    const PlowPrefillSpan* spans;
    const int* positions; /* nullptr when the descriptor does not carry them */
    const unsigned* parked; /* 1 = padding; the complement of `active` */
    unsigned n_spans;
    unsigned real_rows;    /* M */
    unsigned row_capacity;
} PlowTokenBatchView;

typedef struct {
    const PlowPrefillSpan* span;
    unsigned local_row;
    unsigned slot;
    unsigned state_slot;
    unsigned position;
    unsigned active;
} PlowTokenBatchRow;

PLOW_TB_INLINE bool plow_tb_enabled(const PlowProgram* prog) {
    if (!prog) return false;
#if PLOW_TOKEN_BATCH_DESC
    return prog->token_batch != nullptr;
#else
    const bool any = prog->prefill_spans || prog->prefill_parked || prog->n_prefill_spans ||
                     prog->n_prefill_rows;
    const bool all = prog->prefill_spans && prog->prefill_parked && prog->n_prefill_spans &&
                     prog->n_prefill_rows;
    if (any != all) plow_tb_trap();
    return all;
#endif
}

/* Structural validation of the whole table, done once per consumer rather than per row:
 * dense cover of [0, M), monotone row0, no zero-length span, `kv_row0 + n_rows == kv_len`,
 * and M within the compiled capacity. §4.4's invariants, enforced rather than assumed. */
PLOW_TB_INLINE PlowTokenBatchView plow_tb_view(const PlowProgram* prog) {
    PlowTokenBatchView v = {nullptr, nullptr, nullptr, 0u, 0u, 0u};
    if (!plow_tb_enabled(prog)) plow_tb_trap();
#if PLOW_TOKEN_BATCH_DESC
    const PlowTokenBatch* d = prog->token_batch;
    if (d->version != 1u) plow_tb_trap();
    v.spans = d->spans;
    v.positions = (const int*)d->positions;
    v.parked = nullptr; /* the descriptor carries `active`; see plow_tb_row */
    v.n_spans = d->n_spans;
    v.real_rows = d->real_rows;
    v.row_capacity = d->row_capacity;
    if (!v.spans || !v.n_spans || !v.real_rows) plow_tb_trap();
#else
    v.spans = prog->prefill_spans;
    v.positions = nullptr;
    v.parked = prog->prefill_parked;
    v.n_spans = prog->n_prefill_spans;
    v.row_capacity = prog->n_prefill_rows;
    v.real_rows = 0u; /* derived from the cover below */
#endif
    if (v.row_capacity == 0u || v.n_spans == 0u) plow_tb_trap();

    unsigned expect = 0u;
    for (unsigned i = 0; i < v.n_spans; ++i) {
        const PlowPrefillSpan* s = v.spans + i;
        if (!s->n_rows) plow_tb_trap();                 /* no zero-length entries */
        if (s->row0 != expect) plow_tb_trap();          /* dense cover, monotone row0 */
        const unsigned end = s->row0 + s->n_rows;
        if (end < s->row0 || end > v.row_capacity) plow_tb_trap();
        /* §4.4: a span starts at its committed frontier and ends at the new one. */
        if (s->kv_row0 + s->n_rows != s->kv_len) plow_tb_trap();
        expect = end;
    }
#if PLOW_TOKEN_BATCH_DESC
    if (expect != v.real_rows) plow_tb_trap();
#else
    v.real_rows = expect;
#endif
    if (v.real_rows > v.row_capacity) plow_tb_trap();
    return v;
}

PLOW_TB_INLINE const PlowPrefillSpan* plow_tb_span(const PlowTokenBatchView& v, unsigned i) {
    if (i >= v.n_spans) plow_tb_trap();
    return v.spans + i;
}

/* Resolve one packed activation row. Padding is returned inactive; a live row whose derived
 * position disagrees with the descriptor's own `positions[]` traps rather than attending at
 * another request's sequence length — that disagreement IS the class-C hazard. */
PLOW_TB_INLINE PlowTokenBatchRow plow_tb_row(const PlowTokenBatchView& v, unsigned row) {
    PlowTokenBatchRow out = {nullptr, 0u, 0u, 0u, 0u, 0u};
    if (row >= v.row_capacity) plow_tb_trap();
    if (row >= v.real_rows) return out; /* padded tail: inactive, writes nothing */
    if (v.parked && v.parked[row]) return out;

    unsigned lo = 0u, hi = v.n_spans;
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo) / 2u;
        if (v.spans[mid].row0 <= row)
            lo = mid + 1u;
        else
            hi = mid;
    }
    if (!lo) plow_tb_trap();
    const PlowPrefillSpan* s = v.spans + lo - 1u;
    const unsigned local = row - s->row0;
    if (local >= s->n_rows) plow_tb_trap();
    const unsigned position = s->kv_row0 + local;
    if (v.positions && (unsigned)v.positions[row] != position) plow_tb_trap();
    out.span = s;
    out.local_row = local;
    out.slot = s->slot;
    out.state_slot = s->state_slot;
    out.position = position;
    out.active = 1u;
    return out;
}

/* THE ATTENTION PARTITION, defined exactly once.
 *
 * §4.1 forbids the HOST PLAN from classifying a span as decode because its length is one — a
 * final prefill chunk can also have length one. It explicitly allows the other thing: "Kernel
 * selection derives query geometry from span lengths." This is that, and only that: which of
 * two attention kernels covers a span, on a route where both are numerically the same function.
 * A one-row span at frontier p attends over [0, p+1) with its query at p whichever kernel runs
 * it, so a final chunk of length one landing here is CORRECT, not merely tolerated.
 *
 * Spans are decode-first ordered (§4.1), so the partition is a prefix and both consumers agree
 * on it by construction. ndec == 0 (pure prefill) and ndec == n_spans (pure decode) are both
 * legal and both write nothing on the other side — which is how this route serves an
 * intermediate pure-prefill step that the mixed-step resolver refuses outright. */
PLOW_TB_INLINE unsigned plow_tb_decode_spans(const PlowTokenBatchView& v) {
    unsigned n = 0;
    while (n < v.n_spans && v.spans[n].n_rows == 1u) ++n;
    return n;
}

/* A contiguous slice of the span table, for a consumer that owns only part of the partition.
 * `real_rows`/`row_capacity` are carried through unchanged: they describe the BATCH, not the
 * slice, and a consumer that needs the batch extent still gets it. */
PLOW_TB_INLINE PlowTokenBatchView plow_tb_span_range(const PlowTokenBatchView& v, unsigned lo,
                                                     unsigned hi) {
    if (lo > hi || hi > v.n_spans) plow_tb_trap();
    PlowTokenBatchView o = v;
    o.spans = v.spans + lo;
    o.n_spans = hi - lo;
    return o;
}

/* Live rows for a packet compiled at `capacity`. Class-A operators over the whole batch need
 * exactly this and nothing else (§5.1); an attention-partition packet compiled at the decode
 * capacity gets the live decode-span count. Any other capacity is a mis-planned batch. */
PLOW_TB_INLINE unsigned plow_tb_rows(const PlowProgram* prog, unsigned capacity) {
    const PlowTokenBatchView v = plow_tb_view(prog);
    if (capacity == v.row_capacity) return v.real_rows;
    const unsigned ndec = plow_tb_decode_spans(v);
    if (capacity < ndec || capacity >= v.row_capacity) plow_tb_trap();
    return ndec;
}

#undef PLOW_TB_INLINE
#endif /* PLOW_TOKEN_BATCH */

#endif /* PLOW_TOKEN_BATCH_H */
