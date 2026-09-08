/* token_batch.c — the HOST twin of the shared token-batch row resolver.
 *
 * `runtime/common/token_batch.h` is the one resolver: HIP and CUDA compile it as device code
 * and trap on a mis-planned batch; the host compiles the same source and gets `_validate` /
 * `_checked`, which return the same predicates as error codes instead of aborting the process.
 *
 * This file exists so the CPU interpreter and the Rust runtime reach that source through a
 * stable C entry point rather than re-deriving the search. `crates/plow-asset` also carries a
 * Rust twin — the planner has to validate a plan before any device sees it — and
 * `crates/plowrt/tests/token_batch_resolver.rs` runs both over identical span tables, including
 * the malformed ones (gap, overlap, zero length, position mismatch, live mask past M). Two
 * implementations that are never compared are two implementations.
 *
 * The row is flattened to `PlowTokenRowFlat` on the way out: the device struct holds a POINTER
 * to the span, which is meaningless across the FFI boundary, so the wrapper reports the span's
 * INDEX. Everything else is copied through unchanged.
 */
#include "cpu_dev_internal.h"
#include "token_batch.h"

int plow_token_batch_validate_host(const PlowTokenBatch* tb) {
    return plow_token_batch_validate(tb);
}

int plow_token_row_host(const PlowTokenBatch* tb, uint32_t row, PlowTokenRowFlat* out) {
    if (!out) return PLOW_TB_E_NULL;
    PlowTokenRow r;
    const int rc = plow_token_row_checked(tb, row, &r);
    if (rc != PLOW_TB_OK) return rc;
    out->span = r.span ? (uint32_t)(r.span - tb->spans) : PLOW_TB_SPAN_NONE;
    out->local_row = r.local_row;
    out->slot = r.slot;
    out->state_slot = r.state_slot;
    out->position = r.position;
    out->active = r.active;
    return PLOW_TB_OK;
}

int plow_token_sample_row_host(const PlowTokenBatch* tb, uint32_t s, uint32_t* out) {
    if (!out) return PLOW_TB_E_NULL;
    const int rc = plow_token_batch_validate(tb);
    if (rc != PLOW_TB_OK) return rc;
    if (s >= tb->sample_rows) return PLOW_TB_E_SAMPLE;
    *out = plow_token_sample_row(tb, s);
    return PLOW_TB_OK;
}

uint32_t plow_token_batch_descriptor_version(void) { return PLOW_TOKEN_BATCH_VERSION; }
