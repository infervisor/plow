/* interp.h — host serial interpreter for a `.pkt` stream.
 *
 * Walks instructions in issue order, gating each on its wait-counter thresholds
 * and incrementing successor counters on completion. This is the host reference
 * for the on-device persistent-kernel interpreter (out of scope here); it runs
 * the CPU golden kernels to execute a whole program.
 */
#ifndef PLOW_INTERP_H
#define PLOW_INTERP_H

#include <stddef.h>
#include <stdint.h>
#include "dispatch.h"
#include "kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run the program in `buf` against `dt`. `ctx->counters` must have room for the
 * stream's counter ids (zeroed by the caller). `bindings[i]` supplies operand
 * info for instruction i (may be NULL to run dispatch-coverage only).
 * Returns 0 on success, -1 missing kernel, -2 unsatisfied wait (bad order),
 * -3 decode error, -4 a non-control instruction is missing its binding. */
int plow_interp_run(const uint8_t* buf, size_t len, const dispatch_table* dt,
                    kctx* ctx, const PlowBinding* bindings, uint32_t n_bindings);

#ifdef __cplusplus
}
#endif

#endif /* PLOW_INTERP_H */
