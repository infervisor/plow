#include "interp.h"
#include "decode.h"
#include <stdlib.h>

#define PLOW_MAX_INSTS 65536

int plow_interp_run(const uint8_t* buf, size_t len, const dispatch_table* dt,
                    kctx* ctx, const PlowBinding* bindings, uint32_t n_bindings) {
    PlowInst* insts = (PlowInst*)calloc(PLOW_MAX_INSTS, sizeof(PlowInst));
    if (!insts) return -3;

    uint32_t n_insts = 0, n_counters = 0;
    const PlowCounter* counters = NULL;
    int rc = plow_decode(buf, len, insts, PLOW_MAX_INSTS, &n_insts,
                         &counters, &n_counters, NULL, NULL, NULL);
    if (rc != 0) { free(insts); return -3; }

    /* threshold lookup by counter id (ids are small, dense in practice). */
    for (uint32_t i = 0; i < n_insts; i++) {
        const PlowInst* in = &insts[i];

        /* Gate: every wait counter must have reached its threshold. In a valid
         * issue order this already holds for a serial walk. */
        for (uint8_t w = 0; w < in->wait_len; w++) {
            uint32_t cid = in->wait[w];
            uint32_t thr = 0;
            for (uint32_t c = 0; c < n_counters; c++)
                if (counters[c].id == cid) { thr = counters[c].threshold; break; }
            if (cid >= ctx->n_counters || ctx->counters[cid] < thr) {
                free(insts);
                return -2;
            }
        }

        ctx->bind = (bindings && in->index < n_bindings) ? &bindings[in->index] : NULL;
        /* Every family except CONTROL (nop/host-coord) needs a binding to locate
         * its operands. A missing binding would otherwise make the kernel a
         * silent no-op that still reports success, so fail loud here. */
        if (plow_op_family(in->opcode) != PLOW_FAMILY_CONTROL && !ctx->bind) {
            free(insts);
            return -4;
        }
        if (dt_dispatch(dt, in->opcode, in->body, ctx) != 0) {
            free(insts);
            return -1;
        }

        /* Signal: bump every successor counter. */
        for (uint8_t s = 0; s < in->succ_len; s++) {
            uint32_t cid = in->succ[s];
            if (cid < ctx->n_counters) ctx->counters[cid]++;
        }
    }

    free(insts);
    return 0;
}
