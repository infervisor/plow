/* golden/gptoss.h — scalar reference kernels for the GPT-OSS family: MXFP4 dense GEMV (op 91,
 * gptoss.c) and the flat-tensor MXFP4 MoE ops 147-150 (moe.c). Contracts: dev_isa.h. */
#ifndef PLOW_CPU_GOLDEN_GPTOSS_H
#define PLOW_CPU_GOLDEN_GPTOSS_H

#include "golden.h"

/* gptoss.c: t0=C t1=x t2=W(fp4 [N][K/2]) t3=S(e8m0 [N][K/32])  i0=M i1=N i2=K */
G_K(g_gemv_mxfp4);

/* moe.c — routing table entry (dev_isa.h: {u32 eid, f32 gate}, PLOW_EXPERT_UNUSED = skip). */
typedef struct {
    uint32_t eid;
    float gate;
} plow_moe_route;

/* SLICE PARTITION of the MoE ops, shared by every tier (a future AMX tier must mirror it):
 * GV_BLOCKED (g_range) over the flat OUTPUT span, contiguous per slice —
 *   decode  (147/148): (slot, n) = B*k*N items, slot-major;
 *   prefill (149/150): (expert, n) = n_exp*N items, expert-major; a slice computes EVERY gathered
 *                      row of its (expert, column) pairs.
 * Consecutive output columns of one expert are consecutive weight rows (interleaved gate|up rows
 * for layout 0), so each slice streams one contiguous weight range per slot / expert. */
G_K(g_moe_glu_mx);
G_K(g_moe_down_mx);
G_K(g_moe_glu_mx_pf);
G_K(g_moe_down_mx_pf);
/* moe_route.c — routing ops 83/84/87 (op_moe.h ports); 84 runs on slice 0 only. */
G_K(g_moe_router_topk_pf);
G_K(g_moe_align_pf);
G_K(g_moe_combine_pf);

#endif /* PLOW_CPU_GOLDEN_GPTOSS_H */
