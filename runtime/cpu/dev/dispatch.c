#include "cpu_dev_internal.h"
#include <string.h>

static plow_cpu_kernel_fn g_tab[PLOW_CPU_DOP_TABLE];
static int8_t g_tier[PLOW_CPU_DOP_TABLE];

plow_cpu_kernel_fn* plow_cpu_table(void) { return g_tab; }

void plow_cpu_table_reset(void) {
    memset(g_tab, 0, sizeof(g_tab));
    memset(g_tier, -1, sizeof(g_tier));
}

/* Called by plow_cpu_init after each registrar: every entry that changed since the
 * previous pass belongs to `tier`. */
void plow_cpu_table_mark(const plow_cpu_kernel_fn* before, int tier) {
    for (int op = 0; op < PLOW_CPU_DOP_TABLE; op++)
        if (g_tab[op] != before[op]) g_tier[op] = (int8_t)tier;
}

void plow_cpu_table_mark_one(uint16_t op, int tier) {
    if (op < PLOW_CPU_DOP_TABLE) g_tier[op] = (int8_t)tier;
}

int plow_cpu_tier_of(uint16_t op) {
    return (op < PLOW_CPU_DOP_TABLE && g_tab[op]) ? g_tier[op] : -1;
}

int plow_cpu_has(uint16_t op) { return op < PLOW_CPU_DOP_TABLE && g_tab[op] != NULL; }

plow_cpu_kernel_fn plow_cpu_kernel(uint16_t op) {
    return op < PLOW_CPU_DOP_TABLE ? g_tab[op] : NULL;
}

int plow_cpu_exec(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* tensors,
                  PlowCpuCtx* ctx) {
    const plow_cpu_kernel_fn fn = plow_cpu_kernel(in->op);
    if (!fn) return -1;
    fn(in, slice, nblk, tensors, ctx);
    return 0;
}
