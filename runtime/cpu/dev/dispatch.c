#include "cpu_dev_internal.h"
#include <string.h>

static plow_cpu_kernel_fn g_tab[PLOW_CPU_DOP_TABLE];

plow_cpu_kernel_fn* plow_cpu_table(void) { return g_tab; }

void plow_cpu_table_reset(void) { memset(g_tab, 0, sizeof(g_tab)); }

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
