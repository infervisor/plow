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

/* Weak defaults: the golden-only library links and runs before the AVX-512/AMX
 * tiers exist; the avx512 and amx sources provide the strong definitions. */
__attribute__((weak)) void plow_cpu_register_avx512(plow_cpu_kernel_fn* tab) { (void)tab; }
__attribute__((weak)) void plow_cpu_register_amx(plow_cpu_kernel_fn* tab) { (void)tab; }
__attribute__((weak)) int plow_cpu_thread_init_amx(PlowCpuCtx* ctx) { (void)ctx; return 0; }
