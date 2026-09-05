/* cpu_dev_internal.h — shared between dispatch.c, isa_detect.c and the tier registrars. */
#ifndef PLOW_CPU_DEV_INTERNAL_H
#define PLOW_CPU_DEV_INTERNAL_H

#include "cpu_dev.h"

#define PLOW_CPU_SCRATCH_BYTES (1u << 20)

/* The live table (dispatch.c). Registrars write it during plow_cpu_init only. */
plow_cpu_kernel_fn* plow_cpu_table(void);
void plow_cpu_table_reset(void);

/* Tier registrars. Golden fills every op it has; later tiers override entries.
 * The AVX-512/AMX registrars and the AMX per-thread hook have weak no-op defaults
 * (dispatch.c) so the library links before those tiers exist; the avx512 and
 * amx sources define the strong versions. */
void plow_cpu_register_golden(plow_cpu_kernel_fn* tab);
void plow_cpu_register_avx512(plow_cpu_kernel_fn* tab);
void plow_cpu_register_amx(plow_cpu_kernel_fn* tab);
/* Per-thread AMX setup (LDTILECFG palette 1, 8 tiles of 16x64 B). 0 or -errno. */
int plow_cpu_thread_init_amx(PlowCpuCtx* ctx);

#endif /* PLOW_CPU_DEV_INTERNAL_H */
