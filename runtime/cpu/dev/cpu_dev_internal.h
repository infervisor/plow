/* cpu_dev_internal.h — shared between dispatch.c, isa_detect.c and the tier registrars. */
#ifndef PLOW_CPU_DEV_INTERNAL_H
#define PLOW_CPU_DEV_INTERNAL_H

#include "cpu_dev.h"

#define PLOW_CPU_SCRATCH_BYTES (1u << 20)

/* The live table (dispatch.c). Registrars write it during plow_cpu_init only. */
plow_cpu_kernel_fn* plow_cpu_table(void);
void plow_cpu_table_reset(void);

/* Tier registrars. Golden fills every op it has; later tiers override entries.
 * Each is a STRONG definition in its tier's sources (weak defaults do not work through
 * a static archive: the archive member with the weak symbol satisfies the reference and
 * the strong one is never extracted -- measured, the table silently stayed golden). */
void plow_cpu_register_golden(plow_cpu_kernel_fn* tab);
void plow_cpu_register_avx512(plow_cpu_kernel_fn* tab);
void plow_cpu_register_amx(plow_cpu_kernel_fn* tab);
/* fp8 (e4m3) weight family, one registrar per tier (golden/fp8.c, avx512/fp8.c, amx/fp8.c);
 * called right after the tier's base registrar. */
void plow_cpu_register_golden_fp8(plow_cpu_kernel_fn* tab);
void plow_cpu_register_avx512_fp8(plow_cpu_kernel_fn* tab);
void plow_cpu_register_amx_fp8(plow_cpu_kernel_fn* tab);
/* Per-thread AMX setup (LDTILECFG palette 1, 8 tiles of 16x64 B). 0 or -errno. */
int plow_cpu_thread_init_amx(PlowCpuCtx* ctx);

/* dispatch.c: attribute changed table entries to `tier` (see plow_cpu_tier_of). */
void plow_cpu_table_mark(const plow_cpu_kernel_fn* before, int tier);
void plow_cpu_table_mark_one(uint16_t op, int tier);

#endif /* PLOW_CPU_DEV_INTERNAL_H */
