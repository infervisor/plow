/* cpu_dev_internal.h — shared between dispatch.c, isa_detect.c and the tier registrars. */
#ifndef PLOW_CPU_DEV_INTERNAL_H
#define PLOW_CPU_DEV_INTERNAL_H

#include "cpu_dev.h"

#define PLOW_CPU_SCRATCH_BYTES (1u << 20)

/* The live table (dispatch.c). Registrars write it during plow_cpu_init only. */
plow_cpu_kernel_fn* plow_cpu_table(void);
void plow_cpu_table_reset(void);

/* Tier registrars. Golden fills every op it has; later tiers override entries. */
void plow_cpu_register_golden(plow_cpu_kernel_fn* tab);

#endif /* PLOW_CPU_DEV_INTERNAL_H */
