/* avx512/fp8.c — AVX-512 kernels for the fp8 (e4m3) weight family (dequant-on-load via
 * vpermw LUTs). Filled by the fp8 pass; the registrar exists so isa_detect.c links. */
#include "cpu_dev_internal.h"

void plow_cpu_register_avx512_fp8(plow_cpu_kernel_fn* tab) { (void)tab; }
