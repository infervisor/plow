/* amx/fp8.c — AMX kernels for the fp8 (e4m3) weight GEMM family (dequant into the bf16
 * B strip while packing). Filled by the fp8 pass; the registrar exists so isa_detect.c links. */
#include "cpu_dev_internal.h"

void plow_cpu_register_amx_fp8(plow_cpu_kernel_fn* tab) { (void)tab; }
