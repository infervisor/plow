/* golden/fp8.c — scalar kernels for the fp8 (e4m3) weight family. Filled by the fp8 pass;
 * the registrar exists so isa_detect.c links before any fp8 op is ported. */
#include "golden.h"

void plow_cpu_register_golden_fp8(plow_cpu_kernel_fn* tab) { (void)tab; }
