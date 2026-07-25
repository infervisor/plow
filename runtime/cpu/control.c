/* CPU golden CONTROL — nop / host-coordination. No data effect; counters are
 * advanced by the interpreter, not the kernel. */
#include "cpu_kernels.h"

void cpu_host(const void* body, kctx* ctx) {
    (void)body;
    (void)ctx;
}
