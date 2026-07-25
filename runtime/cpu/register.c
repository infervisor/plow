/* Register the CPU (BACKEND_CPU) golden kernels into a dispatch table.
 *
 * The scheduler emits generic opcodes (backend 0); the host harness uses the
 * generic family|variant index, so we register against the generic opcode
 * constants. (A CUDA/ROCm table registers the same family|variant slots with
 * its own kernels — same indices, different backend table.) */
#include "dispatch.h"
#include "cpu_kernels.h"

void plow_register_cpu(dispatch_table* dt) {
    dt_init(dt);
    dt_register(dt, PLOW_HOST_COORD,    cpu_host);
    dt_register(dt, PLOW_NOP,           cpu_host);
    dt_register(dt, PLOW_TMA_LOAD,      cpu_dma);
    dt_register(dt, PLOW_TMA_STORE,     cpu_dma);
    dt_register(dt, PLOW_RDMA,          cpu_rdma);
    dt_register(dt, PLOW_GEMM,          cpu_gemm);
    dt_register(dt, PLOW_FLASH,         cpu_flash);
    dt_register(dt, PLOW_ROW_REDUCE,    cpu_row_reduce);
    dt_register(dt, PLOW_ROW_POINTWISE, cpu_row_pointwise);
    dt_register(dt, PLOW_LAYOUT,        cpu_layout);
}
