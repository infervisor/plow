/* ABI lock: including packet.h fires the _Static_asserts that pin every wire
 * struct to the size the Rust `packet` crate guarantees. If this translation
 * unit compiles, the C view and the Rust view agree byte-for-byte. We also
 * spot-check the body-size dispatch helper at runtime. */
#include "packet.h"
#include "decode.h"
#include <stdio.h>

int main(void) {
    int ok = 1;
    ok &= plow_body_size(PLOW_GEMM)       == sizeof(PlowGemmBody);
    ok &= plow_body_size(PLOW_FLASH)      == sizeof(PlowFlashBody);
    ok &= plow_body_size(PLOW_ROW_REDUCE) == sizeof(PlowRowBody);
    ok &= plow_body_size(PLOW_TMA_LOAD)   == sizeof(PlowDmaBody);
    ok &= plow_body_size(PLOW_RDMA)       == sizeof(PlowRdmaBody);
    ok &= plow_body_size(PLOW_LAYOUT)     == sizeof(PlowLayoutBody);
    ok &= plow_body_size(PLOW_NOP)        == 0;

    ok &= plow_op_family(PLOW_GEMM) == PLOW_FAMILY_GEMM;
    uint16_t cuda_gemm_fp8 = PLOW_OP(PLOW_BACKEND_CUDA, PLOW_FAMILY_GEMM, PLOW_VARIANT_FP8);
    ok &= plow_op_backend(cuda_gemm_fp8) == PLOW_BACKEND_CUDA;
    ok &= plow_op_variant(cuda_gemm_fp8) == PLOW_VARIANT_FP8;

    printf("abi_layout: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
