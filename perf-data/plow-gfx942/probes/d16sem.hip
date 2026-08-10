/* Minimal semantics test: does ds_read_u16_d16_hi PRESERVE the low 16 bits?
 * Seed the destination with a known constant, issue ONE d16_hi, read it back.
 * preserve => 0xKKKK5A5A ; write-32 => 0xKKKK0000 */
#include <hip/hip_runtime.h>
#include <cstdio>
extern "C" __global__ void k(unsigned* out) {
    __shared__ unsigned short pool[256];
    if (threadIdx.x < 256) pool[threadIdx.x] = (unsigned short)(0xAB00 + threadIdx.x);
    __syncthreads();
    unsigned a = (unsigned)(unsigned long long)(__attribute__((address_space(3))) const void*)&pool[0];
    unsigned w = 0x5A5A;
    asm volatile("ds_read_u16_d16_hi %0, %1 offset:8\n\ts_waitcnt lgkmcnt(0)"
                 : "+v"(w) : "v"(a));
    unsigned w2 = 0x5A5A;
    asm volatile("s_waitcnt lgkmcnt(0)\n\tds_read_u16_d16 %0, %1 offset:8\n\ts_waitcnt lgkmcnt(0)"
                 : "+v"(w2) : "v"(a));
    if (threadIdx.x == 0) { out[0] = w; out[1] = w2; }
}
int main() {
    unsigned* d; (void)hipMalloc(&d, 8); (void)hipMemset(d, 0, 8);
    k<<<1, 64>>>(d); (void)hipDeviceSynchronize();
    unsigned h[2]; (void)hipMemcpy(h, d, 8, hipMemcpyDeviceToHost);
    printf("pool[4]=0xab04 seeded reg 0x00005a5a\n");
    printf("  ds_read_u16_d16_hi -> 0x%08x  (preserve => 0xab045a5a, write32 => 0xab040000)\n", h[0]);
    printf("  ds_read_u16_d16    -> 0x%08x  (preserve => 0x0000ab04, write32 => 0x0000ab04)\n", h[1]);
    return 0;
}
