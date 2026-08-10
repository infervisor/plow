// Atomic-scatter throughput probe for the MoE grouped-prefill fusion (gfx942).
//
// QUESTION IT ANSWERS: op 86's epilogue currently issues one non-temporal f32 STORE per output
// element into part[T*k, H] (1.611 GB/layer at T=8192), which op 87 then reads back. A fused
// decomposition replaces that store with an ATOMIC ADD into a T*H accumulator, deleting both
// 1.611 GB traversals. Whether that is a win is entirely a question of atomic throughput on
// this silicon, in THIS access pattern:
//
//   a wave64 holds 64 CONSECUTIVE columns of ONE row; the row start is effectively random
//   (scatter by row_partidx). So each wave-level op is 64 consecutive dwords = 256 B = 2 L2
//   lines at a random 64 B-aligned start.
//
// Arms, all moving the SAME number of ELEMENTS (N = 402,653,184 = 8192*8*6144, i.e. one layer's
// worth at T=8192 TP8):
//   0  nontemporal f32 store     -> 1.611 GB buffer  (the SHIPPED behaviour)
//   1  global_atomic_add_f32     -> 201 MB buffer    (the f32 fusion arm)
//   2  global_atomic_pk_add_bf16 -> 100 MB buffer    (aiter's arm; N/2 instructions)
//   3  plain f32 store           -> 201 MB buffer    (control: same footprint, no atomic)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef __bf16 bf16;
typedef short bf16x2 __attribute__((ext_vector_type(2))); /* raw bits: the v2bf16 atomic operand */

#define WGS 512

/* Each block walks scatter rows; a wave owns a row and sweeps it in 64-column runs. */
__global__ __launch_bounds__(WGS) void k_store_nt(float* __restrict__ p, const unsigned* rows,
                                                  unsigned nrow, unsigned H, unsigned runs) {
    const unsigned lane = threadIdx.x & 63u, wv = threadIdx.x >> 6;
    for (unsigned i = blockIdx.x * (WGS / 64) + wv; i < nrow; i += gridDim.x * (WGS / 64)) {
        const size_t base = (size_t)rows[i] * H;
        for (unsigned r = 0; r < runs; r++)
            __builtin_nontemporal_store(1.0f, p + base + (size_t)r * 64u + lane);
    }
}
__global__ __launch_bounds__(WGS) void k_store_pl(float* __restrict__ p, const unsigned* rows,
                                                  unsigned nrow, unsigned H, unsigned runs) {
    const unsigned lane = threadIdx.x & 63u, wv = threadIdx.x >> 6;
    for (unsigned i = blockIdx.x * (WGS / 64) + wv; i < nrow; i += gridDim.x * (WGS / 64)) {
        const size_t base = (size_t)rows[i] * H;
        for (unsigned r = 0; r < runs; r++) p[base + (size_t)r * 64u + lane] = 1.0f;
    }
}
__global__ __launch_bounds__(WGS) void k_atomic_f32(float* __restrict__ p, const unsigned* rows,
                                                    unsigned nrow, unsigned H, unsigned runs) {
    const unsigned lane = threadIdx.x & 63u, wv = threadIdx.x >> 6;
    for (unsigned i = blockIdx.x * (WGS / 64) + wv; i < nrow; i += gridDim.x * (WGS / 64)) {
        const size_t base = (size_t)rows[i] * H;
        for (unsigned r = 0; r < runs; r++)
            __hip_atomic_fetch_add(p + base + (size_t)r * 64u + lane, 1.0f, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
    }
}
__global__ __launch_bounds__(WGS) void k_atomic_bf16(bf16x2* __restrict__ p, const unsigned* rows,
                                                     unsigned nrow, unsigned H, unsigned runs) {
    const unsigned lane = threadIdx.x & 63u, wv = threadIdx.x >> 6;
    const bf16x2 one = {0x3f80, 0x3f80}; /* bf16(1.0) x2 */
    for (unsigned i = blockIdx.x * (WGS / 64) + wv; i < nrow; i += gridDim.x * (WGS / 64)) {
        const size_t base = (size_t)rows[i] * (H / 2);
        for (unsigned r = 0; r < runs; r++)
            __builtin_amdgcn_global_atomic_fadd_v2bf16(
                (__attribute__((address_space(1))) bf16x2*)(p + base + (size_t)r * 64u + lane), one);
    }
}

#define CK(x) do { hipError_t e = (x); if (e) { printf("ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

int main(void) {
    const unsigned T = 8192, H = 6144, K = 8;
    const unsigned nrow = T * K;                /* 65536 scatter rows */
    const unsigned runs = H / 64;               /* 96 wave-runs of 64 columns per row */
    const size_t N = (size_t)nrow * H;          /* 402,653,184 elements */
    const size_t part_bytes = N * 4;            /* 1.611 GB (shipped part) */
    const size_t acc_bytes = (size_t)T * H * 4; /* 201 MB (fused f32 accumulator) */

    unsigned* h_rows = (unsigned*)malloc(nrow * 4);
    unsigned* h_rows_acc = (unsigned*)malloc(nrow * 4);
    for (unsigned i = 0; i < nrow; i++) {
        h_rows[i] = i;         /* part: every slot has its own row */
        h_rows_acc[i] = i / K; /* accumulator: k=8 slots collide on one token row */
    }
    unsigned *d_rows, *d_rows_acc;
    CK(hipMalloc(&d_rows, nrow * 4)); CK(hipMalloc(&d_rows_acc, nrow * 4));
    CK(hipMemcpy(d_rows, h_rows, nrow * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_rows_acc, h_rows_acc, nrow * 4, hipMemcpyHostToDevice));

    float *d_part, *d_acc;
    CK(hipMalloc(&d_part, part_bytes)); CK(hipMemset(d_part, 0, part_bytes));
    CK(hipMalloc(&d_acc, acc_bytes));   CK(hipMemset(d_acc, 0, acc_bytes));

    const int grid = 304;
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const char* names[4] = {"store_nt -> part[T*k,H] 1.611GB (SHIPPED)",
                            "atomic_add_f32 -> acc[T,H] 201MB",
                            "atomic_pk_add_bf16 -> acc[T,H] 100MB (N/2 ops)",
                            "plain store -> acc[T,H] 201MB (control)"};
    for (int arm = 0; arm < 4; arm++) {
        float best = 1e30f;
        for (int rep = 0; rep < 4; rep++) {
            CK(hipEventRecord(a));
            switch (arm) {
            case 0: k_store_nt<<<grid, WGS>>>(d_part, d_rows, nrow, H, runs); break;
            case 1: k_atomic_f32<<<grid, WGS>>>(d_acc, d_rows_acc, nrow, H, runs); break;
            case 2: k_atomic_bf16<<<grid, WGS>>>((bf16x2*)d_acc, d_rows_acc, nrow, H, runs); break;
            case 3: k_store_pl<<<grid, WGS>>>(d_acc, d_rows_acc, nrow, H, runs); break;
            }
            CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
            float ms; CK(hipEventElapsedTime(&ms, a, b));
            if (rep) best = ms < best ? ms : best;
        }
        const double elem = (double)N;
        printf("%-48s  %8.3f ms   %7.1f G elem/s   %6.0f GB/s(payload)\n", names[arm], best,
               elem / (best * 1e6), (arm == 2 ? elem * 2 : elem * 4) / (best * 1e6));
    }
    return 0;
}
