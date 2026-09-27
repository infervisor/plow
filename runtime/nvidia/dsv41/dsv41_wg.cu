// DeepSeek-V4.1 prefill GEMMs on the Hopper wgmma tile of ../op_wg_sm90.cuh (see there).
#include "../op_wg_sm90.cuh"

// Dense: C (bf16, or f32 when c_f32) [z][M][ldc] = A [z][M][lda] . deq(W8 [z][N][K], sw [z][N/32][K/32])^T.
// grid = (ceil(N / WG_BN), ceil(M / 128), batch), 384 threads, WG_SMEM bytes.
DSV_EXTERN void __launch_bounds__(384, 1)
    dsv_gemm_wg_fp8(void* __restrict__ C, const bf16* __restrict__ A, const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw,
                    int M, int N, int K, long long lda, long long ldc, int c_f32, long long w_bstride, long long s_bstride,
                    long long a_bstride, long long c_bstride) {
    plow_wg::gemm_tile<0>(C, A, W, sw, nullptr, nullptr, nullptr, nullptr, 0, M, N, K, lda, ldc, c_f32, w_bstride, s_bstride, a_bstride, c_bstride,
                          blockIdx.x, blockIdx.y, blockIdx.z);
}

// Grouped fp4 experts: C [p][N] bf16 = A [rows[p] or p][K] (bf16) . deq(W4[e])^T for the 128-row
// tiles of dsv_moe_offsets (bm = 128). grid = (ceil(N / WG_BN), max_tiles), 384 threads, WG_SMEM.
DSV_EXTERN void __launch_bounds__(384, 1)
    dsv_moe_gemm_wg_fp4(bf16* __restrict__ C, const bf16* __restrict__ A, const uint8_t* __restrict__ W, const uint8_t* __restrict__ sw,
                        const int* __restrict__ tiles, const int* __restrict__ meta, const int* __restrict__ offs,
                        const int* __restrict__ rows, int a_by_row, int N, int K, long long w_estride, long long s_estride) {
    plow_wg::gemm_tile<1>(C, A, W, sw, tiles, meta, offs, rows, a_by_row, 0, N, K, K, N, 0, w_estride, s_estride, 0, 0, blockIdx.x,
                          blockIdx.y, 0);
}
