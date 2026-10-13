/* gemm_group_kda_gfx950.hip — GLM-5.3-Flash TP8 KDA input projections at prefill (M=8192, K=4096):
 * q|k|v (N=1024 each), g_a|f_a (N=128), b (N=8). Separate launches with the interpreter's tiles
 * (d_gemm_wide for q/k/v, d_gemm_small for the rest) vs ONE grouped launch over all six. Bit
 * identity per output + timing.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/gemm_group_kda_gfx950.hip
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__
#include "op_gemm.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr unsigned M = 8192, K = 4096, NG = 6;
constexpr unsigned NS[NG] = {1024, 1024, 1024, 128, 128, 8};

struct Group {
    bf16* c[NG];
    const bf16* w[NG];
    unsigned n[NG];
};

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_wide(bf16* C, const bf16* A, const bf16* B, unsigned N) {
#ifdef __HIP_DEVICE_COMPILE__
    extern __shared__ bf16 lds[];
    d_gemm_wide(C, A, B, M, N, K, blockIdx.x, gridDim.x, lds);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_small(bf16* C, const bf16* A, const bf16* B, unsigned N) {
#ifdef __HIP_DEVICE_COMPILE__
    extern __shared__ bf16 lds[];
    d_gemm_small(C, A, B, M, N, K, blockIdx.x, gridDim.x, lds);
#endif
}
/* Tile sweep on one N=3072 (q|k|v) GEMM: tile id 0=256x256 1=192x256 2=128x256 3=128x384. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_tile(bf16* C, const bf16* A, const bf16* B, unsigned N, unsigned tile) {
#ifdef __HIP_DEVICE_COMPILE__
    extern __shared__ bf16 lds[];
    if (tile == 0) d_gemm(C, A, B, M, N, K, blockIdx.x, gridDim.x, lds);
    else if (tile == 1) d_gemm_c5(C, A, B, M, N, K, blockIdx.x, gridDim.x, lds);
    else if (tile == 2) d_gemm_wide(C, A, B, M, N, K, blockIdx.x, gridDim.x, lds);
    else d_gemm_c8(C, A, B, M, N, K, blockIdx.x, gridDim.x, lds);
#endif
}
/* One launch: the first three groups on the wide tile, the narrow tail on the small tile, each
 * workgroup owning exactly one tile of one group. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_group(Group g, const bf16* A) {
#ifdef __HIP_DEVICE_COMPILE__
    extern __shared__ bf16 lds[];
    unsigned b = blockIdx.x;
    for (unsigned i = 0; i < NG; i++) {
        const bool wide = g.n[i] >= 512;
        const unsigned bm = wide ? GM_WD_BM : GM_SM_BM, bn = wide ? GM_WD_BN : GM_SM_BN;
        const unsigned tiles = (M / bm) * ((g.n[i] + bn - 1) / bn);
        if (b < tiles) {
            if (wide)
                d_gemm_wide(g.c[i], A, g.w[i], M, g.n[i], K, b, tiles, lds);
            else
                d_gemm_small(g.c[i], A, g.w[i], M, g.n[i], K, b, tiles, lds);
            return;
        }
        b -= tiles;
    }
#endif
}

static unsigned short f2b(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }

int main() {
    std::vector<unsigned short> ha((size_t)M * K);
    unsigned s = 1;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 9) / 4194304.0f - 1.0f); };
    for (auto& v : ha) v = f2b(rnd());
    bf16 *A, *W[NG], *C0[NG], *C1[NG];
    CK(hipMalloc(&A, ha.size() * 2));
    CK(hipMemcpy(A, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
    Group g;
    unsigned total_tiles = 0;
    for (unsigned i = 0; i < NG; i++) {
        std::vector<unsigned short> hw((size_t)NS[i] * K);
        for (auto& v : hw) v = f2b(rnd() * 0.02f);
        CK(hipMalloc(&W[i], hw.size() * 2));
        CK(hipMemcpy(W[i], hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
        CK(hipMalloc(&C0[i], (size_t)M * NS[i] * 2));
        CK(hipMalloc(&C1[i], (size_t)M * NS[i] * 2));
        g.c[i] = C1[i]; g.w[i] = W[i]; g.n[i] = NS[i];
        const bool wide = NS[i] >= 512;
        const unsigned bm = wide ? 128 : 64, bn = wide ? 256 : 128; /* GM_WD_* / GM_SM_* (gfx950) */
        total_tiles += (M / bm) * ((NS[i] + bn - 1) / bn);
    }
    const unsigned shm = 160 * 1024;
    CK(hipFuncSetAttribute((const void*)k_wide, hipFuncAttributeMaxDynamicSharedMemorySize, shm));
    CK(hipFuncSetAttribute((const void*)k_small, hipFuncAttributeMaxDynamicSharedMemorySize, shm));
    CK(hipFuncSetAttribute((const void*)k_group, hipFuncAttributeMaxDynamicSharedMemorySize, shm));
    auto sep = [&] {
        for (unsigned i = 0; i < NG; i++) {
            if (NS[i] >= 512) k_wide<<<256, PLOW_THREADS, shm>>>(C0[i], A, W[i], NS[i]);
            else k_small<<<256, PLOW_THREADS, shm>>>(C0[i], A, W[i], NS[i]);
        }
    };
    auto grp = [&] { k_group<<<total_tiles, PLOW_THREADS, shm>>>(g, A); };
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto time = [&](auto f, const char* name) {
        for (int r = 0; r < 5; r++) f();
        CK(hipDeviceSynchronize());
        float best = 1e9;
        for (int r = 0; r < 20; r++) {
            CK(hipEventRecord(e0)); f(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); best = ms < best ? ms : best;
        }
        double fl = 0; for (unsigned i = 0; i < NG; i++) fl += 2.0 * M * NS[i] * K;
        printf("%-10s best %.3f ms  %.0f TF/s\n", name, best, fl / best / 1e9);
    };
    time(sep, "separate");
    time(grp, "grouped");
    printf("grouped tiles %u\n", total_tiles);
    {
        bf16 *Wc, *Cc;
        CK(hipMalloc(&Wc, (size_t)3072 * K * 2));
        CK(hipMalloc(&Cc, (size_t)M * 3072 * 2));
        for (unsigned i = 0; i < 3; i++) CK(hipMemcpy(Wc + (size_t)i * 1024 * K, W[i], (size_t)1024 * K * 2, hipMemcpyDeviceToDevice));
        CK(hipFuncSetAttribute((const void*)k_tile, hipFuncAttributeMaxDynamicSharedMemorySize, shm));
        const unsigned bms[4] = {256, 192, 128, 128}, bns[4] = {256, 256, 256, 384};
        const char* names[4] = {"t256x256", "t192x256", "t128x256", "t128x384"};
        for (unsigned tl = 0; tl < 4; tl++) {
            const unsigned tiles = ((M + bms[tl] - 1) / bms[tl]) * ((3072 + bns[tl] - 1) / bns[tl]);
            for (unsigned grid : {tiles, 256u}) {
                auto f = [&] { k_tile<<<grid, PLOW_THREADS, shm>>>(Cc, A, Wc, 3072, tl); };
                for (int r = 0; r < 3; r++) f();
                CK(hipDeviceSynchronize());
                float best = 1e9;
                for (int r = 0; r < 10; r++) {
                    CK(hipEventRecord(e0)); f(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                    float ms; CK(hipEventElapsedTime(&ms, e0, e1)); best = ms < best ? ms : best;
                }
                printf("%s grid %u: %.3f ms %.0f TF/s\n", names[tl], grid, best, 2.0 * M * 3072 * K / best / 1e9);
            }
            std::vector<unsigned short> cc((size_t)M * 3072), ref((size_t)M * 1024);
            CK(hipMemcpy(cc.data(), Cc, cc.size() * 2, hipMemcpyDeviceToHost));
            size_t diff = 0;
            for (unsigned g = 0; g < 3; g++) {
                CK(hipMemcpy(ref.data(), C0[g], ref.size() * 2, hipMemcpyDeviceToHost));
                for (size_t r = 0; r < M; r++)
                    for (size_t c = 0; c < 1024; c++) diff += cc[r * 3072 + g * 1024 + c] != ref[r * 1024 + c];
            }
            printf("%s vs separate wide: differing=%zu\n", names[tl], diff);
        }
    }
    for (unsigned i = 0; i < NG; i++) {
        std::vector<unsigned short> a((size_t)M * NS[i]), b(a.size());
        CK(hipMemcpy(a.data(), C0[i], a.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), C1[i], b.size() * 2, hipMemcpyDeviceToHost));
        size_t diff = 0; for (size_t j = 0; j < a.size(); j++) diff += a[j] != b[j];
        printf("group %u N=%u differing=%zu\n", i, NS[i], diff);
    }
    return 0;
}
