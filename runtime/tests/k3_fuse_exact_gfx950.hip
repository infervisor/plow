/* k3_fuse_exact_gfx950.hip — the two K3 packet fusions are BIT-EXACT, on hardware.
 *
 * Both fusions in this pair are justified by an exactness claim, and an exactness claim is the one
 * kind that a residual-vs-oracle gate cannot check: a 1e-7 relative difference passes every
 * tolerance and still means the two arms are not the same program. So this gate compares the two
 * ARMS to each other and demands `memcmp == 0`.
 *
 *   1. AttnRes absorbing the RMSNorm that always follows it (`op_k3.h`, `d_attn_res`'s `gamma`).
 *        ref   = d_attn_res(out=tmp, gamma=null) ; d_rmsnorm(dst, tmp, gamma)
 *        fused = d_attn_res(out=dst, gamma)
 *   2. The chained residual pair in K3's MoE tail (`op_elementwise.h`, `d_residual`'s `pre`).
 *        ref   = d_residual(ffn, a, b) ; d_residual(dst, pre, ffn)
 *        fused = d_residual(dst, a, b, pre)
 *
 * Every `nb` K3 can emit is swept (0..8), because `nb` changes both the reduction and the number
 * of rows the mix walks, and the fused arm's sum-of-squares is accumulated inside the mix loop.
 *
 *   /opt/rocm/bin/hipcc --offload-arch=gfx950 -O3 -w -DPLOW_K3=1 \
 *       -I runtime/amd -I runtime/common runtime/tests/k3_fuse_exact_gfx950.hip -o /tmp/k3fx
 *   /tmp/k3fx
 */
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amd_common.h"
#include "op_elementwise.h"
#include "op_k3.h"
#include "op_norm.h"

#define HC(x)                                                                                     \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_) {                                                                                 \
            printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__);                              \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

enum { LDS_BYTES = 147464 };

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void k_ar_ref(
    bf16* dst, bf16* tmp, const bf16* prefix, const bf16* blkres, const float* score_w,
    const bf16* gamma, unsigned T, unsigned HID, unsigned NB, float eps) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_attn_res(tmp, prefix, blkres, score_w, T, HID, NB, NB, eps, blockIdx.x, gridDim.x,
               (float*)raw, nullptr, 0u, nullptr);
    d_rmsnorm(dst, tmp, gamma, T, HID, eps, 0u, blockIdx.x, gridDim.x, (float*)raw);
}

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void k_ar_fused(
    bf16* dst, const bf16* prefix, const bf16* blkres, const float* score_w, const bf16* gamma,
    unsigned T, unsigned HID, unsigned NB, float eps) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_attn_res(dst, prefix, blkres, score_w, T, HID, NB, NB, eps, blockIdx.x, gridDim.x,
               (float*)raw, nullptr, 0u, gamma);
}

__global__ void k_res(bf16* dst, const bf16* a, const bf16* b, unsigned n, float scale,
                      const bf16* pre) {
    d_residual(dst, a, b, n, scale, blockIdx.x, gridDim.x, pre);
}

static bf16 f2bf_host(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    /* round-to-nearest-even, matching the device `f2bf` this tree uses. */
    const unsigned r = (u >> 16) & 1u;
    return (bf16)((u + 0x7fffu + r) >> 16);
}

static float frand(void) { return (float)drand48() * 2.0f - 1.0f; }

int main(void) {
    srand48(12345);
    const unsigned HID = 7168, T = 1, NBCAP = 8;
    const float eps = 1e-5f;

    std::vector<bf16> h_prefix(HID), h_gamma(HID), h_blk((size_t)NBCAP * HID), h_a(HID), h_b(HID),
        h_pre(HID);
    std::vector<float> h_sw(HID);
    for (unsigned i = 0; i < HID; i++) {
        h_prefix[i] = f2bf_host(frand());
        h_gamma[i] = f2bf_host(0.5f + 0.5f * frand());
        h_a[i] = f2bf_host(frand());
        h_b[i] = f2bf_host(frand());
        h_pre[i] = f2bf_host(frand());
        h_sw[i] = frand();
    }
    for (size_t i = 0; i < h_blk.size(); i++) h_blk[i] = f2bf_host(frand());

    bf16 *d_prefix, *d_blk, *d_gamma, *d_tmp, *d_ref, *d_fus, *d_a, *d_b, *d_pre, *d_ffn;
    float* d_sw;
    HC(hipMalloc(&d_prefix, HID * 2));
    HC(hipMalloc(&d_gamma, HID * 2));
    HC(hipMalloc(&d_a, HID * 2));
    HC(hipMalloc(&d_b, HID * 2));
    HC(hipMalloc(&d_pre, HID * 2));
    HC(hipMalloc(&d_ffn, HID * 2));
    HC(hipMalloc(&d_blk, h_blk.size() * 2));
    HC(hipMalloc(&d_sw, HID * 4));
    HC(hipMalloc(&d_tmp, HID * 2));
    HC(hipMalloc(&d_ref, HID * 2));
    HC(hipMalloc(&d_fus, HID * 2));
    HC(hipMemcpy(d_prefix, h_prefix.data(), HID * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_gamma, h_gamma.data(), HID * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_a, h_a.data(), HID * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_b, h_b.data(), HID * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_pre, h_pre.data(), HID * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_blk, h_blk.data(), h_blk.size() * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_sw, h_sw.data(), HID * 4, hipMemcpyHostToDevice));

    std::vector<bf16> ref(HID), fus(HID);
    int bad = 0;

    for (unsigned NB = 0; NB <= 8; NB++) {
        HC(hipMemset(d_ref, 0, HID * 2));
        HC(hipMemset(d_fus, 0, HID * 2));
        hipLaunchKernelGGL(k_ar_ref, dim3(1), dim3(PLOW_THREADS), 0, 0, d_ref, d_tmp, d_prefix,
                           d_blk, d_sw, d_gamma, T, HID, NB, eps);
        hipLaunchKernelGGL(k_ar_fused, dim3(1), dim3(PLOW_THREADS), 0, 0, d_fus, d_prefix, d_blk,
                           d_sw, d_gamma, T, HID, NB, eps);
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(ref.data(), d_ref, HID * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(fus.data(), d_fus, HID * 2, hipMemcpyDeviceToHost));
        const int ok = memcmp(ref.data(), fus.data(), HID * 2) == 0;
        unsigned first = 0;
        for (; first < HID && ref[first] == fus[first]; first++) {}
        printf("AttnRes+RmsNorm nb=%u  %s%s\n", NB, ok ? "BIT-IDENTICAL" : "MISMATCH",
               ok ? "" : "");
        if (!ok) {
            printf("   first differing element %u: ref %04x fused %04x\n", first, ref[first],
                   fus[first]);
            bad++;
        }
    }

    /* The residual pair, at the 2-workgroup width the K3 decode packet actually dispatches. */
    for (unsigned nblk = 1; nblk <= 2; nblk++) {
        HC(hipMemset(d_ref, 0, HID * 2));
        HC(hipMemset(d_fus, 0, HID * 2));
        hipLaunchKernelGGL(k_res, dim3(nblk), dim3(PLOW_THREADS), 0, 0, d_ffn, d_a, d_b, HID, 1.0f,
                           (const bf16*)nullptr);
        hipLaunchKernelGGL(k_res, dim3(nblk), dim3(PLOW_THREADS), 0, 0, d_ref, d_pre, d_ffn, HID,
                           1.0f, (const bf16*)nullptr);
        hipLaunchKernelGGL(k_res, dim3(nblk), dim3(PLOW_THREADS), 0, 0, d_fus, d_a, d_b, HID, 1.0f,
                           d_pre);
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(ref.data(), d_ref, HID * 2, hipMemcpyDeviceToHost));
        HC(hipMemcpy(fus.data(), d_fus, HID * 2, hipMemcpyDeviceToHost));
        const int ok = memcmp(ref.data(), fus.data(), HID * 2) == 0;
        printf("Residual chain  nblk=%u  %s\n", nblk, ok ? "BIT-IDENTICAL" : "MISMATCH");
        if (!ok) bad++;
    }

    printf("%s\n", bad ? "FAIL" : "PASS — both fusions are bit-exact");
    return bad ? 1 : 0;
}
