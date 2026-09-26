/* headnorm_rope_ilp_gfx950.hip — d_headnorm_rope_ilp (PLOW_HNR_ILP) vs d_headnorm_rope at the GLM
 * DSA indexer rope shapes (HD=128 interleaved: q 32 heads in place, k 1 head), 256 workgroups.
 * Output must be byte-identical. Timing per call.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/headnorm_rope_ilp_gfx950.hip -o hnr_ilp
 */
#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_norm.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ __launch_bounds__(512, 1) void k_ref(bf16* o, const bf16* x, const float* c, const float* s,
                                                          const int* pos, unsigned ntok, unsigned nh) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_headnorm_rope<128, true>(o, x, nullptr, c, s, pos, ntok, nh, 1e-5f, 0, 0, 0xFFFFFFFFu, 1, blockIdx.x,
                               gridDim.x, 0);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_ref64(bf16* o, const bf16* x, const float* c, const float* s,
                                                            const int* pos, unsigned ntok, unsigned nh) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_headnorm_rope<64, true>(o, x, nullptr, c, s, pos, ntok, nh, 1e-5f, 0, 0, 0xFFFFFFFFu, 1, blockIdx.x,
                              gridDim.x, 0);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_ilp64(bf16* o, const bf16* x, const float* c, const float* s,
                                                            const int* pos, unsigned ntok, unsigned nh) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_headnorm_rope_ilp<64, true>(o, x, nullptr, c, s, pos, ntok, nh, 1e-5f, 0, 0, 0xFFFFFFFFu, 1, blockIdx.x,
                                  gridDim.x, 0);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_ilp(bf16* o, const bf16* x, const float* c, const float* s,
                                                          const int* pos, unsigned ntok, unsigned nh) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_headnorm_rope_ilp<128, true>(o, x, nullptr, c, s, pos, ntok, nh, 1e-5f, 0, 0, 0xFFFFFFFFu, 1, blockIdx.x,
                                   gridDim.x, 0);
#endif
}

int main() {
    const unsigned NT = 16384, NH = 32;
    std::vector<unsigned short> hx((size_t)NT * NH * 128);
    unsigned long long r = 88172645463325252ull;
    for (auto& v : hx) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; float f = ((r >> 40) & 0xFFFF) / 16384.0f - 2.0f; unsigned u; memcpy(&u, &f, 4); v = u >> 16; }
    std::vector<float> hc((size_t)NT * 64), hs((size_t)NT * 64);
    for (unsigned p = 0; p < NT; p++)
        for (unsigned j = 0; j < 64; j++) {
            const double th = j < 32 ? p * pow(10000.0, -2.0 * j / 64) : 0.0;
            hc[p * 64 + j] = (float)cos(th); hs[p * 64 + j] = (float)sin(th);
        }
    std::vector<int> hp(NT);
    for (unsigned t = 0; t < NT; t++) hp[t] = (int)t;
    bf16 *x, *o0, *o1; float *c, *s; int* pos;
    CK(hipMalloc(&x, hx.size() * 2)); CK(hipMalloc(&o0, hx.size() * 2)); CK(hipMalloc(&o1, hx.size() * 2));
    CK(hipMalloc(&c, hc.size() * 4)); CK(hipMalloc(&s, hs.size() * 4)); CK(hipMalloc(&pos, NT * 4));
    CK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(c, hc.data(), hc.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(s, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(pos, hp.data(), NT * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int fails = 0;
    for (unsigned hd : {128u, 64u})
    for (unsigned nh : {32u, 8u, 1u})
        for (unsigned nt : {8192u, 16384u}) {
            if (hd == 128 && nh == 8) continue;
            if (hd == 64 && nh == 32) continue;
            float ms[2];
            for (int k = 0; k < 2; k++) {
                auto fn = hd == 64 ? (k ? k_ilp64 : k_ref64) : (k ? k_ilp : k_ref);
                bf16* o = k ? o1 : o0;
                CK(hipMemset(o, 0, hx.size() * 2));
                hipLaunchKernelGGL(fn, dim3(256), dim3(512), 0, 0, o, x, c, s, pos, nt, nh);
                CK(hipDeviceSynchronize());
                CK(hipEventRecord(e0));
                for (int i = 0; i < 5; i++) hipLaunchKernelGGL(fn, dim3(256), dim3(512), 0, 0, o, x, c, s, pos, nt, nh);
                CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                CK(hipEventElapsedTime(&ms[k], e0, e1)); ms[k] /= 5;
            }
            std::vector<unsigned short> a(hx.size()), b(hx.size());
            CK(hipMemcpy(a.data(), o0, a.size() * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(b.data(), o1, b.size() * 2, hipMemcpyDeviceToHost));
            const bool same = !memcmp(a.data(), b.data(), (size_t)nt * nh * hd * 2);
            printf("hd=%3u ntok=%5u nhead=%2u ref %7.1f us  ilp %7.1f us  %s\n", hd, nt, nh, ms[0] * 1e3, ms[1] * 1e3,
                   same ? "identical" : "DIFFER");
            fails += !same;
        }
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}
