/* dsa_kprep_gfx950.hip — d_dsa_k_prep (PLOW_DSA_PREP) vs the shipped pair d_layernorm_bias ->
 * d_headnorm_rope(_ilp)<128, interleaved> (skip_norm, no gamma) at the GLM DSA indexer key shape
 * (feat 128, eps 1e-6), 256 workgroups x 512 threads. Output must be byte-identical, including
 * a chunk-rebased cache write (out_row0 = c0) and ragged row counts. Timing per call. `feat` is a
 * kernel argument, as in the interpreter: a literal 128 lets hipcc turn the mean's division into a
 * multiply and contract it into `x - mean`, which the interpreter build never does.
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/dsa_kprep_gfx950.hip -o dsa_kprep
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
#include "dsa_prep.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

extern "C" __global__ __launch_bounds__(512, 1) void k_ln(bf16* o, const bf16* x, const bf16* g, const bf16* b,
                                                         unsigned ntok, unsigned feat) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float part[PLOW_WAVES];
    d_layernorm_bias(o, x, g, b, ntok, feat, 1e-6f, 0, blockIdx.x, gridDim.x, part);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_hnr(bf16* o, const bf16* x, const float* c, const float* s,
                                                          const int* pos, unsigned ntok, unsigned row0) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_headnorm_rope<128, true>(o, x, nullptr, c, s, pos, ntok, 1, 1e-5f, row0, 0, 0xFFFFFFFFu, 1, blockIdx.x,
                               gridDim.x, 0);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_hnr_ilp(bf16* o, const bf16* x, const float* c,
                                                              const float* s, const int* pos, unsigned ntok,
                                                              unsigned row0) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_headnorm_rope_ilp<128, true>(o, x, nullptr, c, s, pos, ntok, 1, 1e-5f, row0, 0, 0xFFFFFFFFu, 1, blockIdx.x,
                                   gridDim.x, 0);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_fused(bf16* o, const bf16* x, const bf16* g, const bf16* b,
                                                            const float* c, const float* s, const int* pos,
                                                            unsigned ntok, unsigned row0, unsigned feat) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_dsa_k_prep(o, x, g, b, c, s, pos, ntok, feat, 1e-6f, row0, blockIdx.x, gridDim.x);
#endif
}

static unsigned long long rs = 88172645463325252ull;
static unsigned rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 32); }
static unsigned short tobf(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }

int main() {
    const unsigned NT = 16384, CTX = 32768;
    std::vector<unsigned short> hx((size_t)NT * 128), hg(128), hb(128);
    /* GEMM-output-like rows: per-row offset and scale spanning ~2^-6 .. 2^6, plus exact zeros and
     * near-constant rows (the LayerNorm cancellation case). */
    for (unsigned t = 0; t < NT; t++) {
        const float sc = ldexpf(1.0f, (int)(rnd() % 13) - 6), off = ((int)(rnd() % 2001) - 1000) / 250.0f;
        const unsigned kind = rnd() % 64;
        for (unsigned i = 0; i < 128; i++) {
            float f = off + sc * (((int)(rnd() % 65536) - 32768) / 16384.0f);
            if (kind == 0) f = off + (i == 7 ? sc : 0.0f);
            if (kind == 1) f = 0.0f;
            if (kind == 2 && (i & 3) == 0) f = -0.0f;
            hx[(size_t)t * 128 + i] = tobf(f);
        }
    }
    for (unsigned i = 0; i < 128; i++) {
        hg[i] = tobf(0.5f + (rnd() % 4096) / 2048.0f);
        hb[i] = tobf(((int)(rnd() % 4096) - 2048) / 8192.0f);
    }
    std::vector<float> hc((size_t)CTX * 64), hs((size_t)CTX * 64);
    for (unsigned p = 0; p < CTX; p++)
        for (unsigned j = 0; j < 64; j++) {
            const double th = j < 32 ? p * pow(10000.0, -2.0 * j / 64) : 0.0;
            hc[p * 64 + j] = (float)cos(th); hs[p * 64 + j] = (float)sin(th);
        }
    bf16 *x, *g, *b, *tmp, *o0, *o1, *o2; float *c, *s; int* pos;
    const size_t cache = (size_t)CTX * 128 * 2;
    CK(hipMalloc(&x, hx.size() * 2)); CK(hipMalloc(&g, 256)); CK(hipMalloc(&b, 256));
    CK(hipMalloc(&tmp, hx.size() * 2)); CK(hipMalloc(&o0, cache)); CK(hipMalloc(&o1, cache)); CK(hipMalloc(&o2, cache));
    CK(hipMalloc(&c, hc.size() * 4)); CK(hipMalloc(&s, hs.size() * 4)); CK(hipMalloc(&pos, NT * 4));
    CK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(g, hg.data(), 256, hipMemcpyHostToDevice));
    CK(hipMemcpy(b, hb.data(), 256, hipMemcpyHostToDevice));
    CK(hipMemcpy(c, hc.data(), hc.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(s, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int fails = 0;
    struct Case { unsigned nt, row0; } cases[] = {{8192, 0}, {16384, 0}, {4096, 0}, {8191, 0}, {1000, 0}, {8192, 4096}, {3000, 12288}};
    for (auto cs : cases) {
        std::vector<int> hp(cs.nt);
        for (unsigned t = 0; t < cs.nt; t++) hp[t] = (int)(cs.row0 + t);
        CK(hipMemcpy(pos, hp.data(), cs.nt * 4, hipMemcpyHostToDevice));
        CK(hipMemset(o0, 0x5a, cache)); CK(hipMemset(o1, 0x5a, cache)); CK(hipMemset(o2, 0x5a, cache));
        float ms[3];
        auto run = [&](int k) {
            if (k == 0) {
                hipLaunchKernelGGL(k_ln, dim3(256), dim3(512), 0, 0, tmp, x, g, b, cs.nt, 128u);
                hipLaunchKernelGGL(k_hnr_ilp, dim3(256), dim3(512), 0, 0, o0, tmp, c, s, pos, cs.nt, cs.row0);
            } else if (k == 1) {
                hipLaunchKernelGGL(k_ln, dim3(256), dim3(512), 0, 0, tmp, x, g, b, cs.nt, 128u);
                hipLaunchKernelGGL(k_hnr, dim3(256), dim3(512), 0, 0, o1, tmp, c, s, pos, cs.nt, cs.row0);
            } else {
                hipLaunchKernelGGL(k_fused, dim3(256), dim3(512), 0, 0, o2, x, g, b, c, s, pos, cs.nt, cs.row0, 128u);
            }
        };
        for (int k = 0; k < 3; k++) {
            run(k);
            CK(hipDeviceSynchronize());
            CK(hipEventRecord(e0));
            for (int i = 0; i < 10; i++) run(k);
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms[k], e0, e1)); ms[k] /= 10;
        }
        std::vector<unsigned short> a(cache / 2), bb(cache / 2), f(cache / 2);
        CK(hipMemcpy(a.data(), o0, cache, hipMemcpyDeviceToHost));
        CK(hipMemcpy(bb.data(), o1, cache, hipMemcpyDeviceToHost));
        CK(hipMemcpy(f.data(), o2, cache, hipMemcpyDeviceToHost));
        const bool same = !memcmp(a.data(), f.data(), cache), same2 = !memcmp(bb.data(), f.data(), cache);
        size_t ndiff = 0;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != f[i] && ndiff < 6) {
                const size_t row = i / 128 - cs.row0;
                printf("  row %zu i %zu ref %04x fused %04x in %04x\n", row, i % 128, a[i], f[i],
                       hx[row * 128 + i % 128]);
            }
            ndiff += a[i] != f[i];
        }
        printf("ntok=%5u row0=%5u  ln+hnr_ilp %7.1f us  ln+hnr %7.1f us  fused %6.1f us  %s/%s (%zu diff)\n", cs.nt,
               cs.row0, ms[0] * 1e3, ms[1] * 1e3, ms[2] * 1e3, same ? "identical" : "DIFFER",
               same2 ? "identical" : "DIFFER", ndiff);
        fails += !same || !same2;
    }
    {   /* LayerNorm half alone: identity rope (cos 1, sin 0) must reproduce the LayerNorm output. */
        std::vector<float> one((size_t)CTX * 64, 1.0f), zero((size_t)CTX * 64, 0.0f);
        float *c1, *s1; CK(hipMalloc(&c1, one.size() * 4)); CK(hipMalloc(&s1, zero.size() * 4));
        CK(hipMemcpy(c1, one.data(), one.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(s1, zero.data(), zero.size() * 4, hipMemcpyHostToDevice));
        std::vector<int> hp(8192);
        for (unsigned t = 0; t < 8192; t++) hp[t] = (int)t;
        CK(hipMemcpy(pos, hp.data(), 8192 * 4, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_ln, dim3(256), dim3(512), 0, 0, tmp, x, g, b, 8192u, 128u);
        hipLaunchKernelGGL(k_fused, dim3(256), dim3(512), 0, 0, o2, x, g, b, c1, s1, pos, 8192u, 0u, 128u);
        CK(hipDeviceSynchronize());
        std::vector<unsigned short> a(8192 * 128), f(8192 * 128);
        CK(hipMemcpy(a.data(), tmp, a.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(f.data(), o2, f.size() * 2, hipMemcpyDeviceToHost));
        size_t nd = 0;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != f[i] && (a[i] | f[i]) != 0x8000 && nd < 6)
                printf("  LN row %zu i %zu ref %04x fused %04x\n", i / 128, i % 128, a[i], f[i]);
            nd += a[i] != f[i] && (a[i] | f[i]) != 0x8000;
        }
        printf("LayerNorm half: %zu diff (signed zeros excluded)\n", nd);
    }
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}
