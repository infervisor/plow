/* mla_pf_mha_bench_gfx950.hip — GLM-5.3 TP8 MLA prefill attention per rank (8 heads, causal).
 *   absorbed : d_flash_mla_prefill_v2<512,64> (QK 576 / PV 512 per head), the shipped arm
 *   mha256   : d_flash_prefill<256> on head-major K/V copies (reference for the expanded form)
 *   mla_mha  : d_mla_mha_pf reading q_b / kv_b / k_rope outputs in place (mla_mha_pf.h)
 * mla_mha is checked against mha256 (rel-L2 over all of O).
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common -DPLOW_BUCKET_FLASH \
 *     -DPLOW_WG_WAVES=4 -DFA_DC=256 -DFA_DBUF=1 -DPLOW_MLA_PF_V2_ARM=1 -DPLOW_MLA_PF_SV=1 \
 *     -DPLOW_MLA_PF_TR16=1 -DPLOW_BUCKET_DECODE=0 runtime/tests/mla_pf_mha_bench_gfx950.hip
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__
#include "op_attention.h"
#include "mla_mha_pf.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
#define LDS_BYTES 143360
constexpr unsigned H = 8, KVW = H * 448;

extern "C" __global__ __launch_bounds__(256) void k_absorbed(float* op, float* ml, const bf16* qa, const bf16* qr,
                                                             const bf16* ck, const bf16* kr, const int* kvl,
                                                             unsigned T, float scale) {
#ifdef __HIP_DEVICE_COMPILE__
    extern __shared__ bf16 lds[];
    d_flash_mla_prefill_v2<512, 64, false, false>(op, ml, qa, qr, ck, kr, kvl, 1u, T, H, T, 0u, scale,
                                                  0xffffffffu, blockIdx.x, gridDim.x, lds, nullptr, 0u, 1u,
                                                  false, nullptr, 0u);
#endif
}

extern "C" __global__ __launch_bounds__(256) void k_mha(bf16* o, const bf16* q, const bf16* k, const bf16* v,
                                                        unsigned T, float scale) {
#ifdef __HIP_DEVICE_COMPILE__
    extern __shared__ bf16 lds[];
    d_flash_prefill<256>(nullptr, nullptr, o, q, k, v, T, T, H, H, 0u, 0u, scale, T, 0xffffffffu, 1u,
                         blockIdx.x, gridDim.x, lds);
#endif
}

extern "C" __global__ __launch_bounds__(256) void k_new(bf16* o, const bf16* qn, const bf16* qr, const bf16* kvx,
                                                        const bf16* kr, unsigned T, float scale) {
#ifdef __HIP_DEVICE_COMPILE__
    extern __shared__ bf16 lds[];
    d_mla_mha_pf(o, qn, qr, kvx, kr, kvx + 192, T, H, H * 256, 256, H * 64, 64, KVW, 448, 64, scale,
                 blockIdx.x, gridDim.x, lds);
#endif
}

static unsigned short f2b(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float b2f(unsigned short b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static void fill(std::vector<unsigned short>& v, float amp) {
    for (auto& x : v) x = f2b((float)(rand() % 2001 - 1000) / 1000.0f * amp);
}

int main() {
    for (auto k : {(const void*)k_absorbed, (const void*)k_mha, (const void*)k_new})
        CK(hipFuncSetAttribute(k, hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES));
    const float scale = 1.0f / sqrtf(256.0f);
    int bad = 0;
    for (unsigned T : {1024u, 2048u, 4096u, 8192u, 16384u}) {
        const size_t r = (size_t)T * H;
        std::vector<unsigned short> qa(r * 512), qra(r * 64), ck((size_t)T * 512), kra((size_t)T * 64);
        std::vector<unsigned short> qn(r * 256), qr(r * 64), kvx((size_t)T * KVW), kr((size_t)T * 64);
        fill(qa, 0.25f); fill(qra, 0.25f); fill(ck, 0.25f); fill(kra, 0.25f);
        fill(qn, 1.0f); fill(qr, 1.0f); fill(kvx, 1.0f); fill(kr, 1.0f);
        std::vector<unsigned short> qh(r * 256), kh(r * 256), vh(r * 256);
        for (size_t t = 0; t < T; t++)
            for (unsigned h = 0; h < H; h++) {
                for (unsigned d = 0; d < 256; d++) {
                    qh[(t * H + h) * 256 + d] = d < 192 ? qn[t * H * 256 + h * 256 + d] : qr[(t * H + h) * 64 + d - 192];
                    kh[((size_t)h * T + t) * 256 + d] = d < 192 ? kvx[t * KVW + h * 448 + d] : kr[t * 64 + d - 192];
                    vh[((size_t)h * T + t) * 256 + d] = kvx[t * KVW + h * 448 + 192 + d];
                }
            }
        auto up = [](const std::vector<unsigned short>& v) {
            bf16* d; CK(hipMalloc(&d, v.size() * 2)); CK(hipMemcpy(d, v.data(), v.size() * 2, hipMemcpyHostToDevice)); return d;
        };
        bf16 *dqa = up(qa), *dqra = up(qra), *dck = up(ck), *dkra = up(kra);
        bf16 *dqn = up(qn), *dqr = up(qr), *dkvx = up(kvx), *dkr = up(kr), *dqh = up(qh), *dkh = up(kh), *dvh = up(vh);
        bf16 *dref, *dnew; float *dop, *dml; int* dkvl;
        CK(hipMalloc(&dref, r * 256 * 2)); CK(hipMalloc(&dnew, r * 256 * 2));
        CK(hipMalloc(&dop, r * 512 * 4)); CK(hipMalloc(&dml, r * 2 * 4)); CK(hipMalloc(&dkvl, 4));
        const int kvl = (int)T;
        CK(hipMemcpy(dkvl, &kvl, 4, hipMemcpyHostToDevice));
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        auto timed = [&](auto launch) {
            std::vector<float> ts;
            for (int it = 0; it < 7; it++) {
                CK(hipEventRecord(e0)); launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); if (it) ts.push_back(ms);
            }
            std::sort(ts.begin(), ts.end()); return ts[ts.size() / 2];
        };
        const float ta = timed([&] { hipLaunchKernelGGL(k_absorbed, dim3(256), dim3(256), LDS_BYTES, 0, dop, dml, dqa, dqra, dck, dkra, dkvl, T, scale); });
        const float tm = timed([&] { hipLaunchKernelGGL(k_mha, dim3(256), dim3(256), LDS_BYTES, 0, dref, dqh, dkh, dvh, T, scale); });
        const float tn = timed([&] { hipLaunchKernelGGL(k_new, dim3(256), dim3(256), LDS_BYTES, 0, dnew, dqn, dqr, dkvx, dkr, T, scale); });
        CK(hipGetLastError());
        std::vector<unsigned short> a(r * 256), b(r * 256);
        CK(hipMemcpy(a.data(), dref, a.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), dnew, b.size() * 2, hipMemcpyDeviceToHost));
        double num = 0, den = 0; size_t worst = 0; double wd = 0;
        for (size_t i = 0; i < a.size(); i++) {
            const double x = b2f(a[i]), y = b2f(b[i]), d = (x - y) * (x - y);
            num += d; den += x * x; if (d > wd) { wd = d; worst = i; }
        }
        if (getenv("MHA_DIAG") && T == 1024) {
            std::vector<int> byw(64 * 4, 0), byh(8, 0), byd(256, 0);
            for (size_t i = 0; i < a.size(); i++) {
                const double x = b2f(a[i]), y = b2f(b[i]);
                if (!(fabs(x - y) < 0.05)) {
                    const size_t t = i / 2048, h = (i / 256) % 8, d = i % 256;
                    byw[(t / 128) * 4 + (t % 128) / 32]++; byh[h]++; byd[d]++;
                }
            }
            printf("bad by (qtile,wave):"); for (int k = 0; k < 32; k++) printf(" %d", byw[k]); printf("\n");
            printf("bad by head:"); for (int k = 0; k < 8; k++) printf(" %d", byh[k]); printf("\n");
            printf("bad by d[0..63]:"); for (int k = 0; k < 64; k++) printf(" %d", byd[k]); printf("\n");
            for (size_t t = 0; t < 3; t++) { printf("row t=%zu h0 d0..7 ref/new:", t); for (int d = 0; d < 8; d++) printf(" %.3f/%.3f", b2f(a[t * 2048 + d]), b2f(b[t * 2048 + d])); printf("\n"); }
        }
        const double rel = sqrt(num / (den + 1e-30));
        const bool ok = rel < 1e-2;
        bad += !ok;
        const double causal = (double)T * (T + 1) / 2 * H;
        printf("T=%-6u absorbed %7.3f ms (%4.0f TF/s)  mha256 %7.3f ms (%4.0f)  mla_mha %7.3f ms (%4.0f TF/s)  %.2fx vs absorbed  relL2 %.2e %s",
               T, ta, causal * 2 * 1088 / ta / 1e9, tm, causal * 2 * 512 / tm / 1e9, tn, causal * 2 * 512 / tn / 1e9,
               ta / tn, rel, ok ? "OK" : "BAD");
        if (!ok) printf(" worst @%zu ref %g new %g", worst, b2f(a[worst]), b2f(b[worst]));
        printf("\n");
        for (bf16* p : {dqa, dqra, dck, dkra, dqn, dqr, dkvx, dkr, dqh, dkh, dvh, dref, dnew}) hipFree(p);
        hipFree(dop); hipFree(dml); hipFree(dkvl);
    }
    return bad ? 2 : 0;
}
