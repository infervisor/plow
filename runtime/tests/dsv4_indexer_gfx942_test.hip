/* DeepSeek-V4's lightning indexer score, both arms.  [DSV4-IDX]
 *
 *   score[s][t] = SUM_h  w[s][h] * relu( bf16( q[s][h] . k[t] ) )        model.py:427-428
 *
 * with the `128**-0.5 * 64**-0.5` folded into `w` (model.py:424), so selection is
 * scale-invariant and there is no softmax and no attention sink.
 *
 * 1. DECODE — d_index_score_kpool<128, FAKEQ=true>. V4 fake-quantizes q and the compressed
 *    index keys to fp4 and multiplies straight back, so the operands reaching this kernel are
 *    bf16 with the block scales already folded in and NO per-vector scale. Checked at
 *    index_heads = 64 (V4) and 32 (GLM's count, to show the runtime head axis is really
 *    runtime), and the fp8 arm is re-checked in the same run as a REGRESSION on the shipped
 *    GLM-5.3 path.
 *
 * 2. PREFILL — d_index_score_pf_row<128, HIc, TILE_N>, whose 32x32 MFMA A-tile M axis IS the
 *    head axis and which hard-asserted HIc == 32. V4 has 64. Checked at BOTH 32 and 64 against
 *    the same host reference, so the generalization is proven not to have moved the 32 case.
 *
 * In nix develop:
 *   $PLOW_HIPCC --offload-arch=gfx942 -O3 -w -std=c++17 -Iruntime/amd -Iruntime/common \
 *       -c runtime/tests/dsv4_indexer_gfx942_test.hip -o /tmp/idx.o
 *   c++ /tmp/idx.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/idx
 *   perf-data/tools/gpulease -n 1 v4-indexer /tmp/idx
 */
#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#else
using bf16 = unsigned short;
#endif
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void ck(hipError_t s) {
    if (s != hipSuccess) {
        std::fprintf(stderr, "hip: %s\n", hipGetErrorString(s));
        std::exit(1);
    }
}
template <typename T>
static T* up(const std::vector<T>& v) {
    T* p = nullptr;
    ck(hipMalloc(&p, v.size() * sizeof(T) + 64));
    ck(hipMemset(p, 0, v.size() * sizeof(T) + 64));
    ck(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

__global__ __launch_bounds__(512, 1) void k_kpool_fq(float* Sc, const bf16* Q, const bf16* K,
                                                     const float* W, const int* kv_len,
                                                     unsigned nb, unsigned HI, unsigned stride,
                                                     unsigned psz, float sc) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ uint8_t q[64 * 128 * 2];
    d_index_score_kpool<128, /*FAKEQ=*/true>(Sc, (const uint8_t*)Q, nullptr, (const uint8_t*)K,
                                             nullptr, W, kv_len, nb, HI, stride, psz, false, sc,
                                             blockIdx.x, gridDim.x, q);
#endif
}

__global__ __launch_bounds__(512, 1) void k_kpool_fp8(float* Sc, const uint8_t* Q, const float* Qs,
                                                      const uint8_t* K, const float* Ks,
                                                      const float* W, const int* kv_len,
                                                      unsigned nb, unsigned HI, unsigned stride,
                                                      unsigned psz, float sc) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ uint8_t q[64 * 128];
    d_index_score_kpool<128>(Sc, Q, Qs, K, Ks, W, kv_len, nb, HI, stride, psz, false, sc,
                             blockIdx.x, gridDim.x, q);
#endif
}

template <int HIc>
__global__ __launch_bounds__(512, 1) void k_pfrow(float* Sc, const bf16* Q, const bf16* K,
                                                  const bf16* W, const int* kv_len, unsigned n_tok,
                                                  unsigned kv_stride, float sc) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 kt[64 * (128 + FA_PAD)];
    d_index_score_pf_row<128, HIc, 64u>(Sc, Q, K, W, kv_len, n_tok, kv_stride, sc, blockIdx.x,
                                        gridDim.x, kt);
#endif
}
template __global__ void k_pfrow<32>(float*, const bf16*, const bf16*, const bf16*, const int*,
                                     unsigned, unsigned, float);
template __global__ void k_pfrow<64>(float*, const bf16*, const bf16*, const bf16*, const int*,
                                     unsigned, unsigned, float);

/* ------------------------------------------------------------------ host */
static int fails = 0;
static unsigned rs_ = 0x9E3779B9u;
static float rnd() {
    rs_ ^= rs_ << 13;
    rs_ ^= rs_ >> 17;
    rs_ ^= rs_ << 5;
    return (float)(int)(rs_ >> 8) / 8388608.0f - 1.0f;
}
static bf16 f2b(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static float b2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static float rtb(double v) { return b2f(f2b((float)v)); }
static void say(const char* w, bool ok, const char* d) {
    std::printf("  %-52s %s  %s\n", w, ok ? "ok " : "FAIL", d);
    if (!ok) fails++;
}

static const int DI = 128;

/* ---- decode, fake-quant (bf16) arm ---- */
static void decode_fq(unsigned HI, unsigned n_pool, unsigned pool_size) {
    const unsigned ctx = n_pool * pool_size, stride = n_pool + 8;
    const float sc = std::pow(128.0f, -0.5f) * std::pow((float)HI, -0.5f);
    std::vector<bf16> Q((size_t)HI * DI), K((size_t)stride * DI);
    std::vector<float> W(HI), S(stride, -777.0f);
    for (auto& v : Q) v = f2b(rnd());
    for (auto& v : K) v = f2b(rnd());
    for (auto& v : W) v = rnd();
    std::vector<int> kl(1, (int)ctx);
    auto *dQ = up(Q), *dK = up(K);
    auto *dW = up(W), *dS = up(S);
    auto* dkl = up(kl);
    hipLaunchKernelGGL(k_kpool_fq, dim3(32), dim3(512), 0, 0, dS, dQ, dK, dW, dkl, 1u, HI, stride,
                       pool_size, sc);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(S.data(), dS, S.size() * sizeof(float), hipMemcpyDeviceToHost));

    double worst = 0;
    unsigned past = 0;
    for (unsigned p = 0; p < n_pool; p++) {
        double s = 0;
        for (unsigned h = 0; h < HI; h++) {
            double d = 0;
            for (int i = 0; i < DI; i++) d += (double)b2f(Q[(size_t)h * DI + i]) *
                                              (double)b2f(K[(size_t)p * DI + i]);
            d = rtb(d);
            s += (double)W[h] * (d > 0 ? d : 0.0);
        }
        s *= sc;
        worst = std::fmax(worst, std::fabs(S[p] - s) / std::fmax(std::fabs(s), 1e-4));
    }
    /* Only the COMPLETE pools may be scored: an incomplete trailing pool has no cache entry. */
    for (unsigned p = n_pool; p < stride; p++)
        if (S[p] != -777.0f) past++;
    char det[160];
    std::snprintf(det, sizeof det, "(HI=%u pools=%u worst rel %.2e, %u past-end)", HI, n_pool,
                  worst, past);
    say("decode fake-quant (bf16, no scales)", worst < 5e-3 && past == 0, det);
    hipFree(dQ); hipFree(dK); hipFree(dW); hipFree(dS); hipFree(dkl);
}

/* ---- decode, the SHIPPED fp8 arm: a regression check, not a V4 path ---- */
static void decode_fp8(unsigned HI, unsigned n_pool, unsigned pool_size) {
    const unsigned ctx = n_pool * pool_size, stride = n_pool + 8;
    const float sc = 0.011f;
    std::vector<uint8_t> Q((size_t)HI * DI), K((size_t)stride * DI);
    std::vector<float> Qs(HI), Ks(stride), W(HI), S(stride, -777.0f);
    /* 0x7F / 0xFF is the OCP e4m3 NaN code (e=15, m=7), NOT 480 -- generating it made this
     * regression check fail against a host reference that decoded it as a finite number. Keep
     * the exponent below 15 so every byte is a finite normal or subnormal. */
    auto byte = [] { return (uint8_t)(((unsigned)(rnd() * 1e6f) & 0x87u) |
                                      (((unsigned)(rnd() * 1e6f) % 15u) << 3)); };
    for (auto& v : Q) v = byte();
    for (auto& v : K) v = byte();
    for (auto& v : Qs) v = 0.5f + 0.25f * (rnd() + 1.0f);
    for (auto& v : Ks) v = 0.5f + 0.25f * (rnd() + 1.0f);
    for (auto& v : W) v = rnd();
    std::vector<int> kl(1, (int)ctx);
    auto *dQ = up(Q), *dK = up(K);
    auto *dQs = up(Qs), *dKs = up(Ks), *dW = up(W), *dS = up(S);
    auto* dkl = up(kl);
    hipLaunchKernelGGL(k_kpool_fp8, dim3(32), dim3(512), 0, 0, dS, dQ, dQs, dK, dKs, dW, dkl, 1u,
                       HI, stride, pool_size, sc);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(S.data(), dS, S.size() * sizeof(float), hipMemcpyDeviceToHost));
    /* fp8 host dequant: OCP e4m3, exact into f32. */
    auto deq = [](uint8_t b) -> double {
        const int e = (b >> 3) & 0xF, m = b & 7;
        const double sg = (b & 0x80) ? -1.0 : 1.0;
        return sg * (e == 0 ? std::ldexp((double)m, -9) : std::ldexp(1.0 + m / 8.0, e - 7));
    };
    double worst = 0;
    for (unsigned p = 0; p < n_pool; p++) {
        double s = 0;
        for (unsigned h = 0; h < HI; h++) {
            double d = 0;
            for (int i = 0; i < DI; i++) d += deq(Q[(size_t)h * DI + i]) * deq(K[(size_t)p * DI + i]);
            d = rtb(d) * Ks[p];
            s += (double)W[h] * Qs[h] * (d > 0 ? d : 0.0);
        }
        s *= sc;
        worst = std::fmax(worst, std::fabs(S[p] - s) / std::fmax(std::fabs(s), 1e-4));
    }
    char det[128];
    std::snprintf(det, sizeof det, "(HI=%u pools=%u worst rel %.2e)", HI, n_pool, worst);
    say("decode fp8 arm (SHIPPED GLM path, must not move)", worst < 5e-3, det);
    hipFree(dQ); hipFree(dK); hipFree(dQs); hipFree(dKs); hipFree(dW); hipFree(dS); hipFree(dkl);
}

/* ---- prefill row arm at HIc 32 and 64 ---- */
template <int HIc>
static void prefill_row(unsigned n_tok, unsigned ctx) {
    const unsigned stride = ctx;
    const float sc = std::pow(128.0f, -0.5f) * std::pow((float)HIc, -0.5f);
    std::vector<bf16> Q((size_t)n_tok * HIc * DI), K((size_t)ctx * DI), W((size_t)n_tok * HIc);
    std::vector<float> S((size_t)n_tok * stride, -777.0f);
    for (auto& v : Q) v = f2b(rnd());
    for (auto& v : K) v = f2b(rnd());
    for (auto& v : W) v = f2b(rnd());
    std::vector<int> kl(1, (int)ctx);
    auto *dQ = up(Q), *dK = up(K), *dW = up(W);
    auto* dS = up(S);
    auto* dkl = up(kl);
    hipLaunchKernelGGL(k_pfrow<HIc>, dim3(64), dim3(512), 0, 0, dS, dQ, dK, dW, dkl, n_tok, stride,
                       sc);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(S.data(), dS, S.size() * sizeof(float), hipMemcpyDeviceToHost));

    const unsigned q_pos0 = ctx - n_tok;
    double worst = 0;
    unsigned past = 0;
    for (unsigned t = 0; t < n_tok; t++) {
        const unsigned row_end = q_pos0 + t + 1;
        for (unsigned p = 0; p < stride; p++) {
            if (p >= row_end) { /* strictly causal: nothing may be written past the bound */
                if (S[(size_t)t * stride + p] != -777.0f) past++;
                continue;
            }
            double s = 0;
            for (unsigned h = 0; h < (unsigned)HIc; h++) {
                double d = 0;
                for (int i = 0; i < DI; i++)
                    d += (double)b2f(Q[((size_t)t * HIc + h) * DI + i]) *
                         (double)b2f(K[(size_t)p * DI + i]);
                s += (double)b2f(W[(size_t)t * HIc + h]) * (d > 0 ? d : 0.0);
            }
            s *= sc;
            worst = std::fmax(worst, std::fabs(S[(size_t)t * stride + p] - s) /
                                         std::fmax(std::fabs(s), 1e-3));
        }
    }
    char det[160];
    std::snprintf(det, sizeof det, "(HIc=%d T=%u ctx=%u worst rel %.2e, %u past-causal)", HIc,
                  n_tok, ctx, worst, past);
    char nm[96];
    std::snprintf(nm, sizeof nm, "prefill row arm, HIc=%d", HIc);
    say(nm, worst < 8e-3 && past == 0, det);
    hipFree(dQ); hipFree(dK); hipFree(dW); hipFree(dS); hipFree(dkl);
}

int main() {
    std::printf("DeepSeek-V4 lightning indexer score\n");
    decode_fq(64, 96, 4);  /* V4: 64 index heads, compress_ratio 4 */
    decode_fq(32, 130, 4); /* the head axis really is runtime */
    decode_fp8(32, 96, 4);
    prefill_row<32>(48, 256); /* the SHIPPED GLM tile, must not move */
    prefill_row<64>(48, 256); /* V4 */
    prefill_row<64>(9, 128);
    std::printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
