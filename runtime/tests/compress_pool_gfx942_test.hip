/* d_compress_pool — DeepSeek-V4's learned-pooling KV compressor.  [DSV4-COMPRESS]
 *
 * ORACLE: a host reference written straight off `inference/model.py:322-380` and
 * `inference/kernel.py:36-200`, in double precision for the pooling and the norm, but
 * reproducing every bf16 round trip and the exact quantizer ladders the reference uses (there
 * is no "more accurate" version of a fake quant -- the whole point of the epilogue is to
 * reproduce a specific lossy value, so the oracle must be lossy in the SAME places). The
 * comparison is therefore tight: the device stores bf16 and the reference computes the bf16 it
 * should have stored, so agreement is expected to be EXACT on the fp8/fp4-quantized channels
 * and within one bf16 ulp on the rope channels, where the only divergence is f32-vs-f64
 * ordering inside the norm's sum of squares.
 *
 * Cases cover both shapes the model actually has:
 *   ratio 128, coff 1, d 512, rd 64, qblk 64  -- the HCA layers' attention compressor
 *   ratio 4,   coff 2, d 512, rd 64, qblk 64  -- the CSA layers', WITH the overlap
 *   ratio 4,   coff 2, d 128, rd 64, qblk 32  -- the indexer's, ROTATE (Hadamard + fp4)
 * plus the decode ring arm at ratio 4 (on and off a pool boundary) and a non-zero `out_base`
 * so a chunked prefill cannot silently write every chunk to slot 0 -- the exact bug op 130
 * shipped with and that op_dsa_pool.h's header records at length.
 *
 * In nix develop:
 *   $PLOW_HIPCC --offload-arch=gfx942 -O3 -w -std=c++17 -Iruntime/amd -Iruntime/common \
 *       -c runtime/tests/compress_pool_gfx942_test.hip -o /tmp/cmp.o
 *   c++ /tmp/cmp.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/cmp
 *   perf-data/tools/gpulease -n 1 v4-compress /tmp/cmp
 */
#define PLOW_WG_WAVES 4
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_compress.h"
#else
using bf16 = unsigned short;
#endif
#include <cmath>
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
    ck(hipMalloc(&p, v.size() * sizeof(T)));
    ck(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

/* LDS: max(d, PLOW_THREADS) floats for the row/Hadamard arena + PLOW_WAVES for the reduce. */
#define CMP_LDS 512

__global__ __launch_bounds__(256, 1) void k_cmp(
    bf16* out, const bf16* kv, const bf16* score, const float* ape, const bf16* gamma,
    const float* cosb, const float* sinb, unsigned n_pools, unsigned ratio, unsigned coff,
    unsigned d, unsigned rd, unsigned qblk, float eps, unsigned out_base, const int* pos) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float lds[CMP_LDS];
    __shared__ float part[4]; /* PLOW_WG_WAVES */
    d_compress_pool<false>(out, kv, score, ape, gamma, cosb, sinb, n_pools, ratio, coff, d, rd,
                           qblk, eps, out_base, blockIdx.x, gridDim.x, lds, part, pos);
#endif
}

__global__ __launch_bounds__(256, 1) void k_cmp_rot(
    bf16* out, const bf16* kv, const bf16* score, const float* ape, const bf16* gamma,
    const float* cosb, const float* sinb, unsigned n_pools, unsigned ratio, unsigned coff,
    unsigned d, unsigned rd, unsigned qblk, float eps, unsigned out_base, const int* pos) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float lds[CMP_LDS];
    __shared__ float part[4]; /* PLOW_WG_WAVES */
    d_compress_pool<true>(out, kv, score, ape, gamma, cosb, sinb, n_pools, ratio, coff, d, rd,
                          qblk, eps, out_base, blockIdx.x, gridDim.x, lds, part, pos);
#endif
}

/* ------------------------------------------------------------------ host reference */
static int fails = 0;
static unsigned rs = 0xC0FFEEu;
static float rnd() {
    rs ^= rs << 13;
    rs ^= rs >> 17;
    rs ^= rs << 5;
    return (float)(int)(rs >> 8) / 8388608.0f - 1.0f;
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
static float rt(double v) { return b2f(f2b((float)v)); } /* the bf16 round trip */

/* e4m3 (OCP, no FNUZ) round trip of an already-clamped value; mirrors quant_fp8/dequant_fp8. */
static double q_e4m3(double v) {
    if (v == 0.0) return 0.0;
    const double a = std::fabs(v);
    const double sgn = v < 0 ? -1.0 : 1.0;
    int e;
    std::frexp(a, &e); /* a in [0.5,1) * 2^e  ->  exponent of the value is e-1 */
    int ue = e - 1;
    if (ue < -6) ue = -6; /* e4m3 subnormals share the 2^-6 step */
    double step = std::ldexp(1.0, ue - 3);
    double n = std::nearbyint(a / step); /* RNE, matching the hardware convert */
    double r = n * step;
    if (r > 448.0) r = 448.0;
    return sgn * r;
}
static const double FP4_LADDER[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
/* quant_fp4's ladder: round to nearest with TIES AWAY FROM ZERO, the shipped device form
 * (amd_common.h records why it is not RNE and that changing it would move numerics). */
static double q_e2m1(double v) {
    const double a = std::fabs(v);
    static const double CUT[7] = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0};
    int n = 0;
    for (int i = 0; i < 7; i++) n += !(a < CUT[i]);
    return (v < 0 ? -1.0 : 1.0) * FP4_LADDER[n];
}

struct Case {
    unsigned n_pools, ratio, coff, d, rd, qblk, out_base;
    bool rotate;
    int pos; /* < 0 = prefill */
};

static void run(const char* name, Case cs) {
    const unsigned nslot = cs.coff * cs.ratio, kstr = cs.coff * cs.d;
    const bool decode = cs.pos >= 0;
    const unsigned n_src = decode ? nslot : (cs.n_pools * cs.ratio);
    const unsigned n_out = cs.out_base + (decode ? ((unsigned)cs.pos / cs.ratio + 1) : cs.n_pools);
    const float eps = 1e-6f;

    std::vector<bf16> kv((size_t)n_src * kstr), score((size_t)n_src * kstr), gamma(cs.d);
    std::vector<float> ape((size_t)cs.ratio * kstr);
    for (auto& v : kv) v = f2b(rnd() * 2.0f);
    for (auto& v : score) v = f2b(rnd() * 3.0f);
    for (auto& v : gamma) v = f2b(1.0f + rnd() * 0.25f);
    for (auto& v : ape) v = rnd() * 1.5f;
    /* Decode: the ring's score rows must start at -inf so a cold slot self-masks. Rows the
     * caller has actually stashed are the ones filled above; here every row is live. */

    const unsigned npos = 4096;
    std::vector<float> cosb((size_t)npos * (cs.rd / 2)), sinb((size_t)npos * (cs.rd / 2));
    for (unsigned p = 0; p < npos; p++)
        for (unsigned i = 0; i < cs.rd / 2; i++) {
            const double th = (double)p / std::pow(160000.0, 2.0 * i / (double)cs.rd);
            cosb[(size_t)p * (cs.rd / 2) + i] = (float)std::cos(th);
            sinb[(size_t)p * (cs.rd / 2) + i] = (float)std::sin(th);
        }

    std::vector<bf16> out((size_t)n_out * cs.d, 0x7fc1 /* sentinel */);
    std::vector<int> pv(1, cs.pos < 0 ? 0 : cs.pos);
    auto *dkv = up(kv), *dsc = up(score), *dgm = up(gamma), *dout = up(out);
    auto *dape = up(ape), *dcos = up(cosb), *dsin = up(sinb);
    auto* dpos = up(pv);

    auto* k = cs.rotate ? k_cmp_rot : k_cmp;
    hipLaunchKernelGGL(k, dim3(8), dim3(256), 0, 0, dout, dkv, dsc, dape, dgm,
                       dcos, dsin, decode ? 1u : cs.n_pools, cs.ratio, cs.coff, cs.d, cs.rd,
                       cs.qblk, eps, cs.out_base, decode ? dpos : (const int*)nullptr);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(out.data(), dout, out.size() * sizeof(bf16), hipMemcpyDeviceToHost));

    /* ---- reference ---- */
    const unsigned pbase = decode ? ((unsigned)cs.pos / cs.ratio) : cs.out_base;
    const unsigned np = decode ? 1u : cs.n_pools;
    unsigned bad = 0, checked = 0;
    double worst = 0;
    for (unsigned pool = 0; pool < np; pool++) {
        std::vector<double> v(cs.d);
        for (unsigned c = 0; c < cs.d; c++) {
            double mx = -1e300;
            for (unsigned s = 0; s < nslot; s++) {
                const unsigned half = s / cs.ratio, r = s % cs.ratio;
                const int row = decode ? (int)s
                                       : (int)(pool * cs.ratio + r) -
                                             (int)((cs.coff == 2 && half == 0) ? cs.ratio : 0);
                if (row < 0) continue;
                mx = std::fmax(mx, (double)b2f(score[(size_t)row * kstr + half * cs.d + c]) +
                                       (double)ape[(size_t)r * kstr + half * cs.d + c]);
            }
            double den = 0, acc = 0;
            for (unsigned s = 0; s < nslot; s++) {
                const unsigned half = s / cs.ratio, r = s % cs.ratio;
                const int row = decode ? (int)s
                                       : (int)(pool * cs.ratio + r) -
                                             (int)((cs.coff == 2 && half == 0) ? cs.ratio : 0);
                if (row < 0) continue;
                const double p =
                    std::exp((double)b2f(score[(size_t)row * kstr + half * cs.d + c]) +
                             (double)ape[(size_t)r * kstr + half * cs.d + c] - mx);
                den += p;
                acc += (double)b2f(kv[(size_t)row * kstr + half * cs.d + c]) * p;
            }
            v[c] = rt(den > 0 ? acc / den : 0.0); /* kv.to(bf16) */
        }
        double ms = 0;
        for (unsigned c = 0; c < cs.d; c++) ms += v[c] * v[c];
        const double inv = 1.0 / std::sqrt(ms / cs.d + eps);
        for (unsigned c = 0; c < cs.d; c++) v[c] = rt(v[c] * inv * (double)b2f(gamma[c]));
        if (cs.rd) {
            const unsigned c0 = cs.d - cs.rd;
            const size_t tb = (size_t)((pbase + pool) * cs.ratio) * (cs.rd / 2);
            for (unsigned i = 0; i < cs.rd / 2; i++) {
                const double x0 = v[c0 + 2 * i], x1 = v[c0 + 2 * i + 1];
                const double c_ = cosb[tb + i], s_ = sinb[tb + i];
                v[c0 + 2 * i] = rt(x0 * c_ - x1 * s_);
                v[c0 + 2 * i + 1] = rt(x0 * s_ + x1 * c_);
            }
        }
        if (cs.rotate) { /* Hadamard-128, then 1/sqrt(128), one bf16 round trip */
            std::vector<double> h(v);
            for (unsigned stride = 1; stride < 128; stride <<= 1) {
                std::vector<double> n(128);
                for (unsigned i = 0; i < 128; i++) {
                    const unsigned b2 = 2 * stride, base = (i / b2) * b2, rem = i % b2;
                    const unsigned r = rem < stride ? rem : rem - stride;
                    n[i] = rem < stride ? h[base + r] + h[base + stride + r]
                                        : h[base + r] - h[base + stride + r];
                }
                h = n;
            }
            for (unsigned c = 0; c < cs.d; c++) v[c] = rt(h[c] * 0.08838834764831845);
        }
        const unsigned qspan = cs.rotate ? cs.d : (cs.d - cs.rd);
        const double top = cs.rotate ? 6.0 : 448.0;
        const double flr = cs.rotate ? 6.0 * std::ldexp(1.0, -126) : 1e-4;
        for (unsigned b = 0; b < qspan / cs.qblk; b++) {
            double amax = 0;
            for (unsigned i = 0; i < cs.qblk; i++)
                amax = std::fmax(amax, std::fabs(v[b * cs.qblk + i]));
            amax = std::fmax(amax, flr);
            const double s = std::exp2(std::ceil(std::log2(amax / top)));
            for (unsigned i = 0; i < cs.qblk; i++) {
                const double q = std::fmin(std::fmax(v[b * cs.qblk + i] / s, -top), top);
                v[b * cs.qblk + i] = rt((cs.rotate ? q_e2m1(q) : q_e4m3(q)) * s);
            }
        }
        for (unsigned c = 0; c < cs.d; c++) {
            const float got = b2f(out[(size_t)(pbase + pool) * cs.d + c]);
            const float want = (float)v[c];
            checked++;
            const double e = std::fabs((double)got - (double)want);
            const double tol = 1.0 / 256.0 * std::fmax(std::fabs((double)want), 1e-3);
            worst = std::fmax(worst, e / std::fmax(std::fabs((double)want), 1e-6));
            if (!(e <= tol)) bad++;
        }
    }
    /* Rows the kernel must NOT have touched: every slot below out_base keeps its sentinel.
     * op 130 shipped a bug where every chunk wrote slot 0; this is the check for it. */
    unsigned clobber = 0;
    for (unsigned p = 0; p < pbase; p++)
        for (unsigned c = 0; c < cs.d; c++)
            if (out[(size_t)p * cs.d + c] != 0x7fc1) clobber++;

    const bool ok = bad == 0 && clobber == 0;
    std::printf("  %-52s %s  (%u/%u off, worst rel %.2e, %u clobbered)\n", name,
                ok ? "ok " : "FAIL", bad, checked, worst, clobber);
    if (!ok) fails++;
    hipFree(dkv);
    hipFree(dsc);
    hipFree(dgm);
    hipFree(dout);
    hipFree(dape);
    hipFree(dcos);
    hipFree(dsin);
    hipFree(dpos);
}

/* The decode gate: on a step that is NOT a pool boundary the op must write nothing at all.
 * The reference's `should_compress = (start_pos + 1) % ratio == 0` (model.py:350) is the whole
 * of it, and getting it wrong writes a half-formed pool over a good one. */
static void run_gate(const char* name, unsigned ratio, unsigned d, int pos, bool rotate) {
    const unsigned coff = 2, nslot = coff * ratio, kstr = coff * d, rd = 64, qblk = rotate ? 32 : 64;
    const unsigned n_out = (unsigned)pos / ratio + 2;
    std::vector<bf16> kv((size_t)nslot * kstr), score((size_t)nslot * kstr), gamma(d);
    std::vector<float> ape((size_t)ratio * kstr);
    for (auto& v : kv) v = f2b(rnd() * 2.0f);
    for (auto& v : score) v = f2b(rnd() * 3.0f);
    for (auto& v : gamma) v = f2b(1.0f);
    for (auto& v : ape) v = rnd();
    std::vector<float> cosb((size_t)4096 * (rd / 2), 1.0f), sinb((size_t)4096 * (rd / 2), 0.0f);
    std::vector<bf16> out((size_t)n_out * d, 0x7fc1);
    std::vector<int> pv(1, pos);
    auto *dkv = up(kv), *dsc = up(score), *dgm = up(gamma), *dout = up(out);
    auto *dape = up(ape), *dcos = up(cosb), *dsin = up(sinb);
    auto* dpos = up(pv);
    auto* k = rotate ? k_cmp_rot : k_cmp;
    hipLaunchKernelGGL(k, dim3(8), dim3(256), 0, 0, dout, dkv, dsc, dape, dgm, dcos, dsin, 1u,
                       ratio, coff, d, rd, qblk, 1e-6f, 0u, dpos);
    ck(hipGetLastError());
    ck(hipDeviceSynchronize());
    ck(hipMemcpy(out.data(), dout, out.size() * sizeof(bf16), hipMemcpyDeviceToHost));
    unsigned touched = 0;
    for (auto v : out)
        if (v != 0x7fc1) touched++;
    std::printf("  %-52s %s  (%u of %zu words written; must be 0)\n", name,
                touched ? "FAIL" : "ok ", touched, out.size());
    if (touched) fails++;
    hipFree(dkv); hipFree(dsc); hipFree(dgm); hipFree(dout);
    hipFree(dape); hipFree(dcos); hipFree(dsin); hipFree(dpos);
}

int main() {
    std::printf("d_compress_pool — DeepSeek-V4 learned-pooling KV compressor\n");
    /*             n_pools ratio coff  d   rd qblk out_base rotate pos */
    run("HCA  r128 coff1 d512 prefill",     {3, 128, 1, 512, 64, 64, 0, false, -1});
    run("HCA  r128 coff1 d512 out_base=5",  {2, 128, 1, 512, 64, 64, 5, false, -1});
    run("CSA  r4   coff2 d512 prefill+ovl", {6, 4, 2, 512, 64, 64, 0, false, -1});
    run("CSA  r4   coff2 d512 out_base=3",  {4, 4, 2, 512, 64, 64, 3, false, -1});
    run("IDX  r4   coff2 d128 ROTATE fp4",  {6, 4, 2, 128, 64, 32, 0, true, -1});
    run("IDX  r4   coff2 d128 out_base=2",  {3, 4, 2, 128, 64, 32, 2, true, -1});
    /* decode: pos 11 IS a ratio-4 boundary (12 % 4 == 0), pos 9 is not. */
    run("CSA  r4 decode pos=11 (boundary)",  {1, 4, 2, 512, 64, 64, 0, false, 11});
    run("IDX  r4 decode pos=27 ROTATE",      {1, 4, 2, 128, 64, 32, 0, true, 27});
    /* NOT boundaries: 9+1, 10+1 and 26+1 are not multiples of 4. Nothing may be written. */
    run_gate("CSA  r4 decode pos=9  (NOT a boundary)", 4, 512, 9, false);
    run_gate("CSA  r4 decode pos=10 (NOT a boundary)", 4, 512, 10, false);
    run_gate("IDX  r4 decode pos=26 (NOT a boundary)", 4, 128, 26, true);
    std::printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
