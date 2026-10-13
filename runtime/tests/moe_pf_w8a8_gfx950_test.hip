/* moe_pf_w8a8_gfx950_test.hip — PLOW_MOE_PF_W8A8 (ops 85/86 i[7]=2) against the shipped W8A16
 * grouped pair and an FP32 reference, at GLM-5.3-Flash TP8 shapes (H=4096, I=256, 288 experts,
 * top-8, clamped SwiGLU limit 10).
 *
 * Checks, on sampled routed rows (both paths vs FP32 from the same bf16 x and fp8 weights):
 *   GLU intermediate  W8A16 bf16 fu  vs  W8A8 fq*fs (dequantized)
 *   DOWN output       gate * (h . Wd) rows of part
 * reporting per-row relative L2 error (mean / p99 / max), max abs error, and the rows that carry
 * the injected outlier channels separately. Then times each kernel (median of reps).
 *
 *   nix develop --command bash -c '$PLOW_HIPCC --offload-arch=gfx950 -O3 -w -std=c++17 \
 *       -I runtime/amd -I runtime/common -DPLOW_MOE_PREFILL=1 -DPLOW_MOE_TILE_BINSEARCH=1 \
 *       -DPLOW_MOE_PF_EPI=1 -DPLOW_MOE_PF_BRAW=1 -DPLOW_MOE_PF_GH=1 -DPLOW_MOE_PF_DEEP=1 \
 *       -DPLOW_MOE_PF_HDR=1 runtime/tests/moe_pf_w8a8_gfx950_test.hip -o /tmp/moe_w8'
 *   /tmp/moe_w8 [T=4096] [skew=1.0] [hist_file|-] [xn2.bin|-] [tab.bin|-] [weights_dir|-]
 * hist_file: 288 per-expert row counts; routing is sampled from it instead of the zipf.
 * xn2.bin / tab.bin: a real MoE input (bf16 [>=T,H]) and its routing table ({u32 e, f32 g} x k
 * per token), e.g. PLOW_DUMP_ACT of act.xn2 / act.tab after a prefill; "outlier" tokens are
 * then the 2% with the largest row max|x|. weights_dir: weights.bin + scales.bin from
 * extract_experts.py (one layer's experts, one TP slice) instead of synthetic weights. */
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#ifndef PLOW_MOE_PREFILL
#define PLOW_MOE_PREFILL 1
#endif
#include "amd_common.h"
#ifdef __HIP_DEVICE_COMPILE__
#include "op_gemm.h"
#endif
#include "op_moe.h"
#include "moe_pf_w8a8.h"

#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_glu_base(bf16* fu, const bf16* x, const unsigned long long* wtab, const unsigned long long* stab,
                const int* meta, const unsigned* rt, unsigned I, unsigned H, unsigned n_exp, float lim) {
    __shared__ bf16 lds[MPF_LDS_HALVES];
    d_moe_group_glu_pf(fu, x, wtab, stab, meta, rt, I, H, n_exp, PLOW_MOE_ENC_FP8BLK,
                       PLOW_MOE_ACT_SWIGLU_CLAMP, blockIdx.x, gridDim.x, 0.0f, lim, lds);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_down_base(float* part, const bf16* fu, const unsigned long long* wtab, const unsigned long long* stab,
                 const int* meta, const unsigned* rp, const float* rg, unsigned H, unsigned I, unsigned n_exp) {
    __shared__ bf16 lds[MPF_LDS_HALVES];
    d_moe_group_down_pf(part, fu, wtab, stab, meta, rp, rg, H, I, n_exp, PLOW_MOE_ENC_FP8BLK,
                        blockIdx.x, gridDim.x, lds);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_quant(unsigned char* xq, const bf16* x, float* xs, unsigned T, unsigned H) {
#ifdef __HIP_DEVICE_COMPILE__
    d_quant_fp8_block128(xq, x, xs, T, H, blockIdx.x, gridDim.x);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_glu_w8(unsigned char* fq, float* fs, const unsigned char* xq, const float* xs, unsigned T,
              const unsigned long long* wtab, const unsigned long long* stab, const int* meta,
              const unsigned* rt, unsigned I, unsigned H, unsigned n_exp, float lim) {
    __shared__ __attribute__((aligned(16))) unsigned char lds[MW8_LDS_BYTES];
#if PLOW_CDNA4
    d_moe_pf_w8a8<true>(fq, fs, xq, xs, T, wtab, stab, meta, rt, nullptr, I, H, n_exp,
                        PLOW_MOE_ACT_SWIGLU_CLAMP, 0.0f, lim, 0, blockIdx.x, gridDim.x, lds);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_down_w8(void* part, const unsigned char* fq, const float* fs, const unsigned long long* wtab,
               const unsigned long long* stab, const int* meta, const unsigned* rp, const float* rg,
               unsigned H, unsigned I, unsigned n_exp, unsigned part16) {
    __shared__ __attribute__((aligned(16))) unsigned char lds[MW8_LDS_BYTES];
#if PLOW_CDNA4
    d_moe_pf_w8a8<false>(part, nullptr, fq, fs, 0, wtab, stab, meta, rp, rg, H, I, n_exp, 0, 0.0f,
                         0.0f, part16, blockIdx.x, gridDim.x, lds);
#endif
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_glu_sp(unsigned char* fq, float* fs, const bf16* x, unsigned T,
              const unsigned long long* wtab, const unsigned long long* stab, const int* meta,
              const unsigned* rt, unsigned I, unsigned H, unsigned n_exp, float lim) {
    __shared__ __attribute__((aligned(16))) unsigned char lds[MW8_LDS_BYTES];
#if PLOW_CDNA4
    d_moe_pf_w8a8<true, true>(fq, fs, (const unsigned char*)x, nullptr, T, wtab, stab, meta, rt,
                              nullptr, I, H, n_exp, PLOW_MOE_ACT_SWIGLU_CLAMP, 0.0f, lim, 0,
                              blockIdx.x, gridDim.x, lds);
#endif
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_down_sp(void* part, const unsigned char* fq, const float* fs, const unsigned long long* wtab,
               const unsigned long long* stab, const int* meta, const unsigned* rp, const float* rg,
               unsigned H, unsigned I, unsigned n_exp, unsigned part16) {
    __shared__ __attribute__((aligned(16))) unsigned char lds[MW8_LDS_BYTES];
#if PLOW_CDNA4
    d_moe_pf_w8a8<false, true>(part, nullptr, fq, fs, 0, wtab, stab, meta, rp, rg, H, I, n_exp, 0,
                               0.0f, 0.0f, part16, blockIdx.x, gridDim.x, lds);
#endif
}

static float fp8d(unsigned char b) {
    const unsigned s = b >> 7, e = (b >> 3) & 15u, m = b & 7u;
    float v = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, (int)e - 7);
    if (e == 15 && m == 7) v = NAN;
    return s ? -v : v;
}
static unsigned char fp8e(float x) { /* nearest, ties to even mantissa, saturating */
    static float pos[127];
    static bool init = false;
    if (!init) { for (int i = 0; i < 127; i++) pos[i] = fp8d((unsigned char)i); init = true; }
    const unsigned char s = x < 0 ? 0x80 : 0;
    const float a = std::min(fabsf(x), 448.0f);
    int lo = (int)(std::upper_bound(pos, pos + 127, a) - pos) - 1;
    if (lo < 0) lo = 0;
    int b = lo;
    if (lo + 1 < 127) {
        const float d0 = a - pos[lo], d1 = pos[lo + 1] - a;
        b = (d1 < d0 || (d1 == d0 && ((lo + 1) & 1) == 0)) ? lo + 1 : lo;
    }
    return (unsigned char)(s | b);
}
static unsigned short f2b(float f){unsigned u;memcpy(&u,&f,4);u+=0x7fffu+((u>>16)&1u);return (unsigned short)(u>>16);}
static float bf(unsigned short h){unsigned u=(unsigned)h<<16;float f;memcpy(&f,&u,4);return f;}
static unsigned long long rs_ = 0x9e3779b97f4a7c15ull;
static inline unsigned long long xr() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
static inline float ur() { return (float)((xr() >> 40) & 0xffffff) / 16777216.0f; }
static inline float nr() { float s = 0; for (int i = 0; i < 4; i++) s += ur(); return (s - 2.0f) * 1.732f; }

struct Stat { std::vector<double> rel; double maxabs = 0; };
static void report(const char* name, Stat& s) {
    std::sort(s.rel.begin(), s.rel.end());
    const double mean = std::accumulate(s.rel.begin(), s.rel.end(), 0.0) / std::max<size_t>(1, s.rel.size());
    const double p99 = s.rel.empty() ? 0 : s.rel[(size_t)(0.99 * (s.rel.size() - 1))];
    printf("  %-34s rows=%5zu relL2 mean=%.3e p99=%.3e max=%.3e maxabs=%.3e\n", name, s.rel.size(),
           mean, p99, s.rel.empty() ? 0 : s.rel.back(), s.maxabs);
}
static void acc_row(Stat& s, const float* a, const double* ref, unsigned n) {
    double num = 0, den = 0;
    for (unsigned c = 0; c < n; c++) {
        const double d = (double)a[c] - ref[c];
        num += d * d; den += ref[c] * ref[c];
        s.maxabs = std::max(s.maxabs, fabs(d));
    }
    s.rel.push_back(sqrt(num / std::max(den, 1e-30)));
}

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? (unsigned)atoi(argv[1]) : 4096u;
    const float skew = argc > 2 ? (float)atof(argv[2]) : 1.0f;
    const unsigned H = 4096, I = 256, E = 288, K = 8, KBH = H / 128, KBI = I / 128;
    const float lim = 10.0f;
    printf("W8A8 MoE prefill test: T=%u H=%u I=%u E=%u k=%u skew=%.2f\n", T, H, I, E, K, skew);

    /* ---- routing: zipf(skew) expert popularity or a measured histogram; distinct per token */
    std::vector<double> pe(E);
    bool hist = false;
    auto arg = [&](int i) -> const char* { return argc > i && strcmp(argv[i], "-") ? argv[i] : nullptr; };
    if (arg(3)) {
        FILE* f = fopen(argv[3], "r");
        if (f) { for (unsigned e = 0; e < E; e++) { double v = 0; if (fscanf(f, "%lf", &v) != 1) v = 0; pe[e] = v + 0.01; } fclose(f); hist = true; }
    }
    if (!hist) { for (unsigned e = 0; e < E; e++) pe[e] = 1.0 / pow((double)(e + 1), skew); }
    std::vector<unsigned> perm(E); std::iota(perm.begin(), perm.end(), 0u);
    for (unsigned e = E - 1; e > 0; e--) std::swap(perm[e], perm[xr() % (e + 1)]);
    std::vector<double> cdf(E); double tot = 0;
    for (unsigned e = 0; e < E; e++) { tot += pe[e]; cdf[e] = tot; }
    std::vector<unsigned> tok_e((size_t)T * K); std::vector<float> tok_g((size_t)T * K);
    for (unsigned t = 0; t < T; t++) {
        float gs = 0;
        for (unsigned s = 0; s < K; s++) {
            unsigned e;
            for (;;) {
                const double u = ur() * tot;
                e = perm[(unsigned)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin()) % E];
                bool dup = false; for (unsigned q = 0; q < s; q++) dup |= tok_e[(size_t)t * K + q] == e;
                if (!dup) break;
            }
            tok_e[(size_t)t * K + s] = e; tok_g[(size_t)t * K + s] = 0.2f + ur(); gs += tok_g[(size_t)t * K + s];
        }
        for (unsigned s = 0; s < K; s++) tok_g[(size_t)t * K + s] *= 2.5f / gs;
    }
    if (arg(5)) {
        FILE* f = fopen(argv[5], "rb");
        std::vector<unsigned> tb((size_t)T * K * 2);
        if (!f || fread(tb.data(), 8, (size_t)T * K, f) != (size_t)T * K) { printf("bad tab file\n"); return 1; }
        fclose(f);
        for (size_t i = 0; i < (size_t)T * K; i++) {
            tok_e[i] = tb[2 * i] % E;
            memcpy(&tok_g[i], &tb[2 * i + 1], 4);
        }
        printf("  routing from %s\n", argv[5]);
    }
    /* ---- align (host): the d_moe_align_pf contract, rows padded per expert to MPF_BM */
    std::vector<unsigned> cnt(E, 0);
    for (size_t i = 0; i < tok_e.size(); i++) cnt[tok_e[i]]++;
    std::vector<int> meta(3 * E + 1);
    unsigned off = 0, tiles = 0;
    for (unsigned e = 0; e < E; e++) {
        meta[e] = (int)off; meta[E + e] = (int)cnt[e]; meta[2 * E + e] = (int)tiles;
        const unsigned tl = (cnt[e] + MPF_BM - 1) / MPF_BM; tiles += tl; off += tl * MPF_BM;
    }
    meta[3 * E] = (int)tiles;
    const unsigned rows = off;
    std::vector<unsigned> rt(rows, PLOW_EXPERT_UNUSED), rp(rows, PLOW_EXPERT_UNUSED);
    std::vector<float> rg(rows, 0.0f);
    std::vector<unsigned> cur(E, 0);
    std::vector<unsigned> slot_row((size_t)T * K);
    for (unsigned t = 0; t < T; t++)
        for (unsigned s = 0; s < K; s++) {
            const unsigned e = tok_e[(size_t)t * K + s], r = meta[e] + cur[e]++;
            rt[r] = t; rp[r] = t * K + s; rg[r] = tok_g[(size_t)t * K + s]; slot_row[(size_t)t * K + s] = r;
        }
    {
        std::vector<unsigned> c(cnt); std::sort(c.begin(), c.end());
        unsigned empty = 0, tiny = 0; for (unsigned v : c) { empty += v == 0; tiny += v > 0 && v < 16; }
        printf("  rows %u (live %u, pad %.1f%%), tiles %u, experts empty=%u <16 rows=%u, min/med/max rows %u/%u/%u\n",
               rows, T * K, 100.0 * (rows - T * K) / rows, tiles, empty, tiny, c[0], c[E / 2], c[E - 1]);
    }

    /* ---- activations: post-norm bf16 ~N(0,1) with outlier channels on some tokens */
    std::vector<unsigned short> x((size_t)T * H);
    std::vector<char> outl(T, 0);
    for (unsigned t = 0; t < T; t++) {
        outl[t] = (t % 61) == 7;
        for (unsigned h = 0; h < H; h++) {
            float v = nr();
            if (h % 509 == 3) v *= 8.0f;          /* hot channels, every token */
            if (outl[t] && h % 97 == 5) v *= 60.0f; /* outlier tokens */
            x[(size_t)t * H + h] = f2b(v);
        }
    }
    if (arg(4)) {
        FILE* f = fopen(argv[4], "rb");
        if (!f || fread(x.data(), 2, x.size(), f) != x.size()) { printf("bad xn2 file\n"); return 1; }
        fclose(f);
        std::vector<float> rmax(T);
        for (unsigned t = 0; t < T; t++) {
            float m = 0; for (unsigned h = 0; h < H; h++) m = std::max(m, fabsf(bf(x[(size_t)t * H + h])));
            rmax[t] = m;
        }
        std::vector<float> srt(rmax); std::sort(srt.begin(), srt.end());
        const float thr = srt[(size_t)(0.98 * (T - 1))];
        for (unsigned t = 0; t < T; t++) outl[t] = rmax[t] >= thr && rmax[t] > 0;
        printf("  activations from %s: row max|x| median %.3g p98 %.3g max %.3g\n", argv[4], srt[T / 2], thr, srt[T - 1]);
    }
    /* ---- weights: fp8 payload ~N(0,100) clipped, block scales ~2e-4 */
    unsigned char tbl[65536];
    for (int i = 0; i < 65536; i++) tbl[i] = fp8e(std::max(-448.0f, std::min(448.0f, nr() * 100.0f)));
    const size_t wsz = (size_t)I * H; /* gate, up and down are all I*H bytes */
    std::vector<unsigned char> W((size_t)E * 3 * wsz);
    for (size_t i = 0; i < W.size(); i++) W[i] = tbl[xr() & 0xffff];
    const size_t ssz = (size_t)KBI * KBH;
    std::vector<float> S((size_t)E * 3 * ssz);
    for (size_t i = 0; i < S.size(); i++) S[i] = 2e-4f * (0.5f + ur());
    if (arg(6)) {
        std::string d = argv[6];
        FILE* f = fopen((d + "/weights.bin").c_str(), "rb");
        FILE* g = fopen((d + "/scales.bin").c_str(), "rb");
        if (!f || !g || fread(W.data(), 1, W.size(), f) != W.size() || fread(S.data(), 4, S.size(), g) != S.size()) {
            printf("bad weights dir\n"); return 1;
        }
        fclose(f); fclose(g);
        printf("  weights from %s (scale[0]=%.3g)\n", argv[6], S[0]);
    }
    /* layout: expert e, mat m (0 gate [I,H], 1 up [I,H], 2 down [H,I]) */
    unsigned char* dW; float* dS;
    HC(hipMalloc(&dW, W.size())); HC(hipMemcpy(dW, W.data(), W.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&dS, S.size() * 4)); HC(hipMemcpy(dS, S.data(), S.size() * 4, hipMemcpyHostToDevice));
    std::vector<unsigned long long> wtab(E * 3), stab(E * 3);
    for (unsigned e = 0; e < E; e++)
        for (unsigned m = 0; m < 3; m++) {
            wtab[e * 3 + m] = (unsigned long long)(size_t)(dW + ((size_t)e * 3 + m) * wsz);
            stab[e * 3 + m] = (unsigned long long)(size_t)(dS + ((size_t)e * 3 + m) * ssz);
        }
    auto up = [](const void* h, size_t n) { void* d; HC(hipMalloc(&d, n)); HC(hipMemcpy(d, h, n, hipMemcpyHostToDevice)); return d; };
    auto* dwt = (unsigned long long*)up(wtab.data(), wtab.size() * 8);
    auto* dst = (unsigned long long*)up(stab.data(), stab.size() * 8);
    auto* dmeta = (int*)up(meta.data(), meta.size() * 4);
    auto* drt = (unsigned*)up(rt.data(), rows * 4);
    auto* drp = (unsigned*)up(rp.data(), rows * 4);
    auto* drg = (float*)up(rg.data(), rows * 4);
    auto* dx = (bf16*)up(x.data(), x.size() * 2);
    bf16* dfu; float* dpart; unsigned char *dxq, *dfq; float *dxs, *dfs; void* dpart16;
    HC(hipMalloc(&dfu, (size_t)rows * I * 2)); HC(hipMemset(dfu, 0, (size_t)rows * I * 2));
    HC(hipMalloc(&dpart, (size_t)T * K * H * 4));
    HC(hipMalloc(&dpart16, (size_t)T * K * H * 2));
    HC(hipMalloc(&dxq, (size_t)T * H)); HC(hipMalloc(&dxs, (size_t)T * KBH * 4));
    HC(hipMalloc(&dfq, (size_t)rows * I)); HC(hipMalloc(&dfs, (size_t)rows * KBI * 4));
    unsigned char* dfq2; float* dfs2; HC(hipMalloc(&dfq2, (size_t)rows * I * 2)); HC(hipMalloc(&dfs2, (size_t)rows * KBI * 4));

    const unsigned G = getenv("MW8_G") ? (unsigned)atoi(getenv("MW8_G")) : 512u;
    auto run_base_glu = [&] { hipLaunchKernelGGL(t_glu_base, dim3(G), dim3(PLOW_THREADS), 0, 0, dfu, dx, dwt, dst, dmeta, drt, I, H, E, lim); };
    auto run_base_down = [&] { hipLaunchKernelGGL(t_down_base, dim3(G), dim3(PLOW_THREADS), 0, 0, dpart, dfu, dwt, dst, dmeta, drp, drg, H, I, E); };
    auto run_quant = [&] { hipLaunchKernelGGL(t_quant, dim3(G), dim3(PLOW_THREADS), 0, 0, dxq, dx, dxs, T, H); };
    auto run_w8_glu = [&] { hipLaunchKernelGGL(t_glu_w8, dim3(G), dim3(PLOW_THREADS), 0, 0, dfq, dfs, dxq, dxs, T, dwt, dst, dmeta, drt, I, H, E, lim); };
    auto run_sp_glu = [&] { hipLaunchKernelGGL(t_glu_sp, dim3(G), dim3(PLOW_THREADS), 0, 0, dfq2, dfs2, dx, T, dwt, dst, dmeta, drt, I, H, E, lim); };
    auto run_sp_down = [&](unsigned p16) { hipLaunchKernelGGL(t_down_sp, dim3(G), dim3(PLOW_THREADS), 0, 0, p16 ? dpart16 : (void*)dpart, dfq2, dfs2, dwt, dst, dmeta, drp, drg, H, I, E, p16); };
    auto run_w8_down = [&](unsigned p16) { hipLaunchKernelGGL(t_down_w8, dim3(G), dim3(PLOW_THREADS), 0, 0, p16 ? dpart16 : (void*)dpart, dfq, dfs, dwt, dst, dmeta, drp, drg, H, I, E, p16); };

    /* ---- FP32 reference on sampled slots */
    const unsigned NS = 384;
    std::vector<unsigned> samp;
    for (unsigned i = 0; i < NS / 2; i++) samp.push_back((unsigned)(xr() % ((size_t)T * K)));
    for (unsigned t = 0; t < T && samp.size() < NS; t++) if (outl[t]) samp.push_back(t * K + (unsigned)(xr() % K));
    std::vector<double> href((size_t)samp.size() * I), yref((size_t)samp.size() * H);
    for (size_t si = 0; si < samp.size(); si++) {
        const unsigned slot = samp[si], t = slot / K, e = tok_e[slot];
        const unsigned char* Wg = W.data() + ((size_t)e * 3 + 0) * wsz; const float* Sg = S.data() + ((size_t)e * 3 + 0) * ssz;
        const unsigned char* Wu = W.data() + ((size_t)e * 3 + 1) * wsz; const float* Su = S.data() + ((size_t)e * 3 + 1) * ssz;
        const unsigned char* Wd = W.data() + ((size_t)e * 3 + 2) * wsz; const float* Sd = S.data() + ((size_t)e * 3 + 2) * ssz;
        for (unsigned n = 0; n < I; n++) {
            double g = 0, u = 0;
            for (unsigned h = 0; h < H; h++) {
                const double xv = bf(x[(size_t)t * H + h]);
                g += xv * fp8d(Wg[(size_t)n * H + h]) * Sg[(n / 128) * KBH + h / 128];
                u += xv * fp8d(Wu[(size_t)n * H + h]) * Su[(n / 128) * KBH + h / 128];
            }
            const double gc = std::min(g, (double)lim), uc = std::min(std::max(u, -(double)lim), (double)lim);
            href[si * I + n] = gc / (1.0 + exp(-gc)) * uc;
        }
        for (unsigned n = 0; n < H; n++) {
            double y = 0;
            for (unsigned i = 0; i < I; i++) y += href[si * I + i] * fp8d(Wd[(size_t)n * I + i]) * Sd[(n / 128) * KBI + i / 128];
            yref[si * H + n] = y * tok_g[slot];
        }
    }

    /* ---- run both paths once, compare */
    run_base_glu(); run_base_down(); HC(hipDeviceSynchronize());
    std::vector<unsigned short> fu_h((size_t)rows * I); HC(hipMemcpy(fu_h.data(), dfu, fu_h.size() * 2, hipMemcpyDeviceToHost));
    std::vector<float> part_b((size_t)T * K * H); HC(hipMemcpy(part_b.data(), dpart, part_b.size() * 4, hipMemcpyDeviceToHost));
    HC(hipMemset(dpart, 0xff, (size_t)T * K * H * 4));
    run_quant(); run_w8_glu(); run_w8_down(0); HC(hipDeviceSynchronize());
    std::vector<unsigned char> fq_h((size_t)rows * I); HC(hipMemcpy(fq_h.data(), dfq, fq_h.size(), hipMemcpyDeviceToHost));
    std::vector<float> fs_h((size_t)rows * KBI); HC(hipMemcpy(fs_h.data(), dfs, fs_h.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> part_w((size_t)T * K * H); HC(hipMemcpy(part_w.data(), dpart, part_w.size() * 4, hipMemcpyDeviceToHost));
    run_w8_down(1); HC(hipDeviceSynchronize());
    std::vector<unsigned short> part_w16((size_t)T * K * H); HC(hipMemcpy(part_w16.data(), dpart16, part_w16.size() * 2, hipMemcpyDeviceToHost));
    HC(hipMemset(dpart, 0xff, (size_t)T * K * H * 4));
    run_sp_glu(); run_sp_down(0); HC(hipDeviceSynchronize());
    std::vector<unsigned char> fq2_h((size_t)rows * I * 2); HC(hipMemcpy(fq2_h.data(), dfq2, fq2_h.size(), hipMemcpyDeviceToHost));
    std::vector<float> fs2_h((size_t)rows * KBI); HC(hipMemcpy(fs2_h.data(), dfs2, fs2_h.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> part_s((size_t)T * K * H); HC(hipMemcpy(part_s.data(), dpart, part_s.size() * 4, hipMemcpyDeviceToHost));
    run_sp_down(1); HC(hipDeviceSynchronize());
    std::vector<unsigned short> part_s16((size_t)T * K * H); HC(hipMemcpy(part_s16.data(), dpart16, part_s16.size() * 2, hipMemcpyDeviceToHost));

    /* quantized-input check: xq*xs vs x */
    {
        std::vector<unsigned char> xq_h((size_t)T * H); HC(hipMemcpy(xq_h.data(), dxq, xq_h.size(), hipMemcpyDeviceToHost));
        std::vector<float> xs_h((size_t)T * KBH); HC(hipMemcpy(xs_h.data(), dxs, xs_h.size() * 4, hipMemcpyDeviceToHost));
        Stat sx, sxo;
        std::vector<float> a(H); std::vector<double> r(H);
        for (unsigned t = 0; t < T; t++) {
            for (unsigned h = 0; h < H; h++) { a[h] = fp8d(xq_h[(size_t)t * H + h]) * xs_h[(size_t)(h / 128) * T + t]; r[h] = bf(x[(size_t)t * H + h]); }
            acc_row(outl[t] ? sxo : sx, a.data(), r.data(), H);
        }
        printf("quantized input x (1x128 fp8) vs bf16 x:\n");
        report("x  ordinary tokens", sx); report("x  outlier tokens", sxo);
    }
    Stat gb, gw, gbo, gwo, db, dw, dbo, dwo, dw16, gs, gso, ds, dso, ds16;
    unsigned nan_s = 0;
    std::vector<float> a(H);
    unsigned nan_w = 0;
    for (size_t si = 0; si < samp.size(); si++) {
        const unsigned slot = samp[si], r = slot_row[slot];
        const bool o = outl[slot / K];
        for (unsigned n = 0; n < I; n++) a[n] = bf(fu_h[(size_t)r * I + n]);
        acc_row(o ? gbo : gb, a.data(), &href[si * I], I);
        for (unsigned n = 0; n < I; n++) a[n] = fp8d(fq_h[(size_t)r * I + n]) * fs_h[(size_t)r * KBI + n / 128];
        acc_row(o ? gwo : gw, a.data(), &href[si * I], I);
        acc_row(o ? dbo : db, &part_b[(size_t)slot * H], &yref[si * H], H);
        for (unsigned n = 0; n < H; n++) nan_w += std::isnan(part_w[(size_t)slot * H + n]);
        acc_row(o ? dwo : dw, &part_w[(size_t)slot * H], &yref[si * H], H);
        for (unsigned n = 0; n < H; n++) a[n] = bf(part_w16[(size_t)slot * H + n]);
        acc_row(dw16, a.data(), &yref[si * H], H);
        for (unsigned n = 0; n < I; n++)
            a[n] = (fp8d(fq2_h[(size_t)r * 2 * I + n]) + fp8d(fq2_h[(size_t)r * 2 * I + I + n]) / 16.0f) * fs2_h[(size_t)r * KBI + n / 128];
        acc_row(o ? gso : gs, a.data(), &href[si * I], I);
        for (unsigned n = 0; n < H; n++) nan_s += std::isnan(part_s[(size_t)slot * H + n]);
        acc_row(o ? dso : ds, &part_s[(size_t)slot * H], &yref[si * H], H);
        for (unsigned n = 0; n < H; n++) a[n] = bf(part_s16[(size_t)slot * H + n]);
        acc_row(ds16, a.data(), &yref[si * H], H);
    }
    printf("GLU intermediate h vs FP32:\n");
    report("W8A16 bf16 fu   ordinary", gb); report("W8A8  fq*fs     ordinary", gw);
    report("W8A16 bf16 fu   outlier tok", gbo); report("W8A8  fq*fs     outlier tok", gwo);
    report("SPLIT (hi+lo/16)*fs ordinary", gs); report("SPLIT (hi+lo/16)*fs outlier", gso);
    printf("DOWN output (gate * h.Wd) vs FP32:\n");
    report("W8A16 part f32  ordinary", db); report("W8A8  part f32  ordinary", dw);
    report("W8A16 part f32  outlier tok", dbo); report("W8A8  part f32  outlier tok", dwo);
    report("W8A8  part bf16 all", dw16);
    report("SPLIT part f32  ordinary", ds); report("SPLIT part f32  outlier tok", dso);
    report("SPLIT part bf16 all", ds16);
    printf("  SPLIT NaN in sampled rows: %u\n", nan_s);
    /* every live slot written (no NaN poison left), pad rows untouched */
    unsigned unwritten = 0;
    for (size_t i = 0; i < part_w.size(); i += 61) unwritten += std::isnan(part_w[i]);
    printf("  W8A8 part slots still poisoned (sampled every 61st): %u, NaN in sampled rows: %u\n", unwritten, nan_w);

    /* ---- timing */
    auto timeit = [&](const char* name, auto fn, double flop) {
        hipEvent_t a0, a1; HC(hipEventCreate(&a0)); HC(hipEventCreate(&a1));
        std::vector<float> v;
        for (int r = 0; r < 12; r++) {
            HC(hipEventRecord(a0)); fn(); HC(hipEventRecord(a1)); HC(hipEventSynchronize(a1));
            float ms; HC(hipEventElapsedTime(&ms, a0, a1)); if (r >= 2) v.push_back(ms);
        }
        std::sort(v.begin(), v.end());
        const double us = v[v.size() / 2] * 1000.0;
        printf("  %-22s %9.1f us  %7.1f TF/s\n", name, us, flop > 0 ? flop / us * 1e-6 : 0.0);
        return us;
    };
    const double fg = 2.0 * T * K * 2.0 * I * H, fd = 2.0 * T * K * (double)I * H;
    printf("timing (median of 10, grid %u):\n", G);
    const double b1 = timeit("W8A16 GLU (op85)", run_base_glu, fg);
    const double b2 = timeit("W8A16 DOWN (op86)", run_base_down, fd);
    const double w0 = timeit("quant x 1x128 (op184)", run_quant, 0);
    const double w1 = timeit("W8A8 GLU+quant", run_w8_glu, fg);
    const double w2 = timeit("W8A8 DOWN f32 part", [&] { run_w8_down(0); }, fd);
    const double w3 = timeit("W8A8 DOWN bf16 part", [&] { run_w8_down(1); }, fd);
    const double s1 = timeit("SPLIT GLU (bf16 in)", run_sp_glu, fg);
    const double s2 = timeit("SPLIT DOWN f32 part", [&] { run_sp_down(0); }, fd);
    const double s3 = timeit("SPLIT DOWN bf16 part", [&] { run_sp_down(1); }, fd);
    printf("  pair: W8A16 %.1f us -> SPLIT %.1f us = %.2fx ; bf16 part %.1f us = %.2fx\n", b1 + b2,
           s1 + s2, (b1 + b2) / (s1 + s2), s1 + s3, (b1 + b2) / (s1 + s3));
    printf("  pair: W8A16 %.1f us  ->  W8A8 %.1f us (quant incl.)  = %.2fx ; bf16 part %.1f us\n",
           b1 + b2, w0 + w1 + w2, (b1 + b2) / (w0 + w1 + w2), w0 + w1 + w3);
    return 0;
}
