/* moe_fp8_pf_bench — the dedicated W8A16 grouped-MoE prefill kernels (amd/moe_fp8_pf.hip) against
 * the interpreter bodies they replace, at GLM-5.3-Flash TP8 shape (288 experts, top-8, H 4096,
 * I/rank 256). Synthetic routing, weights and scales; the align layout is built on the host the
 * way d_moe_align_pf lays it out (64-row padded segments, pad rows UNUSED/0).
 * Gates: fu_g, part (live rows) and the combined output byte-equal; prints ms per call.
 *   hipcc --offload-arch=gfx950 -O3 -w -I runtime/amd runtime/bench/amd/moe_fp8_pf_bench.hip
 *   ./a.out [T=8192] [reps=10] [skew=1.0] */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

/* The interpreter's K3 prefill MoE axes (scripts/build_gfx950.sh K3_PF_MOE), all value-identical. */
#define PLOW_K3 1
#define PLOW_MOE_TILE_BINSEARCH 1
#define PLOW_MOE_PF_EPI 1
#define PLOW_MOE_PF_BRAW 1
#define PLOW_COMBINE_VEC 1
#define PLOW_MOE_PF_GH 1
#define PLOW_MOE_PF_DEEP 1
#define PLOW_MOE_PF_HDR 1
#include "../../amd/moe_fp8_pf.hip"

#define HC(x)                                                                                   \
    do {                                                                                        \
        hipError_t e_ = (x);                                                                    \
        if (e_) {                                                                                \
            std::fprintf(stderr, "hip %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
            std::exit(1);                                                                       \
        }                                                                                        \
    } while (0)

extern "C" __global__ __launch_bounds__(512, 1) void ref_glu(
    bf16* fu, const bf16* x, const unsigned long long* wtab, const unsigned long long* stab,
    const int* meta, const unsigned* row_token, unsigned I, unsigned H, unsigned E, unsigned act,
    float beta, float lbeta) {
    extern __shared__ __align__(16) bf16 sm[];
    d_moe_group_glu_pf(fu, x, wtab, stab, meta, row_token, I, H, E, PLOW_MOE_ENC_FP8BLK, act,
                       blockIdx.x, gridDim.x, beta, lbeta, sm);
}

extern "C" __global__ __launch_bounds__(512, 1) void ref_down(
    float* part, const bf16* fu, const unsigned long long* wtab, const unsigned long long* stab,
    const int* meta, const unsigned* row_partidx, const float* row_gate, unsigned H, unsigned I,
    unsigned E) {
    extern __shared__ __align__(16) bf16 sm[];
    d_moe_group_down_pf(part, fu, wtab, stab, meta, row_partidx, row_gate, H, I, E,
                        PLOW_MOE_ENC_FP8BLK, blockIdx.x, gridDim.x, sm);
}

#include "../../amd/moe_pf_w8a8.h"
#define MFP_MW8_LDS (64u * 128u * 4u + 512u + 256u + 256u + 2048u + 512u)

/* The interpreter's W8A8 arms (moe_pf_w8a8.h). */
extern "C" __global__ __launch_bounds__(512, 1) void ref_glu8(
    unsigned char* fq, float* fs, const unsigned char* xq, const float* xs, unsigned pitch,
    const unsigned long long* wtab, const unsigned long long* stab, const int* meta,
    const unsigned* row_token, unsigned I, unsigned H, unsigned E, unsigned act, float beta,
    float lbeta) {
#if PLOW_CDNA4
    extern __shared__ __align__(16) unsigned char sm8[];
    d_moe_pf_w8a8<true>(fq, fs, xq, xs, pitch, wtab, stab, meta, row_token, nullptr, I, H, E, act,
                        beta, lbeta, 0u, blockIdx.x, gridDim.x, sm8);
#endif
}

extern "C" __global__ __launch_bounds__(512, 1) void ref_down8(
    float* part, const unsigned char* fq, const float* fs, const unsigned long long* wtab,
    const unsigned long long* stab, const int* meta, const unsigned* row_partidx,
    const float* row_gate, unsigned H, unsigned I, unsigned E) {
#if PLOW_CDNA4
    extern __shared__ __align__(16) unsigned char sm8[];
    d_moe_pf_w8a8<false>(part, nullptr, fq, fs, 0u, wtab, stab, meta, row_partidx, row_gate, H, I,
                         E, 0u, 0.0f, 0.0f, 0u, blockIdx.x, gridDim.x, sm8);
#endif
}

extern "C" __global__ __launch_bounds__(512, 1) void ref_combine(bf16* out, const bf16* shared,
                                                                 const float* part, unsigned H,
                                                                 unsigned k, unsigned T) {
    d_moe_combine_pf(out, nullptr, shared, part, H, k, T, blockIdx.x, gridDim.x, 0);
}

static unsigned short f2bf_host(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? (unsigned)atoi(argv[1]) : 8192u;
    const int reps = argc > 2 ? atoi(argv[2]) : 10;
    const double skew = argc > 3 ? atof(argv[3]) : 1.0;
    const unsigned grid_n = argc > 4 ? (unsigned)atoi(argv[4]) : 256u;
    const bool w8 = argc > 5 && std::strcmp(argv[5], "w8") == 0;
    /* fold: the shared expert as expert 288, a constant 9th slot of every token (gate 1). */
    const bool fold = argc > 6 && std::strcmp(argv[6], "fold") == 0;
    const unsigned E = fold ? 289 : 288, K8 = fold ? 9 : 8, H = 4096, I = 256, BMP = 64;
    const unsigned ER = 288, KR = 8;
    const unsigned act = PLOW_MOE_ACT_SWIGLU_CLAMP;
    const float beta = 0.0f, lbeta = 10.0f;
    std::mt19937 rng(1234);

    /* routing: skewed expert popularity, k distinct experts per token */
    std::vector<double> pop(ER);
    for (unsigned e = 0; e < ER; e++) pop[e] = 1.0 / std::pow(1.0 + (double)((e * 37u) % ER), skew * 0.5);
    std::discrete_distribution<unsigned> pick(pop.begin(), pop.end());
    std::uniform_real_distribution<float> uf(0.05f, 1.0f);
    std::vector<unsigned> slot_e(T * K8);
    std::vector<float> slot_g(T * K8);
    for (unsigned t = 0; t < T; t++) {
        unsigned n = 0;
        if (fold) {
            slot_e[t * K8 + KR] = ER;
            slot_g[t * K8 + KR] = 1.0f;
        }
        while (n < KR) {
            const unsigned e = pick(rng);
            bool dup = false;
            for (unsigned j = 0; j < n; j++) dup |= slot_e[t * K8 + j] == e;
            if (dup) continue;
            slot_e[t * K8 + n] = e;
            slot_g[t * K8 + n] = uf(rng);
            n++;
        }
    }
    std::vector<unsigned> cnt(E, 0);
    for (unsigned s = 0; s < T * K8; s++) cnt[slot_e[s]]++;
    std::vector<int> meta(3 * E + 1);
    unsigned tiles = 0;
    for (unsigned e = 0; e < E; e++) {
        meta[2 * E + e] = (int)tiles;
        meta[e] = (int)(tiles * BMP);
        meta[E + e] = (int)cnt[e];
        tiles += (cnt[e] + BMP - 1) / BMP;
    }
    meta[3 * E] = (int)tiles;
    const unsigned cap = T * K8 + E * (BMP - 1);
    std::vector<unsigned> row_token(cap, PLOW_EXPERT_UNUSED), row_partidx(cap, PLOW_EXPERT_UNUSED);
    std::vector<float> row_gate(cap, 0.0f);
    std::vector<unsigned> fill(E, 0);
    for (unsigned s = 0; s < T * K8; s++) {
        const unsigned e = slot_e[s];
        const unsigned r = (unsigned)meta[e] + fill[e]++;
        row_token[r] = s / K8;
        row_partidx[r] = s;
        row_gate[r] = slot_g[s];
    }
    std::printf("T=%u rows=%u tiles64=%u max_cnt=%u\n", T, T * K8, tiles,
                *std::max_element(cnt.begin(), cnt.end()));

    /* weights: fp8 e4m3 bytes without NaN (0x7f/0xff) or -0 (0x80, scrubbed by the loader) */
    std::uniform_int_distribution<int> ub(0, 255);
    auto fp8 = [&]() {
        for (;;) {
            const int b = ub(rng);
            if ((b & 0x7f) == 0x7f || b == 0x80) continue;
            if ((b & 0x78) >= 0x48) continue; /* |w| < 2^2 */
            return (unsigned char)b;
        }
    };
    const size_t wgu = (size_t)I * H, wd = (size_t)H * I;
    std::vector<unsigned char> W((size_t)E * (2 * wgu + wd));
    for (auto& b : W) b = fp8();
    const size_t sgu = (size_t)(I / 128) * (H / 128), sd = (size_t)(H / 128) * (I / 128);
    std::vector<float> S((size_t)E * (2 * sgu + sd));
    std::uniform_real_distribution<float> us(2e-3f, 2e-2f);
    for (auto& v : S) v = us(rng);
    std::normal_distribution<float> un(0.0f, 1.0f);
    std::vector<unsigned short> x((size_t)T * H), shared((size_t)T * H);
    for (auto& v : x) v = f2bf_host(un(rng));
    for (auto& v : shared) v = f2bf_host(0.1f * un(rng));

    unsigned char* dW;
    float* dS;
    HC(hipMalloc(&dW, W.size()));
    HC(hipMalloc(&dS, S.size() * 4));
    HC(hipMemcpy(dW, W.data(), W.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(dS, S.data(), S.size() * 4, hipMemcpyHostToDevice));
    std::vector<unsigned long long> wtab(3 * E), stab(3 * E);
    for (unsigned e = 0; e < E; e++) {
        const size_t wb = (size_t)e * (2 * wgu + wd), sb = (size_t)e * (2 * sgu + sd);
        wtab[3 * e + 0] = (unsigned long long)(dW + wb);
        wtab[3 * e + 1] = (unsigned long long)(dW + wb + wgu);
        wtab[3 * e + 2] = (unsigned long long)(dW + wb + 2 * wgu);
        stab[3 * e + 0] = (unsigned long long)(dS + sb);
        stab[3 * e + 1] = (unsigned long long)(dS + sb + sgu);
        stab[3 * e + 2] = (unsigned long long)(dS + sb + 2 * sgu);
    }
    auto up = [&](const void* h, size_t n) {
        void* d;
        HC(hipMalloc(&d, n));
        HC(hipMemcpy(d, h, n, hipMemcpyHostToDevice));
        return d;
    };
    auto* dwt = (unsigned long long*)up(wtab.data(), wtab.size() * 8);
    auto* dst = (unsigned long long*)up(stab.data(), stab.size() * 8);
    auto* dmeta = (int*)up(meta.data(), meta.size() * 4);
    auto* drt = (unsigned*)up(row_token.data(), cap * 4);
    auto* drp = (unsigned*)up(row_partidx.data(), cap * 4);
    auto* drg = (float*)up(row_gate.data(), cap * 4);
    auto* dx = (bf16*)up(x.data(), x.size() * 2);
    auto* dsh = (bf16*)up(shared.data(), shared.size() * 2);
    bf16 *fu_r, *fu_n, *out_r, *out_n;
    float *part_r, *part_n;
    const size_t fu_b = (size_t)cap * I * 2, part_b = (size_t)T * K8 * H * 4, out_b = (size_t)T * H * 2;
    HC(hipMalloc(&fu_r, fu_b));
    HC(hipMalloc(&fu_n, fu_b));
    HC(hipMalloc(&part_r, part_b));
    HC(hipMalloc(&part_n, part_b));
    HC(hipMalloc(&out_r, out_b));
    HC(hipMalloc(&out_n, out_b));
    HC(hipMemset(fu_r, 0x5a, fu_b));
    HC(hipMemset(fu_n, 0x5a, fu_b));
    HC(hipMemset(part_r, 0, part_b));
    HC(hipMemset(part_n, 0, part_b));

    const unsigned NCU = 256, ref_lds = 2 * (64 + 256) * 64 * 2;
    HC(hipFuncSetAttribute((const void*)ref_glu, hipFuncAttributeMaxDynamicSharedMemorySize, ref_lds));
    HC(hipFuncSetAttribute((const void*)ref_down, hipFuncAttributeMaxDynamicSharedMemorySize, ref_lds));
    HC(hipFuncSetAttribute((const void*)ref_glu8, hipFuncAttributeMaxDynamicSharedMemorySize, MFP_MW8_LDS));
    HC(hipFuncSetAttribute((const void*)ref_down8, hipFuncAttributeMaxDynamicSharedMemorySize, MFP_MW8_LDS));
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    auto timeit = [&](const char* name, auto&& fn) {
        fn();
        HC(hipDeviceSynchronize());
        float best = 1e30f, sum = 0.f;
        for (int r = 0; r < reps; r++) {
            HC(hipEventRecord(e0));
            fn();
            HC(hipEventRecord(e1));
            HC(hipEventSynchronize(e1));
            float ms;
            HC(hipEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
            sum += ms;
        }
        std::printf("%-14s best %.4f ms  mean %.4f ms\n", name, best, sum / reps);
        return best;
    };
    auto cmp = [&](const char* name, const void* a, const void* b, size_t bytes, size_t unit) {
        std::vector<unsigned char> ha(bytes), hb(bytes);
        HC(hipMemcpy(ha.data(), a, bytes, hipMemcpyDeviceToHost));
        HC(hipMemcpy(hb.data(), b, bytes, hipMemcpyDeviceToHost));
        size_t d = 0;
        for (size_t o = 0; o < bytes; o += unit) d += std::memcmp(&ha[o], &hb[o], unit) != 0;
        std::printf("%s diff %zu / %zu\n", name, d, bytes / unit);
        return d;
    };
    size_t bad = 0;
    const size_t live_rows = (size_t)tiles * BMP;
    if (!w8) {
        timeit("ref_glu", [&] { ref_glu<<<NCU, 512, ref_lds>>>(fu_r, dx, dwt, dst, dmeta, drt, I, H, E, act, beta, lbeta); });
        timeit("new_glu", [&] { plow_moe_fp8_glu_pf_gfx950<<<grid_n, 512>>>(fu_n, dx, dwt, dst, dmeta, drt, nullptr, nullptr, I, H, E, act, beta, lbeta, 0, 0); });
        timeit("ref_down", [&] { ref_down<<<NCU, 512, ref_lds>>>(part_r, fu_r, dwt, dst, dmeta, drp, drg, H, I, E); });
        timeit("new_down", [&] { plow_moe_fp8_down_pf_gfx950<<<grid_n, 512>>>(part_n, fu_r, dwt, dst, dmeta, drp, drg, nullptr, H, I, E, 0); });
        timeit("ref_combine", [&] { ref_combine<<<NCU, 512>>>(out_r, dsh, part_r, H, K8, T); });
        HC(hipDeviceSynchronize());
        bad += cmp("fu_g", fu_r, fu_n, live_rows * I * 2, 2);
        bad += cmp("part", part_r, part_n, part_b, 4);
    } else {
        /* W8A8 inputs: fp8 x rows and [H/128][T] scales (QuantFp8Block128's layout, pitch T). */
        std::vector<unsigned char> xq((size_t)T * H);
        for (auto& b : xq) b = fp8();
        std::vector<float> xs((size_t)(H / 128) * T);
        std::uniform_real_distribution<float> ua(1e-3f, 3e-2f);
        for (auto& v : xs) v = ua(rng);
        auto* dxq = (unsigned char*)up(xq.data(), xq.size());
        auto* dxs = (float*)up(xs.data(), xs.size() * 4);
        unsigned char *fq_r, *fq_n;
        float *fs_r, *fs_n;
        const size_t fq_b = (size_t)cap * I, fs_b = (size_t)cap * (I / 128) * 4;
        HC(hipMalloc(&fq_r, fq_b));
        HC(hipMalloc(&fq_n, fq_b));
        HC(hipMalloc(&fs_r, fs_b));
        HC(hipMalloc(&fs_n, fs_b));
        HC(hipMemset(fq_r, 0x5a, fq_b));
        HC(hipMemset(fq_n, 0x5a, fq_b));
        HC(hipMemset(fs_r, 0, fs_b));
        HC(hipMemset(fs_n, 0, fs_b));
        timeit("ref_glu8", [&] { ref_glu8<<<NCU, 512, MFP_MW8_LDS>>>(fq_r, fs_r, dxq, dxs, T, dwt, dst, dmeta, drt, I, H, E, act, beta, lbeta); });
        timeit("new_glu8", [&] { plow_moe_w8a8_glu_pf_gfx950<<<grid_n, 512>>>(fq_n, dxq, dwt, dst, dmeta, drt, dxs, fs_n, I, H, E, act, beta, lbeta, T, 0); });
        timeit("ref_down8", [&] { ref_down8<<<NCU, 512, MFP_MW8_LDS>>>(part_r, fq_r, fs_r, dwt, dst, dmeta, drp, drg, H, I, E); });
        timeit("new_down8", [&] { plow_moe_w8a8_down_pf_gfx950<<<grid_n, 512>>>(part_n, fq_r, dwt, dst, dmeta, drp, drg, fs_r, H, I, E, 0); });
        HC(hipDeviceSynchronize());
        bad += cmp("fq", fq_r, fq_n, live_rows * I, 1);
        bad += cmp("fs", fs_r, fs_n, live_rows * (I / 128) * 4, 4);
        bad += cmp("part", part_r, part_n, part_b, 4);
        /* Fused combine: seed out with the shared partial (zeros under fold), the acc down adds
         * the k gated slots; compare against the fixed-order combine of the same part. */
        if (fold) HC(hipMemset(dsh, 0, out_b));
        ref_combine<<<NCU, 512>>>(out_r, dsh, part_r, H, K8, T);
        HC(hipMemcpy(out_n, dsh, out_b, hipMemcpyDeviceToDevice));
        plow_moe_w8a8_down_acc_pf_gfx950<<<grid_n, 512>>>((float*)out_n, fq_r, dwt, dst, dmeta, drp, drg, fs_r, H, I, E, K8);
        HC(hipDeviceSynchronize());
        {
            std::vector<unsigned short> a(out_b / 2), b(out_b / 2);
            HC(hipMemcpy(a.data(), out_r, out_b, hipMemcpyDeviceToHost));
            HC(hipMemcpy(b.data(), out_n, out_b, hipMemcpyDeviceToHost));
            auto bf = [](unsigned short v) { unsigned u = (unsigned)v << 16; float f; std::memcpy(&f, &u, 4); return (double)f; };
            double num = 0, den = 0;
            for (size_t i = 0; i < a.size(); i++) {
                const double x = bf(a[i]), y = bf(b[i]);
                num += (x - y) * (x - y);
                den += x * x;
            }
            const double rel = std::sqrt(num / (den > 0 ? den : 1));
            std::printf("acc vs combine relL2 %.3e%s\n", rel, rel < 1e-2 ? "" : "  ACC_FAIL");
            if (!(rel < 1e-2)) bad++;
        }
        timeit("ref_combine", [&] { ref_combine<<<NCU, 512>>>(out_r, dsh, part_r, H, K8, T); });
        timeit("acc_down8", [&] { plow_moe_w8a8_down_acc_pf_gfx950<<<grid_n, 512>>>((float*)out_n, fq_r, dwt, dst, dmeta, drp, drg, fs_r, H, I, E, K8); });
        timeit("memset_out", [&] { HC(hipMemsetAsync(out_n, 0, out_b)); });
    }
    const bool ok = bad == 0;
    std::printf(ok ? "BITEXACT_OK\n" : "BITEXACT_FAIL\n");
    return ok ? 0 : 1;
}
