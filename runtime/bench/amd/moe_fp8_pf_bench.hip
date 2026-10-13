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

#if PLOW_MOE_PF_DET || PLOW_MOE_PF_ATOMIC
/* The interpreter's fused down -> combine arms (f64 fixed-point / f32 atomic into [T,H]). */
extern "C" __global__ __launch_bounds__(512, 1) void ref_down_acc(
    void* acc, const bf16* fu, const unsigned long long* wtab, const unsigned long long* stab,
    const int* meta, const unsigned* row_partidx, const float* row_gate, unsigned H, unsigned I,
    unsigned E) {
    extern __shared__ __align__(16) bf16 sm[];
    d_moe_group_down_pf((float*)acc, fu, wtab, stab, meta, row_partidx, row_gate, H, I, E,
                        PLOW_MOE_ENC_FP8BLK, blockIdx.x, gridDim.x, sm, nullptr, 0, 0,
#if PLOW_MOE_PF_ATOMIC
                        4u
#else
                        4u
#endif
    );
}
#endif

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
    const unsigned E = 288, K8 = 8, H = 4096, I = 256, BMP = 64;
    const unsigned act = PLOW_MOE_ACT_SWIGLU_CLAMP;
    const float beta = 0.0f, lbeta = 10.0f;
    std::mt19937 rng(1234);

    /* routing: skewed expert popularity, k distinct experts per token */
    std::vector<double> pop(E);
    for (unsigned e = 0; e < E; e++) pop[e] = 1.0 / std::pow(1.0 + (double)((e * 37u) % E), skew * 0.5);
    std::discrete_distribution<unsigned> pick(pop.begin(), pop.end());
    std::uniform_real_distribution<float> uf(0.05f, 1.0f);
    std::vector<unsigned> slot_e(T * K8);
    std::vector<float> slot_g(T * K8);
    for (unsigned t = 0; t < T; t++) {
        unsigned n = 0;
        while (n < K8) {
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
    const unsigned grid_n = argc > 4 ? (unsigned)atoi(argv[4]) : NCU;
    timeit("ref_glu", [&] { ref_glu<<<NCU, 512, ref_lds>>>(fu_r, dx, dwt, dst, dmeta, drt, I, H, E, act, beta, lbeta); });
    timeit("new_glu", [&] { plow_moe_fp8_glu_pf_gfx950<<<grid_n, 512>>>(fu_n, dx, dwt, dst, dmeta, drt, I, H, E, act, beta, lbeta); });
    timeit("ref_down", [&] { ref_down<<<NCU, 512, ref_lds>>>(part_r, fu_r, dwt, dst, dmeta, drp, drg, H, I, E); });
    timeit("new_down", [&] { plow_moe_fp8_down_pf_gfx950<<<grid_n, 512>>>(part_n, fu_r, dwt, dst, dmeta, drp, drg, H, I, E, 0); });
#if PLOW_MOE_PF_DET || PLOW_MOE_PF_ATOMIC
    {
        void* acc;
        const size_t acc_b = (size_t)T * H * (PLOW_MOE_PF_DET ? 8 : 4);
        HC(hipMalloc(&acc, acc_b));
        HC(hipFuncSetAttribute((const void*)ref_down_acc, hipFuncAttributeMaxDynamicSharedMemorySize, ref_lds));
        timeit("acc_memset", [&] { HC(hipMemsetAsync(acc, 0, acc_b)); });
        timeit("ref_down_acc", [&] { ref_down_acc<<<NCU, 512, ref_lds>>>(acc, fu_r, dwt, dst, dmeta, drp, drg, H, I, E); });
        HC(hipFree(acc));
    }
#endif
    timeit("ref_combine", [&] { ref_combine<<<NCU, 512>>>(out_r, dsh, part_r, H, K8, T); });
    timeit("ref_combine_n", [&] { ref_combine<<<NCU, 512>>>(out_n, dsh, part_n, H, K8, T); });
    HC(hipDeviceSynchronize());

    /* compare */
    std::vector<unsigned short> hfr(fu_b / 2), hfn(fu_b / 2);
    HC(hipMemcpy(hfr.data(), fu_r, fu_b, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hfn.data(), fu_n, fu_b, hipMemcpyDeviceToHost));
    size_t fu_diff = 0, fu_live = 0;
    for (unsigned r = 0; r < tiles * BMP; r++)
        for (unsigned c = 0; c < I; c++) {
            fu_live++;
            fu_diff += hfr[(size_t)r * I + c] != hfn[(size_t)r * I + c];
        }
    std::vector<float> hpr(part_b / 4), hpn(part_b / 4);
    HC(hipMemcpy(hpr.data(), part_r, part_b, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hpn.data(), part_n, part_b, hipMemcpyDeviceToHost));
    size_t p_diff = 0;
    double p_max = 0;
    for (size_t i = 0; i < hpr.size(); i++)
        if (std::memcmp(&hpr[i], &hpn[i], 4)) {
            p_diff++;
            p_max = std::max(p_max, (double)std::fabs(hpr[i] - hpn[i]));
        }
    std::vector<unsigned short> hor(out_b / 2), hon(out_b / 2);
    HC(hipMemcpy(hor.data(), out_r, out_b, hipMemcpyDeviceToHost));
    HC(hipMemcpy(hon.data(), out_n, out_b, hipMemcpyDeviceToHost));
    size_t o_diff = 0;
    for (size_t i = 0; i < hor.size(); i++) o_diff += hor[i] != hon[i];
    std::printf("fu_g diff %zu / %zu   part diff %zu / %zu (max %.3g)   out diff %zu / %zu\n",
                fu_diff, fu_live, p_diff, hpr.size(), p_max, o_diff, hor.size());
    const bool ok = fu_diff == 0 && p_diff == 0 && o_diff == 0;
    std::printf(ok ? "BITEXACT_OK\n" : "BITEXACT_FAIL\n");
    return ok ? 0 : 1;
}
