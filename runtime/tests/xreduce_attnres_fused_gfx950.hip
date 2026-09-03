/* Exact single-rank phase-2 gate for the XReduceTwoShot -> AttnRes fold.
 * The peer slots enter with every non-local reduce-scatter row already reduced; this rank's
 * row group is reduced by the production collective body. The reference then gathers the same
 * flat row owners in a separate launch before running the unchanged AttnRes body.
 *
 *   hipcc --offload-arch=gfx950 -O3 -w -I runtime/amd -I runtime/common \
 *     runtime/tests/xreduce_attnres_fused_gfx950.hip -o /tmp/xr_ar_exact
 */
#define PLOW_K3 1
#define PLOW_XR_ATTNRES 1
#define PLOW_XR_NOSIG 1
#define PLOW_XR2_SKIP_RS 1
#define PLOW_XR2_SKIP_AG 1

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../amd/xreduce_attnres_fused.hip"

#define HC(x) do { hipError_t e_ = (x); if (e_) { \
    std::fprintf(stderr, "hip %s @%d\n", hipGetErrorString(e_), __LINE__); std::exit(1); \
} } while (0)

enum { LDS_BYTES = 147464 };

extern "C" __global__ void gather_reference(bf16* reduced, bf16* prefix,
                                              const void* const* peers,
                                              const bf16* residual, uint32_t nranks,
                                              uint32_t T, uint32_t H) {
    const uint32_t m = blockIdx.x;
    const uint32_t owner = m / (T / nranks);
    const bf16* src = (const bf16*)peers[owner];
    for (uint32_t d = threadIdx.x; d < H; d += blockDim.x) {
        const size_t e = (size_t)m * H + d;
        const bf16 v = src[e];
        reduced[e] = v;
        prefix[e] = residual ? f2bf(bf2f(residual[e]) + bf2f(v)) : v;
    }
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void attnres_reference(bf16* out, const bf16* prefix, const bf16* ring,
                       const float* score, const bf16* gamma, uint32_t T,
                       uint32_t H, uint32_t nb, float eps) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_attn_res(out, prefix, ring, score, T, H, nb, nb, eps,
               blockIdx.x, gridDim.x, (float*)raw, nullptr, 0u, gamma);
}

static bf16 f2bf_host(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

int main() {
    constexpr uint32_t NR = 8, RANK = 3, T = 256, H = 7168, NB = 2;
    constexpr float EPS = 1e-5f;
    const size_t n = (size_t)T * H;
    std::vector<std::vector<bf16>> hp(NR, std::vector<bf16>(n));
    for (uint32_t r = 0; r < NR; ++r) {
        for (size_t e = 0; e < n; ++e) {
            const uint32_t owner = (uint32_t)(e / H) / (T / NR);
            if (owner == RANK)
                hp[r][e] = r == RANK
                    ? f2bf_host(((int)(e % 17) - 8) * 0.0078125f + r * 0.03125f)
                    : f2bf_host(0.0f);
            else if (owner == r)
                hp[r][e] = f2bf_host(((int)(e % 23) - 11) * 0.015625f);
        }
    }
    std::vector<bf16> hres(n), hring((size_t)T * NB * H), hgamma(H);
    std::vector<float> hscore(H);
    for (size_t e = 0; e < n; ++e) hres[e] = f2bf_host(((int)(e % 13) - 6) * 0.03125f);
    for (size_t e = 0; e < hring.size(); ++e)
        hring[e] = f2bf_host(((int)(e % 19) - 9) * 0.0078125f);
    for (uint32_t d = 0; d < H; ++d) {
        hscore[d] = ((int)(d % 11) - 5) * 0.015625f;
        hgamma[d] = f2bf_host(0.75f + (d % 7) * 0.03125f);
    }

    std::vector<void*> dp(NR);
    for (uint32_t r = 0; r < NR; ++r) HC(hipMalloc(&dp[r], n * sizeof(bf16)));
    void** dpeers;
    PlowXReduceAttnResArgs* dargs;
    bf16 *dres, *dring, *dgamma, *fred, *fpfx, *fout, *rred, *rpfx, *rout;
    float* dscore;
    HC(hipMalloc(&dpeers, NR * sizeof(void*)));
    HC(hipMalloc(&dargs, sizeof(PlowXReduceAttnResArgs)));
    HC(hipMalloc(&dres, n * 2)); HC(hipMalloc(&dring, hring.size() * 2));
    HC(hipMalloc(&dgamma, H * 2)); HC(hipMalloc(&dscore, H * 4));
    for (bf16** p : {&fred, &fpfx, &fout, &rred, &rpfx, &rout}) HC(hipMalloc(p, n * 2));
    HC(hipMemcpy(dpeers, dp.data(), NR * sizeof(void*), hipMemcpyHostToDevice));
    HC(hipMemcpy(dres, hres.data(), n * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dring, hring.data(), hring.size() * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dgamma, hgamma.data(), H * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dscore, hscore.data(), H * 4, hipMemcpyHostToDevice));

    int bad = 0;
    std::vector<bf16> a(n), b(n);
    for (int with_residual = 0; with_residual <= 1; ++with_residual) {
        for (uint32_t r = 0; r < NR; ++r)
            HC(hipMemcpy(dp[r], hp[r].data(), n * 2, hipMemcpyHostToDevice));
        PlowXReduceAttnResArgs args = {};
        args.reduced = fred; args.prefix = fpfx; args.out = fout;
        args.residual = with_residual ? dres : nullptr;
        args.ring = dring; args.score = dscore; args.gamma = dgamma;
        args.peer_scratch = dpeers; args.xctr = (uint32_t*)dp[RANK];
        args.n = (uint32_t)n; args.rank = RANK; args.nranks = NR;
        args.row_w = H; args.nb = NB; args.nbcap = NB; args.eps = EPS;
        HC(hipMemcpy(dargs, &args, sizeof(args), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(plow_xreduce_attnres_gfx950, dim3(T), dim3(PLOW_THREADS), 0, 0,
                           dargs);
        hipLaunchKernelGGL(gather_reference, dim3(T), dim3(PLOW_THREADS), 0, 0,
                           rred, rpfx, dpeers, with_residual ? dres : nullptr, NR, T, H);
        hipLaunchKernelGGL(attnres_reference, dim3(T), dim3(PLOW_THREADS), 0, 0,
                           rout, rpfx, dring, dscore, dgamma, T, H, NB, EPS);
        HC(hipDeviceSynchronize());
        int mode_bad = 0;
        unsigned which = 0;
        for (auto pair : {std::pair{fred, rred}, std::pair{fpfx, rpfx}, std::pair{fout, rout}}) {
            HC(hipMemcpy(a.data(), pair.first, n * 2, hipMemcpyDeviceToHost));
            HC(hipMemcpy(b.data(), pair.second, n * 2, hipMemcpyDeviceToHost));
            if (std::memcmp(a.data(), b.data(), n * 2) != 0) {
                size_t first = 0;
                while (first < n && a[first] == b[first]) ++first;
                std::printf("  tensor %u first=%zu fused=%04x ref=%04x\n",
                            which, first, a[first], b[first]);
                ++mode_bad;
            }
            ++which;
        }
        bad += mode_bad;
        std::printf("phase2%s: %s\n", with_residual ? "+residual" : " direct",
                    mode_bad ? "MISMATCH" : "BIT-IDENTICAL");
    }
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
