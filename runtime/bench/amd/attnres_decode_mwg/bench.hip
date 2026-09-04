/* Decode AttnRes: single-workgroup arm (control) vs the banded multi-workgroup arm
 * (`d_attn_res_mwg`, PLOW_ATTNRES_DECODE_MWG=1) on the K3 decode shape: T=1, HID=7168,
 * fused output norm, nb = 0..8, 186 distinct (score_w, gamma) sites cycled so the weight
 * stream is as cold as in the model while the rows stay L2-hot (they were written by the
 * previous packet of the same token).
 *
 * The timing loop is inside the kernel, as in runtime/tests/k3_attn_res_bench_gfx950.hip: the
 * op is a packet of a persistent megakernel, not a launch. Consecutive iterations of the
 * candidate are consecutive rounds of the same site's rendezvous scratch, which is exactly the
 * token-to-token reuse the arm has to survive.
 *
 * Exactness: the candidate follows the C3 f32-mix contract (mix in f32 through the output norm)
 * and is checked against a double-precision CPU port of it — every output element within 1 bf16
 * ulp of the reference and relL2 at the bf16 rounding floor — and for byte-stable repeats. The
 * control (bf16-seam arm) is reported against the same reference for scale; it is a different
 * contract and is expected to differ at the seam. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amd_common.h"
#include "op_k3.h"

#define HC(x)                                                                                 \
    do {                                                                                      \
        hipError_t e_ = (x);                                                                  \
        if (e_ != hipSuccess) {                                                               \
            std::fprintf(stderr, "HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

enum { LDS_BYTES = 147464, HID = 7168, NSITE = 186, NBCAP = 8 };

struct Site {
    const float* score;
    const bf16* gamma;
    bf16* prefix; /* hot: every site the same rows (L2); cold: one row set per site (MALL) */
    bf16* ring;
};

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void control(
    bf16* out, const Site* sites, unsigned nb, float eps, unsigned iters) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    for (unsigned i = 0; i < iters; ++i) {
        const Site s = sites[i % NSITE];
        d_attn_res(out, s.prefix, s.ring, s.score, 1u, HID, nb, NBCAP, eps, blockIdx.x,
                   gridDim.x, (float*)raw, nullptr, 0u, s.gamma);
        __syncthreads();
    }
}

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void candidate(
    bf16* out, const Site* sites, unsigned nb, float eps, uint64_t* scratch, unsigned iters) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    for (unsigned i = 0; i < iters; ++i) {
        const Site s = sites[i % NSITE];
        d_attn_res_mwg(out, s.prefix, s.ring, s.score, HID, nb, NBCAP, eps, eps, blockIdx.x,
                       gridDim.x, (float*)raw, nullptr, 0u, s.gamma, nullptr, nullptr, nullptr,
                       scratch);
    }
}

static float bf2f_h(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static bf16 f2bf_h(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
    return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint32_t rng_state = 0x9e3779b9u;
static float frand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) * (1.0f / 16777216.0f);
}
static float nrand() { /* rough gaussian */
    float s = 0.0f;
    for (int i = 0; i < 4; ++i) s += frand() - 0.5f;
    return s * 1.7320508f;
}

/* Double-precision port of the f32-mix contract. */
static void reference(const std::vector<bf16>& prefix, const std::vector<bf16>& ring,
                      const std::vector<float>& score, const std::vector<bf16>& gamma, unsigned nb,
                      float eps, std::vector<double>& y) {
    std::vector<double> sc(nb + 1);
    for (unsigned r = 0; r <= nb; ++r) {
        const bf16* row = (r < nb) ? ring.data() + (size_t)r * HID : prefix.data();
        double ss = 0.0, sw = 0.0;
        for (unsigned d = 0; d < HID; ++d) {
            const double x = bf2f_h(row[d]);
            ss += x * x;
            sw += x * (double)score[d];
        }
        sc[r] = sw / std::sqrt(ss / HID + (double)eps);
    }
    double m = -1e300;
    for (unsigned r = 0; r <= nb; ++r) m = std::max(m, sc[r]);
    double z = 0.0;
    for (unsigned r = 0; r <= nb; ++r) {
        sc[r] = std::exp(sc[r] - m);
        z += sc[r];
    }
    std::vector<double> mixed(HID, 0.0);
    double ss = 0.0;
    for (unsigned d = 0; d < HID; ++d) {
        double a = 0.0;
        for (unsigned r = 0; r <= nb; ++r) {
            const bf16* row = (r < nb) ? ring.data() + (size_t)r * HID : prefix.data();
            a += (sc[r] / z) * (double)bf2f_h(row[d]);
        }
        mixed[d] = a;
        ss += a * a;
    }
    const double inv = 1.0 / std::sqrt(ss / HID + (double)eps);
    y.resize(HID);
    for (unsigned d = 0; d < HID; ++d) y[d] = mixed[d] * inv * (double)bf2f_h(gamma[d]);
}

struct Err {
    double rel_l2;
    int max_ulp;
    unsigned nan;
};
/* bf16 ulps between `got` and the reference rounded to bf16 (0 = the correctly rounded value). */
static Err compare(const std::vector<bf16>& got, const std::vector<double>& y) {
    Err e{0.0, 0, 0};
    double num = 0.0, den = 0.0;
    for (unsigned d = 0; d < HID; ++d) {
        const float g = bf2f_h(got[d]);
        if (std::isnan(g)) {
            e.nan++;
            continue;
        }
        const double diff = (double)g - y[d];
        num += diff * diff;
        den += y[d] * y[d];
        const bf16 want = f2bf_h((float)y[d]);
        auto key = [](bf16 b) { return (b & 0x8000u) ? -(int)(b & 0x7fffu) : (int)(b & 0x7fffu); };
        const int ulp = std::abs(key(got[d]) - key(want));
        e.max_ulp = std::max(e.max_ulp, ulp);
    }
    e.rel_l2 = std::sqrt(num / std::max(den, 1e-300));
    return e;
}

static int schedule_weight(unsigned nb) { /* K3: nb = ceil(layer / 12), 93 layers, 2 sites */
    int n = 0;
    for (unsigned l = 0; l < 93; ++l)
        if ((l + 11) / 12 == nb) n += 2;
    return n;
}

int main(int argc, char** argv) {
    const unsigned iters = argc > 1 ? (unsigned)std::atoi(argv[1]) : 2000u;
    const float eps = 1e-6f;
    std::vector<unsigned> bands = {2, 4, 7, 8, 14, 16};

    std::vector<bf16> h_prefix((size_t)NSITE * HID), h_ring((size_t)NSITE * NBCAP * HID),
        h_gamma((size_t)NSITE * HID);
    std::vector<float> h_score((size_t)NSITE * HID);
    for (auto& v : h_prefix) v = f2bf_h(nrand() * 2.0f);
    for (auto& v : h_ring) v = f2bf_h(nrand() * 2.0f);
    for (auto& v : h_score) v = nrand() * 0.02f;
    for (auto& v : h_gamma) v = f2bf_h(1.0f + nrand() * 0.1f);

    bf16 *d_out, *d_prefix, *d_ring, *d_gamma;
    float* d_score;
    Site *d_sites_hot, *d_sites_cold;
    uint64_t* d_scratch;
    HC(hipMalloc(&d_out, HID * 2));
    HC(hipMalloc(&d_prefix, h_prefix.size() * 2));
    HC(hipMalloc(&d_ring, h_ring.size() * 2));
    HC(hipMalloc(&d_gamma, h_gamma.size() * 2));
    HC(hipMalloc(&d_score, h_score.size() * 4));
    HC(hipMalloc(&d_sites_hot, NSITE * sizeof(Site)));
    HC(hipMalloc(&d_sites_cold, NSITE * sizeof(Site)));
    HC(hipMalloc(&d_scratch, PLOW_ATTNRES_MWG_WORDS * 8));
    HC(hipMemcpy(d_prefix, h_prefix.data(), h_prefix.size() * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_ring, h_ring.data(), h_ring.size() * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_gamma, h_gamma.data(), h_gamma.size() * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(d_score, h_score.data(), h_score.size() * 4, hipMemcpyHostToDevice));
    std::vector<Site> h_sites(NSITE);
    for (unsigned s = 0; s < NSITE; ++s)
        h_sites[s] = Site{d_score + (size_t)s * HID, d_gamma + (size_t)s * HID, d_prefix, d_ring};
    HC(hipMemcpy(d_sites_hot, h_sites.data(), NSITE * sizeof(Site), hipMemcpyHostToDevice));
    for (unsigned s = 0; s < NSITE; ++s) {
        h_sites[s].prefix = d_prefix + (size_t)s * HID;
        h_sites[s].ring = d_ring + (size_t)s * NBCAP * HID;
    }
    HC(hipMemcpy(d_sites_cold, h_sites.data(), NSITE * sizeof(Site), hipMemcpyHostToDevice));
    const Site* d_sites = d_sites_hot;
    HC(hipMemset(d_scratch, 0, PLOW_ATTNRES_MWG_WORDS * 8));

    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    auto time_us = [&](auto launch) {
        launch(20u);
        HC(hipDeviceSynchronize());
        HC(hipEventRecord(e0));
        launch(iters);
        HC(hipEventRecord(e1));
        HC(hipEventSynchronize(e1));
        float ms = 0.0f;
        HC(hipEventElapsedTime(&ms, e0, e1));
        return (double)ms * 1000.0 / iters;
    };
    auto run_control = [&](unsigned nb, unsigned it) {
        hipLaunchKernelGGL(control, dim3(1), dim3(PLOW_THREADS), 0, 0, d_out, d_sites, nb, eps,
                           it);
    };
    /* One site has one band count for the life of a program; the round counter is only a
     * multiple of `nblk` under that invariant, so a band change here starts a fresh scratch. */
    unsigned last_nblk = 0;
    auto run_candidate = [&](unsigned nb, unsigned nblk, unsigned it) {
        if (nblk != last_nblk) {
            HC(hipMemset(d_scratch, 0, PLOW_ATTNRES_MWG_WORDS * 8));
            HC(hipDeviceSynchronize());
            last_nblk = nblk;
        }
        hipLaunchKernelGGL(candidate, dim3(nblk), dim3(PLOW_THREADS), 0, 0, d_out, d_sites, nb,
                           eps, d_scratch, it);
    };

    int bad = 0;
    std::printf("K3 decode AttnRes, T=1 HID=%d fused norm, %u iters, us/site (rows L2-hot, "
                "%d cold sites)\n",
                HID, iters, NSITE);
    std::printf("%4s %-4s %8s", "nb", "rows", "1wg");
    for (unsigned b : bands) std::printf(" %8uwg", b);
    std::printf("\n");
    double mean_ctl[2] = {0.0, 0.0};
    std::vector<double> mean_cand[2] = {std::vector<double>(bands.size(), 0.0),
                                        std::vector<double>(bands.size(), 0.0)};
    int wsum = 0;
    for (unsigned nb = 0; nb <= 8; ++nb) {
        const int w = schedule_weight(nb);
        wsum += w;
        /* Exactness at this nb, site 0 (the first iteration's weights), before timing. */
        std::vector<double> y;
        std::vector<bf16> g0(h_gamma.begin(), h_gamma.begin() + HID);
        std::vector<float> s0(h_score.begin(), h_score.begin() + HID);
        std::vector<bf16> p0(h_prefix.begin(), h_prefix.begin() + HID);
        std::vector<bf16> r0(h_ring.begin(), h_ring.begin() + (size_t)NBCAP * HID);
        reference(p0, r0, s0, g0, nb, eps, y);
        std::vector<bf16> got(HID), got2(HID);
        run_control(nb, 1u);
        HC(hipDeviceSynchronize());
        HC(hipMemcpy(got.data(), d_out, HID * 2, hipMemcpyDeviceToHost));
        const Err ec = compare(got, y);
        std::printf("  nb=%u control  vs f32-mix ref: relL2 %.3e max %d ulp nan %u (bf16-seam "
                    "contract, informational)\n",
                    nb, ec.rel_l2, ec.max_ulp, ec.nan);
        for (unsigned b : bands) {
            run_candidate(nb, b, 1u);
            HC(hipDeviceSynchronize());
            HC(hipMemcpy(got.data(), d_out, HID * 2, hipMemcpyDeviceToHost));
            run_candidate(nb, b, 3u); /* three more rounds, then re-read: byte-stable */
            HC(hipDeviceSynchronize());
            HC(hipMemcpy(got2.data(), d_out, HID * 2, hipMemcpyDeviceToHost));
            /* the last of those three rounds is iteration 2, site 2's weights */
            std::vector<double> y3;
            std::vector<bf16> g3(h_gamma.begin() + 2 * HID, h_gamma.begin() + 3 * HID);
            std::vector<float> s3(h_score.begin() + 2 * HID, h_score.begin() + 3 * HID);
            reference(p0, r0, s3, g3, nb, eps, y3);
            const Err e = compare(got, y), e3 = compare(got2, y3);
            const bool ok = e.nan == 0 && e3.nan == 0 && e.max_ulp <= 1 && e3.max_ulp <= 1 &&
                            e.rel_l2 < 3e-3 && e3.rel_l2 < 3e-3;
            if (!ok) bad++;
            std::printf("  nb=%u %2uwg  vs f32-mix ref: relL2 %.3e max %d ulp nan %u | round 4: "
                        "relL2 %.3e max %d ulp %s\n",
                        nb, b, e.rel_l2, e.max_ulp, e.nan, e3.rel_l2, e3.max_ulp,
                        ok ? "OK" : "FAIL");
            /* determinism: same round count twice must give the same bytes */
            run_candidate(nb, b, 1u);
            HC(hipDeviceSynchronize());
            HC(hipMemcpy(got.data(), d_out, HID * 2, hipMemcpyDeviceToHost));
            run_candidate(nb, b, 1u);
            HC(hipDeviceSynchronize());
            HC(hipMemcpy(got2.data(), d_out, HID * 2, hipMemcpyDeviceToHost));
            if (std::memcmp(got.data(), got2.data(), HID * 2) != 0) {
                std::printf("  nb=%u %2uwg  NOT byte-stable across repeats\n", nb, b);
                bad++;
            }
        }
        for (int cold = 0; cold < 2; ++cold) {
            d_sites = cold ? d_sites_cold : d_sites_hot;
            const double tc = time_us([&](unsigned it) { run_control(nb, it); });
            mean_ctl[cold] += tc * w;
            std::printf("%4u %-4s %8.2f", nb, cold ? "cold" : "hot", tc);
            for (size_t k = 0; k < bands.size(); ++k) {
                const double t = time_us([&](unsigned it) { run_candidate(nb, bands[k], it); });
                mean_cand[cold][k] += t * w;
                std::printf(" %10.2f", t);
            }
            std::printf("\n");
        }
        d_sites = d_sites_hot;
    }
    for (int cold = 0; cold < 2; ++cold) {
        std::printf("%4s %-4s %8.2f", "mean", cold ? "cold" : "hot", mean_ctl[cold] / wsum);
        for (size_t k = 0; k < bands.size(); ++k) std::printf(" %10.2f", mean_cand[cold][k] / wsum);
        std::printf("   (K3 schedule-weighted; x186 = ms/token: %.3f", mean_ctl[cold] / wsum * 186 / 1000);
        for (size_t k = 0; k < bands.size(); ++k)
            std::printf(" %.3f", mean_cand[cold][k] / wsum * 186 / 1000);
        std::printf(")\n");
    }
    std::printf(bad ? "EXACTNESS: %d FAILURES\n" : "EXACTNESS: all candidates within 1 bf16 ulp of the f32-mix reference, byte-stable\n", bad);
    return bad ? 1 : 0;
}
