/* v4_roofline_gfx942.hip — where each DeepSeek-V4 kernel sits against the
 * MEASURED ceilings of this part, at the shapes the model actually runs.
 *
 * DENOMINATORS, and where they come from (docs/bringup/target.md is emphatic
 * that a roofline % whose denominator came from somewhere else is a measurement
 * of nothing):
 *
 *   $BW_BOUND      4164 GB/s. MEASURED on this MI325X
 *                  (runtime/tests/decode_bw_probe.hip, median of three clean
 *                  leased runs) and landed in the registry, where
 *                  `measured_bandwidth_governs_a_reported_bound_on_mi325`
 *                  asserts `bandwidth_for_bound()` returns it. It is 69% of the
 *                  6000 GB/s datasheet peak, so quoting the peak would
 *                  understate efficiency by ~44%.
 *
 *   $VALU_CEIL     304 CU x 64 lanes x 2 flop x 2.1 GHz = 81.7 TF/s fp32 FMA.
 *                  THIS is the compute ceiling that applies here, NOT the
 *                  ~1.3 PF/s bf16 matrix-core figure: not one of these kernels
 *                  issues an MFMA. Every dot product below is scalar f32 FMA on
 *                  bf16 operands converted in the VALU, so the matrix cores are
 *                  idle and quoting them as the ceiling would make a kernel that
 *                  is at 90% of what it can reach look like it is at 6%.
 *                  Analytic, from the part's own clock and CU count — it is a
 *                  bound, not a measurement, and it is stated as such.
 *
 *   ridge point    81.7e12 / 4164e9 = 19.6 flop/byte. Below it a kernel is
 *                  bandwidth bound; above it, VALU bound.
 *
 * BYTES are COMPULSORY traffic — each distinct byte counted once, which is the
 * denominator that says "could this be faster", not "how many loads did I
 * issue". Where a kernel re-reads a value the model only needs once, that shows
 * up as achieved-below-ceiling, which is exactly the finding.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_roofline_gfx942.hip -o /tmp/v4rl && /tmp/v4rl
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amd_common.h"
#include "op_deepseek_v4.h"

#define HC_CHECK(x)                                                                               \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_) {                                                                                 \
            printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__);                              \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

static const double BW_BOUND = 4164.0e9;  /* measured, registry */
static const double VALU_CEIL = 81.7e12;  /* 304 CU x 64 x 2 x 2.1 GHz */

enum { LDS_BYTES = 32768, NCU = 304 };

__global__ __launch_bounds__(PLOW_THREADS) void k_reduce(bf16* out, float* mix, const bf16* x,
                                                         const float* fn, const float* sc,
                                                         const float* ba, unsigned T, unsigned D,
                                                         unsigned HC, unsigned it) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_hc_reduce(out, mix, x, fn, sc, ba, T, D, HC, it, 1e-6f, 1e-6f, blockIdx.x, gridDim.x,
                (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_expand(bf16* out, const bf16* br, const bf16* re,
                                                         const float* mix, unsigned T, unsigned D,
                                                         unsigned HC) {
    d_hc_expand(out, br, re, mix, T, D, HC, blockIdx.x, gridDim.x);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_hcdot(float* part, const bf16* x,
                                                        const float* fn, unsigned T, unsigned D,
                                                        unsigned HC) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_v4_hc_dot(part, x, fn, T, D, HC, blockIdx.x, gridDim.x, (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_hcmix(bf16* out, float* mix, const bf16* x,
                                                        const float* part, const float* sc,
                                                        const float* ba, unsigned T, unsigned D,
                                                        unsigned HC, unsigned it) {
    __shared__ __align__(16) char raw[LDS_BYTES];
    d_v4_hc_mix(out, mix, x, part, sc, ba, T, D, HC, it, 1e-6f, 1e-6f, blockIdx.x, gridDim.x,
                (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_attn(bf16* out, const bf16* q, const bf16* kv,
                                                       const int* idx, const float* sink,
                                                       unsigned T, unsigned H, unsigned D,
                                                       unsigned TOPK, float s) {
    __shared__ __align__(16) char raw[40960];
    d_v4_sparse_attn(out, q, kv, idx, sink, T, H, D, TOPK, s, blockIdx.x, gridDim.x, (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_score(float* sc, const bf16* q, const bf16* ckv,
                                                        const bf16* w, unsigned T, unsigned H,
                                                        unsigned HD, unsigned NC) {
    d_v4_index_score(sc, q, ckv, w, T, H, HD, NC, blockIdx.x, gridDim.x);
}

static bf16 f2bf_host(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* Median of `reps` timed runs after warmup — a mean hides a single outlier and
 * these kernels are short enough that one preemption doubles it. */
template <typename F>
static double time_us(F f, int reps = 21) {
    for (int i = 0; i < 3; i++) f();
    HC_CHECK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; i++) {
        hipEvent_t a, b;
        hipEventCreate(&a);
        hipEventCreate(&b);
        hipEventRecord(a);
        f();
        hipEventRecord(b);
        HC_CHECK(hipEventSynchronize(b));
        float ms = 0;
        hipEventElapsedTime(&ms, a, b);
        t.push_back(ms * 1000.0);
        hipEventDestroy(a);
        hipEventDestroy(b);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static void row(const char* name, double us, double bytes, double flops) {
    const double bw_t = bytes / BW_BOUND * 1e6;    /* us if bandwidth bound   */
    const double cp_t = flops / VALU_CEIL * 1e6;   /* us if VALU bound        */
    const double roof = fmax(bw_t, cp_t);
    const char* side = (cp_t > bw_t) ? "VALU" : "BW";
    printf("  %-22s %8.1f us   roof %7.2f us (%-4s)  %6.1f%% of roof   %7.1f GB/s  %6.2f TF/s  AI %6.1f\n",
           name, us, roof, side, 100.0 * roof / us, bytes / (us * 1e-6) / 1e9,
           flops / (us * 1e-6) / 1e12, flops / bytes);
}

__global__ void k_null(void) {}

int main(void) {
    srand48(3);
    printf("MI325X  BW_BOUND 4164 GB/s (measured)   VALU_CEIL 81.7 TF/s (analytic)   ridge 19.6 flop/B\n");

    /* THE FLOOR. Every row below is one or two KERNEL LAUNCHES, and a launch is
     * not free. In the shipped runtime these ops are PACKETS inside a persistent
     * interpreter — dispatched by a counter DAG, with no per-op launch — so any
     * measured time at or near this floor says nothing about the kernel and
     * everything about the harness. Printed first so no row below can be read
     * without it. */
    const double floor1 = time_us([&] { k_null<<<NCU, PLOW_THREADS>>>(); });
    const double floor2 = time_us([&] {
        k_null<<<NCU, PLOW_THREADS>>>();
        k_null<<<NCU, PLOW_THREADS>>>();
    });
    printf("launch floor: %.2f us for one kernel, %.2f us for two\n\n", floor1, floor2);

    /* ---------------- decode, T=1: the latency-critical shape ------------- */
    {
        const unsigned T = 1, D = 4096, HC = 4, MIX = (2 + HC) * HC, N = HC * D;
        std::vector<bf16> x((size_t)T * N);
        std::vector<float> fn((size_t)MIX * N), base(MIX), sc(3);
        for (auto& v : x) v = f2bf_host(drand48());
        for (auto& v : fn) v = (float)drand48() * 0.01f;
        bf16 *d_x, *d_o, *d_e;
        float *d_fn, *d_sc, *d_ba, *d_mix;
        HC_CHECK(hipMalloc(&d_x, x.size() * 2));
        HC_CHECK(hipMalloc(&d_o, (size_t)T * D * 2));
        HC_CHECK(hipMalloc(&d_e, (size_t)T * N * 2));
        HC_CHECK(hipMalloc(&d_fn, fn.size() * 4));
        HC_CHECK(hipMalloc(&d_sc, 12));
        HC_CHECK(hipMalloc(&d_ba, base.size() * 4));
        HC_CHECK(hipMalloc(&d_mix, (size_t)T * (HC + HC * HC) * 4));
        HC_CHECK(hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice));
        HC_CHECK(hipMemcpy(d_fn, fn.data(), fn.size() * 4, hipMemcpyHostToDevice));

        printf("hyper-connections, decode (T=1, D=4096, hc=4, 86 sites/token)\n");
        const double us1 = time_us([&] {
            k_reduce<<<1, PLOW_THREADS>>>(d_o, d_mix, d_x, d_fn, d_sc, d_ba, T, D, HC, 20);
        });
        /* hc_fn dominates and is read once per token; x once; out/mix are tiny. */
        const double b1 = (double)MIX * N * 4 + (double)N * 2 + (double)D * 2;
        const double f1 = 2.0 * MIX * N + 2.0 * N + 2.0 * HC * D;
        row("hc_reduce", us1, b1, f1);

        const double us2 =
            time_us([&] { k_expand<<<NCU, PLOW_THREADS>>>(d_e, d_o, d_x, d_mix, T, D, HC); });
        const double b2 = (double)D * 2 + 2.0 * N * 2;
        const double f2 = (double)HC * D * (1.0 + 2.0 * HC);
        row("hc_expand", us2, b2, f2);

        /* The split pair: the same reduce as two grid-parallel packets. */
        float* d_part;
        HC_CHECK(hipMalloc(&d_part, (size_t)T * (1 + MIX) * 4));
        bf16* d_o2;
        HC_CHECK(hipMalloc(&d_o2, (size_t)T * D * 2));
        const unsigned GB = NCU;
        const double us3 = time_us([&] {
            HC_CHECK(hipMemsetAsync(d_part, 0, (size_t)T * (1 + MIX) * 4));
            k_hcdot<<<GB, PLOW_THREADS>>>(d_part, d_x, d_fn, T, D, HC);
            k_hcmix<<<GB, PLOW_THREADS>>>(d_o2, d_mix, d_x, d_part, d_sc, d_ba, T, D, HC, 20);
        });
        row("hc_dot+hc_mix (split)", us3, b1, f1);
        /* Attribute the pair: guessing which half is slow has already cost two
         * wrong optimizations here. */
        const double usA = time_us([&] { k_hcdot<<<GB, PLOW_THREADS>>>(d_part, d_x, d_fn, T, D, HC); });
        const double usB = time_us(
            [&] { k_hcmix<<<GB, PLOW_THREADS>>>(d_o2, d_mix, d_x, d_part, d_sc, d_ba, T, D, HC, 20); });
        printf("      of which  hc_dot %.1f us   hc_mix %.1f us   (launch floor %.1f us each)\n",
               usA, usB, floor1);

        /* Both spellings must agree bit-for-bit-ish, or the split is not the
         * same op. Compared here rather than in the oracle because this is
         * where both are already built. */
        std::vector<bf16> a((size_t)T * D), b((size_t)T * D);
        HC_CHECK(hipMemcpy(a.data(), d_o, a.size() * 2, hipMemcpyDeviceToHost));
        HC_CHECK(hipMemcpy(b.data(), d_o2, b.size() * 2, hipMemcpyDeviceToHost));
        unsigned diff = 0;
        for (size_t i = 0; i < a.size(); i++) diff += (a[i] != b[i]);
        printf("    split vs fused reduce: %u/%zu bf16 words differ\n", diff, a.size());
        printf("    per token, all 86 sub-layer sites: %.1f us fused / %.1f us split, "
               "+ %.1f us expand\n\n",
               us1 * 86, us3 * 86, us2 * 86);
        hipFree(d_part);
        hipFree(d_o2);
        hipFree(d_x); hipFree(d_o); hipFree(d_e); hipFree(d_fn); hipFree(d_sc); hipFree(d_ba);
        hipFree(d_mix);
    }

    /* ---------------- attention, decode: 1 query, 640 keys ---------------- */
    {
        const unsigned T = 1, H = 64, D = 512, NKV = 4096, TOPK = 640;
        std::vector<bf16> q((size_t)T * H * D), kv((size_t)NKV * D);
        std::vector<int> idx((size_t)T * TOPK);
        for (auto& v : q) v = f2bf_host(drand48());
        for (auto& v : kv) v = f2bf_host(drand48());
        for (auto& v : idx) v = (int)(lrand48() % NKV);
        bf16 *d_q, *d_kv, *d_o;
        int* d_i;
        float* d_s;
        HC_CHECK(hipMalloc(&d_q, q.size() * 2));
        HC_CHECK(hipMalloc(&d_kv, kv.size() * 2));
        HC_CHECK(hipMalloc(&d_o, q.size() * 2));
        HC_CHECK(hipMalloc(&d_i, idx.size() * 4));
        HC_CHECK(hipMalloc(&d_s, H * 4));
        HC_CHECK(hipMemcpy(d_q, q.data(), q.size() * 2, hipMemcpyHostToDevice));
        HC_CHECK(hipMemcpy(d_kv, kv.data(), kv.size() * 2, hipMemcpyHostToDevice));
        HC_CHECK(hipMemcpy(d_i, idx.data(), idx.size() * 4, hipMemcpyHostToDevice));

        printf("sparse attention, decode (T=1, H=64, D=512, topk=640 = 128 window + 512 indexed)\n");
        const double us = time_us([&] {
            k_attn<<<T * H, PLOW_THREADS>>>(d_o, d_q, d_kv, d_i, d_s, T, H, D, TOPK, 0.0442f);
        });
        /* Compulsory: q once, the 640 gathered KV rows ONCE (all 64 heads share
         * them), the index list, the output. */
        const double b = (double)H * D * 2 + (double)TOPK * D * 2 + (double)TOPK * 4 +
                         (double)H * D * 2;
        const double f = 4.0 * H * TOPK * D; /* qk dot + pv accumulate */
        row("sparse_attn", us, b, f);
        printf("    per token, all 43 layers: %.1f us\n\n", us * 43);
        hipFree(d_q); hipFree(d_kv); hipFree(d_o); hipFree(d_i); hipFree(d_s);
    }

    /* ---------------- indexer scoring, decode ----------------------------- */
    {
        const unsigned T = 1, H = 64, HD = 128, NC = 8192;
        std::vector<bf16> q((size_t)T * H * HD), ckv((size_t)NC * HD), w((size_t)T * H);
        for (auto& v : q) v = f2bf_host(drand48());
        for (auto& v : ckv) v = f2bf_host(drand48());
        bf16 *d_q, *d_c, *d_w;
        float* d_sc;
        HC_CHECK(hipMalloc(&d_q, q.size() * 2));
        HC_CHECK(hipMalloc(&d_c, ckv.size() * 2));
        HC_CHECK(hipMalloc(&d_w, w.size() * 2));
        HC_CHECK(hipMalloc(&d_sc, (size_t)T * NC * 4));
        HC_CHECK(hipMemcpy(d_q, q.data(), q.size() * 2, hipMemcpyHostToDevice));
        HC_CHECK(hipMemcpy(d_c, ckv.data(), ckv.size() * 2, hipMemcpyHostToDevice));

        printf("indexer scoring, decode (T=1, H=64, HD=128, NC=8192 compressed entries)\n");
        const double us = time_us(
            [&] { k_score<<<NCU, PLOW_THREADS>>>(d_sc, d_q, d_c, d_w, T, H, HD, NC); });
        const double b = (double)H * HD * 2 + (double)NC * HD * 2 + (double)NC * 4;
        const double f = 2.0 * H * NC * HD;
        row("index_score", us, b, f);
        printf("    per token, all 21 indexed layers: %.1f us\n\n", us * 21);
        hipFree(d_q); hipFree(d_c); hipFree(d_w); hipFree(d_sc);
    }

    return 0;
}
