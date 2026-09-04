/* L3 microbench: the K3 f_b forget-gate GEMV folded into KdaStateStepG's prologue.
 *
 * ONE GPU, the TP8 per-rank shape (H=12 heads, D=128, N = H*D = 1536, K = 128). Control = the
 * served pair: `d_gemv` (op 10, the `d_gemv_t<1>` staged arm → `gemv_rows_rs`) writing `f_raw`,
 * then `d_kda_state_step_g` (op 112) reading it. Fused = `d_kda_state_step_g` with
 * PLOW_KDA_F_FB_FOLD, `f_a` in the g_raw slot and `W_fb`, one launch. Every iteration draws fresh
 * random q/k/v/f_a/W_fb/beta/state and compares the fused arm's `o` and post-step `state`
 * word-for-word against the control, plus the control's `f_raw` against a double-precision CPU
 * f_b (tolerance, harness sanity only). The packet/gate the fold removes is an interpreter cost;
 * this measures the bodies.
 *
 * Build (gfx950, the served decode recipe's defines):
 *   hipcc --offload-arch=gfx950 -O3 -std=c++17 -DPLOW_BUCKET_DECODE=1 -DPLOW_K3=1 \
 *     -DPLOW_GEMV_MM=1 -DGV_UNROLL=14 -DPLOW_KDA_FB_FOLD=1 -I runtime/amd -I runtime/common \
 *     runtime/bench/amd/kda_fb_fold_bench.hip -o /tmp/kda_fb_fold_bench
 * Run:  GPU_LEASE_TIMEOUT=7200 perf-data/tools/gpulease -n 1 l3 /tmp/kda_fb_fold_bench
 *   [H=12] [D=128] [BV=8] [gemv_nblk=256] [iters=64] [reps=200]
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

/* The op headers are DEVICE code (address-space-qualified pointers); the host pass sees only
 * the kernel signatures with empty bodies, which is all the launch stubs need. */
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_gemm.h"
#include "op_kda.h"
#define BODY(...) __VA_ARGS__
#else
typedef unsigned short bf16;
#define PLOW_THREADS 512
#define PLOW_KDA_GATE_LOWER_BOUND 1
#define PLOW_KDA_F_QK_L2NORM 1u
#define PLOW_KDA_F_FB_FOLD 4u
#define BODY(...) {}
#endif

#define CK(call)                                                                       \
    do {                                                                               \
        const hipError_t e_ = (call);                                                  \
        if (e_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__,                \
                         hipGetErrorString(e_));                                       \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS) __attribute__((amdgpu_waves_per_eu(2, 2)))
void k_gemv(bf16* out, const bf16* x, const bf16* W, unsigned N, unsigned K, unsigned nblk)
BODY({
    __shared__ __align__(16) bf16 lds[GM_LDS_HALVES];
    d_gemv(out, x, W, nullptr, nullptr, 1u, N, K, 0, 0.0f, blockIdx.x, nblk, lds);
})

extern "C" __global__ __launch_bounds__(PLOW_THREADS) __attribute__((amdgpu_waves_per_eu(2, 2)))
void k_step(bf16* o, const bf16* q, const bf16* k, const bf16* v, const bf16* g_raw,
            const bf16* beta_raw, const float* a_log, const float* dt_bias, unsigned gate_mode,
            float lb, float* state, unsigned H, unsigned D, unsigned BV, unsigned flags,
            float scale, unsigned nblk, const bf16* w_fb)
BODY({
    __shared__ __align__(16) float lds[8 * 256];
    d_kda_state_step_g(o, q, k, v, g_raw, beta_raw, a_log, dt_bias, gate_mode, lb, state, 1u, H,
                       D, BV, flags, scale, blockIdx.x, nblk, lds, 0, nullptr, w_fb);
})

static float bf2f_h(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static uint16_t f2bf_h(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

template <class T>
static T* dmalloc(size_t n) {
    T* p;
    CK(hipMalloc(&p, n * sizeof(T)));
    return p;
}

int main(int argc, char** argv) {
    const unsigned H = argc > 1 ? (unsigned)std::atoi(argv[1]) : 12u;
    const unsigned D = argc > 2 ? (unsigned)std::atoi(argv[2]) : 128u;
    const unsigned BV = argc > 3 ? (unsigned)std::atoi(argv[3]) : 8u;
    const unsigned gemv_nblk = argc > 4 ? (unsigned)std::atoi(argv[4]) : 256u;
    const unsigned iters = argc > 5 ? (unsigned)std::atoi(argv[5]) : 64u;
    const unsigned reps = argc > 6 ? (unsigned)std::atoi(argv[6]) : 200u;
    const unsigned P = H * D, K = D;
    const unsigned step_nblk = (P / BV) < 256u ? (P / BV) : 256u;
    const unsigned gate_mode = PLOW_KDA_GATE_LOWER_BOUND;
    const float lb = -5.0f, scale = 1.0f / std::sqrt((float)D);
    const size_t st_elems = (size_t)H * D * D;

    std::mt19937_64 rng(0x4b3fu);
    std::uniform_real_distribution<float> act(-2.0f, 2.0f), wt(-0.08f, 0.08f),
        st(-0.5f, 0.5f), al(-2.0f, 0.5f), db(-0.5f, 0.5f);
    std::vector<uint16_t> h_q(P), h_k(P), h_v(P), h_fa(K), h_w((size_t)P * K), h_beta(H);
    std::vector<float> h_alog(H), h_dtb(P), h_state(st_elems);
    std::vector<uint16_t> h_fraw(P), h_oA(P), h_oB(P);
    std::vector<float> h_sA(st_elems), h_sB(st_elems);

    bf16 *d_q = dmalloc<bf16>(P), *d_k = dmalloc<bf16>(P), *d_v = dmalloc<bf16>(P),
         *d_fa = dmalloc<bf16>(K), *d_w = dmalloc<bf16>((size_t)P * K),
         *d_beta = dmalloc<bf16>(H), *d_fraw = dmalloc<bf16>(P), *d_oA = dmalloc<bf16>(P),
         *d_oB = dmalloc<bf16>(P);
    float *d_alog = dmalloc<float>(H), *d_dtb = dmalloc<float>(P),
          *d_s0 = dmalloc<float>(st_elems), *d_sA = dmalloc<float>(st_elems),
          *d_sB = dmalloc<float>(st_elems);
    const dim3 block(PLOW_THREADS);

    auto upload = [&]() {
        for (auto& x : h_q) x = f2bf_h(act(rng));
        for (auto& x : h_k) x = f2bf_h(act(rng));
        for (auto& x : h_v) x = f2bf_h(act(rng));
        for (auto& x : h_fa) x = f2bf_h(act(rng));
        for (auto& x : h_w) x = f2bf_h(wt(rng));
        for (auto& x : h_beta) x = f2bf_h(act(rng));
        for (auto& x : h_alog) x = al(rng);
        for (auto& x : h_dtb) x = db(rng);
        for (auto& x : h_state) x = st(rng);
        CK(hipMemcpy(d_q, h_q.data(), P * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_k, h_k.data(), P * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_v, h_v.data(), P * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_fa, h_fa.data(), K * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_w, h_w.data(), (size_t)P * K * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_beta, h_beta.data(), H * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_alog, h_alog.data(), H * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_dtb, h_dtb.data(), P * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_s0, h_state.data(), st_elems * 4, hipMemcpyHostToDevice));
    };
    auto control = [&](float* state, bf16* o) {
        k_gemv<<<gemv_nblk, block>>>(d_fraw, d_fa, d_w, P, K, gemv_nblk);
        k_step<<<step_nblk, block>>>(o, d_q, d_k, d_v, d_fraw, d_beta, d_alog, d_dtb, gate_mode,
                                     lb, state, H, D, BV, PLOW_KDA_F_QK_L2NORM, scale, step_nblk,
                                     nullptr);
    };
    auto fused = [&](float* state, bf16* o) {
        k_step<<<step_nblk, block>>>(o, d_q, d_k, d_v, d_fa, d_beta, d_alog, d_dtb, gate_mode,
                                     lb, state, H, D, BV, PLOW_KDA_F_QK_L2NORM | PLOW_KDA_F_FB_FOLD,
                                     scale, step_nblk, d_w);
    };

    uint64_t mism_o = 0, mism_s = 0, checked = 0;
    double max_rel = 0.0;
    unsigned shown = 0;
    for (unsigned it = 0; it < iters; it++) {
        upload();
        CK(hipMemcpy(d_sA, d_s0, st_elems * 4, hipMemcpyDeviceToDevice));
        CK(hipMemcpy(d_sB, d_s0, st_elems * 4, hipMemcpyDeviceToDevice));
        control(d_sA, d_oA);
        fused(d_sB, d_oB);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(h_fraw.data(), d_fraw, P * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_oA.data(), d_oA, P * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_oB.data(), d_oB, P * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_sA.data(), d_sA, st_elems * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_sB.data(), d_sB, st_elems * 4, hipMemcpyDeviceToHost));
        for (unsigned n = 0; n < P; n++) {
            if (h_oA[n] != h_oB[n]) {
                if (shown++ < 8)
                    std::printf("  o[%u]: control %04x fused %04x\n", n, h_oA[n], h_oB[n]);
                mism_o++;
            }
            double ref = 0.0;
            for (unsigned kk = 0; kk < K; kk++)
                ref += (double)bf2f_h(h_w[(size_t)n * K + kk]) * (double)bf2f_h(h_fa[kk]);
            const double got = bf2f_h(h_fraw[n]);
            const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-2);
            if (rel > max_rel) max_rel = rel;
        }
        for (size_t i = 0; i < st_elems; i++)
            if (std::memcmp(&h_sA[i], &h_sB[i], 4) != 0) mism_s++;
        checked += P + st_elems;
    }

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto time = [&](const char* name, auto&& fn) {
        for (unsigned r = 0; r < 20; r++) fn();
        CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0));
        for (unsigned r = 0; r < reps; r++) fn();
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms = 0.0f;
        CK(hipEventElapsedTime(&ms, e0, e1));
        std::printf("%-44s %8.2f us\n", name, 1000.0 * ms / reps);
    };
    std::printf("H=%u D=%u BV=%u N=%u K=%u gemv_nblk=%u step_nblk=%u iters=%u reps=%u\n", H, D,
                BV, P, K, gemv_nblk, step_nblk, iters, reps);
    time("gemv f_b alone (1 launch)", [&]() {
        k_gemv<<<gemv_nblk, block>>>(d_fraw, d_fa, d_w, P, K, gemv_nblk);
    });
    time("step alone, gate from f_raw (1 launch)", [&]() {
        k_step<<<step_nblk, block>>>(d_oA, d_q, d_k, d_v, d_fraw, d_beta, d_alog, d_dtb,
                                     gate_mode, lb, d_sA, H, D, BV, PLOW_KDA_F_QK_L2NORM, scale,
                                     step_nblk, nullptr);
    });
    time("control: gemv + step (2 launches)", [&]() { control(d_sA, d_oA); });
    time("fused: step with f_b fold (1 launch)", [&]() { fused(d_sB, d_oB); });

    std::printf("exact: %llu words checked; fused != control: o %llu, state %llu; control f_raw "
                "vs cpu f64 max rel %.3e\n",
                (unsigned long long)checked, (unsigned long long)mism_o,
                (unsigned long long)mism_s, max_rel);
    const bool ok = mism_o == 0 && mism_s == 0 && max_rel < 1.6e-2;
    std::printf("status %d\n", ok ? 0 : 1);
    return ok ? 0 : 1;
}
