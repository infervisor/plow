/* L7 microbench: the K3 decode KDA chain (Conv3 -> StateStepG -> GatedNorm) as ONE
 * dataflow-gated packet (`d_kda_decode_fused_arm`, op_kda.h) against the served three-op chain.
 *
 * ONE GPU, the TP8 per-rank shape (H=12 heads, D=128, BV=8, W=4, P = H*D = 1536). Control =
 * `d_kda_conv3` (op 109, 256 WGs) -> `d_kda_state_step_g` (op 112, 192 WGs) -> `d_kda_gated_norm`
 * (op 103, 2 WGs), three launches. Fused = `d_kda_decode_fused_arm` on the step's 192 WGs, one
 * launch, the tiles of a head exchanging through the tagged scratch exactly as they do inside
 * the interpreter. Every iteration draws fresh random raw q/k/v, conv windows and taps, gate
 * inputs, beta, o_norm, output-gate logits and state; the fused arm's `y`, post-step `state`
 * and post-conv windows are compared WORD FOR WORD against the control's. Both the plain chain
 * and the L3-folded chain (`PLOW_KDA_F_FB_FOLD`: f_a + W_fb instead of f_raw) are checked.
 * The packets/gates the fusion removes are an interpreter cost; this measures the bodies.
 *
 * Build (gfx950, the served decode recipe's defines):
 *   hipcc --offload-arch=gfx950 -O3 -std=c++17 -DPLOW_BUCKET_DECODE=1 -DPLOW_K3=1 \
 *     -DPLOW_GEMV_MM=1 -DGV_UNROLL=14 -DPLOW_KDA_FB_FOLD=1 -DPLOW_KDA_DECODE_FUSED_ARM=1 \
 *     -I runtime/amd -I runtime/common runtime/bench/amd/kda_decode_fused_arm_bench.hip \
 *     -o /tmp/kda_decode_fused_arm_bench
 * Run:  GPU_LEASE_TIMEOUT=7200 perf-data/tools/gpulease -n 1 l7 /tmp/kda_decode_fused_arm_bench
 *   [H=12] [D=128] [BV=8] [W=4] [iters=64] [reps=200]
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

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
#define PLOW_KDA_F_FUSED_ARM 8u
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
void k_conv3(bf16* oq, bf16* ok, bf16* ov, const bf16* xq, const bf16* xk, const bf16* xv,
             const float* wq, const float* wk, const float* wv, float* sq, float* sk,
             float* sv, unsigned C, unsigned W, unsigned nblk)
BODY({
    d_kda_conv3(oq, ok, ov, xq, xk, xv, wq, wk, wv, sq, sk, sv, 1u, C, W, 1u, blockIdx.x, nblk,
                0, nullptr);
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

extern "C" __global__ __launch_bounds__(PLOW_THREADS) __attribute__((amdgpu_waves_per_eu(2, 2)))
void k_gnorm(bf16* y, const bf16* o, const float* norm_w, const bf16* og_raw, unsigned H,
             unsigned D, float eps, unsigned nblk)
BODY({ d_kda_gated_norm(y, o, norm_w, og_raw, 1u, H, D, eps, blockIdx.x, nblk); })

extern "C" __global__ __launch_bounds__(PLOW_THREADS) __attribute__((amdgpu_waves_per_eu(2, 2)))
void k_fused(bf16* y, const bf16* xq, const bf16* xk, const bf16* xv, const bf16* g_raw,
             const bf16* beta_raw, float* state, const float* wq, const float* wk,
             const float* wv, float* sq, float* sk, float* sv, const float* a_log,
             const float* dt_bias, const float* norm_w, const bf16* og_raw,
             unsigned long long* scratch, unsigned H, unsigned D, unsigned BV, unsigned W,
             unsigned flags, unsigned gate_mode, float scale, float lb, float eps,
             unsigned nblk, const bf16* w_fb)
BODY({
    __shared__ __align__(16) float lds[8 * 256];
    d_kda_decode_fused_arm(y, xq, xk, xv, g_raw, beta_raw, state, wq, wk, wv, sq, sk, sv, a_log,
                           dt_bias, norm_w, og_raw, (uint64_t*)scratch, H, D, BV, W, flags,
                           gate_mode, scale, lb, eps, blockIdx.x, nblk, lds, w_fb);
})

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
    const unsigned W = argc > 4 ? (unsigned)std::atoi(argv[4]) : 4u;
    const unsigned iters = argc > 5 ? (unsigned)std::atoi(argv[5]) : 64u;
    const unsigned reps = argc > 6 ? (unsigned)std::atoi(argv[6]) : 200u;
    const unsigned P = H * D, K = D;
    const unsigned conv_nblk = 256u;
    const unsigned step_nblk = (P / BV) < 256u ? (P / BV) : 256u;
    const unsigned norm_nblk = (H + 7u) / 8u;
    const unsigned gate_mode = PLOW_KDA_GATE_LOWER_BOUND;
    const float lb = -5.0f, scale = 1.0f / std::sqrt((float)D), eps = 1e-6f;
    const size_t st_elems = (size_t)H * D * D, cs_elems = (size_t)P * W;
    const size_t scratch_words = (size_t)H * (1 + 4 * (size_t)D);

    std::mt19937_64 rng(0x7a11u);
    std::uniform_real_distribution<float> act(-2.0f, 2.0f), wt(-0.08f, 0.08f), tap(-0.6f, 0.6f),
        st(-0.5f, 0.5f), al(-2.0f, 0.5f), db(-0.5f, 0.5f), nw(0.5f, 1.5f);
    std::vector<uint16_t> h_q(P), h_k(P), h_v(P), h_fraw(P), h_fa(K), h_wfb((size_t)P * K),
        h_beta(H), h_og(P);
    std::vector<float> h_alog(H), h_dtb(P), h_normw(D), h_state(st_elems), h_cw(3 * cs_elems),
        h_cs(3 * cs_elems);
    std::vector<uint16_t> h_yA(P), h_yB(P);
    std::vector<float> h_sA(st_elems), h_sB(st_elems), h_csA(3 * cs_elems), h_csB(3 * cs_elems);

    bf16 *d_q = dmalloc<bf16>(P), *d_k = dmalloc<bf16>(P), *d_v = dmalloc<bf16>(P),
         *d_fraw = dmalloc<bf16>(P), *d_fa = dmalloc<bf16>(K), *d_wfb = dmalloc<bf16>((size_t)P * K),
         *d_beta = dmalloc<bf16>(H), *d_og = dmalloc<bf16>(P), *d_mq = dmalloc<bf16>(P),
         *d_mk = dmalloc<bf16>(P), *d_mv = dmalloc<bf16>(P), *d_o = dmalloc<bf16>(P),
         *d_yA = dmalloc<bf16>(P), *d_yB = dmalloc<bf16>(P);
    float *d_alog = dmalloc<float>(H), *d_dtb = dmalloc<float>(P), *d_normw = dmalloc<float>(D),
          *d_s0 = dmalloc<float>(st_elems), *d_sA = dmalloc<float>(st_elems),
          *d_sB = dmalloc<float>(st_elems), *d_cw = dmalloc<float>(3 * cs_elems),
          *d_cs0 = dmalloc<float>(3 * cs_elems), *d_csA = dmalloc<float>(3 * cs_elems),
          *d_csB = dmalloc<float>(3 * cs_elems);
    unsigned long long* d_scratch = dmalloc<unsigned long long>(scratch_words);
    CK(hipMemset(d_scratch, 0, scratch_words * 8));
    const dim3 block(PLOW_THREADS);

    auto upload = [&]() {
        for (auto& x : h_q) x = f2bf_h(act(rng));
        for (auto& x : h_k) x = f2bf_h(act(rng));
        for (auto& x : h_v) x = f2bf_h(act(rng));
        for (auto& x : h_fraw) x = f2bf_h(act(rng));
        for (auto& x : h_fa) x = f2bf_h(act(rng));
        for (auto& x : h_wfb) x = f2bf_h(wt(rng));
        for (auto& x : h_beta) x = f2bf_h(act(rng));
        for (auto& x : h_og) x = f2bf_h(act(rng));
        for (auto& x : h_alog) x = al(rng);
        for (auto& x : h_dtb) x = db(rng);
        for (auto& x : h_normw) x = nw(rng);
        for (auto& x : h_state) x = st(rng);
        for (auto& x : h_cw) x = tap(rng);
        for (auto& x : h_cs) x = act(rng);
        CK(hipMemcpy(d_q, h_q.data(), P * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_k, h_k.data(), P * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_v, h_v.data(), P * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_fraw, h_fraw.data(), P * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_fa, h_fa.data(), K * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_wfb, h_wfb.data(), (size_t)P * K * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_beta, h_beta.data(), H * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_og, h_og.data(), P * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_alog, h_alog.data(), H * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_dtb, h_dtb.data(), P * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_normw, h_normw.data(), D * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_s0, h_state.data(), st_elems * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_cw, h_cw.data(), 3 * cs_elems * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_cs0, h_cs.data(), 3 * cs_elems * 4, hipMemcpyHostToDevice));
    };
    /* fold: gate logits from f_a x W_fb inside the step (L3), else from f_raw. */
    auto control = [&](float* state, float* cs, bf16* y, bool fold) {
        k_conv3<<<conv_nblk, block>>>(d_mq, d_mk, d_mv, d_q, d_k, d_v, d_cw, d_cw + cs_elems,
                                      d_cw + 2 * cs_elems, cs, cs + cs_elems, cs + 2 * cs_elems,
                                      P, W, conv_nblk);
        k_step<<<step_nblk, block>>>(d_o, d_mq, d_mk, d_mv, fold ? d_fa : d_fraw, d_beta, d_alog,
                                     d_dtb, gate_mode, lb, state, H, D, BV,
                                     PLOW_KDA_F_QK_L2NORM | (fold ? PLOW_KDA_F_FB_FOLD : 0u),
                                     scale, step_nblk, fold ? d_wfb : nullptr);
        k_gnorm<<<norm_nblk, block>>>(y, d_o, d_normw, d_og, H, D, eps, norm_nblk);
    };
    auto fused = [&](float* state, float* cs, bf16* y, bool fold) {
        k_fused<<<step_nblk, block>>>(
            y, d_q, d_k, d_v, fold ? d_fa : d_fraw, d_beta, state, d_cw, d_cw + cs_elems,
            d_cw + 2 * cs_elems, cs, cs + cs_elems, cs + 2 * cs_elems, d_alog, d_dtb, d_normw,
            d_og, d_scratch, H, D, BV, W,
            PLOW_KDA_F_QK_L2NORM | PLOW_KDA_F_FUSED_ARM | (fold ? PLOW_KDA_F_FB_FOLD : 0u),
            gate_mode, scale, lb, eps, step_nblk, fold ? d_wfb : nullptr);
    };

    uint64_t mism_y = 0, mism_s = 0, mism_cs = 0, checked = 0, nan_y = 0;
    unsigned shown = 0;
    for (unsigned it = 0; it < iters; it++) {
        const bool fold = (it & 1u) != 0u;
        upload();
        CK(hipMemcpy(d_sA, d_s0, st_elems * 4, hipMemcpyDeviceToDevice));
        CK(hipMemcpy(d_sB, d_s0, st_elems * 4, hipMemcpyDeviceToDevice));
        CK(hipMemcpy(d_csA, d_cs0, 3 * cs_elems * 4, hipMemcpyDeviceToDevice));
        CK(hipMemcpy(d_csB, d_cs0, 3 * cs_elems * 4, hipMemcpyDeviceToDevice));
        control(d_sA, d_csA, d_yA, fold);
        fused(d_sB, d_csB, d_yB, fold);
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(h_yA.data(), d_yA, P * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_yB.data(), d_yB, P * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_sA.data(), d_sA, st_elems * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_sB.data(), d_sB, st_elems * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_csA.data(), d_csA, 3 * cs_elems * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_csB.data(), d_csB, 3 * cs_elems * 4, hipMemcpyDeviceToHost));
        for (unsigned n = 0; n < P; n++) {
            if ((h_yB[n] & 0x7f80u) == 0x7f80u) nan_y++;
            if (h_yA[n] != h_yB[n]) {
                if (shown++ < 8)
                    std::printf("  y[%u] (fold %d): control %04x fused %04x\n", n, (int)fold,
                                h_yA[n], h_yB[n]);
                mism_y++;
            }
        }
        for (size_t i = 0; i < st_elems; i++)
            if (std::memcmp(&h_sA[i], &h_sB[i], 4) != 0) mism_s++;
        for (size_t i = 0; i < 3 * cs_elems; i++)
            if (std::memcmp(&h_csA[i], &h_csB[i], 4) != 0) mism_cs++;
        checked += P + st_elems + 3 * cs_elems;
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
        std::printf("%-52s %8.2f us\n", name, 1000.0 * ms / reps);
    };
    std::printf("H=%u D=%u BV=%u W=%u P=%u conv_nblk=%u step_nblk=%u norm_nblk=%u iters=%u "
                "reps=%u\n",
                H, D, BV, W, P, conv_nblk, step_nblk, norm_nblk, iters, reps);
    time("conv3 alone (256 WGs)", [&]() {
        k_conv3<<<conv_nblk, block>>>(d_mq, d_mk, d_mv, d_q, d_k, d_v, d_cw, d_cw + cs_elems,
                                      d_cw + 2 * cs_elems, d_csA, d_csA + cs_elems,
                                      d_csA + 2 * cs_elems, P, W, conv_nblk);
    });
    time("step alone (192 WGs)", [&]() {
        k_step<<<step_nblk, block>>>(d_o, d_mq, d_mk, d_mv, d_fraw, d_beta, d_alog, d_dtb,
                                     gate_mode, lb, d_sA, H, D, BV, PLOW_KDA_F_QK_L2NORM, scale,
                                     step_nblk, nullptr);
    });
    time("gated norm alone (2 WGs)", [&]() {
        k_gnorm<<<norm_nblk, block>>>(d_yA, d_o, d_normw, d_og, H, D, eps, norm_nblk);
    });
    time("control: conv3 + step + norm (3 launches)",
         [&]() { control(d_sA, d_csA, d_yA, false); });
    time("fused arm (1 launch)", [&]() { fused(d_sB, d_csB, d_yB, false); });
    time("control + f_b fold: conv3 + step(fold) + norm",
         [&]() { control(d_sA, d_csA, d_yA, true); });
    time("fused arm + f_b fold (1 launch)", [&]() { fused(d_sB, d_csB, d_yB, true); });

    std::printf("exact: %llu words checked over %u draws (fold alternating); fused != control: "
                "y %llu, state %llu, conv_state %llu; non-finite y %llu\n",
                (unsigned long long)checked, iters, (unsigned long long)mism_y,
                (unsigned long long)mism_s, (unsigned long long)mism_cs,
                (unsigned long long)nan_y);
    const bool ok = mism_y == 0 && mism_s == 0 && mism_cs == 0 && nan_y == 0;
    std::printf("status %d\n", ok ? 0 : 1);
    return ok ? 0 : 1;
}
