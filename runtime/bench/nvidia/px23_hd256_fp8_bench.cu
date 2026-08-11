/* px23_hd256_fp8_bench.cu — the hd256 fp8 flash-prefill arm: correctness vs the PIPE=0 fallback,
 * and what it is worth.
 *
 * WHY.  PX-20 §5: an all-layer e4m3 packet (the configuration vLLM ships by default) emits hd256
 * FLASH_PREFILL_FP8; the PIPE=1 px4 fp8-mma arm is hd512-only and __trap()s on it, so plow falls
 * to the synchronous PIPE=0 staging path at 176.4 s of prefill per 127k request against 34.9 s for
 * the same model at bf16.  That single missing arm is the whole of PX-20's 7.61x conc-8 gap
 * (vLLM 42.55 vs plow 5.59 out tok/s).
 *
 * ARMS.  One source, TWO binaries, because the two arms cannot coexist in a translation unit:
 * d_flash_prefill<...,FP8KV> only exists under `#if !PLOW_NV_FA_PIPE` and px23 only under PIPE=1.
 *   -DPLOW_NV_FA_PIPE=0  -> arm `pipe0` : d_flash_prefill<256,64,32,true>, the SHIPPED fallback
 *   -DPLOW_NV_FA_PIPE=1  -> arm `px23`  : d_flash_prefill_px23<256,64,32>, the new arm
 * Both binaries build the SAME inputs from the same seed and score against the SAME in-bench f32
 * reference kernel, so their error columns and their reference hashes are directly comparable.
 * The reference hash matching across the two binaries is what proves they were fed identical data.
 *
 * THE VACUOUS-GATE TRAP (PX-22 bug 1, which burned an agent on this exact campaign): 0x7f is the
 * E4M3 NaN encoding.  A `rand() & 0x7f` operand fill puts one NaN in every 128 bytes; one NaN
 * anywhere in a reduction NaNs the whole output plane, so every arm hashes identically and the
 * gate passes for the wrong reason.  Three independent defences, all asserted per shape:
 *   1. rnd_e4m3() restricts the exponent field to 5..9 -> every operand finite in [0.25, 7.5];
 *   2. nonfinite counts over every input AND output buffer, asserted 0;
 *   3. the output hash is asserted != the ZERO-plane hash and != the reference hash of a
 *      DIFFERENT shape, so a degenerate (all-zero / all-NaN / stuck) output cannot pass.
 *
 * Run under perf-data/tools/gpulease.  Build: perf-data/px23_build.sh
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "op_attention.cuh"

#if !defined(PLOW_FP8_KV)
#error "build with -DPLOW_FP8_KV=1 (the fp8 prefill arms only exist in fp8 objects)"
#endif

#define CHK(x)                                                                                      \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            printf("CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_));                  \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

static const int HD = 256, BQ = 64, BKV = 32;
static const float E4M3_CEIL = 518.5f; /* TFLOP/s, e4m3 with f32 accumulate, RTX 5090 */

/* ============================== the two arms ============================================== */
#if PLOW_NV_FA_PIPE
#define ARM_NAME "px23"
__global__ __launch_bounds__(PLOW_NV_THREADS, 1) void k_arm(
    float* Opart, float* mlpart, const __nv_bfloat16* Q, const __nv_bfloat16* K,
    const __nv_bfloat16* V, __nv_bfloat16* O, unsigned seq_q, unsigned seq_kv, unsigned n_head,
    unsigned n_kv_head, unsigned q_pos0, unsigned window, unsigned nsplit, unsigned kv_stride,
    unsigned kv_mask, float scale, const float* ksc, const float* vsc) {
    extern __shared__ float lds[];
    d_flash_prefill_px23<HD, BQ, BKV>(Opart, mlpart, Q, K, V, O, seq_q, seq_kv, n_head, n_kv_head,
                                      q_pos0, window, nsplit, kv_stride, kv_mask, scale,
                                      blockIdx.x, gridDim.x, lds, ksc, vsc);
}
#else
#define ARM_NAME "pipe0"
__global__ __launch_bounds__(PLOW_NV_THREADS, 1) void k_arm(
    float* Opart, float* mlpart, const __nv_bfloat16* Q, const __nv_bfloat16* K,
    const __nv_bfloat16* V, __nv_bfloat16* O, unsigned seq_q, unsigned seq_kv, unsigned n_head,
    unsigned n_kv_head, unsigned q_pos0, unsigned window, unsigned nsplit, unsigned kv_stride,
    unsigned kv_mask, float scale, const float* ksc, const float* vsc) {
    extern __shared__ float lds[];
    d_flash_prefill<HD, BQ, BKV, true>(Opart, mlpart, Q, K, V, O, seq_q, seq_kv, n_head, n_kv_head,
                                       q_pos0, window, nsplit, kv_stride, kv_mask, scale,
                                       blockIdx.x, gridDim.x, lds, ksc, vsc);
}
#endif

/* ============================== f32 reference ============================================= */
/* Same math as both arms, in f32, one block per (q row, head).  Mirrors the kernels exactly:
 * exp2 with the log2(e)-folded scale, causal on absolute positions, sliding window, ring read. */
__global__ void k_ref(const __nv_bfloat16* Q, const unsigned char* K8, const unsigned char* V8,
                      const float* ksc, const float* vsc, __nv_bfloat16* O, unsigned seq_q,
                      unsigned seq_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0,
                      unsigned window, unsigned kv_stride, unsigned kv_mask, float scale) {
    const unsigned r = blockIdx.x, h = blockIdx.y;
    if (r >= seq_q) return;
    const unsigned hkv = h / (n_head / n_kv_head);
    const float ls = scale * 1.4426950408889634f;
    const int qabs = (int)(q_pos0 + r);
    __shared__ float acc[HD];
    __shared__ float sm, sl;
    __shared__ float wsum[32];
    for (int i = threadIdx.x; i < HD; i += blockDim.x) acc[i] = 0.0f;
    if (threadIdx.x == 0) { sm = -3.0e38f; sl = 0.0f; }
    __syncthreads();
    const __nv_bfloat16* q = Q + (size_t)r * n_head * HD + (size_t)h * HD;
    for (unsigned kv = 0; kv < seq_kv; kv++) {
        if ((int)kv > qabs) break;
        if (window && (unsigned)(qabs - (int)kv) >= window) continue;
        const size_t row = (size_t)hkv * kv_stride * HD + (size_t)(kv & kv_mask) * HD;
        /* dot in f32 */
        float part = 0.0f;
        for (int i = threadIdx.x; i < HD; i += blockDim.x)
            part += __bfloat162float(q[i]) * fp8_to_f32(K8[row + i]);
        for (int o = 16; o; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
        if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = part;
        __syncthreads();
        float dot = 0.0f;
        if (threadIdx.x == 0) {
            for (int i = 0; i < (int)(blockDim.x >> 5); i++) dot += wsum[i];
            dot *= ls * ksc[(size_t)hkv * kv_stride + (kv & kv_mask)];
            const float mn = fmaxf(sm, dot);
            const float corr = (sm == -3.0e38f) ? 0.0f : exp2f(sm - mn);
            wsum[0] = corr;
            wsum[1] = exp2f(dot - mn);
            sl = sl * corr + wsum[1];
            sm = mn;
        }
        __syncthreads();
        const float corr = wsum[0], p = wsum[1] * vsc[(size_t)hkv * kv_stride + (kv & kv_mask)];
        for (int i = threadIdx.x; i < HD; i += blockDim.x)
            acc[i] = acc[i] * corr + p * fp8_to_f32(V8[row + i]);
        __syncthreads();
    }
    const float inv = (sl > 0.0f) ? 1.0f / sl : 0.0f;
    for (int i = threadIdx.x; i < HD; i += blockDim.x)
        O[(size_t)(r * n_head + h) * HD + i] = __float2bfloat16(acc[i] * inv);
}

/* ============================== operands and hashing ====================================== */
static uint64_t xs = 0x243F6A8885A308D3ull;
static inline uint32_t xr() {
    xs ^= xs << 13;
    xs ^= xs >> 7;
    xs ^= xs << 17;
    return (uint32_t)(xs >> 32);
}
/* HOST-side E4M3 decode, so the anti-vacuity nonfinite check can run on the CPU. */
static inline float fp8_to_f32_host(unsigned char b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN; /* the sole E4M3 NaN encoding */
    float v;
    if (e == 0) v = (float)m * 0.0019531250f;            /* 2^-9 subnormal step */
    else v = ldexpf(1.0f + (float)m * 0.125f, e - 7);
    return s ? -v : v;
}
/* Finite E4M3 only.  exponent field 5..9 -> |v| in [0.25, 7.5].  NEVER 0x7f (the E4M3 NaN). */
static inline unsigned char rnd_e4m3() {
    const unsigned s = xr() & 1u, e = 5u + (xr() % 5u), m = xr() & 7u;
    return (unsigned char)((s << 7) | (e << 3) | m);
}
static inline uint64_t fnv(const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static size_t nonfinite_bf16(const std::vector<__nv_bfloat16>& v) {
    size_t n = 0;
    for (size_t i = 0; i < v.size(); i++)
        if (!isfinite(__bfloat162float(v[i]))) n++;
    return n;
}

struct Shape {
    const char* name;
    unsigned seq_q, seq_kv, n_head, n_kv_head, q_pos0, window, nsplit, ring_log2;
};

int main(int argc, char** argv) {
    int do_perf = 1, do_gate = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gate-only")) do_perf = 0;
        if (!strcmp(argv[i], "--perf-only")) do_gate = 0;
    }
    int dev = 0, sms = 0, optin = 0;
    CHK(cudaGetDevice(&dev));
    CHK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CHK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    const int smem = (int)(FA_PRE_SMEM_FLOATS(HD, BQ, BKV) * sizeof(float));
    printf("# arm=%s SMs=%d optin=%dB arena=%dB", ARM_NAME, sms, optin, smem);
#if PLOW_NV_FA_PIPE
    printf(" px23_claim=%dB", (int)(FA_PX23_SMEM_FLOATS(HD, BQ, BKV) * sizeof(float)));
#endif
    printf("\n");
    if (smem > optin) { printf("FATAL: arena %d > optin %d\n", smem, optin); return 1; }
    CHK(cudaFuncSetAttribute(k_arm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    /* ---------- correctness gate ---------- */
    /* Real Gemma-4 sliding-layer shape: n_head 16, n_kv_head 8 (gqa 2), hd 256, window 1024. */
    const Shape gates[] = {
        {"causal-nowin", 256, 1024, 16, 8, 0, 0, 1, 12},
        {"sliding-1024", 320, 2048, 16, 8, 0, 1024, 1, 12},
        {"ragged-qpos", 200, 1500, 16, 8, 777, 1024, 1, 12},
        {"ring-wrap", 256, 3000, 16, 8, 0, 1024, 1, 11}, /* kv_stride 2048 < seq_kv -> wraps */
        {"nsplit2", 128, 2048, 8, 8, 0, 0, 2, 12},
    };
    uint64_t zero_hash = 0, prev_ref_hash = 0;
    int gate_fail = 0;
    if (do_gate)
        for (unsigned gi = 0; gi < sizeof(gates) / sizeof(gates[0]); gi++) {
            const Shape s = gates[gi];
            const unsigned ring = 1u << s.ring_log2, kv_mask = ring - 1u;
            xs = 0x243F6A8885A308D3ull + gi * 0x9E3779B97F4A7C15ull;
            std::vector<__nv_bfloat16> hQ((size_t)s.seq_q * s.n_head * HD);
            for (size_t i = 0; i < hQ.size(); i++)
                hQ[i] = __float2bfloat16(((float)(xr() % 2001) - 1000.0f) * 0.002f);
            const size_t kvn = (size_t)s.n_kv_head * ring * HD;
            std::vector<unsigned char> hK(kvn), hV(kvn);
            for (size_t i = 0; i < kvn; i++) { hK[i] = rnd_e4m3(); hV[i] = rnd_e4m3(); }
            std::vector<float> hKs((size_t)s.n_kv_head * ring), hVs(hKs.size());
            for (size_t i = 0; i < hKs.size(); i++) {
                hKs[i] = 0.004f + (float)(xr() % 100) * 1e-5f;
                hVs[i] = 0.003f + (float)(xr() % 100) * 1e-5f;
            }
            /* Defence 1+2: no operand may be the E4M3 NaN, and none may be nonfinite. */
            size_t nan_in = 0;
            for (size_t i = 0; i < kvn; i++) {
                if ((hK[i] & 0x7f) == 0x7f || (hV[i] & 0x7f) == 0x7f) nan_in++;
                if (!isfinite(fp8_to_f32_host(hK[i])) || !isfinite(fp8_to_f32_host(hV[i]))) nan_in++;
            }
            nan_in += nonfinite_bf16(hQ);

            __nv_bfloat16 *dQ, *dO, *dOref;
            unsigned char *dK, *dV;
            float *dKs, *dVs, *dOpart, *dml;
            CHK(cudaMalloc(&dQ, hQ.size() * 2));
            CHK(cudaMalloc(&dK, kvn));
            CHK(cudaMalloc(&dV, kvn));
            CHK(cudaMalloc(&dKs, hKs.size() * 4));
            CHK(cudaMalloc(&dVs, hVs.size() * 4));
            const size_t on = (size_t)s.seq_q * s.n_head * HD;
            CHK(cudaMalloc(&dO, on * 2));
            CHK(cudaMalloc(&dOref, on * 2));
            CHK(cudaMalloc(&dOpart, on * s.nsplit * 4));
            CHK(cudaMalloc(&dml, (size_t)s.seq_q * s.n_head * s.nsplit * 2 * 4));
            CHK(cudaMemcpy(dQ, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
            CHK(cudaMemcpy(dK, hK.data(), kvn, cudaMemcpyHostToDevice));
            CHK(cudaMemcpy(dV, hV.data(), kvn, cudaMemcpyHostToDevice));
            CHK(cudaMemcpy(dKs, hKs.data(), hKs.size() * 4, cudaMemcpyHostToDevice));
            CHK(cudaMemcpy(dVs, hVs.data(), hVs.size() * 4, cudaMemcpyHostToDevice));
            CHK(cudaMemset(dO, 0, on * 2));
            CHK(cudaMemset(dOpart, 0, on * s.nsplit * 4));

            k_ref<<<dim3(s.seq_q, s.n_head), 256>>>(dQ, dK, dV, dKs, dVs, dOref, s.seq_q, s.seq_kv,
                                                    s.n_head, s.n_kv_head, s.q_pos0, s.window, ring,
                                                    kv_mask, 1.0f / sqrtf((float)HD));
            CHK(cudaGetLastError());
            k_arm<<<sms, PLOW_NV_THREADS, smem>>>(
                dOpart, dml, dQ, (const __nv_bfloat16*)dK, (const __nv_bfloat16*)dV, dO, s.seq_q,
                s.seq_kv, s.n_head, s.n_kv_head, s.q_pos0, s.window, s.nsplit, ring, kv_mask,
                1.0f / sqrtf((float)HD), dKs, dVs);
            CHK(cudaDeviceSynchronize());
            CHK(cudaGetLastError());

            std::vector<__nv_bfloat16> hO(on), hR(on);
            CHK(cudaMemcpy(hR.data(), dOref, on * 2, cudaMemcpyDeviceToHost));
            std::vector<float> hP;
            if (s.nsplit > 1) {
                /* Merge the split partials the way FLASH_MERGE does, then compare. */
                hP.resize(on * s.nsplit);
                std::vector<float> hM((size_t)s.seq_q * s.n_head * s.nsplit * 2);
                CHK(cudaMemcpy(hP.data(), dOpart, hP.size() * 4, cudaMemcpyDeviceToHost));
                CHK(cudaMemcpy(hM.data(), dml, hM.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t rh = 0; rh < (size_t)s.seq_q * s.n_head; rh++) {
                    float gm = -3.0e38f;
                    for (unsigned k = 0; k < s.nsplit; k++) gm = fmaxf(gm, hM[(rh * s.nsplit + k) * 2]);
                    float den = 0.0f;
                    std::vector<float> num(HD, 0.0f);
                    for (unsigned k = 0; k < s.nsplit; k++) {
                        const float m = hM[(rh * s.nsplit + k) * 2], l = hM[(rh * s.nsplit + k) * 2 + 1];
                        if (l <= 0.0f) continue;
                        const float c = exp2f(m - gm);
                        den += l * c;
                        for (int i = 0; i < HD; i++)
                            num[i] += hP[(rh * s.nsplit + k) * HD + i] * c;
                    }
                    const float inv = (den > 0.0f) ? 1.0f / den : 0.0f;
                    for (int i = 0; i < HD; i++)
                        hO[rh * HD + i] = __float2bfloat16(num[i] * inv);
                }
            } else {
                CHK(cudaMemcpy(hO.data(), dO, on * 2, cudaMemcpyDeviceToHost));
            }

            /* error, measured only where the reference exceeds 5% of its peak magnitude */
            float peak = 0.0f;
            for (size_t i = 0; i < on; i++) peak = fmaxf(peak, fabsf(__bfloat162float(hR[i])));
            double se = 0, sr = 0;
            float mabs = 0, mrel = 0;
            size_t nz = 0;
            for (size_t i = 0; i < on; i++) {
                const float a = __bfloat162float(hO[i]), b = __bfloat162float(hR[i]);
                const float d = fabsf(a - b);
                mabs = fmaxf(mabs, d);
                if (fabsf(b) > 0.05f * peak) mrel = fmaxf(mrel, d / fabsf(b));
                se += (double)d * d;
                sr += (double)b * b;
                if (a != 0.0f) nz++;
            }
            const uint64_t h_arm = fnv(hO.data(), on * 2), h_ref = fnv(hR.data(), on * 2);
            std::vector<__nv_bfloat16> zp(on, __float2bfloat16(0.f));
            const uint64_t h_zero = fnv(zp.data(), on * 2);
            if (!zero_hash) zero_hash = h_zero;
            const size_t nan_out = nonfinite_bf16(hO) + nonfinite_bf16(hR);

            /* Defence 3: not the zero plane, not a stuck value, not the previous shape's plane. */
            const int vac = (h_arm == h_zero) || (h_ref == h_zero) || (h_arm == prev_ref_hash) ||
                            (nz < on / 2);
            const int bad = nan_in || nan_out || vac || !(mrel < 0.05f) || !(peak > 0.0f);
            printf("GATE %-14s %-6s maxabs %.3e maxrel %.3e rms %.3e | hash %016llx ref %016llx "
                   "zero %016llx | nan_in %zu nan_out %zu nz %zu/%zu | %s\n",
                   s.name, ARM_NAME, mabs, mrel, sqrt(se / (sr > 0 ? sr : 1)),
                   (unsigned long long)h_arm, (unsigned long long)h_ref,
                   (unsigned long long)h_zero, nan_in, nan_out, nz, on, bad ? "FAIL" : "PASS");
            gate_fail += bad;
            prev_ref_hash = h_ref;
            cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dKs); cudaFree(dVs);
            cudaFree(dO); cudaFree(dOref); cudaFree(dOpart); cudaFree(dml);
        }

    /* ---------- perf ---------- */
    if (do_perf) {
        /* Real prefill chunk on a Gemma-4 sliding layer: 1024 q rows, n_head 16 / n_kv_head 8,
         * window 1024.  seq_kv sweeps to show the window makes per-tile cost flat. */
        const unsigned seq_q = 1024, n_head = 16, n_kv_head = 8;
        const unsigned ring = 1u << 17, kv_mask = ring - 1u;
        /* rows 0-2: the PRODUCTION sliding shape (window 1024) at three chunk positions.  Per-tile
         * cost must be FLAT, which is the check that the window bound is respected — a chunk deep
         * into a 32k context must cost the same as the first one.
         * rows 3-4: window 0 (FULL causal) at a large q_pos0.  These are the ASYMPTOTIC control:
         * 512 / 1024 KV tiles per work item instead of the sliding shape's ~17, so the per-q-tile
         * prologue (the Q e4m3 quant, which the sliding shape pays every 17 tiles) amortizes and
         * the mainloop rate shows through.  NOTE at seq_q == window == 1024 a window sweep is NOT
         * a control: causal already caps the reach at 1024, so both settings do the same work. */
        const unsigned kvs[] = {1088, 8256, 32832, 17408, 33792};
        const unsigned wins[] = {1024, 1024, 1024, 0, 0};
        const unsigned qp0s[] = {64, 7232, 31808, 16384, 32768};
        xs = 0xDEADBEEFCAFEBABEull;
        std::vector<__nv_bfloat16> hQ((size_t)seq_q * n_head * HD);
        for (size_t i = 0; i < hQ.size(); i++)
            hQ[i] = __float2bfloat16(((float)(xr() % 2001) - 1000.0f) * 0.002f);
        const size_t kvn = (size_t)n_kv_head * ring * HD;
        std::vector<unsigned char> hK(kvn), hV(kvn);
        for (size_t i = 0; i < kvn; i++) { hK[i] = rnd_e4m3(); hV[i] = rnd_e4m3(); }
        std::vector<float> hKs((size_t)n_kv_head * ring, 0.004f), hVs(hKs.size(), 0.003f);
        __nv_bfloat16 *dQ, *dO;
        unsigned char *dK, *dV;
        float *dKs, *dVs, *dOpart, *dml;
        CHK(cudaMalloc(&dQ, hQ.size() * 2));
        CHK(cudaMalloc(&dK, kvn));
        CHK(cudaMalloc(&dV, kvn));
        CHK(cudaMalloc(&dKs, hKs.size() * 4));
        CHK(cudaMalloc(&dVs, hVs.size() * 4));
        const size_t on = (size_t)seq_q * n_head * HD;
        CHK(cudaMalloc(&dO, on * 2));
        CHK(cudaMalloc(&dOpart, on * 4));
        CHK(cudaMalloc(&dml, (size_t)seq_q * n_head * 2 * 4));
        CHK(cudaMemcpy(dQ, hQ.data(), hQ.size() * 2, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dK, hK.data(), kvn, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dV, hV.data(), kvn, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dKs, hKs.data(), hKs.size() * 4, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dVs, hVs.data(), hVs.size() * 4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CHK(cudaEventCreate(&e0));
        CHK(cudaEventCreate(&e1));
        printf("# %-6s %8s %7s %8s %10s %10s %11s %10s %8s\n", "arm", "seq_kv", "window", "q_pos0",
               "ms", "tiles", "tiles/item", "TFLOP/s", "%ceil");
        for (unsigned ki = 0; ki < sizeof(kvs) / sizeof(kvs[0]); ki++) {
            const unsigned seq_kv = kvs[ki], window = wins[ki], q_pos0 = qp0s[ki];
            /* tiles the kernel actually executes — the SAME expression as its nt computation,
             * including the `if (window)` guard (getting that wrong silently mis-scales TFLOP/s) */
            double tiles = 0;
            for (unsigned q0 = 0; q0 < seq_q; q0 += BQ) {
                const long qabs_max = (long)(q_pos0 + q0 + BQ - 1);
                long eff_lo = 0;
                if (window) {
                    const long wfloor = (long)q_pos0 + q0 - (long)window + 1;
                    if (wfloor > 0) eff_lo = (wfloor / BKV) * BKV;
                }
                long cap = (long)seq_kv - 1;
                if (qabs_max < cap) cap = qabs_max;
                const long nt = (cap >= eff_lo) ? (cap - eff_lo) / BKV + 1 : 0;
                tiles += (double)nt * n_head;
            }
            const double flop = tiles * (double)BQ * BKV * 4.0 * HD;
            for (int rep = 0; rep < 3; rep++)
                k_arm<<<sms, PLOW_NV_THREADS, smem>>>(
                    dOpart, dml, dQ, (const __nv_bfloat16*)dK, (const __nv_bfloat16*)dV, dO, seq_q,
                    seq_kv, n_head, n_kv_head, q_pos0, window, 1, ring, kv_mask,
                    1.0f / sqrtf((float)HD), dKs, dVs);
            CHK(cudaDeviceSynchronize());
            const int iters = 20;
            CHK(cudaEventRecord(e0));
            for (int rep = 0; rep < iters; rep++)
                k_arm<<<sms, PLOW_NV_THREADS, smem>>>(
                    dOpart, dml, dQ, (const __nv_bfloat16*)dK, (const __nv_bfloat16*)dV, dO, seq_q,
                    seq_kv, n_head, n_kv_head, q_pos0, window, 1, ring, kv_mask,
                    1.0f / sqrtf((float)HD), dKs, dVs);
            CHK(cudaEventRecord(e1));
            CHK(cudaEventSynchronize(e1));
            float ms = 0;
            CHK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= iters;
            const double tf = flop / (ms * 1e-3) / 1e12;
            /* ns per KV tile PER BLOCK (comparable across arms of the same BKV) */
            const double nspt = ms * 1e6 / (tiles / sms);
            (void)nspt;
            printf("PERF %-6s %8u %7u %8u %10.4f %10.0f %11.1f %10.2f %7.1f%%\n", ARM_NAME, seq_kv,
                   window, q_pos0, ms, tiles, tiles / (seq_q / BQ * (double)n_head), tf,
                   100.0 * tf / E4M3_CEIL);
            if (tf > E4M3_CEIL) { printf("FATAL: %.2f TFLOP/s exceeds the %.1f ceiling\n", tf, E4M3_CEIL); return 1; }
        }
    }
    printf("%s %s\n", gate_fail ? "GATES FAILED" : "GATES PASSED", ARM_NAME);
    return gate_fail ? 1 : 0;
}
