/* Exact K3-DSpark query-block MLA oracle: seven queries share one committed context. */

#ifdef K3DSPARK_MLA_DEVICE
#include <hip/hip_runtime.h>

#include "amd_common.h"
#include "op_attention.h"

struct DsparkMlaArgs {
    unsigned long long opart, mlpart, out, qabs, qrope, ckv, krope, kvlen, wuv;
    unsigned n_tok, n_head, kv_stride, nsplit;
    float scale;
};

template <bool NONCAUSAL>
__device__ void run_flash(DsparkMlaArgs* a) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, 8)];
    d_flash_mla_decode<512, 64, 8, false, false, NONCAUSAL>(
        (float*)a->opart, (float*)a->mlpart, (const bf16*)a->qabs,
        (const bf16*)a->qrope, (const bf16*)a->ckv, (const bf16*)a->krope,
        (const int*)a->kvlen, 1, a->n_head, a->kv_stride, 0, a->scale, a->nsplit,
        PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds, nullptr, 0, a->n_tok);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k_dspark_noncausal(DsparkMlaArgs* a) {
    run_flash<true>(a);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k_dspark_causal_control(DsparkMlaArgs* a) {
    run_flash<false>(a);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1)
void k_dspark_merge(DsparkMlaArgs* a) {
    __shared__ float lds[512 + PLOW_WAVES * PLOW_MLA_FOLD_VT];
    d_mla_merge_fold<512, PLOW_MLA_FOLD_VT>(
        (bf16*)a->out, (const float*)a->opart, (const float*)a->mlpart,
        (const bf16*)a->wuv, a->n_tok, a->n_head, 128, a->nsplit, blockIdx.x,
        gridDim.x, lds);
}

#else
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

using bf16 = uint16_t;

struct DsparkMlaArgs {
    unsigned long long opart, mlpart, out, qabs, qrope, ckv, krope, kvlen, wuv;
    unsigned n_tok, n_head, kv_stride, nsplit;
    float scale;
};

static float bf2f(bf16 b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static bf16 f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

#define HIP_OK(call)                                                                            \
    do {                                                                                        \
        hipError_t e_ = (call);                                                                 \
        if (e_ != hipSuccess) {                                                                 \
            std::fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__,        \
                         __LINE__);                                                              \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

static void* upload(const void* src, size_t bytes) {
    void* dst = nullptr;
    if (hipMalloc(&dst, bytes) != hipSuccess ||
        hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        std::fprintf(stderr, "HIP upload failed\n");
        std::exit(1);
    }
    return dst;
}

static std::vector<double> reference(const std::vector<bf16>& qabs,
                                     const std::vector<bf16>& qrope,
                                     const std::vector<bf16>& ckv,
                                     const std::vector<bf16>& krope,
                                     const std::vector<bf16>& wuv, unsigned nt,
                                     unsigned nh, unsigned ctx, double scale) {
    constexpr unsigned DK = 512, DR = 64, V = 128;
    std::vector<double> out((size_t)nt * nh * V);
    std::vector<double> score(ctx), prob(ctx), latent(DK);
    for (unsigned t = 0; t < nt; ++t) {
        for (unsigned h = 0; h < nh; ++h) {
            const size_t qh = ((size_t)t * nh + h);
            double mx = -INFINITY;
            for (unsigned r = 0; r < ctx; ++r) {
                double s = 0.0;
                for (unsigned d = 0; d < DK; ++d)
                    s += (double)bf2f(qabs[qh * DK + d]) * bf2f(ckv[(size_t)r * DK + d]);
                for (unsigned d = 0; d < DR; ++d)
                    s += (double)bf2f(qrope[qh * DR + d]) * bf2f(krope[(size_t)r * DR + d]);
                score[r] = s * scale;
                mx = std::max(mx, score[r]);
            }
            double z = 0.0;
            for (unsigned r = 0; r < ctx; ++r) {
                prob[r] = std::exp(score[r] - mx);
                z += prob[r];
            }
            std::fill(latent.begin(), latent.end(), 0.0);
            for (unsigned r = 0; r < ctx; ++r) {
                const double p = prob[r] / z;
                for (unsigned d = 0; d < DK; ++d)
                    latent[d] += p * bf2f(ckv[(size_t)r * DK + d]);
            }
            for (unsigned v = 0; v < V; ++v) {
                double y = 0.0;
                for (unsigned d = 0; d < DK; ++d)
                    y += latent[d] * bf2f(wuv[((size_t)h * DK + d) * V + v]);
                out[qh * V + v] = y;
            }
        }
    }
    return out;
}

static double rel_rms(const std::vector<bf16>& got, const std::vector<double>& want) {
    double se = 0.0, sw = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = bf2f(got[i]) - want[i];
        se += d * d;
        sw += want[i] * want[i];
    }
    return std::sqrt(se / (sw + 1e-30));
}

int main(int argc, char** argv) {
    constexpr unsigned NT = 7, NH = 8, CTX = 37, NS = 4;
    constexpr unsigned DK = 512, DR = 64, V = 128;
    constexpr unsigned THREADS = 512;
    const float scale = 0.13085139f;

    hipDeviceProp_t prop{};
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    hipModule_t module{};
    HIP_OK(hipModuleLoad(&module, argc > 1 ? argv[1] : "k3_dspark_mla_gfx942.elf"));
    hipFunction_t flash_nc{}, flash_c{}, merge{};
    HIP_OK(hipModuleGetFunction(&flash_nc, module, "k_dspark_noncausal"));
    HIP_OK(hipModuleGetFunction(&flash_c, module, "k_dspark_causal_control"));
    HIP_OK(hipModuleGetFunction(&merge, module, "k_dspark_merge"));

    std::vector<bf16> qabs((size_t)NT * NH * DK), qrope((size_t)NT * NH * DR);
    std::vector<bf16> ckv((size_t)CTX * DK), krope((size_t)CTX * DR);
    std::vector<bf16> wuv((size_t)NH * DK * V);
    for (size_t i = 0; i < qabs.size(); ++i)
        qabs[i] = f2bf(0.035f + (float)((i * 17u) % 29u) * 0.0007f);
    for (size_t i = 0; i < qrope.size(); ++i)
        qrope[i] = f2bf(((int)((i * 13u) % 23u) - 11) * 0.002f);
    for (unsigned r = 0; r < CTX; ++r)
        for (unsigned d = 0; d < DK; ++d) {
            float x = ((int)(((size_t)r * DK + d) * 11u % 31u) - 15) * 0.002f;
            if (r + 6 >= CTX) x += 0.16f + 0.004f * (float)(r + 6 - CTX);
            ckv[(size_t)r * DK + d] = f2bf(x);
        }
    for (size_t i = 0; i < krope.size(); ++i)
        krope[i] = f2bf(((int)((i * 7u) % 19u) - 9) * 0.002f);
    for (size_t i = 0; i < wuv.size(); ++i)
        wuv[i] = f2bf(((int)((i * 5u) % 37u) - 18) * 0.0015f);

    const auto want = reference(qabs, qrope, ckv, krope, wuv, NT, NH, CTX, scale);
    int hlen = CTX;
    const size_t npart = (size_t)NT * NH * NS * DK;
    const size_t nml = (size_t)NT * NH * NS * 2;
    const size_t no = (size_t)NT * NH * V;
    DsparkMlaArgs a{};
    a.qabs = (uint64_t)upload(qabs.data(), qabs.size() * sizeof(bf16));
    a.qrope = (uint64_t)upload(qrope.data(), qrope.size() * sizeof(bf16));
    a.ckv = (uint64_t)upload(ckv.data(), ckv.size() * sizeof(bf16));
    a.krope = (uint64_t)upload(krope.data(), krope.size() * sizeof(bf16));
    a.kvlen = (uint64_t)upload(&hlen, sizeof(hlen));
    a.wuv = (uint64_t)upload(wuv.data(), wuv.size() * sizeof(bf16));
    a.n_tok = NT;
    a.n_head = NH;
    a.kv_stride = CTX;
    a.nsplit = NS;
    a.scale = scale;

    auto run = [&](hipFunction_t flash) {
        HIP_OK(hipMalloc((void**)&a.opart, npart * sizeof(float)));
        HIP_OK(hipMalloc((void**)&a.mlpart, nml * sizeof(float)));
        HIP_OK(hipMalloc((void**)&a.out, no * sizeof(bf16)));
        void* da = upload(&a, sizeof(a));
        void* args[] = {&da};
        const unsigned flash_blocks = NT * (NH / 8) * NS;
        HIP_OK(hipModuleLaunchKernel(flash, flash_blocks, 1, 1, THREADS, 1, 1, 0, nullptr,
                                    args, nullptr));
        const unsigned merge_blocks = NT * NH * (V / 32);
        HIP_OK(hipModuleLaunchKernel(merge, merge_blocks, 1, 1, THREADS, 1, 1, 0, nullptr,
                                    args, nullptr));
        HIP_OK(hipDeviceSynchronize());
        std::vector<bf16> out(no);
        HIP_OK(hipMemcpy(out.data(), (void*)a.out, no * sizeof(bf16), hipMemcpyDeviceToHost));
        HIP_OK(hipFree(da));
        HIP_OK(hipFree((void*)a.opart));
        HIP_OK(hipFree((void*)a.mlpart));
        HIP_OK(hipFree((void*)a.out));
        return out;
    };

    const auto nc = run(flash_nc);
    const auto causal = run(flash_c);
    const double nc_rms = rel_rms(nc, want);
    const double causal_rms = rel_rms(causal, want);
    double delta = 0.0, base = 0.0;
    for (size_t i = 0; i < nc.size(); ++i) {
        const double d = bf2f(nc[i]) - bf2f(causal[i]);
        delta += d * d;
        base += (double)bf2f(nc[i]) * bf2f(nc[i]);
    }
    const double causal_delta = std::sqrt(delta / (base + 1e-30));
    const bool pass = nc_rms < 1.0e-2 && causal_rms > nc_rms * 2.0 && causal_delta > 1.0e-2;
    std::printf("device=%s T=%u H=%u ctx=%u ns=%u noncausal_rms=%.6g "
                "causal_rms=%.6g causal_delta=%.6g\n%s\n",
                prop.name, NT, NH, CTX, NS, nc_rms, causal_rms, causal_delta,
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
#endif
