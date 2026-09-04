/* Single-GPU oracle + isolated timing for runtime/amd/attn_res_f32mix_gfx950.hip.
 *
 * Reference: a CPU port of vllm/models/kimi_k3/amd/ops/attn_res.py (v0.28.0, BLOCK_L = 1):
 * prefix = bf16(res_pre + bf16(res_a + res_b)); scores on BF16 sources; online softmax mix in
 * f32; output RMSNorm on the unrounded f32 mix with its own epsilon. Row reductions of the port
 * run in double so the comparison measures the GPU's rounding, not the CPU's summation order;
 * everything else (rsqrt, exp, the mix, the norm) is f32 in the reference's order.
 *
 * Three GPU arms are compared per case:
 *   object   the served .elf, loaded with hipModuleLoad and launched with the plowrt kernarg ABI
 *   probe    the same `ar_body` compiled here with an f32 side output (proves the mix before
 *            BF16 rounding; its BF16 output must be byte-identical to the object's)
 *   control  production `d_attn_res` (op_k3.h) at the interpreter's geometry — documents the
 *            BF16-seam semantic delta and is the timing baseline
 *
 *   oracle --elf <path> [--t N] [--nb 0,1,4,8] [--time] [--grid 256,512,1024,2048] [--seed S]
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define PLOW_WG_WAVES 8
#define PLOW_MATERIALIZED_RESIDUAL_INPUT 1
#include "amd_common.h"
#include "op_k3.h"
#define PLOW_ATTNRES_F32MIX_BODY_ONLY 1
#include "attn_res_f32mix_gfx950.hip"

#define HC(x)                                                                                 \
    do {                                                                                      \
        hipError_t e_ = (x);                                                                  \
        if (e_ != hipSuccess) {                                                               \
            std::fprintf(stderr, "HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

enum { HID = 7168, CAP = 8, CONTROL_LDS = 147464 };

struct KernArgs {
    bf16* out;
    bf16* prefix;
    bf16* ring;
    const float* score_w;
    const bf16* push_src;
    const bf16* gamma;
    const bf16* res_a;
    const bf16* res_b;
    const bf16* res_pre;
    unsigned t, hid, nb, push_row, nbcap;
    float eps, out_eps;
    unsigned reserved;
};
static_assert(sizeof(KernArgs) == 104, "kernarg ABI");

__global__ __launch_bounds__(AR_THREADS, 4) void probe_kernel(KernArgs a, float* f32out) {
    ar_body<true>(a.out, a.prefix, a.ring, a.score_w, a.push_src, a.gamma, a.res_a, a.res_b,
                  a.res_pre, a.t, a.hid, a.nb, a.push_row, a.nbcap, a.eps, a.out_eps,
                  a.reserved, f32out);
}

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void control_kernel(KernArgs a,
                                                                              unsigned iters) {
    __shared__ __align__(16) char raw[CONTROL_LDS];
    for (unsigned i = 0; i < iters; ++i) {
        if (a.res_a)
            d_materialize_residual(a.prefix, a.res_a, a.res_b, a.res_pre, a.t, a.hid, blockIdx.x,
                                   gridDim.x);
        d_attn_res(a.out, a.prefix, a.ring, a.score_w, a.t, a.hid, a.nb, a.nbcap, a.eps,
                   blockIdx.x, gridDim.x, (float*)raw, a.push_src, a.push_row, a.gamma);
        __syncthreads();
    }
}

/* Streaming floor for the same access pattern (9 nt row reads + a + b, one bf16 write per
 * token; no barriers): what the object could reach if it were purely bandwidth-bound. */
__global__ __launch_bounds__(AR_THREADS, 4) void bw_kernel(KernArgs a) {
    const unsigned off0 = threadIdx.x * AR_VEC * 2u;
    const auto* rg = as_glob((const bf16*)a.ring);
    for (unsigned t = blockIdx.x; t < a.t; t += gridDim.x) {
        const size_t pofs = (size_t)t * AR_HID, bofs = (size_t)t * a.nbcap * AR_HID;
        ar_vec acc[AR_NV], v[AR_NV];
        ar_load_row(acc, as_glob(a.res_a) + pofs, off0);
        ar_load_row(v, as_glob(a.res_b) + pofs, off0);
        for (unsigned q = 0; q < AR_NV; ++q) acc[q] ^= v[q];
        for (unsigned r = 0; r < a.nb; ++r) {
            ar_load_row_nt(v, rg + bofs + (size_t)r * AR_HID, off0);
            for (unsigned q = 0; q < AR_NV; ++q) acc[q] ^= v[q];
        }
        ar_store_row(as_glob(a.out) + pofs, off0, acc);
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
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static unsigned long long rng = 0x243f6a8885a308d3ull;
static float rnd(float lo, float hi) {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return lo + (hi - lo) * (float)((rng >> 40) / 16777216.0);
}
static unsigned long long fnv(const void* ptr, size_t bytes) {
    const unsigned char* p = (const unsigned char*)ptr;
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

/* attn_res.py in f32 with double row sums. `sources[r]` are the BF16 rows in vLLM's order
 * (ring rows 0..nb-1, prefix last). */
static void vllm_ref(std::vector<float>& out_f32, const std::vector<const bf16*>& sources,
                     const float* w, const bf16* gamma, float eps, float out_eps) {
    const float inv_h = 1.0f / (float)HID;
    float m = -INFINITY, denom = 0.0f;
    std::vector<float> mixed(HID, 0.0f);
    for (const bf16* x : sources) {
        double ss = 0.0, sw = 0.0;
        for (int d = 0; d < HID; ++d) {
            const double f = bf2f_h(x[d]);
            ss += f * f;
            sw += f * (double)w[d];
        }
        const float rstd = 1.0f / std::sqrt((float)ss * inv_h + eps);
        const float s = (float)sw * rstd;
        const float nm = std::max(m, s);
        const float os = std::exp(m - nm), bs = std::exp(s - nm);
        denom = denom * os + bs;
        for (int d = 0; d < HID; ++d) mixed[d] = mixed[d] * os + bs * bf2f_h(x[d]);
        m = nm;
    }
    for (int d = 0; d < HID; ++d) mixed[d] /= denom;
    out_f32.assign(mixed.begin(), mixed.end());
    if (gamma) {
        double ss = 0.0;
        for (int d = 0; d < HID; ++d) ss += (double)mixed[d] * mixed[d];
        const float rstd = 1.0f / std::sqrt((float)ss * inv_h + out_eps);
        for (int d = 0; d < HID; ++d) out_f32[d] = mixed[d] * rstd * bf2f_h(gamma[d]);
    }
}

/* Same contract entirely in double (probs first, then the mix): the ordering-free truth. */
static void ref64(std::vector<double>& out, const std::vector<const bf16*>& sources,
                  const float* w, const bf16* gamma, float eps, float out_eps) {
    const size_t n = sources.size();
    std::vector<double> logit(n);
    for (size_t r = 0; r < n; ++r) {
        double ss = 0.0, sw = 0.0;
        for (int d = 0; d < HID; ++d) {
            const double f = bf2f_h(sources[r][d]);
            ss += f * f;
            sw += f * (double)w[d];
        }
        logit[r] = sw / std::sqrt(ss / HID + (double)eps);
    }
    const double mx = *std::max_element(logit.begin(), logit.end());
    double z = 0.0;
    std::vector<double> p(n);
    for (size_t r = 0; r < n; ++r) z += p[r] = std::exp(logit[r] - mx);
    out.assign(HID, 0.0);
    for (size_t r = 0; r < n; ++r)
        for (int d = 0; d < HID; ++d) out[d] += p[r] / z * bf2f_h(sources[r][d]);
    if (gamma) {
        double ss = 0.0;
        for (double x : out) ss += x * x;
        const double inv = 1.0 / std::sqrt(ss / HID + (double)out_eps);
        for (int d = 0; d < HID; ++d) out[d] = out[d] * inv * bf2f_h(gamma[d]);
    }
}

struct Err {
    double max_abs = 0.0, se = 0.0, sr = 0.0;
    size_t n = 0, exact = 0;
    void add(double got, double ref) {
        const double e = got - ref;
        max_abs = std::max(max_abs, std::abs(e));
        se += e * e;
        sr += ref * ref;
        ++n;
        exact += got == ref;
    }
    double rel_l2() const { return std::sqrt(se / std::max(sr, 1e-300)); }
};

static unsigned g_obj_threads = AR_THREADS; /* --threads: the object's workgroup size */
struct Module {
    hipModule_t mod{};
    hipFunction_t fn{};
    explicit Module(const char* path) {
        HC(hipModuleLoad(&mod, path));
        HC(hipModuleGetFunction(&fn, mod, "plow_attn_res_f32mix_gfx950"));
    }
    void launch(const KernArgs& a, unsigned grid) {
        KernArgs args = a;
        size_t size = sizeof(args);
        void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                         &size, HIP_LAUNCH_PARAM_END};
        HC(hipModuleLaunchKernel(fn, grid, 1, 1, g_obj_threads, 1, 1, 0, nullptr, nullptr, extra));
    }
};

struct Case {
    unsigned t, nb;
    bool res, pre, push, gamma;
};

struct Data {
    unsigned t;
    std::vector<bf16> res_a, res_b, res_pre, prefix, ring, push_src, gamma;
    std::vector<float> score;
    std::vector<bf16> norm, proj;
};

static Data make_data(unsigned t, float scale) {
    Data d;
    d.t = t;
    const size_t rows = (size_t)t * HID;
    d.res_a.resize(rows);
    d.res_b.resize(rows);
    d.res_pre.resize(rows);
    d.prefix.resize(rows);
    d.ring.resize(rows * CAP);
    d.push_src.resize(rows);
    d.gamma.resize(HID);
    d.norm.resize(HID);
    d.proj.resize(HID);
    d.score.resize(HID);
    for (int i = 0; i < HID; ++i) {
        d.norm[i] = f2bf_h(rnd(0.75f, 1.25f));
        d.proj[i] = f2bf_h(rnd(-0.05f, 0.05f));
        d.gamma[i] = f2bf_h(rnd(0.5f, 1.5f));
        d.score[i] = bf2f_h(d.norm[i]) * bf2f_h(d.proj[i]);
    }
    for (size_t i = 0; i < rows; ++i) {
        d.res_a[i] = f2bf_h(scale * rnd(-2.0f, 2.0f));
        d.res_b[i] = f2bf_h(scale * rnd(-0.35f, 0.35f));
        d.res_pre[i] = f2bf_h(scale * rnd(-1.0f, 1.0f));
        d.prefix[i] = f2bf_h(scale * rnd(-2.0f, 2.0f));
        d.push_src[i] = f2bf_h(scale * rnd(-2.0f, 2.0f));
    }
    for (bf16& x : d.ring) x = f2bf_h(scale * rnd(-2.0f, 2.0f));
    return d;
}

struct Dev {
    bf16 *res_a, *res_b, *res_pre, *prefix, *ring, *push_src, *gamma, *out, *out_ctl, *prefix_ctl,
        *ring_ctl;
    float *score, *f32out;
};

static Dev upload(const Data& d) {
    Dev v{};
    const size_t rows = (size_t)d.t * HID;
    auto up = [](bf16** p, const std::vector<bf16>& h) {
        HC(hipMalloc(p, h.size() * 2));
        HC(hipMemcpy(*p, h.data(), h.size() * 2, hipMemcpyHostToDevice));
    };
    up(&v.res_a, d.res_a);
    up(&v.res_b, d.res_b);
    up(&v.res_pre, d.res_pre);
    up(&v.prefix, d.prefix);
    up(&v.ring, d.ring);
    up(&v.push_src, d.push_src);
    up(&v.gamma, d.gamma);
    up(&v.prefix_ctl, d.prefix);
    up(&v.ring_ctl, d.ring);
    HC(hipMalloc(&v.score, HID * 4));
    HC(hipMemcpy(v.score, d.score.data(), HID * 4, hipMemcpyHostToDevice));
    HC(hipMalloc(&v.out, rows * 2));
    HC(hipMalloc(&v.out_ctl, rows * 2));
    HC(hipMalloc(&v.f32out, rows * 4));
    return v;
}

static KernArgs args_for(const Dev& v, const Case& c, bool ctl, float eps, float out_eps) {
    KernArgs a{};
    a.out = ctl ? v.out_ctl : v.out;
    a.prefix = ctl ? v.prefix_ctl : v.prefix;
    a.ring = ctl ? v.ring_ctl : v.ring;
    a.score_w = v.score;
    a.push_src = c.push ? v.push_src : nullptr;
    a.gamma = c.gamma ? v.gamma : nullptr;
    a.res_a = c.res ? v.res_a : nullptr;
    a.res_b = c.res ? v.res_b : nullptr;
    a.res_pre = (c.res && c.pre) ? v.res_pre : nullptr;
    a.t = c.t;
    a.hid = HID;
    a.nb = c.nb;
    a.push_row = c.nb;
    a.nbcap = CAP;
    a.eps = eps;
    a.out_eps = out_eps;
    a.reserved = 0;
    return a;
}

static int run_case(Module& obj, const Data& d, const Dev& v, const Case& c, float eps,
                    float out_eps, unsigned grid) {
    const size_t rows = (size_t)c.t * HID;
    // Reset the tensors the arms write (prefix, ring push row) so every arm starts identically.
    HC(hipMemcpy(v.prefix, d.prefix.data(), rows * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(v.prefix_ctl, d.prefix.data(), rows * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(v.ring, d.ring.data(), rows * CAP * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(v.ring_ctl, d.ring.data(), rows * CAP * 2, hipMemcpyHostToDevice));
    HC(hipMemset(v.out, 0xff, rows * 2));
    HC(hipMemset(v.out_ctl, 0xff, rows * 2));
    HC(hipMemset(v.f32out, 0xff, rows * 4));

    const KernArgs a = args_for(v, c, false, eps, out_eps);
    const KernArgs actl = args_for(v, c, true, eps, out_eps);
    obj.launch(a, std::min(grid, c.t));
    HC(hipDeviceSynchronize());
    std::vector<bf16> obj_out(rows), obj_prefix(rows), obj_ring(rows * CAP);
    HC(hipMemcpy(obj_out.data(), v.out, rows * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(obj_prefix.data(), v.prefix, rows * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(obj_ring.data(), v.ring, rows * CAP * 2, hipMemcpyDeviceToHost));
    // Repeat: byte-stable.
    obj.launch(a, std::min(grid, c.t));
    HC(hipDeviceSynchronize());
    std::vector<bf16> rep(rows);
    HC(hipMemcpy(rep.data(), v.out, rows * 2, hipMemcpyDeviceToHost));
    const bool stable = rep == obj_out;

    // Probe (same body, f32 side output). Rerun from the same reset state.
    HC(hipMemcpy(v.prefix, d.prefix.data(), rows * 2, hipMemcpyHostToDevice));
    HC(hipMemcpy(v.ring, d.ring.data(), rows * CAP * 2, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(probe_kernel, std::min(grid, c.t), AR_THREADS, 0, 0, a, v.f32out);
    HC(hipDeviceSynchronize());
    std::vector<bf16> probe_out(rows);
    std::vector<float> probe_f32(rows);
    HC(hipMemcpy(probe_out.data(), v.out, rows * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(probe_f32.data(), v.f32out, rows * 4, hipMemcpyDeviceToHost));
    const bool probe_same = probe_out == obj_out;

    // Control: production interpreter body.
    hipLaunchKernelGGL(control_kernel, std::min(256u, c.t), PLOW_THREADS, 0, 0, actl, 1u);
    HC(hipDeviceSynchronize());
    std::vector<bf16> ctl_out(rows), ctl_prefix(rows);
    HC(hipMemcpy(ctl_out.data(), v.out_ctl, rows * 2, hipMemcpyDeviceToHost));
    HC(hipMemcpy(ctl_prefix.data(), v.prefix_ctl, rows * 2, hipMemcpyDeviceToHost));

    // CPU reference.
    Err e_f32, e_f64, e_bf16, e_ctl, e_ctl_obj, e_probe64;
    size_t prefix_bad = 0, push_bad = 0;
    for (unsigned t = 0; t < c.t; ++t) {
        const size_t pofs = (size_t)t * HID, bofs = (size_t)t * CAP * HID;
        std::vector<bf16> pfx(HID);
        for (int i = 0; i < HID; ++i) {
            if (c.res) {
                const bf16 inner =
                    f2bf_h(bf2f_h(d.res_a[pofs + i]) + bf2f_h(d.res_b[pofs + i]));
                pfx[i] = c.pre ? f2bf_h(bf2f_h(d.res_pre[pofs + i]) + bf2f_h(inner)) : inner;
            } else {
                pfx[i] = d.prefix[pofs + i];
            }
            prefix_bad += obj_prefix[pofs + i] != pfx[i];
            prefix_bad += ctl_prefix[pofs + i] != pfx[i];
            if (c.push) push_bad += obj_ring[bofs + (size_t)c.nb * HID + i] != d.push_src[pofs + i];
        }
        std::vector<const bf16*> src;
        for (unsigned r = 0; r < c.nb; ++r) src.push_back(d.ring.data() + bofs + (size_t)r * HID);
        src.push_back(pfx.data());
        std::vector<float> rf;
        std::vector<double> r64;
        vllm_ref(rf, src, d.score.data(), c.gamma ? d.gamma.data() : nullptr, eps, out_eps);
        ref64(r64, src, d.score.data(), c.gamma ? d.gamma.data() : nullptr, eps, out_eps);
        for (int i = 0; i < HID; ++i) {
            e_f32.add(probe_f32[pofs + i], rf[i]);
            e_probe64.add(probe_f32[pofs + i], r64[i]);
            e_f64.add(bf2f_h(obj_out[pofs + i]), r64[i]);
            e_bf16.add(bf2f_h(obj_out[pofs + i]), bf2f_h(f2bf_h(rf[i])));
            e_ctl.add(bf2f_h(ctl_out[pofs + i]), bf2f_h(f2bf_h(rf[i])));
            e_ctl_obj.add(bf2f_h(ctl_out[pofs + i]), bf2f_h(obj_out[pofs + i]));
        }
    }
    const bool ok = stable && probe_same && prefix_bad == 0 && push_bad == 0 &&
                    e_f32.rel_l2() <= 1e-6 && e_bf16.max_abs <= 0.0625;
    std::printf(
        "case T=%u nb=%u res=%d pre=%d push=%d gamma=%d grid=%u | f32 vs vllm-port relL2 %.3e "
        "maxabs %.3e | f32 vs f64 relL2 %.3e | bf16 vs bf16(port) relL2 %.3e maxabs %.6g exact "
        "%.4f%% | bf16 vs f64 relL2 %.3e | control(bf16-seam) vs port relL2 %.3e maxabs %.6g | "
        "control vs object relL2 %.3e | prefix_bad %zu push_bad %zu stable %d probe_same %d -> %s\n",
        c.t, c.nb, c.res, c.pre, c.push, c.gamma, grid, e_f32.rel_l2(), e_f32.max_abs,
        e_probe64.rel_l2(), e_bf16.rel_l2(), e_bf16.max_abs, 100.0 * e_bf16.exact / e_bf16.n,
        e_f64.rel_l2(), e_ctl.rel_l2(), e_ctl.max_abs, e_ctl_obj.rel_l2(), prefix_bad, push_bad,
        stable, probe_same, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static double time_ms(const std::function<void()>& f, unsigned warm, unsigned reps) {
    hipEvent_t beg, end;
    HC(hipEventCreate(&beg));
    HC(hipEventCreate(&end));
    for (unsigned i = 0; i < warm; ++i) f();
    HC(hipDeviceSynchronize());
    HC(hipEventRecord(beg));
    for (unsigned i = 0; i < reps; ++i) f();
    HC(hipEventRecord(end));
    HC(hipEventSynchronize(end));
    float ms = 0;
    HC(hipEventElapsedTime(&ms, beg, end));
    HC(hipEventDestroy(beg));
    HC(hipEventDestroy(end));
    return (double)ms / reps;
}

static void timing(Module& obj, const Dev& v, unsigned t, const std::vector<unsigned>& grids,
                   float eps, float out_eps) {
    const Case c{t, 8, true, false, false, true};
    const KernArgs a = args_for(v, c, false, eps, out_eps);
    const KernArgs actl = args_for(v, c, true, eps, out_eps);
    auto ctl = [&] {
        hipLaunchKernelGGL(control_kernel, std::min(256u, t), PLOW_THREADS, 0, 0, actl, 1u);
    };
    std::vector<double> c0;
    std::vector<std::vector<double>> c1(grids.size());
    for (unsigned fold = 0; fold < 4; ++fold) {
        if (fold & 1) {
            for (size_t g = 0; g < grids.size(); ++g)
                c1[g].push_back(time_ms([&] { obj.launch(a, std::min(grids[g], t)); }, 2, 10));
            c0.push_back(time_ms(ctl, 2, 10));
        } else {
            c0.push_back(time_ms(ctl, 2, 10));
            for (size_t g = 0; g < grids.size(); ++g)
                c1[g].push_back(time_ms([&] { obj.launch(a, std::min(grids[g], t)); }, 2, 10));
        }
    }
    auto mid = [](std::vector<double> x) {
        std::sort(x.begin(), x.end());
        return (x[1] + x[2]) / 2;
    };
    std::vector<double> cb;
    for (unsigned fold = 0; fold < 4; ++fold)
        cb.push_back(time_ms([&] {
            hipLaunchKernelGGL(bw_kernel, std::min(512u, t), AR_THREADS, 0, 0, a);
        }, 2, 10));
    const double tb = mid(cb);
    std::printf("timing T=%u streaming floor (same pattern, no barriers, grid 512): %.6f ms\n", t, tb);
    const double t0 = mid(c0);
    const double bytes = (double)t * HID * 2.0 * (8 + 1 + 2 + 1); /* 8 rows + prefix read, a+b read, out */
    std::printf("timing T=%u nb=8 res=1 gamma=1: interpreter body (256 WG x 512) %.6f ms\n", t, t0);
    for (size_t g = 0; g < grids.size(); ++g) {
        const double t1 = mid(c1[g]);
        std::printf("timing T=%u object grid=%u: %.6f ms  gain=%+.2f%%  ~%.2f TB/s HBM-equivalent\n",
                    t, std::min(grids[g], t), t1, (t0 - t1) / t0 * 100.0, bytes / t1 / 1e9);
    }
}

static std::vector<unsigned> parse_list(const char* s) {
    std::vector<unsigned> v;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || *p == 0) {
            if (!cur.empty()) v.push_back((unsigned)std::atoi(cur.c_str()));
            cur.clear();
            if (*p == 0) break;
        } else {
            cur += *p;
        }
    }
    return v;
}

int main(int argc, char** argv) {
    const char* elf = nullptr;
    unsigned t = 1024, grid = 1024;
    std::vector<unsigned> nbs = {0, 1, 4, 8}, grids = {256, 512, 1024, 2048};
    bool do_time = false;
    float scale = 1.0f;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--elf")) elf = argv[++i];
        else if (!std::strcmp(argv[i], "--t")) t = (unsigned)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--nb")) nbs = parse_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--grid")) grids = parse_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--launch-grid")) grid = (unsigned)std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--scale")) scale = (float)std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed")) rng = std::strtoull(argv[++i], nullptr, 0);
        else if (!std::strcmp(argv[i], "--time")) do_time = true;
        else if (!std::strcmp(argv[i], "--threads")) g_obj_threads = (unsigned)std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (!elf) {
        std::fprintf(stderr, "--elf <attn_res_f32mix_gfx950.elf> required\n");
        return 2;
    }
    constexpr float EPS = 1e-5f, OUT_EPS = 1e-5f;
    Module obj(elf);
    const Data d = make_data(t, scale);
    std::printf("contract hid=%d cap=%d score_eps=%.9g output_eps=%.9g output_norm_input=mixed-f32 "
                "order=vllm-0.28-block_l1-online wave=64 threads=%u\n",
                HID, CAP, EPS, OUT_EPS, AR_THREADS);
    std::printf("hash res_a=%016llx res_b=%016llx res_pre=%016llx prefix=%016llx ring=%016llx "
                "push=%016llx norm=%016llx proj=%016llx gamma=%016llx\n",
                fnv(d.res_a.data(), d.res_a.size() * 2), fnv(d.res_b.data(), d.res_b.size() * 2),
                fnv(d.res_pre.data(), d.res_pre.size() * 2), fnv(d.prefix.data(), d.prefix.size() * 2),
                fnv(d.ring.data(), d.ring.size() * 2), fnv(d.push_src.data(), d.push_src.size() * 2),
                fnv(d.norm.data(), HID * 2), fnv(d.proj.data(), HID * 2), fnv(d.gamma.data(), HID * 2));
    const Dev v = upload(d);
    int bad = 0;
    if (!do_time) {
        for (unsigned nb : nbs) {
            // The production shapes: MLP-side (res_a+res_b, fused post-norm), attention-side at a
            // snapshot layer (push, no materialization), plus res_pre and raw-mix variants.
            bad |= run_case(obj, d, v, Case{t, nb, true, false, false, true}, EPS, OUT_EPS, grid);
            bad |= run_case(obj, d, v, Case{t, nb, false, false, nb < CAP, true}, EPS, OUT_EPS, grid);
            bad |= run_case(obj, d, v, Case{t, nb, true, true, false, true}, EPS, OUT_EPS, grid);
            bad |= run_case(obj, d, v, Case{t, nb, false, false, false, false}, EPS, OUT_EPS, grid);
        }
    } else {
        timing(obj, v, t, grids, EPS, OUT_EPS);
    }
    std::puts(bad ? "ORACLE FAIL" : "ORACLE PASS");
    return bad ? 3 : 0;
}
