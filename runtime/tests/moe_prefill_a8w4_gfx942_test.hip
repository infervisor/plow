/* Standalone gfx942 grouped A8W4 experiment.
 *
 * This deliberately does not add a packet opcode or change the production interpreter. Packed
 * MXFP4 weights stay in their checkpoint layout. Staging expands only the e2m1 mantissa to OCP
 * e4m3; the E8M0 scale is promoted into a second f32 accumulator every K=32. Activations are OCP
 * e4m3 with one f32 scale per row. gfx942's matrix core reads e4m3FNUZ, so 0x80 is normalized to
 * zero and the promotion carries the exact x4 correction for the two half-valued operands.
 *
 * Default: two unequal experts, H=512, I=384, 120 live / 192 padded rows, full f64 oracle.
 * MPA8W4_K3=1: K3 TP8 dimensions H=3584, I=384, 896 experts, the existing B4/top16 expert set.
 */

#ifdef MPA8W4_DEVICE

#include <hip/hip_runtime.h>
#include "amd_common.h"

#if !defined(__gfx942__)
#error "the A8W4 experiment is gfx942-only"
#endif

#define A8W4_BM 64u
#define A8W4_BN 256u
#define A8W4_BK 64u
#define A8W4_WM 2u
#define A8W4_WN 4u
#define A8W4_A_BYTES (A8W4_BM * A8W4_BK)
#define A8W4_B_BYTES (A8W4_BN * A8W4_BK)
#define A8W4_LDS_BYTES (A8W4_A_BYTES + A8W4_B_BYTES)
#define A8W4_SWZ(row, off) ((off) ^ (((row) & 7u) << 3))
#define A8W4_UNUSED 0xffffffffu

struct A8W4Args {
    unsigned long long out, aq, ascale, wtab, stab, meta, row_token, row_partidx, row_gate;
    unsigned N, K, n_exp;
};

__device__ __forceinline__ unsigned char fp4_mantissa_to_ocp(unsigned n) {
    const unsigned mag = n & 7u;
    unsigned v;
    switch (mag) {
        case 0: v = 0x00u; break;
        case 1: v = 0x30u; break;
        case 2: v = 0x38u; break;
        case 3: v = 0x3cu; break;
        case 4: v = 0x40u; break;
        case 5: v = 0x44u; break;
        case 6: v = 0x48u; break;
        default: v = 0x4cu; break;
    }
    return (unsigned char)(mag == 0u ? 0u : v | ((n & 8u) << 4));
}

__device__ __forceinline__ unsigned long long mask_ocp_negzero(unsigned long long v) {
    union {
        unsigned long long q;
        unsigned w[2];
    } u{v};
    u.w[0] = plow_fp8_mask_neg0(u.w[0]);
    u.w[1] = plow_fp8_mask_neg0(u.w[1]);
    return u.q;
}

__device__ __forceinline__ unsigned expert_of_tile(const int* tilep, unsigned mt,
                                                    unsigned n_exp) {
    unsigned e = 0;
    while (e + 1u < n_exp && (unsigned)tilep[e + 1] <= mt) ++e;
    return e;
}

template <bool GLU>
__device__ void grouped_a8w4(A8W4Args* a) {
    __shared__ __align__(16) unsigned char lds[A8W4_LDS_BYTES];
    unsigned char* const als = lds;
    unsigned char* const bls = lds + A8W4_A_BYTES;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned wave = tid >> 6;
    const unsigned wm = wave / A8W4_WN;
    const unsigned wn = wave % A8W4_WN;
    const unsigned frow = mfma_frag_row(lane);
    const unsigned NB = GLU ? A8W4_BN / 2u : A8W4_BN;
    const int* const rowoff = (const int*)(size_t)a->meta;
    const int* const tilep = rowoff + 2u * a->n_exp;
    const unsigned total_tiles = (unsigned)tilep[a->n_exp];
    const unsigned tnc = (a->N + NB - 1u) / NB;
    const unsigned ntiles = total_tiles * tnc;
    const unsigned nt = (a->K + A8W4_BK - 1u) / A8W4_BK;
    const unsigned ks = a->K >> 1;
    const unsigned ksc = a->K >> 5;
    const unsigned char* const aq = (const unsigned char*)(size_t)a->aq;
    const float* const ascale = (const float*)(size_t)a->ascale;
    const unsigned long long* const wtab = (const unsigned long long*)(size_t)a->wtab;
    const unsigned long long* const stab = (const unsigned long long*)(size_t)a->stab;
    const unsigned* const row_token = (const unsigned*)(size_t)a->row_token;
    const unsigned* const row_partidx = (const unsigned*)(size_t)a->row_partidx;
    const float* const row_gate = (const float*)(size_t)a->row_gate;

    for (unsigned lin = blockIdx.x; lin < ntiles; lin += gridDim.x) {
        const unsigned mt = lin / tnc;
        const unsigned ntile = lin % tnc;
        const unsigned e = expert_of_tile(tilep, mt, a->n_exp);
        const unsigned rowbase = (unsigned)rowoff[e] + (mt - (unsigned)tilep[e]) * A8W4_BM;
        const unsigned n0 = ntile * NB;
        const unsigned long long w0b = wtab[(size_t)e * 3u + (GLU ? 0u : 2u)];
        if (!w0b) continue;
        const unsigned char* const w0 = (const unsigned char*)(size_t)w0b;
        const unsigned char* const w1 =
            GLU ? (const unsigned char*)(size_t)wtab[(size_t)e * 3u + 1u] : nullptr;
        const unsigned char* const s0 =
            (const unsigned char*)(size_t)stab[(size_t)e * 3u + (GLU ? 0u : 2u)];
        const unsigned char* const s1 =
            GLU ? (const unsigned char*)(size_t)stab[(size_t)e * 3u + 1u] : nullptr;

        f32x16 sum[2] = {(f32x16)(0.0f), (f32x16)(0.0f)};
        for (unsigned kt = 0; kt < nt; ++kt) {
            const unsigned k0 = kt * A8W4_BK;

            /* A: exactly 8 fp8 values per thread. */
            const unsigned ae = tid * 8u;
            const unsigned ar = ae / A8W4_BK;
            const unsigned ak = ae % A8W4_BK;
            const unsigned src = GLU ? row_token[rowbase + ar] : rowbase + ar;
            unsigned long long av = 0;
            if (src != A8W4_UNUSED && k0 + ak < a->K)
                __builtin_memcpy(&av, aq + (size_t)src * a->K + k0 + ak, 8);
            av = mask_ocp_negzero(av);
            *(unsigned long long*)(als + ar * A8W4_BK + A8W4_SWZ(ar, ak)) = av;

            /* B: one packed 32-value MX block per thread, expanded exactly to 32 fp8 bytes. */
            const unsigned br = tid / 2u;
            const unsigned bb = tid & 1u;
            const bool up = GLU && br >= A8W4_BN / 2u;
            const unsigned bn = n0 + (up ? br - A8W4_BN / 2u : br);
            unsigned char packed[16] = {};
            if (bn < a->N) {
                const unsigned char* const wp = up ? w1 : w0;
                __builtin_memcpy(packed, wp + (size_t)bn * ks + (k0 >> 1) + bb * 16u, 16);
            }
            unsigned char expanded[32];
#pragma unroll
            for (unsigned z = 0; z < 16u; ++z) {
                expanded[z * 2u] = fp4_mantissa_to_ocp(packed[z] & 15u);
                expanded[z * 2u + 1u] = fp4_mantissa_to_ocp(packed[z] >> 4);
            }
#pragma unroll
            for (unsigned z = 0; z < 4u; ++z) {
                unsigned long long q;
                __builtin_memcpy(&q, expanded + z * 8u, 8);
                *(unsigned long long*)(bls + br * A8W4_BK +
                                      A8W4_SWZ(br, bb * 32u + z * 8u)) = q;
            }
            __syncthreads();

            /* Two K16 MFMAs form one K32 scale block. Scale promotion cannot cross this edge. */
#pragma unroll
            for (unsigned kb = 0; kb < 2u; ++kb) {
                f32x16 raw[2] = {(f32x16)(0.0f), (f32x16)(0.0f)};
#pragma unroll
                for (unsigned q = 0; q < 2u; ++q) {
                    const unsigned koff = kb * 32u + q * 16u + 8u * (lane / 32u);
                    const unsigned arow = wm * 32u + frow;
                    const unsigned long long af = *(const unsigned long long*)(
                        als + arow * A8W4_BK + A8W4_SWZ(arow, koff));
#pragma unroll
                    for (unsigned j = 0; j < 2u; ++j) {
                        const unsigned brow =
                            (GLU ? j * 128u + wn * 32u : wn * 64u + j * 32u) + frow;
                        const unsigned long long bf = *(const unsigned long long*)(
                            bls + brow * A8W4_BK + A8W4_SWZ(brow, koff));
                        raw[j] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                            (long)af, (long)bf, raw[j], 0, 0, 0);
                    }
                }
#pragma unroll
                for (unsigned j = 0; j < 2u; ++j) {
                    const unsigned nn = n0 + (GLU ? wn * 32u : wn * 64u + j * 32u) +
                                        mfma_acc_n(lane);
                    if (nn < a->N) {
                        const unsigned char* const sp = (GLU && j == 1u) ? s1 : s0;
                        const float scale = e8m0_to_f32(
                            sp[(size_t)nn * ksc + (k0 >> 5) + kb]) * PLOW_FP8_MMA_FIX;
                        sum[j] += raw[j] * scale;
                    }
                }
            }
            __syncthreads();
        }

        if constexpr (GLU) {
            bf16* const out = (bf16*)(size_t)a->out;
            const unsigned nn = n0 + wn * 32u + mfma_acc_n(lane);
            if (nn < a->N) {
#pragma unroll
                for (unsigned el = 0; el < 16u; ++el) {
                    const unsigned rr = wm * 32u + mfma_acc_m(lane, el);
                    const unsigned tok = row_token[rowbase + rr];
                    const float rs = tok == A8W4_UNUSED ? 0.0f : ascale[tok];
                    const float g = sum[0][el] * rs;
                    const float u = sum[1][el] * rs;
                    const float y = 4.0f * fast_tanhf(g * 0.25f) * k3_sigmoid(g) *
                                    25.0f * fast_tanhf(u * 0.04f);
                    st_act1(&out[(size_t)(rowbase + rr) * a->N + nn], f2bf(y));
                }
            }
        } else {
            float* const out = (float*)(size_t)a->out;
#pragma unroll
            for (unsigned j = 0; j < 2u; ++j) {
                const unsigned nn = n0 + wn * 64u + j * 32u + mfma_acc_n(lane);
                if (nn >= a->N) continue;
#pragma unroll
                for (unsigned el = 0; el < 16u; ++el) {
                    const unsigned rr = wm * 32u + mfma_acc_m(lane, el);
                    const unsigned pidx = row_partidx[rowbase + rr];
                    if (pidx != A8W4_UNUSED)
                        out[(size_t)pidx * a->N + nn] = row_gate[rowbase + rr] * sum[j][el] *
                                                       ascale[rowbase + rr];
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(512, 2) void a8w4_group_glu(A8W4Args* a) {
    grouped_a8w4<true>(a);
}

extern "C" __global__ __launch_bounds__(512, 2) void a8w4_group_down(A8W4Args* a) {
    grouped_a8w4<false>(a);
}

#else

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using bf16 = unsigned short;
static constexpr unsigned UNUSED = 0xffffffffu;

struct A8W4Args {
    unsigned long long out, aq, ascale, wtab, stab, meta, row_token, row_partidx, row_gate;
    unsigned N, K, n_exp;
};

static float b2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static bf16 f2b(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static double fp4(unsigned n) {
    static const double v[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (n & 8u) ? -v[n & 7u] : v[n & 7u];
}
static double e4m3(unsigned char b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    double v;
    if (e == 0) v = ldexp((double)m, -9);
    else if (e == 15 && m == 7) v = NAN;
    else v = ldexp(1.0 + m / 8.0, e - 7);
    return s ? -v : v;
}
static unsigned char f2e4m3(float x) {
    if (x == 0.0f) return 0;
    const unsigned sign = std::signbit(x) ? 0x80u : 0u;
    const double a = fabs((double)x);
    unsigned best = 0;
    double bd = a;
    for (unsigned c = 1; c <= 126; ++c) {
        const double d = fabs(a - e4m3((unsigned char)c));
        if (d < bd || (d == bd && !(c & 1u))) {
            best = c;
            bd = d;
        }
    }
    return (unsigned char)(sign | best);
}
static void quant_row(const bf16* x, unsigned char* q, float* scale, unsigned n) {
    float amax = 0.0f;
    for (unsigned i = 0; i < n; ++i) amax = fmaxf(amax, fabsf(b2f(x[i])));
    const float s = fmaxf(amax / 448.0f, 1e-12f);
    *scale = s;
    for (unsigned i = 0; i < n; ++i) q[i] = f2e4m3(b2f(x[i]) / s);
}
static double situ(double g, double u) {
    return 4.0 * tanh(g / 4.0) / (1.0 + exp(-g)) * 25.0 * tanh(u / 25.0);
}
static unsigned long long rng_state = 0x6a09e667f3bcc909ull;
static unsigned rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (unsigned)(rng_state >> 16);
}

#define HC(x)                                                                                  \
    do {                                                                                       \
        hipError_t e_ = (x);                                                                   \
        if (e_ != hipSuccess) {                                                                \
            fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

struct Expert {
    std::vector<unsigned char> wg, sg, wu, su, wd, sd;
};

static void make_weight(unsigned rows, unsigned cols, std::vector<unsigned char>& w,
                        std::vector<unsigned char>& s) {
    w.resize((size_t)rows * cols / 2u);
    s.resize((size_t)rows * cols / 32u);
    for (auto& x : w) x = (unsigned char)rnd();
    for (auto& x : s) x = (unsigned char)(118u + rnd() % 5u);
    if (rows && cols >= 16) {
        for (unsigned z = 0; z < 8; ++z) w[z] = (unsigned char)((2u * z + 1u) << 4 | (2u * z));
    }
}

static double wat(const std::vector<unsigned char>& w, const std::vector<unsigned char>& s,
                  unsigned cols, unsigned r, unsigned k) {
    const unsigned char b = w[(size_t)r * (cols / 2u) + (k >> 1)];
    const unsigned n = (k & 1u) ? b >> 4 : b & 15u;
    return fp4(n) * ldexp(1.0, (int)s[(size_t)r * (cols / 32u) + (k >> 5)] - 127);
}

static void* upload(const void* src, size_t bytes) {
    void* p = nullptr;
    if (hipMalloc(&p, bytes) != hipSuccess) return nullptr;
    if (src && hipMemcpy(p, src, bytes, hipMemcpyHostToDevice) != hipSuccess) return nullptr;
    return p;
}

static void hip_check_abort(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        fprintf(stderr, "HIP %s: %s\n", what, hipGetErrorString(e));
        abort();
    }
}

static double median_ms(hipFunction_t f, void* darg, unsigned grid) {
    void* args[] = {&darg};
    for (int i = 0; i < 20; ++i)
        hip_check_abort(hipModuleLaunchKernel(f, grid, 1, 1, 512, 1, 1, 0, nullptr, args,
                                              nullptr), "warmup launch");
    hip_check_abort(hipDeviceSynchronize(), "warmup sync");
    std::vector<double> samples(12);
    hipEvent_t a, b;
    hip_check_abort(hipEventCreate(&a), "create start event");
    hip_check_abort(hipEventCreate(&b), "create stop event");
    for (double& s : samples) {
        hip_check_abort(hipEventRecord(a), "record start");
        for (int i = 0; i < 4; ++i)
            hip_check_abort(hipModuleLaunchKernel(f, grid, 1, 1, 512, 1, 1, 0, nullptr,
                                                  args, nullptr), "timed launch");
        hip_check_abort(hipEventRecord(b), "record stop");
        hip_check_abort(hipEventSynchronize(b), "timed sync");
        float ms = 0;
        hip_check_abort(hipEventElapsedTime(&ms, a, b), "elapsed time");
        s = ms / 4.0;
    }
    hip_check_abort(hipEventDestroy(a), "destroy start event");
    hip_check_abort(hipEventDestroy(b), "destroy stop event");
    std::sort(samples.begin(), samples.end());
    return 0.5 * (samples[5] + samples[6]);
}

int main(int argc, char** argv) {
    const bool k3 = getenv("MPA8W4_K3") != nullptr;
    const unsigned H = k3 ? 3584u : 512u;
    const unsigned IM = 384u;
    const unsigned E = k3 ? 896u : 2u;
    const unsigned NTOK = k3 ? 4u : 8u;
    const unsigned GRID = k3 ? 304u : 5u;
    const unsigned selected_k3[] = {187, 228, 280, 346, 460, 608, 632, 647,
                                    712, 713, 759, 769, 800, 819, 829, 864};
    std::vector<unsigned> live(E, 0), pad(E, 0);
    for (unsigned e = 0; e < E; ++e) {
        bool selected = !k3;
        if (k3)
            selected = std::find(std::begin(selected_k3), std::end(selected_k3), e) !=
                       std::end(selected_k3);
        live[e] = k3 ? (selected ? 4u : 0u) : (e == 0 ? 70u : 50u);
        pad[e] = live[e] ? (live[e] + 63u) / 64u * 64u : 0u;
    }
    unsigned R = 0, nmt = 0;
    std::vector<int> meta(3u * E + 1u, 0);
    int* tilep = meta.data() + 2u * E;
    for (unsigned e = 0; e < E; ++e) {
        meta[e] = (int)R;
        meta[E + e] = (int)live[e];
        tilep[e] = (int)nmt;
        R += pad[e];
        nmt += pad[e] / 64u;
    }
    tilep[E] = (int)nmt;
    unsigned np = 0;
    std::vector<unsigned> rtok(R, UNUSED), rpix(R, UNUSED);
    std::vector<float> rgate(R, 0.0f);
    for (unsigned e = 0; e < E; ++e) {
        const unsigned base = (unsigned)meta[e];
        for (unsigned r = 0; r < live[e]; ++r) {
            rtok[base + r] = r % NTOK;
            rpix[base + r] = np++;
            rgate[base + r] = ((int)(rnd() % 17u) - 8) / 16.0f;
        }
    }

    std::vector<bf16> x((size_t)NTOK * H);
    for (auto& v : x) v = f2b(((int)(rnd() % 33u) - 16) / 32.0f);
    std::vector<unsigned char> xq((size_t)NTOK * H);
    std::vector<float> xs(NTOK);
    for (unsigned t = 0; t < NTOK; ++t) quant_row(&x[(size_t)t * H], &xq[(size_t)t * H], &xs[t], H);
    xq[0] = 0x80u; /* OCP -0: gfx942 FNUZ would read NaN unless staging normalizes it. */

    std::vector<Expert> ex(E);
    std::vector<unsigned long long> wtab((size_t)E * 3u, 0), stab((size_t)E * 3u, 0);
    for (unsigned e = 0; e < E; ++e) {
        if (!live[e]) continue;
        make_weight(IM, H, ex[e].wg, ex[e].sg);
        make_weight(IM, H, ex[e].wu, ex[e].su);
        make_weight(H, IM, ex[e].wd, ex[e].sd);
        wtab[(size_t)e * 3u] = (unsigned long long)(size_t)upload(ex[e].wg.data(), ex[e].wg.size());
        wtab[(size_t)e * 3u + 1u] =
            (unsigned long long)(size_t)upload(ex[e].wu.data(), ex[e].wu.size());
        wtab[(size_t)e * 3u + 2u] =
            (unsigned long long)(size_t)upload(ex[e].wd.data(), ex[e].wd.size());
        stab[(size_t)e * 3u] = (unsigned long long)(size_t)upload(ex[e].sg.data(), ex[e].sg.size());
        stab[(size_t)e * 3u + 1u] =
            (unsigned long long)(size_t)upload(ex[e].su.data(), ex[e].su.size());
        stab[(size_t)e * 3u + 2u] =
            (unsigned long long)(size_t)upload(ex[e].sd.data(), ex[e].sd.size());
    }

    hipDeviceProp_t prop{};
    HC(hipGetDeviceProperties(&prop, 0));
    if (strstr(prop.gcnArchName, "gfx942") == nullptr) {
        fprintf(stderr, "REFUSED: %s is not gfx942\n", prop.gcnArchName);
        return 2;
    }
    const char* object = argc > 1 ? argv[1] : "moe_prefill_a8w4_gfx942.elf";
    hipModule_t mod;
    HC(hipModuleLoad(&mod, object));
    hipFunction_t fglu, fdown;
    HC(hipModuleGetFunction(&fglu, mod, "a8w4_group_glu"));
    HC(hipModuleGetFunction(&fdown, mod, "a8w4_group_down"));

    void* dmeta = upload(meta.data(), meta.size() * sizeof(int));
    void* drtok = upload(rtok.data(), rtok.size() * sizeof(unsigned));
    void* drpix = upload(rpix.data(), rpix.size() * sizeof(unsigned));
    void* drgate = upload(rgate.data(), rgate.size() * sizeof(float));
    void* dwtab = upload(wtab.data(), wtab.size() * sizeof(unsigned long long));
    void* dstab = upload(stab.data(), stab.size() * sizeof(unsigned long long));
    void* dxq = upload(xq.data(), xq.size());
    void* dxs = upload(xs.data(), xs.size() * sizeof(float));
    void* dfu = nullptr;
    HC(hipMalloc(&dfu, (size_t)R * IM * sizeof(bf16)));
    HC(hipMemset(dfu, 0xee, (size_t)R * IM * sizeof(bf16)));

    A8W4Args ka{};
    ka.out = (unsigned long long)(size_t)dfu;
    ka.aq = (unsigned long long)(size_t)dxq;
    ka.ascale = (unsigned long long)(size_t)dxs;
    ka.wtab = (unsigned long long)(size_t)dwtab;
    ka.stab = (unsigned long long)(size_t)dstab;
    ka.meta = (unsigned long long)(size_t)dmeta;
    ka.row_token = (unsigned long long)(size_t)drtok;
    ka.row_partidx = (unsigned long long)(size_t)drpix;
    ka.row_gate = (unsigned long long)(size_t)drgate;
    ka.N = IM;
    ka.K = H;
    ka.n_exp = E;
    void* dka = upload(&ka, sizeof ka);
    void* args[] = {&dka};
    HC(hipModuleLaunchKernel(fglu, GRID, 1, 1, 512, 1, 1, 0, nullptr, args, nullptr));
    HC(hipDeviceSynchronize());
    std::vector<bf16> fu((size_t)R * IM);
    HC(hipMemcpy(fu.data(), dfu, fu.size() * sizeof(bf16), hipMemcpyDeviceToHost));

    double ge = 0.0, gw = 0.0, gmax = 0.0;
    for (unsigned e = 0; e < E; ++e) {
        const unsigned base = (unsigned)meta[e];
        for (unsigned r = 0; r < live[e]; ++r) {
            const unsigned row = base + r;
            const unsigned tok = rtok[row];
            for (unsigned n = 0; n < IM; ++n) {
                double g = 0.0, u = 0.0;
                for (unsigned kk = 0; kk < H; ++kk) {
                    const double av = e4m3(xq[(size_t)tok * H + kk]) * xs[tok];
                    g += av * wat(ex[e].wg, ex[e].sg, H, n, kk);
                    u += av * wat(ex[e].wu, ex[e].su, H, n, kk);
                }
                const double want = situ(g, u);
                const double got = b2f(fu[(size_t)row * IM + n]);
                ge += (got - want) * (got - want);
                gw += want * want;
                gmax = fmax(gmax, fabs(got - want) / (fabs(want) + 1e-4));
            }
        }
    }
    const double grms = sqrt(ge / (gw + 1e-30));

    std::vector<unsigned char> fuq((size_t)R * IM);
    std::vector<float> fus(R);
    for (unsigned r = 0; r < R; ++r) quant_row(&fu[(size_t)r * IM], &fuq[(size_t)r * IM], &fus[r], IM);
    void* dfuq = upload(fuq.data(), fuq.size());
    void* dfus = upload(fus.data(), fus.size() * sizeof(float));
    void* dpart = nullptr;
    HC(hipMalloc(&dpart, (size_t)np * H * sizeof(float)));
    HC(hipMemset(dpart, 0, (size_t)np * H * sizeof(float)));
    ka.out = (unsigned long long)(size_t)dpart;
    ka.aq = (unsigned long long)(size_t)dfuq;
    ka.ascale = (unsigned long long)(size_t)dfus;
    ka.N = H;
    ka.K = IM;
    HC(hipMemcpy(dka, &ka, sizeof ka, hipMemcpyHostToDevice));
    HC(hipModuleLaunchKernel(fdown, GRID, 1, 1, 512, 1, 1, 0, nullptr, args, nullptr));
    HC(hipDeviceSynchronize());
    std::vector<float> part((size_t)np * H);
    HC(hipMemcpy(part.data(), dpart, part.size() * sizeof(float), hipMemcpyDeviceToHost));

    double de = 0.0, dw = 0.0, dmax = 0.0;
    for (unsigned e = 0; e < E; ++e) {
        const unsigned base = (unsigned)meta[e];
        for (unsigned r = 0; r < live[e]; ++r) {
            const unsigned row = base + r;
            for (unsigned n = 0; n < H; ++n) {
                double y = 0.0;
                for (unsigned kk = 0; kk < IM; ++kk)
                    y += e4m3(fuq[(size_t)row * IM + kk]) * fus[row] *
                         wat(ex[e].wd, ex[e].sd, IM, n, kk);
                const double want = rgate[row] * y;
                const double got = part[(size_t)rpix[row] * H + n];
                de += (got - want) * (got - want);
                dw += want * want;
                dmax = fmax(dmax, fabs(got - want) / (fabs(want) + 1e-4));
            }
        }
    }
    const double drms = sqrt(de / (dw + 1e-30));

    ka.out = (unsigned long long)(size_t)dfu;
    ka.aq = (unsigned long long)(size_t)dxq;
    ka.ascale = (unsigned long long)(size_t)dxs;
    ka.N = IM;
    ka.K = H;
    HC(hipMemcpy(dka, &ka, sizeof ka, hipMemcpyHostToDevice));
    const double glu_ms_fixed = median_ms(fglu, dka, GRID);
    ka.out = (unsigned long long)(size_t)dpart;
    ka.aq = (unsigned long long)(size_t)dfuq;
    ka.ascale = (unsigned long long)(size_t)dfus;
    ka.N = H;
    ka.K = IM;
    HC(hipMemcpy(dka, &ka, sizeof ka, hipMemcpyHostToDevice));
    const double down_ms = median_ms(fdown, dka, GRID);
    int gv = 0, gl = 0, gs = 0, dv = 0, dl = 0, ds = 0;
    HC(hipFuncGetAttribute(&gv, HIP_FUNC_ATTRIBUTE_NUM_REGS, fglu));
    HC(hipFuncGetAttribute(&gl, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fglu));
    HC(hipFuncGetAttribute(&gs, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fglu));
    HC(hipFuncGetAttribute(&dv, HIP_FUNC_ATTRIBUTE_NUM_REGS, fdown));
    HC(hipFuncGetAttribute(&dl, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fdown));
    HC(hipFuncGetAttribute(&ds, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fdown));
    int runtime_version = 0;
    HC(hipRuntimeGetVersion(&runtime_version));
    const char* toolchain = getenv("PLOW_TOOLCHAIN_LABEL");
    const char* build_id = getenv("PLOW_BUILD_ID");
    const char* lease = getenv("PLOW_LEASE_LABEL");
    printf("shape: %s H=%u I=%u E=%u live=%u padded=%u grid=%u\n", k3 ? "k3-b4" : "unequal",
           H, IM, E, np, R, GRID);
    printf("GLU  %.4f ms  rms %.3e worst %.3e  regs=%d lds=%d local=%d\n",
           glu_ms_fixed, grms, gmax, gv, gs, gl);
    printf("DOWN %.4f ms  rms %.3e worst %.3e  regs=%d lds=%d local=%d\n",
           down_ms, drms, dmax, dv, ds, dl);

    const char* jsonl = getenv("MPA8W4_JSONL");
    if (jsonl) {
        FILE* f = fopen(jsonl, "a");
        if (!f) {
            perror(jsonl);
            return 1;
        }
        fprintf(f,
                "{\"experiment\":\"grouped-a8w4-fp8-mfma\",\"arch\":\"gfx942\","
                "\"gcn_arch\":\"%s\",\"gpu\":\"%s\",\"hip_runtime\":%d,"
                "\"toolchain\":\"%s\",\"build_id\":\"%s\",\"lease\":\"%s\","
                "\"source\":\"runtime/tests/moe_prefill_a8w4_gfx942_test.hip\","
                "\"compile\":\"hipcc --genco --offload-arch=gfx942 -O3\","
                "\"object\":\"%s\",\"fixture_seed\":\"0x6a09e667f3bcc909\","
                "\"shape\":\"%s\",\"h\":%u,\"i_tp8\":%u,\"experts\":%u,"
                "\"live_rows\":%u,\"padded_rows\":%u,\"grid\":%u,"
                "\"tile\":\"64x256x64\",\"mfma\":\"v_mfma_f32_32x32x16_fp8_fp8\","
                "\"mfma_per_k64\":8,\"timing_samples\":12,\"timing_repeats\":4,"
                "\"oracle\":\"f64-ocp-e4m3-k32-e8m0\","
                "\"glu_ms\":%.6f,\"down_ms\":%.6f,\"glu_rms\":%.9g,"
                "\"down_rms\":%.9g,\"glu_vgpr\":%d,\"down_vgpr\":%d,"
                "\"glu_lds\":%d,\"down_lds\":%d,\"glu_scratch\":%d,"
                "\"down_scratch\":%d,\"ocp_negzero_injected\":true,"
                "\"k32_scale_promotion\":true}\n",
                prop.gcnArchName, prop.name, runtime_version, toolchain ? toolchain : "",
                build_id ? build_id : "", lease ? lease : "", object,
                k3 ? "k3-b4" : "unequal", H, IM, E, np, R, GRID, glu_ms_fixed, down_ms, grms,
                drms, gv, dv, gs, ds, gl, dl);
        fclose(f);
    }

    const bool pass = grms < 8e-3 && drms < 2e-3 && gl == 0 && dl == 0 && gv <= 256 && dv <= 256;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

#endif
