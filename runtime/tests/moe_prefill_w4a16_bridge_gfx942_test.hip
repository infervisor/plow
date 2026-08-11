/* Standalone gfx942 W4A16 control for K3's grouped GLU -> DOWN bridge.
 *
 * This does not add a packet arm. It keeps the production K3 geometry and data contract:
 * top-16 routing over 896 experts, BM=64 per-expert padding, packed MXFP4+E8M0 Wg/Wu/Wd,
 * SiTUv2, sorted bridge rows, row_partidx scatter, and row_gate. The experimental difference is
 * only the bridge: GLU stores BF16 and DOWN consumes BF16 instead of requantizing it to MXFP4.
 *
 * CMake builds this file twice: MPW4A16_DEVICE emits the standalone gfx942 code object; the host
 * half loads it and checks every padding sentinel plus sampled GLU and DOWN outputs against f64
 * dots over the decoded weight values. The scalar device dots make this a correctness control,
 * not a performance baseline.
 */

#ifdef MPW4A16_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"

struct W4A16Args {
    unsigned long long Cout, Ain, wtab, stab, row_exp, row_token, row_partidx, row_gate;
    unsigned N, K, R, mode;
};

__device__ __forceinline__ float w4a16_fp4(const unsigned char* w, const unsigned char* s,
                                           unsigned K, unsigned row, unsigned k) {
    const unsigned char b = w[(size_t)row * (K >> 1) + (k >> 1)];
    const unsigned nib = (k & 1u) ? (b >> 4) : (b & 15u);
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = (nib & 8u) ? -mag[nib & 7u] : mag[nib & 7u];
    return v * e8m0_to_f32(s[(size_t)row * (K >> 5) + (k >> 5)]);
}

extern "C" __global__ __launch_bounds__(256, 1) void k_w4a16(W4A16Args* a) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)a->R * a->N;
    if ((size_t)idx >= total) return;
    const unsigned row = idx / a->N, n = idx - row * a->N;
    const unsigned* const rtok = (const unsigned*)(size_t)a->row_token;
    if (rtok[row] == 0xFFFFFFFFu) return;
    const unsigned e = ((const unsigned*)(size_t)a->row_exp)[row];
    const unsigned long long* const wt = (const unsigned long long*)(size_t)a->wtab;
    const unsigned long long* const st = (const unsigned long long*)(size_t)a->stab;
    if (a->mode == 0u) {
        const bf16* const x = (const bf16*)(size_t)a->Ain + (size_t)rtok[row] * a->K;
        const unsigned char* const wg = (const unsigned char*)(size_t)wt[(size_t)e * 3];
        const unsigned char* const wu = (const unsigned char*)(size_t)wt[(size_t)e * 3 + 1];
        const unsigned char* const sg = (const unsigned char*)(size_t)st[(size_t)e * 3];
        const unsigned char* const su = (const unsigned char*)(size_t)st[(size_t)e * 3 + 1];
        float g = 0.0f, u = 0.0f;
        for (unsigned k = 0; k < a->K; k++) {
            const float xv = bf2f(x[k]);
            g = fmaf(xv, w4a16_fp4(wg, sg, a->K, n, k), g);
            u = fmaf(xv, w4a16_fp4(wu, su, a->K, n, k), u);
        }
        const float y = k3_situ_gate(g, 4.0f) * k3_situ_up(u, 25.0f);
        ((bf16*)(size_t)a->Cout)[(size_t)row * a->N + n] = f2bf(y);
        return;
    }
    const unsigned pidx = ((const unsigned*)(size_t)a->row_partidx)[row];
    if (pidx == 0xFFFFFFFFu) return;
    const bf16* const bridge = (const bf16*)(size_t)a->Ain + (size_t)row * a->K;
    const unsigned char* const wd = (const unsigned char*)(size_t)wt[(size_t)e * 3 + 2];
    const unsigned char* const sd = (const unsigned char*)(size_t)st[(size_t)e * 3 + 2];
    float acc = 0.0f;
    for (unsigned k = 0; k < a->K; k++)
        acc = fmaf(bf2f(bridge[k]), w4a16_fp4(wd, sd, a->K, n, k), acc);
    ((float*)(size_t)a->Cout)[(size_t)pidx * a->N + n] =
        ((const float*)(size_t)a->row_gate)[row] * acc;
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

typedef unsigned short bf16;
static constexpr unsigned UNUSED_ROW = 0xFFFFFFFFu;
static constexpr bf16 BRIDGE_SENTINEL = 0x7FC1u;

struct W4A16Args {
    unsigned long long Cout, Ain, wtab, stab, row_exp, row_token, row_partidx, row_gate;
    unsigned N, K, R, mode;
};

#define HC(x)                                                                                  \
    do {                                                                                       \
        hipError_t e_ = (x);                                                                   \
        if (e_ != hipSuccess) {                                                                \
            fprintf(stderr, "HIP: %s at line %d\n", hipGetErrorString(e_), __LINE__);        \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

static float b2f(bf16 b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static bf16 f2b(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (bf16)((u >> 16) | 0x40u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

static double fp4_value(unsigned nib) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (nib & 8u) ? -mag[nib & 7u] : mag[nib & 7u];
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

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <moe_prefill_w4a16_bridge_gfx942.elf>\n", argv[0]);
        return 2;
    }

    hipDeviceProp_t prop;
    HC(hipGetDeviceProperties(&prop, 0));
    if (strncmp(prop.gcnArchName, "gfx942", 6) != 0) {
        fprintf(stderr, "REFUSED: this control is gfx942-only, found %s\n", prop.gcnArchName);
        return 2;
    }

    constexpr unsigned H = 3584, IM = 384, NEXP = 896, NTOK = 4, BM = 64;
    static const unsigned selected[] = {
        187, 228, 280, 346, 460, 608, 632, 647,
        712, 713, 759, 769, 800, 819, 829, 864,
    };

    std::vector<unsigned> real(NEXP, 0), pad(NEXP, 0), row_exp;
    for (unsigned e : selected) real[e] = 4;
    unsigned R = 0;
    for (unsigned e = 0; e < NEXP; e++) {
        pad[e] = real[e] ? BM : 0;
        R += pad[e];
    }
    row_exp.resize(R);
    std::vector<unsigned> rtok(R, UNUSED_ROW), rpix(R, UNUSED_ROW), live_rows;
    std::vector<float> rgate(R, 0.0f);

    unsigned np = 0, base = 0;
    for (unsigned e = 0; e < NEXP; e++) {
        for (unsigned r = 0; r < pad[e]; r++) row_exp[base + r] = e;
        for (unsigned r = 0; r < real[e]; r++) {
            const unsigned row = base + r;
            rtok[row] = r % NTOK;
            rpix[row] = np++;
            rgate[row] = ((int)(rnd() % 17u) - 8) / 16.0f;
            live_rows.push_back(row);
        }
        base += pad[e];
    }

    std::vector<bf16> x((size_t)NTOK * H);
    for (bf16& v : x) v = f2b(((int)(rnd() % 33u) - 16) / 32.0f);

    std::vector<std::vector<unsigned char>> weights((size_t)NEXP * 3), scales((size_t)NEXP * 3);
    auto make_weight = [&](unsigned e, unsigned which, unsigned rows, unsigned cols) {
        std::vector<unsigned char>& w = weights[(size_t)e * 3 + which];
        std::vector<unsigned char>& s = scales[(size_t)e * 3 + which];
        w.resize((size_t)rows * cols / 2);
        s.resize((size_t)rows * cols / 32);
        for (unsigned char& b : w) b = (unsigned char)rnd();
        for (unsigned char& b : s) b = (unsigned char)(118u + rnd() % 5u);
        if (rows && cols >= 16)
            for (unsigned z = 0; z < 8; z++)
                w[z] = (unsigned char)((2u * z + 1u) << 4 | (2u * z));
    };
    for (unsigned e : selected) {
        make_weight(e, 0, IM, H);
        make_weight(e, 1, IM, H);
        make_weight(e, 2, H, IM);
    }

    auto weight_at = [&](unsigned e, unsigned which, unsigned K, unsigned row,
                         unsigned k) -> double {
        const std::vector<unsigned char>& w = weights[(size_t)e * 3 + which];
        const std::vector<unsigned char>& s = scales[(size_t)e * 3 + which];
        const unsigned char b = w[(size_t)row * (K / 2) + (k >> 1)];
        const unsigned nib = (k & 1u) ? (b >> 4) : (b & 15u);
        return fp4_value(nib) * ldexp(1.0, (int)s[(size_t)row * (K / 32) + (k >> 5)] - 127);
    };

    std::vector<void*> allocations;
    auto upload = [&](const void* src, size_t bytes) {
        void* d = nullptr;
        HC(hipMalloc(&d, bytes));
        HC(hipMemcpy(d, src, bytes, hipMemcpyHostToDevice));
        allocations.push_back(d);
        return d;
    };

    std::vector<unsigned long long> wtab((size_t)NEXP * 3, 0), stab((size_t)NEXP * 3, 0);
    for (unsigned e : selected)
        for (unsigned which = 0; which < 3; which++) {
            const size_t i = (size_t)e * 3 + which;
            wtab[i] = (unsigned long long)(size_t)upload(weights[i].data(), weights[i].size());
            stab[i] = (unsigned long long)(size_t)upload(scales[i].data(), scales[i].size());
        }

    void* d_x = upload(x.data(), x.size() * sizeof(bf16));
    void* d_wtab = upload(wtab.data(), wtab.size() * sizeof(wtab[0]));
    void* d_stab = upload(stab.data(), stab.size() * sizeof(stab[0]));
    void* d_row_exp = upload(row_exp.data(), row_exp.size() * sizeof(row_exp[0]));
    void* d_rtok = upload(rtok.data(), rtok.size() * sizeof(rtok[0]));
    void* d_rpix = upload(rpix.data(), rpix.size() * sizeof(rpix[0]));
    void* d_rgate = upload(rgate.data(), rgate.size() * sizeof(rgate[0]));

    std::vector<bf16> bridge((size_t)R * IM, BRIDGE_SENTINEL);
    std::vector<float> part((size_t)np * H, NAN);
    void* d_bridge = upload(bridge.data(), bridge.size() * sizeof(bridge[0]));
    void* d_part = upload(part.data(), part.size() * sizeof(part[0]));

    hipModule_t module;
    hipFunction_t kernel;
    HC(hipModuleLoad(&module, argv[1]));
    HC(hipModuleGetFunction(&kernel, module, "k_w4a16"));

    W4A16Args args{};
    args.wtab = (unsigned long long)(size_t)d_wtab;
    args.stab = (unsigned long long)(size_t)d_stab;
    args.row_exp = (unsigned long long)(size_t)d_row_exp;
    args.row_token = (unsigned long long)(size_t)d_rtok;
    args.row_partidx = (unsigned long long)(size_t)d_rpix;
    args.row_gate = (unsigned long long)(size_t)d_rgate;
    args.R = R;
    void* d_args = nullptr;
    HC(hipMalloc(&d_args, sizeof args));
    allocations.push_back(d_args);
    void* launch_args[] = {&d_args};
    auto launch = [&](unsigned mode, void* out, void* in, unsigned N, unsigned K) {
        args.mode = mode;
        args.Cout = (unsigned long long)(size_t)out;
        args.Ain = (unsigned long long)(size_t)in;
        args.N = N;
        args.K = K;
        HC(hipMemcpy(d_args, &args, sizeof args, hipMemcpyHostToDevice));
        const unsigned blocks = (unsigned)(((size_t)R * N + 255) / 256);
        HC(hipModuleLaunchKernel(kernel, blocks, 1, 1, 256, 1, 1, 0, nullptr, launch_args,
                                 nullptr));
        HC(hipDeviceSynchronize());
    };

    launch(0, d_bridge, d_x, IM, H);
    HC(hipMemcpy(bridge.data(), d_bridge, bridge.size() * sizeof(bridge[0]),
                 hipMemcpyDeviceToHost));

    size_t live_bad = 0, pad_bad = 0;
    for (unsigned row = 0; row < R; row++)
        for (unsigned n = 0; n < IM; n++) {
            const bf16 v = bridge[(size_t)row * IM + n];
            if (rtok[row] == UNUSED_ROW) pad_bad += v != BRIDGE_SENTINEL;
            else live_bad += v == BRIDGE_SENTINEL || !isfinite(b2f(v));
        }

    constexpr unsigned NS = 256;
    double glu_se = 0.0, glu_sw = 0.0;
    for (unsigned q = 0; q < NS; q++) {
        const unsigned row = live_rows[q % live_rows.size()];
        const unsigned n = (q * 131u + 17u) % IM;
        const unsigned e = row_exp[row];
        double g = 0.0, u = 0.0;
        for (unsigned k = 0; k < H; k++) {
            const double xv = b2f(x[(size_t)rtok[row] * H + k]);
            g += xv * weight_at(e, 0, H, n, k);
            u += xv * weight_at(e, 1, H, n, k);
        }
        const double want = situ(g, u);
        const double got = b2f(bridge[(size_t)row * IM + n]);
        glu_se += (got - want) * (got - want);
        glu_sw += want * want;
    }
    const double glu_rms = sqrt(glu_se / (glu_sw + 1e-30));

    launch(1, d_part, d_bridge, H, IM);
    HC(hipMemcpy(part.data(), d_part, part.size() * sizeof(part[0]), hipMemcpyDeviceToHost));
    const size_t part_bad = std::count_if(part.begin(), part.end(),
                                          [](float v) { return !isfinite(v); });

    double down_se = 0.0, down_sw = 0.0;
    for (unsigned q = 0; q < NS; q++) {
        const unsigned row = live_rows[q % live_rows.size()];
        const unsigned h = (q * 977u + 29u) % H;
        const unsigned e = row_exp[row];
        double acc = 0.0;
        for (unsigned k = 0; k < IM; k++)
            acc += (double)b2f(bridge[(size_t)row * IM + k]) * weight_at(e, 2, IM, h, k);
        const double want = (double)rgate[row] * acc;
        const double got = part[(size_t)rpix[row] * H + h];
        down_se += (got - want) * (got - want);
        down_sw += want * want;
    }
    const double down_rms = sqrt(down_se / (down_sw + 1e-30));

    printf("K3 W4A16 grouped bridge control: H=%u I=%u experts=%u selected=%zu rows=%u "
           "live=%u BM=%u\n",
           H, IM, NEXP, sizeof(selected) / sizeof(selected[0]), R, np, BM);
    printf("  GLU BF16 bridge vs 256 f64 SiTU dots: rms %.3e; live sentinel/finite failures %zu\n",
           glu_rms, live_bad);
    printf("  padding sentinel failures: %zu / %zu elements\n", pad_bad,
           ((size_t)R - np) * IM);
    printf("  DOWN vs 256 f64 dots over device BF16 bridge: rms %.3e; unwritten/nonfinite %zu\n",
           down_rms, part_bad);

    const bool pass = live_bad == 0 && pad_bad == 0 && part_bad == 0 && glu_rms < 2e-2 &&
                      down_rms < 2e-3;
    printf("%s\n", pass ? "PASS" : "FAIL");

    HC(hipModuleUnload(module));
    for (void* d : allocations) HC(hipFree(d));
    return pass ? 0 : 1;
}
#endif
