/* Exact K3 speculative-verifier journal gate: eight serial rows write eight candidate
 * recurrence/conv states while the committed bank remains byte-identical. */

#ifdef KDAJ_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda.h"

struct KArgs {
    unsigned long long o, q, k, v, g_raw, beta_raw, a_log, dt_bias, state, commit;
    unsigned long long cq_out, ck_out, cv_out, cq_in, ck_in, cv_in;
    unsigned long long wq, wk, wv, sq, sk, sv;
    unsigned T, H, D, BV, C, W;
    float lb, scale;
};

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_journal_state(KArgs* a) {
    __shared__ float lds[4 * 128 + 2 * PLOW_WAVES];
    d_kda_state_step_g(
        (bf16*)a->o, (const bf16*)a->q, (const bf16*)a->k, (const bf16*)a->v,
        (const bf16*)a->g_raw, (const bf16*)a->beta_raw, (const float*)a->a_log,
        (const float*)a->dt_bias, PLOW_KDA_GATE_LOWER_BOUND, a->lb, (float*)a->state, a->T,
        a->H, a->D, a->BV, PLOW_KDA_F_QK_L2NORM | PLOW_KDA_F_JOURNAL, a->scale,
        blockIdx.x, gridDim.x, lds, 0, nullptr, (const unsigned*)a->commit);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_journal_conv(KArgs* a) {
    d_kda_conv3((bf16*)a->cq_out, (bf16*)a->ck_out, (bf16*)a->cv_out,
                (const bf16*)a->cq_in, (const bf16*)a->ck_in, (const bf16*)a->cv_in,
                (const float*)a->wq, (const float*)a->wk, (const float*)a->wv,
                (float*)a->sq, (float*)a->sk, (float*)a->sv, a->T, a->C, a->W, 1,
                blockIdx.x, gridDim.x, 0, nullptr, (const unsigned*)a->commit);
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

typedef unsigned short bf16;

struct KArgs {
    unsigned long long o, q, k, v, g_raw, beta_raw, a_log, dt_bias, state, commit;
    unsigned long long cq_out, ck_out, cv_out, cq_in, ck_in, cv_in;
    unsigned long long wq, wk, wv, sq, sk, sv;
    unsigned T, H, D, BV, C, W;
    float lb, scale;
};

static float b2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static bf16 f2b(float f) {
    unsigned u;
    memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double silu(double x) { return x * sigmoid(x); }

#define HC(expr)                                                                                 \
    do {                                                                                         \
        hipError_t e_ = (expr);                                                                  \
        if (e_ != hipSuccess) {                                                                  \
            fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__);   \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

template <typename T> static void* upload(const std::vector<T>& src) {
    void* dst = nullptr;
    HC(hipMalloc(&dst, src.size() * sizeof(T)));
    HC(hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice));
    return dst;
}

static void launch(hipFunction_t fn, unsigned grid, KArgs& args) {
    void* d_args = nullptr;
    HC(hipMalloc(&d_args, sizeof args));
    HC(hipMemcpy(d_args, &args, sizeof args, hipMemcpyHostToDevice));
    void* argv[] = {&d_args};
    HC(hipModuleLaunchKernel(fn, grid, 1, 1, 512, 1, 1, 0, nullptr, argv, nullptr));
    HC(hipDeviceSynchronize());
    HC(hipFree(d_args));
}

static double rel_rms(const std::vector<float>& got, const std::vector<double>& want) {
    double se = 0.0, sw = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - want[i];
        se += d * d;
        sw += want[i] * want[i];
    }
    return sqrt(se / (sw + 1e-30));
}

static double rel_rms_bf16(const std::vector<bf16>& got, const std::vector<double>& want) {
    double se = 0.0, sw = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)b2f(got[i]) - want[i];
        se += d * d;
        sw += want[i] * want[i];
    }
    return sqrt(se / (sw + 1e-30));
}

static int state_gate(hipFunction_t fn) {
    constexpr unsigned T = 8, H = 12, D = 128, BV = 8, BANKS = 9, COMMIT = 7;
    const size_t hd = (size_t)H * D;
    const size_t bank_elems = hd * D;
    std::vector<bf16> q(T * hd), k(q.size()), v(q.size()), g(q.size()), beta(T * H);
    std::vector<float> alog(H), dt(H * D);
    srand(1701);
    for (auto& x : q) x = f2b(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
    for (auto& x : k) x = f2b(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
    for (auto& x : v) x = f2b(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
    for (auto& x : g) x = f2b(((float)rand() / RAND_MAX - 0.5f) * 4.0f);
    for (auto& x : beta) x = f2b(((float)rand() / RAND_MAX - 0.5f) * 4.0f);
    for (auto& x : alog) x = logf(0.5f + (float)rand() / RAND_MAX * 11.5f);
    for (auto& x : dt) x = -7.0f + (float)rand() / RAND_MAX * 6.0f;

    std::vector<float> banks(BANKS * bank_elems);
    for (unsigned b = 0; b < BANKS; ++b)
        for (size_t i = 0; i < bank_elems; ++i)
            banks[(size_t)b * bank_elems + i] =
                b == COMMIT ? ((float)rand() / RAND_MAX - 0.5f) * 0.5f : 1000.0f + b;
    const std::vector<float> before = banks;
    std::vector<double> current(bank_elems);
    for (size_t i = 0; i < bank_elems; ++i)
        current[i] = banks[(size_t)COMMIT * bank_elems + i];
    std::vector<std::vector<double>> snapshots(T, std::vector<double>(bank_elems));
    std::vector<double> oref(T * hd), qn(D), kn(D), decay(D);
    const double scale = 1.0 / sqrt(128.0);
    for (unsigned t = 0; t < T; ++t) {
        for (unsigned h = 0; h < H; ++h) {
            const size_t xbase = ((size_t)t * H + h) * D;
            double qs = 0.0, ks = 0.0;
            for (unsigned d = 0; d < D; ++d) {
                qn[d] = b2f(q[xbase + d]);
                kn[d] = b2f(k[xbase + d]);
                qs += qn[d] * qn[d];
                ks += kn[d] * kn[d];
            }
            const double rq = scale / sqrt(qs + 1e-6), rk = 1.0 / sqrt(ks + 1e-6);
            for (unsigned d = 0; d < D; ++d) {
                qn[d] *= rq;
                kn[d] *= rk;
                decay[d] = exp(-5.0 * sigmoid(exp((double)alog[h]) *
                                                ((double)b2f(g[xbase + d]) + dt[h * D + d])));
            }
            const double bt = sigmoid((double)b2f(beta[(size_t)t * H + h]));
            for (unsigned j = 0; j < D; ++j) {
                double* col = current.data() + ((size_t)h * D + j) * D;
                double pk = 0.0;
                for (unsigned d = 0; d < D; ++d) {
                    col[d] *= decay[d];
                    pk += col[d] * kn[d];
                }
                const double u = (double)b2f(v[xbase + j]) - pk;
                double pq = 0.0;
                for (unsigned d = 0; d < D; ++d) {
                    col[d] += bt * u * kn[d];
                    pq += col[d] * qn[d];
                }
                oref[xbase + j] = pq;
            }
        }
        snapshots[t] = current;
    }

    std::vector<unsigned> commit = {COMMIT};
    KArgs a{};
    a.q = (uint64_t)upload(q);
    a.k = (uint64_t)upload(k);
    a.v = (uint64_t)upload(v);
    a.g_raw = (uint64_t)upload(g);
    a.beta_raw = (uint64_t)upload(beta);
    a.a_log = (uint64_t)upload(alog);
    a.dt_bias = (uint64_t)upload(dt);
    a.state = (uint64_t)upload(banks);
    a.commit = (uint64_t)upload(commit);
    void* d_o = nullptr;
    HC(hipMalloc(&d_o, T * hd * sizeof(bf16)));
    HC(hipMemset(d_o, 0, T * hd * sizeof(bf16)));
    a.o = (uint64_t)d_o;
    a.T = T;
    a.H = H;
    a.D = D;
    a.BV = BV;
    a.lb = -5.0f;
    a.scale = (float)scale;
    launch(fn, H * D / BV, a);

    std::vector<float> got(banks.size());
    std::vector<bf16> og(T * hd);
    HC(hipMemcpy(got.data(), (void*)a.state, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    HC(hipMemcpy(og.data(), d_o, og.size() * sizeof(bf16), hipMemcpyDeviceToHost));
    const bool committed_same =
        memcmp(got.data() + (size_t)COMMIT * bank_elems,
               before.data() + (size_t)COMMIT * bank_elems, bank_elems * sizeof(float)) == 0;
    double state_worst = 0.0;
    for (unsigned t = 0; t < T; ++t) {
        const unsigned bank = (COMMIT + 1 + t) % BANKS;
        std::vector<float> slice(got.begin() + (size_t)bank * bank_elems,
                                 got.begin() + (size_t)(bank + 1) * bank_elems);
        state_worst = fmax(state_worst, rel_rms(slice, snapshots[t]));
    }
    const double out_rms = rel_rms_bf16(og, oref);
    printf("state: committed=%s candidate_rms=%.3e output_rms=%.3e\n",
           committed_same ? "exact" : "CHANGED", state_worst, out_rms);
    return !committed_same || state_worst >= 1e-4 || out_rms >= 1e-2;
}

static int conv_gate(hipFunction_t fn) {
    constexpr unsigned T = 8, C = 1536, W = 4, BANKS = 9, COMMIT = 8;
    const size_t bank_elems = (size_t)C * W;
    srand(2909);
    std::vector<bf16> input[3];
    std::vector<float> weight[3], state[3], before[3];
    std::vector<std::vector<double>> output_ref(3, std::vector<double>((size_t)T * C));
    std::vector<std::vector<std::vector<float>>> snapshots(
        3, std::vector<std::vector<float>>(T, std::vector<float>(bank_elems)));
    for (unsigned s = 0; s < 3; ++s) {
        input[s].resize((size_t)T * C);
        weight[s].resize(bank_elems);
        state[s].resize(BANKS * bank_elems);
        for (auto& x : input[s]) x = f2b(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
        for (auto& x : weight[s]) x = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        for (unsigned b = 0; b < BANKS; ++b)
            for (size_t i = 0; i < bank_elems; ++i)
                state[s][(size_t)b * bank_elems + i] =
                    b == COMMIT ? ((float)rand() / RAND_MAX - 0.5f) : 2000.0f + b;
        before[s] = state[s];
        std::vector<float> current(state[s].begin() + (size_t)COMMIT * bank_elems,
                                   state[s].begin() + (size_t)(COMMIT + 1) * bank_elems);
        for (unsigned t = 0; t < T; ++t) {
            for (unsigned c = 0; c < C; ++c) {
                float* win = current.data() + (size_t)c * W;
                for (unsigned j = 0; j + 1 < W; ++j) win[j] = win[j + 1];
                win[W - 1] = b2f(input[s][(size_t)t * C + c]);
                double y = 0.0;
                for (unsigned j = 0; j < W; ++j)
                    y += (double)win[j] * weight[s][(size_t)c * W + j];
                output_ref[s][(size_t)t * C + c] = silu(y);
            }
            snapshots[s][t] = current;
        }
    }

    std::vector<unsigned> commit = {COMMIT};
    KArgs a{};
    a.commit = (uint64_t)upload(commit);
    a.cq_in = (uint64_t)upload(input[0]);
    a.ck_in = (uint64_t)upload(input[1]);
    a.cv_in = (uint64_t)upload(input[2]);
    a.wq = (uint64_t)upload(weight[0]);
    a.wk = (uint64_t)upload(weight[1]);
    a.wv = (uint64_t)upload(weight[2]);
    a.sq = (uint64_t)upload(state[0]);
    a.sk = (uint64_t)upload(state[1]);
    a.sv = (uint64_t)upload(state[2]);
    void* d_out[3] = {};
    for (unsigned s = 0; s < 3; ++s) {
        HC(hipMalloc(&d_out[s], (size_t)T * C * sizeof(bf16)));
        HC(hipMemset(d_out[s], 0, (size_t)T * C * sizeof(bf16)));
    }
    a.cq_out = (uint64_t)d_out[0];
    a.ck_out = (uint64_t)d_out[1];
    a.cv_out = (uint64_t)d_out[2];
    a.T = T;
    a.C = C;
    a.W = W;
    launch(fn, 304, a);

    double output_worst = 0.0;
    bool committed_same = true, candidates_exact = true;
    const uint64_t state_ptr[3] = {a.sq, a.sk, a.sv};
    for (unsigned s = 0; s < 3; ++s) {
        std::vector<float> got(state[s].size());
        std::vector<bf16> out((size_t)T * C);
        HC(hipMemcpy(got.data(), (void*)state_ptr[s], got.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
        HC(hipMemcpy(out.data(), d_out[s], out.size() * sizeof(bf16), hipMemcpyDeviceToHost));
        committed_same &=
            memcmp(got.data() + (size_t)COMMIT * bank_elems,
                   before[s].data() + (size_t)COMMIT * bank_elems,
                   bank_elems * sizeof(float)) == 0;
        for (unsigned t = 0; t < T; ++t) {
            const unsigned bank = (COMMIT + 1 + t) % BANKS;
            candidates_exact &=
                memcmp(got.data() + (size_t)bank * bank_elems, snapshots[s][t].data(),
                       bank_elems * sizeof(float)) == 0;
        }
        output_worst = fmax(output_worst, rel_rms_bf16(out, output_ref[s]));
    }
    printf("conv:  committed=%s candidates=%s output_rms=%.3e\n",
           committed_same ? "exact" : "CHANGED", candidates_exact ? "exact" : "MISMATCH",
           output_worst);
    return !committed_same || !candidates_exact || output_worst >= 1e-2;
}

int main(int argc, char** argv) {
    hipDeviceProp_t prop{};
    HC(hipGetDeviceProperties(&prop, 0));
    printf("device: %s\n", prop.gcnArchName);
    hipModule_t module;
    HC(hipModuleLoad(&module, argc > 1 ? argv[1] : "/tmp/kda_journal_gfx942.elf"));
    hipFunction_t state_fn, conv_fn;
    HC(hipModuleGetFunction(&state_fn, module, "k_journal_state"));
    HC(hipModuleGetFunction(&conv_fn, module, "k_journal_conv"));
    const int failed = state_gate(state_fn) | conv_gate(conv_fn);
    printf("%s\n", failed ? "FAIL" : "PASS");
    return failed;
}
#endif
