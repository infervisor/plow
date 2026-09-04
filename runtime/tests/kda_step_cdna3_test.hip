/* kda_step_cdna3_test.hip — the KDA recurrence (ops 102/112) and the gated output norm (op 103)
 * against an f64 host reference, at Kimi-K3's exact geometry, on whatever part runs it.
 *                                                                              [K3-KDA-CDNA3]
 * WHY THIS EXISTS when kda_block_gfx950_test already gates the layer: that gate needs the REAL
 * checkpoint (its fixture is written by kda_real_oracle.py from safetensors), and the K3 snapshot
 * is 1.4 TiB that most boxes — including the gfx942 box this was written on — do not have. The
 * KDA arms are pure VALU and compile everywhere, but "compiles on gfx942" and "computes §2.2 of
 * docs/kimi-k3-kda.md on gfx942" are different claims. This closes the gap with synthetic
 * weights: the recurrence has no learned structure to miss — any full-rank input distinguishes a
 * correct state walk from a transposed, un-decayed, or mis-gated one.
 *
 * WHAT IS PINNED for the recurrence and norm, at H=96 D=128 (the only KDA shape that exists):
 *   - op 112 (`d_kda_state_step_g`), the PRODUCTION form: bounded gate lb*sigmoid(exp(A_log)*
 *     (g+dt_bias)) inlined, beta = sigmoid(raw), qk L2 norm with eps INSIDE the sqrt, q scaled
 *     by D^-1/2, decay -> delta -> rank-1 write -> read of the UPDATED state, V-FIRST [h][v][k]
 *     state layout. The reference implements the spec, not the kernel, so a shared mistake would
 *     have to be made twice independently.
 *   - PREFILL mode (bstride=0, BV=8): T=5 tokens threading serially through ONE state.
 *   - BATCHED-DECODE mode (bstride=H*D*D): 3 rows, each its OWN state — the axis fold whose
 *     absence ran batch-16 decode serially (see the kernel's header).
 *   - op 103 (`d_kda_gated_norm`): RMSNorm over D inside a head, one shared [D] weight, sigmoid
 *     on the RAW gate, gate applied AFTER the norm.
 *   - the standalone BT64 gate-prefix primitive at reduced H=3, D=128: full and ragged chunks,
 *     per-dimension prefix, chunk-local reset, both gate modes, and beta.
 *   - the standalone BC16 diagonal primitive at reduced H=2, D=128: native MFMA QK/KK products,
 *     strict causal masks, f32 triangular solve, and 64/17/1/63-token ragged chunks.
 *   - the full BT64 intra primitive on those same chunks: all six off-diagonal BC16 couplings and
 *     the merged 64x64 inverse, without any cross-workgroup dependency.
 *   - the W/U transform and ordered state/output carry across all four chunks, compared directly
 *     with an f64 serial recurrence from the same bf16 q/k/v and f32 initial state.
 *
 * Two passes, op_kda.h is device-only by include chain:
 *   hipcc --offload-arch=gfx942 -O3 -w -DKDA3_DEVICE --genco \
 *       runtime/tests/kda_step_cdna3_test.hip -o /tmp/kda3.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/kda_step_cdna3_test.hip -o /tmp/kda3 -L/opt/rocm/lib -lamdhip64
 *   perf-data/tools/gpulease -n 1 kda3 '/tmp/kda3 /tmp/kda3.co'
 */

#ifdef KDA3_DEVICE
#define PLOW_PACKED_PREFILL_CONSUMERS 1
#ifndef KDA_CHUNK_QPRE
#define KDA_CHUNK_QPRE 0
#endif
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_elementwise.h" /* act_silu, used by op_kda's conv */
#include "op_kda.h"

struct KArgs {
    unsigned long long o, q, k, v, g_raw, beta_raw, a_log, dt_bias, state, y, norm_w, gate2;
    unsigned long long chunks, gprefix, aqk, ainv, w, u;
    unsigned T, H, D, BV, flags, gate_mode;
    unsigned n_chunks, V;
    float lb, scale, eps;
    unsigned long long bstride;
    unsigned long long spans, parked;
    unsigned nspans, nrows;
};

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_step(KArgs* a) {
    __shared__ float lds[4 * 128 + 2 * PLOW_WAVES];
    d_kda_state_step_g((bf16*)a->o, (const bf16*)a->q, (const bf16*)a->k, (const bf16*)a->v,
                       (const bf16*)a->g_raw, (const bf16*)a->beta_raw, (const float*)a->a_log,
                       (const float*)a->dt_bias, a->gate_mode, a->lb, (float*)a->state, a->T, a->H,
                       a->D, a->BV, a->flags, a->scale, blockIdx.x, gridDim.x, lds,
                       (size_t)a->bstride, nullptr);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_step_packed(KArgs* a) {
    __shared__ float lds[4 * 128 + 2 * PLOW_WAVES];
    PlowProgram prog{};
    prog.prefill_spans = (const PlowPrefillSpan*)a->spans;
    prog.prefill_parked = (const uint32_t*)a->parked;
    prog.n_prefill_spans = a->nspans;
    prog.n_prefill_rows = a->nrows;
    d_kda_state_step_packed<true>(
        (bf16*)a->o, (const bf16*)a->q, (const bf16*)a->k, (const bf16*)a->v, nullptr,
        nullptr, (const bf16*)a->g_raw, (const bf16*)a->beta_raw, (const float*)a->a_log,
        (const float*)a->dt_bias, a->gate_mode, a->lb, (float*)a->state, a->H, a->D, a->BV,
        a->flags, a->scale, blockIdx.x, gridDim.x, lds, &prog);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_norm(KArgs* a) {
    d_kda_gated_norm((bf16*)a->y, (const bf16*)a->o, (const float*)a->norm_w,
                     (const bf16*)a->gate2, a->T, a->H, a->D, a->eps, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_prefix(KArgs* a) {
    d_kda_chunk_gate_prefix_bt64(
        (float*)a->gprefix, (float*)a->o, (const bf16*)a->g_raw,
        (const bf16*)a->beta_raw, (const float*)a->a_log, (const float*)a->dt_bias,
        (const uint2*)a->chunks, a->n_chunks, a->T, a->H, a->D, a->gate_mode, a->lb, blockIdx.x,
        gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_prepare(KArgs* a) {
    d_kda_chunk_prepare_bt64(
        (bf16*)a->q, (bf16*)a->k, (float*)a->gprefix, (float*)a->o,
        (const bf16*)a->g_raw, (const bf16*)a->beta_raw, (const float*)a->a_log,
        (const float*)a->dt_bias, (const uint2*)a->chunks, a->n_chunks, a->T, a->H, a->D,
        a->gate_mode, a->lb, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_intra(KArgs* a) {
    __shared__ float lds[PLOW_WAVES * 16 * 16];
    d_kda_chunk_intra_bc16(
        (float*)a->aqk, (float*)a->ainv, (const bf16*)a->q, (const bf16*)a->k,
        (const float*)a->gprefix, (const float*)a->o, (const uint2*)a->chunks, a->n_chunks,
        a->T, a->H, a->D, a->scale, blockIdx.x, gridDim.x, lds);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_intra_full(KArgs* a) {
    __shared__ float mat[64 * 64];
    d_kda_chunk_intra_bt64(
        (float*)a->aqk, (float*)a->ainv, (const bf16*)a->q, (const bf16*)a->k,
        (const float*)a->gprefix, (const float*)a->o, (const uint2*)a->chunks, a->n_chunks,
        a->T, a->H, a->D, a->scale, blockIdx.x, gridDim.x, mat);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 1) void k_intra_cached(KArgs* a) {
    __shared__ __align__(16) bf16 scratch[57344];
    d_kda_chunk_intra_bt64_cached(
        (float*)a->aqk, (float*)a->ainv, (const bf16*)a->q, (const bf16*)a->k,
        (const float*)a->gprefix, (const float*)a->o, (const uint2*)a->chunks, a->n_chunks,
        a->T, a->H, a->D, a->scale, blockIdx.x, gridDim.x, scratch);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_wu(KArgs* a) {
    d_kda_chunk_wu_bt64(
        (bf16*)a->w, (bf16*)a->u, KDA_CHUNK_QPRE ? (bf16*)a->q : nullptr,
        (const float*)a->ainv, (const bf16*)a->k, (const bf16*)a->v,
        (const float*)a->gprefix, (const float*)a->o, (const uint2*)a->chunks,
        a->n_chunks, a->T, a->H, a->D, a->V, a->scale, KDA_CHUNK_QPRE, blockIdx.x,
        gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_carry(KArgs* a) {
    __shared__ float st[16 * 128];
    __shared__ bf16 vsm[64 * 16];
    __shared__ float osm[64 * 16];
    d_kda_chunk_carry_bt64<KDA_CHUNK_QPRE>(
        (bf16*)a->y, (float*)a->state, (const bf16*)a->q, (const bf16*)a->k,
        (const bf16*)a->w, (const bf16*)a->u, (const float*)a->aqk,
        (const float*)a->gprefix, (const uint2*)a->chunks, a->n_chunks, a->T, a->H, a->D, a->V,
        a->scale, blockIdx.x, gridDim.x, st, vsm, osm);
}
static __device__ PlowProgram packed_prog(const KArgs* a) {
    PlowProgram prog{};
    prog.prefill_spans = (const PlowPrefillSpan*)a->spans;
    prog.prefill_parked = (const uint32_t*)a->parked;
    prog.n_prefill_spans = a->nspans;
    prog.n_prefill_rows = a->nrows;
    return prog;
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_prepare_packed(KArgs* a) {
    PlowProgram prog = packed_prog(a);
    d_kda_chunk_prepare_packed_bt64(
        (bf16*)a->q, (bf16*)a->k, (float*)a->gprefix, (float*)a->o,
        (const bf16*)a->g_raw, (const bf16*)a->beta_raw, (const float*)a->a_log,
        (const float*)a->dt_bias, a->H, a->D, a->gate_mode, a->lb, blockIdx.x, gridDim.x,
        &prog);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_intra_packed(KArgs* a) {
    __shared__ float mat[64 * 64];
    PlowProgram prog = packed_prog(a);
    d_kda_chunk_intra_packed_bt64(
        (float*)a->aqk, (float*)a->ainv, (const bf16*)a->q, (const bf16*)a->k,
        (const float*)a->gprefix, (const float*)a->o, a->H, a->D, a->scale, blockIdx.x,
        gridDim.x, mat, &prog);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_wu_packed(KArgs* a) {
    PlowProgram prog = packed_prog(a);
    d_kda_chunk_wu_packed_bt64(
        (bf16*)a->w, (bf16*)a->u, KDA_CHUNK_QPRE ? (bf16*)a->q : nullptr,
        (const float*)a->ainv, (const bf16*)a->k, (const bf16*)a->v,
        (const float*)a->gprefix, (const float*)a->o, a->H, a->D, a->V, a->scale,
        KDA_CHUNK_QPRE, blockIdx.x, gridDim.x, &prog);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_carry_packed(KArgs* a) {
    __shared__ float st[16 * 128];
    __shared__ bf16 vsm[64 * 16];
    __shared__ float osm[64 * 16];
    PlowProgram prog = packed_prog(a);
    d_kda_chunk_carry_packed_bt64<KDA_CHUNK_QPRE>(
        (bf16*)a->y, (float*)a->state, (const bf16*)a->q, (const bf16*)a->k,
        (const bf16*)a->w, (const bf16*)a->u, (const float*)a->aqk,
        (const float*)a->gprefix, a->H, a->D, a->V, a->scale, blockIdx.x, gridDim.x, st,
        vsm, osm, &prog);
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifndef KDA_BENCH_THREADS
#define KDA_BENCH_THREADS 512
#endif

typedef unsigned short bf16;
enum { PLOW_KDA_GATE_SOFTPLUS = 0, PLOW_KDA_GATE_LOWER_BOUND = 1 };
static float b2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2b(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }
#define HC(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return 1; }        \
    } while (0)
#define HC2(x) if (hipError_t e_ = (x)) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); }

struct KArgs {
    unsigned long long o, q, k, v, g_raw, beta_raw, a_log, dt_bias, state, y, norm_w, gate2;
    unsigned long long chunks, gprefix, aqk, ainv, w, u;
    unsigned T, H, D, BV, flags, gate_mode;
    unsigned n_chunks, V;
    float lb, scale, eps;
    unsigned long long bstride;
    unsigned long long spans, parked;
    unsigned nspans, nrows;
};

struct PrefillSpan {
    unsigned row0, n_rows, slot, flags, kv_row0, kv_len, state_slot, program;
};

/* §2.2 of docs/kimi-k3-kda.md, in f64, from the bf16-rounded inputs the device sees. */
static void ref_step(unsigned T, unsigned H, unsigned D, const std::vector<bf16>& q,
                     const std::vector<bf16>& k, const std::vector<bf16>& v,
                     const std::vector<bf16>& graw, const std::vector<bf16>& braw,
                     const std::vector<float>& alog, const std::vector<float>& dtb, double lb,
                     double scale, std::vector<double>& S /* [rows?][H][V][K] */,
                     std::vector<double>& o, bool rows_independent) {
    std::vector<double> qn(D), kn(D), a(D);
    for (unsigned t = 0; t < T; t++) {
        const unsigned srow = rows_independent ? t : 0;
        for (unsigned h = 0; h < H; h++) {
            const size_t hd = (size_t)t * H * D + (size_t)h * D;
            double qs = 0, ks = 0;
            for (unsigned d = 0; d < D; d++) {
                qn[d] = b2f(q[hd + d]); ks += (double)b2f(k[hd + d]) * b2f(k[hd + d]);
                qs += qn[d] * qn[d]; kn[d] = b2f(k[hd + d]);
            }
            const double rq = scale / sqrt(qs + 1e-6), rk = 1.0 / sqrt(ks + 1e-6);
            for (unsigned d = 0; d < D; d++) { qn[d] *= rq; kn[d] *= rk; }
            const double ah = exp((double)alog[h]);
            for (unsigned d = 0; d < D; d++)
                a[d] = exp(lb * sigm(ah * ((double)b2f(graw[hd + d]) + (double)dtb[h * D + d])));
            const double beta = sigm((double)b2f(braw[(size_t)t * H + h]));
            double* Sh = S.data() + ((size_t)srow * H + h) * D * D; /* V-FIRST [v][k] */
            for (unsigned j = 0; j < D; j++) {
                double* col = Sh + (size_t)j * D;
                double pk = 0;
                for (unsigned d = 0; d < D; d++) { col[d] *= a[d]; pk += col[d] * kn[d]; }
                const double u = (double)b2f(v[hd + j]) - pk;
                double pq = 0;
                for (unsigned d = 0; d < D; d++) {
                    col[d] += beta * u * kn[d];
                    pq += col[d] * qn[d];
                }
                o[hd + j] = pq;
            }
        }
    }
}

int main(int argc, char** argv) {
    hipDeviceProp_t p; HC(hipGetDeviceProperties(&p, 0));
    printf("dev: %s\n", p.gcnArchName);
    hipModule_t M; HC(hipModuleLoad(&M, argc > 1 ? argv[1] : "/tmp/kda3.co"));
    hipFunction_t fstep, fstep_packed, fnorm, fprefix, fprepare, fintra, fintra_full,
                  fintra_cached, fwu, fcarry;
    hipFunction_t fprepare_packed, fintra_packed, fwu_packed, fcarry_packed;
    HC(hipModuleGetFunction(&fstep, M, "k_step"));
    HC(hipModuleGetFunction(&fstep_packed, M, "k_step_packed"));
    HC(hipModuleGetFunction(&fnorm, M, "k_norm"));
    HC(hipModuleGetFunction(&fprefix, M, "k_prefix"));
    HC(hipModuleGetFunction(&fprepare, M, "k_prepare"));
    HC(hipModuleGetFunction(&fintra, M, "k_intra"));
    HC(hipModuleGetFunction(&fintra_full, M, "k_intra_full"));
    HC(hipModuleGetFunction(&fintra_cached, M, "k_intra_cached"));
    HC(hipModuleGetFunction(&fwu, M, "k_wu"));
    HC(hipModuleGetFunction(&fcarry, M, "k_carry"));
    HC(hipModuleGetFunction(&fprepare_packed, M, "k_prepare_packed"));
    HC(hipModuleGetFunction(&fintra_packed, M, "k_intra_packed"));
    HC(hipModuleGetFunction(&fwu_packed, M, "k_wu_packed"));
    HC(hipModuleGetFunction(&fcarry_packed, M, "k_carry_packed"));

    const unsigned H = 96, D = 128, GRID = 37;
    srand(1337);

    auto run_prefix = [&](unsigned mode) -> double {
        const unsigned PT = 139, PH = 3, PD = 128;
        std::vector<uint2> chunks{make_uint2(0, 64), make_uint2(64, 1), make_uint2(65, 7),
                                  make_uint2(72, 64), make_uint2(136, 3)};
        std::vector<bf16> graw((size_t)PT * PH * PD), braw((size_t)PT * PH);
        std::vector<float> alog(PH), dtb((size_t)PH * PD);
        for (auto& x : graw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        for (auto& x : braw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        for (auto& x : alog) x = logf(.5f + (float)rand() / RAND_MAX * 11.5f);
        for (auto& x : dtb) x = -7.f + (float)rand() / RAND_MAX * 6.f;

        std::vector<double> gref(graw.size()), bref(braw.size());
        const double rcp_ln2 = 1.4426950408889634074;
        for (const uint2 c : chunks)
            for (unsigned h = 0; h < PH; h++) {
                bref[(size_t)c.x * PH + h] = sigm((double)b2f(braw[(size_t)c.x * PH + h]));
                for (unsigned d = 0; d < PD; d++) {
                    double acc = 0.0;
                    const double a = exp((double)alog[h]);
                    for (unsigned tl = 0; tl < c.y; tl++) {
                        const size_t row = (size_t)c.x + tl;
                        const size_t i = (row * PH + h) * PD + d;
                        const double s = (double)b2f(graw[i]) + (double)dtb[(size_t)h * PD + d];
                        const double gate = mode == PLOW_KDA_GATE_LOWER_BOUND
                                                ? -5.0 * sigm(a * s)
                                                : -a * log1p(exp(s));
                        acc += gate * rcp_ln2;
                        gref[i] = acc;
                    }
                }
                for (unsigned tl = 1; tl < c.y; tl++) {
                    const size_t row = (size_t)c.x + tl;
                    bref[row * PH + h] = sigm((double)b2f(braw[row * PH + h]));
                }
            }

        void *d_graw, *d_braw, *d_alog, *d_dtb, *d_chunks, *d_gprefix, *d_beta, *d_a;
        HC(hipMalloc(&d_graw, graw.size() * sizeof(bf16)));
        HC(hipMalloc(&d_braw, braw.size() * sizeof(bf16)));
        HC(hipMalloc(&d_alog, alog.size() * sizeof(float)));
        HC(hipMalloc(&d_dtb, dtb.size() * sizeof(float)));
        HC(hipMalloc(&d_chunks, chunks.size() * sizeof(uint2)));
        HC(hipMalloc(&d_gprefix, graw.size() * sizeof(float)));
        HC(hipMalloc(&d_beta, braw.size() * sizeof(float)));
        HC(hipMemcpy(d_graw, graw.data(), graw.size() * sizeof(bf16), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_braw, braw.data(), braw.size() * sizeof(bf16), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_alog, alog.data(), alog.size() * sizeof(float), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_dtb, dtb.data(), dtb.size() * sizeof(float), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_chunks, chunks.data(), chunks.size() * sizeof(uint2), hipMemcpyHostToDevice));

        KArgs a{};
        a.o = (unsigned long long)(size_t)d_beta;
        a.g_raw = (unsigned long long)(size_t)d_graw;
        a.beta_raw = (unsigned long long)(size_t)d_braw;
        a.a_log = (unsigned long long)(size_t)d_alog;
        a.dt_bias = (unsigned long long)(size_t)d_dtb;
        a.chunks = (unsigned long long)(size_t)d_chunks;
        a.gprefix = (unsigned long long)(size_t)d_gprefix;
        a.H = PH; a.D = PD; a.gate_mode = mode; a.n_chunks = chunks.size(); a.lb = -5.0f;
        HC(hipMalloc(&d_a, sizeof a));
        HC(hipMemcpy(d_a, &a, sizeof a, hipMemcpyHostToDevice));
        void* args[] = {&d_a};
        HC(hipModuleLaunchKernel(fprefix, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());

        std::vector<float> gout(graw.size()), bout(braw.size());
        HC(hipMemcpy(gout.data(), d_gprefix, gout.size() * sizeof(float), hipMemcpyDeviceToHost));
        HC(hipMemcpy(bout.data(), d_beta, bout.size() * sizeof(float), hipMemcpyDeviceToHost));
        double max_g = 0.0, max_b = 0.0;
        for (size_t i = 0; i < gout.size(); i++) max_g = fmax(max_g, fabs(gout[i] - gref[i]));
        for (size_t i = 0; i < bout.size(); i++) max_b = fmax(max_b, fabs(bout[i] - bref[i]));
        printf("  chunk gate %-7s max abs %.3e (bar 2e-4) beta %.3e (bar 2e-6)\n",
               mode == PLOW_KDA_GATE_LOWER_BOUND ? "bounded" : "softplus", max_g, max_b);
        return fmax(max_g / 2e-4, max_b / 2e-6);
    };

    auto run_intra = [&]() -> double {
        const unsigned IT = 145, IH = 2, ID = 128;
        const float scale = 1.0f / sqrtf((float)ID);
        std::vector<uint2> chunks{make_uint2(0, 64), make_uint2(64, 17), make_uint2(81, 1),
                                  make_uint2(82, 63)};
        std::vector<bf16> q((size_t)IT * IH * ID), k(q.size());
        std::vector<float> g(q.size()), beta((size_t)IT * IH);
        for (unsigned t = 0; t < IT; t++)
            for (unsigned h = 0; h < IH; h++) {
                double qss = 0.0, kss = 0.0;
                std::vector<float> qr(ID), kr(ID);
                for (unsigned d = 0; d < ID; d++) {
                    qr[d] = (float)rand() / RAND_MAX - .5f;
                    kr[d] = (float)rand() / RAND_MAX - .5f;
                    qss += (double)qr[d] * qr[d];
                    kss += (double)kr[d] * kr[d];
                }
                for (unsigned d = 0; d < ID; d++) {
                    const size_t x = ((size_t)t * IH + h) * ID + d;
                    q[x] = f2b(qr[d] / sqrtf((float)qss));
                    k[x] = f2b(kr[d] / sqrtf((float)kss));
                }
                beta[(size_t)t * IH + h] = .1f + .8f * (float)rand() / RAND_MAX;
            }
        for (const uint2 c : chunks)
            for (unsigned h = 0; h < IH; h++)
                for (unsigned d = 0; d < ID; d++) {
                    float acc = 0.0f;
                    for (unsigned tl = 0; tl < c.y; tl++) {
                        acc -= .01f + .03f * (float)rand() / RAND_MAX;
                        g[(((size_t)c.x + tl) * IH + h) * ID + d] = acc;
                    }
                }

        std::vector<double> qk_ref((size_t)IT * IH * 16), inv_ref(qk_ref.size());
        for (const uint2 c : chunks)
            for (unsigned sub = 0; sub < 4 && sub * 16 < c.y; sub++) {
                const unsigned valid = std::min(16u, c.y - sub * 16);
                const size_t row0 = (size_t)c.x + sub * 16;
                const size_t anchor = row0 + std::min(8u, valid - 1);
                for (unsigned h = 0; h < IH; h++) {
                    double L[16][16] = {}, inv[16][16] = {};
                    for (unsigned m = 0; m < valid; m++)
                        for (unsigned n = 0; n <= m; n++) {
                            double qk_dot = 0.0, kk_dot = 0.0;
                            for (unsigned d = 0; d < ID; d++) {
                                const size_t mi = ((row0 + m) * IH + h) * ID + d;
                                const size_t ni = ((row0 + n) * IH + h) * ID + d;
                                const size_t ai = (anchor * IH + h) * ID + d;
                                const float qm = b2f(f2b(b2f(q[mi]) * exp2f(g[mi] - g[ai])));
                                const float km = b2f(f2b(b2f(k[mi]) * exp2f(g[mi] - g[ai])));
                                const float kn = b2f(f2b(b2f(k[ni]) * exp2f(-(g[ni] - g[ai]))));
                                qk_dot += (double)qm * kn;
                                kk_dot += (double)km * kn;
                            }
                            qk_ref[((row0 + m) * IH + h) * 16 + n] = qk_dot * scale;
                            if (n < m) L[m][n] = kk_dot * beta[(row0 + m) * IH + h];
                        }
                    for (unsigned i = 0; i < valid; i++) {
                        inv[i][i] = 1.0;
                        for (unsigned n = 0; n < i; n++) {
                            double v = -L[i][n];
                            for (unsigned j = n + 1; j < i; j++) v -= L[i][j] * inv[j][n];
                            inv[i][n] = v;
                        }
                    }
                    for (unsigned m = 0; m < valid; m++)
                        for (unsigned n = 0; n <= m; n++)
                            inv_ref[((row0 + m) * IH + h) * 16 + n] = inv[m][n];
                }
            }

        void *d_q, *d_k, *d_g, *d_beta, *d_chunks, *d_qk, *d_inv, *d_a;
        HC(hipMalloc(&d_q, q.size() * sizeof(bf16)));
        HC(hipMalloc(&d_k, k.size() * sizeof(bf16)));
        HC(hipMalloc(&d_g, g.size() * sizeof(float)));
        HC(hipMalloc(&d_beta, beta.size() * sizeof(float)));
        HC(hipMalloc(&d_chunks, chunks.size() * sizeof(uint2)));
        HC(hipMalloc(&d_qk, qk_ref.size() * sizeof(float)));
        HC(hipMalloc(&d_inv, inv_ref.size() * sizeof(float)));
        HC(hipMemcpy(d_q, q.data(), q.size() * sizeof(bf16), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_k, k.data(), k.size() * sizeof(bf16), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_g, g.data(), g.size() * sizeof(float), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_beta, beta.data(), beta.size() * sizeof(float), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_chunks, chunks.data(), chunks.size() * sizeof(uint2), hipMemcpyHostToDevice));

        KArgs a{};
        a.q = (unsigned long long)(size_t)d_q; a.k = (unsigned long long)(size_t)d_k;
        a.gprefix = (unsigned long long)(size_t)d_g; a.o = (unsigned long long)(size_t)d_beta;
        a.chunks = (unsigned long long)(size_t)d_chunks;
        a.aqk = (unsigned long long)(size_t)d_qk; a.ainv = (unsigned long long)(size_t)d_inv;
        a.T = IT; a.H = IH; a.D = ID; a.n_chunks = chunks.size(); a.scale = scale;
        HC(hipMalloc(&d_a, sizeof a));
        HC(hipMemcpy(d_a, &a, sizeof a, hipMemcpyHostToDevice));
        void* args[] = {&d_a};
        HC(hipModuleLaunchKernel(fintra, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());
        std::vector<float> qk(qk_ref.size()), inv(inv_ref.size());
        HC(hipMemcpy(qk.data(), d_qk, qk.size() * sizeof(float), hipMemcpyDeviceToHost));
        HC(hipMemcpy(inv.data(), d_inv, inv.size() * sizeof(float), hipMemcpyDeviceToHost));
        double qk_max = 0.0, inv_max = 0.0;
        for (size_t i = 0; i < qk.size(); i++) qk_max = fmax(qk_max, fabs(qk[i] - qk_ref[i]));
        for (size_t i = 0; i < inv.size(); i++) inv_max = fmax(inv_max, fabs(inv[i] - inv_ref[i]));
        printf("  chunk intra BC16 qk max abs %.3e (bar 5e-5) inv %.3e (bar 2e-4)\n",
               qk_max, inv_max);

        std::vector<double> qk64_ref((size_t)IT * IH * 64), inv64_ref(qk64_ref.size());
        for (const uint2 c : chunks)
            for (unsigned h = 0; h < IH; h++) {
                double L[64][64] = {}, inverse[64][64] = {};
                for (unsigned m = 0; m < c.y; m++)
                    for (unsigned n = 0; n <= m; n++) {
                        const unsigned bm = m / 16, bn = n / 16;
                        const unsigned block_valid = std::min(16u, c.y - bm * 16);
                        const unsigned anchor_local =
                            bm == bn ? bm * 16 + std::min(8u, block_valid - 1) : bm * 16;
                        const size_t anchor = (size_t)c.x + anchor_local;
                        double qk_dot = 0.0, kk_dot = 0.0;
                        for (unsigned d = 0; d < ID; d++) {
                            const size_t mi = (((size_t)c.x + m) * IH + h) * ID + d;
                            const size_t ni = (((size_t)c.x + n) * IH + h) * ID + d;
                            const size_t ai = (anchor * IH + h) * ID + d;
                            const float qm = b2f(f2b(b2f(q[mi]) * exp2f(g[mi] - g[ai])));
                            const float km = b2f(f2b(b2f(k[mi]) * exp2f(g[mi] - g[ai])));
                            const float kn = b2f(f2b(b2f(k[ni]) * exp2f(-(g[ni] - g[ai]))));
                            qk_dot += (double)qm * kn;
                            kk_dot += (double)km * kn;
                        }
                        qk64_ref[(((size_t)c.x + m) * IH + h) * 64 + n] = qk_dot * scale;
                        if (n < m) L[m][n] = kk_dot * beta[((size_t)c.x + m) * IH + h];
                    }
                for (unsigned i = 0; i < c.y; i++) {
                    inverse[i][i] = 1.0;
                    for (unsigned n = 0; n < i; n++) {
                        double v = -L[i][n];
                        for (unsigned j = n + 1; j < i; j++) v -= L[i][j] * inverse[j][n];
                        inverse[i][n] = v;
                    }
                }
                for (unsigned m = 0; m < c.y; m++)
                    for (unsigned n = 0; n <= m; n++)
                        inv64_ref[(((size_t)c.x + m) * IH + h) * 64 + n] = inverse[m][n];
            }
        void *d_qk64, *d_inv64;
        HC(hipMalloc(&d_qk64, qk64_ref.size() * sizeof(float)));
        HC(hipMalloc(&d_inv64, inv64_ref.size() * sizeof(float)));
        a.aqk = (unsigned long long)(size_t)d_qk64;
        a.ainv = (unsigned long long)(size_t)d_inv64;
        HC(hipMemcpy(d_a, &a, sizeof a, hipMemcpyHostToDevice));
        HC(hipModuleLaunchKernel(fintra_full, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());
        std::vector<float> qk64(qk64_ref.size()), inv64(inv64_ref.size());
        HC(hipMemcpy(qk64.data(), d_qk64, qk64.size() * sizeof(float), hipMemcpyDeviceToHost));
        HC(hipMemcpy(inv64.data(), d_inv64, inv64.size() * sizeof(float), hipMemcpyDeviceToHost));
        double qk64_max = 0.0, inv64_max = 0.0;
        for (size_t i = 0; i < qk64.size(); i++)
            qk64_max = fmax(qk64_max, fabs(qk64[i] - qk64_ref[i]));
        for (size_t i = 0; i < inv64.size(); i++)
            inv64_max = fmax(inv64_max, fabs(inv64[i] - inv64_ref[i]));
        printf("  chunk intra BT64 qk max abs %.3e (bar 5e-5) inv %.3e (bar 3e-4)\n",
               qk64_max, inv64_max);

        HC(hipMemset(d_qk64, 0, qk64_ref.size() * sizeof(float)));
        HC(hipMemset(d_inv64, 0, inv64_ref.size() * sizeof(float)));
        HC(hipModuleLaunchKernel(fintra_cached, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0,
                                 args, nullptr));
        HC(hipDeviceSynchronize());
        std::vector<float> qk_cached(qk64_ref.size()), inv_cached(inv64_ref.size());
        HC(hipMemcpy(qk_cached.data(), d_qk64, qk_cached.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
        HC(hipMemcpy(inv_cached.data(), d_inv64, inv_cached.size() * sizeof(float),
                     hipMemcpyDeviceToHost));
        size_t qk_exact_bad = 0, inv_exact_bad = 0;
        double qk_cached_max = 0.0, inv_cached_max = 0.0;
        for (size_t i = 0; i < qk_cached.size(); ++i) {
            qk_exact_bad += qk_cached[i] != qk64[i];
            qk_cached_max = fmax(qk_cached_max, fabs(qk_cached[i] - qk64_ref[i]));
        }
        for (size_t i = 0; i < inv_cached.size(); ++i) {
            inv_exact_bad += inv_cached[i] != inv64[i];
            inv_cached_max = fmax(inv_cached_max, fabs(inv_cached[i] - inv64_ref[i]));
        }
        printf("  chunk intra cached qk max abs %.3e exact_bad %zu/%zu inv max abs %.3e exact_bad %zu/%zu\n",
               qk_cached_max, qk_exact_bad, qk_cached.size(), inv_cached_max, inv_exact_bad,
               inv_cached.size());

        const unsigned IV = 32;
        std::vector<bf16> value((size_t)IT * IH * IV);
        std::vector<float> state0((size_t)IH * IV * ID);
        for (auto& x : value) x = f2b(((float)rand() / RAND_MAX - .5f));
        for (auto& x : state0) x = ((float)rand() / RAND_MAX - .5f) * .1f;
        std::vector<double> state_ref(state0.begin(), state0.end());
        std::vector<double> out_ref((size_t)IT * IH * IV);
        for (const uint2 c : chunks)
            for (unsigned h = 0; h < IH; h++) {
                std::vector<double> prev(ID, 0.0);
                for (unsigned tl = 0; tl < c.y; tl++) {
                    const size_t row = (size_t)c.x + tl;
                    std::vector<double> kn(ID), qn(ID);
                    for (unsigned d = 0; d < ID; d++) {
                        const size_t i = (row * IH + h) * ID + d;
                        kn[d] = b2f(k[i]); qn[d] = b2f(q[i]) * scale;
                        const double decay = exp2((double)g[i] - prev[d]);
                        for (unsigned j = 0; j < IV; j++)
                            state_ref[((size_t)h * IV + j) * ID + d] *= decay;
                        prev[d] = g[i];
                    }
                    for (unsigned j = 0; j < IV; j++) {
                        double pk = 0.0;
                        double* sj = state_ref.data() + ((size_t)h * IV + j) * ID;
                        for (unsigned d = 0; d < ID; d++) pk += sj[d] * kn[d];
                        const double delta =
                            beta[row * IH + h] * ((double)b2f(value[(row * IH + h) * IV + j]) - pk);
                        double oq = 0.0;
                        for (unsigned d = 0; d < ID; d++) {
                            sj[d] += delta * kn[d];
                            oq += sj[d] * qn[d];
                        }
                        out_ref[(row * IH + h) * IV + j] = oq;
                    }
                }
            }

        void *d_value, *d_w, *d_u, *d_state, *d_out;
        HC(hipMalloc(&d_value, value.size() * sizeof(bf16)));
        HC(hipMalloc(&d_w, q.size() * sizeof(bf16)));
        HC(hipMalloc(&d_u, value.size() * sizeof(bf16)));
        HC(hipMalloc(&d_state, state0.size() * sizeof(float)));
        HC(hipMalloc(&d_out, value.size() * sizeof(bf16)));
        HC(hipMemcpy(d_value, value.data(), value.size() * sizeof(bf16), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_state, state0.data(), state0.size() * sizeof(float), hipMemcpyHostToDevice));
        a.v = (unsigned long long)(size_t)d_value;
        a.w = (unsigned long long)(size_t)d_w; a.u = (unsigned long long)(size_t)d_u;
        a.state = (unsigned long long)(size_t)d_state; a.y = (unsigned long long)(size_t)d_out;
        a.V = IV;
        HC(hipMemcpy(d_a, &a, sizeof a, hipMemcpyHostToDevice));
        HC(hipModuleLaunchKernel(fwu, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());
        double qpre_bad = 0.0;
        if (getenv("KDA_QPRE_CHECK")) {
            std::vector<bf16> qgot(q.size());
            HC(hipMemcpy(qgot.data(), d_q, qgot.size() * sizeof(bf16), hipMemcpyDeviceToHost));
            size_t bad = 0;
            for (size_t i = 0; i < q.size(); ++i) {
                const bf16 qs = f2b(b2f(q[i]) * scale);
                const bf16 want = f2b(b2f(qs) * exp2f(g[i]));
                bad += qgot[i] != want;
            }
            qpre_bad = bad ? 2.0 : 0.0;
            printf("  chunk q precompute exact %zu/%zu bad\n", bad, q.size());
        }
        HC(hipModuleLaunchKernel(fcarry, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());
        std::vector<bf16> out(value.size());
        std::vector<float> state(state0.size());
        HC(hipMemcpy(out.data(), d_out, out.size() * sizeof(bf16), hipMemcpyDeviceToHost));
        HC(hipMemcpy(state.data(), d_state, state.size() * sizeof(float), hipMemcpyDeviceToHost));
        double ose = 0.0, osw = 0.0, sse = 0.0, ssw = 0.0;
        for (size_t i = 0; i < out.size(); i++) {
            const double d = b2f(out[i]) - out_ref[i]; ose += d * d; osw += out_ref[i] * out_ref[i];
        }
        for (size_t i = 0; i < state.size(); i++) {
            const double d = state[i] - state_ref[i]; sse += d * d; ssw += state_ref[i] * state_ref[i];
        }
        const double orms = sqrt(ose / (osw + 1e-30)), srms = sqrt(sse / (ssw + 1e-30));
        printf("  chunk carry 4 chunks output rms rel %.3e (bar 1e-2) state %.3e (bar 5e-3)\n",
               orms, srms);
        return fmax(fmax(qk_max / 5e-5, inv_max / 2e-4),
                    fmax(fmax(fmax(qk64_max / 5e-5, inv64_max / 3e-4),
                              fmax(qk_cached_max / 5e-5, inv_cached_max / 3e-4)),
                         fmax(fmax(orms / 1e-2, srms / 5e-3), qpre_bad)));
    };

    double worst = 0;
    worst = fmax(worst, run_prefix(PLOW_KDA_GATE_LOWER_BOUND));
    worst = fmax(worst, run_prefix(PLOW_KDA_GATE_SOFTPLUS));
    worst = fmax(worst, run_intra());

    if (getenv("KDA3_BENCH")) {
        const unsigned BH = getenv("KDA_BENCH_H")
            ? (unsigned)strtoul(getenv("KDA_BENCH_H"), nullptr, 10)
            : 12;
        const unsigned BD = 128, BVV = 128;
        const unsigned GRIDB = getenv("KDA_BENCH_GRID")
            ? (unsigned)strtoul(getenv("KDA_BENCH_GRID"), nullptr, 10)
            : p.multiProcessorCount;
        const unsigned only_t = getenv("KDA_BENCH_T")
            ? (unsigned)strtoul(getenv("KDA_BENCH_T"), nullptr, 10)
            : 0;
        if (GRIDB == 0) {
            fprintf(stderr, "KDA_BENCH_GRID must be positive\n");
            return 2;
        }
        const float scale = 1.0f / sqrtf((float)BD);
        printf("BENCH gfx=%s H=%u D=%u V=%u grid=%u\n", p.gcnArchName, BH, BD, BVV, GRIDB);
        for (const unsigned T : {128u, 511u, 512u, 513u, 2048u, 8191u, 8192u}) {
            if (only_t && T != only_t) continue;
            const unsigned nc = (T + 63u) / 64u;
            std::vector<uint2> chunks(nc);
            for (unsigned c = 0; c < nc; c++)
                chunks[c] = make_uint2(c * 64u, std::min(64u, T - c * 64u));
            std::vector<bf16> qb((size_t)T * BH * BD), kb(qb.size()), vb(qb.size()),
                graw(qb.size()), braw((size_t)T * BH);
            const float norm = 1.0f / sqrtf((float)BD);
            for (size_t i = 0; i < qb.size(); i++) {
                qb[i] = f2b(((i * 17u + 3u) & 1u ? norm : -norm));
                kb[i] = f2b(((i * 29u + i / BD) & 1u ? norm : -norm));
                vb[i] = f2b(((int)(i * 13u % 31u) - 15) / 60.0f);
                graw[i] = f2b(((int)(i * 7u % 17u) - 8) / 32.0f);
            }
            for (size_t i = 0; i < braw.size(); i++)
                braw[i] = f2b(((int)(i * 11u % 13u) - 6) / 8.0f);
            std::vector<float> alog(BH, 0.0f), dtb((size_t)BH * BD, -4.0f);
            std::vector<float> state0((size_t)BH * BVV * BD);
            for (size_t i = 0; i < state0.size(); i++)
                state0[i] = ((int)(i * 19u % 23u) - 11) / 1000.0f;

            auto alloc_copy = [&](const void* src, size_t bytes) {
                void* d = nullptr;
                if (hipMalloc(&d, bytes) != hipSuccess ||
                    hipMemcpy(d, src, bytes, hipMemcpyHostToDevice) != hipSuccess) exit(2);
                return d;
            };
            void* dq = alloc_copy(qb.data(), qb.size() * 2);
            void* dk = alloc_copy(kb.data(), kb.size() * 2);
            void* dv = alloc_copy(vb.data(), vb.size() * 2);
            void* dgraw = alloc_copy(graw.data(), graw.size() * 2);
            void* dbraw = alloc_copy(braw.data(), braw.size() * 2);
            void* dalog = alloc_copy(alog.data(), alog.size() * 4);
            void* ddtb = alloc_copy(dtb.data(), dtb.size() * 4);
            void* dchunks = alloc_copy(chunks.data(), chunks.size() * sizeof(uint2));
            void *dss, *dsc, *dos, *doc, *dg, *dbeta, *daqk, *dainv, *dw, *du;
            HC(hipMalloc(&dss, state0.size() * 4)); HC(hipMalloc(&dsc, state0.size() * 4));
            HC(hipMalloc(&dos, vb.size() * 2)); HC(hipMalloc(&doc, vb.size() * 2));
            HC(hipMalloc(&dg, qb.size() * 4)); HC(hipMalloc(&dbeta, braw.size() * 4));
            HC(hipMalloc(&daqk, (size_t)T * BH * 64u * 4));
            HC(hipMalloc(&dainv, (size_t)T * BH * 64u * 4));
            HC(hipMalloc(&dw, qb.size() * 2)); HC(hipMalloc(&du, vb.size() * 2));

            KArgs as{};
            as.o = (size_t)dos; as.q = (size_t)dq; as.k = (size_t)dk; as.v = (size_t)dv;
            as.g_raw = (size_t)dgraw; as.beta_raw = (size_t)dbraw; as.a_log = (size_t)dalog;
            as.dt_bias = (size_t)ddtb; as.state = (size_t)dss;
            as.T = T; as.H = BH; as.D = BD; as.BV = 8; as.gate_mode = PLOW_KDA_GATE_LOWER_BOUND;
            as.lb = -5.0f; as.scale = scale;
            KArgs ac = as;
            ac.o = (size_t)dbeta; ac.y = (size_t)doc; ac.state = (size_t)dsc;
            ac.chunks = 0; ac.gprefix = (size_t)dg;
            ac.aqk = (size_t)daqk; ac.ainv = (size_t)dainv; ac.w = (size_t)dw; ac.u = (size_t)du;
            ac.n_chunks = nc; ac.V = BVV;
            void *das, *dac; HC(hipMalloc(&das, sizeof as)); HC(hipMalloc(&dac, sizeof ac));
            HC(hipMemcpy(das, &as, sizeof as, hipMemcpyHostToDevice));
            HC(hipMemcpy(dac, &ac, sizeof ac, hipMemcpyHostToDevice));
            void* args_s[] = {&das}; void* args_c[] = {&dac};
            auto serial = [&]() {
                HC2(hipModuleLaunchKernel(fstep, GRIDB, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args_s, nullptr));
            };
            auto chunked = [&]() {
                HC2(hipModuleLaunchKernel(fprepare, GRIDB, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args_c, nullptr));
                HC2(hipModuleLaunchKernel(fintra_cached, GRIDB, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args_c, nullptr));
                HC2(hipModuleLaunchKernel(fwu, GRIDB, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args_c, nullptr));
                HC2(hipModuleLaunchKernel(fcarry, GRIDB, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args_c, nullptr));
            };
            HC(hipMemcpy(dss, state0.data(), state0.size() * 4, hipMemcpyHostToDevice));
            serial(); HC(hipDeviceSynchronize());
            HC(hipMemcpy(dsc, state0.data(), state0.size() * 4, hipMemcpyHostToDevice));
            chunked(); HC(hipDeviceSynchronize());
            std::vector<bf16> os(vb.size()), oc(vb.size());
            std::vector<float> ss(state0.size()), sc(state0.size());
            HC(hipMemcpy(os.data(), dos, os.size() * 2, hipMemcpyDeviceToHost));
            HC(hipMemcpy(oc.data(), doc, oc.size() * 2, hipMemcpyDeviceToHost));
            HC(hipMemcpy(ss.data(), dss, ss.size() * 4, hipMemcpyDeviceToHost));
            HC(hipMemcpy(sc.data(), dsc, sc.size() * 4, hipMemcpyDeviceToHost));
            double ose = 0, osw = 0, sse = 0, ssw = 0;
            for (size_t i = 0; i < os.size(); i++) {
                double d = b2f(oc[i]) - b2f(os[i]); ose += d*d; osw += (double)b2f(os[i])*b2f(os[i]);
            }
            for (size_t i = 0; i < ss.size(); i++) {
                double d = sc[i] - ss[i]; sse += d*d; ssw += (double)ss[i]*ss[i];
            }
            const double out_rel = sqrt(ose / (osw + 1e-30));
            const double state_rel = sqrt(sse / (ssw + 1e-30));
            const unsigned iters = T <= 512 ? 5 : (T <= 2048 ? 3 : 2);
            hipEvent_t beg, end; HC(hipEventCreate(&beg)); HC(hipEventCreate(&end));
            auto time_path = [&](bool use_chunk) {
                float total = 0;
                for (unsigned it = 0; it < iters; it++) {
                    void* ds = use_chunk ? dsc : dss;
                    HC2(hipMemcpy(ds, state0.data(), state0.size() * 4, hipMemcpyHostToDevice));
                    HC2(hipEventRecord(beg));
                    if (use_chunk) chunked(); else serial();
                    HC2(hipEventRecord(end)); HC2(hipEventSynchronize(end));
                    float ms = 0; HC2(hipEventElapsedTime(&ms, beg, end)); total += ms;
                }
                return total / iters;
            };
            const float sms = time_path(false), cms = time_path(true);
            auto time_stage = [&](hipFunction_t fn, bool reset_state) {
                float total = 0;
                for (unsigned it = 0; it < iters; it++) {
                    if (reset_state)
                        HC2(hipMemcpy(dsc, state0.data(), state0.size() * 4,
                                      hipMemcpyHostToDevice));
                    HC2(hipEventRecord(beg));
                    HC2(hipModuleLaunchKernel(fn, GRIDB, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args_c,
                                              nullptr));
                    HC2(hipEventRecord(end)); HC2(hipEventSynchronize(end));
                    float ms = 0; HC2(hipEventElapsedTime(&ms, beg, end)); total += ms;
                }
                return total / iters;
            };
            const float pms = time_stage(fprepare, false);
            const float ims = time_stage(fintra_full, false);
            const float icms = time_stage(fintra_cached, false);
            const float wms = time_stage(fwu, false);
            const float hms = time_stage(fcarry, true);
            printf("BENCH T=%u serial_ms=%.4f chunk_ms=%.4f speedup=%.3f prefix_ms=%.4f intra_ms=%.4f intra_cached_ms=%.4f wu_ms=%.4f carry_ms=%.4f out_rel=%.4e state_rel=%.4e\n",
                   T, sms, cms, sms / cms, pms, ims, icms, wms, hms, out_rel, state_rel);
            if (T >= 512 && (out_rel > 1e-2 || state_rel > 5e-3)) worst = fmax(worst, 2.0);
            for (void* d : {dq,dk,dv,dgraw,dbraw,dalog,ddtb,dchunks,dss,dsc,dos,doc,dg,dbeta,
                            daqk,dainv,dw,du,das,dac}) HC(hipFree(d));
            HC(hipEventDestroy(beg)); HC(hipEventDestroy(end));
        }
    }

    auto run_packed = [&]() -> double {
        const unsigned T = 5, ACTIVE = 4, BV = 8, NSLOT = 3;
        const size_t row_elems = (size_t)H * D, state_elems = row_elems * D;
        std::vector<bf16> q(T * row_elems), k(q.size()), v(q.size()), graw(q.size());
        std::vector<bf16> braw((size_t)T * H);
        std::vector<float> alog(H), dtb(row_elems);
        for (auto& x : q) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : k) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : v) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : graw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        for (auto& x : braw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        for (auto& x : alog) x = logf(.5f + (float)rand() / RAND_MAX * 11.5f);
        for (auto& x : dtb) x = -7.f + (float)rand() / RAND_MAX * 6.f;

        std::vector<float> S0((size_t)NSLOT * state_elems);
        for (size_t i = 0; i < S0.size(); ++i) S0[i] = .125f + (float)(i % 97) / 211.f;
        std::vector<double> Sref(S0.begin(), S0.end()), oref((size_t)T * row_elems, 7.0);
        auto ref_span = [&](unsigned row0, unsigned state_slot, bool reset) {
            auto rows = [&](const std::vector<bf16>& src, size_t width) {
                return std::vector<bf16>(src.begin() + (size_t)row0 * width,
                                         src.begin() + (size_t)(row0 + 2) * width);
            };
            auto qs = rows(q, row_elems), ks = rows(k, row_elems), vs = rows(v, row_elems);
            auto gs = rows(graw, row_elems), bs = rows(braw, H);
            std::vector<double> ss(state_elems);
            if (!reset)
                std::copy(Sref.begin() + (size_t)state_slot * state_elems,
                          Sref.begin() + (size_t)(state_slot + 1) * state_elems, ss.begin());
            std::vector<double> oo(2 * row_elems);
            ref_step(2, H, D, qs, ks, vs, gs, bs, alog, dtb, -5.0, 1.0 / sqrt(128.0), ss,
                     oo, false);
            std::copy(ss.begin(), ss.end(),
                      Sref.begin() + (size_t)state_slot * state_elems);
            std::copy(oo.begin(), oo.end(), oref.begin() + (size_t)row0 * row_elems);
        };
        ref_span(0, 2, true);
        ref_span(2, 0, false);

        std::vector<PrefillSpan> spans{{0, 2, 2, 1, 11, 13, 2, 0},
                                       {2, 2, 0, 0, 7, 9, 0, 0}};
        std::vector<unsigned> parked{0, 0, 0, 0, 1};
        auto upload = [&](const void* src, size_t bytes) {
            void* d = nullptr;
            HC2(hipMalloc(&d, bytes));
            HC2(hipMemcpy(d, src, bytes, hipMemcpyHostToDevice));
            return d;
        };
        KArgs a{};
        a.q = (unsigned long long)(size_t)upload(q.data(), q.size() * 2);
        a.k = (unsigned long long)(size_t)upload(k.data(), k.size() * 2);
        a.v = (unsigned long long)(size_t)upload(v.data(), v.size() * 2);
        a.g_raw = (unsigned long long)(size_t)upload(graw.data(), graw.size() * 2);
        a.beta_raw = (unsigned long long)(size_t)upload(braw.data(), braw.size() * 2);
        a.a_log = (unsigned long long)(size_t)upload(alog.data(), alog.size() * 4);
        a.dt_bias = (unsigned long long)(size_t)upload(dtb.data(), dtb.size() * 4);
        a.state = (unsigned long long)(size_t)upload(S0.data(), S0.size() * 4);
        std::vector<bf16> out((size_t)T * row_elems, f2b(7.f));
        a.o = (unsigned long long)(size_t)upload(out.data(), out.size() * 2);
        a.spans = (unsigned long long)(size_t)upload(spans.data(), spans.size() * sizeof(PrefillSpan));
        a.parked = (unsigned long long)(size_t)upload(parked.data(), parked.size() * 4);
        a.nspans = spans.size(); a.nrows = T; a.H = H; a.D = D; a.BV = BV;
        a.flags = 1u; a.gate_mode = 1u; a.lb = -5.f; a.scale = 1.f / sqrtf(128.f);
        void* d_a = upload(&a, sizeof a); void* args[] = {&d_a};
        HC2(hipModuleLaunchKernel(fstep_packed, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args, nullptr));
        HC2(hipDeviceSynchronize());
        std::vector<float> Sg(S0.size());
        HC2(hipMemcpy(Sg.data(), (void*)(size_t)a.state, Sg.size() * 4, hipMemcpyDeviceToHost));
        HC2(hipMemcpy(out.data(), (void*)(size_t)a.o, out.size() * 2, hipMemcpyDeviceToHost));
        double se = 0, sw = 0, oe = 0, ow = 0;
        for (size_t i = 0; i < Sg.size(); ++i) {
            double d = Sg[i] - Sref[i]; se += d * d; sw += Sref[i] * Sref[i];
        }
        for (size_t i = 0; i < (size_t)ACTIVE * row_elems; ++i) {
            double d = b2f(out[i]) - oref[i]; oe += d * d; ow += oref[i] * oref[i];
        }
        bool untouched = true;
        for (size_t i = state_elems; i < 2 * state_elems; ++i) untouched &= Sg[i] == S0[i];
        for (size_t i = (size_t)ACTIVE * row_elems; i < out.size(); ++i)
            untouched &= out[i] == f2b(7.f);
        const double srms = sqrt(se / (sw + 1e-30)), orms = sqrt(oe / (ow + 1e-30));
        printf("  packed two-span    reset+carry state %.3e output %.3e untouched=%s\n",
               srms, orms, untouched ? "yes" : "NO");
        return fmax(fmax(srms / 1e-4, orms / 1e-2), untouched ? 0.0 : 2.0);
    };

    auto run_packed_chunk = [&]() -> double {
        const unsigned T = 85, ACTIVE = 82, PH = 2, PD = 128, NSLOT = 3;
        const size_t row_elems = (size_t)PH * PD, state_elems = row_elems * PD;
        std::vector<bf16> q(T * row_elems), k(q.size()), v(q.size()), graw(q.size());
        std::vector<bf16> braw((size_t)T * PH);
        std::vector<float> alog(PH), dtb(row_elems);
        for (auto& x : q) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : k) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : v) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : graw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        for (auto& x : braw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        for (auto& x : alog) x = logf(.5f + (float)rand() / RAND_MAX * 11.5f);
        for (auto& x : dtb) x = -7.f + (float)rand() / RAND_MAX * 6.f;
        std::vector<float> state0((size_t)NSLOT * state_elems);
        for (size_t i = 0; i < state0.size(); ++i)
            state0[i] = .125f + (float)(i % 97) / 211.f;
        std::vector<PrefillSpan> spans{{0, 65, 2, 1, 0, 65, 2, 0},
                                       {65, 17, 0, 0, 7, 24, 0, 0}};
        std::vector<unsigned> parked(T, 1);
        std::fill(parked.begin(), parked.begin() + ACTIVE, 0);
        auto upload = [&](const void* src, size_t bytes) {
            void* d = nullptr;
            HC2(hipMalloc(&d, bytes));
            HC2(hipMemcpy(d, src, bytes, hipMemcpyHostToDevice));
            return d;
        };
        void* d_qs = upload(q.data(), q.size() * 2);
        void* d_ks = upload(k.data(), k.size() * 2);
        void* d_qc = upload(q.data(), q.size() * 2);
        void* d_kc = upload(k.data(), k.size() * 2);
        void* d_v = upload(v.data(), v.size() * 2);
        void* d_graw = upload(graw.data(), graw.size() * 2);
        void* d_braw = upload(braw.data(), braw.size() * 2);
        void* d_alog = upload(alog.data(), alog.size() * 4);
        void* d_dtb = upload(dtb.data(), dtb.size() * 4);
        void* d_spans = upload(spans.data(), spans.size() * sizeof(PrefillSpan));
        void* d_parked = upload(parked.data(), parked.size() * 4);
        void* d_ss = upload(state0.data(), state0.size() * 4);
        void* d_sc = upload(state0.data(), state0.size() * 4);
        std::vector<bf16> out0((size_t)T * row_elems, f2b(7.f));
        void* d_os = upload(out0.data(), out0.size() * 2);
        void* d_oc = upload(out0.data(), out0.size() * 2);
        std::vector<float> fscratch((size_t)T * PH * 64u, 19.f);
        std::vector<float> gscratch(q.size(), 19.f), bscratch((size_t)T * PH, 19.f);
        std::vector<bf16> bscratch16(q.size(), f2b(7.f));
        void* d_gp = upload(gscratch.data(), gscratch.size() * 4);
        void* d_beta = upload(bscratch.data(), bscratch.size() * 4);
        void* d_aqk = upload(fscratch.data(), fscratch.size() * 4);
        void* d_ainv = upload(fscratch.data(), fscratch.size() * 4);
        void* d_w = upload(bscratch16.data(), bscratch16.size() * 2);
        void* d_u = upload(bscratch16.data(), bscratch16.size() * 2);

        KArgs as{};
        as.o = (size_t)d_os; as.q = (size_t)d_qs; as.k = (size_t)d_ks; as.v = (size_t)d_v;
        as.g_raw = (size_t)d_graw; as.beta_raw = (size_t)d_braw; as.a_log = (size_t)d_alog;
        as.dt_bias = (size_t)d_dtb; as.state = (size_t)d_ss; as.spans = (size_t)d_spans;
        as.parked = (size_t)d_parked; as.nspans = spans.size(); as.nrows = T;
        as.H = PH; as.D = PD; as.BV = 8; as.flags = 1; as.gate_mode = 1; as.lb = -5.f;
        as.scale = 1.f / sqrtf((float)PD);
        KArgs ac = as;
        ac.o = (size_t)d_beta; ac.y = (size_t)d_oc; ac.q = (size_t)d_qc; ac.k = (size_t)d_kc;
        ac.state = (size_t)d_sc; ac.gprefix = (size_t)d_gp; ac.aqk = (size_t)d_aqk;
        ac.ainv = (size_t)d_ainv; ac.w = (size_t)d_w; ac.u = (size_t)d_u; ac.V = PD;
        void* d_as = upload(&as, sizeof as);
        void* d_ac = upload(&ac, sizeof ac);
        void* args_s[] = {&d_as};
        void* args_c[] = {&d_ac};
        HC2(hipModuleLaunchKernel(fstep_packed, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args_s, nullptr));
        for (hipFunction_t fn : {fprepare_packed, fintra_packed, fwu_packed, fcarry_packed})
            HC2(hipModuleLaunchKernel(fn, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args_c, nullptr));
        HC2(hipDeviceSynchronize());

        std::vector<bf16> os(out0.size()), oc(out0.size()), qc(q.size()), kc(k.size());
        std::vector<float> ss(state0.size()), sc(state0.size()), gp(gscratch.size());
        HC2(hipMemcpy(os.data(), d_os, os.size() * 2, hipMemcpyDeviceToHost));
        HC2(hipMemcpy(oc.data(), d_oc, oc.size() * 2, hipMemcpyDeviceToHost));
        HC2(hipMemcpy(ss.data(), d_ss, ss.size() * 4, hipMemcpyDeviceToHost));
        HC2(hipMemcpy(sc.data(), d_sc, sc.size() * 4, hipMemcpyDeviceToHost));
        HC2(hipMemcpy(qc.data(), d_qc, qc.size() * 2, hipMemcpyDeviceToHost));
        HC2(hipMemcpy(kc.data(), d_kc, kc.size() * 2, hipMemcpyDeviceToHost));
        HC2(hipMemcpy(gp.data(), d_gp, gp.size() * 4, hipMemcpyDeviceToHost));
        double oe = 0, ow = 0, se = 0, sw = 0;
        for (size_t i = 0; i < (size_t)ACTIVE * row_elems; ++i) {
            const double d = b2f(oc[i]) - b2f(os[i]);
            oe += d * d; ow += (double)b2f(os[i]) * b2f(os[i]);
        }
        for (size_t i = 0; i < sc.size(); ++i) {
            const double d = sc[i] - ss[i];
            se += d * d; sw += (double)ss[i] * ss[i];
        }
        bool untouched = true;
        for (size_t i = state_elems; i < 2 * state_elems; ++i)
            untouched &= sc[i] == state0[i];
        for (size_t i = (size_t)ACTIVE * row_elems; i < out0.size(); ++i) {
            untouched &= oc[i] == out0[i] && qc[i] == q[i] && kc[i] == k[i] && gp[i] == 19.f;
        }
        const double orms = sqrt(oe / (ow + 1e-30)), srms = sqrt(se / (sw + 1e-30));
        printf("  packed chunk spans ragged+reset state %.3e output %.3e untouched=%s\n",
               srms, orms, untouched ? "yes" : "NO");
        for (void* d : {d_qs,d_ks,d_qc,d_kc,d_v,d_graw,d_braw,d_alog,d_dtb,d_spans,d_parked,
                        d_ss,d_sc,d_os,d_oc,d_gp,d_beta,d_aqk,d_ainv,d_w,d_u,d_as,d_ac})
            HC2(hipFree(d));
        return fmax(fmax(srms / 5e-3, orms / 1e-2), untouched ? 0.0 : 2.0);
    };

    auto run_mode = [&](const char* nm, unsigned T, unsigned BV, bool batched) -> double {
        const unsigned nstate = batched ? T : 1;
        std::vector<bf16> q(T * H * D), k(q.size()), v(q.size()), graw(q.size()), braw(T * H);
        std::vector<float> alog(H), dtb(H * D);
        for (auto& x : q) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : k) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : v) x = f2b(((float)rand() / RAND_MAX - .5f) * 2.f);
        for (auto& x : graw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        for (auto& x : braw) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        /* checkpoint-shaped: exp(A_log) in [0.5, 12], dt_bias mean ~ -4.6 */
        for (auto& x : alog) x = logf(0.5f + (float)rand() / RAND_MAX * 11.5f);
        for (auto& x : dtb) x = -7.f + (float)rand() / RAND_MAX * 6.f;
        std::vector<float> S0((size_t)nstate * H * D * D);
        for (auto& x : S0) x = ((float)rand() / RAND_MAX - .5f) * .5f;

        std::vector<double> Sref(S0.begin(), S0.end()), oref((size_t)T * H * D, 0.0);
        ref_step(T, H, D, q, k, v, graw, braw, alog, dtb, -5.0, 1.0 / sqrt(128.0), Sref, oref,
                 batched);

#define UP(vec, elt) ({ void* d_; HC2(hipMalloc(&d_, (vec).size() * sizeof(elt)));                \
        HC2(hipMemcpy(d_, (vec).data(), (vec).size() * sizeof(elt), hipMemcpyHostToDevice)); d_; })
        KArgs a{};
        a.q = (unsigned long long)(size_t)UP(q, bf16); a.k = (unsigned long long)(size_t)UP(k, bf16);
        a.v = (unsigned long long)(size_t)UP(v, bf16);
        a.g_raw = (unsigned long long)(size_t)UP(graw, bf16);
        a.beta_raw = (unsigned long long)(size_t)UP(braw, bf16);
        a.a_log = (unsigned long long)(size_t)UP(alog, float);
        a.dt_bias = (unsigned long long)(size_t)UP(dtb, float);
        a.state = (unsigned long long)(size_t)UP(S0, float);
        void* d_o; HC(hipMalloc(&d_o, (size_t)T * H * D * 2)); HC(hipMemset(d_o, 0, (size_t)T * H * D * 2));
        a.o = (unsigned long long)(size_t)d_o;
        a.T = T; a.H = H; a.D = D; a.BV = BV;
        a.flags = 1u /* PLOW_KDA_F_QK_L2NORM */; a.gate_mode = 1u /* LOWER_BOUND */;
        a.lb = -5.0f; a.scale = (float)(1.0 / sqrt(128.0)); a.eps = 1e-5f;
        a.bstride = batched ? (unsigned long long)H * D * D : 0ull;
        void* d_a; HC(hipMalloc(&d_a, sizeof a)); HC(hipMemcpy(d_a, &a, sizeof a, hipMemcpyHostToDevice));
        void* args[] = {&d_a};
        HC(hipModuleLaunchKernel(fstep, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());

        std::vector<float> Sg(S0.size());
        std::vector<bf16> og((size_t)T * H * D);
        HC(hipMemcpy(Sg.data(), (void*)(size_t)a.state, Sg.size() * 4, hipMemcpyDeviceToHost));
        HC(hipMemcpy(og.data(), d_o, og.size() * 2, hipMemcpyDeviceToHost));
        double se = 0, sw = 0, se2 = 0, sw2 = 0;
        for (size_t i = 0; i < Sg.size(); i++) {
            const double d_ = Sg[i] - Sref[i]; se += d_ * d_; sw += Sref[i] * Sref[i];
        }
        for (size_t i = 0; i < og.size(); i++) {
            const double d_ = b2f(og[i]) - oref[i]; se2 += d_ * d_; sw2 += oref[i] * oref[i];
        }
        const double srms = sqrt(se / (sw + 1e-30)), orms = sqrt(se2 / (sw2 + 1e-30));
        printf("  %-18s T=%u BV=%u  state rms rel %.3e (bar 1e-4)   o rms rel %.3e (bar 1e-2)\n",
               nm, T, BV, srms, orms);
        /* op 103 on this mode's output */
        std::vector<float> nw(D);
        std::vector<bf16> g2((size_t)T * H * D);
        for (auto& x : nw) x = .5f + (float)rand() / RAND_MAX;
        for (auto& x : g2) x = f2b(((float)rand() / RAND_MAX - .5f) * 4.f);
        a.y = (unsigned long long)(size_t)UP(g2, bf16); /* reuse as y buffer, overwritten */
        a.norm_w = (unsigned long long)(size_t)UP(nw, float);
        a.gate2 = (unsigned long long)(size_t)UP(g2, bf16);
        HC(hipMemcpy(d_a, &a, sizeof a, hipMemcpyHostToDevice));
        HC(hipModuleLaunchKernel(fnorm, GRID, 1, 1, KDA_BENCH_THREADS, 1, 1, 0, 0, args, nullptr));
        HC(hipDeviceSynchronize());
        std::vector<bf16> yg((size_t)T * H * D);
        HC(hipMemcpy(yg.data(), (void*)(size_t)a.y, yg.size() * 2, hipMemcpyDeviceToHost));
        double se3 = 0, sw3 = 0;
        for (unsigned t = 0; t < T; t++)
            for (unsigned h = 0; h < H; h++) {
                const size_t base = ((size_t)t * H + h) * D;
                double ss = 0;
                for (unsigned d = 0; d < D; d++) { const double x = b2f(og[base + d]); ss += x * x; }
                const double inv = 1.0 / sqrt(ss / D + 1e-5);
                for (unsigned d = 0; d < D; d++) {
                    const double want =
                        (double)b2f(og[base + d]) * inv * nw[d] * sigm((double)b2f(g2[base + d]));
                    const double d_ = b2f(yg[base + d]) - want;
                    se3 += d_ * d_; sw3 += want * want;
                }
            }
        const double yrms = sqrt(se3 / (sw3 + 1e-30));
        printf("  %-18s gated-norm rms rel %.3e (bar 1e-2)\n", nm, yrms);
        return fmax(fmax(srms / 1e-4, orms / 1e-2), yrms / 1e-2);
    };

    worst = fmax(worst, run_mode("prefill (serial)", 5, 8, false));
    worst = fmax(worst, run_mode("batched decode", 3, 16, true));
    worst = fmax(worst, run_packed());
    worst = fmax(worst, run_packed_chunk());
    printf("%s\n", worst < 1.0 ? "PASS" : "FAIL");
    return worst >= 1.0;
}
#endif
