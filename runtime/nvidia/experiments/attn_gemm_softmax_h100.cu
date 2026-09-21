/* attn_gemm_softmax_h100.cu — harness for the vendor-GEMM prefill attention route
 * (PLOW_PF_ATTN_GEMM): per query-row tile, S = Q.K^T (cuBLASLt), the in-place causal softmax
 * of ../attn_softmax_sm90a.cu, O = P.V (cuBLASLt). Gemma-4-12B global layer geometry:
 * 16 query heads, ONE kv head, head_dim 512, scale 1.0.
 *
 *   nvcc -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/nvidia -o attn_h \
 *        runtime/nvidia/experiments/attn_gemm_softmax_h100.cu -lcublasLt
 *   attn_h check                       # f64 CPU oracle on small shapes
 *   attn_h bench <q_rows> <kv_prefix> <tile_rows> <grid> [reps]
 *
 * Timing is per chunk-layer (all tiles), event-timed, min and median over reps with 25 ms idle
 * gaps. A harness number is a hypothesis: the served program is the measurement.
 */
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <unistd.h>
#include <vector>

#define PLOW_ATTN_SM_HARNESS 1
#include "../attn_softmax_sm90a.cu"

#define CK(x)                                                                          \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__);                 \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)
#define LK(x)                                                                          \
    do {                                                                               \
        cublasStatus_t s_ = (x);                                                       \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                             \
            printf("cuBLASLt status %d @%d\n", (int)s_, __LINE__);                     \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static const unsigned HEADS = 16, HD = 512;
/* ATTN_S32=1: the score GEMM widens to f32 (bf16 x bf16 -> f32), P stays bf16. */
static bool S32 = false;
/* ATTN_PITCH_PAD: extra score columns per row beyond the 64-multiple (breaks power-of-two strides). */
static unsigned PITCH_PAD = 0;
/* ATTN_SM_WARP=1: the warp-per-row softmax entries. */
static bool WARP = false;
static const float LOG2E = 1.4426950408889634f;

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    /* round-to-nearest on the dropped 16 mantissa bits */
    uint32_t r = u + 0x8000u;
    return (uint16_t)(r >> 16);
}
static float bf2f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

struct Lt {
    cublasLtHandle_t h;
    void* ws;
    size_t ws_bytes;
    struct Plan {
        cublasLtMatmulDesc_t desc;
        cublasLtMatrixLayout_t a, b, c;
        cublasLtMatmulAlgo_t algo;
    };
    std::map<std::tuple<int, unsigned, unsigned, unsigned>, Plan> plans;

    /* kind 0: S[m][ld] = Q[m][512] . K[n][512]^T ; kind 1: O[m][512] = P[m][ld] . V[n][512] */
    Plan& plan(int kind, unsigned m, unsigned n, unsigned ld) {
        auto key = std::make_tuple(kind, m, n, ld);
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;
        Plan p{};
        LK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        if (kind == 0) {
            cublasOperation_t t = CUBLAS_OP_T;
            LK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t));
            LK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, HD, n, HD));
            LK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, HD, m, HD));
            LK(cublasLtMatrixLayoutCreate(&p.c, S32 ? CUDA_R_32F : CUDA_R_16BF, n, m, ld));
        } else {
            LK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, HD, n, HD));
            LK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, n, m, S32 ? 2 * ld : ld));
            LK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, HD, m, HD));
        }
        cublasLtMatmulPreference_t pref;
        LK(cublasLtMatmulPreferenceCreate(&pref));
        LK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &ws_bytes, sizeof ws_bytes));
        cublasLtMatmulHeuristicResult_t res[4];
        int got = 0;
        LK(cublasLtMatmulAlgoGetHeuristic(h, p.desc, p.a, p.b, p.c, p.c, pref, 4, res, &got));
        cublasLtMatmulPreferenceDestroy(pref);
        if (got == 0) {
            printf("no algorithm kind=%d m=%u n=%u\n", kind, m, n);
            exit(1);
        }
        p.algo = res[0].algo;
        return plans.emplace(key, p).first->second;
    }

    void run(int kind, unsigned m, unsigned n, unsigned ld, float alpha, const void* a,
             const void* b, void* c) {
        Plan& p = plan(kind, m, n, ld);
        const float beta = 0.0f;
        LK(cublasLtMatmul(h, p.desc, &alpha, a, p.a, b, p.b, &beta, c, p.c, c, p.c, &p.algo, ws,
                          ws_bytes, 0));
    }
};

struct Bufs {
    uint16_t *q, *k, *v, *o, *s;
};

/* One chunk-layer: q_rows query rows at absolute positions [kv0, kv0 + q_rows). */
static void route(Lt& lt, const Bufs& d, unsigned q_rows, unsigned kv0, unsigned tile,
                  unsigned grid, cudaEvent_t* ev) {
    for (unsigned q0 = 0; q0 < q_rows; q0 += tile) {
        const unsigned qn = std::min(tile, q_rows - q0);
        const unsigned m = qn * HEADS;
        const unsigned n = kv0 + q0 + qn;
        const unsigned ld = ((n + 63u) & ~63u) + PITCH_PAD;
        const uint16_t* q = d.q + (size_t)q0 * HEADS * HD;
        if (ev) cudaEventRecord(ev[0]);
        lt.run(0, m, n, ld, LOG2E, d.k, q, d.s);
        if (ev) cudaEventRecord(ev[1]);
        PlowAttnSoftmax a{d.s, m, n, ld, HEADS, kv0 + q0 + 1, 0};
        if (WARP && S32)
            plow_attn_softmax_w_f32<<<grid, 256>>>(a);
        else if (WARP)
            plow_attn_softmax_w<<<grid, 256>>>(a);
        else if (S32)
            plow_attn_softmax_f32<<<grid, 256>>>(a);
        else
            plow_attn_softmax<<<grid, 256>>>(a);
        if (ev) cudaEventRecord(ev[2]);
        lt.run(1, m, n, ld, 1.0f, d.v, d.s, d.o + (size_t)q0 * HEADS * HD);
        if (ev) cudaEventRecord(ev[3]);
    }
}

static void fill(std::vector<uint16_t>& x, float amp, unsigned seed) {
    uint32_t s = seed * 2654435761u + 12345u;
    for (auto& e : x) {
        float acc = 0;
        for (int i = 0; i < 4; i++) {
            s = s * 1664525u + 1013904223u;
            acc += (float)(s >> 8) / 16777216.0f - 0.5f;
        }
        e = f2bf(acc * amp);
    }
}

static int check(Lt& lt) {
    int bad = 0;
    const unsigned cases[][3] = {{64, 0, 64}, {37, 100, 16}, {130, 1000, 64}, {96, 29, 96}};
    for (auto& c : cases) {
        const unsigned qr = c[0], kv0 = c[1], tile = c[2], kv = kv0 + qr;
        std::vector<uint16_t> q((size_t)qr * HEADS * HD), k((size_t)kv * HD), v((size_t)kv * HD);
        fill(q, 0.5f, 1);
        fill(k, 0.5f, 2);
        fill(v, 1.0f, 3);
        const unsigned ldmax = (kv + 63u) & ~63u;
        Bufs d;
        CK(cudaMalloc(&d.q, q.size() * 2));
        CK(cudaMalloc(&d.k, k.size() * 2));
        CK(cudaMalloc(&d.v, v.size() * 2));
        CK(cudaMalloc(&d.o, q.size() * 2));
        CK(cudaMalloc(&d.s, (size_t)tile * HEADS * ldmax * 4));
        CK(cudaMemcpy(d.q, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.k, k.data(), k.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d.v, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemset(d.s, 0x7f, (size_t)tile * HEADS * ldmax * 4)); /* NaN-ish garbage */
        route(lt, d, qr, kv0, tile, 132, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> o(q.size());
        CK(cudaMemcpy(o.data(), d.o, o.size() * 2, cudaMemcpyDeviceToHost));
        double err2 = 0, ref2 = 0;
        std::vector<double> sc(kv);
        for (unsigned r = 0; r < qr; r++)
            for (unsigned h = 0; h < HEADS; h++) {
                const uint16_t* qv = &q[((size_t)r * HEADS + h) * HD];
                const unsigned lim = kv0 + r + 1;
                double mx = -1e300;
                for (unsigned j = 0; j < lim; j++) {
                    double s = 0;
                    for (unsigned e = 0; e < HD; e++)
                        s += (double)bf2f(qv[e]) * (double)bf2f(k[(size_t)j * HD + e]);
                    sc[j] = s;
                    mx = std::max(mx, s);
                }
                double den = 0;
                for (unsigned j = 0; j < lim; j++) den += std::exp(sc[j] - mx);
                for (unsigned e = 0; e < HD; e++) {
                    double acc = 0;
                    for (unsigned j = 0; j < lim; j++)
                        acc += std::exp(sc[j] - mx) / den * (double)bf2f(v[(size_t)j * HD + e]);
                    const double got = bf2f(o[((size_t)r * HEADS + h) * HD + e]);
                    err2 += (got - acc) * (got - acc);
                    ref2 += acc * acc;
                }
            }
        const double rel = std::sqrt(err2 / ref2);
        printf("check %s%s q=%u kv0=%u tile=%u: relL2 vs f64 = %.3e\n", S32 ? "s32" : "s16", WARP ? "w" : "", qr,
               kv0, tile, rel);
        bad += !(rel < 2e-2);
        cudaFree(d.q), cudaFree(d.k), cudaFree(d.v), cudaFree(d.o), cudaFree(d.s);
    }
    return bad;
}

int main(int argc, char** argv) {
    S32 = getenv("ATTN_S32") && atoi(getenv("ATTN_S32"));
    PITCH_PAD = getenv("ATTN_PITCH_PAD") ? atoi(getenv("ATTN_PITCH_PAD")) : 0;
    WARP = getenv("ATTN_SM_WARP") && atoi(getenv("ATTN_SM_WARP"));
    Lt lt{};
    LK(cublasLtCreate(&lt.h));
    lt.ws_bytes = 256u << 20;
    CK(cudaMalloc(&lt.ws, lt.ws_bytes));
    if (argc >= 2 && !strcmp(argv[1], "check")) return check(lt);
    if (argc < 6 || strcmp(argv[1], "bench")) {
        printf("usage: %s check | bench q_rows kv_prefix tile grid [reps]\n", argv[0]);
        return 2;
    }
    const unsigned qr = atoi(argv[2]), kv0 = atoi(argv[3]), tile = atoi(argv[4]),
                   grid = atoi(argv[5]);
    const int reps = argc > 6 ? atoi(argv[6]) : 8;
    const unsigned kv = kv0 + qr;
    /* ATTN_SCRATCH_ROWS: size the scratch for this many KV columns (the engine sizes it for max_ctx). */
    const unsigned ldmax = std::max((kv + 63u) & ~63u, (unsigned)atoi(getenv("ATTN_SCRATCH_ROWS") ? getenv("ATTN_SCRATCH_ROWS") : "0")) + PITCH_PAD;
    std::vector<uint16_t> q((size_t)qr * HEADS * HD), k((size_t)kv * HD), v((size_t)kv * HD);
    fill(q, 0.5f, 1);
    fill(k, 0.5f, 2);
    fill(v, 1.0f, 3);
    Bufs d;
    const size_t sbytes = (size_t)std::min(tile, qr) * HEADS * ldmax * (S32 ? 4 : 2);
    CK(cudaMalloc(&d.q, q.size() * 2));
    CK(cudaMalloc(&d.k, k.size() * 2));
    CK(cudaMalloc(&d.v, v.size() * 2));
    CK(cudaMalloc(&d.o, q.size() * 2));
    CK(cudaMalloc(&d.s, sbytes));
    CK(cudaMemcpy(d.q, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.k, k.data(), k.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.v, v.data(), v.size() * 2, cudaMemcpyHostToDevice));

    route(lt, d, qr, kv0, tile, grid, nullptr); /* plans + warm */
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<float> total;
    for (int r = 0; r < reps; r++) {
        usleep(25000);
        cudaEventRecord(e0);
        route(lt, d, qr, kv0, tile, grid, nullptr);
        cudaEventRecord(e1);
        CK(cudaDeviceSynchronize());
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        total.push_back(ms);
    }
    std::sort(total.begin(), total.end());
    /* Phase split of the LAST tile (the widest one). */
    const unsigned q0 = ((qr - 1) / tile) * tile;
    Bufs dl = d;
    cudaEvent_t ev[4];
    for (auto& e : ev) cudaEventCreate(&e);
    float ph[3] = {1e9f, 1e9f, 1e9f};
    for (int r = 0; r < reps; r++) {
        usleep(25000);
        dl.q = d.q + (size_t)q0 * HEADS * HD;
        dl.o = d.o + (size_t)q0 * HEADS * HD;
        route(lt, dl, qr - q0, kv0 + q0, tile, grid, ev);
        CK(cudaDeviceSynchronize());
        for (int i = 0; i < 3; i++) {
            float ms = 0;
            cudaEventElapsedTime(&ms, ev[i], ev[i + 1]);
            ph[i] = std::min(ph[i], ms);
        }
    }
    printf("%s%s q=%u kv0=%u tile=%u grid=%u scratch=%.1f MiB | chunk-layer min %.3f med %.3f ms | "
           "last tile (%u x %u): qk %.3f softmax %.3f pv %.3f ms\n",
           S32 ? "s32" : "s16", WARP ? "w" : "", qr, kv0, tile, grid, sbytes / 1048576.0, total.front(),
           total[total.size() / 2],
           (qr - q0) * HEADS, kv, ph[0], ph[1], ph[2]);
    return 0;
}
