/* sample_topk_test.cu — SampleSmallF32 / SampleFishBf16 (op_elementwise.cuh d_sample_small /
 * d_sample_fish): the radix top-k selection against the K-round block_max selection it replaced,
 * which the same functions still run when the radix scratch does not fit the launch's shared
 * memory. Each case runs twice on identical inputs: once with room for the scratch (radix) and once
 * with exactly the row's arena (the fallback), and every sampled code must be bit-identical.
 *
 * Inputs: recorded Qwen3-TTS logits when given (argv[1] = depth logits `[rows][2048]` f32, argv[2]
 * = backbone logits `[rows][2049]` f32), plus synthetic rows (ties, -inf runs, vocab 64..16384).
 * Sampling: temperatures, top_k 1..64, top_p, repetition-penalty bitmap, min-step stop mask, CFG
 * guide rows, seeds and steps.
 *
 * Build: nvcc -arch=sm_90a -DPLOW_NV_GUMBEL_WARP=1 -O3 -I runtime/common -I runtime/nvidia runtime/tests/sample_topk_test.cu
 *   -o sample_topk_test;  sample_topk_test [depth.f32 backbone.f32] [--bench]
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sm120_common.cuh"
#include "op_elementwise.cuh"

#define CK(x)                                                                       \
    do {                                                                            \
        cudaError_t e_ = (x);                                                       \
        if (e_ != cudaSuccess) {                                                    \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(2);                                                                \
        }                                                                           \
    } while (0)

struct SmallArgs {
    unsigned* out;
    const float* logits;
    const float* params;
    const unsigned* ctl;
    const unsigned long long* seed;
    unsigned* bitmap;
    unsigned rows, vocab, words, min_step;
};

__global__ void small_kernel(SmallArgs a, int* radix_used) {
    extern __shared__ float arena[];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float* L = arena + 3 * PLOW_NV_WARPS + 128;
        unsigned long long* b = (unsigned long long*)(((size_t)(L + a.vocab) + 15u) & ~(size_t)15u);
        *radix_used = smem_fits((unsigned*)(b + 64) + 4 + 256);
    }
    d_sample_small(a.out, a.logits, a.params, a.ctl, a.seed, a.bitmap, nullptr, nullptr, a.rows, a.vocab, 16u, 0u,
                   3u, a.words, 0u, 0u, a.min_step, blockIdx.x, gridDim.x, arena);
}

struct FishArgs {
    unsigned* out;
    const __nv_bfloat16* logits;
    const float* params;
    const unsigned* topk;
    const unsigned long long* seed;
    const unsigned* step;
    unsigned* hist;
    unsigned rows, vocab, window, ras_limit;
};

__global__ void fish_kernel(FishArgs a, int* radix_used) {
    extern __shared__ float arena[];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned short* L = (unsigned short*)(arena + 3 * PLOW_NV_WARPS + 128);
        unsigned long long* b = (unsigned long long*)(((size_t)(L + a.vocab) + 15u) & ~(size_t)15u);
        *radix_used = smem_fits((unsigned*)(b + 64) + 4 + 256);
    }
    d_sample_fish(a.out, a.logits, a.params, a.topk, a.seed, a.step, a.hist, nullptr, a.rows, a.vocab, 16u, 0u, 0u,
                  a.window, a.window ? 1u : 0u, 0u, a.ras_limit, blockIdx.x, gridDim.x, arena);
}

/* Shared bytes the K-round selection needs: reduction slots, 64 classes + values, the row. */
static size_t row_arena(unsigned vocab, size_t elem) {
    return (3 * PLOW_NV_WARPS + 128) * 4 + vocab * elem;
}
static size_t radix_arena(unsigned vocab, size_t elem) {
    return ((row_arena(vocab, elem) + 15) & ~(size_t)15) + 64 * 8 + 260 * 4;
}

static uint64_t rng_state = 0x2545F4914F6CDD1Dull;
static uint32_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 11);
}
static float frand() { return (rnd() & 0xffffff) / 16777216.0f; }

template <class T>
static T* dev(const std::vector<T>& h) {
    T* d;
    CK(cudaMalloc(&d, h.size() * sizeof(T) + 16));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static int failures = 0, cases = 0, rows_checked = 0;
static float bench_ms[2][2];

/* One SampleSmallF32 case on `rows` rows of `vocab` logits: radix vs fallback, bit-exact codes. */
static void small_case(const char* name, const std::vector<float>& logits, unsigned rows, unsigned vocab,
                       bool bench = false) {
    const unsigned words = (vocab + 31) / 32;
    std::vector<float> params(rows * 4);
    std::vector<unsigned> ctl(rows * 2);
    std::vector<unsigned long long> seed(rows);
    std::vector<unsigned> bitmap(rows * words);
    const unsigned min_step = 2;
    for (unsigned r = 0; r < rows; r++) {
        const float temps[] = {0.f, 0.3f, 0.9f, 1.0f, 1.7f};
        const float tops[] = {1.0f, 0.95f, 0.8f, 0.3f};
        const unsigned ks[] = {1, 2, 30, 50, 63, 64};
        params[r * 4 + 0] = temps[rnd() % 5];
        params[r * 4 + 1] = tops[rnd() % 4];
        params[r * 4 + 2] = 1.0f + 0.25f * frand();
        params[r * 4 + 3] = 1.0f + 2.0f * frand();
        unsigned k = ks[rnd() % 6];
        /* Every 7th row a CFG guide over (r, r + 1); its follower then skips. */
        if (r % 7 == 0 && r + 1 < rows) {
            ctl[2 * r] = k | (1u << 31);
            ctl[2 * (r + 1)] = k | (1u << 30);
        } else if (ctl[2 * r] == 0) {
            ctl[2 * r] = k;
        }
        ctl[2 * r + 1] = rnd() % 4; /* steps 0..3: rows below min_step mask the stop class */
        seed[r] = ((unsigned long long)rnd() << 32) | rnd();
        for (unsigned w = 0; w < words; w++) bitmap[r * words + w] = (r % 3 == 0) ? (rnd() & rnd()) : 0u;
    }
    float* d_logits = dev(logits);
    float* d_params = dev(params);
    unsigned* d_ctl = dev(ctl);
    unsigned long long* d_seed = dev(seed);
    std::vector<unsigned> outs[2];
    int* d_used;
    CK(cudaMalloc(&d_used, 4));
    int used[2];
    for (int mode = 0; mode < 2; mode++) {
        unsigned* d_bitmap = dev(bitmap);
        std::vector<unsigned> init(rows * 16, 0xdeadbeefu);
        unsigned* d_out = dev(init);
        const size_t smem = mode == 0 ? radix_arena(vocab, 4) : row_arena(vocab, 4);
        CK(cudaFuncSetAttribute(small_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        SmallArgs a{d_out, d_logits, d_params, d_ctl, d_seed, d_bitmap, rows, vocab, words, min_step};
        small_kernel<<<132, PLOW_NV_THREADS, smem>>>(a, d_used);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&used[mode], d_used, 4, cudaMemcpyDeviceToHost));
        outs[mode].resize(rows * 16);
        CK(cudaMemcpy(outs[mode].data(), d_out, rows * 16 * 4, cudaMemcpyDeviceToHost));
        if (bench) {
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0));
            for (int it = 0; it < 50; it++) small_kernel<<<132, PLOW_NV_THREADS, smem>>>(a, d_used);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            bench_ms[0][mode] = ms / 50;
        }
        CK(cudaFree(d_bitmap));
        CK(cudaFree(d_out));
    }
    cases++;
    rows_checked += rows;
    int bad = 0;
    for (unsigned r = 0; r < rows; r++) {
        if (outs[0][r * 16] != outs[1][r * 16]) {
            if (bad < 4)
                printf("  MISMATCH %s row %u: radix %u fallback %u (k %u t %.2f p %.2f)\n", name, r, outs[0][r * 16],
                       outs[1][r * 16], ctl[2 * r] & 0x3fffffffu, params[r * 4], params[r * 4 + 1]);
            bad++;
        }
    }
    if (used[0] != 1 || used[1] != 0) {
        printf("  %s: radix path not exercised as intended (radix run %d, fallback run %d)\n", name, used[0], used[1]);
        bad++;
    }
    printf("%-34s SampleSmallF32 rows %4u vocab %5u : %s\n", name, rows, vocab, bad ? "FAIL" : "ok");
    failures += bad != 0;
    CK(cudaFree(d_logits));
    CK(cudaFree(d_params));
    CK(cudaFree(d_ctl));
    CK(cudaFree(d_seed));
    CK(cudaFree(d_used));
}

static void fish_case(const char* name, const std::vector<float>& logits, unsigned rows, unsigned vocab,
                      bool bench = false) {
    std::vector<__nv_bfloat16> lb(logits.size());
    for (size_t i = 0; i < logits.size(); i++) lb[i] = __float2bfloat16_rn(logits[i]);
    std::vector<float> params(rows * 4);
    std::vector<unsigned> topk(rows), step(rows);
    std::vector<unsigned long long> seed(rows);
    const unsigned window = 10;
    std::vector<unsigned> hist(rows * window);
    for (unsigned r = 0; r < rows; r++) {
        const unsigned ks[] = {1, 2, 30, 50, 63, 64};
        params[r * 4 + 0] = 0.2f + frand();
        params[r * 4 + 1] = (r & 1) ? 1.0f : 0.7f + 0.3f * frand();
        params[r * 4 + 2] = 0.2f + frand();
        params[r * 4 + 3] = 0.9f;
        topk[r] = ks[rnd() % 6];
        step[r] = rnd() % 20;
        seed[r] = ((unsigned long long)rnd() << 32) | rnd();
        for (unsigned w = 0; w < window; w++) hist[r * window + w] = (rnd() % 3 == 0) ? 1 + rnd() % 40 : 0;
    }
    __nv_bfloat16* d_logits = dev(lb);
    float* d_params = dev(params);
    unsigned* d_topk = dev(topk);
    unsigned* d_step = dev(step);
    unsigned long long* d_seed = dev(seed);
    std::vector<unsigned> outs[2];
    int* d_used;
    CK(cudaMalloc(&d_used, 4));
    int used[2];
    for (int mode = 0; mode < 2; mode++) {
        unsigned* d_hist = dev(hist);
        std::vector<unsigned> init(rows * 16, 0xdeadbeefu);
        unsigned* d_out = dev(init);
        const size_t smem = mode == 0 ? radix_arena(vocab, 2) : row_arena(vocab, 2);
        CK(cudaFuncSetAttribute(fish_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        FishArgs a{d_out, d_logits, d_params, d_topk, d_seed, d_step, d_hist, rows, vocab, window, vocab};
        fish_kernel<<<132, PLOW_NV_THREADS, smem>>>(a, d_used);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&used[mode], d_used, 4, cudaMemcpyDeviceToHost));
        outs[mode].resize(rows * 16);
        CK(cudaMemcpy(outs[mode].data(), d_out, rows * 16 * 4, cudaMemcpyDeviceToHost));
        if (bench) {
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0));
            for (int it = 0; it < 50; it++) fish_kernel<<<132, PLOW_NV_THREADS, smem>>>(a, d_used);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            bench_ms[1][mode] = ms / 50;
        }
        CK(cudaFree(d_hist));
        CK(cudaFree(d_out));
    }
    cases++;
    rows_checked += rows;
    int bad = 0;
    for (unsigned r = 0; r < rows; r++)
        if (outs[0][r * 16] != outs[1][r * 16]) {
            if (bad < 4)
                printf("  MISMATCH %s row %u: radix %u fallback %u (k %u)\n", name, r, outs[0][r * 16],
                       outs[1][r * 16], topk[r]);
            bad++;
        }
    if (used[0] != 1 || used[1] != 0) {
        printf("  %s: radix path not exercised as intended (radix run %d, fallback run %d)\n", name, used[0], used[1]);
        bad++;
    }
    printf("%-34s SampleFishBf16 rows %4u vocab %5u : %s\n", name, rows, vocab, bad ? "FAIL" : "ok");
    failures += bad != 0;
    CK(cudaFree(d_logits));
    CK(cudaFree(d_params));
    CK(cudaFree(d_topk));
    CK(cudaFree(d_step));
    CK(cudaFree(d_seed));
    CK(cudaFree(d_used));
}

static std::vector<float> synthetic(unsigned rows, unsigned vocab, int kind) {
    std::vector<float> v(size_t(rows) * vocab);
    for (unsigned r = 0; r < rows; r++)
        for (unsigned i = 0; i < vocab; i++) {
            float x = 6.0f * frand() - 3.0f;
            if (kind == 1) x = (float)(rnd() % 5) * 0.5f;              /* heavy ties */
            if (kind == 2 && (i % 3)) x = -INFINITY;                     /* mostly masked */
            if (kind == 3 && i >= 20) x = -INFINITY;                     /* fewer finite than top_k */
            if (kind == 4 && i == r % vocab) x = 40.0f;                  /* one dominant class */
            if (kind == 5) x = (i & 1) ? -0.0f : 0.0f;                   /* signed zeros */
            v[size_t(r) * vocab + i] = x;
        }
    return v;
}

static std::vector<float> load(const char* path, unsigned vocab, unsigned* rows) {
    std::vector<float> v;
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f) / 4;
    fseek(f, 0, SEEK_SET);
    *rows = (unsigned)(n / vocab);
    v.resize(size_t(*rows) * vocab);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(2);
    fclose(f);
    return v;
}

/* gumbel_pick_warp against the serial strict-> scan it replaced: one block per case, scores with
 * ties (a 1/16 grid), -inf and NaN runs, n = 0..64. */
__global__ void gumbel_kernel(const float* sv, const unsigned* n, unsigned* got, unsigned* ref) {
    const float* s = sv + (size_t)blockIdx.x * 64u;
    const unsigned step = blockIdx.x * 7u, coord = 3u;
    const unsigned long long seed = 0x9e3779b97f4a7c15ull ^ blockIdx.x;
    if (threadIdx.x < 32u) {
        const unsigned g = gumbel_pick_warp(s, n[blockIdx.x], seed, step, coord);
        if (threadIdx.x == 0) got[blockIdx.x] = g;
    }
    if (threadIdx.x == 0) {
        unsigned best = 0;
        float bv = -INFINITY;
        for (unsigned i = 0; i < n[blockIdx.x]; i++) {
            const float v = s[i] + sf_neg_log_exp1(seed, step, coord, 0u, i);
            if (v > bv) { bv = v; best = i; }
        }
        ref[blockIdx.x] = best;
    }
}
static int gumbel_case() {
    const unsigned cases = 4096;
    std::vector<float> sv((size_t)cases * 64);
    std::vector<unsigned> n(cases);
    unsigned rng = 0x2468ace1u;
    auto next = [&] { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
    for (unsigned c = 0; c < cases; c++) {
        n[c] = c % 65u;
        const unsigned kind = (c / 65u) % 5u;
        for (unsigned i = 0; i < 64; i++) {
            float v = (float)(int)(next() % 64u) / 16.f - 2.f;
            if (kind == 1) v = next() % 2u ? 1e30f : 0.f;           /* ties: the noise vanishes at 1e30 */
            if (kind == 2 && next() % 3u == 0) v = -INFINITY;
            if (kind == 3) v = -INFINITY;
            if (kind == 4 && next() % 4u == 0) v = NAN;
            sv[(size_t)c * 64 + i] = v;
        }
    }
    float* d_sv; unsigned *d_n, *d_got, *d_ref;
    CK(cudaMalloc(&d_sv, sv.size() * 4)); CK(cudaMalloc(&d_n, cases * 4));
    CK(cudaMalloc(&d_got, cases * 4)); CK(cudaMalloc(&d_ref, cases * 4));
    CK(cudaMemcpy(d_sv, sv.data(), sv.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n, n.data(), cases * 4, cudaMemcpyHostToDevice));
    gumbel_kernel<<<cases, 64>>>(d_sv, d_n, d_got, d_ref);
    CK(cudaDeviceSynchronize());
    std::vector<unsigned> got(cases), ref(cases);
    CK(cudaMemcpy(got.data(), d_got, cases * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), d_ref, cases * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (unsigned c = 0; c < cases; c++) bad += got[c] != ref[c];
    printf("%-34s gumbel_pick_warp vs serial scan, %u cases : %s\n", "ties / -inf / NaN, n 0..64", cases,
           bad ? "FAIL" : "ok");
    CK(cudaFree(d_sv)); CK(cudaFree(d_n)); CK(cudaFree(d_got)); CK(cudaFree(d_ref));
    return bad;
}

/* d_gemv_f32out with x staged in the arena (room for it) vs read from global (no arena): the same
 * bits. Shapes: depth heads M = 1..32 x 2048 x 1024, a narrow N that leaves blocks empty, both
 * roundings, and an arena one row short of x (falls back). */
__global__ void gemv_kernel(float* C, const __nv_bfloat16* x, const __nv_bfloat16* W, unsigned M, unsigned N,
                            unsigned K, bool round_bf16) {
    extern __shared__ float arena[];
    d_gemv_f32out(C, x, W, M, N, K, round_bf16, blockIdx.x, gridDim.x, arena);
}
static int gemv_case() {
    struct G { unsigned M, N, K; bool rnd; size_t smem_rows; };
    const G gs[] = {{1, 2048, 1024, false, 1}, {7, 2048, 1024, false, 7}, {16, 2048, 1024, false, 16},
                    {32, 2048, 1024, false, 32}, {16, 2048, 1024, true, 16}, {9, 100, 1024, false, 9},
                    {16, 2048, 1024, false, 15}};
    CK(cudaFuncSetAttribute(gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    int bad_total = 0;
    unsigned rng = 0x1234567u;
    auto next = [&] { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
    for (const G& g : gs) {
        std::vector<__nv_bfloat16> x((size_t)g.M * g.K), w((size_t)g.N * g.K);
        for (auto& v : x) v = __float2bfloat16((float)(int)(next() % 2001u - 1000u) / 997.f);
        for (auto& v : w) v = __float2bfloat16((float)(int)(next() % 2001u - 1000u) / 31337.f);
        __nv_bfloat16 *dx, *dw; float *c0, *c1;
        CK(cudaMalloc(&dx, x.size() * 2)); CK(cudaMalloc(&dw, w.size() * 2));
        CK(cudaMalloc(&c0, (size_t)g.M * g.N * 4)); CK(cudaMalloc(&c1, (size_t)g.M * g.N * 4));
        CK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
        gemv_kernel<<<132, PLOW_NV_THREADS, g.smem_rows * g.K * 2>>>(c0, dx, dw, g.M, g.N, g.K, g.rnd);
        gemv_kernel<<<132, PLOW_NV_THREADS, 0>>>(c1, dx, dw, g.M, g.N, g.K, g.rnd);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> a((size_t)g.M * g.N), b(a.size());
        CK(cudaMemcpy(a.data(), c0, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), c1, b.size() * 4, cudaMemcpyDeviceToHost));
        const int bad = a != b;
        printf("%-34s GemvF32Out M %2u N %4u arena rows %2zu : %s\n", g.rnd ? "staged vs global, bf16-rounded"
               : "staged vs global", g.M, g.N, g.smem_rows, bad ? "FAIL" : "ok");
        bad_total += bad;
        CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(c0)); CK(cudaFree(c1));
    }
    return bad_total;
}

int main(int argc, char** argv) {
    if (argc == 2 && !strcmp(argv[1], "--undersized-arena")) {
        SmallArgs a{};
        a.rows = 1;
        a.vocab = 2048;
        int* marker;
        CK(cudaMalloc(&marker, sizeof(int)));
        small_kernel<<<1, 256, row_arena(a.vocab, sizeof(float)) - 16>>>(a, marker);
        const auto error = cudaDeviceSynchronize();
        if (error != cudaErrorIllegalInstruction && error != cudaErrorLaunchFailure) {
            fprintf(stderr, "FAIL undersized arena: %s\n", cudaGetErrorString(error));
            return 1;
        }
        printf("PASS undersized sampler arena refused before operand access\n");
        return 0;
    }

    bool bench = false;
    std::vector<const char*> files;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench"))
            bench = true;
        else
            files.push_back(argv[i]);
    }
    if (files.size() >= 1) {
        unsigned rows;
        std::vector<float> d = load(files[0], 2048, &rows);
        for (int rep = 0; rep < 4; rep++) {
            small_case("recorded depth logits (2048)", d, rows, 2048, bench && rep == 0);
            fish_case("recorded depth logits (2048)", d, rows, 2048, bench && rep == 0);
        }
    }
    if (files.size() >= 2) {
        unsigned rows;
        std::vector<float> b = load(files[1], 2049, &rows);
        for (int rep = 0; rep < 8; rep++) small_case("recorded backbone logits (2049)", b, rows, 2049);
    }
    const unsigned vocabs[] = {64, 100, 2048, 2049, 3076, 4097, 8192, 16384};
    for (unsigned vocab : vocabs)
        for (int kind = 0; kind < 6; kind++) {
            char name[64];
            snprintf(name, sizeof name, "synthetic kind %d", kind);
            const unsigned rows = vocab >= 8192 ? 64 : 256;
            std::vector<float> v = synthetic(rows, vocab, kind);
            small_case(name, v, rows, vocab);
            fish_case(name, v, rows, vocab);
        }
    if (gumbel_case()) failures++;
    if (gemv_case()) failures++;
    cases += 2;
    if (bench)
        printf("bench (recorded depth rows, us/launch): SampleSmallF32 radix %.1f K-round %.1f; SampleFishBf16 radix "
               "%.1f K-round %.1f\n",
               bench_ms[0][0] * 1e3, bench_ms[0][1] * 1e3, bench_ms[1][0] * 1e3, bench_ms[1][1] * 1e3);
    printf("%s: %d cases, %d rows, %d failing\n", failures ? "FAIL" : "PASS", cases, rows_checked, failures);
    return failures != 0;
}
