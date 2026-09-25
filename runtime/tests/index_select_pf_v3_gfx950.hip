/* index_select_pf_v3_gfx950.hip — DSA prefill select v3 (PLOW_DSA_SELECT_V3, op 118) and the fused
 * select + union (op 118 writing op 119's table), against the shipped v2 select and
 * d_index_union_pf, in the interpreter geometry (256 workgroups x 512 threads, grid-strided).
 *   - v3 sets == v2 sets per row, == a CPU exact top-k on sampled rows, rows >= n_tok untouched;
 *   - fused union table bytes (+ ticket counter) == d_index_union_pf over v2's selection;
 *   - timings. Tie-heavy / overflowing-bin synthetic rows, plus real scores:
 *     isel_v3 [scores.bin n_tok stride]   (f32 [n_tok][stride] rows of an act.iscore_pf dump)
 *   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
 *       runtime/tests/index_select_pf_v3_gfx950.hip -o isel_v3
 */
#define PLOW_WG_WAVES 8
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef ISEL3_PROF
__device__ unsigned long long g_isel3_prof[16];
#endif
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#include "dsa_select_v3.h"
#endif

#define CK(x) do { hipError_t e_ = (x); if (e_) { printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)
constexpr unsigned TOPK = 2048, NWG = 256, PACK = 8;
#ifndef V3_LDS_WORDS
#define V3_LDS_WORDS 34816u
#endif

extern "C" __global__ __launch_bounds__(512, 1) void k_v2(int* idx, const float* sc, const int* len,
                                                         unsigned n_tok, unsigned stride) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned hist[4096 + 16];
    d_index_select_pf_v2(idx, sc, len, n_tok, TOPK, stride, blockIdx.x, gridDim.x, hist, hist + 4096);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_v3(int* idx, const float* sc, const int* len,
                                                         unsigned n_tok, unsigned stride) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned lds[V3_LDS_WORDS];
    static_assert(isel3::WORDS_FUSED == V3_LDS_WORDS, "LDS");
    d_index_select_pf_v3(idx, sc, len, n_tok, TOPK, stride, blockIdx.x, gridDim.x, lds);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_union(unsigned char* uni, unsigned long long* umask,
                                                            const int* idx, const int* len, unsigned n_tok,
                                                            unsigned stride, unsigned cap) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned sc[PLOW_THREADS + 1];
    d_index_union_pf(uni, umask, idx, len, n_tok, TOPK, stride, cap, PACK, blockIdx.x, gridDim.x, sc, 1u);
#endif
}
extern "C" __global__ __launch_bounds__(512, 1) void k_fused(unsigned char* uni, int* idx, const float* sc,
                                                            const int* len, unsigned n_tok, unsigned stride,
                                                            unsigned cap) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ unsigned lds[V3_LDS_WORDS];
    d_index_select_union_pf(uni, idx, sc, len, n_tok, TOPK, stride, cap, 1u, blockIdx.x, gridDim.x, lds);
#endif
}

static unsigned long long rng = 0x9E3779B97F4A7C15ull;
static unsigned nxt() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 11); }
static float gauss() { float a = 0; for (int i = 0; i < 4; i++) a += (nxt() & 0xFFFFF) / 1048576.0f; return a - 2.0f; }

static float gen(unsigned mode) {
    float v = gauss();
    switch (mode) {
    case 1: return (nxt() % 10) < 4 ? 0.0f : v;                           // relu floor plateau
    case 2: return (float)(int)(v * 8.0f) / 8.0f;                          // coarse: huge boundary ties
    case 3: return 1.0f + v * 1e-5f;                                       // one 12-bit bin: overflow
    case 4: return (nxt() % 4) ? 0.5f : 1.0f;                              // pure ties, index decides
    case 5: return (nxt() % 3) ? ((nxt() & 1) ? -0.0f : 0.0f) : -fabsf(v);  // signed zeros, negatives
    case 6: return 1.0f + (float)(nxt() % 3) * 1e-6f;                      // overflow + ties
    default: return v;
    }
}

struct Dev {
    float* sc = nullptr;
    int *len = nullptr, *a = nullptr, *b = nullptr;
    unsigned char *ua = nullptr, *ub = nullptr;
    unsigned long long* umask = nullptr;
};

static float timeit(void (*launch)(), int reps) {
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    launch();
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(e0));
    for (int i = 0; i < reps; i++) launch();
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms;
    CK(hipEventElapsedTime(&ms, e0, e1));
    return ms / reps;
}

static Dev D;
static unsigned g_ntok, g_stride, g_cap, g_len;
static void l_v2() { hipLaunchKernelGGL(k_v2, dim3(NWG), dim3(512), 0, 0, D.a, D.sc, D.len, g_ntok, g_stride); }
static void l_v3() { hipLaunchKernelGGL(k_v3, dim3(NWG), dim3(512), 0, 0, D.b, D.sc, D.len, g_ntok, g_stride); }
static void l_union() {
    const unsigned n_qt = (g_ntok + PACK - 1) / PACK;
    hipLaunchKernelGGL(k_union, dim3(std::min(NWG, n_qt)), dim3(512), 0, 0, D.ua, D.umask, D.a, D.len,
                       g_ntok, g_stride, g_cap);
}
static void l_fused() {
    const unsigned n_qt = (g_ntok + PACK - 1) / PACK;
    hipLaunchKernelGGL(k_fused, dim3(std::min(NWG, n_qt)), dim3(512), 0, 0, D.ub, D.b, D.sc, D.len, g_ntok,
                       g_stride, g_cap);
}

/* h: [rows][stride] scores. Returns failures. */
static int run_case(const char* tag, const std::vector<float>& h, unsigned rows, unsigned stride, unsigned len,
                    unsigned n_tok, int reps) {
    g_ntok = n_tok, g_stride = stride, g_len = len, g_cap = std::min(PACK * TOPK, stride);
    const unsigned q_pos0 = len - n_tok;
    const unsigned n_qt = (n_tok + PACK - 1) / PACK;
    const size_t hdr = (n_qt * 4u + 255u) / 256u * 256u;
    const size_t ubytes = hdr + (size_t)n_qt * g_cap * 12u + 256;
    CK(hipMalloc(&D.sc, (size_t)rows * stride * 4));
    CK(hipMalloc(&D.len, 4));
    CK(hipMalloc(&D.a, (size_t)rows * TOPK * 4));
    CK(hipMalloc(&D.b, (size_t)rows * TOPK * 4));
    CK(hipMalloc(&D.ua, ubytes));
    CK(hipMalloc(&D.ub, ubytes));
    CK(hipMalloc(&D.umask, (size_t)NWG * stride * 8));
    CK(hipMemcpy(D.sc, h.data(), (size_t)rows * stride * 4, hipMemcpyHostToDevice));
    int ilen = (int)len;
    CK(hipMemcpy(D.len, &ilen, 4, hipMemcpyHostToDevice));
    CK(hipMemset(D.a, 0xF9, (size_t)rows * TOPK * 4));
    CK(hipMemset(D.b, 0xF9, (size_t)rows * TOPK * 4));
    const float t2 = timeit(l_v2, reps), t3 = timeit(l_v3, reps);
    std::vector<int> a((size_t)rows * TOPK), b((size_t)rows * TOPK);
    CK(hipMemcpy(a.data(), D.a, a.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), D.b, b.size() * 4, hipMemcpyDeviceToHost));
    unsigned bad = 0, badref = 0, touched = 0;
    for (unsigned r = 0; r < rows; r++) {
        int* ra = &a[(size_t)r * TOPK];
        int* rb = &b[(size_t)r * TOPK];
        if (r >= n_tok) {
            for (unsigned j = 0; j < TOPK; j++) touched += (ra[j] != (int)0xF9F9F9F9) + (rb[j] != (int)0xF9F9F9F9);
            continue;
        }
        std::sort(ra, ra + TOPK);
        std::sort(rb, rb + TOPK);
        if (memcmp(ra, rb, TOPK * 4)) bad++;
        if (r % 97 == 0 || r + 3 >= n_tok) {
            const unsigned rl = q_pos0 + r + 1;
            std::vector<int> ref;
            if (rl <= TOPK) {
                for (unsigned j = 0; j < TOPK; j++) ref.push_back(j < rl ? (int)j : -1);
            } else {
                std::vector<std::pair<unsigned long long, int>> keys(rl);
                for (unsigned s = 0; s < rl; s++) {
                    unsigned sb;
                    memcpy(&sb, &h[(size_t)r * stride + s], 4);
                    sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
                    keys[s] = {((unsigned long long)sb << 24) | (rl - 1 - s), (int)s};
                }
                std::nth_element(keys.begin(), keys.begin() + TOPK, keys.end(),
                                 [](auto& x, auto& y) { return x.first > y.first; });
                for (unsigned j = 0; j < TOPK; j++) ref.push_back(keys[j].second);
            }
            std::sort(ref.begin(), ref.end());
            if (memcmp(ref.data(), rb, TOPK * 4)) badref++;
        }
    }
    /* union: v2 selection -> d_index_union_pf vs fused, same poison, whole buffer. */
    CK(hipMemset(D.ua, 0xA5, ubytes));
    CK(hipMemset(D.ub, 0xA5, ubytes));
    l_union();
    l_fused();
    CK(hipDeviceSynchronize());
    std::vector<unsigned char> ua(ubytes), ub(ubytes);
    CK(hipMemcpy(ua.data(), D.ua, ubytes, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ub.data(), D.ub, ubytes, hipMemcpyDeviceToHost));
    const size_t ctr = hdr + (size_t)n_qt * g_cap * 12u;
    const unsigned ctr_bad = ((unsigned*)(ub.data() + ctr))[0] != 0u || ((unsigned*)(ub.data() + ctr))[1] != 0u;
    size_t ubad = 0, first = ~(size_t)0, entries = 0;
    for (size_t i = 0; i < ubytes; i++)
        if (ua[i] != ub[i]) { ubad++; if (first == ~(size_t)0) first = i; }
    for (unsigned q = 0; q < n_qt; q++) entries += ((unsigned*)ua.data())[q];
    const float tu = timeit(l_union, reps), tf = timeit(l_fused, reps);
    printf("%-10s T=%5u len=%5u  v2 %7.1f us  v3 %7.1f us (%.2fx) | v2+union %7.1f  fused %7.1f us (%.2fx) | "
           "rows-differ %u cpu-bad %u touched %u union-bytes-differ %zu (first %zd) ctr %u entries %zu\n",
           tag, n_tok, len, t2 * 1e3, t3 * 1e3, t2 / t3, (t2 + tu) * 1e3, tf * 1e3, (t2 + tu) / tf, bad, badref,
           touched, ubad, ubad ? (ssize_t)first : (ssize_t)-1, ctr_bad, entries);
    CK(hipFree(D.sc)); CK(hipFree(D.len)); CK(hipFree(D.a)); CK(hipFree(D.b));
    CK(hipFree(D.ua)); CK(hipFree(D.ub)); CK(hipFree(D.umask));
    return (int)(bad + badref + touched + (ubad != 0) + ctr_bad);
}

int main(int argc, char** argv) {
    int fails = 0;
    if (argc >= 4) {
        /* real scores: rows [n_tok][stride] f32 (the act.iscore_pf dump), kv_len = n_tok */
        const unsigned n_tok = (unsigned)atoi(argv[2]), stride = (unsigned)atoi(argv[3]);
        std::vector<float> h((size_t)n_tok * stride);
        FILE* f = fopen(argv[1], "rb");
        if (!f || fread(h.data(), 4, h.size(), f) != h.size()) { printf("read %s failed\n", argv[1]); return 2; }
        fclose(f);
        fails += run_case("real", h, n_tok, stride, n_tok, n_tok, 10);
        printf(fails ? "FAIL\n" : "PASS\n");
        return fails != 0;
    }
    constexpr unsigned S = 16384;
    if (argc == 2 && !strcmp(argv[1], "quick")) {
        std::vector<float> h((size_t)S * S);
        for (size_t r = 0; r < S; r++)
            for (size_t c = 0; c < S; c++) h[r * S + c] = gen(0);
        for (unsigned T : {4096u, 8192u, 16384u}) {
#ifdef ISEL3_PROF
            unsigned long long z[16] = {};
            CK(hipMemcpyToSymbol(HIP_SYMBOL(g_isel3_prof), z, sizeof(z)));
#endif
            fails += run_case("gauss", h, S, S, T, T, 10);
#ifdef ISEL3_PROF
            unsigned long long pr[16];
            CK(hipMemcpyFromSymbol(pr, HIP_SYMBOL(g_isel3_prof), sizeof(pr)));
            const double n = pr[6] ? (double)pr[6] : 1.0;
            printf("prof/row (%llu rows): hist %.0f find %.0f compact %.0f refine %.0f fetch %.0f row %.0f\n",
                   pr[6], pr[0] / n, pr[1] / n, pr[2] / n, pr[3] / n, pr[4] / n, pr[5] / n);
            printf("  compact: masks %.0f alloc %.0f emit %.0f | refine: passes %.0f rank %.0f\n", pr[8] / n,
                   pr[9] / n, pr[10] / n, pr[11] / n, pr[12] / n);
#endif
        }
        return 0;
    }
    {
        std::vector<float> h((size_t)S * S);
        for (size_t r = 0; r < S; r++)
            for (size_t c = 0; c < S; c++) h[r * S + c] = gen(0);
        for (unsigned T : {4096u, 8192u, 16384u}) fails += run_case("gauss", h, S, S, T, T, 10);
        for (size_t r = 0; r < S; r++) {
            const unsigned mode = r % 7;
            for (size_t c = 0; c < S; c++) h[r * S + c] = gen(mode);
        }
        for (unsigned T : {4096u, 8192u, 16384u, 16376u}) fails += run_case("mixed", h, S, S, T, T, 5);
        fails += run_case("offset", h, S, S, 8192u, 8000u, 3);
        fails += run_case("ragged", h, S, S, 5003u, 5003u, 3);
    }
    {
        /* Causal bound past the LDS mask: select-to-idx + windowed union. */
        constexpr unsigned L = 20480, R = 64;
        std::vector<float> h((size_t)R * L);
        for (size_t r = 0; r < R; r++)
            for (size_t c = 0; c < L; c++) h[r * L + c] = gen(r % 7);
        fails += run_case("long", h, R, L, L, R, 2);
        fails += run_case("long-edge", h, R, L, 16384u + 5u, 60u, 2);
    }
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails != 0;
}
