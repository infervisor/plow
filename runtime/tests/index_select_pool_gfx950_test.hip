/* index_select_pool_gfx950_test.hip — pool_size fix for op 59 IndexSelect
 * (d_index_select_coop) and op 118 IndexSelectPf (d_index_select_pf).
 *
 * Bug: both kernels read their `kv_len` operand directly as the score-array-length bound
 * with no division. `IndexScoreKpool` (op 127, already fixed+verified) only writes valid
 * pool scores for `p < kv_len/pool_size` — feeding raw token-granular `kv_len` into these
 * selectors would scan/select from the region past that boundary, which is stale/
 * uninitialised (or, in this test, deliberately POISONED with a value that would win
 * top-k if wrongly included) — the exact documented historical bug at
 * op_attention.h:4477-4491 ("attention read latent rows that were never written...
 * DETERMINISTIC, not a race"), reproduced here on purpose to prove the fix excludes it.
 *
 * Fix: both kernels take a `pool_size` parameter (DEFAULT 1, so every existing call site
 * — including this file's own CASE 1/3 below — compiles and behaves byte-identically
 * unchanged) and divide the live token-granular occupancy down internally before it
 * bounds anything.
 *
 * Three cases, each on real gfx950 hardware:
 *   1. d_index_select_coop, pool_size=1 (explicit, the default every existing call site
 *      gets implicitly): plain top-k over a token-granular score array, matches a CPU
 *      reference top-k exactly — the byte-identical-at-default regression proof.
 *   2. d_index_select_coop, pool_size=4, kv_len NOT a multiple of 4 (kv_len=4*L+2, so
 *      nkv = kv_len/4 = L): the poisoned tail at index >= L must NEVER be selected, and
 *      the selected set must equal case 1's set over the same L valid entries.
 *   3. d_index_select_pf, pool_size=4, 2 rows each with their own row-relative poisoned
 *      tail beyond their own row's pool boundary: must never be selected. `pool_size=1`
 *      is not separately re-tested here — the fix is the identical `x/pool_size` idiom
 *      case 1/2 already prove at pool_size=1 (a compile-time-obvious `x/1==x` identity,
 *      not a second code path), and case 1's own equivalence already covers it.
 *
 * Two passes, the op bodies are device-only by include chain:
 *   hipcc --offload-arch=gfx950 -O3 -w -DISP_DEVICE --genco \
 *       runtime/tests/index_select_pool_gfx950_test.hip -o /tmp/isp.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/index_select_pool_gfx950_test.hip -o /tmp/isp -L/opt/rocm/lib -lamdhip64
 *   sg render -c "perf-data/tools/gpulease -n 1 isp /tmp/isp /tmp/isp.co"
 * (the .co path is argv[1], defaulting to /tmp/isp.co if omitted.)
 */

#define ISP_NCU 4u /* small cooperative grid: this test's len_max is tiny */

#ifdef ISP_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_gemm.h"
#include "op_norm.h"
#include "op_attention.h"

struct CoopArgs {
    unsigned long long idx, score, ghist, gctl, kv_len;
    unsigned len_max, top_k, pool_size;
};
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void k_select_coop(CoopArgs* a) {
    __shared__ unsigned lh[SEL_NB];
    __shared__ unsigned red[4];
    d_index_select_coop<true, true, true>((int*)a->idx, (const float*)a->score, a->len_max,
                                          a->top_k, (unsigned*)a->ghist, (unsigned*)a->gctl,
                                          blockIdx.x, gridDim.x, lh, red,
                                          (const int*)a->kv_len, a->pool_size);
}

struct PfArgs {
    unsigned long long idx, score, kv_len;
    unsigned n_tok, top_k, kv_stride, pool_size;
};
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void k_select_pf(PfArgs* a) {
    __shared__ unsigned lh[SEL_NB];
    __shared__ unsigned red[4];
    d_index_select_pf<true>((int*)a->idx, (const float*)a->score, (const int*)a->kv_len,
                            a->n_tok, a->top_k, a->kv_stride, blockIdx.x, gridDim.x, lh, red,
                            a->pool_size);
}

#else
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

typedef unsigned short bf16;
#define PLOW_THREADS 512u /* amd_common.h:1114 — 8 waves, 2 per SIMD; must match the device build */
#define HC(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return 1; }        \
    } while (0)
/* Same check, `void`-context: for use inside a lambda whose other paths fall off the end
 * (== an implicit `return;`) — `HC`'s `return 1;` there would mismatch the lambda's
 * deduced return type and fall off the end as undefined behavior. */
#define HCV(x)                                                                                   \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return; }          \
    } while (0)

struct CoopArgs {
    unsigned long long idx, score, ghist, gctl, kv_len;
    unsigned len_max, top_k, pool_size;
};
struct PfArgs {
    unsigned long long idx, score, kv_len;
    unsigned n_tok, top_k, kv_stride, pool_size;
};

/* CPU reference top-k: highest `top_k` scores, lowest-index tie-break, over [0, len). */
static std::vector<int> ref_topk(const std::vector<float>& sc, unsigned len, unsigned top_k) {
    std::vector<int> order;
    for (unsigned i = 0; i < len; i++) order.push_back((int)i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (sc[a] != sc[b]) return sc[a] > sc[b];
        return a < b;
    });
    unsigned k = std::min<unsigned>(top_k, len);
    std::vector<int> out(order.begin(), order.begin() + k);
    std::sort(out.begin(), out.end()); /* order-blind emit -> compare as sets */
    return out;
}

static std::vector<int> got_set(const int* idx, unsigned n, unsigned poison_from) {
    std::vector<int> v;
    for (unsigned i = 0; i < n; i++)
        if (idx[i] >= 0) v.push_back(idx[i]);
    std::sort(v.begin(), v.end());
    bool poisoned = false;
    for (int x : v)
        if ((unsigned)x >= poison_from) poisoned = true;
    if (poisoned) printf("  POISONED TAIL SELECTED (bug reproduced)\n");
    return v;
}

int main(int argc, char** argv) {
    const char* co_path = argc > 1 ? argv[1] : "/tmp/isp.co";
    hipModule_t mod;
    HC(hipModuleLoad(&mod, co_path));
    hipFunction_t f_coop, f_pf;
    HC(hipModuleGetFunction(&f_coop, mod, "k_select_coop"));
    HC(hipModuleGetFunction(&f_pf, mod, "k_select_pf"));

    int bad = 0;

    /* ---- Case 1 + 2: d_index_select_coop ---- */
    {
        const unsigned L = 5, POOL = 4, LEN_MAX = 32, TOPK = 3;
        std::vector<float> sc(LEN_MAX, 0.0f);
        /* valid region [0,L): distinct real scores */
        float vals[5] = {2.0f, 5.0f, 1.0f, 4.0f, 3.0f};
        for (unsigned i = 0; i < L; i++) sc[i] = vals[i];
        /* poisoned tail [L, LEN_MAX): +1e30, would dominate top-k if ever scanned */
        for (unsigned i = L; i < LEN_MAX; i++) sc[i] = 1.0e30f;
        auto want = ref_topk(sc, L, TOPK);

        float *d_sc; int *d_idx, *d_kv; unsigned *d_gh, *d_gc;
        HC(hipMalloc(&d_sc, LEN_MAX * 4));
        HC(hipMalloc(&d_idx, TOPK * 4));
        HC(hipMalloc(&d_kv, 4));
        HC(hipMalloc(&d_gh, 7 * 256 * 4));
        HC(hipMalloc(&d_gc, 3 * 4));
        HC(hipMemcpy(d_sc, sc.data(), LEN_MAX * 4, hipMemcpyHostToDevice));

        auto run_coop = [&](unsigned kv_len_val, unsigned pool_size, unsigned len_max,
                             std::vector<int>& out) {
            HCV(hipMemset(d_gh, 0, 7 * 256 * 4));
            HCV(hipMemset(d_gc, 0, 3 * 4));
            HCV(hipMemset(d_idx, 0xFF, TOPK * 4));
            int kvh = (int)kv_len_val;
            HCV(hipMemcpy(d_kv, &kvh, 4, hipMemcpyHostToDevice));
            CoopArgs a{(unsigned long long)d_idx, (unsigned long long)d_sc,
                       (unsigned long long)d_gh,  (unsigned long long)d_gc,
                       (unsigned long long)d_kv,  len_max, TOPK, pool_size};
            CoopArgs* da;
            HCV(hipMalloc(&da, sizeof(a)));
            HCV(hipMemcpy(da, &a, sizeof(a), hipMemcpyHostToDevice));
            void* kp[] = {&da};
            HCV(hipModuleLaunchKernel(f_coop, ISP_NCU, 1, 1, PLOW_THREADS, 1, 1, 0, 0, kp, nullptr));
            HCV(hipDeviceSynchronize());
            std::vector<int> h(TOPK);
            HCV(hipMemcpy(h.data(), d_idx, TOPK * 4, hipMemcpyDeviceToHost));
            HCV(hipFree(da));
            out = got_set(h.data(), TOPK, L);
        };

        std::vector<int> got1, got2;
        /* case 1: pool_size default(1 via explicit 1), kv_len == L exactly (today's shape) */
        run_coop(L, 1u, L, got1);
        printf("coop pool_size=1: got=[");
        for (int x : got1) printf("%d ", x);
        printf("] want=[");
        for (int x : want) printf("%d ", x);
        printf("]\n");
        if (got1 != want) { printf("  MISMATCH\n"); bad++; }

        /* case 2: pool_size=4, kv_len = 4*L+2 (non-pool-aligned) -> nkv = L exactly,
         * but len_max is now LEN_MAX (the pool-granular packet ctx) so a broken kernel
         * would scan straight into the poisoned tail. */
        run_coop(POOL * L + 2, POOL, LEN_MAX, got2);
        printf("coop pool_size=4: got=[");
        for (int x : got2) printf("%d ", x);
        printf("] want=[");
        for (int x : want) printf("%d ", x);
        printf("]\n");
        if (got2 != want) { printf("  MISMATCH\n"); bad++; }

        hipFree(d_sc); hipFree(d_idx); hipFree(d_kv); hipFree(d_gh); hipFree(d_gc);
    }

    /* ---- Case 3 + 4: d_index_select_pf ---- */
    {
        const unsigned POOL = 4, N_TOK = 2, TOPK = 2, KV_STRIDE = 16;
        /* row 0 at absolute token pos P0, row 1 at P0+1. Chunk-end kv_len (token count)
         * covers exactly through row 1: kv_len = P0 + N_TOK. */
        const unsigned P0 = 9; /* -> row0 row_len_tokens=10, row1=11 */
        const unsigned kv_len_tok = P0 + N_TOK; /* 11 */
        std::vector<float> sc(N_TOK * KV_STRIDE, 0.0f);
        /* row 0: valid [0,10), row 1: valid [0,11) -- but for pool_size=4 test only
         * [0, floor(10/4)=2) and [0, floor(11/4)=2) are valid; poison the rest per row. */
        for (unsigned t = 0; t < N_TOK; t++)
            for (unsigned s = 0; s < KV_STRIDE; s++) sc[t * KV_STRIDE + s] = 1.0e30f;
        /* real scores in the small valid windows, distinct per row */
        float row0v[2] = {3.0f, 7.0f};
        float row1v[2] = {6.0f, 2.0f};
        sc[0 * KV_STRIDE + 0] = row0v[0]; sc[0 * KV_STRIDE + 1] = row0v[1];
        sc[1 * KV_STRIDE + 0] = row1v[0]; sc[1 * KV_STRIDE + 1] = row1v[1];

        float* d_sc; int *d_idx, *d_kv;
        HC(hipMalloc(&d_sc, sc.size() * 4));
        HC(hipMalloc(&d_idx, N_TOK * TOPK * 4));
        HC(hipMalloc(&d_kv, 4));
        HC(hipMemcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice));
        int kvh = (int)kv_len_tok;
        HC(hipMemcpy(d_kv, &kvh, 4, hipMemcpyHostToDevice));
        HC(hipMemset(d_idx, 0xFF, N_TOK * TOPK * 4));

        PfArgs a{(unsigned long long)d_idx, (unsigned long long)d_sc, (unsigned long long)d_kv,
                 N_TOK, TOPK, KV_STRIDE, POOL};
        PfArgs* da;
        HC(hipMalloc(&da, sizeof(a)));
        HC(hipMemcpy(da, &a, sizeof(a), hipMemcpyHostToDevice));
        void* kp[] = {&da};
        HC(hipModuleLaunchKernel(f_pf, N_TOK, 1, 1, PLOW_THREADS, 1, 1, 0, 0, kp, nullptr));
        HC(hipDeviceSynchronize());
        std::vector<int> h(N_TOK * TOPK);
        HC(hipMemcpy(h.data(), d_idx, N_TOK * TOPK * 4, hipMemcpyDeviceToHost));
        HC(hipFree(da));

        /* row 0: row_len_tokens=10 -> /4 = 2 -> valid {0,1}, want both (top_k=2, len=2) */
        auto want0 = ref_topk({row0v[0], row0v[1]}, 2, TOPK);
        auto got0 = got_set(h.data() + 0 * TOPK, TOPK, 2);
        printf("pf row0 pool_size=4: got=[");
        for (int x : got0) printf("%d ", x);
        printf("] want=[");
        for (int x : want0) printf("%d ", x);
        printf("]\n");
        if (got0 != want0) { printf("  MISMATCH\n"); bad++; }

        /* row 1: row_len_tokens=11 -> /4 = 2 -> valid {0,1} too */
        auto want1 = ref_topk({row1v[0], row1v[1]}, 2, TOPK);
        auto got1 = got_set(h.data() + 1 * TOPK, TOPK, 2);
        printf("pf row1 pool_size=4: got=[");
        for (int x : got1) printf("%d ", x);
        printf("] want=[");
        for (int x : want1) printf("%d ", x);
        printf("]\n");
        if (got1 != want1) { printf("  MISMATCH\n"); bad++; }

        hipFree(d_sc); hipFree(d_idx); hipFree(d_kv);
    }

    printf("index_select pool_size check: %d mismatch(es)\n", bad);
    return bad ? 1 : 0;
}
#endif
