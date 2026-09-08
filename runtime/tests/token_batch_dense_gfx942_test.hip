/* token_batch_dense_gfx942_test — the Phase 2 acceptance gates for dense GQA on AMD.
 *
 * plans/unified-token-batch.md §9 asks two things of a class-C conversion, and this harness is
 * built to answer exactly those two, on a device, with no model:
 *
 *   "For each converted operator: same tokens as one span vs split across spans, at every span
 *    boundary -> Identical results; the unconverted form is provably unreachable on this route."
 *
 * GATE 1  FlashPrefill, one request's contiguous chunk vs the SAME tokens split at every
 *         interior boundary. Bit-identical, and it is bit-identical for a stated reason, not
 *         by luck: at nsplit == 1 a q-tile whose last row moved only ever gains WHOLLY MASKED
 *         KV tiles, and a fully masked tile is an exact no-op on the online softmax (corr is
 *         exactly 1.0f, every pe is exactly 0.0f). At nsplit > 1 the KV partition itself moves,
 *         so bit-identity is NOT claimed there and the emitter pins nsplit = 1 on this route.
 *
 * GATE 2  FlashPrefill, several requests packed vs each request run isolated in its own slot.
 *         Row for row, bit-identical.
 *
 * GATE 3  The unconverted form is unreachable: the packed launch is given a POISONED
 *         `q_pos0`/`n_q`/`n_kv` (the three packet scalars the pre-conversion kernel derives
 *         every row's position from) and still produces gate 2's answer. The same poison run
 *         through the unconverted kernel is also reported, so "the descriptor is being read"
 *         is a measured claim rather than an assertion about code shape.
 *
 * GATE 4  FlashDecode fills `decode_slot` and the KV length from spans: many requests at
 *         different slots and different lengths, packed vs isolated. Bit-identical.
 *
 * GATE 5  Class-A combined M: one d_gemm over all packed rows vs one d_gemm per span.
 *         Bit-identical, because a row's dot product does not depend on M.
 *
 * GATE 6  Padding never contributes: rows past `real_rows` keep their sentinel.
 *
 * GATE 7  A CPU f64 reference for the smallest case, so the whole chain is anchored to
 *         something other than itself.
 *
 * Build and run (from the repo root, inside nix develop):
 *   $PLOW_HIPCC --offload-arch=gfx942 -O3 -DTB_DEVICE_TU -Iruntime/amd -Iruntime/common \
 *       -c runtime/tests/token_batch_dense_gfx942_test.hip -o /tmp/tb_dev.o
 *   c++ -x c++ -D__HIP_PLATFORM_AMD__ -I$ROCM_PATH/include -Iruntime/common \
 *       -c runtime/tests/token_batch_dense_gfx942_test.hip -o /tmp/tb_host.o
 *   c++ /tmp/tb_host.o /tmp/tb_dev.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 \
 *       -o /tmp/tb_dense
 *   GPU_LEASE_TIMEOUT=21600 perf-data/tools/gpulease -n 1 tb-dense /tmp/tb_dense
 */
#define PLOW_MIXED_STEP 1
#define PLOW_TOKEN_BATCH 1
#define PLOW_WG_WAVES 4
#define FA_DC 256
#define FA_DBUF 1
#define GM_BM 64
#define GM_BN 128

#include <hip/hip_runtime.h>

/* TB_DEVICE_TU selects the DEVICE translation unit. hipcc compiles it twice (device pass and
 * host pass); only the device pass may see op_attention.h, but the extern "C" launchers must
 * exist in the host pass, which is why the split is a -D and not __HIP_DEVICE_COMPILE__. */
#if defined(TB_DEVICE_TU) && defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#include "op_gemm.h"
#else
#include "dev_isa.h"
using bf16 = unsigned short;
#endif

/* ---------------------------------------------------------------- geometry (dense GQA) */
#define TB_HD 128
#define TB_HEADS 8
#define TB_KVHEADS 2
#define TB_KVSTRIDE 512u
#define TB_SLOTS 4u
#define TB_N 256u /* GEMM output width */
#define TB_K 128u /* GEMM contraction width */

#if defined(TB_DEVICE_TU)
/* ================================================================= device entry points */
#if defined(__HIP_DEVICE_COMPILE__)
#define TB_BODY(x) x
#else
#define TB_BODY(x)
#endif

__global__ __launch_bounds__(PLOW_WG_WAVES * 64, 1) void tb_flash_prefill_packed(
    float* Opart, float* mlpart, bf16* O, const bf16* Q, const bf16* K, const bf16* V,
    const PlowProgram* prog, unsigned poison_n_q, unsigned poison_n_kv, unsigned poison_pos0,
    float scale, unsigned nsplit) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[FA_LDS_HALVES(TB_HD)];
    const PlowTokenBatchView tb = plow_tb_view(prog);
    d_flash_prefill<TB_HD, false, TB_HD, /*TB=*/true>(
        Opart, mlpart, O, Q, K, V, poison_n_q, poison_n_kv, TB_HEADS, TB_KVHEADS, poison_pos0,
        /*window=*/0u, scale, TB_KVSTRIDE, PLOW_KV_MASK_NONE, nsplit, blockIdx.x, gridDim.x, lds,
        nullptr, nullptr, &tb);
#endif
}

/* The pre-conversion form: ONE packet scalar for the whole packed batch. Kept in the harness
 * so gate 3's claim is a measurement, not a code review. */
__global__ __launch_bounds__(PLOW_WG_WAVES * 64, 1) void tb_flash_prefill_unconverted(
    float* Opart, float* mlpart, bf16* O, const bf16* Q, const bf16* K, const bf16* V,
    unsigned n_q, unsigned n_kv, unsigned q_pos0, float scale, unsigned nsplit) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[FA_LDS_HALVES(TB_HD)];
    d_flash_prefill<TB_HD>(Opart, mlpart, O, Q, K, V, n_q, n_kv, TB_HEADS, TB_KVHEADS, q_pos0,
                           0u, scale, TB_KVSTRIDE, PLOW_KV_MASK_NONE, nsplit, blockIdx.x,
                           gridDim.x, lds);
#endif
}

__global__ __launch_bounds__(PLOW_WG_WAVES * 64, 1) void tb_flash_decode_packed(
    float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V,
    const PlowProgram* prog, unsigned n_batch, float scale, unsigned nsplit) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float lds[FA_DEC_LDS_FLOATS(TB_HD, PLOW_FA_GF(TB_HD))];
    const PlowTokenBatchView tb = plow_tb_view(prog);
    d_flash_decode<TB_HD, PLOW_FA_GF(TB_HD), false, false, /*SLOTMAP=*/false, /*TB=*/true>(
        Opart, mlpart, Q, K, V, /*kv_len=*/nullptr, n_batch, TB_HEADS, TB_KVHEADS, TB_KVSTRIDE,
        /*window=*/0u, scale, nsplit, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.0f, 0, nullptr, nullptr,
        nullptr, &tb);
#endif
}

__global__ __launch_bounds__(PLOW_WG_WAVES * 64, 1) void tb_flash_decode_isolated(
    float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, const int* kv_len,
    const int* decode_slot, unsigned n_batch, float scale, unsigned nsplit) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ float lds[FA_DEC_LDS_FLOATS(TB_HD, PLOW_FA_GF(TB_HD))];
    d_flash_decode<TB_HD, PLOW_FA_GF(TB_HD), false, false, /*SLOTMAP=*/true>(
        Opart, mlpart, Q, K, V, kv_len, n_batch, TB_HEADS, TB_KVHEADS, TB_KVSTRIDE, 0u, scale,
        nsplit, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, 0.0f, 0, nullptr, nullptr, decode_slot);
#endif
}

__global__ __launch_bounds__(PLOW_WG_WAVES * 64, 1) void tb_flash_merge(
    bf16* out, const float* partial, const float* ml, unsigned rows, unsigned nsplit) {
#if defined(__HIP_DEVICE_COMPILE__)
    d_flash_merge<TB_HD>(out, partial, ml, rows, TB_HEADS, nsplit, blockIdx.x, gridDim.x);
#endif
}

__global__ __launch_bounds__(PLOW_WG_WAVES * 64, 1) void tb_gemm(bf16* C, const bf16* A,
                                                                 const bf16* B, unsigned M) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[GM_LDS_HALVES];
    d_gemm(C, A, B, M, TB_N, TB_K, blockIdx.x, gridDim.x, lds);
#endif
}

extern "C" void tb_launch_prefill_packed(unsigned grid, float* Opart, float* mlpart, bf16* O,
                                         const bf16* Q, const bf16* K, const bf16* V,
                                         const void* prog, unsigned pn_q, unsigned pn_kv,
                                         unsigned ppos, float scale, unsigned nsplit) {
    hipLaunchKernelGGL(tb_flash_prefill_packed, dim3(grid), dim3(PLOW_WG_WAVES * 64), 0, 0, Opart,
                       mlpart, O, Q, K, V, (const PlowProgram*)prog, pn_q, pn_kv, ppos, scale,
                       nsplit);
}
extern "C" void tb_launch_prefill_unconverted(unsigned grid, float* Opart, float* mlpart, bf16* O,
                                              const bf16* Q, const bf16* K, const bf16* V,
                                              unsigned n_q, unsigned n_kv, unsigned pos0,
                                              float scale, unsigned nsplit) {
    hipLaunchKernelGGL(tb_flash_prefill_unconverted, dim3(grid), dim3(PLOW_WG_WAVES * 64), 0, 0,
                       Opart, mlpart, O, Q, K, V, n_q, n_kv, pos0, scale, nsplit);
}
extern "C" void tb_launch_decode_packed(unsigned grid, float* Opart, float* mlpart, const bf16* Q,
                                        const bf16* K, const bf16* V, const void* prog,
                                        unsigned n_batch, float scale, unsigned nsplit) {
    hipLaunchKernelGGL(tb_flash_decode_packed, dim3(grid), dim3(PLOW_WG_WAVES * 64), 0, 0, Opart,
                       mlpart, Q, K, V, (const PlowProgram*)prog, n_batch, scale, nsplit);
}
extern "C" void tb_launch_decode_isolated(unsigned grid, float* Opart, float* mlpart,
                                          const bf16* Q, const bf16* K, const bf16* V,
                                          const int* kv_len, const int* slot, unsigned n_batch,
                                          float scale, unsigned nsplit) {
    hipLaunchKernelGGL(tb_flash_decode_isolated, dim3(grid), dim3(PLOW_WG_WAVES * 64), 0, 0, Opart,
                       mlpart, Q, K, V, kv_len, slot, n_batch, scale, nsplit);
}
extern "C" void tb_launch_merge(unsigned grid, bf16* out, const float* partial, const float* ml,
                                unsigned rows, unsigned nsplit) {
    hipLaunchKernelGGL(tb_flash_merge, dim3(grid), dim3(PLOW_WG_WAVES * 64), 0, 0, out, partial,
                       ml, rows, nsplit);
}
extern "C" void tb_launch_gemm(unsigned grid, bf16* C, const bf16* A, const bf16* B, unsigned M) {
    hipLaunchKernelGGL(tb_gemm, dim3(grid), dim3(PLOW_WG_WAVES * 64), 0, 0, C, A, B, M);
}

#else
/* ======================================================================== host driver */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void tb_launch_prefill_packed(unsigned, float*, float*, bf16*, const bf16*,
                                         const bf16*, const bf16*, const void*, unsigned,
                                         unsigned, unsigned, float, unsigned);
extern "C" void tb_launch_prefill_unconverted(unsigned, float*, float*, bf16*, const bf16*,
                                              const bf16*, const bf16*, unsigned, unsigned,
                                              unsigned, float, unsigned);
extern "C" void tb_launch_decode_packed(unsigned, float*, float*, const bf16*, const bf16*,
                                        const bf16*, const void*, unsigned, float, unsigned);
extern "C" void tb_launch_decode_isolated(unsigned, float*, float*, const bf16*, const bf16*,
                                          const bf16*, const int*, const int*, unsigned, float,
                                          unsigned);
extern "C" void tb_launch_merge(unsigned, bf16*, const float*, const float*, unsigned, unsigned);
extern "C" void tb_launch_gemm(unsigned, bf16*, const bf16*, const bf16*, unsigned);

static unsigned g_fail = 0, g_pass = 0;

static void check(hipError_t s) {
    if (s != hipSuccess) {
        std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(s));
        std::exit(2);
    }
}
template <typename T>
static T* dev_alloc(size_t n) {
    T* p = nullptr;
    check(hipMalloc(&p, n * sizeof(T)));
    return p;
}
template <typename T>
static void up(T* d, const std::vector<T>& h) {
    check(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
}
template <typename T>
static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
    return h;
}

static float bf2f(bf16 v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
static bf16 f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

static uint32_t rng_state = 0x9e3779b9u;
static float rnd() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((int)(rng_state >> 8) - 8388608) * (0.5f / 8388608);
}
static std::vector<bf16> rnd_bf16(size_t n, uint32_t seed) {
    rng_state = seed;
    std::vector<bf16> v(n);
    for (auto& x : v) x = f2bf(rnd());
    return v;
}

static void report(const std::string& name, bool ok, const std::string& extra = "") {
    std::printf("%-58s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL",
                extra.empty() ? "" : "  ", extra.c_str());
    if (ok) ++g_pass; else ++g_fail;
}

/* One admitted request's contribution to a step. */
struct Span {
    unsigned row0, n_rows, slot, kv_row0;
};

/* Build the device-side PlowProgram carrying the descriptor. See runtime/amd/token_batch.h for
 * the assumed layout: until the shared `PlowTokenBatch` lands, the packed-prefill tail IS the
 * descriptor and spans must cover exactly [0, real_rows). */
struct DeviceBatch {
    PlowPrefillSpan* spans = nullptr;
    unsigned* parked = nullptr;
    PlowProgram* prog = nullptr;
    unsigned real_rows = 0, capacity = 0;
};

static DeviceBatch make_batch(const std::vector<Span>& spans, unsigned capacity) {
    DeviceBatch b;
    std::vector<PlowPrefillSpan> hs(spans.size());
    unsigned rows = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        PlowPrefillSpan& s = hs[i];
        std::memset(&s, 0, sizeof(s));
        s.row0 = spans[i].row0;
        s.n_rows = spans[i].n_rows;
        s.slot = spans[i].slot;
        s.state_slot = spans[i].slot;
        s.kv_row0 = spans[i].kv_row0;
        s.kv_len = spans[i].kv_row0 + spans[i].n_rows;
        s.flags = spans[i].kv_row0 == 0 ? PLOW_PREFILL_SPAN_RESET_STATE : 0u;
        rows = s.row0 + s.n_rows;
    }
    std::vector<unsigned> hp(capacity, 0u);
    for (unsigned r = rows; r < capacity; ++r) hp[r] = 1u;

    b.spans = dev_alloc<PlowPrefillSpan>(hs.size());
    b.parked = dev_alloc<unsigned>(capacity);
    up(b.spans, hs);
    up(b.parked, hp);

    PlowProgram hprog;
    std::memset(&hprog, 0, sizeof(hprog));
    hprog.prefill_spans = b.spans;
    hprog.prefill_parked = b.parked;
    hprog.n_prefill_spans = (unsigned)hs.size();
    hprog.n_prefill_rows = capacity;
    b.prog = dev_alloc<PlowProgram>(1);
    check(hipMemcpy(b.prog, &hprog, sizeof(hprog), hipMemcpyHostToDevice));
    b.real_rows = rows;
    b.capacity = capacity;
    return b;
}
static void free_batch(DeviceBatch& b) {
    check(hipFree(b.spans));
    check(hipFree(b.parked));
    check(hipFree(b.prog));
}

static const unsigned GRID = 64;
static const float SCALE = 0.08838834764f;

/* ---------------------------------------------------------------- shared fixture */
struct Fixture {
    bf16 *K = nullptr, *V = nullptr, *Q = nullptr, *O = nullptr;
    float *Opart = nullptr, *ml = nullptr;
    unsigned q_capacity;
    Fixture(unsigned q_cap) : q_capacity(q_cap) {
        const size_t kv = (size_t)TB_SLOTS * TB_KVHEADS * TB_KVSTRIDE * TB_HD;
        K = dev_alloc<bf16>(kv);
        V = dev_alloc<bf16>(kv);
        up(K, rnd_bf16(kv, 12345u));
        up(V, rnd_bf16(kv, 777u));
        Q = dev_alloc<bf16>((size_t)q_cap * TB_HEADS * TB_HD);
        up(Q, rnd_bf16((size_t)q_cap * TB_HEADS * TB_HD, 4242u));
        O = dev_alloc<bf16>((size_t)q_cap * TB_HEADS * TB_HD);
        Opart = dev_alloc<float>((size_t)q_cap * TB_HEADS * TB_HD);
        ml = dev_alloc<float>((size_t)q_cap * TB_HEADS * 2);
    }
    ~Fixture() {
        for (void* p : {(void*)K, (void*)V, (void*)Q, (void*)O, (void*)Opart, (void*)ml})
            check(hipFree(p));
    }
    void reset_out() {
        check(hipMemset(O, 0xA5, (size_t)q_capacity * TB_HEADS * TB_HD * sizeof(bf16)));
        check(hipMemset(Opart, 0x5A, (size_t)q_capacity * TB_HEADS * TB_HD * sizeof(float)));
        check(hipMemset(ml, 0x5A, (size_t)q_capacity * TB_HEADS * 2 * sizeof(float)));
    }
};

static size_t bitdiff(const std::vector<bf16>& a, const std::vector<bf16>& b, size_t lo,
                      size_t hi) {
    size_t n = 0;
    for (size_t i = lo; i < hi; ++i) n += (a[i] != b[i]);
    return n;
}

/* ============================================ GATE 1: one span vs split, every boundary */
static void gate1(Fixture& fx) {
    for (unsigned T : {1u, 5u, 32u, 33u, 64u, 65u, 129u}) {
        for (unsigned P0 : {0u, 7u, 128u}) {
            const unsigned cap = fx.q_capacity;
            DeviceBatch one = make_batch({{0u, T, 0u, P0}}, cap);
            fx.reset_out();
            tb_launch_prefill_packed(GRID, fx.Opart, fx.ml, fx.O, fx.Q, fx.K, fx.V, one.prog, 0u,
                                     0u, 0u, SCALE, 1u);
            check(hipDeviceSynchronize());
            const auto ref = down(fx.O, (size_t)cap * TB_HEADS * TB_HD);
            free_batch(one);

            size_t worst = 0;
            unsigned worst_b = 0;
            for (unsigned b = 1; b < T; ++b) {
                DeviceBatch split =
                    make_batch({{0u, b, 0u, P0}, {b, T - b, 0u, P0 + b}}, cap);
                fx.reset_out();
                tb_launch_prefill_packed(GRID, fx.Opart, fx.ml, fx.O, fx.Q, fx.K, fx.V,
                                         split.prog, 0u, 0u, 0u, SCALE, 1u);
                check(hipDeviceSynchronize());
                const auto got = down(fx.O, (size_t)cap * TB_HEADS * TB_HD);
                const size_t d = bitdiff(ref, got, 0, (size_t)T * TB_HEADS * TB_HD);
                if (d > worst) { worst = d; worst_b = b; }
                free_batch(split);
            }
            char msg[128];
            std::snprintf(msg, sizeof(msg), "T=%u pos0=%u boundaries=%u worst_diff=%zu@b=%u", T,
                          P0, T > 1 ? T - 1 : 0, worst, worst_b);
            report("gate1 flash_prefill one span vs split at every boundary", worst == 0, msg);
        }
    }
}

/* Isolated reference done properly: the solo launch reads Q at the SAME rows the packed launch
 * did (offsetting Q and O by the span's row0), so a row-for-row comparison is over identical
 * inputs. */
static void gate2(Fixture& fx) {
    struct Case { std::vector<Span> spans; const char* name; };
    const unsigned cap = fx.q_capacity;
    std::vector<Case> cases = {
        {{{0u, 1u, 0u, 100u}, {1u, 1u, 1u, 400u}, {2u, 50u, 2u, 70u}, {52u, 80u, 3u, 0u}},
         "gate2 6.1 example (2 decode, completing prompt, mid chunk)"},
        {{{0u, 1u, 3u, 0u}}, "gate2 one-token prompt"},
        {{{0u, 33u, 1u, 0u}, {33u, 33u, 2u, 33u}}, "gate2 two cold chunks, tile-straddling"},
        {{{0u, 128u, 0u, 0u}, {128u, 1u, 1u, 300u}}, "gate2 full tile + a decode row"},
        {{{0u, 7u, 0u, 3u}, {7u, 7u, 0u, 60u}, {14u, 7u, 0u, 120u}},
         "gate2 three spans sharing one slot at different frontiers"},
    };
    const size_t oelems = (size_t)cap * TB_HEADS * TB_HD;
    for (auto& c : cases) {
        unsigned rows = 0;
        for (auto& s : c.spans) rows = s.row0 + s.n_rows;

        DeviceBatch packed = make_batch(c.spans, cap);
        fx.reset_out();
        tb_launch_prefill_packed(GRID, fx.Opart, fx.ml, fx.O, fx.Q, fx.K, fx.V, packed.prog,
                                 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, SCALE, 1u);
        check(hipDeviceSynchronize());
        const auto got = down(fx.O, oelems);
        free_batch(packed);

        std::vector<bf16> ref(oelems, 0);
        for (auto& s : c.spans) {
            DeviceBatch solo = make_batch({{0u, s.n_rows, s.slot, s.kv_row0}}, cap);
            fx.reset_out();
            const size_t qoff = (size_t)s.row0 * TB_HEADS * TB_HD;
            tb_launch_prefill_packed(GRID, fx.Opart, fx.ml, fx.O + qoff, fx.Q + qoff, fx.K, fx.V,
                                     solo.prog, 0u, 0u, 0u, SCALE, 1u);
            check(hipDeviceSynchronize());
            const auto solo_out = down(fx.O, oelems);
            std::memcpy(ref.data() + qoff, solo_out.data() + qoff,
                        (size_t)s.n_rows * TB_HEADS * TB_HD * sizeof(bf16));
            free_batch(solo);
        }
        const size_t d = bitdiff(ref, got, 0, (size_t)rows * TB_HEADS * TB_HD);
        /* GATE 6: the padded tail must be untouched by the packed launch. */
        const auto tail = down(fx.O, oelems); /* re-read is harmless; O still holds the solo run */
        (void)tail;
        char msg[96];
        std::snprintf(msg, sizeof(msg), "rows=%u spans=%zu diff=%zu", rows, c.spans.size(), d);
        report(c.name, d == 0, msg);
    }
}

/* GATE 3 negative control: the same packed batch through the unconverted kernel. */
static void gate3_control(Fixture& fx) {
    const unsigned cap = fx.q_capacity;
    const std::vector<Span> spans = {{0u, 32u, 0u, 0u}, {32u, 32u, 0u, 200u}};
    const unsigned rows = 64;
    const size_t oelems = (size_t)cap * TB_HEADS * TB_HD;

    DeviceBatch packed = make_batch(spans, cap);
    fx.reset_out();
    tb_launch_prefill_packed(GRID, fx.Opart, fx.ml, fx.O, fx.Q, fx.K, fx.V, packed.prog,
                             0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, SCALE, 1u);
    check(hipDeviceSynchronize());
    const auto converted = down(fx.O, oelems);
    free_batch(packed);

    fx.reset_out();
    /* The unconverted form's only choice: one scalar base for the whole pack. */
    tb_launch_prefill_unconverted(GRID, fx.Opart, fx.ml, fx.O, fx.Q, fx.K, fx.V, rows,
                                  spans[0].kv_row0 + rows, spans[0].kv_row0, SCALE, 1u);
    check(hipDeviceSynchronize());
    const auto unconverted = down(fx.O, oelems);
    const size_t d = bitdiff(converted, unconverted, 0, (size_t)rows * TB_HEADS * TB_HD);
    char msg[96];
    std::snprintf(msg, sizeof(msg), "differing bf16 outputs=%zu of %zu", d,
                  (size_t)rows * TB_HEADS * TB_HD);
    /* This one PASSES when the two DIFFER: if they agreed, the packed route would not need the
     * conversion and the gate would be vacuous. */
    report("gate3 unconverted form is a different (wrong) answer", d != 0, msg);
}

/* ============================================ GATE 4: FlashDecode slot/len from spans */
static void gate4(Fixture& fx) {
    const unsigned cap = fx.q_capacity;
    const unsigned nsplit = 4;
    const std::vector<Span> spans = {{0u, 1u, 0u, 100u},
                                     {1u, 1u, 1u, 7u},
                                     {2u, 1u, 3u, 383u},
                                     {3u, 1u, 2u, 0u}};
    const unsigned B = (unsigned)spans.size();
    const size_t pelems = (size_t)cap * TB_HEADS * nsplit * TB_HD;
    const size_t melems = (size_t)cap * TB_HEADS * nsplit * 2;
    float* Opart = dev_alloc<float>(pelems);
    float* ml = dev_alloc<float>(melems);
    bf16* out_a = dev_alloc<bf16>((size_t)cap * TB_HEADS * TB_HD);
    bf16* out_b = dev_alloc<bf16>((size_t)cap * TB_HEADS * TB_HD);

    DeviceBatch batch = make_batch(spans, cap);
    check(hipMemset(Opart, 0, pelems * sizeof(float)));
    check(hipMemset(ml, 0, melems * sizeof(float)));
    tb_launch_decode_packed(GRID, Opart, ml, fx.Q, fx.K, fx.V, batch.prog, B, SCALE, nsplit);
    tb_launch_merge(GRID, out_a, Opart, ml, B, nsplit);
    check(hipDeviceSynchronize());
    const auto packed = down(out_a, (size_t)B * TB_HEADS * TB_HD);
    free_batch(batch);

    std::vector<int> hlen(B), hslot(B);
    for (unsigned i = 0; i < B; ++i) {
        hlen[i] = (int)(spans[i].kv_row0 + 1);
        hslot[i] = (int)spans[i].slot;
    }
    int* dlen = dev_alloc<int>(B);
    int* dslot = dev_alloc<int>(B);
    up(dlen, hlen);
    up(dslot, hslot);
    check(hipMemset(Opart, 0, pelems * sizeof(float)));
    check(hipMemset(ml, 0, melems * sizeof(float)));
    tb_launch_decode_isolated(GRID, Opart, ml, fx.Q, fx.K, fx.V, dlen, dslot, B, SCALE, nsplit);
    tb_launch_merge(GRID, out_b, Opart, ml, B, nsplit);
    check(hipDeviceSynchronize());
    const auto isolated = down(out_b, (size_t)B * TB_HEADS * TB_HD);

    const size_t d = bitdiff(packed, isolated, 0, packed.size());
    char msg[96];
    std::snprintf(msg, sizeof(msg), "B=%u nsplit=%u diff=%zu", B, nsplit, d);
    report("gate4 flash_decode slot+len from spans vs host slot map", d == 0, msg);

    for (void* p : {(void*)Opart, (void*)ml, (void*)out_a, (void*)out_b, (void*)dlen,
                    (void*)dslot})
        check(hipFree(p));
}

/* ============================================ GATE 5: combined-M projections */
static void gate5() {
    const unsigned M = 133; /* 2 decode rows + a 51-row completion + an 80-row chunk */
    const std::vector<unsigned> span_rows = {1, 1, 51, 80};
    bf16* A = dev_alloc<bf16>((size_t)M * TB_K);
    bf16* B = dev_alloc<bf16>((size_t)TB_N * TB_K);
    bf16* Cp = dev_alloc<bf16>((size_t)M * TB_N);
    bf16* Ci = dev_alloc<bf16>((size_t)M * TB_N);
    up(A, rnd_bf16((size_t)M * TB_K, 31u));
    up(B, rnd_bf16((size_t)TB_N * TB_K, 97u));
    check(hipMemset(Cp, 0xA5, (size_t)M * TB_N * sizeof(bf16)));
    check(hipMemset(Ci, 0xA5, (size_t)M * TB_N * sizeof(bf16)));

    tb_launch_gemm(GRID, Cp, A, B, M); /* ONE combined-M call: no phase band */
    unsigned row = 0;
    for (unsigned r : span_rows) {
        tb_launch_gemm(GRID, Ci + (size_t)row * TB_N, A + (size_t)row * TB_K, B, r);
        row += r;
    }
    check(hipDeviceSynchronize());
    const auto p = down(Cp, (size_t)M * TB_N);
    const auto i = down(Ci, (size_t)M * TB_N);
    const size_t d = bitdiff(p, i, 0, p.size());
    char msg[96];
    std::snprintf(msg, sizeof(msg), "M=%u spans=%zu diff=%zu", M, span_rows.size(), d);
    report("gate5 combined-M gemm vs per-span gemm, row for row", d == 0, msg);
    for (void* q : {(void*)A, (void*)B, (void*)Cp, (void*)Ci}) check(hipFree(q));
}

/* ============================================ GATE 6: padding writes nothing */
static void gate6(Fixture& fx) {
    const unsigned cap = fx.q_capacity;
    const std::vector<Span> spans = {{0u, 1u, 0u, 10u}, {1u, 20u, 1u, 5u}};
    const unsigned rows = 21;
    const size_t oelems = (size_t)cap * TB_HEADS * TB_HD;
    DeviceBatch batch = make_batch(spans, cap);
    check(hipMemset(fx.O, 0xA5, oelems * sizeof(bf16)));
    tb_launch_prefill_packed(GRID, fx.Opart, fx.ml, fx.O, fx.Q, fx.K, fx.V, batch.prog, 0u, 0u,
                             0u, SCALE, 1u);
    check(hipDeviceSynchronize());
    const auto got = down(fx.O, oelems);
    free_batch(batch);
    size_t touched = 0;
    for (size_t i = (size_t)rows * TB_HEADS * TB_HD; i < oelems; ++i)
        touched += (got[i] != (bf16)0xA5A5);
    char msg[96];
    std::snprintf(msg, sizeof(msg), "capacity=%u real_rows=%u touched_pad=%zu", cap, rows,
                  touched);
    report("gate6 padded rows past real_rows are never written", touched == 0, msg);
}

/* ============================================ GATE 7: CPU f64 reference */
static void gate7(Fixture& fx) {
    const unsigned cap = fx.q_capacity;
    const std::vector<Span> spans = {{0u, 1u, 0u, 40u}, {1u, 9u, 2u, 3u}};
    const unsigned rows = 10;
    const size_t oelems = (size_t)cap * TB_HEADS * TB_HD;
    DeviceBatch batch = make_batch(spans, cap);
    fx.reset_out();
    tb_launch_prefill_packed(GRID, fx.Opart, fx.ml, fx.O, fx.Q, fx.K, fx.V, batch.prog, 0u, 0u,
                             0u, SCALE, 1u);
    check(hipDeviceSynchronize());
    const auto got = down(fx.O, oelems);
    free_batch(batch);

    const size_t kvn = (size_t)TB_SLOTS * TB_KVHEADS * TB_KVSTRIDE * TB_HD;
    const auto hK = down(fx.K, kvn);
    const auto hV = down(fx.V, kvn);
    const auto hQ = down(fx.Q, (size_t)cap * TB_HEADS * TB_HD);

    double worst = 0.0, scale_ref = 0.0;
    for (const Span& s : spans) {
        for (unsigned j = 0; j < s.n_rows; ++j) {
            const unsigned row = s.row0 + j;
            const unsigned pos = s.kv_row0 + j; /* THE per-row causal bound (§5.2) */
            for (unsigned h = 0; h < TB_HEADS; ++h) {
                const unsigned hkv = h / (TB_HEADS / TB_KVHEADS);
                std::vector<double> logit(pos + 1);
                double mx = -1e300;
                for (unsigned t = 0; t <= pos; ++t) {
                    double acc = 0.0;
                    const size_t kb =
                        ((size_t)s.slot * TB_KVHEADS + hkv) * TB_KVSTRIDE * TB_HD +
                        (size_t)t * TB_HD;
                    for (unsigned d = 0; d < TB_HD; ++d)
                        acc += (double)bf2f(hQ[((size_t)row * TB_HEADS + h) * TB_HD + d]) *
                               (double)bf2f(hK[kb + d]);
                    logit[t] = acc * (double)SCALE;
                    if (logit[t] > mx) mx = logit[t];
                }
                double den = 0.0;
                for (unsigned t = 0; t <= pos; ++t) { logit[t] = std::exp(logit[t] - mx); den += logit[t]; }
                for (unsigned d = 0; d < TB_HD; ++d) {
                    double acc = 0.0;
                    for (unsigned t = 0; t <= pos; ++t) {
                        const size_t vb =
                            ((size_t)s.slot * TB_KVHEADS + hkv) * TB_KVSTRIDE * TB_HD +
                            (size_t)t * TB_HD;
                        acc += logit[t] * (double)bf2f(hV[vb + d]);
                    }
                    const double want = acc / den;
                    const double have = (double)bf2f(got[((size_t)row * TB_HEADS + h) * TB_HD + d]);
                    worst = std::max(worst, std::fabs(want - have));
                    scale_ref = std::max(scale_ref, std::fabs(want));
                }
            }
        }
    }
    const double rel = scale_ref > 0 ? worst / scale_ref : worst;
    char msg[128];
    std::snprintf(msg, sizeof(msg), "rows=%u max_abs=%.3e rel=%.3e (bf16 out, tol 8e-3)", rows,
                  worst, rel);
    report("gate7 packed output vs CPU f64 attention reference", rel < 8e-3, msg);
}

int main() {
    int devs = 0;
    check(hipGetDeviceCount(&devs));
    if (devs < 1) {
        std::fprintf(stderr, "no device\n");
        return 2;
    }
    std::printf("token-batch dense-GQA gates: hd=%d heads=%d kv_heads=%d slots=%u stride=%u\n",
                TB_HD, TB_HEADS, TB_KVHEADS, TB_SLOTS, TB_KVSTRIDE);
    {
        Fixture fx(160);
        gate1(fx);
        gate2(fx);
        gate3_control(fx);
        gate4(fx);
        gate6(fx);
        gate7(fx);
    }
    gate5();
    std::printf("\n%u passed, %u failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
#endif
