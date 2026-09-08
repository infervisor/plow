/* mixed_flash_window_gfx942_test.hip — the WINDOW bound of the packed prefill span loop, against
 * an f64 oracle, plus packed-vs-isolated byte identity.
 *
 * WHY A SECOND MIXED FLASH TEST. `mixed_flash_prefill_gfx942_test.hip` fills V with a constant
 * 1.0 and checks that the fused epilogue writes its output rows where the span says and nowhere
 * else. That is a placement test, and it is blind to the window BY CONSTRUCTION: softmax over any
 * subset of identical value rows is 1.0, so `window = 64` in that file could be `window = 0` or
 * `window = 3` and every assertion would still pass. Nothing in tree scores a windowed
 * `d_flash_prefill` at a NON-ZERO `q_pos0` against arithmetic. Gemma-4-31B runs 50 of its 60
 * layers windowed at 1024, and plans/unified-token-batch.md §9 asks for "windowed layers"
 * alongside prefix 0/nonzero, short final chunks and ring boundaries.
 *
 * THE HAZARD IT IS AIMED AT (§3, class C). `d_flash_prefill` derives every row's absolute
 * position as `q_pos0 + row`, and the window's lower bound from it:
 *
 *     q_tile_first = q_pos0 + q_base
 *     win_lo       = (window && q_tile_first >= window) ? q_tile_first - window + 1 : 0
 *     kv_lo        = (win_lo / BKV) * BKV                     <- the tile carve, and the splits
 *     valid        = (kg <= qg) && (!window || (qg - kg) < window)
 *
 * Under packing `q_pos0` is not a packet immediate but `span->kv_row0`, substituted per span by
 * the loop in interp.hip. So the window bound is per-span already — the question this file
 * answers is whether it is RIGHT, at each of the four places above and at the boundaries where
 * they disagree: a window entirely inside the cached prefix (`win_lo > 0` and no live query row
 * can see the start of the cache), a window that straddles the chunk start, a window wider than
 * the whole cache (must degenerate to plain causal), and a `win_lo` that is not a multiple of
 * BKV, where the tile carve rounds DOWN and the per-element mask has to finish the job.
 *
 * The second half is the packing property itself: three spans at different slots, prefixes and
 * lengths run through one loop must be byte-identical to the same three spans run one at a time.
 * That covers output-row placement and, because `Opart`/`mlpart` are offset per span by the same
 * arithmetic the interpreter uses, split-scratch disjointness.
 *
 * The loop below is a TRANSCRIPTION of PLOW_AMD_FLASH_PREFILL in runtime/amd/interp.hip, offsets
 * included, because the macro lives inside `exec_flash_prefill` and cannot be called from here.
 * If the two ever disagree the transcription is the bug — keep them together.
 *
 * In nix develop:
 *   $PLOW_HIPCC --offload-arch=gfx942 -O3 -Iruntime/amd -Iruntime/common -c \
 *       runtime/tests/mixed_flash_window_gfx942_test.hip -o /tmp/mixed_window.o
 *   c++ /tmp/mixed_window.o -L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib -lamdhip64 -o /tmp/mixed_window
 *   perf-data/tools/gpulease -n 1 mixed-flash-window /tmp/mixed_window
 */
#define PLOW_MIXED_STEP 1
#define PLOW_WG_WAVES 4
#define FA_DC 256
#define FA_DBUF 1
#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__)
#include "op_attention.h"
#else
using bf16 = unsigned short;
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* One request's contribution, in exactly the fields interp.hip reads off a PlowPrefillSpan. */
struct Span {
    unsigned row0;   /* first packed Q row */
    unsigned n_rows; /* query rows this request contributes */
    unsigned slot;   /* its KV slot */
    unsigned kv_row0;/* its absolute position base = `q_pos0` */
    unsigned kv_len; /* kv_row0 + n_rows: live KV after this span */
};

struct Geom {
    unsigned n_head, n_kv_head, kv_stride, kv_mask, window, nsplit;
    float scale;
};

static void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "hip: %s\n", hipGetErrorString(status));
        std::exit(1);
    }
}

template <typename T>
static T* upload(const std::vector<T>& v) {
    T* p;
    check(hipMalloc(&p, v.size() * sizeof(T)));
    check(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

static float b2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static bf16 f2b(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

/* ------------------------------------------------------------------ the device side */

template <int HD>
__global__ __launch_bounds__(256, 1) void k_spans(float* Opart, float* mlpart, bf16* O,
                                                  const bf16* Q, const bf16* K, const bf16* V,
                                                  const Span* spans, unsigned n_spans, Geom g) {
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ bf16 lds[FA_LDS_HALVES(HD)];
    /* TRANSCRIBED from PLOW_AMD_FLASH_PREFILL (runtime/amd/interp.hip). */
    for (unsigned si = 0; si < n_spans; ++si) {
        const Span s = spans[si];
        const size_t qoff = (size_t)s.row0 * g.n_head * HD;
        const size_t kvoff = (size_t)s.slot * g.n_kv_head * g.kv_stride * HD;
        d_flash_prefill<HD>(Opart + qoff * g.nsplit,
                            mlpart + (size_t)s.row0 * g.n_head * g.nsplit * 2u,
                            g.nsplit == 1u ? O + qoff : nullptr, Q + qoff, K + kvoff, V + kvoff,
                            s.n_rows, s.kv_len, g.n_head, g.n_kv_head, s.kv_row0, g.window,
                            g.scale, g.kv_stride, g.kv_mask, g.nsplit, blockIdx.x, gridDim.x, lds);
        __syncthreads();
    }
#else
    (void)Opart; (void)mlpart; (void)O; (void)Q; (void)K; (void)V; (void)spans; (void)n_spans;
    (void)g;
#endif
}

/* The span-loop twin of FlashMerge (interp.hip `exec_mixed_prefill_merge`). */
template <int HD>
__global__ __launch_bounds__(256, 1) void k_merge(bf16* O, const float* Opart, const float* mlpart,
                                                  const Span* spans, unsigned n_spans, Geom g) {
#if defined(__HIP_DEVICE_COMPILE__)
    for (unsigned si = 0; si < n_spans; ++si) {
        const Span s = spans[si];
        const size_t qoff = (size_t)s.row0 * g.n_head * HD;
        d_flash_merge<HD>(O + qoff, Opart + qoff * g.nsplit,
                          mlpart + (size_t)s.row0 * g.n_head * g.nsplit * 2u, s.n_rows, g.n_head,
                          g.nsplit, blockIdx.x, gridDim.x);
        __syncthreads();
    }
#else
    (void)O; (void)Opart; (void)mlpart; (void)spans; (void)n_spans; (void)g;
#endif
}

/* ------------------------------------------------------------------ the f64 oracle */

/* Softmax attention for one packed Q row, written from the CONTRACT at the top of
 * op_attention.h ("keep iff 0 <= q_idx - kv_idx <= window-1", window 0 = full causal) rather
 * than from the kernel's tiling, so a shared mistake would have to be made twice. */
static void oracle_row(std::vector<double>& out, size_t out_off, const std::vector<bf16>& Q,
                       const std::vector<bf16>& K, const std::vector<bf16>& V, size_t qoff,
                       size_t kvoff, unsigned hd, unsigned head, unsigned q_abs, unsigned n_kv,
                       const Geom& g) {
    const unsigned hkv = head / (g.n_head / g.n_kv_head);
    const unsigned last = q_abs < n_kv ? q_abs : n_kv - 1u;
    const unsigned lo = (g.window && q_abs + 1u > g.window) ? q_abs + 1u - g.window : 0u;
    std::vector<double> logit;
    double m = -INFINITY;
    for (unsigned kv = lo; kv <= last; kv++) {
        const size_t krow = kvoff + ((size_t)hkv * g.kv_stride + (kv & g.kv_mask)) * hd;
        double dot = 0.0;
        for (unsigned d = 0; d < hd; d++)
            dot += (double)b2f(Q[qoff + d]) * (double)b2f(K[krow + d]);
        dot *= (double)g.scale;
        logit.push_back(dot);
        if (dot > m) m = dot;
    }
    double sum = 0.0;
    for (double& x : logit) { x = std::exp(x - m); sum += x; }
    for (unsigned d = 0; d < hd; d++) out[out_off + d] = 0.0;
    for (unsigned i = 0; i < logit.size(); i++) {
        const unsigned kv = lo + i;
        const size_t vrow = kvoff + ((size_t)hkv * g.kv_stride + (kv & g.kv_mask)) * hd;
        const double p = logit[i] / sum;
        for (unsigned d = 0; d < hd; d++) out[out_off + d] += p * (double)b2f(V[vrow + d]);
    }
}

static double worst = 0.0;
static int fails = 0;
static void score(const char* nm, const std::vector<bf16>& got, const std::vector<double>& want,
                  double bar) {
    double se = 0, sw = 0, mx = 0, scale = 0;
    for (size_t i = 0; i < want.size(); i++) scale = std::fmax(scale, std::fabs(want[i]));
    for (size_t i = 0; i < want.size(); i++) {
        const double d = (double)b2f(got[i]) - want[i];
        se += d * d;
        sw += want[i] * want[i];
        mx = std::fmax(mx, std::fabs(d));
    }
    const double rms = std::sqrt(se / (sw + 1e-30)), rel = mx / (scale + 1e-30);
    const bool ok = rms < bar && rel < 2 * bar;
    std::printf("  %-44s rms %.3e  max %.3e (bar %.0e)%s\n", nm, rms, rel, bar,
                ok ? "" : "   <-- FAIL");
    worst = std::fmax(worst, std::fmax(rms / bar, rel / (2 * bar)));
    if (!ok) fails++;
}

/* ------------------------------------------------------------------ one case */

template <int HD>
static void run(const char* name, const std::vector<Span>& spans, Geom g, unsigned rows) {
    const unsigned n_slots = 4;
    std::srand(90210 + (int)g.window + (int)g.nsplit + HD);
    const size_t qn = (size_t)rows * g.n_head * HD;
    const size_t kvn = (size_t)n_slots * g.n_kv_head * g.kv_stride * HD;
    std::vector<bf16> Q(qn), K(kvn), V(kvn);
    auto r = [] { return (float)((double)std::rand() / RAND_MAX - 0.5); };
    for (auto& x : Q) x = f2b(r() * 2.0f);
    for (auto& x : K) x = f2b(r() * 2.0f);
    for (auto& x : V) x = f2b(r() * 2.0f);

    std::vector<double> want(qn, 0.0);
    for (const Span& s : spans)
        for (unsigned j = 0; j < s.n_rows; j++)
            for (unsigned h = 0; h < g.n_head; h++) {
                const size_t qoff = ((size_t)(s.row0 + j) * g.n_head + h) * HD;
                oracle_row(want, qoff, Q, K, V, qoff,
                           (size_t)s.slot * g.n_kv_head * g.kv_stride * HD, HD, h, s.kv_row0 + j,
                           s.kv_len, g);
            }

    auto* dq = upload(Q);
    auto* dk = upload(K);
    auto* dv = upload(V);
    auto* dsp = upload(spans);
    auto* dop = upload(std::vector<float>(qn * g.nsplit, 0.0f));
    auto* dml = upload(std::vector<float>((size_t)rows * g.n_head * g.nsplit * 2, 0.0f));
    /* 0xa5 poison: a row no span owns must come back holding it. */
    std::vector<bf16> O(qn, 0xa5a5);
    auto* dout = upload(O);

    hipLaunchKernelGGL((k_spans<HD>), dim3(4), dim3(256), 0, 0, dop, dml, dout, dq, dk, dv, dsp,
                       (unsigned)spans.size(), g);
    check(hipGetLastError());
    if (g.nsplit > 1)
        hipLaunchKernelGGL((k_merge<HD>), dim3(4), dim3(256), 0, 0, dout, dop, dml, dsp,
                           (unsigned)spans.size(), g);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    check(hipMemcpy(O.data(), dout, O.size() * sizeof(bf16), hipMemcpyDeviceToHost));

    /* Score only the rows spans actually own, and check the rest still holds the poison. */
    std::vector<bf16> got;
    std::vector<double> ref;
    for (const Span& s : spans) {
        const size_t a = (size_t)s.row0 * g.n_head * HD, n = (size_t)s.n_rows * g.n_head * HD;
        got.insert(got.end(), O.begin() + a, O.begin() + a + n);
        ref.insert(ref.end(), want.begin() + a, want.begin() + a + n);
    }
    score(name, got, ref, 4e-3);
    unsigned owned = 0;
    for (const Span& s : spans) owned += s.n_rows;
    bool poison_kept = true;
    for (size_t i = (size_t)owned * g.n_head * HD; i < qn; i++) poison_kept &= O[i] == 0xa5a5;
    if (!poison_kept) {
        std::printf("  %-44s padding rows were written   <-- FAIL\n", name);
        fails++;
    }

    for (void* p : {(void*)dq, (void*)dk, (void*)dv, (void*)dsp, (void*)dop, (void*)dml,
                    (void*)dout})
        check(hipFree(p));
}

/* Three spans in one loop vs the same three one at a time — byte identity, no oracle. */
template <int HD>
static void identity(const char* name, const std::vector<Span>& spans, Geom g, unsigned rows) {
    const unsigned n_slots = 4;
    std::srand(4242 + HD + (int)g.window);
    const size_t qn = (size_t)rows * g.n_head * HD;
    const size_t kvn = (size_t)n_slots * g.n_kv_head * g.kv_stride * HD;
    std::vector<bf16> Q(qn), K(kvn), V(kvn);
    auto r = [] { return (float)((double)std::rand() / RAND_MAX - 0.5); };
    for (auto& x : Q) x = f2b(r() * 2.0f);
    for (auto& x : K) x = f2b(r() * 2.0f);
    for (auto& x : V) x = f2b(r() * 2.0f);
    auto* dq = upload(Q);
    auto* dk = upload(K);
    auto* dv = upload(V);

    auto launch = [&](const std::vector<Span>& list, std::vector<bf16>& out) {
        auto* dsp = upload(list);
        auto* dop = upload(std::vector<float>(qn * g.nsplit, 0.0f));
        auto* dml = upload(std::vector<float>((size_t)rows * g.n_head * g.nsplit * 2, 0.0f));
        auto* dout = upload(out);
        hipLaunchKernelGGL((k_spans<HD>), dim3(4), dim3(256), 0, 0, dop, dml, dout, dq, dk, dv,
                           dsp, (unsigned)list.size(), g);
        if (g.nsplit > 1)
            hipLaunchKernelGGL((k_merge<HD>), dim3(4), dim3(256), 0, 0, dout, dop, dml, dsp,
                               (unsigned)list.size(), g);
        check(hipGetLastError());
        check(hipDeviceSynchronize());
        check(hipMemcpy(out.data(), dout, out.size() * sizeof(bf16), hipMemcpyDeviceToHost));
        for (void* p : {(void*)dsp, (void*)dop, (void*)dml, (void*)dout}) check(hipFree(p));
    };

    std::vector<bf16> packed(qn, 0xa5a5), iso(qn, 0xa5a5);
    launch(spans, packed);
    for (const Span& s : spans) {
        std::vector<bf16> one(qn, 0xa5a5);
        launch({s}, one);
        const size_t a = (size_t)s.row0 * g.n_head * HD, n = (size_t)s.n_rows * g.n_head * HD;
        std::copy(one.begin() + a, one.begin() + a + n, iso.begin() + a);
    }
    const bool same = std::memcmp(packed.data(), iso.data(), packed.size() * sizeof(bf16)) == 0;
    std::printf("  %-44s packed==isolated %s%s\n", name, same ? "YES" : "no",
                same ? "" : "   <-- FAIL");
    if (!same) fails++;
    for (void* p : {(void*)dq, (void*)dk, (void*)dv}) check(hipFree(p));
}

int main() {
    hipDeviceProp_t p;
    check(hipGetDeviceProperties(&p, 0));
    std::printf("dev: %s\n", p.gcnArchName);

    /* kv_stride 256 with kv_mask 255 is a 256-row ring; every case below keeps kv_len <= 256
     * except the ring case, which deliberately wraps. */
    auto geom = [](unsigned window, unsigned nsplit, unsigned stride, unsigned mask) {
        Geom g{};
        g.n_head = 2;
        g.n_kv_head = 1;
        g.kv_stride = stride;
        g.kv_mask = mask;
        g.window = window;
        g.nsplit = nsplit;
        g.scale = 0.125f;
        return g;
    };
    auto one = [](unsigned rows, unsigned slot, unsigned pos0) {
        return std::vector<Span>{{0, rows, slot, pos0, pos0 + rows}};
    };

    std::printf("single span, HD=256 (Gemma-4's sliding head):\n");
    run<256>("prefix 0, full causal (window 0)", one(48, 0, 0), geom(0, 1, 256, 255), 48);
    run<256>("prefix 0, window 16", one(48, 1, 0), geom(16, 1, 256, 255), 48);
    run<256>("prefix 96, window 16 (inside the prefix)", one(32, 2, 96), geom(16, 1, 256, 255), 32);
    run<256>("prefix 96, window 100 (straddles the chunk)", one(32, 3, 96), geom(100, 1, 256, 255),
             32);
    run<256>("prefix 96, window 4096 (wider than the KV)", one(32, 0, 96), geom(4096, 1, 256, 255),
             32);
    run<256>("prefix 96, window 37 (win_lo not BKV-aligned)", one(32, 1, 96), geom(37, 1, 256, 255),
             32);
    run<256>("short final chunk: 1 row at position 127", one(1, 2, 127), geom(16, 1, 256, 255), 1);
    run<256>("short final chunk: 3 rows, window 8", one(3, 3, 125), geom(8, 1, 256, 255), 3);
    run<256>("ring wrap: prefix 96, window 24, 64-row ring", one(32, 0, 96), geom(24, 1, 64, 63),
             32);

    std::printf("split merge (nsplit > 1):\n");
    run<256>("prefix 96, window 0, nsplit 4", one(32, 1, 96), geom(0, 4, 256, 255), 32);
    run<256>("prefix 96, window 40, nsplit 4", one(32, 2, 96), geom(40, 4, 256, 255), 32);
    run<256>("prefix 96, window 37, nsplit 3", one(32, 3, 96), geom(37, 3, 256, 255), 32);

    std::printf("single span, HD=128 (Llama/Qwen):\n");
    run<128>("prefix 0, window 24", one(48, 0, 0), geom(24, 1, 256, 255), 48);
    run<128>("prefix 96, window 24", one(32, 1, 96), geom(24, 1, 256, 255), 32);
    run<128>("prefix 96, window 24, nsplit 4", one(32, 2, 96), geom(24, 4, 256, 255), 32);

    /* Three requests at different slots, prefixes and lengths, packed dense from row 0. Each
     * span's live KV length differs, which is the §9 "different live KV lengths" case. */
    const std::vector<Span> pack{{0, 24, 0, 0, 24}, {24, 9, 1, 96, 105}, {33, 31, 2, 200, 231}};
    std::printf("packed spans, HD=256:\n");
    run<256>("3 spans, window 0", pack, geom(0, 1, 256, 255), 64);
    run<256>("3 spans, window 32", pack, geom(32, 1, 256, 255), 64);
    run<256>("3 spans, window 32, nsplit 4", pack, geom(32, 4, 256, 255), 64);
    identity<256>("3 spans, window 32", pack, geom(32, 1, 256, 255), 64);
    identity<256>("3 spans, window 32, nsplit 4", pack, geom(32, 4, 256, 255), 64);
    identity<128>("3 spans, window 24", pack, geom(24, 1, 256, 255), 64);

    std::printf("%s (worst %.3f, %d failing check(s))\n", fails ? "FAIL" : "PASS", worst, fails);
    return fails ? 1 : 0;
}
