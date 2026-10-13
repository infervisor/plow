// mla_pf_dense_gfx950_test — correctness (vs an FP64 reference of exact causal softmax attention,
// merged over the ns split partials) and timing of plow_mla_pf_dense_gfx950.
//   hipcc --offload-arch=gfx950 -O3 -I runtime/amd -I runtime/common \
//     runtime/tests/mla_pf_dense_gfx950_test.hip -o mla_pf_dense_test
//   ./mla_pf_dense_test [perf]
#include "../amd/mla_pf_dense_gfx950.hip"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                          \
    do {                                                                               \
        hipError_t e_ = (x);                                                           \
        if (e_ != hipSuccess) {                                                        \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static uint16_t f2b(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static float b2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static uint64_t rng = 0x12345678abcdefull;
static float frand() {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng >> 40) & 0xffffff) / 16777216.0f * 2.0f - 1.0f;
}

struct Span {
    unsigned row0, n_rows, kv_row0, kv_len, kv_base;
};

// One case: n_spans == 0 -> single sequence (spans[0] describes it, kv_base 0).
static int run_case(const char* name, std::vector<Span> sp, bool packed, unsigned ns, unsigned ctx_cap,
                    bool perf, int check_stride, bool ofold = false) {
    const unsigned H = 8, D = 512;
    unsigned rows_total = 0, slots = 0;
    for (auto& s : sp) {
        rows_total = std::max(rows_total, s.row0 + s.n_rows);
        slots = std::max(slots, s.kv_base / ctx_cap + 1);
    }
    const size_t kv_rows = (size_t)slots * ctx_cap;
    std::vector<uint16_t> hq((size_t)rows_total * H * D), hk(kv_rows * D);
    // Scores with real spread: keys share a common direction so softmax is peaked for some rows.
    std::vector<float> dir(D);
    for (auto& v : dir) v = frand();
    for (size_t i = 0; i < hq.size(); i++) hq[i] = f2b(frand() * 2.0f + 3.0f * dir[i % D]);
    for (size_t i = 0; i < hk.size(); i++)
        hk[i] = f2b(frand() + ((i / D) % 7 == 0 ? 0.5f * dir[i % D] * (float)((i / D) % 5) : 0.0f));
    const float scale = 1.0f / sqrtf(576.0f);

    bf16 *dq, *dk;
    float *dO, *dml;
    int* dlen;
    PlowKvSpan* dsp = nullptr;
    CK(hipMalloc(&dq, hq.size() * 2));
    CK(hipMalloc(&dk, hk.size() * 2));
    CK(hipMalloc(&dO, (size_t)rows_total * H * ns * D * 4));
    CK(hipMalloc(&dml, (size_t)rows_total * H * ns * 2 * 4));
    CK(hipMalloc(&dlen, 4));
    CK(hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemset(dO, 0xff, (size_t)rows_total * H * ns * D * 4));  // NaN poison
    CK(hipMemset(dml, 0xff, (size_t)rows_total * H * ns * 2 * 4));
    unsigned grid = 0;
    for (auto& s : sp) grid += (s.n_rows + 7) / 8 * ns;
    if (packed) {
        std::vector<PlowKvSpan> t(sp.size());
        for (size_t i = 0; i < sp.size(); i++)
            t[i] = PlowKvSpan{sp[i].row0, sp[i].n_rows, sp[i].kv_row0, sp[i].kv_len, sp[i].kv_base, {0, 0, 0}};
        CK(hipMalloc(&dsp, t.size() * sizeof(PlowKvSpan)));
        CK(hipMemcpy(dsp, t.data(), t.size() * sizeof(PlowKvSpan), hipMemcpyHostToDevice));
    } else {
        int len = (int)sp[0].kv_len;
        CK(hipMemcpy(dlen, &len, 4, hipMemcpyHostToDevice));
    }
    const unsigned nsp = packed ? (unsigned)sp.size() : 0u;
    const unsigned n_tok = sp[0].n_rows;
    auto launch = [&]() {
        hipLaunchKernelGGL(plow_mla_pf_dense_gfx950, dim3(grid), dim3(256), 0, 0, dO, dml,
                           (const bf16*)dq, (const bf16*)dk, (const int*)dlen,
                           (const PlowKvSpan*)dsp, nsp, n_tok, H, ctx_cap, 0xffffffffu, ns, scale, ofold ? 1u : 0u);
    };
    launch();
    CK(hipDeviceSynchronize());

    std::vector<float> hO((size_t)rows_total * H * ns * D), hml((size_t)rows_total * H * ns * 2);
    CK(hipMemcpy(hO.data(), dO, hO.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hml.data(), dml, hml.size() * 4, hipMemcpyDeviceToHost));
    if (ofold) {  // normalized bf16 rows: present them as one split with m = 0, l = 1
        std::vector<uint16_t> hb((size_t)rows_total * H * D);
        CK(hipMemcpy(hb.data(), dO, hb.size() * 2, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < hb.size(); i++) hO[i] = b2f(hb[i]);
        for (size_t r = 0; r < (size_t)rows_total * H; r++) { hml[r * 2] = 0.0f; hml[r * 2 + 1] = 1.0f; }
    }
    double num = 0, den = 0, worst = 0;
    int checked = 0, bad = 0;
    for (auto& s : sp) {
        for (unsigned r = 0; r < s.n_rows; r += (r + check_stride < s.n_rows ? check_stride : 1)) {
            const unsigned pos = s.kv_row0 + r;
            for (unsigned h = 0; h < H; h++) {
                const size_t row = (size_t)(s.row0 + r) * H + h;
                const uint16_t* q = &hq[row * D];
                std::vector<double> sc(pos + 1);
                double m = -1e300;
                for (unsigned k = 0; k <= pos; k++) {
                    const uint16_t* kk = &hk[((size_t)s.kv_base + k) * D];
                    double d = 0;
                    for (unsigned j = 0; j < D; j++) d += (double)b2f(q[j]) * b2f(kk[j]);
                    sc[k] = d * scale;
                    m = std::max(m, sc[k]);
                }
                std::vector<double> ref(D, 0.0);
                double l = 0;
                for (unsigned k = 0; k <= pos; k++) {
                    const double p = exp(sc[k] - m);
                    l += p;
                    const uint16_t* kk = &hk[((size_t)s.kv_base + k) * D];
                    for (unsigned j = 0; j < D; j++) ref[j] += p * b2f(kk[j]);
                }
                // merge splits (base-2 frames)
                double M = -1e300;
                for (unsigned p = 0; p < ns; p++) M = std::max(M, (double)hml[(row * ns + p) * 2]);
                double L = 0;
                std::vector<double> got(D, 0.0);
                for (unsigned p = 0; p < ns; p++) {
                    const double wgt = exp2((double)hml[(row * ns + p) * 2] - M);
                    L += wgt * hml[(row * ns + p) * 2 + 1];
                    for (unsigned j = 0; j < D; j++) got[j] += wgt * hO[(row * ns + p) * D + j];
                }
                double rn = 0, rd = 0;
                for (unsigned j = 0; j < D; j++) {
                    const double rv = ref[j] / l, gv = got[j] / L;
                    rn += (rv - gv) * (rv - gv);
                    rd += rv * rv;
                }
                if (!(rn == rn) || !(L > 0)) bad++;
                num += rn;
                den += rd;
                worst = std::max(worst, sqrt(rn / (rd + 1e-300)));
                checked++;
            }
        }
    }
    const double rel = sqrt(num / den);
    const bool ok = bad == 0 && rel < 5e-3 && worst < 2e-2;
    printf("%-28s rows=%u ns=%u checked=%d relL2=%.3e worst=%.3e nan=%d %s\n", name, rows_total, ns,
           checked, rel, worst, bad, ok ? "OK" : "FAIL");
    if (perf) {
        hipEvent_t a, b;
        CK(hipEventCreate(&a));
        CK(hipEventCreate(&b));
        float best = 1e30f;
        for (int rep = 0; rep < 5; rep++) {
            CK(hipEventRecord(a));
            for (int i = 0; i < 10; i++) launch();
            CK(hipEventRecord(b));
            CK(hipEventSynchronize(b));
            float ms;
            CK(hipEventElapsedTime(&ms, a, b));
            best = std::min(best, ms / 10);
        }
        double flop = 0;
        for (auto& s : sp)
            for (unsigned r = 0; r < s.n_rows; r++) flop += 2.0 * 2.0 * 512.0 * H * (s.kv_row0 + r + 1);
        printf("  PERF %-24s %.3f ms/layer  %.1f TF/s (causal-useful)  x11 layers = %.2f ms\n", name, best,
               flop / (best * 1e-3) / 1e12, best * 11);
    }
    CK(hipFree(dq));
    CK(hipFree(dk));
    CK(hipFree(dO));
    CK(hipFree(dml));
    CK(hipFree(dlen));
    if (dsp) CK(hipFree(dsp));
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const bool perf = argc > 1 && !strcmp(argv[1], "perf");
    int fails = 0;
    fails += run_case("fresh T=200 ns=1", {{0, 200, 0, 200, 0}}, false, 1, 4096, false, 1);
    fails += run_case("fresh T=203 ns=2", {{0, 203, 0, 203, 0}}, false, 2, 4096, false, 1);
    fails += run_case("cont T=136 len=1000 ns=2", {{0, 136, 864, 1000, 0}}, false, 2, 4096, false, 3);
    fails += run_case("cont T=77 len=333 ns=3", {{0, 77, 256, 333, 0}}, false, 3, 4096, false, 1);
    fails += run_case("packed 3 spans ns=2",
                      {{0, 100, 0, 100, 0}, {100, 45, 300, 345, 4096}, {145, 64, 640, 704, 2 * 4096}},
                      true, 2, 4096, false, 1);
    fails += run_case("ofold packed 2 spans", {{0, 100, 0, 100, 0}, {100, 45, 300, 345, 4096}}, true, 1,
                      4096, false, 1, true);
    fails += run_case("ofold cont T=136 len=1000", {{0, 136, 864, 1000, 0}}, false, 1, 4096, false, 3, true);
    if (perf) {
        fails += run_case("8k fresh ofold", {{0, 8192, 0, 8192, 0}}, false, 1, 16384, true, 2039, true);
        fails += run_case("8k fresh ns=2", {{0, 8192, 0, 8192, 0}}, false, 2, 16384, true, 1021);
        fails += run_case("8k fresh ns=1", {{0, 8192, 0, 8192, 0}}, false, 1, 16384, true, 2039);
        fails += run_case("16k 2nd chunk ns=2", {{0, 8192, 8192, 16384, 0}}, false, 2, 16384, true, 2039);
    }
    printf(fails ? "MLA_PF_DENSE_TEST FAIL (%d)\n" : "MLA_PF_DENSE_TEST PASS\n", fails);
    return fails ? 1 : 0;
}
