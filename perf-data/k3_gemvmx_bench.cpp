/* Host driver for k3_gemvmx_bench.hip — the pre-fix DUAL-COLUMN mxfp4 GEMV against the R-split one.
 *
 * Interleaved A/B in ONE process (contract 6b-STALE): old, new, old, new, min of the two passes,
 * each pass a median-of-IT hipEvent measurement. hipEvent and not wall clock because the harness
 * this replaces (runtime/ubench/gemv_row_sweep.c) carries a ~31 us per-dispatch floor that buries
 * a 3 us shape entirely; the fix is worth a fraction of the BODY, so the body is what gets timed.
 *
 * WHY nrep. A single [N,K] mxfp4 slab at the small shapes is ~10 MB and lands in cache, so one
 * pass measures a cache-resident latency number rather than the decode stream. Each rep walks a
 * fresh slab out of a 1.5 GB arena, and the reported us/GB\s are per-rep.
 *
 * THE ODD-COLUMN COLUMN. `q` is the per-wave column count: gv_per = ceil(N/grid) columns per
 * workgroup, dealt to PLOW_WAVES waves in stride-PLOW_WAVES fashion. The dual-column loop makes
 * 2*ceil(q/2) column-passes for q columns, so the waste is 1/(q+1) at odd q and ZERO at even q.
 * The even rows are the controls: they must not move, and if they do the measurement is noise.
 *
 * BIT-EXACTNESS IS CHECKED FIRST, elementwise over all N outputs, with real varied weights and
 * scales rather than a memset constant — the fix is only allowed to delete the discarded second
 * stream, so any difference at all is a bug and not a rounding question. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                        \
        if (e_ != hipSuccess) {                                                                     \
            printf("HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));                   \
            exit(1);                                                                                \
        }                                                                                           \
    } while (0)

static hipModule_t M;
static const int T = 512; /* PLOW_THREADS */

static double tm(hipFunction_t f, int g, void** a, int it) {
    for (int i = 0; i < 3; i++) CK(hipModuleLaunchKernel(f, g, 1, 1, T, 1, 1, 0, 0, a, nullptr));
    CK(hipDeviceSynchronize());
    std::vector<double> v;
    hipEvent_t s, e;
    CK(hipEventCreate(&s));
    CK(hipEventCreate(&e));
    for (int i = 0; i < it; i++) {
        CK(hipEventRecord(s, 0));
        CK(hipModuleLaunchKernel(f, g, 1, 1, T, 1, 1, 0, 0, a, nullptr));
        CK(hipEventRecord(e, 0));
        CK(hipEventSynchronize(e));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, s, e));
        v.push_back((double)ms * 1000.0);
    }
    CK(hipEventDestroy(s));
    CK(hipEventDestroy(e));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* Per-wave column count and the resulting column-pass waste of the dual loop, for the table. */
static void wave_cols(unsigned N, unsigned grid, unsigned waves, unsigned* qlo, unsigned* qhi,
                      double* waste) {
    const unsigned per = (N + grid - 1) / grid;
    unsigned passes = 0, cols = 0, lo = ~0u, hi = 0;
    for (unsigned w = 0; w < waves; w++) {
        unsigned q = (per > w) ? ((per - 1 - w) / waves + 1) : 0;
        cols += q;
        passes += 2 * ((q + 1) / 2);
        lo = std::min(lo, q);
        hi = std::max(hi, q);
    }
    *qlo = lo;
    *qhi = hi;
    *waste = cols ? 1.0 - (double)cols / (double)passes : 0.0;
}

int main(int argc, char** argv) {
    CK(hipInit(0));
    CK(hipModuleLoad(&M, argc > 1 ? argv[1] : "/tmp/gvmx.co"));
    const int GRID = argc > 2 ? atoi(argv[2]) : 256;
    const int IT = argc > 3 ? atoi(argv[3]) : 21;
    const unsigned WAVES = (unsigned)T / 64;
    /* argv[4]: a SECOND object to take `k_mx_new` from, so each arm can be measured out of an
     * object that carries only its own body — see the MX_ARM note in the .hip. Omit it and both
     * arms come from one object. Either way the two are interleaved inside this one process. */
    hipModule_t M2 = M;
    if (argc > 4) CK(hipModuleLoad(&M2, argv[4]));
    hipFunction_t Fold, Fnew, Pold = nullptr, Pnew = nullptr;
    CK(hipModuleGetFunction(&Fold, M, "k_mx_old"));
    CK(hipModuleGetFunction(&Fnew, M2, "k_mx_new"));
    /* The per-channel fp8 arm (op 43) is optional: the r2only attribution object does not build it. */
    const int have_fp8 = hipModuleGetFunction(&Pold, M, "k_fp8_old") == hipSuccess &&
                         hipModuleGetFunction(&Pnew, M2, "k_fp8_new") == hipSuccess;

    struct Shape {
        const char* name;
        unsigned N, K;
    } shapes[] = {
        /* the shapes the analysis nominated: odd per-wave column count */
        {"glm dense_down", 6144, 3072},
        {"glm kva_fusionA", 2624, 6144},
        {"glm o_proj", 6144, 4096},
        /* K3-like latent widths, tiny per-wave counts where the waste fraction is largest */
        {"k3 q_a-like", 1536, 7168},
        {"k3 kv_a-like", 512, 7168},
        /* EVEN controls: no duplicate exists, so these must not move */
        {"ctl lm_head_like", 32768, 6144},
        {"ctl even q=6", 12288, 7168},
        /* large ODD: the fix at a shape where the body dominates outright */
        {"big odd q=13", 26624, 6144},
    };

    const size_t ARENA = 1500ull << 20;
    unsigned char* W;
    unsigned char* S;
    unsigned short* C1;
    unsigned short* C2;
    void* xb;
    CK(hipMalloc(&W, ARENA));
    CK(hipMalloc(&S, ARENA / 16 + (1 << 20)));
    CK(hipMalloc(&C1, 1 << 22));
    CK(hipMalloc(&C2, 1 << 22));
    CK(hipMalloc(&xb, 1 << 22));
    float* Sf; /* per-CHANNEL fp8 dequant scale: one f32 per output column, not a block grid */
    CK(hipMalloc(&Sf, 1 << 20));
    { std::vector<float> h(1 << 18, 1.0f); CK(hipMemcpy(Sf, h.data(), h.size() * 4, hipMemcpyHostToDevice)); }

    /* VARIED weights, scales and activations. A memset constant would make the bit-exactness check
     * vacuous (every column identical), which is exactly the check that has to carry weight here. */
    {
        const size_t nfill = 64u << 20; /* fill a 64 MB tile and replicate — the pattern is what
                                         * matters, and 1.5 GB of host randomness is pure wall time */
        std::vector<unsigned char> h(nfill);
        unsigned st = 0x12345677u;
        for (size_t i = 0; i < nfill; i++) {
            st = st * 1664525u + 1013904223u;
            h[i] = (unsigned char)(st >> 24);
        }
        for (size_t off = 0; off < ARENA; off += nfill)
            CK(hipMemcpy(W + off, h.data(), std::min(nfill, ARENA - off), hipMemcpyHostToDevice));
        /* E8M0 bytes in 118..134 -> scales 2^-9 .. 2^7. Byte 0 is a legal encoding but decodes to
         * +0.0 and would make whole blocks vanish, hiding differences rather than exposing them. */
        for (size_t i = 0; i < nfill; i++) h[i] = (unsigned char)(118 + (h[i] % 17));
        const size_t sbytes = ARENA / 16 + (1 << 20);
        for (size_t off = 0; off < sbytes; off += nfill)
            CK(hipMemcpy(S + off, h.data(), std::min(nfill, sbytes - off), hipMemcpyHostToDevice));
        /* x: small bf16 magnitudes around 1.0 so no dot overflows and every lane contributes. */
        std::vector<unsigned short> hx(1 << 21);
        for (size_t i = 0; i < hx.size(); i++) {
            st = st * 1664525u + 1013904223u;
            float f = 1.0f + 0.5f * ((float)(int)((st >> 16) & 0xffffu) / 32768.0f - 1.0f);
            unsigned u;
            memcpy(&u, &f, 4);
            hx[i] = (unsigned short)(u >> 16);
        }
        CK(hipMemcpy(xb, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    }

    printf("grid=%d blockDim=%d waves=%u median-of-%d, min of 2 interleaved passes\n", GRID, T,
           WAVES, IT);
    printf("mxfp4 bytes/weight = 0.5 + 1/32 = 0.53125   denominator 6200 GB/s\n\n");
    int all_exact = 1;
    /* TWO ARMS, same table shape: kind 0 = mxfp4 GEMV (op 91), kind 1 = per-channel fp8 GEMV
     * (op 43). Both carry the identical dual-column defect and take the identical fix, so they are
     * reported side by side out of one run rather than from two campaigns. */
    for (int kind = 0; kind < 2; kind++) {
      if (kind == 1 && !have_fp8) continue;
      hipFunction_t Aold = kind ? Pold : Fold, Anew = kind ? Pnew : Fnew;
      void* Sp = kind ? (void*)Sf : (void*)S;
      const double bpw = kind ? 1.0 : 0.53125;
      printf("\n-- %s   %.5f bytes/weight\n", kind ? "per-channel FP8 GEMV (op 43), w8a16"
                                                   : "MXFP4 GEMV (op 91), w4a16", bpw);
      printf("%-17s %6s %6s %5s %7s | %9s %8s | %9s %8s | %7s %s\n", "shape", "N", "K", "q",
             "dupwast", "old us", "GB/s", "new us", "GB/s", "speedup", "bitexact");
      for (auto& sh : shapes) {
        const size_t wpass = kind ? (size_t)sh.N * sh.K : (size_t)sh.N * sh.K / 2;
        const size_t wb = kind ? (size_t)sh.N * sh.K + (size_t)sh.N * 4
                               : (size_t)sh.N * sh.K / 2 + (size_t)sh.N * sh.K / 32;
        unsigned nrep = (unsigned)std::max<size_t>(1, std::min<size_t>(64, ARENA / wpass));
        unsigned one = 1;
        unsigned qlo, qhi;
        double waste;
        wave_cols(sh.N, (unsigned)GRID, WAVES, &qlo, &qhi, &waste);

        /* --- bit-exactness first, one rep each, into two separate output buffers --- */
        void* aold1[] = {&C1, &xb, &W, &Sp, &one, &sh.N, &sh.K};
        void* anew1[] = {&C2, &xb, &W, &Sp, &one, &sh.N, &sh.K};
        CK(hipMemset(C1, 0xAB, (size_t)sh.N * 2));
        CK(hipMemset(C2, 0xCD, (size_t)sh.N * 2));
        CK(hipModuleLaunchKernel(Aold, GRID, 1, 1, T, 1, 1, 0, 0, aold1, nullptr));
        CK(hipModuleLaunchKernel(Anew, GRID, 1, 1, T, 1, 1, 0, 0, anew1, nullptr));
        CK(hipDeviceSynchronize());
        std::vector<unsigned short> h1(sh.N), h2(sh.N);
        CK(hipMemcpy(h1.data(), C1, (size_t)sh.N * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h2.data(), C2, (size_t)sh.N * 2, hipMemcpyDeviceToHost));
        size_t ndiff = 0, npois = 0;
        for (unsigned i = 0; i < sh.N; i++) {
            if (h1[i] != h2[i]) ndiff++;
            if (h1[i] == 0xABAB || h2[i] == 0xCDCD) npois++;
        }
        if (ndiff || npois) all_exact = 0;

        /* --- timing, interleaved old/new twice --- */
        void* aold[] = {&C1, &xb, &W, &Sp, &nrep, &sh.N, &sh.K};
        void* anew[] = {&C2, &xb, &W, &Sp, &nrep, &sh.N, &sh.K};
        /* PALINDROMIC ORDER: old, new, new, old. `old, new, old, new` puts every `new` sample after
         * an `old` one, so any monotone drift over the four passes — sclk ramp, or the downclock a
         * sustained 1.5 GB stream provokes — lands entirely on one arm. It measured as a flat 2.5%
         * against `new` on the EVEN control shapes, where the two arms execute structurally the
         * same code and the honest answer is 1.000x. Palindromic ordering cancels it to first
         * order, and the even rows are what confirm that it did. */
        double o1 = tm(Aold, GRID, aold, IT), n1 = tm(Anew, GRID, anew, IT);
        double n2 = tm(Anew, GRID, anew, IT), o2 = tm(Aold, GRID, aold, IT);
        const double o = std::min(o1, o2) / nrep, n = std::min(n1, n2) / nrep;
        const double ogb = (double)wb / (o * 1e3), ngb = (double)wb / (n * 1e3);
        char qs[16];
        if (qlo == qhi) snprintf(qs, sizeof qs, "%u", qhi);
        else snprintf(qs, sizeof qs, "%u-%u", qlo, qhi);
        printf("%-17s %6u %6u %5s %6.1f%% | %9.2f %8.0f | %9.2f %8.0f | %6.3fx %s\n", sh.name, sh.N,
               sh.K, qs, 100.0 * waste, o, ogb, n, ngb, o / n,
               npois ? "UNWRITTEN!" : ndiff ? "DIFFERS!" : "exact");
        fflush(stdout);
      }
    }
    printf("\n%s\n", all_exact ? "all shapes BIT-EXACT (old == new elementwise)"
                               : "*** NOT BIT-EXACT — the change is wrong ***");
    return all_exact ? 0 : 1;
}
