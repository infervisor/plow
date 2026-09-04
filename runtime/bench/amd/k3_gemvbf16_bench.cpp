/* Host driver for k3_gemvbf16_bench.hip — the bf16 decode GEMV at K3's own TP8 decode shapes.
 *
 * Shapes and grids come from `plowrt disasm <blob>/model.pkt --program 1` on the shipped 93-layer
 * TP8 fp8-KV asset: (N, K, nblk, instances-per-token). `nblk` is the emitter's own dispatch width
 * and is NOT always 256 — the router GEMV runs 224 and `b_proj` runs 12 — so the grid has to come
 * from the packet, not from the CU count.
 *
 * Arms are interleaved in ONE process, each a median-of-IT hipEvent measurement, and `k_packet_aa`
 * is byte-identical device code to `k_packet` so the run reports its own noise floor. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            printf("HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));                  \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static const int T = 512; /* PLOW_THREADS */

/* `zero`/`zb`: a buffer re-zeroed on the stream before EVERY launch (the k_dyn row-claim
 * cursors), outside the event window. */
static double tm(hipFunction_t f, int g, void** a, int it, void* zero = nullptr, size_t zb = 0) {
    for (int i = 0; i < 3; i++) {
        if (zero) CK(hipMemsetAsync(zero, 0, zb, 0));
        CK(hipModuleLaunchKernel(f, g, 1, 1, T, 1, 1, 0, 0, a, nullptr));
    }
    CK(hipDeviceSynchronize());
    std::vector<double> v;
    hipEvent_t s, e;
    CK(hipEventCreate(&s));
    CK(hipEventCreate(&e));
    for (int i = 0; i < it; i++) {
        if (zero) CK(hipMemsetAsync(zero, 0, zb, 0));
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

int main(int argc, char** argv) {
    CK(hipInit(0));
    hipModule_t M;
    CK(hipModuleLoad(&M, argc > 1 ? argv[1] : "/tmp/gvbf.co"));
    const int IT = argc > 2 ? atoi(argv[2]) : 41;

    const char* sym[] = {"k_steady", "k_packet", "k_packet_aa", "k_r2", "k_wstage", "k_stageonly",
                         "k_noldsx", "k_comb", "k_rA", "k_rL", "k_rF", "k_dyn0", "k_dyn"};
    const int NBASE = 11;
    /* k_dyn0/k_dyn exist only in a -DPLOW_GV_DYNCLAIM=1 object. */
    int NARM = 13;
    hipFunction_t fn[13];
    for (int i = 0; i < NBASE; i++) CK(hipModuleGetFunction(&fn[i], M, sym[i]));
    for (int i = NBASE; i < 13; i++)
        if (hipModuleGetFunction(&fn[i], M, sym[i]) != hipSuccess) NARM = NBASE;
    const bool dyn = NARM > NBASE;
    /* One 128-B cursor line per rep, re-zeroed before every k_dyn launch. */
    const size_t CUR_BYTES = 256 * 128;
    unsigned* cur;
    CK(hipMalloc(&cur, CUR_BYTES));
    CK(hipMemset(cur, 0, CUR_BYTES));

    struct Shape {
        const char* name;
        unsigned N, K, blk, inst;
    } shapes[] = {
        {"moe_down_latent", 3584, 7168, 256, 92}, {"routed_up", 896, 3584, 224, 92},
        {"o_proj", 7168, 1536, 256, 93},          {"router_gate", 896, 7168, 224, 92},
        {"shared_down", 7168, 768, 256, 92},      {"q_a", 1536, 7168, 256, 48},
        {"q_absorb", 6144, 1536, 256, 24},        {"kv_a", 512, 7168, 256, 24},
        {"f_a", 128, 7168, 128, 69},              {"q_rope", 768, 1536, 256, 24},
        {"f_b", 1536, 128, 256, 69},              {"k_rope_d", 64, 7168, 64, 24},
        {"b_proj", 12, 7168, 12, 69},             {"final_gate", 4224, 7168, 249, 2},
        {"final_proj", 7168, 4224, 256, 1},        {"lm_head", 163840, 7168, 256, 1},
    };
    const int NS = (int)(sizeof(shapes) / sizeof(shapes[0]));

    const size_t ARENA = 3ull << 30;
    unsigned short *W, *C1, *C2, *xb;
    CK(hipMalloc(&W, ARENA));
    CK(hipMalloc(&C1, 1u << 22));
    CK(hipMalloc(&C2, 1u << 22));
    CK(hipMalloc(&xb, 1u << 22));
    {
        const size_t nfill = 64u << 20;
        std::vector<unsigned short> h(nfill / 2);
        unsigned st = 0x12345677u;
        for (size_t i = 0; i < h.size(); i++) {
            st = st * 1664525u + 1013904223u;
            /* small bf16 magnitudes: exponent near 127 so the dot products stay finite */
            h[i] = (unsigned short)(0x3B00u | ((st >> 20) & 0x7Fu));
        }
        for (size_t o = 0; o < ARENA; o += nfill)
            CK(hipMemcpy((char*)W + o, h.data(), std::min(nfill, ARENA - o), hipMemcpyHostToDevice));
        CK(hipMemcpy(xb, h.data(), 1u << 22, hipMemcpyHostToDevice));
    }

    printf("%-16s %7s %6s %4s %5s %4s | %9s %9s %9s | %8s %8s %8s %8s %8s %8s %8s %8s%s\n", "shape", "N", "K", "blk",
           "inst", "nrep", "steady_us", "packet_us", "stage_us", "GBs_pkt", "r2/pkt", "wstg/pkt",
           "nolds/pkt", "comb/pkt", "rA/pkt", "rL/pkt", "rF/pkt",
           dyn ? "  dyn0_us   dyn_us dyn0/dyn" : "");
    double tot_pkt = 0, tot_stdy = 0, tot_r2 = 0, tot_ws = 0, tot_nl = 0, tot_cb = 0, tot_ra = 0,
           tot_rl = 0, tot_rf = 0, tot_d0 = 0, tot_dy = 0;
    double aa_lo = 9, aa_hi = 0;
    /* argv[3]: override the grid of every wide (blk >= 64) shape, e.g. 256 = the emitter's current
     * b for these packets (the table's 128 is the older tuned width). */
    const unsigned BLK = argc > 3 ? (unsigned)atoi(argv[3]) : 0u;
    for (int s = 0; s < NS; s++) {
        Shape S = shapes[s];
        if (BLK && S.blk >= 64) S.blk = BLK;
        const size_t slab = (size_t)S.N * S.K * 2;
        unsigned nrep = (unsigned)std::min<size_t>(ARENA / slab, 256);
        if (nrep < 1) nrep = 1;
        void* a[] = {&C1, &xb, &W, &nrep, (void*)&S.N, (void*)&S.K, &cur};
        /* PALINDROMIC interleave: forward pass then reverse pass, min per arm. A monotone drift
         * (clock ramp, a neighbour's kernel arriving) then biases both directions equally. */
        double us[13];
        for (int i = 0; i < NARM; i++) us[i] = 1e30;
        for (int i = 0; i < NARM; i++)
            us[i] = std::min(us[i], tm(fn[i], (int)S.blk, a, IT, i == 12 ? cur : nullptr,
                                       CUR_BYTES) / nrep);
        for (int i = NARM - 1; i >= 0; i--)
            us[i] = std::min(us[i], tm(fn[i], (int)S.blk, a, IT, i == 12 ? cur : nullptr,
                                       CUR_BYTES) / nrep);
        if (dyn) {
            /* k_dyn must be bit-identical to the shipping packet body: only the row->workgroup
             * map moves. Every rep is checked (the cursor is zeroed per launch, one line per rep). */
            unsigned one = 1;
            void* b1[] = {&C1, &xb, &W, &one, (void*)&S.N, (void*)&S.K, &cur};
            void* b2[] = {&C2, &xb, &W, &one, (void*)&S.N, (void*)&S.K, &cur};
            CK(hipMemset(C1, 0, 1u << 22));
            CK(hipMemset(C2, 0, 1u << 22));
            CK(hipMemset(cur, 0, CUR_BYTES));
            CK(hipModuleLaunchKernel(fn[1], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b1, nullptr));
            CK(hipModuleLaunchKernel(fn[12], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b2, nullptr));
            CK(hipDeviceSynchronize());
            std::vector<unsigned short> h1(S.N), h2(S.N);
            CK(hipMemcpy(h1.data(), C1, S.N * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(h2.data(), C2, S.N * 2, hipMemcpyDeviceToHost));
            size_t bad = 0;
            for (unsigned i = 0; i < S.N; i++) bad += (h1[i] != h2[i]);
            if (bad) printf("!! %s: %zu/%u outputs DIFFER (dyn vs packet)\n", S.name, bad, S.N);
            std::vector<unsigned> hc(32);
            CK(hipMemcpy(hc.data(), cur, 32 * 4, hipMemcpyDeviceToHost));
            printf("   %-14s dyn claims %u (pool items + %u empty)\n", S.name, hc[0], S.blk);
        }
        /* BIT-EXACTNESS: the candidate only reorders which wave takes which column and how x is
         * copied into LDS, so every output element must match the shipping body exactly. */
        {
            unsigned one = 1;
            void* b1[] = {&C1, &xb, &W, &one, (void*)&S.N, (void*)&S.K};
            void* b2[] = {&C2, &xb, &W, &one, (void*)&S.N, (void*)&S.K};
            CK(hipMemset(C1, 0, 1u << 22));
            CK(hipMemset(C2, 0, 1u << 22));
            CK(hipModuleLaunchKernel(fn[1], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b1, nullptr));
            CK(hipModuleLaunchKernel(fn[9], (int)S.blk, 1, 1, T, 1, 1, 0, 0, b2, nullptr));
            CK(hipDeviceSynchronize());
            std::vector<unsigned short> h1(S.N), h2(S.N);
            CK(hipMemcpy(h1.data(), C1, S.N * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(h2.data(), C2, S.N * 2, hipMemcpyDeviceToHost));
            size_t bad = 0;
            for (unsigned i = 0; i < S.N; i++) bad += (h1[i] != h2[i]);
            if (bad) printf("!! %s: %zu/%u outputs DIFFER (rL vs packet)\n", S.name, bad, S.N);
        }
        const double gb = (double)slab / 1e9;
        const double aa = us[2] / us[1];
        aa_lo = std::min(aa_lo, aa);
        aa_hi = std::max(aa_hi, aa);
        printf("%-16s %7u %6u %4u %5u %4u | %9.3f %9.3f %9.3f | %8.0f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f",
               S.name, S.N, S.K, S.blk, S.inst, nrep, us[0], us[1], us[5], gb / (us[1] * 1e-6),
               us[1] / us[3], us[1] / us[4], us[1] / us[6], us[1] / us[7], us[1] / us[8],
               us[1] / us[9], us[1] / us[10]);
        if (dyn) {
            printf(" %8.3f %8.3f %8.3f", us[11], us[12], us[11] / us[12]);
            tot_d0 += us[11] * S.inst / 1e3;
            tot_dy += us[12] * S.inst / 1e3;
        }
        printf("\n");
        tot_stdy += us[0] * S.inst / 1e3;
        tot_pkt += us[1] * S.inst / 1e3;
        tot_r2 += us[3] * S.inst / 1e3;
        tot_ws += us[4] * S.inst / 1e3;
        tot_nl += us[6] * S.inst / 1e3;
        tot_cb += us[7] * S.inst / 1e3;
        tot_ra += us[8] * S.inst / 1e3;
        tot_rl += us[9] * S.inst / 1e3;
        tot_rf += us[10] * S.inst / 1e3;
    }
    printf("\nA/A control (pkt_aa / packet): %.4f .. %.4f\n", aa_lo, aa_hi);
    printf("per-token totals over the tabulated instance counts:\n");
    printf("  steady %.3f  packet %.3f  r2 %.3f  wstage %.3f  noldsx %.3f  COMB %.3f  rA %.3f  rL %.3f  rF %.3f ms\n",
           tot_stdy, tot_pkt, tot_r2, tot_ws, tot_nl, tot_cb, tot_ra, tot_rl, tot_rf);
    if (dyn) printf("  dyn0 (production static) %.3f  dyn (PLOW_GV_DYNCLAIM) %.3f ms\n", tot_d0, tot_dy);
    return 0;
}
