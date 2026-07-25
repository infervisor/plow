/* tp_p2p_bench.c — cross-GPU (XGMI / Infinity Fabric) transport de-risk for
 * tensor-parallel decode. Phase 0. Proves peer access works and MEASURES it.
 *
 * Built by scripts/build_tp_p2p.sh (system gcc, clean env). Run on a node with
 * >= 2 idle gfx950 GPUs. Pick the pair on argv:  ./tp_p2p_bench [devA] [devB]
 * (default 0 1). Every number printed is measured on THIS node, this run.
 *
 * What it does, in order:
 *   [1] TOPOLOGY   — HSA link info the collective planner will read.
 *   [2] PEER R/W   — GPU_A kernel writes peer VRAM, GPU_B kernel reads+verifies.
 *   [3] SDMA P2P   — device->device copy bandwidth (big) and latency (8 KB).
 *   [4] HANDSHAKE  — cross-GPU ping-pong on ONE system-scope atomic. The
 *                    decode-relevant number: the counter-gate round-trip.
 *   [5] ONE-SHOT   — the all-reduce reduction step over peer buffers.
 */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
                              else { printf("  ok:   %s\n", msg); } } while (0)

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Kernarg blocks — must match the HIP kernel signatures byte-for-byte. Natural
 * alignment matches the AMDGPU kernarg ABI for scalar/pointer args. */
typedef struct { void* buf; uint32_t n; uint32_t seed; } arg_fill;
typedef struct { const void* buf; uint32_t n; uint32_t seed; void* errs; } arg_verify;
typedef struct { void* p2c; void* c2p; uint32_t iters; uint64_t deadline;
                 void* cycles; void* status; } arg_ping;
typedef struct { void* p2c; void* c2p; uint32_t iters; uint64_t deadline; } arg_pong;
typedef struct { void* out; const void* peers; uint32_t nranks; uint32_t n; } arg_reduce;

/* Read a whole file (the unbundled .elf code object). */
static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(n); if (fread(p, 1, n, f) != (size_t)n) { exit(2); }
    fclose(f); *len = n; return p;
}

int main(int argc, char** argv) {
    const int A = argc > 1 ? atoi(argv[1]) : 0;
    const int B = argc > 2 ? atoi(argv[2]) : 1;
    const char* elf_path = getenv("TP_ELF") ? getenv("TP_ELF")
                                            : "tp_p2p_kernels.elf";

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("plow_hsa_init: %s\n", plow_hsa_last_error()); return 2; }
    const int ndev = plow_hsa_device_count(h);
    printf("== plow cross-GPU transport de-risk (gfx950 / XGMI) ==\n");
    printf("GPUs discovered: %d   test pair: GPU%d <-> GPU%d\n\n", ndev, A, B);
    if (A >= ndev || B >= ndev || A == B) {
        printf("need two distinct valid GPUs\n"); return 2;
    }

    /* s_memrealtime frequency: the constant-rate clock the ping kernel reads.
     * HSA exposes it as the system timestamp frequency. */
    uint64_t ts_freq = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &ts_freq);
    const double tick_ns = ts_freq ? 1e9 / (double)ts_freq : 10.0;

    char nameA[64], nameB[64]; uint32_t cus, lds;
    plow_hsa_device_info(h, A, nameA, &cus, &lds);
    plow_hsa_device_info(h, B, nameB, &cus, &lds);

    /* ---- [1] topology --------------------------------------------------- */
    printf("[1] TOPOLOGY\n");
    printf("  GPU%d = %s (%u CUs)   GPU%d = %s\n", A, nameA, cus, B, nameB);
    printf("  s_memrealtime freq = %.3f MHz  (%.2f ns/tick)\n\n",
           ts_freq / 1e6, tick_ns);

    /* Load the same code object on both GPUs. */
    size_t elf_len; void* elf = slurp(elf_path, &elf_len);
    if (plow_hsa_load_code_object(h, A, elf, elf_len) != 0 ||
        plow_hsa_load_code_object(h, B, elf, elf_len) != 0) {
        printf("load_code_object: %s\n", plow_hsa_last_error()); return 2;
    }
    plow_hsa_kernel k_fill, k_verify_B, k_ping_A, k_pong_B, k_reduce_B;
    if (plow_hsa_get_kernel(h, A, "tp_fill", &k_fill) != 0 ||
        plow_hsa_get_kernel(h, B, "tp_verify", &k_verify_B) != 0 ||
        plow_hsa_get_kernel(h, A, "tp_ping", &k_ping_A) != 0 ||
        plow_hsa_get_kernel(h, B, "tp_pong", &k_pong_B) != 0 ||
        plow_hsa_get_kernel(h, B, "tp_reduce_oneshot", &k_reduce_B) != 0) {
        printf("get_kernel: %s\n", plow_hsa_last_error()); return 2;
    }

    /* ---- [2] peer read/write proof ------------------------------------- */
    printf("[2] PEER READ/WRITE  (GPU%d writes peer VRAM, GPU%d reads it)\n", A, B);
    const uint32_t N = 4u * 1024 * 1024;              /* 16 MB of uint32 */
    uint32_t* buf = (uint32_t*)plow_hsa_alloc_peer(h, A, (size_t)N * 4);
    uint32_t* errs = (uint32_t*)plow_hsa_alloc(h, B, 4);
    if (!buf || !errs) { printf("alloc: %s\n", plow_hsa_last_error()); return 2; }
    const uint32_t seed = 0x1234abcdu, zero = 0;
    plow_hsa_upload(h, B, errs, &zero, 4);

    arg_fill af = { buf, N, seed };
    plow_hsa_launch(h, A, &k_fill, 65536, 1, 1, 256, 1, 1, 0, &af, sizeof af);
    plow_hsa_wait(h, A);

    arg_verify av = { buf, N, seed, errs };
    plow_hsa_launch(h, B, &k_verify_B, 65536, 1, 1, 256, 1, 1, 0, &av, sizeof av);
    plow_hsa_wait(h, B);

    uint32_t nerr = 1; plow_hsa_download(h, B, &nerr, errs, 4);
    CHECK(nerr == 0, "GPU_B kernel read GPU_A's 16 MB peer buffer byte-exact");
    printf("\n");

    /* ---- [3] SDMA P2P bandwidth + latency ------------------------------ */
    printf("[3] SDMA P2P  (hsa_amd_memory_async_copy over XGMI)\n");
    const size_t BIG = 256u * 1024 * 1024;
    void* srcA = plow_hsa_alloc_peer(h, A, BIG);
    void* dstB = plow_hsa_alloc_peer(h, B, BIG);
    /* warm up the copy engine / route */
    for (int i = 0; i < 3; i++) plow_hsa_copy_p2p(h, B, dstB, A, srcA, BIG);
    const int BW_ITERS = 20;
    double t = now_s();
    for (int i = 0; i < BW_ITERS; i++) plow_hsa_copy_p2p(h, B, dstB, A, srcA, BIG);
    double dt = now_s() - t;
    double gbps = (double)BIG * BW_ITERS / dt / 1e9;
    printf("  unidir  GPU%d->GPU%d  %.0f MB x%d  = %.1f GB/s\n",
           A, B, BIG / 1e6, BW_ITERS, gbps);

    /* bidirectional (serialized blocking copies both ways) — full-duplex check */
    t = now_s();
    for (int i = 0; i < BW_ITERS; i++) {
        plow_hsa_copy_p2p(h, B, dstB, A, srcA, BIG);
        plow_hsa_copy_p2p(h, A, srcA, B, dstB, BIG);
    }
    dt = now_s() - t;
    printf("  bidir   GPU%d<->GPU%d aggregate = %.1f GB/s\n",
           A, B, 2.0 * BIG * BW_ITERS / dt / 1e9);

    /* small-message latency: 8 KB (decode all-reduce message size class) */
    const size_t SMALL = 8u * 1024;
    for (int i = 0; i < 100; i++) plow_hsa_copy_p2p(h, B, dstB, A, srcA, SMALL);
    const int LAT_ITERS = 2000;
    t = now_s();
    for (int i = 0; i < LAT_ITERS; i++) plow_hsa_copy_p2p(h, B, dstB, A, srcA, SMALL);
    dt = now_s() - t;
    printf("  latency 8 KB SDMA copy = %.2f us/transfer  (%d iters)\n\n",
           dt / LAT_ITERS * 1e6, LAT_ITERS);

    /* ---- [3b] kernel peer-STORE bandwidth ------------------------------ */
    /* The decode collective does NOT use SDMA: it writes peer VRAM directly
     * from a fused kernel. Measure that path — GPU_A's kernel storing into
     * GPU_B's buffer over XGMI. Large transfer so launch+wait overhead
     * amortizes; the per-message decode transfer is then bytes / this BW. */
    void* storeB = plow_hsa_alloc_peer(h, B, BIG);
    arg_fill asf = { storeB, (uint32_t)(BIG / 4), 0x55u };
    for (int i = 0; i < 3; i++) {
        plow_hsa_launch(h, A, &k_fill, 65536, 1, 1, 256, 1, 1, 0, &asf, sizeof asf);
        plow_hsa_wait(h, A);
    }
    const int ST_ITERS = 30;
    t = now_s();
    for (int i = 0; i < ST_ITERS; i++) {
        plow_hsa_launch(h, A, &k_fill, 65536, 1, 1, 256, 1, 1, 0, &asf, sizeof asf);
        plow_hsa_wait(h, A);
    }
    dt = now_s() - t;
    const double store_gbps = (double)BIG * ST_ITERS / dt / 1e9;
    printf("  kernel peer-store GPU%d->GPU%d = %.1f GB/s  (fused-collective path)\n\n",
           A, B, store_gbps);
    plow_hsa_free(h, storeB);

    /* ---- [4] cross-GPU handshake (system-scope atomic ping-pong) -------- */
    printf("[4] CROSS-GPU HANDSHAKE  (ONE system-scope atomic, no HSA signal)\n");
    /* two flag words in ONE peer buffer on GPU_A, visible to both kernels */
    uint32_t* flags = (uint32_t*)plow_hsa_alloc_peer(h, A, 4 * sizeof(uint32_t));
    uint32_t clr[2] = { 0, 0 }; plow_hsa_upload(h, A, flags, clr, sizeof clr);
    uint64_t* cyc = (uint64_t*)plow_hsa_alloc(h, A, 8);
    uint32_t* pstat = (uint32_t*)plow_hsa_alloc(h, A, 4);
    const uint32_t PP = 10000;
    const uint64_t deadline = (uint64_t)(ts_freq ? ts_freq : 100000000); /* ~1 s */

    /* Launch the responder (GPU_B) FIRST so it is spinning, then the timed
     * pinger (GPU_A). Both queues run concurrently. */
    arg_pong pg = { flags, flags + 1, PP, deadline };
    plow_hsa_launch(h, B, &k_pong_B, 1, 1, 1, 1, 1, 1, 0, &pg, sizeof pg);
    arg_ping pi = { flags, flags + 1, PP, deadline, cyc, pstat };
    plow_hsa_launch(h, A, &k_ping_A, 1, 1, 1, 1, 1, 1, 0, &pi, sizeof pi);
    plow_hsa_wait(h, A); plow_hsa_wait(h, B);

    uint32_t st = 0; uint64_t cycles = 0;
    plow_hsa_download(h, A, &st, pstat, 4);
    plow_hsa_download(h, A, &cycles, cyc, 8);
    if (st == 0) {
        const double rt_ns = (double)cycles * tick_ns / PP;
        printf("  system-scope atomic over XGMI: WORKS (no HSA signal needed)\n");
        printf("  round-trip = %.2f us   one-way handshake = %.2f us  (%u pairs)\n\n",
               rt_ns / 1e3, rt_ns / 2e3, PP);
        CHECK(1, "cross-GPU counter-gate via system-scope atomic");
    } else {
        printf("  system-scope atomic DID NOT propagate (status=0x%08x)\n", st);
        printf("  -> HSA signals required for cross-GPU sync\n\n");
        CHECK(0, "cross-GPU counter-gate via system-scope atomic");
    }

    /* ---- [5] one-shot all-reduce reduction step ------------------------ */
    printf("[5] ONE-SHOT reduction step (sum N peer partials)\n");
    const uint32_t RN = 1920;               /* Gemma-4 hidden = 3840 bf16 ~ 7.7KB */
    const uint32_t NR = 2;
    float* pA = (float*)plow_hsa_alloc_peer(h, A, (size_t)RN * 4);
    float* pB = (float*)plow_hsa_alloc_peer(h, B, (size_t)RN * 4);
    float* rout = (float*)plow_hsa_alloc(h, B, (size_t)RN * 4);
    void** peers = (void**)plow_hsa_alloc_peer(h, B, NR * sizeof(void*));
    float* hs = malloc((size_t)RN * 4);
    for (uint32_t i = 0; i < RN; i++) hs[i] = 1.0f;  plow_hsa_upload(h, A, pA, hs, RN * 4);
    for (uint32_t i = 0; i < RN; i++) hs[i] = 2.0f;  plow_hsa_upload(h, B, pB, hs, RN * 4);
    void* pv[2] = { pA, pB }; plow_hsa_upload(h, B, peers, pv, sizeof pv);
    arg_reduce ar = { rout, peers, NR, RN };
    plow_hsa_launch(h, B, &k_reduce_B, 4096, 1, 1, 256, 1, 1, 0, &ar, sizeof ar);
    plow_hsa_wait(h, B);
    plow_hsa_download(h, B, hs, rout, RN * 4);
    int rok = 1; for (uint32_t i = 0; i < RN; i++) if (hs[i] != 3.0f) { rok = 0; break; }
    CHECK(rok, "one-shot reduce summed 2 ranks' peer buffers (1.0+2.0=3.0)");

    printf("\n%s\n", fails ? "== FAILURES ==" : "== ALL TRANSPORT CHECKS PASSED ==");
    plow_hsa_shutdown(h);
    return fails ? 1 : 0;
}
