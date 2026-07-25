/* tp_allreduce_bench.c — 2-GPU validation of the one-shot all-reduce collective.
 *
 * Standalone correctness + latency proof for the runtime collective (P1-B). Two
 * ranks each publish a known partial vector into peer VRAM; both run op_collective.h's
 * d_xreduce_oneshot (publish-signal + SYSTEM-scope xctr gate + f32-accumulate reduce)
 * and must land the bit-exact element-wise sum. The per-collective latency is timed
 * on rank A's s_memrealtime — it should match the transport budget (~0.35-0.5 us).
 *
 * Built by scripts/build_tp_allreduce.sh (nix hipcc device, system gcc host, clean
 * env — the same toolchain contract as build_tp_p2p.sh). Run on >=2 idle gfx950:
 *   ./tp_allreduce_bench [devA] [devB]      (default 0 1)
 */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

/* Kernarg blocks — must match the HIP kernel signatures byte-for-byte. */
typedef struct { void* part; uint32_t n; uint32_t rank; } arg_fill;
typedef struct {
    void* out; const void* peer_scratch; const void* peer_gate;
    uint32_t nranks; uint32_t rank; uint32_t n; uint32_t slot_bytes;
    uint32_t iters; uint64_t deadline; void* cycles; void* status;
} arg_ar;

static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(n); if (fread(p, 1, n, f) != (size_t)n) exit(2);
    fclose(f); *len = n; return p;
}

int main(int argc, char** argv) {
    const int dev[2] = { argc > 1 ? atoi(argv[1]) : 0, argc > 2 ? atoi(argv[2]) : 1 };
    const uint32_t NR = 2;
    const uint32_t N = 3840;               /* Gemma-4 12B hidden = 7.7 KB bf16 all-reduce */
    const char* elf_path = getenv("TP_ELF") ? getenv("TP_ELF") : "tp_allreduce_kernels.elf";
    int fails = 0;

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("plow_hsa_init: %s\n", plow_hsa_last_error()); return 2; }
    const int ndev = plow_hsa_device_count(h);
    printf("== plow one-shot all-reduce (gfx950 / XGMI, 2 ranks) ==\n");
    printf("GPUs discovered: %d   ranks: GPU%d, GPU%d   N=%u (%.1f KB bf16)\n\n",
           ndev, dev[0], dev[1], N, N * 2 / 1024.0);
    if (dev[0] >= ndev || dev[1] >= ndev || dev[0] == dev[1]) {
        printf("need two distinct valid GPUs\n"); return 2;
    }

    uint64_t ts_freq = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &ts_freq);
    const double tick_ns = ts_freq ? 1e9 / (double)ts_freq : 1.0;

    size_t elf_len; void* elf = slurp(elf_path, &elf_len);
    plow_hsa_kernel k_fill[2], k_ar[2];
    for (int i = 0; i < 2; i++) {
        if (plow_hsa_load_code_object(h, dev[i], elf, elf_len) != 0 ||
            plow_hsa_get_kernel(h, dev[i], "tp_fill_partial", &k_fill[i]) != 0 ||
            plow_hsa_get_kernel(h, dev[i], "tp_allreduce", &k_ar[i]) != 0) {
            printf("load/get_kernel: %s\n", plow_hsa_last_error()); return 2;
        }
    }

    /* Per-rank peer buffers: partial slot, gate word, local output, pointer tables. */
    void* scratch[2]; void* gate[2]; void* out[2]; void* scr_tbl[2]; void* gate_tbl[2];
    uint32_t* cyc = NULL; uint32_t* stat[2];
    for (int i = 0; i < 2; i++) {
        scratch[i] = plow_hsa_alloc_peer(h, dev[i], (size_t)N * 2);
        gate[i]    = plow_hsa_alloc_peer(h, dev[i], 128);       /* one cache-line counter */
        out[i]     = plow_hsa_alloc(h, dev[i], (size_t)N * 2);
        scr_tbl[i] = plow_hsa_alloc(h, dev[i], NR * sizeof(void*));
        gate_tbl[i]= plow_hsa_alloc(h, dev[i], NR * sizeof(void*));
        stat[i]    = (uint32_t*)plow_hsa_alloc(h, dev[i], 4);
        if (!scratch[i] || !gate[i] || !out[i] || !scr_tbl[i] || !gate_tbl[i] || !stat[i]) {
            printf("alloc: %s\n", plow_hsa_last_error()); return 2;
        }
        uint32_t z[32] = {0};
        plow_hsa_upload(h, dev[i], gate[i], z, 128);
        plow_hsa_upload(h, dev[i], stat[i], z, 4);
    }
    cyc = (uint32_t*)plow_hsa_alloc(h, dev[0], 8);

    /* Fill each rank's partial with its element-varying bf16 pattern. */
    for (int i = 0; i < 2; i++) {
        arg_fill af = { scratch[i], N, (uint32_t)i };
        plow_hsa_launch(h, dev[i], &k_fill[i], 4096, 1, 1, 256, 1, 1, 0, &af, sizeof af);
        plow_hsa_wait(h, dev[i]);
    }
    /* Publish the peer pointer tables (same virtual address on every agent). */
    void* scr_pv[2] = { scratch[0], scratch[1] };
    void* gate_pv[2] = { gate[0], gate[1] };
    for (int i = 0; i < 2; i++) {
        plow_hsa_upload(h, dev[i], scr_tbl[i], scr_pv, sizeof scr_pv);
        plow_hsa_upload(h, dev[i], gate_tbl[i], gate_pv, sizeof gate_pv);
    }

    const uint32_t ITERS = 10000;
    const uint64_t deadline = (uint64_t)(ts_freq ? ts_freq : 1000000000); /* ~1 s */
    /* 64 workgroups x 512 threads. Launch rank B first so it is spinning, then the
     * timed rank A; both queues run concurrently. */
    arg_ar aB = { out[1], scr_tbl[1], gate_tbl[1], NR, 1, N, 0, ITERS, deadline, NULL, stat[1] };
    plow_hsa_launch(h, dev[1], &k_ar[1], 64 * 512, 1, 1, 512, 1, 1, 0, &aB, sizeof aB);
    arg_ar aA = { out[0], scr_tbl[0], gate_tbl[0], NR, 0, N, 0, ITERS, deadline, cyc, stat[0] };
    plow_hsa_launch(h, dev[0], &k_ar[0], 64 * 512, 1, 1, 512, 1, 1, 0, &aA, sizeof aA);
    plow_hsa_wait(h, dev[0]); plow_hsa_wait(h, dev[1]);

    uint32_t st0 = 0, st1 = 0; plow_hsa_download(h, dev[0], &st0, stat[0], 4);
    plow_hsa_download(h, dev[1], &st1, stat[1], 4);
    if (st0 || st1) {
        printf("  FAIL: xctr gate did not propagate (status A=0x%08x B=0x%08x)\n", st0, st1);
        printf("  -> system-scope atomic over peer VRAM not coherent on this fabric\n");
        plow_hsa_shutdown(h); return 1;
    }

    /* Correctness: both ranks must hold the bit-exact element-wise sum. */
    bf16* hb = malloc((size_t)N * 2);
    for (int i = 0; i < 2; i++) {
        plow_hsa_download(h, dev[i], hb, out[i], (size_t)N * 2);
        int bad = -1;
        for (uint32_t e = 0; e < N; e++) {
            float sum = 0.0f;
            for (uint32_t r = 0; r < NR; r++)
                sum += bf2f(f2bf((float)(r + 1) * (1.0f + (float)(e & 7u) * 0.125f)));
            if (hb[e] != f2bf(sum)) { bad = (int)e; break; }
        }
        if (bad < 0) printf("  ok:   rank %d output bit-exact (all %u elements)\n", i, N);
        else { printf("  FAIL: rank %d output wrong at e=%d (got %.4f want %.4f)\n",
                      i, bad, bf2f(hb[bad]),
                      bf2f(f2bf(bf2f(f2bf((float)(0 + 1) * (1.0f + (float)(bad & 7) * 0.125f)))
                              + bf2f(f2bf((float)(1 + 1) * (1.0f + (float)(bad & 7) * 0.125f)))))); fails++; }
    }

    uint64_t cycles = 0; plow_hsa_download(h, dev[0], &cycles, cyc, 8);
    const double per_us = (double)cycles * tick_ns / ITERS / 1e3;
    printf("\n  one-shot all-reduce latency = %.3f us/collective  (%u iters, N=%u, %d ranks)\n",
           per_us, ITERS, N, NR);
    printf("  transport budget (tp-transport.md): ~0.35-0.5 us  ->  %s\n",
           per_us < 1.0 ? "WITHIN budget" : "ABOVE budget (investigate)");

    printf("\n%s\n", fails ? "== FAILURES ==" : "== ALL-REDUCE VALIDATED (bit-exact + latency) ==");
    plow_hsa_shutdown(h);
    return fails ? 1 : 0;
}
