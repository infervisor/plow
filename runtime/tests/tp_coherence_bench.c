/* tp_coherence_bench.c — host driver for the adversarial cross-GPU coherence test.
 *
 * Decides whether the flat one-shot all-reduce's system-scope xctr handshake is a
 * real happens-before on gfx950 XGMI, or a hardware accident. See
 * tp_coherence_kernels.hip for the mechanism.
 *
 * For each (variant, GPU-pair) it runs a strict lockstep producer/consumer
 * ping-pong over a tiny resident peer buffer and reports how many of the `iters`
 * observations were STALE (consumer read a value other than this iteration's).
 *
 *   SHIPPED sys/sys   the real xctr path — MUST be 0 stale to claim bit-exact TP.
 *   CTRL rel/none     positive control (no release, no acquire) — MUST be > 0
 *                     stale, else the test has no detection power on this HW.
 *   sys/none          isolates the consumer acquire.
 *   none/sys          isolates the producer release.
 *   agent/agent       what a device-scope-only implementation would give.
 *
 * Build: system gcc, clean env (build_tp_coherence.sh). Needs
 * tp_coherence_kernels.elf in cwd.  Run:  ./tp_coherence_bench [iters] [pairsCSV]
 */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SC_SYS 0
#define SC_AGENT 1
#define SC_RELAXED 2

typedef struct { void* data; uint32_t ndata; void* g_ready; void* g_done;
                 uint32_t iters; int scope; uint64_t deadline; void* status; } arg_prod;
typedef struct { const void* data; uint32_t ndata; void* g_ready; void* g_done;
                 uint32_t iters; int scope; uint64_t deadline; void* stale;
                 void* first_bad; void* status; } arg_cons;

static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n); if (fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f); *len = (size_t)n; return p;
}

typedef struct { const char* name; int rel, acq; } Variant;

int main(int argc, char** argv) {
    const uint32_t iters = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 1000000u;
    const char* pairs_csv = argc > 2 ? argv[2] : "0-1,0-4,4-0,3-4";
    const uint32_t NDATA = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 512;

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("plow_hsa_init: %s\n", plow_hsa_last_error()); return 2; }
    const int ndev = plow_hsa_device_count(h);
    uint64_t ts_freq = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &ts_freq);
    if (!ts_freq) ts_freq = 1000000000ull;      /* gfx950: 1 ns/tick fallback */
    const uint64_t deadline = 3ull * ts_freq;   /* ~3 s per single wait */

    printf("== plow ADVERSARIAL cross-GPU coherence stress (gfx950 / XGMI) ==\n");
    printf("GPUs=%d  iters=%u  ndata=%u words (%u B resident)  ts_freq=%.1f MHz\n",
           ndev, iters, NDATA, NDATA * 4u, ts_freq / 1e6);
    printf("checks per (variant,pair) = %llu\n\n",
           (unsigned long long)iters * NDATA);

    size_t elf_len; void* elf = slurp("tp_coherence_kernels.elf", &elf_len);
    for (int d = 0; d < ndev; d++)
        if (plow_hsa_load_code_object(h, d, elf, elf_len)) {
            printf("dev%d load: %s\n", d, plow_hsa_last_error()); return 2; }

    const Variant V[] = {
        { "SHIPPED  sys/sys ", SC_SYS,     SC_SYS     },
        { "CTRL     rel/none", SC_RELAXED, SC_RELAXED },
        { "acq-only sys/none", SC_SYS,     SC_RELAXED },
        { "rel-only none/sys", SC_RELAXED, SC_SYS     },
        { "agent    agt/agt ", SC_AGENT,   SC_AGENT   },
    };
    const int NV = (int)(sizeof V / sizeof V[0]);

    printf("%-6s %-18s %14s %10s %12s\n", "pair", "variant", "stale", "status", "first_bad");
    printf("--------------------------------------------------------------------\n");

    int any_shipped_stale = 0, control_caught = 0;
    char csv[256]; snprintf(csv, sizeof csv, "%s", pairs_csv);
    for (char* tok = strtok(csv, ","); tok; tok = strtok(NULL, ",")) {
        int A = -1, B = -1;
        if (sscanf(tok, "%d-%d", &A, &B) != 2 || A < 0 || B < 0 || A >= ndev || B >= ndev || A == B) {
            printf("skip bad pair '%s'\n", tok); continue;
        }
        plow_hsa_kernel kp, kc;
        if (plow_hsa_get_kernel(h, A, "coh_producer", &kp) ||
            plow_hsa_get_kernel(h, B, "coh_consumer", &kc)) {
            printf("get_kernel: %s\n", plow_hsa_last_error()); return 2; }

        /* resident peer buffers: data + g_done on A, g_ready on B. */
        uint32_t* data   = (uint32_t*)plow_hsa_alloc_peer(h, A, (size_t)NDATA * 4);
        uint32_t* g_done = (uint32_t*)plow_hsa_alloc_peer(h, A, 4);
        uint32_t* g_ready= (uint32_t*)plow_hsa_alloc_peer(h, B, 4);
        uint32_t* p_stat = (uint32_t*)plow_hsa_alloc(h, A, 4);
        uint32_t* c_stat = (uint32_t*)plow_hsa_alloc(h, B, 4);
        uint32_t* c_stale= (uint32_t*)plow_hsa_alloc(h, B, 4);
        uint32_t* c_fb   = (uint32_t*)plow_hsa_alloc(h, B, 4);
        if (!data || !g_done || !g_ready) { printf("alloc_peer: %s\n", plow_hsa_last_error()); return 2; }

        for (int v = 0; v < NV; v++) {
            uint32_t z = 0, ff = 0xffffffffu;
            /* clean slate every run (§6d discipline). */
            uint32_t* zbuf = (uint32_t*)calloc(NDATA, 4);
            plow_hsa_upload(h, A, data, zbuf, (size_t)NDATA * 4); free(zbuf);
            plow_hsa_upload(h, A, g_done, &z, 4);
            plow_hsa_upload(h, B, g_ready, &z, 4);
            plow_hsa_upload(h, A, p_stat, &z, 4);
            plow_hsa_upload(h, B, c_stat, &z, 4);
            plow_hsa_upload(h, B, c_stale, &z, 4);
            plow_hsa_upload(h, B, c_fb, &ff, 4);

            /* consumer first (spins on g_ready), then producer. */
            arg_cons ac = { data, NDATA, g_ready, g_done, iters, V[v].acq, deadline,
                            c_stale, c_fb, c_stat };
            plow_hsa_launch(h, B, &kc, 1, 1, 1, 1, 1, 1, 0, &ac, sizeof ac);
            arg_prod ap = { data, NDATA, g_ready, g_done, iters, V[v].rel, deadline, p_stat };
            plow_hsa_launch(h, A, &kp, 1, 1, 1, 1, 1, 1, 0, &ap, sizeof ap);
            plow_hsa_wait(h, A); plow_hsa_wait(h, B);

            uint32_t ps = 0, cs = 0, stale = 0, fb = 0xffffffffu;
            plow_hsa_download(h, A, &ps, p_stat, 4);
            plow_hsa_download(h, B, &cs, c_stat, 4);
            plow_hsa_download(h, B, &stale, c_stale, 4);
            plow_hsa_download(h, B, &fb, c_fb, 4);
            const char* stat = (ps == 0 && cs == 0) ? "ok" : "DEADLOCK";
            printf("%d-%-3d %-18s %14u %10s %12s",
                   A, B, V[v].name, stale, stat,
                   fb == 0xffffffffu ? "-" : "");
            if (fb != 0xffffffffu) printf("0x%08x", fb);
            printf("\n"); fflush(stdout);

            if (V[v].rel == SC_SYS && V[v].acq == SC_SYS && ps == 0 && cs == 0 && stale > 0)
                any_shipped_stale = 1;
            if (V[v].rel == SC_RELAXED && V[v].acq == SC_RELAXED && stale > 0)
                control_caught = 1;
        }
        printf("--------------------------------------------------------------------\n");
        plow_hsa_free(h, data); plow_hsa_free(h, g_done); plow_hsa_free(h, g_ready);
    }

    printf("\nVERDICT:\n");
    printf("  positive control detected staleness : %s\n",
           control_caught ? "YES (test has detection power)"
                          : "NO  (test INCONCLUSIVE — HW too coherent for this pattern)");
    printf("  SHIPPED sys/sys path stale          : %s\n",
           any_shipped_stale ? "YES -> cross-GPU acquire/release does NOT synchronize -> TP RACE"
                            : "no  -> system-scope acquire/release synchronizes on this HW");
    plow_hsa_shutdown(h);
    return 0;
}
