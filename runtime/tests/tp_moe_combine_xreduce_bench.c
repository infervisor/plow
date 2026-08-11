/* TP8 full-grid screen for the K3 MoeCombine -> XReduce packet boundary. */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TP_RANKS 8u
#define CTR_WORDS 32u
#define THREADS 512u

typedef uint16_t bf16;

typedef struct {
    void* out;
    const void* peer_scratch;
    const void* peer_gate;
    void* local_gate;
    const void* part;
    uint32_t nranks;
    uint32_t rank;
    uint32_t hidden;
    uint32_t topk;
    uint32_t slot_stride;
    uint32_t iters;
    uint64_t deadline;
    void* cycles;
    void* status;
} arg_cxr;

_Static_assert(sizeof(arg_cxr) == 88, "kernarg layout must match code-object metadata");

typedef struct {
    plow_hsa* h;
    int dev[TP_RANKS];
    plow_hsa_kernel control[TP_RANKS];
    plow_hsa_kernel fused[TP_RANKS];
    void* scratch[TP_RANKS];
    void* gate[TP_RANKS];
    void* out[TP_RANKS];
    void* scratch_table[TP_RANKS];
    void* gate_table[TP_RANKS];
    void* local_gate[TP_RANKS];
    void* part[TP_RANKS];
    void* cycles[TP_RANKS];
    void* status[TP_RANKS];
    uint32_t hidden;
    uint32_t topk;
    uint32_t nblk;
    uint32_t slot_stride;
    uint32_t iters;
    uint64_t deadline;
    double tick_ns;
    const void* zero_gate;
    size_t peer_gate_bytes;
    size_t local_gate_bytes;
} bench;

static float bf2f(bf16 b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static bf16 f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

static float part_value(uint32_t rank, uint32_t j, uint32_t h) {
    return (float)(rank + 1u) * (1.0f / 32.0f) +
           (float)(j + 1u) * (1.0f / 64.0f) +
           (float)(h & 7u) * (1.0f / 256.0f);
}

static bf16 oracle(uint32_t hidden_index, uint32_t topk) {
    float total = 0.0f;
    for (uint32_t r = 0; r < TP_RANKS; r++) {
        float local = 0.0f;
        for (uint32_t j = 0; j < topk; j++) local += part_value(r, j, hidden_index);
        total += bf2f(f2bf(local));
    }
    return f2bf(total);
}

static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f);
    *len = (size_t)n;
    return p;
}

static void check_upload(plow_hsa* h, int dev, void* dst, const void* src, size_t bytes) {
    if (plow_hsa_upload(h, dev, dst, src, bytes) != 0) {
        fprintf(stderr, "upload GPU%d: %s\n", dev, plow_hsa_last_error());
        exit(2);
    }
}

static int cmp_double(const void* a, const void* b) {
    const double x = *(const double*)a;
    const double y = *(const double*)b;
    return (x > y) - (x < y);
}

static double median(const double* in, uint32_t n) {
    double* v = (double*)malloc((size_t)n * sizeof *v);
    memcpy(v, in, (size_t)n * sizeof *v);
    qsort(v, n, sizeof *v, cmp_double);
    const double out = n & 1u ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) * 0.5;
    free(v);
    return out;
}

static double run_one(bench* b, int use_fused, bf16* snapshot, int establish) {
    for (uint32_t r = 0; r < TP_RANKS; r++) {
        check_upload(b->h, b->dev[r], b->gate[r], b->zero_gate, b->peer_gate_bytes);
        check_upload(b->h, b->dev[r], b->local_gate[r], b->zero_gate, b->local_gate_bytes);
        const uint64_t z64 = 0;
        const uint32_t z32 = 0;
        check_upload(b->h, b->dev[r], b->cycles[r], &z64, sizeof z64);
        check_upload(b->h, b->dev[r], b->status[r], &z32, sizeof z32);
    }

    arg_cxr args[TP_RANKS];
    for (uint32_t rr = TP_RANKS; rr-- > 0;) {
        args[rr] = (arg_cxr){
            b->out[rr], b->scratch_table[rr], b->gate_table[rr], b->local_gate[rr],
            b->part[rr], TP_RANKS, rr, b->hidden, b->topk, b->slot_stride, b->iters,
            b->deadline, b->cycles[rr], b->status[rr]};
        const plow_hsa_kernel* k = use_fused ? &b->fused[rr] : &b->control[rr];
        if (plow_hsa_launch(b->h, b->dev[rr], k, b->nblk * THREADS, 1, 1, THREADS, 1, 1,
                            0, &args[rr], sizeof args[rr]) != 0) {
            fprintf(stderr, "launch GPU%d: %s\n", b->dev[rr], plow_hsa_last_error());
            exit(2);
        }
    }
    for (uint32_t r = 0; r < TP_RANKS; r++)
        if (plow_hsa_wait(b->h, b->dev[r]) != 0) {
            fprintf(stderr, "wait GPU%d: %s\n", b->dev[r], plow_hsa_last_error());
            exit(2);
        }

    uint64_t max_ticks = 0;
    bf16* got = (bf16*)malloc((size_t)b->hidden * sizeof *got);
    for (uint32_t r = 0; r < TP_RANKS; r++) {
        uint32_t status = 0;
        uint64_t ticks = 0;
        plow_hsa_download(b->h, b->dev[r], &status, b->status[r], sizeof status);
        plow_hsa_download(b->h, b->dev[r], &ticks, b->cycles[r], sizeof ticks);
        if (status) {
            fprintf(stderr, "%s GPU%u protocol status 0x%08x\n",
                    use_fused ? "fused" : "control", r, status);
            exit(1);
        }
        if (ticks > max_ticks) max_ticks = ticks;
        plow_hsa_download(b->h, b->dev[r], got, b->out[r], (size_t)b->hidden * sizeof *got);
        for (uint32_t h = 0; h < b->hidden; h++) {
            const bf16 want = oracle(h, b->topk);
            if (got[h] != want) {
                fprintf(stderr, "%s GPU%u h=%u got=%g want=%g\n",
                        use_fused ? "fused" : "control", r, h, bf2f(got[h]), bf2f(want));
                exit(1);
            }
        }
        bf16* rank_snapshot = snapshot + (size_t)r * b->hidden;
        if (establish) memcpy(rank_snapshot, got, (size_t)b->hidden * sizeof *got);
        else if (memcmp(rank_snapshot, got, (size_t)b->hidden * sizeof *got) != 0) {
            fprintf(stderr, "%s GPU%u differs from control snapshot\n",
                    use_fused ? "fused" : "control", r);
            exit(1);
        }
    }
    free(got);
    return (double)max_ticks * b->tick_ns / (double)b->iters / 1e3;
}

int main(int argc, char** argv) {
    const char* elf_path = argc > 1 ? argv[1] : "/tmp/tp-moe-cxr.elf";
    const uint32_t samples = (uint32_t)(getenv("TP_SAMPLES") ? atoi(getenv("TP_SAMPLES")) : 12);
    bench b = {0};
    b.hidden = 3584;
    b.topk = 16;
    b.nblk = (b.hidden + THREADS - 1u) / THREADS;
    b.slot_stride = (b.hidden * sizeof(bf16) + 127u) & ~127u;
    b.iters = (uint32_t)(getenv("TP_ITERS") ? atoi(getenv("TP_ITERS")) : 2048);
    if (!samples || !b.iters || b.nblk != 7) {
        fprintf(stderr, "invalid samples/iters/shape\n");
        return 2;
    }
    for (uint32_t r = 0; r < TP_RANKS; r++) b.dev[r] = (int)r;

    b.h = plow_hsa_init();
    if (!b.h) {
        fprintf(stderr, "plow_hsa_init: %s\n", plow_hsa_last_error());
        return 2;
    }
    if (plow_hsa_device_count(b.h) < (int)TP_RANKS) {
        fprintf(stderr, "need 8 GPUs\n");
        return 2;
    }
    uint64_t freq = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    b.tick_ns = freq ? 1e9 / (double)freq : 1.0;
    b.deadline = freq ? freq : 1000000000ull;

    size_t elf_len = 0;
    void* elf = slurp(elf_path, &elf_len);
    for (uint32_t r = 0; r < TP_RANKS; r++) {
        if (plow_hsa_load_code_object(b.h, b.dev[r], elf, elf_len) != 0 ||
            plow_hsa_get_kernel(b.h, b.dev[r], "tp_moe_combine_xreduce_control",
                                &b.control[r]) != 0 ||
            plow_hsa_get_kernel(b.h, b.dev[r], "tp_moe_combine_xreduce_fused",
                                &b.fused[r]) != 0) {
            fprintf(stderr, "load/get GPU%u: %s\n", r, plow_hsa_last_error());
            return 2;
        }
    }

    b.peer_gate_bytes = (size_t)b.iters * CTR_WORDS * sizeof(uint32_t);
    b.local_gate_bytes = (size_t)(b.iters + 1u) * CTR_WORDS * sizeof(uint32_t);
    b.zero_gate = calloc(1, b.local_gate_bytes);
    const size_t scratch_bytes = (size_t)b.slot_stride * b.iters;
    void* scratch_ptrs[TP_RANKS];
    void* gate_ptrs[TP_RANKS];
    for (uint32_t r = 0; r < TP_RANKS; r++) {
        b.scratch[r] = plow_hsa_alloc_peer(b.h, b.dev[r], scratch_bytes);
        b.gate[r] = plow_hsa_alloc_peer(b.h, b.dev[r], b.peer_gate_bytes);
        b.out[r] = plow_hsa_alloc(b.h, b.dev[r], (size_t)b.hidden * sizeof(bf16));
        b.scratch_table[r] = plow_hsa_alloc(b.h, b.dev[r], TP_RANKS * sizeof(void*));
        b.gate_table[r] = plow_hsa_alloc(b.h, b.dev[r], TP_RANKS * sizeof(void*));
        b.local_gate[r] = plow_hsa_alloc(b.h, b.dev[r], b.local_gate_bytes);
        b.part[r] = plow_hsa_alloc(b.h, b.dev[r], (size_t)b.topk * b.hidden * sizeof(float));
        b.cycles[r] = plow_hsa_alloc(b.h, b.dev[r], sizeof(uint64_t));
        b.status[r] = plow_hsa_alloc(b.h, b.dev[r], sizeof(uint32_t));
        if (!b.scratch[r] || !b.gate[r] || !b.out[r] || !b.scratch_table[r] ||
            !b.gate_table[r] || !b.local_gate[r] || !b.part[r] || !b.cycles[r] ||
            !b.status[r]) {
            fprintf(stderr, "alloc GPU%u: %s\n", r, plow_hsa_last_error());
            return 2;
        }
        scratch_ptrs[r] = b.scratch[r];
        gate_ptrs[r] = b.gate[r];
    }
    float* host_part = (float*)malloc((size_t)b.topk * b.hidden * sizeof *host_part);
    for (uint32_t r = 0; r < TP_RANKS; r++) {
        check_upload(b.h, b.dev[r], b.scratch_table[r], scratch_ptrs, sizeof scratch_ptrs);
        check_upload(b.h, b.dev[r], b.gate_table[r], gate_ptrs, sizeof gate_ptrs);
        for (uint32_t j = 0; j < b.topk; j++)
            for (uint32_t h = 0; h < b.hidden; h++)
                host_part[(size_t)j * b.hidden + h] = part_value(r, j, h);
        check_upload(b.h, b.dev[r], b.part[r], host_part,
                     (size_t)b.topk * b.hidden * sizeof *host_part);
    }

    bf16* snapshot = (bf16*)malloc((size_t)TP_RANKS * b.hidden * sizeof *snapshot);
    double* control = (double*)malloc((size_t)samples * sizeof *control);
    double* fused = (double*)malloc((size_t)samples * sizeof *fused);
    printf("K3 combine+xreduce TP8: H=%u topk=%u blocks=%u iters=%u samples=%u\n",
           b.hidden, b.topk, b.nblk, b.iters, samples);
    for (uint32_t s = 0; s < samples; s++) {
        if ((s & 1u) == 0) {
            control[s] = run_one(&b, 0, snapshot, s == 0);
            fused[s] = run_one(&b, 1, snapshot, 0);
        } else {
            fused[s] = run_one(&b, 1, snapshot, 0);
            control[s] = run_one(&b, 0, snapshot, 0);
        }
        printf("  sample %02u control=%8.3f us fused=%8.3f us\n", s, control[s], fused[s]);
    }
    const double cmed = median(control, samples);
    const double fmed = median(fused, samples);
    const double save = cmed - fmed;
    const double token = save * 92.0 / 1000.0;
    printf("median control=%.3f us fused=%.3f us delta=%+.3f us projected92=%+.3f ms/token\n",
           cmed, fmed, -save, token);
    printf("{\"schema\":\"plow.k3-cxr.v1\",\"ranks\":8,\"hidden\":3584,"
           "\"topk\":16,\"blocks\":7,\"iters\":%u,\"samples\":%u,"
           "\"control_us\":%.6f,\"fused_us\":%.6f,\"projected_ms\":%.6f,"
           "\"bit_exact\":true}\n",
           b.iters, samples, cmed, fmed, token);
    plow_hsa_shutdown(b.h);
    return 0;
}
