/* ubench_cu_roofline.c — Plow μBench: single-CU/SM roofline driver for MI350X.
 *
 * Exercises the EXACT production kernels from the Gemma 4 network on a single CU,
 * measures performance via wall-clock timing (and optionally rocprof counters),
 * and reports a roofline analysis showing efficiency vs theoretical ceilings.
 *
 * The benchmark kernels are compiled from bench_cu_gfx950.hip which #includes the
 * production op headers (op_gemm.h, op_norm.h, op_elementwise.h, op_attention.h)
 * directly — same code, same instruction sequence, same register pressure.
 *
 * Usage:
 *   ./ubench_cu_roofline [--kernel name] [--iters N] [--timing-only]
 *                        [--csv pass1.csv,pass2.csv,pass3.csv] [--json out.json]
 *                        [--decode]  (use M=1 decode sizes instead of M=128 prefill)
 */
#define _POSIX_C_SOURCE 199309L

#include "bench_cu.h"
#include "../amd/hsa_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * MI350X Single-CU Theoretical Ceilings
 * From crates/hwspec/src/amd/mi350.rs (measured on hardware)
 * ============================================================================ */
const BenchSpec BENCH_SPEC_MI350X = {
    .name = "MI350X",
    .arch = "gfx950",
    .cu_count = 256,
    .warp_width = 64,
    .max_threads = 2048,
    .shared_mem = 163840,       /* 160 KiB LDS per CU */
    .regs_32bit = 131072,       /* 512 KiB VGPR / CU */
    .clock_ghz = 2.2,
    /* 4 MFMA cores × 512 bf16 MACs/core/cycle × 2 FLOPs/MAC × 2.2 GHz / 1e3 */
    .peak_tflops_bf16 = 4.0 * 512.0 * 2.0 * 2.2 / 1000.0,  /* 9.01 TFLOPS */
    .peak_tflops_fp32 = 4.0 * 256.0 * 2.0 * 2.2 / 1000.0,  /* 4.50 TFLOPS */
    .hbm_bw_gbps = 8000.0 / 256.0,   /* 31.25 GB/s per CU */
    .l2_bw_gbps = 375.0 / 32.0,      /* 11.72 GB/s per CU (within XCD) */
    .ridge_flops_per_byte = 0.0,      /* computed from above */
};

/* ============================================================================
 * Kernel registry — these map to the actual Gemma 4 network ops
 * ============================================================================ */
typedef enum {
    BENCH_GEMM = 0,          /* d_gemm: prefill matmul, 128x128 MFMA tile */
    BENCH_GEMM_NORM,         /* d_gemm_norm: fused RMSNorm + matmul (q/k/v/gate/up) */
    BENCH_GEMV,              /* d_gemv: decode matmul, BW-bound dot product */
    BENCH_GEMV_NORM,         /* d_gemv + fused norm: decode q/k/v projection */
    BENCH_RMSNORM,           /* d_rmsnorm: standalone hidden-dim norm */
    BENCH_ROWRMS,            /* d_rowrms: RMS scalars for fused-norm GEMM */
    BENCH_HEADNORM_ROPE,     /* d_headnorm_rope: per-head norm + RoPE */
    BENCH_RESIDUAL,          /* d_residual: (a+b)*scale */
    BENCH_GLU,               /* d_glu: GeGLU or SwiGLU */
    BENCH_SOFTCAP,           /* d_softcap: cap*tanh(x/cap) */
    BENCH_EMBED,             /* d_embed: gather + scale */
    BENCH_INTERP_OVERHEAD,   /* interpreter gate/fence/signal machinery */
    BENCH__COUNT
} BenchKernel;

static const char* const KERNEL_NAMES[] = {
    "d_gemm", "d_gemm_norm", "d_gemv", "d_gemv_norm",
    "d_rmsnorm", "d_rowrms", "d_headnorm_rope", "d_residual",
    "d_glu", "d_softcap", "d_embed", "interp_overhead"
};

static const char* const KERNEL_SYMBOLS[] = {
    "bench_gemm", "bench_gemm_norm", "bench_gemv", "bench_gemv_norm",
    "bench_rmsnorm", "bench_rowrms", "bench_headnorm_rope", "bench_residual",
    "bench_glu", "bench_softcap", "bench_embed", "bench_interp_overhead"
};

/* Which Gemma 4 fused op uses each kernel (for the table annotation) */
static const char* const KERNEL_ROLE[] = {
    "F1 prefill (o/down/lm_head)",
    "F1 prefill (q/k/v/gate/up + norm)",
    "decode (o/down/lm_head)",
    "decode (q/k/v/gate/up + norm)",
    "S4 final_norm / standalone",
    "norm prologue (rms scalars)",
    "F2/F3 q_norm+rope / k_norm+rope / v_norm",
    "S3 residual + layer_scalar",
    "F5 GeGLU (gelu_tanh(gate)*up)",
    "logit softcap (cap=30)",
    "F4 embed(ids) * sqrt(H)",
    "interpreter spin/fence/signal"
};

/* ============================================================================
 * Gemma 4 31B problem sizes (from the design notes)
 * ============================================================================ */
/* Hidden=5376, Intermediate=21504, Heads=32, KVHeads=16(sliding)/4(global),
 * HeadDim=256(sliding)/512(global), Vocab=262144 */
#define GEMMA4_HIDDEN   5376
#define GEMMA4_INTER    21504
#define GEMMA4_HEADS    32
#define GEMMA4_HD_SLID  256
#define GEMMA4_HD_GLOB  512
#define GEMMA4_VOCAB    262144
#define GEMMA4_EPS      1e-6f
#define GEMMA4_SCALE    73.5f   /* bf16(sqrt(5376)) */
#define GEMMA4_SOFTCAP  30.0f
#define GEMMA4_LSCALAR  1.37f   /* approx layer_scalar */

typedef struct {
    BenchKernel kernel;
    unsigned    M, N, K;       /* GEMM dims */
    unsigned    rows, feat;    /* norm dims */
    unsigned    ntok, nhead, hd; /* headnorm dims */
    unsigned    n;             /* elementwise count */
    unsigned    hidden, vocab; /* embed */
    float       eps, scale;
    unsigned    act;           /* 0=gelu_tanh, 1=silu */
    unsigned    iters;         /* benchmark iterations */
} KernelConfig;

static KernelConfig gemma4_prefill_configs[] = {
    /* Prefill M=128 (a representative single-CU tile from one workgroup) */
    {BENCH_GEMM,          128, GEMMA4_INTER, GEMMA4_HIDDEN, 0,0, 0,0,0, 0, 0,0, 0,0, 0, 20},
    {BENCH_GEMM_NORM,     128, GEMMA4_HIDDEN, GEMMA4_HIDDEN, 0,0, 0,0,0, 0, 0,0, GEMMA4_EPS,0, 0, 20},
    {BENCH_RMSNORM,       0,0,0, 128, GEMMA4_HIDDEN, 0,0,0, 0, 0,0, GEMMA4_EPS,0, 0, 200},
    {BENCH_ROWRMS,        0,0,0, 128, GEMMA4_HIDDEN, 0,0,0, 0, 0,0, GEMMA4_EPS,0, 0, 200},
    {BENCH_HEADNORM_ROPE, 0,0,0, 0,0, 32,GEMMA4_HEADS,GEMMA4_HD_SLID, 0, 0,0, GEMMA4_EPS,0, 0, 200},
    {BENCH_HEADNORM_ROPE, 0,0,0, 0,0, 32,GEMMA4_HEADS,GEMMA4_HD_GLOB, 0, 0,0, GEMMA4_EPS,0, 0, 200},
    {BENCH_RESIDUAL,      0,0,0, 0,0, 0,0,0, 128*GEMMA4_HIDDEN, 0,0, 0,GEMMA4_LSCALAR, 0, 200},
    {BENCH_GLU,           0,0,0, 0,0, 0,0,0, 128*GEMMA4_INTER, 0,0, 0,0, 0, 50},
    {BENCH_SOFTCAP,       0,0,0, 0,0, 0,0,0, 128*GEMMA4_HIDDEN, 0,0, 0,GEMMA4_SOFTCAP, 0, 200},
    {BENCH_EMBED,         0,0,0, 0,0, 0,0,0, 0, GEMMA4_HIDDEN,4096, 0,GEMMA4_SCALE, 0, 100},
    {BENCH_INTERP_OVERHEAD, 0,0,0, 0,0, 0,0,0, 0, 0,0, 0,0, 0, 10000},
};

static KernelConfig gemma4_decode_configs[] = {
    /* Decode M=1 (the latency-critical serving path) */
    {BENCH_GEMV,          1, GEMMA4_INTER, GEMMA4_HIDDEN, 0,0, 0,0,0, 0, 0,0, 0,0, 0, 100},
    {BENCH_GEMV_NORM,     1, GEMMA4_HIDDEN, GEMMA4_HIDDEN, 0,0, 0,0,0, 0, 0,0, GEMMA4_EPS,0, 0, 100},
    {BENCH_RMSNORM,       0,0,0, 1, GEMMA4_HIDDEN, 0,0,0, 0, 0,0, GEMMA4_EPS,0, 0, 1000},
    {BENCH_HEADNORM_ROPE, 0,0,0, 0,0, 1,GEMMA4_HEADS,GEMMA4_HD_SLID, 0, 0,0, GEMMA4_EPS,0, 0, 1000},
    {BENCH_RESIDUAL,      0,0,0, 0,0, 0,0,0, GEMMA4_HIDDEN, 0,0, 0,GEMMA4_LSCALAR, 0, 1000},
    {BENCH_GLU,           0,0,0, 0,0, 0,0,0, GEMMA4_INTER, 0,0, 0,0, 0, 500},
    {BENCH_INTERP_OVERHEAD, 0,0,0, 0,0, 0,0,0, 0, 0,0, 0,0, 0, 10000},
};

#define N_PREFILL_CONFIGS (sizeof(gemma4_prefill_configs)/sizeof(gemma4_prefill_configs[0]))
#define N_DECODE_CONFIGS  (sizeof(gemma4_decode_configs)/sizeof(gemma4_decode_configs[0]))

/* ============================================================================
 * FLOP/byte counting for each kernel
 * ============================================================================ */
static double kernel_flops(const KernelConfig* c) {
    switch (c->kernel) {
    case BENCH_GEMM:
    case BENCH_GEMM_NORM:
        return 2.0 * (double)c->M * c->N * c->K;
    case BENCH_GEMV:
    case BENCH_GEMV_NORM:
        return 2.0 * (double)c->M * c->N * c->K;
    case BENCH_RMSNORM:
        /* sum(x^2): 2*feat, rsqrt: 1, x*inv*g: 3*feat => ~5*feat per row */
        return 5.0 * (double)c->rows * c->feat;
    case BENCH_ROWRMS:
        /* sum(x^2): 2*feat, rsqrt: 1 => ~2*feat per row */
        return 2.0 * (double)c->rows * c->feat;
    case BENCH_HEADNORM_ROPE:
        /* norm: 3*hd, rope: 6*(hd/2) => 6*hd per (tok,head) */
        return 6.0 * (double)c->ntok * c->nhead * c->hd;
    case BENCH_RESIDUAL:
        return 2.0 * (double)c->n;  /* add + mul */
    case BENCH_GLU:
        return 11.0 * (double)c->n; /* gelu_tanh ≈ 10 + 1 mul */
    case BENCH_SOFTCAP:
        return 10.0 * (double)c->n; /* div + tanh(~8) + mul */
    case BENCH_EMBED:
        return (double)c->ntok * c->hidden; /* 1 mul per element (just scale) */
    default: return 0.0;
    }
}

static double kernel_bytes(const KernelConfig* c) {
    switch (c->kernel) {
    case BENCH_GEMM:
        /* A[M,K] + B[N,K] + C[M,N], bf16 */
        return 2.0 * ((double)c->M * c->K + (double)c->N * c->K + (double)c->M * c->N);
    case BENCH_GEMM_NORM:
        /* A[M,K] + B[N,K] + C[M,N] + rms[M] (f32) + gamma[K] (bf16) */
        return 2.0 * ((double)c->M * c->K + (double)c->N * c->K + (double)c->M * c->N)
               + 4.0 * c->M + 2.0 * c->K;
    case BENCH_GEMV:
        /* W[N,K] + x[M,K] + C[M,N], bf16. W dominates. */
        return 2.0 * ((double)c->N * c->K + (double)c->M * c->K + (double)c->M * c->N);
    case BENCH_GEMV_NORM:
        return 2.0 * ((double)c->N * c->K + (double)c->M * c->K + (double)c->M * c->N)
               + 4.0 * c->M + 2.0 * c->K;
    case BENCH_RMSNORM:
        /* x[rows,feat] read + gamma[feat] read + out[rows,feat] write */
        return 2.0 * (2.0 * (double)c->rows * c->feat + c->feat);
    case BENCH_ROWRMS:
        /* x[rows,feat] read + rms[rows] write (f32) */
        return 2.0 * (double)c->rows * c->feat + 4.0 * c->rows;
    case BENCH_HEADNORM_ROPE:
        /* x + gamma + cos/sin(f32) + pos(i32) + out */
        return 2.0 * (double)c->ntok * c->nhead * c->hd * 2
               + 2.0 * c->hd
               + 4.0 * (double)c->ntok * (c->hd / 2) * 2
               + 4.0 * c->ntok;
    case BENCH_RESIDUAL:
        return 2.0 * 3.0 * (double)c->n; /* a + b read, out write */
    case BENCH_GLU:
        return 2.0 * 3.0 * (double)c->n; /* gate + up read, out write */
    case BENCH_SOFTCAP:
        return 2.0 * 2.0 * (double)c->n; /* x read, out write */
    case BENCH_EMBED:
        return 2.0 * 2.0 * (double)c->ntok * c->hidden; /* table read + out write */
    default: return 0.0;
    }
}

/* ============================================================================
 * Roofline analysis
 * ============================================================================ */
typedef struct {
    const char* name;
    const char* role;
    double      achieved_tflops;
    double      peak_tflops;
    double      compute_eff_pct;
    double      achieved_bw_gbps;
    double      peak_bw_gbps;
    double      memory_eff_pct;
    double      arith_intensity;
    double      ridge_point;
    int         is_compute_bound;
    double      elapsed_ms;
    /* From rocprof (if available) */
    double      mfma_util_pct;
    double      lds_conflict_pct;
    double      l2_hit_rate_pct;
} Result;

/* ============================================================================
 * CSV parsing (rocprof output)
 * ============================================================================ */
static int parse_csv_field(const char* path, const char* field, uint64_t* out) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char line[4096];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }

    /* Find column index */
    int col = -1, idx = 0;
    char* tok = strtok(line, ",\n\r");
    while (tok) {
        if (strcmp(tok, field) == 0) { col = idx; break; }
        idx++;
        tok = strtok(NULL, ",\n\r");
    }
    if (col < 0) { fclose(f); return -1; }

    /* Sum data rows */
    *out = 0;
    while (fgets(line, sizeof(line), f)) {
        char* cols[128] = {0};
        int nc = 0;
        tok = strtok(line, ",\n\r");
        while (tok && nc < 128) { cols[nc++] = tok; tok = strtok(NULL, ",\n\r"); }
        if (col < nc) *out += (uint64_t)strtoull(cols[col], NULL, 10);
    }
    fclose(f);
    return 0;
}

/* ============================================================================
 * Terminal output
 * ============================================================================ */
static void print_header(const BenchSpec* spec, int decode) {
    double ridge = spec->peak_tflops_bf16 * 1e3 / spec->hbm_bw_gbps;
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Plow μBench — Single CU Roofline — %s (%s) @ %.2f GHz                           ║\n",
           spec->name, spec->arch, spec->clock_ghz);
    printf("║  Peak: %.2f bf16 TFLOPS | %.1f GB/s HBM | Ridge: %.0f F/B | Mode: %-7s               ║\n",
           spec->peak_tflops_bf16, spec->hbm_bw_gbps, ridge, decode ? "DECODE" : "PREFILL");
    printf("╠═══════════════════════════╤══════════╤═════════╤═════════╤═════════╤══════╤═════════════════╣\n");
    printf("║ Kernel                    │ Time(ms) │ TFLOPS  │ Comp%%   │ GB/s    │ Mem%%  │ Bound / Role    ║\n");
    printf("╟───────────────────────────┼──────────┼─────────┼─────────┼─────────┼──────┼─────────────────╢\n");
}

static void print_row(const Result* r) {
    const char* bound = r->is_compute_bound ? "COMP" : "MEM ";
    printf("║ %-25s │ %7.3f  │ %6.3f  │ %5.1f%%  │ %6.1f  │%5.1f%%│ %s              ║\n",
           r->name, r->elapsed_ms, r->achieved_tflops, r->compute_eff_pct,
           r->achieved_bw_gbps, r->memory_eff_pct, bound);
}

static void print_footer(const Result* results, int n) {
    printf("╚═══════════════════════════╧══════════╧═════════╧═════════╧═════════╧══════╧═════════════════╝\n");
    printf("\nBottleneck Analysis (Gemma 4 31B):\n");
    for (int i = 0; i < n; i++) {
        const Result* r = &results[i];
        if (r->elapsed_ms <= 0.0) continue;
        if (r->is_compute_bound) {
            printf("  ⚡ %-22s COMPUTE (AI=%.0f): %.1f%% MFMA. %s\n",
                   r->name, r->arith_intensity, r->compute_eff_pct, r->role);
        } else {
            printf("  💾 %-22s MEMORY  (AI=%.1f): %.1f%% HBM. %s\n",
                   r->name, r->arith_intensity, r->memory_eff_pct, r->role);
        }
    }
    printf("\n");
}

/* ============================================================================
 * HSA launch helpers
 * ============================================================================ */
typedef unsigned short bf16_h;
static bf16_h f2bf_h(float f) {
    unsigned u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16_h)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16_h)(u >> 16);
}
static float frandf(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

static double timed_launch(plow_hsa* h, int dev, plow_hsa_kernel* kern,
                           void* args, size_t args_sz) {
    /* Single CU: grid=256 threads (1 workgroup), block=256 */
    const unsigned GRID = 256, BLOCK = 256;

    /* Warmup */
    plow_hsa_launch(h, dev, kern, GRID, 1, 1, BLOCK, 1, 1, 0, args, args_sz);
    plow_hsa_wait(h, dev);

    /* Timed */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    plow_hsa_launch(h, dev, kern, GRID, 1, 1, BLOCK, 1, 1, 0, args, args_sz);
    plow_hsa_wait(h, dev);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    return ((double)(t1.tv_sec - t0.tv_sec) * 1e6 +
            (double)(t1.tv_nsec - t0.tv_nsec) / 1e3); /* microseconds */
}

/* ============================================================================
 * Main
 * ============================================================================ */
int main(int argc, char** argv) {
    const char* kernel_filter = NULL;
    const char* json_out = NULL;
    int timing_only = 0;
    int decode_mode = 0;
    int custom_iters = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kernel") == 0 && i+1 < argc) kernel_filter = argv[++i];
        else if (strcmp(argv[i], "--json") == 0 && i+1 < argc) json_out = argv[++i];
        else if (strcmp(argv[i], "--iters") == 0 && i+1 < argc) custom_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--timing-only") == 0) timing_only = 1;
        else if (strcmp(argv[i], "--decode") == 0) decode_mode = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--kernel name] [--iters N] [--timing-only] [--decode] [--json out.json]\n"
                   "\nKernels: d_gemm d_gemm_norm d_gemv d_gemv_norm d_rmsnorm d_rowrms\n"
                   "         d_headnorm_rope d_residual d_glu d_softcap d_embed interp_overhead\n", argv[0]);
            return 0;
        }
    }

    /* HSA init */
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "plow_hsa_init: %s\n", plow_hsa_last_error()); return 1; }

    char nm[64]; uint32_t cus = 0, lds_sz = 0;
    plow_hsa_device_info(h, 0, nm, &cus, &lds_sz);
    printf("Device: %s | CUs: %u | LDS/CU: %u B\n", nm, cus, lds_sz);

    const BenchSpec* spec = &BENCH_SPEC_MI350X;
    printf("Spec: %s — %.2f bf16 TFLOPS/CU, %.1f GB/s HBM/CU\n\n",
           spec->name, spec->peak_tflops_bf16, spec->hbm_bw_gbps);

    /* Load code object */
    const char* elf = "ubench_cu_gfx950.elf";
    FILE* ef = fopen(elf, "rb");
    if (!ef) { elf = "bench_cu_gfx950.elf"; ef = fopen(elf, "rb"); }
    if (!ef) { fprintf(stderr, "Cannot open ELF (tried ubench_cu_gfx950.elf, bench_cu_gfx950.elf)\n"); return 1; }
    fseek(ef, 0, SEEK_END); long co_n = ftell(ef); fseek(ef, 0, SEEK_SET);
    void* co = malloc(co_n);
    if (fread(co, 1, co_n, ef) != (size_t)co_n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(ef);
    if (plow_hsa_load_code_object(h, 0, co, co_n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }

    /* Resolve all kernel symbols */
    plow_hsa_kernel kerns[BENCH__COUNT];
    for (int i = 0; i < BENCH__COUNT; i++) {
        if (plow_hsa_get_kernel(h, 0, KERNEL_SYMBOLS[i], &kerns[i]) != 0) {
            fprintf(stderr, "⚠ symbol '%s' not found (skipping)\n", KERNEL_SYMBOLS[i]);
            memset(&kerns[i], 0, sizeof(kerns[i]));
        }
    }

    /* Allocate device memory.
     * The GEMM weight matrix B[N,K] is the largest: 21504 × 5376 × 2B = ~220 MB.
     * The activation buffer A[M,K] and output C[M,N] are much smaller. */
    srand(42);
    const size_t WEIGHT_SZ = (size_t)GEMMA4_INTER * GEMMA4_HIDDEN * 2; /* B[N,K] bf16 = 220 MB */
    const size_t ACT_SZ    = (size_t)128 * GEMMA4_INTER * 2;           /* largest act = 5.5 MB */
    void* d0 = plow_hsa_alloc(h, 0, ACT_SZ);        /* activation / x */
    void* d1 = plow_hsa_alloc(h, 0, WEIGHT_SZ);     /* weight / B */
    void* d2 = plow_hsa_alloc(h, 0, ACT_SZ);        /* gamma / aux */
    void* d_out = plow_hsa_alloc(h, 0, ACT_SZ);     /* output C */
    void* d_rms = plow_hsa_alloc(h, 0, 128 * 4);    /* f32 rms scalars */

    /* Embed table (small for bench) */
    const unsigned BVOCAB = 4096;
    void* d_embed_tbl = plow_hsa_alloc(h, 0, (size_t)BVOCAB * GEMMA4_HIDDEN * 2);
    void* d_embed_ids = plow_hsa_alloc(h, 0, 128 * 4);

    /* RoPE */
    const unsigned MAXP = 256, HALFHD = GEMMA4_HD_GLOB / 2;
    void* d_cos = plow_hsa_alloc(h, 0, (size_t)MAXP * HALFHD * 4);
    void* d_sin = plow_hsa_alloc(h, 0, (size_t)MAXP * HALFHD * 4);
    void* d_pos = plow_hsa_alloc(h, 0, 128 * 4);

    /* Fill buffers — use plow_hsa_upload (not copy_h2d) since we use malloc'd
     * staging. copy_h2d requires plow_hsa_alloc_host memory; upload handles
     * arbitrary host pointers (pins the pages internally). */
    {
        bf16_h* hbuf = (bf16_h*)malloc(WEIGHT_SZ);
        for (size_t i = 0; i < WEIGHT_SZ / 2; i++) hbuf[i] = f2bf_h(frandf() * 0.1f);
        plow_hsa_upload(h, 0, d0, hbuf, ACT_SZ);
        plow_hsa_upload(h, 0, d1, hbuf, WEIGHT_SZ);
        plow_hsa_upload(h, 0, d2, hbuf, ACT_SZ);
        plow_hsa_upload(h, 0, d_embed_tbl, hbuf, (size_t)BVOCAB * GEMMA4_HIDDEN * 2);

        /* rms scalars (f32, ~1.0) */
        float* hrms = (float*)malloc(128 * 4);
        for (int i = 0; i < 128; i++) hrms[i] = 1.0f / (1.0f + 0.01f * frandf());
        plow_hsa_upload(h, 0, d_rms, hrms, 128 * 4);
        free(hrms);

        /* embed ids */
        int* hids = (int*)malloc(128 * 4);
        for (int i = 0; i < 128; i++) hids[i] = rand() % BVOCAB;
        plow_hsa_upload(h, 0, d_embed_ids, hids, 128 * 4);
        free(hids);

        /* cos/sin */
        float* hc = (float*)malloc((size_t)MAXP * HALFHD * 4);
        float* hs = (float*)malloc((size_t)MAXP * HALFHD * 4);
        for (unsigned p = 0; p < MAXP; p++)
            for (unsigned j = 0; j < HALFHD; j++) {
                double ang = (double)p * pow(1e6, -2.0 * j / (double)GEMMA4_HD_GLOB);
                hc[p * HALFHD + j] = (float)cos(ang);
                hs[p * HALFHD + j] = (float)sin(ang);
            }
        plow_hsa_upload(h, 0, d_cos, hc, (size_t)MAXP * HALFHD * 4);
        plow_hsa_upload(h, 0, d_sin, hs, (size_t)MAXP * HALFHD * 4);
        free(hc); free(hs);

        int* hpos = (int*)malloc(128 * 4);
        for (int i = 0; i < 128; i++) hpos[i] = i;
        plow_hsa_upload(h, 0, d_pos, hpos, 128 * 4);
        free(hpos);
        free(hbuf);
    }

    /* Select config set */
    KernelConfig* configs = decode_mode ? gemma4_decode_configs : gemma4_prefill_configs;
    int n_configs = decode_mode ? (int)N_DECODE_CONFIGS : (int)N_PREFILL_CONFIGS;

    /* Run benchmarks */
    Result results[32];
    int n_results = 0;
    double ridge = spec->peak_tflops_bf16 * 1e3 / spec->hbm_bw_gbps;

    print_header(spec, decode_mode);

    for (int ci = 0; ci < n_configs; ci++) {
        KernelConfig* c = &configs[ci];
        if (custom_iters > 0) c->iters = custom_iters;
        if (kernel_filter && strcmp(kernel_filter, KERNEL_NAMES[c->kernel]) != 0)
            continue;

        double elapsed_us = 0.0;

        /* Build args and launch based on kernel type */
        switch (c->kernel) {
        case BENCH_GEMM: {
            struct __attribute__((packed)) {
                void* C; const void* A; const void* B; unsigned M, N, K, iters;
            } a = {d_out, d0, d1, c->M, c->N, c->K, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_GEMM], &a, sizeof(a));
            break;
        }
        case BENCH_GEMM_NORM: {
            struct __attribute__((packed)) {
                void* C; const void* A; const void* B; const void* rms; const void* gamma;
                unsigned M, N, K, iters;
            } a = {d_out, d0, d1, d_rms, d2, c->M, c->N, c->K, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_GEMM_NORM], &a, sizeof(a));
            break;
        }
        case BENCH_GEMV: {
            struct __attribute__((packed)) {
                void* C; const void* x; const void* W; unsigned M, N, K, iters;
            } a = {d_out, d0, d1, c->M, c->N, c->K, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_GEMV], &a, sizeof(a));
            break;
        }
        case BENCH_GEMV_NORM: {
            struct __attribute__((packed)) {
                void* C; const void* x; const void* W; const void* rms; const void* gamma;
                unsigned M, N, K, iters;
            } a = {d_out, d0, d1, d_rms, d2, c->M, c->N, c->K, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_GEMV_NORM], &a, sizeof(a));
            break;
        }
        case BENCH_RMSNORM: {
            struct __attribute__((packed)) {
                void* out; const void* x; const void* gamma;
                unsigned rows, feat; float eps; unsigned iters;
            } a = {d_out, d0, d1, c->rows, c->feat, c->eps, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_RMSNORM], &a, sizeof(a));
            break;
        }
        case BENCH_ROWRMS: {
            struct __attribute__((packed)) {
                void* rms; const void* x; unsigned rows, feat; float eps; unsigned iters;
            } a = {d_rms, d0, c->rows, c->feat, c->eps, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_ROWRMS], &a, sizeof(a));
            break;
        }
        case BENCH_HEADNORM_ROPE: {
            unsigned ntok = c->ntok ? c->ntok : 32;
            struct __attribute__((packed)) {
                void* out; const void* x; const void* gamma;
                const void* cosb; const void* sinb; const void* pos;
                unsigned ntok, nhead, hd; float eps; unsigned iters;
            } a = {d_out, d0, d1, d_cos, d_sin, d_pos,
                   ntok, c->nhead, c->hd, c->eps, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_HEADNORM_ROPE], &a, sizeof(a));
            break;
        }
        case BENCH_RESIDUAL: {
            struct __attribute__((packed)) {
                void* out; const void* a; const void* b;
                unsigned n; float scale; unsigned iters;
            } a = {d_out, d0, d1, c->n, c->scale, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_RESIDUAL], &a, sizeof(a));
            break;
        }
        case BENCH_GLU: {
            struct __attribute__((packed)) {
                void* out; const void* gate; const void* up;
                unsigned n; unsigned act; unsigned iters;
            } a = {d_out, d0, d1, c->n, c->act, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_GLU], &a, sizeof(a));
            break;
        }
        case BENCH_SOFTCAP: {
            struct __attribute__((packed)) {
                void* out; const void* x; unsigned n; float cap; unsigned iters;
            } a = {d_out, d0, c->n, c->scale, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_SOFTCAP], &a, sizeof(a));
            break;
        }
        case BENCH_EMBED: {
            unsigned ntok = c->ntok ? c->ntok : 128;
            struct __attribute__((packed)) {
                void* out; const void* table; const void* ids;
                unsigned ntok, hidden; float scale; unsigned iters;
            } a = {d_out, d_embed_tbl, d_embed_ids, ntok, c->hidden, c->scale, c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_EMBED], &a, sizeof(a));
            break;
        }
        case BENCH_INTERP_OVERHEAD: {
            struct __attribute__((packed)) { unsigned iters; } a = {c->iters};
            elapsed_us = timed_launch(h, 0, &kerns[BENCH_INTERP_OVERHEAD], &a, sizeof(a));
            break;
        }
        default: continue;
        }

        /* Compute roofline */
        double flops = kernel_flops(c) * c->iters;
        double bytes = kernel_bytes(c) * c->iters;
        double elapsed_s = elapsed_us * 1e-6;

        Result r = {0};
        r.name = KERNEL_NAMES[c->kernel];
        r.role = KERNEL_ROLE[c->kernel];
        r.elapsed_ms = elapsed_us / 1e3;
        r.peak_tflops = spec->peak_tflops_bf16;
        r.peak_bw_gbps = spec->hbm_bw_gbps;
        r.ridge_point = ridge;

        if (elapsed_s > 0.0) {
            r.achieved_tflops = flops / elapsed_s / 1e12;
            r.achieved_bw_gbps = bytes / elapsed_s / 1e9;
        }
        r.compute_eff_pct = 100.0 * r.achieved_tflops / r.peak_tflops;
        r.memory_eff_pct = 100.0 * r.achieved_bw_gbps / r.peak_bw_gbps;
        r.arith_intensity = (bytes > 0) ? flops / bytes : 0.0;
        r.is_compute_bound = (r.arith_intensity > ridge) ? 1 : 0;

        print_row(&r);
        results[n_results++] = r;
    }

    print_footer(results, n_results);

    /* JSON output */
    if (json_out) {
        FILE* jf = fopen(json_out, "w");
        if (jf) {
            fprintf(jf, "{\n  \"device\": \"%s\", \"arch\": \"%s\", \"mode\": \"%s\",\n",
                    spec->name, spec->arch, decode_mode ? "decode" : "prefill");
            fprintf(jf, "  \"peak_tflops_bf16_per_cu\": %.4f, \"hbm_gbps_per_cu\": %.2f,\n",
                    spec->peak_tflops_bf16, spec->hbm_bw_gbps);
            fprintf(jf, "  \"ridge_flops_per_byte\": %.1f,\n  \"results\": [\n", ridge);
            for (int i = 0; i < n_results; i++) {
                Result* r = &results[i];
                fprintf(jf, "    {\"kernel\": \"%s\", \"role\": \"%s\", "
                        "\"elapsed_ms\": %.4f, \"achieved_tflops\": %.6f, "
                        "\"compute_eff\": %.2f, \"achieved_gbps\": %.2f, "
                        "\"memory_eff\": %.2f, \"arith_intensity\": %.2f, "
                        "\"bound\": \"%s\"}%s\n",
                        r->name, r->role, r->elapsed_ms, r->achieved_tflops,
                        r->compute_eff_pct, r->achieved_bw_gbps, r->memory_eff_pct,
                        r->arith_intensity, r->is_compute_bound ? "compute" : "memory",
                        (i < n_results - 1) ? "," : "");
            }
            fprintf(jf, "  ]\n}\n");
            fclose(jf);
            printf("JSON written to %s\n", json_out);
        }
    }

    plow_hsa_shutdown(h);
    free(co);
    return 0;
}
