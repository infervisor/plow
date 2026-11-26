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

#ifdef PLOW_CUDA
/* ============================================================================
 * NVIDIA sm_12x path (GB10 = sm_121a): cuda_backend.c implements the same plow_hsa_* API.
 *
 *   --ceilings            single-SM ceilings at gridDim=1, cross-checked at gridDim=n_sm
 *   --fp4-tile <cubin>    NVFP4 MMA tile loops (fp4_tile_sm120.cu) at gridDim=1: tensor pipe,
 *                         resident smem, and the cp.async / bulk / TMA 2-D ring feeds
 *   --pkt <model.pkt> --cubin <ops cubin> --set <case set>
 *                         the packet's own instructions on ONE SM: per-slice time (L2 flushed
 *                         before every launch, every slice of the instruction's blocks), the
 *                         whole op serially on one SM, and a full-grid vs one-SM output check
 *
 * Bytes/FLOPs and the roofline columns come from scripts/bench/op_roof.py --ubench (op_roof's
 * per-instruction formulas); this driver measures.
 * ============================================================================ */
#include "dev_blob.h"
#include <cuda.h>
#include <signal.h>
#include <unistd.h>

/* GB10 single-SM ceilings, measured by `--ceilings` at gridDim=1 (crates/hwspec/src/peak.rs
 * GB10_SM_CEILINGS is the record; keep the two equal). */
const BenchSpec BENCH_SPEC_GB10 = {
    .name = "GB10",
    .arch = "sm_121a",
    .cu_count = 48,
    .warp_width = 32,
    .max_threads = 1536,
    .shared_mem = 102400,
    .regs_32bit = 65536,
    .clock_ghz = 2.579,
    .peak_tflops_bf16 = 2.640,  /* mma.sync m16n8k16: 1024 FLOP/clk */
    .peak_tflops_fp32 = 0.645,
    .hbm_bw_gbps = 78.1,        /* one SM alone; the whole GPU's 238 GB/s is 4.9 per SM */
    .l2_bw_gbps = 82.6,
    .ridge_flops_per_byte = 2640.0 / 78.1,
    .peak_tflops_fp8 = 5.283,   /* e4m3 m16n8k32, also kind::f8f6f4 and MXFP8 */
    .peak_tflops_fp4 = 10.556,  /* NVFP4 mxf4nvf4 m16n8k64 ue4m3 4X, also MXFP4 */
    .smem_bw_gbps = 329.1,
};

enum { NV_THREADS = 256, NV_MAX_STAMP = 128, NV_SMEM_OPTIN = 101376 };
enum { UB_ZERO = 0, UB_U8, UB_E4M3, UB_UE4M3, UB_BF16, UB_F16, UB_F32, UB_I32, UB_I32_DIV, UB_I32_MOD };

typedef struct {
    plow_hsa* h;
    unsigned n_sm;
    plow_hsa_kernel fill, flush, copy, digest, exec;
    plow_hsa_kernel mma, ffma, stream, cpasync, bulk, smem;
    void* flush_buf;
    size_t flush_bytes;
    unsigned long long* d_stamp;
    void* d_sink;
    unsigned long long* d_acc;
} Nv;

static volatile const char* nv_phase = "init";
static void nv_alarm(int sig) {
    (void)sig;
    fprintf(stderr, "\nHANG: no progress in %s (op waits on other blocks?)\n", (const char*)nv_phase);
    _exit(3);
}

static int nv_kernel(Nv* nv, const char* name, plow_hsa_kernel* k) {
    if (plow_hsa_get_kernel(nv->h, 0, name, k) != 0) {
        fprintf(stderr, "kernel %s: %s\n", name, plow_hsa_last_error());
        return -1;
    }
    return 0;
}

static int nv_load(Nv* nv, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* b = malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return -1; }
    fclose(f);
    int rc = plow_hsa_load_code_object(nv->h, 0, b, (size_t)n);
    if (rc) fprintf(stderr, "load %s: %s\n", path, plow_hsa_last_error());
    free(b);
    return rc;
}

/* Launch `blocks` x `threads` and wait. */
static int nv_run(Nv* nv, const plow_hsa_kernel* k, unsigned blocks, unsigned threads, unsigned dyn,
                  const void* args, size_t n) {
    if (plow_hsa_launch(nv->h, 0, k, blocks * threads, 1, 1, (uint16_t)threads, 1, 1, dyn, args, n) ||
        plow_hsa_wait(nv->h, 0)) {
        /* A device fault poisons the context: every later number would be garbage. */
        fprintf(stderr, "FAIL in %s: %s\n", (const char*)nv_phase, plow_hsa_last_error());
        exit(4);
    }
    return 0;
}

static int nv_fill(Nv* nv, void* p, uint64_t bytes, unsigned kind, unsigned seed, float a, float b) {
    struct __attribute__((packed)) { void* p; uint64_t bytes; uint32_t kind, seed; float a, b; } x =
        {p, bytes, kind, seed, a, b};
    return nv_run(nv, &nv->fill, nv->n_sm * 4, 256, 0, &x, sizeof(x));
}

static int nv_copy(Nv* nv, void* dst, const void* src, uint64_t bytes) {
    struct __attribute__((packed)) { void* d; const void* s; uint64_t n; } x = {dst, src, bytes};
    return nv_run(nv, &nv->copy, nv->n_sm * 4, 256, 0, &x, sizeof(x));
}

static int nv_flush(Nv* nv) {
    struct __attribute__((packed)) { void* p; uint64_t n16; } x = {nv->flush_buf, nv->flush_bytes / 16};
    return nv_run(nv, &nv->flush, nv->n_sm * 4, 512, 0, &x, sizeof(x));
}

static uint64_t nv_digest(Nv* nv, const void* p, uint64_t bytes) {
    uint64_t z = 0, out = 0;
    plow_hsa_upload(nv->h, 0, nv->d_acc, &z, 8);
    struct __attribute__((packed)) { void* o; const void* p; uint64_t n; } x = {nv->d_acc, p, bytes};
    if (nv_run(nv, &nv->digest, nv->n_sm * 4, 256, 0, &x, sizeof(x))) return 0;
    plow_hsa_download(nv->h, 0, &out, nv->d_acc, 8);
    return out;
}

static int cmp_dbl(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static double median(double* v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmp_dbl);
    return n ? v[n / 2] : 0.0;
}

/* --- ceilings ------------------------------------------------------------- */

typedef struct {
    double ghz;
    double mma_tf[6];          /* bf16, e4m3, e4m3 kind::f8f6f4, MXFP8, NVFP4, MXFP4 (TFLOP/s) */
    double mma_flop_clk[6];
    double ffma_tf;
    double ld_gbps[3];         /* ld.global.v4, .nc.L1::no_allocate.v4, .v8 (32 B) */
    double cpasync_gbps, bulk_gbps, l2_gbps, smem_gbps;
} NvCeil;

static const char* const MMA_NAMES[6] = {"bf16 m16n8k16", "e4m3 m16n8k32", "e4m3 kind::f8f6f4",
                                         "mxf8f6f4 ue8m0 1X", "nvfp4 mxf4nvf4 ue4m3 4X", "mxf4 ue8m0 2X"};
static const char* const MMA_KEYS[6] = {"bf16", "fp8", "fp8_f8f6f4", "mxfp8", "nvfp4", "mxfp4"};
static const double MMA_FLOP[6] = {4096, 8192, 8192, 8192, 16384, 16384};

/* Time one ceiling kernel on `blocks` blocks (1 = one SM; n_sm = whole GPU, one block per SM
 * forced by `dyn`). Returns seconds of the timed region (gridDim 1: SM cycles / clock; else the
 * globaltimer span over blocks), and the SM clock from the block-0 stamps. */
static double nv_ceil_time(Nv* nv, const plow_hsa_kernel* k, unsigned blocks, unsigned threads,
                           unsigned dyn, const void* args, size_t n, double* ghz) {
    unsigned long long st[4 * NV_MAX_STAMP];
    if (nv_run(nv, k, blocks, threads, dyn, args, n)) return -1.0;
    plow_hsa_download(nv->h, 0, st, nv->d_stamp, 32ull * blocks);
    unsigned long long t0 = ~0ull, t1 = 0;
    for (unsigned b = 0; b < blocks; b++) {
        if (st[4 * b + 2] < t0) t0 = st[4 * b + 2];
        if (st[4 * b + 3] > t1) t1 = st[4 * b + 3];
    }
    const double cyc = (double)(st[1] - st[0]), ns = (double)(st[3] - st[2]);
    if (ghz) *ghz = ns > 0 ? cyc / ns : 0.0;
    return (double)(t1 - t0) * 1e-9;
}

static int nv_ceilings(Nv* nv, NvCeil* sm, NvCeil* gpu) {
    memset(sm, 0, sizeof(*sm));
    memset(gpu, 0, sizeof(*gpu));
    const unsigned G = nv->n_sm;
    const unsigned one_per_sm = 60 * 1024; /* dyn smem that forbids two blocks per SM */
    double ghz = 0;

    for (int kind = 0; kind < 6; kind++) {
        nv_phase = MMA_NAMES[kind];
        const unsigned iters = 4096;
        double best = 0, best_g = 0;
        for (unsigned thr = 128; thr <= 512; thr *= 2) {
            struct __attribute__((packed)) { void* st; void* sink; uint32_t kind, iters, seed; } a =
                {nv->d_stamp, nv->d_sink, (uint32_t)kind, iters, 1u};
            double g = 0;
            nv_ceil_time(nv, &nv->mma, 1, thr, 0, &a, sizeof(a), NULL); /* warm */
            const double s = nv_ceil_time(nv, &nv->mma, 1, thr, 0, &a, sizeof(a), &g);
            const double fl = (double)(thr / 32) * iters * 8 * MMA_FLOP[kind];
            if (s > 0 && fl / s > best) { best = fl / s; best_g = g; }
            const double sg = nv_ceil_time(nv, &nv->mma, G, thr, one_per_sm, &a, sizeof(a), NULL);
            if (sg > 0 && fl * G / sg > gpu->mma_tf[kind] * 1e12) gpu->mma_tf[kind] = fl * G / sg / 1e12;
        }
        sm->mma_tf[kind] = best / 1e12;
        sm->mma_flop_clk[kind] = best_g > 0 ? best / (best_g * 1e9) : 0;
        if (best_g > ghz) ghz = best_g;
    }
    {
        nv_phase = "ffma";
        const unsigned iters = 2048;
        struct __attribute__((packed)) { void* st; void* sink; uint32_t iters; float x; } a =
            {nv->d_stamp, nv->d_sink, iters, 1.0f};
        for (unsigned thr = 256; thr <= 1024; thr *= 2) {
            const double fl = (double)thr * iters * 16 * 8 * 2;
            nv_ceil_time(nv, &nv->ffma, 1, thr, 0, &a, sizeof(a), NULL);
            const double s = nv_ceil_time(nv, &nv->ffma, 1, thr, 0, &a, sizeof(a), NULL);
            if (s > 0 && fl / s / 1e12 > sm->ffma_tf) sm->ffma_tf = fl / s / 1e12;
            const double sg = nv_ceil_time(nv, &nv->ffma, G, thr, one_per_sm, &a, sizeof(a), NULL);
            if (sg > 0 && fl * G / sg / 1e12 > gpu->ffma_tf) gpu->ffma_tf = fl * G / sg / 1e12;
        }
    }
    /* DRAM: 1 GiB, 4x L2 at least. */
    const uint64_t big = 1ull << 30;
    void* buf = plow_hsa_alloc(nv->h, 0, big);
    if (!buf) { fprintf(stderr, "alloc 1 GiB: %s\n", plow_hsa_last_error()); return -1; }
    nv_fill(nv, buf, big, UB_U8, 3, 0, 0);
    for (unsigned mode = 0; mode < 3; mode++) {
        nv_phase = "dram stream";
        for (unsigned thr = 256; thr <= 1024; thr *= 2) {
            struct __attribute__((packed)) { void* st; void* sink; void* buf; uint64_t bytes; uint32_t mode, passes; } a =
                {nv->d_stamp, nv->d_sink, buf, big / 4, mode, 1};
            nv_flush(nv);
            const double s = nv_ceil_time(nv, &nv->stream, 1, thr, 0, &a, sizeof(a), NULL);
            if (s > 0 && (double)(big / 4) / s / 1e9 > sm->ld_gbps[mode]) sm->ld_gbps[mode] = (double)(big / 4) / s / 1e9;
            a.bytes = big;
            nv_flush(nv);
            const double sg = nv_ceil_time(nv, &nv->stream, G, thr, one_per_sm, &a, sizeof(a), NULL);
            if (sg > 0 && (double)big / sg / 1e9 > gpu->ld_gbps[mode]) gpu->ld_gbps[mode] = (double)big / sg / 1e9;
        }
    }
    for (unsigned tile = 8192; tile <= 16384; tile *= 2) {
        nv_phase = "cp.async";
        for (unsigned thr = 256; thr <= 1024; thr *= 4) {
            struct __attribute__((packed)) { void* st; void* sink; void* buf; uint64_t bytes; uint32_t tile; } a =
                {nv->d_stamp, nv->d_sink, buf, big / 4, tile};
            nv_flush(nv);
            const double s = nv_ceil_time(nv, &nv->cpasync, 1, thr, 4 * tile, &a, sizeof(a), NULL);
            if (s > 0 && (double)(big / 4) / s / 1e9 > sm->cpasync_gbps) sm->cpasync_gbps = (double)(big / 4) / s / 1e9;
            a.bytes = big;
            nv_flush(nv);
            const double sg = nv_ceil_time(nv, &nv->cpasync, G, thr, 4 * tile, &a, sizeof(a), NULL);
            if (sg > 0 && (double)big / sg / 1e9 > gpu->cpasync_gbps) gpu->cpasync_gbps = (double)big / sg / 1e9;
        }
    }
    for (unsigned stages = 2; stages <= 6; stages += 2) {
        nv_phase = "cp.async.bulk";
        const unsigned tile = 16384;
        struct __attribute__((packed)) { void* st; void* sink; void* buf; uint64_t bytes; uint32_t tile, stages; } a =
            {nv->d_stamp, nv->d_sink, buf, big / 4, tile, stages};
        nv_flush(nv);
        const double s = nv_ceil_time(nv, &nv->bulk, 1, 128, stages * tile, &a, sizeof(a), NULL);
        if (s > 0 && (double)(big / 4) / s / 1e9 > sm->bulk_gbps) sm->bulk_gbps = (double)(big / 4) / s / 1e9;
        a.bytes = big;
        nv_flush(nv);
        const double sg = nv_ceil_time(nv, &nv->bulk, G, 128, stages * tile, &a, sizeof(a), NULL);
        if (sg > 0 && (double)big / sg / 1e9 > gpu->bulk_gbps) gpu->bulk_gbps = (double)big / sg / 1e9;
    }
    {
        nv_phase = "l2";
        const uint64_t l2b = 8ull << 20;
        for (unsigned thr = 256; thr <= 1024; thr *= 2) {
            struct __attribute__((packed)) { void* st; void* sink; void* buf; uint64_t bytes; uint32_t mode, passes; } a =
                {nv->d_stamp, nv->d_sink, buf, l2b, 0, 1};
            nv_ceil_time(nv, &nv->stream, 1, thr, 0, &a, sizeof(a), NULL); /* warm */
            a.passes = 32;
            const double s = nv_ceil_time(nv, &nv->stream, 1, thr, 0, &a, sizeof(a), NULL);
            if (s > 0 && 32.0 * l2b / s / 1e9 > sm->l2_gbps) sm->l2_gbps = 32.0 * l2b / s / 1e9;
            a.passes = 64;
            const double sg = nv_ceil_time(nv, &nv->stream, G, thr, one_per_sm, &a, sizeof(a), NULL);
            const double per = (double)((l2b / G) & ~4095ull) * G * 64;
            if (sg > 0 && per / sg / 1e9 > gpu->l2_gbps) gpu->l2_gbps = per / sg / 1e9;
        }
    }
    {
        nv_phase = "smem";
        const unsigned iters = 4096;
        for (unsigned thr = 256; thr <= 1024; thr *= 2) {
            struct __attribute__((packed)) { void* st; void* sink; uint32_t iters; } a = {nv->d_stamp, nv->d_sink, iters};
            const double by = (double)thr * iters * 16 * 16;
            const double s = nv_ceil_time(nv, &nv->smem, 1, thr, 32768, &a, sizeof(a), NULL);
            if (s > 0 && by / s / 1e9 > sm->smem_gbps) sm->smem_gbps = by / s / 1e9;
            const double sg = nv_ceil_time(nv, &nv->smem, G, thr, one_per_sm, &a, sizeof(a), NULL);
            if (sg > 0 && by * G / sg / 1e9 > gpu->smem_gbps) gpu->smem_gbps = by * G / sg / 1e9;
        }
    }
    plow_hsa_free(nv->h, buf);
    sm->ghz = gpu->ghz = ghz;
    return 0;
}

static double nv_dram_sm(const NvCeil* c) {
    double m = c->cpasync_gbps > c->bulk_gbps ? c->cpasync_gbps : c->bulk_gbps;
    for (int i = 0; i < 3; i++) m = c->ld_gbps[i] > m ? c->ld_gbps[i] : m;
    return m;
}

static void nv_ceil_print(const NvCeil* sm, const NvCeil* gpu, unsigned G) {
    printf("\nGB10 single-SM ceilings (gridDim=1; whole GPU = gridDim=%u, one block per SM)\n", G);
    printf("SM clock under load: %.3f GHz\n\n", sm->ghz);
    printf("| ceiling | one SM | FLOP or B / clk | whole GPU | whole GPU / %u |\n|---|---:|---:|---:|---:|\n", G);
    for (int k = 0; k < 6; k++)
        printf("| mma.sync %s | %.3f TFLOP/s | %.0f | %.1f TFLOP/s | %.3f |\n", MMA_NAMES[k], sm->mma_tf[k],
               sm->mma_flop_clk[k], gpu->mma_tf[k], gpu->mma_tf[k] / G);
    printf("| FP32 FFMA | %.3f TFLOP/s | %.0f | %.1f TFLOP/s | %.3f |\n", sm->ffma_tf,
           sm->ffma_tf * 1e3 / sm->ghz, gpu->ffma_tf, gpu->ffma_tf / G);
    static const char* const ldn[3] = {"ld.global.v4", "ld.global.nc.L1::no_allocate.v4", "ld.global.v8 (32 B)"};
    for (int i = 0; i < 3; i++)
        printf("| DRAM %s | %.1f GB/s | %.1f | %.1f GB/s | %.2f |\n", ldn[i], sm->ld_gbps[i],
               sm->ld_gbps[i] / sm->ghz, gpu->ld_gbps[i], gpu->ld_gbps[i] / G);
    printf("| DRAM cp.async.cg 16 B, 4 stages | %.1f GB/s | %.1f | %.1f GB/s | %.2f |\n", sm->cpasync_gbps,
           sm->cpasync_gbps / sm->ghz, gpu->cpasync_gbps, gpu->cpasync_gbps / G);
    printf("| DRAM cp.async.bulk 16 KiB + mbarrier | %.1f GB/s | %.1f | %.1f GB/s | %.2f |\n", sm->bulk_gbps,
           sm->bulk_gbps / sm->ghz, gpu->bulk_gbps, gpu->bulk_gbps / G);
    printf("| L2 hit (8 MiB, ld.global.v4) | %.1f GB/s | %.1f | %.1f GB/s | %.2f |\n", sm->l2_gbps,
           sm->l2_gbps / sm->ghz, gpu->l2_gbps, gpu->l2_gbps / G);
    printf("| shared ld.shared.v4 | %.1f GB/s | %.1f | %.1f GB/s | %.2f |\n", sm->smem_gbps,
           sm->smem_gbps / sm->ghz, gpu->smem_gbps, gpu->smem_gbps / G);
    const BenchSpec* s = &BENCH_SPEC_GB10;
    printf("\nvs the recorded %s spec (hwspec GB10_SM_CEILINGS): bf16 %+.1f%%, fp8 %+.1f%%, nvfp4 %+.1f%%, "
           "DRAM %+.1f%%, L2 %+.1f%%, smem %+.1f%%\n", s->name,
           100.0 * (sm->mma_tf[0] / s->peak_tflops_bf16 - 1), 100.0 * (sm->mma_tf[1] / s->peak_tflops_fp8 - 1),
           100.0 * (sm->mma_tf[4] / s->peak_tflops_fp4 - 1), 100.0 * (nv_dram_sm(sm) / s->hbm_bw_gbps - 1),
           100.0 * (sm->l2_gbps / s->l2_bw_gbps - 1), 100.0 * (sm->smem_gbps / s->smem_bw_gbps - 1));
}

static void nv_ceil_json(FILE* f, const char* key, const NvCeil* c) {
    fprintf(f, "  \"%s\": {\"clock_ghz\": %.4f", key, c->ghz);
    for (int k = 0; k < 6; k++) fprintf(f, ", \"%s_tflops\": %.5f", MMA_KEYS[k], c->mma_tf[k]);
    fprintf(f, ", \"fp32_tflops\": %.5f, \"dram_ld_v4_gbps\": %.3f, \"dram_ld_v4_na_gbps\": %.3f, "
               "\"dram_ld_v8_gbps\": %.3f, \"dram_cpasync_gbps\": %.3f, \"dram_bulk_gbps\": %.3f, "
               "\"dram_gbps\": %.3f, \"l2_gbps\": %.3f, \"smem_gbps\": %.3f}",
            c->ffma_tf, c->ld_gbps[0], c->ld_gbps[1], c->ld_gbps[2], c->cpasync_gbps, c->bulk_gbps,
            nv_dram_sm(c), c->l2_gbps, c->smem_gbps);
}

/* --- NVFP4 tile loops (fp4_tile_sm120.cu) ---------------------------------- */

/* FtOperand / ft_<shape>_m<mode> parameters, natural alignment = the CUDA parameter layout */
typedef struct {
    const void* p;
    const void* s;
    uint32_t kb, nkt, ts;
} FtOp;
typedef struct {
    void* st;
    void* sink;
    FtOp a, b;
    uint32_t nk, pfd;
    const void* tma;
    const void* tmb;
} FtArgs;

enum { FT_A_BYTES = 32u << 20, FT_B_BYTES = 64u << 20 };

/* Operand layouts (kb, nkt, ts) of A then B: step ks reads tile ks / nkt at p + tile * ts + row * kb
 * + (ks % nkt) * 64; ts = 0 keeps an operand L2-resident. up / down / sdown = the Nemotron-3.5-Super
 * routed up (K 1024 gathered, A reused), routed down (K 2688) and shared down (K 5376) row-major
 * weights; tiled = a stage per contiguous block; l2 = both operands L2-resident. */
static void ft_layout(const char* l, unsigned bm, unsigned bn, unsigned nk, uint32_t a[3], uint32_t b[3]) {
    uint32_t v[6] = {4096, 64, 0, nk * 64u, nk, 0};
    if (!strcmp(l, "up")) { uint32_t w[6] = {512, 8, 0, 512, 8, bn * 512u}; memcpy(v, w, sizeof(v)); }
    else if (!strcmp(l, "down")) { uint32_t w[6] = {1344, 21, bm * 1344u, 1344, 21, bn * 1344u}; memcpy(v, w, sizeof(v)); }
    else if (!strcmp(l, "sdown")) { uint32_t w[6] = {2688, 42, 0, 2688, 42, bn * 2688u}; memcpy(v, w, sizeof(v)); }
    else if (!strcmp(l, "tiled")) { uint32_t w[6] = {64, 1, bm * 64u, 64, 1, bn * 64u}; memcpy(v, w, sizeof(v)); }
    else if (!strcmp(l, "l2")) { uint32_t w[6] = {512, 8, 0, 512, 8, 0}; memcpy(v, w, sizeof(v)); }
    memcpy(a, v, 12);
    memcpy(b, v + 3, 12);
}

/* u8 [rows][kb] tensor map over a whole operand buffer, box 64 B x box_rows, 64 B swizzle (= the ring's
 * 16 B chunk ^ (row >> 1 & 3)), uploaded to `dst` (128 B, device) */
static int ft_tmap(Nv* nv, void* dst, const void* base, uint64_t bytes, uint32_t kb, uint32_t box_rows) {
    CUtensorMap m;
    const cuuint64_t dim[2] = {kb, bytes / kb}, stride[1] = {kb};
    const cuuint32_t box[2] = {64, box_rows}, es[2] = {1, 1};
    if (cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)base, dim, stride, box, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed (kb %u, box rows %u)\n", kb, box_rows);
        return -1;
    }
    return plow_hsa_upload(nv->h, 0, dst, &m, sizeof(m));
}

typedef struct {
    void *a, *sa, *b, *sb, *tm;
} FtBufs;

/* One ft_ kernel at gridDim 1, best of `reps` (L2 flushed before each): prints a table row */
static void ft_row(Nv* nv, const FtBufs* bf, const char* tag, unsigned bm, unsigned bn, unsigned nw, unsigned mode,
                   unsigned nst, const char* lay, unsigned nk, unsigned reps, FILE* jf, int* first) {
    char name[64];
    if (!strncmp(tag, "chain", 5)) snprintf(name, sizeof(name), "ft_%s", tag);
    else snprintf(name, sizeof(name), "ft_%s_m%u%s", tag, mode, nst == 4 ? "s4" : "");
    plow_hsa_kernel k;
    nv_phase = name;
    if (nv_kernel(nv, name, &k)) return;
    FtArgs x;
    memset(&x, 0, sizeof(x));
    x.st = nv->d_stamp;
    x.sink = nv->d_sink;
    uint32_t la[3], lb[3];
    ft_layout(lay, bm, bn, nk, la, lb);
    x.a = (FtOp){bf->a, bf->sa, la[0], la[1], la[2]};
    x.b = (FtOp){bf->b, bf->sb, lb[0], lb[1], lb[2]};
    x.nk = nk;
    if (mode == 11) {
        if (ft_tmap(nv, bf->tm, bf->a, FT_A_BYTES, la[0], bm) ||
            ft_tmap(nv, (uint8_t*)bf->tm + 128, bf->b, FT_B_BYTES, lb[0], bn))
            return;
        x.tma = bf->tm;
        x.tmb = (uint8_t*)bf->tm + 128;
    }
    /* + 1 KiB: the kernel aligns its ring to 1 KiB (the TMA swizzle is on absolute smem address bits) */
    const unsigned dyn = (bm + bn) * 72u * (mode >= 3 ? nst : 1u) + 1024u;
    double cyc = 0, ns = 0;
    for (unsigned r = 0; r < reps; r++) {
        unsigned long long st[4];
        nv_flush(nv);
        nv_run(nv, &k, 1, nw * 32u, dyn, &x, sizeof(x));
        plow_hsa_download(nv->h, 0, st, nv->d_stamp, sizeof(st));
        if (!r || (double)(st[1] - st[0]) < cyc) { cyc = (double)(st[1] - st[0]); ns = (double)(st[3] - st[2]); }
    }
    const double tf = 2.0 * bm * bn * 128.0 * nk / (ns * 1e-9) / 1e12, ceil = BENCH_SPEC_GB10.peak_tflops_fp4;
    const double bpc = (double)(bm + bn) * 72.0 * nk / cyc;
    printf("| %s | %u | %u | %s | %s | %.3f | %.1f%% | %.0f | %.1f |\n", name, nw, mode,
           mode >= 3 ? (nst == 4 ? "4" : "3") : "-", lay, tf, 100.0 * tf / ceil, cyc / nk, bpc);
    if (jf)
        fprintf(jf, "%s    {\"kernel\": \"%s\", \"warps\": %u, \"mode\": %u, \"stages\": %u, \"layout\": \"%s\", "
                    "\"tflops\": %.4f, \"pct_ceiling\": %.2f, \"clk_per_k128\": %.1f, \"staged_b_per_clk\": %.2f}",
                *first ? "" : ",\n", name, nw, mode, mode >= 3 ? nst : 0, lay, tf, 100.0 * tf / ceil, cyc / nk, bpc);
    *first = 0;
}

static int nv_fp4_tile(Nv* nv, const char* cubin, unsigned nk, unsigned reps, FILE* jf) {
    if (nv_load(nv, cubin)) return 1;
    FtBufs bf = {plow_hsa_alloc(nv->h, 0, FT_A_BYTES), plow_hsa_alloc(nv->h, 0, FT_A_BYTES / 8),
                 plow_hsa_alloc(nv->h, 0, FT_B_BYTES), plow_hsa_alloc(nv->h, 0, FT_B_BYTES / 8),
                 plow_hsa_alloc(nv->h, 0, 256)};
    if (!bf.a || !bf.sa || !bf.b || !bf.sb || !bf.tm) { fprintf(stderr, "alloc: %s\n", plow_hsa_last_error()); return 1; }
    nv_fill(nv, bf.a, FT_A_BYTES, UB_U8, 1, 0, 0);
    nv_fill(nv, bf.b, FT_B_BYTES, UB_U8, 2, 0, 0);
    nv_fill(nv, bf.sa, FT_A_BYTES / 8, UB_UE4M3, 3, 0, 0);
    nv_fill(nv, bf.sb, FT_B_BYTES / 8, UB_UE4M3, 4, 0, 0);
    printf("NVFP4 tile loops (fp4_tile_sm120.cu), gridDim 1, nk = %u k128 steps, best of %u; ceiling %.3f TF/s\n\n"
           "| kernel | warps | mode | stages | layout | TF/s per SM | %% of ceiling | clk per k128 step | staged B/clk |\n"
           "|---|---:|---:|---:|---|---:|---:|---:|---:|\n", nk, reps, BENCH_SPEC_GB10.peak_tflops_fp4);
    if (jf) fprintf(jf, "  \"nk\": %u,\n  \"fp4_tile\": [\n", nk);
    int first = 1;
    static const struct { const char* tag; unsigned bm, bn, nw; } CH[] = {
        {"chain2", 16, 16, 1}, {"chain4", 16, 32, 1}, {"chain8", 32, 32, 1}, {"chain16", 32, 64, 1},
        {"chain32", 64, 64, 1}, {"chain32_w2", 128, 64, 2}};
    for (size_t i = 0; i < sizeof(CH) / sizeof(CH[0]); i++)
        ft_row(nv, &bf, CH[i].tag, CH[i].bm, CH[i].bn, CH[i].nw, 0, 1, "syn", nk, reps, jf, &first);
    static const struct { const char* tag; unsigned bm, bn, nw; } SH[] = {
        {"128x256_w8", 128, 256, 8}, {"128x192_w8", 128, 192, 8}, {"128x128_w8", 128, 128, 8},
        {"128x128_w4", 128, 128, 4}, {"64x256_w8", 64, 256, 8}};
    for (size_t i = 0; i < sizeof(SH) / sizeof(SH[0]); i++)
        for (unsigned m = 0; m < 3; m++) ft_row(nv, &bf, SH[i].tag, SH[i].bm, SH[i].bn, SH[i].nw, m, 1, "syn", nk, reps, jf, &first);
    static const char* const LAY[] = {"up", "down", "sdown", "tiled", "l2"};
    for (size_t i = 0; i < 2; i++) {
        for (unsigned m = 3; m <= 7; m++)
            for (size_t l = 0; l < 5; l++)
                if (m == 3 || l < 2) ft_row(nv, &bf, SH[i].tag, SH[i].bm, SH[i].bn, 8, m, 3, LAY[l], nk, reps, jf, &first);
        ft_row(nv, &bf, SH[i].tag, SH[i].bm, SH[i].bn, 8, 8, 3, "tiled", nk, reps, jf, &first);
        for (size_t l = 0; l < 5; l++) ft_row(nv, &bf, SH[i].tag, SH[i].bm, SH[i].bn, 8, 11, 3, LAY[l], nk, reps, jf, &first);
    }
    ft_row(nv, &bf, "128x192_w8", 128, 192, 8, 3, 4, "down", nk, reps, jf, &first);
    ft_row(nv, &bf, "128x192_w8", 128, 192, 8, 11, 4, "down", nk, reps, jf, &first);
    ft_row(nv, &bf, "128x128_w8", 128, 128, 8, 8, 3, "tiled", nk, reps, jf, &first);
    ft_row(nv, &bf, "128x128_w8", 128, 128, 8, 11, 3, "down", nk, reps, jf, &first);
    if (jf) fprintf(jf, "\n  ]\n");
    return 0;
}

/* --- packet ---------------------------------------------------------------- */

typedef struct {
    uint8_t* buf;
    PlowBlobHeader h;
    PlowTensorDecl* decl;
    uint8_t* init;
    uint32_t n_prog;
    PlowProgHeader* ph;
    PlowDevInst** insts;
} NvBlob;

/* v5..v7 flat layout (dev_blob.h). Generated (v7) tensors are never materialised: the harness
 * fills every tensor itself, so a v7 blob's RoPE tables are random here like any other input. */
static int nv_blob(const char* path, NvBlob* b) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    b->buf = (uint8_t*)malloc((size_t)n);
    if (fread(b->buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); return -1; }
    fclose(f);
    memcpy(&b->h, b->buf, sizeof(PlowBlobHeader));
    if (memcmp(b->h.magic, PLOW_BLOB_MAGIC, 8) && memcmp(b->h.magic, PLOW_BLOB_MAGIC_V6, 8) &&
        memcmp(b->h.magic, PLOW_BLOB_MAGIC_V7, 8)) {
        fprintf(stderr, "%s: %s\n", path, plow_blob_magic_error(b->h.magic));
        return -1;
    }
    uint8_t* q = b->buf + sizeof(PlowBlobHeader);
    b->decl = (PlowTensorDecl*)q;
    q += (size_t)b->h.n_tensor * sizeof(PlowTensorDecl);
    b->init = q;
    q += b->h.init_bytes;
    q += (size_t)b->h.n_kvrow * 4;
    b->n_prog = b->h.n_prog;
    b->ph = (PlowProgHeader*)calloc(b->n_prog, sizeof(PlowProgHeader));
    b->insts = (PlowDevInst**)calloc(b->n_prog, sizeof(PlowDevInst*));
    for (uint32_t i = 0; i < b->n_prog; i++) {
        memcpy(&b->ph[i], q, sizeof(PlowProgHeader));
        q += sizeof(PlowProgHeader);
        b->insts[i] = (PlowDevInst*)q;
        q += (size_t)b->ph[i].n_inst * sizeof(PlowDevInst);
        q += (size_t)b->ph[i].n_stream * sizeof(PlowStreamEnt);
        q += (size_t)b->h.n_cu * 8;
        q += (size_t)b->ph[i].n_wait * sizeof(PlowWait);
        q += (size_t)b->ph[i].n_succ * 4;
        if (q > b->buf + n) { fprintf(stderr, "%s: truncated program table\n", path); return -1; }
    }
    return 0;
}

/* --- cases ----------------------------------------------------------------
 * One instruction of one program, picked by opcode + a substring of one of its tensor names (nth
 * match), plus the producers it needs to see real-format inputs (routing tables, q tensors, scan
 * state): the nearest preceding instruction of each `setup` opcode, run once on the full grid. */
enum { CASE_PLAIN = 1 }; /* Mamba conv/step: the plain (non-span) body, rowslot operands cleared */

typedef struct {
    const char* set;
    const char* name;
    unsigned t;
    uint16_t op;
    const char* match;
    uint16_t setup[5];
    unsigned flags;
    unsigned ctx; /* decode kv_len / pos; 0 = 1024 */
} NvCase;

#define LOGITS PLOW_DOP_MOE_ROUTER_LOGITS_F32
#define TOPK PLOW_DOP_MOE_ROUTER_SIGMOID_TOPK
static const NvCase NV_CASES[] = {
    /* Nemotron-3.5-Super decode (row-slot object: the MTP verify tick; T = verify rows) */
    {"nemotron-dec", "moe_up_routed", 1, PLOW_DOP_MOE_UP_NVFP4, "cat/", {LOGITS, TOPK}, 0, 0},
    {"nemotron-dec", "moe_up_routed", 4, PLOW_DOP_MOE_UP_NVFP4, "cat/", {LOGITS, TOPK}, 0, 0},
    {"nemotron-dec", "moe_up_routed", 16, PLOW_DOP_MOE_UP_NVFP4, "cat/", {LOGITS, TOPK}, 0, 0},
    {"nemotron-dec", "moe_down_routed", 1, PLOW_DOP_MOE_DOWN_NVFP4, "cat/", {LOGITS, TOPK}, 0, 0},
    {"nemotron-dec", "moe_down_routed", 4, PLOW_DOP_MOE_DOWN_NVFP4, "cat/", {LOGITS, TOPK}, 0, 0},
    {"nemotron-dec", "moe_down_routed", 16, PLOW_DOP_MOE_DOWN_NVFP4, "cat/", {LOGITS, TOPK}, 0, 0},
    {"nemotron-dec", "moe_up_shared", 1, PLOW_DOP_MOE_UP_NVFP4, "shared_experts", {0}, 0, 0},
    {"nemotron-dec", "moe_up_shared", 4, PLOW_DOP_MOE_UP_NVFP4, "shared_experts", {0}, 0, 0},
    {"nemotron-dec", "moe_up_shared", 16, PLOW_DOP_MOE_UP_NVFP4, "shared_experts", {0}, 0, 0},
    {"nemotron-dec", "moe_down_shared", 4, PLOW_DOP_MOE_DOWN_NVFP4, "shared_experts", {0}, 0, 0},
    {"nemotron-dec", "moe_down_shared", 16, PLOW_DOP_MOE_DOWN_NVFP4, "shared_experts", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_in_proj", 1, PLOW_DOP_GEMV_FP8, "in_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_in_proj", 4, PLOW_DOP_GEMV_FP8, "in_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_in_proj", 16, PLOW_DOP_GEMV_FP8, "in_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_out_proj", 1, PLOW_DOP_GEMV_FP8, "out_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_out_proj", 4, PLOW_DOP_GEMV_FP8, "out_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_out_proj", 16, PLOW_DOP_GEMV_FP8, "out_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_attn_q", 4, PLOW_DOP_GEMV_FP8, "q_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_attn_o", 4, PLOW_DOP_GEMV_FP8, "o_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_lm_head", 4, PLOW_DOP_GEMV_FP8, "lm_head", {0}, 0, 0},
    {"nemotron-dec", "gemv_bf16_fc1_latent", 4, PLOW_DOP_GEMV, "fc1_latent", {0}, 0, 0},
    {"nemotron-dec", "gemv_bf16_fc2_latent", 4, PLOW_DOP_GEMV, "fc2_latent", {0}, 0, 0},
    {"nemotron-dec", "router_logits", 4, LOGITS, "", {0}, 0, 0},
    {"nemotron-dec", "router_topk", 4, TOPK, "", {LOGITS}, 0, 0},
    {"nemotron-dec", "router_topk", 16, TOPK, "", {LOGITS}, 0, 0},
    {"nemotron-dec", "mamba2_conv_span", 4, PLOW_DOP_MAMBA2_CONV, "", {0}, 0, 0},
    {"nemotron-dec", "mamba2_step_span", 1, PLOW_DOP_MAMBA2_STEP, "", {0}, 0, 0},
    {"nemotron-dec", "mamba2_step_span", 4, PLOW_DOP_MAMBA2_STEP, "", {0}, 0, 0},
    {"nemotron-dec", "mamba2_step_span", 16, PLOW_DOP_MAMBA2_STEP, "", {0}, 0, 0},
    {"nemotron-dec", "mamba2_step_plain", 4, PLOW_DOP_MAMBA2_STEP, "", {0}, CASE_PLAIN, 0},
    {"nemotron-dec", "mamba2_gated_norm", 4, PLOW_DOP_MAMBA2_GATED_NORM, "", {0}, 0, 0},
    {"nemotron-dec", "rmsnorm", 4, PLOW_DOP_QWEN_RMSNORM, "", {0}, 0, 0},
    {"nemotron-dec", "residual", 4, PLOW_DOP_RESIDUAL, "", {0}, 0, 0},
    {"nemotron-dec", "moe_combine", 4, PLOW_DOP_MOE_COMBINE_PF, "", {0}, 0, 0},
    {"nemotron-dec", "headnorm_rope_kv", 4, PLOW_DOP_QWEN_HEADNORM_ROPE, "", {0}, 0, 0},
    {"nemotron-dec", "flash_decode_gqa16", 4, PLOW_DOP_FLASH_DECODE, "", {0}, 0, 1024},
    {"nemotron-dec", "flash_decode_gqa16_32k", 4, PLOW_DOP_FLASH_DECODE, "", {0}, 0, 32768},
    {"nemotron-dec", "flash_decode_gqa16_32k", 32, PLOW_DOP_FLASH_DECODE, "", {0}, 0, 32768},
    {"nemotron-dec", "flash_decode_gqa16_128k", 4, PLOW_DOP_FLASH_DECODE, "", {0}, 0, 131072},
    {"nemotron-dec", "flash_decode_gqa16_128k", 32, PLOW_DOP_FLASH_DECODE, "", {0}, 0, 131072},
    {"nemotron-dec", "flash_merge_32k", 32, PLOW_DOP_FLASH_MERGE, "", {PLOW_DOP_FLASH_DECODE}, 0, 32768},
    {"nemotron-dec", "flash_merge", 4, PLOW_DOP_FLASH_MERGE, "", {PLOW_DOP_FLASH_DECODE}, 0, 1024},
    {"nemotron-dec", "embed", 4, PLOW_DOP_EMBED, "", {0}, 0, 0},
    /* T=1 (plain rung) and wide verify rungs: identity coverage for every changed body */
    {"nemotron-dec", "router_topk", 1, TOPK, "", {LOGITS}, 0, 0},
    {"nemotron-dec", "router_topk", 32, TOPK, "", {LOGITS}, 0, 0},
    {"nemotron-dec", "mamba2_conv_span", 1, PLOW_DOP_MAMBA2_CONV, "", {0}, 0, 0},
    {"nemotron-dec", "mamba2_conv_span", 16, PLOW_DOP_MAMBA2_CONV, "", {0}, 0, 0},
    {"nemotron-dec", "mamba2_conv_span", 32, PLOW_DOP_MAMBA2_CONV, "", {0}, 0, 0},
    {"nemotron-dec", "flash_decode_gqa16", 1, PLOW_DOP_FLASH_DECODE, "", {0}, 0, 1024},
    {"nemotron-dec", "flash_decode_gqa16_32k", 1, PLOW_DOP_FLASH_DECODE, "", {0}, 0, 32768},
    {"nemotron-dec", "flash_merge", 1, PLOW_DOP_FLASH_MERGE, "", {PLOW_DOP_FLASH_DECODE}, 0, 1024},
    {"nemotron-dec", "flash_merge_32k", 1, PLOW_DOP_FLASH_MERGE, "", {PLOW_DOP_FLASH_DECODE}, 0, 32768},
    {"nemotron-dec", "flash_merge_32k", 4, PLOW_DOP_FLASH_MERGE, "", {PLOW_DOP_FLASH_DECODE}, 0, 32768},
    {"nemotron-dec", "gemv_fp8_attn_q", 1, PLOW_DOP_GEMV_FP8, "q_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_attn_o", 1, PLOW_DOP_GEMV_FP8, "o_proj", {0}, 0, 0},
    {"nemotron-dec", "gemv_fp8_lm_head", 1, PLOW_DOP_GEMV_FP8, "lm_head", {0}, 0, 0},
    /* Nemotron-3.5-Super prefill (one 8192-row chunk) */
    {"nemotron-pf", "moe_quant_nvfp4", 8192, PLOW_DOP_MOE_QUANT_NVFP4, "act.moe.xq", {0}, 0, 0},
    {"nemotron-pf", "moe_align", 8192, PLOW_DOP_MOE_ALIGN_NVFP4, "", {LOGITS, TOPK}, 0, 0},
    {"nemotron-pf", "moe_up_pf_w4a4_routed", 8192, PLOW_DOP_MOE_UP_NVFP4_PF, "cat/",
     {LOGITS, TOPK, PLOW_DOP_MOE_ALIGN_NVFP4, PLOW_DOP_MOE_QUANT_NVFP4}, 0, 0},
    {"nemotron-pf", "moe_down_pf_w4a4_routed", 8192, PLOW_DOP_MOE_DOWN_NVFP4_PF, "cat/",
     {LOGITS, TOPK, PLOW_DOP_MOE_ALIGN_NVFP4, PLOW_DOP_MOE_QUANT_NVFP4, PLOW_DOP_MOE_UP_NVFP4_PF}, 0, 0},
    {"nemotron-pf", "moe_up_pf_w4a4_shared", 8192, PLOW_DOP_MOE_UP_NVFP4_PF, "shared_experts",
     {PLOW_DOP_MOE_QUANT_NVFP4}, 0, 0},
    {"nemotron-pf", "moe_down_pf_w4a4_shared", 8192, PLOW_DOP_MOE_DOWN_NVFP4_PF, "shared_experts",
     {PLOW_DOP_MOE_QUANT_NVFP4, PLOW_DOP_MOE_UP_NVFP4_PF}, 0, 0},
    {"nemotron-pf", "router_logits_pf", 8192, LOGITS, "", {0}, 0, 0},
    {"nemotron-pf", "router_topk_pf", 8192, TOPK, "", {LOGITS}, 0, 0},
    {"nemotron-pf", "moe_combine_pf", 8192, PLOW_DOP_MOE_COMBINE_PF, "",
     {LOGITS, TOPK}, 0, 0},
    {"nemotron-pf", "mamba2_conv_pf", 8192, PLOW_DOP_MAMBA2_CONV, "", {0}, 0, 0},
    {"nemotron-pf", "mamba2_chunk_scan_state", 8192, PLOW_DOP_MAMBA2_CHUNK_STATE, "", {0}, 0, 0},
    {"nemotron-pf", "mamba2_chunk_out", 8192, PLOW_DOP_MAMBA2_CHUNK_OUT, "", {PLOW_DOP_MAMBA2_CHUNK_STATE}, 0, 0},
    {"nemotron-pf", "mamba2_gated_norm_fp8", 8192, PLOW_DOP_MAMBA2_GATED_NORM, "", {0}, 0, 0},
    {"nemotron-pf", "flash_prefill_hd128_e4m3", 8192, PLOW_DOP_FLASH_PREFILL, "", {0}, 0, 0},
    {"nemotron-pf", "quant_fp8", 8192, PLOW_DOP_QUANT_FP8, "", {0}, 0, 0},
    /* FP8 projection GEMMs run in cuBLASLt on this packet (not a plow inner loop) */
    {"nemotron-pf", "gemm_bf16_fc1_latent", 8192, PLOW_DOP_GEMM, "fc1_latent", {0}, 0, 0},
    {"nemotron-pf", "rmsnorm_pf", 8192, PLOW_DOP_QWEN_RMSNORM, "", {0}, 0, 0},
    {"nemotron-pf", "headnorm_rope_pf", 8192, PLOW_DOP_QWEN_HEADNORM_ROPE, "", {0}, 0, 0},
    {"nemotron-pf", "embed_pf", 8192, PLOW_DOP_EMBED, "", {0}, 0, 0},
    /* Qwen3.8-27B FP8 decode (row-slot object) */
    {"qwen-dec", "gdn_conv_span", 4, PLOW_DOP_QWEN_GDN_CONV, "", {0}, 0, 0},
    {"qwen-dec", "gdn_step_span", 1, PLOW_DOP_QWEN_GDN_STEP, "", {0}, 0, 0},
    {"qwen-dec", "gdn_step_span", 4, PLOW_DOP_QWEN_GDN_STEP, "", {0}, 0, 0},
    {"qwen-dec", "gdn_step_span", 16, PLOW_DOP_QWEN_GDN_STEP, "", {0}, 0, 0},
    {"qwen-dec", "gdn_step_plain", 4, PLOW_DOP_QWEN_GDN_STEP, "", {0}, CASE_PLAIN, 0},
    {"qwen-dec", "gated_norm", 4, PLOW_DOP_QWEN_GATED_NORM, "", {0}, 0, 0},
    {"qwen-dec", "gemv_fp8_in_proj_qkv", 1, PLOW_DOP_GEMV_FP8, "in_proj_qkv", {0}, 0, 0},
    {"qwen-dec", "gemv_fp8_in_proj_qkv", 4, PLOW_DOP_GEMV_FP8, "in_proj_qkv", {0}, 0, 0},
    {"qwen-dec", "gemv_fp8_in_proj_qkv", 16, PLOW_DOP_GEMV_FP8, "in_proj_qkv", {0}, 0, 0},
    {"qwen-dec", "gemv_fp8_mlp_down", 1, PLOW_DOP_GEMV_FP8, "down_proj", {0}, 0, 0},
    {"qwen-dec", "gemv_fp8_mlp_down", 4, PLOW_DOP_GEMV_FP8, "down_proj", {0}, 0, 0},
    {"qwen-dec", "gemv_fp8_mlp_down", 16, PLOW_DOP_GEMV_FP8, "down_proj", {0}, 0, 0},
    {"qwen-dec", "glu", 4, PLOW_DOP_GLU, "", {0}, 0, 0},
    {"qwen-dec", "rmsnorm", 4, PLOW_DOP_QWEN_RMSNORM, "", {0}, 0, 0},
    {"qwen-dec", "headnorm_rope_q_hd256", 4, PLOW_DOP_QWEN_HEADNORM_ROPE, "q_norm", {0}, 0, 0},
    {"qwen-dec", "flash_decode_hd256", 4, PLOW_DOP_FLASH_DECODE, "", {0}, 0, 1024},
    {"qwen-dec", "flash_decode_hd256_32k", 4, PLOW_DOP_FLASH_DECODE, "", {0}, 0, 32768},
    {"qwen-dec", "flash_merge_hd256", 4, PLOW_DOP_FLASH_MERGE, "", {PLOW_DOP_FLASH_DECODE}, 0, 1024},
    {"qwen-dec", "embed", 4, PLOW_DOP_EMBED, "", {0}, 0, 0},
    /* Qwen3.8-27B FP8 prefill (8192 rows) */
    {"qwen-pf", "flash_prefill_hd256", 8192, PLOW_DOP_FLASH_PREFILL, "", {0}, 0, 0},
    {"qwen-pf", "gdn_conv_prefill", 8192, PLOW_DOP_QWEN_GDN_CONV_PREFILL, "", {0}, 0, 0},
    {"qwen-pf", "gdn_qkv_prep", 8192, PLOW_DOP_QWEN_GDN_QKV_PREP, "", {0}, 0, 0},
    {"qwen-pf", "gdn_gate_prep", 8192, PLOW_DOP_QWEN_GDN_GATE_PREP, "", {0}, 0, 0},
    {"qwen-pf", "gated_norm_pf", 8192, PLOW_DOP_QWEN_GATED_NORM, "", {0}, 0, 0},
    {"qwen-pf", "glu_pf", 8192, PLOW_DOP_GLU, "", {0}, 0, 0},
    {"qwen-pf", "quant_fp8_pf", 8192, PLOW_DOP_QUANT_FP8, "", {0}, 0, 0},
    {"qwen-pf", "rmsnorm_pf", 8192, PLOW_DOP_QWEN_RMSNORM, "", {0}, 0, 0},
    {"qwen-pf", "headnorm_rope_q_pf", 8192, PLOW_DOP_QWEN_HEADNORM_ROPE, "q_norm", {0}, 0, 0},
};
#undef LOGITS
#undef TOPK
#define N_NV_CASES (sizeof(NV_CASES) / sizeof(NV_CASES[0]))

/* Tensor handles an op reads through its integer words instead of t[] (dispatch in
 * interp_sm120.cu). */
static int nv_iword_handles(const PlowDevInst* in, unsigned* out) {
    int n = 0;
    switch (in->op) {
    case PLOW_DOP_MAMBA2_CONV:
        if (in->i[5] >> 16) out[n++] = in->i[5] >> 16;
        break;
    case PLOW_DOP_MAMBA2_STEP:
        if (in->i[6] >> 16) {
            out[n++] = in->i[6] >> 16;
            if (in->i[5] >> 16) out[n++] = in->i[5] >> 16;
            if (in->i[7] >> 16) out[n++] = in->i[7] >> 16;
        }
        break;
    case PLOW_DOP_QWEN_GDN_STEP:
        if (in->fj[2].u) {
            out[n++] = in->fj[2].u;
            out[n++] = in->i[6];
        }
        break;
    case PLOW_DOP_FLASH_DECODE:
        if (in->fj[2].u) {
            out[n++] = in->fj[2].u & 0xFFFFu;
            out[n++] = in->fj[2].u >> 16;
        }
        break;
    default: break;
    }
    return n;
}

static int has(const char* s, const char* sub) { return strstr(s, sub) != NULL; }
static int starts(const char* s, const char* p) { return strncmp(s, p, strlen(p)) == 0; }

typedef struct { unsigned kind; float a, b; } NvFill;

/* Input data by (opcode, slot) where the slot's dtype is not readable from the name, else by the
 * packet's tensor-naming conventions. Values only need the right format: timing is data-blind
 * except for routing tables, row maps and lengths, which come from setup producers or the case. */
static NvFill nv_fill_rule(const char* name, uint16_t op, int slot, const NvCase* c, unsigned rows, int kv_fp8) {
    const unsigned ctx = c->ctx ? c->ctx : 1024;
    NvFill z = {UB_ZERO, 0, 0}, pos = {UB_F32, 0.005f, 0.01f};
    switch (op) {
    case PLOW_DOP_MOE_UP_NVFP4: case PLOW_DOP_MOE_DOWN_NVFP4:
    case PLOW_DOP_MOE_UP_NVFP4_PF: case PLOW_DOP_MOE_DOWN_NVFP4_PF:
        if (slot == 3) return (NvFill){UB_U8, 0, 0};
        if (slot == 4) return (NvFill){UB_UE4M3, 0, 0};
        if (slot == 5) return pos;
        if (slot == 7 && op == PLOW_DOP_MOE_UP_NVFP4_PF) return pos;
        break;
    case PLOW_DOP_MOE_QUANT_NVFP4:
        if (slot == 2) return pos;
        break;
    case PLOW_DOP_GEMV_FP8:
        if (slot == 2) return (NvFill){UB_E4M3, 0, 0};
        if (slot == 5) return pos;
        break;
    case PLOW_DOP_GEMM_FP8:
        if (slot == 1 || slot == 2) return (NvFill){UB_E4M3, 0, 0};
        if (slot == 3 || slot == 4) return pos;
        break;
    case PLOW_DOP_QUANT_FP8:
    case PLOW_DOP_MAMBA2_GATED_NORM:
        if ((op == PLOW_DOP_QUANT_FP8 && slot == 2) || (op == PLOW_DOP_MAMBA2_GATED_NORM && slot == 5)) return pos;
        break;
    case PLOW_DOP_MOE_ROUTER_SIGMOID_TOPK:
        if (slot == 2) return (NvFill){UB_F32, 0.01f, 0};
        break;
    case PLOW_DOP_EMBED:
        if (slot == 2) return (NvFill){UB_I32_MOD, 0, 1000};
        break;
    default: break;
    }
    if (starts(name, "in.")) {
        if (has(name, "in.ids")) return (NvFill){UB_I32_MOD, 0, 1000};
        if (has(name, "in.pos")) return (NvFill){UB_I32, (float)(ctx - 1), 0};
        if (has(name, "in.kvlen")) return (NvFill){UB_I32, (float)ctx, 0};
        if (has(name, "in.prefill_active")) return (NvFill){UB_I32, (float)rows, 0};
        if (has(name, "in.active")) return (NvFill){UB_I32, 1, 0};
        if (has(name, "in.rowslot")) return (NvFill){UB_I32_DIV, 0, 4};
        if (has(name, "in.cos") || has(name, "in.sin")) return (NvFill){UB_F32, 1.0f, 0};
        return z;
    }
    if (has(name, ".rec") || has(name, ".snap")) return z;
    if (starts(name, "state.")) {
        if (has(name, ".ssm")) return (NvFill){UB_F16, 0.1f, 0};
        if (has(name, ".gdn")) return (NvFill){UB_F32, 0.1f, 0};
        return (NvFill){UB_BF16, 0.1f, 0};
    }
    if (starts(name, "kv.")) return kv_fp8 ? (NvFill){UB_E4M3, 0, 0} : (NvFill){UB_BF16, 1.0f, 0};
    if (starts(name, "act.")) {
        if (has(name, ".fp8.")) return (NvFill){UB_E4M3, 0, 0};
        if (has(name, "act.moe.table") || has(name, "act.moe.meta") || has(name, "act.moe.row_")) return z;
        if (has(name, "act.moe.xq") || has(name, "act.moe.hq") || has(name, "act.moe.fq") || has(name, "act.moe.sfq"))
            return (NvFill){UB_UE4M3, 0, 0};
        if (has(name, "logits") || has(name, "part") || has(name, "cstate") || has(name, ".cum") ||
            has(name, "nemo.dt") || has(name, "alpha") || has(name, "beta") || has(name, "amax"))
            return (NvFill){UB_F32, 1.0f, 0};
        return (NvFill){UB_BF16, 1.0f, 0};
    }
    /* weights */
    if (starts(name, "fp8/")) return has(name, "scale") ? pos : (NvFill){UB_E4M3, 0, 0};
    if (has(name, "weight_scale_2") || has(name, "input_scale")) return pos;
    if (has(name, "norm.weight") || has(name, "layernorm") || has(name, "A_log")) return (NvFill){UB_BF16, 0.25f, 1.0f};
    return (NvFill){UB_BF16, 0.05f, 0};
}

typedef struct {
    unsigned h;
    void* p;
    void* snap;
    uint64_t bytes;
    int weight;
} NvTen;

typedef struct {
    double slice_max_us, slice_mean_us, slice0_us, whole_us, ghz;
    double experts;
    int nblk;
    const char* verify;
    uint64_t digest; /* full-grid outputs: fold of every non-weight tensor digest (old vs new A/B) */
} NvRes;

static int nv_find(const NvBlob* b, unsigned prog, uint16_t op, const char* match, int before) {
    const PlowDevInst* I = b->insts[prog];
    const int n = before >= 0 ? before : (int)b->ph[prog].n_inst;
    if (before >= 0) {
        for (int i = n - 1; i >= 0; i--)
            if (I[i].op == op) return i;
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (I[i].op != op) continue;
        if (!match[0]) return i;
        for (int s = 0; s < 8; s++)
            if (I[i].t[s] != PLOW_TENSOR_NONE && I[i].t[s] < b->h.n_tensor && has(b->decl[I[i].t[s]].name, match))
                return i;
    }
    return -1;
}

/* exec: grid = blocks (one slice per block) or 1 (slices [s0, s0 + ns)); returns block-0 cycles. */
static double nv_exec(Nv* nv, void* d_insts, void* d_T, unsigned grid, unsigned s0, unsigned ns,
                      unsigned dyn, double* ns_out) {
    struct __attribute__((packed)) {
        void* insts; void* T; void* stamp; uint32_t n_inst, slice0, nslice, nblk;
    } a = {d_insts, d_T, nv->d_stamp, 1, s0, ns, 0};
    if (nv_run(nv, &nv->exec, grid, NV_THREADS, dyn, &a, sizeof(a))) return -1.0;
    unsigned long long st[4];
    plow_hsa_download(nv->h, 0, st, nv->d_stamp, sizeof(st));
    if (ns_out) *ns_out = (double)(st[3] - st[2]);
    return (double)(st[1] - st[0]);
}

static int nv_case(Nv* nv, const NvBlob* b, const NvCase* c, int kv_fp8, unsigned reps, double ghz_ref,
                   NvRes* r, FILE* jf, int* first) {
    memset(r, 0, sizeof(*r));
    unsigned prog = ~0u;
    for (unsigned i = 0; i < b->n_prog; i++)
        if (b->ph[i].t == c->t) { prog = i; break; }
    if (prog == ~0u) { printf("| %s | T=%u | no such program |\n", c->name, c->t); return 0; }
    const int idx = nv_find(b, prog, c->op, c->match, -1);
    if (idx < 0) { printf("| %s | T=%u | op %u (%s) not in program |\n", c->name, c->t, c->op, c->match); return 0; }
    int chain[6], nchain = 0;
    for (int k = 0; k < 5 && c->setup[k]; k++) {
        const int j = nv_find(b, prog, c->setup[k], "", idx);
        if (j < 0) { printf("| %s | setup op %u missing |\n", c->name, c->setup[k]); return 0; }
        chain[nchain++] = j;
    }
    chain[nchain++] = idx;
    const unsigned n_inst = b->ph[prog].n_inst;
    PlowDevInst* insts = (PlowDevInst*)malloc((size_t)n_inst * sizeof(PlowDevInst));
    memcpy(insts, b->insts[prog], (size_t)n_inst * sizeof(PlowDevInst));
    if (c->flags & CASE_PLAIN) {
        PlowDevInst* in = &insts[idx];
        if (in->op == PLOW_DOP_MAMBA2_STEP) in->i[6] &= 0xFFFFu;
        if (in->op == PLOW_DOP_MAMBA2_CONV) in->t[6] = PLOW_TENSOR_NONE;
        if (in->op == PLOW_DOP_QWEN_GDN_STEP) in->fj[2].u = 0;
    }
    const PlowDevInst* tin = &insts[idx];
    const unsigned nblk = tin->blocks ? tin->blocks : nv->n_sm;
    r->nblk = (int)nblk;

    /* tensors: every handle the chain names */
    NvTen ten[64];
    int nt = 0;
    uint64_t total = 0;
    for (int ci = 0; ci < nchain; ci++) {
        const PlowDevInst* in = &insts[chain[ci]];
        unsigned hs[24];
        int nh = 0;
        for (int s = 0; s < 8; s++) hs[nh++] = in->t[s];
        nh += nv_iword_handles(in, hs + nh);
        /* The fused Mamba scan (CHUNK_STATE) runs CHUNK_PASS's recurrence on the next record's state. */
        if (in->op == PLOW_DOP_MAMBA2_CHUNK_STATE && (unsigned)chain[ci] + 1 < n_inst &&
            insts[chain[ci] + 1].op == PLOW_DOP_MAMBA2_CHUNK_PASS)
            for (int s = 0; s < 8; s++) hs[nh++] = insts[chain[ci] + 1].t[s];
        for (int k = 0; k < nh; k++) {
            const unsigned hd = hs[k];
            if (hd == PLOW_TENSOR_NONE || hd >= b->h.n_tensor) continue;
            int dup = 0;
            for (int q = 0; q < nt; q++) dup |= ten[q].h == hd;
            if (dup || nt >= 64) continue;
            const PlowTensorDecl* d = &b->decl[hd];
            NvTen* t = &ten[nt++];
            memset(t, 0, sizeof(*t));
            t->h = hd;
            t->bytes = d->bytes ? d->bytes : 16;
            t->weight = !(starts(d->name, "act.") || starts(d->name, "in.") || starts(d->name, "state.") ||
                          starts(d->name, "kv."));
            /* Runtime inputs are re-declared per launch width by the loader; a body may index them
             * by row past the packet's declared size (in.prefill_active), so pad them. */
            if (starts(d->name, "in.") && t->bytes < (1u << 20)) t->bytes = 1u << 20;
            t->p = plow_hsa_alloc(nv->h, 0, t->bytes);
            total += t->bytes;
            if (!t->p) { fprintf(stderr, "%s: alloc %s %llu B: %s\n", c->name, d->name,
                                 (unsigned long long)t->bytes, plow_hsa_last_error()); return -1; }
            const int slot = k < 8 ? k : -1;
            NvFill f = nv_fill_rule(d->name, in->op, slot, c, c->t, kv_fp8);
            if (d->init_off != PLOW_INIT_NONE && d->init_off + d->bytes <= b->h.init_bytes &&
                !has(d->name, "active"))
                plow_hsa_upload(nv->h, 0, t->p, b->init + d->init_off, d->bytes);
            else
                nv_fill(nv, t->p, t->bytes, f.kind, 0x1000u + hd, f.a, f.b);
            if (has(d->name, "in.spec.replay")) {
                /* recurrent_state::Replay: [0] parity, [1] slots, [2 + slot] entry (0 = nothing to replay) */
                const int32_t head[2] = {0, (int32_t)(d->bytes / 4 - 2)};
                plow_hsa_upload(nv->h, 0, t->p, head, sizeof(head));
            }
        }
    }
    void** T = (void**)calloc(b->h.n_tensor, sizeof(void*));
    for (int q = 0; q < nt; q++) T[ten[q].h] = ten[q].p;
    void* d_T = plow_hsa_alloc(nv->h, 0, (size_t)b->h.n_tensor * sizeof(void*));
    void* d_insts = plow_hsa_alloc(nv->h, 0, (size_t)n_inst * sizeof(PlowDevInst));
    plow_hsa_upload(nv->h, 0, d_T, T, (size_t)b->h.n_tensor * sizeof(void*));
    plow_hsa_upload(nv->h, 0, d_insts, insts, (size_t)n_inst * sizeof(PlowDevInst));
    const unsigned dyn = NV_SMEM_OPTIN - nv->exec.group_segment_size;
    char* const di = (char*)d_insts;

    /* producers, on the full grid */
    for (int ci = 0; ci + 1 < nchain; ci++) {
        nv_phase = "setup";
        alarm(60);
        const unsigned g = insts[chain[ci]].blocks ? insts[chain[ci]].blocks : nv->n_sm;
        if (nv_exec(nv, di + (size_t)chain[ci] * sizeof(PlowDevInst), d_T, g, 0, 1, dyn, NULL) < 0) return -1;
    }
    /* touched experts of a routed MoE walk: the table is {u32 expert, f32 gate} x k per row */
    if ((c->op == PLOW_DOP_MOE_UP_NVFP4 || c->op == PLOW_DOP_MOE_DOWN_NVFP4) && tin->t[2] != PLOW_TENSOR_NONE) {
        const unsigned rows = tin->i[0], k = tin->i[1], E = tin->i[4];
        uint32_t* tab = (uint32_t*)malloc((size_t)rows * k * 8);
        unsigned char* seen = (unsigned char*)calloc(E ? E : 1, 1);
        for (int q = 0; q < nt; q++)
            if (ten[q].h == tin->t[2]) plow_hsa_download(nv->h, 0, tab, ten[q].p, (size_t)rows * k * 8);
        for (unsigned i = 0; i < rows * k; i++)
            if (tab[2 * i] < E && !seen[tab[2 * i]]) { seen[tab[2 * i]] = 1; r->experts += 1; }
        free(tab);
        free(seen);
    }

    /* full grid vs one SM (all slices serially), both from the same snapshot of every
     * non-weight tensor; a grid rerun separates nondeterministic ops from mismatches. */
    nv_phase = "verify";
    alarm(120);
    uint64_t snap_bytes = 0;
    for (int q = 0; q < nt; q++)
        if (!ten[q].weight && ten[q].bytes <= (3ull << 30)) {
            ten[q].snap = plow_hsa_alloc(nv->h, 0, ten[q].bytes);
            if (ten[q].snap) { nv_copy(nv, ten[q].snap, ten[q].p, ten[q].bytes); snap_bytes += ten[q].bytes; }
        }
    uint64_t dg[3][64];
    for (int run = 0; run < 3; run++) {
        for (int q = 0; q < nt; q++)
            if (ten[q].snap) nv_copy(nv, ten[q].p, ten[q].snap, ten[q].bytes);
        if (run < 2) nv_exec(nv, di + (size_t)idx * sizeof(PlowDevInst), d_T, nblk, 0, 1, dyn, NULL);
        else nv_exec(nv, di + (size_t)idx * sizeof(PlowDevInst), d_T, 1, 0, nblk, dyn, NULL);
        for (int q = 0; q < nt; q++) dg[run][q] = ten[q].snap ? nv_digest(nv, ten[q].p, ten[q].bytes) : 0;
    }
    int same = 1, grid_same = 1;
    for (int q = 0; q < nt; q++) {
        same &= dg[2][q] == dg[0][q];
        grid_same &= dg[1][q] == dg[0][q];
    }
    r->verify = same ? "bitwise" : grid_same ? "MISMATCH" : "nondet";
    r->digest = 0;
    for (int q = 0; q < nt; q++) r->digest = (r->digest ^ dg[0][q]) * 0x100000001b3ull;
    for (int q = 0; q < nt; q++)
        if (ten[q].snap) nv_copy(nv, ten[q].p, ten[q].snap, ten[q].bytes);

    /* per-slice time, L2 flushed before every launch */
    nv_phase = "time";
    alarm(600);
    double* v = (double*)malloc(sizeof(double) * reps);
    double smax = 0, ssum = 0;
    for (unsigned s = 0; s < nblk; s++) {
        for (unsigned k = 0; k < reps; k++) {
            nv_flush(nv);
            v[k] = nv_exec(nv, di + (size_t)idx * sizeof(PlowDevInst), d_T, 1, s, 1, dyn, NULL);
        }
        const double m = median(v, (int)reps);
        if (s == 0) r->slice0_us = m;
        smax = m > smax ? m : smax;
        ssum += m;
    }
    /* whole op on one SM; its globaltimer span gives this run's clock */
    double ghz = ghz_ref, wsum = 0;
    for (unsigned k = 0; k < 3; k++) {
        double ns = 0;
        nv_flush(nv);
        const double cyc = nv_exec(nv, di + (size_t)idx * sizeof(PlowDevInst), d_T, 1, 0, nblk, dyn, &ns);
        v[k] = cyc;
        if (ns > 2e5) ghz = cyc / ns;
        wsum += cyc;
    }
    (void)wsum;
    r->whole_us = median(v, 3) / (ghz * 1e3);
    r->ghz = ghz;
    r->slice_max_us = smax / (ghz * 1e3);
    r->slice_mean_us = ssum / nblk / (ghz * 1e3);
    r->slice0_us /= ghz * 1e3;
    alarm(0);
    free(v);

    const char* opname = c->name;
    printf("| %-28s | %5u | %4u | %3u | %9.2f | %9.2f | %9.2f | %10.1f | %-8s | %6.0f | %7.0f |\n", opname,
           c->t, idx, nblk, r->slice_max_us, r->slice_mean_us, r->slice0_us, r->whole_us, r->verify,
           r->experts, (double)total / 1048576.0);
    fflush(stdout);
    fprintf(jf, "%s    {\"set\": \"%s\", \"case\": \"%s\", \"program_t\": %u, \"inst\": %d, \"op\": %u, "
                "\"blocks\": %u, \"slice_max_us\": %.4f, \"slice_mean_us\": %.4f, \"slice0_us\": %.4f, "
                "\"whole_us\": %.3f, \"ghz\": %.4f, \"verify\": \"%s\", \"digest\": \"%016llx\", "
                "\"experts\": %.0f, \"ctx\": %u, \"setup\": [",
            *first ? "" : ",\n", c->set, c->name, c->t, idx, c->op, nblk, r->slice_max_us, r->slice_mean_us,
            r->slice0_us, r->whole_us, r->ghz, r->verify, (unsigned long long)r->digest, r->experts,
            c->ctx ? c->ctx : 1024);
    for (int ci = 0; ci + 1 < nchain; ci++) fprintf(jf, "%s%d", ci ? ", " : "", chain[ci]);
    fprintf(jf, "]}");
    fflush(jf);
    *first = 0;

    for (int q = 0; q < nt; q++) {
        plow_hsa_free(nv->h, ten[q].p);
        if (ten[q].snap) plow_hsa_free(nv->h, ten[q].snap);
    }
    plow_hsa_free(nv->h, d_T);
    plow_hsa_free(nv->h, d_insts);
    free(T);
    free(insts);
    (void)snap_bytes;
    return 0;
}

static int nv_main(int argc, char** argv) {
    const char *pkt = NULL, *cubin = NULL, *set = NULL, *only = NULL, *json = NULL, *fp4 = NULL;
    const char* base = "ceiling_sm121a.cubin";
    int ceil = 0, kv_fp8 = -1;
    unsigned reps = 9, nk = 1024;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ceilings")) ceil = 1;
        else if (!strcmp(argv[i], "--fp4-tile") && i + 1 < argc) fp4 = argv[++i];
        else if (!strcmp(argv[i], "--nk") && i + 1 < argc) nk = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pkt") && i + 1 < argc) pkt = argv[++i];
        else if (!strcmp(argv[i], "--cubin") && i + 1 < argc) cubin = argv[++i];
        else if (!strcmp(argv[i], "--base") && i + 1 < argc) base = argv[++i];
        else if (!strcmp(argv[i], "--set") && i + 1 < argc) set = argv[++i];
        else if (!strcmp(argv[i], "--case") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-fp8") && i + 1 < argc) kv_fp8 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--list") && i + 1 < argc) {
            /* distinct case names of a set, one per line (a case name may cover several T) */
            const char* s = argv[++i];
            for (size_t c = 0; c < N_NV_CASES; c++) {
                int seen = 0;
                for (size_t d = 0; d < c; d++)
                    seen |= !strcmp(NV_CASES[d].set, s) && !strcmp(NV_CASES[d].name, NV_CASES[c].name);
                if (!strcmp(NV_CASES[c].set, s) && !seen) printf("%s\n", NV_CASES[c].name);
            }
            return 0;
        } else {
            printf("usage: %s --ceilings [--json out]\n"
                   "       %s --fp4-tile <fp4_tile cubin> [--nk k128 steps, <= 2048] [--reps N] [--json out]\n"
                   "       %s --pkt <assets>/model.pkt --cubin <ubench ops cubin> --set <set> [--case name]\n"
                   "          [--kv-fp8 0|1] [--reps N] [--json out] [--base ceiling cubin]\n"
                   "sets: nemotron-dec nemotron-pf qwen-dec qwen-pf\n", argv[0], argv[0], argv[0]);
            return 2;
        }
    }
    if (!reps) reps = 1;
    signal(SIGALRM, nv_alarm);
    Nv nv;
    memset(&nv, 0, sizeof(nv));
    nv.h = plow_hsa_init();
    if (!nv.h) { fprintf(stderr, "init: %s\n", plow_hsa_last_error()); return 1; }
    char dn[64] = {0};
    uint32_t sms = 0, lds = 0;
    plow_hsa_device_info(nv.h, 0, dn, &sms, &lds);
    nv.n_sm = sms;
    printf("Device: %s | SMs: %u\n", dn, sms);
    if (nv_load(&nv, base)) return 1;
    if (nv_kernel(&nv, "ub_fill", &nv.fill) || nv_kernel(&nv, "ub_flush", &nv.flush) ||
        nv_kernel(&nv, "ub_copy", &nv.copy) || nv_kernel(&nv, "ub_digest", &nv.digest) ||
        nv_kernel(&nv, "ceil_mma", &nv.mma) || nv_kernel(&nv, "ceil_ffma", &nv.ffma) ||
        nv_kernel(&nv, "ceil_stream", &nv.stream) || nv_kernel(&nv, "ceil_cpasync", &nv.cpasync) ||
        nv_kernel(&nv, "ceil_bulk", &nv.bulk) || nv_kernel(&nv, "ceil_smem", &nv.smem))
        return 1;
    nv.d_stamp = (unsigned long long*)plow_hsa_alloc(nv.h, 0, 32 * NV_MAX_STAMP);
    nv.d_sink = plow_hsa_alloc(nv.h, 0, 4096);
    nv.d_acc = (unsigned long long*)plow_hsa_alloc(nv.h, 0, 64);
    nv.flush_bytes = 128ull << 20;
    nv.flush_buf = plow_hsa_alloc(nv.h, 0, nv.flush_bytes);
    if (!nv.d_stamp || !nv.flush_buf) { fprintf(stderr, "alloc: %s\n", plow_hsa_last_error()); return 1; }
    nv_fill(&nv, nv.flush_buf, nv.flush_bytes, UB_ZERO, 0, 0, 0);

    FILE* jf = json ? fopen(json, "w") : NULL;
    if (json && !jf) { fprintf(stderr, "cannot write %s\n", json); return 1; }
    if (ceil) {
        NvCeil sm, gpu;
        alarm(600);
        if (nv_ceilings(&nv, &sm, &gpu)) return 1;
        alarm(0);
        nv_ceil_print(&sm, &gpu, nv.n_sm);
        if (jf) {
            fprintf(jf, "{\n  \"device\": \"%s\", \"n_sm\": %u,\n", dn, nv.n_sm);
            nv_ceil_json(jf, "sm", &sm);
            fprintf(jf, ",\n");
            nv_ceil_json(jf, "gpu", &gpu);
            fprintf(jf, "\n}\n");
            fclose(jf);
        }
        return 0;
    }
    if (fp4) {
        if (!nk || nk > 2048) { fprintf(stderr, "--nk must be 1..2048 (operand buffers)\n"); return 2; }
        if (jf) fprintf(jf, "{\n  \"device\": \"%s\", \"n_sm\": %u,\n", dn, nv.n_sm);
        alarm(600);
        if (nv_fp4_tile(&nv, fp4, nk, reps, jf)) return 1;
        alarm(0);
        if (jf) { fprintf(jf, "}\n"); fclose(jf); }
        return 0;
    }
    if (!pkt || !cubin || !set) { fprintf(stderr, "--pkt, --cubin and --set are required\n"); return 2; }
    if (kv_fp8 < 0) kv_fp8 = starts(set, "nemotron");
    NvBlob blob;
    if (nv_blob(pkt, &blob)) return 1;
    if (nv_load(&nv, cubin) || nv_kernel(&nv, "ubench_exec", &nv.exec)) return 1;
    printf("Packet: %s (%u tensors, %u programs) | set %s | %u reps per slice | KV %s\n", pkt,
           blob.h.n_tensor, blob.n_prog, set, reps, kv_fp8 ? "e4m3" : "bf16");
    printf("exec: %u B static smem, %u B local/thread\n\n", nv.exec.group_segment_size,
           nv.exec.private_segment_size);
    if (jf) fprintf(jf, "{\n  \"device\": \"%s\", \"n_sm\": %u, \"pkt\": \"%s\", \"set\": \"%s\", \"reps\": %u,\n",
                    dn, nv.n_sm, pkt, set, reps);
    double ghz = 0;
    {
        /* SM clock under load for cycles -> us: FFMA chains, ~3 ms */
        struct __attribute__((packed)) { void* st; void* sink; uint32_t iters; float x; } a =
            {nv.d_stamp, nv.d_sink, 8192, 1.0f};
        nv_ceil_time(&nv, &nv.ffma, 1, 1024, 0, &a, sizeof(a), NULL);
        nv_ceil_time(&nv, &nv.ffma, 1, 1024, 0, &a, sizeof(a), &ghz);
        printf("SM clock (FFMA calibration): %.3f GHz\n\n", ghz);
        if (jf) fprintf(jf, "  \"clock_ghz\": %.4f,\n  \"cases\": [\n", ghz);
    }
    printf("| case | T | inst | blk | slice max us | slice mean us | slice 0 us | whole-op 1 SM us | "
           "grid vs 1 SM | experts | MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|\n");
    FILE* sink = jf ? jf : fopen("/dev/null", "w");
    int first = 1;
    for (size_t i = 0; i < N_NV_CASES; i++) {
        const NvCase* c = &NV_CASES[i];
        if (strcmp(c->set, set) || (only && strcmp(c->name, only))) continue;
        NvRes r;
        if (nv_case(&nv, &blob, c, kv_fp8, reps, ghz, &r, sink, &first)) return 1;
    }
    if (jf) { fprintf(jf, "\n  ]\n}\n"); fclose(jf); }
    plow_hsa_shutdown(nv.h);
    return 0;
}
#endif /* PLOW_CUDA */

/* ============================================================================
 * Main
 * ============================================================================ */
int main(int argc, char** argv) {
#ifdef PLOW_CUDA
    return nv_main(argc, argv);
#endif
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
