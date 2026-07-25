/* bench_cu.h — Backend-agnostic single-SM/CU kernel benchmarker interface.
 *
 * The host driver (bench_cu_roofline.c) operates through this abstraction:
 *   - AMD: loads code object via HSA, launches with gridDim=1
 *   - NVIDIA: loads cubin via CUDA driver API, launches with gridDim=1
 *
 * Both pin execution to a single compute unit (CU on AMD, SM on NVIDIA) by
 * launching exactly one workgroup/thread-block. The profiler tool differs:
 *   - AMD: rocprof --counters
 *   - NVIDIA: ncu --set full
 *
 * The roofline analysis is identical: compare achieved metrics against the
 * per-SM/CU theoretical ceiling from the hardware spec.
 */
#ifndef PLOW_BENCH_CU_H
#define PLOW_BENCH_CU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Hardware spec (per single SM/CU)
 * ============================================================================ */
typedef struct {
    const char* name;           /* e.g. "MI350X", "RTX6000", "H100" */
    const char* arch;           /* e.g. "gfx950", "sm_120a", "sm_90a" */
    unsigned    cu_count;       /* total SMs/CUs on device */
    unsigned    warp_width;     /* 64 (AMD) or 32 (NVIDIA) */
    unsigned    max_threads;    /* max threads per SM/CU */
    uint64_t    shared_mem;     /* bytes LDS/SMEM per SM/CU */
    uint64_t    regs_32bit;     /* register file entries */
    double      clock_ghz;     /* boost clock in GHz */
    /* Compute ceilings (per single SM/CU) */
    double      peak_tflops_bf16;  /* bf16 TFLOPS on one SM/CU */
    double      peak_tflops_fp32;  /* fp32 TFLOPS on one SM/CU */
    /* Memory ceilings (per single SM/CU share) */
    double      hbm_bw_gbps;      /* GB/s — total HBM BW / CU count */
    double      l2_bw_gbps;       /* L2 partition BW / CUs_per_partition */
    /* Derived */
    double      ridge_flops_per_byte; /* peak_tflops / hbm_bw => ridge point */
} BenchSpec;

/* Pre-built specs. */
extern const BenchSpec BENCH_SPEC_MI350X;
extern const BenchSpec BENCH_SPEC_MI300X;
extern const BenchSpec BENCH_SPEC_H100;
extern const BenchSpec BENCH_SPEC_RTX6000;

/* Auto-detect from device. Returns NULL if unknown. */
const BenchSpec* bench_spec_detect(void);

/* ============================================================================
 * Kernel descriptors — one per benchmarkable op
 * ============================================================================ */
typedef enum {
    BENCH_OP_GEMM = 0,
    BENCH_OP_RMSNORM,
    BENCH_OP_HEADNORM_ROPE,
    BENCH_OP_RESIDUAL,
    BENCH_OP_GEGLU,
    BENCH_OP_EMBED,
    BENCH_OP_SOFTCAP,
    BENCH_OP_INTERP_OVERHEAD,
    BENCH_OP__COUNT
} BenchOp;

static const char* const BENCH_OP_NAMES[] = {
    "d_gemm", "d_rmsnorm", "d_headnorm_rope", "d_residual",
    "d_geglu", "d_embed", "d_softcap", "interp_overhead"
};

/* Problem size parameters for a benchmark run. */
typedef struct {
    BenchOp   op;
    unsigned  iters;       /* iterations inside kernel loop */
    /* GEMM: M, N, K */
    unsigned  M, N, K;
    /* Row ops: rows, feat / ntok, nhead, hd */
    unsigned  rows, feat;
    unsigned  ntok, nhead, hd;
    /* Elementwise: n (total elements) */
    unsigned  n;
    /* Misc */
    float     eps;
    float     scale;
    unsigned  hidden;      /* embed hidden dim */
    unsigned  vocab;       /* embed vocab size */
} BenchParams;

/* ============================================================================
 * Profiler results — parsed from rocprof CSV / ncu output
 * ============================================================================ */
typedef struct {
    /* Pass 1: ALU/MFMA */
    uint64_t sq_insts_valu;
    uint64_t sq_insts_vmem;
    uint64_t sq_insts_lds;
    uint64_t sq_insts_mfma;
    uint64_t sq_waves;
    uint64_t sq_busy_cycles;
    /* Pass 2: Memory */
    uint64_t tcp_total_read;
    uint64_t tcp_total_write;
    uint64_t tcc_hit;
    uint64_t tcc_miss;
    uint64_t tcc_ea_rdreq;
    uint64_t tcc_ea_wrreq;
    /* Pass 3: LDS / occupancy */
    uint64_t sq_lds_bank_conflict;
    uint64_t sq_wait_inst_lds;
    uint64_t sq_active_inst_valu;
    uint64_t sq_active_inst_vmem;
    /* Timing */
    double   elapsed_ns;
} BenchCounters;

/* ============================================================================
 * Roofline analysis results
 * ============================================================================ */
typedef struct {
    BenchOp op;
    const char* name;
    /* Compute */
    double   achieved_tflops;
    double   peak_tflops;
    double   compute_efficiency_pct;
    /* Memory */
    double   achieved_bw_gbps;
    double   peak_bw_gbps;
    double   memory_efficiency_pct;
    /* Arithmetic intensity */
    double   arith_intensity;    /* FLOPs / byte transferred */
    double   ridge_point;        /* spec ridge point */
    int      is_compute_bound;   /* 1 if AI > ridge, else memory-bound */
    /* Micro-arch */
    double   mfma_utilization_pct;  /* MFMA slots used / available */
    double   lds_conflict_rate;     /* conflicts / LDS instructions */
    double   l2_hit_rate;           /* hits / (hits + misses) */
    double   valu_active_pct;       /* VALU active / busy cycles */
} BenchRoofline;

/* Compute roofline analysis from raw counters. */
BenchRoofline bench_analyze(const BenchSpec* spec, const BenchParams* params,
                            const BenchCounters* ctrs);

/* ============================================================================
 * Output
 * ============================================================================ */

/* Print the roofline table to stdout. */
void bench_print_table(const BenchSpec* spec, const BenchRoofline* results, int n);

/* Print as JSON to a file. */
void bench_print_json(const char* path, const BenchSpec* spec,
                      const BenchRoofline* results, int n);

/* ============================================================================
 * Backend interface — implemented per-vendor
 * ============================================================================ */
typedef struct BenchBackend BenchBackend;

/* Initialize backend (loads code object / cubin). */
BenchBackend* bench_backend_init(const char* code_object_path);

/* Launch one benchmark kernel (gridDim=1, blockDim=warp_width*4). */
int bench_backend_launch(BenchBackend* be, const BenchParams* params);

/* Wait for completion. */
int bench_backend_sync(BenchBackend* be);

/* Cleanup. */
void bench_backend_destroy(BenchBackend* be);

#ifdef __cplusplus
}
#endif

#endif /* PLOW_BENCH_CU_H */
