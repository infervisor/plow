/* amx_common.h — the tile driver gemm_amx.c exposes to sibling AMX kernels (fp8.c). */
#ifndef PLOW_CPU_AMX_COMMON_H
#define PLOW_CPU_AMX_COMMON_H

#include "cpu_dev.h"

typedef struct {
    plow_bf16* C;
    const plow_bf16* A;
    const plow_bf16* W;  /* GEMM: bf16 weight; GLU: gate weight (NULL when Wq is set) */
    const plow_bf16* Wu; /* GLU up weight, else NULL */
    const uint8_t* Wq;   /* fp8 e4m3 weight (dequantized into the B strip while packing) */
    const uint8_t* Wuq;  /* fp8 GLU up weight */
    const float* ws;     /* fp8 per-output-channel scale [N], applied in the epilogue */
    const float* us;     /* fp8 GLU up scale [N] */
    const float* rms;    /* GEMM_NORM */
    const plow_bf16* gamma;
    uint32_t M, N, K, BM, BN, act;
} gemm_args;

/* Scratch large enough and K a multiple of 32 (else the caller falls back to golden). */
int plow_amx_usable(const PlowCpuCtx* ctx, uint32_t K);

/* This slice's tiles of (BM, BN) over C[M,N], golden's SWZ=0 linear tile order. */
void plow_amx_run_tiles(const gemm_args* g, uint32_t slice, uint32_t nblk, PlowCpuCtx* ctx);

#endif /* PLOW_CPU_AMX_COMMON_H */
