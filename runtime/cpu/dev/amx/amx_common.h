/* amx_common.h — the tile driver gemm_amx.c exposes to sibling AMX kernels (fp8.c). */
#ifndef PLOW_CPU_AMX_COMMON_H
#define PLOW_CPU_AMX_COMMON_H

#include <immintrin.h>
#include "cpu_dev.h"

/* In-register 16x16 transpose of 32-bit lanes: r[i] <- lane i of every input row. Every VNNI B
 * pack here (weight strip, x tiles) is this transpose: input row n = 16 k-pairs, output row = one
 * k-pair across 16 columns. Fully unrolled so the 32 vectors stay in zmm0-31 (the rolled version
 * round-tripped every stage through a stack array: ~200 memory ops per tile). Stage 1 is
 * shift+blend (p0 / p05) instead of vpunpck{l,h}dq (p5) so the 64 lane moves spread over two
 * ports; stages 2-4 are the usual unpack-qword / shuffle-i32x4 network. The stage-1 variant
 * lands column 4q+1 and 4q+2 swapped, undone by the final store index. */
static inline __attribute__((always_inline)) void plow_amx_tr16x16(__m512i* r) {
    __m512i t[16];
#pragma GCC unroll 8
    for (int i = 0; i < 8; i++) {
        const __m512i a = r[2 * i], b = r[2 * i + 1];
        t[2 * i] = _mm512_mask_blend_epi32(0xAAAA, a, _mm512_slli_epi64(b, 32));
        t[2 * i + 1] = _mm512_mask_blend_epi32(0x5555, b, _mm512_srli_epi64(a, 32));
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; i++) {
        r[4 * i] = _mm512_unpacklo_epi64(t[4 * i], t[4 * i + 2]);
        r[4 * i + 1] = _mm512_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
        r[4 * i + 2] = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
        r[4 * i + 3] = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; i++) {
        t[i] = _mm512_shuffle_i32x4(r[i], r[i + 4], 0x88);
        t[i + 4] = _mm512_shuffle_i32x4(r[i], r[i + 4], 0xdd);
        t[i + 8] = _mm512_shuffle_i32x4(r[i + 8], r[i + 12], 0x88);
        t[i + 12] = _mm512_shuffle_i32x4(r[i + 8], r[i + 12], 0xdd);
    }
    /* r[k] would hold column 4*(k/4) + (0,2,1,3)[k%4]: write it to the natural index. */
#pragma GCC unroll 8
    for (int i = 0; i < 8; i++) {
        const int o = (i & 3) == 1 ? i + 1 : (i & 3) == 2 ? i - 1 : i;
        r[o] = _mm512_shuffle_i32x4(t[i], t[i + 8], 0x88);
        r[o + 8] = _mm512_shuffle_i32x4(t[i], t[i + 8], 0xdd);
    }
}

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
    const plow_bf16* bias; /* t7: bf16[N], added in f32 before the bf16 store (GEMM / GEMM_NORM);
                              GLU: t6 = gate bias */
    const plow_bf16* bias_up; /* GLU t7: up bias */
    uint32_t M, N, K, BM, BN, act;
    float f0, f1;           /* GLU act immediates */
} gemm_args;

/* Scratch large enough and K a multiple of 32 (else the caller falls back to golden). */
int plow_amx_usable(const PlowCpuCtx* ctx, uint32_t K);

/* This slice's tiles of (BM, BN) over C[M,N], golden's SWZ=0 linear tile order. */
void plow_amx_run_tiles(const gemm_args* g, uint32_t slice, uint32_t nblk, PlowCpuCtx* ctx);

#endif /* PLOW_CPU_AMX_COMMON_H */
