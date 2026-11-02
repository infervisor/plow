#pragma once

#include <stddef.h>
#include <stdint.h>

#define GDN_PREFILL_SM120_ABI "plow_gdn_prefill_sm120:v1:hk=16:hv=48:d=128:dtype=bf16:state=f32_vk"

// 64-bit host/device ABI: nine 8-byte pointers, two i32s, one f32, 4 bytes tail
// padding; size 88, alignment 8. BF16 pointers contain raw 16-bit encodings.
// Launch grid (8, 48, nseq), block (128, 1, 1), dynamic shared memory 0.
// Q/K/V/O must be 16-byte aligned; writable buffers must not alias inputs.
// 0 <= tokens <= 8192, 1 <= nseq <= 64; monotone offsets span [0, tokens].
typedef struct GdnPrefillArgs {
    const uint16_t* q;          //  0: [tokens][16][128], normalized, unscaled
    const uint16_t* k;          //  8: [tokens][16][128], normalized
    const uint16_t* v;          // 16: [tokens][48][128]
    uint16_t* o;                // 24: [tokens][48][128]
    const float* alpha;         // 32: [tokens][48], decay
    const float* beta;          // 40: [tokens][48]
    float* state;               // 48: [nseq][48][V=128][K=128]
    const float* initial;       // 56: [nseq][48][V=128][K=128]
    const int64_t* cu_seqlens;  // 64: [nseq+1]
    int32_t tokens;             // 72
    int32_t nseq;               // 76
    float scale;               // 80: 128^-0.5
} GdnPrefillArgs;

#ifdef __cplusplus
static_assert(sizeof(GdnPrefillArgs) == 88 && alignof(GdnPrefillArgs) == 8);
static_assert(offsetof(GdnPrefillArgs, cu_seqlens) == 64);
static_assert(offsetof(GdnPrefillArgs, tokens) == 72);
static_assert(offsetof(GdnPrefillArgs, nseq) == 76);
static_assert(offsetof(GdnPrefillArgs, scale) == 80);
#endif

#ifdef __CUDACC__
extern "C" __global__ void gdn_prefill_sm120(GdnPrefillArgs a);
#endif
