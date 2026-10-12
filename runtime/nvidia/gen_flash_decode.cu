// Role wrapper for generated flash-decode bodies (scripts/gen_kernels/build_catalog.py, op
// "flash_decode"). PLOW_GEN_BODY names the generated header: it defines
// plow_gen_call_decode(bx, by, K, ML, O, Opart, Q, V, ctr, kv_len, slot, batch, kv_mask,
// kv_stride, kvh, nsplit, scale, window), one (kv head x split = bx, sequence = by) item of
// one-query-row GQA attention over Q/O [rows][heads][hd] and the KV cache
// [slots][kv heads][kv_stride][hd] (row b reads slot[b]; position p at row p & kv_mask; window 0
// = global), plus the PLOW_GEN_* geometry macros. Splits write f32 partials and the last split
// to arrive merges them, so one launch per site; `ctr[row][kv head]` counts arrivals and is left
// at zero. The launch is a plain grid (kv heads x nsplit, rows), not the packet's persistent
// grid: the runtime builds the argument struct itself (exec/gpu/riders.rs, exec/gpu/gen_role.rs).
#include "dev_isa.h"
#include PLOW_GEN_BODY

extern "C" __device__ unsigned plow_gen_flash_decode_abi = 1;
extern "C" __device__ unsigned plow_gen_block = PLOW_GEN_THREADS;
extern "C" __device__ unsigned plow_gen_arena_bytes = PLOW_GEN_ARENA;
extern "C" __device__ unsigned plow_attention_head_dim = PLOW_GEN_HEAD_DIM;
// Capability tiles: query rows per item = the GQA ratio the body is compiled for.
extern "C" __device__ unsigned plow_attention_query_tile = PLOW_GEN_GQA;
extern "C" __device__ unsigned plow_attention_kv_tile = PLOW_GEN_BN;
extern "C" __device__ unsigned plow_attention_warps = PLOW_GEN_THREADS / 32;
extern "C" __device__ unsigned plow_gen_decode_gqa = PLOW_GEN_GQA;
extern "C" __device__ unsigned plow_gen_decode_nsplit_max = PLOW_GEN_NSPLIT_MAX;

// Mirrored by exec/gpu/riders.rs `GenDecodeArgs`.
typedef struct {
    float* opart;            /* [rows][n_head][nsplit_max][hd] */
    float* mlpart;           /* [rows][n_head][nsplit_max][2] */
    const bfloat16_t* q;     /* [rows][n_head][hd] */
    const bfloat16_t* k;     /* [slots][n_kv_head][kv_stride][hd] */
    const bfloat16_t* v;
    const int* kv_len;       /* [rows] */
    const int* slot;         /* [rows] physical KV slot */
    bfloat16_t* out;         /* [rows][n_head][hd] */
    int* ctr;                /* [rows][n_kv_head], zero between launches */
    unsigned rows, n_head, n_kv_head, kv_stride, window, nsplit, kv_mask, hd;
    float scale;
    unsigned pad;
} PlowGenFlashDecode;
static_assert(sizeof(PlowGenFlashDecode) == 9 * 8 + 10 * 4, "generated flash decode role ABI");

extern "C" __global__ __launch_bounds__(PLOW_GEN_THREADS, 1)
void plow_gen_flash_decode(const __grid_constant__ PlowGenFlashDecode a) {
    if (a.hd != PLOW_GEN_HEAD_DIM || a.n_kv_head == 0 || a.n_head != a.n_kv_head * PLOW_GEN_GQA ||
        a.nsplit == 0 || a.nsplit > PLOW_GEN_NSPLIT_MAX || gridDim.x != a.n_kv_head * a.nsplit ||
        gridDim.y != a.rows) {
        __trap();
        return;
    }
    plow_gen_call_decode((int)blockIdx.x, (int)blockIdx.y, a.k, a.mlpart, a.out, a.opart, a.q,
                         a.v, a.ctr, a.kv_len, a.slot, (int)a.rows, (int)a.kv_mask,
                         (int)a.kv_stride, (int)a.n_kv_head, (int)a.nsplit, a.scale,
                         (int)a.window);
}
