/* packet.h — C view of the Infervisor runtime packet ABI.
 *
 * Byte-for-byte mirror of the wire structs in `crates/packet/src/lib.rs`. The
 * persistent-kernel interpreter `reinterpret_cast`s the stream directly into
 * these records. Every struct is `#[repr(C)]` on the Rust side with explicit
 * padding; the `_Static_assert`s below lock the sizes to the Rust contract
 * (see the `record_layout_is_c_compatible` test in lib.rs). A mismatch is a
 * compile error.
 *
 * No dependencies beyond <stdint.h>. Safe to include from C, C++, CUDA, HIP.
 */
#ifndef PLOW_PACKET_H
#define PLOW_PACKET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define PLOW_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define PLOW_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif

/* --- Opcode (u16, structured [backend:4][family:4][variant:8]) ------------ */

#define PLOW_OP(backend, family, variant) \
    ((uint16_t)((((backend) & 0xF) << 12) | (((family) & 0xF) << 8) | ((variant) & 0xFF)))

static inline uint8_t plow_op_backend(uint16_t op) { return (uint8_t)(op >> 12); }
static inline uint8_t plow_op_family(uint16_t op)  { return (uint8_t)((op >> 8) & 0xF); }
static inline uint8_t plow_op_variant(uint16_t op) { return (uint8_t)(op & 0xFF); }

/* Well-known opcodes (Generic backend). */
enum {
    PLOW_NOP           = 0x0000,
    PLOW_HOST_COORD    = 0x0001,
    PLOW_TMA_LOAD      = 0x0100,
    PLOW_TMA_STORE     = 0x0101,
    PLOW_RDMA          = 0x0200,
    PLOW_GEMM          = 0x0300,
    PLOW_FLASH         = 0x0400,
    PLOW_ROW_REDUCE    = 0x0500,
    PLOW_ROW_POINTWISE = 0x0501,
    PLOW_LAYOUT        = 0x0600,
};

/* Families (bits [11:8]) — selects which body struct to cast. */
enum {
    PLOW_FAMILY_CONTROL = 0,
    PLOW_FAMILY_DMA     = 1,
    PLOW_FAMILY_RDMA    = 2,
    PLOW_FAMILY_GEMM    = 3,
    PLOW_FAMILY_FLASH   = 4,
    PLOW_FAMILY_ROW     = 5,
    PLOW_FAMILY_LAYOUT  = 6,
};

/* Backends (bits [15:12]). */
enum {
    PLOW_BACKEND_GENERIC = 0,
    PLOW_BACKEND_CUDA    = 1,
    PLOW_BACKEND_ROCM    = 2,
    PLOW_BACKEND_CPU     = 3,
};

/* Variant convention (bits [7:0]). Mirrors the `VARIANT_*` consts in
 * crates/packet/src/lib.rs — keep in sync. Low nibble = dtype ladder (bf16 = +1);
 * high nibble groups a structural fusion class so the byte never exhausts. */
enum {
    PLOW_VARIANT_GOLDEN  = 0x00, /* naive single-thread reference */
    PLOW_VARIANT_BF16    = 0x01, /* fast.cu / ThunderKittens */
    PLOW_VARIANT_FP8     = 0x02, /* DeepGEMM */
    PLOW_VARIANT_W4A8    = 0x03, /* LiquidGEMM */
    PLOW_VARIANT_GROUPED = 0x10, /* DeepGEMM grouped / MoE */

    /* GEMM (family 3) — Gemma 4 fused */
    PLOW_VARIANT_BF16_SPLITK        = 0x09, /* plain bf16, partial-K accumulate */
    PLOW_VARIANT_NORM_BF16          = 0x21, /* RMSNorm-prologue -> bf16 GEMM */
    PLOW_VARIANT_NORM_SPLITK_BF16   = 0x29, /* norm-prologue bf16, partial-K */

    /* Flash (family 4) */
    PLOW_VARIANT_FLASH_CAUSAL_BF16  = 0x01, /* full causal, FA-2 tiling */
    PLOW_VARIANT_FLASH_SLIDING_BF16 = 0x02, /* sliding-window causal mask */
    PLOW_VARIANT_FLASH_DECODE_BF16  = 0x03, /* single-query, split-KV + merge */

    /* Row (family 5) — variant_is_reduce() must agree (reduce: 0x00, 0x04) */
    PLOW_VARIANT_ROW_RMS_BF16          = 0x04, /* bf16 RMSNorm (reduce) */
    PLOW_VARIANT_ROW_RESIDUAL_ADD_BF16 = 0x06, /* ew add, 2 operands */
    PLOW_VARIANT_ROW_SWIGLU_BF16       = 0x07, /* silu(gate)*up, 2 operands */
    PLOW_VARIANT_ROW_NORMROPE_BF16     = 0x40, /* RMSNorm->RoPE (K path) */
    PLOW_VARIANT_ROW_NORMROPESCALE_BF16 = 0x41,/* RMSNorm->RoPE->scale (Q path) */

    /* Layout (family 6) */
    PLOW_VARIANT_LAYOUT_COPY_BF16       = 0x01, /* vectorized strided copy */
    PLOW_VARIANT_LAYOUT_GATHER_SCALE_BF16 = 0x42, /* gather rows by id + scale */
};

/* Whether a Row-family variant is reduce-shaped (RMSNorm/LayerNorm/softmax) vs
 * pointwise/fused. Mirrors Opcode::variant_is_reduce in crates/packet/src/lib.rs. */
static inline int plow_variant_is_reduce(uint8_t v) {
    return v == PLOW_VARIANT_GOLDEN || v == PLOW_VARIANT_ROW_RMS_BF16;
}

/* Resource class a record runs on (matches ResourceKind, #[repr(u8)]). */
enum {
    PLOW_RES_SM   = 0,
    PLOW_RES_DMA  = 1,
    PLOW_RES_DPU  = 2,
    PLOW_RES_HOST = 3,
};

/* Sentinels. */
#define PLOW_SLOT_NONE   ((uint16_t)0xFFFF)
#define PLOW_TENSOR_NONE ((uint32_t)0xFFFFFFFF)

/* --- C-castable record structs (largest-first + explicit pad) ------------- */

typedef struct {
    uint16_t opcode;
    uint8_t  resource;
    uint8_t  unit;
    uint16_t index;
    uint16_t wait_len;
    uint16_t succ_len;
    uint16_t _pad;
} PlowHeader;

typedef struct {
    uint32_t bytes;
    uint32_t tensor;
    uint16_t slot;
    uint8_t  kind;   /* PLOW_KIND_* (BufKind); PLOW_KIND_UNSPECIFIED if absent */
    uint8_t  access; /* PLOW_ACCESS_* (a load reads, a store writes) */
} PlowDmaBody;

typedef struct {
    uint32_t bytes;
    uint8_t  src_unit;
    uint8_t  dst_unit;
    uint16_t _pad;
} PlowRdmaBody;

typedef struct {
    uint32_t coord0;
    uint32_t coord1;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint16_t bm;
    uint16_t bn;
    uint16_t bk;
    uint16_t out;
    uint16_t tmem;
    uint16_t _pad;
} PlowGemmBody;

typedef struct {
    uint32_t coord0;
    uint32_t coord1;
    uint32_t seq_q;
    uint32_t seq_kv;
    uint16_t head_dim;
    uint16_t bq;
    uint16_t bkv;
    uint16_t heads;
    uint16_t out;
    uint16_t tmem;
} PlowFlashBody;

typedef struct {
    uint32_t coord;
    uint32_t rows;
    uint32_t feat;
    uint16_t br;
    uint16_t out;
    uint8_t  operands;
    uint8_t  _pad[3];
} PlowRowBody;

/* Max tensor rank a LAYOUT descriptor addresses (NCDHW + batch). */
#define PLOW_LAYOUT_MAX_RANK 6

/* LAYOUT body (v4): strided block copy
 *   out[out_base + Σ idx_a·out_stride_a] = in[in_base + Σ idx_a·in_stride_a]
 * over `shape`. kind 0 = contiguous copy (fast path); kind 1 = strided
 * gather/scatter (transpose: permuted in_stride; broadcast: in_stride 0; slice:
 * in_base+extents; inner concat: out_stride). Strides/bases are in elements. */
typedef struct {
    uint8_t  kind;
    uint8_t  rank;
    uint8_t  elem_size;
    uint8_t  _pad0;
    uint16_t out;
    uint16_t _pad1;
    uint32_t shape[PLOW_LAYOUT_MAX_RANK];
    uint32_t in_stride[PLOW_LAYOUT_MAX_RANK];
    uint32_t out_stride[PLOW_LAYOUT_MAX_RANK];
    uint32_t in_base;
    uint32_t out_base;
} PlowLayoutBody;

typedef struct {
    uint32_t id;
    uint32_t threshold;
    uint8_t  scope; /* 0 intra-SM, 1 intra-GPU, 2 cross-unit */
    uint8_t  _pad[3];
} PlowCounter;

/* --- ABI lock: these sizes are the contract with crates/packet ------------ */

PLOW_STATIC_ASSERT(sizeof(PlowHeader)     == 12, "PlowHeader size");
PLOW_STATIC_ASSERT(sizeof(PlowDmaBody)    == 12, "PlowDmaBody size");
PLOW_STATIC_ASSERT(sizeof(PlowRdmaBody)   == 8,  "PlowRdmaBody size");
PLOW_STATIC_ASSERT(sizeof(PlowGemmBody)   == 32, "PlowGemmBody size");
PLOW_STATIC_ASSERT(sizeof(PlowFlashBody)  == 28, "PlowFlashBody size");
PLOW_STATIC_ASSERT(sizeof(PlowRowBody)    == 20, "PlowRowBody size");
PLOW_STATIC_ASSERT(sizeof(PlowLayoutBody) == 88, "PlowLayoutBody size");
PLOW_STATIC_ASSERT(sizeof(PlowCounter)    == 12, "PlowCounter size");

/* Stream header is 20 bytes; MAGIC "INVP"; VERSION 4 (v4 = strided LAYOUT body). */
#define PLOW_MAGIC   ((uint32_t)0x494E5650u)
#define PLOW_VERSION ((uint16_t)5)
#define PLOW_STREAM_HEADER_SIZE 20

#ifdef __cplusplus
}
#endif

#endif /* PLOW_PACKET_H */
