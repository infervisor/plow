/* Infervisor runtime packet ABI v3 — must match `crates/packet/src/lib.rs`.
 *
 * A variable-length stream of tagged #[repr(C)] records the kernel casts directly.
 * Little-endian; every record is 4-byte aligned (Header + bodies are 4-byte
 * multiples), so casting a body or u32 at any record offset is a valid load.
 *
 * Stream header (20 bytes):
 *   u32 magic = 0x494E5650 ("INVP");
 *   u16 version = 3;
 *   u16 bucket_id;              // shape bucket this stream serves
 *   u32 n_insts; u32 n_counters;
 *   u16 plan_gen;               // generation counter for invalidation
 *   u16 flags;                  // bit 0: has_crc; bits 1..15 reserved
 *   <records>                   // n_insts of them
 *   Counter counters[n_counters];
 *
 * Each record:
 *   Header  h;                  // 12 bytes (v3; was 8 in v2)
 *   <Body>  body;              // family-specific (see h.opcode family bits)
 *   uint32_t wait[h.wait_len]; // counter ids to wait on
 *   uint32_t succ[h.succ_len]; // counter ids to increment
 *
 * Walk: `const Header* h = (const Header*)p; p += sizeof(Header);`
 *       uint8_t family = (h->opcode >> 8) & 0xF;
 *       switch (family) { case FAMILY_GEMM: const GemmBody* g = (GemmBody*)p;
 *       p += sizeof(GemmBody); ... } p += (h->wait_len + h->succ_len) * 4;
 *
 * v3 change: wait_len/succ_len widened from u8 to u16 to support fine per-tile
 * counters (removes the 255-counter-per-tile ceiling).
 *
 * Opcode namespace (u16): [backend:4][family:4][variant:8]
 *   Backend: 0=Generic, 1=CUDA, 2=ROCm, 3=CPU
 *   Family:  0=Control, 1=DMA, 2=RDMA, 3=Gemm, 4=Flash, 5=Row, 6=Layout, 7..15=Future
 *   Variant: kernel variant within family (dtype, epilogue, tile-config, etc.)
 */
#ifndef INFERVISOR_PACKET_H
#define INFERVISOR_PACKET_H
#include <stdint.h>

#define INFERVISOR_PACKET_MAGIC   0x494E5650u
#define INFERVISOR_PACKET_VERSION 3
#define PKT_SLOT_NONE             0xFFFFu
#define PKT_TENSOR_NONE           0xFFFFFFFFu

/* Stream header size in bytes */
#define INFERVISOR_STREAM_HEADER_SIZE 20

/* --- Opcode structured namespace ------------------------------------------ */

#define PKT_OPCODE(backend, family, variant) \
    ((uint16_t)(((backend) & 0xF) << 12 | ((family) & 0xF) << 8 | ((variant) & 0xFF)))

#define PKT_OPCODE_BACKEND(op)  (((op) >> 12) & 0xF)
#define PKT_OPCODE_FAMILY(op)   (((op) >> 8) & 0xF)
#define PKT_OPCODE_VARIANT(op)  ((op) & 0xFF)

/* Backends */
#define BACKEND_GENERIC  0
#define BACKEND_CUDA     1
#define BACKEND_ROCM     2
#define BACKEND_CPU      3

/* Families (selects body struct to cast) */
#define FAMILY_CONTROL   0
#define FAMILY_DMA       1
#define FAMILY_RDMA      2
#define FAMILY_GEMM      3
#define FAMILY_FLASH     4
#define FAMILY_ROW       5
#define FAMILY_LAYOUT    6

/* Well-known Generic opcodes */
#define OP_NOP            PKT_OPCODE(0, 0, 0)  /* 0x0000 */
#define OP_HOST_COORD     PKT_OPCODE(0, 0, 1)  /* 0x0001 */
#define OP_TMA_LOAD       PKT_OPCODE(0, 1, 0)  /* 0x0100 */
#define OP_TMA_STORE      PKT_OPCODE(0, 1, 1)  /* 0x0101 */
#define OP_RDMA           PKT_OPCODE(0, 2, 0)  /* 0x0200 */
#define OP_GEMM           PKT_OPCODE(0, 3, 0)  /* 0x0300 */
#define OP_FLASH          PKT_OPCODE(0, 4, 0)  /* 0x0400 */
#define OP_ROW_REDUCE     PKT_OPCODE(0, 5, 0)  /* 0x0500 */
#define OP_ROW_POINTWISE  PKT_OPCODE(0, 5, 1)  /* 0x0501 */
#define OP_LAYOUT         PKT_OPCODE(0, 6, 0)  /* 0x0600 */

/* Resource kinds */
enum ResourceKind { RES_SM = 0, RES_DMA = 1, RES_DPU = 2, RES_HOST = 3 };

/* --- Record structs ------------------------------------------------------- */

typedef struct { uint16_t opcode; uint8_t resource, unit; uint16_t index; uint16_t wait_len, succ_len, _pad; } Header; /* 12 */
typedef struct { uint32_t bytes, tensor; uint16_t slot, _pad; } DmaBody;                                              /* 12 */
typedef struct { uint32_t bytes; uint8_t src_unit, dst_unit; uint16_t _pad; } RdmaBody;                               /* 8 */
typedef struct { uint32_t coord0, coord1, m, n, k; uint16_t bm, bn, bk, out, tmem, _pad; } GemmBody;                  /* 32 */
typedef struct { uint32_t coord0, coord1, seq_q, seq_kv; uint16_t head_dim, bq, bkv, heads, out, tmem; } FlashBody;   /* 28 */
typedef struct { uint32_t coord, rows, feat; uint16_t br, out; uint8_t operands, _pad[3]; } RowBody;                  /* 20 */
typedef struct { uint32_t coord; } LayoutBody;                                                                        /* 4 */
typedef struct { uint32_t id, threshold; uint8_t scope, _pad[3]; } Counter;                                           /* 12 */

#endif /* INFERVISOR_PACKET_H */
