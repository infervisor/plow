/* dev_blob.h — the compiler→runtime container format.
 *
 * plowc writes it, the runtime reads it, and NEITHER hand-rolls the offsets. That rule
 * exists because the format was duplicated in Rust and in three C harnesses, and it broke
 * twice in one afternoon: once when the tensor-name field grew (a silent segfault) and once
 * when an init-data offset was added (a silent misparse that ran a stale program against a
 * fresh interpreter and produced confident garbage). Both were invisible until the model
 * spoke nonsense.
 *
 * So the structs live here, `crates/packet/src/devbuild.rs` mirrors them, and
 * `crates/packet/tests/dev_abi.rs` COMPILES THIS HEADER and asserts every size and offset
 * against the Rust view. Change a field and that test fails — which is the point.
 *
 * Layout, in order:
 *
 *   PlowBlobHeader
 *   PlowTensorDecl[n_tensor]     names + sizes; init_off indexes the init section
 *   uint8_t init[init_bytes]     data the COMPILER computed (RoPE tables)
 *   uint32_t kv_row_insts[n_kvrow]
 *   for each of n_prog:
 *       PlowProgHeader
 *       PlowDevInst  [n_inst]
 *       PlowStreamEnt[n_stream]
 *       uint32_t     stream_ofs[n_cu]
 *       uint32_t     stream_len[n_cu]
 *       PlowWait     [n_wait]
 *       uint32_t     succs[n_succ]
 */
#ifndef PLOW_DEV_BLOB_H
#define PLOW_DEV_BLOB_H

#include <string.h>
#include "dev_isa.h"

#define PLOW_BLOB_MAGIC "PLOWDEV\x07" /* 64-byte PlowDevInst wire format (\x05 was 104B) */
#define PLOW_NAME_LEN 80

/* PlowBlobHeader.flags bit: the blob carries an op-major global-queue packet stream (trailing
 * gq_stream/gq_seg_ofs appendix), so PLOW_GLOBAL_QUEUE=1 can run it. Absent => static-only stream. */
#define PLOW_BLOB_F_GQ 1u

/* A tensor the program addresses by handle. The runtime allocates it and fills the device
 * pointer table; the program only ever sees the handle. */
typedef struct {
    char     name[PLOW_NAME_LEN];
    uint64_t bytes;
    /* Byte offset into the init section, or PLOW_INIT_NONE. Data the compiler already knows
     * — the RoPE tables, whose partial-rotary/NoPE rule must exist in exactly one place. */
    uint64_t init_off;
} PlowTensorDecl;

#define PLOW_INIT_NONE 0xFFFFFFFFFFFFFFFFull

typedef struct {
    char     magic[8];
    uint32_t n_cu;
    uint32_t n_tensor;
    /* Prefill buckets, then the decode program last. All share the tensor table above:
     * prefill fills the KV cache and decode appends to it. */
    uint32_t n_prog;
    /* Instruction indices in the DECODE program whose `i[3]` is the KV-cache write row. The
     * runtime rewrites exactly these each step; everything else it needs (ids, pos, kvlen)
     * is already a tensor. This is the entire dynamic surface of a decode step. */
    uint32_t n_kvrow;
    /* Packet-stream type flags (PLOW_BLOB_F_*). Denotes global-queue vs static-only at the header. */
    uint32_t flags;
    uint32_t _pad;
    uint64_t init_bytes;
    /* Reserved for future metadata. The header is fixed at 64 bytes (one cache line, 8-aligned) so new
     * fields can be carved from this block without moving existing fields or the sections after it. */
    uint64_t reserved[3];
} PlowBlobHeader;

typedef struct {
    uint32_t n_inst;
    uint32_t n_stream;
    uint32_t n_wait;
    uint32_t n_succ;
    uint32_t n_counter;
    /* The T this program was compiled for. The runtime picks the smallest prefill bucket
     * that holds the prompt; the decode program has T = 1. */
    uint32_t t;
} PlowProgHeader;

PLOW_SASSERT(sizeof(PlowTensorDecl) == 96, "PlowTensorDecl size");
PLOW_SASSERT(sizeof(PlowBlobHeader) == 64, "PlowBlobHeader size");
PLOW_SASSERT(sizeof(PlowProgHeader) == 24, "PlowProgHeader size");

/* --- v6 section directory ------------------------------------------------- */

#define PLOW_BLOB_MAGIC_V6 "PLOWDEV\x08" /* sectioned variant of \x07 */
#define PLOW_SECT_MAGIC    "SECT"

#define PLOW_SECT_PROGRAMS      0u
#define PLOW_SECT_CUBIN         1u
#define PLOW_SECT_HSACO         2u
#define PLOW_SECT_WEIGHT_MAP    3u
#define PLOW_SECT_METADATA      4u
#define PLOW_SECT_STATIC_TENSORS 5u
#define PLOW_SECT_GEN_TENSORS   6u

/* --- v7: generated tensors ------------------------------------------------ */

/* v7 = v6 plus a PLOW_SECT_GEN_TENSORS directory. Tensors listed there have
 * init_off == PLOW_INIT_NONE and must be MATERIALISED by the reader from the
 * recipe below, not zero-filled.
 *
 * The RoPE tables are ~403 MB at ctx=131072 and are a pure function of six
 * scalars, so shipping them expanded dominated the blob for no reason.
 *
 * Readers that do not implement this MUST reject the magic. A v6-era reader
 * would see INIT_NONE, zero-fill, and serve a model with cos=sin=0 — fluent
 * output, wrong output, no error. That is why this is a magic bump and not a
 * new optional section. The C harnesses in runtime/tests/ do NOT implement it;
 * compile their blobs with `plowc --no-rope-gen`, which bakes the tables back
 * into the init section and keeps the container at v5/v6. */
#define PLOW_BLOB_MAGIC_V7 "PLOWDEV\x09"

/* PlowGenTensor.kind */
#define PLOW_GEN_ROPE_COS     0u
#define PLOW_GEN_ROPE_SIN     1u
#define PLOW_GEN_ROPE_IDX_COS 2u
#define PLOW_GEN_ROPE_IDX_SIN 3u

/* PlowGenTensor.scale */
#define PLOW_ROPE_SCALE_NONE   0u
#define PLOW_ROPE_SCALE_LLAMA3 1u

/* Mirrors `packet::rope::GenTensor`; locked by crates/packet/tests/dev_abi.rs.
 * Flat union across every kind — slots a kind does not use are zero. */
typedef struct {
    uint32_t tensor;  /* index into the tensor decl table */
    uint32_t kind;    /* PLOW_GEN_* */
    uint32_t ctx;     /* rows */
    uint32_t hd;      /* head_dim; index_dim for the IDX kinds */
    uint32_t aux;     /* rope_hd for the IDX kinds, else 0 */
    uint32_t scale;   /* PLOW_ROPE_SCALE_* */
    double   theta;
    double   frac;    /* partial-rotary fraction; 1.0 = fully rotated */
    double   factor;  /* Llama-3 scaling; all zero when scale == NONE */
    double   low;
    double   high;
    double   orig;
} PlowGenTensor;

PLOW_SASSERT(sizeof(PlowGenTensor) == 72, "PlowGenTensor size");

#define PLOW_SECT_NAME_LEN 24

typedef struct {
    uint32_t kind;
    uint32_t _pad;
    uint64_t offset;
    uint64_t size;
    char     name[PLOW_SECT_NAME_LEN];
} PlowSectionEntry;

PLOW_SASSERT(sizeof(PlowSectionEntry) == 48, "PlowSectionEntry size");

/* Find a section by kind in a v6 blob. Returns 1 if found (out_offset/out_size
 * populated), 0 otherwise. Caller checks magic[7] >= '\x06' before calling. */
static inline int plow_blob_find_section(
    const uint8_t *blob, size_t len, uint32_t kind,
    uint64_t *out_offset, uint64_t *out_size)
{
    const PlowBlobHeader *hdr = (const PlowBlobHeader *)blob;
    uint64_t dir_off = hdr->reserved[0];
    if (dir_off == 0 || dir_off + 8 > len) return 0;
    const uint8_t *dir = blob + dir_off;
    if (memcmp(dir, PLOW_SECT_MAGIC, 4) != 0) return 0;
    uint32_t n = *(const uint32_t *)(dir + 4);
    const PlowSectionEntry *ents = (const PlowSectionEntry *)(dir + 8);
    if (dir_off + 8 + (uint64_t)n * sizeof(PlowSectionEntry) > len) return 0;
    for (uint32_t i = 0; i < n; i++) {
        if (ents[i].kind == kind) {
            *out_offset = ents[i].offset;
            *out_size   = ents[i].size;
            return 1;
        }
    }
    return 0;
}

#endif /* PLOW_DEV_BLOB_H */
