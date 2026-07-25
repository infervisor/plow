/* memmap.h — runtime rebase of the compiler's HBM address map.
 *
 * The compiler emits an address map (`*.map.json`): every logical buffer slot →
 * a byte offset within a single arena. At load time the runtime allocates the
 * arena once and resolves each slot to `arena_base + offset`, filling the
 * `kctx.tensors[]` pointer table the DMA records index by their `tensor` slot id.
 * This is the host-side counterpart of the indirection table / SETUP_INDIRECTION
 * (gpu_hypervisor_architecture.md §7.2): the cached packet stream is address-free;
 * only this map and the chosen `arena_base` bind it to physical memory.
 *
 * The JSON is parsed by the host driver into `PlowMemEntry[]`; this header is the
 * binary contract and the pointer-resolution step (no JSON dependency here).
 */
#ifndef PLOW_MEMMAP_H
#define PLOW_MEMMAP_H

#include <stddef.h>
#include <stdint.h>
#include "packet.h" /* PLOW_LAYOUT_MAX_RANK */

#ifdef __cplusplus
extern "C" {
#endif

/* Lifetime class (mirrors plow_asset::BufClass declaration order). */
enum {
    PLOW_BUF_PERSISTENT = 0, /* weights: stable for the whole program */
    PLOW_BUF_STATIC     = 1, /* compile-time consts (RoPE tables, masks) */
    PLOW_BUF_GROWABLE   = 2, /* KV / context: may extend from growable_base */
    PLOW_BUF_SCRATCH    = 3, /* activations: reused by liveness */
    PLOW_BUF_REQUEST_IO = 4, /* per-request input/output */
};

/* Data-type tag (mirrors plow_asset::BufKind). */
enum {
    PLOW_KIND_WEIGHTS     = 0,
    PLOW_KIND_EMBEDDING   = 1,
    PLOW_KIND_CONST       = 2,
    PLOW_KIND_INPUT       = 3,
    PLOW_KIND_OUTPUT      = 4,
    PLOW_KIND_KV_CACHE    = 5,
    PLOW_KIND_ACTIVATION  = 6,
    PLOW_KIND_UNSPECIFIED = 0xFF, /* DmaBody.kind when the emitter can't resolve it */
};

/* Device-kernel access mode (mirrors plow_asset::Access). */
enum {
    PLOW_ACCESS_READ       = 0,
    PLOW_ACCESS_WRITE      = 1,
    PLOW_ACCESS_READ_WRITE = 2,
};

/* One buffer's placement in the arena. */
typedef struct {
    uint32_t slot;     /* logical slot id (DMA record `tensor` field) */
    uint8_t  class_;   /* PLOW_BUF_* */
    uint8_t  growable; /* runtime may extend this region in place */
    uint8_t  _pad[2];
    uint64_t offset;   /* byte offset into the arena */
    uint64_t reserved; /* bytes reserved at offset (size + growth) */
} PlowMemEntry;

/* The whole map for one bucket. */
typedef struct {
    const PlowMemEntry* entries;
    uint32_t            n_entries;
    uint64_t            arena_bytes;   /* total bytes to allocate */
    uint64_t            growable_base; /* where growable buffers start */
} PlowMemMap;

/* Resolve `map` against an allocated `arena_base`: for each entry, set
 * `tensors[entry.slot] = (uint8_t*)arena_base + entry.offset`. `tensors` must
 * hold at least (max slot id + 1) pointers (`n_tensors` is its capacity).
 * Returns the number of slots written, or 0 if any slot id is out of range or
 * its [offset, offset+reserved) range exceeds `arena_bytes`. */
uint32_t plow_memmap_resolve(const PlowMemMap* map, void* arena_base,
                             void** tensors, uint32_t n_tensors);

/* Highest slot id + 1 in the map (the minimum `tensors[]` capacity). */
uint32_t plow_memmap_n_slots(const PlowMemMap* map);

/* --- Operand views (Phase C) ---------------------------------------------- */

/* A consumer operand resolved against the slot table — a base+offset (and, for a
 * strided view, per-axis element strides). This is what lets a slice/reshape be
 * read in place with no copy: the operand points into the producer's buffer.
 * `device` selects which device's segment owns the buffer (0 on single device;
 * multi-device PGAS resolution is a runtime/topology concern layered on top). */
typedef struct {
    uint32_t slot;       /* buffer slot in the address map */
    uint32_t byte_off;   /* byte offset of the view into that buffer */
    uint8_t  rank;       /* 0 ⇒ contiguous; else `stride` is meaningful */
    uint8_t  device;     /* owning device segment (0 = local/single-device) */
    uint8_t  _pad[2];
    uint32_t stride[PLOW_LAYOUT_MAX_RANK]; /* element strides for a strided view */
} PlowOperand;

/* Resolve an operand to a pointer: `tensors[op->slot] + op->byte_off`, or NULL if
 * the slot is out of range. The caller applies `stride`/`rank` when reading a
 * strided view (the contiguous case needs only the returned base). */
void* plow_operand_ptr(const PlowOperand* op, void* const* tensors, uint32_t n_tensors);

#ifdef __cplusplus
}
#endif

#endif /* PLOW_MEMMAP_H */
