/* decode.h — walk a `.pkt` stream (Program::to_bytes) into decoded records.
 *
 * The stream is a 20-byte header, then variable-length 4-aligned records
 * (PlowHeader + body + wait[] + succ[]), then the counter table. This mirrors
 * `Program::decode` in crates/packet but produces zero-copy views (body and
 * counter pointers alias the input buffer).
 */
#ifndef PLOW_DECODE_H
#define PLOW_DECODE_H

#include <stddef.h>
#include <stdint.h>
#include "packet.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One decoded instruction; `body` aliases the family struct in the buffer. */
typedef struct {
    uint16_t        opcode;
    uint8_t         resource;
    uint8_t         unit;
    uint16_t        index;
    const void*     body;
    const uint32_t* wait;
    const uint32_t* succ;
    uint16_t        wait_len;
    uint16_t        succ_len;
} PlowInst;

/* Byte size of a body for an opcode (0 for control/host/unknown). */
size_t plow_body_size(uint16_t opcode);

/* Decode a stream into `insts` (capacity `max_insts`) and locate the counter
 * table. On success returns 0 and fills the out params; returns:
 *   -1 bad magic/version/truncated, -2 too many insts for `max_insts`.
 * `counters`/`n_counters` may be NULL if not needed. */
int plow_decode(const uint8_t* buf, size_t len,
                PlowInst* insts, uint32_t max_insts, uint32_t* n_insts,
                const PlowCounter** counters, uint32_t* n_counters,
                uint16_t* bucket_id, uint16_t* plan_gen, uint16_t* flags);

#ifdef __cplusplus
}
#endif

#endif /* PLOW_DECODE_H */
