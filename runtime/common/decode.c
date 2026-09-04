#include "decode.h"
#include <string.h>

size_t plow_body_size(uint16_t opcode) {
    switch (plow_op_family(opcode)) {
        case PLOW_FAMILY_DMA:    return sizeof(PlowDmaBody);
        case PLOW_FAMILY_RDMA:   return sizeof(PlowRdmaBody);
        case PLOW_FAMILY_GEMM:   return sizeof(PlowGemmBody);
        case PLOW_FAMILY_FLASH:  return sizeof(PlowFlashBody);
        case PLOW_FAMILY_ROW:    return sizeof(PlowRowBody);
        case PLOW_FAMILY_LAYOUT: return sizeof(PlowLayoutBody);
        default:                 return 0; /* control / host / unknown */
    }
}

static uint32_t rd_u32(const uint8_t* b, size_t at) {
    uint32_t v;
    memcpy(&v, b + at, sizeof(v));
    return v;
}
static uint16_t rd_u16(const uint8_t* b, size_t at) {
    uint16_t v;
    memcpy(&v, b + at, sizeof(v));
    return v;
}

int plow_decode(const uint8_t* buf, size_t len,
                PlowInst* insts, uint32_t max_insts, uint32_t* n_insts,
                const PlowCounter** counters, uint32_t* n_counters,
                uint16_t* bucket_id, uint16_t* plan_gen, uint16_t* flags) {
    if (len < PLOW_STREAM_HEADER_SIZE) return -1;
    if (rd_u32(buf, 0) != PLOW_MAGIC) return -1;
    uint16_t version = rd_u16(buf, 4);
    /* C executors alias body pointers directly into the packet, so they cannot
     * normalize legacy row/Flash layouts. Reject them instead of misparsing. */
    if (version != PLOW_VERSION) return -1;

    uint16_t bucket = rd_u16(buf, 6);
    uint32_t ni = rd_u32(buf, 8);
    uint32_t nc = rd_u32(buf, 12);
    uint16_t gen = rd_u16(buf, 16);
    uint16_t fl = rd_u16(buf, 18);
    if (ni > max_insts) return -2;

    /* v2: 8-byte header with u8 wait_len/succ_len. v3: 12-byte header with u16. */
    size_t hdr_size = (version >= 3) ? sizeof(PlowHeader) : 8;

    size_t i = PLOW_STREAM_HEADER_SIZE;
    for (uint32_t r = 0; r < ni; r++) {
        if (i + hdr_size > len) return -1;

        uint16_t opcode;
        uint8_t resource, unit;
        uint16_t index;
        uint16_t wait_len, succ_len;

        if (version >= 3) {
            PlowHeader h;
            memcpy(&h, buf + i, sizeof(h));
            opcode = h.opcode;
            resource = h.resource;
            unit = h.unit;
            index = h.index;
            wait_len = h.wait_len;
            succ_len = h.succ_len;
        } else {
            /* v2 layout: opcode:u16, resource:u8, unit:u8, index:u16, wait_len:u8, succ_len:u8 */
            opcode = rd_u16(buf, i);
            resource = buf[i + 2];
            unit = buf[i + 3];
            index = rd_u16(buf, i + 4);
            wait_len = buf[i + 6];
            succ_len = buf[i + 7];
        }
        i += hdr_size;

        size_t bsz = plow_body_size(opcode);
        if (i + bsz > len) return -1;
        const void* body = bsz ? (const void*)(buf + i) : NULL;
        i += bsz;

        if (i + (size_t)wait_len * 4 + (size_t)succ_len * 4 > len) return -1;
        const uint32_t* wait = (const uint32_t*)(buf + i);
        i += (size_t)wait_len * 4;
        const uint32_t* succ = (const uint32_t*)(buf + i);
        i += (size_t)succ_len * 4;

        PlowInst* out = &insts[r];
        out->opcode = opcode;
        out->resource = resource;
        out->unit = unit;
        out->index = index;
        out->body = body;
        out->wait = wait_len ? wait : NULL;
        out->succ = succ_len ? succ : NULL;
        out->wait_len = wait_len;
        out->succ_len = succ_len;
    }

    if (i + (size_t)nc * sizeof(PlowCounter) > len) return -1;
    if (counters) *counters = (const PlowCounter*)(buf + i);
    if (n_counters) *n_counters = nc;
    if (n_insts) *n_insts = ni;
    if (bucket_id) *bucket_id = bucket;
    if (plan_gen) *plan_gen = gen;
    if (flags) *flags = fl;
    return 0;
}
