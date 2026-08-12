#include "../../common/control_queue_probe.h"

#include <assert.h>
#include <string.h>

#if defined(__cplusplus)
#define CHECK_OFFSET(type, field, expected) \
    static_assert(offsetof(type, field) == (expected), #type "." #field " ABI offset")
#else
#define CHECK_OFFSET(type, field, expected) \
    _Static_assert(offsetof(type, field) == (expected), #type "." #field " ABI offset")
#endif

CHECK_OFFSET(PlowControlHeaderV1, host_tail_seq, 0x40);
CHECK_OFFSET(PlowControlHeaderV1, dev_head_seq, 0x80);
CHECK_OFFSET(PlowControlHeaderV1, completed_seq, 0x88);
CHECK_OFFSET(PlowControlHeaderV1, stopped_seq, 0x98);
CHECK_OFFSET(PlowControlHeaderV1, fault_seq, 0xc0);
CHECK_OFFSET(PlowControlCommandV1, commit_seq, 0x00);
CHECK_OFFSET(PlowControlCommandV1, opcode, 0x14);
CHECK_OFFSET(PlowControlCommandV1, slot_mask, 0x30);
CHECK_OFFSET(PlowControlCommandV1, input_epoch, 0x40);
CHECK_OFFSET(PlowControlCommandV1, extension_reserved, 0x80);

static PlowControlHeaderV1 valid_header(void) {
    PlowControlHeaderV1 h;
    memset(&h, 0, sizeof(h));
    h.magic = PLOW_CONTROL_MAGIC;
    h.abi_major = PLOW_CONTROL_ABI_MAJOR;
    h.abi_minor = PLOW_CONTROL_ABI_MINOR;
    h.header_bytes = PLOW_CONTROL_HEADER_BYTES;
    h.command_bytes = PLOW_CONTROL_COMMAND_BYTES;
    h.ring_order = 0;
    h.rank = 0;
    h.n_gpu = 1;
    h.required_features = PLOW_CONTROL_FEAT_RUN_DECODE;
    h.supported_features = PLOW_CONTROL_FEAT_RUN_DECODE;
    h.session_nonce = UINT64_C(0x4f93d721c);
    return h;
}

int main(void) {
    PlowControlHeaderV1 h = valid_header();
    PlowControlCommandV1 c;
    memset(&c, 0, sizeof(c));
    c.commit_seq = 1;
    c.session_nonce = h.session_nonce;
    c.record_bytes = PLOW_CONTROL_COMMAND_BYTES;
    c.abi_minor = PLOW_CONTROL_ABI_MINOR;
    c.opcode = PLOW_CONTROL_RUN_DECODE;
    assert(plow_control_header_v1_valid(&h));
    assert(plow_control_command_v1_valid(&h, &c, 1));

    c.commit_seq = 0;
    assert(!plow_control_command_v1_valid(&h, &c, 1));
    c.commit_seq = 1;
    c.session_nonce++;
    assert(!plow_control_command_v1_valid(&h, &c, 1));
    c.session_nonce = h.session_nonce;
    c.opcode = 0xffff;
    assert(!plow_control_command_v1_valid(&h, &c, 1));
    c.opcode = PLOW_CONTROL_STOP;
    assert(plow_control_command_v1_valid(&h, &c, 1));

    h.magic ^= 1;
    assert(!plow_control_header_v1_valid(&h));
    h = valid_header();
    h.abi_major++;
    assert(!plow_control_header_v1_valid(&h));
    h = valid_header();
    h.ring_order = 1;
    assert(!plow_control_header_v1_valid(&h));
    h = valid_header();
    h.required_features |= 1u << 31;
    assert(!plow_control_header_v1_valid(&h));
    return 0;
}
