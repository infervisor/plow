#ifndef PLOW_CONTROL_QUEUE_PROBE_H
#define PLOW_CONTROL_QUEUE_PROBE_H

#include <stddef.h>
#include <stdint.h>

#define PLOW_CONTROL_MAGIC UINT64_C(0x314c5443574f4c50)
#define PLOW_CONTROL_ABI_MAJOR 1u
#define PLOW_CONTROL_ABI_MINOR 0u
#define PLOW_CONTROL_HEADER_BYTES 256u
#define PLOW_CONTROL_COMMAND_BYTES 256u

enum PlowControlOpcode {
    PLOW_CONTROL_RUN_DECODE = 1,
    PLOW_CONTROL_QUIESCE = 2,
    PLOW_CONTROL_STOP = 3,
    PLOW_CONTROL_NOP = 4,
};

enum PlowControlFeature {
    PLOW_CONTROL_FEAT_RUN_DECODE = 1u << 0,
};

typedef struct __attribute__((aligned(64))) PlowControlHeaderV1 {
    uint64_t magic;
    uint16_t abi_major;
    uint16_t abi_minor;
    uint16_t header_bytes;
    uint16_t command_bytes;
    uint16_t ring_order;
    uint16_t rank;
    uint16_t n_gpu;
    uint16_t reserved0;
    uint32_t required_features;
    uint32_t supported_features;
    uint64_t session_nonce;
    uint64_t code_profile_hash;
    uint64_t registry_hash;
    uint64_t reserved1;

    volatile uint64_t host_tail_seq;
    uint64_t host_flags;
    uint64_t host_heartbeat;
    uint64_t host_reserved[5];

    volatile uint64_t dev_head_seq;
    volatile uint64_t completed_seq;
    volatile uint64_t quiesced_seq;
    volatile uint64_t stopped_seq;
    uint64_t device_heartbeat;
    uint32_t state;
    uint32_t phase;
    uint64_t device_reserved[2];

    volatile uint64_t fault_seq;
    uint32_t fault_code;
    uint16_t fault_stage;
    uint16_t fault_rank;
    uint32_t fault_block;
    uint32_t fault_detail;
    uint64_t fault_command_digest;
    uint32_t last_program;
    uint32_t last_rung;
    uint64_t audit_bits;
    uint64_t fault_reserved[2];
} PlowControlHeaderV1;

typedef struct __attribute__((aligned(64))) PlowControlCommandV1 {
    volatile uint64_t commit_seq;
    uint64_t session_nonce;
    uint16_t record_bytes;
    uint16_t abi_minor;
    uint16_t opcode;
    uint16_t flags;
    uint32_t program_id;
    uint32_t program_generation;
    uint64_t program_hash;
    uint64_t command_digest;
    uint64_t slot_mask;
    uint32_t rung_rows;
    uint32_t highest_slot_plus1;
    uint64_t input_epoch;
    uint64_t state_epoch;
    uint64_t counter_epoch;
    uint64_t xctr_epoch;
    uint64_t result_epoch;
    uint32_t audit_policy;
    uint32_t timeout_ticks;
    uint32_t result_slot;
    uint32_t reserved32;
    uint64_t reserved64;
    uint64_t extension_reserved[16];
} PlowControlCommandV1;

typedef struct PlowControlProbeV1 {
    PlowControlHeaderV1 header;
    PlowControlCommandV1 command;
} PlowControlProbeV1;

static inline int plow_control_header_v1_valid(const PlowControlHeaderV1* h) {
    return h != NULL && h->magic == PLOW_CONTROL_MAGIC &&
           h->abi_major == PLOW_CONTROL_ABI_MAJOR &&
           h->header_bytes == PLOW_CONTROL_HEADER_BYTES &&
           h->command_bytes == PLOW_CONTROL_COMMAND_BYTES &&
           h->ring_order == 0 && h->n_gpu == 1 && h->rank == 0 &&
           (h->required_features & ~h->supported_features) == 0;
}

static inline int plow_control_command_v1_valid(const PlowControlHeaderV1* h,
                                                const PlowControlCommandV1* c,
                                                uint64_t expected_seq) {
    return plow_control_header_v1_valid(h) && c != NULL &&
           c->commit_seq == expected_seq && c->session_nonce == h->session_nonce &&
           c->record_bytes == PLOW_CONTROL_COMMAND_BYTES &&
           c->abi_minor <= h->abi_minor &&
           (c->opcode == PLOW_CONTROL_RUN_DECODE || c->opcode == PLOW_CONTROL_STOP);
}

#if defined(__cplusplus)
static_assert(sizeof(PlowControlHeaderV1) == PLOW_CONTROL_HEADER_BYTES,
              "control header ABI size");
static_assert(sizeof(PlowControlCommandV1) == PLOW_CONTROL_COMMAND_BYTES,
              "control command ABI size");
static_assert(alignof(PlowControlHeaderV1) == 64, "control header ABI alignment");
static_assert(alignof(PlowControlCommandV1) == 64, "control command ABI alignment");
static_assert(sizeof(PlowControlProbeV1) == 512, "control probe ABI size");
static_assert(alignof(PlowControlProbeV1) == 64, "control probe ABI alignment");
#else
_Static_assert(sizeof(PlowControlHeaderV1) == PLOW_CONTROL_HEADER_BYTES,
               "control header ABI size");
_Static_assert(sizeof(PlowControlCommandV1) == PLOW_CONTROL_COMMAND_BYTES,
               "control command ABI size");
_Static_assert(_Alignof(PlowControlHeaderV1) == 64, "control header ABI alignment");
_Static_assert(_Alignof(PlowControlCommandV1) == 64, "control command ABI alignment");
_Static_assert(sizeof(PlowControlProbeV1) == 512, "control probe ABI size");
_Static_assert(_Alignof(PlowControlProbeV1) == 64, "control probe ABI alignment");
#endif

#endif
