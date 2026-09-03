#ifndef PLOW_KDA_CHUNK_KEY_FACTOR_COMMON_H
#define PLOW_KDA_CHUNK_KEY_FACTOR_COMMON_H

#ifdef PLOW_CONFIG
#include PLOW_CONFIG
#endif
#ifndef PLOW_PACKET_HASH
#define PLOW_PACKET_HASH 0ull
#endif

#include "op_elementwise.h"
#include "op_kda.h"

extern "C" __device__ unsigned plow_kda_key_factor_abi_1 = 1;
extern "C" __device__ unsigned plow_kda_key_factor_pair_1 = 1;
extern "C" __device__ unsigned plow_kda_key_factor_bt64_d128_v128_1 = 1;
extern "C" __device__ unsigned plow_kda_key_factor_qpre_1 = 1;
extern "C" __device__ unsigned plow_kda_key_factor_wave64_1 = 1;
extern "C" __device__ unsigned plow_kda_key_factor_nospill_1 = 1;
extern "C" __device__ unsigned plow_kda_key_factor_scratch_pair_bf16_1 = 1;

#if PLOW_PACKET_HASH != 0ull
extern "C" __device__ unsigned plow_packet_hash_lo =
    (unsigned)(PLOW_PACKET_HASH & 0xffffffffull);
extern "C" __device__ unsigned plow_packet_hash_hi =
    (unsigned)(PLOW_PACKET_HASH >> 32);
#endif

#endif
