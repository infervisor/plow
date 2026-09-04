#ifdef PLOW_CONFIG
#include PLOW_CONFIG
#endif
#ifndef PLOW_PACKET_HASH
#define PLOW_PACKET_HASH 0ull
#endif

#include "op_collective.h"
#include "op_k3.h"

#ifndef PLOW_XCTR_DEADLINE_TICKS
#define PLOW_XCTR_DEADLINE_TICKS 1000000000ull
#endif

extern "C" __device__ unsigned plow_xr_attnres_segment_1 = 1;
extern "C" __device__ unsigned plow_xr_attnres_wave64_nospill_1 = 1;
#if PLOW_PACKET_HASH != 0ull
extern "C" __device__ unsigned plow_packet_hash_lo =
    (unsigned)(PLOW_PACKET_HASH & 0xffffffffull);
extern "C" __device__ unsigned plow_packet_hash_hi =
    (unsigned)(PLOW_PACKET_HASH >> 32);
#endif

typedef struct {
    bf16* reduced;
    bf16* prefix;
    bf16* out;
    const bf16* residual;
    const bf16* ring;
    const float* score;
    const bf16* gamma;
    const void* const* peer_scratch;
    uint32_t* xctr;
    uint32_t* status;
    uint32_t n;
    uint32_t slot_bytes;
    uint32_t gate_rs;
    uint32_t gate_ag;
    uint32_t rank;
    uint32_t nranks;
    uint32_t row_w;
    uint32_t nb;
    uint32_t nbcap;
    float eps;
} PlowXReduceAttnResArgs;
static_assert(sizeof(PlowXReduceAttnResArgs) == 120, "XR+AttnRes kernarg ABI");

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
__attribute__((amdgpu_waves_per_eu(PLOW_WAVES / 4, PLOW_WAVES / 4)))
void plow_xreduce_attnres_gfx950(const volatile PlowXReduceAttnResArgs* a) {
    __shared__ __align__(16) char raw[147464];
    const size_t xoff = (const char*)a->xctr - (const char*)a->peer_scratch[a->rank];
    d_xreduce_twoshot_mega(a->reduced, a->peer_scratch, a->nranks, a->rank, a->n,
                           a->slot_bytes, xoff, a->gate_rs, a->gate_ag,
                           PLOW_XCTR_DEADLINE_TICKS, a->status,
                           blockIdx.x, gridDim.x, a->residual, nullptr, 0, 0,
                           a->prefix, a->row_w);
    d_attn_res(a->out, a->prefix, a->ring, a->score, a->n / a->row_w, a->row_w,
               a->nb, a->nbcap, a->eps, blockIdx.x, gridDim.x,
               (float*)raw, nullptr, 0u, a->gamma);
}
