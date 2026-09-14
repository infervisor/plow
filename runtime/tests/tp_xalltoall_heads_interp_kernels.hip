/* tp_xalltoall_heads_interp_kernels.hip — drives DevOp::XAllToAllHeads (op 160) through the
 * SAME `PlowDevInst` decode `runtime/amd/interp.hip`'s real dispatch arm uses (the
 * `PLOW_DOP_XALLTOALL_HEADS` case, copied verbatim below), instead of calling
 * `d_xalltoall_heads_mega` with named arguments.
 *
 * The device BODY (wave-per-peer, 16-byte accesses) is already priced standalone, byte-exact,
 * in tp_alltoall_bench.c / tp_alltoall_kernels.hip (job 0-1789234094-a2a-collective-bench3,
 * 291.2/257.8 us at cap 48). What that harness does NOT exercise is the op's OWN ABI wiring:
 * the `DevInst64`/`PlowDevInst` operand decode (i0..i7, t0) this file's dispatch adds to
 * interp.hip, and the `PLOW_ROWSPLIT_A2A` build-flag derivation from `PLOW_HAS_XALLTOALL_HEADS`
 * a generated object's plow_config.h would carry. This harness is built with
 * `-DPLOW_ROWSPLIT_A2A=1` directly (the flag a real object derives from the opcode's presence)
 * to prove that wiring, not just the device body.
 *
 * Q shape: [8192][8][576] per rank -> [1024][64][576] per rank (dir=0), 66.06 MB/rank moved.
 * O shape: [1024][64][512] per rank -> [8192][8][512] per rank (dir=1), 58.72 MB/rank moved.
 * Same shapes, same fill/oracle convention as tp_alltoall_kernels.hip's tp_a2a_fill / the host
 * oracle in tp_xalltoall_heads_interp_bench.c (must match byte-for-byte).
 */
#include "../common/dev_isa.h"
#include "../amd/op_collective.h"

extern "C" __device__ uint32_t txh_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
extern "C" __global__ void txh_fill(bf16* part, uint32_t n, uint32_t rank, uint32_t seed) {
    const uint32_t rs = txh_hash(rank * 0x9E3779B9u + seed);
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) {
        const uint32_t h = txh_hash(e ^ rs);
        part[e] = (bf16)(((h >> 31) << 15) | ((120u + ((h >> 20) & 15u)) << 7) | (h & 0x7fu));
    }
}

__device__ __forceinline__ uint64_t txh_now(void) { return __builtin_amdgcn_s_memrealtime(); }

/* THE DISPATCH ARM, copied verbatim from runtime/amd/interp.hip's
 * `if (in->op == PLOW_DOP_XALLTOALL_HEADS) { ... }` case: same operand indices, same call. The
 * one addition is `tens`, a tiny device-pointer table standing in for the interpreter's real
 * per-program tensor table (`tens[in->t[0]]`) — this harness only ever needs one live handle. */
extern "C" __global__ void txh_dispatch_bench(
    const PlowDevInst* in, void* const* tens, const void* const* peer_scratch, uint32_t n_gpu,
    uint32_t rank, size_t xctr_byte_off, uint64_t deadline_ticks, uint32_t* status,
    uint64_t* ts) {
    const uint64_t t0 = txh_now();
    d_xalltoall_heads_mega(peer_scratch, n_gpu, rank, xctr_byte_off, in->i[4], deadline_ticks,
                           status, blockIdx.x, in->blocks, (bf16*)tens[in->t[0]], in->i[0],
                           in->i[1], in->i[2], in->i[3], in->i[6], in->i[7]);
    __syncthreads();
    if (threadIdx.x == 0 && ts) { ts[2 * blockIdx.x] = t0; ts[2 * blockIdx.x + 1] = txh_now(); }
}
