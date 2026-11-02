/* op_collective.cuh — cross-rank all-reduce bodies for the NVIDIA interpreter.
 *
 * Same packet contract and arithmetic as runtime/amd/op_collective.h (d_xreduce_mega,
 * d_xreduce_twoshot_mega): each rank's producing GEMV has already written its bf16 partial
 * into its own region `peer_scratch[rank] + slot`, the local interpreter gate ordered that
 * write before this packet, and every rank's xctr was zeroed before the launch.
 *
 * The regions may be peer VRAM or page-locked host memory shared between processes (one rank
 * per process, e.g. MIG instances). Read-modify-writes from DIFFERENT devices on mapped host
 * memory are not atomic with respect to each other, so no word here has writers on two
 * devices:
 *   - gate line `id` (128 B) holds one arrival word per SOURCE rank: rank s announces itself
 *     with a system-scope release STORE of 1 into word s of every rank's gate line, and a
 *     rank's gate clears when words 0..n-1 of its own line are all set;
 *   - the two-shot's second rendezvous needs every block of the source rank: the blocks
 *     aggregate on word 31 of the source rank's OWN line (one device), and the block that
 *     completes the count stores the arrival flags for all of them.
 * Polls are ld.relaxed.sys followed by ONE fence.acq_rel.sys once the gate clears; peer data
 * moves with ld/st.relaxed.sys. Accumulation is f32 in strict rank order 0..N-1, so every
 * rank produces bit-identical bf16.
 *
 * A gate that does not clear within kXctrDeadlineNs (%globaltimer) writes 0xDEAD0000|rank to
 * the packet's status line (fj[2] = status id + 1, patched by the host) and skips the reduce.
 * Once the status line is set every later collective of the launch skips at once, so the
 * program still runs to completion (no stranded block) and the host fails the step on its
 * status audit. */
#pragma once

constexpr unsigned long long kXctrDeadlineNs = 10000000000ull;
constexpr uint32_t kXrAggWord = 31u;

__device__ __forceinline__ unsigned long long xr_gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint32_t xr_ld32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void xr_fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void xr_store(uint32_t* p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ float xr_ld(const __nv_bfloat16* p) {
    unsigned short v;
    asm volatile("ld.relaxed.sys.global.u16 %0, [%1];" : "=h"(v) : "l"(p) : "memory");
    return __bfloat162float(__ushort_as_bfloat16(v));
}
__device__ __forceinline__ void xr_st(__nv_bfloat16* p, float f) {
    const unsigned short v = __bfloat16_as_ushort(__float2bfloat16_rn(f));
    asm volatile("st.relaxed.sys.global.u16 [%0], %1;" :: "l"(p), "h"(v) : "memory");
}

__device__ __forceinline__ uint32_t* xr_line(const PlowProgram& prog, uint32_t r, size_t xoff,
                                             uint32_t id) {
    return PLOW_CTR((uint32_t*)((char*)prog.peer_scratch[r] + xoff), id);
}

/* Thread 0: announce this rank on gate `id` of every rank. The leading system fence makes
 * the partial the producer blocks wrote (ordered before this packet by the local gate)
 * visible beyond this device before any peer can observe the flag. */
__device__ __forceinline__ void xr_arrive(const PlowProgram& prog, size_t xoff, uint32_t id) {
    xr_fence_sys();
    for (uint32_t r = 0; r < prog.n_gpu; r++) xr_store(xr_line(prog, r, xoff, id) + prog.rank, 1u);
}

/* Thread 0: wait for every rank's flag on this rank's gate `id`; false on deadline or when an
 * earlier collective of this launch already bailed. */
__device__ __forceinline__ bool xr_wait(const PlowProgram& prog, size_t xoff, uint32_t id,
                                        uint32_t* status) {
    const uint32_t* g = xr_line(prog, prog.rank, xoff, id);
    const unsigned long long t0 = xr_gtime();
    for (uint32_t r = 0; r < prog.n_gpu;) {
        if (xr_ld32(g + r)) {
            r++;
            continue;
        }
        if ((status && xr_ld32(status)) || xr_gtime() - t0 > kXctrDeadlineNs) {
            if (status) xr_store(status, 0xDEAD0000u | prog.rank);
            return false;
        }
        __nanosleep(64);
    }
    xr_fence_sys();
    return true;
}

__device__ __forceinline__ uint32_t* xr_status(const PlowProgram& prog, const PlowDevInst* in) {
    return in->fj[2].u ? PLOW_CTR(prog.xctr, in->fj[2].u - 1u) : nullptr;
}

/* XREDUCE (one-shot): t0=out i0=n i1=n_gpu i2=slot_bytes i3=gate. */
__device__ __forceinline__ void d_xreduce_nv(const PlowProgram& prog, const PlowDevInst* in,
                                             void* const* T, unsigned slice, unsigned nblk) {
    __shared__ int bailed;
    const size_t xoff = (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
    const uint32_t n = in->i[0], nr = prog.n_gpu, slot = in->i[2], gate = in->i[3];
    /* Folded gather / HyperConn post: not on this backend. */
    if (in->i[1] != nr || in->i[5] != 0u || in->t[2] != PLOW_TENSOR_NONE) __trap();
    if (threadIdx.x == 0) {
        if (slice == 0) xr_arrive(prog, xoff, gate);
        bailed = !xr_wait(prog, xoff, gate, xr_status(prog, in));
    }
    __syncthreads();
    if (bailed) return;
    __nv_bfloat16* out = (__nv_bfloat16*)T[in->t[0]];
    for (uint32_t e = slice * blockDim.x + threadIdx.x; e < n; e += nblk * blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < nr; r++)
            acc += xr_ld((const __nv_bfloat16*)((const char*)prog.peer_scratch[r] + slot) + e);
        out[e] = __float2bfloat16_rn(acc);
    }
}

/* XREDUCE2 (two-shot): t0=out t1=resid t2=out2 i0=n i1=n_gpu i2=slot_bytes i3=gate_rs
 * i4=gate_ag i5=e0 (element offset of this band into out and the slot). */
__device__ __forceinline__ void d_xreduce_twoshot_nv(const PlowProgram& prog,
                                                     const PlowDevInst* in, void* const* T,
                                                     unsigned slice, unsigned nblk) {
    __shared__ int bailed;
    const size_t xoff = (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
    const uint32_t nr = prog.n_gpu, n = in->i[0], e0 = in->i[5];
    const uint32_t slot = in->i[2] + e0 * 2u;
    /* Folded gather / AttnRes rows: not on this backend. */
    if (in->i[1] != nr || in->i[7] != 0u || in->t[3] != PLOW_TENSOR_NONE) __trap();
    uint32_t* status = xr_status(prog, in);
    const unsigned tid = slice * blockDim.x + threadIdx.x, stride = nblk * blockDim.x;
    if (threadIdx.x == 0) {
        if (slice == 0) xr_arrive(prog, xoff, in->i[3]);
        bailed = !xr_wait(prog, xoff, in->i[3], status);
    }
    __syncthreads();
    if (bailed) return;
    /* Reduce-scatter: this rank owns [lo, hi) and writes it in place into its own slot. */
    const uint32_t lo = (uint32_t)(((uint64_t)n * prog.rank) / nr);
    const uint32_t hi = (uint32_t)(((uint64_t)n * (prog.rank + 1)) / nr);
    __nv_bfloat16* mine = (__nv_bfloat16*)((char*)prog.peer_scratch[prog.rank] + slot);
    for (uint32_t e = lo + tid; e < hi; e += stride) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < nr; r++)
            acc += xr_ld((const __nv_bfloat16*)((const char*)prog.peer_scratch[r] + slot) + e);
        xr_st(mine + e, acc);
    }
    /* Second rendezvous: the reduced slice is written by ALL blocks of this rank. Each block
     * releases its own stores into the device-local count; the block that completes it
     * acquires them all and announces the rank. */
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t* agg = xr_line(prog, prog.rank, xoff, in->i[4]) + kXrAggWord;
        uint32_t prev;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(prev) : "l"(agg) : "memory");
        if (prev + 1u == nblk) xr_arrive(prog, xoff, in->i[4]);
        bailed = !xr_wait(prog, xoff, in->i[4], status);
    }
    __syncthreads();
    if (bailed) return;
    __nv_bfloat16* out = (__nv_bfloat16*)T[in->t[0]] + e0;
    const __nv_bfloat16* resid =
        in->t[1] == PLOW_TENSOR_NONE ? nullptr : (const __nv_bfloat16*)T[in->t[1]] + e0;
    __nv_bfloat16* out2 =
        in->t[2] == PLOW_TENSOR_NONE ? nullptr : (__nv_bfloat16*)T[in->t[2]] + e0;
    for (uint32_t i = 0; i < nr; i++) {
        const uint32_t s = (slice + prog.rank + i) % nr;
        const uint32_t slo = (uint32_t)(((uint64_t)n * s) / nr);
        const uint32_t shi = (uint32_t)(((uint64_t)n * (s + 1)) / nr);
        const __nv_bfloat16* src =
            (const __nv_bfloat16*)((const char*)prog.peer_scratch[s] + slot);
        for (uint32_t e = slo + tid; e < shi; e += stride) {
            const float v = xr_ld(src + e);
            if (out2)
                out2[e] = __float2bfloat16_rn((resid ? __bfloat162float(resid[e]) : 0.0f) + v);
            else
                out[e] = __float2bfloat16_rn(v);
        }
    }
}
