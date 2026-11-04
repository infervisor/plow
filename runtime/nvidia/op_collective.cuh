/* op_collective.cuh — cross-rank all-reduce bodies for the NVIDIA interpreter.
 *
 * Unstaged transport: same packet contract and arithmetic as runtime/amd/op_collective.h
 * (d_xreduce_mega, d_xreduce_twoshot_mega): each rank's producing GEMV has already written its bf16 partial
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
 * NVIDIA peer tables have n_gpu + 1 entries: peer_scratch[n_gpu] is this rank's
 * VRAM mirror (same region layout), or NULL for the unstaged transport. Only objects
 * exporting plow_xr_staged_1 may receive a non-NULL mirror. Producers write the mirror;
 * collectives publish raw bf16 bits to host and aggregate/poll in local VRAM.
 *
 * Slot reuse requires an intervening dependent collective: publishing new host contents
 * happens only after every remote reader of the previous contents retired through it.
 * Local overwrite follows retirement of local readers through the interpreter's gates.
 *
 * A gate that does not clear within kXctrDeadlineNs (%globaltimer) writes 0xDEAD0000|rank to
 * the packet's status line (fj[2] = status id + 1, patched by the host) and skips the reduce.
 * Once the status line is set every later collective of the launch skips at once, so the
 * program still runs to completion (no stranded block) and the host fails the step on its
 * status audit. */
#pragma once

constexpr unsigned long long kXctrDeadlineNs = 10000000000ull;
constexpr uint32_t kXrAggWord = 31u;
constexpr uint32_t kXrRelayWord = 30u;

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

struct alignas(16) XrVec {
    __nv_bfloat16 v[8];
};

__device__ __forceinline__ uint4 xr_ld8(const __nv_bfloat16* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ float xr_vec_at(const uint4& v, unsigned k) {
    const uint32_t w = k < 2 ? v.x : k < 4 ? v.y : k < 6 ? v.z : v.w;
    return __bfloat162float(__ushort_as_bfloat16((unsigned short)(w >> ((k & 1u) * 16u))));
}

__device__ __forceinline__ void xr_st8(__nv_bfloat16* p, const XrVec& v) {
    const uint4 u = *(const uint4*)&v;
    asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "r"(u.x), "r"(u.y), "r"(u.z), "r"(u.w) : "memory");
}

__device__ __forceinline__ XrVec xr_local_ld8(const __nv_bfloat16* p) {
    XrVec v;
    if (((uintptr_t)p & 15u) == 0)
        *(uint4*)&v = *(const uint4*)p;
    else {
#pragma unroll
        for (unsigned k = 0; k < 8; k++) v.v[k] = p[k];
    }
    return v;
}

__device__ __forceinline__ void xr_local_st8(__nv_bfloat16* p, const XrVec& v) {
    if (((uintptr_t)p & 15u) == 0)
        *(uint4*)p = *(const uint4*)&v;
    else {
#pragma unroll
        for (unsigned k = 0; k < 8; k++) p[k] = v.v[k];
    }
}

__device__ __forceinline__ uint32_t xr_vec_begin(uint32_t slot, uint32_t lo, uint32_t hi) {
    const uint32_t head = (0u - (slot / 2u + lo)) & 7u;
    return lo + (head < hi - lo ? head : hi - lo);
}

template <bool PeerOut>
__device__ __forceinline__ void xr_reduce_range(const PlowProgram& prog, uint32_t slot,
                                                uint32_t lo, uint32_t hi, __nv_bfloat16* out,
                                                unsigned tid, unsigned stride) {
    const uint32_t begin = xr_vec_begin(slot, lo, hi);
    const uint32_t count = (hi - begin) / 8u, end = begin + count * 8u;
    for (uint32_t t = tid; t < begin - lo + hi - end; t += stride) {
        const uint32_t e = t < begin - lo ? lo + t : end + t - (begin - lo);
        float acc = 0.0f;
        for (uint32_t r = 0; r < prog.n_gpu; r++)
            acc += xr_ld((const __nv_bfloat16*)((const char*)prog.peer_scratch[r] + slot) + e);
        if constexpr (PeerOut) xr_st(out + e, acc);
        else out[e] = __float2bfloat16_rn(acc);
    }
    for (uint32_t q = tid; q < count; q += 2u * stride) {
        float acc[2][8] = {};
        for (uint32_t r = 0; r < prog.n_gpu; r += 2u) {
            uint4 v[2][2];
            /* Issue independent peer loads before consuming either rank or vector. */
#pragma unroll
            for (unsigned rr = 0; rr < 2; rr++) {
                if (r + rr < prog.n_gpu) {
                    const __nv_bfloat16* src =
                        (const __nv_bfloat16*)((const char*)prog.peer_scratch[r + rr] + slot);
#pragma unroll
                    for (unsigned j = 0; j < 2; j++)
                        if (q + j * stride < count)
                            v[rr][j] = xr_ld8(src + begin + (q + j * stride) * 8u);
                }
            }
#pragma unroll
            for (unsigned rr = 0; rr < 2; rr++)
                if (r + rr < prog.n_gpu) {
#pragma unroll
                    for (unsigned j = 0; j < 2; j++)
                        if (q + j * stride < count) {
#pragma unroll
                            for (unsigned k = 0; k < 8; k++)
                                acc[j][k] += xr_vec_at(v[rr][j], k);
                        }
                }
        }
#pragma unroll
        for (unsigned j = 0; j < 2; j++)
            if (q + j * stride < count) {
                XrVec v;
#pragma unroll
                for (unsigned k = 0; k < 8; k++) v.v[k] = __float2bfloat16_rn(acc[j][k]);
                __nv_bfloat16* dst = out + begin + (q + j * stride) * 8u;
                if constexpr (PeerOut) xr_st8(dst, v);
                else xr_local_st8(dst, v);
            }
    }
}

__device__ __forceinline__ void xr_gather_range(const __nv_bfloat16* src, uint32_t slot,
                                                uint32_t lo, uint32_t hi, __nv_bfloat16* out,
                                                const __nv_bfloat16* resid, __nv_bfloat16* out2,
                                                unsigned tid, unsigned stride) {
    const uint32_t begin = xr_vec_begin(slot, lo, hi);
    const uint32_t count = (hi - begin) / 8u, end = begin + count * 8u;
    for (uint32_t t = tid; t < begin - lo + hi - end; t += stride) {
        const uint32_t e = t < begin - lo ? lo + t : end + t - (begin - lo);
        const float v = xr_ld(src + e);
        if (out2)
            out2[e] = __float2bfloat16_rn((resid ? __bfloat162float(resid[e]) : 0.0f) + v);
        else
            out[e] = __float2bfloat16_rn(v);
    }
    for (uint32_t q = tid; q < count; q += 2u * stride) {
        uint4 v[2];
#pragma unroll
        for (unsigned j = 0; j < 2; j++)
            if (q + j * stride < count)
                v[j] = xr_ld8(src + begin + (q + j * stride) * 8u);
#pragma unroll
        for (unsigned j = 0; j < 2; j++)
            if (q + j * stride < count) {
                const uint32_t e = begin + (q + j * stride) * 8u;
                XrVec value;
                *(uint4*)&value = v[j];
                if (out2) {
                    const XrVec res = resid ? xr_local_ld8(resid + e) : XrVec{};
#pragma unroll
                    for (unsigned k = 0; k < 8; k++)
                        value.v[k] = __float2bfloat16_rn(__bfloat162float(res.v[k]) +
                                                        xr_vec_at(v[j], k));
                    xr_local_st8(out2 + e, value);
                } else {
                    xr_local_st8(out + e, value);
                }
            }
    }
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

/* Copy bits, including unaligned heads/tails, without a floating-point conversion. */
__device__ __forceinline__ void xr_publish_range(const PlowProgram& prog, uint32_t slot,
                                                 uint32_t lo, uint32_t hi,
                                                 unsigned tid, unsigned stride) {
    const auto* src = (const __nv_bfloat16*)((const char*)prog.peer_scratch[prog.n_gpu] + slot);
    auto* dst = (__nv_bfloat16*)((char*)prog.peer_scratch[prog.rank] + slot);
    const uint32_t begin = xr_vec_begin(slot, lo, hi);
    const uint32_t count = (hi - begin) / 8u, end = begin + count * 8u;
    for (uint32_t t = tid; t < begin - lo + hi - end; t += stride) {
        const uint32_t e = t < begin - lo ? lo + t : end + t - (begin - lo);
        const unsigned short bits = __bfloat16_as_ushort(src[e]);
        asm volatile("st.relaxed.sys.global.u16 [%0], %1;"
                     :: "l"(dst + e), "h"(bits) : "memory");
    }
    for (uint32_t q = tid; q < count; q += stride)
        xr_st8(dst + begin + q * 8u, xr_local_ld8(src + begin + q * 8u));
}

/* CTA stores -> CTA barrier -> acq_rel GPU RMW chain -> elected system release.
 * The elected system acquire -> GPU release/acquire relay carries peer visibility to
 * followers. Their system fence also orders subsequent relaxed system-scope peer loads.
 * Followers never read host flags/status; only a local deadline writes host status. */
__device__ __forceinline__ bool xr_staged_barrier(const PlowProgram& prog, size_t xoff,
                                                  uint32_t gate, uint32_t* status,
                                                  unsigned nblk) {
    __shared__ int bailed;
    __syncthreads();
    if (threadIdx.x == 0) {
        auto* line = PLOW_CTR((uint32_t*)((char*)prog.peer_scratch[prog.n_gpu] + xoff), gate);
        uint32_t prev;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(prev) : "l"(line + kXrAggWord) : "memory");
        if (prev + 1u == nblk) {
            xr_arrive(prog, xoff, gate);
            const uint32_t outcome = xr_wait(prog, xoff, gate, status) ? 1u : 2u;
            asm volatile("st.release.gpu.global.u32 [%0], %1;"
                         :: "l"(line + kXrRelayWord), "r"(outcome) : "memory");
            bailed = outcome == 2u;
        } else {
            const unsigned long long t0 = xr_gtime();
            uint32_t outcome;
            for (;;) {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(outcome) : "l"(line + kXrRelayWord) : "memory");
                if (outcome) break;
                if (xr_gtime() - t0 > kXctrDeadlineNs) {
                    if (status) xr_store(status, 0xDEAD0000u | prog.rank);
                    outcome = 2u;
                    break;
                }
                __nanosleep(64);
            }
            xr_fence_sys();
            bailed = outcome == 2u;
        }
    }
    __syncthreads();
    return !bailed;
}

#if PLOW_PACKET_REQUIRES_XR_PARTIAL_F32 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
__device__ __forceinline__ void xr_publish_f32(const PlowProgram& prog, uint32_t slot,
                                               uint32_t n, unsigned tid, unsigned stride) {
    const auto* src = (const uint32_t*)((const char*)prog.peer_scratch[prog.n_gpu] + slot);
    auto* dst = (uint32_t*)((char*)prog.peer_scratch[prog.rank] + slot);
    for (uint32_t e = tid; e < n; e += stride) {
        const uint32_t bits = xr_ld32(src + e);
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;"
                     :: "l"(dst + e), "r"(bits) : "memory");
    }
}

__device__ __forceinline__ void xr_reduce_f32(const PlowProgram& prog, uint32_t slot,
                                              uint32_t n, __nv_bfloat16* out,
                                              unsigned tid, unsigned stride) {
    for (uint32_t e = tid; e < n; e += stride) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < prog.n_gpu; r++) {
            const void* region = r == prog.rank && prog.peer_scratch[prog.n_gpu]
                                     ? prog.peer_scratch[prog.n_gpu] : prog.peer_scratch[r];
            const auto* src = (const uint32_t*)((const char*)region + slot);
            acc = __fadd_rn(acc, __uint_as_float(xr_ld32(src + e)));
        }
        out[e] = __float2bfloat16_rn(acc);
    }
}
#endif

template <bool Scatter>
__device__ __forceinline__ void xr_staged_reduce_range(const PlowProgram& prog, uint32_t slot,
                                                uint32_t lo, uint32_t hi, __nv_bfloat16* out,
                                                const __nv_bfloat16* resid, __nv_bfloat16* out2,
                                                unsigned tid, unsigned stride) {
    const uint32_t begin = xr_vec_begin(slot, lo, hi);
    const uint32_t count = (hi - begin) / 8u, end = begin + count * 8u;
    for (uint32_t t = tid; t < begin - lo + hi - end; t += stride) {
        const uint32_t e = t < begin - lo ? lo + t : end + t - (begin - lo);
        float acc = 0.0f;
        for (uint32_t r = 0; r < prog.n_gpu; r++) {
            const void* region = prog.peer_scratch[r == prog.rank ? prog.n_gpu : r];
            acc += xr_ld((const __nv_bfloat16*)((const char*)region + slot) + e);
        }
        const __nv_bfloat16 v = __float2bfloat16_rn(acc);
        if constexpr (Scatter) {
            auto* dst = (__nv_bfloat16*)((char*)prog.peer_scratch[prog.rank] + slot) + e;
            const unsigned short bits = __bfloat16_as_ushort(v);
            asm volatile("st.relaxed.sys.global.u16 [%0], %1;"
                         :: "l"(dst), "h"(bits) : "memory");
        }
        if (out2)
            out2[e] = __float2bfloat16_rn((resid ? __bfloat162float(resid[e]) : 0.0f) +
                                        __bfloat162float(v));
        else out[e] = v;
    }
    for (uint32_t q = tid; q < count; q += 2u * stride) {
        float acc[2][8] = {};
        for (uint32_t r = 0; r < prog.n_gpu; r += 2u) {
            uint4 v[2][2];
            /* Issue independent peer loads before consuming either rank or vector. */
#pragma unroll
            for (unsigned rr = 0; rr < 2; rr++) {
                if (r + rr < prog.n_gpu) {
                    const void* region = prog.peer_scratch[r + rr == prog.rank ? prog.n_gpu : r + rr];
                    const auto* src = (const __nv_bfloat16*)((const char*)region + slot);
#pragma unroll
                    for (unsigned j = 0; j < 2; j++)
                        if (q + j * stride < count)
                            v[rr][j] = xr_ld8(src + begin + (q + j * stride) * 8u);
                }
            }
#pragma unroll
            for (unsigned rr = 0; rr < 2; rr++)
                if (r + rr < prog.n_gpu) {
#pragma unroll
                    for (unsigned j = 0; j < 2; j++)
                        if (q + j * stride < count) {
#pragma unroll
                            for (unsigned k = 0; k < 8; k++)
                                acc[j][k] += xr_vec_at(v[rr][j], k);
                        }
                }
        }
#pragma unroll
        for (unsigned j = 0; j < 2; j++)
            if (q + j * stride < count) {
                XrVec v;
#pragma unroll
                for (unsigned k = 0; k < 8; k++) v.v[k] = __float2bfloat16_rn(acc[j][k]);
                const uint32_t e = begin + (q + j * stride) * 8u;
                if constexpr (Scatter)
                    xr_st8((__nv_bfloat16*)((char*)prog.peer_scratch[prog.rank] + slot) + e, v);
                if (out2) {
                    const XrVec res = resid ? xr_local_ld8(resid + e) : XrVec{};
#pragma unroll
                    for (unsigned k = 0; k < 8; k++)
                        v.v[k] = __float2bfloat16_rn(__bfloat162float(res.v[k]) +
                                                   __bfloat162float(v.v[k]));
                    xr_local_st8(out2 + e, v);
                } else xr_local_st8(out + e, v);
            }
    }
}

/* XREDUCE (one-shot): t0=out i0=n i1=n_gpu i2=slot_bytes i3=gate. */
__device__ __forceinline__ void d_xreduce_nv(const PlowProgram& prog, const PlowDevInst* in,
                                             void* const* T, unsigned slice, unsigned nblk) {
#if PLOW_PACKET_REQUIRES_XR_PARTIAL_F32 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    if (in->fj[0].u == PLOW_XREDUCE_PARTIAL_F32_MARK) {
        const uint32_t n = in->i[0], slot = in->i[2];
        if (prog.n_gpu != 2 || in->i[1] != prog.n_gpu || (slot & 3u)
            || in->i[4] || in->i[5] || in->i[6] || in->i[7] || in->fj[1].u)
            __trap();
        const size_t xoff = (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
        const unsigned tid = slice * blockDim.x + threadIdx.x, stride = nblk * blockDim.x;
        uint32_t* status = xr_status(prog, in);
        if (prog.peer_scratch[prog.n_gpu]) {
            xr_publish_f32(prog, slot, n, tid, stride);
            if (!xr_staged_barrier(prog, xoff, in->i[3], status, nblk)) return;
        } else {
            __shared__ int bailed_f32;
            if (threadIdx.x == 0) {
                if (slice == 0) xr_arrive(prog, xoff, in->i[3]);
                bailed_f32 = !xr_wait(prog, xoff, in->i[3], status);
            }
            __syncthreads();
            if (bailed_f32) return;
        }
        xr_reduce_f32(prog, slot, n, (__nv_bfloat16*)T[in->t[0]], tid, stride);
        return;
    }
#endif
    if (prog.peer_scratch[prog.n_gpu]) {
        const size_t xoff = (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
        const uint32_t n = in->i[0], slot = in->i[2];
        if (in->i[1] != prog.n_gpu || in->i[5] != 0u || in->t[2] != PLOW_TENSOR_NONE) __trap();
        const unsigned tid = slice * blockDim.x + threadIdx.x, stride = nblk * blockDim.x;
        xr_publish_range(prog, slot, 0, n, tid, stride);
        if (!xr_staged_barrier(prog, xoff, in->i[3], xr_status(prog, in), nblk)) return;
        xr_staged_reduce_range<false>(prog, slot, 0, n, (__nv_bfloat16*)T[in->t[0]],
                                      nullptr, nullptr, tid, stride);
        return;
    }
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
    xr_reduce_range<false>(prog, slot, 0, n, out, slice * blockDim.x + threadIdx.x,
                           nblk * blockDim.x);
}

/* XREDUCE2 (two-shot): t0=out t1=resid t2=out2 i0=n i1=n_gpu i2=slot_bytes i3=gate_rs
 * i4=gate_ag i5=e0 (element offset of this band into out and the slot). */
__device__ __forceinline__ void d_xreduce_twoshot_nv(const PlowProgram& prog,
                                                     const PlowDevInst* in, void* const* T,
                                                     unsigned slice, unsigned nblk) {
    if (prog.peer_scratch[prog.n_gpu]) {
        const size_t xoff = (const char*)prog.xctr - (const char*)prog.peer_scratch[prog.rank];
        const uint32_t nr = prog.n_gpu, n = in->i[0], e0 = in->i[5];
        const uint32_t slot = in->i[2] + e0 * 2u;
        if (in->i[1] != nr || in->i[7] != 0u || in->t[3] != PLOW_TENSOR_NONE) __trap();
        const unsigned tid = slice * blockDim.x + threadIdx.x, stride = nblk * blockDim.x;
        const uint32_t lo = (uint32_t)(((uint64_t)n * prog.rank) / nr);
        const uint32_t hi = (uint32_t)(((uint64_t)n * (prog.rank + 1)) / nr);
        xr_publish_range(prog, slot, 0, lo, tid, stride);
        xr_publish_range(prog, slot, hi, n, tid, stride);
        uint32_t* status = xr_status(prog, in);
        if (!xr_staged_barrier(prog, xoff, in->i[3], status, nblk)) return;
        auto* out = (__nv_bfloat16*)T[in->t[0]] + e0;
        const auto* resid = in->t[1] == PLOW_TENSOR_NONE ? nullptr
            : (const __nv_bfloat16*)T[in->t[1]] + e0;
        auto* out2 = in->t[2] == PLOW_TENSOR_NONE ? nullptr : (__nv_bfloat16*)T[in->t[2]] + e0;
        xr_staged_reduce_range<true>(prog, slot, lo, hi, out, resid, out2, tid, stride);
        if (!xr_staged_barrier(prog, xoff, in->i[4], status, nblk)) return;
        for (uint32_t i = 0; i < nr; i++) {
            const uint32_t r = (slice + prog.rank + i) % nr;
            if (r == prog.rank) continue;
            const uint32_t rlo = (uint32_t)(((uint64_t)n * r) / nr);
            const uint32_t rhi = (uint32_t)(((uint64_t)n * (r + 1)) / nr);
            const auto* src = (const __nv_bfloat16*)((const char*)prog.peer_scratch[r] + slot);
            xr_gather_range(src, slot, rlo, rhi, out, resid, out2, tid, stride);
        }
        return;
    }
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
    xr_reduce_range<true>(prog, slot, lo, hi, mine, tid, stride);
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
        xr_gather_range(src, slot, slo, shi, out, resid, out2, tid, stride);
    }
}
