/* Causal flash prefill at hd 128 (GQA, NoPE or pre-roped Q/K) for the sm_120 interpreter, FA2
 * form: 8 warps x 16 query rows of one head per 128-row item, Q fragments, scores and P in
 * registers (the m16n8k16 accumulator layout is the next mma's A layout), online softmax per row
 * in registers (quad shuffles), K/V through a 2-stage cp.async ring of 64-key tiles. One barrier
 * per key tile. Items run heaviest first (last query tile first), heads fastest, so the heads of a
 * KV group stream the same K/V tiles together (L2 reuse across blocks).
 *
 * Same operands and math contract as d_flash_prefill (bf16 Q/K/V, f32 scores and accumulation,
 * exp2 with the log2e-folded scale, P rounded to bf16 for P.V, l summed from the f32 P, bf16 O);
 * window 0, nsplit 1 and the fused epilogue only. `req` is the packed chunk's request table
 * ([R, {q0, qlen, slot, kvlen}]), nullptr for one request. */
#pragma once

#define FP2_HD 128
#define FP2_BQ 128
#define FP2_BKV 64
#define FP2_LD (FP2_HD + 8) /* 272 B rows: conflict-free ldmatrix phases */
#define FP2_SMEM_BYTES (4 * FP2_BKV * FP2_LD * 2)
/* KV8: + one raw E4M3 K/V tile behind the bf16 ring. */
#define FP2_SMEM8_BYTES (FP2_SMEM_BYTES + 2 * FP2_BKV * FP2_HD)

static __device__ __forceinline__ void fp2_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0,
                                               unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void fp2_ldsm_x4(unsigned (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
static __device__ __forceinline__ void fp2_ldsm_x4_t(unsigned (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
static __device__ __forceinline__ void fp2_cp16(void* smem, const void* g, int bytes) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(bytes));
}
static __device__ __forceinline__ unsigned fp2_pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&v);
}
/* Two E4M3 bytes as bf16x2 (exact: e4m3 -> f16 -> f32 -> bf16). */
static __device__ __forceinline__ unsigned fp2_e4m3x2_bf16(unsigned two) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xffffu), __NV_E4M3);
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
    return fp2_pack(f.x, f.y);
}
static __device__ __forceinline__ float fp2_ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

/* FP2_INLINE: call form inside the interpreter (A/B: __noinline__ own allocation vs inlined). */
#ifndef FP2_INLINE
#define FP2_INLINE __noinline__
#endif
/* KV8: K / V hold E4M3 bytes at unit scale (PLOW_NV_KV_E4M3, same element offsets, one byte
 * each), FP2_SMEM8_BYTES of arena: tile t + 1's bytes land by cp.async in a raw tile behind the
 * ring during tile t, and each thread dequantizes (exactly) into the bf16 ring only the 16-byte
 * chunks it copied itself, so no barrier sits between the copy and the read; one chunk per P.V
 * k-step of tile t keeps the conversion under the mma stream. The math below is unchanged. */
template <int UNUSED = 0, bool KV8 = false>
static __device__ FP2_INLINE void d_flash_pf128(const int* __restrict__ req,
                                                  const __nv_bfloat16* __restrict__ Q,
                                                  const __nv_bfloat16* __restrict__ K,
                                                  const __nv_bfloat16* __restrict__ V,
                                                  __nv_bfloat16* __restrict__ O, unsigned seq_q,
                                                  unsigned seq_kv, unsigned n_head,
                                                  unsigned n_kv_head, unsigned q_pos0,
                                                  unsigned kv_stride, unsigned kv_mask, float scale,
                                                  unsigned slice, unsigned nblk, float* lds) {
    constexpr int HD = FP2_HD, BQ = FP2_BQ, BKV = FP2_BKV, LD = FP2_LD;
    if (blockDim.x != 256u || n_kv_head == 0u || n_head % n_kv_head) {
        __trap();
        return;
    }
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(lds); /* [2][BKV][LD] */
    __nv_bfloat16* Vs = Ks + 2 * BKV * LD;                     /* [2][BKV][LD] */
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t4 = lane & 3;
    const unsigned gqa = n_head / n_kv_head;
    const float lscale = scale * 1.4426950408889634f;
    const int nreq = req ? req[0] : 1;

    unsigned n_work = 0;
    for (int r = 0; r < nreq; r++) {
        const int qlen = req ? req[2 + 4 * r] : (int)seq_q;
        if (qlen > 0) n_work += (unsigned)((qlen + BQ - 1) / BQ) * n_head;
    }

    for (unsigned w = slice; w < n_work; w += nblk) {
        unsigned rem = w, sq = seq_q, skv = seq_kv, qp0 = q_pos0, ntile;
        size_t qoff = 0, kvoff = 0;
        int r = 0;
        for (;;) {
            const int qlen = req ? req[2 + 4 * r] : (int)seq_q;
            ntile = qlen > 0 ? (unsigned)((qlen + BQ - 1) / BQ) : 0u;
            if (rem < ntile * n_head) break;
            rem -= ntile * n_head;
            r++;
        }
        if (req) {
            sq = (unsigned)req[2 + 4 * r];
            skv = (unsigned)req[4 + 4 * r];
            qp0 = skv - sq;
            qoff = (size_t)req[1 + 4 * r] * n_head * HD;
            kvoff = (size_t)req[3 + 4 * r] * n_kv_head * (size_t)kv_stride * HD;
        }
        const unsigned h = rem % n_head;
        const unsigned q0 = (ntile - 1u - rem / n_head) * BQ;
        const unsigned hkv = h / gqa;
        const __nv_bfloat16* Kb = K + kvoff + (size_t)hkv * kv_stride * HD;
        const __nv_bfloat16* Vb = V + kvoff + (size_t)hkv * kv_stride * HD;
        const unsigned kcap = qp0 + q0 + BQ;
        const unsigned kend = kcap < skv ? kcap : skv;
        const int nt = (int)((kend + BKV - 1) / BKV);

        auto stage = [&](int tile, int b) {
            const unsigned kv0 = (unsigned)tile * BKV;
            __nv_bfloat16* kd = Ks + b * BKV * LD;
            __nv_bfloat16* vd = Vs + b * BKV * LD;
#pragma unroll
            for (int i = 0; i < BKV * (HD / 8) / 256; i++) {
                const int L = tid + i * 256, row = L >> 4, c8 = (L & 15) * 8;
                const unsigned kv = kv0 + (unsigned)row;
                const bool in = kv < kend;
                const size_t off = (size_t)(kv & kv_mask) * HD + c8;
                fp2_cp16(kd + row * LD + c8, in ? Kb + off : Kb, in ? 16 : 0);
                fp2_cp16(vd + row * LD + c8, in ? Vb + off : Vb, in ? 16 : 0);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        unsigned char* F8 = reinterpret_cast<unsigned char*>(lds) + FP2_SMEM_BYTES; /* [K | V][BKV * HD] */
        auto stage8 = [&](int tile) {
            const unsigned kv0 = (unsigned)tile * BKV;
            const unsigned char* K8 = reinterpret_cast<const unsigned char*>(K) + (Kb - K);
            const unsigned char* V8 = reinterpret_cast<const unsigned char*>(V) + (Vb - V);
#pragma unroll
            for (int i = 0; i < BKV * HD / 16 / 256; i++) {
                const int L = tid + i * 256;
                const unsigned kv = kv0 + (unsigned)(L >> 3);
                const bool in = kv < kend;
                const size_t off = (size_t)(kv & kv_mask) * HD + (L & 7) * 16;
                fp2_cp16(F8 + L * 16, in ? K8 + off : K8, in ? 16 : 0);
                fp2_cp16(F8 + BKV * HD + L * 16, in ? V8 + off : V8, in ? 16 : 0);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        auto deq8 = [&](int b, int i, bool v) { /* this thread's chunk i of K (v = 0) or V */
            const int L = tid + i * 256, row = L >> 3, c16 = (L & 7) * 16;
            const uint4 x = *reinterpret_cast<const uint4*>(F8 + (v ? BKV * HD : 0) + L * 16);
            __nv_bfloat16* d = (v ? Vs : Ks) + b * BKV * LD + row * LD + c16;
            const uint4 lo = make_uint4(fp2_e4m3x2_bf16(x.x), fp2_e4m3x2_bf16(x.x >> 16),
                                        fp2_e4m3x2_bf16(x.y), fp2_e4m3x2_bf16(x.y >> 16));
            const uint4 hi = make_uint4(fp2_e4m3x2_bf16(x.z), fp2_e4m3x2_bf16(x.z >> 16),
                                        fp2_e4m3x2_bf16(x.w), fp2_e4m3x2_bf16(x.w >> 16));
            /* chunks 4..7 of a row store their upper half first: each 8-thread store phase
             * covers all 32 banks */
            const bool sw = (L & 4) != 0;
            *reinterpret_cast<uint4*>(d + (sw ? 8 : 0)) = sw ? hi : lo;
            *reinterpret_cast<uint4*>(d + (sw ? 0 : 8)) = sw ? lo : hi;
        };

        __syncthreads(); /* the previous item's last tile is read out before the ring restages */
        if (nt > 0) {
            if constexpr (KV8) {
                stage8(0);
                asm volatile("cp.async.wait_group 0;\n" ::);
                deq8(0, 0, false);
                deq8(0, 1, false);
                deq8(0, 0, true);
                deq8(0, 1, true);
            } else {
                stage(0, 0);
            }
        }

        const unsigned ra = q0 + (unsigned)warp * 16u + (unsigned)g, rb = ra + 8u;
        const bool va = ra < sq, vb = rb < sq;
        const __nv_bfloat16* Qa = Q + qoff + ((size_t)ra * n_head + h) * HD;
        const __nv_bfloat16* Qb = Q + qoff + ((size_t)rb * n_head + h) * HD;
        unsigned qf[HD / 16][4];
#pragma unroll
        for (int kf = 0; kf < HD / 16; kf++) {
            const int c = kf * 16 + 2 * t4;
            qf[kf][0] = va ? *reinterpret_cast<const unsigned*>(Qa + c) : 0u;
            qf[kf][1] = vb ? *reinterpret_cast<const unsigned*>(Qb + c) : 0u;
            qf[kf][2] = va ? *reinterpret_cast<const unsigned*>(Qa + c + 8) : 0u;
            qf[kf][3] = vb ? *reinterpret_cast<const unsigned*>(Qb + c + 8) : 0u;
        }
        const unsigned abs_a = qp0 + ra, abs_b = qp0 + rb;
        const unsigned warp_lo = qp0 + q0 + (unsigned)warp * 16u;
        float o[HD / 8][4];
#pragma unroll
        for (int j = 0; j < HD / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
        float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

        for (int t = 0; t < nt; t++) {
            const int buf = t & 1;
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads(); /* tile t landed for all; tile t-1's buffer is free */
            if (t + 1 < nt) {
                if constexpr (KV8) stage8(t + 1);
                else stage(t + 1, buf ^ 1);
            }
            const unsigned kv0 = (unsigned)t * BKV;
            const __nv_bfloat16* Kt = Ks + buf * BKV * LD;
            const __nv_bfloat16* Vt = Vs + buf * BKV * LD;

            float s[BKV / 8][4];
#pragma unroll
            for (int j = 0; j < BKV / 8; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
            for (int kf = 0; kf < HD / 16; kf++) {
#pragma unroll
                for (int np = 0; np < BKV / 16; np++) {
                    unsigned b[4];
                    fp2_ldsm_x4(b, Kt + (np * 16 + (lane & 7) + ((lane >> 4) << 3)) * LD + kf * 16 +
                                       ((lane >> 3) & 1) * 8);
                    fp2_mma(s[2 * np], qf[kf], b[0], b[1]);
                    fp2_mma(s[2 * np + 1], qf[kf], b[2], b[3]);
                }
            }
            if (kv0 + BKV - 1u > warp_lo) { /* causal edge inside this tile for this warp */
#pragma unroll
                for (int j = 0; j < BKV / 8; j++) {
                    const unsigned kv = kv0 + (unsigned)(j * 8 + 2 * t4);
                    if (kv > abs_a) s[j][0] = -INFINITY;
                    if (kv + 1u > abs_a) s[j][1] = -INFINITY;
                    if (kv > abs_b) s[j][2] = -INFINITY;
                    if (kv + 1u > abs_b) s[j][3] = -INFINITY;
                }
            }
            float x0 = s[0][0], x1 = s[0][2];
#pragma unroll
            for (int j = 0; j < BKV / 8; j++) {
                x0 = fmaxf(x0, fmaxf(s[j][0], s[j][1]));
                x1 = fmaxf(x1, fmaxf(s[j][2], s[j][3]));
            }
            x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
            x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
            x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
            x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
            /* tile 0 always holds key 0 <= every row, so m is finite from the first tile on */
            const float n0 = fmaxf(m0, x0 * lscale), n1 = fmaxf(m1, x1 * lscale);
            const float c0 = fp2_ex2(m0 - n0), c1 = fp2_ex2(m1 - n1);
            m0 = n0;
            m1 = n1;
            float p0 = 0.f, p1 = 0.f;
#pragma unroll
            for (int j = 0; j < BKV / 8; j++) {
                s[j][0] = fp2_ex2(fmaf(s[j][0], lscale, -n0));
                s[j][1] = fp2_ex2(fmaf(s[j][1], lscale, -n0));
                s[j][2] = fp2_ex2(fmaf(s[j][2], lscale, -n1));
                s[j][3] = fp2_ex2(fmaf(s[j][3], lscale, -n1));
                p0 += s[j][0] + s[j][1];
                p1 += s[j][2] + s[j][3];
            }
            l0 = l0 * c0 + p0;
            l1 = l1 * c1 + p1;
#pragma unroll
            for (int j = 0; j < HD / 8; j++) {
                o[j][0] *= c0;
                o[j][1] *= c0;
                o[j][2] *= c1;
                o[j][3] *= c1;
            }
#pragma unroll
            for (int kk = 0; kk < BKV / 16; kk++) {
                if constexpr (KV8) {
                    /* every warp left buf ^ 1 at this tile's barrier. Unconditional: on the
                     * item's last tile this rewrites the idle buffer with stale bytes, which
                     * nothing reads before the next item restages it. */
                    if (kk == 0) asm volatile("cp.async.wait_group 0;\n" ::);
                    deq8(buf ^ 1, kk & 1, kk >> 1);
                }
                const unsigned pa[4] = {fp2_pack(s[2 * kk][0], s[2 * kk][1]),
                                        fp2_pack(s[2 * kk][2], s[2 * kk][3]),
                                        fp2_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                        fp2_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
                for (int dp = 0; dp < HD / 16; dp++) {
                    unsigned b[4];
                    fp2_ldsm_x4_t(b, Vt + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD +
                                         dp * 16 + (lane >> 4) * 8);
                    fp2_mma(o[2 * dp], pa, b[0], b[1]);
                    fp2_mma(o[2 * dp + 1], pa, b[2], b[3]);
                }
            }
        }
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
        __nv_bfloat16* Oa = O + qoff + ((size_t)ra * n_head + h) * HD;
        __nv_bfloat16* Ob = O + qoff + ((size_t)rb * n_head + h) * HD;
#pragma unroll
        for (int j = 0; j < HD / 8; j++) {
            const int c = j * 8 + 2 * t4;
            if (va) *reinterpret_cast<unsigned*>(Oa + c) = fp2_pack(o[j][0] * i0, o[j][1] * i0);
            if (vb) *reinterpret_cast<unsigned*>(Ob + c) = fp2_pack(o[j][2] * i1, o[j][3] * i1);
        }
    }
    if (req) { /* packed chunk: rows past the last request are zero (d_flash_prefill_mux contract) */
        const unsigned real = (unsigned)(req[1 + 4 * (nreq - 1)] + req[2 + 4 * (nreq - 1)]);
        const size_t begin = (size_t)real * n_head * HD, end = (size_t)seq_q * n_head * HD;
        for (size_t i = begin + (size_t)slice * blockDim.x + threadIdx.x; i < end;
             i += (size_t)nblk * blockDim.x)
            O[i] = __float2bfloat16(0.0f);
    }
}

