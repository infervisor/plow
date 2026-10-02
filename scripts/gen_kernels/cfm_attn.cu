/* Standalone probe of the S3Gen CFM attention (AttentionF32, 3xTF32, key prefix): the current
 * interpreter path (d_attention_f32) and candidates, behind a C ABI for bench_cfm.py (ctypes).
 *   nvcc -arch=sm_90a -O3 -shared -Xcompiler -fPIC -I runtime/common -I runtime/nvidia \
 *        scripts/gen_kernels/cfm_attn.cu -o cfm_attn.so      [-DPROF: phase clocks, prof_cfm.py]
 * All candidates run as the interpreter would: 132 blocks x 256 threads, <= SP_ARENA_FLOATS smem.
 *   which 0      interpreter path (sp_attention_tc64p)
 *   1-7          tc64x: K/V split once per block into fragment order, register prefetch
 *   8-11         tc64w: 6 compute warps + 2 staging warps, named-barrier ring
 *   12-16        tc64m: all warps compute and stage, mbarrier ring (warps drift)
 *   20-22, 25    h16m: tc64m on 3xFP16 (m16n8k16, power-of-two scaled operands)
 *   23           h16m with the prefix pre-formatted once (k_fmt_prefix) and bulk-copied: the winner
 * H100, us (which 0 -> 23): b128 q136 474 -> 334, b64 q136 268 -> 187, b128 q72 236 -> 186,
 * b64 q72 146 -> 114; rel. L2 vs fp64 1.5e-6 -> 2.7e-7.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include "op_speech_f32.cuh"
#define SP_TEN(k) (in->t[k] == PLOW_TENSOR_NONE ? nullptr : T[in->t[k]])

/* Candidate: tc64p's packed 16-row query tiles, but K/V split hi/lo once per block into
 * fragment-ordered float4s {hi(x0), hi(x1), lo(x0), lo(x1)} (one LDS.128 per three MMAs instead of
 * two LDS + two splits per warp), P fed to P.V from the accumulator layout by permuting each
 * 8-key block's order (keys 2t, 2t+1 -> k t, t+4; V rows staged to match), next tile's K/V
 * prefetched into registers under the MMAs, one barrier per tile. No bias / causal. */
/* Slot swizzles (float4 index within a 32-slot fragment block, lane = g * 4 + t): staging writes
 * of a quarter warp land on distinct banks. K block (j, kk): lane ^ KX(kk); V block (kk, n):
 * lane ^ (n & 1) << 2. */
#ifndef ABL
#define ABL 0
#endif
__device__ unsigned long long g_prof[16];
#ifdef PROF
/* PT(i): time since the previous mark goes to bucket i (warp 0..7 lane 0 of every block). */
#define PROF_DECL long long pt_last = clock64(); unsigned long long pt_acc[12] = {};
#define PT(i) do { const long long now_ = clock64(); pt_acc[i] += now_ - pt_last; pt_last = now_; } while (0)
#define PROF_FLUSH if (lane == 0) for (int i = 0; i < 12; i++) atomicAdd(&g_prof[i], pt_acc[i]);
#else
#define PROF_DECL
#define PT(i)
#define PROF_FLUSH
#endif
#define KX(kk) ((((kk) & 1) << 2) ^ (((kk) >> 1) & 3))
template <int KT, int HS, int QS, int PS, bool LZ>
static __device__ __noinline__ void sp_attention_tc64x(const PlowDevInst* in, void* const* T, unsigned slice,
                                                       unsigned nblk) {
    float* const arena = sp_smem;
    constexpr int HW = 64, NJ = KT / 8, XS = KT * HW * 4; /* floats per head and stage */
    constexpr int KG = HS * KT * 8, VG = HS * KT * 4;     /* K groups (row, 8 cols), V groups (2 rows, 8 cols) */
    constexpr int KI = (KG + PLOW_NV_THREADS - 1) / PLOW_NV_THREADS, VI = (VG + PLOW_NV_THREADS - 1) / PLOW_NV_THREADS;
    float* out = (float*)SP_TEN(0);
    const float* query = (const float*)SP_TEN(1);
    const float* key = (const float*)SP_TEN(2);
    const float* value = (const float*)SP_TEN(3);
    const unsigned* lengths = (const unsigned*)SP_TEN(4);
    const float* prefix = (const float*)SP_TEN(6);
    const unsigned* pidx = (const unsigned*)SP_TEN(7);
    const unsigned pre = prefix ? in->i[7] : 0u;
    const unsigned batch = in->i[0], q_rows = in->i[1], kv_rows = in->i[2], heads = in->i[3];
    const unsigned width = heads * HW, stride = in->i[5] ? in->i[5] : width;
    const unsigned k_col0 = in->fj[1].u, v_col0 = in->fj[2].u;
    const float scale = in->fj[0].f * 1.4426950408889634f;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned n16 = (q_rows + 15u) / 16u, tiles = heads * n16, runs = (tiles + 7u) / 8u;
    for (unsigned item = slice; item < batch * runs; item += nblk) {
        const unsigned b = item / runs, u0 = (item % runs) * 8u, u = u0 + warp;
        const bool live = u < tiles;
        const unsigned h0 = u0 / n16, nh = min(u0 + 7u, tiles - 1u) / n16 - h0 + 1u;
        const unsigned h = live ? u / n16 : h0, q0 = live ? (u % n16) * 16u : 0u;
        const unsigned klen = lengths && lengths[b] < pre + kv_rows ? lengths[b] : pre + kv_rows;
        const unsigned ra = q0 + g, rb = ra + 8u;
        const float* kvsrc = prefix ? prefix + (size_t)pidx[b] * pre * (2u * width) : nullptr;
        /* Source row of key kj (K at +0, V at +vofs). */
        auto krow = [&](unsigned kj, unsigned hh, unsigned& vofs) -> const float* {
            if (kj < pre) {
                vofs = width;
                return kvsrc + (size_t)kj * (2u * width) + hh * HW;
            }
            vofs = v_col0 - k_col0;
            return key + ((size_t)b * kv_rows + kj - pre) * stride + hh * HW + k_col0;
        };
        float4 kr[KI][2], vr[VI][4];
        auto prefetch = [&](unsigned kt) {
            const unsigned k0 = kt * KT;
#pragma unroll
            for (int i = 0; i < KI; i++) {
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / (KT * 8), r = (e / 8) % KT, kk = e % 8;
                kr[i][0] = kr[i][1] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (e < KG && hs < nh && k0 + r < klen) {
                    unsigned vo;
                    const float* p = krow(k0 + r, h0 + hs, vo) + kk * 8;
                    kr[i][0] = *(const float4*)p;
                    kr[i][1] = *(const float4*)(p + 4);
                }
            }
#pragma unroll
            for (int i = 0; i < VI; i++) {
                /* e -> (head, 8-key block kk, column block n, pair t) */
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / (KT * 4), tt = e % 4, n = (e / 4) % 8,
                               kk = (e / 32) % NJ;
#pragma unroll
                for (int q = 0; q < 4; q++) vr[i][q] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (e < VG && hs < nh) {
#pragma unroll
                    for (int s = 0; s < 2; s++) {
                        const unsigned kj = k0 + kk * 8 + 2 * tt + s;
                        if (kj < klen) {
                            unsigned vo;
                            const float* p = krow(kj, h0 + hs, vo) + vo + n * 8;
                            vr[i][2 * s] = *(const float4*)p;
                            vr[i][2 * s + 1] = *(const float4*)(p + 4);
                        }
                    }
                }
            }
        };
        auto store = [&](unsigned stage) {
            float4* X = (float4*)(arena + stage * HS * XS);
#pragma unroll
            for (int i = 0; i < KI; i++) {
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / (KT * 8), r = (e / 8) % KT, kk = e % 8;
                if (e < KG) {
                    const float x[8] = {kr[i][0].x, kr[i][0].y, kr[i][0].z, kr[i][0].w,
                                        kr[i][1].x, kr[i][1].y, kr[i][1].z, kr[i][1].w};
                    float4* d = X + hs * (XS / 4) + ((r / 8) * 8 + kk) * 32 + (((r % 8) * 4u) ^ ((kk & 1u) << 2));
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        unsigned a0, l0, a1, l1;
                        sp_split(x[c], a0, l0);
                        sp_split(x[c + 4], a1, l1);
                        d[c ^ ((kk >> 1) & 3u)] = make_float4(__uint_as_float(a0), __uint_as_float(a1),
                                                             __uint_as_float(l0), __uint_as_float(l1));
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < VI; i++) {
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / (KT * 4), tt = e % 4, n = (e / 4) % 8,
                               kk = (e / 32) % NJ;
                if (e < VG) {
                    const float y0[8] = {vr[i][0].x, vr[i][0].y, vr[i][0].z, vr[i][0].w,
                                         vr[i][1].x, vr[i][1].y, vr[i][1].z, vr[i][1].w};
                    const float y1[8] = {vr[i][2].x, vr[i][2].y, vr[i][2].z, vr[i][2].w,
                                         vr[i][3].x, vr[i][3].y, vr[i][3].z, vr[i][3].w};
                    float4* d = X + hs * (XS / 4) + KT * 32 + (kk * 8 + n) * 32 + tt;
#pragma unroll
                    for (int c = 0; c < 8; c++) {
                        unsigned a0, l0, a1, l1;
                        sp_split(y0[c], a0, l0);
                        sp_split(y1[c], a1, l1);
                        d[(c ^ (n & 1u)) * 4] = make_float4(__uint_as_float(a0), __uint_as_float(a1),
                                                                 __uint_as_float(l0), __uint_as_float(l1));
                    }
                }
            }
        };
        unsigned qh[8][4], ql[8][4];
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const float* qa = query + ((size_t)b * q_rows + ra) * stride + h * HW + ks * 8 + t;
            const float* qb = query + ((size_t)b * q_rows + rb) * stride + h * HW + ks * 8 + t;
            const bool oa = live && ra < q_rows, ob = live && rb < q_rows;
            const float x[4] = {oa ? qa[0] : 0.f, ob ? qb[0] : 0.f, oa ? qa[4] : 0.f, ob ? qb[4] : 0.f};
#pragma unroll
            for (int r = 0; r < 4; r++) sp_split(x[r], qh[ks][r], ql[ks][r]);
        }
        float o[8][4], mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int r = 0; r < 4; r++) o[n][r] = 0.f;
        const unsigned nkt = (klen + KT - 1) / KT;
        prefetch(0);
        store(0);
        for (unsigned kt = 0; kt < nkt; kt++) {
            if (!(ABL & 4) && kt + 1 < nkt) prefetch(kt + 1);
            __syncthreads();
            if (live) {
                const float4* X = (const float4*)(arena + (kt & 1u) * HS * XS + (h - h0) * XS);
                const float4* Vx = X + KT * 32;
                const unsigned k0 = kt * KT;
                /* QS accumulator sets for the three products (independent MMA chains), summed after. */
                float sq[QS][NJ][4];
#pragma unroll
                for (int a = 0; a < QS; a++)
#pragma unroll
                    for (int j = 0; j < NJ; j++)
#pragma unroll
                        for (int r = 0; r < 4; r++) sq[a][j][r] = 0.f;
#pragma unroll
                for (int kk = 0; kk < 8; kk++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        const float4 kv = X[(j * 8 + kk) * 32 + (lane ^ KX(kk))];
                        const unsigned bh0 = __float_as_uint(kv.x), bh1 = __float_as_uint(kv.y),
                                       bl0 = __float_as_uint(kv.z), bl1 = __float_as_uint(kv.w);
                        if (ABL & 1) { sq[0][j][0] += kv.x; sq[0][j][1] += kv.y; sq[0][j][2] += kv.z; sq[0][j][3] += kv.w + __uint_as_float(qh[kk][0] ^ ql[kk][3]); continue; }
                        sp_mma_tf32(sq[QS - 1][j], ql[kk], bh0, bh1);
                        sp_mma_tf32(sq[QS > 2 ? 1 : QS - 1][j], qh[kk], bl0, bl1);
                        sp_mma_tf32(sq[0][j], qh[kk], bh0, bh1);
                    }
                float sc[NJ][4];
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int r = 0; r < 4; r++)
                        sc[j][r] = QS == 1 ? sq[0][j][r]
                                 : QS == 2 ? sq[0][j][r] + sq[QS - 1][j][r]
                                           : sq[0][j][r] + (sq[1][j][r] + sq[QS - 1][j][r]);
                if (LZ) {
                    /* Lazy rescale: p = 2^(s - mref) with mref the row max at the last rescale; o and
                     * the per-thread row sums are rescaled only when a score passes mref + 8. The
                     * shift cancels in the normalization, so this is the same softmax. */
                    float lm[2] = {-INFINITY, -INFINITY};
#pragma unroll
                    for (int j = 0; j < NJ; j++)
#pragma unroll
                        for (int r = 0; r < 4; r++) {
                            const unsigned kj = k0 + j * 8 + 2 * t + (r & 1);
                            const float v = kj < klen ? sc[j][r] * scale : -INFINITY;
                            sc[j][r] = v;
                            lm[r >> 1] = fmaxf(lm[r >> 1], v);
                        }
                    if (__any_sync(0xffffffffu, lm[0] > mrow[0] + 8.f || lm[1] > mrow[1] + 8.f)) {
#pragma unroll
                        for (int i = 0; i < 2; i++) {
                            lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 1));
                            lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 2));
                            if (lm[i] > mrow[i] + 8.f) {
                                const float corr = mrow[i] == -INFINITY ? 0.f : exp2f(mrow[i] - lm[i]);
                                mrow[i] = lm[i];
                                lrow[i] *= corr;
#pragma unroll
                                for (int n = 0; n < 8; n++) {
                                    o[n][2 * i] *= corr;
                                    o[n][2 * i + 1] *= corr;
                                }
                            }
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NJ; j++)
#pragma unroll
                        for (int r = 0; r < 4; r++) {
                            const float p = exp2f(sc[j][r] - mrow[r >> 1]);
                            sc[j][r] = p;
                            lrow[r >> 1] += p;
                        }
                } else {
                    float mt[2] = {-INFINITY, -INFINITY};
    #pragma unroll
                    for (int j = 0; j < NJ; j++)
    #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            const unsigned kj = k0 + j * 8 + 2 * t + (r & 1);
                            const float v = kj < klen ? sc[j][r] * scale : -INFINITY;
                            sc[j][r] = v;
                            mt[r >> 1] = fmaxf(mt[r >> 1], v);
                        }
                    float corr[2], ps[2] = {0.f, 0.f};
    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        mt[i] = fmaxf(mt[i], __shfl_xor_sync(0xffffffffu, mt[i], 1));
                        mt[i] = fmaxf(mt[i], __shfl_xor_sync(0xffffffffu, mt[i], 2));
                        const float mnew = fmaxf(mrow[i], mt[i]);
                        corr[i] = mnew == -INFINITY ? 1.f : exp2f(mrow[i] - mnew);
                        mrow[i] = mnew;
                    }
    #pragma unroll
                    for (int j = 0; j < NJ; j++)
    #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            const float p = (ABL & 8) ? sc[j][r] : sc[j][r] == -INFINITY ? 0.f : exp2f(sc[j][r] - mrow[r >> 1]);
                            sc[j][r] = p;
                            ps[r >> 1] += p;
                        }
    #pragma unroll
                    for (int i = 0; i < 2; i++) {
                        ps[i] += __shfl_xor_sync(0xffffffffu, ps[i], 1);
                        ps[i] += __shfl_xor_sync(0xffffffffu, ps[i], 2);
                        lrow[i] = lrow[i] * corr[i] + ps[i];
                    }
    #pragma unroll
                    for (int n = 0; n < 8; n++)
    #pragma unroll
                        for (int r = 0; r < 4; r++) o[n][r] *= corr[r >> 1];
                }
                /* PS 1: products straight into o; 2: one tile accumulator (the interpreter's order);
                 * 3: hi.hi and the two lo products in separate tile accumulators. */
                constexpr int PA = PS == 1 ? 1 : PS - 1;
                float pv[PA][8][4];
                if (PS > 1) {
#pragma unroll
                    for (int a = 0; a < PA; a++)
#pragma unroll
                        for (int n = 0; n < 8; n++)
#pragma unroll
                            for (int r = 0; r < 4; r++) pv[a][n][r] = 0.f;
                }
#pragma unroll
                for (int kk = 0; kk < NJ; kk++) {
                    /* k t <- key 2t, k t+4 <- key 2t+1 (V staged in that order). */
                    const float x[4] = {sc[kk][0], sc[kk][2], sc[kk][1], sc[kk][3]};
                    unsigned ah[4], al[4];
#pragma unroll
                    for (int r = 0; r < 4; r++) sp_split(x[r], ah[r], al[r]);
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        const float4 vv = Vx[(kk * 8 + n) * 32 + (lane ^ ((n & 1) << 2))];
                        const unsigned bh0 = __float_as_uint(vv.x), bh1 = __float_as_uint(vv.y),
                                       bl0 = __float_as_uint(vv.z), bl1 = __float_as_uint(vv.w);
                        float(&dl)[4] = PS == 1 ? o[n] : pv[PA - 1][n];
                        float(&dh)[4] = PS == 1 ? o[n] : pv[0][n];
                        if (ABL & 2) { dh[0] += vv.x * x[0]; dh[1] += vv.y; dh[2] += vv.z; dh[3] += vv.w + x[3]; continue; }
                        sp_mma_tf32(dl, al, bh0, bh1);
                        sp_mma_tf32(dl, ah, bl0, bl1);
                        sp_mma_tf32(dh, ah, bh0, bh1);
                    }
                }
                if (PS > 1) {
#pragma unroll
                    for (int n = 0; n < 8; n++)
#pragma unroll
                        for (int r = 0; r < 4; r++)
                            o[n][r] += PS == 2 ? pv[0][n][r] : pv[0][n][r] + pv[PA - 1][n][r];
                }
            }
            if (!(ABL & 4) && kt + 1 < nkt) store((kt + 1) & 1u);
        }
        if (LZ) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                lrow[i] += __shfl_xor_sync(0xffffffffu, lrow[i], 1);
                lrow[i] += __shfl_xor_sync(0xffffffffu, lrow[i], 2);
            }
        }
        __syncthreads();
        if (!live) continue;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const unsigned r = i ? rb : ra;
            if (r >= q_rows) continue;
            const float inv = lrow[i] > 0.f ? 1.0f / lrow[i] : 0.f;
            float* orow = out + ((size_t)b * q_rows + r) * width + h * HW;
#pragma unroll
            for (int n = 0; n < 8; n++)
                *(float2*)(orow + n * 8 + 2 * t) = make_float2(o[n][2 * i] * inv, o[n][2 * i + 1] * inv);
        }
    }
}

/* Warp-specialized variant: warps NC.. stage K/V (load, split hi/lo, fragment order) into a
 * 3-deep ring; warps 0..NC-1 each own one 16-row query tile of a run of NC tiles (at most 2
 * heads) and never sync with each other, so one warp's softmax overlaps another's MMAs. Named
 * barriers 1-3 (stage full) and 4-6 (stage empty); every arrival is matched before return. */
__device__ __forceinline__ void sp_bar_sync(unsigned id, unsigned n) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void sp_bar_arrive(unsigned id, unsigned n) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
template <int KT, int NC, int QS, int PS, bool PD = false>
static __device__ __noinline__ void sp_attention_tc64w(const PlowDevInst* in, void* const* T, unsigned slice,
                                                       unsigned nblk) {
    float* const arena = sp_smem;
    constexpr int HW = 64, NJ = KT / 8, HS = 2, XS = KT * HW * 4, NS = 3;
    constexpr int NP = PLOW_NV_THREADS - NC * 32; /* producer threads */
    constexpr int KG = HS * KT * 8, VG = HS * KT * 4;
    constexpr int KI = (KG + NP - 1) / NP, VI = (VG + NP - 1) / NP;
    static_assert(NS * HS * XS <= SP_ARENA_FLOATS, "ws attention stages");
    float* out = (float*)SP_TEN(0);
    const float* query = (const float*)SP_TEN(1);
    const float* key = (const float*)SP_TEN(2);
    const unsigned* lengths = (const unsigned*)SP_TEN(4);
    const float* prefix = (const float*)SP_TEN(6);
    const unsigned* pidx = (const unsigned*)SP_TEN(7);
    const unsigned pre = prefix ? in->i[7] : 0u;
    const unsigned batch = in->i[0], q_rows = in->i[1], kv_rows = in->i[2], heads = in->i[3];
    const unsigned width = heads * HW, stride = in->i[5] ? in->i[5] : width;
    const unsigned k_col0 = in->fj[1].u, v_col0 = in->fj[2].u;
    const float scale = in->fj[0].f * 1.4426950408889634f;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned n16 = (q_rows + 15u) / 16u, tiles = heads * n16, runs = (tiles + NC - 1u) / NC;
    unsigned c = 0; /* tiles staged / consumed so far (ring position) */
    if (warp >= NC) {
      if constexpr (!PD) {
        const unsigned pt = tid - NC * 32;
        for (unsigned item = slice; item < batch * runs; item += nblk) {
            const unsigned b = item / runs, u0 = (item % runs) * NC;
            const unsigned h0 = u0 / n16, nh = min(u0 + NC - 1u, tiles - 1u) / n16 - h0 + 1u;
            const unsigned klen = lengths && lengths[b] < pre + kv_rows ? lengths[b] : pre + kv_rows;
            const float* kvsrc = prefix ? prefix + (size_t)pidx[b] * pre * (2u * width) : nullptr;
            const unsigned nkt = (klen + KT - 1) / KT;
            for (unsigned kt = 0; kt < nkt; kt++, c++) {
                const unsigned s = c % NS, k0 = kt * KT;
                float4 kr[KI][2], vr[VI][4];
                if (ABL & 32) { if (c >= NS) sp_bar_sync(4 + s, PLOW_NV_THREADS); sp_bar_arrive(1 + s, PLOW_NV_THREADS); continue; }
#pragma unroll
                for (int i = 0; i < KI; i++) {
                    const unsigned e = pt + i * NP, hs = e / (KT * 8), r = (e / 8) % KT, kk = e % 8, kj = k0 + r;
                    kr[i][0] = kr[i][1] = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (e < KG && hs < nh && kj < klen) {
                        const float* p = kj < pre ? kvsrc + (size_t)kj * (2u * width) + (h0 + hs) * HW + kk * 8
                                                  : key + ((size_t)b * kv_rows + kj - pre) * stride + k_col0 +
                                                        (h0 + hs) * HW + kk * 8;
                        kr[i][0] = __ldg((const float4*)p);
                        kr[i][1] = __ldg((const float4*)(p + 4));
                    }
                }
#pragma unroll
                for (int i = 0; i < VI; i++) {
                    const unsigned e = pt + i * NP, hs = e / (KT * 4), tt = e % 4, n = (e / 4) % 8,
                                   kk = (e / 32) % NJ;
#pragma unroll
                    for (int q = 0; q < 4; q++) vr[i][q] = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (e < VG && hs < nh) {
#pragma unroll
                        for (int s2 = 0; s2 < 2; s2++) {
                            const unsigned kj = k0 + kk * 8 + 2 * tt + s2;
                            if (kj < klen) {
                                const float* p = kj < pre ? kvsrc + (size_t)kj * (2u * width) + width + (h0 + hs) * HW + n * 8
                                                          : key + ((size_t)b * kv_rows + kj - pre) * stride + v_col0 +
                                                                (h0 + hs) * HW + n * 8;
                                vr[i][2 * s2] = __ldg((const float4*)p);
                                vr[i][2 * s2 + 1] = __ldg((const float4*)(p + 4));
                            }
                        }
                    }
                }
                if (c >= NS) sp_bar_sync(4 + s, PLOW_NV_THREADS);
                float4* X = (float4*)(arena + s * HS * XS);
#pragma unroll
                for (int i = 0; i < KI; i++) {
                    const unsigned e = pt + i * NP, hs = e / (KT * 8), r = (e / 8) % KT, kk = e % 8;
                    if (e < KG) {
                        const float x[8] = {kr[i][0].x, kr[i][0].y, kr[i][0].z, kr[i][0].w,
                                            kr[i][1].x, kr[i][1].y, kr[i][1].z, kr[i][1].w};
                        float4* d = X + hs * (XS / 4) + ((r / 8) * 8 + kk) * 32 + (((r % 8) * 4u) ^ ((kk & 1u) << 2));
#pragma unroll
                        for (int cc = 0; cc < 4; cc++) {
                            unsigned a0, l0, a1, l1;
                            sp_split(x[cc], a0, l0);
                            sp_split(x[cc + 4], a1, l1);
                            d[cc ^ ((kk >> 1) & 3u)] = make_float4(__uint_as_float(a0), __uint_as_float(a1),
                                                                  __uint_as_float(l0), __uint_as_float(l1));
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < VI; i++) {
                    const unsigned e = pt + i * NP, hs = e / (KT * 4), tt = e % 4, n = (e / 4) % 8,
                                   kk = (e / 32) % NJ;
                    if (e < VG) {
                        const float y0[8] = {vr[i][0].x, vr[i][0].y, vr[i][0].z, vr[i][0].w,
                                             vr[i][1].x, vr[i][1].y, vr[i][1].z, vr[i][1].w};
                        const float y1[8] = {vr[i][2].x, vr[i][2].y, vr[i][2].z, vr[i][2].w,
                                             vr[i][3].x, vr[i][3].y, vr[i][3].z, vr[i][3].w};
                        float4* d = X + hs * (XS / 4) + KT * 32 + (kk * 8 + n) * 32 + tt;
#pragma unroll
                        for (int cc = 0; cc < 8; cc++) {
                            unsigned a0, l0, a1, l1;
                            sp_split(y0[cc], a0, l0);
                            sp_split(y1[cc], a1, l1);
                            d[(cc ^ (n & 1u)) * 4] = make_float4(__uint_as_float(a0), __uint_as_float(a1),
                                                                 __uint_as_float(l0), __uint_as_float(l1));
                        }
                    }
                }
                sp_bar_arrive(1 + s, PLOW_NV_THREADS);
            }
        }
      } else {
        /* Producer: tile n + 1's loads are in flight while tile n is split and stored. */
        const unsigned pt = tid - NC * 32;
        struct It {
            unsigned item, kt, b, h0, nh, klen, nkt;
            const float* kvsrc;
        };
        auto setup = [&](It& x) {
            for (; x.item < batch * runs; x.item += nblk) {
                x.b = x.item / runs;
                const unsigned u0 = (x.item % runs) * NC;
                x.h0 = u0 / n16;
                x.nh = min(u0 + NC - 1u, tiles - 1u) / n16 - x.h0 + 1u;
                x.klen = lengths && lengths[x.b] < pre + kv_rows ? lengths[x.b] : pre + kv_rows;
                x.kvsrc = prefix ? prefix + (size_t)pidx[x.b] * pre * (2u * width) : nullptr;
                x.nkt = (x.klen + KT - 1) / KT;
                if (x.nkt) return true;
            }
            return false;
        };
        auto load = [&](const It& x, float4 (&kr)[KI][2], float4 (&vr)[VI][4]) {
            const unsigned k0 = x.kt * KT;
#pragma unroll
            for (int i = 0; i < KI; i++) {
                const unsigned e = pt + i * NP, hs = e / (KT * 8), r = (e / 8) % KT, kk = e % 8, kj = k0 + r;
                kr[i][0] = kr[i][1] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (e < KG && hs < x.nh && kj < x.klen) {
                    const float* p = kj < pre ? x.kvsrc + (size_t)kj * (2u * width) + (x.h0 + hs) * HW + kk * 8
                                              : key + ((size_t)x.b * kv_rows + kj - pre) * stride + k_col0 +
                                                    (x.h0 + hs) * HW + kk * 8;
                    kr[i][0] = __ldg((const float4*)p);
                    kr[i][1] = __ldg((const float4*)(p + 4));
                }
            }
#pragma unroll
            for (int i = 0; i < VI; i++) {
                const unsigned e = pt + i * NP, hs = e / (KT * 4), tt = e % 4, n = (e / 4) % 8, kk = (e / 32) % NJ;
#pragma unroll
                for (int q = 0; q < 4; q++) vr[i][q] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (e < VG && hs < x.nh) {
#pragma unroll
                    for (int s2 = 0; s2 < 2; s2++) {
                        const unsigned kj = k0 + kk * 8 + 2 * tt + s2;
                        if (kj < x.klen) {
                            const float* p = kj < pre ? x.kvsrc + (size_t)kj * (2u * width) + width + (x.h0 + hs) * HW + n * 8
                                                      : key + ((size_t)x.b * kv_rows + kj - pre) * stride + v_col0 +
                                                            (x.h0 + hs) * HW + n * 8;
                            vr[i][2 * s2] = __ldg((const float4*)p);
                            vr[i][2 * s2 + 1] = __ldg((const float4*)(p + 4));
                        }
                    }
                }
            }
        };
        auto store = [&](unsigned s, const float4 (&kr)[KI][2], const float4 (&vr)[VI][4]) {
            float4* X = (float4*)(arena + s * HS * XS);
#pragma unroll
            for (int i = 0; i < KI; i++) {
                const unsigned e = pt + i * NP, hs = e / (KT * 8), r = (e / 8) % KT, kk = e % 8;
                if (e < KG) {
                    const float x[8] = {kr[i][0].x, kr[i][0].y, kr[i][0].z, kr[i][0].w,
                                        kr[i][1].x, kr[i][1].y, kr[i][1].z, kr[i][1].w};
                    float4* d = X + hs * (XS / 4) + ((r / 8) * 8 + kk) * 32 + (((r % 8) * 4u) ^ ((kk & 1u) << 2));
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) {
                        unsigned a0, l0, a1, l1;
                        sp_split(x[cc], a0, l0);
                        sp_split(x[cc + 4], a1, l1);
                        d[cc ^ ((kk >> 1) & 3u)] = make_float4(__uint_as_float(a0), __uint_as_float(a1),
                                                              __uint_as_float(l0), __uint_as_float(l1));
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < VI; i++) {
                const unsigned e = pt + i * NP, hs = e / (KT * 4), tt = e % 4, n = (e / 4) % 8, kk = (e / 32) % NJ;
                if (e < VG) {
                    const float y0[8] = {vr[i][0].x, vr[i][0].y, vr[i][0].z, vr[i][0].w,
                                         vr[i][1].x, vr[i][1].y, vr[i][1].z, vr[i][1].w};
                    const float y1[8] = {vr[i][2].x, vr[i][2].y, vr[i][2].z, vr[i][2].w,
                                         vr[i][3].x, vr[i][3].y, vr[i][3].z, vr[i][3].w};
                    float4* d = X + hs * (XS / 4) + KT * 32 + (kk * 8 + n) * 32 + tt;
#pragma unroll
                    for (int cc = 0; cc < 8; cc++) {
                        unsigned a0, l0, a1, l1;
                        sp_split(y0[cc], a0, l0);
                        sp_split(y1[cc], a1, l1);
                        d[(cc ^ (n & 1u)) * 4] = make_float4(__uint_as_float(a0), __uint_as_float(a1),
                                                             __uint_as_float(l0), __uint_as_float(l1));
                    }
                }
            }
        };
        auto advance = [&](It& x) {
            if (++x.kt < x.nkt) return true;
            x.kt = 0;
            x.item += nblk;
            return setup(x);
        };
        float4 ka[KI][2], va[VI][4], kb[KI][2], vb[VI][4];
        It x{slice, 0};
        bool more = setup(x);
        if (more) load(x, ka, va);
        auto step = [&](float4 (&kc)[KI][2], float4 (&vc)[VI][4], float4 (&kn)[KI][2], float4 (&vn)[VI][4]) {
            more = advance(x);
            if (more && !(ABL & 32)) load(x, kn, vn);
            const unsigned s = c % NS;
            if (c >= NS) sp_bar_sync(4 + s, PLOW_NV_THREADS);
            if (!(ABL & 32)) store(s, kc, vc);
            sp_bar_arrive(1 + s, PLOW_NV_THREADS);
            c++;
        };
        while (more) {
            step(ka, va, kb, vb);
            if (!more) break;
            step(kb, vb, ka, va);
        }
      }
        /* Match the consumers' last empty arrivals. */
        for (unsigned k = c > NS ? c - NS : 0; k < c; k++) sp_bar_sync(4 + k % NS, PLOW_NV_THREADS);
        return;
    }
    for (unsigned item = slice; item < batch * runs; item += nblk) {
        const unsigned b = item / runs, u0 = (item % runs) * NC, u = u0 + warp;
        const bool live = u < tiles;
        const unsigned h0 = u0 / n16;
        const unsigned h = live ? u / n16 : h0, q0 = live ? (u % n16) * 16u : 0u;
        const unsigned klen = lengths && lengths[b] < pre + kv_rows ? lengths[b] : pre + kv_rows;
        const unsigned ra = q0 + g, rb = ra + 8u;
        unsigned qh[8][4], ql[8][4];
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const float* qa = query + ((size_t)b * q_rows + ra) * stride + h * HW + ks * 8 + t;
            const float* qb = query + ((size_t)b * q_rows + rb) * stride + h * HW + ks * 8 + t;
            const bool oa = live && ra < q_rows, ob = live && rb < q_rows;
            const float x[4] = {oa ? __ldg(qa) : 0.f, ob ? __ldg(qb) : 0.f, oa ? __ldg(qa + 4) : 0.f,
                                ob ? __ldg(qb + 4) : 0.f};
#pragma unroll
            for (int r = 0; r < 4; r++) sp_split(x[r], qh[ks][r], ql[ks][r]);
        }
        float o[8][4], mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int r = 0; r < 4; r++) o[n][r] = 0.f;
        const unsigned nkt = (klen + KT - 1) / KT;
        for (unsigned kt = 0; kt < nkt; kt++, c++) {
            const unsigned s = c % NS;
            sp_bar_sync(1 + s, PLOW_NV_THREADS);
            if (live && !(ABL & 16)) {
                const float4* X = (const float4*)(arena + s * HS * XS + (h - h0) * XS);
                const float4* Vx = X + KT * 32;
                const unsigned k0 = kt * KT;
                float sq[QS][NJ][4];
#pragma unroll
                for (int a = 0; a < QS; a++)
#pragma unroll
                    for (int j = 0; j < NJ; j++)
#pragma unroll
                        for (int r = 0; r < 4; r++) sq[a][j][r] = 0.f;
#pragma unroll
                for (int kk = 0; kk < 8; kk++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        const float4 kv = X[(j * 8 + kk) * 32 + (lane ^ KX(kk))];
                        const unsigned bh0 = __float_as_uint(kv.x), bh1 = __float_as_uint(kv.y),
                                       bl0 = __float_as_uint(kv.z), bl1 = __float_as_uint(kv.w);
                        sp_mma_tf32(sq[QS - 1][j], ql[kk], bh0, bh1);
                        sp_mma_tf32(sq[QS > 2 ? 1 : QS - 1][j], qh[kk], bl0, bl1);
                        sp_mma_tf32(sq[0][j], qh[kk], bh0, bh1);
                    }
                float sc[NJ][4];
                float lm[2] = {-INFINITY, -INFINITY};
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        const float v0 = QS == 1 ? sq[0][j][r]
                                       : QS == 2 ? sq[0][j][r] + sq[QS - 1][j][r]
                                                 : sq[0][j][r] + (sq[1][j][r] + sq[QS - 1][j][r]);
                        const unsigned kj = k0 + j * 8 + 2 * t + (r & 1);
                        const float v = kj < klen ? v0 * scale : -INFINITY;
                        sc[j][r] = v;
                        lm[r >> 1] = fmaxf(lm[r >> 1], v);
                    }
                /* Lazy rescale (see sp_attention_tc64x). */
                if (__any_sync(0xffffffffu, lm[0] > mrow[0] + 8.f || lm[1] > mrow[1] + 8.f)) {
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 1));
                        lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 2));
                        if (lm[i] > mrow[i] + 8.f) {
                            const float corr = mrow[i] == -INFINITY ? 0.f : exp2f(mrow[i] - lm[i]);
                            mrow[i] = lm[i];
                            lrow[i] *= corr;
#pragma unroll
                            for (int n = 0; n < 8; n++) {
                                o[n][2 * i] *= corr;
                                o[n][2 * i + 1] *= corr;
                            }
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int r = 0; r < 4; r++) {
                        const float p = exp2f(sc[j][r] - mrow[r >> 1]);
                        sc[j][r] = p;
                        lrow[r >> 1] += p;
                    }
                constexpr int PA = PS == 1 ? 1 : PS - 1;
                float pv[PA][8][4];
                if (PS > 1) {
#pragma unroll
                    for (int a = 0; a < PA; a++)
#pragma unroll
                        for (int n = 0; n < 8; n++)
#pragma unroll
                            for (int r = 0; r < 4; r++) pv[a][n][r] = 0.f;
                }
#pragma unroll
                for (int kk = 0; kk < NJ; kk++) {
                    const float x[4] = {sc[kk][0], sc[kk][2], sc[kk][1], sc[kk][3]};
                    unsigned ah[4], al[4];
#pragma unroll
                    for (int r = 0; r < 4; r++) sp_split(x[r], ah[r], al[r]);
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        const float4 vv = Vx[(kk * 8 + n) * 32 + (lane ^ ((n & 1) << 2))];
                        const unsigned bh0 = __float_as_uint(vv.x), bh1 = __float_as_uint(vv.y),
                                       bl0 = __float_as_uint(vv.z), bl1 = __float_as_uint(vv.w);
                        float(&dl)[4] = PS == 1 ? o[n] : pv[PA - 1][n];
                        float(&dh)[4] = PS == 1 ? o[n] : pv[0][n];
                        sp_mma_tf32(dl, al, bh0, bh1);
                        sp_mma_tf32(dl, ah, bl0, bl1);
                        sp_mma_tf32(dh, ah, bh0, bh1);
                    }
                }
                if (PS > 1) {
#pragma unroll
                    for (int n = 0; n < 8; n++)
#pragma unroll
                        for (int r = 0; r < 4; r++)
                            o[n][r] += PS == 2 ? pv[0][n][r] : pv[0][n][r] + pv[PA - 1][n][r];
                }
            }
            sp_bar_arrive(4 + s, PLOW_NV_THREADS);
        }
        if (!live) continue;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            lrow[i] += __shfl_xor_sync(0xffffffffu, lrow[i], 1);
            lrow[i] += __shfl_xor_sync(0xffffffffu, lrow[i], 2);
            const unsigned r = i ? rb : ra;
            if (r >= q_rows) continue;
            const float inv = lrow[i] > 0.f ? 1.0f / lrow[i] : 0.f;
            float* orow = out + ((size_t)b * q_rows + r) * width + h * HW;
#pragma unroll
            for (int n = 0; n < 8; n++)
                *(float2*)(orow + n * 8 + 2 * t) = make_float2(o[n][2 * i] * inv, o[n][2 * i + 1] * inv);
        }
    }
}

/* All 8 warps compute (one 16-row query tile each, runs of 8) and stage K/V: each thread loads its
 * share of tile kt + NS - 1 into registers before computing tile kt and splits/stores it after,
 * through an NS-deep ring whose full/empty handoffs are mbarriers, so warps drift apart instead of
 * meeting at a block barrier every tile. */
__device__ __forceinline__ void sp_mma_tf32_z(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}
__device__ __forceinline__ unsigned sp_su32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void sp_mbar_init(uint64_t* b, unsigned n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sp_su32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void sp_mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sp_su32(b)) : "memory");
}
__device__ __forceinline__ void sp_mbar_arrive_relaxed(uint64_t* b) {
    asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" ::"r"(sp_su32(b)) : "memory");
}
__device__ __forceinline__ void sp_mbar_wait(uint64_t* b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}" ::"r"(
                     sp_su32(b)),
                 "r"(parity)
                 : "memory");
}
template <int KT, int HS, int NS, int QS, int PS>
static __device__ __noinline__ void sp_attention_tc64m(const PlowDevInst* in, void* const* T, unsigned slice,
                                                       unsigned nblk) {
    float* const arena = sp_smem;
    constexpr int HW = 64, NJ = KT / 8, XS = KT * HW * 4;
    constexpr int KG = HS * KT * 8, VG = HS * KT * 4;
    constexpr int KI = (KG + PLOW_NV_THREADS - 1) / PLOW_NV_THREADS, VI = (VG + PLOW_NV_THREADS - 1) / PLOW_NV_THREADS;
    static_assert(NS * HS * XS + 4 * NS <= SP_ARENA_FLOATS, "mbar attention stages");
    uint64_t* full = (uint64_t*)(arena + NS * HS * XS);
    uint64_t* empty = full + NS;
    float* out = (float*)SP_TEN(0);
    const float* query = (const float*)SP_TEN(1);
    const float* key = (const float*)SP_TEN(2);
    const unsigned* lengths = (const unsigned*)SP_TEN(4);
    const float* prefix = (const float*)SP_TEN(6);
    const unsigned* pidx = (const unsigned*)SP_TEN(7);
    const unsigned pre = prefix ? in->i[7] : 0u;
    const unsigned batch = in->i[0], q_rows = in->i[1], kv_rows = in->i[2], heads = in->i[3];
    const unsigned width = heads * HW, stride = in->i[5] ? in->i[5] : width;
    const unsigned k_col0 = in->fj[1].u, v_col0 = in->fj[2].u;
    const float scale = in->fj[0].f * 1.4426950408889634f;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned n16 = (q_rows + 15u) / 16u, tiles = heads * n16, runs = (tiles + 7u) / 8u;
    if (tid == 0)
        for (int s = 0; s < NS; s++) {
            sp_mbar_init(full + s, PLOW_NV_THREADS);
            sp_mbar_init(empty + s, PLOW_NV_THREADS);
        }
    __syncthreads();
    unsigned c = 0; /* ring position of this item's tile 0 */
    PROF_DECL
    for (unsigned item = slice; item < batch * runs; item += nblk) {
        const unsigned b = item / runs, u0 = (item % runs) * 8u, u = u0 + warp;
        const bool live = u < tiles;
        const unsigned h0 = u0 / n16, nh = min(u0 + 7u, tiles - 1u) / n16 - h0 + 1u;
        const unsigned h = live ? u / n16 : h0, q0 = live ? (u % n16) * 16u : 0u;
        const unsigned klen = lengths && lengths[b] < pre + kv_rows ? lengths[b] : pre + kv_rows;
        const unsigned ra = q0 + g, rb = ra + 8u;
        const float* kvsrc = prefix ? prefix + (size_t)pidx[b] * pre * (2u * width) : nullptr;
        const unsigned nkt = (klen + KT - 1) / KT;
        float4 kr[KI][2], vr[VI][4];
        auto load = [&](unsigned kt) {
            const unsigned k0 = kt * KT;
#pragma unroll
            for (int i = 0; i < KI; i++) {
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / (KT * 8), r = (e / 8) % KT, kk = e % 8,
                               kj = k0 + r;
                kr[i][0] = kr[i][1] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (e < KG && hs < nh && kj < klen) {
                    const float* p = kj < pre ? kvsrc + (size_t)kj * (2u * width) + (h0 + hs) * HW + kk * 8
                                              : key + ((size_t)b * kv_rows + kj - pre) * stride + k_col0 +
                                                    (h0 + hs) * HW + kk * 8;
                    kr[i][0] = __ldg((const float4*)p);
                    kr[i][1] = __ldg((const float4*)(p + 4));
                }
            }
#pragma unroll
            for (int i = 0; i < VI; i++) {
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / (KT * 4), tt = e % 4, n = (e / 4) % 8,
                               kk = (e / 32) % NJ;
#pragma unroll
                for (int q = 0; q < 4; q++) vr[i][q] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (e < VG && hs < nh) {
#pragma unroll
                    for (int s2 = 0; s2 < 2; s2++) {
                        const unsigned kj = k0 + kk * 8 + 2 * tt + s2;
                        if (kj < klen) {
                            const float* p = kj < pre ? kvsrc + (size_t)kj * (2u * width) + width + (h0 + hs) * HW + n * 8
                                                      : key + ((size_t)b * kv_rows + kj - pre) * stride + v_col0 +
                                                            (h0 + hs) * HW + n * 8;
                            vr[i][2 * s2] = __ldg((const float4*)p);
                            vr[i][2 * s2 + 1] = __ldg((const float4*)(p + 4));
                        }
                    }
                }
            }
        };
        /* Split and store tile at ring position r (waits for its previous use to be consumed). */
        auto store = [&](unsigned r) {
            const unsigned s = r % NS;
            if (r >= NS) sp_mbar_wait(empty + s, (r / NS - 1) & 1u);
            float4* X = (float4*)(arena + s * HS * XS);
#pragma unroll
            for (int i = 0; i < KI; i++) {
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / (KT * 8), rr = (e / 8) % KT, kk = e % 8;
                if (e < KG) {
                    const float x[8] = {kr[i][0].x, kr[i][0].y, kr[i][0].z, kr[i][0].w,
                                        kr[i][1].x, kr[i][1].y, kr[i][1].z, kr[i][1].w};
                    float4* d = X + hs * (XS / 4) + ((rr / 8) * 8 + kk) * 32 + (((rr % 8) * 4u) ^ ((kk & 1u) << 2));
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) {
                        unsigned a0, l0, a1, l1;
                        sp_split(x[cc], a0, l0);
                        sp_split(x[cc + 4], a1, l1);
                        d[cc ^ ((kk >> 1) & 3u)] = make_float4(__uint_as_float(a0), __uint_as_float(a1),
                                                              __uint_as_float(l0), __uint_as_float(l1));
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < VI; i++) {
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / (KT * 4), tt = e % 4, n = (e / 4) % 8,
                               kk = (e / 32) % NJ;
                if (e < VG) {
                    const float y0[8] = {vr[i][0].x, vr[i][0].y, vr[i][0].z, vr[i][0].w,
                                         vr[i][1].x, vr[i][1].y, vr[i][1].z, vr[i][1].w};
                    const float y1[8] = {vr[i][2].x, vr[i][2].y, vr[i][2].z, vr[i][2].w,
                                         vr[i][3].x, vr[i][3].y, vr[i][3].z, vr[i][3].w};
                    float4* d = X + hs * (XS / 4) + KT * 32 + (kk * 8 + n) * 32 + tt;
#pragma unroll
                    for (int cc = 0; cc < 8; cc++) {
                        unsigned a0, l0, a1, l1;
                        sp_split(y0[cc], a0, l0);
                        sp_split(y1[cc], a1, l1);
                        d[(cc ^ (n & 1u)) * 4] = make_float4(__uint_as_float(a0), __uint_as_float(a1),
                                                             __uint_as_float(l0), __uint_as_float(l1));
                    }
                }
            }
            sp_mbar_arrive(full + s);
        };
        unsigned qh[8][4], ql[8][4];
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const float* qa = query + ((size_t)b * q_rows + ra) * stride + h * HW + ks * 8 + t;
            const float* qb = query + ((size_t)b * q_rows + rb) * stride + h * HW + ks * 8 + t;
            const bool oa = live && ra < q_rows, ob = live && rb < q_rows;
            const float x[4] = {oa ? __ldg(qa) : 0.f, ob ? __ldg(qb) : 0.f, oa ? __ldg(qa + 4) : 0.f,
                                ob ? __ldg(qb + 4) : 0.f};
#pragma unroll
            for (int r = 0; r < 4; r++) sp_split(x[r], qh[ks][r], ql[ks][r]);
        }
        float o[8][4], mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int r = 0; r < 4; r++) o[n][r] = 0.f;
        for (unsigned kt = 0; kt + 1 < NS && kt < nkt; kt++) {
            load(kt);
            store(c + kt);
        }
        for (unsigned kt = 0; kt < nkt; kt++) {
            const unsigned r = c + kt, s = r % NS;
            const bool ahead = kt + NS - 1 < nkt;
            PT(0); if (ahead) load(kt + NS - 1);
            PT(1); sp_mbar_wait(full + s, (r / NS) & 1u); PT(2);
            if (live) {
                const float4* X = (const float4*)(arena + s * HS * XS + (h - h0) * XS);
                const float4* Vx = X + KT * 32;
                const unsigned k0 = kt * KT;
                float sq[QS][NJ][4];
#pragma unroll
                for (int a = 0; a < QS; a++)
#pragma unroll
                    for (int j = 0; j < NJ; j++)
#pragma unroll
                        for (int q = 0; q < 4; q++) sq[a][j][q] = 0.f;
#pragma unroll
                for (int kk = 0; kk < 8; kk++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        const float4 kv = X[(j * 8 + kk) * 32 + (lane ^ KX(kk))];
                        const unsigned bh0 = __float_as_uint(kv.x), bh1 = __float_as_uint(kv.y),
                                       bl0 = __float_as_uint(kv.z), bl1 = __float_as_uint(kv.w);
                        sp_mma_tf32(sq[QS - 1][j], ql[kk], bh0, bh1);
                        sp_mma_tf32(sq[QS > 2 ? 1 : QS - 1][j], qh[kk], bl0, bl1);
                        sp_mma_tf32(sq[0][j], qh[kk], bh0, bh1);
                    }
                PT(3);
                const bool kfull = k0 + KT <= klen;
                float sc[NJ][4];
                float lm[2] = {-INFINITY, -INFINITY};
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float v0 = QS == 1 ? sq[0][j][q]
                                       : QS == 2 ? sq[0][j][q] + sq[QS - 1][j][q]
                                                 : sq[0][j][q] + (sq[1][j][q] + sq[QS - 1][j][q]);
                        const unsigned kj = k0 + j * 8 + 2 * t + (q & 1);
                        const float v = (kfull || kj < klen) ? v0 * scale : -INFINITY;
                        sc[j][q] = v;
                        lm[q >> 1] = fmaxf(lm[q >> 1], v);
                    }
                if (__any_sync(0xffffffffu, lm[0] > mrow[0] + 8.f || lm[1] > mrow[1] + 8.f)) {
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 1));
                        lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 2));
                        if (lm[i] > mrow[i] + 8.f) {
                            const float corr = mrow[i] == -INFINITY ? 0.f : exp2f(mrow[i] - lm[i]);
                            mrow[i] = lm[i];
                            lrow[i] *= corr;
#pragma unroll
                            for (int n = 0; n < 8; n++) {
                                o[n][2 * i] *= corr;
                                o[n][2 * i + 1] *= corr;
                            }
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float p = exp2f(sc[j][q] - mrow[q >> 1]);
                        sc[j][q] = p;
                        lrow[q >> 1] += p;
                    }
                PT(4);
                constexpr int PA = PS == 1 ? 1 : PS - 1;
                float pv[PA][8][4];
#pragma unroll
                for (int kk = 0; kk < NJ; kk++) {
                    const float x[4] = {sc[kk][0], sc[kk][2], sc[kk][1], sc[kk][3]};
                    unsigned ah[4], al[4];
#pragma unroll
                    for (int q = 0; q < 4; q++) sp_split(x[q], ah[q], al[q]);
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        const float4 vv = Vx[(kk * 8 + n) * 32 + (lane ^ ((n & 1) << 2))];
                        const unsigned bh0 = __float_as_uint(vv.x), bh1 = __float_as_uint(vv.y),
                                       bl0 = __float_as_uint(vv.z), bl1 = __float_as_uint(vv.w);
                        float(&dl)[4] = PS == 1 ? o[n] : pv[PA - 1][n];
                        float(&dh)[4] = PS == 1 ? o[n] : pv[0][n];
                        if (PS > 1 && kk == 0) sp_mma_tf32_z(dl, al, bh0, bh1);
                        else sp_mma_tf32(dl, al, bh0, bh1);
                        sp_mma_tf32(dl, ah, bl0, bl1);
                        if (PS == 3 && kk == 0) sp_mma_tf32_z(dh, ah, bh0, bh1);
                        else sp_mma_tf32(dh, ah, bh0, bh1);
                    }
                }
                if (PS > 1) {
#pragma unroll
                    for (int n = 0; n < 8; n++)
#pragma unroll
                        for (int q = 0; q < 4; q++)
                            o[n][q] += PS == 2 ? pv[0][n][q] : pv[0][n][q] + pv[PA - 1][n][q];
                }
            }
            PT(5); sp_mbar_arrive(empty + s);
            if (ahead) store(r + NS - 1);
            PT(6);
        }
        c += nkt;
        if (!live) continue;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            lrow[i] += __shfl_xor_sync(0xffffffffu, lrow[i], 1);
            lrow[i] += __shfl_xor_sync(0xffffffffu, lrow[i], 2);
            const unsigned r = i ? rb : ra;
            if (r >= q_rows) continue;
            const float inv = lrow[i] > 0.f ? 1.0f / lrow[i] : 0.f;
            float* orow = out + ((size_t)b * q_rows + r) * width + h * HW;
#pragma unroll
            for (int n = 0; n < 8; n++)
                *(float2*)(orow + n * 8 + 2 * t) = make_float2(o[n][2 * i] * inv, o[n][2 * i + 1] * inv);
        }
    }
    PROF_FLUSH
    __syncthreads();
}

/* 3xFP16 variant of sp_attention_tc64m: operands split hi/lo in fp16 (11 + 11 significant bits, as
 * 3xTF32) on m16n8k16 MMAs, which run at twice the tf32 rate. Each K row, each tile's 8-column V
 * block and each Q row are scaled by a power of two to put their max |x| in [128, 256), so lo stays
 * a normal fp16 wherever it matters; the scales come back out exactly (scores: per row and key,
 * P.V: per tile and column block). P (<= 256 under the lazy rescale) is split unscaled.
 * Fragment-ordered stages: K (8-key block j, 16-col block kb) -> uint4 {b0h, b1h, b0l, b1l} per
 * lane; V (16-key block kb, 8-col block n) -> uint4 {b0h, b0l, b1h, b1l} per lane; then the key
 * and V-block inverse scales. */
__device__ __forceinline__ void sp_mma_f16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void sp_mma_f16_z(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}
/* (x0, x1) -> fp16x2 hi and lo with x = hi + lo to 22 bits. */
__device__ __forceinline__ void sp_split_h2(float x0, float x1, unsigned& hi, unsigned& lo) {
    const __half2 h = __floats2half2_rn(x0, x1);
    const float2 hf = __half22float2(h);
    const __half2 l = __floats2half2_rn(x0 - hf.x, x1 - hf.y);
    hi = *(const unsigned*)&h;
    lo = *(const unsigned*)&l;
}
/* 2^(7 - e) for max |x| = m with exponent e (1 for m == 0): scaled max in [128, 256). */
__device__ __forceinline__ float sp_pow2_scale(float m) {
    const int e = (int)((__float_as_uint(m) >> 23) & 0xffu) - 127;
    return m == 0.f ? 1.f : __uint_as_float((unsigned)(127 + 7 - max(min(e, 100), -100)) << 23);
}
template <int KT, int HS, int NS, bool PF>
static __device__ __noinline__ void sp_attention_h16m(const PlowDevInst* in, void* const* T, unsigned slice,
                                                      unsigned nblk) {
    float* const arena = sp_smem;
    constexpr int HW = 64, NJ = KT / 8, NB = KT / 16;
    /* per head and stage: K and V fragments (KT * 64 * 2 halves each = KT * 64 words), key scales,
     * V block scales */
    constexpr int KW = KT * 64, XS = 2 * KW + KT + NB * 8;
    constexpr int GH = KT * 8, G = HS * GH; /* 16-float groups per head, per stage */
    constexpr int GI = (G + PLOW_NV_THREADS - 1) / PLOW_NV_THREADS;
    static_assert(KT % 16 == 0 && (KT * 4) % 32 == 0, "h16 tile");
    static_assert(NS * HS * XS + 4 * NS <= SP_ARENA_FLOATS, "h16 attention stages");
    uint64_t* full = (uint64_t*)(arena + ((NS * HS * XS + 1) & ~1));
    uint64_t* empty = full + NS;
    float* out = (float*)SP_TEN(0);
    const float* query = (const float*)SP_TEN(1);
    const float* key = (const float*)SP_TEN(2);
    const unsigned* lengths = (const unsigned*)SP_TEN(4);
    const float* prefix = (const float*)SP_TEN(6);
    const unsigned* pidx = (const unsigned*)SP_TEN(7);
    const unsigned pre = prefix ? in->i[7] : 0u;
    const float* pfmt = PF ? (const float*)SP_TEN(5) : nullptr;
    const unsigned npt = PF ? (pre + KT - 1) / KT : 0u, kpre = PF ? npt * KT : pre;
    const unsigned batch = in->i[0], q_rows = in->i[1], kv_rows = in->i[2], heads = in->i[3];
    const unsigned width = heads * HW, stride = in->i[5] ? in->i[5] : width;
    const unsigned k_col0 = in->fj[1].u, v_col0 = in->fj[2].u;
    const float scale = in->fj[0].f * 1.4426950408889634f;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned n16 = (q_rows + 15u) / 16u, tiles = heads * n16, runs = (tiles + 7u) / 8u;
    if (tid == 0)
        for (int s = 0; s < NS; s++) {
            sp_mbar_init(full + s, PLOW_NV_WARPS);
            sp_mbar_init(empty + s, PLOW_NV_WARPS);
        }
    __syncthreads();
    unsigned c = 0;
    PROF_DECL
    for (unsigned item = slice; item < batch * runs; item += nblk) {
        const unsigned b = item / runs, u0 = (item % runs) * 8u, u = u0 + warp;
        const bool live = u < tiles;
        const unsigned h0 = u0 / n16, nh = min(u0 + 7u, tiles - 1u) / n16 - h0 + 1u;
        const unsigned h = live ? u / n16 : h0, q0 = live ? (u % n16) * 16u : 0u;
        const unsigned klen = lengths && lengths[b] < pre + kv_rows ? lengths[b] : pre + kv_rows;
        const unsigned ra = q0 + g, rb = ra + 8u;
        const float* kvsrc = prefix ? prefix + (size_t)pidx[b] * pre * (2u * width) : nullptr;
        /* key position kpos: prefix rows below kpre (valid below pre), own row kpos - kpre */
        const unsigned own = klen - pre, nkt = (kpre + own + KT - 1) / KT;
        auto valid_lim = [&](unsigned kt) { return PF && kt < npt ? pre : kpre + own; };
        /* Group e of a stage: head e / GH; within the head, e < KT * 4: K row e / 4, columns
         * 16 * (e % 4) ..; else V half-group: half e % 2 (keys 2t, 2t+1 or 2t+8, 2t+9), pair t,
         * column block n, 16-key block kb. */
        auto src = [&](unsigned kj, unsigned hh, bool v) -> const float* {
            return kj < kpre ? kvsrc + (size_t)kj * (2u * width) + (v ? width : 0u) + hh * HW
                             : key + ((size_t)b * kv_rows + kj - kpre) * stride + (v ? v_col0 : k_col0) + hh * HW;
        };
        auto load = [&](unsigned kt, float4 (&xr)[GI][4]) {
            const unsigned k0 = kt * KT, klim = valid_lim(kt);
            if (PF && kt < npt) return;
#pragma unroll
            for (int i = 0; i < GI; i++) {
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / GH, w = e % GH;
#pragma unroll
                for (int q = 0; q < 4; q++) xr[i][q] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (e >= G || hs >= nh) continue;
                if (w < KT * 4) {
                    const unsigned kj = k0 + w / 4;
                    if (kj < klim) {
                        const float* p = src(kj, h0 + hs, false) + (w % 4) * 16;
#pragma unroll
                        for (int q = 0; q < 4; q++) xr[i][q] = __ldg((const float4*)(p + 4 * q));
                    }
                } else {
                    const unsigned v = w - KT * 4, half = v % 2, tt = (v / 2) % 4, n = (v / 8) % 8, kb = v / 64;
#pragma unroll
                    for (int s2 = 0; s2 < 2; s2++) {
                        const unsigned kj = k0 + kb * 16 + half * 8 + 2 * tt + s2;
                        if (kj < klim) {
                            const float* p = src(kj, h0 + hs, true) + n * 8;
                            xr[i][2 * s2] = __ldg((const float4*)p);
                            xr[i][2 * s2 + 1] = __ldg((const float4*)(p + 4));
                        }
                    }
                }
            }
        };
        auto store = [&](unsigned r, const float4 (&xr)[GI][4]) {
            const unsigned s = r % NS;
            PT(6);
            if (r >= NS && (!(PF && r - c < npt) || tid == 0)) sp_mbar_wait(empty + s, (r / NS - 1) & 1u);
            PT(7);
            float* X = arena + s * HS * XS;
            if (PF && r - c < npt) {
                /* pre-formatted prefix tile: one bulk copy per head, completion counted in bytes */
                if (tid == 0) {
                    const float* src0 = pfmt + ((size_t)pidx[b] * heads * npt + (r - c)) * XS;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sp_su32(full + s)),
                                 "r"(nh * XS * 4u) : "memory");
                    for (unsigned hs = 0; hs < nh; hs++)
                        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                                     ::"r"(sp_su32(X + hs * XS)), "l"(src0 + (size_t)(h0 + hs) * npt * XS), "r"(XS * 4u),
                                     "r"(sp_su32(full + s)) : "memory");
                } else if (lane == 0) {
                    sp_mbar_arrive(full + s);
                }
                PT(8);
                return;
            }
#pragma unroll
            for (int i = 0; i < GI; i++) {
                const unsigned e = tid + i * PLOW_NV_THREADS, hs = e / GH, w = e % GH;
                if (e >= G) continue;
                const float x[16] = {xr[i][0].x, xr[i][0].y, xr[i][0].z, xr[i][0].w, xr[i][1].x, xr[i][1].y,
                                     xr[i][1].z, xr[i][1].w, xr[i][2].x, xr[i][2].y, xr[i][2].z, xr[i][2].w,
                                     xr[i][3].x, xr[i][3].y, xr[i][3].z, xr[i][3].w};
                float m = 0.f;
#pragma unroll
                for (int q = 0; q < 16; q++) m = fmaxf(m, fabsf(x[q]));
                float* Xh = X + hs * XS;
                if (w < KT * 4) {
                    /* the 4 lanes of a K row */
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                    const float f = sp_pow2_scale(m);
                    const unsigned rr = w / 4, cb = w % 4, j = rr / 8, gg = rr % 8;
                    uint4* d = (uint4*)Xh + (j * 4 + cb) * 32;
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        /* lane (gg, q): cols 2q, 2q + 1 (b0) and 2q + 8, 2q + 9 (b1) */
                        unsigned h0w, l0w, h1w, l1w;
                        sp_split_h2(x[2 * q] * f, x[2 * q + 1] * f, h0w, l0w);
                        sp_split_h2(x[2 * q + 8] * f, x[2 * q + 9] * f, h1w, l1w);
                        d[(gg * 4 + q) ^ cb] = make_uint4(h0w, h1w, l0w, l1w);
                    }
                    if (cb == 0) Xh[2 * KW + rr] = 1.f / f;
                } else {
                    const unsigned v = w - KT * 4, half = v % 2, tt = (v / 2) % 4, n = (v / 8) % 8, kb = v / 64;
                    /* the 8 lanes of a (kb, n) V block */
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
                    const float f = sp_pow2_scale(m);
                    unsigned* d = (unsigned*)Xh + KW + ((kb * 8 + n) * 32) * 4;
#pragma unroll
                    for (int q = 0; q < 8; q++) {
                        /* lane (q, tt): keys 2tt, 2tt + 1 (+ 8 * half) of column n * 8 + q */
                        unsigned hw, lw;
                        sp_split_h2(x[q] * f, x[8 + q] * f, hw, lw);
                        *(uint2*)(d + ((q * 4 + tt) ^ ((n & 1u) << 2)) * 4 + half * 2) = make_uint2(hw, lw);
                    }
                    if (half == 0 && tt == 0) Xh[2 * KW + KT + kb * 8 + n] = 1.f / f;
                }
            }
            __syncwarp();
            if (lane == 0) sp_mbar_arrive(full + s);
            PT(9);
        };
        /* Q: per-row scale, fp16 hi/lo A fragments (16-col block kb: a0 row g cols 2t.., a1 row g + 8,
         * a2 row g cols 2t + 8.., a3 row g + 8). */
        unsigned qh[4][4], ql[4][4];
        float qinv[2];
        {
            float x[2][16];
            const bool oa = live && ra < q_rows, ob = live && rb < q_rows;
            const float* qa = query + ((size_t)b * q_rows + ra) * stride + h * HW + 2 * t;
            const float* qb = query + ((size_t)b * q_rows + rb) * stride + h * HW + 2 * t;
#pragma unroll
            for (int kb = 0; kb < 4; kb++) {
                const float2 a0 = oa ? __ldg((const float2*)(qa + kb * 16)) : make_float2(0.f, 0.f);
                const float2 a2 = oa ? __ldg((const float2*)(qa + kb * 16 + 8)) : make_float2(0.f, 0.f);
                const float2 b0 = ob ? __ldg((const float2*)(qb + kb * 16)) : make_float2(0.f, 0.f);
                const float2 b2 = ob ? __ldg((const float2*)(qb + kb * 16 + 8)) : make_float2(0.f, 0.f);
                x[0][kb * 4 + 0] = a0.x; x[0][kb * 4 + 1] = a0.y; x[0][kb * 4 + 2] = a2.x; x[0][kb * 4 + 3] = a2.y;
                x[1][kb * 4 + 0] = b0.x; x[1][kb * 4 + 1] = b0.y; x[1][kb * 4 + 2] = b2.x; x[1][kb * 4 + 3] = b2.y;
            }
            float f[2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                float m = 0.f;
#pragma unroll
                for (int q = 0; q < 16; q++) m = fmaxf(m, fabsf(x[i][q]));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                f[i] = sp_pow2_scale(m);
                qinv[i] = scale / f[i];
            }
#pragma unroll
            for (int kb = 0; kb < 4; kb++) {
                sp_split_h2(x[0][kb * 4] * f[0], x[0][kb * 4 + 1] * f[0], qh[kb][0], ql[kb][0]);
                sp_split_h2(x[1][kb * 4] * f[1], x[1][kb * 4 + 1] * f[1], qh[kb][1], ql[kb][1]);
                sp_split_h2(x[0][kb * 4 + 2] * f[0], x[0][kb * 4 + 3] * f[0], qh[kb][2], ql[kb][2]);
                sp_split_h2(x[1][kb * 4 + 2] * f[1], x[1][kb * 4 + 3] * f[1], qh[kb][3], ql[kb][3]);
            }
        }
        float o[8][4], mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int q = 0; q < 4; q++) o[n][q] = 0.f;
        float4 xa[GI][4], xb[GI][4];
        for (unsigned kt = 0; kt + 1 < NS && kt < nkt; kt++) {
            if (!(ABL & 4)) load(kt, xa);
            store(c + kt, xa);
        }
        if (!(ABL & 4) && NS - 1 < nkt) load(NS - 1, xa);
        /* cur holds tile kt + NS - 1 (loaded an iteration ago); tile kt + NS goes into nxt. */
        auto step = [&](unsigned kt, float4 (&cur)[GI][4], float4 (&nxt)[GI][4]) {
            const unsigned r = c + kt, s = r % NS;
            PT(0);
            if (!(ABL & 4) && kt + NS < nkt) load(kt + NS, nxt);
            PT(1);
            sp_mbar_wait(full + s, (r / NS) & 1u);
            PT(2);
            if (live) {
                const float* Xh = arena + s * HS * XS + (h - h0) * XS;
                const uint4* Kx = (const uint4*)Xh;
                const uint4* Vx = (const uint4*)(Xh + KW);
                const unsigned k0 = kt * KT;
                float sq[3][NJ][4];
#pragma unroll
                for (int kb = 0; kb < 4; kb++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        /* kv = {b0h, b1h, b0l, b1l} */
                        const uint4 kv = Kx[(j * 4 + kb) * 32 + (lane ^ kb)];
                        if (ABL & 1) {
                            if (kb == 0) for (int q = 0; q < 4; q++) sq[0][j][q] = sq[1][j][q] = sq[2][j][q] = 0.f;
                            sq[0][j][0] += __uint_as_float(kv.x ^ qh[kb][0]); sq[1][j][1] += __uint_as_float(kv.y ^ ql[kb][1]);
                            sq[2][j][2] += __uint_as_float(kv.z ^ qh[kb][2]); sq[0][j][3] += __uint_as_float(kv.w ^ ql[kb][3]);
                        } else if (kb == 0) {
                            sp_mma_f16_z(sq[2][j], ql[kb], kv.x, kv.y);
                            sp_mma_f16_z(sq[1][j], qh[kb], kv.z, kv.w);
                            sp_mma_f16_z(sq[0][j], qh[kb], kv.x, kv.y);
                        } else {
                            sp_mma_f16(sq[2][j], ql[kb], kv.x, kv.y);
                            sp_mma_f16(sq[1][j], qh[kb], kv.z, kv.w);
                            sp_mma_f16(sq[0][j], qh[kb], kv.x, kv.y);
                        }
                    }
                PT(3);
                const unsigned klim = valid_lim(kt);
                const bool kfull = k0 + KT <= klim;
                float sc[NJ][4];
                float lm[2] = {-INFINITY, -INFINITY};
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    const float2 kinv = *(const float2*)(Xh + 2 * KW + j * 8 + 2 * t);
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float v0 = sq[0][j][q] + (sq[1][j][q] + sq[2][j][q]);
                        const unsigned kj = k0 + j * 8 + 2 * t + (q & 1);
                        const float v = (kfull || kj < klim) ? v0 * (qinv[q >> 1] * (q & 1 ? kinv.y : kinv.x)) : -INFINITY;
                        sc[j][q] = v;
                        lm[q >> 1] = fmaxf(lm[q >> 1], v);
                    }
                }
                if (__any_sync(0xffffffffu, lm[0] > mrow[0] + 8.f || lm[1] > mrow[1] + 8.f)) {
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 1));
                        lm[i] = fmaxf(lm[i], __shfl_xor_sync(0xffffffffu, lm[i], 2));
                        if (lm[i] > mrow[i] + 8.f) {
                            const float corr = mrow[i] == -INFINITY ? 0.f : exp2f(mrow[i] - lm[i]);
                            mrow[i] = lm[i];
                            lrow[i] *= corr;
#pragma unroll
                            for (int n = 0; n < 8; n++) {
                                o[n][2 * i] *= corr;
                                o[n][2 * i + 1] *= corr;
                            }
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float p = (ABL & 8) ? sc[j][q] - mrow[q >> 1] : exp2f(sc[j][q] - mrow[q >> 1]);
                        sc[j][q] = p;
                        lrow[q >> 1] += p;
                    }
                PT(4);
#pragma unroll
                for (int kb = 0; kb < NB; kb++) {
                    unsigned ah[4], al[4];
                    sp_split_h2(sc[2 * kb][0], sc[2 * kb][1], ah[0], al[0]);
                    sp_split_h2(sc[2 * kb][2], sc[2 * kb][3], ah[1], al[1]);
                    sp_split_h2(sc[2 * kb + 1][0], sc[2 * kb + 1][1], ah[2], al[2]);
                    sp_split_h2(sc[2 * kb + 1][2], sc[2 * kb + 1][3], ah[3], al[3]);
                    const float4 vi0 = *(const float4*)(Xh + 2 * KW + KT + kb * 8);
                    const float4 vi1 = *(const float4*)(Xh + 2 * KW + KT + kb * 8 + 4);
                    const float vinv[8] = {vi0.x, vi0.y, vi0.z, vi0.w, vi1.x, vi1.y, vi1.z, vi1.w};
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        const uint4 vv = Vx[(kb * 8 + n) * 32 + (lane ^ ((n & 1) << 2))];
                        /* vv = {b0h, b0l, b1h, b1l} */
                        float pv[2][4];
                        if (ABL & 2) {
                            pv[0][0] = __uint_as_float(vv.x ^ ah[0]); pv[0][1] = __uint_as_float(vv.y ^ al[1]);
                            pv[0][2] = __uint_as_float(vv.z ^ ah[2]); pv[0][3] = __uint_as_float(vv.w ^ al[3]);
                            pv[1][0] = pv[1][1] = pv[1][2] = pv[1][3] = 0.f;
                        } else {
                        sp_mma_f16_z(pv[1], al, vv.x, vv.z);
                        sp_mma_f16(pv[1], ah, vv.y, vv.w);
                        sp_mma_f16_z(pv[0], ah, vv.x, vv.z);
                        }
#pragma unroll
                        for (int q = 0; q < 4; q++) o[n][q] = fmaf(pv[0][q] + pv[1][q], vinv[n], o[n][q]);
                    }
                }
            }
            PT(5);
            __syncwarp();
            if (lane == 0) sp_mbar_arrive_relaxed(empty + s);
            if (kt + NS - 1 < nkt) {
                if (ABL & 4) {
                    const unsigned r2 = r + NS - 1;
                    if (r2 >= NS) sp_mbar_wait(empty + r2 % NS, (r2 / NS - 1) & 1u);
                    __syncwarp();
                    if (lane == 0) sp_mbar_arrive(full + r2 % NS);
                } else {
                    store(r + NS - 1, cur);
                }
            }
            PT(6);
        };
        for (unsigned kt = 0; kt < nkt; kt += 2) {
            step(kt, xa, xb);
            if (kt + 1 < nkt) step(kt + 1, xb, xa);
        }
        c += nkt;
        if (!live) continue;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            lrow[i] += __shfl_xor_sync(0xffffffffu, lrow[i], 1);
            lrow[i] += __shfl_xor_sync(0xffffffffu, lrow[i], 2);
            const unsigned r = i ? rb : ra;
            if (r >= q_rows) continue;
            const float inv = lrow[i] > 0.f ? 1.0f / lrow[i] : 0.f;
            float* orow = out + ((size_t)b * q_rows + r) * width + h * HW;
#pragma unroll
            for (int n = 0; n < 8; n++)
                *(float2*)(orow + n * 8 + 2 * t) = make_float2(o[n][2 * i] * inv, o[n][2 * i + 1] * inv);
        }
    }
    PROF_FLUSH
    __syncthreads();
}

/* Probe-side format of the prefix table into h16m's stage layout (what a prefill-time op would write):
 * prefix [blocks][pre][2 * width] f32 -> [blocks][heads][npt][XS], one block of KT * 8 threads per
 * (tile, head, block). */
template <int KT>
__global__ void k_fmt_prefix(float* fmt, const float* prefix, unsigned pre, unsigned heads) {
    constexpr int HW = 64, NB = KT / 16, KW = KT * 64, XS = 2 * KW + KT + NB * 8;
    const unsigned pt = blockIdx.x, hh = blockIdx.y, bk = blockIdx.z, npt = gridDim.x, width = heads * HW;
    const unsigned w = threadIdx.x, k0 = pt * KT;
    float* Xh = fmt + (((size_t)bk * heads + hh) * npt + pt) * XS;
    const float* base = prefix + (size_t)bk * pre * (2u * width) + hh * HW;
    float x[16];
    for (int q = 0; q < 16; q++) x[q] = 0.f;
    if (w < KT * 4) {
        const unsigned kj = k0 + w / 4;
        if (kj < pre)
            for (int q = 0; q < 16; q++) x[q] = base[(size_t)kj * 2 * width + (w % 4) * 16 + q];
    } else {
        const unsigned v = w - KT * 4, half = v % 2, tt = (v / 2) % 4, n = (v / 8) % 8, kb = v / 64;
        for (int s2 = 0; s2 < 2; s2++) {
            const unsigned kj = k0 + kb * 16 + half * 8 + 2 * tt + s2;
            if (kj < pre)
                for (int q = 0; q < 8; q++) x[s2 * 8 + q] = base[(size_t)kj * 2 * width + width + n * 8 + q];
        }
    }
    float m = 0.f;
    for (int q = 0; q < 16; q++) m = fmaxf(m, fabsf(x[q]));
    if (w < KT * 4) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        const float f = sp_pow2_scale(m);
        const unsigned rr = w / 4, cb = w % 4, j = rr / 8, gg = rr % 8;
        uint4* d = (uint4*)Xh + (j * 4 + cb) * 32;
        for (int q = 0; q < 4; q++) {
            unsigned h0w, l0w, h1w, l1w;
            sp_split_h2(x[2 * q] * f, x[2 * q + 1] * f, h0w, l0w);
            sp_split_h2(x[2 * q + 8] * f, x[2 * q + 9] * f, h1w, l1w);
            d[(gg * 4 + q) ^ cb] = make_uint4(h0w, h1w, l0w, l1w);
        }
        if (cb == 0) Xh[2 * KW + rr] = 1.f / f;
    } else {
        const unsigned v = w - KT * 4, half = v % 2, tt = (v / 2) % 4, n = (v / 8) % 8, kb = v / 64;
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
        const float f = sp_pow2_scale(m);
        unsigned* d = (unsigned*)Xh + KW + ((kb * 8 + n) * 32) * 4;
        for (int q = 0; q < 8; q++) {
            unsigned hw, lw;
            sp_split_h2(x[q] * f, x[8 + q] * f, hw, lw);
            *(uint2*)(d + ((q * 4 + tt) ^ ((n & 1u) << 2)) * 4 + half * 2) = make_uint2(hw, lw);
        }
        if (half == 0 && tt == 0) Xh[2 * KW + KT + kb * 8 + n] = 1.f / f;
    }
}

__global__ void __launch_bounds__(256, 1) k_plow(PlowDevInst in, void* const* T) {
    d_attention_f32(&in, T, blockIdx.x, gridDim.x, sp_smem);
}
template <int KT, int HS, int QS, int PS, bool LZ>
__global__ void __launch_bounds__(256, 1) k_x(PlowDevInst in, void* const* T) {
    sp_attention_tc64x<KT, HS, QS, PS, LZ>(&in, T, blockIdx.x, gridDim.x);
}
template <int KT, int QS, int PS, bool LZ = false>
static void launch_x(const PlowDevInst& in, void** table, int nblk, cudaStream_t s) {
    const bool wide = in.i[1] >= 128;
    const int smem2 = 2 * 2 * KT * 64 * 16, smem3 = 2 * 3 * KT * 64 * 16;
    static bool init = false;
    if (!init) {
        init = true;
        cudaFuncSetAttribute(k_x<KT, 2, QS, PS, LZ>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem2);
        cudaFuncSetAttribute(k_x<KT, 3, QS, PS, LZ>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem3);
    }
    if (wide) k_x<KT, 2, QS, PS, LZ><<<nblk, 256, smem2, s>>>(in, table);
    else k_x<KT, 3, QS, PS, LZ><<<nblk, 256, smem3, s>>>(in, table);
}

template <int KT, int NC, int QS, int PS, bool PD>
__global__ void __launch_bounds__(256, 1) k_w(PlowDevInst in, void* const* T) {
    sp_attention_tc64w<KT, NC, QS, PS, PD>(&in, T, blockIdx.x, gridDim.x);
}
template <int KT, int NC, int QS, int PS, bool PD = false>
static void launch_w(const PlowDevInst& in, void** table, int nblk, cudaStream_t s) {
    const int smem = 3 * 2 * KT * 64 * 16;
    static bool init = false;
    if (!init) {
        init = true;
        cudaFuncSetAttribute(k_w<KT, NC, QS, PS, PD>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    k_w<KT, NC, QS, PS, PD><<<nblk, 256, smem, s>>>(in, table);
}

template <int KT, int HS, int NS, int QS, int PS>
__global__ void __launch_bounds__(256, 1) k_m(PlowDevInst in, void* const* T) {
    sp_attention_tc64m<KT, HS, NS, QS, PS>(&in, T, blockIdx.x, gridDim.x);
}
template <int NS2, int NS3, int QS, int PS, int KT2 = 16>
static void launch_m(const PlowDevInst& in, void** table, int nblk, cudaStream_t s) {
    const int smem = SP_ARENA_FLOATS * 4;
    static bool init = false;
    if (!init) {
        init = true;
        cudaFuncSetAttribute(k_m<KT2, 2, NS2, QS, PS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(k_m<16, 3, NS3, QS, PS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    if (in.i[1] > 112) k_m<KT2, 2, NS2, QS, PS><<<nblk, 256, smem, s>>>(in, table);
    else k_m<16, 3, NS3, QS, PS><<<nblk, 256, smem, s>>>(in, table);
}

template <int KT, int HS, int NS, bool PF>
__global__ void __launch_bounds__(256, 1) k_h(PlowDevInst in, void* const* T) {
    sp_attention_h16m<KT, HS, NS, PF>(&in, T, blockIdx.x, gridDim.x);
}
template <int KT, int HS, int NS, bool PF>
__global__ void __launch_bounds__(256, 2) k_h2(PlowDevInst in, void* const* T) {
    sp_attention_h16m<KT, HS, NS, PF>(&in, T, blockIdx.x, gridDim.x);
}
static void launch_h2(PlowDevInst in, void** table, int nblk, cudaStream_t s) {
    const int smem = 3 * 3 * (2 * 16 * 64 + 16 + 8) * 4 + 64;
    static bool init = false;
    if (!init) {
        init = true;
        cudaFuncSetAttribute(k_h2<16, 2, 3, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(k_h2<16, 3, 3, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    in.t[5] = 5;
    if (in.i[1] > 112) k_h2<16, 2, 3, true><<<nblk, 256, smem, s>>>(in, table);
    else k_h2<16, 3, 3, true><<<nblk, 256, smem, s>>>(in, table);
}
template <int KT2, int NS2, int KT3, int NS3, bool PF = false>
static void launch_h(PlowDevInst in, void** table, int nblk, cudaStream_t s) {
    const int smem = SP_ARENA_FLOATS * 4;
    static bool init = false;
    if (!init) {
        init = true;
        cudaFuncSetAttribute(k_h<KT2, 2, NS2, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(k_h<KT3, 3, NS3, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    if (PF) in.t[5] = (in.i[1] > 112 ? KT2 : KT3) == 32 ? 6 : 5;
    if (in.i[1] > 112) k_h<KT2, 2, NS2, PF><<<nblk, 256, smem, s>>>(in, table);
    else k_h<KT3, 3, NS3, PF><<<nblk, 256, smem, s>>>(in, table);
}

struct Handle {
    PlowDevInst in;
    void** table;
};

extern "C" {
void cfm_prof(unsigned long long* out, int reset) {
    cudaMemcpyFromSymbol(out, g_prof, sizeof(g_prof));
    if (reset) { unsigned long long z[16] = {}; cudaMemcpyToSymbol(g_prof, z, sizeof(z)); }
}
/* which: 0 = interpreter path; 1.. = candidates. Tensors are device pointers. */
void* cfm_make(float* out, const float* qkv, const float* prefix, const unsigned* pidx, const unsigned* klen,
               unsigned B, unsigned Tq, unsigned pre) {
    Handle* h = new Handle;
    memset(&h->in, 0, sizeof(h->in));
    for (auto& x : h->in.t) x = PLOW_TENSOR_NONE;
    const unsigned heads = 8, width = heads * 64;
    const unsigned npt = (pre + 15) / 16, nblocks = 4, XS16 = 2 * 16 * 64 + 16 + 8;
    float* fmt = nullptr;
    cudaMalloc(&fmt, (size_t)nblocks * heads * npt * XS16 * 4);
    k_fmt_prefix<16><<<dim3(npt, heads, nblocks), 128>>>(fmt, prefix, pre, heads);
    const unsigned npt32 = (pre + 31) / 32, XS32 = 2 * 32 * 64 + 32 + 16;
    float* fmt32 = nullptr;
    cudaMalloc(&fmt32, (size_t)nblocks * heads * npt32 * XS32 * 4);
    k_fmt_prefix<32><<<dim3(npt32, heads, nblocks), 256>>>(fmt32, prefix, pre, heads);
    void* ptrs[7] = {out, (void*)qkv, (void*)klen, (void*)prefix, (void*)pidx, fmt, fmt32};
    cudaMalloc(&h->table, sizeof(ptrs));
    cudaMemcpy(h->table, ptrs, sizeof(ptrs), cudaMemcpyHostToDevice);
    h->in.op = PLOW_DOP_ATTENTION_F32;
    h->in.t[0] = 0; h->in.t[1] = 1; h->in.t[2] = 1; h->in.t[3] = 1; h->in.t[4] = 2; h->in.t[6] = 3; h->in.t[7] = 4;
    h->in.i[0] = B; h->in.i[1] = Tq; h->in.i[2] = Tq; h->in.i[3] = heads; h->in.i[4] = 64;
    h->in.i[5] = 3 * width; h->in.i[6] = 2u; h->in.i[7] = pre;
    h->in.fj[0].f = 0.125f; h->in.fj[1].u = width; h->in.fj[2].u = 2 * width;
    return h;
}
int cfm_run(void* hp, int which, int nblk, void* stream) {
    Handle* h = (Handle*)hp;
    cudaStream_t s = (cudaStream_t)stream;
    const size_t arena = SP_ARENA_FLOATS * sizeof(float);
    static bool init = false;
    if (!init) {
        init = true;
        cudaFuncSetAttribute(k_plow, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)arena);
    }
    switch (which) {
    case 0: k_plow<<<nblk, 256, arena, s>>>(h->in, h->table); break;
    case 1: launch_x<16, 1, 2>(h->in, h->table, nblk, s); break;
    case 2: launch_x<16, 3, 2>(h->in, h->table, nblk, s); break;
    case 3: launch_x<16, 1, 2, true>(h->in, h->table, nblk, s); break;
    case 4: launch_x<16, 3, 2, true>(h->in, h->table, nblk, s); break;
    case 5: launch_x<16, 1, 1, true>(h->in, h->table, nblk, s); break;
    case 6: launch_x<32, 1, 2, true>(h->in, h->table, nblk, s); break;
    case 7: launch_x<32, 3, 2, true>(h->in, h->table, nblk, s); break;
    case 8: launch_w<16, 6, 3, 2>(h->in, h->table, nblk, s); break;
    case 9: launch_w<16, 6, 1, 2>(h->in, h->table, nblk, s); break;
    case 10: launch_w<16, 6, 3, 2, true>(h->in, h->table, nblk, s); break;
    case 11: launch_w<16, 6, 3, 3>(h->in, h->table, nblk, s); break;
    case 12: launch_m<3, 2, 3, 2>(h->in, h->table, nblk, s); break;
    case 13: launch_m<2, 2, 3, 2>(h->in, h->table, nblk, s); break;
    case 14: launch_m<3, 2, 1, 2>(h->in, h->table, nblk, s); break;
    case 15: launch_m<2, 2, 3, 2, 24>(h->in, h->table, nblk, s); break;
    case 16: launch_m<3, 2, 3, 3>(h->in, h->table, nblk, s); break;
    case 20: launch_h<16, 3, 16, 3>(h->in, h->table, nblk, s); break;
    case 23: launch_h<16, 3, 16, 3, true>(h->in, h->table, nblk, s); break;
    case 24: launch_h2(h->in, h->table, nblk, s); break;
    case 25: launch_h<32, 2, 32, 2, true>(h->in, h->table, nblk, s); break;
    case 26: launch_h<32, 2, 16, 3, true>(h->in, h->table, nblk, s); break;
    case 21: launch_h<32, 2, 32, 2>(h->in, h->table, nblk, s); break;
    case 22: launch_h<32, 3, 16, 3>(h->in, h->table, nblk, s); break;
    default: return -1;
    }
    return (int)cudaGetLastError();
}
}
