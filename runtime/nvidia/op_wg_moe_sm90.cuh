// Hopper (sm_90a) grouped fp4 GEMM for DeepSeek-V4.1's routed experts: the weights ride the MMA's
// M side and the tokens its N side (8 .. C::NT columns), so the tensor work scales with the rows an
// expert actually holds (16 at 1k tokens, TP4).
//
// A work item is 256 weight rows (W0 rows n0 .. n0 + 256, or W0 rows n0 .. n0 + 128 then W1 rows
// n0 .. n0 + 128 -- a gate|up pair) x up to C::NT token rows.
//   * CTA = 384 threads: warpgroup 0 produces by cp.async only -- warps 0-2 the weights and scales
//     (barrier `full`), warp 3 the gathered tokens (`tfull`): one barrier for both serialized the
//     two streams (1.2 GB of weights and 0.3 GB of L2-resident tokens took the SUM of their separate
//     times). A slot holds one stage of both; warpgroups 1, 2 consume.
//   * a stage is 256 K: 128 contiguous bytes of every weight row (64 B chunks cost ~15% of the
//     stream; two producer warps of 16 B requests cap it near 1.7 TB/s, three reach ~3).
//   * consumer cw owns weight rows cw * 64 (block 0) and 128 + cw * 64 (block 1): the gate and the
//     up row of the same output in the same thread. It decodes the packed e2m1 bytes straight into
//     wgmma A registers (value * ue8m0 scale, exact in bf16 -- so no per-32 promotion) and issues
//     m64nNTk16 against the tokens' 128 B-swizzled smem, one wgmma group per 64-K sub-stage; the
//     other consumer warpgroup's decode overlaps this one's tensor work.
//   * setmaxnreg: 96 registers for the producers, 200 for the consumers. Both must be issued inside
//     their role's branch, and the producers must fit in theirs, or ptxas drops them (C7507) and
//     serializes every wgmma.
//   * PERSISTENT over the CTA's items: the producers stream the next item during the epilogue.
// Epilogue: epi.template run<NT>(acc0, acc1, cw, row0, rend, red); thread (warp w, lane g * 4 + t4)
// holds weight rows (w * 16 + g + 8 h) of each block, token columns j * 8 + t4 * 2 + e, as
// acc[j * 4 + h * 2 + e]; red holds 2 x 4 x C::NT floats.
#pragma once
#include "op_wg_sm90.cuh"

#define WGM_SMR(s) asm volatile(s ::: "memory")
#ifndef WGM_PREG
#define WGM_PREG "96"  // registers after setmaxnreg: 128 x producer + 256 x consumer <= 384 x 168
#define WGM_CREG "200"
#endif
#define WGM_ROWS 256
#define WGM_BK 256                      // K per stage
#define WGM_SUBS (WGM_BK / 64)
#define WGM_RAW_LD 128                  // 128 B of fp4 per row and stage, 16 B chunk c at c ^ (row & 7)
#define WGM_SCL_LD 16                   // a row's stage scales: 8 bytes at offset <= 3 (8 B aligned for cp.async 8)
#define WGM_WBYTES (WGM_ROWS * WGM_RAW_LD)

namespace plow_wgm {
using namespace plow_wg;
#include "op_wg_rs_sm90.inc"

template <int NT_, int ST_>
struct Cfg {
    static constexpr int NT = NT_;          // tokens per work item (max)
    static constexpr int ST = ST_;          // stages
    static constexpr int TSUB = NT * 128;   // one 64-K token sub-tile, 128 B swizzled rows
    static constexpr int TBYTES = WGM_SUBS * TSUB;
    static constexpr int SLOT = TBYTES + WGM_WBYTES + WGM_ROWS * WGM_SCL_LD;  // tokens | weights | scales
    static constexpr int RED = 2 * 2 * 4 * NT * 4;
    static constexpr int SMEM = ST * SLOT + 3 * ST * 8 + RED + 1024;
};

// weight stream: the row's next 256 B come into L2 with this request
__device__ __forceinline__ void cp_async16_l2(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
    return v;
}
__device__ __forceinline__ uint32_t lds_u8(uint32_t a) {
    uint32_t v;
    asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(v) : "r"(a));
    return v;
}
// Byte `sel` (0-3) of w = two e2m1 (element k low nibble, k+1 high) -> bf16x2 (k low half) * sc2.
// The byte times 0x40040 lands both magnitudes (bits 0-2, 4-6) at bf16 bits 6-8 / 22-24 -- exponent
// and top mantissa bit, i.e. the value * 2^-126, subnormal 0.5 included -- with the signs at bits
// 9 / 25; shifted 6 more (and masked: the <<18 copy of the low magnitude reaches bit 24) the signs
// sit at 15 / 31. 6 ops per 2 elements (the e4m3 route costs ~10).
__device__ __forceinline__ uint32_t dec_byte(uint32_t w, uint32_t sel, __nv_bfloat162 sc2) {
    const uint32_t p = __byte_perm(w, 0u, sel) * 0x40040u;
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, 0x01C001C0, 0xEC;" : "=r"(r) : "r"(p), "r"((p << 6) & 0x80008000u));  // (a & c) | b
    const __nv_bfloat162 v = __hmul2(*(const __nv_bfloat162*)&r, sc2);
    return *(const uint32_t*)&v;
}

// One work item: A rows row0 .. rend (<= C::NT; row p reads A row rows[p], or p when rows is null)
// against weight rows n0 .. (W1 set: the gate|up pair).
struct Tile {
    int row0, rend, n0;
    const int* rows;
    const uint8_t *W0, *S0, *W1, *S1;
};

template <class C>
struct Ring {
    uint8_t* slot;  // [ST] x {tokens [SUBS][TSUB] | weights [ROWS][RAW_LD] | scales [ROWS][SCL_LD]}
    uint64_t *full, *tfull, *empty;
    float* red;  // [tile parity][2 consumers][4 warps][NT] epilogue exchange
    __device__ __forceinline__ Ring() {
        extern __shared__ __align__(1024) uint8_t wg_smem_raw[];
        slot = (uint8_t*)(((uintptr_t)wg_smem_raw + 1023) & ~(uintptr_t)1023);
        full = (uint64_t*)(slot + C::ST * C::SLOT);
        tfull = full + C::ST;
        empty = tfull + C::ST;
        red = (float*)(empty + C::ST);
    }
};
__device__ __forceinline__ int pick_nt(int n) { return n <= 8 ? 8 : n <= 16 ? 16 : n <= 32 ? 32 : n <= 64 ? 64 : 128; }
// a row's stage scales come as one aligned 8 B piece (window offset 0) when every row start is 8 B aligned
__device__ __forceinline__ bool scale8(int KB, const Tile& t) { return KB % 8 == 0 && (((uintptr_t)t.S0 | (uintptr_t)t.S1) & 7) == 0; }

// A fragments of one 64-K sub-stage for this thread's four weight rows (block b = q >> 1, half
// h = q & 1): row q's 32 fp4 bytes at wslot + roff[q] + sub * 32, its two 32-wide scales at
// wslot + WGM_WBYTES + sco[q] + sub * 2 (soff[q] < 0: past N, zeros).
__device__ __forceinline__ void decode_sub(uint32_t (&a)[2][4][4], uint32_t wslot, const uint32_t* roff, const uint32_t* sco,
                                           const int* soff, int sub, uint32_t bsel) {
#pragma unroll
    for (int b = 0; b < 2; b++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int q = b * 2 + h;
            const uint32_t rp = wslot + roff[q], sw = (uint32_t)(threadIdx.x >> 2) & 7u;  // row & 7 = g
            const uint4 p0 = lds128(rp + (((2u * sub) ^ sw) << 4)), p1 = lds128(rp + (((2u * sub + 1u) ^ sw) << 4));
            const uint32_t wd[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
            const uint32_t sp = wslot + WGM_WBYTES + sco[q] + sub * 2;
            const __nv_bfloat162 z = __floats2bfloat162_rn(0.f, 0.f);
            const __nv_bfloat162 c0 = soff[q] >= 0 ? bf16x2_pow2((int)lds_u8(sp) - 1) : z;
            const __nv_bfloat162 c1 = soff[q] >= 0 ? bf16x2_pow2((int)lds_u8(sp + 1) - 1) : z;
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                a[b][kk][h] = dec_byte(wd[2 * kk], bsel, kk < 2 ? c0 : c1);
                a[b][kk][2 + h] = dec_byte(wd[2 * kk + 1], bsel, kk < 2 ? c0 : c1);
            }
        }
}

// The item's K loop on one consumer warpgroup. Each 64-K sub-stage is one wgmma group (4 k16 steps
// x 2 blocks); a slot is released once the group that read its last sub-stage has retired.
template <class C, int NT, class Epi>
__device__ __forceinline__ void consume(const Ring<C>& R, unsigned& gs, const Tile& t, int N, int K, unsigned par, const Epi& epi) {
    const int tid = threadIdx.x, cw = (tid >> 7) - 1, lt = tid & 127, w = lt >> 5, lane = lt & 31, g = lane >> 2, t4 = lane & 3;
    const int KS = (K + WGM_BK - 1) / WGM_BK, KB = K >> 5;
    const bool pair = t.W1 != nullptr;
    const bool sc8 = scale8(KB, t);
    const uint32_t bsel = 0x4440u | (uint32_t)t4;  // byte t4 to position 0, zeros above
    int soff[4];                                   // per (block, half): offset into the row's scale window, -1 past N
    uint32_t roff[4], sco[4];                      // the row's weight / scale offsets in a slot
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const int rt = cw * 64 + w * 16 + g + 8 * (q & 1), gn = t.n0 + (pair ? rt : (q >> 1) * 128 + rt);
        soff[q] = gn < N ? (sc8 ? 0 : (gn * KB) & 3) : -1;
        const int row = (q >> 1) * 128 + rt;
        roff[q] = C::TBYTES + row * WGM_RAW_LD;
        sco[q] = C::TBYTES + row * WGM_SCL_LD + max(soff[q], 0);
    }
    // every row of this warpgroup is past N (the last gate|up tile at I = 576): nothing to multiply
    const bool idle = t.n0 + cw * 64 >= N;
    float acc0[NT / 2], acc1[NT / 2];
#pragma unroll
    for (int i = 0; i < NT / 2; i++) acc0[i] = acc1[i] = 0.f;
    // One A buffer: a second one (decode during the previous group) runs ptxas out of registers and it
    // serializes every wgmma (C7511/C7512).
    uint32_t a[2][4][4];  // [block][k16 step][fragment register]
    auto step = [&](uint32_t sbase, int sub) {
        decode_sub(a, sbase, roff, sco, soff, sub, bsel);
        wg_fence();
        const uint32_t tsub = sbase + sub * C::TSUB;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const uint64_t db = wg_desc(tsub + kk * 32);
            wg_rs<NT>(acc0, a[0][kk], db);
            wg_rs<NT>(acc1, a[1][kk], db);
        }
        wg_commit();
        wg_wait<0>();
    };
    for (int ks = 0; ks < KS; ks++, gs++) {
        const int nsub = min(WGM_SUBS, (K - ks * WGM_BK) / 64);
        const int s = gs % C::ST;
        mbar_wait(&R.full[s], (gs / C::ST) & 1);
        mbar_wait(&R.tfull[s], (gs / C::ST) & 1);
        const uint32_t sbase = smem_u32(R.slot + s * C::SLOT);
#pragma unroll 1
        for (int sub = 0; sub < (idle ? 0 : nsub); sub++) step(sbase, sub);
        mbar_arrive(&R.empty[s]);  // the stage's last group has retired: holding the slot to the next wait chains every stage to a memory round trip
    }
    epi.template run<NT>(acc0, acc1, cw, t.row0, t.rend, R.red + par * (2 * 4 * C::NT));
}

// Items first, first + stride, ... < count of `tile(item)`; `epi(tile)` makes each one's epilogue.
// Every thread of the 384-thread CTA calls it; it starts and ends on a block barrier.
template <class C, class TileFn, class EpiFn>
__device__ __forceinline__ void moe_run(const bf16* __restrict__ A, long long lda, int N, int K, unsigned first, unsigned count,
                                        unsigned stride, const TileFn& tile, const EpiFn& epi) {
    const Ring<C> R;
    const int tid = threadIdx.x;
    if (tid == 0) {
        for (int s = 0; s < C::ST; s++) {
            mbar_init(&R.full[s], 96);
            mbar_init(&R.tfull[s], 32);
            mbar_init(&R.empty[s], 256);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    const int KS = (K + WGM_BK - 1) / WGM_BK, KB = K >> 5;
    if (tid < 128) {
      WGM_SMR("setmaxnreg.dec.sync.aligned.u32 " WGM_PREG ";\n");
      if (tid < 96) {
        // weights: thread = 16 B chunk (tid & 7) of rows (tid >> 3) + 12 j; one IMAD.WIDE per copy. Lean on
        // purpose: past WGM_PREG registers ptxas ignores setmaxnreg and serializes the consumers' wgmma.
        const long long send = (long long)N * KB;  // scale bytes that exist: a window may not read past them
        const int Kh = K / 2, ci = (tid & 7) * 16, r0 = tid >> 3;
        unsigned gs = 0;
        for (unsigned item = first; item < count; item += stride) {
            const Tile t = tile(item);
            const bool pair = t.W1 != nullptr;
            const bool sc8 = scale8(KB, t);
            // valid rows: pair -> [0, nv) of W0 and [128, 128 + nv) of W1; else [0, nv) of W0
            const int nv = min(pair ? 128 : WGM_ROWS, N - t.n0);
            const int nvlo = min(nv, 128), nvhi = pair ? nv : nv - 128;
            const uint8_t* wlo = t.W0 + (size_t)(t.n0 + r0) * Kh + ci;
            const uint8_t* whi = (pair ? t.W1 + (size_t)(t.n0 + r0) * Kh : t.W0 + (size_t)(t.n0 + 128 + r0) * Kh) + ci;
            for (int ks = 0; ks < KS; ks++, gs++) {
                const int nsub = min(WGM_SUBS, (K - ks * WGM_BK) / 64), s = gs % C::ST;
                if (gs >= (unsigned)C::ST) mbar_wait(&R.empty[s], ((gs / C::ST) - 1) & 1);
                const uint32_t rb = smem_u32(R.slot + s * C::SLOT) + C::TBYTES, sb = rb + WGM_WBYTES;
                if (ci < nsub * 32) {
                    // rows [0, 128) of W0, then rows [128, 256): W1's first 128 (pair) or W0's next
#pragma unroll
                    for (int half = 0; half < 2; half++) {
                        const uint8_t* p = (half ? whi : wlo) + ks * (WGM_BK / 2);
                        const int lim = half ? nvhi : nvlo;
#pragma unroll 1
                        for (int r = r0; r < 128; r += 12, p += 12 * (long long)Kh) {
                            if (r >= lim) break;
                            const int rs = r + half * 128;
                            cp_async16_l2(rb + rs * WGM_RAW_LD + (((ci >> 4) ^ (rs & 7)) << 4), p, 16);
                        }
                    }
                }
#pragma unroll 1
                for (int r = tid; r < WGM_ROWS; r += 96) {
                    const bool hi = pair && r >= 128;
                    const int rr = hi ? r - 128 : r;
                    if (rr >= nv) continue;
                    const uint8_t* sbase = hi ? t.S1 : t.S0;
                    const long long so = (long long)(t.n0 + rr) * KB + ks * (WGM_BK / 32);
                    if (sc8) {  // one aligned 8 B piece: the row's 8 stage scales, window offset 0
                        cp_async8(sb + r * WGM_SCL_LD, sbase + so, (int)max(0ll, min(8ll, send - so)));
                        continue;
                    }
                    const long long a0 = so & ~3ll;
#pragma unroll
                    for (int i = 0; i < 3; i++) {
                        const long long o = a0 + i * 4;
                        cp_async4(sb + r * WGM_SCL_LD + i * 4, sbase + min(o, send - 1), (int)max(0ll, min(4ll, send - o)));
                    }
                }
                mbar_cp_async_arrive(&R.full[s]);
            }
        }
      } else {
        // tokens: lane = row (lane >> 2) + 8 g of the item, 16 B chunks (lane & 3) + 4 i of its 512 B stage;
        // row pointers resolved once per item. Rows past the item are not loaded: their columns are
        // never stored.
        const int lane = tid & 31, q = lane & 3, rl = lane >> 2;
        constexpr int NG = C::NT / 8;
        unsigned gs = 0;
        for (unsigned item = first; item < count; item += stride) {
            const Tile t = tile(item);
            const int n_rows = t.rend - t.row0;
            const bf16* src[NG];
#pragma unroll
            for (int gi = 0; gi < NG; gi++) {
                const int r = rl + 8 * gi;
                src[gi] = r < n_rows ? A + (long long)(t.rows ? t.rows[t.row0 + r] : t.row0 + r) * lda + q * 8 : nullptr;
            }
            for (int ks = 0; ks < KS; ks++, gs++) {
                const int nsub = min(WGM_SUBS, (K - ks * WGM_BK) / 64), s = gs % C::ST;
                if (gs >= (unsigned)C::ST) mbar_wait(&R.empty[s], ((gs / C::ST) - 1) & 1);
                const uint32_t tb = smem_u32(R.slot + s * C::SLOT);
#pragma unroll
                for (int gi = 0; gi < NG; gi++) {
                    if (src[gi] == nullptr) continue;
                    const int r = rl + 8 * gi;
#pragma unroll
                    for (int i = 0; i < 8; i++) {  // chunk q + 4 i: sub-stage i >> 1, 16 B column (q + 4 i) & 7
                        if ((i >> 1) < nsub) wg_cp_async16(tb + (i >> 1) * C::TSUB + sw128(r, (q + 4 * i) & 7), src[gi] + ks * WGM_BK + i * 32, 16);
                    }
                }
                mbar_cp_async_arrive(&R.tfull[s]);
            }
        }
      }
      WGM_SMR("setmaxnreg.inc.sync.aligned.u32 168;\n");
    } else {
        WGM_SMR("setmaxnreg.inc.sync.aligned.u32 " WGM_CREG ";\n");
        unsigned gs = 0, par = 0;
        for (unsigned item = first; item < count; item += stride, par ^= 1u) {
            const Tile t = tile(item);
            const auto e = epi(t);
            const int nt = pick_nt(t.rend - t.row0);
            if (nt <= 8) consume<C, 8>(R, gs, t, N, K, par, e);
            else if (nt <= 16) consume<C, 16>(R, gs, t, N, K, par, e);
            else if (nt <= 32 || C::NT <= 32) consume<C, (C::NT < 32 ? C::NT : 32)>(R, gs, t, N, K, par, e);
            else if (nt <= 64 || C::NT <= 64) consume<C, (C::NT < 64 ? C::NT : 64)>(R, gs, t, N, K, par, e);
            else consume<C, C::NT>(R, gs, t, N, K, par, e);
        }
        // back to the kernel's even split for the code that follows (the consumers free theirs first)
        WGM_SMR("setmaxnreg.dec.sync.aligned.u32 168;\n");
    }
    __syncthreads();
}

}  // namespace plow_wgm
