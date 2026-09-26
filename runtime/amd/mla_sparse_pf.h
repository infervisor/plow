// GLM-5.3 DSA sparse MLA prefill in the absorbed form (the top-k MQA vLLM runs for sparse layers).
// A work item is one PACK of 8 queries; its 64 rows are (query, head) pairs that share one walk over
// the pack's union of selected keys (IndexUnionPf table: ascending positions + a per-key bitmask of
// the pack's queries). Keys are gathered by per-lane DMA; per-query exactness comes from the bits.
//
// Wave w = (row tile rt = w & 1: queries 4rt..4rt+3, key half kh = w >> 1) over 64-key chunks:
//   S^T = K Q^T for its 32 keys x 32 rows (36 k-steps of v_mfma_f32_32x32x16_bf16, rows lane-local
//   as in mla_mha_pf.h), row maxima exchanged with the partner wave (same rt), P halves swapped
//   through LDS, then O^T[dv half kh] += V^T P^T over all 64 keys (V = the latent rows themselves).
// Output is the normalized bf16 latent [T][8][512] that FlashMerge used to produce.
//
// LDS per stage: two 256-wide latent sub-tiles in the mla_mha row-pair layout + 64 rope rows
// (128 B, chunk index XOR row&7). The rope region doubles as the P exchange once QK is done.
// A 3-slot ring prefetches each chunk's positions and masks two chunks ahead.
#pragma once

namespace mla_sp {
constexpr unsigned QP = 8, NH = 8, DK = 512, DR = 64, BKV = 64;
constexpr unsigned PS = 560;                    // latent row-pair stride (halves)
constexpr unsigned SUB = BKV / 2 * PS;          // one 256-wide latent sub-tile (halves)
constexpr unsigned ROPE = BKV * DR;             // halves
constexpr unsigned STAGE = 2 * SUB + ROPE;      // halves
constexpr unsigned STATS_B = 2 * STAGE * 2;     // byte offsets
constexpr unsigned RING_B = STATS_B + 4 * 32 * 4;
constexpr unsigned RING = 3, SLOT_B = 2 * BKV * 4;
constexpr unsigned TICKET_B = RING_B + RING * SLOT_B;
constexpr unsigned LDS_BYTES = TICKET_B + 16;  // 161,808
typedef bf16_t bx4 __attribute__((ext_vector_type(4)));
}  // namespace mla_sp

__device__ __forceinline__ unsigned mla_sp_off(unsigned row, unsigned col) {
    return (row >> 1) * mla_sp::PS + ((col >> 3) * 2u + (row & 1u)) * 8u + (col & 7u);
}

// uni: IndexUnionPf table (pack = 8). Qa [T][8][512], Qr [T][8][64] (roped, or raw with cosb/sinb),
// Ckv [kv][512], Kr [kv][64].
__device__ void d_mla_sparse_pf(bf16* __restrict__ O, const bf16* __restrict__ Qa,
                                const bf16* __restrict__ Qr, const bf16* __restrict__ Ckv,
                                const bf16* __restrict__ Kr, const unsigned char* __restrict__ uni,
                                unsigned T, unsigned cap, float scale, unsigned slice,
                                unsigned nblk, unsigned char* lds,
                                const float* __restrict__ cosb = nullptr,
                                const float* __restrict__ sinb = nullptr,
                                const int* __restrict__ kv_len = nullptr) {
    using namespace mla_sp;
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned g = lane >> 5, r32 = lane & 31u;
    const unsigned g16 = (lane >> 4) & 1u, i16 = lane & 15u;
    const unsigned rt = wave & 1u, kh = wave >> 1, pw = wave ^ 2u;
    const float sl2 = scale * 1.4426950408889634f;
    (void)slice;
    const unsigned n_qt = (T + QP - 1) / QP;
    const unsigned hdr = (n_qt * 4u + 255u) / 256u * 256u;
    bf16* const lh = (bf16*)lds;
    float* const stats = (float*)(lds + STATS_B);
    const unsigned ql = 4u * rt + (r32 >> 3), hh = r32 & 7u;

    // Packs are handed out largest-first from a ticket counter one word past the last union block
    // (union sizes vary ~500x across packs, so a static stride leaves a 1.3-1.5x tail).
    // IndexUnionPf zeroes [ticket, exits]; layers that reuse a union table (span) get no fresh
    // zero, so the last workgroup out resets both for the next flash on this table.
    unsigned* const ticket = (unsigned*)(lds + TICKET_B);
    unsigned* const ctr = (unsigned*)(uni + hdr + (size_t)n_qt * cap * 12u);
    for (;;) {
        __syncthreads();  // previous item's LDS reads (and ticket read) are done
        if (tid == 0u) *ticket = atomicAdd(ctr, 1u);
        __syncthreads();
        const unsigned it = *ticket;
        if (it >= n_qt) {
            if (tid == 0u && atomicAdd(ctr + 1, 1u) == nblk - 1u) {
                atomicExch(ctr, 0u);
                atomicExch(ctr + 1, 0u);
            }
            break;
        }
        const unsigned qt = n_qt - 1u - it;  // longest unions first
        const unsigned ucount = ((const unsigned*)uni)[qt];
        const unsigned char* blk = uni + hdr + (size_t)qt * cap * 12u;
        const int* upos = (const int*)blk;
        const unsigned* umask = (const unsigned*)(blk + (size_t)cap * 4u);
        const unsigned nch = (ucount + BKV - 1) / BKV;

        // Ring slot s: [64 x i32 pos][64 x u32 mask]; waves 0/1 each move 256 B with 16 lanes.
        auto ring = [&](unsigned c) {
            if (wave < 2u && lane < 16u) {
                const unsigned char* src = wave ? (const unsigned char*)(umask + c * BKV)
                                                : (const unsigned char*)(upos + c * BKV);
                cp_async16((const PLOW_GLOB bf16*)as_glob(src + lane * 16u),
                           (bf16*)(lds + RING_B + (c % RING) * SLOT_B + wave * 256u));
            }
        };
        auto stage = [&](unsigned c) {
            bf16* st = lh + (c & 1u) * STAGE;
            const int* pos = (const int*)(lds + RING_B + (c % RING) * SLOT_B);
            const unsigned c0 = c * BKV;
            auto kpos = [&](unsigned row) -> size_t {
                return c0 + row < ucount ? (size_t)pos[row] : 0u;
            };
#pragma unroll
            for (unsigned k = 0; k < 8; k++) {
                const unsigned p = 8u * wave + k, row = 2u * p + (lane & 1u);
                const size_t kv = kpos(row);
#pragma unroll
                for (unsigned u = 0; u < 2; u++)
                    cp_async16(as_glob(Ckv + kv * DK + u * 256u + 8u * (lane >> 1)),
                               st + u * SUB + p * PS);
            }
#pragma unroll
            for (unsigned k = 0; k < 2; k++) {
                const unsigned j = 2u * wave + k, row = 8u * j + (lane >> 3);
                const unsigned lc = (lane & 7u) ^ (row & 7u);
                cp_async16(as_glob(Kr + kpos(row) * DR + 8u * lc), st + 2u * SUB + j * 512u);
            }
        };

        if (nch) ring(0);
        if (nch > 1) ring(1);
        cp_async_wait();
        __syncthreads();
        if (nch) stage(0);

        const unsigned qi = qt * QP + ql;
        const bool qv = qi < T;
        const size_t qrow = (size_t)(qv ? qi : T - 1u) * NH + hh;
        bf16x8 qf[36];
#pragma unroll
        for (unsigned s = 0; s < 36; s++) {
            const unsigned d = 16u * s + 8u * g;
            qf[s] = d < DK ? *(const bf16x8*)(Qa + qrow * DK + d)
                           : *(const bf16x8*)(Qr + qrow * DR + (d - DK));
            if (d >= DK && cosb) {
                // Raw q_rope: GPT-J interleaved RoPE, HeadNormRope's hd=64 skip_norm arithmetic
                // (pair m of the strip at table index pos*32 + m); fresh prefill: pos = kv_len - T + q.
                const unsigned qq = qv ? qi : T - 1u;
                const size_t tb = (size_t)((unsigned)kv_len[0] - T + qq) * 32u + (d - DK) / 2u;
#pragma unroll
                for (unsigned k = 0; k < 4; k++) {
                    const float c = cosb[tb + k], sn = sinb[tb + k];
                    const float a = (float)qf[s][2 * k], b = (float)qf[s][2 * k + 1];
                    qf[s][2 * k] = (bf16_t)(a * c - b * sn);
                    qf[s][2 * k + 1] = (bf16_t)(b * c + a * sn);
                }
            }
        }
        f32x16 o[8];
#pragma unroll
        for (unsigned n = 0; n < 8; n++) o[n] = f32x16{};
        float mrow = -INFINITY, lrow = 0.0f;
        const unsigned R = 32u * kh + r32;  // this lane's K row in the chunk (QK A operand)

        for (unsigned c = 0; c < nch; c++) {
            cp_async_wait();
            __syncthreads();  // B1: chunk c landed; chunk c-1 fully consumed
            if (c + 1 < nch) stage(c + 1);
            if (c + 2 < nch) ring(c + 2);
            const bf16* st = lh + (c & 1u) * STAGE;
            const unsigned* msk = (const unsigned*)(lds + RING_B + (c % RING) * SLOT_B + 256u);
            const unsigned c0 = c * BKV;

            f32x16 s = f32x16{};
            {
                const bf16* kp = st + mla_sp_off(R, 8u * g);
                const bf16* rp = st + 2u * SUB + R * 64u;
                bf16x8 ka[2];
                ka[0] = *(const bf16x8*)kp;
#pragma unroll
                for (unsigned x = 0; x < 36; x++) {
                    if (x + 1 < 36) {
                        const unsigned y = x + 1;
                        ka[y & 1] = y < 32 ? *(const bf16x8*)(kp + (y >> 4) * SUB + 32u * (y & 15u))
                                           : *(const bf16x8*)(rp + 8u * ((2u * (y - 32u) + g) ^ (R & 7u)));
                    }
                    s = plow_mfma_bf16_32x32(ka[x & 1], qf[x], s);
                }
            }
            // Mask: key k = 32kh + 4g + (i&3) + 8(i>>2); four consecutive keys per b128 read.
            float mx = -INFINITY;
#pragma unroll
            for (unsigned j = 0; j < 4; j++) {
                const unsigned kb = 32u * kh + 4u * g + 8u * j;
                const uint4 m4 = *(const uint4*)(msk + kb);
                const unsigned mm[4] = {m4.x, m4.y, m4.z, m4.w};
#pragma unroll
                for (unsigned e = 0; e < 4; e++) {
                    const unsigned i = 4u * j + e;
                    const bool on = c0 + kb + e < ucount && ((mm[e] >> ql) & 1u);
                    const float x = on ? s[i] * sl2 : -INFINITY;
                    s[i] = x;
                    mx = fmaxf(mx, x);
                }
            }
            mx = fmaxf(mx, __shfl_xor(mx, 32));
            if (g == 0u) stats[wave * 32u + r32] = mx;
            __syncthreads();  // B2: row maxima visible; every wave is done reading the rope rows
            mx = fmaxf(mx, stats[pw * 32u + r32]);
            if (__any(mx > mrow + 8.0f)) {
                const float mnew = fmaxf(mrow, mx);
                const float corr = mnew == -INFINITY ? 1.0f : __builtin_amdgcn_exp2f(mrow - mnew);
                lrow *= corr;
#pragma unroll
                for (unsigned n = 0; n < 8; n++) o[n] *= corr;
                mrow = mnew;
            }
            const float msafe = mrow == -INFINITY ? 0.0f : mrow;
            float ls = 0.0f;
            bf16x8 pf[2];
#pragma unroll
            for (unsigned i = 0; i < 16; i++) {
                const float e = __builtin_amdgcn_exp2f(s[i] - msafe);
                ls += e;
                pf[i >> 3][i & 7u] = (bf16_t)e;
            }
            lrow += ls + __shfl_xor(ls, 32);
            bf16x8* pex = (bf16x8*)(st + 2u * SUB);
            pex[(wave * 2u + 0u) * 64u + lane] = pf[0];
            pex[(wave * 2u + 1u) * 64u + lane] = pf[1];
            __syncthreads();  // B3: both P halves visible
            const bf16x8 pp0 = pex[(pw * 2u + 0u) * 64u + lane];
            const bf16x8 pp1 = pex[(pw * 2u + 1u) * 64u + lane];

            // O^T[dv][row] += V^T P^T; V rows = chunk keys of latent sub-tile kh (dv kh*256..+255).
            const bf16* vp = st + kh * SUB + mla_sp_off(4u * g + (i16 >> 2), g16 * 16u + (i16 & 3u) * 4u);
            auto vfrag = [&](unsigned ks, unsigned n) {
                const mla_sp::bx4 lo = mla_pf_ds_read_tr16(vp + ks * 8u * PS + n * 64u);
                const mla_sp::bx4 hi = mla_pf_ds_read_tr16(vp + (ks * 8u + 4u) * PS + n * 64u);
                return bf16x8{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
            };
            bf16x8 va[2];
            va[0] = vfrag(0, 0);
#pragma unroll
            for (unsigned x = 0; x < 32; x++) {
                const unsigned ks = x >> 3, n = x & 7u;
                if (x + 1 < 32) va[(x + 1) & 1] = vfrag((x + 1) >> 3, (x + 1) & 7u);
                const bool own = (ks >> 1) == kh;
                const bf16x8 pb = (ks & 1u) ? (own ? pf[1] : pp1) : (own ? pf[0] : pp0);
                o[n] = plow_mfma_bf16_32x32(va[x & 1], pb, o[n]);
            }
        }

        __syncthreads();
        if (g == 0u) stats[wave * 32u + r32] = lrow;
        __syncthreads();
        const float lt = lrow + stats[pw * 32u + r32];
        if (qv) {
            const float inv = lt > 0.0f ? 1.0f / lt : 0.0f;
            bf16* orow = O + qrow * DK + kh * 256u;
#pragma unroll
            for (unsigned n = 0; n < 8; n++)
#pragma unroll
                for (unsigned i4 = 0; i4 < 4; i4++) {
                    mla_sp::bx4 w;
#pragma unroll
                    for (unsigned c = 0; c < 4; c++) w[c] = (bf16_t)(o[n][i4 * 4 + c] * inv);
                    *(mla_sp::bx4*)(orow + n * 32u + 4u * g + 8u * i4) = w;
                }
        }
    }
}
