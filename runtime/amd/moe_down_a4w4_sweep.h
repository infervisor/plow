// A4W4 grouped MoE DOWN projection as a weight-streaming sweep (gfx950, CDNA4 scaled MFMA).
//
// The interpreter's op86 tiles DOWN like a GEMM (BM64 x BN256 x BK128/256), but DOWN's K is the
// per-rank expert width (GLM-5.3 TP8: I=256), so every tile gets one or two K-steps and never
// reaches a steady pipeline: at T=8192 it visits 27,576 tiles that each reload A and a slice of
// the expert's weights, and it runs at ~9% of the HBM roof at every rung.
//
// Here a work unit is (an m-subtile of TM rows of one expert, one H split). The subtile's fp4
// activations for ALL of K sit in registers for the whole unit, and each wave streams 16 output
// columns at a time straight from global into MFMA operands, DEPTH column groups in flight. No
// LDS, no K loop, no barrier: the op is pure weight streaming plus the output write.
//
// Output contract is op86's: gate-weighted f32 partials at part[row_partidx[row] * H + h], pad
// rows (row_partidx == PLOW_EXPERT_UNUSED) skipped. Accumulation order differs from op86 (16x16
// K128 MFMA vs 32x32 K64), so equality is to f32 rounding, not bitwise.
#pragma once

typedef int moe_sw_i32x8 __attribute__((ext_vector_type(8)));
typedef float moe_sw_f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned moe_sw_u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ moe_sw_i32x8 moe_sw_fp4(const unsigned char* p) {
    const moe_sw_u32x4 q = *reinterpret_cast<const moe_sw_u32x4*>(p);
    return moe_sw_i32x8{(int)q[0], (int)q[1], (int)q[2], (int)q[3], 0, 0, 0, 0};
}

template <int OPA>
__device__ __forceinline__ moe_sw_f32x4 moe_sw_mfma(moe_sw_i32x8 a, moe_sw_i32x8 b, moe_sw_f32x4 c,
                                                    int sa, int sb) {
    // cbsz/blgp = 4: both operands e2m1. OPA selects this 16-row block's byte of the A scale word.
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 4, 4, OPA, sa, 0, sb);
}

template <unsigned MI>
__device__ __forceinline__ void moe_sw_mfma_rows(moe_sw_f32x4* acc, const moe_sw_i32x8* a,
                                                 moe_sw_i32x8 b, int sa, int sb) {
    if constexpr (MI > 0) {
        moe_sw_mfma_rows<MI - 1>(acc, a, b, sa, sb);
        acc[MI - 1] = moe_sw_mfma<MI - 1>(a[MI - 1], b, acc[MI - 1], sa, sb);
    }
}

// TM rows per unit (16 or 64), KS = I / 128 K-steps, DEPTH column groups of loads in flight.
template <unsigned TM, unsigned KS, unsigned DEPTH>
__device__ void d_moe_down_a4w4_sweep(float* __restrict__ part, const unsigned char* __restrict__ fu,
                                      const unsigned char* __restrict__ fu_scale,
                                      const unsigned long long* __restrict__ wtab,
                                      const unsigned long long* __restrict__ stab,
                                      const int* __restrict__ meta,
                                      const unsigned* __restrict__ row_partidx,
                                      const float* __restrict__ row_gate, unsigned H, unsigned n_exp,
                                      unsigned nsplit, unsigned slice, unsigned nblk, unsigned waves) {
    static_assert(TM % 16 == 0 && TM <= 64 && 64 % TM == 0, "TM divides the 64-row routing tile");
    constexpr unsigned MI = TM / 16, SUB = 64 / TM;
    constexpr unsigned I = KS * 128, ROWB = I / 2, GROUPS = I / 32;
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned lane_k = lane >> 4, lane_n = lane & 15u;
    const int* rowoff = meta;
    const int* rowcnt = meta + n_exp;
    const int* tilep = meta + 2 * n_exp;
    const unsigned units = (unsigned)tilep[n_exp] * SUB * nsplit;
    const unsigned span = ((H + nsplit - 1u) / nsplit + 15u) & ~15u;

    for (unsigned u = slice; u < units; u += nblk) {
        const unsigned hs = u % nsplit, ms = u / nsplit;
        const unsigned mt = ms / SUB, sub = ms % SUB;
        unsigned lo = 0, hi = n_exp;  // expert owning 64-row tile mt
        while (lo + 1u < hi) {
            const unsigned mid = (lo + hi) >> 1;
            if ((unsigned)tilep[mid] <= mt) lo = mid; else hi = mid;
        }
        const unsigned e = lo;
        const unsigned rowbase = (unsigned)rowoff[e] + (mt - (unsigned)tilep[e]) * 64u + sub * TM;
        if (rowbase >= (unsigned)rowoff[e] + (unsigned)rowcnt[e]) continue;  // empty subtile
        const unsigned long long wb = wtab[(size_t)e * 3 + 2];
        if (wb == 0ull) continue;
        const unsigned char* W = (const unsigned char*)(size_t)wb;
        const unsigned char* S = (const unsigned char*)(size_t)stab[(size_t)e * 3 + 2];

        // A: every row fragment and scale this lane feeds, all of K, held for the unit.
        moe_sw_i32x8 a[KS][MI];
        int sa[KS];
#pragma unroll
        for (unsigned ks = 0; ks < KS; ks++) {
            sa[ks] = 0;
#pragma unroll
            for (unsigned mi = 0; mi < MI; mi++) {
                const unsigned row = rowbase + mi * 16u + lane_n;
                a[ks][mi] = moe_sw_fp4(fu + (size_t)row * ROWB + ks * 64u + lane_k * 16u);
                sa[ks] |= (int)fu_scale[(size_t)row * GROUPS + ks * 4u + lane_k] << (8 * mi);
            }
        }
        // Output rows this lane owns: 4*lane_k + r within each 16-row block.
        unsigned pid[MI][4];
        float gate[MI][4];
#pragma unroll
        for (unsigned mi = 0; mi < MI; mi++)
#pragma unroll
            for (unsigned r = 0; r < 4; r++) {
                const unsigned row = rowbase + mi * 16u + lane_k * 4u + r;
                pid[mi][r] = row_partidx[row];
                gate[mi][r] = pid[mi][r] != PLOW_EXPERT_UNUSED ? row_gate[row] : 0.0f;
            }

        const unsigned h0 = hs * span, h1 = min(H, h0 + span);
        const unsigned step = waves * 16u;
        // Software pipeline: DEPTH column groups of B operands+scales in registers.
        moe_sw_i32x8 b[DEPTH][KS];
        int sb[DEPTH][KS];
        auto load = [&](unsigned slot, unsigned c) {
            const unsigned col = c + lane_n;
            const bool ok = c < h1;
#pragma unroll
            for (unsigned ks = 0; ks < KS; ks++) {
                b[slot][ks] = ok ? moe_sw_fp4(W + (size_t)col * ROWB + ks * 64u + lane_k * 16u)
                                 : moe_sw_i32x8{0, 0, 0, 0, 0, 0, 0, 0};
                sb[slot][ks] = ok ? (int)S[(size_t)col * GROUPS + ks * 4u + lane_k] : 0;
            }
        };
        unsigned c = h0 + wave * 16u;
#pragma unroll
        for (unsigned d = 0; d < DEPTH; d++) load(d, c + d * step);
        for (; c < h1; c += DEPTH * step) {
#pragma unroll
            for (unsigned d = 0; d < DEPTH; d++) {
                const unsigned cd = c + d * step;
                moe_sw_f32x4 acc[MI];
#pragma unroll
                for (unsigned mi = 0; mi < MI; mi++) acc[mi] = moe_sw_f32x4{0, 0, 0, 0};
#pragma unroll
                for (unsigned ks = 0; ks < KS; ks++)
                    moe_sw_mfma_rows<MI>(acc, a[ks], b[d][ks], sa[ks], sb[d][ks]);
                load(d, cd + DEPTH * step);  // refill this slot while the next group computes
                if (cd < h1) {
                    const unsigned col = cd + lane_n;
#pragma unroll
                    for (unsigned mi = 0; mi < MI; mi++)
#pragma unroll
                        for (unsigned r = 0; r < 4; r++)
                            if (pid[mi][r] != PLOW_EXPERT_UNUSED)
                                part[(size_t)pid[mi][r] * H + col] = gate[mi][r] * acc[mi][r];
                }
            }
        }
    }
}

// Launch-shape choice from the routing metadata, the same in every workgroup: TM=16 while most
// experts hold one 64-row tile or none (decode rows; measured faster at T<=16), else TM=64; the H
// split sized so live units cover ~2 per workgroup. `red` is one word of LDS.
template <unsigned KS>
__device__ void moe_down_a4w4_sweep_auto(float* part, const unsigned char* fu,
                                         const unsigned char* fu_scale,
                                         const unsigned long long* wtab,
                                         const unsigned long long* stab, const int* meta,
                                         const unsigned* row_partidx, const float* row_gate,
                                         unsigned H, unsigned n_exp, unsigned slice, unsigned nblk,
                                         unsigned* red) {
    const unsigned tiles = (unsigned)meta[3 * n_exp];
    const bool small = tiles * 2u < n_exp;
    const unsigned tm = small ? 16u : 64u;
    if (threadIdx.x < 64) {
        unsigned live = 0;
        for (unsigned e = threadIdx.x; e < n_exp; e += 64)
            live += ((unsigned)meta[n_exp + e] + tm - 1u) / tm;
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) live += __shfl_xor(live, off, 64);
        if (threadIdx.x == 0) red[0] = live;
    }
    __syncthreads();
    const unsigned live = red[0];
    __syncthreads();
    unsigned nsplit = live ? (2u * nblk + live - 1u) / live : 1u;
    nsplit = nsplit < 1u ? 1u : (nsplit > H / 64u ? H / 64u : nsplit);
    const unsigned waves = blockDim.x / 64u;
    if (small)
        d_moe_down_a4w4_sweep<16, KS, 4>(part, fu, fu_scale, wtab, stab, meta, row_partidx, row_gate,
                                         H, n_exp, nsplit, slice, nblk, waves);
    else
        d_moe_down_a4w4_sweep<64, KS, 4>(part, fu, fu_scale, wtab, stab, meta, row_partidx, row_gate,
                                         H, n_exp, nsplit, slice, nblk, waves);
}
