/* DSA prefill top-k SELECT v3 (op 118, PLOW_DSA_SELECT_V3) and its fusion with the per-8-query
 * UNION build (op 119's table, written by op 118 when the packet asks for it).
 *
 * Same set as d_index_select_pf<true> / d_index_select_pf_v2: key = monotone fp32 score bits,
 * score descending, lowest index on exact ties; rows <= top_k are the identity (-1 padded).
 *
 * Per row (row resident in VGPRs, the next row fetched while this one is ranked, as in v2):
 *   1. one B1-bit histogram of the top key bits (sign/exponent/mantissa of the fp32 key);
 *   2. every wave scans the whole histogram itself (no cross-wave find, one barrier per pass);
 *   3. one register pass emits every key above the threshold bin and compacts the bin's keys
 *      (score bits, position) into LDS;
 *   4. the remaining score bits and, only on an exact boundary tie, the index bits are resolved
 *      by digit passes over those candidates; a boundary group needed whole ends early, and one
 *      of <= 64 candidates is ranked exactly by a single wave.
 * A threshold bin larger than CAP never truncates: the digit passes then run over the registers.
 * Histograms rotate through three LDS buffers so each pass needs exactly one barrier (a buffer
 * is zeroed one pass after its scan and filled one pass after that).
 * Emission order within a row is unspecified, as for every selector. */
#pragma once

#ifndef IDXSEL_V3_B1
#define IDXSEL_V3_B1 11u
#endif
#ifndef IDXSEL_V3_CAP
#define IDXSEL_V3_CAP 4096u
#endif

namespace isel3 {
constexpr unsigned PER = IDXSEL_V2_PER;
constexpr unsigned MAXROW = PER * PLOW_THREADS;
constexpr unsigned B1 = IDXSEL_V3_B1, SH1 = 32u - B1;
constexpr unsigned NBH = 2048u; /* bins per histogram buffer (the widest digit) */
constexpr unsigned CAP = IDXSEL_V3_CAP;
/* LDS words: hist[3][NBH] | red[32] | fallback[SEL_NB + 8] | cand key[CAP] | cand pos[CAP] |
 * pack mask[MAXROW / 4] (fused only). */
constexpr unsigned RED = 3u * NBH, FB = RED + 32u, CKEY = FB + SEL_NB + 8u, CPOS = CKEY + CAP,
                   MASK = CPOS + CAP;
constexpr unsigned WORDS_SEL = MASK, WORDS_FUSED = MASK + MAXROW / 4u;
constexpr unsigned PACK = 8u;
static_assert(B1 == 11u || B1 == 12u, "first digit is 11 or 12 bits");
static_assert(MAXROW <= 16384u, "index digits cover 14 bits");
static_assert(CAP % PLOW_THREADS == 0u && CAP / PLOW_THREADS <= 32u, "candidate emission bitmask");
}  // namespace isel3

/* Workgroup barrier for LDS traffic only. __syncthreads() also drains this wave's outstanding
 * global loads (vmcnt(0)), which would wait out the next row's prefetch at every barrier. */
__device__ __forceinline__ void isel3_sync() {
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

/* Rotating-histogram pass counter and the parity of the two slot counters (red[0], red[1]). */
struct Isel3State {
    unsigned pass, par;
};
#ifdef ISEL3_PROF
#define ISEL3_T(i) do { if (blockIdx.x == 0 && threadIdx.x == 0) { const unsigned long long n_ = __builtin_amdgcn_s_memtime(); g_isel3_prof[i] += n_ - prof_t; prof_t = n_; } } while (0)
#define ISEL3_T0() unsigned long long prof_t = __builtin_amdgcn_s_memtime()
#else
#define ISEL3_T(i)
#define ISEL3_T0()
#endif

/* Zero all three histograms and both counters. Ends with a barrier. */
__device__ __forceinline__ void isel3_init(unsigned* lds, Isel3State& st) {
    for (unsigned i = threadIdx.x; i < 3u * isel3::NBH; i += PLOW_THREADS) lds[i] = 0u;
    if (threadIdx.x < 2u) lds[isel3::RED + threadIdx.x] = 0u;
    st.pass = 0u;
    st.par = 0u;
    __syncthreads();
}

/* Wave exclusive prefix sum and total of v < 2^nbits, from bit-plane ballots: no LDS round
 * trips (a shuffle scan is six dependent ds_bpermutes). */
__device__ __forceinline__ unsigned isel3_wave_excl(unsigned v, unsigned nbits, unsigned& total) {
    unsigned ex = 0u, tot = 0u;
    for (unsigned b = 0; b < nbits; b++) {
        const unsigned long long m = __ballot((v >> b) & 1u);
        ex += __builtin_amdgcn_mbcnt_hi((unsigned)(m >> 32),
                                        __builtin_amdgcn_mbcnt_lo((unsigned)m, 0u)) << b;
        tot += (unsigned)__builtin_popcountll(m) << b;
    }
    total = tot;
    return ex;
}

/* Barrier after a histogram pass, then this wave's scan of it: the bin d holding the k_rem-th
 * largest (bins counted from the top), the population strictly above it, and its own. The buffer
 * scanned one pass earlier is zeroed here (every wave finished that scan before this barrier;
 * it is filled again only after the next pass's barrier). */
__device__ __forceinline__ void isel3_pass_end(unsigned* lds, Isel3State& st, unsigned k_rem,
                                               unsigned& d, unsigned& above, unsigned& bnd) {
    using namespace isel3;
    isel3_sync();
    const uint4* const h = (const uint4*)(lds + (st.pass % 3u) * NBH);
    unsigned* const z = lds + ((st.pass + 2u) % 3u) * NBH;
    st.pass++;
    for (unsigned i = threadIdx.x; i < NBH; i += PLOW_THREADS) z[i] = 0u;
    const unsigned lane = threadIdx.x & 63u;
    constexpr unsigned G = NBH / 64u / 4u; /* uint4 groups per lane */
    const unsigned g0 = (NBH / 4u) - G * (lane + 1u);
    unsigned g[G], local = 0u;
#pragma unroll
    for (unsigned j = 0; j < G; j++) {
        const uint4 q = h[g0 + j];
        g[j] = q.x + q.y + q.z + q.w;
        local += g[j];
    }
    unsigned total;
    const unsigned excl = isel3_wave_excl(local, 15u, total); /* a row holds <= 16384 keys */
    const unsigned long long hit = __ballot(excl < k_rem && excl + local >= k_rem);
    const unsigned L = (unsigned)__builtin_ctzll(hit);
    unsigned acc = excl, gj = 0u, gacc = excl;
    bool f = false;
#pragma unroll
    for (int j = (int)G - 1; j >= 0; j--) {
        if (!f && acc + g[j] >= k_rem) {
            f = true;
            gj = (unsigned)j;
            gacc = acc;
        }
        acc += g[j];
    }
    const uint4 q = h[g0 + gj];
    const unsigned x[4] = {q.x, q.y, q.z, q.w};
    unsigned dd = 0u, ab = 0u, bb = 0u;
    acc = gacc;
    f = false;
#pragma unroll
    for (int w = 3; w >= 0; w--) {
        if (!f && acc + x[w] >= k_rem) {
            f = true;
            dd = (g0 + gj) * 4u + (unsigned)w;
            ab = acc;
            bb = x[w];
        }
        acc += x[w];
    }
    d = (unsigned)__builtin_amdgcn_readlane((int)dd, (int)L);
    above = (unsigned)__builtin_amdgcn_readlane((int)ab, (int)L);
    bnd = (unsigned)__builtin_amdgcn_readlane((int)bb, (int)L);
}

__device__ __forceinline__ unsigned* isel3_hist(unsigned* lds, const Isel3State& st) {
    return lds + (st.pass % 3u) * isel3::NBH;
}

/* Slot allocation: each thread claims ne emitted and nc candidate slots (each <= 32) from the
 * packed LDS counter (ballot scan, one atomic per wave). Called by every thread of a wave. */
__device__ __forceinline__ void isel3_alloc2(unsigned ne, unsigned nc, unsigned* ctr, unsigned& oe,
                                             unsigned& oc) {
    unsigned te, tc;
    const unsigned xe = isel3_wave_excl(ne, 6u, te), xc = isel3_wave_excl(nc, 6u, tc);
    unsigned base = 0u;
    if ((threadIdx.x & 63u) == 0u && (te | tc) != 0u) base = atomicAdd(ctr, te | (tc << 16));
    base = (unsigned)__builtin_amdgcn_readfirstlane((int)base);
    oe = (base & 0xFFFFu) + xe;
    oc = (base >> 16) + xc;
}
__device__ __forceinline__ unsigned isel3_alloc(unsigned ne, unsigned* ctr) {
    unsigned oe, oc;
    isel3_alloc2(ne, 0u, ctr, oe, oc);
    return oe;
}

/* Taken positions go to one idx row at their slot. */
struct Isel3EmitRow {
    PLOW_GLOB int* row;
    unsigned top_k;
    __device__ __forceinline__ void operator()(unsigned o, unsigned s) const {
        if (o < top_k) st_act<int>(&row[o], (int)s);
    }
};

/* Taken positions set query q's bit in the pack mask (one byte per position); slots unused. */
struct Isel3EmitMask {
    unsigned* mask;
    unsigned bit;
    __device__ __forceinline__ void operator()(unsigned, unsigned s) const {
        atomicOr(&mask[s >> 2], bit << ((s & 3u) * 8u));
    }
};

/* Digit passes over the boundary candidates, from LDS (REGS = false) or, when the threshold bin
 * overflowed CAP, straight from the row registers (REGS = true), then emission of the candidates
 * the digits select. Entered with k_rem < nc (the group is not needed whole). */
template <bool REGS, class Emit>
__device__ __forceinline__ void isel3_refine(const unsigned (&sb)[isel3::PER], unsigned row_len,
                                             unsigned thr, unsigned nc, unsigned k_rem,
                                             unsigned* lds, Isel3State& st, unsigned* ctr,
                                             const Emit& emit) {
    using namespace isel3;
    const unsigned* const ckey = lds + CKEY;
    unsigned* const cpos = lds + CPOS;
    const unsigned tid = threadIdx.x, lane = tid & 63u;
    unsigned sprefix = thr << SH1, smask = ((1u << B1) - 1u) << SH1, iprefix = 0u, imask = 0u;
    /* A zero key marks a slot past the row; with thr == 0 it would match the prefix. */
    auto live = [&](unsigned k, unsigned ik) {
        return k != 0u && (k & smask) == sprefix && (ik & imask) == iprefix;
    };
    unsigned bnd = nc;
    bool ranked = false;
    ISEL3_T0();
    for (unsigned p = 0; p < 4u; p++) {
        if (!REGS && bnd <= 64u) {
            ranked = true;
            break;
        }
        /* score bits below the first digit: 11 + (SH1 - 11); index (< 2^14): 11 + 3. */
        const bool isx = p >= 2u;
        const unsigned sh = p == 0u ? SH1 - 11u : p == 2u ? 3u : 0u;
        const unsigned nb = 1u << (p == 1u ? SH1 - 11u : p == 3u ? 3u : 11u);
        unsigned* const hist = isel3_hist(lds, st);
        if (REGS) {
#pragma unroll
            for (unsigned u = 0; u < PER; u++) {
                if (u * PLOW_THREADS >= row_len) break;
                const unsigned s = tid + u * PLOW_THREADS;
                const unsigned k = sb[u], ik = row_len - 1u - s;
                if (live(k, ik)) atomicAdd(&hist[((isx ? ik : k) >> sh) & (nb - 1u)], 1u);
            }
        } else {
            for (unsigned i = tid; i < nc; i += PLOW_THREADS) {
                const unsigned k = ckey[i], ik = row_len - 1u - cpos[i];
                if (live(k, ik)) atomicAdd(&hist[((isx ? ik : k) >> sh) & (nb - 1u)], 1u);
            }
        }
        unsigned d, above;
        isel3_pass_end(lds, st, k_rem, d, above, bnd);
        k_rem -= above;
        if (isx) {
            iprefix |= d << sh;
            imask |= (nb - 1u) << sh;
        } else {
            sprefix |= d << sh;
            smask |= (nb - 1u) << sh;
        }
        if (bnd == k_rem) break;
    }
    ISEL3_T(11);
    if (!REGS && ranked) {
        /* Wave 0 gathers the bnd candidates matching the prefix and keeps the k_rem best by
         * (score, lowest index); they are flagged with bit 31 of their position. */
        if (tid < 64u) {
            unsigned* const g = lds + FB;
            unsigned got = 0u;
            for (unsigned b = 0; b < nc; b += 64u) {
                const unsigned i = b + lane;
                bool m = false;
                if (i < nc) m = live(ckey[i], row_len - 1u - cpos[i]);
                const unsigned long long bm = __ballot(m);
                if (m) g[got + (unsigned)__builtin_popcountll(bm & ((1ull << lane) - 1ull))] = i;
                got += (unsigned)__builtin_popcountll(bm);
            }
            const unsigned i = lane < bnd ? g[lane] : 0u;
            const unsigned k = lane < bnd ? ckey[i] : 0u;
            const unsigned ik = lane < bnd ? row_len - 1u - cpos[i] : 0u;
            unsigned rank = 0u;
            for (unsigned j = 0; j < bnd; j++) {
                const unsigned kj = (unsigned)__builtin_amdgcn_readlane((int)k, (int)j);
                const unsigned ij = (unsigned)__builtin_amdgcn_readlane((int)ik, (int)j);
                rank += kj > k || (kj == k && ij > ik);
            }
            if (lane < bnd && rank < k_rem) cpos[i] |= 0x80000000u;
        }
        isel3_sync();
    }
    ISEL3_T(12);
    /* Taken = lexicographically above the resolved prefix over (score digits, index digits), or
     * equal to it when the digits resolved the group (not when a wave ranked it). */
    auto take = [&](unsigned k, unsigned ik) {
        const unsigned ks = k & smask, ix = ik & imask;
        return k != 0u && (k >> SH1) == thr &&
               (ks > sprefix || (ks == sprefix && (ix > iprefix || (!ranked && ix == iprefix))));
    };
    if (REGS) {
        unsigned tm = 0u;
#pragma unroll
        for (unsigned u = 0; u < PER; u++) {
            if (u * PLOW_THREADS >= row_len) break;
            const unsigned s = tid + u * PLOW_THREADS;
            tm |= (unsigned)take(sb[u], row_len - 1u - s) << u;
        }
        unsigned o = isel3_alloc((unsigned)__builtin_popcount(tm), ctr);
        while (tm) {
            const unsigned u = (unsigned)__builtin_ctz(tm);
            tm &= tm - 1u;
            emit(o++, tid + u * PLOW_THREADS);
        }
    } else {
        constexpr unsigned NJ = CAP / PLOW_THREADS;
        unsigned tm = 0u;
        for (unsigned j = 0; j < NJ; j++) {
            const unsigned i = tid + j * PLOW_THREADS;
            if (i >= nc) break;
            const unsigned p = cpos[i];
            tm |= (unsigned)((p >> 31) || take(ckey[i], row_len - 1u - p)) << j;
        }
        unsigned o = isel3_alloc((unsigned)__builtin_popcount(tm), ctr);
        while (tm) {
            const unsigned j = (unsigned)__builtin_ctz(tm);
            tm &= tm - 1u;
            emit(o++, cpos[tid + j * PLOW_THREADS] & 0x7FFFFFFFu);
        }
    }
}

/* One row, top_k < row_len <= MAXROW. sb = monotone keys (0 past the row). No trailing barrier:
 * everything a later row writes is ordered behind that row's first histogram barrier. The slot
 * counter alternates per row (reset one row later, behind that barrier); it packs emitted (low
 * 16 bits, <= top_k) and candidate (high 16 bits) slots. */
template <class Emit>
__device__ __forceinline__ void isel3_row(const unsigned (&sb)[isel3::PER], unsigned row_len,
                                          unsigned top_k, unsigned* lds, Isel3State& st,
                                          const Emit& emit) {
    using namespace isel3;
    unsigned* const red = lds + RED;
    unsigned* const ckey = lds + CKEY;
    unsigned* const cpos = lds + CPOS;
    const unsigned tid = threadIdx.x;
    unsigned* const hist = isel3_hist(lds, st);
    ISEL3_T0();
#pragma unroll
    for (unsigned u = 0; u < PER; u++) {
        if (u * PLOW_THREADS >= row_len) break;
        const unsigned k = sb[u];
        if (k != 0u) atomicAdd(&hist[k >> SH1], 1u);
    }
    ISEL3_T(0);
    unsigned thr, above, nc;
    isel3_pass_end(lds, st, top_k, thr, above, nc);
    ISEL3_T(1);
    unsigned* const ctr = red + st.par;
    st.par ^= 1u;
    if (tid == 0u) red[st.par] = 0u;
    const unsigned k_rem = top_k - above;
    const bool whole = nc == k_rem, fits = nc <= CAP;
    const bool keep = !whole && fits;
    unsigned tm = 0u, cm = 0u;
#pragma unroll
    for (unsigned u = 0; u < PER; u++) {
        if (u * PLOW_THREADS >= row_len) break;
        const unsigned k = sb[u], bin = k >> SH1;
        tm |= (unsigned)(k != 0u && (bin > thr || (whole && bin == thr))) << u;
        cm |= (unsigned)(keep && k != 0u && bin == thr) << u;
    }
    ISEL3_T(8);
    unsigned o, oc;
    isel3_alloc2((unsigned)__builtin_popcount(tm), (unsigned)__builtin_popcount(cm), ctr, o, oc);
    ISEL3_T(9);
    while (tm) {
        const unsigned u = (unsigned)__builtin_ctz(tm);
        tm &= tm - 1u;
        emit(o++, tid + u * PLOW_THREADS);
    }
    ISEL3_T(10);
    if (cm) {
#pragma unroll
        for (unsigned u = 0; u < PER; u++) {
            if ((cm >> u) & 1u) {
                ckey[oc] = sb[u];
                cpos[oc] = tid + u * PLOW_THREADS;
                oc++;
            }
        }
    }
    ISEL3_T(2);
    if (!whole) {
        if (fits) {
            isel3_sync();
            isel3_refine<false>(sb, row_len, thr, nc, k_rem, lds, st, ctr, emit);
        } else {
            isel3_refine<true>(sb, row_len, thr, nc, k_rem, lds, st, ctr, emit);
        }
    }
    ISEL3_T(3);
#ifdef ISEL3_PROF
    if (blockIdx.x == 0 && threadIdx.x == 0) g_isel3_prof[6]++;
#endif
}

/* Row fetch with next-row prefetch (v2's pipeline). Raw bits for rows the register path ranks.
 * Unconditional buffer loads (out-of-range lanes read 0): no per-load branch, so the compiler's
 * wait for these loads lands at their use one row later rather than right after the issue. */
struct Isel3Fetch {
    const float* Sc;
    unsigned q_pos0, n_tok, top_k, kv_stride;
    __device__ __forceinline__ void operator()(unsigned (&nxt)[isel3::PER], unsigned t) const {
        const unsigned rl = q_pos0 + t + 1u;
        const bool want = t < n_tok && rl > top_k && rl <= isel3::MAXROW;
        const unsigned long long a = (unsigned long long)(size_t)(Sc + (size_t)t * kv_stride);
        const unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)a);
        const unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(a >> 32));
        const __amdgpu_buffer_rsrc_t r = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(size_t)(((unsigned long long)hi << 32) | lo), (short)0, want ? rl * 4u : 0u,
            PLOW_BUF_RSRC3);
#pragma unroll
        for (unsigned u = 0; u < isel3::PER; u++)
            nxt[u] = __builtin_amdgcn_raw_buffer_load_b32(r, threadIdx.x * 4u, u * PLOW_THREADS * 4u,
                                                          /*glc|slc*/ 3);
    }
};

__device__ __forceinline__ void isel3_keys(unsigned (&sb)[isel3::PER],
                                           const unsigned (&nxt)[isel3::PER], unsigned row_len) {
    const unsigned nv = row_len > threadIdx.x ? (row_len - threadIdx.x + PLOW_THREADS - 1u) / PLOW_THREADS : 0u;
#pragma unroll
    for (unsigned u = 0; u < isel3::PER; u++) {
        const unsigned b = nxt[u];
        sb[u] = u < nv ? ((b & 0x80000000u) ? ~b : (b | 0x80000000u)) : 0u;
    }
}

/* Ranks the fetched row (nxt) and refills nxt with row t_next while this one is ranked. */
template <class Emit>
__device__ __forceinline__ void isel3_step(unsigned (&nxt)[isel3::PER], const Isel3Fetch& fetch,
                                           unsigned t_next, unsigned row_len, unsigned top_k,
                                           unsigned* lds, Isel3State& st, const Emit& emit) {
    ISEL3_T0();
    unsigned sb[isel3::PER];
    isel3_keys(sb, nxt, row_len);
    fetch(nxt, t_next);
    ISEL3_T(4);
    isel3_row(sb, row_len, top_k, lds, st, emit);
    ISEL3_T(5);
}

/* Selects row t into idx (identity rows, register rows, or the long-row fallback), then leaves
 * row t_next in nxt. nxt is dead across the fallback, which keeps it out of its registers. */
__device__ __forceinline__ void isel3_select_row(PLOW_GLOB int* ib, const float* Score, const int* kv_len,
                                                 unsigned n_tok, unsigned top_k,
                                                 unsigned kv_stride, unsigned t, unsigned t_next,
                                                 unsigned (&nxt)[isel3::PER],
                                                 const Isel3Fetch& fetch, unsigned* lds,
                                                 Isel3State& st) {
    const unsigned row_len = fetch.q_pos0 + t + 1u;
    PLOW_GLOB int* const row = ib + (size_t)t * top_k;
    if (row_len <= top_k) {
        for (unsigned s = threadIdx.x; s < top_k; s += PLOW_THREADS)
            st_act<int>(&row[s], s < row_len ? (int)s : -1);
        __syncthreads();
        fetch(nxt, t_next);
    } else if (row_len > isel3::MAXROW) {
        d_index_select_pf<true>((int*)ib, Score, kv_len, n_tok, top_k, kv_stride, 0u, 1u,
                                lds + isel3::FB, lds + isel3::FB + SEL_NB, 1u, t, t + 1u);
        fetch(nxt, t_next);
    } else {
        isel3_step(nxt, fetch, t_next, row_len, top_k, lds, st, Isel3EmitRow{row, top_k});
    }
}

/* op 118 under PLOW_DSA_SELECT_V3: [T][top_k] idx rows, grid-strided over rows. */
__device__ void d_index_select_pf_v3(int* __restrict__ idx, const float* __restrict__ Score,
                                     const int* __restrict__ kv_len, unsigned n_tok,
                                     unsigned top_k, unsigned kv_stride, unsigned slice,
                                     unsigned nblk, unsigned* lds) {
    PLOW_GLOB int* const ib = as_glob(idx);
    const unsigned len = (unsigned)as_glob(kv_len)[0];
    const Isel3Fetch fetch{Score, len - n_tok, n_tok, top_k, kv_stride};
    Isel3State st;
    isel3_init(lds, st);
    unsigned nxt[isel3::PER];
    fetch(nxt, slice);
    for (unsigned t = slice; t < n_tok; t += nblk)
        isel3_select_row(ib, Score, kv_len, n_tok, top_k, kv_stride, t, t + nblk, nxt, fetch, lds,
                         st);
}

/* Appends the mask's positions [w0, w0 + 4*nw) (ascending) to the pack's union block. One
 * barrier per 512-word chunk: wave totals go to a parity slot (red[16..31]) that is rewritten
 * only two chunks later, behind the next chunk's barrier. */
__device__ __forceinline__ unsigned isel3_compact(const unsigned* mask, unsigned nw, unsigned w0,
                                                  unsigned base, unsigned cap, PLOW_GLOB int* upos,
                                                  PLOW_GLOB unsigned* ulo, PLOW_GLOB unsigned* uhi,
                                                  unsigned* red) {
    const unsigned tid = threadIdx.x, wave = tid >> 6;
    for (unsigned c0 = 0, par = 0; c0 < nw; c0 += PLOW_THREADS, par ^= 8u) {
        const unsigned w = c0 + tid;
        const unsigned x = w < nw ? mask[w] : 0u;
        unsigned n = 0u;
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) n += ((x >> (8u * j)) & 0xFFu) != 0u;
        unsigned wt;
        const unsigned wx = isel3_wave_excl(n, 3u, wt);
        if ((tid & 63u) == 0u) red[16u + par + wave] = wt;
        isel3_sync();
        unsigned o = base + wx, total = 0u;
#pragma unroll
        for (unsigned v = 0; v < PLOW_WAVES; v++) {
            const unsigned c = red[16u + par + v];
            o += v < wave ? c : 0u;
            total += c;
        }
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            const unsigned m = (x >> (8u * j)) & 0xFFu;
            if (m != 0u) {
                if (o < cap) {
                    st_act<int>(&upos[o], (int)(w0 + w * 4u + j));
                    st_act<unsigned>(&ulo[o], m);
                    st_act<unsigned>(&uhi[o], 0u);
                }
                o++;
            }
        }
        base += total;
    }
    return base;
}

/* op 118 fused with op 119 (pack = 8): one workgroup per 8-query pack selects its rows straight
 * into an LDS position mask and writes the pack's union block exactly as d_index_union_pf does
 * (count header, ascending positions, maskLo = the pack's query bits, maskHi = 0, count clamped
 * to cap), and zeroes the sparse flash's pack ticket counter when zero_ctr is set. idx is written
 * only for packs whose causal bound exceeds the mask (they select to idx, then build the union
 * from it in MAXROW-position windows). */
__device__ void d_index_select_union_pf(unsigned char* __restrict__ uni, int* __restrict__ idx,
                                        const float* __restrict__ Score,
                                        const int* __restrict__ kv_len, unsigned n_tok,
                                        unsigned top_k, unsigned kv_stride, unsigned cap,
                                        unsigned zero_ctr, unsigned slice, unsigned nblk,
                                        unsigned* lds) {
    using namespace isel3;
    const unsigned tid = threadIdx.x;
    PLOW_GLOB int* const ib = as_glob(idx);
    const unsigned len = (unsigned)as_glob(kv_len)[0];
    const Isel3Fetch fetch{Score, len - n_tok, n_tok, top_k, kv_stride};
    const unsigned q_pos0 = fetch.q_pos0;
    const unsigned n_qt = (n_tok + PACK - 1u) / PACK;
    const unsigned hdr = (n_qt * 4u + 255u) / 256u * 256u;
    PLOW_GLOB unsigned* const cnt = (PLOW_GLOB unsigned*)as_glob(uni);
    unsigned* const mask = lds + MASK;
    unsigned* const red = lds + RED;
    if (zero_ctr && slice == 0u && tid < 2u)
        st_act<unsigned>((PLOW_GLOB unsigned*)(as_glob(uni) + hdr + (size_t)n_qt * cap * 12u) + tid, 0u);
    Isel3State st;
    isel3_init(lds, st);
    unsigned nxt[PER];
    fetch(nxt, slice * PACK);
    for (unsigned qt = slice; qt < n_qt; qt += nblk) {
        const unsigned q_lo = qt * PACK;
        const unsigned q_hi = q_lo + PACK - 1u < n_tok - 1u ? q_lo + PACK - 1u : n_tok - 1u;
        const unsigned tile_end = q_pos0 + q_hi + 1u;
        PLOW_GLOB unsigned char* const blk = as_glob(uni) + hdr + (size_t)qt * cap * 12u;
        PLOW_GLOB int* const upos = (PLOW_GLOB int*)blk;
        PLOW_GLOB unsigned* const ulo = (PLOW_GLOB unsigned*)(blk + (size_t)cap * 4u);
        PLOW_GLOB unsigned* const uhi = (PLOW_GLOB unsigned*)(blk + (size_t)cap * 8u);
        const bool in_lds = tile_end <= MAXROW;
        if (in_lds)
            for (unsigned w = tid; w < (tile_end + 3u) / 4u; w += PLOW_THREADS) mask[w] = 0u;
        isel3_sync();
        for (unsigned t = q_lo; t <= q_hi; t++) {
            const unsigned row_len = q_pos0 + t + 1u;
            const unsigned t_next = t < q_hi ? t + 1u : (qt + nblk) * PACK;
            if (!in_lds) {
                isel3_select_row(ib, Score, kv_len, n_tok, top_k, kv_stride, t, t_next, nxt,
                                 fetch, lds, st);
                continue;
            }
            const unsigned bit = 1u << (t - q_lo);
            if (row_len <= top_k) {
                for (unsigned s = tid; s < row_len; s += PLOW_THREADS)
                    atomicOr(&mask[s >> 2], bit << ((s & 3u) * 8u));
                __syncthreads();
                fetch(nxt, t_next);
            } else {
                isel3_step(nxt, fetch, t_next, row_len, top_k, lds, st,
                           Isel3EmitMask{mask, bit});
            }
        }
        isel3_sync();
        unsigned base = 0u;
        if (in_lds) {
            base = isel3_compact(mask, (tile_end + 3u) / 4u, 0u, 0u, cap, upos, ulo, uhi, red);
        } else {
            /* idx rows were stored by this workgroup; L1-bypassing loads read them back. */
            __threadfence_block();
            __syncthreads();
            for (unsigned w0 = 0; w0 < tile_end; w0 += MAXROW) {
                const unsigned wend = w0 + MAXROW < tile_end ? w0 + MAXROW : tile_end;
                const unsigned nw = (wend - w0 + 3u) / 4u;
                for (unsigned w = tid; w < nw; w += PLOW_THREADS) mask[w] = 0u;
                __syncthreads();
                for (unsigned e = tid; e < (q_hi + 1u - q_lo) * top_k; e += PLOW_THREADS) {
                    const unsigned ql = e / top_k;
                    const int s = __hip_atomic_load(&idx[(size_t)(q_lo + ql) * top_k + e % top_k],
                                                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    if (s >= (int)w0 && (unsigned)s < wend)
                        atomicOr(&mask[(s - w0) >> 2], (1u << ql) << (((s - w0) & 3u) * 8u));
                }
                __syncthreads();
                base = isel3_compact(mask, nw, w0, base, cap, upos, ulo, uhi, red);
            }
        }
        if (tid == 0u) st_act<unsigned>(&cnt[qt], base < cap ? base : cap);
        isel3_sync();
    }
}
