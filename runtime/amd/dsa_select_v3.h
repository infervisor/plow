/* DSA prefill top-k SELECT v3 (op 118, PLOW_DSA_SELECT_V3) and its fusion with the per-8-query
 * UNION build (op 119's table, written by op 118 when the packet asks for it).
 *
 * Same set as d_index_select_pf<true> / d_index_select_pf_v2: key = monotone fp32 score bits,
 * score descending, lowest index on exact ties; rows <= top_k are the identity (-1 padded).
 *
 * ONE WAVE PER ROW, no workgroup barrier inside a row. v2 ranks a row with the whole workgroup
 * and pays ~15 barrier / LDS round trips per row whatever its length; here the eight waves rank
 * eight rows side by side (in the fused form: the eight rows of one query pack):
 *   1. stream the row from global into an 11-bit histogram of the key's top bits (fp32 sign /
 *      exponent / mantissa — not fp16 bins) in the wave's own LDS region;
 *   2. the wave scans it for the threshold bin; while that bin holds more than CAPW keys (and is
 *      not needed whole) further 11-bit digits are resolved by more streaming passes, so an
 *      overflowing bin is never truncated;
 *   3. a second streaming pass emits every key above the resolved prefix and compacts the keys
 *      equal to it (score bits, position) into LDS;
 *   4. 8-bit digit passes over those candidates, score bits first and index bits (lowest index
 *      wins) only on an exact tie, until the boundary group is needed whole or has <= 16 keys,
 *      which are ranked exactly by (score, lowest index).
 * Emission order within a row is unspecified, as for every selector. */
#pragma once

namespace isel3 {
constexpr unsigned NW = PLOW_WAVES;         /* waves per workgroup = rows in flight */
constexpr unsigned REGW = 3840u;            /* LDS words per wave */
constexpr unsigned NBH = 2048u, B1 = 11u;   /* streaming digits */
constexpr unsigned CAPW = (REGW - 256u) / 2u; /* boundary candidates per wave */
constexpr unsigned CKEY = 0u, CPOS = CAPW, SHIST = 2u * CAPW; /* region layout after the scan */
constexpr unsigned MAXPOS = 16384u;         /* positions the fused pack mask covers */
constexpr unsigned MASK = NW * REGW;
constexpr unsigned WORDS_SEL = NW * REGW, WORDS_FUSED = MASK + MAXPOS / 4u;
constexpr unsigned PACK = 8u;
#ifndef IDXSEL_V3_BATCH
#define IDXSEL_V3_BATCH 8u
#endif
constexpr unsigned BATCH = IDXSEL_V3_BATCH;  /* 16-byte loads per lane per streaming step */
static_assert(NW == PACK, "the fused form ranks a pack's rows one per wave");
static_assert(NBH <= REGW && REGW % 4u == 0u && CAPW % 64u == 0u, "wave region layout");
}  // namespace isel3

/* LDS ordering within one wave: its LDS operations execute in order, so waiting for this lane's
 * own makes every lane's earlier writes visible to the reads that follow. */
__device__ __forceinline__ void isel3_wsync() { __asm__ volatile("s_waitcnt lgkmcnt(0)" ::: "memory"); }

/* Workgroup barrier for LDS traffic only (no drain of outstanding global loads). */
__device__ __forceinline__ void isel3_sync() {
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__device__ __forceinline__ unsigned isel3_mbcnt(unsigned long long m) {
    return __builtin_amdgcn_mbcnt_hi((unsigned)(m >> 32), __builtin_amdgcn_mbcnt_lo((unsigned)m, 0u));
}

/* Wave exclusive prefix sum and total of v < 2^nbits from bit-plane ballots (no LDS). */
__device__ __forceinline__ unsigned isel3_wave_excl(unsigned v, unsigned nbits, unsigned& total) {
    unsigned ex = 0u, tot = 0u;
    for (unsigned b = 0; b < nbits; b++) {
        const unsigned long long m = __ballot((v >> b) & 1u);
        ex += isel3_mbcnt(m) << b;
        tot += (unsigned)__builtin_popcountll(m) << b;
    }
    total = tot;
    return ex;
}

/* Scan of a 256*G-bin histogram (bins counted from the top, G uint4 per lane): the bin d
 * holding the k_rem-th largest, the population strictly above it, and its own. */
template <unsigned G>
__device__ __forceinline__ void isel3_scan(const unsigned* hw, unsigned planes, unsigned k_rem,
                                           unsigned& d, unsigned& above, unsigned& bnd) {
    const uint4* const h = (const uint4*)hw;
    const unsigned lane = threadIdx.x & 63u;
    const unsigned g0 = 64u * G - G * (lane + 1u);
    unsigned g[G], local = 0u;
#pragma unroll
    for (unsigned j = 0; j < G; j++) {
        const uint4 q = h[g0 + j];
        g[j] = q.x + q.y + q.z + q.w;
        local += g[j];
    }
    unsigned total;
    const unsigned excl = isel3_wave_excl(local, planes, total);
    const unsigned L = (unsigned)__builtin_ctzll(__ballot(excl < k_rem && excl + local >= k_rem));
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

__device__ __forceinline__ unsigned isel3_key(unsigned b) {
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

/* Streams one row through f(k[4], p0, nv): four consecutive keys at positions p0.. of which
 * the first nv are in the row, 2048 keys per step per wave, the next step's loads issued before
 * this one is consumed. Buffer loads past the row read 0. Every lane calls f the same number of
 * times, so f may ballot. */
template <class F>
__device__ __forceinline__ void isel3_stream(const float* row, unsigned row_len, F&& f) {
    using namespace isel3;
    const unsigned lane = threadIdx.x & 63u;
    const unsigned long long a = (unsigned long long)(size_t)row;
    const unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)a);
    const unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(a >> 32));
    const __amdgpu_buffer_rsrc_t r = __builtin_amdgcn_make_buffer_rsrc(
        (void*)(size_t)(((unsigned long long)hi << 32) | lo), (short)0, row_len * 4u, PLOW_BUF_RSRC3);
    constexpr unsigned STEP = 64u * 4u * BATCH;
    const unsigned ns = (row_len + STEP - 1u) / STEP;
    uint4 cur[BATCH];
#pragma unroll
    for (unsigned j = 0; j < BATCH; j++)
        cur[j] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(
                                               r, lane * 16u, j * 1024u, /*glc|slc*/ 3));
    for (unsigned b = 0; b < ns; b++) {
        uint4 nx[BATCH];
#pragma unroll
        for (unsigned j = 0; j < BATCH; j++)
            nx[j] = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(
                                                  r, lane * 16u, ((b + 1u) * BATCH + j) * 1024u, 3));
#pragma unroll
        for (unsigned j = 0; j < BATCH; j++) {
            const unsigned p0 = (b * BATCH + j) * 256u + lane * 4u;
            const unsigned k[4] = {isel3_key(cur[j].x), isel3_key(cur[j].y), isel3_key(cur[j].z),
                                   isel3_key(cur[j].w)};
            const unsigned nv = p0 < row_len ? (row_len - p0 < 4u ? row_len - p0 : 4u) : 0u;
            f(k, p0, nv);
        }
#pragma unroll
        for (unsigned j = 0; j < BATCH; j++) cur[j] = nx[j];
    }
}

/* Taken positions go to one idx row (wave-compacted slots; n is wave-uniform). group() takes
 * a 4-bit mask of positions p0..p0+3. */
struct Isel3EmitRow {
    PLOW_GLOB int* row;
    unsigned top_k;
    unsigned n;
    __device__ __forceinline__ void operator()(bool take, unsigned s) {
        const unsigned long long m = __ballot(take);
        if (take) {
            const unsigned o = n + isel3_mbcnt(m);
            if (o < top_k) st_act<int>(&row[o], (int)s);
        }
        n += (unsigned)__builtin_popcountll(m);
    }
    __device__ __forceinline__ void group(unsigned bits, unsigned p0) {
        unsigned tot;
        unsigned o = n + isel3_wave_excl((unsigned)__builtin_popcount(bits), 3u, tot);
        n += tot;
        while (bits) {
            const unsigned c = (unsigned)__builtin_ctz(bits);
            bits &= bits - 1u;
            if (o < top_k) st_act<int>(&row[o], (int)(p0 + c));
            o++;
        }
    }
};

/* Taken positions set query q's bit in the pack mask (one byte per position; p0 % 4 == 0, so a
 * group is one mask word). */
struct Isel3EmitMask {
    unsigned* mask;
    unsigned bit;
    __device__ __forceinline__ void operator()(bool take, unsigned s) {
        if (take) atomicOr(&mask[s >> 2], bit << ((s & 3u) * 8u));
    }
    __device__ __forceinline__ void group(unsigned bits, unsigned p0) {
        if (bits) {
            const unsigned w = (bits & 1u) | ((bits & 2u) << 7) | ((bits & 4u) << 14) |
                               ((bits & 8u) << 21);
            atomicOr(&mask[p0 >> 2], w * bit);
        }
    }
};

/* Resolved key prefix: score bits from the top (sres of 32), then index bits (ires of 24; index
 * key = row_len - 1 - pos, larger is better). */
struct Isel3Prefix {
    unsigned sprefix, smask, iprefix, imask, sres, ires;
    /* next digit of at most nbmax bits: shift, width, and whether it is an index digit */
    __device__ __forceinline__ bool next(unsigned nbmax, unsigned& sh, unsigned& nb) const {
        if (sres < 32u) {
            nb = 32u - sres < nbmax ? 32u - sres : nbmax;
            sh = 32u - sres - nb;
            return false;
        }
        nb = 24u - ires < nbmax ? 24u - ires : nbmax;
        sh = 24u - ires - nb;
        return true;
    }
    __device__ __forceinline__ void add(bool isx, unsigned sh, unsigned nb, unsigned d) {
        if (isx) {
            iprefix |= d << sh;
            imask |= ((1u << nb) - 1u) << sh;
            ires += nb;
        } else {
            sprefix |= d << sh;
            smask |= ((1u << nb) - 1u) << sh;
            sres += nb;
        }
    }
    __device__ __forceinline__ bool eq(unsigned k, unsigned ik) const {
        return (k & smask) == sprefix && (ik & imask) == iprefix;
    }
    __device__ __forceinline__ bool gt(unsigned k, unsigned ik) const {
        const unsigned ks = k & smask;
        return ks > sprefix || (ks == sprefix && (ik & imask) > iprefix);
    }
    __device__ __forceinline__ unsigned digit(bool isx, unsigned sh, unsigned nb, unsigned k,
                                              unsigned ik) const {
        return ((isx ? ik : k) >> sh) & ((1u << nb) - 1u);
    }
};

/* One wave ranks one row (top_k < row_len). wl: this wave's REGW-word LDS region. */
template <class Emit>
__device__ __forceinline__ void isel3_row(const float* row, unsigned row_len, unsigned top_k,
                                          unsigned* wl, Emit& emit) {
    using namespace isel3;
    const unsigned lane = threadIdx.x & 63u;
    Isel3Prefix px{0u, 0u, 0u, 0u, 0u, 0u};
    unsigned k_rem = top_k, bnd = row_len;
    /* Streaming digits until the boundary group fits the candidate buffer or is needed whole.
     * Histogram bins are laid out from the top of the NBH-bin buffer, digit value d at bin d. */
    while (bnd != k_rem && bnd > CAPW && (px.sres < 32u || px.ires < 24u)) {
        unsigned sh, nb;
        const bool isx = px.next(B1, sh, nb);
#pragma unroll
        for (unsigned j = 0; j < NBH / 256u; j++) ((uint4*)wl)[j * 64u + lane] = uint4{0u, 0u, 0u, 0u};
        isel3_wsync();
        if (px.sres == 0u) {
            isel3_stream(row, row_len, [&](const unsigned (&k)[4], unsigned, unsigned nv) {
#pragma unroll
                for (unsigned c = 0; c < 4u; c++)
                    if (c < nv) atomicAdd(&wl[k[c] >> (32u - B1)], 1u);
            });
        } else {
            isel3_stream(row, row_len, [&](const unsigned (&k)[4], unsigned p0, unsigned nv) {
#pragma unroll
                for (unsigned c = 0; c < 4u; c++) {
                    const unsigned ik = (row_len - 1u - p0 - c) & 0xFFFFFFu;
                    if (c < nv && px.eq(k[c], ik))
                        atomicAdd(&wl[px.digit(isx, sh, nb, k[c], ik)], 1u);
                }
            });
        }
        isel3_wsync();
        unsigned d, above;
        isel3_scan<NBH / 256u>(wl, 25u, k_rem, d, above, bnd);
        isel3_wsync();
        px.add(isx, sh, nb, d);
        k_rem -= above;
    }
#if defined(ISEL3_PROBE) && ISEL3_PROBE == 1
    if (bnd == 12345u) emit(true, 0u);
    return;
#endif
    const bool whole = bnd == k_rem;
    unsigned* const ckey = wl + CKEY;
    unsigned* const cpos = wl + CPOS;
    unsigned nc = 0u;
    /* Above-prefix keys are emitted, prefix-equal ones compacted (or emitted when needed whole).
     * Until an index digit is resolved the comparison is on score bits alone. */
    auto pass2 = [&](auto idx_digits) {
        constexpr bool IX = decltype(idx_digits)::value;
        isel3_stream(row, row_len, [&](const unsigned (&k)[4], unsigned p0, unsigned nv) {
            unsigned gb = 0u, eb = 0u;
#pragma unroll
            for (unsigned c = 0; c < 4u; c++) {
                const unsigned ks = k[c] & px.smask;
                bool g = ks > px.sprefix, e = ks == px.sprefix;
                if (IX) {
                    const unsigned ix = (row_len - 1u - p0 - c) & 0xFFFFFFu & px.imask;
                    g = g || (e && ix > px.iprefix);
                    e = e && ix == px.iprefix;
                }
                gb |= (unsigned)(c < nv && g) << c;
                eb |= (unsigned)(c < nv && e) << c;
            }
            if (whole) {
                emit.group(gb | eb, p0);
            } else {
                emit.group(gb, p0);
                if (__ballot(eb != 0u)) {
                    unsigned tot;
                    unsigned o = nc + isel3_wave_excl((unsigned)__builtin_popcount(eb), 3u, tot);
                    nc += tot;
#pragma unroll
                    for (unsigned c = 0; c < 4u; c++) {
                        if ((eb >> c) & 1u) {
                            ckey[o] = k[c];
                            cpos[o] = p0 + c;
                            o++;
                        }
                    }
                }
            }
        });
    };
    if (px.ires == 0u)
        pass2(std::false_type{});
    else
        pass2(std::true_type{});
    if (whole) return;
#if defined(ISEL3_PROBE) && ISEL3_PROBE == 2
    return;
#endif
    isel3_wsync();
    /* nc == bnd <= CAPW candidates share the prefix; k_rem < nc of them are needed. */
    unsigned* const sh8 = wl + SHIST;
    bool ranked = false;
    while (bnd != k_rem) {
        if (bnd <= 16u) {
            ranked = true;
            break;
        }
        unsigned sh, nb;
        const bool isx = px.next(8u, sh, nb);
        ((uint4*)sh8)[lane] = uint4{0u, 0u, 0u, 0u};
        isel3_wsync();
        for (unsigned i = lane; i < nc; i += 64u) {
            const unsigned k = ckey[i], ik = (row_len - 1u - cpos[i]) & 0xFFFFFFu;
            if (px.eq(k, ik)) atomicAdd(&sh8[px.digit(isx, sh, nb, k, ik)], 1u);
        }
        isel3_wsync();
        unsigned d, above;
        isel3_scan<1u>(sh8, 12u, k_rem, d, above, bnd);
        isel3_wsync();
        px.add(isx, sh, nb, d);
        k_rem -= above;
    }
    if (ranked) {
        /* the bnd (<= 16) prefix-equal candidates: keep the k_rem best by (score, lowest index),
         * flagged with bit 31 of their position */
        unsigned got = 0u;
        for (unsigned b = 0; b < nc; b += 64u) {
            const unsigned i = b + lane;
            const bool m = i < nc && px.eq(ckey[i], (row_len - 1u - cpos[i]) & 0xFFFFFFu);
            const unsigned long long bm = __ballot(m);
            if (m) sh8[got + isel3_mbcnt(bm)] = i;
            got += (unsigned)__builtin_popcountll(bm);
        }
        isel3_wsync();
        const unsigned i = lane < bnd ? sh8[lane] : 0u;
        const unsigned k = lane < bnd ? ckey[i] : 0u;
        const unsigned ik = lane < bnd ? (row_len - 1u - cpos[i]) & 0xFFFFFFu : 0u;
        unsigned rank = 0u;
        for (unsigned j = 0; j < bnd; j++) {
            const unsigned kj = (unsigned)__builtin_amdgcn_readlane((int)k, (int)j);
            const unsigned ij = (unsigned)__builtin_amdgcn_readlane((int)ik, (int)j);
            rank += kj > k || (kj == k && ij > ik);
        }
        if (lane < bnd && rank < k_rem) cpos[i] |= 0x80000000u;
        isel3_wsync();
    }
    for (unsigned b = 0; b < nc; b += 64u) {
        const unsigned i = b + lane;
        bool t = false;
        unsigned s = 0u;
        if (i < nc) {
            const unsigned p = cpos[i];
            s = p & 0x7FFFFFFFu;
            const unsigned k = ckey[i], ik = (row_len - 1u - s) & 0xFFFFFFu;
            t = (p >> 31) || px.gt(k, ik) || (!ranked && px.eq(k, ik));
        }
        emit(t, s);
    }
}

/* Row t into idx: identity (-1 padded) or ranked. One wave. */
__device__ __forceinline__ void isel3_select_row(PLOW_GLOB int* ib, const float* Score,
                                                 unsigned top_k, unsigned kv_stride, unsigned t,
                                                 unsigned row_len, unsigned* wl) {
    PLOW_GLOB int* const row = ib + (size_t)t * top_k;
    if (row_len <= top_k) {
        for (unsigned s = threadIdx.x & 63u; s < top_k; s += 64u)
            st_act<int>(&row[s], s < row_len ? (int)s : -1);
    } else {
        Isel3EmitRow e{row, top_k, 0u};
        isel3_row(Score + (size_t)t * kv_stride, row_len, top_k, wl, e);
    }
}

/* op 118 under PLOW_DSA_SELECT_V3: [T][top_k] idx rows, one wave per row, no barriers. */
__device__ void d_index_select_pf_v3(int* __restrict__ idx, const float* __restrict__ Score,
                                     const int* __restrict__ kv_len, unsigned n_tok,
                                     unsigned top_k, unsigned kv_stride, unsigned slice,
                                     unsigned nblk, unsigned* lds) {
    using namespace isel3;
    PLOW_GLOB int* const ib = as_glob(idx);
    const unsigned q_pos0 = (unsigned)as_glob(kv_len)[0] - n_tok;
    const unsigned wave = threadIdx.x >> 6;
    unsigned* const wl = lds + wave * REGW;
    for (unsigned t = slice * NW + wave; t < n_tok; t += nblk * NW)
        isel3_select_row(ib, Score, top_k, kv_stride, t, q_pos0 + t + 1u, wl);
}

/* Appends the mask's positions [w0, w0 + 4*nw) (ascending) to the pack's union block. One
 * barrier per 512-word chunk: wave totals go to a parity slot of red[0..15] that is rewritten
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
        if ((tid & 63u) == 0u) red[par + wave] = wt;
        isel3_sync();
        unsigned o = base + wx, total = 0u;
#pragma unroll
        for (unsigned v = 0; v < PLOW_WAVES; v++) {
            const unsigned c = red[par + v];
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

/* op 118 fused with op 119 (pack = 8): one workgroup per 8-query pack, one wave per row, rows
 * selected straight into an LDS position mask; then the pack's union block is written exactly as
 * d_index_union_pf writes it (count header, ascending positions, maskLo = the pack's query bits,
 * maskHi = 0, count clamped to cap), and the sparse flash's pack ticket counter is zeroed when
 * zero_ctr is set. idx is written only for packs whose causal bound exceeds the mask: they
 * select to idx and build the union from it in MAXPOS-position windows. */
__device__ void d_index_select_union_pf(unsigned char* __restrict__ uni, int* __restrict__ idx,
                                        const float* __restrict__ Score,
                                        const int* __restrict__ kv_len, unsigned n_tok,
                                        unsigned top_k, unsigned kv_stride, unsigned cap,
                                        unsigned zero_ctr, unsigned slice, unsigned nblk,
                                        unsigned* lds) {
    using namespace isel3;
    const unsigned tid = threadIdx.x, wave = tid >> 6, lane = tid & 63u;
    PLOW_GLOB int* const ib = as_glob(idx);
    const unsigned q_pos0 = (unsigned)as_glob(kv_len)[0] - n_tok;
    const unsigned n_qt = (n_tok + PACK - 1u) / PACK;
    const unsigned hdr = (n_qt * 4u + 255u) / 256u * 256u;
    PLOW_GLOB unsigned* const cnt = (PLOW_GLOB unsigned*)as_glob(uni);
    unsigned* const mask = lds + MASK;
    unsigned* const wl = lds + wave * REGW;
    unsigned* const red = lds; /* compaction scratch: every row region is idle by then */
    if (zero_ctr && slice == 0u && tid < 2u)
        st_act<unsigned>((PLOW_GLOB unsigned*)(as_glob(uni) + hdr + (size_t)n_qt * cap * 12u) + tid, 0u);
    for (unsigned qt = slice; qt < n_qt; qt += nblk) {
        const unsigned q_lo = qt * PACK;
        const unsigned q_hi = q_lo + PACK - 1u < n_tok - 1u ? q_lo + PACK - 1u : n_tok - 1u;
        const unsigned tile_end = q_pos0 + q_hi + 1u;
        PLOW_GLOB unsigned char* const blk = as_glob(uni) + hdr + (size_t)qt * cap * 12u;
        PLOW_GLOB int* const upos = (PLOW_GLOB int*)blk;
        PLOW_GLOB unsigned* const ulo = (PLOW_GLOB unsigned*)(blk + (size_t)cap * 4u);
        PLOW_GLOB unsigned* const uhi = (PLOW_GLOB unsigned*)(blk + (size_t)cap * 8u);
        const unsigned t = q_lo + wave;
        const unsigned row_len = q_pos0 + t + 1u;
        unsigned base = 0u;
        if (tile_end <= MAXPOS) {
            for (unsigned w = tid; w < (tile_end + 3u) / 4u; w += PLOW_THREADS) mask[w] = 0u;
            isel3_sync();
            if (t <= q_hi) {
                const unsigned bit = 1u << wave;
                if (row_len <= top_k) {
                    for (unsigned s = lane; s < row_len; s += 64u)
                        atomicOr(&mask[s >> 2], bit << ((s & 3u) * 8u));
                } else {
                    Isel3EmitMask e{mask, bit};
                    isel3_row(Score + (size_t)t * kv_stride, row_len, top_k, wl, e);
                }
            }
            isel3_sync();
            base = isel3_compact(mask, (tile_end + 3u) / 4u, 0u, 0u, cap, upos, ulo, uhi, red);
        } else {
            if (t <= q_hi) isel3_select_row(ib, Score, top_k, kv_stride, t, row_len, wl);
            /* idx rows were stored by this workgroup; L1-bypassing loads read them back. */
            __threadfence_block();
            __syncthreads();
            for (unsigned w0 = 0; w0 < tile_end; w0 += MAXPOS) {
                const unsigned wend = w0 + MAXPOS < tile_end ? w0 + MAXPOS : tile_end;
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
