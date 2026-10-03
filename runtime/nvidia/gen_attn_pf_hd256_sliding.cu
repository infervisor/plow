// Generated-kernel catalog entry attn_pf_hd256_sliding: causal sliding-window flash prefill at
// head_dim 256, bf16, sm_90a. A parameterized template: scripts/gen_kernels/attn_pf_hd256_sliding.py
// picks GEN_* per its tuning sweep, builds the cubin and checks it. One object serves any model
// with this op signature: heads, KV heads (even GQA ratio), window (0 = causal only), KV ring
// mask and scale are runtime operands.
//
// Packet contract (role 14 PREFILL_ATTENTION_HD256_GQA2_BKV32's): Q/O [seq_q][heads][256], K/V
// head-major [slot][kv head][kv_stride][256] with position p at row `p & kv_mask` (a ring on
// sliding layers), the packed request table {count, {q0, qlen, slot, kvlen}...}, rows past the
// last request zeroed, the segment successor counters signalled on exit. `mapkv`, when present,
// is the packet's GEN_TMAP_KV_PAIR (a per-slot table of them when packed).
//
// Schedule: persistent, one CTA per SM, three warpgroups. WG0 stages Q and the K/V tiles into
// 128B-swizzled shared memory: TMA through `mapkv` for tiles inside [0, kvlen), cp.async with
// zero fill otherwise; K and V rings of GEN_KSTAGES / GEN_VSTAGES with mbarrier handshakes.
// WG1/WG2 each own one query head of a GQA pair over the same 64 query rows, so one K/V tile
// feeds both: S = Q.K^T (wgmma SS), in-register online softmax, O += P.V (wgmma RS, P from the S
// accumulator), with QK(t) and PV(t-1) in flight together. Only KV tiles inside some row's
// causal window are staged or computed.
#include "dev_isa.h"
#define PLOW_NV_HOPPER 1
#include "sm90_wgmma.cuh"

#ifndef GEN_BN
#define GEN_BN 64
#endif
#ifndef GEN_KSTAGES
#define GEN_KSTAGES 2
#endif
#ifndef GEN_VSTAGES
#define GEN_VSTAGES 2
#endif
// setmaxnreg split of the 384 x 168 launch pool: producer warpgroup / consumer warpgroups.
#ifndef GEN_PREG
#define GEN_PREG 56
#endif
#ifndef GEN_CREG
#define GEN_CREG 224
#endif

namespace hd256 {
constexpr int HD = 256, BM = 64, BN = GEN_BN, KS = GEN_KSTAGES, VS = GEN_VSTAGES, THREADS = 384;
static_assert(KS >= 2 && VS >= 2, "pipeline shape");
static_assert(BN == 32 || BN == 64, "score tile width (a multiple of the 32-row map box)");
static_assert(GEN_PREG * 128 + GEN_CREG * 256 <= 168 * 384, "setmaxnreg pool");
constexpr unsigned Q_BYTES = BM * HD * 2;
constexpr unsigned KV_BYTES = BN * HD * 2;
constexpr unsigned OFF_K = 2 * Q_BYTES;
constexpr unsigned OFF_V = OFF_K + KS * KV_BYTES;
constexpr unsigned OFF_BAR = OFF_V + VS * KV_BYTES;
// fullQ[2] emptyQ[2] fullK[KS] emptyK[KS] fullV[VS] emptyV[VS]
constexpr unsigned NBAR = 4 + 2 * KS + 2 * VS;
constexpr unsigned ARENA = OFF_BAR + NBAR * 8 + 1024;
static_assert(ARENA >= 116 * 1024 && ARENA <= 227 * 1024, "one CTA per SM");
}  // namespace hd256

extern "C" __device__ __constant__ unsigned plow_pf_request_abi = 2;
extern "C" __device__ __constant__ unsigned plow_pf_masked_padding_abi = 1;
extern "C" __device__ unsigned plow_gen_flash_prefill_abi = 1;
extern "C" __device__ unsigned plow_gen_block = hd256::THREADS;
extern "C" __device__ unsigned plow_gen_arena_bytes = hd256::ARENA;
extern "C" __device__ unsigned plow_attention_head_dim = hd256::HD;
extern "C" __device__ unsigned plow_attention_query_tile = hd256::BM;
extern "C" __device__ unsigned plow_attention_kv_tile = hd256::BN;
extern "C" __device__ unsigned plow_attention_warps = hd256::THREADS / 32;

typedef struct {
    const int* requests;
    float* opart;
    float* mlpart;
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    __nv_bfloat16* output;
    const void* mapkv;
    const PlowStreamEnt* entries;
    const unsigned* succs;
    unsigned* counters;
    unsigned seq_q;
    unsigned seq_kv;
    unsigned q_pos0;
    unsigned kv_stride;
    unsigned kv_mask;
    float scale;
    unsigned n_head;
    unsigned n_kv_head;
    unsigned window;
    unsigned reserved;
} PlowGenFlashPrefill;
static_assert(sizeof(PlowGenFlashPrefill) == 128, "generated flash prefill role ABI");

#include "gen_attn_pf_hd256_wgmma.inc"

namespace hd256 {

__device__ __forceinline__ void bar_init(uint32_t b, unsigned n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_arrive(uint32_t b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}
__device__ __forceinline__ void bar_wait(uint32_t b, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nW%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra W%=;\n}\n" ::"r"(b), "r"(parity)
                 : "memory");
}
// A full barrier completes when every loader thread's copies have landed; loaders never block
// on their own copies.
__device__ __forceinline__ void bar_arrive_copies(uint32_t b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(b) : "memory");
}
// cp.async lands through the generic proxy, wgmma reads through the async proxy.
__device__ __forceinline__ void bar_wait_copies(uint32_t b, unsigned parity) {
    bar_wait(b, parity);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void bar_expect_tx(uint32_t b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes)
                 : "memory");
}
// TMA through a GEN_TMAP_KV_PAIR map: rank 3 {256, kv_stride, n_kv_head}, box {64, 32, 1},
// 128B swizzle (role 14's staging contract).
__device__ __forceinline__ void tma3d(uint32_t dst, const void* map, int c0, int c1, int c2,
                                      uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
                 "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
                 : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, unsigned bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ float ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Request r of the packed table; no table is one request covering the launch.
__device__ __forceinline__ unsigned req_count(const PlowGenFlashPrefill& a) {
    return a.requests ? (unsigned)a.requests[0] : 1u;
}
__device__ __forceinline__ uint4 req_at(const PlowGenFlashPrefill& a, unsigned r) {
    if (!a.requests) return make_uint4(0u, a.seq_q, 0u, a.seq_kv);
    const int* e = a.requests + 1 + 4 * r;
    return make_uint4((unsigned)e[0], (unsigned)e[1], (unsigned)e[2], (unsigned)e[3]);
}

// One work item: 64 query rows of one request against one KV head, query heads h0 and h0 + 1.
struct Item {
    unsigned qrow;    // Q/O buffer row of the tile's first query row
    unsigned nrows;   // valid query rows in the tile
    unsigned pos;     // absolute position of the tile's first query row
    unsigned h0, hk, slot, kvlen;
    unsigned lo;      // first KV position any row of the tile attends
    unsigned t0, t1;  // KV tiles [t0, t1)
};

// Items run heaviest-first inside each request (descending query tile, heads inner) and are
// dealt to the persistent CTAs in zig-zag rounds.
__device__ __forceinline__ bool item_at(const PlowGenFlashPrefill& a, unsigned round, Item& it) {
    unsigned local = round * gridDim.x + ((round & 1) ? gridDim.x - 1 - blockIdx.x : blockIdx.x);
    const unsigned pairs = a.n_head / a.n_kv_head / 2;
    const unsigned per_tile = a.n_kv_head * pairs;
    const unsigned count = req_count(a);
    for (unsigned r = 0; r < count; ++r) {
        const uint4 rq = req_at(a, r);
        const unsigned tiles = (rq.y + BM - 1) / BM;
        if (local >= tiles * per_tile) {
            local -= tiles * per_tile;
            continue;
        }
        const unsigned tile = tiles - 1 - local / per_tile;
        const unsigned rem = local % per_tile;
        it.hk = rem / pairs;
        it.h0 = it.hk * (a.n_head / a.n_kv_head) + 2 * (rem % pairs);
        it.slot = rq.z;
        it.kvlen = rq.w;
        it.qrow = rq.x + tile * BM;
        it.nrows = min((unsigned)BM, rq.y - tile * BM);
        it.pos = rq.w - rq.y + tile * BM;
        it.lo = (a.window && it.pos + 1 > a.window) ? it.pos + 1 - a.window : 0u;
        it.t0 = it.lo / BN;
        it.t1 = (it.pos + it.nrows - 1) / BN + 1;
        return true;
    }
    return false;
}

// wgmma writes its accumulators asynchronously: pin them after each wait so no use is hoisted
// above it, and pin operands before an issue so none is defined inside another GEMM's window.
template <int N>
__device__ __forceinline__ void fence_regs(float* r) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N>
__device__ __forceinline__ void fence_regs(uint32_t* r) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}
template <int N>
__device__ __forceinline__ void fence_regs(uint64_t* r) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+l"(r[i])::"memory");
}

// wgmma descriptor halves: lo32 = start address >> 4 | LBO >> 4 << 16, hi32 = SBO 1024 and the
// 128B swizzle mode.
constexpr uint32_t DESC_HI = (1024u >> 4) | (1u << 30);
__device__ __forceinline__ uint32_t desc_lo(uint32_t addr, uint32_t lbo) {
    return ((addr & 0x3FFFF) >> 4) | ((lbo >> 4) << 16);
}

// S[64][BN] = Q[64][256] . K[BN][256]^T (wgmma SS), fence to commit.
__device__ __forceinline__ void issue_qk(float* sc, uint32_t qs, uint32_t ks) {
    const uint32_t qa = desc_lo(qs, 16), kb = desc_lo(ks, 16);
    if constexpr (BN == 64) hd256_qk_n64(sc, qa, kb, DESC_HI);
    else hd256_qk_n32(sc, qa, kb, DESC_HI);
}

// O[64][256] += P[64][BN] . V[BN][256] (wgmma RS): P from registers, since the S accumulator
// layout is the k16 A-fragment layout; V MN-major in place. PV is issued while QK is in flight,
// and a wgmma input defined inside that window serializes the pipeline (ptxas C7513), so the
// caller builds and pins the descriptors and operands before issuing QK.
struct PvDesc {
    uint64_t d[BN / 16];
    __device__ __forceinline__ explicit PvDesc(uint32_t vs) {
        const uint32_t lo = desc_lo(vs, BN * 128);
#pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk) d[kk] = ((uint64_t)DESC_HI << 32) | (lo + kk * 128);
    }
};
__device__ __forceinline__ void issue_pv(float* o, const uint32_t* pa, const PvDesc& vd) {
    if constexpr (BN == 64) hd256_pv_bn64(o, pa, vd.d);
    else hd256_pv_bn32(o, pa, vd.d);
}

// Online softmax over one score tile for this lane's two rows (row, row + 8), in place: S becomes
// the unnormalized fp32 P. Scores stay in the log2 domain. Masks only tiles that cross the causal
// diagonal or the window edge; a row with nothing visible yet keeps max -inf and adds zeros.
struct Softmax {
    unsigned row, col, window, pos;
    float sl, m0, m1, l0, l1;
    __device__ __forceinline__ void reset(unsigned p) {
        pos = p;
        m0 = m1 = -INFINITY;
        l0 = l1 = 0.f;
    }
    __device__ __forceinline__ void step(float* sc, unsigned kv0, float& c0, float& c1) {
        const bool diag = kv0 + BN - 1 > pos;
        const bool edge = window && pos + BM - 1 - kv0 >= window;
        if (diag || edge) {
#pragma unroll
            for (int j = 0; j < BN / 8; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const unsigned q = pos + row + 8 * (e >> 1);
                    const unsigned kv = kv0 + 8 * j + col + (e & 1);
                    if (kv > q || (window && q - kv >= window)) sc[4 * j + e] = -INFINITY;
                }
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int j = 0; j < BN / 8; ++j) {
            mx0 = fmaxf(mx0, fmaxf(sc[4 * j], sc[4 * j + 1]));
            mx1 = fmaxf(mx1, fmaxf(sc[4 * j + 2], sc[4 * j + 3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        const float u0 = mx0 == -INFINITY ? 0.f : mx0 * sl;
        const float u1 = mx1 == -INFINITY ? 0.f : mx1 * sl;
        c0 = ex2(m0 * sl - u0);
        c1 = ex2(m1 * sl - u1);
        m0 = mx0;
        m1 = mx1;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int j = 0; j < BN / 8; ++j) {
            sc[4 * j] = ex2(sc[4 * j] * sl - u0);
            sc[4 * j + 1] = ex2(sc[4 * j + 1] * sl - u0);
            sc[4 * j + 2] = ex2(sc[4 * j + 2] * sl - u1);
            sc[4 * j + 3] = ex2(sc[4 * j + 3] * sl - u1);
            s0 += sc[4 * j] + sc[4 * j + 1];
            s1 += sc[4 * j + 2] + sc[4 * j + 3];
        }
        l0 = l0 * c0 + s0;
        l1 = l1 * c1 + s1;
    }
};

// P as the PV A fragments: k16 step kk is n8 blocks 2kk, 2kk+1 of the score accumulator. Written
// only while no wgmma is in flight.
__device__ __forceinline__ void pack_p(const float* sc, uint32_t* pa) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
        pa[2 * j] = pack_bf16(sc[4 * j], sc[4 * j + 1]);
        pa[2 * j + 1] = pack_bf16(sc[4 * j + 2], sc[4 * j + 3]);
    }
}

struct Bars {
    uint32_t base;
    __device__ __forceinline__ uint32_t fullQ(unsigned w) const { return base + 8 * w; }
    __device__ __forceinline__ uint32_t emptyQ(unsigned w) const { return base + 8 * (2 + w); }
    __device__ __forceinline__ uint32_t fullK(unsigned s) const { return base + 8 * (4 + s); }
    __device__ __forceinline__ uint32_t emptyK(unsigned s) const { return base + 8 * (4 + KS + s); }
    __device__ __forceinline__ uint32_t fullV(unsigned s) const {
        return base + 8 * (4 + 2 * KS + s);
    }
    __device__ __forceinline__ uint32_t emptyV(unsigned s) const {
        return base + 8 * (4 + 2 * KS + VS + s);
    }
};

// TMA: whole tiles inside [0, kvlen) go through the copy engine (thread 0 issues, the other
// loaders just arrive); a partial tile keeps cp.async so its rows past kvlen are zero (0 * a stale
// NaN is NaN in the PV).
template <bool TMA>
__device__ __forceinline__ void produce(const PlowGenFlashPrefill& a, uint32_t base, Bars b,
                                        unsigned tid) {
    const size_t kv_head_elems = (size_t)a.kv_stride * HD;
    // Each producer thread owns one 16 B column chunk `c` of rows r0, r0 + 4, ...
    const unsigned c = tid % 32, r0 = tid / 32;
    unsigned kvit = 0, qph = 0;
    Item it;
    for (unsigned round = 0; item_at(a, round, it); ++round) {
        const size_t head = ((size_t)it.slot * a.n_kv_head + it.hk) * kv_head_elems + c * 8;
        // Packed launches carry a per-slot table of map pairs; an unpacked one the pair itself.
        const char* map = static_cast<const char*>(
            a.requests ? reinterpret_cast<const void*>(
                             reinterpret_cast<const uint64_t*>(a.mapkv)[it.slot])
                       : a.mapkv);
        for (unsigned w = 0; w < 2; ++w) bar_wait(b.emptyQ(w), qph ^ 1);
        for (unsigned w = 0; w < 2; ++w) {
            const __nv_bfloat16* src = a.q + ((size_t)it.qrow * a.n_head + it.h0 + w) * HD + c * 8;
            const uint32_t dst = base + w * Q_BYTES + (c >> 3) * (BM * 128) + ((c & 7) << 4);
#pragma unroll 4
            for (unsigned r = r0; r < BM; r += 4) {
                const bool ok = r < it.nrows;
                cp16((dst + r * 128) ^ ((r & 7) << 4), ok ? src + (size_t)r * a.n_head * HD : a.q,
                     ok ? 16 : 0);
            }
            bar_arrive_copies(b.fullQ(w));
        }
        qph ^= 1;
        for (unsigned t = it.t0; t < it.t1; ++t, ++kvit) {
            for (unsigned which = 0; which < 2; ++which) {
                const unsigned ns = which ? VS : KS;
                const unsigned s = kvit % ns, ph = (kvit / ns) & 1;
                const uint32_t full = which ? b.fullV(s) : b.fullK(s);
                const uint32_t tile = base + (which ? OFF_V : OFF_K) + s * KV_BYTES;
                bar_wait(which ? b.emptyV(s) : b.emptyK(s), ph ^ 1);
                if (TMA && (t + 1) * BN <= it.kvlen) {
                    if (tid == 0) {
                        bar_expect_tx(full, KV_BYTES);
                        const char* m = map + (which ? 128 : 0);
                        const int row = (int)((t * BN) & a.kv_mask);
#pragma unroll
                        for (int blk = 0; blk < HD / 64; ++blk)
#pragma unroll
                            for (int hb = 0; hb < BN / 32; ++hb)
                                tma3d(tile + blk * (BN * 128) + hb * 32 * 128, m, blk * 64,
                                      row + hb * 32, (int)it.hk, full);
                    } else {
                        bar_arrive(full);
                    }
                    continue;
                }
                // 128B swizzle: chunk c of row r lands at chunk c ^ (r & 7) of the 128 B row.
                const __nv_bfloat16* g = (which ? a.v : a.k) + head;
                const uint32_t dst = tile + (c >> 3) * (BN * 128) + ((c & 7) << 4);
#pragma unroll 4
                for (unsigned r = r0; r < BN; r += 4) {
                    const unsigned pos = t * BN + r;
                    const bool ok = pos >= it.lo && pos < it.kvlen;
                    cp16((dst + r * 128) ^ ((r & 7) << 4),
                         ok ? g + (size_t)(pos & a.kv_mask) * HD : a.k, ok ? 16 : 0);
                }
                bar_arrive_copies(full);
            }
        }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
}

__device__ __forceinline__ void consume(const PlowGenFlashPrefill& a, uint32_t base, Bars b,
                                        unsigned tid) {
    const unsigned w = tid / 128 - 1;
    const unsigned lt = tid % 128, warp = lt / 32, lane = lt % 32;
    Softmax sm;
    sm.row = 16 * warp + lane / 4;
    sm.col = 2 * (lane & 3);
    sm.sl = a.scale * 1.4426950408889634f;
    sm.window = a.window;
    const uint32_t qs = base + w * Q_BYTES;
    unsigned kvit = 0, qph = 0;
    Item it;
    for (unsigned round = 0; item_at(a, round, it); ++round) {
        bar_wait_copies(b.fullQ(w), qph);
        qph ^= 1;
        float o[HD / 2];
#pragma unroll
        for (int i = 0; i < HD / 2; ++i) o[i] = 0.f;
        sm.reset(it.pos);
        float sc[BN / 2];
        uint32_t pa[BN / 4];
        // First tile: scores and softmax only; O is still zero.
        {
            const unsigned s = kvit % KS, ph = (kvit / KS) & 1;
            bar_wait_copies(b.fullK(s), ph);
            issue_qk(sc, qs, base + OFF_K + s * KV_BYTES);
            sm90_wg_wait<0>();
            fence_regs<BN / 2>(sc);
            bar_arrive(b.emptyK(s));
            if (it.t0 + 1 == it.t1) bar_arrive(b.emptyQ(w));
            float c0, c1;
            sm.step(sc, it.t0 * BN, c0, c1);
            pack_p(sc, pa);
        }
        // Steady state: QK(t) and PV(t-1) in flight together; softmax(t) runs under PV(t-1).
        for (unsigned t = it.t0 + 1; t < it.t1; ++t) {
            const unsigned sp = kvit % VS, pp = (kvit / VS) & 1;
            const unsigned s = (kvit + 1) % KS, ph = ((kvit + 1) / KS) & 1;
            PvDesc vd(base + OFF_V + sp * KV_BYTES);
            fence_regs<BN / 16>(vd.d);
            fence_regs<HD / 2>(o);
            fence_regs<BN / 4>(pa);
            bar_wait_copies(b.fullK(s), ph);
            issue_qk(sc, qs, base + OFF_K + s * KV_BYTES);
            bar_wait_copies(b.fullV(sp), pp);
            issue_pv(o, pa, vd);
            sm90_wg_wait<1>();
            fence_regs<BN / 2>(sc);
            bar_arrive(b.emptyK(s));
            if (t + 1 == it.t1) bar_arrive(b.emptyQ(w));
            float c0, c1;
            sm.step(sc, t * BN, c0, c1);
            sm90_wg_wait<0>();
            fence_regs<HD / 2>(o);
            fence_regs<BN / 4>(pa);
            fence_regs<BN / 2>(sc);
            bar_arrive(b.emptyV(sp));
#pragma unroll
            for (int j = 0; j < HD / 8; ++j) {
                o[4 * j] *= c0;
                o[4 * j + 1] *= c0;
                o[4 * j + 2] *= c1;
                o[4 * j + 3] *= c1;
            }
            pack_p(sc, pa);
            ++kvit;
        }
        {
            const unsigned s = kvit % VS, ph = (kvit / VS) & 1;
            bar_wait_copies(b.fullV(s), ph);
            issue_pv(o, pa, PvDesc(base + OFF_V + s * KV_BYTES));
            sm90_wg_wait<0>();
            fence_regs<HD / 2>(o);
            fence_regs<BN / 4>(pa);
            bar_arrive(b.emptyV(s));
            ++kvit;
        }

        float l0 = sm.l0, l1 = sm.l1;
        l0 += __shfl_xor_sync(0xffffffff, l0, 1);
        l0 += __shfl_xor_sync(0xffffffff, l0, 2);
        l1 += __shfl_xor_sync(0xffffffff, l1, 1);
        l1 += __shfl_xor_sync(0xffffffff, l1, 2);
        const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
        const unsigned h = it.h0 + w;
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            const unsigned r = sm.row + 8 * hi;
            if (r < it.nrows) {
                __nv_bfloat16* dst =
                    a.output + ((size_t)(it.qrow + r) * a.n_head + h) * HD + sm.col;
                const float inv = hi ? i1 : i0;
#pragma unroll
                for (int j = 0; j < HD / 8; ++j)
                    *reinterpret_cast<uint32_t*>(dst + 8 * j) =
                        pack_bf16(o[4 * j + 2 * hi] * inv, o[4 * j + 2 * hi + 1] * inv);
            }
        }
    }
}

__device__ __forceinline__ void zero_padding(const PlowGenFlashPrefill& a, unsigned lt,
                                             unsigned nt) {
    if (!a.requests) return;
    const unsigned count = (unsigned)a.requests[0];
    const unsigned real =
        count ? (unsigned)a.requests[4 * count - 3] + (unsigned)a.requests[4 * count - 2] : 0u;
    const size_t begin = (size_t)real * a.n_head * HD;
    const size_t end = (size_t)a.seq_q * a.n_head * HD;
    for (size_t i = begin + (size_t)blockIdx.x * nt + lt; i < end; i += (size_t)gridDim.x * nt)
        a.output[i] = __float2bfloat16(0.0f);
}

__device__ __forceinline__ void signal(const PlowStreamEnt* p, const unsigned* succs,
                                       unsigned* counters, unsigned lt, unsigned nt) {
    PlowStreamEnt entry;
    for (int i = 0; i < 3; ++i)
        reinterpret_cast<uint2*>(&entry)[i] = reinterpret_cast<const uint2*>(p)[i];
    for (unsigned s = lt; s < entry.succ_len; s += nt)
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(
                         PLOW_CTR(counters, succs[entry.succ_ofs + s]))
                     : "memory");
}

// The producer warpgroup leaves once its copies land; the consumers zero the padded rows and
// publish the successor counters after both stored their outputs (named barrier 1).
__device__ __forceinline__ void run(const PlowGenFlashPrefill& a, uint8_t* smem,
                                    const PlowStreamEnt* entry, const unsigned* succs,
                                    unsigned* counters) {
    const unsigned tid = threadIdx.x;
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
    const Bars b{base + OFF_BAR};
    if (tid == 0) {
        for (unsigned w = 0; w < 2; ++w) {
            bar_init(b.fullQ(w), 128);
            bar_init(b.emptyQ(w), 128);
        }
        for (unsigned s = 0; s < KS; ++s) {
            bar_init(b.fullK(s), 128);
            bar_init(b.emptyK(s), 256);
        }
        for (unsigned s = 0; s < VS; ++s) {
            bar_init(b.fullV(s), 128);
            bar_init(b.emptyV(s), 256);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (tid < 128) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(GEN_PREG));
        if (a.mapkv) produce<true>(a, base, b, tid);
        else produce<false>(a, base, b, tid);
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(GEN_CREG));
    consume(a, base, b, tid);
    asm volatile("bar.sync 1, 256;" ::: "memory");
    zero_padding(a, tid - 128, 256);
    asm volatile("bar.sync 1, 256;" ::: "memory");
    signal(entry, succs, counters, tid - 128, 256);
}

__device__ __forceinline__ bool valid(const PlowGenFlashPrefill& a) {
    return (a.requests || a.q_pos0 + a.seq_q == a.seq_kv) && a.output && a.n_kv_head &&
           a.n_head % (2 * a.n_kv_head) == 0 &&
           (a.kv_mask == 0xffffffffu || ((a.kv_mask + 1) & a.kv_mask) == 0) &&
           (a.kv_mask == 0xffffffffu || a.kv_mask + 1 >= (unsigned)BN);
}

__device__ __forceinline__ uint8_t* arena() {
    extern __shared__ __align__(1024) uint8_t dyn[];
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(dyn);
    return dyn + ((1024u - (s & 1023u)) & 1023u);
}

}  // namespace hd256

extern "C" __global__ __launch_bounds__(hd256::THREADS, 1)
void plow_gen_flash_prefill_direct(const __grid_constant__ PlowGenFlashPrefill args) {
    if (!hd256::valid(args)) {
        __trap();
        return;
    }
    hd256::run(args, hd256::arena(), args.entries + blockIdx.x, args.succs, args.counters);
}

// Packet entry: one FlashPrefill instruction per segment, `blocks` == gridDim.x.
extern "C" __global__ __launch_bounds__(hd256::THREADS, 1)
void plow_gen_flash_prefill(PlowProgram prog) {
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    const unsigned index = lo + blockIdx.x;
    if (index >= hi) __trap();
    PlowStreamEnt entry;
    for (int i = 0; i < 3; ++i)
        reinterpret_cast<uint2*>(&entry)[i] =
            reinterpret_cast<const uint2*>(prog.gq_stream + index)[i];
    if (entry.flags & PLOW_SE_XCTR) __trap();
    for (unsigned w = threadIdx.x; w < entry.wait_len; w += blockDim.x) {
        const PlowWait wait = prog.waits[entry.wait_ofs + w];
        unsigned value;
        do {
            asm volatile("ld.acquire.gpu.u32 %0, [%1];"
                         : "=r"(value)
                         : "l"(PLOW_CTR(prog.counters, wait.id))
                         : "memory");
        } while (value < wait.threshold);
    }
    __syncthreads();
    const PlowDevInst* in = prog.insts + entry.inst;
    void* const* t = prog.tensors;
    __shared__ PlowGenFlashPrefill a;
    if (threadIdx.x == 0) {
        a.requests = in->t[6] == PLOW_TENSOR_NONE ? nullptr : static_cast<const int*>(t[in->t[6]]);
        a.q = static_cast<const __nv_bfloat16*>(t[in->t[2]]);
        a.k = static_cast<const __nv_bfloat16*>(t[in->t[3]]);
        a.v = static_cast<const __nv_bfloat16*>(t[in->t[4]]);
        a.output = in->t[5] == PLOW_TENSOR_NONE ? nullptr
                                                : static_cast<__nv_bfloat16*>(t[in->t[5]]);
        a.mapkv = in->t[7] == PLOW_TENSOR_NONE ? nullptr : t[in->t[7]];
        a.seq_q = in->i[0];
        a.seq_kv = in->i[1];
        a.q_pos0 = in->i[4];
        a.kv_stride = in->fj[1].u;
        a.kv_mask = in->fj[2].u;
        a.scale = in->fj[0].f;
        a.n_head = in->i[2];
        a.n_kv_head = in->i[3];
        a.window = in->i[5];
    }
    __syncthreads();
    if (in->op != PLOW_DOP_FLASH_PREFILL || in->i[6] != hd256::HD || in->i[7] != 1 ||
        !hd256::valid(a))
        __trap();
    hd256::run(a, hd256::arena(), prog.gq_stream + index, prog.succs, prog.counters);
}
