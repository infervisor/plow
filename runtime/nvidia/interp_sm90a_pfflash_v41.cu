/* interp_sm90a_pfflash_v41.cu -- prefill segment role object for DeepSeek-V4.1's sparse attention,
 * PLOW_DOP_FLASH_MLA_PREFILL (op 51) in its NoPE form, on Hopper wgmma. Three arms:
 *   window   (t7 NONE): the interpreter arm's window partial (op_v41_flash.cuh), same layout;
 *   gathered (t7 = union, t5 NONE or = t4): the union partials, same layout;
 *   FUSED    (t7 = union, t5 = the window rows, a tensor other than t4): one pass over the window rows
 *            of t5 then the union rows of t4 -- kernel.py sparse_attn over cat(window, compressed) --
 *            with t3 = attn_sink (f32 [n_head], or NONE) folded in, normalized, and t0 = O bf16
 *            [n_tok][n_head][512] written directly: no partials, no FlashMerge. i7 must be 0.
 *
 * A work item is M = 64 query rows: P queries x HB = 64 / P heads (the window arm takes
 * P = 64 / min(n_head, 64); the gathered and fused arms take the union tile, j0, 0 = 8). Its key rows
 * stream in tiles of 64 rows of the 512-wide K = V latent:
 *   producer warpgroup: cp.async gathers (two threads per row) of Q and the key rows into 128B-swizzled
 *     K-major tiles, the union membership words beside them (two stages, mbarrier full / empty; the Q
 *     slot is released after the item's last QK, so the next item's Q overlaps its tail);
 *   consumer warpgroups 0 / 1: S = Q K^T over keys [32 wg, 32 wg + 32) (m64n32k16), a cross-group
 *     row max, P (bf16) to shared memory, then O[:, 256 wg .. +256] += P V (m64n256k16, V read as the
 *     MN-major view of the same tile).
 * Scores are in log2 units; a partial is sum p v unnormalized with (m, l) as d_v41_flash_merge reads
 * them, l summing the unrounded p.
 *
 * The same object runs PLOW_DOP_INDEX_SCORE_PF (op 117, op_index_pf.cuh's operands and result) on
 * the same warpgroups: see namespace v41ix.
 */
#include "dev_isa.h"
#include "op_wg_sm90.cuh"
#include <type_traits>

#ifndef FA_ABL
#define FA_ABL 0
#endif
namespace v41fa {
using plow_wg::mbar_arrive;
using plow_wg::mbar_cp_async_arrive;
using plow_wg::mbar_init;
using plow_wg::mbar_wait;
using plow_wg::smem_u32;
using plow_wg::sw128;
using plow_wg::wg_commit;
using plow_wg::wg_cp_async16;
using plow_wg::wg_fence;
using plow_wg::wg_wait;

constexpr unsigned D = 512, M = 64, BN = 64, REGION = M * 128;  // one 64-dim region of a 64-row tile
constexpr unsigned Q_BYTES = 8 * REGION, K_BYTES = 8 * REGION, P_BYTES = REGION;
constexpr unsigned SMEM = 1024 + Q_BYTES + 2 * K_BYTES + P_BYTES + 4 * M * 4 + 2 * BN * 4 + 64;

// K-major SW128 (A operands and the QK B operand): stride between 8-row groups 1024 B
__device__ __forceinline__ uint64_t desc_k(uint32_t addr) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
// MN-major SW128 (V as the PV B operand): 64-dim regions REGION bytes apart (LBO), 8-key groups 1024 B (SBO)
__device__ __forceinline__ uint64_t desc_mn(uint32_t addr) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(REGION >> 4) << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void mma_n32(float* d, uint64_t da, uint64_t db, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
          "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(da), "l"(db), "r"(acc));
}
// m64n256k16, B transposed (MN-major)
__device__ __forceinline__ void mma_n256_tb(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
        "%128, %129, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(da), "l"(db), "r"(1));
}
__device__ __forceinline__ float ex2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void consumer_bar() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }
__device__ __forceinline__ unsigned ctr_poll(const unsigned* p) {
    unsigned value;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}
__device__ __forceinline__ void ctr_signal(unsigned* p) { asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory"); }

#ifndef FA_L2HINT
#define FA_L2HINT 0
#endif
/* 16 B gather, zero-filled when !ok; FA_L2HINT 1/2 = the .L2::128B / .L2::256B prefetch hint */
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) {
#if FA_L2HINT == 1
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0) : "memory");
#elif FA_L2HINT == 2
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0) : "memory");
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0) : "memory");
#endif
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src) : "memory");
}

struct Args {
    float* opart;
    float* ml;
    __nv_bfloat16* out;  // fused: the normalized output
    const __nv_bfloat16* q;
    const __nv_bfloat16* kv;   // union rows (gathered / fused), window rows (window arm)
    const __nv_bfloat16* win;  // window rows
    const unsigned char* uni;
    const float* sinks;
    unsigned n_head, window, n_tok, kv_mask, cap, nsplit, sp0, gsplit, q_pos0, P, HB, n_packs, n_hb, hdr, blocks, slice;
    float sl2;
    bool fused, has_win, valid;
};

/* The item's key rows: window rows [w_lo, w_lo + nw) of `win`, then union entries [c_lo, c_lo + nu). */
struct Rows {
    unsigned w_lo, nw, c_lo, nu;
};
__device__ __forceinline__ Rows item_rows(const Args& a, unsigned pk, unsigned share) {
    Rows r{0, 0, 0, 0};
    if (a.has_win) {
        const unsigned q0 = pk * a.P;
        const unsigned p_first = a.q_pos0 + q0, p_last = a.q_pos0 + min(q0 + a.P, a.n_tok) - 1u;
        r.w_lo = (a.window && p_first + 1u > a.window) ? p_first + 1u - a.window : 0u;
        r.nw = p_last + 1u - r.w_lo;
    }
    if (a.uni) {
        const unsigned c = reinterpret_cast<const unsigned*>(a.uni)[pk];
        r.c_lo = (unsigned)(((unsigned long long)c * share) / a.gsplit);
        r.nu = (unsigned)(((unsigned long long)c * (share + 1u)) / a.gsplit) - r.c_lo;
    }
    return r;
}
}  // namespace v41fa

/* op 117 (the indexer score, op_index_pf.cuh's operands) on this object's warpgroups: an item is 8
 * queries x 32 heads = 256 rows over a span of 128-key tiles. The producer streams K through a
 * 4-stage ring; consumer warpgroup w owns queries 4 w .. 4 w + 3 as two m64 blocks against the same
 * tile (m64n128k16 x 8 each, both in flight, block 0 reduced while block 1 runs), so a K tile crosses
 * L2 once per 8 queries. relu(s) * w_h sums over the heads in registers and by shuffle; the two warps
 * holding a query's 32 heads combine through shared memory. */
namespace v41ix {
constexpr unsigned QN = 8, ROWS = QN * 32, BN = 128, NST = 4, SPAN_TILES = 8;
constexpr unsigned QREG = ROWS * 128, KREG = BN * 128;  // [rows][128 B] per 64-dim region
constexpr unsigned Q_BYTES = 2 * QREG, K_BYTES = 2 * KREG;
constexpr unsigned SMEM = Q_BYTES + NST * K_BYTES + QN * 32 * 4 + QN * BN * 4;
static_assert(SMEM <= v41fa::SMEM, "the index path reuses the flash arena");
__device__ __forceinline__ void mma_n128(float* d, uint64_t da, uint64_t db, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db), "r"(acc));
}
__device__ __forceinline__ void mma_n256(float* d, uint64_t da, uint64_t db, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(da), "l"(db), "r"(acc));
}
__device__ __forceinline__ void wg_bar(unsigned w) { asm volatile("bar.sync %0, 128;\n" ::"r"(2u + w) : "memory"); }

struct Ix {
    float* score;
    const uint16_t *q, *k, *w;
    unsigned q_pos0, n_tok, cols, kv_stride, pool, blocks, slice;
    float scale;
};

template <bool PROD>
__device__ __forceinline__ void run(const Ix& a, unsigned char* dsm, uint64_t* full, uint64_t* empty, uint64_t* qfull, uint64_t* qempty,
                                    unsigned& n_tiles_done, unsigned& n_items_done) {
    using namespace plow_wg;
    unsigned char* Qs = dsm;
    unsigned char* Ks = Qs + Q_BYTES;
    float* part = reinterpret_cast<float*>(Ks + NST * K_BYTES);  // weights [QN][32], staged scores [QN][BN]
    const unsigned tid = threadIdx.x, wg = tid >> 7, lt = tid & 127u;
    const unsigned n_packs = (a.n_tok + QN - 1u) / QN;
    const unsigned n_spans = ((a.cols + BN - 1u) / BN + SPAN_TILES - 1u) / SPAN_TILES;
    for (unsigned item = a.slice; item < n_packs * n_spans; item += a.blocks) {
        const unsigned p = item / n_spans, sp = item % n_spans, t0 = p * QN;
        const unsigned pack_end = (a.q_pos0 + min(t0 + QN - 1u, a.n_tok - 1u) + 1u) / a.pool;
        const unsigned s_begin = sp * SPAN_TILES * BN;
        if (s_begin >= pack_end) continue;  // identical in every role
        const unsigned n_tiles = (min(s_begin + SPAN_TILES * BN, pack_end) - s_begin + BN - 1u) / BN;
        if constexpr (PROD) {
            if (n_items_done) mbar_wait(qempty, (n_items_done - 1u) & 1u);
            /* Q: rows t0 * 32 .. of [T][32][128] are contiguous; 8 lanes per 128 B line */
            const uint32_t qb = smem_u32(Qs);
            for (unsigned idx = lt; idx < ROWS * 16u; idx += 128u) {
                const unsigned r = idx >> 4, c = idx & 15u, row = t0 * 32u + r;
                const bool ok = row < a.n_tok * 32u;
                wg_cp_async16(qb + (c >> 3) * QREG + sw128(r, c & 7u), a.q + (size_t)(ok ? row : 0u) * 128u + c * 8u, ok ? 16 : 0);
            }
            mbar_cp_async_arrive(qfull);
            for (unsigned i = 0; i < n_tiles; i++) {
                const unsigned kt = n_tiles_done + i, s = kt % NST;
                if (kt >= NST) mbar_wait(&empty[s], ((kt / NST) - 1u) & 1u);
                const uint32_t kb = smem_u32(Ks + s * K_BYTES);
                const unsigned s0 = s_begin + i * BN;
                for (unsigned idx = lt; idx < BN * 16u; idx += 128u) {
                    const unsigned r = idx >> 4, c = idx & 15u;
                    const bool ok = s0 + r < a.cols;
                    wg_cp_async16(kb + (c >> 3) * KREG + sw128(r, c & 7u), a.k + (size_t)(ok ? s0 + r : 0u) * 128u + c * 8u, ok ? 16 : 0);
                }
                mbar_cp_async_arrive(&full[s]);
            }
        } else {
            /* S^T = K Q^T: group wg takes keys 64 wg .. of the tile as M (m64n256k16, N = 8 queries x 32 heads),
             * so a thread's accumulator row is one key against (query, head) columns 8 j + 2 t4 + e,
             * j = 4 q + jj: the head sum is 8 products in registers and two shuffles */
            const unsigned wq = lt >> 5, lane = lt & 31u, g = lane >> 2, t4 = lane & 3u, ct = tid;  // ct < 256
            float* wsm = part;              // [QN][32] f32 weights
            float* stg = part + QN * 32u;   // [QN][BN] f32 staged scores
            {
                const unsigned q = ct >> 5, h = ct & 31u, t = t0 + q;
                wsm[ct] = t < a.n_tok ? __uint_as_float((uint32_t)a.w[(size_t)t * 32u + h] << 16) : 0.f;
            }
            mbar_wait(qfull, n_items_done & 1u);
            v41fa::consumer_bar();  // wsm
            /* this thread's weights: query q, heads 8 jj + 2 t4 + e */
            float wr[QN][8];
#pragma unroll
            for (int q = 0; q < (int)QN; q++)
#pragma unroll
                for (int jj = 0; jj < 4; jj++)
#pragma unroll
                    for (int e = 0; e < 2; e++) wr[q][jj * 2 + e] = wsm[q * 32 + jj * 8 + t4 * 2 + e];
            const uint32_t qb = smem_u32(Qs);
            for (unsigned i = 0; i < n_tiles; i++) {
                const unsigned kt = n_tiles_done + i, s = kt % NST, s0 = s_begin + i * BN;
                mbar_wait(&full[s], (kt / NST) & 1u);
                const uint32_t kb = smem_u32(Ks + s * K_BYTES) + 64u * wg * 128u;
                /* two n128 halves (queries 0-3, 4-7): the first reduces while the second is on the tensor cores */
                float dA[64], dB[64];
                wg_fence();
#pragma unroll
                for (int ks = 0; ks < 8; ks++)
                    mma_n128(dA, v41fa::desc_k(kb + (ks >> 2) * KREG + (ks & 3) * 32), v41fa::desc_k(qb + (ks >> 2) * QREG + (ks & 3) * 32), ks);
                wg_commit();
#pragma unroll
                for (int ks = 0; ks < 8; ks++)
                    mma_n128(dB, v41fa::desc_k(kb + (ks >> 2) * KREG + (ks & 3) * 32), v41fa::desc_k(qb + (ks >> 2) * QREG + (ks & 3) * 32 + 128u * 128u),
                             ks);
                wg_commit();
                auto reduce = [&](const float* d, int qbase) {
#pragma unroll
                    for (int q = 0; q < 4; q++)
#pragma unroll
                        for (int hr = 0; hr < 2; hr++) {
                            float x = 0.f;
#pragma unroll
                            for (int jj = 0; jj < 4; jj++)
#pragma unroll
                                for (int e = 0; e < 2; e++) x = fmaf(wr[qbase + q][jj * 2 + e], fmaxf(d[(q * 4 + jj) * 4 + hr * 2 + e], 0.f), x);
                            x += __shfl_xor_sync(0xffffffffu, x, 1);
                            x += __shfl_xor_sync(0xffffffffu, x, 2);
                            if (t4 == 0) stg[wg * QN * 64u + (qbase + q) * 64u + 16u * wq + g + 8u * hr] = x;
                        }
                };
                wg_wait<1>();
                reduce(dA, 0);
                wg_wait<0>();
                mbar_arrive(&empty[s]);
                if (i + 1u == n_tiles) mbar_arrive(qempty);
                reduce(dB, 4);
                wg_bar(wg);
                /* this group's 8 queries x 64 scores, float4 per thread; the two groups drift apart, so
                 * one reduces while the other is on the tensor cores */
                {
                    const unsigned q = lt >> 4, c = (lt & 15u) * 4u, t = t0 + q, pos = s0 + 64u * wg + c;
                    if (t < a.n_tok) {
                        const unsigned rend = (a.q_pos0 + t + 1u) / a.pool;
                        const float4 v = *reinterpret_cast<const float4*>(stg + wg * QN * 64u + q * 64u + c);
                        float* dst = a.score + (size_t)t * a.kv_stride + pos;
                        if (pos + 4u <= rend && (reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
                            *reinterpret_cast<float4*>(dst) = make_float4(v.x * a.scale, v.y * a.scale, v.z * a.scale, v.w * a.scale);
                        } else {
                            const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                            for (int e = 0; e < 4; e++)
                                if (pos + e < rend) dst[e] = vv[e] * a.scale;
                        }
                    }
                }
                wg_bar(wg);  // the stage is rewritten by this group's next tile
            }
            v41fa::consumer_bar();  // stg / wsm are rewritten by the next item
        }
        n_tiles_done += n_tiles;
        n_items_done++;
    }
}
}  // namespace v41ix

extern "C" __device__ unsigned plow_pfflash_v41_abi = 1;
extern "C" __device__ unsigned plow_block_pfflash_v41 = 384;
extern "C" __device__ unsigned plow_arena_bytes_pfflash_v41 = v41fa::SMEM;

extern "C" __global__ __launch_bounds__(384, 1) void plow_sm90a_pfflash_v41(PlowProgram prog) {
    using namespace v41fa;
    extern __shared__ __align__(1024) unsigned char dsm[];
    unsigned char* base = reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(dsm) + 1023) & ~(uintptr_t)1023);
    unsigned char* Qs = base;
    unsigned char* Ks = Qs + Q_BYTES;                            // [2][8 regions][64 rows][128 B]
    unsigned char* Ps = Ks + 2 * K_BYTES;                        // [64 rows][128 B]
    float* redm = reinterpret_cast<float*>(Ps + P_BYTES);        // [2][M]
    float* redl = redm + 2 * M;                                  // [2][M]
    unsigned* kmsk = reinterpret_cast<unsigned*>(redl + 2 * M);  // [2][BN] union membership
    uint64_t* bars = reinterpret_cast<uint64_t*>(kmsk + 2 * BN);   // full[2] empty[2] qfull qempty
    uint64_t* full = bars;
    uint64_t* empty = bars + 2;
    uint64_t* qfull = bars + 4;
    uint64_t* qempty = bars + 5;
    __shared__ PlowStreamEnt entry_s;
    __shared__ Args args;
    __shared__ v41ix::Ix ix;
    __shared__ bool is_index;
    __shared__ uint64_t ixbars[2 * v41ix::NST + 2];  // op 117: full[NST] empty[NST] qfull qempty

    const unsigned tid = threadIdx.x, wg = tid / 128, lt = tid % 128;
    if (tid == 0) {
        for (int s = 0; s < 2; s++) {
            mbar_init(&full[s], 128);
            mbar_init(&empty[s], 256);
        }
        mbar_init(qfull, 128);
        mbar_init(qempty, 256);
        for (unsigned b = 0; b < v41ix::NST; b++) {
            mbar_init(&ixbars[b], 128);
            mbar_init(&ixbars[v41ix::NST + b], 256);
        }
        mbar_init(&ixbars[2 * v41ix::NST], 128);
        mbar_init(&ixbars[2 * v41ix::NST + 1], 256);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    // Each role runs its own copy of the entry loop (same barrier sequence), so setmaxnreg below
    // dominates all of its code and the consumers really get 232 registers.
    auto run = [&](auto producer) {
        constexpr bool PROD = decltype(producer)::value;
        unsigned n_tiles_done = 0, n_items_done = 0;  // barrier phases, identical in every role
        unsigned ix_tiles_done = 0, ix_items_done = 0;
        const unsigned lo = prog.gq_seg_ofs[prog.cur_seg], hi = prog.gq_seg_ofs[prog.cur_seg + 1];
        for (unsigned index = lo + blockIdx.x; index < hi; index += gridDim.x) {
            if (tid == 0) entry_s = prog.gq_stream[index];
            __syncthreads();
            const PlowStreamEnt entry = entry_s;
            if (entry.flags & PLOW_SE_XCTR) __trap();
            for (unsigned w = tid; w < entry.wait_len; w += blockDim.x) {
                const PlowWait wait = prog.waits[entry.wait_ofs + w];
                while (ctr_poll(PLOW_CTR(prog.counters, wait.id)) < wait.threshold) {
                }
            }
            __syncthreads();
            if (tid == 0) {
                const PlowDevInst* const in = prog.insts + entry.inst;
                is_index = in->op == PLOW_DOP_INDEX_SCORE_PF;
                if (is_index) {
                    /* op 117 (op_index_pf.cuh): t0=Score t1=Qidx t2=Kidx t3=W t4=kv_len i0=n_tok i1=heads i2=kv_stride
                     * i3=dim i4=pool f0=scale */
                    const unsigned len_tok = (unsigned)static_cast<const int*>(prog.tensors[in->t[4]])[0], pool = in->i[4] ? in->i[4] : 1u;
                    ix = v41ix::Ix{static_cast<float*>(prog.tensors[in->t[0]]), static_cast<const uint16_t*>(prog.tensors[in->t[1]]),
                                   static_cast<const uint16_t*>(prog.tensors[in->t[2]]), static_cast<const uint16_t*>(prog.tensors[in->t[3]]),
                                   len_tok - in->i[0], in->i[0], len_tok / pool, in->i[2], pool, in->blocks, entry.slice, in->fj[0].f};
                    args.valid = (in->i[1] == 0u || in->i[1] == 32u) && (in->i[3] == 0u || in->i[3] == 128u) && in->blocks > 0 &&
                                 entry.slice < in->blocks;
                    args.blocks = in->blocks;
                    args.slice = entry.slice;
                }
            }
            __syncthreads();
            if (is_index) {
                if (!args.valid) __trap();
                v41ix::run<PROD>(ix, dsm, ixbars, ixbars + v41ix::NST, ixbars + 2 * v41ix::NST, ixbars + 2 * v41ix::NST + 1, ix_tiles_done,
                                 ix_items_done);
                __threadfence();
                __syncthreads();
                for (unsigned s = tid; s < entry.succ_len; s += blockDim.x) ctr_signal(PLOW_CTR(prog.counters, prog.succs[entry.succ_ofs + s]));
                __syncthreads();
                continue;
            }
            if (tid == 0) {
                const PlowDevInst* const in = prog.insts + entry.inst;
                const unsigned H = in->i[1];
                const bool gather = in->t[7] != PLOW_TENSOR_NONE;
                const bool fused = gather && in->t[5] != PLOW_TENSOR_NONE && in->t[5] != in->t[4];
                const unsigned P = gather ? (in->fj[1].u ? in->fj[1].u : 8u) : 64u / min(H, 64u);
                args.valid = in->op == PLOW_DOP_FLASH_MLA_PREFILL && (in->i[3] >> 31) && in->i[0] == 1u && P >= 1u && P <= 64u &&
                             64u % P == 0 && H % (64u / P) == 0 && in->t[0] != PLOW_TENSOR_NONE && in->t[2] != PLOW_TENSOR_NONE &&
                             in->t[4] != PLOW_TENSOR_NONE && in->t[6] != PLOW_TENSOR_NONE &&
                             (fused ? (in->i[7] >> 16) <= 1u : (in->t[1] != PLOW_TENSOR_NONE &&
                                                               (in->t[5] == PLOW_TENSOR_NONE || in->t[5] == in->t[4]))) &&
                             in->blocks > 0 && entry.slice < in->blocks;
                if (args.valid) {
                    args.opart = fused ? nullptr : static_cast<float*>(prog.tensors[in->t[0]]);
                    args.ml = fused ? nullptr : static_cast<float*>(prog.tensors[in->t[1]]);
                    args.out = fused ? static_cast<__nv_bfloat16*>(prog.tensors[in->t[0]]) : nullptr;
                    args.q = static_cast<const __nv_bfloat16*>(prog.tensors[in->t[2]]);
                    args.kv = static_cast<const __nv_bfloat16*>(prog.tensors[in->t[4]]);
                    args.win = fused ? static_cast<const __nv_bfloat16*>(prog.tensors[in->t[5]]) : args.kv;
                    args.uni = gather ? static_cast<const unsigned char*>(prog.tensors[in->t[7]]) : nullptr;
                    args.sinks = fused && in->t[3] != PLOW_TENSOR_NONE ? static_cast<const float*>(prog.tensors[in->t[3]]) : nullptr;
                    args.fused = fused;
                    args.has_win = fused || !gather;
                    args.n_head = H;
                    args.window = in->i[3] & 0x7fffffffu;
                    args.n_tok = in->i[4];
                    args.kv_mask = in->i[5];
                    args.cap = in->i[6];
                    const unsigned sw = in->i[7];
                    args.nsplit = fused ? 1u : (sw ? (sw >> 8) & 0xffu : 1u);
                    args.sp0 = fused ? 0u : sw & 0xffu;
                    args.gsplit = gather && !fused ? ((sw >> 16) ? (sw >> 16) : 1u) : 1u;
                    args.q_pos0 = (unsigned)static_cast<const int*>(prog.tensors[in->t[6]])[0] - in->i[4];
                    args.P = P;
                    args.HB = 64u / P;
                    args.n_packs = (in->i[4] + P - 1u) / P;
                    args.n_hb = H / args.HB;
                    args.hdr = (args.n_packs * 4u + 255u) / 256u * 256u;
                    args.blocks = in->blocks;
                    args.slice = entry.slice;
                    args.sl2 = in->fj[0].f * 1.4426950408889634f;
                }
            }
            __syncthreads();
            if (!args.valid) __trap();
            const Args& a = args;
            const unsigned n_items = a.n_packs * a.n_hb * a.gsplit;

            if constexpr (PROD) {
                /* ---------------- producer: two threads per row, 16 B chunks 2 i + half ---------------- */
                const uint16_t* q16 = reinterpret_cast<const uint16_t*>(a.q);
                const uint16_t* kv16 = reinterpret_cast<const uint16_t*>(a.kv);
                const uint16_t* win16 = reinterpret_cast<const uint16_t*>(a.win);
                /* a warp instruction moves 4 whole 128 B lines: lane chunk c8 = lt % 8 of a 64-dim region, rows
                 * lt / 8 + 16 k (k < 4) */
                const unsigned c8 = lt & 7u, rg = lt >> 3;
                for (unsigned item = a.slice; item < n_items; item += a.blocks) {
                    const unsigned hb = item % a.n_hb, share = (item / a.n_hb) % a.gsplit, pk = item / (a.n_hb * a.gsplit);
                    const unsigned q0 = pk * a.P;
                    const Rows rw = item_rows(a, pk, share);
                    if (n_items_done) mbar_wait(qempty, (n_items_done - 1) & 1);
                    {
                        const uint32_t qb = smem_u32(Qs);
#pragma unroll
                        for (unsigned k = 0; k < 4u; k++) {
                            const unsigned r = rg + 16u * k, t = q0 + r / a.HB, h = hb * a.HB + r % a.HB;
                            const bool ok = t < a.n_tok;
                            const uint16_t* src = q16 + ((size_t)(ok ? t : 0u) * a.n_head + h) * D + c8 * 8u;
#pragma unroll
                            for (unsigned reg = 0; reg < 8u; reg++) cp16(qb + reg * REGION + sw128(r, c8), src + reg * 64u, ok);
                        }
                    }
                    mbar_cp_async_arrive(qfull);
                    const int* upos = a.uni ? reinterpret_cast<const int*>(a.uni + a.hdr + (size_t)pk * a.cap * 12u) + rw.c_lo : nullptr;
                    const unsigned* ulo = a.uni ? reinterpret_cast<const unsigned*>(a.uni + a.hdr + (size_t)pk * a.cap * 12u + (size_t)a.cap * 4u) + rw.c_lo
                                                : nullptr;
                    const unsigned n_rows = rw.nw + rw.nu, n_tiles = (n_rows + BN - 1u) / BN;
                    /* source row g of the item's key list (nullptr past the end) */
                    auto row_src = [&](unsigned g) -> const uint16_t* {
                        if (g < rw.nw) {
                            const unsigned pos = rw.w_lo + g;
                            return win16 + (size_t)(a.kv_mask == 0xFFFFFFFFu ? pos : (pos & a.kv_mask)) * D;
                        }
                        if (g < n_rows) return kv16 + (size_t)upos[g - rw.nw] * D;
                        return nullptr;
                    };
                    for (unsigned i = 0; i < n_tiles; i++) {
                        const unsigned kt = n_tiles_done + i, s = kt & 1u;
                        const uint16_t* src[4];
#pragma unroll
                        for (unsigned k = 0; k < 4u; k++) src[k] = row_src(i * BN + rg + 16u * k);
                        if (kt >= 2) mbar_wait(&empty[s], ((kt >> 1) - 1) & 1);
                        const uint32_t kb = smem_u32(Ks + s * K_BYTES);
#pragma unroll
                        for (unsigned k = 0; k < 4u; k++) {
                            const unsigned r = rg + 16u * k, g = i * BN + r;
                            if (c8 == 0 && g >= rw.nw && g < n_rows) cp_async4(smem_u32(kmsk + s * BN + r), ulo + (g - rw.nw));
                            const bool ok = src[k] != nullptr;
                            const uint16_t* sp = (ok ? src[k] : kv16) + c8 * 8u;
#if FA_ABL != 1
#pragma unroll
                            for (unsigned reg = 0; reg < 8u; reg++) cp16(kb + reg * REGION + sw128(r, c8), sp + reg * 64u, ok);
#endif
                        }
                        mbar_cp_async_arrive(&full[s]);
                    }
                    n_tiles_done += n_tiles;
                    n_items_done++;
                }
            } else {
                /* ---------------- consumers ---------------- */
                const unsigned warp = lt / 32u, lane = lt % 32u, g8 = lane / 4u, t4 = lane % 4u;
                const unsigned r0 = warp * 16u + g8;  // rows r0, r0 + 8
                for (unsigned item = a.slice; item < n_items; item += a.blocks) {
                    const unsigned hb = item % a.n_hb, share = (item / a.n_hb) % a.gsplit, pk = item / (a.n_hb * a.gsplit);
                    const unsigned q0 = pk * a.P;
                    const Rows rw = item_rows(a, pk, share);
                    const unsigned n_rows = rw.nw + rw.nu, n_tiles = (n_rows + BN - 1u) / BN;
                    const unsigned qi[2] = {r0 / a.HB, (r0 + 8u) / a.HB};
                    const unsigned qp[2] = {a.q_pos0 + q0 + qi[0], a.q_pos0 + q0 + qi[1]};
                    const bool qv[2] = {q0 + qi[0] < a.n_tok, q0 + qi[1] < a.n_tok};
                    float o[128];
#pragma unroll
                    for (int j = 0; j < 128; j++) o[j] = 0.f;
                    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
                    mbar_wait(qfull, n_items_done & 1);
                    if (n_tiles == 0) mbar_arrive(qempty);
                    const uint32_t qb = smem_u32(Qs);
                    for (unsigned i = 0; i < n_tiles; i++) {
                        const unsigned kt = n_tiles_done + i, s = kt & 1u;
                        mbar_wait(&full[s], (kt >> 1) & 1);
#if FA_ABL == 3
                        if (i + 1u == n_tiles) mbar_arrive(qempty);
                        mbar_arrive(&empty[s]);
                        continue;
#endif
                        const uint32_t kb = smem_u32(Ks + s * K_BYTES);
                        float sc[16];
#if FA_ABL != 2
                        wg_fence();
#pragma unroll
                        for (int ks = 0; ks < 32; ks++) {
                            const uint32_t off = (ks / 4) * REGION + (ks % 4) * 32;
                            mma_n32(sc, desc_k(qb + off), desc_k(kb + off + wg * 32u * 128u), ks);
                        }
                        wg_commit();
                        wg_wait<0>();
#else
#pragma unroll
                        for (int j = 0; j < 16; j++) sc[j] = 0.f;
#endif
                        if (i + 1u == n_tiles) mbar_arrive(qempty);  // Q is free once the last QK is done
                        /* mask + scale; this thread's keys: 32 wg + 8 j + 2 t4 + e */
                        float pmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
                        for (int j = 0; j < 4; j++)
#pragma unroll
                            for (int e = 0; e < 2; e++) {
                                const unsigned key = wg * 32u + j * 8u + t4 * 2u + e, gk = i * BN + key;
                                unsigned bits = 0;
                                if (gk < rw.nw) {
                                    const unsigned pos = rw.w_lo + gk;
#pragma unroll
                                    for (int h = 0; h < 2; h++)
                                        if (qv[h] && pos <= qp[h] && (!a.window || qp[h] - pos < a.window)) bits |= 1u << qi[h];
                                } else if (gk < n_rows) {
                                    bits = kmsk[s * BN + key];
                                }
#pragma unroll
                                for (int h = 0; h < 2; h++) {
                                    float& v = sc[j * 4 + h * 2 + e];
                                    v = (bits >> qi[h]) & 1u ? v * a.sl2 : -INFINITY;
                                    pmax[h] = fmaxf(pmax[h], v);
                                }
                            }
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            pmax[h] = fmaxf(pmax[h], __shfl_xor_sync(0xffffffffu, pmax[h], 1));
                            pmax[h] = fmaxf(pmax[h], __shfl_xor_sync(0xffffffffu, pmax[h], 2));
                        }
                        if (t4 == 0) {
                            redm[wg * M + r0] = pmax[0];
                            redm[wg * M + r0 + 8u] = pmax[1];
                        }
                        consumer_bar();
                        float msafe[2], alpha[2];
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const unsigned r = r0 + h * 8u;
                            const float mnew = fmaxf(mrow[h], fmaxf(redm[r], redm[M + r]));
                            msafe[h] = mnew == -INFINITY ? 0.f : mnew;
                            alpha[h] = ex2(mrow[h] - msafe[h]);
                            mrow[h] = mnew;
                        }
                        float psum[2] = {0.f, 0.f};
#pragma unroll
                        for (int j = 0; j < 4; j++)
#pragma unroll
                            for (int h = 0; h < 2; h++) {
                                const float p0 = ex2(sc[j * 4 + h * 2] - msafe[h]), p1 = ex2(sc[j * 4 + h * 2 + 1] - msafe[h]);
                                psum[h] += p0 + p1;
                                const unsigned col = wg * 32u + j * 8u + t4 * 2u;
                                *reinterpret_cast<uint32_t*>(Ps + sw128(r0 + h * 8u, col / 8u) + (col % 8u) * 2u) = plow_wg::f2_to_bf2(p0, p1);
                            }
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            psum[h] += __shfl_xor_sync(0xffffffffu, psum[h], 1);
                            psum[h] += __shfl_xor_sync(0xffffffffu, psum[h], 2);
                            lrow[h] = lrow[h] * alpha[h] + psum[h];
                        }
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        consumer_bar();
#pragma unroll
                        for (int j = 0; j < 32; j++) {
                            o[j * 4 + 0] *= alpha[0];
                            o[j * 4 + 1] *= alpha[0];
                            o[j * 4 + 2] *= alpha[1];
                            o[j * 4 + 3] *= alpha[1];
                        }
#if FA_ABL != 2
                        const uint32_t pb = smem_u32(Ps);
                        wg_fence();
#pragma unroll
                        for (int kk = 0; kk < 4; kk++)
                            mma_n256_tb(o, desc_k(pb + kk * 32u), desc_mn(kb + (4u * wg) * REGION + kk * 2u * 1024u));
                        wg_commit();
                        wg_wait<0>();
#endif
                        mbar_arrive(&empty[s]);
                    }
                    /* l over both key halves */
                    if (t4 == 0) {
                        redl[wg * M + r0] = lrow[0];
                        redl[wg * M + r0 + 8u] = lrow[1];
                    }
                    consumer_bar();
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const unsigned r = r0 + h * 8u, t = q0 + r / a.HB, head = hb * a.HB + r % a.HB;
                        if (t >= a.n_tok) continue;
                        const float l = redl[r] + redl[M + r];
                        if (a.fused) {
                            /* the sink joins the denominator with no value row (natural units -> log2) */
                            const float m = mrow[h] == -INFINITY ? 0.f : mrow[h];
                            const float lt_ = l + (a.sinks ? ex2(a.sinks[head] * 1.4426950408889634f - m) : 0.f);
                            const float inv = lt_ > 0.f ? 1.f / lt_ : 0.f;
                            uint32_t* op = reinterpret_cast<uint32_t*>(a.out + ((size_t)t * a.n_head + head) * D + wg * 256u + t4 * 2u);
#pragma unroll
                            for (int j = 0; j < 32; j++) op[j * 4] = plow_wg::f2_to_bf2(o[j * 4 + h * 2] * inv, o[j * 4 + h * 2 + 1] * inv);
                        } else {
                            const size_t bi = ((size_t)t * a.n_head + head) * a.nsplit + a.sp0 + share;
                            float* op = a.opart + bi * D + wg * 256u + t4 * 2u;
#pragma unroll
                            for (int j = 0; j < 32; j++) *reinterpret_cast<float2*>(op + j * 8) = make_float2(o[j * 4 + h * 2], o[j * 4 + h * 2 + 1]);
                            if (wg == 0 && t4 == 0) *reinterpret_cast<float2*>(a.ml + bi * 2) = make_float2(mrow[h] == -INFINITY ? -3.0e38f : mrow[h], l);
                        }
                    }
                    consumer_bar();  // redl reuse
                    n_tiles_done += n_tiles;
                    n_items_done++;
                }
            }
            __threadfence();
            __syncthreads();
            for (unsigned s = tid; s < entry.succ_len; s += blockDim.x) ctr_signal(PLOW_CTR(prog.counters, prog.succs[entry.succ_ofs + s]));
            __syncthreads();
        }
    };
    // the consumers hold O (128 f32) and S in registers: 40 for the producer, 232 for them
    if (wg == 2) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
        run(std::true_type{});
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
        run(std::false_type{});
    }
}
