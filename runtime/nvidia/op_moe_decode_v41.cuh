/* op_moe_decode_v41.cuh -- DeepSeek-V4.1 routed experts at DECODE (T = B <= 64 rows, top-6 of 384)
 * on sm_90a, on the prefill role's operands (interp_sm90a_pfmoe_fp4.cu, ops 85/86): the same
 * wtab/stab [E][3] {gate, up, down} pointer tables (fp4 e2m1 rows [N][K/2], low nibble = even k, ue8m0
 * [N][K/32]), MoeAlignPf's meta (rowoff[E] | cnt[E] | tilep[E+1]), row_token / row_partidx / row_gate,
 * and the same numerics (GluEpiS / DownEpiS): g, u = bf16(fq(x) . W) f32-accumulated; fu =
 * fq(bf16(rw * silu(min(g, L)) * clamp(u, -L, L))) per 32-feature block; part = rw? * bf16(fu . W2).
 *
 * Decode has ~1 row per active expert, so the weights are the whole cost: every active expert's
 * bytes must stream once at HBM rate. One WARP per unit, weights on mma.sync m16n8k16's M side, the
 * expert's tokens (a group of 8 rows) on N:
 *   GLU  item = (expert, group, 32 features): gate and up rows f0..f0+31 (4 m16 tiles), so an item
 *        owns one fq block; 20 chunks of 512 k, alternating gate and up.
 *   DOWN group = (expert, group): H / 32 chunks of 32 output rows.
 * All chunks form one stream cut into equal contiguous per-warp shares (no wave tail at any B). A GLU
 * item cut by a share boundary goes through scratch: the last of its warps (counter, self-resetting)
 * folds the partials in share order.
 * Weights stream through a per-warp ring of TMA copies (+ mbarrier): a GLU chunk is 256 B of each of
 * 32 gate or up rows, ONE tensor copy through the slab map the runtime writes ahead of the weight
 * slab (experts.rs EXPERT_TMAP_BYTES; 32 separate 256 B bulk copies serialized on the SM's TMA unit
 * and capped GLU near 1.1 TB/s); scale words are register-prefetched. A DOWN chunk is 32 whole rows
 * with scales (two bulk copies). Per-lane loads of 64 B row pieces
 * (ld or cp.async) cap near 2 TB/s once decode ALU shares the warp; the bulk ring keeps S - 1 chunks
 * in flight while the warp decodes, across unit boundaries.
 * K is PERMUTED, consistently for weights and tokens: in a 128-wide (GLU) / 64-wide (DOWN) sub-chunk
 * thread t (lane & 3) owns physical k [t*CK/4, (t+1)*CK/4) -- contiguous fp4 bytes of each of its
 * rows, contiguous bf16 of its token, one ue8m0 byte -- and pairs elements (8i+j, 8i+j+4) of each
 * 8-element word into the A/B registers of mma k-steps. A dot product does not care which k slot a
 * product lands in.
 * e2m1 -> bf16 is v * 2^-126 by bit placement (sign 15, magnitude 6-8), times 2^(s-1): exact in bf16,
 * the prefill role's decode.
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace plow_mdv {
constexpr unsigned UNUSED = 0xFFFFFFFFu;
constexpr unsigned ACT_SWIGLU_CLAMP = 4u;
constexpr unsigned TOK = 8;                /* rows per group: one mma n8 */
constexpr unsigned PART = TOK * 64;        /* floats per GLU split partial: [8 rows][64 = gate 32 | up 32] */
constexpr int GLU_S = 3, DOWN_S = 2;
constexpr unsigned GLU_SLOT = 32 * 256;            /* 32 rows x 256 B */
constexpr unsigned RING_BYTES = GLU_S * GLU_SLOT;   /* per warp; DOWN uses DOWN_S slots of 17 I bytes (I <= 576) */
constexpr unsigned BAR_BYTES = 32;                  /* per warp */
constexpr int DOWN_NSUB = 9;                        /* I / 64 <= 9 */

__host__ __device__ constexpr size_t pfx_bytes(unsigned E) { return ((size_t)E * 4 + 15) / 16 * 16; }

__device__ __forceinline__ float rbf(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float round_e4m3(float y) {
    const float a = fabsf(y);
    if (a == 0.f) return y;
    const int e = (int)((__float_as_uint(a) >> 23) & 0xffu) - 127;
    const float quantum = __uint_as_float((uint32_t)((e < -6 ? -6 : e) - 3 + 127) << 23);
    const float q = fminf(rintf(a / quantum) * quantum, 448.f);
    return y < 0.f ? -q : q;
}
/* 8 e2m1 (nibble j = element j) -> pairs (j, j+4) as bf16x2, scaled by sc2. A masked nibble pair times
 * (2^a + 2^b) lands two disjoint copies: magnitude at bits 6-8 from one, sign at 15 from the other. */
__device__ __forceinline__ void dec_word(uint32_t w, __nv_bfloat162 sc2, uint32_t* o) {
    const uint32_t M = 0x81C081C0u, w8 = w >> 8;
    uint32_t r[4];
    r[0] = ((w & 0x000F000Fu) * 0x1040u) & M;
    r[1] = ((w & 0x00F000F0u) * 0x0104u) & M;
    r[2] = ((w8 & 0x000F000Fu) * 0x1040u) & M;
    r[3] = ((w8 & 0x00F000F0u) * 0x0104u) & M;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const __nv_bfloat162 v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&r[j]), sc2);
        o[j] = *reinterpret_cast<const uint32_t*>(&v);
    }
}
/* the same pairing on 8 bf16 (4 words) */
__device__ __forceinline__ void pair_tok(const uint32_t* x, uint32_t* o) {
    o[0] = __byte_perm(x[0], x[2], 0x5410);
    o[1] = __byte_perm(x[0], x[2], 0x7632);
    o[2] = __byte_perm(x[1], x[3], 0x5410);
    o[3] = __byte_perm(x[1], x[3], 0x7632);
}
__device__ __forceinline__ __nv_bfloat162 pow2x2(uint32_t s) {  /* 2^(s-1) in both halves */
    const uint32_t b = (s + 126u) << 7;
    const uint32_t v = b | (b << 16);
    return *reinterpret_cast<const __nv_bfloat162*>(&v);
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void lds128(uint32_t a, uint32_t* v) {
    asm("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(a));
}
__device__ __forceinline__ void lds64(uint32_t a, uint32_t* v) {
    asm("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v[0]), "=r"(v[1]) : "r"(a));
}
__device__ __forceinline__ uint32_t lds8(uint32_t a) {
    uint32_t v;
    asm("ld.shared.u8 %0, [%1];\n" : "=r"(v) : "r"(a));
    return v;
}
__device__ __forceinline__ void mbar_init(uint32_t b) { asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" ::"r"(b) : "memory"); }
__device__ __forceinline__ void mbar_inval(uint32_t b) { asm volatile("mbarrier.inval.shared.b64 [%0];\n" ::"r"(b) : "memory"); }
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
    uint32_t ok = 0;
    while (!ok)
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(ok) : "r"(b), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma3(uint32_t dst, const void* map, unsigned c0, unsigned c1, unsigned c2, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(dst),
                 "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
/* the slab map (experts.rs EXPERT_TMAP_BYTES ahead of expert 0's gate) */
constexpr unsigned TMAP_BYTES = 4096;
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst), "l"(src), "r"(bytes),
                 "r"(bar) : "memory");
}

/* A warp's ring: S slots, one mbarrier each; chunk n of the warp's stream lives in slot n % S and
 * completes phase n / S of its barrier. */
struct Ring {
    uint32_t buf, bar;
    unsigned issued, used;
    __device__ __forceinline__ uint32_t slot(unsigned n, int S, unsigned bytes) const { return buf + (n % S) * bytes; }
    __device__ __forceinline__ uint32_t bar_of(unsigned n, int S) const { return bar + (n % S) * 8u; }
    /* waits for the oldest chunk; returns its slot. The address is laundered through the wait so the
     * (non-volatile) smem reads cannot be scheduled above it. */
    __device__ __forceinline__ uint32_t wait(int S, unsigned bytes) const {
        mbar_wait(bar_of(used, S), (used / S) & 1u);
        uint32_t a = slot(used, S, bytes);
        asm volatile("" : "+r"(a)::"memory");
        return a;
    }
};
__device__ __forceinline__ Ring ring_open(unsigned char* smem, unsigned E, int S) {
    const unsigned wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem) + (uint32_t)pfx_bytes(E);
    /* tensor copies land 128 B aligned; the arena itself is only 16 B aligned */
    const uint32_t ring = (base + nw * BAR_BYTES + 127u) & ~127u;
    Ring r{ring + wid * RING_BYTES, base + wid * BAR_BYTES, 0u, 0u};
    if ((int)(threadIdx.x & 31u) < S) mbar_init(r.bar + (threadIdx.x & 31u) * 8u);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    __syncwarp();
    return r;
}
__device__ __forceinline__ void ring_close(const Ring& r, int S) {
    __syncwarp();
    if ((int)(threadIdx.x & 31u) < S) mbar_inval(r.bar + (threadIdx.x & 31u) * 8u);
}

/* groups (ceil(cnt / TOK)) per expert, inclusive prefix into pfx[E]; all threads; returns the total */
__device__ __forceinline__ unsigned group_prefix(const int* cnt, unsigned E, int* pfx) {
    __syncthreads();
    if (threadIdx.x < 32) {
        const unsigned lane = threadIdx.x, per = (E + 31) / 32, e0 = lane * per;
        unsigned s = 0;
        for (unsigned e = e0; e < min(E, e0 + per); e++) s += (cnt[e] + TOK - 1) / TOK;
        unsigned inc = s;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned v = __shfl_up_sync(0xffffffffu, inc, o);
            if ((int)lane >= o) inc += v;
        }
        unsigned run = inc - s;
        for (unsigned e = e0; e < min(E, e0 + per); e++) {
            run += (cnt[e] + TOK - 1) / TOK;
            pfx[e] = (int)run;
        }
    }
    __syncthreads();
    return (unsigned)pfx[E - 1];
}
/* group -> (expert, group within the expert) */
__device__ __forceinline__ void group_of(const int* pfx, unsigned E, unsigned grp, unsigned& e, unsigned& tg) {
    unsigned lo = 0, hi = E - 1;
    while (lo < hi) {
        const unsigned mid = (lo + hi) >> 1;
        if ((unsigned)pfx[mid] > grp) hi = mid;
        else lo = mid + 1;
    }
    e = lo;
    tg = grp - (lo ? (unsigned)pfx[lo - 1] : 0u);
}

/* Work is a stream of n chunks split into equal contiguous shares: warp w of tw takes
 * [share(w), share(w + 1)), so every warp finishes together; tw <= n keeps every share non-empty. */
__device__ __forceinline__ unsigned share(unsigned w, unsigned n, unsigned tw) { return (unsigned)((unsigned long long)w * n / tw); }
/* the warp whose share holds chunk x */
__device__ __forceinline__ unsigned owner(unsigned x, unsigned n, unsigned tw) {
    return (unsigned)((((unsigned long long)x + 1) * tw - 1) / n);
}

struct GluArgs {
    const unsigned long long *wtab, *stab;
    const int *rowoff, *cnt, *pfx;
    unsigned H, E, ntile, nch, a, b;  /* [a, b): this warp's share of the items * nch chunk stream */
    const void* tmap;
};
struct GluUnit {
    const uint8_t *s0, *s1;            /* gate / up scale rows f0.. */
    int k0, k1, row0, rend;            /* chunks [k0, k1) of item it */
    unsigned f0, it, mat;             /* mat: the gate matrix's index in the slab map */
};
__device__ __forceinline__ GluUnit glu_unit(const GluArgs& A, unsigned it) {
    GluUnit U;
    U.it = it;
    unsigned e, tg;
    group_of(A.pfx, A.E, it / A.ntile, e, tg);
    U.f0 = (it % A.ntile) * 32;
    U.mat = e * 3;
    U.row0 = A.rowoff[e] + (int)(tg * TOK);
    U.rend = min(A.rowoff[e] + A.cnt[e], U.row0 + (int)TOK);
    const size_t lds = A.H / 32;
    U.s0 = (const uint8_t*)A.stab[(size_t)e * 3] + U.f0 * lds;
    U.s1 = (const uint8_t*)A.stab[(size_t)e * 3 + 1] + U.f0 * lds;
    U.k0 = (int)(max(A.a, it * A.nch) - it * A.nch);
    U.k1 = (int)(min(A.b, (it + 1) * A.nch) - it * A.nch);
    return U;
}
/* chunk c: 512 k (kr = c / 2) of gate (c even) or up (c odd) rows f0 .. f0 + 31, 256 B each */
__device__ __forceinline__ void glu_issue(Ring& r, const GluUnit& U, int c, const void* tmap) {
    const uint32_t sl = r.slot(r.issued, GLU_S, GLU_SLOT), b = r.bar_of(r.issued, GLU_S);
    r.issued++;
    if ((threadIdx.x & 31u) == 0) {
        mbar_expect(b, GLU_SLOT);
        tma3(sl, tmap, (unsigned)(c >> 1) * 256u, U.f0, U.mat + (unsigned)(c & 1), b);
    }
}
struct GluCursor {
    unsigned it;
    int c;
    GluUnit U;
};
__device__ __forceinline__ void glu_top_up(Ring& r, GluCursor& q, const GluArgs& A) {
    while (r.issued - r.used < (unsigned)GLU_S) {
        while (q.c >= q.U.k1) {
            if ((q.it + 1) * A.nch >= A.b) return;
            q.U = glu_unit(A, ++q.it);
            q.c = q.U.k0;
        }
        glu_issue(r, q.U, q.c++, A.tmap);
    }
}
/* thread t's token k of k-range kr, sub j: kr*512 + j*128 + t*32 .. +32 (xp includes t*32) */
__device__ __forceinline__ void glu_tok(uint32_t (&b)[16], const __nv_bfloat16* xp, int kr, int j) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        if (xp) {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(xp + (size_t)kr * 512 + j * 128) + i);
            b[4 * i] = v.x, b[4 * i + 1] = v.y, b[4 * i + 2] = v.z, b[4 * i + 3] = v.w;
        } else {
            b[4 * i] = b[4 * i + 1] = b[4 * i + 2] = b[4 * i + 3] = 0u;
        }
    }
}
/* scale word t (bytes 4t .. 4t+3: sub t's four threads) of rows q*16 + h*8 + g of chunk c */
__device__ __forceinline__ void glu_scales(uint32_t (&w)[4], const GluUnit& U, int c, size_t lds) {
    const unsigned lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    const uint8_t* S = (c & 1) ? U.s1 : U.s0;
#pragma unroll
    for (int i = 0; i < 4; i++) w[i] = __ldg(reinterpret_cast<const uint32_t*>(S + ((i >> 1) * 16 + (i & 1) * 8 + g) * lds + (size_t)(c >> 1) * 16 + t * 4));
}
/* one chunk into tiles 2 * mat, 2 * mat + 1 (the pair is swapped through `cur` with selects, so
 * the mma chain and its order are the unrolled form's). Rolled over the four 128-k subs with this
 * sub's tokens loaded one ahead: a compact body, since its cold fetch is paid every step. */
__device__ __forceinline__ void glu_chunk(uint32_t sl, const uint32_t (&sw)[4], float (&acc)[4][4], const __nv_bfloat16* xp, int kr,
                                          unsigned mat) {
    const unsigned lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    float cur[2][4];
#pragma unroll
    for (int q = 0; q < 2; q++)
#pragma unroll
        for (int k = 0; k < 4; k++) cur[q][k] = mat ? acc[2 + q][k] : acc[q][k];
    uint32_t tk[16];
    glu_tok(tk, xp, kr, 0);
#pragma unroll 1
    for (int j = 0; j < 4; j++) {
        uint32_t bp[16];
#pragma unroll
        for (int i = 0; i < 4; i++) pair_tok(&tk[4 * i], &bp[4 * i]);
        if (j < 3) glu_tok(tk, xp, kr, j + 1);
#pragma unroll
        for (int q = 0; q < 2; q++) {
            uint32_t ap[2][16];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned R = q * 16 + h * 8 + g;
                uint32_t a[4];
                lds128(sl + R * 256 + j * 64 + t * 16, a);
                const uint32_t w = __shfl_sync(0xffffffffu, sw[q * 2 + h], (lane & ~3u) | (unsigned)j);
                const __nv_bfloat162 sc = pow2x2((w >> (8 * t)) & 0xffu);
#pragma unroll
                for (int i = 0; i < 4; i++) dec_word(a[i], sc, &ap[h][4 * i]);
            }
#pragma unroll
            for (int st = 0; st < 8; st++) {
                const uint32_t aa[4] = {ap[0][2 * st], ap[1][2 * st], ap[0][2 * st + 1], ap[1][2 * st + 1]};
                mma(cur[q], aa, bp[2 * st], bp[2 * st + 1]);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 2; q++)
#pragma unroll
        for (int k = 0; k < 4; k++) {
            acc[q][k] = mat ? acc[q][k] : cur[q][k];
            acc[2 + q][k] = mat ? cur[q][k] : acc[2 + q][k];
        }
}

struct DownArgs {
    const unsigned long long *wtab, *stab;
    const int *rowoff, *cnt, *pfx;
    unsigned I, E, n, a, b;  /* n = H / 32 chunks per group; [a, b): this warp's share */
};
struct DownUnit {
    const uint8_t *w, *s;
    int c0, c1, row0, rend;
};
__device__ __forceinline__ DownUnit down_unit(const DownArgs& A, unsigned gi) {
    DownUnit U;
    unsigned e, tg;
    group_of(A.pfx, A.E, gi, e, tg);
    U.c0 = (int)(max(A.a, gi * A.n) - gi * A.n);
    U.c1 = (int)(min(A.b, (gi + 1) * A.n) - gi * A.n);
    U.row0 = A.rowoff[e] + (int)(tg * TOK);
    U.rend = min(A.rowoff[e] + A.cnt[e], U.row0 + (int)TOK);
    U.w = (const uint8_t*)A.wtab[(size_t)e * 3 + 2];
    U.s = (const uint8_t*)A.stab[(size_t)e * 3 + 2];
    return U;
}
/* chunk c: output rows 32c .. 32c + 31, contiguous in both the fp4 and the ue8m0 tensor */
__device__ __forceinline__ void down_issue(Ring& r, const DownUnit& U, int c, unsigned I) {
    const uint32_t sl = r.slot(r.issued, DOWN_S, 17 * I), b = r.bar_of(r.issued, DOWN_S);
    r.issued++;
    if ((threadIdx.x & 31u) == 0) {
        mbar_expect(b, 17 * I);
        bulk(sl, U.w + (size_t)c * 16 * I, 16 * I, b);
        bulk(sl + 16 * I, U.s + (size_t)c * I, I, b);
    }
}
struct DownCursor {
    unsigned gi;
    int c;
    DownUnit U;
};
__device__ __forceinline__ void down_top_up(Ring& r, DownCursor& q, const DownArgs& A) {
    while (r.issued - r.used < (unsigned)DOWN_S) {
        while (q.c >= q.U.c1) {
            if ((q.gi + 1) * A.n >= A.b) return;
            q.U = down_unit(A, ++q.gi);
            q.c = q.U.c0;
        }
        down_issue(r, q.U, q.c++, A.I);
    }
}
}  // namespace plow_mdv

/* Dynamic smem both entry points need (16 B aligned): E prefix ints, an mbarrier block and a bulk ring
 * per warp. */
__host__ __device__ constexpr size_t moe_decode_v41_smem_bytes(unsigned E, unsigned warps) {
    return plow_mdv::pfx_bytes(E) + (size_t)warps * (plow_mdv::BAR_BYTES + plow_mdv::RING_BYTES) + 128;
}
/* GLU split-item partials: floats of scratch and u32 counters, for nblk blocks of `warps` warps */
__host__ __device__ constexpr size_t moe_glu_decode_v41_scratch_floats(unsigned nblk, unsigned warps) {
    return (size_t)2 * nblk * warps * plow_mdv::PART;
}
__host__ __device__ constexpr size_t moe_glu_decode_v41_ctrs(unsigned nblk, unsigned warps) { return (size_t)nblk * warps; }

/* GLU: fu bf16 [rows][I] <- x = fq(xn) bf16 [T][H] over MoeAlignPf's rows. I % 32 == 0, H % 512 == 0.
 * An item (group, 32 features) cut by a share boundary is reduced through scratch and ctr (sizes
 * above); ctr must be zero before the first call and is left zero. */
__device__ void d_moe_glu_decode_v41(__nv_bfloat16* fu, const __nv_bfloat16* x, const unsigned long long* wtab,
                                     const unsigned long long* stab, const int* meta, const unsigned* row_token, const float* row_gate,
                                     unsigned I, unsigned H, unsigned E, unsigned act, float lim, float* scratch, unsigned* ctr,
                                     unsigned slice, unsigned nblk, unsigned char* smem) {
    using namespace plow_mdv;
    int* pfx = reinterpret_cast<int*>(smem);
    const unsigned ntile = I / 32, nch = H / 256;
    const unsigned n = group_prefix(meta + E, E, pfx) * ntile * nch;
    const unsigned nw = blockDim.x >> 5, gw = slice * nw + (threadIdx.x >> 5), tw = min(nblk * nw, n);
    const unsigned a = gw < tw ? share(gw, n, tw) : n, b = gw < tw ? share(gw + 1, n, tw) : n;
    const GluArgs A{wtab, stab, meta, meta + E, pfx, H, E, ntile, nch, a, b, (const char*)wtab[0] - TMAP_BYTES};
    const unsigned lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    Ring r = ring_open(smem, E, GLU_S);
    GluCursor q{a / nch, 0, {}};
    q.U.k1 = 0;
    if (a < b) {
        q.U = glu_unit(A, q.it);
        q.c = q.U.k0;
    }
    glu_top_up(r, q, A);
    for (unsigned it = a / nch; a < b && it * nch < b; it++) {
        const GluUnit U = glu_unit(A, it);
        const int p0 = U.row0 + (int)g;
        const __nv_bfloat16* xp = p0 < U.rend ? x + (size_t)row_token[p0] * H + t * 32 : nullptr;
        float acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
        uint32_t sw[4];
        glu_scales(sw, U, U.k0, H / 32);
#pragma unroll 1
        for (int c = U.k0; c < U.k1; c++) {
            uint32_t swn[4];
            if (c + 1 < U.k1) glu_scales(swn, U, c + 1, H / 32);
            const uint32_t sl = r.wait(GLU_S, GLU_SLOT);
            glu_chunk(sl, sw, acc, xp, c >> 1, (unsigned)c & 1u);
            __syncwarp();
            r.used++;
            glu_top_up(r, q, A);
#pragma unroll
            for (int i = 0; i < 4; i++) sw[i] = swn[i];
        }
        if (U.k0 > 0 || U.k1 < (int)nch) {
            /* contributors wf .. wl, in share order; partial slot 0 = a warp's first item, 1 = its last */
            const unsigned wf = owner(it * nch, n, tw), wl = owner(it * nch + nch - 1, n, tw);
            float* mine = scratch + ((size_t)gw * 2 + (it != a / nch)) * PART;
#pragma unroll
            for (int qq = 0; qq < 4; qq++)
#pragma unroll
                for (int k = 0; k < 4; k++) mine[(2 * t + (k & 1)) * 64 + qq * 16 + g + 8 * (k >> 1)] = acc[qq][k];
            __threadfence();
            __syncwarp();
            unsigned last = 0;
            if (lane == 0) last = atomicAdd(ctr + wl, 1u) == wl - wf;
            if (!__shfl_sync(0xffffffffu, last, 0)) continue;
            __threadfence();
#pragma unroll
            for (int qq = 0; qq < 4; qq++) acc[qq][0] = acc[qq][1] = acc[qq][2] = acc[qq][3] = 0.f;
            for (unsigned w = wf; w <= wl; w++) {
                const float* part = scratch + ((size_t)w * 2 + (it != share(w, n, tw) / nch)) * PART;
#pragma unroll
                for (int qq = 0; qq < 4; qq++)
#pragma unroll
                    for (int k = 0; k < 4; k++) acc[qq][k] += __ldcg(part + (2 * t + (k & 1)) * 64 + qq * 16 + g + 8 * (k >> 1));
            }
            if (lane == 0) ctr[wl] = 0;
        }
        /* GluEpiS: thread holds gate (tiles 0, 1) and up (tiles 2, 3) of features f0 + 16 qq + g + 8 h */
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
            const int p = U.row0 + 2 * (int)t + cc;
            const float rw = p < U.rend && row_gate ? row_gate[p] : 1.f;
            float v[2][2], mx = 0.f;
#pragma unroll
            for (int qq = 0; qq < 2; qq++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    float gv = rbf(acc[qq][h * 2 + cc]), uv = rbf(acc[2 + qq][h * 2 + cc]);
                    if (act == ACT_SWIGLU_CLAMP) {
                        uv = fminf(fmaxf(uv, -lim), lim);
                        gv = fminf(gv, lim);
                    }
                    v[qq][h] = rbf(rw * ((gv / (1.f + expf(-gv))) * uv));
                    mx = fmaxf(mx, fabsf(v[qq][h]));
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
            const float tq = fmaxf(mx, 1e-4f) * (1.0f / 448.0f);
            const uint32_t tbits = __float_as_uint(tq);
            const float sc = __uint_as_float((uint32_t)((int)(tbits >> 23) + ((tbits & 0x7fffffu) != 0u)) << 23);
            if (p >= U.rend) continue;
#pragma unroll
            for (int qq = 0; qq < 2; qq++)
#pragma unroll
                for (int h = 0; h < 2; h++)
                    fu[(size_t)p * I + U.f0 + 16 * qq + g + 8 * h] =
                        __float2bfloat16_rn(round_e4m3(fminf(fmaxf(v[qq][h] / sc, -448.f), 448.f)) * sc);
        }
    }
    ring_close(r, GLU_S);
}

/* DOWN: part f32 [T*k][H] at row_partidx (UNUSED skipped), times row_gate when bound. H % 32 == 0,
 * I % 64 == 0, I <= 576. */
__device__ void d_moe_down_decode_v41(float* part, const __nv_bfloat16* fu, const unsigned long long* wtab, const unsigned long long* stab,
                                      const int* meta, const unsigned* row_partidx, const float* row_gate, unsigned H, unsigned I,
                                      unsigned E, unsigned slice, unsigned nblk, unsigned char* smem) {
    using namespace plow_mdv;
    int* pfx = reinterpret_cast<int*>(smem);
    const unsigned nb = H / 32, n = group_prefix(meta + E, E, pfx) * nb;
    const unsigned nw = blockDim.x >> 5, gw = slice * nw + (threadIdx.x >> 5), tw = min(nblk * nw, n);
    const unsigned a = gw < tw ? share(gw, n, tw) : n, b = gw < tw ? share(gw + 1, n, tw) : n;
    const DownArgs A{wtab, stab, meta, meta + E, pfx, I, E, nb, a, b};
    const unsigned nsub = I / 64, lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    Ring r = ring_open(smem, E, DOWN_S);
    DownCursor q{a / nb, 0, {}};
    q.U.c1 = 0;
    if (a < b) {
        q.U = down_unit(A, q.gi);
        q.c = q.U.c0;
    }
    down_top_up(r, q, A);
    for (unsigned gi = a / nb; a < b && gi * nb < b; gi++) {
        const DownUnit U = down_unit(A, gi);
        const int p0 = U.row0 + (int)g;
        /* thread's token row p0, k = s * 64 + t * 16 .. + 16; rows past the group read row0 (their
         * output columns are never stored) */
        const uint4* tp = reinterpret_cast<const uint4*>(fu + (size_t)(p0 < U.rend ? p0 : U.row0) * I + t * 16);
        for (int c = U.c0; c < U.c1; c++) {
            float acc[2][4];
#pragma unroll
            for (int i = 0; i < 2; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
            const uint32_t sl = r.wait(DOWN_S, 17 * I);
            /* rolled over the I / 64 sub-blocks: a compact body (cold code is paid every step) */
            uint4 n0 = tp[0], n1 = tp[1];
#pragma unroll 1
            for (unsigned s = 0; s < nsub; s++) {
                const uint32_t tb[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
                if (s + 1 < nsub) {
                    n0 = tp[(s + 1) * 8];
                    n1 = tp[(s + 1) * 8 + 1];
                }
                uint32_t bp[8];
                pair_tok(&tb[0], &bp[0]);
                pair_tok(&tb[4], &bp[4]);
#pragma unroll
                for (int qq = 0; qq < 2; qq++) {
                    uint32_t ap[2][8];
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const unsigned R = qq * 16 + h * 8 + g;
                        uint32_t a2[2];
                        lds64(sl + R * (I / 2) + s * 32 + t * 8, a2);
                        const __nv_bfloat162 sc = pow2x2(lds8(sl + 16 * I + R * (I / 32) + 2 * s + (t >> 1)));
                        dec_word(a2[0], sc, &ap[h][0]);
                        dec_word(a2[1], sc, &ap[h][4]);
                    }
#pragma unroll
                    for (int st = 0; st < 4; st++) {
                        const uint32_t aa[4] = {ap[0][2 * st], ap[1][2 * st], ap[0][2 * st + 1], ap[1][2 * st + 1]};
                        mma(acc[qq], aa, bp[2 * st], bp[2 * st + 1]);
                    }
                }
            }
            __syncwarp();
            r.used++;
            down_top_up(r, q, A);
#pragma unroll
            for (int cc = 0; cc < 2; cc++) {
                const int p = U.row0 + 2 * (int)t + cc;
                if (p >= U.rend) continue;
                const unsigned pidx = row_partidx[p];
                if (pidx == UNUSED) continue;
                const float rw = row_gate ? row_gate[p] : 1.f;
                float* dst = part + (size_t)pidx * H + c * 32;
#pragma unroll
                for (int qq = 0; qq < 2; qq++)
#pragma unroll
                    for (int h = 0; h < 2; h++) dst[16 * qq + g + 8 * h] = rw * rbf(acc[qq][h * 2 + cc]);
            }
        }
    }
    ring_close(r, DOWN_S);
}
