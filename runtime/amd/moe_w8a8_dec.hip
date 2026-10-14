/* PLOW_MOE_W8A8_DEC (gfx950): the decode-rung W8A8 grouped MoE pair -- ops 85 (GLU, i7 = 6, bf16
 * rows quantized 1x128 in staging) and 86 (DOWN, i7 = 2, f32 part) -- as two standalone kernels
 * launched back to back in one decode segment.
 *
 * The interpreter arms (moe_pf_w8a8.h) run one 8-wave workgroup per CU over 64-row x 256-column
 * tiles, so a decode GLU is one CU walking 32 dependent k-tiles of 2 MB of weights. Here a
 * workgroup is 4 waves and owns ONE work item: one 32-row half of an align tile x GLU 128 fused
 * gate|up columns (one 1x128 output group) or DOWN 256 columns; several workgroups share a CU
 * and the B fragments of the next DEC_PD-1 k-tiles stay in flight in registers. Items of the
 * second halves come last in the grid and exit at once when the half has no live row (their
 * outputs are never read).
 *
 * Per output element: the same fragment map, the same MFMA chain per 128-K block
 * (mfma_scale 32x32x64 twice), the same promotion fmaf(t, as * ws, acc) in ascending K, the same
 * SwiGLU and 1x128 quant expressions -- BIT-IDENTICAL to d_moe_pf_w8a8<true, true, false> /
 * d_moe_pf_w8a8<false>. */
#ifndef MDEC_NO_INCLUDE
#define PLOW_WG_WAVES 8
#define PLOW_MOE_PREFILL 1
#include "amd_common.h"
#include "op_moe.h"
#endif

extern "C" __device__ unsigned plow_moe_w8a8_dec_abi_1 = 1;

#ifndef DEC_PD
#define DEC_PD 2 /* B register stages (k-tiles in flight per wave) */
#endif
static_assert(DEC_PD >= 1 && DEC_PD <= 8 && (DEC_PD & (DEC_PD - 1)) == 0,
              "DEC_PD: power of two <= 8 (the host admits hidden % 1024 == 0)");
#ifndef DEC_OCC
#define DEC_OCC 2 /* min workgroups per CU (launch bounds) */
#endif
/* DEC_BLDS: B k-tiles load as whole 128-byte row segments (8 lanes per row, 32 rows per
 * instruction) into registers and are staged through an LDS double buffer, from which the MFMA
 * fragments are read; 0 = fragments straight from global memory (32 bytes of 32 rows per
 * instruction). Same fragment values either way. */
#ifndef DEC_BLDS
#define DEC_BLDS 0
#endif

namespace mdec {
constexpr unsigned BM = 32, A_BYTES = BM * 128u; /* a work item's rows: one 32-row block */
constexpr unsigned OFF_S = 2u * A_BYTES;      /* f32 [2][32] A row scales */
constexpr unsigned OFF_PIDX = OFF_S + 256u;   /* DOWN u32 [32] */
constexpr unsigned OFF_GATE = OFF_PIDX + 128u;
constexpr unsigned OFF_RED = OFF_GATE + 128u; /* GLU f32 [4][32] */
constexpr unsigned OFF_FS = OFF_RED + 512u;   /* GLU f32 [32] */
constexpr unsigned LDS = OFF_FS + 128u;
} // namespace mdec

#if PLOW_CDNA4
__device__ __forceinline__ unsigned mdec_swz(unsigned r, unsigned c) {
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

/* Global (not flat) load: a flat load may hit LDS, so the waitcnt pass drains vmcnt(0) on it
 * and the B pipeline with it. */
__device__ __forceinline__ float mdec_ldf(const float* p) {
    return *(const PLOW_GLOB float*)(const PLOW_GLOB void*)p;
}

__device__ __forceinline__ fp8v32 mdec_frag(const unsigned char* p) {
    const fp8v16 lo = ld_glob_fp8v16(p), hi = ld_glob_fp8v16(p + 16);
    fp8v32 v;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        v[j] = (int)lo[j];
        v[4 + j] = (int)hi[j];
    }
    return v;
}

/* FUSED (plow_moe_w8a8_dec_pair): GLU and DOWN items in one grid, GLU items first. ctr[2 key]
 * counts the GLU items done for a 32-row half (key = 2 tile + half); a DOWN item prefetches its
 * weights, waits for all K/128 of them, and the last DOWN item of the half re-zeroes both words.
 * Dispatch is in order per XCD, so every GLU item a DOWN item waits on is resident or done. */
template <bool GLU, bool FUSED = false>
__device__ __forceinline__ void mdec_body(void* __restrict__ Cout, float* __restrict__ fs_out,
                                          const unsigned char* __restrict__ Aq,
                                          const float* __restrict__ As,
                                          const unsigned long long* __restrict__ wtab,
                                          const unsigned long long* __restrict__ stab,
                                          const int* __restrict__ meta,
                                          const unsigned* __restrict__ rowmap,
                                          const float* __restrict__ row_gate, unsigned N,
                                          unsigned K, unsigned n_exp, unsigned act, float beta,
                                          float lbeta, unsigned item, unsigned* ctr) {
    using namespace mdec;
    __shared__ __align__(16) unsigned char lds[LDS];
    constexpr unsigned NB = GLU ? 128u : 256u;
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    const unsigned l32 = lane & 31u, hi = lane >> 5;
    const int* tilep = meta + 2 * (int)n_exp;
    const unsigned tn = (N + NB - 1u) / NB;
    const unsigned per = (unsigned)tilep[n_exp] * tn; /* items per 32-row half */
    const unsigned rb = item >= per ? 1u : 0u;
    item -= rb * per;
    if (item >= per) return;
    const unsigned mt = item / tn, nt = item % tn;
    const unsigned e = mpf_expert_of_tile(tilep, mt, n_exp);
    const unsigned toff = (mt - (unsigned)tilep[e]) * 64u + 32u * rb;
    if ((int)toff >= meta[n_exp + e]) return; /* no live row in this half */
    const unsigned rowbase = (unsigned)meta[e] + toff;
    const unsigned n0 = nt * NB;
    const unsigned long long wb0 = wtab[(size_t)e * 3 + (GLU ? 0 : 2)];
    if (wb0 == 0ull) return;
    const unsigned char* W0 = (const unsigned char*)(size_t)wb0;
    const unsigned char* W1 = GLU ? (const unsigned char*)(size_t)wtab[(size_t)e * 3 + 1] : W0;
    const float* S0 = (const float*)(size_t)stab[(size_t)e * 3 + (GLU ? 0 : 2)];
    const float* S1 = GLU ? (const float*)(size_t)stab[(size_t)e * 3 + 1] : S0;
    const unsigned KB = K >> 7;

    const unsigned cb0 = GLU ? n0 + 32u * wave : n0 + 64u * wave;
    const unsigned cb1 = GLU ? cb0 : cb0 + 32u;
    const unsigned brow0 = cb0 + l32, brow1 = cb1 + l32;
    const unsigned char* const bp0 = W0 + (size_t)min(brow0, N - 1u) * K + hi * 32u;
    const unsigned char* const bp1 = W1 + (size_t)min(brow1, N - 1u) * K + hi * 32u;
    const float* const sp0 = S0 + (size_t)(min(cb0, N - 1u) >> 7) * KB;
    const float* const sp1 = S1 + (size_t)(min(cb1, N - 1u) >> 7) * KB;

    float* const lds_s = (float*)(lds + OFF_S);
    unsigned* const lds_pidx = (unsigned*)(lds + OFF_PIDX);
    float* const lds_gate = (float*)(lds + OFF_GATE);
    float* const lds_red = (float*)(lds + OFF_RED);
    float* const lds_fs = (float*)(lds + OFF_FS);

    /* A staging: thread -> (row sr, 16-byte fp8 chunk sc) of the 32 x 128 k-tile. */
    const unsigned sr = tid >> 3, sc = tid & 7u;
    const unsigned src = GLU ? rowmap[rowbase + sr] : rowbase + sr;
    const bool alive = !GLU || src != PLOW_EXPERT_UNUSED;
    const unsigned srcc = alive ? src : 0u;
    const unsigned char* const ap =
        GLU ? Aq + (size_t)srcc * 2u * K + sc * 32u : Aq + (size_t)srcc * K + sc * 16u;
    const float* const asp = GLU ? As : As + (size_t)srcc * KB;
    /* Branch-free software pipeline (conditional loads would make the waitcnt pass drain
     * vmcnt(0) every k-tile): B k-tile kb lives in register slot kb % PD from PD-1 k-tiles ahead,
     * A k-tile kb in slot kb % PD from PD k-tiles ahead until its LDS stage. Tail loads re-read the
     * last k-tile (clamped) and are never consumed. KB % PD == 0. */
    constexpr unsigned PD = GLU ? DEC_PD : 2u;
    fp8v16 ar[PD], ar2[PD];
    float sreg[PD];
#define MDEC_LOAD_A(kb_, s)                                      \
    do {                                                         \
        const unsigned k_ = min((unsigned)(kb_), KB - 1u);       \
        if constexpr (GLU) {                                     \
            ar[s] = ld_glob_fp8v16(ap + k_ * 256u);              \
            ar2[s] = ld_glob_fp8v16(ap + k_ * 256u + 16u);       \
        } else {                                                 \
            ar[s] = ld_glob_fp8v16(ap + k_ * 128u);              \
            sreg[s] = mdec_ldf(asp + k_);                                \
        }                                                        \
    } while (0)
    auto stage = [&](unsigned b, unsigned s) {
        unsigned char* const tb = lds + b * A_BYTES;
        float* const sb = lds_s + b * BM;
        if constexpr (GLU) {
            const fp8v16 keep = (fp8v16)(alive ? ~0u : 0u);
            const fp8v16 a = ar[s] & keep, a2 = ar2[s] & keep;
            float v[16];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                v[2 * q] = __uint_as_float(a[q] << 16);
                v[2 * q + 1] = __uint_as_float(a[q] & 0xffff0000u);
                v[8 + 2 * q] = __uint_as_float(a2[q] << 16);
                v[8 + 2 * q + 1] = __uint_as_float(a2[q] & 0xffff0000u);
            }
            float m = 0.0f;
#pragma unroll
            for (int q = 0; q < 16; q++) m = fmaxf(m, fabsf(v[q]));
            m = fmaxf(m, __shfl_xor(m, 1));
            m = fmaxf(m, __shfl_xor(m, 2));
            m = fmaxf(m, __shfl_xor(m, 4));
            const float s = fmaxf(m, 1e-10f) * (1.0f / 448.0f);
            const float inv = 1.0f / s;
            fp8v16 hq;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const unsigned t0 =
                    __builtin_amdgcn_cvt_pk_fp8_f32(v[4 * q] * inv, v[4 * q + 1] * inv, 0, false);
                hq[q] = __builtin_amdgcn_cvt_pk_fp8_f32(v[4 * q + 2] * inv, v[4 * q + 3] * inv,
                                                        t0, true);
            }
            *(fp8v16*)(tb + mdec_swz(sr, sc)) = hq;
            if (sc == 0u) sb[sr] = alive ? s : 0.0f;
        } else {
            *(fp8v16*)(tb + mdec_swz(sr, sc)) = ar[s];
            if (sc == 0u) sb[sr] = sreg[s];
        }
    };

    constexpr bool BL = DEC_BLDS != 0;
    float wq[PD][2];
#if DEC_BLDS
    /* B tile rows: GLU 0-127 gate / 128-255 up columns n0.., DOWN 0-255 columns n0..; thread ->
     * (rows sr + 32 i, chunk sc). B k-tile kb lives in register slot kb % PD from PD k-tiles
     * ahead until its LDS stage (like A). */
    __shared__ __align__(16) unsigned char ldsb[2u * 256u * 128u];
    const unsigned char* bpl[8];
#pragma unroll
    for (unsigned i = 0; i < 8; i++) {
        const unsigned r = sr + 32u * i;
        const unsigned col = GLU ? n0 + (r & 127u) : n0 + r;
        const unsigned char* const Wb = (GLU && r >= 128u) ? W1 : W0;
        bpl[i] = Wb + (size_t)min(col, N - 1u) * K + sc * 16u;
    }
    fp8v16 breg[PD][8];
    float wnx[2];
#define MDEC_LOAD_B(kb_, s)                                    \
    do {                                                       \
        const unsigned k_ = min((unsigned)(kb_), KB - 1u);     \
        _Pragma("unroll") for (unsigned i_ = 0; i_ < 8; i_++)  \
            breg[s][i_] = ld_glob_fp8v16(bpl[i_] + k_ * 128u); \
        wq[s][0] = mdec_ldf(sp0 + k_);                         \
        wq[s][1] = mdec_ldf(sp1 + k_);                         \
    } while (0)
    auto stage_b = [&](unsigned b, unsigned s) {
#pragma unroll
        for (unsigned i = 0; i < 8; i++)
            *(fp8v16*)(ldsb + b * 32768u + mdec_swz(sr + 32u * i, sc)) = breg[s][i];
        wnx[0] = wq[s][0];
        wnx[1] = wq[s][1];
    };
    const unsigned brl0 = GLU ? 32u * wave + l32 : 64u * wave + l32;
    const unsigned brl1 = GLU ? 128u + 32u * wave + l32 : 64u * wave + 32u + l32;
#else
    fp8v32 bq[PD][2][2];
#define MDEC_LOAD_B(kb_, s)                                  \
    do {                                                     \
        const unsigned k_ = min((unsigned)(kb_), KB - 1u);   \
        bq[s][0][0] = mdec_frag(bp0 + k_ * 128u);            \
        bq[s][0][1] = mdec_frag(bp0 + k_ * 128u + 64u);      \
        bq[s][1][0] = mdec_frag(bp1 + k_ * 128u);            \
        bq[s][1][1] = mdec_frag(bp1 + k_ * 128u + 64u);      \
        wq[s][0] = mdec_ldf(sp0 + k_);                       \
        wq[s][1] = mdec_ldf(sp1 + k_);                       \
    } while (0)
#endif
    constexpr unsigned BPRE = BL ? PD : PD - 1u; /* B k-tiles loaded before the loop */

    const unsigned key = 2u * mt + rb;
    if constexpr (FUSED && !GLU) {
#pragma unroll
        for (unsigned s = 0; s < BPRE; s++) MDEC_LOAD_B(s, s);
        if (tid == 0u)
            while (__hip_atomic_load(ctr + 2u * key, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <
                   KB)
                __builtin_amdgcn_s_sleep(1);
        __syncthreads();
#pragma unroll
        for (unsigned s = 0; s < PD; s++) MDEC_LOAD_A(s, s);
    } else {
#pragma unroll
        for (unsigned s = 0; s < PD; s++) MDEC_LOAD_A(s, s);
#pragma unroll
        for (unsigned s = 0; s < BPRE; s++) MDEC_LOAD_B(s, s);
    }
    stage(0u, 0u);
    MDEC_LOAD_A(PD, 0);
#if DEC_BLDS
    stage_b(0u, 0u);
    MDEC_LOAD_B(PD, 0);
#endif
    if (!GLU && tid < BM) {
        lds_pidx[tid] = rowmap[rowbase + tid];
        lds_gate[tid] = row_gate[rowbase + tid];
    }
    __syncthreads();

    f32x16 accf[2];
    accf[0] = (f32x16)(0.0f);
    accf[1] = (f32x16)(0.0f);

    for (unsigned kb0 = 0; kb0 < KB; kb0 += PD) {
#pragma unroll
        for (unsigned s = 0; s < PD; s++) {
            const unsigned kb = kb0 + s;
            const unsigned buf = (PD % 2u == 0u) ? (s & 1u) : (kb & 1u);
            if constexpr (!BL) MDEC_LOAD_B(kb + PD - 1u, (s + PD - 1u) % PD);
            const unsigned char* abuf = lds + buf * A_BYTES;
            const float* sbuf = lds_s + buf * BM;
            fp8v32 af[2];
#pragma unroll
            for (int t = 0; t < 2; t++) {
                const unsigned c = (unsigned)t * 4u + hi * 2u;
                const fp8v16 lo = *(const fp8v16*)(abuf + mdec_swz(l32, c));
                const fp8v16 up = *(const fp8v16*)(abuf + mdec_swz(l32, c + 1u));
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    af[t][q] = (int)lo[q];
                    af[t][4 + q] = (int)up[q];
                }
            }
            float as[16];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float4 v = *(const float4*)(sbuf + 8u * (unsigned)q + 4u * hi);
                as[4 * q + 0] = v.x;
                as[4 * q + 1] = v.y;
                as[4 * q + 2] = v.z;
                as[4 * q + 3] = v.w;
            }
#if DEC_BLDS
            const unsigned char* bbuf = ldsb + buf * 32768u;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned br = j ? brl1 : brl0;
                fp8v32 bf[2];
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const unsigned c = (unsigned)t * 4u + hi * 2u;
                    const fp8v16 lo = *(const fp8v16*)(bbuf + mdec_swz(br, c));
                    const fp8v16 up = *(const fp8v16*)(bbuf + mdec_swz(br, c + 1u));
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        bf[t][q] = (int)lo[q];
                        bf[t][4 + q] = (int)up[q];
                    }
                }
                f32x16 t = plow_mfma_fp8_32x32(af[0], bf[0], (f32x16)(0.0f));
                t = plow_mfma_fp8_32x32(af[1], bf[1], t);
#pragma unroll
                for (int el = 0; el < 16; el++)
                    accf[j][el] = fmaf(t[el], as[el] * wnx[j], accf[j][el]);
            }
#else
#pragma unroll
            for (int j = 0; j < 2; j++) {
                f32x16 t = plow_mfma_fp8_32x32(af[0], bq[s][j][0], (f32x16)(0.0f));
                t = plow_mfma_fp8_32x32(af[1], bq[s][j][1], t);
#pragma unroll
                for (int el = 0; el < 16; el++)
                    accf[j][el] = fmaf(t[el], as[el] * wq[s][j], accf[j][el]);
            }
#endif
            /* The last k-tile stages a clamped re-read into the idle buffer: never consumed. */
            stage(buf ^ 1u, (s + 1u) % PD);
            MDEC_LOAD_A(kb + 1u + PD, (s + 1u) % PD);
#if DEC_BLDS
            stage_b(buf ^ 1u, (s + 1u) % PD);
            MDEC_LOAD_B(kb + 1u + PD, (s + 1u) % PD);
#endif
            __syncthreads();
        }
    }
#undef MDEC_LOAD_B
#undef MDEC_LOAD_A
    if constexpr (FUSED && !GLU) {
        if (tid == 0u &&
            __hip_atomic_fetch_add(ctr + 2u * key + 1u, 1u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT) == tn - 1u) {
            __hip_atomic_store(ctr + 2u * key, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(ctr + 2u * key + 1u, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    if constexpr (GLU) {
#pragma unroll
        for (int el = 0; el < 16; el++) {
            accf[0][el] = moe_glu(accf[0][el], accf[1][el], act, beta, lbeta);
            float m = fabsf(accf[0][el]);
            m = fmaxf(m, __shfl_xor(m, 1));
            m = fmaxf(m, __shfl_xor(m, 2));
            m = fmaxf(m, __shfl_xor(m, 4));
            m = fmaxf(m, __shfl_xor(m, 8));
            m = fmaxf(m, __shfl_xor(m, 16));
            if (l32 == 0u) lds_red[wave * BM + mfma_acc_m(lane, el)] = m;
        }
        __syncthreads();
        const unsigned groups = N >> 7;
        if (tid < BM) {
            float amax = 1e-10f; /* AITER floors amax, not the scale */
#pragma unroll
            for (int w = 0; w < 4; w++) amax = fmaxf(amax, lds_red[(unsigned)w * BM + tid]);
            const float s = amax * (1.0f / 448.0f);
            lds_fs[tid] = s;
            st_act<float>(as_glob(fs_out) + (size_t)(rowbase + tid) * groups + (n0 >> 7), s);
        }
        __syncthreads();
        unsigned char* const fq = (unsigned char*)Cout;
        const unsigned col = n0 + 32u * wave + l32;
#pragma unroll
        for (int el = 0; el < 16; el++) {
            const unsigned r = mfma_acc_m(lane, el);
            const float inv = 1.0f / lds_fs[r];
            const float xv = accf[0][el] * inv;
            const unsigned q = __builtin_amdgcn_cvt_pk_fp8_f32(xv, 0.0f, 0u, false);
            st_act1_u8(as_glob(fq) + (size_t)(rowbase + r) * N + col, (unsigned char)q);
        }
        if constexpr (FUSED) {
            __threadfence();
            __syncthreads();
            if (tid == 0u)
                __hip_atomic_fetch_add(ctr + 2u * key, 1u, __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_AGENT);
        }
    } else {
#pragma unroll
        for (int el = 0; el < 16; el++) {
            const unsigned r = mfma_acc_m(lane, el);
            const unsigned pidx = lds_pidx[r];
            if (pidx == PLOW_EXPERT_UNUSED) continue;
            const float gv = lds_gate[r];
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const unsigned bcol = j ? brow1 : brow0;
                if (bcol >= N) continue;
                __builtin_nontemporal_store(
                    gv * accf[j][el], (PLOW_GLOB float*)&((float*)Cout)[(size_t)pidx * N + bcol]);
            }
        }
    }
}
#endif

/* ABI = MoeFp8GluArgs / MoeFp8DownArgs (exec/amd.rs). GLU x is the bf16 [T,H] rows (i7 = 6),
 * x_scale / x_pitch unused. Grid: 2 * tiles_max * N/128 (GLU), 2 * tiles_max * N/256 (DOWN). */
extern "C" __global__ __launch_bounds__(256, DEC_OCC) void plow_moe_w8a8_dec_glu(
    unsigned char* __restrict__ out, const unsigned short* __restrict__ x,
    const unsigned long long* __restrict__ wtab, const unsigned long long* __restrict__ stab,
    const int* __restrict__ meta, const unsigned* __restrict__ row_token,
    const float* __restrict__ x_scale, float* __restrict__ out_scale, unsigned inter,
    unsigned hidden, unsigned experts, unsigned act, float beta, float lbeta, unsigned x_pitch,
    unsigned reserved) {
#if PLOW_CDNA4
    (void)x_scale;
    (void)x_pitch;
    (void)reserved;
    if (inter % 128u != 0u || hidden % (128u * DEC_PD) != 0u) return;
    mdec_body<true>(out, out_scale, (const unsigned char*)x, nullptr, wtab, stab, meta, row_token,
                    nullptr, inter, hidden, experts, act, beta, lbeta, blockIdx.x, nullptr);
#endif
}

extern "C" __global__ __launch_bounds__(256, DEC_OCC) void plow_moe_w8a8_dec_down(
    float* __restrict__ part, const unsigned char* __restrict__ fu,
    const unsigned long long* __restrict__ wtab, const unsigned long long* __restrict__ stab,
    const int* __restrict__ meta, const unsigned* __restrict__ row_partidx,
    const float* __restrict__ row_gate, const float* __restrict__ fu_scale, unsigned hidden,
    unsigned inter, unsigned experts, unsigned reserved) {
#if PLOW_CDNA4
    (void)reserved;
    if (hidden % 128u != 0u || inter % 256u != 0u) return;
    mdec_body<false>(part, nullptr, fu, fu_scale, wtab, stab, meta, row_partidx, row_gate, hidden,
                     inter, experts, 0u, 0.0f, 0.0f, blockIdx.x, nullptr);
#endif
}

/* One launch for the pair (exec/amd.rs MoeW8a8DecPairArgs): blocks [0, grid_glu) are the GLU
 * grid, the rest the DOWN grid; ctr is a zeroed u32 [2 * 2 * tiles] the kernel leaves zeroed. */
struct MdecGluArgs {
    unsigned char* out;
    const unsigned short* x;
    const unsigned long long* wtab;
    const unsigned long long* stab;
    const int* meta;
    const unsigned* row_token;
    const float* x_scale;
    float* out_scale;
    unsigned inter, hidden, experts, act;
    float beta, lbeta;
    unsigned x_pitch, reserved;
};
struct MdecDownArgs {
    float* part;
    const unsigned char* fu;
    const unsigned long long* wtab;
    const unsigned long long* stab;
    const int* meta;
    const unsigned* row_partidx;
    const float* row_gate;
    const float* fu_scale;
    unsigned hidden, inter, experts, reserved;
};
struct MdecPairArgs {
    MdecGluArgs g;
    MdecDownArgs d;
    unsigned* ctr;
    unsigned grid_glu, reserved;
};
static_assert(sizeof(MdecGluArgs) == 96 && sizeof(MdecDownArgs) == 80 && sizeof(MdecPairArgs) == 192,
              "MdecPairArgs ABI");

extern "C" __global__ __launch_bounds__(256, DEC_OCC) void plow_moe_w8a8_dec_pair(MdecPairArgs a) {
#if PLOW_CDNA4
    if (a.g.inter % 256u != 0u || a.g.hidden % (128u * DEC_PD) != 0u || a.d.inter != a.g.inter)
        return;
    if (blockIdx.x < a.grid_glu)
        mdec_body<true, true>(a.g.out, a.g.out_scale, (const unsigned char*)a.g.x, nullptr,
                              a.g.wtab, a.g.stab, a.g.meta, a.g.row_token, nullptr, a.g.inter,
                              a.g.hidden, a.g.experts, a.g.act, a.g.beta, a.g.lbeta, blockIdx.x,
                              a.ctr);
    else
        mdec_body<false, true>(a.d.part, nullptr, a.d.fu, a.d.fu_scale, a.d.wtab, a.d.stab,
                               a.d.meta, a.d.row_partidx, a.d.row_gate, a.d.hidden, a.d.inter,
                               a.d.experts, 0u, 0.0f, 0.0f, blockIdx.x - a.grid_glu, a.ctr);
#endif
}
