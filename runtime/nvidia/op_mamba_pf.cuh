/* op_mamba_pf.cuh — Mamba-2 chunked-prefill bodies on tensor cores (ops 240 prefill, 243-245).
 *
 * Same operands and scratch as op_mamba.cuh; the prefill object of a Mamba packet runs these.
 * Arithmetic is vLLM's ssd_* contract: bf16 operands into mma.sync m16n8k16 with f32
 * accumulation, rounded to bf16 exactly where vLLM rounds (decay-scaled B in the chunk state,
 * the entering state and the decay-masked CB in the chunk scan); f32 dt, cumsum, recurrence.
 * 256 threads (8 warps); smem tiles use 272 B / 144 B row strides (odd multiples of 16 B) so
 * every ldmatrix phase is conflict-free.
 */
#pragma once
#include "op_mamba.cuh"

#define MPF_SN 136u /* bf16 row stride of 128-wide tiles */
#define MPF_SP 72u  /* bf16 row stride of 64-wide tiles */
/* ChunkOut: C (or B) [128][136] + E [64][136] (or x [128][72]) bf16 + cum/dt f32. */
#define MAMBA_PF_ARENA_BYTES (2u * (128u * MPF_SN + 128u * MPF_SP) + 8u * MAMBA_L + 1024u)

__device__ __forceinline__ void mpf_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mpf_ldsm(unsigned (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void mpf_ldsm_t(unsigned (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ unsigned mpf_bf2(float a, float b) {
    __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&r);
}
/* 8 bf16 of a global row (16 B aligned) or zeros. */
__device__ __forceinline__ uint4 mpf_ld8(const __nv_bfloat16* p, bool ok) {
    return ok ? *reinterpret_cast<const uint4*>(p) : make_uint4(0u, 0u, 0u, 0u);
}

/* Packed prefill (`req` != nullptr, host-bound at t7): req[0] = n, request r = {row0, len, slot,
 * kv end} at req[1 + 4r], rows contiguous from row 0. Request r's rows form ceil(len / L) virtual
 * chunks at the same offsets from its first row as a serial launch's chunks, so no chunk crosses
 * a request and each sees the serial chunk's rows. Scratch: cstate[v], dtv/cum[h][v * L + t]
 * with h-stride T + L n. The launch's tensor table is unshifted: state lives at `slot`. */
static __device__ __forceinline__ unsigned mpf_len(const int* req, unsigned r) {
    const int l = req[2 + 4 * r];
    return l > 0 ? (unsigned)l : 0u;
}
/* Rows the requests cover (pad rows after them are zero). */
static __device__ __forceinline__ unsigned mpf_real(const int* req) {
    const unsigned n = (unsigned)req[0];
    return n ? (unsigned)req[1 + 4 * (n - 1)] + mpf_len(req, n - 1) : 0u;
}
static __device__ __forceinline__ unsigned mpf_nchunk(const int* req) {
    unsigned v = 0;
    for (unsigned r = 0; r < (unsigned)req[0]; r++) v += (mpf_len(req, r) + MAMBA_L - 1) / MAMBA_L;
    return v;
}
/* Virtual chunk `c`: first row, real rows and slot. Serial: chunk c of the R real rows, slot 0. */
static __device__ __forceinline__ void mpf_chunk(const int* req, unsigned c, unsigned rows, unsigned& t0,
                                                 unsigned& clen, unsigned& slot) {
    if (!req) {
        t0 = c * MAMBA_L;
        clen = rows - t0 < MAMBA_L ? rows - t0 : MAMBA_L;
        slot = 0;
        return;
    }
    for (unsigned r = 0;; r++) {
        const unsigned len = mpf_len(req, r), nc = (len + MAMBA_L - 1) / MAMBA_L;
        if (c < nc) {
            t0 = (unsigned)req[1 + 4 * r] + c * MAMBA_L;
            clen = len - c * MAMBA_L < MAMBA_L ? len - c * MAMBA_L : MAMBA_L;
            slot = (unsigned)req[3 + 4 * r];
            return;
        }
        c -= nc;
    }
}
/* Zero rows [from, tokens) of a `width`-element bf16 activation (width % 8 == 0). */
static __device__ __forceinline__ void mpf_zero_rows(__nv_bfloat16* p, unsigned width, unsigned from,
                                                     unsigned tokens, unsigned slice, unsigned nblk) {
    const size_t begin = (size_t)from * width / 8u, end = (size_t)tokens * width / 8u;
    for (size_t i = begin + (size_t)slice * blockDim.x + threadIdx.x; i < end; i += (size_t)nblk * blockDim.x)
        reinterpret_cast<uint4*>(p)[i] = make_uint4(0u, 0u, 0u, 0u);
}

/* MAMBA2_CONV prefill. Head items (one per channel and request) as op_mamba.cuh: rows 0..W-2 of
 * the request from its slot history, then the history rewrite. Body: thread per (8 rows,
 * 8 channels) of one request from W-1+7 uint4 row loads. `channels` % 8 == 0, W == 4. */
template <bool PK = false>
static __device__ __noinline__ void d_mamba2_conv_pf(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ history, unsigned channels, unsigned width, unsigned rows,
    unsigned tokens, unsigned source, unsigned slice, unsigned nblk, const int* __restrict__ req = nullptr) {
    if (!PK) req = nullptr; /* the serial instantiation folds every packed branch away */
    if (width != 4u || channels % 8u != 0u || tokens < width - 1u || rows > tokens ||
        (req && mpf_real(req) > tokens)) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned is = source ? (source & 0xFFFFu) : channels;
    in += source >> 16;
    const unsigned hw = 3u, c8n = channels / 8u, nreq = req ? (unsigned)req[0] : 1u;
    unsigned rb = (tokens + 7u) / 8u;
    if (req) {
        rb = 0;
        for (unsigned r = 0; r < nreq; r++) rb += (mpf_len(req, r) + 7u) / 8u;
    }
    const unsigned head = channels * nreq, body = rb * c8n;
    for (unsigned idx = slice * blockDim.x + threadIdx.x; idx < head + body; idx += nblk * blockDim.x) {
        if (idx < head) {
            const unsigned r = idx / channels, c = idx - r * channels;
            const unsigned row0 = req ? (unsigned)req[1 + 4 * r] : 0u, len = req ? mpf_len(req, r) : rows;
            if (req && len == 0u) continue;
            __nv_bfloat16* h = history + ((size_t)(req ? (unsigned)req[3 + 4 * r] : 0u) * channels + c) * hw;
            const __nv_bfloat16* x = in + (size_t)row0 * is;
            __nv_bfloat16* o = out + (size_t)row0 * channels;
            const __nv_bfloat16* w = weight + (size_t)c * width;
            const float b = bias ? mamba_bf(bias, c) : 0.f;
            float old[3];
            for (unsigned j = 0; j < hw; j++) old[j] = __bfloat162float(h[j]);
            for (unsigned t = 0; t < hw; t++) {
                if (req && t >= len) break; /* the next request's rows */
                float sum = b;
                for (unsigned k = 0; k < width; k++) {
                    const int pos = (int)t - (int)hw + (int)k;
                    const float v = pos < 0 ? old[pos + (int)hw]
                                  : ((unsigned)pos < len ? mamba_bf(x, (size_t)pos * is + c) : 0.f);
                    sum = fmaf(v, mamba_bf(w, k), sum);
                }
                o[(size_t)t * channels + c] = __float2bfloat16(t < len ? mamba_silu(sum) : 0.f);
            }
            for (unsigned j = 0; j < hw; j++) {
                const int src = (int)len - (int)hw + (int)j;
                const float v = src < 0 ? old[src + (int)hw] : mamba_bf(x, (size_t)src * is + c);
                h[j] = __float2bfloat16(v);
            }
            continue;
        }
        unsigned e = idx - head, rows_r = rows;
        const __nv_bfloat16* xin = in;
        __nv_bfloat16* xout = out;
        if (req) { /* (request, 8-row group of it, 8 channels) */
            for (unsigned r = 0;; r++) {
                const unsigned len = mpf_len(req, r), n = (len + 7u) / 8u * c8n;
                if (e < n) {
                    rows_r = len;
                    xin = in + (size_t)req[1 + 4 * r] * is;
                    xout = out + (size_t)req[1 + 4 * r] * channels;
                    break;
                }
                e -= n;
            }
        }
        const unsigned r8 = e / c8n, c0 = (e - r8 * c8n) * 8u, t0 = r8 * 8u;
        float wk[4][8], bv[8];
        {
            const uint4 b4 = bias ? *reinterpret_cast<const uint4*>(bias + c0) : make_uint4(0u, 0u, 0u, 0u);
            const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&b4);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                bv[j] = __bfloat162float(pb[j]);
                const uint2 w2 = *reinterpret_cast<const uint2*>(weight + (size_t)(c0 + j) * 4u);
                const __nv_bfloat16* pw = reinterpret_cast<const __nv_bfloat16*>(&w2);
#pragma unroll
                for (int k = 0; k < 4; k++) wk[k][j] = __bfloat162float(pw[k]);
            }
        }
        float win[3][8];
#pragma unroll
        for (int k = 0; k < 3; k++) {
            const int pos = (int)t0 - 3 + k;
            const uint4 v = mpf_ld8(xin + (size_t)(pos < 0 ? 0 : pos) * is + c0, pos >= 0 && (unsigned)pos < rows_r);
            const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
            for (int j = 0; j < 8; j++) win[k][j] = __bfloat162float(pv[j]);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned t = t0 + (unsigned)i;
            if (t >= (req ? rows_r : tokens)) break;
            const uint4 v = mpf_ld8(xin + (size_t)t * is + c0, t < rows_r);
            const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
            float cur[8];
#pragma unroll
            for (int j = 0; j < 8; j++) cur[j] = __bfloat162float(pv[j]);
            if (t >= hw) {
                __nv_bfloat16 o[8];
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    float s = bv[j];
                    s = fmaf(win[0][j], wk[0][j], s);
                    s = fmaf(win[1][j], wk[1][j], s);
                    s = fmaf(win[2][j], wk[2][j], s);
                    s = fmaf(cur[j], wk[3][j], s);
                    o[j] = __float2bfloat16(t < rows_r ? mamba_silu(s) : 0.f);
                }
                *reinterpret_cast<uint4*>(xout + (size_t)t * channels + c0) = *reinterpret_cast<const uint4*>(o);
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                win[0][j] = win[1][j];
                win[1][j] = win[2][j];
                win[2][j] = cur[j];
            }
        }
    }
    if (req) mpf_zero_rows(out, channels, mpf_real(req), tokens, slice, nblk);
}

/* One chunk of MAMBA2_CHUNK_STATE for head h: dt and the decay cumsum of the chunk's rows (written
 * to dtv / cum), then the chunk's local state X^T bf16(B exp(cum_last - cum) dt) on tensor cores into
 * acc (warp tile p [wp, wp + 32) x n [wn, wn + 32)). Returns cum at the chunk's last row. */
static __device__ __forceinline__ float mpf_state_chunk(float (&acc)[2][4][4], const __nv_bfloat16* __restrict__ xbc,
    const __nv_bfloat16* __restrict__ proj, const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias, float* __restrict__ dtv, float* __restrict__ cum, unsigned h,
    unsigned grp, unsigned c, unsigned t0, unsigned clen, size_t ts, unsigned d_inner, unsigned conv_dim,
    unsigned ds, unsigned dcol, float dt_lo, float dt_hi, float* s_w, float* s_part, __nv_bfloat16* Xs,
    __nv_bfloat16* Bs) {
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned wp = (warp >> 2) * 32u, wn = (warp & 3u) * 32u;
    float a = 0.f, dt = 0.f;
    if (tid < MAMBA_L) {
        const unsigned tok = t0 + tid;
        if (tid < clen) {
            dt = mamba_dt(mamba_bf(proj, (size_t)tok * ds + dcol + h), mamba_bf(dt_bias, h), dt_lo, dt_hi);
            a = dt * -__expf(mamba_bf(A_log, h));
        }
        float v = a;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float u = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= (unsigned)o) v += u;
        }
        if (lane == 31) s_part[warp] = v;
        a = v;
    }
    __syncthreads();
    if (tid < MAMBA_L) {
        for (unsigned w = 0; w < warp; w++) a += s_part[w];
        dtv[(size_t)h * ts + c * MAMBA_L + tid] = dt;
        cum[(size_t)h * ts + c * MAMBA_L + tid] = a;
        s_w[tid] = a;
    }
    __syncthreads();
    const float last = s_w[MAMBA_L - 1];
    __syncthreads();
    if (tid < MAMBA_L) s_w[tid] = __expf(fminf(last - s_w[tid], 0.f)) * dt;
    __syncthreads();
    /* Stage x (raw bf16) and B' (scaled, bf16). */
    for (unsigned e = tid; e < MAMBA_L * (MAMBA_P / 8u); e += 256u) {
        const unsigned r = e / (MAMBA_P / 8u), q = (e % (MAMBA_P / 8u)) * 8u, tok = t0 + r;
        *reinterpret_cast<uint4*>(Xs + r * MPF_SP + q) =
            mpf_ld8(xbc + (size_t)tok * conv_dim + h * MAMBA_P + q, r < clen);
    }
    for (unsigned e = tid; e < MAMBA_L * (MAMBA_N / 8u); e += 256u) {
        const unsigned r = e / (MAMBA_N / 8u), q = (e % (MAMBA_N / 8u)) * 8u, tok = t0 + r;
        const uint4 v = mpf_ld8(xbc + (size_t)tok * conv_dim + d_inner + grp * MAMBA_N + q, r < clen);
        const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
        const float w = s_w[r];
        uint4 o;
        o.x = mpf_bf2(__bfloat162float(pv[0]) * w, __bfloat162float(pv[1]) * w);
        o.y = mpf_bf2(__bfloat162float(pv[2]) * w, __bfloat162float(pv[3]) * w);
        o.z = mpf_bf2(__bfloat162float(pv[4]) * w, __bfloat162float(pv[5]) * w);
        o.w = mpf_bf2(__bfloat162float(pv[6]) * w, __bfloat162float(pv[7]) * w);
        *reinterpret_cast<uint4*>(Bs + r * MPF_SN + q) = o;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;
#pragma unroll 2
    for (unsigned k0 = 0; k0 < MAMBA_L; k0 += 16u) {
        unsigned af[2][4], bf[2][4];
        /* A = X^T [p][t]: X stored [t][p] -> transposed load, k rows (l&7)+((l>>4)<<3). */
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
            mpf_ldsm_t(af[mi], Xs + (k0 + (lane & 7u) + ((lane >> 4) << 3)) * MPF_SP + wp + mi * 16u +
                                   ((lane >> 3) & 1u) * 8u);
        /* B = B'[t][n] stored [k][n] -> transposed load, rows (l&15), cols (l>>4)*8. */
#pragma unroll
        for (int p = 0; p < 2; p++)
            mpf_ldsm_t(bf[p], Bs + (k0 + (lane & 15u)) * MPF_SN + wn + p * 16u + (lane >> 4) * 8u);
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int nj = 0; nj < 4; nj++)
                mpf_mma(acc[mi][nj], af[mi], bf[nj >> 1][(nj & 1) * 2], bf[nj >> 1][(nj & 1) * 2 + 1]);
    }
    return last;
}

/* MAMBA2_CHUNK_STATE: block per (chunk, head). S[p][n] = sum_t x[t][p] B'[t][n],
 * B' = bf16(B exp(min(cum_last - cum_t, 0)) dt_t). Warps 2 (p) x 4 (n), warp tile 32 x 32. */
template <bool PK = false>
static __device__ __noinline__ void d_mamba2_chunk_state_tc(float* __restrict__ cstate, const __nv_bfloat16* __restrict__ xbc,
    const __nv_bfloat16* __restrict__ proj, const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias, float* __restrict__ dtv, float* __restrict__ cum,
    unsigned heads, unsigned hdim, unsigned rows, unsigned tokens, unsigned dstate, unsigned groups,
    float dt_lo, float dt_hi, unsigned dt_source, float* __restrict__ smem, unsigned slice, unsigned nblk,
    const int* __restrict__ req = nullptr) {
    if (!PK) req = nullptr; /* the serial instantiation folds every packed branch away */
    if (!mamba_chunk_shape_ok(hdim, dstate, rows, tokens) || heads % groups != 0 || !dt_source ||
        (req && mpf_real(req) > tokens)) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned d_inner = heads * MAMBA_P, conv_dim = d_inner + 2u * groups * MAMBA_N;
    const unsigned nchunk = req ? mpf_nchunk(req) : (rows + MAMBA_L - 1) / MAMBA_L, hpg = heads / groups;
    const size_t ts = req ? (size_t)tokens + (size_t)MAMBA_L * (unsigned)req[0] : tokens;
    const unsigned ds = dt_source & 0xFFFFu, dcol = dt_source >> 16;
    float* s_w = smem;                                                  /* [L] */
    float* s_part = smem + MAMBA_L;                                     /* [8] */
    __nv_bfloat16* Xs = reinterpret_cast<__nv_bfloat16*>(smem + MAMBA_L + 32u); /* [128 t][72] */
    __nv_bfloat16* Bs = Xs + MAMBA_L * MPF_SP;                          /* [128 t][136] */
    const unsigned wp = (warp >> 2) * 32u, wn = (warp & 3u) * 32u, g = lane >> 2, tq = lane & 3u;
    for (unsigned item = slice; item < nchunk * heads; item += nblk) {
        const unsigned c = item / heads, h = item % heads, grp = h / hpg;
        unsigned t0, clen, slot;
        mpf_chunk(req, c, rows, t0, clen, slot);
        float acc[2][4][4];
        mpf_state_chunk(acc, xbc, proj, A_log, dt_bias, dtv, cum, h, grp, c, t0, clen, ts, d_inner, conv_dim, ds,
                        dcol, dt_lo, dt_hi, s_w, s_part, Xs, Bs);
        float* dst = cstate + ((size_t)c * heads + h) * MAMBA_P * MAMBA_N;
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int nj = 0; nj < 4; nj++) {
                const unsigned p = wp + mi * 16u + g, n = wn + nj * 8u + 2u * tq;
                *reinterpret_cast<float2*>(dst + (size_t)p * MAMBA_N + n) = make_float2(acc[mi][nj][0], acc[mi][nj][1]);
                *reinterpret_cast<float2*>(dst + (size_t)(p + 8u) * MAMBA_N + n) =
                    make_float2(acc[mi][nj][2], acc[mi][nj][3]);
            }
        __syncthreads();
    }
}

/* MAMBA2_CHUNK_PASS: op_mamba.cuh's recurrence with the local states of 8 chunks loaded ahead
 * of each serial group of updates (independent loads in flight per thread). Packed: one walk per
 * request from its slot state over its own chunks. */
template <bool PK = false>
static __device__ __noinline__ void d_mamba2_chunk_pass_pf(float* __restrict__ cstate, const float* __restrict__ cum,
    __half* __restrict__ state, unsigned heads, unsigned hdim, unsigned rows, unsigned tokens,
    unsigned dstate, unsigned state_bytes, unsigned slice, unsigned nblk, const int* __restrict__ req = nullptr) {
    if (!PK) req = nullptr; /* the serial instantiation folds every packed branch away */
    if (!mamba_chunk_shape_ok(hdim, dstate, rows, tokens) || state_bytes != MAMBA_STATE_BYTES ||
        (req && mpf_real(req) > tokens)) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned q = MAMBA_N / 4u, per_seg = heads * MAMBA_P * q, nseg = req ? (unsigned)req[0] : 1u;
    const size_t per_chunk = (size_t)heads * MAMBA_P * MAMBA_N;
    const size_t ts = req ? (size_t)tokens + (size_t)MAMBA_L * nseg : tokens;
    for (unsigned idx = slice * blockDim.x + threadIdx.x; idx < nseg * per_seg; idx += nblk * blockDim.x) {
        const unsigned seg = idx / per_seg, e = idx - seg * per_seg, h = e / (MAMBA_P * q);
        unsigned cb = 0, nchunk = (rows + MAMBA_L - 1) / MAMBA_L;
        size_t soff = 0;
        if (req) {
            for (unsigned r = 0; r < seg; r++) cb += (mpf_len(req, r) + MAMBA_L - 1) / MAMBA_L;
            nchunk = (mpf_len(req, seg) + MAMBA_L - 1) / MAMBA_L;
            if (nchunk == 0u) continue;
            soff = (size_t)req[3 + 4 * seg] * per_chunk;
        }
        const size_t off = (size_t)e * 4u;
        float4 carry = mamba_ld_h4(state + soff + off);
        for (unsigned c0 = 0; c0 < nchunk; c0 += 8u) {
            float4 loc[8];
            float d[8];
#pragma unroll
            for (unsigned j = 0; j < 8u; j++) {
                if (c0 + j < nchunk) {
                    loc[j] = *(const float4*)(cstate + (cb + c0 + j) * per_chunk + off);
                    d[j] = __expf(cum[(size_t)h * ts + (cb + c0 + j) * MAMBA_L + MAMBA_L - 1]);
                }
            }
#pragma unroll
            for (unsigned j = 0; j < 8u; j++) {
                if (c0 + j < nchunk) {
                    *(float4*)(cstate + (cb + c0 + j) * per_chunk + off) = carry;
                    carry.x = fmaf(d[j], carry.x, loc[j].x);
                    carry.y = fmaf(d[j], carry.y, loc[j].y);
                    carry.z = fmaf(d[j], carry.z, loc[j].z);
                    carry.w = fmaf(d[j], carry.w, loc[j].w);
                }
            }
        }
        mamba_st_h4_rn(state + soff + off, carry);
    }
}

/* MAMBA2_CHUNK_STATE + MAMBA2_CHUNK_PASS in one walk (PLOW_NV_MAMBA_SCAN_FUSE): block per head
 * (packed: per request and head) over its chunks in order. Each chunk is mpf_state_chunk (the same
 * mma sequence per element as CHUNK_STATE), then CHUNK_PASS's recurrence on the accumulator
 * fragment: cstate[c] = entering state, carry = exp(cum_last) carry + local. Bit-identical to the
 * two ops; the per-chunk local states never reach memory. `state` is CHUNK_PASS's operand. `ebf`:
 * the entering states are stored as CHUNK_OUT stages them, bf16(fp16(E)), in the buffer's first half
 * (CHUNK_OUT with `ebf` reads them as is). */
template <bool PK = false>
static __device__ __noinline__ void d_mamba2_chunk_scan_state_tc(float* __restrict__ cstate,
    const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ proj,
    const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ dtv, float* __restrict__ cum, __half* __restrict__ state, unsigned heads,
    unsigned hdim, unsigned rows, unsigned tokens, unsigned dstate, unsigned groups, float dt_lo,
    float dt_hi, unsigned dt_source, float* __restrict__ smem, unsigned slice, unsigned nblk,
    const int* __restrict__ req = nullptr, bool ebf = false) {
    if (!PK) req = nullptr;
    if (!mamba_chunk_shape_ok(hdim, dstate, rows, tokens) || heads % groups != 0 || !dt_source || !state ||
        (req && mpf_real(req) > tokens)) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned d_inner = heads * MAMBA_P, conv_dim = d_inner + 2u * groups * MAMBA_N, hpg = heads / groups;
    const unsigned nseg = req ? (unsigned)req[0] : 1u;
    const size_t ts = req ? (size_t)tokens + (size_t)MAMBA_L * nseg : tokens;
    const size_t per_chunk = (size_t)heads * MAMBA_P * MAMBA_N;
    const unsigned ds = dt_source & 0xFFFFu, dcol = dt_source >> 16;
    float* s_w = smem;
    float* s_part = smem + MAMBA_L;
    __nv_bfloat16* Xs = reinterpret_cast<__nv_bfloat16*>(smem + MAMBA_L + 32u);
    __nv_bfloat16* Bs = Xs + MAMBA_L * MPF_SP;
    const unsigned wp = (warp >> 2) * 32u, wn = (warp & 3u) * 32u, g = lane >> 2, tq = lane & 3u;
    for (unsigned item = slice; item < nseg * heads; item += nblk) {
        const unsigned seg = item / heads, h = item - seg * heads, grp = h / hpg;
        unsigned cb = 0u, nc = (rows + MAMBA_L - 1u) / MAMBA_L;
        size_t soff = 0;
        if (req) {
            for (unsigned r = 0; r < seg; r++) cb += (mpf_len(req, r) + MAMBA_L - 1u) / MAMBA_L;
            nc = (mpf_len(req, seg) + MAMBA_L - 1u) / MAMBA_L;
            if (nc == 0u) continue;
            soff = (size_t)req[3 + 4 * seg] * per_chunk;
        }
        const __half* sh = state + soff + (size_t)h * MAMBA_P * MAMBA_N;
        float carry[2][4][4];
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int nj = 0; nj < 4; nj++) {
                const unsigned p = wp + mi * 16u + g, n = wn + nj * 8u + 2u * tq;
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(sh + (size_t)p * MAMBA_N + n));
                const float2 b = __half22float2(*reinterpret_cast<const __half2*>(sh + (size_t)(p + 8u) * MAMBA_N + n));
                carry[mi][nj][0] = a.x;
                carry[mi][nj][1] = a.y;
                carry[mi][nj][2] = b.x;
                carry[mi][nj][3] = b.y;
            }
        for (unsigned j = 0; j < nc; j++) {
            const unsigned c = cb + j;
            unsigned t0, clen, slot;
            mpf_chunk(req, c, rows, t0, clen, slot);
            float acc[2][4][4];
            const float last = mpf_state_chunk(acc, xbc, proj, A_log, dt_bias, dtv, cum, h, grp, c, t0, clen, ts,
                                               d_inner, conv_dim, ds, dcol, dt_lo, dt_hi, s_w, s_part, Xs, Bs);
            const float d = __expf(last);
            float* dst = cstate + (size_t)c * per_chunk + (size_t)h * MAMBA_P * MAMBA_N;
            __nv_bfloat16* dsb = reinterpret_cast<__nv_bfloat16*>(cstate) + (size_t)c * per_chunk +
                                 (size_t)h * MAMBA_P * MAMBA_N;
            auto h16 = [](float x) { return __half2float(__float2half_rn(x)); };
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
#pragma unroll
                for (int nj = 0; nj < 4; nj++) {
                    const unsigned p = wp + mi * 16u + g, n = wn + nj * 8u + 2u * tq;
                    float* cr = carry[mi][nj];
                    if (ebf) { /* what CHUNK_OUT stages: bf16 of the fp16-rounded entering state */
                        *reinterpret_cast<unsigned*>(dsb + (size_t)p * MAMBA_N + n) = mpf_bf2(h16(cr[0]), h16(cr[1]));
                        *reinterpret_cast<unsigned*>(dsb + (size_t)(p + 8u) * MAMBA_N + n) =
                            mpf_bf2(h16(cr[2]), h16(cr[3]));
                    } else {
                        *reinterpret_cast<float2*>(dst + (size_t)p * MAMBA_N + n) = make_float2(cr[0], cr[1]);
                        *reinterpret_cast<float2*>(dst + (size_t)(p + 8u) * MAMBA_N + n) = make_float2(cr[2], cr[3]);
                    }
#pragma unroll
                    for (int q = 0; q < 4; q++) cr[q] = fmaf(d, cr[q], acc[mi][nj][q]);
                }
            __syncthreads();
        }
        __half* so = state + soff + (size_t)h * MAMBA_P * MAMBA_N;
#pragma unroll
        for (int mi = 0; mi < 2; mi++)
#pragma unroll
            for (int nj = 0; nj < 4; nj++) {
                const unsigned p = wp + mi * 16u + g, n = wn + nj * 8u + 2u * tq;
                *reinterpret_cast<__half2*>(so + (size_t)p * MAMBA_N + n) =
                    __floats2half2_rn(carry[mi][nj][0], carry[mi][nj][1]);
                *reinterpret_cast<__half2*>(so + (size_t)(p + 8u) * MAMBA_N + n) =
                    __floats2half2_rn(carry[mi][nj][2], carry[mi][nj][3]);
            }
    }
}

/* MAMBA2_CHUNK_OUT: block per (chunk, head); warp w owns rows t in [16w, 16w+16).
 * y = bf16(exp(cum_t) C . bf16(E)^T + sum_{s<=t} bf16(CB exp(min(cum_t-cum_s,0)) dt_s) x_s + D x_t). */
template <bool PK = false>
static __device__ __noinline__ void d_mamba2_chunk_out_tc(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ xbc,
    const float* __restrict__ enter, const float* __restrict__ cum, const float* __restrict__ dtv,
    const __nv_bfloat16* __restrict__ Dp, unsigned heads, unsigned hdim, unsigned rows,
    unsigned tokens, unsigned dstate, unsigned groups, float* __restrict__ smem, unsigned slice,
    unsigned nblk, const int* __restrict__ req = nullptr, bool ebf = false) {
    if (!PK) req = nullptr; /* the serial instantiation folds every packed branch away */
    if (!mamba_chunk_shape_ok(hdim, dstate, rows, tokens) || heads % groups != 0 ||
        (req && mpf_real(req) > tokens)) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, tq = lane & 3u;
    const __nv_bfloat16* enter_bf = reinterpret_cast<const __nv_bfloat16*>(enter);
    const unsigned d_inner = heads * MAMBA_P, conv_dim = d_inner + 2u * groups * MAMBA_N;
    const unsigned live = req ? mpf_nchunk(req) : (rows + MAMBA_L - 1) / MAMBA_L, hpg = heads / groups;
    const unsigned nchunk = req ? live : tokens / MAMBA_L;
    const size_t ts = req ? (size_t)tokens + (size_t)MAMBA_L * (unsigned)req[0] : tokens;
    float* s_cum = smem;                                                  /* [L] */
    float* s_dt = s_cum + MAMBA_L;                                        /* [L] */
    __nv_bfloat16* T0 = reinterpret_cast<__nv_bfloat16*>(smem + 2u * MAMBA_L); /* C, then B: [128][136] */
    __nv_bfloat16* T1 = T0 + MAMBA_L * MPF_SN;                            /* E [64][136], then x [128][72] */
    const unsigned r0 = warp * 16u;
    for (unsigned item = slice; item < nchunk * heads; item += nblk) {
        const unsigned c = item / heads, h = item % heads, grp = h / hpg;
        if (c >= live) {
            const unsigned t0 = c * MAMBA_L;
            for (unsigned e = tid; e < MAMBA_L * (MAMBA_P / 8u); e += 256u) {
                const unsigned r = e / (MAMBA_P / 8u), q = (e % (MAMBA_P / 8u)) * 8u;
                *reinterpret_cast<uint4*>(y + (size_t)(t0 + r) * d_inner + h * MAMBA_P + q) = make_uint4(0u, 0u, 0u, 0u);
            }
            continue;
        }
        unsigned t0, clen, slot;
        mpf_chunk(req, c, rows, t0, clen, slot);
        const size_t crow = d_inner + groups * MAMBA_N + grp * MAMBA_N, brow = d_inner + grp * MAMBA_N;
        if (tid < MAMBA_L) {
            s_cum[tid] = cum[(size_t)h * ts + c * MAMBA_L + tid];
            s_dt[tid] = dtv[(size_t)h * ts + c * MAMBA_L + tid];
        }
        /* Phase 1: C [t][n], E [p][n]: the f32 entering state through the fp16 state dtype, then
         * bf16 (the peer stores entering states as fp16 and casts them to bf16 for the dot). */
        auto h16 = [](float x) { return __half2float(__float2half_rn(x)); };
        for (unsigned e = tid; e < MAMBA_L * (MAMBA_N / 8u); e += 256u) {
            const unsigned r = e / (MAMBA_N / 8u), q = (e % (MAMBA_N / 8u)) * 8u, tok = t0 + r;
            *reinterpret_cast<uint4*>(T0 + r * MPF_SN + q) = mpf_ld8(xbc + (size_t)tok * conv_dim + crow + q, r < clen);
        }
        const float* E = enter + ((size_t)c * heads + h) * MAMBA_P * MAMBA_N;
        const __nv_bfloat16* Eb = enter_bf + ((size_t)c * heads + h) * MAMBA_P * MAMBA_N;
        for (unsigned e = tid; e < MAMBA_P * (MAMBA_N / 8u); e += 256u) {
            const unsigned p = e / (MAMBA_N / 8u), q = (e % (MAMBA_N / 8u)) * 8u;
            if (ebf) { /* the fused walk stored bf16(fp16(E)) */
                *reinterpret_cast<uint4*>(T1 + p * MPF_SN + q) =
                    *reinterpret_cast<const uint4*>(Eb + (size_t)p * MAMBA_N + q);
                continue;
            }
            const float4 v0 = *reinterpret_cast<const float4*>(E + (size_t)p * MAMBA_N + q);
            const float4 v1 = *reinterpret_cast<const float4*>(E + (size_t)p * MAMBA_N + q + 4u);
            uint4 o;
            o.x = mpf_bf2(h16(v0.x), h16(v0.y));
            o.y = mpf_bf2(h16(v0.z), h16(v0.w));
            o.z = mpf_bf2(h16(v1.x), h16(v1.y));
            o.w = mpf_bf2(h16(v1.z), h16(v1.w));
            *reinterpret_cast<uint4*>(T1 + p * MPF_SN + q) = o;
        }
        __syncthreads();
        unsigned cf[8][4];
#pragma unroll
        for (int k = 0; k < 8; k++)
            mpf_ldsm(cf[k], T0 + (r0 + (lane & 15u)) * MPF_SN + k * 16u + (lane >> 4) * 8u);
        float acc[8][4];
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[j][q] = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) {
#pragma unroll
            for (int pp = 0; pp < 4; pp++) {
                unsigned bf[4];
                /* B = E^T [n][p]: E stored [p][n] (k contiguous) -> plain load. */
                mpf_ldsm(bf, T1 + (pp * 16u + (lane & 7u) + ((lane >> 4) << 3)) * MPF_SN + k * 16u +
                                  ((lane >> 3) & 1u) * 8u);
                mpf_mma(acc[2 * pp], cf[k], bf[0], bf[1]);
                mpf_mma(acc[2 * pp + 1], cf[k], bf[2], bf[3]);
            }
        }
        const unsigned ta = r0 + g, tb = ta + 8u;
        const float cta = s_cum[ta], ctb = s_cum[tb];
        {
            const float sa = __expf(cta), sb = __expf(ctb);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[j][0] *= sa;
                acc[j][1] *= sa;
                acc[j][2] *= sb;
                acc[j][3] *= sb;
            }
        }
        __syncthreads();
        /* Phase 2: B [s][n] over C's tile, x [s][p] over E's. */
        for (unsigned e = tid; e < MAMBA_L * (MAMBA_N / 8u); e += 256u) {
            const unsigned r = e / (MAMBA_N / 8u), q = (e % (MAMBA_N / 8u)) * 8u, tok = t0 + r;
            *reinterpret_cast<uint4*>(T0 + r * MPF_SN + q) = mpf_ld8(xbc + (size_t)tok * conv_dim + brow + q, r < clen);
        }
        for (unsigned e = tid; e < MAMBA_L * (MAMBA_P / 8u); e += 256u) {
            const unsigned r = e / (MAMBA_P / 8u), q = (e % (MAMBA_P / 8u)) * 8u, tok = t0 + r;
            *reinterpret_cast<uint4*>(T1 + r * MPF_SP + q) =
                mpf_ld8(xbc + (size_t)tok * conv_dim + h * MAMBA_P + q, r < clen);
        }
        __syncthreads();
        for (unsigned j = 0; j <= warp; j++) {
            const unsigned s0 = j * 16u;
            float cb[2][4];
#pragma unroll
            for (int a = 0; a < 2; a++)
#pragma unroll
                for (int q = 0; q < 4; q++) cb[a][q] = 0.f;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                unsigned bf[4];
                /* B = B^T [n][s]: B stored [s][n] (k contiguous) -> plain load. */
                mpf_ldsm(bf, T0 + (s0 + (lane & 7u) + ((lane >> 4) << 3)) * MPF_SN + k * 16u + ((lane >> 3) & 1u) * 8u);
                mpf_mma(cb[0], cf[k], bf[0], bf[1]);
                mpf_mma(cb[1], cf[k], bf[2], bf[3]);
            }
            unsigned af[4];
#pragma unroll
            for (int a = 0; a < 2; a++) {
                const unsigned sa = s0 + a * 8u + 2u * tq, sb = sa + 1u;
                const float csa = s_cum[sa], csb = s_cum[sb], dsa = s_dt[sa], dsb = s_dt[sb];
                const float v0 = sa <= ta ? cb[a][0] * __expf(fminf(cta - csa, 0.f)) * dsa : 0.f;
                const float v1 = sb <= ta ? cb[a][1] * __expf(fminf(cta - csb, 0.f)) * dsb : 0.f;
                const float v2 = sa <= tb ? cb[a][2] * __expf(fminf(ctb - csa, 0.f)) * dsa : 0.f;
                const float v3 = sb <= tb ? cb[a][3] * __expf(fminf(ctb - csb, 0.f)) * dsb : 0.f;
                af[2 * a] = mpf_bf2(v0, v1);
                af[2 * a + 1] = mpf_bf2(v2, v3);
            }
#pragma unroll
            for (int pp = 0; pp < 4; pp++) {
                unsigned bf[4];
                /* B = x [s][p] stored [k][n] -> transposed load. */
                mpf_ldsm_t(bf, T1 + (s0 + (lane & 15u)) * MPF_SP + pp * 16u + (lane >> 4) * 8u);
                mpf_mma(acc[2 * pp], af, bf[0], bf[1]);
                mpf_mma(acc[2 * pp + 1], af, bf[2], bf[3]);
            }
        }
        const float Dh = mamba_bf(Dp, h);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const unsigned p = j * 8u + 2u * tq;
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const unsigned t = hh ? tb : ta, tok = t0 + t;
                if (req && t >= clen) continue; /* the next request's rows */
                const __nv_bfloat16* xr = T1 + t * MPF_SP + p;
                const float o0 = t < clen ? fmaf(Dh, __bfloat162float(xr[0]), acc[j][2 * hh]) : 0.f;
                const float o1 = t < clen ? fmaf(Dh, __bfloat162float(xr[1]), acc[j][2 * hh + 1]) : 0.f;
                *reinterpret_cast<unsigned*>(y + (size_t)tok * d_inner + h * MAMBA_P + p) = mpf_bf2(o0, o1);
            }
        }
        __syncthreads();
    }
    if (req) mpf_zero_rows(y, d_inner, mpf_real(req), tokens, slice, nblk);
}
