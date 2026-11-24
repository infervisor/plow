/* op_mamba_span.cuh — span-mode (speculative verify) bodies of MAMBA2_CONV and MAMBA2_STEP, compiled
 * into the row-slot objects only (PLOW_NV_ROW_SLOT; devgen spec/span_verify.rs, plowrt
 * exec/gpu/spec.rs, plow_asset::recurrent_state::Replay).
 *
 * Rows r0..r0+n-1 carrying the same rowslot entry (bit 31 clear) are one sequence's consecutive
 * tokens and chain through the slot's conv history and SSM state (the span is at most depth + 1
 * rows). A span writes no state: it records each row's step inputs at rec[parity][row] (parity =
 * table[0]) and leaves the buffers at the state before the span. A commit of c rows defers them:
 * the next row-slot launch replays records [row0, row0 + c) of the entry's parity onto the slot
 * (table[2 + slot], plow_asset Replay::entry) before anything reads it, in the slot's span item
 * when the slot runs a span there, else in a replay-only item; the replay program is such a launch
 * with no span rows. Padding rows (bit 31) and inactive rows run nothing. The arithmetic is the
 * plain bodies' (op_mamba.cuh d_mamba2_conv / d_mamba2_step), replayed in the same order.
 */
#pragma once
#if PLOW_NV_ROW_SLOT
#include <cuda_fp16.h>

static __device__ __forceinline__ unsigned mamba_span_len(const int* rowslot, unsigned r0,
                                                          unsigned rows, unsigned depth) {
    const int tag = rowslot[r0];
    if (tag < 0 || (r0 > 0 && rowslot[r0 - 1] == tag)) return 0;
    unsigned n = 1;
    while (r0 + n < rows && rowslot[r0 + n] == tag) n++;
    if (n > depth + 1u) __trap();
    return n;
}

/* One work item of a span-mode launch: unit u < rows is the span starting at row u (n > 0 rows),
 * unit rows + s a replay-only item for slot s; rep = slot's replay-table entry. False: no work. */
static __device__ __forceinline__ bool mamba_span_unit(const int* __restrict__ rowslot,
    const int* __restrict__ active, const int* __restrict__ table, unsigned u, unsigned rows,
    unsigned slots, unsigned depth, size_t& slot, unsigned& n, unsigned& rep) {
    n = 0;
    if (u < rows) {
        if (active && active[u] <= 0) return false;
        n = mamba_span_len(rowslot, u, rows, depth);
        if (!n) return false;
        slot = (size_t)rowslot[u];
        if (slot >= slots || u + n > slots) __trap();
        rep = (unsigned)table[2 + slot];
        if ((rep & 0xFFu) && !(rep >> 17 & 1u)) __trap();
        return true;
    }
    slot = u - rows;
    rep = (unsigned)table[2 + slot];
    return (rep & 0xFFu) && !(rep >> 17 & 1u);
}

/* The units of a span-mode launch that have work (span starts, replay-only slots; at most one per
 * slot), compacted once per block (block-uniform call): testing every (unit, ...) item walked the
 * dependent rowslot loads for every row and padding slot, most of which have no work. Entry:
 * u | n << 12 | slot << 16, the replay entry beside it; list order is arbitrary. */
#define MAMBA_SPAN_UNITS 256u
static __device__ unsigned mamba_span_units(const int* __restrict__ rowslot,
    const int* __restrict__ active, const int* __restrict__ table, unsigned rows, unsigned slots,
    unsigned depth, const unsigned*& ent, const unsigned*& reps) {
    __shared__ unsigned s_ent[MAMBA_SPAN_UNITS], s_rep[MAMBA_SPAN_UNITS], s_wn[32];
    if (slots > MAMBA_SPAN_UNITS || rows + slots > 4096u || depth > 14u || blockDim.x > 1024u) __trap();
    const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned base = 0;
    for (unsigned u0 = 0; u0 < rows + slots; u0 += blockDim.x) {
        const unsigned u = u0 + threadIdx.x;
        size_t slot = 0;
        unsigned n = 0, rep = 0;
        const bool has = u < rows + slots && mamba_span_unit(rowslot, active, table, u, rows, slots, depth, slot, n, rep);
        const unsigned m = __ballot_sync(0xFFFFFFFFu, has);
        __syncthreads();
        if (lane == 0) s_wn[w] = __popc(m);
        __syncthreads();
        unsigned k = base + __popc(m & ((1u << lane) - 1u));
        for (unsigned j = 0; j < nw; j++) {
            k += j < w ? s_wn[j] : 0u;
            base += s_wn[j];
        }
        if (has) {
            if (k >= MAMBA_SPAN_UNITS) __trap();
            s_ent[k] = u | n << 12 | (unsigned)slot << 16;
            s_rep[k] = rep;
        }
    }
    __syncthreads();
    ent = s_ent;
    reps = s_rep;
    return base;
}

/* MAMBA2_CONV span mode: one thread per (unit, channel). t6 = rowslot, t7 = the conv-input
 * record [2][slots][C], i5 & 0xFFFF = depth, i5 >> 16 = the replay table. */
static __device__ void d_mamba2_conv_span(__nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ history,
    const int* __restrict__ active, unsigned channels, unsigned width, unsigned rows,
    unsigned source, const int* __restrict__ rowslot, __nv_bfloat16* __restrict__ rec,
    unsigned depth, const int* __restrict__ table, unsigned slice, unsigned nblk) {
    if (width < 2 || width > 8 || !rec) { if (threadIdx.x == 0) __trap(); return; }
    const unsigned is = source ? (source & 0xFFFFu) : channels;
    in += source >> 16;
    const unsigned hw = width - 1;
    const unsigned slots = (unsigned)table[1], wpar = (unsigned)table[0] & 1u;
    const unsigned *ent, *reps;
    const unsigned nu = mamba_span_units(rowslot, active, table, rows, slots, depth, ent, reps);
    for (unsigned idx = slice * blockDim.x + threadIdx.x; idx < nu * channels;
         idx += nblk * blockDim.x) {
        const unsigned c = idx % channels, e = ent[idx / channels], rep = reps[idx / channels];
        const unsigned u = e & 0xFFFu, n = e >> 12 & 0xFu;
        const size_t slot = e >> 16;
        __nv_bfloat16* live = history + (slot * channels + c) * hw;
        __nv_bfloat16 h[7];
        for (unsigned j = 0; j < hw; j++) h[j] = live[j];
        if (rep & 0xFFu) {
            const __nv_bfloat16* src =
                rec + ((size_t)(rep >> 16 & 1u) * slots + (rep >> 8 & 0xFFu)) * channels + c;
            for (unsigned s = 0; s < (rep & 0xFFu); s++) {
                for (unsigned j = 0; j + 1 < hw; j++) h[j] = h[j + 1];
                h[hw - 1] = src[(size_t)s * channels];
            }
            for (unsigned j = 0; j < hw; j++) live[j] = h[j];
        }
        const __nv_bfloat16* w = weight + (size_t)c * width;
        const float b = bias ? mamba_bf(bias, c) : 0.f;
        for (unsigned s = 0; s < n; s++) {
            const unsigned row = u + s;
            float sum = b;
            for (unsigned j = 0; j < hw; j++) sum = fmaf(__bfloat162float(h[j]), mamba_bf(w, j), sum);
            const __nv_bfloat16 x = in[(size_t)row * is + c];
            sum = fmaf(__bfloat162float(x), mamba_bf(w, hw), sum);
            for (unsigned j = 0; j + 1 < hw; j++) h[j] = h[j + 1];
            h[hw - 1] = x;
            out[(size_t)row * channels + c] = __float2bfloat16(mamba_silu(sum));
            rec[((size_t)wpar * slots + row) * channels + c] = x;
        }
    }
}

static __device__ __forceinline__ void mamba_span_prefetch(const void* p) {
    asm volatile("prefetch.global.L1 [%0];" ::"l"(p));
}

/* State element I/O: f32 (float4 per lane), or fp16 stored round-to-nearest (vLLM's speculative
 * path stores its fp16 state without stochastic rounding). */
static __device__ __forceinline__ float4 mamba_span_ld(const float* p) { return *(const float4*)p; }
static __device__ __forceinline__ float4 mamba_span_ld(const __half* p) {
    const uint2 u = *(const uint2*)p;
    const float2 a = __half22float2(*(const __half2*)&u.x), b = __half22float2(*(const __half2*)&u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
static __device__ __forceinline__ void mamba_span_st(float* p, float4 v) { *(float4*)p = v; }
static __device__ __forceinline__ void mamba_span_st(__half* p, float4 v) {
    const __half2 a = __floats2half2_rn(v.x, v.y), b = __floats2half2_rn(v.z, v.w);
    uint2 u;
    u.x = *(const unsigned*)&a;
    u.y = *(const unsigned*)&b;
    *(uint2*)p = u;
}

/* The value the next span row continues from: what a commit would leave in the buffer. */
static __device__ __forceinline__ float4 mamba_span_round(float4 v, const float*) { return v; }
static __device__ __forceinline__ float4 mamba_span_round(float4 v, const __half*) {
    return make_float4(__half2float(__float2half_rn(v.x)), __half2float(__float2half_rn(v.y)),
                       __half2float(__float2half_rn(v.z)), __half2float(__float2half_rn(v.w)));
}

/* MAMBA2_STEP span mode: one warp per (unit, head, 8 head-dim rows) replays the slot's deferred
 * records, then walks the span in order with the state in registers (the slot state is read once,
 * and written once only after a replay). Operands ride the high halves of the plain op's
 * immediates: i5 >> 8 & 0xFF = depth (i5 & 0xFF = state element bytes, 0 = f32), i5 >> 16 = the
 * replay table, i6 >> 16 = in.rowslot, i7 >> 16 = the step record [2][slots][row]: x (H*P bf16),
 * B (G*N bf16), dt (H f32), the row padded to 16 bytes. */
template <typename S>
static __device__ void d_mamba2_step_span(__nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ xbc, const __nv_bfloat16* __restrict__ proj,
    const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ Dp,
    const __nv_bfloat16* __restrict__ dt_bias, S* __restrict__ state,
    const int* __restrict__ active, unsigned heads, unsigned hdim, unsigned dstate,
    unsigned groups, unsigned batch, unsigned dt_source, float dt_lo, float dt_hi,
    const int* __restrict__ rowslot, unsigned char* __restrict__ rec,
    const int* __restrict__ table, unsigned depth, unsigned slice, unsigned nblk) {
    if (dstate != 128 || hdim % 8 != 0 || groups == 0 || heads % groups != 0 || !dt_source || !rec) {
        if (threadIdx.x == 0) __trap();
        return;
    }
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const unsigned d_inner = heads * hdim, conv_dim = d_inner + 2u * groups * dstate;
    const unsigned hpg = heads / groups, ptiles = hdim / 8u;
    const unsigned ds = dt_source & 0xFFFFu, dcol = dt_source >> 16;
    const size_t slot_elems = (size_t)heads * hdim * dstate;
    const unsigned xb = d_inner + groups * dstate;
    const size_t rec_row = ((size_t)xb * 2u + (size_t)heads * 4u + 15u) & ~(size_t)15u;
    const unsigned slots = (unsigned)table[1], wpar = (unsigned)table[0] & 1u;
    const unsigned *ent, *reps;
    const unsigned nu = mamba_span_units(rowslot, active, table, batch, slots, depth, ent, reps);
    const unsigned per = heads * ptiles;
    for (unsigned item = slice * warps + warp; item < nu * per; item += nblk * warps) {
        const unsigned e = ent[item / per], rep = reps[item / per];
        const unsigned u = e & 0xFFFu, n = e >> 12 & 0xFu;
        const size_t slot = e >> 16;
        const unsigned h = (item / ptiles) % heads, p0 = (item % ptiles) * 8u;
        const size_t local = ((size_t)h * hdim + p0) * dstate + 4u * lane;
        const unsigned g = h / hpg;
        /* Every row's (and replayed record's) inputs requested up front into L1, before the state
         * loads: the walks below then wait on one round trip, not one per row. No registers held. */
        for (unsigned s = 0; s < (rep & 0xFFu); s++) {
            const unsigned char* rr = rec + ((size_t)(rep >> 16 & 1u) * slots + (rep >> 8 & 0xFFu) + s) * rec_row;
            mamba_span_prefetch(rr + ((size_t)d_inner + g * dstate + 4u * lane) * 2u);
            if (lane == 0) mamba_span_prefetch(rr + ((size_t)h * hdim + p0) * 2u);
            if (lane == 1) mamba_span_prefetch(rr + (size_t)xb * 2u + h * 4u);
        }
        for (unsigned s = 0; s < n; s++) {
            const __nv_bfloat16* row = xbc + (size_t)(u + s) * conv_dim;
            mamba_span_prefetch(row + d_inner + g * dstate + 4u * lane);
            mamba_span_prefetch(row + d_inner + groups * dstate + g * dstate + 4u * lane);
            if (lane == 0) mamba_span_prefetch(row + h * hdim + p0);
            if (lane == 1) mamba_span_prefetch(proj + (size_t)(u + s) * ds + dcol + h);
        }
        const float dneg = -__expf(mamba_bf(A_log, h));
        const float Dh = mamba_bf(Dp, h);
        S* live = state + slot * slot_elems + local;
        float4 st[8];
#pragma unroll
        for (int i = 0; i < 8; i++) st[i] = mamba_span_ld(live + (size_t)i * dstate);
        if (rep & 0xFFu) {
            for (unsigned s = 0; s < (rep & 0xFFu); s++) {
                const unsigned char* rr =
                    rec + ((size_t)(rep >> 16 & 1u) * slots + (rep >> 8 & 0xFFu) + s) * rec_row;
                const __nv_bfloat16* rx = (const __nv_bfloat16*)rr;
                const float dt = ((const float*)(rr + (size_t)xb * 2u))[h];
                const float dA = __expf(dt * dneg);
                float Bn[4];
#pragma unroll
                for (int j = 0; j < 4; j++) Bn[j] = mamba_bf(rx, d_inner + g * dstate + 4u * lane + j) * dt;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const float x = mamba_bf(rx, h * hdim + p0 + i);
                    st[i].x = fmaf(st[i].x, dA, x * Bn[0]);
                    st[i].y = fmaf(st[i].y, dA, x * Bn[1]);
                    st[i].z = fmaf(st[i].z, dA, x * Bn[2]);
                    st[i].w = fmaf(st[i].w, dA, x * Bn[3]);
                    st[i] = mamba_span_round(st[i], live);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; i++) mamba_span_st(live + (size_t)i * dstate, st[i]);
        }
        for (unsigned s = 0; s < n; s++) {
            const unsigned b = u + s;
            const __nv_bfloat16* row = xbc + (size_t)b * conv_dim;
            const float dt = mamba_dt(mamba_bf(proj, (size_t)b * ds + dcol + h), mamba_bf(dt_bias, h), dt_lo, dt_hi);
            const float dA = __expf(dt * dneg);
            unsigned char* wr = rec + ((size_t)wpar * slots + b) * rec_row;
            __nv_bfloat16* wx = (__nv_bfloat16*)wr;
            if (lane < 8) wx[h * hdim + p0 + lane] = row[h * hdim + p0 + lane];
            if (p0 == 0 && h % hpg == 0)
                *(uint2*)(wx + d_inner + g * dstate + 4u * lane) =
                    *(const uint2*)(row + d_inner + g * dstate + 4u * lane);
            if (p0 == 0 && lane == 0) ((float*)(wr + (size_t)xb * 2u))[h] = dt;
            float Bn[4], Cn[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                Bn[j] = mamba_bf(row, d_inner + g * dstate + 4u * lane + j) * dt;
                Cn[j] = mamba_bf(row, d_inner + groups * dstate + g * dstate + 4u * lane + j);
            }
            float x[8], acc[8];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                x[i] = mamba_bf(row, h * hdim + p0 + i);
                st[i].x = fmaf(st[i].x, dA, x[i] * Bn[0]);
                st[i].y = fmaf(st[i].y, dA, x[i] * Bn[1]);
                st[i].z = fmaf(st[i].z, dA, x[i] * Bn[2]);
                st[i].w = fmaf(st[i].w, dA, x[i] * Bn[3]);
                acc[i] = st[i].x * Cn[0] + st[i].y * Cn[1] + st[i].z * Cn[2] + st[i].w * Cn[3];
                st[i] = mamba_span_round(st[i], live);
            }
            float v = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float o = fmaf(Dh, x[i], mamba_warp_sum(acc[i]));
                v = lane == (unsigned)i ? o : v;
            }
            if (lane < 8) y[(size_t)b * d_inner + h * hdim + p0 + lane] = __float2bfloat16(v);
        }
    }
}
#endif /* PLOW_NV_ROW_SLOT */
