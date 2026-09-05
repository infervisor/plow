/* golden/moe.c — GPT-OSS flat-tensor MXFP4 MoE, scalar reference (dev_isa.h ops 147-150).
 *
 * Expert e of a [E][N][K/2] fp4 tensor starts at W + e*N*(K/2), scales at S + e*N*(K/32), bias at
 * b + e*N. Rows are dequantized on the fly (mxfp4_common.h plow_mxfp4_row_dot: f32 per 32-block,
 * block scale once per block). GLU: g/u get their per-expert bias in f32, then the pair-form act
 * (act 3 = swiglu_oai) and one bf16 round. DOWN: part = gate * (dot + bias) in f32, no round.
 * Sentinel (eid >= n_exp): GLU writes nothing, DOWN zeroes part[slot]. Slice partition: gptoss.h. */
#include "gptoss.h"
#include "../mxfp4_common.h"

/* Gate / up row of output n under i4=layout: 0 interleaved (2n, 2n+1), 1 blocked (n, I+n). */
static inline uint32_t glu_row_g(uint32_t n, uint32_t layout) { return layout ? n : 2u * n; }
static inline uint32_t glu_row_u(uint32_t n, uint32_t I, uint32_t layout) {
    return layout ? I + n : 2u * n + 1u;
}

/* t0=fu(bf16 [B*k][I]) t1=x(bf16 [B][K]) t2=table([B*k]) t3=W_gu(fp4 [E][2I][K/2])
 * t4=S_gu(e8m0 [E][2I][K/32]) t5=bias_gu?(bf16 [E][2I])
 * i0=k i1=I i2=K i3=n_exp i4=layout i5=act i6=n_batch(0 = 1)  f0/f1 = act immediates.
 * Slot s reads x row s/k and writes fu row s. */
G_K(g_moe_glu_mx) {
    (void)ctx;
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* tab = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const uint32_t k = in->i[0], I = in->i[1], K = in->i[2], E = in->i[3], layout = in->i[4];
    const uint32_t act = in->i[5], B = in->i[6] ? in->i[6] : 1u;
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    const size_t N2 = 2u * I, ldw = K / 2u, lds = K / PLOW_MX_BLK;
    uint32_t lo, hi;
    g_range(B * k * I, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t s = idx / I, n0 = idx % I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t eid = tab[s].eid;
        if (eid >= E) continue;
        const plow_bf16* xr = x + (size_t)(s / k) * K;
        const uint8_t* We = W + (size_t)eid * N2 * ldw;
        const uint8_t* Se = S + (size_t)eid * N2 * lds;
        const plow_bf16* be = bias ? bias + (size_t)eid * N2 : NULL;
        for (uint32_t n = n0; n < n1; n++) {
            const uint32_t rg = glu_row_g(n, layout), ru = glu_row_u(n, I, layout);
            float g = plow_mxfp4_row_dot(We + (size_t)rg * ldw, Se + (size_t)rg * lds, xr, K);
            float u = plow_mxfp4_row_dot(We + (size_t)ru * ldw, Se + (size_t)ru * lds, xr, K);
            if (be) { g += plow_bf2f(be[rg]); u += plow_bf2f(be[ru]); }
            fu[(size_t)s * I + n] = plow_f2bf(g_glu_pair(g, u, act, f0, f1));
        }
    }
}

/* t0=part(f32 [B*k][H]) t1=fu(bf16 [B*k][I]) t2=table t3=W_d(fp4 [E][H][I/2])
 * t4=S_d(e8m0 [E][H][I/32]) t5=bias_d?(bf16 [E][H])  i0=k i1=H i2=I i3=n_exp i6=n_batch(0 = 1)
 * part[s][h] = gate_s * (W_e[h] . fu[s] + b_e[h]); sentinel slot -> part[s][*] = 0. */
G_K(g_moe_down_mx) {
    (void)ctx;
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const plow_moe_route* tab = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 3);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const uint32_t k = in->i[0], H = in->i[1], I = in->i[2], E = in->i[3];
    const uint32_t B = in->i[6] ? in->i[6] : 1u;
    const size_t ldw = I / 2u, lds = I / PLOW_MX_BLK;
    uint32_t lo, hi;
    g_range(B * k * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t s = idx / H, h0 = idx % H;
        const uint32_t h1 = h0 + (hi - idx) < H ? h0 + (hi - idx) : H;
        idx += h1 - h0;
        float* pr = part + (size_t)s * H;
        const uint32_t eid = tab[s].eid;
        if (eid >= E) {
            for (uint32_t h = h0; h < h1; h++) pr[h] = 0.0f;
            continue;
        }
        const float gate = tab[s].gate;
        const plow_bf16* fr = fu + (size_t)s * I;
        const uint8_t* We = W + (size_t)eid * H * ldw;
        const uint8_t* Se = S + (size_t)eid * H * lds;
        const plow_bf16* be = bias ? bias + (size_t)eid * H : NULL;
        for (uint32_t h = h0; h < h1; h++) {
            float acc = plow_mxfp4_row_dot(We + (size_t)h * ldw, Se + (size_t)h * lds, fr, I);
            if (be) acc += plow_bf2f(be[h]);
            pr[h] = gate * acc;
        }
    }
}

/* PREFILL GLU: t0=fu_g(bf16 [rows][I]) t1=xn2(bf16 [T][K]) t2=W_gu t3=S_gu t4=meta(i32)
 * t5=row_token(u32) t6=bias_gu?  i0=I i1=K i2=n_exp i3=layout i5=act  f0/f1.
 * meta: [0,n_exp) rowoff, [n_exp,2n_exp) cnt. Rows [rowoff[e], rowoff[e]+cnt[e]) belong to
 * expert e; a row whose token is PLOW_EXPERT_UNUSED (padding) is skipped. */
G_K(g_moe_glu_mx_pf) {
    (void)ctx;
    plow_bf16* fu = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 4);
    const uint32_t* row_token = PLOW_CPU_TEN(in, T, 5);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 6);
    const uint32_t I = in->i[0], K = in->i[1], E = in->i[2], layout = in->i[3], act = in->i[5];
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    const size_t N2 = 2u * I, ldw = K / 2u, lds = K / PLOW_MX_BLK;
    uint32_t lo, hi;
    g_range(E * I, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / I, n0 = idx % I;
        const uint32_t n1 = n0 + (hi - idx) < I ? n0 + (hi - idx) : I;
        idx += n1 - n0;
        const uint32_t r0 = (uint32_t)meta[e], cnt = (uint32_t)meta[E + e];
        if (cnt == 0u) continue;
        const uint8_t* We = W + (size_t)e * N2 * ldw;
        const uint8_t* Se = S + (size_t)e * N2 * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * N2 : NULL;
        for (uint32_t r = r0; r < r0 + cnt; r++) {
            const uint32_t tok = row_token[r];
            if (tok == PLOW_EXPERT_UNUSED) continue;
            const plow_bf16* xr = x + (size_t)tok * K;
            for (uint32_t n = n0; n < n1; n++) {
                const uint32_t rg = glu_row_g(n, layout), ru = glu_row_u(n, I, layout);
                float g = plow_mxfp4_row_dot(We + (size_t)rg * ldw, Se + (size_t)rg * lds, xr, K);
                float u = plow_mxfp4_row_dot(We + (size_t)ru * ldw, Se + (size_t)ru * lds, xr, K);
                if (be) { g += plow_bf2f(be[rg]); u += plow_bf2f(be[ru]); }
                fu[(size_t)r * I + n] = plow_f2bf(g_glu_pair(g, u, act, f0, f1));
            }
        }
    }
}

/* PREFILL DOWN: t0=part(f32 [T*k][H]) t1=fu_g t2=W_d t3=S_d t4=meta t5=bias_d?
 * t6=row_partidx(u32) t7=row_gate(f32)  i0=H i1=I i2=n_exp
 * part[row_partidx[r]][h] = row_gate[r] * (W_e[h] . fu_g[r] + b_e[h]); pad rows dropped. */
G_K(g_moe_down_mx_pf) {
    (void)ctx;
    float* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* fu = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* W = PLOW_CPU_TEN(in, T, 2);
    const uint8_t* S = PLOW_CPU_TEN(in, T, 3);
    const int32_t* meta = PLOW_CPU_TEN(in, T, 4);
    const plow_bf16* bias = PLOW_CPU_TEN(in, T, 5);
    const uint32_t* row_partidx = PLOW_CPU_TEN(in, T, 6);
    const float* row_gate = PLOW_CPU_TEN(in, T, 7);
    const uint32_t H = in->i[0], I = in->i[1], E = in->i[2];
    const size_t ldw = I / 2u, lds = I / PLOW_MX_BLK;
    uint32_t lo, hi;
    g_range(E * H, slice, nblk, &lo, &hi);
    for (uint32_t idx = lo; idx < hi;) {
        const uint32_t e = idx / H, h0 = idx % H;
        const uint32_t h1 = h0 + (hi - idx) < H ? h0 + (hi - idx) : H;
        idx += h1 - h0;
        const uint32_t r0 = (uint32_t)meta[e], cnt = (uint32_t)meta[E + e];
        if (cnt == 0u) continue;
        const uint8_t* We = W + (size_t)e * H * ldw;
        const uint8_t* Se = S + (size_t)e * H * lds;
        const plow_bf16* be = bias ? bias + (size_t)e * H : NULL;
        for (uint32_t r = r0; r < r0 + cnt; r++) {
            const uint32_t pidx = row_partidx[r];
            if (pidx == PLOW_EXPERT_UNUSED) continue;
            const plow_bf16* fr = fu + (size_t)r * I;
            float* pr = part + (size_t)pidx * H;
            for (uint32_t h = h0; h < h1; h++) {
                float acc = plow_mxfp4_row_dot(We + (size_t)h * ldw, Se + (size_t)h * lds, fr, I);
                if (be) acc += plow_bf2f(be[h]);
                pr[h] = row_gate[r] * acc;
            }
        }
    }
}
