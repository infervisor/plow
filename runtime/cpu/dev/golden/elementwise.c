/* Pointwise ops: op_elementwise.h ported 1:1 (f32 math, bf16 round on store). */
#include "golden.h"

/* t0=out t1=a t2=b t3=pre?  i0=n  f0=scale.  out = (a + b) * scale, or with pre:
 * out = (pre + bf16(a + b)) * scale — the inner sum is rounded like the packet it replaced. */
G_K(g_residual) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* a = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* b = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* pre = PLOW_CPU_TEN(in, T, 3);
    const float scale = in->fj[0].f;
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) {
        const float s = plow_bf2f(a[i]) + plow_bf2f(b[i]);
        out[i] = pre ? plow_f2bf((plow_bf2f(pre[i]) + plow_bf2f(plow_f2bf(s))) * scale)
                     : plow_f2bf(s * scale);
    }
}

/* t0=out t1=gate t2=up  i0=n i1=act  f0/f1 = the act's immediates (act 3 swiglu_oai: alpha, limit) */
G_K(g_glu) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* gate = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* up = PLOW_CPU_TEN(in, T, 2);
    const uint32_t act = in->i[1];
    const float f0 = in->fj[0].f, f1 = in->fj[1].f;
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++)
        out[i] = plow_f2bf(g_glu_pair(plow_bf2f(gate[i]), plow_bf2f(up[i]), act, f0, f1));
}

/* t0=out t1=x  i0=n  f0=cap.  out = cap * tanh(x / cap). */
G_K(g_softcap) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const float cap = in->fj[0].f, inv = 1.0f / cap;
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) out[i] = plow_f2bf(cap * tanhf(plow_bf2f(x[i]) * inv));
}

/* t0=out t1=table t2=ids(i32)  i0=ntok i1=hidden  f0=scale (bf16-rounded sqrt(hidden)). */
G_K(g_embed) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* table = PLOW_CPU_TEN(in, T, 1);
    const int32_t* ids = PLOW_CPU_TEN(in, T, 2);
    const uint32_t ntok = in->i[0], hidden = in->i[1];
    const float scale = in->fj[0].f;
    for (uint32_t t = slice; t < ntok; t += nblk) {
        const plow_bf16* src = table + (size_t)ids[t] * hidden;
        plow_bf16* dst = out + (size_t)t * hidden;
        for (uint32_t i = 0; i < hidden; i++) dst[i] = plow_f2bf(plow_bf2f(src[i]) * scale);
    }
}

/* t0=part(u64[n_batch][nblk]) t1=x  i0=n i1=n_batch(0/1 = one sequence).
 * Each slice reduces its share of every row to one packed key; the max is partition-independent. */
G_K(g_argmax) {
    (void)ctx;
    uint64_t* part = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t n = in->i[0], B = in->i[1] ? in->i[1] : 1u;
    uint32_t lo, hi;
    g_range(n, slice, nblk, &lo, &hi);
    for (uint32_t b = 0; b < B; b++) {
        const plow_bf16* xb = x + (size_t)b * n;
        uint64_t best = 0;
        for (uint32_t i = lo; i < hi; i++) {
            const uint64_t p = g_amax_pack(xb[i], i);
            best = p > best ? p : best;
        }
        part[(size_t)b * nblk + slice] = best;
    }
}

/* t0=ids(i32) t1=part  i0=nparts i1=n_batch.  Slice 0 folds the partials. */
G_K(g_argmax_fin) {
    (void)ctx; (void)nblk;
    if (slice != 0) return;
    int32_t* ids = PLOW_CPU_TEN(in, T, 0);
    const uint64_t* part = PLOW_CPU_TEN(in, T, 1);
    const uint32_t nparts = in->i[0], B = in->i[1] ? in->i[1] : 1u;
    for (uint32_t b = 0; b < B; b++) {
        const uint64_t* pb = part + (size_t)b * nparts;
        uint64_t best = 0;
        for (uint32_t i = 0; i < nparts; i++) best = pb[i] > best ? pb[i] : best;
        ids[b] = (int32_t) ~(uint32_t)(best & 0xFFFFFFFFull);
    }
}
