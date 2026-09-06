/* MoE routing ops (83 router top-k, 84 align, 87 combine) vs naive f64 references at GPT-OSS
 * geometry (E=32, k=4, H=2880) and a GLM-like one (E=256, k=8, sigmoid + bias). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpu_dev.h"
#include "golden/gptoss.h"

static uint64_t rs = 0x243F6A8885A308D3ull;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static double nr(void) { return sqrt(-2 * log(ur() + 1e-12)) * cos(6.283185307 * ur()); }
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static void run_op(plow_cpu_kernel_fn f, PlowDevInst* in, void** T, uint32_t nblk, PlowCpuCtx* ctx) {
    in->blocks = (uint16_t)nblk;
    for (uint32_t s = 0; s < nblk; s++) f(in, s, nblk, T, ctx);
}

static void test_route(uint32_t E, uint32_t k, uint32_t T, uint32_t flags, int with_bias, uint32_t nblk, PlowCpuCtx* ctx) {
    plow_bf16* logit = malloc((size_t)T * E * 2);
    float* bias = malloc(E * 4);
    plow_moe_route* tab = calloc((size_t)T * k, sizeof(plow_moe_route));
    for (size_t i = 0; i < (size_t)T * E; i++) logit[i] = plow_f2bf((float)(nr() * 2.0));
    for (uint32_t e = 0; e < E; e++) bias[e] = (float)(nr() * 0.05);
    void* Tt[4] = {tab, logit, NULL, with_bias ? bias : NULL};
    PlowDevInst in; memset(&in, 0, sizeof in);
    in.op = PLOW_DOP_MOE_ROUTER_TOPK_PF; in.t[0] = 0; in.t[1] = 1; in.t[2] = PLOW_TENSOR_NONE; in.t[3] = with_bias ? 3 : PLOW_TENSOR_NONE;
    for (int i = 4; i < 8; i++) in.t[i] = PLOW_TENSOR_NONE;
    in.i[1] = E; in.i[2] = k; in.i[3] = flags; in.i[4] = T; in.fj[0].f = 1.0f;
    run_op(plow_cpu_kernel(PLOW_DOP_MOE_ROUTER_TOPK_PF), &in, Tt, nblk, ctx);
    double* sc = malloc(E * 8);
    for (uint32_t t = 0; t < T; t++) {
        const plow_bf16* lg = logit + (size_t)t * E;
        if (flags & 1) { for (uint32_t e = 0; e < E; e++) sc[e] = 1.0 / (1.0 + exp(-(double)plow_bf2f(lg[e]))); }
        else { double m = -1e300, s = 0; for (uint32_t e = 0; e < E; e++) { sc[e] = plow_bf2f(lg[e]); if (sc[e] > m) m = sc[e]; }
               for (uint32_t e = 0; e < E; e++) { sc[e] = exp(sc[e] - m); s += sc[e]; } for (uint32_t e = 0; e < E; e++) sc[e] /= s; }
        /* reference top-k by biased score, lower id wins ties */
        uint32_t used[256] = {0}; double gsum = 0; uint32_t win[16];
        for (uint32_t j = 0; j < k; j++) {
            int best = -1; double bv = -1e300;
            for (uint32_t e = 0; e < E; e++) { if (used[e]) continue; double v = sc[e] + (with_bias ? bias[e] : 0.0); if (v > bv) { bv = v; best = (int)e; } }
            used[best] = 1; win[j] = (uint32_t)best; gsum += sc[best];
        }
        const plow_moe_route* row = tab + (size_t)t * k;
        for (uint32_t j = 0; j < k; j++) {
            CHECK(row[j].eid == win[j], "route E=%u k=%u tok %u slot %u: eid %u want %u", E, k, t, j, row[j].eid, win[j]);
            double g = sc[win[j]]; if (flags & 2) g /= gsum;
            CHECK(fabs(row[j].gate - g) <= 1e-5 + 1e-3 * fabs(g), "route gate tok %u slot %u: %g want %g", t, j, row[j].gate, g);
        }
    }
    /* align */
    int32_t* meta = calloc(3u * E + 1u, 4);
    uint32_t rows_max = T * k + E * 63u;
    uint32_t* row_token = malloc(rows_max * 4); uint32_t* row_partidx = malloc(rows_max * 4); float* row_gate = malloc(rows_max * 4);
    void* Ta[5] = {meta, tab, row_token, row_partidx, row_gate};
    PlowDevInst ia; memset(&ia, 0, sizeof ia);
    ia.op = PLOW_DOP_MOE_ALIGN_PF; for (int i = 0; i < 8; i++) ia.t[i] = i < 5 ? (uint16_t)i : PLOW_TENSOR_NONE;
    ia.i[0] = T; ia.i[1] = E; ia.i[2] = k;
    run_op(plow_cpu_kernel(PLOW_DOP_MOE_ALIGN_PF), &ia, Ta, nblk, ctx);
    uint32_t seen = 0, off = 0, tiles = 0;
    for (uint32_t e = 0; e < E; e++) {
        uint32_t cnt = 0; for (uint32_t s = 0; s < T * k; s++) cnt += tab[s].eid == e;
        CHECK((uint32_t)meta[E + e] == cnt, "align cnt[%u] %d want %u", e, meta[E + e], cnt);
        CHECK((uint32_t)meta[e] == off, "align rowoff[%u] %d want %u", e, meta[e], off);
        CHECK((uint32_t)meta[2 * E + e] == tiles, "align tilep[%u]", e);
        for (uint32_t r = 0; r < cnt; r++) {
            uint32_t s = row_partidx[off + r];
            CHECK(s < T * k && tab[s].eid == e && row_token[off + r] == s / k && row_gate[off + r] == tab[s].gate,
                  "align row %u of expert %u inconsistent", r, e);
            CHECK(r == 0 || row_partidx[off + r - 1] < s, "align rows of expert %u not in slot order", e);
            seen++;
        }
        uint32_t t = (cnt + 63u) / 64u;
        for (uint32_t r = cnt; r < t * 64u; r++) CHECK(row_partidx[off + r] == PLOW_EXPERT_UNUSED, "pad row not marked");
        tiles += t; off += t * 64u;
    }
    CHECK(seen == T * k, "align: %u rows for %u slots", seen, T * k);
    CHECK((uint32_t)meta[3 * E] == tiles, "align tilep[E]");
    /* combine */
    const uint32_t H = 96;
    float* part = malloc((size_t)T * k * H * 4); plow_bf16* resid = malloc((size_t)T * H * 2); plow_bf16* out = malloc((size_t)T * H * 2);
    for (size_t i = 0; i < (size_t)T * k * H; i++) part[i] = (float)nr();
    for (size_t i = 0; i < (size_t)T * H; i++) resid[i] = plow_f2bf((float)nr());
    void* Tc[4] = {out, resid, NULL, part};
    PlowDevInst ic; memset(&ic, 0, sizeof ic);
    ic.op = PLOW_DOP_MOE_COMBINE_PF; ic.t[0] = 0; ic.t[1] = 1; ic.t[2] = PLOW_TENSOR_NONE; ic.t[3] = 3; for (int i = 4; i < 8; i++) ic.t[i] = PLOW_TENSOR_NONE;
    ic.i[0] = H; ic.i[1] = k; ic.i[2] = T;
    for (int live = 0; live < 2; live++) {
        memset(out, 0, (size_t)T * H * 2);
        run_op(live ? plow_cpu_kernel(PLOW_DOP_MOE_COMBINE_PF) : g_moe_combine_pf, &ic, Tc, nblk, ctx);
        for (uint32_t t = 0; t < T; t++)
            for (uint32_t h = 0; h < H; h++) {
                double r = plow_bf2f(resid[(size_t)t * H + h]);
                for (uint32_t j = 0; j < k; j++) r += part[((size_t)t * k + j) * H + h];
                double got = plow_bf2f(out[(size_t)t * H + h]);
                CHECK(fabs(got - r) <= 1e-2 * fmax(1.0, fabs(r)), "combine%s tok %u h %u: %g want %g",
                      live ? "" : " (golden)", t, h, got, r);
            }
    }
    free(logit); free(bias); free(tab); free(sc); free(meta); free(row_token); free(row_partidx); free(row_gate); free(part); free(resid); free(out);
}

int main(void) {
    /* The live table: top-k and align have no fast arm and resolve to golden, combine does.
     * The golden combine is checked against the same reference inside test_route. */
    plow_cpu_init(PLOW_CPU_ISA_AMX);
    PlowCpuCtx ctx; memset(&ctx, 0, sizeof ctx);
    ctx.scratch_bytes = plow_cpu_scratch_bytes(); ctx.scratch = aligned_alloc(64, ctx.scratch_bytes);
    plow_cpu_thread_init(&ctx);
    for (uint32_t nblk = 1; nblk <= 16; nblk *= 4) {
        test_route(32, 4, 1, 2, 0, nblk, &ctx);      /* GPT-OSS decode: softmax + norm_topk */
        test_route(32, 4, 37, 2, 0, nblk, &ctx);     /* GPT-OSS prefill */
        test_route(256, 8, 5, 1, 1, nblk, &ctx);     /* GLM-style: sigmoid + selection bias */
        test_route(64, 2, 129, 0, 0, nblk, &ctx);    /* plain softmax, unnormalised */
    }
    printf(fails ? "cpu_dev_route: %d failures\n" : "cpu_dev_route: all passed\n", fails);
    return fails ? 1 : 0;
}
