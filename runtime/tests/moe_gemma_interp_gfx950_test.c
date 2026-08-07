/* moe_gemma_interp_gfx950_test.c — the Gemma-4 MoE DECODE block through the PERSISTENT
 * interpreter, i.e. through the dispatch ARMS, not just the op bodies. [GEMMA4-MOE-AMD]
 *
 * WHY THIS EXISTS SEPARATELY FROM moe_gemma_gfx950_test.c. That test drives the __device__
 * bodies through standalone wrappers, so it proves the ARITHMETIC. It cannot prove the half of
 * the port that is `case PLOW_DOP_...:` — which tensor handle feeds which parameter, which `i[]`
 * slot carries k versus n_exp, whether the slot offset in i[4] is applied. Get that wrong and
 * every body is correct and the model is not.
 *
 * And on AMD the failure is SILENT: the dispatch `default:` writes nothing (it does not trap,
 * unlike sm_120), so an opcode with no arm — the state all nineteen of these were in — leaves an
 * untouched buffer that reads as a result. Every output tensor here is therefore POISONED with
 * bf16 NaN before the launch. An arm that never ran leaves NaN, the reference comparison is
 * NaN-safe (a naive `if (rel > worst)` is FALSE for NaN and prints PASS), and the op fails.
 *
 * The program is one counter-chained stream, ONE launch:
 *
 *   ROUTER_GEMMA_SCORE -> ROUTER_GEMMA_TOPK -> ROUTER_GEMMA (fused, cross-checked)
 *     -> EXPERT_GLU_GEMMA -> EXPERT_GLU_NORM_GEMMA -> EXPERT_DOWN_GEMMA
 *     -> COMBINE_GEMMA -> COMBINE_NORM_GEMMA -> COMBINE_RESID_NORM_GEMMA
 *
 * Truth is the same CPU reference discipline: transcribed from runtime/nvidia/op_moe.cuh.
 *
 * Build:
 *   cc -O2 -std=gnu11 -o t_moe_gemma_interp runtime/tests/moe_gemma_interp_gfx950_test.c \
 *      runtime/amd/hsa_backend.c -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
 * Run:  ./t_moe_gemma_interp <interp_decode.elf built with -DPLOW_MOE_GEMMA=1>
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H      2816u
#define I_MOE   704u
#define N_EXP   128u
#define TOPK      8u
#define EPS   1e-6f
#define LS    0.7f

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static uint64_t rs = 0x243F6A8885A308D3ull;
static uint32_t r32(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static float frand(void) { return ((float)(r32() % 4001u) - 2000.0f) / 8000.0f; }
static float gelu_tanh(float x) {
    const float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

static int fails = 0, checks = 0;
static double relerr(double g, double w) {
    if (isnan(g) || isnan(w)) return INFINITY; /* NaN must FAIL, not score 0 */
    return fabs(g - w) / (fabs(w) + 1e-3);
}
static void check_bf16(const char* what, const bf16* got, const float* want, size_t n, double tol) {
    double worst = 0.0; size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = relerr(bf2f(got[i]), want[i]);
        if (!(d <= worst)) { worst = d; at = i; }
    }
    checks++;
    const int ok = worst < tol;
    printf("  %-36s %s  (worst rel %.3e)\n", what, ok ? "PASS" : "FAIL", worst);
    if (!ok) { printf("      [%zu] got %.6g want %.6g\n", at, bf2f(got[at]), want[at]); fails++; }
}
static void check_f32(const char* what, const float* got, const float* want, size_t n, double tol) {
    double worst = 0.0; size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = relerr(got[i], want[i]);
        if (!(d <= worst)) { worst = d; at = i; }
    }
    checks++;
    const int ok = worst < tol;
    printf("  %-36s %s  (worst rel %.3e)\n", what, ok ? "PASS" : "FAIL", worst);
    if (!ok) { printf("      [%zu] got %.6g want %.6g\n", at, got[at], want[at]); fails++; }
}

/* tensor handles */
enum { T_RESID = 0, T_PROJ, T_SCALE, T_PES, T_SCORE, T_TABLE, T_TABLE2, T_X, T_EWT, T_GAMMA,
       T_FU, T_FU2, T_PART, T_MOE, T_OUT, T_HN, T_XRES, T_H1, T_GPO, T_GN, T_N };
#define NOPS 9u

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char* elf = argc > 1 ? argv[1] : "interp_decode.elf";
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, ldsb = 0;
    plow_hsa_device_info(h, 0, nm, &cus, &ldsb);
    printf("dev0: %s  CUs=%u  LDS=%u B\n", nm, cus, ldsb);

    FILE* f = fopen(elf, "rb");
    if (!f) { perror(elf); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc((size_t)co_n);
    if (fread(co, 1, (size_t)co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, (size_t)co_n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }
    /* Same rule as crates/plowrt/src/exec/amd.rs: the symbol is derived from the LIVE agent
     * name, so this runs against whatever part it is pointed at. */
    char sym[96];
    snprintf(sym, sizeof(sym), "plow_interp_dec_%s", nm);
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, sym, &k) != 0) {
        fprintf(stderr, "symbol %s: %s\n", sym, plow_hsa_last_error());
        fprintf(stderr, "  (is this object built with -DPLOW_BUCKET_DECODE=1?)\n");
        return 1;
    }
    const unsigned NCU = cus;
    printf("interpreter %s resolved (kernarg=%u B, LDS=%u B)\n\n", sym, k.kernarg_size,
           k.group_segment_size);

    /* ---- inputs ---- */
    bf16* resid = malloc(H * 2);
    bf16* proj = malloc((size_t)N_EXP * H * 2);
    bf16* scal = malloc(H * 2);
    bf16* pes = malloc(N_EXP * 2);
    bf16* xin = malloc(H * 2);
    bf16* gam = malloc(H * 2);
    bf16* h1 = malloc(H * 2);
    bf16* gpo = malloc(H * 2);
    bf16* gn = malloc(H * 2);
    bf16* xres = malloc(H * 2);
    for (unsigned i = 0; i < H; i++) {
        resid[i] = f2bf(frand()); scal[i] = f2bf(1.0f + frand()); xin[i] = f2bf(frand());
        gam[i] = f2bf(1.0f + frand()); h1[i] = f2bf(frand()); gpo[i] = f2bf(1.0f + frand());
        gn[i] = f2bf(1.0f + frand()); xres[i] = f2bf(frand());
    }
    for (size_t i = 0; i < (size_t)N_EXP * H; i++) proj[i] = f2bf(frand());
    for (unsigned i = 0; i < N_EXP; i++) pes[i] = f2bf(1.0f + frand());
    const float root = 1.0f / sqrtf((float)H);

    /* ---- CPU reference: router ---- */
    float* h2 = malloc(H * sizeof(float));
    float* sc = malloc(N_EXP * sizeof(float));
    {
        double ss = 0.0;
        for (unsigned i = 0; i < H; i++) { const float v = bf2f(resid[i]); ss += (double)v * v; }
        const float inv = 1.0f / sqrtf((float)(ss / (double)H) + EPS);
        for (unsigned i = 0; i < H; i++) h2[i] = bf2f(resid[i]) * inv * bf2f(scal[i]) * root;
        for (unsigned e = 0; e < N_EXP; e++) {
            float acc = 0.0f;
            const bf16* pr = proj + (size_t)e * H;
            for (unsigned i = 0; i < H; i++) acc = fmaf(h2[i], bf2f(pr[i]), acc);
            sc[e] = acc;
        }
    }
    unsigned rid[TOPK]; float rga[TOPK];
    {
        float* p = malloc(N_EXP * sizeof(float));
        memcpy(p, sc, N_EXP * sizeof(float));
        float m = -1e30f;
        for (unsigned e = 0; e < N_EXP; e++) m = fmaxf(m, p[e]);
        float s = 0.0f;
        for (unsigned e = 0; e < N_EXP; e++) { p[e] = expf(p[e] - m); s += p[e]; }
        for (unsigned e = 0; e < N_EXP; e++) p[e] /= s;
        for (unsigned j = 0; j < TOPK; j++) {
            unsigned long long best = 0ull; unsigned bid = 0;
            for (unsigned e = 0; e < N_EXP; e++) {
                unsigned sb; const float v = p[e]; memcpy(&sb, &v, 4);
                sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
                const unsigned long long key = ((unsigned long long)sb << 20) |
                                               (unsigned long long)((N_EXP - 1u - e) & 0xFFFFFu);
                if (key > best) { best = key; bid = e; }
            }
            rid[j] = bid; rga[j] = p[bid]; p[bid] = -1e30f;
        }
        float gs = 0.0f;
        for (unsigned j = 0; j < TOPK; j++) gs += rga[j];
        for (unsigned j = 0; j < TOPK; j++) { if (gs != 0.0f) rga[j] /= gs; rga[j] *= bf2f(pes[rid[j]]); }
        free(p);
    }

    /* ---- expert weights for the routed 8 only (the rest of ewt is the EP null) ---- */
    const size_t GU_N = (size_t)2 * I_MOE * H, DN_N = (size_t)H * I_MOE;
    bf16** gu = calloc(N_EXP, sizeof(bf16*));
    bf16** dn = calloc(N_EXP, sizeof(bf16*));
    unsigned long long* ewt = calloc(N_EXP * 2, sizeof(unsigned long long));
    for (unsigned j = 0; j < TOPK; j++) {
        const unsigned e = rid[j];
        gu[e] = malloc(GU_N * 2); dn[e] = malloc(DN_N * 2);
        for (size_t i = 0; i < GU_N; i++) gu[e][i] = f2bf(frand());
        for (size_t i = 0; i < DN_N; i++) dn[e][i] = f2bf(frand());
        void* a = plow_hsa_alloc(h, 0, GU_N * 2);
        void* b = plow_hsa_alloc(h, 0, DN_N * 2);
        plow_hsa_upload(h, 0, a, gu[e], GU_N * 2);
        plow_hsa_upload(h, 0, b, dn[e], DN_N * 2);
        ewt[(size_t)e * 2 + 0] = (unsigned long long)(size_t)a;
        ewt[(size_t)e * 2 + 1] = (unsigned long long)(size_t)b;
    }

    /* ---- device tensors; every OUTPUT is poisoned with bf16 NaN ---- */
    void* T[T_N];
    const size_t FU_N = (size_t)TOPK * I_MOE, PART_N = (size_t)TOPK * H;
#define MK(idx, bytes) T[idx] = plow_hsa_alloc(h, 0, (bytes))
    MK(T_RESID, H * 2); MK(T_PROJ, (size_t)N_EXP * H * 2); MK(T_SCALE, H * 2);
    MK(T_PES, N_EXP * 2); MK(T_SCORE, N_EXP * 4); MK(T_TABLE, TOPK * 8); MK(T_TABLE2, TOPK * 8);
    MK(T_X, H * 2); MK(T_EWT, N_EXP * 2 * 8); MK(T_GAMMA, H * 2);
    MK(T_FU, FU_N * 2); MK(T_FU2, FU_N * 2); MK(T_PART, PART_N * 4); MK(T_MOE, H * 2);
    MK(T_OUT, H * 2); MK(T_HN, H * 2); MK(T_XRES, H * 2); MK(T_H1, H * 2);
    MK(T_GPO, H * 2); MK(T_GN, H * 2);
#undef MK
    plow_hsa_upload(h, 0, T[T_RESID], resid, H * 2);
    plow_hsa_upload(h, 0, T[T_PROJ], proj, (size_t)N_EXP * H * 2);
    plow_hsa_upload(h, 0, T[T_SCALE], scal, H * 2);
    plow_hsa_upload(h, 0, T[T_PES], pes, N_EXP * 2);
    plow_hsa_upload(h, 0, T[T_X], xin, H * 2);
    plow_hsa_upload(h, 0, T[T_EWT], ewt, N_EXP * 2 * 8);
    plow_hsa_upload(h, 0, T[T_GAMMA], gam, H * 2);
    plow_hsa_upload(h, 0, T[T_H1], h1, H * 2);
    plow_hsa_upload(h, 0, T[T_GPO], gpo, H * 2);
    plow_hsa_upload(h, 0, T[T_GN], gn, H * 2);
    plow_hsa_upload(h, 0, T[T_XRES], xres, H * 2);
    {
        /* 0x7FC0 is bf16 qNaN; the same word read as f32 is a NaN too, so one filler poisons
         * both the bf16 and the f32 outputs. */
        const size_t big = PART_N * 4;
        unsigned* p = malloc(big);
        for (size_t i = 0; i < big / 4; i++) p[i] = 0x7FC07FC0u;
        const int outs[] = { T_SCORE, T_TABLE, T_TABLE2, T_FU, T_FU2, T_PART, T_MOE, T_OUT,
                             T_HN };
        const size_t sz[] = { N_EXP * 4, TOPK * 8, TOPK * 8, FU_N * 2, FU_N * 2, PART_N * 4,
                              H * 2, H * 2, H * 2 };
        for (unsigned i = 0; i < sizeof(outs) / sizeof(outs[0]); i++)
            plow_hsa_upload(h, 0, T[outs[i]], p, sz[i]);
        free(p);
    }
    void* d_tensors = plow_hsa_alloc(h, 0, sizeof(T));
    plow_hsa_upload(h, 0, d_tensors, T, sizeof(T));

    /* ---- the program ---- */
    PlowDevInst in[NOPS];
    memset(in, 0, sizeof(in));
    /* ZERO IS A VALID TENSOR HANDLE (it is T_RESID here). The absent-operand sentinel is
     * PLOW_TENSOR_NONE — memset leaves 0, which silently binds a real tensor. */
    for (unsigned i = 0; i < NOPS; i++) {
        for (unsigned j = 0; j < 8; j++) in[i].t[j] = PLOW_TENSOR_NONE;
        in[i].blocks = (uint16_t)NCU;
    }
    unsigned n = 0;
    /* op67 SCORE: t0=score t1=resid t2=proj t3=scale; i0=H i1=n_exp i2=B; f0=root f1=eps */
    in[n].op = PLOW_DOP_MOE_ROUTER_GEMMA_SCORE;
    in[n].t[0] = T_SCORE; in[n].t[1] = T_RESID; in[n].t[2] = T_PROJ; in[n].t[3] = T_SCALE;
    in[n].i[0] = H; in[n].i[1] = N_EXP; in[n].fj[0].f = root; in[n].fj[1].f = EPS; n++;
    /* op68 TOPK: t0=table t1=score t2=pes; i1=n_exp i2=k i3=B */
    in[n].op = PLOW_DOP_MOE_ROUTER_GEMMA_TOPK;
    in[n].t[0] = T_TABLE; in[n].t[1] = T_SCORE; in[n].t[2] = T_PES;
    in[n].i[1] = N_EXP; in[n].i[2] = TOPK; n++;
    /* op61 ROUTER (fused): t0=table t1=resid t2=proj t3=scale t4=pes */
    in[n].op = PLOW_DOP_MOE_ROUTER_GEMMA;
    in[n].t[0] = T_TABLE2; in[n].t[1] = T_RESID; in[n].t[2] = T_PROJ; in[n].t[3] = T_SCALE;
    in[n].t[4] = T_PES;
    in[n].i[0] = H; in[n].i[1] = N_EXP; in[n].i[2] = TOPK;
    in[n].fj[0].f = root; in[n].fj[1].f = EPS; n++;
    /* op62 GLU: t0=fu t1=x t2=table t3=ewt; i0=k i1=I i2=H i3=n_exp i4=slot_off i5=B */
    in[n].op = PLOW_DOP_MOE_EXPERT_GLU_GEMMA;
    in[n].t[0] = T_FU; in[n].t[1] = T_X; in[n].t[2] = T_TABLE; in[n].t[3] = T_EWT;
    in[n].i[0] = TOPK; in[n].i[1] = I_MOE; in[n].i[2] = H; in[n].i[3] = N_EXP; n++;
    /* op71 GLU_NORM: t0=fu t1=resid t2=table t3=ewt t4=gamma; i0=k i1=I i2=H i3=n_exp; f0=eps */
    in[n].op = PLOW_DOP_MOE_EXPERT_GLU_NORM_GEMMA;
    in[n].t[0] = T_FU2; in[n].t[1] = T_RESID; in[n].t[2] = T_TABLE; in[n].t[3] = T_EWT;
    in[n].t[4] = T_GAMMA;
    in[n].i[0] = TOPK; in[n].i[1] = I_MOE; in[n].i[2] = H; in[n].i[3] = N_EXP;
    in[n].fj[0].f = EPS; n++;
    /* op63 DOWN: t0=part t1=fu t2=table t3=ewt; i0=k i1=H i2=I i3=n_exp */
    in[n].op = PLOW_DOP_MOE_EXPERT_DOWN_GEMMA;
    in[n].t[0] = T_PART; in[n].t[1] = T_FU; in[n].t[2] = T_TABLE; in[n].t[3] = T_EWT;
    in[n].i[0] = TOPK; in[n].i[1] = H; in[n].i[2] = I_MOE; in[n].i[3] = N_EXP; n++;
    /* op64 COMBINE: t0=moe t1=part; i0=H i1=k */
    in[n].op = PLOW_DOP_MOE_COMBINE_GEMMA;
    in[n].t[0] = T_MOE; in[n].t[1] = T_PART; in[n].i[0] = H; in[n].i[1] = TOPK; n++;
    /* op70 COMBINE_NORM: t0=out t1=part t2=resid t3=gamma; i0=H i1=k i2=B; f0=eps */
    in[n].op = PLOW_DOP_MOE_COMBINE_NORM_GEMMA;
    in[n].t[0] = T_OUT; in[n].t[1] = T_PART; in[n].t[2] = T_RESID; in[n].t[3] = T_GAMMA;
    in[n].i[0] = H; in[n].i[1] = TOPK; in[n].fj[0].f = EPS; n++;
    /* op72 CRN: t0=hn t1=x t2=part t3=h1 t4=g_pf2 t5=g_po t6=gn; i0=H i1=k; f0=eps f1=ls */
    in[n].op = PLOW_DOP_MOE_COMBINE_RESID_NORM_GEMMA;
    in[n].t[0] = T_HN; in[n].t[1] = T_XRES; in[n].t[2] = T_PART; in[n].t[3] = T_H1;
    in[n].t[4] = T_GAMMA; in[n].t[5] = T_GPO; in[n].t[6] = T_GN;
    in[n].i[0] = H; in[n].i[1] = TOPK; in[n].fj[0].f = EPS; in[n].fj[1].f = LS; n++;

    PlowWait waits[NOPS - 1];
    uint32_t succs[NOPS - 1];
    for (unsigned i = 0; i + 1 < NOPS; i++) { waits[i].id = i; waits[i].threshold = NCU; succs[i] = i; }

    PlowStreamEnt* st = calloc((size_t)NCU * NOPS, sizeof(PlowStreamEnt));
    uint32_t* sofs = malloc(4 * NCU);
    uint32_t* slen = malloc(4 * NCU);
    for (unsigned cu = 0; cu < NCU; cu++) {
        sofs[cu] = cu * NOPS; slen[cu] = NOPS;
        for (unsigned i = 0; i < NOPS; i++) {
            PlowStreamEnt* e = &st[(size_t)cu * NOPS + i];
            e->inst = (uint16_t)i;
            e->slice = (uint16_t)cu;
            if (i > 0) { e->wait_len = 1; e->wait_ofs = (uint16_t)(i - 1); }
            if (i + 1 < NOPS) { e->succ_len = 1; e->succ_ofs = (uint16_t)i; }
        }
    }
    uint32_t* ctr = calloc((NOPS - 1) * PLOW_CTR_STRIDE, 4);

    void* d_in = plow_hsa_alloc(h, 0, sizeof(in));
    void* d_st = plow_hsa_alloc(h, 0, sizeof(PlowStreamEnt) * NCU * NOPS);
    void* d_so = plow_hsa_alloc(h, 0, 4 * NCU);
    void* d_sl = plow_hsa_alloc(h, 0, 4 * NCU);
    void* d_wa = plow_hsa_alloc(h, 0, sizeof(waits));
    void* d_su = plow_hsa_alloc(h, 0, sizeof(succs));
    void* d_ct = plow_hsa_alloc(h, 0, (NOPS - 1) * PLOW_CTR_STRIDE * 4);
    plow_hsa_upload(h, 0, d_in, in, sizeof(in));
    plow_hsa_upload(h, 0, d_st, st, sizeof(PlowStreamEnt) * NCU * NOPS);
    plow_hsa_upload(h, 0, d_so, sofs, 4 * NCU);
    plow_hsa_upload(h, 0, d_sl, slen, 4 * NCU);
    plow_hsa_upload(h, 0, d_wa, waits, sizeof(waits));
    plow_hsa_upload(h, 0, d_su, succs, sizeof(succs));
    plow_hsa_upload(h, 0, d_ct, ctr, (NOPS - 1) * PLOW_CTR_STRIDE * 4);

    PlowProgram prog;
    memset(&prog, 0, sizeof(prog));
    prog.insts = (const PlowDevInst*)d_in;
    prog.stream = (const PlowStreamEnt*)d_st;
    prog.stream_ofs = (const uint32_t*)d_so;
    prog.stream_len = (const uint32_t*)d_sl;
    prog.waits = (const PlowWait*)d_wa;
    prog.succs = (const uint32_t*)d_su;
    prog.counters = (uint32_t*)d_ct;
    prog.tensors = (void* const*)d_tensors;

    printf("launching: 1 kernel, %u workgroups, %u-op Gemma MoE decode block\n", NCU, NOPS);
    if (plow_hsa_launch(h, 0, &k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &prog,
                        sizeof(prog)) != 0) {
        fprintf(stderr, "launch: %s\n", plow_hsa_last_error()); return 1;
    }
    if (plow_hsa_wait(h, 0) != 0) { fprintf(stderr, "wait failed\n"); return 1; }
    plow_hsa_download(h, 0, ctr, d_ct, (NOPS - 1) * PLOW_CTR_STRIDE * 4);
    int ctr_ok = 1;
    for (unsigned i = 0; i + 1 < NOPS; i++) if (ctr[i * PLOW_CTR_STRIDE] != NCU) ctr_ok = 0;
    printf("counters: %s (expect %u on each of %u gates)\n\n", ctr_ok ? "PASS" : "FAIL", NCU,
           NOPS - 1);
    checks++;
    if (!ctr_ok) fails++;

    /* ---- readback + reference ---- */
    float* g_sc = malloc(N_EXP * 4);
    unsigned char tb[TOPK * 8], tb2[TOPK * 8];
    bf16* g_fu = malloc(FU_N * 2);
    bf16* g_fu2 = malloc(FU_N * 2);
    float* g_part = malloc(PART_N * 4);
    bf16* g_moe = malloc(H * 2);
    bf16* g_out = malloc(H * 2);
    bf16* g_hn = malloc(H * 2);
    bf16* g_x = malloc(H * 2);
    plow_hsa_download(h, 0, g_sc, T[T_SCORE], N_EXP * 4);
    plow_hsa_download(h, 0, tb, T[T_TABLE], sizeof(tb));
    plow_hsa_download(h, 0, tb2, T[T_TABLE2], sizeof(tb2));
    plow_hsa_download(h, 0, g_fu, T[T_FU], FU_N * 2);
    plow_hsa_download(h, 0, g_fu2, T[T_FU2], FU_N * 2);
    plow_hsa_download(h, 0, g_part, T[T_PART], PART_N * 4);
    plow_hsa_download(h, 0, g_moe, T[T_MOE], H * 2);
    plow_hsa_download(h, 0, g_out, T[T_OUT], H * 2);
    plow_hsa_download(h, 0, g_hn, T[T_HN], H * 2);
    plow_hsa_download(h, 0, g_x, T[T_XRES], H * 2);

    /* 1e-4, not tighter: the exact scorer's fmaf chain matches element for element, but
     * `invrms` is a block reduction on device and a sequential sum here, so every h2 term
     * carries one f32 ulp of that scalar. See the note in moe_gemma_gfx950_test.c. */
    check_f32("op67 SCORE (arm)", g_sc, sc, N_EXP, 1e-4);
    {
        unsigned gi[TOPK], gi2[TOPK]; float gg[TOPK], gg2[TOPK];
        for (unsigned j = 0; j < TOPK; j++) {
            memcpy(&gi[j], tb + (size_t)j * 8, 4); memcpy(&gg[j], tb + (size_t)j * 8 + 4, 4);
            memcpy(&gi2[j], tb2 + (size_t)j * 8, 4); memcpy(&gg2[j], tb2 + (size_t)j * 8 + 4, 4);
        }
        size_t bad = 0, bad2 = 0;
        for (unsigned j = 0; j < TOPK; j++) { if (gi[j] != rid[j]) bad++; if (gi2[j] != rid[j]) bad2++; }
        checks += 2;
        printf("  %-36s %s  (%zu/%u ids wrong)\n", "op68 TOPK ids (arm)", bad ? "FAIL" : "PASS",
               bad, TOPK);
        if (bad) fails++;
        printf("  %-36s %s  (%zu/%u ids wrong)\n", "op61 ROUTER fused ids (arm)",
               bad2 ? "FAIL" : "PASS", bad2, TOPK);
        if (bad2) fails++;
        check_f32("op68 TOPK gates (arm)", gg, rga, TOPK, 2e-3);
        check_f32("op61 ROUTER fused gates (arm)", gg2, rga, TOPK, 2e-3);
    }
    {
        float* want = malloc(FU_N * sizeof(float));
        float* want2 = malloc(FU_N * sizeof(float));
        double ss = 0.0;
        for (unsigned i = 0; i < H; i++) { const float v = bf2f(resid[i]); ss += (double)v * v; }
        const float inv = 1.0f / sqrtf((float)(ss / (double)H) + EPS);
        float* xn = malloc(H * sizeof(float));
        for (unsigned i = 0; i < H; i++) xn[i] = bf2f(resid[i]) * inv * bf2f(gam[i]);
        for (unsigned s = 0; s < TOPK; s++) {
            const bf16* w = gu[rid[s]];
            for (unsigned q = 0; q < I_MOE; q++) {
                const bf16* gr = w + (size_t)q * H;
                const bf16* ur = w + (size_t)(I_MOE + q) * H;
                float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
                for (unsigned i = 0; i < H; i++) {
                    const float xv = bf2f(xin[i]);
                    a = fmaf(xv, bf2f(gr[i]), a);
                    b = fmaf(xv, bf2f(ur[i]), b);
                    c = fmaf(xn[i], bf2f(gr[i]), c);
                    d = fmaf(xn[i], bf2f(ur[i]), d);
                }
                want[(size_t)s * I_MOE + q] = gelu_tanh(a) * b;
                want2[(size_t)s * I_MOE + q] = gelu_tanh(c) * d;
            }
        }
        check_bf16("op62 EXPERT_GLU (arm)", g_fu, want, FU_N, 2e-2);
        check_bf16("op71 EXPERT_GLU_NORM (arm)", g_fu2, want2, FU_N, 2e-2);
        free(want); free(want2); free(xn);
    }
    float* r_part = malloc(PART_N * sizeof(float));
    for (unsigned s = 0; s < TOPK; s++) {
        const bf16* w = dn[rid[s]];
        const bf16* fs = g_fu + (size_t)s * I_MOE;
        for (unsigned i = 0; i < H; i++) {
            float acc = 0.0f;
            const bf16* wr = w + (size_t)i * I_MOE;
            for (unsigned q = 0; q < I_MOE; q++) acc = fmaf(bf2f(fs[q]), bf2f(wr[q]), acc);
            r_part[(size_t)s * H + i] = rga[s] * acc;
        }
    }
    check_f32("op63 EXPERT_DOWN (arm)", g_part, r_part, PART_N, 5e-3);
    {
        float* sum = malloc(H * sizeof(float));
        for (unsigned i = 0; i < H; i++) {
            float acc = 0.0f;
            for (unsigned s = 0; s < TOPK; s++) acc += g_part[(size_t)s * H + i];
            sum[i] = acc;
        }
        check_bf16("op64 COMBINE (arm)", g_moe, sum, H, 1e-2);
        double ss = 0.0;
        for (unsigned i = 0; i < H; i++) ss += (double)sum[i] * sum[i];
        const float i1 = 1.0f / sqrtf((float)(ss / (double)H) + EPS);
        float* want = malloc(H * sizeof(float));
        for (unsigned i = 0; i < H; i++)
            want[i] = sum[i] * i1 * bf2f(gam[i]) + bf2f(resid[i]);
        check_bf16("op70 COMBINE_NORM (arm)", g_out, want, H, 1e-2);

        float* b = malloc(H * sizeof(float));
        double s2 = 0.0;
        for (unsigned i = 0; i < H; i++) {
            b[i] = bf2f(f2bf(sum[i] * i1 * bf2f(gam[i]) + bf2f(h1[i])));
            s2 += (double)b[i] * b[i];
        }
        const float i2 = 1.0f / sqrtf((float)(s2 / (double)H) + EPS);
        float* rr = malloc(H * sizeof(float));
        double s3 = 0.0;
        for (unsigned i = 0; i < H; i++) {
            rr[i] = bf2f(f2bf((bf2f(xres[i]) + b[i] * i2 * bf2f(gpo[i])) * LS));
            s3 += (double)rr[i] * rr[i];
        }
        const float i3 = 1.0f / sqrtf((float)(s3 / (double)H) + EPS);
        float* whn = malloc(H * sizeof(float));
        for (unsigned i = 0; i < H; i++) whn[i] = rr[i] * i3 * bf2f(gn[i]);
        check_bf16("op72 CRN x(resid) (arm)", g_x, rr, H, 1e-2);
        check_bf16("op72 CRN hn (arm)", g_hn, whn, H, 1e-2);
        free(sum); free(want); free(b); free(rr); free(whn);
    }

    printf("\n%d/%d checks passed — %s\n", checks - fails, checks,
           fails ? "FAILED" : "GEMMA-4 MoE DECODE RUNS THROUGH THE PERSISTENT INTERPRETER");
    plow_hsa_shutdown(h);
    return fails ? 1 : 0;
}
