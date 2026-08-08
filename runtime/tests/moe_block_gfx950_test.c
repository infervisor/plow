/* moe_block_gfx950_test.c — the GLM-5.2 MoE dispatch prototype on the real interpreter.
 *
 * Builds, by hand, a decode program of N stacked MoE FFN blocks — the data-dependent
 * counter-gate in full (the design notes §2-§3):
 *
 *     for each block b:
 *        MoeRouter(x[b], Wr[b]) -> routing_table            (writes {expert_id, gate}[K])
 *        for slot 0..K-1:  MoeExpertGlu -> fu[slot]         (weight base from the table;
 *                          MoeExpertDown -> part[slot]       sentinel slot skips, streams 0)
 *        MoeCombine(x[b], part[]) -> x[b+1]                 (residual + Σ part, fixed order)
 *
 * ONE persistent-interpreter launch runs the whole stack; the router's completion counter
 * unblocks the K expert slots ON DEVICE (no host round-trip), and each expert resolves its
 * weight base by the id the router chose. Synthetic seeded weights (fixed bits every run).
 * Truth is an independent fp32/bf16 CPU reference that mirrors the SAME sequential-dot op
 * bodies, so an exact match validates the dispatch, the routing, and the combine — not a
 * second copy of the arithmetic.
 *
 * Scope: the routed MoE FFN sublayer stacked N times (block 0 dense-equivalent numerically;
 * shared expert = NONE here, exercised bit-exact in the Rust core test crates/packet/
 * tests/moe_dispatch.rs). GLM-5.2 RouterCfg (sigmoid, norm_topk, route_scale=2.5),
 * cardinality-scaled (E=8, K=2)
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- scaled GLM-5.2 config ---- */
#define H       64      /* hidden (scaled small for a fast, exact CPU reference)      */
#define I_MOE   32      /* per-expert intermediate                                    */
#define N_EXP   8       /* routed experts (E)                                         */
#define K       2       /* top-k                                                      */
#define N_BLK   3       /* stacked MoE blocks                                         */
#define ACT     1u      /* SwiGLU (silu) — GLM                                        */
#define ROUTE_SCALE 2.5f
#define FLAGS   (1u | 2u) /* bit0 sigmoid, bit1 norm_topk                             */
#define N_SHARED 0

typedef uint16_t bf16;
static float b2f(bf16 v) { union { uint32_t u; float f; } c; c.u = (uint32_t)v << 16; return c.f; }
static bf16 f2b(float f) {
    union { float f; uint32_t u; } c; c.f = f;
    uint32_t r = c.u + 0x7fff + ((c.u >> 16) & 1);
    return (bf16)(r >> 16);
}

/* ---- deterministic seeded weights (identical bits every run) ---- */
static uint64_t rng_s;
static void seed(uint64_t s) { rng_s = s * 6364136223846793005ULL + 1442695040888963407ULL; }
static float frand(void) {
    rng_s = rng_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((int32_t)(rng_s >> 33) % 2001 - 1000) / 4000.0f; /* +-0.25 */
}

/* ---- the op bodies, mirrored sequentially (the CPU reference == the device kernels) ---- */
static float mdot(const bf16* x, const bf16* w, int Kk) {
    float acc = 0.0f;
    for (int i = 0; i < Kk; i++) acc += b2f(x[i]) * b2f(w[i]);
    return b2f(f2b(acc)); /* bf16-rounded like the op store */
}
static float act_silu(float x) { return x / (1.0f + expf(-x)); }

/* reference router: mirrors d_moe_router bit-for-bit (sigmoid, k-pass masked argmax with
 * lowest-id tie-break via the packed key, norm_topk, route_scale). */
static void ref_router(const bf16* x, const bf16* Wr, unsigned* out_id, float* out_gate) {
    float score[N_EXP];
    for (int e = 0; e < N_EXP; e++) {
        float logit = mdot(x, Wr + (size_t)e * H, H);
        score[e] = 1.0f / (1.0f + expf(-logit)); /* sigmoid */
    }
    float live[N_EXP];
    memcpy(live, score, sizeof(score));
    for (int j = 0; j < K; j++) {
        unsigned long long best = 0ull; int best_id = 0;
        for (int e = 0; e < N_EXP; e++) {
            unsigned sb; float sc = live[e]; memcpy(&sb, &sc, 4);
            sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
            unsigned long long key = ((unsigned long long)sb << 20) |
                                     (unsigned long long)((N_EXP - 1 - e) & 0xFFFFFu);
            if (key > best) { best = key; best_id = e; }
        }
        out_id[j] = (unsigned)best_id;
        out_gate[j] = live[best_id];
        live[best_id] = -1e30f;
    }
    float sum = 0.0f;
    for (int j = 0; j < K; j++) sum += out_gate[j];
    for (int j = 0; j < K; j++) {
        if (FLAGS & 2u) out_gate[j] /= sum; /* norm_topk */
        out_gate[j] *= ROUTE_SCALE;
    }
}

/* reference one MoE block: x_out = x + Σ_j gate_j · expert(x). */
static void ref_block(const bf16* x, const bf16* Wr, const bf16* const gate_w[N_EXP],
                      const bf16* const up_w[N_EXP], const bf16* const down_w[N_EXP],
                      bf16* x_out) {
    unsigned id[K]; float g[K];
    ref_router(x, Wr, id, g);
    float part[K][H];
    for (int slot = 0; slot < K; slot++) {
        unsigned e = id[slot];
        bf16 fu[I_MOE];
        for (int n = 0; n < I_MOE; n++) {
            float gg = mdot(x, gate_w[e] + (size_t)n * H, H);
            float uu = mdot(x, up_w[e] + (size_t)n * H, H);
            fu[n] = f2b(act_silu(gg) * uu);
        }
        for (int hh = 0; hh < H; hh++) {
            float y = mdot(fu, down_w[e] + (size_t)hh * I_MOE, I_MOE);
            part[slot][hh] = g[slot] * y;
        }
    }
    for (int hh = 0; hh < H; hh++) {
        float acc = b2f(x[hh]);
        for (int slot = 0; slot < K; slot++) acc += part[slot][hh];
        x_out[hh] = f2b(acc);
    }
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "interp_decode.elf";
    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, gfx, &cus, &lds);
    printf("dev0: %s  CUs=%u\n", gfx, cus);
    const unsigned NCU = cus;

    FILE* f = fopen(elf, "rb");
    if (!f) { printf("%s missing\n", elf); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc((size_t)co_n);
    if (fread(co, 1, (size_t)co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, (size_t)co_n)) { printf("load failed\n"); return 1; }
    plow_hsa_kernel kern;
    if (plow_hsa_get_kernel(h, 0, "plow_interp_dec_gfx950", &kern)) { printf("no kernel\n"); return 1; }

    /* ---- host weights (seeded), device allocations ---- */
    static bf16 h_Wr[N_BLK][N_EXP * H];
    static bf16 h_gate[N_BLK][N_EXP][I_MOE * H];
    static bf16 h_up[N_BLK][N_EXP][I_MOE * H];
    static bf16 h_down[N_BLK][N_EXP][H * I_MOE];
    static bf16 h_x0[H];
    for (int b = 0; b < N_BLK; b++) {
        seed(0xABCD ^ (uint64_t)b);
        for (int i = 0; i < N_EXP * H; i++) h_Wr[b][i] = f2b(frand());
        for (int e = 0; e < N_EXP; e++)
            for (int i = 0; i < I_MOE * H; i++) h_gate[b][e][i] = f2b(frand());
        for (int e = 0; e < N_EXP; e++)
            for (int i = 0; i < I_MOE * H; i++) h_up[b][e][i] = f2b(frand());
        for (int e = 0; e < N_EXP; e++)
            for (int i = 0; i < H * I_MOE; i++) h_down[b][e][i] = f2b(frand());
    }
    seed(777);
    for (int i = 0; i < H; i++) h_x0[i] = f2b(frand());

    void* d_Wr[N_BLK];
    void* d_gate[N_BLK][N_EXP];
    void* d_up[N_BLK][N_EXP];
    void* d_down[N_BLK][N_EXP];
    void* d_wtab[N_BLK];              /* [N_EXP][3] uint64 device pointers */
    void* d_x[N_BLK + 1];            /* residual stream x[0..N_BLK]        */
    void* d_table = plow_hsa_alloc(h, 0, K * 8);                 /* shared scratch */
    void* d_fu = plow_hsa_alloc(h, 0, (size_t)K * I_MOE * 2);
    void* d_part = plow_hsa_alloc(h, 0, (size_t)K * H * 4);      /* f32 partials */

    for (int b = 0; b < N_BLK; b++) {
        d_Wr[b] = plow_hsa_alloc(h, 0, N_EXP * H * 2);
        plow_hsa_upload(h, 0, d_Wr[b], h_Wr[b], N_EXP * H * 2);
        uint64_t wtab[N_EXP * 3];
        for (int e = 0; e < N_EXP; e++) {
            d_gate[b][e] = plow_hsa_alloc(h, 0, I_MOE * H * 2);
            d_up[b][e] = plow_hsa_alloc(h, 0, I_MOE * H * 2);
            d_down[b][e] = plow_hsa_alloc(h, 0, H * I_MOE * 2);
            plow_hsa_upload(h, 0, d_gate[b][e], h_gate[b][e], I_MOE * H * 2);
            plow_hsa_upload(h, 0, d_up[b][e], h_up[b][e], I_MOE * H * 2);
            plow_hsa_upload(h, 0, d_down[b][e], h_down[b][e], H * I_MOE * 2);
            wtab[e * 3 + 0] = (uint64_t)(uintptr_t)d_gate[b][e];
            wtab[e * 3 + 1] = (uint64_t)(uintptr_t)d_up[b][e];
            wtab[e * 3 + 2] = (uint64_t)(uintptr_t)d_down[b][e];
        }
        d_wtab[b] = plow_hsa_alloc(h, 0, sizeof(wtab));
        plow_hsa_upload(h, 0, d_wtab[b], wtab, sizeof(wtab));
    }
    for (int b = 0; b <= N_BLK; b++) d_x[b] = plow_hsa_alloc(h, 0, H * 2);
    plow_hsa_upload(h, 0, d_x[0], h_x0, H * 2);

    /* ---- tensor table: give every referenced buffer a handle ---- */
    /* handle layout: [x0..xN][table][fu][part][ per block: Wr, wtab ] */
    enum { T_X0 = 0, T_TABLE = N_BLK + 1, T_FU, T_PART, T_PERBLK };
    const int n_tensor = T_PERBLK + N_BLK * 2;
    void** h_tens = calloc(n_tensor, sizeof(void*));
    for (int b = 0; b <= N_BLK; b++) h_tens[b] = d_x[b];
    h_tens[T_TABLE] = d_table;
    h_tens[T_FU] = d_fu;
    h_tens[T_PART] = d_part;
    for (int b = 0; b < N_BLK; b++) {
        h_tens[T_PERBLK + b * 2 + 0] = d_Wr[b];
        h_tens[T_PERBLK + b * 2 + 1] = d_wtab[b];
    }
    void* d_tens = plow_hsa_alloc(h, 0, (size_t)n_tensor * sizeof(void*));
    plow_hsa_upload(h, 0, d_tens, h_tens, (size_t)n_tensor * sizeof(void*));
    const int TH_WR = T_PERBLK, TH_WTAB = T_PERBLK + 1; /* stride 2 per block */
#define HWR(b) (TH_WR + (b) * 2)
#define HWTAB(b) (TH_WTAB + (b) * 2)

    /* ---- build the program: 2K+2 ops per block ---- */
    const int OPB = 2 * K + 2;
    const int n_ops = N_BLK * OPB;
    PlowDevInst* insts = calloc(n_ops, sizeof(PlowDevInst));
    /* waits: worst-case one per op-dependency; count as we go */
    PlowWait* waits = calloc(n_ops * (K + 1), sizeof(PlowWait));
    uint32_t* succs = calloc(n_ops, sizeof(uint32_t));
    int nw = 0;
    char is_router[512] = {0};
    for (int op = 0; op < n_ops; op++) succs[op] = (uint32_t)op; /* op signals its own counter */

    typedef struct { uint32_t wait_ofs, succ_ofs; uint16_t wait_len, succ_len; } PlowGate;
    PlowGate* gates = calloc((size_t)n_ops, sizeof(PlowGate)); /* copied onto StreamEnts below */
    int prev_combine = -1; /* counter of the previous block's combine */
    for (int b = 0; b < N_BLK; b++) {
        const int base = b * OPB;
        const int i_router = base;
        const int i_glu = base + 1;          /* glu/down interleaved: glu = base+1+2*slot */
        const int i_combine = base + OPB - 1;

        /* router: 1 CU. reads x[b], Wr[b]; writes table. */
        is_router[i_router] = 1;
        PlowDevInst* R = &insts[i_router];
        R->op = PLOW_DOP_MOE_ROUTER; R->blocks = 1;
        R->t[0] = T_TABLE; R->t[1] = (uint32_t)b; R->t[2] = (uint32_t)HWR(b);
        R->i[0] = H; R->i[1] = N_EXP; R->i[2] = K; R->i[3] = FLAGS; R->fj[0].f = ROUTE_SCALE;
        gates[i_router].succ_ofs = (uint32_t)i_router; gates[i_router].succ_len = 1;
        if (prev_combine >= 0) {
            gates[i_router].wait_ofs = nw; waits[nw].id = prev_combine; waits[nw].threshold = NCU; nw++;
            gates[i_router].wait_len = 1;
        }

        for (int slot = 0; slot < K; slot++) {
            const int ig = i_glu + 2 * slot, id = ig + 1;
            PlowDevInst* G = &insts[ig];
            G->op = PLOW_DOP_MOE_EXPERT_GLU; G->blocks = NCU;
            G->t[0] = T_FU; G->t[1] = (uint32_t)b; G->t[2] = T_TABLE; G->t[3] = (uint32_t)HWTAB(b);
            G->i[0] = (uint32_t)slot; G->i[1] = I_MOE; G->i[2] = H; G->i[3] = N_EXP; G->i[5] = ACT;
            gates[ig].wait_ofs = nw; waits[nw].id = i_router; waits[nw].threshold = 1; nw++;
            gates[ig].wait_len = 1; gates[ig].succ_ofs = (uint32_t)ig; gates[ig].succ_len = 1;

            PlowDevInst* D = &insts[id];
            D->op = PLOW_DOP_MOE_EXPERT_DOWN; D->blocks = NCU;
            D->t[0] = T_PART; D->t[1] = T_FU; D->t[2] = T_TABLE; D->t[3] = (uint32_t)HWTAB(b);
            D->i[0] = (uint32_t)slot; D->i[1] = H; D->i[2] = I_MOE; D->i[3] = N_EXP;
            gates[id].wait_ofs = nw; waits[nw].id = ig; waits[nw].threshold = NCU; nw++;
            gates[id].wait_len = 1; gates[id].succ_ofs = (uint32_t)id; gates[id].succ_len = 1;
        }

        /* combine: waits all K down slots; writes x[b+1]. */
        PlowDevInst* C = &insts[i_combine];
        C->op = PLOW_DOP_MOE_COMBINE; C->blocks = NCU;
        C->t[0] = (uint32_t)(b + 1); C->t[1] = (uint32_t)b; C->t[2] = PLOW_TENSOR_NONE; C->t[3] = T_PART;
        C->i[0] = H; C->i[1] = K;
        gates[i_combine].wait_ofs = nw;
        for (int slot = 0; slot < K; slot++) {
            waits[nw].id = i_glu + 2 * slot + 1; waits[nw].threshold = NCU; nw++;
        }
        gates[i_combine].wait_len = K; gates[i_combine].succ_ofs = (uint32_t)i_combine; gates[i_combine].succ_len = 1;
        prev_combine = i_combine;
    }

    /* ---- per-CU streams (coarse; flags=0). Router ops only on CU 0. ---- */
    uint32_t* h_sofs = malloc(4u * NCU);
    uint32_t* h_slen = malloc(4u * NCU);
    /* count total entries */
    size_t total = 0;
    for (unsigned cu = 0; cu < NCU; cu++)
        for (int op = 0; op < n_ops; op++)
            if (!(is_router[op] && cu != 0)) total++;
    PlowStreamEnt* h_stream = calloc(total, sizeof(PlowStreamEnt));
    size_t si = 0;
    for (unsigned cu = 0; cu < NCU; cu++) {
        h_sofs[cu] = (uint32_t)si;
        for (int op = 0; op < n_ops; op++) {
            if (is_router[op] && cu != 0) continue;
            h_stream[si].inst = (uint32_t)op;
            h_stream[si].slice = is_router[op] ? 0u : cu; /* slice = local index in the op's CU set */
            h_stream[si].wait_ofs = gates[op].wait_ofs; h_stream[si].wait_len = gates[op].wait_len;
            h_stream[si].succ_ofs = gates[op].succ_ofs; h_stream[si].succ_len = gates[op].succ_len;
            si++;
        }
        h_slen[cu] = (uint32_t)si - h_sofs[cu];
    }

    /* ---- upload program ---- */
    void* d_inst = plow_hsa_alloc(h, 0, (size_t)n_ops * sizeof(PlowDevInst));
    void* d_stream = plow_hsa_alloc(h, 0, total * sizeof(PlowStreamEnt));
    void* d_sofs = plow_hsa_alloc(h, 0, 4u * NCU);
    void* d_slen = plow_hsa_alloc(h, 0, 4u * NCU);
    void* d_waits = plow_hsa_alloc(h, 0, (size_t)(nw ? nw : 1) * sizeof(PlowWait));
    void* d_succs = plow_hsa_alloc(h, 0, (size_t)n_ops * 4);
    void* d_ctr = plow_hsa_alloc(h, 0, (size_t)n_ops * PLOW_CTR_STRIDE * 4);
    plow_hsa_upload(h, 0, d_inst, insts, (size_t)n_ops * sizeof(PlowDevInst));
    plow_hsa_upload(h, 0, d_stream, h_stream, total * sizeof(PlowStreamEnt));
    plow_hsa_upload(h, 0, d_sofs, h_sofs, 4u * NCU);
    plow_hsa_upload(h, 0, d_slen, h_slen, 4u * NCU);
    if (nw) plow_hsa_upload(h, 0, d_waits, waits, (size_t)nw * sizeof(PlowWait));
    plow_hsa_upload(h, 0, d_succs, succs, (size_t)n_ops * 4);
    uint32_t* zc = calloc((size_t)n_ops * PLOW_CTR_STRIDE, 4);
    plow_hsa_upload(h, 0, d_ctr, zc, (size_t)n_ops * PLOW_CTR_STRIDE * 4);

    PlowProgram prog;
    memset(&prog, 0, sizeof(prog));
    prog.insts = d_inst; prog.stream = d_stream; prog.stream_ofs = d_sofs;
    prog.stream_len = d_slen; prog.waits = d_waits; prog.succs = d_succs;
    prog.counters = d_ctr; prog.tensors = (void* const*)d_tens;

    printf("program: %d ops (%d blocks x %d), %zu workgroup-packets, %d counters\n\n",
           n_ops, N_BLK, OPB, total, n_ops);

    if (plow_hsa_launch(h, 0, &kern, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &prog,
                        sizeof(prog))) {
        printf("LAUNCH FAILED\n"); return 1;
    }
    plow_hsa_wait(h, 0);

    /* ---- CPU reference over the N blocks ---- */
    bf16 ref[N_BLK + 1][H];
    memcpy(ref[0], h_x0, sizeof(h_x0));
    for (int b = 0; b < N_BLK; b++) {
        const bf16* gate_w[N_EXP]; const bf16* up_w[N_EXP]; const bf16* down_w[N_EXP];
        for (int e = 0; e < N_EXP; e++) { gate_w[e] = h_gate[b][e]; up_w[e] = h_up[b][e]; down_w[e] = h_down[b][e]; }
        ref_block(ref[b], h_Wr[b], gate_w, up_w, down_w, ref[b + 1]);
    }

    /* ---- compare EVERY block boundary bit-exactly + report routing ---- */
    int ok = 1;
    printf("  %-10s %10s %10s\n", "boundary", "bit-exact", "worst abs");
    for (int b = 1; b <= N_BLK; b++) {
        bf16 got[H];
        plow_hsa_download(h, 0, got, d_x[b], H * 2);
        int exact = 1; double worst = 0;
        for (int i = 0; i < H; i++) {
            if (got[i] != ref[b][i]) exact = 0;
            double d = fabs(b2f(got[i]) - b2f(ref[b][i]));
            if (d > worst) worst = d;
        }
        ok &= exact;
        printf("  x[%d]       %10s %10.6f\n", b, exact ? "YES" : "NO", worst);
    }

    /* routing decisions: recompute the reference route per block for the log */
    printf("\n  routing (reference top-%d, lowest-id tie-break):\n", K);
    for (int b = 0; b < N_BLK; b++) {
        unsigned id[K]; float g[K];
        ref_router(ref[b], h_Wr[b], id, g);
        printf("   block %d: ", b);
        for (int j = 0; j < K; j++) printf("e%u(g=%.4f) ", id[j], g[j]);
        printf("\n");
    }

    /* counters: every op fired (executed == total: skip or compute). */
    plow_hsa_download(h, 0, zc, d_ctr, (size_t)n_ops * PLOW_CTR_STRIDE * 4);
    int ctr_ok = 1;
    for (int op = 0; op < n_ops; op++) {
        uint32_t got = zc[(size_t)op * PLOW_CTR_STRIDE];
        uint32_t want = insts[op].blocks;
        if (got != want) { ctr_ok = 0; printf("   counter %d: %u != %u\n", op, got, want); }
    }
    printf("\n  executed==total (every packet fired, skip or compute): %s\n", ctr_ok ? "YES" : "NO");

    printf("\n%s\n", (ok && ctr_ok) ? "MoE DISPATCH BIT-EXACT vs CPU REFERENCE"
                                    : "*** MoE DISPATCH MISMATCH ***");
    plow_hsa_shutdown(h);
    return (ok && ctr_ok) ? 0 : 1;
}
