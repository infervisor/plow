/* kda_fuse_bench_gfx950.c — what the KDA decode chain's PACKETS cost.      [K3-KDA-FUSE]
 *
 * The GEMV tuning campaign measured every K3 decode GEMV at M=1 and found exactly one of them
 * (bf16 lm_head) bandwidth-bound; every other one sat at a ~0.032 ms floor using under 3% of
 * achievable bandwidth. At batch 1 a KDA layer is therefore launch/protocol bound, and the only
 * lever that moves is the PACKET COUNT. `GemvQkvg` already demonstrated it: -207 packets/token
 * measured -22..27%.
 *
 * This is the instrument for the K3-specific remainder — the four KDA opcodes, which the emitter
 * spends SIX packets on per layer:
 *
 *     KdaConv(q) KdaConv(k) KdaConv(v)  KdaGate  KdaStateStep  KdaGatedNorm
 *
 * and which fuse to THREE:
 *
 *     KdaConv3                          KdaStateStep(+gate)   KdaGatedNorm
 *
 * WHAT IT MEASURES AND WHAT IT DOES NOT. It builds `L` layers of that chain, CHAINED — layer l's
 * first packet waits on layer l-1's last — and times the whole dispatch. It does NOT include the
 * seven projection GEMVs: those are unchanged by this work, they dominate the byte count, and
 * including them would bury the packet-count delta under 700 MiB/layer of weight streaming. The
 * number this prints is the MARGINAL cost of the KDA chain's packets, which is the quantity the
 * fusion moves. Wall time for the whole layer is the real-weight gate's business, not this one's.
 *
 * TP MATTERS MORE THAN ANYTHING ELSE HERE, so it is the first argument. At TP1 (H=96) the state
 * step moves 12 MiB per layer and is genuinely doing work; at TP8 (H=12, BV=8 — the emitter drops
 * BV with the local head count) it moves 1.5 MiB and the chain is almost pure protocol. TP8 is
 * where K3 decode actually runs.
 *
 * Weights are random. This measures TIME; `runtime/tests/k3_block_gfx950_test.c` against
 * `k3_real_oracle.py`'s fixture measures CORRECTNESS, and neither substitutes for the other.
 *
 * MEASURED, gfx950 / MI355X, 256 CU, one uncontended GPU, 69 layers, T=1, medians of 60:
 *
 *      shape            packets/layer   chain wall            per-CU conv channels
 *      TP1 H=96 BV=16     6 -> 3        6.814 -> 4.128 ms     48 -> 144   (-39.4%)
 *      TP4 H=24 BV=8      6 -> 3        5.251 -> 3.225 ms      6 -> 18    (-38.6%)
 *      TP8 H=12 BV=8      6 -> 3        4.910 -> 3.042 ms      6 -> 18    (-38.1%)
 *
 * and `state-step blocks` is IDENTICAL either way at every shape — 256/256 at TP1 and TP4, 192/256
 * at TP8, which is `state_step_blocks` doing what it does and not something the fusion moved.
 *
 * THE COST IS LINEAR IN THE PACKET COUNT, which is the finding under the finding. At TP8 the
 * six-packet chain measures 0.096 ms at L=1, 0.413 at L=5, 1.283 at L=17 and 5.027 at L=69: a
 * slope of 12.08 us per packet and an intercept of 0.02 ms, against 108 MiB of state traffic that
 * is ~17 us at roofline for the whole 69 layers. Nothing here is bandwidth.
 *
 *   ./kda_fuse_bench <interp_decode.elf> [H] [BV] [LAYERS] [ITERS] [TRAP]
 *      H=96 BV=16  -> TP1        H=12 BV=8 -> TP8 (the shipping shape)
 *
 * Runs BOTH modes back to back in one process against the SAME device buffers, so the delta is not
 * confounded by allocation, clocks or which GPU the pinning landed on.
 *
 * TRAP = `conv3` or `stepg` skips the timing and dispatches ONE packet with a demoted operand set
 * to PLOW_TENSOR_NONE. It must DIE. This interpreter's dispatch `default:` is a silent NOP, so an
 * arm that does not refuse a malformed packet leaves an output finite, fluent and wrong — and
 * `KdaConv3` carries four demoted handles and `KdaStateStepG` two, every one of which is a chance
 * to convolve two streams of three or run the recurrence against an uninitialised gate. A trap
 * that is never fired is a trap nobody has checked, so it is fired here.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

/* The two fused opcodes. Defined here as a fallback ONLY so this file still compiles against a
 * dev_isa.h that predates them; if the header has them, the header wins and a drift in either
 * direction is a compile error rather than a silently mis-dispatched packet. */
#ifndef PLOW_DOP_KDA_CONV3
#define PLOW_DOP_KDA_CONV3 109
#endif
#ifndef PLOW_DOP_KDA_STATE_STEP_G
#define PLOW_DOP_KDA_STATE_STEP_G 110
#endif

#define MAXOPS 2048
#define MAXW 8192
#define MAXTEN 4096

static PlowDevInst g_inst[MAXOPS];
static PlowWait g_wait[MAXW];
static uint32_t g_succ[MAXOPS];
typedef struct { uint32_t wait_ofs, succ_ofs; uint16_t wait_len, succ_len; } PlowGate;
static PlowGate g_gate[MAXOPS];
static int g_nops = 0, g_nw = 0;
static void* g_tens[MAXTEN];
static int g_nt = 0;

static int reg(void* p) { g_tens[g_nt] = p; return g_nt++; }
static int emitop(uint16_t op, uint16_t blocks) {
    int i = g_nops++;
    if (i >= MAXOPS) { printf("MAXOPS\n"); exit(1); }
    memset(&g_inst[i], 0, sizeof(g_inst[i]));
    for (int k = 0; k < 8; k++) g_inst[i].t[k] = PLOW_TENSOR_NONE;
    g_inst[i].op = op; g_inst[i].blocks = blocks;
    g_gate[i].wait_ofs = 0; g_gate[i].wait_len = 0;
    g_gate[i].succ_ofs = i; g_gate[i].succ_len = 1; g_succ[i] = i;
    return i;
}
static void addwait(int op, int producer, uint32_t thr) {
    if (g_gate[op].wait_len == 0) g_gate[op].wait_ofs = g_nw;
    if (g_nw >= MAXW) { printf("MAXW\n"); exit(1); }
    g_wait[g_nw].id = producer; g_wait[g_nw].threshold = thr; g_nw++;
    g_gate[op].wait_len++;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

/* bf16 from a float, round-to-nearest-even is not needed for random input. */
static uint16_t f2b(float f) { union { float f; uint32_t u; } c; c.f = f; return (uint16_t)(c.u >> 16); }
static uint32_t rs = 0x1234567u;
static float rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return ((float)(rs & 0xFFFFFF) / 16777216.0f - 0.5f) * 0.5f;
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "interp_decode.elf";
    const int H = argc > 2 ? atoi(argv[2]) : 12;
    const int BV = argc > 3 ? atoi(argv[3]) : 8;
    const int L = argc > 4 ? atoi(argv[4]) : 69;
    const int ITERS = argc > 5 ? atoi(argv[5]) : 50;
    const int D = 128, W = 4, T = 1, GMODE = 1;
    const int P = H * D;
    const float EPS = 1e-5f, LB = -5.0f, SCALE = 1.0f / sqrtf((float)D);
    setbuf(stdout, NULL);

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, gfx, &cus, &lds);
    const unsigned NCU = cus;
    printf("dev0: %s CUs=%u\n", gfx, cus);
    printf("KDA chain: L=%d layers  T=%d H=%d D=%d W=%d BV=%d  (P=H*D=%d)\n", L, T, H, D, W, BV, P);

    FILE* f = fopen(elf, "rb");
    if (!f) { printf("%s missing\n", elf); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(co_n);
    if (fread(co, 1, co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, co_n)) { printf("load failed\n"); return 1; }
    plow_hsa_kernel kern;
    if (plow_hsa_get_kernel(h, 0, "plow_interp_dec_gfx950", &kern)) { printf("no kernel\n"); return 1; }

    /* ---- device buffers, one set per layer (the real thing has per-layer weights and a per-layer
     * state; sharing them would make this measure an L2 hit rate instead of a chain). ---- */
    const size_t bp = (size_t)P * 2, fp = (size_t)P * 4;
    const size_t cwb = (size_t)P * W * 4, stb = (size_t)H * D * D * 4;
    float* hbuf = malloc(stb > cwb ? stb : cwb);

#define DRND_F32(bytes) ({ size_t _n = (bytes) / 4; for (size_t _i = 0; _i < _n; _i++) ((float*)hbuf)[_i] = rnd(); \
                           void* _d = plow_hsa_alloc(h, 0, (bytes)); plow_hsa_upload(h, 0, _d, hbuf, (bytes)); reg(_d); })
#define DRND_BF(bytes)  ({ size_t _n = (bytes) / 2; for (size_t _i = 0; _i < _n; _i++) ((uint16_t*)hbuf)[_i] = f2b(rnd()); \
                           void* _d = plow_hsa_alloc(h, 0, (bytes)); plow_hsa_upload(h, 0, _d, hbuf, (bytes)); reg(_d); })
#define DNEW(bytes)     ({ void* _d = plow_hsa_alloc(h, 0, (bytes)); reg(_d); })

    typedef struct {
        int raw[3], mix[3], cw[3], cs[3];
        int graw, braw, alog, dtb, gate, beta, o, y, onorm, state;
    } Layer;
    Layer* ly = calloc(L, sizeof(Layer));
    for (int l = 0; l < L; l++) {
        for (int s = 0; s < 3; s++) {
            ly[l].raw[s] = DRND_BF(bp);
            ly[l].mix[s] = DNEW(bp);
            ly[l].cw[s] = DRND_F32(cwb);
            ly[l].cs[s] = DRND_F32(cwb);
        }
        ly[l].graw = DRND_BF(bp);
        ly[l].braw = DRND_BF((size_t)H * 2);
        ly[l].alog = DRND_F32((size_t)H * 4);
        ly[l].dtb = DRND_F32(fp);
        ly[l].gate = DNEW(fp);
        ly[l].beta = DNEW((size_t)H * 4);
        ly[l].o = DNEW(bp);
        ly[l].y = DNEW(bp);
        ly[l].onorm = DRND_F32((size_t)D * 4);
        ly[l].state = DRND_F32(stb);
    }
    printf("tensors: %d  (%.0f MiB of per-layer state)\n", g_nt, (double)stb * L / (1 << 20));

    void* d_tens = plow_hsa_alloc(h, 0, (size_t)g_nt * sizeof(void*));
    plow_hsa_upload(h, 0, d_tens, g_tens, (size_t)g_nt * sizeof(void*));

    const uint16_t ALL = (uint16_t)NCU;
    unsigned items = (unsigned)(P / BV);
    uint16_t sb = (uint16_t)(items < NCU ? (items ? items : 1) : NCU);
    const char* trap = argc > 6 ? argv[6] : NULL;

    /* ---- the refusal check. One packet, one demoted handle removed. ---- */
    if (trap) {
        Layer* z = &ly[0];
        int i;
        if (!strcmp(trap, "conv3")) {
            i = emitop(PLOW_DOP_KDA_CONV3, ALL);
            g_inst[i].t[0] = z->mix[0]; g_inst[i].t[1] = z->mix[1]; g_inst[i].t[2] = z->mix[2];
            g_inst[i].t[3] = z->raw[0]; g_inst[i].t[4] = z->raw[1]; g_inst[i].t[5] = z->raw[2];
            g_inst[i].t[6] = z->cw[0];  g_inst[i].t[7] = z->cw[1];
            g_inst[i].i[0] = T; g_inst[i].i[1] = P; g_inst[i].i[2] = W; g_inst[i].i[3] = 1;
            g_inst[i].i[4] = PLOW_TENSOR_NONE;      /* <- the v TAPS, absent */
            g_inst[i].i[5] = (uint32_t)z->cs[0];
            g_inst[i].i[6] = (uint32_t)z->cs[1];
            g_inst[i].i[7] = (uint32_t)z->cs[2];
            printf("TRAP CHECK: KdaConv3 with i4 (w_v) = PLOW_TENSOR_NONE\n");
        } else {
            i = emitop(PLOW_DOP_KDA_STATE_STEP_G, sb);
            g_inst[i].t[0] = z->o; g_inst[i].t[1] = z->mix[0];
            g_inst[i].t[2] = z->mix[1]; g_inst[i].t[3] = z->mix[2];
            g_inst[i].t[4] = z->graw; g_inst[i].t[5] = z->braw;
            g_inst[i].t[6] = z->state; g_inst[i].t[7] = z->alog;
            g_inst[i].i[0] = T; g_inst[i].i[1] = H; g_inst[i].i[2] = D;
            g_inst[i].i[3] = BV; g_inst[i].i[4] = 1;
            g_inst[i].i[5] = PLOW_TENSOR_NONE;      /* <- dt_bias, absent */
            g_inst[i].i[6] = GMODE;
            g_inst[i].fj[0].f = SCALE; g_inst[i].fj[1].f = LB;
            printf("TRAP CHECK: KdaStateStepG with i5 (dt_bias) = PLOW_TENSOR_NONE\n");
        }
        const int n_ops = g_nops;
        uint32_t *sofs = calloc(NCU, 4), *slen = calloc(NCU, 4);
        size_t total = 0;
        for (unsigned cu = 0; cu < NCU; cu++)
            if (cu < g_inst[0].blocks) total++;
        PlowStreamEnt* stream = calloc(total, sizeof(PlowStreamEnt));
        size_t si = 0;
        for (unsigned cu = 0; cu < NCU; cu++) {
            sofs[cu] = (uint32_t)si;
            if (cu < g_inst[0].blocks) {
                stream[si].inst = 0; stream[si].slice = cu;
                stream[si].succ_ofs = 0; stream[si].succ_len = 1;
                si++;
            }
            slen[cu] = (uint32_t)si - sofs[cu];
        }
        void* d_inst = plow_hsa_alloc(h, 0, (size_t)n_ops * sizeof(PlowDevInst));
        void* d_stream = plow_hsa_alloc(h, 0, total * sizeof(PlowStreamEnt));
        void* d_sofs = plow_hsa_alloc(h, 0, 4u * NCU);
        void* d_slen = plow_hsa_alloc(h, 0, 4u * NCU);
        void* d_waits = plow_hsa_alloc(h, 0, sizeof(PlowWait));
        void* d_succs = plow_hsa_alloc(h, 0, (size_t)n_ops * 4);
        void* d_ctr = plow_hsa_alloc(h, 0, (size_t)n_ops * PLOW_CTR_STRIDE * 4);
        plow_hsa_upload(h, 0, d_inst, g_inst, (size_t)n_ops * sizeof(PlowDevInst));
        plow_hsa_upload(h, 0, d_stream, stream, total * sizeof(PlowStreamEnt));
        plow_hsa_upload(h, 0, d_sofs, sofs, 4u * NCU);
        plow_hsa_upload(h, 0, d_slen, slen, 4u * NCU);
        plow_hsa_upload(h, 0, d_succs, g_succ, (size_t)n_ops * 4);
        uint32_t* zc = calloc((size_t)n_ops * PLOW_CTR_STRIDE, 4);
        plow_hsa_upload(h, 0, d_ctr, zc, (size_t)n_ops * PLOW_CTR_STRIDE * 4);
        PlowProgram prog; memset(&prog, 0, sizeof(prog));
        prog.insts = d_inst; prog.stream = d_stream; prog.stream_ofs = d_sofs;
        prog.stream_len = d_slen; prog.waits = d_waits; prog.succs = d_succs;
        prog.counters = d_ctr; prog.tensors = (void* const*)d_tens;
        printf("dispatching... a NOP here is the bug this check exists for\n");
        plow_hsa_launch(h, 0, &kern, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &prog,
                        sizeof(prog));
        plow_hsa_wait(h, 0);
        printf("*** SURVIVED — THE ARM DID NOT REFUSE THE PACKET ***\n");
        return 1;
    }

    double ms[2];
    long pk[2];
    for (int mode = 0; mode < 2; mode++) {
        g_nops = 0; g_nw = 0;
        int prev = -1;
        for (int l = 0; l < L; l++) {
            Layer* z = &ly[l];
            int i_step, i_norm;
            if (mode == 0) {
                /* ---- BASELINE: three convs, a gate, the step, the gated norm. ---- */
                int i_conv[3];
                for (int s = 0; s < 3; s++) {
                    int i = emitop(PLOW_DOP_KDA_CONV, ALL);
                    g_inst[i].t[0] = z->mix[s]; g_inst[i].t[1] = z->raw[s];
                    g_inst[i].t[2] = z->cw[s]; g_inst[i].t[3] = z->cs[s];
                    g_inst[i].i[0] = T; g_inst[i].i[1] = P; g_inst[i].i[2] = W; g_inst[i].i[3] = 1;
                    if (prev >= 0) addwait(i, prev, ALL);
                    i_conv[s] = i;
                }
                int i_gate = emitop(PLOW_DOP_KDA_GATE, ALL);
                g_inst[i_gate].t[0] = z->gate; g_inst[i_gate].t[1] = z->beta;
                g_inst[i_gate].t[2] = z->graw; g_inst[i_gate].t[3] = z->braw;
                g_inst[i_gate].t[4] = z->alog; g_inst[i_gate].t[5] = z->dtb;
                g_inst[i_gate].i[0] = T; g_inst[i_gate].i[1] = H; g_inst[i_gate].i[2] = D;
                g_inst[i_gate].i[3] = GMODE; g_inst[i_gate].fj[0].f = LB;
                if (prev >= 0) addwait(i_gate, prev, ALL);

                i_step = emitop(PLOW_DOP_KDA_STATE_STEP, sb);
                g_inst[i_step].t[0] = z->o; g_inst[i_step].t[1] = z->mix[0];
                g_inst[i_step].t[2] = z->mix[1]; g_inst[i_step].t[3] = z->mix[2];
                g_inst[i_step].t[4] = z->gate; g_inst[i_step].t[5] = z->beta;
                g_inst[i_step].t[6] = z->state;
                g_inst[i_step].i[0] = T; g_inst[i_step].i[1] = H; g_inst[i_step].i[2] = D;
                g_inst[i_step].i[3] = BV; g_inst[i_step].i[4] = 1;
                g_inst[i_step].fj[0].f = SCALE;
                for (int s = 0; s < 3; s++) addwait(i_step, i_conv[s], ALL);
                addwait(i_step, i_gate, ALL);
            } else {
                /* ---- FUSED: one conv over the 3*P concatenated channel axis, and the gate
                 * folded into the state step's LDS staging. ---- */
                int i_c3 = emitop(PLOW_DOP_KDA_CONV3, ALL);
                g_inst[i_c3].t[0] = z->mix[0]; g_inst[i_c3].t[1] = z->mix[1];
                g_inst[i_c3].t[2] = z->mix[2];
                g_inst[i_c3].t[3] = z->raw[0]; g_inst[i_c3].t[4] = z->raw[1];
                g_inst[i_c3].t[5] = z->raw[2];
                g_inst[i_c3].t[6] = z->cw[0]; g_inst[i_c3].t[7] = z->cw[1];
                g_inst[i_c3].i[0] = T; g_inst[i_c3].i[1] = P; g_inst[i_c3].i[2] = W;
                g_inst[i_c3].i[3] = 1;
                g_inst[i_c3].i[4] = (uint32_t)z->cw[2];
                g_inst[i_c3].i[5] = (uint32_t)z->cs[0];
                g_inst[i_c3].i[6] = (uint32_t)z->cs[1];
                g_inst[i_c3].i[7] = (uint32_t)z->cs[2];
                if (prev >= 0) addwait(i_c3, prev, ALL);

                i_step = emitop(PLOW_DOP_KDA_STATE_STEP_G, sb);
                g_inst[i_step].t[0] = z->o; g_inst[i_step].t[1] = z->mix[0];
                g_inst[i_step].t[2] = z->mix[1]; g_inst[i_step].t[3] = z->mix[2];
                g_inst[i_step].t[4] = z->graw; g_inst[i_step].t[5] = z->braw;
                g_inst[i_step].t[6] = z->state; g_inst[i_step].t[7] = z->alog;
                g_inst[i_step].i[0] = T; g_inst[i_step].i[1] = H; g_inst[i_step].i[2] = D;
                g_inst[i_step].i[3] = BV; g_inst[i_step].i[4] = 1;
                g_inst[i_step].i[5] = (uint32_t)z->dtb; g_inst[i_step].i[6] = GMODE;
                g_inst[i_step].fj[0].f = SCALE; g_inst[i_step].fj[1].f = LB;
                addwait(i_step, i_c3, ALL);
            }
            i_norm = emitop(PLOW_DOP_KDA_GATED_NORM, ALL);
            g_inst[i_norm].t[0] = z->y; g_inst[i_norm].t[1] = z->o;
            g_inst[i_norm].t[2] = z->onorm; g_inst[i_norm].t[3] = z->graw;
            g_inst[i_norm].i[0] = T; g_inst[i_norm].i[1] = H; g_inst[i_norm].i[2] = D;
            g_inst[i_norm].fj[0].f = EPS;
            addwait(i_norm, i_step, sb);
            prev = i_norm;
        }
        const int n_ops = g_nops;
        pk[mode] = n_ops;

        uint32_t *sofs = calloc(NCU, 4), *slen = calloc(NCU, 4);
        size_t total = 0;
        for (unsigned cu = 0; cu < NCU; cu++)
            for (int op = 0; op < n_ops; op++)
                if (cu < g_inst[op].blocks) total++;
        PlowStreamEnt* stream = calloc(total, sizeof(PlowStreamEnt));
        size_t si = 0;
        for (unsigned cu = 0; cu < NCU; cu++) {
            sofs[cu] = (uint32_t)si;
            for (int op = 0; op < n_ops; op++) {
                if (cu >= g_inst[op].blocks) continue;
                stream[si].inst = (uint32_t)op; stream[si].slice = cu;
                stream[si].wait_ofs = g_gate[op].wait_ofs; stream[si].wait_len = g_gate[op].wait_len;
                stream[si].succ_ofs = g_gate[op].succ_ofs; stream[si].succ_len = g_gate[op].succ_len;
                si++;
            }
            slen[cu] = (uint32_t)si - sofs[cu];
        }
        void* d_inst = plow_hsa_alloc(h, 0, (size_t)n_ops * sizeof(PlowDevInst));
        void* d_stream = plow_hsa_alloc(h, 0, total * sizeof(PlowStreamEnt));
        void* d_sofs = plow_hsa_alloc(h, 0, 4u * NCU);
        void* d_slen = plow_hsa_alloc(h, 0, 4u * NCU);
        void* d_waits = plow_hsa_alloc(h, 0, (size_t)(g_nw ? g_nw : 1) * sizeof(PlowWait));
        void* d_succs = plow_hsa_alloc(h, 0, (size_t)n_ops * 4);
        void* d_ctr = plow_hsa_alloc(h, 0, (size_t)n_ops * PLOW_CTR_STRIDE * 4);
        plow_hsa_upload(h, 0, d_inst, g_inst, (size_t)n_ops * sizeof(PlowDevInst));
        plow_hsa_upload(h, 0, d_stream, stream, total * sizeof(PlowStreamEnt));
        plow_hsa_upload(h, 0, d_sofs, sofs, 4u * NCU);
        plow_hsa_upload(h, 0, d_slen, slen, 4u * NCU);
        if (g_nw) plow_hsa_upload(h, 0, d_waits, g_wait, (size_t)g_nw * sizeof(PlowWait));
        plow_hsa_upload(h, 0, d_succs, g_succ, (size_t)n_ops * 4);
        uint32_t* zc = calloc((size_t)n_ops * PLOW_CTR_STRIDE, 4);

        PlowProgram prog; memset(&prog, 0, sizeof(prog));
        prog.insts = d_inst; prog.stream = d_stream; prog.stream_ofs = d_sofs;
        prog.stream_len = d_slen; prog.waits = d_waits; prog.succs = d_succs;
        prog.counters = d_ctr; prog.tensors = (void* const*)d_tens;

        /* The counters are CUMULATIVE and the thresholds absolute, so a second launch against a
         * dirty counter block passes every gate immediately and measures nothing. Re-zeroing is
         * outside the timed region. */
        double* s = malloc(sizeof(double) * ITERS);
        int nrun = 0;
        for (int it = 0; it < ITERS + 3; it++) {
            plow_hsa_upload(h, 0, d_ctr, zc, (size_t)n_ops * PLOW_CTR_STRIDE * 4);
            plow_hsa_wait(h, 0);
            double t0 = now_ms();
            if (plow_hsa_launch(h, 0, &kern, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0,
                                &prog, sizeof(prog))) { printf("LAUNCH FAILED\n"); return 1; }
            plow_hsa_wait(h, 0);
            double dt = now_ms() - t0;
            if (it >= 3) s[nrun++] = dt; /* 3 warmups: first touch of every page */
        }
        qsort(s, nrun, sizeof(double), cmpd);
        ms[mode] = s[nrun / 2];
        printf("\n%-6s  %4d packets (%.2f/layer)  median %.4f ms   min %.4f  p90 %.4f\n",
               mode ? "FUSED" : "BASE", n_ops, (double)n_ops / L, s[nrun / 2], s[0],
               s[(nrun * 9) / 10]);
        printf("        per-CU work: conv %s, state-step %u of %u CUs (%.1f%%), items=%u\n",
               mode ? "3*P/nblk channels" : "P/nblk channels", sb, NCU, 100.0 * sb / NCU, items);
        free(s); free(zc); free(stream); free(sofs); free(slen);
    }

    printf("\n==== %d layers, H=%d BV=%d ====\n", L, H, BV);
    printf("  packets/layer  %.0f -> %.0f\n", (double)pk[0] / L, (double)pk[1] / L);
    printf("  chain wall     %.4f ms -> %.4f ms   (%+.1f%%)\n", ms[0], ms[1],
           100.0 * (ms[1] - ms[0]) / ms[0]);
    printf("  per packet     %.1f us -> %.1f us\n", ms[0] * 1e3 / pk[0], ms[1] * 1e3 / pk[1]);
    return 0;
}
