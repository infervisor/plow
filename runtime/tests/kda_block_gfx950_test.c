/* kda_block_gfx950_test.c — REAL-WEIGHT single-layer numeric gate for Kimi-K3 KDA.  [K3-KDA-GATE]
 *
 * The analogue of glm52_real_block_gfx950_test.c, for the mixer in 69 of K3's 93 layers. It mmaps
 * the fixture written by runtime/tests/kda_real_oracle.py (real layer weights + a per-stage fp32
 * reference), builds the thirteen-packet KDA block as a DevInst program, dispatches it on the
 * persistent gfx950 interpreter, and prints a per-stage residual table.
 *
 *   ./kda_block_test [interp_decode.elf] [kda_fixture.bin]
 *
 * WHY A SINGLE LAYER. Section 5 of plans/knob-contract.md: single-block runs before full-network
 * launches, always. GLM's B4 gate de-risked one real-weight layer before anyone tried 78, and that
 * is why GLM's bugs were findable. A 93-layer K3 blob would fail here in a way nobody could
 * localise.
 *
 * WHAT IT IS ACTUALLY GATING, in order of how silently each fails:
 *
 *  1. THE V-FIRST STATE. K3 stores the recurrent state [h][v][k], not [h][k][v]. V == K == 128, so
 *     a transposed state has EXACTLY the right norm and every magnitude check passes. The fixture
 *     carries the reference state V-first and this test diffs it elementwise; that diff is the
 *     only thing that catches the transpose.
 *  2. THE FOUR NEW ARMS ACTUALLY RUNNING. AMD's dispatch `default:` is a silent NOP, not a trap —
 *     an opcode with no arm leaves the output buffer untouched. That is exactly how op 90
 *     (MAMBA2_SCAN) computes nothing on gfx950 today. The per-op counter check below fails loudly
 *     if any packet did not execute on every workgroup it was sliced across, and a stage whose arm
 *     is missing shows up as a residual of 1.0 rather than as silence.
 *  3. THE GATE BRANCH. K3 is the first checkpoint to ship `gate_lower_bound`, so the bounded
 *     branch `lb*sigmoid(exp(A_log)*(g+dt_bias))` has no released implementation to inherit from.
 *     `g` is diffed in f32, before it is exponentiated, because after `exp()` a wrong gate near
 *     `g = 0` is numerically invisible.
 *  4. THE A_log NARROW. The fixture ships A_log at [96]; the checkpoint has [128]. If the loader
 *     had not narrowed, the byte size would not match and the oracle would have refused.
 *
 * The state and conv-state are diffed as f32 with no bf16 round trip, for the same reason the GLM
 * gate diffs `expert_sum` in f32: a bf16 comparison hides cancellation.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

typedef uint16_t bf16;
static float b2f(bf16 b) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)b << 16;
    return c.f;
}

/* ---- program builder (verbatim from glm52_real_block_gfx950_test.c) ---- */
static PlowDevInst g_inst[512];
static PlowWait g_wait[2048];
static uint32_t g_succ[512];
typedef struct { uint32_t wait_ofs, succ_ofs; uint16_t wait_len, succ_len; } PlowGate;
static PlowGate g_gate[512];
static int g_nops = 0, g_nw = 0;
static void* g_tens[256];
static int g_nt = 0;
static int reg(void* p) { g_tens[g_nt] = p; return g_nt++; }
static int emitop(uint16_t op, uint16_t blocks) {
    int i = g_nops++;
    g_inst[i].op = op; g_inst[i].blocks = blocks;
    g_gate[i].succ_ofs = i; g_gate[i].succ_len = 1; g_succ[i] = i;
    return i;
}
static void addwait(int op, int producer, uint32_t thr) {
    if (g_gate[op].wait_len == 0) g_gate[op].wait_ofs = g_nw;
    g_wait[g_nw].id = producer; g_wait[g_nw].threshold = thr; g_nw++;
    g_gate[op].wait_len++;
}

static double relerr(const bf16* got, const bf16* want, size_t n, double* worst) {
    double mw = 0, md = 0, se = 0, sw = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(b2f(got[i]) - b2f(want[i])), w = fabs(b2f(want[i]));
        mw = fmax(mw, w); md = fmax(md, d); se += d * d; sw += (double)b2f(want[i]) * b2f(want[i]);
    }
    *worst = md / (mw + 1e-9);
    return sqrt(se / (double)n) / (sqrt(sw / (double)n) + 1e-9);
}
static double relerr_f32(const float* got, const float* want, size_t n, double* worst) {
    double mw = 0, md = 0, se = 0, sw = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - want[i]), w = fabs((double)want[i]);
        mw = fmax(mw, w); md = fmax(md, d); se += d * d; sw += (double)want[i] * want[i];
    }
    *worst = md / (mw + 1e-9);
    return sqrt(se / (double)n) / (sqrt(sw / (double)n) + 1e-9);
}

static int T, H, D, HID, W, GMODE, BV, P;
static float EPS, LB, SCALE;

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "interp_decode.elf";
    const char* fix = argc > 2 ? argv[2] : "kda_fixture.bin";
    setbuf(stdout, NULL);

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, gfx, &cus, &lds);
    const unsigned NCU = cus;
    printf("dev0: %s CUs=%u\n", gfx, cus);

    FILE* f = fopen(elf, "rb");
    if (!f) { printf("%s missing\n", elf); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(co_n);
    if (fread(co, 1, co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, co_n)) { printf("load failed\n"); return 1; }
    plow_hsa_kernel kern;
    if (plow_hsa_get_kernel(h, 0, "plow_interp_dec_gfx950", &kern)) { printf("no kernel\n"); return 1; }

    int fd = open(fix, O_RDONLY);
    if (fd < 0) { perror(fix); return 1; }
    struct stat st; fstat(fd, &st);
    char* base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    int32_t* hdr = (int32_t*)base;
    if (hdr[0] != 0x4B444131) { printf("bad magic %x (want KDA1 0x4B444131)\n", hdr[0]); return 1; }
    T = hdr[1]; H = hdr[2]; D = hdr[3]; HID = hdr[4]; W = hdr[5]; GMODE = hdr[6]; BV = hdr[7];
    float* fh = (float*)(base + 8 * 4);
    EPS = fh[0]; LB = fh[1]; SCALE = fh[2];
    P = H * D;
    printf("KDA layer: T=%d H=%d D=%d hidden=%d W=%d gate_mode=%d BV=%d lb=%.1f scale=%.6f eps=%.1e\n",
           T, H, D, HID, W, GMODE, BV, LB, SCALE, EPS);

    size_t off = 8 * 4 + 3 * 4;
#define NEXT(cnt, elt) ({ void* _p = base + off; off += (size_t)(cnt) * (elt); _p; })
    /* Order MUST mirror kda_real_oracle.py's write order exactly; the size assert below is the
     * only thing that catches a layout drift. */
    bf16* P_hidden = NEXT((size_t)T * HID, 2);
    bf16* P_lnw = NEXT((size_t)HID, 2);
    bf16* P_wq = NEXT((size_t)P * HID, 2);
    bf16* P_wk = NEXT((size_t)P * HID, 2);
    bf16* P_wv = NEXT((size_t)P * HID, 2);
    bf16* P_wg = NEXT((size_t)P * HID, 2);
    bf16* P_wo = NEXT((size_t)HID * P, 2);
    bf16* P_wfa = NEXT((size_t)D * HID, 2);
    bf16* P_wfb = NEXT((size_t)P * D, 2);
    bf16* P_wb = NEXT((size_t)H * HID, 2);
    float* P_cw[3];
    for (int s = 0; s < 3; s++) P_cw[s] = NEXT((size_t)P * W, 4);
    float* P_alog = NEXT((size_t)H, 4);          /* [96] — ALREADY narrowed from the ckpt's [128] */
    float* P_dtb = NEXT((size_t)P, 4);
    float* P_onorm = NEXT((size_t)D, 4);
    float* P_cs_in[3];
    for (int s = 0; s < 3; s++) P_cs_in[s] = NEXT((size_t)P * W, 4);
    float* P_state_in = NEXT((size_t)H * D * D, 4); /* V-FIRST [h][v][k] */
    /* references */
    bf16* R_x = NEXT((size_t)T * HID, 2);
    bf16* R_qc = NEXT((size_t)T * P, 2);
    bf16* R_kc = NEXT((size_t)T * P, 2);
    bf16* R_vc = NEXT((size_t)T * P, 2);
    float* R_gate = NEXT((size_t)T * P, 4);
    float* R_beta = NEXT((size_t)T * H, 4);
    bf16* R_o = NEXT((size_t)T * P, 2);
    float* R_state = NEXT((size_t)H * D * D, 4);
    bf16* R_y = NEXT((size_t)T * P, 2);
    bf16* R_out = NEXT((size_t)T * HID, 2);
    float* R_cs[3];
    for (int s = 0; s < 3; s++) R_cs[s] = NEXT((size_t)P * W, 4);
    if (off != (size_t)st.st_size) {
        printf("FIXTURE SIZE MISMATCH: consumed %zu, file %zu\n", off, (size_t)st.st_size);
        return 1;
    }

    /* ---- device buffers ---- */
#define DUP(ptr, bytes) ({ void* _d = plow_hsa_alloc(h, 0, (bytes)); \
                           plow_hsa_upload(h, 0, _d, (ptr), (bytes)); reg(_d); })
#define DNEW(bytes) ({ void* _d = plow_hsa_alloc(h, 0, (bytes)); reg(_d); })
    const size_t tp = (size_t)T * P, thid = (size_t)T * HID;
    int t_hidden = DUP(P_hidden, thid * 2), t_lnw = DUP(P_lnw, HID * 2);
    int t_wq = DUP(P_wq, (size_t)P * HID * 2), t_wk = DUP(P_wk, (size_t)P * HID * 2);
    int t_wv = DUP(P_wv, (size_t)P * HID * 2), t_wg = DUP(P_wg, (size_t)P * HID * 2);
    int t_wo = DUP(P_wo, (size_t)HID * P * 2), t_wfa = DUP(P_wfa, (size_t)D * HID * 2);
    int t_wfb = DUP(P_wfb, (size_t)P * D * 2), t_wb = DUP(P_wb, (size_t)H * HID * 2);
    int t_cw[3], t_cs[3];
    for (int s = 0; s < 3; s++) {
        t_cw[s] = DUP(P_cw[s], (size_t)P * W * 4);
        t_cs[s] = DUP(P_cs_in[s], (size_t)P * W * 4); /* IN/OUT — uploaded, read back below */
    }
    int t_alog = DUP(P_alog, (size_t)H * 4), t_dtb = DUP(P_dtb, (size_t)P * 4);
    int t_onorm = DUP(P_onorm, (size_t)D * 4);
    int t_state = DUP(P_state_in, (size_t)H * D * D * 4); /* IN/OUT, V-FIRST */
    int t_x = DNEW(thid * 2);
    int t_raw[3], t_mix[3];
    for (int s = 0; s < 3; s++) { t_raw[s] = DNEW(tp * 2); t_mix[s] = DNEW(tp * 2); }
    int t_graw = DNEW(tp * 2), t_fa = DNEW((size_t)T * D * 2), t_fraw = DNEW(tp * 2);
    int t_braw = DNEW((size_t)T * H * 2);
    int t_gate = DNEW(tp * 4), t_beta = DNEW((size_t)T * H * 4);
    int t_o = DNEW(tp * 2), t_y = DNEW(tp * 2), t_attn = DNEW(thid * 2), t_out = DNEW(thid * 2);

    /* ---- the thirteen packets. Shape and gating mirror crates/devgen/src/kda.rs exactly. ---- */
    const uint16_t ALL = (uint16_t)NCU;
#define GEMV(o_, x_, w_, N_, K_, dep_) ({                                                        \
        int _i = emitop(PLOW_DOP_GEMV, ALL);                                                     \
        g_inst[_i].t[0] = o_; g_inst[_i].t[1] = x_; g_inst[_i].t[2] = w_;                         \
        g_inst[_i].t[3] = PLOW_TENSOR_NONE; g_inst[_i].t[4] = PLOW_TENSOR_NONE;                   \
        g_inst[_i].i[0] = T; g_inst[_i].i[1] = (N_); g_inst[_i].i[2] = (K_);                      \
        addwait(_i, dep_, ALL); _i; })

    int i_ln = emitop(PLOW_DOP_RMSNORM, ALL);
    g_inst[i_ln].t[0] = t_x; g_inst[i_ln].t[1] = t_hidden; g_inst[i_ln].t[2] = t_lnw;
    g_inst[i_ln].i[0] = T; g_inst[i_ln].i[1] = HID; g_inst[i_ln].fj[0].f = EPS;

    /* P1-P6: six independent GEMVs, all gated only on the pre-norm. */
    int i_q = GEMV(t_raw[0], t_x, t_wq, P, HID, i_ln);
    int i_k = GEMV(t_raw[1], t_x, t_wk, P, HID, i_ln);
    int i_v = GEMV(t_raw[2], t_x, t_wv, P, HID, i_ln);
    int i_g = GEMV(t_graw, t_x, t_wg, P, HID, i_ln);
    int i_fa = GEMV(t_fa, t_x, t_wfa, D, HID, i_ln);
    int i_bb = GEMV(t_braw, t_x, t_wb, H, HID, i_ln);
    int i_fb = GEMV(t_fraw, t_fa, t_wfb, P, D, i_fa);

    /* P8a-c: three short convs. */
    int i_conv[3], src[3] = { i_q, i_k, i_v };
    for (int s = 0; s < 3; s++) {
        i_conv[s] = emitop(PLOW_DOP_KDA_CONV, ALL);
        g_inst[i_conv[s]].t[0] = t_mix[s]; g_inst[i_conv[s]].t[1] = t_raw[s];
        g_inst[i_conv[s]].t[2] = t_cw[s]; g_inst[i_conv[s]].t[3] = t_cs[s];
        g_inst[i_conv[s]].i[0] = T; g_inst[i_conv[s]].i[1] = P;
        g_inst[i_conv[s]].i[2] = W; g_inst[i_conv[s]].i[3] = 1; /* silu */
        addwait(i_conv[s], src[s], ALL);
    }

    /* P9: gate + beta. */
    int i_gate = emitop(PLOW_DOP_KDA_GATE, ALL);
    g_inst[i_gate].t[0] = t_gate; g_inst[i_gate].t[1] = t_beta;
    g_inst[i_gate].t[2] = t_fraw; g_inst[i_gate].t[3] = t_braw;
    g_inst[i_gate].t[4] = t_alog; g_inst[i_gate].t[5] = t_dtb;
    g_inst[i_gate].i[0] = T; g_inst[i_gate].i[1] = H; g_inst[i_gate].i[2] = D;
    g_inst[i_gate].i[3] = GMODE; g_inst[i_gate].fj[0].f = LB;
    addwait(i_gate, i_fb, ALL);
    addwait(i_gate, i_bb, ALL);

    /* P10: the recurrence. blocks = min(H*D/BV, NCU) — the check against 256 that section 7.3
     * requires of every proposal. Head-parallelism alone would put this at 96/256 = 37.5%. */
    unsigned items = (unsigned)(P / BV);
    uint16_t step_blocks = (uint16_t)(items < NCU ? items : NCU);
    int i_step = emitop(PLOW_DOP_KDA_STATE_STEP, step_blocks);
    g_inst[i_step].t[0] = t_o; g_inst[i_step].t[1] = t_mix[0];
    g_inst[i_step].t[2] = t_mix[1]; g_inst[i_step].t[3] = t_mix[2];
    g_inst[i_step].t[4] = t_gate; g_inst[i_step].t[5] = t_beta; g_inst[i_step].t[6] = t_state;
    g_inst[i_step].i[0] = T; g_inst[i_step].i[1] = H; g_inst[i_step].i[2] = D;
    g_inst[i_step].i[3] = BV; g_inst[i_step].i[4] = 1; /* flags bit0 = L2 norm q/k in kernel */
    g_inst[i_step].fj[0].f = SCALE;
    for (int s = 0; s < 3; s++) addwait(i_step, i_conv[s], ALL);
    addwait(i_step, i_gate, ALL);

    /* P11: gated output norm. */
    int i_norm = emitop(PLOW_DOP_KDA_GATED_NORM, ALL);
    g_inst[i_norm].t[0] = t_y; g_inst[i_norm].t[1] = t_o;
    g_inst[i_norm].t[2] = t_onorm; g_inst[i_norm].t[3] = t_graw;
    g_inst[i_norm].i[0] = T; g_inst[i_norm].i[1] = H; g_inst[i_norm].i[2] = D;
    g_inst[i_norm].fj[0].f = EPS;
    addwait(i_norm, i_step, step_blocks);
    addwait(i_norm, i_g, ALL);

    /* P12/P13. */
    int i_o = GEMV(t_attn, t_y, t_wo, HID, P, i_norm);
    int i_res = emitop(PLOW_DOP_RESIDUAL, ALL);
    g_inst[i_res].t[0] = t_out; g_inst[i_res].t[1] = t_hidden; g_inst[i_res].t[2] = t_attn;
    g_inst[i_res].i[0] = (uint32_t)thid; g_inst[i_res].fj[0].f = 1.0f;
    addwait(i_res, i_o, ALL);

    const int n_ops = g_nops;
    printf("program: %d packets, %d tensors; state_step blocks=%u of %u CUs (%.1f%%), items=%u\n",
           n_ops, g_nt, step_blocks, NCU, 100.0 * step_blocks / NCU, items);

    /* ---- per-CU streams, upload, launch ---- */
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
    void* d_tens = plow_hsa_alloc(h, 0, (size_t)g_nt * sizeof(void*));
    plow_hsa_upload(h, 0, d_tens, g_tens, (size_t)g_nt * sizeof(void*));
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
    plow_hsa_upload(h, 0, d_ctr, zc, (size_t)n_ops * PLOW_CTR_STRIDE * 4);
    PlowProgram prog; memset(&prog, 0, sizeof(prog));
    prog.insts = d_inst; prog.stream = d_stream; prog.stream_ofs = d_sofs; prog.stream_len = d_slen;
    prog.waits = d_waits; prog.succs = d_succs; prog.counters = d_ctr;
    prog.tensors = (void* const*)d_tens;
    if (plow_hsa_launch(h, 0, &kern, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &prog,
                        sizeof(prog))) {
        printf("LAUNCH FAILED\n");
        return 1;
    }

    /* ---- read back ---- */
    bf16* o_qc = malloc(tp * 2); bf16* o_kc = malloc(tp * 2); bf16* o_vc = malloc(tp * 2);
    bf16* o_x = malloc(thid * 2); bf16* o_o = malloc(tp * 2); bf16* o_y = malloc(tp * 2);
    bf16* o_out = malloc(thid * 2);
    float* o_gate = malloc(tp * 4); float* o_beta = malloc((size_t)T * H * 4);
    float* o_state = malloc((size_t)H * D * D * 4);
    float* o_cs[3];
    plow_hsa_download(h, 0, o_x, g_tens[t_x], thid * 2);
    plow_hsa_download(h, 0, o_qc, g_tens[t_mix[0]], tp * 2);
    plow_hsa_download(h, 0, o_kc, g_tens[t_mix[1]], tp * 2);
    plow_hsa_download(h, 0, o_vc, g_tens[t_mix[2]], tp * 2);
    plow_hsa_download(h, 0, o_gate, g_tens[t_gate], tp * 4);
    plow_hsa_download(h, 0, o_beta, g_tens[t_beta], (size_t)T * H * 4);
    plow_hsa_download(h, 0, o_o, g_tens[t_o], tp * 2);
    plow_hsa_download(h, 0, o_state, g_tens[t_state], (size_t)H * D * D * 4);
    plow_hsa_download(h, 0, o_y, g_tens[t_y], tp * 2);
    plow_hsa_download(h, 0, o_out, g_tens[t_out], thid * 2);
    for (int s = 0; s < 3; s++) {
        o_cs[s] = malloc((size_t)P * W * 4);
        plow_hsa_download(h, 0, o_cs[s], g_tens[t_cs[s]], (size_t)P * W * 4);
    }
    plow_hsa_download(h, 0, zc, d_ctr, (size_t)n_ops * PLOW_CTR_STRIDE * 4);

    /* EVERY packet must have executed on every workgroup it was sliced across. An arm that does
     * not exist still SIGNALS its counter — the interpreter bumps successors after `plow_exec`
     * regardless — so this catches a mis-sliced stream, not a missing arm. A missing arm shows up
     * as a residual of 1.0 in the table below, because the output buffer is never written. */
    int exec_ok = 1;
    for (int op = 0; op < n_ops; op++)
        if (zc[op * PLOW_CTR_STRIDE] != g_inst[op].blocks) {
            printf("  op %d (dop %u): executed %u of %u\n", op, g_inst[op].op,
                   zc[op * PLOW_CTR_STRIDE], g_inst[op].blocks);
            exec_ok = 0;
        }
    printf("all packets executed on every slice: %s\n", exec_ok ? "YES" : "NO");

    /* ---- residual table ---- */
    double w_[16], r_x, r_q, r_k, r_v, r_g, r_b, r_o, r_s, r_y, r_out, r_cs_[3];
    r_x = relerr(o_x, R_x, thid, &w_[0]);
    r_q = relerr(o_qc, R_qc, tp, &w_[1]);
    r_k = relerr(o_kc, R_kc, tp, &w_[2]);
    r_v = relerr(o_vc, R_vc, tp, &w_[3]);
    r_g = relerr_f32(o_gate, R_gate, tp, &w_[4]);
    r_b = relerr_f32(o_beta, R_beta, (size_t)T * H, &w_[5]);
    r_o = relerr(o_o, R_o, tp, &w_[6]);
    r_s = relerr_f32(o_state, R_state, (size_t)H * D * D, &w_[7]);
    r_y = relerr(o_y, R_y, tp, &w_[8]);
    r_out = relerr(o_out, R_out, thid, &w_[9]);
    for (int s = 0; s < 3; s++) r_cs_[s] = relerr_f32(o_cs[s], R_cs[s], (size_t)P * W, &w_[10 + s]);

    printf("\n  stage                       rms rel     worst rel\n");
    printf("  P0  pre-norm  x            %10.3e  %10.3e\n", r_x, w_[0]);
    printf("  P8  conv+silu q            %10.3e  %10.3e\n", r_q, w_[1]);
    printf("  P8  conv+silu k            %10.3e  %10.3e\n", r_k, w_[2]);
    printf("  P8  conv+silu v            %10.3e  %10.3e\n", r_v, w_[3]);
    printf("  P9  gate g       (f32)     %10.3e  %10.3e\n", r_g, w_[4]);
    printf("  P9  beta         (f32)     %10.3e  %10.3e\n", r_b, w_[5]);
    printf("  P10 state read  o          %10.3e  %10.3e\n", r_o, w_[6]);
    printf("  P10 STATE V-FIRST(f32)     %10.3e  %10.3e\n", r_s, w_[7]);
    printf("  P8  conv_state q (f32)     %10.3e  %10.3e\n", r_cs_[0], w_[10]);
    printf("  P8  conv_state k (f32)     %10.3e  %10.3e\n", r_cs_[1], w_[11]);
    printf("  P8  conv_state v (f32)     %10.3e  %10.3e\n", r_cs_[2], w_[12]);
    printf("  P11 gated norm  y          %10.3e  %10.3e\n", r_y, w_[8]);
    printf("  P13 BLOCK out              %10.3e  %10.3e\n", r_out, w_[9]);

    /* The transpose check, stated as a number rather than as a hope: diff the device state against
     * the K-FIRST reading of the same reference. If the kernel were storing [h][k][v] this would
     * be the SMALL one, and `r_s` would be large — and both states have the same norm, so nothing
     * else in this table would notice. */
    double wt;
    float* R_state_T = malloc((size_t)H * D * D * 4);
    for (int hh = 0; hh < H; hh++)
        for (int a = 0; a < D; a++)
            for (int bq = 0; bq < D; bq++)
                R_state_T[((size_t)hh * D + a) * D + bq] = R_state[((size_t)hh * D + bq) * D + a];
    double r_sT = relerr_f32(o_state, R_state_T, (size_t)H * D * D, &wt);
    printf("\n  state vs the TRANSPOSED reference: %10.3e  (must be LARGE — if it is the small "
           "one,\n  the kernel is storing [h][k][v] and every other row above still passes)\n", r_sT);

    /* Tolerances: the GLM B4 gate's, glm52_real_block_gfx950_test.c:363-364. bf16 stages 1.5e-2,
     * f32 accumulators 2e-2. Nothing here is fp8, so the loose 4e-2/6e-2 arm does not apply. */
    const double TOL = 1.5e-2, TOL_F32 = 2e-2;
    int ok = exec_ok && r_x < TOL && r_q < TOL && r_k < TOL && r_v < TOL && r_g < TOL_F32 &&
             r_b < TOL_F32 && r_o < TOL && r_s < TOL_F32 && r_y < TOL && r_out < TOL &&
             r_cs_[0] < TOL_F32 && r_cs_[1] < TOL_F32 && r_cs_[2] < TOL_F32 && r_sT > 10 * r_s;

    printf("\n=> %s\n", ok ? "KDA REAL-WEIGHT SINGLE-LAYER OK — plow matches the fla reference on "
                             "real Kimi-K3 layer weights"
                           : "*** KDA REAL-WEIGHT MISMATCH ***");
    return ok ? 0 : 1;
}
