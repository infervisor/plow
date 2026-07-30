/* k3_block_gfx950_test.c — REAL-WEIGHT gate for ONE COMPLETE Kimi-K3 BLOCK.       [K3-BLOCK-GATE]
 *
 * The KDA gate (kda_block_gfx950_test.c) proved the MIXER on real layer-0 weights. This proves the
 * BLOCK AROUND IT, which is where K3 stops looking like every other model in this tree:
 *
 *     AttnRes -> attention -> AttnRes -> FFN,   NOT   residual + attn ; residual + mlp.
 *
 * It mmaps the fixture written by runtime/tests/k3_real_oracle.py (real layer-0 weights + a
 * per-stage fp32 reference), builds the block as a DevInst program, dispatches it on the persistent
 * gfx950 interpreter, and prints a per-stage residual table.
 *
 *   ./k3_block_test [interp_decode.elf] [k3_fixture.bin]
 *   PLOW_KDA_FUSE=0 ./k3_block_test ...      # the six-packet KDA chain instead of the three
 *
 * TWENTY packets, or twenty-two with `PLOW_KDA_FUSE=0`. The KDA chain has two spellings — see the
 * P8/P9/P10 note below — and this gate scores BOTH against the same fixture, because "the fusion
 * did not move the numbers" is a claim that only means something if the unfused numbers came out
 * of the same binary on the same run.
 *
 * WHAT IT GATES THAT THE KDA GATE DID NOT, in order of how silently each fails:
 *
 *  1. THE BLOCK WIRING. K3 layer 0's output is `KDA_out + MLP_out` — the embedding hidden state
 *     reaches it only THROUGH the AttnRes mix. A `hidden + attn + mlp` wiring has the same shapes,
 *     the same dtypes and a similar magnitude. The oracle prints the distance between the two and
 *     ASSERTS it is large, so this gate is falsifiable rather than decorative.
 *  2. ATTNRES ITSELF. RMS-normalize each of nb+1 rows, score each against ONE folded [H] vector,
 *     softmax over the rows, then mix the RAW rows. Mixing the NORMALIZED rows instead is the
 *     natural misreading and gives the right shape with the wrong per-row magnitude.
 *  3. `situ`. It transforms the UP branch as well as the gate. Applying only the gate transform —
 *     which is what a third `act` code in the existing `act(g)*u` epilogue would do — is a small
 *     error at |u| < 25 that grows with the tail. `ff_gate` and `ff_act` are diffed separately so
 *     a GEMV bug and an activation bug cannot be confused.
 *  4. THE TWO NEW ARMS ACTUALLY RUNNING. AMD's dispatch `default:` is a silent NOP. A stage whose
 *     arm is missing shows up as a residual of 1.0, not as silence.
 *
 * `block_residual` is bound to the SAME device buffer as `hidden`. That is not a shortcut: at
 * layer 0 the snapshot IS the layer input (`block_residual = cat([], prefix_sum)` with
 * prefix_sum = hidden), and [T,1,HID] and [T,HID] are the identical byte layout.
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

/* ---- program builder (verbatim from kda_block_gfx950_test.c) ---- */
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
    /* SLOT 0 IS A LEGAL TENSOR HANDLE, so an unset `t[k]` in a static (zeroed) g_inst names
     * tensor 0 rather than "absent". The moe and mla gates have always done this; this one did
     * not, and it stayed harmless only for as long as no op read a slot this file leaves unset.
     *
     * ATTN_RES then grew an OPTIONAL t4 (`push_src`, the snapshot the ring is seeded from), and
     * this loop's absence turned it live: t4 read as tensor 0, so every AttnRes here requested a
     * push. At T=1 that pushed the layer input onto ring row 0 — which this fixture aliases to
     * `hidden` anyway, so it was a self-copy and the 19 rows still passed. At T>1 the arm poisons
     * (a multi-workgroup push has no barrier), so S6-S10 came back NaN and read as a regression
     * in op_k3.h. It was this line. */
    for (int k = 0; k < 8; k++) g_inst[i].t[k] = PLOW_TENSOR_NONE;
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

static int T, H, D, HID, W, GMODE, BV, INTER, NB, P;
static float EPS, LB, SCALE, BETA, LBETA;

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "interp_decode.elf";
    const char* fix = argc > 2 ? argv[2] : "k3_fixture.bin";
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
    if (hdr[0] != 0x4B334231) { printf("bad magic %x (want K3B1 0x4B334231)\n", hdr[0]); return 1; }
    T = hdr[1]; H = hdr[2]; D = hdr[3]; HID = hdr[4]; W = hdr[5]; GMODE = hdr[6]; BV = hdr[7];
    INTER = hdr[8]; NB = hdr[9];
    float* fh = (float*)(base + 10 * 4);
    EPS = fh[0]; LB = fh[1]; SCALE = fh[2]; BETA = fh[3]; LBETA = fh[4];
    P = H * D;
    printf("K3 block: T=%d H=%d D=%d hidden=%d W=%d BV=%d inter=%d nb=%d\n", T, H, D, HID, W, BV,
           INTER, NB);
    printf("          lb=%.1f scale=%.6f eps=%.1e situ beta=%.1f linear_beta=%.1f\n", LB, SCALE,
           EPS, BETA, LBETA);

    size_t off = 10 * 4 + 5 * 4;
#define NEXT(cnt, elt) ({ void* _p = base + off; off += (size_t)(cnt) * (elt); _p; })
    /* Order MUST mirror k3_real_oracle.py's write order exactly; the size assert below is the
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
    float* P_alog = NEXT((size_t)H, 4);
    float* P_dtb = NEXT((size_t)P, 4);
    float* P_onorm = NEXT((size_t)D, 4);
    float* P_cs_in[3];
    for (int s = 0; s < 3; s++) P_cs_in[s] = NEXT((size_t)P * W, 4);
    float* P_state_in = NEXT((size_t)H * D * D, 4);
    float* P_score_w = NEXT((size_t)HID, 4);       /* norm.weight * proj.weight, FOLDED */
    bf16* P_postln = NEXT((size_t)HID, 2);
    bf16* P_wgate = NEXT((size_t)INTER * HID, 2);
    bf16* P_wup = NEXT((size_t)INTER * HID, 2);
    bf16* P_wdown = NEXT((size_t)HID * INTER, 2);
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
    bf16* R_attn = NEXT((size_t)T * HID, 2);
    bf16* R_h2 = NEXT((size_t)T * HID, 2);
    bf16* R_h3 = NEXT((size_t)T * HID, 2);
    bf16* R_ffg = NEXT((size_t)T * INTER, 2);
    bf16* R_ffa = NEXT((size_t)T * INTER, 2);
    bf16* R_ffo = NEXT((size_t)T * HID, 2);
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
    const size_t tp = (size_t)T * P, thid = (size_t)T * HID, ti = (size_t)T * INTER;
    int t_hidden = DUP(P_hidden, thid * 2), t_lnw = DUP(P_lnw, HID * 2);
    int t_wq = DUP(P_wq, (size_t)P * HID * 2), t_wk = DUP(P_wk, (size_t)P * HID * 2);
    int t_wv = DUP(P_wv, (size_t)P * HID * 2), t_wg = DUP(P_wg, (size_t)P * HID * 2);
    int t_wo = DUP(P_wo, (size_t)HID * P * 2), t_wfa = DUP(P_wfa, (size_t)D * HID * 2);
    int t_wfb = DUP(P_wfb, (size_t)P * D * 2), t_wb = DUP(P_wb, (size_t)H * HID * 2);
    int t_cw[3], t_cs[3];
    for (int s = 0; s < 3; s++) {
        t_cw[s] = DUP(P_cw[s], (size_t)P * W * 4);
        t_cs[s] = DUP(P_cs_in[s], (size_t)P * W * 4);
    }
    int t_alog = DUP(P_alog, (size_t)H * 4), t_dtb = DUP(P_dtb, (size_t)P * 4);
    int t_onorm = DUP(P_onorm, (size_t)D * 4);
    int t_state = DUP(P_state_in, (size_t)H * D * D * 4);
    int t_scorew = DUP(P_score_w, (size_t)HID * 4);
    int t_postln = DUP(P_postln, (size_t)HID * 2);
    int t_wgate = DUP(P_wgate, (size_t)INTER * HID * 2);
    int t_wup = DUP(P_wup, (size_t)INTER * HID * 2);
    int t_wdown = DUP(P_wdown, (size_t)HID * INTER * 2);
    int t_x = DNEW(thid * 2);
    int t_raw[3], t_mix[3];
    for (int s = 0; s < 3; s++) { t_raw[s] = DNEW(tp * 2); t_mix[s] = DNEW(tp * 2); }
    int t_graw = DNEW(tp * 2), t_fa = DNEW((size_t)T * D * 2), t_fraw = DNEW(tp * 2);
    int t_braw = DNEW((size_t)T * H * 2);
    int t_gate = DNEW(tp * 4), t_beta = DNEW((size_t)T * H * 4);
    int t_o = DNEW(tp * 2), t_y = DNEW(tp * 2), t_attn = DNEW(thid * 2);
    int t_h2 = DNEW(thid * 2), t_h3 = DNEW(thid * 2);
    int t_ffg = DNEW(ti * 2), t_ffu = DNEW(ti * 2), t_ffa = DNEW(ti * 2), t_ffo = DNEW(thid * 2);
    int t_out = DNEW(thid * 2);
    /* block_residual = [prefix_sum] = [hidden] at layer 0. [T,1,HID] IS [T,HID]. */
    const int t_blkres = t_hidden;

    const uint16_t ALL = (uint16_t)NCU;
#define GEMV(o_, x_, w_, N_, K_, dep_) ({                                                        \
        int _i = emitop(PLOW_DOP_GEMV, ALL);                                                     \
        g_inst[_i].t[0] = o_; g_inst[_i].t[1] = x_; g_inst[_i].t[2] = w_;                         \
        g_inst[_i].t[3] = PLOW_TENSOR_NONE; g_inst[_i].t[4] = PLOW_TENSOR_NONE;                   \
        g_inst[_i].i[0] = T; g_inst[_i].i[1] = (N_); g_inst[_i].i[2] = (K_);                      \
        addwait(_i, dep_, ALL); _i; })

    /* ================= the KDA mixer, identical to kda_block_gfx950_test.c ================= */
    int i_ln = emitop(PLOW_DOP_RMSNORM, ALL);
    g_inst[i_ln].t[0] = t_x; g_inst[i_ln].t[1] = t_hidden; g_inst[i_ln].t[2] = t_lnw;
    g_inst[i_ln].i[0] = T; g_inst[i_ln].i[1] = HID; g_inst[i_ln].fj[0].f = EPS;

    int i_q = GEMV(t_raw[0], t_x, t_wq, P, HID, i_ln);
    int i_k = GEMV(t_raw[1], t_x, t_wk, P, HID, i_ln);
    int i_v = GEMV(t_raw[2], t_x, t_wv, P, HID, i_ln);
    int i_g = GEMV(t_graw, t_x, t_wg, P, HID, i_ln);
    int i_fa = GEMV(t_fa, t_x, t_wfa, D, HID, i_ln);
    int i_bb = GEMV(t_braw, t_x, t_wb, H, HID, i_ln);
    int i_fb = GEMV(t_fraw, t_fa, t_wfb, P, D, i_fa);

    /* ---- P8/P9/P10: SIX packets or THREE. -----------------------------------------------
     *
     * `PLOW_KDA_FUSE=0` selects the decomposed spelling, mirroring `crates/devgen/src/kda.rs`'s
     * `fuse_kda`. Both are scored against the SAME fixture in the SAME binary, which is what makes
     * "the fusion did not move the numbers" a measurement rather than an assurance. The fused
     * spelling is the default, because it is what the emitter emits.
     *
     * THE GATE PACKET SURVIVES IN FUSED MODE, as a PROBE. `KdaStateStepG` computes g and beta in
     * registers and never writes them, so rows S2 would otherwise read an unwritten buffer. It is
     * emitted after the step, gated on it, writing only `t_gate`/`t_beta` — nothing downstream
     * reads them. That keeps all 19 rows comparable across the two spellings, and it is honest
     * about what S2 then proves: S2 checks the STANDALONE gate against the oracle, and S3/state
     * check the INLINE one, because the inline gate is the only thing the recurrence consumed. */
    const int FUSE = !(getenv("PLOW_KDA_FUSE") && !strcmp(getenv("PLOW_KDA_FUSE"), "0"));
    unsigned items = (unsigned)(P / BV);
    uint16_t step_blocks = (uint16_t)(items < NCU ? items : NCU);
    int i_conv[3], src[3] = { i_q, i_k, i_v }, i_step;
    (void)i_conv;

    if (FUSE) {
        int i_c3 = emitop(PLOW_DOP_KDA_CONV3, ALL);
        g_inst[i_c3].t[0] = t_mix[0]; g_inst[i_c3].t[1] = t_mix[1]; g_inst[i_c3].t[2] = t_mix[2];
        g_inst[i_c3].t[3] = t_raw[0]; g_inst[i_c3].t[4] = t_raw[1]; g_inst[i_c3].t[5] = t_raw[2];
        g_inst[i_c3].t[6] = t_cw[0];  g_inst[i_c3].t[7] = t_cw[1];
        g_inst[i_c3].i[0] = T; g_inst[i_c3].i[1] = P; g_inst[i_c3].i[2] = W; g_inst[i_c3].i[3] = 1;
        g_inst[i_c3].i[4] = (uint32_t)t_cw[2];
        g_inst[i_c3].i[5] = (uint32_t)t_cs[0];
        g_inst[i_c3].i[6] = (uint32_t)t_cs[1];
        g_inst[i_c3].i[7] = (uint32_t)t_cs[2];
        for (int s = 0; s < 3; s++) addwait(i_c3, src[s], ALL);

        i_step = emitop(PLOW_DOP_KDA_STATE_STEP_G, step_blocks);
        g_inst[i_step].t[0] = t_o; g_inst[i_step].t[1] = t_mix[0];
        g_inst[i_step].t[2] = t_mix[1]; g_inst[i_step].t[3] = t_mix[2];
        g_inst[i_step].t[4] = t_fraw; g_inst[i_step].t[5] = t_braw;
        g_inst[i_step].t[6] = t_state; g_inst[i_step].t[7] = t_alog;
        g_inst[i_step].i[0] = T; g_inst[i_step].i[1] = H; g_inst[i_step].i[2] = D;
        g_inst[i_step].i[3] = BV; g_inst[i_step].i[4] = 1;
        g_inst[i_step].i[5] = (uint32_t)t_dtb; g_inst[i_step].i[6] = GMODE;
        g_inst[i_step].fj[0].f = SCALE; g_inst[i_step].fj[1].f = LB;
        addwait(i_step, i_c3, ALL);
        addwait(i_step, i_fb, ALL);
        addwait(i_step, i_bb, ALL);

        /* The S2 probe. Gated on the step so it cannot be mistaken for a producer of it. */
        int i_probe = emitop(PLOW_DOP_KDA_GATE, ALL);
        g_inst[i_probe].t[0] = t_gate; g_inst[i_probe].t[1] = t_beta;
        g_inst[i_probe].t[2] = t_fraw; g_inst[i_probe].t[3] = t_braw;
        g_inst[i_probe].t[4] = t_alog; g_inst[i_probe].t[5] = t_dtb;
        g_inst[i_probe].i[0] = T; g_inst[i_probe].i[1] = H; g_inst[i_probe].i[2] = D;
        g_inst[i_probe].i[3] = GMODE; g_inst[i_probe].fj[0].f = LB;
        addwait(i_probe, i_step, step_blocks);
    } else {
        for (int s = 0; s < 3; s++) {
            i_conv[s] = emitop(PLOW_DOP_KDA_CONV, ALL);
            g_inst[i_conv[s]].t[0] = t_mix[s]; g_inst[i_conv[s]].t[1] = t_raw[s];
            g_inst[i_conv[s]].t[2] = t_cw[s]; g_inst[i_conv[s]].t[3] = t_cs[s];
            g_inst[i_conv[s]].i[0] = T; g_inst[i_conv[s]].i[1] = P;
            g_inst[i_conv[s]].i[2] = W; g_inst[i_conv[s]].i[3] = 1;
            addwait(i_conv[s], src[s], ALL);
        }

        int i_gate = emitop(PLOW_DOP_KDA_GATE, ALL);
        g_inst[i_gate].t[0] = t_gate; g_inst[i_gate].t[1] = t_beta;
        g_inst[i_gate].t[2] = t_fraw; g_inst[i_gate].t[3] = t_braw;
        g_inst[i_gate].t[4] = t_alog; g_inst[i_gate].t[5] = t_dtb;
        g_inst[i_gate].i[0] = T; g_inst[i_gate].i[1] = H; g_inst[i_gate].i[2] = D;
        g_inst[i_gate].i[3] = GMODE; g_inst[i_gate].fj[0].f = LB;
        addwait(i_gate, i_fb, ALL);
        addwait(i_gate, i_bb, ALL);

        i_step = emitop(PLOW_DOP_KDA_STATE_STEP, step_blocks);
        g_inst[i_step].t[0] = t_o; g_inst[i_step].t[1] = t_mix[0];
        g_inst[i_step].t[2] = t_mix[1]; g_inst[i_step].t[3] = t_mix[2];
        g_inst[i_step].t[4] = t_gate; g_inst[i_step].t[5] = t_beta; g_inst[i_step].t[6] = t_state;
        g_inst[i_step].i[0] = T; g_inst[i_step].i[1] = H; g_inst[i_step].i[2] = D;
        g_inst[i_step].i[3] = BV; g_inst[i_step].i[4] = 1;
        g_inst[i_step].fj[0].f = SCALE;
        for (int s = 0; s < 3; s++) addwait(i_step, i_conv[s], ALL);
        addwait(i_step, i_gate, ALL);
    }

    int i_norm = emitop(PLOW_DOP_KDA_GATED_NORM, ALL);
    g_inst[i_norm].t[0] = t_y; g_inst[i_norm].t[1] = t_o;
    g_inst[i_norm].t[2] = t_onorm; g_inst[i_norm].t[3] = t_graw;
    g_inst[i_norm].i[0] = T; g_inst[i_norm].i[1] = H; g_inst[i_norm].i[2] = D;
    g_inst[i_norm].fj[0].f = EPS;
    addwait(i_norm, i_step, step_blocks);
    addwait(i_norm, i_g, ALL);

    int i_o = GEMV(t_attn, t_y, t_wo, HID, P, i_norm);

    /* ================= THE BLOCK. This is what is new. =================
     *
     * There is NO `residual + attn` here, and its absence is the point. At layer 0 the snapshot
     * set prefix_sum to None, so the prefix sum entering the mlp-side AttnRes is the BARE
     * attention output. The layer input reaches the output only through `block_residual`. */
    uint16_t ar_blocks = (uint16_t)(T < (int)NCU ? T : (int)NCU);
    int i_ar = emitop(PLOW_DOP_ATTN_RES, ar_blocks);
    g_inst[i_ar].t[0] = t_h2; g_inst[i_ar].t[1] = t_attn; g_inst[i_ar].t[2] = t_blkres;
    g_inst[i_ar].t[3] = t_scorew;
    g_inst[i_ar].i[0] = T; g_inst[i_ar].i[1] = HID; g_inst[i_ar].i[2] = NB;
    /* i[4] IS THE RING CAPACITY, AND OMITTING IT POISONS. `d_attn_res` strides the ring as
     * `[T][NBCAP][HID]` and refuses `NBCAP < NB`, because striding a T > 1 ring by the LIVE
     * count would give every layer a differently-strided view of one buffer — no fault, no
     * NaN, a fluent wrong model. Unset, i[4] read 0 and the arm poisoned every run: S6-S10
     * came back NaN on real weights while S0-S5 (the whole KDA mixer) passed at 4.5e-3.
     *
     * This file is an INDEPENDENT transcription of the graph the emitter builds, which is the
     * point of it — and the cost is that a new operand has to be carried here by hand. That has
     * now happened twice on this same op: the comment at the top of `emitop` records t4
     * (`push_src`) going live because `t[]` was never zeroed, with the same NaN signature at
     * the same stages. Capacity == NB here because `t_blkres` aliases `t_hidden`, one row per
     * token, which is exactly the live count at this rung. */
    g_inst[i_ar].i[4] = NB;
    g_inst[i_ar].fj[0].f = EPS;
    addwait(i_ar, i_o, ALL);

    int i_pn = emitop(PLOW_DOP_RMSNORM, ALL);
    g_inst[i_pn].t[0] = t_h3; g_inst[i_pn].t[1] = t_h2; g_inst[i_pn].t[2] = t_postln;
    g_inst[i_pn].i[0] = T; g_inst[i_pn].i[1] = HID; g_inst[i_pn].fj[0].f = EPS;
    addwait(i_pn, i_ar, ar_blocks);

    /* gate and up UNFUSED, as two Gemvs plus a standalone SituGlu, rather than one GemvGlu with a
     * third `act` code. Two reasons, and the first is the load-bearing one:
     *   - it makes `ff_gate` and `ff_act` separately diffable, so a GEMV bug and an activation bug
     *     cannot be confused in the table below — the whole point of a stage-by-stage gate;
     *   - the fused epilogue would have to grow a second transform on the UP branch at seven
     *     sites, two of which are the PREFILL GEMM at 256 VGPR / occ 2 / 2 spills.
     * The fused form is the PERFORMANCE answer and is not this. See op_k3.h. */
    int i_ffg = GEMV(t_ffg, t_h3, t_wgate, INTER, HID, i_pn);
    int i_ffu = GEMV(t_ffu, t_h3, t_wup, INTER, HID, i_pn);

    int i_situ = emitop(PLOW_DOP_SITU_GLU, ALL);
    g_inst[i_situ].t[0] = t_ffa; g_inst[i_situ].t[1] = t_ffg; g_inst[i_situ].t[2] = t_ffu;
    g_inst[i_situ].i[0] = (uint32_t)ti;
    g_inst[i_situ].fj[0].f = BETA; g_inst[i_situ].fj[1].f = LBETA;
    addwait(i_situ, i_ffg, ALL);
    addwait(i_situ, i_ffu, ALL);

    int i_ffo = GEMV(t_ffo, t_ffa, t_wdown, HID, INTER, i_situ);

    /* The block output IS the prefix sum: attn + ffn. */
    int i_res = emitop(PLOW_DOP_RESIDUAL, ALL);
    g_inst[i_res].t[0] = t_out; g_inst[i_res].t[1] = t_attn; g_inst[i_res].t[2] = t_ffo;
    g_inst[i_res].i[0] = (uint32_t)thid; g_inst[i_res].fj[0].f = 1.0f;
    addwait(i_res, i_ffo, ALL);

    const int n_ops = g_nops;
    printf("program: %d packets, %d tensors   KDA chain: %s\n", n_ops, g_nt,
           FUSE ? "FUSED (KdaConv3 + KdaStateStepG, 3 packets + an S2 probe)"
                : "DECOMPOSED (3x KdaConv + KdaGate + KdaStateStep, 6 packets)");
    printf("  %s blocks=%u of %u CUs (%.1f%%), items=%u\n",
           FUSE ? "KdaStateStepG" : "KdaStateStep ", step_blocks, NCU,
           100.0 * step_blocks / NCU, items);
    printf("  AttnRes      blocks=%u of %u CUs (%.1f%%)  <- one workgroup per token, KNOWN gap\n",
           ar_blocks, NCU, 100.0 * ar_blocks / NCU);

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
    bf16* o_x = malloc(thid * 2);
    bf16* o_qc = malloc(tp * 2); bf16* o_kc = malloc(tp * 2); bf16* o_vc = malloc(tp * 2);
    bf16* o_o = malloc(tp * 2); bf16* o_y = malloc(tp * 2);
    bf16* o_attn = malloc(thid * 2); bf16* o_h2 = malloc(thid * 2); bf16* o_h3 = malloc(thid * 2);
    bf16* o_ffg = malloc(ti * 2); bf16* o_ffa = malloc(ti * 2); bf16* o_ffo = malloc(thid * 2);
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
    plow_hsa_download(h, 0, o_attn, g_tens[t_attn], thid * 2);
    plow_hsa_download(h, 0, o_h2, g_tens[t_h2], thid * 2);
    plow_hsa_download(h, 0, o_h3, g_tens[t_h3], thid * 2);
    plow_hsa_download(h, 0, o_ffg, g_tens[t_ffg], ti * 2);
    plow_hsa_download(h, 0, o_ffa, g_tens[t_ffa], ti * 2);
    plow_hsa_download(h, 0, o_ffo, g_tens[t_ffo], thid * 2);
    plow_hsa_download(h, 0, o_out, g_tens[t_out], thid * 2);
    for (int s = 0; s < 3; s++) {
        o_cs[s] = malloc((size_t)P * W * 4);
        plow_hsa_download(h, 0, o_cs[s], g_tens[t_cs[s]], (size_t)P * W * 4);
    }
    plow_hsa_download(h, 0, zc, d_ctr, (size_t)n_ops * PLOW_CTR_STRIDE * 4);

    int exec_ok = 1;
    for (int op = 0; op < n_ops; op++)
        if (zc[op * PLOW_CTR_STRIDE] != g_inst[op].blocks) {
            printf("  op %d (dop %u): executed %u of %u\n", op, g_inst[op].op,
                   zc[op * PLOW_CTR_STRIDE], g_inst[op].blocks);
            exec_ok = 0;
        }
    printf("all packets executed on every slice: %s\n", exec_ok ? "YES" : "NO");

    /* ---- residual table ---- */
    double w_[24];
    double r_x = relerr(o_x, R_x, thid, &w_[0]);
    double r_q = relerr(o_qc, R_qc, tp, &w_[1]);
    double r_k = relerr(o_kc, R_kc, tp, &w_[2]);
    double r_v = relerr(o_vc, R_vc, tp, &w_[3]);
    double r_g = relerr_f32(o_gate, R_gate, tp, &w_[4]);
    double r_b = relerr_f32(o_beta, R_beta, (size_t)T * H, &w_[5]);
    double r_o = relerr(o_o, R_o, tp, &w_[6]);
    double r_s = relerr_f32(o_state, R_state, (size_t)H * D * D, &w_[7]);
    double r_y = relerr(o_y, R_y, tp, &w_[8]);
    double r_at = relerr(o_attn, R_attn, thid, &w_[9]);
    double r_h2 = relerr(o_h2, R_h2, thid, &w_[10]);
    double r_h3 = relerr(o_h3, R_h3, thid, &w_[11]);
    double r_fg = relerr(o_ffg, R_ffg, ti, &w_[12]);
    double r_fa = relerr(o_ffa, R_ffa, ti, &w_[13]);
    double r_fo = relerr(o_ffo, R_ffo, thid, &w_[14]);
    double r_out = relerr(o_out, R_out, thid, &w_[15]);
    double r_cs_[3];
    for (int s = 0; s < 3; s++) r_cs_[s] = relerr_f32(o_cs[s], R_cs[s], (size_t)P * W, &w_[16 + s]);

    printf("\n  stage                            rms rel     worst rel\n");
    printf("  S0  pre-norm      x             %10.3e  %10.3e\n", r_x, w_[0]);
    printf("  S1  conv+silu     q             %10.3e  %10.3e\n", r_q, w_[1]);
    printf("  S1  conv+silu     k             %10.3e  %10.3e\n", r_k, w_[2]);
    printf("  S1  conv+silu     v             %10.3e  %10.3e\n", r_v, w_[3]);
    printf("  S2  gate g        (f32)         %10.3e  %10.3e\n", r_g, w_[4]);
    printf("  S2  beta          (f32)         %10.3e  %10.3e\n", r_b, w_[5]);
    printf("  S3  state read    o             %10.3e  %10.3e\n", r_o, w_[6]);
    printf("  S3  STATE V-FIRST (f32)         %10.3e  %10.3e\n", r_s, w_[7]);
    printf("  S1  conv_state q  (f32)         %10.3e  %10.3e\n", r_cs_[0], w_[16]);
    printf("  S1  conv_state k  (f32)         %10.3e  %10.3e\n", r_cs_[1], w_[17]);
    printf("  S1  conv_state v  (f32)         %10.3e  %10.3e\n", r_cs_[2], w_[18]);
    printf("  S4  gated norm    y             %10.3e  %10.3e\n", r_y, w_[8]);
    printf("  S5  KDA out       attn          %10.3e  %10.3e\n", r_at, w_[9]);
    printf("  S6  ATTNRES       h2            %10.3e  %10.3e\n", r_h2, w_[10]);
    printf("  S7  post-attn nrm h3            %10.3e  %10.3e\n", r_h3, w_[11]);
    printf("  S8  ffn gate      (pre-act)     %10.3e  %10.3e\n", r_fg, w_[12]);
    printf("  S8  SITU          act           %10.3e  %10.3e\n", r_fa, w_[13]);
    printf("  S9  ffn down      out           %10.3e  %10.3e\n", r_fo, w_[14]);
    printf("  S10 BLOCK out                   %10.3e  %10.3e\n", r_out, w_[15]);

    /* The V-first transpose check, carried over from the KDA gate. */
    double wt;
    float* R_state_T = malloc((size_t)H * D * D * 4);
    for (int hh = 0; hh < H; hh++)
        for (int a = 0; a < D; a++)
            for (int bq = 0; bq < D; bq++)
                R_state_T[((size_t)hh * D + a) * D + bq] = R_state[((size_t)hh * D + bq) * D + a];
    double r_sT = relerr_f32(o_state, R_state_T, (size_t)H * D * D, &wt);
    printf("\n  state vs the TRANSPOSED reference: %10.3e  (must be LARGE)\n", r_sT);

    /* THE CONTROL THAT MAKES THIS GATE FALSIFIABLE. Diff the device block output against the
     * PLAIN `hidden + attn + mlp` wiring every other model in this tree uses. If that number were
     * small, a block wired the plain way would have passed every row above. The oracle asserts the
     * same thing on the reference side; this is the device-side half. */
    double wp;
    bf16* R_plain = malloc(thid * 2);
    for (size_t i = 0; i < thid; i++) {
        float p = b2f(P_hidden[i]) + b2f(R_attn[i]) + b2f(R_ffo[i]);
        union { uint32_t u; float f; } c; c.f = p;
        R_plain[i] = (bf16)(c.u >> 16);
    }
    double r_plain = relerr(o_out, R_plain, thid, &wp);
    printf("  block out vs the PLAIN `hidden+attn+mlp` wiring: %10.3e\n", r_plain);
    printf("  (must be LARGE — it is exactly what this gate would NOT have caught)\n");

    const double TOL = 1.5e-2, TOL_F32 = 2e-2;
    int ok = exec_ok && r_x < TOL && r_q < TOL && r_k < TOL && r_v < TOL && r_g < TOL_F32 &&
             r_b < TOL_F32 && r_o < TOL && r_s < TOL_F32 && r_y < TOL && r_at < TOL &&
             r_h2 < TOL && r_h3 < TOL && r_fg < TOL && r_fa < TOL && r_fo < TOL && r_out < TOL &&
             r_cs_[0] < TOL_F32 && r_cs_[1] < TOL_F32 && r_cs_[2] < TOL_F32 &&
             r_sT > 10 * r_s && r_plain > 0.1;

    printf("\n=> %s\n", ok ? "K3 REAL-WEIGHT SINGLE-BLOCK OK — AttnRes + KDA + situ dense FFN "
                             "match the reference on real Kimi-K3 layer-0 weights"
                           : "*** K3 REAL-WEIGHT BLOCK MISMATCH ***");
    return ok ? 0 : 1;
}
