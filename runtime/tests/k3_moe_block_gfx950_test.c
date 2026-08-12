/* k3_moe_block_gfx950_test.c — REAL-WEIGHT gate for a COMPLETE Kimi-K3 MoE BLOCK.  [K3-MOE-GATE]
 *
 * Rung 2. Rung 1 (k3_block_gfx950_test.c) gated layer 0: KDA + ONE AttnRes + a dense situ FFN.
 * This gates layer 1, which adds the two things layer 0 does not have:
 *
 *   * BOTH AttnRes applications, with DIFFERENT folded weights. Layer 1 takes no block-residual
 *     snapshot (1 % 12 != 0), so `prefix_sum` survives the attention and the attn-side AttnRes —
 *     dead at layer 0 — is live. Layer 0 never reads `self_attention_res_*` at all.
 *   * STABLE LATENTMOE: 7168 --down--> 3584 --896 experts, top-16, MXFP4, K=3584, I=3072-->
 *     --combine--> --RMSNorm--> --up--> 7168, plus a shared expert that reads the PRE-DOWN hidden.
 *
 * WHERE THIS GATE HAS TO LOOK, and it is not where rung 1 looked. At layer 1 there is no snapshot,
 * so `prefix = prefix_in + attn` and the block output is `prefix_in + attn + moe` — which is
 * EXACTLY what a plain-residual wiring produces. The oracle measures the difference at **3.0e-3**.
 * A block-output-only gate would NOT catch AttnRes wired as a plain residual here. So `h_a` and
 * `h2` — the two AttnRes outputs, i.e. the attention input and the MLP input — are diffed as their
 * own rows, and the controls are taken there (the oracle prints 8.1e-1 and 7.6e-1 for how far
 * AttnRes moves each). The block output is the last row, not the argument.
 *
 * ONLY THE SELECTED EXPERTS ARE UPLOADED. 896 x 3 mxfp4 tensors is 15.7 GB; top-16 is 280 MB. The
 * pointer table is NULL for every unselected expert and the kernel skips a null base — so if
 * plow's router chose differently from the oracle, that slot's partial is zero and the residual
 * explodes rather than quietly drifting. The routing table is ALSO downloaded and diffed expert id
 * by expert id, so a divergence is localized instead of merely detected.
 *
 *   ./k3_moe_test [interp_decode.elf] [k3_moe_fixture.bin]
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
#include "k3_test_arch.h"
#include "../common/dev_isa.h"

typedef uint16_t bf16;
static float b2f(bf16 b) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)b << 16;
    return c.f;
}
static void upload_or_die(plow_hsa* h, void* dst, const void* src, size_t bytes) {
    if (plow_hsa_upload(h, 0, dst, src, bytes)) {
        fprintf(stderr, "upload %zu failed: %s\n", bytes, plow_hsa_last_error());
        exit(1);
    }
}

static PlowDevInst g_inst[512];
static PlowWait g_wait[4096];
static uint32_t g_succ[512];
typedef struct { uint32_t wait_ofs, succ_ofs; uint16_t wait_len, succ_len; } PlowGate;
static PlowGate g_gate[512];
static int g_nops = 0, g_nw = 0;
static void* g_tens[512];
static int g_nt = 0;
static int reg(void* p) { g_tens[g_nt] = p; return g_nt++; }
/* Every tensor slot defaults to ABSENT. Slot 0 is a legal handle, so a `memset`-zeroed DevInst
 * silently aliases tensor 0 into any operand the op reads and the emitter forgot — and
 * `d_moe_combine` reads t1/t2 as optional pointers, which is precisely such an op. */
static int emitop(uint16_t op, uint16_t blocks) {
    int i = g_nops++;
    g_inst[i].op = op; g_inst[i].blocks = blocks;
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

/* The MoE activation codes op_moe.h defines. situ is 2 and it is NOT a `moe_act` value —
 * `moe_act` returns NaN for it on purpose, because situ transforms the up branch too. */
#define K3_MOE_ACT_SITU 2u
#define K3_MOE_ENC_MXFP4 2u

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "interp_decode.elf";
    const char* fix = argc > 2 ? argv[2] : "k3_moe_fixture.bin";
    setbuf(stdout, NULL);

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus = 0, ldsz = 0;
    plow_hsa_device_info(h, 0, gfx, &cus, &ldsz);
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
    if (plow_hsa_get_kernel(h, 0, PLOW_K3_DECODE_KERNEL, &kern)) { printf("no kernel\n"); return 1; }

    int fd = open(fix, O_RDONLY);
    if (fd < 0) { perror(fix); return 1; }
    struct stat st; fstat(fd, &st);
    char* base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    int32_t* hdr = (int32_t*)base;
    if (hdr[0] != 0x4B334D31) { printf("bad magic %x (want K3M1)\n", hdr[0]); return 1; }
    const int T = hdr[1], H = hdr[2], D = hdr[3], HID = hdr[4], W = hdr[5], GMODE = hdr[6],
              BV = hdr[7], IMOE = hdr[8], HE = hdr[9], NEXP = hdr[10], TOPK = hdr[11],
              NSEL = hdr[12], SHI = hdr[13], NB = hdr[14], RFLAGS = hdr[15];
    float* fh = (float*)(base + 16 * 4);
    const float EPS = fh[0], LB = fh[1], SCALE = fh[2], BETA = fh[3], LBETA = fh[4],
                RSCALE = fh[5];
    const int P = H * D;
    printf("K3 MoE block: T=%d hidden=%d latent=%d experts=%d top-%d I_moe=%d shared_inter=%d "
           "nb=%d\n", T, HID, HE, NEXP, TOPK, IMOE, SHI, NB);
    printf("              situ beta=%.1f linear_beta=%.1f  router flags=%d scale=%.2f  "
           "%d experts materialized\n", BETA, LBETA, RFLAGS, RSCALE, NSEL);
    if (T != 1) { printf("this gate is the DECODE MoE path: T must be 1\n"); return 1; }

    const size_t W1P = (size_t)IMOE * (HE / 2), W1S = (size_t)IMOE * (HE / 32);
    const size_t W2P = (size_t)HE * (IMOE / 2), W2S = (size_t)HE * (IMOE / 32);

    size_t off = 16 * 4 + 6 * 4;
#define NEXT(cnt, elt) ({ void* _p = base + off; off += (size_t)(cnt) * (elt); _p; })
    bf16* P_prefix_in = NEXT((size_t)T * HID, 2);
    bf16* P_blkres = NEXT((size_t)T * NB * HID, 2);
    float* P_asw = NEXT((size_t)HID, 4);
    float* P_msw = NEXT((size_t)HID, 4);
    bf16* P_lnw = NEXT((size_t)HID, 2);
    bf16* P_postln = NEXT((size_t)HID, 2);
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
    bf16* P_wrouter = NEXT((size_t)NEXP * HID, 2);
    float* P_rbias = NEXT((size_t)NEXP, 4);
    bf16* P_wdownl = NEXT((size_t)HE * HID, 2);
    bf16* P_latnorm = NEXT((size_t)HE, 2);
    bf16* P_wupl = NEXT((size_t)HID * HE, 2);
    bf16* P_wshg = NEXT((size_t)SHI * HID, 2);
    bf16* P_wshu = NEXT((size_t)SHI * HID, 2);
    bf16* P_wshd = NEXT((size_t)HID * SHI, 2);
    uint32_t* P_eid = NEXT((size_t)NSEL, 4);
    uint8_t* P_exp[64][6];
    for (int j = 0; j < NSEL; j++) {
        P_exp[j][0] = NEXT(W1P, 1); P_exp[j][1] = NEXT(W1S, 1);   /* w1 = GATE */
        P_exp[j][2] = NEXT(W1P, 1); P_exp[j][3] = NEXT(W1S, 1);   /* w3 = UP   */
        P_exp[j][4] = NEXT(W2P, 1); P_exp[j][5] = NEXT(W2S, 1);   /* w2 = DOWN */
    }
    bf16* R_ha = NEXT((size_t)T * HID, 2);
    bf16* R_x = NEXT((size_t)T * HID, 2);
    float* R_state = NEXT((size_t)H * D * D, 4);
    bf16* R_attn = NEXT((size_t)T * HID, 2);
    bf16* R_prefix = NEXT((size_t)T * HID, 2);
    bf16* R_h2 = NEXT((size_t)T * HID, 2);
    bf16* R_h3 = NEXT((size_t)T * HID, 2);
    bf16* R_logit = NEXT((size_t)T * NEXP, 2);
    uint32_t* R_sel = NEXT((size_t)TOPK, 4);
    float* R_gate = NEXT((size_t)TOPK, 4);
    bf16* R_xe = NEXT((size_t)T * HE, 2);
    bf16* R_fu = NEXT((size_t)TOPK * IMOE, 2);
    float* R_part = NEXT((size_t)TOPK * HE, 4);
    bf16* R_ylat = NEXT((size_t)T * HE, 2);
    bf16* R_yn = NEXT((size_t)T * HE, 2);
    bf16* R_yh = NEXT((size_t)T * HID, 2);
    bf16* R_shd = NEXT((size_t)T * HID, 2);
    bf16* R_out = NEXT((size_t)T * HID, 2);
    float* R_cs[3];
    for (int s = 0; s < 3; s++) R_cs[s] = NEXT((size_t)P * W, 4);
    if (off != (size_t)st.st_size) {
        printf("FIXTURE SIZE MISMATCH: consumed %zu, file %zu\n", off, (size_t)st.st_size);
        return 1;
    }

#define DUP(ptr, bytes) ({ void* _d = plow_hsa_alloc(h, 0, (bytes)); \
                           upload_or_die(h, _d, (ptr), (bytes)); reg(_d); })
#define DNEW(bytes) ({ void* _d = plow_hsa_alloc(h, 0, (bytes)); reg(_d); })
    const size_t tp = (size_t)T * P, thid = (size_t)T * HID;
    int t_prefix_in = DUP(P_prefix_in, thid * 2);
    int t_blkres = DUP(P_blkres, (size_t)T * NB * HID * 2);
    int t_asw = DUP(P_asw, (size_t)HID * 4), t_msw = DUP(P_msw, (size_t)HID * 4);
    int t_lnw = DUP(P_lnw, HID * 2), t_postln = DUP(P_postln, HID * 2);
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
    int t_wrouter = DUP(P_wrouter, (size_t)NEXP * HID * 2);
    int t_rbias = DUP(P_rbias, (size_t)NEXP * 4);
    int t_wdownl = DUP(P_wdownl, (size_t)HE * HID * 2);
    int t_latnorm = DUP(P_latnorm, (size_t)HE * 2);
    int t_wupl = DUP(P_wupl, (size_t)HID * HE * 2);
    int t_wshg = DUP(P_wshg, (size_t)SHI * HID * 2);
    int t_wshu = DUP(P_wshu, (size_t)SHI * HID * 2);
    int t_wshd = DUP(P_wshd, (size_t)HID * SHI * 2);

    /* THE EXPERT POINTER TABLES. `[NEXP][3]` u64 of DEVICE addresses in {gate, up, down} order —
     * op_moe.h reads wtab[eid*3+0]=gate, +1=up, +2=down, and `stab` the same for the E8M0 scale
     * rows under MXFP4. A zero base means "not mine" and the kernel skips the slot, which is what
     * every unselected expert gets here. */
    unsigned long long* wtab = calloc((size_t)NEXP * 3, 8);
    unsigned long long* stab = calloc((size_t)NEXP * 3, 8);
    size_t exp_bytes = 0;
    for (int j = 0; j < NSEL; j++) {
        void* dw[6];
        const size_t sz[6] = { W1P, W1S, W1P, W1S, W2P, W2S };
        for (int q = 0; q < 6; q++) {
            dw[q] = plow_hsa_alloc(h, 0, sz[q]);
            upload_or_die(h, dw[q], P_exp[j][q], sz[q]);
            exp_bytes += sz[q];
        }
        const unsigned e = P_eid[j];
        if (e >= (unsigned)NEXP) { printf("bad expert id %u\n", e); return 1; }
        wtab[(size_t)e * 3 + 0] = (unsigned long long)(size_t)dw[0];
        stab[(size_t)e * 3 + 0] = (unsigned long long)(size_t)dw[1];
        wtab[(size_t)e * 3 + 1] = (unsigned long long)(size_t)dw[2];
        stab[(size_t)e * 3 + 1] = (unsigned long long)(size_t)dw[3];
        wtab[(size_t)e * 3 + 2] = (unsigned long long)(size_t)dw[4];
        stab[(size_t)e * 3 + 2] = (unsigned long long)(size_t)dw[5];
    }
    int t_ewt = DUP(wtab, (size_t)NEXP * 3 * 8);
    int t_est = DUP(stab, (size_t)NEXP * 3 * 8);
    printf("              uploaded %.1f MB of mxfp4 expert weights (%.1f GB if all %d were "
           "materialized)\n", exp_bytes / 1e6, exp_bytes / 1e9 * NEXP / NSEL, NEXP);

    int t_ha = DNEW(thid * 2), t_x = DNEW(thid * 2);
    int t_raw[3], t_mix[3];
    for (int s = 0; s < 3; s++) { t_raw[s] = DNEW(tp * 2); t_mix[s] = DNEW(tp * 2); }
    int t_graw = DNEW(tp * 2), t_fa = DNEW((size_t)T * D * 2), t_fraw = DNEW(tp * 2);
    int t_braw = DNEW((size_t)T * H * 2);
    int t_gate = DNEW(tp * 4), t_beta = DNEW((size_t)T * H * 4);
    int t_o = DNEW(tp * 2), t_y = DNEW(tp * 2), t_attn = DNEW(thid * 2);
    int t_prefix = DNEW(thid * 2), t_h2 = DNEW(thid * 2), t_h3 = DNEW(thid * 2);
    int t_logit = DNEW((size_t)NEXP * 2);
    int t_tab = DNEW((size_t)TOPK * 8);
    int t_xe = DNEW((size_t)HE * 2);
    int t_fu = DNEW((size_t)TOPK * IMOE * 2);
    int t_part = DNEW((size_t)TOPK * HE * 4);
    int t_ylat = DNEW((size_t)HE * 2), t_yn = DNEW((size_t)HE * 2), t_yh = DNEW(thid * 2);
    int t_shg = DNEW((size_t)SHI * 2), t_shu = DNEW((size_t)SHI * 2), t_sha = DNEW((size_t)SHI * 2);
    int t_shd = DNEW(thid * 2), t_moe = DNEW(thid * 2), t_out = DNEW(thid * 2);

    const uint16_t ALL = (uint16_t)NCU;
#define GEMV(o_, x_, w_, N_, K_, dep_) ({                                                        \
        int _i = emitop(PLOW_DOP_GEMV, ALL);                                                     \
        g_inst[_i].t[0] = o_; g_inst[_i].t[1] = x_; g_inst[_i].t[2] = w_;                         \
        g_inst[_i].i[0] = T; g_inst[_i].i[1] = (N_); g_inst[_i].i[2] = (K_);                      \
        addwait(_i, dep_, ALL); _i; })
#define RMSN(o_, x_, g_, feat_, dep_, dblk_) ({                                                  \
        int _i = emitop(PLOW_DOP_RMSNORM, ALL);                                                  \
        g_inst[_i].t[0] = o_; g_inst[_i].t[1] = x_; g_inst[_i].t[2] = g_;                         \
        g_inst[_i].i[0] = T; g_inst[_i].i[1] = (feat_); g_inst[_i].fj[0].f = EPS;                \
        addwait(_i, dep_, dblk_); _i; })

    const uint16_t AR = (uint16_t)(T < (int)NCU ? T : (int)NCU);
    /* A0 — the ATTN-SIDE AttnRes, with `self_attention_res_*`. Dead at layer 0, live here. */
    int i_ar1 = emitop(PLOW_DOP_ATTN_RES, AR);
    g_inst[i_ar1].t[0] = t_ha; g_inst[i_ar1].t[1] = t_prefix_in; g_inst[i_ar1].t[2] = t_blkres;
    g_inst[i_ar1].t[3] = t_asw;
    g_inst[i_ar1].i[0] = T; g_inst[i_ar1].i[1] = HID; g_inst[i_ar1].i[2] = NB;
    /* i[4] IS THE RING CAPACITY. `d_attn_res` strides `[T][NBCAP][HID]` and refuses
     * `NBCAP < NB`; unset it reads 0 and the arm poisons, which is NaN from A0 down through
     * every later stage. `t_blkres` is allocated `T * NB * HID`, so capacity IS NB here.
     * Layer 1 is a non-snapshot layer (1 % 12 != 0), so no push and both mixes see NB rows. */
    g_inst[i_ar1].i[4] = NB;
    g_inst[i_ar1].fj[0].f = EPS;

    int i_ln = RMSN(t_x, t_ha, t_lnw, HID, i_ar1, AR);

    int i_q = GEMV(t_raw[0], t_x, t_wq, P, HID, i_ln);
    int i_k = GEMV(t_raw[1], t_x, t_wk, P, HID, i_ln);
    int i_v = GEMV(t_raw[2], t_x, t_wv, P, HID, i_ln);
    int i_g = GEMV(t_graw, t_x, t_wg, P, HID, i_ln);
    int i_fa = GEMV(t_fa, t_x, t_wfa, D, HID, i_ln);
    int i_bb = GEMV(t_braw, t_x, t_wb, H, HID, i_ln);
    int i_fb = GEMV(t_fraw, t_fa, t_wfb, P, D, i_fa);

    int i_conv[3], srcs[3] = { i_q, i_k, i_v };
    for (int s = 0; s < 3; s++) {
        i_conv[s] = emitop(PLOW_DOP_KDA_CONV, ALL);
        g_inst[i_conv[s]].t[0] = t_mix[s]; g_inst[i_conv[s]].t[1] = t_raw[s];
        g_inst[i_conv[s]].t[2] = t_cw[s]; g_inst[i_conv[s]].t[3] = t_cs[s];
        g_inst[i_conv[s]].i[0] = T; g_inst[i_conv[s]].i[1] = P;
        g_inst[i_conv[s]].i[2] = W; g_inst[i_conv[s]].i[3] = 1;
        addwait(i_conv[s], srcs[s], ALL);
    }
    int i_kgate = emitop(PLOW_DOP_KDA_GATE, ALL);
    g_inst[i_kgate].t[0] = t_gate; g_inst[i_kgate].t[1] = t_beta;
    g_inst[i_kgate].t[2] = t_fraw; g_inst[i_kgate].t[3] = t_braw;
    g_inst[i_kgate].t[4] = t_alog; g_inst[i_kgate].t[5] = t_dtb;
    g_inst[i_kgate].i[0] = T; g_inst[i_kgate].i[1] = H; g_inst[i_kgate].i[2] = D;
    g_inst[i_kgate].i[3] = GMODE; g_inst[i_kgate].fj[0].f = LB;
    addwait(i_kgate, i_fb, ALL);
    addwait(i_kgate, i_bb, ALL);

    unsigned items = (unsigned)(P / BV);
    uint16_t step_blocks = (uint16_t)(items < NCU ? items : NCU);
    int i_step = emitop(PLOW_DOP_KDA_STATE_STEP, step_blocks);
    g_inst[i_step].t[0] = t_o; g_inst[i_step].t[1] = t_mix[0];
    g_inst[i_step].t[2] = t_mix[1]; g_inst[i_step].t[3] = t_mix[2];
    g_inst[i_step].t[4] = t_gate; g_inst[i_step].t[5] = t_beta; g_inst[i_step].t[6] = t_state;
    g_inst[i_step].i[0] = T; g_inst[i_step].i[1] = H; g_inst[i_step].i[2] = D;
    g_inst[i_step].i[3] = BV; g_inst[i_step].i[4] = 1; g_inst[i_step].fj[0].f = SCALE;
    for (int s = 0; s < 3; s++) addwait(i_step, i_conv[s], ALL);
    addwait(i_step, i_kgate, ALL);

    int i_gnorm = emitop(PLOW_DOP_KDA_GATED_NORM, ALL);
    g_inst[i_gnorm].t[0] = t_y; g_inst[i_gnorm].t[1] = t_o;
    g_inst[i_gnorm].t[2] = t_onorm; g_inst[i_gnorm].t[3] = t_graw;
    g_inst[i_gnorm].i[0] = T; g_inst[i_gnorm].i[1] = H; g_inst[i_gnorm].i[2] = D;
    g_inst[i_gnorm].fj[0].f = EPS;
    addwait(i_gnorm, i_step, step_blocks);
    addwait(i_gnorm, i_g, ALL);

    int i_o = GEMV(t_attn, t_y, t_wo, HID, P, i_gnorm);

    /* A1 — prefix_sum SURVIVES at a non-snapshot layer, so it accumulates here. */
    int i_pfx = emitop(PLOW_DOP_RESIDUAL, ALL);
    g_inst[i_pfx].t[0] = t_prefix; g_inst[i_pfx].t[1] = t_prefix_in; g_inst[i_pfx].t[2] = t_attn;
    g_inst[i_pfx].i[0] = (uint32_t)thid; g_inst[i_pfx].fj[0].f = 1.0f;
    addwait(i_pfx, i_o, ALL);

    /* A2 — the MLP-SIDE AttnRes, with the OTHER fold (`mlp_res_*`). */
    int i_ar2 = emitop(PLOW_DOP_ATTN_RES, AR);
    g_inst[i_ar2].t[0] = t_h2; g_inst[i_ar2].t[1] = t_prefix; g_inst[i_ar2].t[2] = t_blkres;
    g_inst[i_ar2].t[3] = t_msw;
    g_inst[i_ar2].i[0] = T; g_inst[i_ar2].i[1] = HID; g_inst[i_ar2].i[2] = NB;
    g_inst[i_ar2].i[4] = NB; /* capacity, as at A0 above */
    g_inst[i_ar2].fj[0].f = EPS;
    addwait(i_ar2, i_pfx, ALL);

    int i_pn = RMSN(t_h3, t_h2, t_postln, HID, i_ar2, AR);

    /* ================= STABLE LATENTMOE ================= */
    int i_rl = GEMV(t_logit, t_h3, t_wrouter, NEXP, HID, i_pn);
    int i_rt = emitop(PLOW_DOP_MOE_ROUTER_TOPK, 1);   /* 1 CU by construction */
    g_inst[i_rt].t[0] = t_tab; g_inst[i_rt].t[1] = t_logit; g_inst[i_rt].t[3] = t_rbias;
    g_inst[i_rt].i[1] = NEXP; g_inst[i_rt].i[2] = TOPK; g_inst[i_rt].i[3] = RFLAGS;
    g_inst[i_rt].i[6] = 1; g_inst[i_rt].i[7] = 1;     /* n_group / topk_group: inert at 1 */
    g_inst[i_rt].fj[0].f = RSCALE;
    addwait(i_rt, i_rl, ALL);

    /* The DOWN projection is what makes this LatentMoE: every expert below runs at K = HE = 3584,
     * not at hidden = 7168. The kernels take H as a runtime operand, so they need no change; it is
     * the GRAPH that has to know there are two widths. */
    int i_xe = GEMV(t_xe, t_h3, t_wdownl, HE, HID, i_pn);

    int i_ed[32];
    for (int j = 0; j < TOPK; j++) {
        int ig = emitop(PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK, ALL);
        g_inst[ig].t[0] = t_fu; g_inst[ig].t[1] = t_xe; g_inst[ig].t[2] = t_tab;
        g_inst[ig].t[3] = t_ewt; g_inst[ig].t[4] = t_est;
        g_inst[ig].i[0] = (uint32_t)j; g_inst[ig].i[1] = IMOE; g_inst[ig].i[2] = HE;
        g_inst[ig].i[3] = NEXP; g_inst[ig].i[5] = K3_MOE_ACT_SITU; g_inst[ig].i[6] = K3_MOE_ENC_MXFP4;
        g_inst[ig].fj[0].f = BETA; g_inst[ig].fj[1].f = LBETA;
        addwait(ig, i_rt, 1);
        addwait(ig, i_xe, ALL);
        i_ed[j] = emitop(PLOW_DOP_MOE_EXPERT_DOWN_FP8_BLK, ALL);
        g_inst[i_ed[j]].t[0] = t_part; g_inst[i_ed[j]].t[1] = t_fu; g_inst[i_ed[j]].t[2] = t_tab;
        g_inst[i_ed[j]].t[3] = t_ewt; g_inst[i_ed[j]].t[4] = t_est;
        g_inst[i_ed[j]].i[0] = (uint32_t)j; g_inst[i_ed[j]].i[1] = HE; g_inst[i_ed[j]].i[2] = IMOE;
        g_inst[i_ed[j]].i[3] = NEXP; g_inst[i_ed[j]].i[6] = K3_MOE_ENC_MXFP4;
        addwait(i_ed[j], ig, ALL);
    }

    /* MoeCombine with NO residual and NO shared expert. At hidden width the residual add happens
     * here; at latent width there is nothing of width 3584 to add — the residual add is after the
     * UP projection. `residual == nullptr` was a device fault before this change, because TEN()
     * maps TENSOR_NONE to nullptr and the kernel dereferenced it unconditionally. */
    int i_cmb = emitop(PLOW_DOP_MOE_COMBINE, ALL);
    g_inst[i_cmb].t[0] = t_ylat; g_inst[i_cmb].t[3] = t_part;   /* t1 residual, t2 shared: ABSENT */
    g_inst[i_cmb].i[0] = HE; g_inst[i_cmb].i[1] = TOPK;
    for (int j = 0; j < TOPK; j++) addwait(i_cmb, i_ed[j], ALL);

    int i_ln2 = RMSN(t_yn, t_ylat, t_latnorm, HE, i_cmb, ALL);
    int i_up = GEMV(t_yh, t_yn, t_wupl, HID, HE, i_ln2);

    /* The shared expert reads the PRE-DOWN hidden (`identity` in KimiSparseMoeBlock.forward), so
     * it is gated on the post-attention norm, NOT on the latent path. Feeding it `xe` would be a
     * plausible mistake: right dtype, wrong width, and it would fail loudly — but feeding it
     * `h2` instead of `h3` would fail quietly. */
    int i_shg = GEMV(t_shg, t_h3, t_wshg, SHI, HID, i_pn);
    int i_shu = GEMV(t_shu, t_h3, t_wshu, SHI, HID, i_pn);
    int i_sha = emitop(PLOW_DOP_SITU_GLU, ALL);
    g_inst[i_sha].t[0] = t_sha; g_inst[i_sha].t[1] = t_shg; g_inst[i_sha].t[2] = t_shu;
    g_inst[i_sha].i[0] = (uint32_t)SHI;
    g_inst[i_sha].fj[0].f = BETA; g_inst[i_sha].fj[1].f = LBETA;
    addwait(i_sha, i_shg, ALL);
    addwait(i_sha, i_shu, ALL);
    int i_shd = GEMV(t_shd, t_sha, t_wshd, HID, SHI, i_sha);

    int i_moe = emitop(PLOW_DOP_RESIDUAL, ALL);
    g_inst[i_moe].t[0] = t_moe; g_inst[i_moe].t[1] = t_yh; g_inst[i_moe].t[2] = t_shd;
    g_inst[i_moe].i[0] = (uint32_t)thid; g_inst[i_moe].fj[0].f = 1.0f;
    addwait(i_moe, i_up, ALL);
    addwait(i_moe, i_shd, ALL);

    int i_out = emitop(PLOW_DOP_RESIDUAL, ALL);
    g_inst[i_out].t[0] = t_out; g_inst[i_out].t[1] = t_prefix; g_inst[i_out].t[2] = t_moe;
    g_inst[i_out].i[0] = (uint32_t)thid; g_inst[i_out].fj[0].f = 1.0f;
    addwait(i_out, i_moe, ALL);

    const int n_ops = g_nops;
    printf("program: %d packets, %d tensors;  AttnRes blocks=%u of %u (%.1f%%), "
           "MoeRouterTopk blocks=1 (0.4%%)\n", n_ops, g_nt, AR, NCU, 100.0 * AR / NCU);

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
    upload_or_die(h, d_tens, g_tens, (size_t)g_nt * sizeof(void*));
    void* d_inst = plow_hsa_alloc(h, 0, (size_t)n_ops * sizeof(PlowDevInst));
    void* d_stream = plow_hsa_alloc(h, 0, total * sizeof(PlowStreamEnt));
    void* d_sofs = plow_hsa_alloc(h, 0, 4u * NCU);
    void* d_slen = plow_hsa_alloc(h, 0, 4u * NCU);
    void* d_waits = plow_hsa_alloc(h, 0, (size_t)(g_nw ? g_nw : 1) * sizeof(PlowWait));
    void* d_succs = plow_hsa_alloc(h, 0, (size_t)n_ops * 4);
    void* d_ctr = plow_hsa_alloc(h, 0, (size_t)n_ops * PLOW_CTR_STRIDE * 4);
    upload_or_die(h, d_inst, g_inst, (size_t)n_ops * sizeof(PlowDevInst));
    upload_or_die(h, d_stream, stream, total * sizeof(PlowStreamEnt));
    upload_or_die(h, d_sofs, sofs, 4u * NCU);
    upload_or_die(h, d_slen, slen, 4u * NCU);
    if (g_nw) upload_or_die(h, d_waits, g_wait, (size_t)g_nw * sizeof(PlowWait));
    upload_or_die(h, d_succs, g_succ, (size_t)n_ops * 4);
    uint32_t* zc = calloc((size_t)n_ops * PLOW_CTR_STRIDE, 4);
    upload_or_die(h, d_ctr, zc, (size_t)n_ops * PLOW_CTR_STRIDE * 4);
    PlowProgram prog; memset(&prog, 0, sizeof(prog));
    prog.insts = d_inst; prog.stream = d_stream; prog.stream_ofs = d_sofs; prog.stream_len = d_slen;
    prog.waits = d_waits; prog.succs = d_succs; prog.counters = d_ctr;
    prog.tensors = (void* const*)d_tens;
    if (plow_hsa_launch(h, 0, &kern, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &prog,
                        sizeof(prog))) {
        printf("LAUNCH FAILED\n");
        return 1;
    }
    if (plow_hsa_wait(h, 0)) { printf("WAIT FAILED\n"); return 1; }

    bf16* o_ha = malloc(thid * 2); bf16* o_x = malloc(thid * 2);
    bf16* o_attn = malloc(thid * 2); bf16* o_prefix = malloc(thid * 2);
    bf16* o_h2 = malloc(thid * 2); bf16* o_h3 = malloc(thid * 2);
    bf16* o_logit = malloc((size_t)NEXP * 2);
    uint8_t* o_tab = malloc((size_t)TOPK * 8);
    bf16* o_xe = malloc((size_t)HE * 2);
    bf16* o_fu = malloc((size_t)TOPK * IMOE * 2);
    float* o_part = malloc((size_t)TOPK * HE * 4);
    bf16* o_ylat = malloc((size_t)HE * 2); bf16* o_yn = malloc((size_t)HE * 2);
    bf16* o_yh = malloc(thid * 2); bf16* o_shd = malloc(thid * 2); bf16* o_out = malloc(thid * 2);
    float* o_state = malloc((size_t)H * D * D * 4);
    plow_hsa_download(h, 0, o_ha, g_tens[t_ha], thid * 2);
    plow_hsa_download(h, 0, o_x, g_tens[t_x], thid * 2);
    plow_hsa_download(h, 0, o_state, g_tens[t_state], (size_t)H * D * D * 4);
    plow_hsa_download(h, 0, o_attn, g_tens[t_attn], thid * 2);
    plow_hsa_download(h, 0, o_prefix, g_tens[t_prefix], thid * 2);
    plow_hsa_download(h, 0, o_h2, g_tens[t_h2], thid * 2);
    plow_hsa_download(h, 0, o_h3, g_tens[t_h3], thid * 2);
    plow_hsa_download(h, 0, o_logit, g_tens[t_logit], (size_t)NEXP * 2);
    plow_hsa_download(h, 0, o_tab, g_tens[t_tab], (size_t)TOPK * 8);
    plow_hsa_download(h, 0, o_xe, g_tens[t_xe], (size_t)HE * 2);
    plow_hsa_download(h, 0, o_fu, g_tens[t_fu], (size_t)TOPK * IMOE * 2);
    plow_hsa_download(h, 0, o_part, g_tens[t_part], (size_t)TOPK * HE * 4);
    plow_hsa_download(h, 0, o_ylat, g_tens[t_ylat], (size_t)HE * 2);
    plow_hsa_download(h, 0, o_yn, g_tens[t_yn], (size_t)HE * 2);
    plow_hsa_download(h, 0, o_yh, g_tens[t_yh], thid * 2);
    plow_hsa_download(h, 0, o_shd, g_tens[t_shd], thid * 2);
    plow_hsa_download(h, 0, o_out, g_tens[t_out], thid * 2);
    plow_hsa_download(h, 0, zc, d_ctr, (size_t)n_ops * PLOW_CTR_STRIDE * 4);

    int exec_ok = 1;
    for (int op = 0; op < n_ops; op++)
        if (zc[op * PLOW_CTR_STRIDE] != g_inst[op].blocks) {
            printf("  op %d (dop %u): executed %u of %u\n", op, g_inst[op].op,
                   zc[op * PLOW_CTR_STRIDE], g_inst[op].blocks);
            exec_ok = 0;
        }
    printf("all packets executed on every slice: %s\n", exec_ok ? "YES" : "NO");

    /* ---- THE ROUTING TABLE, expert id by expert id. Diffed BEFORE the residual table, because a
     * routing divergence makes every downstream row meaningless and must not be read as an
     * arithmetic error. The reference selection is computed from the SAME bf16 logits the device
     * is given (the oracle reports the fp32-vs-bf16 disagreement separately). ---- */
    int route_ok = 1, order_ok = 1;
    double gate_max = 0;
    printf("\n  routing table (slot: device -> reference)\n");
    for (int j = 0; j < TOPK; j++) {
        unsigned eid; float g_;
        memcpy(&eid, o_tab + (size_t)j * 8, 4);
        memcpy(&g_, o_tab + (size_t)j * 8 + 4, 4);
        const int same = (eid == R_sel[j]);
        order_ok &= same;
        int present = 0;
        for (int q = 0; q < TOPK; q++) present |= (eid == R_sel[q]);
        route_ok &= present;
        gate_max = fmax(gate_max, fabs((double)g_ - R_gate[j]) / (fabs(R_gate[j]) + 1e-30));
        if (j < 4 || !same)
            printf("    %2d: e=%-4u g=%.6f  ->  e=%-4u g=%.6f%s\n", j, eid, g_, R_sel[j],
                   R_gate[j], same ? "" : "   <-- DIFFERS");
    }
    printf("    ... (%d slots)   SET matches: %s   ORDER matches: %s   max gate rel err: %.3e\n",
           TOPK, route_ok ? "YES" : "NO", order_ok ? "YES" : "NO", gate_max);
    if (!route_ok)
        printf("    NOTE: a divergent SET means unselected experts have NULL table entries, so the\n"
               "          partials below are zero by construction. Read the routing row, not them.\n");

    double w_[24];
    double r_ha = relerr(o_ha, R_ha, thid, &w_[0]);
    double r_x = relerr(o_x, R_x, thid, &w_[1]);
    double r_st = relerr_f32(o_state, R_state, (size_t)H * D * D, &w_[2]);
    double r_at = relerr(o_attn, R_attn, thid, &w_[3]);
    double r_pf = relerr(o_prefix, R_prefix, thid, &w_[4]);
    double r_h2 = relerr(o_h2, R_h2, thid, &w_[5]);
    double r_h3 = relerr(o_h3, R_h3, thid, &w_[6]);
    double r_lg = relerr(o_logit, R_logit, (size_t)NEXP, &w_[7]);
    double r_xe = relerr(o_xe, R_xe, (size_t)HE, &w_[8]);
    double r_fu = relerr(o_fu, R_fu, (size_t)TOPK * IMOE, &w_[9]);
    double r_pt = relerr_f32(o_part, R_part, (size_t)TOPK * HE, &w_[10]);
    double r_yl = relerr(o_ylat, R_ylat, (size_t)HE, &w_[11]);
    double r_yn = relerr(o_yn, R_yn, (size_t)HE, &w_[12]);
    double r_yh = relerr(o_yh, R_yh, thid, &w_[13]);
    double r_sh = relerr(o_shd, R_shd, thid, &w_[14]);
    double r_out = relerr(o_out, R_out, thid, &w_[15]);

    printf("\n  stage                            rms rel     worst rel\n");
    printf("  A0  ATTNRES (attn side)  h_a    %10.3e  %10.3e\n", r_ha, w_[0]);
    printf("  A1  pre-norm             x      %10.3e  %10.3e\n", r_x, w_[1]);
    printf("  A2  KDA STATE V-FIRST  (f32)    %10.3e  %10.3e\n", r_st, w_[2]);
    printf("  A3  KDA out              attn   %10.3e  %10.3e\n", r_at, w_[3]);
    printf("  A4  prefix_sum accum     prefix %10.3e  %10.3e\n", r_pf, w_[4]);
    printf("  A5  ATTNRES (mlp side)   h2     %10.3e  %10.3e\n", r_h2, w_[5]);
    printf("  A6  post-attn norm       h3     %10.3e  %10.3e\n", r_h3, w_[6]);
    printf("  M0  router logits        [896]  %10.3e  %10.3e\n", r_lg, w_[7]);
    printf("  M1  LATENT down          xe     %10.3e  %10.3e\n", r_xe, w_[8]);
    printf("  M2  expert situ GLU      fu     %10.3e  %10.3e\n", r_fu, w_[9]);
    printf("  M3  expert down partials (f32)  %10.3e  %10.3e\n", r_pt, w_[10]);
    printf("  M4  MoeCombine (no resid) ylat  %10.3e  %10.3e\n", r_yl, w_[11]);
    printf("  M5  latent RMSNorm       yn     %10.3e  %10.3e\n", r_yn, w_[12]);
    printf("  M6  LATENT up            yh     %10.3e  %10.3e\n", r_yh, w_[13]);
    printf("  M7  shared expert (situ) shd    %10.3e  %10.3e\n", r_sh, w_[14]);
    printf("  M8  BLOCK out                   %10.3e  %10.3e\n", r_out, w_[15]);

    /* The controls, at the two points where AttnRes actually acts. The oracle's note applies: at a
     * NON-SNAPSHOT layer the block output is `prefix_in + attn + moe` either way, so the last row
     * above is NOT the falsifier here — rows A0 and A5 are. */
    double wc;
    double c_ha = relerr(o_ha, P_prefix_in, thid, &wc);
    double c_h2 = relerr(o_h2, R_prefix, thid, &wc);
    printf("\n  [control] device h_a vs a PLAIN residual (= prefix_in): %10.3e\n", c_ha);
    printf("  [control] device h2  vs a PLAIN residual (= prefix)   : %10.3e\n", c_h2);
    printf("  (both must be LARGE — at this layer the BLOCK OUTPUT cannot tell the two wirings\n"
           "   apart, so these two rows are what makes the gate falsifiable)\n");

    /* bf16 tolerances as the GLM B4 gate, `expert_sum` at the looser 6e-2 arm: the expert partials
     * are an mxfp4 w4a16 product summed over 16 slots, which is the same cancellation shape. */
    const double TOL = 1.5e-2, TOL_F32 = 2e-2, TOL_EXP = 6e-2;
    int ok = exec_ok && route_ok && order_ok && gate_max < TOL_F32 &&
             r_ha < TOL && r_x < TOL && r_st < TOL_F32 && r_at < TOL && r_pf < TOL &&
             r_h2 < TOL && r_h3 < TOL && r_lg < TOL && r_xe < TOL && r_fu < TOL_EXP &&
             r_pt < TOL_EXP && r_yl < TOL_EXP && r_yn < TOL_EXP && r_yh < TOL_EXP &&
             r_sh < TOL && r_out < TOL && c_ha > 0.1 && c_h2 > 0.1;

    printf("\n=> %s\n", ok ? "K3 MoE BLOCK OK — both AttnRes applications, KDA, Stable LatentMoE "
                             "(896 experts, top-16, MXFP4, latent 3584) and the situ shared expert "
                             "match the reference on real Kimi-K3 layer-1 weights"
                           : "*** K3 MoE BLOCK MISMATCH ***");
    return ok ? 0 : 1;
}
