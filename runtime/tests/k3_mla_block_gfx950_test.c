/* k3_mla_block_gfx950_test.c — REAL-WEIGHT gate for a COMPLETE Kimi-K3 **GATED MLA** BLOCK.
 *                                                                              [K3-MLA-GATE]
 *
 * Rung 3, and the last of K3's three block types: 24 of the 93 layers are this. Rung 1
 * (`k3_block_gfx950_test.c`, layer 0) gated KDA + one AttnRes + a dense `situ` FFN; rung 2
 * (`k3_moe_block_gfx950_test.c`, layer 1) gated both AttnRes applications + KDA + Stable LatentMoE.
 * This gates layer 3 — the same block SHAPE as rung 2 with the mixer swapped:
 *
 *     AttnRes -> Gated MLA -> AttnRes -> Stable LatentMoE
 *
 * and it is the first thing in this tree to carry a KV CACHE: rungs 1 and 2 were stateless.
 *
 * ============================================================================================
 * THE FOUR THINGS THIS GATE EXISTS TO PROVE
 * ============================================================================================
 *
 * 1. **MLA ABSORPTION, fed to a kernel for the first time.** `scripts/kimi_k3_prep.py` computed
 *    `Wqa = einsum('hpl,hpk->hlk', k_nope, q_nope)` and `Wuv = value^T` and verified them
 *    numerically — but nothing had ever consumed the output. plow's MLA decode NEVER materializes
 *    q_pass / k_pass / value: FLASH_MLA_DECODE scores the absorbed query against the 512-wide
 *    LATENT, and MLA_MERGE_FOLD folds W_uv on the way out. Row **M2** is scored against the
 *    MODEL'S OWN DEFINITION (k_pass and value materialized per cached row), not against the
 *    absorbed form, so a wrong absorption cannot agree with itself. The oracle prints the
 *    absorbed-vs-unabsorbed difference separately.
 *
 * 2. **The MLA OUTPUT GATE** (`mla_use_output_gate: true`) — gap #5, never touched, `PLOW_DOP_
 *    MLA_OUT_GATE = 106` added for it. `g = sigmoid(g_proj(hidden_states)); attn *= g` BEFORE
 *    o_proj, where `hidden_states` is the post-`input_layernorm` MLA INPUT and NOT the attention
 *    output. Rows **G0/G1**, with a gated-vs-ungated control (the oracle measures 5.2e-1) so a
 *    gate that silently evaluated to 1 could not pass.
 *
 * 3. **NoPE** (`mla_use_nope: true`) — gap #6, AND IT IS NOT A REMOVAL. The k-side
 *    `HeadNormRope` is the ONLY writer of the `kv.{l}.krot` cache row AND is the instruction
 *    `plowrt::exec::amd::kv_write_row_field` and `glm52_decode.c:419` both SCAN FOR to patch that
 *    row's position each step; deleting it drops the layer out of the KV-row-writer list with no
 *    count check. So both `HeadNormRope`s are EMITTED, with an identity cos=1 / sin=0 table, and
 *    rows **Q3/K1** are checked **BITWISE**, not to a tolerance — a table that were merely
 *    nearly-identity would pass 1.5e-2 while quietly rotating. The harness also proves the k-side
 *    op wrote the cache at row `qpos` and NOWHERE ELSE, which is the property the runtime's
 *    per-step `i[3]` patch depends on.
 *
 * 4. **`MLA_MERGE_FOLD` at V=128.** `exec_mla_merge_fold` picked `VT=256` whenever
 *    `bh*8 > nblk`; the fast body needs `v1-v0 == VT`, and `vtiles = ceil(128/256) = 1` gives
 *    128 != 256, so at K3's `v_head_dim` every workgroup silently took the scalar fallback the
 *    2026-07-28 rewrite replaced for being 7.7x slower. This gate runs at `bh = 96`, `nblk = 256`,
 *    i.e. `96*8 = 768 > 256` — EXACTLY the arm that was broken. Row M2 is what says the fixed
 *    `<512,128>` instantiation computes the same numbers.
 *
 * ============================================================================================
 * WHERE THIS GATE HAS TO LOOK
 * ============================================================================================
 * Layer 3 takes no block-residual snapshot (3 % 12 != 0), so `prefix = prefix_in + attn` and the
 * block output is `prefix_in + attn + moe` — EXACTLY what a plain-residual wiring produces. The
 * oracle measures the difference at **2.9e-3** at the block output against **8.0e-1** and
 * **7.7e-1** at the two AttnRes outputs. A block-output-only gate does not see AttnRes at 85 of
 * K3's 93 layers, so `h_a` and `h2` are diffed as their own rows and the controls are taken there.
 * The same discipline is applied to everything new here: every one of rows M2 / G1 / Q3 / K1 has a
 * control next to it saying what a wrong version would have looked like.
 *
 *   ./k3_mla_test [interp_decode.elf] [k3_mla_fixture.bin]
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
 * silently aliases tensor 0 into any operand the op reads and the emitter forgot. */
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

#define K3_MOE_ACT_SITU 2u
#define K3_MOE_ENC_MXFP4 2u
#define KV_MASK_NONE 0xFFFFFFFFu

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "interp_decode.elf";
    const char* fix = argc > 2 ? argv[2] : "k3_mla_fixture.bin";
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
    if (plow_hsa_get_kernel(h, 0, "plow_interp_dec_gfx950", &kern)) { printf("no kernel\n"); return 1; }

    int fd = open(fix, O_RDONLY);
    if (fd < 0) { perror(fix); return 1; }
    struct stat st; fstat(fd, &st);
    char* base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    int32_t* hdr = (int32_t*)base;
    if (hdr[0] != 0x4B334D41) { printf("bad magic %x (want K3MA)\n", hdr[0]); return 1; }
    const int T = hdr[1], NH = hdr[2], HID = hdr[3], QL = hdr[4], DK = hdr[5], DR = hdr[6],
              QN = hdr[7], VD = hdr[8], L = hdr[9], QPOS = hdr[10], IMOE = hdr[11], HE = hdr[12],
              NEXP = hdr[13], TOPK = hdr[14], NSEL = hdr[15];
    int32_t* h2i = (int32_t*)(base + 16 * 4);
    const int SHI = h2i[0], NB = h2i[1], RFLAGS = h2i[2], GF = h2i[3], NSPLIT = h2i[4];
    float* fh = (float*)(base + 24 * 4);
    const float EPS = fh[0], SCALE = fh[1], BETA = fh[2], LBETA = fh[3], RSCALE = fh[4];
    /* `qk=nope+rope`, in that order — the same order the config, the prep script and the gap docs
     * use. Printing it the other way round reads as a different model at a glance. */
    printf("K3 GATED MLA block: T=%d hidden=%d heads=%d q_lora=%d kv_lora=%d qk=%d+%d v_head=%d\n",
           T, HID, NH, QL, DK, QN, DR, VD);
    printf("              ctx=%d qpos=%d gf=%d nsplit=%d scale=%.6f | latent MoE %d experts "
           "top-%d I=%d he=%d shared=%d\n", L, QPOS, GF, NSPLIT, SCALE, NEXP, TOPK, IMOE, HE, SHI);
    if (T != 1) { printf("this gate is the DECODE path: T must be 1\n"); return 1; }
    if (NH % GF) { printf("n_head %d is not divisible by GF %d — d_flash_mla_decode truncates "
                          "n_grp and MLA_MERGE_FOLD then reads uninitialised Opart for the tail "
                          "heads, silently\n", NH, GF); return 1; }
    if (DR != 64) { printf("qk_rope %d: interp.hip's HEADNORM_ROPE arm has NO final else, so an "
                           "unmatched head_dim writes NOTHING\n", DR); return 1; }

    const size_t W1P = (size_t)IMOE * (HE / 2), W1S = (size_t)IMOE * (HE / 32);
    const size_t W2P = (size_t)HE * (IMOE / 2), W2S = (size_t)HE * (IMOE / 32);
    const size_t NHVD = (size_t)NH * VD, thid = (size_t)T * HID;

    size_t off = 24 * 4 + 6 * 4;
#define NEXT(cnt, elt) ({ void* _p = base + off; off += (size_t)(cnt) * (elt); _p; })
    bf16* P_prefix_in = NEXT(thid, 2);
    bf16* P_blkres = NEXT((size_t)T * NB * HID, 2);
    float* P_asw = NEXT((size_t)HID, 4);
    float* P_msw = NEXT((size_t)HID, 4);
    bf16* P_lnw = NEXT((size_t)HID, 2);
    bf16* P_postln = NEXT((size_t)HID, 2);
    bf16* P_qad = NEXT((size_t)QL * HID, 2);
    bf16* P_gqa = NEXT((size_t)QL, 2);
    bf16* P_wqa = NEXT((size_t)NH * DK * QL, 2);
    bf16* P_wqr = NEXT((size_t)NH * DR * QL, 2);
    bf16* P_ckvd = NEXT((size_t)DK * HID, 2);
    bf16* P_krotd = NEXT((size_t)DR * HID, 2);
    bf16* P_gkva = NEXT((size_t)DK, 2);
    bf16* P_wuv = NEXT((size_t)NH * DK * VD, 2);
    bf16* P_wg = NEXT(NHVD * HID, 2);
    bf16* P_wo = NEXT((size_t)HID * NHVD, 2);
    bf16* P_ckv_hist = NEXT((size_t)(L - 1) * DK, 2);
    bf16* P_krot_hist = NEXT((size_t)(L - 1) * DR, 2);
    float* P_cos = NEXT((size_t)L * (DR / 2), 4);
    float* P_sin = NEXT((size_t)L * (DR / 2), 4);
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
    bf16* R_ha = NEXT(thid, 2);
    bf16* R_x = NEXT(thid, 2);
    bf16* R_qlat = NEXT((size_t)QL, 2);
    bf16* R_qa = NEXT((size_t)NH * DK, 2);
    bf16* R_qrr = NEXT((size_t)NH * DR, 2);
    bf16* R_ckvcur = NEXT((size_t)DK, 2);
    bf16* R_krotcur = NEXT((size_t)DR, 2);
    bf16* R_oat = NEXT(NHVD, 2);
    bf16* R_gl = NEXT(NHVD, 2);
    bf16* R_oatg = NEXT(NHVD, 2);
    bf16* R_attn = NEXT(thid, 2);
    bf16* R_prefix = NEXT(thid, 2);
    bf16* R_h2 = NEXT(thid, 2);
    bf16* R_h3 = NEXT(thid, 2);
    bf16* R_logit = NEXT((size_t)NEXP, 2);
    uint32_t* R_sel = NEXT((size_t)TOPK, 4);
    float* R_gate = NEXT((size_t)TOPK, 4);
    bf16* R_xe = NEXT((size_t)HE, 2);
    bf16* R_fu = NEXT((size_t)TOPK * IMOE, 2);
    float* R_part = NEXT((size_t)TOPK * HE, 4);
    bf16* R_ylat = NEXT((size_t)HE, 2);
    bf16* R_yn = NEXT((size_t)HE, 2);
    bf16* R_yh = NEXT(thid, 2);
    bf16* R_shd = NEXT(thid, 2);
    bf16* R_out = NEXT(thid, 2);
    if (off != (size_t)st.st_size) {
        printf("FIXTURE SIZE MISMATCH: consumed %zu, file %zu\n", off, (size_t)st.st_size);
        return 1;
    }

#define DUP(ptr, bytes) ({ void* _d = plow_hsa_alloc(h, 0, (bytes)); \
                           plow_hsa_upload(h, 0, _d, (ptr), (bytes)); reg(_d); })
#define DNEW(bytes) ({ void* _d = plow_hsa_alloc(h, 0, (bytes)); reg(_d); })

    int t_prefix_in = DUP(P_prefix_in, thid * 2);
    int t_blkres = DUP(P_blkres, (size_t)T * NB * HID * 2);
    int t_asw = DUP(P_asw, (size_t)HID * 4), t_msw = DUP(P_msw, (size_t)HID * 4);
    int t_lnw = DUP(P_lnw, HID * 2), t_postln = DUP(P_postln, HID * 2);
    int t_qad = DUP(P_qad, (size_t)QL * HID * 2), t_gqa = DUP(P_gqa, (size_t)QL * 2);
    int t_wqa = DUP(P_wqa, (size_t)NH * DK * QL * 2);
    int t_wqr = DUP(P_wqr, (size_t)NH * DR * QL * 2);
    int t_ckvd = DUP(P_ckvd, (size_t)DK * HID * 2);
    int t_krotd = DUP(P_krotd, (size_t)DR * HID * 2);
    int t_gkva = DUP(P_gkva, (size_t)DK * 2);
    int t_wuv = DUP(P_wuv, (size_t)NH * DK * VD * 2);
    int t_wg = DUP(P_wg, NHVD * HID * 2);
    int t_wo = DUP(P_wo, (size_t)HID * NHVD * 2);
    int t_cos = DUP(P_cos, (size_t)L * (DR / 2) * 4);
    int t_sin = DUP(P_sin, (size_t)L * (DR / 2) * 4);

    /* THE KV CACHE. `[L][DK]` latent + `[L][DR]` rope, exactly the two `kv.{l}.*` rows the shipping
     * emitter declares. Rows 0..L-2 are the oracle's history; row QPOS is left as a POISON pattern
     * so that "the layer forgot to write its own KV row" cannot look like a small residual — the
     * flash decode would then attend over 0x7F7F (a huge bf16) and the output would explode rather
     * than drift. Rungs 1 and 2 had no cache at all, so this is the first time it matters. */
    bf16* kvinit = malloc((size_t)L * DK * 2);
    memcpy(kvinit, P_ckv_hist, (size_t)(L - 1) * DK * 2);
    for (int d = 0; d < DK; d++) kvinit[(size_t)QPOS * DK + d] = 0x7F7F;
    int t_ckv = DUP(kvinit, (size_t)L * DK * 2);
    bf16* krinit = malloc((size_t)L * DR * 2);
    memcpy(krinit, P_krot_hist, (size_t)(L - 1) * DR * 2);
    for (int d = 0; d < DR; d++) krinit[(size_t)QPOS * DR + d] = 0x7F7F;
    int t_krot = DUP(krinit, (size_t)L * DR * 2);
    /* The ckv WRITE targets a pre-offset handle at row QPOS: `d_rmsnorm` writes row 0 of its
     * output, and the shipping runtime instead patches `i[2]` per step (exec/amd.rs
     * `kv_write_row_field`). Both land in the same place; the harness takes the pointer form
     * because it has no per-step loop, and exercises the `i[3]` form on the krot side below —
     * so BOTH row-targeting mechanisms the runtime uses are covered. */
    int t_ckvrow = reg((char*)g_tens[t_ckv] + (size_t)QPOS * DK * 2);

    int32_t klen = L, pos = QPOS;
    int t_klen = DUP(&klen, 4), t_pos = DUP(&pos, 4);

    int t_wrouter = DUP(P_wrouter, (size_t)NEXP * HID * 2);
    int t_rbias = DUP(P_rbias, (size_t)NEXP * 4);
    int t_wdownl = DUP(P_wdownl, (size_t)HE * HID * 2);
    int t_latnorm = DUP(P_latnorm, (size_t)HE * 2);
    int t_wupl = DUP(P_wupl, (size_t)HID * HE * 2);
    int t_wshg = DUP(P_wshg, (size_t)SHI * HID * 2);
    int t_wshu = DUP(P_wshu, (size_t)SHI * HID * 2);
    int t_wshd = DUP(P_wshd, (size_t)HID * SHI * 2);

    unsigned long long* wtab = calloc((size_t)NEXP * 3, 8);
    unsigned long long* stab = calloc((size_t)NEXP * 3, 8);
    size_t exp_bytes = 0;
    for (int j = 0; j < NSEL; j++) {
        void* dw[6];
        const size_t sz[6] = { W1P, W1S, W1P, W1S, W2P, W2S };
        for (int q = 0; q < 6; q++) {
            dw[q] = plow_hsa_alloc(h, 0, sz[q]);
            plow_hsa_upload(h, 0, dw[q], P_exp[j][q], sz[q]);
            exp_bytes += sz[q];
        }
        const unsigned e = P_eid[j];
        if (e >= (unsigned)NEXP) { printf("bad expert id %u\n", e); return 1; }
        for (int q = 0; q < 3; q++) {
            wtab[(size_t)e * 3 + q] = (unsigned long long)(size_t)dw[q * 2];
            stab[(size_t)e * 3 + q] = (unsigned long long)(size_t)dw[q * 2 + 1];
        }
    }
    int t_ewt = DUP(wtab, (size_t)NEXP * 3 * 8);
    int t_est = DUP(stab, (size_t)NEXP * 3 * 8);
    printf("              uploaded %.1f MB of mxfp4 expert weights (%.1f GB if all %d were "
           "materialized)\n", exp_bytes / 1e6, exp_bytes / 1e9 * NEXP / NSEL, NEXP);

    int t_ha = DNEW(thid * 2), t_x = DNEW(thid * 2);
    int t_qlr = DNEW((size_t)QL * 2), t_qlat = DNEW((size_t)QL * 2);
    int t_qa = DNEW((size_t)NH * DK * 2);
    int t_qrr = DNEW((size_t)NH * DR * 2), t_qr = DNEW((size_t)NH * DR * 2);
    int t_ckvraw = DNEW((size_t)DK * 2), t_krr = DNEW((size_t)DR * 2);
    int t_opart = DNEW((size_t)NH * NSPLIT * DK * 4);
    int t_mlpart = DNEW((size_t)NH * NSPLIT * 2 * 4);
    int t_oat = DNEW(NHVD * 2), t_gl = DNEW(NHVD * 2), t_oatg = DNEW(NHVD * 2);
    int t_attn = DNEW(thid * 2);
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
        g_inst[_i].fj[0].f = 1.0f;                                                                \
        addwait(_i, dep_, ALL); _i; })
#define RMSN(o_, x_, g_, feat_, dep_, dblk_) ({                                                  \
        int _i = emitop(PLOW_DOP_RMSNORM, ALL);                                                  \
        g_inst[_i].t[0] = o_; g_inst[_i].t[1] = x_; g_inst[_i].t[2] = g_;                         \
        g_inst[_i].i[0] = T; g_inst[_i].i[1] = (feat_); g_inst[_i].fj[0].f = EPS;                \
        addwait(_i, dep_, dblk_); _i; })

    const uint16_t AR = (uint16_t)(T < (int)NCU ? T : (int)NCU);
    /* A0 — the ATTN-SIDE AttnRes, with `self_attention_res_*`. */
    int i_ar1 = emitop(PLOW_DOP_ATTN_RES, AR);
    g_inst[i_ar1].t[0] = t_ha; g_inst[i_ar1].t[1] = t_prefix_in; g_inst[i_ar1].t[2] = t_blkres;
    g_inst[i_ar1].t[3] = t_asw;
    g_inst[i_ar1].i[0] = T; g_inst[i_ar1].i[1] = HID; g_inst[i_ar1].i[2] = NB;
    g_inst[i_ar1].fj[0].f = EPS;

    /* A1 — input_layernorm. `t_x` is `hidden_states` as KimiMLAAttention.forward sees it, and it
     * feeds the q down-proj, the kv down-proj, the k_rope down-proj AND `g_proj`. */
    int i_ln = RMSN(t_x, t_ha, t_lnw, HID, i_ar1, AR);

    /* ===================== GATED MLA ===================== */
    int i_qad = GEMV(t_qlr, t_x, t_qad, QL, HID, i_ln);
    int i_rnq = RMSN(t_qlat, t_qlr, t_gqa, QL, i_qad, ALL);
    /* THE ABSORPTION. `q_absorb` is [NH*DK, QL] and its output is the query IN LATENT SPACE — the
     * kernel never sees q_pass, k_pass or qk_nope at all. */
    int i_qa = GEMV(t_qa, t_qlat, t_wqa, NH * DK, QL, i_rnq);
    int i_qrr = GEMV(t_qrr, t_qlat, t_wqr, NH * DR, QL, i_rnq);
    int i_ckvd = GEMV(t_ckvraw, t_x, t_ckvd, DK, HID, i_ln);
    int i_krr = GEMV(t_krr, t_x, t_krotd, DR, HID, i_ln);

    /* Q3 — the q-side HeadNormRope, with the IDENTITY table. gamma ABSENT and skip_norm=1, so with
     * cos=1/sin=0 the whole op is `f2bf(bf2f(x))` — a bit-exact copy. Emitted rather than skipped
     * because K3's shipping graph must keep the op (see the k-side note below), and because an op
     * that is emitted-and-neutralized is checkable while an op that is absent is not. */
    int i_qr = emitop(PLOW_DOP_HEADNORM_ROPE, ALL);
    g_inst[i_qr].t[0] = t_qr; g_inst[i_qr].t[1] = t_qrr; g_inst[i_qr].t[2] = PLOW_TENSOR_NONE;
    g_inst[i_qr].t[3] = t_cos; g_inst[i_qr].t[4] = t_sin; g_inst[i_qr].t[5] = t_pos;
    g_inst[i_qr].i[0] = T; g_inst[i_qr].i[1] = NH; g_inst[i_qr].i[2] = DR;
    g_inst[i_qr].i[3] = 0; g_inst[i_qr].i[4] = 1;         /* out_row0=0 (q is not cached), skip_norm */
    g_inst[i_qr].fj[0].f = EPS;
    g_inst[i_qr].fj[1].u = 0;                             /* out_stride: plain [ntok][nhead][hd] */
    g_inst[i_qr].fj[2].u = KV_MASK_NONE;
    addwait(i_qr, i_qrr, ALL);

    /* K0 — kv_a_layernorm, writing the LATENT cache row directly (pre-offset handle). */
    int i_rnkv = RMSN(t_ckvrow, t_ckvraw, t_gkva, DK, i_ckvd, ALL);

    /* K1 — the k-side HeadNormRope. THE ONE THAT MUST NOT BE DELETED. It is the only writer of the
     * `kv.{l}.krot` row and it is the instruction the runtime SCANS FOR to patch `i[3]` each step;
     * here `i[3] = QPOS` is that patch, applied once. With the identity table it is a bit-exact
     * copy of `t_krr` into row QPOS — the WRITE kept, the ROTATION removed. */
    int i_krd = emitop(PLOW_DOP_HEADNORM_ROPE, ALL);
    g_inst[i_krd].t[0] = t_krot; g_inst[i_krd].t[1] = t_krr; g_inst[i_krd].t[2] = PLOW_TENSOR_NONE;
    g_inst[i_krd].t[3] = t_cos; g_inst[i_krd].t[4] = t_sin; g_inst[i_krd].t[5] = t_pos;
    g_inst[i_krd].i[0] = T; g_inst[i_krd].i[1] = 1; g_inst[i_krd].i[2] = DR;
    g_inst[i_krd].i[3] = (uint32_t)QPOS; g_inst[i_krd].i[4] = 1;
    g_inst[i_krd].fj[0].f = EPS;
    g_inst[i_krd].fj[1].u = 0;
    g_inst[i_krd].fj[2].u = KV_MASK_NONE;
    addwait(i_krd, i_krr, ALL);

    /* M1 — FLASH_MLA_DECODE over the LATENT. One "kv head": every query head reads the same
     * 512-wide latent plus the shared 64-wide rope row. i[3]=window=0 (dense, full causal). */
    int i_fl = emitop(PLOW_DOP_FLASH_MLA_DECODE, ALL);
    g_inst[i_fl].t[0] = t_opart; g_inst[i_fl].t[1] = t_mlpart;
    g_inst[i_fl].t[2] = t_qa; g_inst[i_fl].t[3] = t_qr;
    g_inst[i_fl].t[4] = t_ckv; g_inst[i_fl].t[5] = t_krot; g_inst[i_fl].t[6] = t_klen;
    g_inst[i_fl].i[0] = 1; g_inst[i_fl].i[1] = NH; g_inst[i_fl].i[2] = L;
    g_inst[i_fl].i[3] = 0; g_inst[i_fl].i[4] = NSPLIT; g_inst[i_fl].i[5] = KV_MASK_NONE;
    g_inst[i_fl].i[7] = GF;
    g_inst[i_fl].fj[0].f = SCALE;
    addwait(i_fl, i_qa, ALL); addwait(i_fl, i_qr, ALL);
    addwait(i_fl, i_rnkv, ALL); addwait(i_fl, i_krd, ALL);

    /* M2 — MLA_MERGE_FOLD. bh = 1*96 = 96 and nblk = 256, so 96*8 = 768 > 256: this takes the
     * `else` arm — the arm that at V=128 silently fell to the 7.7x-slower scalar body until the
     * `<512,128>` instantiation was added. Fixing the dispatch does not change the ANSWER, which
     * is what this row is here to say. */
    int i_uv = emitop(PLOW_DOP_MLA_MERGE_FOLD, ALL);
    g_inst[i_uv].t[0] = t_oat; g_inst[i_uv].t[1] = t_opart; g_inst[i_uv].t[2] = t_mlpart;
    g_inst[i_uv].t[3] = t_wuv;
    g_inst[i_uv].i[0] = 1; g_inst[i_uv].i[1] = NH; g_inst[i_uv].i[2] = VD; g_inst[i_uv].i[4] = NSPLIT;
    addwait(i_uv, i_fl, ALL);

    /* G0/G1 — THE OUTPUT GATE. `g_proj` reads `t_x` (the input_layernorm output), NOT `t_oat`:
     * HF gates on `hidden_states`. The two have DIFFERENT widths here (7168 vs 12288), so that
     * particular mistake fails loudly — but gating on the post-attn norm instead would not, which
     * is why the dependency is on `i_ln` and is written down. */
    int i_gl = GEMV(t_gl, t_x, t_wg, (uint32_t)NHVD, HID, i_ln);
    int i_gt = emitop(PLOW_DOP_MLA_OUT_GATE, ALL);
    g_inst[i_gt].t[0] = t_oatg; g_inst[i_gt].t[1] = t_oat; g_inst[i_gt].t[2] = t_gl;
    g_inst[i_gt].i[0] = (uint32_t)NHVD;
    addwait(i_gt, i_uv, ALL); addwait(i_gt, i_gl, ALL);

    /* A2 — o_proj, on the GATED output. */
    int i_op = GEMV(t_attn, t_oatg, t_wo, HID, (uint32_t)NHVD, i_gt);

    /* A3 — prefix_sum accumulates (no snapshot at this layer). */
    int i_pfx = emitop(PLOW_DOP_RESIDUAL, ALL);
    g_inst[i_pfx].t[0] = t_prefix; g_inst[i_pfx].t[1] = t_prefix_in; g_inst[i_pfx].t[2] = t_attn;
    g_inst[i_pfx].i[0] = (uint32_t)thid; g_inst[i_pfx].fj[0].f = 1.0f;
    addwait(i_pfx, i_op, ALL);

    /* A4 — the MLP-SIDE AttnRes, with the OTHER fold (`mlp_res_*`). */
    int i_ar2 = emitop(PLOW_DOP_ATTN_RES, AR);
    g_inst[i_ar2].t[0] = t_h2; g_inst[i_ar2].t[1] = t_prefix; g_inst[i_ar2].t[2] = t_blkres;
    g_inst[i_ar2].t[3] = t_msw;
    g_inst[i_ar2].i[0] = T; g_inst[i_ar2].i[1] = HID; g_inst[i_ar2].i[2] = NB;
    g_inst[i_ar2].fj[0].f = EPS;
    addwait(i_ar2, i_pfx, ALL);

    int i_pn = RMSN(t_h3, t_h2, t_postln, HID, i_ar2, AR);

    /* ================= STABLE LATENTMOE (rung 2's graph, unchanged) ================= */
    int i_rl = GEMV(t_logit, t_h3, t_wrouter, NEXP, HID, i_pn);
    int i_rt = emitop(PLOW_DOP_MOE_ROUTER_TOPK, 1);
    g_inst[i_rt].t[0] = t_tab; g_inst[i_rt].t[1] = t_logit; g_inst[i_rt].t[3] = t_rbias;
    g_inst[i_rt].i[1] = NEXP; g_inst[i_rt].i[2] = TOPK; g_inst[i_rt].i[3] = RFLAGS;
    g_inst[i_rt].i[6] = 1; g_inst[i_rt].i[7] = 1;
    g_inst[i_rt].fj[0].f = RSCALE;
    addwait(i_rt, i_rl, ALL);

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

    int i_cmb = emitop(PLOW_DOP_MOE_COMBINE, ALL);
    g_inst[i_cmb].t[0] = t_ylat; g_inst[i_cmb].t[3] = t_part;   /* t1 residual, t2 shared: ABSENT */
    g_inst[i_cmb].i[0] = HE; g_inst[i_cmb].i[1] = TOPK;
    for (int j = 0; j < TOPK; j++) addwait(i_cmb, i_ed[j], ALL);

    int i_ln2 = RMSN(t_yn, t_ylat, t_latnorm, HE, i_cmb, ALL);
    int i_up = GEMV(t_yh, t_yn, t_wupl, HID, HE, i_ln2);

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

    bf16* o_ha = malloc(thid * 2); bf16* o_x = malloc(thid * 2);
    bf16* o_qlat = malloc((size_t)QL * 2);
    bf16* o_qa = malloc((size_t)NH * DK * 2);
    bf16* o_qrr = malloc((size_t)NH * DR * 2); bf16* o_qr = malloc((size_t)NH * DR * 2);
    bf16* o_krr = malloc((size_t)DR * 2);
    bf16* o_ckv = malloc((size_t)L * DK * 2); bf16* o_krot = malloc((size_t)L * DR * 2);
    bf16* o_oat = malloc(NHVD * 2); bf16* o_gl = malloc(NHVD * 2); bf16* o_oatg = malloc(NHVD * 2);
    bf16* o_attn = malloc(thid * 2); bf16* o_prefix = malloc(thid * 2);
    bf16* o_h2 = malloc(thid * 2); bf16* o_h3 = malloc(thid * 2);
    bf16* o_logit = malloc((size_t)NEXP * 2);
    uint8_t* o_tab = malloc((size_t)TOPK * 8);
    bf16* o_xe = malloc((size_t)HE * 2);
    bf16* o_fu = malloc((size_t)TOPK * IMOE * 2);
    float* o_part = malloc((size_t)TOPK * HE * 4);
    bf16* o_ylat = malloc((size_t)HE * 2); bf16* o_yn = malloc((size_t)HE * 2);
    bf16* o_yh = malloc(thid * 2); bf16* o_shd = malloc(thid * 2); bf16* o_out = malloc(thid * 2);
    plow_hsa_download(h, 0, o_ha, g_tens[t_ha], thid * 2);
    plow_hsa_download(h, 0, o_x, g_tens[t_x], thid * 2);
    plow_hsa_download(h, 0, o_qlat, g_tens[t_qlat], (size_t)QL * 2);
    plow_hsa_download(h, 0, o_qa, g_tens[t_qa], (size_t)NH * DK * 2);
    plow_hsa_download(h, 0, o_qrr, g_tens[t_qrr], (size_t)NH * DR * 2);
    plow_hsa_download(h, 0, o_qr, g_tens[t_qr], (size_t)NH * DR * 2);
    plow_hsa_download(h, 0, o_krr, g_tens[t_krr], (size_t)DR * 2);
    plow_hsa_download(h, 0, o_ckv, g_tens[t_ckv], (size_t)L * DK * 2);
    plow_hsa_download(h, 0, o_krot, g_tens[t_krot], (size_t)L * DR * 2);
    plow_hsa_download(h, 0, o_oat, g_tens[t_oat], NHVD * 2);
    plow_hsa_download(h, 0, o_gl, g_tens[t_gl], NHVD * 2);
    plow_hsa_download(h, 0, o_oatg, g_tens[t_oatg], NHVD * 2);
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

    /* ---- NoPE, BITWISE. Read BEFORE the residual table, because "the rotation is the identity" is
     * a claim about BYTES and a 1.5e-2 residual cannot distinguish an identity table from a table
     * whose angles happen to be small. Two properties, both required:
     *   (a) the op is a bit-exact copy on both the q side and the k side, and
     *   (b) the k-side write landed at row QPOS and NOWHERE ELSE — the property the runtime's
     *       per-step `i[3]` patch depends on, and the reason the op must be KEPT rather than
     *       deleted (it is also the instruction the kv-row-writer scan looks for). ---- */
    int qbits = memcmp(o_qr, o_qrr, (size_t)NH * DR * 2) == 0;
    int kbits = memcmp(o_krot + (size_t)QPOS * DR, o_krr, (size_t)DR * 2) == 0;
    int hist_intact = memcmp(o_krot, P_krot_hist, (size_t)(L - 1) * DR * 2) == 0;
    /* Would a non-identity table have been visible here? Only if the input is not already a RoPE
     * fixed point. Every-element-equal is the fixed point; assert we are not sitting on it. */
    int q_nonzero = 0;
    for (size_t i = 0; i < (size_t)NH * DR; i++) if (o_qrr[i] != 0) q_nonzero++;
    printf("\n  NoPE (identity cos=1/sin=0 table), checked BITWISE not to a tolerance\n");
    printf("    q_rope  HeadNormRope is a bit-exact copy : %s   (%d of %zu inputs nonzero)\n",
           qbits ? "YES" : "*** NO ***", q_nonzero, (size_t)NH * DR);
    printf("    k_rope  HeadNormRope is a bit-exact copy : %s\n", kbits ? "YES" : "*** NO ***");
    printf("    it wrote kv.krot row %d and left rows 0..%d untouched : %s\n", QPOS, QPOS - 1,
           hist_intact ? "YES" : "*** NO ***");
    printf("    (the op is KEPT and NEUTRALIZED, not deleted: it is the ONLY writer of the krot\n"
           "     cache row AND the instruction plowrt's kv-row-writer scan matches on. Deleting it\n"
           "     drops the layer from that list with no count check.)\n");

    /* ---- The ckv cache row, and the poison. If the layer had failed to write its own KV row the
     * flash decode would have attended over 0x7F7F — a huge bf16 — so this cannot fail quietly. */
    double wc0;
    double r_ckv = relerr(o_ckv + (size_t)QPOS * DK, R_ckvcur, (size_t)DK, &wc0);
    double r_krot = relerr(o_krot + (size_t)QPOS * DR, R_krotcur, (size_t)DR, &wc0);

    /* ---- The routing table, expert id by expert id, BEFORE the residual table, because a routing
     * divergence makes every downstream row meaningless and must not be read as an arithmetic
     * error.
     *
     * SET is a hard requirement. **ORDER IS NOT**, and that is a deliberate weakening of rung 2's
     * contract with a reason and a check attached.
     *
     * The MoE output is order-invariant: `d_moe_expert_down_fp8_blk` writes `part_slot[h] = gate*y`
     * per slot and `d_moe_combine` SUMS the slots, so permuting the top-k table cannot change
     * `ylat` — and indeed rows M5..M9 below pass at 2e-3 through a permutation that makes rows
     * M3/M4 (which ARE per-slot) read 5e-1. Requiring exact order therefore requires the device's
     * rank pass to reproduce `torch.topk`'s tie-break, which is not a property of the model.
     *
     * But "it's just a tie" is exactly the kind of excuse that hides a real ranking bug, so it is
     * CHECKED rather than asserted. The selection key is `sigmoid(logit) + bias`, and the device
     * and the reference are given DIFFERENT bf16 logits (row M0 measures the difference). So:
     *   dmax = max_e |sigmoid(dev_logit_e) - sigmoid(ref_logit_e)|   over the selected experts
     * bounds how far the logit discrepancy alone can move any key. A pair whose REFERENCE keys are
     * closer together than 2*dmax cannot be ordered by either party — the ranking is genuinely
     * unresolvable at this precision. A pair further apart than that is a BUG, and fails.
     * Everything here is computed from the fixture the harness already has; nothing is tuned. ---- */
    double* refkey = malloc((size_t)NEXP * sizeof(double));
    double* devkey = malloc((size_t)NEXP * sizeof(double));
    for (int e = 0; e < NEXP; e++) {
        refkey[e] = 1.0 / (1.0 + exp(-(double)b2f(R_logit[e]))) + P_rbias[e];
        devkey[e] = 1.0 / (1.0 + exp(-(double)b2f(o_logit[e]))) + P_rbias[e];
    }
    unsigned dev_e[64]; float dev_g[64];
    int route_ok = 1, order_ok = 1;
    double gate_max = 0, dmax = 0;
    int perm[64];                       /* perm[j] = the REFERENCE slot holding device slot j's id */
    printf("\n  routing table (slot: device -> reference)\n");
    for (int j = 0; j < TOPK; j++) {
        memcpy(&dev_e[j], o_tab + (size_t)j * 8, 4);
        memcpy(&dev_g[j], o_tab + (size_t)j * 8 + 4, 4);
        perm[j] = -1;
        for (int q = 0; q < TOPK; q++) if (dev_e[j] == R_sel[q]) perm[j] = q;
        if (perm[j] < 0) { route_ok = 0; perm[j] = j; }
        else dmax = fmax(dmax, fabs(devkey[dev_e[j]] - refkey[dev_e[j]]));
        const int same = (dev_e[j] == R_sel[j]);
        order_ok &= same;
        /* Compare the gate to the reference entry for the SAME EXPERT, not the same slot: under a
         * permutation a slot-wise comparison reports the permutation, not the arithmetic. */
        gate_max = fmax(gate_max,
                        fabs((double)dev_g[j] - R_gate[perm[j]]) / (fabs(R_gate[perm[j]]) + 1e-30));
        if (j < 4 || !same)
            printf("    %2d: e=%-4u g=%.6f  ->  e=%-4u g=%.6f%s\n", j, dev_e[j], dev_g[j], R_sel[j],
                   R_gate[j], same ? "" : "   <-- REORDERED");
    }
    /* Every out-of-order pair must be an unresolvable tie. */
    int tie_ok = 1;
    double worst_pair_gap = 0, tie_bound = 2.0 * dmax;
    for (int j = 0; j < TOPK && route_ok; j++) {
        if (dev_e[j] == R_sel[j]) continue;
        const double gap = fabs(refkey[dev_e[j]] - refkey[R_sel[j]]);
        worst_pair_gap = fmax(worst_pair_gap, gap);
        if (gap > tie_bound) {
            tie_ok = 0;
            printf("    slot %2d: e=%u and e=%u are separated by %.3e in selection key, which is "
                   "MORE than the %.3e the logit discrepancy can explain — this is a RANKING BUG, "
                   "not a tie\n", j, dev_e[j], R_sel[j], gap, tie_bound);
        }
    }
    printf("    ... (%d slots)   SET matches: %s   max gate rel err (per EXPERT): %.3e\n",
           TOPK, route_ok ? "YES" : "*** NO ***", gate_max);
    printf("    ORDER matches: %s", order_ok ? "YES" : "NO");
    if (!order_ok)
        printf("  — reordered pairs differ by at most %.3e in selection key, against a %.3e bound "
               "from the bf16 logit discrepancy (max |dsigmoid| = %.3e): %s",
               worst_pair_gap, tie_bound, dmax,
               tie_ok ? "an UNRESOLVABLE TIE, and the MoE combine is order-invariant"
                      : "*** RESOLVABLE, so this is a ranking bug ***");
    printf("\n");
    if (!route_ok)
        printf("    NOTE: a divergent SET means unselected experts have NULL table entries, so the\n"
               "          per-slot partials below are zero by construction. Read this row, not them.\n");

    double w_[32];
    double r_ha = relerr(o_ha, R_ha, thid, &w_[0]);
    double r_x = relerr(o_x, R_x, thid, &w_[1]);
    double r_qlat = relerr(o_qlat, R_qlat, (size_t)QL, &w_[2]);
    double r_qa = relerr(o_qa, R_qa, (size_t)NH * DK, &w_[3]);
    double r_qr = relerr(o_qr, R_qrr, (size_t)NH * DR, &w_[4]);
    double r_oat = relerr(o_oat, R_oat, NHVD, &w_[5]);
    double r_gl = relerr(o_gl, R_gl, NHVD, &w_[6]);
    double r_og = relerr(o_oatg, R_oatg, NHVD, &w_[7]);
    double r_at = relerr(o_attn, R_attn, thid, &w_[8]);
    double r_pf = relerr(o_prefix, R_prefix, thid, &w_[9]);
    double r_h2 = relerr(o_h2, R_h2, thid, &w_[10]);
    double r_h3 = relerr(o_h3, R_h3, thid, &w_[11]);
    double r_lg = relerr(o_logit, R_logit, (size_t)NEXP, &w_[12]);
    double r_xe = relerr(o_xe, R_xe, (size_t)HE, &w_[13]);
    /* M3/M4 are the only PER-SLOT rows in the table, so they are the only ones a routing
     * PERMUTATION can break — and it breaks them loudly (measured 5.1e-1 / 4.7e-1 on a
     * two-slot swap) while every downstream row stays at 2e-3, because the combine is a sum.
     * Diff device slot j against the REFERENCE SLOT HOLDING THE SAME EXPERT. This is not a
     * loosening: `perm` is only well-defined when the SET matches, which is a hard gate condition
     * above, and at `perm == identity` these are byte-for-byte the same comparisons as before. */
    bf16* R_fu_p = malloc((size_t)TOPK * IMOE * 2);
    float* R_part_p = malloc((size_t)TOPK * HE * 4);
    for (int j = 0; j < TOPK; j++) {
        memcpy(R_fu_p + (size_t)j * IMOE, R_fu + (size_t)perm[j] * IMOE, (size_t)IMOE * 2);
        memcpy(R_part_p + (size_t)j * HE, R_part + (size_t)perm[j] * HE, (size_t)HE * 4);
    }
    double r_fu = relerr(o_fu, R_fu_p, (size_t)TOPK * IMOE, &w_[14]);
    double r_pt = relerr_f32(o_part, R_part_p, (size_t)TOPK * HE, &w_[15]);
    double r_yl = relerr(o_ylat, R_ylat, (size_t)HE, &w_[16]);
    double r_yn = relerr(o_yn, R_yn, (size_t)HE, &w_[17]);
    double r_yh = relerr(o_yh, R_yh, thid, &w_[18]);
    double r_sh = relerr(o_shd, R_shd, thid, &w_[19]);
    double r_out = relerr(o_out, R_out, thid, &w_[20]);
    double wck, wkr;
    relerr(o_ckv + (size_t)QPOS * DK, R_ckvcur, (size_t)DK, &wck);
    relerr(o_krot + (size_t)QPOS * DR, R_krotcur, (size_t)DR, &wkr);

    printf("\n  stage                             rms rel     worst rel\n");
    printf("  A0  ATTNRES (attn side)   h_a    %10.3e  %10.3e\n", r_ha, w_[0]);
    printf("  A1  input_layernorm       x      %10.3e  %10.3e\n", r_x, w_[1]);
    printf("  Q1  q_a down + q_a_norm   qlat   %10.3e  %10.3e\n", r_qlat, w_[2]);
    printf("  Q2  ABSORBED q_nope       qa     %10.3e  %10.3e\n", r_qa, w_[3]);
    printf("  Q3  q_rope (NoPE copy)    qr     %10.3e  %10.3e\n", r_qr, w_[4]);
    printf("  K0  kv_a_norm -> kv.ckv[%2d]      %10.3e  %10.3e\n", QPOS, r_ckv, wck);
    printf("  K1  k_rope    -> kv.krot[%2d]     %10.3e  %10.3e\n", QPOS, r_krot, wkr);
    printf("  M2  FLASH_MLA + MERGE_FOLD oat   %10.3e  %10.3e   <- vs the model's OWN k_pass/value\n",
           r_oat, w_[5]);
    printf("  G0  g_proj logits         g      %10.3e  %10.3e\n", r_gl, w_[6]);
    printf("  G1  MLA OUTPUT GATE       oatg   %10.3e  %10.3e\n", r_og, w_[7]);
    printf("  A2  o_proj                attn   %10.3e  %10.3e\n", r_at, w_[8]);
    printf("  A3  prefix_sum accum      prefix %10.3e  %10.3e\n", r_pf, w_[9]);
    printf("  A4  ATTNRES (mlp side)    h2     %10.3e  %10.3e\n", r_h2, w_[10]);
    printf("  A5  post-attn norm        h3     %10.3e  %10.3e\n", r_h3, w_[11]);
    printf("  M0  router logits         [896]  %10.3e  %10.3e\n", r_lg, w_[12]);
    printf("  M1  LATENT down           xe     %10.3e  %10.3e\n", r_xe, w_[13]);
    printf("  M3  expert situ GLU       fu     %10.3e  %10.3e   %s\n", r_fu, w_[14],
           order_ok ? "" : "<- slots aligned by EXPERT ID");
    printf("  M4  expert down partials (f32)   %10.3e  %10.3e   %s\n", r_pt, w_[15],
           order_ok ? "" : "<- slots aligned by EXPERT ID");
    printf("  M5  MoeCombine (no resid) ylat   %10.3e  %10.3e\n", r_yl, w_[16]);
    printf("  M6  latent RMSNorm        yn     %10.3e  %10.3e\n", r_yn, w_[17]);
    printf("  M7  LATENT up             yh     %10.3e  %10.3e\n", r_yh, w_[18]);
    printf("  M8  shared expert (situ)  shd    %10.3e  %10.3e\n", r_sh, w_[19]);
    printf("  M9  BLOCK out                    %10.3e  %10.3e\n", r_out, w_[20]);

    /* ---- CONTROLS. Every new thing above gets one, because every row above can be green while the
     * thing it names is wrong. ---- */
    double wc;
    double c_ha = relerr(o_ha, P_prefix_in, thid, &wc);
    double c_h2 = relerr(o_h2, R_prefix, thid, &wc);
    double c_gate = relerr(o_oatg, o_oat, NHVD, &wc);
    printf("\n  [control] device h_a vs a PLAIN residual (= prefix_in) : %10.3e\n", c_ha);
    printf("  [control] device h2  vs a PLAIN residual (= prefix)    : %10.3e\n", c_h2);
    printf("            (both must be LARGE — at a non-snapshot layer the BLOCK OUTPUT cannot tell\n"
           "             the two wirings apart, so these are what makes rows A0/A4 falsifiable)\n");
    printf("  [control] device GATED vs UNGATED attention output     : %10.3e\n", c_gate);
    printf("            (must be LARGE — a gate that evaluated to 1 would pass row G1 silently)\n");

    const double TOL = 1.5e-2, TOL_F32 = 2e-2, TOL_EXP = 6e-2;
    /* `order_ok` is NOT a gate condition; `tie_ok` is. See the routing-table note above: the MoE
     * combine sums the slots, so a permutation of the top-k table cannot change the model's
     * output — but a permutation the logit discrepancy CANNOT explain is a ranking bug and fails. */
    int ok = exec_ok && route_ok && tie_ok && gate_max < TOL_F32 &&
             qbits && kbits && hist_intact && q_nonzero > 0 &&
             r_ha < TOL && r_x < TOL && r_qlat < TOL && r_qa < TOL && r_qr < TOL &&
             r_ckv < TOL && r_krot < TOL && r_oat < TOL && r_gl < TOL && r_og < TOL &&
             r_at < TOL && r_pf < TOL && r_h2 < TOL && r_h3 < TOL && r_lg < TOL &&
             r_xe < TOL && r_fu < TOL_EXP && r_pt < TOL_EXP && r_yl < TOL_EXP &&
             r_yn < TOL_EXP && r_yh < TOL_EXP && r_sh < TOL && r_out < TOL &&
             c_ha > 0.1 && c_h2 > 0.1 && c_gate > 0.1;

    printf("\n=> %s\n", ok ? "K3 GATED MLA BLOCK OK — absorbed MLA over a real KV cache, NoPE as a "
                             "bit-exact krot write, the sigmoid output gate, both AttnRes "
                             "applications and Stable LatentMoE match the reference on real "
                             "Kimi-K3 layer-3 weights"
                           : "*** K3 GATED MLA BLOCK MISMATCH ***");
    return ok ? 0 : 1;
}
