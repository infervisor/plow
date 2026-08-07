/* interp_gfx950_test.c — proves the persistent on-device interpreter.
 *
 * Builds a 4-op program whose ops are chained purely by counters:
 *
 *     embed -> rmsnorm -> residual(norm, embed) -> softcap
 *
 * ONE kernel launch. 256 workgroups, one per CU, each walking its own stream.
 * Nothing is launched per op. If the counter protocol (agent-scope fence +
 * threshold gate) were wrong, a consumer would read a partially-written producer
 * buffer and the chained result would drift — so an exact match here is a real
 * test of the gating, not just of the arithmetic.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned short bf16;

static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

/* Tensor handles, in the order we fill the device pointer table. */
enum { T_OUT_EMB = 0, T_TABLE, T_IDS, T_OUT_NRM, T_GAMMA, T_OUT_RES, T_OUT_CAP, T_N };

int main(void) {
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 1; }

    char nm[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, nm, &cus, &lds);
    printf("dev0: %s  CUs=%u  LDS=%u B\n", nm, cus, lds);

    FILE* f = fopen("interp.elf", "rb");
    if (!f) { perror("interp.elf"); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(co_n);
    if (fread(co, 1, co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, co_n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }

    /* The interpreter's symbol carries the ISA name, and the runtime resolves it from the LIVE
     * agent name -- crates/plowrt/src/exec/amd.rs formats "{base}_{arch}" out of
     * HSA_AGENT_INFO_NAME. Do the same here instead of hardcoding one arch, so this test runs
     * against whatever part it is pointed at. */
    char sym[96];
    snprintf(sym, sizeof(sym), "plow_interp_%s", nm);
    plow_hsa_kernel k_interp;
    if (plow_hsa_get_kernel(h, 0, sym, &k_interp) != 0) {
        fprintf(stderr, "symbol %s: %s\n", sym, plow_hsa_last_error()); return 1;
    }
    printf("interpreter resolved (kernarg=%u B, LDS=%u B)\n\n",
           k_interp.kernarg_size, k_interp.group_segment_size);

    /* --- problem ---------------------------------------------------------- */
    const unsigned NCU = cus;          /* one workgroup per CU — co-residency */
    const unsigned VOCAB = 8192, HID = 5376, NTOK = 64;
    const size_t ACT = (size_t)NTOK * HID;
    const float SCALE = 73.5f, CAP = 30.0f, EPS = 1e-6f;

    srand(11);
    bf16* h_table = plow_hsa_alloc_host(h, (size_t)VOCAB * HID * 2);
    bf16* h_gamma = plow_hsa_alloc_host(h, HID * 2);
    int*  h_ids   = plow_hsa_alloc_host(h, NTOK * 4);
    bf16* h_out   = plow_hsa_alloc_host(h, ACT * 2);
    for (size_t i = 0; i < (size_t)VOCAB * HID; i++) h_table[i] = f2bf(frand());
    for (unsigned i = 0; i < HID; i++) h_gamma[i] = f2bf(1.0f + 0.1f * frand());
    for (unsigned t = 0; t < NTOK; t++) h_ids[t] = rand() % VOCAB;

    /* Device tensors. */
    void* d_table = plow_hsa_alloc(h, 0, (size_t)VOCAB * HID * 2);
    void* d_gamma = plow_hsa_alloc(h, 0, HID * 2);
    void* d_ids   = plow_hsa_alloc(h, 0, NTOK * 4);
    void* d_emb   = plow_hsa_alloc(h, 0, ACT * 2);
    void* d_nrm   = plow_hsa_alloc(h, 0, ACT * 2);
    void* d_res   = plow_hsa_alloc(h, 0, ACT * 2);
    void* d_cap   = plow_hsa_alloc(h, 0, ACT * 2);
    plow_hsa_copy_h2d(h, 0, d_table, h_table, (size_t)VOCAB * HID * 2);
    plow_hsa_copy_h2d(h, 0, d_gamma, h_gamma, HID * 2);
    plow_hsa_copy_h2d(h, 0, d_ids, h_ids, NTOK * 4);

    void* h_tensors[T_N];
    h_tensors[T_OUT_EMB] = d_emb;
    h_tensors[T_TABLE]   = d_table;
    h_tensors[T_IDS]     = d_ids;
    h_tensors[T_OUT_NRM] = d_nrm;
    h_tensors[T_GAMMA]   = d_gamma;
    h_tensors[T_OUT_RES] = d_res;
    h_tensors[T_OUT_CAP] = d_cap;
    void* d_tensors = plow_hsa_alloc(h, 0, sizeof(h_tensors));
    plow_hsa_upload(h, 0, d_tensors, h_tensors, sizeof(h_tensors));

    /* --- the program ------------------------------------------------------ *
     * 4 ops, each sliced across all NCU workgroups. Counter i is signalled once
     * per workgroup by op i, so the consumer's threshold is NCU.              */
    PlowDevInst insts[4];
    memset(insts, 0, sizeof(insts));
    /* ZERO IS A VALID TENSOR HANDLE. `memset` leaves every unused t[] slot at 0, which is
     * T_OUT_EMB here -- so PLOW_DOP_RESIDUAL's OPTIONAL third operand (`pre`, t[3]) resolved to
     * the embedding and the op computed (pre + a + b) * scale, adding the embedding twice.
     * The absent-operand sentinel is PLOW_TENSOR_NONE (0xFFFF), not 0. Symptom was a chained
     * result ~2x high with the counter protocol passing, i.e. it looked like a coherence bug
     * and was an ABI one. */
    for (unsigned i = 0; i < 4; i++)
        for (unsigned j = 0; j < 8; j++) insts[i].t[j] = PLOW_TENSOR_NONE;
    PlowWait waits[3];
    uint32_t succs[3] = {0, 1, 2};

    /* op0: embed+scale -> d_emb.  no wait.  signals c0 */
    insts[0].op = PLOW_DOP_EMBED;   insts[0].blocks = NCU;
    insts[0].t[0] = T_OUT_EMB; insts[0].t[1] = T_TABLE; insts[0].t[2] = T_IDS;
    insts[0].i[0] = NTOK; insts[0].i[1] = HID; insts[0].fj[0].f = SCALE;

    /* op1: rmsnorm(d_emb) -> d_nrm.  waits c0 >= NCU.  signals c1 */
    waits[0].id = 0; waits[0].threshold = NCU;
    insts[1].op = PLOW_DOP_RMSNORM; insts[1].blocks = NCU;
    insts[1].t[0] = T_OUT_NRM; insts[1].t[1] = T_OUT_EMB; insts[1].t[2] = T_GAMMA;
    insts[1].i[0] = NTOK; insts[1].i[1] = HID; insts[1].fj[0].f = EPS;

    /* op2: residual(d_nrm, d_emb) * 1.37 -> d_res.  waits c1.  signals c2 */
    waits[1].id = 1; waits[1].threshold = NCU;
    const float LS = 1.37f;
    insts[2].op = PLOW_DOP_RESIDUAL; insts[2].blocks = NCU;
    insts[2].t[0] = T_OUT_RES; insts[2].t[1] = T_OUT_NRM; insts[2].t[2] = T_OUT_EMB;
    insts[2].i[0] = (unsigned)ACT; insts[2].fj[0].f = LS;

    /* op3: softcap(d_res) -> d_cap.  waits c2. */
    waits[2].id = 2; waits[2].threshold = NCU;
    insts[3].op = PLOW_DOP_SOFTCAP; insts[3].blocks = NCU;
    insts[3].t[0] = T_OUT_CAP; insts[3].t[1] = T_OUT_RES;
    insts[3].i[0] = (unsigned)ACT; insts[3].fj[0].f = CAP;

    /* Streams: every CU runs all 4 ops, taking slice = cu.
     * calloc, NOT malloc: a stream entry carries `flags`, and a garbage PLOW_SE_FINE bit
     * sends the interpreter to a wait list that does not exist. Zero == the coarse path. */
    PlowStreamEnt* h_stream = calloc(NCU * 4, sizeof(PlowStreamEnt));
    uint32_t* h_sofs = malloc(4 * NCU);
    uint32_t* h_slen = malloc(4 * NCU);
    for (unsigned cu = 0; cu < NCU; cu++) {
        h_sofs[cu] = cu * 4;
        h_slen[cu] = 4;
        for (unsigned k = 0; k < 4; k++) {
            h_stream[cu * 4 + k].inst = k;
            h_stream[cu * 4 + k].slice = cu;
            /* Gates live on the stream entries (64-byte PlowDevInst carries none):
             * op k waits on counter k-1 (k>0) and signals counter k (k<3). */
            if (k > 0) { h_stream[cu * 4 + k].wait_len = 1; h_stream[cu * 4 + k].wait_ofs = k - 1; }
            if (k < 3) { h_stream[cu * 4 + k].succ_len = 1; h_stream[cu * 4 + k].succ_ofs = k; }
        }
    }
    /* strided one cache line apart — see PLOW_CTR_STRIDE */
    uint32_t h_ctr[3 * PLOW_CTR_STRIDE] = {0};

    void* d_insts  = plow_hsa_alloc(h, 0, sizeof(insts));
    void* d_stream = plow_hsa_alloc(h, 0, sizeof(PlowStreamEnt) * NCU * 4);
    void* d_sofs   = plow_hsa_alloc(h, 0, 4 * NCU);
    void* d_slen   = plow_hsa_alloc(h, 0, 4 * NCU);
    void* d_waits  = plow_hsa_alloc(h, 0, sizeof(waits));
    void* d_succs  = plow_hsa_alloc(h, 0, sizeof(succs));
    void* d_ctr    = plow_hsa_alloc(h, 0, sizeof(h_ctr));
    plow_hsa_upload(h, 0, d_insts, insts, sizeof(insts));
    plow_hsa_upload(h, 0, d_stream, h_stream, sizeof(PlowStreamEnt) * NCU * 4);
    plow_hsa_upload(h, 0, d_sofs, h_sofs, 4 * NCU);
    plow_hsa_upload(h, 0, d_slen, h_slen, 4 * NCU);
    plow_hsa_upload(h, 0, d_waits, waits, sizeof(waits));
    plow_hsa_upload(h, 0, d_succs, succs, sizeof(succs));
    plow_hsa_upload(h, 0, d_ctr, h_ctr, sizeof(h_ctr));

    PlowProgram prog;
    memset(&prog, 0, sizeof(prog)); /* trace == NULL disables tracing */
    prog.insts      = (const PlowDevInst*)d_insts;
    prog.stream     = (const PlowStreamEnt*)d_stream;
    prog.stream_ofs = (const uint32_t*)d_sofs;
    prog.stream_len = (const uint32_t*)d_slen;
    prog.waits      = (const PlowWait*)d_waits;
    prog.succs      = (const uint32_t*)d_succs;
    prog.counters   = (uint32_t*)d_ctr;
    prog.tensors    = (void* const*)d_tensors;

    /* --- ONE launch ------------------------------------------------------- */
    printf("launching persistent interpreter: 1 kernel, %u workgroups, 4-op program\n", NCU);
    if (plow_hsa_launch(h, 0, &k_interp, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0,
                        &prog, sizeof(prog)) != 0) {
        fprintf(stderr, "launch: %s\n", plow_hsa_last_error()); return 1;
    }
    if (plow_hsa_wait(h, 0) != 0) { fprintf(stderr, "wait failed\n"); return 1; }
    printf("interpreter returned\n\n");

    plow_hsa_copy_d2h(h, 0, h_out, d_cap, ACT * 2);
    plow_hsa_download(h, 0, h_ctr, d_ctr, sizeof(h_ctr));
    printf("counters: c0=%u c1=%u c2=%u  (expect %u each)\n",
           h_ctr[0], h_ctr[PLOW_CTR_STRIDE], h_ctr[2 * PLOW_CTR_STRIDE], NCU);

    /* --- CPU reference for the whole chain -------------------------------- */
    float* emb = malloc(ACT * sizeof(float));
    float* want = malloc(ACT * sizeof(float));
    for (unsigned t = 0; t < NTOK; t++)
        for (unsigned i = 0; i < HID; i++)
            emb[(size_t)t * HID + i] =
                bf2f(f2bf(bf2f(h_table[(size_t)h_ids[t] * HID + i]) * SCALE));
    for (unsigned t = 0; t < NTOK; t++) {
        double ss = 0.0;
        for (unsigned i = 0; i < HID; i++) {
            const double v = emb[(size_t)t * HID + i];
            ss += v * v;
        }
        const double inv = pow(ss / HID + EPS, -0.5);
        for (unsigned i = 0; i < HID; i++) {
            const size_t k = (size_t)t * HID + i;
            const float nrm = (float)bf2f(f2bf((float)(emb[k] * inv * bf2f(h_gamma[i]))));
            const float res = (float)bf2f(f2bf((nrm + emb[k]) * LS));
            want[k] = CAP * tanhf(res / CAP);
        }
    }

    double worst = 0.0; size_t at = 0;
    for (size_t i = 0; i < ACT; i++) {
        const double d = fabs(bf2f(h_out[i]) - want[i]) / (fabs(want[i]) + 1e-3);
        if (d > worst) { worst = d; at = i; }
    }
    const int ctr_ok = (h_ctr[0] == NCU && h_ctr[PLOW_CTR_STRIDE] == NCU &&
                        h_ctr[2 * PLOW_CTR_STRIDE] == NCU);
    const int num_ok = worst < 2e-2;
    printf("chained result: %s (worst rel %.5f at %zu)\n", num_ok ? "PASS" : "FAIL", worst, at);
    if (!num_ok) printf("   got %.5f want %.5f\n", bf2f(h_out[at]), want[at]);
    printf("counter protocol: %s\n\n", ctr_ok ? "PASS" : "FAIL");

    const int ok = ctr_ok && num_ok;
    printf("%s\n", ok ? "PERSISTENT INTERPRETER WORKS" : "FAILED");
    plow_hsa_shutdown(h);
    return ok ? 0 : 1;
}
