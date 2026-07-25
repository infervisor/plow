/* net_gemma_block_test.c — run a REAL network end to end on the interpreter.
 *
 * plowc (crates/plowc/src/bin/tinygemma.rs) compiles a Gemma-4-shaped prefill network
 * into a packet blob. This loads it, generates the weights from a fixed seed, hands the
 * packets to the persistent interpreter, and checks the logits against an independent
 * fp32 CPU reference. Truth is the CPU reference.
 *
 * Then it prints the packet trace. That is the point of the whole exercise: a single-op
 * benchmark cannot tell you whether the SCHEDULE is any good, and the schedule is what
 * plow exists to optimise. The trace says, per packet, how long each workgroup STALLED
 * on its producers versus how long it did WORK — and therefore where the machine is
 * idle and why.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- the network. Must match crates/plowc/src/bin/tinygemma.rs. ---- */
#define T 512
#define H 1024
#define II 2048
#define N_HEAD 4
#define HD 256
#define N_KV_HEAD 2
#define LAYERS 2
#define VOCAB 1024
#define EPSV 1e-6f
#define SOFTCAPV 30.0f
#define LAYER_SCALAR 0.75f
#define QDIM (N_HEAD * HD)
#define KVDIM (N_KV_HEAD * HD)
static const unsigned WINDOW[LAYERS] = {256, 0};

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

typedef uint16_t bf16;
static inline float b2f(bf16 v) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)v << 16;
    return c.f;
}
static inline bf16 f2b(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    uint32_t r = c.u + 0x7fff + ((c.u >> 16) & 1);
    return (bf16)(r >> 16);
}

/* ---- blob ---- */
#define NAME_LEN 80
typedef struct {
    char name[NAME_LEN];
    uint64_t bytes;
    uint64_t init_off; /* UINT64_MAX = no compiler-supplied data */
} TensorDecl;
typedef struct {
    uint32_t n_cu, n_inst, n_stream, n_wait, n_succ, n_counter, n_tensor, _pad;
    PlowDevInst* insts;
    PlowStreamEnt* stream;
    uint32_t *stream_ofs, *stream_len, *succs;
    PlowWait* waits;
    TensorDecl* tensors;
    void* raw;
} Blob;

static int load_blob(const char* path, Blob* b) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* p = malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
    if (memcmp(p, "PLOWDEV\x01", 8)) { printf("bad magic\n"); return 1; }
    b->raw = p;
    uint32_t* hdr = (uint32_t*)(p + 8);
    b->n_cu = hdr[0]; b->n_inst = hdr[1]; b->n_stream = hdr[2];
    b->n_wait = hdr[3]; b->n_succ = hdr[4]; b->n_counter = hdr[5]; b->n_tensor = hdr[6];
    uint8_t* q = p + 8 + 32;
    b->insts = (PlowDevInst*)q;               q += (size_t)b->n_inst * sizeof(PlowDevInst);
    b->stream = (PlowStreamEnt*)q;            q += (size_t)b->n_stream * sizeof(PlowStreamEnt);
    b->stream_ofs = (uint32_t*)q;             q += (size_t)b->n_cu * 4;
    b->stream_len = (uint32_t*)q;             q += (size_t)b->n_cu * 4;
    b->waits = (PlowWait*)q;                  q += (size_t)b->n_wait * sizeof(PlowWait);
    b->succs = (uint32_t*)q;                  q += (size_t)b->n_succ * 4;
    q += 8; /* init-blob size */
    b->tensors = (TensorDecl*)q;
    return 0;
}

/* ---- deterministic weights: same bits every run, so a mismatch is a real bug ---- */
static uint64_t rng_s;
static void seed(uint64_t s) { rng_s = s * 6364136223846793005ULL + 1442695040888963407ULL; }
static float frand(void) {
    rng_s = rng_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((int32_t)(rng_s >> 33) % 2001 - 1000) / 20000.0f; /* +-0.05 */
}

/* ---- fp32 CPU reference ---- */
static float *A_, *Bt_;
static void rms_norm(float* out, const float* x, const bf16* g, int rows, int feat, float eps) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * feat;
        double ss = 0;
        for (int i = 0; i < feat; i++) ss += (double)xr[i] * xr[i];
        float inv = 1.0f / sqrtf((float)(ss / feat) + eps);
        for (int i = 0; i < feat; i++) {
            float gg = g ? b2f(g[i]) : 1.0f;
            out[(size_t)r * feat + i] = b2f(f2b(xr[i] * inv * gg)); /* op writes bf16 */
        }
    }
}
static void gemm(float* C, const float* A, const bf16* B, int M, int N, int K) {
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0;
            const float* a = A + (size_t)m * K;
            const bf16* b = B + (size_t)n * K;
            for (int k = 0; k < K; k++) acc += a[k] * b2f(b[k]);
            C[(size_t)m * N + n] = b2f(f2b(acc)); /* the op stores bf16 */
        }
}
/* per-head RMSNorm over head_dim, then optional half-split RoPE */
static void headnorm_rope(float* out, const float* x, const bf16* g, const float* cs,
                          const float* sn, const int* pos, int ntok, int nhead, int hd,
                          float eps) {
    const int H2 = hd / 2;
    for (int t = 0; t < ntok; t++)
        for (int h = 0; h < nhead; h++) {
            const float* xi = x + ((size_t)t * nhead + h) * hd;
            float* oi = out + ((size_t)t * nhead + h) * hd;
            double ss = 0;
            for (int i = 0; i < hd; i++) ss += (double)xi[i] * xi[i];
            float inv = 1.0f / sqrtf((float)(ss / hd) + eps);
            float v[512];
            for (int i = 0; i < hd; i++) v[i] = xi[i] * inv * (g ? b2f(g[i]) : 1.0f);
            if (cs) {
                float r[512];
                for (int i = 0; i < hd; i++) {
                    int j = (i < H2) ? i : i - H2;
                    float c = cs[(size_t)pos[t] * H2 + j], s = sn[(size_t)pos[t] * H2 + j];
                    r[i] = (i < H2) ? (v[i] * c - v[i + H2] * s) : (v[i] * c + v[i - H2] * s);
                }
                memcpy(v, r, sizeof(float) * hd);
            }
            for (int i = 0; i < hd; i++) oi[i] = b2f(f2b(v[i]));
        }
}
static void flash(float* O, const float* Q, const float* K, const float* V, int nq, int nkv,
                  int nhead, int nkvhead, int window, int hd) {
    const int gqa = nhead / nkvhead;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < nq; t++)
        for (int h = 0; h < nhead; h++) {
            const int hk = h / gqa;
            float m = -1e30f, l = 0.0f;
            float acc[512];
            for (int d = 0; d < hd; d++) acc[d] = 0;
            for (int s = 0; s <= t; s++) {
                if (window && (t - s) >= window) continue; /* INCLUSIVE window */
                const float* qq = Q + ((size_t)t * nhead + h) * hd;
                const float* kk = K + ((size_t)s * nkvhead + hk) * hd;
                float dot = 0;
                for (int d = 0; d < hd; d++) dot += qq[d] * kk[d]; /* scale = 1.0 */
                float mn = dot > m ? dot : m;
                float corr = expf(m - mn), p = expf(dot - mn);
                const float* vv = V + ((size_t)s * nkvhead + hk) * hd;
                for (int d = 0; d < hd; d++) acc[d] = acc[d] * corr + p * vv[d];
                l = l * corr + p;
                m = mn;
            }
            float* oo = O + ((size_t)t * nhead + h) * hd;
            for (int d = 0; d < hd; d++) oo[d] = b2f(f2b(acc[d] / (l > 0 ? l : 1)));
        }
}
static float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

int main(int argc, char** argv) {
    const char* blobp = argc > 1 ? argv[1] : "tinygemma.pkt";
    Blob B;
    if (load_blob(blobp, &B)) return 1;

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, gfx, &cus, &lds);
    printf("dev0: %s  CUs=%u\n", gfx, cus);
    if (B.n_cu != cus) printf("  (blob built for %u CUs, device has %u)\n", B.n_cu, cus);

    FILE* f = fopen("interp.elf", "rb");
    if (!f) { printf("interp.elf missing\n"); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc((size_t)co_n);
    if (fread(co, 1, (size_t)co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, (size_t)co_n)) { printf("load failed\n"); return 1; }
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, "plow_interp_gfx950", &k)) { printf("no kernel\n"); return 1; }

    printf("program: %u packets, %u workgroup-packets, %u counters, %u tensors\n\n",
           B.n_inst, B.n_stream, B.n_counter, B.n_tensor);

    /* ---- host tensors, deterministic ---- */
    void** dev = calloc(B.n_tensor, sizeof(void*));
    void** hostp = calloc(B.n_tensor, sizeof(void*));
    for (uint32_t i = 0; i < B.n_tensor; i++) {
        hostp[i] = calloc(1, B.tensors[i].bytes);
        dev[i] = plow_hsa_alloc(h, 0, B.tensors[i].bytes);
    }
    int th(const char* nm) {
        for (uint32_t i = 0; i < B.n_tensor; i++)
            if (!strcmp(B.tensors[i].name, nm)) return (int)i;
        printf("missing tensor %s\n", nm);
        exit(1);
    }
    /* weights + inputs */
    for (uint32_t i = 0; i < B.n_tensor; i++) {
        const char* nm = B.tensors[i].name;
        size_t n2 = B.tensors[i].bytes / 2;
        if (!strcmp(nm, "ids")) {
            seed(11);
            int* p = hostp[i];
            for (int t = 0; t < T; t++) p[t] = (int)((rng_s >> 20) % VOCAB), rng_s = rng_s * 6364136223846793005ULL + 1;
        } else if (!strcmp(nm, "pos")) {
            int* p = hostp[i];
            for (int t = 0; t < T; t++) p[t] = t;
        } else if (!strcmp(nm, "cos") || !strcmp(nm, "sin")) {
            float* p = hostp[i];
            const int H2 = HD / 2;
            for (int t = 0; t < T; t++)
                for (int j = 0; j < H2; j++) {
                    double invf = pow(10000.0, -2.0 * j / (double)HD);
                    double a = t * invf;
                    p[(size_t)t * H2 + j] = (float)(nm[0] == 'c' ? cos(a) : sin(a));
                }
        } else if (strstr(nm, ".qn")) {
            /* q_norm carries the attention scale.
             *
             * Gemma sets the attention scale to 1.0 -- there is NO 1/sqrt(head_dim) in
             * the kernel -- because the TRAINED q_norm weight absorbs it. A synthetic
             * q_norm of ~1.0 therefore leaves the q.k dot products at ~sqrt(head_dim)
             * (~16 here), which makes the softmax pathologically peaked: a 0.4% bf16
             * error in q or k moves a logit by ~0.06 and the exponential turns that into
             * several percent on the output. Stacking two such layers is what took the
             * end-to-end error from 0.004 to 0.020 -- an artefact of the test weights,
             * not of the kernels. Scaling q_norm by 1/sqrt(head_dim), as the real trained
             * weight effectively does, restores a well-conditioned softmax. */
            seed(100 + i);
            bf16* p = hostp[i];
            const float s = 1.0f / sqrtf((float)HD);
            for (size_t j = 0; j < n2; j++) p[j] = f2b(s * (1.0f + frand()));
        } else if (strstr(nm, ".g_") || strstr(nm, ".kn") || !strcmp(nm, "g_final")) {
            seed(100 + i); /* norm gammas near 1 */
            bf16* p = hostp[i];
            for (size_t j = 0; j < n2; j++) p[j] = f2b(1.0f + frand());
        } else if (strstr(nm, ".w") || !strcmp(nm, "emb")) {
            seed(1000 + i);
            bf16* p = hostp[i];
            for (size_t j = 0; j < n2; j++) p[j] = f2b(frand());
        }
        plow_hsa_upload(h, 0, dev[i], hostp[i], B.tensors[i].bytes);
    }

    /* ---- upload the program ---- */
    void* d_inst = plow_hsa_alloc(h, 0, (size_t)B.n_inst * sizeof(PlowDevInst));
    void* d_stream = plow_hsa_alloc(h, 0, (size_t)B.n_stream * sizeof(PlowStreamEnt));
    void* d_sofs = plow_hsa_alloc(h, 0, (size_t)B.n_cu * 4);
    void* d_slen = plow_hsa_alloc(h, 0, (size_t)B.n_cu * 4);
    void* d_waits = plow_hsa_alloc(h, 0, (size_t)(B.n_wait ? B.n_wait : 1) * sizeof(PlowWait));
    void* d_succs = plow_hsa_alloc(h, 0, (size_t)(B.n_succ ? B.n_succ : 1) * 4);
    void* d_ctr = plow_hsa_alloc(h, 0, (size_t)B.n_counter * PLOW_CTR_STRIDE * 4);
    void* d_tens = plow_hsa_alloc(h, 0, (size_t)B.n_tensor * sizeof(void*));
    void* d_trace = plow_hsa_alloc(h, 0, (size_t)B.n_stream * sizeof(PlowTraceRec));
    plow_hsa_upload(h, 0, d_inst, B.insts, (size_t)B.n_inst * sizeof(PlowDevInst));
    plow_hsa_upload(h, 0, d_stream, B.stream, (size_t)B.n_stream * sizeof(PlowStreamEnt));
    plow_hsa_upload(h, 0, d_sofs, B.stream_ofs, (size_t)B.n_cu * 4);
    plow_hsa_upload(h, 0, d_slen, B.stream_len, (size_t)B.n_cu * 4);
    if (B.n_wait) plow_hsa_upload(h, 0, d_waits, B.waits, (size_t)B.n_wait * sizeof(PlowWait));
    if (B.n_succ) plow_hsa_upload(h, 0, d_succs, B.succs, (size_t)B.n_succ * 4);
    plow_hsa_upload(h, 0, d_tens, dev, (size_t)B.n_tensor * sizeof(void*));
    uint32_t* zc = calloc((size_t)B.n_counter * PLOW_CTR_STRIDE, 4);
    plow_hsa_upload(h, 0, d_ctr, zc, (size_t)B.n_counter * PLOW_CTR_STRIDE * 4);

    PlowProgram prog;
    memset(&prog, 0, sizeof(prog));
    prog.insts = d_inst; prog.stream = d_stream; prog.stream_ofs = d_sofs;
    prog.stream_len = d_slen; prog.waits = d_waits; prog.succs = d_succs;
    prog.counters = d_ctr; prog.tensors = (void* const*)d_tens; prog.trace = d_trace;

    /* ---- run ---- */
    const double t0 = now();
    if (plow_hsa_launch(h, 0, &k, B.n_cu * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &prog,
                        sizeof(prog))) {
        printf("LAUNCH FAILED\n");
        return 1;
    }
    plow_hsa_wait(h, 0);
    const double wall = now() - t0;
    printf("interpreter: 1 launch, %u workgroups, %u packets  ->  %.3f ms\n\n", B.n_cu, B.n_inst,
           wall * 1e3);

    /* ---- CPU reference ---- */
    printf("computing fp32 CPU reference...\n");
    const double rt0 = now();
    static float x[T * H], hn[T * H], qg[T * QDIM], kg[T * KVDIM], vg[T * KVDIM];
    static float qq[T * QDIM], kk[T * KVDIM], vv[T * KVDIM], at[T * QDIM];
    static float og[T * H], on[T * H], gt[T * II], ut[T * II], fu[T * II];
    static float dg[T * H], dn[T * H], lg[T * VOCAB];
    const int* ids = hostp[th("ids")];
    const int* pos = hostp[th("pos")];
    const float* cs = hostp[th("cos")];
    const float* sn = hostp[th("sin")];
    const bf16* emb = hostp[th("emb")];

    float escale = b2f(f2b(sqrtf((float)H)));
    for (int t = 0; t < T; t++)
        for (int i = 0; i < H; i++)
            x[(size_t)t * H + i] = b2f(f2b(b2f(emb[(size_t)ids[t] * H + i]) * escale));

    char nm[64];
    for (int l = 0; l < LAYERS; l++) {
#define W(s) ((const bf16*)hostp[(sprintf(nm, "l%d." s, l), th(nm))])
        rms_norm(hn, x, W("g_in"), T, H, EPSV);
        gemm(qg, hn, W("wq"), T, QDIM, H);
        gemm(kg, hn, W("wk"), T, KVDIM, H);
        gemm(vg, hn, W("wv"), T, KVDIM, H);
        headnorm_rope(qq, qg, W("qn"), cs, sn, pos, T, N_HEAD, HD, EPSV);
        headnorm_rope(kk, kg, W("kn"), cs, sn, pos, T, N_KV_HEAD, HD, EPSV);
        headnorm_rope(vv, vg, NULL, NULL, NULL, pos, T, N_KV_HEAD, HD, EPSV); /* v_norm */
        flash(at, qq, kk, vv, T, T, N_HEAD, N_KV_HEAD, (int)WINDOW[l], HD);
        gemm(og, at, W("wo"), T, H, QDIM);
        rms_norm(on, og, W("g_pa"), T, H, EPSV);
        for (int i = 0; i < T * H; i++) x[i] = b2f(f2b(x[i] + on[i]));
        rms_norm(hn, x, W("g_pf"), T, H, EPSV);
        gemm(gt, hn, W("wg"), T, II, H);
        gemm(ut, hn, W("wu"), T, II, H);
        for (int i = 0; i < T * II; i++) fu[i] = b2f(f2b(gelu_tanh(gt[i]) * ut[i]));
        gemm(dg, fu, W("wd"), T, H, II);
        rms_norm(dn, dg, W("g_po"), T, H, EPSV);
        for (int i = 0; i < T * H; i++) x[i] = b2f(f2b((x[i] + dn[i]) * LAYER_SCALAR));
#undef W
    }
    rms_norm(hn, x, (const bf16*)hostp[th("g_final")], T, H, EPSV);
    gemm(lg, hn, emb, T, VOCAB, H);
    for (int i = 0; i < T * VOCAB; i++)
        lg[i] = b2f(f2b(SOFTCAPV * tanhf(lg[i] / SOFTCAPV)));
    printf("  reference took %.1f s\n\n", now() - rt0);

    /* ---- compare INTERMEDIATES first: a single end-of-network number cannot tell a
     * real bug from accumulated bf16 rounding, and "it matched the threshold" is not a
     * result. Checking the tensors the layers actually produce localises any error. ---- */
    struct { const char* nm; const float* ref; size_t n; } chk[] = {
        {"at", at, (size_t)T * QDIM},   /* attention out, last layer  */
        {"fu", fu, (size_t)T * II},     /* GeGLU out, last layer      */
        {"x",  x,  (size_t)T * H},      /* residual stream, final     */
        {"hn", hn, (size_t)T * H},      /* final norm                 */
    };
    printf("  %-8s %10s %10s\n", "tensor", "rel(rms)", "worst abs");
    for (unsigned c = 0; c < sizeof(chk) / sizeof(chk[0]); c++) {
        bf16* g = malloc(chk[c].n * 2);
        plow_hsa_download(h, 0, g, dev[th(chk[c].nm)], chk[c].n * 2);
        double nu = 0, de = 0, wo = 0;
        for (size_t i = 0; i < chk[c].n; i++) {
            double a = b2f(g[i]), b = chk[c].ref[i];
            nu += (a - b) * (a - b); de += b * b;
            if (fabs(a - b) > wo) wo = fabs(a - b);
        }
        printf("  %-8s %10.5f %10.4f\n", chk[c].nm, sqrt(nu / (de + 1e-30)), wo);
        free(g);
    }
    printf("\n");

    bf16* got = malloc((size_t)T * VOCAB * 2);
    plow_hsa_download(h, 0, got, dev[th("logits")], (size_t)T * VOCAB * 2);
    double num = 0, den = 0, worst = 0;
    int wi = 0;
    for (int i = 0; i < T * VOCAB; i++) {
        double g = b2f(got[i]), w = lg[i];
        num += (g - w) * (g - w);
        den += w * w;
        double d = fabs(g - w);
        if (d > worst) { worst = d; wi = i; }
    }
    double rel = sqrt(num / (den + 1e-30));
    int ok = rel < 0.02;
    printf("LOGITS: rel(rms) = %.5f  worst abs = %.4f at %d (got %.4f want %.4f)  -> %s\n\n", rel,
           worst, wi, b2f(got[wi]), lg[wi], ok ? "MATCH" : "MISMATCH");

    /* ---- trace ---- */
    PlowTraceRec* tr = malloc((size_t)B.n_stream * sizeof(PlowTraceRec));
    plow_hsa_download(h, 0, tr, d_trace, (size_t)B.n_stream * sizeof(PlowTraceRec));
    static const char* OPN[] = {"nop", "rmsnorm", "rowrms", "headnorm_rope", "residual",
                                "glu", "embed",   "softcap", "gemm",  "gemm_norm",
                                "gemv", "flash_prefill", "flash_decode", "flash_merge"};
    uint64_t t_min = ~0ull, t_max = 0;
    for (uint32_t i = 0; i < B.n_stream; i++) {
        if (tr[i].t_arrive && tr[i].t_arrive < t_min) t_min = tr[i].t_arrive;
        if (tr[i].t_end > t_max) t_max = tr[i].t_end;
    }
    /* derive the tick rate from wall time — no need to guess s_memrealtime's frequency */
    const double us_per_tick = (wall * 1e6) / (double)(t_max - t_min);

    printf("PACKET TRACE (%u records). stall = waiting on producers, work = op body.\n",
           B.n_stream);
    printf("  tick rate derived from wall clock: %.1f MHz\n\n", 1.0 / us_per_tick);
    printf("  %-3s %-14s %4s  %9s %9s  %9s %9s\n", "pkt", "op", "CUs", "start(us)", "end(us)",
           "stall(us)", "work(us)");
    double busy = 0;
    for (uint32_t p = 0; p < B.n_inst; p++) {
        uint64_t s = ~0ull, e = 0;
        double stall = 0, work = 0;
        int n = 0;
        for (uint32_t i = 0; i < B.n_stream; i++) {
            if (tr[i].inst != p) continue;
            if (tr[i].t_arrive < s) s = tr[i].t_arrive;
            if (tr[i].t_end > e) e = tr[i].t_end;
            stall += (double)(tr[i].t_ready - tr[i].t_arrive);
            work += (double)(tr[i].t_end - tr[i].t_ready);
            n++;
        }
        if (!n) continue;
        busy += work * us_per_tick;
        printf("  %-3u %-14s %4d  %9.1f %9.1f  %9.1f %9.1f\n", p, OPN[B.insts[p].op], n,
               (double)(s - t_min) * us_per_tick, (double)(e - t_min) * us_per_tick,
               stall / n * us_per_tick, work / n * us_per_tick);
    }
    const double span = (double)(t_max - t_min) * us_per_tick;
    printf("\n  span %.1f us   CU-time busy %.1f us of %.1f available  -> %.1f%% machine utilisation\n",
           span, busy, span * B.n_cu, 100.0 * busy / (span * B.n_cu));

    printf("\n%s\n", ok ? "NETWORK END-TO-END CORRECT" : "*** NETWORK MISMATCH ***");
    plow_hsa_shutdown(h);
    return ok ? 0 : 1;
}
