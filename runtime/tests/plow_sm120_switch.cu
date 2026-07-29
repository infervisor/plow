/* plow_sm120_switch.cu — TWO models resident, interleaved decode, measured switch cost.
 *
 *   plow_sm120_switch A.pkt A-dir B.pkt B-dir prompt.ids N
 *
 * ===================== WHAT A "MODEL SWITCH" ACTUALLY IS HERE =====================
 *
 * The load-bearing question (assignment step 0) is whether anything bakes an ABSOLUTE
 * device address into the compiled program. It does not, and this harness is the proof
 * by construction:
 *
 *   - PlowDevInst.t[0..7] are tensor HANDLES (slot indices), not pointers. The device
 *     resolves them through `prog.tensors[handle]` in plow_exec's TEN() macro
 *     (interp_sm120.cu:87). The .pkt therefore contains no address at all.
 *   - PlowTensorDecl carries {name, bytes, init_off} — a SIZE, never a location.
 *   - runtime/common/memmap.c resolves an address map as `arena_base + offset`, i.e.
 *     base-relative by construction, and this harness does not even need it: it binds
 *     one cudaMalloc per tensor and hands the interpreter the resulting pointer table.
 *
 * So a "switch" is: pass a different `PlowProgram` struct to the launch. Every pointer
 * the device will follow — instruction table, stream, counters, and the tensor table
 * (hence weights AND KV) — is a field of that 128-byte struct.
 *
 * WHICH MEANS THE HOST-SIDE SWITCH COST IS NOT THE INTERESTING NUMBER. Selecting a
 * different struct is a pointer assignment; timing it measures the C compiler. The
 * cooperative kernel ALREADY exits and is relaunched every decode step (see
 * qwen3_sm120_chat.cu), so S1 "relaunch on switch" adds no launch that a same-model
 * step was not already paying. The only cost a switch can have is therefore on the
 * DEVICE — L2 eviction, TLB pressure, page residency from touching the other model's
 * 3-7 GiB of weights — and no host timer can see it.
 *
 * So the measurement here is differential, and that is deliberate:
 *
 *     switch_cost(M) = mean(step of M | previous step was the OTHER model)
 *                    - mean(step of M | previous step was M)
 *
 * with both terms taken at the SAME context length on the SAME resident allocation, so
 * the only difference between them is the interleaving. This is reported per model.
 *
 * ============================== CORRECTNESS GATE ==============================
 *
 * Interleaving is only safe if each model's per-step state is fully private. This
 * harness keeps per model: its own tensor table (so its own KV), its own counter array,
 * its own host-side instruction shadow (kv-row patch sites), and its own position. It
 * then requires the interleaved run to produce TOKEN-FOR-TOKEN the same output as that
 * model's own solo run, AND a bit-identical step-0 logits row (token match alone is
 * coarse — this campaign already measured zeroing 62% of an o_proj changing no tokens).
 *
 * Two negative controls reintroduce exactly the failure modes a switch invites. Both
 * are safe (no OOB, no deadlock: a counter left too HIGH opens gates early, it does not
 * hang) and both must break the match or the gate is worthless:
 *
 *   PLOW_NEGCTRL_SKIP_CTRZERO — do not zero the counter array on a step that follows a
 *       switch. Stale counters from this model's PREVIOUS step leave every gate already
 *       satisfied, so packets run before their producers.
 *   PLOW_NEGCTRL_STALE_POS   — on a switch, feed the OTHER model's position/kvlen, i.e.
 *       forget to restore per-model sequence state across the switch.
 *
 * BUILD NOTE — GQA FUSION. interp_sm120.cu ships PLOW_NV_FA_GF=4 and traps unless
 * gqa % GF == 0. Qwen3-4B is gqa=4 but Qwen3-1.7B is gqa=16/8=2, so a build that runs
 * BOTH must use GF=2 (4%2==0 and 2%2==0). That costs Qwen3-4B flash-decode throughput
 * and is stated rather than hidden; the Qwen3-4B reference is re-established under GF=2
 * against HF so the correctness anchor is not merely self-consistent.
 */
#include "../common/dev_blob.h"
#include "../common/dev_isa.h"
#include "../common/safetensors.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern "C" int plow_sm120_grid(int dev);
extern "C" size_t plow_sm120_smem(void);
extern "C" int plow_sm120_launch(PlowProgram* prog, int grid, cudaStream_t stream);
extern "C" int plow_sm120_sched(void);

#define CK(x)                                                                        \
    do {                                                                             \
        cudaError_t _e = (x);                                                        \
        if (_e != cudaSuccess) {                                                     \
            printf("CUDA FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #x,            \
                   cudaGetErrorString(_e));                                          \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static float b2f(uint16_t v) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)v << 16;
    return c.f;
}

typedef struct {
    PlowProgHeader h;
    PlowDevInst* insts;
    PlowStreamEnt* stream;
    uint32_t *stream_ofs, *stream_len, *succs;
    PlowWait* waits;
    void *d_inst, *d_stream, *d_sofs, *d_slen, *d_waits, *d_succs, *d_ctr;
} Prog;

typedef struct {
    PlowBlobHeader h;
    PlowTensorDecl* tensors;
    uint8_t* init;
    uint32_t* kvrow;
    Prog* prog;
} Blob;

/* One fully resident model: weights, KV, activations, program tables, and the
 * PlowProgram struct that names them all. THIS STRUCT IS THE SWITCH UNIT. */
typedef struct {
    const char* tag;
    Blob B;
    Prog* g;              /* the decode program */
    void** devp;          /* [n_tensor] host copy of the device pointer table */
    void* d_tens;         /* device copy of the above */
    PlowDevInst* h_inst;  /* pinned host shadow, kv-row patched per step */
    uint32_t* zc;         /* pinned zeroed counter block */
    size_t zc_bytes;
    int32_t* h_scalar;    /* pinned staging for ids/pos/kvlen */
    uint16_t* logit;      /* pinned logits readback */
    int t_ids, t_pos, t_kvlen, t_logits;
    uint32_t vocab;
    int max_ctx;
    PlowProgram pr;
    /* per-sequence state */
    int pos;              /* next absolute position to write */
    int best;             /* token most recently sampled */
    /* accounting */
    uint64_t bytes_weights, bytes_kv, bytes_act, bytes_other;
} Model;

static int load_blob(const char* path, Blob* b) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* p = (uint8_t*)malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
    memcpy(&b->h, p, sizeof(PlowBlobHeader));
    { const char* e = plow_blob_magic_error(b->h.magic);
      if (e) { printf("%s: %s\n", path, e); return 1; } }
    uint8_t* q = p + sizeof(PlowBlobHeader);
    b->tensors = (PlowTensorDecl*)q; q += (size_t)b->h.n_tensor * sizeof(PlowTensorDecl);
    b->init = q;                     q += b->h.init_bytes;
    b->kvrow = (uint32_t*)q;         q += (size_t)b->h.n_kvrow * 4;
    b->prog = (Prog*)calloc(b->h.n_prog, sizeof(Prog));
    for (uint32_t i = 0; i < b->h.n_prog; i++) {
        Prog* g = &b->prog[i];
        memcpy(&g->h, q, sizeof(PlowProgHeader)); q += sizeof(PlowProgHeader);
        g->insts = (PlowDevInst*)q;    q += (size_t)g->h.n_inst * sizeof(PlowDevInst);
        g->stream = (PlowStreamEnt*)q; q += (size_t)g->h.n_stream * sizeof(PlowStreamEnt);
        g->stream_ofs = (uint32_t*)q;  q += (size_t)b->h.n_cu * 4;
        g->stream_len = (uint32_t*)q;  q += (size_t)b->h.n_cu * 4;
        g->waits = (PlowWait*)q;       q += (size_t)g->h.n_wait * sizeof(PlowWait);
        g->succs = (uint32_t*)q;       q += (size_t)g->h.n_succ * 4;
    }
    return 0;
}

/* Bring one model fully resident and build its PlowProgram. */
static void model_load(Model* m, const char* tag, const char* pkt, const char* dir, int grid) {
    m->tag = tag;
    if (load_blob(pkt, &m->B)) exit(1);
    Blob* B = &m->B;
    if (grid != (int)B->h.n_cu) {
        printf("FATAL[%s]: interpreter grid %d != packet n_cu %u; recompile with n_cu=%d\n",
               tag, grid, B->h.n_cu, grid);
        exit(1);
    }
    Safet S;
    if (st_open(&S, dir)) { printf("no safetensors in %s\n", dir); exit(1); }

    const size_t STAGE = 64u << 20;
    void* stage = NULL;
    CK(cudaMallocHost(&stage, STAGE));
    m->devp = (void**)calloc(B->h.n_tensor, sizeof(void*));
    m->t_ids = m->t_pos = m->t_kvlen = m->t_logits = -1;
    int nw = 0;
    const double t0 = now();
    for (uint32_t i = 0; i < B->h.n_tensor; i++) {
        PlowTensorDecl* td = &B->tensors[i];
        CK(cudaMalloc(&m->devp[i], td->bytes));
        if (!strcmp(td->name, "in.ids")) m->t_ids = (int)i;
        if (!strcmp(td->name, "in.pos")) m->t_pos = (int)i;
        if (!strcmp(td->name, "in.kvlen")) m->t_kvlen = (int)i;
        if (!strcmp(td->name, "act.logits")) m->t_logits = (int)i;

        if (!strncmp(td->name, "kv.", 3)) m->bytes_kv += td->bytes;
        else if (!strncmp(td->name, "act.", 4)) m->bytes_act += td->bytes;
        else if (!strncmp(td->name, "model.", 6)) m->bytes_weights += td->bytes;
        else m->bytes_other += td->bytes;

        if (!strncmp(td->name, "model.", 6)) {
            uint64_t got = 0;
            const uint8_t* src = st_find(&S, td->name, &got);
            if (!src) { printf("MISSING WEIGHT [%s]: %s\n", tag, td->name); exit(1); }
            if (got != td->bytes) {
                printf("SIZE MISMATCH [%s] %s (want %llu got %llu)\n", tag, td->name,
                       (unsigned long long)td->bytes, (unsigned long long)got);
                exit(1);
            }
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, src + o, n);
                CK(cudaMemcpy((uint8_t*)m->devp[i] + o, stage, n, cudaMemcpyHostToDevice));
            }
            nw++;
        } else if (td->init_off != PLOW_INIT_NONE) {
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, B->init + td->init_off + o, n);
                CK(cudaMemcpy((uint8_t*)m->devp[i] + o, stage, n, cudaMemcpyHostToDevice));
            }
        } else {
            CK(cudaMemset(m->devp[i], 0, td->bytes));
        }
    }
    CK(cudaFreeHost(stage));
    if (m->t_ids < 0 || m->t_pos < 0 || m->t_kvlen < 0 || m->t_logits < 0) {
        printf("[%s] blob missing in.ids/in.pos/in.kvlen/act.logits\n", tag);
        exit(1);
    }
    CK(cudaMalloc(&m->d_tens, (size_t)B->h.n_tensor * sizeof(void*)));
    CK(cudaMemcpy(m->d_tens, m->devp, (size_t)B->h.n_tensor * sizeof(void*),
                  cudaMemcpyHostToDevice));

    /* decode program = last */
    m->g = &B->prog[B->h.n_prog - 1];
    Prog* g = m->g;
    if (g->h.t != 1) { printf("[%s] last program is not T=1 decode\n", tag); exit(1); }
    for (uint32_t j = 0; j < g->h.n_stream; j++)
        if (g->stream[j].seg != 0 || (g->stream[j].flags & (PLOW_SE_FINE | PLOW_SE_XCTR))) {
            printf("FATAL[%s]: stream entry %u is segmented/fine-gated; this interpreter "
                   "implements the coarse single-segment path only.\n", tag, j);
            exit(1);
        }
    CK(cudaMalloc(&g->d_inst, (size_t)g->h.n_inst * sizeof(PlowDevInst)));
    CK(cudaMalloc(&g->d_stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt)));
    CK(cudaMalloc(&g->d_sofs, (size_t)B->h.n_cu * 4));
    CK(cudaMalloc(&g->d_slen, (size_t)B->h.n_cu * 4));
    CK(cudaMalloc(&g->d_waits, (size_t)(g->h.n_wait ? g->h.n_wait : 1) * sizeof(PlowWait)));
    CK(cudaMalloc(&g->d_succs, (size_t)(g->h.n_succ ? g->h.n_succ : 1) * 4));
    CK(cudaMalloc(&g->d_ctr, (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4));
    CK(cudaMemcpy(g->d_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_stream, g->stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_sofs, g->stream_ofs, (size_t)B->h.n_cu * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_slen, g->stream_len, (size_t)B->h.n_cu * 4, cudaMemcpyHostToDevice));
    if (g->h.n_wait)
        CK(cudaMemcpy(g->d_waits, g->waits, (size_t)g->h.n_wait * sizeof(PlowWait), cudaMemcpyHostToDevice));
    if (g->h.n_succ)
        CK(cudaMemcpy(g->d_succs, g->succs, (size_t)g->h.n_succ * 4, cudaMemcpyHostToDevice));

    CK(cudaMallocHost(&m->h_inst, (size_t)g->h.n_inst * sizeof(PlowDevInst)));
    memcpy(m->h_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    m->zc_bytes = (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4;
    CK(cudaMallocHost(&m->zc, m->zc_bytes));
    memset(m->zc, 0, m->zc_bytes);
    CK(cudaMallocHost(&m->h_scalar, 64));
    m->vocab = (uint32_t)(B->tensors[m->t_logits].bytes / 2);
    m->max_ctx = (int)(B->tensors[m->t_pos].bytes / 4);
    CK(cudaMallocHost(&m->logit, (size_t)m->vocab * 2));

    memset(&m->pr, 0, sizeof(m->pr));
    m->pr.insts = (const PlowDevInst*)g->d_inst;
    m->pr.stream = (const PlowStreamEnt*)g->d_stream;
    m->pr.stream_ofs = (const uint32_t*)g->d_sofs;
    m->pr.stream_len = (const uint32_t*)g->d_slen;
    m->pr.waits = (const PlowWait*)g->d_waits;
    m->pr.succs = (const uint32_t*)g->d_succs;
    m->pr.counters = (uint32_t*)g->d_ctr;
    m->pr.tensors = (void* const*)m->d_tens;
    m->pr.trace = NULL;

    printf("[%s] %s: %u tensors, %d weights (%.3f GiB), KV %.3f GiB, act %.3f GiB, "
           "other %.3f GiB | decode %u packets %u wg-packets %u counters | vocab %u max_ctx %d "
           "| loaded in %.1f s\n",
           tag, dir, B->h.n_tensor, nw, m->bytes_weights / 1073741824.0,
           m->bytes_kv / 1073741824.0, m->bytes_act / 1073741824.0,
           m->bytes_other / 1073741824.0, g->h.n_inst, g->h.n_stream, g->h.n_counter,
           m->vocab, m->max_ctx, now() - t0);
}

/* Reset a model's sequence: zero its whole KV cache and rewind the position. Weights and
 * program tables are untouched — this is a per-REQUEST reset, not a reload. */
static void model_reset_seq(Model* m) {
    for (uint32_t i = 0; i < m->B.h.n_tensor; i++)
        if (!strncmp(m->B.tensors[i].name, "kv.", 3))
            CK(cudaMemset(m->devp[i], 0, m->B.tensors[i].bytes));
    m->pos = 0;
    m->best = -1;
}

static int g_nc_skip_ctrzero = 0;
static int g_nc_stale_pos = 0;

/* One decode step of `m` at its own current position. `feed >= 0` forces in.ids (prompt
 * consumption); feed < 0 keeps whatever the device's own ARGMAX_FIN left there.
 * `switched` says the PREVIOUS launch ran the other model — the negative controls key
 * on it, because the whole point is state that a switch must restore. */
static void model_step(Model* m, int feed, int switched, int other_pos) {
    Prog* g = m->g;
    const int pos = m->pos;
    uint32_t lo = g->h.n_inst - 1, hi = 0;
    for (uint32_t k = 0; k < m->B.h.n_kvrow; k++) {
        const uint32_t ix = m->B.kvrow[k];
        m->h_inst[ix].i[3] = (uint32_t)pos;
        if (ix < lo) lo = ix;
        if (ix > hi) hi = ix;
    }
    if (m->B.h.n_kvrow)
        CK(cudaMemcpy((uint8_t*)g->d_inst + (size_t)lo * sizeof(PlowDevInst), &m->h_inst[lo],
                      (size_t)(hi - lo + 1) * sizeof(PlowDevInst), cudaMemcpyHostToDevice));
    if (feed >= 0) {
        m->h_scalar[0] = (int32_t)feed;
        CK(cudaMemcpy(m->devp[m->t_ids], m->h_scalar, 4, cudaMemcpyHostToDevice));
    }
    /* NEGATIVE CONTROL: on a switch, fail to restore this model's own sequence position
     * and use the other model's instead. Both are < max_ctx, so this is a wrong-answer
     * bug, not a memory fault — exactly the class of silent corruption a switch invites. */
    const int use_pos = (g_nc_stale_pos && switched) ? other_pos : pos;
    m->h_scalar[0] = (int32_t)use_pos;
    CK(cudaMemcpy(m->devp[m->t_pos], m->h_scalar, 4, cudaMemcpyHostToDevice));
    m->h_scalar[0] = (int32_t)(use_pos + 1);
    CK(cudaMemcpy(m->devp[m->t_kvlen], m->h_scalar, 4, cudaMemcpyHostToDevice));
    /* NEGATIVE CONTROL: skip the counter zeroing on a switch step. The counters still
     * hold this model's PREVIOUS step's final values, so every wait threshold is already
     * met and packets run before their producers. Cannot deadlock (gates open early, not
     * late) — it produces wrong arithmetic, which is what the gate must catch. */
    if (!(g_nc_skip_ctrzero && switched))
        CK(cudaMemcpy(g->d_ctr, m->zc, m->zc_bytes, cudaMemcpyHostToDevice));

    const int rc = plow_sm120_launch(&m->pr, plow_sm120_grid(0), 0);
    if (rc != cudaSuccess) {
        printf("LAUNCH FAILED [%s] at pos %d: %s\n", m->tag, pos,
               cudaGetErrorString((cudaError_t)rc));
        exit(1);
    }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(m->h_scalar, m->devp[m->t_ids], 4, cudaMemcpyDeviceToHost));
    m->best = (int)m->h_scalar[0];
    m->pos = pos + 1;
}

static void dump_logits(Model* m, uint16_t* out) {
    CK(cudaMemcpy(out, m->devp[m->t_logits], (size_t)m->vocab * 2, cudaMemcpyDeviceToHost));
}

static void top5(Model* m, const uint16_t* logit) {
    int ti[5];
    float tv[5];
    for (int k = 0; k < 5; k++) { ti[k] = -1; tv[k] = -1e30f; }
    for (uint32_t v = 0; v < m->vocab; v++) {
        float x = b2f(logit[v]);
        for (int k = 0; k < 5; k++)
            if (x > tv[k]) {
                for (int j = 4; j > k; j--) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; }
                tv[k] = x; ti[k] = (int)v;
                break;
            }
    }
    printf("[%s] step-0 top-5:", m->tag);
    for (int k = 0; k < 5; k++) printf("  %d(%+.5f)", ti[k], tv[k]);
    printf("\n");
}

/* Run `m` solo: consume the prompt, then generate n_gen tokens greedily.
 * `tok` receives the generated ids; `l0` receives the step-0 (post-prompt) logits row. */
static void run_solo(Model* m, const int32_t* prompt, int n_prompt, int n_gen, int* tok,
                     uint16_t* l0, double* step_ms, int warmup) {
    model_reset_seq(m);
    for (int i = 0; i < n_prompt; i++) model_step(m, prompt[i], 0, 0);
    dump_logits(m, l0);
    for (int s = 0; s < n_gen; s++) {
        tok[s] = m->best;
        const double t0 = now();
        model_step(m, -1, 0, 0);
        const double d = now() - t0;
        if (s >= warmup) step_ms[s - warmup] = d * 1e3;
    }
}

static void stats(const char* label, const double* v, int n) {
    if (n <= 0) { printf("  %-28s  (no samples)\n", label); return; }
    double s = 0, mn = 1e30, mx = -1e30;
    for (int i = 0; i < n; i++) { s += v[i]; if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i]; }
    const double mean = s / n;
    double var = 0;
    for (int i = 0; i < n; i++) { const double e = v[i] - mean; var += e * e; }
    const double sd = n > 1 ? sqrt(var / (n - 1)) : 0.0;
    double* c = (double*)malloc(sizeof(double) * (size_t)n);
    memcpy(c, v, sizeof(double) * (size_t)n);
    for (int i = 1; i < n; i++) {
        const double x = c[i];
        int j = i - 1;
        while (j >= 0 && c[j] > x) { c[j + 1] = c[j]; j--; }
        c[j + 1] = x;
    }
    const double med = n & 1 ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    printf("  %-28s  n=%-4d mean %7.3f  med %7.3f  sd %6.3f  min %7.3f  max %7.3f ms\n",
           label, n, mean, med, sd, mn, mx);
    free(c);
}
static double meanof(const double* v, int n) {
    if (n <= 0) return 0;
    double s = 0;
    for (int i = 0; i < n; i++) s += v[i];
    return s / n;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("usage: %s A.pkt A-dir B.pkt B-dir prompt.ids [n_gen]\n", argv[0]);
        return 1;
    }
    const int n_gen = argc > 6 ? atoi(argv[6]) : 24;
    const int warmup = getenv("PLOW_WARMUP") ? atoi(getenv("PLOW_WARMUP")) : 4;
    g_nc_skip_ctrzero = getenv("PLOW_NEGCTRL_SKIP_CTRZERO") != NULL;
    g_nc_stale_pos = getenv("PLOW_NEGCTRL_STALE_POS") != NULL;
    if (g_nc_skip_ctrzero) printf("*** NEGATIVE CONTROL: counters NOT zeroed on switch steps ***\n");
    if (g_nc_stale_pos) printf("*** NEGATIVE CONTROL: stale (other model's) position on switch steps ***\n");

    int dev = 0;
    cudaDeviceProp dp;
    CK(cudaGetDeviceProperties(&dp, dev));
    const int grid = plow_sm120_grid(dev);
    printf("device: %s cc %d.%d SMs=%d | grid=%d smem=%zu B | scheduler: %s\n", dp.name,
           dp.major, dp.minor, dp.multiProcessorCount, grid, plow_sm120_smem(),
           plow_sm120_sched() ? "GLOBAL QUEUE" : "STATIC per-block stream");
    size_t mfree0 = 0, mtot = 0;
    CK(cudaMemGetInfo(&mfree0, &mtot));
    printf("HBM before load: %.3f / %.3f GiB free\n", mfree0 / 1073741824.0, mtot / 1073741824.0);

    FILE* pf = fopen(argv[5], "rb");
    if (!pf) { printf("no %s\n", argv[5]); return 1; }
    fseek(pf, 0, SEEK_END);
    long pn = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    const int n_prompt = (int)(pn / 4);
    int32_t* prompt = (int32_t*)malloc((size_t)pn);
    if (fread(prompt, 1, (size_t)pn, pf) != (size_t)pn) return 1;
    fclose(pf);
    printf("prompt: %d tokens\n\n", n_prompt);

    Model MA, MB;
    memset(&MA, 0, sizeof(MA));
    memset(&MB, 0, sizeof(MB));
    model_load(&MA, "A", argv[1], argv[2], grid);
    model_load(&MB, "B", argv[3], argv[4], grid);
    Model* M[2] = {&MA, &MB};

    size_t mfree1 = 0;
    CK(cudaMemGetInfo(&mfree1, &mtot));
    printf("\nRESIDENT FOOTPRINT (both models live simultaneously)\n");
    printf("  A weights %.3f  KV %.3f  act %.3f  other %.3f GiB\n",
           MA.bytes_weights / 1073741824.0, MA.bytes_kv / 1073741824.0,
           MA.bytes_act / 1073741824.0, MA.bytes_other / 1073741824.0);
    printf("  B weights %.3f  KV %.3f  act %.3f  other %.3f GiB\n",
           MB.bytes_weights / 1073741824.0, MB.bytes_kv / 1073741824.0,
           MB.bytes_act / 1073741824.0, MB.bytes_other / 1073741824.0);
    printf("  HBM consumed by load: %.3f GiB   free after: %.3f / %.3f GiB\n",
           (mfree0 - mfree1) / 1073741824.0, mfree1 / 1073741824.0, mtot / 1073741824.0);
    printf("  shared-scratch upper bound (if act arenas were unified): saves %.3f GiB\n\n",
           (MA.bytes_act < MB.bytes_act ? MA.bytes_act : MB.bytes_act) / 1073741824.0);

    for (int k = 0; k < 2; k++)
        if (n_prompt + n_gen + 2 > M[k]->max_ctx) {
            printf("prompt %d + n_gen %d exceeds [%s] max_ctx %d\n", n_prompt, n_gen,
                   M[k]->tag, M[k]->max_ctx);
            return 1;
        }
    if (n_gen <= warmup + 2) { printf("n_gen must exceed warmup+2\n"); return 1; }
    const int ntimed = n_gen - warmup;

    /* ---------- PHASE 1: solo reference per model ---------- */
    int* refA = (int*)calloc(n_gen, sizeof(int));
    int* refB = (int*)calloc(n_gen, sizeof(int));
    uint16_t* l0A = (uint16_t*)malloc((size_t)MA.vocab * 2);
    uint16_t* l0B = (uint16_t*)malloc((size_t)MB.vocab * 2);
    double* soloA = (double*)calloc(ntimed, sizeof(double));
    double* soloB = (double*)calloc(ntimed, sizeof(double));

    printf("=== PHASE 1: solo reference ===\n");
    run_solo(&MA, prompt, n_prompt, n_gen, refA, l0A, soloA, warmup);
    printf("[A] tokens:");
    for (int i = 0; i < n_gen; i++) printf(" %d", refA[i]);
    printf("\n");
    top5(&MA, l0A);
    run_solo(&MB, prompt, n_prompt, n_gen, refB, l0B, soloB, warmup);
    printf("[B] tokens:");
    for (int i = 0; i < n_gen; i++) printf(" %d", refB[i]);
    printf("\n");
    top5(&MB, l0B);
    if (getenv("PLOW_DUMP_LOGITS_A")) {
        FILE* f = fopen(getenv("PLOW_DUMP_LOGITS_A"), "wb");
        if (f) { fwrite(l0A, 2, MA.vocab, f); fclose(f); }
    }
    if (getenv("PLOW_DUMP_LOGITS_B")) {
        FILE* f = fopen(getenv("PLOW_DUMP_LOGITS_B"), "wb");
        if (f) { fwrite(l0B, 2, MB.vocab, f); fclose(f); }
    }

    /* ---------- PHASE 2: interleaved A/B/A/B ---------- */
    printf("\n=== PHASE 2: interleaved A/B/A/B (both resident, switching every step) ===\n");
    model_reset_seq(&MA);
    model_reset_seq(&MB);
    /* Prompt both, still alternating, so even prompt consumption is switched. */
    for (int i = 0; i < n_prompt; i++) {
        model_step(&MA, prompt[i], 1, MB.pos);
        model_step(&MB, prompt[i], 1, MA.pos);
    }
    uint16_t* i0A = (uint16_t*)malloc((size_t)MA.vocab * 2);
    uint16_t* i0B = (uint16_t*)malloc((size_t)MB.vocab * 2);
    dump_logits(&MA, i0A);
    dump_logits(&MB, i0B);

    int* intA = (int*)calloc(n_gen, sizeof(int));
    int* intB = (int*)calloc(n_gen, sizeof(int));
    double* swA = (double*)calloc(ntimed, sizeof(double));
    double* swB = (double*)calloc(ntimed, sizeof(double));
    /* HOST-SIDE REBIND COST: what the "switch" costs before any device work — selecting
     * the other model's PlowProgram. Accumulated over every switch and reported, because
     * the design claim is that this is the whole host-side cost. */
    double rebind_ns = 0;
    int n_rebind = 0;
    PlowProgram* cur = NULL;
    for (int s = 0; s < n_gen; s++) {
        intA[s] = MA.best;
        intB[s] = MB.best;
        const double r0 = now();
        cur = &MA.pr;                 /* THE SWITCH: one pointer, nothing else */
        const double r1 = now();
        rebind_ns += (r1 - r0) * 1e9;
        n_rebind++;
        const double a0 = now();
        model_step(&MA, -1, 1, MB.pos);
        const double a1 = now();
        cur = &MB.pr;
        const double b0 = now();
        model_step(&MB, -1, 1, MA.pos);
        const double b1 = now();
        rebind_ns += (b0 - a1) * 1e9;
        n_rebind++;
        if (s >= warmup) { swA[s - warmup] = (a1 - a0) * 1e3; swB[s - warmup] = (b1 - b0) * 1e3; }
    }
    (void)cur;

    printf("[A] tokens:");
    for (int i = 0; i < n_gen; i++) printf(" %d", intA[i]);
    printf("\n[B] tokens:");
    for (int i = 0; i < n_gen; i++) printf(" %d", intB[i]);
    printf("\n");

    /* ---------- PHASE 3: solo AGAIN, to quantify the drift confound ----------
     * Phase 1 and phase 2 are separated in time, so any difference between them could be
     * GPU clock/thermal drift rather than switching. Re-running solo AFTER the interleaved
     * phase bounds that: solo1 vs solo3 is drift with NO switching involved, and it is the
     * yardstick the switch delta has to beat to mean anything. Without this the switch cost
     * is an unfalsifiable number. */
    printf("\n=== PHASE 3: solo again (drift control, no switching) ===\n");
    int* re3A = (int*)calloc(n_gen, sizeof(int));
    int* re3B = (int*)calloc(n_gen, sizeof(int));
    uint16_t* l3A = (uint16_t*)malloc((size_t)MA.vocab * 2);
    uint16_t* l3B = (uint16_t*)malloc((size_t)MB.vocab * 2);
    double* solo3A = (double*)calloc(ntimed, sizeof(double));
    double* solo3B = (double*)calloc(ntimed, sizeof(double));
    run_solo(&MA, prompt, n_prompt, n_gen, re3A, l3A, solo3A, warmup);
    run_solo(&MB, prompt, n_prompt, n_gen, re3B, l3B, solo3B, warmup);
    int rp3 = 1;
    for (int i = 0; i < n_gen; i++) if (re3A[i] != refA[i] || re3B[i] != refB[i]) rp3 = 0;
    printf("  solo-run repeatability (phase 1 vs phase 3 tokens): %s\n",
           rp3 ? "IDENTICAL" : "*** DIFFERS — the harness is not deterministic ***");

    /* ---------- CORRECTNESS ---------- */
    int mmA = 0, mmB = 0;
    for (int i = 0; i < n_gen; i++) {
        if (intA[i] != refA[i]) mmA++;
        if (intB[i] != refB[i]) mmB++;
    }
    const int lA = memcmp(i0A, l0A, (size_t)MA.vocab * 2);
    const int lB = memcmp(i0B, l0B, (size_t)MB.vocab * 2);
    /* Report the worst logit delta too — memcmp says "differs", not "how much". */
    double dA = 0, dB = 0;
    for (uint32_t v = 0; v < MA.vocab; v++) {
        const double d = fabs((double)b2f(i0A[v]) - (double)b2f(l0A[v]));
        if (d > dA) dA = d;
    }
    for (uint32_t v = 0; v < MB.vocab; v++) {
        const double d = fabs((double)b2f(i0B[v]) - (double)b2f(l0B[v]));
        if (d > dB) dB = d;
    }
    printf("\n=== CORRECTNESS UNDER SWITCHING ===\n");
    printf("  [A] token mismatches vs solo : %d / %d   %s\n", mmA, n_gen, mmA ? "*** FAIL ***" : "PASS");
    printf("  [B] token mismatches vs solo : %d / %d   %s\n", mmB, n_gen, mmB ? "*** FAIL ***" : "PASS");
    printf("  [A] step-0 logits row        : %s (max |delta| %.6g)\n",
           lA ? "*** DIFFERS ***" : "BIT-IDENTICAL", dA);
    printf("  [B] step-0 logits row        : %s (max |delta| %.6g)\n",
           lB ? "*** DIFFERS ***" : "BIT-IDENTICAL", dB);
    const int pass = !mmA && !mmB && !lA && !lB;
    printf("  VERDICT: %s\n", pass ? "PASS — interleaving is bit-exact per model"
                                   : "*** FAIL — switching perturbed at least one model ***");

    /* ---------- SWITCH COST ---------- */
    printf("\n=== SWITCH COST (differential; same ctx, same allocation) ===\n");
    stats("A solo phase1 (no switching)", soloA, ntimed);
    stats("A interleaved (switched)", swA, ntimed);
    stats("A solo phase3 (drift ctrl)", solo3A, ntimed);
    stats("B solo phase1 (no switching)", soloB, ntimed);
    stats("B interleaved (switched)", swB, ntimed);
    stats("B solo phase3 (drift ctrl)", solo3B, ntimed);
    const double ma = meanof(soloA, ntimed), msa = meanof(swA, ntimed), m3a = meanof(solo3A, ntimed);
    const double mb = meanof(soloB, ntimed), msb = meanof(swB, ntimed), m3b = meanof(solo3B, ntimed);
    /* Compare the switched mean against the MEAN OF THE TWO SOLO RUNS that bracket it in
     * time, so first-order drift cancels; report the solo1-vs-solo3 spread as the noise
     * floor the switch delta must clear. */
    const double ba = 0.5 * (ma + m3a), bb = 0.5 * (mb + m3b);
    printf("  drift floor A (solo1 vs solo3): %+.3f ms   |  B: %+.3f ms\n", m3a - ma, m3b - mb);
    printf("  switch cost A: %+.3f ms vs bracketed solo %.3f ms  (%+.2f%%)  = %+.1f us\n",
           msa - ba, ba, 100.0 * (msa - ba) / ba, (msa - ba) * 1e3);
    printf("  switch cost B: %+.3f ms vs bracketed solo %.3f ms  (%+.2f%%)  = %+.1f us\n",
           msb - bb, bb, 100.0 * (msb - bb) / bb, (msb - bb) * 1e3);
    printf("  host-side rebind: %.1f ns mean over %d switches (pointer assignment only)\n",
           rebind_ns / n_rebind, n_rebind);
    printf("PLOW_SWITCH solo_a=%.4f sw_a=%.4f solo3_a=%.4f solo_b=%.4f sw_b=%.4f solo3_b=%.4f "
           "d_a_us=%.1f d_b_us=%.1f drift_a_us=%.1f drift_b_us=%.1f "
           "mmA=%d mmB=%d logitA=%d logitB=%d rep3=%d pass=%d\n",
           ma, msa, m3a, mb, msb, m3b, (msa - ba) * 1e3, (msb - bb) * 1e3,
           (m3a - ma) * 1e3, (m3b - mb) * 1e3, mmA, mmB, lA != 0, lB != 0, rp3, pass);
    return pass ? 0 : 2;
}
