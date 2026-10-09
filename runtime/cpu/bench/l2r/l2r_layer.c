/* l2r_layer.c <refdir> <steps>: one real Gemma-4 text decoder layer (BF16 weights) as a weight-stationary stage.
 *
 * Input: a ref_layer.py dump (BF16 weights, the decode step's inputs, the layer's KV cache, FP32 boundaries).
 * One worker per cpu of L2R_CPUS (default the 90 isolated inference cores). Every GEMV is output-row
 * partitioned; each worker keeps its row slices of all matrices in one node-local, THP-backed arena that it
 * touches first. KV rows are split by position across workers (node-local); attention is split-K with per-head
 * (max, sum, partial out) partials combined by element slices. Norms, RoPE and residuals are recomputed by every
 * worker from the shared vectors, so a decode step has 8 barriers:
 *   qkv | attention | combine | o | gate,up,gelu | down | ple gate | ple proj
 * Numerics as ref_layer.py's BF16 mode: BF16 weights, GEMV inputs rounded to BF16, BF16 KV, FP32 accumulation.
 * GEMV kernel: L2R_GEMV=avx (VDPBF16PS, default) or amx (TDPBF16PS, weights packed in 16-row VNNI tiles).
 * L2R_CTX_REPEAT=n tiles the dumped KV n times (long-context timing; numerics are then not checked).
 * Every step recomputes the same token (the new KV row overwrites itself), so weights stay hot and outputs repeat.
 * L2R_RESIDENT_KIB=n (AVX only): keep n KiB of each worker's slice L2-resident; the FFN rows beyond it stream from L3
 * with PREFETCHNTA (the E4B split: one E4B layer is ~2 MiB per worker).
 * L2R_LOCK=1 (with L2R_RESIDENT_KIB): lock the resident part in L2 through /dev/pseudo_lock (level 2, owner = the worker's
 *   cpu); the rest streams normally, confined by CAT to the unlocked ways. Prints the minimum held fraction.
 * L2R_BCAST=direct|rep|repcld|repnt|fid: how the output vectors reach every worker. direct: read the shared vector (homed
 *   wherever main touched it). rep: each producer copies its slice (64-byte aligned segment) into one replica per
 *   SNC node, homed on that node; readers gather from their own node's replica. repcld: rep + CLDEMOTE of the
 *   written lines, so readers hit their node's L3 instead of snooping the producer's core. repnt: rep with
 *   non-temporal full-line stores (readers fetch from their node's memory, no remote snoop). fid: repnt where each
 *   line carries the step's tag at both ends, readers poll until it matches, and only the barrier before the
 *   attention combine remains (7 of 8 barriers removed).
 * L2R_BARRIER=diss|hier: dissemination flags, or a per-node arrival counter -> node leaders -> per-node gate (all
 *   lines homed on their node).
 * P4 KV policy knobs:
 *   L2R_KV_COPIES=r: r copies of every worker's KV slice, step st reads copy st % r (r x KV > L3 = path A, KV from
 *     DRAM every step; r = 1 = path B, KV left wherever the previous step put it).
 *   L2R_KV_TILE_KIB=t: attention in tiles of t KiB of K+V per KV head (online softmax across tiles); 0 = one tile.
 *   L2R_KV_PFD=d, L2R_KV_PFH=t0|t1|t2|nta: software prefetch of the K / V row d rows ahead (path C, in-loop).
 *   L2R_KV_EARLY_KIB=n: at the start of the step, before the qkv GEMV, prefetch (L2R_KV_PFH) the first n KiB of this
 *     step's K and V slice per KV head (path C, staged ahead; its cost is inside the step).
 *   L2R_ATTN_ONLY=1: skip every GEMV (timing of attention, combine and sync alone; numerics not checked).
 * L2R_PERF_CTL=<fifo>: write enable / disable to a `perf stat -D -1 --control fifo:<fifo>` around the timed steps.
 * Batch (P5): L2R_BATCH=b (1..16) decode tokens per step, one per sequence; L2R_ROWS=dir,dir,... gives each row its own
 *   dump of the same layer and context length (row r uses dir r % n; default <refdir>): its own hidden state,
 *   per-layer input, RoPE, KV cache and FP32 reference. Weights come from <refdir>. Every GEMV is batch b (AMX: A tile
 *   rows = b), norms / RoPE / residuals per row, broadcasts carry b rows. Attention is split by row group: row r gets
 *   workers [NW r / b, NW (r + 1) / b), which split that row's KV positions and combine only each other's partials
 *   (b = 1: the group is every worker, the P2-P4 scheme). b > 1 needs
 *   L2R_BCAST=direct|rep|repcld|repnt and no L2R_NOBCAST or streamed rows.
 * 12B / 26B / 31B dumps: no per-layer input (ple = 0: phases 6-7 are empty and the step ends after down), full layers
 *   with attention_k_eq_v (no v_proj: V is the K projection, normed without scale). A tensor-parallel socket slice
 *   (ref_layer.py REF_TP, meta tp > 1) holds whole heads and a block of FFN rows; o and down are the slice's partial sums,
 *   and the other ranks' share (o_rest / down_rest of each row's dump) is added where the cross-socket all-reduce lands.
 * MoE (26B-A4B; ref_layer.py tp = moe:G): stage "head" adds the router rows (input rms(h1) * scale / sqrt(hidden)) to
 *   the gate / up phase and combines h2 = h1 + rms(rms(down, pf1) + rms(moe_rest, pf2), pf), moe_rest being the expert
 *   sockets' weighted sum. Stage "experts" is one expert socket: every local expert is striped over all workers (gate /
 *   up rows and down rows by output rows); phase 0 runs gate / up + gelu for the tokens that picked each expert and
 *   all-gathers the activations, phase 1 runs the down rows, weights them and all-gathers the socket's partial sum.
 * L2R_NOBCAST=1: after step 0 every worker reads private snapshots of the shared vectors (identical values, since
 * the step repeats), which removes the activation all-gather from the timing; writes are unchanged.
 * Prints one JSON line: per-boundary error vs the FP32 reference (first and last step), step p50/p95/p99,
 * per-phase critical path (max over workers of phase compute) and barrier time. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <x86intrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAXW 192
#define NPH 8
typedef uint16_t bf16;
typedef struct { _Alignas(64) volatile uint64_t v; char pad[56]; } line_t;

static const char* DIR;
static int NW, STEPS, AMX, REP, NOBCAST;
static size_t RESKIB;
static int PLFD = -1;
#define MAXB 16
static int KVR = 1, TILE_KIB, PFD, PFH, NB = 1, ATTN_ONLY;
static const char* ROWDIR[MAXB];
static size_t EARLY_KIB;
static int PERF_FD = -1; /* L2R_PERF_CTL: perf stat --control fifo, enabled for the timed steps only */
static int BCAST, BARR;               /* L2R_BCAST: 0 direct, 1 rep, 2 repcld, 3 repnt, 4 fid; L2R_BARRIER: 0 diss, 1 hier */
static int NNODE, nodeof[MAXW], node_first[8], node_count[8];
/* One broadcast vector of `rows` rows: producer w owns elements [off, off + len) of every row; in the per-node replicas
 * its segment starts at byte seg and holds its rows rb bytes apart (64-byte aligned, so no line has two writers).
 * rep[n] is homed on node n. */
typedef struct { int esz, rows, off[MAXW], len[MAXW], row[MAXW]; size_t seg[MAXW], rb[MAXW], bytes; uint8_t* rep[8]; } bc_t;
static bc_t bc_q, bc_o, bc_down, bc_attn, bc_act, bc_pact, bc_po, bc_pml;
static line_t *b_arrive[8], *b_done[8], *b_gate[8];
/* Flag-in-data vectors (L2R_BCAST=fid): every 64-byte line is [tag u32 | 56 payload bytes | tag u32], written with
 * one non-temporal full-line store into each node's replica (parity-buffered by step); a reader polls its node's
 * lines until both tags equal the step's tag, so the gather itself is the synchronisation and no barrier is needed. */
typedef struct { int esz, epl, off[MAXW], len[MAXW], line0[MAXW], lines; uint8_t* rep[8][2]; } fv_t;
static fv_t fv_q, fv_k, fv_v, fv_attn, fv_o, fv_act, fv_down, fv_pact, fv_pp;
static uint8_t* line_nodes; /* [NB][NH*HD/16]: bit k = a combine worker of the row's group on node k reads this line */
/* attention row groups: row r is attended by workers [grp0(r), grp0(r + 1)) */
static inline int grp0(int r) { return (int)((long)NW * r / NB); }
static inline int row_of(int w) { int r = 0; while (grp0(r + 1) <= w) r++; return r; }
static inline int gsplit(int n, int w, int r) { const int g0 = grp0(r), gs = grp0(r + 1) - g0; return (int)((long)n * (w - g0) / gs); }
struct pl_lock_req { uint64_t addr, len; int32_t cpu; uint32_t level, id, pad; };
struct pl_measure { uint32_t id, pad; uint64_t lines, l1_l2, l3, dram, p50, cal_l2, cal_l3, cal_dram; };
#define PL_IOC_LOCK _IOWR('P', 11, struct pl_lock_req)
#define PL_IOC_MEASURE _IOWR('P', 13, struct pl_measure)
static int cpus[MAXW];
static double tsc_ghz;

/* ---- model ---- */
static int H, NH, KVH, HD, I, PLE, WIN, CL0, CL; /* CL0: dumped cache rows, CL: rows attended before the new token */
static int KVEQ, TP, STAGE; /* STAGE: 0 layer, 1 MoE head, 2 MoE experts */
static float *moe_rest, *w_pf1, *w_pf2, *w_rsc, *rs; /* head: [NB][H] experts' sum, norms, router scale, scores */
static int NE, EI, TOPK, NP, *pb, *pel, *el_p0, *el_n; static float *pw, *h1x, *part, *chk_xn3, *chk_eact;
static bf16 *Wgu, *Wdn, *eact_b; static bc_t bc_eact, bc_part;
static float *o_rest, *down_rest; /* [NB][H], TP > 1 */
static float EPS, SCALAR;
static float *x_in, *pli, *cosv, *sinv; /* [NB][H], [NB][PLE], [NB][HD], [NB][HD] */
static bf16 *kc0[MAXB], *vc0[MAXB];
#define NMAT 10
static const char* WN[NMAT] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "per_layer_input_gate", "per_layer_projection",
                               "router.proj"};
static bf16* Wfull[NMAT];
static int Wn[NMAT], Wk[NMAT];
static float *w_in, *w_pa, *w_pf, *w_pff, *w_pn, *w_qn, *w_kn;

/* ---- shared vectors, all [NB][n] ---- */
static float *q, *k, *v, *o, *gate, *up, *down, *pg, *pp, *outv; /* outv [2][NB][H]: step 0, last step */
static bf16 *attn_b, *act_b, *pact_b;
static float *pm, *pl, *po; /* partials [NW][NB][NH], [NW][NB][NH], [NW][NB][NH][HD] */
static float *chk_qn, *chk_h1, *chk_xn2, *chk_h2, *chk_attn, *chk_act, *chk_pact;

/* ---- per worker ---- */
typedef struct {
    int id, node;
    int r0[NMAT], r1[NMAT];
    int nres[NMAT];          /* rows of the slice kept L2-resident; the rest stream with PREFETCHNTA */
    bf16* ws[NMAT];
    bf16* w[NMAT];
    int p0, p1;           /* KV positions [p0, p1) of 0..CL-1; the last worker also owns the new row */
    bf16 *kv;             /* [KVR][NB][K, V][KVH][np][HD] */
    size_t kvblk;         /* elements of one [KVH][np][HD] block */
    int e0, e1;           /* attention-combine element slice of NH*HD (of the worker's row) */
    int row, g0, g1;      /* attention row and its worker group [g0, g1) */
    uint32_t lock_id; double held0, held1;
    uint64_t ph[NPH], wt[NPH], pro[NPH]; /* pro: the phase's redundant prologue (reads the shared vectors) */
    uint64_t* steps_t;
} worker_t;
static worker_t WK[MAXW];
static line_t dis[MAXW][8];
static pthread_barrier_t pbar;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double calib(void) { double t0 = now_s(); uint64_t c0 = __rdtsc(); while (now_s() - t0 < 0.2) ; return (__rdtsc() - c0) / (now_s() - t0) / 1e9; }

static inline float bf2f(bf16 b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline bf16 f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (bf16)(u >> 16); }

static void* load_in(const char* dir, const char* name, size_t* n_out, char* dt_out) {
    char path[512], line[512];
    snprintf(path, sizeof path, "%s/manifest.txt", dir);
    FILE* m = fopen(path, "r");
    if (!m) { perror(path); exit(1); }
    while (fgets(line, sizeof line, m)) {
        char nm[256], dt[8]; int off;
        if (sscanf(line, "%255s %7s %n", nm, dt, &off) < 2 || strcmp(nm, name)) continue;
        size_t n = 1; long d; char* p = line + off;
        while (sscanf(p, "%ld%n", &d, &off) == 1) { n *= (size_t)d; p += off; }
        fclose(m);
        const size_t es = strcmp(dt, "bf16") ? 4 : 2;
        snprintf(path, sizeof path, "%s/%s.%s", dir, name, dt);
        FILE* f = fopen(path, "rb");
        if (!f) { perror(path); exit(1); }
        void* b = aligned_alloc(64, (n * es + 63) / 64 * 64);
        if (fread(b, es, n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
        fclose(f);
        if (n_out) *n_out = n;
        if (dt_out) *dt_out = es == 2 ? 'b' : 'f';
        return b;
    }
    fprintf(stderr, "%s not in manifest of %s\n", name, dir); exit(1);
}
static void* load(const char* name, size_t* n_out, char* dt_out) { return load_in(DIR, name, n_out, dt_out); }
static int in_manifest(const char* dir, const char* name) {
    char path[512], line[512], nm[256];
    snprintf(path, sizeof path, "%s/manifest.txt", dir);
    FILE* m = fopen(path, "r");
    if (!m) return 0;
    int hit = 0;
    while (!hit && fgets(line, sizeof line, m)) hit = sscanf(line, "%255s", nm) == 1 && !strcmp(nm, name);
    fclose(m);
    return hit;
}

static float* loadbf_as_f(const char* name) {
    size_t n; char dt; void* p = load(name, &n, &dt);
    if (dt == 'f') return p;
    float* f = malloc(n * 4);
    for (size_t i = 0; i < n; i++) f[i] = bf2f(((bf16*)p)[i]);
    free(p);
    return f;
}

static int meta_int(const char* js, const char* key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\": ", key);
    const char* p = strstr(js, pat);
    if (!p) { fprintf(stderr, "meta: %s\n", key); exit(1); }
    p += strlen(pat);
    return strncmp(p, "null", 4) ? atoi(p) : 0;
}

static double meta_f(const char* js, const char* key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\": ", key);
    const char* p = strstr(js, pat);
    return p ? atof(p + strlen(pat)) : 0;
}

static void split(int n, int unit, int w, int* a, int* b);

static void* node_alloc(size_t bytes, int node) {
    bytes = (bytes + 4095) / 4096 * 4096;
    void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    unsigned long mask = 1ul << node;
    if (syscall(SYS_mbind, p, bytes, 2 /* MPOL_BIND */, &mask, 64, 0)) { perror("mbind"); exit(1); }
    memset(p, 0, bytes);
    return p;
}

static void bc_init(bc_t* b, int esz, int n, int unit, int whole_blocks, int rows) {
    b->esz = esz; b->rows = rows;
    size_t at = 0;
    for (int w = 0; w < NW; w++) {
        int a, z;
        if (whole_blocks) { a = w * n; z = a + n; } else split(n, unit, w, &a, &z);
        b->off[w] = a; b->len[w] = z - a; b->seg[w] = at; b->row[w] = -1;
        b->rb[w] = ((size_t)(z - a) * esz + 63) / 64 * 64;
        at += b->rb[w] * rows;
    }
    b->bytes = at;
    for (int k = 0; k < NNODE; k++) b->rep[k] = node_alloc(at, k);
}
/* one row of n per producer: worker w owns elements [gsplit(n, w), gsplit(n, w + 1)) of its attention row */
static void bc_init_grp(bc_t* b, int esz, int n) {
    b->esz = esz; b->rows = 1;
    size_t at = 0;
    for (int w = 0; w < NW; w++) {
        const int r = row_of(w), a = gsplit(n, w, r), z = w + 1 == grp0(r + 1) ? n : gsplit(n, w + 1, r);
        b->off[w] = a; b->len[w] = z - a; b->seg[w] = at; b->row[w] = r;
        b->rb[w] = ((size_t)(z - a) * esz + 63) / 64 * 64;
        at += b->rb[w];
    }
    b->bytes = at;
    for (int k = 0; k < NNODE; k++) b->rep[k] = node_alloc(at, k);
}

/* producer w copies its elements of every row (src points at element off[w] of row 0; rows ld elements apart) into
 * every node's replica */
static inline void bc_publish(const bc_t* b, int w, const void* src, size_t ld) {
    const size_t n = (size_t)b->len[w] * b->esz;
    for (int r = 0; r < b->rows; r++) {
        const uint8_t* s = (const uint8_t*)src + (size_t)r * ld * b->esz;
        const size_t o = b->seg[w] + (size_t)r * b->rb[w];
        if (BCAST >= 3) { /* whole 64-byte lines (the segment is padded), streamed past the caches to the replica's node */
            for (size_t i = 0; i < n; i += 64) {
                const __mmask64 k = n - i >= 64 ? ~0ull : (1ull << (n - i)) - 1;
                const __m512i v = _mm512_maskz_loadu_epi8(k, s + i);
                for (int nd = 0; nd < NNODE; nd++) _mm512_stream_si512((__m512i*)(b->rep[nd] + o + i), v);
            }
            continue;
        }
        for (int nd = 0; nd < NNODE; nd++) {
            uint8_t* d = b->rep[nd] + o;
            memcpy(d, s, n);
            if (BCAST == 2) for (size_t i = 0; i < n; i += 64) _cldemote(d + i);
        }
    }
    if (BCAST >= 3) _mm_sfence();
}

/* reader on `node` gathers the whole vector into dst (element 0 of row 0 at dst, rows ld elements apart) */
static inline void bc_gather(const bc_t* b, int node, void* dst, size_t ld) {
    /* all lines requested up front: the copy below then finds them in flight instead of missing one segment at a time */
    for (size_t i = 0; i < b->bytes; i += 64) _mm_prefetch((const char*)b->rep[node] + i, _MM_HINT_T0);
    for (int w = 0; w < NW; w++)
        for (int r = 0; r < b->rows; r++)
            memcpy((uint8_t*)dst + ((size_t)(b->row[w] < 0 ? r : b->row[w]) * ld + b->off[w]) * b->esz,
                   b->rep[node] + b->seg[w] + (size_t)r * b->rb[w], (size_t)b->len[w] * b->esz);
}
/* only row r of a vector replicated per row (bc_init rows > 1), into dst (element 0 of that row at dst) */
static inline void bc_gather_row(const bc_t* b, int node, void* dst, int r) {
    for (int w = 0; w < NW; w++)
        for (size_t i = 0; i < (size_t)b->len[w] * b->esz; i += 64) _mm_prefetch((const char*)b->rep[node] + b->seg[w] + (size_t)r * b->rb[w] + i, _MM_HINT_T0);
    for (int w = 0; w < NW; w++)
        memcpy((uint8_t*)dst + (size_t)b->off[w] * b->esz, b->rep[node] + b->seg[w] + (size_t)r * b->rb[w], (size_t)b->len[w] * b->esz);
}

static void fv_init(fv_t* f, int esz, int n, int unit) {
    f->esz = esz; f->epl = 56 / esz;
    int at = 0;
    for (int w = 0; w < NW; w++) {
        int a, z; split(n, unit, w, &a, &z);
        f->off[w] = a; f->len[w] = z - a; f->line0[w] = at;
        at += (z - a + f->epl - 1) / f->epl;
    }
    f->lines = at;
    for (int k = 0; k < NNODE; k++)
        for (int par = 0; par < 2; par++) f->rep[k][par] = node_alloc((size_t)at * 64, k);
}

static inline void fv_publish(const fv_t* f, int w, const void* src, uint32_t tag, int par) {
    const uint8_t* s8 = src;
    const int n = f->len[w];
    for (int i = 0, li = f->line0[w]; i < n; i += f->epl, li++) {
        _Alignas(64) uint8_t line[64] = {0};
        const int c = n - i < f->epl ? n - i : f->epl;
        memcpy(line, &tag, 4); memcpy(line + 60, &tag, 4);
        memcpy(line + 4, s8 + (size_t)i * f->esz, (size_t)c * f->esz);
        const __m512i v = _mm512_load_si512(line);
        for (int k = 0; k < NNODE; k++) _mm512_stream_si512((__m512i*)(f->rep[k][par] + (size_t)li * 64), v);
    }
    _mm_sfence();
}

static inline void fv_gather(const fv_t* f, int node, void* dst, uint32_t tag, int par) {
    uint8_t* d8 = dst;
    const uint8_t* base = f->rep[node][par];
    for (int w = 0; w < NW; w++) {
        const int n = f->len[w];
        for (int i = 0, li = f->line0[w]; i < n; i += f->epl, li++) {
            _Alignas(64) uint8_t line[64];
            for (;;) {
                _mm512_store_si512(line, _mm512_load_si512(base + (size_t)li * 64));
                uint32_t t0, t1; memcpy(&t0, line, 4); memcpy(&t1, line + 60, 4);
                if (t0 == tag && t1 == tag) break;
                _mm_pause();
            }
            const int c = n - i < f->epl ? n - i : f->epl;
            memcpy(d8 + (size_t)(f->off[w] + i) * f->esz, line + 4, (size_t)c * f->esz);
        }
    }
}

/* ---- barrier: dissemination, epoch-tagged flags, one line each ---- */
static inline void barrier(int id, uint64_t e) {
    if (BARR) { /* hierarchical: node-local arrival counter -> node leaders exchange -> node-local gate */
        const int n = nodeof[id];
        __atomic_fetch_add(&b_arrive[n]->v, 1, __ATOMIC_ACQ_REL);
        if (id == node_first[n]) {
            while (__atomic_load_n(&b_arrive[n]->v, __ATOMIC_ACQUIRE) < (uint64_t)node_count[n] * e) _mm_pause();
            __atomic_store_n(&b_done[n]->v, e, __ATOMIC_RELEASE);
            for (int m = 0; m < NNODE; m++) while (__atomic_load_n(&b_done[m]->v, __ATOMIC_ACQUIRE) < e) _mm_pause();
            __atomic_store_n(&b_gate[n]->v, e, __ATOMIC_RELEASE);
        } else while (__atomic_load_n(&b_gate[n]->v, __ATOMIC_ACQUIRE) < e) _mm_pause();
        return;
    }
    for (int r = 0, d = 1; d < NW; r++, d <<= 1) {
        dis[(id + d) % NW][r].v = e;
        while (dis[id][r].v < e) _mm_pause();
    }
}

/* ---- GEMV kernels: y[b][r] = sum_k W[r][k] * x[b][k], W rows bf16 [n][K], x bf16 [nb][K], y rows ldy apart ---- */
/* 4 weight rows x Bc batch rows; Bc is a compile-time constant at every call so the accumulators stay in registers */
static inline __attribute__((always_inline)) void avx_blk(const int Bc, const bf16* w0, int K, const bf16* x, float* y, int ldy) {
    const bf16 *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
    __m512 a[4][Bc];
    for (int j = 0; j < 4; j++) for (int b = 0; b < Bc; b++) a[j][b] = _mm512_setzero_ps();
    for (int c = 0; c < K; c += 32) {
        const __m512bh v0 = (__m512bh)_mm512_loadu_si512(w0 + c), v1 = (__m512bh)_mm512_loadu_si512(w1 + c);
        const __m512bh v2 = (__m512bh)_mm512_loadu_si512(w2 + c), v3 = (__m512bh)_mm512_loadu_si512(w3 + c);
        for (int b = 0; b < Bc; b++) {
            const __m512bh xv = (__m512bh)_mm512_loadu_si512(x + (size_t)b * K + c);
            a[0][b] = _mm512_dpbf16_ps(a[0][b], v0, xv); a[1][b] = _mm512_dpbf16_ps(a[1][b], v1, xv);
            a[2][b] = _mm512_dpbf16_ps(a[2][b], v2, xv); a[3][b] = _mm512_dpbf16_ps(a[3][b], v3, xv);
        }
    }
    for (int b = 0; b < Bc; b++) for (int j = 0; j < 4; j++) y[(size_t)b * ldy + j] = _mm512_reduce_add_ps(a[j][b]);
}
static void gemv_avx(const bf16* W, int n, int K, const bf16* x, int nb, float* y, int ldy) {
    int r = 0;
    for (; r + 4 <= n; r += 4)
        for (int b0 = 0; b0 < nb; b0 += 4) {
            const bf16* xb = x + (size_t)b0 * K; float* yb = y + (size_t)b0 * ldy + r;
            switch (nb - b0 < 4 ? nb - b0 : 4) {
            case 1: avx_blk(1, W + (size_t)r * K, K, xb, yb, ldy); break;
            case 2: avx_blk(2, W + (size_t)r * K, K, xb, yb, ldy); break;
            case 3: avx_blk(3, W + (size_t)r * K, K, xb, yb, ldy); break;
            default: avx_blk(4, W + (size_t)r * K, K, xb, yb, ldy);
            }
        }
    for (; r < n; r++)
        for (int b = 0; b < nb; b++) {
            const bf16* w0 = W + (size_t)r * K;
            __m512 a0 = _mm512_setzero_ps();
            for (int c = 0; c < K; c += 32)
                a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_loadu_si512(w0 + c), (__m512bh)_mm512_loadu_si512(x + (size_t)b * K + c));
            y[(size_t)b * ldy + r] = _mm512_reduce_add_ps(a0);
        }
}

/* AMX: rows packed per 16-row group g, per 32-wide K chunk c, as one 1 KiB B tile [16 k-pairs][16 rows][2].
 * A = the nb batch rows' x chunk (tile rows = nb, set in the tile config); C[b][0..15] = the group's 16 outputs of
 * batch row b. n must be a multiple of 16. */
typedef struct { uint8_t palette, start; uint8_t res[14]; uint16_t colsb[16]; uint8_t rows[16]; } tilecfg_t;
static void pack_amx(const bf16* src, int n, int K, bf16* dst) {
    for (int g = 0; g < n / 16; g++)
        for (int c = 0; c < K / 32; c++) {
            bf16* t = dst + ((size_t)g * (K / 32) + c) * 512;
            for (int kp = 0; kp < 16; kp++)
                for (int r = 0; r < 16; r++)
                    for (int e = 0; e < 2; e++)
                        t[kp * 32 + r * 2 + e] = src[(size_t)(g * 16 + r) * K + c * 32 + kp * 2 + e];
        }
}
static void gemv_amx(const bf16* Wp, int n, int K, const bf16* x, int nb, float* y, int ldy) {
    _Alignas(64) float cbuf[16 * 16];
    const int nc = K / 32;
    const size_t xs = (size_t)K * 2;
#define CST(t, gg) do { _tile_stored(t, cbuf, 64); for (int b_ = 0; b_ < nb; b_++) memcpy(y + (size_t)b_ * ldy + (gg) * 16, cbuf + b_ * 16, 64); } while (0)
    int g = 0;
    for (; g + 4 <= n / 16; g += 4) {
        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
        const bf16* b = Wp + (size_t)g * nc * 512;
        for (int c = 0; c < nc; c++) {
            _tile_loadd(4, x + c * 32, xs);
            _tile_loadd(5, b + (size_t)c * 512, 64); _tile_dpbf16ps(0, 4, 5);
            _tile_loadd(6, b + ((size_t)nc + c) * 512, 64); _tile_dpbf16ps(1, 4, 6);
            _tile_loadd(7, b + ((size_t)2 * nc + c) * 512, 64); _tile_dpbf16ps(2, 4, 7);
            _tile_loadd(5, b + ((size_t)3 * nc + c) * 512, 64); _tile_dpbf16ps(3, 4, 5);
        }
        CST(0, g); CST(1, g + 1); CST(2, g + 2); CST(3, g + 3);
    }
    for (; g < n / 16; g++) {
        _tile_zero(0);
        const bf16* b = Wp + (size_t)g * nc * 512;
        for (int c = 0; c < nc; c++) { _tile_loadd(4, x + c * 32, xs); _tile_loadd(5, b + (size_t)c * 512, 64); _tile_dpbf16ps(0, 4, 5); }
        CST(0, g);
    }
#undef CST
}
/* Streamed rows: same math as gemv_avx; the next 4-row block is prefetched non-temporally (into L1, not L2) while the
 * current one is computed, so streaming rows from L3 does not evict the L2-resident part of the slice. */
static void gemv_nta(const bf16* W, int n, int K, const bf16* x, float* y) {
    if (ATTN_ONLY) return;
    int r = 0;
    for (; r + 4 <= n; r += 4) {
        const bf16 *w0 = W + (size_t)r * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K, *nx = w0 + (size_t)4 * K;
        const int pf = r + 8 <= n;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int c = 0; c < K; c += 32) {
            if (pf) {
                _mm_prefetch((const char*)(nx + c), _MM_HINT_NTA); _mm_prefetch((const char*)(nx + K + c), _MM_HINT_NTA);
                _mm_prefetch((const char*)(nx + 2 * K + c), _MM_HINT_NTA); _mm_prefetch((const char*)(nx + 3 * K + c), _MM_HINT_NTA);
            }
            const __m512bh xv = (__m512bh)_mm512_loadu_si512(x + c);
            a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_loadu_si512(w0 + c), xv);
            a1 = _mm512_dpbf16_ps(a1, (__m512bh)_mm512_loadu_si512(w1 + c), xv);
            a2 = _mm512_dpbf16_ps(a2, (__m512bh)_mm512_loadu_si512(w2 + c), xv);
            a3 = _mm512_dpbf16_ps(a3, (__m512bh)_mm512_loadu_si512(w3 + c), xv);
        }
        y[r] = _mm512_reduce_add_ps(a0); y[r + 1] = _mm512_reduce_add_ps(a1);
        y[r + 2] = _mm512_reduce_add_ps(a2); y[r + 3] = _mm512_reduce_add_ps(a3);
    }
    if (r < n) gemv_avx(W + (size_t)r * K, n - r, K, x, 1, y + r, 0);
}

static void gemv_avx1(const bf16* W, int n, int K, const bf16* x, float* y) { if (n > 0 && !ATTN_ONLY) gemv_avx(W, n, K, x, 1, y, 0); }

static inline void gemv(const bf16* W, int n, int K, const bf16* x, float* y, int ldy) {
    if (n <= 0 || ATTN_ONLY) return;
    if (AMX) gemv_amx(W, n, K, x, NB, y, ldy); else gemv_avx(W, n, K, x, NB, y, ldy);
}

/* ---- small ops ---- */
/* n is a multiple of 16 (hidden, head dim) */
static void rms(const float* x, const float* w, int n, float* y) {
    __m512 a = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) { const __m512 t = _mm512_loadu_ps(x + i); a = _mm512_fmadd_ps(t, t, a); }
    const __m512 r = _mm512_set1_ps(powf(_mm512_reduce_add_ps(a) / n + EPS, -0.5f));
    for (int i = 0; i < n; i += 16) {
        const __m512 t = _mm512_mul_ps(_mm512_loadu_ps(x + i), r);
        _mm512_storeu_ps(y + i, w ? _mm512_mul_ps(t, _mm512_loadu_ps(w + i)) : t);
    }
}
/* round-to-nearest-even, n a multiple of 32 */
static void tobf(const float* x, int n, bf16* y) {
    for (int i = 0; i < n; i += 32)
        _mm512_storeu_si512(y + i, (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(x + i)));
}
static void rope(float* x, int row) {
    const int h = HD / 2;
    const float *cs = cosv + (size_t)row * HD, *sn = sinv + (size_t)row * HD;
    float t[1024];
    for (int i = 0; i < HD; i++) t[i] = i < h ? -x[i + h] : x[i - h];
    for (int i = 0; i < HD; i++) x[i] = x[i] * cs[i] + t[i] * sn[i];
}

/* exp for x <= 0 (softmax): x = n ln2 + r, |r| <= ln2/2, degree-7 Taylor (< 1 ulp), 2^n by scalef */
static inline __m512 exp512(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 y = _mm512_set1_ps(1.f / 5040);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f / 720));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f / 120));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f / 24));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f / 6));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(0.5f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f));
    return _mm512_scalef_ps(y, n);
}

/* out[i] = bf16(gelu_tanh(g[i]) * u[i]), chk[i] = the FP32 value (chk may be NULL); tanh(y) = sign(y) (1 - e) / (1 + e),
 * e = exp(-2 |y|) <= 1 */
static void gelu_mul(const float* g, const float* u, int n, bf16* out, float* chk) {
    const __m512 c0 = _mm512_set1_ps(0.7978845608028654f), c1 = _mm512_set1_ps(0.044715f), one = _mm512_set1_ps(1.f), half = _mm512_set1_ps(0.5f);
    for (int i = 0; i < n; i += 16) {
        const __mmask16 k = n - i >= 16 ? 0xffff : (__mmask16)((1u << (n - i)) - 1);
        const __m512 x = _mm512_maskz_loadu_ps(k, g + i);
        const __m512 y = _mm512_mul_ps(c0, _mm512_fmadd_ps(_mm512_mul_ps(c1, _mm512_mul_ps(x, x)), x, x));
        const __m512 e = exp512(_mm512_mul_ps(_mm512_set1_ps(-2.f), _mm512_abs_ps(y)));
        const __m512 th = _mm512_div_ps(_mm512_sub_ps(one, e), _mm512_add_ps(one, e));
        const __m512 t = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(th),
                             _mm512_and_si512(_mm512_castps_si512(y), _mm512_set1_epi32((int)0x80000000))));
        const __m512 a = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(half, x), _mm512_add_ps(one, t)), _mm512_maskz_loadu_ps(k, u + i));
        _mm256_mask_storeu_epi16(out + i, k, (__m256i)_mm512_cvtneps_pbh(a));
        if (chk) _mm512_mask_storeu_ps(chk + i, k, a);
    }
}

static inline void pf_line(const void* a) {
    switch (PFH) {
    case 0: _mm_prefetch((const char*)a, _MM_HINT_T0); break;
    case 1: _mm_prefetch((const char*)a, _MM_HINT_T1); break;
    case 2: _mm_prefetch((const char*)a, _MM_HINT_T2); break;
    default: _mm_prefetch((const char*)a, _MM_HINT_NTA);
    }
}
static inline void pf_row(const bf16* r) { for (int i = 0; i < HD; i += 32) pf_line(r + i); }

/* Attention partials of one KV head for its G query heads over np rows: scores in sc[p][G]; per head the max m, the
 * sum l of exp(s - m) and the unnormalised output o = sum exp(s - m) * V. Each K / V row is widened once for all G
 * heads; G is a compile-time constant at every call so the accumulators stay in registers. */
static inline __attribute__((always_inline)) void attn_group(const int G, const float* q, const bf16* K, const bf16* V,
                                                              int np, int T, float* sc, float* m, float* l, float* o) {
    for (int j = 0; j < G; j++) m[j] = -INFINITY, l[j] = 0;
    for (int t0 = 0; t0 < np; t0 += T) {
        const int nt = np - t0 < T ? np - t0 : T;
        float mt[G], sf[G];
        for (int j = 0; j < G; j++) mt[j] = m[j];
        for (int p = 0; p < nt; p++) {
            __m512 a[G];
            for (int j = 0; j < G; j++) a[j] = _mm512_setzero_ps();
            const bf16* kr = K + (size_t)(t0 + p) * HD;
            if (PFD && t0 + p + PFD < np) pf_row(kr + (size_t)PFD * HD);
            for (int d = 0; d < HD; d += 16) {
                const __m512 kf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(kr + d))), 16));
                for (int j = 0; j < G; j++) a[j] = _mm512_fmadd_ps(kf, _mm512_loadu_ps(q + j * HD + d), a[j]);
            }
            for (int j = 0; j < G; j++) {
                const float x = _mm512_reduce_add_ps(a[j]);
                sc[(size_t)p * G + j] = x;
                if (x > mt[j]) mt[j] = x;
            }
        }
        /* online softmax: rescale the running sum and output by exp(m_old - m_new) */
        for (int j = 0; j < G; j++) { sf[j] = m[j] == -INFINITY ? 0.f : expf(m[j] - mt[j]); m[j] = mt[j]; l[j] *= sf[j]; }
        {
            float mp[16];
            for (int i = 0; i < 16; i++) mp[i] = m[i % G];
            const __m512 mv = _mm512_loadu_ps(mp);
            __m512 lv = _mm512_setzero_ps();
            const int n = nt * G;
            for (int i = 0; i < n; i += 16) {
                const __mmask16 k = n - i >= 16 ? 0xffff : (__mmask16)((1u << (n - i)) - 1);
                const __m512 w = exp512(_mm512_sub_ps(_mm512_maskz_loadu_ps(k, sc + i), mv));
                _mm512_mask_storeu_ps(sc + i, k, w);
                lv = _mm512_add_ps(lv, _mm512_maskz_mov_ps(k, w));
            }
            float lp[16];
            _mm512_storeu_ps(lp, lv);
            for (int i = 0; i < 16; i++) l[i % G] += lp[i];
        }
        const bf16* Vt = V + (size_t)t0 * HD;
        for (int d = 0; d < HD; d += 16) {
            __m512 acc[G];
            for (int j = 0; j < G; j++) acc[j] = t0 ? _mm512_mul_ps(_mm512_loadu_ps(o + (size_t)j * HD + d), _mm512_set1_ps(sf[j])) : _mm512_setzero_ps();
            for (int p = 0; p < nt; p++) {
                if (PFD && !d && t0 + p + PFD < np) pf_row(Vt + (size_t)(p + PFD) * HD);
                const __m512 vf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(Vt + (size_t)p * HD + d))), 16));
                for (int j = 0; j < G; j++) acc[j] = _mm512_fmadd_ps(_mm512_set1_ps(sc[(size_t)p * G + j]), vf, acc[j]);
            }
            for (int j = 0; j < G; j++) _mm512_storeu_ps(o + (size_t)j * HD + d, acc[j]);
        }
    }
    if (!np) for (int j = 0; j < G; j++) { m[j] = -INFINITY; for (int d = 0; d < HD; d++) o[(size_t)j * HD + d] = 0; }
}

/* ---- the decode step ---- */
static void split(int n, int unit, int w, int* a, int* b) {
    const int u = n / unit;
    *a = (int)((long)u * w / NW) * unit;
    *b = (int)((long)u * (w + 1) / NW) * unit;
}

static void* run(void* arg) {
    worker_t* me = arg;
    const int id = me->id;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus[id], &s); sched_setaffinity(0, sizeof s, &s);
    if (AMX) {
        if (syscall(SYS_arch_prctl, 0x1023, 18)) { perror("amx"); exit(1); }
        tilecfg_t c; memset(&c, 0, sizeof c); c.palette = 1;
        for (int t = 0; t < 5; t++) { c.colsb[t] = 64; c.rows[t] = NB; }
        for (int t = 5; t < 8; t++) { c.colsb[t] = 64; c.rows[t] = 16; }
        _tile_loadconfig(&c);
    }
    /* weights: one arena, 2 MiB aligned, THP, first touch here. With L2R_RESIDENT_KIB, FFN rows beyond the budget
     * (gate/up paired, down proportional) go after all resident rows and stream through gemv_nta. */
    size_t bytes = 0;
    for (int m = 0; m < NMAT; m++) {
        split(Wn[m], AMX ? 16 : 4, id, &me->r0[m], &me->r1[m]);
        me->nres[m] = me->r1[m] - me->r0[m];
        bytes += (size_t)(me->r1[m] - me->r0[m]) * Wk[m] * 2;
    }
    if (RESKIB && bytes > RESKIB << 10) {
        if (AMX || NB > 1) { fprintf(stderr, "streamed rows (slice %zu B > L2R_RESIDENT_KIB) need the AVX GEMV and L2R_BATCH=1\n", bytes); exit(1); }
        const size_t ffn = (size_t)(me->nres[4] + me->nres[5]) * H * 2 + (size_t)me->nres[6] * I * 2;
        const double f = (double)(bytes - (RESKIB << 10)) / ffn;
        const int sgu = (int)(me->nres[4] * f + 3) / 4 * 4, sd = (int)(me->nres[6] * f + 3) / 4 * 4;
        me->nres[4] -= sgu < me->nres[4] ? sgu : me->nres[4]; me->nres[5] = me->nres[4];
        me->nres[6] -= sd < me->nres[6] ? sd : me->nres[6];
    }
    const size_t HUGE = 2u << 20, al = (bytes + HUGE - 1) / HUGE * HUGE;
    uint8_t* raw = mmap(NULL, al + HUGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint8_t* arena = (uint8_t*)(((uintptr_t)raw + HUGE - 1) & ~(HUGE - 1));
    madvise(arena, al, MADV_HUGEPAGE);
    size_t off = 0;
    for (int m = 0; m < NMAT; m++) {
        const int n = me->nres[m];
        me->w[m] = (bf16*)(arena + off);
        if (AMX) pack_amx(Wfull[m] + (size_t)me->r0[m] * Wk[m], n, Wk[m], me->w[m]);
        else memcpy(me->w[m], Wfull[m] + (size_t)me->r0[m] * Wk[m], (size_t)n * Wk[m] * 2);
        off += (size_t)n * Wk[m] * 2;
    }
    const size_t resident = off;
    for (int m = 0; m < NMAT; m++) {
        const int n = me->r1[m] - me->r0[m] - me->nres[m];
        me->ws[m] = (bf16*)(arena + off);
        memcpy(me->ws[m], Wfull[m] + (size_t)(me->r0[m] + me->nres[m]) * Wk[m], (size_t)n * Wk[m] * 2);
        off += (size_t)n * Wk[m] * 2;
    }
    if (PLFD >= 0) {
        struct pl_lock_req rq = {(uintptr_t)arena, (resident + 4095) / 4096 * 4096, cpus[id], 2, 0, 0};
        if (ioctl(PLFD, PL_IOC_LOCK, &rq)) { fprintf(stderr, "cpu %d LOCK %zu B: %m\n", cpus[id], (size_t)rq.len); exit(1); }
        me->lock_id = rq.id;
        struct pl_measure ms = {rq.id, 0};
        ioctl(PLFD, PL_IOC_MEASURE, &ms); me->held0 = ms.lines ? (double)ms.l1_l2 / ms.lines : -1;
    }
    /* KV slice of the worker's attention row, node-local */
    me->row = row_of(id); me->g0 = grp0(me->row); me->g1 = grp0(me->row + 1);
    const int row = me->row, gs = me->g1 - me->g0;
    me->p0 = (int)((long)CL * (id - me->g0) / gs); me->p1 = (int)((long)CL * (id - me->g0 + 1) / gs);
    const int last = id == me->g1 - 1, np = me->p1 - me->p0 + last;
    me->kvblk = ((size_t)KVH * np * HD + 31) / 32 * 32;
    {
        const size_t kb = (size_t)KVR * 2 * me->kvblk * 2, ka = (kb + HUGE - 1) / HUGE * HUGE;
        uint8_t* kraw = mmap(NULL, ka + HUGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        me->kv = (bf16*)(((uintptr_t)kraw + HUGE - 1) & ~(HUGE - 1));
        madvise(me->kv, ka, MADV_HUGEPAGE);
    }
    for (int c = 0; c < KVR; c++) {
        bf16 *kc = me->kv + (size_t)c * 2 * me->kvblk, *vc = kc + me->kvblk;
        const bf16 *ks = kc0[row], *vs = vc0[row];
        for (int h = 0; h < KVH; h++)
            for (int p = me->p0; p < me->p1; p++) {
                memcpy(kc + ((size_t)h * np + p - me->p0) * HD, ks + ((size_t)h * CL0 + p % CL0) * HD, HD * 2);
                memcpy(vc + ((size_t)h * np + p - me->p0) * HD, vs + ((size_t)h * CL0 + p % CL0) * HD, HD * 2);
            }
    }
    int T = TILE_KIB ? (int)(((size_t)TILE_KIB << 10) / (2 * HD * 2)) : np + 1;
    if (T < 1) T = 1;
    me->e0 = gsplit(NH * HD, id, row); me->e1 = last ? NH * HD : gsplit(NH * HD, id + 1, row);

    const int A = NH * HD, KD = KVH * HD;
    float* xn = aligned_alloc(64, (size_t)H * 4); bf16* xb = aligned_alloc(64, (size_t)NB * H * 2 + 64);
    float* qn = aligned_alloc(64, (size_t)A * 4);
    float* h1 = aligned_alloc(64, (size_t)NB * H * 4); float* h2 = aligned_alloc(64, (size_t)NB * H * 4);
    float* tmp = aligned_alloc(64, (size_t)(H > I ? H : I) * 4);
    bf16* x2b = aligned_alloc(64, (size_t)NB * H * 2 + 64); bf16* h2b = aligned_alloc(64, (size_t)NB * H * 2 + 64);
    float* sc = aligned_alloc(64, (size_t)(np + 1) * 4 * NH + 64);
    float fw[MAXW];
    /* read side of every shared vector; L2R_NOBCAST=1 points them at private snapshots after step 0 */
    const float *rq = q, *rk = k, *rv = v, *rpm = pm, *rpl = pl, *rpo = po, *ro = o, *rdown = down;
    const bf16 *rattn = attn_b, *ract = act_b, *rpact = pact_b;
    const int g = NH / KVH;
    const int nd = nodeof[id];
    float *lq = aligned_alloc(64, (size_t)NB * A * 4), *lo = aligned_alloc(64, (size_t)NB * H * 4), *ldown = aligned_alloc(64, (size_t)NB * H * 4);
    bf16 *lattn = aligned_alloc(64, (size_t)NB * A * 2 + 64), *lact = aligned_alloc(64, (size_t)NB * I * 2 + 64), *lpact = aligned_alloc(64, (size_t)NB * PLE * 2 + 64);
    float *lk = aligned_alloc(64, (size_t)KD * 4), *lv = aligned_alloc(64, (size_t)KD * 4), *lpp = aligned_alloc(64, (size_t)H * 4);
    if (BCAST == 4) { rk = lk; rv = lv; }
    /* attention partials are owner-homed: each worker writes its block into the replica of its own node only and
     * combine readers fetch producer w's block from w's node (one reader per line, so no replication) */
    const float* pow_[MAXW]; const float* pmw[MAXW];
    float *my_po = po + (size_t)id * A, *my_pm = pm + id * NH, *my_pl = pl + id * NH;
    if (BCAST) {
        rq = lq; ro = lo; rdown = ldown; rattn = lattn; ract = lact; rpact = lpact;
        for (int w = 0; w < NW; w++) {
            pow_[w] = (const float*)(bc_po.rep[nodeof[w]] + bc_po.seg[w]);
            pmw[w] = (const float*)(bc_pml.rep[nodeof[w]] + bc_pml.seg[w]);
        }
        for (int w = 0; w < NW; w++) pow_[w] = (const float*)(bc_po.rep[nd] + bc_po.seg[w]); /* reader-homed */
        for (int w = 0; w < NW; w++) pmw[w] = (const float*)(bc_pml.rep[nd] + bc_pml.seg[w]);
    }
    me->steps_t = calloc(STEPS, 8);

    pthread_barrier_wait(&pbar);
    uint64_t e = 0;
    for (int st = 0; st < STEPS; st++) {
        if (id == 0 && PERF_FD >= 0 && st == STEPS / 10 && write(PERF_FD, "enable\n", 7) != 7) perror("perf ctl");
        uint64_t t = __rdtsc(), t0 = t, u;
#define PHASE(i) do { u = __rdtsc(); me->ph[i] += u - t; ++e; if (BCAST != 4 || (i) == 1) barrier(id, e); t = __rdtsc(); me->wt[i] += t - u; } while (0)
        const uint32_t tag = (uint32_t)st + 1; const int par = st & 1;
        const bf16* kvs = me->kv + (size_t)(st % KVR) * 2 * me->kvblk; /* this step's copy */
        if (EARLY_KIB)
            for (int x = 0; x < 2; x++)
                for (int h = 0; h < KVH; h++) {
                        const uint8_t* a = (const uint8_t*)(kvs + (size_t)x * me->kvblk + (size_t)h * np * HD);
                        const size_t n = (size_t)np * HD * 2 < EARLY_KIB << 10 ? (size_t)np * HD * 2 : EARLY_KIB << 10;
                        for (size_t i = 0; i < n; i += 64) pf_line(a + i);
                }
        /* 0: input norm (redundant), qkv rows */
        for (int b = 0; b < NB; b++) { rms(x_in + (size_t)b * H, w_in, H, xn); tobf(xn, H, xb + (size_t)b * H); }
        gemv(me->w[0], me->r1[0] - me->r0[0], H, xb, q + me->r0[0], A);
        gemv(me->w[1], me->r1[1] - me->r0[1], H, xb, k + me->r0[1], KD);
        gemv(me->w[2], me->r1[2] - me->r0[2], H, xb, v + me->r0[2], KD);
        if (BCAST == 4) { fv_publish(&fv_k, id, k + me->r0[1], tag, par); fv_publish(&fv_v, id, v + me->r0[2], tag, par); }
        if (BCAST == 4) fv_publish(&fv_q, id, q + me->r0[0], tag, par); else if (BCAST) bc_publish(&bc_q, id, q + me->r0[0], A);
        PHASE(0);
        /* 1: q/k norm + rope, v norm (redundant); attention over this worker's KV positions */
        const float* rqr = rq + (size_t)row * A;
        if (BCAST == 4) fv_gather(&fv_q, nd, lq, tag, par); else if (BCAST) { bc_gather_row(&bc_q, nd, lq, row); rqr = lq; }
        for (int h = 0; h < NH; h++) { rms(rqr + h * HD, w_qn, HD, qn + h * HD); rope(qn + h * HD, row); }
        me->pro[1] += __rdtsc() - t;
        if (st == 0 && id == me->g0) memcpy(chk_qn + (size_t)row * A, qn, (size_t)A * 4);
        if (last) {
            if (BCAST == 4) { fv_gather(&fv_k, nd, lk, tag, par); fv_gather(&fv_v, nd, lv, tag, par); }
            float kn[1024], vn[1024];
            for (int h = 0; h < KVH; h++) {
                rms(rk + (size_t)row * KD + h * HD, w_kn, HD, kn); rope(kn, row); rms((KVEQ ? rk : rv) + (size_t)row * KD + h * HD, NULL, HD, vn);
                tobf(kn, HD, (bf16*)kvs + ((size_t)h * np + np - 1) * HD);
                tobf(vn, HD, (bf16*)kvs + me->kvblk + ((size_t)h * np + np - 1) * HD);
            }
        }
        for (int kh = 0; kh < KVH; kh++) {
                float* m = my_pm + kh * g; float* l = my_pl + kh * g;
                float* oh = my_po + (size_t)kh * g * HD;
                const float* qh = qn + (size_t)kh * g * HD;
                const bf16 *kr = kvs + (size_t)kh * np * HD, *vr = kr + me->kvblk;
                for (int j = 0; j < g;) { /* groups above 8 heads (12B full layer: 16 per KV head) in chunks */
                    const int c = g - j >= 8 ? 8 : g - j >= 4 ? 4 : g - j >= 2 ? 2 : 1;
                    const float* qj = qh + (size_t)j * HD; float* oj = oh + (size_t)j * HD;
                    if (c == 8) attn_group(8, qj, kr, vr, np, T, sc, m + j, l + j, oj);
                    else if (c == 4) attn_group(4, qj, kr, vr, np, T, sc, m + j, l + j, oj);
                    else if (c == 2) attn_group(2, qj, kr, vr, np, T, sc, m + j, l + j, oj);
                    else attn_group(1, qj, kr, vr, np, T, sc, m + j, l + j, oj);
                    j += c;
                }
            }
        if (BCAST) { /* each line of the partial block goes only to the node(s) whose combine workers read it */
            const uint8_t* src = (const uint8_t*)my_po;
            for (int li = 0; li < A / 16; li++) {
                const __m512i v = _mm512_loadu_si512(src + (size_t)li * 64);
                for (int k = 0; k < NNODE; k++)
                    if (line_nodes[(size_t)row * (A / 16) + li] >> k & 1) {
                        uint8_t* d = bc_po.rep[k] + bc_po.seg[id] + (size_t)li * 64;
                        if (BCAST >= 3) _mm512_stream_si512((__m512i*)d, v); else _mm512_store_si512(d, v);
                    }
            }
            float rec[2 * 64];
            for (int h = 0; h < NH; h++) { rec[h] = my_pm[h]; rec[NH + h] = my_pl[h]; }
            bc_publish(&bc_pml, id, rec, 0);
        }
        PHASE(1);
        /* 2: combine the row group's partials for this worker's slice of NH*HD: one scale per (worker, head) */
        for (int h = me->e0 / HD; me->e1 > me->e0 && h <= (me->e1 - 1) / HD; h++) {
            float M = -INFINITY, den = 0;
#define PM_(w) (BCAST ? pmw[w][h] : rpm[(w) * NH + h])
#define PL_(w) (BCAST ? pmw[w][NH + h] : rpl[(w) * NH + h])
            for (int w = me->g0; w < me->g1; w++) if (PM_(w) > M) M = PM_(w);
            for (int w = me->g0; w < me->g1; w++) { fw[w] = PM_(w) == -INFINITY ? 0.f : expf(PM_(w) - M); den += fw[w] * PL_(w); }
            const int a = h * HD > me->e0 ? h * HD : me->e0, b = (h + 1) * HD < me->e1 ? (h + 1) * HD : me->e1;
            for (int i = a; i < b; i++) {
                const int d = i - h * HD;
                float num = 0;
                if (BCAST) for (int w = me->g0; w < me->g1; w++) num += fw[w] * pow_[w][(size_t)h * HD + d];
                else for (int w = me->g0; w < me->g1; w++) num += fw[w] * rpo[((size_t)w * NH + h) * HD + d];
                attn_b[(size_t)row * A + i] = f2bf(num / den);
                if (st == 0) chk_attn[(size_t)row * A + i] = num / den;
            }
        }
        if (BCAST == 4) fv_publish(&fv_attn, id, attn_b + me->e0, tag, par); else if (BCAST) bc_publish(&bc_attn, id, attn_b + (size_t)row * A + me->e0, 0);
        PHASE(2);
        /* 3: o rows */
        if (BCAST == 4) fv_gather(&fv_attn, nd, lattn, tag, par); else if (BCAST) bc_gather(&bc_attn, nd, lattn, A);
        gemv(me->w[3], me->r1[3] - me->r0[3], A, rattn, o + me->r0[3], H);
        if (BCAST == 4) fv_publish(&fv_o, id, o + me->r0[3], tag, par); else if (BCAST) bc_publish(&bc_o, id, o + me->r0[3], H);
        PHASE(3);
        /* 4: residual + norms (redundant); gate/up rows, gelu * up */
        if (BCAST == 4) fv_gather(&fv_o, nd, lo, tag, par); else if (BCAST) bc_gather(&bc_o, nd, lo, H);
        for (int b = 0; b < NB; b++) {
            float* hb = h1 + (size_t)b * H;
            const float* ob = ro + (size_t)b * H;
            if (TP > 1) { for (int i = 0; i < H; i++) tmp[i] = ob[i] + o_rest[(size_t)b * H + i]; ob = tmp; }
            rms(ob, w_pa, H, tmp); for (int i = 0; i < H; i++) hb[i] = x_in[(size_t)b * H + i] + tmp[i];
            rms(hb, w_pf, H, tmp); tobf(tmp, H, x2b + (size_t)b * H);
            if (st == 0 && id == 0) memcpy(chk_xn2 + (size_t)b * H, tmp, H * 4);
        }
        me->pro[4] += __rdtsc() - t;
        if (st == 0 && id == 0) memcpy(chk_h1, h1, (size_t)NB * H * 4);
        if (STAGE == 1) {
            for (int b = 0; b < NB; b++) {
                rms(h1 + (size_t)b * H, w_rsc, H, tmp);
                for (int i = 0; i < H; i++) tmp[i] *= 1.f / sqrtf((float)H);
                tobf(tmp, H, h2b + (size_t)b * H);
            }
            gemv(me->w[9], me->r1[9] - me->r0[9], H, h2b, rs + me->r0[9], Wn[9]);
        }
        gemv(me->w[4], me->nres[4], H, x2b, gate + me->r0[4], I);
        gemv(me->w[5], me->nres[5], H, x2b, up + me->r0[5], I);
        (PLFD >= 0 ? gemv_avx1 : gemv_nta)(me->ws[4], me->r1[4] - me->r0[4] - me->nres[4], H, x2b, gate + me->r0[4] + me->nres[4]);
        (PLFD >= 0 ? gemv_avx1 : gemv_nta)(me->ws[5], me->r1[5] - me->r0[5] - me->nres[5], H, x2b, up + me->r0[5] + me->nres[5]);
        for (int b = 0; b < NB; b++)
            gelu_mul(gate + (size_t)b * I + me->r0[4], up + (size_t)b * I + me->r0[4], me->r1[4] - me->r0[4],
                     act_b + (size_t)b * I + me->r0[4], st == 0 ? chk_act + (size_t)b * I + me->r0[4] : NULL);
        if (BCAST == 4) fv_publish(&fv_act, id, act_b + me->r0[4], tag, par); else if (BCAST) bc_publish(&bc_act, id, act_b + me->r0[4], I);
        PHASE(4);
        /* 5: down rows */
        if (BCAST == 4) fv_gather(&fv_act, nd, lact, tag, par); else if (BCAST) bc_gather(&bc_act, nd, lact, I);
        gemv(me->w[6], me->nres[6], I, ract, down + me->r0[6], H);
        (PLFD >= 0 ? gemv_avx1 : gemv_nta)(me->ws[6], me->r1[6] - me->r0[6] - me->nres[6], I, ract, down + me->r0[6] + me->nres[6]);
        if (BCAST == 4) fv_publish(&fv_down, id, down + me->r0[6], tag, par); else if (BCAST) bc_publish(&bc_down, id, down + me->r0[6], H);
        PHASE(5);
        /* 6: residual (redundant; without the per-layer input only worker 0 forms the output); ple gate rows,
         * gelu * per-layer input */
        if (PLE || id == 0) {
            if (BCAST == 4) fv_gather(&fv_down, nd, ldown, tag, par); else if (BCAST) bc_gather(&bc_down, nd, ldown, H);
            for (int b = 0; b < NB; b++) {
                float* hb = h2 + (size_t)b * H;
                const float* db = rdown + (size_t)b * H;
                if (TP > 1) { for (int i = 0; i < H; i++) tmp[i] = db[i] + down_rest[(size_t)b * H + i]; db = tmp; }
                if (STAGE == 1) {
                    float* t2 = lpp; /* H floats of scratch, unused without PLE */
                    rms(db, w_pf1, H, tmp); rms(moe_rest + (size_t)b * H, w_pf2, H, t2);
                    for (int i = 0; i < H; i++) tmp[i] += t2[i];
                    db = tmp;
                }
                rms(db, w_pff, H, tmp); for (int i = 0; i < H; i++) hb[i] = h1[(size_t)b * H + i] + tmp[i];
                tobf(hb, H, h2b + (size_t)b * H);
            }
        }
        if (!PLE) {
            if (id == 0) {
                memcpy(outv + (size_t)(st == 0 ? 0 : 1) * NB * H, h2, (size_t)NB * H * 4);
                for (size_t i = 0; i < (size_t)NB * H; i++) outv[(size_t)(st == 0 ? 0 : 1) * NB * H + i] *= SCALAR;
                if (st == 0) memcpy(chk_h2, h2, (size_t)NB * H * 4);
            }
            me->steps_t[st] = __rdtsc() - t0;
            continue;
        }
        me->pro[6] += __rdtsc() - t;
        if (st == 0 && id == 0) memcpy(chk_h2, h2, (size_t)NB * H * 4);
        gemv(me->w[7], me->r1[7] - me->r0[7], H, h2b, pg + me->r0[7], PLE);
        for (int b = 0; b < NB; b++)
            gelu_mul(pg + (size_t)b * PLE + me->r0[7], pli + (size_t)b * PLE + me->r0[7], me->r1[7] - me->r0[7],
                     pact_b + (size_t)b * PLE + me->r0[7], st == 0 ? chk_pact + (size_t)b * PLE + me->r0[7] : NULL);
        if (BCAST == 4) fv_publish(&fv_pact, id, pact_b + me->r0[7], tag, par); else if (BCAST) bc_publish(&bc_pact, id, pact_b + me->r0[7], PLE);
        PHASE(6);
        /* 7: ple projection rows */
        if (BCAST == 4) fv_gather(&fv_pact, nd, lpact, tag, par); else if (BCAST) bc_gather(&bc_pact, nd, lpact, PLE);
        gemv(me->w[8], me->r1[8] - me->r0[8], PLE, rpact, pp + me->r0[8], H);
        if (BCAST == 4) fv_publish(&fv_pp, id, pp + me->r0[8], tag, par);
        PHASE(7);
        if (id == 0) {
            if (BCAST == 4) fv_gather(&fv_pp, nd, lpp, tag, par);
            for (int b = 0; b < NB; b++) {
                rms(BCAST == 4 ? lpp : pp + (size_t)b * H, w_pn, H, tmp);
                float* ov = outv + ((size_t)(st == 0 ? 0 : 1) * NB + b) * H;
                for (int i = 0; i < H; i++) ov[i] = (h2[(size_t)b * H + i] + tmp[i]) * SCALAR;
            }
        }
        me->steps_t[st] = __rdtsc() - t0;
        if (st == 0 && NOBCAST) {
#define SNAP(dst, src, bytes) do { void* c_ = aligned_alloc(64, ((bytes) + 63) / 64 * 64); memcpy(c_, src, bytes); dst = c_; } while (0)
            SNAP(rq, q, (size_t)NH * HD * 4); SNAP(rk, k, (size_t)KVH * HD * 4); SNAP(rv, v, (size_t)KVH * HD * 4);
            SNAP(rpm, pm, (size_t)NW * NH * 4); SNAP(rpl, pl, (size_t)NW * NH * 4); SNAP(rpo, po, (size_t)NW * NH * HD * 4);
            SNAP(ro, o, (size_t)H * 4); SNAP(rdown, down, (size_t)H * 4);
            SNAP(rattn, attn_b, (size_t)NH * HD * 2); SNAP(ract, act_b, (size_t)I * 2); SNAP(rpact, pact_b, (size_t)PLE * 2);
        }
    }
    if (id == 0 && PERF_FD >= 0 && write(PERF_FD, "disable\n", 8) != 8) perror("perf ctl");
    if (PLFD >= 0) { struct pl_measure ms = {me->lock_id, 0}; ioctl(PLFD, PL_IOC_MEASURE, &ms); me->held1 = ms.lines ? (double)ms.l1_l2 / ms.lines : -1; }
    if (AMX) _tile_release();
    return NULL;
}

/* ---- checks ---- */
typedef struct { double rel_rms, cos, max_abs; } err_t;
static err_t err(const float* a, const float* r, int n) {
    double d2 = 0, r2 = 0, a2 = 0, ar = 0, mx = 0;
    for (int i = 0; i < n; i++) {
        const double d = (double)a[i] - r[i];
        d2 += d * d; r2 += (double)r[i] * r[i]; a2 += (double)a[i] * a[i]; ar += (double)a[i] * r[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    return (err_t){sqrt(d2 / (r2 ? r2 : 1)), ar / sqrt(a2 * r2 + 1e-300), mx};
}
static int cmpu64(const void* a, const void* b) { uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b; return x < y ? -1 : x > y; }

/* ---- MoE expert socket (STAGE 2) ---- */
static inline void gemvn(const bf16* W, int n, int K, const bf16* x, int nb, float* y, int ldy) {
    if (n <= 0 || nb <= 0) return;
    if (AMX) gemv_amx(W, n, K, x, nb, y, ldy); else gemv_avx(W, n, K, x, nb, y, ldy);
}

static void* run_experts(void* arg) {
    worker_t* me = arg;
    const int id = me->id;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpus[id], &cs); sched_setaffinity(0, sizeof cs, &cs);
    if (AMX) {
        if (syscall(SYS_arch_prctl, 0x1023, 18)) { perror("amx"); exit(1); }
        tilecfg_t c; memset(&c, 0, sizeof c); c.palette = 1;
        for (int t = 0; t < 5; t++) { c.colsb[t] = 64; c.rows[t] = NB; }
        for (int t = 5; t < 8; t++) { c.colsb[t] = 64; c.rows[t] = 16; }
        _tile_loadconfig(&c);
    }
    const int u = AMX ? 16 : 4;
    int ga, gz, da, dz;
    split(EI, u, id, &ga, &gz); split(H, u, id, &da, &dz);
    const int ng = gz - ga, ndn = dz - da;
    /* per local expert: gate rows [ga, gz), up rows [EI + ga, EI + gz) (K = H), down rows [da, dz) (K = EI) */
    const size_t eg = (size_t)ng * H, ed = (size_t)ndn * EI, per = 2 * eg + ed, bytes = per * NE * 2;
    const size_t HUGE = 2u << 20, al = (bytes + HUGE - 1) / HUGE * HUGE + HUGE;
    uint8_t* raw = mmap(NULL, al + HUGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    bf16* arena = (bf16*)(((uintptr_t)raw + HUGE - 1) & ~(HUGE - 1));
    madvise(arena, al, MADV_HUGEPAGE);
    for (int e = 0; e < NE; e++) {
        const bf16* gu = Wgu + (size_t)e * 2 * EI * H;
        const bf16* dn = Wdn + (size_t)e * H * EI;
        bf16* a = arena + (size_t)e * per;
        if (AMX) {
            pack_amx(gu + (size_t)ga * H, ng, H, a); pack_amx(gu + (size_t)(EI + ga) * H, ng, H, a + eg);
            pack_amx(dn + (size_t)da * EI, ndn, EI, a + 2 * eg);
        } else {
            memcpy(a, gu + (size_t)ga * H, eg * 2); memcpy(a + eg, gu + (size_t)(EI + ga) * H, eg * 2);
            memcpy(a + 2 * eg, dn + (size_t)da * EI, ed * 2);
        }
    }
    if (PLFD >= 0) {
        struct pl_lock_req rq = {(uintptr_t)arena, (bytes + 4095) / 4096 * 4096, cpus[id], 2, 0, 0};
        if (ioctl(PLFD, PL_IOC_LOCK, &rq)) { fprintf(stderr, "cpu %d LOCK %zu B: %m\n", cpus[id], (size_t)rq.len); exit(1); }
        me->lock_id = rq.id;
        struct pl_measure ms = {rq.id, 0};
        ioctl(PLFD, PL_IOC_MEASURE, &ms); me->held0 = ms.lines ? (double)ms.l1_l2 / ms.lines : -1;
    }
    float* xn = aligned_alloc(64, (size_t)H * 4);
    bf16* xb = aligned_alloc(64, (size_t)NB * H * 2 + 64);
    bf16* xe = aligned_alloc(64, (size_t)NB * H * 2 + 64);
    float* gb = aligned_alloc(64, (size_t)NB * (ng + 16) * 4 + 64);
    float* ub = aligned_alloc(64, (size_t)NB * (ng + 16) * 4 + 64);
    float* yb = aligned_alloc(64, (size_t)NB * (ndn + 16) * 4 + 64);
    bf16* lact = aligned_alloc(64, (size_t)(NP + NB) * EI * 2 + 64);
    float* lpart = aligned_alloc(64, (size_t)NB * H * 4);
    memset(lact, 0, (size_t)(NP + NB) * EI * 2);
    const int nd = nodeof[id];
    me->steps_t = calloc(STEPS, 8);
    pthread_barrier_wait(&pbar);
    uint64_t e = 0;
    for (int st = 0; st < STEPS; st++) {
        if (id == 0 && PERF_FD >= 0 && st == STEPS / 10 && write(PERF_FD, "enable\n", 7) != 7) perror("perf ctl");
        uint64_t t = __rdtsc(), t0 = t, u_;
#undef PHASE
#define PHASE(i) do { u_ = __rdtsc(); me->ph[i] += u_ - t; ++e; barrier(id, e); t = __rdtsc(); me->wt[i] += t - u_; } while (0)
        /* 0: pre-FFN-2 norm of every row (redundant), gate / up rows of every picked expert, gelu * up */
        for (int b = 0; b < NB; b++) {
            rms(h1x + (size_t)b * H, w_pf2, H, xn); tobf(xn, H, xb + (size_t)b * H);
            if (st == 0 && id == 0) memcpy(chk_xn3 + (size_t)b * H, xn, (size_t)H * 4);
        }
        me->pro[0] += __rdtsc() - t;
        for (int el = 0; el < NE; el++) {
            const int m = el_n[el], p0 = el_p0[el];
            if (!m || !ng) continue;
            for (int i = 0; i < m; i++) memcpy(xe + (size_t)i * H, xb + (size_t)pb[p0 + i] * H, (size_t)H * 2);
            const bf16* a = arena + (size_t)el * per;
            gemvn(a, ng, H, xe, m, gb, ng); gemvn(a + eg, ng, H, xe, m, ub, ng);
            for (int i = 0; i < m; i++)
                gelu_mul(gb + (size_t)i * ng, ub + (size_t)i * ng, ng, eact_b + (size_t)(p0 + i) * EI + ga,
                         st == 0 ? chk_eact + (size_t)(p0 + i) * EI + ga : NULL);
        }
        if (NP) bc_publish(&bc_eact, id, eact_b + ga, EI);
        PHASE(0);
        /* 1: down rows of every picked expert, times the routing weight, summed per token */
        if (NP) bc_gather(&bc_eact, nd, lact, EI);
        me->pro[1] += __rdtsc() - t;
        for (int b = 0; b < NB; b++) memset(part + (size_t)b * H + da, 0, (size_t)ndn * 4);
        for (int el = 0; el < NE; el++) {
            const int m = el_n[el], p0 = el_p0[el];
            if (!m || !ndn) continue;
            gemvn(arena + (size_t)el * per + 2 * eg, ndn, EI, lact + (size_t)p0 * EI, m, yb, ndn);
            for (int i = 0; i < m; i++) {
                float* d = part + (size_t)pb[p0 + i] * H + da;
                const float w = pw[p0 + i];
                for (int r = 0; r < ndn; r++) d[r] += w * yb[(size_t)i * ndn + r];
            }
        }
        bc_publish(&bc_part, id, part + da, H);
        PHASE(1);
        if (id == 0) { /* the socket's partial sum, as it would leave for the combining head socket */
            bc_gather(&bc_part, nd, lpart, H);
            memcpy(outv + (size_t)(st == 0 ? 0 : 1) * NB * H, lpart, (size_t)NB * H * 4);
        }
#undef PHASE
        me->steps_t[st] = __rdtsc() - t0;
    }
    if (id == 0 && PERF_FD >= 0 && write(PERF_FD, "disable\n", 8) != 8) perror("perf ctl");
    if (PLFD >= 0) { struct pl_measure ms = {me->lock_id, 0}; ioctl(PLFD, PL_IOC_MEASURE, &ms); me->held1 = ms.lines ? (double)ms.l1_l2 / ms.lines : -1; }
    if (AMX) _tile_release();
    return NULL;
}

static void print_err(const char* dir, const char* name, const float* a, int n, int first) {
    char nm[64]; snprintf(nm, sizeof nm, "ref.%s", !strcmp(name, "out_last") ? "out" : name);
    float* r = in_manifest(dir, nm) ? load_in(dir, nm, NULL, NULL) : NULL;
    if (r && n > 0) {
        err_t e = err(a, r, n);
        printf("%s\"%s\":[%.3e,%.8f,%.3e]", first ? "" : ",", name, e.rel_rms, e.cos, e.max_abs);
    } else printf("%s\"%s\":null", first ? "" : ",", name);
    free(r);
}

static int experts_main(const char* js) {
    H = meta_int(js, "hidden"); EPS = (float)meta_f(js, "eps"); NE = meta_int(js, "experts"); EI = meta_int(js, "moe_inter");
    TOPK = meta_int(js, "top_k");
    const int E0 = meta_int(js, "e0");
    if (!BCAST || BCAST == 4 || NOBCAST || RESKIB) { fprintf(stderr, "experts stage: L2R_BCAST=rep|repcld|repnt, no L2R_NOBCAST / L2R_RESIDENT_KIB\n"); return 1; }
    Wgu = load("w.experts.gate_up_proj", NULL, NULL); Wdn = load("w.experts.down_proj", NULL, NULL);
    w_pf2 = loadbf_as_f("w.pre_feedforward_layernorm_2.weight");
    h1x = malloc((size_t)NB * H * 4);
    int* ridx = malloc((size_t)NB * TOPK * sizeof(int)); float* rwt = malloc((size_t)NB * TOPK * 4);
    for (int b = 0; b < NB; b++) {
        float* t = load_in(ROWDIR[b], "h1", NULL, NULL); memcpy(h1x + (size_t)b * H, t, (size_t)H * 4); free(t);
        float* ix = load_in(ROWDIR[b], "router_idx", NULL, NULL); float* wv = load_in(ROWDIR[b], "router_w", NULL, NULL);
        for (int j = 0; j < TOPK; j++) { ridx[b * TOPK + j] = (int)ix[j]; rwt[b * TOPK + j] = wv[j]; }
        free(ix); free(wv);
    }
    /* (token, expert) pairs of this socket, grouped by local expert; experts in order of first pick, so with one row the
     * pairs keep the router's top-k order (ref.eact) */
    pb = malloc((size_t)NB * TOPK * sizeof(int)); pel = malloc((size_t)NB * TOPK * sizeof(int)); pw = malloc((size_t)NB * TOPK * 4);
    el_p0 = calloc(NE + 1, sizeof(int)); el_n = calloc(NE + 1, sizeof(int));
    int* order = malloc((size_t)NE * sizeof(int)); int no = 0;
    for (int b = 0; b < NB; b++)
        for (int j = 0; j < TOPK; j++) {
            const int el = ridx[b * TOPK + j] - E0;
            if (el < 0 || el >= NE) continue;
            if (!el_n[el]) order[no++] = el;
            el_n[el]++;
        }
    NP = 0;
    for (int o = 0; o < no; o++) {
        const int el = order[o];
        el_p0[el] = NP;
        for (int b = 0; b < NB; b++)
            for (int j = 0; j < TOPK; j++)
                if (ridx[b * TOPK + j] - E0 == el) { pb[NP] = b; pel[NP] = el; pw[NP] = rwt[b * TOPK + j]; NP++; }
    }
    eact_b = aligned_alloc(64, (size_t)(NP + 1) * EI * 2 + 64);
#define ZA(p, n) p = aligned_alloc(64, ((size_t)(n) * 4 + 63) / 64 * 64), memset(p, 0, (size_t)(n) * 4)
    ZA(part, NB * H); ZA(outv, 2 * NB * H); ZA(chk_xn3, NB * H); ZA(chk_eact, (NP + 1) * EI);
#undef ZA
    for (int i = 0; i < NW; i++) {
        nodeof[i] = 0;
        for (int k = 0; k < 8; k++) {
            char np_[96]; snprintf(np_, sizeof np_, "/sys/devices/system/cpu/cpu%d/node%d", cpus[i], k);
            if (!access(np_, F_OK)) { nodeof[i] = k; break; }
        }
        if (nodeof[i] + 1 > NNODE) NNODE = nodeof[i] + 1;
    }
    const int u = AMX ? 16 : 4;
    bc_init(&bc_eact, 2, EI, u, 0, NP > 0 ? NP : 1); bc_init(&bc_part, 4, H, u, 0, NB);
    tsc_ghz = calib();
    pthread_barrier_init(&pbar, NULL, NW);
    pthread_t th[MAXW];
    for (int i = 0; i < NW; i++) { WK[i].id = i; pthread_create(&th[i], NULL, run_experts, &WK[i]); }
    for (int i = 0; i < NW; i++) pthread_join(th[i], NULL);
    printf("{\"ref\":\"%s\",\"stage\":\"experts\",\"batch\":%d,\"gemv\":\"%s\",\"bcast\":%d,\"workers\":%d,\"steps\":%d,\"experts\":%d,\"pairs\":%d,\"tsc_ghz\":%.3f,\"err\":{",
           DIR, NB, AMX ? "amx" : "avx", BCAST, NW, STEPS, NE, NP, tsc_ghz);
    print_err(DIR, "xn3", chk_xn3, H, 1);
    if (NB == 1) print_err(DIR, "eact", chk_eact, NP * EI, 0);
    print_err(DIR, "out", outv, H, 0); print_err(DIR, "out_last", outv + (size_t)NB * H, H, 0);
    const int skip = STEPS / 10;
    uint64_t* stt = malloc(STEPS * 8);
    int n = 0;
    for (int s2 = skip; s2 < STEPS; s2++) { uint64_t m = 0; for (int w = 0; w < NW; w++) if (WK[w].steps_t[s2] > m) m = WK[w].steps_t[s2]; stt[n++] = m; }
    qsort(stt, n, 8, cmpu64);
    const double us = 1e-3 / tsc_ghz;
    double mean = 0; for (int i = 0; i < n; i++) mean += stt[i]; mean /= n;
    printf("},\"step_us\":{\"mean\":%.2f,\"p50\":%.2f,\"p95\":%.2f,\"p99\":%.2f,\"max\":%.2f},\"phase_us\":[", mean * us,
           stt[n / 2] * us, stt[n * 95 / 100] * us, stt[n * 99 / 100] * us, stt[n - 1] * us);
    for (int p = 0; p < 2; p++) {
        double mx = 0, avg = 0, wmx = 0, pr = 0;
        for (int w = 0; w < NW; w++) { double c = (double)WK[w].ph[p] / STEPS; avg += c; if (c > mx) mx = c; wmx += (double)WK[w].wt[p] / STEPS; pr += (double)WK[w].pro[p] / STEPS; }
        printf("%s{\"compute_max\":%.3f,\"compute_mean\":%.3f,\"prologue_mean\":%.3f,\"barrier_mean\":%.3f}", p ? "," : "", mx * us, avg / NW * us, pr / NW * us, wmx / NW * us);
    }
    printf("]");
    if (NB > 1) {
        printf(",\"rows\":[");
        for (int b = 0; b < NB; b++) {
            printf("%s{\"dir\":\"%s\",\"err\":{", b ? "," : "", ROWDIR[b]);
            print_err(ROWDIR[b], "xn3", chk_xn3 + (size_t)b * H, H, 1);
            print_err(ROWDIR[b], "out", outv + (size_t)b * H, H, 0);
            print_err(ROWDIR[b], "out_last", outv + (size_t)(NB + b) * H, H, 0);
            printf("}}");
        }
        printf("]");
    }
    double h0 = 1, h1v = 1;
    for (int w = 0; w < NW; w++) { if (WK[w].held0 < h0) h0 = WK[w].held0; if (WK[w].held1 < h1v) h1v = WK[w].held1; }
    if (PLFD >= 0) printf(",\"lock\":{\"held_l2_before_min\":%.4f,\"held_l2_after_min\":%.4f}", h0, h1v);
    const size_t wb = (size_t)NE * 3 * EI * H * 2;
    printf(",\"weight_bytes\":%zu,\"weight_bytes_per_worker\":%.0f}\n", wb, (double)wb / NW);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: l2r_layer <refdir> <steps>\n"); return 1; }
    DIR = argv[1]; STEPS = atoi(argv[2]);
    const char* g = getenv("L2R_GEMV"); AMX = g && !strcmp(g, "amx");
    NOBCAST = getenv("L2R_NOBCAST") && atoi(getenv("L2R_NOBCAST"));
    { const char* b = getenv("L2R_BCAST"); BCAST = !b || !strcmp(b, "direct") ? 0 : !strcmp(b, "rep") ? 1 : !strcmp(b, "repcld") ? 2 : !strcmp(b, "repnt") ? 3 : 4; }
    { const char* b = getenv("L2R_BARRIER"); BARR = b && !strcmp(b, "hier"); }
    RESKIB = getenv("L2R_RESIDENT_KIB") ? strtoull(getenv("L2R_RESIDENT_KIB"), 0, 10) : 0;
    if (getenv("L2R_LOCK") && atoi(getenv("L2R_LOCK"))) {
        if (!RESKIB && !getenv("L2R_EXPERTS_LOCK")) { fprintf(stderr, "L2R_LOCK needs L2R_RESIDENT_KIB\n"); return 1; }
        if ((PLFD = open("/dev/pseudo_lock", O_RDWR)) < 0) { perror("/dev/pseudo_lock"); return 1; }
    }
#define ENVI(v, n) if (getenv(n)) v = atoi(getenv(n))
    ENVI(KVR, "L2R_KV_COPIES"); ENVI(TILE_KIB, "L2R_KV_TILE_KIB"); ENVI(PFD, "L2R_KV_PFD"); ENVI(NB, "L2R_BATCH");
    ENVI(ATTN_ONLY, "L2R_ATTN_ONLY");
    EARLY_KIB = getenv("L2R_KV_EARLY_KIB") ? strtoull(getenv("L2R_KV_EARLY_KIB"), 0, 10) : 0;
    { const char* h = getenv("L2R_KV_PFH"); PFH = !h || !strcmp(h, "t0") ? 0 : !strcmp(h, "t1") ? 1 : !strcmp(h, "t2") ? 2 : 3; }
    if (KVR < 1) KVR = 1;
    if (getenv("L2R_PERF_CTL") && (PERF_FD = open(getenv("L2R_PERF_CTL"), O_WRONLY)) < 0) { perror("L2R_PERF_CTL"); return 1; }
    if (NB < 1 || NB > MAXB || (NB > 1 && (BCAST == 4 || NOBCAST))) { fprintf(stderr, "L2R_BATCH: 1..16, >1 needs L2R_BCAST=direct|rep|repcld|repnt and no L2R_NOBCAST\n"); return 1; }
    {
        static char rows[4096]; int ndir = 0;
        snprintf(rows, sizeof rows, "%s", getenv("L2R_ROWS") ? getenv("L2R_ROWS") : DIR);
        for (char* t = strtok(rows, ","); t && ndir < MAXB; t = strtok(NULL, ",")) ROWDIR[ndir++] = t;
        for (int b = ndir; b < NB; b++) ROWDIR[b] = ROWDIR[b % ndir];
    }
    REP = getenv("L2R_CTX_REPEAT") ? atoi(getenv("L2R_CTX_REPEAT")) : 1;
    const char* cl = getenv("L2R_CPUS") ? getenv("L2R_CPUS") : "2-31,34-63,66-95";
    for (const char* p = cl; *p;) {
        int a = (int)strtol(p, (char**)&p, 10), b = a;
        if (*p == '-') b = (int)strtol(p + 1, (char**)&p, 10);
        for (int c = a; c <= b && NW < MAXW; c++) cpus[NW++] = c;
        if (*p == ',') p++;
    }
    if (NB > NW) { fprintf(stderr, "L2R_BATCH %d > %d workers\n", NB, NW); return 1; }
    char path[512]; snprintf(path, sizeof path, "%s/meta.json", DIR);
    FILE* f = fopen(path, "r"); if (!f) { perror(path); return 1; }
    static char js[1 << 20]; js[fread(js, 1, sizeof js - 1, f)] = 0; fclose(f);
    STAGE = strstr(js, "\"stage\": \"experts\"") ? 2 : strstr(js, "\"stage\": \"head\"") ? 1 : 0;
    if (STAGE == 2) return experts_main(js);
    H = meta_int(js, "hidden"); NH = meta_int(js, "heads"); KVH = meta_int(js, "kv_heads"); HD = meta_int(js, "head_dim");
    I = meta_int(js, "inter"); PLE = meta_int(js, "ple"); WIN = meta_int(js, "window"); CL0 = meta_int(js, "cache_len");
    EPS = (float)meta_f(js, "eps");
    KVEQ = !in_manifest(DIR, "w.self_attn.v_proj.weight"); TP = strstr(js, "\"tp\": ") ? meta_int(js, "tp") : 1;
    if (TP > 1 && (NOBCAST || BCAST == 4)) { fprintf(stderr, "tp > 1 needs L2R_BCAST=direct|rep|repcld|repnt\n"); return 1; }
    CL = CL0 * REP;
    for (int m = 0; m < NMAT; m++) {
        char nm[128]; snprintf(nm, sizeof nm, "w.%s.weight", WN[m]);
        if ((m == 2 && KVEQ) || ((m == 7 || m == 8) && !PLE) || (m == 9 && STAGE != 1)) { Wfull[m] = NULL; continue; }
        size_t n; Wfull[m] = load(nm, &n, NULL);
    }
    Wn[0] = NH * HD; Wk[0] = H; Wn[1] = KVH * HD; Wn[2] = KVEQ ? 0 : KVH * HD; Wk[1] = Wk[2] = H; Wn[3] = H; Wk[3] = NH * HD;
    Wn[4] = Wn[5] = I; Wk[4] = Wk[5] = H; Wn[6] = H; Wk[6] = I; Wn[7] = PLE; Wk[7] = H; Wn[8] = H; Wk[8] = PLE;
    Wn[9] = 0; Wk[9] = H;
    if (STAGE == 1) {
        size_t ne; free(load("w.router.proj.weight", &ne, NULL)); Wn[9] = (int)(ne / H);
        w_pf1 = loadbf_as_f("w.post_feedforward_layernorm_1.weight"); w_pf2 = loadbf_as_f("w.pre_feedforward_layernorm_2.weight");
        free(w_pf2); w_pf2 = loadbf_as_f("w.post_feedforward_layernorm_2.weight"); w_rsc = loadbf_as_f("w.router.scale");
    }
    w_in = loadbf_as_f("w.input_layernorm.weight"); w_pa = loadbf_as_f("w.post_attention_layernorm.weight");
    w_pf = loadbf_as_f("w.pre_feedforward_layernorm.weight"); w_pff = loadbf_as_f("w.post_feedforward_layernorm.weight");
    if (PLE) w_pn = loadbf_as_f("w.post_per_layer_input_norm.weight");
    w_qn = loadbf_as_f("w.self_attn.q_norm.weight"); w_kn = loadbf_as_f("w.self_attn.k_norm.weight");
    float* ls = load("layer_scalar", NULL, NULL); SCALAR = ls[0];
    if (TP > 1) { o_rest = malloc((size_t)NB * H * 4); down_rest = malloc((size_t)NB * H * 4); }
    if (STAGE == 1) moe_rest = malloc((size_t)NB * H * 4);
    x_in = malloc((size_t)NB * H * 4); pli = malloc((size_t)NB * PLE * 4 + 64); cosv = malloc((size_t)NB * HD * 4); sinv = malloc((size_t)NB * HD * 4);
    for (int b = 0; b < NB; b++) {
        snprintf(path, sizeof path, "%s/meta.json", ROWDIR[b]);
        if (!(f = fopen(path, "r"))) { perror(path); return 1; }
        static char rj[1 << 20]; rj[fread(rj, 1, sizeof rj - 1, f)] = 0; fclose(f);
        if (meta_int(rj, "layer") != meta_int(js, "layer") || meta_int(rj, "cache_len") != CL0 || meta_int(rj, "hidden") != H) {
            fprintf(stderr, "%s: not the same layer / cache length as %s\n", ROWDIR[b], DIR); return 1;
        }
#define ROWLD(dst, nm, n) do { float* t_ = load_in(ROWDIR[b], nm, NULL, NULL); memcpy(dst + (size_t)b * (n), t_, (size_t)(n) * 4); free(t_); } while (0)
        ROWLD(x_in, "x_in", H); ROWLD(cosv, "cos", HD); ROWLD(sinv, "sin", HD);
        if (PLE) ROWLD(pli, "per_layer_input", PLE);
        if (TP > 1) { ROWLD(o_rest, "o_rest", H); ROWLD(down_rest, "down_rest", H); }
        if (STAGE == 1) ROWLD(moe_rest, "moe_rest", H);
        kc0[b] = load_in(ROWDIR[b], "kcache", NULL, NULL); vc0[b] = load_in(ROWDIR[b], "vcache", NULL, NULL);
    }
    const int A = NH * HD;
#define ZA(p, n) p = aligned_alloc(64, ((size_t)(n) * 4 + 63) / 64 * 64), memset(p, 0, (size_t)(n) * 4)
    ZA(q, NB * A); ZA(k, NB * KVH * HD); ZA(v, NB * KVH * HD); ZA(o, NB * H); ZA(gate, NB * I); ZA(up, NB * I); ZA(down, NB * H);
    ZA(pg, NB * PLE); ZA(pp, NB * H); ZA(outv, 2 * NB * H); ZA(rs, NB * (Wn[9] + 16));
    ZA(pm, NW * NH); ZA(pl, NW * NH); ZA(po, (size_t)NW * NH * HD);
    ZA(chk_qn, NB * A); ZA(chk_h1, NB * H); ZA(chk_xn2, NB * H); ZA(chk_h2, NB * H); ZA(chk_attn, NB * A); ZA(chk_act, NB * I); ZA(chk_pact, NB * PLE);
    attn_b = aligned_alloc(64, (size_t)NB * A * 2 + 64); act_b = aligned_alloc(64, (size_t)NB * I * 2 + 64); pact_b = aligned_alloc(64, (size_t)NB * PLE * 2 + 64);
    for (int i = 0; i < NW; i++) {
        nodeof[i] = 0;
        for (int k = 0; k < 8; k++) {
            char np_[96]; snprintf(np_, sizeof np_, "/sys/devices/system/cpu/cpu%d/node%d", cpus[i], k);
            if (!access(np_, F_OK)) { nodeof[i] = k; break; }
        }
        if (nodeof[i] + 1 > NNODE) NNODE = nodeof[i] + 1;
    }
    for (int k = 0; k < NNODE; k++) { node_first[k] = -1; node_count[k] = 0; }
    for (int i = 0; i < NW; i++) { if (node_first[nodeof[i]] < 0) node_first[nodeof[i]] = i; node_count[nodeof[i]]++; }
    for (int k = 0; k < NNODE; k++) {
        if (!node_count[k]) { node_first[k] = -1; }
        b_arrive[k] = node_alloc(64, k); b_done[k] = node_alloc(64, k); b_gate[k] = node_alloc(64, k);
        if (!node_count[k]) b_done[k]->v = ~0ull; /* nodes without workers never hold the leaders back */
    }
    if (BCAST) {
        const int u = AMX ? 16 : 4;
        bc_init(&bc_q, 4, A, u, 0, NB); bc_init(&bc_o, 4, H, u, 0, NB); bc_init(&bc_down, 4, H, u, 0, NB);
        bc_init_grp(&bc_attn, 2, A); bc_init(&bc_act, 2, I, u, 0, NB); if (PLE) bc_init(&bc_pact, 2, PLE, u, 0, NB);
        bc_init(&bc_po, 4, A, 0, 1, 1); bc_init(&bc_pml, 4, 2 * NH, 0, 1, 1);
        if (BCAST == 4) {
            fv_init(&fv_q, 4, A, u); fv_init(&fv_k, 4, KVH * HD, u); fv_init(&fv_v, 4, KVH * HD, u); fv_init(&fv_attn, 2, A, 1);
            fv_init(&fv_o, 4, H, u); fv_init(&fv_act, 2, I, u); fv_init(&fv_down, 4, H, u); fv_init(&fv_pact, 2, PLE, u);
            fv_init(&fv_pp, 4, H, u);
        }
        line_nodes = calloc((size_t)NB * A / 16, 1);
        for (int w = 0; w < NW; w++)
            for (int i = bc_attn.off[w]; i < bc_attn.off[w] + bc_attn.len[w]; i++) line_nodes[(size_t)bc_attn.row[w] * (A / 16) + i / 16] |= 1 << nodeof[w];
    }
    tsc_ghz = calib();
    pthread_barrier_init(&pbar, NULL, NW);
    pthread_t th[MAXW];
    for (int i = 0; i < NW; i++) { WK[i].id = i; pthread_create(&th[i], NULL, run, &WK[i]); }
    for (int i = 0; i < NW; i++) pthread_join(th[i], NULL);

    /* numerics (step 0 boundaries; out of step 0 and of the last step) */
    struct { const char* n; float* a; int len; } B[] = {
        {"q", q, A}, {"k", k, KVH * HD}, {"v", KVEQ ? k : v, KVH * HD}, {"qn", chk_qn, A}, {"attn", chk_attn, A}, {"o", o, H},
        {"h1", chk_h1, H}, {"xn2", chk_xn2, H}, {"gate", gate, I}, {"up", up, I}, {"act", chk_act, I},
        {"down", down, H}, {"rs", rs, Wn[9]}, {"h2", chk_h2, H}, {"pg", pg, PLE}, {"pact", chk_pact, PLE}, {"pp", pp, H},
        {"out", outv, H}, {"out_last", outv + (size_t)NB * H, H}};
    printf("{\"ref\":\"%s\",\"batch\":%d,\"gemv\":\"%s\",\"nobcast\":%d,\"bcast\":%d,\"barrier\":\"%s\",\"resident_kib\":%zu,\"workers\":%d,\"steps\":%d,\"ctx_rows\":%d,\"tsc_ghz\":%.3f,\"err\":{", DIR, NB, AMX ? "amx" : "avx", NOBCAST, BCAST, BARR ? "hier" : "diss", RESKIB, NW, STEPS, CL + 1, tsc_ghz);
    for (size_t b = 0; b < sizeof B / sizeof B[0]; b++) {
        char nm[64]; snprintf(nm, sizeof nm, "ref.%s", !strcmp(B[b].n, "out_last") ? "out" : B[b].n);
        float* r = REP == 1 && in_manifest(DIR, nm) ? load(nm, NULL, NULL) : NULL;
        if (r) {
            err_t e = err(B[b].a, r, B[b].len);
            printf("%s\"%s\":[%.3e,%.8f,%.3e]", b ? "," : "", B[b].n, e.rel_rms, e.cos, e.max_abs);
            free(r);
        } else printf("%s\"%s\":null", b ? "," : "", B[b].n);
    }
    /* timing: step 0..skip warm-up excluded */
    const int skip = STEPS / 10;
    uint64_t* st = malloc(STEPS * 8);
    int n = 0;
    for (int s = skip; s < STEPS; s++) { uint64_t m = 0; for (int w = 0; w < NW; w++) if (WK[w].steps_t[s] > m) m = WK[w].steps_t[s]; st[n++] = m; }
    qsort(st, n, 8, cmpu64);
    const double us = 1e-3 / tsc_ghz;
    double mean = 0; for (int i = 0; i < n; i++) mean += st[i]; mean /= n;
    printf("},\"step_us\":{\"mean\":%.2f,\"p50\":%.2f,\"p95\":%.2f,\"p99\":%.2f,\"max\":%.2f},\"phase_us\":[", mean * us,
           st[n / 2] * us, st[n * 95 / 100] * us, st[n * 99 / 100] * us, st[n - 1] * us);
    for (int p = 0; p < NPH; p++) {
        double mx = 0, avg = 0, wmx = 0, pr = 0;
        for (int w = 0; w < NW; w++) { double c = (double)WK[w].ph[p] / STEPS; avg += c; if (c > mx) mx = c; wmx += (double)WK[w].wt[p] / STEPS; pr += (double)WK[w].pro[p] / STEPS; }
        printf("%s{\"compute_max\":%.3f,\"compute_mean\":%.3f,\"prologue_mean\":%.3f,\"barrier_mean\":%.3f}", p ? "," : "", mx * us, avg / NW * us, pr / NW * us, wmx / NW * us);
    }
    size_t wb = 0; for (int m = 0; m < NMAT; m++) wb += (size_t)Wn[m] * Wk[m] * 2;
    double h0 = 1, h1 = 1;
    for (int w = 0; w < NW; w++) { if (WK[w].held0 < h0) h0 = WK[w].held0; if (WK[w].held1 < h1) h1 = WK[w].held1; }
    printf("]");
    if (NB > 1) { /* every row against its own dump's reference */
        printf(",\"rows\":[");
        for (int b = 0; b < NB; b++) {
            printf("%s{\"dir\":\"%s\",\"err\":{", b ? "," : "", ROWDIR[b]);
            for (size_t i = 0; i < sizeof B / sizeof B[0]; i++) {
                char nm[64]; snprintf(nm, sizeof nm, "ref.%s", !strcmp(B[i].n, "out_last") ? "out" : B[i].n);
                float* r = REP == 1 && in_manifest(ROWDIR[b], nm) ? load_in(ROWDIR[b], nm, NULL, NULL) : NULL;
                if (r) {
                    err_t e = err(B[i].a + (size_t)b * B[i].len, r, B[i].len);
                    printf("%s\"%s\":[%.3e,%.8f,%.3e]", i ? "," : "", B[i].n, e.rel_rms, e.cos, e.max_abs);
                    free(r);
                } else printf("%s\"%s\":null", i ? "," : "", B[i].n);
            }
            printf("}}");
        }
        printf("]");
    }
    printf(",\"kv\":{\"copies\":%d,\"rows\":%d,\"tile_kib\":%d,\"pfd\":%d,\"pfh\":%d,\"early_kib\":%zu,\"attn_only\":%d,\"bytes_per_step\":%zu}",
           KVR, NB, TILE_KIB, PFD, PFH, EARLY_KIB, ATTN_ONLY, (size_t)NB * 2 * KVH * (CL + 1) * HD * 2);
    if (PLFD >= 0) printf(",\"lock\":{\"held_l2_before_min\":%.4f,\"held_l2_after_min\":%.4f}", h0, h1);
    printf(",\"weight_bytes\":%zu,\"weight_bytes_per_worker\":%.0f}\n", wb, (double)wb / NW);
    return 0;
}
