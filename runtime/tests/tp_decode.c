/* tp_decode.c — multi-GPU tensor-parallel decode ORCHESTRATION (P1-C + P1-D).
 *
 * This is the host side of plow TP decode: the N-device launch, the peer-buffer
 * setup, the per-token decode loop across N GPUs, and the 1-token decode sweep
 * harness. It is the un-hardcoded twin of gemma4_chat.c (which drives dev=0
 * only): every backend call here takes an explicit `dev`, and the token step
 * fans out over dev = 0..N-1.
 *
 * SOURCE OF TRUTH: the design notes (§6-§8 orchestration, §12 interface
 * contract) + the transport notes (plow_hsa_alloc_peer, the ~90ns system-scope
 * atomic handshake). The backend (hsa_backend.c) is ALREADY N-device — per-GPU
 * queues/signals/pools exist and every call takes `dev`; this file is the
 * orchestration that was missing, not a backend change.
 *
 * WHAT INTEGRATES WITH tp-core (§12): tp-core owns the sharded per-rank packets
 * (--tp N compiler output), the XREDUCE/XARGMAX_FIN collective ops, and the
 * xctr counters. Where this file allocates peer_scratch and builds the per-rank
 * peer pointer table, the four §12 PlowProgram fields (xctr, rank, n_gpu,
 * peer_scratch[]) get wired in — the WIRE-IN point is marked `[§12 HOOK]`.
 * Those fields are not in the PlowProgram ABI yet (tp-core adds them), so the
 * standalone build here runs each GPU's FULL model independently (DP-style): it
 * proves the N-device launch + per-GPU dispatch + peer setup + handshake without
 * the sharded packets. When tp-core lands, flip the hook on and each rank binds
 * its 1/N shard instead of the full model.
 *
 * MODES:
 *   --verify prompt.ids [ngen]   DP correctness: prime KV by stepping the prompt
 *                                through the decode program on all N GPUs, then
 *                                generate ngen tokens. Asserts every rank emits
 *                                the identical token stream and rank-0 argmax ==
 *                                a host recompute of the logits (device==host).
 *   --sweep 1k,4k,8k,...         1-token decode sweep (P1-D): for each ctx, time
 *                                M decode steps on all N GPUs concurrently, report
 *                                median ms/tok. One row per ctx; the runner script
 *                                assembles the TP x ctx table.
 *
 * BUILD: scripts/build_tp_decode.sh (system gcc host, clean env — same contract
 * as build_tp_p2p.sh). Needs interp_decode.elf in cwd (static decode object);
 * tp_p2p_kernels.elf too if you want the cross-GPU handshake probe.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_blob.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAX_DEV 8

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
static int cmp_dbl(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* ---- blob: parsed through the SHARED structs in dev_blob.h (decode-only) ----
 * Trimmed from gemma4_chat.c's load_blob: we only ever run the decode program
 * (the last one), so we parse every program's tables for the offsets but keep the
 * decode program's pointers. The GQ appendix (op-major stream) is parsed too: the
 * harness runs the GLOBAL-QUEUE decode interpreter by default (~35% less fixed overhead),
 * falling back to the static interpreter when the blob lacks it or PLOW_STATIC=1 is set. */
typedef struct {
    PlowProgHeader h;
    PlowDevInst* insts;
    PlowStreamEnt* stream;
    uint32_t *stream_ofs, *stream_len, *succs;
    PlowWait* waits;
    /* GQ (global-queue) appendix: op-major permutation of `stream` + per-segment window bounds.
     * Present iff the blob carries PLOW_BLOB_F_GQ. Decode is a single wave-class (8), so n_seg==1
     * and the one launch walks the whole [0,n_stream) window via the shared fetch-add cursor. */
    PlowStreamEnt* gq_stream;
    uint32_t* gq_seg_ofs;
    uint32_t gq_n_seg;
    /* segmented-dispatch metadata (derived from stream .seg tags; TP prefill, mirrors gemma4_chat.c):
     * n_seg = max seg + 1; a segment is wave-class 4 iff it holds a FLASH_PREFILL op (else class 8). */
    uint32_t n_seg;
    uint8_t seg_class[512];
} Prog;
typedef struct {
    PlowBlobHeader h;
    PlowTensorDecl* tensors;
    uint8_t* init;
    uint32_t* kvrow;
    Prog* prog;
    uint8_t* raw;
} Blob;

static int load_blob(const char* path, Blob* b) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* p = malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { fclose(f); return 1; }
    fclose(f);
    b->raw = p;
    memcpy(&b->h, p, sizeof(PlowBlobHeader));
    { const char* e = plow_blob_magic_error(b->h.magic);
      if (e) { printf("%s\n", e); return 1; } }
    uint8_t* q = p + sizeof(PlowBlobHeader);
    b->tensors = (PlowTensorDecl*)q; q += (size_t)b->h.n_tensor * sizeof(PlowTensorDecl);
    b->init = q;                     q += b->h.init_bytes;
    b->kvrow = (uint32_t*)q;         q += (size_t)b->h.n_kvrow * 4;
    b->prog = calloc(b->h.n_prog, sizeof(Prog));
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
    /* GQ appendix (mirrors gemma4_chat.c load_blob): "GQ01" then per program
     * { n_seg:u32, gq_stream[n_stream], gq_seg_ofs[n_seg+1] }. The header flag is authoritative;
     * the tag guards a truncated appendix. A static-only blob leaves gq_stream NULL. */
    if (b->h.flags & PLOW_BLOB_F_GQ) {
        if ((size_t)(q - p) + 4 > (size_t)n || memcmp(q, "GQ01", 4)) {
            printf("blob flags GQ but the gq appendix is missing/corrupt\n"); return 1;
        }
        q += 4;
        for (uint32_t i = 0; i < b->h.n_prog; i++) {
            Prog* g = &b->prog[i];
            memcpy(&g->gq_n_seg, q, 4);       q += 4;
            g->gq_stream = (PlowStreamEnt*)q; q += (size_t)g->h.n_stream * sizeof(PlowStreamEnt);
            g->gq_seg_ofs = (uint32_t*)q;     q += (size_t)(g->gq_n_seg + 1) * 4;
        }
    }
    /* Derive per-program segment metadata from the stream's seg tags (as gemma4_chat.c does). */
    for (uint32_t i = 0; i < b->h.n_prog; i++) {
        Prog* g = &b->prog[i];
        uint32_t ns = 1;
        for (uint32_t j = 0; j < g->h.n_stream; j++)
            if ((uint32_t)g->stream[j].seg + 1 > ns) ns = g->stream[j].seg + 1;
        if (ns > 512) ns = 512;
        g->n_seg = ns;
        for (uint32_t s = 0; s < ns; s++) g->seg_class[s] = 8;
        for (uint32_t j = 0; j < g->h.n_stream; j++)
            if (g->insts[g->stream[j].inst].op == PLOW_DOP_FLASH_PREFILL && g->stream[j].seg < 512)
                g->seg_class[g->stream[j].seg] = 4;
    }
    return 0;
}

/* ---- safetensors (copied verbatim from gemma4_chat.c: same bind path) ---- */
#define MAX_SHARD 8
typedef struct {
    int n;
    uint8_t* base[MAX_SHARD];
    char* hdr[MAX_SHARD];
    size_t hdr_len[MAX_SHARD];
    uint64_t data0[MAX_SHARD];
} Safet;
static int st_open(Safet* s, const char* dir) {
    s->n = 0;
    int total = 0;
    for (int cand = 1; cand <= MAX_SHARD; cand++) {
        char p[512];
        snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, 1, cand);
        if (access(p, R_OK) == 0) { total = cand; break; }
    }
    for (int i = 1; total && i <= total; i++) {
        char p[512];
        snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, i, total);
        int fd = open(p, O_RDONLY);
        if (fd < 0) break;
        struct stat st; fstat(fd, &st);
        uint8_t* m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) return 1;
        uint64_t hn = *(uint64_t*)m;
        s->base[s->n] = m; s->hdr[s->n] = (char*)(m + 8);
        s->hdr_len[s->n] = (size_t)hn; s->data0[s->n] = 8 + hn;
        s->n++;
    }
    return s->n ? 0 : 1;
}
/* Find a tensor by name. Returns its data pointer and byte count; if `shape`/`ndim`
 * are non-NULL, also parses up to 2 leading shape dims (enough for the [out,in] weight
 * matrices TP shards — the strided row-parallel slice needs out=shape[0], in=shape[1]). */
static const uint8_t* st_find_ex(Safet* s, const char* name, uint64_t* nb,
                                 uint64_t shape[2], int* ndim) {
    char key[256];
    int kl = snprintf(key, sizeof(key), "\"%s\":", name);
    for (int i = 0; i < s->n; i++) {
        const char* h = s->hdr[i];
        const char* end = h + s->hdr_len[i];
        const char* p = NULL;
        for (const char* c = h; c + kl <= end; c++)
            if (!memcmp(c, key, (size_t)kl)) { p = c + kl; break; }
        if (!p) continue;
        if (shape && ndim) {
            *ndim = 0; shape[0] = shape[1] = 0;
            const char* sh = strstr(p, "\"shape\":[");
            if (sh && sh < end) {
                sh += strlen("\"shape\":[");
                while (*sh != ']' && sh < end && *ndim < 2) {
                    while (*sh == ' ' || *sh == ',') sh++;
                    if (*sh == ']') break;
                    shape[*ndim] = strtoull(sh, (char**)&sh, 10);
                    (*ndim)++;
                }
                /* count any remaining dims so callers can detect >2-D */
                while (*sh != ']' && sh < end) { if (*sh == ',') (*ndim)++; sh++; }
            }
        }
        const char* d = strstr(p, "\"data_offsets\":[");
        if (!d || d > end) continue;
        d += strlen("\"data_offsets\":[");
        unsigned long long a = strtoull(d, (char**)&d, 10);
        d++;
        unsigned long long e = strtoull(d, (char**)&d, 10);
        *nb = (uint64_t)(e - a);
        return s->base[i] + s->data0[i] + a;
    }
    return NULL;
}
static const uint8_t* st_find(Safet* s, const char* name, uint64_t* nb) {
    return st_find_ex(s, name, nb, NULL, NULL);
}

/* ---- per-GPU rank state --------------------------------------------------- */
typedef struct {
    int id;                 /* physical device index (== TP rank here)          */
    int n_gpu;              /* TP degree (for the §12 PlowProgram fields)        */
    uint32_t hidden;        /* H — peer-scratch partial slot size (H*2 bytes)    */
    void** tens;            /* [n_tensor] device pointers                        */
    void* d_tens;           /* device-side tensor pointer table                  */
    int t_ids, t_pos, t_kvlen, t_logits;
    int t_og_tp, t_dg_tp;   /* peer-mapped partial slots (bound into peer, §7a)  */
    /* decode-program device tables */
    void *d_inst, *d_stream, *d_sofs, *d_slen, *d_waits, *d_succs, *d_ctr;
    /* GQ (global-queue) decode: op-major stream + per-segment window bounds + the shared cursor.
     * NULL when the harness runs the static decode interpreter. gq_ncursor = n_seg (cursor words). */
    void *d_gq_stream, *d_gq_seg, *d_gq_cursor;
    uint32_t gq_ncursor;
    /* pinned host staging */
    PlowDevInst* h_inst;    /* patched decode inst stream (kv-row immediates)    */
    int32_t* h_scalar;      /* per-step scalars (ids/pos/kvlen)                  */
    /* peer-mapped reduction region this rank owns (§7a), + the [N] pointer table */
    void* peer;             /* plow_hsa_alloc_peer region, owner == this rank    */
    void* d_peer_tbl;       /* device array of [N] peer bases (this rank's view) */
    void* d_trace;          /* rank-0 per-packet PlowTraceRec buffer (PLOW_TRACE_RAW), else NULL */
    /* --- TP PREFILL --- peer layout is [T,hidden]-sized, not [1,hidden]:
     * partial_A @ 0, partial_B @ slot_b = rows_max*hidden*2, xctr @ 2*slot_b. slot_b/peer_bytes are
     * discovered from the blob's down-XReduce i[2]; they SUPERSEDE the old 4*H / 64KB decode layout. */
    uint32_t slot_b;        /* partial_B byte offset in peer = rows_max*hidden*2  */
    size_t   peer_bytes;    /* 2*slot_b + xctr region                            */
    /* the active prefill-bucket program's device tables (weights + tensor table are shared) */
    void *pd_inst, *pd_stream, *pd_sofs, *pd_slen, *pd_waits, *pd_succs, *pd_ctr;
    void *pd_gqs, *pd_gqseg, *pd_gqc; uint32_t pd_gqn;
    PlowDevInst* ph_inst;   /* pinned patched prefill inst stream                */
    int pdp;                /* which bucket program is bound in pd_*             */
} Dev;

/* §7a peer-scratch region per GPU: [partial_A][partial_B][xctr]. H<=6144 covers
 * every Gemma-4 (31B H=5376). Sized generously; the exact offsets are the
 * compiler's (tp-core) once XREDUCE is emitted — here it is the allocation +
 * peer-map + pointer-table facility the design calls the "single new memory
 * facility" (§7a). Tiny (~31KB), so over-allocating costs nothing. */
#define PEER_SCRATCH_BYTES (64u * 1024u)

/* Is `name` a column-parallel weight? (output-dim sharded: HF [out,in] row-major ⇒ a
 * CONTIGUOUS row-range, the design notes §3a/§3b.) q/k/v/gate/up. lm_head is REPLICATED
 * (compiler keeps it full-vocab under TP), so it is NOT here. */
static int is_col_parallel(const char* n) {
    return strstr(n, "q_proj.weight") || strstr(n, "k_proj.weight") ||
           strstr(n, "v_proj.weight") || strstr(n, "gate_proj.weight") ||
           strstr(n, "up_proj.weight");
}
/* Is `name` a row-parallel weight? (input-dim sharded ⇒ a STRIDED column-range.) o/down. */
static int is_row_parallel(const char* n) {
    return strstr(n, "o_proj.weight") || strstr(n, "down_proj.weight");
}

/* Bind one rank's 1/N WEIGHT SLICE (the design notes §3b — the real work). Column-parallel
 * weights (q/k/v/gate/up) are a contiguous output-row range of the [out,in] checkpoint matrix;
 * row-parallel weights (o/down) are a strided input-column range; everything else (norms,
 * embed_tokens/lm_head) is replicated full. tp==1 loads full (byte-identical to the old path).
 * Uploads the decode program's device tables, allocates the pinned per-step buffers. */
static int setup_dev(plow_hsa* h, Dev* d, int id, int N, Blob* B, Safet* S, int dp,
                     void* stage, size_t STAGE, uint32_t VOCAB, int use_gq,
                     Safet* Sf, int have_fp8) {
    d->id = id;
    d->n_gpu = N;
    d->t_ids = d->t_pos = d->t_kvlen = d->t_logits = -1;
    d->t_og_tp = d->t_dg_tp = -1;
    d->tens = calloc(B->h.n_tensor, sizeof(void*));
    uint64_t wb = 0; int nw = 0;
    const double t0 = now();
    for (uint32_t i = 0; i < B->h.n_tensor; i++) {
        PlowTensorDecl* td = &B->tensors[i];
        d->tens[i] = plow_hsa_alloc(h, id, td->bytes);
        if (!d->tens[i]) { printf("dev%d: VRAM alloc failed: %s\n", id, td->name); return 1; }
        if (!strcmp(td->name, "in.ids")) d->t_ids = (int)i;
        if (!strcmp(td->name, "in.pos")) d->t_pos = (int)i;
        if (!strcmp(td->name, "in.kvlen")) d->t_kvlen = (int)i;
        if (!strcmp(td->name, "act.logits")) d->t_logits = (int)i;
        if (!strcmp(td->name, "act.og_tp")) d->t_og_tp = (int)i;
        if (!strcmp(td->name, "act.dg_tp")) d->t_dg_tp = (int)i;
        if (!strncmp(td->name, "model.", 6) || !strncmp(td->name, "lm_head", 7)) {
            uint64_t got = 0, shp[2] = {0, 0}; int nd = 0;
            const uint8_t* src = st_find_ex(S, td->name, &got, shp, &nd);
            if (!src) { printf("dev%d: MISSING WEIGHT: %s\n", id, td->name); return 1; }
            const int col = N > 1 && is_col_parallel(td->name);
            const int row = N > 1 && is_row_parallel(td->name);
            if (col) {
                /* contiguous output-row range: rank id owns rows [shard_idx*out_sh,...) = a byte
                 * range. Usually the full matrix splits into N slices (shard_idx==id); but a
                 * kv-replicated weight (k/v_proj at tp > n_kv_heads, §3a/§13.2) splits into FEWER
                 * slices (n_shards = full/shard < N), and tp/n_shards ranks SHARE each slice — so
                 * shard_idx = id / (N / n_shards) folds the sharing group. q/gate/up: n_shards==N. */
                if (td->bytes == 0 || got % td->bytes != 0) {
                    printf("dev%d: COL SHARD MISMATCH %s (full %llu not a multiple of shard %llu)\n",
                           id, td->name, (unsigned long long)got, (unsigned long long)td->bytes); return 1; }
                const uint64_t n_shards = got / td->bytes;
                if (n_shards == 0 || (uint64_t)N % n_shards != 0) {
                    printf("dev%d: COL SHARD %s: %llu shards do not divide N=%d (kv-replication needs N%%n_shards==0)\n",
                           id, td->name, (unsigned long long)n_shards, N); return 1; }
                const uint64_t shard_idx = (uint64_t)id / ((uint64_t)N / n_shards);
                const uint8_t* ss = src + shard_idx * td->bytes;
                for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                    size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                    memcpy(stage, ss + o, n);
                    plow_hsa_copy_h2d(h, id, (uint8_t*)d->tens[i] + o, stage, n);
                }
            } else if (row) {
                /* strided input-column range: [out,in] ⇒ rank id owns cols [id*in/N,...) of every
                 * row. Gather the strided slice into a contiguous host buffer, then chunk-upload. */
                if (nd != 2 || shp[0] == 0 || shp[1] == 0) {
                    printf("dev%d: ROW SHARD needs 2-D shape for %s (got nd=%d)\n", id, td->name, nd); return 1; }
                const uint64_t out = shp[0], in_full = shp[1];
                if (in_full % (uint64_t)N) {
                    printf("dev%d: ROW SHARD %s in=%llu not divisible by N=%d\n", id, td->name,
                           (unsigned long long)in_full, N); return 1; }
                const uint64_t in_sh = in_full / (uint64_t)N;
                if (out * in_sh * 2 != td->bytes) {
                    printf("dev%d: ROW SHARD MISMATCH %s (%llu*%llu*2 != %llu)\n", id, td->name,
                           (unsigned long long)out, (unsigned long long)in_sh,
                           (unsigned long long)td->bytes); return 1; }
                uint8_t* rb = malloc(td->bytes);
                if (!rb) { printf("dev%d: row-gather malloc failed\n", id); return 1; }
                for (uint64_t rr = 0; rr < out; rr++)
                    memcpy(rb + rr * in_sh * 2,
                           src + (rr * in_full + (uint64_t)id * in_sh) * 2, (size_t)in_sh * 2);
                for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                    size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                    memcpy(stage, rb + o, n);
                    plow_hsa_copy_h2d(h, id, (uint8_t*)d->tens[i] + o, stage, n);
                }
                free(rb);
            } else {
                /* replicated (norms, embed_tokens/lm_head) or tp==1: full copy */
                if (got != td->bytes) { printf("dev%d: SIZE MISMATCH %s (full %llu != %llu)\n", id,
                    td->name, (unsigned long long)got, (unsigned long long)td->bytes); return 1; }
                for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                    size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                    memcpy(stage, src + o, n);
                    plow_hsa_copy_h2d(h, id, (uint8_t*)d->tens[i] + o, stage, n);
                }
            }
            wb += td->bytes; nw++;
        } else if (!strncmp(td->name, "fp8/", 4)) {
            /* FP8 DECODE twin: quantized projection weight (e4m3, 1 B/elt) or its per-output-channel
             * f32 scale, loaded from the SEPARATE fp8 checkpoint (PLOW_FP8_DIR). "fp8/" is stripped
             * before the st_find lookup. Sharded exactly like the bf16 twin: q/k/v/gate/up (+their
             * [out] scale) are output-contiguous col slices; o/down weight is an input-strided row
             * slice while its [out] scale is NOT sharded (output full) -> replicated. */
            if (!have_fp8) { printf("dev%d: fp8 pkt needs PLOW_FP8_DIR (missing %s)\n", id, td->name); return 1; }
            const char* key = td->name + 4;
            const int is_scale = strstr(key, "_scale") != NULL;
            const uint64_t esz = is_scale ? 4 : 1;          /* scale=f32, weight=e4m3 */
            uint64_t got = 0, shp[2] = {0, 0}; int nd = 0;
            const uint8_t* src = st_find_ex(Sf, key, &got, shp, &nd);
            if (!src) { printf("dev%d: MISSING FP8 WEIGHT: %s\n", id, key); return 1; }
            const int col = N > 1 && is_col_parallel(key);
            const int row = N > 1 && is_row_parallel(key) && !is_scale;
            if (col) {
                if (td->bytes == 0 || got % td->bytes != 0) {
                    printf("dev%d: FP8 COL SHARD MISMATCH %s (full %llu not a multiple of shard %llu)\n",
                           id, key, (unsigned long long)got, (unsigned long long)td->bytes); return 1; }
                const uint64_t n_shards = got / td->bytes;
                if (n_shards == 0 || (uint64_t)N % n_shards != 0) {
                    printf("dev%d: FP8 COL SHARD %s: %llu shards do not divide N=%d\n",
                           id, key, (unsigned long long)n_shards, N); return 1; }
                const uint64_t shard_idx = (uint64_t)id / ((uint64_t)N / n_shards);
                const uint8_t* ss = src + shard_idx * td->bytes;
                for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                    size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                    memcpy(stage, ss + o, n);
                    plow_hsa_copy_h2d(h, id, (uint8_t*)d->tens[i] + o, stage, n);
                }
            } else if (row) {
                if (nd != 2 || shp[0] == 0 || shp[1] == 0) {
                    printf("dev%d: FP8 ROW SHARD needs 2-D shape for %s (got nd=%d)\n", id, key, nd); return 1; }
                const uint64_t out = shp[0], in_full = shp[1];
                if (in_full % (uint64_t)N) {
                    printf("dev%d: FP8 ROW SHARD %s in=%llu not divisible by N=%d\n", id, key,
                           (unsigned long long)in_full, N); return 1; }
                const uint64_t in_sh = in_full / (uint64_t)N;
                if (out * in_sh * esz != td->bytes) {
                    printf("dev%d: FP8 ROW SHARD MISMATCH %s (%llu*%llu*%llu != %llu)\n", id, key,
                           (unsigned long long)out, (unsigned long long)in_sh, (unsigned long long)esz,
                           (unsigned long long)td->bytes); return 1; }
                uint8_t* rb = malloc(td->bytes);
                if (!rb) { printf("dev%d: fp8 row-gather malloc failed\n", id); return 1; }
                for (uint64_t rr = 0; rr < out; rr++)
                    memcpy(rb + rr * in_sh * esz,
                           src + (rr * in_full + (uint64_t)id * in_sh) * esz, (size_t)in_sh * esz);
                for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                    size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                    memcpy(stage, rb + o, n);
                    plow_hsa_copy_h2d(h, id, (uint8_t*)d->tens[i] + o, stage, n);
                }
                free(rb);
            } else {
                if (got != td->bytes) { printf("dev%d: FP8 SIZE MISMATCH %s (full %llu != %llu)\n", id,
                    key, (unsigned long long)got, (unsigned long long)td->bytes); return 1; }
                for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                    size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                    memcpy(stage, src + o, n);
                    plow_hsa_copy_h2d(h, id, (uint8_t*)d->tens[i] + o, stage, n);
                }
            }
            wb += td->bytes; nw++;
        } else if (td->init_off != PLOW_INIT_NONE) {
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, B->init + td->init_off + o, n);
                plow_hsa_copy_h2d(h, id, (uint8_t*)d->tens[i] + o, stage, n);
            }
        }
    }
    if (d->t_ids < 0 || d->t_pos < 0 || d->t_kvlen < 0 || d->t_logits < 0) {
        printf("dev%d: missing an in.* / act.logits tensor\n", id); return 1;
    }
    d->d_tens = plow_hsa_alloc(h, id, (size_t)B->h.n_tensor * sizeof(void*));
    plow_hsa_upload(h, id, d->d_tens, d->tens, (size_t)B->h.n_tensor * sizeof(void*));

    Prog* g = &B->prog[dp];
    d->d_inst  = plow_hsa_alloc(h, id, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    d->d_stream= plow_hsa_alloc(h, id, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
    d->d_sofs  = plow_hsa_alloc(h, id, (size_t)B->h.n_cu * 4);
    d->d_slen  = plow_hsa_alloc(h, id, (size_t)B->h.n_cu * 4);
    d->d_waits = plow_hsa_alloc(h, id, (size_t)(g->h.n_wait ? g->h.n_wait : 1) * sizeof(PlowWait));
    d->d_succs = plow_hsa_alloc(h, id, (size_t)(g->h.n_succ ? g->h.n_succ : 1) * 4);
    d->d_ctr   = plow_hsa_alloc(h, id, (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4);
    plow_hsa_upload(h, id, d->d_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    plow_hsa_upload(h, id, d->d_stream, g->stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
    plow_hsa_upload(h, id, d->d_sofs, g->stream_ofs, (size_t)B->h.n_cu * 4);
    plow_hsa_upload(h, id, d->d_slen, g->stream_len, (size_t)B->h.n_cu * 4);
    if (g->h.n_wait) plow_hsa_upload(h, id, d->d_waits, g->waits, (size_t)g->h.n_wait * sizeof(PlowWait));
    if (g->h.n_succ) plow_hsa_upload(h, id, d->d_succs, g->succs, (size_t)g->h.n_succ * 4);

    /* GQ decode: upload this rank's op-major gq_stream + segment bounds, and allocate its OWN
     * fetch-add cursor (one word per segment, PLOW_CTR-strided, zeroed per launch like the counters).
     * The gq_stream is a rank-agnostic permutation of `stream`, so every rank uploads the same table;
     * the cursor is per-rank device state. XReduce entries carry SE_XCTR and gate on peer xctr — the
     * GQ scheduler routes them exactly like any other packet (interp.hip §6a). */
    d->d_gq_stream = d->d_gq_seg = d->d_gq_cursor = NULL;
    d->gq_ncursor = 0;
    if (use_gq && g->gq_stream) {
        d->gq_ncursor = g->gq_n_seg ? g->gq_n_seg : 1;
        d->d_gq_stream = plow_hsa_alloc(h, id, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
        d->d_gq_seg    = plow_hsa_alloc(h, id, (size_t)(g->gq_n_seg + 1) * 4);
        d->d_gq_cursor = plow_hsa_alloc(h, id, (size_t)d->gq_ncursor * PLOW_CTR_STRIDE * 4);
        plow_hsa_upload(h, id, d->d_gq_stream, g->gq_stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
        plow_hsa_upload(h, id, d->d_gq_seg, g->gq_seg_ofs, (size_t)(g->gq_n_seg + 1) * 4);
    }

    d->h_inst = plow_hsa_alloc_host(h, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    memcpy(d->h_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    d->h_scalar = plow_hsa_alloc_host(h, 64);
    (void)VOCAB;
    printf("dev%d: bound %d weights (%.1f GiB) in %.1f s\n", id, nw,
           wb / 1073741824.0, now() - t0);
    return 0;
}

/* Zero all ranks' counters (the deadlock-safety obligation of §6d: every rank's
 * counters — and, once tp-core lands, xctr — must be zero before ANY rank is
 * launched, so a cross-rank publish can never race a stale consume). */
static void zero_all_counters(plow_hsa* h, Dev* devs, int N, Blob* B, int dp,
                              const uint32_t* zc) {
    Prog* g = &B->prog[dp];
    for (int r = 0; r < N; r++) {
        plow_hsa_copy_h2d(h, devs[r].id, devs[r].d_ctr, zc,
                          (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4);
        /* §6d: the cross-GPU xctr counters must ALSO be zero on every rank before ANY rank
         * launches — the coarse XReduce gate (threshold n_gpu) needs a clean slate each token,
         * or a stale count would let a consume race a publish. xctr lives at byte 4*H into the
         * peer region (after partial_A/partial_B); zero from there to the end (partials are
         * overwritten by o_proj/down before XReduce reads them, so leaving them is fine). */
        if (N > 1) {
            const size_t xoff = (size_t)2 * devs[r].slot_b;   /* xctr @ after both partials */
            plow_hsa_copy_h2d(h, devs[r].id, (char*)devs[r].peer + xoff, zc,
                              devs[r].peer_bytes - xoff);
        }
        /* GQ: the shared fetch-add cursor must be zero before each launch (same lifecycle as the
         * counters — the scheduler claims [0,n_stream) by fetch_add from 0). */
        if (devs[r].d_gq_cursor)
            plow_hsa_copy_h2d(h, devs[r].id, devs[r].d_gq_cursor, zc,
                              (size_t)devs[r].gq_ncursor * PLOW_CTR_STRIDE * 4);
    }
}

/* Fill a PlowProgram for rank r. [§12 HOOK] is where tp-core's four fields go. */
static void fill_program(PlowProgram* pr, Dev* d, Blob* B, int dp) {
    Prog* g = &B->prog[dp];
    memset(pr, 0, sizeof(*pr));
    pr->insts = d->d_inst; pr->stream = d->d_stream; pr->stream_ofs = d->d_sofs;
    pr->stream_len = d->d_slen; pr->waits = d->d_waits; pr->succs = d->d_succs;
    pr->counters = d->d_ctr; pr->tensors = (void* const*)d->d_tens;
    pr->trace = NULL; pr->cur_seg = 0;
    /* GQ decode: the interpreter reads gq_stream/gq_seg_ofs and drives the shared cursor when the
     * _gq object is loaded; NULL here => the static per-CU stream path. Decode is one wave-class
     * (n_seg==1), so cur_seg==0 walks the whole [gq_seg_ofs[0], gq_seg_ofs[1]) window. */
    pr->gq_stream = d->d_gq_stream; pr->gq_seg_ofs = d->d_gq_seg; pr->gq_cursor = d->d_gq_cursor;
    pr->trace = d->d_trace;   /* NULL on all ranks unless PLOW_TRACE_RAW set (rank 0 only) */
    (void)g;
    /* [§12 HOOK] — wire tp-core's four cross-GPU PlowProgram fields (the design notes §7a,
     * §12). peer_scratch[rank] layout: [partial_A(H*2)][partial_B(H*2)][xctr(...)]; og_tp/dg_tp
     * are bound into partial_A/partial_B (see main), so the row-parallel o_proj/down write their
     * partials peer-visibly and XReduce sums the N peer slots. xctr sits at byte 2*H*2 = 4*H into
     * this rank's own peer region; ALL ranks lay it out at the same offset, so a peer's counter
     * base is peer_scratch[r] + (xctr - peer_scratch[rank]) (interp derives it, no 5th field). */
    if (d->n_gpu > 1) {
        pr->rank = (uint32_t)d->id;
        pr->n_gpu = (uint32_t)d->n_gpu;
        pr->peer_scratch = (void* const*)d->d_peer_tbl;
        pr->xctr = (uint32_t*)((char*)d->peer + (size_t)2 * d->slot_b);
    }
}

/* One decode token across N GPUs (§8 orchestration half): broadcast `token` to
 * every rank's in.ids, patch the kv-append row, set pos/kvlen, launch all N
 * megakernels (async — the queues run concurrently), wait all, read each rank's
 * argmax back. Returns rank-0's sampled id; fills out_ids[r] for agreement
 * checks. This is the host step; the collectives themselves are inline
 * counter-gated packets INSIDE each megakernel (nothing for the host to do
 * between launch and wait — plow's structural win over launched RCCL). */
static int decode_step(plow_hsa* h, Dev* devs, int N, Blob* B, int dp,
                       plow_hsa_kernel* kdec /* [N], one per device */, const uint32_t* zc,
                       int token, int pos, int kvlen, int32_t* out_ids,
                       double* gpu_ms) {
    Prog* g = &B->prog[dp];
    /* The only per-step change to the ~68 KB inst stream is `i[3]` (the KV write row) of the
     * n_kvrow kvrow insts, so re-upload just the [lo,hi] slice that spans them, not the whole
     * array. On Gemma-31B the kvrow insts are the 120 k/v HeadNormRope immediates SCATTERED in
     * pairs across all 60 layers — span [4,664] of 676, i.e. 98% of the stream — so a per-inst
     * SCATTER upload (120 tiny h2d) would cost more in submission overhead than one contiguous
     * copy; the slice copy is the sound win (never more submissions, never more bytes). This is
     * host prep before the timed launch, so it does not move decode ms/tok — it only trims the
     * host loop. lo/hi are index-only (device-independent), computed once. */
    uint32_t lo = g->h.n_inst ? g->h.n_inst - 1 : 0, hi = 0;
    for (uint32_t i = 0; i < B->h.n_kvrow; i++) {
        uint32_t idx = B->kvrow[i];
        if (idx < lo) lo = idx;
        if (idx > hi) hi = idx;
    }
    for (int r = 0; r < N; r++) {
        Dev* d = &devs[r];
        for (uint32_t i = 0; i < B->h.n_kvrow; i++)
            d->h_inst[B->kvrow[i]].i[3] = (uint32_t)pos;
        if (B->h.n_kvrow)
            plow_hsa_copy_h2d(h, d->id,
                              (uint8_t*)d->d_inst + (size_t)lo * sizeof(PlowDevInst),
                              &d->h_inst[lo], (size_t)(hi - lo + 1) * sizeof(PlowDevInst));
        else
            plow_hsa_copy_h2d(h, d->id, d->d_inst, d->h_inst,
                              (size_t)g->h.n_inst * sizeof(PlowDevInst));
        d->h_scalar[0] = token; plow_hsa_copy_h2d(h, d->id, d->tens[d->t_ids], d->h_scalar, 4);
        d->h_scalar[0] = pos;   plow_hsa_copy_h2d(h, d->id, d->tens[d->t_pos], d->h_scalar, 4);
        d->h_scalar[0] = kvlen; plow_hsa_copy_h2d(h, d->id, d->tens[d->t_kvlen], d->h_scalar, 4);
    }
    zero_all_counters(h, devs, N, B, dp, zc);   /* §6d: all ranks before any launch */

    const double t0 = now();
    /* launch ALL ranks (async), THEN drain ALL — the N megakernels are co-resident
     * and step the token in parallel. */
    const int dbg = getenv("PLOW_TP_DBG") != NULL;
    for (int r = 0; r < N; r++) {
        Dev* d = &devs[r];
        PlowProgram pr; fill_program(&pr, d, B, dp);
        if (dbg) { fprintf(stderr, "[launch dev%d]\n", d->id); fflush(stderr); }
        if (plow_hsa_launch(h, d->id, &kdec[r], B->h.n_cu * PLOW_WG_THREADS, 1, 1,
                            PLOW_WG_THREADS, 1, 1, 0, &pr, sizeof(pr))) {
            printf("dev%d: LAUNCH FAILED\n", d->id); return 1;
        }
        if (getenv("PLOW_TP_SERIAL")) plow_hsa_wait(h, d->id);
    }
    if (!getenv("PLOW_TP_SERIAL")) for (int r = 0; r < N; r++) plow_hsa_wait(h, devs[r].id);
    if (dbg) { fprintf(stderr, "[waited all]\n"); fflush(stderr); }
    const double t1 = now();
    if (gpu_ms) *gpu_ms = (t1 - t0) * 1e3;

    /* collect the sampled id the device wrote into in.ids on each rank */
    for (int r = 0; r < N; r++) {
        plow_hsa_copy_d2h(h, devs[r].id, devs[r].h_scalar, devs[r].tens[devs[r].t_ids], 4);
        out_ids[r] = devs[r].h_scalar[0];
    }
    return 0;
}

/* ============================ TP PREFILL ============================
 * The un-hardcoded twin of gemma4_chat.c's prefill, sharded across N GPUs. Each rank streams
 * its 1/N weight shard, computes its head-shard of flash_prefill over the full prompt, and the
 * two XReduce all-reduces/layer (after o_proj and down) fire INLINE on the [T,hidden] partials —
 * the bandwidth-bound regime (each partial is T rows, not decode's 1). The prompt is fed as
 * ceil(n/C) chunks of ONE bucket size C; the last chunk pads with masked garbage. Non-segmented:
 * one launch per chunk per rank on the static prefill interpreter (the SEG_OFF baseline — correct,
 * and the simplest orchestration for the N-rank collective rendezvous). */

/* Bind bucket program `pdp`'s device tables on one rank (weights + tensor table already bound
 * by setup_dev; only the per-program inst/stream/counter tables differ). */
static int bind_prefill_prog(plow_hsa* h, Dev* d, Blob* B, int pdp, int use_gq) {
    Prog* g = &B->prog[pdp];
    d->pdp = pdp;
    d->pd_inst   = plow_hsa_alloc(h, d->id, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    d->pd_stream = plow_hsa_alloc(h, d->id, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
    d->pd_sofs   = plow_hsa_alloc(h, d->id, (size_t)B->h.n_cu * 4);
    d->pd_slen   = plow_hsa_alloc(h, d->id, (size_t)B->h.n_cu * 4);
    d->pd_waits  = plow_hsa_alloc(h, d->id, (size_t)(g->h.n_wait ? g->h.n_wait : 1) * sizeof(PlowWait));
    d->pd_succs  = plow_hsa_alloc(h, d->id, (size_t)(g->h.n_succ ? g->h.n_succ : 1) * 4);
    d->pd_ctr    = plow_hsa_alloc(h, d->id, (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4);
    plow_hsa_upload(h, d->id, d->pd_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    plow_hsa_upload(h, d->id, d->pd_stream, g->stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
    plow_hsa_upload(h, d->id, d->pd_sofs, g->stream_ofs, (size_t)B->h.n_cu * 4);
    plow_hsa_upload(h, d->id, d->pd_slen, g->stream_len, (size_t)B->h.n_cu * 4);
    if (g->h.n_wait) plow_hsa_upload(h, d->id, d->pd_waits, g->waits, (size_t)g->h.n_wait * sizeof(PlowWait));
    if (g->h.n_succ) plow_hsa_upload(h, d->id, d->pd_succs, g->succs, (size_t)g->h.n_succ * 4);
    d->pd_gqs = d->pd_gqseg = d->pd_gqc = NULL; d->pd_gqn = 0;
    if (use_gq && g->gq_stream) {
        d->pd_gqn = g->gq_n_seg ? g->gq_n_seg : 1;
        d->pd_gqs   = plow_hsa_alloc(h, d->id, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
        d->pd_gqseg = plow_hsa_alloc(h, d->id, (size_t)(g->gq_n_seg + 1) * 4);
        d->pd_gqc   = plow_hsa_alloc(h, d->id, (size_t)d->pd_gqn * PLOW_CTR_STRIDE * 4);
        plow_hsa_upload(h, d->id, d->pd_gqs, g->gq_stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
        plow_hsa_upload(h, d->id, d->pd_gqseg, g->gq_seg_ofs, (size_t)(g->gq_n_seg + 1) * 4);
    }
    d->ph_inst = plow_hsa_alloc_host(h, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    memcpy(d->ph_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    if (!d->pd_inst || !d->pd_stream || !d->pd_sofs || !d->pd_slen || !d->pd_waits ||
        !d->pd_succs || !d->pd_ctr || !d->ph_inst) {
        printf("dev%d: bind_prefill_prog ALLOC FAILED (pd_inst=%p stream=%p ctr=%p)\n",
               d->id, d->pd_inst, d->pd_stream, d->pd_ctr); return 1;
    }
    if (getenv("PLOW_PF_DBG"))
        fprintf(stderr, "[bind] dev%d bucket T=%u n_inst=%u n_stream=%u n_ctr=%u pd_inst=%p d_tens=%p tens[0]=%p\n",
                d->id, g->h.t, g->h.n_inst, g->h.n_stream, g->h.n_counter, d->pd_inst, d->d_tens, d->tens[0]);
    return 0;
}

/* Fill a PlowProgram for the bound prefill bucket on rank r (mirror of fill_program). */
static void fill_prefill_program(PlowProgram* pr, Dev* d, Blob* B) {
    memset(pr, 0, sizeof(*pr));
    pr->insts = d->pd_inst; pr->stream = d->pd_stream; pr->stream_ofs = d->pd_sofs;
    pr->stream_len = d->pd_slen; pr->waits = d->pd_waits; pr->succs = d->pd_succs;
    pr->counters = d->pd_ctr; pr->tensors = (void* const*)d->d_tens;
    pr->trace = NULL; pr->cur_seg = 0;
    pr->gq_stream = d->pd_gqs; pr->gq_seg_ofs = d->pd_gqseg; pr->gq_cursor = d->pd_gqc;
    if (d->n_gpu > 1) {
        pr->rank = (uint32_t)d->id;
        pr->n_gpu = (uint32_t)d->n_gpu;
        pr->peer_scratch = (void* const*)d->d_peer_tbl;
        pr->xctr = (uint32_t*)((char*)d->peer + (size_t)2 * d->slot_b);
    }
    (void)B;
}

/* Zero the bound prefill bucket's counters + peer xctr on every rank (deadlock safety §6d). */
static void zero_prefill_counters(plow_hsa* h, Dev* devs, int N, Blob* B, const uint32_t* zc) {
    for (int r = 0; r < N; r++) {
        Prog* g = &B->prog[devs[r].pdp];
        plow_hsa_copy_h2d(h, devs[r].id, devs[r].pd_ctr, zc,
                          (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4);
        if (N > 1) {
            const size_t xoff = (size_t)2 * devs[r].slot_b;
            plow_hsa_copy_h2d(h, devs[r].id, (char*)devs[r].peer + xoff, zc,
                              devs[r].peer_bytes - xoff);
        }
        if (devs[r].pd_gqc)
            plow_hsa_copy_h2d(h, devs[r].id, devs[r].pd_gqc, zc,
                              (size_t)devs[r].pd_gqn * PLOW_CTR_STRIDE * 4);
    }
}

/* Find the lm_head matmul (the one writing act.logits) in a program. */
static int find_lm(Prog* g, int t_logits) {
    for (uint32_t i = 0; i < g->h.n_inst; i++) {
        const uint16_t o = g->insts[i].op;
        const int mm = (o == PLOW_DOP_GEMM || o == PLOW_DOP_GEMM_SMALL ||
                        o == PLOW_DOP_GEMM_MED || o == PLOW_DOP_GEMV);
        if (mm && g->insts[i].t[0] == (uint32_t)t_logits) return (int)i;
    }
    return -1;
}

/* Prefill the whole prompt across N ranks, chunked by a single bucket size C.
 * Returns rank-0's first sampled token; fills out_ids[r] (agreement check) and *ttft_ms. */
static int prefill_run(plow_hsa* h, Dev* devs, int N, Blob* B, plow_hsa_kernel* kpre,
                       plow_hsa_kernel* kflash, uint32_t flash_threads,
                       int pdp, const uint32_t* zc, const int32_t* prompt, int n_prompt,
                       int32_t* out_ids, int32_t* pbuf, double* ttft_ms) {
    const uint32_t CH = B->prog[pdp].h.t;
    for (int r = 0; r < N; r++) if (devs[r].pdp != pdp && bind_prefill_prog(h, &devs[r], B, pdp, 0)) return 1;
    const int lm = find_lm(&B->prog[pdp], devs[0].t_logits);
    if (lm < 0) { printf("prefill: no lm_head matmul in bucket T=%u\n", CH); return 1; }

    const double p0 = now();
    uint32_t c0 = 0;
    while ((int)c0 < n_prompt) {
        const uint32_t clen = ((uint32_t)n_prompt - c0 < CH) ? ((uint32_t)n_prompt - c0) : CH;
        for (uint32_t i = 0; i < CH; i++) pbuf[i] = (i < clen) ? prompt[c0 + i] : 0;
        for (int r = 0; r < N; r++) {
            Dev* d = &devs[r];
            Prog* g = &B->prog[pdp];
            plow_hsa_copy_h2d(h, d->id, d->tens[d->t_ids], pbuf, (size_t)CH * 4);
            for (uint32_t i = 0; i < CH; i++) pbuf[i] = (int32_t)(c0 + i);
            plow_hsa_copy_h2d(h, d->id, d->tens[d->t_pos], pbuf, (size_t)CH * 4);
            for (uint32_t i = 0; i < CH; i++) pbuf[i] = prompt ? ((i < clen) ? prompt[c0 + i] : 0) : 0;
            /* patch the three per-chunk immediates (found by IDENTITY, as gemma4_chat.c) */
            memcpy(d->ph_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst));
            for (uint32_t i = 0; i < g->h.n_inst; i++) {
                if (d->ph_inst[i].op == PLOW_DOP_HEADNORM_ROPE && d->ph_inst[i].fj[1].u != 0)
                    d->ph_inst[i].i[3] = c0;                 /* KV cache write row */
                else if (d->ph_inst[i].op == PLOW_DOP_FLASH_PREFILL) {
                    d->ph_inst[i].i[4] = c0;                 /* q_pos0 */
                    d->ph_inst[i].i[1] = c0 + clen;          /* n_kv so far */
                }
            }
            d->ph_inst[lm].i[4] = clen - 1;                  /* lm_head reads the last real row */
            plow_hsa_copy_h2d(h, d->id, d->pd_inst, d->ph_inst,
                              (size_t)g->h.n_inst * sizeof(PlowDevInst));
        }
        zero_prefill_counters(h, devs, N, B, zc);
        /* SEGMENTED dispatch (Gemma flash_prefill needs the 4-wave object; single-launch is wrong).
         * Per rank, enqueue ALL segments async (queue-ordered by the AQL barrier bit), then drain all
         * ranks. The XReduce collectives (class-8 segments) rendezvous across ranks via the inline
         * system-scope gate: each rank's queue reaches the collective and waits N arrivals. */
        Prog* gp = &B->prog[pdp];
        const int pfdbg = getenv("PLOW_PF_DBG") != NULL;
        if (pfdbg && c0 == 0) { fprintf(stderr, "[prefill] n_seg=%u classes:", gp->n_seg);
            for (uint32_t s = 0; s < gp->n_seg; s++) fprintf(stderr, " %u", gp->seg_class[s]); fprintf(stderr, "\n"); }
        /* PER-SEGMENT, ALL-RANKS: launch segment s on EVERY rank, then drain all ranks before s+1.
         * This host barrier per segment guarantees every rank runs a class-8 segment's XReduce
         * collectives CONCURRENTLY (the segment holds both all-reduces of a layer), so the inline
         * system-scope gate rendezvouses immediately instead of spinning to its deadline. (Per-rank-
         * all-segments let the ranks desync — a lagging rank made peers time out and bail, giving a
         * WRONG, 100x-slow reduction at TP>=4.) */
        for (uint32_t s = 0; s < gp->n_seg; s++) {
            const int use4 = gp->seg_class[s] == 4;
            const uint32_t th = use4 ? flash_threads : (uint32_t)PLOW_WG_THREADS;
            for (int r = 0; r < N; r++) {
                PlowProgram pr; fill_prefill_program(&pr, &devs[r], B);
                pr.cur_seg = s;
                plow_hsa_kernel* kk = use4 ? &kflash[r] : &kpre[r];
                if (plow_hsa_launch(h, devs[r].id, kk, B->h.n_cu * th, 1, 1, th, 1, 1,
                                    0, &pr, sizeof(pr))) {
                    printf("dev%d: PREFILL LAUNCH FAILED (chunk c0=%u seg %u)\n", devs[r].id, c0, s);
                    return 1;
                }
            }
            for (int r = 0; r < N; r++) plow_hsa_wait(h, devs[r].id);
            if (pfdbg && c0 == 0) fprintf(stderr, "[prefill] seg %u class%u OK\n", s, gp->seg_class[s]);
        }
        c0 += CH;
    }
    *ttft_ms = (now() - p0) * 1e3;
    for (int r = 0; r < N; r++) {
        plow_hsa_copy_d2h(h, devs[r].id, devs[r].h_scalar, devs[r].tens[devs[r].t_ids], 4);
        out_ids[r] = devs[r].h_scalar[0];
    }
    return 0;
}

/* ---- [4] cross-GPU handshake probe from the real orchestrator (tp-transport
 * pattern), proving the ~90ns system-scope atomic gate works from THIS launch
 * path — the primitive the inline collectives ride on. Loads tp_p2p_kernels.elf
 * on ranks A,B; runs the ping/pong over a peer flag word. No-op if the elf is
 * absent. */
typedef struct { void* p2c; void* c2p; uint32_t iters; uint64_t deadline;
                 void* cycles; void* status; } arg_ping;
typedef struct { void* p2c; void* c2p; uint32_t iters; uint64_t deadline; } arg_pong;

static void handshake_probe(plow_hsa* h, Dev* devs, int N, double tick_ns) {
    if (N < 2) return;
    if (access("tp_p2p_kernels.elf", R_OK) != 0) {
        printf("[handshake] tp_p2p_kernels.elf not in cwd — skipping (build it to probe)\n");
        return;
    }
    FILE* f = fopen("tp_p2p_kernels.elf", "rb");
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc((size_t)n);
    if (fread(co, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(co); return; }
    fclose(f);
    const int A = devs[0].id, Bd = devs[1].id;
    plow_hsa_kernel k_ping, k_pong;
    if (plow_hsa_load_code_object(h, A, co, (size_t)n) ||
        plow_hsa_load_code_object(h, Bd, co, (size_t)n) ||
        plow_hsa_get_kernel(h, A, "tp_ping", &k_ping) ||
        plow_hsa_get_kernel(h, Bd, "tp_pong", &k_pong)) {
        printf("[handshake] could not load tp_p2p kernels — skipping\n"); free(co); return;
    }
    uint32_t* flags = (uint32_t*)plow_hsa_alloc_peer(h, A, 4 * sizeof(uint32_t));
    uint32_t clr[2] = {0, 0}; plow_hsa_upload(h, A, flags, clr, sizeof clr);
    uint64_t* cyc = (uint64_t*)plow_hsa_alloc(h, A, 8);
    uint32_t* pstat = (uint32_t*)plow_hsa_alloc(h, A, 4);
    const uint32_t PP = 10000;
    const uint64_t deadline = (uint64_t)(1.0 / (tick_ns * 1e-9)); /* ~1 s of ticks */
    arg_pong pg = { flags, flags + 1, PP, deadline };
    plow_hsa_launch(h, Bd, &k_pong, 1, 1, 1, 1, 1, 1, 0, &pg, sizeof pg);
    arg_ping pi = { flags, flags + 1, PP, deadline, cyc, pstat };
    plow_hsa_launch(h, A, &k_ping, 1, 1, 1, 1, 1, 1, 0, &pi, sizeof pi);
    plow_hsa_wait(h, A); plow_hsa_wait(h, Bd);
    uint32_t st = 1; uint64_t cycles = 0;
    plow_hsa_download(h, A, &st, pstat, 4);
    plow_hsa_download(h, A, &cycles, cyc, 8);
    if (st == 0) {
        const double rt_ns = (double)cycles * tick_ns / PP;
        printf("[handshake] GPU%d<->GPU%d system-scope atomic gate: round-trip %.0f ns, "
               "one-way %.0f ns (%u pairs) — the cross-GPU counter-gate WORKS from the "
               "launch path\n", A, Bd, rt_ns, rt_ns / 2.0, PP);
    } else {
        printf("[handshake] system-scope atomic did NOT propagate (status=0x%08x)\n", st);
    }
    plow_hsa_free(h, flags); plow_hsa_free(h, cyc); plow_hsa_free(h, pstat); free(co);
}

static int parse_ctx(const char* s) { /* "32k" -> 32768, "1000" -> 1000 */
    char* e; long v = strtol(s, &e, 10);
    if (*e == 'k' || *e == 'K') v *= 1024;
    return (int)v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s model.pkt <model-dir> [--tp N] "
               "[--verify prompt.ids [ngen]] [--sweep 1k,4k,8k,16k,32k] [--steps M]\n", argv[0]);
        return 1;
    }
    const char* pkt = argv[1];
    const char* mdir = argv[2];
    int N = 1, ngen = 8, steps = 21, pf_chunk = 0, pf_ntok = 0;
    const char* verify_ids = NULL;
    const char* sweep_list = NULL;
    const char* prefill_ids = NULL;   /* --prefill prompt.ids (or "synth" + --synth N) */
    const char* pf_sweep = NULL;      /* --pf-sweep 8k,32k,64k,128k : TTFT sweep, one weight-load */
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--tp") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify") && i + 1 < argc) {
            verify_ids = argv[++i];
            if (i + 1 < argc && argv[i + 1][0] != '-') ngen = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--prefill") && i + 1 < argc) prefill_ids = argv[++i];
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) pf_chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--synth") && i + 1 < argc) pf_ntok = parse_ctx(argv[++i]);
        else if (!strcmp(argv[i], "--pf-sweep") && i + 1 < argc) pf_sweep = argv[++i];
        else if (!strcmp(argv[i], "--sweep") && i + 1 < argc) sweep_list = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    }
    if (N < 1) N = 1;
    if (N > MAX_DEV) N = MAX_DEV;

    Blob B;
    if (load_blob(pkt, &B)) return 1;
    Safet S;
    if (st_open(&S, mdir)) { printf("no safetensors in %s\n", mdir); return 1; }
    /* FP8 DECODE: the quantized projection weights + scales load from a SEPARATE fp8 checkpoint
     * (PLOW_FP8_DIR), leaving mdir as the bf16 source. Opened only if the pkt declares "fp8/"
     * tensors (harmless otherwise). Mirrors gemma4_chat.c's mixed bf16/fp8 load. */
    Safet Sf; int have_fp8 = 0;
    { const char* fd = getenv("PLOW_FP8_DIR");
      if (fd && !st_open(&Sf, fd)) { have_fp8 = 1; printf("fp8 weights dir: %s\n", fd); } }

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("plow_hsa_init: %s\n", plow_hsa_last_error()); return 1; }
    const int ndev = plow_hsa_device_count(h);
    if (N > ndev) { printf("requested TP=%d but only %d GPUs\n", N, ndev); N = ndev; }
    printf("== plow TP decode orchestration ==  TP=%d over %d GPUs\n", N, ndev);

    /* gfx950 s_memrealtime = 1 ns/tick (tp-transport.md §1); used by the probe. */
    const double tick_ns = 1.0;

    const int dp = (int)B.h.n_prog - 1;           /* decode program = last */

    /* Discover the blob's sharding from its XReduce packets (self-describing, §8a): the compiler
     * bakes i[0]=hidden and i[1]=n_gpu into every XReduce. If the caller asked for TP>1 the blob
     * MUST be a --tp N sharded blob (else the weight slices/peer partials are wrong); if it is a
     * tp==1 blob it carries no XReduce and runs replicated (DP). */
    uint32_t hidden = 0, blob_tp = 1, slot_b = 0;
    for (uint32_t i = 0; i < B.prog[dp].h.n_inst; i++) {
        if (B.prog[dp].insts[i].op == PLOW_DOP_XREDUCE) {
            if (hidden == 0) { hidden = B.prog[dp].insts[i].i[0]; blob_tp = B.prog[dp].insts[i].i[1]; }
            /* partial_B XReduce carries i[2] = slot_b = rows_max*hidden*2 (the fixed dg_tp offset,
             * the design notes); partial_A carries i[2]=0. Take the max => slot_b. */
            if (B.prog[dp].insts[i].i[2] > slot_b) slot_b = B.prog[dp].insts[i].i[2];
        }
    }
    if (N > 1) {
        if (blob_tp <= 1 || hidden == 0) {
            printf("ERROR: --tp %d but the packet is NOT sharded (no XReduce). "
                   "Recompile with: plowc gemma4 --tp %d <model> <ctx> <out.pkt>\n", N, N);
            return 1;
        }
        if ((int)blob_tp != N) {
            printf("ERROR: packet is sharded for tp=%u but you launched --tp %d. "
                   "The tensor slices and peer partials only match at tp=%u.\n", blob_tp, N, blob_tp);
            return 1;
        }
        printf("sharded packet: tp=%u, hidden=%u (peer partial slot = %u B)\n",
               blob_tp, hidden, hidden * 2);
    }

    /* find act.logits size for VOCAB (host argmax check) */
    uint32_t vocab = 0;
    for (uint32_t i = 0; i < B.h.n_tensor; i++)
        if (!strcmp(B.tensors[i].name, "act.logits")) vocab = (uint32_t)(B.tensors[i].bytes / 2);

    /* DECODE SCHEDULER — global-queue (default, faster) vs static per-CU stream.
     * The GQ decode object (interp_decode_gq.elf / plow_interp_dec_gfx950_gq) cuts the fixed
     * per-packet overhead ~35% (14.55 vs 19.7 ms baseline on 31B) by walking one shared fetch-add
     * cursor over the op-major gq_stream instead of a static per-CU stream. The XReduce collective
     * packets ride the same scheduler (SE_XCTR gate, interp.hip §6a). Falls back to static if the
     * blob lacks the GQ appendix, the _gq object is absent, or PLOW_STATIC=1 asks for it. */
    int use_gq = getenv("PLOW_STATIC") ? 0 : 1;
    if (use_gq && !(B.h.flags & PLOW_BLOB_F_GQ)) { use_gq = 0; printf("pkt has no GQ stream -> static decode\n"); }
    if (use_gq && access("interp_decode_gq.elf", R_OK) != 0) { use_gq = 0; printf("no interp_decode_gq.elf -> static decode\n"); }
    /* FP8 decode object selection (mirror gemma4_chat.c). A GEMV_FP8 pkt runs the fp8-weight decode
     * object; a FLASH_DECODE_FP8 pkt (fp8-KV, which is also fp8-weight) runs the fp8kv decode object.
     * The exported symbol is unchanged (only one decode object is ever loaded per rank). */
    int is_fp8_pkt = 0, is_fp8kv_pkt = 0;
    for (uint32_t pi = 0; pi < B.h.n_prog; pi++)
        for (uint32_t j = 0; j < B.prog[pi].h.n_inst; j++) {
            uint32_t op = B.prog[pi].insts[j].op;
            if (op == PLOW_DOP_GEMV_FP8) is_fp8_pkt = 1;
            if (op == PLOW_DOP_FLASH_DECODE_FP8) is_fp8kv_pkt = 1;
        }
    const char* dec_elf =
        is_fp8kv_pkt ? (use_gq ? "interp_decode_fp8kv_gq.elf" : "interp_decode_fp8kv.elf")
        : is_fp8_pkt ? (use_gq ? "interp_decode_fp8_gq.elf"   : "interp_decode_fp8.elf")
                     : (use_gq ? "interp_decode_gq.elf"       : "interp_decode.elf");
    const char* dec_sym = use_gq ? "plow_interp_dec_gfx950_gq" : "plow_interp_dec_gfx950";
    printf("decode scheduler: %s (%s%s)\n", use_gq ? "GLOBAL QUEUE" : "static", dec_elf,
           is_fp8kv_pkt ? " [fp8-KV]" : is_fp8_pkt ? " [fp8-weight]" : "");

    plow_hsa_kernel kdec[MAX_DEV];
    { FILE* f = fopen(dec_elf, "rb");
      if (!f) { printf("missing %s in cwd\n", dec_elf); return 1; }
      fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
      void* co = malloc((size_t)n);
      if (fread(co, 1, (size_t)n, f) != (size_t)n) return 1;
      fclose(f);
      for (int r = 0; r < N; r++) {
          if (plow_hsa_load_code_object(h, r, co, (size_t)n)) { printf("dev%d: load interp failed\n", r); return 1; }
          if (plow_hsa_get_kernel(h, r, dec_sym, &kdec[r])) {
              printf("dev%d: no %s\n", r, dec_sym); return 1;
          }
      }
    }

    /* max_ctx from in.pos */
    int mctx = 0;
    for (uint32_t i = 0; i < B.h.n_tensor; i++)
        if (!strcmp(B.tensors[i].name, "in.pos")) mctx = (int)(B.tensors[i].bytes / 4);
    printf("decode program: %u packets, %u counters | vocab %u | max_ctx %d\n\n",
           B.prog[dp].h.n_inst, B.prog[dp].h.n_counter, vocab, mctx);

    /* ---- bind every rank (full model = DP standalone) ---- */
    const size_t STAGE = 64u << 20;
    void* stage = plow_hsa_alloc_host(h, STAGE);
    Dev devs[MAX_DEV];
    /* peer layout (the design notes): partial_A[0,slot_b) partial_B[slot_b,2*slot_b) xctr[2*slot_b,...).
     * slot_b = rows_max*hidden*2 from the blob (=2*hidden for a decode-only tp==1 fallback where it is 0). */
    /* 512 counters, 128B each = 64 KB. The TWO-SHOT prefill all-reduce (the design notes)
     * uses 2 xctr gate ids per collective (reduce-scatter + all-gather) = 4/layer × 60 layers
     * = 240 ids/program; one-shot decode uses 2/layer = 120. 512 leaves comfortable headroom. */
    const size_t XCTR_BYTES = 512u * PLOW_CTR_STRIDE * 4u;
    if (N > 1 && slot_b == 0) { printf("ERROR: sharded blob but no partial_B slot in XReduce\n"); return 1; }
    const size_t peer_bytes = (N > 1) ? ((size_t)2 * slot_b + XCTR_BYTES) : PEER_SCRATCH_BYTES;
    for (int r = 0; r < N; r++) {
        if (setup_dev(h, &devs[r], r, N, &B, &S, dp, stage, STAGE, vocab, use_gq, &Sf, have_fp8)) return 1;
        devs[r].hidden = hidden;   /* peer partial slot size, for the §12 fields */
        devs[r].slot_b = slot_b;
        devs[r].peer_bytes = peer_bytes;
        devs[r].pdp = -1;
        devs[r].d_trace = NULL;
    }
    /* STEP 1 diagnostic (tp-optimize): per-packet critical-path trace on rank 0, dumped for
     * decode_critpath.py. Only rank 0 is traced (the sharded-GEMV bandwidth is a per-rank
     * property). Enabling trace costs an s_memrealtime per packet; fine for a diagnostic run. */
    const char* trace_raw = getenv("PLOW_TRACE_RAW");
    if (trace_raw)
        devs[0].d_trace = plow_hsa_alloc(h, devs[0].id,
                              (size_t)B.prog[dp].h.n_stream * sizeof(PlowTraceRec));

    /* ---- PEER-BUFFER SETUP (§7a): one peer-mapped reduction region per GPU,
     * all-to-all peer access (plow_hsa_alloc_peer maps to EVERY agent), and the
     * per-rank [N] peer-base pointer table wired for §12. ---- */
    void* peer_base[MAX_DEV];
    for (int r = 0; r < N; r++) {
        devs[r].peer = plow_hsa_alloc_peer(h, devs[r].id, peer_bytes);
        if (!devs[r].peer) { printf("dev%d: alloc_peer failed: %s\n", r, plow_hsa_last_error()); return 1; }
        peer_base[r] = devs[r].peer;
    }
    for (int r = 0; r < N; r++) {
        devs[r].d_peer_tbl = plow_hsa_alloc(h, devs[r].id, (size_t)N * sizeof(void*));
        plow_hsa_upload(h, devs[r].id, devs[r].d_peer_tbl, peer_base, (size_t)N * sizeof(void*));
    }
    /* §7a: BIND og_tp/dg_tp INTO this rank's peer region so the row-parallel o_proj/down write
     * their partial H-vectors peer-visibly. partial_A at byte 0, partial_B at byte H*2; XReduce
     * reads peer_scratch[r]+slot (slot 0 / H*2). We overwrite the two tensor-table entries (which
     * setup_dev pointed at throwaway local VRAM) and re-upload the table. */
    if (N > 1) {
        for (int r = 0; r < N; r++) {
            if (devs[r].t_og_tp < 0 || devs[r].t_dg_tp < 0) {
                printf("dev%d: sharded packet missing act.og_tp/act.dg_tp\n", r); return 1; }
            devs[r].tens[devs[r].t_og_tp] = devs[r].peer;                        /* partial_A @ 0     */
            devs[r].tens[devs[r].t_dg_tp] = (char*)devs[r].peer + (size_t)slot_b; /* partial_B @ slot_b */
            plow_hsa_upload(h, devs[r].id, devs[r].d_tens, devs[r].tens,
                            (size_t)B.h.n_tensor * sizeof(void*));
        }
    }
    printf("peer setup: %d reduction regions (%zu B each) peer-mapped all-to-all; "
           "per-rank peer_scratch[%d] tables built%s\n",
           N, peer_bytes, N,
           N > 1 ? "; og_tp/dg_tp @ 0/slot_b; xctr @ 2*slot_b" : "");

    /* [4] cross-GPU handshake from the launch path (the ~90ns gate). */
    if (!getenv("PLOW_NO_HANDSHAKE")) handshake_probe(h, devs, N, tick_ns);
    printf("\n");

    /* shared zero-counter buffer (pinned): sized to the LARGEST program's counter set (prefill
     * buckets have more counters than decode) AND at least the xctr region, since zero_* reuses it
     * to clear both the counters and the peer xctr. */
    uint32_t max_ctr = 0;
    for (uint32_t i = 0; i < B.h.n_prog; i++)
        if (B.prog[i].h.n_counter > max_ctr) max_ctr = B.prog[i].h.n_counter;
    size_t zc_bytes = (size_t)max_ctr * PLOW_CTR_STRIDE * 4;
    if (zc_bytes < XCTR_BYTES) zc_bytes = XCTR_BYTES;
    uint32_t* zc = plow_hsa_alloc_host(h, zc_bytes);
    memset(zc, 0, zc_bytes);

    int32_t out_ids[MAX_DEV];

    /* ================= PREFILL: TP prefill (TTFT) + first-token bit-exactness ============ */
    if (prefill_ids) {
        /* load the 8-wave prefill interp + the 4-wave flash object on every rank (segmented dispatch) */
        plow_hsa_kernel kpre[MAX_DEV], kflash[MAX_DEV];
        { FILE* f = fopen("interp_prefill.elf", "rb");
          if (!f) { printf("missing interp_prefill.elf in cwd\n"); return 1; }
          fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
          void* co = malloc((size_t)n);
          if (fread(co, 1, (size_t)n, f) != (size_t)n) return 1;
          fclose(f);
          for (int r = 0; r < N; r++) {
              if (plow_hsa_load_code_object(h, r, co, (size_t)n)) { printf("dev%d: load interp_prefill failed\n", r); return 1; }
              if (plow_hsa_get_kernel(h, r, "plow_interp_gfx950", &kpre[r])) { printf("dev%d: no plow_interp_gfx950\n", r); return 1; }
          }
          free(co);
        }
        uint32_t flash_threads = 256;
        { const char* e = getenv("PLOW_FLASH_THREADS"); if (e) flash_threads = (uint32_t)atoi(e); }
        { FILE* f = fopen("interp_flash.elf", "rb");
          if (!f) { printf("missing interp_flash.elf in cwd\n"); return 1; }
          fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
          void* co = malloc((size_t)n);
          if (fread(co, 1, (size_t)n, f) != (size_t)n) return 1;
          fclose(f);
          for (int r = 0; r < N; r++) {
              if (plow_hsa_load_code_object(h, r, co, (size_t)n)) { printf("dev%d: load interp_flash failed\n", r); return 1; }
              if (plow_hsa_get_kernel(h, r, "plow_interp_flash_gfx950", &kflash[r])) { printf("dev%d: no plow_interp_flash_gfx950\n", r); return 1; }
          }
          free(co);
        }
        /* build the list of prompt lengths to run: --pf-sweep list, else the single --synth/file. */
        int lens[16]; int nlen = 0;
        if (pf_sweep) {
            char buf[256]; snprintf(buf, sizeof buf, "%s", pf_sweep);
            for (char* t = strtok(buf, ","); t && nlen < 16; t = strtok(NULL, ",")) lens[nlen++] = parse_ctx(t);
        } else {
            lens[nlen++] = pf_ntok > 0 ? pf_ntok : (int)(prefill_ids && strcmp(prefill_ids, "synth") ? 0 : 4096);
        }
        /* load a real token file once (if given) so every sweep point reuses the same prompt prefix. */
        int32_t* file_ids = NULL; int file_n = 0;
        if (strcmp(prefill_ids, "synth")) {
            FILE* pf = fopen(prefill_ids, "rb");
            if (!pf) { printf("no %s\n", prefill_ids); return 1; }
            fseek(pf, 0, SEEK_END); long pn = ftell(pf); fseek(pf, 0, SEEK_SET);
            file_n = (int)(pn / 4); file_ids = malloc((size_t)pn);
            if (fread(file_ids, 1, (size_t)pn, pf) != (size_t)pn) return 1;
            fclose(pf);
            if (nlen == 1 && lens[0] == 0) lens[0] = file_n;
        }
        /* chunk bucket = --chunk C, else the largest bucket (fewest launches) */
        int pdp = -1; uint32_t best_t = 0;
        for (uint32_t i = 0; i + 1 < B.h.n_prog; i++) {
            uint32_t t = B.prog[i].h.t;
            if (t == 0) continue;
            if (pf_chunk) { if ((int)t == pf_chunk) { pdp = (int)i; best_t = t; } }
            else if (t > best_t) { pdp = (int)i; best_t = t; }
        }
        int32_t* pbuf = plow_hsa_alloc_host(h, (size_t)best_t * 4);
        int32_t* prompt = malloc((size_t)(mctx) * 4);
        uint16_t* logit = plow_hsa_alloc_host(h, (size_t)vocab * 2);
        printf("PREFILL (TP=%d): chunk bucket T=%u | %-8s %10s %12s %8s %s\n",
               N, best_t, "ctx", "TTFT ms", "tok/s", "tok0", "bitexact");
        for (int li = 0; li < nlen; li++) {
            int n_prompt = lens[li];
            if (n_prompt > mctx - 1) n_prompt = mctx - 1;
            if (n_prompt <= 0) continue;
            for (int i = 0; i < n_prompt; i++)
                prompt[i] = file_ids ? file_ids[i % file_n]
                                     : (int32_t)(101u + (((uint64_t)i * 1103515245ull + 12345ull) % 30000ull));
            double ttft = 0, best_ttft = 1e30; int reps = (n_prompt <= 16384) ? 2 : 1;
            for (int rep = 0; rep < reps; rep++) {
                if (prefill_run(h, devs, N, &B, kpre, kflash, flash_threads, pdp, zc, prompt, n_prompt,
                                out_ids, pbuf, &ttft)) return 1;
                if (ttft < best_ttft) best_ttft = ttft;
            }
            int mism = 0;
            for (int r = 1; r < N; r++) if (out_ids[r] != out_ids[0]) mism++;
            plow_hsa_copy_d2h(h, devs[0].id, logit, devs[0].tens[devs[0].t_logits], (size_t)vocab * 2);
            int hb = 0; float hv = -1e30f;
            for (uint32_t v = 0; v < vocab; v++) { float x = b2f(logit[v]); if (x > hv) { hv = x; hb = (int)v; } }
            printf("PREFILL: %-8d %10.1f %12.0f %8d %s\n",
                   n_prompt, best_ttft, n_prompt / (best_ttft / 1e3), out_ids[0],
                   (mism == 0 && out_ids[0] == hb) ? "OK(dev==host,ranks agree)" : "MISMATCH");
        }
        free(prompt);
    }

    /* ================= VERIFY: DP correctness ================= */
    if (verify_ids) {
        FILE* pf = fopen(verify_ids, "rb");
        if (!pf) { printf("no %s\n", verify_ids); return 1; }
        fseek(pf, 0, SEEK_END); long pn = ftell(pf); fseek(pf, 0, SEEK_SET);
        int n_prompt = (int)(pn / 4);
        int32_t* prompt = malloc((size_t)pn);
        if (fread(prompt, 1, (size_t)pn, pf) != (size_t)pn) return 1;
        fclose(pf);
        if (n_prompt > 64) n_prompt = 64;   /* decode-only priming; keep it short */
        printf("VERIFY (DP): priming %d prompt tokens through the decode program on "
               "%d GPU(s), then generating %d tokens.\n", n_prompt, N, ngen);

        int best = prompt[0];
        int mismatches = 0;
        /* prime: step each prompt token (append its KV row), predict the next */
        for (int i = 0; i < n_prompt; i++) {
            if (decode_step(h, devs, N, &B, dp, kdec, zc, prompt[i], i, i + 1, out_ids, NULL))
                return 1;
            for (int r = 1; r < N; r++) if (out_ids[r] != out_ids[0]) mismatches++;
            best = out_ids[0];
        }
        printf("generated:");
        int ctx = n_prompt;
        for (int s = 0; s < ngen && ctx < mctx; s++) {
            if (decode_step(h, devs, N, &B, dp, kdec, zc, best, ctx, ctx + 1, out_ids, NULL))
                return 1;
            for (int r = 1; r < N; r++) if (out_ids[r] != out_ids[0]) {
                mismatches++;
                printf("\n  RANK DISAGREE at step %d: rank0=%d rank%d=%d", s, out_ids[0], r, out_ids[r]);
            }
            best = out_ids[0];
            printf(" %d", best); fflush(stdout);
            ctx++;
            if (best == 1 || best == 106 || best == 50) break;   /* Gemma EOS */
        }
        printf("\n");
        /* device==host argmax check on rank 0's logits */
        uint16_t* logit = plow_hsa_alloc_host(h, (size_t)vocab * 2);
        plow_hsa_copy_d2h(h, devs[0].id, logit, devs[0].tens[devs[0].t_logits], (size_t)vocab * 2);
        int hb = 0; float hv = -1e30f;
        for (uint32_t v = 0; v < vocab; v++) { float x = b2f(logit[v]); if (x > hv) { hv = x; hb = (int)v; } }
        printf("VERIFY: rank-argmax mismatches=%d | rank0 device-argmax=%d host-argmax=%d -> %s\n",
               mismatches, best, hb, (mismatches == 0) ? "N-DEVICE LAUNCH OK (ranks agree, device==host)"
                                                       : "MISMATCH");
    }

    /* ================= SWEEP: 1-token decode, TP x ctx ================= */
    if (sweep_list) {
        int ctxs[16]; int nc = 0;
        char buf[256]; snprintf(buf, sizeof buf, "%s", sweep_list);
        for (char* t = strtok(buf, ","); t && nc < 16; t = strtok(NULL, ",")) ctxs[nc++] = parse_ctx(t);
        printf("SWEEP (decode-only, 1 tok, median of %d): TP=%d\n", steps, N);
        printf("  %-8s %12s %10s\n", "ctx", "ms/tok", "tok/s");
        for (int c = 0; c < nc; c++) {
            int ctx = ctxs[c];
            if (ctx > mctx) { printf("  %-8d   (exceeds pkt max_ctx %d — recompile pkt)\n", ctx, mctx); continue; }
            if (ctx >= mctx) ctx = mctx - 1;  /* pkt holds positions 0..max_ctx-1 */
            double* samp = malloc((size_t)steps * sizeof(double));
            /* warm */
            for (int w = 0; w < 2; w++) decode_step(h, devs, N, &B, dp, kdec, zc, 42, ctx, ctx + 1, out_ids, NULL);
            for (int s = 0; s < steps; s++) {
                double ms = 0;
                decode_step(h, devs, N, &B, dp, kdec, zc, 42, ctx, ctx + 1, out_ids, &ms);
                samp[s] = ms;
            }
            qsort(samp, (size_t)steps, sizeof(double), cmp_dbl);
            double med = samp[steps / 2];
            printf("  %-8d %12.3f %10.1f\n", ctx, med, 1000.0 / med);
            free(samp);
            /* STEP 1: dump rank-0's per-packet trace for this ctx (decode_critpath.py input). */
            if (trace_raw && devs[0].d_trace) {
                uint32_t nrec = B.prog[dp].h.n_stream;
                PlowTraceRec* tr = plow_hsa_alloc_host(h, (size_t)nrec * sizeof(PlowTraceRec));
                plow_hsa_copy_d2h(h, devs[0].id, tr, devs[0].d_trace,
                                  (size_t)nrec * sizeof(PlowTraceRec));
                char fn[512]; snprintf(fn, sizeof fn, "%s.tp%d.ctx%d.bin", trace_raw, N, ctx);
                FILE* tf = fopen(fn, "wb");
                if (tf) { fwrite(tr, sizeof(PlowTraceRec), nrec, tf); fclose(tf);
                    printf("    raw trace -> %s (%u recs, ms=%.3f)\n", fn, nrec, med); }
                plow_hsa_free(h, tr);
            }
        }
    }

    plow_hsa_shutdown(h);
    return 0;
}
