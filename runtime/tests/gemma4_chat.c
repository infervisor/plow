/* gemma4_chat.c — the closed loop. Prefill + KV-cached decode for Gemma 4 on one GPU.
 *
 *   plowc gemma4 <model-dir> <max_ctx> model.pkt     compile: prefill buckets + decode
 *   gemma4_chat model.pkt <model-dir> prompt.ids N   run
 *
 * Two persistent interpreters (one code object per bucket -- prefill carries the MFMA GEMM
 * and flash-prefill, decode carries the bandwidth-bound GEMV and flash-decode). Weights are
 * bound ONCE, by name, and both phases address the same KV cache.
 *
 * A decode step is ONE launch. The only thing that changes per step is:
 *   ids[0]    the token we just produced
 *   pos[0]    its position (feeds RoPE)
 *   kvlen[0]  how much cache flash_decode should read
 *   i[3] of the k/v HeadNormRope packets -- the cache row to append at
 * Everything else is already a tensor. That last one is an immediate, so the compiler hands
 * us the list of instructions to patch (`kv_row_insts`) and we rewrite exactly those.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_blob.h"

#include <dirent.h>
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

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
/* MEASUREMENT (runtime-launch-verify): host-side accounting of segmented dispatch.
 * g_seg_enq_us  = wall time the host spends WRITING all n_seg AQL packets into the queue
 *                 (should be tiny — proves the host is not in the loop between segments).
 * g_seg_drain_us = wall time in the single post-loop plow_hsa_wait (the GPU running every
 *                 segment back to back, chained by the AQL barrier bit + agent fences). */
static double g_seg_enq_us = 0, g_seg_drain_us = 0;
static long   g_seg_launches = 0, g_runseg_calls = 0;
static float b2f(uint16_t v) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)v << 16;
    return c.f;
}

/* ---- blob: parsed through the SHARED structs in dev_blob.h, never hand-rolled ----
 *
 * This used to be a hand-written parser with its own copy of the field layout. It broke
 * twice: once when the name field grew (segfault) and once when an init offset was added
 * (silent misparse -> a stale program ran against a fresh interpreter and the model spoke
 * confident nonsense). dev_abi.rs now compiles dev_blob.h and asserts every offset against
 * the Rust writer, so the two cannot drift. */
typedef struct {
    PlowProgHeader h;
    PlowDevInst* insts;
    PlowStreamEnt* stream;
    uint32_t *stream_ofs, *stream_len, *succs;
    PlowWait* waits;
    void *d_inst, *d_stream, *d_sofs, *d_slen, *d_waits, *d_succs, *d_ctr;
    /* Segmented dispatch (derived from the stream, not stored in the blob): number of wave-class
     * segments and the wave count (8 or 4) of each. The host relaunches the interpreter once per
     * segment on the matching code object. See the design notes */
    uint32_t n_seg;
    uint8_t seg_class[512];
    /* GLOBAL-QUEUE (Experiment E1): op-major stream + per-segment window bounds + per-segment
     * fetch-add cursor lines. Present only if the pkt carries the "GQ01" appendix; else NULL. */
    PlowStreamEnt* gq_stream;
    uint32_t* gq_seg_ofs;
    uint32_t gq_n_seg;
    void *d_gq_stream, *d_gq_seg, *d_gq_cursor;
} Prog;
typedef struct {
    PlowBlobHeader h;
    PlowTensorDecl* tensors;
    uint8_t* init;
    uint32_t* kvrow;
    Prog* prog;
} Blob;

static int load_blob(const char* path, Blob* b) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* p = malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
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
    /* GLOBAL-QUEUE appendix (Experiment E1): when the header flags the stream global-queue-capable,
     * a trailing "GQ01" section follows the programs — per program { n_seg:u32, gq_stream[n_stream],
     * gq_seg_ofs[n_seg+1] }. The header flag is authoritative; the "GQ01" tag only guards against a
     * truncated/corrupt appendix. A static-only blob leaves gq_stream NULL and PLOW_GLOBAL_QUEUE off. */
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
    return 0;
}

/* ---- safetensors: mmap the shards, find a tensor's byte range in the JSON header ---- */
#define MAX_SHARD 128
typedef struct {
    int n;
    uint8_t* base[MAX_SHARD];
    char* hdr[MAX_SHARD];
    size_t hdr_len[MAX_SHARD];
    uint64_t data0[MAX_SHARD];
} Safet;

/* Parse "model-{i}-of-{n}[.partial].safetensors". Returns 1 on match, filling idx/tot/part.
 * The suffix must be EXACTLY ".safetensors" so sidecars like
 * "model-00001-of-00002.safetensors.header.json" (present in the 31B partial dir) don't match.
 * Digit widths are not fixed at 5 -- they're parsed as integers, so there is no shard ceiling
 * baked into the pattern. Mirrors crates/plowrt/src/memory/container.rs::parse_shard_name. */
static int st_parse_name(const char* f, int* idx, int* tot, int* part) {
    if (strncmp(f, "model-", 6)) return 0;
    const char* p = f + 6;
    char* e;
    long i = strtol(p, &e, 10);
    if (e == p || strncmp(e, "-of-", 4)) return 0;
    p = e + 4;
    long t = strtol(p, &e, 10);
    if (e == p) return 0;
    if (!strcmp(e, ".safetensors")) *part = 0;
    else if (!strcmp(e, ".partial.safetensors")) *part = 1;
    else return 0;
    *idx = (int)i; *tot = (int)t;
    return 1;
}

static int st_mmap_one(Safet* s, const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { printf("st_open: cannot open %s\n", path); return 1; }
    struct stat st;
    if (fstat(fd, &st) || (size_t)st.st_size < 8) { close(fd); printf("st_open: %s too short\n", path); return 1; }
    uint8_t* m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { printf("st_open: mmap failed for %s\n", path); return 1; }
    uint64_t hn = *(uint64_t*)m;
    if (8 + hn > (uint64_t)st.st_size) { printf("st_open: %s header len %llu exceeds file\n", path, (unsigned long long)hn); return 1; }
    s->base[s->n] = m; s->hdr[s->n] = (char*)(m + 8);
    s->hdr_len[s->n] = (size_t)hn; s->data0[s->n] = 8 + hn;
    s->n++;
    return 0;
}

/* Discover and mmap a checkpoint's shards.
 *
 * FIXED (was two bugs): the old probe called access() on "model-00001-of-%05d.safetensors" for
 * total in 1..8 only. That (a) capped the shard count at 8 and (b) never matched the
 * ".partial.safetensors" naming of the Gemma-4 31B partial checkpoint, so that dir loaded ZERO
 * shards; and the single-file "model.safetensors" fallback was documented here but never
 * written, so Gemma-4 12B (one unsharded 23 GB file, no index) could not load at all.
 *
 * Now: one readdir pass. A complete non-partial set wins; a partial set is used when that's all
 * there is; single-file "model.safetensors" is the last resort. Anything ambiguous (a hole in
 * the set, or two different "-of-N" totals) is a hard failure that names what it saw, because a
 * silently-absent weight still generates fluent text. */
static int st_open(Safet* s, const char* dir) {
    s->n = 0;
    DIR* d = opendir(dir);
    if (!d) { printf("st_open: cannot opendir %s\n", dir); return 1; }
    /* files[part][total][index] -> seen; keep totals we observe */
    /* Just the dirent name (NAME_MAX+1); the dir prefix is joined at use. */
    static char names[2][MAX_SHARD + 1][MAX_SHARD + 1][256];
    static int  seen [2][MAX_SHARD + 1];
    /* st_open is called more than once per process (model dir, then the fp8 dir), so these
     * must be cleared every call -- stale entries would fabricate a "complete" set. */
    memset(names, 0, sizeof(names));
    memset(seen, 0, sizeof(seen));
    int have_single = 0, totals[2][MAX_SHARD + 1];
    memset(totals, 0, sizeof(totals));
    struct dirent* de;
    while ((de = readdir(d))) {
        if (!strcmp(de->d_name, "model.safetensors")) { have_single = 1; continue; }
        int i, t, part;
        if (!st_parse_name(de->d_name, &i, &t, &part)) continue;
        if (t < 1 || t > MAX_SHARD || i < 1 || i > t) {
            printf("st_open: ignoring %s (shard %d of %d outside 1..%d)\n", de->d_name, i, t, MAX_SHARD);
            continue;
        }
        if (!names[part][t][i][0]) { snprintf(names[part][t][i], 256, "%s", de->d_name); seen[part][t]++; }
        totals[part][t] = 1;
    }
    closedir(d);

    /* Collect complete sets. A partial set is a subset by construction, so a complete
     * non-partial set of the SAME total wins over it; anything else with more than one
     * complete set is ambiguous and must not be resolved by guessing. */
    int n_full = 0, n_part = 0, tot_full = -1, tot_part = -1;
    for (int t = 1; t <= MAX_SHARD; t++) {
        if (totals[0][t] && seen[0][t] == t) { n_full++; tot_full = t; }
        if (totals[1][t] && seen[1][t] == t) { n_part++; tot_part = t; }
    }
    int chosen_part = -1, chosen_tot = -1;
    if (n_full == 1 && (n_part == 0 || (n_part == 1 && tot_part == tot_full))) { chosen_part = 0; chosen_tot = tot_full; }
    else if (n_full == 0 && n_part == 1) { chosen_part = 1; chosen_tot = tot_part; }
    else if (n_full + n_part > 1) {
        printf("st_open: %s is ambiguous -- %d complete shard set(s) (%d full, %d partial); a stray "
               "shard-named file silently changes what loads\n", dir, n_full + n_part, n_full, n_part);
        return 1;
    }
    if (chosen_tot > 0) {
        for (int i = 1; i <= chosen_tot; i++) {
            char p[512];
            snprintf(p, sizeof(p), "%s/%s", dir, names[chosen_part][chosen_tot][i]);
            if (st_mmap_one(s, p)) return 1;
        }
        return 0;
    }
    /* Nothing complete: if we saw a partial set at all, name the holes rather than fall through. */
    for (int part = 0; part < 2; part++)
        for (int t = 1; t <= MAX_SHARD; t++)
            if (totals[part][t]) {
                printf("st_open: %s has an incomplete shard set (-of-%05d%s): %d of %d present, missing",
                       dir, t, part ? " .partial" : "", seen[part][t], t);
                for (int i = 1; i <= t; i++) if (!names[part][t][i][0]) printf(" %d", i);
                printf("\n");
                return 1;
            }
    /* THE single-file fallback that was documented but missing. */
    if (have_single) {
        char p[512];
        snprintf(p, sizeof(p), "%s/model.safetensors", dir);
        return st_mmap_one(s, p);
    }
    printf("st_open: %s holds no safetensors checkpoint (looked for "
           "model-{i}-of-{n}[.partial].safetensors and model.safetensors)\n", dir);
    return 1;
}
static const uint8_t* st_find(Safet* s, const char* name, uint64_t* nb) {
    char key[256];
    int kl = snprintf(key, sizeof(key), "\"%s\":", name);
    for (int i = 0; i < s->n; i++) {
        const char* h = s->hdr[i];
        const char* end = h + s->hdr_len[i];
        const char* p = NULL;
        for (const char* c = h; c + kl <= end; c++)
            if (!memcmp(c, key, (size_t)kl)) { p = c + kl; break; }
        if (!p) continue;
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

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: %s model.pkt <model-dir> prompt.ids [n_gen]\n", argv[0]);
        return 1;
    }
    const int n_gen = argc > 4 ? atoi(argv[4]) : 64;
    Blob B;
    if (load_blob(argv[1], &B)) return 1;

    /* Derive per-program segment metadata from the stream's seg tags (plowc wrote them; they are
     * not a separate blob field). n_seg = max seg + 1; a segment is wave-class 4 iff it holds a
     * FlashPrefill, else 8. See the design notes */
    for (uint32_t pi = 0; pi < B.h.n_prog; pi++) {
        Prog* g = &B.prog[pi];
        uint32_t ns = 1;
        for (uint32_t j = 0; j < g->h.n_stream; j++)
            if ((uint32_t)g->stream[j].seg + 1 > ns) ns = g->stream[j].seg + 1;
        if (ns > 512) ns = 512;
        g->n_seg = ns;
        for (uint32_t s = 0; s < ns; s++) g->seg_class[s] = 8;
        /* A segment is wave-class 4 iff EVERY op in it is a flash_prefill, in either precision.
         *
         * "iff it HOLDS a flash_prefill" is what this used to say, and it is a trapdoor. The
         * 4-wave object is compiled PLOW_BUCKET_FLASH: its whole body is
         * `if (op == FLASH_PREFILL{,_FP8}) ...` and there is no switch at all — every other
         * opcode is silently dropped. So classifying a MIXED segment as 4 runs its GEMMs, norms
         * and lm_head on an object that cannot dispatch them, and writes nothing.
         *
         * That is not hypothetical. A packet whose `seg` tags are all 0 (plowc emitted exactly
         * that after a regression — measured on a 2026-07-27 build) puts the entire prefill
         * program in segment 0 alongside its flash packets, so the whole program was classified
         * 4 and ran on the flash-only object: prefill "completed" in 8.7 ms instead of 72.1 ms
         * and act.logits was all zero, which reads as a numerics bug and is a dispatch bug.
         * With `all` the same packet is correct (verified: logit 20.75, same token ids as a
         * properly segmented build) — just unsegmented, i.e. slower, which is what an
         * unsegmented packet should cost.
         *
         * Testing only the bf16 opcode was the earlier version of this same mistake: it left
         * every fp8-KV packet's flash segments classified 8, so they ran on the 8-wave
         * interpreter without the 512-register budget the D=512 Q-hoist is built for (its 228
         * spills are the deliberate trade; the 8-wave object caps at 256).
         *
         * NOTE the MLA prefill ops (51/55) are deliberately NOT flash here — they run on the
         * 8-wave prefill_mla object, and interp_flash dispatches only FLASH_PREFILL{,_FP8}. */
        for (uint32_t s = 0; s < ns; s++) g->seg_class[s] = 0; /* 0 = "nothing seen yet" */
        for (uint32_t j = 0; j < g->h.n_stream; j++) {
            const uint16_t o = g->insts[g->stream[j].inst].op;
            const uint16_t s = g->stream[j].seg;
            if (s >= 512) continue;
            const uint8_t want =
                (o == PLOW_DOP_FLASH_PREFILL || o == PLOW_DOP_FLASH_PREFILL_FP8) ? 4 : 8;
            /* 8 wins: one non-flash op in the segment forces the general interpreter. */
            if (g->seg_class[s] == 0 || want == 8) g->seg_class[s] = want;
        }
        for (uint32_t s = 0; s < ns; s++)
            if (g->seg_class[s] == 0) g->seg_class[s] = 8; /* empty segment: harmless either way */
        /* MEASUREMENT: PLOW_SEG_OFF collapses every entry into segment 0 -> ONE launch per chunk
         * (the pre-segmentation baseline: all ops on the 8-wave interp, in emit order — identical
         * numerics). The A/B against the default gives the pure per-segment-transition cost.
         *
         * IT REWRITES ONLY THE STATIC STREAM. The global queue does not read stream[].seg at all;
         * it bounds each launch by gq_seg_ofs[cur_seg..cur_seg+1] over the op-major gq_stream,
         * which this cannot rewrite (collapsing it would mean re-deriving the op-major order).
         * So SEG_OFF + GQ ran ONE SEGMENT'S WORTH of packets and reported a time for it: measured
         * 0.9 ms for a 164 ms prefill, i.e. it looked like a 180x speedup and had simply done
         * almost nothing. It is refused below rather than silently truncating. */
        if (getenv("PLOW_SEG_OFF")) {
            for (uint32_t j = 0; j < g->h.n_stream; j++) g->stream[j].seg = 0;
            g->n_seg = 1;
        }
    }
    Safet S;
    if (st_open(&S, argv[2])) { printf("no safetensors\n"); return 1; }
    /* FP8 decode: the quantized projection weights + scales load from PLOW_FP8_DIR (the fp8
     * checkpoint), leaving the primary dir (argv[2]) as the bf16 source for prefill. Opened only
     * when the pkt actually declares "fp8/"-prefixed tensors; harmless otherwise. */
    Safet Sf; int have_fp8 = 0;
    { const char* fd = getenv("PLOW_FP8_DIR");
      if (fd && !st_open(&Sf, fd)) { have_fp8 = 1; printf("fp8 weights dir: %s\n", fd); } }

    FILE* pf = fopen(argv[3], "rb");
    if (!pf) { printf("no %s\n", argv[3]); return 1; }
    fseek(pf, 0, SEEK_END); long pn = ftell(pf); fseek(pf, 0, SEEK_SET);
    int n_prompt = (int)(pn / 4);
    int32_t* prompt = malloc((size_t)pn + 4 * (size_t)n_gen);
    if (fread(prompt, 1, (size_t)pn, pf) != (size_t)pn) return 1;
    fclose(pf);

    plow_hsa* h = plow_hsa_init();
    if (!h) return 1;
    char gfx[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, gfx, &cus, &lds);

    /* Two interpreters, two code objects, two SYMBOLS. The prefill bucket carries the MFMA
     * GEMM + flash-prefill; the decode bucket carries the bandwidth-bound GEMV +
     * flash-decode. They are separate kernels because register allocation is per-kernel and
     * carrying both op sets costs the 8-wave dispatch. */
    plow_hsa_kernel k_pre, k_dec, k_flash;
    /* SCHEDULER — global queue (one shared atomic cursor over the op-major gq_stream) vs static per-CU
     * streams. Op kernels/tiles/registers are byte-identical, so it's a pure scheduling change, bit-exact
     * vs static. Selectable PER PHASE because prefill and decode are separate kernels/objects and the E1
     * data splits by phase: GQ wins decode on both models, but prefill is model-dependent (neutral on the
     * 31B, +2-3% on the 4B). So static-prefill + GQ-decode is a valid hybrid — kept selectable pending
     * more runs to settle the default.
     *   default:            both GQ
     *   PLOW_STATIC=1 / PLOW_GLOBAL_QUEUE=0 : both static
     *   PLOW_STATIC_PREFILL=1               : static prefill, GQ decode (the hybrid)
     *   PLOW_STATIC_DECODE=1                : GQ prefill, static decode
     * Auto-falls-back to static per phase if the pkt lacks the GQ stream (F_GQ) or the _gq object is absent. */
    int gq_pre = 1, gq_dec = 1;
    { const char* e = getenv("PLOW_GLOBAL_QUEUE"); if (e) { int v = atoi(e) != 0; gq_pre = v; gq_dec = v; } }
    if (getenv("PLOW_STATIC"))         { gq_pre = 0; gq_dec = 0; }
    if (getenv("PLOW_STATIC_PREFILL")) gq_pre = 0;
    if (getenv("PLOW_STATIC_DECODE"))  gq_dec = 0;
    if ((gq_pre || gq_dec) && !(B.h.flags & PLOW_BLOB_F_GQ)) { gq_pre = gq_dec = 0; printf("pkt has no GQ stream -> static\n"); }
    /* PLOW_SEG_OFF only collapses the STATIC stream (see above), so combining it with the global
     * queue measures one segment and calls it a prefill. Refuse: a benchmark that quietly does
     * less work is worse than one that does not run. PLOW_STATIC=1 is the supported pairing. */
    if (getenv("PLOW_SEG_OFF") && (gq_pre || gq_dec)) {
        printf("PLOW_SEG_OFF is a STATIC-scheduler measurement and cannot collapse the global\n"
               "queue's gq_seg_ofs window — it would run one segment and report it as the whole\n"
               "prefill. Re-run with PLOW_STATIC=1 (or PLOW_STATIC_PREFILL=1).\n");
        return 1;
    }
    if (gq_pre && access("interp_prefill_gq.elf", R_OK) != 0) { gq_pre = 0; printf("no prefill _gq object -> static prefill\n"); }
    if (gq_dec && access("interp_decode_gq.elf",  R_OK) != 0) { gq_dec = 0; printf("no decode _gq object -> static decode\n"); }
    printf("scheduler: prefill=%s decode=%s\n", gq_pre ? "GLOBAL QUEUE" : "static", gq_dec ? "GLOBAL QUEUE" : "static");
    /* FP8 decode object: if any program emits a GEMV_FP8 packet, the decode phase must run the
     * SEPARATE interp_decode_fp8[_gq].elf (fp8 w8a16 GEMV arms in place of the bf16 GEMV_GLU/QKV).
     * The exported symbol is unchanged (only one decode object is ever loaded). */
    int is_fp8_pkt = 0;
    for (uint32_t pi = 0; pi < B.h.n_prog && !is_fp8_pkt; pi++)
        for (uint32_t j = 0; j < B.prog[pi].h.n_inst; j++)
            if (B.prog[pi].insts[j].op == PLOW_DOP_GEMV_FP8) { is_fp8_pkt = 1; break; }
    if (is_fp8_pkt) printf("fp8 weights: interp_{prefill,decode}_fp8%s.elf\n", gq_dec ? "_gq" : "");
    /* FP8 KV-CACHE object: an fp8-KV pkt emits FLASH_DECODE_FP8 (K/V stored+read as e4m3). It needs
     * BOTH interpreter objects rebuilt with the fp8 flash + HeadNormRopeFp8 arms — the decode object
     * that reads the fp8 cache AND the prefill object that fills it. These _fp8kv objects supersede
     * the plain/fp8 selection above. (An fp8-KV pkt is also fp8-weight, so the decode object still
     * carries the fp8 GEMV arms.) */
    int is_fp8kv_pkt = 0;
    for (uint32_t pi = 0; pi < B.h.n_prog && !is_fp8kv_pkt; pi++)
        for (uint32_t j = 0; j < B.prog[pi].h.n_inst; j++)
            if (B.prog[pi].insts[j].op == PLOW_DOP_FLASH_DECODE_FP8) { is_fp8kv_pkt = 1; break; }
    if (is_fp8kv_pkt) printf("fp8 KV-cache: interp_{prefill,decode}_fp8kv%s.elf\n", gq_dec ? "_gq" : "");
    /* THE PREFILL OBJECT MUST FOLLOW THE WEIGHT PRECISION TOO, not just the KV precision. This
     * used to select only between fp8kv and bf16, so a w8a8 packet (fp8 weights, bf16 KV — the
     * PLOW_FP8=1 profile) was handed interp_prefill.elf, which carries no GEMM_FP8 /
     * GEMM_MED_FP8 / GEMM_SMALL_FP8 / GEMM_GLU_FP8 / QUANT_FP8 arms at all. Those opcodes would
     * fall through the interpreter's switch to `default:` and write NOTHING — a silently zeroed
     * prefill, the same failure mode as the unreachable-op-93 arm and the misclassified fp8
     * flash segment. Verified in the objects: interp_prefill.elf has 0 v_mfma_f32_32x32x64_f8f6f4
     * and interp_prefill_fp8.elf has them. */
    const char* elfs[2] = {
        is_fp8kv_pkt ? (gq_pre ? "interp_prefill_fp8kv_gq.elf" : "interp_prefill_fp8kv.elf")
        : is_fp8_pkt ? (gq_pre ? "interp_prefill_fp8_gq.elf" : "interp_prefill_fp8.elf")
                     : (gq_pre ? "interp_prefill_gq.elf" : "interp_prefill.elf"),
        is_fp8kv_pkt ? (gq_dec ? "interp_decode_fp8kv_gq.elf" : "interp_decode_fp8kv.elf")
        : is_fp8_pkt ? (gq_dec ? "interp_decode_fp8_gq.elf" : "interp_decode_fp8.elf")
                     : (gq_dec ? "interp_decode_gq.elf"  : "interp_decode.elf") };
    const char* syms[2] = { gq_pre ? "plow_interp_gfx950_gq"     : "plow_interp_gfx950",
                            gq_dec ? "plow_interp_dec_gfx950_gq" : "plow_interp_dec_gfx950" };
    plow_hsa_kernel* ks[2] = {&k_pre, &k_dec};
    for (int i = 0; i < 2; i++) {
        FILE* f = fopen(elfs[i], "rb");
        if (!f) { printf("missing %s\n", elfs[i]); return 1; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        void* co = malloc((size_t)n);
        if (fread(co, 1, (size_t)n, f) != (size_t)n) return 1;
        fclose(f);
        if (plow_hsa_load_code_object(h, 0, co, (size_t)n)) { printf("load %s failed\n", elfs[i]); return 1; }
        if (plow_hsa_get_kernel(h, 0, syms[i], ks[i])) { printf("no %s in %s\n", syms[i], elfs[i]); return 1; }
    }
    /* Thread count the flash code object was built for: 256 (4-wave Gemma D=256/512) or 512
     * (8-wave Llama/Qwen D=128). head_dim 128 is compiled as a D=128-only 8-wave object (see
     * build.sh / PLOW_FLASH_HD128), so pick 512 automatically when the model's flash op is
     * head_dim 128; PLOW_FLASH_THREADS overrides. Must match how interp_flash.elf was built. */
    uint32_t g_flash_threads = 256u;
    for (uint32_t pi = 0; pi < B.h.n_prog && g_flash_threads == 256u; pi++) {
        Prog* g = &B.prog[pi];
        for (uint32_t j = 0; j < g->h.n_inst; j++)
            if ((g->insts[j].op == PLOW_DOP_FLASH_PREFILL ||
                 g->insts[j].op == PLOW_DOP_FLASH_PREFILL_FP8) && g->insts[j].i[6] == 128) {
                g_flash_threads = 512u; break;
            }
    }
    { const char* e = getenv("PLOW_FLASH_THREADS"); if (e) g_flash_threads = (uint32_t)atoi(e); }
    /* Optional flash code object for segmented dispatch. If absent, the segment
     * loop runs every segment on the 8-wave interpreter (correct, just no flash speedup). */
    /* THE FLASH OBJECT FOLLOWS THE KV PRECISION, exactly as the prefill/decode objects do. It
     * did not: this always loaded the bf16 `interp_flash.elf`, whose only dispatch is
     * `if (in->op == PLOW_DOP_FLASH_PREFILL)` — the fp8-KV twin is built with the FP8 arm
     * SWAPPED IN, not added. So on an fp8-KV packet (which emits FLASH_PREFILL_FP8 and never
     * the bf16 opcode) every class-4 segment fell through and wrote NOTHING: all attention
     * output zero, silently. The cmake table has carried the `interp_flash_fp8kv` row and
     * scripts/gfx950_objects.py has selected it since it landed; only this driver did not.
     * (plowrt's Rust engine already selects it — exec/amd.rs object_name(Phase::Flash, ...).)
     * Same shape as the class-4 wave-class bug: an arm that exists, is correct, and has
     * nothing routing to it. */
    plow_hsa_kernel* k_flash_p = NULL;
    { const char* felf = is_fp8kv_pkt ? (gq_pre ? "interp_flash_fp8kv_gq.elf" : "interp_flash_fp8kv.elf")
                                      : (gq_pre ? "interp_flash_gq.elf" : "interp_flash.elf"); /* flash is a prefill segment */
      const char* fsym = gq_pre ? "plow_interp_flash_gfx950_gq" : "plow_interp_flash_gfx950";
      FILE* f = fopen(felf, "rb");
      if (f) {
          fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
          void* co = malloc((size_t)n);
          if (fread(co, 1, (size_t)n, f) == (size_t)n && !plow_hsa_load_code_object(h, 0, co, (size_t)n)
              && !plow_hsa_get_kernel(h, 0, fsym, &k_flash)) {
              k_flash_p = &k_flash;
              printf("segmented dispatch: 4-wave flash code object loaded (%s)\n", felf);
          }
          fclose(f);
      } else {
          /* LOUD. An fp8-KV packet whose class-4 twin is missing runs its flash on the 8-wave
           * interpreter — which for fp8-KV is correct but register-starved, and if the whole
           * file is absent it is not obvious from the timings alone. */
          printf("no %s -> flash segments run on the 8-wave interpreter\n", felf);
      } }

    printf("dev0: %s CUs=%u\n", gfx, cus);
    printf("%u programs, %u tensors, %u kv-row patch sites\n", B.h.n_prog, B.h.n_tensor, B.h.n_kvrow);

    /* ---- bind ---- */
    const size_t STAGE = 64u << 20;
    void* stage = plow_hsa_alloc_host(h, STAGE);
    void** dev = calloc(B.h.n_tensor, sizeof(void*));
    int t_ids = -1, t_pos = -1, t_kvlen = -1, t_logits = -1;
    uint64_t wb = 0, kvb = 0;
    int nw = 0;
    const double lt0 = now();
    for (uint32_t i = 0; i < B.h.n_tensor; i++) {
        PlowTensorDecl* td = &B.tensors[i];
        dev[i] = plow_hsa_alloc(h, 0, td->bytes);
        if (!dev[i]) { printf("VRAM alloc failed: %s (%llu B)\n", td->name, (unsigned long long)td->bytes); return 1; }
        if (!strcmp(td->name, "in.ids")) t_ids = (int)i;
        if (!strcmp(td->name, "in.pos")) t_pos = (int)i;
        if (!strcmp(td->name, "in.kvlen")) t_kvlen = (int)i;
        if (!strcmp(td->name, "act.logits")) t_logits = (int)i;
        if (!strncmp(td->name, "kv.", 3)) kvb += td->bytes;
        /* A WEIGHT is anything bound from the checkpoint by name: "model.*" for every arch, plus
         * Llama's untied top-level "lm_head.weight". Inputs (in.*), the KV cache (kv.*) and
         * activations (act.*) are not.
         *
         * FP8 DECODE (PLOW_FP8): the quantized projection twins are declared under an "fp8/" name
         * prefix and bound from a SEPARATE fp8 checkpoint dir (env PLOW_FP8_DIR); the "fp8/" is
         * stripped before the st_find key lookup. The plain bf16 "model.*" weights still come from
         * the primary dir (argv[2]) and feed prefill's GEMM. This is the mixed bf16-prefill /
         * fp8-decode load the design calls for. */
        int is_fp8 = !strncmp(td->name, "fp8/", 4);
        if (is_fp8 && !have_fp8) { printf("fp8 pkt needs PLOW_FP8_DIR (missing %s)\n", td->name); return 1; }
        if (!strncmp(td->name, "model.", 6) || !strncmp(td->name, "lm_head", 7) || is_fp8) {
            uint64_t got = 0;
            /* THE TWO SIDES DISAGREE ON WHETHER THE "fp8/" PREFIX IS PART OF THE KEY, so accept
             * both. The emitter declares the twin as `fp8/<name>`; this loader stripped the
             * prefix before the lookup, while perf-data/harness/quantize_fp8.py writes the key
             * WITH it ("keyed EXACTLY as the emitter declares the twins"). Neither convention is
             * obviously canonical and picking one silently would just move the failure — so try
             * the declared name first, then the stripped one, and report BOTH spellings if
             * neither is present. A checkpoint keyed either way now loads.
             *
             * RESOLVED: the README now specifies the key VERBATIM INCLUDING the prefix on all
             * three sides (emitter declaration, quantizer output, loader). The stripped-name
             * fallback is kept only so a checkpoint generated before that ruling still loads;
             * it can be deleted once none are left. The verbatim form is tried first, so the
             * fallback never shadows a correct checkpoint. */
            const char* key = is_fp8 ? td->name + 4 : td->name;
            const uint8_t* src = st_find(is_fp8 ? &Sf : &S, td->name, &got);
            if (!src && is_fp8) src = st_find(&Sf, key, &got);
            if (!src) { printf("MISSING WEIGHT: %s%s%s\n", is_fp8 ? "[fp8] " : "", td->name,
                               is_fp8 ? " (tried both with and without the \"fp8/\" prefix)" : "");
                        return 1; }
            if (got != td->bytes) { printf("SIZE MISMATCH %s (want %llu got %llu)\n", td->name,
                                           (unsigned long long)td->bytes, (unsigned long long)got); return 1; }
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, src + o, n);
                plow_hsa_copy_h2d(h, 0, (uint8_t*)dev[i] + o, stage, n);
            }
            wb += td->bytes; nw++;
        } else if (td->init_off != PLOW_INIT_NONE) {
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, B.init + td->init_off + o, n);
                plow_hsa_copy_h2d(h, 0, (uint8_t*)dev[i] + o, stage, n);
            }
        }
    }
    void* d_tens = plow_hsa_alloc(h, 0, (size_t)B.h.n_tensor * sizeof(void*));
    plow_hsa_upload(h, 0, d_tens, dev, (size_t)B.h.n_tensor * sizeof(void*));
    printf("bound %d weights (%.1f GiB) + %.2f GiB KV cache in %.1f s\n", nw,
           wb / 1073741824.0, kvb / 1073741824.0, now() - lt0);

    /* ---- upload every program's tables ---- */
    for (uint32_t i = 0; i < B.h.n_prog; i++) {
        Prog* g = &B.prog[i];
        g->d_inst = plow_hsa_alloc(h, 0, (size_t)g->h.n_inst * sizeof(PlowDevInst));
        g->d_stream = plow_hsa_alloc(h, 0, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
        g->d_sofs = plow_hsa_alloc(h, 0, (size_t)B.h.n_cu * 4);
        g->d_slen = plow_hsa_alloc(h, 0, (size_t)B.h.n_cu * 4);
        g->d_waits = plow_hsa_alloc(h, 0, (size_t)(g->h.n_wait ? g->h.n_wait : 1) * sizeof(PlowWait));
        g->d_succs = plow_hsa_alloc(h, 0, (size_t)(g->h.n_succ ? g->h.n_succ : 1) * 4);
        g->d_ctr = plow_hsa_alloc(h, 0, (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4);
        plow_hsa_upload(h, 0, g->d_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst));
        plow_hsa_upload(h, 0, g->d_stream, g->stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
        plow_hsa_upload(h, 0, g->d_sofs, g->stream_ofs, (size_t)B.h.n_cu * 4);
        plow_hsa_upload(h, 0, g->d_slen, g->stream_len, (size_t)B.h.n_cu * 4);
        if (g->h.n_wait) plow_hsa_upload(h, 0, g->d_waits, g->waits, (size_t)g->h.n_wait * sizeof(PlowWait));
        if (g->h.n_succ) plow_hsa_upload(h, 0, g->d_succs, g->succs, (size_t)g->h.n_succ * 4);
        /* GLOBAL-QUEUE tables (Experiment E1): op-major stream, per-segment window bounds, and one
         * fetch-add cursor line per segment (PLOW_CTR-strided, zeroed per launch like the counters). */
        if (g->gq_stream) {
            g->d_gq_stream = plow_hsa_alloc(h, 0, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
            g->d_gq_seg    = plow_hsa_alloc(h, 0, (size_t)(g->gq_n_seg + 1) * 4);
            g->d_gq_cursor = plow_hsa_alloc(h, 0, (size_t)(g->gq_n_seg ? g->gq_n_seg : 1) * PLOW_CTR_STRIDE * 4);
            plow_hsa_upload(h, 0, g->d_gq_stream, g->gq_stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt));
            plow_hsa_upload(h, 0, g->d_gq_seg, g->gq_seg_ofs, (size_t)(g->gq_n_seg + 1) * 4);
        }
    }

    /* THE PROMPT MUST FIT THE KV CACHE, NOT A SINGLE BUCKET.
     *
     * This used to demand a prefill bucket >= n_prompt, which was right before chunked prefill
     * and is wrong after it: a prompt is now fed as ceil(n/C) chunks of at most PLOW_MAX_CHUNK,
     * so the bucket ladder only has to reach MAX_CHUNK, not the context. Capping the ladder
     * there (every rung above it was dead code) made this legacy guard start rejecting every
     * prompt longer than one chunk.
     *
     * The real bound is the KV cache. `in.pos` is one i32 per context row and is the only
     * activation still sized by the CONTEXT rather than the chunk, so it carries that bound. */
    const int max_ctx = (int)(B.tensors[t_pos].bytes / 4);
    if (n_prompt > max_ctx) {
        printf("prompt (%d) exceeds max context (%d)\n", n_prompt, max_ctx);
        return 1;
    }
    /* Largest prefill bucket: the fallback program, and what PLOW_TRACE_PREFILL traces. */
    int bp = 0;
    for (uint32_t i = 0; i + 1 < B.h.n_prog; i++)
        if (B.prog[i].h.t > B.prog[bp].h.t) bp = (int)i;
    const int dp = (int)B.h.n_prog - 1;
    /* Vocab from the LOGITS TENSOR, not from "whatever the last instruction's i[0] happens to
     * be". It used to be the latter, which silently meant SOFTCAP -- until ARGMAX_FIN became
     * the last instruction and VOCAB quietly became 64 (its block count). The host then
     * argmaxed the first 64 logits, picked a nonsense token, and everything downstream looked
     * like a kernel numerics bug. Read the size of the thing you are reading. */
    const uint32_t VOCAB = (uint32_t)(B.tensors[t_logits].bytes / 2);
    printf("prompt %d tokens -> prefill bucket T=%u; decode program has %u packets\n\n",
           n_prompt, B.prog[bp].h.t, B.prog[dp].h.n_inst);

    /* Pinned staging for the per-step scalars and the patched decode instructions.
     *
     * Sized from the LARGEST BUCKET IN THE BLOB, not from a magic 4096. It was the latter, which
     * held only because the bucket ladder happened to stop at T=4096: raising max_ctx to 8192
     * made the prefill's `ids`/`pos` fill walk straight off the end of it. Same family as the
     * counter-zeroing buffer below, and the same failure mode — a hardcoded limit that is right
     * until the day the workload grows. */
    uint32_t max_t = 1;
    for (uint32_t i = 0; i < B.h.n_prog; i++)
        if (B.prog[i].h.t > max_t) max_t = B.prog[i].h.t;
    int32_t* h_scalar = plow_hsa_alloc_host(h, (size_t)max_t * 4);
    PlowDevInst* h_inst = plow_hsa_alloc_host(h, (size_t)B.prog[dp].h.n_inst * sizeof(PlowDevInst));
    memcpy(h_inst, B.prog[dp].insts, (size_t)B.prog[dp].h.n_inst * sizeof(PlowDevInst));
    /* PINNED. plow_hsa_copy_h2d's contract is that the host side comes from
     * plow_hsa_alloc_host: SDMA can only read pages ROCr has registered, and a plain
     * calloc() here faults the GPU on a host heap address. */
    /* Sized from the BLOB, not from a magic number. It used to be a flat 4096 counters, which
     * silently sufficed only because one-counter-per-op happened to stay under it. Per-slice
     * gates (PLOW_SE_FINE) push the decode program to ~16k counters, and the h2d below then
     * read 2 MB out of a 512 KB pinned buffer -- which presents as a GPU memory access fault
     * in the interpreter, i.e. it looks like a kernel bug and is a host allocation bug. */
    uint32_t max_ctr = 0;
    for (uint32_t i = 0; i < B.h.n_prog; i++)
        if (B.prog[i].h.n_counter > max_ctr) max_ctr = B.prog[i].h.n_counter;
    const size_t zc_bytes = (size_t)max_ctr * PLOW_CTR_STRIDE * 4;
    uint32_t* zc = plow_hsa_alloc_host(h, zc_bytes);
    memset(zc, 0, zc_bytes);
    /* PINNED, and allocated once. plow_hsa_download() pins the host buffer on every call
     * (hsa_amd_memory_lock is syscall-class) -- doing that per decode step cost more than the
     * whole forward pass. */
    uint16_t* logit = plow_hsa_alloc_host(h, (size_t)VOCAB * 2);
    const int do_trace = getenv("PLOW_TRACE") != NULL;
    /* PLOW_TRACE_PREFILL traces the PREFILL program instead of decode. Prefill had never been
     * profiled at all, which is how its GEMM epilogue kept storing two bytes at a time. */
    const int trace_prog = getenv("PLOW_TRACE_PREFILL") ? bp : dp;
    void* d_trace = do_trace
        ? plow_hsa_alloc(h, 0, (size_t)B.prog[trace_prog].h.n_stream * sizeof(PlowTraceRec))
        : NULL;

#define RUN(P, K)                                                                             \
    do {                                                                                       \
        Prog* g = &B.prog[P];                                                                  \
        PlowProgram pr;                                                                        \
        memset(&pr, 0, sizeof(pr));                                                            \
        pr.insts = g->d_inst; pr.stream = g->d_stream; pr.stream_ofs = g->d_sofs;              \
        pr.stream_len = g->d_slen; pr.waits = g->d_waits; pr.succs = g->d_succs;               \
        pr.counters = g->d_ctr; pr.tensors = (void* const*)d_tens;                              \
        pr.trace = ((P) == trace_prog) ? (PlowTraceRec*)d_trace : NULL;                          \
        pr.gq_stream = g->d_gq_stream; pr.gq_seg_ofs = g->d_gq_seg; pr.gq_cursor = g->d_gq_cursor; \
        plow_hsa_copy_h2d(h, 0, g->d_ctr, zc, (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4);                       \
        if (g->d_gq_cursor) plow_hsa_copy_h2d(h, 0, g->d_gq_cursor, zc,                          \
                            (size_t)(g->gq_n_seg ? g->gq_n_seg : 1) * PLOW_CTR_STRIDE * 4);       \
        if (plow_hsa_launch(h, 0, &(K), B.h.n_cu * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, \
                            0, &pr, sizeof(pr))) { printf("LAUNCH FAILED\n"); return 1; }       \
        plow_hsa_wait(h, 0);                                                                   \
    } while (0)

/* Segmented launch: relaunch the interpreter once per wave-class segment. Counters are zeroed ONCE
 * (not per segment) so a segment's producers — run in earlier launches — stay satisfied; the
 * per-segment host wait is the cross-segment barrier. Class-4 segments run on the 4-wave flash code
 * object at 256 threads; class-8 on the 8-wave interp at 512. K4P is a kernel* or NULL (NULL =>
 * every segment on the 8-wave interp, i.e. baseline split into launches). See segmented-dispatch.md. */
#define RUNSEG(P, K8, K4P)                                                                     \
    do {                                                                                        \
        Prog* g = &B.prog[P];                                                                   \
        PlowProgram pr;                                                                         \
        memset(&pr, 0, sizeof(pr));                                                             \
        pr.insts = g->d_inst; pr.stream = g->d_stream; pr.stream_ofs = g->d_sofs;               \
        pr.stream_len = g->d_slen; pr.waits = g->d_waits; pr.succs = g->d_succs;                \
        pr.counters = g->d_ctr; pr.tensors = (void* const*)d_tens;                              \
        pr.trace = ((P) == trace_prog) ? (PlowTraceRec*)d_trace : NULL;                         \
        pr.gq_stream = g->d_gq_stream; pr.gq_seg_ofs = g->d_gq_seg; pr.gq_cursor = g->d_gq_cursor; \
        plow_hsa_copy_h2d(h, 0, g->d_ctr, zc, (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4);    \
        if (g->d_gq_cursor) plow_hsa_copy_h2d(h, 0, g->d_gq_cursor, zc,                         \
                            (size_t)(g->gq_n_seg ? g->gq_n_seg : 1) * PLOW_CTR_STRIDE * 4);      \
        const double _seg_t0 = now();                                                           \
        for (uint32_t s = 0; s < g->n_seg; s++) {                                               \
            int use4 = (K4P) != NULL && g->seg_class[s] == 4;                                   \
            plow_hsa_kernel* kk = use4 ? (K4P) : &(K8);                                         \
            /* The flash code object's wave count is a BUILD choice: Gemma's D=256/512 object is  \
             * 4-wave (256 threads); the D=128 (Llama/Qwen) object is 8-wave (512 threads) so its \
             * small-D tile runs at 2 waves/SIMD and hides its barrier+softmax latency. The thread \
             * count of the loaded flash object must match how it was built — PLOW_FLASH_THREADS   \
             * carries that (default 256). */                                                     \
            uint32_t th = use4 ? g_flash_threads : (uint32_t)PLOW_WG_THREADS;                    \
            pr.cur_seg = s;                                                                     \
            if (plow_hsa_launch(h, 0, kk, B.h.n_cu * th, 1, 1, th, 1, 1, 0, &pr, sizeof(pr))) { \
                printf("LAUNCH FAILED seg %u\n", s); return 1; }                                \
        }                                                                                       \
        const double _seg_t1 = now();  /* all n_seg packets are in the queue now */              \
        plow_hsa_wait(h, 0); /* ONE drain: each launch has its own kernarg-ring slot (its own    \
                              * cur_seg) and the AQL BARRIER bit + agent fences serialise the     \
                              * segments on the queue, so no per-segment host round-trip. */      \
        const double _seg_t2 = now();                                                            \
        g_seg_enq_us += (_seg_t1 - _seg_t0) * 1e6;                                                \
        g_seg_drain_us += (_seg_t2 - _seg_t1) * 1e6;                                              \
        g_seg_launches += g->n_seg; g_runseg_calls++;                                             \
    } while (0)

    /* ---------------- PREFILL: CHUNKED ----------------
     *
     * A chunk is EXACTLY the existing T=C prefill bucket, launched ceil(n_prompt/C) times with
     * three patched immediates. Nothing new is compiled; the packet fields already exist.
     *
     * WHY. The bucket ladder is powers of two, so a one-shot prefill pays for the whole bucket:
     * measured, a 10000-token prompt lands in the T=16384 program and does 16384 rows of work —
     * 4230 ms, against 1703 ms for 8000 tokens in the T=8192 bucket. 10k, 12k and 16k all take
     * the SAME 4230 ms because they all run the same program. That is up to 64% wasted work.
     * Chunking pads only the LAST chunk: at C=2048 a 10k prompt wastes 2.4%, not 64%.
     *
     * It is also the prerequisite for the sliding-window KV ring (50 of 60 layers only need a
     * 1024-row window, but a one-shot prefill writes all T rows at once and every CU's flash
     * reads them concurrently, so a ring would be clobbered before it was read).
     *
     * THE THREE PATCHES, all found BY IDENTITY, never by position (see the lm_head note below):
     *   HEADNORM_ROPE with j[0] != 0  -> i[3] = c0        the KV cache write row
     *   FLASH_PREFILL                 -> i[4] = c0        q_pos0, the absolute position of row 0
     *                                    i[1] = c0 + clen n_kv: how much cache is valid so far
     *   the matmul into `logits`      -> i[4] = clen - 1  a_row0: the chunk's last row
     *
     * Padding rows in the last chunk write garbage KV at rows >= n_prompt, which nothing ever
     * reads (decode's kvlen is n_prompt, and the causal mask excludes them within the chunk). */
    /* CHOOSE THE CHUNK SIZE. It is not a constant — it is a per-prompt decision, and getting it
     * wrong costs more than chunking gains.
     *
     * Two opposing costs:
     *   PADDING   the bucket ladder is powers of two, so one chunk of C does ceil(n/C)*C rows of
     *             work no matter how few tokens are real. A 10000-token prompt in the T=16384
     *             bucket does 64% wasted work.
     *   LAUNCHES  each chunk is a separate dispatch with less parallelism than a big one.
     *             Measured: 8000 tokens as 1x8192 = 1693 ms, as 4x2048 = 1841 (+9%).
     *
     * cost = padded_rows + LAUNCH_ROWS * n_chunks   reproduces every measured winner:
     *
     *   n=8000   C=8192 (1 chunk )  1693 ms  <- picked      C=2048 (4)  1841
     *   n=10000  C=2048 (5 chunks)  2434     <- picked      C=16384(1)  3667   1.51x
     *   n=12000  C=4096 (3 chunks)  2969     <- picked      C=16384(1)  3903   1.31x
     *   n=16000  C=16384(1 chunk )  4221     <- picked      C=2048 (8)  4523
     *
     * LAUNCH_ROWS is the per-chunk cost in rows-of-work, and it is NOT dispatch overhead -- it is
     * mostly the WEIGHT STREAM. Every chunk re-reads all 57.2 GiB of weights through the GEMM, so
     * a chunk of T rows amortises them over T rows and a smaller chunk over fewer. That is the
     * real force pushing towards BIG chunks, and padding is only the force pushing the other way.
     *
     * Fitted from two measurements at n=3326 (1x4096 = 760 ms; 4 chunks of 3328 rows = 841 ms),
     * solving cost = rows*r + launches*L:  r = 0.168 ms/row, L = 70 ms  ->  L/r = 416 rows.
     *
     * The first cut used 250 and was wrong in a VISIBLE way: at n=3326 it chose
     * [2048+1024+128+128] (3328 rows, 0% pad, 4 launches) over a single 4096 (19% pad, 1 launch)
     * and measured 841 ms against 760. Zero padding is not the objective. */
    /* CHUNKS OF MIXED SIZES, chosen by DP over the bucket ladder.
     *
     * The first version repeated ONE bucket, and that is strictly worse. n=10000 as 5 x 2048 pads
     * 2% but costs FIVE launches; as 8192 + 2048 it pads the same 2% and costs TWO. The ladder is
     * powers of two, so a mixed cover is almost always cheaper than a uniform one.
     *
     *     cost(r) = min over buckets t of  ( t + LAUNCH_ROWS + cost(max(0, r - t)) )
     *
     * Exact, and the state space is tiny (r quantised to the smallest bucket). It reproduces every
     * measured winner and finds the mixed ones the uniform search could not:
     *
     *     n= 8000  ->  8192                  1 launch    ( 2% pad)
     *     n=10000  ->  8192 + 2048           2 launches  ( 2% pad)   was 5 x 2048
     *     n=12000  ->  8192 + 4096           2 launches  ( 2% pad)   was 3 x 4096
     *     n=16000  ->  16384                 1 launch    ( 2% pad)   mixed 8192+8192 is dearer
     *
     * ORDER DOES NOT MATTER for attention cost: the total is sum_i C_i*(prefix_i) + sum C_i^2/2,
     * and sum_i C_i * prefix_i = sum over unordered PAIRS C_i*C_j — symmetric. So any order costs
     * the same, and we emit largest-first purely so the ragged chunk is last.
     *
     * (When the sliding-window KV ring lands it will need R >= W + max(C) - 1, so the DP will
     * gain a cap on the largest bucket. Not yet — there is no ring.) */
    /* PLOW_LAUNCH_ROWS overrides the pad/launch tradeoff for A/B tuning per model. Measured on
     * Qwen3-4B/MI350X the default 416 is already right (if anything low): each extra chunk costs
     * ~30-40 ms of fixed launch overhead even at identical total work (n=8192 as [8192]=514 ms vs
     * [4096+4096]=545 vs [1024]x8=786), and small chunks waste GEMM (128 tok = 3.3k tok/s vs 4096 =
     * 22k). So padding a ragged prompt into ONE bigger bucket beats a zero-pad multi-chunk cover:
     * n=3326 as [4096] (23% pad) = 176 ms, but the zero-pad [2048+1024+128+128] = 275 ms (+56%).
     * The DP picks the fast option; do NOT lower LAUNCH_ROWS to chase zero padding. */
    uint32_t LAUNCH_ROWS = 416;
    { const char* e = getenv("PLOW_LAUNCH_ROWS"); if (e) LAUNCH_ROWS = (uint32_t)atoi(e); }
    uint32_t bkt[16];
    int n_bkt = 0;
    /* CAPPED AT PLOW_MAX_CHUNK. The sliding-window KV ring holds PLOW_KV_RING rows and a chunk's
     * queries span window + C - 1 of them, so a chunk larger than the cap would clobber rows it
     * had not yet read. dev_isa.h owns both constants. */
    for (uint32_t i = 0; i + 1 < B.h.n_prog && n_bkt < 16; i++)
        if (B.prog[i].h.t && B.prog[i].h.t <= PLOW_MAX_CHUNK) bkt[n_bkt++] = B.prog[i].h.t;
    const uint32_t QUANT = bkt[0]; /* the smallest bucket: the DP's step */
    const int NS = (n_prompt + (int)QUANT - 1) / (int)QUANT + 1;
    uint64_t* cost = calloc((size_t)NS, sizeof(uint64_t));
    int* pick = calloc((size_t)NS, sizeof(int));
    for (int r = 1; r < NS; r++) {
        cost[r] = ~0ull;
        for (int b = 0; b < n_bkt; b++) {
            const int step = (int)(bkt[b] / QUANT);
            const int rest = (r - step > 0) ? (r - step) : 0;
            const uint64_t c = (uint64_t)bkt[b] + LAUNCH_ROWS + cost[rest];
            if (c < cost[r]) { cost[r] = c; pick[r] = b; }
        }
    }
    /* Reconstruct, largest-first. */
    uint32_t chunk[64];
    int n_chunk = 0;
    for (int r = NS - 1; r > 0 && n_chunk < 64;) {
        const int b = pick[r];
        chunk[n_chunk++] = bkt[b];
        const int step = (int)(bkt[b] / QUANT);
        r = (r - step > 0) ? (r - step) : 0;
    }
    for (int i = 0; i < n_chunk / 2; i++) { /* largest first */
        const uint32_t t = chunk[i];
        chunk[i] = chunk[n_chunk - 1 - i];
        chunk[n_chunk - 1 - i] = t;
    }
    free(cost);
    free(pick);
    if (getenv("PLOW_CHUNK")) { /* override, for the A/B */
        const uint32_t want = (uint32_t)atoi(getenv("PLOW_CHUNK"));
        n_chunk = 0;
        for (uint32_t c0 = 0; c0 < (uint32_t)n_prompt && n_chunk < 64; c0 += want)
            chunk[n_chunk++] = want;
    }
    uint32_t total = 0;
    printf("prefill: %d tokens -> %d chunk(s) [", n_prompt, n_chunk);
    for (int i = 0; i < n_chunk; i++) { printf("%s%u", i ? "+" : "", chunk[i]); total += chunk[i]; }
    printf("] = %u rows, %.0f%% pad\n", total, 100.0 * (total - n_prompt) / n_prompt);

    /* Every prefill bucket has the same instruction COUNT (same network, different T), so one
     * staging buffer sized off the largest serves them all. */
    uint32_t max_inst = 0;
    for (uint32_t i = 0; i + 1 < B.h.n_prog; i++)
        if (B.prog[i].h.n_inst > max_inst) max_inst = B.prog[i].h.n_inst;
    PlowDevInst* tmp = plow_hsa_alloc_host(h, (size_t)max_inst * sizeof(PlowDevInst));
    const double p0 = now();
    uint32_t c0 = 0;
    for (int c = 0; c < n_chunk; c++) {
        const uint32_t CH = chunk[c]; /* chunks may DIFFER in size — see the DP above */
        const uint32_t clen = ((uint32_t)n_prompt - c0 < CH) ? ((uint32_t)n_prompt - c0) : CH;

        int cp = -1; /* the bucket program compiled for THIS chunk's T */
        for (uint32_t i = 0; i + 1 < B.h.n_prog; i++)
            if (B.prog[i].h.t == CH) { cp = (int)i; break; }
        if (cp < 0) { printf("no bucket for chunk T=%u\n", CH); return 1; }
        Prog* gp = &B.prog[cp];

        /* The lm_head is M=1 over ONE row (i4 = a_row0), and each bucket bakes its own T-1.
         *
         * Found BY IDENTITY: the matmul whose destination is the logits tensor. It used to be
         * found by POSITION -- `insts[n_inst - 2]`, "the one before the softcap" -- and when two
         * packets were later appended to the program that silently became a different
         * instruction. The lm_head then kept the bucket's baked-in T-1 and read row 127 of a
         * 21-token prompt, i.e. padding: garbage logits that looked exactly like a kernel
         * numerics bug for an hour. Do not address instructions by position. */
        int lm = -1;
        for (uint32_t i = 0; i < gp->h.n_inst; i++) {
            const uint16_t o = gp->insts[i].op;
            /* Every matmul opcode that can carry the lm_head, in any precision. The bf16-only
             * list made the search fail outright on a packet whose lm_head is quantized —
             * "could not find the lm_head instruction" — which at least is loud, unlike the two
             * bugs above. Kept complete so a new precision does not resurrect it. */
            const int is_matmul = (o == PLOW_DOP_GEMM || o == PLOW_DOP_GEMM_SMALL ||
                                   o == PLOW_DOP_GEMM_MED || o == PLOW_DOP_GEMV ||
                                   o == PLOW_DOP_GEMM_FP8 || o == PLOW_DOP_GEMM_MED_FP8 ||
                                   o == PLOW_DOP_GEMM_SMALL_FP8 || o == PLOW_DOP_GEMV_FP8 ||
                                   o == PLOW_DOP_GEMV_FP8_BLK || o == PLOW_DOP_GEMM_MXFP4 ||
                                   o == PLOW_DOP_GEMV_MXFP4);
            if (is_matmul && gp->insts[i].t[0] == (uint32_t)t_logits) { lm = (int)i; break; }
        }
        if (lm < 0) { printf("could not find the lm_head instruction\n"); return 1; }

        for (uint32_t i = 0; i < CH; i++)
            h_scalar[i] = (i < clen) ? prompt[c0 + i] : 0;
        plow_hsa_copy_h2d(h, 0, dev[t_ids], h_scalar, (size_t)CH * 4);
        for (uint32_t i = 0; i < CH; i++) h_scalar[i] = (int32_t)(c0 + i); /* ABSOLUTE positions */
        plow_hsa_copy_h2d(h, 0, dev[t_pos], h_scalar, (size_t)CH * 4);

        memcpy(tmp, gp->insts, (size_t)gp->h.n_inst * sizeof(PlowDevInst));
        for (uint32_t i = 0; i < gp->h.n_inst; i++) {
            if (tmp[i].op == PLOW_DOP_HEADNORM_ROPE && tmp[i].fj[1].u != 0)
                tmp[i].i[3] = c0; /* j[0] != 0 is the head-major KV write; the q norm has 0 */
            /* BOTH precisions: an fp8-KV packet emits FLASH_PREFILL_FP8 and never the bf16 twin,
             * so testing only the bf16 opcode left q_pos0 and n_kv at their baked-in bucket
             * values on every fp8 chunk — chunk 2 onward would attend over the wrong KV extent
             * and mask from the wrong absolute position. Silent, and only on fp8. */
            else if (tmp[i].op == PLOW_DOP_FLASH_PREFILL ||
                     tmp[i].op == PLOW_DOP_FLASH_PREFILL_FP8) {
                tmp[i].i[4] = c0;        /* q_pos0 */
                tmp[i].i[1] = c0 + clen; /* n_kv: everything written so far */
            }
        }
        tmp[lm].i[4] = clen - 1;
        plow_hsa_copy_h2d(h, 0, gp->d_inst, tmp, (size_t)gp->h.n_inst * sizeof(PlowDevInst));

        RUNSEG(cp, k_pre, k_flash_p);
        c0 += CH;
    }
    const double pdt = now() - p0;

    /* The token was sampled ON DEVICE (ARGMAX + ARGMAX_FIN) and already sits in `in.ids`,
     * which is exactly where the next step's EMBED reads it. We pull back 4 bytes to print it
     * and to test for EOS -- not the 512 KB logit row, and we write nothing back. */
    int best = 0;
    plow_hsa_copy_d2h(h, 0, h_scalar, dev[t_ids], 4);
    best = (int)h_scalar[0];
    if (getenv("PLOW_CHECK_ARGMAX")) {
        plow_hsa_copy_d2h(h, 0, logit, dev[t_logits], (size_t)VOCAB * 2);
        int hb = 0; float hv = -1e30f;
        for (uint32_t v = 0; v < VOCAB; v++) { float x = b2f(logit[v]); if (x > hv) { hv = x; hb = (int)v; } }
        printf("[argmax] device=%d host=%d  (logit[dev]=%.6f logit[host]=%.6f)\n",
               best, hb, b2f(logit[best < 0 ? 0 : best]), hv);
        /* how many tokens TIE at the max, in bf16? */
        int nties = 0;
        for (uint32_t v = 0; v < VOCAB; v++) if (b2f(logit[v]) == hv) nties++;
        printf("[argmax] VOCAB=%u  logit[100]=%.4f  logit[47]=%.4f\n", VOCAB,
               b2f(logit[100]), b2f(logit[47]));
        printf("[argmax] %d token(s) tie at the bf16 max %.6f:", nties, hv);
        for (uint32_t v = 0; v < VOCAB && nties; v++)
            if (b2f(logit[v]) == hv) printf(" %u", v);
        printf("\n");
    }
    printf("prefill: %d tokens in %.0f ms  (%.0f tok/s)\n", n_prompt, pdt * 1e3, n_prompt / pdt);
    /* MEASUREMENT: segmented-dispatch host accounting (see runtime-launch-verify). */
    printf("  SEG: %ld launch(es) over %ld RUNSEG call(s) = %.1f seg/chunk | "
           "host enqueue %.1f us total (%.2f us/launch) | GPU drain %.1f ms total\n",
           g_seg_launches, g_runseg_calls,
           g_runseg_calls ? (double)g_seg_launches / g_runseg_calls : 0.0,
           g_seg_enq_us, g_seg_launches ? g_seg_enq_us / g_seg_launches : 0.0,
           g_seg_drain_us / 1e3);
    printf("generated:");
    fflush(stdout);

    /* ---------------- DECODE: one launch per token, KV cache reused ---------------- */
    int ctx = n_prompt;
    double dsum = 0, host_us = 0, gpu_us = 0, argmax_us = 0;
    int ngen = 0;
    for (int step = 0; step < n_gen; step++) {
        printf(" %d", best);
        fflush(stdout);
        prompt[ctx] = best;
        /* EOS. Gemma: 1/106/50. Llama-3.1: 128001/128008/128009. Qwen3: 151645. Override with
         * PLOW_EOS="a,b,c" for other tokenizers. */
        int is_eos = 0;
        const char* eosenv = getenv("PLOW_EOS");
        if (eosenv) {
            char buf[128]; snprintf(buf, sizeof(buf), "%s", eosenv);
            for (char* tok = strtok(buf, ","); tok; tok = strtok(NULL, ","))
                if (best == atoi(tok)) is_eos = 1;
        } else {
            is_eos = (best == 1 || best == 106 || best == 50 || best == 128001 ||
                      best == 128008 || best == 128009 || best == 151645);
        }
        if (is_eos) break;

        const int pos = ctx;   /* the new token sits at this position */
        ctx++;
        /* Bound by the KV CACHE, not by a prefill bucket. Same legacy assumption as the prompt
         * guard above: `bp` used to be a bucket big enough for the whole sequence, so it doubled
         * as the context limit. It is now just the largest CHUNK (<= PLOW_MAX_CHUNK), and decode
         * appends one row at a time -- it is bounded by how much history the cache can hold. */
        if (ctx > max_ctx) break;

        Prog* g = &B.prog[dp];
        /* the only per-step immediates: the KV-cache row each k/v norm appends at. Re-upload just
         * the [lo,hi] slice spanning the n_kvrow patch sites, not the whole ~68 KB inst stream.
         * The kvrow insts are SCATTERED in k/v pairs across all layers (Gemma-31B: span [4,664] of
         * 676), so a per-inst scatter upload would cost more in h2d submission overhead than one
         * contiguous slice — the slice is the sound win (fewer bytes, same one submission). */
        uint32_t lo = g->h.n_inst ? g->h.n_inst - 1 : 0, hi = 0;
        for (uint32_t i = 0; i < B.h.n_kvrow; i++) {
            uint32_t idx = B.kvrow[i];
            h_inst[idx].i[3] = (uint32_t)pos;
            if (idx < lo) lo = idx;
            if (idx > hi) hi = idx;
        }
        const double d0 = now();
        if (B.h.n_kvrow)
            plow_hsa_copy_h2d(h, 0, (uint8_t*)g->d_inst + (size_t)lo * sizeof(PlowDevInst),
                              &h_inst[lo], (size_t)(hi - lo + 1) * sizeof(PlowDevInst));
        else
            plow_hsa_copy_h2d(h, 0, g->d_inst, h_inst, (size_t)g->h.n_inst * sizeof(PlowDevInst));
        /* NOT in.ids: the device wrote it itself at the end of the previous step. */
        h_scalar[0] = pos;       plow_hsa_copy_h2d(h, 0, dev[t_pos], h_scalar, 4);
        h_scalar[0] = ctx;       plow_hsa_copy_h2d(h, 0, dev[t_kvlen], h_scalar, 4);
        const double d1 = now();
        RUN(dp, k_dec);
        const double d2 = now();
        host_us += (d1 - d0) * 1e6;
        gpu_us += (d2 - d1) * 1e6;
        dsum += now() - d0;
        ngen++;

        const double a0 = now();
        plow_hsa_copy_d2h(h, 0, h_scalar, dev[t_ids], 4); /* 4 bytes, not 512 KB */
        best = (int)h_scalar[0];
        argmax_us += (now() - a0) * 1e6;
        dsum += now() - a0;
    }
    printf("\n");
    if (do_trace) {
        Prog* g = &B.prog[trace_prog];
        PlowTraceRec* tr = plow_hsa_alloc_host(h, (size_t)g->h.n_stream * sizeof(PlowTraceRec));
        plow_hsa_copy_d2h(h, 0, tr, d_trace, (size_t)g->h.n_stream * sizeof(PlowTraceRec));
        if (getenv("PLOW_TRACE_RAW")) {
            FILE* f = fopen(getenv("PLOW_TRACE_RAW"), "wb");
            fwrite(tr, sizeof(PlowTraceRec), g->h.n_stream, f);
            fclose(f);
            printf("raw trace -> %s (%u records)\n", getenv("PLOW_TRACE_RAW"), g->h.n_stream);
        }
        /* Indexed by PLOW_DOP_*. KEEP IT DENSE AND IN STEP WITH dev_isa.h: an op past the end
         * of this table is silently attributed to "nop" (the lookup clamps), so a new opcode
         * does not show up as unknown -- it shows up as someone else's cost, or as nothing at
         * all. GEMM_GLU=20 did exactly that, and the gemm row lost 15k workgroup-packets to a
         * "nop" that was really the biggest op in the MLP. */
        static const char* OPN[] = {"nop","rmsnorm","rowrms","headnorm_rope","residual","glu",
                                    "embed","softcap","gemm","gemm_norm","gemv","flash_prefill",
                                    "flash_decode","flash_merge","gemm_small","gemm_med",
                                    "norm_residual","argmax","argmax_fin","gemv_glu","gemm_glu",
                                    "add_norm","gemv_qkv","norm_residual_norm",
                                    /* 24-29 TP collectives */
                                    "xreduce","xreducescatter","xallgather","xflashmerge",
                                    "xargmax_fin","xreduce2",
                                    /* 30-39 fp8 (renumbered +6 past XREDUCE2 on the tp merge) */
                                    "gemv_fp8","gemv_glu_fp8","quant_fp8","gemm_fp8","gemm_med_fp8",
                                    "gemm_small_fp8","gemm_glu_fp8","headnorm_rope_fp8",
                                    "flash_decode_fp8","flash_prefill_fp8"};
        const int NOPN = (int)(sizeof(OPN) / sizeof(OPN[0]));
        uint64_t t0 = ~0ull, t1 = 0;
        for (uint32_t i = 0; i < g->h.n_stream; i++) {
            if (tr[i].t_arrive && tr[i].t_arrive < t0) t0 = tr[i].t_arrive;
            if (tr[i].t_end > t1) t1 = tr[i].t_end;
        }
        const double us = (gpu_us / ngen) / (double)(t1 - t0);
        /* aggregate by OP, not by packet: 1134 packets is too many to read one by one */
        double w[64] = {0}, st[64] = {0};
        int cnt[64] = {0};
        for (uint32_t i = 0; i < g->h.n_stream; i++) {
            int o = tr[i].op < NOPN ? tr[i].op : 0;
            w[o] += (double)(tr[i].t_end - tr[i].t_ready) * us;
            st[o] += (double)(tr[i].t_ready - tr[i].t_arrive) * us;
            cnt[o]++;
        }
        printf("\nDECODE TRACE (one step, %u workgroup-packets over %.1f ms):\n",
               g->h.n_stream, (double)(t1 - t0) * us / 1e3);
        printf("  %-14s %7s %11s %11s\n", "op", "wg-pkts", "work(ms)", "stall(ms)");
        for (int o = 0; o < NOPN; o++)
            if (cnt[o])
                printf("  %-14s %7d %11.2f %11.2f\n", OPN[o], cnt[o], w[o] / 1e3, st[o] / 1e3);

        /* CONCURRENCY CHECK. plow schedules WORKGROUPS, not ops: independent projections
         * (q/k/v, gate/up) are handed DISJOINT CU sets by split3/split2 in plowc and are
         * meant to be in flight together. That is a claim about the schedule, and it stays
         * a claim until something measures it -- so measure it.
         *
         * Two instructions ran CONCURRENTLY iff their CU sets are disjoint AND their
         * [t_ready, t_end] intervals overlap. The overlap can be NEGATIVE (a gap), which is
         * what a truly dependent pair looks like; that is the control that makes this
         * falsifiable. Without it, "concurrent" is unfalsifiable and therefore worthless.
         *
         * This is why q/k/v are NOT fused into one GEMV: the fusion would buy only the
         * one-tensor-on-256-CUs bandwidth delta, and the schedule already overlaps them. */
        enum { NI = 512 };
        static uint64_t lo[NI], hi[NI];
        static uint32_t cu_lo[NI], cu_hi[NI], opi[NI];
        uint32_t ni = 0;
        for (uint32_t k = 0; k < NI; k++) { lo[k] = ~0ull; cu_lo[k] = ~0u; }
        for (uint32_t i = 0; i < g->h.n_stream; i++) {
            uint32_t k = tr[i].inst;
            if (k >= NI) continue;
            if (k + 1 > ni) ni = k + 1;
            opi[k] = tr[i].op < NOPN ? tr[i].op : 0;
            if (tr[i].t_ready < lo[k]) lo[k] = tr[i].t_ready;
            if (tr[i].t_end > hi[k]) hi[k] = tr[i].t_end;
            if (tr[i].cu < cu_lo[k]) cu_lo[k] = tr[i].cu;
            if (tr[i].cu > cu_hi[k]) cu_hi[k] = tr[i].cu;
        }
        printf("\n  CONCURRENCY (independent ops must overlap in time on disjoint CUs):\n");
        printf("  %-22s %-11s %-11s %9s  %s\n", "pair", "CUs(a)", "CUs(b)", "overlap", "verdict");
        int bad = 0;
        for (uint32_t a = 0; a + 1 < ni && a < 20; a++) {
            uint32_t b = a + 1;
            if (hi[a] == 0 || hi[b] == 0) continue;
            /* disjoint CU sets? split3/split2 always hand out CONTIGUOUS ranges, so
             * comparing the ranges is exact here, not an approximation. */
            const int disjoint = (cu_hi[a] < cu_lo[b]) || (cu_hi[b] < cu_lo[a]);
            if (!disjoint) continue; /* shares CUs -> cannot be concurrent by construction */
            /* Disjoint CUs is NOT enough to expect overlap: a producer and its consumer can
             * land on disjoint ranges too (gemv->headnorm, gemv->glu both do) and they are
             * SUPPOSED to serialise. The pairs split3/split2 fans out are the same-opcode
             * siblings -- q/k/v and gate/up. Those, and only those, must overlap; anything
             * else here is a true dependency and its cost belongs in the stall table above. */
            if (opi[a] != opi[b]) continue;
            const int64_t ov = (int64_t)((hi[a] < hi[b]) ? hi[a] : hi[b]) -
                               (int64_t)((lo[a] > lo[b]) ? lo[a] : lo[b]);
            char pr[32];
            snprintf(pr, sizeof pr, "%s|%s", OPN[opi[a]], OPN[opi[b]]);
            const int conc = ov > 0;
            printf("  %-22s %3u-%-7u %3u-%-7u %9lld  %s\n", pr, cu_lo[a], cu_hi[a], cu_lo[b],
                   cu_hi[b], (long long)ov, conc ? "CONCURRENT" : "SERIAL (gap!)");
            if (!conc) bad++;
        }
        printf("  -> %d disjoint-CU pair(s) failed to overlap%s\n", bad,
               bad ? " -- the schedule is NOT overlapping them" : "");
    }
    if (ngen) {
        printf("decode: %d tokens, %.1f ms/token (%.1f tok/s)\n", ngen, dsum / ngen * 1e3,
               ngen / dsum);
        printf("  per token: host-setup %.1f ms | GPU %.1f ms | token readback %.1f ms\n",
               host_us / ngen / 1e3, gpu_us / ngen / 1e3, argmax_us / ngen / 1e3);
        /* `wb`, the bytes actually BOUND, not a literal. This was hardcoded to bf16 Gemma-4 31B
         * (57.2 GiB / 61.4e9), so every fp8 run divided the bf16 weight stream by the fp8 time
         * and reported a bandwidth ~1.9x higher than the hardware delivered — a w8a8 31B decode
         * printed "4.55 TB/s" against a 29.9 GiB stream that is really 2.27 TB/s. A benchmark
         * that flatters exactly the configuration you are trying to evaluate is worse than none. */
        printf("  GPU: %.1f GiB of weights in %.1f ms = %.2f TB/s (bf16 GEMV measured 4.3 TB/s)\n",
               wb / 1073741824.0, gpu_us / ngen / 1e3,
               (double)wb / (gpu_us / ngen / 1e6) / 1e12);
    }
    plow_hsa_shutdown(h);
    return 0;
}
