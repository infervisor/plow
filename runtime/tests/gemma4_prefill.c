/* gemma4_prefill.c — run the REAL Gemma-4 31B on the interpreter.
 *
 *   plowc gemma4 <model-dir> <T> prog.pkt      compiles the network to packets
 *   gemma4_prefill prog.pkt <model-dir> ids    binds the weights and runs it
 *
 * Weights are bound BY NAME from the safetensors shards, and a missing name is a hard
 * error. That matters more than it sounds: a silently-absent weight (v_norm has no
 * checkpoint tensor; layer_scalar is easy to skip) does not crash and does not produce
 * garbage -- it produces fluent, confident, WRONG text. Every load is accounted for.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

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
static float b2f(uint16_t v) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)v << 16;
    return c.f;
}

/* ---------------- blob ---------------- */
#define NAME_LEN 80
typedef struct {
    char name[NAME_LEN];
    uint64_t bytes;
    uint64_t init_off; /* UINT64_MAX = no compiler-supplied data */
} TensorDecl;
typedef struct {
    uint32_t n_cu, n_inst, n_stream, n_wait, n_succ, n_counter, n_tensor;
    PlowDevInst* insts;
    PlowStreamEnt* stream;
    uint32_t *stream_ofs, *stream_len, *succs;
    PlowWait* waits;
    TensorDecl* tensors;
    uint8_t* init;
} Blob;

static int load_blob(const char* path, Blob* b) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* p = malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
    if (memcmp(p, "PLOWDEV\x01", 8)) { printf("bad magic\n"); return 1; }
    uint32_t* h = (uint32_t*)(p + 8);
    b->n_cu = h[0]; b->n_inst = h[1]; b->n_stream = h[2];
    b->n_wait = h[3]; b->n_succ = h[4]; b->n_counter = h[5]; b->n_tensor = h[6];
    uint8_t* q = p + 8 + 32;
    b->insts = (PlowDevInst*)q;    q += (size_t)b->n_inst * sizeof(PlowDevInst);
    b->stream = (PlowStreamEnt*)q; q += (size_t)b->n_stream * sizeof(PlowStreamEnt);
    b->stream_ofs = (uint32_t*)q;  q += (size_t)b->n_cu * 4;
    b->stream_len = (uint32_t*)q;  q += (size_t)b->n_cu * 4;
    b->waits = (PlowWait*)q;       q += (size_t)b->n_wait * sizeof(PlowWait);
    b->succs = (uint32_t*)q;       q += (size_t)b->n_succ * 4;
    uint64_t init_bytes = *(uint64_t*)q; q += 8;
    b->tensors = (TensorDecl*)q;   q += (size_t)b->n_tensor * sizeof(TensorDecl);
    b->init = q;
    (void)init_bytes;
    return 0;
}

/* ---------------- safetensors ----------------
 * Minimal reader: mmap each shard, scan its JSON header for `"name":{...
 * "data_offsets":[a,b]}`. No JSON library, because the only thing we need out of the
 * header is a byte range, and a wrong byte range fails loudly (size mismatch). */
#define MAX_SHARD 8
typedef struct {
    int n;
    uint8_t* base[MAX_SHARD];
    size_t len[MAX_SHARD];
    char* hdr[MAX_SHARD];
    size_t hdr_len[MAX_SHARD];
    uint64_t data0[MAX_SHARD];
} Safet;

static int st_open(Safet* s, const char* dir) {
    s->n = 0;
    for (int i = 1; i <= MAX_SHARD; i++) {
        char p[512];
        snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, i, 2);
        int fd = open(p, O_RDONLY);
        if (fd < 0) break;
        struct stat st;
        fstat(fd, &st);
        uint8_t* m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) { printf("mmap failed %s\n", p); return 1; }
        uint64_t hn = *(uint64_t*)m;
        s->base[s->n] = m;
        s->len[s->n] = (size_t)st.st_size;
        s->hdr[s->n] = (char*)(m + 8);
        s->hdr_len[s->n] = (size_t)hn;
        s->data0[s->n] = 8 + hn;
        s->n++;
    }
    return s->n ? 0 : 1;
}

/* find "<name>":{...,"data_offsets":[a,b]} -> returns pointer + size, or NULL */
static const uint8_t* st_find(Safet* s, const char* name, uint64_t* out_bytes) {
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
        d++; /* comma */
        unsigned long long b = strtoull(d, (char**)&d, 10);
        *out_bytes = (uint64_t)(b - a);
        return s->base[i] + s->data0[i] + a;
    }
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: %s prog.pkt <model-dir> prompt.ids [n_gen]\n", argv[0]);
        return 1;
    }
    const int n_gen = argc > 4 ? atoi(argv[4]) : 1;
    Blob B;
    if (load_blob(argv[1], &B)) return 1;
    Safet S;
    if (st_open(&S, argv[2])) { printf("no safetensors in %s\n", argv[2]); return 1; }

    /* prompt */
    FILE* pf = fopen(argv[3], "rb");
    if (!pf) { printf("no %s\n", argv[3]); return 1; }
    fseek(pf, 0, SEEK_END); long pn = ftell(pf); fseek(pf, 0, SEEK_SET);
    int n_prompt = (int)(pn / 4);
    int32_t* prompt = malloc((size_t)pn);
    if (fread(prompt, 1, (size_t)pn, pf) != (size_t)pn) return 1;
    fclose(pf);

    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, gfx, &cus, &lds);
    printf("dev0: %s  CUs=%u\n", gfx, cus);
    printf("program: %u packets, %u workgroup-packets, %u tensors\n", B.n_inst, B.n_stream,
           B.n_tensor);
    printf("prompt: %d tokens\n\n", n_prompt);

    FILE* f = fopen("interp.elf", "rb");
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc((size_t)co_n);
    if (fread(co, 1, (size_t)co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, (size_t)co_n)) { printf("co load failed\n"); return 1; }
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, "plow_interp_gfx950", &k)) { printf("no kernel\n"); return 1; }

    /* ---- bind tensors ---- */
    const size_t STAGE = 64u << 20;
    void* stage = plow_hsa_alloc_host(h, STAGE);
    void** dev = calloc(B.n_tensor, sizeof(void*));
    int t_ids = -1, t_pos = -1, t_logits = -1;
    uint64_t wbytes = 0;
    int n_weight = 0, n_init = 0, n_act = 0;
    const double lt0 = now();
    for (uint32_t i = 0; i < B.n_tensor; i++) {
        TensorDecl* td = &B.tensors[i];
        dev[i] = plow_hsa_alloc(h, 0, td->bytes);
        if (!dev[i]) { printf("VRAM alloc failed for %s (%llu B)\n", td->name,
                               (unsigned long long)td->bytes); return 1; }
        if (!strcmp(td->name, "in.ids")) t_ids = (int)i;
        if (!strcmp(td->name, "in.pos")) t_pos = (int)i;
        if (!strcmp(td->name, "act.logits")) t_logits = (int)i;

        if (!strncmp(td->name, "model.", 6)) {
            /* a checkpoint weight: bind by name, hard-fail if absent */
            uint64_t got = 0;
            const uint8_t* src = st_find(&S, td->name, &got);
            if (!src) { printf("MISSING WEIGHT: %s\n", td->name); return 1; }
            if (got != td->bytes) {
                printf("SIZE MISMATCH %s: checkpoint %llu B, program expects %llu B\n", td->name,
                       (unsigned long long)got, (unsigned long long)td->bytes);
                return 1;
            }
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, src + o, n);
                plow_hsa_copy_h2d(h, 0, (uint8_t*)dev[i] + o, stage, n);
            }
            wbytes += td->bytes;
            n_weight++;
        } else if (td->init_off != UINT64_MAX) {
            /* compiler-supplied data (the RoPE tables) */
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, B.init + td->init_off + o, n);
                plow_hsa_copy_h2d(h, 0, (uint8_t*)dev[i] + o, stage, n);
            }
            n_init++;
        } else {
            n_act++;
        }
    }
    printf("bound %d weights (%.1f GiB), %d compiler tables, %d activations in %.1f s\n",
           n_weight, wbytes / 1073741824.0, n_init, n_act, now() - lt0);

    /* ---- program tables ---- */
    void* d_inst = plow_hsa_alloc(h, 0, (size_t)B.n_inst * sizeof(PlowDevInst));
    void* d_stream = plow_hsa_alloc(h, 0, (size_t)B.n_stream * sizeof(PlowStreamEnt));
    void* d_sofs = plow_hsa_alloc(h, 0, (size_t)B.n_cu * 4);
    void* d_slen = plow_hsa_alloc(h, 0, (size_t)B.n_cu * 4);
    void* d_waits = plow_hsa_alloc(h, 0, (size_t)(B.n_wait ? B.n_wait : 1) * sizeof(PlowWait));
    void* d_succs = plow_hsa_alloc(h, 0, (size_t)(B.n_succ ? B.n_succ : 1) * 4);
    void* d_ctr = plow_hsa_alloc(h, 0, (size_t)B.n_counter * 4);
    void* d_tens = plow_hsa_alloc(h, 0, (size_t)B.n_tensor * sizeof(void*));
    plow_hsa_upload(h, 0, d_inst, B.insts, (size_t)B.n_inst * sizeof(PlowDevInst));
    plow_hsa_upload(h, 0, d_stream, B.stream, (size_t)B.n_stream * sizeof(PlowStreamEnt));
    plow_hsa_upload(h, 0, d_sofs, B.stream_ofs, (size_t)B.n_cu * 4);
    plow_hsa_upload(h, 0, d_slen, B.stream_len, (size_t)B.n_cu * 4);
    if (B.n_wait) plow_hsa_upload(h, 0, d_waits, B.waits, (size_t)B.n_wait * sizeof(PlowWait));
    if (B.n_succ) plow_hsa_upload(h, 0, d_succs, B.succs, (size_t)B.n_succ * 4);
    plow_hsa_upload(h, 0, d_tens, dev, (size_t)B.n_tensor * sizeof(void*));

    const uint32_t T = B.insts[0].i[0];       /* embed's ntok == the program's T */
    const uint32_t VOCAB = B.insts[B.n_inst - 1].i[0] / T;
    printf("T=%u vocab=%u  generating %d token(s) greedily\n\n", T, VOCAB, n_gen);

    int32_t* ids = calloc(T, 4);
    int32_t* posv = calloc(T, 4);
    for (uint32_t i = 0; i < T; i++) posv[i] = (int32_t)i;
    plow_hsa_upload(h, 0, dev[t_pos], posv, (size_t)T * 4);

    uint32_t* zc = calloc(B.n_counter, 4);
    uint16_t* logit_row = malloc((size_t)VOCAB * 2);
    PlowProgram prog;
    memset(&prog, 0, sizeof(prog));
    prog.insts = d_inst; prog.stream = d_stream; prog.stream_ofs = d_sofs;
    prog.stream_len = d_slen; prog.waits = d_waits; prog.succs = d_succs;
    prog.counters = d_ctr; prog.tensors = (void* const*)d_tens; prog.trace = NULL;

    int n_ctx = n_prompt;
    printf("generated ids:");
    fflush(stdout);
    for (int g = 0; g < n_gen; g++) {
        if ((uint32_t)n_ctx > T) { printf("\n context exceeds compiled T=%u\n", T); break; }
        /* Positions past the prompt are padding; the causal mask means only rows < n_ctx
         * matter, and we read the logits of row n_ctx-1. */
        memset(ids, 0, (size_t)T * 4);
        memcpy(ids, prompt, (size_t)n_ctx * 4);
        plow_hsa_upload(h, 0, dev[t_ids], ids, (size_t)T * 4);
        plow_hsa_upload(h, 0, d_ctr, zc, (size_t)B.n_counter * 4);

        const double t0 = now();
        if (plow_hsa_launch(h, 0, &k, B.n_cu * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0,
                            &prog, sizeof(prog))) {
            printf("\nLAUNCH FAILED\n");
            return 1;
        }
        plow_hsa_wait(h, 0);
        const double dt = now() - t0;

        plow_hsa_download(h, 0, logit_row,
                          (uint8_t*)dev[t_logits] + (size_t)(n_ctx - 1) * VOCAB * 2,
                          (size_t)VOCAB * 2);
        int best = 0;
        float bv = -1e30f;
        for (uint32_t v = 0; v < VOCAB; v++) {
            float x = b2f(logit_row[v]);
            if (x > bv) { bv = x; best = (int)v; }
        }
        printf(" %d", best);
        fflush(stdout);
        if (g == 0)
            printf("   [prefill %d tok in %.0f ms | top logit %.2f]", n_ctx, dt * 1e3, bv);
        prompt = realloc(prompt, (size_t)(n_ctx + 1) * 4);
        prompt[n_ctx++] = best;
        if (best == 1 || best == 106 || best == 50) break; /* eos */
    }
    printf("\n");
    plow_hsa_shutdown(h);
    return 0;
}
