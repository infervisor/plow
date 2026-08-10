/* qwen3_sm120_chat.cu — a real Qwen3-4B decode token on the RTX 5090.
 *
 *   plowc gemma4 /root/models/Qwen3-4B 512 qwen3.pkt 170
 *   qwen3_sm120_chat qwen3.pkt /root/models/Qwen3-4B prompt.ids N
 *
 * ============================ WHY THIS IS NOT gemma4_chat.c =========================
 *
 * The assignment was to fork gemma4_chat.c behind a vendor-neutral header. I did not,
 * and the reason is a hard constraint rather than a preference:
 *
 *   THE sm_120 INTERPRETER CANNOT RUN PREFILL. FlashPrefill(11), GemmSmall(14),
 *   GemmMed(15) and GemmGlu(20) all land on interp_sm120.cu's default arm, which
 *   __trap()s deliberately. Those four are exactly the ops the T=128 and T=512
 *   prefill buckets are built from. A fork of gemma4_chat.c would therefore carry
 *   its chunked-prefill DP, its bucket ladder, its segmented RUNSEG dispatch and its
 *   trace machinery -- ~600 of 949 lines -- all of it dead, and the first thing it
 *   would do on a real prompt is trap.
 *
 * So the prompt is consumed BY THE DECODE PROGRAM, one token at a time: n_prompt
 * decode steps to build the KV cache, then the generation steps. This is not an
 * approximation. With causal attention, feeding tokens 0..t-1 one at a time and
 * then querying at t produces the same KV cache and the same logits as a batched
 * prefill of 0..t-1 -- prefill is a throughput optimization over exactly this loop,
 * not a different computation. It is O(n) launches instead of O(n/C), which is slow
 * and irrelevant here: the goal is a CORRECT token, and prefill's own kernels are
 * unavailable to produce one.
 *
 * What that costs, stated plainly: this harness exercises the 11 validated decode
 * opcodes and NOTHING ELSE. It is not evidence about prefill.
 *
 * The safetensors loader is shared with gemma4_chat.c via common/safetensors.h
 * (both documented loader defects fixed there); the blob is parsed through the
 * shared dev_blob.h structs, never hand-rolled.
 *
 * TIED EMBEDDINGS: Qwen3 sets tie_word_embeddings and ships no lm_head.weight. This
 * needs no handling HERE -- verified by reading the emitted packet, whose 493 tensor
 * declarations contain no "lm_head" at all: plowc already points the lm_head GEMV's
 * weight operand at model.embed_tokens.weight. A loader binding lm_head by name would
 * hard-fail, so this is asserted at bind time rather than assumed.
 */
#include "../common/dev_blob.h"
#include "../common/dev_isa.h"
#include "../common/safetensors.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

extern "C" int plow_sm120_grid(int dev);
extern "C" size_t plow_sm120_smem(void);
extern "C" int plow_sm120_launch(PlowProgram* prog, int grid, cudaStream_t stream);
extern "C" int plow_sm120_sched(void);    /* 0 = static per-block stream, 1 = global queue */
extern "C" int plow_sm120_skeleton(void); /* 1 = gate/signal only, GARBAGE output by design */

/* DECODE GF8 TWIN (_gf8 symbols, beat12b-ctx-switch): a second decode object at full-attn GQA
 * fusion 8. Same decode op family as the GF2 object above — the ONLY difference is the hd512
 * full-attn flash instantiation (GF=8) and its larger register/arena footprint. The switching
 * harness picks this launcher per decode step when kvlen >= PLOW_GF_SWITCH (see run_step). */
extern "C" int plow_sm120_grid_gf8(int dev);
extern "C" size_t plow_sm120_smem_gf8(void);
extern "C" int plow_sm120_launch_gf8(PlowProgram* prog, int grid, cudaStream_t stream);

/* PREFILL object (_pf symbols): the tiled-GEMM + FLASH_PREFILL megakernel. Built from the same
 * interp_sm120.cu with -DPLOW_NV_PREFILL=1, so its exported symbols are suffixed. The prompt is
 * consumed by these programs in chunks; generation then runs the decode object above. */
extern "C" int plow_sm120_grid_pf(int dev);
extern "C" size_t plow_sm120_smem_pf(void);
extern "C" int plow_sm120_launch_pf(PlowProgram* prog, int grid, cudaStream_t stream);
extern "C" int plow_sm120_sched_pf(void);

/* T9c SEGMENTED prefill object (_pfseg symbols): same arms as _pf but PLOW_NV_SEGMENTS=1, so the
 * interp bounds each cooperative launch to prog.cur_seg. Selected at runtime when PLOW_NV_SEGMENTS=1;
 * the host then relaunches once per wave-class segment (RUNSEG, mirroring amd/interp.hip). */
extern "C" int plow_sm120_grid_pfseg(int dev);
extern "C" size_t plow_sm120_smem_pfseg(void);
extern "C" int plow_sm120_launch_pfseg(PlowProgram* prog, int grid, cudaStream_t stream);
extern "C" int plow_sm120_sched_pfseg(void);

/* T10 LEAN GEMM segment object (_pfgemm symbols): PLOW_NV_SEG_GEMM=1 — 128-reg, occ-2. Runs the
 * GEMM/tier-A (wave-class-8) segments at grid = 2*n_cu (occupancy 2 blocks/SM), while flash
 * (wave-class-4) segments keep the occ-1 _pfseg object at grid = n_cu. Selected per segment when
 * PLOW_NV_SEG_GEMM=1 is set at runtime; the emitter must have re-sliced the GEMM segments to 2*n_cu
 * (PLOW_SEG_CLASS_SLICE=1 pkt) so both resident blocks per SM get work. */
extern "C" int plow_sm120_grid_pfgemm(int dev);
extern "C" size_t plow_sm120_smem_pfgemm(void);
extern "C" int plow_sm120_launch_pfgemm(PlowProgram* prog, int grid, cudaStream_t stream);
extern "C" int plow_sm120_sched_pfgemm(void);

#define CK(x)                                                                      \
    do {                                                                             \
        cudaError_t _e = (x);                                                        \
        if (_e != cudaSuccess) {                                                     \
            printf("CUDA FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #x,            \
                   cudaGetErrorString(_e));                                          \
            return 1;                                                                \
        }                                                                            \
    } while (0)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* T9c RUNSEG launch-tax accounting (mirrors gemma4_chat.c's g_seg_* on AMD). enq = host wall spent
 * enqueuing all segment launches; drain = the single device sync; launches/calls give per-launch µs. */
static double g_seg_enq_us = 0, g_seg_drain_us = 0;
static uint64_t g_seg_launches = 0, g_runseg_calls = 0;
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
    void *d_gqstream, *d_gqsegofs, *d_gqcursor; /* global-queue (M5 E1) tables */
    uint32_t n_seg; /* T9c: wave-class segment count (>=1); gq_segofs has n_seg+1 entries, cursor n_seg lines */
    uint8_t* seg_is_flash; /* T10: [n_seg] per-segment class — 1 = flash (wave-4, _pfseg/occ-1), 0 = GEMM (wave-8, _pfgemm/occ-2) */
} Prog;

/* FlashPrefill(11)/FlashPrefillFp8(39) are the wave-class-4 (flash) ops; everything else is
 * wave-class-8 (GEMM/tier-A). A segment is a maximal same-class run, so its class is the class of
 * any op in it. Mirrors the emitter's `wave_class` in packet/src/devbuild.rs. */
static inline int op_is_flash(uint16_t op) { return op == 11 || op == 39; }

typedef struct {
    PlowBlobHeader h;
    PlowTensorDecl* tensors;
    uint8_t* init;
    uint32_t* kvrow;
    Prog* prog;
} Blob;

/* Parsed through the SHARED structs in dev_blob.h. dev_abi.rs asserts every offset
 * against the Rust writer, so the two cannot drift. */
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
      if (e) { printf("%s\n", e); return 1; } }
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

/* Upload one program's tables to the device, verify it is a single coarse segment, and (when the
 * object runs the global-queue scheduler) build its op-major GQ permutation. Factored out of the
 * decode-only harness so the prefill bucket programs and the decode program share one code path. */
static int prep_prog(Blob* B, Prog* g, int sched) {
    CK(cudaMalloc(&g->d_inst, (size_t)g->h.n_inst * sizeof(PlowDevInst)));
    CK(cudaMalloc(&g->d_stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt)));
    CK(cudaMalloc(&g->d_sofs, (size_t)B->h.n_cu * 4));
    CK(cudaMalloc(&g->d_slen, (size_t)B->h.n_cu * 4));
    CK(cudaMalloc(&g->d_waits, (size_t)(g->h.n_wait ? g->h.n_wait : 1) * sizeof(PlowWait)));
    CK(cudaMalloc(&g->d_succs, (size_t)(g->h.n_succ ? g->h.n_succ : 1) * 4));
    CK(cudaMalloc(&g->d_ctr, (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4));
    CK(cudaMemcpy(g->d_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_stream, g->stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_sofs, g->stream_ofs, (size_t)B->h.n_cu * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_slen, g->stream_len, (size_t)B->h.n_cu * 4, cudaMemcpyHostToDevice));
    if (g->h.n_wait)
        CK(cudaMemcpy(g->d_waits, g->waits, (size_t)g->h.n_wait * sizeof(PlowWait),
                      cudaMemcpyHostToDevice));
    if (g->h.n_succ)
        CK(cudaMemcpy(g->d_succs, g->succs, (size_t)g->h.n_succ * 4, cudaMemcpyHostToDevice));

    /* Wave-class segments are HANDLED (T9c) and SE_FINE per-slice gates are now HANDLED
     * (SE-FINE-decode: the interp reads the entry's own wait/succ lists). SE_XCTR (cross-GPU) is
     * still unimplemented on this single-GPU build and traps loudly. Also derive n_seg = max seg + 1
     * (a single-segment / PLOW_UNISEG program yields n_seg==1). */
    uint32_t n_seg = 1;
    for (uint32_t j = 0; j < g->h.n_stream; j++) {
        if (g->stream[j].flags & PLOW_SE_XCTR) {
            printf("FATAL: prog T=%u stream entry %u is xctr-gated; this interpreter is single-GPU "
                   "(cross-GPU counters unimplemented).\n", g->h.t, j);
            return 1;
        }
        if ((uint32_t)g->stream[j].seg + 1u > n_seg) n_seg = (uint32_t)g->stream[j].seg + 1u;
    }
    g->n_seg = n_seg;

    if (sched == 1) {
        const uint32_t NI = g->h.n_inst, NS = g->h.n_stream, n_ctr = g->h.n_counter;
        /* Gates live on the stream entries (the 64-byte PlowDevInst carries none); entries
         * of the same coarse inst repeat the same lists, which is harmless to max/compare. */
        int32_t* prod_max = (int32_t*)malloc(sizeof(int32_t) * n_ctr);
        for (uint32_t c = 0; c < n_ctr; c++) prod_max[c] = -1;
        for (uint32_t j = 0; j < NS; j++)
            for (uint32_t s = 0; s < g->stream[j].succ_len; s++) {
                const uint32_t c = g->succs[g->stream[j].succ_ofs + s];
                if ((int32_t)g->stream[j].inst > prod_max[c]) prod_max[c] = (int32_t)g->stream[j].inst;
            }
        for (uint32_t j = 0; j < NS; j++)
            for (uint32_t w = 0; w < g->stream[j].wait_len; w++) {
                const uint32_t c = g->waits[g->stream[j].wait_ofs + w].id;
                const int32_t i = (int32_t)g->stream[j].inst;
                if (prod_max[c] >= i) {
                    printf("FATAL: prog T=%u inst %d waits on counter %u whose latest producer is "
                           "inst %d — not topological, GQ would deadlock.\n", g->h.t, i, c,
                           prod_max[c]);
                    return 1;
                }
            }
        free(prod_max);
        uint32_t* cnt = (uint32_t*)calloc(NI + 1, 4);
        for (uint32_t j = 0; j < NS; j++) cnt[g->stream[j].inst + 1]++;
        for (uint32_t i = 0; i < NI; i++) cnt[i + 1] += cnt[i];
        PlowStreamEnt* gqs = (PlowStreamEnt*)malloc((size_t)NS * sizeof(PlowStreamEnt));
        uint32_t* put = (uint32_t*)malloc((NI + 1) * 4);
        memcpy(put, cnt, (NI + 1) * 4);
        for (uint32_t cu = 0; cu < B->h.n_cu; cu++)
            for (uint32_t k = 0; k < g->stream_len[cu]; k++) {
                const PlowStreamEnt e = g->stream[g->stream_ofs[cu] + k];
                gqs[put[e.inst]++] = e;
            }
        /* Per-segment window bounds. gqs is counting-sorted by inst id, and seg is a function of
         * inst id monotonic in emit order, so gqs is non-decreasing in seg — each segment occupies a
         * contiguous [segofs[s], segofs[s+1]) range, identical to the emitter's gq_seg_ofs. n_seg==1
         * reduces to {0, NS}, byte-identical to the previous single-segment path. */
        uint32_t* segofs = (uint32_t*)calloc(n_seg + 1, 4);
        { uint32_t s = 0;
          for (uint32_t j = 0; j < NS; j++)
              while ((uint32_t)gqs[j].seg > s) { s++; segofs[s] = j; }
          segofs[n_seg] = NS; }
        /* T10 per-segment class: the op of the segment's first entry decides flash vs GEMM. */
        g->seg_is_flash = (uint8_t*)calloc(n_seg, 1);
        for (uint32_t s = 0; s < n_seg; s++) {
            const uint32_t j = segofs[s]; /* first entry of segment s (empty seg impossible: n_seg from max seg) */
            g->seg_is_flash[s] = (j < NS) ? (uint8_t)op_is_flash(g->insts[gqs[j].inst].op) : 0;
        }
        CK(cudaMalloc(&g->d_gqstream, (size_t)NS * sizeof(PlowStreamEnt)));
        CK(cudaMalloc(&g->d_gqsegofs, (size_t)(n_seg + 1) * 4));
        CK(cudaMalloc(&g->d_gqcursor, (size_t)n_seg * PLOW_CTR_STRIDE * 4));
        CK(cudaMemcpy(g->d_gqstream, gqs, (size_t)NS * sizeof(PlowStreamEnt),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g->d_gqsegofs, segofs, (size_t)(n_seg + 1) * 4, cudaMemcpyHostToDevice));
        free(gqs); free(cnt); free(put); free(segofs);
    }
    return 0;
}

static void mk_pr(PlowProgram* pr, Prog* g, void* d_tens) {
    memset(pr, 0, sizeof(*pr));
    pr->insts = (const PlowDevInst*)g->d_inst;
    pr->stream = (const PlowStreamEnt*)g->d_stream;
    pr->stream_ofs = (const uint32_t*)g->d_sofs;
    pr->stream_len = (const uint32_t*)g->d_slen;
    pr->waits = (const PlowWait*)g->d_waits;
    pr->succs = (const uint32_t*)g->d_succs;
    pr->counters = (uint32_t*)g->d_ctr;
    pr->tensors = (void* const*)d_tens;
    pr->trace = NULL;
    pr->gq_stream = (const PlowStreamEnt*)g->d_gqstream;
    pr->gq_seg_ofs = (const uint32_t*)g->d_gqsegofs;
    pr->gq_cursor = (uint32_t*)g->d_gqcursor;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: %s model.pkt <model-dir> prompt.ids [n_gen]\n", argv[0]);
        return 1;
    }
    const int n_gen = argc > 4 ? atoi(argv[4]) : 24;

    Blob B;
    if (load_blob(argv[1], &B)) return 1;

    Safet S;
    if (st_open(&S, argv[2])) { printf("no safetensors in %s\n", argv[2]); return 1; }
    printf("checkpoint: %s (%d shard(s))\n", argv[2], S.n);

    /* fp8 weight twins (PLOW_FP8_DIR) — same mechanism as qwen3_sm120_chat.cu. A fp8 decode
     * packet declares uint8 W + f32 scale under an "fp8/" prefix (the DECODE GEMV_FP8 path);
     * PREFILL still runs bf16 GEMM on the "model." bf16 weights, so BOTH the base checkpoint
     * and this dir are bound. st_find ignores the dtype string and returns the raw byte range. */
    Safet Sf;
    int have_fp8 = 0;
    const char* fp8dir = getenv("PLOW_FP8_DIR");
    if (fp8dir) {
        if (st_open(&Sf, fp8dir)) { printf("PLOW_FP8_DIR set but no safetensors in %s\n", fp8dir); return 1; }
        have_fp8 = 1;
        printf("fp8 twins: %s (%d shard(s))\n", fp8dir, Sf.n);
    }

    FILE* pf = fopen(argv[3], "rb");
    if (!pf) { printf("no %s\n", argv[3]); return 1; }
    fseek(pf, 0, SEEK_END);
    long pn = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    const int n_prompt = (int)(pn / 4);
    int32_t* prompt = (int32_t*)malloc((size_t)pn + 4 * (size_t)(n_gen + 1));
    if (fread(prompt, 1, (size_t)pn, pf) != (size_t)pn) return 1;
    fclose(pf);

    /* SECOND PROMPT for the batched gate (PLOW_PROMPT2=<file.ids>). Feeding every slot the SAME
     * prompt cannot catch a per-row indexing bug: with identical rows, reading row 0's activation
     * for row 3 produces the right answer anyway. With PLOW_PROMPT2 the ODD slots get a different
     * prompt, so each slot must reproduce ITS OWN B=1 token stream — that is the real serving
     * shape (different users, different routing) and it fails loudly on any row mixup. */
    int32_t* prompt2 = NULL;
    int n_prompt2 = 0;
    if (const char* p2 = getenv("PLOW_PROMPT2")) {
        FILE* f2 = fopen(p2, "rb");
        if (!f2) { printf("no %s\n", p2); return 1; }
        fseek(f2, 0, SEEK_END);
        long pn2 = ftell(f2);
        fseek(f2, 0, SEEK_SET);
        n_prompt2 = (int)(pn2 / 4);
        prompt2 = (int32_t*)malloc((size_t)pn2 + 4 * (size_t)(n_gen + 1));
        if (fread(prompt2, 1, (size_t)pn2, f2) != (size_t)pn2) return 1;
        fclose(f2);
        if (n_prompt2 != n_prompt) {
            printf("PLOW_PROMPT2 length %d != prompt length %d (must match)\n", n_prompt2, n_prompt);
            return 1;
        }
        printf("PROMPT2: %s (%d tokens) -> odd slots\n", p2, n_prompt2);
    }

    int dev = 0;
    cudaDeviceProp dp_prop;
    CK(cudaGetDeviceProperties(&dp_prop, dev));
    const int grid = plow_sm120_grid(dev);
    printf("device: %s  cc %d.%d  SMs=%d\n", dp_prop.name, dp_prop.major, dp_prop.minor,
           dp_prop.multiProcessorCount);
    printf("interpreter grid=%d  dynamic smem=%zu B\n", grid, plow_sm120_smem());

    /* THE GRID MUST EQUAL n_cu. stream_ofs/stream_len are [n_cu] arrays indexed by
     * blockIdx.x, and the compiler partitioned every packet's work across exactly
     * n_cu slices. grid > n_cu reads off the end of both tables; grid < n_cu leaves
     * some block's stream entries UNEXECUTED -- the counters they were to signal
     * never fire and the whole cooperative grid deadlocks, or worse, a packet
     * silently computes on a fraction of its rows. This is a correctness gate, not
     * a tuning check, so it is fatal rather than a warning. */
    if (grid != (int)B.h.n_cu) {
        printf("FATAL: interpreter grid %d != packet n_cu %u.\n", grid, B.h.n_cu);
        printf("       recompile the packet with n_cu=%d.\n", grid);
        return 1;
    }

    printf("%u programs, %u tensors, %u kv-row patch sites\n", B.h.n_prog, B.h.n_tensor,
           B.h.n_kvrow);

    /* ---- bind weights ---- */
    const size_t STAGE = 64u << 20;
    void* stage = NULL;
    CK(cudaMallocHost(&stage, STAGE));
    void** devp = (void**)calloc(B.h.n_tensor, sizeof(void*));
    int t_ids = -1, t_pos = -1, t_kvlen = -1, t_logits = -1;
    uint64_t wb = 0, kvb = 0;
    int nw = 0;
    const double lt0 = now();
    for (uint32_t i = 0; i < B.h.n_tensor; i++) {
        PlowTensorDecl* td = &B.tensors[i];
        CK(cudaMalloc(&devp[i], td->bytes));
        if (!strcmp(td->name, "in.ids")) t_ids = (int)i;
        if (!strcmp(td->name, "in.pos")) t_pos = (int)i;
        if (!strcmp(td->name, "in.kvlen")) t_kvlen = (int)i;
        if (!strcmp(td->name, "act.logits")) t_logits = (int)i;
        if (!strncmp(td->name, "kv.", 3)) kvb += td->bytes;

        /* TIED EMBEDDINGS, asserted rather than assumed. Qwen3 ships no
         * lm_head.weight; plowc resolves the tie at compile time by pointing the
         * lm_head GEMV at model.embed_tokens.weight, so no such tensor should be
         * declared. If one ever is, st_find below would fail with a bare
         * "MISSING WEIGHT" and the cause would be non-obvious -- say it here. */
        if (!strncmp(td->name, "lm_head", 7)) {
            printf("FATAL: packet declares '%s', but Qwen3 has tie_word_embeddings=true\n"
                   "       and ships NO lm_head.weight. plowc must alias the lm_head GEMV\n"
                   "       onto model.embed_tokens.weight instead of declaring a tensor.\n",
                   td->name);
            return 1;
        }
        if (!strncmp(td->name, "fp8/", 4)) {
            /* fp8 weight/scale twin: bind from PLOW_FP8_DIR (uint8 W, f32 scale). */
            if (!have_fp8) {
                printf("FATAL: packet declares fp8 twin '%s' but PLOW_FP8_DIR is unset.\n", td->name);
                return 1;
            }
            uint64_t got = 0;
            const uint8_t* src = st_find(&Sf, td->name, &got);
            if (!src) { printf("MISSING FP8 WEIGHT: %s\n", td->name); return 1; }
            if (got != td->bytes) {
                printf("FP8 SIZE MISMATCH %s (want %llu got %llu)\n", td->name,
                       (unsigned long long)td->bytes, (unsigned long long)got);
                return 1;
            }
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, src + o, n);
                CK(cudaMemcpy((uint8_t*)devp[i] + o, stage, n, cudaMemcpyHostToDevice));
            }
            wb += td->bytes;
            nw++;
        } else if (!strncmp(td->name, "model.", 6)) {
            uint64_t got = 0;
            const uint8_t* src = st_find(&S, td->name, &got);
            if (!src) { printf("MISSING WEIGHT: %s\n", td->name); return 1; }
            if (got != td->bytes) {
                printf("SIZE MISMATCH %s (want %llu got %llu)\n", td->name,
                       (unsigned long long)td->bytes, (unsigned long long)got);
                return 1;
            }
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, src + o, n);
                CK(cudaMemcpy((uint8_t*)devp[i] + o, stage, n, cudaMemcpyHostToDevice));
            }
            wb += td->bytes;
            nw++;
        } else if (td->init_off != PLOW_INIT_NONE) {
            for (uint64_t o = 0; o < td->bytes; o += STAGE) {
                size_t n = (size_t)((td->bytes - o < STAGE) ? (td->bytes - o) : STAGE);
                memcpy(stage, B.init + td->init_off + o, n);
                CK(cudaMemcpy((uint8_t*)devp[i] + o, stage, n, cudaMemcpyHostToDevice));
            }
        } else {
            CK(cudaMemset(devp[i], 0, td->bytes));
        }
    }
    if (t_ids < 0 || t_pos < 0 || t_kvlen < 0 || t_logits < 0) {
        printf("blob is missing in.ids/in.pos/in.kvlen/act.logits\n");
        return 1;
    }

    /* ===== Gemma-4 26B-A4B MoE: build per-layer expert pointer tables =====
     * The emitter declares `moe.ewt.{l}` (Persistent u64[E*2]) and the two FUSED expert tensors
     * `...layers.{l}.experts.{gate_up_proj,down_proj}` bound above. The SM reads
     * ewt[eid*2+0]=gate_up base, ewt[eid*2+1]=down base; the per-expert bases are byte offsets into
     * the two fused tensors (base + e*stride). In an fp8 packet, `moe.est.{l}` similarly points
     * at the two per-output-row f32 scale tensors. Dims come from tensor sizes only (no hardcoding):
     * E = ewt_bytes/16, gate_up stride = gu_bytes/E, down stride = dn_bytes/E. Guarded on the
     * moe.ewt tensors' presence, so dense (12B/31B/Qwen) blobs are untouched. */
    for (uint32_t i = 0; i < B.h.n_tensor; i++) {
        if (strncmp(B.tensors[i].name, "moe.ewt.", 8)) continue;
        const int layer = atoi(B.tensors[i].name + 8);
        char suf_gu[96], suf_dn[96], suf_gs[104], suf_ds[104];
        snprintf(suf_gu, sizeof suf_gu, "layers.%d.experts.gate_up_proj", layer);
        snprintf(suf_dn, sizeof suf_dn, "layers.%d.experts.down_proj", layer);
        snprintf(suf_gs, sizeof suf_gs, "layers.%d.experts.gate_up_proj_scale", layer);
        snprintf(suf_ds, sizeof suf_ds, "layers.%d.experts.down_proj_scale", layer);
        int gu = -1, dn = -1, gs = -1, ds = -1, est = -1;
        for (uint32_t j = 0; j < B.h.n_tensor; j++) {
            const char* nm = B.tensors[j].name;
            size_t ln = strlen(nm);
            if (ln >= strlen(suf_gu) && !strcmp(nm + ln - strlen(suf_gu), suf_gu)) gu = (int)j;
            if (ln >= strlen(suf_dn) && !strcmp(nm + ln - strlen(suf_dn), suf_dn)) dn = (int)j;
            if (ln >= strlen(suf_gs) && !strcmp(nm + ln - strlen(suf_gs), suf_gs)) gs = (int)j;
            if (ln >= strlen(suf_ds) && !strcmp(nm + ln - strlen(suf_ds), suf_ds)) ds = (int)j;
            char est_name[32]; snprintf(est_name, sizeof est_name, "moe.est.%d", layer);
            if (!strcmp(nm, est_name)) est = (int)j;
        }
        if (gu < 0 || dn < 0) { printf("MoE: layer %d missing fused expert tensor(s)\n", layer); return 1; }
        const uint64_t E = B.tensors[i].bytes / 16ull;         /* ewt = E*2 u64 */
        const uint64_t gu_stride = B.tensors[gu].bytes / E;    /* bytes per expert's gate_up */
        const uint64_t dn_stride = B.tensors[dn].bytes / E;    /* bytes per expert's down    */
        uint64_t* h_ewt = (uint64_t*)malloc((size_t)E * 2 * 8);
        for (uint64_t e = 0; e < E; e++) {
            h_ewt[e * 2 + 0] = (uint64_t)devp[gu] + e * gu_stride;
            h_ewt[e * 2 + 1] = (uint64_t)devp[dn] + e * dn_stride;
        }
        CK(cudaMemcpy(devp[i], h_ewt, (size_t)E * 2 * 8, cudaMemcpyHostToDevice));
        free(h_ewt);
        const int fp8_experts = !strncmp(B.tensors[gu].name, "fp8/", 4);
        if (fp8_experts) {
            if (est < 0 || gs < 0 || ds < 0) {
                printf("MoE fp8: layer %d missing expert scale tensor/table\n", layer);
                return 1;
            }
            const uint64_t gs_stride = B.tensors[gs].bytes / E;
            const uint64_t ds_stride = B.tensors[ds].bytes / E;
            uint64_t* h_est = (uint64_t*)malloc((size_t)E * 2 * 8);
            for (uint64_t e = 0; e < E; e++) {
                h_est[e * 2 + 0] = (uint64_t)devp[gs] + e * gs_stride;
                h_est[e * 2 + 1] = (uint64_t)devp[ds] + e * ds_stride;
            }
            CK(cudaMemcpy(devp[est], h_est, (size_t)E * 2 * 8, cudaMemcpyHostToDevice));
            free(h_est);
        } else if (est >= 0 || gs >= 0 || ds >= 0) {
            printf("MoE bf16: layer %d unexpectedly declares fp8 scale state\n", layer);
            return 1;
        }
    }
    /* NEGATIVE CONTROL (PLOW_NEGCTRL_WEIGHT=n): zero the first n bf16 elements of
     * layer 18's o_proj (default 4096 = exactly ONE output channel, since K=4096).
     *
     * THE FIRST VERSION OF THIS CONTROL ZEROED ONE SCALAR AND DID NOT FIRE -- all 24
     * tokens still matched. That is reported rather than quietly retuned, because it
     * is a real statement about this gate's sensitivity: one bf16 element out of the
     * 6.5M in this tensor perturbs one of 2560 output elements in 1 of 36 layers, and
     * bf16 rounding absorbs it before the argmax sees it. A greedy-token gate is
     * COARSE -- it sees only the top-1 ordering of the final row -- so it cannot be
     * expected to detect an arbitrarily small perturbation, and claiming otherwise
     * would be the "negative control that nearly produced a fake PASS" failure mode.
     * One whole output channel is the honest calibration point: still a 1/6656 slice
     * of one tensor, and it does fire. */
    if (const char* nw_env = getenv("PLOW_NEGCTRL_WEIGHT")) {
        const char* target = "model.layers.18.self_attn.o_proj.weight";
        int found = -1;
        for (uint32_t i = 0; i < B.h.n_tensor; i++)
            if (!strcmp(B.tensors[i].name, target)) { found = (int)i; break; }
        if (found < 0) { printf("negctrl: no tensor %s\n", target); return 1; }
        size_t nz = (size_t)atoi(nw_env);
        if (nz == 0) nz = 4096;
        if (nz * 2 > B.tensors[found].bytes) nz = B.tensors[found].bytes / 2;
        CK(cudaMemset(devp[found], 0, nz * 2));
        printf("*** NEGATIVE CONTROL: zeroed %zu bf16 element(s) of %s ***\n", nz, target);
    }

    void* d_tens = NULL;
    CK(cudaMalloc(&d_tens, (size_t)B.h.n_tensor * sizeof(void*)));
    CK(cudaMemcpy(d_tens, devp, (size_t)B.h.n_tensor * sizeof(void*), cudaMemcpyHostToDevice));
    printf("bound %d weights (%.2f GiB) + %.3f GiB KV cache in %.1f s\n", nw,
           wb / 1073741824.0, kvb / 1073741824.0, now() - lt0);

    /* ---- upload ALL programs: the prefill buckets (run under the _pf object) and the T=1
     * decode program (run under the decode object). Both share the tensor table above. ---- */
    const int dp = (int)B.h.n_prog - 1;
    Prog* g = &B.prog[dp]; /* decode program is the last one */
    /* BATCHED DECODE (PLOW_DECODE_BATCH=B): the decode program's T IS the batch B — it drives B
     * independent sequence slots in one launch. B=1 is the historical single-slot program.
     * Programs are keyed by INDEX (the last one is decode), never by "t == 1": at B>1 the decode
     * program's t collides with the prefill bucket sizes. */
    const int DB = (int)g->h.t;
    if (DB < 1 || DB > 32) { printf("decode program batch %d out of range\n", DB); return 1; }
    if (DB > 1) printf("BATCHED DECODE: %d sequence slots per launch\n", DB);
    const int grid_pf = plow_sm120_grid_pf(dev);
    if (grid_pf != (int)B.h.n_cu) {
        printf("FATAL: prefill grid %d != packet n_cu %u\n", grid_pf, B.h.n_cu);
        return 1;
    }
    /* T9c: PLOW_NV_SEGMENTS=1 routes prefill through the segmented object (RUNSEG). Default off keeps
     * the byte-identical single-launch _pf object. The segmented object's occupancy must also give a
     * co-resident grid == n_cu (cooperative launch enforces co-residency per segment). */
    const int use_seg = getenv("PLOW_NV_SEGMENTS") && atoi(getenv("PLOW_NV_SEGMENTS")) != 0;
    const int grid_seg = use_seg ? plow_sm120_grid_pfseg(dev) : grid_pf;
    if (use_seg && grid_seg != (int)B.h.n_cu) {
        printf("FATAL: segmented prefill grid %d != packet n_cu %u\n", grid_seg, B.h.n_cu);
        return 1;
    }
    /* T10: PLOW_NV_SEG_GEMM=1 drives the occ-2 lean object on the GEMM (wave-8) segments at
     * grid_gemm = 2*n_cu; flash (wave-4) segments stay on _pfseg at grid_seg = n_cu. The
     * per-segment grid relaxes the old blanket `grid==n_cu` assert: GEMM-class launches carry a
     * 2*n_cu grid whose co-residency the cooperative launch validates at launch time (occ-2 x n_cu
     * SMs). The pkt MUST be re-sliced (PLOW_SEG_CLASS_SLICE=1) so a GEMM op has 2*n_cu blocks. */
    const int use_gemm = use_seg && getenv("PLOW_NV_SEG_GEMM") && atoi(getenv("PLOW_NV_SEG_GEMM")) != 0;
    const int grid_gemm = use_gemm ? plow_sm120_grid_pfgemm(dev) : grid_seg;
    if (use_gemm) {
        printf("T10 occ-2 segmented: GEMM-class grid=%d (2*n_cu=%u) on _pfgemm, flash grid=%d on _pfseg\n",
               grid_gemm, 2u * B.h.n_cu, grid_seg);
        if (grid_gemm < (int)B.h.n_cu) {
            printf("FATAL: _pfgemm grid %d < n_cu %u — lean object failed to reach occ>=1\n",
                   grid_gemm, B.h.n_cu);
            return 1;
        }
    }
    for (uint32_t i = 0; i < B.h.n_prog; i++) {
        const int sched = (i == (uint32_t)dp) ? plow_sm120_sched() : plow_sm120_sched_pf();
        if (prep_prog(&B, &B.prog[i], sched)) return 1;
    }
    printf("uploaded %u programs (prefill grid=%d smem=%zu; decode smem=%zu)\n", B.h.n_prog,
           grid_pf, plow_sm120_smem_pf(), plow_sm120_smem());

    /* logits are [B][vocab] under a batched decode program. */
    const uint32_t VOCAB = (uint32_t)(B.tensors[t_logits].bytes / 2) / (uint32_t)DB;
    const int max_ctx = (int)(B.tensors[t_pos].bytes / 4);
    if (n_prompt + n_gen > max_ctx) {
        printf("prompt %d + n_gen %d exceeds max context %d\n", n_prompt, n_gen, max_ctx);
        return 1;
    }
    printf("vocab=%u  max_ctx=%d  decode program: %u packets, %u wg-packets\n\n", VOCAB,
           max_ctx, g->h.n_inst, g->h.n_stream);

    PlowDevInst* h_inst = NULL;
    CK(cudaMallocHost(&h_inst, (size_t)g->h.n_inst * sizeof(PlowDevInst)));
    memcpy(h_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst));
    int32_t* h_scalar = NULL;
    CK(cudaMallocHost(&h_scalar, 64));
    const size_t zc_bytes = (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4;
    uint32_t* zc = NULL;
    CK(cudaMallocHost(&zc, zc_bytes));
    memset(zc, 0, zc_bytes);
    uint16_t* logit = NULL;
    CK(cudaMallocHost(&logit, (size_t)VOCAB * 2));

    PlowProgram pr;
    mk_pr(&pr, g, d_tens);
    printf("scheduler: %s%s\n", plow_sm120_sched() ? "GLOBAL QUEUE (one atomic cursor)"
                                                   : "STATIC per-block stream",
           plow_sm120_skeleton() ? "   *** SKELETON BUILD: NO OP BODIES, OUTPUT IS GARBAGE ***"
                                 : "");

    /* ===== CTX-REGIME KERNEL SWITCHING (beat12b-ctx-switch) ==========================
     * PLOW_GF_SWITCH=<kvlen>  → for a decode step whose kvlen >= this threshold, launch the
     *                           GF8 twin object on the LONG decode program (PLOW_PKT_LONG, an
     *                           ns94 packet) instead of the shipped GF2 object + this blob's
     *                           (ns47) decode program. Unset/0/off → never switch (this run is
     *                           byte-identical to the pre-change harness).
     * PLOW_GF_SWITCH=alt      → force-alternate GF2/GF8 every step regardless of kvlen. Isolates
     *                           the pure per-step function/program-switch COST from the regime win.
     * The long program shares the SAME tensor table (d_tens): the two packets differ only in the
     * decode flash op's nsplit field + dep map (tensor-table region cmp-identical), and opart/mlpart
     * are sized by the prefill buckets, so the ns94 decode partials fit the ns47 packet's scratch. */
    const char* gfsw_s = getenv("PLOW_GF_SWITCH");
    const bool gfsw_alt = gfsw_s && !strcmp(gfsw_s, "alt");
    const long gf_switch = (gfsw_s && !gfsw_alt && strcmp(gfsw_s, "off")) ? atol(gfsw_s) : 0;
    const bool gf_switch_on = gfsw_alt || gf_switch > 0;
    Blob BL; Prog* gL = NULL; PlowProgram prL; PlowDevInst* h_instL = NULL;
    int grid_gf8 = 0;
    if (gf_switch_on) {
        const char* pktL = getenv("PLOW_PKT_LONG");
        if (!pktL) { printf("FATAL: PLOW_GF_SWITCH set but PLOW_PKT_LONG (ns94 pkt) unset\n"); return 1; }
        if (load_blob(pktL, &BL)) return 1;
        if (BL.h.n_cu != B.h.n_cu || BL.h.n_tensor != B.h.n_tensor) {
            printf("FATAL: long pkt n_cu/n_tensor (%u/%u) != main (%u/%u) — tensor table must match\n",
                   BL.h.n_cu, BL.h.n_tensor, B.h.n_cu, B.h.n_tensor);
            return 1;
        }
        grid_gf8 = plow_sm120_grid_gf8(dev);
        if (grid_gf8 != (int)B.h.n_cu) {
            printf("FATAL: GF8 twin grid %d != n_cu %u\n", grid_gf8, B.h.n_cu);
            return 1;
        }
        gL = &BL.prog[BL.h.n_prog - 1]; /* long decode program is the last */
        if (gL->h.t != g->h.t) { printf("FATAL: long decode batch %u != %u\n", gL->h.t, g->h.t); return 1; }
        /* zc is sized by the SHORT program's counter block and re-zeros the active program's d_ctr
         * each step; the long program must not need more. Same program shape → equal in practice. */
        if (gL->h.n_counter > g->h.n_counter) {
            printf("FATAL: long decode n_counter %u > short %u (zc buffer too small)\n",
                   gL->h.n_counter, g->h.n_counter);
            return 1;
        }
        if (prep_prog(&BL, gL, plow_sm120_sched())) return 1;
        CK(cudaMallocHost(&h_instL, (size_t)gL->h.n_inst * sizeof(PlowDevInst)));
        memcpy(h_instL, gL->insts, (size_t)gL->h.n_inst * sizeof(PlowDevInst));
        mk_pr(&prL, gL, d_tens);
        printf("CTX-SWITCH ON: %s (threshold kvlen>=%ld) GF8 grid=%d smem=%zu; short GF2 smem=%zu\n",
               gfsw_alt ? "ALTERNATE-every-step" : "by-kvlen", gf_switch, grid_gf8,
               plow_sm120_smem_gf8(), plow_sm120_smem());
    }
    long gf_alt_ctr = 0; long gf_long_steps = 0; long gf_short_steps = 0;

    /* ================= CHUNKED PREFILL (the G5 path) =================================
     * Consume the prompt through the PREFILL bucket programs (tiled GEMM + FLASH_PREFILL) rather
     * than one decode launch per token. Each chunk picks the smallest bucket >= its real row count
     * (capped at the largest bucket); the final partial chunk PADS to the bucket and runs only its
     * real rows via patched op fields:
     *   - every KV-writing HeadNormRope (j0!=0): out_row0 (i3) = c0        -> writes cache rows c0..
     *   - FLASH_PREFILL: seq_kv (i1) = c0+real, q_pos0 (i4) = c0           -> causal over history
     *   - lm_head GEMM (M==1): a_row0 (i4) = real-1                        -> logits of the LAST real row
     *   - in.pos[i] = c0+i ; in.ids[0..real-1] = chunk tokens (rest padded)
     * After prefill, in.ids holds the FIRST generated token (argmax over the last prompt row), exactly
     * as the decode-only path's step n_prompt-1 would produce; generation then runs the decode object.
     * Set PLOW_PREFILL=0 to fall back to the decode-only prompt consumption (the Phase-0 reference). */
    /* The prefill buckets fill ONE KV slot, so a batched run consumes the prompt through the
     * decode program instead — which is also exactly what the B=1..B parity gate wants: the same
     * prompt fed to every slot, one batched launch per token. */
    int use_prefill = getenv("PLOW_PREFILL") ? atoi(getenv("PLOW_PREFILL")) : 1;
    if (DB > 1 && use_prefill) {
        printf("batched decode: prompt consumed through the DECODE program (prefill is 1-slot)\n");
        use_prefill = 0;
    }
    /* Bucket sizes present in the blob (every program except the decode one, which is last). */
    uint32_t bucket[32]; int nbuckets = 0, maxbucket = 0;
    for (uint32_t i = 0; i < B.h.n_prog; i++)
        if (i != (uint32_t)dp) {
            bucket[nbuckets++] = B.prog[i].h.t;
            if ((int)B.prog[i].h.t > maxbucket) maxbucket = (int)B.prog[i].h.t;
        }
    /* Reusable host scratch for the ids/pos uploads (sized to the largest bucket). */
    int32_t* h_ids = (int32_t*)malloc((size_t)maxbucket * 4);
    int32_t* h_posv = (int32_t*)malloc((size_t)maxbucket * 4);
    int prefill_first = -1; /* the first generated token, produced by prefill */

    auto find_prog = [&](int t) -> Prog* {
        for (uint32_t i = 0; i < B.h.n_prog; i++)
            if ((int)B.prog[i].h.t == t) return &B.prog[i];
        return NULL;
    };

    /* ==== PLOW_CHUNK_PROF=1: per-chunk overhead decomposition (beat-chunk-overhead) ====
     * Host-side phase timestamps + CUDA events around every chunk launch. Records are
     * printed AFTER the loop so the instrumentation itself stays off the timed path as
     * far as possible (event record ~1us each). Default (unset) adds nothing. */
    const int chunk_prof = getenv("PLOW_CHUNK_PROF") ? atoi(getenv("PLOW_CHUNK_PROF")) : 0;
    enum { CP_MAX = 96 };
    struct ChunkRec {
        int c0, tc, real;
        double patch_us, inst_us, ids_us, memset_us, enq_us, sync_us; /* host phases */
        float kern_ms;  /* ev_start..ev_end */
        float gap_ms;   /* prev ev_end .. this ev_start (device idle) */
    };
    ChunkRec cp_rec[CP_MAX];
    int cp_n = 0;
    cudaEvent_t cp_evs[CP_MAX], cp_eve[CP_MAX];
    if (chunk_prof)
        for (int i = 0; i < CP_MAX; i++) {
            CK(cudaEventCreate(&cp_evs[i]));
            CK(cudaEventCreate(&cp_eve[i]));
        }

    /* ==== PLOW_CHUNK_ASYNC=1: back-to-back chunk launches, ONE host sync ====
     * (beat-chunk-overhead lever A — the multi-step-decode analogue for prefill.)
     * Every chunk's patched instruction stream + ids/pos/kvlen are pre-staged in
     * PINNED memory, then the whole prefill is enqueued on stream 0 as
     * [inst H2D -> ids/pos/kvlen H2D -> ctr/gq memset -> cooperative launch] x N
     * with a single cudaDeviceSynchronize at the end. Stream order makes the
     * in-place d_inst re-patch safe (chunk i+1's copy waits for kernel i).
     * Default (unset) keeps the per-chunk synchronous loop byte-identical. */
    const int chunk_async =
        (getenv("PLOW_CHUNK_ASYNC") ? atoi(getenv("PLOW_CHUNK_ASYNC")) : 0) && !use_seg;
    uint8_t* ca_slab = NULL; /* pinned staging, sized before the timed region */
    size_t ca_slab_bytes = 0;
    uint32_t ca_ninst_max = 0;
    if (chunk_async) {
        for (uint32_t i = 0; i < B.h.n_prog; i++)
            if (i != (uint32_t)dp && B.prog[i].h.n_inst > ca_ninst_max)
                ca_ninst_max = B.prog[i].h.n_inst;
        const int ca_chunks_max = n_prompt / (maxbucket > 0 ? maxbucket : 1) + 2;
        ca_slab_bytes = (size_t)ca_chunks_max *
                        ((size_t)ca_ninst_max * sizeof(PlowDevInst) + (size_t)maxbucket * 8 + 64);
        CK(cudaMallocHost(&ca_slab, ca_slab_bytes));
    }

    double prefill_ms = 0;
    if (use_prefill && n_prompt > 0 && nbuckets > 0 && chunk_async) {
        const double pf0 = now();
        /* pass 1+2: chunk plan + host-side patch into the pinned slab */
        struct CaChunk {
            Prog* gp;
            int c0, tc, real;
            PlowDevInst* inst;
            int32_t *ids, *posv, *kvl;
        };
        CaChunk ca[CP_MAX];
        int nck = 0;
        uint8_t* cur = ca_slab;
        for (int c0 = 0; c0 < n_prompt && nck < CP_MAX;) {
            const int rem = n_prompt - c0;
            int Tc = maxbucket;
            for (int bi = 0; bi < nbuckets; bi++)
                if ((int)bucket[bi] >= rem && (int)bucket[bi] < Tc) Tc = (int)bucket[bi];
            const int real = rem < Tc ? rem : Tc;
            Prog* gp = find_prog(Tc);
            if (!gp) { printf("no prefill bucket for Tc=%d\n", Tc); return 1; }
            CaChunk* k = &ca[nck++];
            k->gp = gp; k->c0 = c0; k->tc = Tc; k->real = real;
            k->inst = (PlowDevInst*)cur; cur += (size_t)gp->h.n_inst * sizeof(PlowDevInst);
            k->ids = (int32_t*)cur;  cur += (size_t)Tc * 4;
            k->posv = (int32_t*)cur; cur += (size_t)Tc * 4;
            k->kvl = (int32_t*)cur;  cur += 64; /* keep chunks 64B-aligned */
            memcpy(k->inst, gp->insts, (size_t)gp->h.n_inst * sizeof(PlowDevInst));
            for (uint32_t j = 0; j < gp->h.n_inst; j++) {
                PlowDevInst* in = &k->inst[j];
                if (in->op == PLOW_DOP_HEADNORM_ROPE && in->fj[1].u != 0)
                    in->i[3] = (uint32_t)c0;
                else if (in->op == PLOW_DOP_FLASH_PREFILL) {
                    in->i[1] = (uint32_t)(c0 + real);
                    in->i[4] = (uint32_t)c0;
                } else if ((in->op == PLOW_DOP_GEMM || in->op == PLOW_DOP_GEMM_SMALL ||
                            in->op == PLOW_DOP_GEMM_MED) && in->i[0] == 1)
                    in->i[4] = (uint32_t)(real - 1);
            }
            for (int i = 0; i < Tc; i++) {
                k->ids[i] = (i < real) ? prompt[c0 + i] : 0;
                k->posv[i] = c0 + i;
            }
            k->kvl[0] = c0 + real;
            c0 += real;
        }
        if ((size_t)(cur - ca_slab) > ca_slab_bytes) {
            printf("FATAL: chunk-async slab overflow (%zu > %zu)\n", (size_t)(cur - ca_slab),
                   ca_slab_bytes);
            return 1;
        }
        const double ca_t_patch = now();
        /* pass 3: enqueue EVERYTHING, no host round-trips */
        PlowProgram ppr;
        for (int i = 0; i < nck; i++) {
            CaChunk* k = &ca[i];
            Prog* gp = k->gp;
            CK(cudaMemcpyAsync(gp->d_inst, k->inst,
                               (size_t)gp->h.n_inst * sizeof(PlowDevInst),
                               cudaMemcpyHostToDevice, 0));
            CK(cudaMemcpyAsync(devp[t_ids], k->ids, (size_t)k->tc * 4,
                               cudaMemcpyHostToDevice, 0));
            CK(cudaMemcpyAsync(devp[t_pos], k->posv, (size_t)k->tc * 4,
                               cudaMemcpyHostToDevice, 0));
            CK(cudaMemcpyAsync(devp[t_kvlen], k->kvl, 4, cudaMemcpyHostToDevice, 0));
            CK(cudaMemsetAsync(gp->d_ctr, 0, (size_t)gp->h.n_counter * PLOW_CTR_STRIDE * 4, 0));
            if (gp->d_gqcursor) CK(cudaMemsetAsync(gp->d_gqcursor, 0, 4, 0));
            mk_pr(&ppr, gp, d_tens);
            if (chunk_prof && i < CP_MAX) CK(cudaEventRecord(cp_evs[i], 0));
            const int rc = plow_sm120_launch_pf(&ppr, grid_pf, 0);
            if (rc != cudaSuccess) {
                printf("PREFILL ASYNC LAUNCH FAILED chunk c0=%d Tc=%d: %s\n", k->c0, k->tc,
                       cudaGetErrorString((cudaError_t)rc));
                return 1;
            }
            if (chunk_prof && i < CP_MAX) CK(cudaEventRecord(cp_eve[i], 0));
        }
        const double ca_t_enq = now();
        CK(cudaDeviceSynchronize());
        const double ca_t_sync = now();
        CK(cudaMemcpy(h_scalar, devp[t_ids], 4, cudaMemcpyDeviceToHost));
        prefill_first = (int)h_scalar[0];
        prefill_ms = (now() - pf0) * 1e3;
        if (chunk_prof) {
            float s_kern = 0, s_gap = 0;
            for (int i = 0; i < nck && i < CP_MAX; i++) {
                float k_ms = 0, g_ms = 0;
                CK(cudaEventElapsedTime(&k_ms, cp_evs[i], cp_eve[i]));
                if (i > 0) CK(cudaEventElapsedTime(&g_ms, cp_eve[i - 1], cp_evs[i]));
                s_kern += k_ms; s_gap += g_ms;
                printf("CHUNK_ASYNC %3d c0=%7d Tc=%6d kern_ms=%9.3f gap_ms=%8.3f\n", i,
                       ca[i].c0, ca[i].tc, k_ms, g_ms);
            }
            printf("CHUNK_ASYNC_SUM chunks=%d patch_ms=%.3f enq_ms=%.3f sync_wait_ms=%.3f "
                   "kern_ms=%.3f gap_ms=%.3f wall_ms=%.2f nonkern_ms=%.3f\n",
                   nck, (ca_t_patch - pf0) * 1e3, (ca_t_enq - ca_t_patch) * 1e3,
                   (ca_t_sync - ca_t_enq) * 1e3, s_kern, s_gap, prefill_ms,
                   prefill_ms - s_kern);
        }
        printf("PREFILL: %d tokens in %.1f ms (%.0f tok/s), first gen token=%d  [CHUNK_ASYNC]\n",
               n_prompt, prefill_ms, n_prompt / (prefill_ms / 1e3), prefill_first);
        printf("PLOW_PREFILL_RESULT n_prompt=%d prefill_ms=%.2f tok_per_s=%.1f\n", n_prompt,
               prefill_ms, n_prompt / (prefill_ms / 1e3));
    } else if (use_prefill && n_prompt > 0 && nbuckets > 0) {
        const double pf0 = now();
        int c0 = 0;
        while (c0 < n_prompt) {
            int rem = n_prompt - c0;
            /* smallest bucket >= rem, else the largest (a full max-size chunk) */
            int Tc = maxbucket, real;
            for (int bi = 0; bi < nbuckets; bi++)
                if ((int)bucket[bi] >= rem && (int)bucket[bi] < Tc) Tc = (int)bucket[bi];
            real = rem < Tc ? rem : Tc;
            Prog* gp = find_prog(Tc);
            if (!gp) { printf("no prefill bucket for Tc=%d\n", Tc); return 1; }
            const double cp_t0 = now();

            /* host-patch this bucket's instruction stream for the chunk */
            PlowDevInst* hi = (PlowDevInst*)malloc((size_t)gp->h.n_inst * sizeof(PlowDevInst));
            memcpy(hi, gp->insts, (size_t)gp->h.n_inst * sizeof(PlowDevInst));
            for (uint32_t k = 0; k < gp->h.n_inst; k++) {
                PlowDevInst* in = &hi[k];
                if (in->op == PLOW_DOP_HEADNORM_ROPE && in->fj[1].u != 0)
                    in->i[3] = (uint32_t)c0; /* KV write out_row0 */
                else if (in->op == PLOW_DOP_FLASH_PREFILL) {
                    in->i[1] = (uint32_t)(c0 + real); /* seq_kv = history + this chunk */
                    in->i[4] = (uint32_t)c0;          /* q_pos0 */
                } else if ((in->op == PLOW_DOP_GEMM || in->op == PLOW_DOP_GEMM_SMALL ||
                            in->op == PLOW_DOP_GEMM_MED) && in->i[0] == 1)
                    in->i[4] = (uint32_t)(real - 1); /* lm_head: logits of the last REAL row */
            }
            const double cp_t1 = now();
            CK(cudaMemcpy(gp->d_inst, hi, (size_t)gp->h.n_inst * sizeof(PlowDevInst),
                          cudaMemcpyHostToDevice));
            free(hi);
            const double cp_t2 = now();

            /* ids (real tokens + pad) and positions for this chunk */
            for (int i = 0; i < Tc; i++) {
                h_ids[i] = (i < real) ? prompt[c0 + i] : 0;
                h_posv[i] = c0 + i;
            }
            CK(cudaMemcpy(devp[t_ids], h_ids, (size_t)Tc * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(devp[t_pos], h_posv, (size_t)Tc * 4, cudaMemcpyHostToDevice));
            const int32_t kvl = c0 + real;
            CK(cudaMemcpy(devp[t_kvlen], &kvl, 4, cudaMemcpyHostToDevice));
            const double cp_t3 = now();
            /* Zero this program's OWN counter block (the prefill buckets have more counters than
             * the decode program, so the decode-sized `zc` staging buffer cannot cover them). */
            CK(cudaMemset(gp->d_ctr, 0, (size_t)gp->h.n_counter * PLOW_CTR_STRIDE * 4));

            PlowProgram ppr;
            mk_pr(&ppr, gp, d_tens);
            if (use_seg) {
                /* RUNSEG (T9c, mirrors amd/interp.hip): zero ALL n_seg cursor lines ONCE, then
                 * enqueue one cooperative launch per wave-class segment on the SAME stream. Stream
                 * order serialises the segments, so a producer run in an earlier segment's launch has
                 * its counter (zeroed once, above) satisfied when a later segment gates on it. One
                 * device sync drains them all — the K-1 extra launch+sync tax is measured here. */
                if (gp->d_gqcursor)
                    CK(cudaMemset(gp->d_gqcursor, 0, (size_t)gp->n_seg * PLOW_CTR_STRIDE * 4));
                const double se0 = now();
                for (uint32_t s = 0; s < gp->n_seg; s++) {
                    ppr.cur_seg = s;
                    /* T10: GEMM-class segments run on the occ-2 _pfgemm object at grid_gemm; flash
                     * (and the whole T9c path when use_gemm==0) stay on _pfseg at grid_seg. */
                    const int gemm_seg =
                        use_gemm && gp->seg_is_flash && !gp->seg_is_flash[s];
                    const int rc = gemm_seg ? plow_sm120_launch_pfgemm(&ppr, grid_gemm, 0)
                                            : plow_sm120_launch_pfseg(&ppr, grid_seg, 0);
                    if (rc != cudaSuccess) {
                        printf("PREFILL SEG LAUNCH FAILED chunk c0=%d Tc=%d seg %u/%u: %s\n", c0, Tc,
                               s, gp->n_seg, cudaGetErrorString((cudaError_t)rc));
                        return 1;
                    }
                }
                const double se1 = now();
                CK(cudaDeviceSynchronize());
                g_seg_enq_us += (se1 - se0) * 1e6;
                g_seg_drain_us += (now() - se1) * 1e6;
                g_seg_launches += gp->n_seg;
                g_runseg_calls++;
            } else {
                if (gp->d_gqcursor) CK(cudaMemset(gp->d_gqcursor, 0, 4));
                const double cp_t4 = now();
                if (chunk_prof && cp_n < CP_MAX) CK(cudaEventRecord(cp_evs[cp_n], 0));
                const int rc = plow_sm120_launch_pf(&ppr, grid_pf, 0);
                if (rc != cudaSuccess) {
                    printf("PREFILL LAUNCH FAILED chunk c0=%d Tc=%d: %s\n", c0, Tc,
                           cudaGetErrorString((cudaError_t)rc));
                    return 1;
                }
                if (chunk_prof && cp_n < CP_MAX) CK(cudaEventRecord(cp_eve[cp_n], 0));
                const double cp_t5 = now();
                CK(cudaDeviceSynchronize());
                if (chunk_prof && cp_n < CP_MAX) {
                    ChunkRec* r = &cp_rec[cp_n];
                    r->c0 = c0; r->tc = Tc; r->real = real;
                    r->patch_us = (cp_t1 - cp_t0) * 1e6;
                    r->inst_us = (cp_t2 - cp_t1) * 1e6;
                    r->ids_us = (cp_t3 - cp_t2) * 1e6;
                    r->memset_us = (cp_t4 - cp_t3) * 1e6;
                    r->enq_us = (cp_t5 - cp_t4) * 1e6;
                    r->sync_us = (now() - cp_t5) * 1e6;
                    cp_n++;
                }
            }
            c0 += real;
        }
        const double cp_rb0 = now();
        CK(cudaMemcpy(h_scalar, devp[t_ids], 4, cudaMemcpyDeviceToHost));
        const double cp_rb_us = (now() - cp_rb0) * 1e6;
        prefill_first = (int)h_scalar[0];
        prefill_ms = (now() - pf0) * 1e3;
        if (chunk_prof && cp_n > 0) {
            double s_patch = 0, s_inst = 0, s_ids = 0, s_ms = 0, s_enq = 0, s_sync = 0,
                   s_kern = 0, s_gap = 0;
            printf("CHUNK_PROF chunks=%d (host us | device ms)\n", cp_n);
            printf("%3s %7s %6s | %8s %8s %8s %8s %8s %10s | %9s %8s %9s\n", "i", "c0", "Tc",
                   "patch_us", "inst_us", "ids_us", "mset_us", "enq_us", "sync_us", "kern_ms",
                   "gap_ms", "ovh_us");
            for (int i = 0; i < cp_n; i++) {
                ChunkRec* r = &cp_rec[i];
                CK(cudaEventElapsedTime(&r->kern_ms, cp_evs[i], cp_eve[i]));
                r->gap_ms = 0.f;
                if (i > 0) CK(cudaEventElapsedTime(&r->gap_ms, cp_eve[i - 1], cp_evs[i]));
                const double ovh = r->patch_us + r->inst_us + r->ids_us + r->memset_us +
                                   r->enq_us + (r->sync_us - r->kern_ms * 1e3);
                printf("%3d %7d %6d | %8.1f %8.1f %8.1f %8.1f %8.1f %10.1f | %9.3f %8.3f %9.1f\n",
                       i, r->c0, r->tc, r->patch_us, r->inst_us, r->ids_us, r->memset_us,
                       r->enq_us, r->sync_us, r->kern_ms, r->gap_ms, ovh);
                s_patch += r->patch_us; s_inst += r->inst_us; s_ids += r->ids_us;
                s_ms += r->memset_us; s_enq += r->enq_us; s_sync += r->sync_us;
                s_kern += r->kern_ms; s_gap += r->gap_ms;
            }
            printf("CHUNK_PROF_SUM chunks=%d patch_us=%.1f inst_us=%.1f ids_us=%.1f mset_us=%.1f "
                   "enq_us=%.1f sync_us=%.1f kern_ms=%.3f gap_ms=%.3f readback_us=%.1f "
                   "wall_ms=%.2f nonkern_ms=%.3f\n",
                   cp_n, s_patch, s_inst, s_ids, s_ms, s_enq, s_sync, s_kern, s_gap, cp_rb_us,
                   (now() - pf0) * 1e3, (now() - pf0) * 1e3 - s_kern);
        }
        printf("PREFILL: %d tokens in %.1f ms (%.0f tok/s), first gen token=%d\n", n_prompt,
               prefill_ms, n_prompt / (prefill_ms / 1e3), prefill_first);
        printf("PLOW_PREFILL_RESULT n_prompt=%d prefill_ms=%.2f tok_per_s=%.1f\n", n_prompt,
               prefill_ms, n_prompt / (prefill_ms / 1e3));
        if (use_seg)
            printf("PLOW_SEG_TAX runseg_calls=%llu launches=%llu enq_us=%.1f drain_us=%.1f "
                   "enq_per_launch_us=%.2f\n",
                   (unsigned long long)g_runseg_calls, (unsigned long long)g_seg_launches,
                   g_seg_enq_us, g_seg_drain_us,
                   g_seg_launches ? g_seg_enq_us / (double)g_seg_launches : 0.0);
    }

    /* ONE decode step at absolute position `pos`, with kv valid up to pos+1.
     * `feed` >= 0 forces in.ids (prompt consumption); feed < 0 leaves in.ids as the
     * device wrote it at the end of the previous step (generation).
     *
     * THIS WAS A MACRO AND THE MACRO WAS WRONG. Its body declared `uint32_t i` to walk
     * the kvrow table, which SHADOWED the caller's `for (int i = 0; i < n_prompt; i++)`
     * -- so `STEP(i, ...)` expanded to patch the KV row with the kvrow table INDEX
     * (0..71) instead of the position, and every prompt token would have appended to the
     * wrong cache row. nvcc flagged it (warning #780-D) and it would otherwise have
     * presented as a numerics bug in flash-decode. A lambda has real parameters and
     * cannot capture-by-accident like this; the macro bought nothing. */
    /* NEGATIVE CONTROLS. A 24/24 token match against HF is only evidence if this
     * harness can be made to FAIL, so both knobs below reintroduce a specific real
     * defect and must break the match. Off by default; each is one env var.
     *
     *  PLOW_NEGCTRL_KVROW reproduces EXACTLY the macro-shadowing bug this file was
     *      written with: patch the KV write row with the kvrow table index k instead
     *      of the position. It is the highest-value control here because it is not a
     *      hypothetical -- it is the bug nvcc's warning #780-D caught, and this proves
     *      the difference between the broken and fixed forms is observable in tokens.
     *  PLOW_NEGCTRL_WEIGHT zeroes one bf16 element of one mid-stack weight tensor.
     *      A single scalar out of 4.0e9 -- the tightest perturbation that should still
     *      be caught, and the one that shows the gate is not merely detecting rubble. */
    const int nc_kvrow = getenv("PLOW_NEGCTRL_KVROW") != NULL;
    if (nc_kvrow) printf("*** NEGATIVE CONTROL: KV row = kvrow index, not position ***\n");

    /* HOST-OVERHEAD BREAKDOWN. Every step re-uploads the kv-row-patched instruction range,
     * three scalars and the zeroed counter block before launching. vLLM's tpot is measured
     * with CUDA graphs ON and pays none of this, so an undecomposed plow ms/token compares a
     * host-prologue+kernel against a graph replay. These accumulate the two halves so the
     * asymmetry is reported as a number instead of being buried in the mean. */
    double host_pro = 0, dev_ker = 0;
    int brk_on = 0;
    /* Per-slot scalar staging: at B>1 every slot gets its own ids/pos/kvlen entry. This gate
     * drives all B slots with the SAME prompt and the SAME position, so every slot must emit
     * the same token as the B=1 run; a per-slot indexing bug shows up as a slot that diverges. */
    int32_t* h_slot = NULL;
    CK(cudaMallocHost(&h_slot, (size_t)DB * 4 * 3));
    auto run_step = [&](int pos, int feed) -> int {
        const double b0 = brk_on ? now() : 0;
        /* CTX-REGIME SWITCH: pick the active decode object+program for THIS step. kvlen = pos+1.
         * alt mode toggles every step (switch-cost isolation); by-kvlen uses the threshold. When
         * off, aP/apr/ainst/... are exactly the shipped GF2 path, so the launch is byte-identical. */
        const int kvlen = pos + 1;
        bool use_long = false;
        if (gf_switch_on) use_long = gfsw_alt ? ((gf_alt_ctr++ & 1) != 0) : (kvlen >= gf_switch);
        Prog* aP = use_long ? gL : g;
        PlowProgram* apr = use_long ? &prL : &pr;
        PlowDevInst* ainst = use_long ? h_instL : h_inst;
        const uint32_t* akvrow = use_long ? BL.kvrow : B.kvrow;
        const uint32_t an_kvrow = use_long ? BL.h.n_kvrow : B.h.n_kvrow;
        int (*alaunch)(PlowProgram*, int, cudaStream_t) = use_long ? plow_sm120_launch_gf8
                                                                   : plow_sm120_launch;
        const int agrid = use_long ? grid_gf8 : grid;
        if (use_long) gf_long_steps++; else gf_short_steps++;
        /* The kv-row patch is a B=1 mechanism: at B>1 the KV write row is derived on-device
         * from each slot's own pos[b] (op_norm.cuh n_batch_kv), so patching would be wrong. */
        if (DB == 1) {
            uint32_t lo = aP->h.n_inst - 1, hi = 0;
            for (uint32_t k = 0; k < an_kvrow; k++) {
                const uint32_t ix = akvrow[k];
                ainst[ix].i[3] = nc_kvrow ? (uint32_t)k : (uint32_t)pos;
                if (ix < lo) lo = ix;
                if (ix > hi) hi = ix;
            }
            if (an_kvrow)
                CK(cudaMemcpy((uint8_t*)aP->d_inst + (size_t)lo * sizeof(PlowDevInst),
                              &ainst[lo], (size_t)(hi - lo + 1) * sizeof(PlowDevInst),
                              cudaMemcpyHostToDevice));
        }
        if (feed >= 0) {
            /* `feed` is the PROMPT INDEX (not the token) so odd slots can be fed their own
             * stream when PLOW_PROMPT2 is set. */
            for (int b = 0; b < DB; b++)
                h_slot[b] = (prompt2 && (b & 1)) ? prompt2[feed] : prompt[feed];
            CK(cudaMemcpy(devp[t_ids], h_slot, (size_t)DB * 4, cudaMemcpyHostToDevice));
        }
        for (int b = 0; b < DB; b++) h_slot[b] = (int32_t)pos;
        CK(cudaMemcpy(devp[t_pos], h_slot, (size_t)DB * 4, cudaMemcpyHostToDevice));
        for (int b = 0; b < DB; b++) h_slot[b] = (int32_t)(pos + 1);
        CK(cudaMemcpy(devp[t_kvlen], h_slot, (size_t)DB * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(aP->d_ctr, zc, zc_bytes, cudaMemcpyHostToDevice));
        /* The GQ cursor has the same lifecycle as the counters: one launch consumes it
         * from 0 to n_stream, so it must be re-zeroed before every launch. */
        if (aP->d_gqcursor) CK(cudaMemset(aP->d_gqcursor, 0, 4));
        /* The prologue is all synchronous cudaMemcpy on the default stream, so it has
         * already retired on the device by the time the last one returns; timing the split
         * here does not perturb the launch that follows. */
        const double b1 = brk_on ? now() : 0;
        const int rc = alaunch(apr, agrid, 0);
        if (rc != cudaSuccess) {
            printf("LAUNCH FAILED at pos %d: %s\n", pos,
                   cudaGetErrorString((cudaError_t)rc));
            return 1;
        }
        CK(cudaDeviceSynchronize());
        if (brk_on) { host_pro += b1 - b0; dev_ker += now() - b1; }
        return 0;
    };

    /* ---- consume the prompt ----
     * PREFILL path: already done above (the KV cache is built and in.ids holds the first gen
     * token). DECODE-ONLY path (PLOW_PREFILL=0, the Phase-0 reference): one decode step per token. */
    int best;
    if (use_prefill && prefill_first >= 0) {
        best = prefill_first;
        printf("prompt consumed via PREFILL buckets (%.1f ms)\n", prefill_ms);
    } else {
        const double p0 = now();
        for (int i = 0; i < n_prompt; i++)
            if (run_step(i, i)) return 1; /* second arg is the PROMPT INDEX */
        const double pdt = now() - p0;
        printf("prompt: %d tokens through the DECODE program in %.0f ms (%.1f tok/s)\n",
               n_prompt, pdt * 1e3, n_prompt / pdt);
        CK(cudaMemcpy(h_scalar, devp[t_ids], 4, cudaMemcpyDeviceToHost));
        best = (int)h_scalar[0];
    }
    /* SLOT PARITY: read all B slot tokens. Slots fed the same prompt must produce the same
     * token; with PLOW_PROMPT2 the even and odd slots form two groups, each of which must be
     * internally identical AND must match that prompt's own B=1 stream (compared off-line from
     * the PLOW_IDS / PLOW_IDS_ODD dumps). Returns slot 0's token to drive the printed stream. */
    int slot_div = 0, slot_div_step = -1;
    std::vector<int> odd_ids;
    auto read_slots = [&](int step) -> int {
        CK(cudaMemcpy(h_slot, devp[t_ids], (size_t)DB * 4, cudaMemcpyDeviceToHost));
        for (int b = 1; b < DB; b++) {
            const int ref = (prompt2 && (b & 1)) ? (int)h_slot[1] : (int)h_slot[0];
            if ((int)h_slot[b] != ref) {
                if (!slot_div) {
                    slot_div_step = step;
                    printf("\n*** SLOT DIVERGENCE at step %d: slot %d gave %d, expected %d\n",
                           step, b, (int)h_slot[b], ref);
                }
                slot_div++;
            }
        }
        if (prompt2 && DB > 1) odd_ids.push_back((int)h_slot[1]);
        return (int)h_slot[0];
    };
    if (!use_prefill) best = read_slots(-1);

    /* TOP-5 AT STEP 0, against the HF reference's own top-5. The logits row is the
     * single most diagnostic artifact here: a matching argmax with a mismatched
     * ordering underneath means the model is nearly right and about to diverge. */
    CK(cudaMemcpy(logit, devp[t_logits], (size_t)VOCAB * 2, cudaMemcpyDeviceToHost));
    {
        int ti[5];
        float tv[5];
        for (int k = 0; k < 5; k++) { ti[k] = -1; tv[k] = -1e30f; }
        for (uint32_t v = 0; v < VOCAB; v++) {
            float x = b2f(logit[v]);
            for (int k = 0; k < 5; k++)
                if (x > tv[k]) {
                    for (int j = 4; j > k; j--) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; }
                    tv[k] = x; ti[k] = (int)v;
                    break;
                }
        }
        printf("\nstep 0 top-5 logits (plow sm_120):\n");
        for (int k = 0; k < 5; k++) printf("  id=%-8d logit=%+10.5f\n", ti[k], tv[k]);
        /* The device argmax and a host scan of the same row must agree. They read
         * the SAME bf16 logits, so this isolates the ARGMAX/ARGMAX_FIN packed-key
         * reduction from the rest of the network -- a mismatch here is a reduction
         * bug, not a numerics drift. */
        printf("argmax check: device=%d host=%d  %s\n", best, ti[0],
               best == ti[0] ? "AGREE" : "*** DISAGREE ***");
        /* BIT-IDENTITY ARTIFACT. Reordering *scheduling* must not change arithmetic, so the
         * step-0 logits row is dumped raw for an external memcmp between the static and GQ
         * runs. Top-5 agreement is far too coarse: this campaign already measured a case
         * where zeroing 62% of an o_proj changed NO tokens. */
        if (getenv("PLOW_DUMP_LOGITS")) {
            FILE* lf = fopen(getenv("PLOW_DUMP_LOGITS"), "wb");
            if (lf) {
                fwrite(logit, 2, VOCAB, lf);
                fclose(lf);
                printf("logits dumped: %s (%u bf16)\n", getenv("PLOW_DUMP_LOGITS"), VOCAB);
            }
        }
    }

    /* ---- greedy generation ---- */
    printf("\nPLOW_IDS");
    fflush(stdout);
    int ctx = n_prompt;
    double dsum = 0;
    int ngen = 0;
    /* STEADY STATE. The first decode steps after the prompt loop are not representative:
     * the clocks have just been loaded by n_prompt back-to-back launches, and the very
     * first step also faults in the freshly-grown KV pages. PLOW_WARMUP steps are run and
     * TIMED BUT DISCARDED from the reported mean, so the reported number is a warm
     * steady state rather than an average over the ramp. Per-step times are kept so the
     * spread can be reported: a mean without a spread cannot be checked for the
     * per-launch artifact this campaign already hit once. */
    const int warmup = getenv("PLOW_WARMUP") ? atoi(getenv("PLOW_WARMUP")) : 16;
    double* dt = (double*)malloc(sizeof(double) * (size_t)(n_gen > 0 ? n_gen : 1));
    for (int step = 0; step < n_gen; step++) {
        printf(" %d", best);
        fflush(stdout);
        prompt[ctx] = best;
        if (ctx + 1 >= max_ctx) break;
        const double d0 = now();
        /* in.ids already holds `best`, written by the device's own ARGMAX_FIN */
        if (run_step(ctx, -1)) return 1;
        const double d = now() - d0;
        if (step >= warmup) { dt[ngen] = d; dsum += d; ngen++; }
        brk_on = (step + 1 >= warmup); /* accumulate the split over the timed steps only */
        ctx++;
        best = read_slots(step);
    }
    printf("\n");
    if (DB > 1)
        printf("SLOT PARITY: %d slots, %d divergent slot-steps%s\n", DB, slot_div,
               slot_div ? " *** FAIL ***" : " -- every slot matched its prompt group");
    if (prompt2 && !odd_ids.empty()) {
        printf("PLOW_IDS_ODD");
        for (size_t i = 0; i < odd_ids.size(); i++) printf(" %d", odd_ids[i]);
        printf("\n");
    }
    (void)slot_div_step;
    if (ngen) {
        const double mean = dsum / ngen;
        double var = 0, mn = 1e30, mx = -1e30;
        for (int i = 0; i < ngen; i++) {
            const double e = dt[i] - mean;
            var += e * e;
            if (dt[i] < mn) mn = dt[i];
            if (dt[i] > mx) mx = dt[i];
        }
        const double sd = ngen > 1 ? sqrt(var / (ngen - 1)) : 0.0;
        if (getenv("PLOW_DUMP_STEPS")) {
            printf("\nPLOW_STEPS_RAW");
            for (int i = 0; i < ngen; i++) printf(" %.3f", dt[i] * 1e3);
            printf("\n");
        }
        /* median: insertion sort, n<=few hundred */
        for (int i = 1; i < ngen; i++) {
            const double v = dt[i];
            int j = i - 1;
            while (j >= 0 && dt[j] > v) { dt[j + 1] = dt[j]; j--; }
            dt[j + 1] = v;
        }
        const double med = ngen & 1 ? dt[ngen / 2] : 0.5 * (dt[ngen / 2 - 1] + dt[ngen / 2]);
        if (getenv("PLOW_DUMP_STEPS")) {
            printf("\nPLOW_STEPS_SORTED");
            for (int i = 0; i < ngen; i++) printf(" %.3f", dt[i] * 1e3);
            printf("\n");
        }
        printf("\ndecode STEADY STATE (%d warmup steps discarded):\n", warmup);
        printf("  timed steps : %d   ctx %d -> %d\n", ngen, n_prompt + warmup, ctx);
        printf("  mean        : %.3f ms/token (%.1f tok/s)\n", mean * 1e3, 1.0 / mean);
        printf("  median      : %.3f ms/token\n", med * 1e3);
        /* Batched: one launch advances all B slots, so the launch time IS each slot's TPOT and
         * the aggregate is B/launch. Reported explicitly so a batch run is never read as if the
         * per-launch mean were the throughput. */
        printf("  PLOW_BATCH_PERF B=%d tpot_ms=%.3f tpot_med_ms=%.3f agg_tok_s=%.1f\n", DB,
               mean * 1e3, med * 1e3, (double)DB / mean);
        printf("  stddev      : %.3f ms  (%.2f%% of mean)\n", sd * 1e3, 100.0 * sd / mean);
        printf("  min / max   : %.3f / %.3f ms\n", mn * 1e3, mx * 1e3);
        printf("  host prologue: %.3f ms/step  (kv-row patch + 3 scalars + counter zero)\n",
               host_pro / ngen * 1e3);
        printf("  launch+sync  : %.3f ms/step  <- the part vLLM's graph replay also pays\n",
               dev_ker / ngen * 1e3);
        printf("PLOW_RESULT ctx=%d mean_ms=%.4f median_ms=%.4f sd_ms=%.4f n=%d "
               "host_ms=%.4f kernel_ms=%.4f\n",
               n_prompt, mean * 1e3, med * 1e3, sd * 1e3, ngen,
               host_pro / ngen * 1e3, dev_ker / ngen * 1e3);
        if (gf_switch_on)
            printf("PLOW_GF_SWITCH_STATS mode=%s threshold=%ld long_steps=%ld short_steps=%ld\n",
                   gfsw_alt ? "alt" : "kvlen", gf_switch, gf_long_steps, gf_short_steps);
    }
    free(dt);

    /* ---- PLOW_DUMP_KV: raw KV-ring dump for the P10 kv-zip audit (KV-0) --
     * Dumps every kv.* cache tensor VERBATIM (head-major ring bytes exactly as d_headnorm_rope
     * wrote them) plus a manifest carrying the final ctx, so the offline audit can slice valid
     * rows. Tensors are dumped WHOLE rather than [0..ctx): ring geometry (kvh/ring/hd) is
     * inferred offline from the byte size, and a partial dump would bake that inference in here. */
    if (getenv("PLOW_DUMP_KV")) {
        const char* kdir = getenv("PLOW_DUMP_KV");
        char kpath[1024];
        snprintf(kpath, sizeof kpath, "%s/manifest.txt", kdir);
        FILE* mf = fopen(kpath, "w");
        if (!mf) {
            printf("PLOW_DUMP_KV: cannot open %s\n", kpath);
        } else {
            fprintf(mf, "ctx %d n_prompt %d n_batch %d\n", ctx, n_prompt, DB);
            uint64_t dumped = 0;
            void* hbuf = NULL;
            size_t hcap = 0;
            for (uint32_t i = 0; i < B.h.n_tensor; i++) {
                PlowTensorDecl* td = &B.tensors[i];
                if (strncmp(td->name, "kv.", 3)) continue;
                if (td->bytes > hcap) {
                    free(hbuf);
                    hcap = td->bytes;
                    hbuf = malloc(hcap);
                    if (!hbuf) { printf("PLOW_DUMP_KV: OOM at %llu B\n",
                                        (unsigned long long)td->bytes); break; }
                }
                CK(cudaMemcpy(hbuf, devp[i], td->bytes, cudaMemcpyDeviceToHost));
                snprintf(kpath, sizeof kpath, "%s/%s.raw", kdir, td->name);
                FILE* tf = fopen(kpath, "wb");
                if (!tf || fwrite(hbuf, 1, td->bytes, tf) != td->bytes) {
                    printf("PLOW_DUMP_KV: write failed %s\n", kpath);
                    if (tf) fclose(tf);
                    break;
                }
                fclose(tf);
                fprintf(mf, "%s %llu\n", td->name, (unsigned long long)td->bytes);
                dumped += td->bytes;
            }
            free(hbuf);
            fclose(mf);
            printf("PLOW_DUMP_KV: %.2f GiB of kv.* tensors -> %s\n",
                   dumped / 1073741824.0, kdir);
        }
    }

    /* ---- PLOW_NV_TRACE dump: per-op cycle attribution for block 0 ---- */
#if PLOW_NV_TRACE
    {
        // Externs sized to the interp's PLOW_TRACE_MAX (4096) so cudaMemcpyFromSymbol has a
        // complete type when this TU links libplow_interp_sm120 (incomplete [] fails to compile).
        enum { PLOW_TRACE_MAX = 4096 };
        extern __device__ unsigned g_tr_n;
        extern __device__ unsigned g_tr_op[PLOW_TRACE_MAX];
        extern __device__ unsigned g_tr_wait[PLOW_TRACE_MAX];
        extern __device__ unsigned long long g_tr_gate[PLOW_TRACE_MAX];
        extern __device__ unsigned long long g_tr_body[PLOW_TRACE_MAX];
        extern __device__ unsigned long long g_tr_sig[PLOW_TRACE_MAX];
        unsigned h_n = 0;
        CK(cudaMemcpyFromSymbol(&h_n, g_tr_n, sizeof(unsigned)));
        if (h_n > 0) {
            if (h_n > 4096) h_n = 4096;
            unsigned* h_op = (unsigned*)malloc(h_n * sizeof(unsigned));
            unsigned* h_wait = (unsigned*)malloc(h_n * sizeof(unsigned));
            unsigned long long* h_gate = (unsigned long long*)malloc(h_n * 8);
            unsigned long long* h_body = (unsigned long long*)malloc(h_n * 8);
            unsigned long long* h_sig = (unsigned long long*)malloc(h_n * 8);
            CK(cudaMemcpyFromSymbol(h_op, g_tr_op, h_n * sizeof(unsigned)));
            CK(cudaMemcpyFromSymbol(h_wait, g_tr_wait, h_n * sizeof(unsigned)));
            CK(cudaMemcpyFromSymbol(h_gate, g_tr_gate, h_n * 8));
            CK(cudaMemcpyFromSymbol(h_body, g_tr_body, h_n * 8));
            CK(cudaMemcpyFromSymbol(h_sig, g_tr_sig, h_n * 8));
            unsigned long long total_gate = 0, total_body = 0, total_sig = 0;
            for (unsigned i = 0; i < h_n; i++) {
                total_gate += h_gate[i];
                total_body += h_body[i];
                total_sig += h_sig[i];
            }
            printf("\n== PLOW TRACE (block 0, %u packets) ==\n", h_n);
            printf("  total_gate: %llu cycles (%.1f%%)\n", total_gate,
                   100.0 * total_gate / (total_gate + total_body + total_sig));
            printf("  total_body: %llu cycles (%.1f%%)\n", total_body,
                   100.0 * total_body / (total_gate + total_body + total_sig));
            printf("  total_sig:  %llu cycles (%.1f%%)\n", total_sig,
                   100.0 * total_sig / (total_gate + total_body + total_sig));
            printf("\n  top gate waits (cycles desc):\n");
            /* sort by gate time, print top 20 */
            unsigned* idx = (unsigned*)malloc(h_n * sizeof(unsigned));
            for (unsigned i = 0; i < h_n; i++) idx[i] = i;
            for (unsigned i = 0; i < h_n && i < 20; i++)
                for (unsigned j = i + 1; j < h_n; j++)
                    if (h_gate[idx[j]] > h_gate[idx[i]]) { unsigned t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
            for (unsigned i = 0; i < 20 && i < h_n; i++) {
                unsigned k = idx[i];
                printf("    [%3u] op=%2u  wait=%u  gate=%7llu  body=%7llu  sig=%5llu\n",
                       k, h_op[k], h_wait[k], h_gate[k], h_body[k], h_sig[k]);
            }
            free(h_op); free(h_wait); free(h_gate); free(h_body); free(h_sig); free(idx);
        }
    }
#endif

    return 0;
}
