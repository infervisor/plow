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

extern "C" int plow_sm120_grid(int dev);
extern "C" size_t plow_sm120_smem(void);
extern "C" int plow_sm120_launch(PlowProgram* prog, int grid, cudaStream_t stream);
extern "C" int plow_sm120_sched(void);    /* 0 = static per-block stream, 1 = global queue */
extern "C" int plow_sm120_skeleton(void); /* 1 = gate/signal only, GARBAGE output by design */

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
} Prog;

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

    /* fp8 weight twins (PLOW_FP8_DIR). A fp8 decode packet declares uint8 weights + f32 scales
     * under an "fp8/" name prefix (crates/plowc/src/bin/gemma4.rs w8()/sc()); this dir holds
     * their quantized bytes (perf-data/tools/quantize_fp8.py output, one model.safetensors).
     * st_find ignores the dtype string, so the raw byte range is bound verbatim. */
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
            /* fp8 weight/scale twin: bind from the fp8 checkpoint (uint8 W, f32 scale). */
            if (!have_fp8) {
                printf("FATAL: packet declares fp8 twin '%s' but PLOW_FP8_DIR is unset.\n"
                       "       Compile the fp8 twins with quantize_fp8.py and point\n"
                       "       PLOW_FP8_DIR at their directory.\n", td->name);
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

    /* ---- upload the DECODE program's tables (the only program we can run) ---- */
    const int dp = (int)B.h.n_prog - 1;
    Prog* g = &B.prog[dp];
    if (g->h.t != 1) { printf("last program is not the T=1 decode program\n"); return 1; }
    CK(cudaMalloc(&g->d_inst, (size_t)g->h.n_inst * sizeof(PlowDevInst)));
    CK(cudaMalloc(&g->d_stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt)));
    CK(cudaMalloc(&g->d_sofs, (size_t)B.h.n_cu * 4));
    CK(cudaMalloc(&g->d_slen, (size_t)B.h.n_cu * 4));
    CK(cudaMalloc(&g->d_waits, (size_t)(g->h.n_wait ? g->h.n_wait : 1) * sizeof(PlowWait)));
    CK(cudaMalloc(&g->d_succs, (size_t)(g->h.n_succ ? g->h.n_succ : 1) * 4));
    CK(cudaMalloc(&g->d_ctr, (size_t)g->h.n_counter * PLOW_CTR_STRIDE * 4));
    CK(cudaMemcpy(g->d_inst, g->insts, (size_t)g->h.n_inst * sizeof(PlowDevInst),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_stream, g->stream, (size_t)g->h.n_stream * sizeof(PlowStreamEnt),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_sofs, g->stream_ofs, (size_t)B.h.n_cu * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g->d_slen, g->stream_len, (size_t)B.h.n_cu * 4, cudaMemcpyHostToDevice));
    if (g->h.n_wait)
        CK(cudaMemcpy(g->d_waits, g->waits, (size_t)g->h.n_wait * sizeof(PlowWait),
                      cudaMemcpyHostToDevice));
    if (g->h.n_succ)
        CK(cudaMemcpy(g->d_succs, g->succs, (size_t)g->h.n_succ * 4, cudaMemcpyHostToDevice));

    /* The decode program must be a SINGLE coarse segment: this interpreter traps on
     * PLOW_SE_FINE / PLOW_SE_XCTR entries and has no segmented relaunch path. Check
     * it here, where the message can say so, rather than discovering it as a trap. */
    for (uint32_t j = 0; j < g->h.n_stream; j++) {
        if (g->stream[j].seg != 0 ||
            (g->stream[j].flags & (PLOW_SE_FINE | PLOW_SE_XCTR))) {
            printf("FATAL: decode stream entry %u is segmented/fine-gated; this "
                   "interpreter implements the coarse single-segment path only.\n", j);
            return 1;
        }
    }

    /* ================= GLOBAL QUEUE (M5, the AMD Experiment E1 analogue) =================
     * The GQ kernel needs an OP-MAJOR permutation of the very same stream entries: all entries
     * of instruction 0, then all of instruction 1, ... Every entry is preserved exactly once,
     * so the set of (inst, slice) work items is identical to the static path and the arithmetic
     * cannot change — only which block runs which item.
     *
     * DEADLOCK FREEDOM rests on ONE property: instruction indices must be a TOPOLOGICAL order
     * of the counter DAG. Then the globally-minimum claimed-but-unretired entry m has every
     * producer of every counter it waits on at a LOWER instruction index, hence at a lower
     * gq index, hence already retired (claims are dense and monotone from 0). So m always makes
     * progress. That property is ASSERTED below against the actual packet, not assumed: an
     * out-of-order packet would hang the GPU, and a hang is the worst possible way to find out. */
    if (plow_sm120_sched() == 1) {
        const uint32_t NI = g->h.n_inst, NS = g->h.n_stream;

        /* producer_max[c] = highest instruction index that SIGNALS counter c. */
        uint32_t n_ctr = g->h.n_counter;
        int32_t* prod_max = (int32_t*)malloc(sizeof(int32_t) * n_ctr);
        for (uint32_t c = 0; c < n_ctr; c++) prod_max[c] = -1;
        /* Gates live on the stream entries (64-byte PlowDevInst carries none). */
        for (uint32_t k = 0; k < NS; k++)
            for (uint32_t s = 0; s < g->stream[k].succ_len; s++) {
                const uint32_t c = g->succs[g->stream[k].succ_ofs + s];
                const int32_t i = (int32_t)g->stream[k].inst;
                if (i > prod_max[c]) prod_max[c] = i;
            }
        /* Every counter an instruction WAITS on must have all producers strictly before it. */
        for (uint32_t k = 0; k < NS; k++)
            for (uint32_t w = 0; w < g->stream[k].wait_len; w++) {
                const uint32_t c = g->waits[g->stream[k].wait_ofs + w].id;
                const uint32_t i = g->stream[k].inst;
                if (prod_max[c] >= (int32_t)i) {
                    printf("FATAL: instruction order is NOT topological — inst %u waits on "
                           "counter %u whose latest producer is inst %d. The global queue "
                           "would deadlock; refusing to launch.\n", i, c, prod_max[c]);
                    return 1;
                }
            }
        free(prod_max);
        printf("GQ: instruction order verified TOPOLOGICAL over %u counters\n", n_ctr);

        /* Counting sort by inst — a stable op-major permutation, O(NS). */
        uint32_t* cnt = (uint32_t*)calloc(NI + 1, 4);
        for (uint32_t j = 0; j < NS; j++) cnt[g->stream[j].inst + 1]++;
        for (uint32_t i = 0; i < NI; i++) cnt[i + 1] += cnt[i];
        PlowStreamEnt* gqs = (PlowStreamEnt*)malloc((size_t)NS * sizeof(PlowStreamEnt));
        uint32_t* put = (uint32_t*)malloc((NI + 1) * 4);
        memcpy(put, cnt, (NI + 1) * 4);
        /* Walk the flat stream in per-block order so ties keep static's slice ordering. */
        for (uint32_t cu = 0; cu < B.h.n_cu; cu++)
            for (uint32_t k = 0; k < g->stream_len[cu]; k++) {
                const PlowStreamEnt e = g->stream[g->stream_ofs[cu] + k];
                gqs[put[e.inst]++] = e;
            }
        for (uint32_t i = 0; i < NI; i++)
            if (put[i] != cnt[i + 1]) { printf("FATAL: gq permutation lost entries\n"); return 1; }

        const uint32_t segofs[2] = {0u, NS};
        CK(cudaMalloc(&g->d_gqstream, (size_t)NS * sizeof(PlowStreamEnt)));
        CK(cudaMalloc(&g->d_gqsegofs, 8));
        CK(cudaMalloc(&g->d_gqcursor, PLOW_CTR_STRIDE * 4)); /* own cache line */
        CK(cudaMemcpy(g->d_gqstream, gqs, (size_t)NS * sizeof(PlowStreamEnt),
                      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g->d_gqsegofs, segofs, 8, cudaMemcpyHostToDevice));
        free(gqs); free(cnt); free(put);
        printf("GQ: op-major stream built, %u entries over %u instructions\n", NS, NI);
    }

    const uint32_t VOCAB = (uint32_t)(B.tensors[t_logits].bytes / 2);
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
    memset(&pr, 0, sizeof(pr));
    pr.insts = (const PlowDevInst*)g->d_inst;
    pr.stream = (const PlowStreamEnt*)g->d_stream;
    pr.stream_ofs = (const uint32_t*)g->d_sofs;
    pr.stream_len = (const uint32_t*)g->d_slen;
    pr.waits = (const PlowWait*)g->d_waits;
    pr.succs = (const uint32_t*)g->d_succs;
    pr.counters = (uint32_t*)g->d_ctr;
    pr.tensors = (void* const*)d_tens;
    pr.trace = NULL;
    pr.gq_stream = (const PlowStreamEnt*)g->d_gqstream;
    pr.gq_seg_ofs = (const uint32_t*)g->d_gqsegofs;
    pr.gq_cursor = (uint32_t*)g->d_gqcursor;
    printf("scheduler: %s%s\n", plow_sm120_sched() ? "GLOBAL QUEUE (one atomic cursor)"
                                                   : "STATIC per-block stream",
           plow_sm120_skeleton() ? "   *** SKELETON BUILD: NO OP BODIES, OUTPUT IS GARBAGE ***"
                                 : "");

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
    auto run_step = [&](int pos, int feed) -> int {
        const double b0 = brk_on ? now() : 0;
        uint32_t lo = g->h.n_inst - 1, hi = 0;
        for (uint32_t k = 0; k < B.h.n_kvrow; k++) {
            const uint32_t ix = B.kvrow[k];
            h_inst[ix].i[3] = nc_kvrow ? (uint32_t)k : (uint32_t)pos;
            if (ix < lo) lo = ix;
            if (ix > hi) hi = ix;
        }
        if (B.h.n_kvrow)
            CK(cudaMemcpy((uint8_t*)g->d_inst + (size_t)lo * sizeof(PlowDevInst),
                          &h_inst[lo], (size_t)(hi - lo + 1) * sizeof(PlowDevInst),
                          cudaMemcpyHostToDevice));
        if (feed >= 0) {
            h_scalar[0] = (int32_t)feed;
            CK(cudaMemcpy(devp[t_ids], h_scalar, 4, cudaMemcpyHostToDevice));
        }
        h_scalar[0] = (int32_t)pos;
        CK(cudaMemcpy(devp[t_pos], h_scalar, 4, cudaMemcpyHostToDevice));
        h_scalar[0] = (int32_t)(pos + 1);
        CK(cudaMemcpy(devp[t_kvlen], h_scalar, 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g->d_ctr, zc, zc_bytes, cudaMemcpyHostToDevice));
        /* The GQ cursor has the same lifecycle as the counters: one launch consumes it
         * from 0 to n_stream, so it must be re-zeroed before every launch. */
        if (g->d_gqcursor) CK(cudaMemset(g->d_gqcursor, 0, 4));
        /* The prologue is all synchronous cudaMemcpy on the default stream, so it has
         * already retired on the device by the time the last one returns; timing the split
         * here does not perturb the launch that follows. */
        const double b1 = brk_on ? now() : 0;
        const int rc = plow_sm120_launch(&pr, grid, 0);
        if (rc != cudaSuccess) {
            printf("LAUNCH FAILED at pos %d: %s\n", pos,
                   cudaGetErrorString((cudaError_t)rc));
            return 1;
        }
        CK(cudaDeviceSynchronize());
        if (brk_on) { host_pro += b1 - b0; dev_ker += now() - b1; }
        return 0;
    };

    /* ---- consume the prompt, one token per decode step (see the header note) ---- */
    const double p0 = now();
    for (int i = 0; i < n_prompt; i++)
        if (run_step(i, prompt[i])) return 1;
    const double pdt = now() - p0;
    printf("prompt: %d tokens through the DECODE program in %.0f ms (%.1f tok/s)\n",
           n_prompt, pdt * 1e3, n_prompt / pdt);

    /* The token was sampled ON DEVICE (ARGMAX + ARGMAX_FIN) and already sits in
     * in.ids, which is exactly where the next step's EMBED reads it. */
    CK(cudaMemcpy(h_scalar, devp[t_ids], 4, cudaMemcpyDeviceToHost));
    int best = (int)h_scalar[0];

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
        CK(cudaMemcpy(h_scalar, devp[t_ids], 4, cudaMemcpyDeviceToHost));
        best = (int)h_scalar[0];
    }
    printf("\n");
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
    }
    free(dt);
    return 0;
}
