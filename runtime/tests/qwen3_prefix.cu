/* qwen3_prefix.cu — prefix caching over plow's HEAD-MAJOR KV pool, measured.
 *
 *   PLOW_PREFIX_SHARED=2048 gpulease run ./qwen3_prefix qwen3_4256.pkt \
 *       /root/models/Qwen3-4B /workspace/p4096.ids 24
 *
 * Derived from qwen3_sm120_chat.cu (same loader, same run_step, same decode program);
 * only the driver below the lambda is new. See the big comment at the driver for the
 * ABI finding this is built on — in short, FlashDecode's kv_mask is a power-of-two RING
 * MODULO, not a page table, so it can address exactly ONE contiguous run per (b,kv_head)
 * and CANNOT express shared-prefix + private-suffix. This harness measures what the
 * current ABI *can* do (prefix restore by strided D2D blit) and gates it on tokens.
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
#include <math.h>

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
        if (!strncmp(td->name, "model.", 6)) {
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


    /* ================= PREFIX CACHE over the HEAD-MAJOR KV pool =================
     *
     * THE ABI FACT THIS EXPERIMENT IS BUILT ON, read off the emitter
     * (crates/plowc/src/bin/gemma4.rs:1165-1168) and the consumer
     * (runtime/nvidia/op_attention.cuh:174) rather than crates/packet/src/dev.rs:
     *
     *   kbase = K + ((b*n_kv_head + hkv) * kv_stride) * D;   krow = kbase + (kv & kv_mask)*D;
     *
     * kv_mask is a POWER-OF-TWO RING MODULO (KV_MASK_NONE=0xFFFFFFFF on a full layer,
     * kvr-1 on a sliding one -- gemma4.rs kv_ring()). It is a sliding-window wrap, NOT a
     * page table. There is NO index array on this path: FlashDecode can address exactly
     * ONE contiguous run per (b, kv_head). It CANNOT express "shared prefix run + private
     * suffix run". the design notes B.2 assumed the kv_mask ring "already supports a
     * discontinuity" and could be verified for two runs -- that assumption is WRONG, and
     * the correct reading is recorded in the report.
     *
     * So zero-copy cross-sequence sharing is not expressible today. What IS expressible,
     * and what this measures, is PREFIX RESTORE: because a full layer has kv_mask=~0, KV
     * row index == absolute token position exactly, so the KV a prefix produces is
     * bit-identical for every sequence carrying that prefix. The prefix is therefore a
     * STRIDED SET OF RUNS -- one per (layer, kv/v, head) -- that can be blitted D2D into a
     * fresh sequence's cache instead of being recomputed by n_shared decode steps.
     *
     * Run count for Qwen3-4B: 36 layers x 2 (K,V) x 8 kv-heads = 576 runs, each
     * P*head_dim*2 bytes contiguous. That is the "strided set of runs" the head-major
     * layout forces, made concrete. */
    const int n_shared = getenv("PLOW_PREFIX_SHARED") ? atoi(getenv("PLOW_PREFIX_SHARED")) : 0;
    if (n_shared <= 0 || n_shared >= n_prompt) {
        printf("set PLOW_PREFIX_SHARED to the shared prefix length (0 < P < %d)\n", n_prompt);
        return 1;
    }

    /* --- KV geometry, derived from the packet's own FlashDecode operands --- */
    uint32_t kvh = 0, kv_stride = 0, hd = 0, kvm = 0;
    int n_fd = 0;
    for (uint32_t i = 0; i < g->h.n_inst; i++) {
        if (g->insts[i].op != PLOW_DOP_FLASH_DECODE) continue;
        const PlowDevInst* d = &g->insts[i];
        if (n_fd == 0) { kvh = d->i[2]; kv_stride = d->i[3]; hd = d->i[6]; kvm = d->i[7]; }
        else if (d->i[2] != kvh || d->i[3] != kv_stride || d->i[6] != hd || d->i[7] != kvm) {
            printf("FATAL: heterogeneous FlashDecode KV geometry across layers; this "
                   "harness assumes one (kvh,kv_stride,hd,kv_mask) for the whole model\n");
            return 1;
        }
        n_fd++;
    }
    if (!n_fd) { printf("FATAL: no FLASH_DECODE instruction in the packet\n"); return 1; }
    printf("\nKV geometry from the packet: n_kv_head=%u kv_stride=%u head_dim=%u "
           "kv_mask=0x%08X  (%d FlashDecode packets)\n", kvh, kv_stride, hd, kvm, n_fd);
    /* The restore is only bit-exact if row == position, i.e. the ring is the identity.
     * On a sliding layer (kvm = kvr-1) rows wrap and a prefix longer than the ring is
     * simply not retained -- that is a real limitation, so it is asserted, not papered over. */
    if (kvm != 0xFFFFFFFFu) {
        printf("FATAL: kv_mask=0x%08X is a RING (sliding window). Prefix restore assumes\n"
               "       row==position (kv_mask=~0). A sliding layer needs the gather arm.\n", kvm);
        return 1;
    }
    if ((uint32_t)n_shared > kv_stride) {
        printf("FATAL: shared prefix %d exceeds kv_stride %u\n", n_shared, kv_stride);
        return 1;
    }

    /* --- the run list: one run per (kv tensor, kv head) --- */
    struct Run { int tensor; size_t dev_off; size_t snap_off; size_t bytes; };
    const size_t elem = 2; /* bf16 cache */
    const size_t head_slot = (size_t)kv_stride * hd * elem;     /* one (kv,head) head-slot */
    const size_t run_bytes = (size_t)n_shared * hd * elem;      /* the prefix inside it */
    int n_kvt = 0;
    for (uint32_t i = 0; i < B.h.n_tensor; i++)
        if (!strncmp(B.tensors[i].name, "kv.", 3)) n_kvt++;
    Run* runs = (Run*)malloc(sizeof(Run) * (size_t)n_kvt * kvh);
    int n_run = 0;
    size_t snap_bytes = 0;
    for (uint32_t i = 0; i < B.h.n_tensor; i++) {
        if (strncmp(B.tensors[i].name, "kv.", 3)) continue;
        if (B.tensors[i].bytes != head_slot * kvh) {
            printf("FATAL: %s is %llu B, expected kvh*kv_stride*hd*2 = %llu B\n",
                   B.tensors[i].name, (unsigned long long)B.tensors[i].bytes,
                   (unsigned long long)(head_slot * kvh));
            return 1;
        }
        for (uint32_t h = 0; h < kvh; h++) {
            runs[n_run].tensor = (int)i;
            runs[n_run].dev_off = (size_t)h * head_slot; /* head-major: head h's rows start here */
            runs[n_run].snap_off = snap_bytes;
            runs[n_run].bytes = run_bytes;
            snap_bytes += run_bytes;
            n_run++;
        }
    }
    printf("prefix P=%d tokens -> %d runs (%d kv tensors x %u heads), %.2f KiB each, "
           "%.2f MiB total\n", n_shared, n_run, n_kvt, kvh, run_bytes / 1024.0,
           snap_bytes / (1024.0 * 1024.0));
    printf("full KV footprint of one sequence: %.2f MiB (kv_stride=%u)\n",
           (double)kvb / (1024.0 * 1024.0), kv_stride);

    void* d_snap = NULL;
    CK(cudaMalloc(&d_snap, snap_bytes));

    /* NEGATIVE CONTROL. A prefix cache that returns the WRONG KV produces fluent wrong
     * text -- the worst failure mode in this stack -- so the gate has to be shown able to
     * fail. PLOW_NEGCTRL_PREFIX=n slides ONE run's restore destination by n KV ROWS
     * (n*head_dim*2 bytes) inside its own head-slot. Nothing is out of bounds, no size
     * changes, refcounts and the run list stay structurally valid: exactly one head of one
     * layer gets its prefix keys off by n positions. That is precisely the class of bug a
     * page-table-over-pool introduces.
     *
     * MEASURED SENSITIVITY -- and this is the headline result of the control, not a
     * footnote. At P=2048/N=4096 the corruption does NOT break the token match:
     *
     *   shift        step-0 logits differing       24-token greedy match
     *   1 row        121027 / 151936  (dmax 0.156)  IDENTICAL  <-- token gate BLIND
     *   64 rows      129196 / 151936  (dmax 0.188)  IDENTICAL  <-- token gate BLIND
     *
     * 80-85% of the vocabulary's logits move and greedy decoding still picks the same 24
     * tokens. So a prefix cache serving WRONG KV would have passed a token-match gate
     * outright -- exactly the "fluent wrong text" failure this cache must never have. The
     * gate that fires is the BIT-EXACT step-0 logits row, which is why the pass condition
     * below is `tok_match && lg_match` and not `tok_match`. A token-only gate on this
     * feature would be a fake PASS. */
    const int nc_prefix = getenv("PLOW_NEGCTRL_PREFIX") ? atoi(getenv("PLOW_NEGCTRL_PREFIX")) : 0;
    const int nc_run = getenv("PLOW_NEGCTRL_RUN") ? atoi(getenv("PLOW_NEGCTRL_RUN")) : 0;
    if (nc_prefix)
        printf("*** NEGATIVE CONTROL: run %d restored %d KV rows off (%zu B) ***\n",
               nc_run, nc_prefix, (size_t)nc_prefix * hd * elem);

    /* CK() returns 1 on failure, so every lambda that uses it is int-returning and its
     * result must be checked -- a swallowed cudaMemset failure here would silently serve
     * a stale cache, which is exactly the bug class this harness exists to catch. */
    auto zero_kv = [&]() -> int {
        for (uint32_t i = 0; i < B.h.n_tensor; i++)
            if (!strncmp(B.tensors[i].name, "kv.", 3))
                CK(cudaMemset(devp[i], 0, B.tensors[i].bytes));
        return 0;
    };
    auto snapshot = [&]() -> int {
        for (int r = 0; r < n_run; r++)
            CK(cudaMemcpy((uint8_t*)d_snap + runs[r].snap_off,
                          (uint8_t*)devp[runs[r].tensor] + runs[r].dev_off, runs[r].bytes,
                          cudaMemcpyDeviceToDevice));
        return 0;
    };
    auto restore = [&]() -> int {
        for (int r = 0; r < n_run; r++) {
            size_t off = runs[r].dev_off;
            if (nc_prefix && r == nc_run) off += (size_t)nc_prefix * hd * elem;
            CK(cudaMemcpy((uint8_t*)devp[runs[r].tensor] + off,
                          (uint8_t*)d_snap + runs[r].snap_off, runs[r].bytes,
                          cudaMemcpyDeviceToDevice));
        }
        return 0;
    };

    const int NG = n_gen;
    int* tok_cold = (int*)malloc(sizeof(int) * (size_t)(NG > 0 ? NG : 1));
    int* tok_warm = (int*)malloc(sizeof(int) * (size_t)(NG > 0 ? NG : 1));
    uint16_t* lg_cold = (uint16_t*)malloc((size_t)VOCAB * 2);
    uint16_t* lg_warm = (uint16_t*)malloc((size_t)VOCAB * 2);

    /* One request: feed prompt[start..n_prompt), then greedily generate NG tokens.
     * `start` > 0 means rows 0..start-1 were supplied by the prefix restore. */
    auto serve = [&](int start, int* out_tok, uint16_t* out_lg, double* ttft) -> int {
        const double t0 = now();
        for (int i = start; i < n_prompt; i++)
            if (run_step(i, prompt[i])) return 1;
        CK(cudaMemcpy(h_scalar, devp[t_ids], 4, cudaMemcpyDeviceToHost));
        *ttft = now() - t0;
        int best = (int)h_scalar[0];
        CK(cudaMemcpy(out_lg, devp[t_logits], (size_t)VOCAB * 2, cudaMemcpyDeviceToHost));
        int ctx = n_prompt;
        for (int s = 0; s < NG; s++) {
            out_tok[s] = best;
            if (ctx + 1 >= max_ctx) { for (int j = s + 1; j < NG; j++) out_tok[j] = -1; break; }
            if (run_step(ctx, -1)) return 1;
            ctx++;
            CK(cudaMemcpy(h_scalar, devp[t_ids], 4, cudaMemcpyDeviceToHost));
            best = (int)h_scalar[0];
        }
        return 0;
    };

    /* ---- pass 0: POPULATE. A previous request whose prompt was the shared prefix. ---- */
    if (zero_kv()) return 1;
    printf("\n[populate] feeding the %d-token shared prefix to fill the cache...\n", n_shared);
    const double pop0 = now();
    for (int i = 0; i < n_shared; i++)
        if (run_step(i, prompt[i])) return 1;
    const double pop_dt = now() - pop0;
    const double snp0 = now();
    if (snapshot()) return 1;
    CK(cudaDeviceSynchronize());
    const double snp_dt = now() - snp0;
    printf("[populate] %d tokens in %.0f ms; snapshot of %d runs (%.2f MiB) in %.3f ms "
           "(%.0f GB/s D2D)\n", n_shared, pop_dt * 1e3, n_run,
           snap_bytes / (1024.0 * 1024.0), snp_dt * 1e3,
           2.0 * snap_bytes / snp_dt / 1e9);

    /* ---- pass 1: COLD. Same full prompt, empty cache. The reference. ---- */
    if (zero_kv()) return 1;
    double ttft_cold = 0;
    printf("\n[cold] serving the %d-token request with an EMPTY cache...\n", n_prompt);
    if (serve(0, tok_cold, lg_cold, &ttft_cold)) return 1;
    printf("[cold] TTFT = %.1f ms  (%d prompt tokens through the decode program)\n",
           ttft_cold * 1e3, n_prompt);

    /* ---- pass 2: WARM. Restore the shared runs, prefill only the private suffix. ---- */
    if (zero_kv()) return 1;
    double ttft_warm = 0;
    printf("\n[warm] restoring %d runs, then serving only the %d-token private suffix...\n",
           n_run, n_prompt - n_shared);
    const double w0 = now();
    if (restore()) return 1;
    CK(cudaDeviceSynchronize());
    const double rst_dt = now() - w0;
    double ttft_suffix = 0;
    if (serve(n_shared, tok_warm, lg_warm, &ttft_suffix)) return 1;
    ttft_warm = rst_dt + ttft_suffix;
    printf("[warm] restore = %.3f ms (%.0f GB/s D2D), suffix prefill = %.1f ms, "
           "TTFT = %.1f ms\n", rst_dt * 1e3, 2.0 * snap_bytes / rst_dt / 1e9,
           ttft_suffix * 1e3, ttft_warm * 1e3);

    /* ---- the gates ---- */
    int tok_match = 1, first_div = -1;
    for (int s = 0; s < NG; s++)
        if (tok_cold[s] != tok_warm[s]) { tok_match = 0; if (first_div < 0) first_div = s; }
    const int lg_match = memcmp(lg_cold, lg_warm, (size_t)VOCAB * 2) == 0;
    int lg_ndiff = 0;
    double lg_maxabs = 0;
    for (uint32_t v = 0; v < VOCAB; v++)
        if (lg_cold[v] != lg_warm[v]) {
            lg_ndiff++;
            const double d = fabs((double)b2f(lg_cold[v]) - (double)b2f(lg_warm[v]));
            if (d > lg_maxabs) lg_maxabs = d;
        }

    printf("\n================================ RESULT ================================\n");
    printf("cold tokens :");
    for (int s = 0; s < NG; s++) printf(" %d", tok_cold[s]);
    printf("\nwarm tokens :");
    for (int s = 0; s < NG; s++) printf(" %d", tok_warm[s]);
    printf("\n");
    printf("token match : %s  (%d/%d)%s\n", tok_match ? "IDENTICAL" : "*** DIVERGED ***",
           NG - (first_div < 0 ? 0 : NG - first_div), NG,
           first_div >= 0 ? "  first divergence at step " : "");
    if (first_div >= 0) printf("              first divergence: step %d\n", first_div);
    printf("step-0 logits: %s  (%d/%u bf16 words differ, max |delta| = %.6g)\n",
           lg_match ? "BIT-IDENTICAL" : "*** DIFFER ***", lg_ndiff, VOCAB, lg_maxabs);
    printf("TTFT cold   : %10.1f ms\n", ttft_cold * 1e3);
    printf("TTFT warm   : %10.1f ms   (restore %.3f ms + %d suffix steps)\n",
           ttft_warm * 1e3, rst_dt * 1e3, n_prompt - n_shared);
    printf("speedup     : %10.1fx   prefill work skipped: %d/%d tokens (%.1f%%)\n",
           ttft_cold / ttft_warm, n_shared, n_prompt,
           100.0 * n_shared / n_prompt);
    printf("KV bytes reused (would be recomputed cold): %.2f MiB across %d runs\n",
           snap_bytes / (1024.0 * 1024.0), n_run);
    printf("PLOW_PREFIX P=%d N=%d ttft_cold_ms=%.3f ttft_warm_ms=%.3f restore_ms=%.4f "
           "speedup=%.3f runs=%d snap_mib=%.3f tok_match=%d lg_bitexact=%d lg_ndiff=%d\n",
           n_shared, n_prompt, ttft_cold * 1e3, ttft_warm * 1e3, rst_dt * 1e3,
           ttft_cold / ttft_warm, n_run, snap_bytes / (1024.0 * 1024.0), tok_match,
           lg_match, lg_ndiff);
    printf("========================================================================\n");

    /* The harness must FAIL when the cache is wrong. Under the negative control the
     * expectation inverts: divergence is the pass. */
    if (nc_prefix) return (tok_match && lg_match) ? 1 : 0;
    return (tok_match && lg_match) ? 0 : 1;
}
