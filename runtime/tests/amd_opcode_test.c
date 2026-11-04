/* Build from the repo root inside nix develop:
 * cc -O2 -std=gnu11 -Wall -Wextra -Werror runtime/tests/amd_opcode_test.c \
 *   runtime/amd/hsa_backend.c -I"$ROCM_PATH/include" -L"$ROCM_PATH/lib" \
 *   -Wl,-rpath,"$ROCM_PATH/lib" -lhsa-runtime64 -lm -o /tmp/amd_opcode_test
 * Run beside test_kernels.elf, through gpuq:
 *   amd_opcode_test interp_prefill.elf
 *   amd_opcode_test interp_tokbatch.elf token-batch
 * The second run uses the build script's four-wave interp_tokbatch object
 * (PLOW_TOKEN_BATCH=1, PLOW_TOKEN_BATCH_DESC=1, PLOW_MIXED_STEP=1). It checks live
 * rows below capacity with untouched padding for PER_LAYER_INPUT and embeddings.
 * test_kernels.elf and this host must have matching PLOW_WG_WAVES (default 8).
 * The current interp_tokbatch build is blocked by op_norm.h's unrelated
 * d_rmsnorm_q128 static_assert(rn_groups == 1); the live-row cases await that fix.
 * GEMV_ARGMAX has fixed sample rows; its packet case checks i4 and operand slots.
 * Trap cases are intentionally omitted: this HSA harness has no recoverable
 * device-trap isolation, so a trap may hang the queue/device.
 * Ragged-K GEMV is compared to zero-padded, eight-aligned ordinary GEMV.
 * `dense` selects only the K64 GEMM geometry cases for packet-specific objects;
 * the full opcode suite needs an unconfigured object with every tested arm.
 * `residual-f32` needs typed interpreter objects and test wrappers compiled with
 * PLOW_PACKET_REQUIRES_RESIDUAL_F32=1. It checks four ops against independent CPU
 * math at rows1/134 and hidden960/3840/4096, including aliasing and optional gamma.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t bf16;
static unsigned rng = 12345;
static int fails;
static int token_batch;
static plow_hsa_kernel interp;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float roundbf(float f) { return bf2f(f2bf(f)); }
static float random_value(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return ((int)(rng % 257) - 128) / 256.0f;
}
static void* alloc(plow_hsa* h, size_t bytes) {
    void* p = plow_hsa_alloc_host(h, bytes);
    if (!p) { fprintf(stderr, "%s\n", plow_hsa_last_error()); exit(1); }
    memset(p, 0, bytes);
    return p;
}
static bf16* random_bf(plow_hsa* h, size_t n) {
    bf16* p = alloc(h, n * sizeof(*p));
    for (size_t i = 0; i < n; i++) p[i] = f2bf(random_value());
    return p;
}
static void launch(plow_hsa* h, const char* name, unsigned blocks, void* args, size_t size) {
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, name, &k) ||
        plow_hsa_launch(h, 0, &k, blocks * PLOW_WG_THREADS, 1, 1,
                        PLOW_WG_THREADS, 1, 1, 0, args, size) || plow_hsa_wait(h, 0)) {
        fprintf(stderr, "%s: %s\n", name, plow_hsa_last_error()); exit(1);
    }
}
static void check(const char* name, const bf16* got, const bf16* ref, size_t n, float tol) {
    for (size_t i = 0; i < n; i++) {
        float a = bf2f(got[i]), b = bf2f(ref[i]);
        if (!isfinite(a) || fabsf(a - b) > tol * (1.0f + fabsf(b))) {
            fprintf(stderr, "%s[%zu]: %.8g vs %.8g\n", name, i, a, b);
            fails++; return;
        }
    }
}
static uint64_t pack(bf16 b, unsigned i) {
    unsigned u = (b & 0x8000u) ? (unsigned)(bf16)~b : (unsigned)(b | 0x8000u);
    return ((uint64_t)u << 32) | (uint32_t)~i;
}

static void exact(const char* name, const void* got, const void* ref, size_t bytes) {
    if (memcmp(got, ref, bytes)) { fprintf(stderr, "%s: byte mismatch\n", name); fails++; }
}

static PlowDevInst instruction(unsigned op, void* tensors[8]) {
    PlowDevInst in = {0};
    in.op = op;
    for (unsigned i = 0; i < 8; i++) in.t[i] = tensors[i] ? i : PLOW_TENSOR_NONE;
    return in;
}

static void packet(plow_hsa* h, PlowDevInst in, void* tensors[8], unsigned blocks,
                   unsigned live_rows) {
    PlowDevInst* inst = alloc(h, sizeof(*inst));
    PlowStreamEnt* stream = alloc(h, blocks * sizeof(*stream));
    uint32_t *ofs = alloc(h, blocks * 4), *len = alloc(h, blocks * 4);
    void** table = alloc(h, 8 * sizeof(*table));
    PlowTokenBatch* tb = alloc(h, sizeof(*tb));
    PlowPrefillSpan* span = alloc(h, sizeof(*span));
    uint32_t* gq_cursor=alloc(h,128);
    uint32_t* gq_bounds=alloc(h,8);
    gq_bounds[1]=blocks;
    *inst = in;
    inst->blocks = blocks;
    /* Reverse handles so dispatch must resolve the operand table, not slot indices. */
    for (unsigned i = 0; i < 8; i++) {
        table[7 - i] = tensors[i];
        if (inst->t[i] != PLOW_TENSOR_NONE) inst->t[i] = 7 - i;
    }
    for (unsigned b = 0; b < blocks; b++) {
        ofs[b] = b; len[b] = 1; stream[b].slice = b;
    }
    tb->version = PLOW_TOKEN_BATCH_VERSION;
    tb->row_capacity = in.i[0]; tb->real_rows = live_rows;
    tb->n_spans = 1; tb->spans = span; span->n_rows = live_rows;
    span->kv_len = live_rows; span->flags = PLOW_PREFILL_SPAN_RESET_STATE;
    PlowProgram prog = {0};
    prog.insts = inst; prog.stream = stream; prog.stream_ofs = ofs; prog.stream_len = len;
    prog.tensors = table; prog.token_batch = token_batch ? tb : NULL;
    prog.gq_stream=stream;prog.gq_seg_ofs=gq_bounds;prog.gq_cursor=gq_cursor;
    const unsigned threads = token_batch ? 256u : PLOW_WG_THREADS;
    if (plow_hsa_launch(h, 0, &interp, blocks * threads, 1, 1,
                        threads, 1, 1, 0, &prog, sizeof(prog)) || plow_hsa_wait(h, 0)) {
        fprintf(stderr, "packet op %u: %s\n", in.op, plow_hsa_last_error()); exit(1);
    }
    plow_hsa_free(h, inst); plow_hsa_free(h, stream); plow_hsa_free(h, ofs);
    plow_hsa_free(h, len); plow_hsa_free(h, table); plow_hsa_free(h, tb); plow_hsa_free(h, span);
    plow_hsa_free(h,gq_cursor);plow_hsa_free(h,gq_bounds);
}

static bf16* partial_input(plow_hsa* h,unsigned rows,unsigned k,int structured,int weight) {
    if(!structured)return random_bf(h,(size_t)rows*k);
    bf16* a=alloc(h,(size_t)rows*k*2);
    for(unsigned r=0;r<rows;r++)for(unsigned j=0;j<k;j++)
        a[(size_t)r*k+j]=f2bf(weight?((int)(r%17)-8-(int)(j%8))*0x1p-7f:((int)(r%13)-6+(int)(j%8))*0x1p-6f);
    return a;
}
static void partial_f32(plow_hsa* h, unsigned rows, unsigned n, unsigned k,int structured) {
    bf16* a = partial_input(h,rows,k,structured,0);
    bf16* w = partial_input(h,n,k,structured,1);
    float* out = alloc(h, ((size_t)rows*n+16)*4);
    float* expected = alloc(h, (size_t)rows*n*4);
    for (unsigned m=0; m<rows; m++) for (unsigned c=0; c<n; c++) {
        double sum=0;
        for (unsigned j=0; j<(structured?8:k); j++) sum += (double)bf2f(a[(size_t)m*k+j])*bf2f(w[(size_t)c*k+j]);
        if(structured)sum*=k/8;
        expected[(size_t)m*n+c]=(float)sum;
    }
    for (int direct=0; direct<2; direct++) {
        for (size_t i=0; i<(size_t)rows*n+16; i++) out[i]=12345.0f;
        if (direct) {
            struct {float* out; const bf16* a; const bf16* w; unsigned m,n,k;} args={out,a,w,rows,n,k};
            launch(h,"gemm_f32_parts_test",64,&args,sizeof(args));
        } else {
            void* tensors[8]={out,a,w,NULL,NULL,NULL,NULL,NULL};
            PlowDevInst in=instruction(PLOW_DOP_GEMM_F32,tensors);
            in.i[0]=rows; in.i[1]=n; in.i[2]=k;
            packet(h,in,tensors,64,rows);
        }
        double err=0,den=0;
        for (size_t i=0;i<(size_t)rows*n;i++) {
            double d=(double)out[i]-expected[i]; err+=d*d;den+=(double)expected[i]*expected[i];
        }
        double rel=sqrt(err/fmax(den,1e-30));
        if (!isfinite(rel)||rel>1e-5) fails++;
        for(size_t i=(size_t)rows*n;i<(size_t)rows*n+16;i++) if(out[i]!=12345.0f)fails++;
        printf("partial-f32 rows=%u n=%u k=%u direct=%d rel=%g failures=%d\n",rows,n,k,direct,rel,fails);
    }
    bf16* a2=partial_input(h,rows,k,structured,0);
    bf16* w2=partial_input(h,n,k,structured,1);
    float* part2=alloc(h,((size_t)rows*n+16)*4);
    bf16* reduced=alloc(h,((size_t)rows*n+16)*2);
    struct {float* out;const bf16* a;const bf16* w;unsigned m,n,k;} args={part2,a2,w2,rows,n,k};
    launch(h,"gemm_f32_parts_test",64,&args,sizeof(args));
    for(size_t i=0;i<(size_t)rows*n+16;i++)reduced[i]=0x42f6u;
    struct {bf16* out;const float* a;const float* b;unsigned n;} ar={reduced,out,part2,rows*n};
    launch(h,"reduce_f32_parts_test",8,&ar,sizeof(ar));
    double err=0,den=0,rne_err=0,rne_den=0;
    for(unsigned m=0;m<rows;m++)for(unsigned c=0;c<n;c++){
        double sum=0;
        for(unsigned j=0;j<(structured?8:k);j++)sum+=(double)bf2f(a[(size_t)m*k+j])*bf2f(w[(size_t)c*k+j])+(double)bf2f(a2[(size_t)m*k+j])*bf2f(w2[(size_t)c*k+j]);
        if(structured)sum*=k/8;
        float got=bf2f(reduced[(size_t)m*n+c]),rne=roundbf((float)sum);
        double d=got-sum,rd=(double)got-rne;err+=d*d;den+=sum*sum;rne_err+=rd*rd;rne_den+=(double)rne*rne;
    }
    double rel=sqrt(err/fmax(den,1e-30)),rne=sqrt(rne_err/fmax(rne_den,1e-30));
    if(!isfinite(rel)||!isfinite(rne)||rel>0x1p-8||rne>1e-4)fails++;
    for(size_t i=(size_t)rows*n;i<(size_t)rows*n+16;i++)if(reduced[i]!=0x42f6u)fails++;
    printf("projection-collective rows=%u n=%u k=%u rel=%g rne=%g failures=%d\n",rows,n,k,rel,rne,fails);
    plow_hsa_free(h,a2);plow_hsa_free(h,w2);plow_hsa_free(h,part2);plow_hsa_free(h,reduced);
    plow_hsa_free(h,a);plow_hsa_free(h,w);plow_hsa_free(h,out);plow_hsa_free(h,expected);
}

static void residual_f32(plow_hsa* h, unsigned rows, unsigned feat, int weighted,
                         int alias, float scale) {
    const size_t n = (size_t)rows * feat;
    const float eps = 1e-6f;
    float* a = alloc(h, (n + 16) * 4);
    float* original = alloc(h, n * 4);
    float* residual = alias ? a : alloc(h, (n + 16) * 4);
    float* expected = alloc(h, n * 4);
    bf16 *b = random_bf(h, n), *gamma = random_bf(h, feat);
    bf16* out = alloc(h, (n + 16) * 2);
    float* norm = alloc(h, n * 4);
    int* ids = alloc(h, rows * 4);
    for (size_t i = 0; i < n; i++) original[i] = random_value() + 0.000173f;
    for (unsigned i = 0; i < feat; i++) gamma[i] = f2bf(1.0f + bf2f(gamma[i]));
    for (unsigned r = 0; r < rows; r++) ids[r] = (int)(rows - 1 - r);
    const bf16* g = weighted ? gamma : NULL;
    for (unsigned op = 0; op < 4; op++) {
        for (unsigned row = 0; row < rows; row++) {
            const size_t base = (size_t)row * feat;
            double sum = 0;
            for (unsigned i = 0; i < feat; i++) {
                const double v = op == 1 ? original[base + i] : bf2f(b[base + i]);
                sum += v * v;
            }
            const float inv = (float)(1.0 / sqrt(sum / feat + eps));
            double ss = 0;
            for (unsigned i = 0; i < feat; i++) {
                const size_t j = base + i;
                const float weight = g ? bf2f(g[i]) : 1.0f;
                if (op == 0) expected[j] = bf2f(b[(size_t)ids[row] * feat + i]) * scale;
                else if (op == 1) norm[j] = original[j] * inv * weight;
                else expected[j] = (original[j] + bf2f(b[j]) * inv * weight) * scale;
                if (op == 3) ss += (double)expected[j] * expected[j];
            }
            if (op == 3) {
                const float rinv = (float)(1.0 / sqrt(ss / feat + eps));
                for (unsigned i = 0; i < feat; i++)
                    norm[base + i] = expected[base + i] * rinv * (g ? bf2f(g[i]) : 1.0f);
            }
        }
        for (unsigned direct = 0; direct < 2; direct++) {
            memcpy(a, original, n * 4);
            if (!alias) for (size_t i = 0; i < n + 16; i++) residual[i] = NAN;
            for (size_t i = n; i < n + 16; i++) residual[i] = 912.25f;
            for (size_t i = 0; i < n + 16; i++) out[i] = f2bf(i < n ? NAN : 912.0f);
            void* output = op == 0 || op == 2 ? (void*)residual : (void*)out;
            if (direct) {
                struct { unsigned op; void* out; float* resid; const float* a;
                    const bf16 *b, *gb, *gn; const int* ids; unsigned rows, feat;
                    float eps, scale; } args = {op, output, residual, a, b, g, g, ids,
                                               rows, feat, eps, scale};
                launch(h, "residual_f32_test", 7, &args, sizeof(args));
            } else {
                void* tensors[8] = {0};
                const unsigned codes[] = {PLOW_DOP_EMBED, PLOW_DOP_RMSNORM,
                    PLOW_DOP_NORM_RESIDUAL, PLOW_DOP_NORM_RESIDUAL_NORM};
                tensors[0] = output;
                if (op == 0) { tensors[1] = b; tensors[2] = ids; }
                if (op == 1) { tensors[1] = a; tensors[2] = (void*)g; }
                if (op == 2) { tensors[1] = a; tensors[2] = b; tensors[3] = (void*)g; }
                if (op == 3) { tensors[1] = residual; tensors[2] = a; tensors[3] = b;
                    tensors[4] = (void*)g; tensors[5] = (void*)g; }
                PlowDevInst in = instruction(codes[op], tensors);
                in.i[0] = rows; in.i[1] = feat; in.i[6] = PLOW_RESIDUAL_F32_FLAG;
                in.fj[0].f = op == 0 ? scale : eps; in.fj[1].f = scale;
                packet(h, in, tensors, 7, rows);
            }
            double diff = 0, denominator = 0, rounding_diff = 0, rounding_den = 0;
            for (size_t i = 0; i < n; i++) {
                if (op != 1) {
                    if (!isfinite(residual[i])) { fails++; break; }
                    const double d = residual[i] - expected[i];
                    diff += d * d; denominator += (double)expected[i] * expected[i];
                    if (op == 0 && residual[i] != expected[i]) { fails++; break; }
                }
                if (op == 1 || op == 3) {
                    const float value = bf2f(out[i]), rounded = roundbf(norm[i]);
                    if (!isfinite(value)) { fails++; break; }
                    const double d = value - rounded;
                    rounding_diff += d * d; rounding_den += (double)rounded * rounded;
                    if (op == 1) { const double e = value - norm[i];
                        diff += e * e; denominator += (double)norm[i] * norm[i]; }
                }
            }
            const double rel = sqrt(diff / fmax(denominator, 1e-30));
            const double rne = sqrt(rounding_diff / fmax(rounding_den, 1e-30));
            if (rel > (op == 1 ? 1.0 / 256 : 1e-5) || rne > 1e-4) {
                fprintf(stderr, "residual-f32 op=%u direct=%u rows=%u feat=%u weighted=%d alias=%d scale=%g rel=%g rne=%g\n",
                    op, direct, rows, feat, weighted, alias, scale, rel, rne); fails++;
            }
            for (size_t i = n; i < n + 16; i++)
                if (residual[i] != 912.25f || out[i] != f2bf(912.0f)) { fails++; break; }
        }
    }
    printf("residual-f32 rows=%u feat=%u gamma=%d alias=%d scale=%g failures=%d\n",
           rows, feat, weighted, alias, scale, fails);
    plow_hsa_free(h, a); plow_hsa_free(h, original);
    if (!alias) plow_hsa_free(h, residual);
    plow_hsa_free(h, expected); plow_hsa_free(h, b); plow_hsa_free(h, gamma);
    plow_hsa_free(h, out); plow_hsa_free(h, norm); plow_hsa_free(h, ids);
}

static void finish_argmax(plow_hsa* h, int* ids, uint64_t* part, unsigned blocks, unsigned M) {
    struct { int* ids; uint64_t* part; unsigned blocks, M; } args = {ids, part, blocks, M};
    launch(h, "argmax_fin_test", 1, &args, sizeof(args));
}

static void dense_gemm(plow_hsa* h, unsigned M, unsigned N, unsigned K) {
    bf16* x = alloc(h, (size_t)M * K * 2);
    bf16* w = alloc(h, (size_t)N * K * 2);
    bf16* out = alloc(h, (size_t)M * N * 2);
    bf16* ref = alloc(h, (size_t)M * N * 2);
    /* Small dyadic products and sums are exact in FP32, including cancellation. */
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = f2bf(roundf(random_value() * 16) / 16);
    for (size_t i = 0; i < (size_t)N * K; i++) w[i] = f2bf(roundf(random_value() * 16) / 16);
    for (unsigned m = 0; m < M; m++) for (unsigned n = 0; n < N; n++) {
        double sum = 0;
        for (unsigned k = 0; k < K; k++) sum += bf2f(x[(size_t)m * K + k]) * bf2f(w[(size_t)n * K + k]);
        ref[(size_t)m * N + n] = f2bf((float)sum);
        out[(size_t)m * N + n] = f2bf(NAN);
    }
    void* tensors[8] = {out, x, w};
    PlowDevInst in = instruction(PLOW_DOP_GEMM_SMALL, tensors);
    in.i[0] = M; in.i[1] = N; in.i[2] = K;
    struct { bf16 *out, *x, *w; unsigned M, N, K; } args = {out, x, w, M, N, K};
    printf("direct GEMM M=%u N=%u K=%u\n", M, N, K);
    launch(h, "gemm_c4", 64, &args, sizeof(args));
    check("direct dense GEMM", out, ref, (size_t)M * N, 0);
    for (size_t i = 0; i < (size_t)M * N; i++) out[i] = f2bf(NAN);
    printf("dense GEMM M=%u N=%u K=%u\n", M, N, K);
    packet(h, in, tensors, 64, M);
    check("packet dense GEMM", out, ref, (size_t)M * N, 0);
    plow_hsa_free(h, x); plow_hsa_free(h, w); plow_hsa_free(h, out); plow_hsa_free(h, ref);
}

/* pattern 0: random/tied maxima; 1: rounding boundary; 2: negative last winner;
 * 3: nonzero interior winner; 4: last winner; 5/6: at/below the rounding tie. */
static void gemv(plow_hsa* h, unsigned M, unsigned N, unsigned K, unsigned blocks,
                 float cap, unsigned pattern) {
    bf16* x = random_bf(h, (M + 1) * K);
    bf16* w = random_bf(h, N * K);
    bf16* out = random_bf(h, M * N);
    bf16* raw = alloc(h, M * N * 2);
    bf16* ref = alloc(h, M * N * 2);
    uint64_t *part = alloc(h, M * blocks * 8), *refpart = alloc(h, M * blocks * 8);
    int *ids = alloc(h, M * 4), *refids = alloc(h, M * 4);
    unsigned winner = pattern >= 5 ? 0u : pattern == 1 ? 1u : pattern == 3 ? N / 2 : pattern ? N - 1 : 0;
    if (!pattern) {
        for (unsigned k = 0; k < K; k++) w[k] = w[(N - 1) * K + k] = f2bf(1.0f);
        for (unsigned m = 1; m <= M; m++)
            for (unsigned k = 0; k < K; k++) x[m * K + k] = f2bf(fabsf(bf2f(x[m * K + k])));
    } else {
        memset(w, 0, N * K * 2);
        for (unsigned m = 1; m <= M; m++)
            for (unsigned k = 0; k < K; k++) x[m * K + k] = f2bf(1.0f);
        if (pattern == 1 || pattern >= 5) {
            w[0] = w[K] = f2bf(1.0f); w[K + 1] = f2bf(0x1p-8f);
            const float tail = pattern == 1 ? 0x1p-25f : pattern == 5 ? 0.0f : -0x1p-25f;
            for (unsigned k = 512; k < 520; k++) w[K + k] = f2bf(tail);
        } else {
            for (unsigned n = 0; n < N; n++) w[n * K] = f2bf(-2.0f);
            w[winner * K] = f2bf(pattern == 2 ? -1.0f : 1.0f);
        }
    }
    const unsigned refK = (K + 7u) & ~7u;
    bf16 *refx = alloc(h, M * refK * 2), *refw = alloc(h, N * refK * 2);
    for (unsigned m = 0; m < M; m++) memcpy(refx + m * refK, x + (m + 1) * K, K * 2);
    for (unsigned n = 0; n < N; n++) memcpy(refw + n * refK, w + n * K, K * 2);
    struct { bf16 *out, *x, *w; unsigned M, N, K; } ga = {raw, refx, refw, M, N, refK};
    launch(h, "gemv_unfused_test", blocks, &ga, sizeof(ga));
    if (cap > 0) {
        struct { bf16 *out, *x; unsigned n; float cap; } sa = {ref, raw, M * N, cap};
        launch(h, "gemma_logit_softcap_bf16", blocks, &sa, sizeof(sa));
    } else memcpy(ref, raw, M * N * 2);
    struct { uint64_t* part; bf16* x; unsigned N, M; } aa = {refpart, ref, N, M};
    launch(h, "argmax_test", blocks, &aa, sizeof(aa));
    finish_argmax(h, refids, refpart, blocks, M);
    for (unsigned m = 0; m < M; m++) {
        if ((unsigned)refids[m] != winner) {
            fprintf(stderr, "unfused pattern %u cap %g: token %d vs %u\n", pattern, cap, refids[m], winner);
            fails++;
        }
    }
    struct { bf16 *out, *x, *w; uint64_t* part; unsigned M, N, K; float cap; }
        args = {out, x + K, w, part, M, N, K, cap};
    launch(h, "gemv_argmax_test", blocks, &args, sizeof(args));
    exact("gemv logits", out, ref, M * N * 2);
    finish_argmax(h, ids, part, blocks, M);
    exact("gemv token", ids, refids, M * 4);
    for (unsigned m = 0; m < M; m++) for (unsigned b = 0; b < blocks; b++) {
        unsigned per = (N + blocks - 1) / blocks;
        uint64_t best = 0;
        for (unsigned n = b * per; n < N && n < (b + 1) * per; n++) {
            uint64_t key = pack(ref[m * N + n], n);
            if (key > best) best = key;
        }
        exact("argmax partial", part + m * blocks + b, &best, sizeof(best));
    }
    memset(out, 0xa5, M * N * 2); memset(part, 0xa5, M * blocks * 8);
    void* tensors[8] = {out, x, w, part};
    PlowDevInst in = instruction(PLOW_DOP_GEMV_ARGMAX, tensors);
    in.i[0] = M; in.i[1] = N; in.i[2] = K; in.i[4] = 1; in.fj[0].f = cap;
    packet(h, in, tensors, blocks, M);
    exact("packet gemv logits", out, ref, M * N * 2);
    finish_argmax(h, ids, part, blocks, M);
    exact("packet gemv token", ids, refids, M * 4);
    plow_hsa_free(h, x); plow_hsa_free(h, w); plow_hsa_free(h, out); plow_hsa_free(h, raw);
    plow_hsa_free(h, ref); plow_hsa_free(h, part); plow_hsa_free(h, refpart);
    plow_hsa_free(h, ids); plow_hsa_free(h, refids);
    plow_hsa_free(h, refx); plow_hsa_free(h, refw);
}

static void per_layer(plow_hsa* h, unsigned rows, unsigned H, unsigned P, int next) {
    const unsigned col0 = 7, stride = P + 19, capacity = rows + 2;
    const float eps = 1e-6f, scalar = 0.75f;
    bf16 *x = random_bf(h, capacity * H), *wg = random_bf(h, P * H), *wp = random_bf(h, H * P);
    bf16 *gamma = random_bf(h, H), *ple = random_bf(h, capacity * stride), *gnext = random_bf(h, H);
    bf16* hn = random_bf(h, capacity * H);
    bf16* initial = malloc(capacity * H * 2);
    memcpy(initial, x, capacity * H * 2);
    bf16* ref = malloc(rows * H * 2), *nr = malloc(rows * H * 2);
    float* a = malloc(P * sizeof(float)), *y = malloc(H * sizeof(float));
    memcpy(ref, x, rows * H * 2);
    for (unsigned t = 0; t < rows; t++) {
        for (unsigned p = 0; p < P; p++) {
            double sum = 0;
            for (unsigned j = 0; j < H; j++) sum += (double)bf2f(wg[p * H + j]) * bf2f(ref[t * H + j]);
            float v = roundbf((float)sum);
            float g = roundbf(0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))));
            a[p] = roundbf(g * bf2f(ple[t * stride + col0 + p]));
        }
        double ss = 0;
        for (unsigned j = 0; j < H; j++) {
            double sum = 0;
            for (unsigned p = 0; p < P; p++) sum += (double)bf2f(wp[j * P + p]) * a[p];
            y[j] = roundbf((float)sum); ss += (double)y[j] * y[j];
        }
        float inv = 1.0f / sqrtf((float)ss / H + eps);
        ss = 0;
        for (unsigned j = 0; j < H; j++) {
            float n = roundbf(y[j] * inv * (next == 2 ? 1.0f : bf2f(gamma[j])));
            ref[t * H + j] = f2bf(roundbf(bf2f(ref[t * H + j]) + n) * scalar);
            float v = bf2f(ref[t * H + j]); ss += (double)v * v;
        }
        inv = 1.0f / sqrtf((float)ss / H + eps);
        for (unsigned j = 0; j < H; j++) nr[t * H + j] = f2bf(bf2f(ref[t * H + j]) * inv * (next == 2 ? 1.0f : bf2f(gnext[j])));
    }
    struct { bf16 *x, *wg, *wp, *gamma, *ple, *hn, *gnext; unsigned rows, H, P, col0, stride; float eps, scalar; }
        args = {x, wg, wp, next == 2 ? NULL : gamma, ple, next ? hn : NULL,
                next == 2 ? NULL : gnext, rows, H, P, col0, stride, eps, scalar};
    launch(h, "per_layer_input_test", 3, &args, sizeof(args));
    check("per-layer x", x, ref, rows * H, 0.025f);
    if (next) check("per-layer next", hn, nr, rows * H, 0.025f);
    void* tensors[8] = {x, wg, wp, args.gamma, ple, args.hn, args.gnext};
    PlowDevInst in = instruction(PLOW_DOP_PER_LAYER_INPUT, tensors);
    in.i[0] = token_batch ? capacity : rows; in.i[1] = H; in.i[2] = P;
    in.i[3] = col0; in.i[4] = stride; in.fj[0].f = eps; in.fj[1].f = scalar;
    const unsigned block_counts[] = {1, 3, rows + 2};
    for (unsigned b = 0; b < sizeof(block_counts) / sizeof(block_counts[0]); b++) {
        /* Reuse GPU-written host memory: dispatch must acquire the CPU reset. */
        memcpy(x, initial, capacity * H * 2);
        memset(hn, 0xa5, capacity * H * 2);
        packet(h, in, tensors, block_counts[b], rows);
        check("packet per-layer x", x, ref, rows * H, 0.025f);
        if (next) check("packet per-layer next", hn, nr, rows * H, 0.025f);
        exact("per-layer padding", x + rows * H, initial + rows * H, 2 * H * 2);
    }
    memset(initial, 0xa5, 2 * H * 2);
    exact("per-layer next padding", hn + rows * H, initial, 2 * H * 2);
    free(initial); free(ref); free(nr); free(a); free(y);
    plow_hsa_free(h, x); plow_hsa_free(h, wg); plow_hsa_free(h, wp); plow_hsa_free(h, gamma);
    plow_hsa_free(h, ple); plow_hsa_free(h, gnext); plow_hsa_free(h, hn);
}

static void embeds(plow_hsa* h, unsigned width) {
    const unsigned rows = 7, capacity = 9, vocab = 13, overlays = 3, positions = 9, K = 3, cb_size = 5;
    bf16 *table = random_bf(h, vocab * width), *pt = random_bf(h, positions * width);
    bf16 *cb = random_bf(h, K * cb_size * width), *out = random_bf(h, capacity * width);
    bf16* ref = malloc(capacity * width * 2);
    bf16* initial = malloc(capacity * width * 2);
    memcpy(initial, out, capacity * width * 2);
    float *ov = alloc(h, overlays * width * 4), *nw = alloc(h, width * 4);
    unsigned *tok = alloc(h, capacity * 4), *sel = alloc(h, capacity * 4), *pos = alloc(h, capacity * 4);
    unsigned *base = alloc(h, capacity * 4), *codes = alloc(h, capacity * K * 4);
    for (unsigned i = 0; i < overlays * width; i++) ov[i] = random_value() * 1.003f;
    for (unsigned i = 0; i < width; i++) nw[i] = 1.0f + random_value();
    for (unsigned r = 0; r < rows; r++) {
        tok[r] = r + 1; base[r] = 100 + r * 3; pos[r] = base[r] + r;
        sel[r] = r % 3 == 0 ? UINT32_MAX : (r % 3 == 1 ? UINT32_MAX - 1 : r % overlays);
        for (unsigned k = 0; k < K; k++) codes[r * K + k] = (r + k) % cb_size;
    }
    memcpy(ref, out, capacity * width * 2);
    for (unsigned r = 0; r < rows; r++) for (unsigned i = 0; i < width; i++) {
        if (sel[r] == UINT32_MAX) ref[r * width + i] = table[tok[r] * width + i];
        else if (sel[r] != UINT32_MAX - 1) ref[r * width + i] = f2bf(ov[sel[r] * width + i]);
    }
    struct { bf16 *out, *table; unsigned* tok; float* ov; unsigned* sel; unsigned rows, width, vocab, overlays; }
        oa = {out, table, tok, ov, sel, rows, width, vocab, overlays};
    launch(h, "embed_overlay_bf16_test", 3, &oa, sizeof(oa));
    check("overlay", out, ref, rows * width, 0);
    memcpy(out, initial, capacity * width * 2);
    void* tensors[8] = {out, table, tok, ov, sel};
    PlowDevInst in = instruction(PLOW_DOP_EMBED_OVERLAY_BF16, tensors);
    in.i[0] = token_batch ? capacity : rows; in.i[1] = width;
    in.i[2] = vocab; in.i[3] = overlays;
    packet(h, in, tensors, 3, rows);
    exact("packet overlay", out, ref, capacity * width * 2);
    for (unsigned r = 0; r < rows; r++) for (unsigned i = 0; i < width; i++)
        ref[r * width + i] = f2bf(bf2f(table[tok[r] * width + i]) + bf2f(pt[(pos[r] - base[r]) * width + i]));
    struct { bf16 *out, *table; unsigned* tok; bf16* pt; unsigned *pos, *base; unsigned rows, width, vocab, positions; }
        pa = {out, table, tok, pt, pos, base, rows, width, vocab, positions};
    launch(h, "embed_pos_bf16_test", 3, &pa, sizeof(pa));
    check("position", out, ref, rows * width, 0);
    memcpy(out, initial, capacity * width * 2);
    tensors[3] = pt; tensors[4] = pos; tensors[5] = base;
    in = instruction(PLOW_DOP_EMBED_POS_BF16, tensors);
    in.i[0] = token_batch ? capacity : rows; in.i[1] = width;
    in.i[2] = vocab; in.i[3] = positions;
    packet(h, in, tensors, 3, rows);
    exact("packet position", out, ref, capacity * width * 2);
    for (unsigned mode = 0; mode < 4; mode++) {
        for (unsigned r = 0; r < rows; r++) {
            double ss = 0;
            for (unsigned i = 0; i < width; i++) {
                if (tok[r] - 3u >= 3u) { ref[r * width + i] = table[tok[r] * width + i]; continue; }
                float sum = 0;
                for (unsigned k = 0; k < K; k++) sum += bf2f(cb[(k * cb_size + codes[r * K + k]) * width + i]);
                float v = roundbf(sum);
                if (!(mode & 1)) v = roundbf(bf2f(table[tok[r] * width + i]) + v);
                ref[r * width + i] = f2bf(v / 1.75f);
                v = bf2f(ref[r * width + i]); ss += (double)v * v;
            }
            if ((mode & 2) && tok[r] - 3u < 3u) {
                float inv = 1.0f / sqrtf((float)ss / width + 1e-5f);
                for (unsigned i = 0; i < width; i++) ref[r * width + i] = f2bf(bf2f(ref[r * width + i]) * inv * nw[i]);
            }
        }
        struct { bf16 *out, *table; unsigned* tok; bf16* cb; unsigned* codes; float* nw;
                 unsigned rows, width, vocab, K, cb_size, frame_lo, frame_rows, mode; float div, eps; }
            fa = {out, table, tok, cb, codes, (mode & 2) ? nw : NULL,
                  rows, width, vocab, K, cb_size, 3, 3, mode, 1.75f, 1e-5f};
        launch(h, "embed_frame_bf16_test", 3, &fa, sizeof(fa));
        check("frame", out, ref, rows * width, (mode & 2) ? 0.01f : 0);
        memcpy(out, initial, capacity * width * 2);
        tensors[3] = cb; tensors[4] = codes; tensors[5] = fa.nw;
        in = instruction(PLOW_DOP_EMBED_FRAME_BF16, tensors);
        in.i[0] = token_batch ? capacity : rows; in.i[1] = width;
        in.i[2] = vocab; in.i[3] = K; in.i[4] = cb_size;
        in.i[5] = 3; in.i[6] = 3; in.i[7] = mode;
        in.fj[0].f = 1.75f; in.fj[1].f = 1e-5f;
        packet(h, in, tensors, 3, rows);
        check("packet frame", out, ref, rows * width, (mode & 2) ? 0.01f : 0);
        exact("frame padding", out + rows * width, initial + rows * width, 2 * width * 2);
    }
    free(initial); free(ref); plow_hsa_free(h, table); plow_hsa_free(h, pt); plow_hsa_free(h, cb); plow_hsa_free(h, out);
    plow_hsa_free(h, ov); plow_hsa_free(h, nw); plow_hsa_free(h, tok); plow_hsa_free(h, sel);
    plow_hsa_free(h, pos); plow_hsa_free(h, base); plow_hsa_free(h, codes);
}

static void load_object(plow_hsa* h, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); rewind(f);
    if (bytes <= 0) exit(1);
    void* elf = malloc((size_t)bytes);
    if (!elf || fread(elf, 1, (size_t)bytes, f) != (size_t)bytes) exit(1);
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, elf, (size_t)bytes)) {
        fprintf(stderr, "%s: %s\n", path, plow_hsa_last_error()); exit(1);
    }
    free(elf);
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 3 || (argc == 3 && strcmp(argv[2], "token-batch") && strcmp(argv[2], "dense") && strcmp(argv[2], "residual-f32") && strcmp(argv[2], "partial-f32") && strcmp(argv[2],"partial-prefill-f32"))) {
        fprintf(stderr, "usage: %s [interp_prefill.elf [dense | residual-f32] | interp_decode_gq.elf partial-f32 | interp_tokbatch.elf token-batch]\n", argv[0]);
        return 1;
    }
    const int residual_only = argc == 3 && !strcmp(argv[2], "residual-f32");
    const int partial_prefill=argc==3&&!strcmp(argv[2],"partial-prefill-f32");
    const int partial_only = partial_prefill || (argc == 3 && !strcmp(argv[2], "partial-f32"));
    const int dense_only = argc == 3 && !strcmp(argv[2], "dense");
    token_batch = argc == 3 && !dense_only && !residual_only && !partial_only;
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    load_object(h, argc > 1 ? argv[1] : "interp_prefill.elf");
    char arch[64], symbol[96];
    uint32_t cus, lds_bytes;
    if (plow_hsa_device_info(h, 0, arch, &cus, &lds_bytes)) return 1;
    snprintf(symbol, sizeof(symbol), partial_prefill?"plow_interp_%s_gq":partial_only ? "plow_interp_dec_%s_gq" : token_batch ? "plow_interp_tokbatch_%s" : "plow_interp_%s", arch);
    if (plow_hsa_get_kernel(h, 0, symbol, &interp)) {
        fprintf(stderr, "%s: %s\n", symbol, plow_hsa_last_error()); return 1;
    }
    load_object(h, "test_kernels.elf");
    unsigned* threads = alloc(h, sizeof(*threads));
    struct { unsigned* out; } probe = {threads};
    launch(h, "plow_probe_wg_threads", 1, &probe, sizeof(probe));
    if (*threads != PLOW_WG_THREADS) { fprintf(stderr, "workgroup width mismatch\n"); return 1; }
    for (unsigned i = 0; i < 8; i++) {
        const unsigned value = 0x12340000u + i * 257u;
        *threads = value;
        launch(h, "plow_probe_host_roundtrip", 1, &probe, sizeof(probe));
        if (*threads != value + 1u) {
            fprintf(stderr, "host-memory roundtrip: %u vs %u\n", *threads, value + 1u);
            return 1;
        }
    }
    plow_hsa_free(h, threads);
    if (!partial_only && !residual_only && !token_batch && (dense_only || !strcmp(arch, "gfx1201"))) {
        const unsigned saved_rng = rng;
        dense_gemm(h, 128, 1024, 1024);
        dense_gemm(h, 128, 960, 1024); dense_gemm(h, 128, 1024, 960);
        dense_gemm(h, 128, 960, 960); dense_gemm(h, 128, 320, 960);
        dense_gemm(h, 128, 2560, 960); dense_gemm(h, 128, 960, 2560);
        dense_gemm(h, 1, 960, 960); dense_gemm(h, 1, 320, 960);
        rng = saved_rng;
    }
    if (residual_only) {
        const unsigned widths[] = {960, 3840, 4096};
        for (unsigned w = 0; w < 3; w++) for (unsigned rows = 1; rows <= 134; rows += 133)
            for (int g = 0; g < 2; g++) for (int alias = 0; alias < 2; alias++) {
                residual_f32(h, rows, widths[w], g, alias, 1.0f);
                residual_f32(h, rows, widths[w], g, alias, 0.052978515625f);
            }
    }
    if (partial_only) {
        const unsigned ks[]={2048,4096,7680};
        const unsigned ms[]={1,8,128,512,1024,2048,4096,8192};
        for (unsigned i=partial_prefill?2:0;i<(partial_prefill?8:3);i++)
            for(unsigned j=0;j<3;j++)partial_f32(h,ms[i],3840,ks[j],partial_prefill);
    }
    if (!dense_only && !residual_only && !partial_only) {
        gemv(h, 1, 259, 512, 3, 0, 0); gemv(h, 3, 259, 520, 5, 30, 0);
        gemv(h, 2, 7, 67, 11, 0, 0); gemv(h, 1, 259, 3840, 3, 30, 0);
        gemv(h, 1, 259, 8192, 3, 0, 0);
        for (unsigned pattern = 1; pattern <= 6; pattern++) {
            gemv(h, 2, 19, 1024, 3, 0, pattern);
            gemv(h, 2, 19, 1024, 3, 30, pattern);
        }
        per_layer(h, 1, 1536, 256, 0); per_layer(h, 7, 64, 32, 1);
        per_layer(h, 3, 67, 35, 2); per_layer(h, 1, 2560, 256, 1);
        embeds(h, 64); embeds(h, 67); embeds(h, 1024);
    }
    plow_hsa_shutdown(h);
    printf("amd_opcode_test: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
