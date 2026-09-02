/* indexer_qscore_gfx950_test.hip — GLM-5.3-Flash's DSA indexer query-side fp8 quant
 * (op 126, DsaQQuant) and fp8 pooled score (op 127, IndexScoreKpool) against
 * /home/shaswot/plow-work/indexer_qscore_oracle.npz — built by actually RUNNING vLLM's
 * real fwht128_quant_fp8 (kpool_compress.py) and fp8_mqa_logits_torch
 * (rocm_aiter_mla_sparse.py) on this box's GPU. See op_dsa_pool.h / op_attention.h's
 * header comments and crates/packet/src/dev.rs's DevOp::DsaQQuant / IndexScoreKpool doc
 * comments for the full contract.
 *
 * Two passes, the op bodies are device-only by include chain:
 *   hipcc --offload-arch=gfx950 -O3 -w -DIQS_DEVICE --genco \
 *       runtime/tests/indexer_qscore_gfx950_test.hip -o /tmp/iqs.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/indexer_qscore_gfx950_test.hip -o /tmp/iqs -Iruntime/tests \
 *       -L/opt/rocm/lib -lamdhip64
 *   perf-data/tools/gpulease -n 1 iqs 'sg render -c "/tmp/iqs /tmp/iqs.co"'
 */

#ifdef IQS_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_gemm.h"
#include "op_norm.h"
#include "op_dsa_pool.h"
#include "op_attention.h"

struct QArgs {
    unsigned long long q_fp8, q_scale, q_raw;
    unsigned n_rows, head_dim;
};
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_qquant(QArgs* a) {
    __shared__ float lds[128];
    __shared__ float part[PLOW_WAVES];
    d_dsa_q_quant((uint8_t*)a->q_fp8, (float*)a->q_scale, (const bf16*)a->q_raw, a->n_rows,
                  a->head_dim, blockIdx.x, gridDim.x, lds, part);
}

struct SArgs {
    unsigned long long score, qfp8, qscale, kfp8, kscale, w, kv_len;
    unsigned n_batch, index_heads, pool_stride, pool_size;
    float scale;
};
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_score(SArgs* a) {
    __shared__ uint8_t qlds[32 * 128];
    d_index_score_kpool<128>((float*)a->score, (const uint8_t*)a->qfp8, (const float*)a->qscale,
                             (const uint8_t*)a->kfp8, (const float*)a->kscale, (const float*)a->w,
                             (const int*)a->kv_len, a->n_batch, a->index_heads, a->pool_stride,
                             a->pool_size, false, a->scale, blockIdx.x, gridDim.x, qlds);
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "indexer_qscore_oracle_fixture.h"

typedef unsigned short bf16;
static bf16 f2b(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
#define HC(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return 1; }        \
    } while (0)

struct QArgs {
    unsigned long long q_fp8, q_scale, q_raw;
    unsigned n_rows, head_dim;
};
struct SArgs {
    unsigned long long score, qfp8, qscale, kfp8, kscale, w, kv_len;
    unsigned n_batch, index_heads, pool_stride, pool_size;
    float scale;
};

int main(int argc, char** argv) {
    if (argc != 2) { printf("usage: %s <iqs.co>\n", argv[0]); return 1; }
    HC(hipSetDevice(0));
    hipModule_t mod;
    HC(hipModuleLoad(&mod, argv[1]));
    hipFunction_t f_qquant, f_score;
    HC(hipModuleGetFunction(&f_qquant, mod, "k_qquant"));
    HC(hipModuleGetFunction(&f_score, mod, "k_score"));

    const unsigned HI = IQS_HI, DI = IQS_DI, M = IQS_M, NP = IQS_N_POOLS;
    const unsigned N_ROWS = M * HI;
    int bad = 0;

    /* ---- Pass 1: DsaQQuant (op 126) ---- */
    std::vector<bf16> q_raw_h(N_ROWS * DI);
    for (unsigned i = 0; i < N_ROWS * DI; i++) q_raw_h[i] = f2b(iqs_q_raw[i]);

    void *d_q_raw, *d_q_fp8, *d_q_scale;
    HC(hipMalloc(&d_q_raw, N_ROWS * DI * sizeof(bf16)));
    HC(hipMalloc(&d_q_fp8, N_ROWS * DI * sizeof(unsigned char)));
    HC(hipMalloc(&d_q_scale, N_ROWS * sizeof(float)));
    HC(hipMemcpy(d_q_raw, q_raw_h.data(), N_ROWS * DI * sizeof(bf16), hipMemcpyHostToDevice));

    QArgs qa = {};
    qa.q_fp8 = (unsigned long long)d_q_fp8;
    qa.q_scale = (unsigned long long)d_q_scale;
    qa.q_raw = (unsigned long long)d_q_raw;
    qa.n_rows = N_ROWS;
    qa.head_dim = DI;
    void* d_qa;
    HC(hipMalloc(&d_qa, sizeof(qa)));
    HC(hipMemcpy(d_qa, &qa, sizeof(qa), hipMemcpyHostToDevice));
    void* qparams[] = {&d_qa};
    HC(hipModuleLaunchKernel(f_qquant, 4, 1, 1, 512, 1, 1, 0, 0, qparams, nullptr));
    HC(hipDeviceSynchronize());

    std::vector<unsigned char> q_fp8_h(N_ROWS * DI);
    std::vector<float> q_scale_h(N_ROWS);
    HC(hipMemcpy(q_fp8_h.data(), d_q_fp8, N_ROWS * DI, hipMemcpyDeviceToHost));
    HC(hipMemcpy(q_scale_h.data(), d_q_scale, N_ROWS * sizeof(float), hipMemcpyDeviceToHost));

    int qquant_bad = 0;
    for (unsigned i = 0; i < N_ROWS; i++) {
        if (q_scale_h[i] != iqs_q_scale_want[i]) {
            if (qquant_bad < 5) printf("MISMATCH q_scale[%u]: got %.9g want %.9g\n", i, q_scale_h[i], iqs_q_scale_want[i]);
            qquant_bad++;
        }
    }
    for (unsigned i = 0; i < N_ROWS * DI; i++) {
        if (q_fp8_h[i] != iqs_q_fp8_want[i]) {
            if (qquant_bad < 5) printf("MISMATCH q_fp8[%u]: got %u want %u\n", i, q_fp8_h[i], iqs_q_fp8_want[i]);
            qquant_bad++;
        }
    }
    printf("DsaQQuant: %d mismatch(es) over %u scales + %u fp8 bytes\n", qquant_bad, N_ROWS, N_ROWS * DI);
    bad += qquant_bad;

    /* ---- Pass 2: IndexScoreKpool (op 127) ---- */
    /* NON-POOL-ALIGNED OCCUPANCY REGRESSION TEST. A prior version of this op took a
     * pool-granular `pool_len` straight from the caller; the fix takes the SAME
     * token-granular `kv_len` every other GLM decode op reads and divides by `pool_size`
     * internally, FLOOR (see op_attention.h / dev.rs's DevOp::IndexScoreKpool doc for
     * why floor, not ceiling). A test that only ever passes an exact multiple of
     * pool_size cannot distinguish "derives len correctly" from "takes len directly" —
     * so row 0 here uses kv_len=7 with pool_size=4 -> len=1 (a genuinely INCOMPLETE
     * second pool, deliberately excluded from scoring), row 1 uses kv_len=8 -> len=2
     * (exact multiple, the original fully-aligned case, kept as a regression check).
     * Duplicate the shared k pool array per batch row -- matches d_index_score's own
     * per-batch-row kv_stride indexing convention (Kg + (b*stride+p)*DI). */
    const unsigned POOL_SIZE = 4;
    std::vector<unsigned char> k_fp8_dup(M * NP * DI);
    std::vector<float> k_scale_dup(M * NP);
    for (unsigned b = 0; b < M; b++) {
        memcpy(&k_fp8_dup[b * NP * DI], iqs_k_fp8, NP * DI);
        memcpy(&k_scale_dup[b * NP], iqs_k_scale, NP * sizeof(float));
    }
    std::vector<int> kv_len_h(M);
    kv_len_h[0] = 7; /* -> len = 7/4 = 1, pool 1 must NOT be touched */
    for (unsigned b = 1; b < M; b++) kv_len_h[b] = (int)(NP * POOL_SIZE); /* fully aligned */

    const float POISON = -999.0f;
    std::vector<float> score_init(M * NP, POISON);

    void *d_kfp8, *d_kscale, *d_w, *d_kvlen, *d_score;
    HC(hipMalloc(&d_kfp8, k_fp8_dup.size()));
    HC(hipMalloc(&d_kscale, k_scale_dup.size() * sizeof(float)));
    HC(hipMalloc(&d_w, M * HI * sizeof(float)));
    HC(hipMalloc(&d_kvlen, M * sizeof(int)));
    HC(hipMalloc(&d_score, M * NP * sizeof(float)));
    HC(hipMemcpy(d_kfp8, k_fp8_dup.data(), k_fp8_dup.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_kscale, k_scale_dup.data(), k_scale_dup.size() * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_w, iqs_weights_raw, M * HI * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_kvlen, kv_len_h.data(), M * sizeof(int), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_score, score_init.data(), score_init.size() * sizeof(float), hipMemcpyHostToDevice));

    SArgs sa = {};
    sa.score = (unsigned long long)d_score;
    sa.qfp8 = (unsigned long long)d_q_fp8; /* reuse pass-1's quantized q (M*HI rows, contiguous) */
    sa.qscale = (unsigned long long)d_q_scale;
    sa.kfp8 = (unsigned long long)d_kfp8;
    sa.kscale = (unsigned long long)d_kscale;
    sa.w = (unsigned long long)d_w;
    sa.kv_len = (unsigned long long)d_kvlen;
    sa.n_batch = M;
    sa.index_heads = HI;
    sa.pool_stride = NP;
    sa.pool_size = POOL_SIZE;
    sa.scale = 1.0f / (sqrtf((float)DI) * sqrtf((float)HI));
    void* d_sa;
    HC(hipMalloc(&d_sa, sizeof(sa)));
    HC(hipMemcpy(d_sa, &sa, sizeof(sa), hipMemcpyHostToDevice));
    void* sparams[] = {&d_sa};
    HC(hipModuleLaunchKernel(f_score, 4, 1, 1, 512, 1, 1, 0, 0, sparams, nullptr));
    HC(hipDeviceSynchronize());

    std::vector<float> score_h(M * NP);
    HC(hipMemcpy(score_h.data(), d_score, M * NP * sizeof(float), hipMemcpyDeviceToHost));

    int score_bad = 0;
    double max_rel = 0.0;
    for (unsigned b = 0; b < M; b++) {
        const unsigned len = (unsigned)kv_len_h[b] / POOL_SIZE;
        for (unsigned p = 0; p < NP; p++) {
            const unsigned i = b * NP + p;
            const double got = score_h[i];
            if (p >= len) {
                /* Excluded pool (incomplete trailing pool, or beyond live occupancy):
                 * must be left UNTOUCHED at the poison sentinel -- this is the actual
                 * regression check for the granularity fix. A pre-fix kernel that took
                 * pool_len directly (misinterpreting kv_len=7 as "7 pools") would instead
                 * try to score pools far past pool_stride, corrupting or crashing. */
                if (got != POISON) {
                    printf("MISMATCH score[%u] (b=%u,p=%u, EXCLUDED pool): got %.9g, expected "
                           "untouched poison %.9g\n", i, b, p, got, POISON);
                    score_bad++;
                }
                continue;
            }
            const double want = iqs_logits_want[i];
            const double rel = fabs(got - want) / (fabs(want) > 1e-6 ? fabs(want) : 1.0);
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-3) {
                printf("MISMATCH score[%u] (b=%u,p=%u): got %.9g want %.9g (rel %.6g)\n", i, b, p,
                       got, want, rel);
                score_bad++;
            }
        }
    }
    printf("IndexScoreKpool: %d mismatch(es) over %u logits (granularity-aware), max relerr %.6g\n",
           score_bad, M * NP, max_rel);
    bad += score_bad;

    printf("indexer qscore check: %d mismatch(es)\n", bad);
    return bad ? 1 : 0;
}
#endif
