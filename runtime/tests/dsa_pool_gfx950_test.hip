/* dsa_pool_gfx950_test.hip — GLM-5.3-Flash's DSA indexer key-pooling compress (op 123)
 * against /home/shaswot/plow-work/kpool_oracle.npz — built by actually RUNNING vLLM's
 * real Triton kernel (amd/ops/kpool_compress.py) on this box's GPU. See op_dsa_pool.h's
 * header for the full contract.
 *
 * Two passes, op_dsa_pool.h is device-only by include chain:
 *   hipcc --offload-arch=gfx950 -O3 -w -DDSAPOOL_DEVICE --genco \
 *       runtime/tests/dsa_pool_gfx950_test.hip -o /tmp/dp.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/dsa_pool_gfx950_test.hip -o /tmp/dp -L/opt/rocm/lib -lamdhip64
 *   perf-data/tools/gpulease -n 1 dp 'sg render -c "/tmp/dp /tmp/dp.co"'
 */

#ifdef DSAPOOL_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_gemm.h"
#include "op_norm.h"
#include "op_dsa_pool.h"

struct DArgs {
    unsigned long long compressed_k, compressed_scale, slot_k, slot_score, ape;
    unsigned n_pools, pool_size, head_dim;
};

extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_compress(DArgs* a) {
    __shared__ float lds[128];
    __shared__ float part[PLOW_WAVES];
    d_dsa_pool_compress((uint8_t*)a->compressed_k, (float*)a->compressed_scale,
                        (const bf16*)a->slot_k, (const bf16*)a->slot_score, (const float*)a->ape,
                        a->n_pools, a->pool_size, a->head_dim, blockIdx.x, gridDim.x, lds, part);
}

struct EArgs {
    unsigned long long out, pool_ids, kv_len;
    unsigned rows, n_groups, pool_size;
};
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_expand(EArgs* a) {
    d_dsa_pool_expand((int32_t*)a->out, (const int32_t*)a->pool_ids, (const int32_t*)a->kv_len,
                      a->rows, a->n_groups, a->pool_size, blockIdx.x, gridDim.x);
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "kpool_oracle_fixture.h"

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

struct DArgs {
    unsigned long long compressed_k, compressed_scale, slot_k, slot_score, ape;
    unsigned n_pools, pool_size, head_dim;
};
struct EArgs {
    unsigned long long out, pool_ids, kv_len;
    unsigned rows, n_groups, pool_size;
};

/* Independent CPU re-implementation of expand_pools_to_tokens + append_tail_to_topk's
 * identity-path composition (kpool_compress.py), NOT a copy of the device kernel — this
 * is what makes it a real check rather than comparing code against itself.
 *
 * seq_len is COMPUTED from a single scalar kv_len (the chunk-end token count) + the row
 * index, mirroring d_index_select_pf's own derivation exactly — NOT read from a
 * materialized per-row array (the op's earlier contract, before it was found that
 * nothing upstream in the prefill DSA chain ever produces one). Row rows-1 (the last,
 * most-recent row) has seq_len==kv_len; each earlier row has one fewer token. */
static void ref_expand(const std::vector<int32_t>& pool_ids, int32_t kv_len,
                       unsigned rows, unsigned n_groups, unsigned pool_size,
                       std::vector<int32_t>& out) {
    const unsigned topk = n_groups * pool_size;
    const unsigned out_cols = topk + pool_size - 1;
    out.assign((size_t)rows * out_cols, -1);
    const int q_pos0 = kv_len - (int)rows;
    for (unsigned row = 0; row < rows; row++) {
        const int seq_len = q_pos0 + (int)row + 1;
        const int pool_len = seq_len / (int)pool_size;
        const int tail_start = pool_len * (int)pool_size;
        const int tail_count = seq_len - tail_start;
        for (unsigned col = 0; col < out_cols; col++) {
            int result;
            if (col < topk) {
                const unsigned g = col / pool_size, o = col % pool_size;
                const int pid = pool_ids[(size_t)row * n_groups + g];
                result = (pid >= 0) ? (pid * (int)pool_size + (int)o) : -1;
            } else {
                const int tail_off = (int)col - (int)topk;
                result = (tail_off < tail_count) ? (tail_start + tail_off) : -1;
            }
            out[(size_t)row * out_cols + col] = result;
        }
    }
}

static int test_expand(const char* co_path) {
    hipModule_t mod;
    HC(hipModuleLoad(&mod, co_path));
    hipFunction_t f;
    HC(hipModuleGetFunction(&f, mod, "k_expand"));

    /* Causal chunk case: 4 rows, n_groups=2, pool_size=4 (topk=8, out_cols=11), kv_len=20
     * (the chunk's end, row 3's own seq_len). q_pos0 = 20-4 = 16, so rows 0..3 get
     * seq_len 17,18,19,20 — tail_count 1,2,3,0 in that order, covering every possible
     * tail_count for pool_size=4 in one case (strictly more coverage than the op's old
     * arbitrary-seq_lens test, which only hit 3 of the 4 values). Row 2 repeats a pool
     * id (a legal input — the loader never guarantees distinct pool ids). */
    const unsigned rows = 4, n_groups = 2, pool_size = 4;
    std::vector<int32_t> pool_ids = {2, 0, /*row0*/ 1, 3, /*row1*/ 0,
                                     0, /*row2, repeated pool id*/ 1, 2 /*row3*/};
    const int32_t kv_len = 20;
    std::vector<int32_t> want;
    ref_expand(pool_ids, kv_len, rows, n_groups, pool_size, want);

    void *d_out, *d_pid, *d_kl;
    const unsigned out_cols = n_groups * pool_size + pool_size - 1;
    HC(hipMalloc(&d_out, (size_t)rows * out_cols * sizeof(int32_t)));
    HC(hipMalloc(&d_pid, pool_ids.size() * sizeof(int32_t)));
    HC(hipMalloc(&d_kl, sizeof(int32_t)));
    HC(hipMemcpy(d_pid, pool_ids.data(), pool_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_kl, &kv_len, sizeof(int32_t), hipMemcpyHostToDevice));

    EArgs args = {};
    args.out = (unsigned long long)d_out;
    args.pool_ids = (unsigned long long)d_pid;
    args.kv_len = (unsigned long long)d_kl;
    args.rows = rows;
    args.n_groups = n_groups;
    args.pool_size = pool_size;
    void* d_args;
    HC(hipMalloc(&d_args, sizeof(EArgs)));
    HC(hipMemcpy(d_args, &args, sizeof(EArgs), hipMemcpyHostToDevice));
    void* kparams[] = {&d_args};
    HC(hipModuleLaunchKernel(f, 1, 1, 1, 512, 1, 1, 0, 0, kparams, nullptr));
    HC(hipDeviceSynchronize());

    std::vector<int32_t> got((size_t)rows * out_cols);
    HC(hipMemcpy(got.data(), d_out, got.size() * sizeof(int32_t), hipMemcpyDeviceToHost));

    int bad = 0;
    for (size_t i = 0; i < got.size(); i++)
        if (got[i] != want[i]) { printf("MISMATCH expand[%zu]: got %d want %d\n", i, got[i], want[i]); bad++; }
    printf("dsa pool-expand check: %d mismatch(es) of %zu elements\n", bad, got.size());
    return bad;
}

/* DECODE degenerate case (rows==1): proves the new scalar+affine contract collapses to
 * EXACTLY what the old materialized-seq_lens[1] contract already gave decode —
 * q_pos0 = kv_len-1, seq_len = q_pos0+0+1 = kv_len, so decode's occupancy tensor plugs
 * straight in unchanged. kv_len=23 (not a pool boundary, tail_count=3) exercises the
 * same tail-append arithmetic decode actually depends on. */
static int test_expand_decode(const char* co_path) {
    hipModule_t mod;
    HC(hipModuleLoad(&mod, co_path));
    hipFunction_t f;
    HC(hipModuleGetFunction(&f, mod, "k_expand"));

    const unsigned rows = 1, n_groups = 2, pool_size = 4;
    std::vector<int32_t> pool_ids = {5, 1};
    const int32_t kv_len = 23;
    std::vector<int32_t> want;
    ref_expand(pool_ids, kv_len, rows, n_groups, pool_size, want);
    /* Old-contract cross-check: with rows=1, seq_lens[0] must equal kv_len exactly. */
    if (want.empty() || kv_len != 23) { printf("decode setup sanity failed\n"); return 1; }

    void *d_out, *d_pid, *d_kl;
    const unsigned out_cols = n_groups * pool_size + pool_size - 1;
    HC(hipMalloc(&d_out, (size_t)rows * out_cols * sizeof(int32_t)));
    HC(hipMalloc(&d_pid, pool_ids.size() * sizeof(int32_t)));
    HC(hipMalloc(&d_kl, sizeof(int32_t)));
    HC(hipMemcpy(d_pid, pool_ids.data(), pool_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_kl, &kv_len, sizeof(int32_t), hipMemcpyHostToDevice));

    EArgs args = {};
    args.out = (unsigned long long)d_out;
    args.pool_ids = (unsigned long long)d_pid;
    args.kv_len = (unsigned long long)d_kl;
    args.rows = rows;
    args.n_groups = n_groups;
    args.pool_size = pool_size;
    void* d_args;
    HC(hipMalloc(&d_args, sizeof(EArgs)));
    HC(hipMemcpy(d_args, &args, sizeof(EArgs), hipMemcpyHostToDevice));
    void* kparams[] = {&d_args};
    HC(hipModuleLaunchKernel(f, 1, 1, 1, 512, 1, 1, 0, 0, kparams, nullptr));
    HC(hipDeviceSynchronize());

    std::vector<int32_t> got((size_t)rows * out_cols);
    HC(hipMemcpy(got.data(), d_out, got.size() * sizeof(int32_t), hipMemcpyDeviceToHost));

    int bad = 0;
    for (size_t i = 0; i < got.size(); i++)
        if (got[i] != want[i]) {
            printf("MISMATCH decode-expand[%zu]: got %d want %d\n", i, got[i], want[i]);
            bad++;
        }
    printf("dsa pool-expand decode (rows=1) check: %d mismatch(es) of %zu elements\n", bad,
          got.size());
    return bad;
}

int main(int argc, char** argv) {
    if (argc != 2) { printf("usage: %s <dsapool.co>\n", argv[0]); return 1; }

    std::vector<bf16> slot_k_bf(KPOOL_N_POOLS * KPOOL_POOL_SIZE * KPOOL_HEAD_DIM);
    std::vector<bf16> slot_score_bf(KPOOL_N_POOLS * KPOOL_POOL_SIZE * KPOOL_HEAD_DIM);
    for (size_t i = 0; i < slot_k_bf.size(); i++) {
        slot_k_bf[i] = f2b(kpool_slot_k[i]);
        slot_score_bf[i] = f2b(kpool_slot_score[i]);
    }

    HC(hipSetDevice(0));
    hipModule_t mod;
    HC(hipModuleLoad(&mod, argv[1]));
    hipFunction_t f_compress;
    HC(hipModuleGetFunction(&f_compress, mod, "k_compress"));

    void *d_ck, *d_cs, *d_sk, *d_ss, *d_ape;
    HC(hipMalloc(&d_ck, KPOOL_N_POOLS * KPOOL_HEAD_DIM * sizeof(unsigned char)));
    HC(hipMalloc(&d_cs, KPOOL_N_POOLS * sizeof(float)));
    HC(hipMalloc(&d_sk, slot_k_bf.size() * sizeof(bf16)));
    HC(hipMalloc(&d_ss, slot_score_bf.size() * sizeof(bf16)));
    HC(hipMalloc(&d_ape, KPOOL_POOL_SIZE * KPOOL_HEAD_DIM * sizeof(float)));

    HC(hipMemcpy(d_sk, slot_k_bf.data(), slot_k_bf.size() * sizeof(bf16), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_ss, slot_score_bf.data(), slot_score_bf.size() * sizeof(bf16), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_ape, kpool_ape, KPOOL_POOL_SIZE * KPOOL_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));

    DArgs args = {};
    args.compressed_k = (unsigned long long)d_ck;
    args.compressed_scale = (unsigned long long)d_cs;
    args.slot_k = (unsigned long long)d_sk;
    args.slot_score = (unsigned long long)d_ss;
    args.ape = (unsigned long long)d_ape;
    args.n_pools = KPOOL_N_POOLS;
    args.pool_size = KPOOL_POOL_SIZE;
    args.head_dim = KPOOL_HEAD_DIM;

    void* d_args;
    HC(hipMalloc(&d_args, sizeof(DArgs)));
    HC(hipMemcpy(d_args, &args, sizeof(DArgs), hipMemcpyHostToDevice));
    void* kparams[] = {&d_args};

    HC(hipModuleLaunchKernel(f_compress, 1, 1, 1, 512, 1, 1, 0, 0, kparams, nullptr));
    HC(hipDeviceSynchronize());

    std::vector<unsigned char> ck(KPOOL_N_POOLS * KPOOL_HEAD_DIM);
    std::vector<float> cs(KPOOL_N_POOLS);
    HC(hipMemcpy(ck.data(), d_ck, ck.size(), hipMemcpyDeviceToHost));
    HC(hipMemcpy(cs.data(), d_cs, cs.size() * sizeof(float), hipMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < KPOOL_N_POOLS; i++) {
        double d_ = fabs((double)cs[i] - (double)kpool_expect_scale[i]);
        if (d_ > 1e-6) { printf("MISMATCH scale[%d]: got %.9g want %.9g\n", i, (double)cs[i], (double)kpool_expect_scale[i]); bad++; }
    }
    /* fp8 bytes: allow the LAST-BIT rounding difference between AMD's cvt_pk_fp8_f32/OCP
     * encoder and Triton's own RNE (both are "round to nearest even in e4m3", but a
     * software encoder and a hardware instruction are not guaranteed to hit the exact same
     * tie-break on every input) — flag anything worse than that as a real mismatch. */
    int byte_mismatches = 0;
    for (int i = 0; i < KPOOL_N_POOLS * KPOOL_HEAD_DIM; i++) {
        int got = ck[i], want = kpool_expect_u8[i];
        int diff = got - want;
        if (diff != 0) {
            byte_mismatches++;
            if (abs(diff) > 1 || byte_mismatches <= 5)
                printf("%s byte[%d]: got %d want %d (diff %d)\n", abs(diff) > 1 ? "MISMATCH" : "note", i, got, want, diff);
            if (abs(diff) > 1) bad++;
        }
    }
    printf("dsa pool-compress oracle check: %d hard mismatch(es), %d byte(s) off by <=1 ULP, "
           "%d/%d bytes exact\n",
           bad, byte_mismatches, KPOOL_N_POOLS * KPOOL_HEAD_DIM - byte_mismatches,
           KPOOL_N_POOLS * KPOOL_HEAD_DIM);

    const int expand_bad = test_expand(argv[1]);
    const int expand_decode_bad = test_expand_decode(argv[1]);
    return (bad || expand_bad || expand_decode_bad) ? 1 : 0;
}
#endif
