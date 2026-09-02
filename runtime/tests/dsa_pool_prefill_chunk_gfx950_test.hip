/* dsa_pool_prefill_chunk_gfx950_test.hip — MULTI-CHUNK regression test for op 123
 * `DsaPoolCompress`'s PREFILL-mode chunk-base addressing (`chunk_base` parameter).
 *
 * Same bug class as the already-fixed decode-mode addressing
 * (dsa_pool_decode_mb_gfx950_test.hip): a prior version of `d_dsa_pool_compress` had no
 * way to place a prefill chunk's pools anywhere but cache offset 0 — every chunk after
 * the first would silently overwrite the previous chunk's compressed output. Fixed by
 * adding `chunk_base` (token-granular, host-patched per chunk via
 * `crates/plowrt/src/exec/amd.rs`'s `kv_write_row_field`), computing
 * `pool_idx = chunk_base / pool_size` exactly like decode's `pos[0] / pool_size`. See
 * op_dsa_pool.h's file header and `d_dsa_pool_compress`'s own doc comment.
 *
 * This test drives TWO chunks (pool_size=4, n_pools=2 each) into a SHARED, GROWING
 * 4-pool cache buffer — chunk 0 at chunk_base=0 (pool_idx=0, writes pools 0-1), chunk 1
 * at chunk_base=8 (pool_idx=2, writes pools 2-3) — and re-checks pools 0-1's bytes are
 * still correct AFTER chunk 1 writes. Both chunks reuse the SAME oracle-verified input
 * (kpool_oracle_fixture.h, the same fixture dsa_pool_gfx950_test.hip's prefill-mode
 * check already validates against the real Triton kernel) so the expected output for
 * BOTH chunks is identical, known-correct bytes — the only thing under test is WHERE
 * they land, not what they are.
 *
 * Two passes, op_dsa_pool.h is device-only by include chain:
 *   hipcc --offload-arch=gfx950 -O3 -w -DDSAPOOL_CHUNK_DEVICE --genco \
 *       runtime/tests/dsa_pool_prefill_chunk_gfx950_test.hip -o /tmp/dpc.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/dsa_pool_prefill_chunk_gfx950_test.hip -o /tmp/dpc -Iruntime/tests -L/opt/rocm/lib -lamdhip64
 *   perf-data/tools/gpulease -n 1 dpc 'sg render -c "/tmp/dpc /tmp/dpc.co"'
 */

#ifdef DSAPOOL_CHUNK_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_gemm.h"
#include "op_norm.h"
#include "op_dsa_pool.h"

struct CompressArgs {
    unsigned long long compressed_k, compressed_scale, slot_k, slot_score, ape;
    unsigned n_pools, pool_size, head_dim, chunk_base;
};
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_compress_chunk(CompressArgs* a) {
    __shared__ float lds[128];
    __shared__ float part[PLOW_WAVES];
    d_dsa_pool_compress((uint8_t*)a->compressed_k, (float*)a->compressed_scale,
                        (const bf16*)a->slot_k, (const bf16*)a->slot_score, (const float*)a->ape,
                        a->n_pools, a->pool_size, a->head_dim, blockIdx.x, gridDim.x, lds, part,
                        /*pos=*/nullptr, a->chunk_base);
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

struct CompressArgs {
    unsigned long long compressed_k, compressed_scale, slot_k, slot_score, ape;
    unsigned n_pools, pool_size, head_dim, chunk_base;
};

int main(int argc, char** argv) {
    if (argc != 2) { printf("usage: %s <dsapool_chunk.co>\n", argv[0]); return 1; }
    HC(hipSetDevice(0));
    hipModule_t mod;
    HC(hipModuleLoad(&mod, argv[1]));
    hipFunction_t f_compress;
    HC(hipModuleGetFunction(&f_compress, mod, "k_compress_chunk"));

    const unsigned PS = KPOOL_POOL_SIZE, HD = KPOOL_HEAD_DIM, NP_CHUNK = KPOOL_N_POOLS;
    const unsigned NP_TOTAL = NP_CHUNK * 2; /* two chunks' worth, shared growing cache */

    std::vector<bf16> slot_k_bf(NP_CHUNK * PS * HD), slot_score_bf(NP_CHUNK * PS * HD);
    for (size_t i = 0; i < slot_k_bf.size(); i++) {
        slot_k_bf[i] = f2b(kpool_slot_k[i]);
        slot_score_bf[i] = f2b(kpool_slot_score[i]);
    }

    void *d_sk, *d_ss, *d_ape, *d_ck, *d_cs;
    HC(hipMalloc(&d_sk, slot_k_bf.size() * sizeof(bf16)));
    HC(hipMalloc(&d_ss, slot_score_bf.size() * sizeof(bf16)));
    HC(hipMalloc(&d_ape, PS * HD * sizeof(float)));
    HC(hipMalloc(&d_ck, (size_t)NP_TOTAL * HD * sizeof(unsigned char)));
    HC(hipMalloc(&d_cs, NP_TOTAL * sizeof(float)));
    HC(hipMemcpy(d_sk, slot_k_bf.data(), slot_k_bf.size() * sizeof(bf16), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_ss, slot_score_bf.data(), slot_score_bf.size() * sizeof(bf16), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_ape, kpool_ape, PS * HD * sizeof(float), hipMemcpyHostToDevice));

    /* Poison the WHOLE growing cache ONCE, before either chunk -- the point of this test
     * is checking chunk 0's write survives chunk 1's, so the cache must accumulate across
     * chunks, not reset between them (same discipline as dsa_pool_decode_mb_gfx950_test). */
    std::vector<unsigned char> poison_ck((size_t)NP_TOTAL * HD, 0xAB);
    HC(hipMemcpy(d_ck, poison_ck.data(), (size_t)NP_TOTAL * HD, hipMemcpyHostToDevice));
    std::vector<float> poison_cs(NP_TOTAL, -999.0f);
    HC(hipMemcpy(d_cs, poison_cs.data(), NP_TOTAL * sizeof(float), hipMemcpyHostToDevice));

    void* d_args;
    HC(hipMalloc(&d_args, sizeof(CompressArgs)));

    /* Chunk 0: chunk_base=0 tokens -> pool_idx=0, writes pools [0,1].
     * Chunk 1: chunk_base=NP_CHUNK*PS=8 tokens -> pool_idx=2, writes pools [2,3].
     * Same input data both times (kpool_slot_k/score, oracle-verified for pools 0-1) --
     * only the destination offset differs, so a correct kernel reproduces the SAME
     * expected bytes at BOTH pool pairs. */
    const unsigned chunk_bases[2] = {0u, NP_CHUNK * PS};
    for (int chunk = 0; chunk < 2; chunk++) {
        CompressArgs args = {};
        args.compressed_k = (unsigned long long)d_ck;     /* SHARED growing cache */
        args.compressed_scale = (unsigned long long)d_cs; /* SHARED growing cache */
        args.slot_k = (unsigned long long)d_sk;
        args.slot_score = (unsigned long long)d_ss;
        args.ape = (unsigned long long)d_ape;
        args.n_pools = NP_CHUNK;
        args.pool_size = PS;
        args.head_dim = HD;
        args.chunk_base = chunk_bases[chunk];
        HC(hipMemcpy(d_args, &args, sizeof(args), hipMemcpyHostToDevice));
        void* kparams[] = {&d_args};
        HC(hipModuleLaunchKernel(f_compress, 1, 1, 1, 512, 1, 1, 0, 0, kparams, nullptr));
        HC(hipDeviceSynchronize());
    }

    std::vector<unsigned char> ck_all((size_t)NP_TOTAL * HD);
    std::vector<float> cs_all(NP_TOTAL);
    HC(hipMemcpy(ck_all.data(), d_ck, ck_all.size(), hipMemcpyDeviceToHost));
    HC(hipMemcpy(cs_all.data(), d_cs, NP_TOTAL * sizeof(float), hipMemcpyDeviceToHost));

    /* THE ACTUAL REGRESSION CHECK: pool pairs [0,1] (chunk 0) and [2,3] (chunk 1) must
     * BOTH hold the SAME oracle-verified bytes -- if chunk_base were ignored (the fixed
     * bug), chunk 1 would have overwritten pools 0-1 with its own write and left pools
     * 2-3 as poison, or all four pools would collide at [0,1]. */
    int bad = 0;
    for (int chunk = 0; chunk < 2; chunk++) {
        for (unsigned p = 0; p < NP_CHUNK; p++) {
            const unsigned dst_pool = chunk * NP_CHUNK + p;
            double d_ = fabs((double)cs_all[dst_pool] - (double)kpool_expect_scale[p]);
            if (d_ > 1e-6) {
                printf("chunk %d pool %u: MISMATCH scale got %.9g want %.9g\n", chunk, dst_pool,
                       (double)cs_all[dst_pool], (double)kpool_expect_scale[p]);
                bad++;
            }
            int byte_bad = 0;
            for (unsigned d = 0; d < HD; d++) {
                int got = ck_all[(size_t)dst_pool * HD + d];
                int want = kpool_expect_u8[(size_t)p * HD + d];
                if (abs(got - want) > 1) byte_bad++; /* same <=1 ULP encoder slack as the prefill single-chunk test */
            }
            if (byte_bad) {
                printf("chunk %d pool %u: %d byte mismatch(es) -- chunk_base addressing broken\n",
                       chunk, dst_pool, byte_bad);
                bad += byte_bad;
            }
        }
    }
    printf("dsa pool-compress MULTI-CHUNK check: %d mismatch(es) across %u pools\n", bad, NP_TOTAL);
    return bad ? 1 : 0;
}
#endif
