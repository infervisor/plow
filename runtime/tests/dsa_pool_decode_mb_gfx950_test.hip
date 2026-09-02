/* dsa_pool_decode_mb_gfx950_test.hip — MULTI-BOUNDARY regression test for op 123
 * `DsaPoolCompress`'s decode-mode write addressing (paired with op 125 `DsaPoolStash`).
 *
 * A prior version of `d_dsa_pool_compress` always wrote its decode-mode output to offset
 * 0 of the growing per-layer pool cache, regardless of which pool boundary was being
 * crossed — it read `pos[0]` only for the fire/no-fire gate, never to compute WHICH pool
 * slot (0, 1, 2, ...) this boundary belongs to. Across a real multi-boundary decode
 * sequence every pool after the first silently overwrote pool 0. The original
 * `dsa_pool_decode_gfx950_test.hip` only crossed ONE boundary (6 steps, pool_size=4,
 * fires once at step 3), so "always writes slot 0" was indistinguishable from "writes the
 * right slot" in that test — it could not have caught this class of bug.
 *
 * This test crosses TWO boundaries (8 steps, pool_size=4: fires at step 3 -> pool0 and
 * step 7 -> pool1) against a SHARED, GROWING 2-pool cache buffer, and checks pool 0's
 * bytes are still correct and untouched AFTER pool 1 is written — the actual regression
 * check for the fixed bug. Oracle: /home/shaswot/plow-work/decode_kpool_oracle_
 * multiboundary.npz, built by running vLLM's real `kpool_decode_update_and_maybe_write_
 * cache_batched` (amd/ops/kpool_compress.py) on this box's GPU for 8 steps instead of the
 * original oracle's 6 — see decode_kpool_oracle_mb_fixture.h's header for the exact
 * generation script.
 *
 * Two passes, op_dsa_pool.h is device-only by include chain:
 *   hipcc --offload-arch=gfx950 -O3 -w -DDSAPOOL_MB_DEVICE --genco \
 *       runtime/tests/dsa_pool_decode_mb_gfx950_test.hip -o /tmp/dpdmb.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/dsa_pool_decode_mb_gfx950_test.hip -o /tmp/dpdmb -Iruntime/tests -L/opt/rocm/lib -lamdhip64
 *   perf-data/tools/gpulease -n 1 dpdmb 'sg render -c "/tmp/dpdmb /tmp/dpdmb.co"'
 */

#ifdef DSAPOOL_MB_DEVICE
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_gemm.h"
#include "op_norm.h"
#include "op_dsa_pool.h"

struct StashArgs {
    unsigned long long ring_k, ring_score, cur_k, cur_score, pos;
    unsigned pool_size, head_dim;
};
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_stash(StashArgs* a) {
    d_dsa_pool_stash((bf16*)a->ring_k, (bf16*)a->ring_score, (const bf16*)a->cur_k,
                     (const bf16*)a->cur_score, (const int*)a->pos, a->pool_size, a->head_dim);
}

/* compressed_k/compressed_scale point at the GROWING n_pools-wide cache (n_pools=2 here),
 * not a single-pool scratch buffer -- this is the exact shape a real decode program uses. */
struct CompressArgs {
    unsigned long long compressed_k, compressed_scale, slot_k, slot_score, ape, pos;
    unsigned n_pools, pool_size, head_dim;
};
extern "C" __global__ __launch_bounds__(PLOW_THREADS, 2) void k_compress_decode(CompressArgs* a) {
    __shared__ float lds[128];
    __shared__ float part[PLOW_WAVES];
    d_dsa_pool_compress((uint8_t*)a->compressed_k, (float*)a->compressed_scale,
                        (const bf16*)a->slot_k, (const bf16*)a->slot_score, (const float*)a->ape,
                        a->n_pools, a->pool_size, a->head_dim, blockIdx.x, gridDim.x, lds, part,
                        (const int*)a->pos);
}

#else
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "decode_kpool_oracle_mb_fixture.h"

typedef unsigned short bf16;
static bf16 f2b(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static float b2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
#define HC(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_) { printf("hip err %s @%d\n", hipGetErrorString(e_), __LINE__); return 1; }        \
    } while (0)

struct StashArgs {
    unsigned long long ring_k, ring_score, cur_k, cur_score, pos;
    unsigned pool_size, head_dim;
};
struct CompressArgs {
    unsigned long long compressed_k, compressed_scale, slot_k, slot_score, ape, pos;
    unsigned n_pools, pool_size, head_dim;
};

int main(int argc, char** argv) {
    if (argc != 2) { printf("usage: %s <dsapool_decode_mb.co>\n", argv[0]); return 1; }
    HC(hipSetDevice(0));
    hipModule_t mod;
    HC(hipModuleLoad(&mod, argv[1]));
    hipFunction_t f_stash, f_compress;
    HC(hipModuleGetFunction(&f_stash, mod, "k_stash"));
    HC(hipModuleGetFunction(&f_compress, mod, "k_compress_decode"));

    const unsigned PS = DKPOOL_POOL_SIZE, HD = DKPOOL_HEAD_DIM, NP = DKPOOL_N_POOLS;

    void *d_ring_k, *d_ring_score, *d_cur_k, *d_cur_score, *d_pos;
    void *d_ck, *d_cs, *d_ape;
    HC(hipMalloc(&d_ring_k, PS * HD * sizeof(bf16)));
    HC(hipMalloc(&d_ring_score, PS * HD * sizeof(bf16)));
    HC(hipMalloc(&d_cur_k, HD * sizeof(bf16)));
    HC(hipMalloc(&d_cur_score, HD * sizeof(bf16)));
    HC(hipMalloc(&d_pos, sizeof(int)));
    /* THE GROWING CACHE -- n_pools wide, shared/reused exactly like a real per-layer
     * indexer key cache would be across the whole decode sequence. This is what makes the
     * addressing bug visible: a wrong kernel writes every boundary to bytes [0, HD), a
     * correct one writes pool p's boundary to bytes [p*HD, (p+1)*HD). */
    HC(hipMalloc(&d_ck, (size_t)NP * HD * sizeof(unsigned char)));
    HC(hipMalloc(&d_cs, NP * sizeof(float)));
    HC(hipMalloc(&d_ape, PS * HD * sizeof(float)));
    HC(hipMemset(d_ring_k, 0, PS * HD * sizeof(bf16)));
    HC(hipMemset(d_ring_score, 0, PS * HD * sizeof(bf16)));
    HC(hipMemcpy(d_ape, dkpool_ape, PS * HD * sizeof(float), hipMemcpyHostToDevice));
    /* Poison the WHOLE growing cache ONCE, before the loop -- unlike the single-boundary
     * test (which can afford to re-poison every step since it only ever checks the latest
     * write), this test's whole point is checking EARLIER pools survive LATER writes, so
     * the cache must accumulate across steps, not reset. */
    std::vector<unsigned char> poison_ck((size_t)NP * HD, 0xAB);
    HC(hipMemcpy(d_ck, poison_ck.data(), (size_t)NP * HD, hipMemcpyHostToDevice));
    std::vector<float> poison_cs(NP, -999.0f);
    HC(hipMemcpy(d_cs, poison_cs.data(), NP * sizeof(float), hipMemcpyHostToDevice));

    void *d_sargs, *d_cargs;
    HC(hipMalloc(&d_sargs, sizeof(StashArgs)));
    HC(hipMalloc(&d_cargs, sizeof(CompressArgs)));

    int bad = 0;
    std::vector<unsigned char> ck_all((size_t)NP * HD);
    std::vector<float> cs_all(NP);

    const float* want_ring_k[DKPOOL_N_STEPS] = {
        dkpool_ring_after_step0_k, dkpool_ring_after_step1_k, dkpool_ring_after_step2_k,
        dkpool_ring_after_step3_k, dkpool_ring_after_step4_k, dkpool_ring_after_step5_k,
        dkpool_ring_after_step6_k, dkpool_ring_after_step7_k,
    };
    const float* want_ring_score[DKPOOL_N_STEPS] = {
        dkpool_ring_after_step0_score, dkpool_ring_after_step1_score, dkpool_ring_after_step2_score,
        dkpool_ring_after_step3_score, dkpool_ring_after_step4_score, dkpool_ring_after_step5_score,
        dkpool_ring_after_step6_score, dkpool_ring_after_step7_score,
    };
    const unsigned char* want_pool_k_u8[2] = {dkpool_boundary_pool0_compressed_k_u8, dkpool_boundary_pool1_compressed_k_u8};
    const float want_pool_scale[2] = {dkpool_boundary_pool0_compressed_scale, dkpool_boundary_pool1_compressed_scale};

    for (int pos = 0; pos < DKPOOL_N_STEPS; pos++) {
        std::vector<bf16> cur_k(HD), cur_score(HD);
        for (unsigned d = 0; d < HD; d++) {
            cur_k[d] = f2b(dkpool_all_keys[pos * HD + d]);
            cur_score[d] = f2b(dkpool_all_scores[pos * HD + d]);
        }
        HC(hipMemcpy(d_cur_k, cur_k.data(), HD * sizeof(bf16), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_cur_score, cur_score.data(), HD * sizeof(bf16), hipMemcpyHostToDevice));
        HC(hipMemcpy(d_pos, &pos, sizeof(int), hipMemcpyHostToDevice));

        StashArgs sargs = {};
        sargs.ring_k = (unsigned long long)d_ring_k;
        sargs.ring_score = (unsigned long long)d_ring_score;
        sargs.cur_k = (unsigned long long)d_cur_k;
        sargs.cur_score = (unsigned long long)d_cur_score;
        sargs.pos = (unsigned long long)d_pos;
        sargs.pool_size = PS;
        sargs.head_dim = HD;
        HC(hipMemcpy(d_sargs, &sargs, sizeof(sargs), hipMemcpyHostToDevice));
        void* skparams[] = {&d_sargs};
        HC(hipModuleLaunchKernel(f_stash, 1, 1, 1, 512, 1, 1, 0, 0, skparams, nullptr));
        HC(hipDeviceSynchronize());

        CompressArgs cargs = {};
        cargs.compressed_k = (unsigned long long)d_ck;      /* the SHARED growing cache */
        cargs.compressed_scale = (unsigned long long)d_cs;  /* the SHARED growing cache */
        cargs.slot_k = (unsigned long long)d_ring_k;
        cargs.slot_score = (unsigned long long)d_ring_score;
        cargs.ape = (unsigned long long)d_ape;
        cargs.pos = (unsigned long long)d_pos;
        cargs.n_pools = 1; /* decode-mode contract: n_pools MUST be 1 */
        cargs.pool_size = PS;
        cargs.head_dim = HD;
        HC(hipMemcpy(d_cargs, &cargs, sizeof(cargs), hipMemcpyHostToDevice));
        void* ckparams[] = {&d_cargs};
        HC(hipModuleLaunchKernel(f_compress, 1, 1, 1, 512, 1, 1, 0, 0, ckparams, nullptr));
        HC(hipDeviceSynchronize());

        std::vector<bf16> ring_k_h(PS * HD), ring_score_h(PS * HD);
        HC(hipMemcpy(ring_k_h.data(), d_ring_k, PS * HD * sizeof(bf16), hipMemcpyDeviceToHost));
        HC(hipMemcpy(ring_score_h.data(), d_ring_score, PS * HD * sizeof(bf16), hipMemcpyDeviceToHost));
        int ring_bad = 0;
        for (unsigned i = 0; i < PS * HD; i++) {
            float gk = b2f(ring_k_h[i]), gs = b2f(ring_score_h[i]);
            if (fabsf(gk - want_ring_k[pos][i]) > 1e-3f) ring_bad++;
            if (fabsf(gs - want_ring_score[pos][i]) > 1e-3f) ring_bad++;
        }
        if (ring_bad) { printf("step %d: %d ring mismatch(es)\n", pos, ring_bad); bad += ring_bad; }

        /* THE ACTUAL REGRESSION CHECK: after EVERY step, re-read the WHOLE growing cache
         * and verify EVERY pool boundary crossed so far still holds its OWN correct bytes
         * -- not just the latest one. This is what a single-boundary test structurally
         * cannot express. */
        HC(hipMemcpy(ck_all.data(), d_ck, (size_t)NP * HD, hipMemcpyDeviceToHost));
        HC(hipMemcpy(cs_all.data(), d_cs, NP * sizeof(float), hipMemcpyDeviceToHost));
        for (int p = 0; p < 2; p++) {
            const bool crossed = pos >= (p + 1) * (int)PS - 1;
            if (!crossed) continue;
            /* Byte-exact against the oracle's OWN raw fp8 bytes (kv_cache's uint8 view,
             * not a float round-trip) -- same idiom as the already-passing
             * dsa_pool_gfx950_test.hip's prefill-mode check. Allow the same <=1 ULP
             * AMD-encoder-vs-Triton-RNE tie-break slack that test documents; anything
             * worse is a real mismatch, not just this bug's addressing but any
             * regression in the compress math itself. */
            int pool_bad = 0;
            for (unsigned d = 0; d < HD; d++) {
                int got = ck_all[(size_t)p * HD + d], want = want_pool_k_u8[p][d];
                if (abs(got - want) > 1) pool_bad++;
            }
            if (fabsf(cs_all[p] - want_pool_scale[p]) > 1e-6f) pool_bad++;
            if (pool_bad) {
                printf("step %d: pool%d corrupted (%d byte/scale mismatch(es)) -- "
                       "EARLIER pool overwritten by a LATER boundary write\n", pos, p, pool_bad);
                bad += pool_bad;
            }
        }
        printf("step %d: ring OK, pool cache OK so far (%s)\n", pos,
               (pos % (int)PS == (int)PS - 1) ? "boundary" : "stash-only");
    }

    printf("dsa pool decode-side MULTI-BOUNDARY check: %d mismatch(es)\n", bad);
    return bad ? 1 : 0;
}
#endif
