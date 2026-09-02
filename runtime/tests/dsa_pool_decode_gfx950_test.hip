/* dsa_pool_decode_gfx950_test.hip — GLM-5.3-Flash's DSA indexer key-pooling, DECODE side
 * (op 125 `DsaPoolStash` + op 123 `DsaPoolCompress`'s decode-mode gate) against
 * /home/shaswot/plow-work/decode_kpool_oracle.npz — built by actually RUNNING vLLM's real
 * Triton decode-step kernel (`_kpool_decode_update_batched_kernel`,
 * amd/ops/kpool_compress.py) on this box's GPU. See op_dsa_pool.h's header and
 * crates/packet/src/dev.rs's `DsaPoolStash`/`DsaPoolCompress` doc comments for the full
 * contract.
 *
 * Simulates 6 sequential single-token decode steps (pool_size=4): every step stashes into
 * the ring (`k_stash`), then calls the decode-mode compress (`k_compress_decode`), which is
 * a no-op except at step 3 (the first pool boundary). Checks the ring bytes after EVERY
 * step (proves the stash-only steps are correct, not just the boundary one) and the
 * compressed pool output only at step 3 — also proving non-boundary steps leave the
 * compressed buffer untouched (pre-poisoned with a sentinel before the loop).
 *
 * Two passes, op_dsa_pool.h is device-only by include chain:
 *   hipcc --offload-arch=gfx950 -O3 -w -DDSAPOOL_DECODE_DEVICE --genco \
 *       runtime/tests/dsa_pool_decode_gfx950_test.hip -o /tmp/dpd.co -Iruntime/amd -Iruntime/common
 *   hipcc -O2 -w -x c++ -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       runtime/tests/dsa_pool_decode_gfx950_test.hip -o /tmp/dpd -L/opt/rocm/lib -lamdhip64
 *   perf-data/tools/gpulease -n 1 dpd 'sg render -c "/tmp/dpd /tmp/dpd.co"'
 */

#ifdef DSAPOOL_DECODE_DEVICE
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

#include "decode_kpool_oracle_fixture.h"

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
    if (argc != 2) { printf("usage: %s <dsapool_decode.co>\n", argv[0]); return 1; }
    HC(hipSetDevice(0));
    hipModule_t mod;
    HC(hipModuleLoad(&mod, argv[1]));
    hipFunction_t f_stash, f_compress;
    HC(hipModuleGetFunction(&f_stash, mod, "k_stash"));
    HC(hipModuleGetFunction(&f_compress, mod, "k_compress_decode"));

    const unsigned PS = DKPOOL_POOL_SIZE, HD = DKPOOL_HEAD_DIM;

    void *d_ring_k, *d_ring_score, *d_cur_k, *d_cur_score, *d_pos;
    void *d_ck, *d_cs, *d_ape;
    HC(hipMalloc(&d_ring_k, PS * HD * sizeof(bf16)));
    HC(hipMalloc(&d_ring_score, PS * HD * sizeof(bf16)));
    HC(hipMalloc(&d_cur_k, HD * sizeof(bf16)));
    HC(hipMalloc(&d_cur_score, HD * sizeof(bf16)));
    HC(hipMalloc(&d_pos, sizeof(int)));
    HC(hipMalloc(&d_ck, HD * sizeof(unsigned char)));
    HC(hipMalloc(&d_cs, sizeof(float)));
    HC(hipMalloc(&d_ape, PS * HD * sizeof(float)));
    HC(hipMemset(d_ring_k, 0, PS * HD * sizeof(bf16)));
    HC(hipMemset(d_ring_score, 0, PS * HD * sizeof(bf16)));
    HC(hipMemcpy(d_ape, dkpool_ape, PS * HD * sizeof(float), hipMemcpyHostToDevice));

    void *d_sargs, *d_cargs;
    HC(hipMalloc(&d_sargs, sizeof(StashArgs)));
    HC(hipMalloc(&d_cargs, sizeof(CompressArgs)));

    int bad = 0;
    std::vector<unsigned char> ck(HD);
    std::vector<float> cs(1);
    std::vector<bf16> ring_k_h(PS * HD), ring_score_h(PS * HD);

    for (int pos = 0; pos < DKPOOL_N_STEPS; pos++) {
        /* Poison the compress output before every step so a wrongly-fired compress at a
         * non-boundary step is visible as "didn't match the poison" even without an
         * oracle value to compare against for that step. */
        HC(hipMemset(d_ck, 0xAB, HD * sizeof(unsigned char)));
        float poison_scale = -999.0f;
        HC(hipMemcpy(d_cs, &poison_scale, sizeof(float), hipMemcpyHostToDevice));

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
        cargs.compressed_k = (unsigned long long)d_ck;
        cargs.compressed_scale = (unsigned long long)d_cs;
        cargs.slot_k = (unsigned long long)d_ring_k;
        cargs.slot_score = (unsigned long long)d_ring_score;
        cargs.ape = (unsigned long long)d_ape;
        cargs.pos = (unsigned long long)d_pos;
        cargs.n_pools = 1;
        cargs.pool_size = PS;
        cargs.head_dim = HD;
        HC(hipMemcpy(d_cargs, &cargs, sizeof(cargs), hipMemcpyHostToDevice));
        void* ckparams[] = {&d_cargs};
        HC(hipModuleLaunchKernel(f_compress, 1, 1, 1, 512, 1, 1, 0, 0, ckparams, nullptr));
        HC(hipDeviceSynchronize());

        HC(hipMemcpy(ring_k_h.data(), d_ring_k, PS * HD * sizeof(bf16), hipMemcpyDeviceToHost));
        HC(hipMemcpy(ring_score_h.data(), d_ring_score, PS * HD * sizeof(bf16), hipMemcpyDeviceToHost));

        const float* want_ring_k = nullptr;
        const float* want_ring_score = nullptr;
        switch (pos) {
        case 0: want_ring_k = dkpool_ring_after_step0_k; want_ring_score = dkpool_ring_after_step0_score; break;
        case 1: want_ring_k = dkpool_ring_after_step1_k; want_ring_score = dkpool_ring_after_step1_score; break;
        case 2: want_ring_k = dkpool_ring_after_step2_k; want_ring_score = dkpool_ring_after_step2_score; break;
        case 3: want_ring_k = dkpool_ring_after_step3_k; want_ring_score = dkpool_ring_after_step3_score; break;
        case 4: want_ring_k = dkpool_ring_after_step4_k; want_ring_score = dkpool_ring_after_step4_score; break;
        case 5: want_ring_k = dkpool_ring_after_step5_k; want_ring_score = dkpool_ring_after_step5_score; break;
        }
        int ring_bad = 0;
        for (unsigned i = 0; i < PS * HD; i++) {
            float gk = b2f(ring_k_h[i]), gs = b2f(ring_score_h[i]);
            if (fabsf(gk - want_ring_k[i]) > 1e-3f) { ring_bad++; if (ring_bad <= 3) printf("MISMATCH ring_k step%d[%u]: got %.6f want %.6f\n", pos, i, gk, want_ring_k[i]); }
            if (fabsf(gs - want_ring_score[i]) > 1e-3f) { ring_bad++; if (ring_bad <= 3) printf("MISMATCH ring_score step%d[%u]: got %.6f want %.6f\n", pos, i, gs, want_ring_score[i]); }
        }
        if (ring_bad) { printf("step %d: %d ring mismatch(es)\n", pos, ring_bad); bad += ring_bad; }
        else printf("step %d: ring OK (%s)\n", pos, (pos % (int)PS == (int)PS - 1) ? "boundary" : "stash-only");

        if (pos == 3) {
            /* The compress MATH (softmax/Hadamard/fp8-quant) is already proven byte-exact
             * by dsa_pool_gfx950_test.hip against the same reference family — this test's
             * job is the GATING mechanism (ring stash + conditional fire), so the scale
             * match plus "not still poison" is the right-scoped check here, not a second
             * byte-exact fp8 decode. */
            HC(hipMemcpy(ck.data(), d_ck, HD * sizeof(unsigned char), hipMemcpyDeviceToHost));
            HC(hipMemcpy(cs.data(), d_cs, sizeof(float), hipMemcpyDeviceToHost));
            double sdiff = fabs((double)cs[0] - (double)dkpool_boundary_compressed_scale);
            if (sdiff > 1e-6) { printf("MISMATCH boundary scale: got %.9g want %.9g\n", cs[0], dkpool_boundary_compressed_scale); bad++; }
            bool poisoned = true;
            for (unsigned i = 0; i < HD; i++) if (ck[i] != 0xAB) { poisoned = false; break; }
            if (poisoned) { printf("MISMATCH: boundary step wrote nothing (still poison bytes)\n"); bad++; }
            else printf("step 3: boundary compress fired, scale=%.9g (want %.9g)\n", cs[0], dkpool_boundary_compressed_scale);
        } else {
            HC(hipMemcpy(ck.data(), d_ck, HD * sizeof(unsigned char), hipMemcpyDeviceToHost));
            HC(hipMemcpy(cs.data(), d_cs, sizeof(float), hipMemcpyDeviceToHost));
            bool touched = false;
            for (unsigned i = 0; i < HD; i++) if (ck[i] != 0xAB) { touched = true; break; }
            if (cs[0] != poison_scale) touched = true;
            if (touched) { printf("MISMATCH: non-boundary step %d fired the compress (gate broken)\n", pos); bad++; }
        }
    }

    printf("dsa pool decode-side check: %d mismatch(es)\n", bad);
    return bad ? 1 : 0;
}
#endif
