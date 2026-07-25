/* mla_correct_sm120.cu — NUMERIC ORACLE for the sm_120 MLA decode + merge-fold arms (P1).
 *
 * The NVIDIA analogue of runtime/tests/mla_gfx950_test.c: it drives the two wired MLA op bodies
 * (op_mla.cuh d_flash_mla_decode_sm120 + d_mla_merge_fold_sm120) against the portable f32 golden
 * produced by runtime/tests/mla_ref.rs (the "MLA1" fixture) and checks the device output at
 * rel_rms < 5e-3 per case — the same tolerance the AMD harness uses. Each dense/gather case is run
 * at BOTH GF=2 and GF=4 to exercise both template instantiations the interp GF-ladder dispatches.
 *
 * These are the SAME device functions interp_sm120.cu inlines into the megakernel; the wrappers
 * call them with the identical operand order as the FLASH_MLA_DECODE / MLA_MERGE_FOLD dispatch
 * arms, so a pass proves the kernel math AND the operand contract the interpreter relies on.
 *
 * Build (needs a GPU at run time):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -I runtime/common -I runtime/nvidia runtime/tests/mla_correct_sm120.cu -o mla_correct
 * Run:  mla_correct fixture.bin
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "op_mla.cuh"

typedef __nv_bfloat16 bf16;
#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA ERR %s: %s\n",#x,cudaGetErrorString(e_));exit(2);} } while(0)

/* ---- wrappers: exactly the interp dispatch call, with slice=blockIdx nblk=gridDim ---- */
template <int GF, bool GATHER>
__global__ void k_mla_decode(float* Op, float* Ml, const bf16* Qa, const bf16* Qr, const bf16* Ckv,
                             const bf16* Kr, const int* kvlen, unsigned nb, unsigned nh,
                             unsigned kvs, float scale, unsigned nsplit, const int* idx,
                             unsigned top_k) {
    extern __shared__ float arena[];
    d_flash_mla_decode_sm120<512, 64, GF, GATHER>(Op, Ml, Qa, Qr, Ckv, Kr, kvlen, nb, nh, kvs,
                                                  /*window*/ 0u, scale, nsplit,
                                                  /*kv_mask*/ 0xFFFFFFFFu, blockIdx.x, gridDim.x,
                                                  arena, idx, top_k);
}
__global__ void k_merge_fold(bf16* O, const float* Op, const float* Ml, const bf16* Wuv,
                             unsigned nb, unsigned nh, unsigned V, unsigned nsplit) {
    extern __shared__ float smem[];
    d_mla_merge_fold_sm120<512, 256>(O, Op, Ml, Wuv, nb, nh, V, nsplit, blockIdx.x, gridDim.x, smem);
}

static float bf2f(uint16_t b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }

static int g_fail = 0;

/* Output-scaled error, identical metric to mla_gfx950_test.c::check. */
static void check(const char* what, const std::vector<uint16_t>& got, const float* want, size_t n) {
    double max_w = 0, max_d = 0, se = 0, sw = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)bf2f(got[i]) - want[i]);
        max_w = fmax(max_w, fabs(want[i]));
        max_d = fmax(max_d, d);
        se += d * d;
        sw += (double)want[i] * want[i];
    }
    double rel_max = max_d / (max_w + 1e-12);
    double rel_rms = sqrt(se / n) / (sqrt(sw / n) + 1e-12);
    bool ok = rel_max < 2e-2 && rel_rms < 5e-3;
    printf("  %-52s %s  (rel_max %.4f  rel_rms %.5f  |O|max=%.3f)\n", what, ok ? "PASS" : "FAIL",
           rel_max, rel_rms, max_w);
    if (!ok) g_fail = 1;
}

/* run decode(GF,GATHER) + merge-fold for one case, return device O (bf16). */
template <int GF, bool GATHER>
static std::vector<uint16_t> run(unsigned nh, unsigned V, unsigned ctx, unsigned nsplit,
                                 unsigned top_k, const bf16* dCkv, const bf16* dKr, const bf16* dQa,
                                 const bf16* dQr, const bf16* dWuv, const int* dLen,
                                 const int* dIdx) {
    const unsigned nb = 1, kvs = ctx;
    const float scale = 0.08838835f; /* fixture: 1/sqrt(128) */
    float *dOp, *dMl;
    CK(cudaMalloc(&dOp, (size_t)nb * nh * nsplit * 512 * 4));
    CK(cudaMalloc(&dMl, (size_t)nb * nh * nsplit * 2 * 4));
    bf16* dO;
    CK(cudaMalloc(&dO, (size_t)nb * nh * V * 2));
    const unsigned n_work = nb * (nh / GF) * nsplit;
    const size_t smem = (size_t)MLA_DEC_SMEM_FLOATS(512, 64, GF) * sizeof(float);
    CK(cudaFuncSetAttribute(k_mla_decode<GF, GATHER>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    k_mla_decode<GF, GATHER><<<n_work, 256, smem>>>(dOp, dMl, dQa, dQr, dCkv, dKr, dLen, nb, nh, kvs,
                                                    scale, nsplit, dIdx, top_k);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const size_t msmem = (size_t)512 * sizeof(float);
    k_merge_fold<<<nb * nh, 256, msmem>>>(dO, dOp, dMl, dWuv, nb, nh, V, nsplit);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> got((size_t)nh * V);
    CK(cudaMemcpy(got.data(), dO, got.size() * 2, cudaMemcpyDeviceToHost));
    cudaFree(dOp); cudaFree(dMl); cudaFree(dO);
    return got;
}

int main(int argc, char** argv) {
    const char* fx = argc > 1 ? argv[1] : "fixture.bin";
    FILE* f = fopen(fx, "rb");
    if (!f) { perror(fx); return 1; }
    fseek(f, 0, SEEK_END); long fn = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(fn);
    if (fread(buf.data(), 1, fn, f) != (size_t)fn) return 1;
    fclose(f);
    const uint8_t* P = buf.data();
    auto ru32 = [&]() { uint32_t v; memcpy(&v, P, 4); P += 4; return v; };
    auto rf32 = [&]() { float v; memcpy(&v, P, 4); P += 4; return v; };
    if (ru32() != 0x4d4c4131u) { fprintf(stderr, "bad fixture magic\n"); return 1; }
    const uint32_t n_cases = ru32();

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("dev0: %s  SMs=%d\n\n", prop.name, prop.multiProcessorCount);
    printf("MLA decode + merge-fold (device vs mla_ref f32 golden):\n");

    for (uint32_t ci = 0; ci < n_cases; ci++) {
        const uint32_t nh = ru32(), DK = ru32(), DR = ru32(), V = ru32();
        const uint32_t ctx = ru32(), nsplit = ru32(), top_k = ru32();
        (void)rf32(); /* scale (we hardcode the fixture's 1/sqrt(128)) */
        const int32_t* hIdx = nullptr;
        if (top_k) { hIdx = (const int32_t*)P; P += (size_t)top_k * 4; }

        const size_t nckv = (size_t)ctx * DK, nkr = (size_t)ctx * DR;
        const size_t nqa = (size_t)nh * DK, nqr = (size_t)nh * DR;
        const size_t nwuv = (size_t)nh * DK * V, no = (size_t)nh * V;
        const bf16* hCkv = (const bf16*)P; P += nckv * 2;
        const bf16* hKr = (const bf16*)P; P += nkr * 2;
        const bf16* hQa = (const bf16*)P; P += nqa * 2;
        const bf16* hQr = (const bf16*)P; P += nqr * 2;
        const bf16* hWuv = (const bf16*)P; P += nwuv * 2;
        const float* golden = (const float*)P; P += no * 4;

        bf16 *dCkv, *dKr, *dQa, *dQr, *dWuv;
        CK(cudaMalloc(&dCkv, nckv * 2)); CK(cudaMemcpy(dCkv, hCkv, nckv * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dKr, nkr * 2)); CK(cudaMemcpy(dKr, hKr, nkr * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dQa, nqa * 2)); CK(cudaMemcpy(dQa, hQa, nqa * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dQr, nqr * 2)); CK(cudaMemcpy(dQr, hQr, nqr * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dWuv, nwuv * 2)); CK(cudaMemcpy(dWuv, hWuv, nwuv * 2, cudaMemcpyHostToDevice));
        int L = (int)ctx; int* dLen; CK(cudaMalloc(&dLen, 4)); CK(cudaMemcpy(dLen, &L, 4, cudaMemcpyHostToDevice));
        int* dIdx = nullptr;
        if (top_k) { CK(cudaMalloc(&dIdx, (size_t)top_k * 4)); CK(cudaMemcpy(dIdx, hIdx, (size_t)top_k * 4, cudaMemcpyHostToDevice)); }

        char lbl[128];
        for (int gfsel = 0; gfsel < 3; gfsel++) {
            const unsigned GF = gfsel == 0 ? 2u : gfsel == 1 ? 4u : 8u;
            if (nh % GF != 0) continue;
            const char* kind = top_k ? "gather" : "dense ";
            snprintf(lbl, sizeof(lbl), "%s nh=%2u ctx=%u ns=%u tk=%u GF=%u", kind, nh, ctx, nsplit,
                     top_k, GF);
            std::vector<uint16_t> got;
            if (top_k) {
                if (GF == 2) got = run<2, true>(nh, V, ctx, nsplit, top_k, dCkv, dKr, dQa, dQr, dWuv, dLen, dIdx);
                else if (GF == 4) got = run<4, true>(nh, V, ctx, nsplit, top_k, dCkv, dKr, dQa, dQr, dWuv, dLen, dIdx);
                else got = run<8, true>(nh, V, ctx, nsplit, top_k, dCkv, dKr, dQa, dQr, dWuv, dLen, dIdx);
            } else {
                if (GF == 2) got = run<2, false>(nh, V, ctx, nsplit, top_k, dCkv, dKr, dQa, dQr, dWuv, dLen, nullptr);
                else if (GF == 4) got = run<4, false>(nh, V, ctx, nsplit, top_k, dCkv, dKr, dQa, dQr, dWuv, dLen, nullptr);
                else got = run<8, false>(nh, V, ctx, nsplit, top_k, dCkv, dKr, dQa, dQr, dWuv, dLen, nullptr);
            }
            check(lbl, got, golden, no);
        }

        cudaFree(dCkv); cudaFree(dKr); cudaFree(dQa); cudaFree(dQr); cudaFree(dWuv); cudaFree(dLen);
        if (dIdx) cudaFree(dIdx);
    }
    printf("\nRESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
