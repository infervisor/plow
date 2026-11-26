/* sm120_interp_op_test.cu — numeric validation of the sm_120 packet-interpreter op bodies
 * (runtime/nvidia/op_*.cuh) against an inline f32 CPU oracle.
 *
 * These are the EXACT device templates the megakernel dispatches (this TU #includes the same
 * headers), so a PASS here is a statement about the interpreter's arithmetic, decoupled from the
 * whole-network gate in qwen3_sm120_chat / gemma4_sm120_chat. Method per op:
 *   1. gen f32 inputs, round to bf16 (the oracle sees exactly what the GPU sees, so input
 *      quantization is not charged against the kernel — only its own arithmetic is),
 *   2. run the f32 CPU reference,
 *   3. run the device template through a thin __global__ wrapper,
 *   4. diff: relL2 = ||gpu-ref|| / ||ref|| and max abs err.
 *
 * COVERAGE is the Gemma-4 decode gap (rtx-06 G1–G4):
 *   - d_headnorm_rope<HD> at HD = 128 / 256 / 512   (norm + half-split RoPE)
 *   - d_flash_decode<HD,GF> + d_flash_merge<HD> at (128,4) / (256,2) / (512,2)
 *   - d_norm_residual_norm / d_norm_residual        (Gemma sandwich norm + folded layer_scalar)
 *   - d_softcap                                     (30*tanh final logit softcap)
 *
 * NEGATIVE CONTROL: compiled a second time with -DFA_NV_WAVE64_NEGCTRL (CMake target
 * sm120_interp_op_test_w64, WILL_FAIL). That flips FA_RED_OFF0 to the naive wave64 offset (32 on
 * a 32-lane warp), which leaves half of every warp reduction unfolded — the flash/headnorm/norm
 * ops MUST fail, which is what makes the passing run mean something.
 *
 * Build (standalone):
 *   nvcc -arch=sm_120a -I ../common -I . sm120_interp_op_test.cu -o sm120_interp_op_test
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef PLOW_PACKET_LINEAR_BIAS
#define PLOW_PACKET_LINEAR_BIAS 1
#endif
/* The sm_12x tensor-core W8A16 decode GEMV (op_gemv_fp8_tc.cuh) behind the arena overload, as a
 * Nemotron-H decode object (tuning `fp8_decode_tc` = 4) compiles it. */
#ifndef PLOW_NV_FP8_DECODE_TC_ON
#define PLOW_NV_FP8_DECODE_TC_ON 1
#endif
#ifndef PLOW_NV_FP8_DECODE_TC
#define PLOW_NV_FP8_DECODE_TC 4
#endif

#include "sm120_common.cuh"   /* pulls op_attention.cuh (flash + bf16v8/reductions) */
#include "op_norm.cuh"
#include "op_elementwise.cuh"
#include "op_gemm.cuh"        /* prefill: d_gemm / d_gemm_glu; decode: d_gemv_fp8 / d_gemv_glu_fp8 */
#include "dev_isa.h"          /* PLOW_EXPERT_UNUSED for the grouped-MoE prefill bodies */
#include "op_moe.cuh"         /* Gemma-4 26B-A4B bf16 MoE decode bodies + grouped prefill bodies */
#include <cuda_fp8.h>

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

/* ---- host helpers -------------------------------------------------------- */
static uint32_t rng_s = 0x2468ace0u;
static float rnd() { rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
    return (float)((int32_t)rng_s) / 2147483648.0f; }
static void seed(uint32_t s) { rng_s = s ? s : 1; }
static float bf16_rt(float x) { return __bfloat162float(__float2bfloat16(x)); }
static std::vector<float> gen_bf16(size_t n, float amp) {
    std::vector<float> v(n); for (size_t i=0;i<n;i++) v[i]=bf16_rt(rnd()*amp); return v;
}
static bf16* to_dev(const std::vector<float>& h) {
    std::vector<bf16> t(h.size()); for (size_t i=0;i<h.size();i++) t[i]=__float2bfloat16(h[i]);
    bf16* d=nullptr; CK(cudaMalloc(&d,h.size()*sizeof(bf16)));
    CK(cudaMemcpy(d,t.data(),h.size()*sizeof(bf16),cudaMemcpyHostToDevice)); return d;
}
static float* to_dev_f(const std::vector<float>& h) {
    float* d=nullptr; CK(cudaMalloc(&d,h.size()*sizeof(float)));
    CK(cudaMemcpy(d,h.data(),h.size()*sizeof(float),cudaMemcpyHostToDevice)); return d;
}
static float* dev_f(size_t n){ float* d=nullptr; CK(cudaMalloc(&d,n*sizeof(float)));
    CK(cudaMemset(d,0,n*sizeof(float))); return d; }
static bf16* dev_bf(size_t n){ bf16* d=nullptr; CK(cudaMalloc(&d,n*sizeof(bf16)));
    CK(cudaMemset(d,0,n*sizeof(bf16))); return d; }
static int* to_dev_i(const std::vector<int>& h){ int* d=nullptr; CK(cudaMalloc(&d,h.size()*4));
    CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static std::vector<float> from_dev(const bf16* d, size_t n){
    std::vector<bf16> t(n); CK(cudaMemcpy(t.data(),d,n*sizeof(bf16),cudaMemcpyDeviceToHost));
    std::vector<float> h(n); for(size_t i=0;i<n;i++) h[i]=__bfloat162float(t[i]); return h;
}

static int g_fail = 0;
static void report(const char* name, const std::vector<float>& ref,
                   const std::vector<float>& got, bool expect_pass, double tol=2e-2) {
    if (ref.size()!=got.size()){ printf("  %-40s SIZE MISMATCH\n",name); g_fail=1; return; }
    double num=0,den=0,maxabs=0; size_t nan=0;
    for (size_t i=0;i<ref.size();i++){
        double r=ref[i], g=got[i];
        if (std::isnan(g)||std::isinf(g)){ nan++; continue; }
        double d=g-r; num+=d*d; den+=r*r; if (fabs(d)>maxabs) maxabs=fabs(d);
    }
    double relL2 = den>0 ? sqrt(num/den) : (num>0?1e30:0);
    bool pass = (relL2<=tol) && nan==0;
    printf("  %-40s n=%-8zu relL2=%-11.4g maxabs=%-11.4g%s -> %s\n",
           name, ref.size(), relL2, maxabs,
           nan? " NaN!" : "", pass?"PASS":"FAIL");
    if (pass != expect_pass) g_fail = 1;
}

/* ==================== fp8 (w8a16) DECODE GEMV (G7) ======================== */
/* Reference is the DEQUANTIZED-weight f32 matmul: the weight is per-row e4m3-quantized
 * (scale=amax/448, exactly the quantize_fp8.py convention), then dequantized back to
 * float for the golden. So this isolates the KERNEL's arithmetic (bf16 x, f32 accumulate,
 * scale factored once into the epilogue) from the e4m3 quantization error itself — the
 * device and the reference see the SAME weight values, and e4m3->fp16->f32 on device is
 * exact, so the only gap is bf16-x rounding + warp-tree accumulation order (~1e-3). */
static uint8_t e4m3_enc(float v){ __nv_fp8_e4m3 q(v); return *(const uint8_t*)&q; }
static float   e4m3_dec(uint8_t b){ __nv_fp8_e4m3 q; *(uint8_t*)&q=b; return (float)q; }

__global__ void k_gemv_fp8(bf16* C, const bf16* x, const uint8_t* W, const float* sc,
                           unsigned M, unsigned N, unsigned K) {
    d_gemv_fp8(C, x, W, sc, M, N, K, blockIdx.x, gridDim.x);
}
/* BATCH>1 (PLOW_DECODE_BATCH up to 32): C is [M][N] and EVERY row must be written from its own
 * x row. The pre-batch fp8 arm was scalar-accumulator, so rows 1..M-1 came back as whatever the
 * buffer held; dC is pre-poisoned with NaN so an unwritten row fails loudly instead of passing
 * on a zeroed allocation. Ragged M (5, 17) exercises the {1,2,4,8}-ladder + block-walk remainder. */
/* The interpreter's arena overload: from M = PLOW_NV_FP8_DECODE_TC the tensor-core walk on 48 CTAs
 * (GB10's SM count, the per-CTA row split of the served object). */
__global__ void k_gemv_fp8_arena(bf16* C, const bf16* x, const uint8_t* W, const float* sc,
                                 unsigned M, unsigned N, unsigned K) {
    extern __shared__ __align__(16) unsigned char gemv_arena[];
    d_gemv_fp8(C, x, W, sc, M, N, K, blockIdx.x, gridDim.x, (bf16*)gemv_arena);
}
static void test_gemv_fp8(const char* label, unsigned N, unsigned K, unsigned M = 1, bool arena = false) {
    seed(0x5000u + N*3 + K);
    std::vector<float> W = gen_bf16((size_t)N*K, 1.0f);
    std::vector<float> x = gen_bf16((size_t)M*K, 1.0f);
    std::vector<uint8_t> W8((size_t)N*K);
    std::vector<float> sc(N), ref((size_t)M*N);
    for (unsigned n=0;n<N;n++){
        float amax=0; for(unsigned k=0;k<K;k++) amax=fmaxf(amax,fabsf(W[(size_t)n*K+k]));
        float scale = amax>0.0f ? amax/448.0f : 1.0f; sc[n]=scale;
        for(unsigned k=0;k<K;k++) W8[(size_t)n*K+k]=e4m3_enc(W[(size_t)n*K+k]/scale);
        for (unsigned m=0;m<M;m++){
            double acc=0;
            for(unsigned k=0;k<K;k++)
                acc += (double)e4m3_dec(W8[(size_t)n*K+k]) * (double)x[(size_t)m*K+k];
            ref[(size_t)m*N+n]=(float)(acc*(double)scale);
        }
    }
    bf16* dx=to_dev(x); bf16* dC=dev_bf((size_t)M*N); float* dsc=to_dev_f(sc);
    CK(cudaMemset(dC,0x7f,(size_t)M*N*sizeof(bf16))); /* NaN-poison: unwritten row -> FAIL */
    uint8_t* dW; CK(cudaMalloc(&dW,(size_t)N*K));
    CK(cudaMemcpy(dW,W8.data(),(size_t)N*K,cudaMemcpyHostToDevice));
    if (arena) {
        const unsigned bytes = PLOW_FP8TC_ARENA_BYTES > 2u * K ? PLOW_FP8TC_ARENA_BYTES : 2u * K;
        k_gemv_fp8_arena<<<48,256,bytes>>>(dC,dx,dW,dsc,M,N,K);
    } else {
        k_gemv_fp8<<<64,256>>>(dC,dx,dW,dsc,M,N,K);
    }
    CK(cudaDeviceSynchronize());
    report(label, ref, from_dev(dC,(size_t)M*N), true, 5e-3);
    cudaFree(dx);cudaFree(dC);cudaFree(dW);cudaFree(dsc);
}

__global__ void k_gemv_glu_fp8(bf16* C, const bf16* x, const uint8_t* Wg, const uint8_t* Wu,
                               const float* sg, const float* su, unsigned M, unsigned N,
                               unsigned K, unsigned act) {
    d_gemv_glu_fp8(C, x, Wg, Wu, sg, su, M, N, K, act, blockIdx.x, gridDim.x);
}
static void test_gemv_glu_fp8(const char* label, unsigned N, unsigned K, unsigned act,
                              unsigned M = 1) {
    seed(0x6000u + N*3 + K + act);
    std::vector<float> Wg=gen_bf16((size_t)N*K,1.0f), Wu=gen_bf16((size_t)N*K,1.0f);
    std::vector<float> x=gen_bf16((size_t)M*K,1.0f);
    std::vector<uint8_t> Wg8((size_t)N*K), Wu8((size_t)N*K);
    std::vector<float> sg(N), su(N), ref((size_t)M*N);
    auto quant_row=[&](const std::vector<float>& W, std::vector<uint8_t>& W8,
                       std::vector<float>& sc, unsigned n){
        float amax=0; for(unsigned k=0;k<K;k++) amax=fmaxf(amax,fabsf(W[(size_t)n*K+k]));
        float scale=amax>0.0f?amax/448.0f:1.0f; sc[n]=scale;
        for(unsigned k=0;k<K;k++) W8[(size_t)n*K+k]=e4m3_enc(W[(size_t)n*K+k]/scale);
    };
    auto dot_row=[&](const std::vector<uint8_t>& W8, const std::vector<float>& sc,
                     unsigned n, unsigned m)->float{
        double acc=0;
        for(unsigned k=0;k<K;k++) acc += (double)e4m3_dec(W8[(size_t)n*K+k])*(double)x[(size_t)m*K+k];
        return (float)(acc*(double)sc[n]);
    };
    for (unsigned n=0;n<N;n++){
        quant_row(Wg,Wg8,sg,n); quant_row(Wu,Wu8,su,n);
        for (unsigned m=0;m<M;m++){
            float g=dot_row(Wg8,sg,n,m), u=dot_row(Wu8,su,n,m);
            float a = act==PLOW_ACT_SILU_ ? (g/(1.0f+expf(-g)))
                : 0.5f*g*(1.0f+tanhf(0.7978845608028654f*(g+0.044715f*g*g*g)));
            ref[(size_t)m*N+n]=a*u;
        }
    }
    bf16* dx=to_dev(x); bf16* dC=dev_bf((size_t)M*N);
    CK(cudaMemset(dC,0x7f,(size_t)M*N*sizeof(bf16))); /* NaN-poison */
    float* dsg=to_dev_f(sg); float* dsu=to_dev_f(su);
    uint8_t *dWg,*dWu; CK(cudaMalloc(&dWg,(size_t)N*K)); CK(cudaMalloc(&dWu,(size_t)N*K));
    CK(cudaMemcpy(dWg,Wg8.data(),(size_t)N*K,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWu,Wu8.data(),(size_t)N*K,cudaMemcpyHostToDevice));
    k_gemv_glu_fp8<<<64,256>>>(dC,dx,dWg,dWu,dsg,dsu,M,N,K,act);
    CK(cudaDeviceSynchronize());
    report(label, ref, from_dev(dC,(size_t)M*N), true, 1e-2);
    cudaFree(dx);cudaFree(dC);cudaFree(dsg);cudaFree(dsu);cudaFree(dWg);cudaFree(dWu);
}

__global__ void k_gemv_glu_fp8_arena(bf16* C, const bf16* x, const uint8_t* Wg, const uint8_t* Wu,
                                     const float* sg, const float* su, unsigned M, unsigned N,
                                     unsigned K, unsigned act) {
    extern __shared__ __align__(16) unsigned char gemv_arena[];
    d_gemv_glu_fp8(C, x, Wg, Wu, sg, su, M, N, K, act, blockIdx.x, gridDim.x, (bf16*)gemv_arena);
}
/* Row-count invariance of the decode FP8 GEMV as the interpreter runs it (arena overload, 48 CTAs):
 * row m of every M-row call (M = 1..32) is bitwise equal to row m of the 32-row call over the same
 * x rows, so a span verify reproduces plain decode at any other row count. Must hold from the TC
 * rung on with PLOW_NV_FP8TC_ROW_INVARIANT; otherwise the differing M are only listed. act < 0 =
 * single matrix, else GLU with that activation. */
static void test_gemv_fp8_rows(const char* label, unsigned N, unsigned K, int act = -1) {
    const unsigned MX = 32;
    const bool glu = act >= 0;
    uint64_t s = 0x9e3779b97f4a7c15ull ^ ((uint64_t)N << 20) ^ K;
    auto bytes = [&](size_t n) {
        std::vector<uint8_t> w(n);
        for (size_t i = 0; i < n; i++) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            uint8_t b = (uint8_t)(s >> 56);
            w[i] = (b & 0x7f) == 0x7f ? (uint8_t)(b ^ 1u) : b; /* no e4m3 NaN */
        }
        return w;
    };
    seed(0x7000u + N * 3 + K);
    std::vector<float> x = gen_bf16((size_t)MX * K, 1.0f), sg(N), su(N);
    for (unsigned n = 0; n < N; n++) { sg[n] = (1.5f + rnd()) / 448.0f; su[n] = (1.5f + rnd()) / 448.0f; }
    std::vector<uint8_t> Wg = bytes((size_t)N * K), Wu = glu ? bytes((size_t)N * K) : std::vector<uint8_t>();
    bf16* dx = to_dev(x); bf16* dC = dev_bf((size_t)MX * N);
    float* dsg = to_dev_f(sg); float* dsu = to_dev_f(su);
    uint8_t *dWg, *dWu = nullptr;
    CK(cudaMalloc(&dWg, (size_t)N * K)); CK(cudaMemcpy(dWg, Wg.data(), (size_t)N * K, cudaMemcpyHostToDevice));
    if (glu) { CK(cudaMalloc(&dWu, (size_t)N * K)); CK(cudaMemcpy(dWu, Wu.data(), (size_t)N * K, cudaMemcpyHostToDevice)); }
    const unsigned smem = PLOW_FP8TC_ARENA_BYTES > 2u * K ? PLOW_FP8TC_ARENA_BYTES : 2u * K;
    auto run = [&](unsigned M) {
        CK(cudaMemset(dC, 0x7f, (size_t)MX * N * sizeof(bf16)));
        if (glu) k_gemv_glu_fp8_arena<<<48, 256, smem>>>(dC, dx, dWg, dWu, dsg, dsu, M, N, K, (unsigned)act);
        else k_gemv_fp8_arena<<<48, 256, smem>>>(dC, dx, dWg, dsg, M, N, K);
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> h((size_t)M * N);
        CK(cudaMemcpy(h.data(), dC, h.size() * 2, cudaMemcpyDeviceToHost));
        return h;
    };
    const std::vector<uint16_t> full = run(MX);
    char bad[256] = ""; size_t nb = 0; unsigned first_tc_bad = 0;
    for (unsigned M = 1; M < MX; M++) {
        const std::vector<uint16_t> got = run(M);
        if (memcmp(got.data(), full.data(), got.size() * 2) == 0) continue;
        if (M >= PLOW_NV_FP8_DECODE_TC && !first_tc_bad) first_tc_bad = M;
        if (nb < sizeof(bad) - 8) nb += snprintf(bad + nb, sizeof(bad) - nb, " %u", M);
    }
    const bool must = PLOW_NV_FP8TC_ROW_INVARIANT != 0;
    printf("  %-40s M=1..%u vs %u: %s%s%s\n", label, MX - 1, MX, nb ? "differ at M =" : "bitwise", bad,
           first_tc_bad ? (must ? " -> FAIL" : " (TC rows; route not row-invariant)") : (nb ? " (below the TC rung)" : " -> PASS"));
    if (must && first_tc_bad) g_fail = 1;
    cudaFree(dx); cudaFree(dC); cudaFree(dsg); cudaFree(dsu); cudaFree(dWg); if (dWu) cudaFree(dWu);
}

int main() {
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840", 4096, 3840);
    test_gemv_fp8("gemv_fp8 o_proj N3840 K4096", 3840, 4096);
    test_gemv_fp8("gemv_fp8 down   N3840 K15360", 3840, 15360);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840", 15360, 3840, PLOW_ACT_GELU_TANH_);
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840 M=2",  4096, 3840, 2);
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840 M=8",  4096, 3840, 8);
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840 M=32", 4096, 3840, 32);
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840 M=5 (ragged)",  4096, 3840, 5);
    test_gemv_fp8("gemv_fp8 o_proj N3840 K4096 M=17 (ragged)", 3840, 4096, 17);
    test_gemv_fp8("gemv_fp8 down N3840 K15360 M=32", 3840, 15360, 32);
    test_gemv_fp8("gemv_fp8 tc in_proj N18560 K4096 M=4", 18560, 4096, 4, true);
    test_gemv_fp8("gemv_fp8 tc in_proj N18560 K4096 M=5", 18560, 4096, 5, true);
    test_gemv_fp8("gemv_fp8 tc in_proj N18560 K4096 M=16", 18560, 4096, 16, true);
    test_gemv_fp8("gemv_fp8 tc in_proj N18560 K4096 M=32", 18560, 4096, 32, true);
    test_gemv_fp8("gemv_fp8 tc out_proj N4096 K8192 M=4", 4096, 8192, 4, true);
    test_gemv_fp8("gemv_fp8 tc out_proj N4096 K8192 M=17", 4096, 8192, 17, true);
    test_gemv_fp8("gemv_fp8 tc q_proj N4096 K4096 M=8", 4096, 4096, 8, true);
    test_gemv_fp8("gemv_fp8 tc tiny N40 K128 M=4", 40, 128, 4, true);
    test_gemv_fp8("gemv_fp8 tc N1000 K192 M=7", 1000, 192, 7, true);
    test_gemv_fp8_rows("rows qwen in_proj_qkv N10240 K5120", 10240, 5120);
    test_gemv_fp8_rows("rows qwen in_proj_z N6144 K5120", 6144, 5120);
    test_gemv_fp8_rows("rows qwen in_proj_a N48 K5120", 48, 5120);
    test_gemv_fp8_rows("rows qwen in_proj_ab N96 K5120", 96, 5120);
    test_gemv_fp8_rows("rows qwen out_proj/o_proj N5120 K6144", 5120, 6144);
    test_gemv_fp8_rows("rows qwen q_proj N12288 K5120", 12288, 5120);
    test_gemv_fp8_rows("rows qwen k/v_proj N1024 K5120", 1024, 5120);
    test_gemv_fp8_rows("rows qwen down N5120 K17408", 5120, 17408);
    test_gemv_fp8_rows("rows qwen gate/up silu N17408 K5120", 17408, 5120, PLOW_ACT_SILU_);
    test_gemv_fp8_rows("rows nemo in_proj N18560 K4096", 18560, 4096);
    test_gemv_fp8_rows("rows nemo out_proj N4096 K8192", 4096, 8192);
    test_gemv_fp8_rows("rows nemo q/o_proj N4096 K4096", 4096, 4096);
    test_gemv_fp8_rows("rows nemo k/v_proj N256 K4096", 256, 4096);
    test_gemv_fp8_rows("rows nemo lm_head N131072 K4096", 131072, 4096);
    test_gemv_fp8_rows("rows glu gelu N15360 K3840", 15360, 3840, PLOW_ACT_GELU_TANH_);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840 M=2",  15360, 3840, PLOW_ACT_GELU_TANH_, 2);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840 M=8",  15360, 3840, PLOW_ACT_GELU_TANH_, 8);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840 M=32", 15360, 3840, PLOW_ACT_GELU_TANH_, 32);
    test_gemv_glu_fp8("gemv_glu_fp8 silu N15360 K3840 M=5 (ragged)", 15360, 3840, PLOW_ACT_SILU_, 5);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840 M=17 (ragged)", 15360, 3840, PLOW_ACT_GELU_TANH_, 17);
    printf("fp8 gemv driver %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
