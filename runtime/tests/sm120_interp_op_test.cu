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

/* ==================== HEADNORM + ROPE ===================================== */
template <int HD>
__global__ void k_headnorm_rope(bf16* out, const bf16* x, const bf16* gamma,
                                const float* cosb, const float* sinb, const int* pos,
                                unsigned ntok, unsigned nhead, float eps) {
    d_headnorm_rope<HD>(out, x, gamma, cosb, sinb, pos, ntok, nhead, eps,
                        /*out_row0*/0, /*out_stride*/0, /*kv_mask*/0xFFFFFFFFu,
                        /*skip_norm*/0, blockIdx.x, gridDim.x);
}
template <int HD>
static void test_headnorm_rope(const char* label, unsigned ntok, unsigned nhead) {
    seed(0x1000u + HD + ntok*7 + nhead);
    const float eps = 1e-6f, theta = 1e6f;
    const unsigned H2 = HD/2;
    std::vector<float> X = gen_bf16((size_t)ntok*nhead*HD, 1.0f);
    std::vector<float> gam = gen_bf16(HD, 1.0f);
    std::vector<int> pos(ntok); for (unsigned t=0;t<ntok;t++) pos[t]=(int)(3+t);
    /* cos/sin table [maxpos*H2], full rotary (freq i for i in [0,H2)). */
    unsigned maxpos = 0; for (int p:pos) if ((unsigned)p>maxpos) maxpos=p;
    std::vector<float> cosb((size_t)(maxpos+1)*H2), sinb((size_t)(maxpos+1)*H2);
    for (unsigned p=0;p<=maxpos;p++) for (unsigned i=0;i<H2;i++){
        float freq = powf(theta, -2.0f*(float)i/(float)HD);
        float ang = (float)p*freq;
        cosb[(size_t)p*H2+i]=cosf(ang); sinb[(size_t)p*H2+i]=sinf(ang);
    }
    /* CPU ref */
    std::vector<float> Oref((size_t)ntok*nhead*HD);
    for (unsigned t=0;t<ntok;t++) for (unsigned h=0;h<nhead;h++){
        const float* xr = &X[((size_t)t*nhead+h)*HD];
        std::vector<float> v(HD);
        double ss=0; for (unsigned d=0;d<HD;d++) ss += (double)xr[d]*xr[d];
        float inv = 1.0f/sqrtf((float)(ss/(double)HD)+eps);
        for (unsigned d=0;d<HD;d++) v[d]=xr[d]*inv*gam[d];
        for (unsigned i=0;i<H2;i++){
            float c=cosb[(size_t)pos[t]*H2+i], s=sinb[(size_t)pos[t]*H2+i];
            float x0=v[i], x1=v[i+H2];
            v[i]=x0*c-x1*s; v[i+H2]=x0*s+x1*c;
        }
        for (unsigned d=0;d<HD;d++) Oref[((size_t)t*nhead+h)*HD+d]=v[d];
    }
    bf16 *dX=to_dev(X), *dG=to_dev(gam), *dO=dev_bf((size_t)ntok*nhead*HD);
    float *dC=to_dev_f(cosb), *dS=to_dev_f(sinb); int* dP=to_dev_i(pos);
    k_headnorm_rope<HD><<<64,256>>>(dO,dX,dG,dC,dS,dP,ntok,nhead,eps);
    CK(cudaDeviceSynchronize());
    report(label, Oref, from_dev(dO,(size_t)ntok*nhead*HD), true);
    cudaFree(dX);cudaFree(dG);cudaFree(dO);cudaFree(dC);cudaFree(dS);cudaFree(dP);
}

/* ==================== FLASH DECODE + MERGE ================================ */
template <int D, int GF>
__global__ void k_flash_decode(float* Opart, float* mlpart, const bf16* Q, const bf16* K,
                               const bf16* V, const int* kvlen, unsigned nb, unsigned nh,
                               unsigned nkv, unsigned kvs, unsigned win, float scale,
                               unsigned nsplit) {
    extern __shared__ float arena[];
    d_flash_decode<D,GF>(Opart, mlpart, Q, K, V, kvlen, nb, nh, nkv, kvs, win, scale, nsplit,
                         0xFFFFFFFFu, blockIdx.x, gridDim.x, arena);
}
template <int D>
__global__ void k_flash_merge(bf16* O, const float* Opart, const float* mlpart,
                              unsigned nb, unsigned nh, unsigned nsplit) {
    d_flash_merge<D>(O, Opart, mlpart, nb, nh, nsplit, blockIdx.x, gridDim.x);
}
template <int D, int GF>
static void test_flash(const char* label, unsigned nh, unsigned nkv, unsigned len) {
    seed(0x2000u + D + nh*13 + len);
    const unsigned nb=1, nsplit=4, kvs=len;
    const float scale = 1.0f/sqrtf((float)D);
    std::vector<float> Q = gen_bf16((size_t)nh*D, 1.0f);
    std::vector<float> K = gen_bf16((size_t)nkv*kvs*D, 1.0f);
    std::vector<float> V = gen_bf16((size_t)nkv*kvs*D, 1.0f);
    std::vector<int> kvlen(nb, (int)len);
    const unsigned gqa = nh/nkv;
    /* CPU ref: full causal-at-newest attention (query = row len-1, attends 0..len-1). */
    std::vector<float> Oref((size_t)nh*D, 0.f);
    for (unsigned h=0;h<nh;h++){
        const unsigned hkv=h/gqa;
        std::vector<float> s(len); float mx=-INFINITY;
        for (unsigned r=0;r<len;r++){
            double dot=0; for (unsigned d=0;d<D;d++) dot += (double)Q[(size_t)h*D+d]*K[((size_t)hkv*kvs+r)*D+d];
            s[r]=(float)dot*scale; if (s[r]>mx) mx=s[r];
        }
        float sum=0; for (unsigned r=0;r<len;r++){ s[r]=expf(s[r]-mx); sum+=s[r]; }
        for (unsigned d=0;d<D;d++){ double a=0;
            for (unsigned r=0;r<len;r++) a += (s[r]/sum)*V[((size_t)hkv*kvs+r)*D+d];
            Oref[(size_t)h*D+d]=(float)a; }
    }
    bf16 *dQ=to_dev(Q), *dK=to_dev(K), *dV=to_dev(V); int* dL=to_dev_i(kvlen);
    float *dOp=dev_f((size_t)nb*nh*nsplit*D), *dMl=dev_f((size_t)nb*nh*nsplit*2);
    bf16* dO=dev_bf((size_t)nb*nh*D);
    const unsigned n_work = nb*(nh/GF)*nsplit;
    const size_t smem = (size_t)FA_DEC_SMEM_FLOATS(D,GF)*sizeof(float);
    k_flash_decode<D,GF><<<n_work,256,smem>>>(dOp,dMl,dQ,dK,dV,dL,nb,nh,nkv,kvs,0/*full*/,scale,nsplit);
    CK(cudaDeviceSynchronize());
    k_flash_merge<D><<<nb*nh,256>>>(dO,dOp,dMl,nb,nh,nsplit);
    CK(cudaDeviceSynchronize());
    report(label, Oref, from_dev(dO,(size_t)nb*nh*D), true);
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dL);cudaFree(dOp);cudaFree(dMl);cudaFree(dO);
}

/* ==================== NORM_RESIDUAL_NORM / NORM_RESIDUAL ================== */
__global__ void k_nrn(bf16* out, bf16* resid, const bf16* a, const bf16* b, const bf16* gb,
                      const bf16* gn, unsigned rows, unsigned feat, float eps, float scale) {
    extern __shared__ float part[];
    d_norm_residual_norm(out, resid, a, b, gb, gn, rows, feat, eps, scale, blockIdx.x, gridDim.x, part);
}
__global__ void k_nr(bf16* out, const bf16* a, const bf16* b, const bf16* g,
                     unsigned rows, unsigned feat, float eps, float scale) {
    extern __shared__ float part[];
    d_norm_residual(out, a, b, g, rows, feat, eps, scale, blockIdx.x, gridDim.x, part);
}
static void rms(std::vector<float>& o, const std::vector<float>& x, const std::vector<float>& g,
                unsigned rows, unsigned k, float eps) {
    o.resize(x.size());
    for (unsigned r=0;r<rows;r++){ double ss=0; for(unsigned i=0;i<k;i++){double v=x[(size_t)r*k+i]; ss+=v*v;}
        float inv=1.0f/sqrtf((float)(ss/(double)k)+eps);
        for(unsigned i=0;i<k;i++) o[(size_t)r*k+i]=x[(size_t)r*k+i]*inv*g[i]; }
}
static void test_nrn(const char* label, unsigned rows, unsigned feat, float scale) {
    seed(0x3000u + feat + rows);
    const float eps=1e-6f;
    std::vector<float> A=gen_bf16((size_t)rows*feat,1.0f), B=gen_bf16((size_t)rows*feat,1.0f);
    std::vector<float> gb=gen_bf16(feat,1.0f), gn=gen_bf16(feat,1.0f);
    /* ref: resid=(a+rms(b,gb))*scale rounded to bf16; out=rms(resid,gn) */
    std::vector<float> nb; rms(nb,B,gb,rows,feat,eps);
    std::vector<float> resid((size_t)rows*feat);
    for (size_t i=0;i<resid.size();i++) resid[i]=bf16_rt((A[i]+nb[i])*scale);
    std::vector<float> Oref; rms(Oref,resid,gn,rows,feat,eps);
    for (float& v:Oref) v=bf16_rt(v);
    bf16 *dA=to_dev(A),*dB=to_dev(B),*dgb=to_dev(gb),*dgn=to_dev(gn);
    bf16 *dO=dev_bf((size_t)rows*feat), *dR=dev_bf((size_t)rows*feat);
    k_nrn<<<rows,256,PLOW_NV_WARPS*sizeof(float)>>>(dO,dR,dA,dB,dgb,dgn,rows,feat,eps,scale);
    CK(cudaDeviceSynchronize());
    report(label, Oref, from_dev(dO,(size_t)rows*feat), true);
    cudaFree(dA);cudaFree(dB);cudaFree(dgb);cudaFree(dgn);cudaFree(dO);cudaFree(dR);
}
static void test_nr(const char* label, unsigned rows, unsigned feat, float scale) {
    seed(0x3500u + feat + rows);
    const float eps=1e-6f;
    std::vector<float> A=gen_bf16((size_t)rows*feat,1.0f), B=gen_bf16((size_t)rows*feat,1.0f);
    std::vector<float> g=gen_bf16(feat,1.0f);
    std::vector<float> nb; rms(nb,B,g,rows,feat,eps);
    std::vector<float> Oref((size_t)rows*feat);
    for (size_t i=0;i<Oref.size();i++) Oref[i]=bf16_rt((A[i]+nb[i])*scale);
    bf16 *dA=to_dev(A),*dB=to_dev(B),*dg=to_dev(g), *dO=dev_bf((size_t)rows*feat);
    k_nr<<<rows,256,PLOW_NV_WARPS*sizeof(float)>>>(dO,dA,dB,dg,rows,feat,eps,scale);
    CK(cudaDeviceSynchronize());
    report(label, Oref, from_dev(dO,(size_t)rows*feat), true);
    cudaFree(dA);cudaFree(dB);cudaFree(dg);cudaFree(dO);
}

/* ==================== SOFTCAP ============================================ */
__global__ void k_softcap(bf16* out, const bf16* x, unsigned n, float cap) {
    d_softcap(out, x, n, cap, blockIdx.x, gridDim.x);
}
static void test_softcap(const char* label, unsigned n, float cap) {
    seed(0x4000u + n);
    std::vector<float> X = gen_bf16(n, 40.0f); /* wide enough to exercise the tanh saturation */
    std::vector<float> Oref(n);
    for (unsigned i=0;i<n;i++) Oref[i]=bf16_rt(cap*tanhf(X[i]/cap));
    bf16 *dX=to_dev(X), *dO=dev_bf(n);
    k_softcap<<<64,256>>>(dO,dX,n,cap);
    CK(cudaDeviceSynchronize());
    report(label, Oref, from_dev(dO,n), true);
    cudaFree(dX);cudaFree(dO);
}

/* ==================== PREFILL TILED GEMM / GEMM_GLU ====================== */
__global__ void k_gemm(bf16* C, const bf16* A, const bf16* B, unsigned m, unsigned n, unsigned k,
                       unsigned a_row0) {
    extern __shared__ bf16 smg[];
    d_gemm(C, A, B, m, n, k, a_row0, blockIdx.x, gridDim.x, smg);
}
__global__ void k_gemm_glu(bf16* C, const bf16* A, const bf16* Wg, const bf16* Wu, unsigned m,
                           unsigned n, unsigned k, unsigned act) {
    extern __shared__ bf16 smg[];
    d_gemm_glu(C, A, Wg, Wu, m, n, k, act, blockIdx.x, gridDim.x, smg);
}
static void test_gemm(const char* label, unsigned m, unsigned n, unsigned k, unsigned a_row0) {
    seed(0x5000u + m*3 + n*5 + k + a_row0);
    std::vector<float> A = gen_bf16((size_t)(m+a_row0)*k, 1.0f);
    std::vector<float> B = gen_bf16((size_t)n*k, 1.0f);
    std::vector<float> Cref((size_t)m*n);
    for (unsigned i=0;i<m;i++) for (unsigned j=0;j<n;j++){ double acc=0;
        for (unsigned kk=0;kk<k;kk++) acc += (double)A[(size_t)(a_row0+i)*k+kk]*B[(size_t)j*k+kk];
        Cref[(size_t)i*n+j]=bf16_rt((float)acc); }
    bf16 *dA=to_dev(A), *dB=to_dev(B), *dC=dev_bf((size_t)m*n);
    const size_t smem = (size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_gemm<<<188,256,smem>>>(dC,dA,dB,m,n,k,a_row0);
    CK(cudaDeviceSynchronize());
    report(label, Cref, from_dev(dC,(size_t)m*n), true);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
}
static void test_gemm_glu(const char* label, unsigned m, unsigned n, unsigned k, unsigned act) {
    seed(0x5500u + m + n + k);
    std::vector<float> A = gen_bf16((size_t)m*k, 1.0f);
    std::vector<float> Wg = gen_bf16((size_t)n*k, 1.0f), Wu = gen_bf16((size_t)n*k, 1.0f);
    auto gelu=[&](float x){ float c=0.7978845608028654f*(x+0.044715f*x*x*x); return 0.5f*x*(1.0f+tanhf(c)); };
    auto silu=[&](float x){ return x/(1.0f+expf(-x)); };
    std::vector<float> Cref((size_t)m*n);
    for (unsigned i=0;i<m;i++) for (unsigned j=0;j<n;j++){ double ag=0,au=0;
        for (unsigned kk=0;kk<k;kk++){ ag += (double)A[(size_t)i*k+kk]*Wg[(size_t)j*k+kk];
                                       au += (double)A[(size_t)i*k+kk]*Wu[(size_t)j*k+kk]; }
        float a = act==1 ? silu((float)ag) : gelu((float)ag);
        Cref[(size_t)i*n+j]=bf16_rt(a*(float)au); }
    bf16 *dA=to_dev(A), *dg=to_dev(Wg), *du=to_dev(Wu), *dC=dev_bf((size_t)m*n);
    const size_t smem = (size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_gemm_glu,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_gemm_glu<<<188,256,smem>>>(dC,dA,dg,du,m,n,k,act);
    CK(cudaDeviceSynchronize());
    report(label, Cref, from_dev(dC,(size_t)m*n), true);
    cudaFree(dA);cudaFree(dg);cudaFree(du);cudaFree(dC);
}

/* ==================== FLASH PREFILL (fused ns=1, and split ns>1 + merge) === */
template <int HD, int BQ, int BKV>
__global__ void k_flash_prefill(float* Op, float* Ml, bf16* O, const bf16* Q, const bf16* K,
                                const bf16* V, unsigned sq, unsigned skv, unsigned nh, unsigned nkv,
                                unsigned qpos0, unsigned win, unsigned nsplit, float scale) {
    extern __shared__ float sm[];
    d_flash_prefill<HD,BQ,BKV>(Op, Ml, Q, K, V, O, sq, skv, nh, nkv, qpos0, win, nsplit,
                               /*kv_stride*/skv, /*kv_mask*/0xFFFFFFFFu, scale, blockIdx.x,
                               gridDim.x, sm);
}
/* seq_q query rows sit at absolute positions [q_pos0, q_pos0+seq_q); K/V hold seq_kv rows at
 * absolute positions [0, seq_kv). Causal: kv <= q_pos0+q. Sliding: also q_pos0+q - kv < window.
 * This is the CHUNKED-prefill shape (q_pos0>0, seq_kv>seq_q), the real interpreter case. */
template <int HD, int BQ, int BKV>
static void test_flash_prefill(const char* label, unsigned nh, unsigned nkv, unsigned seq_q,
                               unsigned seq_kv, unsigned q_pos0, unsigned window, unsigned nsplit,
                               float scale = 1.0f) {
    seed(0x6000u + HD + nh*13 + seq_q + seq_kv*3 + q_pos0*5 + window + nsplit*7);
    /* scale=1.0 with amp-1.0 hd256/512 data yields dot~O(HD) scores -> a NEAR-ONE-HOT softmax,
     * where P quantizes to bf16 losslessly (the mma P.V A operand). A SMALL scale (~1/sqrt(HD))
     * gives a SOFT, spread-out softmax that actually exercises the bf16-P rounding in d_flash_prefill's
     * tensor-core P.V (T4) — the numeric regime the real model runs in. */
    std::vector<float> Q = gen_bf16((size_t)seq_q*nh*HD, 1.0f);
    std::vector<float> K = gen_bf16((size_t)nkv*seq_kv*HD, 1.0f);
    std::vector<float> V = gen_bf16((size_t)nkv*seq_kv*HD, 1.0f);
    const unsigned gqa = nh/nkv;
    /* CPU ref: causal (+ optional sliding) attention. Q token-major, K/V head-major. */
    std::vector<float> Oref((size_t)seq_q*nh*HD, 0.f);
    for (unsigned q=0;q<seq_q;q++) for (unsigned h=0;h<nh;h++){
        const unsigned qabs = q_pos0+q;
        const unsigned hkv=h/gqa; std::vector<float> s(seq_kv); float mx=-INFINITY;
        for (unsigned kv=0;kv<seq_kv;kv++){
            bool masked = kv>qabs; if (window) masked |= (qabs-kv >= window);
            if (masked){ s[kv]=-INFINITY; continue; }
            double dot=0; for (unsigned d=0;d<HD;d++)
                dot += (double)Q[((size_t)q*nh+h)*HD+d]*K[((size_t)hkv*seq_kv+kv)*HD+d];
            s[kv]=(float)dot*scale; if (s[kv]>mx) mx=s[kv]; }
        /* Masked entries were flagged -INFINITY (excluded from the max). Convert them to 0 HERE so
         * the O-accumulation `==0.f` skip below actually drops them: leaving them -INFINITY made
         * `(-inf/sum)*V` poison Oref with inf, which collapsed report()'s num/den to a spurious
         * relL2=0 (maxabs=inf) — a DEGENERATE ref that PASSED any kernel, even the wave64 negctrl.
         * With the fix the flash-prefill oracle actually charges the kernel's softmax + bf16-P P.V. */
        float sum=0; for (unsigned kv=0;kv<seq_kv;kv++){ if (s[kv]==-INFINITY){ s[kv]=0.f; continue; }
            s[kv]=expf(s[kv]-mx); sum+=s[kv]; }
        for (unsigned d=0;d<HD;d++){ double a=0; for (unsigned kv=0;kv<seq_kv;kv++){
            if (s[kv]==0.f) continue; a += (s[kv]/sum)*V[((size_t)hkv*seq_kv+kv)*HD+d]; }
            Oref[((size_t)q*nh+h)*HD+d]=(float)a; }
    }
    bf16 *dQ=to_dev(Q), *dK=to_dev(K), *dV=to_dev(V);
    bf16 *dO = (nsplit==1) ? dev_bf((size_t)seq_q*nh*HD) : nullptr;
    const size_t smem = (size_t)FA_PRE_SMEM_FLOATS(HD,BQ,BKV)*sizeof(float);
    CK(cudaFuncSetAttribute(k_flash_prefill<HD,BQ,BKV>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    std::vector<float> got;
    if (nsplit==1) {
        k_flash_prefill<HD,BQ,BKV><<<188,256,smem>>>(nullptr,nullptr,dO,dQ,dK,dV,seq_q,seq_kv,nh,nkv,q_pos0,window,1,scale);
        CK(cudaDeviceSynchronize());
        got = from_dev(dO,(size_t)seq_q*nh*HD);
    } else {
        float* dOp=dev_f((size_t)seq_q*nh*nsplit*HD); float* dMl=dev_f((size_t)seq_q*nh*nsplit*2);
        k_flash_prefill<HD,BQ,BKV><<<188,256,smem>>>(dOp,dMl,nullptr,dQ,dK,dV,seq_q,seq_kv,nh,nkv,q_pos0,window,nsplit,scale);
        CK(cudaDeviceSynchronize());
        bf16* dO2=dev_bf((size_t)seq_q*nh*HD);
        k_flash_merge<HD><<<seq_q*nh,256>>>(dO2,dOp,dMl,seq_q,nh,nsplit);
        CK(cudaDeviceSynchronize());
        got = from_dev(dO2,(size_t)seq_q*nh*HD);
        cudaFree(dOp);cudaFree(dMl);cudaFree(dO2);
    }
    report(label, Oref, got, true);
    cudaFree(dQ);cudaFree(dK);cudaFree(dV); if(dO) cudaFree(dO);
}

/* ==================== FLASH PREFILL VARLEN MUX (PX-1 stage 2) ============= */
/* Block-diagonal cross-request pack: R requests' chunks packed row-contiguous in Q, each with its
 * own seq-slot in a batch-major ring cache. Validated two ways:
 *   1. ORACLE: per-request T4-fixed f32 reference (same math as test_flash_prefill, chunked
 *      q_pos0 = kvlen-qlen) — relL2 within the flash tolerance;
 *   2. BIT-EXACT vs the stage-1 per-request-serial path (d_flash_prefill launched once per
 *      request with offset Q/O and slot-offset K/V bases) — the varlen work enumeration must
 *      reproduce the serial call byte-for-byte, including request boundaries mid-tile. */
template <int HD, int BQ, int BKV>
__global__ void k_flash_prefill_mux(const int* req, bf16* O, const bf16* Q, const bf16* K,
                                    const bf16* V, unsigned nh, unsigned nkv, unsigned win,
                                    unsigned kv_stride, float scale) {
    extern __shared__ float sm[];
    d_flash_prefill_mux<HD,BQ,BKV>(req, nullptr, nullptr, Q, K, V, O, /*seq_q*/0, /*seq_kv*/0,
                                   nh, nkv, /*q_pos0*/0, win, /*nsplit*/1, kv_stride,
                                   /*kv_mask*/0xFFFFFFFFu, scale, blockIdx.x, gridDim.x, sm);
}
template <int HD, int BQ, int BKV>
__global__ void k_flash_prefill_ctrl(bf16* O, const bf16* Q, const bf16* K, const bf16* V,
                                     unsigned sq, unsigned skv, unsigned nh, unsigned nkv,
                                     unsigned qpos0, unsigned win, unsigned kv_stride, float scale) {
    extern __shared__ float sm[];
    d_flash_prefill<HD,BQ,BKV>(nullptr, nullptr, Q, K, V, O, sq, skv, nh, nkv, qpos0, win, 1,
                               kv_stride, 0xFFFFFFFFu, scale, blockIdx.x, gridDim.x, sm);
}
template <int HD, int BQ, int BKV>
static void test_flash_prefill_varlen(const char* label, unsigned nh, unsigned nkv,
                                      const std::vector<int>& qlens,
                                      const std::vector<int>& kvlens,
                                      const std::vector<int>& slots, unsigned window,
                                      float scale) {
    const int R = (int)qlens.size();
    seed(0x7A00u + HD + nh*13 + (unsigned)R*29 + window + (unsigned)kvlens[0]);
    const unsigned gqa = nh/nkv;
    int nslot = 0, kvmax = 0, total = 0;
    for (int r=0;r<R;r++){ if (slots[r]+1>nslot) nslot=slots[r]+1;
        if (kvlens[r]>kvmax) kvmax=kvlens[r]; total += qlens[r]; }
    const unsigned kv_stride = (unsigned)((kvmax + 63) & ~63); /* no ring wrap; mask=full */

    /* Packed Q [total][nh][HD]; batch-major KV cache [nslot][nkv][kv_stride][HD] (slot ring),
     * request r's logical rows [0,kvlen_r) at the front of its slot's ring. */
    std::vector<float> Q = gen_bf16((size_t)total*nh*HD, 1.0f);
    std::vector<float> KC((size_t)nslot*nkv*kv_stride*HD, 0.f), VC(KC.size(), 0.f);
    std::vector<int> req; req.push_back(R);
    int cur = 0;
    for (int r=0;r<R;r++){
        std::vector<float> Kr = gen_bf16((size_t)nkv*kvlens[r]*HD, 1.0f);
        std::vector<float> Vr = gen_bf16((size_t)nkv*kvlens[r]*HD, 1.0f);
        for (unsigned hk=0;hk<nkv;hk++)
            for (int kv=0;kv<kvlens[r];kv++)
                for (int d=0;d<HD;d++){
                    const size_t base = (((size_t)slots[r]*nkv+hk)*kv_stride+kv)*HD+d;
                    KC[base]=Kr[((size_t)hk*kvlens[r]+kv)*HD+d];
                    VC[base]=Vr[((size_t)hk*kvlens[r]+kv)*HD+d];
                }
        req.push_back(cur); req.push_back(qlens[r]); req.push_back(slots[r]); req.push_back(kvlens[r]);
        cur += qlens[r];
    }
    /* Per-request f32 oracle into the packed Oref (chunked: q rows at abs pos kvlen-qlen..kvlen). */
    std::vector<float> Oref((size_t)total*nh*HD, 0.f);
    for (int r=0;r<R;r++){
        const int q0r = req[1+4*r], qlen = qlens[r], kvlen = kvlens[r];
        const unsigned qp0 = (unsigned)(kvlen - qlen);
        for (int q=0;q<qlen;q++) for (unsigned h=0;h<nh;h++){
            const unsigned qabs = qp0 + (unsigned)q, hkv = h/gqa;
            std::vector<float> s(kvlen); float mx=-INFINITY;
            for (int kv=0;kv<kvlen;kv++){
                bool masked = (unsigned)kv>qabs; if (window) masked |= (qabs-(unsigned)kv >= window);
                if (masked){ s[kv]=-INFINITY; continue; }
                double dot=0; for (int d=0;d<HD;d++)
                    dot += (double)Q[((size_t)(q0r+q)*nh+h)*HD+d]
                         * KC[(((size_t)slots[r]*nkv+hkv)*kv_stride+(unsigned)kv)*HD+d];
                s[kv]=(float)dot*scale; if (s[kv]>mx) mx=s[kv]; }
            float sum=0; for (int kv=0;kv<kvlen;kv++){ if (s[kv]==-INFINITY){ s[kv]=0.f; continue; }
                s[kv]=expf(s[kv]-mx); sum+=s[kv]; }
            for (int d=0;d<HD;d++){ double a=0; for (int kv=0;kv<kvlen;kv++){
                if (s[kv]==0.f) continue;
                a += (s[kv]/sum)*VC[(((size_t)slots[r]*nkv+hkv)*kv_stride+(unsigned)kv)*HD+d]; }
                Oref[((size_t)(q0r+q)*nh+h)*HD+d]=(float)a; }
        }
    }
    bf16 *dQ=to_dev(Q), *dK=to_dev(KC), *dV=to_dev(VC);
    bf16 *dOm=dev_bf((size_t)total*nh*HD), *dOs=dev_bf((size_t)total*nh*HD);
    int* dreq=to_dev_i(req);
    const size_t smem = (size_t)FA_PRE_SMEM_FLOATS(HD,BQ,BKV)*sizeof(float);
    CK(cudaFuncSetAttribute(k_flash_prefill_mux<HD,BQ,BKV>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(k_flash_prefill_ctrl<HD,BQ,BKV>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    /* Varlen (one launch, all requests). */
    k_flash_prefill_mux<HD,BQ,BKV><<<188,256,smem>>>(dreq,dOm,dQ,dK,dV,nh,nkv,window,kv_stride,scale);
    CK(cudaDeviceSynchronize());
    /* Stage-1 serial control (one launch per request, offset bases). */
    for (int r=0;r<R;r++){
        const size_t qoff = (size_t)req[1+4*r]*nh*HD;
        const size_t kvoff = (size_t)slots[r]*nkv*(size_t)kv_stride*HD;
        k_flash_prefill_ctrl<HD,BQ,BKV><<<188,256,smem>>>(dOs+qoff,dQ+qoff,dK+kvoff,dV+kvoff,
            (unsigned)qlens[r],(unsigned)kvlens[r],nh,nkv,(unsigned)(kvlens[r]-qlens[r]),
            window,kv_stride,scale);
    }
    CK(cudaDeviceSynchronize());
    std::vector<bf16> hm((size_t)total*nh*HD), hs(hm.size());
    CK(cudaMemcpy(hm.data(),dOm,hm.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(),dOs,hs.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
    const bool bitexact = memcmp(hm.data(),hs.data(),hm.size()*sizeof(bf16))==0;
    printf("  %-40s varlen-vs-serial %s\n", label, bitexact?"BIT-EXACT":"MISMATCH");
    if (!bitexact) g_fail = 1;
    report(label, Oref, from_dev(dOm,(size_t)total*nh*HD), true);
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dOm);cudaFree(dOs);cudaFree(dreq);
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
static void test_gemv_fp8(const char* label, unsigned N, unsigned K, unsigned M = 1) {
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
    k_gemv_fp8<<<64,256>>>(dC,dx,dW,dsc,M,N,K);
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

/* ==================== Gemma-4 26B-A4B bf16 MoE (decode) ================== */
static uint64_t* to_dev_u64(const std::vector<uint64_t>& h){ uint64_t* d=nullptr;
    CK(cudaMalloc(&d,h.size()*8)); CK(cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice)); return d; }

/* The f32 CPU golden for the router selection — byte-identical algorithm to the device
 * (softmax over probs, k-pass masked argmax with the packed-key LOWEST-ID tie-break,
 * norm_topk, per-expert scale). Fills id[k], gate[k]. */
static void router_ref(const std::vector<float>& resid, const std::vector<float>& proj,
                       const std::vector<float>& scale, const std::vector<float>& pes,
                       unsigned H, unsigned E, unsigned k, float root, float eps,
                       std::vector<unsigned>& id, std::vector<float>& gate) {
    double ss=0; for(unsigned h=0;h<H;h++) ss += (double)resid[h]*resid[h];
    float invrms = 1.0f/sqrtf((float)(ss/(double)H)+eps);
    std::vector<float> h2(H); for(unsigned h=0;h<H;h++) h2[h]=resid[h]*invrms*scale[h]*root;
    std::vector<float> sc(E);
    for(unsigned e=0;e<E;e++){ double acc=0; for(unsigned h=0;h<H;h++) acc += (double)h2[h]*proj[(size_t)e*H+h];
        sc[e]=(float)acc; }
    float m=-1e30f; for(unsigned e=0;e<E;e++) m=fmaxf(m,sc[e]);
    float s=0; for(unsigned e=0;e<E;e++){ sc[e]=expf(sc[e]-m); s+=sc[e]; }
    for(unsigned e=0;e<E;e++) sc[e]/=s;
    id.assign(k,0); gate.assign(k,0);
    for(unsigned j=0;j<k;j++){ unsigned long long best=0; unsigned bid=0;
        for(unsigned e=0;e<E;e++){ unsigned sb; float v=sc[e]; memcpy(&sb,&v,4);
            sb=(sb&0x80000000u)?~sb:(sb|0x80000000u);
            unsigned long long key=((unsigned long long)sb<<20)|(unsigned long long)((E-1u-e)&0xFFFFFu);
            if(key>best){best=key;bid=e;} }
        id[j]=bid; gate[j]=sc[bid]; sc[bid]=-1e30f; }
    float gs=0; for(unsigned j=0;j<k;j++) gs+=gate[j];
    for(unsigned j=0;j<k;j++){ if(gs!=0) gate[j]/=gs; gate[j]*=pes[id[j]]; }
}

__global__ void k_moe_router(unsigned char* table, const bf16* resid, const bf16* proj,
                             const bf16* scale, const bf16* pes, unsigned H, unsigned E,
                             unsigned k, float root, float eps, unsigned B) {
    extern __shared__ float arena[];
    d_moe_router_gemma(table, resid, proj, scale, pes, H, E, k, root, eps,
                       blockIdx.x, gridDim.x, B, arena);
}
__global__ void k_moe_router_score(float* score, const bf16* resid, const bf16* proj,
                                   const bf16* scale, unsigned H, unsigned E,
                                   float root, float eps, unsigned B) {
    d_moe_router_gemma_score(score, resid, proj, scale, H, E, root, eps,
                             blockIdx.x, gridDim.x, B);
}
__global__ void k_moe_router_score_fast(float* score, const bf16* resid, const bf16* proj,
                                        const bf16* scale, unsigned H, unsigned E,
                                        float root, float eps, unsigned B) {
    d_moe_router_gemma_score_fast(score, resid, proj, scale, H, E, root, eps,
                                  blockIdx.x, gridDim.x, B);
}
__global__ void k_moe_router_topk(unsigned char* table, const float* score, const bf16* pes,
                                  unsigned E, unsigned k, unsigned B) {
    extern __shared__ float arena[];
    d_moe_router_gemma_topk(table, score, pes, E, k, blockIdx.x, gridDim.x, B, arena);
}

/* Compare a [B][k] device routing table against the per-row f32 CPU golden. */
static void check_table(const char* label, const std::vector<unsigned char>& hT,
                        const std::vector<unsigned>& rid, const std::vector<float>& rgate,
                        unsigned B, unsigned k) {
    std::vector<float> gref((size_t)B*k), ggot((size_t)B*k); int idmis=0;
    for(unsigned s=0;s<B*k;s++){ unsigned gid=*(unsigned*)(hT.data()+(size_t)s*8);
        float gg=*(float*)(hT.data()+(size_t)s*8+4);
        if(gid!=rid[s]) idmis++; gref[s]=rgate[s]; ggot[s]=gg; }
    printf("  %-40s ids %s (%d mismatch)\n", label, idmis?"MISMATCH":"match", idmis);
    if(idmis) g_fail=1;
    report(label, gref, ggot, true, 1e-2);
}

/* B rows of residual; row r is scaled by (1 + 0.37*r) so the rows route DIFFERENTLY (the
 * softmax ordering is not scale-invariant once the weightless RMS renormalizes it). */
static std::vector<float> gen_rows(unsigned B, unsigned H) {
    std::vector<float> base=gen_bf16(H,1.0f), all((size_t)B*H);
    for(unsigned r=0;r<B;r++)
        for(unsigned h=0;h<H;h++) all[(size_t)r*H+h]=bf16_rt(base[h]*(1.0f+0.37f*(float)r));
    return all;
}

static void test_moe_router_split(const char* label, unsigned H, unsigned E, unsigned k,
                                  bool fast=false, unsigned B=1) {
    seed(0x6700u + H + E*7 + k);
    const float eps=1e-6f, root=1.0f/sqrtf((float)H);
    std::vector<float> resid=gen_rows(B,H), proj=gen_bf16((size_t)E*H,1.0f),
                       scale=gen_bf16(H,1.0f), pes=gen_bf16(E,1.0f);
    std::vector<unsigned> rid((size_t)B*k); std::vector<float> rgate((size_t)B*k);
    for(unsigned r=0;r<B;r++){
        std::vector<float> row(resid.begin()+(size_t)r*H, resid.begin()+(size_t)(r+1)*H);
        std::vector<unsigned> id1; std::vector<float> g1;
        router_ref(row,proj,scale,pes,H,E,k,root,eps,id1,g1);
        for(unsigned j=0;j<k;j++){ rid[(size_t)r*k+j]=id1[j]; rgate[(size_t)r*k+j]=g1[j]; } }
    bf16 *dR=to_dev(resid),*dP=to_dev(proj),*dS=to_dev(scale),*dE=to_dev(pes);
    float* dScore=nullptr; CK(cudaMalloc(&dScore,(size_t)B*E*sizeof(float)));
    unsigned char* dT=nullptr; CK(cudaMalloc(&dT,(size_t)B*k*8)); CK(cudaMemset(dT,0,(size_t)B*k*8));
    const unsigned blocks=(B*E+7)/8;
    if(fast) k_moe_router_score_fast<<<blocks,256>>>(dScore,dR,dP,dS,H,E,root,eps,B);
    else     k_moe_router_score<<<blocks,256>>>(dScore,dR,dP,dS,H,E,root,eps,B);
    k_moe_router_topk<<<B,256,(size_t)E*sizeof(float)>>>(dT,dScore,dE,E,k,B);
    CK(cudaDeviceSynchronize());
    std::vector<unsigned char> hT((size_t)B*k*8);
    CK(cudaMemcpy(hT.data(),dT,(size_t)B*k*8,cudaMemcpyDeviceToHost));
    check_table(label,hT,rid,rgate,B,k);
    cudaFree(dR);cudaFree(dP);cudaFree(dS);cudaFree(dE);cudaFree(dScore);cudaFree(dT);
}
static void test_moe_router(const char* label, unsigned H, unsigned E, unsigned k, unsigned B=1) {
    seed(0x6100u + H + E*7 + k);
    const float eps=1e-6f, root=1.0f/sqrtf((float)H);
    std::vector<float> resid=gen_rows(B,H), proj=gen_bf16((size_t)E*H,1.0f),
                       scale=gen_bf16(H,1.0f), pes=gen_bf16(E,1.0f);
    std::vector<unsigned> rid((size_t)B*k); std::vector<float> rgate((size_t)B*k);
    for(unsigned r=0;r<B;r++){
        std::vector<float> row(resid.begin()+(size_t)r*H, resid.begin()+(size_t)(r+1)*H);
        std::vector<unsigned> id1; std::vector<float> g1;
        router_ref(row,proj,scale,pes,H,E,k,root,eps,id1,g1);
        for(unsigned j=0;j<k;j++){ rid[(size_t)r*k+j]=id1[j]; rgate[(size_t)r*k+j]=g1[j]; } }
    bf16 *dR=to_dev(resid),*dP=to_dev(proj),*dS=to_dev(scale),*dE=to_dev(pes);
    unsigned char* dT=nullptr; CK(cudaMalloc(&dT,(size_t)B*k*8)); CK(cudaMemset(dT,0,(size_t)B*k*8));
    const size_t smem=(size_t)(H+E)*sizeof(float);
    CK(cudaFuncSetAttribute(k_moe_router,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_moe_router<<<1,256,smem>>>(dT,dR,dP,dS,dE,H,E,k,root,eps,B);
    CK(cudaDeviceSynchronize());
    std::vector<unsigned char> hT((size_t)B*k*8);
    CK(cudaMemcpy(hT.data(),dT,(size_t)B*k*8,cudaMemcpyDeviceToHost));
    check_table(label,hT,rid,rgate,B,k);
    cudaFree(dR);cudaFree(dP);cudaFree(dS);cudaFree(dE);cudaFree(dT);
}
/* Deliberate TIE: 8 experts (ids 0..7) carry an identical strong logit, ids 8..15 carry 0.
 * top-k must select the k LOWEST ids of the tied block. resid=scale=1 => h2 is a positive
 * constant, so logit[e] = const * rowsum(proj[e]); we set rowsum 100 for e<8, 0 otherwise. */
static void test_moe_router_tie(const char* label, unsigned H, unsigned k, unsigned B=1) {
    const unsigned E=16; const float eps=1e-6f, root=1.0f/sqrtf((float)H);
    std::vector<float> resid((size_t)B*H,1.0f), scale(H,1.0f), pes(E,1.0f), proj((size_t)E*H,0.0f);
    for(unsigned e=0;e<8;e++) proj[(size_t)e*H+0]=100.0f; /* rowsum 100, all 8 identical */
    std::vector<unsigned> rid; std::vector<float> rgate;
    std::vector<float> row(resid.begin(),resid.begin()+H);
    router_ref(row,proj,scale,pes,H,E,k,root,eps,rid,rgate);
    bf16 *dR=to_dev(resid),*dP=to_dev(proj),*dS=to_dev(scale),*dE=to_dev(pes);
    unsigned char* dT=nullptr; CK(cudaMalloc(&dT,(size_t)B*k*8)); CK(cudaMemset(dT,0,(size_t)B*k*8));
    const size_t smem=(size_t)(H+E)*sizeof(float);
    CK(cudaFuncSetAttribute(k_moe_router,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_moe_router<<<1,256,smem>>>(dT,dR,dP,dS,dE,H,E,k,root,eps,B);
    CK(cudaDeviceSynchronize());
    std::vector<unsigned char> hT((size_t)B*k*8);
    CK(cudaMemcpy(hT.data(),dT,(size_t)B*k*8,cudaMemcpyDeviceToHost));
    /* EVERY row must pick exactly 0,1,..,k-1 (the k lowest tied ids). */
    int bad=0; for(unsigned r=0;r<B;r++) for(unsigned j=0;j<k;j++){
        unsigned gid=*(unsigned*)(hT.data()+(size_t)(r*k+j)*8);
        if(gid!=j || rid[j]!=j) bad++; }
    printf("  %-40s selected ids=%u,%u,%u,%u %s\n", label,
           *(unsigned*)(hT.data()+0), *(unsigned*)(hT.data()+8),
           *(unsigned*)(hT.data()+16), *(unsigned*)(hT.data()+24), bad?"FAIL":"PASS");
    if(bad) g_fail=1;
    cudaFree(dR);cudaFree(dP);cudaFree(dS);cudaFree(dE);cudaFree(dT);
}

static void test_moe_router_split_tie(const char* label, unsigned H, unsigned k,
                                      bool fast=false, unsigned B=1) {
    const unsigned E=16; const float eps=1e-6f, root=1.0f/sqrtf((float)H);
    std::vector<float> resid((size_t)B*H,1.0f), scale(H,1.0f), pes(E,1.0f), proj((size_t)E*H,0.0f);
    for(unsigned e=0;e<8;e++) proj[(size_t)e*H]=100.0f;
    bf16 *dR=to_dev(resid),*dP=to_dev(proj),*dS=to_dev(scale),*dE=to_dev(pes);
    float* dScore=nullptr; CK(cudaMalloc(&dScore,(size_t)B*E*sizeof(float)));
    unsigned char* dT=nullptr; CK(cudaMalloc(&dT,(size_t)B*k*8)); CK(cudaMemset(dT,0,(size_t)B*k*8));
    if(fast) k_moe_router_score_fast<<<2*B,256>>>(dScore,dR,dP,dS,H,E,root,eps,B);
    else     k_moe_router_score<<<2*B,256>>>(dScore,dR,dP,dS,H,E,root,eps,B);
    k_moe_router_topk<<<B,256,(size_t)E*sizeof(float)>>>(dT,dScore,dE,E,k,B);
    CK(cudaDeviceSynchronize());
    std::vector<unsigned char> hT((size_t)B*k*8);
    CK(cudaMemcpy(hT.data(),dT,(size_t)B*k*8,cudaMemcpyDeviceToHost));
    int bad=0; for(unsigned s=0;s<B*k;s++) if(*(unsigned*)(hT.data()+(size_t)s*8)!=(s%k)) bad++;
    printf("  %-40s selected ids=%u,%u,%u,%u %s\n", label,
           *(unsigned*)(hT.data()+0), *(unsigned*)(hT.data()+8),
           *(unsigned*)(hT.data()+16), *(unsigned*)(hT.data()+24), bad?"FAIL":"PASS");
    if(bad) g_fail=1;
    cudaFree(dR);cudaFree(dP);cudaFree(dS);cudaFree(dE);cudaFree(dScore);cudaFree(dT);
}

/* Build a routing table (host) selecting `sel` experts with gates `g`. */
static unsigned char* make_table(const std::vector<unsigned>& sel, const std::vector<float>& g){
    std::vector<unsigned char> t(sel.size()*8);
    for(size_t j=0;j<sel.size();j++){ *(unsigned*)(t.data()+j*8)=sel[j]; *(float*)(t.data()+j*8+4)=g[j]; }
    unsigned char* d=nullptr; CK(cudaMalloc(&d,t.size())); CK(cudaMemcpy(d,t.data(),t.size(),cudaMemcpyHostToDevice));
    return d;
}
/* B*k routing slots. share=1: every row routes to the SAME expert set (maximum weight reuse —
 * the case the channel-major sweep exists for). share=0: the rows route to DISJOINT experts
 * (needs B*k <= E), the worst case for reuse and the one that catches a row/slot index mixup. */
static std::vector<unsigned> gen_sel(unsigned B, unsigned k, unsigned E, int share, unsigned salt){
    std::vector<unsigned> sel((size_t)B*k);
    for(unsigned r=0;r<B;r++) for(unsigned j=0;j<k;j++)
        sel[(size_t)r*k+j] = share ? ((j*13u+salt)%E) : ((r*k+j)%E);
    return sel;
}
__global__ void k_moe_glu(bf16* fu, const bf16* x, const unsigned char* tab, const uint64_t* ewt,
                          unsigned k, unsigned I, unsigned H, unsigned E, unsigned B){
    d_moe_expert_glu_gemma(fu,x,tab,(const unsigned long long*)ewt,k,I,H,E,blockIdx.x,gridDim.x,B);
}
static void test_moe_glu(const char* label, unsigned E, unsigned k, unsigned I, unsigned H,
                         unsigned B=1, int share=1){
    seed(0x6200u+E+k+I+H);
    std::vector<unsigned> sel=gen_sel(B,k,E,share,1u); std::vector<float> g((size_t)B*k);
    for(size_t s=0;s<g.size();s++) g[s]=0.1f*(float)(s%k+1);
    std::vector<float> x=gen_rows(B,H);
    std::vector<float> gu=gen_bf16((size_t)E*2*I*H,1.0f);          /* [E,2I,H] fused gate_up */
    bf16* dGU=to_dev(gu); bf16* dx=to_dev(x);
    std::vector<uint64_t> ewt((size_t)E*2,0);
    for(unsigned e=0;e<E;e++) ewt[(size_t)e*2+0]=(uint64_t)(dGU+(size_t)e*2*I*H);
    uint64_t* dEwt=to_dev_u64(ewt);
    unsigned char* dTab=make_table(sel,g);
    bf16* dFu=dev_bf((size_t)B*k*I);
    auto gelu=[&](float v){ float c=0.7978845608028654f*(v+0.044715f*v*v*v); return 0.5f*v*(1.0f+tanhf(c)); };
    std::vector<float> ref((size_t)B*k*I);
    for(unsigned s=0;s<B*k;s++){ const float* gub=&gu[(size_t)sel[s]*2*I*H];
        const float* xr=&x[(size_t)(s/k)*H];
        for(unsigned n=0;n<I;n++){ double ag=0,au=0;
            for(unsigned h=0;h<H;h++){ ag+=(double)xr[h]*gub[(size_t)n*H+h];
                                       au+=(double)xr[h]*gub[(size_t)(I+n)*H+h]; }
            ref[(size_t)s*I+n]=bf16_rt(gelu((float)ag)*(float)au); } }
    k_moe_glu<<<188,256>>>(dFu,dx,dTab,dEwt,k,I,H,E,B);
    CK(cudaDeviceSynchronize());
    report(label, ref, from_dev(dFu,(size_t)B*k*I), true, 2e-2);
    cudaFree(dGU);cudaFree(dx);cudaFree(dEwt);cudaFree(dTab);cudaFree(dFu);
}

/* op71: fused pre-FFN-norm-2 + expert GLU — the SHIPPING bf16 decode arm. x is derived inside
 * the kernel as resid[r]*rsqrt(mean(resid[r]^2)+eps)*gamma, one RMS scalar per batched row. */
__global__ void k_moe_glu_norm(bf16* fu, const bf16* resid, const bf16* gamma,
                               const unsigned char* tab, const uint64_t* ewt, unsigned k,
                               unsigned I, unsigned H, unsigned E, float eps, unsigned B){
    d_moe_expert_glu_norm_gemma(fu,resid,gamma,tab,(const unsigned long long*)ewt,k,I,H,E,eps,
                                blockIdx.x,gridDim.x,B);
}
static void test_moe_glu_norm(const char* label, unsigned E, unsigned k, unsigned I, unsigned H,
                              float eps, unsigned B=1, int share=1){
    seed(0x7100u+E+k+I+H);
    std::vector<unsigned> sel=gen_sel(B,k,E,share,3u); std::vector<float> g((size_t)B*k);
    for(size_t s=0;s<g.size();s++) g[s]=0.1f*(float)(s%k+1);
    std::vector<float> resid=gen_rows(B,H), gamma=gen_bf16(H,1.0f);
    std::vector<float> gu=gen_bf16((size_t)E*2*I*H,1.0f);
    bf16* dGU=to_dev(gu); bf16* dR=to_dev(resid); bf16* dG=to_dev(gamma);
    std::vector<uint64_t> ewt((size_t)E*2,0);
    for(unsigned e=0;e<E;e++) ewt[(size_t)e*2+0]=(uint64_t)(dGU+(size_t)e*2*I*H);
    uint64_t* dEwt=to_dev_u64(ewt);
    unsigned char* dTab=make_table(sel,g);
    bf16* dFu=dev_bf((size_t)B*k*I);
    auto gelu=[&](float v){ float c=0.7978845608028654f*(v+0.044715f*v*v*v); return 0.5f*v*(1.0f+tanhf(c)); };
    /* golden: per-row RMS then the same dots as op62 */
    std::vector<float> xn((size_t)B*H);
    for(unsigned r=0;r<B;r++){ double ss=0;
        for(unsigned h=0;h<H;h++){ float v=resid[(size_t)r*H+h]; ss+=(double)v*v; }
        float inv=1.f/sqrtf((float)(ss/(double)H)+eps);
        for(unsigned h=0;h<H;h++) xn[(size_t)r*H+h]=resid[(size_t)r*H+h]*inv*gamma[h]; }
    std::vector<float> ref((size_t)B*k*I);
    for(unsigned s=0;s<B*k;s++){ const float* gub=&gu[(size_t)sel[s]*2*I*H];
        const float* xr=&xn[(size_t)(s/k)*H];
        for(unsigned n=0;n<I;n++){ double ag=0,au=0;
            for(unsigned h=0;h<H;h++){ ag+=(double)xr[h]*gub[(size_t)n*H+h];
                                       au+=(double)xr[h]*gub[(size_t)(I+n)*H+h]; }
            ref[(size_t)s*I+n]=bf16_rt(gelu((float)ag)*(float)au); } }
    k_moe_glu_norm<<<188,256>>>(dFu,dR,dG,dTab,dEwt,k,I,H,E,eps,B);
    CK(cudaDeviceSynchronize());
    report(label, ref, from_dev(dFu,(size_t)B*k*I), true, 2e-2);
    cudaFree(dGU);cudaFree(dR);cudaFree(dG);cudaFree(dEwt);cudaFree(dTab);cudaFree(dFu);
}
__global__ void k_moe_down(float* part, const bf16* fu, const unsigned char* tab, const uint64_t* ewt,
                           unsigned k, unsigned H, unsigned I, unsigned E, unsigned B){
    d_moe_expert_down_gemma(part,fu,tab,(const unsigned long long*)ewt,k,H,I,E,blockIdx.x,gridDim.x,B);
}
static void test_moe_down(const char* label, unsigned E, unsigned k, unsigned I, unsigned H,
                          unsigned B=1, int share=1){
    seed(0x6300u+E+k+I+H);
    std::vector<unsigned> sel=gen_sel(B,k,E,share,2u); std::vector<float> g((size_t)B*k);
    for(size_t s=0;s<g.size();s++) g[s]=0.25f*(float)(s%k+1);
    std::vector<float> fu=gen_bf16((size_t)B*k*I,1.0f);
    std::vector<float> dw=gen_bf16((size_t)E*H*I,1.0f);            /* [E,H,I] down */
    bf16* dDW=to_dev(dw); bf16* dFu=to_dev(fu);
    std::vector<uint64_t> ewt((size_t)E*2,0);
    for(unsigned e=0;e<E;e++) ewt[(size_t)e*2+1]=(uint64_t)(dDW+(size_t)e*H*I);
    uint64_t* dEwt=to_dev_u64(ewt);
    unsigned char* dTab=make_table(sel,g);
    float* dPart=dev_f((size_t)B*k*H);
    std::vector<float> ref((size_t)B*k*H);
    for(unsigned s=0;s<B*k;s++){ const float* wd=&dw[(size_t)sel[s]*H*I];
        for(unsigned h=0;h<H;h++){ double a=0;
            for(unsigned i=0;i<I;i++) a += (double)fu[(size_t)s*I+i]*wd[(size_t)h*I+i];
            ref[(size_t)s*H+h]=g[s]*(float)a; } }
    k_moe_down<<<188,256>>>(dPart,dFu,dTab,dEwt,k,H,I,E,B);
    CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)B*k*H);
    CK(cudaMemcpy(got.data(),dPart,(size_t)B*k*H*4,cudaMemcpyDeviceToHost));
    report(label, ref, got, true, 2e-2);
    cudaFree(dDW);cudaFree(dFu);cudaFree(dEwt);cudaFree(dTab);cudaFree(dPart);
}

/* Per-row e4m3 oracle for the Gemma fused expert tensors. The reference consumes the SAME
 * dequantized e4m3 values as the device, isolating kernel indexing/scaling/GELU/gate semantics
 * from the expected model-level quantization error. */
static void quant_rows(const std::vector<float>& w, unsigned N, unsigned K,
                       std::vector<uint8_t>& q, std::vector<float>& sc) {
    q.resize((size_t)N*K); sc.resize(N);
    for (unsigned n=0;n<N;n++) {
        float amax=0; for(unsigned x=0;x<K;x++) amax=fmaxf(amax,fabsf(w[(size_t)n*K+x]));
        sc[n]=amax>0.0f?amax/448.0f:1.0f;
        for(unsigned x=0;x<K;x++) q[(size_t)n*K+x]=e4m3_enc(w[(size_t)n*K+x]/sc[n]);
    }
}
__global__ void k_moe_glu_fp8(bf16* fu, const bf16* x, const unsigned char* tab,
                              const uint64_t* ewt, const uint64_t* est,
                              unsigned k, unsigned I, unsigned H, unsigned E, unsigned B) {
    d_moe_expert_glu_gemma_fp8(fu,x,tab,(const unsigned long long*)ewt,
                               (const unsigned long long*)est,k,I,H,E,blockIdx.x,gridDim.x,B);
}
/* B>1: per-row x, [B][k] routing table, [B][k][I] fu; the flat sweep is channel-major. share=1
 * = every row routes to the same expert set (weight reuse), share=0 = disjoint (catches a
 * row/slot index mixup — a row-1-unwritten bug lands as huge relL2 on those outputs). */
static void test_moe_glu_fp8(const char* label, unsigned E, unsigned k, unsigned I, unsigned H,
                             unsigned B=1, int share=1) {
    seed(0x6500u+E+k+I+H);
    std::vector<unsigned> sel=gen_sel(B,k,E,share,1u); std::vector<float> gate((size_t)B*k);
    for(size_t s=0;s<gate.size();s++) gate[s]=0.1f*(float)(s%k+1);
    std::vector<float> x=gen_rows(B,H), gu=gen_bf16((size_t)E*2*I*H,1.0f);
    std::vector<uint8_t> gu8; std::vector<float> gs;
    quant_rows(gu,E*2*I,H,gu8,gs);
    uint8_t* dGU=nullptr; float* dGS=to_dev_f(gs); bf16* dx=to_dev(x);
    CK(cudaMalloc(&dGU,gu8.size())); CK(cudaMemcpy(dGU,gu8.data(),gu8.size(),cudaMemcpyHostToDevice));
    std::vector<uint64_t> ewt((size_t)E*2,0),est((size_t)E*2,0);
    for(unsigned e=0;e<E;e++){ ewt[(size_t)e*2]=(uint64_t)(dGU+(size_t)e*2*I*H);
                              est[(size_t)e*2]=(uint64_t)(dGS+(size_t)e*2*I); }
    uint64_t *dEwt=to_dev_u64(ewt),*dEst=to_dev_u64(est);
    unsigned char* dTab=make_table(sel,gate); bf16* dFu=dev_bf((size_t)B*k*I);
    auto gelu=[](float v){ float c=0.7978845608028654f*(v+0.044715f*v*v*v);
                           return 0.5f*v*(1.0f+tanhf(c)); };
    std::vector<float> ref((size_t)B*k*I);
    for(unsigned s=0;s<B*k;s++){ const float* xr=&x[(size_t)(s/k)*H];
        for(unsigned n=0;n<I;n++) {
        size_t er=(size_t)sel[s]*2*I; double ag=0,au=0;
        for(unsigned h=0;h<H;h++) { ag+=(double)xr[h]*e4m3_dec(gu8[(er+n)*H+h]);
                                    au+=(double)xr[h]*e4m3_dec(gu8[(er+I+n)*H+h]); }
        float g=(float)(ag*gs[er+n]),u=(float)(au*gs[er+I+n]);
        ref[(size_t)s*I+n]=bf16_rt(gelu(g)*u);
    } }
    k_moe_glu_fp8<<<188,256>>>(dFu,dx,dTab,dEwt,dEst,k,I,H,E,B); CK(cudaDeviceSynchronize());
    report(label,ref,from_dev(dFu,(size_t)B*k*I),true,2e-2);
    cudaFree(dGU);cudaFree(dGS);cudaFree(dx);cudaFree(dEwt);cudaFree(dEst);cudaFree(dTab);cudaFree(dFu);
}
__global__ void k_moe_down_fp8(float* part, const bf16* fu, const unsigned char* tab,
                               const uint64_t* ewt, const uint64_t* est,
                               unsigned k, unsigned H, unsigned I, unsigned E, unsigned B) {
    d_moe_expert_down_gemma_fp8(part,fu,tab,(const unsigned long long*)ewt,
                                (const unsigned long long*)est,k,H,I,E,blockIdx.x,gridDim.x,B);
}
static void test_moe_down_fp8(const char* label, unsigned E, unsigned k, unsigned I, unsigned H,
                              unsigned B=1, int share=1) {
    seed(0x6600u+E+k+I+H);
    std::vector<unsigned> sel=gen_sel(B,k,E,share,2u); std::vector<float> gate((size_t)B*k);
    for(size_t s=0;s<gate.size();s++) gate[s]=0.25f*(float)(s%k+1);
    std::vector<float> fu=gen_bf16((size_t)B*k*I,1.0f),dw=gen_bf16((size_t)E*H*I,1.0f);
    std::vector<uint8_t> dw8; std::vector<float> ds; quant_rows(dw,E*H,I,dw8,ds);
    uint8_t* dDW=nullptr; float* dDS=to_dev_f(ds); bf16* dFu=to_dev(fu);
    CK(cudaMalloc(&dDW,dw8.size())); CK(cudaMemcpy(dDW,dw8.data(),dw8.size(),cudaMemcpyHostToDevice));
    std::vector<uint64_t> ewt((size_t)E*2,0),est((size_t)E*2,0);
    for(unsigned e=0;e<E;e++){ ewt[(size_t)e*2+1]=(uint64_t)(dDW+(size_t)e*H*I);
                              est[(size_t)e*2+1]=(uint64_t)(dDS+(size_t)e*H); }
    uint64_t *dEwt=to_dev_u64(ewt),*dEst=to_dev_u64(est);
    unsigned char* dTab=make_table(sel,gate); float* dPart=dev_f((size_t)B*k*H);
    std::vector<float> ref((size_t)B*k*H);
    for(unsigned s=0;s<B*k;s++) for(unsigned h=0;h<H;h++) { size_t er=((size_t)sel[s]*H+h);
        double a=0; for(unsigned i=0;i<I;i++) a+=(double)fu[(size_t)s*I+i]*e4m3_dec(dw8[er*I+i]);
        ref[(size_t)s*H+h]=gate[s]*(float)(a*ds[er]); }
    k_moe_down_fp8<<<188,256>>>(dPart,dFu,dTab,dEwt,dEst,k,H,I,E,B); CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)B*k*H); CK(cudaMemcpy(got.data(),dPart,got.size()*4,cudaMemcpyDeviceToHost));
    report(label,ref,got,true,2e-2);
    cudaFree(dDW);cudaFree(dDS);cudaFree(dFu);cudaFree(dEwt);cudaFree(dEst);cudaFree(dTab);cudaFree(dPart);
}
__global__ void k_moe_combine(bf16* moe, const float* part, unsigned H, unsigned k){
    d_moe_combine_gemma(moe,part,H,k,blockIdx.x,gridDim.x);
}
static void test_moe_combine(const char* label, unsigned H, unsigned k){
    seed(0x6400u+H+k);
    std::vector<float> part((size_t)k*H); for(size_t i=0;i<part.size();i++) part[i]=rnd();
    std::vector<float> ref(H,0.f);
    for(unsigned h=0;h<H;h++){ float a=0; for(unsigned j=0;j<k;j++) a+=part[(size_t)j*H+h]; ref[h]=bf16_rt(a); }
    float* dPart=to_dev_f(part); bf16* dMoe=dev_bf(H);
    k_moe_combine<<<64,256>>>(dMoe,dPart,H,k);
    CK(cudaDeviceSynchronize());
    report(label, ref, from_dev(dMoe,H), true, 2e-2);
    cudaFree(dPart);cudaFree(dMoe);
}

__global__ void k_moe_combine_norm(bf16* out, const float* part, const bf16* resid,
                                   const bf16* gamma, unsigned H, unsigned k, float eps,
                                   unsigned B){
    extern __shared__ float arena_cn[];
    d_moe_combine_norm_gemma(out,part,resid,gamma,H,k,eps,blockIdx.x,gridDim.x,B,arena_cn);
}
static void test_moe_combine_norm(const char* label, unsigned H, unsigned k, float eps,
                                  unsigned B=1){
    seed(0x6500u+H+k);
    std::vector<float> part((size_t)B*k*H); for(size_t i=0;i<part.size();i++) part[i]=rnd()*0.1f;
    std::vector<float> resid((size_t)B*H); for(size_t i=0;i<resid.size();i++) resid[i]=bf16_rt(rnd()*0.5f);
    std::vector<float> gamma(H); for(size_t i=0;i<gamma.size();i++) gamma[i]=bf16_rt(0.5f+rnd()*0.5f);
    /* reference: per row, combine -> rmsnorm -> + resid (gamma/resid are bf16 on device) */
    std::vector<float> ref((size_t)B*H);
    for(unsigned r=0;r<B;r++){
        const float* pt=&part[(size_t)r*k*H];
        std::vector<float> sum(H,0.f);
        for(unsigned h=0;h<H;h++){ float a=0; for(unsigned j=0;j<k;j++) a+=pt[(size_t)j*H+h]; sum[h]=a; }
        double ss=0; for(unsigned h=0;h<H;h++) ss+=(double)sum[h]*(double)sum[h];
        float inv=1.f/sqrtf((float)(ss/(double)H)+eps);
        for(unsigned h=0;h<H;h++)
            ref[(size_t)r*H+h]=bf16_rt(sum[h]*inv*gamma[h]+resid[(size_t)r*H+h]); }
    float* dPart=to_dev_f(part); bf16* dResid=to_dev(resid); bf16* dGamma=to_dev(gamma);
    bf16* dOut=dev_bf((size_t)B*H);
    k_moe_combine_norm<<<B,256,H*sizeof(float)>>>(dOut,dPart,dResid,dGamma,H,k,eps,B);
    CK(cudaDeviceSynchronize());
    report(label, ref, from_dev(dOut,(size_t)B*H), true, 5e-3);
    cudaFree(dPart);cudaFree(dResid);cudaFree(dGamma);cudaFree(dOut);
}

/* op72: fused MoE tail — combine + post_ffn norm + sandwich residual + next input norm.
 * Golden composes the op70 reference with the NormResidualNorm reference, with the SAME
 * bf16 rounding points the pair had (b and the new residual rounded before re-reduction). */
__global__ void k_moe_comb_resid_norm(bf16* hn, bf16* x, const float* part, const bf16* h1,
                                      const bf16* gpf2, const bf16* gpo, const bf16* gn,
                                      unsigned H, unsigned k, float eps, float ls){
    extern __shared__ float arena_crn[];
    d_moe_combine_resid_norm_gemma(hn,x,part,h1,gpf2,gpo,gn,H,k,eps,ls,blockIdx.x,arena_crn);
}
static void test_moe_comb_resid_norm(const char* label, unsigned H, unsigned k, float eps, float ls){
    seed(0x7200u+H+k);
    std::vector<float> part((size_t)k*H); for(size_t i=0;i<part.size();i++) part[i]=rnd()*0.1f;
    std::vector<float> h1(H); for(size_t i=0;i<h1.size();i++) h1[i]=bf16_rt(rnd()*0.5f);
    std::vector<float> x0(H); for(size_t i=0;i<x0.size();i++) x0[i]=bf16_rt(rnd()*0.5f);
    std::vector<float> gpf2(H), gpo(H), gn(H);
    for(unsigned h=0;h<H;h++){ gpf2[h]=bf16_rt(0.5f+rnd()*0.5f); gpo[h]=bf16_rt(0.5f+rnd()*0.5f); gn[h]=bf16_rt(0.5f+rnd()*0.5f); }
    /* golden: comb -> inv1 -> b(rounded) -> inv2 -> r(rounded) -> inv3 -> hn */
    std::vector<float> comb(H);
    double ss=0;
    for(unsigned h=0;h<H;h++){ float a=0; for(unsigned j=0;j<k;j++) a+=part[(size_t)j*H+h]; comb[h]=a; ss+=(double)a*a; }
    float inv1=1.f/sqrtf((float)(ss/(double)H)+eps);
    std::vector<float> b(H); ss=0;
    for(unsigned h=0;h<H;h++){ b[h]=bf16_rt(comb[h]*inv1*gpf2[h]+h1[h]); ss+=(double)b[h]*b[h]; }
    float inv2=1.f/sqrtf((float)(ss/(double)H)+eps);
    std::vector<float> r(H); ss=0;
    for(unsigned h=0;h<H;h++){ r[h]=bf16_rt((x0[h]+b[h]*inv2*gpo[h])*ls); ss+=(double)r[h]*r[h]; }
    float inv3=1.f/sqrtf((float)(ss/(double)H)+eps);
    std::vector<float> ref_hn(H);
    for(unsigned h=0;h<H;h++) ref_hn[h]=bf16_rt(r[h]*inv3*gn[h]);
    float* dPart=to_dev_f(part); bf16* dH1=to_dev(h1); bf16* dX=to_dev(x0);
    bf16* dGpf2=to_dev(gpf2); bf16* dGpo=to_dev(gpo); bf16* dGn=to_dev(gn); bf16* dHn=dev_bf(H);
    k_moe_comb_resid_norm<<<1,256,H*sizeof(float)>>>(dHn,dX,dPart,dH1,dGpf2,dGpo,dGn,H,k,eps,ls);
    CK(cudaDeviceSynchronize());
    report(label, ref_hn, from_dev(dHn,H), true, 5e-3);
    char lx[96]; snprintf(lx,sizeof(lx),"%s (resid x)",label);
    report(lx, r, from_dev(dX,H), true, 5e-3);
    cudaFree(dPart);cudaFree(dH1);cudaFree(dX);cudaFree(dGpf2);cudaFree(dGpo);cudaFree(dGn);cudaFree(dHn);
}

/* ==================== PREFILL fp8 (w8a16) GEMM / GEMM_GLU (T6 L2) ============
 * Same dequantized-weight f32 reference as the fp8 GEMV: the weight is per-ROW e4m3-quantized
 * (scale=amax/448, the quantize_fp8.py convention) and dequantized back to float for the golden,
 * so only the kernel's own arithmetic (bf16 A, f32 mma-accumulate, per-channel scale factored into
 * the epilogue) is charged — not the e4m3 quantization error. e4m3->half->f32 on device is exact. */
__global__ void k_gemm_fp8(bf16* C, const bf16* A, const uint8_t* B, const float* sc,
                           unsigned m, unsigned n, unsigned k, unsigned a_row0) {
    extern __shared__ bf16 smg[];
    d_gemm_fp8(C, A, B, sc, m, n, k, a_row0, blockIdx.x, gridDim.x, smg);
}
static void test_gemm_fp8(const char* label, unsigned m, unsigned n, unsigned k, unsigned a_row0) {
    seed(0x5A00u + m*3 + n*5 + k + a_row0);
    std::vector<float> A = gen_bf16((size_t)(m+a_row0)*k, 1.0f);
    std::vector<float> B = gen_bf16((size_t)n*k, 1.0f);
    std::vector<uint8_t> B8((size_t)n*k);
    std::vector<float> sc(n), Cref((size_t)m*n);
    for (unsigned j=0;j<n;j++){
        float amax=0; for(unsigned kk=0;kk<k;kk++) amax=fmaxf(amax,fabsf(B[(size_t)j*k+kk]));
        float scale = amax>0.0f ? amax/448.0f : 1.0f; sc[j]=scale;
        for(unsigned kk=0;kk<k;kk++) B8[(size_t)j*k+kk]=e4m3_enc(B[(size_t)j*k+kk]/scale);
    }
    for (unsigned i=0;i<m;i++) for (unsigned j=0;j<n;j++){ double acc=0;
        for (unsigned kk=0;kk<k;kk++) acc += (double)A[(size_t)(a_row0+i)*k+kk]*(double)e4m3_dec(B8[(size_t)j*k+kk]);
        Cref[(size_t)i*n+j]=bf16_rt((float)(acc*(double)sc[j])); }
    bf16 *dA=to_dev(A), *dC=dev_bf((size_t)m*n); float* dsc=to_dev_f(sc);
    uint8_t* dB; CK(cudaMalloc(&dB,(size_t)n*k)); CK(cudaMemcpy(dB,B8.data(),(size_t)n*k,cudaMemcpyHostToDevice));
    const size_t smem = (size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_gemm_fp8,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_gemm_fp8<<<188,256,smem>>>(dC,dA,dB,dsc,m,n,k,a_row0);
    CK(cudaDeviceSynchronize());
    report(label, Cref, from_dev(dC,(size_t)m*n), true, 5e-3);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dsc);
}
__global__ void k_gemm_glu_fp8(bf16* C, const bf16* A, const uint8_t* Wg, const uint8_t* Wu,
                               const float* sg, const float* su, unsigned m, unsigned n,
                               unsigned k, unsigned act) {
    extern __shared__ bf16 smg[];
    d_gemm_glu_fp8(C, A, Wg, Wu, sg, su, m, n, k, act, blockIdx.x, gridDim.x, smg);
}
static void test_gemm_glu_fp8(const char* label, unsigned m, unsigned n, unsigned k, unsigned act) {
    seed(0x5F00u + m + n + k + act);
    std::vector<float> A = gen_bf16((size_t)m*k, 1.0f);
    std::vector<float> Wg = gen_bf16((size_t)n*k, 1.0f), Wu = gen_bf16((size_t)n*k, 1.0f);
    std::vector<uint8_t> Wg8((size_t)n*k), Wu8((size_t)n*k);
    std::vector<float> sg(n), su(n), Cref((size_t)m*n);
    auto quant=[&](const std::vector<float>& W, std::vector<uint8_t>& W8, std::vector<float>& sc){
        for (unsigned j=0;j<n;j++){ float amax=0; for(unsigned kk=0;kk<k;kk++) amax=fmaxf(amax,fabsf(W[(size_t)j*k+kk]));
            float scale=amax>0.0f?amax/448.0f:1.0f; sc[j]=scale;
            for(unsigned kk=0;kk<k;kk++) W8[(size_t)j*k+kk]=e4m3_enc(W[(size_t)j*k+kk]/scale); } };
    quant(Wg,Wg8,sg); quant(Wu,Wu8,su);
    auto gelu=[&](float x){ float c=0.7978845608028654f*(x+0.044715f*x*x*x); return 0.5f*x*(1.0f+tanhf(c)); };
    auto silu=[&](float x){ return x/(1.0f+expf(-x)); };
    for (unsigned i=0;i<m;i++) for (unsigned j=0;j<n;j++){ double ag=0,au=0;
        for (unsigned kk=0;kk<k;kk++){ ag += (double)A[(size_t)i*k+kk]*(double)e4m3_dec(Wg8[(size_t)j*k+kk]);
                                       au += (double)A[(size_t)i*k+kk]*(double)e4m3_dec(Wu8[(size_t)j*k+kk]); }
        float g=(float)(ag*(double)sg[j]), u=(float)(au*(double)su[j]);
        float a = act==PLOW_ACT_SILU_ ? silu(g) : gelu(g);
        Cref[(size_t)i*n+j]=bf16_rt(a*u); }
    bf16 *dA=to_dev(A), *dC=dev_bf((size_t)m*n);
    float* dsg=to_dev_f(sg); float* dsu=to_dev_f(su);
    uint8_t *dWg,*dWu; CK(cudaMalloc(&dWg,(size_t)n*k)); CK(cudaMalloc(&dWu,(size_t)n*k));
    CK(cudaMemcpy(dWg,Wg8.data(),(size_t)n*k,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWu,Wu8.data(),(size_t)n*k,cudaMemcpyHostToDevice));
    const size_t smem = (size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_gemm_glu_fp8,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_gemm_glu_fp8<<<188,256,smem>>>(dC,dA,dWg,dWu,dsg,dsu,m,n,k,act);
    CK(cudaDeviceSynchronize());
    report(label, Cref, from_dev(dC,(size_t)m*n), true, 1e-2);
    cudaFree(dA);cudaFree(dWg);cudaFree(dWu);cudaFree(dC);cudaFree(dsg);cudaFree(dsu);
}

/* ==================== PREFILL fp8 (w8a8) GEMM / GEMM_GLU / QUANT (T7 L2) =====================
 * Now BOTH operands are e4m3: the ACTIVATION is per-M-row quantized (a_scale=amax/448, the
 * d_quant_fp8 convention) as well as the per-channel weight. Two references are computed:
 *   GATE ref = both operands DEQUANTIZED back to f32 (kernel and ref see the SAME e4m3 values),
 *     so the gate charges ONLY the kernel's own arithmetic: e4m3*e4m3 products are EXACT in f32
 *     (e4m3<->f32 lossless), the m16n8k32 tensor core f32-accumulates them, and the store rounds
 *     to bf16. The only gap vs the dequant ref is the K-accumulation ORDER + the bf16 output round
 *     -> ~1e-3 relL2, the SAME band as the w8a16 gate (tol 5e-3). A fragment-map or epilogue error
 *     blows straight past that band. e4m3-aware tolerance HONESTLY derived: since the ref shares the
 *     kernel's quantized inputs, the e4m3 ROUNDING (max rel 2^-4=6.25%/elt) is NOT in the gate — it
 *     is characterized separately below, not gated (that error is a MODELING choice, not a bug).
 *   CHAR = relL2 vs the FULL-PRECISION bf16 matmul (unquantized), PRINTED to quantify the w8a8
 *     quantization divergence honestly (the campaign's numerics note); informational, not gated. */
static void quant_row_act(const std::vector<float>& x, std::vector<uint8_t>& xq,
                          std::vector<float>& as, unsigned M, unsigned K) {
    for (unsigned m=0;m<M;m++){ float amax=0; for(unsigned kk=0;kk<K;kk++) amax=fmaxf(amax,fabsf(x[(size_t)m*K+kk]));
        float s=fmaxf(amax/448.0f,1e-12f); as[m]=s;
        for(unsigned kk=0;kk<K;kk++) xq[(size_t)m*K+kk]=e4m3_enc(x[(size_t)m*K+kk]/s); }
}
static void quant_col_wt(const std::vector<float>& W, std::vector<uint8_t>& W8,
                         std::vector<float>& sc, unsigned N, unsigned K) {
    for (unsigned j=0;j<N;j++){ float amax=0; for(unsigned kk=0;kk<K;kk++) amax=fmaxf(amax,fabsf(W[(size_t)j*K+kk]));
        float s=amax>0.0f?amax/448.0f:1.0f; sc[j]=s;
        for(unsigned kk=0;kk<K;kk++) W8[(size_t)j*K+kk]=e4m3_enc(W[(size_t)j*K+kk]/s); }
}
__global__ void k_gemm_w8a8(bf16* C, const uint8_t* A, const uint8_t* B, const float* as,
                            const float* ws, unsigned m, unsigned n, unsigned k, unsigned a_row0) {
    extern __shared__ bf16 smw[];
    d_gemm_w8a8(C, A, B, as, ws, m, n, k, a_row0, blockIdx.x, gridDim.x, smw);
}
static void test_gemm_w8a8(const char* label, unsigned m, unsigned n, unsigned k, unsigned a_row0) {
    seed(0x7A00u + m*3 + n*5 + k + a_row0);
    std::vector<float> A = gen_bf16((size_t)(m+a_row0)*k, 1.0f);
    std::vector<float> B = gen_bf16((size_t)n*k, 1.0f);
    std::vector<uint8_t> A8((size_t)(m+a_row0)*k), B8((size_t)n*k);
    std::vector<float> as(m+a_row0), ws(n), Cref((size_t)m*n), Cfull((size_t)m*n);
    quant_row_act(A, A8, as, m+a_row0, k);
    quant_col_wt(B, B8, ws, n, k);
    for (unsigned i=0;i<m;i++) for (unsigned j=0;j<n;j++){ double acc=0,facc=0;
        for (unsigned kk=0;kk<k;kk++){
            acc  += (double)e4m3_dec(A8[(size_t)(a_row0+i)*k+kk])*(double)e4m3_dec(B8[(size_t)j*k+kk]);
            facc += (double)A[(size_t)(a_row0+i)*k+kk]*(double)B[(size_t)j*k+kk]; }
        Cref[(size_t)i*n+j]=bf16_rt((float)(acc*(double)as[a_row0+i]*(double)ws[j]));
        Cfull[(size_t)i*n+j]=(float)facc; }
    bf16* dC=dev_bf((size_t)m*n); float* das=to_dev_f(as); float* dws=to_dev_f(ws);
    uint8_t *dA,*dB; CK(cudaMalloc(&dA,(size_t)(m+a_row0)*k)); CK(cudaMalloc(&dB,(size_t)n*k));
    CK(cudaMemcpy(dA,A8.data(),(size_t)(m+a_row0)*k,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,B8.data(),(size_t)n*k,cudaMemcpyHostToDevice));
    const size_t smem=(size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_gemm_w8a8,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_gemm_w8a8<<<188,256,smem>>>(dC,dA,dB,das,dws,m,n,k,a_row0);
    CK(cudaDeviceSynchronize());
    std::vector<float> got=from_dev(dC,(size_t)m*n);
    /* CHAR: quantization divergence vs the full-precision matmul (printed, not gated). */
    double num=0,den=0; for(size_t i=0;i<got.size();i++){ double d=(double)got[i]-Cfull[i]; num+=d*d; den+=(double)Cfull[i]*Cfull[i]; }
    printf("  [w8a8 quant-err vs full-precision] relL2=%.4g\n", den>0?sqrt(num/den):0.0);
    report(label, Cref, got, true, 5e-3);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(das);cudaFree(dws);
}
__global__ void k_gemm_glu_w8a8(bf16* C, const uint8_t* A, const uint8_t* Wg, const uint8_t* Wu,
                                const float* as, const float* sg, const float* su, unsigned m,
                                unsigned n, unsigned k, unsigned act) {
    extern __shared__ bf16 smw[];
    d_gemm_glu_w8a8(C, A, Wg, Wu, as, sg, su, m, n, k, act, blockIdx.x, gridDim.x, smw);
}
static void test_gemm_glu_w8a8(const char* label, unsigned m, unsigned n, unsigned k, unsigned act) {
    seed(0x7F00u + m + n + k + act);
    std::vector<float> A = gen_bf16((size_t)m*k, 1.0f);
    std::vector<float> Wg = gen_bf16((size_t)n*k, 1.0f), Wu = gen_bf16((size_t)n*k, 1.0f);
    std::vector<uint8_t> A8((size_t)m*k), Wg8((size_t)n*k), Wu8((size_t)n*k);
    std::vector<float> as(m), sg(n), su(n), Cref((size_t)m*n);
    quant_row_act(A, A8, as, m, k);
    quant_col_wt(Wg, Wg8, sg, n, k); quant_col_wt(Wu, Wu8, su, n, k);
    auto gelu=[&](float x){ float c=0.7978845608028654f*(x+0.044715f*x*x*x); return 0.5f*x*(1.0f+tanhf(c)); };
    auto silu=[&](float x){ return x/(1.0f+expf(-x)); };
    for (unsigned i=0;i<m;i++) for (unsigned j=0;j<n;j++){ double ag=0,au=0;
        for (unsigned kk=0;kk<k;kk++){ ag += (double)e4m3_dec(A8[(size_t)i*k+kk])*(double)e4m3_dec(Wg8[(size_t)j*k+kk]);
                                       au += (double)e4m3_dec(A8[(size_t)i*k+kk])*(double)e4m3_dec(Wu8[(size_t)j*k+kk]); }
        float g=(float)(ag*(double)as[i]*(double)sg[j]), u=(float)(au*(double)as[i]*(double)su[j]);
        float a = act==PLOW_ACT_SILU_ ? silu(g) : gelu(g);
        Cref[(size_t)i*n+j]=bf16_rt(a*u); }
    bf16* dC=dev_bf((size_t)m*n); float* das=to_dev_f(as); float* dsg=to_dev_f(sg); float* dsu=to_dev_f(su);
    uint8_t *dA,*dWg,*dWu; CK(cudaMalloc(&dA,(size_t)m*k)); CK(cudaMalloc(&dWg,(size_t)n*k)); CK(cudaMalloc(&dWu,(size_t)n*k));
    CK(cudaMemcpy(dA,A8.data(),(size_t)m*k,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWg,Wg8.data(),(size_t)n*k,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWu,Wu8.data(),(size_t)n*k,cudaMemcpyHostToDevice));
    const size_t smem=(size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_gemm_glu_w8a8,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_gemm_glu_w8a8<<<188,256,smem>>>(dC,dA,dWg,dWu,das,dsg,dsu,m,n,k,act);
    CK(cudaDeviceSynchronize());
    report(label, Cref, from_dev(dC,(size_t)m*n), true, 8e-3);
    cudaFree(dA);cudaFree(dWg);cudaFree(dWu);cudaFree(dC);cudaFree(das);cudaFree(dsg);cudaFree(dsu);
}
/* Per-row activation quant op (QUANT_FP8): compare xq DEQUANTIZED + the a_scale against a host
 * quantizer of the same (amax/448) convention. Bit-exact scale; xq matches host RN e4m3 encoding. */
__global__ void k_quant_fp8(uint8_t* xq, const bf16* x, float* as, unsigned M, unsigned K) {
    d_quant_fp8(xq, x, as, M, K, blockIdx.x, gridDim.x);
}
static void test_quant_fp8(const char* label, unsigned M, unsigned K) {
    seed(0x7C00u + M*3 + K);
    std::vector<float> x = gen_bf16((size_t)M*K, 1.0f);
    std::vector<uint8_t> xq_ref((size_t)M*K); std::vector<float> as_ref(M);
    quant_row_act(x, xq_ref, as_ref, M, K);
    bf16* dx=to_dev(x); float* das; uint8_t* dxq;
    CK(cudaMalloc(&das,M*sizeof(float))); CK(cudaMalloc(&dxq,(size_t)M*K));
    k_quant_fp8<<<188,256>>>(dxq, dx, das, M, K);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> xq((size_t)M*K); std::vector<float> as(M);
    CK(cudaMemcpy(xq.data(),dxq,(size_t)M*K,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(as.data(),das,M*sizeof(float),cudaMemcpyDeviceToHost));
    /* compare via DEQUANTIZED values (byte-level RN can differ by 1 ulp only at exact ties, which
     * bf16 inputs almost never hit; the dequantized value + scale is the load-bearing contract). */
    std::vector<float> gref((size_t)M*K), ggot((size_t)M*K);
    for (unsigned m=0;m<M;m++) for (unsigned kk=0;kk<K;kk++){
        gref[(size_t)m*K+kk]=e4m3_dec(xq_ref[(size_t)m*K+kk])*as_ref[m];
        ggot[(size_t)m*K+kk]=e4m3_dec(xq[(size_t)m*K+kk])*as[m]; }
    report(label, gref, ggot, true, 1e-3);
    cudaFree(dx);cudaFree(das);cudaFree(dxq);
}

/* ==================== PREFILL grouped-MoE (router_pf/align/glu/down/combine, ops 73-77) ======
 * End-to-end oracle: the exact 5-op device chain (T-token router -> align/sort -> grouped gate_up
 * GLU GEMM -> grouped down GEMM + scatter -> T-row combine+sandwich) vs an f32 CPU golden. The two
 * norms feeding it (xn2 = pre_ffn_norm_2(resid); h1 = post_ffn_norm_1(dense)) are computed on host
 * as bf16 inputs so only the NEW ops' arithmetic is charged. Also verifies the align/sort invariants
 * (histogram counts, unique part-row per routing slot, pad rows UNUSED) and the router selection. */
static unsigned char* dev_u8(size_t n){ unsigned char* d=nullptr; CK(cudaMalloc(&d,n)); return d; }
static unsigned* dev_u32(size_t n){ unsigned* d=nullptr; CK(cudaMalloc(&d,n*4)); return d; }
static int* dev_i32(size_t n){ int* d=nullptr; CK(cudaMalloc(&d,n*4)); return d; }

__global__ void k_moe_router_pf(unsigned char* table, const bf16* resid, const bf16* proj,
                                const bf16* scale, const bf16* pes, unsigned H, unsigned E,
                                unsigned k, unsigned T, float root, float eps){
    extern __shared__ float ar[];
    d_moe_router_gemma_pf(table,resid,proj,scale,pes,H,E,k,T,root,eps,blockIdx.x,gridDim.x,ar);
}
__global__ void k_moe_align_pf(int* meta, const unsigned char* table, unsigned* rt, unsigned* rp,
                               float* rg, unsigned T, unsigned E, unsigned k){
    d_moe_align_gemma_pf(meta,table,rt,rp,rg,T,E,k,blockIdx.x);
}
__global__ void k_moe_group_glu_pf(bf16* fu, const bf16* xn2, const uint64_t* ewt, const int* meta,
                                   const unsigned* rt, unsigned I, unsigned H, unsigned E, unsigned act){
    extern __shared__ bf16 smg[];
    d_moe_group_glu_gemma_pf(fu,xn2,(const unsigned long long*)ewt,meta,rt,I,H,E,act,blockIdx.x,gridDim.x,smg);
}
__global__ void k_moe_group_down_pf(float* part, const bf16* fu, const uint64_t* ewt, const int* meta,
                                    const unsigned* rp, const float* rg, unsigned H, unsigned I, unsigned E){
    extern __shared__ bf16 smg[];
    d_moe_group_down_gemma_pf(part,fu,(const unsigned long long*)ewt,meta,rp,rg,H,I,E,blockIdx.x,gridDim.x,smg);
}
__global__ void k_moe_combine_pf(bf16* out, const float* part, const bf16* h1, const bf16* gamma,
                                 unsigned H, unsigned k, unsigned T, float eps){
    extern __shared__ float ar[];
    d_moe_combine_norm_gemma_pf(out,part,h1,gamma,H,k,T,eps,blockIdx.x,gridDim.x,ar);
}

static void chk(const char* name, bool ok){ printf("  %-40s %s\n", name, ok?"PASS":"FAIL"); if(!ok) g_fail=1; }

/* rmsnorm one row -> bf16 (matches d_rmsnorm/combine: f32 accumulate, gamma bf16). */
static std::vector<float> rmsnorm_row(const std::vector<float>& x, size_t off,
                                      const std::vector<float>& g, unsigned H, float eps){
    double ss=0; for(unsigned h=0;h<H;h++) ss+=(double)x[off+h]*x[off+h];
    float inv=1.f/sqrtf((float)(ss/(double)H)+eps);
    std::vector<float> o(H); for(unsigned h=0;h<H;h++) o[h]=bf16_rt(x[off+h]*inv*g[h]); return o;
}

static void test_moe_prefill_e2e(const char* label, unsigned T, unsigned E, unsigned k,
                                 unsigned I, unsigned H){
    seed(0x7000u+T*11+E+k+I+H);
    const float eps=1e-6f, root=1.0f/sqrtf((float)H);
    std::vector<float> resid=gen_bf16((size_t)T*H,0.6f);
    std::vector<float> proj =gen_bf16((size_t)E*H,1.0f);
    std::vector<float> scale=gen_bf16(H,1.0f);
    std::vector<float> pes  =gen_bf16(E,1.0f);
    std::vector<float> gu   =gen_bf16((size_t)E*2*I*H,1.0f);   /* [E,2I,H] fused gate_up */
    std::vector<float> dw   =gen_bf16((size_t)E*H*I,1.0f);     /* [E,H,I] down */
    std::vector<float> dg   =gen_bf16((size_t)T*H,0.6f);       /* dense MLP output */
    std::vector<float> g_pf1=gen_bf16(H,1.0f), g_pf2=gen_bf16(H,1.0f), g_pre2=gen_bf16(H,1.0f);
    // host xn2 (expert input) and h1 (dense sandwich half), bf16.
    std::vector<float> xn2((size_t)T*H), h1((size_t)T*H);
    for(unsigned t=0;t<T;t++){ auto a=rmsnorm_row(resid,(size_t)t*H,g_pre2,H,eps);
        auto b=rmsnorm_row(dg,(size_t)t*H,g_pf1,H,eps);
        for(unsigned h=0;h<H;h++){ xn2[(size_t)t*H+h]=a[h]; h1[(size_t)t*H+h]=b[h]; } }
    // host router per token.
    std::vector<unsigned> id((size_t)T*k); std::vector<float> gate((size_t)T*k);
    std::vector<unsigned> hcnt(E,0);
    for(unsigned t=0;t<T;t++){ std::vector<float> r(resid.begin()+(size_t)t*H, resid.begin()+(size_t)(t+1)*H);
        std::vector<unsigned> ii; std::vector<float> gg;
        router_ref(r,proj,scale,pes,H,E,k,root,eps,ii,gg);
        for(unsigned j=0;j<k;j++){ id[(size_t)t*k+j]=ii[j]; gate[(size_t)t*k+j]=gg[j]; hcnt[ii[j]]++; } }
    auto gelu=[&](float v){ float c=0.7978845608028654f*(v+0.044715f*v*v*v); return 0.5f*v*(1.0f+tanhf(c)); };
    // host part[(t*k+j)][h] and combine golden.
    std::vector<float> part_g((size_t)T*k*H);
    for(unsigned t=0;t<T;t++) for(unsigned j=0;j<k;j++){ unsigned e=id[(size_t)t*k+j];
        const float* gub=&gu[(size_t)e*2*I*H]; const float* wd=&dw[(size_t)e*H*I];
        std::vector<float> fu(I);
        for(unsigned n=0;n<I;n++){ double ag=0,au=0;
            for(unsigned h=0;h<H;h++){ ag+=(double)xn2[(size_t)t*H+h]*gub[(size_t)n*H+h];
                                       au+=(double)xn2[(size_t)t*H+h]*gub[(size_t)(I+n)*H+h]; }
            fu[n]=bf16_rt(gelu((float)ag)*(float)au); }
        for(unsigned h=0;h<H;h++){ double a=0; for(unsigned i=0;i<I;i++) a+=(double)fu[i]*wd[(size_t)h*I+i];
            part_g[((size_t)t*k+j)*H+h]=gate[(size_t)t*k+j]*(float)a; } }
    std::vector<float> comb_g((size_t)T*H);
    for(unsigned t=0;t<T;t++){ std::vector<float> sum(H,0.f);
        for(unsigned h=0;h<H;h++){ for(unsigned j=0;j<k;j++) sum[h]+=part_g[((size_t)t*k+j)*H+h]; }
        double ss=0; for(unsigned h=0;h<H;h++) ss+=(double)sum[h]*sum[h];
        float inv=1.f/sqrtf((float)(ss/(double)H)+eps);
        for(unsigned h=0;h<H;h++) comb_g[(size_t)t*H+h]=bf16_rt(sum[h]*inv*g_pf2[h]+h1[(size_t)t*H+h]); }

    // device pipeline.
    const unsigned BM=128, total_pad=T*k+E*BM;
    bf16 *dResid=to_dev(resid),*dProj=to_dev(proj),*dScale=to_dev(scale),*dPes=to_dev(pes);
    bf16 *dGU=to_dev(gu),*dDW=to_dev(dw),*dXn2=to_dev(xn2),*dH1=to_dev(h1),*dGpf2=to_dev(g_pf2);
    std::vector<uint64_t> ewt((size_t)E*2,0);
    for(unsigned e=0;e<E;e++){ ewt[(size_t)e*2+0]=(uint64_t)(dGU+(size_t)e*2*I*H);
                               ewt[(size_t)e*2+1]=(uint64_t)(dDW+(size_t)e*H*I); }
    uint64_t* dEwt=to_dev_u64(ewt);
    unsigned char* dTab=dev_u8((size_t)T*k*8);
    int* dMeta=dev_i32((size_t)3*E+2);
    unsigned *dRt=dev_u32(total_pad),*dRp=dev_u32(total_pad); float* dRg=to_dev_f(std::vector<float>(total_pad,0.f));
    /* The align op only initializes the [0, total_tiles*128) rows it actually uses; the allocated
     * tail (total_tiles*128 .. total_pad) is never touched by any kernel (the grouped GEMMs
     * enumerate exactly total_tiles tiles from meta). Preset the whole buffer to UNUSED (0xFF) so
     * the HOST invariant scan below skips that untouched tail instead of reading cudaMalloc garbage. */
    CK(cudaMemset(dRt,0xFF,(size_t)total_pad*4)); CK(cudaMemset(dRp,0xFF,(size_t)total_pad*4));
    bf16* dFug=dev_bf((size_t)total_pad*I);
    float* dPart=dev_f((size_t)T*k*H);
    bf16* dOut=dev_bf((size_t)T*H);
    const size_t smg=(size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_moe_group_glu_pf,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smg));
    CK(cudaFuncSetAttribute(k_moe_group_down_pf,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smg));
    k_moe_router_pf<<<188,256,(size_t)(H+E)*sizeof(float)>>>(dTab,dResid,dProj,dScale,dPes,H,E,k,T,root,eps);
    k_moe_align_pf<<<188,256>>>(dMeta,dTab,dRt,dRp,dRg,T,E,k);
    k_moe_group_glu_pf<<<188,256,smg>>>(dFug,dXn2,dEwt,dMeta,dRt,I,H,E,PLOW_ACT_GELU_TANH_);
    k_moe_group_down_pf<<<188,256,smg>>>(dPart,dFug,dEwt,dMeta,dRp,dRg,H,I,E);
    k_moe_combine_pf<<<188,256,(size_t)H*sizeof(float)>>>(dOut,dPart,dH1,dGpf2,H,k,T,eps);
    CK(cudaDeviceSynchronize());

    // (a) router table vs golden (ids exact, gates close).
    std::vector<unsigned char> hTab((size_t)T*k*8); CK(cudaMemcpy(hTab.data(),dTab,hTab.size(),cudaMemcpyDeviceToHost));
    unsigned idbad=0; double gerr=0;
    for(unsigned s=0;s<T*k;s++){ unsigned e=*(unsigned*)(hTab.data()+(size_t)s*8);
        float g=*(float*)(hTab.data()+(size_t)s*8+4); if(e!=id[s]) idbad++; gerr+=fabs(g-gate[s]); }
    { char nm[96]; snprintf(nm,sizeof nm,"%s router T*k ids", label); chk(nm, idbad==0); }
    // (b) align/sort invariants.
    std::vector<int> hMeta((size_t)3*E+2); CK(cudaMemcpy(hMeta.data(),dMeta,hMeta.size()*4,cudaMemcpyDeviceToHost));
    std::vector<unsigned> hRt(total_pad),hRp(total_pad); std::vector<float> hRg(total_pad);
    CK(cudaMemcpy(hRt.data(),dRt,total_pad*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hRp.data(),dRp,total_pad*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hRg.data(),dRg,total_pad*4,cudaMemcpyDeviceToHost));
    /* align only COPIES the device routing table's gate into row_gate, so validate against the
     * device table's own gate (bit-exact), not the host router_ref gate (which drifts by __expf). */
    bool align_ok=true; std::vector<unsigned> seen((size_t)T*k,0); const char* why="";
    for(unsigned e=0;e<E;e++){ if(hMeta[E+e]!=(int)hcnt[e]){ align_ok=false; why="cnt"; } }
    unsigned live=0;
    for(unsigned r=0;r<total_pad;r++){ if(hRp[r]==PLOW_EXPERT_UNUSED) continue; live++;
        unsigned pidx=hRp[r]; if(pidx>=T*k){align_ok=false;why="pidx-oob";continue;} seen[pidx]++;
        if(hRt[r]!=pidx/k){ align_ok=false; why="rowtok"; }
        float tg=*(float*)(hTab.data()+(size_t)pidx*8+4);
        if(hRg[r]!=tg){ align_ok=false; why="gate-copy"; } }
    if(live!=T*k){ align_ok=false; why="live!=T*k"; }
    for(unsigned s=0;s<T*k;s++) if(seen[s]!=1){ align_ok=false; why="dup/miss"; }
    { char nm[96]; snprintf(nm,sizeof nm,"%s align (cnt/unique/pad/gate)%s%s", label,
        align_ok?"":" why=", why); chk(nm, align_ok); }
    // (c) final combine vs golden (whole grouped GLU/down/scatter/combine chain).
    { char nm[96]; snprintf(nm,sizeof nm,"%s combine vs golden", label);
      report(nm, comb_g, from_dev(dOut,(size_t)T*H), true, 2e-2); }
    (void)gerr;
    cudaFree(dResid);cudaFree(dProj);cudaFree(dScale);cudaFree(dPes);cudaFree(dGU);cudaFree(dDW);
    cudaFree(dXn2);cudaFree(dH1);cudaFree(dGpf2);cudaFree(dEwt);cudaFree(dTab);cudaFree(dMeta);
    cudaFree(dRt);cudaFree(dRp);cudaFree(dRg);cudaFree(dFug);cudaFree(dPart);cudaFree(dOut);
}

/* Router T>1 with a deliberate 8-way TIE per token: ids 0..7 carry identical strong logits.
 * Lowest-id tie-break must pick 0..k-1 for EVERY token. */
static void test_moe_router_pf_tie(const char* label, unsigned T, unsigned H, unsigned k){
    seed(0x7700u+T+H+k); const unsigned E=16; const float eps=1e-6f, root=1.0f/sqrtf((float)H);
    // ALL-POSITIVE resid/scale/row so the shared-row logit is strictly positive -> experts 0..7
    // (identical rows) tie ABOVE the zero-logit experts 8..15; lowest-id tie must pick 0..k-1.
    std::vector<float> resid(T*H),scale(H); std::vector<float> pes(E,1.0f);
    for(size_t i=0;i<resid.size();i++) resid[i]=bf16_rt(0.2f+0.4f*fabsf(rnd()));
    for(size_t i=0;i<scale.size();i++) scale[i]=bf16_rt(0.5f+0.5f*fabsf(rnd()));
    std::vector<float> proj((size_t)E*H,0.f), row(H);
    for(size_t i=0;i<row.size();i++) row[i]=bf16_rt(0.2f+0.8f*fabsf(rnd()));
    for(unsigned e=0;e<8;e++) for(unsigned h=0;h<H;h++) proj[(size_t)e*H+h]=row[h];
    bf16 *dResid=to_dev(resid),*dProj=to_dev(proj),*dScale=to_dev(scale),*dPes=to_dev(pes);
    unsigned char* dTab=dev_u8((size_t)T*k*8);
    k_moe_router_pf<<<188,256,(size_t)(H+E)*sizeof(float)>>>(dTab,dResid,dProj,dScale,dPes,H,E,k,T,root,eps);
    CK(cudaDeviceSynchronize());
    std::vector<unsigned char> hTab((size_t)T*k*8); CK(cudaMemcpy(hTab.data(),dTab,hTab.size(),cudaMemcpyDeviceToHost));
    bool ok=true; for(unsigned t=0;t<T;t++) for(unsigned j=0;j<k;j++)
        if(*(unsigned*)(hTab.data()+((size_t)t*k+j)*8)!=j) ok=false;
    chk(label, ok);
    cudaFree(dResid);cudaFree(dProj);cudaFree(dScale);cudaFree(dPes);cudaFree(dTab);
}

#if PLOW_NV_W8A8
/* ==================== PREFILL grouped-MoE w8a8 (e4m3, ops _W8A8) =============================
 * Coverage for the native w8a8 grouped gate/up (GeGLU) + down GEMMs — d_moe_group_glu_gemma_pf_w8a8
 * / d_moe_group_down_gemma_pf_w8a8 (op_moe.cuh). Under PLOW_NV_HOPPER these are the wgmma.m64n128k32
 * .f32.e4m3.e4m3 FORKS (op_moe_sm90.cuh); WITHOUT it they are the mma.sync.m16n8k32 #else bodies.
 * Same source drives both — only PLOW_NV_W8A8 gates the arms.
 *
 * ORACLE: an f32 CPU golden over the SAME e4m3 bytes the kernel reads (activations + weights are
 * random e4m3, scales are arbitrary positive floats — this is NOT a quant round-trip, it charges
 * the kernel's own arithmetic: the row_token GATHER, the fp8 mma fragment map, the per-token x
 * per-channel scale epilogue, GeGLU, gate/fscale scatter, and the pad-row zero-fill). e4m3*e4m3
 * products are EXACT in f32, so the mma.sync path matches the oracle to ~0 (only the bf16 fu store
 * rounds); the wgmma fork's accumulator is NOT true f32 and its error grows with K (K=H for GLU,
 * K=I for DOWN) — see sm90_wgmma.cuh.
 *
 * TOLERANCE: GLU 4e-3, DOWN 1e-3. Rationale — the out-of-band A/B (E=3 m_e=100 I=144 H=256) measured
 * the wgmma fork at relL2 1.06e-3 (GLU, K=256) / 1.10e-4 (DOWN, K=144) vs this oracle. 4e-3 leaves
 * ~3.8x margin on GLU and 1e-3 leaves ~9x on DOWN — wide enough NOT to trip on the documented fp8
 * accumulator drift (which scales ~linearly with K: dense w8a8 runs 3.9e-5@K=32 -> 1.14e-3@K=3840),
 * yet a fragment-map / gather / scale bug misplaces whole rows and blows past these by 10x+. The
 * shapes here keep K small (<=512) so the accumulator term stays far under tol; the margin is real,
 * not a K-inflated tripwire. Both bodies also run under the plain (mma.sync) build at ~0 error. */
__global__ void k_moe_group_glu_w8a8(bf16* fu, const uint8_t* xq8, const float* asc,
        const unsigned long long* ewt, const unsigned long long* est, const int* meta,
        const unsigned* rt, unsigned I, unsigned H, unsigned E, unsigned act) {
    extern __shared__ bf16 sm8[];
    d_moe_group_glu_gemma_pf_w8a8(fu,xq8,asc,ewt,est,meta,rt,I,H,E,act,blockIdx.x,gridDim.x,sm8);
}
__global__ void k_moe_group_down_w8a8(float* part, const uint8_t* fu8, const float* fsc,
        const unsigned long long* ewt, const unsigned long long* est, const int* meta,
        const unsigned* rp, const float* rg, unsigned H, unsigned I, unsigned E) {
    extern __shared__ bf16 sm8[];
    d_moe_group_down_gemma_pf_w8a8(part,fu8,fsc,ewt,est,meta,rp,rg,H,I,E,blockIdx.x,gridDim.x,sm8);
}

static uint8_t* to_dev_u8(const std::vector<uint8_t>& h){ uint8_t* d=nullptr; CK(cudaMalloc(&d,h.size()));
    CK(cudaMemcpy(d,h.data(),h.size(),cudaMemcpyHostToDevice)); return d; }
static unsigned long long* to_dev_u64_(const std::vector<unsigned long long>& h){
    unsigned long long* d=nullptr; CK(cudaMalloc(&d,h.size()*8));
    CK(cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice)); return d; }

/* E experts, ragged per-expert row count m_e (m_e NOT a multiple of BM exercises the pad tail). */
static void test_moe_group_w8a8(const char* label, unsigned E, unsigned me, unsigned I, unsigned H){
    seed(0x8A00u + E*131 + me*7 + I*5 + H);
    const unsigned BM = PGM_BM;
    /* meta[3E+2]: rowoff[e]=meta[e], count[e]=meta[E+e], tilep[e]=meta[2E+e], total_tiles=meta[3E]. */
    std::vector<int> meta(3*E+2, 0);
    unsigned tiles=0, rows=0;
    for(unsigned e=0;e<E;e++){ meta[e]=(int)rows; meta[E+e]=(int)me; meta[2*E+e]=(int)tiles;
        unsigned t=(me+BM-1)/BM; tiles+=t; rows+=t*BM; }
    meta[3*E]=(int)tiles;
    const unsigned Mtot=rows, ntok=E*me;

    /* row_token / row_partidx: live rows -> global slot; pad rows -> UNUSED. */
    std::vector<unsigned> rt(Mtot, PLOW_EXPERT_UNUSED), rp(Mtot, PLOW_EXPERT_UNUSED);
    std::vector<float> rg(Mtot, 0.f);
    unsigned slot=0;
    for(unsigned e=0;e<E;e++){ unsigned b=(unsigned)meta[e];
        for(unsigned r=0;r<me;r++){ rt[b+r]=slot; rp[b+r]=slot; rg[b+r]=0.5f+0.1f*(float)(slot%3); slot++; } }

    /* random e4m3 bytes + arbitrary positive scales (oracle reads the SAME bytes). */
    auto genq=[&](size_t n){ std::vector<uint8_t> v(n); for(auto&b:v) b=e4m3_enc(rnd()*0.8f); return v; };
    auto genp=[&](size_t n,float lo,float hi){ std::vector<float> v(n);
        for(auto&x:v) x=lo+hi*fabsf(rnd()); return v; };
    std::vector<uint8_t> hx =genq((size_t)ntok*H);       /* GLU activation [ntok][H] */
    std::vector<uint8_t> hgu=genq((size_t)E*2*I*H);      /* [E][2I][H] fused gate_up */
    std::vector<uint8_t> hdw=genq((size_t)E*H*I);        /* [E][H][I] down */
    std::vector<uint8_t> hfu=genq((size_t)Mtot*I);       /* DOWN activation [Mtot][I] */
    std::vector<float> hasc=genp(ntok,0.01f,0.002f);     /* per-token act scale */
    std::vector<float> hsc =genp((size_t)E*2*I,0.02f,0.004f); /* per-expert [2I] gate|up */
    std::vector<float> hdsc=genp((size_t)E*H,0.03f,0.005f);   /* per-expert [H] down */
    std::vector<float> hfsc=genp(Mtot,0.015f,0.003f);        /* per-row fu scale */

    /* ---- f32 oracle over the SAME e4m3 bytes ---- */
    std::vector<float> glu_g((size_t)Mtot*I, 0.f), dn_g((size_t)ntok*H, 0.f);
    auto gelu=[&](float v){ float c=0.7978845608028654f*(v+0.044715f*v*v*v); return 0.5f*v*(1.f+tanhf(c)); };
    for(unsigned e=0;e<E;e++){ unsigned base=(unsigned)meta[e];
        const uint8_t* Wg=&hgu[(size_t)e*2*I*H]; const uint8_t* Wu=Wg+(size_t)I*H;
        const float* sc=&hsc[(size_t)e*2*I];
        for(unsigned r=0;r<me;r++){ unsigned tok=e*me+r; float as=hasc[tok];
            for(unsigned n=0;n<I;n++){ double ag=0,au=0;
                for(unsigned kk=0;kk<H;kk++){ float x=e4m3_dec(hx[(size_t)tok*H+kk]);
                    ag+=(double)x*e4m3_dec(Wg[(size_t)n*H+kk]);
                    au+=(double)x*e4m3_dec(Wu[(size_t)n*H+kk]); }
                float g=(float)ag*as*sc[n], u=(float)au*as*sc[I+n];
                glu_g[(size_t)(base+r)*I+n]=bf16_rt(gelu(g)*u); } }
        const uint8_t* Wd=&hdw[(size_t)e*H*I]; const float* dsc=&hdsc[(size_t)e*H];
        for(unsigned r=0;r<me;r++){ unsigned row=base+r, tok=e*me+r;
            for(unsigned h=0;h<H;h++){ double a=0;
                for(unsigned kk=0;kk<I;kk++) a+=(double)e4m3_dec(hfu[(size_t)row*I+kk])*e4m3_dec(Wd[(size_t)h*I+kk]);
                dn_g[(size_t)tok*H+h]=rg[row]*hfsc[row]*dsc[h]*(float)a; } } }

    /* ---- device ---- */
    uint8_t *dx=to_dev_u8(hx),*dgu=to_dev_u8(hgu),*ddw=to_dev_u8(hdw),*dfu8=to_dev_u8(hfu);
    float *dasc=to_dev_f(hasc),*dsc=to_dev_f(hsc),*ddsc=to_dev_f(hdsc),*dfsc=to_dev_f(hfsc),*drg=to_dev_f(rg);
    int* dMeta=to_dev_i(meta);
    unsigned* dRt=dev_u32(Mtot); CK(cudaMemcpy(dRt,rt.data(),Mtot*4,cudaMemcpyHostToDevice));
    unsigned* dRp=dev_u32(Mtot); CK(cudaMemcpy(dRp,rp.data(),Mtot*4,cudaMemcpyHostToDevice));
    std::vector<unsigned long long> ewt(2*E), est(2*E);
    for(unsigned e=0;e<E;e++){ ewt[2*e+0]=(unsigned long long)(dgu+(size_t)e*2*I*H);
        ewt[2*e+1]=(unsigned long long)(ddw+(size_t)e*H*I);
        est[2*e+0]=(unsigned long long)(dsc+(size_t)e*2*I);
        est[2*e+1]=(unsigned long long)(ddsc+(size_t)e*H); }
    unsigned long long *dEwt=to_dev_u64_(ewt),*dEst=to_dev_u64_(est);
    bf16*  dGlu=dev_bf((size_t)Mtot*I);
    float* dDn =dev_f((size_t)ntok*H);
    const size_t smem=(size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_moe_group_glu_w8a8, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(k_moe_group_down_w8a8,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_moe_group_glu_w8a8<<<188,256,smem>>>(dGlu,dx,dasc,dEwt,dEst,dMeta,dRt,I,H,E,PLOW_ACT_GELU_TANH_);
    k_moe_group_down_w8a8<<<188,256,smem>>>(dDn,dfu8,dfsc,dEwt,dEst,dMeta,dRp,drg,H,I,E);
    CK(cudaDeviceSynchronize());

    std::vector<float> got_glu=from_dev(dGlu,(size_t)Mtot*I);
    std::vector<float> got_dn((size_t)ntok*H);
    CK(cudaMemcpy(got_dn.data(),dDn,(size_t)ntok*H*4,cudaMemcpyDeviceToHost));
    { char nm[96]; snprintf(nm,sizeof nm,"%s GLU vs oracle", label);  report(nm, glu_g, got_glu, true, 4e-3); }
    { char nm[96]; snprintf(nm,sizeof nm,"%s DOWN vs oracle", label); report(nm, dn_g,  got_dn,  true, 1e-3); }

    cudaFree(dx);cudaFree(dgu);cudaFree(ddw);cudaFree(dfu8);cudaFree(dasc);cudaFree(dsc);
    cudaFree(ddsc);cudaFree(dfsc);cudaFree(drg);cudaFree(dMeta);cudaFree(dRt);cudaFree(dRp);
    cudaFree(dEwt);cudaFree(dEst);cudaFree(dGlu);cudaFree(dDn);
}
#endif /* PLOW_NV_W8A8 */

int main() {
    int dev=0; cudaDeviceProp p; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p,dev));
    printf("device: %s  sm_%d%d  SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
#ifdef FA_NV_WAVE64_NEGCTRL
    printf("*** FA_NV_WAVE64_NEGCTRL build: warp reductions use the wrong wave64 offset; "
           "flash/headnorm/norm ops MUST fail ***\n");
#endif
    printf("\n== headnorm+rope ==\n");
    test_headnorm_rope<128>("headnorm_rope hd128 t3 h8", 3, 8);
    test_headnorm_rope<256>("headnorm_rope hd256 t3 h4", 3, 4);
    test_headnorm_rope<512>("headnorm_rope hd512 t3 h4", 3, 4);

    printf("\n== flash decode+merge ==\n");
    test_flash<128,4>("flash hd128 GF4 h8 kv2 len40", 8, 2, 40);
    test_flash<256,2>("flash hd256 GF2 h4 kv2 len40", 4, 2, 40);
    test_flash<512,2>("flash hd512 GF2 h4 kv1 len40", 4, 1, 40);
    test_flash<256,2>("flash hd256 GF2 h4 kv2 len300",4, 2, 300); /* multi-tile KV */

    printf("\n== norm_residual(_norm) (Gemma sandwich; hidden 3840) ==\n");
    test_nrn("norm_residual_norm rows4 f3840 s1.0", 4, 3840, 1.0f);
    test_nrn("norm_residual_norm rows4 f3840 s0.8", 4, 3840, 0.8f); /* folded layer_scalar */
    test_nr ("norm_residual      rows4 f3840 s1.0", 4, 3840, 1.0f);
    /* Gemma-31B hidden 5376 > old RN_REG*256=4096: exercised the scalar slow path before the
     * RN_REG 16->24 bump, now takes the register-vector fast path (5376 <= 6144, 8-aligned). */
    test_nrn("norm_residual_norm rows4 f5376 s1.0", 4, 5376, 1.0f);
    test_nrn("norm_residual_norm rows4 f5376 s0.8", 4, 5376, 0.8f);
    test_nr ("norm_residual      rows4 f5376 s1.0", 4, 5376, 1.0f);

    printf("\n== softcap ==\n");
    test_softcap("softcap n=262144 cap=30", 262144, 30.0f);

    printf("\n== prefill tiled GEMM (m16n8k16) ==\n");
    test_gemm("gemm m64 n256 k3840 (q_proj-ish)", 64, 256, 3840, 0);
    test_gemm("gemm m200 n512 k3840 (M-ragged)", 200, 512, 3840, 0);
    test_gemm("gemm m1 n300 k3840 a_row0=63 (lm_head-ish)", 1, 300, 3840, 63);

    printf("\n== prefill GEMM_GLU (gate|up fused) ==\n");
    test_gemm_glu("gemm_glu m64 n256 k3840 gelu", 64, 256, 3840, 0);
    test_gemm_glu("gemm_glu m64 n256 k3840 silu", 64, 256, 3840, 1);

    printf("\n== flash prefill mma.sync QK^T (fused ns=1) ==\n");
    /* args: nh, nkv, seq_q, seq_kv, q_pos0, window, nsplit */
    test_flash_prefill<256,64,32>("flash_pre hd256 h4 kv2 len128 causal fused (tile-exact)", 4, 2, 128, 128, 0, 0, 1);
    test_flash_prefill<256,64,32>("flash_pre hd256 h4 kv2 len200 causal fused (ragged q+kv)", 4, 2, 200, 200, 0, 0, 1);
    test_flash_prefill<512,32,16>("flash_pre hd512 h4 kv1 len200 causal fused", 4, 1, 200, 200, 0, 0, 1);
    test_flash_prefill<256,64,32>("flash_pre hd256 h4 kv2 len300 window128 fused (straddle)", 4, 2, 300, 300, 0, 128, 1);
    /* CHUNKED prefill: seq_q<seq_kv, q_pos0>0 (a later chunk attending prior history). */
    test_flash_prefill<256,64,32>("flash_pre hd256 h4 kv2 chunk sq100 skv612 qp0=512 causal", 4, 2, 100, 612, 512, 0, 1);
    test_flash_prefill<256,64,32>("flash_pre hd256 h4 kv2 chunk sq100 skv900 qp0=800 window128", 4, 2, 100, 900, 800, 128, 1);
    test_flash_prefill<512,32,16>("flash_pre hd512 h4 kv1 chunk sq96 skv500 qp0=404 causal", 4, 1, 96, 500, 404, 0, 1);

    printf("\n== flash prefill SOFT softmax (scale~1/sqrt(HD)) — exercises bf16-P in the mma P.V (T4) ==\n");
    /* Soft, spread-out softmax so P is a genuine distribution (NOT one-hot): this is the case that
     * charges the tensor-core P.V's bf16 P-operand rounding against the f32 CPU ref. Multi-KV-tile
     * (len>>BKV) so the online rescale + corr path is stressed across many tiles. */
    test_flash_prefill<256,64,32>("flash_pre hd256 soft len512 causal",      4, 2, 512,  512,  0, 0,   1, 0.0625f);
    test_flash_prefill<256,64,32>("flash_pre hd256 soft len512 window128",   4, 2, 512,  512,  0, 128, 1, 0.0625f);
    test_flash_prefill<512,32,16>("flash_pre hd512 soft len512 causal",      4, 1, 512,  512,  0, 0,   1, 0.0442f);
    test_flash_prefill<256,64,32>("flash_pre hd256 soft chunk sq128 skv2048",4, 2, 128,  2048, 1920, 0, 1, 0.0625f);
    test_flash_prefill<256,64,32>("flash_pre hd256 soft len512 causal ns4",  4, 2, 512,  512,  0, 0,   4, 0.0625f);

    /* T5 cp.async KV-stream pipeline (PLOW_NV_FA_PIPE): natural-K + prefetched staging changes the
     * tile enumeration and adds the last-tile empty-commit edge. These stress LONG multi-tile runs
     * with RAGGED last tiles (skv not a BKV multiple) and windows straddling tile boundaries, where a
     * miscounted cp.async wait_group or a lost partial tile would corrupt the tail. Bit-exact under
     * both PIPE settings (natural-K is the T3-proven non-.trans equivalent; cp.async moves the same
     * bytes), so these PASS at the same relL2 band whether or not the pipeline is compiled in. */
    printf("\n== flash prefill T5 pipeline edges (long multi-tile, ragged last tile) ==\n");
    test_flash_prefill<256,64,32>("flash_pre hd256 soft len1000 causal (ragged last tile)", 4, 2, 1000, 1000, 0, 0,   1, 0.0625f);
    test_flash_prefill<256,64,32>("flash_pre hd256 soft len2000 window256 (mid-tile floor)", 4, 2, 2000, 2000, 0, 256, 1, 0.0625f);
    test_flash_prefill<512,32,16>("flash_pre hd512 soft chunk sq50 skv3000 qp0=2950 causal", 4, 1, 50, 3000, 2950, 0, 1, 0.0442f);
    test_flash_prefill<512,32,16>("flash_pre hd512 soft len1024 window300 (straddle+ragged)", 4, 1, 1024, 1024, 0, 300, 1, 0.0442f);
    test_flash_prefill<256,64,32>("flash_pre hd256 soft chunk sq100 skv2050 qp0=1950 ns3", 4, 2, 100, 2050, 1950, 0, 3, 0.0625f);

    /* T7 L1 2-stage V double-buffer on the hd512 arm (VBUF=2): V[t+1] is prefetched into the OTHER
     * ring buffer during P.V[t], so the wait_group depth AND the ring-buffer reuse parity are new
     * boundary conditions the T5 single-buffer path never had. These pin the corners a miscounted
     * cp.async group or a clobbered ring slot would corrupt (hd256 stays VBUF=1, unaffected):
     *   nt=1  : prologue-V + DOUBLE empty-commit tail (both K and V empty on the only tile);
     *   nt=2  : minimal ring, each buffer used exactly once, tail on parity-1;
     *   nt=3  : ring WRAPAROUND — V[2] restages b0, which P.V[0] read and freed 2 tiles earlier;
     *   ragged: last V tile partial (skv not a BKV multiple) under the double-buffer prefetch;
     *   window: sliding straddle + eff_lo floor interacting with the V prefetch. */
    printf("\n== flash prefill T7 hd512 2-stage V double-buffer edges ==\n");
    test_flash_prefill<512,32,16>("flash_pre hd512 soft len16 causal (nt=1 double-empty tail)", 4, 1, 16, 16, 0, 0, 1, 0.0442f);
    test_flash_prefill<512,32,16>("flash_pre hd512 soft len32 causal (nt=2 minimal ring)", 4, 1, 32, 32, 0, 0, 1, 0.0442f);
    test_flash_prefill<512,32,16>("flash_pre hd512 soft len48 causal (nt=3 ring wraparound)", 4, 1, 48, 48, 0, 0, 1, 0.0442f);
    test_flash_prefill<512,32,16>("flash_pre hd512 soft len40 causal (ragged last V tile)", 4, 1, 40, 40, 0, 0, 1, 0.0442f);
    test_flash_prefill<512,32,16>("flash_pre hd512 soft len96 window40 (straddle+dbuf)", 4, 1, 96, 96, 0, 40, 1, 0.0442f);
    test_flash_prefill<512,32,16>("flash_pre hd512 soft chunk sq33 skv97 qp0=64 causal (ragged+chunk dbuf)", 4, 1, 33, 97, 64, 0, 1, 0.0442f);

    printf("\n== flash prefill VARLEN mux (PX-1 stage 2, block-diagonal pack) ==\n");
    /* Mid-tile request boundaries (qlen % BQ != 0), shuffled slots, soft softmax (bf16-P regime).
     * hd256: BQ=64 sliding-layer shape; hd512: BQ=32 full-layer shape. */
    test_flash_prefill_varlen<256,64,32>("varlen hd256 R4 fresh causal midtile", 4, 2,
        {33,64,100,47}, {33,64,100,47}, {2,0,3,1}, 0, 0.0625f);
    test_flash_prefill_varlen<256,64,32>("varlen hd256 R4 chunked window128", 4, 2,
        {33,64,100,47}, {533,64,612,147}, {2,0,3,1}, 128, 0.0625f);
    test_flash_prefill_varlen<256,64,32>("varlen hd256 R4 tiny reqs (<1 tile each)", 4, 2,
        {5,7,3,1}, {5,7,3,1}, {1,3,0,2}, 0, 0.0625f);
    test_flash_prefill_varlen<256,64,32>("varlen hd256 R1 degenerate table", 4, 2,
        {200}, {200}, {0}, 0, 0.0625f);
    test_flash_prefill_varlen<512,32,16>("varlen hd512 R3 chunked causal midtile", 4, 1,
        {17,40,9}, {217,40,105}, {2,0,1}, 0, 0.0442f);
    test_flash_prefill_varlen<512,32,16>("varlen hd512 R4 fresh causal", 4, 1,
        {32,90,15,64}, {32,90,15,64}, {3,1,0,2}, 0, 0.0442f);

    printf("\n== flash prefill mma.sync QK^T (split ns>1 + merge) ==\n");
    test_flash_prefill<256,64,32>("flash_pre hd256 h4 kv2 len200 causal ns3", 4, 2, 200, 200, 0, 0, 3);
    test_flash_prefill<512,32,16>("flash_pre hd512 h4 kv1 len300 causal ns4", 4, 1, 300, 300, 0, 0, 4);
    test_flash_prefill<256,64,32>("flash_pre hd256 h4 kv2 chunk sq128 skv2048 qp0=1920 causal ns4", 4, 2, 128, 2048, 1920, 0, 4);

    printf("\n== fp8 (w8a16) decode GEMV (G7) ==\n");
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840", 4096, 3840);
    test_gemv_fp8("gemv_fp8 o_proj N3840 K4096", 3840, 4096);
    test_gemv_fp8("gemv_fp8 down   N3840 K15360", 3840, 15360);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840", 15360, 3840, PLOW_ACT_GELU_TANH_);
    /* BATCHED fp8 decode GEMV (PLOW_DECODE_BATCH 2..32). M=1 is the byte-identical B=1 path;
     * 2/8/32 are the ladder + block-walk multiples; 5/17 are ragged remainders. */
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840 M=2",  4096, 3840, 2);
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840 M=8",  4096, 3840, 8);
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840 M=32", 4096, 3840, 32);
    test_gemv_fp8("gemv_fp8 q_proj N4096 K3840 M=5 (ragged)",  4096, 3840, 5);
    test_gemv_fp8("gemv_fp8 o_proj N3840 K4096 M=17 (ragged)", 3840, 4096, 17);
    test_gemv_fp8("gemv_fp8 down N3840 K15360 M=32", 3840, 15360, 32);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840 M=2",  15360, 3840, PLOW_ACT_GELU_TANH_, 2);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840 M=8",  15360, 3840, PLOW_ACT_GELU_TANH_, 8);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840 M=32", 15360, 3840, PLOW_ACT_GELU_TANH_, 32);
    test_gemv_glu_fp8("gemv_glu_fp8 silu N15360 K3840 M=5 (ragged)", 15360, 3840, PLOW_ACT_SILU_, 5);
    test_gemv_glu_fp8("gemv_glu_fp8 gelu N15360 K3840 M=17 (ragged)", 15360, 3840, PLOW_ACT_GELU_TANH_, 17);

    printf("\n== Gemma-4 26B-A4B bf16 MoE decode (router/glu/down/combine) ==\n");
    /* Real 26B geometry: H=2816, E=128, k=8, moe_inter I=704. */
    test_moe_router("moe_router H2816 E128 k8", 2816, 128, 8);
    test_moe_router("moe_router H2816 E16 k4",  2816, 16, 4);
    test_moe_router_tie("moe_router TIE lowest-id (8 tied, top4)", 2816, 4);
    test_moe_router_split("moe_router_split H2816 E128 k8", 2816, 128, 8);
    test_moe_router_split("moe_router_split H2816 E16 k4", 2816, 16, 4);
    test_moe_router_split_tie("moe_router_split TIE lowest-id", 2816, 4);
    test_moe_router_split("moe_router_fast H2816 E128 k8", 2816, 128, 8, true);
    test_moe_router_split("moe_router_fast H2816 E16 k4", 2816, 16, 4, true);
    test_moe_router_split_tie("moe_router_fast TIE lowest-id", 2816, 4, true);
    test_moe_glu("moe_glu E128 k8 I704 H2816", 128, 8, 704, 2816);
    test_moe_down("moe_down E128 k8 I704 H2816", 128, 8, 704, 2816);
    test_moe_glu_fp8("moe_glu_fp8 E4 k2 I704 H2816", 4, 2, 704, 2816);
    test_moe_down_fp8("moe_down_fp8 E4 k2 I704 H2816", 4, 2, 704, 2816);
    test_moe_combine("moe_combine H2816 k8", 2816, 8);
    test_moe_combine_norm("moe_combine_norm H2816 k8", 2816, 8, 1e-6f);
    test_moe_glu_norm("moe_glu_norm E128 k8 I704 H2816", 128, 8, 704, 2816, 1e-6f);

    /* ---- BATCHED DECODE (B>1, plans/batch-decode.md): every decode MoE op vs the SAME per-row
     * f32 golden. Each op is covered with rows routing to the SAME experts (share=1, the weight
     * reuse case the channel-major sweep is built for) and to DISJOINT experts (share=0, which
     * catches a row/slot index mixup), plus a top-k tie replicated across rows. ---- */
    printf("-- batched MoE decode (B>1)\n");
    test_moe_router("moe_router B2 E128 k8", 2816, 128, 8, 2);
    test_moe_router("moe_router B8 E128 k8", 2816, 128, 8, 8);
    test_moe_router_tie("moe_router TIE B4 (every row lowest-id)", 2816, 4, 4);
    test_moe_router_split("moe_router_split B2 E128 k8", 2816, 128, 8, false, 2);
    test_moe_router_split("moe_router_split B8 E128 k8", 2816, 128, 8, false, 8);
    test_moe_router_split("moe_router_split B32 E128 k8", 2816, 128, 8, false, 32);
    test_moe_router_split("moe_router_fast B8 E128 k8", 2816, 128, 8, true, 8);
    test_moe_router_split_tie("moe_router_split TIE B4", 2816, 4, false, 4);
    test_moe_glu("moe_glu B2 SHARED expert", 128, 8, 704, 2816, 2, 1);
    test_moe_glu("moe_glu B2 DISJOINT experts", 128, 8, 704, 2816, 2, 0);
    test_moe_glu("moe_glu B8 DISJOINT experts", 128, 8, 704, 2816, 8, 0);
    test_moe_glu_norm("moe_glu_norm B2 SHARED expert", 128, 8, 704, 2816, 1e-6f, 2, 1);
    test_moe_glu_norm("moe_glu_norm B2 DISJOINT experts", 128, 8, 704, 2816, 1e-6f, 2, 0);
    test_moe_glu_norm("moe_glu_norm B8 SHARED expert", 128, 8, 704, 2816, 1e-6f, 8, 1);
    test_moe_glu_norm("moe_glu_norm B8 DISJOINT experts", 128, 8, 704, 2816, 1e-6f, 8, 0);
    test_moe_glu_norm("moe_glu_norm B32 SHARED expert", 128, 8, 704, 2816, 1e-6f, 32, 1);
    test_moe_down("moe_down B2 SHARED expert", 128, 8, 704, 2816, 2, 1);
    test_moe_down("moe_down B2 DISJOINT experts", 128, 8, 704, 2816, 2, 0);
    test_moe_down("moe_down B8 DISJOINT experts", 128, 8, 704, 2816, 8, 0);
    test_moe_down("moe_down B32 SHARED expert", 128, 8, 704, 2816, 32, 1);
    /* fp8 expert twins batched (this branch): channel-major B*k sweep, per-row x/part. E small so
     * DISJOINT (B*k<=E) exercises distinct-expert weight streams; SHARED exercises L2 reuse. */
    test_moe_glu_fp8("moe_glu_fp8 B2 SHARED expert", 4, 2, 704, 2816, 2, 1);
    test_moe_glu_fp8("moe_glu_fp8 B2 DISJOINT experts", 8, 2, 704, 2816, 2, 0);
    test_moe_glu_fp8("moe_glu_fp8 B8 DISJOINT experts", 128, 8, 704, 2816, 8, 0);
    test_moe_glu_fp8("moe_glu_fp8 B32 SHARED expert", 16, 4, 704, 2816, 32, 1);
    test_moe_down_fp8("moe_down_fp8 B2 SHARED expert", 4, 2, 704, 2816, 2, 1);
    test_moe_down_fp8("moe_down_fp8 B2 DISJOINT experts", 8, 2, 704, 2816, 2, 0);
    test_moe_down_fp8("moe_down_fp8 B8 DISJOINT experts", 128, 8, 704, 2816, 8, 0);
    test_moe_down_fp8("moe_down_fp8 B32 SHARED expert", 16, 4, 704, 2816, 32, 1);
    test_moe_combine_norm("moe_combine_norm B2 H2816 k8", 2816, 8, 1e-6f, 2);
    test_moe_combine_norm("moe_combine_norm B8 H2816 k8", 2816, 8, 1e-6f, 8);
    test_moe_combine_norm("moe_combine_norm B32 H2816 k8", 2816, 8, 1e-6f, 32);
    test_moe_comb_resid_norm("moe_comb_resid_norm H2816 k8", 2816, 8, 1e-6f, 0.515625f);

    printf("\n== Gemma-4 26B-A4B bf16 grouped-MoE PREFILL (router_pf/align/glu/down/combine) ==\n");
    /* Real 26B geometry (H=2816, E=128, k=8, I=704) at several chunk sizes, incl. a partial-tile T. */
    test_moe_router_pf_tie("moe_router_pf TIE lowest-id T5", 5, 2816, 8);
    test_moe_prefill_e2e("moe_prefill T16  E128 k8 I704 H2816", 16, 128, 8, 704, 2816);
    test_moe_prefill_e2e("moe_prefill T129 E128 k8 I704 H2816", 129, 128, 8, 704, 2816); /* >1 m-tile */
    test_moe_prefill_e2e("moe_prefill T512 E128 k8 I704 H2816", 512, 128, 8, 704, 2816);
    test_moe_prefill_e2e("moe_prefill T64  E16  k4 I704 H2816", 64, 16, 4, 704, 2816);

#if PLOW_NV_W8A8
    /* PREFILL grouped-MoE w8a8 (e4m3) — d_moe_group_{glu,down}_gemma_pf_w8a8 vs f32 oracle over the
     * SAME e4m3 bytes. Ragged m_e (not a multiple of BM=128) + multiple experts exercise the pad
     * tail and per-expert tiling; mirrors the out-of-band A/B reference shape (E3 m100 I144 H256). */
    printf("\n-- prefill grouped-MoE w8a8 (e4m3) --\n");
    test_moe_group_w8a8("moe_group_w8a8 E3 m100 I144 H256 (ref)",   3, 100, 144, 256);
    test_moe_group_w8a8("moe_group_w8a8 E2 m192 I256 H512 (ragged)", 2, 192, 256, 512);
    test_moe_group_w8a8("moe_group_w8a8 E4 m64  I128 H256 (m<BM)",   4, 64,  128, 256);
#endif

    /* PREFILL fp8 (w8a16) GEMM / GEMM_GLU — T6 L2. Same shapes as the bf16 GEMM cases (q/k/v/o/down
     * proj-ish, M-ragged, lm_head M=1 a_row0) plus the fused GLU (gate|up, gelu+silu). */
    printf("\n-- prefill fp8 (w8a16) GEMM / GEMM_GLU --\n");
    test_gemm_fp8("gemm_fp8 m64 n256 k3840 (q_proj-ish)", 64, 256, 3840, 0);
    test_gemm_fp8("gemm_fp8 m200 n512 k3840 (M-ragged)", 200, 512, 3840, 0);
    test_gemm_fp8("gemm_fp8 m1 n300 k3840 a_row0=63 (lm_head-ish)", 1, 300, 3840, 63);
    test_gemm_fp8("gemm_fp8 m128 n3840 k15360 (down-ish)", 128, 3840, 15360, 0);
    test_gemm_glu_fp8("gemm_glu_fp8 m64 n256 k3840 gelu", 64, 256, 3840, PLOW_ACT_GELU_TANH_);
    test_gemm_glu_fp8("gemm_glu_fp8 m200 n512 k3840 silu (M-ragged)", 200, 512, 3840, PLOW_ACT_SILU_);

    /* PREFILL fp8 (w8a8) — T7 L2. BOTH operands e4m3 + per-row act quant. Same q/k/v/o/down/lm_head
     * shapes; e4m3-aware tol (dequant-both gate 5e-3; quant-err vs full-precision printed, not gated). */
    printf("\n-- prefill fp8 (w8a8) QUANT / GEMM / GEMM_GLU --\n");
    test_quant_fp8("quant_fp8 m64 k3840", 64, 3840);
    test_quant_fp8("quant_fp8 m200 k4096 (M-ragged)", 200, 4096);
    test_gemm_w8a8("gemm_w8a8 m64 n256 k3840 (q_proj-ish)", 64, 256, 3840, 0);
    test_gemm_w8a8("gemm_w8a8 m200 n512 k3840 (M-ragged)", 200, 512, 3840, 0);
    test_gemm_w8a8("gemm_w8a8 m1 n300 k3840 a_row0=63 (lm_head-ish)", 1, 300, 3840, 63);
    test_gemm_w8a8("gemm_w8a8 m128 n3840 k15360 (down-ish)", 128, 3840, 15360, 0);
    test_gemm_glu_w8a8("gemm_glu_w8a8 m64 n256 k3840 gelu", 64, 256, 3840, PLOW_ACT_GELU_TANH_);
    test_gemm_glu_w8a8("gemm_glu_w8a8 m200 n512 k3840 silu (M-ragged)", 200, 512, 3840, PLOW_ACT_SILU_);

    /* 31B dims: hidden H=5376 (K for q/k/v/o/gate/up), inter I=21504 (K for down, N for gate/up).
     * All K,N multiples of the fp8 BK8=64 / BN tile — charges the w8a8 mma on the real 31B shapes. */
    printf("\n-- prefill fp8 (w8a8) — 31B shapes (H=5376 I=21504) --\n");
    /* M kept small on the big-N/K 31B tiles: the host golden is O(m*n*k) and M is only rows —
     * the 31B K/N dims (5376/21504) are what charge the w8a8 mma fragment map + epilogue. */
    test_quant_fp8("quant_fp8 m64 k5376 (31B hidden act)", 64, 5376);
    test_quant_fp8("quant_fp8 m32 k21504 (31B down act, M-ragged)", 32, 21504);
    test_gemm_w8a8("gemm_w8a8 m64 n5376 k5376 (31B o_proj-ish)", 64, 5376, 5376, 0);
    test_gemm_w8a8("gemm_w8a8 m40 n5376 k21504 (31B down, M-ragged)", 40, 5376, 21504, 0);
    test_gemm_w8a8("gemm_w8a8 m1 n300 k5376 a_row0=63 (31B lm_head-ish)", 1, 300, 5376, 63);
    test_gemm_glu_w8a8("gemm_glu_w8a8 m32 n21504 k5376 gelu (31B gate/up)", 32, 21504, 5376, PLOW_ACT_GELU_TANH_);
    test_gemm_glu_w8a8("gemm_glu_w8a8 m40 n21504 k5376 gelu (31B gate/up, M-ragged)", 40, 21504, 5376, PLOW_ACT_GELU_TANH_);

    printf("\n%s\n", g_fail ? "sm120_interp_op_test: FAIL" : "sm120_interp_op_test: ok");
    return g_fail;
}
