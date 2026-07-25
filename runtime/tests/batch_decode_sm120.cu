/* batch_decode_sm120.cu — numeric oracle for BATCH>1 DECODE on sm_120 (serving pending #4).
 *
 * Proves the batch>1 decode kernel layer is CORRECT and has NO cross-sequence bleed, against an
 * inline f32 CPU reference. These are the exact device templates the megakernel dispatches
 * (this TU #includes the same op_*.cuh), so a PASS is a statement about the interpreter's batch
 * arithmetic. Cases:
 *
 *   G  batched GEMV family (d_gemv / d_gemv_qkv / d_gemv_glu) at M in {1,2,4,8} vs f32 ref —
 *      EVERY row written and correct.  NEGATIVE CONTROL: gemv_rows<1> at M=2 leaves row 1 at the
 *      sentinel (the historical blocker #2 bug), so the checker's "written" test is trustworthy.
 *   F  batched d_flash_decode with per-batch KV rings [B][kv_head][ring][D], DIFFERENT data and
 *      DIFFERENT kvlen per sequence, vs a per-sequence ref. ISOLATION: zeroing sequence 1's ring
 *      must not move sequence 0's output (no bleed).
 *   H  batched d_headnorm_rope KV write: two sequences at DIFFERENT positions each land in their
 *      OWN ring at their OWN pos, vs ref.
 *   T  the PLOW_NV_KVBOUNDS trap: flash_decode with n_batch=2 but a KV capacity sized for 1
 *      TRAPS (the b>=1 OOB the fix closes); sized for 2 it runs clean. Only when built with
 *      -DPLOW_NV_KVBOUNDS=1.
 *
 * Build:
 *   nvcc -std=c++17 -arch=sm_120a -Iruntime/common -Iruntime/nvidia -include cstdint \
 *     runtime/tests/batch_decode_sm120.cu -o batch_decode
 *   nvcc ... -DPLOW_NV_KVBOUNDS=1 ... -o batch_decode_trap   (arms case T)
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sm120_common.cuh"   /* pulls op_attention.cuh */
#include "op_norm.cuh"
#include "op_elementwise.cuh"
#include "op_gemm.cuh"

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

static const unsigned GRID = 170, BLOCK = 256;
static const unsigned short SENTINEL = 0x7A5A;

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
static bf16* dev_bf_sentinel(size_t n){ bf16* d=nullptr; CK(cudaMalloc(&d,n*sizeof(bf16)));
    std::vector<unsigned short> s(n, SENTINEL);
    CK(cudaMemcpy(d,s.data(),n*sizeof(bf16),cudaMemcpyHostToDevice)); return d; }
static int* to_dev_i(const std::vector<int>& h){ int* d=nullptr; CK(cudaMalloc(&d,h.size()*4));
    CK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static std::vector<unsigned short> raw_dev(const bf16* d, size_t n){
    std::vector<unsigned short> t(n); CK(cudaMemcpy(t.data(),d,n*sizeof(bf16),cudaMemcpyDeviceToHost));
    return t; }
static std::vector<float> from_dev(const bf16* d, size_t n){
    auto t = raw_dev(d,n); std::vector<float> h(n);
    for(size_t i=0;i<n;i++) h[i]=__bfloat162float(*(bf16*)&t[i]); return h; }

static int g_fail = 0;
/* relL2 over ROWS [r0,r1); also reports whether every element in the range differs from SENTINEL. */
static void report_rows(const char* name, const std::vector<float>& ref,
                        const bf16* dC, unsigned M, unsigned N,
                        unsigned r0, unsigned r1, bool expect_written, double tol=3e-2) {
    auto raw = raw_dev(dC, (size_t)M*N);
    double num=0,den=0; bool all_written=true; size_t nan=0;
    for (unsigned m=r0;m<r1;m++) for (unsigned n=0;n<N;n++){
        unsigned short bits = raw[(size_t)m*N+n];
        if (bits==SENTINEL) all_written=false;
        double g=__bfloat162float(*(bf16*)&bits), r=ref[(size_t)m*N+n];
        if (std::isnan(g)||std::isinf(g)){ nan++; continue; }
        double d=g-r; num+=d*d; den+=r*r;
    }
    double relL2 = den>0? sqrt(num/den):(num>0?1e30:0);
    bool numeric_ok = (relL2<=tol) && nan==0;
    bool pass = (all_written==expect_written) && (expect_written ? numeric_ok : true);
    printf("  %-34s rows[%u,%u) written=%d relL2=%-10.4g%s -> %s\n",
           name, r0, r1, all_written, relL2, nan?" NaN!":"",
           pass?"PASS":"FAIL");
    if (!pass) g_fail=1;
}

/* ============================ G: batched GEMV family ============================ */
__global__ void k_gemv(bf16* C, const bf16* x, const bf16* W, unsigned M, unsigned N, unsigned K){
    d_gemv(C, x, W, M, N, K, blockIdx.x, gridDim.x);
}
__global__ void k_gemv_rows1(bf16* C, const bf16* x, const bf16* W, unsigned M, unsigned N, unsigned K){
    gemv_rows<1>(C, x, W, M, N, K, blockIdx.x, gridDim.x); /* the OLD (broken) d_gemv body */
}
__global__ void k_gemv_qkv(bf16* Cq, bf16* Ck, bf16* Cv, const bf16* x, const bf16* Wq,
                           const bf16* Wk, const bf16* Wv, unsigned M, unsigned Nq, unsigned Nk,
                           unsigned Nv, unsigned K){
    d_gemv_qkv(Cq,Ck,Cv,x,Wq,Wk,Wv,M,Nq,Nk,Nv,K,blockIdx.x,gridDim.x);
}
__global__ void k_gemv_glu(bf16* C, const bf16* x, const bf16* Wg, const bf16* Wu,
                           unsigned M, unsigned N, unsigned K, unsigned act){
    d_gemv_glu(C,x,Wg,Wu,M,N,K,act,blockIdx.x,gridDim.x);
}
static void ref_gemv(std::vector<float>& out, const std::vector<float>& x,
                     const std::vector<float>& W, unsigned M, unsigned N, unsigned K){
    out.assign((size_t)M*N,0.f);
    for (unsigned m=0;m<M;m++) for (unsigned n=0;n<N;n++){
        double a=0; for (unsigned k=0;k<K;k++) a += (double)x[(size_t)m*K+k]*W[(size_t)n*K+k];
        out[(size_t)m*N+n]=(float)a;
    }
}
static void test_gemv(unsigned M){
    seed(0x100u+M);
    const unsigned N=512, K=2560;
    auto x=gen_bf16((size_t)M*K,0.1f), W=gen_bf16((size_t)N*K,0.1f);
    std::vector<float> ref; ref_gemv(ref,x,W,M,N,K);
    bf16 *dx=to_dev(x),*dW=to_dev(W),*dC=dev_bf_sentinel((size_t)M*N);
    k_gemv<<<GRID,BLOCK>>>(dC,dx,dW,M,N,K); CK(cudaDeviceSynchronize());
    char lbl[64]; snprintf(lbl,64,"d_gemv M=%u",M);
    report_rows(lbl, ref, dC, M, N, 0, M, true);
    cudaFree(dx);cudaFree(dW);cudaFree(dC);
}
static void test_gemv_negctrl(){
    seed(0x777u);
    const unsigned M=2,N=512,K=2560;
    auto x=gen_bf16((size_t)M*K,0.1f), W=gen_bf16((size_t)N*K,0.1f);
    std::vector<float> ref; ref_gemv(ref,x,W,M,N,K);
    bf16 *dx=to_dev(x),*dW=to_dev(W),*dC=dev_bf_sentinel((size_t)M*N);
    k_gemv_rows1<<<GRID,BLOCK>>>(dC,dx,dW,M,N,K); CK(cudaDeviceSynchronize());
    /* row 0 correct+written; row 1 UNWRITTEN (the bug the fix removes). expect_written=false. */
    report_rows("NEGCTRL gemv_rows<1> row0", ref, dC, M, N, 0, 1, true);
    report_rows("NEGCTRL gemv_rows<1> row1", ref, dC, M, N, 1, 2, false);
    cudaFree(dx);cudaFree(dW);cudaFree(dC);
}
static void test_qkv(unsigned M){
    seed(0x200u+M);
    const unsigned Nq=512,Nk=128,Nv=128,K=2560;
    auto x=gen_bf16((size_t)M*K,0.1f);
    auto Wq=gen_bf16((size_t)Nq*K,0.1f),Wk=gen_bf16((size_t)Nk*K,0.1f),Wv=gen_bf16((size_t)Nv*K,0.1f);
    std::vector<float> rq,rk,rv; ref_gemv(rq,x,Wq,M,Nq,K); ref_gemv(rk,x,Wk,M,Nk,K); ref_gemv(rv,x,Wv,M,Nv,K);
    bf16 *dx=to_dev(x),*dWq=to_dev(Wq),*dWk=to_dev(Wk),*dWv=to_dev(Wv);
    bf16 *dCq=dev_bf_sentinel((size_t)M*Nq),*dCk=dev_bf_sentinel((size_t)M*Nk),*dCv=dev_bf_sentinel((size_t)M*Nv);
    k_gemv_qkv<<<GRID,BLOCK>>>(dCq,dCk,dCv,dx,dWq,dWk,dWv,M,Nq,Nk,Nv,K); CK(cudaDeviceSynchronize());
    char lbl[64]; snprintf(lbl,64,"d_gemv_qkv M=%u (q)",M); report_rows(lbl,rq,dCq,M,Nq,0,M,true);
    snprintf(lbl,64,"d_gemv_qkv M=%u (v)",M); report_rows(lbl,rv,dCv,M,Nv,0,M,true);
    cudaFree(dx);cudaFree(dWq);cudaFree(dWk);cudaFree(dWv);cudaFree(dCq);cudaFree(dCk);cudaFree(dCv);
}
static void test_glu(unsigned M){
    seed(0x300u+M);
    const unsigned N=512,K=2560;
    auto x=gen_bf16((size_t)M*K,0.1f),Wg=gen_bf16((size_t)N*K,0.1f),Wu=gen_bf16((size_t)N*K,0.1f);
    std::vector<float> g,u; ref_gemv(g,x,Wg,M,N,K); ref_gemv(u,x,Wu,M,N,K);
    std::vector<float> ref((size_t)M*N);
    for(size_t i=0;i<ref.size();i++){ float gg=g[i]; float a=0.5f*gg*(1.f+tanhf(0.7978845608f*(gg+0.044715f*gg*gg*gg))); ref[i]=a*u[i]; }
    bf16 *dx=to_dev(x),*dWg=to_dev(Wg),*dWu=to_dev(Wu),*dC=dev_bf_sentinel((size_t)M*N);
    k_gemv_glu<<<GRID,BLOCK>>>(dC,dx,dWg,dWu,M,N,K,/*act=GeGLU*/0); CK(cudaDeviceSynchronize());
    char lbl[64]; snprintf(lbl,64,"d_gemv_glu M=%u",M);
    report_rows(lbl,ref,dC,M,N,0,M,true,5e-2);
    cudaFree(dx);cudaFree(dWg);cudaFree(dWu);cudaFree(dC);
}

/* ============================ F: batched flash-decode ============================ */
template<int D,int GF>
__global__ void k_flash(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
                        const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
                        float scale, unsigned nsplit, unsigned kv_cap){
    extern __shared__ float arena[];
    d_flash_decode<D,GF>(Op,Ml,Q,K,V,kvlen,nb,nh,nkv,kvs,0/*full*/,scale,nsplit,0xFFFFFFFFu,
                         blockIdx.x,gridDim.x,arena,kv_cap);
}
template<int D>
__global__ void k_merge(bf16* O, const float* Op, const float* Ml, unsigned nb, unsigned nh, unsigned nsplit){
    d_flash_merge<D>(O,Op,Ml,nb,nh,nsplit,blockIdx.x,gridDim.x);
}
/* Per-sequence ref: sequence b attends its query row against ITS ring for kvlen[b] rows. */
static void ref_flash(std::vector<float>& out, const std::vector<float>& Q, const std::vector<float>& K,
                      const std::vector<float>& V, const std::vector<int>& kvlen, unsigned B,
                      unsigned nh, unsigned nkv, unsigned kvs, unsigned D, float scale){
    out.assign((size_t)B*nh*D,0.f);
    const unsigned gqa=nh/nkv;
    for(unsigned b=0;b<B;b++) for(unsigned h=0;h<nh;h++){
        const unsigned hkv=h/gqa; const unsigned len=(unsigned)kvlen[b];
        const float* q=&Q[((size_t)b*nh+h)*D];
        const float* kb=&K[((size_t)b*nkv+hkv)*kvs*D];
        const float* vb=&V[((size_t)b*nkv+hkv)*kvs*D];
        std::vector<float> s(len); float mx=-INFINITY;
        for(unsigned r=0;r<len;r++){ double d=0; for(unsigned e=0;e<D;e++) d+=(double)q[e]*kb[(size_t)r*D+e];
            s[r]=(float)d*scale; if(s[r]>mx)mx=s[r]; }
        float sum=0; for(unsigned r=0;r<len;r++){ s[r]=expf(s[r]-mx); sum+=s[r]; }
        for(unsigned e=0;e<D;e++){ double a=0; for(unsigned r=0;r<len;r++) a+=(s[r]/sum)*vb[(size_t)r*D+e];
            out[((size_t)b*nh+h)*D+e]=(float)a; }
    }
}
template<int D,int GF>
static void test_flash(unsigned nh, unsigned nkv){
    seed(0x2000u+D+nh);
    const unsigned B=2, kvs=96, nsplit=4;
    const float scale=1.f/sqrtf((float)D);
    std::vector<int> kvlen={ (int)80, (int)57 };   /* DIFFERENT lengths per sequence */
    auto Q=gen_bf16((size_t)B*nh*D,1.0f);
    auto K=gen_bf16((size_t)B*nkv*kvs*D,1.0f);
    auto V=gen_bf16((size_t)B*nkv*kvs*D,1.0f);
    std::vector<float> ref; ref_flash(ref,Q,K,V,kvlen,B,nh,nkv,kvs,D,scale);
    bf16 *dQ=to_dev(Q),*dK=to_dev(K),*dV=to_dev(V); int* dL=to_dev_i(kvlen);
    float *dOp=dev_f((size_t)B*nh*nsplit*D),*dMl=dev_f((size_t)B*nh*nsplit*2);
    bf16* dO=dev_bf_sentinel((size_t)B*nh*D);
    const unsigned n_work=B*(nh/GF)*nsplit;
    const size_t smem=(size_t)FA_DEC_SMEM_FLOATS(D,GF)*sizeof(float);
    const unsigned cap=B*nkv*kvs; /* correctly-sized capacity */
    k_flash<D,GF><<<n_work,256,smem>>>(dOp,dMl,dQ,dK,dV,dL,B,nh,nkv,kvs,scale,nsplit,cap);
    CK(cudaDeviceSynchronize());
    k_merge<D><<<B*nh,256>>>(dO,dOp,dMl,B,nh,nsplit); CK(cudaDeviceSynchronize());
    char lbl[80]; snprintf(lbl,80,"flash_decode D=%d B=2 per-seq",D);
    report_rows(lbl, ref, dO, B*nh, D, 0, B*nh, true, 3e-2);

    /* ISOLATION: zero sequence 1's rings, rerun — sequence 0's output must be UNCHANGED. */
    auto K2=K, V2=V;
    for(size_t i=(size_t)1*nkv*kvs*D;i<(size_t)2*nkv*kvs*D;i++){ K2[i]=0; V2[i]=0; }
    std::vector<float> ref0; ref_flash(ref0,Q,K2,V2,kvlen,B,nh,nkv,kvs,D,scale);
    bf16 *dK2=to_dev(K2),*dV2=to_dev(V2);
    CK(cudaMemset(dOp,0,(size_t)B*nh*nsplit*D*sizeof(float)));
    CK(cudaMemset(dMl,0,(size_t)B*nh*nsplit*2*sizeof(float)));
    k_flash<D,GF><<<n_work,256,smem>>>(dOp,dMl,dQ,dK2,dV2,dL,B,nh,nkv,kvs,scale,nsplit,cap);
    CK(cudaDeviceSynchronize());
    k_merge<D><<<B*nh,256>>>(dO,dOp,dMl,B,nh,nsplit); CK(cudaDeviceSynchronize());
    snprintf(lbl,80,"flash_decode D=%d seq0 unbled by zeroing seq1",D);
    report_rows(lbl, ref0, dO, B*nh, D, 0, nh, true, 3e-2); /* only seq 0's nh rows */
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dL);cudaFree(dOp);cudaFree(dMl);cudaFree(dO);
    cudaFree(dK2);cudaFree(dV2);
}

/* ============================ H: batched headnorm KV write ============================ */
template<int HD>
__global__ void k_hnr_batch(bf16* out, const bf16* x, const bf16* gamma, const float* cosb,
                            const float* sinb, const int* pos, unsigned ntok, unsigned nhead,
                            float eps, unsigned ring, unsigned mask, unsigned nb){
    d_headnorm_rope<HD>(out,x,gamma,cosb,sinb,pos,ntok,nhead,eps,
                        /*out_row0*/0,/*out_stride*/ring,/*kv_mask*/mask,/*skip_norm*/0,
                        blockIdx.x,gridDim.x,nb);
}
template<int HD>
static void test_headnorm_batch(){
    seed(0x3000u+HD);
    const unsigned B=2, nkv=2, ring=64, mask=ring-1; const float eps=1e-6f,theta=1e6f;
    const unsigned H2=HD/2;
    std::vector<int> pos={ 40, 13 };                 /* different positions per sequence */
    std::vector<float> X=gen_bf16((size_t)B*nkv*HD,1.0f), gam=gen_bf16(HD,1.0f);
    unsigned maxpos=0; for(int p:pos) if((unsigned)p>maxpos)maxpos=p;
    std::vector<float> cosb((size_t)(maxpos+1)*H2),sinb((size_t)(maxpos+1)*H2);
    for(unsigned p=0;p<=maxpos;p++) for(unsigned i=0;i<H2;i++){
        float f=powf(theta,-2.f*(float)i/(float)HD),a=(float)p*f;
        cosb[(size_t)p*H2+i]=cosf(a); sinb[(size_t)p*H2+i]=sinf(a); }
    /* ref cache [B][nkv][ring][HD], zeroed; write each (b,h) at row pos[b]&mask. */
    std::vector<float> ref((size_t)B*nkv*ring*HD,0.f);
    for(unsigned b=0;b<B;b++) for(unsigned h=0;h<nkv;h++){
        const float* xr=&X[((size_t)b*nkv+h)*HD];
        std::vector<float> v(HD); double ss=0; for(unsigned d=0;d<HD;d++) ss+=(double)xr[d]*xr[d];
        float inv=1.f/sqrtf((float)(ss/(double)HD)+eps);
        for(unsigned d=0;d<HD;d++) v[d]=xr[d]*inv*gam[d];
        for(unsigned i=0;i<H2;i++){ float c=cosb[(size_t)pos[b]*H2+i],s=sinb[(size_t)pos[b]*H2+i];
            float x0=v[i],x1=v[i+H2]; v[i]=x0*c-x1*s; v[i+H2]=x0*s+x1*c; }
        unsigned row=(unsigned)pos[b]&mask;
        for(unsigned d=0;d<HD;d++) ref[(((size_t)b*nkv+h)*ring+row)*HD+d]=v[d];
    }
    bf16 *dX=to_dev(X),*dG=to_dev(gam),*dO=dev_bf_sentinel((size_t)B*nkv*ring*HD);
    CK(cudaMemset(dO,0,(size_t)B*nkv*ring*HD*sizeof(bf16)));
    float *dC=to_dev_f(cosb),*dS=to_dev_f(sinb); int* dP=to_dev_i(pos);
    k_hnr_batch<HD><<<64,256>>>(dO,dX,dG,dC,dS,dP,B,nkv,eps,ring,mask,B);
    CK(cudaDeviceSynchronize());
    /* compare only the two written rows (rest is zero in both). */
    auto got=from_dev(dO,(size_t)B*nkv*ring*HD);
    double num=0,den=0;
    for(unsigned b=0;b<B;b++) for(unsigned h=0;h<nkv;h++){ unsigned row=(unsigned)pos[b]&mask;
        for(unsigned d=0;d<HD;d++){ size_t idx=(((size_t)b*nkv+h)*ring+row)*HD+d;
            double dd=got[idx]-ref[idx]; num+=dd*dd; den+=ref[idx]*ref[idx]; } }
    double relL2=den>0?sqrt(num/den):0;
    char lbl[64]; snprintf(lbl,64,"headnorm_rope KV-write HD=%d B=2",HD);
    printf("  %-34s relL2=%-10.4g -> %s\n",lbl,relL2, relL2<=3e-2?"PASS":"FAIL");
    if(relL2>3e-2) g_fail=1;
    cudaFree(dX);cudaFree(dG);cudaFree(dO);cudaFree(dC);cudaFree(dS);cudaFree(dP);
}

/* ============================ T: the OOB bounds trap ============================ */
#if PLOW_NV_KVBOUNDS
static void test_trap(){
    seed(0x4000u);
    const unsigned B=2,nh=8,nkv=2,D=128,kvs=64,nsplit=4; const float scale=1.f/sqrtf((float)D);
    std::vector<int> kvlen={ (int)40,(int)40 };
    auto Q=gen_bf16((size_t)B*nh*D,1.f);
    /* KV allocated for ONE sequence only (the historical under-allocation). */
    auto K=gen_bf16((size_t)1*nkv*kvs*D,1.f), V=gen_bf16((size_t)1*nkv*kvs*D,1.f);
    bf16 *dQ=to_dev(Q),*dK=to_dev(K),*dV=to_dev(V); int* dL=to_dev_i(kvlen);
    float *dOp=dev_f((size_t)B*nh*nsplit*D),*dMl=dev_f((size_t)B*nh*nsplit*2);
    const unsigned n_work=B*(nh/2)*nsplit; const size_t smem=(size_t)FA_DEC_SMEM_FLOATS(D,2)*sizeof(float);
    /* cap sized for ONE sequence -> the b=1 read must TRAP. */
    const unsigned cap_bad=1*nkv*kvs;
    k_flash<D,2><<<n_work,256,smem>>>(dOp,dMl,dQ,dK,dV,dL,B,nh,nkv,kvs,scale,nsplit,cap_bad);
    cudaError_t e=cudaDeviceSynchronize();
    printf("  %-34s n_batch=2, cap_for_1 -> %s\n","KVBOUNDS trap (under-alloc)",
           e!=cudaSuccess ? "TRAPPED (PASS)" : "NO TRAP (FAIL)");
    if(e==cudaSuccess) g_fail=1;
    /* reset device after the trap so the clean run is valid. */
    cudaGetLastError(); cudaDeviceReset();
}
#endif

int main(){
#if PLOW_NV_KVBOUNDS
    printf("=== batch decode oracle: KVBOUNDS TRAP build ===\n");
    test_trap();
#else
    printf("=== batch decode oracle (sm_120, serving pending #4) ===\n");
    printf("-- G: batched GEMV family vs f32 ref --\n");
    test_gemv_negctrl();
    /* B>8 (PLOW_DECODE_BATCH up to 32): 16/32 are block-walk multiples, 5/17 ragged
     * remainders. Before the walk, d_gemv_qkv/d_gemv_glu fed M>8 to gemv_*_rows<8> and left
     * rows 8..M-1 UNWRITTEN. */
    for(unsigned M: {1u,2u,4u,8u,16u,32u,5u,17u}) test_gemv(M);
    for(unsigned M: {1u,2u,4u,8u,16u,32u,5u,17u}) test_qkv(M);
    for(unsigned M: {1u,2u,4u,8u,16u,32u,5u,17u}) test_glu(M);
    printf("-- F: batched flash-decode, per-sequence KV rings --\n");
    test_flash<128,4>(8,2);
    test_flash<256,2>(4,2);
    test_flash<512,2>(4,1);
    printf("-- H: batched headnorm KV write --\n");
    test_headnorm_batch<256>();
    test_headnorm_batch<512>();
#endif
    printf(g_fail? "\nRESULT: FAIL\n" : "\nRESULT: PASS\n");
    return g_fail;
}
