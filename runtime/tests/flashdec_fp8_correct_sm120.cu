/* flashdec_fp8_correct_sm120.cu — NUMERIC ORACLE for the fp8 (e4m3-KV) flash-DECODE arm.
 *
 * sm120_interp_op_test covers only d_flash_decode<HD,GF> with FP8KV=false (bf16 KV). This harness
 * closes that gap for beat26b-flashdec: it exercises d_flash_decode<...,FP8KV=true> (+ d_flash_merge)
 * against an f32 CPU reference that dequantizes the SAME e4m3 bytes the kernel reads. Compiled twice:
 *   default        -> shipped fp8 arm (e4m3 -> bf16 -> f32 double round-trip)
 *   -DPLOW_FP8_FAST -> new arm (e4m3 -> f32 direct, 8B loads)
 * Both must pass the e4m3 budget; FAST should be at least as tight (it drops one rounding).
 *
 * Build:
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -I runtime/common -I runtime/nvidia -include cstdint runtime/tests/flashdec_fp8_correct_sm120.cu -o fd_correct
 *   ... add -DPLOW_FP8_FAST for the fast arm.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "sm120_common.cuh"

typedef __nv_bfloat16 bf16;
#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA ERR %s: %s\n",#x,cudaGetErrorString(e_));exit(2);} } while(0)

static uint32_t rs=0x1234567u;
static float rnd(){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return (float)((int32_t)rs)/2147483648.0f; }
static uint8_t e4m3_enc(float v){ __nv_fp8_e4m3 q(v); return *(const uint8_t*)&q; }
static float   e4m3_dec(uint8_t b){ __nv_fp8_e4m3 q; *(uint8_t*)&q=b; return (float)q; }

template<int D,int GF>
__global__ void k_flash_fp8(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
                            const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
                            float scale, unsigned nsplit, const float* ks, const float* vs){
    extern __shared__ float arena[];
    d_flash_decode<D,GF,true>(Op,Ml,Q,K,V,kvlen,nb,nh,nkv,kvs,0u,scale,nsplit,0xFFFFFFFFu,
                              blockIdx.x,gridDim.x,arena,0u,ks,vs);
}
template<int D>
__global__ void k_merge(bf16* O, const float* Op, const float* Ml, unsigned nb, unsigned nh, unsigned ns){
    d_flash_merge<D>(O,Op,Ml,nb,nh,ns,blockIdx.x,gridDim.x);
}

static int g_fail=0;
template<int D,int GF>
static void test(unsigned nh, unsigned nkv, unsigned len, unsigned nsplit){
    const unsigned nb=1, kvs=len; const float scale=1.f/sqrtf((float)D);
    const unsigned gqa=nh/nkv;
    /* f32 K/V, quantized per-row to e4m3 with scale = max|row|/448 (the write-side convention). */
    std::vector<float> Qf((size_t)nh*D); for(auto&x:Qf)x=rnd()*0.6f;
    std::vector<bf16> Qb(Qf.size()); for(size_t i=0;i<Qf.size();i++)Qb[i]=__float2bfloat16(Qf[i]);
    const size_t nrows=(size_t)nkv*kvs;
    std::vector<uint8_t> Kq(nrows*D), Vq(nrows*D);
    std::vector<float> Ks(nrows), Vs(nrows);
    std::vector<float> Kdq(nrows*D), Vdq(nrows*D); /* dequantized (what BOTH kernel and ref see) */
    for(size_t r=0;r<nrows;r++){
        float amk=1e-6f, amv=1e-6f; std::vector<float> kr(D),vr(D);
        for(unsigned d=0;d<D;d++){ kr[d]=rnd()*0.8f; vr[d]=rnd()*0.8f; amk=fmaxf(amk,fabsf(kr[d])); amv=fmaxf(amv,fabsf(vr[d])); }
        float sk=amk/448.f, sv=amv/448.f; Ks[r]=sk; Vs[r]=sv;
        for(unsigned d=0;d<D;d++){ uint8_t kb=e4m3_enc(kr[d]/sk), vb=e4m3_enc(vr[d]/sv);
            Kq[r*D+d]=kb; Vq[r*D+d]=vb; Kdq[r*D+d]=e4m3_dec(kb)*sk; Vdq[r*D+d]=e4m3_dec(vb)*sv; }
    }
    /* f32 reference attention on the dequantized cache. */
    std::vector<float> ref((size_t)nh*D,0.f);
    for(unsigned h=0;h<nh;h++){ unsigned hkv=h/gqa; const float* q=&Qf[(size_t)h*D];
        const float* kb=&Kdq[(size_t)hkv*kvs*D]; const float* vb=&Vdq[(size_t)hkv*kvs*D];
        std::vector<float> s(len); float mx=-1e30f;
        for(unsigned r=0;r<len;r++){ double d=0; for(unsigned e=0;e<D;e++)d+=(double)__bfloat162float(Qb[(size_t)h*D+e])*kb[(size_t)r*D+e];
            s[r]=(float)d*scale; if(s[r]>mx)mx=s[r]; }
        float sum=0; for(unsigned r=0;r<len;r++){ s[r]=expf(s[r]-mx); sum+=s[r]; }
        for(unsigned e=0;e<D;e++){ double a=0; for(unsigned r=0;r<len;r++)a+=(s[r]/sum)*vb[(size_t)r*D+e]; ref[(size_t)h*D+e]=(float)a; }
    }
    /* device */
    bf16* dQ; CK(cudaMalloc(&dQ,Qb.size()*2)); CK(cudaMemcpy(dQ,Qb.data(),Qb.size()*2,cudaMemcpyHostToDevice));
    void* dK; void* dV; CK(cudaMalloc(&dK,Kq.size())); CK(cudaMalloc(&dV,Vq.size()));
    CK(cudaMemcpy(dK,Kq.data(),Kq.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dV,Vq.data(),Vq.size(),cudaMemcpyHostToDevice));
    float *dKs,*dVs; CK(cudaMalloc(&dKs,nrows*4)); CK(cudaMalloc(&dVs,nrows*4));
    CK(cudaMemcpy(dKs,Ks.data(),nrows*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dVs,Vs.data(),nrows*4,cudaMemcpyHostToDevice));
    int L=(int)len; int* dL; CK(cudaMalloc(&dL,4)); CK(cudaMemcpy(dL,&L,4,cudaMemcpyHostToDevice));
    float *dOp,*dMl; CK(cudaMalloc(&dOp,(size_t)nh*nsplit*D*4)); CK(cudaMalloc(&dMl,(size_t)nh*nsplit*2*4));
    bf16* dO; CK(cudaMalloc(&dO,(size_t)nh*D*2));
    const unsigned n_work=nb*(nh/GF)*nsplit; const size_t smem=(size_t)FA_DEC_SMEM_FLOATS(D,GF)*sizeof(float);
    CK(cudaFuncSetAttribute(k_flash_fp8<D,GF>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    k_flash_fp8<D,GF><<<n_work,256,smem>>>(dOp,dMl,dQ,(bf16*)dK,(bf16*)dV,dL,nb,nh,nkv,kvs,scale,nsplit,dKs,dVs);
    CK(cudaDeviceSynchronize());
    k_merge<D><<<nb*nh,256>>>(dO,dOp,dMl,nb,nh,nsplit); CK(cudaDeviceSynchronize());
    std::vector<unsigned short> got(nh*D); CK(cudaMemcpy(got.data(),dO,(size_t)nh*D*2,cudaMemcpyDeviceToHost));
    double num=0,den=0; for(size_t i=0;i<(size_t)nh*D;i++){ float g=__bfloat162float(*(bf16*)&got[i]); double d=g-ref[i]; num+=d*d; den+=ref[i]*ref[i]; }
    double relL2=den>0?sqrt(num/den):0;
#ifdef PLOW_FP8_FAST
    const char* arm="FAST(f32)";
#else
    const char* arm="shipped(bf16rt)";
#endif
    bool pass=relL2<=2e-2;
    printf("  fp8 flash D=%d GF=%d nh=%u nkv=%u len=%u ns=%u  [%s]  relL2=%.4g -> %s\n",
           D,GF,nh,nkv,len,nsplit,arm,relL2,pass?"PASS":"FAIL");
    if(!pass)g_fail=1;
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dKs);cudaFree(dVs);cudaFree(dL);cudaFree(dOp);cudaFree(dMl);cudaFree(dO);
}

int main(){
    printf("=== fp8 flash-decode numeric oracle (Gemma-4-26B full-attn hd512 GF4 + others) ===\n");
    test<512,4>(16,2,300,4);   /* the 26B full-attn geometry */
    test<512,4>(16,2,777,8);
    test<512,2>(4,1,200,4);
    test<256,2>(4,2,150,4);    /* sliding geometry sanity */
    printf(g_fail?"\nRESULT: FAIL\n":"\nRESULT: PASS\n");
    return g_fail;
}
