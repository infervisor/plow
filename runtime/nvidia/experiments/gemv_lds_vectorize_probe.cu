// gemv_lds_vectorize_probe.cu — decode-GEMV shared-load (LDS.U16) vectorization probe (sm_90a).
//
//   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin \
//     nvcc -arch=sm_90a -O3 -o /tmp/gemv_lds gemv_lds_vectorize_probe.cu && \
//     flock /tmp/plow_gpu.lock /tmp/gemv_lds
//
// QUESTION it answers -----------------------------------------------------------------------
// SASS of the built decode cubin shows 13,824 `LDS.U16` (2-byte scalar shared loads) vs 673
// `LDS.128`. The stated hypothesis was that these come from the M=1 decode GEMV inner loop
// staging x in smem and reading it 2 bytes at a time, and that vectorizing to LDS.128 would cut
// shared-load instructions 4-8x.
//
// ANSWER: **the hypothesis is REFUTED on both counts. Do NOT vectorize op_gemm.cuh.**
//
// (1) WRONG SITE. Rebuilt the decode cubin with -lineinfo and attributed every LDS by source
//     line (nvdisasm -g):
//         13824 LDS.U16  -> op_mla.cuh:87       (100%, ALL of them)
//             0 LDS.U16  -> op_gemm.cuh          (none)
//           244 LDS.128  -> op_attention.cuh:80  (= `ld_smem8`, the GEMV/flash x-tile load)
//     Every LDS.U16 in the cubin uses `[UR+IMM]` addressing — a warp-UNIFORM base plus a
//     compile-time constant. The GEMV x-tile address is `xs + kk[u]`, `kk[u] = (c+u)*GV_STEP +
//     lane*8`, which is lane-DEPENDENT and therefore can never be `[UR+IMM]`. The immediates
//     form one dense contiguous run 0x18d0..0x4cce step 2 (6656 bf16 slots) = a fully-unrolled
//     sweep of the MLA query tile (qsm/qrsm), not a GEMV.
//     Instruction-count check, exact: the MLA score loop does (DK/8=64 + DR/8=8) = 72 smem
//     vec-loads per GF per thread, fully unrolled, over instantiations GF in {4,8} x GATHER in
//     {false,true}:  72*(4+8)*2 = 1728 vec-loads -> scalarized 1728*8 = 13,824 LDS.U16. And the
//     GF=2 instantiations stay vectorized: 72*2*2 = 288 -> 288 LDS.128, which is exactly the
//     LDS.128 count attributed to op_mla.cuh:87. Both numbers land on the nose.
//     => the LDS.U16 live in the DeepSeek/GLM/Kimi MLA attention arm, which a Gemma decode never
//        executes. They are STATIC instruction counts in a cold arm of a megakernel; static SASS
//        histograms over a megakernel say nothing about the dynamic hot path.
//
// (2) GEMV IS ALREADY VECTORIZED, and cannot be made otherwise. `ld_smem8`'s uint4 load survives
//     to LDS.128 in the production cubin AND in version A of this probe. Compiled at
//     -maxrregcount = 255/64/40/32, version A emits LDS.U16=0, LDS.128=8 per kernel at every cap.
//     The real mechanism behind (1) is register pressure, not source form: `mla_dot8` ALREADY
//     uses packed `__bfloat1622float2` (op_mla.cuh:97) — i.e. the proposed fix is already in the
//     code there — and ptxas still rematerializes each bf16 from smem at GF=4/8 (84/128 regs) to
//     relieve pressure, while leaving GF=2 (62 regs) vectorized. op_mla.cuh:89-96 already records
//     that this exact vectorization was tried against this exact plateau and measured at ~1%.
//
// This probe is the A/B that backs (2): version A = the production loop verbatim (ld_smem8 +
// dot8/dot8_fp8); version B = smem x-tile read as int4 and converted as __nv_bfloat162 pairs.
// Both emit LDS.128, both validate against an f32 CPU oracle, and B is worth nothing.
//
// MEASURED (H100 NVL, 132 SM, 60 MB L2, 4023 GB/s spec; 3 runs, all agree):
//   correctness   A and B relL2 IDENTICAL to 4 s.f. at every shape (1.6e-3, all PASS)
//   speedup A/B   0.999 - 1.005 across all 6 shape x dtype cells = NOISE, no measurable win
//   registers     bf16 A=60 B=60 ; fp8 A=38 B=40  (B costs +2 regs on the fp8 arm)
//   bandwidth     lm_head bf16 2767 GB/s (69% of spec) — decode GEMV is HBM-bound; LDS is off
//                 the critical path entirely, which is why vectorizing it cannot help.
// Consistent with experiments/README.md: warpspec_ab / gemv_warpspec_prod_cons already found
// BW-bound decode GEMV gains nothing from restructuring the load path.
//
// Conventions borrowed from the sm_120 harnesses (experiments/README.md): weights replicated
// past L2 (60 MB on H100 NVL) and cycled per-iter to force COLD HBM reads; blockIdx-driven
// column slicing; f32 CPU oracle for correctness.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- production constants (op_gemm.cuh / op_attention.cuh) ---------------------------------
#define NV_WARPS      8u
#define NV_LANE_MASK  31u
#define NV_WARP_SHIFT 5u
#define NV_THREADS    (NV_WARPS*32u)
#define GV_STEP       (32u*8u)   // 256 : 8 bf16/lane * 32 lanes
#define GV_UNROLL     8
#define GV_UNROLL_FP8 8

// ---- faithful copies of the production types / load helpers --------------------------------
struct bf16v8 { __nv_bfloat16 x[8]; };
__device__ __forceinline__ bf16v8 bf16v8_zero(){ bf16v8 r; *(uint4*)&r = make_uint4(0,0,0,0); return r; }
__device__ __forceinline__ bf16v8 ld_glob8(const __nv_bfloat16* p){ bf16v8 r; *(uint4*)&r=*(const uint4*)p; return r; }
__device__ __forceinline__ bf16v8 ld_smem8(const __nv_bfloat16* p){ bf16v8 r; *(uint4*)&r=*(const uint4*)p; return r; }
// dot8 / dot8_fp8 : BYTE-IDENTICAL to op_gemm.cuh — this is the LDS.U16 producer.
__device__ __forceinline__ float dot8(const bf16v8& a,const bf16v8& b,float acc){
#pragma unroll
    for(int i=0;i<8;i++) acc=fmaf(__bfloat162float(a.x[i]),__bfloat162float(b.x[i]),acc);
    return acc;
}
__device__ __forceinline__ float dot8_fp8(const uint2& w8,const bf16v8& x,float acc){
    const uint16_t* wp=(const uint16_t*)&w8;
#pragma unroll
    for(int j=0;j<4;j++){
        __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j],__NV_E4M3);
        float2 f=__half22float2(*reinterpret_cast<__half2*>(&h));
        acc=fmaf(f.x,__bfloat162float(x.x[2*j]),acc);
        acc=fmaf(f.y,__bfloat162float(x.x[2*j+1]),acc);
    }
    return acc;
}
__device__ __forceinline__ float warp_sum32(float v){
#pragma unroll
    for(int off=16;off>0;off>>=1) v+=__shfl_xor_sync(0xffffffffu,v,off,32);
    return v;
}

// ---- VERSION B smem readers : force LDS.128 -------------------------------------------------
// Read the staged x tile (8 contiguous bf16 = 16 B) as ONE int4, then convert as 4 packed
// __nv_bfloat162 pairs. The packed __bfloat1622float2 keeps the value in a 128-bit register
// quad (no per-element sub-word extraction), so ptxas emits LDS.128 for the x operand.
__device__ __forceinline__ float dot8_vecB(const bf16v8& w,const __nv_bfloat16* xs,float acc){
    int4 packed = *(const int4*)xs;                       // <-- LDS.128
    const __nv_bfloat162* xp=(const __nv_bfloat162*)&packed;
#pragma unroll
    for(int j=0;j<4;j++){
        float2 xf=__bfloat1622float2(xp[j]);
        acc=fmaf(__bfloat162float(w.x[2*j]),   xf.x, acc);
        acc=fmaf(__bfloat162float(w.x[2*j+1]), xf.y, acc);
    }
    return acc;
}
__device__ __forceinline__ float dot8_fp8_vecB(const uint2& w8,const __nv_bfloat16* xs,float acc){
    int4 packed = *(const int4*)xs;                       // <-- LDS.128
    const __nv_bfloat162* xp=(const __nv_bfloat162*)&packed;
    const uint16_t* wp=(const uint16_t*)&w8;
#pragma unroll
    for(int j=0;j<4;j++){
        __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)wp[j],__NV_E4M3);
        float2 f=__half22float2(*reinterpret_cast<__half2*>(&h));
        float2 xf=__bfloat1622float2(xp[j]);
        acc=fmaf(f.x,xf.x,acc);
        acc=fmaf(f.y,xf.y,acc);
    }
    return acc;
}

// ---- kernels : x staged in smem, weight streamed from gmem, warp-shuffle reduce ------------
// bf16.  VER: 0 = version A (ld_smem8+dot8), 1 = version B (int4+bfloat162).
template<int VER>
__global__ void gemv_bf16(__nv_bfloat16* __restrict__ C,const __nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ W,unsigned N,unsigned K){
    extern __shared__ __nv_bfloat16 xs[];
    for(unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
    __syncthreads();
    const unsigned lane=threadIdx.x&NV_LANE_MASK, warp=threadIdx.x>>NV_WARP_SHIFT;
    const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
    const unsigned nblk=gridDim.x, per=(N+nblk-1)/nblk, n0=blockIdx.x*per;
    const unsigned n1=(n0+per<N)?(n0+per):N;
    for(unsigned n=n0+warp;n<n1;n+=NV_WARPS){
        const __nv_bfloat16* wrow=W+(size_t)n*K;
        float acc=0.0f;
        for(unsigned c=0;c<nchunk;c+=GV_UNROLL){
            bf16v8 wv[GV_UNROLL]; unsigned kk[GV_UNROLL];
#pragma unroll
            for(int u=0;u<GV_UNROLL;u++){ unsigned k=(c+u)*GV_STEP+lane*8; kk[u]=k;
                wv[u]=(k<K)?ld_glob8(wrow+k):bf16v8_zero(); }
#pragma unroll
            for(int u=0;u<GV_UNROLL;u++){ if(kk[u]>=K) continue;
                acc = VER? dot8_vecB(wv[u],xs+kk[u],acc) : dot8(wv[u],ld_smem8(xs+kk[u]),acc); }
        }
        float t=warp_sum32(acc);
        if(lane==0) C[n]=__float2bfloat16(t);
    }
}
// fp8 e4m3.  W is [N,K] e4m3 bytes; scale[n] applied post-reduce.
template<int VER>
__global__ void gemv_fp8(__nv_bfloat16* __restrict__ C,const __nv_bfloat16* __restrict__ x,
                         const uint8_t* __restrict__ W,const float* __restrict__ scale,
                         unsigned N,unsigned K){
    extern __shared__ __nv_bfloat16 xs[];
    for(unsigned i=threadIdx.x;i<K;i+=blockDim.x) xs[i]=x[i];
    __syncthreads();
    const unsigned lane=threadIdx.x&NV_LANE_MASK, warp=threadIdx.x>>NV_WARP_SHIFT;
    const unsigned nchunk=(K+GV_STEP-1)/GV_STEP;
    const unsigned nblk=gridDim.x, per=(N+nblk-1)/nblk, n0=blockIdx.x*per;
    const unsigned n1=(n0+per<N)?(n0+per):N;
    for(unsigned n=n0+warp;n<n1;n+=NV_WARPS){
        const uint8_t* wrow=W+(size_t)n*K;
        float acc=0.0f;
        for(unsigned c=0;c<nchunk;c+=GV_UNROLL_FP8){
            uint2 wv[GV_UNROLL_FP8]; unsigned kk[GV_UNROLL_FP8];
#pragma unroll
            for(int u=0;u<GV_UNROLL_FP8;u++){ unsigned k=(c+u)*GV_STEP+lane*8; kk[u]=k;
                wv[u]=(k<K)?*(const uint2*)(wrow+k):make_uint2(0u,0u); }
#pragma unroll
            for(int u=0;u<GV_UNROLL_FP8;u++){ if(kk[u]>=K) continue;
                acc = VER? dot8_fp8_vecB(wv[u],xs+kk[u],acc) : dot8_fp8(wv[u],ld_smem8(xs+kk[u]),acc); }
        }
        float t=warp_sum32(acc);
        if(lane==0) C[n]=__float2bfloat16(t*scale[n]);
    }
}

// device helpers to build the fp8 weight (bytes + exact dequant) so the oracle matches ---------
__global__ void quant_e4m3(uint8_t* W8,float* Wdq,const float* Wf,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    __nv_fp8_e4m3 q(Wf[i]); W8[i]=*(uint8_t*)&q; Wdq[i]=(float)q;
}

// ---- host ----------------------------------------------------------------------------------
static float frand(){ return (float)rand()/RAND_MAX*2.0f-1.0f; }

struct Shape{ const char* name; unsigned N,K; };

static double relL2(const std::vector<float>& a,const std::vector<float>& r){
    double num=0,den=0; for(size_t i=0;i<r.size();i++){ double d=a[i]-r[i]; num+=d*d; den+=(double)r[i]*r[i]; }
    return sqrt(num/(den+1e-30));
}

int main(){
    srand(1234);
    CK(cudaSetDevice(0));
    int l2=0; cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,0);
    const size_t POOL_BYTES = (size_t)768<<20;  // 768 MB > 2*L2 (60 MB) -> weights cycle cold
    printf("H100 NVL, L2=%.0f MB, replica pool=%zu MB\n\n",l2/1048576.0,POOL_BYTES>>20);

    Shape shapes[] = {
        {"qkv/attn  N4096", 4096, 3840},
        {"o_proj    N2048", 2048, 3840},
        {"lm_head   N15360",15360,3840},
    };
    const int NSH=3, ITERS=200, WARM=10;

    const unsigned K=3840;
    std::vector<__nv_bfloat16> hx(K);
    std::vector<float> hxf(K);
    for(unsigned k=0;k<K;k++){ float v=frand()*0.5f; hx[k]=__float2bfloat16(v); hxf[k]=__bfloat162float(hx[k]); }
    __nv_bfloat16* dx; CK(cudaMalloc(&dx,K*sizeof(__nv_bfloat16)));
    CK(cudaMemcpy(dx,hx.data(),K*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));

    size_t smem = K*sizeof(__nv_bfloat16);
    CK(cudaFuncSetAttribute(gemv_bf16<0>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(gemv_bf16<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(gemv_fp8<0>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(gemv_fp8<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    printf("=== CORRECTNESS (relL2 vs f32 CPU oracle; PASS bf16<2e-3, fp8<3e-3) ===\n");
    printf("%-18s %-6s %-12s %-12s %s\n","shape","dtype","A relL2","B relL2","RESULT");

    struct Res{ double nsA,nsB,gbA,gbB; };
    std::vector<Res> bf, fp;

    for(int si=0; si<NSH; si++){
        const Shape&s=shapes[si];
        const unsigned N=s.N; const size_t WN=(size_t)N*K;
        std::vector<__nv_bfloat16> hW(WN);
        for(size_t i=0;i<WN;i++) hW[i]=__float2bfloat16(frand()*0.5f);
        std::vector<float> oracle_bf(N);
        for(unsigned n=0;n<N;n++){ float acc=0; for(unsigned k=0;k<K;k++) acc+=__bfloat162float(hW[(size_t)n*K+k])*hxf[k]; oracle_bf[n]=acc; }

        size_t wbytes=WN*sizeof(__nv_bfloat16);
        int NREP=(int)(POOL_BYTES/wbytes); if(NREP<1) NREP=1; if(NREP>64) NREP=64;
        __nv_bfloat16* dW; CK(cudaMalloc(&dW,(size_t)NREP*wbytes));
        for(int r=0;r<NREP;r++) CK(cudaMemcpy(dW+(size_t)r*WN,hW.data(),wbytes,cudaMemcpyHostToDevice));
        __nv_bfloat16* dC; CK(cudaMalloc(&dC,N*sizeof(__nv_bfloat16)));

        unsigned nblk=132*4; if(nblk>(N+7)/8) nblk=(N+7)/8;
        auto run=[&](int ver,int rep){ if(ver==0) gemv_bf16<0><<<nblk,NV_THREADS,smem>>>(dC,dx,dW+(size_t)rep*WN,N,K);
                                       else       gemv_bf16<1><<<nblk,NV_THREADS,smem>>>(dC,dx,dW+(size_t)rep*WN,N,K); };
        std::vector<__nv_bfloat16> hC(N); std::vector<float> cA(N),cB(N);
        run(0,0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hC.data(),dC,N*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
        for(unsigned n=0;n<N;n++) cA[n]=__bfloat162float(hC[n]);
        run(1,0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hC.data(),dC,N*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
        for(unsigned n=0;n<N;n++) cB[n]=__bfloat162float(hC[n]);
        double rA=relL2(cA,oracle_bf), rB=relL2(cB,oracle_bf);
        bool pass=(rA<2e-3)&&(rB<2e-3);
        printf("%-18s %-6s %-12.3e %-12.3e %s\n",s.name,"bf16",rA,rB,pass?"PASS":"FAIL");

        Res r{};
        for(int v=0;v<2;v++){
            for(int i=0;i<WARM;i++) run(v,i%NREP);
            CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
            for(int i=0;i<ITERS;i++) run(v,i%NREP);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
            double ns=ms*1e6/ITERS, gb=(double)wbytes/ns; // bytes/ns == GB/s
            if(v==0){ r.nsA=ns; r.gbA=gb; } else { r.nsB=ns; r.gbB=gb; }
        }
        bf.push_back(r);
        CK(cudaFree(dW)); CK(cudaFree(dC));
    }

    printf("\n");
    for(int si=0; si<NSH; si++){
        const Shape&s=shapes[si];
        const unsigned N=s.N; const size_t WN=(size_t)N*K;
        std::vector<float> hWf(WN);
        for(size_t i=0;i<WN;i++) hWf[i]=frand()*0.5f;
        float* dWf; CK(cudaMalloc(&dWf,WN*sizeof(float)));
        CK(cudaMemcpy(dWf,hWf.data(),WN*sizeof(float),cudaMemcpyHostToDevice));
        uint8_t* dW8_1; float* dWdq; CK(cudaMalloc(&dW8_1,WN)); CK(cudaMalloc(&dWdq,WN*sizeof(float)));
        quant_e4m3<<<(unsigned)((WN+255)/256),256>>>(dW8_1,dWdq,dWf,WN); CK(cudaDeviceSynchronize());
        std::vector<float> hWdq(WN); CK(cudaMemcpy(hWdq.data(),dWdq,WN*sizeof(float),cudaMemcpyDeviceToHost));
        std::vector<float> hsc(N); for(unsigned n=0;n<N;n++) hsc[n]=0.5f+0.01f*(float)(n%7);
        float* dsc; CK(cudaMalloc(&dsc,N*sizeof(float))); CK(cudaMemcpy(dsc,hsc.data(),N*sizeof(float),cudaMemcpyHostToDevice));
        std::vector<float> oracle_fp(N);
        for(unsigned n=0;n<N;n++){ float acc=0; for(unsigned k=0;k<K;k++) acc+=hWdq[(size_t)n*K+k]*hxf[k]; oracle_fp[n]=acc*hsc[n]; }

        size_t wbytes=WN; // 1 byte/elem
        int NREP=(int)(POOL_BYTES/wbytes); if(NREP<1) NREP=1; if(NREP>128) NREP=128;
        uint8_t* dW8; CK(cudaMalloc(&dW8,(size_t)NREP*wbytes));
        for(int r=0;r<NREP;r++) CK(cudaMemcpy(dW8+(size_t)r*WN,dW8_1,wbytes,cudaMemcpyDeviceToDevice));
        __nv_bfloat16* dC; CK(cudaMalloc(&dC,N*sizeof(__nv_bfloat16)));

        unsigned nblk=132*4; if(nblk>(N+7)/8) nblk=(N+7)/8;
        auto run=[&](int ver,int rep){ if(ver==0) gemv_fp8<0><<<nblk,NV_THREADS,smem>>>(dC,dx,dW8+(size_t)rep*WN,dsc,N,K);
                                       else       gemv_fp8<1><<<nblk,NV_THREADS,smem>>>(dC,dx,dW8+(size_t)rep*WN,dsc,N,K); };
        std::vector<__nv_bfloat16> hC(N); std::vector<float> cA(N),cB(N);
        run(0,0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hC.data(),dC,N*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
        for(unsigned n=0;n<N;n++) cA[n]=__bfloat162float(hC[n]);
        run(1,0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hC.data(),dC,N*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
        for(unsigned n=0;n<N;n++) cB[n]=__bfloat162float(hC[n]);
        double rA=relL2(cA,oracle_fp), rB=relL2(cB,oracle_fp);
        bool pass=(rA<3e-3)&&(rB<3e-3);
        printf("%-18s %-6s %-12.3e %-12.3e %s\n",s.name,"fp8",rA,rB,pass?"PASS":"FAIL");

        Res r{};
        for(int v=0;v<2;v++){
            for(int i=0;i<WARM;i++) run(v,i%NREP);
            CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
            for(int i=0;i<ITERS;i++) run(v,i%NREP);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
            double ns=ms*1e6/ITERS, gb=(double)wbytes/ns;
            if(v==0){ r.nsA=ns; r.gbA=gb; } else { r.nsB=ns; r.gbB=gb; }
        }
        fp.push_back(r);
        CK(cudaFree(dWf)); CK(cudaFree(dW8_1)); CK(cudaFree(dWdq)); CK(cudaFree(dsc)); CK(cudaFree(dW8)); CK(cudaFree(dC));
    }

    printf("\n=== BENCHMARK (200 iters/10 warm, HBM-resident; GB/s = weight bytes / time) ===\n");
    printf("ceiling: 4023 GB/s spec, ~3300-3400 GB/s achievable\n");
    printf("%-18s %-6s %10s %10s %10s %10s %8s\n","shape","dtype","A ns","B ns","A GB/s","B GB/s","A/B");
    for(int i=0;i<NSH;i++){ Res&r=bf[i];
        printf("%-18s %-6s %10.0f %10.0f %10.0f %10.0f %8.3f\n",shapes[i].name,"bf16",r.nsA,r.nsB,r.gbA,r.gbB,r.nsA/r.nsB); }
    for(int i=0;i<NSH;i++){ Res&r=fp[i];
        printf("%-18s %-6s %10.0f %10.0f %10.0f %10.0f %8.3f\n",shapes[i].name,"fp8",r.nsA,r.nsB,r.gbA,r.gbB,r.nsA/r.nsB); }
    return 0;
}
