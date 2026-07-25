// fa_gf_full_ab.cu — GQA-fusion factor A/B for the FULL-attention decode layer on sm_120a.
//
// QUESTION: Gemma-4 full layers are kvh=2, hd=512, 16 q heads => GQA=8. Production runs the
// flash-decode kernel at GF=2 (PLOW_NV_FA_GF_FULL default), so each KV row is read by GQA/GF=4
// work-groups (4x re-read). Does raising GF_FULL to 4 or 8 (fewer re-reads, but 4-8 hd512
// accumulators per thread => register-spill risk) actually cut decode time at long ctx? And does a
// pure L2-coschedule of the GF=2 re-reads recover the same win without the register cost?
//
// This harness #includes the REAL production kernel (runtime/nvidia/op_attention.cuh) and drives
// d_flash_decode<512,GF> + d_flash_merge<512> exactly as interp_sm120.cu does, so GF=2 here is
// byte-identical to what ships. Correctness is checked against an f32 CPU reference every run.
//
//   variants:  GF=2 (baseline), GF=4, GF=8, and GF=2 + L2-coschedule (same-KV items adjacent).
//   ctx:       32k / 64k / 128k     kvh=2  hd=512  n_head=16  bf16 KV (k_eq_v: K and V distinct bufs)
//
// build:  /usr/local/cuda/bin/nvcc -arch=sm_120a -O3 --ptxas-options=-v \
//             -o /tmp/fa_gf_full_ab runtime/nvidia/experiments/fa_gf_full_ab.cu
// run:    gpulease p9-flash /tmp/fa_gf_full_ab

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#define PLOW_NV_KVBOUNDS 0
#include "../op_attention.cuh"   // d_flash_decode<D,GF>, d_flash_merge<D>, FA_DEC_SMEM_FLOATS

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const int   D    = 512;   // full-layer head dim
static const int   NH   = 16;    // q heads
static const int   KVH  = 2;     // full-layer kv heads  -> GQA = 8
static const float SCALE = 0.0441941738f; // 1/sqrt(512)

// ---- launch wrapper: one block per work item, slice = perm[blockIdx] (identity or coschedule).
// nblk is passed huge so the kernel's grid-stride loop runs exactly once, on work item `w`.
template <int GF>
__global__ void decode_launch(float* Opart, float* mlpart,
                              const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                              const int* kv_len, unsigned n_head, unsigned n_kv_head,
                              unsigned kv_stride, unsigned window, float scale, unsigned nsplit,
                              unsigned kv_mask, const int* perm) {
    extern __shared__ float lds[];
    const unsigned w = (unsigned)perm[blockIdx.x];
    d_flash_decode<D, GF>(Opart, mlpart, Q, K, V, kv_len, /*n_batch*/1, n_head, n_kv_head,
                          kv_stride, window, scale, nsplit, kv_mask, /*slice*/w,
                          /*nblk*/0x40000000u, lds, 0);
}

__global__ void merge_launch(__nv_bfloat16* O, const float* Opart, const float* mlpart,
                             unsigned n_head, unsigned nsplit) {
    d_flash_merge<D>(O, Opart, mlpart, 1, n_head, nsplit, blockIdx.x, gridDim.x);
}

static __nv_bfloat16 rbf(){ return __float2bfloat16((float)rand()/RAND_MAX*2.f-1.f); }

int main(){
    srand(1234);
    const int ctxs[] = {32768, 65536, 131072};
    struct Cfg { int gf; int nsplit; int cosched; const char* name; };
    // GF=2 baseline uses production nsplit=24 (n_grp=8 -> 192 blocks). GF=4/8 keep the SAME 192-ish
    // fill (n_grp*nsplit) so the comparison isolates re-read, not occupancy. Also underfill rows.
    Cfg cfgs[] = {
        {2, 24, 0, "GF2  ns24 (baseline)"},
        {2, 24, 1, "GF2  ns24 L2-cosched"},
        {4, 48, 0, "GF4  ns48 (filled)"},
        {8, 96, 0, "GF8  ns96 (filled)"},
        {4, 24, 0, "GF4  ns24 (underfill)"},
        {8, 24, 0, "GF8  ns24 (underfill)"},
    };

    for (int ci = 0; ci < 3; ci++) {
        const int ctx = ctxs[ci];
        const unsigned kv_stride = ctx;         // linear full-attention cache
        const unsigned kv_mask   = 0xFFFFFFFFu; // no ring
        const unsigned window    = 0;           // full attention

        std::vector<__nv_bfloat16> hQ((size_t)NH*D), hK((size_t)KVH*ctx*D), hV((size_t)KVH*ctx*D);
        for (auto& x: hQ) x = rbf();
        for (auto& x: hK) x = rbf();
        for (auto& x: hV) x = rbf();
        int hlen = ctx;

        __nv_bfloat16 *dQ,*dK,*dV,*dO; int *dLen;
        CK(cudaMalloc(&dQ, hQ.size()*2)); CK(cudaMalloc(&dK, hK.size()*2)); CK(cudaMalloc(&dV, hV.size()*2));
        CK(cudaMemcpy(dQ,hQ.data(),hQ.size()*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dK,hK.data(),hK.size()*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV,hV.data(),hV.size()*2,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dLen,4)); CK(cudaMemcpy(dLen,&hlen,4,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dO, (size_t)NH*D*2));

        // CPU reference (newest query token, full window).
        std::vector<float> ref((size_t)NH*D);
        {
            const int qpos = ctx-1;
            for (int h=0; h<NH; h++){
                int hkv = h/(NH/KVH);
                std::vector<float> sc(ctx);
                float m=-1e30f;
                for (int r=0;r<=qpos;r++){
                    float d=0; const __nv_bfloat16* q=&hQ[(size_t)h*D];
                    const __nv_bfloat16* k=&hK[((size_t)hkv*ctx+r)*D];
                    for(int i=0;i<D;i++) d+=__bfloat162float(q[i])*__bfloat162float(k[i]);
                    sc[r]=d*SCALE; if(sc[r]>m)m=sc[r];
                }
                float l=0; for(int r=0;r<=qpos;r++){ sc[r]=expf(sc[r]-m); l+=sc[r]; }
                for(int i=0;i<D;i++){ float a=0;
                    for(int r=0;r<=qpos;r++) a+=sc[r]*__bfloat162float(hV[((size_t)hkv*ctx+r)*D+i]);
                    ref[(size_t)h*D+i]=a/l; }
            }
        }

        for (auto& cf : cfgs) {
            const int GF = cf.gf, nsplit = cf.nsplit;
            const int n_grp = NH/GF;
            const int n_work = n_grp*nsplit;

            std::vector<int> perm(n_work);
            if (!cf.cosched){ for(int i=0;i<n_work;i++) perm[i]=i; }
            else {
                const int gpk = (NH/KVH)/GF;   // groups sharing one kv-head = GQA/GF
                int idx=0;
                for (int hkv=0; hkv<KVH; hkv++)
                  for (int sp=0; sp<nsplit; sp++)
                    for (int gw=0; gw<gpk; gw++){
                        int hg = hkv*gpk + gw;
                        int w  = hg*nsplit + sp;   // kernel's w = (b*n_grp+hg)*nsplit+sp, b=0
                        perm[idx++]=w;
                    }
            }
            int* dPerm; CK(cudaMalloc(&dPerm,n_work*4));
            CK(cudaMemcpy(dPerm,perm.data(),n_work*4,cudaMemcpyHostToDevice));
            float *dOp,*dMl;
            CK(cudaMalloc(&dOp,(size_t)NH*nsplit*D*4));
            CK(cudaMalloc(&dMl,(size_t)NH*nsplit*2*4));

            size_t smem;
            void(*kern)(float*,float*,const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*,
                        const int*,unsigned,unsigned,unsigned,unsigned,float,unsigned,unsigned,const int*);
            if(GF==2){ kern=decode_launch<2>; smem=FA_DEC_SMEM_FLOATS(512,2)*4; }
            else if(GF==4){ kern=decode_launch<4>; smem=FA_DEC_SMEM_FLOATS(512,4)*4; }
            else { kern=decode_launch<8>; smem=FA_DEC_SMEM_FLOATS(512,8)*4; }
            if(smem>48*1024){
                CK(cudaFuncSetAttribute((void*)kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
            }

            auto run = [&](){
                kern<<<n_work,256,smem>>>(dOp,dMl,dQ,dK,dV,dLen,NH,KVH,kv_stride,window,SCALE,nsplit,kv_mask,dPerm);
                merge_launch<<<NH,256>>>(dO,dOp,dMl,NH,nsplit);
            };
            run(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

            std::vector<__nv_bfloat16> hO((size_t)NH*D);
            CK(cudaMemcpy(hO.data(),dO,hO.size()*2,cudaMemcpyDeviceToHost));
            double maxrel=0; for(size_t i=0;i<hO.size();i++){
                float o=__bfloat162float(hO[i]), r=ref[i];
                double rel=fabs(o-r)/(fabs(r)+1e-3); if(rel>maxrel)maxrel=rel; }

            const int iters=30;
            cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
            for(int w=0;w<3;w++) run();
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(a));
            for(int it=0;it<iters;it++) run();
            CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            float ms; CK(cudaEventElapsedTime(&ms,a,b)); ms/=iters;

            double logical = (double)n_grp*ctx*D*2.0*2.0;   // K+V, each group reads its kv-head once
            double uniqueB = (double)KVH*ctx*D*2.0*2.0;
            double gbps = logical/1e9/(ms/1e3);
            double reread = logical/uniqueB;
            printf("ctx %6d | %-22s | %7.3f ms | %7.0f GB/s(logical) | reread %.0fx | maxrel %.1e\n",
                   ctx, cf.name, ms, gbps, reread, maxrel);
            CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
            CK(cudaFree(dPerm)); CK(cudaFree(dOp)); CK(cudaFree(dMl));
        }
        printf("\n");
        CK(cudaFree(dQ));CK(cudaFree(dK));CK(cudaFree(dV));CK(cudaFree(dO));CK(cudaFree(dLen));
    }
    return 0;
}
