/* flashpre_fp8_bw_sm120.cu — KV-read BANDWIDTH + correctness microbench for the FAST (PIPE=1
 * cp.async) hd512 FULL-attention flash-PREFILL px4 arm on sm_120 (beat-fp8-prefill Exp1).
 *
 * GOAL (GO/NO-GO gate, plan step 1): the shipped fp8-KV prefill runs only on the slow PIPE=0
 * synchronous-staging arm (~20-30% slower than bf16 cp.async). This harness isolates the new
 * fp8 arm of d_flash_prefill_px4<512,32,16,true> — raw e4m3 staged via the cp.async ring, dequant
 * after cp_wait, K-scale post-mma, V-scale into P — vs the bf16 px4 baseline, at seq_kv
 * {8k..128k}. It answers ONE question before any full-model work:
 *
 *   fp8 FASTER than bf16 at >= 32k?  GB/s not worse?  relL2 (fp8 vs bf16) in the ~3-6e-3 band?
 *   If NOT faster at 32k -> STOP.
 *
 * Config: ONE query tile (BQ=32) at absolute position seq_kv-32 so it attends the FULL causal KV
 * (the O(ctx^2) long-ctx tail this arm dominates). nsplit fills the machine (~1 item/SM). Both
 * arms launched with the SAME grid/smem so the ms delta is purely the KV-read + dequant cost.
 *
 * bytes_issued = n_head * seq_kv * HD * elem * 2 (K+V; each head reads its kv_head's KV once)
 * bytes_phys   = n_kv_head * seq_kv * HD * elem * 2 (distinct HBM bytes)
 *
 * Build (plain env, sm_120a — MUST define PLOW_FP8_KV so the px4 arena reserves the e4m3 staging):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -DPLOW_FP8_KV=1 -I runtime/common -I runtime/nvidia -include cstdint \
 *     runtime/tests/flashpre_fp8_bw_sm120.cu -o flashpre_bw
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sm120_common.cuh" /* pulls op_attention.cuh (d_flash_prefill_px4 / d_flash_merge) */

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

static uint32_t rng_s = 0x13572468u;
static float rnd() { rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
    return (float)((int32_t)rng_s) / 2147483648.0f; }

static const unsigned NH = 16, NKV = 2, HD = 512; /* Gemma-4 full-attention geometry */
static const int BQ = 32, BKV = 16;

/* Standalone launchers pinned to 1 block/SM, matching the megakernel's __launch_bounds__(256,1). */
__global__ void __launch_bounds__(256,1)
k_px4_bf16(float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, bf16* O,
           unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0,
           unsigned nsplit, unsigned kv_stride, float scale) {
    extern __shared__ float arena[];
    d_flash_prefill_px4<512,32,16,false>(Opart,mlpart,Q,K,V,O,seq_q,seq_kv,n_head,n_kv_head,q_pos0,
                                         0u/*window=full*/,nsplit,kv_stride,0xFFFFFFFFu,scale,
                                         blockIdx.x,gridDim.x,arena);
}
__global__ void __launch_bounds__(256,1)
k_px4_fp8(float* Opart, float* mlpart, const bf16* Q, const bf16* K, const bf16* V, bf16* O,
          unsigned seq_q, unsigned seq_kv, unsigned n_head, unsigned n_kv_head, unsigned q_pos0,
          unsigned nsplit, unsigned kv_stride, float scale, const float* ks, const float* vs) {
    extern __shared__ float arena[];
    d_flash_prefill_px4<512,32,16,true>(Opart,mlpart,Q,K,V,O,seq_q,seq_kv,n_head,n_kv_head,q_pos0,
                                        0u/*window=full*/,nsplit,kv_stride,0xFFFFFFFFu,scale,
                                        blockIdx.x,gridDim.x,arena,ks,vs);
}
__global__ void __launch_bounds__(256,1)
k_merge(bf16* O, const float* Opart, const float* mlpart, unsigned n_batch, unsigned n_head,
        unsigned nsplit) {
    d_flash_merge<512>(O,Opart,mlpart,n_batch,n_head,nsplit,blockIdx.x,gridDim.x);
}

struct Res { double ms; double gbps_issued; double gbps_phys; };

/* One measured config. Fills O for the correctness check; returns min ms over iters (px4 only).
 * seq_q = BQ (legacy single-q-tile latency regime) or 8192 (CHUNK regime: 256 q-tiles x 16 heads
 * = 4096 work items on a 188-block persistent grid, KV re-read across q-tiles L2-served — the
 * regime the real chunked-prefill TTFT ladder exercises; nsplit=1, fused epilogue). */
static Res run(bool fp8, unsigned ctx, unsigned seq_q, unsigned nsplit, int iters,
               std::vector<float>* Oout) {
    rng_s = 0x0BADF00Du + ctx; /* SAME inputs for the bf16 and fp8 runs of a given ctx (relL2 valid) */
    const unsigned q_pos0 = ctx - seq_q;          /* placed at the end -> attends full causal KV */
    const unsigned kvs = ctx;
    const float scale = 1.f/sqrtf((float)HD);
    const size_t nkv_elems = (size_t)NKV*kvs*HD;

    void *dK=nullptr,*dV=nullptr; float *dKs=nullptr,*dVs=nullptr;
    /* Reference bf16 KV values (both arms see the same rounded numbers: fp8 quantizes them). */
    std::vector<float> kref(nkv_elems), vref(nkv_elems);
    for (size_t i=0;i<nkv_elems;i++){ kref[i]=rnd()*0.5f; vref[i]=rnd()*0.5f; }
    if (fp8){
        std::vector<uint8_t> k(nkv_elems), v(nkv_elems);
        for (size_t i=0;i<nkv_elems;i++){ __nv_fp8_e4m3 a(kref[i]), b(vref[i]);
            k[i]=*(uint8_t*)&a; v[i]=*(uint8_t*)&b; }
        CK(cudaMalloc(&dK,nkv_elems)); CK(cudaMalloc(&dV,nkv_elems));
        CK(cudaMemcpy(dK,k.data(),nkv_elems,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV,v.data(),nkv_elems,cudaMemcpyHostToDevice));
        const size_t nrows=(size_t)NKV*kvs;
        std::vector<float> ks(nrows,1.0f), vs(nrows,1.0f); /* unit scales: numerics == bf16 cache */
        CK(cudaMalloc(&dKs,nrows*4)); CK(cudaMalloc(&dVs,nrows*4));
        CK(cudaMemcpy(dKs,ks.data(),nrows*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dVs,vs.data(),nrows*4,cudaMemcpyHostToDevice));
    } else {
        std::vector<bf16> k(nkv_elems), v(nkv_elems);
        for (size_t i=0;i<nkv_elems;i++){ k[i]=__float2bfloat16(kref[i]); v[i]=__float2bfloat16(vref[i]); }
        CK(cudaMalloc(&dK,nkv_elems*2)); CK(cudaMalloc(&dV,nkv_elems*2));
        CK(cudaMemcpy(dK,k.data(),nkv_elems*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV,v.data(),nkv_elems*2,cudaMemcpyHostToDevice));
    }
    std::vector<bf16> q((size_t)seq_q*NH*HD); for (auto& x:q) x=__float2bfloat16(rnd()*0.5f);
    bf16* dQ=nullptr; CK(cudaMalloc(&dQ,q.size()*2));
    CK(cudaMemcpy(dQ,q.data(),q.size()*2,cudaMemcpyHostToDevice));

    float *dOp=nullptr,*dMl=nullptr; bf16* dO=nullptr;
    CK(cudaMalloc(&dOp,(size_t)seq_q*NH*nsplit*HD*4));
    CK(cudaMalloc(&dMl,(size_t)seq_q*NH*nsplit*2*4));
    CK(cudaMalloc(&dO,(size_t)seq_q*NH*HD*2));

    /* Both arms launched with the fp8 arena size (the larger one) so the grid/smem is identical.
     * Legacy regime (seq_q==BQ): grid == n_work (192, ~1/SM). Chunk regime: persistent 188. */
    const size_t smem=(size_t)FA_PX4_SMEM_FLOATS(512,32,16)*sizeof(float);
    const unsigned n_work = (seq_q/BQ) * NH * nsplit;
    const unsigned grid = (seq_q == (unsigned)BQ) ? n_work : ((n_work < 188u) ? n_work : 188u);
    auto launch=[&](){
        if (fp8) k_px4_fp8<<<grid,256,smem>>>(dOp,dMl,dQ,(bf16*)dK,(bf16*)dV,dO,seq_q,kvs,NH,NKV,q_pos0,nsplit,kvs,scale,dKs,dVs);
        else     k_px4_bf16<<<grid,256,smem>>>(dOp,dMl,dQ,(bf16*)dK,(bf16*)dV,dO,seq_q,kvs,NH,NKV,q_pos0,nsplit,kvs,scale);
    };
    if (fp8) CK(cudaFuncSetAttribute(k_px4_fp8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    else     CK(cudaFuncSetAttribute(k_px4_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    for (int w=0;w<5;w++) launch();
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double best=1e30;
    for (int it=0; it<iters; it++){
        CK(cudaEventRecord(a));
        launch();
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
        if (ms<best) best=ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);

    /* Merge the split partials -> final O, then copy back for the relL2 check.
     * nsplit==1 (chunk regime): the kernel's fused epilogue already wrote the final bf16 O. */
    if (Oout){
        if (nsplit > 1) k_merge<<<seq_q*NH,256>>>(dO,dOp,dMl,seq_q,NH,nsplit);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        std::vector<bf16> oh((size_t)seq_q*NH*HD);
        CK(cudaMemcpy(oh.data(),dO,oh.size()*2,cudaMemcpyDeviceToHost));
        Oout->resize(oh.size());
        for (size_t i=0;i<oh.size();i++) (*Oout)[i]=__bfloat162float(oh[i]);
    }

    cudaFree(dK);cudaFree(dV);cudaFree(dQ);cudaFree(dOp);cudaFree(dMl);cudaFree(dO);
    if(dKs)cudaFree(dKs); if(dVs)cudaFree(dVs);

    const double elem = fp8?1.0:2.0;
    const double bytes_issued=(double)NH *ctx*HD*elem*2.0;
    const double bytes_phys  =(double)NKV*ctx*HD*elem*2.0;
    Res r; r.ms=best; r.gbps_issued=bytes_issued/(best*1e-3)/1e9; r.gbps_phys=bytes_phys/(best*1e-3)/1e9;
    return r;
}

static double relL2(const std::vector<float>& a, const std::vector<float>& b){
    double num=0,den=0; for (size_t i=0;i<a.size();i++){ double d=a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; }
    return den>0? sqrt(num/den):0.0;
}

int main(int argc, char** argv){
    int iters = argc>1?atoi(argv[1]):50;
    unsigned only_ctx = argc>2?(unsigned)atoi(argv[2]):0; /* 0 = all */
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("# device %s, SMs %d, iters %d\n", p.name, p.multiProcessorCount, iters);
    printf("# geom: n_head=%u n_kv_head=%u HD=%u BQ=%d BKV=%d (px4 hd512 FULL-attn prefill)\n", NH,NKV,HD,BQ,BKV);
    printf("# HBM ceiling assumed 1535 GB/s; px4 smem/arm = %zu B\n",
           (size_t)FA_PX4_SMEM_FLOATS(512,32,16)*sizeof(float));

    const unsigned ctxs[]={8192u,16384u,32768u,65536u,131072u};
    const int NC=5;
    /* nsplit fills ~192 blocks (1/SM): NH=16 -> nsplit=12 -> n_work=192. */
    const unsigned nsplit=12u;

    printf("\n## px4 hd512 flash-PREFILL KV-read microbench (best of %d, nsplit=%u)\n", iters, nsplit);
    printf("%-6s %-7s | %10s | %10s %10s | %9s | %10s\n",
           "cfg","ctx","ms","GBps_iss","GBps_phys","%ceil","relL2(vsbf16)");

    double ms_bf[5], ms_fp[5];
    for (int ci=0; ci<NC; ci++){
        unsigned ctx=ctxs[ci];
        if (only_ctx && ctx != only_ctx) { ms_bf[ci]=0; ms_fp[ci]=1; continue; }
        std::vector<float> Obf, Ofp;
        Res rb=run(false,ctx,BQ,nsplit,iters,&Obf); ms_bf[ci]=rb.ms;
        Res rf=run(true, ctx,BQ,nsplit,iters,&Ofp); ms_fp[ci]=rf.ms;
        double rl=relL2(Ofp,Obf);
        printf("%-6s %-7u | %10.4f | %10.1f %10.1f | %9.1f | %10s\n",
               "bf16",ctx,rb.ms,rb.gbps_issued,rb.gbps_phys,100.0*rb.gbps_issued/1535.0,"-");
        printf("%-6s %-7u | %10.4f | %10.1f %10.1f | %9.1f | %10.2e\n",
               "fp8",ctx,rf.ms,rf.gbps_issued,rf.gbps_phys,100.0*rf.gbps_issued/1535.0,rl);
        printf("       speedup(bf16/fp8) = %.3fx %s\n\n", rb.ms/rf.ms,
               (rf.ms<rb.ms)?"(fp8 FASTER)":"(fp8 slower)");
    }

    printf("## GO/NO-GO, legacy single-q-tile latency regime (fp8 FASTER at >= 32k)\n");
    for (int ci=0; ci<NC; ci++){
        if (ctxs[ci] < 32768u) continue;
        printf("  ctx %-7u : fp8 %s bf16 (%.3fx)\n", ctxs[ci],
               (ms_fp[ci]<ms_bf[ci])?"FASTER  ":"SLOWER  ", ms_bf[ci]/ms_fp[ci]);
    }

    /* CHUNK regime — the ladder-faithful gate: a trailing 8192-token chunk (256 q-tiles x 16
     * heads on a persistent 188-block grid, nsplit=1, fused epilogue). KV is re-read across
     * q-tiles through L2, exactly as the e2e chunked prefill runs the px4 arm. */
    const int citers = (iters > 10) ? 10 : iters;
    printf("\n## CHUNK regime (seq_q=8192, nsplit=1, grid 188, best of %d)\n", citers);
    printf("%-6s %-7s | %10s | %10s\n", "cfg","ctx","ms","relL2(vsbf16)");
    double cms_bf[5], cms_fp[5];
    for (int ci=0; ci<NC; ci++){
        unsigned ctx=ctxs[ci];
        if (only_ctx && ctx != only_ctx) { cms_bf[ci]=0; cms_fp[ci]=1; continue; }
        std::vector<float> Obf, Ofp;
        Res rb=run(false,ctx,8192u,1u,citers,&Obf); cms_bf[ci]=rb.ms;
        Res rf=run(true, ctx,8192u,1u,citers,&Ofp); cms_fp[ci]=rf.ms;
        double rl=relL2(Ofp,Obf);
        printf("%-6s %-7u | %10.4f | %10s\n","bf16",ctx,rb.ms,"-");
        printf("%-6s %-7u | %10.4f | %10.2e\n","fp8",ctx,rf.ms,rl);
        printf("       speedup(bf16/fp8) = %.3fx %s\n\n", rb.ms/rf.ms,
               (rf.ms<rb.ms)?"(fp8 FASTER)":"(fp8 slower)");
    }
    printf("## GO/NO-GO, CHUNK regime (fp8 FASTER at >= 32k)\n");
    for (int ci=0; ci<NC; ci++){
        if (ctxs[ci] < 32768u) continue;
        printf("  ctx %-7u : fp8 %s bf16 (%.3fx)\n", ctxs[ci],
               (cms_fp[ci]<cms_bf[ci])?"FASTER  ":"SLOWER  ", cms_bf[ci]/cms_fp[ci]);
    }
    return 0;
}
