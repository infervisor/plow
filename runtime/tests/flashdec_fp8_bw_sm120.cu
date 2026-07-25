/* flashdec_fp8_bw_sm120.cu — KV-read BANDWIDTH microbench for the hd512 FULL-attention
 * flash-DECODE arm on sm_120 (Gemma-4-26B-A4B full layers: n_head=16, n_kv_head=2, gqa=8, D=512).
 *
 * GOAL (beat26b-flashdec): isolate FlashDecodeFp8<512> at KV lengths {32k,64k,96k,128k} and
 * measure achieved KV-read GB/s vs the 1535 GB/s HBM ceiling and vs the bf16 FlashDecode path.
 * The full-model campaign measured plow's marginal KV cost as ~21.3 ns/tok vs vLLM's ~12.0
 * (~1.8x). This harness A/Bs the named levers WITHOUT touching the model:
 *
 *   GF=4  (SHIPPED): gqa/GF = 2 head-groups per kv_head  -> each KV byte read TWICE.
 *   GF=8  (CANDIDATE): GF == gqa -> one group per kv_head -> each KV byte read ONCE (half HBM).
 *
 * KV-2 (plans/p10-kv-zip.md): + an SZ arm — the KVZIP-SZ12 v1.2 lossless row blob (800 B/row vs
 * 1024, 1.28x fewer KV bytes) decoded INLINE in d_flash_decode. KV content comes from a REAL
 * plow dump (PLOW_KV_DUMP, default /dev/shm/kv0/prose, tensor PLOW_KV_TENSOR, default kv.47 —
 * a full-attention hd512 layer) so escape rows and exponent spread are real; rows are tiled to
 * the target ctx. The sz arm is gated on output being BIT-EXACT vs the bf16 arm on the same
 * bytes (lossless => identical dot8 inputs => identical Opart/mlpart). Without a dump the sz
 * configs are skipped and the bf16/fp8 sweep behaves as before (random fill).
 *
 * The megakernel pins 1 block/SM (__launch_bounds__(256,1)); this harness pins the same so the
 * per-item register/occupancy footprint matches the model. Each work item = one block; the grid
 * is n_work = n_batch * (n_head/GF) * nsplit, exactly the megakernel's flash-decode item count.
 *
 * bytes_issued  = n_grp * ctx * D * elem * 2(K,V)   (what the kernel demands from the mem system)
 * bytes_phys    = n_kv_head * ctx * D * elem * 2     (distinct HBM bytes; GF8 issues == phys)
 * achieved GB/s = bytes_issued / time  (redundant-read-aware: the real HBM demand)
 * eff GB/s      = bytes_phys / time    (vLLM-comparable: distinct KV moved per unit time)
 *
 * Build (plain env, sm_120a):
 *   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin nvcc -std=c++17 -arch=sm_120a -O3 \
 *     -I runtime/common -I runtime/nvidia -include cstdint \
 *     runtime/tests/flashdec_fp8_bw_sm120.cu -o flashdec_bw
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sm120_common.cuh" /* pulls op_attention.cuh (d_flash_decode / d_flash_merge) */

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

static uint32_t rng_s = 0x2468ace0u;
static float rnd() { rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
    return (float)((int32_t)rng_s) / 2147483648.0f; }

/* Standalone launchers pinned to 1 block/SM, matching the megakernel's __launch_bounds__(256,1). */
template<int D,int GF>
__global__ void __launch_bounds__(256,1)
k_flash_bf16(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
             const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
             float scale, unsigned nsplit){
    extern __shared__ float arena[];
    d_flash_decode<D,GF,false>(Op,Ml,Q,K,V,kvlen,nb,nh,nkv,kvs,0u/*window=full*/,scale,nsplit,
                               0xFFFFFFFFu,blockIdx.x,gridDim.x,arena,0u);
}
template<int D,int GF>
__global__ void __launch_bounds__(256,1)
k_flash_fp8(float* Op, float* Ml, const bf16* Q, const bf16* K, const bf16* V,
            const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
            float scale, unsigned nsplit, const float* ks, const float* vs){
    extern __shared__ float arena[];
    d_flash_decode<D,GF,true>(Op,Ml,Q,K,V,kvlen,nb,nh,nkv,kvs,0u/*window=full*/,scale,nsplit,
                              0xFFFFFFFFu,blockIdx.x,gridDim.x,arena,0u,ks,vs);
}
template<int D,int GF>
__global__ void __launch_bounds__(256,1)
k_flash_sz(float* Op, float* Ml, const bf16* Q, const uint8_t* Kz, const uint8_t* Vz,
           const int* kvlen, unsigned nb, unsigned nh, unsigned nkv, unsigned kvs,
           float scale, unsigned nsplit){
    extern __shared__ float arena[];
    d_flash_decode<D,GF,false,true>(Op,Ml,Q,(const bf16*)Kz,(const bf16*)Vz,kvlen,nb,nh,nkv,kvs,
                                    0u/*window=full*/,scale,nsplit,
                                    0xFFFFFFFFu,blockIdx.x,gridDim.x,arena,0u);
}

static const unsigned NH = 16, NKV = 2, D = 512; /* Gemma-4-26B full-attention geometry */
static const unsigned ROWB = D/2 + D + 32;       /* KVZIP-SZ12 v1.2: 800 B/row at hd512 */

/* ---- real-dump KV rows (PLOW_KV_DUMP) -------------------------------------------------------- */
static std::vector<uint16_t> g_krows, g_vrows; /* [g_nrows][D] valid ring rows of a full layer */
static unsigned g_nrows = 0;

static bool load_dump(){
    const char* dir = getenv("PLOW_KV_DUMP");   if (!dir) dir = "/dev/shm/kv0/prose";
    const char* ten = getenv("PLOW_KV_TENSOR"); if (!ten) ten = "kv.47";
    char p[512];
    snprintf(p,sizeof p,"%s/manifest.txt",dir);
    FILE* mf = fopen(p,"r"); if (!mf) return false;
    unsigned ctx=0; char tok[128];
    while (fscanf(mf,"%127s",tok)==1){ if (!strcmp(tok,"ctx")){ if(fscanf(mf,"%u",&ctx)!=1) ctx=0; break; } }
    fclose(mf);
    if (!ctx) return false;
    const unsigned ring = 32768; /* full-attention hd512 ring of the 12B dump */
    g_nrows = ctx < ring ? ctx : ring;
    for (int kv=0; kv<2; kv++){
        std::vector<uint16_t>& dst = kv ? g_vrows : g_krows;
        snprintf(p,sizeof p,"%s/%s.%s.raw",dir,ten,kv?"v":"k");
        FILE* f = fopen(p,"rb"); if (!f){ g_nrows=0; return false; }
        dst.resize((size_t)g_nrows*D);
        if (fread(dst.data(),2,dst.size(),f)!=dst.size()){ fclose(f); g_nrows=0; return false; }
        fclose(f);
    }
    printf("# KV dump: %s %s, %u valid rows\n",dir,ten,g_nrows);
    return true;
}
/* cache row -> dump row, decorrelated per kv-head, identical across arms (bit-exact A/B). */
static inline unsigned src_row(unsigned head, unsigned i){ return (i + head*10007u) % g_nrows; }

/* ---- host KVZIP-SZ12 v1.2 encoder (normative spec: perf-data/harness/kvzip_oracle.py) ------- */
static void sz_encode_row(const uint16_t* x, uint8_t* blob){
    uint8_t e[D], lo[D]; int hist[256]={0};
    for (unsigned i=0;i<D;i++){ e[i]=(x[i]>>7)&0xFF; lo[i]=((x[i]>>8)&0x80)|(x[i]&0x7F); hist[e[i]]++; }
    int cur=0; for (int s=0;s<15;s++) cur+=hist[s];
    int best=cur, bs=0;
    for (int s=1;s<=241;s++){ cur+=hist[s+14]-hist[s-1]; if (cur>best){best=cur;bs=s;} }
    memset(blob+D/2+D,0xFF,32); blob[D/2+D+31]=0;
    unsigned nesc=0;
    for (unsigned i=0;i<D;i+=2){
        unsigned c0, c1;
        int d0=e[i]-bs, d1=e[i+1]-bs;
        if (d0>=0 && d0<15) c0=(unsigned)d0; else {
            if (nesc>=9){ fprintf(stderr,"sz_encode: >9 escapes/row\n"); exit(3); }
            uint8_t* sl=blob+D/2+D+4+3*nesc++; sl[0]=e[i]; sl[1]=i&0xFF; sl[2]=i>>8; c0=15;
        }
        if (d1>=0 && d1<15) c1=(unsigned)d1; else {
            if (nesc>=9){ fprintf(stderr,"sz_encode: >9 escapes/row\n"); exit(3); }
            uint8_t* sl=blob+D/2+D+4+3*nesc++; sl[0]=e[i+1]; sl[1]=(i+1)&0xFF; sl[2]=(i+1)>>8; c1=15;
        }
        blob[i/2]=(uint8_t)(c0|(c1<<4));
    }
    memcpy(blob+D/2,lo,D);
    uint32_t hdr=(uint32_t)bs|((uint32_t)nesc<<8);
    memcpy(blob+D/2+D,&hdr,4);
}

struct Res { double ms; double gbps_issued; double gbps_phys; };

/* KV-2 numerics gate: bf16 arm and sz arm on the SAME logical KV must produce BIT-IDENTICAL
 * Opart/mlpart (lossless decode feeds the same bf16v8 into the same dot8/fma chain). */
template<int GF>
static bool verify(unsigned ctx, unsigned nsplit){
    const unsigned nb=1, kvs=ctx, n_grp=NH/GF, n_work=nb*n_grp*nsplit;
    const float scale = 1.f/sqrtf((float)D);
    const size_t nkv_elems=(size_t)nb*NKV*kvs*D, nrows=(size_t)nb*NKV*kvs;
    std::vector<bf16> k(nkv_elems), v(nkv_elems);
    std::vector<uint8_t> kz(nrows*ROWB), vz(nrows*ROWB);
    size_t esc_rows=0;
    for (unsigned h=0;h<NKV;h++)
        for (unsigned i=0;i<kvs;i++){
            const unsigned s=src_row(h,i); const size_t r=(size_t)h*kvs+i;
            memcpy(&k[r*D],&g_krows[(size_t)s*D],D*2);
            memcpy(&v[r*D],&g_vrows[(size_t)s*D],D*2);
            sz_encode_row(&g_krows[(size_t)s*D],&kz[r*ROWB]);
            sz_encode_row(&g_vrows[(size_t)s*D],&vz[r*ROWB]);
            esc_rows += (kz[r*ROWB+D/2+D+1]!=0) + (vz[r*ROWB+D/2+D+1]!=0);
        }
    bf16 *dK,*dV; uint8_t *dKz,*dVz;
    CK(cudaMalloc(&dK,nkv_elems*2)); CK(cudaMalloc(&dV,nkv_elems*2));
    CK(cudaMalloc(&dKz,kz.size())); CK(cudaMalloc(&dVz,vz.size()));
    CK(cudaMemcpy(dK,k.data(),nkv_elems*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,v.data(),nkv_elems*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dKz,kz.data(),kz.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVz,vz.data(),vz.size(),cudaMemcpyHostToDevice));
    rng_s = 0x13579bdfu;
    std::vector<bf16> q((size_t)nb*NH*D); for (auto& x:q) x=__float2bfloat16(rnd()*0.5f);
    bf16* dQ; CK(cudaMalloc(&dQ,q.size()*2));
    CK(cudaMemcpy(dQ,q.data(),q.size()*2,cudaMemcpyHostToDevice));
    std::vector<int> kvlen(nb,(int)ctx); int* dL; CK(cudaMalloc(&dL,nb*4));
    CK(cudaMemcpy(dL,kvlen.data(),nb*4,cudaMemcpyHostToDevice));
    const size_t opN=(size_t)nb*NH*nsplit*D, mlN=(size_t)nb*NH*nsplit*2;
    float *dOp,*dMl; CK(cudaMalloc(&dOp,opN*4)); CK(cudaMalloc(&dMl,mlN*4));
    const size_t smem=(size_t)FA_DEC_SMEM_FLOATS(D,GF)*sizeof(float);
    CK(cudaFuncSetAttribute(k_flash_bf16<D,GF>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    CK(cudaFuncSetAttribute(k_flash_sz<D,GF>,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

    std::vector<float> op_bf(opN), ml_bf(mlN), op_sz(opN), ml_sz(mlN);
    k_flash_bf16<D,GF><<<n_work,256,smem>>>(dOp,dMl,dQ,dK,dV,dL,nb,NH,NKV,kvs,scale,nsplit);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    CK(cudaMemcpy(op_bf.data(),dOp,opN*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ml_bf.data(),dMl,mlN*4,cudaMemcpyDeviceToHost));
    k_flash_sz<D,GF><<<n_work,256,smem>>>(dOp,dMl,dQ,dKz,dVz,dL,nb,NH,NKV,kvs,scale,nsplit);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    CK(cudaMemcpy(op_sz.data(),dOp,opN*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ml_sz.data(),dMl,mlN*4,cudaMemcpyDeviceToHost));

    const bool ok = !memcmp(op_bf.data(),op_sz.data(),opN*4) &&
                    !memcmp(ml_bf.data(),ml_sz.data(),mlN*4);
    printf("# verify GF%d ctx %u: sz12 vs bf16 %s (%zu escape rows in cache)\n",
           GF,ctx,ok?"BIT-EXACT":"MISMATCH",esc_rows);
    cudaFree(dK);cudaFree(dV);cudaFree(dKz);cudaFree(dVz);cudaFree(dQ);cudaFree(dL);
    cudaFree(dOp);cudaFree(dMl);
    return ok;
}

/* One measured config. dt: 0=bf16 1=fp8 2=sz. Returns min ms over iters. */
template<int GF>
static Res run(int dt, unsigned ctx, unsigned nsplit, int iters){
    const unsigned nb=1, kvs=ctx;
    const unsigned n_grp = NH/GF;
    const unsigned n_work = nb*n_grp*nsplit;
    const float scale = 1.f/sqrtf((float)D);
    const size_t nkv_elems = (size_t)nb*NKV*kvs*D;
    const size_t nrows = (size_t)nb*NKV*kvs;

    void *dK=nullptr,*dV=nullptr; float *dKs=nullptr,*dVs=nullptr;
    if (dt==1){
        std::vector<uint8_t> k(nkv_elems), v(nkv_elems);
        for (size_t i=0;i<nkv_elems;i++){ __nv_fp8_e4m3 a(rnd()*0.5f), b(rnd()*0.5f);
            k[i]=*(uint8_t*)&a; v[i]=*(uint8_t*)&b; }
        CK(cudaMalloc(&dK,nkv_elems)); CK(cudaMalloc(&dV,nkv_elems));
        CK(cudaMemcpy(dK,k.data(),nkv_elems,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV,v.data(),nkv_elems,cudaMemcpyHostToDevice));
        std::vector<float> ks(nrows,1.0f), vs(nrows,1.0f);
        CK(cudaMalloc(&dKs,nrows*4)); CK(cudaMalloc(&dVs,nrows*4));
        CK(cudaMemcpy(dKs,ks.data(),nrows*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dVs,vs.data(),nrows*4,cudaMemcpyHostToDevice));
    } else if (dt==2){
        std::vector<uint8_t> kz(nrows*ROWB), vz(nrows*ROWB);
        for (unsigned h=0;h<NKV;h++)
            for (unsigned i=0;i<kvs;i++){
                const unsigned s=src_row(h,i); const size_t r=(size_t)h*kvs+i;
                sz_encode_row(&g_krows[(size_t)s*D],&kz[r*ROWB]);
                sz_encode_row(&g_vrows[(size_t)s*D],&vz[r*ROWB]);
            }
        CK(cudaMalloc(&dK,kz.size())); CK(cudaMalloc(&dV,vz.size()));
        CK(cudaMemcpy(dK,kz.data(),kz.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV,vz.data(),vz.size(),cudaMemcpyHostToDevice));
    } else {
        std::vector<bf16> k(nkv_elems), v(nkv_elems);
        if (g_nrows){ /* same logical bytes as the sz arm: the A/B compares like with like */
            for (unsigned h=0;h<NKV;h++)
                for (unsigned i=0;i<kvs;i++){
                    const unsigned s=src_row(h,i); const size_t r=((size_t)h*kvs+i)*D;
                    memcpy(&k[r],&g_krows[(size_t)s*D],D*2);
                    memcpy(&v[r],&g_vrows[(size_t)s*D],D*2);
                }
        } else {
            for (size_t i=0;i<nkv_elems;i++){ k[i]=__float2bfloat16(rnd()*0.5f); v[i]=__float2bfloat16(rnd()*0.5f); }
        }
        CK(cudaMalloc(&dK,nkv_elems*2)); CK(cudaMalloc(&dV,nkv_elems*2));
        CK(cudaMemcpy(dK,k.data(),nkv_elems*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV,v.data(),nkv_elems*2,cudaMemcpyHostToDevice));
    }
    rng_s = 0x13579bdfu; /* same Q for every arm */
    std::vector<bf16> q((size_t)nb*NH*D); for (auto& x:q) x=__float2bfloat16(rnd()*0.5f);
    bf16* dQ=nullptr; CK(cudaMalloc(&dQ,q.size()*2));
    CK(cudaMemcpy(dQ,q.data(),q.size()*2,cudaMemcpyHostToDevice));
    std::vector<int> kvlen(nb,(int)ctx); int* dL=nullptr; CK(cudaMalloc(&dL,nb*4));
    CK(cudaMemcpy(dL,kvlen.data(),nb*4,cudaMemcpyHostToDevice));
    float *dOp=nullptr,*dMl=nullptr;
    const size_t opN=(size_t)nb*NH*nsplit*D, mlN=(size_t)nb*NH*nsplit*2;
    CK(cudaMalloc(&dOp,opN*4)); CK(cudaMalloc(&dMl,mlN*4));

    const size_t smem=(size_t)FA_DEC_SMEM_FLOATS(D,GF)*sizeof(float);
    auto launch=[&](){
        if (dt==1)      k_flash_fp8<D,GF><<<n_work,256,smem>>>(dOp,dMl,dQ,(bf16*)dK,(bf16*)dV,dL,nb,NH,NKV,kvs,scale,nsplit,dKs,dVs);
        else if (dt==2) k_flash_sz<D,GF><<<n_work,256,smem>>>(dOp,dMl,dQ,(uint8_t*)dK,(uint8_t*)dV,dL,nb,NH,NKV,kvs,scale,nsplit);
        else            k_flash_bf16<D,GF><<<n_work,256,smem>>>(dOp,dMl,dQ,(bf16*)dK,(bf16*)dV,dL,nb,NH,NKV,kvs,scale,nsplit);
    };
    if (dt==1)      CK(cudaFuncSetAttribute(k_flash_fp8<D,GF>,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    else if (dt==2) CK(cudaFuncSetAttribute(k_flash_sz<D,GF>,   cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    else            CK(cudaFuncSetAttribute(k_flash_bf16<D,GF>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

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
    cudaFree(dK);cudaFree(dV);cudaFree(dQ);cudaFree(dL);cudaFree(dOp);cudaFree(dMl);
    if(dKs)cudaFree(dKs); if(dVs)cudaFree(dVs);

    const double elem = dt==1 ? 1.0 : (dt==2 ? (double)ROWB/(double)D : 2.0);
    const double bytes_issued = (double)n_grp*ctx*D*elem*2.0;   /* K+V, per group */
    const double bytes_phys   = (double)NKV  *ctx*D*elem*2.0;   /* distinct HBM bytes */
    Res r; r.ms=best; r.gbps_issued=bytes_issued/(best*1e-3)/1e9; r.gbps_phys=bytes_phys/(best*1e-3)/1e9;
    return r;
}

int main(int argc, char** argv){
    int iters = argc>1?atoi(argv[1]):50;
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("# device %s, SMs %d, iters %d\n", p.name, p.multiProcessorCount, iters);
    printf("# geom: n_head=%u n_kv_head=%u gqa=%u D=%u (Gemma-4-26B-A4B full-attn)\n", NH,NKV,NH/NKV,D);
    printf("# HBM ceiling assumed 1535 GB/s\n");
    const bool have_dump = load_dump();
    if (!have_dump) printf("# no KV dump found: sz arm SKIPPED, bf16/fp8 random-filled\n");
    if (have_dump){ /* numerics gate BEFORE any timing: bit-exact or the sz numbers are void */
        if (!verify<4>(32768u,48) || !verify<8>(32768u,96)){
            printf("# sz12 verify FAILED — aborting\n"); return 4;
        }
    }

    const unsigned ctxs[]={32768u,65536u,98304u,131072u};
    /* nsplit chosen to keep n_work ~= 188 (1 item/SM at 1 block/SM): GF4 n_grp=4 -> ns48 (192);
     * GF8 n_grp=2 -> ns96 (192). Grid-aligned to the shipped packet's PLOW_NS_FULL_ABS=48. */

    printf("\n## FlashDecode<512> KV-read microbench (best of %d)\n", iters);
    printf("%-6s %-6s %-4s | %8s | %10s %10s | %10s\n",
           "cfg","ctx","GF","ms","GBps_iss","GBps_phys","%ceil_iss");

    double ms_bf4[4], ms_fp4[4], ms_fp8gf8[4], ms_bf8[4], ms_sz4[4], ms_sz8[4];

    for (int ci=0; ci<4; ci++){
        unsigned ctx=ctxs[ci];
        Res r;
        r=run<4>(0,ctx,48,iters); ms_bf4[ci]=r.ms;
        printf("%-6s %-6u %-4d | %8.4f | %10.1f %10.1f | %9.1f\n","bf16",ctx,4,r.ms,r.gbps_issued,r.gbps_phys,100.0*r.gbps_issued/1535.0);
        r=run<8>(0,ctx,96,iters); ms_bf8[ci]=r.ms;
        printf("%-6s %-6u %-4d | %8.4f | %10.1f %10.1f | %9.1f\n","bf16",ctx,8,r.ms,r.gbps_issued,r.gbps_phys,100.0*r.gbps_issued/1535.0);
        if (have_dump){
            r=run<4>(2,ctx,48,iters); ms_sz4[ci]=r.ms;
            printf("%-6s %-6u %-4d | %8.4f | %10.1f %10.1f | %9.1f  (vs bf16: %.3fx)\n","sz12",ctx,4,r.ms,r.gbps_issued,r.gbps_phys,100.0*r.gbps_issued/1535.0,ms_bf4[ci]/r.ms);
            r=run<8>(2,ctx,96,iters); ms_sz8[ci]=r.ms;
            printf("%-6s %-6u %-4d | %8.4f | %10.1f %10.1f | %9.1f  (vs bf16: %.3fx)\n","sz12",ctx,8,r.ms,r.gbps_issued,r.gbps_phys,100.0*r.gbps_issued/1535.0,ms_bf8[ci]/r.ms);
        }
        r=run<4>(1,ctx,48,iters); ms_fp4[ci]=r.ms;
        printf("%-6s %-6u %-4d | %8.4f | %10.1f %10.1f | %9.1f\n","fp8",ctx,4,r.ms,r.gbps_issued,r.gbps_phys,100.0*r.gbps_issued/1535.0);
        r=run<8>(1,ctx,96,iters); ms_fp8gf8[ci]=r.ms;
        printf("%-6s %-6u %-4d | %8.4f | %10.1f %10.1f | %9.1f\n","fp8",ctx,8,r.ms,r.gbps_issued,r.gbps_phys,100.0*r.gbps_issued/1535.0);
        printf("\n");
    }

    /* Per-layer marginal ns per KV-token, 32k->128k slope, and the x5-full-layer model projection. */
    auto slope=[&](double* ms){ /* ms over ctxs[]; ns per ctx-token = d(ms)*1e6 / d(ctx) */
        return (ms[3]-ms[0])*1e6/(double)(ctxs[3]-ctxs[0]); };
    printf("## Marginal KV cost (32k->128k), per full-attn LAYER and x5 model projection\n");
    printf("%-14s %12s %12s\n","cfg","ns/tok(layer)","ns/tok(x5)");
    printf("%-14s %12.3f %12.3f\n","bf16 GF4",slope(ms_bf4),5*slope(ms_bf4));
    printf("%-14s %12.3f %12.3f\n","bf16 GF8",slope(ms_bf8),5*slope(ms_bf8));
    if (have_dump){
        printf("%-14s %12.3f %12.3f\n","sz12 GF4",slope(ms_sz4),5*slope(ms_sz4));
        printf("%-14s %12.3f %12.3f\n","sz12 GF8",slope(ms_sz8),5*slope(ms_sz8));
    }
    printf("%-14s %12.3f %12.3f\n","fp8  GF4",slope(ms_fp4),5*slope(ms_fp4));
    printf("%-14s %12.3f %12.3f\n","fp8  GF8",slope(ms_fp8gf8),5*slope(ms_fp8gf8));
    printf("# campaign model marginal: plow 21.3 ns/tok, vLLM 12.0 ns/tok\n");
    return 0;
}
