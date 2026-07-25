/* moe_group_bench.cu — SINGLE-BLOCK grouped-GEMM microbench for the 26B-A4B MoE PREFILL
 * ops 75 (MOE_GROUP_GLU_GEMMA_PF) and 76 (MOE_GROUP_DOWN_GEMMA_PF), in ISOLATION.
 *
 * Mandate: measure the grouped GLU/DOWN GEMM tile/scheduler variants BEFORE full-model sweeps,
 * pick the winner, then emit the full blob. This bench:
 *   (1) times the CURRENT bf16 grouped GLU + DOWN (op_moe.cuh d_moe_group_*_pf) at real geometry
 *       (H2816/E128/k8/I704) over the prefill chunk ladder T in {512,2048,8192};
 *   (2) implements + validates + times the fp8 (w8a8) grouped GLU + DOWN reusing the 08a2bdd
 *       dense w8a8 mainloop (pgm_*_w8a8 / pgm_mma_fp8_k32, BK8=64, m16n8k32 e4m3);
 *   (3) tile A/B is via -DPGM_BN / -DPGM_STAGES / -DPGM_GLU_STAGES at build time.
 *
 * Same launch config as the megakernel: grid=188 blocks, 256 threads, block-stride work list.
 * Build: nvcc -arch=sm_120a -O3 -DPLOW_NV_GEMMA=1 -I ../common -I ../nvidia moe_group_bench.cu
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sm120_common.cuh"
#include "op_norm.cuh"
#include "op_elementwise.cuh"
#include "op_gemm.cuh"
#include "dev_isa.h"
#include "op_moe.cuh"

typedef __nv_bfloat16 bf16;

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess){ \
    printf("CUDA ERROR %s at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(2);} } while(0)

static uint32_t rng_s = 0x2468ace0u;
static float rnd() { rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
    return (float)((int32_t)rng_s) / 2147483648.0f; }
static void seed(uint32_t s) { rng_s = s ? s : 1; }
static float bf16_rt(float x) { return __bfloat162float(__float2bfloat16(x)); }
static std::vector<float> gen_bf16(size_t n, float amp) {
    std::vector<float> v(n); for (size_t i=0;i<n;i++) v[i]=bf16_rt(rnd()*amp); return v;
}
template<class T> static T* to_dev(const std::vector<T>& h){ T* d=nullptr; CK(cudaMalloc(&d,h.size()*sizeof(T)));
    CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice)); return d; }
static bf16* to_dev_bf(const std::vector<float>& h){ std::vector<bf16> t(h.size());
    for(size_t i=0;i<h.size();i++) t[i]=__float2bfloat16(h[i]); return to_dev(t); }

/* per-row (per-output-channel / per-token) e4m3 quant: scale = rowmax/448, q = round_e4m3(v/scale). */
static void quant_rows(const std::vector<float>& v, size_t rows, size_t width,
                       std::vector<uint8_t>& q, std::vector<float>& scale) {
    q.resize(rows*width); scale.resize(rows);
    for (size_t r=0;r<rows;r++){ float amax=0; for(size_t c=0;c<width;c++) amax=fmaxf(amax,fabsf(v[r*width+c]));
        float s = amax>0 ? amax/448.0f : 1.0f; scale[r]=s;
        for(size_t c=0;c<width;c++){ __nv_fp8_e4m3 e(v[r*width+c]/s); q[r*width+c]=e.__x; } }
}

/* ============================ fp8 (w8a8) GROUPED GEMM (NEW — bench) ============================
 * Reuses the dense w8a8 helpers from op_gemm.cuh (pgm_stage_a8/b8, pgm_load_*frags_w8a8,
 * pgm_mma_fp8_k32, pgm_sw8). Only the A stage (gathered) and the epilogue (per-token/per-row
 * activation scale x per-channel weight scale) differ from d_gemm_*_w8a8. */

/* gather e4m3 A rows by row_token into a swizzled [BM][BK8] fp8 tile (K contiguous). */
__device__ __forceinline__ void pgm_stage_a8_gather(uint8_t* Ad8, const uint8_t* __restrict__ A,
        const unsigned* __restrict__ rowsrc, int tid, int rowbase, int kbase, unsigned k) {
    const int LCH = PGM_BK8 / 16;
    for (int L = tid; L < PGM_BM * LCH; L += (int)PLOW_NV_THREADS) {
        const int row = L / LCH, kk16 = (L % LCH) * 16;
        const unsigned src = rowsrc[rowbase + row];
        const int kk = kbase + kk16;
        const bool in = (src != PLOW_EXPERT_UNUSED) && (kk + 16 <= (int)k);
        const uint8_t* g = in ? A + (size_t)src * k + kk : A;
        pgm_cp_async_cg16(&Ad8[pgm_sw8(row * PGM_BK8 + kk16)], g, in ? 16 : 0);
    }
}

/* GROUPED GLU, w8a8. A gathered fp8 (xq8 by row_token) + per-token ascale; Wg/Wu fp8 + per-chan sg/su. */
static __device__ void d_moe_group_glu_gemma_pf_w8a8(
        bf16* __restrict__ fu, const uint8_t* __restrict__ xq8, const float* __restrict__ ascale,
        const unsigned long long* __restrict__ ewt, const unsigned long long* __restrict__ est,
        const int* __restrict__ meta, const unsigned* __restrict__ row_token,
        unsigned I_moe, unsigned H, unsigned n_exp, unsigned act, unsigned slice, unsigned nblk,
        bf16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)I_moe + PGM_BN - 1) / PGM_BN;
    const int ntiles = total_tiles * tiles_n;
    const unsigned K = H;
    const int ksteps = ((int)K + PGM_BK8 - 1) / PGM_BK8;
    uint8_t* As = (uint8_t*)arena;
    uint8_t* Bg = As + PGM_GLU_STAGES * PGM_A8BUF8;
    uint8_t* Bu = Bg + PGM_GLU_STAGES * PGM_B8BUF8;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n, ntile = tile % tiles_n;
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * PGM_BM;
        const int tn = ntile * PGM_BN;
        const uint8_t* Wg = (const uint8_t*)(size_t)ewt[(size_t)e * 2 + 0];
        const uint8_t* Wu = Wg + (size_t)I_moe * H;
        const float* sc = (const float*)(size_t)est[(size_t)e * 2 + 0]; /* [2*I_moe] */

        float accg[PGM_MFRAG][PGM_NFRAG][4], accu[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++) for (int j = 0; j < PGM_NFRAG; j++)
            for (int ee = 0; ee < 4; ee++) { accg[i][j][ee]=0.f; accu[i][j][ee]=0.f; }

        auto stage = [&](int ks, int buf) {
            pgm_stage_a8_gather(As + buf * PGM_A8BUF8, xq8, row_token, tid, rowbase, ks * PGM_BK8, K);
            pgm_stage_b8(Bg + buf * PGM_B8BUF8, Wg, tid, tn, ks * PGM_BK8, I_moe, K, (int)K);
            pgm_stage_b8(Bu + buf * PGM_B8BUF8, Wu, tid, tn, ks * PGM_BK8, I_moe, K, (int)K);
        };
#pragma unroll
        for (int s = 0; s < PGM_GLU_STAGES - 1; s++) { if (s < ksteps) stage(s, s); pgm_cp_commit(); }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_GLU_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_GLU_STAGES);
            pgm_cp_commit(); pgm_cp_wait<PGM_GLU_STAGES - 1>(); __syncthreads();
            const int cb = ks % PGM_GLU_STAGES;
#pragma unroll
            for (int kf = 0; kf < PGM_BK8; kf += 32) {
                unsigned af[PGM_MFRAG][4]; pgm_load_afrags_w8a8(af, As + cb * PGM_A8BUF8, wm, kf, lane);
                unsigned bg[PGM_NFRAG][2], bu[PGM_NFRAG][2];
                pgm_load_bfrags_w8a8(bg, Bg + cb * PGM_B8BUF8, wn, kf, lane);
                pgm_load_bfrags_w8a8(bu, Bu + cb * PGM_B8BUF8, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++) {
                        pgm_mma_fp8_k32(accg[mi][nj], af[mi], bg[nj], accg[mi][nj]);
                        pgm_mma_fp8_k32(accu[mi][nj], af[mi], bu[nj], accu[mi][nj]);
                    }
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int ee = 0; ee < 4; ee++) {
                    int rr = gr + (ee / 2) * 8, cc = tn + gc + (ee % 2);
                    if (rr < PGM_BM && cc < (int)I_moe) {
                        const unsigned tok = row_token[rowbase + rr];
                        if (tok == PLOW_EXPERT_UNUSED) continue; /* pad row: no valid token/ascale */
                        const float as = ascale[tok];
                        float g = accg[mi][nj][ee] * as * sc[cc];
                        float u = accu[mi][nj][ee] * as * sc[I_moe + cc];
                        float a = (act == PLOW_ACT_SILU_) ? act_silu(g) : act_gelu_tanh(g);
                        fu[(size_t)(rowbase + rr) * I_moe + cc] = __float2bfloat16(a * u);
                    }
                }
            }
        __syncthreads();
    }
}

/* GROUPED DOWN, w8a8. A = fu8 (contiguous, per gathered-row scale fscale); Wd fp8 + per-chan dscale. */
static __device__ void d_moe_group_down_gemma_pf_w8a8(
        float* __restrict__ part, const uint8_t* __restrict__ fu8, const float* __restrict__ fscale,
        const unsigned long long* __restrict__ ewt, const unsigned long long* __restrict__ est,
        const int* __restrict__ meta, const unsigned* __restrict__ row_partidx,
        const float* __restrict__ row_gate, unsigned H, unsigned I_moe, unsigned n_exp,
        unsigned slice, unsigned nblk, bf16* arena) {
    const int* rowoff = meta;
    const int* tilep = meta + 2 * (int)n_exp;
    const int total_tiles = tilep[n_exp];
    const int tiles_n = ((int)H + PGM_BN - 1) / PGM_BN;
    const int ntiles = total_tiles * tiles_n;
    const unsigned K = I_moe;
    const int ksteps = ((int)K + PGM_BK8 - 1) / PGM_BK8;
    uint8_t* As = (uint8_t*)arena;
    uint8_t* Bs = As + PGM_STAGES * PGM_A8BUF8;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / PGM_WARPS_N, wn = warp % PGM_WARPS_N;

    for (int tile = (int)slice; tile < ntiles; tile += (int)nblk) {
        const int mtile = tile / tiles_n, ntile = tile % tiles_n;
        const int e = pgm_moe_expert_of_mtile(tilep, mtile, (int)n_exp);
        const int rowbase = rowoff[e] + (mtile - tilep[e]) * PGM_BM;
        const int tn = ntile * PGM_BN;
        const uint8_t* Wd = (const uint8_t*)(size_t)ewt[(size_t)e * 2 + 1];
        const float* dsc = (const float*)(size_t)est[(size_t)e * 2 + 1]; /* [H] */

        float acc[PGM_MFRAG][PGM_NFRAG][4];
#pragma unroll
        for (int i = 0; i < PGM_MFRAG; i++) for (int j = 0; j < PGM_NFRAG; j++)
            for (int ee = 0; ee < 4; ee++) acc[i][j][ee]=0.f;

        auto stage = [&](int ks, int buf) {
            pgm_stage_a8(As + buf * PGM_A8BUF8, fu8, tid, 0, ks * PGM_BK8, PGM_BM, K, rowbase);
            pgm_stage_b8(Bs + buf * PGM_B8BUF8, Wd, tid, tn, ks * PGM_BK8, H, K, (int)K);
        };
#pragma unroll
        for (int s = 0; s < PGM_STAGES - 1; s++) { if (s < ksteps) stage(s, s); pgm_cp_commit(); }
        for (int ks = 0; ks < ksteps; ks++) {
            const int fetch = ks + PGM_STAGES - 1;
            if (fetch < ksteps) stage(fetch, fetch % PGM_STAGES);
            pgm_cp_commit(); pgm_cp_wait<PGM_STAGES - 1>(); __syncthreads();
            const int cb = ks % PGM_STAGES;
#pragma unroll
            for (int kf = 0; kf < PGM_BK8; kf += 32) {
                unsigned af[PGM_MFRAG][4]; pgm_load_afrags_w8a8(af, As + cb * PGM_A8BUF8, wm, kf, lane);
                unsigned bf[PGM_NFRAG][2]; pgm_load_bfrags_w8a8(bf, Bs + cb * PGM_B8BUF8, wn, kf, lane);
#pragma unroll
                for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
                    for (int nj = 0; nj < PGM_NFRAG; nj++)
                        pgm_mma_fp8_k32(acc[mi][nj], af[mi], bf[nj], acc[mi][nj]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < PGM_MFRAG; mi++)
#pragma unroll
            for (int nj = 0; nj < PGM_NFRAG; nj++) {
                int gr = wm * PGM_WM + mi * 16 + (lane / 4);
                int gc = wn * PGM_WN + nj * 8 + (lane % 4) * 2;
#pragma unroll
                for (int ee = 0; ee < 4; ee++) {
                    int rr = gr + (ee / 2) * 8, cc = tn + gc + (ee % 2);
                    if (rr < PGM_BM && cc < (int)H) {
                        const unsigned pidx = row_partidx[rowbase + rr];
                        if (pidx != PLOW_EXPERT_UNUSED)
                            part[(size_t)pidx * H + cc] =
                                row_gate[rowbase + rr] * fscale[rowbase + rr] * dsc[cc] * acc[mi][nj][ee];
                    }
                }
            }
        __syncthreads();
    }
}

/* ---- wrapper kernels ---- */
__global__ void k_align(int* meta, const unsigned char* table, unsigned* rt, unsigned* rp,
                        float* rg, unsigned T, unsigned E, unsigned k){
    d_moe_align_gemma_pf(meta,table,rt,rp,rg,T,E,k,blockIdx.x);
}
__global__ void k_glu_bf16(bf16* fu, const bf16* xn2, const uint64_t* ewt, const int* meta,
                          const unsigned* rt, unsigned I, unsigned H, unsigned E, unsigned act){
    extern __shared__ bf16 sm[];
    d_moe_group_glu_gemma_pf(fu,xn2,(const unsigned long long*)ewt,meta,rt,I,H,E,act,blockIdx.x,gridDim.x,sm);
}
__global__ void k_down_bf16(float* part, const bf16* fu, const uint64_t* ewt, const int* meta,
                           const unsigned* rp, const float* rg, unsigned H, unsigned I, unsigned E){
    extern __shared__ bf16 sm[];
    d_moe_group_down_gemma_pf(part,fu,(const unsigned long long*)ewt,meta,rp,rg,H,I,E,blockIdx.x,gridDim.x,sm);
}
__global__ void k_glu_fp8(bf16* fu, const uint8_t* xq8, const float* ascale, const uint64_t* ewt,
                         const uint64_t* est, const int* meta, const unsigned* rt,
                         unsigned I, unsigned H, unsigned E, unsigned act){
    extern __shared__ bf16 sm[];
    d_moe_group_glu_gemma_pf_w8a8(fu,xq8,ascale,(const unsigned long long*)ewt,(const unsigned long long*)est,
                                  meta,rt,I,H,E,act,blockIdx.x,gridDim.x,sm);
}
__global__ void k_down_fp8(float* part, const uint8_t* fu8, const float* fscale, const uint64_t* ewt,
                          const uint64_t* est, const int* meta, const unsigned* rp, const float* rg,
                          unsigned H, unsigned I, unsigned E){
    extern __shared__ bf16 sm[];
    d_moe_group_down_gemma_pf_w8a8(part,fu8,fscale,(const unsigned long long*)ewt,(const unsigned long long*)est,
                                   meta,rp,rg,H,I,E,blockIdx.x,gridDim.x,sm);
}

static double reL2(const std::vector<float>& a, const std::vector<float>& b){
    double num=0,den=0; for(size_t i=0;i<a.size();i++){ double d=(double)a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; }
    return den>0? sqrt(num/den) : 0.0;
}

int main(int argc, char** argv){
    int dev=0; cudaDeviceProp p; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p,dev));
    printf("device: %s sm_%d%d SMs=%d | tile BM=%d BN=%d BK=%d BK8=%d GLU_STAGES=%d STAGES=%d\n",
           p.name,p.major,p.minor,p.multiProcessorCount,PGM_BM,PGM_BN,PGM_BK,PGM_BK8,PGM_GLU_STAGES,PGM_STAGES);
    const unsigned H=2816,E=128,k=8,I=704;
    const int GRID=p.multiProcessorCount, ITERS=50, WARM=10;
    std::vector<unsigned> Ts;
    if(argc>1){ for(int i=1;i<argc;i++) Ts.push_back((unsigned)atoi(argv[i])); }
    else { Ts={512,2048,8192}; }

    const size_t smg=(size_t)PGM_ARENA_BF16*sizeof(bf16);
    CK(cudaFuncSetAttribute(k_glu_bf16,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smg));
    CK(cudaFuncSetAttribute(k_down_bf16,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smg));
    CK(cudaFuncSetAttribute(k_glu_fp8,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smg));
    CK(cudaFuncSetAttribute(k_down_fp8,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smg));

    /* shared expert weights (per expert): gate_up [2I,H] bf16 + e4m3/scale; down [H,I] bf16 + e4m3/scale. */
    seed(1234);
    std::vector<float> gu=gen_bf16((size_t)E*2*I*H,0.05f);   /* small amp keeps fp8 range sane */
    std::vector<float> dw=gen_bf16((size_t)E*H*I,0.05f);
    std::vector<uint8_t> gu8,dw8; std::vector<float> guS,dwS;
    quant_rows(gu,(size_t)E*2*I,H,gu8,guS);   /* per-out-channel over the [E*2I] rows */
    quant_rows(dw,(size_t)E*H,I,dw8,dwS);
    bf16* dGU=to_dev_bf(gu); bf16* dDW=to_dev_bf(dw);
    uint8_t* dGU8=to_dev(gu8); uint8_t* dDW8=to_dev(dw8);
    float* dGUs=to_dev(guS); float* dDWs=to_dev(dwS);
    std::vector<uint64_t> ewt((size_t)E*2), est((size_t)E*2);
    for(unsigned e=0;e<E;e++){ ewt[e*2+0]=(uint64_t)(dGU+(size_t)e*2*I*H); ewt[e*2+1]=(uint64_t)(dDW+(size_t)e*H*I);
        est[e*2+0]=0; est[e*2+1]=0; }
    std::vector<uint64_t> ewt8((size_t)E*2), est8((size_t)E*2);
    for(unsigned e=0;e<E;e++){ ewt8[e*2+0]=(uint64_t)(dGU8+(size_t)e*2*I*H); ewt8[e*2+1]=(uint64_t)(dDW8+(size_t)e*H*I);
        est8[e*2+0]=(uint64_t)(dGUs+(size_t)e*2*I); est8[e*2+1]=(uint64_t)(dDWs+(size_t)e*H); }
    uint64_t* dEwt=to_dev(ewt); uint64_t* dEwt8=to_dev(ewt8); uint64_t* dEst8=to_dev(est8);

    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    printf("\n%-6s %-22s %10s %10s %8s\n","T","op","ms","GFLOP/s","relL2");
    for(unsigned T: Ts){
        seed(0x900u+T);
        std::vector<float> xn2=gen_bf16((size_t)T*H,1.0f);
        /* random routing: each of T*k slots -> random expert, gate ~ U(0,1). table = [T*k]{u32 eid, f32 gate}. */
        std::vector<unsigned char> tab((size_t)T*k*8);
        for(unsigned s=0;s<T*k;s++){ unsigned e=((unsigned)(fabsf(rnd())*E))%E; float g=fabsf(rnd());
            *(unsigned*)(tab.data()+(size_t)s*8)=e; *(float*)(tab.data()+(size_t)s*8+4)=g; }
        unsigned char* dTab=to_dev(tab);
        const unsigned BM=128, total_pad=T*k+E*BM;
        bf16* dXn2=to_dev_bf(xn2);
        int* dMeta=nullptr; CK(cudaMalloc(&dMeta,(size_t)(3*E+2)*4));
        unsigned *dRt=nullptr,*dRp=nullptr; float* dRg=nullptr;
        CK(cudaMalloc(&dRt,(size_t)total_pad*4)); CK(cudaMalloc(&dRp,(size_t)total_pad*4));
        CK(cudaMalloc(&dRg,(size_t)total_pad*4));
        CK(cudaMemset(dRt,0xFF,(size_t)total_pad*4)); CK(cudaMemset(dRp,0xFF,(size_t)total_pad*4));
        CK(cudaMemset(dRg,0,(size_t)total_pad*4));
        bf16* dFu=nullptr; CK(cudaMalloc(&dFu,(size_t)total_pad*I*sizeof(bf16)));
        float* dPart=nullptr; CK(cudaMalloc(&dPart,(size_t)T*k*H*sizeof(float)));
        k_align<<<GRID,256>>>(dMeta,dTab,dRt,dRp,dRg,T,E,k); CK(cudaDeviceSynchronize());

        /* fp8 activation twin: xq8 [T,H] + ascale[T]. */
        std::vector<uint8_t> xq8; std::vector<float> xaS; quant_rows(xn2,T,H,xq8,xaS);
        uint8_t* dXq8=to_dev(xq8); float* dXaS=to_dev(xaS);

        auto timeit=[&](auto launch)->double{
            for(int i=0;i<WARM;i++) launch(); CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(t0)); for(int i=0;i<ITERS;i++) launch(); CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1)); float ms=0; CK(cudaEventElapsedTime(&ms,t0,t1)); return ms/ITERS;
        };

        /* --- bf16 GLU --- */
        double glu_ms=timeit([&]{ k_glu_bf16<<<GRID,256,smg>>>(dFu,dXn2,dEwt,dMeta,dRt,I,H,E,PLOW_ACT_GELU_TANH_); });
        std::vector<float> fu_bf16((size_t)total_pad*I);
        { std::vector<bf16> t((size_t)total_pad*I); CK(cudaMemcpy(t.data(),dFu,t.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
          for(size_t i=0;i<t.size();i++) fu_bf16[i]=__bfloat162float(t[i]); }
        double gflop_glu = 2.0*(double)(T*k)*(2.0*I)*H/1e9;

        /* --- bf16 DOWN --- */
        double dn_ms=timeit([&]{ k_down_bf16<<<GRID,256,smg>>>(dPart,dFu,dEwt,dMeta,dRp,dRg,H,I,E); });
        std::vector<float> part_bf16((size_t)T*k*H); CK(cudaMemcpy(part_bf16.data(),dPart,part_bf16.size()*4,cudaMemcpyDeviceToHost));
        double gflop_dn = 2.0*(double)(T*k)*H*I/1e9;

        /* --- fp8 GLU --- */
        double glu8_ms=timeit([&]{ k_glu_fp8<<<GRID,256,smg>>>(dFu,dXq8,dXaS,dEwt8,dEst8,dMeta,dRt,I,H,E,PLOW_ACT_GELU_TANH_); });
        std::vector<float> fu_fp8((size_t)total_pad*I);
        { std::vector<bf16> t((size_t)total_pad*I); CK(cudaMemcpy(t.data(),dFu,t.size()*sizeof(bf16),cudaMemcpyDeviceToHost));
          for(size_t i=0;i<t.size();i++) fu_fp8[i]=__bfloat162float(t[i]); }

        /* fp8 DOWN: quantize the bf16 GLU output (shared) so the down fp8 error is isolated. */
        std::vector<uint8_t> fu8; std::vector<float> fuS; quant_rows(fu_bf16,total_pad,I,fu8,fuS);
        uint8_t* dFu8=to_dev(fu8); float* dFuS=to_dev(fuS);
        double dn8_ms=timeit([&]{ k_down_fp8<<<GRID,256,smg>>>(dPart,dFu8,dFuS,dEwt8,dEst8,dMeta,dRp,dRg,H,I,E); });
        std::vector<float> part_fp8((size_t)T*k*H); CK(cudaMemcpy(part_fp8.data(),dPart,part_fp8.size()*4,cudaMemcpyDeviceToHost));

        printf("%-6u %-22s %10.4f %10.1f %8s\n",T,"GLU bf16",glu_ms,gflop_glu/(glu_ms/1e3),"-");
        printf("%-6u %-22s %10.4f %10.1f %8.1e\n",T,"GLU fp8(w8a8)",glu8_ms,gflop_glu/(glu8_ms/1e3),reL2(fu_fp8,fu_bf16));
        printf("%-6u %-22s %10.4f %10.1f %8s\n",T,"DOWN bf16",dn_ms,gflop_dn/(dn_ms/1e3),"-");
        printf("%-6u %-22s %10.4f %10.1f %8.1e\n",T,"DOWN fp8(w8a8)",dn8_ms,gflop_dn/(dn8_ms/1e3),reL2(part_fp8,part_bf16));
        printf("%-6u %-22s %10.4f fp8/bf16 GLU=%.2fx DOWN=%.2fx\n",T,"SUMMARY",
               (glu_ms+dn_ms),glu8_ms>0?glu_ms/glu8_ms:0,dn8_ms>0?dn_ms/dn8_ms:0);
        fflush(stdout);
        cudaFree(dTab);cudaFree(dXn2);cudaFree(dMeta);cudaFree(dRt);cudaFree(dRp);cudaFree(dRg);
        cudaFree(dFu);cudaFree(dPart);cudaFree(dXq8);cudaFree(dXaS);cudaFree(dFu8);cudaFree(dFuS);
    }
    return 0;
}
