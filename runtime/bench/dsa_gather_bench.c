/* dsa_gather_bench.c — single-GLM-block DSA sparse-attention lever, MEASURED end-to-end. [DSA/G5]
 *
 * Drives, per ctx in {8k,32k,128k}, one GLM MLA decode block's attention two ways and times each
 * kernel directly (no interpreter): the REAL lightning indexer -> radix top-k select -> gathered
 * flash, versus the dense full-ctx flash. Produces the first in-block dense-vs-gather number.
 *
 *   index_score_128 / index_score_fast_128  score[t]=sum_h w[h]*ReLU(q[h].k[t])   (G3 + perf follow-up)
 *   index_select                            radix top-k over the HBM scores -> idx  (G4, the selector)
 *   mla_gather_decode_512 -> merge -> fold  gathered MLA over the top_k=2048 rows   (op 54 path)
 *   mla_flash_decode_512  -> merge -> fold  dense MLA over all ctx rows             (op 50 path)
 *
 * Correctness gates (independent CPU references over the same reduction boundaries):
 *   1. index_score  vs CPU weighted-ReLU score  (relmax)
 *   2. index_select set == CPU radix top-k set   (EXACT, lowest-index tie-break)
 *   3. gathered MLA over the DEVICE-selected set vs CPU MLA over the SAME set (tolerance) — the
 *      gather KERNEL correctness, isolated from selection.
 * Also reports dense-full vs gather relmax as the sparsification error (synthetic data => the top-k
 * of random scores is not the top-k of attention, so this number is a data-independence sanity read,
 * NOT the model-accuracy claim — that needs real weights, G6).
 *
 * Build: scripts/build_dsa_bench.sh ; run under `sg render`, pin with ROCR_VISIBLE_DEVICES.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NH      8       /* tp8 attention heads (matches the design-doc lever table)  */
#define DK      512     /* kv_lora_rank (latent)                                     */
#define DR      64      /* rope dim                                                  */
#define VD      128     /* v_head_dim                                                */
#define HI      32      /* index_n_heads                                             */
#define DI      128     /* index_head_dim                                            */
#define TOPK    2048    /* index_topk                                                */
#define NSPLIT  16      /* split-K (ns16, matches the lever table)                   */
#define SCALE_MLA 0.08838835f
#define SCALE_IDX 0.08838835f  /* 1/sqrt(128) */

typedef uint16_t bf16;
static float b2f(bf16 v){ union{uint32_t u;float f;}c; c.u=(uint32_t)v<<16; return c.f; }
static bf16 f2b(float f){ union{float f;uint32_t u;}c; c.f=f; uint32_t r=c.u+0x7fff+((c.u>>16)&1); return (bf16)(r>>16); }
static uint64_t rs; static void seed(uint64_t s){ rs=s*6364136223846793005ULL+1442695040888963407ULL; }
static float frand(void){ rs=rs*6364136223846793005ULL+1442695040888963407ULL; return (float)((int32_t)(rs>>33)%2001-1000)/4000.0f; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static plow_hsa* H;
static void* dev(size_t n){ return plow_hsa_alloc(H,0,n); }
static void up(void* d, const void* s, size_t n){ plow_hsa_upload(H,0,d,s,n); }

/* ---- CPU references ---- */
static void ref_score(const bf16* Qi,const bf16* Ki,const bf16* W,unsigned ctx,float* sc){
    for(unsigned t=0;t<ctx;t++){ float s=0;
        for(int h=0;h<HI;h++){ float d=0;
            for(int i=0;i<DI;i++) d+=b2f(Qi[h*DI+i])*b2f(Ki[(size_t)t*DI+i]);
            s+=b2f(W[h])*(d>0?d:0); }
        sc[t]=s*SCALE_IDX; }
}
/* radix-ref: the SAME packed key + top-k as the kernel; returns a selected[ctx] bitmap. */
/* Top-k over the FIRST `len` scores only, selecting `k` of them. `len`/`k` are parameters (not the
 * ctx/TOPK constants they used to be) because the short-context gate needs the reference to model
 * what the device is now contractually required to do: rank only the rows INDEX_SCORE actually
 * wrote (kv_len of them) and select min(top_k, kv_len). */
static void ref_select_k(const float* sc,unsigned len,unsigned k,char* selected,unsigned selbytes){
    memset(selected,0,selbytes);
    if(len==0||k==0) return;
    if(k>len) k=len;
    unsigned long long* key=malloc((size_t)len*8);
    for(unsigned t=0;t<len;t++){ float x=sc[t]; unsigned sb; memcpy(&sb,&x,4);
        sb=(sb&0x80000000u)?~sb:(sb|0x80000000u);
        key[t]=((unsigned long long)sb<<20)|(unsigned long long)((len-1-t)&0xFFFFF); }
    /* rank by count-greater (unique keys) — but O(n^2) is too slow at 128k; use nth_element via
     * a copy+partial selection: find the k-th largest key by a simple quickselect. */
    unsigned long long* tmp=malloc((size_t)len*8); memcpy(tmp,key,(size_t)len*8);
    /* quickselect for the (len-k)-th smallest == k-th largest threshold */
    unsigned lo=0,hi=len-1,want=len-k; unsigned long long thr=0;
    while(lo<hi){ unsigned long long p=tmp[(lo+hi)>>1]; unsigned i=lo,j=hi;
        while(i<=j){ while(tmp[i]<p)i++; while(tmp[j]>p)j--;
            if(i<=j){ unsigned long long t2=tmp[i];tmp[i]=tmp[j];tmp[j]=t2; i++; if(j)j--; else break; } }
        if(want<=j) hi=j; else if(want>=i) lo=i; else break; }
    thr=tmp[want];
    for(unsigned t=0;t<len;t++) if(key[t]>=thr) selected[t]=1;
    free(key); free(tmp);
}
static void ref_select(const float* sc,unsigned ctx,char* selected){
    ref_select_k(sc,ctx,TOPK,selected,ctx);
}
/* absorbed-MLA over a given selected index list (mirrors the gather kernel reduction). */
static void ref_mla(const bf16* Qa,const bf16* Qr,const bf16* Ckv,const bf16* Kr,const bf16* Wuv,
                    const int* sel,int nsel,bf16* o){
    float* p=malloc((size_t)nsel*4);
    for(int h=0;h<NH;h++){ float mx=-1e30f;
        for(int s=0;s<nsel;s++){ int t=sel[s]; float d=0;
            for(int l=0;l<DK;l++) d+=b2f(Qa[h*DK+l])*b2f(Ckv[(size_t)t*DK+l]);
            for(int r=0;r<DR;r++) d+=b2f(Qr[h*DR+r])*b2f(Kr[(size_t)t*DR+r]);
            p[s]=d*SCALE_MLA; if(p[s]>mx)mx=p[s]; }
        float sum=0; for(int s=0;s<nsel;s++){ p[s]=expf(p[s]-mx); sum+=p[s]; }
        float inv=sum>0?1.0f/sum:0; bf16 oacc[DK];
        for(int l=0;l<DK;l++){ float a=0; for(int s=0;s<nsel;s++) a+=p[s]*b2f(Ckv[(size_t)sel[s]*DK+l]); oacc[l]=f2b(a*inv); }
        for(int v=0;v<VD;v++){ float a=0; for(int l=0;l<DK;l++) a+=b2f(oacc[l])*b2f(Wuv[(size_t)(h*DK+l)*VD+v]); o[h*VD+v]=f2b(a); }
    }
    free(p);
}

static double relmax(const bf16* a,const bf16* b,int n){ double md=0,mw=0;
    for(int i=0;i<n;i++){ double d=fabs(b2f(a[i])-b2f(b[i])),w=fabs(b2f(b[i])); md=fmax(md,d); mw=fmax(mw,w);} return md/(mw+1e-12); }

/* time R back-to-back launches of a prepared kernel+args, one drain, return mean us. */
#define TIME(R, LAUNCH) ({ for(int _w=0;_w<3;_w++){ LAUNCH; } plow_hsa_wait(H,0); \
    double _t0=now(); for(int _r=0;_r<(R);_r++){ LAUNCH; } plow_hsa_wait(H,0); \
    (now()-_t0)/(R)*1e6; })

static const unsigned CTXS[]={8192,32768,131072,262144};
#define NCTX ((int)(sizeof(CTXS)/sizeof(CTXS[0])))
static plow_hsa_kernel kSc,kScF,kScM,kSelC,kSelA,kGat,kDen,kMrg,kFld;

int main(int argc,char** argv){
    const char* elf=argc>1?argv[1]:"test_kernels.elf";
    H=plow_hsa_init(); if(!H){ printf("hsa init failed\n"); return 1; }
    char nm[64]; uint32_t cus=0,lds=0; plow_hsa_device_info(H,0,nm,&cus,&lds);
    const unsigned NCU=cus; printf("dev0: %s CUs=%u\n\n",nm,cus);
    FILE* f=fopen(elf,"rb"); if(!f){ printf("%s missing\n",elf); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); void* co=malloc(n);
    if(fread(co,1,n,f)!=(size_t)n) return 1; fclose(f);
    if(plow_hsa_load_code_object(H,0,co,n)){ printf("load failed\n"); return 1; }
    if(plow_hsa_get_kernel(H,0,"index_score_128",&kSc)||plow_hsa_get_kernel(H,0,"index_score_fast_128",&kScF)||
       plow_hsa_get_kernel(H,0,"index_score_mfma_128",&kScM)||
       plow_hsa_get_kernel(H,0,"index_select_coop",&kSelC)||
       plow_hsa_get_kernel(H,0,"index_select_coop_a",&kSelA)||
       plow_hsa_get_kernel(H,0,"mla_gather_decode_512",&kGat)||
       plow_hsa_get_kernel(H,0,"mla_flash_decode_512",&kDen)||plow_hsa_get_kernel(H,0,"gemma_flash_merge_512",&kMrg)||
       plow_hsa_get_kernel(H,0,"mla_o_uv_fold_512",&kFld)){ printf("kernel missing: %s\n",plow_hsa_last_error()); return 1; }

    printf("%-7s | %10s %10s | %10s %10s %10s %8s %8s | %9s | %-22s\n",
           "ctx","dense us","gather us","idx-scalar","idx-fast","idx-mfma","sel us","sel-a us","dense/gat","gates (score/sel/gather)");
    printf("--------|-----------------------|-------------------------------------------|-----------|--------------------------\n");

    for(unsigned ci=0;ci<NCTX;ci++){
        const unsigned ctx=CTXS[ci];
        /* ---- host data ---- */
        bf16* hQi=malloc((size_t)HI*DI*2); bf16* hKi=malloc((size_t)ctx*DI*2); bf16* hW=malloc((size_t)HI*2);
        bf16* hQa=malloc((size_t)NH*DK*2); bf16* hQr=malloc((size_t)NH*DR*2);
        bf16* hCkv=malloc((size_t)ctx*DK*2); bf16* hKr=malloc((size_t)ctx*DR*2);
        bf16* hWuv=malloc((size_t)NH*DK*VD*2);
        seed(0x5E1EC7u+ctx);
        for(size_t i=0;i<(size_t)HI*DI;i++) hQi[i]=f2b(frand());
        for(size_t i=0;i<(size_t)ctx*DI;i++) hKi[i]=f2b(frand());
        for(int i=0;i<HI;i++) hW[i]=f2b(frand()*0.5f+0.5f);
        for(int i=0;i<NH*DK;i++) hQa[i]=f2b(frand()*0.2f);
        for(int i=0;i<NH*DR;i++) hQr[i]=f2b(frand()*0.2f);
        for(size_t i=0;i<(size_t)ctx*DK;i++) hCkv[i]=f2b(frand()*0.4f);
        for(size_t i=0;i<(size_t)ctx*DR;i++) hKr[i]=f2b(frand()*0.4f);
        for(size_t i=0;i<(size_t)NH*DK*VD;i++) hWuv[i]=f2b(frand()*0.1f);
        int32_t klen=(int)ctx;

        void* dQi=dev((size_t)HI*DI*2); up(dQi,hQi,(size_t)HI*DI*2);
        void* dKi=dev((size_t)ctx*DI*2); up(dKi,hKi,(size_t)ctx*DI*2);
        void* dW=dev((size_t)HI*2); up(dW,hW,(size_t)HI*2);
        void* dQa=dev((size_t)NH*DK*2); up(dQa,hQa,(size_t)NH*DK*2);
        void* dQr=dev((size_t)NH*DR*2); up(dQr,hQr,(size_t)NH*DR*2);
        void* dCkv=dev((size_t)ctx*DK*2); up(dCkv,hCkv,(size_t)ctx*DK*2);
        void* dKr=dev((size_t)ctx*DR*2); up(dKr,hKr,(size_t)ctx*DR*2);
        void* dWuv=dev((size_t)NH*DK*VD*2); up(dWuv,hWuv,(size_t)NH*DK*VD*2);
        void* dLen=dev(4); up(dLen,&klen,4);
        void* dSc=dev((size_t)ctx*4);
        void* dIdx=dev((size_t)TOPK*4);
        void* dCtl=dev(4*4);       /* coop select ctl: arrive, generation, emit-slot        */
        void* dHist=dev(4*8192*4); /* >= SEL_NPASS x SEL_NB radix histograms (kernel clears)  */
        { unsigned z[4]; memset(z,0,sizeof(z)); up(dCtl,z,4*4); }
        void* dOp=dev((size_t)NH*NSPLIT*DK*4);
        void* dMl=dev((size_t)NH*NSPLIT*2*4);
        void* dOlat=dev((size_t)NH*DK*2);
        void* dO=dev((size_t)NH*VD*2);

        /* ---- kernarg structs (mirror the wrapper signatures) ---- */
        struct __attribute__((packed)){ void *sc,*qi,*ki,*w,*len; unsigned nb,ih,ks; float scale; } aSc=
            {dSc,dQi,dKi,dW,dLen,1,HI,ctx,SCALE_IDX};
        struct __attribute__((packed)){ void *idx; const void *sc; unsigned len,tk; void *hist,*ctl;
                                        const void *klen; } aSel=
            {dIdx,dSc,ctx,TOPK,dHist,dCtl,dLen};
        struct __attribute__((packed)){ void *op,*ml; const void *qa,*qr,*ckv,*kr,*len,*idx;
            unsigned tk,nb,nh,ks; float scale; unsigned ns; } aGat=
            {dOp,dMl,dQa,dQr,dCkv,dKr,dLen,dIdx,TOPK,1,NH,ctx,SCALE_MLA,NSPLIT};
        struct __attribute__((packed)){ void *op,*ml; const void *qa,*qr,*ckv,*kr,*len;
            unsigned nb,nh,ks,win; float scale; unsigned ns; } aDen=
            {dOp,dMl,dQa,dQr,dCkv,dKr,dLen,1,NH,ctx,0,SCALE_MLA,NSPLIT};
        struct __attribute__((packed)){ void* o; const void *op,*ml; unsigned nb,nh,ns; } aMrg=
            {dOlat,dOp,dMl,1,NH,NSPLIT};
        struct __attribute__((packed)){ void* o; const void *olat,*wuv; unsigned nb,nh,v; } aFld=
            {dO,dOlat,dWuv,1,NH,VD};
        #define L(K,A) plow_hsa_launch(H,0,&(K),NCU*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&(A),sizeof(A))
        /* The cooperative selector is BARRIER/CONTENTION-bound, not bandwidth-bound (the score array
         * is only ctx*4 = 512 KiB @128k). Fewer co-resident WGs => less atomic contention on the
         * grid-barrier counter and the shared histogram bins. SELWG (env DSA_SELWG, default 64) is the
         * grid width for the selector only; it must stay <= NCU (co-residency). */
        unsigned SELWG=32; { const char* e=getenv("DSA_SELWG"); if(e){ unsigned v=(unsigned)atoi(e); if(v>=1&&v<=NCU) SELWG=v; } }
        #define LS(A) plow_hsa_launch(H,0,&kSelC,SELWG*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&(A),sizeof(A))
        #define LSA(A) plow_hsa_launch(H,0,&kSelA,SELWG*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&(A),sizeof(A))

        const int R=30;
        double us_sc =TIME(R, L(kSc ,aSc ));
        double us_scf=TIME(R, L(kScF,aSc ));
        double us_scm=TIME(R, L(kScM,aSc ));
        L(kSc,aSc); plow_hsa_wait(H,0); /* scores in HBM for the select timing */
        double us_sel=TIME(R, LS(aSel));
        L(kSc,aSc); plow_hsa_wait(H,0);
        double us_sela=TIME(R, LSA(aSel));
        double us_gat=TIME(R, L(kGat,aGat));
        double us_den=TIME(R, L(kDen,aDen));

        /* ---- correctness ---- */
        /* score: run scalar once, compare to CPU */
        L(kSc,aSc); plow_hsa_wait(H,0);
        float* hSc=malloc((size_t)ctx*4); plow_hsa_download(H,0,hSc,dSc,(size_t)ctx*4);
        float* hScF=malloc((size_t)ctx*4);
        L(kScF,aSc); plow_hsa_wait(H,0); plow_hsa_download(H,0,hScF,dSc,(size_t)ctx*4);
        float* hScM=malloc((size_t)ctx*4);
        L(kScM,aSc); plow_hsa_wait(H,0); plow_hsa_download(H,0,hScM,dSc,(size_t)ctx*4);
        float* refSc=malloc((size_t)ctx*4); ref_score(hQi,hKi,hW,ctx,refSc);
        double smax=0,sfmax=0,smmax=0,wmax=0;
        for(unsigned t=0;t<ctx;t++){ double w=fabs(refSc[t]); wmax=fmax(wmax,w);
            smax=fmax(smax,fabs(hSc[t]-refSc[t])); sfmax=fmax(sfmax,fabs(hScF[t]-refSc[t]));
            smmax=fmax(smmax,fabs(hScM[t]-refSc[t])); }
        double sc_rel=smax/(wmax+1e-12), scf_rel=sfmax/(wmax+1e-12), scm_rel=smmax/(wmax+1e-12);

        /* select: run once (scalar score already in dSc), compare set */
        L(kSc,aSc); plow_hsa_wait(H,0);
        LS(aSel); plow_hsa_wait(H,0);
        int* hIdx=malloc((size_t)TOPK*4); plow_hsa_download(H,0,hIdx,dIdx,(size_t)TOPK*4);
        char* selRef=malloc(ctx); ref_select(hSc,ctx,selRef);
        char* selDev=calloc(ctx,1); int idx_ok=1,cnt=0;
        for(int j=0;j<TOPK;j++){ int t=hIdx[j]; if(t<0||t>=(int)ctx){idx_ok=0;continue;} selDev[t]=1; cnt++; }
        int set_eq=idx_ok&&(cnt==TOPK); if(set_eq) for(unsigned t=0;t<ctx;t++) if(selDev[t]!=selRef[t]){set_eq=0;break;}

        /* variant select EXACT check (atomic-gen barrier): must match the CPU radix set exactly. */
        L(kSc,aSc); plow_hsa_wait(H,0);
        LSA(aSel); plow_hsa_wait(H,0);
        int* hIdxA=malloc((size_t)TOPK*4); plow_hsa_download(H,0,hIdxA,dIdx,(size_t)TOPK*4);
        char* selDevA=calloc(ctx,1); int idx_okA=1,cntA=0;
        for(int j=0;j<TOPK;j++){ int t=hIdxA[j]; if(t<0||t>=(int)ctx){idx_okA=0;continue;} selDevA[t]=1; cntA++; }
        int set_eqA=idx_okA&&(cntA==TOPK); if(set_eqA) for(unsigned t=0;t<ctx;t++) if(selDevA[t]!=selRef[t]){set_eqA=0;break;}
        free(hIdxA); free(selDevA);

        /* TIE-STRESS gate for the FAST (fewer-passes) variant: the timing data has unique scores so
         * it always fast-exits after 4 passes and NEVER exercises the index tie-break passes. Real
         * ReLU indexer scores produce large exact-tie groups (many zeros), so the boundary can land
         * inside a tie that MUST be split by lowest-index. Construct a score array with a handful of
         * unique highs and a huge equal-valued group straddling the top_k boundary, forcing red[2] >
         * k_rem and the index passes; the selected set must still be EXACT vs the CPU radix ref. */
        float* hScT=malloc((size_t)ctx*4);
        for(unsigned t=0;t<ctx;t++) hScT[t]=(t<500)?(10.0f+(float)(500-t)*0.01f):1.0f;
        up(dSc,hScT,(size_t)ctx*4);
        LSA(aSel); plow_hsa_wait(H,0);
        int* hIdxT=malloc((size_t)TOPK*4); plow_hsa_download(H,0,hIdxT,dIdx,(size_t)TOPK*4);
        char* selRefT=malloc(ctx); ref_select(hScT,ctx,selRefT);
        char* selDevT=calloc(ctx,1); int idx_okT=1,cntT=0;
        for(int j=0;j<TOPK;j++){ int t=hIdxT[j]; if(t<0||t>=(int)ctx){idx_okT=0;continue;} selDevT[t]=1; cntT++; }
        int tie_eqA=idx_okT&&(cntT==TOPK); if(tie_eqA) for(unsigned t=0;t<ctx;t++) if(selDevT[t]!=selRefT[t]){tie_eqA=0;break;}
        free(hScT); free(hIdxT); free(selRefT); free(selDevT);
        /* ---- SHORT-CONTEXT GATE: kv_len < max_ctx, the ONLY state a real decode step is ever in.
         *
         * Everything above runs kv_len == ctx, which is why this harness passed for the entire life
         * of the bug it is now pinned against. The emitter bakes `i[0] = max_ctx` into INDEX_SELECT
         * but INDEX_SCORE writes `Score[pos]` only for `pos < kv_len`, so the selector used to rank
         * `max_ctx - kv_len` words the score kernel never touched. DSA arms only above a 64k
         * crossover, so in production that gap is essentially the whole array.
         *
         * Reproduced faithfully rather than simulated: POISON the full score buffer, then let the
         * real score kernel overwrite only [0, LIVE) by uploading kv_len = LIVE. The tail that
         * survives is exactly what production leaves there. The poison is +1e30 so a selector that
         * still scans max_ctx MUST pick it and the failure is unambiguous, not probabilistic.
         *
         * Two regimes, both required:
         *   LIVE_A > TOPK  — the ordinary long-prompt case; every index must land inside [0,LIVE).
         *   LIVE_B < TOPK  — a short prompt, where fewer than top_k rows EXIST. The selector must
         *                    emit exactly LIVE_B of them and the gather must walk exactly that many
         *                    slots (`tk_live`), because slots [LIVE_B, TOPK) of idx[] were never
         *                    written and the GATHER path applies no mask. */
        {
            const unsigned LIVES[2]={ctx/4>TOPK?ctx/4:TOPK+1, 777};
            for(int li=0;li<2;li++){
                const unsigned LIVE=LIVES[li];
                if(LIVE>=ctx) continue;
                const unsigned KEXP=TOPK<LIVE?TOPK:LIVE;   /* min(top_k, kv_len) */
                float* hP=malloc((size_t)ctx*4);
                for(unsigned t=0;t<ctx;t++) hP[t]=1e30f;    /* poison the whole buffer */
                up(dSc,hP,(size_t)ctx*4);
                int32_t kl=(int)LIVE; up(dLen,&kl,4);       /* score writes only [0,LIVE) */
                L(kScM,aSc); plow_hsa_wait(H,0);
                float* hS=malloc((size_t)ctx*4); plow_hsa_download(H,0,hS,dSc,(size_t)ctx*4);
                LSA(aSel); plow_hsa_wait(H,0);
                int* hI=malloc((size_t)TOPK*4); plow_hsa_download(H,0,hI,dIdx,(size_t)TOPK*4);
                char* rS=malloc(ctx); ref_select_k(hS,LIVE,KEXP,rS,ctx);
                char* dS=calloc(ctx,1); int ok=1,cnt=0,oob=0;
                for(unsigned j=0;j<KEXP;j++){ int t=hI[j];
                    if(t<0||(unsigned)t>=LIVE){oob++;ok=0;continue;} dS[t]=1; cnt++; }
                if(ok&&cnt!=(int)KEXP) ok=0;
                if(ok) for(unsigned t=0;t<ctx;t++) if(dS[t]!=rS[t]){ok=0;break;}
                /* the gather must agree with a CPU MLA over exactly the KEXP live slots */
                double grel=-1.0;
                if(ok){ L(kGat,aGat); plow_hsa_wait(H,0); L(kMrg,aMrg); plow_hsa_wait(H,0);
                    L(kFld,aFld); plow_hsa_wait(H,0);
                    bf16* hOs=malloc((size_t)NH*VD*2); plow_hsa_download(H,0,hOs,dO,(size_t)NH*VD*2);
                    bf16* rOs=malloc((size_t)NH*VD*2); ref_mla(hQa,hQr,hCkv,hKr,hWuv,hI,(int)KEXP,rOs);
                    grel=relmax(hOs,rOs,NH*VD); free(hOs); free(rOs); }
                printf("        kv_len=%-7u (max_ctx=%u, expect %u idx) sel %s%s gather %s\n",
                       LIVE,ctx,KEXP, ok?"EXACT":"MISMATCH",
                       oob?" [INDEX OUT OF RANGE — selector scanned past kv_len]":"",
                       grel<0?"-":(grel<3e-2?"PASS":"FAIL"));
                free(hP);free(hS);free(hI);free(rS);free(dS);
            }
            int32_t kb=(int)ctx; up(dLen,&kb,4); /* restore full length for anything after */
        }

        /* restore real scores in dSc AND regenerate dIdx (the tie test clobbered both) so the gather
         * checks below run over the same real top-k set that hIdx holds. */
        L(kSc,aSc); plow_hsa_wait(H,0); LS(aSel); plow_hsa_wait(H,0);

        /* gather output vs CPU-MLA over the SAME device-selected set (kernel correctness) */
        L(kGat,aGat); plow_hsa_wait(H,0); L(kMrg,aMrg); plow_hsa_wait(H,0); L(kFld,aFld); plow_hsa_wait(H,0);
        bf16* hOg=malloc((size_t)NH*VD*2); plow_hsa_download(H,0,hOg,dO,(size_t)NH*VD*2);
        bf16* refOg=malloc((size_t)NH*VD*2); ref_mla(hQa,hQr,hCkv,hKr,hWuv,hIdx,TOPK,refOg);
        double gat_rel=relmax(hOg,refOg,NH*VD);

        /* dense output, and dense-vs-gather sparsification error */
        L(kDen,aDen); plow_hsa_wait(H,0); L(kMrg,aMrg); plow_hsa_wait(H,0); L(kFld,aFld); plow_hsa_wait(H,0);
        bf16* hOd=malloc((size_t)NH*VD*2); plow_hsa_download(H,0,hOd,dO,(size_t)NH*VD*2);
        double sparse_rel=relmax(hOg,hOd,NH*VD);

        printf("%-7u | %10.1f %10.1f | %10.1f %10.1f %10.1f %8.1f %8.1f | %8.2fx | score s%.4f/f%.4f/m%.4f sel %s/%s gat %s (sparse rel %.3f)\n",
               ctx, us_den, us_gat, us_sc, us_scf, us_scm, us_sel, us_sela, us_den/us_gat,
               sc_rel, scf_rel, scm_rel, set_eq?"EXACT":"MISMATCH",
               (set_eqA&&tie_eqA)?"EXACT":"MISMATCH",
               gat_rel<3e-2?"PASS":"FAIL", sparse_rel);
        fflush(stdout);

        free(hQi);free(hKi);free(hW);free(hQa);free(hQr);free(hCkv);free(hKr);free(hWuv);
        free(hSc);free(hScF);free(hScM);free(refSc);free(hIdx);free(selRef);free(selDev);free(hOg);free(refOg);free(hOd);
        plow_hsa_free(H,dQi);plow_hsa_free(H,dKi);plow_hsa_free(H,dW);plow_hsa_free(H,dQa);plow_hsa_free(H,dQr);
        plow_hsa_free(H,dCkv);plow_hsa_free(H,dKr);plow_hsa_free(H,dWuv);plow_hsa_free(H,dLen);plow_hsa_free(H,dSc);
        plow_hsa_free(H,dIdx);plow_hsa_free(H,dCtl);plow_hsa_free(H,dHist);
        plow_hsa_free(H,dOp);plow_hsa_free(H,dMl);plow_hsa_free(H,dOlat);plow_hsa_free(H,dO);
    }
    printf("\nNote: dense/gather timings are data-independent; sparse rel is a synthetic-data read\n"
           "(top-k of random scores), NOT a model-accuracy claim — that needs real weights (G6).\n");
    plow_hsa_shutdown(H);
    return 0;
}
