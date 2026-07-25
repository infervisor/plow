/* glm_block_gfx950_test.c — a FULL GLM-style decoder block on the real interpreter.
 *
 * Assembles plow's EXISTING dense GQA flash-decode attention with the shared MoE FFN core
 * into one decode block, stacked N times, driven by ONE persistent-interpreter launch over a
 * real ~4k KV context. Per block (decode, bs=1, one new token):
 *
 *   RMSNorm(x)                                    -> xn
 *   GEMV W_q · xn -> Q[nh][hd]
 *   GEMV W_k · xn -> K_new[nkv][hd] ; HEADNORM_ROPE scatters it into the head-major K cache[qpos]
 *   GEMV W_v · xn -> V_new[nkv][hd] ; HEADNORM_ROPE scatters it into the head-major V cache[qpos]
 *   FLASH_DECODE (GQA, split-KV) -> (O,m,l) partials
 *   FLASH_MERGE  -> O[nh][hd]
 *   GEMV W_o · O -> attn_out
 *   RESIDUAL x + attn_out -> x_mid
 *   RMSNorm(x_mid) -> xn2
 *   MoeRouter -> K×(MoeExpertGlu -> MoeExpertDown) -> MoeCombine(x_mid + Σ part) -> x_next
 *
 * The MoE FFN is the same data-dependent counter-gate core the DeepSeek block uses; only the
 * attention differs (dense GQA flash vs MLA latent flash) — this is the GLM instance of the
 * config-driven AttnKind. RoPE's rotary twist is out of scope (synthetic weights); HEADNORM_ROPE
 * runs with skip_norm and no cos/sin, i.e. it is used purely as the head-major cache scatter, and
 * the flash still streams the full ~4k KV loop per query head.
 *
 * VALIDATION (truth = independent CPU reference over the SAME op reduction boundaries):
 *   - MoE dispatch BIT-EXACT: the oracle MoE is fed the DEVICE's normed input (xn2) + residual
 *     (x_mid) and must reproduce x_next byte-for-byte (router lowest-id tie-break, weight-base
 *     indirection, fixed-order combine).
 *   - Flash attention to the established output-scaled tolerance (~1e-3): online-softmax
 *     reassociation vs the reference's single-pass softmax (same metric attention_gfx950_test.c
 *     uses). The oracle is fed the DEVICE's RMSNorm output (xn) so the non-reproducible block
 *     reduction (block_sum + rsqrtf) doesn't pollute the compare.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- dense-GQA config ---- */
#define H        1024
#define HD       128     /* head_dim (flash template 128) */
#define NH       8       /* query heads */
#define NKV      2       /* kv heads (GQA=4) */
#define CTX      4096
#define NSPLIT   8
#define SCALE    0.08838835f  /* 1/sqrt(128) */
/* MoE (scaled GLM RouterCfg) */
#define I_MOE    32
#define N_EXP    8
#define K        2
#define ACT      1u
#define ROUTE_SCALE 2.5f
#define FLAGS    (1u | 2u)
#define N_BLK    3
#define EPS      1e-6f

typedef uint16_t bf16;
static float b2f(bf16 v){ union{uint32_t u;float f;}c; c.u=(uint32_t)v<<16; return c.f; }
static bf16 f2b(float f){ union{float f;uint32_t u;}c; c.f=f; uint32_t r=c.u+0x7fff+((c.u>>16)&1); return (bf16)(r>>16); }
static uint64_t rs;
static void seed(uint64_t s){ rs=s*6364136223846793005ULL+1442695040888963407ULL; }
static float frand(void){ rs=rs*6364136223846793005ULL+1442695040888963407ULL; return (float)((int32_t)(rs>>33)%2001-1000)/4000.0f; }

static float mdot(const bf16* x, const bf16* w, int Kk){ float a=0; for(int i=0;i<Kk;i++) a+=b2f(x[i])*b2f(w[i]); return b2f(f2b(a)); }
static float silu(float x){ return x/(1.0f+expf(-x)); }

/* dense GQA flash reference (single-pass softmax; O rounded to bf16 like the merge writes). */
static void ref_attn(const bf16* Q, const bf16* Kc, const bf16* Vc, bf16* O) {
    const int gqa = NH/NKV;
    for (int h=0;h<NH;h++){
        const int kh = h/gqa;
        float sc[CTX], mx=-1e30f;
        for (int t=0;t<CTX;t++){ float d=0; for(int i=0;i<HD;i++) d+=b2f(Q[h*HD+i])*b2f(Kc[((size_t)kh*CTX+t)*HD+i]);
            sc[t]=d*SCALE; if(sc[t]>mx) mx=sc[t]; }
        float sum=0,p[CTX]; for(int t=0;t<CTX;t++){ p[t]=expf(sc[t]-mx); sum+=p[t]; }
        float inv=sum>0?1.0f/sum:0.0f;
        for (int v=0;v<HD;v++){ float a=0; for(int t=0;t<CTX;t++) a+=p[t]*b2f(Vc[((size_t)kh*CTX+t)*HD+v]);
            O[h*HD+v]=f2b(a*inv); }
    }
}

static void ref_router(const bf16* x, const bf16* Wr, unsigned* id, float* g) {
    float s[N_EXP]; for (int e=0;e<N_EXP;e++) s[e]=1.0f/(1.0f+expf(-mdot(x,Wr+(size_t)e*H,H)));
    float live[N_EXP]; memcpy(live,s,sizeof(s));
    for (int j=0;j<K;j++){ unsigned long long best=0; int bi=0;
        for (int e=0;e<N_EXP;e++){ unsigned sb; float sc=live[e]; memcpy(&sb,&sc,4);
            sb=(sb&0x80000000u)?~sb:(sb|0x80000000u);
            unsigned long long key=((unsigned long long)sb<<20)|(unsigned long long)((N_EXP-1-e)&0xFFFFF);
            if(key>best){best=key;bi=e;} }
        id[j]=bi; g[j]=live[bi]; live[bi]=-1e30f; }
    float sum=0; for(int j=0;j<K;j++) sum+=g[j];
    for (int j=0;j<K;j++){ if(FLAGS&2u) g[j]/=sum; g[j]*=ROUTE_SCALE; }
}
static void ref_moe(const bf16* xn2, const bf16* xmid, const bf16* Wr,
                    const bf16* const gw[N_EXP], const bf16* const uw[N_EXP],
                    const bf16* const dw[N_EXP], bf16* xnext) {
    unsigned id[K]; float g[K]; ref_router(xn2,Wr,id,g);
    float part[K][H];
    for (int slot=0;slot<K;slot++){ unsigned e=id[slot]; bf16 fu[I_MOE];
        for (int n=0;n<I_MOE;n++){ float gg=mdot(xn2,gw[e]+(size_t)n*H,H); float uu=mdot(xn2,uw[e]+(size_t)n*H,H); fu[n]=f2b(silu(gg)*uu); }
        for (int hh=0;hh<H;hh++) part[slot][hh]=g[slot]*mdot(fu,dw[e]+(size_t)hh*I_MOE,I_MOE); }
    for (int hh=0;hh<H;hh++){ float a=b2f(xmid[hh]); for(int slot=0;slot<K;slot++) a+=part[slot][hh]; xnext[hh]=f2b(a); }
}

/* ---- program builder ---- */
static PlowDevInst g_inst[4096]; static PlowWait g_wait[8192]; static uint32_t g_succ[4096];
typedef struct { uint32_t wait_ofs, succ_ofs; uint16_t wait_len, succ_len; } PlowGate;
static PlowGate g_gate[4096]; /* per-inst gates, copied onto every StreamEnt (64B DevInst carries none) */
static int g_nops=0, g_nw=0;
static int emitop(uint16_t op, uint16_t blocks){ int i=g_nops++; g_inst[i].op=op; g_inst[i].blocks=blocks;
    g_gate[i].succ_ofs=i; g_gate[i].succ_len=1; g_succ[i]=i; return i; }
static void addwait(int op,int producer,uint32_t thr){ if(g_gate[op].wait_len==0) g_gate[op].wait_ofs=g_nw;
    g_wait[g_nw].id=producer; g_wait[g_nw].threshold=thr; g_nw++; g_gate[op].wait_len++; }
static void* g_tens[1024]; static int g_nt=0;
static int reg(void* p){ g_tens[g_nt]=p; return g_nt++; }
static float relerr(const bf16* got,const bf16* want,int n,double* worst){
    double mw=0,md=0,se=0,sw=0;
    for(int i=0;i<n;i++){ double d=fabs(b2f(got[i])-b2f(want[i])),w=fabs(b2f(want[i]));
        mw=fmax(mw,w); md=fmax(md,d); se+=d*d; sw+=(double)b2f(want[i])*b2f(want[i]); }
    *worst=md/(mw+1e-12); return (float)(sqrt(se/n)/(sqrt(sw/n)+1e-12)); }

int main(int argc, char** argv){
    const char* elf=argc>1?argv[1]:"interp_decode.elf";
    plow_hsa* h=plow_hsa_init(); if(!h){ printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus=0,lds=0; plow_hsa_device_info(h,0,gfx,&cus,&lds); const unsigned NCU=cus;
    printf("dev0: %s  CUs=%u\n", gfx, cus);
    FILE* f=fopen(elf,"rb"); if(!f){ printf("%s missing\n",elf); return 1; }
    fseek(f,0,SEEK_END); long co_n=ftell(f); fseek(f,0,SEEK_SET); void* co=malloc(co_n);
    if(fread(co,1,co_n,f)!=(size_t)co_n) return 1; fclose(f);
    if(plow_hsa_load_code_object(h,0,co,co_n)){ printf("load failed\n"); return 1; }
    plow_hsa_kernel kern; if(plow_hsa_get_kernel(h,0,"plow_interp_dec_gfx950",&kern)){ printf("no kernel\n"); return 1; }

    static bf16 h_g1[N_BLK][H],h_g2[N_BLK][H];
    static bf16 h_Wq[N_BLK][NH*HD*H],h_Wk[N_BLK][NKV*HD*H],h_Wv[N_BLK][NKV*HD*H],h_Wo[N_BLK][H*NH*HD];
    static bf16 h_Wr[N_BLK][N_EXP*H];
    static bf16 h_gate[N_BLK][N_EXP][I_MOE*H],h_up[N_BLK][N_EXP][I_MOE*H],h_down[N_BLK][N_EXP][H*I_MOE];
    static bf16 h_kc[N_BLK][(size_t)NKV*CTX*HD],h_vc[N_BLK][(size_t)NKV*CTX*HD];
    static bf16 h_x0[H];
    for(int b=0;b<N_BLK;b++){
        seed(0x6100^(uint64_t)b);
        for(int i=0;i<H;i++){ h_g1[b][i]=f2b(1.0f+frand()*0.1f); h_g2[b][i]=f2b(1.0f+frand()*0.1f); }
        for(size_t i=0;i<(size_t)NH*HD*H;i++)  h_Wq[b][i]=f2b(frand()*0.2f);
        for(size_t i=0;i<(size_t)NKV*HD*H;i++){ h_Wk[b][i]=f2b(frand()*0.2f); h_Wv[b][i]=f2b(frand()*0.2f); }
        for(size_t i=0;i<(size_t)H*NH*HD;i++)  h_Wo[b][i]=f2b(frand()*0.1f);
        for(int i=0;i<N_EXP*H;i++) h_Wr[b][i]=f2b(frand());
        for(int e=0;e<N_EXP;e++){ for(size_t i=0;i<I_MOE*H;i++){ h_gate[b][e][i]=f2b(frand()); h_up[b][e][i]=f2b(frand()); }
            for(size_t i=0;i<H*I_MOE;i++) h_down[b][e][i]=f2b(frand()); }
        for(size_t i=0;i<(size_t)NKV*CTX*HD;i++){ h_kc[b][i]=f2b(frand()*0.4f); h_vc[b][i]=f2b(frand()*0.4f); }
    }
    seed(999); for(int i=0;i<H;i++) h_x0[i]=f2b(frand());

#define UP(dp,src,bytes) do{ dp=plow_hsa_alloc(h,0,bytes); plow_hsa_upload(h,0,dp,src,bytes);}while(0)
    void *d_x[N_BLK+1],*d_xn[N_BLK],*d_attn[N_BLK],*d_xmid[N_BLK],*d_xn2[N_BLK];
    for(int b=0;b<=N_BLK;b++) d_x[b]=plow_hsa_alloc(h,0,H*2);
    plow_hsa_upload(h,0,d_x[0],h_x0,H*2);
    for(int b=0;b<N_BLK;b++){ d_xn[b]=plow_hsa_alloc(h,0,H*2); d_attn[b]=plow_hsa_alloc(h,0,H*2);
        d_xmid[b]=plow_hsa_alloc(h,0,H*2); d_xn2[b]=plow_hsa_alloc(h,0,H*2); }
    void *d_g1[N_BLK],*d_g2[N_BLK],*d_Wq[N_BLK],*d_Wk[N_BLK],*d_Wv[N_BLK],*d_Wo[N_BLK];
    void *d_Wr[N_BLK],*d_wtab[N_BLK],*d_kc[N_BLK],*d_vc[N_BLK];
    void *d_gate[N_BLK][N_EXP],*d_up[N_BLK][N_EXP],*d_down[N_BLK][N_EXP];
    for(int b=0;b<N_BLK;b++){
        UP(d_g1[b],h_g1[b],H*2); UP(d_g2[b],h_g2[b],H*2);
        UP(d_Wq[b],h_Wq[b],(size_t)NH*HD*H*2); UP(d_Wk[b],h_Wk[b],(size_t)NKV*HD*H*2);
        UP(d_Wv[b],h_Wv[b],(size_t)NKV*HD*H*2); UP(d_Wo[b],h_Wo[b],(size_t)H*NH*HD*2);
        UP(d_Wr[b],h_Wr[b],N_EXP*H*2);
        UP(d_kc[b],h_kc[b],(size_t)NKV*CTX*HD*2); UP(d_vc[b],h_vc[b],(size_t)NKV*CTX*HD*2);
        uint64_t wt[N_EXP*3];
        for(int e=0;e<N_EXP;e++){ UP(d_gate[b][e],h_gate[b][e],I_MOE*H*2); UP(d_up[b][e],h_up[b][e],I_MOE*H*2);
            UP(d_down[b][e],h_down[b][e],H*I_MOE*2);
            wt[e*3+0]=(uint64_t)(uintptr_t)d_gate[b][e]; wt[e*3+1]=(uint64_t)(uintptr_t)d_up[b][e]; wt[e*3+2]=(uint64_t)(uintptr_t)d_down[b][e]; }
        UP(d_wtab[b],wt,sizeof(wt));
    }
    void* d_Q=plow_hsa_alloc(h,0,(size_t)NH*HD*2);
    void* d_Knew=plow_hsa_alloc(h,0,(size_t)NKV*HD*2);
    void* d_Vnew=plow_hsa_alloc(h,0,(size_t)NKV*HD*2);
    void* d_Opart=plow_hsa_alloc(h,0,(size_t)NH*NSPLIT*HD*4);
    void* d_ml=plow_hsa_alloc(h,0,(size_t)NH*NSPLIT*2*4);
    void* d_O=plow_hsa_alloc(h,0,(size_t)NH*HD*2);
    void* d_tab=plow_hsa_alloc(h,0,K*8);
    void* d_fu=plow_hsa_alloc(h,0,(size_t)K*I_MOE*2);
    void* d_part=plow_hsa_alloc(h,0,(size_t)K*H*4);
    int32_t klen=CTX; void* d_klen; UP(d_klen,&klen,4);

    int TQ=reg(d_Q),TKN=reg(d_Knew),TVN=reg(d_Vnew),TOP=reg(d_Opart),TML=reg(d_ml),TO=reg(d_O),
        TTAB=reg(d_tab),TFU=reg(d_fu),TPART=reg(d_part),TKLEN=reg(d_klen);
    int TX[N_BLK+1]; for(int b=0;b<=N_BLK;b++) TX[b]=reg(d_x[b]);
    int TXN[N_BLK],TATT[N_BLK],TXMID[N_BLK],TXN2[N_BLK];
    int TG1[N_BLK],TG2[N_BLK],TWQ[N_BLK],TWK[N_BLK],TWV[N_BLK],TWO[N_BLK],TWR[N_BLK],TWTAB[N_BLK],TKC[N_BLK],TVC[N_BLK];
    const size_t qpos=CTX-1;
    for(int b=0;b<N_BLK;b++){
        TXN[b]=reg(d_xn[b]); TATT[b]=reg(d_attn[b]); TXMID[b]=reg(d_xmid[b]); TXN2[b]=reg(d_xn2[b]);
        TG1[b]=reg(d_g1[b]); TG2[b]=reg(d_g2[b]); TWQ[b]=reg(d_Wq[b]); TWK[b]=reg(d_Wk[b]); TWV[b]=reg(d_Wv[b]);
        TWO[b]=reg(d_Wo[b]); TWR[b]=reg(d_Wr[b]); TWTAB[b]=reg(d_wtab[b]); TKC[b]=reg(d_kc[b]); TVC[b]=reg(d_vc[b]);
    }

#define GEMV(o_h,x_h,w_h,N_,K_) do{ int _i=emitop(PLOW_DOP_GEMV,NCU); PlowDevInst*_d=&g_inst[_i]; \
        _d->t[0]=o_h;_d->t[1]=x_h;_d->t[2]=w_h;_d->t[3]=PLOW_TENSOR_NONE;_d->t[4]=PLOW_TENSOR_NONE; \
        _d->i[0]=1;_d->i[1]=(N_);_d->i[2]=(K_);_d->i[3]=0;_d->i[4]=0;_d->fj[0].f=1.0f; _last=_i; }while(0)
    int prev_combine=-1,_last=-1;
    for(int b=0;b<N_BLK;b++){
        int i_rn1=emitop(PLOW_DOP_RMSNORM,1);
        { PlowDevInst* d=&g_inst[i_rn1]; d->t[0]=TXN[b]; d->t[1]=TX[b]; d->t[2]=TG1[b]; d->i[0]=1; d->i[1]=H; d->fj[0].f=EPS; }
        if(prev_combine>=0) addwait(i_rn1,prev_combine,NCU);

        GEMV(TQ,  TXN[b], TWQ[b], NH*HD,  H); int i_q=_last;  addwait(i_q,i_rn1,1);
        GEMV(TKN, TXN[b], TWK[b], NKV*HD, H); int i_k=_last;  addwait(i_k,i_rn1,1);
        GEMV(TVN, TXN[b], TWV[b], NKV*HD, H); int i_v=_last;  addwait(i_v,i_rn1,1);

        /* HEADNORM_ROPE as the head-major cache scatter: skip_norm=1, no cos/sin. */
        int i_hk=emitop(PLOW_DOP_HEADNORM_ROPE,NCU);
        { PlowDevInst* d=&g_inst[i_hk]; d->t[0]=TKC[b]; d->t[1]=TKN; d->t[2]=PLOW_TENSOR_NONE;
          d->t[3]=PLOW_TENSOR_NONE; d->t[4]=PLOW_TENSOR_NONE; d->t[5]=PLOW_TENSOR_NONE;
          d->i[0]=1; d->i[1]=NKV; d->i[2]=HD; d->i[3]=(uint32_t)qpos; d->i[4]=1; /*skip_norm*/
          d->fj[1].u=CTX; d->fj[2].u=PLOW_KV_MASK_NONE; d->fj[0].f=EPS; }
        addwait(i_hk,i_k,NCU);
        int i_hv=emitop(PLOW_DOP_HEADNORM_ROPE,NCU);
        { PlowDevInst* d=&g_inst[i_hv]; d->t[0]=TVC[b]; d->t[1]=TVN; d->t[2]=PLOW_TENSOR_NONE;
          d->t[3]=PLOW_TENSOR_NONE; d->t[4]=PLOW_TENSOR_NONE; d->t[5]=PLOW_TENSOR_NONE;
          d->i[0]=1; d->i[1]=NKV; d->i[2]=HD; d->i[3]=(uint32_t)qpos; d->i[4]=1;
          d->fj[1].u=CTX; d->fj[2].u=PLOW_KV_MASK_NONE; d->fj[0].f=EPS; }
        addwait(i_hv,i_v,NCU);

        int i_fl=emitop(PLOW_DOP_FLASH_DECODE,NCU);
        { PlowDevInst* d=&g_inst[i_fl]; d->t[0]=TOP; d->t[1]=TML; d->t[2]=TQ; d->t[3]=TKC[b]; d->t[4]=TVC[b]; d->t[5]=TKLEN;
          d->i[0]=1; d->i[1]=NH; d->i[2]=NKV; d->i[3]=CTX; d->i[4]=0; d->i[5]=NSPLIT; d->i[6]=HD; d->i[7]=PLOW_KV_MASK_NONE; d->fj[0].f=SCALE; }
        addwait(i_fl,i_q,NCU); addwait(i_fl,i_hk,NCU); addwait(i_fl,i_hv,NCU);

        int i_mg=emitop(PLOW_DOP_FLASH_MERGE,NCU);
        { PlowDevInst* d=&g_inst[i_mg]; d->t[0]=TO; d->t[1]=TOP; d->t[2]=TML; d->i[0]=1; d->i[1]=NH; d->i[2]=NSPLIT; d->i[3]=HD; }
        addwait(i_mg,i_fl,NCU);

        GEMV(TATT[b], TO, TWO[b], H, NH*HD); int i_op=_last; addwait(i_op,i_mg,NCU);

        int i_rs=emitop(PLOW_DOP_RESIDUAL,1);
        { PlowDevInst* d=&g_inst[i_rs]; d->t[0]=TXMID[b]; d->t[1]=TX[b]; d->t[2]=TATT[b]; d->i[0]=H; d->fj[0].f=1.0f; }
        addwait(i_rs,i_op,NCU);
        int i_rn2=emitop(PLOW_DOP_RMSNORM,1);
        { PlowDevInst* d=&g_inst[i_rn2]; d->t[0]=TXN2[b]; d->t[1]=TXMID[b]; d->t[2]=TG2[b]; d->i[0]=1; d->i[1]=H; d->fj[0].f=EPS; }
        addwait(i_rn2,i_rs,1);

        int i_router=emitop(PLOW_DOP_MOE_ROUTER,1);
        { PlowDevInst* d=&g_inst[i_router]; d->t[0]=TTAB; d->t[1]=TXN2[b]; d->t[2]=TWR[b];
          d->i[0]=H; d->i[1]=N_EXP; d->i[2]=K; d->i[3]=FLAGS; d->fj[0].f=ROUTE_SCALE; }
        addwait(i_router,i_rn2,1);
        int downs[K];
        for(int slot=0;slot<K;slot++){
            int i_g=emitop(PLOW_DOP_MOE_EXPERT_GLU,NCU);
            { PlowDevInst* d=&g_inst[i_g]; d->t[0]=TFU; d->t[1]=TXN2[b]; d->t[2]=TTAB; d->t[3]=TWTAB[b];
              d->i[0]=slot; d->i[1]=I_MOE; d->i[2]=H; d->i[3]=N_EXP; d->i[5]=ACT; }
            addwait(i_g,i_router,1);
            int i_d=emitop(PLOW_DOP_MOE_EXPERT_DOWN,NCU);
            { PlowDevInst* d=&g_inst[i_d]; d->t[0]=TPART; d->t[1]=TFU; d->t[2]=TTAB; d->t[3]=TWTAB[b];
              d->i[0]=slot; d->i[1]=H; d->i[2]=I_MOE; d->i[3]=N_EXP; }
            addwait(i_d,i_g,NCU); downs[slot]=i_d;
        }
        int i_cmb=emitop(PLOW_DOP_MOE_COMBINE,NCU);
        { PlowDevInst* d=&g_inst[i_cmb]; d->t[0]=TX[b+1]; d->t[1]=TXMID[b]; d->t[2]=PLOW_TENSOR_NONE; d->t[3]=TPART; d->i[0]=H; d->i[1]=K; }
        for(int slot=0;slot<K;slot++) addwait(i_cmb,downs[slot],NCU);
        prev_combine=i_cmb;
    }
    const int n_ops=g_nops;

    void* d_tens=plow_hsa_alloc(h,0,(size_t)g_nt*sizeof(void*));
    plow_hsa_upload(h,0,d_tens,g_tens,(size_t)g_nt*sizeof(void*));
    uint32_t* sofs=malloc(4u*NCU); uint32_t* slen=malloc(4u*NCU);
    size_t total=0; for(unsigned cu=0;cu<NCU;cu++) for(int op=0;op<n_ops;op++) if(!(g_inst[op].blocks==1&&cu!=0)) total++;
    PlowStreamEnt* stream=calloc(total,sizeof(PlowStreamEnt)); size_t si=0;
    for(unsigned cu=0;cu<NCU;cu++){ sofs[cu]=(uint32_t)si;
        for(int op=0;op<n_ops;op++){ if(g_inst[op].blocks==1&&cu!=0) continue;
            stream[si].inst=(uint32_t)op; stream[si].slice=(g_inst[op].blocks==1)?0u:cu;
            stream[si].wait_ofs=g_gate[op].wait_ofs; stream[si].wait_len=g_gate[op].wait_len;
            stream[si].succ_ofs=g_gate[op].succ_ofs; stream[si].succ_len=g_gate[op].succ_len; si++; }
        slen[cu]=(uint32_t)si-sofs[cu]; }
    void* d_inst=plow_hsa_alloc(h,0,(size_t)n_ops*sizeof(PlowDevInst));
    void* d_stream=plow_hsa_alloc(h,0,total*sizeof(PlowStreamEnt));
    void* d_sofs=plow_hsa_alloc(h,0,4u*NCU); void* d_slen=plow_hsa_alloc(h,0,4u*NCU);
    void* d_waits=plow_hsa_alloc(h,0,(size_t)(g_nw?g_nw:1)*sizeof(PlowWait));
    void* d_succs=plow_hsa_alloc(h,0,(size_t)n_ops*4);
    void* d_ctr=plow_hsa_alloc(h,0,(size_t)n_ops*PLOW_CTR_STRIDE*4);
    plow_hsa_upload(h,0,d_inst,g_inst,(size_t)n_ops*sizeof(PlowDevInst));
    plow_hsa_upload(h,0,d_stream,stream,total*sizeof(PlowStreamEnt));
    plow_hsa_upload(h,0,d_sofs,sofs,4u*NCU); plow_hsa_upload(h,0,d_slen,slen,4u*NCU);
    if(g_nw) plow_hsa_upload(h,0,d_waits,g_wait,(size_t)g_nw*sizeof(PlowWait));
    plow_hsa_upload(h,0,d_succs,g_succ,(size_t)n_ops*4);
    uint32_t* zc=calloc((size_t)n_ops*PLOW_CTR_STRIDE,4); plow_hsa_upload(h,0,d_ctr,zc,(size_t)n_ops*PLOW_CTR_STRIDE*4);
    PlowProgram prog; memset(&prog,0,sizeof(prog));
    prog.insts=d_inst; prog.stream=d_stream; prog.stream_ofs=d_sofs; prog.stream_len=d_slen;
    prog.waits=d_waits; prog.succs=d_succs; prog.counters=d_ctr; prog.tensors=(void* const*)d_tens;

    printf("program: %d ops, %zu workgroup-packets, ctx=%d nsplit=%d GQA=%d/%d hd=%d\n\n",
           n_ops,total,CTX,NSPLIT,NH,NKV,HD);
    if(plow_hsa_launch(h,0,&kern,NCU*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&prog,sizeof(prog))){ printf("LAUNCH FAILED\n"); return 1; }
    plow_hsa_wait(h,0);

    int ok=1;
    printf("  %-8s %-30s %-30s\n","block","dense attn (rel rms / max)","MoE dispatch (bit-exact)");
    for(int b=0;b<N_BLK;b++){
        bf16 xn[H],att[H],xmid[H],xn2[H];
        plow_hsa_download(h,0,xn,d_xn[b],H*2); plow_hsa_download(h,0,att,d_attn[b],H*2);
        plow_hsa_download(h,0,xmid,d_xmid[b],H*2); plow_hsa_download(h,0,xn2,d_xn2[b],H*2);

        /* attention leg: oracle fed device xn -> attn_out, tolerance */
        static bf16 Q[NH*HD],Kn[NKV*HD],Vn[NKV*HD];
        for(int n=0;n<NH*HD;n++) Q[n]=f2b(mdot(xn,h_Wq[b]+(size_t)n*H,H));
        for(int n=0;n<NKV*HD;n++){ Kn[n]=f2b(mdot(xn,h_Wk[b]+(size_t)n*H,H)); Vn[n]=f2b(mdot(xn,h_Wv[b]+(size_t)n*H,H)); }
        static bf16 kc[(size_t)NKV*CTX*HD],vc[(size_t)NKV*CTX*HD];
        memcpy(kc,h_kc[b],sizeof(kc)); memcpy(vc,h_vc[b],sizeof(vc));
        for(int kh=0;kh<NKV;kh++) for(int i=0;i<HD;i++){ kc[((size_t)kh*CTX+qpos)*HD+i]=Kn[kh*HD+i]; vc[((size_t)kh*CTX+qpos)*HD+i]=Vn[kh*HD+i]; }
        bf16 O[NH*HD], att_ref[H];
        ref_attn(Q,kc,vc,O);
        for(int hh=0;hh<H;hh++) att_ref[hh]=f2b(mdot(O,h_Wo[b]+(size_t)hh*NH*HD,NH*HD));
        double amax; float arms=relerr(att,att_ref,H,&amax);
        int a_ok=(amax<3e-2 && arms<1e-2);

        const bf16 *gw[N_EXP],*uw[N_EXP],*dw[N_EXP];
        for(int e=0;e<N_EXP;e++){ gw[e]=h_gate[b][e]; uw[e]=h_up[b][e]; dw[e]=h_down[b][e]; }
        bf16 xnext[H]; ref_moe(xn2,xmid,h_Wr[b],gw,uw,dw,xnext);
        bf16 dev_next[H]; plow_hsa_download(h,0,dev_next,d_x[b+1],H*2);
        int moe_exact=1; for(int i=0;i<H;i++) if(dev_next[i]!=xnext[i]) moe_exact=0;
        ok &= (a_ok && moe_exact);
        char c1[64]; snprintf(c1,sizeof(c1),"%s rms=%.5f max=%.4f",a_ok?"PASS":"FAIL",arms,amax);
        printf("  %-8d %-30s %-30s\n",b,c1,moe_exact?"PASS (byte-identical)":"*** FAIL ***");
        unsigned id[K]; float g[K]; ref_router(xn2,h_Wr[b],id,g);
        printf("           routing: "); for(int j=0;j<K;j++) printf("e%u(g=%.4f) ",id[j],g[j]); printf("\n");
    }
    plow_hsa_download(h,0,zc,d_ctr,(size_t)n_ops*PLOW_CTR_STRIDE*4);
    int ctr_ok=1; for(int op=0;op<n_ops;op++) if(zc[(size_t)op*PLOW_CTR_STRIDE]!=g_inst[op].blocks) ctr_ok=0;
    printf("\n  executed==total (every packet fired): %s\n", ctr_ok?"YES":"NO");
    printf("\n%s\n",(ok&&ctr_ok)?"GLM BLOCK OK — dense attn within tolerance, MoE dispatch BIT-EXACT"
                                :"*** GLM BLOCK MISMATCH ***");
    plow_hsa_shutdown(h);
    return (ok&&ctr_ok)?0:1;
}
