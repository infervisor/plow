/* attn_select_gfx950_test.c — the DeepSeek DSA sparse path end-to-end (select -> gather). [DEEPSEEK-MLA]
 *
 * Drives, in ONE persistent-interpreter launch (decode), the full sparse-attention chain:
 *   ATTN_SELECT (op 53)         Qidx·Kidx over ctx -> top_k idx table (lowest-index tie-break)
 *   FLASH_GATHER_DECODE (op 54) MLA latent flash over ONLY the selected latent rows
 *   FLASH_MERGE<512>            latent-wide LSE merge of the split-KV partials
 *   O_UV_FOLD (op 52)           fold the merged latent to v_head_dim
 *
 * Truth (independent CPU reference over the same op reduction boundaries):
 *   1. the SELECTED SET: the device idx table must equal the reference top-k (score + packed-key
 *      lowest-index tie-break) — a set compare, exact.
 *   2. the GATHERED OUTPUT: absorbed-MLA over the DEVICE-selected set (fed the device idx) must
 *      match o[nh][V] within the established MLA tolerance (online-softmax reassociation).
 * The two checks isolate the selector (exact) from the gather+merge+fold (tolerance).
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NH      8
#define DK      512
#define DR      64
#define VD      128
#define IDXD    64        /* indexer dim (a small scorer head) */
#define CTX     4096
#define TOPK    512
#define NSPLIT  4
#define SCALE_MLA 0.08838835f
#define SCALE_SEL 0.125f  /* 1/sqrt(64) */

typedef uint16_t bf16;
static float b2f(bf16 v){ union{uint32_t u;float f;}c; c.u=(uint32_t)v<<16; return c.f; }
static bf16 f2b(float f){ union{float f;uint32_t u;}c; c.f=f; uint32_t r=c.u+0x7fff+((c.u>>16)&1); return (bf16)(r>>16); }
static uint64_t rs; static void seed(uint64_t s){ rs=s*6364136223846793005ULL+1442695040888963407ULL; }
static float frand(void){ rs=rs*6364136223846793005ULL+1442695040888963407ULL; return (float)((int32_t)(rs>>33)%2001-1000)/4000.0f; }

/* reference selector: mirrors d_attn_select's packed-key rank exactly. */
static void ref_select(const bf16* Qi, const bf16* Ki, int* sel /*[TOPK]*/) {
    static unsigned long long key[CTX];
    for (int t=0;t<CTX;t++){
        float d=0; for(int i=0;i<IDXD;i++) d+=b2f(Qi[i])*b2f(Ki[(size_t)t*IDXD+i]);
        float sc=d*SCALE_SEL; unsigned sb; memcpy(&sb,&sc,4); sb=(sb&0x80000000u)?~sb:(sb|0x80000000u);
        key[t]=((unsigned long long)sb<<20)|(unsigned long long)((CTX-1-t)&0xFFFFF);
    }
    for (int t=0;t<CTX;t++){ unsigned rank=0; for(int u=0;u<CTX;u++) rank+=(key[u]>key[t]); if(rank<TOPK) sel[rank]=t; }
}

/* absorbed-MLA over a given selected set (mirrors mla_ref.rs, gather case). */
static void ref_mla_gather(const bf16* Qa,const bf16* Qr,const bf16* Ckv,const bf16* Kr,
                           const bf16* Wuv,const int* sel,int nsel,bf16* o){
    for (int h=0;h<NH;h++){
        float sc[TOPK],mx=-1e30f;
        for (int s=0;s<nsel;s++){ int t=sel[s]; float d=0;
            for(int l=0;l<DK;l++) d+=b2f(Qa[h*DK+l])*b2f(Ckv[(size_t)t*DK+l]);
            for(int r=0;r<DR;r++) d+=b2f(Qr[h*DR+r])*b2f(Kr[(size_t)t*DR+r]);
            sc[s]=d*SCALE_MLA; if(sc[s]>mx) mx=sc[s]; }
        float sum=0,p[TOPK]; for(int s=0;s<nsel;s++){ p[s]=expf(sc[s]-mx); sum+=p[s]; }
        float inv=sum>0?1.0f/sum:0.0f; bf16 oacc[DK];
        for(int l=0;l<DK;l++){ float a=0; for(int s=0;s<nsel;s++) a+=p[s]*b2f(Ckv[(size_t)sel[s]*DK+l]); oacc[l]=f2b(a*inv); }
        for(int v=0;v<VD;v++){ float a=0; for(int l=0;l<DK;l++) a+=b2f(oacc[l])*b2f(Wuv[(size_t)(h*DK+l)*VD+v]); o[h*VD+v]=f2b(a); }
    }
}

static PlowDevInst g_inst[64]; static PlowWait g_wait[64]; static uint32_t g_succ[64]; static int g_nops=0,g_nw=0;
typedef struct { uint32_t wait_ofs, succ_ofs; uint16_t wait_len, succ_len; } PlowGate;
static PlowGate g_gate[64]; /* per-inst gates, copied onto every StreamEnt (64B DevInst carries none) */
static int emitop(uint16_t op,uint16_t blocks){ int i=g_nops++; g_inst[i].op=op; g_inst[i].blocks=blocks;
    g_gate[i].succ_ofs=i; g_gate[i].succ_len=1; g_succ[i]=i; return i; }
static void addwait(int op,int prod,uint32_t thr){ if(g_gate[op].wait_len==0) g_gate[op].wait_ofs=g_nw;
    g_wait[g_nw].id=prod; g_wait[g_nw].threshold=thr; g_nw++; g_gate[op].wait_len++; }
static void* g_tens[64]; static int g_nt=0; static int reg(void* p){ g_tens[g_nt]=p; return g_nt++; }

int main(int argc,char** argv){
    const char* elf=argc>1?argv[1]:"interp_decode.elf";
    plow_hsa* h=plow_hsa_init(); if(!h){ printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus=0,lds=0; plow_hsa_device_info(h,0,gfx,&cus,&lds); const unsigned NCU=cus;
    printf("dev0: %s  CUs=%u\n",gfx,cus);
    FILE* f=fopen(elf,"rb"); if(!f){ printf("%s missing\n",elf); return 1; }
    fseek(f,0,SEEK_END); long co_n=ftell(f); fseek(f,0,SEEK_SET); void* co=malloc(co_n);
    if(fread(co,1,co_n,f)!=(size_t)co_n) return 1; fclose(f);
    if(plow_hsa_load_code_object(h,0,co,co_n)){ printf("load failed\n"); return 1; }
    plow_hsa_kernel kern; if(plow_hsa_get_kernel(h,0,"plow_interp_dec_gfx950",&kern)){ printf("no kernel\n"); return 1; }

    static bf16 h_Qi[IDXD], h_Ki[(size_t)CTX*IDXD];
    static bf16 h_Qa[NH*DK], h_Qr[NH*DR], h_Ckv[(size_t)CTX*DK], h_Kr[(size_t)CTX*DR], h_Wuv[NH*DK*VD];
    seed(0x5E1EC7);
    for(int i=0;i<IDXD;i++) h_Qi[i]=f2b(frand());
    for(size_t i=0;i<(size_t)CTX*IDXD;i++) h_Ki[i]=f2b(frand());
    for(int i=0;i<NH*DK;i++) h_Qa[i]=f2b(frand()*0.2f);
    for(int i=0;i<NH*DR;i++) h_Qr[i]=f2b(frand()*0.2f);
    for(size_t i=0;i<(size_t)CTX*DK;i++) h_Ckv[i]=f2b(frand()*0.4f);
    for(size_t i=0;i<(size_t)CTX*DR;i++) h_Kr[i]=f2b(frand()*0.4f);
    for(size_t i=0;i<(size_t)NH*DK*VD;i++) h_Wuv[i]=f2b(frand()*0.1f);

#define UP(dp,src,bytes) do{ dp=plow_hsa_alloc(h,0,bytes); plow_hsa_upload(h,0,dp,src,bytes);}while(0)
    void *d_Qi,*d_Ki,*d_Qa,*d_Qr,*d_Ckv,*d_Kr,*d_Wuv;
    UP(d_Qi,h_Qi,sizeof(h_Qi)); UP(d_Ki,h_Ki,sizeof(h_Ki)); UP(d_Qa,h_Qa,sizeof(h_Qa));
    UP(d_Qr,h_Qr,sizeof(h_Qr)); UP(d_Ckv,h_Ckv,sizeof(h_Ckv)); UP(d_Kr,h_Kr,sizeof(h_Kr)); UP(d_Wuv,h_Wuv,sizeof(h_Wuv));
    void* d_idx=plow_hsa_alloc(h,0,TOPK*4);
    void* d_Op=plow_hsa_alloc(h,0,(size_t)NH*NSPLIT*DK*4);
    void* d_ml=plow_hsa_alloc(h,0,(size_t)NH*NSPLIT*2*4);
    void* d_Olat=plow_hsa_alloc(h,0,(size_t)NH*DK*2);
    void* d_o=plow_hsa_alloc(h,0,(size_t)NH*VD*2);
    int32_t klen=CTX; void* d_klen; UP(d_klen,&klen,4);

    int TIDX=reg(d_idx),TQI=reg(d_Qi),TKI=reg(d_Ki),TKLEN=reg(d_klen),TQA=reg(d_Qa),TQR=reg(d_Qr),
        TCKV=reg(d_Ckv),TKR=reg(d_Kr),TWUV=reg(d_Wuv),TOP=reg(d_Op),TML=reg(d_ml),TOLAT=reg(d_Olat),TO=reg(d_o);

    int i_sel=emitop(PLOW_DOP_ATTN_SELECT,1);
    { PlowDevInst* d=&g_inst[i_sel]; d->t[0]=TIDX; d->t[1]=TQI; d->t[2]=TKI; d->t[3]=TKLEN;
      d->i[0]=1; d->i[1]=IDXD; d->i[2]=CTX; d->i[3]=TOPK; d->fj[0].f=SCALE_SEL; }
    int i_ga=emitop(PLOW_DOP_FLASH_GATHER_DECODE,NCU);
    { PlowDevInst* d=&g_inst[i_ga]; d->t[0]=TOP; d->t[1]=TML; d->t[2]=TQA; d->t[3]=TQR; d->t[4]=TCKV; d->t[5]=TKR;
      d->t[6]=TKLEN; d->t[7]=TIDX; d->i[0]=1; d->i[1]=NH; d->i[2]=CTX; d->i[3]=0; d->i[4]=NSPLIT;
      d->i[5]=PLOW_KV_MASK_NONE; d->i[6]=TOPK; d->fj[0].f=SCALE_MLA; }
    addwait(i_ga,i_sel,1);
    int i_mg=emitop(PLOW_DOP_FLASH_MERGE,NCU);
    { PlowDevInst* d=&g_inst[i_mg]; d->t[0]=TOLAT; d->t[1]=TOP; d->t[2]=TML; d->i[0]=1; d->i[1]=NH; d->i[2]=NSPLIT; d->i[3]=512; }
    addwait(i_mg,i_ga,NCU);
    int i_uv=emitop(PLOW_DOP_O_UV_FOLD,NCU);
    { PlowDevInst* d=&g_inst[i_uv]; d->t[0]=TO; d->t[1]=TOLAT; d->t[2]=TWUV; d->i[0]=1; d->i[1]=NH; d->i[2]=VD; }
    addwait(i_uv,i_mg,NCU);
    const int n_ops=g_nops;

    void* d_tens=plow_hsa_alloc(h,0,(size_t)g_nt*sizeof(void*)); plow_hsa_upload(h,0,d_tens,g_tens,(size_t)g_nt*sizeof(void*));
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
    void* d_succs=plow_hsa_alloc(h,0,(size_t)n_ops*4); void* d_ctr=plow_hsa_alloc(h,0,(size_t)n_ops*PLOW_CTR_STRIDE*4);
    plow_hsa_upload(h,0,d_inst,g_inst,(size_t)n_ops*sizeof(PlowDevInst));
    plow_hsa_upload(h,0,d_stream,stream,total*sizeof(PlowStreamEnt));
    plow_hsa_upload(h,0,d_sofs,sofs,4u*NCU); plow_hsa_upload(h,0,d_slen,slen,4u*NCU);
    if(g_nw) plow_hsa_upload(h,0,d_waits,g_wait,(size_t)g_nw*sizeof(PlowWait));
    plow_hsa_upload(h,0,d_succs,g_succ,(size_t)n_ops*4);
    uint32_t* zc=calloc((size_t)n_ops*PLOW_CTR_STRIDE,4); plow_hsa_upload(h,0,d_ctr,zc,(size_t)n_ops*PLOW_CTR_STRIDE*4);
    PlowProgram prog; memset(&prog,0,sizeof(prog));
    prog.insts=d_inst; prog.stream=d_stream; prog.stream_ofs=d_sofs; prog.stream_len=d_slen;
    prog.waits=d_waits; prog.succs=d_succs; prog.counters=d_ctr; prog.tensors=(void* const*)d_tens;

    printf("program: %d ops (select->gather->merge->fold), ctx=%d top_k=%d nsplit=%d\n\n",n_ops,CTX,TOPK,NSPLIT);
    if(plow_hsa_launch(h,0,&kern,NCU*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&prog,sizeof(prog))){ printf("LAUNCH FAILED\n"); return 1; }
    plow_hsa_wait(h,0);

    /* 1. selection set compare */
    int idx_dev[TOPK]; plow_hsa_download(h,0,idx_dev,d_idx,TOPK*4);
    static int sel_ref[TOPK]; ref_select(h_Qi,h_Ki,sel_ref);
    static char in_dev[CTX], in_ref[CTX]; memset(in_dev,0,sizeof(in_dev)); memset(in_ref,0,sizeof(in_ref));
    int idx_range_ok=1;
    for(int j=0;j<TOPK;j++){ if(idx_dev[j]<0||idx_dev[j]>=CTX){ idx_range_ok=0; continue; } in_dev[idx_dev[j]]=1; }
    for(int j=0;j<TOPK;j++) in_ref[sel_ref[j]]=1;
    int set_eq=idx_range_ok; for(int t=0;t<CTX;t++) if(in_dev[t]!=in_ref[t]) set_eq=0;
    printf("  ATTN_SELECT set == reference top-%d: %s\n", TOPK, set_eq?"YES (exact)":"*** NO ***");

    /* 2. gathered MLA over the DEVICE-selected set, tolerance compare */
    bf16 o_dev[NH*VD]; plow_hsa_download(h,0,o_dev,d_o,(size_t)NH*VD*2);
    static bf16 o_ref[NH*VD]; ref_mla_gather(h_Qa,h_Qr,h_Ckv,h_Kr,h_Wuv,idx_dev,TOPK,o_ref);
    double mw=0,md=0,se=0,sw=0;
    for(int i=0;i<NH*VD;i++){ double d=fabs(b2f(o_dev[i])-b2f(o_ref[i])),w=fabs(b2f(o_ref[i]));
        mw=fmax(mw,w); md=fmax(md,d); se+=d*d; sw+=(double)b2f(o_ref[i])*b2f(o_ref[i]); }
    double rmax=md/(mw+1e-12), rrms=sqrt(se/(NH*VD))/(sqrt(sw/(NH*VD))+1e-12);
    int gather_ok=(rmax<3e-2 && rrms<1e-2);
    printf("  FLASH_GATHER MLA output vs reference: %s  (rel rms=%.5f max=%.4f)\n",
           gather_ok?"PASS":"FAIL", rrms, rmax);

    plow_hsa_download(h,0,zc,d_ctr,(size_t)n_ops*PLOW_CTR_STRIDE*4);
    int ctr_ok=1; for(int op=0;op<n_ops;op++) if(zc[(size_t)op*PLOW_CTR_STRIDE]!=g_inst[op].blocks) ctr_ok=0;
    printf("  executed==total: %s\n", ctr_ok?"YES":"NO");
    int ok=set_eq&&gather_ok&&ctr_ok;
    printf("\n%s\n", ok?"DSA SELECT->GATHER OK — selection exact, gathered MLA within tolerance"
                      :"*** DSA PATH MISMATCH ***");
    plow_hsa_shutdown(h);
    return ok?0:1;
}
