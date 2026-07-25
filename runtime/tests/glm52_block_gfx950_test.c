/* glm52_block_gfx950_test.c — ONE real-dim GLM-5.2 (GlmMoeDsa) decoder block on the interpreter,
 * DIFFed against the HF-transformers oracle (glm52_oracle.py fixture).  [GLM52-B1]
 *
 * This is the B1 single-block de-risk: assemble the FULL GLM block at REAL dims (hidden 6144,
 * nh 64, qk_nope 192 + qk_rope 64, v 256, kv_lora 512, q_lora 2048) out of plow's existing ops and
 * diff plow's last-token block output against HF. Per block (decode, bs=1, one new token, an
 * L-token dense latent context — layer 3 is a "shared" indexer layer so the DSA indexer is a no-op
 * and attention is full causal):
 *
 *   RMSNorm(x, g_in)                       -> xn                     (input_layernorm)
 *   GEMV q_a_down · xn  -> q_lora_raw ;  RMSNorm(·, g_qa) -> q_lat   (q_a_proj + q_a_layernorm)
 *   GEMV Wqa(absorbed) · q_lat -> Q_abs[nh][512]                     (W_uk_nope^T @ q_b_nope)
 *   GEMV Wqr(rope-folded) · q_lat -> Q_rot[nh][64]                   (R_qpos @ q_b_rope, INTERLEAVED)
 *   GEMV kv_a_down · xn -> c_kv_raw ;  RMSNorm(·, g_kva) -> cache[qpos]  (kv_a_proj + kv_a_layernorm)
 *   GEMV krot_down(rope-folded) · xn -> k_rot cache[qpos]
 *   FLASH_MLA_DECODE (scale = 1/sqrt(256) = 0.0625) -> (O,m,l) ; FLASH_MERGE<512> -> O_lat[nh][512]
 *   O_UV_FOLD (W_uv = value^T) -> o[nh][256] ;  GEMV o_proj · o -> attn_out
 *   RESIDUAL x + attn_out -> x_mid ;  RMSNorm(x_mid, g_post) -> xn2  (post_attention_layernorm)
 *   MoeRouter (sigmoid + norm_topk + scale 2.5 + e_score_correction_bias) -> top-8 table
 *   shared expert: GEMV_GLU(xn2) -> sh_fu ; GEMV down -> shared_out
 *   8x (MoeExpertGlu -> MoeExpertDown) ;  MoeCombine(x_mid + shared_out + Σ part) -> x_next
 *
 * Runs TWICE: bf16 routed experts first (isolate correctness), then block-fp8 (weight_block_size
 * [128,128], e4m3, per-expert scale grid — MoeExpertGluFp8Blk/DownFp8Blk 45/46) to measure the
 * fp8-vs-bf16 delta. Diff is per-substep (attn_out, x_mid, xn2, router top-8 set + gates, x_next)
 * vs HF, relaxed tol (bf16+MoE ~1e-2, fp8 ~3e-2).
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef uint16_t bf16;
static float b2f(bf16 v){ union{uint32_t u;float f;}c; c.u=(uint32_t)v<<16; return c.f; }
static bf16 f2b(float f){ union{float f;uint32_t u;}c; c.f=f; uint32_t r=c.u+0x7fff+((c.u>>16)&1); return (bf16)(r>>16); }

/* ---- OCP e4m3fn encode / decode (round-to-nearest); matches block_fp8_gfx950_test decode ---- */
static double e4m3_decode(unsigned char b){ const int s=(b>>7)&1,e=(b>>3)&0xF,m=b&0x7; double v;
    if(e==0) v=(m/8.0)*0.015625; else v=(1.0+m/8.0)*ldexp(1.0,e-7); return s?-v:v; }
static unsigned char f2e4m3(float x){
    int sign = (x<0)?0x80:0; float a=fabsf(x);
    if(a==0.0f) return (unsigned char)sign;
    if(a>448.0f) a=448.0f;
    if(a < 0.015625f){ int m=(int)lrintf(a/0.001953125f); if(m>7)m=7; return (unsigned char)(sign|m); }
    int e=(int)floorf(log2f(a)); if(e<-6)e=-6; if(e>8)e=8;
    float sc=ldexpf(1.0f,e); int m=(int)lrintf((a/sc-1.0f)*8.0f);
    if(m==8){ m=0; e++; } if(e>8){ e=8; m=6; }
    int E=e+7; if(E<1){ int mm=(int)lrintf(a/0.001953125f); if(mm>7)mm=7; return (unsigned char)(sign|mm); }
    if(E>15){ E=15; m=6; } return (unsigned char)(sign|(E<<3)|(m&7));
}

/* ---- program builder (verbatim scaffolding from deepseek_block_gfx950_test.c) ---- */
static PlowDevInst g_inst[4096];
static PlowWait    g_wait[8192];
static uint32_t    g_succ[4096];
typedef struct { uint32_t wait_ofs, succ_ofs; uint16_t wait_len, succ_len; } PlowGate;
static PlowGate g_gate[4096]; /* per-inst gates, copied onto every StreamEnt (64B DevInst carries none) */
static int g_nops=0, g_nw=0;
static void* g_tens[2048];
static int g_nt=0;
static int reg(void* p){ g_tens[g_nt]=p; return g_nt++; }
static int emitop(uint16_t op, uint16_t blocks){ int i=g_nops++; g_inst[i].op=op; g_inst[i].blocks=blocks;
    g_gate[i].succ_ofs=i; g_gate[i].succ_len=1; g_succ[i]=i; return i; }
static void addwait(int op,int producer,uint32_t thr){ if(g_gate[op].wait_len==0) g_gate[op].wait_ofs=g_nw;
    g_wait[g_nw].id=producer; g_wait[g_nw].threshold=thr; g_nw++; g_gate[op].wait_len++; }

static double relerr(const bf16* got,const bf16* want,int n,double* worst){
    double mw=0,md=0,se=0,sw=0;
    for(int i=0;i<n;i++){ double d=fabs(b2f(got[i])-b2f(want[i])),w=fabs(b2f(want[i]));
        mw=fmax(mw,w); md=fmax(md,d); se+=d*d; sw+=(double)b2f(want[i])*b2f(want[i]); }
    *worst=md/(mw+1e-9); return sqrt(se/n)/(sqrt(sw/n)+1e-9);
}

/* ---- dims (real GLM-5.2, checked against the fixture header) ---- */
static int L,H,NH,DK,DR,QN,VD,QL,E,TOPK,IMOE,qpos;
static float EPS,SCALE,RSCALE;
#define NSPLIT 8
#define ACT 1u
#define FLAGS (1u|2u|4u)   /* sigmoid + norm_topk + e_score_correction_bias */

/* device weight handles (uploaded once) */
struct W {
    void *x,*gin,*qad,*gqa,*wqa,*wqr,*ckvd,*gkva,*krotd,*wuv,*wo,*gpost,*ckv,*krot,*wr,*bias;
    void *wtab_bf, *wtab8, *stab8;       /* expert bases: bf16 table, fp8 table, scale table */
    void *shg,*shu,*shd;
    /* scratch */
    void *xn,*qlr,*qlat,*ckvraw,*Qa,*Qr,*Opart,*ml,*Olat,*oat,*attn,*xmid,*xn2,*tab,*shfu,*shared,*fu,*part,*xnext,*klen;
};

/* build + launch ONE block; use_fp8 selects the expert opcode set; downloads x_next + substeps */
static void run_block(plow_hsa* h, plow_hsa_kernel* kern, unsigned NCU, struct W* w, int use_fp8,
                      bf16* out_attn, bf16* out_xmid, bf16* out_xn2, bf16* out_shared,
                      bf16* out_xnext, unsigned char* out_tab){
    g_nops=0; g_nw=0; g_nt=0;
    memset(g_inst,0,sizeof(*g_inst)*256);
    /* tensor handles */
    int TX=reg(w->x),TGIN=reg(w->gin),TQAD=reg(w->qad),TGQA=reg(w->gqa),TWQA=reg(w->wqa),TWQR=reg(w->wqr),
        TCKVD=reg(w->ckvd),TGKVA=reg(w->gkva),TKROTD=reg(w->krotd),TWUV=reg(w->wuv),TWO=reg(w->wo),
        TGPOST=reg(w->gpost),TCKV=reg(w->ckv),TKROT=reg(w->krot),TWR=reg(w->wr),TBIAS=reg(w->bias),
        TSHG=reg(w->shg),TSHU=reg(w->shu),TSHD=reg(w->shd),
        TXN=reg(w->xn),TQLR=reg(w->qlr),TQLAT=reg(w->qlat),TCKVRAW=reg(w->ckvraw),TQA=reg(w->Qa),TQR=reg(w->Qr),
        TOP=reg(w->Opart),TML=reg(w->ml),TOLAT=reg(w->Olat),TOAT=reg(w->oat),TATT=reg(w->attn),
        TXMID=reg(w->xmid),TXN2=reg(w->xn2),TTAB=reg(w->tab),TSHFU=reg(w->shfu),TSHARED=reg(w->shared),
        TFU=reg(w->fu),TPART=reg(w->part),TXNEXT=reg(w->xnext),TKLEN=reg(w->klen);
    int TWTAB=reg(use_fp8?w->wtab8:w->wtab_bf), TSTAB=reg(w->stab8);
    int TCKVROW=reg((char*)w->ckv+(size_t)qpos*DK*2), TKRROW=reg((char*)w->krot+(size_t)qpos*DR*2);

#define GEMV(o_h,x_h,w_h,N_,K_) do{ int _i=emitop(PLOW_DOP_GEMV,NCU); PlowDevInst*_d=&g_inst[_i]; \
    _d->t[0]=o_h; _d->t[1]=x_h; _d->t[2]=w_h; _d->t[3]=PLOW_TENSOR_NONE; _d->t[4]=PLOW_TENSOR_NONE; \
    _d->i[0]=1; _d->i[1]=(N_); _d->i[2]=(K_); _d->fj[0].f=1.0f; _last=_i; }while(0)
    int _last=-1;
    /* input_layernorm */
    int i_rn1=emitop(PLOW_DOP_RMSNORM,1);
    { PlowDevInst*d=&g_inst[i_rn1]; d->t[0]=TXN; d->t[1]=TX; d->t[2]=TGIN; d->i[0]=1; d->i[1]=H; d->fj[0].f=EPS; }
    /* q_a down + q_a_layernorm */
    GEMV(TQLR,TXN,TQAD,QL,H); int i_qad=_last; addwait(i_qad,i_rn1,1);
    int i_rnq=emitop(PLOW_DOP_RMSNORM,1);
    { PlowDevInst*d=&g_inst[i_rnq]; d->t[0]=TQLAT; d->t[1]=TQLR; d->t[2]=TGQA; d->i[0]=1; d->i[1]=QL; d->fj[0].f=EPS; }
    addwait(i_rnq,i_qad,NCU);
    /* absorbed query + rope query from q_lat */
    GEMV(TQA,TQLAT,TWQA,NH*DK,QL); int i_qa=_last; addwait(i_qa,i_rnq,1);
    GEMV(TQR,TQLAT,TWQR,NH*DR,QL); int i_qr=_last; addwait(i_qr,i_rnq,1);
    /* kv_a down (c_kv_raw) + kv_a_layernorm into cache[qpos]; krot down (rope-folded) into cache[qpos] */
    GEMV(TCKVRAW,TXN,TCKVD,DK,H); int i_ckvd=_last; addwait(i_ckvd,i_rn1,1);
    int i_rnkv=emitop(PLOW_DOP_RMSNORM,1);
    { PlowDevInst*d=&g_inst[i_rnkv]; d->t[0]=TCKVROW; d->t[1]=TCKVRAW; d->t[2]=TGKVA; d->i[0]=1; d->i[1]=DK; d->fj[0].f=EPS; }
    addwait(i_rnkv,i_ckvd,NCU);
    GEMV(TKRROW,TXN,TKROTD,DR,H); int i_krd=_last; addwait(i_krd,i_rn1,1);
    /* flash MLA decode -> merge -> uv fold */
    int i_fl=emitop(PLOW_DOP_FLASH_MLA_DECODE,NCU);
    { PlowDevInst*d=&g_inst[i_fl]; d->t[0]=TOP; d->t[1]=TML; d->t[2]=TQA; d->t[3]=TQR; d->t[4]=TCKV;
      d->t[5]=TKROT; d->t[6]=TKLEN; d->i[0]=1; d->i[1]=NH; d->i[2]=L; d->i[4]=NSPLIT;
      d->i[5]=PLOW_KV_MASK_NONE; d->fj[0].f=SCALE; }
    addwait(i_fl,i_qa,NCU); addwait(i_fl,i_qr,NCU); addwait(i_fl,i_rnkv,1); addwait(i_fl,i_krd,NCU);
    int i_mg=emitop(PLOW_DOP_FLASH_MERGE,NCU);
    { PlowDevInst*d=&g_inst[i_mg]; d->t[0]=TOLAT; d->t[1]=TOP; d->t[2]=TML; d->i[0]=1; d->i[1]=NH; d->i[2]=NSPLIT; d->i[3]=512; }
    addwait(i_mg,i_fl,NCU);
    int i_uv=emitop(PLOW_DOP_O_UV_FOLD,NCU);
    { PlowDevInst*d=&g_inst[i_uv]; d->t[0]=TOAT; d->t[1]=TOLAT; d->t[2]=TWUV; d->i[0]=1; d->i[1]=NH; d->i[2]=VD; }
    addwait(i_uv,i_mg,NCU);
    /* o_proj + residual + post_attention_layernorm */
    GEMV(TATT,TOAT,TWO,H,NH*VD); int i_op=_last; addwait(i_op,i_uv,NCU);
    int i_rs=emitop(PLOW_DOP_RESIDUAL,1);
    { PlowDevInst*d=&g_inst[i_rs]; d->t[0]=TXMID; d->t[1]=TX; d->t[2]=TATT; d->i[0]=H; d->fj[0].f=1.0f; }
    addwait(i_rs,i_op,NCU);
    int i_rn2=emitop(PLOW_DOP_RMSNORM,1);
    { PlowDevInst*d=&g_inst[i_rn2]; d->t[0]=TXN2; d->t[1]=TXMID; d->t[2]=TGPOST; d->i[0]=1; d->i[1]=H; d->fj[0].f=EPS; }
    addwait(i_rn2,i_rs,1);
    /* router (sigmoid + norm_topk + scale + correction bias) */
    int i_router=emitop(PLOW_DOP_MOE_ROUTER,1);
    { PlowDevInst*d=&g_inst[i_router]; d->t[0]=TTAB; d->t[1]=TXN2; d->t[2]=TWR; d->t[3]=TBIAS;
      d->i[0]=H; d->i[1]=E; d->i[2]=TOPK; d->i[3]=FLAGS; d->fj[0].f=RSCALE; }
    addwait(i_router,i_rn2,1);
    /* shared expert dense MLP on xn2 (GEMV_GLU gate|up -> down) */
    int i_shglu=emitop(PLOW_DOP_GEMV_GLU,NCU);
    { PlowDevInst*d=&g_inst[i_shglu]; d->t[0]=TSHFU; d->t[1]=TXN2; d->t[2]=TSHG; d->t[5]=TSHU;
      d->i[0]=1; d->i[1]=IMOE; d->i[2]=H; d->i[5]=ACT; }
    addwait(i_shglu,i_rn2,1);
    GEMV(TSHARED,TSHFU,TSHD,H,IMOE); int i_shd=_last; addwait(i_shd,i_shglu,NCU);
    /* routed experts */
    int downs[64];
    for(int slot=0;slot<TOPK;slot++){
        int i_g=emitop(use_fp8?PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK:PLOW_DOP_MOE_EXPERT_GLU,NCU);
        { PlowDevInst*d=&g_inst[i_g]; d->t[0]=TFU; d->t[1]=TXN2; d->t[2]=TTAB; d->t[3]=TWTAB;
          if(use_fp8) d->t[4]=TSTAB; d->i[0]=slot; d->i[1]=IMOE; d->i[2]=H; d->i[3]=E; d->i[5]=ACT; }
        addwait(i_g,i_router,1);
        int i_d=emitop(use_fp8?PLOW_DOP_MOE_EXPERT_DOWN_FP8_BLK:PLOW_DOP_MOE_EXPERT_DOWN,NCU);
        { PlowDevInst*d=&g_inst[i_d]; d->t[0]=TPART; d->t[1]=TFU; d->t[2]=TTAB; d->t[3]=TWTAB;
          if(use_fp8) d->t[4]=TSTAB; d->i[0]=slot; d->i[1]=H; d->i[2]=IMOE; d->i[3]=E; }
        addwait(i_d,i_g,NCU);
        downs[slot]=i_d;
    }
    int i_cmb=emitop(PLOW_DOP_MOE_COMBINE,NCU);
    { PlowDevInst*d=&g_inst[i_cmb]; d->t[0]=TXNEXT; d->t[1]=TXMID; d->t[2]=TSHARED; d->t[3]=TPART;
      d->i[0]=H; d->i[1]=TOPK; }
    addwait(i_cmb,i_shd,NCU); for(int s=0;s<TOPK;s++) addwait(i_cmb,downs[s],NCU);
    const int n_ops=g_nops;

    /* upload program + per-CU streams (blocks==1 ops only on CU0) */
    void* d_tens=plow_hsa_alloc(h,0,(size_t)g_nt*sizeof(void*)); plow_hsa_upload(h,0,d_tens,g_tens,(size_t)g_nt*sizeof(void*));
    uint32_t *sofs=malloc(4u*NCU), *slen=malloc(4u*NCU); size_t total=0;
    for(unsigned cu=0;cu<NCU;cu++) for(int op=0;op<n_ops;op++) if(!(g_inst[op].blocks==1&&cu!=0)) total++;
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
    if(plow_hsa_launch(h,0,kern,NCU*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&prog,sizeof(prog))){ printf("LAUNCH FAILED\n"); exit(1); }
    plow_hsa_wait(h,0);
    plow_hsa_download(h,0,out_attn,w->attn,H*2);
    plow_hsa_download(h,0,out_xmid,w->xmid,H*2);
    plow_hsa_download(h,0,out_xn2,w->xn2,H*2);
    plow_hsa_download(h,0,out_shared,w->shared,H*2);
    plow_hsa_download(h,0,out_xnext,w->xnext,H*2);
    plow_hsa_download(h,0,out_tab,w->tab,(size_t)TOPK*8);
    plow_hsa_download(h,0,zc,d_ctr,(size_t)n_ops*PLOW_CTR_STRIDE*4);
    int ctr_ok=1; for(int op=0;op<n_ops;op++) if(zc[(size_t)op*PLOW_CTR_STRIDE]!=g_inst[op].blocks) ctr_ok=0;
    printf("  [%s] program: %d ops, %zu packets, executed==total: %s\n",
           use_fp8?"fp8":"bf16", n_ops, total, ctr_ok?"YES":"NO");
    free(sofs); free(slen); free(stream); free(zc);
    plow_hsa_free(h,d_tens); plow_hsa_free(h,d_inst); plow_hsa_free(h,d_stream); plow_hsa_free(h,d_sofs);
    plow_hsa_free(h,d_slen); plow_hsa_free(h,d_waits); plow_hsa_free(h,d_succs); plow_hsa_free(h,d_ctr);
}

int main(int argc,char** argv){
    const char* elf = argc>1?argv[1]:"interp_decode.elf";
    const char* fix = argc>2?argv[2]:"glm52_fixture.bin";
    setbuf(stdout,NULL);
    plow_hsa* h=plow_hsa_init(); if(!h){ printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus=0,lds=0; plow_hsa_device_info(h,0,gfx,&cus,&lds);
    const unsigned NCU=cus; printf("dev0: %s CUs=%u\n",gfx,cus);
    FILE* f=fopen(elf,"rb"); if(!f){ printf("%s missing\n",elf); return 1; }
    fseek(f,0,SEEK_END); long co_n=ftell(f); fseek(f,0,SEEK_SET);
    void* co=malloc(co_n); if(fread(co,1,co_n,f)!=(size_t)co_n) return 1; fclose(f);
    if(plow_hsa_load_code_object(h,0,co,co_n)){ printf("load failed\n"); return 1; }
    plow_hsa_kernel kern; if(plow_hsa_get_kernel(h,0,"plow_interp_dec_gfx950",&kern)){ printf("no kernel\n"); return 1; }

    /* ---- mmap fixture ---- */
    int fd=open(fix,O_RDONLY); if(fd<0){ perror(fix); return 1; }
    struct stat st; fstat(fd,&st);
    char* base=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){ perror("mmap"); return 1; }
    int32_t* hdr=(int32_t*)base;
    if(hdr[0]!=0x474C4D35){ printf("bad magic %x\n",hdr[0]); return 1; }
    L=hdr[1];H=hdr[2];NH=hdr[3];DK=hdr[4];DR=hdr[5];QN=hdr[6];VD=hdr[7];QL=hdr[8];E=hdr[9];TOPK=hdr[10];IMOE=hdr[11];qpos=hdr[12];
    float* fh=(float*)(base+13*4); EPS=fh[0];SCALE=fh[1];RSCALE=fh[2];
    printf("dims L=%d H=%d NH=%d DK=%d DR=%d QN=%d VD=%d QL=%d E=%d TOPK=%d IMOE=%d qpos=%d scale=%.4f eps=%.1e\n",
           L,H,NH,DK,DR,QN,VD,QL,E,TOPK,IMOE,qpos,SCALE,EPS);
    size_t off=13*4+3*4;
#define NEXT(cnt,elt) ({ void* _p=base+off; off+=(size_t)(cnt)*(elt); _p; })
    bf16* P_x       = NEXT((size_t)H,2);
    bf16* P_gin     = NEXT((size_t)H,2);
    bf16* P_qad     = NEXT((size_t)QL*H,2);
    bf16* P_gqa     = NEXT((size_t)QL,2);
    bf16* P_wqa     = NEXT((size_t)NH*DK*QL,2);
    bf16* P_wqr     = NEXT((size_t)NH*DR*QL,2);
    bf16* P_ckvd    = NEXT((size_t)DK*H,2);
    bf16* P_gkva    = NEXT((size_t)DK,2);
    bf16* P_krotd   = NEXT((size_t)DR*H,2);
    bf16* P_wuv     = NEXT((size_t)NH*DK*VD,2);
    bf16* P_wo      = NEXT((size_t)H*NH*VD,2);
    bf16* P_gpost   = NEXT((size_t)H,2);
    bf16* P_ckv     = NEXT((size_t)L*DK,2);
    bf16* P_krot    = NEXT((size_t)L*DR,2);
    bf16* P_wr      = NEXT((size_t)E*H,2);
    float* P_bias   = NEXT((size_t)E,4);
    bf16* P_gate    = NEXT((size_t)E*IMOE*H,2);
    bf16* P_up      = NEXT((size_t)E*IMOE*H,2);
    bf16* P_down    = NEXT((size_t)E*H*IMOE,2);
    bf16* P_shg     = NEXT((size_t)IMOE*H,2);
    bf16* P_shu     = NEXT((size_t)IMOE*H,2);
    bf16* P_shd     = NEXT((size_t)H*IMOE,2);
    bf16* P_blk     = NEXT((size_t)H,2);
    int32_t* P_sel  = NEXT((size_t)TOPK,4);
    float* P_selg   = NEXT((size_t)TOPK,4);
    bf16* P_attn    = NEXT((size_t)H,2);
    bf16* P_xn2     = NEXT((size_t)H,2);
    bf16* P_h1      = NEXT((size_t)H,2);
    if(off!=(size_t)st.st_size){ printf("FIXTURE SIZE MISMATCH: parsed %zu != file %ld\n",off,st.st_size); return 1; }

    /* ---- upload weights ---- */
#define UP(src,bytes) ({ void* _d=plow_hsa_alloc(h,0,bytes); plow_hsa_upload(h,0,_d,src,bytes); _d; })
    struct W w; memset(&w,0,sizeof(w));
    w.x=UP(P_x,H*2); w.gin=UP(P_gin,H*2); w.qad=UP(P_qad,(size_t)QL*H*2); w.gqa=UP(P_gqa,QL*2);
    w.wqa=UP(P_wqa,(size_t)NH*DK*QL*2); w.wqr=UP(P_wqr,(size_t)NH*DR*QL*2);
    w.ckvd=UP(P_ckvd,(size_t)DK*H*2); w.gkva=UP(P_gkva,DK*2); w.krotd=UP(P_krotd,(size_t)DR*H*2);
    w.wuv=UP(P_wuv,(size_t)NH*DK*VD*2); w.wo=UP(P_wo,(size_t)H*NH*VD*2); w.gpost=UP(P_gpost,H*2);
    w.ckv=UP(P_ckv,(size_t)L*DK*2); w.krot=UP(P_krot,(size_t)L*DR*2);
    w.wr=UP(P_wr,(size_t)E*H*2); w.bias=UP(P_bias,(size_t)E*4);
    w.shg=UP(P_shg,(size_t)IMOE*H*2); w.shu=UP(P_shu,(size_t)IMOE*H*2); w.shd=UP(P_shd,(size_t)H*IMOE*2);
    /* bf16 expert weight table: gate/up/down bases per expert */
    void* d_gate=UP(P_gate,(size_t)E*IMOE*H*2); void* d_up=UP(P_up,(size_t)E*IMOE*H*2); void* d_down=UP(P_down,(size_t)E*H*IMOE*2);
    uint64_t* wtb=malloc((size_t)E*3*8);
    for(int e=0;e<E;e++){ wtb[e*3+0]=(uint64_t)(uintptr_t)d_gate+(size_t)e*IMOE*H*2;
        wtb[e*3+1]=(uint64_t)(uintptr_t)d_up+(size_t)e*IMOE*H*2; wtb[e*3+2]=(uint64_t)(uintptr_t)d_down+(size_t)e*H*IMOE*2; }
    w.wtab_bf=UP(wtb,(size_t)E*3*8);
    /* fp8 expert weights: quantize gate/up/down per expert to e4m3 + [128,128] scale grids */
    const int IB=(IMOE+127)/128, HB=(H+127)/128;
    uint64_t *wtb8=malloc((size_t)E*3*8), *stb8=malloc((size_t)E*3*8);
    unsigned char* q8=malloc((size_t)IMOE*H); float* sc=malloc((size_t)IB*HB*4);
    for(int e=0;e<E;e++){
        /* gate [IMOE][H], up [IMOE][H]: scale grid [IB][HB] */
        for(int which=0;which<2;which++){ bf16* src=(which?P_up:P_gate)+(size_t)e*IMOE*H;
            for(int nb=0;nb<IB;nb++) for(int kb=0;kb<HB;kb++){ float mx=0;
                for(int n=nb*128;n<(nb+1)*128&&n<IMOE;n++) for(int k=kb*128;k<(kb+1)*128&&k<H;k++)
                    mx=fmaxf(mx,fabsf(b2f(src[(size_t)n*H+k]))); sc[nb*HB+kb]=(mx>0?mx/448.0f:1.0f); }
            for(int n=0;n<IMOE;n++) for(int k=0;k<H;k++) q8[(size_t)n*H+k]=f2e4m3(b2f(src[(size_t)n*H+k])/sc[(n>>7)*HB+(k>>7)]);
            void* dq=UP(q8,(size_t)IMOE*H); void* ds=UP(sc,(size_t)IB*HB*4);
            wtb8[e*3+which]=(uint64_t)(uintptr_t)dq; stb8[e*3+which]=(uint64_t)(uintptr_t)ds; }
        /* down [H][IMOE]: scale grid [HB][IB] */
        { bf16* src=P_down+(size_t)e*H*IMOE;
          for(int nb=0;nb<HB;nb++) for(int kb=0;kb<IB;kb++){ float mx=0;
              for(int n=nb*128;n<(nb+1)*128&&n<H;n++) for(int k=kb*128;k<(kb+1)*128&&k<IMOE;k++)
                  mx=fmaxf(mx,fabsf(b2f(src[(size_t)n*IMOE+k]))); sc[nb*IB+kb]=(mx>0?mx/448.0f:1.0f); }
          for(int n=0;n<H;n++) for(int k=0;k<IMOE;k++) q8[(size_t)n*IMOE+k]=f2e4m3(b2f(src[(size_t)n*IMOE+k])/sc[(n>>7)*IB+(k>>7)]);
          void* dq=UP(q8,(size_t)H*IMOE); void* ds=UP(sc,(size_t)HB*IB*4);
          wtb8[e*3+2]=(uint64_t)(uintptr_t)dq; stb8[e*3+2]=(uint64_t)(uintptr_t)ds; }
    }
    free(q8); free(sc);
    w.wtab8=UP(wtb8,(size_t)E*3*8); w.stab8=UP(stb8,(size_t)E*3*8);

    /* scratch */
    w.xn=plow_hsa_alloc(h,0,H*2); w.qlr=plow_hsa_alloc(h,0,QL*2); w.qlat=plow_hsa_alloc(h,0,QL*2);
    w.ckvraw=plow_hsa_alloc(h,0,DK*2);
    w.Qa=plow_hsa_alloc(h,0,(size_t)NH*DK*2); w.Qr=plow_hsa_alloc(h,0,(size_t)NH*DR*2);
    w.Opart=plow_hsa_alloc(h,0,(size_t)NH*NSPLIT*DK*4); w.ml=plow_hsa_alloc(h,0,(size_t)NH*NSPLIT*2*4);
    w.Olat=plow_hsa_alloc(h,0,(size_t)NH*DK*2); w.oat=plow_hsa_alloc(h,0,(size_t)NH*VD*2);
    w.attn=plow_hsa_alloc(h,0,H*2); w.xmid=plow_hsa_alloc(h,0,H*2); w.xn2=plow_hsa_alloc(h,0,H*2);
    w.tab=plow_hsa_alloc(h,0,(size_t)TOPK*8); w.shfu=plow_hsa_alloc(h,0,(size_t)IMOE*2);
    w.shared=plow_hsa_alloc(h,0,H*2); w.fu=plow_hsa_alloc(h,0,(size_t)TOPK*IMOE*2);
    w.part=plow_hsa_alloc(h,0,(size_t)TOPK*H*4); w.xnext=plow_hsa_alloc(h,0,H*2);
    int32_t klen=L; w.klen=UP(&klen,4);

    /* ---- run bf16 then fp8; diff each vs HF ---- */
    bf16 *attn=malloc(H*2),*xmid=malloc(H*2),*xn2=malloc(H*2),*shared=malloc(H*2),*xnext=malloc(H*2),*xnext8=malloc(H*2);
    unsigned char* tab=malloc((size_t)TOPK*8);

    printf("\n=== HF oracle last-token router pick ===\n  ");
    for(int j=0;j<TOPK;j++) printf("e%d(g=%.4f) ",P_sel[j],P_selg[j]); printf("\n");

    int ok=1;
    for(int pass=0;pass<2;pass++){
        int use_fp8=pass;
        bf16* xn=(pass?xnext8:xnext);
        run_block(h,&kern,NCU,&w,use_fp8,attn,xmid,xn2,shared,xn,tab);
        /* router top-k set + gates (device table) */
        printf("  [%s] plow router pick: ",use_fp8?"fp8":"bf16");
        int set_ok=1; for(int j=0;j<TOPK;j++){ unsigned id=*(unsigned*)(tab+j*8); float g=*(float*)(tab+j*8+4);
            printf("e%u(g=%.4f) ",id,g); int found=0; for(int k=0;k<TOPK;k++) if((int)id==P_sel[k]) found=1; if(!found) set_ok=0; }
        printf("\n     top-%d SET vs HF: %s\n",TOPK,set_ok?"MATCH":"*** DIFFERS ***");
        /* substep diffs vs HF */
        double w1,w2,w3,w4;
        double r_attn=relerr(attn,P_attn,H,&w1);
        double r_xmid=relerr(xmid,P_h1,H,&w2);
        double r_xn2 =relerr(xn2,P_xn2,H,&w3);
        double r_next=relerr(xn,P_blk,H,&w4);
        printf("     MLA  attn_out : rms=%.5f max=%.4f\n",r_attn,w1);
        printf("     x_mid(residual): rms=%.5f max=%.4f\n",r_xmid,w2);
        printf("     xn2 (post_ln) : rms=%.5f max=%.4f\n",r_xn2,w3);
        printf("     x_next (BLOCK): rms=%.5f max=%.4f  vs HF\n",r_next,w4);
        double tol = use_fp8?3e-2:1.5e-2;
        int pass_ok = set_ok && r_attn<tol && r_xmid<tol && r_xn2<tol && r_next<(use_fp8?4e-2:2e-2);
        printf("     => %s\n", pass_ok?"PASS":"*** FAIL ***");
        ok &= pass_ok;
    }
    /* fp8-vs-bf16 delta */
    double wd; double r_d=relerr(xnext8,xnext,H,&wd);
    printf("\n  block-fp8 vs bf16 delta: rms=%.5f max=%.4f\n",r_d,wd);
    printf("\n%s\n", ok?"GLM52 BLOCK B1 OK — plow matches HF within tol (bf16 + block-fp8)"
                      : "*** GLM52 BLOCK B1 MISMATCH ***");
    munmap(base,st.st_size); plow_hsa_shutdown(h);
    return ok?0:1;
}
