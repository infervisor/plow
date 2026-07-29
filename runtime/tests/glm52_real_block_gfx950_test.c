/* glm52_real_block_gfx950_test.c — ONE REAL-WEIGHT GLM-5.2 (GlmMoeDsa) MoE decoder block on the
 * interpreter, DIFFed against the HF-transformers oracle (glm52_real_oracle.py fixture).  [GLM52-B4]
 *
 * The B1 harness (glm52_block_gfx950_test.c) ran SYNTHETIC weights + E=16 experts and re-quantised
 * bf16->fp8 on the fly. This one loads the REAL zai-org/GLM-5.2-FP8 layer-3 weights: the REAL 256
 * [128,128]-block-fp8 experts with their REAL weight_scale_inv grids fed VERBATIM to plow's Sg/Sd
 * scale tables (no transpose). It reconfirms what B1 simplified:
 *   (1) 256-wide top-8 SELECTION on the device router (bf16 dot) vs HF fp32,
 *   (2) the real block-fp8 scale layout through MoeExpertGluFp8Blk/DownFp8Blk (45/46).
 *
 * KEY DE-RISK METRIC: the layer-3 RMSNorm gammas are small, so the MoE contribution is tiny in
 * ABSOLUTE terms (~0.05% of the residual). The block output is therefore INSENSITIVE to fp8 expert
 * error. So the fp8 de-risk is the EXPERT PATH diffed DIRECTLY in f32: the interp's per-slot `part`
 * ([TOPK,H] f32, gate-scaled down outputs) is downloaded and summed on host to expert_sum_plow, and
 * diffed against the f32 reference expert_sum — NEVER recovered via the catastrophic bf16 xnext-xmid.
 *
 * Two passes: bf16 experts (host-dequantised from the real fp8+scales; isolates router/MLA/dispatch)
 * then block-fp8 (the real kernels). Runs on ONE gfx950 GPU (~29GB: 9.7GB fp8 + 19GB bf16 experts).
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
/* OCP e4m3fn decode (matches block_fp8_gfx950_test / torch.float8_e4m3fn) */
static float e4m3_decode(unsigned char b){ const int s=(b>>7)&1,e=(b>>3)&0xF,m=b&0x7; double v;
    if(e==0) v=(m/8.0)*0.015625; else v=(1.0+m/8.0)*ldexp(1.0,e-7); return (float)(s?-v:v); }

/* ---- program builder (verbatim from glm52_block_gfx950_test.c) ---- */
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
static double relerr_f32(const float* got,const float* want,int n,double* worst){
    double mw=0,md=0,se=0,sw=0;
    for(int i=0;i<n;i++){ double d=fabs((double)got[i]-want[i]),w=fabs((double)want[i]);
        mw=fmax(mw,w); md=fmax(md,d); se+=d*d; sw+=(double)want[i]*want[i]; }
    *worst=md/(mw+1e-9); return sqrt(se/n)/(sqrt(sw/n)+1e-9);
}

/* ---- dims (real GLM-5.2, from fixture header) ---- */
static int L,H,NH,DK,DR,QN,VD,QL,E,TOPK,IMOE,qpos;
static float EPS,SCALE,RSCALE;
#define NSPLIT 8
#define ACT 1u
#define FLAGS (1u|2u|4u)   /* sigmoid + norm_topk + e_score_correction_bias */

struct W {
    void *x,*gin,*qad,*gqa,*wqa,*wqr,*ckvd,*gkva,*krotd,*wuv,*wo,*gpost,*ckv,*krot,*wr,*bias;
    void *wtab_bf, *wtab8, *stab8;
    void *shg,*shu,*shd;
    void *xn,*qlr,*qlat,*ckvraw,*Qa,*Qr,*Opart,*ml,*Olat,*oat,*attn,*xmid,*xn2,*tab,*shfu,*shared,*fu,*part,*xnext,*klen;
};

/* build + launch ONE block; use_fp8 selects the expert opcode set; downloads x_next + substeps + part */
static void run_block(plow_hsa* h, plow_hsa_kernel* kern, unsigned NCU, struct W* w, int use_fp8,
                      bf16* out_attn, bf16* out_xmid, bf16* out_xn2, bf16* out_shared,
                      bf16* out_xnext, unsigned char* out_tab, float* out_part){
    g_nops=0; g_nw=0; g_nt=0;
    memset(g_inst,0,sizeof(*g_inst)*256);
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
    int i_rn1=emitop(PLOW_DOP_RMSNORM,1);
    { PlowDevInst*d=&g_inst[i_rn1]; d->t[0]=TXN; d->t[1]=TX; d->t[2]=TGIN; d->i[0]=1; d->i[1]=H; d->fj[0].f=EPS; }
    GEMV(TQLR,TXN,TQAD,QL,H); int i_qad=_last; addwait(i_qad,i_rn1,1);
    int i_rnq=emitop(PLOW_DOP_RMSNORM,1);
    { PlowDevInst*d=&g_inst[i_rnq]; d->t[0]=TQLAT; d->t[1]=TQLR; d->t[2]=TGQA; d->i[0]=1; d->i[1]=QL; d->fj[0].f=EPS; }
    addwait(i_rnq,i_qad,NCU);
    GEMV(TQA,TQLAT,TWQA,NH*DK,QL); int i_qa=_last; addwait(i_qa,i_rnq,1);
    GEMV(TQR,TQLAT,TWQR,NH*DR,QL); int i_qr=_last; addwait(i_qr,i_rnq,1);
    GEMV(TCKVRAW,TXN,TCKVD,DK,H); int i_ckvd=_last; addwait(i_ckvd,i_rn1,1);
    int i_rnkv=emitop(PLOW_DOP_RMSNORM,1);
    { PlowDevInst*d=&g_inst[i_rnkv]; d->t[0]=TCKVROW; d->t[1]=TCKVRAW; d->t[2]=TGKVA; d->i[0]=1; d->i[1]=DK; d->fj[0].f=EPS; }
    addwait(i_rnkv,i_ckvd,NCU);
    GEMV(TKRROW,TXN,TKROTD,DR,H); int i_krd=_last; addwait(i_krd,i_rn1,1);
    int i_fl=emitop(PLOW_DOP_FLASH_MLA_DECODE,NCU);
    { PlowDevInst*d=&g_inst[i_fl]; d->t[0]=TOP; d->t[1]=TML; d->t[2]=TQA; d->t[3]=TQR; d->t[4]=TCKV;
      d->t[5]=TKROT; d->t[6]=TKLEN; d->i[0]=1; d->i[1]=NH; d->i[2]=L; d->i[4]=NSPLIT;
      d->i[5]=PLOW_KV_MASK_NONE; d->fj[0].f=SCALE; }
    addwait(i_fl,i_qa,NCU); addwait(i_fl,i_qr,NCU); addwait(i_fl,i_rnkv,1); addwait(i_fl,i_krd,NCU);
    /* Latent epilogue. GLM_FUSED_FOLD=1 runs the SINGLE fused op (57, d_mla_merge_fold) that the
     * real GLM decode packet emits; the default keeps the original FLASH_MERGE<512> + O_UV_FOLD
     * pair this gate was written against. Both are checked, because this gate is the only numeric
     * instrument on the epilogue — mla_test does not exercise d_mla_merge_fold at all — and
     * `MLA attn_out` below is the stage directly downstream of it. Same operands either way:
     * the fused op reads (Opart, mlpart, W_uv) and writes o[head][V] without the Olat round-trip. */
    int i_uv;
    if(getenv("GLM_FUSED_FOLD") && atoi(getenv("GLM_FUSED_FOLD"))){
        i_uv=emitop(PLOW_DOP_MLA_MERGE_FOLD,NCU);
        { PlowDevInst*d=&g_inst[i_uv]; d->t[0]=TOAT; d->t[1]=TOP; d->t[2]=TML; d->t[3]=TWUV;
          d->i[0]=1; d->i[1]=NH; d->i[2]=VD; d->i[4]=NSPLIT; }
        addwait(i_uv,i_fl,NCU);
    } else {
        int i_mg=emitop(PLOW_DOP_FLASH_MERGE,NCU);
        { PlowDevInst*d=&g_inst[i_mg]; d->t[0]=TOLAT; d->t[1]=TOP; d->t[2]=TML; d->i[0]=1; d->i[1]=NH; d->i[2]=NSPLIT; d->i[3]=512; }
        addwait(i_mg,i_fl,NCU);
        i_uv=emitop(PLOW_DOP_O_UV_FOLD,NCU);
        { PlowDevInst*d=&g_inst[i_uv]; d->t[0]=TOAT; d->t[1]=TOLAT; d->t[2]=TWUV; d->i[0]=1; d->i[1]=NH; d->i[2]=VD; }
        addwait(i_uv,i_mg,NCU);
    }
    GEMV(TATT,TOAT,TWO,H,NH*VD); int i_op=_last; addwait(i_op,i_uv,NCU);
    int i_rs=emitop(PLOW_DOP_RESIDUAL,1);
    { PlowDevInst*d=&g_inst[i_rs]; d->t[0]=TXMID; d->t[1]=TX; d->t[2]=TATT; d->i[0]=H; d->fj[0].f=1.0f; }
    addwait(i_rs,i_op,NCU);
    int i_rn2=emitop(PLOW_DOP_RMSNORM,1);
    { PlowDevInst*d=&g_inst[i_rn2]; d->t[0]=TXN2; d->t[1]=TXMID; d->t[2]=TGPOST; d->i[0]=1; d->i[1]=H; d->fj[0].f=EPS; }
    addwait(i_rn2,i_rs,1);
    int i_router=emitop(PLOW_DOP_MOE_ROUTER,1);
    { PlowDevInst*d=&g_inst[i_router]; d->t[0]=TTAB; d->t[1]=TXN2; d->t[2]=TWR; d->t[3]=TBIAS;
      d->i[0]=H; d->i[1]=E; d->i[2]=TOPK; d->i[3]=FLAGS; d->fj[0].f=RSCALE; }
    addwait(i_router,i_rn2,1);
    int i_shglu=emitop(PLOW_DOP_GEMV_GLU,NCU);
    { PlowDevInst*d=&g_inst[i_shglu]; d->t[0]=TSHFU; d->t[1]=TXN2; d->t[2]=TSHG; d->t[5]=TSHU;
      d->i[0]=1; d->i[1]=IMOE; d->i[2]=H; d->i[5]=ACT; }
    addwait(i_shglu,i_rn2,1);
    GEMV(TSHARED,TSHFU,TSHD,H,IMOE); int i_shd=_last; addwait(i_shd,i_shglu,NCU);
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
    plow_hsa_download(h,0,out_part,w->part,(size_t)TOPK*H*4);
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
    const char* fix = argc>2?argv[2]:"glm52_real_fixture.bin";
    setbuf(stdout,NULL);
    plow_hsa* h=plow_hsa_init(); if(!h){ printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus=0,lds=0; plow_hsa_device_info(h,0,gfx,&cus,&lds);
    const unsigned NCU=cus; printf("dev0: %s CUs=%u\n",gfx,cus);
    FILE* f=fopen(elf,"rb"); if(!f){ printf("%s missing\n",elf); return 1; }
    fseek(f,0,SEEK_END); long co_n=ftell(f); fseek(f,0,SEEK_SET);
    void* co=malloc(co_n); if(fread(co,1,co_n,f)!=(size_t)co_n) return 1; fclose(f);
    if(plow_hsa_load_code_object(h,0,co,co_n)){ printf("load failed\n"); return 1; }
    plow_hsa_kernel kern; if(plow_hsa_get_kernel(h,0,"plow_interp_dec_gfx950",&kern)){ printf("no kernel\n"); return 1; }

    int fd=open(fix,O_RDONLY); if(fd<0){ perror(fix); return 1; }
    struct stat st; fstat(fd,&st);
    char* base=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){ perror("mmap"); return 1; }
    int32_t* hdr=(int32_t*)base;
    if(hdr[0]!=0x474C4D36){ printf("bad magic %x (want GLM6 0x474C4D36)\n",hdr[0]); return 1; }
    L=hdr[1];H=hdr[2];NH=hdr[3];DK=hdr[4];DR=hdr[5];QN=hdr[6];VD=hdr[7];QL=hdr[8];E=hdr[9];TOPK=hdr[10];IMOE=hdr[11];qpos=hdr[12];
    float* fh=(float*)(base+13*4); EPS=fh[0];SCALE=fh[1];RSCALE=fh[2];
    printf("dims L=%d H=%d NH=%d DK=%d DR=%d QN=%d VD=%d QL=%d E=%d TOPK=%d IMOE=%d qpos=%d scale=%.4f eps=%.1e\n",
           L,H,NH,DK,DR,QN,VD,QL,E,TOPK,IMOE,qpos,SCALE,EPS);
    const int IB=(IMOE+127)/128, HB=(H+127)/128;   /* 16, 48 */
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
    unsigned char* P_efp8 = (unsigned char*)NEXT((size_t)E*3*IMOE*H,1);      /* per e: gate,up,down fp8 */
    float* P_esc    = (float*)NEXT((size_t)E*3*IB*HB,4);                     /* per e: Sg,Su,Sd f32 */
    bf16* P_shg     = NEXT((size_t)IMOE*H,2);
    bf16* P_shu     = NEXT((size_t)IMOE*H,2);
    bf16* P_shd     = NEXT((size_t)H*IMOE,2);
    bf16* P_blk     = NEXT((size_t)H,2);
    int32_t* P_sel  = NEXT((size_t)TOPK,4);
    float* P_selg   = NEXT((size_t)TOPK,4);
    bf16* P_attn    = NEXT((size_t)H,2);
    bf16* P_xn2     = NEXT((size_t)H,2);
    bf16* P_h1      = NEXT((size_t)H,2);
    float* P_esum   = (float*)NEXT((size_t)H,4);      /* HF expert_sum (f32) — the fp8 de-risk target */
    bf16* P_shref   = NEXT((size_t)H,2);              /* HF shared_out (bf16) */
    if(off!=(size_t)st.st_size){ printf("FIXTURE SIZE MISMATCH: parsed %zu != file %ld\n",off,st.st_size); return 1; }

#define UP(src,bytes) ({ void* _d=plow_hsa_alloc(h,0,bytes); plow_hsa_upload(h,0,_d,src,bytes); _d; })
    struct W w; memset(&w,0,sizeof(w));
    w.x=UP(P_x,H*2); w.gin=UP(P_gin,H*2); w.qad=UP(P_qad,(size_t)QL*H*2); w.gqa=UP(P_gqa,QL*2);
    w.wqa=UP(P_wqa,(size_t)NH*DK*QL*2); w.wqr=UP(P_wqr,(size_t)NH*DR*QL*2);
    w.ckvd=UP(P_ckvd,(size_t)DK*H*2); w.gkva=UP(P_gkva,DK*2); w.krotd=UP(P_krotd,(size_t)DR*H*2);
    w.wuv=UP(P_wuv,(size_t)NH*DK*VD*2); w.wo=UP(P_wo,(size_t)H*NH*VD*2); w.gpost=UP(P_gpost,H*2);
    w.ckv=UP(P_ckv,(size_t)L*DK*2); w.krot=UP(P_krot,(size_t)L*DR*2);
    w.wr=UP(P_wr,(size_t)E*H*2); w.bias=UP(P_bias,(size_t)E*4);
    w.shg=UP(P_shg,(size_t)IMOE*H*2); w.shu=UP(P_shu,(size_t)IMOE*H*2); w.shd=UP(P_shd,(size_t)H*IMOE*2);

    const size_t EW = (size_t)IMOE*H;            /* gate/up/down each IMOE*H fp8 bytes */
    const size_t ESC = (size_t)IB*HB;            /* each scale grid IB*HB f32 */
    /* fp8 experts: upload the whole real fp8 blob + scale blob ONCE; point wtab8/stab8 into them */
    printf("uploading %.2f GB real fp8 experts + scales ...\n",(double)E*3*EW/1e9);
    void* d_efp8=UP(P_efp8,(size_t)E*3*EW);
    void* d_esc =UP(P_esc,(size_t)E*3*ESC*4);
    uint64_t *wtb8=malloc((size_t)E*3*8), *stb8=malloc((size_t)E*3*8);
    for(int e=0;e<E;e++) for(int j=0;j<3;j++){
        wtb8[e*3+j]=(uint64_t)(uintptr_t)d_efp8+((size_t)e*3+j)*EW;
        stb8[e*3+j]=(uint64_t)(uintptr_t)d_esc +((size_t)(e*3+j)*ESC)*4; }
    w.wtab8=UP(wtb8,(size_t)E*3*8); w.stab8=UP(stb8,(size_t)E*3*8);
    /* bf16 experts: host-dequantise the real fp8+scales into one big GPU buffer (isolate router/MLA) */
    printf("dequantising %.2f GB bf16 experts (host) ...\n",(double)E*3*EW*2/1e9);
    void* d_ebf=plow_hsa_alloc(h,0,(size_t)E*3*EW*2);
    bf16* scr=malloc(3*EW*2);
    for(int e=0;e<E;e++){
        unsigned char* efp=P_efp8+(size_t)e*3*EW; float* esc=P_esc+(size_t)e*3*ESC;
        /* gate,up: [IMOE][H], scale [IB][HB] */
        for(int which=0;which<2;which++){ unsigned char* Wq=efp+(size_t)which*EW; float* S=esc+(size_t)which*ESC;
            bf16* dst=scr+(size_t)which*EW;
            for(int n=0;n<IMOE;n++) for(int k=0;k<H;k++)
                dst[(size_t)n*H+k]=f2b(e4m3_decode(Wq[(size_t)n*H+k])*S[(n>>7)*HB+(k>>7)]); }
        /* down: [H][IMOE], scale [HB][IB] */
        { unsigned char* Wq=efp+(size_t)2*EW; float* S=esc+(size_t)2*ESC; bf16* dst=scr+(size_t)2*EW;
          for(int n=0;n<H;n++) for(int k=0;k<IMOE;k++)
              dst[(size_t)n*IMOE+k]=f2b(e4m3_decode(Wq[(size_t)n*IMOE+k])*S[(n>>7)*IB+(k>>7)]); }
        plow_hsa_upload(h,0,(char*)d_ebf+(size_t)e*3*EW*2,scr,3*EW*2);
    }
    free(scr);
    uint64_t* wtb=malloc((size_t)E*3*8);
    for(int e=0;e<E;e++) for(int j=0;j<3;j++) wtb[e*3+j]=(uint64_t)(uintptr_t)d_ebf+((size_t)e*3+j)*EW*2;
    w.wtab_bf=UP(wtb,(size_t)E*3*8);

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

    bf16 *attn=malloc(H*2),*xmid=malloc(H*2),*xn2=malloc(H*2),*shared=malloc(H*2),*xnext=malloc(H*2),*xnext8=malloc(H*2);
    unsigned char* tab=malloc((size_t)TOPK*8);
    float* part=malloc((size_t)TOPK*H*4);
    float* esum=malloc((size_t)H*4);

    printf("\n=== HF oracle last-token router pick (fp32) ===\n  ");
    for(int j=0;j<TOPK;j++) printf("e%d(g=%.4f) ",P_sel[j],P_selg[j]); printf("\n");
    printf("  ref: expert_sum(f32) present, shared_out(bf16) present\n");
    printf("  latent epilogue: %s\n", (getenv("GLM_FUSED_FOLD") && atoi(getenv("GLM_FUSED_FOLD")))
                                          ? "FUSED MLA_MERGE_FOLD (op 57) — what the real packet emits"
                                          : "FLASH_MERGE<512> + O_UV_FOLD (the separate pair)");

    int ok=1;
    for(int pass=0;pass<2;pass++){
        int use_fp8=pass;
        bf16* xn=(pass?xnext8:xnext);
        run_block(h,&kern,NCU,&w,use_fp8,attn,xmid,xn2,shared,xn,tab,part);
        printf("  [%s] plow router pick: ",use_fp8?"fp8":"bf16");
        int set_ok=1; for(int j=0;j<TOPK;j++){ unsigned id=*(unsigned*)(tab+j*8); float g=*(float*)(tab+j*8+4);
            printf("e%u(g=%.4f) ",id,g); int found=0; for(int k=0;k<TOPK;k++) if((int)id==P_sel[k]) found=1; if(!found) set_ok=0; }
        printf("\n     top-%d SET vs HF: %s\n",TOPK,set_ok?"MATCH":"*** DIFFERS ***");
        /* expert_sum(f32) = Σ_slot part[slot] — the fp8 de-risk, diffed directly (no bf16 cancellation) */
        for(int i=0;i<H;i++){ double a=0; for(int s=0;s<TOPK;s++) a+=part[(size_t)s*H+i]; esum[i]=(float)a; }
        double we,ws2,w1,w3,w4;
        double r_esum = relerr_f32(esum,P_esum,H,&we);
        double r_shar = relerr(shared,P_shref,H,&ws2);
        double r_attn = relerr(attn,P_attn,H,&w1);
        double r_xn2  = relerr(xn2,P_xn2,H,&w3);
        double r_next = relerr(xn,P_blk,H,&w4);
        printf("     MLA  attn_out    : rms=%.5f max=%.4f\n",r_attn,w1);
        printf("     xn2  (post_ln)   : rms=%.5f max=%.4f\n",r_xn2,w3);
        printf("     shared_expert    : rms=%.5f max=%.4f\n",r_shar,ws2);
        printf("     >> EXPERT_SUM f32 : rms=%.5f max=%.4f   <<< fp8 de-risk (real [128,128] scales)\n",r_esum,we);
        printf("     x_next (BLOCK)   : rms=%.5f max=%.4f  (residual-dominated; sanity only)\n",r_next,w4);
        double tol = use_fp8?4e-2:1.5e-2;
        int pass_ok = set_ok && r_attn<tol && r_xn2<tol && r_esum<(use_fp8?6e-2:2e-2) && r_shar<2e-2;
        printf("     => %s\n", pass_ok?"PASS":"*** FAIL ***");
        ok &= pass_ok;
    }
    double wd; double r_d=relerr(xnext8,xnext,H,&wd);
    printf("\n  block-fp8 vs bf16 x_next delta: rms=%.5f max=%.4f (small — MoE is ~0.05%% of residual)\n",r_d,wd);
    printf("\n%s\n", ok?"GLM52 REAL-WEIGHT SINGLE-LAYER B4 OK — plow matches HF (real 256 experts, real block-fp8 scales)"
                      : "*** GLM52 REAL-WEIGHT B4 MISMATCH ***");
    munmap(base,st.st_size); plow_hsa_shutdown(h);
    return ok?0:1;
}
