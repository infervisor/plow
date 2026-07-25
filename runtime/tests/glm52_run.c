/* glm52_run.c — MILESTONE-1 CLOSE for the GLM-5.2 (GlmMoeDsa) serving emitter.  [GLM52-ms1]
 *
 * Drives the EMITTED single-layer .pkt (crates/plowc/src/bin/gemma4.rs::glm_main, Arch::GlmMoeDsa)
 * end-to-end and diffs it against the HF single-layer fixture — the real ms1 gate. Unlike the B4
 * harness (glm52_real_block_gfx950_test.c), which hand-built the 34-op sequence, this loads the
 * COMPILER's program + binds every weight BY NAME from the host-prepped weight dir (scripts/
 * glm52_prep.py). It closes the loop the offline glm_tests only asserted structurally.
 *
 * THREE pieces of loader glue the task calls out:
 *   1. expert tables — after binding, fill mlp.expert_weight_table[e*3+j] / expert_scale_table
 *      with the device addresses of the bound experts.{e}.{gate,up,down}_proj.weight(+_scale_inv),
 *      exactly as the B4 harness built wtb8/stb8 by hand.
 *   2. current-token cache-row write (out_row0) — the emitted RMSNORM/GEMV that write the latent
 *      (kv.L.ckv) and rope (kv.L.krot) caches target the cache BASE (row 0); for the fixed-position
 *      ms1 validation the current token lives at row qpos=L-1. We pre-populate the full [L] cache
 *      from the fixture and repoint just those two writer instructions to base+qpos (the B4 harness
 *      aliased TCKVROW/TKRROW the same way). FLASH still reads the whole [0,L) cache.
 *   3. input + reference — act.x is the fixture's block-input hidden[qpos]; the diff targets
 *      (block_out, router pick, attn, xn2, expert_sum f32, shared_out) come from the fixture.
 *
 * The block-fp8 expert opcodes (45/46) are compiled unconditionally into interp_decode.elf, so ONE
 * interpreter object runs both the bf16 (PLOW_FP8 unset) and block-fp8 pkt. Runs on ONE gfx950 GPU
 * (~30 GB: 10 GB prepped weights + activations). PLOW_FP8=1 must match the pkt the emitter produced.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_blob.h"

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
/* OCP e4m3fn decode (matches the fixture / torch.float8_e4m3fn) — bf16-pkt host dequant path. */
static float e4m3_decode(unsigned char b){ const int s=(b>>7)&1,e=(b>>3)&0xF,m=b&0x7; double v;
    if(e==0) v=(m/8.0)*0.015625; else v=(1.0+m/8.0)*ldexp(1.0,e-7); return (float)(s?-v:v); }

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

/* ---- blob (verbatim struct/loader from gemma4_chat.c; static-stream single program) ---- */
typedef struct {
    PlowProgHeader h;
    PlowDevInst* insts;
    PlowStreamEnt* stream;
    uint32_t *stream_ofs, *stream_len, *succs;
    PlowWait* waits;
    void *d_inst, *d_stream, *d_sofs, *d_slen, *d_waits, *d_succs, *d_ctr;
} Prog;
typedef struct {
    PlowBlobHeader h;
    PlowTensorDecl* tensors;
    uint8_t* init;
    uint32_t* kvrow;
    Prog* prog;
} Blob;
static int load_blob(const char* path, Blob* b) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* p = malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
    memcpy(&b->h, p, sizeof(PlowBlobHeader));
    if (memcmp(b->h.magic, PLOW_BLOB_MAGIC, 8)) { printf("bad blob magic\n"); return 1; }
    uint8_t* q = p + sizeof(PlowBlobHeader);
    b->tensors = (PlowTensorDecl*)q; q += (size_t)b->h.n_tensor * sizeof(PlowTensorDecl);
    b->init = q;                     q += b->h.init_bytes;
    b->kvrow = (uint32_t*)q;         q += (size_t)b->h.n_kvrow * 4;
    b->prog = calloc(b->h.n_prog, sizeof(Prog));
    for (uint32_t i = 0; i < b->h.n_prog; i++) {
        Prog* g = &b->prog[i];
        memcpy(&g->h, q, sizeof(PlowProgHeader)); q += sizeof(PlowProgHeader);
        g->insts = (PlowDevInst*)q;    q += (size_t)g->h.n_inst * sizeof(PlowDevInst);
        g->stream = (PlowStreamEnt*)q; q += (size_t)g->h.n_stream * sizeof(PlowStreamEnt);
        g->stream_ofs = (uint32_t*)q;  q += (size_t)b->h.n_cu * 4;
        g->stream_len = (uint32_t*)q;  q += (size_t)b->h.n_cu * 4;
        g->waits = (PlowWait*)q;       q += (size_t)g->h.n_wait * sizeof(PlowWait);
        g->succs = (uint32_t*)q;       q += (size_t)g->h.n_succ * 4;
    }
    return 0;
}

/* ---- safetensors (verbatim from gemma4_chat.c) ---- */
#define MAX_SHARD 8
typedef struct { int n; uint8_t* base[MAX_SHARD]; char* hdr[MAX_SHARD]; size_t hdr_len[MAX_SHARD]; uint64_t data0[MAX_SHARD]; } Safet;
static int st_open(Safet* s, const char* dir) {
    s->n = 0; int total = 0;
    for (int cand = 1; cand <= MAX_SHARD; cand++) { char p[512];
        snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, 1, cand);
        if (access(p, R_OK) == 0) { total = cand; break; } }
    for (int i = 1; total && i <= total; i++) { char p[512];
        snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, i, total);
        int fd = open(p, O_RDONLY); if (fd < 0) break;
        struct stat st; fstat(fd, &st);
        uint8_t* m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd); if (m == MAP_FAILED) return 1;
        uint64_t hn = *(uint64_t*)m;
        s->base[s->n] = m; s->hdr[s->n] = (char*)(m + 8); s->hdr_len[s->n] = (size_t)hn;
        s->data0[s->n] = 8 + hn; s->n++; }
    return s->n ? 0 : 1;
}
static const uint8_t* st_find(Safet* s, const char* name, uint64_t* nb) {
    char key[256]; int kl = snprintf(key, sizeof(key), "\"%s\":", name);
    for (int i = 0; i < s->n; i++) { const char* h = s->hdr[i]; const char* end = h + s->hdr_len[i];
        const char* p = NULL;
        for (const char* c = h; c + kl <= end; c++) if (!memcmp(c, key, (size_t)kl)) { p = c + kl; break; }
        if (!p) continue;
        const char* d = strstr(p, "\"data_offsets\":["); if (!d || d > end) continue;
        d += strlen("\"data_offsets\":[");
        unsigned long long a = strtoull(d, (char**)&d, 10); d++;
        unsigned long long e = strtoull(d, (char**)&d, 10);
        *nb = (uint64_t)(e - a); return s->base[i] + s->data0[i] + a; }
    return NULL;
}

static int find_tensor(Blob* B, const char* name) {
    for (uint32_t i = 0; i < B->h.n_tensor; i++) if (!strcmp(B->tensors[i].name, name)) return (int)i;
    return -1;
}

/* ---- DENSE (layers 0-2) block validation: emitted dense .pkt vs the HF dense fixture (GLM7).
 * Self-contained (own HSA init) so the MoE path below stays pristine. No experts/tables — just the
 * MLA derived weights + block-fp8 dense gate/up/down + scales, all bound individually by name. ---- */
static int run_dense(Blob* B, Safet* S, const char* prep, const char* fixpath, int LAYER) {
    int fd=open(fixpath,O_RDONLY); if(fd<0){ perror(fixpath); return 1; }
    struct stat st; fstat(fd,&st);
    char* base=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); if(base==MAP_FAILED){ perror("mmap"); return 1; }
    int32_t* hdr=(int32_t*)base;
    if(hdr[0]!=0x474C4D37){ printf("bad dense fixture magic %x (want GLM7)\n",hdr[0]); return 1; }
    int L=hdr[1],H=hdr[2],NH=hdr[3],DK=hdr[4],DR=hdr[5],DI=hdr[9],qpos=hdr[10]; (void)NH;
    size_t off=11*4+2*4;
#define DN(cnt,elt) ({ void* _p=base+off; off+=(size_t)(cnt)*(elt); _p; })
    bf16* P_x=DN((size_t)H,2); bf16* P_ckv=DN((size_t)L*DK,2); bf16* P_krot=DN((size_t)L*DR,2);
    bf16* P_attn=DN((size_t)H,2); bf16* P_xn2=DN((size_t)H,2); bf16* P_blk=DN((size_t)H,2);
    float* P_dffn=(float*)DN((size_t)H,4);
    if(off!=(size_t)st.st_size){ printf("dense fixture size mismatch %zu != %ld\n",off,st.st_size); return 1; }
    printf("dense fixture: L=%d H=%d DI=%d qpos=%d layer=%d\n",L,H,DI,qpos,LAYER);

    plow_hsa* h=plow_hsa_init(); if(!h){ printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus=0,lds=0; plow_hsa_device_info(h,0,gfx,&cus,&lds);
    const uint32_t NCU=B->h.n_cu; printf("dev0: %s CUs=%u (pkt n_cu=%u)\n",gfx,cus,NCU);
    const char* elf=getenv("PLOW_INTERP")?getenv("PLOW_INTERP"):"interp_decode.elf";
    FILE* ef=fopen(elf,"rb"); if(!ef){ printf("%s missing\n",elf); return 1; }
    fseek(ef,0,SEEK_END); long co_n=ftell(ef); fseek(ef,0,SEEK_SET);
    void* co=malloc(co_n); if(fread(co,1,co_n,ef)!=(size_t)co_n) return 1; fclose(ef);
    if(plow_hsa_load_code_object(h,0,co,co_n)){ printf("load %s failed\n",elf); return 1; }
    plow_hsa_kernel kern; if(plow_hsa_get_kernel(h,0,"plow_interp_dec_gfx950",&kern)){ printf("no kernel\n"); return 1; }

    const size_t STAGE=64u<<20; void* stage=plow_hsa_alloc_host(h,STAGE);
    void** dev=calloc(B->h.n_tensor+2,sizeof(void*)); int nw=0,nmiss=0; uint64_t wb=0;
    for(uint32_t i=0;i<B->h.n_tensor;i++){ PlowTensorDecl* td=&B->tensors[i];
        int is_model=!strncmp(td->name,"model.",6)||!strncmp(td->name,"lm_head",7);
        if(is_model){ uint64_t got=0; const uint8_t* src=st_find(S,td->name,&got);
            if(!src){ nmiss++; continue; }
            if(got!=td->bytes){ printf("SIZE MISMATCH %s want %llu got %llu\n",td->name,
                                (unsigned long long)td->bytes,(unsigned long long)got); return 1; }
            dev[i]=plow_hsa_alloc(h,0,td->bytes);
            for(uint64_t o=0;o<td->bytes;o+=STAGE){ size_t n=(size_t)((td->bytes-o<STAGE)?(td->bytes-o):STAGE);
                memcpy(stage,src+o,n); plow_hsa_copy_h2d(h,0,(uint8_t*)dev[i]+o,stage,n); }
            wb+=td->bytes; nw++;
        } else { dev[i]=plow_hsa_alloc(h,0,td->bytes);
            if(td->init_off!=PLOW_INIT_NONE)
                for(uint64_t o=0;o<td->bytes;o+=STAGE){ size_t n=(size_t)((td->bytes-o<STAGE)?(td->bytes-o):STAGE);
                    memcpy(stage,B->init+td->init_off+o,n); plow_hsa_copy_h2d(h,0,(uint8_t*)dev[i]+o,stage,n); } }
    }
    printf("bound %d dense weights (%.2f GiB); %d absent globals skipped\n",nw,wb/1073741824.0,nmiss);

    int i_x=find_tensor(B,"act.x"),i_kvlen=find_tensor(B,"in.kvlen"),i_pos=find_tensor(B,"in.pos");
    char cn[80]; snprintf(cn,80,"kv.%d.ckv",LAYER); int i_ckv=find_tensor(B,cn);
    snprintf(cn,80,"kv.%d.krot",LAYER); int i_krot=find_tensor(B,cn);
    int i_xnext=find_tensor(B,"act.xnext"),i_attn=find_tensor(B,"act.attn"),i_xn2=find_tensor(B,"act.xn2"),
        i_sh=find_tensor(B,"act.shared");
    if(i_x<0||i_kvlen<0||i_pos<0||i_ckv<0||i_krot<0||i_xnext<0||i_sh<0){ printf("dense: required tensor missing\n"); return 1; }
    plow_hsa_upload(h,0,dev[i_x],P_x,(size_t)H*2);
    plow_hsa_upload(h,0,dev[i_ckv],P_ckv,(size_t)L*DK*2);
    { bf16* ik=malloc((size_t)L*DR*2); const int half=DR/2;
      for(int r=0;r<L;r++) for(int m=0;m<half;m++){ ik[(size_t)r*DR+2*m]=P_krot[(size_t)r*DR+m];
          ik[(size_t)r*DR+2*m+1]=P_krot[(size_t)r*DR+m+half]; }
      plow_hsa_upload(h,0,dev[i_krot],ik,(size_t)L*DR*2); free(ik); }
    int32_t klen=L; plow_hsa_upload(h,0,dev[i_kvlen],&klen,4);
    int32_t* posbuf=calloc(L,4); posbuf[0]=qpos; plow_hsa_upload(h,0,dev[i_pos],posbuf,(size_t)L*4);

    Prog* g=&B->prog[0];
    int pc=0,pk=0; for(uint32_t k=0;k<g->h.n_inst;k++){ PlowDevInst* d=&g->insts[k];
        if(d->op==PLOW_DOP_RMSNORM && d->t[0]==(uint32_t)i_ckv){ d->i[2]=(uint32_t)qpos; pc++; }
        if(d->op==PLOW_DOP_HEADNORM_ROPE && d->t[0]==(uint32_t)i_krot){ d->i[3]=(uint32_t)qpos; pk++; } }
    if(pc!=1||pk!=1){ printf("dense cache patch ckv=%d krot=%d\n",pc,pk); return 1; }

    void* d_tens=plow_hsa_alloc(h,0,(size_t)(B->h.n_tensor+2)*sizeof(void*));
    plow_hsa_upload(h,0,d_tens,dev,(size_t)(B->h.n_tensor+2)*sizeof(void*));
    g->d_inst=plow_hsa_alloc(h,0,(size_t)g->h.n_inst*sizeof(PlowDevInst));
    g->d_stream=plow_hsa_alloc(h,0,(size_t)g->h.n_stream*sizeof(PlowStreamEnt));
    g->d_sofs=plow_hsa_alloc(h,0,(size_t)NCU*4); g->d_slen=plow_hsa_alloc(h,0,(size_t)NCU*4);
    g->d_waits=plow_hsa_alloc(h,0,(size_t)(g->h.n_wait?g->h.n_wait:1)*sizeof(PlowWait));
    g->d_succs=plow_hsa_alloc(h,0,(size_t)(g->h.n_succ?g->h.n_succ:1)*4);
    g->d_ctr=plow_hsa_alloc(h,0,(size_t)g->h.n_counter*PLOW_CTR_STRIDE*4);
    plow_hsa_upload(h,0,g->d_inst,g->insts,(size_t)g->h.n_inst*sizeof(PlowDevInst));
    plow_hsa_upload(h,0,g->d_stream,g->stream,(size_t)g->h.n_stream*sizeof(PlowStreamEnt));
    plow_hsa_upload(h,0,g->d_sofs,g->stream_ofs,(size_t)NCU*4);
    plow_hsa_upload(h,0,g->d_slen,g->stream_len,(size_t)NCU*4);
    if(g->h.n_wait) plow_hsa_upload(h,0,g->d_waits,g->waits,(size_t)g->h.n_wait*sizeof(PlowWait));
    if(g->h.n_succ) plow_hsa_upload(h,0,g->d_succs,g->succs,(size_t)g->h.n_succ*4);
    uint32_t* zc=calloc((size_t)g->h.n_counter*PLOW_CTR_STRIDE,4);
    plow_hsa_upload(h,0,g->d_ctr,zc,(size_t)g->h.n_counter*PLOW_CTR_STRIDE*4);
    PlowProgram pr; memset(&pr,0,sizeof(pr));
    pr.insts=g->d_inst; pr.stream=g->d_stream; pr.stream_ofs=g->d_sofs; pr.stream_len=g->d_slen;
    pr.waits=g->d_waits; pr.succs=g->d_succs; pr.counters=g->d_ctr; pr.tensors=(void* const*)d_tens;
    if(plow_hsa_launch(h,0,&kern,NCU*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&pr,sizeof(pr))){ printf("LAUNCH FAILED\n"); return 1; }
    plow_hsa_wait(h,0);
    plow_hsa_download(h,0,zc,g->d_ctr,(size_t)g->h.n_counter*PLOW_CTR_STRIDE*4);
    int ctr_ok=1; for(uint32_t k=0;k<g->h.n_inst;k++) if(zc[(size_t)k*PLOW_CTR_STRIDE]!=g->insts[k].blocks) ctr_ok=0;

    bf16 *xnext=malloc(H*2),*attn=malloc(H*2),*xn2=malloc(H*2),*shd=malloc(H*2);
    plow_hsa_download(h,0,xnext,dev[i_xnext],H*2);
    if(i_attn>=0) plow_hsa_download(h,0,attn,dev[i_attn],H*2);
    if(i_xn2>=0)  plow_hsa_download(h,0,xn2,dev[i_xn2],H*2);
    plow_hsa_download(h,0,shd,dev[i_sh],H*2);        /* dense FFN output (reused n.shared) */
    float* dffn=malloc((size_t)H*4); for(int i=0;i<H;i++) dffn[i]=b2f(shd[i]);
    printf("\n[emitted dense pkt] %u ops, executed==total: %s\n",g->h.n_inst,ctr_ok?"YES":"NO");
    double w1,w2,w3,w4; int ok=ctr_ok;
    if(i_attn>=0){ double r=relerr(attn,P_attn,H,&w1); printf("  MLA  attn_out    : rms=%.5f max=%.4f\n",r,w1); ok&=r<2e-2; }
    if(i_xn2>=0){ double r=relerr(xn2,P_xn2,H,&w2); printf("  xn2  (post_ln)   : rms=%.5f max=%.4f\n",r,w2); ok&=r<2e-2; }
    double r_dffn=relerr_f32(dffn,P_dffn,H,&w3); printf("  >> DENSE_FFN f32 : rms=%.5f max=%.4f  <<< op47/op44 block-fp8 de-risk\n",r_dffn,w3); ok&=r_dffn<6e-2;
    double r_next=relerr(xnext,P_blk,H,&w4); printf("  x_next (BLOCK)   : rms=%.5f max=%.4f\n",r_next,w4); ok&=r_next<2e-2;
    printf("\n%s\n", ok?"GLM52 B3 DENSE OK — emitted dense .pkt + prepped real weights match HF"
                      : "*** GLM52 B3 DENSE MISMATCH ***");
    munmap(base,st.st_size); plow_hsa_shutdown(h);
    return ok?0:1;
#undef DN
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s model.pkt <prep-dir> <fixture.bin> [layer]\n", argv[0]); return 1; }
    const int LAYER = argc > 4 ? atoi(argv[4]) : 3;
    setbuf(stdout, NULL);
    Blob B; if (load_blob(argv[1], &B)) return 1;
    Safet S; if (st_open(&S, argv[2])) { printf("no safetensors in %s\n", argv[2]); return 1; }
    /* Dense (layers 0-2) pkt? -> the self-contained dense validation path (op 47 present). */
    for (uint32_t k=0;k<B.prog[0].h.n_inst;k++)
        if (B.prog[0].insts[k].op==PLOW_DOP_DENSE_GLU_FP8_BLK) return run_dense(&B,&S,argv[2],argv[3],LAYER);
    /* pkt arm: block-fp8 experts (production) vs bf16 experts (isolation). The prep dir keeps experts
     * fp8; a bf16 pkt needs them host-dequantised (as the B4 harness did). */
    int use_fp8=0; for(uint32_t k=0;k<B.prog[0].h.n_inst;k++)
        if(B.prog[0].insts[k].op==PLOW_DOP_MOE_EXPERT_GLU_FP8_BLK){ use_fp8=1; break; }
    printf("pkt experts: %s\n", use_fp8?"block-fp8 (45/46)":"bf16 (41/42, host-dequant)");

    /* ---- fixture: dims + input hidden + full [L] cache + diff refs (B4 layout) ---- */
    int fd = open(argv[3], O_RDONLY); if (fd < 0) { perror(argv[3]); return 1; }
    struct stat st; fstat(fd, &st);
    char* base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    int32_t* hdr = (int32_t*)base;
    if (hdr[0] != 0x474C4D36) { printf("bad fixture magic %x\n", hdr[0]); return 1; }
    int L=hdr[1],H=hdr[2],NH=hdr[3],DK=hdr[4],DR=hdr[5],QN=hdr[6],VD=hdr[7],QL=hdr[8],E=hdr[9],TOPK=hdr[10],IMOE=hdr[11],qpos=hdr[12];
    float* fh=(float*)(base+13*4); float EPS=fh[0],SCALE=fh[1]; (void)EPS;(void)SCALE;(void)QN;(void)VD;(void)QL;
    const int IB=(IMOE+127)/128, HB=(H+127)/128;
    size_t off=13*4+3*4;
#define NEXT(cnt,elt) ({ void* _p=base+off; off+=(size_t)(cnt)*(elt); _p; })
    bf16* P_x=NEXT((size_t)H,2); NEXT((size_t)H,2); NEXT((size_t)QL*H,2); NEXT((size_t)QL,2);
    NEXT((size_t)NH*DK*QL,2); NEXT((size_t)NH*DR*QL,2); NEXT((size_t)DK*H,2); NEXT((size_t)DK,2);
    NEXT((size_t)DR*H,2); NEXT((size_t)NH*DK*VD,2); NEXT((size_t)H*NH*VD,2); NEXT((size_t)H,2);
    bf16* P_ckv=NEXT((size_t)L*DK,2); bf16* P_krot=NEXT((size_t)L*DR,2);
    NEXT((size_t)E*H,2); NEXT((size_t)E,4);
    NEXT((size_t)E*3*IMOE*H,1); NEXT((size_t)E*3*IB*HB,4);
    NEXT((size_t)IMOE*H,2); NEXT((size_t)IMOE*H,2); NEXT((size_t)H*IMOE,2);
    bf16* P_blk=NEXT((size_t)H,2); int32_t* P_sel=NEXT((size_t)TOPK,4); float* P_selg=NEXT((size_t)TOPK,4);
    bf16* P_attn=NEXT((size_t)H,2); bf16* P_xn2=NEXT((size_t)H,2); NEXT((size_t)H,2);
    float* P_esum=(float*)NEXT((size_t)H,4); bf16* P_shref=NEXT((size_t)H,2);
    if (off != (size_t)st.st_size) { printf("FIXTURE SIZE MISMATCH parsed %zu != %ld\n", off, st.st_size); return 1; }
    printf("fixture: L=%d H=%d NH=%d E=%d TOPK=%d IMOE=%d qpos=%d layer=%d\n", L,H,NH,E,TOPK,IMOE,qpos,LAYER);

    /* ---- HSA + interpreter (one object: bf16 MLA/MoE ops + block-fp8 experts 45/46) ---- */
    plow_hsa* h = plow_hsa_init(); if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus=0,lds=0; plow_hsa_device_info(h,0,gfx,&cus,&lds);
    const uint32_t NCU = B.h.n_cu; printf("dev0: %s CUs=%u  (pkt n_cu=%u)\n", gfx, cus, NCU);
    const char* elf = getenv("PLOW_INTERP") ? getenv("PLOW_INTERP") : "interp_decode.elf";
    FILE* ef=fopen(elf,"rb"); if(!ef){ printf("%s missing\n",elf); return 1; }
    fseek(ef,0,SEEK_END); long co_n=ftell(ef); fseek(ef,0,SEEK_SET);
    void* co=malloc(co_n); if(fread(co,1,co_n,ef)!=(size_t)co_n) return 1; fclose(ef);
    if(plow_hsa_load_code_object(h,0,co,co_n)){ printf("load %s failed\n",elf); return 1; }
    plow_hsa_kernel kern; if(plow_hsa_get_kernel(h,0,"plow_interp_dec_gfx950",&kern)){ printf("no kernel\n"); return 1; }

    /* ---- bind. Allocate a device buffer only for tensors the block actually uses:
     *   - the 256*3 block-fp8 experts + scale grids go into TWO packed buffers (like the B4 harness),
     *     never 1536 separate allocations, and the expert tables point into them;
     *   - other model.* weights (norms/derived/o_proj/router/shared) bind individually from the dir;
     *   - unreferenced absent globals (embed/norm/lm_head, declared but not used by ONE layer) skipped.
     * The interpreter only ever dereferences a tensor HANDLE that an instruction names, and the per-
     * expert handles are named nowhere (the ops index the tables) — so leaving dev[expert]=NULL is safe. */
    const size_t STAGE = 64u<<20;
    void* stage = plow_hsa_alloc_host(h, STAGE);
    void** dev = calloc(B.h.n_tensor + 2, sizeof(void*));  /* +2 for the cache-row aliases */
    uint64_t wb=0; int nw=0, nmiss=0;
    char pfx[80]; snprintf(pfx, sizeof(pfx), "model.layers.%d.", LAYER);
    size_t plen=strlen(pfx);
    for (uint32_t i=0;i<B.h.n_tensor;i++) {
        PlowTensorDecl* td=&B.tensors[i];
        if(strstr(td->name,"mlp.experts.")) continue;   /* packed below; no individual alloc */
        int is_table = strstr(td->name,"expert_weight_table") || strstr(td->name,"expert_scale_table");
        int is_model = (!strncmp(td->name,"model.",6) || !strncmp(td->name,"lm_head",7)) && !is_table;
        if(is_table){ dev[i]=plow_hsa_alloc(h,0,td->bytes);   /* loader-filled below, not on disk */
            if(!dev[i]){ printf("VRAM alloc failed %s\n",td->name); return 1; } }
        else if(is_model){ uint64_t got=0; const uint8_t* src=st_find(&S,td->name,&got);
            if(!src){ nmiss++; continue; }               /* absent unreferenced global -> dev=NULL */
            if(got!=td->bytes){ printf("SIZE MISMATCH %s want %llu got %llu\n",td->name,
                                (unsigned long long)td->bytes,(unsigned long long)got); return 1; }
            dev[i]=plow_hsa_alloc(h,0,td->bytes);
            if(!dev[i]){ printf("VRAM alloc failed %s\n",td->name); return 1; }
            for(uint64_t o=0;o<td->bytes;o+=STAGE){ size_t n=(size_t)((td->bytes-o<STAGE)?(td->bytes-o):STAGE);
                memcpy(stage,src+o,n); plow_hsa_copy_h2d(h,0,(uint8_t*)dev[i]+o,stage,n); }
            wb+=td->bytes; nw++;
        } else {
            dev[i]=plow_hsa_alloc(h,0,td->bytes);        /* act.* / in.* / kv.* — referenced */
            if(!dev[i]){ printf("VRAM alloc failed %s\n",td->name); return 1; }
            /* init-data tensors (the emitter's rope cos/sin tables) live in the blob's init section. */
            if(td->init_off!=PLOW_INIT_NONE)
                for(uint64_t o=0;o<td->bytes;o+=STAGE){ size_t n=(size_t)((td->bytes-o<STAGE)?(td->bytes-o):STAGE);
                    memcpy(stage,B.init+td->init_off+o,n); plow_hsa_copy_h2d(h,0,(uint8_t*)dev[i]+o,stage,n); }
        }
    }

    /* ---- expert tables: pack all 256 experts into two buffers, fill tables (loader glue #1) ---- */
    const size_t EW=(size_t)IMOE*H, ESC=(size_t)IB*HB;   /* fp8 bytes / f32 scale count per proj */
    void* d_efp8=plow_hsa_alloc(h,0,(size_t)E*3*EW);
    void* d_esc =plow_hsa_alloc(h,0,(size_t)E*3*ESC*4);
    if(!d_efp8||!d_esc){ printf("expert buffer alloc failed (%.2f GB)\n",(double)E*3*EW/1e9); return 1; }
    uint64_t* wtb=malloc((size_t)E*3*8); uint64_t* stb=malloc((size_t)E*3*8);
    const char* projs[3]={"gate_proj","up_proj","down_proj"};
    for(int e=0;e<E;e++) for(int j=0;j<3;j++){ char nm[96]; uint64_t gw=0,gs=0;
        snprintf(nm,sizeof(nm),"%smlp.experts.%d.%s.weight",pfx,e,projs[j]);
        const uint8_t* ws=st_find(&S,nm,&gw);
        snprintf(nm,sizeof(nm),"%smlp.experts.%d.%s.weight_scale_inv",pfx,e,projs[j]);
        const uint8_t* ss=st_find(&S,nm,&gs);
        if(!ws||!ss||gw!=EW||gs!=ESC*4){ printf("expert %d %s bad (%llu/%llu)\n",e,projs[j],
            (unsigned long long)gw,(unsigned long long)gs); return 1; }
        void* wdst=(char*)d_efp8+((size_t)e*3+j)*EW; void* sdst=(char*)d_esc+((size_t)e*3+j)*ESC*4;
        memcpy(stage,ws,EW); plow_hsa_copy_h2d(h,0,wdst,stage,EW);
        memcpy(stage,ss,ESC*4); plow_hsa_copy_h2d(h,0,sdst,stage,ESC*4);
        wtb[e*3+j]=(uint64_t)(uintptr_t)wdst; stb[e*3+j]=(uint64_t)(uintptr_t)sdst;
        wb+=EW+ESC*4; }
    nw+=E*3;
    /* bf16 pkt: MoeExpertGlu/Down read bf16 weights, so host-dequantise the fp8+scales into a bf16
     * buffer and repoint wtb (mirrors the B4 harness d_ebf path). est is unused by the bf16 ops. */
    if(!use_fp8){
        void* d_ebf=plow_hsa_alloc(h,0,(size_t)E*3*EW*2);
        if(!d_ebf){ printf("bf16 expert buffer alloc failed\n"); return 1; }
        bf16* scr=malloc(3*EW*2);
        for(int e=0;e<E;e++){ char nm[96];
            for(int j=0;j<3;j++){ uint64_t gw=0,gs=0;
                snprintf(nm,sizeof(nm),"%smlp.experts.%d.%s.weight",pfx,e,projs[j]);
                const unsigned char* Wq=st_find(&S,nm,&gw);
                snprintf(nm,sizeof(nm),"%smlp.experts.%d.%s.weight_scale_inv",pfx,e,projs[j]);
                const float* Sc=(const float*)st_find(&S,nm,&gs);
                bf16* dst=scr+(size_t)j*EW;
                if(j<2)  /* gate/up [IMOE,H], scale [IB,HB] */
                    for(int nn=0;nn<IMOE;nn++) for(int kk=0;kk<H;kk++)
                        dst[(size_t)nn*H+kk]=f2b(e4m3_decode(Wq[(size_t)nn*H+kk])*Sc[(nn>>7)*HB+(kk>>7)]);
                else     /* down [H,IMOE], scale [HB,IB] */
                    for(int nn=0;nn<H;nn++) for(int kk=0;kk<IMOE;kk++)
                        dst[(size_t)nn*IMOE+kk]=f2b(e4m3_decode(Wq[(size_t)nn*IMOE+kk])*Sc[(nn>>7)*IB+(kk>>7)]); }
            for(int j=0;j<3;j++){ void* d=(char*)d_ebf+((size_t)e*3+j)*EW*2;  /* stage per proj (EW*2=25MB<STAGE) */
                memcpy(stage,scr+(size_t)j*EW,EW*2); plow_hsa_copy_h2d(h,0,d,stage,EW*2);
                wtb[e*3+j]=(uint64_t)(uintptr_t)d; } }
        free(scr);
    }
    char tn[96]; snprintf(tn,sizeof(tn),"%smlp.expert_weight_table",pfx); int i_ewt=find_tensor(&B,tn);
    snprintf(tn,sizeof(tn),"%smlp.expert_scale_table",pfx); int i_est=find_tensor(&B,tn);
    if(i_ewt<0||i_est<0||!dev[i_ewt]||!dev[i_est]){ printf("expert tables not declared/allocated\n"); return 1; }
    plow_hsa_upload(h,0,dev[i_ewt],wtb,(size_t)E*3*8);
    plow_hsa_upload(h,0,dev[i_est],stb,(size_t)E*3*8);
    printf("bound %d weights (%.2f GiB) from %s; %d unreferenced globals skipped\n",
           nw, wb/1073741824.0, argv[2], nmiss);
    (void)plen;

    /* ---- input + full cache; kvlen (loader glue #3) ---- */
    int i_x=find_tensor(&B,"act.x"), i_kvlen=find_tensor(&B,"in.kvlen");
    char cn[80]; snprintf(cn,80,"kv.%d.ckv",LAYER); int i_ckv=find_tensor(&B,cn);
    snprintf(cn,80,"kv.%d.krot",LAYER); int i_krot=find_tensor(&B,cn);
    int i_xnext=find_tensor(&B,"act.xnext"), i_attn=find_tensor(&B,"act.attn"),
        i_xn2=find_tensor(&B,"act.xn2"), i_tab=find_tensor(&B,"act.tab"), i_part=find_tensor(&B,"act.part");
    int i_pos=find_tensor(&B,"in.pos");
    if(i_x<0||i_kvlen<0||i_ckv<0||i_krot<0||i_xnext<0||i_part<0||i_pos<0){ printf("required act/kv/in tensor missing\n"); return 1; }
    plow_hsa_upload(h,0,dev[i_x],P_x,(size_t)H*2);
    plow_hsa_upload(h,0,dev[i_ckv],P_ckv,(size_t)L*DK*2);
    /* k_rope cache layout: HF apply_rotary_pos_emb_interleave writes HALF-SPLIT output (cat of the
     * rotated even/odd slices) — that is what the fixture stores. plow's interleaved HeadNormRope
     * kernel writes INTERLEAVED-POSITION output (out[2m]=hf[m], out[2m+1]=hf[m+DR/2]). The flash dot
     * q_rope·k_rope is permutation-invariant, so a full plow decode (all k kernel-written) is correct
     * in its own basis; but this test mixes the kernel's current-token k with the fixture's history,
     * so permute the fixture history to the SAME interleaved-position layout the kernel produces. */
    { bf16* ik=malloc((size_t)L*DR*2); const int half=DR/2;
      for(int r=0;r<L;r++) for(int m=0;m<half;m++){ ik[(size_t)r*DR+2*m]=P_krot[(size_t)r*DR+m];
          ik[(size_t)r*DR+2*m+1]=P_krot[(size_t)r*DR+m+half]; }
      plow_hsa_upload(h,0,dev[i_krot],ik,(size_t)L*DR*2); free(ik); }
    int32_t klen=L; plow_hsa_upload(h,0,dev[i_kvlen],&klen,4);
    int32_t* posbuf=calloc(L,4); posbuf[0]=qpos;   /* decode t=0 is the current token at qpos */
    plow_hsa_upload(h,0,dev[i_pos],posbuf,(size_t)L*4);

    /* ---- cache-row write (loader glue #2). Both writers take an out_row0 immediate = the current
     * position: ckv RMSNORM i[2], k_rope HeadNormRope i[3]. Patched per step in the decode loop; here
     * set to qpos. FLASH reads the full [0,L) caches (pre-populated from the fixture). ---- */
    Prog* g=&B.prog[0];
    int patched_ckv=0, patched_krot=0;
    for(uint32_t k=0;k<g->h.n_inst;k++){ PlowDevInst* d=&g->insts[k];
        if(d->op==PLOW_DOP_RMSNORM && d->t[0]==(uint32_t)i_ckv){ d->i[2]=(uint32_t)qpos; patched_ckv++; }
        if(d->op==PLOW_DOP_HEADNORM_ROPE && d->t[0]==(uint32_t)i_krot){ d->i[3]=(uint32_t)qpos; patched_krot++; } }
    if(patched_ckv!=1||patched_krot!=1){ printf("cache-row patch: ckv=%d krot=%d (want 1/1)\n",patched_ckv,patched_krot); return 1; }
    printf("cache-row write set to qpos=%d (ckv RMSNORM out_row0 i[2] + k_rope HeadNormRope out_row0 i[3])\n",qpos);

    void* d_tens=plow_hsa_alloc(h,0,(size_t)(B.h.n_tensor+2)*sizeof(void*));
    plow_hsa_upload(h,0,d_tens,dev,(size_t)(B.h.n_tensor+2)*sizeof(void*));

    /* ---- upload the program tables + launch (static per-CU stream, single launch) ---- */
    g->d_inst=plow_hsa_alloc(h,0,(size_t)g->h.n_inst*sizeof(PlowDevInst));
    g->d_stream=plow_hsa_alloc(h,0,(size_t)g->h.n_stream*sizeof(PlowStreamEnt));
    g->d_sofs=plow_hsa_alloc(h,0,(size_t)NCU*4); g->d_slen=plow_hsa_alloc(h,0,(size_t)NCU*4);
    g->d_waits=plow_hsa_alloc(h,0,(size_t)(g->h.n_wait?g->h.n_wait:1)*sizeof(PlowWait));
    g->d_succs=plow_hsa_alloc(h,0,(size_t)(g->h.n_succ?g->h.n_succ:1)*4);
    g->d_ctr=plow_hsa_alloc(h,0,(size_t)g->h.n_counter*PLOW_CTR_STRIDE*4);
    plow_hsa_upload(h,0,g->d_inst,g->insts,(size_t)g->h.n_inst*sizeof(PlowDevInst));
    plow_hsa_upload(h,0,g->d_stream,g->stream,(size_t)g->h.n_stream*sizeof(PlowStreamEnt));
    plow_hsa_upload(h,0,g->d_sofs,g->stream_ofs,(size_t)NCU*4);
    plow_hsa_upload(h,0,g->d_slen,g->stream_len,(size_t)NCU*4);
    if(g->h.n_wait) plow_hsa_upload(h,0,g->d_waits,g->waits,(size_t)g->h.n_wait*sizeof(PlowWait));
    if(g->h.n_succ) plow_hsa_upload(h,0,g->d_succs,g->succs,(size_t)g->h.n_succ*4);
    uint32_t* zc=calloc((size_t)g->h.n_counter*PLOW_CTR_STRIDE,4);
    plow_hsa_upload(h,0,g->d_ctr,zc,(size_t)g->h.n_counter*PLOW_CTR_STRIDE*4);

    PlowProgram pr; memset(&pr,0,sizeof(pr));
    pr.insts=g->d_inst; pr.stream=g->d_stream; pr.stream_ofs=g->d_sofs; pr.stream_len=g->d_slen;
    pr.waits=g->d_waits; pr.succs=g->d_succs; pr.counters=g->d_ctr; pr.tensors=(void* const*)d_tens;
    if(plow_hsa_launch(h,0,&kern,NCU*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&pr,sizeof(pr))){ printf("LAUNCH FAILED\n"); return 1; }
    plow_hsa_wait(h,0);

    /* ---- counters + downloads + diff ---- */
    plow_hsa_download(h,0,zc,g->d_ctr,(size_t)g->h.n_counter*PLOW_CTR_STRIDE*4);
    int ctr_ok=1; for(uint32_t k=0;k<g->h.n_inst;k++) if(zc[(size_t)k*PLOW_CTR_STRIDE]!=g->insts[k].blocks) ctr_ok=0;
    bf16 *xnext=malloc(H*2),*attn=malloc(H*2),*xn2=malloc(H*2),*shared=NULL;
    unsigned char* tab=malloc((size_t)TOPK*8); float* part=malloc((size_t)TOPK*H*4);
    plow_hsa_download(h,0,xnext,dev[i_xnext],H*2);
    if(i_attn>=0) plow_hsa_download(h,0,attn,dev[i_attn],H*2);
    if(i_xn2>=0)  plow_hsa_download(h,0,xn2,dev[i_xn2],H*2);
    if(i_tab>=0)  plow_hsa_download(h,0,tab,dev[i_tab],(size_t)TOPK*8);
    plow_hsa_download(h,0,part,dev[i_part],(size_t)TOPK*H*4);
    int i_sh=find_tensor(&B,"act.shared"); if(i_sh>=0){ shared=malloc(H*2); plow_hsa_download(h,0,shared,dev[i_sh],H*2); }

    printf("\n[emitted pkt] %u ops, executed==total: %s\n", g->h.n_inst, ctr_ok?"YES":"NO");
    printf("HF router pick (fp32): "); for(int j=0;j<TOPK;j++) printf("e%d(g=%.4f) ",P_sel[j],P_selg[j]); printf("\n");
    if(i_tab>=0){ printf("plow router pick     : ");
        int set_ok=1; for(int j=0;j<TOPK;j++){ unsigned id=*(unsigned*)(tab+j*8); float gg=*(float*)(tab+j*8+4);
            printf("e%u(g=%.4f) ",id,gg); int found=0; for(int k=0;k<TOPK;k++) if((int)id==P_sel[k]) found=1; if(!found) set_ok=0; }
        printf("\n  top-%d SET vs HF: %s\n",TOPK,set_ok?"MATCH":"*** DIFFERS ***");
    }
    float* esum=malloc((size_t)H*4);
    for(int i=0;i<H;i++){ double a=0; for(int s=0;s<TOPK;s++) a+=part[(size_t)s*H+i]; esum[i]=(float)a; }
    double w1,w2,w3,w4,w5; int ok=ctr_ok;
    if(i_attn>=0){ double r=relerr(attn,P_attn,H,&w1); printf("  MLA  attn_out    : rms=%.5f max=%.4f\n",r,w1); ok&=r<2e-2; }
    if(i_xn2>=0){ double r=relerr(xn2,P_xn2,H,&w2); printf("  xn2  (post_ln)   : rms=%.5f max=%.4f\n",r,w2); ok&=r<2e-2; }
    if(shared){ double r=relerr(shared,P_shref,H,&w5); printf("  shared_expert    : rms=%.5f max=%.4f\n",r,w5); ok&=r<2e-2; }
    double r_esum=relerr_f32(esum,P_esum,H,&w3); printf("  >> EXPERT_SUM f32: rms=%.5f max=%.4f\n",r_esum,w3); ok&=r_esum<6e-2;
    double r_next=relerr(xnext,P_blk,H,&w4); printf("  x_next (BLOCK)   : rms=%.5f max=%.4f (residual-dominated)\n",r_next,w4);

    printf("\n%s\n", ok ? "GLM52 ms1 CLOSED — emitted .pkt + prepped real weights match HF single-layer"
                        : "*** GLM52 ms1 MISMATCH ***");
    munmap(base,st.st_size); plow_hsa_shutdown(h);
    return ok?0:1;
}
