/* glm52_decode.c — GLM-5.2-FP8 full-model DECODE driver (serving path).  [GLM52-B5 + TP4]
 *
 *   glm52_decode <model.pkt> <weight-dir> [prompt.ids] [--steps M] [--sweep 1k,4k,..] [--tp N] [--gen G]
 *
 * Loads the emitted full 78-layer decode .pkt (gemma4.rs glm_emit_full, GLM_FULL=1), binds the whole
 * model BY NAME from the plow-ready weight dir (79 shards + model.safetensors.index.json), and runs a
 * KV-cached decode: prefill the prompt then greedily generate. Each step the interpreter appends the
 * current token's latent/rope to every layer's KV cache at the sequence position — patched per step
 * via the ckv RMSNorm out_row0 (i[2]) and the k_rope HeadNormRope out_row0 (i[3]).
 *
 * MULTI-GPU (--tp N): mirrors runtime/tests/tp_decode.c. The .pkt must be emitted with `--tp N` (the
 * emitter shards the MLA head-projections, dense FFN and the 256 experts column/row-parallel and
 * inserts an XReduce all-reduce after o_proj and after the FFN down/combine). This host loads each
 * rank's 1/N WEIGHT SLICE (column-parallel = contiguous output-row byte range; row-parallel = strided
 * input-column gather; both fp8 weight + f32 block-scale sharded), packs each layer's sharded experts,
 * peer-maps the reduction region (og_tp @ 0, dg_tp @ slot_b, xctr @ 2*slot_b), and fans the token step
 * over dev = 0..N-1 with the XReduce rendezvous inline in each rank's megakernel. lm_head is REPLICATED
 * (full vocab) so every rank argmaxes the same id — no cross-rank fold. --tp 1 is byte-identical to B5.
 *
 * --sweep matches tp_decode.c's format (SWEEP rows: ctx / ms/tok / tok/s).
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
#include <time.h>
#include <unistd.h>

#define MAX_DEV 8

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
typedef uint16_t bf16;
static float b2f(bf16 v){ union{uint32_t u;float f;}c; c.u=(uint32_t)v<<16; return c.f; }
static int cmp_dbl(const void* a,const void* b){ double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }

/* ---- blob (from gemma4_chat.c/glm52_run.c) ---- */
typedef struct {
    PlowProgHeader h; PlowDevInst* insts; PlowStreamEnt* stream;
    uint32_t *stream_ofs,*stream_len,*succs; PlowWait* waits;
} Prog;
typedef struct { PlowBlobHeader h; PlowTensorDecl* tensors; uint8_t* init; uint32_t* kvrow; Prog* prog; } Blob;
static int load_blob(const char* path, Blob* b){
    FILE* f=fopen(path,"rb"); if(!f){ printf("cannot open %s\n",path); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t* p=malloc((size_t)n); if(fread(p,1,(size_t)n,f)!=(size_t)n) return 1; fclose(f);
    memcpy(&b->h,p,sizeof(PlowBlobHeader));
    { const char* e = plow_blob_magic_error(b->h.magic);
      if (e) { printf("%s\n", e); return 1; } }
    uint8_t* q=p+sizeof(PlowBlobHeader);
    b->tensors=(PlowTensorDecl*)q; q+=(size_t)b->h.n_tensor*sizeof(PlowTensorDecl);
    b->init=q; q+=b->h.init_bytes; b->kvrow=(uint32_t*)q; q+=(size_t)b->h.n_kvrow*4;
    b->prog=calloc(b->h.n_prog,sizeof(Prog));
    for(uint32_t i=0;i<b->h.n_prog;i++){ Prog* g=&b->prog[i];
        memcpy(&g->h,q,sizeof(PlowProgHeader)); q+=sizeof(PlowProgHeader);
        g->insts=(PlowDevInst*)q; q+=(size_t)g->h.n_inst*sizeof(PlowDevInst);
        g->stream=(PlowStreamEnt*)q; q+=(size_t)g->h.n_stream*sizeof(PlowStreamEnt);
        g->stream_ofs=(uint32_t*)q; q+=(size_t)b->h.n_cu*4;
        g->stream_len=(uint32_t*)q; q+=(size_t)b->h.n_cu*4;
        g->waits=(PlowWait*)q; q+=(size_t)g->h.n_wait*sizeof(PlowWait);
        g->succs=(uint32_t*)q; q+=(size_t)g->h.n_succ*4; }
    return 0;
}

/* ---- safetensors: mmap all shards (naming model-%05d-of-%05d per index.json) ----
 * name->shard hash index: st_find on a 116454-tensor / 79-shard model is called ~115k times per rank
 * (every expert's gate/up/down weight+scale). A linear header scan per lookup is O(tensors*lookups)
 * and never finishes; the index makes each lookup scan ONE shard's header instead of all 79. */
/* 79 base shards + up to 78 append-only `model-idx-*` shards (the DSA indexer prep and the
 * block-fp8 linear prep both add one per layer) already exceeds 128, and the overflow is SILENT:
 * st_open just stops mmapping and the bind loop reports MISSING WEIGHT for the tail layers. */
#define MAX_SHARD 256
typedef struct { int n; uint8_t* base[MAX_SHARD]; char* hdr[MAX_SHARD]; size_t hdr_len[MAX_SHARD]; uint64_t data0[MAX_SHARD];
    uint32_t icap; const char** iname; uint32_t* inlen; int8_t* ishard; } Safet;
static uint64_t st_hash(const char* s, int n){ uint64_t h=1469598103934665603ull; for(int i=0;i<n;i++){ h^=(uint8_t)s[i]; h*=1099511628211ull; } return h; }
static void st_index_put(Safet* s, const char* name, int nlen, int shard){
    uint32_t m=s->icap-1, p=(uint32_t)st_hash(name,nlen)&m;
    while(s->iname[p]){ p=(p+1)&m; }
    s->iname[p]=name; s->inlen[p]=(uint32_t)nlen; s->ishard[p]=(int8_t)shard;
}
static int st_index_get(Safet* s, const char* name){   /* -1 if absent */
    int nlen=(int)strlen(name); uint32_t m=s->icap-1, p=(uint32_t)st_hash(name,nlen)&m;
    while(s->iname[p]){ if((int)s->inlen[p]==nlen && !memcmp(s->iname[p],name,nlen)) return s->ishard[p]; p=(p+1)&m; }
    return -1;
}
/* Parse one shard's JSON header, inserting each top-level tensor key -> shard into the index. */
static void st_index_shard(Safet* s, int shard){
    const char* p=s->hdr[shard]; const char* end=p+s->hdr_len[shard];
    while(p<end && *p!='{') p++; if(p<end) p++;
    while(p<end){
        while(p<end && (*p==' '||*p==','||*p=='\n'||*p=='\t'||*p=='\r')) p++;
        if(p>=end || *p=='}') break;
        if(*p!='"') break;
        const char* k=++p; while(p<end && *p!='"') p++; int klen=(int)(p-k); if(p<end) p++;
        while(p<end && (*p==' '||*p==':')) p++;
        if(p<end && *p=='{'){
            if(!(klen==12 && !memcmp(k,"__metadata__",12))) st_index_put(s,k,klen,shard);
            int d=0; do{ if(*p=='{') d++; else if(*p=='}') d--; p++; }while(p<end && d>0);
        }
    }
}
static int st_open(Safet* s, const char* dir){
    s->n=0; int total=0;
    for(int cand=1;cand<=MAX_SHARD;cand++){ char p[512];
        snprintf(p,sizeof(p),"%s/model-%05d-of-%05d.safetensors",dir,1,cand);
        if(access(p,R_OK)==0){ total=cand; break; } }
    if(!total){ char p[512]; snprintf(p,sizeof(p),"%s/model.safetensors",dir);   /* single-file fallback */
        if(access(p,R_OK)==0) total=-1; }
    if(total==-1){ char p[512]; snprintf(p,sizeof(p),"%s/model.safetensors",dir);
        int fd=open(p,O_RDONLY); if(fd<0) return 1; struct stat st; fstat(fd,&st);
        uint8_t* m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
        if(m==MAP_FAILED) return 1; uint64_t hn=*(uint64_t*)m;
        s->base[0]=m; s->hdr[0]=(char*)(m+8); s->hdr_len[0]=(size_t)hn; s->data0[0]=8+hn; s->n=1; return 0; }
    for(int i=1;total&&i<=total;i++){ char p[512];
        snprintf(p,sizeof(p),"%s/model-%05d-of-%05d.safetensors",dir,i,total);
        int fd=open(p,O_RDONLY); if(fd<0) break; struct stat st; fstat(fd,&st);
        uint8_t* m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
        if(m==MAP_FAILED) return 1; uint64_t hn=*(uint64_t*)m;
        s->base[s->n]=m; s->hdr[s->n]=(char*)(m+8); s->hdr_len[s->n]=(size_t)hn; s->data0[s->n]=8+hn; s->n++; }
    /* DSA: also load the incremental indexer shards (model-idx-{layer:05d}-of-idx.safetensors), added
     * append-only by scripts/glm52_prep_indexer.py without recopying the base weights. [GLM52-DSA] */
    for(int L=0;L<256 && s->n<MAX_SHARD;L++){ char p[512];
        snprintf(p,sizeof(p),"%s/model-idx-%05d-of-idx.safetensors",dir,L);
        int fd=open(p,O_RDONLY); if(fd<0) continue; struct stat st; fstat(fd,&st);
        uint8_t* m=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
        if(m==MAP_FAILED) continue; uint64_t hn=*(uint64_t*)m;
        s->base[s->n]=m; s->hdr[s->n]=(char*)(m+8); s->hdr_len[s->n]=(size_t)hn; s->data0[s->n]=8+hn; s->n++; }
    if(!s->n) return 1;
    /* build the name->shard index (over-provision to keep the open-addressing load factor low) */
    s->icap=1u<<20;
    s->iname=calloc(s->icap,sizeof(char*)); s->inlen=calloc(s->icap,4); s->ishard=malloc(s->icap);
    for(int i=0;i<s->n;i++) st_index_shard(s,i);
    return 0;
}
/* Find a tensor; return data ptr + byte count. If shape/ndim non-NULL, parse up to 2 leading dims
 * (row-parallel gather needs out=shape[0], in=shape[1]). O(1) shard lookup via the name index. */
static const uint8_t* st_find_ex(Safet* s, const char* name, uint64_t* nb, uint64_t shape[2], int* ndim){
    char key[256]; int kl=snprintf(key,sizeof(key),"\"%s\":",name);
    int sh=st_index_get(s,name); int i0=sh>=0?sh:0, i1=sh>=0?sh+1:s->n;
    for(int i=i0;i<i1;i++){ const char* h=s->hdr[i]; const char* end=h+s->hdr_len[i]; const char* p=NULL;
        for(const char* c=h;c+kl<=end;c++) if(!memcmp(c,key,(size_t)kl)){ p=c+kl; break; }
        if(!p) continue;
        if(shape&&ndim){ *ndim=0; shape[0]=shape[1]=0;
            const char* sh=strstr(p,"\"shape\":["); if(sh&&sh<end){ sh+=strlen("\"shape\":[");
                while(*sh!=']'&&sh<end&&*ndim<2){ while(*sh==' '||*sh==',') sh++; if(*sh==']') break;
                    shape[*ndim]=strtoull(sh,(char**)&sh,10); (*ndim)++; }
                while(*sh!=']'&&sh<end){ if(*sh==',') (*ndim)++; sh++; } } }
        const char* d=strstr(p,"\"data_offsets\":["); if(!d||d>end) continue;
        d+=strlen("\"data_offsets\":[");
        unsigned long long a=strtoull(d,(char**)&d,10); d++; unsigned long long e=strtoull(d,(char**)&d,10);
        *nb=(uint64_t)(e-a); return s->base[i]+s->data0[i]+a; }
    return NULL;
}
static const uint8_t* st_find(Safet* s, const char* name, uint64_t* nb){ return st_find_ex(s,name,nb,NULL,NULL); }
static int find_tensor(Blob* B, const char* name){
    for(uint32_t i=0;i<B->h.n_tensor;i++) if(!strcmp(B->tensors[i].name,name)) return (int)i; return -1;
}
static int parse_ctx(const char* s){ int v=atoi(s); const char* p=s; while(*p&&(*p>='0'&&*p<='9'))p++;
    if(*p=='k'||*p=='K') v*=1024; else if(*p=='m'||*p=='M') v*=1024*1024; return v; }

/* ---- GLM TP weight sharding ------------------------------------------------------------------
 * COLUMN-PARALLEL (output-row sharded): each rank owns a CONTIGUOUS row-range = a contiguous byte
 *   range of the [out,in] row-major matrix (weight AND its block-scale grid, whose rows shard the
 *   same way). q/v absorb, q_rope, shared+dense+expert gate/up + *_scale_inv.
 * ROW-PARALLEL (input-col sharded): each rank owns a STRIDED column-range. o_proj, shared+dense down,
 *   dense-down scale. Gathered into a contiguous per-rank buffer. */
static int glm_col(const char* n){
    return strstr(n,"derived.q_absorb")||strstr(n,"derived.q_rope")||strstr(n,"derived.v_absorb")
         ||strstr(n,"shared_experts.gate_proj")||strstr(n,"shared_experts.up_proj")
         ||strstr(n,"mlp.gate_proj.")||strstr(n,"mlp.up_proj.")    /* dense gate/up weight + _scale_inv */
         /* lm_head is VOCAB-column-parallel only when the packet asks for it — with GLM_SHARD_HEAD
          * the declared size is vocab/tp*hidden and an XARGMAX_FIN folds the per-rank maxima;
          * without it the packet declares the full table and it must stay REPLICATED. This says
          * only HOW it would shard; the bind loop below gates the col branch on the declared size
          * so a replicated packet still takes the replicated path. */
         ||!strcmp(n,"lm_head.weight");
}
static int glm_row(const char* n){
    return strstr(n,"o_proj.weight")||strstr(n,"shared_experts.down_proj")||strstr(n,"mlp.down_proj.");
}
/* Gather rank's strided col-slice [rank*in_sh, ) of a [out, in_full] row-major tensor into dst. */
static void gather_row(void* dst, const uint8_t* src, uint64_t out, uint64_t in_full,
                       int rank, int N, int elsize){
    uint64_t in_sh=in_full/(uint64_t)N;
    for(uint64_t r=0;r<out;r++)
        memcpy((char*)dst + (size_t)r*in_sh*elsize,
               src + ((size_t)r*in_full + (uint64_t)rank*in_sh)*elsize, (size_t)in_sh*elsize);
}

/* ---- per-GPU rank state ---- */
typedef struct {
    int id, n_gpu;
    void** dev;             /* [n_tensor] */
    void* d_tens;
    int i_ids,i_pos,i_kvlen,i_logits,i_og,i_dg;
    Prog* g;                /* decode program (per-rank device tables live here) */
    void *d_inst,*d_stream,*d_sofs,*d_slen,*d_waits,*d_succs,*d_ctr;
    PlowDevInst* h_inst;    /* pinned patched inst stream */
    int32_t *h_ids,*h_pos,*h_kvlen;
    /* peer reduction region (og_tp @ 0, dg_tp @ slot_b, xctr @ 2*slot_b) + [N] peer base table */
    void* peer; void* d_peer_tbl;
    uint32_t slot_b; size_t peer_bytes;
    /* per-layer KV-row writers */
    int nlk, ckv_ins[128], krot_ins[128];
} Dev;

int main(int argc, char** argv){
    if(argc<3){ printf("usage: %s <model.pkt> <weight-dir> [prompt.ids] [--steps M] [--sweep list] [--tp N] [--gen G]\n",argv[0]); return 1; }
    const char* pkt=argv[1]; const char* wdir=argv[2]; const char* prompt_file=NULL;
    const char* sweep_list=NULL; int steps=21, N=1, ngen=16, ep=0, want_gen=0;
    for(int i=3;i<argc;i++){
        if(!strcmp(argv[i],"--steps")&&i+1<argc) steps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--sweep")&&i+1<argc) sweep_list=argv[++i];
        else if(!strcmp(argv[i],"--tp")&&i+1<argc) N=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ep")) ep=1;   /* EP: bind WHOLE experts per rank (256/N), NULL remote */
        /* --gen WITH --sweep runs the token-identity generate in the SAME process, after the
         * sweep. The 4-minute 183 GiB/rank weight load is the whole cost of a run, and an A/B
         * that has to pay it twice per blob (once for the timing, once for the ids) prices an
         * afternoon of lease time. Every STEP re-patches ids/pos/kvlen and the per-layer KV-row
         * writers, so the generate starting at pos=0 overwrites whatever cache rows the sweep
         * left behind before it reads them. */
        else if(!strcmp(argv[i],"--gen")&&i+1<argc){ ngen=atoi(argv[++i]); want_gen=1; }
        else if(argv[i][0]!='-') prompt_file=argv[i];
    }
    setbuf(stdout,NULL);
    if(N<1) N=1; if(N>MAX_DEV) N=MAX_DEV;
    Blob B; if(load_blob(pkt,&B)) return 1;
    Safet S; if(st_open(&S,wdir)){ printf("no safetensors in %s\n",wdir); return 1; }
    const int dp=(int)B.h.n_prog-1;   /* decode program = last */
    Prog* G0=&B.prog[dp];
    printf("pkt: %u tensors, %u prog, decode %u ops | weights: %d shards\n",
           B.h.n_tensor,B.h.n_prog,G0->h.n_inst,S.n);

    /* Discover sharding from the blob's XReduce packets (self-describing): i0=hidden i1=tp i2=slot. */
    uint32_t hidden=0, blob_tp=1, slot_b=0;
    for(uint32_t i=0;i<G0->h.n_inst;i++) if(G0->insts[i].op==PLOW_DOP_XREDUCE){
        if(hidden==0){ hidden=G0->insts[i].i[0]; blob_tp=G0->insts[i].i[1]; }
        if(G0->insts[i].i[2]>slot_b) slot_b=G0->insts[i].i[2];
    }
    /* Fallback: the act.og_tp tensor (declared only when tp>1) is the authoritative sharding marker;
     * use it when the pkt carries no XReduce (the PLOW_NO_XREDUCE diagnostic pkt). hidden = og_tp/2. */
    if(blob_tp<=1 && N>1){ int io=find_tensor(&B,"act.og_tp");
        if(io>=0){ hidden=(uint32_t)(B.tensors[io].bytes/2); blob_tp=(uint32_t)N; slot_b=hidden*2; } }
    if(N>1){
        if(blob_tp<=1||hidden==0){ printf("ERROR: --tp %d but pkt is NOT sharded (no XReduce/og_tp). Recompile: gemma4 ... --tp %d\n",N,N); return 1; }
        if((int)blob_tp!=N){ printf("ERROR: pkt sharded for tp=%u but launched --tp %d\n",blob_tp,N); return 1; }
        if(slot_b==0){ printf("ERROR: sharded pkt but no dg_tp slot\n"); return 1; }
        printf("sharded pkt: tp=%u hidden=%u (peer slot=%uB, slot_b=%uB)\n",blob_tp,hidden,hidden*2,slot_b);
    }

    plow_hsa* h=plow_hsa_init(); if(!h){ printf("hsa init failed\n"); return 1; }
    const int ndev=plow_hsa_device_count(h);
    if(N>ndev){ printf("requested --tp %d but only %d GPUs\n",N,ndev); N=ndev; }
    char gfx[64]; uint32_t cus=0,lds=0; plow_hsa_device_info(h,0,gfx,&cus,&lds);
    const uint32_t NCU=B.h.n_cu; printf("== GLM-5.2 decode ==  TP=%d over %d GPUs | dev0 %s CUs=%u (pkt n_cu=%u)\n",N,ndev,gfx,cus,NCU);
    const char* elf=getenv("PLOW_INTERP")?getenv("PLOW_INTERP"):"interp_decode.elf";
    FILE* ef=fopen(elf,"rb"); if(!ef){ printf("%s missing\n",elf); return 1; }
    fseek(ef,0,SEEK_END); long co_n=ftell(ef); fseek(ef,0,SEEK_SET);
    void* co=malloc(co_n); if(fread(co,1,co_n,ef)!=(size_t)co_n) return 1; fclose(ef);
    plow_hsa_kernel kern[MAX_DEV];
    for(int r=0;r<N;r++){
        if(plow_hsa_load_code_object(h,r,co,co_n)){ printf("dev%d: load %s failed\n",r,elf); return 1; }
        if(plow_hsa_get_kernel(h,r,"plow_interp_dec_gfx950",&kern[r])){ printf("dev%d: no kernel\n",r); return 1; }
    }

    int i_logits=find_tensor(&B,"act.logits"); uint32_t VOCAB=i_logits>=0?(uint32_t)(B.tensors[i_logits].bytes/2):0;
    int i_pos=find_tensor(&B,"in.pos"); int MCTX=i_pos>=0?(int)(B.tensors[i_pos].bytes/4):0;
    int i_ids=find_tensor(&B,"in.ids"), i_kvlen=find_tensor(&B,"in.kvlen");
    int i_og=find_tensor(&B,"act.og_tp"), i_dg=find_tensor(&B,"act.dg_tp");
    if(i_ids<0||i_pos<0||i_kvlen<0||i_logits<0){ printf("*** missing in.ids/pos/kvlen/logits\n"); return 1; }
    if(N>1&&(i_og<0||i_dg<0)){ printf("*** sharded pkt missing act.og_tp/act.dg_tp\n"); return 1; }

    const size_t STAGE=64u<<20; void* stage=plow_hsa_alloc_host(h,STAGE);
    const size_t XCTR_BYTES=512u*PLOW_CTR_STRIDE*4u;
    const size_t peer_bytes=(N>1)?((size_t)2*slot_b+XCTR_BYTES):0;
    Dev devs[MAX_DEV]; memset(devs,0,sizeof(devs));

    /* ---- bind every rank ---- */
    for(int r=0;r<N;r++){
        Dev* D=&devs[r]; D->id=r; D->n_gpu=N; D->slot_b=slot_b; D->peer_bytes=peer_bytes;
        D->i_ids=i_ids; D->i_pos=i_pos; D->i_kvlen=i_kvlen; D->i_logits=i_logits; D->i_og=i_og; D->i_dg=i_dg;
        D->g=&B.prog[dp];
        D->dev=calloc(B.h.n_tensor,sizeof(void*));
        uint64_t wb=0; int nw=0; const double lt0=now();
        for(uint32_t i=0;i<B.h.n_tensor;i++){ PlowTensorDecl* td=&B.tensors[i];
            if(strstr(td->name,"mlp.experts.")) continue;                    /* packed per-layer below */
            int is_table=strstr(td->name,"expert_weight_table")||strstr(td->name,"expert_scale_table");
            int is_model=(!strncmp(td->name,"model.",6)||!strncmp(td->name,"lm_head",7))&&!is_table;
            if(is_model){
                uint64_t got=0, shp[2]={0,0}; int nd=0;
                const uint8_t* src=st_find_ex(&S,td->name,&got,shp,&nd);
                if(!src){ printf("dev%d: MISSING WEIGHT %s\n",r,td->name); return 1; }
                D->dev[i]=plow_hsa_alloc(h,r,td->bytes); if(!D->dev[i]){ printf("dev%d: VRAM alloc failed %s\n",r,td->name); return 1; }
                /* The name predicate says HOW a tensor shards; the declared-vs-disk size says
                 * WHETHER this packet sharded it. lm_head is the one tensor whose answer is a
                 * packet property (GLM_SHARD_HEAD) rather than a fixed rule, so the col branch is
                 * taken only when the packet actually declared a 1/N slice. A col/row tensor whose
                 * size does not match still fails — as the replicated SIZE MISMATCH below. */
                const int col=N>1&&glm_col(td->name)&&got==(uint64_t)td->bytes*(uint64_t)N;
                const int row=N>1&&glm_row(td->name);
                if(col){
                    /* contiguous output-row slice: full = td->bytes * N, offset = r*td->bytes */
                    if(got!=(uint64_t)td->bytes*N){ printf("dev%d: COL SHARD %s full %llu != %llu*%d\n",r,td->name,
                        (unsigned long long)got,(unsigned long long)td->bytes,N); return 1; }
                    const uint8_t* ss=src+(uint64_t)r*td->bytes;
                    for(uint64_t o=0;o<td->bytes;o+=STAGE){ size_t nn=(size_t)((td->bytes-o<STAGE)?(td->bytes-o):STAGE);
                        memcpy(stage,ss+o,nn); plow_hsa_copy_h2d(h,r,(uint8_t*)D->dev[i]+o,stage,nn); }
                } else if(row){
                    /* strided input-col gather: out=shp[0], in_full=shp[1], elsize from got/(out*in_full) */
                    if(nd!=2||shp[0]==0||shp[1]==0){ printf("dev%d: ROW SHARD needs 2-D shape %s (nd=%d)\n",r,td->name,nd); return 1; }
                    const uint64_t out=shp[0], in_full=shp[1];
                    if(in_full%(uint64_t)N){ printf("dev%d: ROW SHARD %s in=%llu not /%d\n",r,td->name,(unsigned long long)in_full,N); return 1; }
                    const int elsize=(int)(got/(out*in_full));
                    if((uint64_t)out*(in_full/N)*elsize!=td->bytes){ printf("dev%d: ROW SHARD MISMATCH %s\n",r,td->name); return 1; }
                    uint8_t* rb=malloc(td->bytes); if(!rb){ printf("dev%d: row-gather malloc\n",r); return 1; }
                    gather_row(rb,src,out,in_full,r,N,elsize);
                    for(uint64_t o=0;o<td->bytes;o+=STAGE){ size_t nn=(size_t)((td->bytes-o<STAGE)?(td->bytes-o):STAGE);
                        memcpy(stage,rb+o,nn); plow_hsa_copy_h2d(h,r,(uint8_t*)D->dev[i]+o,stage,nn); }
                    free(rb);
                } else {
                    /* replicated (norms, q_a_proj, kv_a_latent, k_rope, router, embed, lm_head) or tp==1 */
                    if(got!=td->bytes){ printf("dev%d: SIZE MISMATCH %s want %llu got %llu\n",r,td->name,
                        (unsigned long long)td->bytes,(unsigned long long)got); return 1; }
                    for(uint64_t o=0;o<td->bytes;o+=STAGE){ size_t nn=(size_t)((td->bytes-o<STAGE)?(td->bytes-o):STAGE);
                        memcpy(stage,src+o,nn); plow_hsa_copy_h2d(h,r,(uint8_t*)D->dev[i]+o,stage,nn); }
                }
                wb+=td->bytes; nw++;
            } else {
                D->dev[i]=plow_hsa_alloc(h,r,td->bytes); if(!D->dev[i]){ printf("dev%d: VRAM alloc failed %s\n",r,td->name); return 1; }
                if(td->init_off!=PLOW_INIT_NONE)
                    for(uint64_t o=0;o<td->bytes;o+=STAGE){ size_t nn=(size_t)((td->bytes-o<STAGE)?(td->bytes-o):STAGE);
                        memcpy(stage,B.init+td->init_off+o,nn); plow_hsa_copy_h2d(h,r,(uint8_t*)D->dev[i]+o,stage,nn); }
            }
        }
        /* experts: one buffer of fp8 weights + one of f32 scales PER MoE LAYER, sharded per rank.
         * gate/up = contiguous output-row slice; down = strided input-col gather (+ scale grid same). */
        const char* projs[3]={"gate_proj","up_proj","down_proj"};
        for(uint32_t li=0;li<B.h.n_tensor;li++){ PlowTensorDecl* td=&B.tensors[li];
            if(!strstr(td->name,"mlp.expert_weight_table")) continue;
            int layer; { const char* p=strstr(td->name,"layers."); layer=atoi(p+7); }
            char pfx[80]; snprintf(pfx,sizeof(pfx),"model.layers.%d.",layer);
            char nm[96]; uint64_t ew=0, esc=0; uint64_t gshp[2]={0,0}; int gnd=0;
            snprintf(nm,sizeof(nm),"%smlp.experts.0.gate_proj.weight",pfx); if(!st_find_ex(&S,nm,&ew,gshp,&gnd)){ printf("no experts L%d\n",layer); return 1; }
            snprintf(nm,sizeof(nm),"%smlp.experts.0.gate_proj.weight_scale_inv",pfx); st_find(&S,nm,&esc);
            /* full gate = [imoe, h] fp8 => ew=imoe*h; sharded slot ew/N. scale [ib,hb] => esc/N. */
            const uint64_t ew_sh=ew/(uint64_t)N, esc_sh=esc/(uint64_t)N;
            int E=0; while(1){ char n2[96]; uint64_t b2; snprintf(n2,sizeof(n2),"%smlp.experts.%d.gate_proj.weight",pfx,E); if(!st_find(&S,n2,&b2)) break; E++; }
            /* EP: this rank owns the CONTIGUOUS expert block [r*E/N, (r+1)*E/N); each LOCAL expert is
             * bound WHOLE (full ew/esc, no slice), and remote experts get a NULL table entry (the kernel
             * skips a null base). The per-rank buffer holds only E/N experts' full weights. TP (ep=0)
             * keeps the old per-expert slice. */
            const int e_lo = ep ? (int)((int64_t)r*E/N) : 0;
            const int e_hi = ep ? (int)((int64_t)(r+1)*E/N) : E;
            const int n_local = e_hi - e_lo;
            const uint64_t ew_e = ep ? ew : ew_sh, esc_e = ep ? esc : esc_sh;
            void* d_w=plow_hsa_alloc(h,r,(size_t)(ep?n_local:E)*3*ew_e); void* d_s=plow_hsa_alloc(h,r,(size_t)(ep?n_local:E)*3*esc_e);
            if(!d_w||!d_s){ printf("dev%d: expert buf alloc L%d\n",r,layer); return 1; }
            uint64_t* wtb=malloc((size_t)E*3*8); uint64_t* stb=malloc((size_t)E*3*8);
            memset(wtb,0,(size_t)E*3*8); memset(stb,0,(size_t)E*3*8);  /* NULL = remote (EP) / unused */
            for(int e=0;e<E;e++){
                if(ep && (e<e_lo || e>=e_hi)) continue;  /* remote expert: leave wtb/stb NULL */
                for(int j=0;j<3;j++){ uint64_t gw=0,gs=0; uint64_t wshp[2]={0,0},sshp[2]={0,0}; int wnd=0,snd=0;
                snprintf(nm,sizeof(nm),"%smlp.experts.%d.%s.weight",pfx,e,projs[j]); const uint8_t* ws=st_find_ex(&S,nm,&gw,wshp,&wnd);
                snprintf(nm,sizeof(nm),"%smlp.experts.%d.%s.weight_scale_inv",pfx,e,projs[j]); const uint8_t* ss=st_find_ex(&S,nm,&gs,sshp,&snd);
                if(!ws||!ss){ printf("dev%d: expert %d.%s missing L%d\n",r,e,projs[j],layer); return 1; }
                const int islot = ep ? (e - e_lo) : e;
                void* wd=(char*)d_w+((size_t)islot*3+j)*ew_e; void* sd=(char*)d_s+((size_t)islot*3+j)*esc_e;
                if(ep){ /* whole local expert, no slice (full ew/esc) */
                    memcpy(stage,ws,gw); plow_hsa_copy_h2d(h,r,wd,stage,gw);
                    memcpy(stage,ss,gs); plow_hsa_copy_h2d(h,r,sd,stage,gs); }
                else if(N==1){ memcpy(stage,ws,gw); plow_hsa_copy_h2d(h,r,wd,stage,gw);
                          memcpy(stage,ss,gs); plow_hsa_copy_h2d(h,r,sd,stage,gs); }
                else if(j<2){ /* gate/up: contiguous output-row slice */
                    memcpy(stage,ws+(uint64_t)r*ew_sh,ew_sh); plow_hsa_copy_h2d(h,r,wd,stage,ew_sh);
                    memcpy(stage,ss+(uint64_t)r*esc_sh,esc_sh); plow_hsa_copy_h2d(h,r,sd,stage,esc_sh);
                } else { /* down: strided input-col gather (weight [h,imoe] fp8; scale [hb,ib] f32).
                          * Gather into the PINNED stage buffer (copy_h2d blocks + does not pin src,
                          * so a malloc'd gather buffer would fault the SDMA). */
                    gather_row(stage,ws,wshp[0],wshp[1],r,N,1); plow_hsa_copy_h2d(h,r,wd,stage,ew_sh);
                    gather_row(stage,ss,sshp[0],sshp[1],r,N,4); plow_hsa_copy_h2d(h,r,sd,stage,esc_sh);
                }
                wtb[e*3+j]=(uint64_t)(uintptr_t)wd; stb[e*3+j]=(uint64_t)(uintptr_t)sd; wb+=ew_e+esc_e; } }
            char sn[96]; snprintf(sn,sizeof(sn),"%smlp.expert_scale_table",pfx); int i_est=find_tensor(&B,sn);
            plow_hsa_upload(h,r,D->dev[(int)li],wtb,(size_t)E*3*8); plow_hsa_upload(h,r,D->dev[i_est],stb,(size_t)E*3*8);
            free(wtb); free(stb);
        }
        D->d_tens=plow_hsa_alloc(h,r,(size_t)B.h.n_tensor*sizeof(void*));
        plow_hsa_upload(h,r,D->d_tens,D->dev,(size_t)B.h.n_tensor*sizeof(void*));
        printf("dev%d: bound %d named weights + packed experts (%.1f GiB) in %.1f s\n",r,nw,wb/1073741824.0,now()-lt0);
    }

    /* ---- PEER SETUP (§7a): one peer-mapped reduction region per GPU, all-to-all, + [N] table ---- */
    if(N>1){
        void* peer_base[MAX_DEV];
        for(int r=0;r<N;r++){ devs[r].peer=plow_hsa_alloc_peer(h,r,peer_bytes);
            if(!devs[r].peer){ printf("dev%d: alloc_peer failed: %s\n",r,plow_hsa_last_error()); return 1; }
            peer_base[r]=devs[r].peer; }
        for(int r=0;r<N;r++){ devs[r].d_peer_tbl=plow_hsa_alloc(h,r,(size_t)N*sizeof(void*));
            plow_hsa_upload(h,r,devs[r].d_peer_tbl,peer_base,(size_t)N*sizeof(void*)); }
        /* bind og_tp @ peer+0, dg_tp @ peer+slot_b so o_proj/down write peer-visible partials */
        for(int r=0;r<N;r++){ Dev* D=&devs[r];
            D->dev[i_og]=D->peer; D->dev[i_dg]=(char*)D->peer+(size_t)slot_b;
            plow_hsa_upload(h,r,D->d_tens,D->dev,(size_t)B.h.n_tensor*sizeof(void*)); }
        printf("peer setup: %d regions (%zuB each) peer-mapped; og_tp@0 dg_tp@%u xctr@%u\n",
               N,peer_bytes,slot_b,2*slot_b);
    }

    /* ---- per-layer KV-row writers (ckv RMSNorm + k_rope HeadNormRope), + program tables per rank ---- */
    for(int r=0;r<N;r++){ Dev* D=&devs[r]; Prog* g=D->g;
        D->nlk=0;
        for(uint32_t k=0;k<g->h.n_inst;k++){ PlowDevInst* d=&g->insts[k];
            if(d->op==PLOW_DOP_RMSNORM){ int t0=d->t[0]; if(t0<(int)B.h.n_tensor&&!strncmp(B.tensors[t0].name,"kv.",3)&&strstr(B.tensors[t0].name,".ckv")) D->ckv_ins[D->nlk]=k; }
            if(d->op==PLOW_DOP_HEADNORM_ROPE){ int t0=d->t[0]; if(t0<(int)B.h.n_tensor&&!strncmp(B.tensors[t0].name,"kv.",3)&&strstr(B.tensors[t0].name,".krot")){ D->krot_ins[D->nlk]=k; D->nlk++; } }
        }
        D->d_inst=plow_hsa_alloc(h,r,(size_t)g->h.n_inst*sizeof(PlowDevInst));
        D->d_stream=plow_hsa_alloc(h,r,(size_t)g->h.n_stream*sizeof(PlowStreamEnt));
        D->d_sofs=plow_hsa_alloc(h,r,(size_t)NCU*4); D->d_slen=plow_hsa_alloc(h,r,(size_t)NCU*4);
        D->d_waits=plow_hsa_alloc(h,r,(size_t)(g->h.n_wait?g->h.n_wait:1)*sizeof(PlowWait));
        D->d_succs=plow_hsa_alloc(h,r,(size_t)(g->h.n_succ?g->h.n_succ:1)*4);
        D->d_ctr=plow_hsa_alloc(h,r,(size_t)g->h.n_counter*PLOW_CTR_STRIDE*4);
        plow_hsa_upload(h,r,D->d_stream,g->stream,(size_t)g->h.n_stream*sizeof(PlowStreamEnt));
        plow_hsa_upload(h,r,D->d_sofs,g->stream_ofs,(size_t)NCU*4);
        plow_hsa_upload(h,r,D->d_slen,g->stream_len,(size_t)NCU*4);
        if(g->h.n_wait) plow_hsa_upload(h,r,D->d_waits,g->waits,(size_t)g->h.n_wait*sizeof(PlowWait));
        if(g->h.n_succ) plow_hsa_upload(h,r,D->d_succs,g->succs,(size_t)g->h.n_succ*4);
        D->h_inst=plow_hsa_alloc_host(h,(size_t)g->h.n_inst*sizeof(PlowDevInst));
        memcpy(D->h_inst,g->insts,(size_t)g->h.n_inst*sizeof(PlowDevInst));
        D->h_ids=plow_hsa_alloc_host(h,4); D->h_pos=plow_hsa_alloc_host(h,4); D->h_kvlen=plow_hsa_alloc_host(h,4);
    }
    printf("KV-row writers: %d layers (ckv RMSNorm i[2] + k_rope HeadNormRope i[3])\n",devs[0].nlk);

    /* shared zero buffer for counters + peer xctr */
    size_t zc_bytes=(size_t)G0->h.n_counter*PLOW_CTR_STRIDE*4; if(zc_bytes<XCTR_BYTES) zc_bytes=XCTR_BYTES;
    uint32_t* zc=plow_hsa_alloc_host(h,zc_bytes); memset(zc,0,zc_bytes);

    /* ---- TRACE (PLOW_TRACE_RAW=<prefix>): per-(workgroup,packet) PlowTraceRec buffer.
     * EVERY rank is traced when PLOW_TRACE_ALLRANKS=1, else rank 0 only (the historical form).
     * All-rank tracing is what prices CROSS-RANK ARRIVAL SKEW: the 156 XReduce rendezvous hard-
     * synchronise the ranks, so the work each rank does BETWEEN two consecutive collectives is
     * comparable across ranks WITHOUT any cross-device clock sync — only durations inside one
     * rank's own s_memrealtime domain are ever differenced.
     * A .insts.txt sidecar maps inst -> op + operand tensor names/bytes so the python pass can label
     * each op and compute per-op HBM bytes. Per-ctx dumps land at <prefix>[.rk<R>].tp<N>.ctx<C>.bin. */
    const char* trace_raw=getenv("PLOW_TRACE_RAW");
    const int trace_all = getenv("PLOW_TRACE_ALLRANKS") && atoi(getenv("PLOW_TRACE_ALLRANKS"));
    void* d_trace[MAX_DEV]; for(int r=0;r<MAX_DEV;r++) d_trace[r]=NULL;
    void* d_trace0=NULL;
    if(trace_raw){
        d_trace0=plow_hsa_alloc(h,0,(size_t)G0->h.n_stream*sizeof(PlowTraceRec));
        d_trace[0]=d_trace0;
        if(trace_all) for(int r=1;r<N;r++)
            d_trace[r]=plow_hsa_alloc(h,r,(size_t)devs[r].g->h.n_stream*sizeof(PlowTraceRec));
        char fn[512]; snprintf(fn,sizeof(fn),"%s.insts.txt",trace_raw); FILE* sf=fopen(fn,"w");
        if(sf){ fprintf(sf,"# inst op blocks | t[k]=idx:name:bytes ... | i0 i1 i2 i3\n");
            for(uint32_t k=0;k<G0->h.n_inst;k++){ PlowDevInst* d=&G0->insts[k];
                fprintf(sf,"%u %u %u |",k,d->op,d->blocks);
                for(int tk=0;tk<8;tk++){ uint32_t ti=d->t[tk]; if(ti==PLOW_TENSOR_NONE) continue;
                    fprintf(sf," t%d=%u:%s:%llu",tk,ti,ti<B.h.n_tensor?B.tensors[ti].name:"?",
                            (unsigned long long)(ti<B.h.n_tensor?B.tensors[ti].bytes:0)); }
                fprintf(sf," | %u %u %u %u\n",d->i[0],d->i[1],d->i[2],d->i[3]); }
            fclose(sf); printf("trace sidecar -> %s.insts.txt (%u insts)\n",trace_raw,G0->h.n_inst); }
    }

    /* decode_step: all ranks. patch ids/pos/kvlen + KV-row writes to `pos`, zero ctr+xctr, launch all,
     * wait all, read rank0 next id into `outtok`. Returns gpu ms (wall of the N-rank co-resident step). */
    #define STEP(tok,pos,kvl,outtok,msout) do{ \
        for(int r=0;r<N;r++){ Dev* D=&devs[r]; \
            *D->h_ids=(tok); plow_hsa_copy_h2d(h,r,D->dev[i_ids],D->h_ids,4); \
            *D->h_pos=(pos); plow_hsa_copy_h2d(h,r,D->dev[i_pos],D->h_pos,4); \
            *D->h_kvlen=(kvl); plow_hsa_copy_h2d(h,r,D->dev[i_kvlen],D->h_kvlen,4); \
            for(int L=0;L<D->nlk;L++){ D->h_inst[D->ckv_ins[L]].i[2]=(uint32_t)(pos); D->h_inst[D->krot_ins[L]].i[3]=(uint32_t)(pos); } \
            plow_hsa_copy_h2d(h,r,D->d_inst,D->h_inst,(size_t)D->g->h.n_inst*sizeof(PlowDevInst)); \
            plow_hsa_copy_h2d(h,r,D->d_ctr,zc,(size_t)D->g->h.n_counter*PLOW_CTR_STRIDE*4); \
            if(N>1){ size_t xo=(size_t)2*D->slot_b; plow_hsa_copy_h2d(h,r,(char*)D->peer+xo,zc,D->peer_bytes-xo); } } \
        double _t0=now(); \
        for(int r=0;r<N;r++){ Dev* D=&devs[r]; PlowProgram pr; memset(&pr,0,sizeof(pr)); \
            pr.insts=D->d_inst; pr.stream=D->d_stream; pr.stream_ofs=D->d_sofs; pr.stream_len=D->d_slen; \
            pr.waits=D->d_waits; pr.succs=D->d_succs; pr.counters=D->d_ctr; pr.tensors=(void* const*)D->d_tens; \
            if(N>1){ pr.rank=(uint32_t)D->id; pr.n_gpu=(uint32_t)N; pr.peer_scratch=(void* const*)D->d_peer_tbl; \
                     pr.xctr=(uint32_t*)((char*)D->peer+(size_t)2*D->slot_b); } \
            if(D->id<MAX_DEV && d_trace[D->id]) pr.trace=(PlowTraceRec*)d_trace[D->id]; \
            if(plow_hsa_launch(h,r,&kern[r],NCU*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&pr,sizeof(pr))){ printf("dev%d LAUNCH FAILED\n",r); return 1; } } \
        for(int r=0;r<N;r++) plow_hsa_wait(h,r); \
        { double _ms=(now()-_t0)*1e3; double* _mp=(msout); if(_mp)*_mp=_ms; } \
        plow_hsa_copy_d2h(h,0,devs[0].h_ids,devs[0].dev[i_ids],4); (outtok)=*devs[0].h_ids; \
    }while(0)

    /* ---- SWEEP (decode-only, 1 tok): tp_decode format ---- */
    if(sweep_list){
        int ctxs[16], nc=0; char buf[256]; snprintf(buf,sizeof(buf),"%s",sweep_list);
        for(char* t=strtok(buf,",");t&&nc<16;t=strtok(NULL,",")) ctxs[nc++]=parse_ctx(t);
        printf("\nSWEEP (decode-only, 1 tok, median of %d): TP=%d\n",steps,N);
        printf("  %-8s %12s %10s\n","ctx","ms/tok","tok/s");
        for(int c=0;c<nc;c++){ int ctx=ctxs[c]; int tok;
            if(MCTX&&ctx>MCTX){ printf("  %-8d   (exceeds pkt max_ctx %d)\n",ctx,MCTX); continue; }
            if(MCTX&&ctx>=MCTX) ctx=MCTX-1;   /* pkt holds positions 0..MCTX-1 (mirror tp_decode.c) */
            double* samp=malloc((size_t)steps*sizeof(double));
            for(int w=0;w<2;w++) STEP(42,ctx,ctx+1,tok,NULL);
            for(int s=0;s<steps;s++){ double ms=0; STEP(42,ctx,ctx+1,tok,&ms); samp[s]=ms; }
            qsort(samp,(size_t)steps,sizeof(double),cmp_dbl);
            double med=samp[steps/2]; printf("  %-8d %12.3f %10.1f\n",ctx,med,1000.0/med); free(samp);
            if(trace_raw && d_trace0){
                /* one traced step at this ctx, dump every traced rank's PlowTraceRec[n_stream]. */
                double ms=0; STEP(42,ctx,ctx+1,tok,&ms);
                for(int r=0;r<N;r++){
                    if(!d_trace[r]) continue;
                    uint32_t nrec=devs[r].g->h.n_stream;
                    PlowTraceRec* tr=plow_hsa_alloc_host(h,(size_t)nrec*sizeof(PlowTraceRec));
                    plow_hsa_copy_d2h(h,r,tr,d_trace[r],(size_t)nrec*sizeof(PlowTraceRec));
                    char fn[512];
                    if(trace_all) snprintf(fn,sizeof(fn),"%s.rk%d.tp%d.ctx%d.bin",trace_raw,r,N,ctx);
                    else          snprintf(fn,sizeof(fn),"%s.tp%d.ctx%d.bin",trace_raw,N,ctx);
                    FILE* tf=fopen(fn,"wb");
                    if(tf){ fwrite(tr,sizeof(PlowTraceRec),nrec,tf); fclose(tf);
                        printf("    raw trace -> %s (%u recs, traced-ms=%.3f)\n",fn,nrec,ms); }
                }
            }
        }
        if(!want_gen){ plow_hsa_shutdown(h); return 0; }
    }

    /* ---- generate from a prompt (or a default) ---- */
    int32_t prompt[64]; int np=0;
    if(prompt_file){ FILE* pf=fopen(prompt_file,"rb"); if(pf){ np=(int)fread(prompt,4,64,pf); fclose(pf); } }
    if(np<=0){ int32_t d[]={100,264,6722,315,9822,374}; np=6; memcpy(prompt,d,sizeof(d)); printf("(no prompt file — using placeholder ids)\n"); }
    printf("\nprompt %d tokens; generating %d:\n  ids:",np,ngen);
    int pos=0, tok=prompt[0], maxtok=0, disagree=0;
    for(int p=0;p<np;p++){ STEP(prompt[p],pos,pos+1,tok,NULL); pos++; }   /* prefill: last STEP's next id is 1st gen */
    for(int gi=0;gi<ngen;gi++){ printf(" %d",tok); if(tok>maxtok) maxtok=tok;
        STEP(tok,pos,pos+1,tok,NULL); pos++;
        /* CROSS-RANK AGREEMENT. With a REPLICATED lm_head every rank argmaxes the full vocab and
         * trivially agrees. With a VOCAB-SHARDED one they only agree if XARGMAX_FIN folded — and a
         * fold that silently no-ops leaves each rank holding its own shard's winner, which the
         * rank-0-only readback above cannot see. Costs one 4-byte D2H per rank per token. */
        for(int r=1;r<N;r++){ int32_t t2; plow_hsa_copy_d2h(h,r,devs[r].h_ids,devs[r].dev[i_ids],4);
            t2=*devs[r].h_ids; if(t2!=tok) disagree++; } }
    printf("\n(mechanics: %d decode steps ran, ids in [0,%u), max id %d, cross-rank disagreements %d)\n",
           np+ngen,VOCAB,maxtok,disagree);
    plow_hsa_shutdown(h); return 0;
}
