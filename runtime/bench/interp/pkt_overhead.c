/* pkt_overhead.c — what does ONE PACKET of the persistent interpreter cost?
 *
 * plow's whole claim is that a resident interpreter walking a packet stream beats per-op kernel
 * launches. That is only worth anything if the packet PROTOCOL is cheap, so this measures it.
 *
 * TWO TRAPS THIS HARNESS EXISTS TO AVOID, both of which produce confident wrong answers:
 *
 * 1. A NOP-ONLY STREAM CANNOT ANSWER THIS. It generates no compute load, the DPM governor parks
 *    the part at its idle clock (132 MHz measured on MI300X, against 2100 boost), and every
 *    "overhead" number lands ~16x pessimistic. A NOP stream measured 26.7 us/packet here; the
 *    same protocol under load is a small fraction of that. Do not benchmark an idle GPU.
 *    `rocm-smi --setperflevel high` needs privileges this container does not have.
 *
 * 2. WALL CLOCK PER LAUNCH CONFLATES WORK WITH PROTOCOL. So the method is FIXED TOTAL WORK,
 *    VARYING PACKET COUNT: the same 268M-element RESIDUAL is cut into P packets and P is swept.
 *    The math per launch is identical; only the packet count moves, and the slope in P is the
 *    protocol cost, measured while the machine is busy.
 *
 * Timestamps come from the interpreter's own trace (s_memrealtime, a constant ~97.1 MHz on
 * MI300X -- calibrated here, not assumed) because the shader clock moves with DVFS.
 *
 * READ THE GATE NUMBER CAREFULLY: t_ready - t_arrive on one workgroup includes waiting for the
 * SLOWEST workgroup of the previous packet. It is convergence latency PLUS load-imbalance tail,
 * not pure protocol, and the two cannot be separated from inside the interpreter.
 *
 * MEASURED, MI300X (gfx942), 304 workgroups, 8-wave decode object:
 *   - P <= 8: total time FLAT (0.52-0.57 ms). Protocol is entirely hidden while packets are big.
 *   - P = 64: gate ~12 us, stream walk + decode ~4.8 us, per-packet slope ~24 us (independent)
 *   - PLOW_GATE_HIER (two-level, 8 L2 domains instead of 304 workgroups polling): NO measurable
 *     change, 4.130 -> 4.102 ms at P=64, inside noise. The gate is not poll-traffic bound and it
 *     is not cache-maintenance bound; it is convergence and tail. Reducing poll traffic is the
 *     wrong lever -- reduce the NUMBER of gated boundaries, or the imbalance feeding them.
 *
 * Build:
 *   /usr/bin/env -i PATH=/usr/bin:/bin HOME="$HOME" /usr/bin/gcc -O2 -std=gnu11 -o pkt_overhead \
 *     runtime/bench/interp/pkt_overhead.c runtime/amd/hsa_backend.c \
 *     -I runtime/amd -I runtime/common -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -lm
 * Run:  ./pkt_overhead <interp object.elf>      PLOW_HIER=1 engages the hierarchical gate.
 */
#define _POSIX_C_SOURCE 200809L
#include <time.h>
#include "hsa_backend.h"
#include "dev_isa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef unsigned short bf16;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static plow_hsa* H; static plow_hsa_kernel K; static unsigned NCU;
static double TR_G,TR_E,TR_L; static int HIER=0;

/* P packets, each RESIDUAL over ELEMS elements. gated: 0 = independent, 1 = counter-chained. */
static double run(unsigned P, unsigned ELEMS, int gated, void** tens4) {
    PlowDevInst* insts = calloc(P, sizeof(PlowDevInst));
    for (unsigned k=0;k<P;k++){
        for (unsigned j=0;j<8;j++) insts[k].t[j]=PLOW_TENSOR_NONE;
        insts[k].op=PLOW_DOP_RESIDUAL; insts[k].blocks=NCU;
        insts[k].t[0]=0; insts[k].t[1]=1; insts[k].t[2]=2;
        insts[k].i[0]=ELEMS; insts[k].fj[0].f=1.0f;
    }
    PlowStreamEnt* st = calloc((size_t)NCU*P, sizeof(PlowStreamEnt));
    uint32_t* sofs=malloc(4*NCU); uint32_t* slen=malloc(4*NCU);
    PlowWait* waits=calloc(P?P:1,sizeof(PlowWait)); uint32_t* succs=calloc(P?P:1,4);
    for (unsigned k=0;k<P;k++){ succs[k]=k; waits[k].id=(k?k-1:0); waits[k].threshold=NCU; }
    for (unsigned b=0;b<NCU;b++){
        sofs[b]=b*P; slen[b]=P;
        for (unsigned k=0;k<P;k++){
            PlowStreamEnt* e=&st[(size_t)b*P+k]; e->inst=k; e->slice=b;
            if (gated){ e->succ_len=1; e->succ_ofs=k; if(k){e->wait_len=1;e->wait_ofs=k;}
                        if (HIER) e->flags |= (uint16_t)(((NCU/8u) << PLOW_SE_NPER_SHIFT) & PLOW_SE_NPER_MASK); }
        }
    }
    void* d_i=plow_hsa_alloc(H,0,(size_t)P*sizeof(PlowDevInst));
    void* d_s=plow_hsa_alloc(H,0,(size_t)NCU*P*sizeof(PlowStreamEnt));
    void* d_so=plow_hsa_alloc(H,0,4*NCU); void* d_sl=plow_hsa_alloc(H,0,4*NCU);
    void* d_w=plow_hsa_alloc(H,0,(size_t)(P?P:1)*sizeof(PlowWait));
    void* d_su=plow_hsa_alloc(H,0,(P?P:1)*4);
    void* d_c=plow_hsa_alloc(H,0,((size_t)(P+1)*PLOW_CTR_STRIDE + (size_t)P*8*3 + 64)*4);
    void* d_t=plow_hsa_alloc(H,0,64);
    void* d_tr=plow_hsa_alloc(H,0,(size_t)NCU*P*sizeof(PlowTraceRec));
    plow_hsa_upload(H,0,d_i,insts,(size_t)P*sizeof(PlowDevInst));
    plow_hsa_upload(H,0,d_s,st,(size_t)NCU*P*sizeof(PlowStreamEnt));
    plow_hsa_upload(H,0,d_so,sofs,4*NCU); plow_hsa_upload(H,0,d_sl,slen,4*NCU);
    plow_hsa_upload(H,0,d_w,waits,(size_t)(P?P:1)*sizeof(PlowWait));
    plow_hsa_upload(H,0,d_su,succs,(P?P:1)*4);
    plow_hsa_upload(H,0,d_t,tens4,3*sizeof(void*));
    PlowProgram pr; memset(&pr,0,sizeof(pr));
    pr.insts=(const PlowDevInst*)d_i; pr.stream=(const PlowStreamEnt*)d_s;
    pr.stream_ofs=(const uint32_t*)d_so; pr.stream_len=(const uint32_t*)d_sl;
    pr.waits=(const PlowWait*)d_w; pr.succs=(const uint32_t*)d_su;
    pr.counters=(uint32_t*)d_c; pr.tensors=(void* const*)d_t; pr.trace=(PlowTraceRec*)d_tr;
    if (HIER && gated) { pr.l2_domains = 8; pr.hier_base = (P+1)*PLOW_CTR_STRIDE; }
    size_t czn=(size_t)(P+1)*PLOW_CTR_STRIDE + (size_t)P*8*3 + 64;
    uint32_t* zc=calloc(czn,4);
    double best=1e9;
    for(int r=0;r<7;r++){
        plow_hsa_upload(H,0,d_c,zc,czn*4); plow_hsa_wait(H,0);
        double t0=now();
        plow_hsa_launch(H,0,&K,NCU*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&pr,sizeof(pr));
        plow_hsa_wait(H,0); double dt=now()-t0;
        if(r>=2 && dt<best) best=dt;
    }
    {   PlowTraceRec* tr=malloc((size_t)NCU*P*sizeof(PlowTraceRec));
        plow_hsa_download(H,0,tr,d_tr,(size_t)NCU*P*sizeof(PlowTraceRec));
        double g=0,e2=0,lp=0; unsigned c2=0;
        for(unsigned k=0;k+1<P;k++){
            g += (double)(tr[k].t_ready - tr[k].t_arrive);
            e2+= (double)(tr[k].t_end   - tr[k].t_ready);
            lp+= (double)(tr[k+1].t_arrive - tr[k].t_end);
            c2++;
        }
        if(c2){TR_G=g/c2;TR_E=e2/c2;TR_L=lp/c2;}
        free(tr); }
    plow_hsa_free(H,d_tr);
    free(insts);free(st);free(sofs);free(slen);free(waits);free(succs);free(zc);
    plow_hsa_free(H,d_i);plow_hsa_free(H,d_s);plow_hsa_free(H,d_so);plow_hsa_free(H,d_sl);
    plow_hsa_free(H,d_w);plow_hsa_free(H,d_su);plow_hsa_free(H,d_c);plow_hsa_free(H,d_t);
    return best;
}
int main(int argc,char**argv){
    H=plow_hsa_init(); char nm[64];unsigned cu,l; plow_hsa_device_info(H,0,nm,&cu,&l); NCU=cu;
    printf("dev %s CUs=%u\n",nm,cu);
    const char* elf=argc>1?argv[1]:"interp_decode.elf";
    FILE*f=fopen(elf,"rb"); if(!f){perror(elf);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    void*co=malloc(n); if(fread(co,1,n,f)!=(size_t)n)return 1; fclose(f);
    plow_hsa_load_code_object(H,0,co,n);
    char sym[96]; snprintf(sym,sizeof(sym),"plow_interp_dec_%s",nm);
    if(plow_hsa_get_kernel(H,0,sym,&K)){snprintf(sym,sizeof(sym),"plow_interp_%s",nm);
        if(plow_hsa_get_kernel(H,0,sym,&K)){fprintf(stderr,"no symbol\n");return 1;}}
    const unsigned TOTAL = 1u<<28;  /* 268M elems -> 1.6 GB of traffic; enough to hold boost clocks */
    void* da=plow_hsa_alloc(H,0,(size_t)TOTAL*2);
    void* db=plow_hsa_alloc(H,0,(size_t)TOTAL*2);
    void* dc=plow_hsa_alloc(H,0,(size_t)TOTAL*2);
    void* tens[3]={dc,da,db};
    printf("\nFIXED TOTAL WORK (%u elems of RESIDUAL), split into P packets.\n", TOTAL);
    printf("Slope in P is the per-packet protocol cost, measured while the GPU is busy.\n\n");
    HIER = getenv("PLOW_HIER") ? atoi(getenv("PLOW_HIER")) : 0;
    printf("  hierarchical gate: %s\n", HIER?"ON (PLOW_GATE_HIER, 8 L2 domains)":"off");
    printf("  %-6s %14s %14s %14s %14s\n","P","indep ms","indep ns/pkt","gated ms","gated ns/pkt");
    double p0i=0,p0g=0; unsigned P0=0;
    for (unsigned P=1;P<=64;P*=2){
        unsigned E=TOTAL/P;
        double ti=run(P,E,0,tens), tg=run(P,E,1,tens);
        if(P0==0){P0=P;p0i=ti;p0g=tg;}
        double si = (P>P0)? (ti-p0i)/(P-P0)*1e9 : 0;
        double sg = (P>P0)? (tg-p0g)/(P-P0)*1e9 : 0;
        printf("  %-6u %14.4f %14.0f %14.4f %14.0f   | trace(gated) gate=%.1fus exec=%.1fus loop=%.1fus\n",
               P,ti*1e3,si,tg*1e3,sg, TR_G*10.298e-3, TR_E*10.298e-3, TR_L*10.298e-3);
    }
    return 0;
}
