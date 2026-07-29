/* moe_prefill_gfx950_test.hip — ops 83/84/86/87 and GROUP-LIMITED top-k, on hardware.
 *
 * Validates the MoE prefill family AHEAD OF ANY EMITTER, using analytic oracles rather than a
 * checkpoint, so the arms are proven before a packet exists to exercise them.
 *
 * THE ORACLES, and why each is exact rather than a tolerance:
 *   ROUTER_TOPK_PF (83)  is d_moe_router_topk with a token loop around it, so every token's
 *                        table must be BIT-IDENTICAL to the decode router run on that token's
 *                        logit row. memcmp, not epsilon.
 *   GROUP-LIMITED top-k  at n_group<=1 must be bit-identical to the flat path (that is the
 *                        no-regression guarantee for every GLM/Qwen/Mixtral packet); at
 *                        n_group>1 it must equal an independent CPU implementation of the
 *                        noaux_tc rule. THIS IS THE CASE THAT WAS SILENTLY WRONG: the body
 *                        applied a flat top-k while its header claimed group-limited, and at
 *                        DeepSeek-V3's 8 groups / topk_group 4 that is a different expert set.
 *   ALIGN_PF (84)        is checked STRUCTURALLY and exhaustively: every live (token,slot) slot
 *                        appears exactly once among the gathered rows, lands inside its own
 *                        expert's padded range, carries the right token/partidx/gate, and every
 *                        other row is PLOW_EXPERT_UNUSED. The scatter uses an LDS atomic cursor
 *                        so row ORDER within an expert is not deterministic — the test asserts
 *                        the multiset, which is exactly the property the kernel guarantees.
 *   COMBINE_PF (87)      at T=1 must be bit-identical to the decode d_moe_combine.
 *
 *   nix develop --command bash -c '/opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 -w \
 *       -I runtime/amd -I runtime/common runtime/tests/moe_prefill_gfx950_test.hip -o /tmp/moe_pf'
 *   sg render -c 'HIP_VISIBLE_DEVICES=1 /tmp/moe_pf'
 */
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#define PLOW_MOE_PREFILL 1
#include "amd_common.h"
#include "op_moe.h"

#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_router_pf(unsigned char* tab, const bf16* logit, const float* bias, unsigned n_exp,
                 unsigned k, unsigned flags, float rs, unsigned T, unsigned ng, unsigned tg) {
    __shared__ float lds[8192];
    d_moe_router_topk_pf(tab, logit, bias, n_exp, k, flags, rs, T, blockIdx.x, gridDim.x, lds, ng, tg);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_router_1(unsigned char* tab, const bf16* logit, const float* bias, unsigned n_exp,
                unsigned k, unsigned flags, float rs, unsigned ng, unsigned tg) {
    __shared__ float lds[8192];
    d_moe_router_topk(tab, logit, bias, n_exp, k, flags, rs, blockIdx.x, gridDim.x, lds, ng, tg);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_align(int* meta, const unsigned char* tab, unsigned* rt, unsigned* rp, float* rg,
             unsigned T, unsigned n_exp, unsigned k) {
    __shared__ unsigned lds[4096];
    d_moe_align_pf(meta, tab, rt, rp, rg, T, n_exp, k, blockIdx.x, lds);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_comb_pf(bf16* o, const bf16* r, const bf16* sh, const float* p, unsigned H, unsigned k, unsigned T) {
    d_moe_combine_pf(o, r, sh, p, H, k, T, blockIdx.x, gridDim.x);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_comb_1(bf16* o, const bf16* r, const bf16* sh, const float* p, unsigned H, unsigned k) {
    d_moe_combine(o, r, sh, p, H, k, blockIdx.x, gridDim.x);
}

/* DECODE expert (op 45) with the MXFP4 encoding (i[6]=2). */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)
void t_expert_glu_mx(bf16* fu, const bf16* x, const unsigned char* tab,
                     const unsigned long long* wtab, const unsigned long long* stab,
                     unsigned slot, unsigned I, unsigned H, unsigned n_exp, unsigned act) {
    d_moe_expert_glu_fp8_blk(fu, x, tab, wtab, stab, slot, I, H, n_exp, act, blockIdx.x,
                             gridDim.x, PLOW_MOE_ENC_MXFP4);
}

static int fails = 0;
static void ck(const char* w, bool ok, const char* extra=""){
    printf("  %-56s %s %s\n", w, ok?"PASS":"FAIL", extra); if(!ok) fails++;
}
static float fp4d(unsigned n){const float T[8]={0,.5f,1,1.5f,2,3,4,6};return (n&8)?-T[n&7]:T[n&7];}
static float e8(unsigned char b){return ldexpf(1.f,(int)b-127);}
static unsigned short f2b(float f){unsigned u;memcpy(&u,&f,4);u+=0x7fffu+((u>>16)&1u);return (unsigned short)(u>>16);}
static float bf(unsigned short h){unsigned u=(unsigned)h<<16;float f;memcpy(&f,&u,4);return f;}

int main(){
  const unsigned n_exp=384, k=8, T=37, flags=1u|2u|4u; /* sigmoid + norm + bias, the GLM/DS path */
  const float rs=2.5f;
  std::vector<unsigned short> logit((size_t)T*n_exp);
  std::vector<float> bias(n_exp);
  for(size_t i=0;i<logit.size();i++) logit[i]=f2b(sinf((float)i*0.37f)*3.0f);
  for(unsigned e=0;e<n_exp;e++) bias[e]=cosf((float)e*0.11f)*0.4f;
  bf16* dl; float* db; unsigned char *dt_pf,*dt_1;
  HC(hipMalloc(&dl,logit.size()*2)); HC(hipMemcpy(dl,logit.data(),logit.size()*2,hipMemcpyHostToDevice));
  HC(hipMalloc(&db,n_exp*4)); HC(hipMemcpy(db,bias.data(),n_exp*4,hipMemcpyHostToDevice));
  HC(hipMalloc(&dt_pf,(size_t)T*k*8)); HC(hipMalloc(&dt_1,(size_t)T*k*8));
  std::vector<unsigned char> tpf((size_t)T*k*8), t1((size_t)T*k*8);

  printf("MoE PREFILL family (ops 83/84/87) + group-limited top-k, gfx950:\n");

  for (unsigned gcase=0; gcase<2; ++gcase){
    const unsigned ng = gcase? 8u:1u, tg = gcase? 4u:1u;
    HC(hipMemset(dt_pf,0,(size_t)T*k*8));
    hipLaunchKernelGGL(t_router_pf,dim3(64),dim3(PLOW_THREADS),0,0,dt_pf,dl,db,n_exp,k,flags,rs,T,ng,tg);
    HC(hipDeviceSynchronize());
    HC(hipMemcpy(tpf.data(),dt_pf,tpf.size(),hipMemcpyDeviceToHost));
    /* oracle A: the DECODE router, one launch per token */
    for(unsigned t=0;t<T;t++){
      hipLaunchKernelGGL(t_router_1,dim3(1),dim3(PLOW_THREADS),0,0,dt_1,dl+(size_t)t*n_exp,db,n_exp,k,flags,rs,ng,tg);
      HC(hipDeviceSynchronize());
      HC(hipMemcpy(t1.data()+(size_t)t*k*8,dt_1,k*8,hipMemcpyDeviceToHost));
    }
    char lbl[96]; snprintf(lbl,sizeof lbl,"ROUTER_TOPK_PF == decode router per token (n_group=%u)",ng);
    ck(lbl, memcmp(tpf.data(),t1.data(),tpf.size())==0);

    /* oracle B: an independent CPU noaux_tc implementation */
    unsigned bad=0;
    for(unsigned t=0;t<T;t++){
      std::vector<float> sc(n_exp), bs(n_exp);
      for(unsigned e=0;e<n_exp;e++){ sc[e]=1.f/(1.f+expf(-bf(logit[(size_t)t*n_exp+e]))); bs[e]=sc[e]+bias[e]; }
      std::vector<char> live(n_exp,1);
      if(ng>1){
        const unsigned gsz=n_exp/ng; std::vector<std::pair<float,unsigned>> gs(ng);
        for(unsigned g=0;g<ng;g++){ std::vector<float> v(bs.begin()+g*gsz,bs.begin()+(g+1)*gsz);
          std::partial_sort(v.begin(),v.begin()+2,v.end(),std::greater<float>()); gs[g]={v[0]+v[1],g}; }
        std::stable_sort(gs.begin(),gs.end(),[](auto&a,auto&b){return a.first>b.first;});
        std::fill(live.begin(),live.end(),0);
        for(unsigned i=0;i<tg;i++){unsigned g=gs[i].second; for(unsigned j=0;j<gsz;j++) live[g*gsz+j]=1;}
      }
      std::vector<std::pair<float,unsigned>> cand;
      for(unsigned e=0;e<n_exp;e++) if(live[e]) cand.push_back({bs[e],e});
      std::stable_sort(cand.begin(),cand.end(),[](auto&a,auto&b){return a.first>b.first;});
      float sum=0; for(unsigned j=0;j<k;j++) sum+=sc[cand[j].second];
      for(unsigned j=0;j<k;j++){
        unsigned gid; float gg; memcpy(&gid,tpf.data()+((size_t)t*k+j)*8,4);
        memcpy(&gg,tpf.data()+((size_t)t*k+j)*8+4,4);
        float want=sc[cand[j].second]/sum*rs;
        if(gid!=cand[j].second || fabsf(gg-want)>1e-5f*fmaxf(1.f,fabsf(want))) bad++;
      }
    }
    snprintf(lbl,sizeof lbl,"  ...and == an independent CPU noaux_tc (n_group=%u tg=%u)",ng,tg);
    ck(lbl, bad==0, bad?"":"");
  }

  /* ---- ALIGN_PF: structural, exhaustive ---- */
  {
    const unsigned pad = T*k + n_exp*(MPF_BM-1);
    int* dmeta; unsigned *drt,*drp; float* drg;
    HC(hipMalloc(&dmeta,(3*n_exp+1)*4)); HC(hipMalloc(&drt,pad*4));
    HC(hipMalloc(&drp,pad*4)); HC(hipMalloc(&drg,pad*4));
    hipLaunchKernelGGL(t_align,dim3(1),dim3(PLOW_THREADS),0,0,dmeta,dt_pf,drt,drp,drg,T,n_exp,k);
    HC(hipDeviceSynchronize());
    std::vector<int> meta(3*n_exp+1); std::vector<unsigned> rt(pad),rp(pad); std::vector<float> rg(pad);
    HC(hipMemcpy(meta.data(),dmeta,meta.size()*4,hipMemcpyDeviceToHost));
    HC(hipMemcpy(rt.data(),drt,pad*4,hipMemcpyDeviceToHost));
    HC(hipMemcpy(rp.data(),drp,pad*4,hipMemcpyDeviceToHost));
    HC(hipMemcpy(rg.data(),drg,pad*4,hipMemcpyDeviceToHost));
    std::vector<unsigned> cnt(n_exp,0);
    for(unsigned s=0;s<T*k;s++){unsigned e; memcpy(&e,tpf.data()+(size_t)s*8,4); if(e<n_exp) cnt[e]++;}
    bool ok=true; unsigned total=(unsigned)meta[3*n_exp]*MPF_BM;
    for(unsigned e=0;e<n_exp;e++) if((unsigned)meta[n_exp+e]!=cnt[e]) ok=false;
    std::vector<char> seen(T*k,0);
    for(unsigned r=0;r<total;r++){
      if(rp[r]==PLOW_EXPERT_UNUSED){ if(rt[r]!=PLOW_EXPERT_UNUSED) ok=false; continue; }
      if(rp[r]>=T*k || seen[rp[r]]) { ok=false; continue; }
      seen[rp[r]]=1;
      unsigned e; float g; memcpy(&e,tpf.data()+(size_t)rp[r]*8,4); memcpy(&g,tpf.data()+(size_t)rp[r]*8+4,4);
      if(rt[r]!=rp[r]/k || rg[r]!=g) ok=false;
      unsigned lo=(unsigned)meta[e], hi=lo+((cnt[e]+MPF_BM-1)/MPF_BM)*MPF_BM;
      if(r<lo||r>=hi) ok=false;                     /* row must sit in its OWN expert's range */
    }
    for(unsigned s=0;s<T*k;s++){unsigned e; memcpy(&e,tpf.data()+(size_t)s*8,4);
      if(e<n_exp && !seen[s]) ok=false;}            /* every live slot placed exactly once */
    ck("ALIGN_PF: counts, ranges, bijection, pad marking", ok);
  }

  /* ---- COMBINE_PF at T=1 == decode COMBINE, bit-exact ---- */
  {
    const unsigned H=1024;
    std::vector<unsigned short> res(H),sh(H); std::vector<float> part((size_t)k*H);
    for(unsigned h=0;h<H;h++){res[h]=f2b(sinf(h*0.1f)); sh[h]=f2b(cosf(h*0.2f));}
    for(size_t i=0;i<part.size();i++) part[i]=sinf((float)i*0.03f)*0.5f;
    bf16 *dr,*dsh,*o1,*o2; float* dp;
    HC(hipMalloc(&dr,H*2));HC(hipMalloc(&dsh,H*2));HC(hipMalloc(&o1,H*2));HC(hipMalloc(&o2,H*2));
    HC(hipMalloc(&dp,part.size()*4));
    HC(hipMemcpy(dr,res.data(),H*2,hipMemcpyHostToDevice));
    HC(hipMemcpy(dsh,sh.data(),H*2,hipMemcpyHostToDevice));
    HC(hipMemcpy(dp,part.data(),part.size()*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(t_comb_pf,dim3(64),dim3(PLOW_THREADS),0,0,o1,dr,dsh,dp,H,k,1u);
    hipLaunchKernelGGL(t_comb_1,dim3(64),dim3(PLOW_THREADS),0,0,o2,dr,dsh,dp,H,k);
    HC(hipDeviceSynchronize());
    std::vector<unsigned short> a(H),b(H);
    HC(hipMemcpy(a.data(),o1,H*2,hipMemcpyDeviceToHost));
    HC(hipMemcpy(b.data(),o2,H*2,hipMemcpyDeviceToHost));
    ck("COMBINE_PF at T=1 == decode COMBINE (bit-exact)", memcmp(a.data(),b.data(),H*2)==0);
  }

  /* ---- DECODE experts with MXFP4 weights (op 45, i[6]=2) -----------------------------------
   * The blocker for an all-MXFP4 asset: MXFP4 weights on disk ARE fp4 + E8M0, so op 45 run
   * against them with the block-fp8 body reads packed nibbles as bf16 and emits noise. This
   * checks the fp4 body against a CPU reference of the same math. Tolerance rather than memcmp
   * because wave_dot_mxfp4 reduces across 64 lanes and the reference sums serially -- the
   * SCALE fold is exact (E8M0 is a power of two), so anything beyond f32 reassociation here
   * would be a real decode error. */
  {
    const unsigned H2=2048, I2=256, NE=4, slot=1, act=1;
    std::vector<unsigned short> xv(H2);
    for(unsigned i=0;i<H2;i++) xv[i]=f2b(sinf((float)i*0.21f)*0.7f);
    std::vector<std::vector<unsigned char>> Wg(NE),Wu(NE),Sg(NE),Su(NE);
    for(unsigned e=0;e<NE;e++){
      Wg[e].resize((size_t)I2*H2/2); Wu[e].resize((size_t)I2*H2/2);
      Sg[e].resize((size_t)I2*H2/32); Su[e].resize((size_t)I2*H2/32);
      for(size_t i=0;i<Wg[e].size();i++){Wg[e][i]=(unsigned char)((i*13+e)&0x77);Wu[e][i]=(unsigned char)((i*7+e*3)&0x77);}
      for(size_t i=0;i<Sg[e].size();i++){Sg[e][i]=(unsigned char)(126+(i%3));Su[e][i]=(unsigned char)(127+(i%2));}
    }
    unsigned char tabh[64]={0}; const unsigned eid=2; float gval=1.0f;
    memcpy(tabh+slot*8,&eid,4); memcpy(tabh+slot*8+4,&gval,4);
    bf16* dx; unsigned char* dtab; bf16* dfu; unsigned long long *dwt2,*dst2;
    HC(hipMalloc(&dx,H2*2)); HC(hipMemcpy(dx,xv.data(),H2*2,hipMemcpyHostToDevice));
    HC(hipMalloc(&dtab,64)); HC(hipMemcpy(dtab,tabh,64,hipMemcpyHostToDevice));
    HC(hipMalloc(&dfu,(size_t)8*I2*2)); HC(hipMemset(dfu,0,(size_t)8*I2*2));
    std::vector<unsigned long long> wt2(NE*3,0),st2(NE*3,0);
    for(unsigned e=0;e<NE;e++){
      void *a,*b,*c,*d; HC(hipMalloc(&a,Wg[e].size()));HC(hipMalloc(&b,Wu[e].size()));
      HC(hipMalloc(&c,Sg[e].size()));HC(hipMalloc(&d,Su[e].size()));
      HC(hipMemcpy(a,Wg[e].data(),Wg[e].size(),hipMemcpyHostToDevice));
      HC(hipMemcpy(b,Wu[e].data(),Wu[e].size(),hipMemcpyHostToDevice));
      HC(hipMemcpy(c,Sg[e].data(),Sg[e].size(),hipMemcpyHostToDevice));
      HC(hipMemcpy(d,Su[e].data(),Su[e].size(),hipMemcpyHostToDevice));
      wt2[e*3]=(unsigned long long)a; wt2[e*3+1]=(unsigned long long)b;
      st2[e*3]=(unsigned long long)c; st2[e*3+1]=(unsigned long long)d;
    }
    HC(hipMalloc(&dwt2,wt2.size()*8)); HC(hipMemcpy(dwt2,wt2.data(),wt2.size()*8,hipMemcpyHostToDevice));
    HC(hipMalloc(&dst2,st2.size()*8)); HC(hipMemcpy(dst2,st2.data(),st2.size()*8,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(t_expert_glu_mx,dim3(32),dim3(PLOW_THREADS),0,0,dfu,dx,dtab,dwt2,dst2,slot,I2,H2,NE,act);
    HC(hipDeviceSynchronize());
    std::vector<unsigned short> got((size_t)8*I2);
    HC(hipMemcpy(got.data(),dfu,got.size()*2,hipMemcpyDeviceToHost));
    double worst=0;
    for(unsigned n=0;n<I2;n++){
      double g=0,u=0;
      for(unsigned k=0;k<H2;k++){
        unsigned gn=(Wg[eid][(size_t)n*H2/2+k/2]>>((k&1)*4))&0xF;
        unsigned un=(Wu[eid][(size_t)n*H2/2+k/2]>>((k&1)*4))&0xF;
        double xv2=bf(xv[k]);
        g+=xv2*fp4d(gn)*e8(Sg[eid][(size_t)n*H2/32+k/32]);
        u+=xv2*fp4d(un)*e8(Su[eid][(size_t)n*H2/32+k/32]);
      }
      double want=(g/(1.0+exp(-g)))*u;
      double gotv=bf(got[(size_t)slot*I2+n]);
      worst=fmax(worst, fabs(gotv-want)/(fabs(want)+1e-2));
    }
    char lbl[96]; snprintf(lbl,sizeof lbl,"EXPERT_GLU op45 enc=MXFP4 vs CPU (worst rel %.2e)",worst);
    ck(lbl, worst<2e-2);
  }

  printf("\n%s (%d failure%s)\n", fails?"MOE PREFILL FAILED":"MOE PREFILL CORRECT", fails, fails==1?"":"s");
  return fails?1:0;
}
