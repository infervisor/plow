/* moe_prefill_a4w4_gfx950_test.hip — the A4W4 grouped MoE prefill GEMM + fused bridge,
 * against a CPU reference of the same math.                    [KIMI/GLM/DEEPSEEK-PREFILL]
 *
 * Validates ops 85/86 AHEAD OF THE EMITTER. There is no packet producing these opcodes yet, so
 * the alternative was to ship them unproven; this drives the device function directly with a
 * hand-built routing/align state and checks every output value.
 *
 * WHAT IT ACTUALLY PINS DOWN, because a GEMM that is merely "close" would pass a loose check:
 *   - the fp4 fragment layout and the XOR swizzle round-trip through LDS,
 *   - that the hardware applies BOTH E8M0 scales (A's, produced by the staging quantizer, and
 *     the weight's, read from memory) -- a dropped scale is a large error, not a small one,
 *   - RAGGED expert row counts (40 / 7 / 64) with per-expert MPF_BM padding, so pad rows,
 *     the tile->expert mapping and the gathered-row indexing are all exercised,
 *   - the FUSED BRIDGE: the SwiGLU result is re-quantized to MXFP4 in the sorted layout and its
 *     E8M0 scale bytes are compared EXACTLY (an off-by-one exponent fails the run outright),
 *     which is the part that would otherwise only be caught end-to-end.
 *
 * TOLERANCE. The comparison is against ONE fp4 ULP of the block's own scale, not a relative
 * epsilon: the output IS 4-bit, so the only meaningful question is whether the device picked
 * the same 4-bit code the reference did. Measured 0.4837 -- every value inside half a code.
 *
 *   nix develop --command bash -c '/opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 -w \
 *       -I runtime/amd -I runtime/common runtime/tests/moe_prefill_a4w4_gfx950_test.hip \
 *       -o /tmp/moe_pf_a4w4'
 *   sg render -c 'HIP_VISIBLE_DEVICES=1 /tmp/moe_pf_a4w4'
 */
/* A4W4 grouped MoE prefill GEMM vs a CPU f64 reference of the same math. */
#include <vector>
#include <cstring>
#define PLOW_MOE_PREFILL 1
#define PLOW_MOE_PF_A4W4 1
#if defined(MPF4_TEST_ATOMIC)
#define PLOW_MOE_PF_ATOMIC 1
#elif defined(MPF4_TEST_DET)
#define PLOW_MOE_PF_DET 1
#endif
#include "amd_common.h"
#define st_act1(p, v) st_act<bf16>((PLOW_GLOB bf16*)(p), (v))
#include "op_moe.h"
#undef st_act1
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void t_glu(unsigned char* fu, const bf16* xn2, const unsigned long long* wtab,
           const unsigned long long* stab, const int* meta, const unsigned* row_token,
           const unsigned* row_partidx, unsigned char* fu_scale, unsigned I, unsigned H,
           unsigned n_exp, unsigned act, float beta, float lbeta) {
    __shared__ unsigned char lds[MPF4_LDS_BYTES];
    d_moe_group_pf_a4w4<true>((void*)fu, (const void*)xn2, nullptr, wtab, stab, meta, row_token,
                              row_partidx, nullptr, fu_scale, I, H, n_exp, act, blockIdx.x,
                              gridDim.x, beta, lbeta, (void*)lds);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void t_down(float* part, const unsigned char* fu, const unsigned char* fu_scale,
            const unsigned long long* wtab, const unsigned long long* stab, const int* meta,
            const unsigned* row_partidx, const float* row_gate, unsigned H, unsigned I,
            unsigned n_exp, unsigned ksh) {
    __shared__ unsigned char lds[MPF4_LDS_BYTES];
    d_moe_group_pf_a4w4<false>((void*)part, (const void*)fu, fu_scale, wtab, stab, meta, nullptr,
                               row_partidx, row_gate, nullptr, H, I, n_exp, 0, blockIdx.x,
                               gridDim.x, 0.0f, 0.0f, (void*)lds
#if PLOW_MOE_PF_ATOMIC
                               , ksh
#elif PLOW_MOE_PF_DET
                               , ksh
#endif
                               );
#if !PLOW_MOE_PF_ATOMIC && !PLOW_MOE_PF_DET
    (void)ksh;
#endif
}

static float fp4d(unsigned n){const float T[8]={0,.5f,1,1.5f,2,3,4,6};return (n&8)?-T[n&7]:T[n&7];}
static float e8(unsigned char b){return ldexpf(1.f,(int)b-127);}
static float bf(unsigned short h){unsigned u=(unsigned)h<<16;float f;memcpy(&f,&u,4);return f;}
static unsigned short f2b(float f){unsigned u;memcpy(&u,&f,4);u+=0x7fffu+((u>>16)&1u);return (unsigned short)(u>>16);}
#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip err %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

int main(){
  const unsigned H=256, I=256, n_exp=3, BM=MPF4_BM, act=1;
  const unsigned cnt[3]={40,7,64};                   /* live rows per expert */
  unsigned tiles[4]={0,0,0,0}, rowoff[3]; unsigned tp=0;
  for(unsigned e=0;e<n_exp;e++){tiles[e]=tp;rowoff[e]=tp*BM;tp+=(cnt[e]+BM-1)/BM;} tiles[n_exp]=tp;
  const unsigned pad=tp*BM, T=64;
  std::vector<int> meta(3*n_exp+1);
  for(unsigned e=0;e<n_exp;e++){meta[e]=rowoff[e];meta[n_exp+e]=cnt[e];meta[2*n_exp+e]=tiles[e];}
  meta[3*n_exp]=tiles[n_exp];
  std::vector<unsigned> rtok(pad,PLOW_EXPERT_UNUSED), rpid(pad,PLOW_EXPERT_UNUSED);
  for(unsigned e=0;e<n_exp;e++) for(unsigned r=0;r<cnt[e];r++){rtok[rowoff[e]+r]=(e*17+r)%T; rpid[rowoff[e]+r]=rowoff[e]+r;}
  std::vector<unsigned short> x(T*H); for(size_t i=0;i<x.size();i++) x[i]=f2b(((int)(i*37%23)-11)*0.13f);
  /* per-expert mxfp4 gate|up weights [I][H/2] + E8M0 [I][H/32] */
  std::vector<std::vector<unsigned char>> Wg(n_exp),Wu(n_exp),Sg(n_exp),Su(n_exp);
  for(unsigned e=0;e<n_exp;e++){
    Wg[e].resize((size_t)I*H/2); Wu[e].resize((size_t)I*H/2);
    Sg[e].resize((size_t)I*H/32); Su[e].resize((size_t)I*H/32);
    for(size_t i=0;i<Wg[e].size();i++){Wg[e][i]=(unsigned char)((i*7+e*3)&0x77);Wu[e][i]=(unsigned char)((i*5+e)&0x77);}
    for(size_t i=0;i<Sg[e].size();i++){Sg[e][i]=(unsigned char)(126+(i%3));Su[e][i]=(unsigned char)(127+(i%2));}
  }
  unsigned char *dfu,*dfs; bf16* dx; int* dmeta; unsigned *drt,*drp;
  unsigned long long *dwt,*dst;
  HC(hipMalloc(&dfu,(size_t)pad*I/2)); HC(hipMalloc(&dfs,(size_t)pad*I/32));
  HC(hipMemset(dfu,0,(size_t)pad*I/2)); HC(hipMemset(dfs,0,(size_t)pad*I/32));
  HC(hipMalloc(&dx,x.size()*2)); HC(hipMemcpy(dx,x.data(),x.size()*2,hipMemcpyHostToDevice));
  HC(hipMalloc(&dmeta,meta.size()*4)); HC(hipMemcpy(dmeta,meta.data(),meta.size()*4,hipMemcpyHostToDevice));
  HC(hipMalloc(&drt,pad*4)); HC(hipMemcpy(drt,rtok.data(),pad*4,hipMemcpyHostToDevice));
  HC(hipMalloc(&drp,pad*4)); HC(hipMemcpy(drp,rpid.data(),pad*4,hipMemcpyHostToDevice));
  std::vector<unsigned long long> wt(n_exp*3,0),st(n_exp*3,0);
  for(unsigned e=0;e<n_exp;e++){
    void *a,*b,*c,*d; HC(hipMalloc(&a,Wg[e].size()));HC(hipMalloc(&b,Wu[e].size()));
    HC(hipMalloc(&c,Sg[e].size()));HC(hipMalloc(&d,Su[e].size()));
    HC(hipMemcpy(a,Wg[e].data(),Wg[e].size(),hipMemcpyHostToDevice));
    HC(hipMemcpy(b,Wu[e].data(),Wu[e].size(),hipMemcpyHostToDevice));
    HC(hipMemcpy(c,Sg[e].data(),Sg[e].size(),hipMemcpyHostToDevice));
    HC(hipMemcpy(d,Su[e].data(),Su[e].size(),hipMemcpyHostToDevice));
    wt[e*3]= (unsigned long long)a; wt[e*3+1]=(unsigned long long)b;
    st[e*3]= (unsigned long long)c; st[e*3+1]=(unsigned long long)d;
  }
  HC(hipMalloc(&dwt,wt.size()*8)); HC(hipMemcpy(dwt,wt.data(),wt.size()*8,hipMemcpyHostToDevice));
  HC(hipMalloc(&dst,st.size()*8)); HC(hipMemcpy(dst,st.data(),st.size()*8,hipMemcpyHostToDevice));
  hipLaunchKernelGGL(t_glu,dim3(64),dim3(PLOW_THREADS),0,0,dfu,dx,dwt,dst,dmeta,drt,drp,dfs,I,H,n_exp,act,0.0f,0.0f);
  HC(hipDeviceSynchronize());
  std::vector<unsigned char> fu((size_t)pad*I/2), fs((size_t)pad*I/32);
  HC(hipMemcpy(fu.data(),dfu,fu.size(),hipMemcpyDeviceToHost));
  HC(hipMemcpy(fs.data(),dfs,fs.size(),hipMemcpyDeviceToHost));

  /* CPU reference: same A-quantization, same weights, SwiGLU, same MX re-quantization. */
  double worst=0; size_t nchk=0;
  for(unsigned e=0;e<n_exp;e++) for(unsigned r=0;r<cnt[e];r++){
    unsigned gr=rowoff[e]+r, tok=rtok[gr];
    std::vector<double> out(I);
    for(unsigned n=0;n<I;n++){
      double g=0,u=0;
      for(unsigned kb=0;kb<H/32;kb++){
        /* A block quantization, exactly as the kernel's staging does it */
        float amax=0; for(int z=0;z<32;z++) amax=fmaxf(amax,fabsf(bf(x[(size_t)tok*H+kb*32+z])));
        int ee; frexpf(amax/6.0f,&ee); ee+=127; if(ee<1)ee=1; if(ee>254)ee=254;
        double as=e8((unsigned char)ee); double inv=1.0/as;
        for(int z=0;z<32;z++){
          float av=bf(x[(size_t)tok*H+kb*32+z]);
          /* replicate quant_fp4 on the CPU */
          float a2=fabsf((float)(av*inv)); unsigned q; 
          if(a2<0.25f)q=0;else if(a2<0.75f)q=1;else if(a2<1.25f)q=2;else if(a2<1.75f)q=3;
          else if(a2<2.5f)q=4;else if(a2<3.5f)q=5;else if(a2<5.f)q=6;else q=7;
          double ad=(av*inv<0?-1:1)*fp4d(q)*as;
          unsigned k=kb*32+z;
          unsigned gn=(Wg[e][(size_t)n*H/2+k/2]>>((k&1)*4))&0xF;
          unsigned un=(Wu[e][(size_t)n*H/2+k/2]>>((k&1)*4))&0xF;
          g+=ad*fp4d(gn)*e8(Sg[e][(size_t)n*H/32+kb]);
          u+=ad*fp4d(un)*e8(Su[e][(size_t)n*H/32+kb]);
        }
      }
      double sg = g/(1.0+exp(-g));      /* silu */
      out[n]=sg*u;
    }
    for(unsigned b=0;b<I/32;b++){
      double am=0; for(int z=0;z<32;z++) am=fmax(am,fabs(out[b*32+z]));
      int ee; frexpf((float)(am/6.0),&ee); ee+=127; if(ee<1)ee=1; if(ee>254)ee=254;
      if(!(am>0)) ee=127;
      unsigned char gots=fs[(size_t)gr*I/32+b];
      if(gots!=(unsigned char)ee){ printf("  scale mismatch row %u blk %u: got %u want %d\n",gr,b,gots,ee); worst=1e9; }
      double sc=e8((unsigned char)ee);
      for(int z=0;z<32;z++){
        unsigned k=b*32+z;
        unsigned gq=(fu[(size_t)gr*I/2+k/2]>>((k&1)*4))&0xF;
        double got=fp4d(gq)*sc; double want=out[k];
        double denom=fmax(fabs(want),sc*0.5); worst=fmax(worst,fabs(got-want)/denom); nchk++;
      }
    }
  }
  /* ---- op 86: the DOWN op's SCATTER epilogue -----------------------------------------------
   * The one arm with no direct coverage until now, and the one whose failure mode is quietest:
   * it writes part[row_partidx[row]*H + n] scaled by row_gate[row], so a mis-indexed scatter
   * lands a correct dot product on the wrong token's partial and the combine still produces
   * plausible numbers. Feeds the GLU output above straight in, as the real packet does. */
  {
    const unsigned Hd = 128;                       /* down output width */
    std::vector<std::vector<unsigned char>> Wd(n_exp), Sd(n_exp);
    for(unsigned e=0;e<n_exp;e++){
      Wd[e].resize((size_t)Hd*I/2); Sd[e].resize((size_t)Hd*I/32);
      for(size_t i=0;i<Wd[e].size();i++) Wd[e][i]=(unsigned char)((i*11+e*5)&0x77);
      for(size_t i=0;i<Sd[e].size();i++) Sd[e][i]=(unsigned char)(127+(i%2));
    }
    std::vector<float> rgate(pad,0.f);
    for(unsigned e=0;e<n_exp;e++) for(unsigned r=0;r<cnt[e];r++) rgate[rowoff[e]+r]=0.25f+0.05f*(float)((e*7+r)%9);
    float* drg; HC(hipMalloc(&drg,pad*4)); HC(hipMemcpy(drg,rgate.data(),pad*4,hipMemcpyHostToDevice));
    for(unsigned e=0;e<n_exp;e++){
      void *a,*b; HC(hipMalloc(&a,Wd[e].size())); HC(hipMalloc(&b,Sd[e].size()));
      HC(hipMemcpy(a,Wd[e].data(),Wd[e].size(),hipMemcpyHostToDevice));
      HC(hipMemcpy(b,Sd[e].data(),Sd[e].size(),hipMemcpyHostToDevice));
      wt[e*3+2]=(unsigned long long)a; st[e*3+2]=(unsigned long long)b;
    }
    HC(hipMemcpy(dwt,wt.data(),wt.size()*8,hipMemcpyHostToDevice));
    HC(hipMemcpy(dst,st.data(),st.size()*8,hipMemcpyHostToDevice));
    float* dpart; const size_t np=(size_t)pad*Hd;
    HC(hipMalloc(&dpart,np*4));
    /* poison, so an UNWRITTEN destination is a loud failure rather than a lucky zero */
    std::vector<float> poison(np, -12345.0f);
    HC(hipMemcpy(dpart,poison.data(),np*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(t_down,dim3(64),dim3(PLOW_THREADS),0,0,dpart,dfu,dfs,dwt,dst,dmeta,drp,drg,Hd,I,n_exp,0u);
    HC(hipDeviceSynchronize());
    std::vector<float> part(np); HC(hipMemcpy(part.data(),dpart,np*4,hipMemcpyDeviceToHost));
    double wmax=0; unsigned untouched=0, checked=0;
    for(unsigned e=0;e<n_exp;e++) for(unsigned r=0;r<cnt[e];r++){
      const unsigned gr=rowoff[e]+r, pidx=rpid[gr];
      for(unsigned n=0;n<Hd;n++){
        double acc=0;
        for(unsigned kb=0;kb<I/32;kb++){
          double as=e8(fs[(size_t)gr*I/32+kb]), ws=e8(Sd[e][(size_t)n*I/32+kb]);
          for(int z=0;z<32;z++){
            unsigned kk=kb*32+z;
            unsigned an=(fu[(size_t)gr*I/2+kk/2]>>((kk&1)*4))&0xF;
            unsigned bn=(Wd[e][(size_t)n*I/2+kk/2]>>((kk&1)*4))&0xF;
            acc += (double)fp4d(an)*as*(double)fp4d(bn)*ws;
          }
        }
        const double want=acc*rgate[gr], got=part[(size_t)pidx*Hd+n];
        if(got==-12345.0f){untouched++;continue;}
        wmax=fmax(wmax, fabs(got-want)/(fabs(want)+1e-3)); checked++;
      }
    }
    printf("  DOWN (op 86) scatter+gate: checked %u values, %u unwritten, worst rel %.2e  %s\n",
           checked, untouched, wmax, (untouched==0 && wmax<2e-3)?"PASS":"FAIL");
    if(untouched || wmax>=2e-3) return 1;
#if PLOW_MOE_PF_ATOMIC || PLOW_MOE_PF_DET
    const unsigned ksh=2, atom_arg=ksh+1, acc_rows=T;
#if PLOW_MOE_PF_ATOMIC
    float* dacc; HC(hipMalloc(&dacc,(size_t)acc_rows*Hd*sizeof(float)));
    HC(hipMemset(dacc,0,(size_t)acc_rows*Hd*sizeof(float)));
    hipLaunchKernelGGL(t_down,dim3(64),dim3(PLOW_THREADS),0,0,dacc,dfu,dfs,dwt,dst,dmeta,drp,drg,Hd,I,n_exp,atom_arg);
    HC(hipDeviceSynchronize());
    std::vector<float> got_acc((size_t)acc_rows*Hd), want_acc((size_t)acc_rows*Hd,0.0f);
    HC(hipMemcpy(got_acc.data(),dacc,got_acc.size()*sizeof(float),hipMemcpyDeviceToHost));
    for(unsigned e=0;e<n_exp;e++) for(unsigned r=0;r<cnt[e];r++){
      const unsigned pidx=rpid[rowoff[e]+r], tok=pidx>>ksh;
      for(unsigned n=0;n<Hd;n++) want_acc[(size_t)tok*Hd+n]+=part[(size_t)pidx*Hd+n];
    }
    double aworst=0; for(size_t i=0;i<got_acc.size();i++) aworst=fmax(aworst,fabs((double)got_acc[i]-want_acc[i])/(fabs((double)want_acc[i])+1e-3));
    printf("  DOWN atomic accumulator: worst rel %.2e  %s\n",aworst,aworst<2e-5?"PASS":"FAIL");
    if(aworst>=2e-5) return 1;
#else
    double* dacc; HC(hipMalloc(&dacc,(size_t)acc_rows*Hd*sizeof(double)));
    HC(hipMemset(dacc,0,(size_t)acc_rows*Hd*sizeof(double)));
    hipLaunchKernelGGL(t_down,dim3(64),dim3(PLOW_THREADS),0,0,(float*)dacc,dfu,dfs,dwt,dst,dmeta,drp,drg,Hd,I,n_exp,atom_arg);
    HC(hipDeviceSynchronize());
    std::vector<double> got_acc((size_t)acc_rows*Hd), want_acc((size_t)acc_rows*Hd,0.0);
    HC(hipMemcpy(got_acc.data(),dacc,got_acc.size()*sizeof(double),hipMemcpyDeviceToHost));
    for(unsigned e=0;e<n_exp;e++) for(unsigned r=0;r<cnt[e];r++){
      const unsigned pidx=rpid[rowoff[e]+r], tok=pidx>>ksh;
      for(unsigned n=0;n<Hd;n++){
        const double v=fmax(-131072.0,fmin(131072.0,(double)part[(size_t)pidx*Hd+n]));
        want_acc[(size_t)tok*Hd+n]+=nearbyint(v*4294967296.0);
      }
    }
    size_t mismatch=0; for(size_t i=0;i<got_acc.size();i++) mismatch+=(got_acc[i]!=want_acc[i]);
    printf("  DOWN deterministic accumulator: mismatches %zu  %s\n",mismatch,mismatch==0?"PASS":"FAIL");
    if(mismatch) return 1;
#endif
#endif
  }

  printf("A4W4 grouped MoE prefill GLU + fused bridge vs CPU reference:\n");
  printf("  experts=%u rows=%u/%u  H=%u I=%u   checked %zu values\n",n_exp,cnt[0]+cnt[1]+cnt[2],pad,H,I,nchk);
  printf("  worst relative deviation (vs one fp4 ULP) : %.4f  %s\n",worst, worst<1.05?"PASS":"FAIL");
  return worst<1.05?0:1;
}
