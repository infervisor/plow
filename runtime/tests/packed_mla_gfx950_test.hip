/* Packed-prefill MLA cache/flash and chunk-KDA consumers versus the same low-level kernels
 * launched once per span. The focused consumer-suite wrapper verifies the required markers on
 * the three shipping family objects, but executes this separately compiled oracle object. The
 * production end-to-end segmented operator-chain gate remains separate and open.
 *
 *   nix develop --command hipcc --offload-arch=gfx950 -O3 -w -DPACKED_MLA_DEVICE \
 *     --genco runtime/tests/packed_mla_gfx950_test.hip -o /tmp/packed_mla.co \
 *     -Iruntime/amd -Iruntime/common
 *   nix develop --command hipcc -O2 -w runtime/tests/packed_mla_gfx950_test.hip \
 *     -o /tmp/packed_mla -lamdhip64
 *   scripts/test_packed_prefill_consumers_gfx950.sh /path/to/packed-family-hsaco
 */

#ifdef PACKED_MLA_DEVICE
#define PLOW_PACKED_PREFILL_CONSUMERS 1
#define PLOW_MLA_PREFILL 1
#define PLOW_K3 1
#define PLOW_KDA_CHUNK 1
#define PLOW_BUCKET_DECODE 0
#include "interp.hip"

struct Args {
    unsigned long long rms, rope, x512, x64, gamma, cosb, sinb, pos;
    unsigned long long qa, qr, ckv, kr, kvlen, op, ml, spans, parked;
    unsigned row0, nrows, slot, kv0, nspans, rung, ctx, nhead;
};

struct KdaArgs {
    unsigned long long out, q, k, v, graw, braw, alog, dtb, state;
    unsigned long long gprefix, beta, aqk, ainv, w, u, spans, parked;
    unsigned row0, nrows, slot, flags, nspans, rung, h, d;
    float lb, scale;
};

static __device__ PlowProgram packed_program(const Args* a) {
    PlowProgram p{};
    p.prefill_spans = (const PlowPrefillSpan*)a->spans;
    p.prefill_parked = (const uint32_t*)a->parked;
    p.n_prefill_spans = a->nspans;
    p.n_prefill_rows = a->rung;
    return p;
}

extern "C" __global__ void packed_cache(Args* a) {
    __shared__ float part[PLOW_WAVES];
    PlowProgram p = packed_program(a);
    d_rmsnorm((bf16*)a->rms, (const bf16*)a->x512, (const bf16*)a->gamma, a->rung, 512,
              1e-5f, 0, blockIdx.x, gridDim.x, part, nullptr, nullptr, &p, a->ctx);
    __syncthreads();
    d_headnorm_rope<64, true>((bf16*)a->rope, (const bf16*)a->x64, nullptr,
                              (const float*)a->cosb, (const float*)a->sinb,
                              (const int*)a->pos, a->rung, 1, 0.0f, 0, a->ctx,
                              PLOW_KV_MASK_NONE, 1, blockIdx.x, gridDim.x, 0, &p, a->ctx);
}

extern "C" __global__ void serial_cache(Args* a) {
    __shared__ float part[PLOW_WAVES];
    d_rmsnorm((bf16*)a->rms + (size_t)a->slot * a->ctx * 512,
              (const bf16*)a->x512 + (size_t)a->row0 * 512, (const bf16*)a->gamma,
              a->nrows, 512, 1e-5f, a->kv0, blockIdx.x, gridDim.x, part);
    __syncthreads();
    d_headnorm_rope<64, true>((bf16*)a->rope + (size_t)a->slot * a->ctx * 64,
                              (const bf16*)a->x64 + (size_t)a->row0 * 64, nullptr,
                              (const float*)a->cosb, (const float*)a->sinb,
                              (const int*)a->pos + a->row0, a->nrows, 1, 0.0f, a->kv0,
                              a->ctx, PLOW_KV_MASK_NONE, 1, blockIdx.x, gridDim.x);
}

extern "C" __global__ void packed_flash(Args* a) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    PlowProgram p = packed_program(a);
    void* tensors[8] = {(void*)a->op, (void*)a->ml, (void*)a->qa, (void*)a->qr,
                        (void*)a->ckv, (void*)a->kr, (void*)a->kvlen, nullptr};
    PlowDevInst d{};
    d.op = PLOW_DOP_FLASH_MLA_PREFILL;
    for (unsigned i = 0; i < 7; ++i) d.t[i] = i;
    d.t[7] = PLOW_TENSOR_NONE;
    d.i[0] = 1; d.i[1] = a->nhead; d.i[2] = a->ctx; d.i[3] = 0;
    d.i[4] = a->rung; d.i[5] = PLOW_KV_MASK_NONE; d.i[7] = GLM_MLA_GF;
    d.fj[0].f = 0.0883883476f;
    exec_flash_mla_prefill((CInst)&d, (CTens)tensors, blockIdx.x, gridDim.x, lds, &p);
}

extern "C" __global__ void serial_flash(Args* a) {
    __shared__ float lds[FA_MLA_DEC_LDS_FLOATS(512, 64, GLM_MLA_GF)];
    d_flash_mla_prefill<512, 64, GLM_MLA_GF>(
        (float*)a->op + (size_t)a->row0 * a->nhead * 512,
        (float*)a->ml + (size_t)a->row0 * a->nhead * 2,
        (const bf16*)a->qa + (size_t)a->row0 * a->nhead * 512,
        (const bf16*)a->qr + (size_t)a->row0 * a->nhead * 64,
        (const bf16*)a->ckv + (size_t)a->slot * a->ctx * 512,
        (const bf16*)a->kr + (size_t)a->slot * a->ctx * 64,
        (const int*)a->kvlen + a->slot, 1, a->nrows, a->nhead, a->ctx, 0,
        0.0883883476f, PLOW_KV_MASK_NONE, blockIdx.x, gridDim.x, lds);
}

static __device__ PlowProgram packed_kda_program(const KdaArgs* a) {
    PlowProgram p{};
    p.prefill_spans = (const PlowPrefillSpan*)a->spans;
    p.prefill_parked = (const uint32_t*)a->parked;
    p.n_prefill_spans = a->nspans;
    p.n_prefill_rows = a->rung;
    return p;
}

extern "C" __global__ void packed_kda_prepare(KdaArgs* a) {
    PlowProgram p = packed_kda_program(a);
    d_kda_chunk_prepare_packed_bt64(
        (bf16*)a->q, (bf16*)a->k, (float*)a->gprefix, (float*)a->beta,
        (const bf16*)a->graw, (const bf16*)a->braw, (const float*)a->alog,
        (const float*)a->dtb, a->h, a->d, PLOW_KDA_GATE_LOWER_BOUND, a->lb, blockIdx.x,
        gridDim.x, &p);
}

extern "C" __global__ void packed_kda_intra(KdaArgs* a) {
    __shared__ float mat[64 * 64];
    PlowProgram p = packed_kda_program(a);
    d_kda_chunk_intra_packed_bt64(
        (float*)a->aqk, (float*)a->ainv, (const bf16*)a->q, (const bf16*)a->k,
        (const float*)a->gprefix, (const float*)a->beta, a->h, a->d, a->scale,
        blockIdx.x, gridDim.x, mat, &p);
}

extern "C" __global__ void packed_kda_wu(KdaArgs* a) {
    PlowProgram p = packed_kda_program(a);
    d_kda_chunk_wu_packed_bt64(
        (bf16*)a->w, (bf16*)a->u, nullptr, (const float*)a->ainv,
        (const bf16*)a->k, (const bf16*)a->v, (const float*)a->gprefix,
        (const float*)a->beta, a->h, a->d, a->d, a->scale, false, blockIdx.x,
        gridDim.x, &p);
}

extern "C" __global__ void packed_kda_carry(KdaArgs* a) {
    __shared__ float st[16 * 128];
    __shared__ bf16 vsm[64 * 16];
    __shared__ float osm[64 * 16];
    PlowProgram p = packed_kda_program(a);
    d_kda_chunk_carry_packed_bt64(
        (bf16*)a->out, (float*)a->state, (const bf16*)a->q, (const bf16*)a->k,
        (const bf16*)a->w, (const bf16*)a->u, (const float*)a->aqk,
        (const float*)a->gprefix, a->h, a->d, a->d, a->scale, blockIdx.x, gridDim.x, st,
        vsm, osm, &p);
}

extern "C" __global__ void serial_kda(KdaArgs* a) {
    __shared__ float lds[4 * 128 + 2 * PLOW_WAVES];
    const size_t rd = (size_t)a->row0 * a->h * a->d;
    float* state = (float*)a->state + (size_t)a->slot * a->h * a->d * a->d;
    d_kda_state_step_g(
        (bf16*)a->out + rd, (const bf16*)a->q + rd, (const bf16*)a->k + rd,
        (const bf16*)a->v + rd, (const bf16*)a->graw + rd,
        (const bf16*)a->braw + (size_t)a->row0 * a->h, (const float*)a->alog,
        (const float*)a->dtb, PLOW_KDA_GATE_LOWER_BOUND, a->lb, state, a->nrows, a->h,
        a->d, 8, PLOW_KDA_F_QK_L2NORM, a->scale, blockIdx.x, gridDim.x, lds, 0, nullptr);
}

#else
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef unsigned short bf16;
struct Span { unsigned row0, nrows, slot, flags, kv0, kvlen, state_slot, program; };
struct Args {
    unsigned long long rms, rope, x512, x64, gamma, cosb, sinb, pos;
    unsigned long long qa, qr, ckv, kr, kvlen, op, ml, spans, parked;
    unsigned row0, nrows, slot, kv0, nspans, rung, ctx, nhead;
};
struct KdaArgs {
    unsigned long long out, q, k, v, graw, braw, alog, dtb, state;
    unsigned long long gprefix, beta, aqk, ainv, w, u, spans, parked;
    unsigned row0, nrows, slot, flags, nspans, rung, h, d;
    float lb, scale;
};
#define HC(x) do { hipError_t e=(x); if(e){std::fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(e),__LINE__);return 1;} } while(0)
#define HC2(x) do { hipError_t e=(x); if(e){std::fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(e),__LINE__);std::exit(1);} } while(0)
static bf16 f2b(float f) { unsigned u; std::memcpy(&u,&f,4); u += 0x7fffu+((u>>16)&1u); return u>>16; }
static float b2f(bf16 b) { unsigned u=(unsigned)b<<16;float f;std::memcpy(&f,&u,4);return f; }
static void fill(std::vector<bf16>& v, unsigned seed) { for(auto& x:v){seed=seed*1664525u+1013904223u; x=f2b(((int)(seed>>9)%2048-1024)/1024.f);} }

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "/tmp/packed_mla.co";
    hipModule_t mod; HC(hipModuleLoad(&mod, co));
    hipFunction_t kp_cache, ks_cache, kp_flash, ks_flash;
    hipFunction_t kk_prepare, kk_intra, kk_wu, kk_carry, ks_kda;
    HC(hipModuleGetFunction(&kp_cache,mod,"packed_cache")); HC(hipModuleGetFunction(&ks_cache,mod,"serial_cache"));
    HC(hipModuleGetFunction(&kp_flash,mod,"packed_flash")); HC(hipModuleGetFunction(&ks_flash,mod,"serial_flash"));
    HC(hipModuleGetFunction(&kk_prepare,mod,"packed_kda_prepare"));
    HC(hipModuleGetFunction(&kk_intra,mod,"packed_kda_intra"));
    HC(hipModuleGetFunction(&kk_wu,mod,"packed_kda_wu"));
    HC(hipModuleGetFunction(&kk_carry,mod,"packed_kda_carry"));
    HC(hipModuleGetFunction(&ks_kda,mod,"serial_kda"));
    const unsigned R=85,C=128,NH=1; const size_t S=3;
    auto run_case = [&](const char* name, std::vector<Span> spans,
                        std::vector<unsigned> parked) -> bool {
    size_t active=0; for(const auto&s:spans)active+=s.nrows;
    std::vector<bf16> x512(R*512),x64(R*64),gamma(512),qa(R*NH*512),qr(R*NH*64);
    fill(x512,1);fill(x64,2);fill(gamma,3);fill(qa,4);fill(qr,5);
    std::vector<float> cs(C*32,1.f), sn(C*32,0.f); std::vector<int> pos(R,0);
    std::vector<int> lens(S,1);
    for(const auto&s:spans){lens[s.slot]=std::max(lens[s.slot],(int)(s.kv0+s.nrows));
        for(unsigned t=0;t<s.nrows;++t)pos[s.row0+t]=s.kv0+t;}
    const bf16 poison=f2b(6.f); const float fpoison=12345.f;
    std::vector<bf16> cache0(S*C*512,poison), rope0(S*C*64,poison);
    std::vector<float> op0(R*NH*512,fpoison),ml0(R*NH*2,fpoison);
    auto up=[&](const void* h,size_t n){void*d=nullptr;HC2(hipMalloc(&d,n));HC2(hipMemcpy(d,h,n,hipMemcpyHostToDevice));return d;};
    void *dx512=up(x512.data(),x512.size()*2),*dx64=up(x64.data(),x64.size()*2),*dg=up(gamma.data(),gamma.size()*2);
    void *dcs=up(cs.data(),cs.size()*4),*dsn=up(sn.data(),sn.size()*4),*dpos=up(pos.data(),pos.size()*4);
    void *dqa=up(qa.data(),qa.size()*2),*dqr=up(qr.data(),qr.size()*2),*dlens=up(lens.data(),lens.size()*4);
    void *dsp=up(spans.data(),spans.size()*sizeof(Span)),*dpark=up(parked.data(),parked.size()*4);
    auto buffers=[&](void*&r,void*&k,void*&o,void*&m){r=up(cache0.data(),cache0.size()*2);k=up(rope0.data(),rope0.size()*2);o=up(op0.data(),op0.size()*4);m=up(ml0.data(),ml0.size()*4);};
    void *pr,*pk,*po,*pm,*sr,*sk,*so,*sm; buffers(pr,pk,po,pm);buffers(sr,sk,so,sm);
    Args base{}; base.x512=(size_t)dx512;base.x64=(size_t)dx64;base.gamma=(size_t)dg;base.cosb=(size_t)dcs;base.sinb=(size_t)dsn;base.pos=(size_t)dpos;
    base.qa=(size_t)dqa;base.qr=(size_t)dqr;base.kvlen=(size_t)dlens;base.spans=(size_t)dsp;base.parked=(size_t)dpark;base.nspans=spans.size();base.rung=R;base.ctx=C;base.nhead=NH;
    auto launch=[&](hipFunction_t f,Args a){void*da=up(&a,sizeof a);void*av[]={&da};HC(hipModuleLaunchKernel(f,64*32,1,1,512,1,1,0,0,av,nullptr));HC(hipDeviceSynchronize());return 0;};
    Args a=base;a.rms=(size_t)pr;a.rope=(size_t)pk; if(launch(kp_cache,a))return false;
    for(const auto&s:spans){a=base;a.rms=(size_t)sr;a.rope=(size_t)sk;a.row0=s.row0;a.nrows=s.nrows;a.slot=s.slot;a.kv0=s.kv0;if(launch(ks_cache,a))return false;}
    a=base;a.op=(size_t)po;a.ml=(size_t)pm;a.ckv=(size_t)pr;a.kr=(size_t)pk;if(launch(kp_flash,a))return false;
    for(const auto&s:spans){a=base;a.op=(size_t)so;a.ml=(size_t)sm;a.ckv=(size_t)sr;a.kr=(size_t)sk;a.row0=s.row0;a.nrows=s.nrows;a.slot=s.slot;a.kv0=s.kv0;if(launch(ks_flash,a))return false;}
    auto get=[&](void*d,void*h,size_t n){HC(hipMemcpy(h,d,n,hipMemcpyDeviceToHost));return 0;};
    std::vector<bf16> hpr(cache0.size()),hsr(cache0.size()),hpk(rope0.size()),hsk(rope0.size());
    std::vector<float> hpo(op0.size()),hso(op0.size()),hpm(ml0.size()),hsm(ml0.size());
    if(get(pr,hpr.data(),hpr.size()*2)||get(sr,hsr.data(),hsr.size()*2)||get(pk,hpk.data(),hpk.size()*2)||get(sk,hsk.data(),hsk.size()*2)||get(po,hpo.data(),hpo.size()*4)||get(so,hso.data(),hso.size()*4)||get(pm,hpm.data(),hpm.size()*4)||get(sm,hsm.data(),hsm.size()*4))return false;
    bool ok=hpr==hsr&&hpk==hsk; double md=0;
    for(size_t i=0;i<hpo.size();++i){if(i>=active*NH*512)ok&=hpo[i]==fpoison&&hso[i]==fpoison;else md=fmax(md,fabs(hpo[i]-hso[i]));}
    for(size_t i=0;i<hpm.size();++i){if(i>=active*NH*2)ok&=hpm[i]==fpoison&&hsm[i]==fpoison;else md=fmax(md,fabs(hpm[i]-hsm[i]));}
    for(size_t slot=0;slot<S;++slot)for(size_t row=0;row<C;++row){
        bool written=false;for(const auto&s:spans)written|=slot==s.slot&&row>=s.kv0&&row<s.kv0+s.nrows;
        if(!written){for(size_t d=0;d<512;++d)ok&=hpr[(slot*C+row)*512+d]==poison&&hsr[(slot*C+row)*512+d]==poison;
                     for(size_t d=0;d<64;++d)ok&=hpk[(slot*C+row)*64+d]==poison&&hsk[(slot*C+row)*64+d]==poison;}
    }
    ok &= md < 1e-6;
    const unsigned KH=2,KD=128; const size_t krow=(size_t)KH*KD,kstate=krow*KD;
    std::vector<bf16> kq(R*krow),kk(kq.size()),kv(kq.size()),kg(kq.size()),kb((size_t)R*KH);
    fill(kq,11);fill(kk,12);fill(kv,13);fill(kg,14);fill(kb,15);
    std::vector<float> ka(KH),kdt(krow);for(unsigned h=0;h<KH;++h)ka[h]=std::log(.5f+h);
    for(size_t i=0;i<kdt.size();++i)kdt[i]=-6.f+(i%17)/17.f;
    std::vector<float> state0(S*kstate);for(size_t i=0;i<state0.size();++i)state0[i]=.25f+(i%89)/127.f;
    std::vector<bf16> kout0(R*krow,poison);std::vector<float> gp0(kq.size(),fpoison),bt0(R*KH,fpoison),
        mat0((size_t)R*KH*64,fpoison);std::vector<bf16> wu0(kq.size(),poison);
    void *dqs=up(kq.data(),kq.size()*2),*dks=up(kk.data(),kk.size()*2);
    void *dqc=up(kq.data(),kq.size()*2),*dkc=up(kk.data(),kk.size()*2),*dkv=up(kv.data(),kv.size()*2);
    void *dkg=up(kg.data(),kg.size()*2),*dkb=up(kb.data(),kb.size()*2),*dka=up(ka.data(),ka.size()*4),*dkdt=up(kdt.data(),kdt.size()*4);
    void *dss=up(state0.data(),state0.size()*4),*dsc=up(state0.data(),state0.size()*4);
    void *dkos=up(kout0.data(),kout0.size()*2),*dkoc=up(kout0.data(),kout0.size()*2);
    void *dgp=up(gp0.data(),gp0.size()*4),*dbt=up(bt0.data(),bt0.size()*4);
    void *daq=up(mat0.data(),mat0.size()*4),*dai=up(mat0.data(),mat0.size()*4);
    void *dkw=up(wu0.data(),wu0.size()*2),*dku=up(wu0.data(),wu0.size()*2);
    auto klaunch=[&](hipFunction_t f,KdaArgs a){void*da=up(&a,sizeof a);void*av[]={&da};HC(hipModuleLaunchKernel(f,37,1,1,512,1,1,0,0,av,nullptr));HC(hipDeviceSynchronize());HC(hipFree(da));return 0;};
    KdaArgs ks{};ks.out=(size_t)dkos;ks.q=(size_t)dqs;ks.k=(size_t)dks;ks.v=(size_t)dkv;
    ks.graw=(size_t)dkg;ks.braw=(size_t)dkb;ks.alog=(size_t)dka;ks.dtb=(size_t)dkdt;
    ks.state=(size_t)dss;ks.h=KH;ks.d=KD;ks.lb=-5.f;ks.scale=1.f/std::sqrt((float)KD);
    for(const auto&s:spans){if(s.flags&1)HC2(hipMemset((char*)dss+(size_t)s.state_slot*kstate*4,0,kstate*4));
        ks.row0=s.row0;ks.nrows=s.nrows;ks.slot=s.state_slot;if(klaunch(ks_kda,ks))return false;}
    KdaArgs kc=ks;kc.out=(size_t)dkoc;kc.q=(size_t)dqc;kc.k=(size_t)dkc;kc.state=(size_t)dsc;
    kc.gprefix=(size_t)dgp;kc.beta=(size_t)dbt;kc.aqk=(size_t)daq;kc.ainv=(size_t)dai;
    kc.w=(size_t)dkw;kc.u=(size_t)dku;kc.spans=(size_t)dsp;kc.parked=(size_t)dpark;
    kc.nspans=spans.size();kc.rung=R;
    for(hipFunction_t f:{kk_prepare,kk_intra,kk_wu,kk_carry})if(klaunch(f,kc))return false;
    std::vector<bf16> kos(kout0.size()),koc(kout0.size()),kqc(kq.size()),kkc(kk.size());
    std::vector<float> sgs(state0.size()),sgc(state0.size()),gpc(gp0.size());
    if(get(dkos,kos.data(),kos.size()*2)||get(dkoc,koc.data(),koc.size()*2)||
       get(dss,sgs.data(),sgs.size()*4)||get(dsc,sgc.data(),sgc.size()*4)||
       get(dqc,kqc.data(),kqc.size()*2)||get(dkc,kkc.data(),kkc.size()*2)||
       get(dgp,gpc.data(),gpc.size()*4))return false;
    double koe=0,kow=0,kse=0,ksw=0;for(size_t i=0;i<active*krow;++i){double d=(double)b2f(koc[i])-b2f(kos[i]);koe+=d*d;kow+=(double)b2f(kos[i])*b2f(kos[i]);}
    for(size_t i=0;i<sgc.size();++i){double d=sgc[i]-sgs[i];kse+=d*d;ksw+=(double)sgs[i]*sgs[i];}
    bool kpoison=true;for(size_t i=kstate;i<2*kstate;++i)kpoison&=sgc[i]==state0[i];
    for(size_t i=active*krow;i<kout0.size();++i)kpoison&=koc[i]==poison&&kqc[i]==kq[i]&&kkc[i]==kk[i]&&gpc[i]==fpoison;
    const double korms=std::sqrt(koe/(kow+1e-30)),ksrms=std::sqrt(kse/(ksw+1e-30));
    ok&=korms<1e-2&&ksrms<5e-3&&kpoison;
    std::printf("%-10s cache=%s flash=%.3e kda_out=%.3e kda_state=%.3e poison=%s\n",name,hpr==hsr&&hpk==hsk?"exact":"FAIL",md,korms,ksrms,ok?"untouched":"FAIL");
    return ok;
    };
    std::vector<unsigned> single_park(R,1);single_park[0]=single_park[1]=0;
    bool ok=run_case("single",{{0,2,2,1,3,5,2,0}},single_park);
    std::vector<unsigned> ragged_park(R,1);std::fill(ragged_park.begin(),ragged_park.begin()+82,0);
    ok&=run_case("ragged",{{0,65,2,1,3,68,2,0},{65,17,0,0,7,24,0,0}},ragged_park);
    std::puts(ok?"PASS":"FAIL"); return ok?0:1;
}
#endif
