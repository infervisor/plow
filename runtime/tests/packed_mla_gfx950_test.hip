/* Packed-prefill MLA consumers versus the same low-level kernels launched once per span.
 *
 *   nix develop --command hipcc --offload-arch=gfx950 -O3 -w -DPACKED_MLA_DEVICE \
 *     --genco runtime/tests/packed_mla_gfx950_test.hip -o /tmp/packed_mla.co \
 *     -Iruntime/amd -Iruntime/common
 *   nix develop --command hipcc -O2 -w runtime/tests/packed_mla_gfx950_test.hip \
 *     -o /tmp/packed_mla -lamdhip64
 *   perf-data/tools/gpulease -n 1 packed-mla /tmp/packed_mla /tmp/packed_mla.co
 */

#ifdef PACKED_MLA_DEVICE
#define PLOW_PACKED_PREFILL_CONSUMERS 1
#define PLOW_MLA_PREFILL 1
#define PLOW_BUCKET_DECODE 0
#include "interp.hip"

struct Args {
    unsigned long long rms, rope, x512, x64, gamma, cosb, sinb, pos;
    unsigned long long qa, qr, ckv, kr, kvlen, op, ml, spans, parked;
    unsigned row0, nrows, slot, kv0, nspans, rung, ctx, nhead;
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

#else
#include <hip/hip_runtime.h>
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
#define HC(x) do { hipError_t e=(x); if(e){std::fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(e),__LINE__);return 1;} } while(0)
#define HC2(x) do { hipError_t e=(x); if(e){std::fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(e),__LINE__);std::exit(1);} } while(0)
static bf16 f2b(float f) { unsigned u; std::memcpy(&u,&f,4); u += 0x7fffu+((u>>16)&1u); return u>>16; }
static void fill(std::vector<bf16>& v, unsigned seed) { for(auto& x:v){seed=seed*1664525u+1013904223u; x=f2b(((int)(seed>>9)%2048-1024)/1024.f);} }

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "/tmp/packed_mla.co";
    hipModule_t mod; HC(hipModuleLoad(&mod, co));
    hipFunction_t kp_cache, ks_cache, kp_flash, ks_flash;
    HC(hipModuleGetFunction(&kp_cache,mod,"packed_cache")); HC(hipModuleGetFunction(&ks_cache,mod,"serial_cache"));
    HC(hipModuleGetFunction(&kp_flash,mod,"packed_flash")); HC(hipModuleGetFunction(&ks_flash,mod,"serial_flash"));
    const unsigned R=8,C=16,NH=1; const size_t S=3;
    auto run_case = [&](const char* name, std::vector<Span> spans,
                        std::vector<unsigned> parked) -> bool {
    size_t active=0; for(const auto&s:spans)active+=s.nrows;
    std::vector<bf16> x512(R*512),x64(R*64),gamma(512),qa(R*NH*512),qr(R*NH*64);
    fill(x512,1);fill(x64,2);fill(gamma,3);fill(qa,4);fill(qr,5);
    std::vector<float> cs(C*32,1.f), sn(C*32,0.f); std::vector<int> pos{3,4,7,8,9,0,0,0};
    std::vector<int> lens{10,13,5};
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
    std::printf("%-10s cache=%s flash_max_abs=%.3e outside/tail=%s\n",name,hpr==hsr&&hpk==hsk?"exact":"FAIL",md,ok?"untouched":"FAIL");
    return ok;
    };
    bool ok=run_case("single",{{0,2,2,0,3,5,2,0}},{0,0,1,1,1,1,1,1});
    ok&=run_case("ragged-2+3",{{0,2,2,0,3,5,2,0},{2,3,0,0,7,10,0,0}},
                 {0,0,0,0,0,1,1,1});
    std::puts(ok?"PASS":"FAIL"); return ok?0:1;
}
#endif
