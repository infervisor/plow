/* Exact KDA decode semantic-boundary screen: conv update + recurrent state step +
 * gated RMSNorm. B is runtime-selected; H=12, D=128 and W=4 match the emitted
 * workload manifest. The control calls production Plow device bodies. The fused
 * arm is an isolated rejected candidate retained so later schedules compare
 * against the same oracle and timing boundary.
 *
 * Plow stores convolution state as f32. A vLLM fused_kda_decode comparison that
 * stores it as bf16 is shape- and semantics-matched, but not dtype-matched.
 *
 * Usage: kda_decode_fused_exact <code-object> <B:1|8> [state-step-workgroups]
 *                               [production-fused-object].
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP failure: %s\n", hipGetErrorString(e_)); std::exit(1); } } while (0)

namespace {
constexpr unsigned H = 12, D = 128, BV = 8, W = 4, P = H * D, Threads = 512;
constexpr unsigned Shared = (3 * D + 2 * 8 + BV) * sizeof(float);

uint16_t bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t)(u >> 16); }
float f32(uint16_t x) { uint32_t u = (uint32_t)x << 16; float v; std::memcpy(&v, &u, 4); return v; }
float rnd(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return ((float)(s & 65535u) / 65536.0f - 0.5f) * 0.2f; }

template<class T> T* upload(const std::vector<T>& x) { T* p; CK(hipMalloc(&p, x.size()*sizeof(T)));
    CK(hipMemcpy(p, x.data(), x.size()*sizeof(T), hipMemcpyHostToDevice)); return p; }
template<class T> T* alloc(size_t n) { T* p; CK(hipMalloc(&p, n*sizeof(T))); return p; }
void launch(hipFunction_t f, unsigned blocks, unsigned shared, void** args) {
    CK(hipModuleLaunchKernel(f, blocks,1,1, Threads,1,1, shared, nullptr,args,nullptr)); }
void launch128(hipFunction_t f, unsigned blocks, void** args) {
    CK(hipModuleLaunchKernel(f, blocks,1,1, 128,1,1, 0, nullptr,args,nullptr)); }
void launch256(hipFunction_t f, unsigned blocks, void** args) {
    CK(hipModuleLaunchKernel(f, blocks,1,1, 256,1,1, 0, nullptr,args,nullptr)); }
double median(std::vector<double> x) { std::sort(x.begin(),x.end()); return x[x.size()/2]; }

struct Err { double rel, max; size_t bad; };
template<class T, class F> Err compare(T* a, T* b, size_t n, F cvt) {
    std::vector<T> x(n), y(n); CK(hipMemcpy(x.data(),a,n*sizeof(T),hipMemcpyDeviceToHost));
    CK(hipMemcpy(y.data(),b,n*sizeof(T),hipMemcpyDeviceToHost));
    double e2=0,r2=0,m=0; size_t bad=0;
    for(size_t i=0;i<n;++i){ double p=cvt(x[i]),q=cvt(y[i]); if(!std::isfinite(p)||!std::isfinite(q)){++bad;continue;}
        double d=p-q;e2+=d*d;r2+=p*p;m=std::max(m,std::abs(d)); }
    return {std::sqrt(e2/std::max(r2,1e-30)),m,bad};
}
}

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/kda_decode_fused_exact_gfx950.co";
    unsigned B = argc > 2 ? (unsigned)std::strtoul(argv[2],nullptr,10) : 1;
    const unsigned requested_step_blocks =
        argc > 3 ? (unsigned)std::strtoul(argv[3],nullptr,10) : 0;
    const char* production_object = argc > 4 ? argv[4] : nullptr;
    if (B != 1 && B != 8) { std::fprintf(stderr,"B must be 1 or 8\n"); return 2; }
    const size_t bp=(size_t)B*P, cs=3*bp*W, ss=(size_t)B*H*D*D;
    uint32_t rng=0x950128u+B;
    std::vector<uint16_t> qr(bp),kr(bp),vr(bp),forget(bp),out_gate(bp),beta((size_t)B*H);
    std::vector<float> weight(3ull*P*W),conv0(cs),alog(H),dt(P),state0(ss),nw(D);
    for(auto* x:{&qr,&kr,&vr,&forget,&out_gate,&beta}) for(auto& v:*x) v=bf16(rnd(rng));
    for(auto& v:weight)v=rnd(rng);
    for(auto& v:conv0)v=rnd(rng);
    for(auto& v:alog)v=-2.0f+rnd(rng);
    for(auto& v:dt)v=rnd(rng);
    for(auto& v:state0)v=rnd(rng);
    for(auto& v:nw)v=1.0f+rnd(rng);
    CK(hipInit(0)); hipModule_t mod; CK(hipModuleLoad(&mod,object));
    hipFunction_t fc,fs,fn,ff,fx,fz,fw; CK(hipModuleGetFunction(&fc,mod,"kda_conv_control"));
    CK(hipModuleGetFunction(&fs,mod,"kda_step_control")); CK(hipModuleGetFunction(&fn,mod,"kda_norm_control"));
    CK(hipModuleGetFunction(&ff,mod,"kda_decode_fused"));
    CK(hipModuleGetFunction(&fx,mod,"kda_step_bv8x16"));
    CK(hipModuleGetFunction(&fz,mod,"kda_decode_fused_bv8x16"));
    CK(hipModuleGetFunction(&fw,mod,"kda_decode_fused_256x16"));
    hipModule_t production_mod{}; hipFunction_t production_fn{};
    if (production_object) {
        CK(hipModuleLoad(&production_mod,production_object));
        CK(hipModuleGetFunction(&production_fn,production_mod,"plow_kda_decode_fused_256x16_v2"));
    }
    auto dqr=upload(qr),dkr=upload(kr),dvr=upload(vr),df=upload(forget),dg=upload(out_gate),db=upload(beta);
    auto dw=upload(weight),dc0=upload(conv0),da=upload(alog),dd=upload(dt),ds0=upload(state0),dn=upload(nw);
    auto mix=alloc<uint16_t>(3*bp),oc=alloc<uint16_t>(bp),ox=alloc<uint16_t>(bp),yc=alloc<uint16_t>(bp),yx=alloc<uint16_t>(bp),yf=alloc<uint16_t>(bp),yz=alloc<uint16_t>(bp),yw=alloc<uint16_t>(bp),yp=alloc<uint16_t>(bp);
    auto cc=alloc<float>(cs),cf=alloc<float>(cs),cz=alloc<float>(cs),cw=alloc<float>(cs),cp=alloc<float>(cs),sc=alloc<float>(ss),sx=alloc<float>(ss),sf=alloc<float>(ss),sz=alloc<float>(ss),sw=alloc<float>(ss),sp=alloc<float>(ss);
    auto reset=[&]{ CK(hipMemcpy(cc,dc0,cs*4,hipMemcpyDeviceToDevice)); CK(hipMemcpy(cf,dc0,cs*4,hipMemcpyDeviceToDevice));
        CK(hipMemcpy(cz,dc0,cs*4,hipMemcpyDeviceToDevice)); CK(hipMemcpy(cw,dc0,cs*4,hipMemcpyDeviceToDevice));
        CK(hipMemcpy(cp,dc0,cs*4,hipMemcpyDeviceToDevice));
        CK(hipMemcpy(sc,ds0,ss*4,hipMemcpyDeviceToDevice)); CK(hipMemcpy(sx,ds0,ss*4,hipMemcpyDeviceToDevice));
        CK(hipMemcpy(sf,ds0,ss*4,hipMemcpyDeviceToDevice)); CK(hipMemcpy(sz,ds0,ss*4,hipMemcpyDeviceToDevice));
        CK(hipMemcpy(sw,ds0,ss*4,hipMemcpyDeviceToDevice)); CK(hipMemcpy(sp,ds0,ss*4,hipMemcpyDeviceToDevice)); };
    const unsigned step_blocks=requested_step_blocks ? requested_step_blocks : std::min(256u,B*H*D/BV);
    const unsigned norm_blocks=(B*H+7)/8;
    auto conv=[&]{ void* ac[]={&mix,&dqr,&dkr,&dvr,&dw,&cc,&B}; launch(fc,304,0,ac); };
    auto step=[&]{ void* as[]={&oc,&mix,&df,&db,&da,&dd,&sc,&B}; launch(fs,step_blocks,Shared,as); };
    auto step_bv8x16=[&]{ void* as[]={&ox,&mix,&df,&db,&da,&dd,&sx,&B}; launch128(fx,B*H*D/BV,as); };
    auto norm=[&]{ void* an[]={&yc,&oc,&dn,&dg,&B}; launch(fn,norm_blocks,0,an); };
    auto norm_bv8x16=[&]{ void* an[]={&yx,&ox,&dn,&dg,&B}; launch(fn,norm_blocks,0,an); };
    auto control=[&]{ conv(); step(); norm(); };
    auto control_bv8x16=[&]{ conv(); step_bv8x16(); norm_bv8x16(); };
    auto fused=[&]{ void* af[]={&yf,&dqr,&dkr,&dvr,&dw,&dc0,&cf,&df,&db,&dg,&da,&dd,&sf,&dn,&B};
        launch(ff,B*H,0,af); };
    auto fused_bv8x16=[&]{ void* af[]={&yz,&dqr,&dkr,&dvr,&dw,&dc0,&cz,&df,&db,&dg,&da,&dd,&sz,&dn,&B};
        launch128(fz,B*H,af); };
    auto fused_256x16=[&]{ void* af[]={&yw,&dqr,&dkr,&dvr,&dw,&dc0,&cw,&df,&db,&dg,&da,&dd,&sw,&dn,&B};
        launch256(fw,B*H,af); };
    auto production=[&]{
        auto wq=dw, wk=dw+(size_t)P*W, wv=dw+(size_t)2u*P*W;
        auto csq=cp, csk=cp+bp*W, csv=cp+2u*bp*W;
        unsigned* parked=nullptr; unsigned rows=B,heads=H,dim=D,bv=BV,conv_w=W;
        unsigned flags=1u|(B>1?2u:0u),gate_mode=1u;
        float lower_bound=-5.0f,scale=1.0f/std::sqrt((float)D),norm_eps=1.0e-5f;
        void* ap[]={&yp,&dqr,&dkr,&dvr,&wq,&wk,&wv,&csq,&csk,&csv,&df,&db,&dg,&da,&dd,&sp,&dn,
                    &parked,&rows,&heads,&dim,&bv,&conv_w,&flags,&gate_mode,&lower_bound,&scale,&norm_eps};
        launch256(production_fn,B*H,ap);
    };
    reset(); control(); fused(); step_bv8x16(); norm_bv8x16(); fused_bv8x16(); fused_256x16();
    if(production_object) production(); CK(hipDeviceSynchronize());
    auto ce=compare(cc,cf,cs,[](float x){return x;}); auto se=compare(sc,sf,ss,[](float x){return x;});
    auto ye=compare(yc,yf,bp,[](uint16_t x){return f32(x);});
    auto xse=compare(sc,sx,ss,[](float x){return x;}); auto xoe=compare(oc,ox,bp,[](uint16_t x){return f32(x);});
    auto xye=compare(yc,yx,bp,[](uint16_t x){return f32(x);});
    auto zce=compare(cc,cz,cs,[](float x){return x;}); auto zse=compare(sc,sz,ss,[](float x){return x;});
    auto zye=compare(yc,yz,bp,[](uint16_t x){return f32(x);});
    auto wce=compare(cc,cw,cs,[](float x){return x;}); auto wse=compare(sc,sw,ss,[](float x){return x;});
    auto wye=compare(yc,yw,bp,[](uint16_t x){return f32(x);});
    Err pce{0,0,0},pse{0,0,0},pye{0,0,0};
    if(production_object) {
        pce=compare(cc,cp,cs,[](float x){return x;}); pse=compare(sc,sp,ss,[](float x){return x;});
        pye=compare(yc,yp,bp,[](uint16_t x){return f32(x);});
    }
    const bool ok=!ce.bad&&!se.bad&&!ye.bad&&!xse.bad&&!xoe.bad&&!xye.bad&&!zce.bad&&!zse.bad&&
                  !zye.bad&&!wce.bad&&!wse.bad&&!wye.bad&&ce.rel<1e-7&&se.rel<2e-5&&ye.rel<2e-4&&
                  xse.rel<2e-5&&xoe.rel<2e-4&&xye.rel<2e-4&&zce.rel<1e-7&&zse.rel<2e-5&&
                  zye.rel<2e-4&&wce.rel<1e-7&&wse.rel<2e-5&&wye.rel<2e-4&&
                  (!production_object||(pce.rel<1e-7&&pse.rel<2e-5&&pye.rel<2e-4&&!pce.bad&&!pse.bad&&!pye.bad));
    if(!ok){ std::fprintf(stderr,"FAIL conv %.3e state %.3e y %.3e bv8_state %.3e bv8_output %.3e bv8_y %.3e fused_bv8_conv %.3e fused_bv8_state %.3e fused_bv8_y %.3e fused_256_conv %.3e fused_256_state %.3e fused_256_y %.3e production_conv %.3e production_state %.3e production_y %.3e\n",ce.rel,se.rel,ye.rel,xse.rel,xoe.rel,xye.rel,zce.rel,zse.rel,zye.rel,wce.rel,wse.rel,wye.rel,pce.rel,pse.rel,pye.rel); return 2; }
    auto measure=[&](auto body){ hipEvent_t b,e; CK(hipEventCreate(&b));CK(hipEventCreate(&e));std::vector<double> samples;
        for(unsigned s=0;s<15;++s){ reset();CK(hipDeviceSynchronize());CK(hipEventRecord(b));
            for(unsigned i=0;i<1000;++i) body();
            CK(hipEventRecord(e));CK(hipEventSynchronize(e));float ms;CK(hipEventElapsedTime(&ms,b,e));
            if(s>=3)samples.push_back(ms); } CK(hipEventDestroy(b));CK(hipEventDestroy(e));return median(samples)/1000.0; };
    double c=measure(control),cx=measure(control_bv8x16),f=measure(fused),fzv=measure(fused_bv8x16),fwv=measure(fused_256x16),pv=production_object?measure(production):0.0,cv=measure(conv),st=measure(step),xs=measure(step_bv8x16),nm=measure(norm);
    std::printf("{\"schema\":\"plow.kda-decode-fused-exact.v2\",\"batch\":%u,\"heads\":12,\"dim\":128,\"conv_width\":4,\"step_blocks\":%u,\"control_us\":%.3f,\"control_bv8x16_us\":%.3f,\"fused_us\":%.3f,\"fused_bv8x16_us\":%.3f,\"fused_256x16_us\":%.3f,\"production_fused_us\":%.3f,\"speedup\":%.4f,\"conv_us\":%.3f,\"step_us\":%.3f,\"step_bv8x16_us\":%.3f,\"norm_us\":%.3f,\"conv_rel_l2\":%.9g,\"state_rel_l2\":%.9g,\"output_rel_l2\":%.9g,\"output_max_abs\":%.9g,\"bv8x16_state_rel_l2\":%.9g,\"bv8x16_output_rel_l2\":%.9g,\"bv8x16_output_max_abs\":%.9g,\"bv8x16_final_rel_l2\":%.9g,\"bv8x16_final_max_abs\":%.9g,\"fused_bv8x16_conv_rel_l2\":%.9g,\"fused_bv8x16_state_rel_l2\":%.9g,\"fused_bv8x16_final_rel_l2\":%.9g,\"fused_bv8x16_final_max_abs\":%.9g,\"fused_256x16_conv_rel_l2\":%.9g,\"fused_256x16_state_rel_l2\":%.9g,\"fused_256x16_final_rel_l2\":%.9g,\"fused_256x16_final_max_abs\":%.9g,\"production_conv_rel_l2\":%.9g,\"production_state_rel_l2\":%.9g,\"production_final_rel_l2\":%.9g,\"production_final_max_abs\":%.9g}\n",
        B,step_blocks,c*1000.0,cx*1000.0,f*1000.0,fzv*1000.0,fwv*1000.0,pv*1000.0,c/f,cv*1000.0,st*1000.0,xs*1000.0,nm*1000.0,ce.rel,se.rel,ye.rel,ye.max,xse.rel,xoe.rel,xoe.max,xye.rel,xye.max,zce.rel,zse.rel,zye.rel,zye.max,wce.rel,wse.rel,wye.rel,wye.max,pce.rel,pse.rel,pye.rel,pye.max);
    if(production_object) CK(hipModuleUnload(production_mod)); CK(hipModuleUnload(mod)); return 0;
}
