#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CK(e) do { auto s=(e); if(s!=hipSuccess) { std::fprintf(stderr,"%s: %s line%d\n",#e,hipGetErrorString(s),__LINE__); std::exit(1); } } while(0)
template<class T> T* alloc(size_t n) { T* p; CK(hipMalloc(&p,n*sizeof(T))); return p; }
template<class T> T* upload(const std::vector<T>& v) { auto p=alloc<T>(v.size()); CK(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice)); return p; }
uint32_t rng=173;
uint32_t next() { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
uint16_t bf(float f) { uint32_t b; std::memcpy(&b,&f,4); return (b+0x7fff+((b>>16)&1))>>16; }
float f32(uint16_t v) { uint32_t b=uint32_t(v)<<16; float f; std::memcpy(&f,&b,4); return f; }
std::vector<uint16_t> random_bf(size_t n) { std::vector<uint16_t> v(n); for(auto& x:v)x=bf((int(next()%2001)-1000)/1000.f); return v; }
struct Gold { size_t row; std::vector<double> output; };
std::vector<Gold> reference(unsigned rows,const std::vector<uint16_t>& q,const std::vector<uint16_t>& kv,const std::vector<uint32_t>& idx,float scale) {
    std::vector<unsigned> picks{0,rows/2,rows-1}; picks.erase(std::unique(picks.begin(),picks.end()),picks.end());
    std::vector<Gold> gold;
    for(unsigned r:picks)for(unsigned h:{0u,7u}) {
        double query[576];for(unsigned d=0;d<576;++d)query[d]=f32(q[(size_t(r)*8+h)*576+d]);
        std::vector<double> scores(2048),out(512,0);double maximum=-INFINITY,den=0;
        for(unsigned k=0;k<2048;++k) {double dot=0;size_t at=size_t(idx[size_t(r)*2048+k])*576;for(unsigned d=0;d<576;++d)dot+=query[d]*double(f32(kv[at+d]));scores[k]=dot*scale;maximum=std::max(maximum,scores[k]);}
        for(unsigned k=0;k<2048;++k) {double p=std::exp(scores[k]-maximum);den+=p;size_t at=size_t(idx[size_t(r)*2048+k])*576;for(unsigned d=0;d<512;++d)out[d]+=p*double(f32(kv[at+d]));}
        for(auto& x:out)x/=den;
        gold.push_back({size_t(r)*8+h,std::move(out)});
    }
    return gold;
}
int main(int argc,char** argv) {
    if(argc!=2) {std::fprintf(stderr,"usage: %s pinned-qh8-v3.co\n",argv[0]);return 2;}
    CK(hipInit(0));hipModule_t mod;hipFunction_t kernel;CK(hipModuleLoad(&mod,argv[1]));CK(hipModuleGetFunction(&kernel,mod,"_ZN5aiter36mla_a16w16_qh8_qseqlen1_gqaratio8_v3E"));
    hipEvent_t begin,end;CK(hipEventCreate(&begin));CK(hipEventCreate(&end));
    constexpr unsigned guard=128;float scale=1.f/std::sqrt(576.f);
    std::puts("rows,context,overlap,splits,median_us,max_abs,relative_l2");
    for(unsigned rows:{1u,8u,128u,512u,2048u,8192u})for(unsigned ctx:{16384u,81920u})for(unsigned overlap:{0u,1u}) {
        auto qh=random_bf(size_t(rows)*8*576),kvh=random_bf(size_t(ctx)*576);
        std::vector<uint32_t> indices(size_t(rows)*2048),qp(rows+1),kp(rows+1),sp(rows+1),last(rows,1);
        unsigned past=ctx-rows,step=37;while(std::gcd(step,past)!=1)++step;
        for(unsigned r=0;r<rows;++r)for(unsigned k=0;k<2048;++k)indices[size_t(r)*2048+k]=((size_t(k)+(overlap?size_t(r)*2048:0))*step)%past;
        std::iota(qp.begin(),qp.end(),0);for(unsigned r=0;r<=rows;++r)kp[r]=r*2048;
        auto gold=reference(rows,qh,kvh,indices,scale);
        auto q=upload(qh),kv=upload(kvh);auto idx=upload(indices),qptr=upload(qp),kptr=upload(kp),lptr=upload(last),splitptr=upload(sp);
        auto out=alloc<float>(size_t(rows)*4*8*512+2*guard),lse=alloc<float>(size_t(rows)*4*8+2*guard);
        for(unsigned ns:{2u,1u,4u}) {
            for(unsigned r=0;r<=rows;++r)sp[r]=r*ns;CK(hipMemcpy(splitptr,sp.data(),sp.size()*4,hipMemcpyHostToDevice));
            uint64_t args[40]={};args[0]=(uint64_t)(out+guard);args[2]=(uint64_t)(lse+guard);args[4]=(uint64_t)q;args[6]=(uint64_t)kv;args[8]=(uint64_t)kptr;args[10]=(uint64_t)idx;args[12]=(uint64_t)lptr;std::memcpy(&args[14],&scale,4);args[16]=8;args[18]=ns;args[20]=8*576*2;args[22]=576*2;args[26]=(uint64_t)qptr;args[28]=(uint64_t)splitptr;
            size_t args_size=sizeof(args);void* config[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,args,HIP_LAUNCH_PARAM_BUFFER_SIZE,&args_size,HIP_LAUNCH_PARAM_END};
            auto launch=[&] {CK(hipModuleLaunchKernel(kernel,1,rows,ns,256,1,1,0,nullptr,nullptr,config));};
            double max_abs=0,error=0,magnitude=0;
            std::vector<float> host(size_t(rows)*ns*8*512+2*guard),host_lse(size_t(rows)*ns*8+2*guard);
            for(unsigned reuse=0;reuse<3;++reuse) {
                std::fill(host.begin(),host.end(),NAN);std::fill(host_lse.begin(),host_lse.end(),NAN);
                CK(hipMemcpy(out,host.data(),host.size()*4,hipMemcpyHostToDevice));CK(hipMemcpy(lse,host_lse.data(),host_lse.size()*4,hipMemcpyHostToDevice));launch();CK(hipDeviceSynchronize());
                CK(hipMemcpy(host.data(),out,host.size()*4,hipMemcpyDeviceToHost));CK(hipMemcpy(host_lse.data(),lse,host_lse.size()*4,hipMemcpyDeviceToHost));
                for(unsigned i=0;i<guard;++i)if(!std::isnan(host[i])||!std::isnan(host[host.size()-guard+i])||!std::isnan(host_lse[i])||!std::isnan(host_lse[host_lse.size()-guard+i]))return 3;
                for(size_t i=guard;i<host.size()-guard;++i)if(!std::isfinite(host[i])) {std::fprintf(stderr,"nonfinite rows%u ns%u index%zu\n",rows,ns,i);return 4;}
                for(const auto& g:gold) {
                    unsigned r=g.row/8,h=g.row%8;double m=-INFINITY,weights[4],den=0;
                    for(unsigned s=0;s<ns;++s)m=std::max(m,double(host_lse[guard+(size_t(r)*ns+s)*8+h]));
                    for(unsigned s=0;s<ns;++s){weights[s]=std::exp(double(host_lse[guard+(size_t(r)*ns+s)*8+h])-m);den+=weights[s];}
                    for(unsigned d=0;d<512;++d) {double value=0;for(unsigned s=0;s<ns;++s)value+=weights[s]*host[guard+((size_t(r)*ns+s)*8+h)*512+d];double delta=value/den-g.output[d];if(!std::isfinite(delta))return 5;max_abs=std::max(max_abs,std::abs(delta));error+=delta*delta;magnitude+=g.output[d]*g.output[d];}
                }
            }
            std::vector<float> times;
            for(unsigned rep=0;rep<11;++rep){CK(hipEventRecord(begin));launch();CK(hipEventRecord(end));CK(hipEventSynchronize(end));float ms;CK(hipEventElapsedTime(&ms,begin,end));if(rep>=2)times.push_back(ms*1000);}
            std::sort(times.begin(),times.end());double rel=std::sqrt(error/magnitude);
            std::printf("%u,%u,%s,%u,%.6f,%.9g,%.9g\n",rows,ctx,overlap?"distinct":"shared",ns,times[times.size()/2],max_abs,rel);std::fflush(stdout);
            if(max_abs>.02||rel>.02)return 6;
        }
        for(void* p:{(void*)q,(void*)kv,(void*)idx,(void*)qptr,(void*)kptr,(void*)lptr,(void*)splitptr,(void*)out,(void*)lse})CK(hipFree(p));
    }
    CK(hipModuleUnload(mod));CK(hipEventDestroy(begin));CK(hipEventDestroy(end));
}
