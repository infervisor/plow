#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::exit(1); \
} } while (0)

extern "C" void k_wu_control();
extern "C" void k_wu_key_factors();
extern "C" void k_carry_control();
extern "C" void k_carry_precomputed();

static uint16_t bf16(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    const uint32_t bias = 0x7fffu + ((u >> 16) & 1u);
    return uint16_t((u + bias) >> 16);
}

template<class T> static T* dev(const std::vector<T>& h) {
    T* p; CHECK(hipMalloc(&p, h.size() * sizeof(T)));
    CHECK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end()); return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 8192;
    const unsigned H = argc > 2 ? std::strtoul(argv[2], nullptr, 0) : 12;
    const unsigned D = argc > 3 ? std::strtoul(argv[3], nullptr, 0) : 128;
    const unsigned V = argc > 4 ? std::strtoul(argv[4], nullptr, 0) : D;
    const unsigned samples = argc > 5 ? std::strtoul(argv[5], nullptr, 0) : 21;
    if (!T || !H || D != 128 || V != 128) return 2;
    const size_t kd = (size_t)T * H * D, vv = (size_t)T * H * V;
    const size_t aa = (size_t)T * H * 64u, ss = (size_t)H * V * D;
    std::vector<uint16_t> q(kd), k(kd), v(vv);
    std::vector<float> aqk(aa), ainv(aa), g(kd), beta((size_t)T*H), state0(ss);
    for (size_t i = 0; i < kd; ++i) {
        q[i] = bf16(float(int((i * 17u) % 31u) - 15) * 0.003f);
        k[i] = bf16(float(int((i * 29u) % 37u) - 18) * 0.002f);
        g[i] = -0.0002f * float(1u + ((i / ((size_t)H * D)) & 63u));
    }
    for (size_t i = 0; i < vv; ++i)
        v[i] = bf16(float(int((i * 47u) % 43u) - 21) * 0.001f);
    for (size_t i = 0; i < aa; ++i) {
        aqk[i] = float(int((i * 53u) % 47u) - 23) * 0.00003f;
        ainv[i] = (i % 65u == 0 ? 1.0f : float(int((i * 61u) % 43u) - 21) * 0.00002f);
    }
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = 0.25f + 0.001f * float(i % 37u);
    for (size_t i = 0; i < ss; ++i)
        state0[i] = float(int((i * 59u) % 53u) - 26) * 0.0001f;

    auto dq0=dev(q), dq1=dev(q), dk=dev(k), dv=dev(v); auto da=dev(aqk), di=dev(ainv), dg=dev(g), db=dev(beta);
    auto ds0=dev(state0), ds1=dev(state0); uint16_t *do0, *do1, *dhi, *dlo;
    uint16_t *dw0, *du0, *dw1, *du1;
    CHECK(hipMalloc(&do0, vv*2)); CHECK(hipMalloc(&do1, vv*2));
    CHECK(hipMalloc(&dhi, kd*2)); CHECK(hipMalloc(&dlo, kd*2));
    CHECK(hipMalloc(&dw0,kd*2)); CHECK(hipMalloc(&dw1,kd*2));
    CHECK(hipMalloc(&du0,vv*2)); CHECK(hipMalloc(&du1,vv*2));
    const dim3 block(256), pre_grid(256), carry_grid(H * ((V + 15u) / 16u));
    const size_t lds = 16u * D * 4u + 64u * 16u * 2u + 64u * 16u * 4u;
    const float scale = 1.0f / 128.0f;
    void* w0_args[] = {&dw0,&du0,&dq0,&di,&dk,&dv,&dg,&db,(void*)&T,(void*)&H,(void*)&scale};
    void* w1_args[] = {&dw1,&du1,&dhi,&dlo,&dq1,&di,&dk,&dv,&dg,&db,(void*)&T,(void*)&H,(void*)&scale};
    CHECK(hipLaunchKernel((const void*)k_wu_control,pre_grid,block,w0_args,0,nullptr));
    CHECK(hipLaunchKernel((const void*)k_wu_key_factors,pre_grid,block,w1_args,0,nullptr));
    void* c0_args[] = {&do0,&ds0,&dq0,&dk,&dw0,&du0,&da,&dg,(void*)&T,(void*)&H};
    void* c1_args[] = {&do1,&ds1,&dq1,&dk,&dhi,&dlo,&dw1,&du1,&da,&dg,(void*)&T,(void*)&H};
    CHECK(hipLaunchKernel((const void*)k_carry_control, carry_grid, block, c0_args, lds, nullptr));
    CHECK(hipLaunchKernel((const void*)k_carry_precomputed, carry_grid, block, c1_args, lds, nullptr));
    CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> o0(vv), o1(vv); std::vector<float> s0(ss), s1(ss);
    CHECK(hipMemcpy(o0.data(), do0, vv*2, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(o1.data(), do1, vv*2, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(s0.data(), ds0, ss*4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(s1.data(), ds1, ss*4, hipMemcpyDeviceToHost));
    std::vector<uint16_t> wh0(kd),wh1(kd),uh0(vv),uh1(vv),qh0(kd),qh1(kd);
    CHECK(hipMemcpy(wh0.data(),dw0,kd*2,hipMemcpyDeviceToHost)); CHECK(hipMemcpy(wh1.data(),dw1,kd*2,hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(uh0.data(),du0,vv*2,hipMemcpyDeviceToHost)); CHECK(hipMemcpy(uh1.data(),du1,vv*2,hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(qh0.data(),dq0,kd*2,hipMemcpyDeviceToHost)); CHECK(hipMemcpy(qh1.data(),dq1,kd*2,hipMemcpyDeviceToHost));
    size_t qm=0,wm=0,um=0,om=0,sm=0; for(size_t i=0;i<kd;++i) { qm += qh0[i]!=qh1[i]; wm += wh0[i]!=wh1[i]; }
    for(size_t i=0;i<vv;++i) { um += uh0[i]!=uh1[i]; om += o0[i]!=o1[i]; }
    for(size_t i=0;i<ss;++i) { uint32_t a,b; std::memcpy(&a,&s0[i],4); std::memcpy(&b,&s1[i],4); sm += a!=b; }
    std::printf("shape T=%u H=%u D=%u V=%u grid=%u lds=%zu\n",T,H,D,V,carry_grid.x,lds);
    std::printf("oracle q_mismatch=%zu/%zu W_mismatch=%zu/%zu U_mismatch=%zu/%zu output_mismatch=%zu/%zu state_mismatch=%zu/%zu\n",qm,kd,wm,kd,um,vv,om,vv,sm,ss);
    if (qm || wm || um || om || sm) return 3;

    hipEvent_t a,b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    auto time = [&](bool candidate) {
        std::vector<float> times;
        for (unsigned it=0; it<samples+3; ++it) {
            CHECK(hipEventRecord(a));
            if (candidate) {
                CHECK(hipLaunchKernel((const void*)k_wu_key_factors,pre_grid,block,w1_args,0,nullptr));
                CHECK(hipLaunchKernel((const void*)k_carry_precomputed,carry_grid,block,c1_args,lds,nullptr));
            } else {
                CHECK(hipLaunchKernel((const void*)k_wu_control,pre_grid,block,w0_args,0,nullptr));
                CHECK(hipLaunchKernel((const void*)k_carry_control,carry_grid,block,c0_args,lds,nullptr));
            }
            CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b)); float ms; CHECK(hipEventElapsedTime(&ms,a,b));
            if (it>=3) times.push_back(ms);
        } return median(times);
    };
    const float control=time(false), candidate=time(true);
    std::printf("median_ms wu_plus_carry_control=%.6f wu_keyfactor_plus_carry=%.6f delta=%.6f speedup=%.3fx samples=%u\n",
                control,candidate,candidate-control,control/candidate,samples);
    return 0;
}
