/* Generic AttnRes + RMSNorm semantic/performance gate for gfx950.
 *
 * The candidate follows the pinned-vLLM order: score BF16 sources in f32, mix in f32, and
 * normalize the unrounded f32 mix with an independent epsilon.  The control is Plow's current
 * exact-semantics fusion, whose norm intentionally rereads a BF16-rounded mix.  This bench is not
 * a model implementation: eligibility is the op contract plus HID<=8192, HID%8==0, wave64.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amd_common.h"
#include "op_k3.h"

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) {                         \
    std::fprintf(stderr, "HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_)); \
    std::exit(1); } } while (0)

enum { LDS_BYTES = 147464 };
#ifndef PLOW_ATTNRES_TOKENS
#define PLOW_ATTNRES_TOKENS 1
#endif

struct Args {
    bf16* out;
    const bf16* prefix;
    const bf16* ring;
    const float* score;
    const bf16* gamma;
};

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void control(
    const Args* ap, unsigned hid, unsigned nb, float score_eps, unsigned iters) {
    const Args a = *ap;
    __shared__ __align__(16) char raw[LDS_BYTES];
    for (unsigned i = 0; i < iters; ++i) {
        d_attn_res(a.out, a.prefix, a.ring, a.score, PLOW_ATTNRES_TOKENS, hid, nb, 8u, score_eps, blockIdx.x,
                   gridDim.x, (float*)raw, nullptr, 0u, a.gamma);
        __syncthreads();
    }
}

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void candidate(
    const Args* ap, unsigned hid, unsigned nb, float score_eps, float output_eps, unsigned iters) {
    const Args a = *ap;
    __shared__ __align__(16) char raw[LDS_BYTES];
    for (unsigned i = 0; i < iters; ++i) {
        d_attn_res<true>(a.out, a.prefix, a.ring, a.score, PLOW_ATTNRES_TOKENS, hid, nb, 8u, score_eps, blockIdx.x,
                         gridDim.x, (float*)raw, nullptr, 0u, a.gamma, output_eps);
        __syncthreads();
    }
}

static float bf2f_h(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static bf16 f2bf_h(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static unsigned long long rng = 0x243f6a8885a308d3ull;
static float rnd(float lo, float hi) {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return lo + (hi - lo) * (float)((rng >> 40) / 16777216.0);
}
static unsigned long long fnv(const void* ptr, size_t bytes) {
    const unsigned char* p = (const unsigned char*)ptr;
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

static void vllm_order(std::vector<bf16>& out, const std::vector<bf16>& prefix,
                       const std::vector<bf16>& ring, const std::vector<bf16>& norm,
                       const std::vector<bf16>& proj, const std::vector<bf16>& gamma,
                       unsigned hid, unsigned nb, float score_eps, float output_eps) {
    std::vector<double> logits(nb + 1), prob(nb + 1), mix(hid, 0.0);
    for (unsigned r = 0; r <= nb; ++r) {
        const bf16* x = r < nb ? ring.data() + (size_t)r * hid : prefix.data();
        double ss = 0.0, sw = 0.0;
        for (unsigned d = 0; d < hid; ++d) {
            const double f = bf2f_h(x[d]);
            ss += f * f;
            sw += f * (double)bf2f_h(norm[d]) * (double)bf2f_h(proj[d]);
        }
        logits[r] = sw / std::sqrt(ss / hid + score_eps);
    }
    const double mx = *std::max_element(logits.begin(), logits.end());
    double z = 0.0;
    for (unsigned r = 0; r <= nb; ++r) z += prob[r] = std::exp(logits[r] - mx);
    for (unsigned r = 0; r <= nb; ++r) {
        const bf16* x = r < nb ? ring.data() + (size_t)r * hid : prefix.data();
        for (unsigned d = 0; d < hid; ++d) mix[d] += prob[r] / z * bf2f_h(x[d]);
    }
    double ss = 0.0;
    for (double x : mix) ss += x * x;
    const double inv = 1.0 / std::sqrt(ss / hid + output_eps);
    out.resize(hid);
    for (unsigned d = 0; d < hid; ++d)
        out[d] = f2bf_h((float)(mix[d] * inv * bf2f_h(gamma[d])));
}

struct Error { double max_abs, rel_l2, exact; };
static Error error(const std::vector<bf16>& got, const std::vector<bf16>& ref) {
    double mx = 0.0, se = 0.0, sr = 0.0, eq = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = bf2f_h(got[i]), r = bf2f_h(ref[i]), e = g - r;
        mx = std::max(mx, std::abs(e)); se += e * e; sr += r * r; eq += got[i] == ref[i];
    }
    return {mx, std::sqrt(se / std::max(sr, 1e-30)), eq / got.size()};
}

#if PLOW_ATTNRES_TOKENS == 1
int main() {
    constexpr unsigned HID = 7168, CAP = 8;
    /* The two operands stay distinct even though this exact boundary uses the configured 1e-5
     * for both.  A second compile is not needed to prove independence: each rides a separate
     * kernel argument and only OUTPUT_EPS reaches the final reduction. */
    constexpr float SCORE_EPS = 1e-5f, OUTPUT_EPS = 1e-5f;
    std::vector<bf16> raw_prefix(HID), delta(HID), prefix(HID), ring((size_t)CAP * HID),
        norm(HID), proj(HID), gamma(HID);
    std::vector<float> score(HID);
    for (unsigned d = 0; d < HID; ++d) {
        raw_prefix[d] = f2bf_h(rnd(-2.0f, 2.0f));
        delta[d] = f2bf_h(rnd(-0.35f, 0.35f));
        prefix[d] = f2bf_h(bf2f_h(raw_prefix[d]) + bf2f_h(delta[d]));
        norm[d] = f2bf_h(rnd(0.75f, 1.25f));
        proj[d] = f2bf_h(rnd(-0.05f, 0.05f));
        gamma[d] = f2bf_h(rnd(0.5f, 1.5f));
        score[d] = bf2f_h(norm[d]) * bf2f_h(proj[d]);
    }
    for (bf16& x : ring) x = f2bf_h(rnd(-2.0f, 2.0f));
    std::printf("contract operation=softmax-rms-residual-mix rounding=add-f32-round-bf16 "
                "ring=token-block-hidden hid=%u cap=%u write_idx=-1 score_eps=%.9g "
                "output_eps=%.9g output_norm_input=mixed-f32 wave=64\n", HID, CAP, SCORE_EPS,
                OUTPUT_EPS);
    std::printf("hash prefix=%016llx delta=%016llx rounded_prefix=%016llx ring=%016llx "
                "norm=%016llx proj=%016llx gamma=%016llx\n",
                fnv(raw_prefix.data(), HID*2), fnv(delta.data(), HID*2), fnv(prefix.data(), HID*2),
                fnv(ring.data(), ring.size()*2), fnv(norm.data(), HID*2), fnv(proj.data(), HID*2),
                fnv(gamma.data(), HID*2));

    bf16 *dp, *dr, *dg, *do0, *do1;
    float* ds;
    HC(hipMalloc(&dp, HID*2)); HC(hipMalloc(&dr, ring.size()*2)); HC(hipMalloc(&dg, HID*2));
    HC(hipMalloc(&ds, HID*4)); HC(hipMalloc(&do0, HID*2)); HC(hipMalloc(&do1, HID*2));
    HC(hipMemcpy(dp, prefix.data(), HID*2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dr, ring.data(), ring.size()*2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dg, gamma.data(), HID*2, hipMemcpyHostToDevice));
    HC(hipMemcpy(ds, score.data(), HID*4, hipMemcpyHostToDevice));
    Args a0{do0,dp,dr,ds,dg}, a1{do1,dp,dr,ds,dg}, *da0, *da1;
    HC(hipMalloc(&da0,sizeof(Args))); HC(hipMalloc(&da1,sizeof(Args)));
    HC(hipMemcpy(da0,&a0,sizeof(Args),hipMemcpyHostToDevice));
    HC(hipMemcpy(da1,&a1,sizeof(Args),hipMemcpyHostToDevice));
    hipEvent_t beg,end; HC(hipEventCreate(&beg)); HC(hipEventCreate(&end));
    auto timed = [&](bool cand, unsigned nb) {
        const unsigned n1=300,n2=1300;
        if (cand) hipLaunchKernelGGL(candidate,1,PLOW_THREADS,0,0,da1,HID,nb,SCORE_EPS,OUTPUT_EPS,100);
        else hipLaunchKernelGGL(control,1,PLOW_THREADS,0,0,da0,HID,nb,SCORE_EPS,100);
        HC(hipDeviceSynchronize()); float x=0,y=0;
        HC(hipEventRecord(beg));
        if (cand) hipLaunchKernelGGL(candidate,1,PLOW_THREADS,0,0,da1,HID,nb,SCORE_EPS,OUTPUT_EPS,n1);
        else hipLaunchKernelGGL(control,1,PLOW_THREADS,0,0,da0,HID,nb,SCORE_EPS,n1);
        HC(hipEventRecord(end)); HC(hipEventSynchronize(end)); HC(hipEventElapsedTime(&x,beg,end));
        HC(hipEventRecord(beg));
        if (cand) hipLaunchKernelGGL(candidate,1,PLOW_THREADS,0,0,da1,HID,nb,SCORE_EPS,OUTPUT_EPS,n2);
        else hipLaunchKernelGGL(control,1,PLOW_THREADS,0,0,da0,HID,nb,SCORE_EPS,n2);
        HC(hipEventRecord(end)); HC(hipEventSynchronize(end)); HC(hipEventElapsedTime(&y,beg,end));
        return (double)(y-x)*1000.0/(n2-n1);
    };
    int bad=0;
    for (unsigned nb : {0u,1u,4u,8u}) {
        hipLaunchKernelGGL(control,1,PLOW_THREADS,0,0,da0,HID,nb,SCORE_EPS,1);
        hipLaunchKernelGGL(candidate,1,PLOW_THREADS,0,0,da1,HID,nb,SCORE_EPS,OUTPUT_EPS,1);
        HC(hipDeviceSynchronize());
        std::vector<bf16> h0(HID),h1(HID),ref;
        HC(hipMemcpy(h0.data(),do0,HID*2,hipMemcpyDeviceToHost));
        HC(hipMemcpy(h1.data(),do1,HID*2,hipMemcpyDeviceToHost));
        vllm_order(ref,prefix,ring,norm,proj,gamma,HID,nb,SCORE_EPS,OUTPUT_EPS);
        const Error e0=error(h0,ref),e1=error(h1,ref),ec=error(h1,h0);
        const double t0=timed(false,nb),t1=timed(true,nb),gain=(t0-t1)/t0*100.0;
        std::printf("nb=%u current %.3fus f32 %.3fus gain=%+.2f%% | current relL2 %.3e "
                    "f32 relL2 %.3e maxabs %.6g f32 exact %.2f%% current->f32 relL2 %.3e\n",
                    nb,t0,t1,gain,e0.rel_l2,e1.rel_l2,e1.max_abs,100.0*e1.exact,ec.rel_l2);
        /* CPU double is an ordering oracle, not a reduction-tree oracle.  The BF16 result must be
         * close and repeated GPU calls must be byte stable; promotion additionally needs 10%. */
        std::vector<bf16> repeat(HID);
        hipLaunchKernelGGL(candidate,1,PLOW_THREADS,0,0,da1,HID,nb,SCORE_EPS,OUTPUT_EPS,1);
        HC(hipDeviceSynchronize()); HC(hipMemcpy(repeat.data(),do1,HID*2,hipMemcpyDeviceToHost));
        if (repeat != h1 || e1.rel_l2 > 2e-3 || gain < 10.0) bad=1;
    }

    hipFuncAttributes ca{},fa{}; HC(hipFuncGetAttributes(&ca,(const void*)control));
    HC(hipFuncGetAttributes(&fa,(const void*)candidate));
    std::printf("resources current regs=%d private=%zu candidate regs=%d private=%zu\n",
                ca.numRegs,ca.localSizeBytes,fa.numRegs,fa.localSizeBytes);
    HC(hipFree(da0)); HC(hipFree(da1));
    std::puts(bad ? "REJECT" : "QUALIFIED");
    return bad ? 3 : 0;
}
#else
int main() {
    constexpr unsigned T=PLOW_ATTNRES_TOKENS,HID=7168,CAP=8,NB=8,GRID=256;
    constexpr float SCORE_EPS=1e-5f,OUTPUT_EPS=1e-5f;
    const size_t rows=(size_t)T*HID,rings=rows*CAP;
    bf16 *prefix,*ring,*gamma,*out0,*out1;
    float* score;
    HC(hipMalloc(&prefix,rows*2)); HC(hipMalloc(&ring,rings*2));
    HC(hipMalloc(&gamma,HID*2)); HC(hipMalloc(&score,HID*4));
    HC(hipMalloc(&out0,rows*2)); HC(hipMalloc(&out1,rows*2));
    HC(hipMemset(prefix,0x3f,rows*2)); HC(hipMemset(ring,0xbe,rings*2));
    HC(hipMemset(gamma,0x3f,HID*2)); HC(hipMemset(score,0,HID*4));
    Args h0{out0,prefix,ring,score,gamma},h1{out1,prefix,ring,score,gamma},*a0,*a1;
    HC(hipMalloc(&a0,sizeof(Args))); HC(hipMalloc(&a1,sizeof(Args)));
    HC(hipMemcpy(a0,&h0,sizeof(Args),hipMemcpyHostToDevice));
    HC(hipMemcpy(a1,&h1,sizeof(Args),hipMemcpyHostToDevice));
    hipEvent_t beg,end; HC(hipEventCreate(&beg)); HC(hipEventCreate(&end));
    auto timed=[&](bool cand) {
        constexpr unsigned WARM=2,REPS=12;
        if(cand) hipLaunchKernelGGL(candidate,GRID,PLOW_THREADS,0,0,a1,HID,NB,SCORE_EPS,OUTPUT_EPS,WARM);
        else hipLaunchKernelGGL(control,GRID,PLOW_THREADS,0,0,a0,HID,NB,SCORE_EPS,WARM);
        HC(hipDeviceSynchronize()); float ms=0; HC(hipEventRecord(beg));
        if(cand) hipLaunchKernelGGL(candidate,GRID,PLOW_THREADS,0,0,a1,HID,NB,SCORE_EPS,OUTPUT_EPS,REPS);
        else hipLaunchKernelGGL(control,GRID,PLOW_THREADS,0,0,a0,HID,NB,SCORE_EPS,REPS);
        HC(hipEventRecord(end)); HC(hipEventSynchronize(end)); HC(hipEventElapsedTime(&ms,beg,end));
        return (double)ms/REPS;
    };
    std::vector<double> c0,c1;
    for(unsigned fold=0;fold<4;++fold) {
        if(fold&1){c1.push_back(timed(true));c0.push_back(timed(false));}
        else{c0.push_back(timed(false));c1.push_back(timed(true));}
    }
    std::sort(c0.begin(),c0.end()); std::sort(c1.begin(),c1.end());
    const double t0=(c0[1]+c0[2])/2,t1=(c1[1]+c1[2])/2,gain=(t0-t1)/t0*100.0;
    std::printf("T%u nb=8 current %.6fms f32 %.6fms gain=%+.2f%% (four alternating folds)\n",
                T,t0,t1,gain);
    hipFuncAttributes ca{},fa{}; HC(hipFuncGetAttributes(&ca,(const void*)control));
    HC(hipFuncGetAttributes(&fa,(const void*)candidate));
    std::printf("resources current regs=%d private=%zu candidate regs=%d private=%zu\n",
                ca.numRegs,ca.localSizeBytes,fa.numRegs,fa.localSizeBytes);
    std::puts(gain>=10.0?"QUALIFIED":"REJECT");
    return gain>=10.0?0:3;
}
#endif
