/* Standalone gfx950 proof for row/head-owned decode fusion. This deliberately does not enter
 * the packet ABI or persistent interpreter. The control kernels call the production
 * Conv3 -> StateStepG -> GatedNorm bodies; k_fused owns one independent (row, head), including
 * all three convolution windows and the complete V-first recurrent state. */
#ifdef KDA_FUSED_DEVICE
#define PLOW_WG_WAVES 4
#include <hip/hip_runtime.h>
#include "amd_common.h"
#include "op_elementwise.h"
#include "op_kda.h"

struct KdaFusedArgs {
    unsigned long long y, o, q, k, v, q_raw, k_raw, v_raw, decay_raw, beta_raw, output_gate;
    unsigned long long wq, wk, wv, csq, csk, csv, state, a_log, dt_bias, norm_w, parked;
    unsigned B, H, D, W, gate_mode;
    float lower_bound, scale, eps;
};

template <unsigned PL>
__device__ __forceinline__ void kda_decode_fused_body(KdaFusedArgs* a, float* shared) {
    if (!a->B || !a->H || a->D != PL * PLOW_WAVE || !a->W || a->W > 8)
        __builtin_trap();
    const unsigned tid = threadIdx.x, lane = tid & 63u, wave = tid >> 6;
    float* q = shared;
    float* k = q + a->D;
    float* g = k + a->D;
    float* v = g + a->D;
    float* o = v + a->D;
    float* reduce = o + a->D;

    for (unsigned item = blockIdx.x; item < a->B * a->H; item += gridDim.x) {
        const unsigned row = item / a->H, h = item % a->H;
        if (((const unsigned*)a->parked)[row]) continue;
        const size_t hd0 = ((size_t)row * a->H + h) * a->D;
        const size_t c0 = (size_t)h * a->D;
        const size_t cs0 = ((size_t)row * a->H + h) * a->D * a->W;

        /* One rolled implementation for q/k/v keeps only one W-tap window live. Convolution is
         * rounded through bf16 before q/k normalization and recurrence, exactly like op 111. */
#pragma unroll 1
        for (unsigned stream = 0; stream < 3; ++stream) {
            const bf16* raw = stream == 0 ? (const bf16*)a->q_raw
                              : stream == 1 ? (const bf16*)a->k_raw
                                            : (const bf16*)a->v_raw;
            const float* weight = stream == 0 ? (const float*)a->wq
                                  : stream == 1 ? (const float*)a->wk
                                                : (const float*)a->wv;
            float* conv = stream == 0 ? (float*)a->csq
                          : stream == 1 ? (float*)a->csk
                                        : (float*)a->csv;
            float* dst = stream == 0 ? q : (stream == 1 ? k : v);
            for (unsigned d = tid; d < a->D; d += PLOW_THREADS) {
                float win[8], tap[8];
#pragma unroll
                for (unsigned j = 0; j < 8; ++j) {
                    win[j] = j < a->W ? conv[cs0 + (size_t)d * a->W + j] : 0.0f;
                    tap[j] = j < a->W ? weight[(c0 + d) * a->W + j] : 0.0f;
                }
#pragma unroll
                for (unsigned j = 0; j + 1 < 8; ++j) win[j] = win[j + 1];
                win[a->W - 1] = bf2f(raw[hd0 + d]);
                float value = 0.0f;
#pragma unroll
                for (unsigned j = 0; j < 8; ++j) value += win[j] * tap[j];
                dst[d] = bf2f(f2bf(act_silu(value)));
#pragma unroll
                for (unsigned j = 0; j < 8; ++j)
                    if (j < a->W) conv[cs0 + (size_t)d * a->W + j] = win[j];
            }
            __syncthreads();
        }

        float qs = 0.0f, ks = 0.0f;
        for (unsigned d = tid; d < a->D; d += PLOW_THREADS) {
            const float s = bf2f(((const bf16*)a->decay_raw)[hd0 + d]) +
                            ((const float*)a->dt_bias)[c0 + d];
            const float ah = __expf(((const float*)a->a_log)[h]);
            const float gate = a->gate_mode == PLOW_KDA_GATE_LOWER_BOUND
                                   ? a->lower_bound * kda_sigmoid(ah * s)
                                   : -ah * kda_softplus(s);
            g[d] = __expf(gate);
            qs += q[d] * q[d];
            ks += k[d] * k[d];
        }
        qs = block_sum(qs, reduce);
        ks = block_sum(ks, reduce);
        const float rq = a->scale / sqrtf(qs + 1.0e-6f);
        const float rk = 1.0f / sqrtf(ks + 1.0e-6f);
        for (unsigned d = tid; d < a->D; d += PLOW_THREADS) {
            q[d] *= rq;
            k[d] *= rk;
        }
        __syncthreads();

        const float beta = kda_sigmoid(bf2f(((const bf16*)a->beta_raw)[(size_t)row * a->H + h]));
        float* state = (float*)a->state + ((size_t)row * a->H + h) * a->D * a->D;
        for (unsigned j = wave; j < a->D; j += PLOW_WAVES) {
            float* column = state + (size_t)j * a->D;
            float sc[PL], pk = 0.0f;
#pragma unroll
            for (unsigned r = 0; r < PL; ++r) {
                const unsigned d = r * PLOW_WAVE + lane;
                sc[r] = column[d] * g[d];
                pk += sc[r] * k[d];
            }
            pk = wave_sum(pk);
            const float update = beta * (v[j] - pk);
            float pq = 0.0f;
#pragma unroll
            for (unsigned r = 0; r < PL; ++r) {
                const unsigned d = r * PLOW_WAVE + lane;
                sc[r] += update * k[d];
                column[d] = sc[r];
                pq += sc[r] * q[d];
            }
            pq = wave_sum(pq);
            if (lane == 0) o[j] = bf2f(f2bf(pq));
        }
        __syncthreads();

        /* op 103 uses one wave per head. Keeping its association here makes the fused result a
         * direct oracle against the old chain rather than an accuracy-only comparison. */
        if (wave == 0) {
            float ss = 0.0f;
            for (unsigned d = lane; d < a->D; d += PLOW_WAVE) ss += o[d] * o[d];
            const float inv = rsqrtf(wave_sum(ss) / (float)a->D + a->eps);
            for (unsigned d = lane; d < a->D; d += PLOW_WAVE) {
                const float gate = kda_sigmoid(bf2f(((const bf16*)a->output_gate)[hd0 + d]));
                ((bf16*)a->y)[hd0 + d] = f2bf(o[d] * inv * ((const float*)a->norm_w)[d] * gate);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void kda_decode_fused_poc(KdaFusedArgs* a) {
    __shared__ float shared[5 * 256 + PLOW_WAVES];
    if (a->D == 64) kda_decode_fused_body<1>(a, shared);
    else if (a->D == 128) kda_decode_fused_body<2>(a, shared);
    else if (a->D == 256) kda_decode_fused_body<4>(a, shared);
    else __builtin_trap();
}

extern "C" __global__ __launch_bounds__(256, 2) void kda_decode_control_conv(KdaFusedArgs* a) {
    const size_t stride = (size_t)a->H * a->D * a->W;
    d_kda_conv3((bf16*)a->q, (bf16*)a->k, (bf16*)a->v,
                (const bf16*)a->q_raw, (const bf16*)a->k_raw, (const bf16*)a->v_raw,
                (const float*)a->wq, (const float*)a->wk, (const float*)a->wv,
                (float*)a->csq, (float*)a->csk, (float*)a->csv, a->B, a->H * a->D, a->W, 1,
                blockIdx.x, gridDim.x, stride, (const unsigned*)a->parked);
}

extern "C" __global__ __launch_bounds__(256, 2) void kda_decode_control_step(KdaFusedArgs* a) {
    __shared__ float lds[4 * 256 + 2 * PLOW_WAVES];
    d_kda_state_step_g((bf16*)a->o, (const bf16*)a->q, (const bf16*)a->k,
                       (const bf16*)a->v, (const bf16*)a->decay_raw,
                       (const bf16*)a->beta_raw, (const float*)a->a_log,
                       (const float*)a->dt_bias, a->gate_mode, a->lower_bound,
                       (float*)a->state, a->B, a->H, a->D, PLOW_WAVES,
                       PLOW_KDA_F_QK_L2NORM | PLOW_KDA_F_SEQ_ROWS, a->scale,
                       blockIdx.x, gridDim.x, lds, (size_t)a->H * a->D * a->D,
                       (const unsigned*)a->parked);
}

extern "C" __global__ __launch_bounds__(256, 2) void kda_decode_control_norm(KdaFusedArgs* a) {
    d_kda_gated_norm((bf16*)a->y, (const bf16*)a->o, (const float*)a->norm_w,
                     (const bf16*)a->output_gate, a->B, a->H, a->D, a->eps,
                     blockIdx.x, gridDim.x);
}

#else
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

typedef unsigned short bf16;
struct KdaFusedArgs {
    unsigned long long y, o, q, k, v, q_raw, k_raw, v_raw, decay_raw, beta_raw, output_gate;
    unsigned long long wq, wk, wv, csq, csk, csv, state, a_log, dt_bias, norm_w, parked;
    unsigned B, H, D, W, gate_mode;
    float lower_bound, scale, eps;
};
static float b2f(bf16 b) { unsigned u = (unsigned)b << 16; float x; memcpy(&x, &u, 4); return x; }
static bf16 f2b(float x) { unsigned u; memcpy(&u, &x, 4); u += 0x7fffu + ((u >> 16) & 1u); return u >> 16; }
#define HC(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP %s at %d\n", hipGetErrorString(e), __LINE__); return false; } } while (0)

template <class T> static T* upload(const std::vector<T>& h) {
    T* p = nullptr;
    if (hipMalloc(&p, h.size() * sizeof(T)) != hipSuccess) return nullptr;
    if (hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) return nullptr;
    return p;
}
static unsigned next_u32() { static unsigned x = 0x91e10da5u; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
static float rnd(float scale = 1.0f) { return scale * ((int)(next_u32() & 0xffff) - 32768) / 32768.0f; }

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return .5 * (values[(values.size() - 1) / 2] + values[values.size() / 2]);
}

struct TimingVerdict { double ratio, mad; unsigned wins; bool pass; };
static TimingVerdict timing_verdict(const std::vector<double>& control,
                                    const std::vector<double>& fused) {
    if (control.size() != fused.size() || control.size() < 9) return {0, 0, 0, false};
    std::vector<double> ratios, deviations;
    unsigned wins = 0;
    for (size_t i = 0; i < control.size(); ++i) {
        if (!(control[i] > 0) || !(fused[i] > 0) || !std::isfinite(control[i]) ||
            !std::isfinite(fused[i])) return {0, 0, 0, false};
        ratios.push_back(fused[i] / control[i]);
        wins += fused[i] < control[i];
    }
    const double ratio = median(ratios);
    for (double r : ratios) deviations.push_back(std::abs(r - ratio));
    const double mad = median(deviations);
    const unsigned required_wins = (3 * (unsigned)ratios.size() + 3) / 4;
    return {ratio, mad, wins,
            ratio <= .97 && ratio + 3.0 * mad < 1.0 && wins >= required_wins};
}

static int timing_selftest() {
    const std::vector<double> control(11, 10.0);
    std::vector<double> clear{8.9,9.0,9.1,9.0,8.8,9.2,9.0,9.1,8.9,9.0,9.1};
    std::vector<double> marginal(11, 9.8);
    std::vector<double> noisy{8.0,11.0,8.0,11.0,8.0,11.0,8.0,11.0,8.0,11.0,8.0};
    const bool ok = timing_verdict(control, clear).pass &&
                    !timing_verdict(control, marginal).pass &&
                    !timing_verdict(control, noisy).pass &&
                    !timing_verdict({10.0}, {9.0}).pass;
    puts(ok ? "timing verdict selftest: PASS" : "timing verdict selftest: FAIL");
    return ok ? 0 : 1;
}

struct Shape { unsigned B, H, D, W; };
static bool run_shape(hipModule_t module, Shape s) {
    const size_t n = (size_t)s.B * s.H * s.D, nh = (size_t)s.B * s.H;
    const size_t csn = n * s.W, sn = n * s.D, wn = (size_t)s.H * s.D * s.W;
    std::vector<bf16> rawq(n), rawk(n), rawv(n), decay(n), out_gate(n);
    std::vector<bf16> beta(nh), poison(n, f2b(29.0f));
    std::vector<float> wq(wn), wk(wn), wv(wn), csq(csn), csk(csn), csv(csn), state(sn);
    std::vector<float> alog(s.H), dt((size_t)s.H * s.D), norm(s.D);
    std::vector<unsigned> parked(s.B, 0);
    if (s.B > 1) { parked[0] = 1; parked[s.B / 2] = 1; }
    for (size_t i = 0; i < n; ++i) {
        rawq[i] = f2b(rnd(.5f)); rawk[i] = f2b(rnd(.5f)); rawv[i] = f2b(rnd(.5f));
        decay[i] = f2b(rnd(2.0f)); out_gate[i] = f2b(rnd(2.0f));
    }
    for (size_t i = 0; i < nh; ++i) beta[i] = f2b(rnd(2.0f));
    for (auto& x : wq) x = rnd(.25f); for (auto& x : wk) x = rnd(.25f); for (auto& x : wv) x = rnd(.25f);
    for (auto& x : csq) x = rnd(.3f); for (auto& x : csk) x = rnd(.3f); for (auto& x : csv) x = rnd(.3f);
    for (auto& x : state) x = rnd(.05f);
    for (auto& x : alog) x = logf(.5f + fabsf(rnd(11.5f)));
    for (auto& x : dt) x = -4.5f + rnd(2.0f);
    for (auto& x : norm) x = .5f + fabsf(rnd());
    const bf16 nan = f2b(std::numeric_limits<float>::quiet_NaN());
    for (unsigned r = 0; r < s.B; ++r) if (parked[r]) {
        for (size_t i = (size_t)r * s.H * s.D; i < (size_t)(r + 1) * s.H * s.D; ++i)
            rawq[i] = rawk[i] = rawv[i] = decay[i] = out_gate[i] = nan;
        for (size_t i = (size_t)r * s.H; i < (size_t)(r + 1) * s.H; ++i) beta[i] = nan;
    }

    bf16 *dq = upload(rawq), *dk = upload(rawk), *dv = upload(rawv), *dg = upload(decay);
    bf16 *db = upload(beta), *dog = upload(out_gate);
    float *dwq = upload(wq), *dwk = upload(wk), *dwv = upload(wv);
    float *da = upload(alog), *ddt = upload(dt), *dnw = upload(norm);
    unsigned* dp = upload(parked);
    float *csq0 = upload(csq), *csk0 = upload(csk), *csv0 = upload(csv), *st0 = upload(state);
    float *csq1 = upload(csq), *csk1 = upload(csk), *csv1 = upload(csv), *st1 = upload(state);
    float *csqb = upload(csq), *cskb = upload(csk), *csvb = upload(csv), *stb = upload(state);
    bf16 *q = upload(poison), *k = upload(poison), *v = upload(poison), *o = upload(poison);
    bf16 *yc = upload(poison), *yf = upload(poison);
    if (!dq || !dk || !dv || !dg || !db || !dog || !dwq || !dwk || !dwv || !da || !ddt ||
        !dnw || !dp || !csq0 || !csk0 || !csv0 || !st0 || !csq1 || !csk1 || !csv1 ||
        !st1 || !csqb || !cskb || !csvb || !stb || !q || !k || !v || !o || !yc || !yf)
        return false;

    KdaFusedArgs c{};
    c.y=(size_t)yc; c.o=(size_t)o; c.q=(size_t)q; c.k=(size_t)k; c.v=(size_t)v;
    c.q_raw=(size_t)dq; c.k_raw=(size_t)dk; c.v_raw=(size_t)dv; c.decay_raw=(size_t)dg;
    c.beta_raw=(size_t)db; c.output_gate=(size_t)dog; c.wq=(size_t)dwq; c.wk=(size_t)dwk;
    c.wv=(size_t)dwv; c.csq=(size_t)csq0; c.csk=(size_t)csk0; c.csv=(size_t)csv0;
    c.state=(size_t)st0; c.a_log=(size_t)da; c.dt_bias=(size_t)ddt; c.norm_w=(size_t)dnw;
    c.parked=(size_t)dp; c.B=s.B; c.H=s.H; c.D=s.D; c.W=s.W; c.gate_mode=1;
    c.lower_bound=-5.0f; c.scale=1.0f/sqrtf((float)s.D); c.eps=1.0e-5f;
    KdaFusedArgs f=c; f.y=(size_t)yf; f.csq=(size_t)csq1; f.csk=(size_t)csk1;
    f.csv=(size_t)csv1; f.state=(size_t)st1;
    KdaFusedArgs *dc=nullptr, *df=nullptr; HC(hipMalloc(&dc,sizeof(c))); HC(hipMalloc(&df,sizeof(f)));
    HC(hipMemcpy(dc,&c,sizeof(c),hipMemcpyHostToDevice)); HC(hipMemcpy(df,&f,sizeof(f),hipMemcpyHostToDevice));
    hipFunction_t fc, fs, fn, ff;
    HC(hipModuleGetFunction(&fc,module,"kda_decode_control_conv"));
    HC(hipModuleGetFunction(&fs,module,"kda_decode_control_step"));
    HC(hipModuleGetFunction(&fn,module,"kda_decode_control_norm"));
    HC(hipModuleGetFunction(&ff,module,"kda_decode_fused_poc"));
    /* Keep the launch grid identical to the existing interpreter chain. Shrinking only the fused
     * grid to active heads makes a launch-overhead comparison look better than production. */
    void* ac[]={&dc}; void* af[]={&df}; const unsigned grid = 256;
    auto control = [&]() -> bool {
        HC(hipModuleLaunchKernel(fc,256,1,1,256,1,1,0,0,ac,nullptr));
        HC(hipModuleLaunchKernel(fs,256,1,1,256,1,1,0,0,ac,nullptr));
        HC(hipModuleLaunchKernel(fn,256,1,1,256,1,1,0,0,ac,nullptr));
        return true;
    };
    auto fused = [&]() -> bool {
        HC(hipModuleLaunchKernel(ff,grid,1,1,256,1,1,0,0,af,nullptr));
        return true;
    };
    if (!control() || !fused()) return false;
    HC(hipDeviceSynchronize());

    std::vector<bf16> hyc(n), hyf(n), hq(n), ho(n); std::vector<float> hs0(sn), hs1(sn);
    std::vector<float> hcq0(csn),hcq1(csn),hck0(csn),hck1(csn),hcv0(csn),hcv1(csn);
    HC(hipMemcpy(hyc.data(),yc,n*2,hipMemcpyDeviceToHost)); HC(hipMemcpy(hyf.data(),yf,n*2,hipMemcpyDeviceToHost));
    HC(hipMemcpy(hq.data(),q,n*2,hipMemcpyDeviceToHost)); HC(hipMemcpy(ho.data(),o,n*2,hipMemcpyDeviceToHost));
    HC(hipMemcpy(hs0.data(),st0,sn*4,hipMemcpyDeviceToHost)); HC(hipMemcpy(hs1.data(),st1,sn*4,hipMemcpyDeviceToHost));
#define COPY_CS(dst,src) HC(hipMemcpy((dst).data(),src,csn*4,hipMemcpyDeviceToHost))
    COPY_CS(hcq0,csq0); COPY_CS(hcq1,csq1); COPY_CS(hck0,csk0); COPY_CS(hck1,csk1); COPY_CS(hcv0,csv0); COPY_CS(hcv1,csv1);
    double ye=0,yw=0,se=0,sw=0; bool poison_ok=true, finite=true, conv_exact=true;
    for (unsigned r=0;r<s.B;++r) {
        const size_t rb=(size_t)r*s.H*s.D, re=rb+(size_t)s.H*s.D;
        const size_t sb=(size_t)r*s.H*s.D*s.D, ce=(size_t)(r+1)*s.H*s.D*s.W;
        if (parked[r]) {
            for(size_t i=rb;i<re;++i) poison_ok &= hyf[i]==poison[i] && hq[i]==poison[i] && ho[i]==poison[i];
            for(size_t i=sb;i<sb+(size_t)s.H*s.D*s.D;++i) poison_ok &= hs1[i]==state[i] && hs0[i]==state[i];
            for(size_t i=(size_t)r*s.H*s.D*s.W;i<ce;++i)
                poison_ok &= hcq1[i]==csq[i] && hck1[i]==csk[i] && hcv1[i]==csv[i] && hcq0[i]==csq[i] && hck0[i]==csk[i] && hcv0[i]==csv[i];
        } else {
            for(size_t i=rb;i<re;++i) { double d=b2f(hyf[i])-b2f(hyc[i]); ye+=d*d; yw+=(double)b2f(hyc[i])*b2f(hyc[i]); finite &= std::isfinite(b2f(hyf[i])); }
            for(size_t i=sb;i<sb+(size_t)s.H*s.D*s.D;++i) { double d=hs1[i]-hs0[i]; se+=d*d; sw+=(double)hs0[i]*hs0[i]; finite &= std::isfinite(hs1[i]); }
            for(size_t i=(size_t)r*s.H*s.D*s.W;i<ce;++i) conv_exact &= hcq1[i]==hcq0[i] && hck1[i]==hck0[i] && hcv1[i]==hcv0[i];
        }
    }
    const double yr=sqrt(ye/(yw+1e-30)), sr=sqrt(se/(sw+1e-30));
    bool ok=yr<=1e-6 && sr<=1e-6 && poison_ok && finite && conv_exact;
    printf("B=%u H=%u D=%u W=%u y=%.3e state=%.3e conv=%s poison=%s finite=%s %s\n",
           s.B,s.H,s.D,s.W,yr,sr,conv_exact?"exact":"DIFF",poison_ok?"ok":"FAIL",finite?"yes":"NO",ok?"PASS":"FAIL");
    if (ok) {
        constexpr unsigned warmup_pairs = 4, sample_pairs = 15, iterations = 32;
        hipEvent_t begin, end;
        HC(hipEventCreate(&begin)); HC(hipEventCreate(&end));
        std::vector<double> ct, ft;
        auto reset_arm = [&](bool is_fused) -> bool {
            float *aq=is_fused?csq1:csq0, *ak=is_fused?csk1:csk0;
            float *av=is_fused?csv1:csv0, *ast=is_fused?st1:st0;
            HC(hipMemcpy(aq,csqb,csn*4,hipMemcpyDeviceToDevice));
            HC(hipMemcpy(ak,cskb,csn*4,hipMemcpyDeviceToDevice));
            HC(hipMemcpy(av,csvb,csn*4,hipMemcpyDeviceToDevice));
            HC(hipMemcpy(ast,stb,sn*4,hipMemcpyDeviceToDevice));
            return true;
        };
        auto measure = [&](bool is_fused, double& us) -> bool {
            if (!reset_arm(is_fused)) return false;
            HC(hipDeviceSynchronize());
            HC(hipEventRecord(begin,nullptr));
            for(unsigned i=0;i<iterations;++i)
                if (!(is_fused ? fused() : control())) return false;
            HC(hipEventRecord(end,nullptr)); HC(hipEventSynchronize(end));
            float ms=0; HC(hipEventElapsedTime(&ms,begin,end));
            us=(double)ms*1000.0/iterations;
            return true;
        };
        for(unsigned pair=0;pair<warmup_pairs+sample_pairs;++pair) {
            double ctime=0,ftime=0;
            const bool control_first=(pair&1u)==0;
            if (control_first) {
                if(!measure(false,ctime)||!measure(true,ftime)) return false;
            } else {
                if(!measure(true,ftime)||!measure(false,ctime)) return false;
            }
            if(pair>=warmup_pairs) { ct.push_back(ctime); ft.push_back(ftime); }
        }
        HC(hipEventDestroy(begin)); HC(hipEventDestroy(end));
        const TimingVerdict tv=timing_verdict(ct,ft);
        printf("  timing control=%.3fus fused=%.3fus ratio=%.4f mad=%.4f wins=%u/%zu threshold=0.9700 %s\n",
               median(ct),median(ft),tv.ratio,tv.mad,tv.wins,ct.size(),tv.pass?"PASS":"FAIL");
        ok=tv.pass;
    }
    for(void* p0:{(void*)dq,(void*)dk,(void*)dv,(void*)dg,(void*)db,(void*)dog,(void*)dwq,(void*)dwk,(void*)dwv,(void*)da,(void*)ddt,(void*)dnw,(void*)dp,(void*)csq0,(void*)csk0,(void*)csv0,(void*)st0,(void*)csq1,(void*)csk1,(void*)csv1,(void*)st1,(void*)csqb,(void*)cskb,(void*)csvb,(void*)stb,(void*)q,(void*)k,(void*)v,(void*)o,(void*)yc,(void*)yf,(void*)dc,(void*)df}) hipFree(p0);
    return ok;
}

int main(int argc, char** argv) {
    if (argc == 2 && !strcmp(argv[1],"--selftest-timing")) return timing_selftest();
    if (argc != 2) { fprintf(stderr,"usage: %s kda_decode_fused_poc.co\n",argv[0]); return 2; }
    hipDeviceProp_t p; if (hipGetDeviceProperties(&p,0)!=hipSuccess) return 2;
    if (!strstr(p.gcnArchName,"gfx950")) { fprintf(stderr,"refusing non-gfx950 device: %s\n",p.gcnArchName); return 2; }
    hipModule_t module; if (hipModuleLoad(&module,argv[1])!=hipSuccess) return 2;
    const Shape shapes[]={{1,1,64,1},{5,3,128,4},{7,2,256,8},{9,12,128,3}};
    bool ok=true; for(auto s:shapes) ok=run_shape(module,s)&&ok;
    puts(ok?"PASS":"FAIL"); return ok?0:1;
}
#endif
