/* sm120_numeric_test.cu — numerically validate the registered sm_120 bf16
 * kernels (gemma_sm120.cu, wired by register_rtx6000.cu) against the CPU golden
 * f32 oracle (runtime/cpu/*.c + runtime/common/interp.c).
 *
 * Method, per kernel:
 *   1. generate f32 inputs, ROUND THEM TO bf16 first. The rounded values (widened
 *      back to f32) are what the CPU oracle consumes, so input quantization is
 *      NOT charged against the kernel — only the kernel's own arithmetic is.
 *   2. run the CPU oracle (dispatch table from plow_register_cpu via
 *      plow_interp_run on a real 1-instruction wire stream, where a base opcode
 *      exists; composed *_ref calls where the sm_120 op is a fusion with no
 *      single CPU counterpart).
 *   3. run the SAME wire packet through the sm_120 kernel via the dispatch table
 *      from plow_register_cuda_rtx6000 + plow_interp_run.
 *   4. diff: max abs err, max rel err, argmax index.
 *
 * max rel err is |gpu-ref| / max(|ref|, 0.01*max|ref|) — the floor keeps
 * near-zero outputs from producing meaningless ratios. Both the floor and the
 * raw max abs err are printed so nothing is hidden.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

#include "packet.h"
#include "kernel.h"
#include "dispatch.h"
#include "interp.h"

extern "C" void plow_register_cuda_rtx6000(dispatch_table* dt);

typedef __nv_bfloat16 bf16;

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERROR %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
           cudaGetErrorString(e_)); exit(2); } } while (0)

/* ---- wire stream builder (same layout as Program::to_bytes) --------------- */
struct Buf {
    std::vector<uint8_t> p;
    void put(const void* s, size_t n) { const uint8_t* b=(const uint8_t*)s; p.insert(p.end(), b, b+n); }
    void u32(uint32_t v) { put(&v,4); }
    void u16(uint16_t v) { put(&v,2); }
};
static Buf build_1inst(uint16_t opcode, const void* body, size_t bsz) {
    Buf b;
    b.u32(PLOW_MAGIC); b.u16(PLOW_VERSION); b.u16(0);
    b.u32(1); b.u32(0); b.u16(0); b.u16(0);
    PlowHeader h = { opcode, PLOW_RES_SM, 0, 0, 0, 0, 0 };
    b.put(&h, sizeof(h));
    if (bsz) b.put(body, bsz);
    return b;
}

/* ---- host bf16 round-trip ------------------------------------------------ */
static float bf16_rt(float x) { return __bfloat162float(__float2bfloat16(x)); }

static uint32_t rng_s = 0x12345678u;
static float rnd() { /* xorshift -> [-1,1) */
    rng_s ^= rng_s<<13; rng_s ^= rng_s>>17; rng_s ^= rng_s<<5;
    return (float)((int32_t)rng_s) / 2147483648.0f;
}
static void seed(uint32_t s) { rng_s = s ? s : 1; }

/* inputs rounded to bf16 so the oracle sees exactly what the GPU sees */
static std::vector<float> gen_bf16(size_t n, float amp) {
    std::vector<float> v(n);
    for (size_t i=0;i<n;i++) v[i] = bf16_rt(rnd()*amp);
    return v;
}
/* small integers: exactly representable in bf16, products/sums exact in f32 */
static std::vector<float> gen_int(size_t n, int lo, int hi) {
    std::vector<float> v(n);
    int span = hi-lo+1;
    for (size_t i=0;i<n;i++) { int r=(int)((rnd()*0.5f+0.5f)*span); if(r>=span)r=span-1; v[i]=(float)(lo+r); }
    return v;
}

/* ---- device helpers ------------------------------------------------------ */
static bf16* to_dev_bf16(const std::vector<float>& h) {
    std::vector<bf16> t(h.size());
    for (size_t i=0;i<h.size();i++) t[i] = __float2bfloat16(h[i]);
    bf16* d=nullptr; CUDA_OK(cudaMalloc(&d, h.size()*sizeof(bf16)));
    CUDA_OK(cudaMemcpy(d, t.data(), h.size()*sizeof(bf16), cudaMemcpyHostToDevice));
    return d;
}
static float* dev_f32(size_t n) {
    float* d=nullptr; CUDA_OK(cudaMalloc(&d, n*sizeof(float)));
    CUDA_OK(cudaMemset(d, 0, n*sizeof(float))); return d;
}
static bf16* dev_bf16(size_t n) {
    bf16* d=nullptr; CUDA_OK(cudaMalloc(&d, n*sizeof(bf16)));
    CUDA_OK(cudaMemset(d, 0, n*sizeof(bf16))); return d;
}
static std::vector<float> from_dev_bf16(const bf16* d, size_t n) {
    std::vector<bf16> t(n);
    CUDA_OK(cudaMemcpy(t.data(), d, n*sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<float> h(n); for (size_t i=0;i<n;i++) h[i]=__bfloat162float(t[i]);
    return h;
}
static std::vector<float> from_dev_f32(const float* d, size_t n) {
    std::vector<float> h(n);
    CUDA_OK(cudaMemcpy(h.data(), d, n*sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

/* ---- diff + report ------------------------------------------------------- */
static int g_fail = 0;
static const double TOL = 1e-2;

static void report(const char* name, const std::vector<float>& ref,
                   const std::vector<float>& got, bool expect_pass = true) {
    if (ref.size() != got.size()) { printf("  %-34s SIZE MISMATCH\n", name); g_fail=1; return; }
    double mx = 0; for (float v : ref) mx = fmax(mx, fabs((double)v));
    double floor_ = 0.01*mx;
    double maxabs=0, maxrel=0; size_t ia=0, ir=0;
    size_t nan_ct=0;
    for (size_t i=0;i<ref.size();i++) {
        double r=ref[i], g=got[i];
        if (std::isnan(g) || std::isinf(g)) { nan_ct++; continue; }
        double a=fabs(g-r);
        if (a>maxabs){maxabs=a; ia=i;}
        double den = fmax(fabs(r), floor_);
        double rel = den>0 ? a/den : (a>0?1e30:0);
        if (rel>maxrel){maxrel=rel; ir=i;}
    }
    bool pass = (maxrel <= TOL) && nan_ct==0;
    printf("  %-34s n=%-9zu max|ref|=%-11.5g maxabs=%-11.5g maxrel=%-11.5g "
           "argmax_abs=%zu(ref=%.6g got=%.6g) argmax_rel=%zu(ref=%.6g got=%.6g)%s  -> %s\n",
           name, ref.size(), mx, maxabs, maxrel,
           ia, ref.size()?(double)ref[ia]:0.0, ref.size()?(double)got[ia]:0.0,
           ir, ref.size()?(double)ref[ir]:0.0, ref.size()?(double)got[ir]:0.0,
           nan_ct? (std::string(" NaN/Inf=")+std::to_string(nan_ct)).c_str() : "",
           pass?"PASS":"FAIL");
    if (pass != expect_pass) g_fail = 1;
}

/* ---- oracle helpers not covered by a single CPU opcode ------------------- */
static void rmsnorm_rows(std::vector<float>& out, const std::vector<float>& x,
                         const std::vector<float>& gamma, uint32_t rows, uint32_t k, float eps) {
    out.resize(x.size());
    for (uint32_t r=0;r<rows;r++){
        double ss=0; for (uint32_t i=0;i<k;i++){ double v=x[(size_t)r*k+i]; ss+=v*v; }
        float inv = 1.0f/sqrtf((float)(ss/(double)k) + eps);
        for (uint32_t i=0;i<k;i++) out[(size_t)r*k+i] = x[(size_t)r*k+i]*inv*gamma[i];
    }
}

/* =========================================================================== */
static dispatch_table dt_cpu, dt_gpu;
static cudaStream_t g_stream;

static kctx mk_ctx(void** slots, uint32_t n, void* stream) {
    kctx c; memset(&c,0,sizeof(c)); c.slots=slots; c.n_slots=n; c.stream=stream; return c;
}

static int run_gpu(const Buf& b, void** slots, uint32_t n, const PlowBinding* bind) {
    kctx c = mk_ctx(slots, n, g_stream);
    int rc = plow_interp_run(b.p.data(), b.p.size(), &dt_gpu, &c, bind, 1);
    if (rc != 0) return rc;
    cudaError_t e = cudaStreamSynchronize(g_stream);
    if (e != cudaSuccess) { printf("  [launch/exec error: %s]\n", cudaGetErrorString(e)); return -100; }
    return 0;
}
static int run_cpu(const Buf& b, void** slots, uint32_t n, const PlowBinding* bind) {
    kctx c = mk_ctx(slots, n, nullptr);
    return plow_interp_run(b.p.data(), b.p.size(), &dt_cpu, &c, bind, 1);
}

/* --------------------------------------------------------------- GEMM bf16 */
static void test_gemm_bf16(const char* label, uint32_t M, uint32_t N, uint32_t K,
                           bool exact, int perturb_idx = -1) {
    seed(0xA1B2C300u + M + N + K);
    std::vector<float> A = exact ? gen_int((size_t)M*K, -2, 2) : gen_bf16((size_t)M*K, 1.0f);
    std::vector<float> B = exact ? gen_int((size_t)N*K, -2, 2) : gen_bf16((size_t)N*K, 1.0f);

    /* CPU oracle via the real wire packet + plow_register_cpu */
    std::vector<float> Cref((size_t)M*N, 0.f);
    PlowGemmBody g = {0,0,M,N,K,(uint16_t)M,(uint16_t)N,(uint16_t)K,/*out*/2,PLOW_SLOT_NONE,0};
    Buf pkt_cpu = build_1inst(PLOW_GEMM, &g, sizeof(g));
    PlowBinding bcpu = {0,1,PLOW_SLOT_NONE,PLOW_ACT_NONE,0,0.0f,0};
    void* cs[3] = { A.data(), B.data(), Cref.data() };
    if (run_cpu(pkt_cpu, cs, 3, &bcpu) != 0) { printf("  %s: CPU oracle failed\n", label); g_fail=1; return; }

    /* GPU: same body, fused opcode */
    std::vector<float> Bg = B;
    if (perturb_idx >= 0) Bg[perturb_idx] = bf16_rt(Bg[perturb_idx] + 1.0f);
    bf16 *dA=to_dev_bf16(A), *dB=to_dev_bf16(Bg), *dC=dev_bf16((size_t)M*N);
    Buf pkt = build_1inst(PLOW_OP(0,PLOW_FAMILY_GEMM,PLOW_VARIANT_BF16), &g, sizeof(g));
    PlowBinding bg = {0,1,PLOW_SLOT_NONE,PLOW_ACT_NONE,0,0.0f,0};
    void* gs[3] = { dA, dB, dC };
    int rc = run_gpu(pkt, gs, 3, &bg);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Cref, from_dev_bf16(dC,(size_t)M*N), perturb_idx < 0);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

/* ---------------------------------------------------------- GEMM norm bf16 */
static void test_gemm_norm_bf16(const char* label, uint32_t M, uint32_t N, uint32_t K) {
    seed(0xB2C3D400u + M + N + K);
    const float eps = 1e-6f;
    std::vector<float> A = gen_bf16((size_t)M*K, 1.0f);
    std::vector<float> W = gen_bf16((size_t)N*K, 1.0f);
    std::vector<float> gam = gen_bf16(K, 1.0f);

    /* oracle: RMSNorm(A) rounded to bf16 (the kernel stages the normed A as bf16
     * before the mma), then plow_gemm_ref. */
    std::vector<float> An; rmsnorm_rows(An, A, gam, M, K, eps);
    for (float& v : An) v = bf16_rt(v);
    std::vector<float> Cref((size_t)M*N, 0.f);
    plow_gemm_ref(Cref.data(), An.data(), W.data(), nullptr, M, N, K, PLOW_ACT_NONE);

    bf16 *dA=to_dev_bf16(A), *dW=to_dev_bf16(W), *dG=to_dev_bf16(gam), *dC=dev_bf16((size_t)M*N);
    PlowGemmBody g = {0,0,M,N,K,(uint16_t)M,(uint16_t)N,(uint16_t)K,/*out*/3,PLOW_SLOT_NONE,0};
    Buf pkt = build_1inst(PLOW_OP(0,PLOW_FAMILY_GEMM,PLOW_VARIANT_NORM_BF16), &g, sizeof(g));
    PlowBinding bd = {0,1,2,PLOW_ACT_NONE,0,eps,0};
    void* gs[4] = { dA, dW, dG, dC };
    int rc = run_gpu(pkt, gs, 4, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Cref, from_dev_bf16(dC,(size_t)M*N));
    cudaFree(dA); cudaFree(dW); cudaFree(dG); cudaFree(dC);
}

/* -------------------------------------------------------- GEMM split-K bf16 */
static void test_gemm_splitk(const char* label, uint32_t M, uint32_t N, uint32_t K,
                             int split, bool norm) {
    seed(0xC3D4E500u + M + N + K + split);
    const float eps = 1e-6f;
    std::vector<float> A = gen_bf16((size_t)M*K, 1.0f);
    std::vector<float> W = gen_bf16((size_t)N*K, 1.0f);
    std::vector<float> gam = gen_bf16(K, 1.0f);

    std::vector<float> Aeff = A;
    if (norm) { rmsnorm_rows(Aeff, A, gam, M, K, eps); for (float& v:Aeff) v=bf16_rt(v); }
    std::vector<float> Cref((size_t)M*N, 0.f);
    plow_gemm_ref(Cref.data(), Aeff.data(), W.data(), nullptr, M, N, K, PLOW_ACT_NONE);

    bf16 *dA=to_dev_bf16(A), *dW=to_dev_bf16(W), *dG=to_dev_bf16(gam);
    float* dCp = dev_f32((size_t)split*M*N);
    PlowGemmBody g = {0, /*coord1: split in high 16*/ (uint32_t)split<<16,
                      M,N,K,(uint16_t)M,(uint16_t)N,(uint16_t)K,/*out*/3,PLOW_SLOT_NONE,0};
    uint16_t op = norm ? PLOW_OP(0,PLOW_FAMILY_GEMM,PLOW_VARIANT_NORM_SPLITK_BF16)
                       : PLOW_OP(0,PLOW_FAMILY_GEMM,PLOW_VARIANT_BF16_SPLITK);
    Buf pkt = build_1inst(op, &g, sizeof(g));
    PlowBinding bd = {0,1,(uint16_t)(norm?2:PLOW_SLOT_NONE),PLOW_ACT_NONE,0,eps,0};
    void* gs[4] = { dA, dW, dG, dCp };
    int rc = run_gpu(pkt, gs, 4, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else {
        std::vector<float> parts = from_dev_f32(dCp, (size_t)split*M*N);
        std::vector<float> Cgot((size_t)M*N, 0.f);
        for (int s=0;s<split;s++)
            for (size_t i=0;i<(size_t)M*N;i++) Cgot[i] += parts[(size_t)s*M*N+i];
        report(label, Cref, Cgot);
    }
    cudaFree(dA); cudaFree(dW); cudaFree(dG); cudaFree(dCp);
}

/* --------------------------------------------------------------- FLASH */
static void test_flash_prefill(const char* label, uint32_t heads, uint32_t sq, uint32_t skv, int perturb_idx = -1) {
    const uint32_t HD = 128;
    seed(0xD4E5F600u + heads + sq);
    std::vector<float> Q = gen_bf16((size_t)heads*sq*HD, 1.0f);
    std::vector<float> K = gen_bf16((size_t)heads*skv*HD, 1.0f);
    std::vector<float> V = gen_bf16((size_t)heads*skv*HD, 1.0f);

    /* CPU oracle through the real wire packet + plow_register_cpu (causal). */
    std::vector<float> Oref((size_t)heads*sq*HD, 0.f);
    PlowFlashBody f = {0,0,sq,skv,(uint16_t)HD,64,64,(uint16_t)heads,/*out*/3,/*tmem=kv_heads*/(uint16_t)heads};
    Buf pc = build_1inst(PLOW_FLASH, &f, sizeof(f));
    PlowBinding bc = {0,1,2,/*detail: causal*/1,0,0.0f,0};
    void* cs[4] = { Q.data(), K.data(), V.data(), Oref.data() };
    if (run_cpu(pc, cs, 4, &bc) != 0) { printf("  %s: CPU oracle failed\n", label); g_fail=1; return; }

    std::vector<float> Kg = K;
    if (perturb_idx >= 0) Kg[perturb_idx] = bf16_rt(Kg[perturb_idx] + 1.0f);
    bf16 *dQ=to_dev_bf16(Q), *dK=to_dev_bf16(Kg), *dV=to_dev_bf16(V);
    bf16* dO=dev_bf16((size_t)heads*sq*HD);
    Buf pkt = build_1inst(PLOW_OP(0,PLOW_FAMILY_FLASH,PLOW_VARIANT_FLASH_CAUSAL_BF16), &f, sizeof(f));
    PlowBinding bd = {0,1,2,0,0,/*scale*/1.0f/sqrtf((float)HD),0};
    void* gs[4] = { dQ, dK, dV, dO };
    int rc = run_gpu(pkt, gs, 4, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Oref, from_dev_bf16(dO,(size_t)heads*sq*HD), perturb_idx < 0);
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
}

/* Sliding-window causal attention. No CPU opcode carries a window, so the
 * reference is written out here: key j attends iff j <= i AND i - j < window
 * (matches the kernel's mask: masked |= (kv > q_abs); masked |= (q_abs-kv >= window)). */
static void test_flash_sliding(const char* label, uint32_t heads, uint32_t sq,
                               uint32_t skv, uint32_t window) {
    const uint32_t HD = 128;
    seed(0xF6070800u + heads + sq + window);
    std::vector<float> Q = gen_bf16((size_t)heads*sq*HD, 1.0f);
    std::vector<float> K = gen_bf16((size_t)heads*skv*HD, 1.0f);
    std::vector<float> V = gen_bf16((size_t)heads*skv*HD, 1.0f);
    float scale = 1.0f/sqrtf((float)HD);

    std::vector<float> Oref((size_t)heads*sq*HD, 0.f);
    std::vector<float> row(skv);
    for (uint32_t h=0;h<heads;h++) for (uint32_t i=0;i<sq;i++) {
        const float* qh=&Q[((size_t)h*sq+i)*HD];
        float mx=-INFINITY; uint32_t lo = (i>=window)? i-window+1 : 0;
        for (uint32_t j=lo;j<=i && j<skv;j++){
            double s=0; for(uint32_t d=0;d<HD;d++) s += (double)qh[d]*K[((size_t)h*skv+j)*HD+d];
            row[j]=(float)s*scale; if(row[j]>mx) mx=row[j];
        }
        float sum=0; for(uint32_t j=lo;j<=i&&j<skv;j++){ row[j]=expf(row[j]-mx); sum+=row[j]; }
        for(uint32_t d=0;d<HD;d++){ double acc=0;
            for(uint32_t j=lo;j<=i&&j<skv;j++) acc += (row[j]/sum)*V[((size_t)h*skv+j)*HD+d];
            Oref[((size_t)h*sq+i)*HD+d]=(float)acc; }
    }



    bf16 *dQ=to_dev_bf16(Q), *dK=to_dev_bf16(K), *dV=to_dev_bf16(V);
    bf16* dO=dev_bf16((size_t)heads*sq*HD);
    PlowFlashBody f = {/*coord0=window*/window,0,sq,skv,(uint16_t)HD,64,64,
                       (uint16_t)heads,/*out*/3,(uint16_t)heads};
    Buf pkt = build_1inst(PLOW_OP(0,PLOW_FAMILY_FLASH,PLOW_VARIANT_FLASH_SLIDING_BF16), &f, sizeof(f));
    PlowBinding bd = {0,1,2,0,0,scale,0};
    void* gs[4] = { dQ, dK, dV, dO };
    int rc = run_gpu(pkt, gs, 4, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Oref, from_dev_bf16(dO,(size_t)heads*sq*HD));
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
}

static void test_flash_decode(const char* label, uint32_t heads, uint32_t skv) {
    const uint32_t HD = 128, sq = 1;
    seed(0xE5F60700u + heads + skv);
    std::vector<float> Q = gen_bf16((size_t)heads*sq*HD, 1.0f);
    std::vector<float> K = gen_bf16((size_t)heads*skv*HD, 1.0f);
    std::vector<float> V = gen_bf16((size_t)heads*skv*HD, 1.0f);

    std::vector<float> Oref((size_t)heads*sq*HD, 0.f);
    PlowFlashBody f = {0,0,sq,skv,(uint16_t)HD,1,64,(uint16_t)heads,/*out*/3,(uint16_t)heads};
    Buf pc = build_1inst(PLOW_FLASH, &f, sizeof(f));
    PlowBinding bc = {0,1,2,/*causal (sq=1: no-op)*/0,0,0.0f,0};
    void* cs[4] = { Q.data(), K.data(), V.data(), Oref.data() };
    if (run_cpu(pc, cs, 4, &bc) != 0) { printf("  %s: CPU oracle failed\n", label); g_fail=1; return; }

    bf16 *dQ=to_dev_bf16(Q), *dK=to_dev_bf16(K), *dV=to_dev_bf16(V);
    bf16* dO=dev_bf16((size_t)heads*HD);
    Buf pkt = build_1inst(PLOW_OP(0,PLOW_FAMILY_FLASH,PLOW_VARIANT_FLASH_DECODE_BF16), &f, sizeof(f));
    PlowBinding bd = {0,1,2,0,0,1.0f/sqrtf((float)HD),0};
    void* gs[4] = { dQ, dK, dV, dO };
    int rc = run_gpu(pkt, gs, 4, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Oref, from_dev_bf16(dO,(size_t)heads*HD));
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
}

/* --------------------------------------------------------------- ROW ops */
static void test_row_rmsnorm(const char* label, uint32_t rows, uint32_t feat,
                             int perturb_idx = -1) {
    seed(0x11223344u + rows + feat);
    const float eps = 1e-6f;
    std::vector<float> X = gen_bf16((size_t)rows*feat, 1.0f);
    std::vector<float> gam = gen_bf16(feat, 1.0f);

    std::vector<float> Oref((size_t)rows*feat, 0.f);
    PlowRowBody r = {0, rows, feat, 0, /*out*/2, /*operands*/2, {0,0,0}};
    Buf pc = build_1inst(PLOW_ROW_REDUCE, &r, sizeof(r));
    PlowBinding bc = {0,1,PLOW_SLOT_NONE,PLOW_NORM_RMS,0,eps,0};
    void* cs[3] = { X.data(), gam.data(), Oref.data() };
    if (run_cpu(pc, cs, 3, &bc) != 0) { printf("  %s: CPU oracle failed\n", label); g_fail=1; return; }

    std::vector<float> gamg = gam;
    if (perturb_idx >= 0) gamg[perturb_idx] = bf16_rt(gamg[perturb_idx] + 1.0f);
    bf16 *dX=to_dev_bf16(X), *dG=to_dev_bf16(gamg), *dO=dev_bf16((size_t)rows*feat);
    Buf pkt = build_1inst(PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_RMS_BF16), &r, sizeof(r));
    PlowBinding bd = {0,1,PLOW_SLOT_NONE,0,0,eps,0};
    void* gs[3] = { dX, dG, dO };
    int rc = run_gpu(pkt, gs, 3, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Oref, from_dev_bf16(dO,(size_t)rows*feat), perturb_idx < 0);
    cudaFree(dX);cudaFree(dG);cudaFree(dO);
}

static void test_row_residual(const char* label, uint32_t rows, uint32_t feat) {
    seed(0x22334455u + rows + feat);
    std::vector<float> X = gen_bf16((size_t)rows*feat, 1.0f);
    std::vector<float> R = gen_bf16((size_t)rows*feat, 1.0f);

    std::vector<float> Oref((size_t)rows*feat, 0.f);
    PlowRowBody r = {0, rows, feat, 0, /*out*/2, /*operands*/2, {0,0,0}};
    Buf pc = build_1inst(PLOW_ROW_POINTWISE, &r, sizeof(r));
    PlowBinding bc = {0,1,PLOW_SLOT_NONE,(uint8_t)(PLOW_EW_ADD<<4),0,0.0f,0};
    void* cs[3] = { X.data(), R.data(), Oref.data() };
    if (run_cpu(pc, cs, 3, &bc) != 0) { printf("  %s: CPU oracle failed\n", label); g_fail=1; return; }

    bf16 *dX=to_dev_bf16(X), *dR=to_dev_bf16(R), *dO=dev_bf16((size_t)rows*feat);
    Buf pkt = build_1inst(PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_RESIDUAL_ADD_BF16), &r, sizeof(r));
    PlowBinding bd = {0,1,PLOW_SLOT_NONE,0,0,0.0f,0};
    void* gs[3] = { dX, dR, dO };
    int rc = run_gpu(pkt, gs, 3, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Oref, from_dev_bf16(dO,(size_t)rows*feat));
    cudaFree(dX);cudaFree(dR);cudaFree(dO);
}

static void test_row_swiglu(const char* label, uint32_t rows, uint32_t feat) {
    seed(0x33445566u + rows + feat);
    size_t n = (size_t)rows*feat;
    std::vector<float> G = gen_bf16(n, 2.0f);
    std::vector<float> U = gen_bf16(n, 2.0f);
    /* oracle: silu(gate)*up, composed from the CPU pointwise ref */
    std::vector<float> sg(n), Oref(n);
    plow_row_pointwise_ref(sg.data(), G.data(), nullptr, (uint32_t)n, PLOW_ACT_SILU, 0);
    plow_row_pointwise_ref(Oref.data(), sg.data(), U.data(), (uint32_t)n, 0, PLOW_EW_MUL);

    bf16 *dG=to_dev_bf16(G), *dU=to_dev_bf16(U), *dO=dev_bf16(n);
    PlowRowBody r = {0, rows, feat, 0, /*out*/2, 2, {0,0,0}};
    Buf pkt = build_1inst(PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_SWIGLU_BF16), &r, sizeof(r));
    PlowBinding bd = {0,1,PLOW_SLOT_NONE,0,0,0.0f,0};
    void* gs[3] = { dG, dU, dO };
    int rc = run_gpu(pkt, gs, 3, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Oref, from_dev_bf16(dO,n));
    cudaFree(dG);cudaFree(dU);cudaFree(dO);
}

/* NormRope(Scale) — fused, no single CPU opcode; reference mirrors the
 * documented semantics (RMSNorm over head_dim, RoPE over first head_dim/2 dims
 * pairing (i, i+half), theta hardcoded 1e6 in the launcher). */
static void test_row_normrope(const char* label, uint32_t rows, uint32_t feat,
                              uint32_t head_dim, bool scale_variant, uint32_t pos_base) {
    seed(0x44556677u + rows + feat + head_dim + (scale_variant?1:0));
    const float eps = 1e-6f, theta = 1e6f;
    size_t n = (size_t)rows*feat;
    std::vector<float> X = gen_bf16(n, 1.0f);
    std::vector<float> gam = gen_bf16(head_dim, 1.0f);
    uint32_t heads = feat/head_dim, rot = head_dim/2, half = rot/2;
    float outscale = scale_variant ? 1.0f/sqrtf((float)head_dim) : 1.0f;

    std::vector<float> Oref(n, 0.f);
    for (uint32_t r=0;r<rows;r++) for (uint32_t h=0;h<heads;h++) {
        const float* xr = &X[(size_t)r*feat + (size_t)h*head_dim];
        std::vector<float> nv(head_dim);
        double ss=0; for (uint32_t d=0;d<head_dim;d++) ss += (double)xr[d]*xr[d];
        float inv = 1.0f/sqrtf((float)(ss/(double)head_dim)+eps);
        for (uint32_t d=0;d<head_dim;d++) nv[d] = xr[d]*inv*gam[d];
        uint32_t pos = pos_base + r;
        for (uint32_t i=0;i<half;i++) {
            float freq = powf(theta, -2.0f*(float)i/(float)rot);
            float ang = (float)pos*freq, c=cosf(ang), s=sinf(ang);
            float x0=nv[i], x1=nv[i+half];
            nv[i]=x0*c-x1*s; nv[i+half]=x0*s+x1*c;
        }
        for (uint32_t d=0;d<head_dim;d++) Oref[(size_t)r*feat+(size_t)h*head_dim+d] = nv[d]*outscale;
    }

    bf16 *dX=to_dev_bf16(X), *dG=to_dev_bf16(gam), *dO=dev_bf16(n);
    PlowRowBody r = {pos_base, rows, feat, (uint16_t)head_dim, /*out*/2, 2, {0,0,0}};
    uint16_t op = scale_variant ? PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_NORMROPESCALE_BF16)
                                : PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_NORMROPE_BF16);
    Buf pkt = build_1inst(op, &r, sizeof(r));
    PlowBinding bd = {0,1,PLOW_SLOT_NONE,0,0,eps,0};
    void* gs[3] = { dX, dG, dO };
    int rc = run_gpu(pkt, gs, 3, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Oref, from_dev_bf16(dO,n));
    cudaFree(dX);cudaFree(dG);cudaFree(dO);
}

static void test_layout_gather_scale(const char* label, uint32_t tokens, uint32_t d,
                                     uint32_t vocab) {
    seed(0x55667788u + tokens + d);
    std::vector<float> table = gen_bf16((size_t)vocab*d, 1.0f);
    std::vector<int32_t> ids(tokens);
    for (uint32_t t=0;t<tokens;t++) ids[t] = (int)((rnd()*0.5f+0.5f)*vocab) % vocab;
    float sc = sqrtf((float)d);

    std::vector<float> Oref((size_t)tokens*d);
    for (uint32_t t=0;t<tokens;t++)
        for (uint32_t i=0;i<d;i++) Oref[(size_t)t*d+i] = table[(size_t)ids[t]*d+i]*sc;

    int32_t* dI=nullptr; CUDA_OK(cudaMalloc(&dI, tokens*sizeof(int32_t)));
    CUDA_OK(cudaMemcpy(dI, ids.data(), tokens*sizeof(int32_t), cudaMemcpyHostToDevice));
    bf16 *dT=to_dev_bf16(table), *dO=dev_bf16((size_t)tokens*d);
    PlowLayoutBody L; memset(&L,0,sizeof(L));
    L.kind=1; L.rank=2; L.elem_size=2; L.out=2;
    L.shape[0]=tokens; L.shape[1]=d;
    Buf pkt = build_1inst(PLOW_OP(0,PLOW_FAMILY_LAYOUT,PLOW_VARIANT_LAYOUT_GATHER_SCALE_BF16), &L, sizeof(L));
    PlowBinding bd = {0,1,PLOW_SLOT_NONE,0,0,sc,0};
    void* gs[3] = { dI, dT, dO };
    int rc = run_gpu(pkt, gs, 3, &bd);
    if (rc != 0) { printf("  %-34s DISPATCH/RUN FAILED rc=%d\n", label, rc); g_fail=1; }
    else report(label, Oref, from_dev_bf16(dO,(size_t)tokens*d));
    cudaFree(dI);cudaFree(dT);cudaFree(dO);
}

/* =========================================================================== */
/* A CUDA "misaligned address" fault is STICKY: it poisons the context and every
 * subsequent call fails. So each case runs in its own process, selected by argv.
 * `main` with no argument prints the case list; the driver script runs each. */
static const char* g_sel = nullptr;
static int g_matched = 0;
static bool want(const char* k) {
    if (!g_sel) return true;
    if (strcmp(g_sel, k) != 0) return false;
    g_matched = 1;
    return true;
}

int main(int argc, char** argv) {
    if (argc > 1) g_sel = argv[1];
    int dev=0; cudaDeviceProp prop;
    CUDA_OK(cudaGetDevice(&dev));
    CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s  sm_%d%d  SMs=%d  smem/block(optin)=%zu KiB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (size_t)prop.sharedMemPerBlockOptin/1024);
    printf("PLOW_SM120_SMS (persistent grid compiled into gemma_sm120.cu) = %d\n",
#ifdef PLOW_SM120_SMS
           PLOW_SM120_SMS
#else
           188
#endif
    );
    CUDA_OK(cudaStreamCreate(&g_stream));
    plow_register_cpu(&dt_cpu);
    plow_register_cuda_rtx6000(&dt_gpu);

    /* confirm every slot we intend to test is actually registered */
    struct { const char* n; uint16_t op; } regs[] = {
      {"GEMM bf16",          PLOW_OP(0,PLOW_FAMILY_GEMM,PLOW_VARIANT_BF16)},
      {"GEMM norm bf16",     PLOW_OP(0,PLOW_FAMILY_GEMM,PLOW_VARIANT_NORM_BF16)},
      {"GEMM bf16 splitk",   PLOW_OP(0,PLOW_FAMILY_GEMM,PLOW_VARIANT_BF16_SPLITK)},
      {"GEMM norm splitk",   PLOW_OP(0,PLOW_FAMILY_GEMM,PLOW_VARIANT_NORM_SPLITK_BF16)},
      {"FLASH causal bf16",  PLOW_OP(0,PLOW_FAMILY_FLASH,PLOW_VARIANT_FLASH_CAUSAL_BF16)},
      {"FLASH sliding bf16", PLOW_OP(0,PLOW_FAMILY_FLASH,PLOW_VARIANT_FLASH_SLIDING_BF16)},
      {"FLASH decode bf16",  PLOW_OP(0,PLOW_FAMILY_FLASH,PLOW_VARIANT_FLASH_DECODE_BF16)},
      {"ROW normrope",       PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_NORMROPE_BF16)},
      {"ROW normropescale",  PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_NORMROPESCALE_BF16)},
      {"ROW swiglu",         PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_SWIGLU_BF16)},
      {"ROW residual",       PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_RESIDUAL_ADD_BF16)},
      {"ROW rmsnorm",        PLOW_OP(0,PLOW_FAMILY_ROW,PLOW_VARIANT_ROW_RMS_BF16)},
      {"LAYOUT gather_scale",PLOW_OP(0,PLOW_FAMILY_LAYOUT,PLOW_VARIANT_LAYOUT_GATHER_SCALE_BF16)},
    };
    if (!g_sel) {
        printf("\n== registration (plow_register_cuda_rtx6000) ==\n");
        for (auto& r : regs)
            printf("  %-22s op=0x%04x  %s\n", r.n, r.op,
                   dt_lookup(&dt_gpu, r.op) ? "registered" : "*** MISSING ***");
    }

    if (want("gemm_exact_1tile")) test_gemm_bf16("gemm_bf16 128x64x32 exact",   128, 64, 32, true);
    if (want("gemm_exact_multi")) test_gemm_bf16("gemm_bf16 256x128x64 exact",  256, 128, 64, true);
    if (want("gemm_exact_ragged"))test_gemm_bf16("gemm_bf16 120x56x32 ragged",  120, 56, 32, true);
    /* B staging loads 8 consecutive k as one 16B chunk when k and the slice base
     * are 8-aligned, else a scalar fallback. k=28 forces the fallback; k=40 is
     * 8-aligned but not GM_BK(32)-aligned, so the last chunk of the tile runs
     * off the end of K and must be zero-filled, not left stale. */
    if (want("gemm_exact_ragged_k"))  test_gemm_bf16("gemm_bf16 120x56x28 ragged k",  120, 56, 28, true);
    if (want("gemm_exact_ragged_k40"))test_gemm_bf16("gemm_bf16 72x40x40 ragged k40", 72, 40, 40, true);
    if (want("gemm_real"))        test_gemm_bf16("gemm_bf16 256x512x1024",      256, 512, 1024, false);
    if (want("gemm_norm"))        test_gemm_norm_bf16("gemm_norm_bf16 256x512x1024", 256, 512, 1024);
    if (want("gemm_norm_small"))  test_gemm_norm_bf16("gemm_norm_bf16 128x64x256",   128, 64, 256);
    if (want("gemm_splitk"))      test_gemm_splitk("gemm_bf16_splitk 64x256x1024 s4",  64, 256, 1024, 4, false);
    if (want("gemm_norm_splitk")) test_gemm_splitk("gemm_norm_splitk 64x256x1024 s4",  64, 256, 1024, 4, true);

    if (want("negctl_flash")) test_flash_prefill("NEGCTL flash K[999]+1", 4, 128, 128, 999);
    if (want("flash_causal_small")) test_flash_prefill("flash_causal h4 sq128 skv128",  4, 128, 128);
    if (want("flash_causal_big"))   test_flash_prefill("flash_causal h8 sq256 skv256",  8, 256, 256);
    if (want("flash_decode"))       test_flash_decode ("flash_decode h8 skv512",        8, 512);
    /* window 96 deliberately straddles the FA_BKV=64 tile boundary so the
     * kernel's whole-tile skip and its per-element mask are both exercised. */
    if (want("flash_sliding"))      test_flash_sliding("flash_sliding h4 sq256 w96",    4, 256, 256, 96);
    if (want("flash_sliding_small"))test_flash_sliding("flash_sliding h4 sq256 w32",    4, 256, 256, 32);

    if (want("row_rmsnorm"))   test_row_rmsnorm ("row_rmsnorm 64x1024",  64, 1024);
    if (want("row_residual"))  test_row_residual("row_residual 64x1024", 64, 1024);
    if (want("row_swiglu"))    test_row_swiglu  ("row_swiglu 64x1024",   64, 1024);
    if (want("row_normrope"))      test_row_normrope("row_normrope hd128 8x512",     8, 512, 128, false, 0);
    if (want("row_normropescale")) test_row_normrope("row_normropescale hd128 8x512",8, 512, 128, true,  0);
    if (want("row_normrope_pos"))  test_row_normrope("row_normrope hd128 pos_base=7",8, 512, 128, false, 7);
    /* head_dim sweep: the RoPE step reads nv[i+half] from a PER-THREAD local
     * array. Lane L only ever writes nv[L], nv[L+32], ... so the pair (i,i+half)
     * is lane-local ONLY when half==32, i.e. head_dim==128. Predict: 128 passes,
     * everything else reads another lane's (uninitialized) local memory. */
    if (want("row_normrope_hd64")) test_row_normrope("row_normrope hd64 8x512",      8, 512,  64, false, 0);
    if (want("row_normrope_hd32")) test_row_normrope("row_normrope hd32 8x512",      8, 512,  32, false, 0);
    if (want("layout_gather"))     test_layout_gather_scale("layout_gather_scale 64x1024", 64, 1024, 4096);

    /* NEGATIVE CONTROL: perturb exactly ONE element of B by +1.0. The harness
     * must report FAIL here; if it does not, the harness is worthless. */
    if (want("negctl_gemm"))
        test_gemm_bf16("NEGCTL gemm 256x128x64 B[777]+1", 256, 128, 64, true, 777);
    if (want("negctl_rmsnorm"))
        test_row_rmsnorm("NEGCTL row_rmsnorm gamma[500]+1", 64, 1024, 500);
    /* the ragged-k shape must be able to fail too, or its maxrel=0 proves nothing */
    if (want("negctl_gemm_ragged_k"))
        test_gemm_bf16("NEGCTL gemm 120x56x28 B[333]+1", 120, 56, 28, true, 333);

    /* A case name that matches nothing must NOT look like a pass: a typo'd or
     * renamed CTest entry would otherwise report success while running zero
     * checks. */
    if (g_sel && !g_matched) {
        printf("%s: UNKNOWN CASE (no test ran)\n", g_sel);
        return 3;
    }
    printf("%s: %s\n", g_sel ? g_sel : "sm120_numeric_test", g_fail ? "FAIL" : "ok");
    return g_fail;
}
