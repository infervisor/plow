/* speech_f32_op_test.cu — the interpreter's FP32 speech arm (op_speech_f32.cuh d_speech_f32,
 * the exact function interp_sm120.cu dispatches ops 163-178/180/195-203 to) against the CPU golden
 * runtime/cpu/dev/golden/f32_primitives.c, one PlowDevInst per case.
 *
 * Pass criteria: every element within 1e-5 of rms(ref) (fp32 paths). Where the op bf16-rounds
 * its output: bf16 on both sides and at most 1 bf16 ulp apart, unless the difference is itself
 * within that fp32 bound (a near-zero value, whose ulp is below fp32 accumulation noise).
 * `exact` counts bit-identical elements.
 *
 * Build (standalone):
 *   gcc -O2 -ffp-contract=off -c -I runtime/common -I runtime/cpu/dev \
 *       runtime/cpu/dev/golden/f32_primitives.c
 *   nvcc -arch=sm_90a -O3 -I runtime/common -I runtime/nvidia -I runtime/cpu/dev \
 *        runtime/tests/speech_f32_op_test.cu f32_primitives.o -o speech_f32_op_test
 *   speech_f32_op_test [--bench | --bench-only]
 */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "op_speech_f32.cuh"
#include "cpu_dev.h"

#define GOLDEN(name) void name(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*)
extern "C" {
GOLDEN(g_q8_gemm_f32);
GOLDEN(g_layernorm_f32);
GOLDEN(g_scaled_add_f32);
GOLDEN(g_glu_f32);
GOLDEN(g_causal_depthwise_conv1d_f32);
GOLDEN(g_relative_attention_f32);
GOLDEN(g_silu_f32);
GOLDEN(g_dense_gemm_f32);
GOLDEN(g_embed_f16_f32);
GOLDEN(g_lstm_cell_f32);
GOLDEN(g_argmax_f32);
GOLDEN(g_relu_f32);
GOLDEN(g_broadcast_add_f32);
GOLDEN(g_conv2d_f32);
GOLDEN(g_pack_ncfw_rows_f32);
GOLDEN(g_grouped_attention_f32);
GOLDEN(g_gemm_f32);
GOLDEN(g_gather_rows_f32);
GOLDEN(g_copy_cols_f32);
GOLDEN(g_conv1d_f32);
GOLDEN(g_conv_transpose1d_f32);
GOLDEN(g_unary_f32);
GOLDEN(g_binary_f32);
GOLDEN(g_cumsum_f64);
GOLDEN(g_rand_f32);
GOLDEN(g_attention_f32);
}
typedef void (*golden_fn)(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERROR %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(2); } } while (0)

__global__ void __launch_bounds__(256, 1) k_speech(PlowDevInst in, void* const* T) {
    extern __shared__ float arena[];
    d_speech_f32(&in, T, blockIdx.x, gridDim.x, arena);
}

static uint32_t rng_s = 0x13572468u;
static float rnd() {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5;
    return (float)(int32_t)rng_s / 2147483648.0f;
}
static uint16_t f2bf(float f) { return plow_f2bf(f); }
static uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }

/* One case: host buffers mirror device buffers; `outs` lists output tensor indices. */
struct Case {
    std::vector<std::vector<uint8_t>> host;
    std::vector<void*> dev;
    PlowDevInst in;
    Case() { memset(&in, 0, sizeof(in)); for (auto& t : in.t) t = PLOW_TENSOR_NONE; }
    ~Case() { for (void* p : dev) cudaFree(p); }
    unsigned add(size_t bytes) {
        host.emplace_back(bytes ? bytes : 4);
        return (unsigned)host.size() - 1;
    }
    unsigned f32(size_t n, float amp, float bias = 0.f) {
        unsigned id = add(n * 4);
        float* p = (float*)host[id].data();
        for (size_t i = 0; i < n; i++) p[i] = bias + amp * rnd();
        return id;
    }
    unsigned bf16(size_t n, float amp) {
        unsigned id = add(n * 2);
        uint16_t* p = (uint16_t*)host[id].data();
        for (size_t i = 0; i < n; i++) p[i] = f2bf(amp * rnd());
        return id;
    }
    unsigned f16(size_t n, float amp) {
        unsigned id = add(n * 2);
        uint16_t* p = (uint16_t*)host[id].data();
        for (size_t i = 0; i < n; i++) p[i] = f2h(amp * rnd());
        return id;
    }
    unsigned u32(std::vector<uint32_t> v) {
        unsigned id = add(v.size() * 4);
        memcpy(host[id].data(), v.data(), v.size() * 4);
        return id;
    }
    unsigned out(size_t n) {
        unsigned id = add(n * 4);
        uint32_t* p = (uint32_t*)host[id].data();
        for (size_t i = 0; i < n; i++) p[i] = 0x7fc00bad;
        return id;
    }
    void upload() {
        for (auto& h : host) {
            void* d;
            CK(cudaMalloc(&d, h.size()));
            CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
            dev.push_back(d);
        }
    }
};

static void** upload_table(const std::vector<void*>& ptrs) {
    void** d;
    CK(cudaMalloc(&d, ptrs.size() * sizeof(void*)));
    CK(cudaMemcpy(d, ptrs.data(), ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice));
    return d;
}

static int g_nblk = 132, g_fail = 0;
static const size_t kSmem = SP_ARENA_FLOATS * sizeof(float);

enum Mode { FP32, BF16, INT, F64 };

static void run(const char* name, Case& c, golden_fn golden, std::vector<unsigned> outs, Mode mode,
                double tol = 1e-5) {
    c.upload();
    std::vector<void*> th;
    for (auto& h : c.host) th.push_back(h.data());
    golden(&c.in, 0, 1, th.data(), nullptr);
    void** td = upload_table(c.dev);
    k_speech<<<g_nblk, 256, kSmem>>>(c.in, td);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaFree(td));
    for (unsigned o : outs) {
        if (mode == F64) {
            const size_t n = c.host[o].size() / 8;
            std::vector<double> got(n);
            CK(cudaMemcpy(got.data(), c.dev[o], n * 8, cudaMemcpyDeviceToHost));
            const double* ref = (const double*)c.host[o].data();
            double rms = 0, maxd = 0;
            for (size_t i = 0; i < n; i++) rms += ref[i] * ref[i];
            rms = sqrt(rms / (double)n);
            for (size_t i = 0; i < n; i++) maxd = std::max(maxd, fabs(got[i] - ref[i]) / (rms ? rms : 1.0));
            const bool pass = maxd <= tol;
            printf("  %-34s t%-2u n=%-8zu f64 max|d|/rms=%-9.3g -> %s\n", name, o, n, maxd, pass ? "PASS" : "FAIL");
            if (!pass) g_fail = 1;
            continue;
        }
        const size_t n = c.host[o].size() / 4;
        std::vector<uint32_t> got(n);
        CK(cudaMemcpy(got.data(), c.dev[o], n * 4, cudaMemcpyDeviceToHost));
        const uint32_t* ref = (const uint32_t*)c.host[o].data();
        size_t exact = 0, bad = 0;
        double rms = 0, maxd = 0;
        unsigned maxulp = 0;
        for (size_t i = 0; i < n; i++) {
            float r, g;
            memcpy(&r, &ref[i], 4); memcpy(&g, &got[i], 4);
            if (ref[i] != 0x7fc00bad && mode != INT) rms += (double)r * r;
        }
        rms = sqrt(rms / (double)n);
        if (rms == 0) rms = 1;
        for (size_t i = 0; i < n; i++) {
            if (ref[i] == got[i]) { exact++; continue; }
            float r, g;
            memcpy(&r, &ref[i], 4); memcpy(&g, &got[i], 4);
            if (mode == INT || std::isnan(r) || std::isnan(g)) { bad++; continue; }
            const double d = fabs((double)g - r) / rms;
            if (d > maxd) maxd = d;
            if (mode == BF16) {
                const unsigned u = (unsigned)abs((int)(ref[i] >> 16) - (int)(got[i] >> 16));
                if (u > maxulp) maxulp = u;
                if ((ref[i] & 0xffff) || (got[i] & 0xffff) || ((u > 1 || ((ref[i] ^ got[i]) >> 31)) && d > tol)) {
                    if (bad < 4) printf("    [%zu] ref=%.9g got=%.9g\n", i, r, g);
                    bad++;
                }
            } else if (d > tol) {
                bad++;
            }
        }
        const bool pass = bad == 0;
        printf("  %-34s t%-2u n=%-8zu exact=%6.2f%%  max|d|/rms=%-9.3g%s -> %s\n", name, o, n,
               100.0 * exact / n, maxd,
               mode == BF16 ? (maxulp ? "  (<=1 bf16 ulp)" : "") : "", pass ? "PASS" : "FAIL");
        if (!pass) {
            printf("    %zu bad elements\n", bad);
            g_fail = 1;
        }
    }
}

static void t_dense(const char* name, unsigned m, unsigned n, unsigned k, unsigned flags,
                    unsigned act, unsigned row0, unsigned stride, unsigned onehot, bool bias) {
    Case c;
    const unsigned ws = stride ? stride : k;
    unsigned o = c.out((size_t)m * n);
    unsigned x = c.f32((size_t)(m + row0) * k, 1.f);
    unsigned w = flags & 4u ? c.bf16((size_t)n * ws, 0.06f) : c.f32((size_t)n * ws, 0.06f);
    unsigned b = bias ? c.f32(n, 0.5f) : PLOW_TENSOR_NONE;
    c.in.op = PLOW_DOP_DENSE_GEMM_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b;
    c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k; c.in.i[3] = act; c.in.i[4] = row0;
    c.in.i[5] = stride; c.in.i[6] = onehot; c.in.i[7] = flags;
    run(name, c, g_dense_gemm_f32, {o}, flags & 3u ? BF16 : FP32);
}

static void t_gemm_bf16(unsigned m, unsigned n, unsigned k) {
    Case c;
    unsigned o = c.out((size_t)m * n), x = c.bf16((size_t)m * k, 1.f), w = c.bf16((size_t)n * k, 0.1f);
    c.in.op = PLOW_DOP_GEMM_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w;
    c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k;
    run("GemmF32 bf16 33x70x96", c, g_gemm_f32, {o}, FP32);
}

static void t_q8(unsigned m, unsigned n, unsigned k, unsigned act) {
    Case c;
    unsigned o = c.out((size_t)m * n), x = c.f32((size_t)(m + 1) * k, 1.f);
    unsigned w = c.add((size_t)n * (k / 32) * 34);
    uint8_t* wp = c.host[w].data();
    for (size_t blk = 0; blk < (size_t)n * (k / 32); blk++) {
        uint16_t s = f2h(0.01f + 0.005f * rnd());
        memcpy(wp + blk * 34, &s, 2);
        for (int i = 0; i < 32; i++) wp[blk * 34 + 2 + i] = (uint8_t)(int8_t)(127.f * rnd());
    }
    unsigned b = c.f32(n, 0.3f);
    c.in.op = PLOW_DOP_Q8_GEMM_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b;
    c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k; c.in.i[3] = act; c.in.i[4] = 1;
    run(act ? "Q8GemmF32 silu 20x96x256" : "Q8GemmF32 70x130x512", c, g_q8_gemm_f32, {o}, FP32);
}

static void t_conv(const char* name, unsigned batches, unsigned frames, unsigned width, unsigned ic,
                   unsigned oc, unsigned kernel, unsigned stride, unsigned pb, unsigned pa,
                   unsigned flags) {
    Case c;
    const unsigned of = (frames + pb + pa - kernel) / stride + 1, ow = (width + pb + pa - kernel) / stride + 1;
    const unsigned stored = flags & 1u ? 1u : ic;
    unsigned o = c.out((size_t)batches * of * ow * oc);
    unsigned x = c.f32((size_t)batches * frames * width * ic, 1.f);
    const size_t wn = (size_t)oc * stored * kernel * kernel;
    unsigned w = flags & 64u ? c.f32(wn, 0.1f) : c.f16(wn, 0.1f);
    unsigned b = c.f32(oc, 0.2f);
    c.in.op = PLOW_DOP_CONV2D_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b;
    c.in.i[0] = frames; c.in.i[1] = width; c.in.i[2] = ic; c.in.i[3] = oc;
    c.in.i[4] = kernel; c.in.i[5] = stride; c.in.i[6] = pb; c.in.i[7] = pa;
    c.in.fj[1].u = flags; c.in.fj[2].u = batches;
    run(name, c, g_conv2d_f32, {o}, flags & 128u ? BF16 : FP32);
}

static void t_layernorm(const char* name, unsigned rows, unsigned feat, unsigned flags, bool affine) {
    Case c;
    unsigned o = c.out((size_t)rows * feat), x = c.f32((size_t)rows * feat, 2.f, 0.5f);
    unsigned g = affine ? c.f32(feat, 0.5f, 1.f) : PLOW_TENSOR_NONE;
    unsigned b = affine ? c.f32(feat, 0.2f) : PLOW_TENSOR_NONE;
    c.in.op = PLOW_DOP_LAYERNORM_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = g; c.in.t[3] = b;
    c.in.i[0] = rows; c.in.i[1] = feat; c.in.i[2] = flags; c.in.fj[0].f = 1e-5f;
    run(name, c, g_layernorm_f32, {o}, flags & 1u ? BF16 : FP32);
}

static void t_grouped(const char* name, unsigned rows, unsigned width, unsigned hw, unsigned group,
                      unsigned flags, int valid) {
    Case c;
    const size_t n = (size_t)rows * width;
    unsigned o = c.out(n), q = c.f32(n, 1.f), k = c.f32(n, 1.f), v = c.f32(n, 1.f);
    unsigned vr = valid >= 0 ? c.u32({(uint32_t)valid}) : PLOW_TENSOR_NONE;
    c.in.op = PLOW_DOP_GROUPED_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = vr;
    c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = hw; c.in.i[3] = group; c.in.i[4] = flags;
    run(name, c, g_grouped_attention_f32, {o}, flags & 4u ? BF16 : FP32);
}

static void t_relative(const char* name, unsigned rows, unsigned width, unsigned heads, unsigned chunk,
                       unsigned left) {
    Case c;
    const size_t n = (size_t)rows * width;
    unsigned o = c.out(n), q = c.f32(n, 1.f), k = c.f32(n, 1.f), v = c.f32(n, 1.f);
    unsigned p = c.f32((size_t)(2 * rows - 1) * width, 1.f), u = c.f32(width, 0.3f), bv = c.f32(width, 0.3f);
    c.in.op = PLOW_DOP_RELATIVE_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = p; c.in.t[5] = u; c.in.t[6] = bv;
    c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = heads; c.in.i[3] = chunk; c.in.i[4] = left;
    run(name, c, g_relative_attention_f32, {o}, FP32);
}

static void t_elementwise() {
    {
        Case c; const unsigned n = 100003;
        unsigned o = c.out(n), a = c.f32(n, 3.f), b = c.f32(n, 3.f);
        c.in.op = PLOW_DOP_SCALED_ADD_F32; c.in.t[0] = o; c.in.t[1] = a; c.in.t[2] = b;
        c.in.i[0] = n; c.in.i[1] = 1; c.in.fj[0].f = 0.75f;
        run("ScaledAddF32 bf16", c, g_scaled_add_f32, {o}, BF16);
    }
    {
        Case c; const unsigned n = 4099;
        unsigned o = c.out(n), a = c.f32(n, 3.f), b = c.f32(n, 3.f);
        c.in.op = PLOW_DOP_SCALED_ADD_F32; c.in.t[0] = o; c.in.t[1] = a; c.in.t[2] = b;
        c.in.i[0] = n; c.in.i[1] = 0; c.in.fj[0].f = -1.5f;
        run("ScaledAddF32", c, g_scaled_add_f32, {o}, FP32);
    }
    {
        Case c; const unsigned rows = 37, width = 300;
        unsigned o = c.out(rows * width), x = c.f32(rows * width * 2, 4.f);
        c.in.op = PLOW_DOP_GLU_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.i[0] = rows; c.in.i[1] = width;
        run("GluF32", c, g_glu_f32, {o}, FP32);
    }
    {
        Case c; const unsigned n = 5000;
        unsigned o = c.out(n), x = c.f32(n, 6.f);
        c.in.op = PLOW_DOP_SILU_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.i[0] = n;
        run("SiluF32", c, g_silu_f32, {o}, FP32);
    }
    {
        Case c; const unsigned n = 5000;
        unsigned o = c.out(n), x = c.f32(n, 6.f);
        c.in.op = PLOW_DOP_RELU_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.i[0] = n;
        run("ReluF32", c, g_relu_f32, {o}, FP32);
    }
    {
        Case c; const unsigned rows = 41, width = 257;
        unsigned o = c.out(rows * width), m = c.f32(rows * width, 2.f), v = c.f32(width, 2.f);
        c.in.op = PLOW_DOP_BROADCAST_ADD_F32; c.in.t[0] = o; c.in.t[1] = m; c.in.t[2] = v;
        c.in.i[0] = rows; c.in.i[1] = width;
        run("BroadcastAddF32", c, g_broadcast_add_f32, {o}, FP32);
    }
    {
        Case c; const unsigned rows = 50, ch = 96, kernel = 9;
        unsigned o = c.out(rows * ch), x = c.f32(rows * ch, 1.f), w = c.f16(ch * kernel, 0.3f);
        c.in.op = PLOW_DOP_CAUSAL_DEPTHWISE_CONV1D_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w;
        c.in.i[0] = rows; c.in.i[1] = ch; c.in.i[2] = kernel;
        run("CausalDepthwiseConv1dF32", c, g_causal_depthwise_conv1d_f32, {o}, FP32);
    }
    {
        Case c; const unsigned vocab = 50, width = 300;
        unsigned o = c.out(width), t = c.f16(vocab * width, 2.f), tok = c.u32({7});
        c.in.op = PLOW_DOP_EMBED_F16_F32; c.in.t[0] = o; c.in.t[1] = t; c.in.t[2] = tok;
        c.in.i[0] = vocab; c.in.i[1] = width;
        run("EmbedF16F32", c, g_embed_f16_f32, {o}, FP32);
    }
    {
        Case c; const unsigned width = 640;
        unsigned h = c.out(width), cn = c.out(width), g = c.f32(4 * width, 4.f), cp = c.f32(width, 2.f);
        c.in.op = PLOW_DOP_LSTM_CELL_F32; c.in.t[0] = h; c.in.t[1] = cn; c.in.t[2] = g; c.in.t[3] = cp;
        c.in.i[0] = width;
        run("LstmCellF32", c, g_lstm_cell_f32, {h, cn}, FP32);
    }
    {
        Case c; const unsigned rows = 6, width = 3001;
        unsigned o = c.out(rows), x = c.f32(rows * width, 1.f);
        float* xp = (float*)c.host[x].data();
        xp[1 * width + 100] = 5.f; xp[1 * width + 2000] = 5.f;       /* tie -> first */
        xp[2 * width + 0] = NAN;                                     /* NaN first -> 0 */
        xp[3 * width + 5] = NAN; xp[3 * width + 2999] = 7.f;         /* NaN never wins */
        for (unsigned i = 0; i < width; i++) xp[4 * width + i] = -INFINITY;
        c.in.op = PLOW_DOP_ARGMAX_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.i[0] = rows; c.in.i[1] = width;
        run("ArgmaxF32", c, g_argmax_f32, {o}, INT);
    }
    {
        Case c; const unsigned batches = 3, ch = 8, frames = 5, width = 13, rows = 37;
        unsigned o = c.out(rows * ch * frames), x = c.f32(batches * ch * frames * width, 1.f);
        c.in.op = PLOW_DOP_PACK_NCFW_ROWS_F32; c.in.t[0] = o; c.in.t[1] = x;
        c.in.i[0] = rows; c.in.i[1] = ch; c.in.i[2] = frames; c.in.i[3] = width; c.in.i[4] = batches;
        run("PackNcfwRowsF32", c, g_pack_ncfw_rows_f32, {o}, FP32);
    }
}

static void bench_conv(unsigned batches, unsigned frames, unsigned width, unsigned ic, unsigned oc) {
    const unsigned of = (frames - 1) / 2 + 1, ow = (width - 1) / 2 + 1;
    float *o, *x, *w, *b;
    CK(cudaMalloc(&o, (size_t)batches * of * ow * oc * 4));
    CK(cudaMalloc(&x, (size_t)batches * frames * width * ic * 4));
    CK(cudaMalloc(&w, (size_t)oc * ic * 9 * 4)); CK(cudaMalloc(&b, (size_t)oc * 4));
    CK(cudaMemset(x, 0, (size_t)batches * frames * width * ic * 4));
    CK(cudaMemset(w, 0, (size_t)oc * ic * 9 * 4)); CK(cudaMemset(b, 0, oc * 4));
    std::vector<void*> ptrs = {o, x, w, b};
    void** td = upload_table(ptrs);
    PlowDevInst in;
    memset(&in, 0, sizeof(in));
    for (auto& t : in.t) t = PLOW_TENSOR_NONE;
    in.op = PLOW_DOP_CONV2D_F32;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3;
    in.i[0] = frames; in.i[1] = width; in.i[2] = ic; in.i[3] = oc; in.i[4] = 3; in.i[5] = 2;
    in.i[6] = 1; in.i[7] = 1; in.fj[1].u = 64 | 128 | (2 << 2) | (2 << 4); in.fj[2].u = batches;
    k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int reps = 5;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= reps;
    printf("  bench Conv2dF32 3x3/2 %ux%ux%u ic=%u oc=%u: %.3f ms  %.1f GFLOP/s\n", batches, frames, width,
           ic, oc, ms, 2.0 * batches * of * ow * oc * ic * 9 / (ms * 1e6));
    cudaFree(o); cudaFree(x); cudaFree(w); cudaFree(b); cudaFree(td);
}

static void bench_dense(unsigned m, unsigned n, unsigned k) {
    float *o, *x, *b;
    uint16_t* w;
    CK(cudaMalloc(&o, (size_t)m * n * 4)); CK(cudaMalloc(&x, (size_t)m * k * 4));
    CK(cudaMalloc(&w, (size_t)n * k * 2)); CK(cudaMalloc(&b, (size_t)n * 4));
    CK(cudaMemset(x, 0, (size_t)m * k * 4)); CK(cudaMemset(w, 0, (size_t)n * k * 2)); CK(cudaMemset(b, 0, n * 4));
    std::vector<void*> ptrs = {o, x, w, b};
    void** td = upload_table(ptrs);
    PlowDevInst in;
    memset(&in, 0, sizeof(in));
    for (auto& t : in.t) t = PLOW_TENSOR_NONE;
    in.op = PLOW_DOP_DENSE_GEMM_F32;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3;
    in.i[0] = m; in.i[1] = n; in.i[2] = k; in.i[7] = 1u | 4u;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < 3; i++) k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    const int reps = 20;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= reps;
    printf("  bench DenseGemmF32 bf16w M=%u N=%u K=%u nblk=%d: %.3f ms  %.1f GFLOP/s\n", m, n, k, g_nblk,
           ms, 2.0 * m * n * k / (ms * 1e6));
    cudaFree(o); cudaFree(x); cudaFree(w); cudaFree(b); cudaFree(td);
}

/* ---- generic signal ops (195-203) ---------------------------------------------------------- */

static unsigned pads(unsigned before, unsigned after) { return before | after << 16; }
static unsigned conv_flags(unsigned mode, unsigned pre, unsigned post, bool f16) {
    return mode | pre << 4 | post << 8 | (f16 ? 1u << 12 : 0u);
}

static void t_gather() {
    {
        Case c; const unsigned items = 2, per_item = 30, repeat = 3, vocab = 12, width = 70;
        unsigned o = c.out(items * per_item * width), t = c.f32(items * vocab * width, 1.f);
        std::vector<uint32_t> idx(2 * 10);
        for (unsigned i = 0; i < idx.size(); i++) idx[i] = i % 7 == 3 ? 0xFFFFFFFFu : (i * 5) % vocab;
        unsigned ix = c.u32(idx);
        c.in.op = PLOW_DOP_GATHER_ROWS_F32; c.in.t[0] = o; c.in.t[1] = t; c.in.t[2] = ix;
        c.in.i[0] = items * per_item; c.in.i[1] = width; c.in.i[2] = vocab; c.in.i[3] = per_item;
        c.in.i[4] = repeat; c.in.i[5] = 10; c.in.i[6] = vocab;
        run("GatherRowsF32 index repeat items", c, g_gather_rows_f32, {o}, FP32);
    }
    {
        Case c; const unsigned rows = 50, vocab = 40, width = 33, stride = 100;
        unsigned o = c.f32(rows * stride, 1.f), t = c.f16(vocab * width, 2.f);
        c.in.op = PLOW_DOP_GATHER_ROWS_F32; c.in.t[0] = o; c.in.t[1] = t;
        c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = vocab; c.in.i[4] = 2; c.in.i[7] = 3;
        c.in.fj[1].u = stride; c.in.fj[2].u = 17;
        run("GatherRowsF32 f16 upsample accumulate", c, g_gather_rows_f32, {o}, FP32);
    }
}

static void t_copy_cols() {
    Case c; const unsigned items = 3, rows = 40, cols = 80, in_stride = 80, out_stride = 320;
    unsigned o = c.f32(items * rows * out_stride, 1.f), x = c.f32(rows * in_stride, 1.f);
    c.in.op = PLOW_DOP_COPY_COLS_F32; c.in.t[0] = o; c.in.t[1] = x;
    c.in.i[0] = items; c.in.i[1] = rows; c.in.i[2] = cols; c.in.i[3] = in_stride; c.in.i[4] = 0;
    c.in.i[5] = out_stride; c.in.i[6] = 160; c.in.fj[1].u = 0; c.in.fj[2].u = rows * out_stride;
    run("CopyColsF32 broadcast into concat", c, g_copy_cols_f32, {o}, FP32);
}

struct ConvSpec {
    unsigned batch, in_rows, cin, cout, kernel, stride, dil_or_opad, groups, before, after;
    unsigned mode, pre, post;
    bool f16, bias, alpha, residual;
    std::vector<uint32_t> lengths;
    float slope;
};

static unsigned conv_out_rows(const ConvSpec& s, bool transpose) {
    if (transpose) return (s.in_rows - 1) * s.stride + s.kernel + s.dil_or_opad - s.before - s.after;
    return (s.in_rows + s.before + s.after - s.dil_or_opad * (s.kernel - 1) - 1) / s.stride + 1;
}

static void build_conv(Case& c, const ConvSpec& s, bool transpose, unsigned& o) {
    const unsigned out_rows = conv_out_rows(s, transpose);
    o = c.out((size_t)s.batch * out_rows * s.cout);
    unsigned x = c.f32((size_t)s.batch * s.in_rows * s.cin, 1.f);
    const size_t wn = (size_t)s.cout * (s.cin / s.groups) * s.kernel;
    const float amp = 1.5f / sqrtf((float)(s.cin / s.groups * s.kernel));
    unsigned w = s.f16 ? c.f16(wn, amp) : c.f32(wn, amp);
    unsigned b = s.bias ? c.f32(s.cout, 0.2f) : PLOW_TENSOR_NONE;
    unsigned a = s.alpha ? c.f32(std::max(s.cin, s.cout), 0.4f, 0.6f) : PLOW_TENSOR_NONE;
    unsigned r = s.residual ? c.f32((size_t)s.batch * out_rows * s.cout, 1.f) : PLOW_TENSOR_NONE;
    unsigned l = s.lengths.empty() ? PLOW_TENSOR_NONE : c.u32(s.lengths);
    c.in.op = transpose ? PLOW_DOP_CONV_TRANSPOSE1D_F32 : PLOW_DOP_CONV1D_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b; c.in.t[4] = a; c.in.t[5] = r; c.in.t[6] = l;
    c.in.i[0] = s.batch; c.in.i[1] = s.in_rows; c.in.i[2] = s.cin; c.in.i[3] = s.cout;
    c.in.i[4] = s.kernel; c.in.i[5] = s.stride; c.in.i[6] = s.dil_or_opad; c.in.i[7] = s.groups;
    c.in.fj[0].f = s.slope; c.in.fj[1].u = pads(s.before, s.after);
    c.in.fj[2].u = conv_flags(s.mode, s.pre, s.post, s.f16);
}

static void t_conv1d(const char* name, const ConvSpec& s, double tol = 2e-5) {
    Case c; unsigned o;
    build_conv(c, s, false, o);
    run(name, c, g_conv1d_f32, {o}, FP32, tol);
}

static void t_convt1d(const char* name, const ConvSpec& s, double tol = 2e-5) {
    Case c; unsigned o;
    build_conv(c, s, true, o);
    run(name, c, g_conv_transpose1d_f32, {o}, FP32, tol);
}

static void t_unary() {
    static const char* names[16] = {"none", "tanh", "sin", "cos", "exp", "abs", "sigmoid", "silu", "elu",
                                    "leaky", "mish", "gelu_erf", "snake", "clamp", "scale_shift", "relu"};
    for (unsigned kind = 0; kind < 16; kind++) {
        Case c; const unsigned rows = 97, width = 40, stride = 50, col0 = 7;
        unsigned o = c.f32(rows * stride, 1.f), x = c.f32(rows * stride, 4.f);
        unsigned p = kind == 12 || kind == 14 ? c.f32(width, 0.5f, 0.7f) : PLOW_TENSOR_NONE;
        c.in.op = PLOW_DOP_UNARY_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = p;
        c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = kind; c.in.i[3] = stride; c.in.i[4] = col0;
        c.in.fj[0].f = kind == 13 ? -0.5f : 0.1f; c.in.fj[1].f = 0.8f;
        char name[64];
        snprintf(name, sizeof(name), "UnaryF32 %s", names[kind]);
        run(name, c, g_unary_f32, {o}, FP32, kind == 12 ? 1e-5 : 2e-6);
    }
}

static void t_binary() {
    struct B { const char* name; unsigned op, bis, brs, bcs, bn, flags; };
    const unsigned items = 3, rows = 45, width = 37;
    const B cases[] = {
        {"BinaryF32 add full", 0, rows * width, width, 1, items * rows * width, 0},
        {"BinaryF32 sub row-vector scaled", 1, 0, 0, 1, width, 1},
        {"BinaryF32 mul column-vector", 2, rows, 1, 0, items * rows, 0},
        {"BinaryF32 div per-item scalar", 3, 1, 0, 0, items, 0},
        {"BinaryF32 max scalar", 4, 0, 0, 0, 1, 0},
        {"BinaryF32 min shared column", 5, 0, 1, 0, rows, 1},
    };
    for (const B& t : cases) {
        Case c;
        unsigned o = c.out(items * rows * width), a = c.f32(items * rows * width, 2.f), b = c.f32(t.bn, 1.f, 1.5f);
        c.in.op = PLOW_DOP_BINARY_F32; c.in.t[0] = o; c.in.t[1] = a; c.in.t[2] = b;
        c.in.i[0] = items; c.in.i[1] = rows; c.in.i[2] = width; c.in.i[3] = t.op;
        c.in.i[4] = t.bis; c.in.i[5] = t.brs; c.in.i[6] = t.bcs; c.in.i[7] = t.flags; c.in.fj[0].f = -0.7f;
        run(t.name, c, g_binary_f32, {o}, FP32, 0.0);
    }
}

static void t_cumsum() {
    {
        /* SineGen-style phase: upsampled f0 (one column) -> 9 harmonics, wrapped, x 2pi. */
        Case c; const unsigned items = 2, rows = 48000, width = 9;
        unsigned o = c.out(items * rows * width), x = c.f32(items * rows, 100.f, 200.f);
        std::vector<float> h(width);
        for (unsigned i = 0; i < width; i++) h[i] = (float)(i + 1);
        unsigned cs = c.add(width * 4);
        memcpy(c.host[cs].data(), h.data(), width * 4);
        unsigned l = c.u32({48000, 30011});
        c.in.op = PLOW_DOP_CUMSUM_F64; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = cs; c.in.t[3] = l;
        c.in.i[0] = items; c.in.i[1] = rows; c.in.i[2] = width; c.in.i[3] = 1; c.in.i[4] = 2;
        c.in.fj[0].f = 1.0f / 24000.0f; c.in.fj[1].f = 6.2831855f;
        run("CumSumF64 wrap harmonics 2x48000x9", c, g_cumsum_f64, {o}, FP32, 1e-5);
    }
    {
        Case c; const unsigned items = 3, rows = 700, width = 5;
        unsigned o = c.add((size_t)items * rows * width * 8), x = c.f32(items * rows * width, 3.f);
        c.in.op = PLOW_DOP_CUMSUM_F64; c.in.t[0] = o; c.in.t[1] = x;
        c.in.i[0] = items; c.in.i[1] = rows; c.in.i[2] = width; c.in.i[4] = 1 | 4;
        c.in.fj[0].f = 1.0f; c.in.fj[1].f = 1.0f;
        run("CumSumF64 exclusive f64-out", c, g_cumsum_f64, {o}, F64, 1e-12);
    }
}

/* Verbatim copies of the counter generators the op replaces (runtime/nvidia/s3gen/s3gen.cu
 * rhash/rnormal/runiform, runtime/nvidia/snac/snac.cu noise_at). */
namespace ref {
__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
__device__ __forceinline__ unsigned long long rhash(unsigned long long seed, unsigned stream, unsigned a,
                                                    unsigned b) {
  unsigned long long key = ((unsigned long long)stream << 56) ^ ((unsigned long long)a << 32) ^ b;
  return mix64(seed ^ mix64(key));
}
__device__ __forceinline__ float rnormal(unsigned long long seed, unsigned stream, unsigned a, unsigned b) {
  unsigned long long h1 = rhash(seed, stream, a, b), h2 = mix64(h1);
  float u1 = (float)((h1 >> 40) + 1) * (1.0f / 16777216.0f);  // (0,1]
  float u2 = (float)(h2 >> 40) * (1.0f / 16777216.0f);        // [0,1)
  return sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
}
__device__ __forceinline__ float runiform(unsigned long long seed, unsigned stream, unsigned a, unsigned b) {
  return (float)(rhash(seed, stream, a, b) >> 40) * (1.0f / 16777216.0f);  // [0,1)
}
__device__ __forceinline__ float noise_at(unsigned long long seed, int blk, int b, int t) {
  unsigned long long key = ((unsigned long long)blk << 58) | ((unsigned long long)b << 32) |
                           (unsigned)t;
  unsigned long long h1 = mix64(seed ^ mix64(key));
  unsigned long long h2 = mix64(h1);
  float u1 = (float)((h1 >> 40) + 1) * (1.0f / 16777216.0f);  // (0,1]
  float u2 = (float)(h2 >> 40) * (1.0f / 16777216.0f);        // [0,1)
  return sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
}
}  // namespace ref

/* which: 1 s3gen CFM noise z[b][t][c], 2 s3gen harmonic phase [b][i-1], 3 s3gen source noise
 * [b][n][i], 4 snac NoiseBlock n[b][t]. */
__global__ void k_ref_rand(float* out, const unsigned long long* seed, int which, int items, int rows,
                           int width, int blk) {
    const int n = items * rows * width;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) {
        const int c = e % width, r = e / width % rows, b = e / (width * rows);
        float v;
        if (which == 1) v = ref::rnormal(seed[b], 1, r, c);
        else if (which == 2) v = (ref::runiform(seed[b], 2, r + 1, 0) * 2.0f - 1.0f) * 3.14159265358979f;
        else if (which == 3) v = ref::rnormal(seed[b], 3, c, r);
        else v = ref::noise_at(seed[0], blk, b, r);
        out[e] = v;
    }
}

static void t_rand_case(const char* name, int which, unsigned items, unsigned rows, unsigned width,
                        unsigned stream, unsigned shift, unsigned coords, unsigned a_off, unsigned flags,
                        float scale, float offset) {
    Case c;
    unsigned o = c.out((size_t)items * rows * width);
    std::vector<uint64_t> seeds(items);
    for (unsigned i = 0; i < items; i++) seeds[i] = 0x9E3779B97F4A7C15ull * (i + 3) ^ 0x1234567ull;
    unsigned sd = c.add(items * 8);
    memcpy(c.host[sd].data(), seeds.data(), items * 8);
    c.in.op = PLOW_DOP_RAND_F32; c.in.t[0] = o; c.in.t[1] = sd;
    c.in.i[0] = items; c.in.i[1] = rows; c.in.i[2] = width; c.in.i[3] = stream; c.in.i[4] = shift;
    c.in.i[5] = coords; c.in.i[6] = a_off; c.in.fj[0].f = scale; c.in.fj[1].f = offset; c.in.fj[2].u = flags;
    char gname[96];
    snprintf(gname, sizeof(gname), "%s vs golden", name);
    run(gname, c, g_rand_f32, {o}, flags & 1u ? FP32 : INT, 1e-6);
    const size_t n = (size_t)items * rows * width;
    float* ref;
    CK(cudaMalloc(&ref, n * 4));
    k_ref_rand<<<256, 256>>>(ref, (const unsigned long long*)c.dev[sd], which, items, rows, width, stream);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(n), want(n);
    CK(cudaMemcpy(got.data(), c.dev[o], n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(want.data(), ref, n * 4, cudaMemcpyDeviceToHost));
    cudaFree(ref);
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) bad += got[i] != want[i];
    printf("  %-34s     n=%-8zu bit-exact vs native generator: %s\n", name, n, bad ? "FAIL" : "PASS");
    if (bad) { printf("    %zu differing elements\n", bad); g_fail = 1; }
}

static void t_rand() {
    t_rand_case("RandF32 s3gen CFM noise", 1, 2, 700, 80, 1, 56, 0 | 1 << 2, 0, 1, 1.f, 0.f);
    t_rand_case("RandF32 s3gen harmonic phase", 2, 2, 8, 1, 2, 56, 0 | 1 << 2, 1, 0, 6.2831855f, -0.5f);
    t_rand_case("RandF32 s3gen source noise", 3, 2, 30000, 9, 3, 56, 1 | 0 << 2, 0, 1, 1.f, 0.f);
    t_rand_case("RandF32 snac NoiseBlock", 4, 3, 2048, 1, 2, 58, 2 | 0 << 2, 0, 1 | 2, 1.f, 0.f);
}

static void t_attention(const char* name, unsigned batch, unsigned rows, unsigned heads, unsigned hw,
                        std::vector<uint32_t> lengths, bool causal, bool bias, bool fused) {
    Case c;
    const unsigned width = heads * hw, stride = fused ? 3 * width : width;
    unsigned o = c.out((size_t)batch * rows * width);
    unsigned q, k, v;
    if (fused) {
        q = k = v = c.f32((size_t)batch * rows * stride, 1.f);
    } else {
        q = c.f32((size_t)batch * rows * width, 1.f);
        k = c.f32((size_t)batch * rows * width, 1.f);
        v = c.f32((size_t)batch * rows * width, 1.f);
    }
    unsigned l = lengths.empty() ? PLOW_TENSOR_NONE : c.u32(lengths);
    unsigned bs = bias ? c.f32((size_t)heads * rows * rows, 1.f) : PLOW_TENSOR_NONE;
    c.in.op = PLOW_DOP_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = l; c.in.t[5] = bs;
    c.in.i[0] = batch; c.in.i[1] = rows; c.in.i[2] = rows; c.in.i[3] = heads; c.in.i[4] = hw;
    c.in.i[5] = fused ? stride : 0; c.in.i[6] = causal; c.in.i[7] = rows * rows;
    c.in.fj[0].f = 1.0f / sqrtf((float)hw);
    c.in.fj[1].u = fused ? width : 0; c.in.fj[2].u = fused ? 2 * width : 0;
    run(name, c, g_attention_f32, {o}, FP32, 2e-5);
}

static void bench(const char* name, Case& c, double flops, double bytes, int reps = 20) {
    c.upload();
    void** td = upload_table(c.dev);
    for (int i = 0; i < 2; i++) k_speech<<<g_nblk, 256, kSmem>>>(c.in, td);
    CK(cudaGetLastError());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) k_speech<<<g_nblk, 256, kSmem>>>(c.in, td);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= reps;
    printf("  bench %-44s %8.3f ms", name, ms);
    if (flops > 0) printf("  %8.1f GFLOP/s", flops / (ms * 1e6));
    if (bytes > 0) printf("  %8.1f GB/s", bytes / (ms * 1e6));
    printf("\n");
    CK(cudaFree(td));
}

static void bench_signal() {
    {
        Case c; const unsigned m = 16384, n = 512, k = 512;
        unsigned o = c.out((size_t)m * n), x = c.f32((size_t)m * k, 1.f), w = c.f32((size_t)n * k, 0.05f);
        c.in.op = PLOW_DOP_DENSE_GEMM_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w;
        c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k;
        bench("DenseGemmF32 f32w 16384x512x512 (reference)", c, 2.0 * m * n * k, 0);
    }
    {
        ConvSpec s{2, 8192, 512, 512, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, false, {}, 0.f};
        Case c; unsigned o; build_conv(c, s, false, o);
        bench("Conv1dF32 pointwise 512->512 16k rows", c, 2.0 * 16384 * 512 * 512, 0);
    }
    {
        ConvSpec s{2, 8192, 512, 512, 1, 1, 1, 1, 0, 0, 0, 12, 0, false, true, true, true, {}, 0.f};
        Case c; unsigned o; build_conv(c, s, false, o);
        bench("Conv1dF32 pointwise +snake +residual 16k", c, 2.0 * 16384 * 512 * 512, 0);
    }
    for (unsigned dil : {1u, 3u, 9u}) {
        ConvSpec s{2, 2048, 512, 512, 7, 1, dil, 512, 3 * dil, 3 * dil, 0, 12, 0, false, true, true, false, {}, 0.f};
        Case c; unsigned o; build_conv(c, s, false, o);
        char name[64];
        snprintf(name, sizeof(name), "Conv1dF32 depthwise k7 d%u snake [2][2048][512]", dil);
        bench(name, c, 2.0 * 2 * 2048 * 512 * 7, 2.0 * 2 * 2048 * 512 * 4);
    }
    {
        ConvSpec s{2, 2048, 512, 512, 7, 1, 1, 1, 3, 3, 0, 12, 0, false, true, true, false, {}, 0.f};
        Case c; unsigned o; build_conv(c, s, false, o);
        bench("Conv1dF32 dense k7 512->512 snake [2][2048]", c, 2.0 * 4096 * 512 * 512 * 7, 0);
    }
    {
        ConvSpec s{2, 256, 1024, 512, 16, 8, 0, 1, 4, 4, 0, 12, 0, false, true, true, false, {}, 0.f};
        Case c; unsigned o; build_conv(c, s, true, o);
        const double out_rows = conv_out_rows(s, true);
        bench("ConvTranspose1dF32 1024->512 k16 s8 [2][256]", c, 2.0 * 2 * out_rows * 512 * 1024 * 2, 0);
    }
    {
        Case c;
        const unsigned batch = 2, rows = 1200, heads = 8, hw = 64, width = heads * hw;
        unsigned o = c.out((size_t)batch * rows * width), q = c.f32((size_t)batch * rows * width, 1.f);
        unsigned k = c.f32((size_t)batch * rows * width, 1.f), v = c.f32((size_t)batch * rows * width, 1.f);
        unsigned l = c.u32({1200, 1000});
        c.in.op = PLOW_DOP_ATTENTION_F32;
        c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = l;
        c.in.i[0] = batch; c.in.i[1] = rows; c.in.i[2] = rows; c.in.i[3] = heads; c.in.i[4] = hw;
        c.in.fj[0].f = 0.125f;
        bench("AttentionF32 8x64 2x1200 (len 1200/1000)", c, 4.0 * heads * hw * rows * (1200.0 + 1000.0), 0);
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    g_nblk = prop.multiProcessorCount;
    CK(cudaFuncSetAttribute(k_speech, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem));
    printf("speech_f32_op_test on %s (%d SMs), nblk=%d, arena=%zu B\n", prop.name, g_nblk, g_nblk, kSmem);

    if (argc > 1 && !strcmp(argv[1], "--bench-only")) {
        bench_signal();
        return 0;
    }
    t_dense("DenseGemmF32 qwen q 300x1024x1024", 300, 1024, 1024, 1 | 4, 0, 0, 0, 0, true);
    t_dense("DenseGemmF32 gelu 200x512x384", 200, 512, 384, 2 | 4, 0, 0, 0, 0, true);
    t_dense("DenseGemmF32 f32w relu row0", 77, 130, 70, 0, 1, 3, 0, 0, true);
    t_dense("DenseGemmF32 f32w onehot", 45, 67, 33, 0, 0, 0, 35, 34, false);
    t_dense("DenseGemmF32 bf16w onehot bf16", 45, 67, 36, 1 | 4, 0, 0, 37, 36, true);
    t_gemm_bf16(33, 70, 96);
    t_q8(70, 130, 512, 0);
    t_q8(20, 96, 256, 1);
    /* Qwen3-ASR conv stages: NCFW in/out, f32 weights, GELU bf16; then other layouts/flags. */
    t_conv("Conv2dF32 qwen stage1", 2, 128, 100, 1, 48, 3, 2, 1, 1, 64 | 128 | (2 << 2) | (2 << 4));
    t_conv("Conv2dF32 qwen stage2", 2, 64, 50, 48, 64, 3, 2, 1, 1, 64 | 128 | (2 << 2) | (2 << 4));
    t_conv("Conv2dF32 f16 NFCW->NFWC relu", 2, 17, 11, 12, 20, 3, 1, 1, 0, 2 | (0 << 2) | (1 << 4));
    t_conv("Conv2dF32 depthwise f16 NFWC->NCFW", 2, 17, 11, 24, 24, 3, 2, 1, 1, 1 | 2 | (2 << 2));
    t_conv("Conv2dF32 depthwise f32 gelu", 1, 9, 30, 16, 16, 5, 1, 2, 2, 1 | 64 | 128 | (1 << 2) | (1 << 4));
    t_layernorm("LayerNormF32 ordered bf16", 100, 1024, 3, true);
    t_layernorm("LayerNormF32 ordered ragged", 45, 1000, 2, true);
    t_layernorm("LayerNormF32 double", 70, 1024, 0, true);
    t_layernorm("LayerNormF32 double bf16 noaffine", 33, 513, 1, false);
    t_grouped("GroupedAttentionF32 qwen 300 v290", 300, 1024, 64, 104, 7, 290);
    t_grouped("GroupedAttentionF32 fp32 g256 (global K)", 300, 256, 64, 256, 0, -1);
    t_grouped("GroupedAttentionF32 hw80 g50", 130, 320, 80, 50, 1, 111);
    t_relative("RelativeAttentionF32 chunked", 40, 256, 4, 8, 2);
    t_relative("RelativeAttentionF32 full hw320", 21, 640, 2, 1, 0xFFFFFFFFu);
    t_elementwise();
    t_gather();
    t_copy_cols();
    /* SNAC decoder block shapes: depthwise k7 dilations 1/3/9 with snake, pointwise 512->512. */
    for (unsigned dil : {1u, 3u, 9u}) {
        char name[64];
        snprintf(name, sizeof(name), "Conv1dF32 depthwise k7 d%u snake", dil);
        t_conv1d(name, {2, 2048, 512, 512, 7, 1, dil, 512, 3 * dil, 3 * dil, 0, 12, 0, false, true, true, false,
                        {}, 0.f});
    }
    t_conv1d("Conv1dF32 pointwise 512 16k +res", {2, 8192, 512, 512, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true,
                                                   false, true, {}, 0.f});
    t_conv1d("Conv1dF32 dense k7 reflect leaky->tanh len", {2, 300, 64, 80, 7, 1, 3, 1, 9, 9, 1, 9, 1, true, true,
                                                              false, false, {300, 211}, 0.1f});
    t_conv1d("Conv1dF32 causal k3 mish-out f16 len", {3, 257, 96, 128, 3, 1, 1, 1, 2, 0, 0, 0, 10, true, true,
                                                        false, true, {257, 100, 3}, 0.f});
    t_conv1d("Conv1dF32 stride2 replicate groups2", {2, 199, 64, 64, 5, 2, 1, 2, 2, 2, 2, 8, 0, false, false,
                                                       false, false, {}, 0.f});
    t_conv1d("Conv1dF32 narrow 64->1 k7 tanh (direct)", {2, 1000, 64, 1, 7, 1, 1, 1, 3, 3, 0, 9, 1, false, true,
                                                           false, false, {1000, 500}, 0.01f});
    t_conv1d("Conv1dF32 odd channels 30->50 k5 elu", {1, 90, 30, 50, 5, 1, 2, 1, 4, 4, 0, 8, 7, false, true,
                                                       false, false, {}, 0.f});
    /* SNAC / HiFT upsamplers and the iSTFT overlap-add form. */
    t_convt1d("ConvT1dF32 1024->512 k16 s8 snake", {2, 128, 1024, 512, 16, 8, 0, 1, 4, 4, 0, 12, 0, false, true,
                                                     true, false, {128, 77}, 0.f});
    t_convt1d("ConvT1dF32 k11 s5 leaky f16 opad", {2, 60, 128, 64, 11, 5, 1, 1, 3, 2, 0, 9, 0, true, true,
                                                    false, true, {}, 0.1f});
    t_convt1d("ConvT1dF32 groups2 k7 s3", {1, 50, 64, 96, 7, 3, 0, 2, 2, 2, 0, 0, 0, false, true, false, false,
                                            {}, 0.f});
    t_convt1d("ConvT1dF32 istft 18->1 k16 s4 (direct)", {2, 400, 18, 1, 16, 4, 0, 1, 8, 8, 0, 0, 0, false, false,
                                                           false, false, {400, 250}, 0.f});
    t_unary();
    t_binary();
    t_cumsum();
    t_rand();
    t_attention("AttentionF32 8x64 2x1200 lengths", 2, 1200, 8, 64, {1200, 1000}, false, false, false);
    t_attention("AttentionF32 2x128 causal bias fused", 2, 300, 2, 128, {300, 217}, true, true, true);
    t_attention("AttentionF32 4x64 ragged rows", 1, 77, 4, 64, {}, false, false, false);

    if (argc > 1 && !strcmp(argv[1], "--bench")) {
        bench_signal();
        bench_dense(1500, 1280, 1280);
        bench_dense(1500, 1024, 1024);
        bench_dense(1500, 4096, 1024);
        bench_dense(1500, 1024, 4096);
        bench_dense(1500, 5120, 1280);
        bench_conv(30, 64, 50, 480, 480);
    }
    printf(g_fail ? "FAIL\n" : "ALL PASS\n");
    return g_fail;
}
