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
 *   nvcc -arch=sm_90a -O3 -DPLOW_NV_HOPPER=1 -I runtime/common -I runtime/nvidia -I runtime/cpu/dev \
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
GOLDEN(g_stream_tail_f32);
GOLDEN(g_reduce_rows_f32);
GOLDEN(g_length_affine_u32);
GOLDEN(g_trim_silence_f32);
GOLDEN(g_conv1d_f32);
GOLDEN(g_conv_transpose1d_f32);
GOLDEN(g_unary_f32);
GOLDEN(g_binary_f32);
GOLDEN(g_cumsum_f64);
GOLDEN(g_rand_f32);
GOLDEN(g_attention_f32);
GOLDEN(g_row_stats_f32);
}
typedef void (*golden_fn)(const PlowDevInst*, uint32_t, uint32_t, void* const*, PlowCpuCtx*);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERROR %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(2); } } while (0)

__global__ void __launch_bounds__(256, 1) k_speech(PlowDevInst in, void* const* T) {
    extern __shared__ float arena[];
#ifdef SPEECH_TEST_RING_ONLY
    if (in.op == PLOW_DOP_STREAM_TAIL_F32) d_stream_tail_f32(&in, T, blockIdx.x, gridDim.x);
    else d_attention_f32(&in, T, blockIdx.x, gridDim.x, arena);
#elif defined(SPEECH_TEST_SHORT_ATTENTION_ONLY)
    d_attention_f32(&in, T, blockIdx.x, gridDim.x, arena);
#else
#if defined(PLOW_NV_HOPPER)
    if (sp_wg_conv1d(&in, T, blockIdx.x, gridDim.x)) return;
#endif
    d_speech_f32(&in, T, blockIdx.x, gridDim.x, arena);
#endif
}
/* sp_conv1d_direct alone: the reference the staged few-output conv must match bit for bit. */
__global__ void __launch_bounds__(256, 1) k_conv_direct(PlowDevInst in, void* const* T) {
    SpConvArgs a;
    if (sp_conv_args(&in, T, false, a)) sp_conv1d_direct(a, blockIdx.x, gridDim.x);
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
/* SP_ARENA_FLOATS depends on __CUDA_ARCH__ (sm_89/sm_120: 99 KiB blocks), which the host pass
 * does not see: main() reads the device's value. */
__device__ unsigned k_arena_floats = SP_LAUNCH_ARENA_FLOATS;
static size_t kSmem;

enum Mode { FP32, BF16, INT, F64 };

/* flips: BF16 elements allowed past 1 ulp. A bf16-rounded GELU amplifies a one-ulp flip of its
 * rounded input (an fp32 summation-order effect) to several output ulps; the FFMA and tensor-core
 * GEMMs both show 3e-6..3e-5 of elements so at K = 1024..4320 (the budget is 5e-5). */
/* pre: a GPU instruction run first (preparing an input); ref: the golden's instruction (c.in). */
static void run(const char* name, Case& c, golden_fn golden, std::vector<unsigned> outs, Mode mode,
                double tol = 1e-5, size_t flips = 0, const PlowDevInst* pre = nullptr,
                const PlowDevInst* ref_in = nullptr, size_t npre = 1) {
    c.upload();
    std::vector<void*> th;
    for (auto& h : c.host) th.push_back(h.data());
    golden(ref_in ? ref_in : &c.in, 0, 1, th.data(), nullptr);
    void** td = upload_table(c.dev);
    if (pre)
        for (size_t i = 0; i < npre; i++) k_speech<<<g_nblk, 256, kSmem>>>(pre[i], td);
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
        const bool pass = bad <= flips;
        printf("  %-34s t%-2u n=%-8zu exact=%6.2f%%  max|d|/rms=%-9.3g%s -> %s\n", name, o, n,
               100.0 * exact / n, maxd,
               mode == BF16 ? (bad ? "  (gelu flips)" : maxulp ? "  (<=1 bf16 ulp)" : "") : "", pass ? "PASS" : "FAIL");
        if (bad && pass) printf("    %zu flipped elements (budget %zu)\n", bad, flips);
        if (!pass) {
            printf("    %zu bad elements\n", bad);
            g_fail = 1;
        }
    }
}

/* Round a host f32 tensor to bf16-representable values (what flag "bf16-exact" promises). */
static void round_bf16(Case& c, unsigned id) {
    float* p = (float*)c.host[id].data();
    for (size_t i = 0; i < c.host[id].size() / 4; i++) {
        const uint32_t u = (uint32_t)f2bf(p[i]) << 16;
        memcpy(&p[i], &u, 4);
    }
}
/* Conv2dF32 flag 1024 weights from bf16-valued f32 [oc][ic][taps]: bf16 [oc][ic / 16][tap][16]. */
static unsigned blocked_weights(Case& c, unsigned w, unsigned oc, unsigned ic, unsigned taps) {
    const unsigned id = c.add((size_t)oc * ic * taps * 2);
    const uint32_t* src = (const uint32_t*)c.host[w].data();
    uint16_t* dst = (uint16_t*)c.host[id].data();
    for (unsigned n = 0; n < oc; n++)
        for (unsigned ch = 0; ch < ic; ch++)
            for (unsigned t = 0; t < taps; t++)
                dst[(((size_t)n * (ic / 16) + ch / 16) * taps + t) * 16 + ch % 16] =
                    (uint16_t)(src[((size_t)n * ic + ch) * taps + t] >> 16);
    return id;
}
/* Split-K scratch for the tensor-core GEMMs: tickets + one 64x64 partial per block, or the packed
 * reference split's slots. */
static unsigned splitk_scratch(Case& c) {
    const size_t bytes = (SPT_TICKETS + (size_t)g_nblk * SPN_SPLIT_FLOATS) * 4;
    unsigned id = c.add(bytes);
    memset(c.host[id].data(), 0, bytes);
    return id;
}

static void t_dense(const char* name, unsigned m, unsigned n, unsigned k, unsigned flags,
                    unsigned act, unsigned row0, unsigned stride, unsigned onehot, bool bias,
                    bool scratch = false, const float* post = nullptr) {
    Case c;
    const unsigned ws = stride ? stride : k;
    unsigned o = c.out((size_t)m * n);
    unsigned x = c.f32((size_t)(m + row0) * k, 1.f);
    unsigned w = flags & 4u ? c.bf16((size_t)n * ws, 0.06f) : c.f32((size_t)n * ws, 0.06f);
    if (flags & 8u) {
        round_bf16(c, x);
        if (!(flags & 4u)) round_bf16(c, w);
    }
    unsigned b = bias ? c.f32(n, 0.5f) : PLOW_TENSOR_NONE;
    c.in.op = PLOW_DOP_DENSE_GEMM_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b;
    if (scratch) c.in.t[4] = splitk_scratch(c);
    if (post) {
        c.in.t[5] = c.add(16);
        memcpy(c.host[c.in.t[5]].data(), post, 16);
    }
    c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k; c.in.i[3] = act; c.in.i[4] = row0;
    c.in.i[5] = stride; c.in.i[6] = onehot; c.in.i[7] = flags;
    run(name, c, g_dense_gemm_f32, {o}, flags & 3u ? BF16 : FP32, 1e-5,
        flags & 0x202u ? (size_t)m * n / 20000 : 0);
}

/* Flag 128: rows packed from several requests sum K as each request's own split-K launch would.
 * Each reference runs alone at devgen's block count (split-K on); the packed run (one launch,
 * reference rows per row) must match it bit for bit. */
static unsigned dense_ref_blocks(unsigned rows, unsigned n, unsigned k) {
    const unsigned tiles = (rows + 63) / 64 * ((n + 63) / 64), steps = (k + 63) / 64;
    if (tiles > SPT_TICKETS || tiles * 2 > 132u) return std::min(tiles, 132u);
    const unsigned chunk = (steps + std::min(132u / tiles, steps) - 1) / std::min(132u / tiles, steps);
    return tiles * ((steps + chunk - 1) / chunk);
}
static void t_dense_ref(unsigned n, unsigned k, std::vector<unsigned> items) {
    Case c;
    unsigned m = 0;
    for (unsigned r : items) m += r;
    unsigned o = c.out((size_t)m * n), x = c.f32((size_t)m * k, 1.f), w = c.bf16((size_t)n * k, 0.06f);
    round_bf16(c, x);
    unsigned b = c.f32(n, 0.5f), scratch = splitk_scratch(c);
    std::vector<uint32_t> ref;
    for (unsigned r : items) ref.insert(ref.end(), r, r);
    unsigned table = c.u32(ref);
    c.upload();
    void** td = upload_table(c.dev);
    PlowDevInst in = c.in;
    in.op = PLOW_DOP_DENSE_GEMM_F32;
    in.t[0] = o; in.t[1] = x; in.t[2] = w; in.t[3] = b; in.t[4] = table; in.t[5] = scratch;
    in.i[0] = m; in.i[1] = n; in.i[2] = k; in.i[7] = 1 | 4 | 8 | 128; in.fj[1].u = 132;
    k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> packed((size_t)m * n), alone;
    CK(cudaMemcpy(packed.data(), c.dev[o], packed.size() * 4, cudaMemcpyDeviceToHost));
    size_t diff = 0;
    unsigned row0 = 0;
    for (unsigned r : items) {
        PlowDevInst one = in;
        one.t[4] = scratch; one.i[0] = r; one.i[4] = row0; one.i[7] = 1 | 4 | 8; one.fj[1].u = 0;
        k_speech<<<dense_ref_blocks(r, n, k), 256, kSmem>>>(one, td);
        CK(cudaDeviceSynchronize());
        alone.resize((size_t)r * n);
        CK(cudaMemcpy(alone.data(), c.dev[o], alone.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < alone.size(); i++) diff += alone[i] != packed[(size_t)row0 * n + i];
        row0 += r;
    }
    CK(cudaFree(td));
    printf("  DenseGemmF32 ref split %ux%u, %zu requests packed: %zu differ -> %s\n", n, k, items.size(), diff,
           diff ? "FAIL" : "PASS");
    if (diff) g_fail = 1;
}

/* Reference-split DenseGemmF32 (flag 128) at M rows of `ref`-row requests, whole device. */
static void bench_dense_ref(unsigned m, unsigned n, unsigned k, unsigned post, unsigned ref) {
    Case c;
    unsigned o = c.out((size_t)m * n), x = c.add((size_t)m * k * 4), w = c.add((size_t)n * k * 2);
    unsigned b = c.f32(n, 0.5f), scratch = splitk_scratch(c), table = c.u32(std::vector<uint32_t>(m, ref));
    c.upload();
    void** td = upload_table(c.dev);
    PlowDevInst in = c.in;
    in.op = PLOW_DOP_DENSE_GEMM_F32;
    in.t[0] = o; in.t[1] = x; in.t[2] = w; in.t[3] = b; in.t[4] = table; in.t[5] = scratch;
    in.i[0] = m; in.i[1] = n; in.i[2] = k; in.i[7] = post | 4 | 8 | 128; in.fj[1].u = 132;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < 3; i++) k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    const int reps = 50;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaFree(td));
    printf("  bench DenseGemmF32 ref-split M=%u N=%u K=%u ref=%u nblk=%d: %.1f us  %.1f TFLOP/s\n", m, n, k, ref,
           g_nblk, 1e3 * ms / reps, 2.0 * m * n * k / (ms / reps * 1e9));
}

/* sp_gelu_erf_bf16 on every bf16 input against the double exp / __fdiv_rn spelling of the golden. */
__global__ void k_gelu_all(unsigned* differ) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < 65536u; i += gridDim.x * blockDim.x) {
        const float value = __uint_as_float(i << 16);
        const float z = __fmul_rn(value, 0.7071067811865475f);
        const float t = __fdiv_rn(1.0f, __fadd_rn(1.0f, __fmul_rn(0.3275911f, fabsf(z))));
        float p = __fsub_rn(__fmul_rn(1.061405429f, t), 1.453152027f);
        p = __fadd_rn(__fmul_rn(p, t), 1.421413741f);
        p = __fsub_rn(__fmul_rn(p, t), 0.284496736f);
        p = __fadd_rn(__fmul_rn(p, t), 0.254829592f);
        const float r = __fmul_rn(__fmul_rn(p, t), (float)exp((double)__fmul_rn(-z, z)));
        const float want = sp_bf16(__fmul_rn(__fmul_rn(0.5f, value), z < 0.0f ? r : __fsub_rn(2.0f, r)));
        const float got = sp_gelu_erf_bf16(value);
        if (__float_as_uint(want) != __float_as_uint(got) && !(want != want && got != got)) atomicAdd(differ, 1u);
    }
}
static void t_gelu_all() {
    unsigned* differ;
    CK(cudaMalloc(&differ, 4));
    CK(cudaMemset(differ, 0, 4));
    k_gelu_all<<<64, 256>>>(differ);
    CK(cudaDeviceSynchronize());
    unsigned n;
    CK(cudaMemcpy(&n, differ, 4, cudaMemcpyDeviceToHost));
    CK(cudaFree(differ));
    printf("  erf-GELU over the 65536 bf16 inputs: %u differ -> %s\n", n, n ? "FAIL" : "PASS");
    if (n) g_fail = 1;
}

/* Continuous-accumulation DenseGemmF32 (flag 0x800): requests packed vs each alone (bitwise), an
 * FNV-1a of the packed output (compare across builds), and the whole-device time. */
static void t_dense_cont(unsigned n, unsigned k, unsigned post, std::vector<unsigned> items) {
    Case c;
    unsigned m = 0;
    for (unsigned r : items) m += r;
    unsigned o = c.out((size_t)m * n), x = c.f32((size_t)m * k, 1.f), w = c.bf16((size_t)n * k, 0.06f);
    round_bf16(c, x);
    unsigned b = c.f32(n, 0.5f);
    c.upload();
    void** td = upload_table(c.dev);
    PlowDevInst in = c.in;
    in.op = PLOW_DOP_DENSE_GEMM_F32;
    in.t[0] = o; in.t[1] = x; in.t[2] = w; in.t[3] = b;
    in.i[0] = m; in.i[1] = n; in.i[2] = k; in.i[7] = post | 4 | 8 | 0x800;
    k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> packed((size_t)m * n), alone;
    CK(cudaMemcpy(packed.data(), c.dev[o], packed.size() * 4, cudaMemcpyDeviceToHost));
    uint64_t fnv = 1469598103934665603ull;
    for (uint32_t v : packed) fnv = (fnv ^ v) * 1099511628211ull;
    size_t diff = 0;
    unsigned row0 = 0;
    for (unsigned r : items) {
        PlowDevInst one = in;
        one.i[0] = r; one.i[4] = row0;
        k_speech<<<44, 256, kSmem>>>(one, td);
        CK(cudaDeviceSynchronize());
        alone.resize((size_t)r * n);
        CK(cudaMemcpy(alone.data(), c.dev[o], alone.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < alone.size(); i++) diff += alone[i] != packed[(size_t)row0 * n + i];
        row0 += r;
    }
    in.i[4] = 0;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int reps = 50;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaFree(td));
    printf("  DenseGemmF32 continuous M=%u N=%u K=%u, %zu requests: %zu differ, fnv %016llx, %.1f us  %.1f TFLOP/s -> %s\n",
           m, n, k, items.size(), diff, (unsigned long long)fnv, 1e3 * ms / reps, 2.0 * m * n * k / (ms / reps * 1e9),
           diff ? "FAIL" : "PASS");
    if (diff) g_fail = 1;
}

static void t_gemm_bf16(unsigned m, unsigned n, unsigned k) {
    Case c;
    unsigned o = c.out((size_t)m * n), x = c.bf16((size_t)m * k, 1.f), w = c.bf16((size_t)n * k, 0.1f);
    c.in.op = PLOW_DOP_GEMM_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w;
    c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k;
    run("GemmF32 bf16 33x70x96", c, g_gemm_f32, {o}, FP32);
}

static unsigned q8_weights(Case& c, unsigned n, unsigned k) {
    unsigned w = c.add((size_t)n * (k / 32) * 34);
    uint8_t* wp = c.host[w].data();
    for (size_t blk = 0; blk < (size_t)n * (k / 32); blk++) {
        uint16_t s = f2h(0.01f + 0.005f * rnd());
        memcpy(wp + blk * 34, &s, 2);
        for (int i = 0; i < 32; i++) wp[blk * 34 + 2 + i] = (uint8_t)(int8_t)(127.f * rnd());
    }
    return w;
}

static void t_q8(unsigned m, unsigned n, unsigned k, unsigned act) {
    Case c;
    unsigned o = c.out((size_t)m * n), x = c.f32((size_t)(m + 1) * k, 1.f);
    unsigned w = q8_weights(c, n, k);
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
    const unsigned ws = flags >> 16 ? flags >> 16 : stride;
    const unsigned of = (frames + pb + pa - kernel) / stride + 1, ow = (width + pb + pa - kernel) / ws + 1;
    const unsigned stored = flags & 1u ? 1u : ic;
    unsigned o = c.out((size_t)batches * of * ow * oc);
    unsigned x = c.f32((size_t)batches * frames * width * ic, 1.f);
    const size_t wn = (size_t)oc * stored * kernel * kernel;
    unsigned w = flags & (64u | 1024u) ? c.f32(wn, 0.1f) : c.f16(wn, 0.1f);
    if ((flags & 256u) || strstr(name, "same data")) { round_bf16(c, x); round_bf16(c, w); }
    if (flags & 1024u) w = blocked_weights(c, w, oc, ic, kernel * kernel);
    unsigned b = c.f32(oc, 0.2f);
    c.in.op = PLOW_DOP_CONV2D_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b;
    c.in.i[0] = frames; c.in.i[1] = width; c.in.i[2] = ic; c.in.i[3] = oc;
    c.in.i[4] = kernel; c.in.i[5] = stride; c.in.i[6] = pb; c.in.i[7] = pa;
    c.in.fj[1].u = flags; c.in.fj[2].u = batches;
    const size_t n = (size_t)batches * of * ow * oc;
    run(name, c, g_conv2d_f32, {o}, flags & 128u ? BF16 : FP32, 1e-5, flags & 128u ? n / 20000 : 0);
}

/* Packing parity: clip `pick` of a `batches`-wide launch against the same clip launched alone,
 * bitwise, at each block count. Device against device, no golden. */
static void t_conv_packing(const char* name, unsigned batches, unsigned frames, unsigned width, unsigned ic,
                           unsigned oc, unsigned flags) {
    const unsigned of = (frames - 1) / 2 + 1, ow = (width - 1) / 2 + 1;
    const size_t xn = (size_t)frames * width * ic, on = (size_t)of * ow * oc;
    for (unsigned nblk : {132u, 44u}) {
        Case c;
        unsigned o = c.out(batches * on), x = c.f32(batches * xn, 1.f), w = c.f32((size_t)oc * ic * 9, 0.1f);
        if (flags & 256u) { round_bf16(c, x); round_bf16(c, w); }
        const unsigned wf = w;
        if (flags & 1024u) w = blocked_weights(c, w, oc, ic, 9);
        unsigned b = c.f32(oc, 0.2f), o1 = c.out(on), x1 = c.add(xn * 4);
        c.upload();
        PlowDevInst in = c.in;
        in.op = PLOW_DOP_CONV2D_F32;
        in.t[0] = o; in.t[1] = x; in.t[2] = w; in.t[3] = b;
        in.i[0] = frames; in.i[1] = width; in.i[2] = ic; in.i[3] = oc; in.i[4] = 3; in.i[5] = 2;
        in.i[6] = 1; in.i[7] = 1; in.fj[1].u = flags; in.fj[2].u = batches;
        void** td = upload_table(c.dev);
        k_speech<<<nblk, 256, kSmem>>>(in, td);
        CK(cudaDeviceSynchronize());
        std::vector<uint32_t> packed(batches * on), alone(on);
        CK(cudaMemcpy(packed.data(), c.dev[o], batches * on * 4, cudaMemcpyDeviceToHost));
        size_t differ = 0, clips = 0;
        double maxd = 0;
        if (flags & 1024u) {
            /* The f32-stored weights give the same bits. */
            std::vector<uint32_t> ref(batches * on);
            in.t[2] = wf; in.fj[1].u = (flags & ~1024u) | 64u;
            k_speech<<<nblk, 256, kSmem>>>(in, td);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(ref.data(), c.dev[o], batches * on * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < ref.size(); i++) differ += ref[i] != packed[i];
            in.t[2] = w; in.fj[1].u = flags;
        }
        in.t[0] = o1; in.t[1] = x1; in.fj[2].u = 1;
        for (unsigned pick : {0u, 1u, batches / 2, batches - 1}) {
            if (pick >= batches) continue;
            CK(cudaMemcpy(c.dev[x1], (float*)c.dev[x] + pick * xn, xn * 4, cudaMemcpyDeviceToDevice));
            k_speech<<<nblk, 256, kSmem>>>(in, td);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(alone.data(), c.dev[o1], on * 4, cudaMemcpyDeviceToHost));
            size_t d = 0;
            for (size_t i = 0; i < on; i++) {
                if (alone[i] == packed[pick * on + i]) continue;
                float g, r;
                memcpy(&g, &packed[pick * on + i], 4); memcpy(&r, &alone[i], 4);
                maxd = std::max(maxd, fabs((double)g - r));
                d++;
            }
            differ += d; clips += d != 0;
        }
        uint64_t fnv = 1469598103934665603ull;
        for (uint32_t v : packed) fnv = (fnv ^ v) * 1099511628211ull;
        in.t[0] = o; in.t[1] = x; in.fj[2].u = batches;
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        const int reps = batches >= 64 ? 10 : 30;
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; i++) k_speech<<<nblk, 256, kSmem>>>(in, td);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaFree(td));
        printf("  %-30s b=%-3u nblk=%-3u packed vs alone: %zu elements differ in %zu clips, max|d|=%.3g, fnv %016llx, %.0f us -> %s\n",
               name, batches, nblk, differ, clips, maxd, (unsigned long long)fnv, 1e3 * ms / reps, differ ? "FAIL" : "PASS");
        if (differ) g_fail = 1;
    }
}

static void t_layernorm(const char* name, unsigned rows, unsigned feat, unsigned flags, bool affine, bool add = false) {
    Case c;
    unsigned o = c.out((size_t)rows * feat), x = c.f32((size_t)rows * feat, 2.f, 0.5f);
    unsigned g = affine ? c.f32(feat, 0.5f, 1.f) : PLOW_TENSOR_NONE;
    unsigned b = affine ? c.f32(feat, 0.2f) : PLOW_TENSOR_NONE;
    c.in.op = PLOW_DOP_LAYERNORM_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = g; c.in.t[3] = b;
    c.in.t[4] = add ? c.f32(feat, 0.7f) : PLOW_TENSOR_NONE;
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

/* flags 8: packed utterances, a {count, (first, rows) x count} group table. */
static void t_grouped_table(const char* name, unsigned rows, std::vector<uint32_t> table) {
    Case c;
    const size_t n = (size_t)rows * 1024;
    unsigned o = c.out(n), q = c.f32(n, 1.f), k = c.f32(n, 1.f), v = c.f32(n, 1.f);
    unsigned vr = c.u32(table);
    c.in.op = PLOW_DOP_GROUPED_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = vr;
    c.in.i[0] = rows; c.in.i[1] = 1024; c.in.i[2] = 64; c.in.i[3] = 104; c.in.i[4] = 7 | 8;
    run(name, c, g_grouped_attention_f32, {o}, BF16);
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
        Case c; const unsigned rows = 50, ch = 96, kernel = 5;
        unsigned o = c.out(rows * ch), x = c.f32(rows * ch, 1.f), w = c.f32(ch * kernel, 0.3f);
        c.in.op = PLOW_DOP_CAUSAL_DEPTHWISE_CONV1D_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w;
        c.in.i[0] = rows; c.in.i[1] = ch; c.in.i[2] = kernel; c.in.i[3] = 25; c.in.i[4] = 3;
        run("CausalDepthwiseConv1dF32 f32w round items", c, g_causal_depthwise_conv1d_f32, {o}, FP32);
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

static void bench_conv(unsigned batches, unsigned frames, unsigned width, unsigned ic, unsigned oc,
                       unsigned extra = 0) {
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
    in.i[6] = 1; in.i[7] = 1; in.fj[1].u = extra | 64 | 128 | (2 << 2) | (2 << 4); in.fj[2].u = batches;
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
    printf("  bench Conv2dF32%s%s 3x3/2 %ux%ux%u ic=%u oc=%u: %.3f ms  %.1f GFLOP/s\n", extra & 256u ? " tc" : "", extra & 512u ? " tf32x3" : "",
           batches, frames, width, ic, oc, ms, 2.0 * batches * of * ow * oc * ic * 9 / (ms * 1e6));
    cudaFree(o); cudaFree(x); cudaFree(w); cudaFree(b); cudaFree(td);
}

static void bench_dense(unsigned m, unsigned n, unsigned k, unsigned flags = 1u | 4u) {
    float *o, *x, *b, *sc;
    uint16_t* w;
    const size_t sc_bytes = (SPT_TICKETS + (size_t)g_nblk * SPT_BM * SPT_BN) * 4;
    CK(cudaMalloc(&o, (size_t)m * n * 4)); CK(cudaMalloc(&x, (size_t)m * k * 4));
    CK(cudaMalloc(&w, (size_t)n * k * 2)); CK(cudaMalloc(&b, (size_t)n * 4)); CK(cudaMalloc(&sc, sc_bytes));
    CK(cudaMemset(x, 0, (size_t)m * k * 4)); CK(cudaMemset(w, 0, (size_t)n * k * 2)); CK(cudaMemset(b, 0, n * 4));
    CK(cudaMemset(sc, 0, sc_bytes));
    std::vector<void*> ptrs = {o, x, w, b, sc};
    void** td = upload_table(ptrs);
    PlowDevInst in;
    memset(&in, 0, sizeof(in));
    for (auto& t : in.t) t = PLOW_TENSOR_NONE;
    in.op = PLOW_DOP_DENSE_GEMM_F32;
    in.t[0] = 0; in.t[1] = 1; in.t[2] = 2; in.t[3] = 3;
    if (flags & 8u) in.t[4] = 4;
    in.i[0] = m; in.i[1] = n; in.i[2] = k; in.i[7] = flags;
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
    printf("  bench DenseGemmF32 bf16w%s M=%u N=%u K=%u nblk=%d: %.4f ms  %.1f GFLOP/s  %.1f GB/s(W)\n",
           flags & 8u ? " tc" : "", m, n, k, g_nblk, ms, 2.0 * m * n * k / (ms * 1e6), 2.0 * n * k / (ms * 1e6));
    cudaFree(o); cudaFree(x); cudaFree(w); cudaFree(b); cudaFree(sc); cudaFree(td);
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

/* StreamTailF32: gather (fresh streams zero their tail, column window zc0) and store (z rows
 * n..n+L), against the golden; the chained case gathers then stores on the device. Copies only:
 * tol 0. */
static void st_inst(PlowDevInst& in, unsigned z, unsigned src, unsigned state, unsigned slot, unsigned n,
                    unsigned pos, unsigned B, unsigned slots, unsigned L, unsigned N, unsigned C, unsigned Cz,
                    unsigned zc0, unsigned src_stride, unsigned mode) {
    in.op = PLOW_DOP_STREAM_TAIL_F32;
    in.t[0] = z; in.t[1] = src; in.t[2] = state; in.t[3] = slot; in.t[4] = n; in.t[5] = pos;
    in.i[0] = B; in.i[1] = L; in.i[2] = N; in.i[3] = C; in.i[4] = Cz; in.i[5] = zc0; in.i[6] = src_stride;
    in.i[7] = mode; in.fj[1].u = slots;
}
static const PlowDevInst* g_chain_pre;
static void g_stream_tail_chain(const PlowDevInst* in, uint32_t b, uint32_t nb, void* const* t, PlowCpuCtx* ctx) {
    g_stream_tail_f32(g_chain_pre, b, nb, t, ctx);
    g_stream_tail_f32(in, b, nb, t, ctx);
}
static void t_stream_tail() {
    struct G { const char* name; unsigned L, N, C, Cz, zc0, src_stride, B; };
    /* Quad path: C, Cz, zc0 and the source stride multiples of 4 (C96 full width is the 12 Hz
     * codec's shape); a stride of 100 keeps the state quads but forces the scalar source read. */
    for (const G& g : {G{"StreamTailF32 gather window zc0=9", 5, 7, 24, 40, 9, 48, 4},
                       G{"StreamTailF32 gather window zc0=8", 5, 7, 24, 40, 8, 48, 4},
                       G{"StreamTailF32 gather full width L>N", 8, 3, 32, 32, 0, 32, 4},
                       G{"StreamTailF32 gather wide zc0=256", 8, 17, 512, 1024, 256, 1024, 6},
                       G{"StreamTailF32 gather C96 full width", 6, 17, 96, 96, 0, 96, 4},
                       G{"StreamTailF32 gather C96 window zc0=8", 6, 17, 96, 112, 8, 112, 4},
                       G{"StreamTailF32 gather C96 odd src stride", 6, 17, 96, 96, 0, 98, 4}}) {
        Case c; const unsigned B = g.B, slots = 6;
        unsigned z = c.out((size_t)B * (g.L + g.N) * g.Cz);
        unsigned src = c.f32((size_t)(B * g.N - 1) * g.src_stride + g.Cz, 1.f);
        unsigned state = c.f32((size_t)slots * g.L * g.C, 1.f);
        std::vector<uint32_t> vs{3, 0, 5, 1, 2, 4}, vn{g.N, 0, 3, 1, g.N, 2}, vp{0, 12, 4, 0, 30, 2};
        vs.resize(B); vn.resize(B); vp.resize(B);
        unsigned slot = c.u32(vs), n = c.u32(vn), pos = c.u32(vp);
        st_inst(c.in, z, src, state, slot, n, pos, B, slots, g.L, g.N, g.C, g.Cz, g.zc0, g.src_stride, 0);
        run(g.name, c, g_stream_tail_f32, {z}, INT);
    }
    /* n = 0, n < L, n == L, n > L (= N) and a window: state rows read z rows n..n+L. zc0 = 9 takes
     * the scalar walk, zc0 = 8 the float4 one. */
    for (unsigned zc0 : {9u, 8u}) {
        Case c; const unsigned B = 4, slots = 6, L = 5, N = 7, C = 24, Cz = 40;
        unsigned z = c.f32((size_t)B * (L + N) * Cz, 1.f);
        unsigned state = c.f32((size_t)slots * L * C, 1.f);
        unsigned slot = c.u32({2, 5, 0, 4}), n = c.u32({0, 3, 5, 7});
        st_inst(c.in, PLOW_TENSOR_NONE, z, state, slot, n, PLOW_TENSOR_NONE, B, slots, L, N, C, Cz, zc0, 0, 1);
        run(zc0 == 9u ? "StreamTailF32 store n=0,<L,=L,>L zc0=9" : "StreamTailF32 store n=0,<L,=L,>L zc0=8",
            c, g_stream_tail_f32, {state}, INT);
    }
    {
        /* Quad store at the codec's C96 (full width) and with an aligned window. */
        for (unsigned zc0 : {0u, 8u}) {
            Case c; const unsigned B = 4, slots = 6, L = 6, N = 17, C = 96, Cz = 96 + 2 * zc0;
            unsigned z = c.f32((size_t)B * (L + N) * Cz, 1.f);
            unsigned state = c.f32((size_t)slots * L * C, 1.f);
            unsigned slot = c.u32({2, 5, 0, 4}), n = c.u32({0, 3, 6, 17});
            st_inst(c.in, PLOW_TENSOR_NONE, z, state, slot, n, PLOW_TENSOR_NONE, B, slots, L, N, C, Cz, zc0, 0, 1);
            run(zc0 ? "StreamTailF32 store C96 quads zc0=8" : "StreamTailF32 store C96 quads", c, g_stream_tail_f32,
                {state}, FP32, 0.0);
        }
    }
    /* Device gather then store against the chained golden; z starts poisoned. Items with pos = 0
     * and n = 0 must leave zeros. zc0 = 4 takes the float4 walks, zc0 = 5 the scalar ones. */
    for (unsigned zc0 : {4u, 5u}) {
        Case c; const unsigned B = 5, slots = 5, L = 6, N = 4, C = 16, Cz = 24, ss = 24;
        unsigned z = c.out((size_t)B * (L + N) * Cz);
        unsigned src = c.f32((size_t)B * N * ss, 1.f);
        unsigned state = c.f32((size_t)slots * L * C, 1.f);
        unsigned slot = c.u32({4, 1, 3, 0, 2}), n = c.u32({0, 4, 2, 0, 3}), pos = c.u32({0, 9, 0, 7, 3});
        PlowDevInst gather;
        memset(&gather, 0, sizeof(gather));
        for (auto& t : gather.t) t = PLOW_TENSOR_NONE;
        st_inst(gather, z, src, state, slot, n, pos, B, slots, L, N, C, Cz, zc0, ss, 0);
        st_inst(c.in, PLOW_TENSOR_NONE, z, state, slot, n, PLOW_TENSOR_NONE, B, slots, L, N, C, Cz, zc0, 0, 1);
        g_chain_pre = &gather;
        run(zc0 == 4u ? "StreamTailF32 chain zc0=4" : "StreamTailF32 chain zc0=5", c, g_stream_tail_chain,
            {z, state}, INT, 0.0, 0, &gather);
        const float* st = (const float*)c.host[state].data();
        bool zeros = true;
        for (size_t i = 0; i < (size_t)L * C; i++) zeros &= st[(size_t)4 * L * C + i] == 0.f;
        printf("  %-34s fresh n=0 slot zero -> %s\n", "StreamTailF32 chain", zeros ? "PASS" : "FAIL");
        if (!zeros) g_fail = 1;
    }
    {
        /* The 12 Hz codec's shape: C = Cz = 96, 17 frames of 10 rows; gather (quads) then store
         * (quads) on the device, n = 0 / < L / = L / = N. */
        Case c; const unsigned B = 4, slots = 5, L = 6, N = 170, C = 96, Cz = 96, zc0 = 0, ss = 96;
        unsigned z = c.out((size_t)B * (L + N) * Cz);
        unsigned src = c.f32((size_t)B * N * ss, 1.f);
        unsigned state = c.f32((size_t)slots * L * C, 1.f);
        unsigned slot = c.u32({4, 1, 3, 0}), n = c.u32({0, 3, 6, 170}), pos = c.u32({0, 5, 9, 2});
        PlowDevInst gather;
        memset(&gather, 0, sizeof(gather));
        for (auto& t : gather.t) t = PLOW_TENSOR_NONE;
        st_inst(gather, z, src, state, slot, n, pos, B, slots, L, N, C, Cz, zc0, ss, 0);
        st_inst(c.in, PLOW_TENSOR_NONE, z, state, slot, n, PLOW_TENSOR_NONE, B, slots, L, N, C, Cz, zc0, 0, 1);
        std::vector<void*> th;
        for (auto& h : c.host) th.push_back(h.data());
        g_stream_tail_f32(&gather, 0, 1, th.data(), nullptr);
        run("StreamTailF32 C96 gather->store chain", c, g_stream_tail_f32, {state}, FP32, 0.0, 0, &gather);
    }
}

struct ConvSpec {
    unsigned batch, in_rows, cin, cout, kernel, stride, dil_or_opad, groups, before, after;
    unsigned mode, pre, post;
    bool f16, bias, alpha, residual;
    std::vector<uint32_t> lengths;
    float slope;
    bool tc = false; /* flags bit 13: 3xTF32 tensor cores */
    bool row_scale = false; /* t7 (Conv1d) */
    bool wide = false; /* flags bit 14: split-bf16 wide tiles */
    bool tapmajor = false; /* flags bit 15: weights [cout][kernel][cin/groups] */
    bool wg = false; /* flags bit 17: the wgmma arm of the pointwise wide tile (sm_90a) */
    bool dual = false; /* flags bit 19: alpha [cin | cout], an input then an output snake */
    bool rows = false; /* flags bit 16: the small-M rows arm (else the warp arm at cout <= 4) */
    bool splitk = false; /* flags bit 20: t7 is the wgmma tile's split-K scratch */
    unsigned pin = 0; /* flags bits 21..23: 1 pinned arms, 2 FFMA tiles, 4 a 128-wide tile */
    unsigned sk = 0; /* flags bits 24..29 (pinned): fixed split-K parts; t7 = the conv-own scratch */
    bool beta = false; /* flags bit 2 (post snake): SnakeBeta output, alpha [a | r] of 2 * cout */
    bool ragged = false; /* flags bit 3: wgmma tiles may span taps (widths % 4, not % 64) */
};
/* Pinned split-K scratch: SPC_TICKETS zeroed tickets, then tiles x parts 128x64 partials. */
static unsigned splitk_scratch_pinned(Case& c, const ConvSpec& s, unsigned out_rows) {
    const size_t tiles = ((size_t)s.batch * out_rows + 127) / 128 * ((s.cout + 63) / 64);
    const size_t nk = ((size_t)s.cin * s.kernel + 63) / 64, chunk = std::max<size_t>(1, (nk + s.sk - 1) / s.sk), parts = (nk + chunk - 1) / chunk;
    const size_t bytes = (SPC_TICKETS + tiles * parts * 128 * 64) * 4;
    unsigned id = c.add(bytes);
    memset(c.host[id].data(), 0, bytes);
    return id;
}

static unsigned conv_out_rows(const ConvSpec& s, bool transpose) {
    if (transpose) return (s.in_rows - 1) * s.stride + s.kernel + s.dil_or_opad - s.before - s.after;
    return (s.in_rows + s.before + s.after - s.dil_or_opad * (s.kernel - 1) - 1) / s.stride + 1;
}

static uint16_t bf16_rn(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
/* Appends the wgmma pre-split image (Conv1dF32 flags bit 18) of f32 weights [n][k]: at the next
 * 1024-byte boundary, per 64-column k-tile the hi then lo bf16 halves of n rows padded to 128,
 * each row's 16-byte chunk c at chunk c ^ (row & 7). */
static void append_wsplit(std::vector<uint8_t>& w, unsigned n, unsigned k, unsigned channel_kernel = 0) {
    if (channel_kernel && (k % channel_kernel || (k / channel_kernel) % 64u)) exit(2);
    const size_t base = (w.size() + 1023) & ~(size_t)1023, np = (n + 127) & ~127u, kt = (k + 63) / 64;
    std::vector<float> f(w.size() / 4);
    memcpy(f.data(), w.data(), f.size() * 4);
    w.resize(base + kt * 2 * np * 128, 0);
    for (size_t t = 0; t < kt; t++)
        for (size_t r = 0; r < n; r++)
            for (size_t j = 0; j < 64 && t * 64 + j < k; j++) {
                const size_t col = channel_kernel ? ((t / channel_kernel) * 64u + j) * channel_kernel + t % channel_kernel
                                                   : t * 64 + j;
                const float v = f[r * k + col];
                const uint16_t hi = bf16_rn(v);
                const uint32_t hb = (uint32_t)hi << 16;
                float hf;
                memcpy(&hf, &hb, 4);
                const uint16_t lo = bf16_rn(v - hf);
                const size_t off = base + t * 2 * np * 128 + r * 128 + (((j / 8) ^ (r & 7)) * 16) + (j % 8) * 2;
                memcpy(&w[off], &hi, 2);
                memcpy(&w[off + np * 128], &lo, 2);
            }
}
static void build_conv(Case& c, const ConvSpec& s, bool transpose, unsigned& o) {
    const unsigned out_rows = conv_out_rows(s, transpose);
    o = c.out((size_t)s.batch * out_rows * s.cout);
    unsigned x = c.f32((size_t)s.batch * s.in_rows * s.cin, 1.f);
    const size_t wn = (size_t)s.cout * (s.cin / s.groups) * s.kernel;
    const float amp = 1.5f / sqrtf((float)(s.cin / s.groups * s.kernel));
    unsigned w = s.f16 ? c.f16(wn, amp) : c.f32(wn, amp);
    if (s.tapmajor && transpose) {
        /* [cin][cout][k] -> [stride][cout][ceil(k/stride)][cin] (groups == 1), zero past k. */
        const unsigned taps = (s.kernel + s.stride - 1) / s.stride, es = s.f16 ? 2 : 4;
        std::vector<uint8_t> src = c.host[w];
        c.host[w].assign((size_t)s.stride * s.cout * taps * s.cin * es, 0);
        for (unsigned i = 0; i < s.cin; i++)
            for (unsigned o = 0; o < s.cout; o++)
                for (unsigned k = 0; k < s.kernel; k++)
                    memcpy(&c.host[w][((((size_t)(k % s.stride) * s.cout + o) * taps + k / s.stride) * s.cin + i) * es],
                           &src[(((size_t)i * s.cout + o) * s.kernel + k) * es], es);
    } else if (s.tapmajor) {
        const unsigned cg = s.cin / s.groups, es = s.f16 ? 2 : 4;
        std::vector<uint8_t> src = c.host[w];
        for (unsigned o = 0; o < s.cout; o++)
            for (unsigned i = 0; i < cg; i++)
                for (unsigned k = 0; k < s.kernel; k++)
                    memcpy(&c.host[w][(((size_t)o * s.kernel + k) * cg + i) * es], &src[(((size_t)o * cg + i) * s.kernel + k) * es], es);
    }
    /* ConvTranspose: the merged-phase image [stride*cout][2*cin] of phase-major weights. */
    const bool wsplit = s.wg && s.groups == 1 && !s.f16 && (!transpose || (s.tapmajor && s.kernel == 2 * s.stride));
    if (wsplit && transpose) append_wsplit(c.host[w], s.stride * s.cout, 2 * s.cin);
    else if (wsplit) append_wsplit(c.host[w], s.cout, s.cin * s.kernel,
        !s.tapmajor && s.kernel > 1u && s.cin % 64u == 0u ? s.kernel : 0u);
    unsigned b = s.bias ? c.f32(s.cout, 0.2f) : PLOW_TENSOR_NONE;
    unsigned a = s.alpha ? c.f32(s.beta ? 2 * s.cout : s.dual ? s.cin + s.cout : std::max(s.cin, s.cout), 0.4f, 0.6f)
                         : PLOW_TENSOR_NONE;
    unsigned r = s.residual ? c.f32((size_t)s.batch * out_rows * s.cout, 1.f) : PLOW_TENSOR_NONE;
    unsigned l = s.lengths.empty() ? PLOW_TENSOR_NONE : c.u32(s.lengths);
    if (s.row_scale) c.in.t[7] = c.f32((size_t)s.batch * out_rows, 1.f);
    if (s.splitk) c.in.t[7] = splitk_scratch(c);
    if (s.sk) c.in.t[7] = splitk_scratch_pinned(c, s, out_rows);
    c.in.op = transpose ? PLOW_DOP_CONV_TRANSPOSE1D_F32 : PLOW_DOP_CONV1D_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b; c.in.t[4] = a; c.in.t[5] = r; c.in.t[6] = l;
    c.in.i[0] = s.batch; c.in.i[1] = s.in_rows; c.in.i[2] = s.cin; c.in.i[3] = s.cout;
    c.in.i[4] = s.kernel; c.in.i[5] = s.stride; c.in.i[6] = s.dil_or_opad; c.in.i[7] = s.groups;
    c.in.fj[0].f = s.slope; c.in.fj[1].u = pads(s.before, s.after);
    c.in.fj[2].u = conv_flags(s.mode, s.pre, s.post, s.f16) | (s.tc ? 1u << 13 : 0u) | (s.wide ? 1u << 14 : 0u) |
                   (s.tapmajor ? 1u << 15 : 0u) | (s.wg ? 1u << 17 : 0u) | (wsplit ? 1u << 18 : 0u) |
                   (s.dual ? 1u << 19 : 0u) | (s.rows ? 1u << 16 : 0u) | (s.splitk || s.sk ? 1u << 20 : 0u) |
                   (s.pin << 21) | (s.sk << 24) | (s.beta ? 1u << 2 : 0u) | (s.ragged ? 1u << 3 : 0u);
}

static void t_conv1d(const char* name, const ConvSpec& s, double tol = 2e-5) {
    Case c; unsigned o;
    build_conv(c, s, false, o);
    run(name, c, g_conv1d_f32, {o}, FP32, tol);
    if (!s.splitk && !s.sk) return;
    /* The reducers re-arm every ticket. */
    const unsigned nt = s.sk ? SPC_TICKETS : SPT_TICKETS;
    std::vector<uint32_t> tk(nt);
    CK(cudaMemcpy(tk.data(), c.dev[c.in.t[7]], nt * 4, cudaMemcpyDeviceToHost));
    bool zero = true;
    for (uint32_t t : tk) zero &= t == 0u;
    printf("  %-34s tickets re-armed -> %s\n", name, zero ? "PASS" : "FAIL");
    if (!zero) g_fail = 1;
}

static void t_convt1d(const char* name, const ConvSpec& s, double tol = 2e-5) {
    Case c; unsigned o;
    build_conv(c, s, true, o);
    run(name, c, g_conv_transpose1d_f32, {o}, FP32, tol);
}

__global__ void __launch_bounds__(256, 1) k_conv_warp_ref(PlowDevInst in, void* const* T) {
    SpConvArgs a;
    if (sp_conv_args(&in, T, false, a)) sp_conv1d_warp(a, blockIdx.x, gridDim.x);
}
/* Bit-16 cout <= 4 shapes the lanes arm takes: against the golden, then bit for bit against the
 * warp arm it replaces. */
static void t_conv1d_lanes(const char* name, ConvSpec s, bool tapmajor = true) {
    s.rows = true;
    s.tapmajor = tapmajor;
    Case c; unsigned o;
    build_conv(c, s, false, o);
    run(name, c, g_conv1d_f32, {o}, FP32, 2e-5);
    std::vector<void*> ptrs = c.dev;
    const size_t bytes = c.host[o].size();
    CK(cudaMalloc(&ptrs[o], bytes));
    void** td = upload_table(ptrs);
    k_conv_warp_ref<<<g_nblk, 256>>>(c.in, td);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(bytes), ref(bytes);
    CK(cudaMemcpy(got.data(), c.dev[o], bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), ptrs[o], bytes, cudaMemcpyDeviceToHost));
    CK(cudaFree(ptrs[o]));
    CK(cudaFree(td));
    const bool same = got == ref;
    printf("  %-34s vs warp arm bit-identical -> %s\n", name, same ? "PASS" : "FAIL");
    if (!same) g_fail = 1;
}

/* Batch invariance of a conv: item 0's output bits match whether it runs alone or in the batch,
 * on all blocks or one. On the wgmma tile the f32 weights are NaN (the golden keeps them), so a
 * fallback off it fails. */
static void t_conv_invariant(const char* name, const ConvSpec& s, bool transpose, double tol = 1e-4) {
    Case c; unsigned o;
    build_conv(c, s, transpose, o);
    const PlowDevInst ref = c.in;
    if (s.wg && (transpose || s.cin % 64u == 0)) {
        const unsigned w = c.in.t[2], wp = c.add(c.host[w].size());
        c.host[wp] = c.host[w];
        const float nan = NAN;
        const size_t nf = transpose ? (size_t)s.stride * s.cout * 2 * s.cin : (size_t)s.cout * s.cin * s.kernel;
        for (size_t i = 0; i < nf; i++) memcpy(&c.host[wp][i * 4], &nan, 4);
        c.in.t[2] = wp;
    }
    run(name, c, transpose ? g_conv_transpose1d_f32 : g_conv1d_f32, {o}, FP32, tol, 0, nullptr, &ref);
    const size_t item = (size_t)conv_out_rows(s, transpose) * s.cout, n = c.host[o].size() / 4;
    std::vector<uint32_t> full(n), got(n);
    CK(cudaMemcpy(full.data(), c.dev[o], n * 4, cudaMemcpyDeviceToHost));
    void** td = upload_table(c.dev);
    struct V { const char* what; unsigned batch, nblk; } vs[] = {
        {"alone", 1, (unsigned)g_nblk}, {"alone 1 block", 1, 1}, {"batch 7 blocks", s.batch, 7}, {"batch 1 block", s.batch, 1}};
    for (const V& v : vs) {
        PlowDevInst in = c.in;
        in.i[0] = v.batch;
        CK(cudaMemset(c.dev[o], 0xff, n * 4));
        k_speech<<<v.nblk, 256, kSmem>>>(in, td);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), c.dev[o], n * 4, cudaMemcpyDeviceToHost));
        const size_t cmp = v.batch == 1u ? item : n;
        const bool same = !memcmp(got.data(), full.data(), cmp * 4);
        printf("  %-34s %s bitwise == batch %u -> %s\n", name, v.what, s.batch, same ? "PASS" : "FAIL");
        if (!same) g_fail = 1;
    }
    CK(cudaFree(td));
}

/* A SnakeBeta output (post snake + flags bit 2, no residual / row scale): against the golden, batch-
 * invariant, and bitwise equal to the plain conv followed by UnaryF32 SnakeBeta (kind 19) over the
 * same [a | r]. */
static void t_conv_beta(const char* name, const ConvSpec& s) {
    /* The snake's slope scales the split-bf16 conv error: 3e-4 against the f64 golden. */
    t_conv_invariant(name, s, false, 3e-4);
    Case c; unsigned o;
    build_conv(c, s, false, o);
    const size_t n = c.host[o].size() / 4;
    const unsigned u = c.add(n * 4);
    run(name, c, g_conv1d_f32, {o}, FP32, 3e-4);
    std::vector<uint32_t> fused(n), two(n);
    CK(cudaMemcpy(fused.data(), c.dev[o], n * 4, cudaMemcpyDeviceToHost));
    void** td = upload_table(c.dev);
    PlowDevInst conv = c.in;
    conv.fj[2].u &= ~(4u | (15u << 8));
    k_speech<<<g_nblk, 256, kSmem>>>(conv, td);
    PlowDevInst un{};
    un.op = PLOW_DOP_UNARY_F32; un.t[0] = u; un.t[1] = o; un.t[2] = c.in.t[4];
    for (unsigned i = 3; i < 8; i++) un.t[i] = PLOW_TENSOR_NONE;
    un.i[0] = (unsigned)(n / s.cout); un.i[1] = s.cout; un.i[2] = 19;
    CK(cudaDeviceSynchronize());
    k_speech<<<g_nblk, 256, kSmem>>>(un, td);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(two.data(), c.dev[u], n * 4, cudaMemcpyDeviceToHost));
    /* Masked rows: the conv stores 0, the unary maps 0 to 0. */
    const bool same = !memcmp(fused.data(), two.data(), n * 4);
    printf("  %-34s fused bitwise == conv + unary -> %s\n", name, same ? "PASS" : "FAIL");
    if (!same) g_fail = 1;
    CK(cudaFree(td));
}

/* The staged few-output conv (cout <= 4, ungrouped) against sp_conv1d_direct: bitwise equal, and
 * batch-invariant. */
static void t_conv_staged(const char* name, const ConvSpec& s) {
    t_conv_invariant(name, s, false);
    Case c; unsigned o;
    build_conv(c, s, false, o);
    run(name, c, g_conv1d_f32, {o}, FP32, 1e-4);
    const size_t n = c.host[o].size() / 4;
    std::vector<uint32_t> got(n), ref(n);
    CK(cudaMemcpy(got.data(), c.dev[o], n * 4, cudaMemcpyDeviceToHost));
    void** td = upload_table(c.dev);
    CK(cudaMemset(c.dev[o], 0xff, n * 4));
    k_conv_direct<<<g_nblk, 256>>>(c.in, td);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), c.dev[o], n * 4, cudaMemcpyDeviceToHost));
    const bool same = !memcmp(got.data(), ref.data(), n * 4);
    printf("  %-34s staged bitwise == direct -> %s\n", name, same ? "PASS" : "FAIL");
    if (!same) g_fail = 1;
    CK(cudaFree(td));
}

/* Two specs over the same data (same seed): each against the golden, then bitwise against each
 * other (`same` = expected equal, e.g. the two pinned tile widths; else expected to differ, e.g.
 * 3xTF32 vs FFMA). */
static void t_conv_pair(const char* name, const ConvSpec& a, const ConvSpec& b, bool transpose, bool same,
                        double tol = 2e-5) {
    std::vector<uint32_t> bits[2];
    const uint32_t seed = rng_s;
    for (int i = 0; i < 2; i++) {
        rng_s = seed;
        Case c; unsigned o;
        build_conv(c, i ? b : a, transpose, o);
        char nm[96];
        snprintf(nm, sizeof(nm), "%s [%s]", name, i ? "b" : "a");
        run(nm, c, transpose ? g_conv_transpose1d_f32 : g_conv1d_f32, {o}, FP32, tol);
        bits[i].resize(c.host[o].size() / 4);
        CK(cudaMemcpy(bits[i].data(), c.dev[o], bits[i].size() * 4, cudaMemcpyDeviceToHost));
    }
    const bool eq = bits[0] == bits[1];
    if (same) {
        printf("  %-34s a bitwise == b -> %s\n", name, eq ? "PASS" : "FAIL");
        if (!eq) g_fail = 1;
    } else {
        printf("  %-34s a bitwise == b -> %s (different arms, informational)\n", name, eq ? "yes" : "no");
    }
}

/* Pinned arms (flags bit 21): the 3xTF32 / FFMA choice, the tile width and the warp arm come from
 * the flags, never from the batch. Mimi's ELU-pre k3 convs at both pins; the 12 Hz codec's 96->1
 * output conv on the warp arm at M 1 / 32 / 33 and batched; the wide and wgmma tiles at both
 * widths (bit-identical: per output the k order is the tile's). */
static void t_conv_pinned() {
    {
        ConvSpec tc{4, 1024, 512, 256, 3, 1, 1, 1, 1, 1, 0, 8, 0, false, true, false, false, {}, 1.f, true};
        tc.pin = 1;
        ConvSpec ff = tc;
        ff.pin = 1 | 2;
        t_conv_pair("Conv1dF32 pinned k3 elu tf32 vs ffma", tc, ff, false, false);
        t_conv_invariant("Conv1dF32 pinned k3 elu tf32", tc, false);
        t_conv_invariant("Conv1dF32 pinned k3 elu ffma", ff, false);
        /* 122880 rows per item: the former crossover would have switched to FFMA; pinned it stays. */
        ConvSpec big = tc;
        big.batch = 2; big.in_rows = 30720;
        t_conv_invariant("Conv1dF32 pinned k3 elu tf32 big M", big, false);
    }
    {
        ConvSpec w{3, 300, 96, 1, 7, 1, 1, 1, 6, 0, 0, 0, 0, false, true, false, false, {300, 100, 0}, 0.f, true};
        w.tapmajor = true; w.rows = true; w.pin = 1;
        t_conv_invariant("Conv1dF32 pinned warp k7 96->1 len", w, false);
        ConvSpec d = w;
        d.rows = false;
        t_conv_pair("Conv1dF32 pinned 96->1 warp vs direct", w, d, false, false);
        t_conv_staged("Conv1dF32 staged 96->1 k7 len", d);
        ConvSpec big = d;
        big.batch = 2; big.in_rows = 9000; big.lengths = {9000, 4321};
        t_conv_staged("Conv1dF32 staged 96->1 9k rows", big);
        ConvSpec st{3, 301, 96, 3, 5, 2, 3, 1, 4, 4, 1, 9, 11, false, true, false, true, {301, 150, 7}, 0.2f, true};
        st.pin = 1;
        t_conv_staged("Conv1dF32 staged 96->3 s2 d3 reflect pre/post/res", st);
        ConvSpec rp = st;
        rp.mode = 2; rp.cin = 256; rp.cout = 4; rp.kernel = 7; rp.stride = 1; rp.dil_or_opad = 1; rp.f16 = true;
        t_conv_staged("Conv1dF32 staged 256->4 k7 replicate f16", rp);
        for (unsigned rows : {7u, 38u, 39u}) {
            ConvSpec m = w;
            m.batch = 1; m.in_rows = rows; m.before = 0; m.lengths = {};
            char name[64];
            snprintf(name, sizeof(name), "Conv1dF32 pinned warp 96->1 M%u", rows - 6);
            t_conv1d(name, m);
        }
    }
    {
        /* SnakeBeta output on every arm the 12 Hz c1 convs (and others) can take. */
        ConvSpec t{3, 300, 192, 192, 7, 1, 3, 1, 18, 0, 0, 0, 12, false, true, true, false, {300, 50, 1}, 0.f, true};
        t.wide = true; t.wg = true; t.tapmajor = true; t.pin = 1; t.beta = true;
        t_conv_beta("Conv1dF32 beta wgmma taps 192 d3", t);
        ConvSpec t128 = t;
        t128.pin = 1 | 4; t128.cin = t128.cout = 384;
        t_conv_beta("Conv1dF32 beta wgmma taps 384 bn128", t128);
        ConvSpec w = t;
        w.cin = w.cout = 96; w.wg = false;
        t_conv_beta("Conv1dF32 beta wide 96 d3", w);
        ConvSpec w128 = w;
        w128.pin = 1 | 4;
        t_conv_beta("Conv1dF32 beta wide 96 bn128", w128);
        ConvSpec tc = t;
        tc.wide = tc.wg = false; tc.tapmajor = false;
        t_conv_beta("Conv1dF32 beta 3xTF32 192", tc);
        ConvSpec ff = tc;
        ff.pin = 1 | 2;
        t_conv_beta("Conv1dF32 beta FFMA 192", ff);
        ConvSpec st = t;
        st.cin = 96; st.cout = 3; st.wide = st.wg = false;
        t_conv_beta("Conv1dF32 beta staged 96->3", st);
        ConvSpec pw = t;
        /* The pointwise wgmma tile has no SnakeBeta epilogue: the wide tile takes it. */
        pw.kernel = 1; pw.dil_or_opad = 1; pw.before = 0; pw.wg = false;
        t_conv_beta("Conv1dF32 beta wide pointwise", pw);
    }
    {
        /* Arms first combined by the integration: pinned widths and the SnakeBeta epilogue on
         * ragged wgmma tiles (96 channels: k-tiles span taps, the N tile is partial), the lanes
         * arm under a pinned bit-16 class, and the widest parts field beside the split-image
         * bits (30 / 31 stay clear). */
        ConvSpec r{3, 300, 96, 96, 7, 1, 3, 1, 18, 0, 0, 0, 0, false, true, false, true, {300, 50, 1}, 0.f, true};
        r.wide = true; r.wg = true; r.tapmajor = true; r.pin = 1; r.ragged = true;
        ConvSpec r128 = r;
        r128.pin = 1 | 4;
        t_conv_pair("Conv1dF32 pinned ragged 96 bn64 vs bn128", r, r128, false, true, 1e-4);
        t_conv_invariant("Conv1dF32 pinned ragged 96 bn64", r, false);
        t_conv_invariant("Conv1dF32 pinned ragged 96 bn128", r128, false);
        ConvSpec rb = r;
        rb.residual = false; rb.alpha = true; rb.post = 12; rb.beta = true;
        t_conv_beta("Conv1dF32 beta ragged 96 d3", rb);
        ConvSpec rb128 = rb;
        rb128.pin = 1 | 4;
        t_conv_beta("Conv1dF32 beta ragged 96 bn128", rb128);
        ConvSpec ln{3, 300, 96, 1, 7, 1, 1, 1, 6, 0, 0, 0, 0, false, true, false, false, {300, 100, 0}, 0.f, true};
        ln.pin = 1;
        t_conv1d_lanes("Conv1dF32 pinned lanes 96->1", ln);
        ln.rows = true; ln.tapmajor = true;
        t_conv_invariant("Conv1dF32 pinned lanes 96->1", ln, false);
        ConvSpec p63{1, 64, 4096, 1024, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true};
        p63.wide = true; p63.wg = true; p63.pin = 1; p63.sk = 63;
        t_conv1d("Conv1dF32 pinned splitk P63 4096->1024", p63, 1e-4);
        t_conv_invariant("Conv1dF32 pinned splitk P63 4096->1024", p63, false);
    }
    {
        /* Wide tile (split-bf16, no wgmma): both pinned widths over the same data. */
        ConvSpec a{5, 700, 96, 96, 7, 1, 3, 1, 9, 9, 0, 0, 0, false, true, false, true, {}, 0.f, true};
        a.wide = true; a.tapmajor = true; a.pin = 1;
        ConvSpec b = a;
        b.pin = 1 | 4;
        t_conv_pair("Conv1dF32 pinned wide k7 96 bn64 vs bn128", a, b, false, true, 1e-4);
        t_conv_invariant("Conv1dF32 pinned wide k7 96 bn128", b, false);
    }
#if defined(PLOW_NV_HOPPER)
    {
        ConvSpec a{5, 300, 512, 512, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, true, {}, 0.f, true};
        a.wide = true; a.wg = true; a.pin = 1;
        ConvSpec b = a;
        b.pin = 1 | 4;
        t_conv_pair("Conv1dF32 pinned wgmma pointwise bn64 vs bn128", a, b, false, true, 1e-4);
        ConvSpec t{5, 300, 384, 384, 7, 1, 3, 1, 9, 9, 0, 0, 0, false, true, false, true, {300, 50, 1, 300, 299}, 0.f, true};
        t.wide = true; t.wg = true; t.tapmajor = true; t.pin = 1;
        ConvSpec u = t;
        u.pin = 1 | 4;
        t_conv_pair("Conv1dF32 pinned wgmma k7 384 bn64 vs bn128", t, u, false, true, 1e-4);
        t_conv_invariant("Conv1dF32 pinned wgmma k7 384 bn128", u, false);
        ConvSpec ct{6, 40, 1024, 512, 16, 8, 0, 1, 4, 4, 0, 0, 0, false, true, false, false, {40, 0, 1, 39, 40, 20}, 0.f, true};
        ct.wide = true; ct.wg = true; ct.tapmajor = true; ct.pin = 1;
        ConvSpec cu = ct;
        cu.pin = 1 | 4;
        t_conv_pair("ConvT pinned wgmma s8 1024->512 bn64 vs bn128", ct, cu, true, true, 1e-4);
        t_conv_invariant("ConvT pinned wgmma s8 1024->512 bn128", cu, true);
    }
    {
        /* Fixed split-K (bits 24..29): the transformer denses at the step's 64 rows (8 tiles of
         * 1024->512 on 132 blocks), parts that do not divide nk (16 k-tiles: 5 -> chunk 4 -> 4
         * parts), parts past nk (32 -> 16), a taps k7 conv, a split with the residual epilogue;
         * every split batch-invariant (alone / 1 block / batched bitwise) and golden-checked. */
        ConvSpec wo{3, 64, 1024, 512, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true};
        wo.wide = true; wo.wg = true; wo.pin = 1;
        for (unsigned P : {2u, 5u, 16u, 32u}) {
            ConvSpec a = wo;
            a.sk = P;
            char name[64];
            snprintf(name, sizeof(name), "Conv1dF32 pinned splitk P%u 1024->512 M64", P);
            t_conv1d(name, a, 1e-4);
            t_conv_invariant(name, a, false);
        }
        ConvSpec w1 = wo;
        w1.cin = 512; w1.cout = 3072; w1.sk = 3; /* nk 8: chunk 3 -> 3 parts (3, 3, 2) */
        t_conv1d("Conv1dF32 pinned splitk P3 512->3072 M64", w1, 1e-4);
        t_conv_invariant("Conv1dF32 pinned splitk P3 512->3072 M64", w1, false);
        ConvSpec res = wo;
        res.residual = true; res.sk = 8; res.lengths = {64, 10, 0};
        t_conv1d("Conv1dF32 pinned splitk P8 +res len", res, 1e-4);
        t_conv_invariant("Conv1dF32 pinned splitk P8 +res len", res, false);
        ConvSpec one = wo;
        one.batch = 1; one.sk = 16;
        t_conv1d("Conv1dF32 pinned splitk P16 one item", one, 1e-4);
        ConvSpec taps{2, 150, 1024, 1536, 7, 1, 1, 1, 6, 0, 0, 0, 0, false, true, false, false, {150, 77}, 0.f, true};
        taps.wide = true; taps.wg = true; taps.tapmajor = true; taps.pin = 1; taps.sk = 4;
        t_conv1d("Conv1dF32 pinned splitk P4 k7 1024->1536", taps, 1e-4);
        t_conv_invariant("Conv1dF32 pinned splitk P4 k7 1024->1536", taps, false);
        /* P = 1 is the unsplit tile (bit 20 set, no scratch use beyond the tickets). */
        ConvSpec p1 = wo;
        p1.sk = 1;
        t_conv_pair("Conv1dF32 pinned splitk P1 vs unsplit", p1, wo, false, true, 1e-4);
    }
#endif
}

static void t_conv_padding() {
    for (unsigned mode : {0u, 1u, 2u})
        for (unsigned kernel : {1u, 3u})
            for (bool leaky : {false, true}) {
                const unsigned pad = kernel == 1u ? 2u : 3u;
                ConvSpec s{5, 129, 64, 96, kernel, 1, kernel == 1u ? 1u : 3u, 1, pad, pad,
                           mode, leaky ? 9u : 0u, 0, false, true, false, true, {129, 93, 17, 1, 0}, 0.1f, true};
                s.wide = s.tapmajor = s.row_scale = true;
                Case c; unsigned o;
                build_conv(c, s, false, o);
                float* x = (float*)c.host[c.in.t[1]].data();
                for (unsigned b = 0; b < s.batch; b++)
                    for (unsigned t = s.lengths[b]; t < s.in_rows; t++)
                        for (unsigned ch = 0; ch < s.cin; ch++) x[((size_t)b * s.in_rows + t) * s.cin + ch] = NAN;
                char name[96];
                snprintf(name, sizeof(name), "Conv1dF32 wide pad%u k%u leaky%u poison", mode, kernel, leaky);
                run(name, c, g_conv1d_f32, {o}, FP32, 1e-4);
                std::vector<float> got(c.host[o].size() / 4);
                CK(cudaMemcpy(got.data(), c.dev[o], c.host[o].size(), cudaMemcpyDeviceToHost));
                const float* ref = (const float*)c.host[o].data();
                bool finite = true;
                for (size_t i = 0; i < got.size(); i++) finite &= std::isfinite(got[i]) && std::isfinite(ref[i]);
                printf("  %-34s finite -> %s\n", name, finite ? "PASS" : "FAIL");
                if (!finite) g_fail = 1;
            }
    for (unsigned cin : {64u, 128u})
        for (unsigned kernel : {3u, 5u}) {
            const unsigned pad = kernel - 1u;
            ConvSpec s{5, 129, cin, 96, kernel, 1, 2, 1, pad, pad,
                       0, 0, 0, false, true, false, true, {129, 93, 17, 1, 0}, 0.f, true};
            s.wide = s.wg = true;
            Case c; unsigned o;
            build_conv(c, s, false, o);
            float* x = (float*)c.host[c.in.t[1]].data();
            for (unsigned b = 0; b < s.batch; b++)
                for (unsigned t = s.lengths[b]; t < s.in_rows; t++)
                    for (unsigned ch = 0; ch < cin; ch++) x[((size_t)b * s.in_rows + t) * cin + ch] = NAN;
            char name[96];
            snprintf(name, sizeof(name), "Conv1dF32 channel64 c%u k%u poison", cin, kernel);
            run(name, c, g_conv1d_f32, {o}, FP32, 1e-4);
            std::vector<float> got(c.host[o].size() / 4);
            CK(cudaMemcpy(got.data(), c.dev[o], c.host[o].size(), cudaMemcpyDeviceToHost));
            const float* ref = (const float*)c.host[o].data();
            bool finite = true;
            for (size_t i = 0; i < got.size(); i++) finite &= std::isfinite(got[i]) && std::isfinite(ref[i]);
            printf("  %-34s finite -> %s\n", name, finite ? "PASS" : "FAIL");
            if (!finite) g_fail = 1;
        }
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
    {
        /* ACT_LOG (20) over [1e-10, 10]. */
        Case c; const unsigned rows = 97, width = 40, stride = 50, col0 = 7;
        unsigned o = c.f32(rows * stride, 1.f), x = c.f32(rows * stride, 5.f, 5.f);
        float* xs = (float*)c.host[x].data();
        for (unsigned i = 0; i < rows * stride; i++) xs[i] = fabsf(xs[i]) + (i % 13 == 0 ? 1e-10f : 1e-3f);
        c.in.op = PLOW_DOP_UNARY_F32; c.in.t[0] = o; c.in.t[1] = x;
        c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = 20; c.in.i[3] = stride; c.in.i[4] = col0;
        run("UnaryF32 log", c, g_unary_f32, {o}, FP32, 2e-6);
    }
}

/* ReduceRowsF32: every op over ceil-mode segments with lengths (short last segment, empty and
 * single-row segments) and a column window; then global fbank-sized means. f64 in row order on
 * both sides: bit-exact expected. */
static void t_reduce_rows() {
    static const char* names[5] = {"sum", "mean", "max", "std", "mean_std"};
    for (unsigned op = 0; op < 5; op++) {
        Case c; const unsigned items = 4, rows = 250, width = 37, seg = 100, stride = 50, col0 = 9;
        const unsigned segs = 3, ow = op == 4 ? 2 * width : width;
        unsigned o = c.out((size_t)items * segs * ow), x = c.f32((size_t)items * rows * stride, 3.f, 0.5f);
        unsigned l = c.u32({250, 173, 201, 1000});
        c.in.op = PLOW_DOP_REDUCE_ROWS_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = l;
        c.in.i[0] = items; c.in.i[1] = rows; c.in.i[2] = width; c.in.i[3] = op; c.in.i[4] = seg;
        c.in.i[5] = stride; c.in.i[6] = col0;
        char name[64];
        snprintf(name, sizeof(name), "ReduceRowsF32 seg100 %s", names[op]);
        run(name, c, g_reduce_rows_f32, {o}, FP32, 0.0);
    }
    for (unsigned op : {1u, 4u}) {
        Case c; const unsigned items = 2, rows = 1000, width = 80;
        unsigned o = c.out((size_t)items * (op == 4 ? 2 : 1) * width), x = c.f32((size_t)items * rows * width, 4.f, -6.f);
        unsigned l = c.u32({1000, 637});
        c.in.op = PLOW_DOP_REDUCE_ROWS_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = l;
        c.in.i[0] = items; c.in.i[1] = rows; c.in.i[2] = width; c.in.i[3] = op;
        run(op == 1 ? "ReduceRowsF32 global mean 2x1000x80" : "ReduceRowsF32 global mean_std", c, g_reduce_rows_f32,
            {o}, FP32, 0.0);
    }
    {
        Case c; const unsigned items = 3, rows = 64, width = 512;
        unsigned o = c.out((size_t)items * width), x = c.f32((size_t)items * rows * width, 2.f);
        c.in.op = PLOW_DOP_REDUCE_ROWS_F32; c.in.t[0] = o; c.in.t[1] = x;
        c.in.i[0] = items; c.in.i[1] = rows; c.in.i[2] = width; c.in.i[3] = 2;
        run("ReduceRowsF32 max no lengths", c, g_reduce_rows_f32, {o}, FP32, 0.0);
    }
}

/* LengthAffineU32: floor division of negative values, clamps, the elementwise bound; div 0 writes
 * nothing (both sides keep the fill). */
static void t_length_affine() {
    struct A { const char* name; unsigned mul; int add; unsigned div, lo, hi; bool bound; };
    for (const A& a : {A{"LengthAffineU32 VE windows bound", 1, -51, 77, 1, 0xFFFFFFFFu, true},
                       A{"LengthAffineU32 fbank frames clamp", 1, -400, 160, 0, 50, false},
                       A{"LengthAffineU32 x2+1 /4", 2, 1, 4, 0, 0xFFFFFFFFu, true},
                       A{"LengthAffineU32 div 0 no write", 1, 0, 0, 0, 0xFFFFFFFFu, false}}) {
        Case c; const unsigned items = 300;
        std::vector<uint32_t> in(items), bound(items);
        for (unsigned i = 0; i < items; i++) {
            in[i] = i < 200 ? i * 3 : (rng_s = rng_s * 1664525u + 1013904223u);
            bound[i] = (i * 7919u) % 600u;
        }
        unsigned o = c.out(items), x = c.u32(in), m = a.bound ? c.u32(bound) : PLOW_TENSOR_NONE;
        c.in.op = PLOW_DOP_LENGTH_AFFINE_U32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = m;
        c.in.i[0] = items; c.in.i[1] = a.mul; c.in.i[2] = (unsigned)a.add; c.in.i[3] = a.div;
        c.in.i[4] = a.lo; c.in.i[5] = a.hi;
        run(a.name, c, g_length_affine_u32, {o}, INT);
    }
}

/* LstmCellF32 rows: c in place, h rows into a time-major [x | h] buffer at a column offset. */
static void t_lstm_rows() {
    Case c; const unsigned width = 256, rows = 25, zw = 296, t = 3;
    const size_t zn = (size_t)(t + 2) * rows * zw;
    unsigned z = c.f32(zn, 1.f), cs = c.f32((size_t)rows * width, 2.f), g = c.f32((size_t)rows * 4 * width, 4.f);
    c.in.op = PLOW_DOP_LSTM_CELL_F32; c.in.t[0] = z; c.in.t[1] = cs; c.in.t[2] = g; c.in.t[3] = cs;
    c.in.i[0] = width; c.in.i[1] = rows; c.in.i[2] = zw; c.in.i[3] = (t + 1) * rows * zw + 40;
    run("LstmCellF32 25 rows h strided", c, g_lstm_cell_f32, {z, cs}, FP32);
}

/* AttentionF32 head_width 256: separate query rows and a fused [K | V] buffer under one stride. */
static void t_attention_w256(const char* name, unsigned batch, unsigned q_rows, unsigned kv_rows, unsigned heads,
                             std::vector<uint32_t> lengths, bool causal, bool bias, unsigned pad) {
    Case c;
    const unsigned hw = 256, width = heads * hw, stride = 2 * width + pad;
    unsigned o = c.out((size_t)batch * q_rows * width);
    unsigned q = c.f32((size_t)batch * q_rows * stride, 1.f), kv = c.f32((size_t)batch * kv_rows * stride, 1.f);
    unsigned l = lengths.empty() ? PLOW_TENSOR_NONE : c.u32(lengths);
    unsigned bs = bias ? c.f32((size_t)heads * q_rows * kv_rows, 1.f) : PLOW_TENSOR_NONE;
    c.in.op = PLOW_DOP_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = kv; c.in.t[3] = kv; c.in.t[4] = l; c.in.t[5] = bs;
    c.in.i[0] = batch; c.in.i[1] = q_rows; c.in.i[2] = kv_rows; c.in.i[3] = heads; c.in.i[4] = hw;
    c.in.i[5] = stride; c.in.i[6] = causal; c.in.i[7] = q_rows * kv_rows;
    c.in.fj[0].f = 1.0f / 16.0f;
    c.in.fj[1].u = pad; c.in.fj[2].u = width + pad;
    run(name, c, g_attention_f32, {o}, FP32, 2e-5);
}

/* TrimSilenceF32 against librosa 0.11 effects.trim(frame 2048, hop 512) indices, computed in
 * Python over these exact signals (xorshift-free LCG noise times piecewise amplitudes). */
struct TrimSeg { unsigned end; float amp; };
struct TrimVec { unsigned seed, len_in; std::vector<TrimSeg> segs; unsigned s20, e20, s60, e60; };
static std::vector<TrimVec> trim_vectors() {
    std::vector<TrimSeg> f;
    for (unsigned k = 0; k < 16; k++) f.push_back({1500u * (k + 1), ldexpf(1.f, -(int)k)});
    f.push_back({~0u, 0.f});
    return {
        {1, 20000, {{5000, 0.f}, {14000, 0.5f}, {~0u, 0.001f}}, 4096, 15360, 4096, 20000},
        {2, 7000, {{~0u, 0.f}}, 0, 7000, 0, 7000},
        {3, 12345, {{~0u, 0.25f}}, 0, 12345, 0, 12345},
        {4, 300, {{~0u, 0.5f}}, 0, 300, 0, 300},
        {5, 30000, {{3000, 3e-3f}, {9000, 0.5f}, {~0u, 0.f}}, 2048, 10240, 0, 10240},
        {6, 24000, f, 0, 6656, 0, 16384},
        {7, 22000, {{2000, 0.f}, {4000, 0.3f}, {15000, 0.f}, {16000, 0.3f}, {~0u, 0.f}}, 1024, 17408, 1024, 17408},
    };
}
static void trim_signal(const TrimVec& v, float* x, unsigned cap) {
    uint32_t st = v.seed;
    unsigned si = 0;
    const unsigned len = v.len_in < cap ? v.len_in : cap;
    for (unsigned i = 0; i < cap; i++) {
        st = st * 1664525u + 1013904223u;
        const float u = (float)((int)(st >> 8) - 8388608) / 8388608.0f;
        while (i >= v.segs[si].end) si++;
        x[i] = i < len ? u * v.segs[si].amp : 0.f;
    }
}
static void t_trim_silence() {
    const std::vector<TrimVec> vs = trim_vectors();
    const unsigned cap = 24000, items = (unsigned)vs.size();
    for (float top_db : {20.f, 60.f, 0.f}) {
        Case c;
        unsigned o = c.out((size_t)items * cap), x = c.add((size_t)items * cap * 4), lo = c.out(items);
        std::vector<uint32_t> li;
        for (unsigned b = 0; b < items; b++) {
            trim_signal(vs[b], (float*)c.host[x].data() + (size_t)b * cap, cap);
            li.push_back(vs[b].len_in);
        }
        unsigned l = c.u32(li);
        c.in.op = PLOW_DOP_TRIM_SILENCE_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = l; c.in.t[3] = lo;
        c.in.i[0] = items; c.in.i[1] = cap; c.in.i[2] = 2048; c.in.i[3] = 512; c.in.fj[0].f = top_db;
        char name[64];
        snprintf(name, sizeof(name), "TrimSilenceF32 top_db %g", top_db);
        run(name, c, g_trim_silence_f32, {o, lo}, INT);
        /* The golden's result (host buffers) against librosa's [start, end). */
        const float* xs = (const float*)c.host[x].data();
        const float* os = (const float*)c.host[o].data();
        const uint32_t* ls = (const uint32_t*)c.host[lo].data();
        bool ok = true;
        for (unsigned b = 0; b < items; b++) {
            const unsigned s0 = top_db == 0.f ? 0 : top_db == 20.f ? vs[b].s20 : vs[b].s60;
            const unsigned e0 = top_db == 0.f ? 0 : top_db == 20.f ? vs[b].e20 : vs[b].e60;
            if (ls[b] != e0 - s0) ok = false;
            for (unsigned i = 0; i < cap && ok; i++)
                if (os[(size_t)b * cap + i] != (i < e0 - s0 ? xs[(size_t)b * cap + s0 + i] : 0.f)) ok = false;
        }
        printf("  %-34s librosa indices -> %s\n", name, ok ? "PASS" : "FAIL");
        if (!ok) g_fail = 1;
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
        {"BinaryF32 add full round", 0, rows * width, width, 1, items * rows * width, 2},
        {"BinaryF32 sub row-vector scaled round", 1, 0, 0, 1, width, 3},
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

static void t_vec_elementwise() {
    {
        Case c; const unsigned rows = 97, width = 40, stride = 48, col0 = 8;
        unsigned o = c.f32(rows * stride, 1.f), x = c.f32(rows * stride, 4.f), p = c.f32(width, 0.5f, 0.7f);
        c.in.op = PLOW_DOP_UNARY_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = p;
        c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = 12; c.in.i[3] = stride; c.in.i[4] = col0;
        run("UnaryF32 snake float4", c, g_unary_f32, {o}, FP32, 1e-5);
    }
    /* SnakeBeta (19), param [a | r]: the decoder widths (quad path, 8 quads per thread) and an
     * unaligned window (scalar path); 1500 x 768 runs every thread's batches more than once at
     * 132 blocks (288000 quads > 33792). */
    for (unsigned wsel : {96u, 192u, 384u, 768u, 40u, 1768u}) {
        unsigned width = wsel;
        Case c; const unsigned rows = width == 40 ? 97 : width == 1768 ? 1500 : 61, stride = width == 40 ? 50 : width == 1768 ? 768 : width, col0 = width == 40 ? 7 : 0;
        if (width == 1768) width = 768;
        unsigned o = c.f32(rows * stride, 1.f), x = c.f32(rows * stride, 4.f), p = c.f32(2 * width, 0.5f, 0.7f);
        c.in.op = PLOW_DOP_UNARY_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = p;
        c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = 19; c.in.i[3] = stride; c.in.i[4] = col0;
        char name[64];
        snprintf(name, sizeof(name), "UnaryF32 snake_beta %s w%u", width == 40 ? "scalar" : "float4", width);
        run(name, c, g_unary_f32, {o}, FP32, 1e-5);
    }
    {
        /* The SnakeBeta quad and scalar paths over the same values (rows of 96 at stride 96 vs
         * copied into stride 98): the same bits. */
        const unsigned rows = 300, width = 96, s2 = 98;
        Case a;
        unsigned oa = a.out(rows * width), xa = a.f32(rows * width, 4.f), pa = a.f32(2 * width, 0.5f, 0.7f);
        a.in.op = PLOW_DOP_UNARY_F32; a.in.t[0] = oa; a.in.t[1] = xa; a.in.t[2] = pa;
        a.in.i[0] = rows; a.in.i[1] = width; a.in.i[2] = 19; a.in.i[3] = width; a.in.i[4] = 0;
        run("UnaryF32 snake_beta quad path", a, g_unary_f32, {oa}, FP32, 1e-5);
        Case b;
        unsigned ob = b.out(rows * s2), xb = b.f32(rows * s2, 4.f), pb = b.f32(2 * width, 0.5f, 0.7f);
        b.host[pb] = a.host[pa];
        for (unsigned r = 0; r < rows; r++)
            memcpy(&b.host[xb][(size_t)r * s2 * 4], &a.host[xa][(size_t)r * width * 4], width * 4);
        b.in.op = PLOW_DOP_UNARY_F32; b.in.t[0] = ob; b.in.t[1] = xb; b.in.t[2] = pb;
        b.in.i[0] = rows; b.in.i[1] = width; b.in.i[2] = 19; b.in.i[3] = s2; b.in.i[4] = 0;
        run("UnaryF32 snake_beta scalar path", b, g_unary_f32, {ob}, FP32, 1e-5);
        std::vector<float> ya(rows * width), yb(rows * s2);
        CK(cudaMemcpy(ya.data(), a.dev[oa], ya.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yb.data(), b.dev[ob], yb.size() * 4, cudaMemcpyDeviceToHost));
        bool same = true;
        for (unsigned r = 0; r < rows && same; r++) same = !memcmp(&ya[(size_t)r * width], &yb[(size_t)r * s2], width * 4);
        printf("  %-34s quad bitwise == scalar -> %s\n", "UnaryF32 snake_beta paths", same ? "PASS" : "FAIL");
        if (!same) g_fail = 1;
    }
    struct B { const char* name; unsigned op, bis, brs, bcs, bn, flags; };
    const unsigned items = 3, rows = 45, width = 36;
    const B cases[] = {
        {"BinaryF32 add full float4", 0, rows * width, width, 1, items * rows * width, 0},
        {"BinaryF32 mul row-scalar float4", 2, rows, 1, 0, items * rows, 1},
        {"BinaryF32 sub item-scalar float4", 1, 1, 0, 0, items, 0},
        {"BinaryF32 add full float4 round", 0, rows * width, width, 1, items * rows * width, 2},
        {"BinaryF32 mul row-scalar float4 scaled round", 2, rows, 1, 0, items * rows, 3},
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

/* RowStatsF32 vs golden; then LayerNormF32 -> DenseGemmF32 against the fused prologue
 * (RowStatsF32 -> DenseGemmF32 flag 32), both on the GPU: bit-identical outputs expected. */
static void t_row_stats(const char* name, unsigned rows, unsigned feat, unsigned flags) {
    Case c;
    unsigned o = c.out((size_t)rows * 2), x = c.f32((size_t)rows * feat, 2.f, 0.5f);
    c.in.op = PLOW_DOP_ROW_STATS_F32; c.in.t[0] = o; c.in.t[1] = x;
    c.in.i[0] = rows; c.in.i[1] = feat; c.in.i[2] = flags; c.in.fj[0].f = 1e-5f;
    run(name, c, g_row_stats_f32, {o}, FP32, 0.0);
}
static void launch(const PlowDevInst& in, const std::vector<void*>& ptrs) {
    void** td = upload_table(ptrs);
    k_speech<<<g_nblk, 256, kSmem>>>(in, td);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaFree(td));
}
static void t_ln_gemm(const char* name, unsigned m, unsigned n, unsigned k, unsigned ln_flags, unsigned dense_flags,
                      unsigned row0) {
    /* Golden fused vs GPU fused. */
    Case c;
    const unsigned rows = m + row0;
    unsigned o = c.out((size_t)m * n), x = c.f32((size_t)rows * k, 2.f, 0.3f);
    unsigned w = dense_flags & 4u ? c.bf16((size_t)n * k, 0.06f) : c.f32((size_t)n * k, 0.06f);
    unsigned b = c.f32(n, 0.5f), sc = splitk_scratch(c), st = c.out((size_t)rows * 2);
    unsigned g = c.f32(k, 0.5f, 1.f), be = c.f32(k, 0.2f);
    c.upload();
    std::vector<float> stats((size_t)rows * 2);
    {   /* stats on the GPU (RowStatsF32), mirrored to the host for the golden */
        PlowDevInst in; memset(&in, 0, sizeof(in)); for (auto& t : in.t) t = PLOW_TENSOR_NONE;
        in.op = PLOW_DOP_ROW_STATS_F32; in.t[0] = st; in.t[1] = x;
        in.i[0] = rows; in.i[1] = k; in.i[2] = ln_flags; in.fj[0].f = 1e-5f;
        launch(in, c.dev);
        CK(cudaMemcpy(c.host[st].data(), c.dev[st], rows * 8, cudaMemcpyDeviceToHost));
        memcpy(stats.data(), c.host[st].data(), rows * 8);
        std::vector<float> ref(rows * 2);
        std::vector<void*> th; for (auto& h : c.host) th.push_back(h.data());
        std::vector<uint8_t> save = c.host[st];
        g_row_stats_f32(&in, 0, 1, th.data(), nullptr);
        if (memcmp(save.data(), c.host[st].data(), rows * 8)) { printf("  %s: RowStats mismatch\n", name); g_fail = 1; }
    }
    /* Unfused GPU chain: LayerNormF32 -> DenseGemmF32 (flags without the prologue bits). */
    float* ln_out; float* chain;
    CK(cudaMalloc(&ln_out, (size_t)rows * k * 4)); CK(cudaMalloc(&chain, (size_t)m * n * 4));
    {
        std::vector<void*> ptrs = c.dev; ptrs.push_back(ln_out); ptrs.push_back(chain);
        const unsigned lo = (unsigned)ptrs.size() - 2, ch = lo + 1;
        PlowDevInst in; memset(&in, 0, sizeof(in)); for (auto& t : in.t) t = PLOW_TENSOR_NONE;
        in.op = PLOW_DOP_LAYERNORM_F32; in.t[0] = lo; in.t[1] = x; in.t[2] = g; in.t[3] = be;
        in.i[0] = rows; in.i[1] = k; in.i[2] = ln_flags; in.fj[0].f = 1e-5f;
        launch(in, ptrs);
        memset(&in, 0, sizeof(in)); for (auto& t : in.t) t = PLOW_TENSOR_NONE;
        in.op = PLOW_DOP_DENSE_GEMM_F32; in.t[0] = ch; in.t[1] = lo; in.t[2] = w; in.t[3] = b; in.t[4] = sc;
        in.i[0] = m; in.i[1] = n; in.i[2] = k; in.i[4] = row0; in.i[7] = dense_flags;
        launch(in, ptrs);
    }
    c.in.op = PLOW_DOP_DENSE_GEMM_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b; c.in.t[4] = sc; c.in.t[5] = st; c.in.t[6] = g;
    c.in.t[7] = be;
    c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k; c.in.i[4] = row0;
    c.in.i[7] = dense_flags | 32u | (ln_flags & 1u ? 64u : 0u);
    /* run() re-uploads host buffers: keep the stats the GPU produced. */
    memcpy(c.host[st].data(), stats.data(), rows * 8);
    for (void* p : c.dev) cudaFree(p);
    c.dev.clear();
    run(name, c, g_dense_gemm_f32, {o}, dense_flags & 3u ? BF16 : FP32, 1e-5, dense_flags & 2u ? (size_t)m * n / 20000 : 0);
    std::vector<uint32_t> fused((size_t)m * n), unfused((size_t)m * n);
    CK(cudaMemcpy(fused.data(), c.dev[o], fused.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(unfused.data(), chain, unfused.size() * 4, cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < fused.size(); i++) diff += fused[i] != unfused[i];
    printf("  %-34s     n=%-8zu fused vs LayerNorm->Dense on GPU: %s\n", name, fused.size(), diff ? "DIFFER" : "bit-identical");
    if (diff) { printf("    %zu differing elements\n", diff); g_fail = 1; }
    cudaFree(ln_out); cudaFree(chain);
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
                        std::vector<uint32_t> lengths, bool causal, bool bias, bool fused, bool tc = false) {
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
    c.in.i[5] = fused ? stride : 0; c.in.i[6] = causal | (tc ? 2u : 0u); c.in.i[7] = rows * rows;
    c.in.fj[0].f = 1.0f / sqrtf((float)hw);
    c.in.fj[1].u = fused ? width : 0; c.in.fj[2].u = fused ? 2 * width : 0;
    run(name, c, g_attention_f32, {o}, FP32, 2e-5);
}

/* Stream mode: fused QKV item rows [hist carried | n new]; positions mask absent history, and the
 * TC flag must fall back to the fp32 kernel. */
static void t_ring_kv() {
    const unsigned L = 249, slots = 512;
    for (unsigned B : {7u, 33u}) {
        std::vector<uint32_t> vs(B), vn(B), vp(B);
        const uint32_t positions[] = {0, 1, 248, 249, 250, 10000, 256};
        for (unsigned N : {1u, 2u, 3u, 4u}) {
            Case c;
            const unsigned H = 2, width = H * 64u, stride = 3 * width;
            unsigned out = c.out((size_t)B * N * width);
            unsigned qkv = c.f32((size_t)B * N * stride, .2f);
            unsigned state = c.f32((size_t)slots * L * 2 * width, .2f);
            for (unsigned b = 0; b < B; b++) {
                vn[b] = b % 7 == 3 ? 0 : b % 7 == 2 ? N - 1 : N;
                vs[b] = vn[b] ? (b * 13 + 5) % slots : slots + 99;
                vp[b] = vn[b] ? positions[b % 7] : UINT32_MAX;
            }
            unsigned slot = c.u32(vs), n = c.u32(vn), pos = c.u32(vp);
            c.in.op = PLOW_DOP_ATTENTION_F32;
            const unsigned ts[] = {out, qkv, state, slot, n, pos};
            for (unsigned k = 0; k < 6; k++) c.in.t[k] = ts[k];
            const unsigned is[] = {B, N, L, H, 64, stride, 0x109u | (250u << 16), slots};
            for (unsigned k = 0; k < 8; k++) c.in.i[k] = is[k];
            c.in.fj[0].f = .125f;
            c.in.fj[1].u = width; c.in.fj[2].u = 2 * width;
            char name[80]; snprintf(name, sizeof(name), "ring attention B%u N%u sparse/zero/wrap", B, N);
            run(name, c, g_attention_f32, {out}, FP32);
        }
        for (unsigned N : {1u, 2u, 17u, 34u, 300u}) {
            for (unsigned mode : {2u, 3u}) {
                Case c;
                const unsigned C = 128, Cz = 192, zc0 = 64;
                unsigned z = mode == 3 ? c.out((size_t)B * (L + N) * Cz) : PLOW_TENSOR_NONE;
                unsigned src = c.f32((size_t)B * N * Cz, .2f);
                unsigned state = c.f32((size_t)slots * L * C, .2f);
                for (unsigned b = 0; b < B; b++) {
                    vn[b] = b % 7 == 3 ? 0 : b % 7 == 2 ? N / 2 : N;
                    vs[b] = vn[b] ? (b * 13 + 5) % slots : slots + 99;
                    vp[b] = vn[b] ? positions[b % 7] : UINT32_MAX;
                }
                unsigned slot = c.u32(vs), n = c.u32(vn), pos = c.u32(vp);
                st_inst(c.in, z, src, state, slot, n, pos, B, slots, L, N, C, Cz, zc0, Cz, mode);
                char name[80]; snprintf(name, sizeof(name), "ring tail B%u N%u mode%u sparse/zero/wrap", B, N, mode);
                run(name, c, g_stream_tail_f32, {mode == 3 ? z : state}, INT);
            }
        }
    }
}

static void t_attention_stream(const char* name, unsigned batch, unsigned hist, unsigned n, unsigned win,
                               std::vector<uint32_t> pos, std::vector<uint32_t> lengths, bool tc) {
    Case c;
    const unsigned heads = 4, hw = 64, width = heads * hw, stride = 3 * width, rows = hist + n;
    unsigned o = c.out((size_t)batch * n * width);
    unsigned qkv = c.f32((size_t)batch * rows * stride, 1.f);
    unsigned ps = c.u32(pos);
    unsigned l = lengths.empty() ? PLOW_TENSOR_NONE : c.u32(lengths);
    c.in.op = PLOW_DOP_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = qkv; c.in.t[2] = qkv; c.in.t[3] = qkv; c.in.t[4] = l; c.in.t[5] = PLOW_TENSOR_NONE;
    c.in.t[6] = PLOW_TENSOR_NONE; c.in.t[7] = ps;
    c.in.i[0] = batch; c.in.i[1] = n; c.in.i[2] = rows; c.in.i[3] = heads; c.in.i[4] = hw;
    c.in.i[5] = stride; c.in.i[6] = 1u | 8u | (tc ? 2u : 0u) | (win << 16); c.in.i[7] = hist;
    c.in.fj[0].f = 0.125f;
    c.in.fj[1].u = width; c.in.fj[2].u = 2 * width;
    run(name, c, g_attention_f32, {o}, FP32, 2e-5);
}

/* Keys = [prefix block index[b] (pre rows, [K | V] rows) | own rows of a fused QKV tensor]. */
static void t_attention_prefix(const char* name, unsigned batch, unsigned rows, unsigned pre, unsigned blocks,
                               std::vector<uint32_t> index, std::vector<uint32_t> lengths, bool tc) {
    Case c;
    const unsigned heads = 8, hw = 64, width = heads * hw, stride = 3 * width;
    unsigned o = c.out((size_t)batch * rows * width);
    unsigned qkv = c.f32((size_t)batch * rows * stride, 1.f);
    unsigned pf = c.f32((size_t)blocks * pre * 2 * width, 1.f);
    unsigned ix = c.u32(index);
    unsigned l = lengths.empty() ? PLOW_TENSOR_NONE : c.u32(lengths);
    c.in.op = PLOW_DOP_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = qkv; c.in.t[2] = qkv; c.in.t[3] = qkv; c.in.t[4] = l; c.in.t[5] = PLOW_TENSOR_NONE;
    c.in.t[6] = pf; c.in.t[7] = ix;
    c.in.i[0] = batch; c.in.i[1] = rows; c.in.i[2] = rows; c.in.i[3] = heads; c.in.i[4] = hw;
    c.in.i[5] = stride; c.in.i[6] = tc ? 2u : 0u; c.in.i[7] = pre;
    c.in.fj[0].f = 0.125f;
    c.in.fj[1].u = width; c.in.fj[2].u = 2 * width;
    run(name, c, g_attention_f32, {o}, FP32, 2e-5);
}

#if defined(PLOW_NV_HOPPER)
/* t_attention_prefix with the prefix formatted on the GPU (CopyColsF32 format mode) and read by
 * sp_attention_h16 (flags bit 4), against the golden on the f32 prefix. */
static void t_attention_prefix_h16(const char* name, unsigned batch, unsigned rows, unsigned pre, unsigned blocks,
                                   std::vector<uint32_t> index, std::vector<uint32_t> lengths) {
    Case c;
    const unsigned heads = 8, hw = 64, width = heads * hw, stride = 3 * width, npt = (pre + SPH_KT - 1) / SPH_KT;
    unsigned o = c.out((size_t)batch * rows * width);
    unsigned qkv = c.f32((size_t)batch * rows * stride, 1.f);
    unsigned pf = c.f32((size_t)blocks * pre * 2 * width, 1.f);
    unsigned fm = c.out((size_t)blocks * heads * npt * SPH_XS);
    unsigned ix = c.u32(index);
    unsigned l = lengths.empty() ? PLOW_TENSOR_NONE : c.u32(lengths);
    PlowDevInst f;
    memset(&f, 0, sizeof(f));
    for (auto& x : f.t) x = PLOW_TENSOR_NONE;
    f.op = PLOW_DOP_COPY_COLS_F32;
    f.t[0] = fm; f.t[1] = pf;
    f.i[0] = blocks; f.i[1] = pre; f.i[2] = 2 * width; f.i[3] = 2 * width; f.i[7] = 1;
    f.fj[1].u = pre * 2 * width; f.fj[2].u = heads * npt * SPH_XS;
    c.in.op = PLOW_DOP_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = qkv; c.in.t[2] = qkv; c.in.t[3] = qkv; c.in.t[4] = l; c.in.t[5] = PLOW_TENSOR_NONE;
    c.in.t[6] = pf; c.in.t[7] = ix;
    c.in.i[0] = batch; c.in.i[1] = rows; c.in.i[2] = rows; c.in.i[3] = heads; c.in.i[4] = hw;
    c.in.i[5] = stride; c.in.i[6] = 2u; c.in.i[7] = pre;
    c.in.fj[0].f = 0.125f;
    c.in.fj[1].u = width; c.in.fj[2].u = 2 * width;
    const PlowDevInst ref = c.in;
    c.in.i[6] = 2u | 16u;
    c.in.t[6] = fm;
    run(name, c, g_attention_f32, {o}, FP32, 2e-5, 0, &f, &ref);
}
#endif

/* Time-major encoder attention: batch 1, items * hpi heads, bias a relative-position table
 * [t][heads][2t - 1], key lengths per item (group of hpi heads). */
static void t_attention_rel(const char* name, unsigned items, unsigned t, unsigned hpi, std::vector<uint32_t> lengths) {
    Case c;
    const unsigned heads = items * hpi, hw = 64, width = heads * hw, m = 2 * t - 1;
    unsigned o = c.out((size_t)t * width);
    unsigned q = c.f32((size_t)t * width, 1.f), k = c.f32((size_t)t * width, 1.f), v = c.f32((size_t)t * width, 1.f);
    unsigned bs = c.f32((size_t)t * heads * m, 1.f), l = c.u32(lengths);
    c.in.op = PLOW_DOP_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = l; c.in.t[5] = bs;
    c.in.i[0] = 1; c.in.i[1] = t; c.in.i[2] = t; c.in.i[3] = heads; c.in.i[4] = hw;
    c.in.i[5] = 0; c.in.i[6] = 2u | 4u | hpi << 8; c.in.i[7] = m;
    c.in.fj[0].f = 0.125f;
    run(name, c, g_attention_f32, {o}, FP32, 2e-5);
}

static void bench(const char* name, Case& c, double flops, double bytes, int reps = 20,
                  const PlowDevInst* pre = nullptr) {
    c.upload();
    void** td = upload_table(c.dev);
    if (pre) k_speech<<<g_nblk, 256, kSmem>>>(*pre, td);
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

static void t_attention_short(unsigned batch, unsigned qr, unsigned kr, bool causal, bool fused, bool timing) {
    Case c;
    constexpr unsigned heads = 16, width = heads * 64;
    const unsigned stride = fused ? 3 * width : width;
    const unsigned out = c.out((size_t)batch * qr * width + 16);
    unsigned q, k, v;
    if (fused && qr == kr) q = k = v = c.f32((size_t)batch * kr * stride, 1.f);
    else {
        q = c.f32((size_t)batch * qr * stride, 1.f);
        k = v = c.f32((size_t)batch * kr * stride, 1.f);
    }
    std::vector<uint32_t> lengths(batch);
    for (unsigned b = 0; b < batch; b++) lengths[b] = timing ? kr : b % 4 == 0 ? 0 : b % 4 == 1 ? 1 : kr + b % 2;
    c.in.op = PLOW_DOP_ATTENTION_F32;
    c.in.t[0] = out; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = c.u32(lengths);
    c.in.i[0] = batch; c.in.i[1] = qr; c.in.i[2] = kr; c.in.i[3] = heads; c.in.i[4] = 64;
    c.in.i[5] = stride; c.in.i[6] = causal;
    c.in.fj[0].f = .125f; c.in.fj[1].u = fused ? width : 0; c.in.fj[2].u = fused ? 2 * width : 0;
    char name[160];
    snprintf(name, sizeof(name), "AttentionF32 short b%u q%u k%u causal%u fused%u", batch, qr, kr, causal, fused);
    if (!timing) {
        run(name, c, g_attention_f32, {out}, FP32, 2e-5);
        return;
    }
    for (unsigned arm : {2u, 0u, 2u, 0u}) {
        c.in.i[6] = causal | arm;
        char label[192];
        snprintf(label, sizeof(label), "%s %s", name, arm ? "tc" : "scalar");
        bench(label, c, 4.0 * batch * qr * kr * width, 0, 50);
    }
}

static void bench_signal() {
    {
        Case c; const unsigned rows = 32 * 16384, width = 64;
        unsigned o = c.out((size_t)rows * width), x = c.f32((size_t)rows * width, 4.f);
        unsigned p = c.f32(width, 0.5f, 0.7f);
        c.in.op = PLOW_DOP_UNARY_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = p;
        c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = 12;
        bench("UnaryF32 snake [32*16384][64]", c, 0, 8.0 * rows * width, 5);
    }
    {
        Case c; const unsigned rows = 32 * 16384, width = 64;
        unsigned o = c.out((size_t)rows * width), a = c.f32((size_t)rows * width, 1.f), b = c.f32((size_t)rows * width, 1.f);
        c.in.op = PLOW_DOP_BINARY_F32; c.in.t[0] = o; c.in.t[1] = a; c.in.t[2] = b;
        c.in.i[0] = 1; c.in.i[1] = rows; c.in.i[2] = width; c.in.i[3] = 0; c.in.i[4] = rows * width;
        c.in.i[5] = width; c.in.i[6] = 1;
        bench("BinaryF32 add full [32*16384][64]", c, 0, 12.0 * rows * width, 5);
    }
    {
        Case c; const unsigned rows = 32 * 16384, width = 64;
        unsigned o = c.out((size_t)rows * width), a = c.f32((size_t)rows * width, 1.f), b = c.f32(rows, 1.f);
        c.in.op = PLOW_DOP_BINARY_F32; c.in.t[0] = o; c.in.t[1] = a; c.in.t[2] = b;
        c.in.i[0] = 1; c.in.i[1] = rows; c.in.i[2] = width; c.in.i[3] = 2; c.in.i[4] = rows; c.in.i[5] = 1;
        c.in.i[6] = 0;
        bench("BinaryF32 mul row-scalar [32*16384][64]", c, 0, 8.0 * rows * width, 5);
    }
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
        ConvSpec s{64, 256, 512, 256, 16, 8, 0, 1, 4, 4, 0, 12, 0, false, true, true, false, {}, 0.f, true};
        for (bool pm : {false, true}) {
            s.tapmajor = pm;
            Case c; unsigned o; build_conv(c, s, true, o);
            const double out_rows = conv_out_rows(s, true);
            bench(pm ? "ConvTranspose1dF32 phase-major 512->256 k16 s8 [64][256]" : "ConvTranspose1dF32 512->256 k16 s8 [64][256]",
                  c, 2.0 * 64 * out_rows * 256 * 512 * 2, 0);
        }
    }
    for (bool tc : {false, true}) {
        ConvSpec s{2, 256, 1024, 512, 16, 8, 0, 1, 4, 4, 0, 12, 0, false, true, true, false, {}, 0.f, tc};
        Case c; unsigned o; build_conv(c, s, true, o);
        const double out_rows = conv_out_rows(s, true);
        bench(tc ? "ConvTranspose1dF32 tc 1024->512 k16 s8 [2][256]" : "ConvTranspose1dF32 1024->512 k16 s8 [2][256]",
              c, 2.0 * 2 * out_rows * 512 * 1024 * 2, 0);
    }
    for (bool tc : {false, true}) {
        ConvSpec s{1, 32, 1024, 512, 16, 8, 0, 1, 4, 4, 0, 12, 0, false, true, true, false, {32}, 0.f, tc};
        Case c; unsigned o; build_conv(c, s, true, o);
        const double out_rows = conv_out_rows(s, true);
        bench(tc ? "ConvTranspose1dF32 tc 1024->512 k16 s8 [1][32]" : "ConvTranspose1dF32 1024->512 k16 s8 [1][32]",
              c, 2.0 * out_rows * 512 * 1024 * 2, 0);
    }
    for (bool tc : {false, true}) {
        ConvSpec s{32, 2048, 256, 256, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, true, {}, 0.f, tc};
        Case c; unsigned o; build_conv(c, s, false, o);
        bench(tc ? "Conv1dF32 tc pointwise 256->256 +res [32][2048]" : "Conv1dF32 pointwise 256->256 +res [32][2048]",
              c, 2.0 * 32 * 2048 * 256 * 256, 0);
    }
    for (bool tc : {false, true}) {
        ConvSpec s{2, 8192, 512, 512, 1, 1, 1, 1, 0, 0, 0, 12, 0, false, true, true, true, {}, 0.f, tc};
        Case c; unsigned o; build_conv(c, s, false, o);
        bench(tc ? "Conv1dF32 tc pointwise +snake +residual 16k" : "Conv1dF32 pointwise +snake +residual 16k (2)",
              c, 2.0 * 16384 * 512 * 512, 0);
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

static std::vector<uint32_t> fetch(Case& c, unsigned id) {
    std::vector<uint32_t> v(c.host[id].size() / 4);
    CK(cudaMemcpy(v.data(), c.dev[id], c.host[id].size(), cudaMemcpyDeviceToHost));
    return v;
}
static void verdict(const char* name, const char* what, bool pass) {
    printf("  %-34s %s -> %s\n", name, what, pass ? "PASS" : "FAIL");
    if (!pass) g_fail = 1;
}

/* Q8GemmF32 with a fixed path (i7): against the golden, and each of the first `m_small` rows bit
 * for bit as in a launch of only those rows. */
static void t_q8_path(unsigned path, unsigned m_small, unsigned m_big, unsigned n, unsigned k) {
    char name[64];
    snprintf(name, sizeof name, "Q8GemmF32 path %u %ux%ux%u", path, m_big, n, k);
    Case c;
    unsigned big = c.out((size_t)m_big * n), small = c.out((size_t)m_small * n), last = c.out((size_t)m_small * n);
    unsigned x = c.f32((size_t)m_big * k, 1.f), w = q8_weights(c, n, k), b = c.f32(n, 0.3f);
    c.in.op = PLOW_DOP_Q8_GEMM_F32;
    c.in.t[0] = big; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b;
    c.in.i[0] = m_big; c.in.i[1] = n; c.in.i[2] = k; c.in.i[7] = path;
    run(name, c, g_q8_gemm_f32, {big}, FP32);
    PlowDevInst alone = c.in;
    alone.t[0] = small; alone.i[0] = m_small;
    launch(alone, c.dev);
    /* The launch's last rows too (a_row0): they sit in its last group of rows. */
    PlowDevInst tail = alone;
    tail.t[0] = last; tail.i[4] = m_big - m_small;
    launch(tail, c.dev);
    const std::vector<uint32_t> batched = fetch(c, big), own = fetch(c, small), own_last = fetch(c, last);
    verdict(name, "rows alone == rows in the launch",
            std::equal(own.begin(), own.end(), batched.begin()) &&
                std::equal(own_last.begin(), own_last.end(), batched.end() - own_last.size()));
}

/* Q8GemmF32's GEMV walk (path 1) bit for bit against its order on the host: lane l's fma over blocks
 * of x[r][32b + l] * (scale_b * q_b[l]) from 0, the butterfly over lanes (offsets 16..1), the bias. */
static void t_q8_gemv_order(unsigned m, unsigned n, unsigned k) {
    char name[64];
    snprintf(name, sizeof name, "Q8GemmF32 gemv order %ux%ux%u", m, n, k);
    Case c;
    unsigned o = c.out((size_t)m * n), x = c.f32((size_t)m * k, 1.f), w = q8_weights(c, n, k), b = c.f32(n, 0.3f);
    c.in.op = PLOW_DOP_Q8_GEMM_F32;
    c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b;
    c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k; c.in.i[7] = 1;
    run(name, c, g_q8_gemm_f32, {o}, FP32);
    const std::vector<uint32_t> got = fetch(c, o);
    const float* X = (const float*)c.host[x].data();
    const float* B = (const float*)c.host[b].data();
    size_t bad = 0;
    for (unsigned col = 0; col < n; col++) {
        const uint8_t* wc = c.host[w].data() + (size_t)col * (k / 32) * 34;
        for (unsigned r = 0; r < m; r++) {
            float p[32], q[32];
            for (unsigned l = 0; l < 32; l++) {
                float acc = 0.f;
                for (unsigned blk = 0; blk < k / 32; blk++) {
                    uint16_t hs;
                    memcpy(&hs, wc + blk * 34, 2);
                    const float wv = __half2float(__ushort_as_half(hs)) * (float)(int8_t)wc[blk * 34 + 2 + l];
                    acc = fmaf(X[(size_t)r * k + 32 * blk + l], wv, acc);
                }
                p[l] = acc;
            }
            for (unsigned off = 16; off > 0; off >>= 1) {
                for (unsigned l = 0; l < 32; l++) q[l] = p[l] + p[l ^ off];
                memcpy(p, q, sizeof p);
            }
            const float want = p[0] + B[col];
            uint32_t bits;
            memcpy(&bits, &want, 4);
            bad += got[(size_t)r * n + col] != bits;
        }
    }
    verdict(name, "== its order on the host", bad == 0);
}

/* Q8GemmF32 tiles (path 2): 128x128 tiles (as many slices as tiles) give the 64x64 tiles' bits. */
static void t_q8_tiles(unsigned m, unsigned n, unsigned k) {
    char name[64];
    snprintf(name, sizeof name, "Q8GemmF32 tiles %ux%ux%u", m, n, k);
    Case c;
    unsigned small = c.out((size_t)m * n), large = c.out((size_t)m * n);
    unsigned x = c.f32((size_t)m * k, 1.f), w = q8_weights(c, n, k), b = c.f32(n, 0.3f);
    c.in.op = PLOW_DOP_Q8_GEMM_F32;
    c.in.t[0] = small; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b;
    c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k; c.in.i[7] = 2;
    run(name, c, g_q8_gemm_f32, {small}, FP32);
    PlowDevInst one = c.in;
    one.t[0] = large;
    const int slices = g_nblk;
    g_nblk = (int)(((m + SPG_BM - 1) / SPG_BM) * ((n + SPG_BN - 1) / SPG_BN));
    launch(one, c.dev);
    g_nblk = slices;
    verdict(name, "128x128 tiles == 64x64 tiles", fetch(c, large) == fetch(c, small));
}

/* Item `b` of an [items][bytes] tensor as a tensor of its own. */
static unsigned item_copy(Case& c, unsigned all, unsigned b, size_t bytes) {
    unsigned id = c.add(bytes);
    memcpy(c.host[id].data(), c.host[all].data() + b * bytes, bytes);
    return id;
}

/* RelativeAttentionF32 over items (i6), each with its own key_start: against the golden, and each
 * item bit for bit as in a launch of its own. */
static void t_relative_items(const char* name, unsigned rows, unsigned width, unsigned heads, unsigned chunk,
                             unsigned left, unsigned query_row0, std::vector<uint32_t> starts) {
    Case c;
    const unsigned items = (unsigned)starts.size();
    const size_t n = (size_t)rows * width;
    unsigned o = c.out(items * n), q = c.f32(items * n, 1.f), k = c.f32(items * n, 1.f), v = c.f32(items * n, 1.f);
    unsigned p = c.f32((size_t)(2 * rows - 1) * width, 1.f), u = c.f32(width, 0.3f), bv = c.f32(width, 0.3f);
    unsigned ks = c.u32(starts);
    std::vector<PlowDevInst> alone;
    for (unsigned b = 0; b < items; b++) {
        PlowDevInst one;
        memset(&one, 0, sizeof one);
        for (auto& t : one.t) t = PLOW_TENSOR_NONE;
        one.t[0] = c.out(n); one.t[1] = item_copy(c, q, b, n * 4); one.t[2] = item_copy(c, k, b, n * 4);
        one.t[3] = item_copy(c, v, b, n * 4); one.t[7] = c.u32({starts[b]});
        alone.push_back(one);
    }
    c.in.op = PLOW_DOP_RELATIVE_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = p; c.in.t[5] = u; c.in.t[6] = bv;
    c.in.t[7] = ks;
    c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = heads; c.in.i[3] = chunk; c.in.i[4] = left;
    c.in.i[5] = query_row0; c.in.i[6] = items;
    run(name, c, g_relative_attention_f32, {o}, FP32);
    const std::vector<uint32_t> batched = fetch(c, o);
    bool same = true;
    for (unsigned b = 0; b < items; b++) {
        PlowDevInst one = c.in;
        for (int t : {0, 1, 2, 3, 7}) one.t[t] = alone[b].t[t];
        one.i[6] = 0;
        launch(one, c.dev);
        const std::vector<uint32_t> own = fetch(c, one.t[0]);
        same = same && std::equal(own.begin(), own.end(), batched.begin() + b * n);
    }
    verdict(name, "each item == the item alone", same);
}

/* CausalDepthwiseConv1dF32 over items of item_rows rows (i3): against the golden, and each item bit
 * for bit as in a launch of its own. */
static void t_dwconv_items(unsigned items, unsigned item_rows, unsigned ch, unsigned kernel) {
    const char* name = "CausalDepthwiseConv1dF32 items";
    Case c;
    const size_t n = (size_t)item_rows * ch;
    unsigned o = c.out(items * n), x = c.f32(items * n, 1.f), w = c.f16((size_t)ch * kernel, 0.3f);
    std::vector<std::pair<unsigned, unsigned>> alone;
    for (unsigned b = 0; b < items; b++) alone.push_back({c.out(n), item_copy(c, x, b, n * 4)});
    c.in.op = PLOW_DOP_CAUSAL_DEPTHWISE_CONV1D_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w;
    c.in.i[0] = items * item_rows; c.in.i[1] = ch; c.in.i[2] = kernel; c.in.i[3] = item_rows;
    run(name, c, g_causal_depthwise_conv1d_f32, {o}, FP32);
    const std::vector<uint32_t> batched = fetch(c, o);
    bool same = true;
    for (unsigned b = 0; b < items; b++) {
        PlowDevInst one = c.in;
        one.t[0] = alone[b].first; one.t[1] = alone[b].second; one.i[0] = item_rows; one.i[3] = 0;
        launch(one, c.dev);
        const std::vector<uint32_t> own = fetch(c, one.t[0]);
        same = same && std::equal(own.begin(), own.end(), batched.begin() + b * n);
    }
    verdict(name, "each item == the item alone", same);
}

/* Q8GemmF32 at Nemotron's shapes: the GEMV walk (path 1) against tiles (path 2) by row count, then
 * tiles alone at an upload launch's rows. */
static void bench_q8_paths() {
    const unsigned shapes[][2] = {{1024, 1024}, {4096, 1024}, {1024, 4096}, {2048, 1024}, {13088, 640}, {2560, 640}};
    for (const auto& shape : shapes) {
        const unsigned n = shape[0], k = shape[1];
        for (unsigned m : {1u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u}) {
            for (unsigned path : {1u, 2u}) {
                if (path == 1u && m > 1024u) continue;
                Case c;
                unsigned o = c.out((size_t)m * n), x = c.f32((size_t)m * k, 1.f), w = q8_weights(c, n, k);
                c.in.op = PLOW_DOP_Q8_GEMM_F32;
                c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w;
                c.in.i[0] = m; c.in.i[1] = n; c.in.i[2] = k; c.in.i[7] = path;
                char name[64];
                snprintf(name, sizeof name, "Q8GemmF32 %s M=%u N=%u K=%u", path == 1 ? "gemv" : "tiles", m, n, k);
                bench(name, c, 2.0 * m * n * k, (double)n * (k / 32) * 34);
            }
        }
    }
}

/* RelativeAttentionF32 at Nemotron's batched stream steps: `items` windows of `rows` rows, the rows from
 * query_row0 querying (live 60/56, upload 88/56), every key filled. */
static void bench_relative_items(unsigned items, unsigned rows, unsigned query_row0) {
    Case c;
    const unsigned width = 1024;
    const size_t n = (size_t)items * rows * width;
    unsigned o = c.out(n), q = c.f32(n, 1.f), k = c.f32(n, 1.f), v = c.f32(n, 1.f);
    unsigned p = c.f32((size_t)(2 * rows - 1) * width, 1.f), u = c.f32(width, 0.3f), bv = c.f32(width, 0.3f);
    c.in.op = PLOW_DOP_RELATIVE_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = p; c.in.t[5] = u; c.in.t[6] = bv;
    c.in.t[7] = c.u32(std::vector<uint32_t>(items, 0u));
    c.in.i[0] = rows; c.in.i[1] = width; c.in.i[2] = 8; c.in.i[3] = 4; c.in.i[4] = 14;
    c.in.i[5] = query_row0; c.in.i[6] = items;
    char name[64];
    snprintf(name, sizeof name, "RelativeAttentionF32 items=%u rows=%u from %u", items, rows, query_row0);
    bench(name, c, 0, 0);
}

/* Per-item prefix rows (AttentionF32 flags bit 5, t5 = pre_b per item). Separate Q/K/V; own K/V
 * rows past each item's length are NaN / 1e30; prefix rows no user of a block reads are NaN
 * (fp32 / tc routes) or, for h16, formatted tiles past the last one any user reads are NaN.
 * route: 0 fp32 kernel (tc flag off), 1 tc (kernel chosen by q_rows), 2 h16. When every
 * pre_b >= i7 the output is also compared bitwise with the flag-off instruction. */
static void t_attention_prefix_rows(const char* name, int route, unsigned batch, unsigned q_rows, unsigned hw,
                                    unsigned pre, unsigned blocks, std::vector<uint32_t> index,
                                    std::vector<uint32_t> rows, std::vector<uint32_t> lengths) {
    Case c;
    const unsigned heads = 4, width = heads * hw, kv_rows = 50;
    auto pre_b = [&](unsigned b) { return std::min(rows[b], pre); };
    auto klen = [&](unsigned b) {
        const unsigned all = pre_b(b) + kv_rows;
        return lengths.empty() ? all : std::min(lengths[b], all);
    };
    unsigned o = c.out((size_t)batch * q_rows * width);
    unsigned q = c.f32((size_t)batch * q_rows * width, 1.f);
    unsigned k = c.f32((size_t)batch * kv_rows * width, 1.f);
    unsigned v = c.f32((size_t)batch * kv_rows * width, 1.f);
    unsigned pf = c.f32((size_t)blocks * pre * 2 * width, 1.f);
    unsigned ix = c.u32(index), pr = c.u32(rows);
    unsigned l = lengths.empty() ? PLOW_TENSOR_NONE : c.u32(lengths);
    std::vector<unsigned> used(blocks, 0);
    for (unsigned b = 0; b < batch; b++)
        used[index[b]] = std::max(used[index[b]], route == 2 ? pre_b(b) : std::min(pre_b(b), klen(b)));
    if (route != 2) {
        float* P = (float*)c.host[pf].data();
        for (unsigned blk = 0; blk < blocks; blk++)
            for (size_t e = ((size_t)blk * pre + used[blk]) * 2 * width; e < (size_t)(blk + 1) * pre * 2 * width; e++)
                P[e] = NAN;
    }
    float *K = (float*)c.host[k].data(), *V = (float*)c.host[v].data();
    for (unsigned b = 0; b < batch; b++) {
        const unsigned own = klen(b) > pre_b(b) ? klen(b) - pre_b(b) : 0u;
        for (size_t e = ((size_t)b * kv_rows + own) * width; e < (size_t)(b + 1) * kv_rows * width; e++) {
            K[e] = NAN;
            V[e] = 1e30f;
        }
    }
    c.in.op = PLOW_DOP_ATTENTION_F32;
    c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = l; c.in.t[5] = pr;
    c.in.t[6] = pf; c.in.t[7] = ix;
    c.in.i[0] = batch; c.in.i[1] = q_rows; c.in.i[2] = kv_rows; c.in.i[3] = heads; c.in.i[4] = hw;
    c.in.i[5] = 0; c.in.i[6] = (route ? 2u : 0u) | 32u; c.in.i[7] = pre;
    c.in.fj[0].f = 0.125f;
    const PlowDevInst ref = c.in;
    std::vector<PlowDevInst> pres;
#if defined(PLOW_NV_HOPPER)
    if (route == 2) {
        const unsigned npt = (pre + SPH_KT - 1) / SPH_KT, blk_floats = heads * npt * SPH_XS;
        unsigned fm = c.out((size_t)blocks * blk_floats);
        unsigned nan = c.out(blk_floats);
        PlowDevInst f;
        memset(&f, 0, sizeof(f));
        for (auto& x : f.t) x = PLOW_TENSOR_NONE;
        f.op = PLOW_DOP_COPY_COLS_F32;
        f.t[0] = fm; f.t[1] = pf;
        f.i[0] = blocks; f.i[1] = pre; f.i[2] = 2 * width; f.i[3] = 2 * width; f.i[7] = 1;
        f.fj[1].u = pre * 2 * width; f.fj[2].u = blk_floats;
        pres.push_back(f);
        for (unsigned blk = 0; blk < blocks; blk++) {
            const unsigned nu = (used[blk] + SPH_KT - 1) / SPH_KT;
            if (nu == npt) continue;
            PlowDevInst z = f;
            z.t[1] = nan;
            z.i[0] = heads; z.i[1] = 1; z.i[2] = (npt - nu) * SPH_XS; z.i[3] = 0; z.i[4] = 0; z.i[5] = 0;
            z.i[6] = blk * blk_floats + nu * SPH_XS; z.i[7] = 0;
            z.fj[1].u = 0; z.fj[2].u = npt * SPH_XS;
            pres.push_back(z);
        }
        c.in.i[6] |= 16u;
        c.in.t[6] = fm;
    }
#endif
    run(name, c, g_attention_f32, {o}, FP32, 2e-5, 0, pres.empty() ? nullptr : pres.data(), &ref, pres.size());
    bool full = true;
    for (unsigned b = 0; b < batch; b++) full = full && rows[b] >= pre;
    if (full) {
        const std::vector<uint32_t> on = fetch(c, o);
        PlowDevInst off = c.in;
        off.i[6] &= ~32u;
        off.t[5] = PLOW_TENSOR_NONE;
        launch(off, c.dev);
        verdict(name, "bitwise == flag off", fetch(c, o) == on);
    }
}

/* CopyColsF32 flags bit 1 (plain): items land at group g; capacity in flag bits 8..31. */
static void t_copy_cols_group(const char* name, unsigned g, unsigned capacity) {
    Case c;
    const unsigned items = 2, rows = 40, cols = 80, in_stride = 96, out_stride = 88;
    unsigned o = c.out((size_t)capacity * rows * out_stride);
    unsigned x = c.f32((size_t)items * rows * in_stride, 1.f);
    unsigned gb = c.u32({g});
    c.in.op = PLOW_DOP_COPY_COLS_F32; c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = gb;
    c.in.i[0] = items; c.in.i[1] = rows; c.in.i[2] = cols; c.in.i[3] = in_stride; c.in.i[4] = 8;
    c.in.i[5] = out_stride; c.in.i[6] = 3; c.in.i[7] = 2u | capacity << 8;
    c.in.fj[1].u = rows * in_stride; c.in.fj[2].u = rows * out_stride;
    run(name, c, g_copy_cols_f32, {o}, FP32, 0.0);
}

#if defined(PLOW_NV_HOPPER)
/* CopyColsF32 format mode with a group: equals an ungrouped format shifted by g * items blocks;
 * out-of-capacity groups write nothing; a short item over a long one leaves the short format's
 * tiles exactly as a fresh write and the long one's later tiles untouched. */
static void t_format_group() {
    const char* name = "CopyColsF32 h16 format group";
    Case c;
    const unsigned heads = 4, width = heads * 64, items = 2, L = 45, S = 20, cap = 6;
    const unsigned blk = heads * ((L + SPH_KT - 1) / SPH_KT) * SPH_XS;
    const unsigned ns = heads * ((S + SPH_KT - 1) / SPH_KT) * SPH_XS;
    unsigned xl = c.f32((size_t)items * L * 2 * width, 1.f), xs = c.f32((size_t)items * L * 2 * width, 1.f);
    unsigned fresh = c.out((size_t)cap * blk), grp = c.out((size_t)cap * blk), guard = c.out((size_t)cap * blk);
    unsigned freshs = c.out((size_t)cap * blk), reuse = c.out((size_t)cap * blk);
    unsigned g2 = c.u32({2}), g0 = c.u32({0});
    c.upload();
    PlowDevInst f;
    memset(&f, 0, sizeof(f));
    for (auto& t : f.t) t = PLOW_TENSOR_NONE;
    f.op = PLOW_DOP_COPY_COLS_F32;
    f.i[0] = items; f.i[1] = L; f.i[2] = 2 * width; f.i[3] = 2 * width; f.i[7] = 1;
    f.fj[1].u = L * 2 * width; f.fj[2].u = blk;
    auto go = [&](unsigned out, unsigned x, unsigned rows, int g, unsigned capacity) {
        PlowDevInst i = f;
        i.t[0] = out; i.t[1] = x; i.i[1] = rows;
        if (g >= 0) {
            i.t[2] = g == 2 ? g2 : g0;
            i.i[7] = 1u | 2u | capacity << 8;
        }
        launch(i, c.dev);
    };
    go(fresh, xl, L, -1, 0);
    go(grp, xl, L, 2, cap);
    go(guard, xl, L, 2, cap - 1);
    go(freshs, xs, S, 0, cap);
    go(reuse, xl, L, 2, cap);
    go(reuse, xs, S, 2, cap);
    const std::vector<uint32_t> a = fetch(c, fresh), b = fetch(c, grp), z = fetch(c, guard);
    const std::vector<uint32_t> fs = fetch(c, freshs), r = fetch(c, reuse);
    bool shift = true, none = true, tail = true;
    for (size_t e = 0; e < (size_t)cap * blk; e++) {
        const size_t it = e / blk, w = e % blk;
        const bool in2 = it >= 2 * items && it < 3 * items;
        shift = shift && b[e] == (in2 ? a[e - 2 * items * blk] : 0x7fc00badu);
        none = none && z[e] == 0x7fc00badu;
        const uint32_t want = !in2 ? 0x7fc00badu : w < ns ? fs[e - 2 * items * blk] : b[e];
        tail = tail && r[e] == want;
    }
    verdict(name, "g=2 == ungrouped shifted", shift);
    verdict(name, "(g+1)*items > capacity writes nothing", none);
    verdict(name, "long->short reuse == fresh short", tail);
}
#endif

#if defined(PLOW_NV_HOPPER) && PLOW_SPEECH_SPLIT_A
/* The split image of an f32 activation [rows][width] (devgen `split_image`): per 64-column k-tile
 * the bf16 hi then lo halves of all rows, 128-byte rows, chunk c at c ^ (row & 7). */
static std::vector<uint8_t> host_split_image(const float* x, unsigned rows, unsigned width) {
    std::vector<uint8_t> img((size_t)rows * width * 4, 0);
    for (size_t r = 0; r < rows; r++)
        for (size_t j = 0; j < width; j++) {
            const float v = x[r * width + j];
            const uint16_t hi = bf16_rn(v);
            const uint32_t hb = (uint32_t)hi << 16;
            float hf;
            memcpy(&hf, &hb, 4);
            const uint16_t lo = bf16_rn(v - hf);
            const size_t off = (j / 64) * 2 * (size_t)rows * 128 + r * 128 + (((j % 64 / 8) ^ (r & 7)) * 16) + (j % 8) * 2;
            memcpy(&img[off], &hi, 2);
            memcpy(&img[off + (size_t)rows * 128], &lo, 2);
        }
    return img;
}
static void split_verdict(const char* name, const char* what, bool ok) {
    printf("  %-34s %-40s -> %s\n", name, what, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
}
static std::vector<uint8_t> download(void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    CK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
    return h;
}
/* Split-image hand-offs between the CFM transformer block's ops: a producer's image equals the
 * host split of its f32 output byte for byte, and a GEMM reading the image equals the f32-input
 * GEMM bit for bit (same split, products and accumulation order). Rows 300: a 128-row tail. */
static void t_split_image() {
    const unsigned rows = 300, feat = 256;
    {
        /* LayerNormF32 flag bit 2 vs the f32 rows. */
        Case c;
        unsigned y = c.out((size_t)rows * feat), img = c.out((size_t)rows * feat), x = c.f32((size_t)rows * feat, 2.f, 0.5f);
        unsigned g = c.f32(feat, 0.5f, 1.f), b = c.f32(feat, 0.2f);
        c.in.op = PLOW_DOP_LAYERNORM_F32;
        c.in.t[0] = y; c.in.t[1] = x; c.in.t[2] = g; c.in.t[3] = b;
        c.in.i[0] = rows; c.in.i[1] = feat; c.in.i[2] = 0; c.in.fj[0].f = 1e-5f;
        PlowDevInst li = c.in;
        li.t[0] = img; li.i[2] = 4u;
        c.upload();
        void** td = upload_table(c.dev);
        k_speech<<<g_nblk, 256, kSmem>>>(c.in, td);
        k_speech<<<g_nblk, 256, kSmem>>>(li, td);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        const std::vector<uint8_t> yh = download(c.dev[y], (size_t)rows * feat * 4), ih = download(c.dev[img], (size_t)rows * feat * 4);
        split_verdict("LayerNormF32 split image", "== host split of the f32 rows", ih == host_split_image((const float*)yh.data(), rows, feat));
        CK(cudaFree(td));
    }
    struct G { const char* name; unsigned cin, cout, post; bool res, ao; unsigned nblk; };
    /* The served shapes, then: a 128-wide tile with a column tail (N 320), one k-tile (K 64), and
     * one-block launches (every CTA walks many tiles: output-stage reuse behind the bulk stores). */
    const G gs[] = {
        {"qkv 256->1536", 256, 1536, 0, false, false, 0},
        {"out 512->256 +res", 512, 256, 0, true, false, 0},
        {"ff1 256->1024 gelu -> image", 256, 1024, 11, false, true, 0},
        {"ff2 1024->256 +res", 1024, 256, 0, true, false, 0},
        {"256->320 (BN 128, column tail)", 256, 320, 0, false, false, 0},
        {"64->256 (one k-tile)", 64, 256, 0, false, false, 0},
        {"ff2 1024->256 +res, 1 block", 1024, 256, 0, true, false, 1},
        {"ff1 256->1024 gelu -> image, 1 block", 256, 1024, 11, false, true, 1},
    };
    for (const G& g : gs) {
        /* Conv1dF32 flags bit 30 (A image) [| bit 31 (image out)] vs the f32 path, one Case. */
        const unsigned nblk = g.nblk ? g.nblk : g_nblk;
        ConvSpec s{1, rows, g.cin, g.cout, 1, 1, 1, 1, 0, 0, 0, 0, g.post, false, true, false, g.res, {}, 0.f, true};
        s.wide = true;
        s.wg = true;
        Case c; unsigned o;
        build_conv(c, s, false, o);
        const unsigned x = c.in.t[1];
        const unsigned xi = c.add(c.host[x].size()), o2 = c.out((size_t)rows * g.cout);
        c.host[xi] = host_split_image((const float*)c.host[x].data(), rows, g.cin);
        PlowDevInst in2 = c.in;
        in2.t[0] = o2; in2.t[1] = xi; in2.fj[2].u |= (1u << 30) | (g.ao ? 1u << 31 : 0u);
        c.upload();
        void** td = upload_table(c.dev);
        k_speech<<<nblk, 256, kSmem>>>(c.in, td);
        k_speech<<<nblk, 256, kSmem>>>(in2, td);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        const std::vector<uint8_t> a = download(c.dev[o], (size_t)rows * g.cout * 4), b = download(c.dev[o2], (size_t)rows * g.cout * 4);
        char name[96];
        snprintf(name, sizeof(name), "Conv1dF32 image A %s", g.name);
        if (g.ao) split_verdict(name, "image out == host split of f32 out", b == host_split_image((const float*)a.data(), rows, g.cout));
        else split_verdict(name, "== f32-input wgmma GEMM, bit for bit", a == b);
        CK(cudaFree(td));
    }
    {
        /* AttentionF32 h16 flag bit 7 vs its f32 rows: 4 CFG items x 72 rows, 306-row prefix. */
        const unsigned batch = 4, q = 72, heads = 8, hw = 64, width = heads * hw, stride = 3 * width, pre = 306;
        std::vector<uint32_t> ix(batch);
        for (unsigned i = 0; i < batch; i++) ix[i] = i % 2;
        Case c;
        unsigned o = c.out((size_t)batch * q * width), img = c.out((size_t)batch * q * width), qkv = c.f32((size_t)batch * q * stride, 1.f);
        unsigned pf = c.f32((size_t)2 * pre * 2 * width, 1.f), x = c.u32(ix);
        const unsigned npt = (pre + SPH_KT - 1) / SPH_KT, fm = c.out((size_t)2 * heads * npt * SPH_XS);
        PlowDevInst f;
        memset(&f, 0, sizeof(f));
        for (auto& t : f.t) t = PLOW_TENSOR_NONE;
        f.op = PLOW_DOP_COPY_COLS_F32;
        f.t[0] = fm; f.t[1] = pf;
        f.i[0] = 2; f.i[1] = pre; f.i[2] = 2 * width; f.i[3] = 2 * width; f.i[7] = 1;
        f.fj[1].u = pre * 2 * width; f.fj[2].u = heads * npt * SPH_XS;
        c.in.op = PLOW_DOP_ATTENTION_F32;
        c.in.t[0] = o; c.in.t[1] = qkv; c.in.t[2] = qkv; c.in.t[3] = qkv; c.in.t[4] = PLOW_TENSOR_NONE;
        c.in.t[5] = PLOW_TENSOR_NONE; c.in.t[6] = fm; c.in.t[7] = x;
        c.in.i[0] = batch; c.in.i[1] = q; c.in.i[2] = q; c.in.i[3] = heads; c.in.i[4] = hw;
        c.in.i[5] = stride; c.in.i[6] = 2u | 16u; c.in.i[7] = pre;
        c.in.fj[0].f = 0.125f; c.in.fj[1].u = width; c.in.fj[2].u = 2 * width;
        PlowDevInst ai = c.in;
        ai.t[0] = img; ai.i[6] |= 128u;
        c.upload();
        void** td = upload_table(c.dev);
        k_speech<<<g_nblk, 256, kSmem>>>(f, td);
        k_speech<<<g_nblk, 256, kSmem>>>(c.in, td);
        k_speech<<<g_nblk, 256, kSmem>>>(ai, td);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        const std::vector<uint8_t> a = download(c.dev[o], (size_t)batch * q * width * 4), b = download(c.dev[img], (size_t)batch * q * width * 4);
        split_verdict("AttentionF32 h16 split image", "== host split of the f32 rows", b == host_split_image((const float*)a.data(), batch * q, width));
        CK(cudaFree(td));
    }
}
#endif
template <unsigned STEP>
__global__ void k_bf16_mma_replay(const float* x, const float* w, const float* bias, float* y,
                                 unsigned M, unsigned N, unsigned K, unsigned span, unsigned mode) {
    const unsigned lane = threadIdx.x & 31u, g = lane >> 2, p = 2u * (lane & 3u);
    const unsigned tile = blockIdx.x * 8u + threadIdx.x / 32u, tn = (N + 7u) / 8u;
    if (tile >= ((M + 15u) / 16u) * tn) return;
    const unsigned r = (tile / tn) * 16u + g, c = (tile % tn) * 8u;
    float total[4] = {};
    for (unsigned lo = 0; lo < K; lo += span) {
        const unsigned end = min(K, lo + span);
        float d[4] = {};
        for (unsigned k = lo; k < end; k += STEP) {
            unsigned a[4] = {}, b[2] = {};
#pragma unroll
            for (unsigned h = 0; h < STEP / 8u; h++) {
#pragma unroll
                for (unsigned j = 0; j < 2u; j++) {
                    const unsigned ci = k + p + h * 8u + j;
#pragma unroll
                    for (unsigned hi = 0; hi < 2u; hi++) {
                        const unsigned ri = r + hi * 8u;
                        const float v = ri < M && ci < end ? x[(size_t)ri * K + ci] : 0.f;
                        a[2u * h + hi] |= (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(v)) << (16u * j);
                    }
                    const float v = c + g < N && ci < end ? w[(size_t)(c + g) * K + ci] : 0.f;
                    b[h] |= (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(v)) << (16u * j);
                }
            }
            if constexpr (STEP == 8u) {
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(a[0]), "r"(a[1]), "r"(b[0]));
            } else {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
        }
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            const float part = mode == 1u || (mode == 2u && lo == 0u) ? sp_bf16(d[j]) : d[j];
            total[j] = __fadd_rn(total[j], part);
            if (mode == 2u) total[j] = sp_bf16(total[j]);
        }
    }
#pragma unroll
    for (unsigned j = 0; j < 4u; j++) {
        const unsigned ri = r + (j / 2u) * 8u, ci = c + p + (j & 1u);
        if (ri < M && ci < N)
            y[(size_t)ri * N + ci] = sp_bf16(__fadd_rn(sp_bf16(total[j]), bias[ci]));
    }
}

static void native_bf16_dense_replay(const std::vector<float>& x, const std::vector<float>& w,
                                     const std::vector<float>& bias, std::vector<float>& actual,
                                     unsigned M, unsigned N, unsigned K, unsigned span, unsigned mode, unsigned step) {
    std::vector<float> total((size_t)M * N, 0.f), partial(total.size());
    for (unsigned lo = 0; lo < K; lo += span) {
        const unsigned width = std::min(span, K - lo), padded = (width + 7u) & ~7u;
        Case c;
        const unsigned a = c.add((size_t)M * padded * 4u), b = c.add((size_t)N * padded * 2u), o = c.out(total.size());
        for (unsigned row = 0; row < M; row++)
            memcpy(c.host[a].data() + (size_t)row * padded * 4u, x.data() + (size_t)row * K + lo, width * 4u);
        auto* wb = (uint16_t*)c.host[b].data();
        for (unsigned row = 0; row < N; row++) for (unsigned col = 0; col < width; col++)
            wb[(size_t)row * padded + col] = f2bf(w[(size_t)row * K + lo + col]);
        c.in.op = PLOW_DOP_DENSE_GEMM_F32;
        c.in.t[0] = o; c.in.t[1] = a; c.in.t[2] = b;
        c.in.i[0] = M; c.in.i[1] = N; c.in.i[2] = padded;
        const bool round = mode == 1u || (mode == 2u && lo == 0u);
        c.in.i[7] = 4u | 8u | 0x800u | (step == 8u ? 0x1000u : 0u) | (unsigned)round;
        c.upload();
        void** table = upload_table(c.dev);
        k_speech<<<g_nblk, 256, kSmem>>>(c.in, table);
        CK(cudaGetLastError());
        CK(cudaMemcpy(partial.data(), c.dev[o], partial.size() * 4u, cudaMemcpyDeviceToHost));
        CK(cudaFree(table));
        for (size_t i = 0; i < total.size(); i++) {
            total[i] += partial[i];
            if (mode == 2u) total[i] = plow_bf2f(f2bf(total[i]));
        }
    }
    for (size_t i = 0; i < actual.size(); i++)
        actual[i] = plow_bf2f(f2bf(plow_bf2f(f2bf(total[i])) + bias[i % N]));
}

static int t_bf16_mma_replay(const char* input, const char* output, bool native = false) {
    FILE* in = fopen(input, "rb");
    FILE* out = fopen(output, "wb");
    if (!in || !out) { perror("BF16 MMA fixture"); return 2; }
    auto read = [&](void* p, size_t bytes) {
        if (fread(p, 1, bytes, in) != bytes) { fprintf(stderr, "truncated BF16 MMA fixture\n"); exit(2); }
    };
    auto write = [&](const void* p, size_t bytes) {
        if (fwrite(p, 1, bytes, out) != bytes) { perror("BF16 MMA output"); exit(2); }
    };
    uint32_t header[2];
    read(header, sizeof(header));
    if (header[0] != 0x31464d42u || !header[1] || header[1] > 4096u) return 2;
    write(header, sizeof(header));
    size_t differences[2] = {}, exact[2] = {};
    bool finite = true;
    for (unsigned item = 0; item < header[1]; item++) {
        // BMF1: rows, columns, K, span, reduction (0 full F32, 1 parallel BF16, 2 serial BF16, 3 partitioned F32).
        uint32_t shape[5];
        read(shape, sizeof(shape));
        const unsigned M = shape[0], N = shape[1], K = shape[2], span = shape[3], mode = shape[4];
        if (!M || !N || !K || !span || span > K || mode > 3u || (mode == 0u && span != K) ||
            (uint64_t)M * K > (1u << 27) || (uint64_t)N * K > (1u << 27) ||
            (uint64_t)M * N > (1u << 27)) return 2;
        std::vector<float> x((size_t)M * K), w((size_t)N * K), bias(N), expected((size_t)M * N), actual(expected.size());
        read(x.data(), x.size() * 4u);
        read(w.data(), w.size() * 4u);
        read(bias.data(), bias.size() * 4u);
        read(expected.data(), expected.size() * 4u);
        for (const auto* values : {&x, &w, &bias}) for (float v : *values) {
            uint32_t bits; memcpy(&bits, &v, 4);
            if (!std::isfinite(v) || (bits & 0xffffu)) return 2;
        }
        float *dx = nullptr, *dw = nullptr, *db = nullptr, *dy = nullptr;
        if (!native) {
            CK(cudaMalloc(&dx, x.size() * 4u)); CK(cudaMalloc(&dw, w.size() * 4u));
            CK(cudaMalloc(&db, bias.size() * 4u)); CK(cudaMalloc(&dy, actual.size() * 4u));
            CK(cudaMemcpy(dx, x.data(), x.size() * 4u, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dw, w.data(), w.size() * 4u, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(db, bias.data(), bias.size() * 4u, cudaMemcpyHostToDevice));
        }
        write(shape, sizeof(shape));
        const unsigned blocks = ((((M + 15u) / 16u) * ((N + 7u) / 8u)) + 7u) / 8u;
        for (unsigned variant = 0; variant < 2u; variant++) {
            if (native) native_bf16_dense_replay(x, w, bias, actual, M, N, K, span, mode, variant == 0u ? 8u : 16u);
            else {
                CK(cudaMemset(dy, 0xff, actual.size() * 4u));
                if (variant == 0u) k_bf16_mma_replay<8><<<blocks, 256>>>(dx, dw, db, dy, M, N, K, span, mode);
                else k_bf16_mma_replay<16><<<blocks, 256>>>(dx, dw, db, dy, M, N, K, span, mode);
                CK(cudaGetLastError());
                CK(cudaMemcpy(actual.data(), dy, actual.size() * 4u, cudaMemcpyDeviceToHost));
            }
            size_t changed = 0;
            for (size_t i = 0; i < actual.size(); i++) {
                finite = finite && std::isfinite(actual[i]);
                changed += memcmp(&actual[i], &expected[i], 4) != 0;
            }
            differences[variant] += changed;
            exact[variant] += changed == 0;
            write(actual.data(), actual.size() * 4u);
            printf("mma_replay case=%u backend=%s kstep=%u rows=%u width=%u k=%u span=%u mode=%u changed=%zu\n",
                   item, native ? "native_dense_cpu_reduce" : "intrinsic", variant == 0u ? 8u : 16u, M, N, K, span, mode, changed);
        }
        if (!native) { CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(db)); CK(cudaFree(dy)); }
    }
    if (fgetc(in) != EOF || ferror(in) || fclose(in) != 0 || fclose(out) != 0) return 2;
    printf("mma_replay cases=%u k8_exact=%zu k16_exact=%zu k8_differences=%zu k16_differences=%zu finite=%u\n",
           header[1], exact[0], exact[1], differences[0], differences[1], (unsigned)finite);
    return finite && !differences[0] && !differences[1] ? 0 : 1;
}

int main(int argc, char** argv) {
#ifdef SPEECH_TEST_SHORT_ATTENTION_ONLY
    if (argc != 2 || (strcmp(argv[1], "--short-attention") && strcmp(argv[1], "--bench-short-attention"))) return 2;
#endif
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    g_nblk = prop.multiProcessorCount;
    unsigned arena_floats;
    CK(cudaMemcpyFromSymbol(&arena_floats, k_arena_floats, sizeof(arena_floats)));
    kSmem = (size_t)arena_floats * sizeof(float);
    CK(cudaFuncSetAttribute(k_speech, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem));
    printf("speech_f32_op_test on %s (%d SMs), nblk=%d, arena=%zu B\n", prop.name, g_nblk, g_nblk, kSmem);

#ifdef SPEECH_TEST_RING_ONLY
    t_ring_kv();
    printf(g_fail ? "FAIL\n" : "ALL PASS\n");
    return g_fail;
#endif

    if (argc > 1 && !strcmp(argv[1], "--short-attention")) {
        for (unsigned b : {3u, 16u, 33u})
            for (unsigned q : {1u, 2u, 17u, 32u})
                for (unsigned k : {1u, 7u, 32u})
                    for (bool causal : {false, true})
                        for (bool fused : {false, true}) t_attention_short(b, q, k, causal, fused, false);
        return g_fail;
    }
    if (argc > 1 && !strcmp(argv[1], "--bench-short-attention")) {
        for (unsigned b : {32u, 64u, 128u, 200u, 224u})
            for (unsigned q : {1u, 2u, 26u}) t_attention_short(b, q, q == 1 ? 32u : q, q != 1, true, true);
        return g_fail;
    }

    if (argc > 1 && !strcmp(argv[1], "--check-split")) {
#if defined(PLOW_NV_HOPPER) && PLOW_SPEECH_SPLIT_A
        t_split_image();
#else
        printf("  FAIL split-image arms not compiled (PLOW_NV_HOPPER && PLOW_SPEECH_SPLIT_A)\n");
        g_fail = true;
#endif
        printf(g_fail ? "FAIL\n" : "ALL PASS\n");
        return g_fail;
    }
    if (argc == 4 && !strcmp(argv[1], "--bf16-mma-replay")) return t_bf16_mma_replay(argv[2], argv[3]);
    if (argc == 4 && !strcmp(argv[1], "--bf16-native-replay")) return t_bf16_mma_replay(argv[2], argv[3], true);

    if (argc > 1 && !strcmp(argv[1], "--conv-padding-only")) {
        t_conv_padding();
        return g_fail;
    }
    if (argc > 1 && !strcmp(argv[1], "--gelu-only")) {
        t_gelu_all();
        printf(g_fail ? "FAIL\n" : "ALL PASS\n");
        return g_fail;
    }
    if (argc > 1 && !strcmp(argv[1], "--dense-cont-only")) {
        t_dense("DenseGemmF32 cont16 golden 390x1024x1024", 390, 1024, 1024, 1 | 4 | 8 | 0x800, 0, 0, 0, 0, true);
        t_dense("DenseGemmF32 cont16 gelu 208x4096x1024", 208, 4096, 1024, 2 | 4 | 8 | 0x800, 0, 0, 0, 0, true);
        t_dense("DenseGemmF32 cont16 ragged 77x200x328", 77, 200, 328, 1 | 4 | 8 | 0x800, 0, 3, 0, 0, true);
        for (unsigned reqs : {2u, 8u, 16u}) {
            t_dense_cont(1024, 1024, 1, std::vector<unsigned>(reqs, 104));
            t_dense_cont(4096, 1024, 2, std::vector<unsigned>(reqs, 104));
            t_dense_cont(1024, 4096, 1, std::vector<unsigned>(reqs, 104));
        }
        t_dense_cont(1024, 7680, 1, std::vector<unsigned>(16, 104));
        t_dense_cont(1024, 1024, 1, {104, 52, 156, 13, 390, 208, 390, 390, 260});
        t_dense_cont(896, 3584, 1, {91, 13, 26, 39, 52});
        /* Unrounded sums: K % 16 == 8, ragged tiles. */
        t_dense_cont(1024, 1016, 0, std::vector<unsigned>(16, 104));
        t_dense_cont(1000, 4088, 0, {390, 77, 13, 390, 390, 131});
        printf(g_fail ? "FAIL\n" : "ALL PASS\n");
        return g_fail;
    }
    if (argc > 1 && !strcmp(argv[1], "--dense-ref-only")) {
        /* The Qwen3-ASR 1.7B encoder layer at a 128-chunk pack (16 requests of 104 rows). */
        t_dense_ref(1024, 1024, std::vector<unsigned>(16, 104));
        t_dense_ref(4096, 1024, std::vector<unsigned>(16, 104));
        t_dense_ref(1024, 4096, std::vector<unsigned>(16, 104));
        t_dense_ref(1024, 1024, {104, 52, 156, 13, 390, 208, 390, 390, 260});
        t_dense_ref(1024, 4096, {104, 52, 260});
        t_dense_ref(4096, 1024, {104, 52});
        t_dense_ref(1024, 7680, {104, 52, 156});
        for (unsigned m : {208u, 832u, 1664u}) {
            bench_dense_ref(m, 1024, 1024, 1, 104);
            bench_dense_ref(m, 4096, 1024, 2, 104);
            bench_dense_ref(m, 1024, 4096, 1, 104);
        }
        bench_dense_ref(1664, 1024, 7680, 1, 104);
        printf(g_fail ? "FAIL\n" : "ALL PASS\n");
        return g_fail;
    }
    if (argc > 1 && !strcmp(argv[1], "--conv-packing-only")) {
        /* The Qwen3-ASR front end: conv1 (3xTF32, f32 input), conv2 / conv3 (bf16-exact tensor cores). */
        for (unsigned b : {2u, 16u, 128u}) {
            t_conv_packing("conv1 1->480 128x100", b, 128, 100, 1, 480, 744);
            t_conv_packing("conv2 480->480 64x50", b, 64, 50, 480, 480, 488);
            t_conv_packing("conv3 480->480 32x25", b, 32, 25, 480, 480, 488);
            /* Blocked bf16 weights, also against the f32-stored weights' output. */
            t_conv_packing("conv2 480->480 64x50 blocked", b, 64, 50, 480, 480, (488 & ~64) | 1024);
            t_conv_packing("conv3 480->480 32x25 blocked", b, 32, 25, 480, 480, (488 & ~64) | 1024);
        }
        t_conv("Conv2dF32 tc-bf16 blocked weights multi-tile", 8, 64, 50, 48, 320, 3, 2, 1, 1, 256 | 1024 | 128 | (2 << 2) | (2 << 4));
        t_conv("Conv2dF32 tc-bf16 blocked weights odd width", 3, 16, 25, 64, 96, 3, 2, 1, 1, 256 | 1024 | 128 | (2 << 2) | (2 << 4));
        printf(g_fail ? "FAIL\n" : "ALL PASS\n");
        return g_fail;
    }
    if (argc > 1 && !strcmp(argv[1], "--conv-pinned-only")) {
        t_conv_pinned();
        return g_fail;
    }

    if (argc > 1 && !strcmp(argv[1], "--bench-cfm")) {
        /* The S3Gen CFM estimator's GEMMs at csynth.b32.t64 (64 CFG items x 136 rows). */
        struct G { const char* name; unsigned rows, cin, cout, k, post; bool res, len; };
        const G gs[] = {
            {"qkv 256->1536", 8704, 256, 1536, 1, 0, false, false},
            {"ff1 256->1024 gelu", 8704, 256, 1024, 1, 11, false, false},
            {"ff2 1024->256 +res", 8704, 1024, 256, 1, 0, true, false},
            {"out 512->256 +res", 8704, 512, 256, 1, 0, true, false},
            {"res 320->256 +res len", 136, 320, 256, 1, 0, true, true},
            {"c1 k3 320->256 len", 136, 320, 256, 3, 0, false, true},
        };
        for (const G& g : gs)
            for (bool wg : {false, true}) {
                const unsigned batch = g.len ? 64 : 1;
                ConvSpec s{batch, g.rows, g.cin, g.cout, g.k, 1, 1, 1, g.k - 1u, 0, 0, 0, g.post, false, true, false, g.res,
                           g.len ? std::vector<uint32_t>(batch, g.rows) : std::vector<uint32_t>{}, 0.f, true};
                s.wide = true;
                s.tapmajor = g.k > 1;
                s.wg = wg;
                Case c; unsigned o; build_conv(c, s, false, o);
                char name[96];
                snprintf(name, sizeof(name), "CFM %s%s", g.name, wg ? " wgmma" : " mma.sync");
                bench(name, c, 2.0 * batch * g.rows * g.cin * g.cout * g.k, 0);
            }
        for (bool wg : {false, true}) {
            /* Encoder relative-position table: grouped 1x1, 8 heads x 64 -> 756 (b64, t 378). */
            ConvSpec s{1, 24192, 512, 8 * 756, 1, 1, 1, 8, 0, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true};
            s.wide = true;
            s.wg = wg;
            Case c; unsigned o; build_conv(c, s, false, o);
            bench(wg ? "ENC relpos g8 512->6048 wgmma" : "ENC relpos g8 512->6048 3xTF32", c, 2.0 * 24192 * 64 * 6048, 0);
        }
        {
            Case c; const unsigned rows = 8704, feat = 256;
            unsigned o = c.out((size_t)rows * feat), x = c.f32((size_t)rows * feat, 2.f, 0.5f);
            unsigned g = c.f32(feat, 0.5f, 1.f), b = c.f32(feat, 0.2f);
            c.in.op = PLOW_DOP_LAYERNORM_F32;
            c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = g; c.in.t[3] = b;
            c.in.i[0] = rows; c.in.i[1] = feat; c.in.i[2] = 0; c.in.fj[0].f = 1e-5f;
            bench("CFM LayerNormF32 8704x256", c, 0, 8.0 * rows * feat);
            c.in.t[4] = c.f32(feat, 0.7f);
            c.in.i[2] = 10u << 4;
            bench("CFM LayerNormF32 + Mish + row 8704x256", c, 0, 8.0 * rows * feat);
        }
        /* CFM self-attention: 8 heads x 64, fused QKV, 306-row prompt prefix; (rows, CFG items) =
         * csynth.b32.t64 (136, 64), the 72-row window at 64 items, and csynth.b64.t32 (72, 128). */
        for (auto rb : {std::pair<unsigned, unsigned>{136u, 64u}, {72u, 64u}, {72u, 128u}}) {
            Case c; const unsigned rows = rb.first, batch = rb.second, heads = 8, hw = 64, width = heads * hw, stride = 3 * width, pre = 306;
            std::vector<uint32_t> ix(batch);
            for (unsigned i = 0; i < batch; i++) ix[i] = i % 2;
            unsigned o = c.out((size_t)batch * rows * width), qkv = c.f32((size_t)batch * rows * stride, 1.f);
            unsigned pf = c.f32((size_t)2 * pre * 2 * width, 1.f), x = c.u32(ix);
            c.in.op = PLOW_DOP_ATTENTION_F32;
            c.in.t[0] = o; c.in.t[1] = qkv; c.in.t[2] = qkv; c.in.t[3] = qkv; c.in.t[4] = PLOW_TENSOR_NONE;
            c.in.t[5] = PLOW_TENSOR_NONE; c.in.t[6] = pf; c.in.t[7] = x;
            c.in.i[0] = batch; c.in.i[1] = rows; c.in.i[2] = rows; c.in.i[3] = heads; c.in.i[4] = hw;
            c.in.i[5] = stride; c.in.i[6] = 2u; c.in.i[7] = pre;
            c.in.fj[0].f = 0.125f; c.in.fj[1].u = width; c.in.fj[2].u = 2 * width;
            char name[80];
            snprintf(name, sizeof(name), "CFM AttentionF32 tc b%u q%u kv%u+306", batch, rows, rows);
            bench(name, c, 4.0 * batch * heads * hw * rows * (rows + pre), 0);
#if defined(PLOW_NV_HOPPER)
            /* the same with the prefix pre-formatted (sp_attention_h16) */
            const unsigned npt = (pre + SPH_KT - 1) / SPH_KT;
            Case h;
            h.in = c.in;
            for (auto& hb : c.host) h.host.push_back(hb);
            const unsigned fm = h.out((size_t)2 * heads * npt * SPH_XS);
            PlowDevInst f;
            memset(&f, 0, sizeof(f));
            for (auto& x : f.t) x = PLOW_TENSOR_NONE;
            f.op = PLOW_DOP_COPY_COLS_F32;
            f.t[0] = fm; f.t[1] = pf;
            f.i[0] = 2; f.i[1] = pre; f.i[2] = 2 * width; f.i[3] = 2 * width; f.i[7] = 1;
            f.fj[1].u = pre * 2 * width; f.fj[2].u = heads * npt * SPH_XS;
            h.in.i[6] = 2u | 16u;
            h.in.t[6] = fm;
            snprintf(name, sizeof(name), "CFM AttentionF32 h16 b%u q%u kv%u+306", batch, rows, rows);
            bench(name, h, 4.0 * batch * heads * hw * rows * (rows + pre), 0, 20, &f);
#endif
        }
        /* The transformer-block GEMMs at csynth.b64.t32 (128 CFG items x 72 rows = 9216 rows), wgmma. */
        for (const G& g : gs) {
            if (g.len) continue;
            ConvSpec s{1, 9216, g.cin, g.cout, g.k, 1, 1, 1, g.k - 1u, 0, 0, 0, g.post, false, true, false, g.res, {}, 0.f, true};
            s.wide = true;
            s.wg = true;
            Case c; unsigned o; build_conv(c, s, false, o);
            char name[96];
            snprintf(name, sizeof(name), "CFM %s wgmma M9216", g.name);
            bench(name, c, 2.0 * 9216 * g.cin * g.cout * g.k, 0);
        }
#if PLOW_SPEECH_SPLIT_A
        /* The same four GEMMs reading the split image of their input (flags bit 30; ff1 also
         * writing one, bit 31), at both c200 capacities. */
        for (unsigned M : {8704u, 9216u})
            for (const G& g : gs) {
                if (g.len) continue;
                ConvSpec s{1, M, g.cin, g.cout, g.k, 1, 1, 1, g.k - 1u, 0, 0, 0, g.post, false, true, false, g.res, {}, 0.f, true};
                s.wide = true;
                s.wg = true;
                Case c; unsigned o; build_conv(c, s, false, o);
                const unsigned x = c.in.t[1], xi = c.add(c.host[x].size());
                c.host[xi] = host_split_image((const float*)c.host[x].data(), M, g.cin);
                c.in.t[1] = xi;
                c.in.fj[2].u |= (1u << 30) | (g.post == 11u ? 1u << 31 : 0u);
                char name[96];
                snprintf(name, sizeof(name), "CFM %s image-A M%u", g.name, M);
                bench(name, c, 2.0 * M * g.cin * g.cout * g.k, 0);
            }
#endif
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--bench-splitk")) {
        /* Fixed split-K screen for the 12 Hz codec's transformer/dec.in classes at the served
         * rows (b1.f1 64, b32.f17 544, b16.f4 1088 rows): parts 1..16, the same data. */
        struct C { unsigned cin, cout, k; } cls[] = {{1024, 512, 1}, {512, 1024, 1}, {512, 3072, 1}, {4096, 1024, 1},
                                                    {1024, 4096, 1}, {1024, 1536, 7}, {512, 1024, 3}};
        for (const C& cl : cls)
            for (unsigned rows : {64u, 544u, 1088u})
                for (unsigned P : {1u, 2u, 4u, 8u, 16u}) {
                    const unsigned r = cl.k == 1 ? rows : (cl.k == 7 ? rows * 4 : rows * 2);
                    ConvSpec s{1, r, cl.cin, cl.cout, cl.k, 1, 1, 1, cl.k - 1, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true};
                    s.wide = true; s.wg = true; s.tapmajor = cl.k > 1; s.pin = 1; s.sk = P;
                    Case c; unsigned o; build_conv(c, s, false, o);
                    char name[80];
                    snprintf(name, sizeof(name), "splitk %u->%u k%u rows %u P%u", cl.cin, cl.cout, cl.k, r, P);
                    bench(name, c, 2.0 * r * cl.cin * cl.cout * cl.k, 0, 50);
                }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--bench-codec")) {
        for (bool tc : {false, true}) {
            ConvSpec s{32, 16384, 64, 64, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, true,
                       std::vector<uint32_t>(32, 16384), 0.f, tc};
            Case c; unsigned o; build_conv(c, s, false, o);
            bench(tc ? "Conv1dF32 tc pointwise 64->64 +res len [32][16384]" : "Conv1dF32 pointwise 64->64 +res len [32][16384]",
                  c, 2.0 * 32 * 16384 * 64 * 64, 12.0 * 32 * 16384 * 64, 5);
        }
        for (bool tc : {false, true}) {
            ConvSpec s{32, 2048, 256, 256, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, true,
                       std::vector<uint32_t>(32, 2048), 0.f, tc};
            Case c; unsigned o; build_conv(c, s, false, o);
            bench(tc ? "Conv1dF32 tc pointwise 256->256 +res len [32][2048]" : "Conv1dF32 pointwise 256->256 +res len [32][2048]",
                  c, 2.0 * 32 * 2048 * 256 * 256, 12.0 * 32 * 2048 * 256, 5);
        }
        /* S3Gen CFM transformer block shapes (B=1 T=128 -> 1140 rows). */
        const unsigned cfm[][2] = {{256, 1536}, {512, 256}, {256, 1024}, {1024, 256}};
        for (auto& sh : cfm) {
            ConvSpec s{1, 1140, sh[0], sh[1], 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true};
            Case c; unsigned o; build_conv(c, s, false, o);
            char name[80];
            snprintf(name, sizeof(name), "Conv1dF32 tc pointwise %u->%u [1][1140]", sh[0], sh[1]);
            bench(name, c, 2.0 * 1140 * sh[0] * sh[1], 0);
        }
        {
            Case c;
            unsigned o = c.out(1140 * 256), x = c.f32(1140 * 256, 2.f, 0.5f), g = c.f32(256, 0.5f, 1.f), b = c.f32(256, 0.2f);
            c.in.op = PLOW_DOP_LAYERNORM_F32;
            c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = g; c.in.t[3] = b;
            c.in.i[0] = 1140; c.in.i[1] = 256; c.in.i[2] = 0; c.in.fj[0].f = 1e-5f;
            bench("LayerNormF32 double 1140x256", c, 0, 8.0 * 1140 * 256);
        }
        {
            Case c;
            const unsigned batch = 2, rows = 570, heads = 8, hw = 64, width = heads * hw, stride = 3 * width;
            unsigned o = c.out((size_t)batch * rows * width), q = c.f32((size_t)batch * rows * stride, 1.f);
            c.in.op = PLOW_DOP_ATTENTION_F32;
            c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = q; c.in.t[3] = q;
            c.in.i[0] = batch; c.in.i[1] = rows; c.in.i[2] = rows; c.in.i[3] = heads; c.in.i[4] = hw; c.in.i[5] = stride;
            c.in.fj[0].f = 0.125f; c.in.fj[1].u = width; c.in.fj[2].u = 2 * width;
            bench("AttentionF32 8x64 2x570 fused", c, 4.0 * batch * heads * hw * rows * rows, 0);
            c.in.i[6] = 2;
            bench("AttentionF32 tc 8x64 2x570 fused", c, 4.0 * batch * heads * hw * rows * rows, 0);
        }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--bench-dense")) {
        for (unsigned m : {104u, 390u}) {
            bench_dense(m, 1024, 1024, 1 | 4 | 8);
            bench_dense(m, 4096, 1024, 2 | 4 | 8);
            bench_dense(m, 1024, 4096, 1 | 4 | 8);
        }
        bench_dense(104, 1024, 7680, 1 | 4 | 8);
        bench_dense(1, 64, 64, 1 | 4 | 8);

        bench_conv(8, 128, 100, 1, 480);
        bench_conv(8, 128, 100, 1, 480, 512);
        bench_conv(8, 64, 50, 480, 480);
        bench_conv(8, 64, 50, 480, 480, 256);
        bench_conv(8, 32, 25, 480, 480, 256);
        bench_conv(30, 64, 50, 480, 480, 256);
        for (unsigned rows : {104u, 390u}) {
            Case c;
            unsigned o = c.out((size_t)rows * 1024), x = c.f32((size_t)rows * 1024, 2.f, 0.5f);
            unsigned g = c.f32(1024, 0.5f, 1.f), b = c.f32(1024, 0.2f);
            c.in.op = PLOW_DOP_LAYERNORM_F32;
            c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = g; c.in.t[3] = b;
            c.in.i[0] = rows; c.in.i[1] = 1024; c.in.i[2] = 3; c.in.fj[0].f = 1e-5f;
            char name[64];
            snprintf(name, sizeof(name), "LayerNormF32 ordered bf16 %ux1024", rows);
            bench(name, c, 0, 8.0 * rows * 1024);
        }
        for (unsigned rows : {104u, 390u}) {
            Case c;
            const size_t n = (size_t)rows * 1024;
            unsigned o = c.out(n), q = c.f32(n, 1.f), k = c.f32(n, 1.f), v = c.f32(n, 1.f);
            unsigned vr = c.u32({rows - 10});
            c.in.op = PLOW_DOP_GROUPED_ATTENTION_F32;
            c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = vr;
            c.in.i[0] = rows; c.in.i[1] = 1024; c.in.i[2] = 64; c.in.i[3] = 104; c.in.i[4] = 7;
            char name[64];
            snprintf(name, sizeof(name), "GroupedAttentionF32 qwen %ux1024 g104", rows);
            bench(name, c, 4.0 * rows * 104 * 1024, 0);
        }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--bench-qwen-asr")) {
        /* The Qwen3-ASR encoder's instructions at a 64-chunk bucket (832 rows) with the packet's
         * flags; argv[2] = d_model (896: 0.6B, 1024: 1.7B), argv[3] = bucket rows (13 per chunk).
         * Requests of 7 chunks (91 rows) set the reference split and the attention group table. */
        const unsigned d = argc > 2 ? (unsigned)atoi(argv[2]) : 1024u, M = argc > 3 ? (unsigned)atoi(argv[3]) : 832u;
        const unsigned B = M / 13u;
        std::vector<uint32_t> rows(M), table{0};
        for (unsigned r0 = 0; r0 < M; r0 += 91u) {
            const unsigned n = std::min(91u, M - r0);
            for (unsigned r = r0; r < r0 + n; r++) rows[r] = n;
            table[0]++;
            table.push_back(r0);
            table.push_back(n);
        }
        auto dense = [&](const char* name, unsigned n, unsigned k, unsigned flags, bool bias) {
            Case c;
            unsigned o = c.out((size_t)M * n), x = c.f32((size_t)M * k, 1.f), w = c.bf16((size_t)n * k, 0.05f);
            unsigned b = bias ? c.f32(n, 0.1f) : PLOW_TENSOR_NONE, r = c.u32(rows), sc = splitk_scratch(c);
            c.in.op = PLOW_DOP_DENSE_GEMM_F32;
            c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = w; c.in.t[3] = b; c.in.t[4] = r; c.in.t[5] = sc;
            c.in.i[0] = M; c.in.i[1] = n; c.in.i[2] = k; c.in.i[7] = flags; c.in.fj[1].u = (unsigned)g_nblk;
            char full[96];
            snprintf(full, sizeof(full), "DenseGemmF32 %s %ux%ux%u f%u", name, M, n, k, flags);
            bench(full, c, 2.0 * M * n * k, 0);
        };
        bench_conv(B, 128, 100, 1, 480, 512);
        bench_conv(B, 64, 50, 480, 480, 256);
        bench_conv(B, 32, 25, 480, 480, 256);
        {
            Case c;
            unsigned o = c.out((size_t)M * 480 * 16), x = c.f32((size_t)B * 480 * 16 * 13, 1.f);
            c.in.op = PLOW_DOP_PACK_NCFW_ROWS_F32; c.in.t[0] = o; c.in.t[1] = x;
            c.in.i[0] = M; c.in.i[1] = 480; c.in.i[2] = 16; c.in.i[3] = 13; c.in.i[4] = B;
            bench("PackNcfwRowsF32 Mx480x16", c, 0, 8.0 * M * 480 * 16);
        }
        dense("conv_out", d, 7680, 141, false);
        dense("q/k/v/o", d, d, 141, true);
        dense("fc1 gelu", 4 * d, d, 143, true);
        dense("fc2", d, 4 * d, 141, true);
        {
            Case c;
            const size_t n = (size_t)M * d;
            unsigned o = c.out(n), a = c.f32(n, 3.f), b = c.f32(n, 3.f);
            c.in.op = PLOW_DOP_SCALED_ADD_F32; c.in.t[0] = o; c.in.t[1] = a; c.in.t[2] = b;
            c.in.i[0] = (unsigned)n; c.in.i[1] = 1; c.in.fj[0].f = 1.f;
            bench("ScaledAddF32 bf16 Mxd", c, 0, 12.0 * n);
        }
        {
            Case c;
            unsigned o = c.out((size_t)M * d), x = c.f32((size_t)M * d, 2.f, 0.5f);
            unsigned g = c.f32(d, 0.5f, 1.f), b = c.f32(d, 0.2f);
            c.in.op = PLOW_DOP_LAYERNORM_F32;
            c.in.t[0] = o; c.in.t[1] = x; c.in.t[2] = g; c.in.t[3] = b;
            c.in.i[0] = M; c.in.i[1] = d; c.in.i[2] = 3; c.in.fj[0].f = 1e-5f;
            bench("LayerNormF32 ordered bf16 Mxd", c, 0, 8.0 * M * d);
        }
        {
            Case c;
            const size_t n = (size_t)M * d;
            unsigned o = c.out(n), q = c.f32(n, 1.f), k = c.f32(n, 1.f), v = c.f32(n, 1.f), vr = c.u32(table);
            c.in.op = PLOW_DOP_GROUPED_ATTENTION_F32;
            c.in.t[0] = o; c.in.t[1] = q; c.in.t[2] = k; c.in.t[3] = v; c.in.t[4] = vr;
            c.in.i[0] = M; c.in.i[1] = d; c.in.i[2] = 64; c.in.i[3] = 104; c.in.i[4] = 15;
            bench("GroupedAttentionF32 Mxd g104 packed", c, 4.0 * M * 104 * d, 0);
        }
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--test-qwen-asr")) {
        /* The Qwen3-ASR encoder's ops only, for an object built with the packet's PLOW_SPEECH_OPS. */
        t_dense("DenseGemmF32 qwen q 300x1024x1024", 300, 1024, 1024, 1 | 4, 0, 0, 0, 0, true);
        t_dense("DenseGemmF32 gelu 200x512x384", 200, 512, 384, 2 | 4, 0, 0, 0, 0, true);
        t_dense("DenseGemmF32 tc-bf16 qwen q 104x1024x1024 splitk", 104, 1024, 1024, 1 | 4 | 8, 0, 0, 0, 0, true, true);
        t_dense("DenseGemmF32 tc-bf16 fc2 104x1024x4096 splitk", 104, 1024, 4096, 1 | 4 | 8, 0, 0, 0, 0, true, true);
        t_dense_ref(1024, 1024, {104, 52, 156, 13, 390, 208});
        t_dense_ref(1024, 4096, {104, 52, 260});
        t_dense_ref(4096, 1024, {104, 52});
        t_dense_ref(1024, 7680, {104, 52, 156});
        t_dense_ref(896, 3584, {91, 13, 26, 39, 52});
        t_dense_ref(896, 896, {91, 65, 13, 130});
        t_dense_ref(896, 7680, {91, 91, 26});
        t_dense_ref(896, 3584, {91, 91, 91, 91, 91, 91, 91, 91, 91, 13});
        t_dense_ref(896, 896, {91, 65, 13, 39});
        t_dense_ref(3584, 896, {91, 91, 26});
        t_conv("Conv2dF32 tc-3xtf32 qwen stage1", 2, 128, 100, 1, 48, 3, 2, 1, 1, 512 | 64 | 128 | (2 << 2) | (2 << 4));
        t_conv("Conv2dF32 tc-bf16 qwen stage2", 2, 64, 50, 48, 64, 3, 2, 1, 1, 256 | 64 | 128 | (2 << 2) | (2 << 4));
        t_conv("Conv2dF32 tc-bf16 wide multi-tile", 8, 64, 50, 48, 320, 3, 2, 1, 1, 256 | 64 | 128 | (2 << 2) | (2 << 4));
        {
            const uint32_t seed = rng_s;
            t_conv("Conv2dF32 tc-bf16 qwen stage3 480", 3, 16, 25, 480, 480, 3, 2, 1, 1, 256 | 64 | 128 | (2 << 2) | (2 << 4));
            rng_s = seed;
            t_conv("Conv2dF32 ffma qwen stage3 480 (same data)", 3, 16, 25, 480, 480, 3, 2, 1, 1, 64 | 128 | (2 << 2) | (2 << 4));
        }
        t_layernorm("LayerNormF32 ordered bf16", 100, 1024, 3, true);
        t_layernorm("LayerNormF32 ordered bf16 d896", 156, 896, 3, true);
        t_layernorm("LayerNormF32 ordered ragged", 45, 1000, 2, true);
        t_layernorm("LayerNormF32 ordered odd width", 37, 1001, 2, true);
        t_layernorm("LayerNormF32 ordered bf16 two chunks", 19, 2050, 3, true);
        t_grouped("GroupedAttentionF32 qwen 300 v290", 300, 1024, 64, 104, 7, 290);
        t_grouped_table("GroupedAttentionF32 qwen packed table", 13 * 19, {4, 0, 104, 104, 36, 143, 35, 182, 60});
        t_grouped_table("GroupedAttentionF32 qwen packed table 10 ragged", 832,
                        {10, 0, 104, 104, 104, 208, 104, 312, 52, 364, 104, 468, 91, 559, 67, 626, 37, 663, 27, 690, 12});
        printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--bench-only")) {
        bench_signal();
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--bench-asr")) {
        bench_q8_paths();
        bench_relative_items(128, 60, 56);
        bench_relative_items(32, 88, 56);
        return 0;
    }
    t_dense("DenseGemmF32 qwen q 300x1024x1024", 300, 1024, 1024, 1 | 4, 0, 0, 0, 0, true);
    t_dense("DenseGemmF32 gelu 200x512x384", 200, 512, 384, 2 | 4, 0, 0, 0, 0, true);
    t_dense("DenseGemmF32 f32w relu row0", 77, 130, 70, 0, 1, 3, 0, 0, true);
    t_dense("DenseGemmF32 f32w onehot", 45, 67, 33, 0, 0, 0, 35, 34, false);
    t_dense("DenseGemmF32 bf16w onehot bf16", 45, 67, 36, 1 | 4, 0, 0, 37, 36, true);
    t_gemm_bf16(33, 70, 96);
    t_dense_ref(1024, 1024, {104, 52, 156, 13, 390, 208});
    t_dense_ref(1024, 1024, {104, 52, 156, 13, 390, 208, 390, 390, 260});
    t_dense_ref(1024, 4096, {104, 52, 260});
    t_dense_ref(4096, 1024, {104, 52});
    t_dense_ref(1024, 7680, {104, 52, 156});
    t_q8(70, 130, 512, 0);
    t_q8(20, 96, 256, 1);
    t_q8_path(1, 4, 20, 96, 256);
    t_q8_path(1, 5, 37, 130, 512);
    /* A batched joint's rows: GEMV passes of 32 rows (32, 32, 6) against an 8-row launch. */
    t_q8_path(1, 8, 70, 640, 640);
    /* Passes of 8 (odd n), 16, 32 + 8 rows (several columns per warp), 16 + 8 at k 1024, 8 filling
     * the arena; then the untransposed walk, staged and from global. */
    t_q8_gemv_order(1, 300, 640);
    t_q8_gemv_order(8, 301, 640);
    t_q8_gemv_order(16, 300, 640);
    t_q8_gemv_order(37, 2701, 640);
    t_q8_gemv_order(20, 150, 1024);
    t_q8_gemv_order(8, 100, 2048);
    t_q8_gemv_order(4, 200, 4096);
    t_q8_gemv_order(5, 96, 8192);
    t_q8_tiles(200, 300, 256);
    t_q8_tiles(150, 260, 1024);
    t_q8_path(2, 4, 70, 130, 512);
    /* Nemotron's stream window: 14 chunks of 4 rows of keys, then the step's 4 query rows. */
    t_relative_items("RelativeAttentionF32 stream items", 60, 1024, 8, 4, 14, 56, {56, 52, 0, 8, 56});
    /* An upload stream's window: the same 56 rows of keys, then 32 query rows. */
    t_relative_items("RelativeAttentionF32 upload items", 88, 1024, 8, 4, 14, 56, {56, 24, 0, 8, 56});
    t_relative_items("RelativeAttentionF32 chunked items", 40, 256, 4, 8, 2, 0, {0, 3, 17});
    t_relative_items("RelativeAttentionF32 full items", 21, 640, 2, 1, 0xFFFFFFFFu, 0, {21, 9, 15});
    t_dwconv_items(5, 12, 1024, 9);
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
    t_grouped_table("GroupedAttentionF32 qwen packed table", 13 * 19,
                    {4, 0, 104, 104, 36, 143, 35, 182, 60});
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
    t_conv_pinned();
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
    /* Tensor-core GEMM paths (after the fixed-seed cases above, so their data is unchanged). */
    t_vec_elementwise();
    t_row_stats("RowStatsF32 ordered 104x1024", 104, 1024, 3);
    t_row_stats("RowStatsF32 double 1140x256", 1140, 256, 0);
    t_ln_gemm("DenseGemmF32 LN-prologue tc-bf16 qwen q", 104, 1024, 1024, 3, 1 | 4 | 8, 0);
    t_ln_gemm("DenseGemmF32 LN-prologue tc-bf16 gelu fc1", 104, 4096, 1024, 3, 2 | 4 | 8, 0);
    t_ln_gemm("DenseGemmF32 LN-prologue ffma bf16w row0", 77, 130, 72, 3, 1 | 4, 3);
    t_ln_gemm("DenseGemmF32 LN-prologue 3xtf32 s3gen", 1140, 256, 256, 0, 16, 0);
    t_ln_gemm("DenseGemmF32 LN-prologue ffma f32w", 60, 100, 256, 0, 0, 2);
    {
        ConvSpec rs[] = {
            {2, 1024, 128, 128, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, true, {1024, 700}, 0.f, true, true},
            {2, 300, 64, 64, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, true, {}, 0.f, false, true},
            {2, 256, 64, 64, 7, 1, 3, 64, 9, 9, 0, 12, 0, false, true, true, true, {}, 0.f, false, true},
        };
        t_conv1d("Conv1dF32 row_scale tc pointwise +res len", rs[0]);
        t_conv1d("Conv1dF32 row_scale ffma pointwise +res", rs[1]);
        t_conv1d("Conv1dF32 row_scale depthwise snake +res", rs[2]);
    }
    {
        /* SNAC residual unit: the depthwise conv applies the pointwise conv's snake too (bit 19). */
        for (unsigned dil : {1u, 3u, 9u}) {
            char name[64];
            snprintf(name, sizeof(name), "Conv1dF32 depthwise k7 d%u snake->snake dual len", dil);
            ConvSpec d = {2, 2048, 512, 512, 7, 1, dil, 512, 3 * dil, 3 * dil, 0, 12, 12, false, true, true, false,
                          {2048, 1501}, 0.f};
            d.dual = true;
            t_conv1d(name, d);
        }
        ConvSpec r = {2, 300, 64, 64, 7, 1, 3, 64, 9, 9, 0, 12, 12, false, true, true, true, {300, 211}, 0.f};
        r.dual = true;
        t_conv1d("Conv1dF32 depthwise snake->snake dual +res", r);
        ConvSpec t = {2, 257, 64, 96, 3, 1, 1, 1, 1, 1, 0, 12, 12, false, true, true, false, {257, 100}, 0.f, true};
        t.dual = true;
        t_conv1d("Conv1dF32 dense k3 snake->snake dual tc", t);
    }
    t_attention("AttentionF32 tc 8x64 2x1200 lengths", 2, 1200, 8, 64, {1200, 1000}, false, false, false, true);
    t_attention("AttentionF32 tc 8x64 causal bias fused", 2, 300, 8, 64, {300, 217}, true, true, true, true);
    t_attention("AttentionF32 tc 4x64 ragged rows", 1, 77, 4, 64, {}, false, false, false, true);
    t_attention("AttentionF32 tc s3gen 8x64 2x570 fused", 2, 570, 8, 64, {}, false, false, true, true);
    t_attention("AttentionF32 tc 8x64 causal bias 2x72", 2, 72, 8, 64, {72, 50}, true, true, true, true);
    t_attention_prefix("AttentionF32 prefix 306+72 lengths", 4, 72, 306, 2, {0, 1, 0, 1}, {378, 350, 320, 306}, false);
    t_attention_prefix("AttentionF32 tc prefix 306+72 lengths", 4, 72, 306, 2, {0, 1, 0, 1}, {378, 350, 320, 306}, true);
    t_attention_prefix("AttentionF32 tc prefix 306+136", 2, 136, 306, 2, {1, 0}, {}, true);
    t_attention_rel("AttentionF32 tc relative table 3x8 t120 lengths", 3, 120, 8, {120, 80, 7});
    t_attention_rel("AttentionF32 tc relative table 2x8 t150 lengths", 2, 150, 8, {150, 99});
    /* Split-bf16 wide tiles (~2^-17 per product) and tap-major weights. */
    {
        ConvSpec w1{2, 4200, 256, 512, 1, 1, 1, 1, 0, 0, 0, 0, 11, false, true, false, true, {4200, 3000}, 0.f, true};
        w1.wide = true;
        t_conv1d("Conv1dF32 wide pointwise gelu +res len", w1, 1e-4);
        ConvSpec w2{16, 570, 256, 256, 3, 1, 1, 1, 2, 0, 0, 9, 10, false, true, false, false,
                    {570, 400, 570, 1, 570, 570, 300, 570, 570, 570, 2, 570, 570, 569, 570, 100}, 0.1f, true};
        w2.wide = w2.tapmajor = true;
        t_conv1d("Conv1dF32 wide k3 tap-major leaky->mish len", w2, 1e-4);
        ConvSpec w3{4, 8192, 64, 64, 11, 1, 3, 1, 15, 15, 0, 12, 0, false, true, true, true, {8192, 5000, 8192, 100},
                    0.f, true};
        w3.wide = w3.tapmajor = true;
        t_conv1d("Conv1dF32 wide k11 d3 snake 64 tap-major +res", w3, 1e-4);
        ConvSpec w4{2, 100, 64, 96, 5, 1, 2, 1, 4, 4, 0, 0, 7, false, true, false, false, {}, 0.f, true};
        w4.wide = w4.tapmajor = true;
        t_conv1d("Conv1dF32 wide tap-major k5 small M", w4, 1e-4);
        ConvSpec w6{4, 8192, 64, 64, 7, 1, 3, 1, 9, 9, 0, 0, 12, false, true, true, false, {8192, 5000, 1, 8192}, 0.f, true};
        w6.wide = w6.tapmajor = true;
        t_conv1d("Conv1dF32 wide k7 d3 snake-out tap-major len", w6, 1e-4);
#if defined(PLOW_NV_HOPPER)
        ConvSpec g1 = w1;
        g1.wg = true;
        t_conv1d("Conv1dF32 wide wgmma pointwise gelu +res len", g1, 1e-4);
        ConvSpec g2{1, 1000, 256, 392, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true};
        g2.wide = g2.wg = true;
        t_conv1d("Conv1dF32 wide wgmma pointwise 256->392 ragged", g2, 1e-4);
        ConvSpec g3{3, 150, 320, 256, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, true, {150, 77, 3}, 0.f, true};
        g3.wide = g3.wg = true;
        t_conv1d("Conv1dF32 wide wgmma pointwise +res len 3 items", g3, 1e-4);
        for (ConvSpec* w : {&w2, &w3, &w4, &w6}) {
            ConvSpec g = *w;
            g.wg = true;
            char name[96];
            snprintf(name, sizeof(name), "Conv1dF32 wide wgmma k%u d%u pre%u post%u", g.kernel, g.dil_or_opad, g.pre, g.post);
            t_conv1d(name, g, 1e-4);
        }
        ConvSpec g5{3, 700, 128, 192, 3, 1, 5, 1, 5, 5, 0, 9, 12, false, true, true, true, {700, 333, 64}, 0.1f, true};
        g5.wide = g5.tapmajor = g5.wg = true;
        t_conv1d("Conv1dF32 wide wgmma k3 d5 leaky->snake +res len", g5, 1e-4);
        ConvSpec g6{64, 136, 320, 256, 3, 1, 1, 1, 2, 0, 0, 0, 0, false, true, false, false, std::vector<uint32_t>(64, 136), 0.f, true};
        for (unsigned i = 0; i < 64; i += 3) g6.lengths[i] = 40 + i;
        g6.wide = g6.tapmajor = g6.wg = true;
        t_conv1d("Conv1dF32 wide wgmma k3 causal 320->256 len (CFM c1)", g6, 1e-4);
        /* Split-K (tiles * 2 <= blocks, post 0): S parts per tile folded in part order, one-k-tile
         * parts, odd chunks, a K tail, short lengths. */
        struct SK { const char* name; ConvSpec s; };
        SK sks[] = {
            {"Conv1dF32 wgmma split-K k7 768 M=1", {1, 1, 768, 768, 7, 1, 1, 1, 6, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true}},
            {"Conv1dF32 wgmma split-K k7 d3 +res len M=34", {2, 17, 384, 384, 7, 1, 3, 1, 18, 0, 0, 0, 0, false, true, false, true, {17, 9}, 0.f, true}},
            {"Conv1dF32 wgmma split-K pointwise K tail M=32", {4, 8, 1000, 512, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true}},
            {"Conv1dF32 wgmma split-K k3 704->768 leaky-in chunk 3", {1, 5, 704, 768, 3, 1, 1, 1, 2, 0, 0, 9, 0, false, true, false, true, {5}, 0.1f, true}},
        };
        for (SK& t : sks) {
            t.s.wide = t.s.tapmajor = t.s.wg = t.s.splitk = true;
            t_conv1d(t.name, t.s, 1e-4);
        }
        ConvSpec g7{1, 1000, 512, 8 * 380, 1, 1, 1, 8, 0, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true};
        g7.wide = g7.wg = true;
        t_conv1d("Conv1dF32 wgmma grouped pointwise g8 64->380 (relpos)", g7, 1e-4);
        ConvSpec g8{2, 300, 256, 4 * 68, 1, 1, 1, 4, 0, 0, 0, 0, 0, false, true, false, true, {300, 123}, 0.f, true};
        g8.wide = g8.wg = true;
        t_conv1d("Conv1dF32 wgmma grouped pointwise g4 +res len", g8, 1e-4);
        /* Merged-phase ConvTranspose1d (kernel 2*stride, phase-major): SNAC's four upsamplers,
         * Fish's causal crop, T=1, lengths 0/1/T-1, no bias. */
        struct CT { const char* name; ConvSpec s; };
        const CT cts[] = {
            {"ConvT wgmma s8 1024->512 crop4 len", {2, 33, 1024, 512, 16, 8, 0, 1, 4, 4, 0, 0, 0, false, true, false, false, {33, 17}, 0.f, true}},
            {"ConvT wgmma s8 512->256 T1", {4, 1, 512, 256, 16, 8, 0, 1, 4, 4, 0, 0, 0, false, true, false, false, {}, 0.f, true}},
            {"ConvT wgmma s4 256->128 len 0/1/T-1", {4, 200, 256, 128, 8, 4, 0, 1, 2, 2, 0, 0, 0, false, true, false, false, {200, 0, 1, 199}, 0.f, true}},
            {"ConvT wgmma s2 128->64 len", {2, 1000, 128, 64, 4, 2, 0, 1, 1, 1, 0, 0, 0, false, true, false, false, {1000, 513}, 0.f, true}},
            {"ConvT wgmma s8 causal 512->256 nobias", {3, 64, 512, 256, 16, 8, 0, 1, 0, 8, 0, 0, 0, false, false, false, false, {64, 63, 5}, 0.f, true}},
            {"ConvT wgmma s4 big M 256->128", {32, 257, 256, 128, 8, 4, 0, 1, 2, 2, 0, 0, 0, false, true, false, false, {}, 0.f, true}},
        };
        /* The GPU reads a weight copy whose f32 part is NaN (golden keeps the clean one), so a
         * fallback off the split image fails. */
        for (const CT& t : cts) {
            ConvSpec g = t.s;
            g.wide = g.tapmajor = g.wg = true;
            Case c; unsigned o;
            build_conv(c, g, true, o);
            const PlowDevInst ref = c.in;
            const unsigned w = c.in.t[2], wp = c.add(c.host[w].size());
            c.host[wp] = c.host[w];
            const float nan = NAN;
            for (size_t i = 0; i < (size_t)g.stride * g.cout * 2 * g.cin; i++) memcpy(&c.host[wp][i * 4], &nan, 4);
            c.in.t[2] = wp;
            run(t.name, c, g_conv_transpose1d_f32, {o}, FP32, 1e-4, 0, nullptr, &ref);
        }
#endif
        /* Batch invariance of the Qwen3-TTS codec's convs without arms (its recipe drops the
         * per-size arm table): wgmma taps / pointwise / merged-phase ConvT, the wide tile (96
         * channels), the direct form (cout 1), 3xTF32 phase-major ConvT; small and large M. */
        struct BI { const char* name; ConvSpec s; bool transpose; };
        BI bis[] = {
            {"Conv1dF32 inv pointwise 4096->1024", {9, 8, 4096, 1024, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, true, {}, 0.f, true}, false},
            {"Conv1dF32 inv k7 d3 384 len", {7, 19, 384, 384, 7, 1, 3, 1, 18, 0, 0, 0, 0, false, true, false, false, {19, 4, 19, 1, 19, 19, 7}, 0.f, true}, false},
            {"Conv1dF32 inv pointwise 512 big M", {40, 500, 512, 512, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true}, false},
            {"Conv1dF32 inv k3 512->1024 big M", {20, 500, 512, 1024, 3, 1, 1, 1, 2, 0, 0, 0, 0, false, true, false, true, {}, 0.f, true}, false},
            {"Conv1dF32 inv wide k7 96 +res", {40, 60, 96, 96, 7, 1, 1, 1, 6, 0, 0, 0, 0, false, true, false, true, {}, 0.f, true}, false},
            {"Conv1dF32 inv direct k7 96->1", {40, 300, 96, 1, 7, 1, 1, 1, 6, 0, 0, 0, 0, false, true, false, false, {}, 0.f, true}, false},
            {"ConvT inv wgmma s8 1024->512", {12, 9, 1024, 512, 16, 8, 0, 1, 4, 4, 0, 0, 0, false, true, false, false, {}, 0.f, true}, true},
            {"ConvT inv wgmma s4 256->128 big M", {64, 200, 256, 128, 8, 4, 0, 1, 2, 2, 0, 0, 0, false, true, false, false, {}, 0.f, true}, true},
            {"ConvT1dF32 inv tc phase-major k6 s3", {40, 300, 192, 96, 6, 3, 0, 1, 3, 3, 0, 0, 0, false, true, false, false, {}, 0.f, true}, true},
        };
#if defined(PLOW_NV_HOPPER)
        const bool hopper = true;
#else
        const bool hopper = false;
#endif
        for (BI& t : bis) {
            /* The merged-phase wgmma ConvT needs kernel 2*stride and cout % 64; the k6 s3 one is 3xTF32. */
            t.s.tapmajor = true;
            t.s.wg = hopper && t.s.cout > 1u && (!t.transpose || (t.s.kernel == 2u * t.s.stride && t.s.cout % 64u == 0));
            t.s.wide = t.s.wg || (!t.transpose && t.s.cout > 1u);
            t_conv_invariant(t.name, t.s, t.transpose);
        }
        /* Rows arm (bit 16, M <= 32): M=17 and 32, partial 8-channel blocks, K tails past a 512
         * chunk, short lengths, M=1; M=33 and cout <= 4 fall back to the warp arm. */
        struct RA { const char* name; ConvSpec s; };
        RA ras[] = {
            {"Conv1dF32 rows k7 snake-in 256->203 +res M=17", {1, 17, 256, 203, 7, 1, 1, 1, 6, 0, 0, 12, 0, false, true, true, true, {}, 0.f}},
            {"Conv1dF32 rows k7 d3 100->50 leaky len M=32", {2, 16, 100, 50, 7, 1, 3, 1, 18, 0, 0, 0, 9, false, true, false, true, {16, 5}, 0.1f}},
            {"Conv1dF32 rows k7 1024->1536 M=1", {1, 1, 1024, 1536, 7, 1, 1, 1, 6, 0, 0, 0, 0, false, true, false, false, {}, 0.f}},
            {"Conv1dF32 rows M=33 cout 3 -> warp", {1, 33, 64, 3, 7, 1, 1, 1, 6, 0, 0, 0, 1, false, true, false, false, {}, 0.f}},
        };
        for (RA& t : ras) {
            t.s.rows = t.s.tapmajor = true;
            t_conv1d(t.name, t.s);
        }
        /* Lanes arm (bit 16, cout <= 4, tap-major, cg % 32 == 0): multi-tile items, reflect and
         * replicate padding, short lengths, stride and dilation. */
        RA las[] = {
            {"Conv1dF32 lanes k7 snake-in 96->1 reflect len", {3, 700, 96, 1, 7, 1, 1, 1, 3, 3, 1, 12, 0, false, true, true, true, {700, 451, 9}, 0.f}},
            {"Conv1dF32 lanes k7 d3 s2 leaky 64->3 replicate", {2, 1100, 64, 3, 7, 2, 3, 1, 9, 9, 2, 9, 0, false, true, false, false, {1100, 530}, 0.1f}},
            {"Conv1dF32 lanes k3 128->4 zero", {1, 520, 128, 4, 3, 1, 1, 1, 1, 1, 0, 0, 0, false, false, false, false, {}, 0.f}},
        };
        for (RA& t : las) t_conv1d_lanes(t.name, t.s);
        const int nblk = g_nblk;
        g_nblk = 2; /* several tiles per block */
        t_conv1d_lanes("Conv1dF32 lanes 96->2 asym pad nblk=2",
                       {4, 900, 96, 2, 7, 1, 1, 1, 6, 0, 0, 12, 0, true, true, true, false, {900, 1, 0, 640}, 0.f});
        g_nblk = nblk;
        /* Channel-major weights at kernel 7 (the codec's final 96 -> 1 conv). */
        RA lcs[] = {
            {"Conv1dF32 lanes ch-major k7 snake 96->1 len", {3, 700, 96, 1, 7, 1, 1, 1, 3, 3, 0, 12, 0, false, true, true, false, {700, 451, 9}, 0.f}},
            {"Conv1dF32 lanes ch-major k7 d2 s3 64->4 refl", {2, 900, 64, 4, 7, 3, 2, 1, 6, 6, 1, 9, 0, false, true, false, true, {900, 333}, 0.1f}},
        };
        for (RA& t : lcs) t_conv1d_lanes(t.name, t.s, false);
        g_nblk = 3;
        t_conv1d_lanes("Conv1dF32 lanes ch-major 96->1 nblk=3",
                       {2, 1500, 96, 1, 7, 1, 1, 1, 6, 0, 0, 12, 0, true, true, true, false, {1500, 2}, 0.f}, false);
        g_nblk = nblk;
        ConvSpec w7{2, 300, 64, 64, 7, 1, 3, 1, 9, 9, 0, 0, 12, false, true, true, true, {300, 170}, 0.f, true};
        t_conv1d("Conv1dF32 k7 snake-out +res (64x64 tiles)", w7);
        ConvSpec w5{2, 1000, 64, 1, 7, 1, 1, 1, 3, 3, 0, 9, 1, false, true, false, false, {1000, 500}, 0.01f, true};
        w5.tapmajor = true;
        t_conv1d("Conv1dF32 tap-major narrow 64->1 (direct)", w5);
    }
    {
        ConvSpec p1{2, 128, 1024, 512, 16, 8, 0, 1, 4, 4, 0, 12, 0, false, true, true, false, {128, 77}, 0.f, true};
        p1.tapmajor = true;
        t_convt1d("ConvT1dF32 phase-major 1024->512 k16 s8 snake", p1);
        ConvSpec p2{8, 256, 128, 64, 7, 3, 0, 1, 2, 2, 0, 9, 0, false, true, false, true, {256, 100, 1, 256, 256, 256, 30, 256}, 0.1f, true};
        p2.tapmajor = true;
        t_convt1d("ConvT1dF32 phase-major k7 s3 leaky +res len", p2);
    }
    t_layernorm("LayerNormF32 double rows 9120x256", 9120, 256, 0, true);
    t_layernorm("LayerNormF32 double rows 1001x512 bf16", 1001, 512, 1, true);
    t_layernorm("LayerNormF32 rows Mish + row 9216x256", 9216, 256, 10u << 4, true, true);
    t_layernorm("LayerNormF32 rows Mish 1001x256", 1001, 256, 10u << 4, true);
    t_layernorm("LayerNormF32 rows GELU + row bf16 777x512", 777, 512, 11u << 4 | 1u, true, true);
    t_layernorm("LayerNormF32 double Mish + row 33x513", 33, 513, 10u << 4, true, true);
    {
        ConvSpec cs[] = {
            {2, 1024, 512, 512, 1, 1, 1, 1, 0, 0, 0, 0, 0, false, true, false, true, {}, 0.f, true},
            {2, 2048, 512, 512, 1, 1, 1, 1, 0, 0, 0, 12, 0, false, true, true, true, {2048, 1500}, 0.f, true},
            {2, 300, 64, 80, 7, 1, 3, 1, 9, 9, 1, 9, 1, true, true, false, false, {300, 211}, 0.1f, true},
            {2, 199, 64, 64, 5, 2, 1, 2, 2, 2, 2, 8, 0, false, false, false, false, {}, 0.f, true},
        };
        const char* names[] = {"Conv1dF32 tc pointwise 512 2k +res", "Conv1dF32 tc pointwise snake +res len",
                               "Conv1dF32 tc dense k7 reflect leaky->tanh len", "Conv1dF32 tc stride2 replicate groups2"};
        for (int i = 0; i < 4; i++) t_conv1d(names[i], cs[i]);
        ConvSpec ts[] = {
            {2, 128, 1024, 512, 16, 8, 0, 1, 4, 4, 0, 12, 0, false, true, true, false, {128, 77}, 0.f, true},
            {2, 60, 128, 64, 11, 5, 1, 1, 3, 2, 0, 9, 0, true, true, false, true, {}, 0.1f, true},
            {1, 50, 64, 96, 7, 3, 0, 2, 2, 2, 0, 0, 0, false, true, false, false, {}, 0.f, true},
        };
        const char* tnames[] = {"ConvT1dF32 tc 1024->512 k16 s8 snake", "ConvT1dF32 tc k11 s5 leaky f16 opad",
                                "ConvT1dF32 tc groups2 k7 s3"};
        for (int i = 0; i < 3; i++) t_convt1d(tnames[i], ts[i]);
    }
    t_conv("Conv2dF32 tc-3xtf32 qwen stage1", 2, 128, 100, 1, 48, 3, 2, 1, 1, 512 | 64 | 128 | (2 << 2) | (2 << 4));
    t_conv("Conv2dF32 tc-bf16 qwen stage2", 2, 64, 50, 48, 64, 3, 2, 1, 1, 256 | 64 | 128 | (2 << 2) | (2 << 4));
    t_conv("Conv2dF32 tc-bf16 wide multi-tile", 8, 64, 50, 48, 320, 3, 2, 1, 1, 256 | 64 | 128 | (2 << 2) | (2 << 4));
    t_conv("Conv2dF32 tc-bf16 wide NFWC out", 8, 64, 50, 48, 320, 3, 2, 1, 1, 256 | 64 | (0 << 2) | (2 << 4));
    {
        /* Same data through the FFMA path: its GELU flip count is the baseline the budget covers. */
        const uint32_t seed = rng_s;
        t_conv("Conv2dF32 tc-bf16 qwen stage3 480", 3, 16, 25, 480, 480, 3, 2, 1, 1, 256 | 64 | 128 | (2 << 2) | (2 << 4));
        rng_s = seed;
        t_conv("Conv2dF32 ffma qwen stage3 480 (same data)", 3, 16, 25, 480, 480, 3, 2, 1, 1, 64 | 128 | (2 << 2) | (2 << 4));
    }
    t_conv("Conv2dF32 tc-3xtf32 f32 NFCW->NFWC relu", 2, 17, 11, 12, 20, 3, 1, 1, 0, 512 | 64 | 2 | (1 << 4));
    t_dense("DenseGemmF32 tc-bf16 qwen q 104x1024x1024 splitk", 104, 1024, 1024, 1 | 4 | 8, 0, 0, 0, 0, true, true);
    t_dense("DenseGemmF32 tc-bf16 fc2 104x1024x4096 splitk", 104, 1024, 4096, 1 | 4 | 8, 0, 0, 0, 0, true, true);
    t_dense("DenseGemmF32 tc-bf16 390x4096x1024", 390, 4096, 1024, 1 | 4 | 8, 0, 0, 0, 0, true, true);
    t_dense("DenseGemmF32 tc-bf16 gelu 390x4096x1024", 390, 4096, 1024, 2 | 4 | 8, 0, 0, 0, 0, true, true);
    t_dense("DenseGemmF32 tc-bf16 conv_out 77x1024x7680", 77, 1024, 7680, 1 | 4 | 8, 0, 0, 0, 0, false, true);
    t_dense("DenseGemmF32 tc-bf16 nosplit ragged 45x67x36", 45, 67, 36, 1 | 4 | 8, 0, 0, 37, 36, true);
    t_dense("DenseGemmF32 tc-bf16 f32w relu row0", 77, 130, 70, 8, 1, 3, 0, 0, true, true);
    t_dense("DenseGemmF32 tc-3xtf32 f32w 300x520x700", 300, 520, 700, 16, 0, 0, 0, 0, true);
    t_dense("DenseGemmF32 tc-3xtf32 f32w splitk 50x130x2000", 50, 130, 2000, 16, 1, 2, 0, 0, true, true);
    t_dense("DenseGemmF32 tc-3xtf32 onehot", 45, 67, 33, 16, 0, 0, 35, 34, false);
    {
        /* Post chain (flags 0x100 clamp, 0x200 SiLU + round, 0x400 clamp_next). */
        static const float pc[4] = {-0.9f, 1.1f, -0.3f, 0.4f};
        t_dense("DenseGemmF32 tc-bf16 post all 250x4096x1024", 250, 4096, 1024, 1 | 4 | 8 | 0x700, 0, 0, 0, 0,
                false, true, pc);
        t_dense("DenseGemmF32 tc-bf16 post clamp splitk 250x1024x4096", 250, 1024, 4096, 1 | 4 | 8 | 0x100, 0, 0,
                0, 0, false, true, pc);
        t_dense("DenseGemmF32 bf16w post silu+clamp 45x67x36", 45, 67, 36, 1 | 4 | 0x600, 0, 0, 0, 0, true, false,
                pc);
    }
    /* last: keeps the earlier cases' random data */
#if defined(PLOW_NV_HOPPER)
    t_attention_prefix_h16("AttentionF32 h16 prefix 306+72 lengths", 4, 72, 306, 2, {0, 1, 0, 1}, {378, 350, 320, 306});
    t_attention_prefix_h16("AttentionF32 h16 prefix 306+136", 2, 136, 306, 2, {1, 0}, {});
    t_attention_prefix_h16("AttentionF32 h16 prefix 300+264 lengths", 3, 264, 300, 3, {2, 0, 1}, {564, 400, 300});
#endif
    t_stream_tail();
    t_reduce_rows();
    t_length_affine();
    t_attention_stream("attn stream L127 n17 w128", 4, 127, 17, 128, {0, 5, 127, 4000}, {144, 140, 130, 144}, false);
    t_attention_stream("attn stream L127 n17 w128 tc", 3, 127, 17, 128, {0, 1, 300}, {}, true);
    t_attention_stream("attn stream L71 n80 w72", 2, 71, 80, 72, {0, 1000}, {151, 100}, false);
    t_attention_stream("attn stream L127 n1 w128", 5, 127, 1, 128, {0, 1, 126, 127, 128}, {}, false);
    /* Voice-conditioning extensions (after the fixed-seed cases, so their data is unchanged). */
    t_lstm_rows();
    t_trim_silence();
#if defined(PLOW_NV_HOPPER) && PLOW_SPEECH_SPLIT_A
    t_split_image();
#endif
    t_attention_w256("AttentionF32 w256 4x256 q32 kv150 lengths", 3, 32, 150, 4, {150, 97, 1}, false, false, 0);
    t_attention_w256("AttentionF32 w256 q70 kv150 unaligned", 2, 70, 150, 4, {}, false, false, 1);
    t_attention_w256("AttentionF32 w256 causal bias", 2, 100, 100, 2, {100, 61}, true, true, 0);
    t_conv("Conv2dF32 fcm 3x3 s(1,2) ffma", 2, 150, 80, 32, 32, 3, 1, 1, 1, 64 | 2 | (2u << 16));
    t_conv("Conv2dF32 fcm 3x3 s(1,2) tc-3xtf32 NFCW", 1, 150, 80, 32, 32, 3, 1, 1, 1, 512 | 64 | (1 << 2) | (2u << 16));
    t_conv("Conv2dF32 fcm 1x1 s(1,2) shortcut tc-3xtf32", 1, 150, 80, 32, 32, 1, 1, 0, 0, 512 | 64 | (2u << 16));
    t_conv("Conv2dF32 3x3 s(2,1) tc-bf16 wide", 2, 64, 50, 48, 320, 3, 2, 1, 1, 256 | 64 | (2 << 2) | (1u << 16));
    t_conv("Conv2dF32 depthwise s(1,3) f16 NCFW in", 2, 17, 31, 24, 24, 3, 1, 1, 1, 1 | (2 << 4) | (3u << 16));
    /* Per-item prefix rows (AttentionF32 flag bit 5) on every route, and CopyColsF32 groups. */
    {
        const std::vector<uint32_t> idx = {0, 1, 0, 2, 1, 0, 2, 0};
        const std::vector<uint32_t> sweep45 = {0, 1, 15, 16, 17, 44, 45, 900};
        const std::vector<uint32_t> sweep48 = {17, 0, 47, 1, 48, 16, 15, 2};
        /* cutting inside the own region, inside the prefix, and past the end */
        const std::vector<uint32_t> len45 = {3, 1, 40, 16, 20, 60, 47, 90};
        const std::vector<uint32_t> full = {45, 45, 45, 1000, 45, 45, 45, 45};
        struct R { const char* name; int route; unsigned q, hw; };
        const R routes[] = {
            {"fp32 hw64", 0, 40, 64},   {"fp32 hw128", 0, 20, 128}, {"fp32 ragged hw72", 0, 33, 72},
            {"tc64", 1, 16, 64},        {"tc64p<16,5>", 1, 40, 64}, {"tc64p<24,3>", 1, 72, 64},
            {"tc64p<32,2>", 1, 136, 64},
#if defined(PLOW_NV_HOPPER)
            {"h16<3>", 2, 72, 64},      {"h16<2>", 2, 136, 64},
#endif
        };
        for (const R& r : routes) {
            char name[96];
            snprintf(name, sizeof(name), "pre_b %s i7=45 sweep", r.name);
            t_attention_prefix_rows(name, r.route, 8, r.q, r.hw, 45, 3, idx, sweep45, {});
            snprintf(name, sizeof(name), "pre_b %s i7=45 lengths", r.name);
            t_attention_prefix_rows(name, r.route, 8, r.q, r.hw, 45, 3, idx, sweep45, len45);
            snprintf(name, sizeof(name), "pre_b %s i7=48 mixed", r.name);
            t_attention_prefix_rows(name, r.route, 8, r.q, r.hw, 48, 3, {2, 2, 0, 1, 1, 0, 2, 0}, sweep48, {});
            snprintf(name, sizeof(name), "pre_b %s all zero", r.name);
            t_attention_prefix_rows(name, r.route, 8, r.q, r.hw, 45, 3, idx, std::vector<uint32_t>(8, 0), len45);
            snprintf(name, sizeof(name), "pre_b %s full", r.name);
            t_attention_prefix_rows(name, r.route, 8, r.q, r.hw, 45, 3, idx, full, {});
            snprintf(name, sizeof(name), "pre_b %s full lengths", r.name);
            t_attention_prefix_rows(name, r.route, 8, r.q, r.hw, 45, 3, idx, full, len45);
        }
        t_copy_cols_group("CopyColsF32 group g=0", 0, 6);
        t_copy_cols_group("CopyColsF32 group g=2 (at capacity)", 2, 6);
        t_copy_cols_group("CopyColsF32 group g=1 cap 4", 1, 4);
        t_copy_cols_group("CopyColsF32 group g=3 guard", 3, 6);
        t_copy_cols_group("CopyColsF32 group g=max guard", 0xFFFFFFFFu, 6);
#if defined(PLOW_NV_HOPPER)
        t_format_group();
#endif
    }

    if (argc > 1 && !strcmp(argv[1], "--bench")) {
        bench_signal();
        bench_dense(1500, 1280, 1280);
        bench_dense(1500, 1024, 1024);
        bench_dense(1500, 4096, 1024);
        bench_dense(1500, 1024, 4096);
        bench_dense(1500, 5120, 1280);
        for (unsigned m : {104u, 390u, 1500u}) {
            bench_dense(m, 1024, 1024, 1 | 4 | 8);
            bench_dense(m, 4096, 1024, 2 | 4 | 8);
            bench_dense(m, 1024, 4096, 1 | 4 | 8);
        }
        bench_dense(104, 1024, 7680, 1 | 4 | 8);
        bench_dense(104, 1024, 1024);
        bench_conv(30, 64, 50, 480, 480);
    }
    t_conv_padding();
    printf(g_fail ? "FAIL\n" : "ALL PASS\n");
    return g_fail;
}
