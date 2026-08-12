/* Model-free full-grid screen for converting K3's dense BF16 decode weights to block FP8. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(X)                                                                                      \
    do {                                                                                           \
        const hipError_t e_ = (X);                                                                 \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "HIP FAIL %s @%d: %s\n", #X, __LINE__, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static constexpr unsigned THREADS = 512;
static constexpr size_t BF16_ARENA = 3ull << 30;
static constexpr size_t W8_ARENA = 1536ull << 20;
static constexpr size_t W4_ARENA = 1ull << 30;
static constexpr size_t TARGET_STREAM = 512ull << 20;
static constexpr unsigned MAX_REP = 1024;

struct Shape {
    const char* name;
    unsigned n;
    unsigned k;
    unsigned blocks;
    unsigned instances;
};

static const Shape SHAPES[] = {
    {"o_proj", 7168, 1536, 128, 93},       {"router", 896, 7168, 128, 92},
    {"routed_up", 896, 3584, 128, 92},     {"shared_down", 7168, 768, 128, 92},
    {"latent_down", 3584, 7168, 128, 92},  {"kda_f_b", 1536, 128, 128, 69},
    {"kda_f_a", 128, 7168, 128, 69},       {"kda_b", 12, 7168, 12, 69},
    {"mla_q_a_g", 1536, 7168, 128, 48},    {"mla_k_rope_down", 64, 7168, 64, 24},
    {"mla_q_rope", 768, 1536, 128, 24},    {"mla_q_absorb", 6144, 1536, 128, 24},
    {"mla_kv_a", 512, 7168, 128, 24},      {"output_attn_res", 4224, 7168, 128, 2},
    {"lm_head", 163840, 7168, 304, 1},     {"output_attn_proj", 7168, 4224, 128, 1},
};

static unsigned rng_state = 0x12345677u;
static float rnd() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (static_cast<int>((rng_state >> 8) & 0xffffu) - 32768) / 32768.0f;
}

static uint16_t f2bf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

static float bf2f(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static float e4m3_decode(uint8_t value) {
    const int sign = value >> 7;
    const int exponent = (value >> 3) & 15;
    const int mantissa = value & 7;
    float result;
    if (exponent == 0)
        result = (mantissa / 8.0f) * 0.015625f;
    else
        result = (1.0f + mantissa / 8.0f) * std::ldexp(1.0f, exponent - 7);
    return sign ? -result : result;
}

struct Fp8Value {
    float value;
    uint8_t bits;
};

static const std::vector<Fp8Value>& fp8_values() {
    static const std::vector<Fp8Value> values = [] {
        std::vector<Fp8Value> table;
        for (unsigned bits = 0; bits < 256; ++bits) {
            if (bits == 0x7f || bits == 0xff || bits == 0x80) continue;
            table.push_back({e4m3_decode(static_cast<uint8_t>(bits)), static_cast<uint8_t>(bits)});
        }
        std::sort(table.begin(), table.end(), [](const Fp8Value& a, const Fp8Value& b) {
            return a.value < b.value;
        });
        return table;
    }();
    return values;
}

static uint8_t e4m3_encode(float value) {
    const auto& values = fp8_values();
    const auto it = std::lower_bound(values.begin(), values.end(), value,
                                     [](const Fp8Value& a, float b) { return a.value < b; });
    if (it == values.begin()) return it->bits;
    if (it == values.end()) return values.back().bits;
    const auto prev = it - 1;
    return std::fabs(prev->value - value) <= std::fabs(it->value - value) ? prev->bits : it->bits;
}

static uint8_t fp4_encode(float value) {
    const uint8_t sign = value < 0.0f ? 8u : 0u;
    const float a = std::fabs(value);
    const uint8_t mag = static_cast<uint8_t>(
        !(a < 0.25f) + !(a < 0.75f) + !(a < 1.25f) + !(a < 1.75f) +
        !(a < 2.5f) + !(a < 3.5f) + !(a < 5.0f));
    return sign | mag;
}

static float fp4_decode(uint8_t value) {
    static constexpr float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float out = mag[value & 7u];
    return value & 8u ? -out : out;
}

static uint8_t e8m0_for_amax_host(float amax) {
    if (!(amax > 0.0f)) return 127u;
    int exponent = 0;
    std::frexp(amax / 6.0f, &exponent);
    return static_cast<uint8_t>(std::clamp(exponent + 127, 1, 254));
}

static double median_time(hipFunction_t kernel, unsigned blocks, void** args, int samples) {
    for (int i = 0; i < 2; ++i)
        CK(hipModuleLaunchKernel(kernel, blocks, 1, 1, THREADS, 1, 1, 0, nullptr, args, nullptr));
    CK(hipDeviceSynchronize());
    hipEvent_t begin, end;
    CK(hipEventCreate(&begin));
    CK(hipEventCreate(&end));
    std::vector<double> elapsed;
    elapsed.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        CK(hipEventRecord(begin));
        CK(hipModuleLaunchKernel(kernel, blocks, 1, 1, THREADS, 1, 1, 0, nullptr, args, nullptr));
        CK(hipEventRecord(end));
        CK(hipEventSynchronize(end));
        float ms = 0.0f;
        CK(hipEventElapsedTime(&ms, begin, end));
        elapsed.push_back(ms * 1000.0);
    }
    CK(hipEventDestroy(begin));
    CK(hipEventDestroy(end));
    std::sort(elapsed.begin(), elapsed.end());
    return elapsed[elapsed.size() / 2];
}

static void fill_bf16(uint16_t* device, size_t bytes, uint16_t value) {
    const size_t chunk_bytes = 64u << 20;
    std::vector<uint16_t> chunk(chunk_bytes / 2, value);
    for (size_t offset = 0; offset < bytes; offset += chunk_bytes)
        CK(hipMemcpy(reinterpret_cast<char*>(device) + offset, chunk.data(),
                     std::min(chunk_bytes, bytes - offset), hipMemcpyHostToDevice));
}

static bool run_oracle(hipFunction_t bf16, hipFunction_t w8a16, hipFunction_t w8a16_row,
                       hipFunction_t mxfp4) {
    constexpr unsigned N = 257;
    constexpr unsigned K = 7168;
    constexpr unsigned BLOCKS = 128;
    const unsigned nb = (N + 127u) / 128u;
    const unsigned kb = (K + 127u) / 128u;
    std::vector<float> original(static_cast<size_t>(N) * K);
    std::vector<uint16_t> weight_bf16(original.size());
    std::vector<uint8_t> weight_fp8(original.size());
    std::vector<uint8_t> weight_row_fp8(original.size());
    std::vector<uint8_t> weight_mxfp4(original.size() / 2);
    std::vector<uint8_t> scales_mxfp4(original.size() / 32);
    std::vector<float> scales(static_cast<size_t>(nb) * kb);
    std::vector<float> row_scales(N);
    std::vector<uint16_t> x(K);
    for (auto& value : original) value = rnd() * 0.08f;
    for (auto& value : x) value = f2bf(rnd() * 0.5f);
    for (unsigned bn = 0; bn < nb; ++bn) {
        for (unsigned bk = 0; bk < kb; ++bk) {
            float amax = 0.0f;
            for (unsigned n = bn * 128; n < std::min(N, (bn + 1) * 128); ++n)
                for (unsigned k = bk * 128; k < std::min(K, (bk + 1) * 128); ++k)
                    amax = std::max(amax, std::fabs(original[static_cast<size_t>(n) * K + k]));
            const float scale = amax == 0.0f ? 1.0f : amax / 448.0f;
            scales[static_cast<size_t>(bn) * kb + bk] = scale;
            for (unsigned n = bn * 128; n < std::min(N, (bn + 1) * 128); ++n) {
                for (unsigned k = bk * 128; k < std::min(K, (bk + 1) * 128); ++k) {
                    const size_t index = static_cast<size_t>(n) * K + k;
                    weight_bf16[index] = f2bf(original[index]);
                    weight_fp8[index] = e4m3_encode(original[index] / scale);
                }
            }
        }
    }
    for (unsigned n = 0; n < N; ++n) {
        float amax = 0.0f;
        for (unsigned k = 0; k < K; ++k)
            amax = std::max(amax, std::fabs(original[static_cast<size_t>(n) * K + k]));
        row_scales[n] = amax == 0.0f ? 1.0f : amax / 448.0f;
        for (unsigned k = 0; k < K; ++k) {
            const size_t index = static_cast<size_t>(n) * K + k;
            weight_row_fp8[index] = e4m3_encode(original[index] / row_scales[n]);
        }
        for (unsigned bk = 0; bk < K / 32; ++bk) {
            float amax = 0.0f;
            for (unsigned k = bk * 32; k < (bk + 1) * 32; ++k)
                amax = std::max(amax, std::fabs(original[static_cast<size_t>(n) * K + k]));
            const uint8_t sb = e8m0_for_amax_host(amax);
            const float scale = std::ldexp(1.0f, static_cast<int>(sb) - 127);
            scales_mxfp4[static_cast<size_t>(n) * (K / 32) + bk] = sb;
            for (unsigned p = 0; p < 16; ++p) {
                const unsigned k = bk * 32 + p * 2;
                const uint8_t lo = fp4_encode(original[static_cast<size_t>(n) * K + k] / scale);
                const uint8_t hi = fp4_encode(original[static_cast<size_t>(n) * K + k + 1] / scale);
                weight_mxfp4[(static_cast<size_t>(n) * K + k) / 2] = lo | (hi << 4);
            }
        }
    }

    uint16_t *d_bf16, *d_x, *d_out_bf16, *d_out_fp8, *d_out_row_fp8, *d_out_mxfp4;
    uint8_t *d_fp8, *d_row_fp8, *d_mxfp4, *d_mxscale;
    float *d_scale, *d_row_scale;
    CK(hipMalloc(&d_bf16, weight_bf16.size() * 2));
    CK(hipMalloc(&d_fp8, weight_fp8.size()));
    CK(hipMalloc(&d_row_fp8, weight_row_fp8.size()));
    CK(hipMalloc(&d_mxfp4, weight_mxfp4.size()));
    CK(hipMalloc(&d_mxscale, scales_mxfp4.size()));
    CK(hipMalloc(&d_scale, scales.size() * 4));
    CK(hipMalloc(&d_row_scale, row_scales.size() * 4));
    CK(hipMalloc(&d_x, x.size() * 2));
    CK(hipMalloc(&d_out_bf16, N * 2));
    CK(hipMalloc(&d_out_fp8, N * 2));
    CK(hipMalloc(&d_out_row_fp8, N * 2));
    CK(hipMalloc(&d_out_mxfp4, N * 2));
    CK(hipMemcpy(d_bf16, weight_bf16.data(), weight_bf16.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_fp8, weight_fp8.data(), weight_fp8.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(d_row_fp8, weight_row_fp8.data(), weight_row_fp8.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(d_mxfp4, weight_mxfp4.data(), weight_mxfp4.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(d_mxscale, scales_mxfp4.data(), scales_mxfp4.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(d_scale, scales.data(), scales.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_row_scale, row_scales.data(), row_scales.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    unsigned one = 1;
    unsigned n = N, k = K;
    void* bf_args[] = {&d_out_bf16, &d_x, &d_bf16, &one, &n, &k};
    void* fp_args[] = {&d_out_fp8, &d_x, &d_fp8, &d_scale, &one, &n, &k};
    void* row_fp_args[] = {&d_out_row_fp8, &d_x, &d_row_fp8, &d_row_scale, &one, &n, &k};
    void* mx_args[] = {&d_out_mxfp4, &d_x, &d_mxfp4, &d_mxscale, &one, &n, &k};
    CK(hipMemset(d_out_bf16, 0xab, N * 2));
    CK(hipMemset(d_out_fp8, 0xcd, N * 2));
    CK(hipMemset(d_out_row_fp8, 0xef, N * 2));
    CK(hipMemset(d_out_mxfp4, 0x91, N * 2));
    CK(hipModuleLaunchKernel(bf16, BLOCKS, 1, 1, THREADS, 1, 1, 0, nullptr, bf_args, nullptr));
    CK(hipModuleLaunchKernel(w8a16, BLOCKS, 1, 1, THREADS, 1, 1, 0, nullptr, fp_args, nullptr));
    CK(hipModuleLaunchKernel(w8a16_row, BLOCKS, 1, 1, THREADS, 1, 1, 0, nullptr, row_fp_args,
                             nullptr));
    CK(hipModuleLaunchKernel(mxfp4, BLOCKS, 1, 1, THREADS, 1, 1, 0, nullptr, mx_args, nullptr));
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> out_bf16(N), out_fp8(N), out_row_fp8(N), out_mxfp4(N);
    CK(hipMemcpy(out_bf16.data(), d_out_bf16, N * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(out_fp8.data(), d_out_fp8, N * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(out_row_fp8.data(), d_out_row_fp8, N * 2, hipMemcpyDeviceToHost));
    CK(hipMemcpy(out_mxfp4.data(), d_out_mxfp4, N * 2, hipMemcpyDeviceToHost));

    double ref2 = 0.0, bf_err2 = 0.0, fp_err2 = 0.0, row_err2 = 0.0, mx_err2 = 0.0,
           mx_quant_ref2 = 0.0, mx_quant_err2 = 0.0, cross_err2 = 0.0;
    double dot = 0.0, row_dot = 0.0, mx_dot = 0.0, bf2 = 0.0, fp2 = 0.0, row2 = 0.0,
           mx2 = 0.0;
    unsigned nonfinite = 0;
    for (unsigned row = 0; row < N; ++row) {
        double ref = 0.0;
        double mx_ref = 0.0;
        for (unsigned col = 0; col < K; ++col)
            ref += static_cast<double>(bf2f(x[col])) *
                   bf2f(weight_bf16[static_cast<size_t>(row) * K + col]);
        for (unsigned col = 0; col < K; ++col) {
            const uint8_t packed = weight_mxfp4[(static_cast<size_t>(row) * K + col) / 2];
            const uint8_t code = col & 1u ? packed >> 4 : packed & 15u;
            const uint8_t sb = scales_mxfp4[static_cast<size_t>(row) * (K / 32) + col / 32];
            const float scale = std::ldexp(1.0f, static_cast<int>(sb) - 127);
            mx_ref += static_cast<double>(bf2f(x[col])) * fp4_decode(code) * scale;
        }
        const double b = bf2f(out_bf16[row]);
        const double f = bf2f(out_fp8[row]);
        const double r = bf2f(out_row_fp8[row]);
        const double m = bf2f(out_mxfp4[row]);
        nonfinite += !std::isfinite(b) || !std::isfinite(f) || !std::isfinite(r) ||
                     !std::isfinite(m);
        ref2 += ref * ref;
        bf_err2 += (b - ref) * (b - ref);
        fp_err2 += (f - ref) * (f - ref);
        row_err2 += (r - ref) * (r - ref);
        mx_err2 += (m - ref) * (m - ref);
        mx_quant_ref2 += mx_ref * mx_ref;
        mx_quant_err2 += (m - mx_ref) * (m - mx_ref);
        cross_err2 += (f - b) * (f - b);
        dot += b * f;
        row_dot += b * r;
        mx_dot += b * m;
        bf2 += b * b;
        fp2 += f * f;
        row2 += r * r;
        mx2 += m * m;
    }
    const double bf_rel = std::sqrt(bf_err2 / ref2);
    const double fp_rel = std::sqrt(fp_err2 / ref2);
    const double row_rel = std::sqrt(row_err2 / ref2);
    const double mx_rel = std::sqrt(mx_err2 / ref2);
    const double mx_quant_rel = std::sqrt(mx_quant_err2 / mx_quant_ref2);
    const double cross_rel = std::sqrt(cross_err2 / bf2);
    const double cosine = dot / std::sqrt(bf2 * fp2);
    const double row_cosine = row_dot / std::sqrt(bf2 * row2);
    const double mx_cosine = mx_dot / std::sqrt(bf2 * mx2);
    const bool ok = nonfinite == 0 && bf_rel < 0.01 && fp_rel < 0.08 && row_rel < 0.08 &&
                    mx_rel < 0.20 && mx_quant_rel < 0.01 && cosine > 0.995 &&
                    row_cosine > 0.995 && mx_cosine > 0.98;
    std::printf("oracle N=%u K=%u: bf16_rel=%.6g block_rel=%.6g block_drift=%.6g "
                "block_cos=%.8f row_rel=%.6g row_cos=%.8f mx_rel=%.6g mx_qrel=%.6g "
                "mx_cos=%.8f %s\n",
                N, K, bf_rel, fp_rel, cross_rel, cosine, row_rel, row_cosine, mx_rel,
                mx_quant_rel, mx_cosine, ok ? "PASS" : "FAIL");
    for (void* pointer : {static_cast<void*>(d_bf16), static_cast<void*>(d_fp8),
                          static_cast<void*>(d_row_fp8), static_cast<void*>(d_mxfp4),
                          static_cast<void*>(d_mxscale), static_cast<void*>(d_scale),
                          static_cast<void*>(d_row_scale), static_cast<void*>(d_x),
                          static_cast<void*>(d_out_bf16), static_cast<void*>(d_out_fp8),
                          static_cast<void*>(d_out_row_fp8), static_cast<void*>(d_out_mxfp4)})
        CK(hipFree(pointer));
    return ok;
}

int main(int argc, char** argv) {
    const char* object = argc > 1 ? argv[1] : "/tmp/k3_gemv_w8a16.co";
    const int samples = argc > 2 ? std::atoi(argv[2]) : 9;
    if (samples < 3 || (samples & 1) == 0) {
        std::fprintf(stderr, "samples must be an odd integer >= 3\n");
        return 2;
    }
    CK(hipInit(0));
    hipModule_t module;
    hipFunction_t bf16, bf16_aa, w8a16, w8a16_row, mxfp4;
    CK(hipModuleLoad(&module, object));
    CK(hipModuleGetFunction(&bf16, module, "k_bf16"));
    CK(hipModuleGetFunction(&bf16_aa, module, "k_bf16_aa"));
    CK(hipModuleGetFunction(&w8a16, module, "k_w8a16"));
    CK(hipModuleGetFunction(&w8a16_row, module, "k_w8a16_row"));
    CK(hipModuleGetFunction(&mxfp4, module, "k_mxfp4"));
    if (!run_oracle(bf16, w8a16, w8a16_row, mxfp4)) return 3;

    uint16_t *weights_bf16, *x, *out;
    uint8_t* weights_fp8;
    uint8_t *weights_mxfp4, *scales_mxfp4;
    float* scales;
    CK(hipMalloc(&weights_bf16, BF16_ARENA));
    CK(hipMalloc(&weights_fp8, W8_ARENA));
    CK(hipMalloc(&weights_mxfp4, W4_ARENA));
    CK(hipMalloc(&scales_mxfp4, W4_ARENA / 16));
    CK(hipMalloc(&x, 1u << 22));
    CK(hipMalloc(&out, 1u << 22));
    CK(hipMalloc(&scales, 1u << 20));
    fill_bf16(weights_bf16, BF16_ARENA, f2bf(0.015625f));
    CK(hipMemset(weights_fp8, 0x08, W8_ARENA));
    CK(hipMemset(weights_mxfp4, 0x11, W4_ARENA));
    CK(hipMemset(scales_mxfp4, 127, W4_ARENA / 16));
    std::vector<uint16_t> host_x((1u << 22) / 2);
    for (auto& value : host_x) value = f2bf(rnd() * 0.5f);
    CK(hipMemcpy(x, host_x.data(), 1u << 22, hipMemcpyHostToDevice));
    std::vector<float> host_scales((1u << 20) / 4, 1.0f);
    CK(hipMemcpy(scales, host_scales.data(), 1u << 20, hipMemcpyHostToDevice));

    FILE* json = nullptr;
    if (const char* path = std::getenv("PLOW_K3_W8A16_JSONL")) {
        json = std::fopen(path, "w");
        if (!json) {
            std::perror(path);
            return 2;
        }
    }
    double bf_total = 0.0, bf_aa_total = 0.0, block_total = 0.0, row_total = 0.0,
           mx_total = 0.0;
    double aa_low = 10.0, aa_high = 0.0;
    std::printf("%-18s %7s %6s %5s | %9s %6s | %9s %7s | %9s %7s | %9s %7s\n",
                "shape", "N", "K", "grid", "bf16_us", "A/A", "block_us", "speedup",
                "row_us", "speedup", "mxfp4_us", "speedup");
    for (const Shape& shape : SHAPES) {
        const size_t bf_slab = static_cast<size_t>(shape.n) * shape.k * 2;
        const size_t fp_slab = static_cast<size_t>(shape.n) * shape.k;
        const size_t mx_slab = static_cast<size_t>(shape.n) * shape.k * 17 / 32;
        unsigned rb = static_cast<unsigned>(std::max<size_t>(1, TARGET_STREAM / bf_slab));
        unsigned r8 = static_cast<unsigned>(std::max<size_t>(1, TARGET_STREAM / fp_slab));
        unsigned r4 = static_cast<unsigned>(std::max<size_t>(1, TARGET_STREAM / mx_slab));
        rb = std::min<unsigned>(rb, std::min<unsigned>(MAX_REP, BF16_ARENA / bf_slab));
        r8 = std::min<unsigned>(r8, std::min<unsigned>(MAX_REP, W8_ARENA / fp_slab));
        r4 = std::min<unsigned>(r4, std::min<unsigned>(MAX_REP, W4_ARENA / mx_slab));
        unsigned n = shape.n, k = shape.k;
        void* bf_args[] = {&out, &x, &weights_bf16, &rb, &n, &k};
        void* fp_args[] = {&out, &x, &weights_fp8, &scales, &r8, &n, &k};
        void* row_args[] = {&out, &x, &weights_fp8, &scales, &r8, &n, &k};
        void* mx_args[] = {&out, &x, &weights_mxfp4, &scales_mxfp4, &r4, &n, &k};
        double b1 = median_time(bf16, shape.blocks, bf_args, samples);
        double a1 = median_time(bf16_aa, shape.blocks, bf_args, samples);
        double m1 = median_time(mxfp4, shape.blocks, mx_args, samples);
        double f1 = median_time(w8a16, shape.blocks, fp_args, samples);
        double r1 = median_time(w8a16_row, shape.blocks, row_args, samples);
        double r2 = median_time(w8a16_row, shape.blocks, row_args, samples);
        double f2 = median_time(w8a16, shape.blocks, fp_args, samples);
        double m2 = median_time(mxfp4, shape.blocks, mx_args, samples);
        double a2 = median_time(bf16_aa, shape.blocks, bf_args, samples);
        double b2 = median_time(bf16, shape.blocks, bf_args, samples);
        const double b = std::min(b1, b2) / rb;
        const double f = std::min(f1, f2) / r8;
        const double r = std::min(r1, r2) / r8;
        const double m = std::min(m1, m2) / r4;
        const double aa = (std::min(a1, a2) / rb) / b;
        aa_low = std::min(aa_low, aa);
        aa_high = std::max(aa_high, aa);
        bf_total += b * shape.instances / 1000.0;
        bf_aa_total += b * aa * shape.instances / 1000.0;
        block_total += f * shape.instances / 1000.0;
        row_total += r * shape.instances / 1000.0;
        mx_total += m * shape.instances / 1000.0;
        std::printf("%-18s %7u %6u %5u | %9.3f %6.3f | %9.3f %7.3f | %9.3f %7.3f | "
                    "%9.3f %7.3f\n",
                    shape.name, shape.n, shape.k, shape.blocks, b, aa, f, b / f, r, b / r, m,
                    b / m);
        if (json)
            std::fprintf(json,
                         "{\"schema\":\"plow.k3-gemv-w8a16.v1\",\"shape\":\"%s\","
                         "\"n\":%u,\"k\":%u,\"grid\":%u,\"instances\":%u,\"bf16_us\":%.6f,"
                         "\"block_us\":%.6f,\"row_us\":%.6f,\"block_speedup\":%.6f,"
                         "\"row_speedup\":%.6f,\"mxfp4_us\":%.6f,"
                         "\"mxfp4_speedup\":%.6f,\"correct\":true}\n",
                         shape.name, shape.n, shape.k, shape.blocks, shape.instances, b, f, r,
                         b / f, b / r, m, b / m);
    }
    if (json) std::fclose(json);
    const double block_saving = bf_total - block_total;
    const double row_saving = bf_total - row_total;
    const double mx_saving = bf_total - mx_total;
    const double aa_weighted = bf_aa_total / bf_total;
    const bool stable = aa_weighted >= 0.98 && aa_weighted <= 1.02;
    std::printf("A/A bf16 range %.4f..%.4f, weighted %.4f %s\n", aa_low, aa_high, aa_weighted,
                stable ? "PASS" : "NOISY");
    const bool promote = mx_saving >= 5.0 && stable;
    std::printf("weighted body projection: bf16 %.3f ms, block %.3f (save %.3f), "
                "row %.3f (save %.3f), mxfp4 %.3f (save %.3f), %s\n",
                bf_total, block_total, block_saving, row_total, row_saving, mx_total, mx_saving,
                promote ? "PROMOTE MXFP4" : "STOP");
    return stable ? 0 : 4;
}
