#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include "gdn_prefill_sm120.h"

static_assert(std::is_trivial<GdnPrefillArgs>::value && std::is_standard_layout<GdnPrefillArgs>::value);

#define CUDA_CHECK(call) do { \
    const cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { CUDA_CHECK(cudaMalloc(&p, std::max(n, size_t(1)) * sizeof(T))); }
    explicit Device(const std::vector<T>& x) : Device(x.size()) {
        if (!x.empty()) CUDA_CHECK(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device() { CUDA_CHECK(cudaFree(p)); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void read(std::vector<T>& x) const {
        if (!x.empty()) CUDA_CHECK(cudaMemcpy(x.data(), p, x.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

static float from_bf16(uint16_t x) {
    const uint32_t bits = uint32_t(x) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t to_bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

struct Error {
    double max_abs = 0, max_rel = 0, diff2 = 0, ref2 = 0;
    bool finite = true;
    void add(double actual, double expected) {
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            finite = false;
            return;
        }
        const double diff = std::abs(actual - expected);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / std::max(std::abs(expected), 1e-12));
        diff2 += diff * diff;
        ref2 += expected * expected;
    }
    double rel_l2() const { return finite ? std::sqrt(diff2 / std::max(ref2, 1e-30)) : INFINITY; }
    void print(const char* label) const {
        std::printf("  %s: max_abs=%.9g max_rel=%.9g rel-L2=%.9g finite=%d\n",
                    label, max_abs, max_rel, rel_l2(), int(finite));
    }
};

static void normalized(std::vector<uint16_t>& x, std::mt19937& rng) {
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    for (size_t base = 0; base < x.size(); base += 128) {
        float values[128];
        double norm2 = 0;
        for (float& v : values) { v = random(rng); norm2 += double(v) * v; }
        for (int d = 0; d < 128; ++d)
            x[base + d] = to_bf16(float(values[d] / std::sqrt(norm2)));
    }
}

static bool run_case(const char* name, const std::vector<int>& lengths, bool timed = false) {
    std::vector<int64_t> offsets(1, 0);
    for (int n : lengths) offsets.push_back(offsets.back() + n);
    const int tokens = int(offsets.back()), nseq = int(lengths.size());
    const size_t nout = size_t(tokens) * 48 * 128;
    const size_t nstate = size_t(nseq) * 48 * 128 * 128;
    const float scale = 1.f / std::sqrt(128.f);
    std::printf("%s: nseq=%d tokens=%d\n", name, nseq, tokens);
    std::fflush(stdout);

    std::mt19937 rng(1234 + tokens + nseq);
    std::uniform_real_distribution<float> random(-1.f, 1.f), unit(0.001f, 0.999f);
    std::vector<uint16_t> q(size_t(tokens) * 16 * 128), k(q.size()), v(nout), output(nout);
    std::vector<float> alpha(size_t(tokens) * 48), beta(alpha.size()), initial(nstate), final(nstate);
    normalized(q, rng);
    normalized(k, rng);
    for (auto& x : v) x = to_bf16(random(rng));
    for (auto& x : alpha) x = 0.5f + 0.5f * unit(rng);
    for (auto& x : beta) x = unit(rng);
    for (auto& x : initial) x = 0.1f * random(rng);

    Device<uint16_t> dq(q), dk(k), dv(v), dout(nout);
    Device<float> da(alpha), db(beta), di(initial), ds(nstate);
    Device<int64_t> dc(offsets);
    CUDA_CHECK(cudaMemset(dout.p, 0xff, std::max(nout, size_t(1)) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(ds.p, 0xff, nstate * sizeof(float)));
    GdnPrefillArgs args{dq.p, dk.p, dv.p, dout.p, da.p, db.p, ds.p, di.p, dc.p, tokens, nseq, scale};
    const dim3 grid(8, 48, nseq);
    gdn_prefill_sm120<<<grid, 128>>>(args);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dout.read(output);
    ds.read(final);

    Error rounded, unrounded, state_error;
    bool empty_exact = true;
    // FP64 recurrence on the actual BF16 inputs, with FP32 q*scale as in decode.
    for (int seq = 0; seq < nseq; ++seq) {
        for (int head = 0; head < 48; ++head) {
            const size_t state0 = (size_t(seq) * 48 + head) * 128 * 128;
            std::vector<double> state(initial.begin() + state0, initial.begin() + state0 + 128 * 128);
            for (int64_t t = offsets[seq]; t < offsets[seq + 1]; ++t) {
                double qt[128], kt[128];
                const size_t qk0 = (size_t(t) * 16 + head / 3) * 128;
                for (int d = 0; d < 128; ++d) {
                    qt[d] = from_bf16(q[qk0 + d]) * scale;
                    kt[d] = from_bf16(k[qk0 + d]);
                }
                const double decay = alpha[size_t(t) * 48 + head];
                const double b = beta[size_t(t) * 48 + head];
                const size_t out0 = (size_t(t) * 48 + head) * 128;
                for (int row = 0; row < 128; ++row) {
                    double* s = state.data() + row * 128;
                    double proj = 0;
                    for (int d = 0; d < 128; ++d) {
                        s[d] *= decay;
                        proj += s[d] * kt[d];
                    }
                    const double delta = (from_bf16(v[out0 + row]) - proj) * b;
                    double result = 0;
                    for (int d = 0; d < 128; ++d) {
                        s[d] += delta * kt[d];
                        result += s[d] * qt[d];
                    }
                    const double actual = from_bf16(output[out0 + row]);
                    rounded.add(actual, from_bf16(to_bf16(float(result))));
                    unrounded.add(actual, result);
                }
            }
            for (int i = 0; i < 128 * 128; ++i)
                state_error.add(final[state0 + i], state[i]);
            if (lengths[seq] == 0)
                empty_exact &= std::memcmp(final.data() + state0, initial.data() + state0,
                                           128 * 128 * sizeof(float)) == 0;
        }
    }
    rounded.print("o vs BF16-rounded reference");
    unrounded.print("o vs unrounded reference");
    state_error.print("final state");
    di.read(final);
    const bool initial_unchanged = std::memcmp(initial.data(), final.data(), nstate * sizeof(float)) == 0;
    const bool pass = rounded.rel_l2() < 2e-3 && state_error.rel_l2() < 1e-4 &&
                      empty_exact && initial_unchanged;
    std::printf("  %s (empty copy exact=%d, initial unchanged=%d)\n",
                pass ? "PASS" : "FAIL", int(empty_exact), int(initial_unchanged));

    if (timed && pass) {
        cudaEvent_t begin, end;
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
        gdn_prefill_sm120<<<grid, 128>>>(args);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(begin));
        constexpr int repeats = 10;
        for (int i = 0; i < repeats; ++i) gdn_prefill_sm120<<<grid, 128>>>(args);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        std::printf("  kernel time: %.6f ms (mean of %d launches)\n", ms / repeats, repeats);
        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(end));
    }
    return pass;
}

int main() {
    bool pass = true;
    pass &= run_case("empty", {0, 0});
    pass &= run_case("length 1", {1});
    pass &= run_case("length 63", {63});
    pass &= run_case("length 64", {64});
    pass &= run_case("length 65", {65});
    pass &= run_case("ragged", {0, 1, 63, 0, 64, 65, 1000, 0});
    pass &= run_case("1 x 4096", {4096}, true);
    pass &= run_case("8 x 512", std::vector<int>(8, 512), true);
    std::vector<int> limits(64, 0);
    for (int i = 1; i < 64; i += 2) limits[i] = 256;
    pass &= run_case("64 sequences / 8192 tokens", limits);
    return pass ? 0 : 1;
}
