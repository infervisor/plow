// Shared helpers for the exact-object attention ladder replays (gfx942).
#pragma once
#include <hip/hip_runtime.h>
#include "dev_isa.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(expr) do { auto e_ = (expr); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s: %s at %s:%d\n", #expr, hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr unsigned GRID = 304, GUARD = 128;

template<class T> T* dalloc(size_t n) { T* p; CK(hipMalloc(&p, n * sizeof(T))); return p; }
template<class T> T* upload(const std::vector<T>& x) {
    T* p = dalloc<T>(x.size()); CK(hipMemcpy(p, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice)); return p;
}
template<class T> void download(std::vector<T>& x, const T* p) {
    CK(hipMemcpy(x.data(), p, x.size() * sizeof(T), hipMemcpyDeviceToHost));
}

static uint32_t g_rng = 73;
static inline uint32_t rnext() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static inline uint16_t bf16_of(float f) { uint32_t b; std::memcpy(&b, &f, 4); b += 0x7fff + ((b >> 16) & 1); return b >> 16; }
static inline float from_bf16(uint16_t b) { uint32_t bits = uint32_t(b) << 16; float f; std::memcpy(&f, &bits, 4); return f; }
// OCP E4M3FN (the runtime's KV encoding). 0x7f/0xff = NaN.
static inline float from_fp8(uint8_t b) {
    if ((b & 0x7f) == 0x7f) return NAN;
    const int exp = (b >> 3) & 15, mantissa = b & 7;
    const float value = exp ? std::ldexp(float(8 + mantissa), exp - 10) : std::ldexp(float(mantissa), -9);
    return b & 128 ? -value : value;
}
static inline std::vector<uint16_t> random_bf(size_t n) {
    std::vector<uint16_t> v(n); for (auto& x : v) x = bf16_of((int(rnext() % 2001) - 1000) / 1000.f); return v;
}
static inline std::vector<uint8_t> random_fp8(size_t n) {
    // exponents 4..8, mantissa any, both signs: magnitudes 2^-3 .. ~2 (finite, no NaN byte)
    std::vector<uint8_t> v(n); for (auto& x : v) x = 0x20 + (rnext() % 33) + ((rnext() & 1) << 7); return v;
}
static inline std::vector<float> random_scales(size_t n) {
    std::vector<float> v(n); for (auto& x : v) x = 0.5f + (rnext() % 101) / 100.f; return v;
}

struct Kernel { std::string label; hipModule_t mod{}; hipFunction_t fn{}; unsigned threads{}; };
static inline Kernel load_kernel(const char* label, const std::string& path, const char* symbol, unsigned threads) {
    Kernel k; k.label = label; k.threads = threads;
    CK(hipModuleLoad(&k.mod, path.c_str()));
    if (hipModuleGetFunction(&k.fn, k.mod, symbol) != hipSuccess) {
        std::fprintf(stderr, "symbol %s missing in %s\n", symbol, path.c_str()); std::exit(2);
    }
    return k;
}

// One-instruction interpreter program. `blocks` stream entries (slice i -> entry i), the
// persistent grid of GRID workgroups drains them through the global-queue cursor.
struct Replay {
    PlowProgram p{};
    PlowDevInst* ins{};
    std::vector<PlowStreamEnt> entries;
    PlowStreamEnt* d_entries{};
    uint32_t* d_seg_ofs{};
    hipEvent_t start{}, stop{};
    Replay() {
        entries.resize(GRID);
        for (unsigned i = 0; i < GRID; ++i) entries[i].slice = i;
        d_entries = upload(entries);
        d_seg_ofs = dalloc<uint32_t>(2);
        p.gq_stream = d_entries;
        p.gq_seg_ofs = d_seg_ofs;
        p.gq_cursor = dalloc<uint32_t>(32);
        p.n_seg = p.n_gpu = 1;
        ins = dalloc<PlowDevInst>(1);
        p.insts = ins;
        CK(hipEventCreate(&start)); CK(hipEventCreate(&stop));
    }
    void set_tensors(void** table) { p.tensors = table; }
    void set_inst(const PlowDevInst& d) {
        const uint32_t ofs[2] = {0u, d.blocks};
        CK(hipMemcpy(d_seg_ofs, ofs, 8, hipMemcpyHostToDevice));
        CK(hipMemcpy(ins, &d, sizeof(d), hipMemcpyHostToDevice));
    }
    void run(Kernel& k) {
        CK(hipMemsetAsync(p.gq_cursor, 0, 128));
        void* args[] = {&p};
        CK(hipModuleLaunchKernel(k.fn, GRID, 1, 1, k.threads, 1, 1, 0, nullptr, args, nullptr));
    }
    // median of `reps` timed launches (cursor reset outside the event)
    double time_us(Kernel& k, unsigned reps = 9) {
        std::vector<double> t;
        for (unsigned r = 0; r < reps + 1; ++r) {
            CK(hipMemsetAsync(p.gq_cursor, 0, 128));
            CK(hipEventRecord(start));
            void* args[] = {&p};
            CK(hipModuleLaunchKernel(k.fn, GRID, 1, 1, k.threads, 1, 1, 0, nullptr, args, nullptr));
            CK(hipEventRecord(stop)); CK(hipEventSynchronize(stop));
            float ms; CK(hipEventElapsedTime(&ms, start, stop));
            if (r) t.push_back(ms * 1000.);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    }
};

// NaN-poison a device float buffer with guard bands and check the guards after a run.
static inline void poison(float* base, size_t n_total) {
    std::vector<float> v(n_total, NAN);
    CK(hipMemcpy(base, v.data(), n_total * 4, hipMemcpyHostToDevice));
}
static inline bool guards_intact(const std::vector<float>& host, size_t payload) {
    for (unsigned i = 0; i < GUARD; ++i)
        if (!std::isnan(host[i]) || !std::isnan(host[GUARD + payload + i])) return false;
    return true;
}

// CPU merge of nsplit (m,l,O) partials in the interpreter's exp2 frame: O = sum w_s O_s / sum w_s l_s.
static inline void merge_partials(const float* opart, const float* mlpart, unsigned nsplit, unsigned dk,
                                  std::vector<double>& out) {
    out.assign(dk, 0.0);
    double maxm = -INFINITY;
    for (unsigned s = 0; s < nsplit; ++s) maxm = std::max(maxm, double(mlpart[s * 2]));
    double den = 0;
    std::vector<double> w(nsplit);
    for (unsigned s = 0; s < nsplit; ++s) {
        const double m = mlpart[s * 2];
        w[s] = (m == -INFINITY || !(m > -3e38)) ? 0.0 : std::exp2(m - maxm);
        den += w[s] * mlpart[s * 2 + 1];
    }
    for (unsigned c = 0; c < dk; ++c) {
        double v = 0;
        for (unsigned s = 0; s < nsplit; ++s) v += w[s] * opart[size_t(s) * dk + c];
        out[c] = (den > 0) ? v / den : NAN;
    }
}
