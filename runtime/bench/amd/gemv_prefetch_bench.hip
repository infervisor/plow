/* L8 microbench: claim-ahead weight prefetch for the decode bf16 GEMV (`gemv_prefetch_slice`,
 * op_gemm.h, `PLOW_GEMV_PREFETCH`).
 *
 * ONE GPU. A persistent 256-workgroup kernel replays the interpreter's per-packet loop over a
 * chain of `Gemv` packets (K3 decode shapes at TP8: N x K = 7168 x 1536 (o_proj / expert up,
 * 22 MB), 1536 x 7168 (a single projection, 22 MB), 1536 x 128 (f_b, 0.4 MB)), each packet with
 * its own cold weight matrix and a simulated gate that opens `period` us after the previous one:
 *
 *     [prefetch this slice's rows]  ->  poll the gate (a global atomic load per poll, as
 *     `ctr_poll` does, so the poll's vmcnt(0) covers the prefetch exactly as in the
 *     interpreter)  ->  t_ready  ->  vmcnt(0)  ->  body (`d_gemv`, the served arm)  ->  t_end
 *
 * Reported per shape and period: mean/median per-workgroup body (t_end - t_ready), the packet
 * span (max t_end - min t_ready) and the chain makespan, prefetch off vs on. `period = 0` is the
 * back-to-back chain (no gate slack: the cost side), `period >= body` the case where the gate
 * wait hides the whole prefetch. Every output word of the prefetch-on run is compared against
 * the prefetch-off run (loads only, so equality is expected — the check is the harness's).
 *
 * Build (gfx950, the served decode recipe's defines):
 *   hipcc --offload-arch=gfx950 -O3 -std=c++17 -DPLOW_BUCKET_DECODE=1 -DPLOW_K3=1 \
 *     -DPLOW_GEMV_MM=1 -DGV_UNROLL=14 -DPLOW_GEMV_PREFETCH=1 -I runtime/amd -I runtime/common \
 *     runtime/bench/amd/gemv_prefetch_bench.hip -o /tmp/gemv_prefetch_bench
 * Run:  GPU_LEASE_TIMEOUT=7200 perf-data/tools/gpulease -n 1 l8 /tmp/gemv_prefetch_bench
 *   [packets=24] [reps=5]
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(__HIP_DEVICE_COMPILE__)
#include "op_gemm.h"
#define BODY(...) __VA_ARGS__
#else
typedef unsigned short bf16;
#define PLOW_THREADS 512
#define BODY(...) {}
#endif

#define CK(call)                                                                       \
    do {                                                                               \
        const hipError_t e_ = (call);                                                  \
        if (e_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__,                \
                         hipGetErrorString(e_));                                       \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

struct Packet {
    const bf16* W;
    bf16* C;
    unsigned N, K;
};

/* rec[(p * gridDim + wg) * 2 + {0,1}] = t_ready, t_end (s_memrealtime, 100 MHz). */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) __attribute__((amdgpu_waves_per_eu(2, 2)))
void k_chain(const Packet* pk, unsigned n_pk, const bf16* x, unsigned* gate,
             unsigned long long t0, unsigned long long period, unsigned prefetch,
             unsigned long long* rec)
BODY({
    __shared__ __align__(16) bf16 lds[GM_LDS_HALVES];
    __shared__ unsigned open;
    const unsigned slice = blockIdx.x, nblk = gridDim.x;
    for (unsigned p = 0; p < n_pk; ++p) {
        const Packet q = pk[p];
        __syncthreads();
#if PLOW_GEMV_PREFETCH
        if (prefetch) gemv_prefetch_slice(q.W, q.N, q.K, slice, nblk, lds);
#endif
        /* Gate: opens at t0 + p*period. One thread polls (as the interpreter's designated
         * pollers do), with a global atomic load per iteration so the vmcnt(0) the poll needs
         * sits where the interpreter's sits. */
        if (threadIdx.x == 0) {
            for (;;) {
                const unsigned g = __hip_atomic_load(gate, __ATOMIC_RELAXED,
                                                     __HIP_MEMORY_SCOPE_AGENT);
                const unsigned long long now = __builtin_amdgcn_s_memrealtime();
                if (now >= t0 + p * period || g == 0xffffffffu) break;
                __builtin_amdgcn_s_sleep(8);
            }
            open = 1;
        }
        __syncthreads();
        const unsigned long long t_ready = __builtin_amdgcn_s_memrealtime();
#if PLOW_GEMV_PREFETCH
        if (prefetch) cp_async_wait();
#endif
        d_gemv(q.C, x, q.W, nullptr, nullptr, 1u, q.N, q.K, 0, 0.0f, slice, nblk, lds);
        __syncthreads();
        if (threadIdx.x == 0) {
            rec[((size_t)p * nblk + slice) * 2] = t_ready;
            rec[((size_t)p * nblk + slice) * 2 + 1] = __builtin_amdgcn_s_memrealtime();
        }
    }
    (void)open;
})

extern "C" __global__ void k_probe(unsigned long long* p)
BODY({ *p = __builtin_amdgcn_s_memrealtime(); })

static uint16_t f2bf_h(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

template <class T>
static T* dmalloc(size_t n) {
    T* p;
    CK(hipMalloc(&p, n * sizeof(T)));
    return p;
}

int main(int argc, char** argv) {
    const unsigned n_pk = argc > 1 ? (unsigned)std::atoi(argv[1]) : 24u;
    const unsigned reps = argc > 2 ? (unsigned)std::atoi(argv[2]) : 5u;
    const unsigned nblk = 256u;
    struct Shape {
        const char* name;
        unsigned N, K;
    };
    /* The K3 TP8 decode `Gemv` census (flag-on emit, per token): 93 x (7168,1536) o_proj,
     * 92 x (7168,768), 92 x (3584,7168) latent up, 48 x (1536,7168), 24 x (6144,1536), …;
     * 69 x (1536,128) f_b when L3 is off. */
    const Shape shapes[] = {{"N=7168 K=1536 (o_proj, 22 MB)", 7168u, 1536u},
                            {"N=7168 K=768 (11 MB)", 7168u, 768u},
                            {"N=3584 K=7168 (latent up, 51 MB)", 3584u, 7168u},
                            {"N=1536 K=7168 (single proj, 22 MB)", 1536u, 7168u},
                            {"N=1536 K=128 (f_b, 0.4 MB)", 1536u, 128u}};
    const unsigned long long periods_us[] = {0ull, 6ull, 12ull, 20ull};

    std::mt19937_64 rng(0x5eedu);
    std::uniform_real_distribution<float> wt(-0.05f, 0.05f), act(-1.0f, 1.0f);
    unsigned* d_gate = dmalloc<unsigned>(1);
    CK(hipMemset(d_gate, 0, 4));
    unsigned long long* d_probe = dmalloc<unsigned long long>(1);
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("%s: packets=%u reps=%u workgroups=%u\n", prop.gcnArchName, n_pk, reps, nblk);

    for (const Shape& s : shapes) {
        const size_t w_elems = (size_t)s.N * s.K;
        std::vector<uint16_t> h_w(w_elems), h_x(s.K);
        std::vector<bf16*> d_w(n_pk), d_cA(n_pk), d_cB(n_pk);
        for (unsigned p = 0; p < n_pk; ++p) {
            for (auto& v : h_w) v = f2bf_h(wt(rng));
            d_w[p] = dmalloc<bf16>(w_elems);
            CK(hipMemcpy(d_w[p], h_w.data(), w_elems * 2, hipMemcpyHostToDevice));
            d_cA[p] = dmalloc<bf16>(s.N);
            d_cB[p] = dmalloc<bf16>(s.N);
        }
        for (auto& v : h_x) v = f2bf_h(act(rng));
        bf16* d_x = dmalloc<bf16>(s.K);
        CK(hipMemcpy(d_x, h_x.data(), s.K * 2, hipMemcpyHostToDevice));
        std::vector<Packet> h_pkA(n_pk), h_pkB(n_pk);
        for (unsigned p = 0; p < n_pk; ++p) {
            h_pkA[p] = Packet{d_w[p], d_cA[p], s.N, s.K};
            h_pkB[p] = Packet{d_w[p], d_cB[p], s.N, s.K};
        }
        Packet* d_pkA = dmalloc<Packet>(n_pk);
        Packet* d_pkB = dmalloc<Packet>(n_pk);
        CK(hipMemcpy(d_pkA, h_pkA.data(), n_pk * sizeof(Packet), hipMemcpyHostToDevice));
        CK(hipMemcpy(d_pkB, h_pkB.data(), n_pk * sizeof(Packet), hipMemcpyHostToDevice));
        unsigned long long* d_rec = dmalloc<unsigned long long>((size_t)n_pk * nblk * 2);
        std::vector<unsigned long long> h_rec((size_t)n_pk * nblk * 2);
        /* A 300 MB sweep between runs pushes the weights out of L2 and the MALL. */
        const size_t flush_elems = 150u << 20;
        bf16* d_flush = dmalloc<bf16>(flush_elems);

        std::printf("\n== %s: %u packets x %.1f KB/WG\n", s.name, n_pk,
                    (double)((s.N + nblk - 1) / nblk) * s.K * 2.0 / 1024.0);
        std::printf("%-10s %-4s %10s %10s %10s %12s\n", "period", "pf", "body mean", "body p50",
                    "pkt span", "makespan/pk");
        for (unsigned long long per : periods_us) {
            double bmean[2] = {0, 0}, bp50[2] = {0, 0}, span[2] = {0, 0}, mk[2] = {0, 0};
            for (unsigned pf = 0; pf < 2; ++pf) {
                std::vector<double> bodies, spans, makes;
                for (unsigned r = 0; r < reps; ++r) {
                    CK(hipMemset(d_flush, 0, flush_elems * 2));
                    CK(hipDeviceSynchronize());
                    /* t0 = device clock now + 200 us, so packet 0's gate is a real wait and
                     * the chain's schedule is anchored on the device clock, not the host's. */
                    unsigned long long t0 = 0;
                    k_probe<<<1, 1>>>(d_probe);
                    CK(hipMemcpy(&t0, d_probe, 8, hipMemcpyDeviceToHost));
                    t0 += 200ull * 100ull;
                    k_chain<<<nblk, PLOW_THREADS>>>(pf ? d_pkB : d_pkA, n_pk, d_x, d_gate, t0,
                                                    per * 100ull, pf, d_rec);
                    CK(hipDeviceSynchronize());
                    CK(hipMemcpy(h_rec.data(), d_rec, h_rec.size() * 8,
                                 hipMemcpyDeviceToHost));
                    unsigned long long first = ~0ull, last = 0;
                    for (unsigned p = 1; p < n_pk; ++p) { /* skip packet 0: launch ramp */
                        unsigned long long rmin = ~0ull, emax = 0;
                        for (unsigned wg = 0; wg < nblk; ++wg) {
                            const unsigned long long tr = h_rec[((size_t)p * nblk + wg) * 2];
                            const unsigned long long te = h_rec[((size_t)p * nblk + wg) * 2 + 1];
                            bodies.push_back((double)(te - tr) / 100.0);
                            rmin = std::min(rmin, tr);
                            emax = std::max(emax, te);
                        }
                        spans.push_back((double)(emax - rmin) / 100.0);
                        first = std::min(first, rmin);
                        last = std::max(last, emax);
                    }
                    makes.push_back((double)(last - first) / 100.0 / (double)(n_pk - 1));
                }
                std::sort(bodies.begin(), bodies.end());
                double bs = 0;
                for (double b : bodies) bs += b;
                bmean[pf] = bs / (double)bodies.size();
                bp50[pf] = bodies[bodies.size() / 2];
                double ss = 0;
                for (double v : spans) ss += v;
                span[pf] = ss / (double)spans.size();
                double ms = 0;
                for (double v : makes) ms += v;
                mk[pf] = ms / (double)makes.size();
            }
            for (unsigned pf = 0; pf < 2; ++pf)
                std::printf("%-10llu %-4s %8.2f us %8.2f us %8.2f us %10.2f us\n", per,
                            pf ? "on" : "off", bmean[pf], bp50[pf], span[pf], mk[pf]);
        }
        /* Exactness: on == off word for word. */
        uint64_t mism = 0;
        std::vector<uint16_t> a(s.N), b(s.N);
        for (unsigned p = 0; p < n_pk; ++p) {
            CK(hipMemcpy(a.data(), d_cA[p], s.N * 2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(b.data(), d_cB[p], s.N * 2, hipMemcpyDeviceToHost));
            for (unsigned n = 0; n < s.N; ++n)
                if (a[n] != b[n]) mism++;
        }
        std::printf("exact: %u x %u outputs, prefetch-on != off: %llu\n", n_pk, s.N,
                    (unsigned long long)mism);
        if (mism) return 1;
        for (unsigned p = 0; p < n_pk; ++p) {
            CK(hipFree(d_w[p]));
            CK(hipFree(d_cA[p]));
            CK(hipFree(d_cB[p]));
        }
        CK(hipFree(d_x));
        CK(hipFree(d_pkA));
        CK(hipFree(d_pkB));
        CK(hipFree(d_rec));
        CK(hipFree(d_flush));
    }
    std::printf("status 0\n");
    return 0;
}
