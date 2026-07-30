/* k3_attn_res_bench_gfx950.hip — AttnRes (op 104): CORRECTNESS + LATENCY, on hardware.
 *
 * AttnRes runs 186 times per K3 token (2 per layer x 93 layers) on ONE workgroup, because both
 * reductions span the full 7168-wide row and the softmax couples the rows
 * (`crates/devgen/src/k3.rs::emit_attn_res`). `perf-data/kimi-k3-kernel-gap.md` §10 item 7 forbids
 * splitting it into a second packet, so the only lever is the body of the single workgroup.
 *
 * This file holds BOTH bodies:
 *   d_attn_res_v0 — a VERBATIM copy of the pre-optimisation arm, kept so the speedup is measured
 *                   against something that still exists rather than against a remembered number.
 *   d_attn_res    — the live arm, from op_k3.h.
 * Both are checked against a double-precision CPU oracle, and v1 is additionally checked to agree
 * with v0 to well inside the gate's 1.5e-2.
 *
 * WHY THE TIMING LOOP IS INSIDE THE KERNEL. The body is single-digit microseconds and a HIP launch
 * is ~2-5 us, so back-to-back launches measure the launch. In the real program the op is a PACKET
 * inside a persistent megakernel — no launch at all — so an in-kernel repeat loop is the honest
 * model of it. Two residencies are reported because they bracket reality: the block-residual rows
 * were written by an earlier packet (L2-hot) while `score_w` is a per-site weight (5.3 MB across
 * the 186 sites, so MALL-warm at best).
 *
 *   /opt/rocm/bin/hipcc --offload-arch=gfx950 -O3 -w -DPLOW_K3=1 \
 *       -I runtime/amd -I runtime/common runtime/tests/k3_attn_res_bench_gfx950.hip -o /tmp/k3ar
 *   /tmp/k3ar
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "amd_common.h"
#include "op_k3.h"

#define HC(x)                                                                                   \
    do {                                                                                        \
        hipError_t e_ = (x);                                                                    \
        if (e_) {                                                                               \
            printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__);                            \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)

/* ------------------------------------------------------------------ the OLD body, verbatim. */
__device__ void d_attn_res_v0(bf16* __restrict__ out, const bf16* __restrict__ prefix,
                              const bf16* __restrict__ blkres, const float* __restrict__ score_w,
                              unsigned T, unsigned HID, unsigned NB, float eps, unsigned slice,
                              unsigned nblk, float* __restrict__ lds) {
    if (NB > PLOW_ATTNRES_MAXB) return;

    float* part = lds;
    float* sco = lds + PLOW_WAVES;

    for (unsigned t = slice; t < T; t += nblk) {
        const size_t pofs = (size_t)t * HID;
        const size_t bofs = (size_t)t * NB * HID;

        for (unsigned r = 0; r <= NB; r++) {
            const bf16* vr = (r < NB) ? blkres + bofs + (size_t)r * HID : prefix + pofs;
            float ss = 0.0f, sw = 0.0f;
            for (unsigned d = threadIdx.x; d < HID; d += PLOW_THREADS) {
                const float x = bf2f(vr[d]);
                ss += x * x;
                sw += x * score_w[d];
            }
            ss = block_sum(ss, part);
            sw = block_sum(sw, part);
            if (threadIdx.x == 0) sco[r] = sw * rsqrtf(ss / (float)HID + eps);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float m = -INFINITY;
            for (unsigned r = 0; r <= NB; r++) m = fmaxf(m, sco[r]);
            float s = 0.0f;
            for (unsigned r = 0; r <= NB; r++) {
                const float e = __expf(sco[r] - m);
                sco[r] = e;
                s += e;
            }
            const float inv = 1.0f / s;
            for (unsigned r = 0; r <= NB; r++) sco[r] *= inv;
        }
        __syncthreads();

        for (unsigned d = threadIdx.x; d < HID; d += PLOW_THREADS) {
            float acc = 0.0f;
            for (unsigned r = 0; r < NB; r++)
                acc += sco[r] * bf2f(blkres[bofs + (size_t)r * HID + d]);
            acc += sco[NB] * bf2f(prefix[pofs + d]);
            out[pofs + d] = f2bf(acc);
        }
        __syncthreads();
    }
}

/* The interpreter's LDS arena, verbatim from interp.hip's `plow_smem` (144 KiB on the decode
 * object; read out of the built code object's .group_segment_fixed_size = 147464). */
enum { BENCH_LDS_BYTES = 147464 };

/* ------------------------------------------------------------------------------- test kernels */
struct Rep {
    const bf16* prefix;
    const bf16* blkres;
    const float* score_w;
    bf16* out;
};

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void t_v0(Rep* reps, unsigned nrep,
                                                                     unsigned T, unsigned HID,
                                                                     unsigned NB, float eps,
                                                                     unsigned iters) {
    __shared__ __align__(16) char raw[BENCH_LDS_BYTES];
    for (unsigned i = 0; i < iters; i++) {
        const Rep r = reps[i % nrep];
        d_attn_res_v0(r.out, r.prefix, r.blkres, r.score_w, T, HID, NB, eps, blockIdx.x, gridDim.x,
                      (float*)raw);
        __syncthreads();
    }
}

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void t_v1(Rep* reps, unsigned nrep,
                                                                     unsigned T, unsigned HID,
                                                                     unsigned NB, float eps,
                                                                     unsigned iters) {
    __shared__ __align__(16) char raw[BENCH_LDS_BYTES];
    for (unsigned i = 0; i < iters; i++) {
        const Rep r = reps[i % nrep];
        /* NBCAP == NB: the bench allocates the ring at exactly the live count, which is the one
         * capacity a T = 1 measurement can distinguish from nothing. No push — the snapshot arm is
         * a separate store pass and is not what this bench times. */
        d_attn_res(r.out, r.prefix, r.blkres, r.score_w, T, HID, NB, NB, eps, blockIdx.x,
                   gridDim.x, (float*)raw, nullptr, 0u);
        __syncthreads();
    }
}

/* ---------------------------------------------------------------------------------------------
 * THE COST OF THE THING THIS OP DOES NOT DO: a CROSS-WORKGROUP BARRIER.
 *
 * The obvious structural fix for a 1-of-256-CU op is to split the 7168 reduction across N blocks.
 * §10 item 7 forbids finishing it in a SECOND PACKET, so the split would have to barrier INSIDE
 * the packet — every participating workgroup bumps a device-scope counter, spins until all N have
 * arrived, and takes one acquire fence. That is legal here (the interpreter is persistent, one
 * workgroup per CU, so all N are co-resident by construction) and it is what this measures.
 *
 * TWO barriers are needed per invocation: one after the partial statistics, one after the softmax
 * — the same protocol interp.hip's `ctr_poll`/`ctr_signal` use between packets, including the
 * agent-scope acquire whose `buffer_inv` that file's comment calls out as the expensive part.
 * If 2 barriers cost more than the op does, the split cannot win however perfectly it scales. */
__device__ __forceinline__ void bench_ctr_wait(uint32_t* ctr, uint32_t target) {
    while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target)
        __builtin_amdgcn_s_sleep(8);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

__global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void t_barrier(uint32_t* ctr,
                                                                          unsigned iters) {
    const unsigned n = gridDim.x;
    for (unsigned i = 0; i < iters; i++) {
        __syncthreads();
        if (threadIdx.x == 0)
            __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        if (threadIdx.x == 0) bench_ctr_wait(ctr, (i + 1) * n);
        __syncthreads();
    }
}

/* ------------------------------------------------------------------------------- CPU oracle. */
static float bf2f_h(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}
static bf16 f2bf_h(float f) {
    unsigned u;
    __builtin_memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

static void oracle(std::vector<float>& out, const std::vector<bf16>& prefix,
                   const std::vector<bf16>& blkres, const std::vector<float>& sw, unsigned T,
                   unsigned HID, unsigned NB, float eps) {
    out.assign((size_t)T * HID, 0.0f);
    std::vector<double> sco(NB + 1);
    for (unsigned t = 0; t < T; t++) {
        const size_t pofs = (size_t)t * HID, bofs = (size_t)t * NB * HID;
        for (unsigned r = 0; r <= NB; r++) {
            const bf16* vr = (r < NB) ? &blkres[bofs + (size_t)r * HID] : &prefix[pofs];
            double ss = 0.0, s = 0.0;
            for (unsigned d = 0; d < HID; d++) {
                const double x = bf2f_h(vr[d]);
                ss += x * x;
                s += x * (double)sw[d];
            }
            sco[r] = s / std::sqrt(ss / (double)HID + (double)eps);
        }
        double m = -1e300;
        for (unsigned r = 0; r <= NB; r++) m = std::max(m, sco[r]);
        double z = 0.0;
        for (unsigned r = 0; r <= NB; r++) {
            sco[r] = std::exp(sco[r] - m);
            z += sco[r];
        }
        for (unsigned r = 0; r <= NB; r++) sco[r] /= z;
        for (unsigned d = 0; d < HID; d++) {
            double acc = 0.0;
            for (unsigned r = 0; r < NB; r++)
                acc += sco[r] * bf2f_h(blkres[bofs + (size_t)r * HID + d]);
            acc += sco[NB] * bf2f_h(prefix[pofs + d]);
            out[pofs + d] = (float)acc;
        }
    }
}

/* ------------------------------------------------------------------------------------ driver */
static unsigned long long rng = 0x9e3779b97f4a7c15ull;
static float frand(float lo, float hi) {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return lo + (hi - lo) * (float)((rng >> 40) / 16777216.0);
}

struct Res {
    double us_hot, us_mix, us_cold;
    double maxrel;
};

static Res run(int which, unsigned T, unsigned HID, unsigned NB, float eps, bool verify) {
    /* Replica count for the COLD residency: enough of a working set to blow past a 4 MiB XCD L2.
     * Sized in bytes so nb=1 and nb=8 see the same footprint, not the same replica count. */
    const size_t bytes_one =
        (size_t)T * HID * 2 + (size_t)T * NB * HID * 2 + (size_t)HID * 4 + (size_t)T * HID * 2;
    const unsigned nrep = (unsigned)std::max<size_t>(2, std::min<size_t>(256, (64u << 20) / bytes_one));

    std::vector<bf16> h_pre((size_t)T * HID), h_blk((size_t)T * NB * HID);
    std::vector<float> h_sw(HID);
    for (auto& v : h_pre) v = f2bf_h(frand(-2.0f, 2.0f));
    for (auto& v : h_blk) v = f2bf_h(frand(-2.0f, 2.0f));
    for (auto& v : h_sw) v = frand(-0.05f, 0.05f);

    std::vector<Rep> h_reps(nrep);
    for (unsigned i = 0; i < nrep; i++) {
        bf16 *dp, *db, *dout;
        float* dsw;
        HC(hipMalloc(&dp, h_pre.size() * 2));
        HC(hipMalloc(&db, std::max<size_t>(1, h_blk.size()) * 2));
        HC(hipMalloc(&dsw, HID * 4));
        HC(hipMalloc(&dout, h_pre.size() * 2));
        HC(hipMemcpy(dp, h_pre.data(), h_pre.size() * 2, hipMemcpyHostToDevice));
        if (!h_blk.empty())
            HC(hipMemcpy(db, h_blk.data(), h_blk.size() * 2, hipMemcpyHostToDevice));
        HC(hipMemcpy(dsw, h_sw.data(), HID * 4, hipMemcpyHostToDevice));
        HC(hipMemset(dout, 0, h_pre.size() * 2));
        h_reps[i] = Rep{dp, db, dsw, dout};
    }
    /* THE RESIDENCY THAT MATCHES K3 DECODE. The block-residual rows and the prefix were written
     * by an EARLIER PACKET of the same token, so they are L2-hot; `score_w` is a per-site weight
     * and there are 186 distinct ones (5.3 MB/token), so it is not. `d_mix` rotates score_w only,
     * over replicas that share replica 0's rows — which is what separates a loop nest that reads
     * score_w ONCE from one that reads it (nb+1) times. */
    std::vector<Rep> h_mix(nrep);
    for (unsigned i = 0; i < nrep; i++)
        h_mix[i] = Rep{h_reps[0].prefix, h_reps[0].blkres, h_reps[i].score_w, h_reps[0].out};

    Rep *d_reps, *d_mix;
    HC(hipMalloc(&d_reps, nrep * sizeof(Rep)));
    HC(hipMemcpy(d_reps, h_reps.data(), nrep * sizeof(Rep), hipMemcpyHostToDevice));
    HC(hipMalloc(&d_mix, nrep * sizeof(Rep)));
    HC(hipMemcpy(d_mix, h_mix.data(), nrep * sizeof(Rep), hipMemcpyHostToDevice));

    auto launch = [&](Rep* rp, unsigned nr, unsigned iters) {
        if (which == 0)
            hipLaunchKernelGGL(t_v0, dim3(1), dim3(PLOW_THREADS), 0, 0, rp, nr, T, HID, NB, eps,
                               iters);
        else
            hipLaunchKernelGGL(t_v1, dim3(1), dim3(PLOW_THREADS), 0, 0, rp, nr, T, HID, NB, eps,
                               iters);
    };

    Res res{0, 0, 0, 0};

    /* correctness on replica 0 */
    launch(d_reps, 1, 1);
    HC(hipDeviceSynchronize());
    if (verify) {
        std::vector<bf16> got((size_t)T * HID);
        HC(hipMemcpy(got.data(), h_reps[0].out, got.size() * 2, hipMemcpyDeviceToHost));
        std::vector<float> ref;
        oracle(ref, h_pre, h_blk, h_sw, T, HID, NB, eps);
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            const double g = bf2f_h(got[i]), r = ref[i];
            const double den = std::max(1e-3, std::fabs(r));
            worst = std::max(worst, std::fabs(g - r) / den);
        }
        res.maxrel = worst;
    }

    hipEvent_t a, b;
    HC(hipEventCreate(&a));
    HC(hipEventCreate(&b));
    auto timeit = [&](Rep* rp, unsigned nr) {
        const unsigned WARM = 200, N1 = 400, N2 = 1400;
        launch(rp, nr, WARM);
        HC(hipDeviceSynchronize());
        float t1 = 0, t2 = 0;
        HC(hipEventRecord(a));
        launch(rp, nr, N1);
        HC(hipEventRecord(b));
        HC(hipEventSynchronize(b));
        HC(hipEventElapsedTime(&t1, a, b));
        HC(hipEventRecord(a));
        launch(rp, nr, N2);
        HC(hipEventRecord(b));
        HC(hipEventSynchronize(b));
        HC(hipEventElapsedTime(&t2, a, b));
        /* Slope, so the launch constant cancels. */
        return (double)(t2 - t1) * 1000.0 / (double)(N2 - N1);
    };
    res.us_hot = timeit(d_reps, 1);
    res.us_mix = timeit(d_mix, nrep);
    res.us_cold = timeit(d_reps, nrep);

    for (unsigned i = 0; i < nrep; i++) {
        HC(hipFree((void*)h_reps[i].prefix));
        HC(hipFree((void*)h_reps[i].blkres));
        HC(hipFree((void*)h_reps[i].score_w));
        HC(hipFree(h_reps[i].out));
    }
    HC(hipFree(d_reps));
    HC(hipFree(d_mix));
    HC(hipEventDestroy(a));
    HC(hipEventDestroy(b));
    return res;
}

static double barrier_us(unsigned nblk) {
    uint32_t* ctr;
    HC(hipMalloc(&ctr, 4));
    HC(hipMemset(ctr, 0, 4));
    hipEvent_t a, b;
    HC(hipEventCreate(&a));
    HC(hipEventCreate(&b));
    const unsigned N1 = 200, N2 = 1200;
    float t1 = 0, t2 = 0;
    hipLaunchKernelGGL(t_barrier, dim3(nblk), dim3(PLOW_THREADS), 0, 0, ctr, 200);
    HC(hipDeviceSynchronize());
    HC(hipMemset(ctr, 0, 4));
    HC(hipEventRecord(a));
    hipLaunchKernelGGL(t_barrier, dim3(nblk), dim3(PLOW_THREADS), 0, 0, ctr, N1);
    HC(hipEventRecord(b));
    HC(hipEventSynchronize(b));
    HC(hipEventElapsedTime(&t1, a, b));
    HC(hipMemset(ctr, 0, 4));
    HC(hipEventRecord(a));
    hipLaunchKernelGGL(t_barrier, dim3(nblk), dim3(PLOW_THREADS), 0, 0, ctr, N2);
    HC(hipEventRecord(b));
    HC(hipEventSynchronize(b));
    HC(hipEventElapsedTime(&t2, a, b));
    HC(hipFree(ctr));
    HC(hipEventDestroy(a));
    HC(hipEventDestroy(b));
    return (double)(t2 - t1) * 1000.0 / (double)(N2 - N1);
}

int main(int argc, char** argv) {
    const float eps = 1e-6f;
    const unsigned HID = argc > 1 ? (unsigned)atoi(argv[1]) : 7168u;
    printf("AttnRes bench — gfx950, %u threads / %u waves, HID=%u\n", PLOW_THREADS, PLOW_WAVES,
           HID);
    printf("%-4s %-3s | %-30s | %-30s | maxrel old/new\n", "T", "nb",
           "old us (hot / mix / cold)", "new us (hot / mix / cold)");
    /* nb 0..8 is K3's whole range: `nb = ceil(layer / attn_res_block_size)` with 93 layers and a
     * block size of 12. T=8 at nb=8 is the batched shape, to show the emitter's `blocks = min(T,
     * n_cu)` still scales. */
    const unsigned Ts[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 8};
    const unsigned NBs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 8};
    const int NC = 10;
    int bad = 0;
    double hot0[9], hot1[9], mix0[9], mix1[9];
    for (int c = 0; c < NC; c++) {
        const unsigned T = Ts[c], NB = NBs[c];
        Res r0 = run(0, T, HID, NB, eps, true);
        Res r1 = run(1, T, HID, NB, eps, true);
        if (T == 1) {
            hot0[NB] = r0.us_hot;
            hot1[NB] = r1.us_hot;
            mix0[NB] = r0.us_mix;
            mix1[NB] = r1.us_mix;
        }
        printf("%-4u %-3u | %7.3f /%7.3f /%7.3f  | %7.3f /%7.3f /%7.3f  | %.1e/%.1e %s %.2fx/%.2fx\n",
               T, NB, r0.us_hot, r0.us_mix, r0.us_cold, r1.us_hot, r1.us_mix, r1.us_cold, r0.maxrel,
               r1.maxrel, (r1.maxrel < 1.5e-2 ? "OK " : "FAIL"), r0.us_hot / r1.us_hot,
               r0.us_mix / r1.us_mix);
        if (!(r1.maxrel < 1.5e-2) || !(r0.maxrel < 1.5e-2)) bad = 1;
    }

    /* K3 decode, one token: 93 layers x 2 AttnRes sites, `nb = ceil(l / 12)` at layer l
     * (crates/devgen/src/k3.rs: `n_block_res` / `attn_res_block_size = 12`). */
    double tok0 = 0, tok1 = 0, tkm0 = 0, tkm1 = 0;
    for (unsigned l = 0; l < 93; l++) {
        const unsigned nb = (l + 11u) / 12u;
        tok0 += 2.0 * hot0[nb];
        tok1 += 2.0 * hot1[nb];
        tkm0 += 2.0 * mix0[nb];
        tkm1 += 2.0 * mix1[nb];
    }
    printf("\nK3 decode, 186 AttnRes sites/token (nb = ceil(l/12), 93 layers):\n");
    printf("  all L2-hot     old %7.3f ms/token   new %7.3f ms/token   %.2fx\n", tok0 / 1000.0,
           tok1 / 1000.0, tok0 / tok1);
    printf("  rows hot, w cold  old %7.3f ms/token   new %7.3f ms/token   %.2fx   <- the real one\n",
           tkm0 / 1000.0, tkm1 / 1000.0, tkm0 / tkm1);

    /* The alternative that is not taken. */
    printf("\nCross-workgroup barrier (device-scope counter + agent acquire, the ONLY in-packet\n"
           "way to split the reduction across CUs). A split needs TWO per invocation:\n");
    for (unsigned n : {2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u})
        printf("  %3u blocks: %7.3f us / barrier  -> %7.3f us of barrier per AttnRes\n", n,
               barrier_us(n), 2.0 * barrier_us(n));

    printf(bad ? "\nFAIL\n" : "\nPASS\n");
    return bad;
}
