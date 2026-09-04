/* L4 microbench: K3 latent MoeCombine folded into the tagged one-shot XReduce publish.
 *
 * ONE GPU. The `nranks` peers are `nranks` buffers on the same device, all ranks' workgroups
 * launched in one grid (`rank = blockIdx.x / nblk`, all co-resident), so the tagged protocol
 * runs unmodified through `d_xreduce_tagged_mega` while no fabric is involved — this measures
 * the fold's BODY (16 f32 slot loads per element on the publishing workgroups vs one bf16
 * copy) and proves bit-exactness; the packet/gate it removes is an interpreter cost, not
 * measurable here.
 *
 * Control = today's two packets: `d_moe_combine` into the plain slot (one launch), then the
 * copy-publish `d_xreduce_tagged_mega` (second launch). Fused = `d_xreduce_tagged_mega` with
 * `cpart/ck` set (one launch, plain slot never written). Every iteration uses random f32 slot
 * partials (random routing = random data; the combine order is fixed 0..k-1 by contract) and
 * compares every rank's output word-for-word against the control AND a CPU reference
 * (f32 slot sum -> bf16 -> f32 rank sum in rank order -> bf16).
 *
 * Build (gfx950):
 *   hipcc --offload-arch=gfx950 -O3 -std=c++17 -DPLOW_BUCKET_DECODE=1 -DPLOW_XR_TAGGED=1 \
 *     -DPLOW_XR_COMBINE_FOLD=1 -I runtime/amd -I runtime/common \
 *     runtime/bench/amd/xr_combine_fold_bench.hip -o /tmp/xr_combine_fold_bench
 * Run:  GPU_LEASE_TIMEOUT=7200 perf-data/tools/gpulease -n 1 l4 /tmp/xr_combine_fold_bench
 *   [n=3584] [k=16] [nranks=8] [nblk=7] [iters=200]
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "op_collective.h"
#include "op_moe.h"

#define CK(call)                                                                       \
    do {                                                                               \
        const hipError_t e_ = (call);                                                  \
        if (e_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__,                \
                         hipGetErrorString(e_));                                       \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

/* Per-peer buffer layout (identical offset in every peer, as in PeerLayout). */
constexpr uint32_t SLOT_OFF = 0;             /* plain bf16 partial slot */
constexpr uint32_t XCTR_OFF = 64u << 10;     /* audit counters: one 128 B line per gate */
constexpr uint32_t XCTR_LINES = 4096u;       /* 4 gates per iteration -> iters < 1024 */
constexpr uint32_t TAG_OFF = XCTR_OFF + XCTR_LINES * 128u; /* 4 tagged slots */
constexpr uint32_t PEER_BYTES = TAG_OFF + 4u * PLOW_XR_TAG_SLOT_BYTES;
constexpr uint64_t DEADLINE = 200000000ull; /* 2 s at 100 MHz */

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_combine(
    const void* const* peer_scratch, const float* part, uint32_t nranks, uint32_t n,
    uint32_t k, uint32_t nblk) {
    const uint32_t rank = blockIdx.x / nblk, slice = blockIdx.x % nblk;
    if (rank >= nranks) return;
    bf16* local = (bf16*)((char*)peer_scratch[rank] + SLOT_OFF);
    d_moe_combine(local, nullptr, nullptr, part + (size_t)rank * k * n, n, k, slice, nblk);
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS) void k_xreduce(
    bf16* out, const void* const* peer_scratch, const float* part, uint32_t nranks,
    uint32_t n, uint32_t k, uint32_t nblk, uint32_t gate, uint32_t* status, uint32_t fold) {
    const uint32_t rank = blockIdx.x / nblk, slice = blockIdx.x % nblk;
    if (rank >= nranks) return;
    d_xreduce_tagged_mega(out + (size_t)rank * n, peer_scratch, nranks, rank, n, SLOT_OFF,
                          XCTR_OFF, gate, DEADLINE, status + rank, slice, nblk, 0u, 0u, 0u,
                          TAG_OFF, PLOW_XR_TAG_SLOT_BYTES,
                          fold ? part + (size_t)rank * k * n : nullptr, fold ? k : 0u);
}

static float bf2f_h(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static uint16_t f2bf_h(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

int main(int argc, char** argv) {
    const uint32_t n = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 3584u;
    const uint32_t k = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 16u;
    const uint32_t nranks = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 8u;
    const uint32_t nblk = argc > 4 ? (uint32_t)std::atoi(argv[4]) : 7u;
    const uint32_t iters = argc > 5 ? (uint32_t)std::atoi(argv[5]) : 200u;
    if (nranks > PLOW_XRT_MAXR || n > PLOW_XR_TAG_SLOT_BYTES / 8 * 3 ||
        4u * iters + 4u > XCTR_LINES) {
        std::fprintf(stderr, "shape outside the tagged contract\n");
        return 2;
    }

    std::vector<void*> peers(nranks);
    for (uint32_t r = 0; r < nranks; r++) {
        CK(hipMalloc(&peers[r], PEER_BYTES));
        CK(hipMemset(peers[r], 0, PEER_BYTES));
    }
    void** d_peers;
    CK(hipMalloc(&d_peers, nranks * sizeof(void*)));
    CK(hipMemcpy(d_peers, peers.data(), nranks * sizeof(void*), hipMemcpyHostToDevice));
    const size_t part_elems = (size_t)nranks * k * n;
    float* d_part;
    CK(hipMalloc(&d_part, part_elems * 4));
    uint16_t *d_out_ctl, *d_out_fused;
    CK(hipMalloc(&d_out_ctl, (size_t)nranks * n * 2));
    CK(hipMalloc(&d_out_fused, (size_t)nranks * n * 2));
    uint32_t* d_status;
    CK(hipMalloc(&d_status, nranks * 4));
    CK(hipMemset(d_status, 0, nranks * 4));

    std::mt19937_64 rng(0x4c34u);
    std::uniform_real_distribution<float> uni(-4.0f, 4.0f);
    std::vector<float> h_part(part_elems);
    std::vector<uint16_t> h_ctl((size_t)nranks * n), h_fused((size_t)nranks * n);
    std::vector<uint16_t> h_ref((size_t)nranks * n);
    const dim3 grid(nranks * nblk), block(PLOW_THREADS);

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    double ms_ctl = 0, ms_fused = 0, ms_xr = 0;
    uint64_t mism_ctl = 0, mism_ref = 0, mism_ctl_ref = 0, mism_self = 0, checked = 0;
    unsigned shown = 0;
    uint16_t* d_out_ctl2;
    CK(hipMalloc(&d_out_ctl2, (size_t)nranks * n * 2));
    std::vector<uint16_t> h_fused2((size_t)nranks * n);
    /* Gate ids alternate parity per collective and the tag is gate+1, unique per
     * iteration: nothing is zeroed between iterations, exactly like consecutive layers. */
    uint32_t gate = 0;
    for (uint32_t it = 0; it < iters; it++) {
        for (auto& v : h_part) v = uni(rng);
        CK(hipMemcpy(d_part, h_part.data(), part_elems * 4, hipMemcpyHostToDevice));

        /* control: combine packet, then copy-publish XReduce packet */
        CK(hipEventRecord(e0));
        k_combine<<<grid, block>>>(d_peers, d_part, nranks, n, k, nblk);
        k_xreduce<<<grid, block>>>(d_out_ctl, d_peers, d_part, nranks, n, k, nblk, gate,
                                   d_status, 0u);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float ms;
        CK(hipEventElapsedTime(&ms, e0, e1));
        if (it >= 10) ms_ctl += ms;
        gate++;

        /* copy-publish XReduce alone (the packet the fold keeps) */
        CK(hipEventRecord(e0));
        k_xreduce<<<grid, block>>>(d_out_ctl, d_peers, d_part, nranks, n, k, nblk, gate,
                                   d_status, 0u);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        CK(hipEventElapsedTime(&ms, e0, e1));
        if (it >= 10) ms_xr += ms;
        gate++;

        /* fused: one packet, the plain slot is scribbled first so a copy-publish would be
         * caught */
        for (uint32_t r = 0; r < nranks; r++)
            CK(hipMemset((char*)peers[r] + SLOT_OFF, 0x5a, (size_t)n * 2));
        CK(hipEventRecord(e0));
        k_xreduce<<<grid, block>>>(d_out_fused, d_peers, d_part, nranks, n, k, nblk, gate,
                                   d_status, 1u);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        CK(hipEventElapsedTime(&ms, e0, e1));
        if (it >= 10) ms_fused += ms;
        gate++;

        /* fused again on the same data: a self-inconsistent result is a race, not rounding */
        k_xreduce<<<grid, block>>>(d_out_ctl2, d_peers, d_part, nranks, n, k, nblk, gate,
                                   d_status, 1u);
        CK(hipDeviceSynchronize());
        gate++;

        CK(hipMemcpy(h_ctl.data(), d_out_ctl, h_ctl.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_fused.data(), d_out_fused, h_fused.size() * 2, hipMemcpyDeviceToHost));
        CK(hipMemcpy(h_fused2.data(), d_out_ctl2, h_fused2.size() * 2, hipMemcpyDeviceToHost));
        for (uint32_t h = 0; h < n; h++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < nranks; r++) {
                float c = 0.0f;
                for (uint32_t s = 0; s < k; s++) c += h_part[((size_t)r * k + s) * n + h];
                acc += bf2f_h(f2bf_h(c));
            }
            const uint16_t ref = f2bf_h(acc);
            for (uint32_t r = 0; r < nranks; r++) {
                const size_t i = (size_t)r * n + h;
                const bool bad = h_fused[i] != h_ctl[i] || h_fused[i] != ref;
                mism_ctl += h_fused[i] != h_ctl[i];
                mism_ref += h_fused[i] != ref;
                mism_ctl_ref += h_ctl[i] != ref;
                mism_self += h_fused[i] != h_fused2[i];
                checked++;
                if (bad && shown < 8) {
                    shown++;
                    std::printf("  mismatch it=%u rank=%u h=%u fused=%04x fused2=%04x ctl=%04x "
                                "ref=%04x\n",
                                it, r, h, h_fused[i], h_fused2[i], h_ctl[i], ref);
                }
            }
        }
    }
    std::vector<uint32_t> st(nranks);
    CK(hipMemcpy(st.data(), d_status, nranks * 4, hipMemcpyDeviceToHost));
    uint32_t bad = 0;
    for (auto s : st) bad |= s;
    const double timed = iters > 10 ? (double)(iters - 10) : 1.0;
    std::printf("xr_combine_fold n=%u k=%u nranks=%u nblk=%u iters=%u\n", n, k, nranks, nblk,
                iters);
    std::printf("  control  (combine + copy-publish XReduce, 2 launches): %.3f us\n",
                1e3 * ms_ctl / timed);
    std::printf("  xreduce  (copy-publish XReduce alone, 1 launch):       %.3f us\n",
                1e3 * ms_xr / timed);
    std::printf("  fused    (combine-in-publish XReduce, 1 launch):       %.3f us\n",
                1e3 * ms_fused / timed);
    std::printf("  exact: %llu words checked; fused != control %llu, fused != cpu ref %llu, "
                "control != cpu ref %llu, fused != fused(rerun) %llu; status 0x%08x\n",
                (unsigned long long)checked, (unsigned long long)mism_ctl,
                (unsigned long long)mism_ref, (unsigned long long)mism_ctl_ref,
                (unsigned long long)mism_self, bad);
    return (mism_ctl || mism_ref || mism_self || bad) ? 1 : 0;
}
