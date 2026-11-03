/* Standalone build (also use -arch=sm_120a):
 * nix develop --command bash -c 'env -i PATH="$PLOW_NVCC_PATH" ROCM_PATH="$ROCM_PATH" \
 *     NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS" "$PLOW_NVCC" -std=c++17 -arch=sm_90a \
 *     -I runtime/common -I runtime/nvidia runtime/tests/xreduce_body_nv.cu -o /tmp/xr_test'
 * The clean compiler environment matches runtime/cmake/nvcc_cubin.sh.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dev_isa.h"
#include "op_collective.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

__global__ void k_xreduce(PlowProgram prog, PlowDevInst in, void* out, void* resid,
                          void* out2, bool twoshot) {
    void* tensors[] = {out, resid, out2};
    if (twoshot) d_xreduce_twoshot_nv(prog, &in, tensors, blockIdx.x, gridDim.x);
    else d_xreduce_nv(prog, &in, tensors, blockIdx.x, gridDim.x);
}

static float from_bf16(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static uint16_t input(unsigned r, uint32_t e) {
    if (e % 16u == 0) {
        const uint16_t cancellation[] = {0x4c00, 0x3f80, 0xcc00, 0x3f80};
        return cancellation[r % 4];
    }
    if (e % 16u == 1) return r == 0 ? 0x3f80 : 0x3b80;
    uint32_t h = (e + 1u) * 0x9e3779b9u + (r + 1u) * 0x85ebca6bu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    return (uint16_t)((h & 0x807fu) | ((112u + ((h >> 8) % 30u)) << 7));
}

static void mismatch(const char* what, unsigned nr, unsigned rank, uint32_t n, uint32_t e0,
                     unsigned mode, uint32_t e, uint16_t got, uint16_t expected) {
    std::fprintf(stderr,
                 "%s: N=%u rank=%u n=%u e0=%u mode=%u e=%u got=%04x expected=%04x\n",
                 what, nr, rank, n, e0, mode, e, got, expected);
    std::exit(1);
}

static void run_case(unsigned nr, uint32_t n, uint32_t e0, bool staged) {
    constexpr uint32_t slot = 128;
    constexpr uint16_t guard = 0x5a5a;
    const uint32_t span = e0 + n + 8u;
    const size_t bytes = (size_t)span * sizeof(uint16_t);
    const size_t xoff = (slot + bytes + 127u) & ~size_t(127u);
    const size_t region = xoff + 3u * 128u;
    char *host, *mapped;
    CK(cudaHostAlloc(&host, nr * region, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer(&mapped, host, 0));
    std::vector<void*> peers(nr + 1, nullptr);
    std::vector<char*> local(nr, nullptr);
    if (staged)
        for (auto& p : local) CK(cudaMalloc(&p, region));
    for (unsigned r = 0; r < nr; r++) peers[r] = mapped + r * region;
    void** device_peers;
    CK(cudaMalloc(&device_peers, (nr + 1) * sizeof(void*)));
    CK(cudaMemcpy(device_peers, peers.data(), (nr + 1) * sizeof(void*), cudaMemcpyHostToDevice));

    std::vector<uint16_t> partials((size_t)nr * span), reduced(span), residual(span), fused(span);
    for (uint32_t e = 0; e < span; e++) {
        float sum = 0.0f;
        for (unsigned r = 0; r < nr; r++) {
            const uint16_t v = input(r, e);
            partials[(size_t)r * span + e] = v;
            sum += from_bf16(v);
        }
        reduced[e] = to_bf16(sum);
        residual[e] = input(9, e);
        fused[e] = to_bf16(from_bf16(residual[e]) + from_bf16(reduced[e]));
    }
    uint16_t *out, *resid, *out2;
    CK(cudaMalloc(&out, bytes));
    CK(cudaMalloc(&resid, bytes));
    CK(cudaMalloc(&out2, bytes));
    CK(cudaMemcpy(resid, residual.data(), bytes, cudaMemcpyHostToDevice));
    std::vector<uint16_t> initial(span, guard), got(span), untouched(span);

    // Preset gates cannot order concurrent RS writers. Run one rank at a time, with one
    // unstaged RS CTA; the staged relay orders multiple CTAs. Seed peers' completed bands.
    for (unsigned mode = 0; mode < 4; mode++) {
        const bool twoshot = mode != 0, use_out2 = mode >= 2;
        for (unsigned rank = 0; rank < nr; rank++) {
            std::memset(host, 0, nr * region);
            for (unsigned r = 0; r < nr; r++) {
                auto* data = (uint16_t*)(host + r * region + slot);
                std::memcpy(data, partials.data() + (size_t)r * span, bytes);
                if (twoshot && r != rank) {
                    const uint32_t lo = (uint64_t)n * r / nr;
                    const uint32_t hi = (uint64_t)n * (r + 1u) / nr;
                    for (uint32_t e = lo; e < hi; e++) data[e0 + e] = reduced[e0 + e];
                }
                auto* gates = (uint32_t*)(host + r * region + xoff);
                for (unsigned g = 0; g < 2; g++)
                    for (unsigned s = 0; s < nr; s++) PLOW_CTR(gates, g)[s] = 1u;
            }
            if (staged) {
                CK(cudaMemset(local[rank], 0, region));
                CK(cudaMemcpy(local[rank] + slot, partials.data() + (size_t)rank * span,
                              bytes, cudaMemcpyHostToDevice));
                peers[nr] = local[rank];
                CK(cudaMemcpy(device_peers, peers.data(), (nr + 1) * sizeof(void*),
                              cudaMemcpyHostToDevice));
                auto* own = (uint16_t*)(host + rank * region + slot);
                const uint32_t begin = twoshot ? e0 : 0;
                for (uint32_t e = begin; e < begin + n; e++) own[e] = guard;
            }
            CK(cudaMemcpy(out, initial.data(), bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(out2, initial.data(), bytes, cudaMemcpyHostToDevice));

            PlowProgram prog{};
            prog.peer_scratch = device_peers;
            prog.n_gpu = nr;
            prog.rank = rank;
            prog.xctr = (uint32_t*)(mapped + rank * region + xoff);
            PlowDevInst in{};
            for (auto& t : in.t) t = PLOW_TENSOR_NONE;
            in.t[0] = 0;
            if (mode == 2) in.t[1] = 1;
            if (use_out2) in.t[2] = 2;
            in.i[0] = n;
            in.i[1] = nr;
            in.i[2] = slot;
            in.i[3] = 0;
            in.i[4] = 1;
            in.i[5] = twoshot ? e0 : 0;
            in.fj[2].u = 3;
            // One-shot has no band operand; e0 instead exercises an unaligned local output.
            const unsigned nblk = twoshot && !staged ? 1u : 3u;
            k_xreduce<<<nblk, 256>>>(prog, in, twoshot ? out : out + e0, resid, out2, twoshot);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(got.data(), use_out2 ? out2 : out, bytes, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(untouched.data(), use_out2 ? out : out2, bytes, cudaMemcpyDeviceToHost));
            for (uint32_t e = 0; e < span; e++) {
                uint16_t expected = guard;
                if (e >= e0 && e < e0 + n)
                    expected = mode == 2 ? fused[e] : reduced[twoshot ? e : e - e0];
                if (got[e] != expected)
                    mismatch("output", nr, rank, n, e0, mode, e, got[e], expected);
                if (untouched[e] != guard)
                    mismatch("unused output", nr, rank, n, e0, mode, e, untouched[e], guard);
            }
            for (unsigned r = 0; r < nr; r++) {
                const auto* data = (const uint16_t*)(host + r * region + slot);
                const uint32_t lo = e0 + (uint64_t)n * r / nr;
                const uint32_t hi = e0 + (uint64_t)n * (r + 1u) / nr;
                for (uint32_t e = 0; e < span; e++) {
                    const uint16_t expected = twoshot && e >= lo && e < hi
                        ? reduced[e] : partials[(size_t)r * span + e];
                    if (data[e] != expected)
                        mismatch("scratch", nr, rank, n, e0, mode, e, data[e], expected);
                }
                const auto* gates = (const uint32_t*)(host + r * region + xoff);
                if (*PLOW_CTR(gates, 2) != 0 ||
                    PLOW_CTR(gates, 1)[kXrAggWord] != unsigned(twoshot && r == rank && !staged)) {
                    std::fprintf(stderr, "unexpected status/aggregation word\n");
                    std::exit(1);
                }
            }
            if (staged) {
                uint32_t lines[3 * PLOW_CTR_STRIDE];
                CK(cudaMemcpy(lines, local[rank] + xoff, sizeof(lines), cudaMemcpyDeviceToHost));
                if (lines[kXrAggWord] != nblk || lines[kXrRelayWord] != 1u ||
                    PLOW_CTR(lines, 1)[kXrAggWord] != (twoshot ? nblk : 0u) ||
                    PLOW_CTR(lines, 1)[kXrRelayWord] != unsigned(twoshot) ||
                    *PLOW_CTR(lines, 2) != 0u) {
                    std::fprintf(stderr, "unexpected local status/aggregation/relay word\n");
                    std::exit(1);
                }
            }
        }
    }
    for (auto* p : local) if (p) CK(cudaFree(p));
    CK(cudaFree(out));
    CK(cudaFree(resid));
    CK(cudaFree(out2));
    CK(cudaFree(device_peers));
    CK(cudaFreeHost(host));
    std::printf("PASS %s N=%u n=%u e0=%u (one-shot, two-shot, residual, absent residual)\n",
                staged ? "staged" : "host", nr, n, e0);
}

int main() {
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    const uint32_t lengths[] = {1, 7, 8, 9, 2560, 2561, 4096 * 2560 + 3};
    for (unsigned nr : {2u, 4u})
        for (uint32_t n : lengths)
            for (uint32_t e0 : {0u, 5u})
                for (bool staged : {false, true}) run_case(nr, n, e0, staged);
    return 0;
}
