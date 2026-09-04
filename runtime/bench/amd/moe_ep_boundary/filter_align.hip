#include <hip/hip_runtime.h>

struct RouteEntry {
    unsigned expert;
    float gate;
};

extern "C" __device__ unsigned plow_moe_ep_filter_align_abi_1 = 1;
extern "C" __device__ unsigned plow_moe_ep_filter_align_wave64_1 = 1;
extern "C" __device__ unsigned plow_moe_ep_filter_align_stable_1 = 1;
extern "C" __device__ unsigned plow_moe_ep_filter_align_parallel_1 = 1;
extern "C" __device__ unsigned plow_moe_ep_filter_align_no_spill_1 = 1;

extern "C" __global__ __launch_bounds__(256, 2)
void plow_moe_ep_filter_align_gfx950(
    const RouteEntry* __restrict__ routes,
    unsigned* __restrict__ meta,
    unsigned* __restrict__ partial,
    unsigned* __restrict__ row_token,
    unsigned* __restrict__ row_partidx,
    float* __restrict__ row_gate,
    unsigned tokens,
    unsigned topk,
    unsigned experts,
    unsigned expert_begin,
    unsigned expert_end,
    unsigned row_capacity,
    unsigned phase,
    unsigned npart) {
    const unsigned tid = threadIdx.x;
    const unsigned slice = blockIdx.x;
    const size_t nroutes = static_cast<size_t>(tokens) * topk;
    unsigned* rowoff = meta;
    unsigned* count = meta + experts;
    unsigned* tilep = meta + 2u * experts;
    __shared__ unsigned scratch[1024];

    if (expert_end <= expert_begin || expert_end > experts || experts > 1024u || npart == 0)
        return;
    if (phase == 1u) {
        for (unsigned e = tid; e < experts; e += blockDim.x) scratch[e] = 0u;
        __syncthreads();
        const size_t first = nroutes * slice / npart;
        const size_t last = nroutes * (slice + 1u) / npart;
        for (size_t p = first + tid; p < last; p += blockDim.x) {
            const unsigned expert = routes[p].expert;
            if (expert >= expert_begin && expert < expert_end)
                atomicAdd(&scratch[expert], 1u);
        }
        __syncthreads();
        for (unsigned e = tid; e < experts; e += blockDim.x)
            partial[static_cast<size_t>(slice) * experts + e] = scratch[e];
        return;
    }
    if (phase == 2u) {
        if (slice != 0) return;
        const unsigned chunk = (experts + blockDim.x - 1u) / blockDim.x;
        const unsigned begin = tid * chunk;
        const unsigned end = min(begin + chunk, experts);
        unsigned tiles = 0u;
        for (unsigned e = begin; e < end; ++e) {
            unsigned total = 0u;
            for (unsigned b = 0; b < npart; ++b) {
                const size_t at = static_cast<size_t>(b) * experts + e;
                const unsigned n = partial[at];
                partial[at] = total;
                total += n;
            }
            count[e] = total;
            tiles += (total + 63u) / 64u;
        }
        scratch[tid] = tiles;
        __syncthreads();
        for (unsigned offset = 1u; offset < blockDim.x; offset <<= 1u) {
            const unsigned add = tid >= offset ? scratch[tid - offset] : 0u;
            __syncthreads();
            scratch[tid] += add;
            __syncthreads();
        }
        unsigned prefix = tid == 0 ? 0u : scratch[tid - 1u];
        for (unsigned e = begin; e < end; ++e) {
            tilep[e] = prefix;
            rowoff[e] = prefix * 64u;
            prefix += (count[e] + 63u) / 64u;
        }
        if (tid == blockDim.x - 1u)
            tilep[experts] = scratch[tid] * 64u <= row_capacity ? scratch[tid] : ~0u;
        return;
    }
    if (tilep[experts] == ~0u) return;
    const unsigned rows = tilep[experts] * 64u;
    if (phase == 3u) {
        for (unsigned row = slice * blockDim.x + tid; row < rows;
             row += npart * blockDim.x) {
            row_token[row] = ~0u;
            row_partidx[row] = ~0u;
            row_gate[row] = 0.0f;
        }
        return;
    }
    if (phase != 4u || tid >= 64u) return;
    const unsigned lane = tid;
    const size_t first = nroutes * slice / npart;
    const size_t last = nroutes * (slice + 1u) / npart;
    for (size_t base = first; base < last; base += 64u) {
        const size_t p = base + lane;
        const unsigned global = p < last ? routes[p].expert : ~0u;
        const unsigned owned = global >= expert_begin && global < expert_end ? global : ~0u;
        const unsigned long long peers = __match_any(owned);
        const unsigned leader = __builtin_ctzll(peers);
        const unsigned rank = __builtin_popcountll(peers & ((1ull << lane) - 1ull));
        unsigned pos0 = 0u;
        if (lane == leader && owned < experts) {
            const size_t at = static_cast<size_t>(slice) * experts + owned;
            pos0 = rowoff[owned] + partial[at];
            partial[at] += __builtin_popcountll(peers);
        }
        pos0 = __shfl(pos0, leader, 64);
        if (owned < experts) {
            const unsigned row = pos0 + rank;
            row_token[row] = static_cast<unsigned>(p / topk);
            row_partidx[row] = static_cast<unsigned>(p);
            row_gate[row] = routes[p].gate;
        }
    }
}
