#include "amd_common.h"
#include "op_collective.h"
#include "op_k3.h"

#define D6_DOMAINS 8u
#define D6_MAXB 16u
#define D6_LINE 32u
#define D6_GRID 14u

struct D6Sync {
    float stats[D6_DOMAINS][2u * (D6_MAXB + 1u)];
    float probs[D6_MAXB + 1u];
    float norm[D6_DOMAINS];
    float inv;
    uint32_t phase[5][D6_LINE];
    uint32_t local_arrive[D6_DOMAINS][D6_LINE];
    uint32_t local_done[D6_DOMAINS][D6_LINE];
    uint32_t control[2][D6_LINE];
    uint32_t successor[D6_LINE];
};

struct D6Args {
    bf16* reduced;
    bf16* prefix;
    bf16* mixed;
    bf16* block_residual;
    const float* score_weight;
    const bf16* gamma;
    const bf16* residual_other;
    const bf16* residual_pre;
    const bf16* push_src;
    const void* const* peer_scratch;
    uint32_t* const* peer_gate;
    D6Sync* sync;
    uint64_t* cycles;
    uint32_t* status;
    uint32_t nranks;
    uint32_t rank;
    uint32_t hidden;
    uint32_t nb;
    uint32_t nb_cap;
    uint32_t contract;
    uint32_t push;
    uint32_t use_gamma;
    uint32_t iters;
    uint64_t deadline;
    float eps;
};
static_assert(sizeof(D6Args) == 168, "D6Args ABI");
static_assert(sizeof(D6Sync) == 4264, "D6Sync ABI");

__device__ __forceinline__ void d6_wait(uint32_t* p, uint32_t target, const D6Args& a) {
    const uint64_t start = __builtin_amdgcn_s_memrealtime();
    while (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
        if (__builtin_amdgcn_s_memrealtime() - start > a.deadline) {
            if (a.status) *a.status = 0xD6000000u | (blockIdx.x & 0xffffu);
            break;
        }
        __builtin_amdgcn_s_sleep(8);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}
__device__ __forceinline__ void d6_release(uint32_t* p) {
    __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ const bf16* d6_materialize(const D6Args& a, unsigned domain) {
    if (a.contract == 0u) return a.reduced;
    const unsigned lane = threadIdx.x & 63u;
    if (threadIdx.x < 64u) {
        const unsigned old_thread = domain * 64u + lane;
        const auto* rg = as_glob(a.reduced);
        const auto* og = as_glob(a.residual_other);
        const auto* pg = as_glob(a.residual_pre);
        auto* dst = as_glob(a.prefix);
        for (unsigned d = old_thread * 8u; d < a.hidden; d += 512u * 8u) {
            const unsigned lim = min(8u, a.hidden - d);
            for (unsigned j = 0; j < lim; ++j) {
                const bf16 aa = a.contract == 1u ? rg[d + j] : og[d + j];
                const bf16 bb = a.contract == 1u ? og[d + j] : rg[d + j];
                const bf16 inner = f2bf(bf2f(aa) + bf2f(bb));
                dst[d + j] = a.residual_pre
                                 ? f2bf(bf2f(pg[d + j]) + bf2f(inner))
                                 : inner;
            }
        }
    }
    return a.prefix;
}

__device__ __forceinline__ void d6_attn_leader(const D6Args& a, unsigned domain,
                                                uint32_t iter) {
    D6Sync* s = a.sync;
    const unsigned lane = threadIdx.x & 63u;
    const unsigned wave = threadIdx.x >> 6;
    const bool active = wave == 0u;
    const unsigned old_thread = domain * 64u + lane;
    const bf16* prefix = d6_materialize(a, domain);
    if (a.push && active)
        for (unsigned d = old_thread; d < a.hidden; d += 512u)
            a.block_residual[(size_t)a.nb * a.hidden + d] = a.push_src[d];
    __syncthreads();

    const unsigned nv = (a.hidden & 7u) ? 0u : a.hidden >> 3;
    const unsigned pstr = 2u * (a.nb + 1u);
    if (active) {
        for (unsigned g = 0; g <= a.nb; g += PLOW_ATTNRES_RG) {
            float ss[PLOW_ATTNRES_RG], sw[PLOW_ATTNRES_RG];
#pragma unroll
            for (int k = 0; k < PLOW_ATTNRES_RG; ++k) ss[k] = sw[k] = 0.0f;
            for (unsigned i = old_thread; i < nv; i += 512u) {
                const f32x4 w0 = *(const PLOW_GLOB f32x4*)(a.score_weight + i * 8u);
                const f32x4 w1 = *(const PLOW_GLOB f32x4*)(a.score_weight + i * 8u + 4u);
#pragma unroll
                for (int k = 0; k < PLOW_ATTNRES_RG; ++k) {
                    const unsigned r = g + (unsigned)k;
                    if (r > a.nb) break;
                    const bf16* row = r < a.nb
                                          ? a.block_residual + (size_t)r * a.hidden
                                          : prefix;
                    const bf16v8 v = ld_glob8(row + i * 8u);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float x = bf2f(v[j]);
                        ss[k] += x * x;
                        sw[k] += x * (j < 4 ? w0[j] : w1[j & 3]);
                    }
                }
            }
            for (unsigned d = nv * 8u + old_thread; d < a.hidden; d += 512u) {
                const float wd = a.score_weight[d];
#pragma unroll
                for (int k = 0; k < PLOW_ATTNRES_RG; ++k) {
                    const unsigned r = g + (unsigned)k;
                    if (r > a.nb) break;
                    const bf16* row = r < a.nb
                                          ? a.block_residual + (size_t)r * a.hidden
                                          : prefix;
                    const float x = bf2f(row[d]);
                    ss[k] += x * x;
                    sw[k] += x * wd;
                }
            }
#pragma unroll
            for (int k = 0; k < PLOW_ATTNRES_RG; ++k) {
                const unsigned r = g + (unsigned)k;
                if (r > a.nb) break;
                const float x = wave_sum(ss[k]), y = wave_sum(sw[k]);
                if (lane == 0u) {
                    s->stats[domain][2u * r] = x;
                    s->stats[domain][2u * r + 1u] = y;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0u) d6_release(&s->phase[1][0]);

    if (domain == 0u) {
        if (threadIdx.x == 0u) d6_wait(&s->phase[1][0], iter * D6_DOMAINS, a);
        __syncthreads();
        if (active) {
            float ss = 0.0f, sw = 0.0f;
            if (lane <= a.nb) {
#pragma unroll
                for (unsigned d = 0; d < D6_DOMAINS; ++d) {
                    ss += s->stats[d][2u * lane];
                    sw += s->stats[d][2u * lane + 1u];
                }
            }
            const float score = lane <= a.nb
                                    ? sw * rsqrtf(ss / (float)a.hidden + a.eps)
                                    : -INFINITY;
            const float mx = half_wave_max(score);
            const float e = lane <= a.nb ? __expf(score - mx) : 0.0f;
            const float z = half_wave_sum(e);
            if (lane <= a.nb) s->probs[lane] = e / z;
        }
        __syncthreads();
        if (threadIdx.x == 0u) d6_release(&s->phase[2][0]);
    } else {
        if (threadIdx.x == 0u) d6_wait(&s->phase[2][0], iter, a);
        __syncthreads();
    }

    float local_ss = 0.0f;
    if (active) {
        for (unsigned i = old_thread; i < nv; i += 512u) {
            float acc[8] = {};
            for (unsigned r = 0; r <= a.nb; ++r) {
                const bf16* row = r < a.nb
                                      ? a.block_residual + (size_t)r * a.hidden
                                      : prefix;
                const bf16v8 v = ld_glob8(row + i * 8u);
                const float p = s->probs[r];
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[j] += p * bf2f(v[j]);
            }
            bf16v8 out;
#pragma unroll
            for (int j = 0; j < 8; ++j) out[j] = f2bf(acc[j]);
            if (a.use_gamma)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float x = bf2f(out[j]);
                    local_ss += x * x;
                }
            st_glob8(a.mixed + i * 8u, out);
        }
        for (unsigned d = nv * 8u + old_thread; d < a.hidden; d += 512u) {
            float acc = 0.0f;
            for (unsigned r = 0; r <= a.nb; ++r) {
                const bf16* row = r < a.nb
                                      ? a.block_residual + (size_t)r * a.hidden
                                      : prefix;
                acc += s->probs[r] * bf2f(row[d]);
            }
            const bf16 out = f2bf(acc);
            if (a.use_gamma) {
                const float x = bf2f(out);
                local_ss += x * x;
            }
            a.mixed[d] = out;
        }
        if (a.use_gamma) {
            local_ss = wave_sum(local_ss);
            if (lane == 0u) s->norm[domain] = local_ss;
        }
    }
    __syncthreads();
    if (!a.use_gamma) return;
    if (threadIdx.x == 0u) d6_release(&s->phase[3][0]);
    if (domain == 0u) {
        if (threadIdx.x == 0u) {
            d6_wait(&s->phase[3][0], iter * D6_DOMAINS, a);
            float ss = 0.0f;
#pragma unroll
            for (unsigned d = 0; d < D6_DOMAINS; ++d) ss += s->norm[d];
            s->inv = rsqrtf(ss / (float)a.hidden + a.eps);
            d6_release(&s->phase[4][0]);
        }
        __syncthreads();
    } else {
        if (threadIdx.x == 0u) d6_wait(&s->phase[4][0], iter, a);
        __syncthreads();
    }
    if (active) {
        const auto* gg = as_glob(a.gamma);
        auto* og = as_glob(a.mixed);
        for (unsigned i = old_thread; i < nv; i += 512u) {
            const bf16v8 o = ld_glob8(og + i * 8u);
            const bf16v8 g = ld_glob8(gg + i * 8u);
            bf16v8 n;
#pragma unroll
            for (int j = 0; j < 8; ++j) n[j] = f2bf(bf2f(o[j]) * s->inv * bf2f(g[j]));
            st_glob8(og + i * 8u, n);
        }
        for (unsigned d = nv * 8u + old_thread; d < a.hidden; d += 512u)
            a.mixed[d] = f2bf(bf2f(a.mixed[d]) * s->inv * bf2f(a.gamma[d]));
    }
    __syncthreads();
}

template <bool Fused>
__device__ __forceinline__ void d6_kernel(const D6Args& a, float* lds) {
    __shared__ unsigned role;
    const uint64_t start = __builtin_amdgcn_s_memrealtime();
    for (uint32_t iter = 1; iter <= a.iters; ++iter) {
        d_xreduce_oneshot(a.reduced, a.peer_scratch, a.peer_gate, a.nranks, a.rank,
                          a.hidden, 0u, iter * a.nranks, blockIdx.x == 0u,
                          a.deadline, a.status, blockIdx.x, D6_GRID);
        if constexpr (!Fused) {
            __syncthreads();
            if (threadIdx.x == 0u) d6_release(&a.sync->control[0][0]);
            if (blockIdx.x == 0u) {
                if (threadIdx.x == 0u) d6_wait(&a.sync->control[0][0], iter * D6_GRID, a);
                __syncthreads();
                const bf16* prefix = a.reduced;
                if (a.contract != 0u) {
                    const bf16* aa = a.contract == 1u ? a.reduced : a.residual_other;
                    const bf16* bb = a.contract == 1u ? a.residual_other : a.reduced;
                    d_materialize_residual(a.prefix, aa, bb, a.residual_pre, 1u, a.hidden, 0u, 1u);
                    prefix = a.prefix;
                }
                d_attn_res(a.mixed, prefix, a.block_residual, a.score_weight, 1u,
                           a.hidden, a.nb, a.nb_cap, a.eps, 0u, 1u, lds,
                           a.push ? a.push_src : nullptr, a.nb,
                           a.use_gamma ? a.gamma : nullptr);
                if (threadIdx.x == 0u) d6_release(&a.sync->control[1][0]);
            } else {
                if (threadIdx.x == 0u) d6_wait(&a.sync->control[1][0], iter, a);
                __syncthreads();
            }
        } else {
            __syncthreads();
            const unsigned domain =
                __builtin_amdgcn_s_getreg((20u) | (0u << 6) | ((32u - 1u) << 11)) & 0xfu;
            const unsigned nper = domain < 6u ? 2u : 1u;
            if (threadIdx.x == 0u)
                role = __hip_atomic_fetch_add(&a.sync->local_arrive[domain][0], 1u,
                                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                       == iter * nper - 1u;
            __syncthreads();
            if (role) {
                if (threadIdx.x == 0u) d6_release(&a.sync->phase[0][0]);
                if (threadIdx.x == 0u) d6_wait(&a.sync->phase[0][0], iter * D6_DOMAINS, a);
                __syncthreads();
                d6_attn_leader(a, domain, iter);
                if (threadIdx.x == 0u) d6_release(&a.sync->local_done[domain][0]);
            } else {
                if (threadIdx.x == 0u) d6_wait(&a.sync->local_done[domain][0], iter, a);
                __syncthreads();
            }
        }
        if (threadIdx.x == 0u) d6_release(&a.sync->successor[0]);
        if (threadIdx.x == 0u) d6_wait(&a.sync->successor[0], iter * D6_GRID, a);
        __syncthreads();
    }
    const uint64_t end = __builtin_amdgcn_s_memrealtime();
    if (blockIdx.x == 0u && threadIdx.x == 0u && a.cycles) *a.cycles = end - start;
}

extern "C" __global__ __launch_bounds__(512, 2)
__attribute__((amdgpu_waves_per_eu(2, 2))) void d6_control(const D6Args* a) {
    __shared__ __align__(16) volatile char envelope[147452];
    envelope[threadIdx.x] = 0;
    d6_kernel<false>(*a, (float*)(void*)envelope);
}
extern "C" __global__ __launch_bounds__(512, 2)
__attribute__((amdgpu_waves_per_eu(2, 2))) void d6_fused(const D6Args* a) {
    __shared__ __align__(16) volatile char envelope[147452];
    envelope[threadIdx.x] = 0;
    d6_kernel<true>(*a, (float*)(void*)envelope);
}

extern "C" __global__ void d6_fill_bf16(bf16* out, uint32_t n, uint32_t kind, uint32_t rank) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float v = 0.0f;
        if (kind == 0u) v = (float)(rank + 1u) * (0.25f + (float)(i & 15u) * 0.03125f);
        else if (kind == 1u) v = -0.75f + (float)(i % 29u) * 0.046875f;
        else if (kind == 2u) v = 0.5f + (float)(i % 17u) * 0.015625f;
        else v = -0.25f + (float)(i % 23u) * 0.0234375f;
        out[i] = f2bf(v);
    }
}
extern "C" __global__ void d6_fill_f32(float* out, uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        out[i] = -0.03125f + (float)(i % 31u) * 0.001953125f;
}
extern "C" __global__ __launch_bounds__(512, 2)
__attribute__((amdgpu_waves_per_eu(2, 2))) void d6_xcd_map(uint32_t* out) {
    __shared__ __align__(16) volatile char envelope[147460];
    envelope[threadIdx.x] = 0;
    if (threadIdx.x == 0u)
        out[blockIdx.x] =
            __builtin_amdgcn_s_getreg((20u) | (0u << 6) | ((32u - 1u) << 11)) & 0xfu;
}
extern "C" __device__ unsigned plow_d6_oracle_wave64 = 64;
extern "C" __device__ unsigned plow_d6_oracle_grid = D6_GRID;
extern "C" __device__ unsigned plow_d6_oracle_domains = D6_DOMAINS;
extern "C" __device__ unsigned plow_d6_oracle_args_size = sizeof(D6Args);
extern "C" __device__ unsigned plow_d6_oracle_sync_size = sizeof(D6Sync);
