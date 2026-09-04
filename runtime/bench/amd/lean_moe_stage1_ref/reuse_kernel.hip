#define PLOW_MOE_PREFILL 1
#define PLOW_MOE_PF_A4W4 1
#define PLOW_WG_WAVES 8
#define MPF4_BK 256
#include "amd_common.h"
#include "op_moe.h"

using u32x4 = unsigned __attribute__((ext_vector_type(4)));
using i32x8 = int __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ i32x8 reuse_fp4_operand(const unsigned char* p) {
    const u32x4 q = *reinterpret_cast<const u32x4*>(p);
    return i32x8{static_cast<int>(q[0]), static_cast<int>(q[1]),
                 static_cast<int>(q[2]), static_cast<int>(q[3]), 0, 0, 0, 0};
}

template <int OpA, int OpB>
static __device__ __forceinline__ f32x4 reuse_mfma(
    i32x8 a, i32x8 b, f32x4 c, int scale_a, int scale_b) {
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 4, 4, OpA, scale_a, OpB, scale_b);
}

extern "C" __device__ unsigned plow_moe1_a4_reuse_abi_1 = 1;
extern "C" __device__ unsigned plow_moe1_a4_reuse_wave64_1 = 1;
extern "C" __device__ unsigned plow_moe1_a4_reuse_four_wave_1 = 1;
extern "C" __device__ unsigned plow_moe1_a4_reuse_a_only_lds_8192 = 1;
extern "C" __device__ unsigned plow_moe1_a4_reuse_register_b_1 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_abi_1 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_bm64_bn256_bk256_1 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_wave64_1 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_no_spill_1 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_dynamic_lds_119808 = 1;
extern "C" __device__ unsigned plow_moe1_mxfp4_stage1_vgpr_le_192 = 1;

extern "C" __global__ __launch_bounds__(256, 2)
void plow_moe1_quant_sort_a4_gfx950(
    unsigned char* __restrict__ out,
    unsigned char* __restrict__ out_scale,
    const bf16* __restrict__ activation,
    const unsigned* __restrict__ row_token,
    const int* __restrict__ meta,
    unsigned row_capacity,
    unsigned experts,
    unsigned hidden) {
    const unsigned rows = min(
        static_cast<unsigned>(meta[2u * experts + experts]), row_capacity / 64u) * 64u;
    const size_t groups = hidden / 32u;
    const size_t count = static_cast<size_t>(rows) * groups;
    for (size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < count; item += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const unsigned row = static_cast<unsigned>(item / groups);
        const unsigned group = static_cast<unsigned>(item - static_cast<size_t>(row) * groups);
        const unsigned src_row = row_token[row];
        unsigned char* dst = out + (static_cast<size_t>(row) * groups + group) * 16u;
        unsigned char* dst_scale = out_scale + item;
        if (src_row == ~0u) {
            *reinterpret_cast<u32x4*>(dst) = u32x4{0, 0, 0, 0};
            *dst_scale = PLOW_E8M0_ONE;
        } else {
            const mpf4_a32 v = mpf4_quant_load(
                activation + static_cast<size_t>(src_row) * hidden + group * 32u);
            mpf4_quant_commit(v, dst, dst_scale);
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 2)
__attribute__((amdgpu_waves_per_eu(2, 2)))
void plow_moe1_mxfp4_bk256_gfx950(
    unsigned char* out, const bf16* activation, const unsigned long long* weight_table,
    const unsigned long long* weight_scale_table, const int* meta,
    const unsigned* row_token, const unsigned* row_partidx, unsigned char* out_scale,
    unsigned inter_dim, unsigned model_dim, unsigned experts, unsigned act,
    float beta, float linear_beta, unsigned reserved, unsigned reserved2) {
    (void)reserved; (void)reserved2;
    extern __shared__ unsigned char lds[];
    d_moe_group_pf_a4w4<true>(out, activation, nullptr, weight_table, weight_scale_table,
        meta, row_token, row_partidx, nullptr, out_scale, inter_dim, model_dim, experts,
        act, blockIdx.x, gridDim.x, beta, linear_beta, lds);
}

struct ReuseBStage {
    i32x8 gate[2];
    i32x8 up[2];
    int gate_scale;
    int up_scale;
};

static __device__ __forceinline__ ReuseBStage reuse_load_b(
    const unsigned char* gate, const unsigned char* up,
    const unsigned char* gate_scale, const unsigned char* up_scale,
    unsigned kt, unsigned n0, unsigned wave, unsigned lane_k, unsigned lane_n,
    unsigned hidden, unsigned inter) {
    ReuseBStage stage;
    const unsigned kbytes = hidden / 2u;
    const unsigned groups = hidden / 32u;
#pragma unroll
    for (unsigned ni = 0; ni < 2; ++ni) {
        const unsigned n = n0 + wave * 32u + ni * 16u + lane_n;
        const size_t offset = static_cast<size_t>(n) * kbytes
            + static_cast<size_t>(kt) * 64u + lane_k * 16u;
        const i32x8 zero{0, 0, 0, 0, 0, 0, 0, 0};
        stage.gate[ni] = n < inter ? reuse_fp4_operand(gate + offset) : zero;
        stage.up[ni] = n < inter ? reuse_fp4_operand(up + offset) : zero;
    }
    const unsigned n = n0 + wave * 32u + lane_n;
    const unsigned group = kt * 4u + lane_k;
    if (n < inter) {
        stage.gate_scale = static_cast<int>(gate_scale[static_cast<size_t>(n) * groups + group])
            | (static_cast<int>(gate_scale[static_cast<size_t>(n + 16u) * groups + group]) << 8);
        stage.up_scale = static_cast<int>(up_scale[static_cast<size_t>(n) * groups + group])
            | (static_cast<int>(up_scale[static_cast<size_t>(n + 16u) * groups + group]) << 8);
    } else {
        stage.gate_scale = 0;
        stage.up_scale = 0;
    }
    return stage;
}

static __device__ __forceinline__ unsigned reuse_expert_for_tile(
    const int* tile_prefix, unsigned tile, unsigned experts) {
    unsigned lo = 0, hi = experts;
    while (lo + 1u < hi) {
        const unsigned mid = lo + (hi - lo) / 2u;
        if (static_cast<unsigned>(tile_prefix[mid]) <= tile) lo = mid;
        else hi = mid;
    }
    return lo;
}

extern "C" __global__ __launch_bounds__(256, 2)
void plow_moe1_a4_reuse_16x16x128_gfx950(
    unsigned char* __restrict__ out,
    const unsigned char* __restrict__ activation,
    const unsigned long long* __restrict__ weight_table,
    const unsigned char* __restrict__ activation_scale,
    const unsigned long long* __restrict__ weight_scale_table,
    const int* __restrict__ meta,
    const unsigned* __restrict__ row_partidx,
    unsigned char* __restrict__ out_scale,
    unsigned inter,
    unsigned hidden,
    unsigned experts,
    unsigned act,
    float beta,
    float linear_beta) {
    constexpr unsigned TM = 64, TN = 128, TK = 128, KBYTES = TK / 2;
    extern __shared__ __align__(16) unsigned char a_lds[];
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid / 64u;
    const unsigned lane = tid % 64u;
    const unsigned lane_k = lane / 16u;
    const unsigned lane_n = lane % 16u;
    const unsigned n_tiles = (inter + TN - 1u) / TN;
    const unsigned mt = blockIdx.x / n_tiles;
    const unsigned nt = blockIdx.x - mt * n_tiles;
    const int* tile_prefix = meta + 2 * experts;
    const unsigned total_tiles = static_cast<unsigned>(tile_prefix[experts]);
    if (mt >= total_tiles || hidden % TK != 0 || inter == 0) return;
    const unsigned expert = reuse_expert_for_tile(tile_prefix, mt, experts);
    const unsigned rowbase = static_cast<unsigned>(meta[expert])
        + (mt - static_cast<unsigned>(tile_prefix[expert])) * TM;
    const unsigned n0 = nt * TN;
    const auto gate = reinterpret_cast<const unsigned char*>(
        static_cast<size_t>(weight_table[static_cast<size_t>(expert) * 3u]));
    const auto up = reinterpret_cast<const unsigned char*>(
        static_cast<size_t>(weight_table[static_cast<size_t>(expert) * 3u + 1u]));
    const auto gate_scale = reinterpret_cast<const unsigned char*>(
        static_cast<size_t>(weight_scale_table[static_cast<size_t>(expert) * 3u]));
    const auto up_scale = reinterpret_cast<const unsigned char*>(
        static_cast<size_t>(weight_scale_table[static_cast<size_t>(expert) * 3u + 1u]));
    if (!gate || !up || !gate_scale || !up_scale) return;
    const unsigned a_stride = hidden / 2u;
    const unsigned as_stride = hidden / 32u;

    f32x4 acc[4][4];
#pragma unroll
    for (unsigned mi = 0; mi < 4; ++mi)
#pragma unroll
        for (unsigned bi = 0; bi < 4; ++bi)
            acc[mi][bi] = f32x4{0, 0, 0, 0};

    auto load_a = [&](unsigned kt) {
        const unsigned row = tid / 4u;
        const unsigned chunk = tid % 4u;
        return *reinterpret_cast<const u32x4*>(
            activation + static_cast<size_t>(rowbase + row) * a_stride
            + kt * KBYTES + chunk * 16u);
    };
    auto store_a = [&](u32x4 value, unsigned slot) {
        const unsigned row = tid / 4u;
        const unsigned chunk = tid % 4u;
        const unsigned swizzled = chunk ^ (row & 3u);
        *reinterpret_cast<u32x4*>(
            a_lds + slot * TM * KBYTES + row * KBYTES + swizzled * 16u) = value;
    };

    const unsigned ntk = hidden / TK;
    ReuseBStage b_cur = reuse_load_b(gate, up, gate_scale, up_scale,
                                     0, n0, wave, lane_k, lane_n, hidden, inter);
    store_a(load_a(0), 0);
    __syncthreads();
    for (unsigned kt = 0; kt < ntk; ++kt) {
        int a_scale_word = 0;
        i32x8 a_frag[4];
#pragma unroll
        for (unsigned mi = 0; mi < 4; ++mi) {
            const unsigned row = lane_n + mi * 16u;
            const unsigned swizzled = lane_k ^ (row & 3u);
            a_frag[mi] = reuse_fp4_operand(
                a_lds + (kt & 1u) * TM * KBYTES + row * KBYTES + swizzled * 16u);
            const unsigned sb = activation_scale[
                static_cast<size_t>(rowbase + row) * as_stride + kt * 4u + lane_k];
            a_scale_word |= static_cast<int>(sb) << (mi * 8u);
        }
        ReuseBStage b_next;
        u32x4 a_next{0, 0, 0, 0};
        if (kt + 1u < ntk) {
            b_next = reuse_load_b(gate, up, gate_scale, up_scale,
                                  kt + 1u, n0, wave, lane_k, lane_n, hidden, inter);
            a_next = load_a(kt + 1u);
        }
        __builtin_amdgcn_s_setprio(1);
#define REUSE_MFMA_ROW(MI)                                                                    \
        acc[MI][0] = reuse_mfma<MI, 0>(a_frag[MI], b_cur.gate[0], acc[MI][0],                 \
                                        a_scale_word, b_cur.gate_scale);                       \
        acc[MI][1] = reuse_mfma<MI, 1>(a_frag[MI], b_cur.gate[1], acc[MI][1],                 \
                                        a_scale_word, b_cur.gate_scale);                       \
        acc[MI][2] = reuse_mfma<MI, 0>(a_frag[MI], b_cur.up[0], acc[MI][2],                   \
                                        a_scale_word, b_cur.up_scale);                         \
        acc[MI][3] = reuse_mfma<MI, 1>(a_frag[MI], b_cur.up[1], acc[MI][3],                   \
                                        a_scale_word, b_cur.up_scale)
        REUSE_MFMA_ROW(0);
        REUSE_MFMA_ROW(1);
        REUSE_MFMA_ROW(2);
        REUSE_MFMA_ROW(3);
#undef REUSE_MFMA_ROW
        __builtin_amdgcn_s_setprio(0);
        if (kt + 1u < ntk) {
            store_a(a_next, (kt + 1u) & 1u);
            b_cur = b_next;
        }
        __syncthreads();
    }

    float* bridge = reinterpret_cast<float*>(a_lds);
#pragma unroll
    for (unsigned mi = 0; mi < 4; ++mi)
#pragma unroll
        for (unsigned ni = 0; ni < 2; ++ni)
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned row = mi * 16u + lane_k * 4u + i;
                const unsigned col = wave * 32u + ni * 16u + lane_n;
                if (n0 + col < inter)
                    bridge[row * TN + col] = mpf4_glu(
                        acc[mi][ni][i], acc[mi][2u + ni][i], act, beta, linear_beta);
            }
    __syncthreads();
    const unsigned row = tid / 4u;
    const unsigned block = tid % 4u;
    if (row_partidx[rowbase + row] != ~0u && n0 + block * 32u < inter) {
        const float* src = bridge + row * TN + block * 32u;
        float amax = 0.0f;
#pragma unroll
        for (unsigned i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(src[i]));
        const unsigned char sb = e8m0_for_amax(amax);
        const float inv = e8m0_inv_f32(sb);
        u32x4 q;
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            unsigned word = 0;
#pragma unroll
            for (unsigned j = 0; j < 8; ++j)
                word |= quant_fp4(src[i * 8u + j] * inv) << (j * 4u);
            q[i] = word;
        }
        *reinterpret_cast<u32x4*>(out + static_cast<size_t>(rowbase + row) * (inter / 2u)
                                 + (n0 + block * 32u) / 2u) = q;
        out_scale[static_cast<size_t>(rowbase + row) * (inter / 32u)
                  + (n0 + block * 32u) / 32u] = sb;
    }
}
