#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

using u32x4 = unsigned __attribute__((ext_vector_type(4)));
using i32x8 = int __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ i32x8 fp4_operand(const unsigned char* p) {
    const u32x4 q = *reinterpret_cast<const u32x4*>(p);
    return i32x8{static_cast<int>(q[0]), static_cast<int>(q[1]),
                 static_cast<int>(q[2]), static_cast<int>(q[3]), 0, 0, 0, 0};
}

template <int OpA, int OpB>
static __device__ __forceinline__ f32x4 mfma_fp4(
    i32x8 a, i32x8 b, f32x4 c, int scale_a, int scale_b) {
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 4, 4, OpA, scale_a, OpB, scale_b);
}

struct BStage {
    i32x8 frag[4];
    int scale[2];
};

static __device__ __forceinline__ BStage load_b_stage(
    const unsigned char* weight, const unsigned* weight_scale,
    unsigned kt, unsigned n0, unsigned wave, unsigned lane_k, unsigned lane_n,
    int expert, int model_dim, unsigned weight_stride_n0,
    size_t expert_weight_stride, unsigned scale_stride_n0) {
    BStage stage;
#pragma unroll
    for (unsigned ni = 0; ni < 4; ++ni) {
        const unsigned global_n = n0 + wave * 64u + ni * 16u + lane_n;
        const unsigned n_block = global_n / 16u;
        const size_t offset = static_cast<size_t>(expert) * expert_weight_stride
            + static_cast<size_t>(n_block) * weight_stride_n0
            + static_cast<size_t>(kt) * 1024u + lane_k * 256u + lane_n * 16u;
        stage.frag[ni] = global_n < static_cast<unsigned>(model_dim)
            ? fp4_operand(weight + offset) : i32x8{0, 0, 0, 0, 0, 0, 0, 0};
    }
#pragma unroll
    for (unsigned np = 0; np < 2; ++np) {
        const unsigned n_pair = n0 + wave * 64u + np * 32u;
        const unsigned mni = (static_cast<unsigned>(expert) * model_dim + n_pair) / 32u;
        const unsigned index = mni * scale_stride_n0 + (kt / 2u) * 64u
            + lane_k * 16u + lane_n;
        stage.scale[np] = static_cast<int>(weight_scale[index] >> ((kt & 1u) * 16u));
    }
    return stage;
}

extern "C" __global__ __launch_bounds__(256, 2)
void plow_moe2_mxfp4_16x16x128_gfx950(
    __hip_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    const unsigned* __restrict__ activation_scale,
    const unsigned* __restrict__ weight_scale,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ num_valid_ids,
    const float* __restrict__ bias,
    int tokens, int model_dim, int inter_dim, int expert_blocks) {
    (void)bias;
    (void)expert_blocks;
    constexpr unsigned TM = 32;
    constexpr unsigned TN = 256;
    constexpr unsigned TK = 128;
    constexpr unsigned WG = 256;
    constexpr unsigned WAVE = 64;
    constexpr unsigned KBYTES = TK / 2;

    extern __shared__ __align__(16) unsigned char shared[];
    unsigned char* a_lds = shared;
    __hip_bfloat16* c_lds = reinterpret_cast<__hip_bfloat16*>(shared);
    int* row_ids_lds = reinterpret_cast<int*>(shared + 16384);
    float* row_gates_lds = reinterpret_cast<float*>(shared + 16384 + TM * sizeof(int));

    const unsigned tid = threadIdx.x;
    const unsigned wave = tid / WAVE;
    const unsigned lane = tid % WAVE;
    const unsigned lane_k = lane / 16;
    const unsigned lane_n = lane % 16;
    const unsigned block_m = blockIdx.y;
    const unsigned n0 = blockIdx.x * TN;
    const int valid_rows = *num_valid_ids;
    const int expert = expert_ids[block_m];
    const unsigned kbytes = static_cast<unsigned>(inter_dim) / 2;
    const unsigned scale_k_padded = (static_cast<unsigned>(inter_dim) + 255u) & ~255u;
    const unsigned scale_k_groups = scale_k_padded / 256u;
    const unsigned scale_stride_n0 = scale_k_groups * 64u;
    const unsigned weight_stride_n0 = (kbytes / 64u) * 1024u;
    const size_t expert_weight_stride =
        static_cast<size_t>(model_dim / 16) * weight_stride_n0;
    if (inter_dim != 384)
        return;
    if (tid < TM) {
        const unsigned sorted_row = block_m * TM + tid;
        row_ids_lds[tid] = sorted_row < static_cast<unsigned>(valid_rows)
            ? sorted_token_ids[sorted_row] : -1;
        row_gates_lds[tid] = sorted_row < static_cast<unsigned>(valid_rows)
            ? sorted_weights[sorted_row] : 0.0f;
    }
    __syncthreads();

    f32x4 acc[2][4];
#pragma unroll
    for (unsigned mi = 0; mi < 2; ++mi)
#pragma unroll
        for (unsigned ni = 0; ni < 4; ++ni)
            acc[mi][ni] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};

    auto load_a_stage = [&](unsigned kt) {
        u32x4 value{0, 0, 0, 0};
        if (tid < TM * 4) {
            const unsigned row = tid / 4;
            const unsigned chunk = tid % 4;
            const int fused = row_ids_lds[row];
            const unsigned token = static_cast<unsigned>(fused) & 0x00ffffffu;
            const unsigned slot = static_cast<unsigned>(fused) >> 24;
            if (fused >= 0 && token < static_cast<unsigned>(tokens)) {
                const size_t src = (static_cast<size_t>(token) * 16u + slot) * kbytes
                    + kt * KBYTES + chunk * 16u;
                value = *reinterpret_cast<const u32x4*>(activation + src);
            }
        }
        return value;
    };
    auto store_a_stage = [&](u32x4 value, unsigned slot) {
        if (tid < TM * 4) {
            const unsigned row = tid / 4;
            const unsigned chunk = tid % 4;
            const unsigned swizzled = chunk ^ (row & 3u);
            *reinterpret_cast<u32x4*>(
                a_lds + slot * TM * KBYTES + row * KBYTES + swizzled * 16u) = value;
        }
    };

    BStage b_cur = load_b_stage(
        weight, weight_scale, 0, n0, wave, lane_k, lane_n, expert, model_dim,
        weight_stride_n0, expert_weight_stride, scale_stride_n0);
    store_a_stage(load_a_stage(0), 0);
    __syncthreads();

#pragma unroll
    for (unsigned kt = 0; kt < 3; ++kt) {
        i32x8 a_frag[2];
        const unsigned a_scale_index = block_m * scale_stride_n0 + (kt / 2u) * 64u
            + lane_k * 16u + lane_n;
        const int a_scale_pair =
            static_cast<int>(activation_scale[a_scale_index] >> ((kt & 1u) * 16u));
#pragma unroll
        for (unsigned mi = 0; mi < 2; ++mi) {
            const unsigned row = lane_n + mi * 16u;
            const unsigned swizzled = lane_k ^ (row & 3u);
            a_frag[mi] = fp4_operand(
                a_lds + (kt & 1u) * TM * KBYTES + row * KBYTES + swizzled * 16u);
        }

        BStage b_next;
        u32x4 a_next{0, 0, 0, 0};
        if (kt < 2) {
            b_next = load_b_stage(
                weight, weight_scale, kt + 1, n0, wave, lane_k, lane_n, expert, model_dim,
                weight_stride_n0, expert_weight_stride, scale_stride_n0);
            a_next = load_a_stage(kt + 1);
        }
        __builtin_amdgcn_s_setprio(1);
        acc[0][0] = mfma_fp4<0, 0>(a_frag[0], b_cur.frag[0], acc[0][0], a_scale_pair, b_cur.scale[0]);
        acc[0][1] = mfma_fp4<0, 1>(a_frag[0], b_cur.frag[1], acc[0][1], a_scale_pair, b_cur.scale[0]);
        acc[0][2] = mfma_fp4<0, 0>(a_frag[0], b_cur.frag[2], acc[0][2], a_scale_pair, b_cur.scale[1]);
        acc[0][3] = mfma_fp4<0, 1>(a_frag[0], b_cur.frag[3], acc[0][3], a_scale_pair, b_cur.scale[1]);
        acc[1][0] = mfma_fp4<1, 0>(a_frag[1], b_cur.frag[0], acc[1][0], a_scale_pair, b_cur.scale[0]);
        acc[1][1] = mfma_fp4<1, 1>(a_frag[1], b_cur.frag[1], acc[1][1], a_scale_pair, b_cur.scale[0]);
        acc[1][2] = mfma_fp4<1, 0>(a_frag[1], b_cur.frag[2], acc[1][2], a_scale_pair, b_cur.scale[1]);
        acc[1][3] = mfma_fp4<1, 1>(a_frag[1], b_cur.frag[3], acc[1][3], a_scale_pair, b_cur.scale[1]);
        __builtin_amdgcn_s_setprio(0);
        if (kt < 2) {
            store_a_stage(a_next, (kt + 1) & 1u);
            b_cur = b_next;
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned mi = 0; mi < 2; ++mi)
#pragma unroll
        for (unsigned ni = 0; ni < 4; ++ni)
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned row = mi * 16u + lane_k * 4u + i;
                const unsigned col = wave * 64u + ni * 16u + lane_n;
                const float gate = row_gates_lds[row];
                c_lds[row * TN + col] = __float2bfloat16(acc[mi][ni][i] * gate);
            }
    __syncthreads();

    const unsigned pair_count = TM * TN / 2u;
    for (unsigned pair = tid; pair < pair_count; pair += WG) {
        const unsigned row = pair / (TN / 2u);
        const unsigned col = (pair % (TN / 2u)) * 2u;
        const int fused = row_ids_lds[row];
        const unsigned token = static_cast<unsigned>(fused) & 0x00ffffffu;
        if (fused >= 0 && token < static_cast<unsigned>(tokens) &&
            n0 + col + 1u < static_cast<unsigned>(model_dim)) {
            const __hip_bfloat162 value = *reinterpret_cast<const __hip_bfloat162*>(
                c_lds + row * TN + col);
            unsafeAtomicAdd(reinterpret_cast<__hip_bfloat162*>(
                out + static_cast<size_t>(token) * model_dim + n0 + col), value);
        }
    }
}
