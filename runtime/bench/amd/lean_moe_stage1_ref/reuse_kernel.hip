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
    unsigned hidden
#if defined(PLOW_MOE1_BODY) && PLOW_MOE1_BODY == 2
    , unsigned tokens
#endif
    ) {
#if defined(PLOW_MOE1_BODY) && PLOW_MOE1_BODY == 2
    /* TOKEN-GATHER form: quantize each of the T tokens ONCE into A4 row t; the GEMM reads row
     * row_token[r]. The sorted form re-reads and re-quantizes every token top-k times (~9x the
     * bf16 bytes, from HBM). A wave takes a pair of rows with every 64 B block of both in flight
     * before any is quantized. Same mpf4_quant_load/commit per block, so each row's bytes are
     * the sorted form's bytes. */
    (void)row_token; (void)meta; (void)row_capacity; (void)experts;
    const unsigned rows = tokens;
    constexpr unsigned GPL = 4;
    const unsigned groups = hidden / 32u;
    if (groups > GPL * 64u) __builtin_trap();
    const unsigned lane = threadIdx.x % 64u;
    const unsigned nw = gridDim.x * (blockDim.x / 64u);
    const unsigned w = __builtin_amdgcn_readfirstlane(blockIdx.x * (blockDim.x / 64u) + threadIdx.x / 64u);
    for (unsigned row0 = w * 2u; row0 < rows; row0 += nw * 2u) {
        mpf4_a32 v[2][GPL];
#pragma unroll
        for (unsigned r = 0; r < 2; ++r) {
            const bf16* src = activation + static_cast<size_t>(min(row0 + r, rows - 1u)) * hidden;
#pragma unroll
            for (unsigned j = 0; j < GPL; ++j) {
                const unsigned g = lane + j * 64u;
                if (j * 64u < groups) v[r][j] = mpf4_quant_load(src + min(g, groups - 1u) * 32u);
            }
        }
#pragma unroll
        for (unsigned r = 0; r < 2; ++r) {
            const size_t row = row0 + r;
#pragma unroll
            for (unsigned j = 0; j < GPL; ++j) {
                const unsigned g = lane + j * 64u;
                if (j * 64u >= groups || g >= groups || row >= rows) continue;
                mpf4_quant_commit(v[r][j], out + (row * groups + g) * 16u, out_scale + row * groups + g);
            }
        }
    }
#else
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
#endif
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

#ifndef PLOW_MOE1_BODY
#define PLOW_MOE1_BODY 0
#endif
#if PLOW_MOE1_BODY == 1
#ifndef PLOW_MOE1_XCD
#define PLOW_MOE1_XCD 1
#endif
extern "C" __device__ unsigned plow_moe1_a4_reuse_body_k256_1 = 1;

/* Two-k-step (K256) iteration body. Same MFMA sequence per accumulator as the K128 body below
 * (k-steps issued in the same order with the same operands), so the output is bit-identical;
 * only data movement changes: full 128-byte B row lines per iteration, A tile rows of 128 B,
 * E8M0 scales staged through LDS in packed lane order, one barrier per two k-steps.
 * LDS: A 2x8192, B scales 2x2048, A scales 2x512 = 21,504 B < the 32,768 B bridge alias. */
struct BodyBStage {
    i32x8 gate[2][2];
    i32x8 up[2][2];
};

static __device__ __forceinline__ BodyBStage body_load_b(
    const unsigned char* gate, const unsigned char* up, unsigned kt0, bool has_k1,
    unsigned n0, unsigned wave, unsigned lane_k, unsigned lane_n, unsigned hidden,
    unsigned inter) {
    BodyBStage stage;
    const unsigned kbytes = hidden / 2u;
    const i32x8 zero{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (unsigned ni = 0; ni < 2; ++ni) {
        const unsigned n = n0 + wave * 32u + ni * 16u + lane_n;
        const size_t offset = static_cast<size_t>(n) * kbytes
            + static_cast<size_t>(kt0) * 64u + lane_k * 16u;
        stage.gate[0][ni] = n < inter ? reuse_fp4_operand(gate + offset) : zero;
        stage.up[0][ni] = n < inter ? reuse_fp4_operand(up + offset) : zero;
        stage.gate[1][ni] = has_k1 && n < inter ? reuse_fp4_operand(gate + offset + 64u) : zero;
        stage.up[1][ni] = has_k1 && n < inter ? reuse_fp4_operand(up + offset + 64u) : zero;
    }
    return stage;
}

#ifndef PLOW_MOE1_WAVES
#define PLOW_MOE1_WAVES 2
#endif
extern "C" __global__ __launch_bounds__(256, PLOW_MOE1_WAVES)
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
    constexpr unsigned TM = 64, TN = 128, TK = 128, ROWB = 128, A_SLOT = TM * ROWB;
    constexpr unsigned BS_OFF = 2 * A_SLOT, BS_SLOT = 2048, AS_OFF = BS_OFF + 2 * BS_SLOT,
                       AS_SLOT = 512;
    extern __shared__ __align__(16) unsigned char a_lds[];
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid / 64u;
    const unsigned lane = tid % 64u;
    const unsigned lane_k = lane / 16u;
    const unsigned lane_n = lane % 16u;
    const unsigned n_tiles = (inter + TN - 1u) / TN;
    unsigned bid = blockIdx.x;
#if PLOW_MOE1_XCD
    /* Hardware places consecutive workgroups on consecutive XCDs; fold the grid so that
     * consecutive tiles (same expert, shared B rows) land on one XCD's L2. Bijective. */
    {
        const unsigned per = gridDim.x / 8u;
        if (bid < per * 8u) bid = (bid % 8u) * per + bid / 8u;
    }
#endif
    const unsigned mt = bid / n_tiles;
    const unsigned nt = bid - mt * n_tiles;
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
    const unsigned groups = hidden / 32u;
    const unsigned ntk = hidden / TK;
    const unsigned niter = (ntk + 1u) / 2u;

    f32x4 acc[4][4];
#pragma unroll
    for (unsigned mi = 0; mi < 4; ++mi)
#pragma unroll
        for (unsigned bi = 0; bi < 4; ++bi)
            acc[mi][bi] = f32x4{0, 0, 0, 0};

    /* A tile: 64 rows x 128 B per iteration; thread -> row tid/4, 32 B at chunk pair (tid%4)*2. */
    const unsigned a_row = tid / 4u;
    const unsigned a_chunk = (tid % 4u) * 2u;
    auto load_a = [&](unsigned it, bool has_k1, u32x4* v) {
        const unsigned char* src = activation + static_cast<size_t>(rowbase + a_row) * a_stride
            + it * ROWB + a_chunk * 16u;
        const bool valid = has_k1 || a_chunk < 4u;
        v[0] = valid ? *reinterpret_cast<const u32x4*>(src) : u32x4{0, 0, 0, 0};
        v[1] = valid ? *reinterpret_cast<const u32x4*>(src + 16u) : u32x4{0, 0, 0, 0};
    };
    auto store_a = [&](const u32x4* v, unsigned slot) {
        unsigned char* dst = a_lds + slot * A_SLOT + a_row * ROWB;
        *reinterpret_cast<u32x4*>(dst + ((a_chunk ^ (a_row & 7u)) * 16u)) = v[0];
        *reinterpret_cast<u32x4*>(dst + (((a_chunk + 1u) ^ (a_row & 7u)) * 16u)) = v[1];
    };
    /* B scales: thread -> (branch tid>>7, column tid&127); 8 groups per iteration as 2 dwords.
     * Stored as [branch][group][wave][lane_n][ni] so a lane reads its packed pair with one u16. */
    const unsigned bs_branch = tid >> 7;
    const unsigned bs_col = tid & 127u;
    const unsigned char* bs_src = (bs_branch ? up_scale : gate_scale)
        + static_cast<size_t>(n0 + bs_col) * groups;
    auto load_bs = [&](unsigned it, bool has_k1, unsigned* w) {
        if (n0 + bs_col < inter) {
            w[0] = *reinterpret_cast<const unsigned*>(bs_src + it * 8u);
            w[1] = has_k1 ? *reinterpret_cast<const unsigned*>(bs_src + it * 8u + 4u) : 0u;
        } else {
            w[0] = 0u;
            w[1] = 0u;
        }
    };
    auto store_bs = [&](const unsigned* w, unsigned slot) {
        unsigned char* dst = a_lds + BS_OFF + slot * BS_SLOT + bs_branch * 1024u
            + (bs_col / 32u) * 32u + (bs_col & 15u) * 2u + ((bs_col / 16u) & 1u);
#pragma unroll
        for (unsigned g = 0; g < 8; ++g)
            dst[g * 128u] = static_cast<unsigned char>(w[g / 4u] >> ((g % 4u) * 8u));
    };
    /* A scales: threads < 64 -> row; stored as [group][lane_n][mi] so a lane reads one dword. */
    auto load_as = [&](unsigned it, bool has_k1, unsigned* w) {
        if (tid < TM) {
            const unsigned char* src = activation_scale
                + static_cast<size_t>(rowbase + tid) * groups + it * 8u;
            w[0] = *reinterpret_cast<const unsigned*>(src);
            w[1] = has_k1 ? *reinterpret_cast<const unsigned*>(src + 4u) : 0u;
        }
    };
    auto store_as = [&](const unsigned* w, unsigned slot) {
        if (tid < TM) {
            unsigned char* dst = a_lds + AS_OFF + slot * AS_SLOT + (tid & 15u) * 4u + tid / 16u;
#pragma unroll
            for (unsigned g = 0; g < 8; ++g)
                dst[g * 64u] = static_cast<unsigned char>(w[g / 4u] >> ((g % 4u) * 8u));
        }
    };

    auto mfma_step = [&](const i32x8* bg, const i32x8* bu, unsigned slot, unsigned s) {
        const unsigned char* a_slot = a_lds + slot * A_SLOT;
        const unsigned chunk = s * 4u + lane_k;
        const int a_scale_word = *reinterpret_cast<const int*>(
            a_lds + AS_OFF + slot * AS_SLOT + (chunk * 16u + lane_n) * 4u);
        const unsigned char* bs = a_lds + BS_OFF + slot * BS_SLOT + chunk * 128u
            + wave * 32u + lane_n * 2u;
        const int gate_scale_word = *reinterpret_cast<const unsigned short*>(bs);
        const int up_scale_word = *reinterpret_cast<const unsigned short*>(bs + 1024u);
        i32x8 a_frag[4];
#pragma unroll
        for (unsigned mi = 0; mi < 4; ++mi) {
            const unsigned row = lane_n + mi * 16u;
            a_frag[mi] = reuse_fp4_operand(
                a_slot + row * ROWB + ((chunk ^ (row & 7u)) * 16u));
        }
        __builtin_amdgcn_s_setprio(1);
#define BODY_MFMA_ROW(MI)                                                                     \
        acc[MI][0] = reuse_mfma<MI, 0>(a_frag[MI], bg[0], acc[MI][0],                         \
                                        a_scale_word, gate_scale_word);                        \
        acc[MI][1] = reuse_mfma<MI, 1>(a_frag[MI], bg[1], acc[MI][1],                         \
                                        a_scale_word, gate_scale_word);                        \
        acc[MI][2] = reuse_mfma<MI, 0>(a_frag[MI], bu[0], acc[MI][2],                         \
                                        a_scale_word, up_scale_word);                          \
        acc[MI][3] = reuse_mfma<MI, 1>(a_frag[MI], bu[1], acc[MI][3],                          \
                                        a_scale_word, up_scale_word)
        BODY_MFMA_ROW(0);
        BODY_MFMA_ROW(1);
        BODY_MFMA_ROW(2);
        BODY_MFMA_ROW(3);
#undef BODY_MFMA_ROW
        __builtin_amdgcn_s_setprio(0);
    };

    {
        u32x4 a0[2];
        unsigned bs0[2], as0[2] = {0u, 0u};
        const bool has_k1 = ntk > 1u;
        load_a(0, has_k1, a0);
        load_bs(0, has_k1, bs0);
        load_as(0, has_k1, as0);
        store_a(a0, 0);
        store_bs(bs0, 0);
        store_as(as0, 0);
    }
    BodyBStage b_cur = body_load_b(gate, up, 0, ntk > 1u, n0, wave, lane_k, lane_n, hidden, inter);
    __syncthreads();
    for (unsigned it = 0; it < niter; ++it) {
        const unsigned slot = it & 1u;
        const bool has_k1 = 2u * it + 1u < ntk;
        const bool has_next = it + 1u < niter;
        BodyBStage b_next;
        u32x4 a_next[2];
        unsigned bs_next[2], as_next[2] = {0u, 0u};
        if (has_next) {
            const bool next_k1 = 2u * it + 3u < ntk;
            b_next = body_load_b(gate, up, 2u * it + 2u, next_k1, n0, wave, lane_k, lane_n,
                                 hidden, inter);
            load_a(it + 1u, next_k1, a_next);
            load_bs(it + 1u, next_k1, bs_next);
            load_as(it + 1u, next_k1, as_next);
        }
        mfma_step(b_cur.gate[0], b_cur.up[0], slot, 0);
        if (has_k1) mfma_step(b_cur.gate[1], b_cur.up[1], slot, 1);
        if (has_next) {
            store_a(a_next, slot ^ 1u);
            store_bs(bs_next, slot ^ 1u);
            store_as(as_next, slot ^ 1u);
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
#elif PLOW_MOE1_BODY == 2
extern "C" __device__ unsigned plow_moe1_a4_reuse_pipe_k256_1 = 1;
/* ABI marker: quant takes `tokens` and writes A4 by TOKEN; this kernel takes `row_token` last. */
extern "C" __device__ unsigned plow_moe1_a4_token_gather_1 = 1;

/* PIPELINED body (PLOW_MOE_STAGE1_PIPE). Same MFMA chain per accumulator as the K128 body below
 * (16x16x128 k-steps in order from zero, same operands and scale bytes), so the output is
 * bit-identical; only data movement changes. A rows are gathered by token from the quant
 * kernel's per-token A4 (see plow_moe1_a4_token_gather_1). The K128 body waits on every load it issues
 * before the MFMAs that do not need it: B and scales come through FLAT loads (the weight
 * pointers arrive as integers), the scale bytes are combined inside `n < inter` branches, and
 * A's scale bytes are read from global in the step that consumes them, so each k-step exposes
 * about two full memory latencies. Here every load is a branchless global load issued a whole
 * K256 stage before its use (B two stages ahead, A one), so the compiler's vmcnt counts leave
 * the next stage in flight across the MFMA block.
 * A's E8M0 rows for the whole K are staged into LDS once, in MFMA lane order.
 * LDS: A 2x8192 + A scales 64x192 = 28,672 B, under the 32 KiB bridge alias. */
struct PipeStage {
    u32x4 b[2][2][2];   /* [branch][k-step][ni] */
    uint2 bs[2][2];     /* [branch][ni]: 8 E8M0 bytes, groups 8S..8S+7 */
    u32x4 a[2];         /* A rows a_row0 and a_row0 + 32, 16 B each */
};

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
    float linear_beta,
    const unsigned* __restrict__ row_token) {
    constexpr unsigned TM = 64, TN = 128, ROWB = 128, A_SLOT = TM * ROWB;
    constexpr unsigned AS_OFF = 2 * A_SLOT;
    extern __shared__ __align__(16) unsigned char a_lds[];
    const unsigned tid = threadIdx.x;
    const unsigned wave = tid / 64u;
    const unsigned lane = tid % 64u;
    const unsigned lane_k = lane / 16u;
    const unsigned lane_n = lane % 16u;
    if (hidden % 512u != 0u || hidden > 6144u || inter % TN != 0u) __builtin_trap();
    const unsigned n_tiles = inter / TN;
    const int* tile_prefix = meta + 2 * experts;
    const unsigned total_tiles = static_cast<unsigned>(tile_prefix[experts]);
    /* The grid is sized from the row capacity; only the first live WGs have work. Fold the live
     * ones so consecutive tiles (same expert, shared B rows) land on one XCD's L2. Bijective. */
    const unsigned live = total_tiles * n_tiles;
    unsigned bid = blockIdx.x;
    if (bid >= live) return;
    {
        const unsigned per = live / 8u;
        if (bid < per * 8u) bid = (bid % 8u) * per + bid / 8u;
    }
    const unsigned mt = bid / n_tiles;
    const unsigned nt = bid - mt * n_tiles;
    const unsigned expert = reuse_expert_for_tile(tile_prefix, mt, experts);
    const unsigned rowbase = static_cast<unsigned>(meta[expert])
        + (mt - static_cast<unsigned>(tile_prefix[expert])) * TM;
    const unsigned n0 = nt * TN;
    const PLOW_GLOB unsigned char* wb[2] = {
        reinterpret_cast<const PLOW_GLOB unsigned char*>(
            static_cast<size_t>(weight_table[static_cast<size_t>(expert) * 3u])),
        reinterpret_cast<const PLOW_GLOB unsigned char*>(
            static_cast<size_t>(weight_table[static_cast<size_t>(expert) * 3u + 1u]))};
    const PLOW_GLOB unsigned char* sbp[2] = {
        reinterpret_cast<const PLOW_GLOB unsigned char*>(
            static_cast<size_t>(weight_scale_table[static_cast<size_t>(expert) * 3u])),
        reinterpret_cast<const PLOW_GLOB unsigned char*>(
            static_cast<size_t>(weight_scale_table[static_cast<size_t>(expert) * 3u + 1u]))};
    if (!wb[0] || !wb[1] || !sbp[0] || !sbp[1]) return;
    const unsigned kbytes = hidden / 2u;
    const unsigned groups = hidden / 32u;
    const unsigned nst = hidden / 256u;

    f32x4 acc[4][4];
#pragma unroll
    for (unsigned mi = 0; mi < 4; ++mi)
#pragma unroll
        for (unsigned bi = 0; bi < 4; ++bi)
            acc[mi][bi] = f32x4{0, 0, 0, 0};

    const unsigned b_off = (n0 + wave * 32u + lane_n) * kbytes + lane_k * 16u;
    const unsigned bs_off = (n0 + wave * 32u + lane_n) * groups;
    const unsigned a_row = tid / 8u;
    const unsigned a_chunk = tid % 8u;
    /* Pad rows (row_token ~0) read token 0; their outputs are never stored. */
    auto tok_of = [&](unsigned r) {
        const unsigned t = row_token[rowbase + r];
        return static_cast<size_t>(t == ~0u ? 0u : t);
    };
    /* A: instruction i covers rows 32i + tid/8, whole 128 B lines (8 lanes per row). */
    const PLOW_GLOB unsigned char* a_src[2] = {
        as_glob(activation) + tok_of(a_row) * kbytes + a_chunk * 16u,
        as_glob(activation) + tok_of(a_row + 32u) * kbytes + a_chunk * 16u};
    {
        /* A scales, all K: 64 rows x groups bytes as 16 B pieces, stored [group][lane_n][mi]. */
        const unsigned pieces = groups / 16u;
        for (unsigned c = tid; c < TM * pieces; c += 256u) {
            const unsigned r = c / pieces, part = c - r * pieces;
            const u32x4 v = *reinterpret_cast<const PLOW_GLOB u32x4*>(
                as_glob(activation_scale) + tok_of(r) * groups + part * 16u);
            unsigned char* d = a_lds + AS_OFF + (part * 16u) * 64u + (r & 15u) * 4u + r / 16u;
#pragma unroll
            for (unsigned b = 0; b < 16; ++b)
                d[b * 64u] = static_cast<unsigned char>(v[b / 4u] >> ((b % 4u) * 8u));
        }
    }

    auto load = [&](unsigned s, PipeStage& st) {
        s = min(s, nst - 1u);
        const unsigned k0 = s * 128u;
        st.a[0] = *reinterpret_cast<const PLOW_GLOB u32x4*>(a_src[0] + k0);
        st.a[1] = *reinterpret_cast<const PLOW_GLOB u32x4*>(a_src[1] + k0);
#pragma unroll
        for (unsigned br = 0; br < 2; ++br) {
#pragma unroll
            for (unsigned ni = 0; ni < 2; ++ni) {
                const PLOW_GLOB unsigned char* p = wb[br] + b_off + ni * 16u * kbytes + k0;
                st.b[br][0][ni] = *reinterpret_cast<const PLOW_GLOB u32x4*>(p);
                st.b[br][1][ni] = *reinterpret_cast<const PLOW_GLOB u32x4*>(p + 64u);
                st.bs[br][ni] = *reinterpret_cast<const PLOW_GLOB uint2*>(
                    sbp[br] + bs_off + ni * 16u * groups + s * 8u);
            }
        }
    };
    auto commit_a = [&](const PipeStage& st, unsigned slot) {
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            const unsigned r = a_row + h * 32u;
            *reinterpret_cast<u32x4*>(a_lds + slot * A_SLOT + r * ROWB
                                      + ((a_chunk ^ ((r >> 1) & 7u)) * 16u)) = st.a[h];
        }
    };
    auto compute = [&](const PipeStage& st, unsigned slot, unsigned s) {
        const unsigned char* a_slot = a_lds + slot * A_SLOT;
#pragma unroll
        for (unsigned ks = 0; ks < 2; ++ks) {
            const unsigned chunk = ks * 4u + lane_k;
            const int a_scale_word = *reinterpret_cast<const int*>(
                a_lds + AS_OFF + ((s * 8u + chunk) * 16u + lane_n) * 4u);
            i32x8 a_frag[4];
#pragma unroll
            for (unsigned mi = 0; mi < 4; ++mi) {
                const unsigned row = lane_n + mi * 16u;
                a_frag[mi] = reuse_fp4_operand(
                    a_slot + row * ROWB + ((chunk ^ ((row >> 1) & 7u)) * 16u));
            }
            const unsigned sh = lane_k * 8u;
            int bsw[2];
#pragma unroll
            for (unsigned br = 0; br < 2; ++br) {
                const unsigned w0 = ks ? st.bs[br][0].y : st.bs[br][0].x;
                const unsigned w1 = ks ? st.bs[br][1].y : st.bs[br][1].x;
                bsw[br] = static_cast<int>(((w0 >> sh) & 0xffu) | (((w1 >> sh) & 0xffu) << 8));
            }
            i32x8 bf[2][2];
#pragma unroll
            for (unsigned br = 0; br < 2; ++br)
#pragma unroll
                for (unsigned ni = 0; ni < 2; ++ni) {
                    const u32x4 q = st.b[br][ks][ni];
                    bf[br][ni] = i32x8{static_cast<int>(q[0]), static_cast<int>(q[1]),
                                       static_cast<int>(q[2]), static_cast<int>(q[3]), 0, 0, 0, 0};
                }
            __builtin_amdgcn_s_setprio(1);
#define PIPE_MFMA_ROW(MI)                                                                     \
            acc[MI][0] = reuse_mfma<MI, 0>(a_frag[MI], bf[0][0], acc[MI][0], a_scale_word, bsw[0]); \
            acc[MI][1] = reuse_mfma<MI, 1>(a_frag[MI], bf[0][1], acc[MI][1], a_scale_word, bsw[0]); \
            acc[MI][2] = reuse_mfma<MI, 0>(a_frag[MI], bf[1][0], acc[MI][2], a_scale_word, bsw[1]); \
            acc[MI][3] = reuse_mfma<MI, 1>(a_frag[MI], bf[1][1], acc[MI][3], a_scale_word, bsw[1])
            PIPE_MFMA_ROW(0);
            PIPE_MFMA_ROW(1);
            PIPE_MFMA_ROW(2);
            PIPE_MFMA_ROW(3);
#undef PIPE_MFMA_ROW
            __builtin_amdgcn_s_setprio(0);
        }
    };

    PipeStage p0, p1;
    load(0, p0);
    load(1, p1);
    commit_a(p0, 0);
    __syncthreads();
    for (unsigned s = 0; s < nst; s += 2) {
        compute(p0, 0, s);
        commit_a(p1, 1);
        load(s + 2u, p0);
        __syncthreads();
        compute(p1, 1, s + 1u);
        commit_a(p0, 0);
        load(s + 3u, p1);
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
                bridge[row * TN + col] = mpf4_glu(
                    acc[mi][ni][i], acc[mi][2u + ni][i], act, beta, linear_beta);
            }
    __syncthreads();
    const unsigned row = tid / 4u;
    const unsigned block = tid % 4u;
    if (row_partidx[rowbase + row] != ~0u) {
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
#else
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
#endif
