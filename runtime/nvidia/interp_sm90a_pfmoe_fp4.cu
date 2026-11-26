/* interp_sm90a_pfmoe_fp4.cu -- prefill segment role object for DeepSeek-V4.1's routed experts:
 * PLOW_DOP_MOE_GROUP_GLU_PF (85) and PLOW_DOP_MOE_GROUP_DOWN_PF (86) with MXFP4 weights (i3 = 2), on
 * the Hopper wgmma tile of op_wg_moe_sm90.cuh (fp4 e2m1 + per-row 32-wide ue8m0 scales decoded to bf16
 * in registers). Geometry (384 threads, ~211 KiB of shared memory, setmaxnreg) is why it is a role
 * object.
 *
 * Numerics are the reference's W4A8 Expert.forward (model.py), not the AMD arm's A4W4:
 *   GLU:  fu[r] = fq(bf16(w[r] * silu(min(g, L)) * clamp(u, -L, L))), g/u = bf16(fq(x) . W1/W3)
 *         fq = act_quant(., 32, ue8m0) as a bf16 fake quant, the down projection's input
 *   DOWN: part[row_partidx[r]] = bf16(fu[r] . W2) (times row_gate[r] when t7 is bound)
 * so on NVIDIA:
 *   GLU  t0 = fu bf16 [rows][I] (not packed fp4)  t1 = fq(xn2) bf16 [T][H] (PLOW_DOP_ACT_QUANT_MX
 *        upstream)  t2/t3 = wtab/stab [E][3] {gate, up, down}  t4 = meta  t5 = row_token
 *        t7 = row_gate f32 (NONE = 1)  i0 = I  i1 = H  i2 = E  i3 = 2  i5 = act (1 silu,
 *        4 clamped swiglu)  f1 = limit.
 *   DOWN t0 = part f32 [T*k][H]  t1 = fu  t2/t3/t4 as GLU  t6 = row_partidx  t7 = row_gate or NONE
 *        (NONE when GLU applied it -- the reference's order)  i0 = H  i1 = I  i2 = E  i3 = 2.
 * meta is MoeAlignPf's rowoff[E] | cnt[E] | tilep[E+1]; each expert's rows are rowoff[e] ..
 * rowoff[e] + cnt[e], run in chunks of <= C::NT tokens (weights on the MMA's M side, tokens on
 * N = 8 .. 64). Work items are (expert, feature tile, chunk), chunk fastest, so the chunks that
 * share an expert's weights run side by side and meet in L2.
 */
#include "dev_isa.h"
#include "op_wg_moe_sm90.cuh"


extern "C" __device__ unsigned plow_pfmoe_fp4_abi = 1;
extern "C" __device__ unsigned plow_block_pfmoe_fp4 = 384;


#define PFMOE_ENC_MXFP4 2u
#define PFMOE_ACT_SILU 1u
#define PFMOE_ACT_SWIGLU_CLAMP 4u
#define PFMOE_MAX_E 512
#define PFMOE_UNUSED 0xFFFFFFFFu

namespace pfmoe {
using plow_wg::bf16;
// Two rings, picked per op by the weight passes each needs (sum over experts of ceil(cnt / NT)): the
// narrow one keeps a fourth stage for the weight stream, ~7% faster per pass (1k tokens, TP4: 16
// rows per expert); the wide one needs fewer passes once experts pass 32 rows (hot experts of real
// routing, or 4k tokens and up). One ring with runtime geometry, or GLU and DOWN sharing one
// instantiation, measured slower.
using Cfg = plow_wgm::Cfg<32, 4>;
using CfgL = plow_wgm::Cfg<64, 3>;
constexpr unsigned smem_max(unsigned a, unsigned b) { return a > b ? a : b; }

__device__ __forceinline__ float rbf(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float round_e4m3(float y) {
    const float a = fabsf(y);
    if (a == 0.f) return y;
    const int e = (int)((__float_as_uint(a) >> 23) & 0xffu) - 127;
    const float quantum = __uint_as_float((uint32_t)((e < -6 ? -6 : e) - 3 + 127) << 23);
    const float q = fminf(rintf(a / quantum) * quantum, 448.f);
    return y < 0.f ? -q : q;
}

/* skinny-tile epilogues: thread (w, g, t4) holds weight rows w * 16 + g + 8 h of block 0 (acc0) and
 * block 1 (acc1), token columns j * 8 + t4 * 2 + e. A 32-feature quant block spans warps w and
 * w ^ 1, so its amax meets in `red`. */
struct GluEpiS {
    bf16* fu;
    const float* gate;
    int f0, I, act;
    float lim;
    // Two passes over the accumulators, the activated value written back over the gate one: at
    // NT = 128 separate value and amax arrays on top of 128 accumulators spill.
    template <int NT>
    __device__ __forceinline__ void run(float* acc0, float* acc1, int cw, int row0, int rend, float* red) const {
        const int lt = threadIdx.x & 127, w = lt >> 5, lane = lt & 31, g = lane >> 2, t4 = lane & 3;
#pragma unroll
        for (int j = 0; j < NT / 8; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int p = row0 + j * 8 + t4 * 2 + e;
                const float rw = p < rend && gate ? gate[p] : 1.f;
                float m = 0.f;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    float gv = rbf(acc0[j * 4 + h * 2 + e]), uv = rbf(acc1[j * 4 + h * 2 + e]);
                    if (act == (int)PFMOE_ACT_SWIGLU_CLAMP) {
                        uv = fminf(fmaxf(uv, -lim), lim);
                        gv = fminf(gv, lim);
                    }
                    const float x = rbf(rw * ((gv / (1.f + expf(-gv))) * uv));
                    acc0[j * 4 + h * 2 + e] = x;
                    m = fmaxf(m, fabsf(x));
                }
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
                if (g == 0) red[(cw * 4 + w) * NT + j * 8 + t4 * 2 + e] = m;
            }
        asm volatile("bar.sync %0, 128;" ::"r"(1 + cw) : "memory");
#pragma unroll
        for (int j = 0; j < NT / 8; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int c = j * 8 + t4 * 2 + e, p = row0 + c;
                const float t = fmaxf(fmaxf(red[(cw * 4 + w) * NT + c], red[(cw * 4 + (w ^ 1)) * NT + c]), 1e-4f) * (1.0f / 448.0f);
                const uint32_t tb = __float_as_uint(t);
                const float sc = __uint_as_float((uint32_t)((int)(tb >> 23) + ((tb & 0x7fffffu) != 0u)) << 23);
                if (p >= rend) continue;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int f = f0 + cw * 64 + w * 16 + g + 8 * h;
                    if (f < I)
                        fu[(long long)p * I + f] =
                            __float2bfloat16_rn(round_e4m3(fminf(fmaxf(acc0[j * 4 + h * 2 + e] / sc, -448.f), 448.f)) * sc);
                }
            }
    }
};

struct DownEpiS {
    float* part;
    const unsigned* partidx;
    const float* gate;
    int n0, H;
    template <int NT>
    __device__ __forceinline__ void run(float* acc0, float* acc1, int cw, int row0, int rend, float*) const {
        const int lt = threadIdx.x & 127, w = lt >> 5, lane = lt & 31, g = lane >> 2, t4 = lane & 3;
#pragma unroll
        for (int j = 0; j < NT / 8; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int p = row0 + j * 8 + t4 * 2 + e;
                if (p >= rend) continue;
                const unsigned pidx = partidx[p];
                if (pidx == PFMOE_UNUSED) continue;
                const float rw = gate ? gate[p] : 1.f;
                float* dst = part + (long long)pidx * H;
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int f = n0 + cw * 64 + w * 16 + g + 8 * h;
                    if (f < H) dst[f] = rw * rbf(acc0[j * 4 + h * 2 + e]);
                    if (f + 128 < H) dst[f + 128] = rw * rbf(acc1[j * 4 + h * 2 + e]);
                }
            }
    }
};

struct Args {
    void* out;
    const bf16* a;
    const unsigned long long* wtab;
    const unsigned long long* stab;
    const int* meta;
    const unsigned* rowmap;  // GLU: row_token (A gather); DOWN: row_partidx (scatter)
    const float* gate;
    unsigned n, k, e, act, blocks;
    float lim;
    bool glu, valid;
};

// Work items (expert, feature tile, chunk of C::NT rows), chunk fastest, prefix-summed per expert
// into pfx; then the grouped GEMM over this CTA's items.
template <class C>
__device__ __forceinline__ void run(const Args& a, int* pfx, unsigned slice) {
    const unsigned E = a.e;
    const int* rowoff = a.meta;
    const int* cnt = a.meta + E;
    const unsigned tn = a.glu ? (a.n + 127) / 128 : (a.n + 255) / 256;
    for (unsigned e = threadIdx.x; e <= E; e += blockDim.x) pfx[e] = e < E ? (cnt[e] + C::NT - 1) / C::NT * tn : 0;
    __syncthreads();
    for (unsigned off = 1; off <= E; off <<= 1) {
        int v[2];
        for (unsigned i = 0; i < 2; i++) {
            const unsigned e = threadIdx.x + i * blockDim.x;
            v[i] = e <= E && e >= off ? pfx[e - off] : 0;
        }
        __syncthreads();
        for (unsigned i = 0; i < 2; i++) {
            const unsigned e = threadIdx.x + i * blockDim.x;
            if (e <= E) pfx[e] += v[i];
        }
        __syncthreads();
    }
    auto tile = [&](unsigned item) {
        unsigned lo_e = 0, hi_e = E - 1;  // first e with pfx[e] > item
        while (lo_e < hi_e) {
            const unsigned mid = (lo_e + hi_e) >> 1;
            if ((unsigned)pfx[mid] > item) hi_e = mid;
            else lo_e = mid + 1;
        }
        const unsigned e = lo_e, local = item - (e ? (unsigned)pfx[e - 1] : 0u), nch = (cnt[e] + C::NT - 1) / C::NT;
        const int row0 = rowoff[e] + (int)(local % nch * C::NT), rend = min(rowoff[e] + cnt[e], row0 + C::NT);
        const unsigned nt = local / nch;
        const unsigned long long* wt = a.wtab + (size_t)e * 3;
        const unsigned long long* st = a.stab + (size_t)e * 3;
        return a.glu ? plow_wgm::Tile{row0, rend, (int)(nt * 128), (const int*)a.rowmap, (const uint8_t*)wt[0], (const uint8_t*)st[0],
                                      (const uint8_t*)wt[1], (const uint8_t*)st[1]}
                     : plow_wgm::Tile{row0, rend, (int)(nt * 256), nullptr, (const uint8_t*)wt[2], (const uint8_t*)st[2], nullptr, nullptr};
    };
    if (a.glu)
        plow_wgm::moe_run<C>(a.a, a.k, (int)a.n, (int)a.k, slice, (unsigned)pfx[E - 1], a.blocks, tile,
                             [&](const plow_wgm::Tile& t) { return GluEpiS{(bf16*)a.out, a.gate, t.n0, (int)a.n, (int)a.act, a.lim}; });
    else
        plow_wgm::moe_run<C>(a.a, a.k, (int)a.n, (int)a.k, slice, (unsigned)pfx[E - 1], a.blocks, tile,
                             [&](const plow_wgm::Tile& t) { return DownEpiS{(float*)a.out, a.rowmap, a.gate, t.n0, (int)a.n}; });
}
}  // namespace pfmoe

extern "C" __device__ unsigned plow_arena_bytes_pfmoe_fp4 = pfmoe::smem_max(pfmoe::Cfg::SMEM, pfmoe::CfgL::SMEM);

__device__ __forceinline__ unsigned pfmoe_ctr_poll(const unsigned* p) {
    unsigned value;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}
__device__ __forceinline__ void pfmoe_ctr_signal(unsigned* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}

extern "C" __global__ __launch_bounds__(384, 1) void plow_sm90a_pfmoe_fp4(PlowProgram prog) {
    __shared__ PlowStreamEnt entry_s;
    __shared__ pfmoe::Args args;
    __shared__ int sk_pfx[PFMOE_MAX_E + 1];  // work items up to expert e (inclusive after the scan)
    __shared__ int passes[2];                // weight passes over all experts, narrow | wide ring
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    for (unsigned index = lo + blockIdx.x; index < hi; index += gridDim.x) {
        if (threadIdx.x == 0) entry_s = prog.gq_stream[index];
        __syncthreads();
        const PlowStreamEnt entry = entry_s;
        if (entry.flags & PLOW_SE_XCTR) __trap();
        for (unsigned w = threadIdx.x; w < entry.wait_len; w += blockDim.x) {
            const PlowWait wait = prog.waits[entry.wait_ofs + w];
            while (pfmoe_ctr_poll(PLOW_CTR(prog.counters, wait.id)) < wait.threshold) {
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const PlowDevInst* const in = prog.insts + entry.inst;
            pfmoe::Args& a = args;
            a.glu = in->op == PLOW_DOP_MOE_GROUP_GLU_PF;
            const bool down = in->op == PLOW_DOP_MOE_GROUP_DOWN_PF;
            const unsigned map_slot = a.glu ? 5u : 6u;
            a.valid = (a.glu || down) && in->i[3] == PFMOE_ENC_MXFP4 && in->i[0] > 0 && in->i[1] % 64 == 0 &&
                      in->i[2] > 0 && in->i[2] <= PFMOE_MAX_E && in->i[6] == 0 && in->i[7] == 0 &&
                      (!down || (in->i[4] == 0 && in->i[5] == 0)) &&
                      (!a.glu || in->i[5] == PFMOE_ACT_SILU || in->i[5] == PFMOE_ACT_SWIGLU_CLAMP) &&
                      (!a.glu || in->i[0] % 32 == 0) && in->t[0] != PLOW_TENSOR_NONE && in->t[1] != PLOW_TENSOR_NONE &&
                      in->t[2] != PLOW_TENSOR_NONE && in->t[3] != PLOW_TENSOR_NONE && in->t[4] != PLOW_TENSOR_NONE &&
                      in->t[map_slot] != PLOW_TENSOR_NONE && in->blocks > 0 && entry.slice < in->blocks;
            if (a.valid) {
                a.out = prog.tensors[in->t[0]];
                a.a = static_cast<const pfmoe::bf16*>(prog.tensors[in->t[1]]);
                a.wtab = static_cast<const unsigned long long*>(prog.tensors[in->t[2]]);
                a.stab = static_cast<const unsigned long long*>(prog.tensors[in->t[3]]);
                a.meta = static_cast<const int*>(prog.tensors[in->t[4]]);
                a.rowmap = static_cast<const unsigned*>(prog.tensors[in->t[map_slot]]);
                a.gate = in->t[7] == PLOW_TENSOR_NONE ? nullptr : static_cast<const float*>(prog.tensors[in->t[7]]);
                a.n = in->i[0];
                a.k = in->i[1];
                a.e = in->i[2];
                a.act = in->i[5];
                a.lim = in->fj[1].f;
                a.blocks = in->blocks;
            }
        }
        __syncthreads();
        if (!args.valid) __trap();
        if (threadIdx.x < 2) passes[threadIdx.x] = 0;
        __syncthreads();
        for (unsigned e = threadIdx.x; e < args.e; e += blockDim.x) {
            const int c = args.meta[args.e + e];
            atomicAdd(&passes[0], (c + pfmoe::Cfg::NT - 1) / pfmoe::Cfg::NT);
            atomicAdd(&passes[1], (c + pfmoe::CfgL::NT - 1) / pfmoe::CfgL::NT);
        }
        __syncthreads();
        if (15 * passes[1] >= 14 * passes[0]) pfmoe::run<pfmoe::Cfg>(args, sk_pfx, entry.slice);
        else pfmoe::run<pfmoe::CfgL>(args, sk_pfx, entry.slice);
        __threadfence();
        __syncthreads();
        for (unsigned s = threadIdx.x; s < entry.succ_len; s += blockDim.x)
            pfmoe_ctr_signal(PLOW_CTR(prog.counters, prog.succs[entry.succ_ofs + s]));
        __syncthreads();
    }
}
