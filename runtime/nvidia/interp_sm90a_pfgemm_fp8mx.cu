/* interp_sm90a_pfgemm_fp8mx.cu -- prefill segment role object for PLOW_DOP_GEMM_FP8_MX (op 210).
 *
 * DeepSeek-V4.1's block-fp8 projections at a [32, 32] ue8m0 grid on the Hopper wgmma tile of
 * op_wg_sm90.cuh: one 128 x 256 output tile per iteration, the producer warpgroup decoding the e4m3
 * weights to bf16 in shared memory, two consumer warpgroups on wgmma. Its geometry (384 threads,
 * ~206 KiB of dynamic shared memory) is not the interpreter's, which is why it is a role object.
 *
 * A stream entry's slice runs tiles slice, slice + blocks, ... of the groups x ceil(T/128) x
 * ceil(N/256) grid, so any `blocks` covers the op. x is the bf16 activation as the packet holds it --
 * after PLOW_DOP_ACT_QUANT_MX on the reference-exact emit. i3 = groups (block-diagonal: group g
 * multiplies x columns g*K.. by weight rows g*N.. into out columns g*N..); i4 / i5 = a_row0 / c_row0,
 * a row band of A and C (the banded TP seam), as the AMD arm reads them.
 *
 * i6 = 1: x is e4m3 and t4 its ue8m0 scales [rows][groups * K / 32] -- act_quant's (xq, xs), the
 * reference fp8_gemm operands -- and the tile is the fp8 wgmma of op_wg_fp8_sm90.cuh with per-32
 * promotion (K % 128 == 0). i6 = 0 is the bf16-decode tile.
 */
#include "dev_isa.h"
#include "op_wg_fp8_sm90.cuh"

extern "C" __device__ unsigned plow_pfgemm_fp8mx_abi = 1;
extern "C" __device__ unsigned plow_block_pfgemm_fp8mx = 384;
extern "C" __device__ unsigned plow_arena_bytes_pfgemm_fp8mx = WG_SMEM > F8_SMEM ? WG_SMEM : F8_SMEM;

struct Fp8MxArgs {
    __nv_bfloat16* out;
    const __nv_bfloat16* x;
    const uint8_t* w;
    const uint8_t* scale;
    const uint8_t* xs;
    unsigned t, n, k, groups, blocks, fp8;
    bool valid;
};

__device__ __forceinline__ unsigned fp8mx_ctr_poll(const unsigned* p) {
    unsigned value;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}
__device__ __forceinline__ void fp8mx_ctr_signal(unsigned* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}

extern "C" __global__ __launch_bounds__(384, 1) void plow_sm90a_pfgemm_fp8mx(PlowProgram prog) {
    __shared__ PlowStreamEnt entry_s;
    __shared__ Fp8MxArgs args;
    const unsigned lo = prog.gq_seg_ofs[prog.cur_seg];
    const unsigned hi = prog.gq_seg_ofs[prog.cur_seg + 1];
    for (unsigned index = lo + blockIdx.x; index < hi; index += gridDim.x) {
        if (threadIdx.x == 0) entry_s = prog.gq_stream[index];
        __syncthreads();
        const PlowStreamEnt entry = entry_s;
        if (entry.flags & PLOW_SE_XCTR) __trap();
        for (unsigned w = threadIdx.x; w < entry.wait_len; w += blockDim.x) {
            const PlowWait wait = prog.waits[entry.wait_ofs + w];
            while (fp8mx_ctr_poll(PLOW_CTR(prog.counters, wait.id)) < wait.threshold) {
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const PlowDevInst* const in = prog.insts + entry.inst;
            args.valid = in->op == PLOW_DOP_GEMM_FP8_MX && in->i[0] > 0 && in->i[1] > 0 && in->i[2] > 0 && in->i[2] % 64 == 0 &&
                         in->t[0] != PLOW_TENSOR_NONE && in->t[1] != PLOW_TENSOR_NONE && in->t[2] != PLOW_TENSOR_NONE &&
                         in->t[3] != PLOW_TENSOR_NONE && in->blocks > 0 && entry.slice < in->blocks &&
                         (in->i[3] <= 1u || in->i[1] % 32 == 0) && in->i[6] <= 1u &&
                         (in->i[6] == 0u || (in->i[2] % 128 == 0 && in->t[4] != PLOW_TENSOR_NONE));
            if (args.valid) {
                args.groups = in->i[3] > 1u ? in->i[3] : 1u;
                args.out = static_cast<__nv_bfloat16*>(prog.tensors[in->t[0]]) + (size_t)in->i[5] * args.groups * in->i[1];
                args.fp8 = in->i[6];
                args.x = static_cast<const __nv_bfloat16*>(prog.tensors[in->t[1]]);
                if (args.fp8) {
                    args.x = (const __nv_bfloat16*)((const uint8_t*)args.x + (size_t)in->i[4] * args.groups * in->i[2]);
                    args.xs = static_cast<const uint8_t*>(prog.tensors[in->t[4]]) + (size_t)in->i[4] * args.groups * (in->i[2] / 32);
                } else {
                    args.x += (size_t)in->i[4] * args.groups * in->i[2];
                }
                args.w = static_cast<const uint8_t*>(prog.tensors[in->t[2]]);
                args.scale = static_cast<const uint8_t*>(prog.tensors[in->t[3]]);
                args.t = in->i[0];
                args.n = in->i[1];
                args.k = in->i[2];
                args.blocks = in->blocks;
            }
        }
        __syncthreads();
        if (!args.valid) __trap();
        if (args.fp8) {
            const unsigned g = args.groups;
            const plow_wg8::Fp8Job job{args.out, (long long)g * args.n, (const uint8_t*)args.x, (long long)g * args.k, args.xs,
                                       (long long)g * (args.k / 32), args.w, args.scale, (int)args.t, (int)args.n, (int)args.k, (int)g};
            plow_wg8::gemm_fp8(job, entry.slice, args.blocks);
        } else {
        const unsigned tn = (args.n + WG_BN - 1) / WG_BN;
        const unsigned per_group = tn * ((args.t + WG_BM - 1) / WG_BM);
        const unsigned g = args.groups;
        for (unsigned tile = entry.slice; tile < per_group * g; tile += args.blocks) {
            const unsigned r = tile % per_group;
            plow_wg::gemm_tile<0>(args.out, args.x, args.w, args.scale, nullptr, nullptr, nullptr, nullptr, 0, (int)args.t,
                                  (int)args.n, (int)args.k, (long long)g * args.k, (long long)g * args.n, 0,
                                  (long long)args.n * args.k, (long long)(args.n / 32) * ((args.k + 31) / 32), args.k, args.n,
                                  (int)(r % tn), (int)(r / tn), (int)(tile / per_group));
        }
        }
        __threadfence();
        __syncthreads();
        for (unsigned s = threadIdx.x; s < entry.succ_len; s += blockDim.x)
            fp8mx_ctr_signal(PLOW_CTR(prog.counters, prog.succs[entry.succ_ofs + s]));
        __syncthreads();
    }
}
