#pragma once

template <unsigned Rows>
__device__ __forceinline__ void decode_wgmma_fp8(float (&acc)[Rows / 2],
    uint64_t weights, uint64_t inputs, int accumulate) {
    if constexpr (Rows == 8) {
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3}, %4, %5, p, 1, 1; }"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "l"(weights), "l"(inputs), "r"(accumulate));
    } else {
        static_assert(Rows == 16);
        asm volatile("{ .reg .pred p; setp.ne.b32 p, %10, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1; }"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
              "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
            : "l"(weights), "l"(inputs), "r"(accumulate));
    }
}

template <unsigned Rows, bool Glu>
constexpr unsigned decode_wgmma_bytes = 2 * (128 * (Glu ? 2 : 1) + Rows) * 128 + 1024;

template <unsigned Rows, bool Glu, bool Promote, bool CombinedQkv = false>
__global__ __launch_bounds__(256) void decode_w8a8(
    __nv_bfloat16* output, const uint8_t* x, const uint8_t* weights,
    const uint8_t* up_weights, const float* xscale, const float* wscale,
    const float* uscale, unsigned M, unsigned N, unsigned K,
    unsigned first_slice, unsigned nblk) {
    unsigned slice = first_slice + blockIdx.x;
    if constexpr (CombinedQkv) {
        static_assert(!Glu);
        const unsigned preceding = slice < 66 ? 0 : slice < 99 ? 8192 : 12288;
        output += (size_t)M * preceding;
        weights += (size_t)preceding * K;
        wscale += preceding;
        N = slice < 66 ? 8192 : 4096;
        nblk = slice < 66 ? 66 : 33;
        slice -= slice < 66 ? 0 : slice < 99 ? 66 : 99;
    }
    extern __shared__ __align__(16) uint8_t arena[];
    auto* base = reinterpret_cast<uint8_t*>(sm90_align1024(arena));
    constexpr unsigned weight_bytes = 128 * 128;
    constexpr unsigned input_offset = weight_bytes * (Glu ? 2 : 1);
    constexpr unsigned stage_bytes = input_offset + Rows * 128;
    constexpr unsigned Acc = Rows / 2;
    const unsigned tid = threadIdx.x, group = tid / 128, lane = tid & 31, warp = (tid / 32) & 3;
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned first = slice * per, end = min(first + per, N);
    const unsigned stages = (K + 127) / 128;
    auto stage = [&](unsigned buffer, unsigned col, unsigned kb) {
        uint8_t* memory = base + buffer * stage_bytes;
        pgm90_stage_fp8(memory, weights, tid, 128, col, kb, end, K);
        if constexpr (Glu) pgm90_stage_fp8(memory + weight_bytes, up_weights, tid, 128, col, kb, end, K);
        pgm90_stage_fp8(memory + input_offset, x, tid, Rows, 0, kb, M, K);
    };
    for (unsigned col = first; col < end; col += 128) {
        float gate[Acc] = {}, up[Acc] = {}, gate_sum[Acc] = {}, up_sum[Acc] = {};
        stage(0, col, 0); sm90_cp_commit();
        for (unsigned step = 0; step < stages; ++step) {
            if (step + 1 < stages) stage((step + 1) & 1, col, (step + 1) * 128);
            sm90_cp_commit();
            sm90_cp_wait<1>();
            __syncthreads();
            uint8_t* current = base + (step & 1) * stage_bytes;
            sm90_wg_fence();
#pragma unroll
            for (unsigned sub = 0; sub < 4; ++sub) {
                const uint64_t dw = sm90_desc(current + group * 64 * 128 + sub * 32);
                const uint64_t dx = sm90_desc(current + input_offset + sub * 32);
                const int accumulate = Promote ? sub != 0 : step != 0 || sub != 0;
                decode_wgmma_fp8<Rows>(gate, dw, dx, accumulate);
                if constexpr (Glu) decode_wgmma_fp8<Rows>(up,
                    sm90_desc(current + weight_bytes + group * 64 * 128 + sub * 32), dx, accumulate);
            }
            sm90_wg_commit();
            sm90_wg_wait<0>();
            if constexpr (Promote) {
#pragma unroll
                for (unsigned i = 0; i < Acc; ++i) {
                    gate_sum[i] += gate[i];
                    if constexpr (Glu) up_sum[i] += up[i];
                }
            }
            // Both warpgroups must finish reading a stage before it can be reused.
            __syncthreads();
        }
        sm90_cp_wait<0>();
        const unsigned n0 = col + group * 64 + warp * 16 + (lane >> 2);
        const unsigned m0 = (lane & 3) * 2;
#pragma unroll
        for (unsigned panel = 0; panel < Rows / 8; ++panel)
#pragma unroll
            for (unsigned high = 0; high < 2; ++high)
#pragma unroll
                for (unsigned low = 0; low < 2; ++low) {
                    const unsigned m = m0 + panel * 8 + low, n = n0 + high * 8;
                    const unsigned index = panel * 4 + high * 2 + low;
                    if (m < M && n < end) {
                        const float sum = Promote ? gate_sum[index] : gate[index];
                        const float scaled = __fmul_rn(wscale[n], __fmul_rn(xscale[m], sum));
                        float result = scaled;
                        if constexpr (Glu) {
                            const float usum = Promote ? up_sum[index] : up[index];
                            const float u = __fmul_rn(uscale[n], __fmul_rn(xscale[m], usum));
                            result = act_gelu_tanh(scaled) * u;
                        }
                        output[(size_t)m * N + n] = __float2bfloat16(result);
                    }
                }
        __syncthreads();
    }
}
