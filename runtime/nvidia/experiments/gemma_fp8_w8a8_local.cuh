#pragma once

template <unsigned Rows, bool Glu>
constexpr unsigned decode_wgmma_local_bytes = decode_wgmma_bytes<Rows, Glu> + 64;

template <unsigned Rows, bool Glu, bool CombinedQkv = false>
__global__ __launch_bounds__(256) void decode_w8a8_local(
    __nv_bfloat16* output, const __nv_bfloat16* x, const uint8_t* weights,
    const uint8_t* up_weights, const float* wscale, const float* uscale,
    unsigned M, unsigned N, unsigned K, unsigned first_slice, unsigned nblk) {
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
    const unsigned per = (N + nblk - 1) / nblk;
    const unsigned first = slice * per, end = min(first + per, N);
    if (first >= end) return;
    extern __shared__ __align__(16) uint8_t local_arena[];
    auto* base = reinterpret_cast<uint8_t*>(sm90_align1024(local_arena));
    constexpr unsigned weight_bytes = 128 * 128;
    constexpr unsigned input_offset = weight_bytes * (Glu ? 2 : 1);
    constexpr unsigned stage_bytes = input_offset + Rows * 128;
    constexpr unsigned Acc = Rows / 2;
    auto* scales = reinterpret_cast<float*>(base + 2 * stage_bytes);
    const unsigned tid = threadIdx.x, group = tid / 128, lane = tid & 31, warp = (tid / 32) & 3;
    for (unsigned row = tid / 32; row < M; row += 8) {
        float maximum = 0;
        for (unsigned k = lane * 8; k < K; k += 256) {
            const bf16v8 values = ld_glob8(x + (size_t)row * K + k);
#pragma unroll
            for (unsigned j = 0; j < 8; ++j) maximum = fmaxf(maximum, fabsf(__bfloat162float(values.x[j])));
        }
        maximum = warp_max32(maximum);
        if (lane == 0) scales[row] = fmaxf(__fdiv_rn(maximum, 448.f), 1.f / (448.f * 512.f));
    }
    __syncthreads();
    auto stage = [&](unsigned buffer, unsigned col, unsigned kb) {
        uint8_t* memory = base + buffer * stage_bytes;
        pgm90_stage_fp8(memory, weights, tid, 128, col, kb, end, K);
        if constexpr (Glu) pgm90_stage_fp8(memory + weight_bytes, up_weights, tid, 128, col, kb, end, K);
        for (unsigned line = tid; line < Rows * 16; line += 256) {
            const unsigned row = line / 16, column = (line % 16) * 8;
            uint8_t bytes[8] = {};
            if (row < M && kb + column < K) {
                const bf16v8 values = ld_glob8(x + (size_t)row * K + kb + column);
#pragma unroll
                for (unsigned j = 0; j < 8; ++j) {
                    const float value = __fdiv_rn(__bfloat162float(values.x[j]), scales[row]);
                    __nv_fp8_e4m3 q(fmaxf(-448.f, fminf(value, 448.f)));
                    bytes[j] = *reinterpret_cast<const uint8_t*>(&q);
                }
            }
            auto* destination = memory + input_offset + sm90_swz_off<128, 16>(row, column / 16) + (column & 15);
            *reinterpret_cast<uint2*>(destination) = *reinterpret_cast<const uint2*>(bytes);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    };
    const unsigned stages = (K + 127) / 128;
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
                const uint64_t dx = sm90_desc(current + input_offset + sub * 32);
                decode_wgmma_fp8<Rows>(gate, sm90_desc(current + group * 64 * 128 + sub * 32), dx, sub != 0);
                if constexpr (Glu) decode_wgmma_fp8<Rows>(up,
                    sm90_desc(current + weight_bytes + group * 64 * 128 + sub * 32), dx, sub != 0);
            }
            sm90_wg_commit();
            sm90_wg_wait<0>();
#pragma unroll
            for (unsigned i = 0; i < Acc; ++i) {
                gate_sum[i] += gate[i];
                if constexpr (Glu) up_sum[i] += up[i];
            }
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
                        const float g = __fmul_rn(wscale[n], __fmul_rn(scales[m], gate_sum[index]));
                        float result = g;
                        if constexpr (Glu) result = act_gelu_tanh(g) * __fmul_rn(uscale[n], __fmul_rn(scales[m], up_sum[index]));
                        output[(size_t)m * N + n] = __float2bfloat16(result);
                    }
                }
        __syncthreads();
    }
}
