#include <metal_stdlib>
using namespace metal;

kernel void asr_causal_conv3_f16(
    device const float *x [[buffer(0)]],
    device const half *w [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint *shape [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    uint frames = shape[0], width = shape[1], input_channels = shape[2];
    uint output_channels = shape[3], depthwise = shape[4], relu = shape[5];
    uint output_frames = frames / 2u + 1u, output_width = width / 2u + 1u;
    uint count = output_frames * output_width * output_channels;
    if (index >= count) return;
    uint output_channel = index % output_channels;
    uint output_x = (index / output_channels) % output_width;
    uint output_frame = index / (output_channels * output_width);
    uint channel_start = depthwise != 0u ? output_channel : 0u;
    uint channel_end = depthwise != 0u ? output_channel + 1u : input_channels;
    uint stored_channels = depthwise != 0u ? 1u : input_channels;
    float sum = bias[output_channel];
    for (uint ky = 0u; ky < 3u; ++ky) {
        int input_frame = int(output_frame * 2u + ky) - 2;
        if (input_frame < 0 || input_frame >= int(frames)) continue;
        for (uint kx = 0u; kx < 3u; ++kx) {
            int input_x = int(output_x * 2u + kx) - 2;
            if (input_x < 0 || input_x >= int(width)) continue;
            for (uint input_channel = channel_start; input_channel < channel_end; ++input_channel) {
                uint weight_channel = depthwise != 0u ? 0u : input_channel;
                uint weight_index = (((output_channel * stored_channels + weight_channel) * 3u + ky) * 3u) + kx;
                uint input_index = (uint(input_frame) * width + uint(input_x)) * input_channels + input_channel;
                sum = fma(x[input_index], float(w[weight_index]), sum);
            }
        }
    }
    y[index] = relu != 0u ? max(sum, 0.0f) : sum;
}

kernel void asr_pointwise_f16(
    device const float *x [[buffer(0)]],
    device const half *w [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint *shape [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m = shape[0], n = shape[1], k = shape[2], relu = shape[3];
    uint n_tiles = (n + 63u) / 64u;
    uint row = (group / n_tiles) * 32u, col = (group % n_tiles) * 64u;
    uint simd_row = (simd / 4u) * 16u, simd_col = (simd % 4u) * 16u;
    threadgroup float a_tile[1024], b_tile[2048];
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
    for (uint base = 0u; base < k; base += 32u) {
        for (uint e = lid; e < 1024u; e += 256u) {
            uint r = e / 32u, c = e % 32u;
            a_tile[e] = row + r < m && base + c < k ? x[(row + r) * k + base + c] : 0.0f;
        }
        for (uint e = lid; e < 2048u; e += 256u) {
            uint r = e / 32u, c = e % 32u;
            b_tile[e] = col + r < n && base + c < k ? float(w[(col + r) * k + base + c]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 32u; q += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, a_tile + simd_row * 32u + q, 32);
            simdgroup_load(a1, a_tile + (simd_row + 8u) * 32u + q, 32);
            simdgroup_load(b0, b_tile + simd_col * 32u + q, 32, ulong2(0), true);
            simdgroup_load(b1, b_tile + (simd_col + 8u) * 32u + q, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(acc0, a0, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a0, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a1, b0, acc2);
            simdgroup_multiply_accumulate(acc3, a1, b1, acc3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint quad = lane / 4u;
    uint fragment_row = (quad & 4u) + ((lane / 2u) % 4u);
    uint fragment_col = (quad & 2u) * 2u + (lane % 2u) * 2u;
    for (uint e = 0u; e < 2u; ++e) {
        uint r = row + simd_row + fragment_row, c = col + simd_col + fragment_col + e;
        if (r < m && c < n) {
            float value = acc0.thread_elements()[e] + bias[c];
            y[r * n + c] = relu != 0u ? max(value, 0.0f) : value;
        }
        if (r < m && c + 8u < n) {
            float value = acc1.thread_elements()[e] + bias[c + 8u];
            y[r * n + c + 8u] = relu != 0u ? max(value, 0.0f) : value;
        }
        if (r + 8u < m && c < n) {
            float value = acc2.thread_elements()[e] + bias[c];
            y[(r + 8u) * n + c] = relu != 0u ? max(value, 0.0f) : value;
        }
        if (r + 8u < m && c + 8u < n) {
            float value = acc3.thread_elements()[e] + bias[c + 8u];
            y[(r + 8u) * n + c + 8u] = relu != 0u ? max(value, 0.0f) : value;
        }
    }
}
