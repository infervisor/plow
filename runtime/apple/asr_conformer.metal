#include <metal_stdlib>
using namespace metal;

kernel void conformer_layer_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint *shape [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint rows = shape[0], width = shape[1];
    if (row >= rows) return;
    float sum = 0.0f;
    for (uint column = lane; column < width; column += 32u) {
        sum += x[ulong(row) * width + column];
    }
    float mean = simd_sum(sum) / float(width);
    float square_sum = 0.0f;
    for (uint column = lane; column < width; column += 32u) {
        float centered = x[ulong(row) * width + column] - mean;
        square_sum = fma(centered, centered, square_sum);
    }
    float inverse = rsqrt(simd_sum(square_sum) / float(width) + 1e-5f);
    for (uint column = lane; column < width; column += 32u) {
        ulong index = ulong(row) * width + column;
        y[index] = fma((x[index] - mean) * inverse, weight[column], bias[column]);
    }
}

kernel void conformer_silu(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    float value = x[index];
    y[index] = value / (1.0f + exp(-value));
}

kernel void conformer_scaled_add(
    device const float *x [[buffer(0)]],
    device const float *other [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant float &scale [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) y[index] = fma(scale, other[index], x[index]);
}

kernel void conformer_glu(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint *shape [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    uint count = shape[0] * shape[1];
    if (index >= count) return;
    uint width = shape[1], row = index / width, column = index % width;
    float gate = x[ulong(row) * width * 2u + width + column];
    y[index] = x[ulong(row) * width * 2u + column] / (1.0f + exp(-gate));
}

kernel void conformer_depthwise_causal(
    device const float *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint *shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint rows = shape[0], width = shape[1], kernel_width = shape[2];
    if (index >= rows * width) return;
    uint row = index / width, channel = index % width;
    float sum = 0.0f;
    for (uint tap = 0; tap < kernel_width; ++tap) {
        int source = int(row) + int(tap) + 1 - int(kernel_width);
        if (source >= 0) {
            sum = fma(x[ulong(source) * width + channel],
                      float(weight[ulong(channel) * kernel_width + tap]), sum);
        }
    }
    y[index] = sum;
}

kernel void conformer_relative_attention_fused(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device const float *position [[buffer(3)]],
    device const float *bias_u [[buffer(4)]],
    device const float *bias_v [[buffer(5)]],
    device float *context [[buffer(6)]],
    constant uint *shape [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint rows = shape[0], width = shape[1], heads = shape[2];
    uint chunk_size = shape[3], left_chunks = shape[4];
    uint head = group % heads, query_row = group / heads;
    if (query_row >= rows) return;
    uint query_chunk = query_row / chunk_size;
    uint first_chunk = query_chunk > left_chunks ? query_chunk - left_chunks : 0u;
    uint first_key = first_chunk * chunk_size;
    uint last_key = min(rows, (query_chunk + 1u) * chunk_size);
    uint key_count = last_key - first_key;
    uint head_width = width / heads;
    uint query_base = query_row * width + head * head_width;
    uint bias_base = head * head_width;
    threadgroup float scores[64];
    for (uint local_key = simd; local_key < key_count; local_key += 4u) {
        uint key_row = first_key + local_key;
        uint key_base = key_row * width + head * head_width;
        uint relative_row = rows - 1u + key_row - query_row;
        uint position_base = relative_row * width + head * head_width;
        float content = 0.0f, relative = 0.0f;
        for (uint column = lane; column < head_width; column += 32u) {
            float q = query[query_base + column];
            content = fma(key[key_base + column], q + bias_u[bias_base + column], content);
            relative = fma(position[position_base + column], q + bias_v[bias_base + column], relative);
        }
        float score = simd_sum(content + relative) * rsqrt(float(head_width));
        if (lane == 0u) scores[local_key] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
        float maximum = -INFINITY;
        for (uint key_index = 0; key_index < key_count; ++key_index) {
            maximum = max(maximum, scores[key_index]);
        }
        float sum = 0.0f;
        for (uint key_index = 0; key_index < key_count; ++key_index) {
            scores[key_index] = exp(scores[key_index] - maximum);
            sum += scores[key_index];
        }
        float inverse = 1.0f / sum;
        for (uint key_index = 0; key_index < key_count; ++key_index) scores[key_index] *= inverse;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = thread_index; column < head_width; column += 128u) {
        float sum = 0.0f;
        for (uint local_key = 0; local_key < key_count; ++local_key) {
            uint key_row = first_key + local_key;
            ulong value_index = ulong(key_row) * width + head * head_width + column;
            sum = fma(scores[local_key], value[value_index], sum);
        }
        context[ulong(query_row) * width + head * head_width + column] = sum;
    }
}
