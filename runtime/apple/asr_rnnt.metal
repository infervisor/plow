#include <metal_stdlib>
using namespace metal;

kernel void rnnt_q8_gemv4(
    device const float *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint *shape [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint rows = shape[0], columns = shape[1], width = shape[2], simdgroups = shape[3];
    uint groups_per_row = (columns + simdgroups * 4u - 1u) / (simdgroups * 4u);
    uint batch = group / groups_per_row;
    uint output_group = group % groups_per_row;
    uint row_in_simd = lane >> 3u;
    uint row_lane = lane & 7u;
    uint column = (output_group * simdgroups + simd) * 4u + row_in_simd;
    if (batch >= rows || column >= columns) return;
    uint blocks = width / 32u;
    device const uchar *weight_row = w + ulong(column) * blocks * 34u;
    device const float *input_row = x + ulong(batch) * width;
    float sum = 0.0f;
    for (uint block = row_lane; block < blocks; block += 8u) {
        device const uchar *q8 = weight_row + block * 34u;
        ushort scale_bits = ushort(q8[0]) | ushort(ushort(q8[1]) << 8u);
        half scale = as_type<half>(scale_bits);
        uint x0 = block * 32u;
        for (uint i = 0; i < 32u; i += 4u) {
            packed_char4 packed =
                *reinterpret_cast<device const packed_char4 *>(q8 + 2u + i);
            float4 activation =
                *reinterpret_cast<device const float4 *>(input_row + x0 + i);
            sum = fma(dot(float4(packed), activation), float(scale), sum);
        }
    }
    sum += simd_shuffle_xor(sum, 4u);
    sum += simd_shuffle_xor(sum, 2u);
    sum += simd_shuffle_xor(sum, 1u);
    if (row_lane == 0u) y[ulong(batch) * columns + column] = sum + bias[column];
}

kernel void rnnt_embed_f16_f32(
    device float *out [[buffer(0)]],
    device const half *table [[buffer(1)]],
    device const uint *token [[buffer(2)]],
    constant uint *shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < shape[1] && *token < shape[0]) {
        out[index] = float(table[ulong(*token) * shape[1] + index]);
    }
}

kernel void rnnt_scaled_add(
    device float *out [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *other [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant float &scale [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) out[index] = fma(scale, other[index], x[index]);
}

kernel void rnnt_lstm_cell(
    device float *h [[buffer(0)]],
    device float *c [[buffer(1)]],
    device const float *gates [[buffer(2)]],
    device const float *previous [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    float input = 1.0f / (1.0f + exp(-gates[index]));
    float forget = 1.0f / (1.0f + exp(-gates[width + index]));
    float output = 1.0f / (1.0f + exp(-gates[3u * width + index]));
    float cell = fma(forget, previous[index], input * tanh(gates[2u * width + index]));
    c[index] = cell;
    h[index] = output * tanh(cell);
}

kernel void rnnt_broadcast_add(
    device float *out [[buffer(0)]],
    device const float *matrix [[buffer(1)]],
    device const float *vector [[buffer(2)]],
    constant uint *shape [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint count = shape[0] * shape[1];
    if (index < count) out[index] = matrix[index] + vector[index % shape[1]];
}

kernel void rnnt_relu(
    device float *out [[buffer(0)]],
    device const float *x [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) out[index] = max(x[index], 0.0f);
}

kernel void rnnt_argmax(
    device uint *ids [[buffer(0)]],
    device const float *x [[buffer(1)]],
    constant uint *shape [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best = -INFINITY;
    uint best_id = 0u;
    for (uint column = lid; column < shape[1]; column += 256u) {
        float value = x[ulong(row) * shape[1] + column];
        if (value > best || (value == best && column < best_id)) {
            best = value;
            best_id = column;
        }
    }
    values[lid] = best;
    indices[lid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            float other = values[lid + stride];
            uint other_id = indices[lid + stride];
            if (other > values[lid] || (other == values[lid] && other_id < indices[lid])) {
                values[lid] = other;
                indices[lid] = other_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u && row < shape[0]) ids[row] = indices[0];
}
