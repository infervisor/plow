#include <metal_stdlib>
using namespace metal;

kernel void q8_0_gemv(
    device const float *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint *shape [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint n = group * shape[2] + simd;
    if (n >= shape[0]) return;
    uint blocks = shape[1] / 32u;
    device const uchar *row = w + ulong(n) * blocks * 34u;
    float sum = 0.0f;
    for (uint block = lane; block < blocks; block += 32u) {
        device const uchar *q8 = row + block * 34u;
        ushort scale_bits = ushort(q8[0]) | ushort(ushort(q8[1]) << 8u);
        half scale = as_type<half>(scale_bits);
        uint x0 = block * 32u;
        for (uint i = 0; i < 32u; i += 4u) {
            packed_char4 packed = *reinterpret_cast<device const packed_char4 *>(q8 + 2u + i);
            float4 activation = *reinterpret_cast<device const float4 *>(x + x0 + i);
            sum = fma(dot(float4(packed), activation), float(scale), sum);
        }
    }
    sum = simd_sum(sum);
    if (lane == 0u) y[n] = sum;
}

kernel void q8_0_gemv4(
    device const float *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint *shape [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint row_in_simd = lane >> 3u;
    uint row_lane = lane & 7u;
    uint n = (group * shape[2] + simd) * 4u + row_in_simd;
    if (n >= shape[0]) return;
    uint blocks = shape[1] / 32u;
    device const uchar *row = w + ulong(n) * blocks * 34u;
    float sum = 0.0f;
    for (uint block = row_lane; block < blocks; block += 8u) {
        device const uchar *q8 = row + block * 34u;
        ushort scale_bits = ushort(q8[0]) | ushort(ushort(q8[1]) << 8u);
        half scale = as_type<half>(scale_bits);
        uint x0 = block * 32u;
        for (uint i = 0; i < 32u; i += 4u) {
            packed_char4 packed = *reinterpret_cast<device const packed_char4 *>(q8 + 2u + i);
            float4 activation = *reinterpret_cast<device const float4 *>(x + x0 + i);
            sum = fma(dot(float4(packed), activation), float(scale), sum);
        }
    }
    sum += simd_shuffle_xor(sum, 4u);
    sum += simd_shuffle_xor(sum, 2u);
    sum += simd_shuffle_xor(sum, 1u);
    if (row_lane == 0u) y[n] = sum;
}

kernel void q8_0_gemm(
    device const float *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint *shape [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m = shape[0], n = shape[1], k = shape[2], use_bias = shape[3];
    uint n_tiles = (n + 63u) / 64u;
    uint row = (group / n_tiles) * 32u, col = (group % n_tiles) * 64u;
    uint simd_row = (simd / 4u) * 16u, simd_col = (simd % 4u) * 16u;
    uint row_bytes = (k / 32u) * 34u;
    threadgroup float a_tile[1024], b_tile[2048];
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);

    for (uint base = 0u; base < k; base += 32u) {
        for (uint e = lid; e < 1024u; e += 256u) {
            uint r = e / 32u, c = e % 32u;
            a_tile[e] = row + r < m ? x[(row + r) * k + base + c] : 0.0f;
        }
        for (uint e = lid; e < 2048u; e += 256u) {
            uint r = e / 32u, c = e % 32u;
            if (col + r < n) {
                device const uchar *block = w + ulong(col + r) * row_bytes + (base / 32u) * 34u;
                ushort scale_bits = ushort(block[0]) | ushort(ushort(block[1]) << 8u);
                b_tile[e] = float(as_type<char>(block[2u + c])) * float(as_type<half>(scale_bits));
            } else {
                b_tile[e] = 0.0f;
            }
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
        if (r < m && c < n) y[r * n + c] = acc0.thread_elements()[e] + (use_bias != 0u ? bias[c] : 0.0f);
        if (r < m && c + 8u < n) y[r * n + c + 8u] = acc1.thread_elements()[e] + (use_bias != 0u ? bias[c + 8u] : 0.0f);
        if (r + 8u < m && c < n) y[(r + 8u) * n + c] = acc2.thread_elements()[e] + (use_bias != 0u ? bias[c] : 0.0f);
        if (r + 8u < m && c + 8u < n) y[(r + 8u) * n + c + 8u] = acc3.thread_elements()[e] + (use_bias != 0u ? bias[c + 8u] : 0.0f);
    }
}

kernel void q8_0_gemm64(
    device const float *x [[buffer(0)]],
    device const uchar *w [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint *shape [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m = shape[0], n = shape[1], k = shape[2], use_bias = shape[3];
    uint n_tiles = (n + 63u) / 64u;
    uint row = (group / n_tiles) * 64u, col = (group % n_tiles) * 64u;
    uint simd_row = (simd / 4u) * 32u, simd_col = (simd % 4u) * 16u;
    uint row_bytes = (k / 32u) * 34u;
    threadgroup float a_tile[2048], b_tile[2048];
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
    simdgroup_float8x8 acc4(0.0f), acc5(0.0f), acc6(0.0f), acc7(0.0f);

    for (uint base = 0u; base < k; base += 32u) {
        for (uint e = lid; e < 2048u; e += 256u) {
            uint r = e / 32u, c = e % 32u;
            a_tile[e] = row + r < m ? x[(row + r) * k + base + c] : 0.0f;
        }
        for (uint e = lid; e < 2048u; e += 256u) {
            uint r = e / 32u, c = e % 32u;
            if (col + r < n) {
                device const uchar *block = w + ulong(col + r) * row_bytes + (base / 32u) * 34u;
                ushort scale_bits = ushort(block[0]) | ushort(ushort(block[1]) << 8u);
                b_tile[e] = float(as_type<char>(block[2u + c])) * float(as_type<half>(scale_bits));
            } else {
                b_tile[e] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0u; q < 32u; q += 8u) {
            simdgroup_float8x8 a0, a1, a2, a3, b0, b1;
            simdgroup_load(a0, a_tile + simd_row * 32u + q, 32);
            simdgroup_load(a1, a_tile + (simd_row + 8u) * 32u + q, 32);
            simdgroup_load(a2, a_tile + (simd_row + 16u) * 32u + q, 32);
            simdgroup_load(a3, a_tile + (simd_row + 24u) * 32u + q, 32);
            simdgroup_load(b0, b_tile + simd_col * 32u + q, 32, ulong2(0), true);
            simdgroup_load(b1, b_tile + (simd_col + 8u) * 32u + q, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(acc0, a0, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a0, b1, acc1);
            simdgroup_multiply_accumulate(acc2, a1, b0, acc2);
            simdgroup_multiply_accumulate(acc3, a1, b1, acc3);
            simdgroup_multiply_accumulate(acc4, a2, b0, acc4);
            simdgroup_multiply_accumulate(acc5, a2, b1, acc5);
            simdgroup_multiply_accumulate(acc6, a3, b0, acc6);
            simdgroup_multiply_accumulate(acc7, a3, b1, acc7);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint quad = lane / 4u;
    uint fragment_row = (quad & 4u) + ((lane / 2u) % 4u);
    uint fragment_col = (quad & 2u) * 2u + (lane % 2u) * 2u;
    for (uint e = 0u; e < 2u; ++e) {
        uint r = row + simd_row + fragment_row, c = col + simd_col + fragment_col + e;
        if (r < m && c < n) y[r * n + c] = acc0.thread_elements()[e] + (use_bias != 0u ? bias[c] : 0.0f);
        if (r < m && c + 8u < n) y[r * n + c + 8u] = acc1.thread_elements()[e] + (use_bias != 0u ? bias[c + 8u] : 0.0f);
        if (r + 8u < m && c < n) y[(r + 8u) * n + c] = acc2.thread_elements()[e] + (use_bias != 0u ? bias[c] : 0.0f);
        if (r + 8u < m && c + 8u < n) y[(r + 8u) * n + c + 8u] = acc3.thread_elements()[e] + (use_bias != 0u ? bias[c + 8u] : 0.0f);
        if (r + 16u < m && c < n) y[(r + 16u) * n + c] = acc4.thread_elements()[e] + (use_bias != 0u ? bias[c] : 0.0f);
        if (r + 16u < m && c + 8u < n) y[(r + 16u) * n + c + 8u] = acc5.thread_elements()[e] + (use_bias != 0u ? bias[c + 8u] : 0.0f);
        if (r + 24u < m && c < n) y[(r + 24u) * n + c] = acc6.thread_elements()[e] + (use_bias != 0u ? bias[c] : 0.0f);
        if (r + 24u < m && c + 8u < n) y[(r + 24u) * n + c + 8u] = acc7.thread_elements()[e] + (use_bias != 0u ? bias[c + 8u] : 0.0f);
    }
}
