#include "golden.h"
#include <stdint.h>
#include <stdlib.h>

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp = (h >> 10) & 31u;
    uint32_t frac = h & 1023u;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) bits = sign;
        else {
            uint32_t shift = 0;
            while ((frac & 1024u) == 0) { frac <<= 1; shift++; }
            bits = sign | ((127u - 14u - shift) << 23) | ((frac & 1023u) << 13);
        }
    } else if (exp == 31u) bits = sign | 0x7f800000u | (frac << 13);
    else bits = sign | ((exp + 112u) << 23) | (frac << 13);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

G_K(g_q8_gemm_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const uint8_t* weight = PLOW_CPU_TEN(in, T, 2);
    const float* bias = PLOW_CPU_TEN(in, T, 3);
    const uint32_t m = in->i[0], n = in->i[1], k = in->i[2], activation = in->i[3];
    x += (size_t)in->i[4] * k;
    const uint32_t row_bytes = (k / 32u) * 34u;
    uint32_t lo, hi;
    g_range(n, slice, nblk, &lo, &hi);
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t column = lo; column < hi; column++) {
            const uint8_t* wr = weight + (size_t)column * row_bytes;
            float sum = bias ? bias[column] : 0.0f;
            for (uint32_t block = 0; block < k / 32u; block++) {
                uint16_t hs;
                memcpy(&hs, wr + block * 34u, sizeof(hs));
                const float scale = f16_to_f32(hs);
                const int8_t* q = (const int8_t*)(wr + block * 34u + 2u);
                float part = 0.0f;
                for (uint32_t i = 0; i < 32u; i++)
                    part += (float)q[i] * x[(size_t)row * k + block * 32u + i];
                sum += scale * part;
            }
            out[(size_t)row * n + column] = activation == 1u ? g_silu(sum) : sum;
        }
    }
}

G_K(g_layernorm_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const float* gamma = PLOW_CPU_TEN(in, T, 2);
    const float* beta = PLOW_CPU_TEN(in, T, 3);
    const uint32_t rows = in->i[0], feat = in->i[1];
    for (uint32_t row = slice; row < rows; row += nblk) {
        const float* xr = x + (size_t)row * feat;
        float* yr = out + (size_t)row * feat;
        float mean, inv;
        if (in->i[2] & 2u) {
            float sum = 0.0f, square_sum = 0.0f;
            for (uint32_t i = 0; i < feat; i++) sum += xr[i];
            mean = sum / feat;
            for (uint32_t i = 0; i < feat; i++) {
                const float v = xr[i] - mean;
                square_sum += v * v;
            }
            inv = 1.0f / sqrtf(square_sum / feat + in->fj[0].f);
        } else {
            double sum = 0.0;
            for (uint32_t i = 0; i < feat; i++) sum += xr[i];
            mean = (float)(sum / feat);
            double square_sum = 0.0;
            for (uint32_t i = 0; i < feat; i++) {
                const double v = xr[i] - mean;
                square_sum += v * v;
            }
            inv = 1.0f / sqrtf((float)(square_sum / feat) + in->fj[0].f);
        }
        for (uint32_t i = 0; i < feat; i++) {
            float value = (xr[i] - mean) * inv * (gamma ? gamma[i] : 1.0f) +
                          (beta ? beta[i] : 0.0f);
            yr[i] = in->i[2] & 1u ? plow_bf2f(plow_f2bf(value)) : value;
        }
    }
}

G_K(g_scaled_add_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* a = PLOW_CPU_TEN(in, T, 1);
    const float* b = PLOW_CPU_TEN(in, T, 2);
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) {
        float value = a[i] + in->fj[0].f * b[i];
        out[i] = in->i[1] & 1u ? plow_bf2f(plow_f2bf(value)) : value;
    }
}

G_K(g_glu_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t rows = in->i[0], width = in->i[1], count = rows * width;
    uint32_t lo, hi;
    g_range(count, slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) {
        const uint32_t row = i / width, column = i % width;
        out[i] = x[(size_t)row * 2u * width + column] *
                 g_sigmoid(x[(size_t)row * 2u * width + width + column]);
    }
}

G_K(g_silu_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) out[i] = g_silu(x[i]);
}

G_K(g_dense_gemm_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const void* weight = PLOW_CPU_TEN(in, T, 2);
    const float* bias = PLOW_CPU_TEN(in, T, 3);
    const uint32_t m = in->i[0], n = in->i[1], k = in->i[2], activation = in->i[3];
    const uint32_t weight_stride = in->i[5] ? in->i[5] : k;
    const uint32_t onehot = in->i[6];
    const uint32_t flags = in->i[7];
    x += (size_t)in->i[4] * k;
    uint32_t lo, hi;
    g_range(n, slice, nblk, &lo, &hi);
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t column = lo; column < hi; column++) {
            double sum = bias ? bias[column] : 0.0;
            for (uint32_t inner = 0; inner < k; inner++) {
                const size_t wi = (size_t)column * weight_stride + inner;
                const float w = flags & 4u ? plow_bf2f(((const plow_bf16*)weight)[wi])
                                           : ((const float*)weight)[wi];
                sum += (double)x[(size_t)row * k + inner] * w;
            }
            if (in->i[5] && onehot < weight_stride) {
                const size_t wi = (size_t)column * weight_stride + onehot;
                sum += flags & 4u ? plow_bf2f(((const plow_bf16*)weight)[wi])
                                  : ((const float*)weight)[wi];
            }
            float value = (float)sum;
            if (flags & 2u) {
                value = plow_bf2f(plow_f2bf(value));
                const float z = value * 0.7071067811865475f;
                const float t = 1.0f / (1.0f + 0.3275911f * fabsf(z));
                const float r = (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t -
                                    0.284496736f) * t + 0.254829592f) * t) * expf(-z * z);
                value = plow_bf2f(plow_f2bf(0.5f * value * (z < 0.0f ? r : 2.0f - r)));
            } else if (flags & 1u) {
                value = plow_bf2f(plow_f2bf(value));
            }
            out[(size_t)row * n + column] = activation == 1u ? fmaxf(value, 0.0f) : value;
        }
    }
}

G_K(g_gemm_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* x = PLOW_CPU_TEN(in, T, 1);
    const plow_bf16* weight = PLOW_CPU_TEN(in, T, 2);
    const uint32_t m = in->i[0], n = in->i[1], k = in->i[2];
    uint32_t lo, hi;
    g_range(n, slice, nblk, &lo, &hi);
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t column = lo; column < hi; column++) {
            float sum = 0.0f;
            for (uint32_t inner = 0; inner < k; inner++)
                sum += plow_bf2f(x[(size_t)row * k + inner]) *
                       plow_bf2f(weight[(size_t)column * k + inner]);
            out[(size_t)row * n + column] = sum;
        }
    }
}

G_K(g_conv2d_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const void* weight = PLOW_CPU_TEN(in, T, 2);
    const float* bias = PLOW_CPU_TEN(in, T, 3);
    const uint32_t frames = in->i[0], width = in->i[1];
    const uint32_t input_channels = in->i[2], output_channels = in->i[3];
    const uint32_t kernel = in->i[4], stride = in->i[5];
    const uint32_t pad_before = in->i[6], pad_after = in->i[7], flags = in->fj[1].u;
    const uint32_t batches = in->fj[2].u ? in->fj[2].u : 1u;
    if (!kernel || !stride || frames + pad_before + pad_after < kernel ||
        width + pad_before + pad_after < kernel) return;
    const uint32_t output_frames = (frames + pad_before + pad_after - kernel) / stride + 1u;
    const uint32_t output_width = (width + pad_before + pad_after - kernel) / stride + 1u;
    const uint32_t depthwise = flags & 1u, relu = flags & 2u;
    const uint32_t output_layout = (flags >> 2u) & 3u, input_layout = (flags >> 4u) & 3u;
    const uint32_t weight_f32 = flags & 64u, gelu_erf_bf16 = flags & 128u;
    if (output_layout > 2u || input_layout > 2u) return;
    const uint32_t stored_channels = depthwise ? 1u : input_channels;
    const uint64_t count64 = (uint64_t)batches * output_frames * output_width * output_channels;
    if (count64 > UINT32_MAX) return;
    uint32_t lo, hi;
    g_range((uint32_t)count64, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t output_channel = index % output_channels;
        const uint32_t output_x = (index / output_channels) % output_width;
        const uint32_t output_frame =
            (index / (output_channels * output_width)) % output_frames;
        const uint32_t batch = index / (output_channels * output_width * output_frames);
        const uint32_t channel_start = depthwise ? output_channel : 0u;
        const uint32_t channel_end = depthwise ? output_channel + 1u : input_channels;
        double sum = bias[output_channel];
        for (uint32_t ky = 0; ky < kernel; ky++) {
            const int64_t input_frame = (int64_t)output_frame * stride + ky - pad_before;
            if (input_frame < 0 || input_frame >= frames) continue;
            for (uint32_t kx = 0; kx < kernel; kx++) {
                const int64_t input_x = (int64_t)output_x * stride + kx - pad_before;
                if (input_x < 0 || input_x >= width) continue;
                for (uint32_t input_channel = channel_start; input_channel < channel_end;
                     input_channel++) {
                    const uint32_t weight_channel = depthwise ? 0u : input_channel;
                    const size_t wi = (((size_t)output_channel * stored_channels + weight_channel) *
                                       kernel + ky) * kernel + kx;
                    size_t xi;
                    if (input_layout == 0u)
                        xi = (((size_t)batch * frames + input_frame) * width + input_x) *
                                 input_channels + input_channel;
                    else if (input_layout == 1u)
                        xi = (((size_t)batch * frames + input_frame) * input_channels +
                              input_channel) * width + input_x;
                    else
                        xi = (((size_t)batch * input_channels + input_channel) * frames +
                              input_frame) * width + input_x;
                    const float w = weight_f32 ? ((const float*)weight)[wi]
                                               : f16_to_f32(((const uint16_t*)weight)[wi]);
                    sum += (double)x[xi] * w;
                }
            }
        }
        float value = (float)sum;
        if (gelu_erf_bf16) {
            value = plow_bf2f(plow_f2bf(value));
            const float z = value * 0.7071067811865475f;
            const float t = 1.0f / (1.0f + 0.3275911f * fabsf(z));
            const float r = (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t -
                                0.284496736f) * t + 0.254829592f) * t) * expf(-z * z);
            value = plow_bf2f(plow_f2bf(0.5f * value * (z < 0.0f ? r : 2.0f - r)));
        }
        if (relu) value = fmaxf(value, 0.0f);
        size_t oi;
        if (output_layout == 0u)
            oi = (((size_t)batch * output_frames + output_frame) * output_width + output_x) *
                     output_channels + output_channel;
        else if (output_layout == 1u)
            oi = (((size_t)batch * output_frames + output_frame) * output_channels +
                  output_channel) * output_width + output_x;
        else
            oi = (((size_t)batch * output_channels + output_channel) * output_frames +
                  output_frame) * output_width + output_x;
        out[oi] = value;
    }
}

G_K(g_pack_ncfw_rows_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t rows = in->i[0], channels = in->i[1];
    const uint32_t frames = in->i[2], width = in->i[3], batches = in->i[4];
    const uint64_t count64 = (uint64_t)rows * channels * frames;
    const uint64_t capacity = (uint64_t)batches * width;
    if (!rows || !channels || !frames || !width || !batches || rows > capacity ||
        count64 > UINT32_MAX) return;
    uint32_t lo, hi;
    g_range((uint32_t)count64, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t row_width = channels * frames;
        const uint32_t row = index / row_width, column = index % row_width;
        const uint32_t batch = row / width, position = row % width;
        const uint32_t channel = column / frames, frame = column % frames;
        out[index] = x[((size_t)batch * channels + channel) * frames * width +
                       (size_t)frame * width + position];
    }
}

G_K(g_embed_f16_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const uint16_t* table = PLOW_CPU_TEN(in, T, 1);
    const uint32_t token = *(const uint32_t*)PLOW_CPU_TEN(in, T, 2);
    const uint32_t vocab = in->i[0], width = in->i[1];
    if (token >= vocab) return;
    uint32_t lo, hi;
    g_range(width, slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) out[i] = f16_to_f32(table[(size_t)token * width + i]);
}

G_K(g_embed_overlay_bf16) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* table = PLOW_CPU_TEN(in, T, 1);
    const uint32_t* tokens = PLOW_CPU_TEN(in, T, 2);
    const float* overlay = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* overlay_index = PLOW_CPU_TEN(in, T, 4);
    const uint32_t rows = in->i[0], width = in->i[1], vocab = in->i[2];
    const uint32_t overlay_rows = in->i[3];
    const uint64_t count64 = (uint64_t)rows * width;
    if (!rows || !width || !vocab || count64 > UINT32_MAX) return;
    uint32_t lo, hi;
    g_range((uint32_t)count64, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t row = index / width, column = index % width;
        const uint32_t selected = overlay_index[row];
        if (selected == UINT32_MAX) {
            if (tokens[row] >= vocab) return;
            out[index] = table[(size_t)tokens[row] * width + column];
        } else {
            if (selected >= overlay_rows) return;
            out[index] = plow_f2bf(overlay[(size_t)selected * width + column]);
        }
    }
}

G_K(g_embed_pos_bf16) {
    (void)ctx;
    plow_bf16* out = PLOW_CPU_TEN(in, T, 0);
    const plow_bf16* table = PLOW_CPU_TEN(in, T, 1);
    const uint32_t* tokens = PLOW_CPU_TEN(in, T, 2);
    const plow_bf16* pos_table = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* pos = PLOW_CPU_TEN(in, T, 4);
    const uint32_t* base = PLOW_CPU_TEN(in, T, 5);
    const uint32_t rows = in->i[0], width = in->i[1], vocab = in->i[2], pos_rows = in->i[3];
    const uint64_t count64 = (uint64_t)rows * width;
    if (!rows || !width || !vocab || !pos_rows || count64 > UINT32_MAX) return;
    uint32_t lo, hi;
    g_range((uint32_t)count64, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t row = index / width, column = index % width;
        const uint32_t p = pos[row] - base[row];
        if (tokens[row] >= vocab || pos[row] < base[row] || p >= pos_rows) return;
        out[index] = plow_f2bf(plow_bf2f(table[(size_t)tokens[row] * width + column]) +
                               plow_bf2f(pos_table[(size_t)p * width + column]));
    }
}

G_K(g_lstm_cell_f32) {
    (void)ctx;
    float* h_new = PLOW_CPU_TEN(in, T, 0);
    float* c_new = PLOW_CPU_TEN(in, T, 1);
    const float* gates = PLOW_CPU_TEN(in, T, 2);
    const float* c_prev = PLOW_CPU_TEN(in, T, 3);
    const uint32_t width = in->i[0];
    uint32_t lo, hi;
    g_range(width, slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) {
        const float input = g_sigmoid(gates[i]);
        const float forget = g_sigmoid(gates[width + i]);
        const float cell = tanhf(gates[2u * width + i]);
        const float output = g_sigmoid(gates[3u * width + i]);
        c_new[i] = forget * c_prev[i] + input * cell;
        h_new[i] = output * tanhf(c_new[i]);
    }
}

G_K(g_argmax_f32) {
    (void)ctx;
    uint32_t* ids = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t rows = in->i[0], width = in->i[1];
    for (uint32_t row = slice; row < rows; row += nblk) {
        uint32_t best = 0;
        float maximum = x[(size_t)row * width];
        for (uint32_t i = 1; i < width; i++) {
            const float value = x[(size_t)row * width + i];
            if (value > maximum) { maximum = value; best = i; }
        }
        ids[row] = best;
    }
}

G_K(g_relu_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    uint32_t lo, hi;
    g_range(in->i[0], slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) out[i] = fmaxf(x[i], 0.0f);
}

G_K(g_broadcast_add_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* matrix = PLOW_CPU_TEN(in, T, 1);
    const float* vector = PLOW_CPU_TEN(in, T, 2);
    const uint32_t rows = in->i[0], width = in->i[1];
    uint32_t lo, hi;
    g_range(rows * width, slice, nblk, &lo, &hi);
    for (uint32_t i = lo; i < hi; i++) out[i] = matrix[i] + vector[i % width];
}

G_K(g_causal_depthwise_conv1d_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const uint16_t* weight = PLOW_CPU_TEN(in, T, 2);
    const uint32_t rows = in->i[0], channels = in->i[1], kernel = in->i[2];
    uint32_t lo, hi;
    g_range(rows * channels, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t row = index / channels, channel = index % channels;
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < kernel; tap++) {
            const int64_t source = (int64_t)row + tap + 1 - kernel;
            if (source >= 0) sum += x[(size_t)source * channels + channel] *
                                   f16_to_f32(weight[(size_t)channel * kernel + tap]);
        }
        out[index] = sum;
    }
}

static float attention_score(const float* q, const float* k, const float* p,
                             const float* u, const float* v, uint32_t width) {
    double content = 0.0, relative = 0.0;
    for (uint32_t i = 0; i < width; i++) {
        content += (double)k[i] * (q[i] + u[i]);
        relative += (double)p[i] * (q[i] + v[i]);
    }
    return (float)((content + relative) / sqrt((double)width));
}

G_K(g_relative_attention_f32) {
    (void)ctx;
    float* context = PLOW_CPU_TEN(in, T, 0);
    const float* query = PLOW_CPU_TEN(in, T, 1);
    const float* key = PLOW_CPU_TEN(in, T, 2);
    const float* value = PLOW_CPU_TEN(in, T, 3);
    const float* position = PLOW_CPU_TEN(in, T, 4);
    const float* bias_u = PLOW_CPU_TEN(in, T, 5);
    const float* bias_v = PLOW_CPU_TEN(in, T, 6);
    const uint32_t rows = in->i[0], width = in->i[1], heads = in->i[2];
    const uint32_t chunk = in->i[3], left_chunks = in->i[4], head_width = width / heads;
    for (uint32_t item = slice; item < rows * heads; item += nblk) {
        const uint32_t qr = item / heads, head = item % heads;
        uint32_t first = 0, last = rows;
        if (left_chunks != UINT32_MAX) {
            const uint32_t qc = qr / chunk;
            first = (qc > left_chunks ? qc - left_chunks : 0u) * chunk;
            last = (qc + 1u) * chunk; if (last > rows) last = rows;
        }
        float scores[rows];
        float maximum = -INFINITY;
        const float* q = query + (size_t)qr * width + head * head_width;
        for (uint32_t kr = first; kr < last; kr++) {
            const uint32_t pr = rows - 1u + kr - qr;
            scores[kr] = attention_score(q, key + (size_t)kr * width + head * head_width,
                                         position + (size_t)pr * width + head * head_width,
                                         bias_u + head * head_width, bias_v + head * head_width,
                                         head_width);
            maximum = fmaxf(maximum, scores[kr]);
        }
        float denominator = 0.0f;
        for (uint32_t kr = first; kr < last; kr++) denominator += expf(scores[kr] - maximum);
        for (uint32_t col = 0; col < head_width; col++) {
            float sum = 0.0f;
            for (uint32_t kr = first; kr < last; kr++)
                sum += expf(scores[kr] - maximum) / denominator *
                       value[(size_t)kr * width + head * head_width + col];
            context[(size_t)qr * width + head * head_width + col] = sum;
        }
    }
}

G_K(g_grouped_attention_f32) {
    (void)ctx;
    float* context = PLOW_CPU_TEN(in, T, 0);
    const float* query = PLOW_CPU_TEN(in, T, 1);
    const float* key = PLOW_CPU_TEN(in, T, 2);
    const float* value = PLOW_CPU_TEN(in, T, 3);
    const uint32_t rows = in->i[0], width = in->i[1], head_width = in->i[2];
    const uint32_t group_rows = in->i[3], flags = in->i[4];
    const uint32_t valid_rows = in->t[4] == PLOW_TENSOR_NONE
        ? rows : *(const uint32_t*)T[in->t[4]];
    if (head_width == 0 || width % head_width != 0 || group_rows == 0 ||
        group_rows > 256 || valid_rows == 0 || valid_rows > rows)
        return;
    const uint32_t heads = width / head_width;
    for (uint32_t item = slice; item < valid_rows * heads; item += nblk) {
        const uint32_t row = item / heads, head = item % heads;
        const uint32_t first = row / group_rows * group_rows;
        uint32_t last = first + group_rows;
        if (last > valid_rows) last = valid_rows;
        float scores[256], maximum = -INFINITY;
        const size_t qb = (size_t)row * width + head * head_width;
        for (uint32_t kr = first; kr < last; kr++) {
            const size_t kb = (size_t)kr * width + head * head_width;
            float score = 0.0f;
            for (uint32_t col = 0; col < head_width; col++)
                score += query[qb + col] * key[kb + col];
            if (flags & 1u) score = plow_bf2f(plow_f2bf(score));
            score /= sqrtf((float)head_width);
            scores[kr - first] = score;
            maximum = fmaxf(maximum, score);
        }
        float denominator = 0.0f;
        for (uint32_t kr = first; kr < last; kr++) {
            scores[kr - first] = expf(scores[kr - first] - maximum);
            denominator += scores[kr - first];
        }
        for (uint32_t kr = first; kr < last; kr++) {
            float probability = scores[kr - first] / denominator;
            scores[kr - first] = flags & 2u ? plow_bf2f(plow_f2bf(probability)) : probability;
        }
        for (uint32_t col = 0; col < head_width; col++) {
            float sum = 0.0f;
            for (uint32_t kr = first; kr < last; kr++)
                sum += scores[kr - first] *
                       value[(size_t)kr * width + head * head_width + col];
            context[qb + col] = flags & 4u ? plow_bf2f(plow_f2bf(sum)) : sum;
        }
    }
}

/* ---- generic FP32 signal ops (195-203). Activation codes: packet::dev::ACT_*. ---- */

static float g_act_f32(uint32_t kind, float x, float p0, float p1) {
    switch (kind) {
    case 1: return tanhf(x);
    case 2: return sinf(x);
    case 3: return cosf(x);
    case 4: return expf(x);
    case 5: return fabsf(x);
    case 6: return 1.0f / (1.0f + expf(-x));
    case 7: return x / (1.0f + expf(-x));
    case 8: return x > 0.0f ? x : expm1f(x);
    case 9: return x >= 0.0f ? x : x * p0;
    case 10: return x * tanhf(log1pf(expf(x)));
    case 11: return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    case 12: {
        const float s = sinf(p0 * x);
        return x + 1.0f / (p0 + 1e-9f) * (s * s);
    }
    case 13: return fminf(fmaxf(x, p0), p1);
    case 14: return x * p0 + p1;
    case 15: return x > 0.0f ? x : 0.0f;
    default: return x;
    }
}

static int g_act_valid_conv(uint32_t kind) { return kind != 13u && kind != 14u; }

static float g_weight(const void* weight, int f16, size_t i) {
    return f16 ? f16_to_f32(((const uint16_t*)weight)[i]) : ((const float*)weight)[i];
}

G_K(g_gather_rows_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const void* table = PLOW_CPU_TEN(in, T, 1);
    const uint32_t* index = PLOW_CPU_TEN(in, T, 2);
    const uint32_t rows = in->i[0], width = in->i[1], vocab = in->i[2];
    const uint32_t per_item = in->i[3] ? in->i[3] : rows, repeat = in->i[4] ? in->i[4] : 1u;
    const uint32_t flags = in->i[7];
    const uint32_t out_stride = in->fj[1].u ? in->fj[1].u : width, out_col0 = in->fj[2].u;
    uint32_t lo, hi;
    g_range(rows, slice, nblk, &lo, &hi);
    for (uint32_t row = lo; row < hi; row++) {
        const uint32_t item = row / per_item, local = row % per_item / repeat;
        const uint32_t source = index ? index[(size_t)item * in->i[5] + local] : local;
        const size_t table_row = (size_t)item * in->i[6] + source;
        for (uint32_t column = 0; column < width; column++) {
            const float value = source < vocab
                ? g_weight(table, flags & 1u, table_row * width + column) : 0.0f;
            float* o = out + (size_t)row * out_stride + out_col0 + column;
            *o = flags & 2u ? *o + value : value;
        }
    }
}

G_K(g_copy_cols_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const uint32_t items = in->i[0], rows = in->i[1], cols = in->i[2];
    uint32_t lo, hi;
    g_range(items * rows, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t item = index / rows, row = index % rows;
        const float* src = x + (size_t)item * in->fj[1].u + (size_t)row * in->i[3] + in->i[4];
        float* dst = out + (size_t)item * in->fj[2].u + (size_t)row * in->i[5] + in->i[6];
        for (uint32_t column = 0; column < cols; column++) dst[column] = src[column];
    }
}

/* Input row `u` of an item with `length` valid rows under pad `mode`; -1 reads as zero. */
static int64_t g_pad_row(int64_t u, int64_t length, uint32_t mode) {
    if (u >= 0 && u < length) return u;
    if (mode == 1u) u = u < 0 ? -u : 2 * (length - 1) - u;
    else if (mode == 2u) u = u < 0 ? 0 : length - 1;
    else return -1;
    return u >= 0 && u < length ? u : -1;
}

static uint32_t g_item_rows(const uint32_t* lengths, uint32_t item, uint32_t rows) {
    return lengths && lengths[item] < rows ? lengths[item] : rows;
}

G_K(g_conv1d_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const void* weight = PLOW_CPU_TEN(in, T, 2);
    const float* bias = PLOW_CPU_TEN(in, T, 3);
    const float* alpha = PLOW_CPU_TEN(in, T, 4);
    const float* residual = PLOW_CPU_TEN(in, T, 5);
    const uint32_t* lengths = PLOW_CPU_TEN(in, T, 6);
    const uint32_t batch = in->i[0], in_rows = in->i[1], cin = in->i[2], cout = in->i[3];
    const uint32_t kernel = in->i[4], stride = in->i[5], dilation = in->i[6], groups = in->i[7];
    const uint32_t pad_before = in->fj[1].u & 0xFFFFu, pad_after = in->fj[1].u >> 16;
    const uint32_t flags = in->fj[2].u, mode = flags & 3u;
    const uint32_t pre = (flags >> 4) & 15u, post = (flags >> 8) & 15u;
    const float slope = in->fj[0].f;
    if (!kernel || !stride || !dilation || !groups || cin % groups || cout % groups ||
        mode > 2u || !g_act_valid_conv(pre) || !g_act_valid_conv(post) || post == 12u)
        return;
    const uint64_t span = (uint64_t)dilation * (kernel - 1u) + 1u;
    if (in_rows + pad_before + pad_after < span) return;
    const uint32_t out_rows = (uint32_t)((in_rows + pad_before + pad_after - span) / stride + 1u);
    const uint32_t cg = cin / groups, ng = cout / groups;
    uint32_t lo, hi;
    g_range(batch * out_rows, slice, nblk, &lo, &hi);
    for (uint32_t m = lo; m < hi; m++) {
        const uint32_t b = m / out_rows, t = m % out_rows;
        const uint32_t length = g_item_rows(lengths, b, in_rows);
        const uint64_t padded = (uint64_t)length + pad_before + pad_after;
        const uint64_t out_length = padded >= span ? (padded - span) / stride + 1u : 0u;
        for (uint32_t o = 0; o < cout; o++) {
            const size_t oi = (size_t)m * cout + o;
            if (t >= out_length) { out[oi] = 0.0f; continue; }
            const uint32_t c0 = o / ng * cg;
            double sum = bias ? bias[o] : 0.0;
            for (uint32_t k = 0; k < kernel; k++) {
                const int64_t u = g_pad_row((int64_t)t * stride + (int64_t)k * dilation - pad_before,
                                            length, mode);
                if (u < 0) continue;
                for (uint32_t i = 0; i < cg; i++) {
                    const uint32_t c = c0 + i;
                    const float v = g_act_f32(pre, x[((size_t)b * in_rows + u) * cin + c],
                                              alpha ? alpha[c] : slope, 0.0f);
                    sum += (double)v * g_weight(weight, (flags >> 12) & 1u,
                                                ((size_t)o * cg + i) * kernel + k);
                }
            }
            float value = g_act_f32(post, (float)sum, alpha ? alpha[o] : slope, 0.0f);
            out[oi] = residual ? value + residual[oi] : value;
        }
    }
}

G_K(g_conv_transpose1d_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const void* weight = PLOW_CPU_TEN(in, T, 2);
    const float* bias = PLOW_CPU_TEN(in, T, 3);
    const float* alpha = PLOW_CPU_TEN(in, T, 4);
    const float* residual = PLOW_CPU_TEN(in, T, 5);
    const uint32_t* lengths = PLOW_CPU_TEN(in, T, 6);
    const uint32_t batch = in->i[0], in_rows = in->i[1], cin = in->i[2], cout = in->i[3];
    const uint32_t kernel = in->i[4], stride = in->i[5], output_padding = in->i[6];
    const uint32_t groups = in->i[7];
    const uint32_t crop_before = in->fj[1].u & 0xFFFFu, crop_after = in->fj[1].u >> 16;
    const uint32_t flags = in->fj[2].u;
    const uint32_t pre = (flags >> 4) & 15u, post = (flags >> 8) & 15u;
    const float slope = in->fj[0].f;
    if (!in_rows || !kernel || !stride || !groups || cin % groups || cout % groups ||
        !g_act_valid_conv(pre) || !g_act_valid_conv(post) || post == 12u)
        return;
    const int64_t full = (int64_t)(in_rows - 1u) * stride + kernel + output_padding;
    if (full <= (int64_t)crop_before + crop_after) return;
    const uint32_t out_rows = (uint32_t)(full - crop_before - crop_after);
    const uint32_t cg = cin / groups, ng = cout / groups;
    uint32_t lo, hi;
    g_range(batch * out_rows, slice, nblk, &lo, &hi);
    for (uint32_t m = lo; m < hi; m++) {
        const uint32_t b = m / out_rows, t = m % out_rows;
        const uint32_t length = g_item_rows(lengths, b, in_rows);
        const int64_t out_length = length
            ? (int64_t)(length - 1u) * stride + kernel + output_padding - crop_before - crop_after
            : 0;
        for (uint32_t o = 0; o < cout; o++) {
            const size_t oi = (size_t)m * cout + o;
            if ((int64_t)t >= out_length) { out[oi] = 0.0f; continue; }
            const uint32_t c0 = o / ng * cg, on = o % ng;
            double sum = bias ? bias[o] : 0.0;
            for (uint32_t k = 0; k < kernel; k++) {
                const int64_t num = (int64_t)t + crop_before - k;
                if (num < 0 || num % stride) continue;
                const int64_t s = num / stride;
                if (s >= length) continue;
                for (uint32_t i = 0; i < cg; i++) {
                    const uint32_t c = c0 + i;
                    const float v = g_act_f32(pre, x[((size_t)b * in_rows + s) * cin + c],
                                              alpha ? alpha[c] : slope, 0.0f);
                    sum += (double)v * g_weight(weight, (flags >> 12) & 1u,
                                                ((size_t)c * ng + on) * kernel + k);
                }
            }
            float value = g_act_f32(post, (float)sum, alpha ? alpha[o] : slope, 0.0f);
            out[oi] = residual ? value + residual[oi] : value;
        }
    }
}

G_K(g_unary_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const float* param = PLOW_CPU_TEN(in, T, 2);
    const uint32_t rows = in->i[0], width = in->i[1], kind = in->i[2];
    const uint32_t stride = in->i[3] ? in->i[3] : width, col0 = in->i[4];
    uint32_t lo, hi;
    g_range(rows, slice, nblk, &lo, &hi);
    for (uint32_t row = lo; row < hi; row++)
        for (uint32_t column = 0; column < width; column++) {
            const size_t i = (size_t)row * stride + col0 + column;
            out[i] = g_act_f32(kind, x[i], param ? param[column] : in->fj[0].f, in->fj[1].f);
        }
}

G_K(g_binary_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* a = PLOW_CPU_TEN(in, T, 1);
    const float* b = PLOW_CPU_TEN(in, T, 2);
    const uint32_t items = in->i[0], rows = in->i[1], width = in->i[2], op = in->i[3];
    if (op > 5u) return;
    uint32_t lo, hi;
    g_range(items * rows, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t item = index / rows, row = index % rows;
        for (uint32_t column = 0; column < width; column++) {
            const size_t i = (size_t)index * width + column;
            const float av = a[i];
            const float bv = b[(size_t)item * in->i[4] + (size_t)row * in->i[5] +
                               (size_t)column * in->i[6]];
            float v = op == 0u ? av + bv : op == 1u ? av - bv : op == 2u ? av * bv
                    : op == 3u ? av / bv : op == 4u ? fmaxf(av, bv) : fminf(av, bv);
            out[i] = in->i[7] & 1u ? in->fj[0].f * v : v;
        }
    }
}

G_K(g_cumsum_f64) {
    (void)ctx;
    void* out = PLOW_CPU_TEN(in, T, 0);
    const float* x = PLOW_CPU_TEN(in, T, 1);
    const float* column_scale = PLOW_CPU_TEN(in, T, 2);
    const uint32_t* lengths = PLOW_CPU_TEN(in, T, 3);
    const uint32_t items = in->i[0], rows = in->i[1], width = in->i[2];
    const uint32_t x_width = in->i[3] ? in->i[3] : width, flags = in->i[4];
    uint32_t lo, hi;
    g_range(items * width, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t item = index / width, column = index % width;
        const uint32_t length = g_item_rows(lengths, item, rows);
        const double scale = (double)(column_scale ? column_scale[column] : 1.0f) * in->fj[0].f;
        double sum = 0.0;
        for (uint32_t row = 0; row < rows; row++) {
            const double value = row < length
                ? (double)x[((size_t)item * rows + row) * x_width + column % x_width] : 0.0;
            if (!(flags & 1u)) sum += value;
            double v = sum * scale;
            if (flags & 2u) v -= floor(v);
            v *= in->fj[1].f;
            const size_t o = ((size_t)item * rows + row) * width + column;
            if (flags & 4u) ((double*)out)[o] = v;
            else ((float*)out)[o] = (float)v;
            if (flags & 1u) sum += value;
        }
    }
}

static uint64_t g_mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

G_K(g_rand_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const uint64_t* seed = PLOW_CPU_TEN(in, T, 1);
    const uint32_t items = in->i[0], rows = in->i[1], width = in->i[2];
    const uint32_t stream = in->i[3], shift = in->i[4], coords = in->i[5];
    const uint32_t flags = in->fj[2].u;
    if (shift > 63u || (coords & 3u) > 2u || ((coords >> 2) & 3u) > 2u) return;
    uint32_t lo, hi;
    g_range(items * rows, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t item = index / rows, row = index % rows;
        const uint64_t s = seed[flags & 2u ? 0u : item];
        for (uint32_t column = 0; column < width; column++) {
            const uint32_t coord[3] = {row, column, item};
            const uint64_t a = (uint32_t)(coord[coords & 3u] + in->i[6]);
            const uint64_t b = (uint32_t)(coord[(coords >> 2) & 3u] + in->i[7]);
            const uint64_t h = g_mix64(s ^ g_mix64(((uint64_t)stream << shift) ^ (a << 32) ^ b));
            float v;
            if (flags & 1u) {
                const double u1 = (double)((h >> 40) + 1u) / 16777216.0;
                const double u2 = (double)(g_mix64(h) >> 40) / 16777216.0;
                v = (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
            } else {
                v = (float)(h >> 40) * (1.0f / 16777216.0f);
            }
            out[(size_t)index * width + column] = (v + in->fj[1].f) * in->fj[0].f;
        }
    }
}

G_K(g_attention_f32) {
    (void)ctx;
    float* out = PLOW_CPU_TEN(in, T, 0);
    const float* query = PLOW_CPU_TEN(in, T, 1);
    const float* key = PLOW_CPU_TEN(in, T, 2);
    const float* value = PLOW_CPU_TEN(in, T, 3);
    const uint32_t* key_lengths = PLOW_CPU_TEN(in, T, 4);
    const float* bias = PLOW_CPU_TEN(in, T, 5);
    const uint32_t batch = in->i[0], q_rows = in->i[1], kv_rows = in->i[2], heads = in->i[3];
    const uint32_t hw = in->i[4], width = heads * hw;
    const uint32_t stride = in->i[5] ? in->i[5] : width, causal = in->i[6] & 1u;
    const uint32_t k_col0 = in->fj[1].u, v_col0 = in->fj[2].u;
    if (!hw || !kv_rows) return;
    double* p = malloc(sizeof(double) * kv_rows);
    double* acc = malloc(sizeof(double) * hw);
    uint32_t lo, hi;
    g_range(batch * heads * q_rows, slice, nblk, &lo, &hi);
    for (uint32_t index = lo; index < hi; index++) {
        const uint32_t b = index / (heads * q_rows), h = index / q_rows % heads, r = index % q_rows;
        uint32_t visible = g_item_rows(key_lengths, b, kv_rows);
        if (causal && r + 1u < visible) visible = r + 1u;
        const float* q = query + ((size_t)b * q_rows + r) * stride + (size_t)h * hw;
        double maximum = -INFINITY;
        for (uint32_t j = 0; j < visible; j++) {
            const float* k = key + ((size_t)b * kv_rows + j) * stride + k_col0 + (size_t)h * hw;
            double dot = 0.0;
            for (uint32_t c = 0; c < hw; c++) dot += (double)q[c] * k[c];
            p[j] = dot * in->fj[0].f +
                   (bias ? bias[(size_t)h * in->i[7] + (size_t)r * kv_rows + j] : 0.0f);
            if (p[j] > maximum) maximum = p[j];
        }
        double denominator = 0.0;
        for (uint32_t c = 0; c < hw; c++) acc[c] = 0.0;
        for (uint32_t j = 0; j < visible; j++) {
            const double e = exp(p[j] - maximum);
            const float* v = value + ((size_t)b * kv_rows + j) * stride + v_col0 + (size_t)h * hw;
            denominator += e;
            for (uint32_t c = 0; c < hw; c++) acc[c] += e * v[c];
        }
        float* o = out + ((size_t)b * q_rows + r) * width + (size_t)h * hw;
        for (uint32_t c = 0; c < hw; c++) o[c] = visible ? (float)(acc[c] / denominator) : 0.0f;
    }
    free(p);
    free(acc);
}
