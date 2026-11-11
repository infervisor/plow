#pragma once

static __device__ __forceinline__ bool sp_is_ring_attention(const PlowDevInst* in) {
    return (in->i[6] & 0xff08u) == 0x108u;
}

static __device__ __forceinline__ const float* sp_ring_key(
    const float* qkv, const float* state, unsigned b, unsigned slot, unsigned rows,
    unsigned history, unsigned stride, unsigned width, unsigned head, unsigned pos,
    unsigned ring0, unsigned long long absolute, bool value) {
    if (absolute < pos) {
        unsigned index = ring0 + (unsigned)(absolute + history - pos);
        if (index >= history) index -= history;
        return state + ((size_t)slot * history + index) * (2u * width) +
               head * 64u + (value ? width : 0u);
    }
    return qkv + ((size_t)b * rows + (absolute - pos)) * stride +
           head * 64u + (value ? 2u * width : width);
}

static __device__ __noinline__ void d_attention_ring_f32(
    const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
    const unsigned B = in->i[0], N = in->i[1], L = in->i[2], H = in->i[3];
    const unsigned stride = in->i[5], W = in->i[6] >> 16, slots = in->i[7];
    if (!B || !N || N > 4u || !L || L > 255u || !H || in->i[4] != 64u ||
        (uint64_t)B * N > 0xffffffffull || (uint64_t)B * N * H > 0xffffffffull ||
        stride % 4u || (uint64_t)H * 192u > stride ||
        in->fj[1].u != H * 64u || in->fj[2].u != H * 128u || W != L + 1u ||
        (in->i[6] & 0xffffu) != 0x109u || !slots) { __trap(); return; }
    const auto* qkv = (const float*)SP_TEN(1);
    const auto* state = (const float*)SP_TEN(2);
    const auto* slot = (const unsigned*)SP_TEN(3);
    const auto* lengths = (const unsigned*)SP_TEN(4);
    const auto* position = (const unsigned*)SP_TEN(5);
    auto* out = (float*)SP_TEN(0);
    if ((((size_t)qkv | (size_t)state | (size_t)out) & 15u) != 0u) { __trap(); return; }
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned width = H * 64u;
    float* query = sp_smem;
    float* scores = query + 64;
    float* part = scores + 256;
    auto* partial = (float2*)(part + PLOW_NV_WARPS);
    for (uint64_t work = slice; work < (uint64_t)B * N * H; work += nblk) {
        const unsigned item = (unsigned)work;
        const unsigned h = item % H, r = item / H % N, b = item / (H * N);
        const unsigned n = lengths[b];
        if (n > N) { __trap(); return; }
        const size_t output = ((size_t)b * N + r) * width + h * 64u;
        if (r >= n) {
            if (tid < 32u) ((float2*)(out + output))[tid] = make_float2(0.f, 0.f);
            continue;
        }
        const unsigned s = slot[b], pos = position[b], ring0 = pos % L;
        if (s >= slots) { __trap(); return; }
        const unsigned long long end = (unsigned long long)pos + r + 1u;
        const unsigned long long first = end > W ? end - W : 0u;
        const unsigned count = (unsigned)(end - first);
        if (tid < 64u) query[tid] = qkv[((size_t)b * N + r) * stride + h * 64u + tid];
        scores[tid] = 0.f;
        __syncthreads();
        const unsigned group = tid >> 2, component = (tid & 3u) * 4u;
        const unsigned mask = 15u << (lane & ~3u);
        for (unsigned j = group; j < count; j += PLOW_NV_THREADS / 4u) {
            const float* key = sp_ring_key(qkv, state, b, s, N, L, stride, width, h,
                                            pos, ring0, first + j, false);
            float4 values[4];
#pragma unroll
            for (unsigned k = 0; k < 4; k++) values[k] = *(const float4*)(key + component + k * 16u);
            float dot = 0.f;
#pragma unroll
            for (unsigned k = 0; k < 4; k++) {
                const float4 q = *(const float4*)(query + component + k * 16u);
                dot = __fmaf_rn(q.x, values[k].x, dot);
                dot = __fmaf_rn(q.y, values[k].y, dot);
                dot = __fmaf_rn(q.z, values[k].z, dot);
                dot = __fmaf_rn(q.w, values[k].w, dot);
            }
            dot += __shfl_down_sync(mask, dot, 2, 4);
            dot += __shfl_down_sync(mask, dot, 1, 4);
            if ((tid & 3u) == 0) scores[j] = dot * in->fj[0].f;
        }
        __syncthreads();
        const float maximum = block_max(tid < count ? scores[tid] : -INFINITY, part);
        const float probability = tid < count ? expf(scores[tid] - maximum) : 0.f;
        const float sum = block_sum(probability, part);
        scores[tid] = sum > 0.f ? probability / sum : 0.f;
        __syncthreads();
        // Keep the global loads ahead of the dependent weighted-sum chain.
        float2 values[32];
#pragma unroll
        for (unsigned k = 0; k < 32; k++) {
            const unsigned j = warp + k * PLOW_NV_WARPS;
            values[k] = j < count ? *(const float2*)(sp_ring_key(qkv, state, b, s, N, L,
                stride, width, h, pos, ring0, first + j, true) + lane * 2u) : make_float2(0.f, 0.f);
        }
        float2 value = make_float2(0.f, 0.f);
#pragma unroll
        for (unsigned k = 0; k < 32; k++) {
            const float p = scores[warp + k * PLOW_NV_WARPS];
            value.x = __fmaf_rn(p, values[k].x, value.x);
            value.y = __fmaf_rn(p, values[k].y, value.y);
        }
        partial[tid] = value;
        __syncthreads();
        if (tid < 32u) {
            float2 value = partial[tid];
#pragma unroll
            for (unsigned w = 1; w < PLOW_NV_WARPS; w++) {
                const float2 other = partial[w * 32u + tid];
                value.x += other.x;
                value.y += other.y;
            }
            ((float2*)(out + output))[tid] = value;
        }
        __syncthreads();
    }
}

static __device__ __noinline__ void d_stream_ring_f32(
    const PlowDevInst* in, void* const* T, unsigned slice, unsigned nblk) {
    const unsigned B = in->i[0], L = in->i[1], N = in->i[2], C = in->i[3];
    const unsigned Cz = in->i[4], offset = in->i[5], stride = in->i[6], mode = in->i[7];
    const unsigned slots = in->fj[1].u;
    if (!L || !N || !C || C % 4u || Cz % 4u || offset % 4u || stride % 4u || stride < Cz ||
        (uint64_t)offset + C > Cz || (mode != 2u && mode != 3u) || !slots ||
        (uint64_t)L + N > 0xffffffffull || ((uint64_t)L + N) * Cz > 0xffffffffull ||
        (uint64_t)B * ((uint64_t)L + N) * Cz > 0xffffffffull) { __trap(); return; }
    const auto* src = (const float*)SP_TEN(1);
    auto* state = (float*)SP_TEN(2);
    const auto* slot = (const unsigned*)SP_TEN(3);
    const auto* length = (const unsigned*)SP_TEN(4);
    const auto* position = (const unsigned*)SP_TEN(5);
    if ((((size_t)src | (size_t)state | (mode == 3u ? (size_t)SP_TEN(0) : 0u)) & 15u) != 0u) {
        __trap(); return;
    }
    if (mode == 3u) {
        auto* z = (float4*)SP_TEN(0);
        const unsigned cols = Cz / 4u, rows = L + N;
        for (uint64_t work = (uint64_t)slice * PLOW_NV_THREADS + threadIdx.x;
             work < (uint64_t)B * rows * cols; work += (uint64_t)nblk * PLOW_NV_THREADS) {
            const unsigned e = (unsigned)work;
            const unsigned c = e % cols * 4u, row = e / cols % rows, b = e / (cols * rows);
            const unsigned n = length[b];
            if (n > N) { __trap(); return; }
            float4 value = make_float4(0.f, 0.f, 0.f, 0.f);
            if (n && row >= L) value = *(const float4*)(src + ((size_t)b * N + row - L) * stride + c);
            else if (n && c >= offset && c - offset < C && row >= L - min(position[b], L)) {
                if (slot[b] >= slots) { __trap(); return; }
                const unsigned index = (position[b] % L + row) % L;
                value = *(const float4*)(state + ((size_t)slot[b] * L + index) * C + c - offset);
            }
            z[e] = value;
        }
        return;
    }
    const unsigned rows = min(N, L);
    for (uint64_t work = slice; work < (uint64_t)B * rows; work += nblk) {
        const unsigned item = (unsigned)work;
        const unsigned b = item / rows, r = item % rows, n = length[b];
        if (n > N) { __trap(); return; }
        const unsigned kept = min(n, L);
        if (r >= kept) continue;
        if (slot[b] >= slots) { __trap(); return; }
        const unsigned input = n - kept + r;
        const unsigned index = (unsigned)(((uint64_t)position[b] + input) % L);
        for (unsigned c = threadIdx.x * 4u; c < C; c += PLOW_NV_THREADS * 4u)
            *(float4*)(state + ((size_t)slot[b] * L + index) * C + c) =
                *(const float4*)(src + ((size_t)b * N + input) * stride + offset + c);
    }
}
