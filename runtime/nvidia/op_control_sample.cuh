#pragma once
#include "control_sample.h"
#include "op_elementwise.cuh"

extern "C" __device__ const unsigned plow_control_sample_abi = 1;

static __device__ unsigned long long cs_mix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static __device__ unsigned long long cs_key(float v, unsigned i) {
    const unsigned u = __float_as_uint(v);
    return ((unsigned long long)(u & 0x80000000u ? ~u : u ^ 0x80000000u) << 32) | ~i;
}
static __device__ void cs_sort(unsigned long long* keys) {
    for (unsigned k = 2; k <= 4096; k *= 2)
        for (unsigned j = k / 2; j; j /= 2) {
            for (unsigned i = threadIdx.x; i < 4096; i += blockDim.x) {
                const unsigned other = i ^ j;
                if (other > i) {
                    const auto a = keys[i], b = keys[other];
                    const bool descending = (i & k) == 0;
                    keys[i] = descending ? max(a, b) : min(a, b);
                    keys[other] = descending ? min(a, b) : max(a, b);
                }
            }
            __syncthreads();
        }
}
static __device__ void cs_softmax(float* p, const float* l, unsigned n, float* scratch) {
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (unsigned i = 0; i < n; i++) m = fmaxf(m, l[i]);
        *scratch = m;
    }
    __syncthreads();
    const float m = *scratch;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) p[i] = expf(__fsub_rn(l[i], m));
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0;
        for (unsigned i = 0; i < n; i++) sum = __fadd_rn(sum, p[i]);
        *scratch = sum > 0 ? __fdiv_rn(1.f, sum) : 0.f;
    }
    __syncthreads();
    const float inv = *scratch;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) p[i] = __fmul_rn(p[i], inv);
    __syncthreads();
}
static __device__ void cs_emitted(ControlSampleState& s, unsigned code) {
    const auto& g = s.plan;
    const unsigned f = ++s.frames;
    for (unsigned i = 1; i < g.hist; i++) s.tail[i - 1] = s.tail[i];
    if (g.hist) s.tail[g.hist - 1] = code;
    bool suspended = false;
    for (unsigned i = 0; i < g.n_windows; i++)
        suspended |= f >= g.windows[i][0] && f <= g.windows[i][1];
    if (f >= g.cap || f < g.start || !g.every || f % g.every || suspended) return;
    bool hit = false;
    if (f >= max(12u, g.stall) && g.stall && g.stall <= g.hist && g.period_win <= g.hist) {
        hit = true;
        for (unsigned i = g.hist - g.stall + 1; i < g.hist; i++)
            hit &= s.tail[i] == s.tail[g.hist - g.stall];
        if (!hit && f >= max(max(12u, g.stall), g.period_win)) {
            for (unsigned p = g.pmin; p <= g.pmax && p < g.period_win; p++) {
                unsigned same = 0;
                for (unsigned i = g.hist - g.period_win; i < g.hist - p; i++) same += s.tail[i + p] == s.tail[i];
                hit |= __fdiv_rn((float)same, (float)(g.period_win - p)) >= g.period_thr;
            }
        }
    }
    if (hit) {
        if (s.boost_at == 0xffffffffu) { s.boost_at = f; s.aeos_bias = g.boost; }
        if (f - s.boost_at >= g.backstop) s.backstop = 1;
    } else if (s.boost_at != 0xffffffffu) { s.aeos_bias = 0; s.boost_at = 0xffffffffu; }
}

extern "C" __global__ void control_sample(
    ControlSampleState* states, const ControlSampleRow* rows, const unsigned short* logits,
    unsigned* codes, unsigned count, unsigned stride, unsigned classes, unsigned stop) {
    if (blockIdx.x >= count) return;
    if (classes > 4096 || !classes || stop + 1 != classes || stride < classes) { asm("trap;"); return; }
    const auto row = rows[blockIdx.x];
    auto& s = states[row.slot];
    const auto& c = s.plan;
    extern __shared__ unsigned long long cs_mem[];
    auto* keys = cs_mem;
    auto* l = (float*)(keys + 4096);
    auto* p = l + 4096;
    auto* scratch = p + 4096;
    for (unsigned i = threadIdx.x; i < 4096; i += blockDim.x) {
        float v = -INFINITY;
        if (i < classes) {
            v = __fdiv_rn(__uint_as_float((unsigned)logits[(size_t)row.logits * stride + i] << 16), c.temperature);
            if (i == stop) v = __fadd_rn(v, s.aeos_bias);
            if (i < stop && c.presence_penalty != 0) {
                bool seen = false;
                for (unsigned j = 0; j < c.penalty_window; j++) seen |= s.recent[j] == (int)i;
                if (seen) v = __fsub_rn(v, c.presence_penalty);
            }
        }
        l[i] = v;
        keys[i] = cs_key(v, i);
    }
    __syncthreads();
    const unsigned k = min(max(c.top_k, 1u), classes);
    unsigned kth;
    if (k <= 64u) {
        auto* sel = (unsigned*)keys;
        auto* ctl = sel + 64;
        auto* hist = ctl + 4;
        auto* buf = (unsigned long long*)(hist + 256);
        auto* pm = buf + 64;
        block_topk_u64([l](unsigned i) { return cs_key(l[i], i); }, classes, k, 56, sel, ctl, hist, buf, pm);
        kth = sel[k - 1];
    } else {
        cs_sort(keys);
        kth = ~(unsigned)keys[k - 1];
    }
    const float threshold = l[kth];
    for (unsigned i = threadIdx.x; i < classes; i += blockDim.x) if (l[i] < threshold) l[i] = -INFINITY;
    __syncthreads();
    if (c.top_p < 1.f) {
        cs_softmax(p, l, classes, scratch);
        for (unsigned i = threadIdx.x; i < 4096; i += blockDim.x) keys[i] = cs_key(i < classes ? p[i] : -1.f, i);
        __syncthreads();
        cs_sort(keys);
        if (threadIdx.x == 0) {
            float cum = 0;
            for (unsigned j = 0; j < classes; j++) {
                const unsigned i = ~(unsigned)keys[j];
                if (p[i] <= 0) break;
                if (cum > c.top_p) l[i] = -INFINITY;
                cum = __fadd_rn(cum, p[i]);
            }
        }
        __syncthreads();
    }
    cs_softmax(p, l, classes, scratch);
    if (threadIdx.x == 0) {
        const float pa = p[stop];
        unsigned am = 0;
        for (unsigned i = 1; i < classes; i++) if (p[i] > p[am]) am = i;
        const bool can = s.ticks >= c.min_frames;
        if (can) s.hazard = __fsub_rn(1.f, __fmul_rn(__fsub_rn(1.f, s.hazard), __fsub_rn(1.f, pa)));
        s.stopped = s.backstop || (can && (am == stop || pa >= c.stop_p || s.hazard >= c.stop_hazard));
        p[stop] = 0;
        *scratch = 0;
    }
    __syncthreads();
    unsigned long long best = 0;
    for (unsigned i = threadIdx.x; i < stop; i += blockDim.x) if (p[i] > 0) {
        const auto h = cs_mix(cs_mix(c.seed ^ ((unsigned long long)s.ticks << 32)) ^ i);
        const float u = __fmul_rn(__fadd_rn((float)(h >> 40), .5f), 0x1p-24f);
        const float score = __fsub_rn(logf(p[i]), logf(-logf(u)));
        // The shared helper normalizes signed zero and breaks ties at the lowest class.
        if (score > -INFINITY) best = max(best, f32_key(score, i));
    }
    best = block_max_u64(best, keys);
    if (threadIdx.x == 0) {
        const unsigned code = best ? ~(unsigned)best : 0u;
        ++s.ticks;
        for (unsigned i = 1; i < c.penalty_window; i++) s.recent[i - 1] = s.recent[i];
        if (c.penalty_window) s.recent[c.penalty_window - 1] = code;
        if (!s.stopped) { s.last = code; cs_emitted(s, code); }
        codes[row.output] = s.last;
    }
}
extern "C" __global__ void control_sample_finish(
    const ControlSampleState* states, const ControlSampleRow* rows, unsigned* codes,
    unsigned count, unsigned codebooks, unsigned stop) {
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;
    const auto row = rows[r];
    if (states[row.slot].stopped) {
        codes[(size_t)row.slot * codebooks] = stop;
        for (unsigned k = 1; k < codebooks; k++) codes[(size_t)row.slot * codebooks + k] = 0;
    }
}

extern "C" __global__ void tick_codec_codes(
    const unsigned* source, const unsigned* slots, unsigned* target,
    unsigned rows, unsigned capacity, unsigned codebooks, unsigned stop) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)capacity * codebooks) return;
    const unsigned r = i / codebooks, k = i % codebooks;
    unsigned code = 0;
    if (r < rows) {
        const size_t base = (size_t)slots[r] * codebooks;
        if (source[base] != stop) code = source[base + k];
    }
    target[i] = code;
}
extern "C" __global__ void tick_pack_pcm16(
    const unsigned* codes, const unsigned* slots, const float* pcm, unsigned char* packet,
    unsigned rows, unsigned codebooks, unsigned samples, unsigned stride, unsigned stop) {
    const unsigned r = blockIdx.x;
    if (r >= rows) return;
    const size_t base = (size_t)slots[r] * codebooks;
    auto* output = packet + (size_t)r * stride;
    for (unsigned k = threadIdx.x; k < codebooks; k += blockDim.x)
        ((unsigned*)output)[k] = codes[base + k];
    const bool stopped = codes[base] == stop;
    auto* audio = (short*)(output + codebooks * 4);
    for (unsigned s = threadIdx.x; s < samples; s += blockDim.x) {
        float x = stopped ? 0.f : pcm[(size_t)r * samples + s];
        x = isnan(x) ? 0.f : fminf(1.f, fmaxf(-1.f, x));
        audio[s] = (short)roundf(__fmul_rn(x, 32767.f));
    }
}

extern "C" __global__ void tick_codec_lengths(
    const unsigned* codes, const unsigned* slots, unsigned* lengths,
    unsigned rows, unsigned capacity, unsigned codebooks, unsigned scale, unsigned stop) {
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < capacity) lengths[r] = r < rows && codes[(size_t)slots[r] * codebooks] != stop ? scale : 0;
}

extern "C" __global__ void handoff_pack_pcm16(
    const ControlSampleState* states, const unsigned* slots, const float* pcm,
    unsigned char* packet, unsigned rows, unsigned capacity, unsigned samples, unsigned stride) {
    const unsigned r = blockIdx.x;
    if (r >= rows) return;
    if (!capacity || rows > capacity || !samples || stride < 8 || (stride & 3)
        || samples > (stride - 8) / 2) { asm("trap;"); return; }
    const unsigned slot = slots[r];
    if (slot >= capacity) { asm("trap;"); return; }
    const bool stopped = states[slot].stopped != 0;
    auto* output = packet + (size_t)r * stride;
    if (threadIdx.x == 0) {
        ((unsigned*)output)[0] = stopped;
        ((unsigned*)output)[1] = states[slot].last;
    }
    auto* audio = (short*)(output + 8);
    for (size_t s = threadIdx.x; s < samples; s += blockDim.x) {
        float x = stopped ? 0.f : pcm[(size_t)r * samples + s];
        x = isnan(x) ? 0.f : fminf(1.f, fmaxf(-1.f, x));
        audio[s] = (short)roundf(__fmul_rn(x, 32767.f));
    }
    for (size_t b = 8 + (size_t)samples * 2 + threadIdx.x; b < stride; b += blockDim.x)
        output[b] = 0;
}

extern "C" __global__ void handoff_pack_codes(
    const unsigned* source, const unsigned* slots, unsigned* target,
    unsigned rows, unsigned source_capacity, unsigned capacity, unsigned codebooks) {
    if (!source_capacity || !capacity || !codebooks || rows > capacity) { asm("trap;"); return; }
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)capacity * codebooks) return;
    const size_t r = i / codebooks, k = i % codebooks;
    unsigned value = 0;
    if (r < rows) {
        const unsigned slot = slots[r];
        if (slot >= source_capacity) { asm("trap;"); return; }
        value = source[(size_t)slot * codebooks + k];
    }
    target[i] = value;
}

extern "C" __global__ void handoff_codec_lengths(
    const ControlSampleState* states, const unsigned* slots, unsigned* lengths,
    unsigned rows, unsigned state_capacity, unsigned capacity, unsigned scale) {
    if (!state_capacity || !capacity || !scale || rows > capacity) { asm("trap;"); return; }
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= capacity) return;
    unsigned value = 0;
    if (r < rows) {
        const unsigned slot = slots[r];
        if (slot >= state_capacity) { asm("trap;"); return; }
        value = states[slot].stopped ? 0 : scale;
    }
    lengths[r] = value;
}
