/* decode_bw_probe.hip — what does the MEMORY SYSTEM give this access pattern?
 *
 * The decode kernel reads the KV cache two different ways, and they are not equally
 * cheap even though they move the same bytes:
 *
 *   K phase: one lane owns one WHOLE KV row and walks it with 16-byte loads. Across a
 *            wave that is 64 lanes on 64 DIFFERENT rows, so one instruction generates 64
 *            distinct cache-line requests; the line is then re-requested by the same
 *            lane's next 7 loads (it hits, but it still costs a request slot).
 *   V phase: D/8 lanes cover one row, so one instruction is 64 lanes on 1024 CONTIGUOUS
 *            bytes -- 8 lines, fully consumed, no re-request.
 *
 * These probes strip everything else away (no softmax, no LDS, no barriers) and just
 * read a big buffer in each shape, so the decode kernel's measured GB/s can be compared
 * against the ceiling its own addressing implies rather than against HBM peak.
 */
#include "amd_common.h"

typedef unsigned u32v4 __attribute__((ext_vector_type(4)));

/* Finite bf16 data, initialized outside the timed region. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS) void bw_probe_init(
    bf16* dst, unsigned long long n8) {
    auto* p = (PLOW_GLOB u32v4*)(PLOW_GLOB void*)as_glob(dst);
    const unsigned long long stride = (unsigned long long)gridDim.x * PLOW_THREADS;
    for (unsigned long long i = (unsigned long long)blockIdx.x * PLOW_THREADS + threadIdx.x;
         i < n8; i += stride) {
        u32v4 v;
#pragma unroll
        for (unsigned u = 0; u < 4; u++) {
            unsigned x = (unsigned)(i * 4 + u);
            x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
            v[u] = (x & 0x007f007f) | 0x3f003f00;
        }
        p[i] = v;
    }
}

/* Fully coalesced streaming read: 16 B per lane, contiguous across the wave. */
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void bw_probe_stream(
    float* out, const bf16* src, unsigned long long n8 /* number of bf16v8 chunks */) {
    const auto* p = as_glob(src);
    unsigned long long i = (unsigned long long)blockIdx.x * PLOW_THREADS + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * PLOW_THREADS;
    u32v4 acc[4] = {};
    /* Four independent loads/checksums avoid serial bf16 conversions and adds. */
    for (; i < n8; i += 4 * stride) {
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const unsigned long long j = i + u * stride;
            if (j < n8) acc[u] += __builtin_bit_cast(u32v4, ld_glob8(p + j * 8));
        }
    }
    const u32v4 sum = acc[0] + acc[1] + acc[2] + acc[3];
    const unsigned checksum = sum[0] + sum[1] + sum[2] + sum[3];
    out[blockIdx.x * PLOW_THREADS + threadIdx.x] = __builtin_bit_cast(float, checksum);
}

/* ROW-PER-LANE read, the K phase's shape: lane L walks row (base + L) end to end. */
template <int D>
__device__ __forceinline__ void probe_rows(float* out, const bf16* src, unsigned n_rows) {
    const auto* p = as_glob(src);
    const unsigned tid = threadIdx.x;
    float acc = 0.0f;
    for (unsigned r0 = blockIdx.x * PLOW_THREADS; r0 < n_rows; r0 += gridDim.x * PLOW_THREADS) {
        const unsigned r = r0 + tid;
        if (r >= n_rows) break;
        const auto* row = p + (size_t)r * D;
#pragma unroll
        for (int d = 0; d < D; d += 8) {
            const bf16v8 v = ld_glob8(row + d);
#pragma unroll
            for (int u = 0; u < 8; u++) acc += bf2f(v[u]);
        }
    }
    if (acc == 1.2345e-30f) out[0] = acc;
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void bw_probe_rows_512(
    float* out, const bf16* src, unsigned n_rows) {
    probe_rows<512>(out, src, n_rows);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void bw_probe_rows_256(
    float* out, const bf16* src, unsigned n_rows) {
    probe_rows<256>(out, src, n_rows);
}

/* EIGHT lanes per row. One instruction is then 8 rows x 128 contiguous bytes = 8 cache
 * lines fully consumed -- the SAME line count per instruction as the 64-lane cooperative
 * form -- while the dot-product reduction it implies is only log2(8) = 3 lane-exchanges
 * per row-group instead of 6. This is the shape the K phase can actually adopt. */
template <int D>
__device__ __forceinline__ void probe_rows8(float* out, const bf16* src, unsigned n_rows) {
    const auto* p = as_glob(src);
    const unsigned lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const unsigned sub = lane & 7, rl = lane >> 3; /* 8 lanes/row, 8 rows/wave-pass */
    float acc = 0.0f;
    for (unsigned r0 = blockIdx.x * PLOW_THREADS; r0 < n_rows; r0 += gridDim.x * PLOW_THREADS) {
#pragma unroll
        for (int pass = 0; pass < 8; pass++) {
            const unsigned r = r0 + wave * 64 + pass * 8 + rl;
            if (r >= n_rows) break;
            const auto* row = p + (size_t)r * D;
#pragma unroll
            for (int j = 0; j < D / 64; j++) {
                const bf16v8 v = ld_glob8(row + j * 64 + sub * 8);
#pragma unroll
                for (int u = 0; u < 8; u++) acc += bf2f(v[u]);
            }
        }
    }
    if (acc == 1.2345e-30f) out[0] = acc;
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void bw_probe_rows8_512(
    float* out, const bf16* src, unsigned n_rows) {
    probe_rows8<512>(out, src, n_rows);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void bw_probe_rows8_256(
    float* out, const bf16* src, unsigned n_rows) {
    probe_rows8<256>(out, src, n_rows);
}

/* COOPERATIVE row read, the V phase's shape: D/8 lanes cover one row, NG row-groups. */
template <int D>
__device__ __forceinline__ void probe_coop(float* out, const bf16* src, unsigned n_rows) {
    constexpr unsigned NDT = D / 8, NG = PLOW_THREADS / NDT;
    const auto* p = as_glob(src);
    const unsigned dbase = (threadIdx.x % NDT) * 8, grp = threadIdx.x / NDT;
    float acc = 0.0f;
    for (unsigned r0 = blockIdx.x * PLOW_THREADS; r0 < n_rows; r0 += gridDim.x * PLOW_THREADS) {
        for (unsigned r = grp; r < PLOW_THREADS && r0 + r < n_rows; r += NG) {
            const bf16v8 v = ld_glob8(p + (size_t)(r0 + r) * D + dbase);
#pragma unroll
            for (int u = 0; u < 8; u++) acc += bf2f(v[u]);
        }
    }
    if (acc == 1.2345e-30f) out[0] = acc;
}

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void bw_probe_coop_512(
    float* out, const bf16* src, unsigned n_rows) {
    probe_coop<512>(out, src, n_rows);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4) void bw_probe_coop_256(
    float* out, const bf16* src, unsigned n_rows) {
    probe_coop<256>(out, src, n_rows);
}
