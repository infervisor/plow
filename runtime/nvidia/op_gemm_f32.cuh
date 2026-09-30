/* op_gemm_f32.cuh -- fp32-output projections on the warp32 interpreters.
 *
 * PLOW_DOP_GEMV_F32 (135): C f32[M][N] = x bf16[M][K] . W f32[N][K]^T. The reference keeps these
 * weights in fp32 (DeepSeek-V4.1's mHC `hc_*_fn`, N = 24 over K = 4 * hidden; GLM's indexer gate),
 * so the arithmetic stays fp32 (tf32 would round W). Many rows: a CTA owns one row per warp and
 * stages each K chunk of W (N <= 32; the widest of 256 / 128 / 64 the object's arena holds) in the
 * arena, so W crosses L2 once per 8 rows.
 * PLOW_DOP_GEMM_F32 (180): C f32[M][N] = A bf16[M][K] . W bf16[N][K]^T (router logits). A bf16
 * product is exact in fp32, so bf16 tensor cores with fp32 accumulation compute the reference's
 * fp32 matmul up to summation order: 128 x 64 tiles, 8 warps of 32 x 32 on ldmatrix fragments, 64 K
 * per stage, GF_STAGES deep through cp.async. With a scratch and S > 1 splits a work item is (tile, K slice): the slices
 * write f32 partials and the last to arrive sums them in slice order (gemm_f32_scratch_bytes).
 * Both: small output grids (decode) take the dot form -- one CTA per output element.
 */
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace plow_f32 {
__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* Small output grids (decode: the mHC mixes' 24 or the router's 384 outputs of one token): one CTA
 * per output element, the block reducing over K through `red` (>= 32 floats), so a decode step's
 * few outputs spread over the machine instead of one warp walking all of K. */
template <bool W_BF16>
__device__ __forceinline__ void dot_form(float* __restrict__ C, const uint16_t* __restrict__ a16, const void* __restrict__ W, unsigned M,
                                         unsigned N, unsigned K, unsigned slice, unsigned nblk, float* red) {
    const unsigned ln = threadIdx.x & 31u, wv = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (unsigned e = slice; e < M * N; e += nblk) {
        const unsigned m = e / N, n = e % N;
        const uint16_t* a = a16 + (size_t)m * K;
        float acc = 0.f;
        if (W_BF16) {
            const uint16_t* w = static_cast<const uint16_t*>(W) + (size_t)n * K;
            for (unsigned k = threadIdx.x; k < K; k += blockDim.x) acc = fmaf(bf(a[k]), bf(w[k]), acc);
        } else {
            const float* w = static_cast<const float*>(W) + (size_t)n * K;
            for (unsigned k = threadIdx.x; k < K; k += blockDim.x) acc = fmaf(bf(a[k]), w[k], acc);
        }
        acc = warp_sum(acc);
        if (ln == 0) red[wv] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (unsigned w2 = 0; w2 < warps; w2++) s += red[w2];
            C[(size_t)m * N + n] = s;
        }
        __syncthreads();
    }
}
#define PLOW_F32_DOT_MAX_OUT 1024u
}  // namespace plow_f32

namespace plow_f32 {
__device__ __forceinline__ uint32_t pack_bf(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}
/* w = hi + mid + lo, each bf16, exact for every f32 w in bf16's exponent range */
__device__ __forceinline__ void split3(float w, float& hi, float& mid, float& lo) {
    hi = __bfloat162float(__float2bfloat16_rn(w));
    const float r = w - hi;
    mid = __bfloat162float(__float2bfloat16_rn(r));
    lo = r - mid;
}
}  // namespace plow_f32

/* Many rows (prefill): bf16 tensor cores on the three-way bf16 split of W, so every product is
 * exact and accumulates in f32 -- the reference's fp32 matmul up to summation order. A CTA owns 16
 * rows and walks K in 128-wide chunks: x[16][128] and W[<=32][128] staged by cp.async, GV_STAGES
 * deep; warp w takes K columns w*16.. of each chunk and the arena reduces the 8 partials. */
namespace plow_f32 {
constexpr unsigned GV_KC = 128, GV_LDX = GV_KC + 8, GV_LDW = GV_KC + 4, GV_STAGES = 4;
constexpr unsigned GV_STAGE_FLOATS = (16 * GV_LDX) / 2 + 32 * GV_LDW;
constexpr unsigned GV_ARENA_FLOATS = GV_STAGES * GV_STAGE_FLOATS;
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}
}  // namespace plow_f32

__device__ __noinline__ void d_gemv_f32_mma(float* __restrict__ C, const uint16_t* __restrict__ x, const float* __restrict__ W,
                                            unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk, float* arena) {
    using namespace plow_f32;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t4 = lane & 3u;
    const unsigned nt = (N + 7u) / 8u, nrows = nt * 8u, chunks = K / GV_KC;
    auto xs = [&](unsigned b) { return reinterpret_cast<uint16_t*>(arena + b * GV_STAGE_FLOATS); };
    auto ws = [&](unsigned b) { return arena + b * GV_STAGE_FLOATS + (16 * GV_LDX) / 2; };
    for (unsigned rb = slice; rb * 16u < M; rb += nblk) {
        const unsigned r0 = rb * 16u;
        auto load = [&](unsigned c) {
            const unsigned b = c % GV_STAGES, k0 = c * GV_KC;
            for (unsigned e = tid; e < 16u * (GV_KC / 8u); e += blockDim.x) {  // 8 bf16 per 16 B
                const unsigned r = e / (GV_KC / 8u), col = (e % (GV_KC / 8u)) * 8u;
                const bool ok = r0 + r < M;
                cp16(xs(b) + r * GV_LDX + col, x + (size_t)(ok ? r0 + r : 0) * K + k0 + col, ok);
            }
            for (unsigned e = tid; e < nrows * (GV_KC / 4u); e += blockDim.x) {  // 4 f32 per 16 B
                const unsigned n = e / (GV_KC / 4u), col = (e % (GV_KC / 4u)) * 4u;
                const bool ok = n < N;
                cp16(ws(b) + n * GV_LDW + col, W + (size_t)(ok ? n : 0) * K + k0 + col, ok);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        float acc[4][4] = {};
        __syncthreads();  // the previous item's reduction is done with the arena
        for (unsigned c = 0; c + 1u < GV_STAGES; c++) {
            if (c < chunks) load(c);
            else asm volatile("cp.async.commit_group;\n" ::);
        }
        for (unsigned c = 0; c < chunks; c++) {
            if (c + GV_STAGES - 1u < chunks) load(c + GV_STAGES - 1u);
            else asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group %0;\n" ::"n"(GV_STAGES - 1));
            __syncthreads();
            const uint16_t* xb = xs(c % GV_STAGES);
            const float* wb = ws(c % GV_STAGES);
            {
                const unsigned kk = warp * 16u;
                const unsigned kc = kk + t4 * 2u;
                const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(xb + g * GV_LDX + kc),
                                       *reinterpret_cast<const uint32_t*>(xb + (g + 8u) * GV_LDX + kc),
                                       *reinterpret_cast<const uint32_t*>(xb + g * GV_LDX + kc + 8u),
                                       *reinterpret_cast<const uint32_t*>(xb + (g + 8u) * GV_LDX + kc + 8u)};
#pragma unroll
                for (unsigned j = 0; j < 4u; j++) {
                    if (j >= nt) break;
                    const float* wr = wb + (j * 8u + g) * GV_LDW + kc;
                    const float2 w0 = *reinterpret_cast<const float2*>(wr), w1 = *reinterpret_cast<const float2*>(wr + 8);
                    float h[4], m[4], l[4];
                    split3(w0.x, h[0], m[0], l[0]);
                    split3(w0.y, h[1], m[1], l[1]);
                    split3(w1.x, h[2], m[2], l[2]);
                    split3(w1.y, h[3], m[3], l[3]);
                    const uint32_t bh[2] = {pack_bf(h[0], h[1]), pack_bf(h[2], h[3])};
                    const uint32_t bm[2] = {pack_bf(m[0], m[1]), pack_bf(m[2], m[3])};
                    const uint32_t bl[2] = {pack_bf(l[0], l[1]), pack_bf(l[2], l[3])};
                    mma_bf16(acc[j], a, bl);
                    mma_bf16(acc[j], a, bm);
                    mma_bf16(acc[j], a, bh);
                }
            }
            __syncthreads();  // buffer c % GV_STAGES is refilled next iteration
        }
        float* red = arena;  // [8 warps][16][32], over the drained stage buffers
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            if (j >= nt) break;
            float* rw = red + warp * 512u;
            rw[g * 32u + j * 8u + t4 * 2u] = acc[j][0];
            rw[g * 32u + j * 8u + t4 * 2u + 1u] = acc[j][1];
            rw[(g + 8u) * 32u + j * 8u + t4 * 2u] = acc[j][2];
            rw[(g + 8u) * 32u + j * 8u + t4 * 2u + 1u] = acc[j][3];
        }
        __syncthreads();
        for (unsigned e = tid; e < 16u * N; e += blockDim.x) {
            const unsigned r = e / N, n = e % N;
            if (r0 + r >= M) continue;
            float sum = 0.f;
            for (unsigned w2 = 0; w2 < 8u; w2++) sum += red[w2 * 512u + r * 32u + n];
            C[(size_t)(r0 + r) * N + n] = sum;
        }
    }
    __syncthreads();
}

/* Split-K form for many rows: item = (64-row block, K slice s of S). The CTA reduces its slice
 * into scratch partials [S][M][N]; the last CTA to finish a row block (a per-block counter in the
 * scratch head, zero at load and reset by that CTA) sums the S partials IN ORDER s = 0..S-1, so the
 * result is deterministic. Each CTA reads only its K slice of W, which is what the 16-row form
 * above re-reads whole per row block. 8 warps: warp w owns m-tile w & 3 and half (w >> 2) of each
 * 128-wide K chunk. scratch: u32 counters[ceil(M/64)] padded to 256 B, then the partials. */
namespace plow_f32 {
constexpr unsigned SK_ROWS = 64, SK_LDX = GV_KC + 8, SK_STAGES = 3;
constexpr unsigned SK_STAGE_FLOATS = (SK_ROWS * SK_LDX) / 2 + 32 * GV_LDW;
constexpr unsigned SK_ARENA_FLOATS = SK_STAGES * SK_STAGE_FLOATS;
__host__ __device__ constexpr unsigned long long splitk_scratch_bytes(unsigned M, unsigned N, unsigned S) {
    return (((M + SK_ROWS - 1) / SK_ROWS) * 4ull + 255ull) / 256ull * 256ull + (unsigned long long)S * M * N * 4ull;
}
}  // namespace plow_f32

__device__ __noinline__ void d_gemv_f32_splitk(float* __restrict__ C, const uint16_t* __restrict__ x, const float* __restrict__ W,
                                               unsigned M, unsigned N, unsigned K, unsigned S, unsigned char* __restrict__ scratch,
                                               unsigned slice, unsigned nblk, float* arena) {
    using namespace plow_f32;
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, g = lane >> 2, t4 = lane & 3u;
    const unsigned mt = warp & 3u, kh = warp >> 2;
    const unsigned nt = (N + 7u) / 8u, nrows = nt * 8u;
    const unsigned n_rb = (M + SK_ROWS - 1u) / SK_ROWS, kslice = K / S, chunks = kslice / GV_KC;
    unsigned* counters = reinterpret_cast<unsigned*>(scratch);
    float* partial = reinterpret_cast<float*>(scratch + ((n_rb * 4ull + 255ull) / 256ull) * 256ull);
    __shared__ unsigned last_flag;
    auto xs = [&](unsigned b) { return reinterpret_cast<uint16_t*>(arena + b * SK_STAGE_FLOATS); };
    auto ws = [&](unsigned b) { return arena + b * SK_STAGE_FLOATS + (SK_ROWS * SK_LDX) / 2; };
    for (unsigned item = slice; item < n_rb * S; item += nblk) {
        const unsigned rb = item / S, sk = item % S, r0 = rb * SK_ROWS, kbase = sk * kslice;
        auto load = [&](unsigned c) {
            const unsigned b = c % SK_STAGES, k0 = kbase + c * GV_KC;
            for (unsigned e = tid; e < SK_ROWS * (GV_KC / 8u); e += blockDim.x) {
                const unsigned r = e / (GV_KC / 8u), col = (e % (GV_KC / 8u)) * 8u;
                const bool ok = r0 + r < M;
                cp16(xs(b) + r * SK_LDX + col, x + (size_t)(ok ? r0 + r : 0) * K + k0 + col, ok);
            }
            for (unsigned e = tid; e < nrows * (GV_KC / 4u); e += blockDim.x) {
                const unsigned n = e / (GV_KC / 4u), col = (e % (GV_KC / 4u)) * 4u;
                const bool ok = n < N;
                cp16(ws(b) + n * GV_LDW + col, W + (size_t)(ok ? n : 0) * K + k0 + col, ok);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        float acc[4][4] = {};
        __syncthreads();
        for (unsigned c = 0; c + 1u < SK_STAGES; c++) {
            if (c < chunks) load(c);
            else asm volatile("cp.async.commit_group;\n" ::);
        }
        for (unsigned c = 0; c < chunks; c++) {
            if (c + SK_STAGES - 1u < chunks) load(c + SK_STAGES - 1u);
            else asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group %0;\n" ::"n"(SK_STAGES - 1));
            __syncthreads();
            const uint16_t* xb = xs(c % SK_STAGES) + mt * 16u * SK_LDX;
            const float* wb = ws(c % SK_STAGES);
#pragma unroll
            for (unsigned kk = kh * 64u; kk < kh * 64u + 64u; kk += 16u) {
                const unsigned kc = kk + t4 * 2u;
                const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(xb + g * SK_LDX + kc),
                                       *reinterpret_cast<const uint32_t*>(xb + (g + 8u) * SK_LDX + kc),
                                       *reinterpret_cast<const uint32_t*>(xb + g * SK_LDX + kc + 8u),
                                       *reinterpret_cast<const uint32_t*>(xb + (g + 8u) * SK_LDX + kc + 8u)};
#pragma unroll
                for (unsigned j = 0; j < 4u; j++) {
                    if (j >= nt) break;
                    const float* wr = wb + (j * 8u + g) * GV_LDW + kc;
                    const float2 w0 = *reinterpret_cast<const float2*>(wr), w1 = *reinterpret_cast<const float2*>(wr + 8);
                    float h[4], m[4], l[4];
                    split3(w0.x, h[0], m[0], l[0]);
                    split3(w0.y, h[1], m[1], l[1]);
                    split3(w1.x, h[2], m[2], l[2]);
                    split3(w1.y, h[3], m[3], l[3]);
                    const uint32_t bh[2] = {pack_bf(h[0], h[1]), pack_bf(h[2], h[3])};
                    const uint32_t bm[2] = {pack_bf(m[0], m[1]), pack_bf(m[2], m[3])};
                    const uint32_t bl[2] = {pack_bf(l[0], l[1]), pack_bf(l[2], l[3])};
                    mma_bf16(acc[j], a, bl);
                    mma_bf16(acc[j], a, bm);
                    mma_bf16(acc[j], a, bh);
                }
            }
            __syncthreads();
        }
        /* the two K halves of each m-tile meet in the arena, then this slice's partial goes out */
        float* red = arena;  // [2][64][32]
#pragma unroll
        for (unsigned j = 0; j < 4u; j++) {
            if (j >= nt) break;
            float* rw = red + kh * (SK_ROWS * 32u) + (mt * 16u) * 32u;
            rw[g * 32u + j * 8u + t4 * 2u] = acc[j][0];
            rw[g * 32u + j * 8u + t4 * 2u + 1u] = acc[j][1];
            rw[(g + 8u) * 32u + j * 8u + t4 * 2u] = acc[j][2];
            rw[(g + 8u) * 32u + j * 8u + t4 * 2u + 1u] = acc[j][3];
        }
        __syncthreads();
        for (unsigned e = tid; e < SK_ROWS * N; e += blockDim.x) {
            const unsigned r = e / N, n = e % N;
            if (r0 + r < M) partial[((size_t)sk * M + r0 + r) * N + n] = red[r * 32u + n] + red[SK_ROWS * 32u + r * 32u + n];
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) last_flag = atomicAdd(&counters[rb], 1u) == S - 1u;
        __syncthreads();
        if (last_flag) {
            __threadfence();
            for (unsigned e = tid; e < SK_ROWS * N; e += blockDim.x) {
                const unsigned r = e / N, n = e % N;
                if (r0 + r >= M) continue;
                float sum = 0.f;
                for (unsigned s2 = 0; s2 < S; s2++) sum += __ldcg(&partial[((size_t)s2 * M + r0 + r) * N + n]);
                C[(size_t)(r0 + r) * N + n] = sum;
            }
            if (tid == 0) counters[rb] = 0u;
        }
    }
    __syncthreads();
}

/* Decode form (M <= 8, i4 = 1): CTA s of S owns K slice s for EVERY output and writes its partial
 * C[s][M][N]; the consumer (HyperConnPre, i6 = S) sums the S slices in order. Each CTA reads its x
 * slice once and no CTA waits on another. Both loops stay ROLLED: the decode interpreter is one huge
 * function whose op bodies are cold in the instruction cache every step, so an unrolled body costs
 * more in i-cache misses than it saves -- cp.async keeps every load in flight instead. */
namespace plow_f32 {
__host__ __device__ constexpr unsigned kpart_arena_floats(unsigned M, unsigned N, unsigned ks) {
    return N * (ks + 4u) + M * (ks + 8u) / 2u;
}
}  // namespace plow_f32

__device__ __forceinline__ void d_gemv_f32_kpart(float* __restrict__ C, const uint16_t* __restrict__ x, const float* __restrict__ W,
                                                 unsigned M, unsigned N, unsigned K, unsigned S, unsigned slice, unsigned nblk,
                                                 float* arena) {
    using namespace plow_f32;
    __shared__ float red[512];
    const unsigned ks = K / S, lw = ks + 4u, lx = ks + 8u;  // padded rows: conflict-free float4 reads
    float* ws = arena;
    uint16_t* xs = reinterpret_cast<uint16_t*>(arena + N * lw);
    const unsigned wc = N * (ks / 4u), xc = M * (ks / 8u);
    for (unsigned s = slice; s < S; s += nblk) {
#pragma unroll 1
        for (unsigned e = threadIdx.x; e < wc + xc; e += blockDim.x) {
            if (e < wc) {
                const unsigned n = e / (ks / 4u), c = (e % (ks / 4u)) * 4u;
                cp16(ws + n * lw + c, W + (size_t)n * K + (size_t)s * ks + c, true);
            } else {
                const unsigned m = (e - wc) / (ks / 8u), c = ((e - wc) % (ks / 8u)) * 8u;
                cp16(xs + m * lx + c, x + (size_t)m * K + (size_t)s * ks + c, true);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        /* thread (output, half of the slice) */
#pragma unroll 1
        for (unsigned o = threadIdx.x; o < M * N * 2u; o += blockDim.x) {
            const unsigned out = o >> 1, h = (o & 1u) * (ks / 2u);
            const float* wr = ws + (out % N) * lw + h;
            const uint16_t* xr = xs + (out / N) * lx + h;
            float acc = 0.f;
#pragma unroll 2
            for (unsigned k = 0; k < ks / 2u; k += 4u) {
                const float4 w4 = *reinterpret_cast<const float4*>(wr + k);
                const uint2 x2 = *reinterpret_cast<const uint2*>(xr + k);
                acc = fmaf(__uint_as_float(x2.x << 16), w4.x, acc);
                acc = fmaf(__uint_as_float(x2.x & 0xffff0000u), w4.y, acc);
                acc = fmaf(__uint_as_float(x2.y << 16), w4.z, acc);
                acc = fmaf(__uint_as_float(x2.y & 0xffff0000u), w4.w, acc);
            }
            red[o] = acc;
        }
        __syncthreads();
        for (unsigned out = threadIdx.x; out < M * N; out += blockDim.x)
            C[((size_t)s * M + out / N) * N + out % N] = red[2u * out] + red[2u * out + 1u];
        __syncthreads();
    }
}

/* Pre-gate L2 prefetch of the W slice `d_gemv_f32_kpart` reads: W does not depend on the producer,
 * so its DRAM latency hides behind the wait. */
__device__ __forceinline__ void d_gemv_f32_kpart_pf(const float* __restrict__ W, unsigned N, unsigned K, unsigned S, unsigned slice) {
    const unsigned lines = K / S * 4u / 128u;
    for (unsigned e = threadIdx.x; e < N * lines; e += blockDim.x)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const char*>(W + (size_t)(e / lines) * K + (size_t)slice * (K / S)) +
                                                     (e % lines) * 128u));
}

__device__ __forceinline__ void d_gemv_f32(float* __restrict__ C, const __nv_bfloat16* __restrict__ X, const float* __restrict__ W,
                                           unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk, float* arena,
                                           unsigned arena_floats, unsigned char* scratch = nullptr, unsigned splits = 0,
                                           unsigned partial_out = 0) {
    using namespace plow_f32;
    const uint16_t* x = reinterpret_cast<const uint16_t*>(X);
    if (partial_out) {
        const unsigned ks = splits ? K / splits : 0u;
        if (splits < 2u || M > 8u || K % splits || ks % 8u || M * N * 2u > 512u || arena_floats < kpart_arena_floats(M, N, ks))
            __trap();
        d_gemv_f32_kpart(C, x, W, M, N, K, splits, slice, nblk, arena);
        return;
    }
    if (scratch && splits > 1u && N <= 32u && K % (splits * GV_KC) == 0 && blockDim.x == 256u && arena_floats >= SK_ARENA_FLOATS) {
        d_gemv_f32_splitk(C, x, W, M, N, K, splits, scratch, slice, nblk, arena);
        return;
    }
    if (M * N > PLOW_F32_DOT_MAX_OUT && N <= 32u && K % GV_KC == 0 && blockDim.x == 256u && arena_floats >= GV_ARENA_FLOATS) {
        d_gemv_f32_mma(C, x, W, M, N, K, slice, nblk, arena);
        return;
    }
    const unsigned ch = N * 256u <= arena_floats ? 256u : N * 128u <= arena_floats ? 128u : 64u;
    if (M * N <= PLOW_F32_DOT_MAX_OUT || N > 32u || N * ch > arena_floats || K % ch) {
        dot_form<false>(C, x, W, M, N, K, slice, nblk, arena);
        return;
    }
    const unsigned warps = blockDim.x >> 5, wv = threadIdx.x >> 5, ln = threadIdx.x & 31u;
    float* ws = arena;  // [N][ch] f32 chunk of W
    for (unsigned m0 = slice * warps; m0 < M; m0 += nblk * warps) {
        const unsigned m = m0 + wv;
        const bool live = m < M;
        float acc[32];
#pragma unroll
        for (int n = 0; n < 32; n++) acc[n] = 0.f;
        for (unsigned k0 = 0; k0 < K; k0 += ch) {
            __syncthreads();
            for (unsigned e = threadIdx.x * 4u; e < N * ch; e += blockDim.x * 4u) {
                const unsigned n = e / ch, k = e % ch;
                *reinterpret_cast<float4*>(ws + e) = *reinterpret_cast<const float4*>(W + (size_t)n * K + k0 + k);
            }
            __syncthreads();
            if (live && ln * 8u < ch) {
                const uint4 u = *reinterpret_cast<const uint4*>(x + (size_t)m * K + k0 + ln * 8u);
                const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
                float a[8];
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    a[2 * q] = __uint_as_float(w4[q] << 16);
                    a[2 * q + 1] = __uint_as_float(w4[q] & 0xffff0000u);
                }
#pragma unroll
                for (unsigned n = 0; n < 32u; n++) {
                    if (n >= N) break;
                    const float4 wa = *reinterpret_cast<const float4*>(ws + n * ch + ln * 8u);
                    const float4 wb = *reinterpret_cast<const float4*>(ws + n * ch + ln * 8u + 4u);
                    const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
                    for (int u2 = 0; u2 < 8; u2++) acc[n] = fmaf(a[u2], w[u2], acc[n]);
                }
            }
        }
#pragma unroll
        for (unsigned n = 0; n < 32u; n++) {
            if (n >= N) break;
            const float s = warp_sum(acc[n]);
            if (live && ln == 0) C[(size_t)m * N + n] = s;
        }
    }
    __syncthreads();
}

namespace plow_f32 {
__host__ __device__ constexpr unsigned long long gemm_f32_scratch_bytes(unsigned M, unsigned N, unsigned S) {
    return (((M + 127ull) / 128ull) * ((N + 63ull) / 64ull) * 4ull + 255ull) / 256ull * 256ull + (unsigned long long)S * M * N * 4ull;
}
}  // namespace plow_f32

__device__ __forceinline__ void d_gemm_f32(float* __restrict__ C, const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
                                           unsigned M, unsigned N, unsigned K, unsigned slice, unsigned nblk, float* arena,
                                           unsigned arena_floats, unsigned char* scratch = nullptr, unsigned splits = 0) {
    using namespace plow_f32;
    const uint16_t* a16 = reinterpret_cast<const uint16_t*>(A);
    const uint16_t* w16 = reinterpret_cast<const uint16_t*>(W);
    constexpr unsigned LD = 72;  // bf16 row pitch in the arena: 64 + 8 (conflict-free fragment loads)
    constexpr unsigned GF_STAGES = 4, STAGE = (128 + 64) * LD;  // bf16 elements per stage: A[128] then W[64]
    if (M * N <= PLOW_F32_DOT_MAX_OUT || K % 64u || blockDim.x != 256u || arena_floats < GF_STAGES * STAGE / 2u) {
        dot_form<true>(C, a16, W, M, N, K, slice, nblk, arena);
        return;
    }
    if (M <= 8u && K % 8u == 0u && arena_floats >= 256u) {
        /* decode rows: a block takes columns slice + j*nblk (4 at a time), its 256 threads split K 8
         * elements each, warp sums meet in the arena in warp order. Exact bf16 products, f32 sums. */
        const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, warps = blockDim.x >> 5;
        float* red = arena;  // [warps][4 cols][8 rows]
        for (unsigned n0 = slice; n0 < N; n0 += 4u * nblk) {
            float acc[4][8] = {};
            /* NI K-slots per thread per pass, every weight load of the pass issued before any math:
             * one HBM round trip per pass instead of one per slot */
            constexpr unsigned NI = 3;
            for (unsigned kb = tid * 8u; kb < K; kb += NI * blockDim.x * 8u) {
                uint4 wv[NI][4];
#pragma unroll
                for (unsigned i = 0; i < NI; i++) {
                    const unsigned k = kb + i * blockDim.x * 8u;
#pragma unroll
                    for (unsigned j = 0; j < 4u; j++) {
                        const unsigned n = n0 + j * nblk;
                        wv[i][j] = k < K && n < N ? __ldg(reinterpret_cast<const uint4*>(w16 + (size_t)n * K + k)) : make_uint4(0, 0, 0, 0);
                    }
                }
#pragma unroll
                for (unsigned i = 0; i < NI; i++) {
                    const unsigned k = kb + i * blockDim.x * 8u;
                    if (k >= K) break;
#pragma unroll
                    for (unsigned m = 0; m < 8u; m++) {
                        if (m >= M) break;
                        const uint4 au = *reinterpret_cast<const uint4*>(a16 + (size_t)m * K + k);
                        const uint32_t aa[4] = {au.x, au.y, au.z, au.w};
#pragma unroll
                        for (unsigned j = 0; j < 4u; j++) {
                            const uint32_t ww[4] = {wv[i][j].x, wv[i][j].y, wv[i][j].z, wv[i][j].w};
#pragma unroll
                            for (int q = 0; q < 4; q++) {
                                acc[j][m] = fmaf(__uint_as_float(aa[q] << 16), __uint_as_float(ww[q] << 16), acc[j][m]);
                                acc[j][m] = fmaf(__uint_as_float(aa[q] & 0xffff0000u), __uint_as_float(ww[q] & 0xffff0000u), acc[j][m]);
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (unsigned j = 0; j < 4u; j++)
#pragma unroll
                for (unsigned m = 0; m < 8u; m++) {
                    const float v = warp_sum(acc[j][m]);
                    if (lane == 0) red[warp * 32u + j * 8u + m] = v;
                }
            __syncthreads();
            if (tid < 32u) {
                const unsigned j = tid >> 3, m = tid & 7u, n = n0 + j * nblk;
                float v = 0.f;
                for (unsigned w2 = 0; w2 < warps; w2++) v += red[w2 * 32u + tid];
                if (m < M && n < N) C[(size_t)m * N + n] = v;
            }
            __syncthreads();
        }
        return;
    }
    uint16_t* const base = reinterpret_cast<uint16_t*>(arena);
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    const unsigned wm = warp & 3u, wn = warp >> 2;  // warp tile rows wm*32, cols wn*32
    const unsigned g = lane >> 2, t4 = lane & 3u;
    const unsigned S = scratch && splits > 1u && K % (splits * 64u) == 0 ? splits : 1u;
    const unsigned tn = (N + 63u) / 64u, tiles = tn * ((M + 127u) / 128u), chunks = K / S / 64u;
    /* ldmatrix lane addresses: A x4 = rows lane % 16, k half lane / 16; W x4 = n rows (lane & 7) + 8 (lane / 16),
     * k half (lane / 8) & 1 -- two n8 fragments per load */
    const unsigned a_off = (wm * 32u + (lane & 15u)) * LD + (lane >> 4) * 8u;
    const unsigned w_off = 128u * LD + (wn * 32u + (lane & 7u) + ((lane >> 4) << 3)) * LD + ((lane >> 3) & 1u) * 8u;
    unsigned* counters = reinterpret_cast<unsigned*>(scratch);
    float* partial = reinterpret_cast<float*>(scratch + ((tiles * 4ull + 255ull) / 256ull) * 256ull);
    __shared__ unsigned last_flag;
    for (unsigned item = slice; item < tiles * S; item += nblk) {
        const unsigned tile = item / S, sk = item % S;
        const unsigned m0 = (tile / tn) * 128u, n0 = (tile % tn) * 64u;
        auto load = [&](unsigned c) {
            if (c < chunks) {
                uint16_t* As = base + (c % GF_STAGES) * STAGE;
                const unsigned k0 = (sk * chunks + c) * 64u;
#pragma unroll
                for (unsigned i = 0; i < 6u; i++) {  // 192 rows (A 128, W 64) x 8 chunks of 16 B
                    const unsigned e = tid + i * 256u, r = e >> 3, off = (e & 7u) * 8u;
                    if (r < 128u) cp16(As + r * LD + off, a16 + (size_t)(m0 + r < M ? m0 + r : 0) * K + k0 + off, m0 + r < M);
                    else {
                        const unsigned n = n0 + r - 128u;
                        cp16(As + r * LD + off, w16 + (size_t)(n < N ? n : 0) * K + k0 + off, n < N);
                    }
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        float acc[2][4][4] = {};
        __syncthreads();  // the previous tile's readers are done with the stages
        for (unsigned c = 0; c + 1u < GF_STAGES; c++) load(c);
        for (unsigned c = 0; c < chunks; c++) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(GF_STAGES - 2));
            __syncthreads();  // chunk c landed for every thread; stage (c - 1) % GF_STAGES is free
            load(c + GF_STAGES - 1u);
            const uint32_t sb = (uint32_t)__cvta_generic_to_shared(base + (c % GF_STAGES) * STAGE);
#pragma unroll
            for (unsigned kk = 0; kk < 64u; kk += 16u) {
                uint32_t af[2][4], bfr[4][2];
#pragma unroll
                for (unsigned i = 0; i < 2u; i++)
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3])
                                 : "r"(sb + (a_off + i * 16u * LD + kk) * 2u));
#pragma unroll
                for (unsigned j = 0; j < 4u; j += 2u)
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(bfr[j][0]), "=r"(bfr[j][1]), "=r"(bfr[j + 1][0]), "=r"(bfr[j + 1][1])
                                 : "r"(sb + (w_off + j * 8u * LD + kk) * 2u));
#pragma unroll
                for (unsigned i = 0; i < 2u; i++)
#pragma unroll
                    for (unsigned j = 0; j < 4u; j++) mma_bf16(acc[i][j], af[i], bfr[j]);
            }
        }
        float* const dst = S > 1u ? partial + (size_t)sk * M * N : C;
#pragma unroll
        for (unsigned i = 0; i < 2u; i++)
#pragma unroll
            for (unsigned j = 0; j < 4u; j++)
#pragma unroll
                for (unsigned h = 0; h < 2u; h++) {
                    const unsigned m = m0 + wm * 32 + i * 16 + g + h * 8, n = n0 + wn * 32 + j * 8 + t4 * 2;
                    if (m >= M) continue;
                    if (n < N) dst[(size_t)m * N + n] = acc[i][j][h * 2];
                    if (n + 1 < N) dst[(size_t)m * N + n + 1] = acc[i][j][h * 2 + 1];
                }
        if (S > 1u) {
            __threadfence();
            __syncthreads();
            if (tid == 0) last_flag = atomicAdd(&counters[tile], 1u) == S - 1u;
            __syncthreads();
            if (last_flag) {
                __threadfence();
                for (unsigned e = tid; e < 128u * 64u; e += blockDim.x) {
                    const unsigned m = m0 + e / 64u, n = n0 + e % 64u;
                    if (m >= M || n >= N) continue;
                    float sum = 0.f;
                    for (unsigned s2 = 0; s2 < S; s2++) sum += __ldcg(&partial[((size_t)s2 * M + m) * N + n]);
                    C[(size_t)m * N + n] = sum;
                }
                if (tid == 0) counters[tile] = 0u;
            }
        }
    }
    __syncthreads();
}
