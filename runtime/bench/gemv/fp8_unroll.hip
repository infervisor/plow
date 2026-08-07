// Is the megakernel's fp8 decode GEMV slow because of the LOOP, or because of the CONTEXT?
//
// Measured in the real trace (Gemma-4 12B, fp8, occ4), per-op union interval:
//     q_proj  N=4096 K=3840 b=152 -> 37.7 us    o_proj N=3840 K=4096  b=304 -> 35.5 us
//     k/v     N=2048 K=3840 b=76  -> 37.5 us    down   N=3840 K=15360 b=304 -> 46.1 us
// Every op <=16 MB sits on the SAME ~36 us floor regardless of size, which is not what a
// bandwidth-bound kernel does. This is gemv_rows_fp8 lifted out verbatim -- same two-column
// pairing, same buf_rsrc_fp8, same CDNA3 dequant, same grid -- so whatever the megakernel adds
// is exactly the difference between these numbers and those.
//
// CAVEAT, and it matters when reading the absolute numbers: this reproduces the PAIR loop only
// and omits gemv_fp8_rows_1, the odd-column tail. At these shapes that is 10 of 13 columns for
// o_proj and 22 of 27 for k/v, so the per-op microseconds here are ~20-30% OPTIMISTIC. The UN
// RANKING is unaffected (every arm drops the same tail), and that ranking is what this is for.
//
// It also sweeps UN. d_gemv_fp8 hardcodes UN=3 for everything except K==9728, and its comment
// names the K's it was tuned for (2560/4096/9728) -- none of which are Gemma-4's. With
// step=1024, K=3840 and K=4096 both give nchunk=4, so UN=3 issues c=0,3 => 6 chunk-steps for 4
// real chunks: 50% of the converts and dots are on zeros.
#include <hip/hip_runtime.h>
#include <cstdio>

typedef __bf16 bf16;
typedef bf16 bf16v8 __attribute__((ext_vector_type(8)));
typedef bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef unsigned fp8v16 __attribute__((ext_vector_type(4)));   // 4 u32 = 16 fp8 per lane
typedef float f32x2 __attribute__((ext_vector_type(2)));
#define PLOW_BUF_RSRC3 0x00020000u
#define PLOW_WAVE 64
#define PLOW_WAVES 8

__device__ __forceinline__ __amdgpu_buffer_rsrc_t buf_rsrc_fp8(const unsigned char* b, unsigned n) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)b, (short)0, n, PLOW_BUF_RSRC3);
}
__device__ __forceinline__ fp8v16 buf_ld_fp8(__amdgpu_buffer_rsrc_t r, unsigned off) {
    return __builtin_bit_cast(fp8v16, __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 3));
}
__device__ __forceinline__ unsigned plow_pack_bf16x2(float lo, float hi) {
    unsigned a, b; __builtin_memcpy(&a, &lo, 4); __builtin_memcpy(&b, &hi, 4);
    return (a >> 16) | (b & 0xffff0000u);
}
__device__ __forceinline__ unsigned plow_fp8_mask_neg0(unsigned w) {
    const unsigned t = w ^ 0x80808080u;
    const unsigned z = (t - 0x01010101u) & ~t & 0x80808080u;
    return w & ~(z | (z - (z >> 7)));
}
__device__ __forceinline__ void plow_fp8x4_ocp_to_bf16(unsigned w, unsigned& lo, unsigned& hi) {
    const unsigned m = plow_fp8_mask_neg0(w);
    const f32x2 a = __builtin_amdgcn_cvt_pk_f32_fp8(m, false);
    const f32x2 c = __builtin_amdgcn_cvt_pk_f32_fp8(m, true);
    lo = plow_pack_bf16x2(a[0] * 2.0f, a[1] * 2.0f);
    hi = plow_pack_bf16x2(c[0] * 2.0f, c[1] * 2.0f);
}
__device__ __forceinline__ void fp8_to_bf16v8(fp8v16 w, bf16v8& lo, bf16v8& hi) {
    union { bf16v8 v; unsigned u[4]; } ol, oh;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        auto& o = (i < 2) ? ol : oh;
        const int b = (i & 1) * 2;
        plow_fp8x4_ocp_to_bf16(w[i], o.u[b + 0], o.u[b + 1]);
    }
    lo = ol.v; hi = oh.v;
}
// gfx942 has no v_dot2c_f32_bf16: widen and FMA (plow_dot2_bf16's CDNA3 arm).
__device__ __forceinline__ float dot2(bf16x2 a, bf16x2 b, float acc) {
    const unsigned pa = __builtin_bit_cast(unsigned, a), pb = __builtin_bit_cast(unsigned, b);
    float fa0, fb0, fa1, fb1; unsigned t;
    t = pa << 16; __builtin_memcpy(&fa0, &t, 4);
    t = pa & 0xffff0000u; __builtin_memcpy(&fa1, &t, 4);
    t = pb << 16; __builtin_memcpy(&fb0, &t, 4);
    t = pb & 0xffff0000u; __builtin_memcpy(&fb1, &t, 4);
    return __builtin_fmaf(fa0, fb0, __builtin_fmaf(fa1, fb1, acc));
}
union pairs { bf16v8 v; bf16x2 p[4]; };
__device__ __forceinline__ float dot8(bf16v8 a, bf16v8 b, float acc) {
    pairs x{a}, y{b};
#pragma unroll
    for (int j = 0; j < 4; j++) acc = dot2(x.p[j], y.p[j], acc);
    return acc;
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int o = 32; o >= 1; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}

template <int UN, int LDSB>
__global__ __launch_bounds__(512, 2) void gemv_fp8(bf16* __restrict__ C, const bf16* __restrict__ x,
                                                   const unsigned char* __restrict__ W,
                                                   const float* __restrict__ wscale, unsigned N,
                                                   unsigned K) {
    __shared__ bf16 lds[LDSB / 2];
    const unsigned lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    for (unsigned i = threadIdx.x; i < K / 8; i += 512)
        *(bf16v8*)(lds + i * 8) = *(const bf16v8*)(x + i * 8);
    __syncthreads();

    const unsigned step = PLOW_WAVE * 16;
    const unsigned nchunk = (K + step - 1) / step;
    const unsigned per = (N + gridDim.x - 1) / gridDim.x;
    const unsigned n0 = blockIdx.x * per, n1 = (n0 + per < N) ? n0 + per : N;

    unsigned n = n0 + wave;
    for (; n + PLOW_WAVES < n1; n += PLOW_WAVES * 2) {
        const unsigned n2 = n + PLOW_WAVES;
        const __amdgpu_buffer_rsrc_t wr = buf_rsrc_fp8(W + (size_t)n * K, K);
        const __amdgpu_buffer_rsrc_t wr2 = buf_rsrc_fp8(W + (size_t)n2 * K, K);
        float acc = 0.f, acc2 = 0.f;
        for (unsigned c = 0; c < nchunk; c += UN) {
            fp8v16 wv[UN], wv2[UN];
#pragma unroll
            for (int u = 0; u < UN; u++) wv[u] = buf_ld_fp8(wr, (c + (unsigned)u) * step + lane * 16);
#pragma unroll
            for (int u = 0; u < UN; u++) wv2[u] = buf_ld_fp8(wr2, (c + (unsigned)u) * step + lane * 16);
#pragma unroll
            for (int u = 0; u < UN; u++) {
                const unsigned k = (c + (unsigned)u) * step + lane * 16;
                const unsigned kx = (k < K) ? k : 0u;
                bf16v8 wlo, whi, wlo2, whi2;
                fp8_to_bf16v8(wv[u], wlo, whi);
                fp8_to_bf16v8(wv2[u], wlo2, whi2);
                const bf16v8 xlo = *(const bf16v8*)(lds + kx);
                const bf16v8 xhi = *(const bf16v8*)(lds + kx + 8);
                acc  += dot8(whi,  xhi, dot8(wlo,  xlo, 0.0f));
                acc2 += dot8(whi2, xhi, dot8(wlo2, xlo, 0.0f));
            }
        }
        const float t = wave_sum(acc) * wscale[n], t2 = wave_sum(acc2) * wscale[n2];
        if (lane == 0) { C[n] = (bf16)t; C[n2] = (bf16)t2; }
    }
}

#define CK(e) do{hipError_t r=(e); if(r){printf("ERR %s\n", hipGetErrorString(r)); return 1;}}while(0)

template <int UN>
static double run(unsigned char** Wb, int nbuf, bf16* x, bf16* C, float* ws, unsigned N, unsigned K,
                  unsigned nblk, int iters) {
    for (int i = 0; i < 4; i++)
        hipLaunchKernelGGL((gemv_fp8<UN, 16384>), dim3(nblk), dim3(512), 0, 0, C, x, Wb[i%nbuf], ws, N, K);
    CK(hipDeviceSynchronize());
    hipEvent_t a, b; (void)hipEventCreate(&a); (void)hipEventCreate(&b);
    (void)hipEventRecord(a);
    for (int i = 0; i < iters; i++)
        hipLaunchKernelGGL((gemv_fp8<UN, 16384>), dim3(nblk), dim3(512), 0, 0, C, x, Wb[i%nbuf], ws, N, K);
    (void)hipEventRecord(b); CK(hipDeviceSynchronize());
    float ms; (void)hipEventElapsedTime(&ms, a, b);
    return (double)ms * 1e3 / iters;   // microseconds per op
}

int main() {
    struct { const char* name; unsigned N, K, b; double meas; } S[] = {
        {"q_proj  ", 4096, 3840, 152, 37.7}, {"k_proj  ", 2048, 3840,  76, 37.5},
        {"v_proj  ", 2048, 3840,  76, 37.6}, {"o_proj  ", 3840, 4096, 304, 35.5},
        {"down    ", 3840, 15360, 304, 46.1},
    };
    printf("%-9s %6s %6s %4s | %8s %8s %8s %8s | %9s %9s\n", "shape", "N", "K", "wg",
           "UN=2 us", "UN=3 us", "UN=4 us", "UN=5 us", "best GB/s", "megakern");
    for (auto s : S) {
        size_t rb = (size_t)s.N * s.K;
        int nbuf = (int)(3ull * (256ull << 20) / rb) + 1; if (nbuf > 24) nbuf = 24; if (nbuf < 2) nbuf = 2;
        bf16 *x, *C; float* ws; unsigned char* Wb[24];
        CK(hipMalloc(&x, s.K * 2)); CK(hipMalloc(&C, s.N * 2)); CK(hipMalloc(&ws, s.N * 4));
        CK(hipMemset(x, 0x3c, s.K * 2)); CK(hipMemset(ws, 0x3c, s.N * 4));
        for (int i = 0; i < nbuf; i++) { CK(hipMalloc(&Wb[i], rb)); CK(hipMemset(Wb[i], 0x38, rb)); }
        int iters = (int)(60e-3 / (rb / 1.0e12)); if (iters < 200) iters = 200;
        double u2 = run<2>(Wb, nbuf, x, C, ws, s.N, s.K, s.b, iters);
        double u3 = run<3>(Wb, nbuf, x, C, ws, s.N, s.K, s.b, iters);
        double u4 = run<4>(Wb, nbuf, x, C, ws, s.N, s.K, s.b, iters);
        double u5 = run<5>(Wb, nbuf, x, C, ws, s.N, s.K, s.b, iters);
        double best = u2; if (u3 < best) best = u3; if (u4 < best) best = u4; if (u5 < best) best = u5;
        printf("%-9s %6u %6u %4u | %8.2f %8.2f %8.2f %8.2f | %9.0f %8.1fus\n", s.name, s.N, s.K, s.b,
               u2, u3, u4, u5, rb / (best * 1e-6) / 1e9, s.meas);
        for (int i = 0; i < nbuf; i++) hipFree(Wb[i]);
        hipFree(x); hipFree(C); hipFree(ws);
    }
    return 0;
}
