// Does the decode GEMV actually go faster at 4 waves/SIMD than at 2?
//
// The megakernel pins interp_decode_gq to 2 waves/SIMD twice over -- LDS 64,520 B of 65,536
// (one workgroup per CU) and VGPR 253 -- and BOTH numbers are unions over every op in the
// dispatch switch, not the GEMV's own need. Before paying for resource-class segmentation in
// the emitter, prove the premise: the SAME inner loop, the SAME shapes, the SAME 304x512 grid,
// varying only the two things the megakernel forces on it.
//
// CAVEAT: the LDSB=16384 and 8192 rows OVERFLOW for K=15360 (x staging needs 30,720 B), so the
// `down 3840x15360` numbers in those two columns are INVALID -- the kernel writes past `lds`.
// The K<=4096 shapes are unaffected. Kept as measured; the 32 KB column is the one that matches
// what a real decode object can reach, because M*K must still fit for the staged-x arm.
//
// WORKING SET > INFINITY CACHE. A single 31 MB weight fits in MI300X's 256 MiB, and a repeat
// loop over one buffer measures cache bandwidth (which is how plow's own decode-GEMV rung reads
// 2.9 TB/s). NBUF is chosen so the rotation exceeds it ~3x, reproducing the real decode's
// stream-once behaviour.
#include <hip/hip_runtime.h>
#include <cstdio>

typedef __bf16 bf16;
typedef bf16 bf16v8 __attribute__((ext_vector_type(8)));
typedef bf16 bf16x2 __attribute__((ext_vector_type(2)));
#define PLOW_BUF_RSRC3 0x00020000u

__device__ __forceinline__ bf16v8 buf_ld8(__amdgpu_buffer_rsrc_t r, unsigned off) {
    return __builtin_bit_cast(bf16v8, __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, 3));
}
__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(const bf16* b, unsigned nh) {
    unsigned long long u = (unsigned long long)(size_t)b;
    unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)u);
    unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(u >> 32));
    return __builtin_amdgcn_make_buffer_rsrc(
        (void*)(size_t)(((unsigned long long)hi << 32) | lo), (short)0, nh * 2u, PLOW_BUF_RSRC3);
}
// The gfx942 fallback, verbatim: no v_dot2c_f32_bf16, so widen and FMA.
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
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 32; o >= 1; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}

// LDSB = the LDS the megakernel's union forces on this kernel. WPE = requested waves/EU.
template <int LDSB, int WPE>
__global__ __launch_bounds__(512, WPE) void gemv(bf16* __restrict__ C, const bf16* __restrict__ x,
                                                 const bf16* __restrict__ W, unsigned N,
                                                 unsigned K) {
    __shared__ bf16 lds[LDSB / 2];
    const unsigned lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    for (unsigned i = threadIdx.x; i < K / 8; i += 512)
        *(bf16v8*)(lds + i * 8) = *(const bf16v8*)(x + i * 8);
    __syncthreads();
    const unsigned step = 512, nch = (K + step - 1) / step;
    const unsigned per = (N + gridDim.x - 1) / gridDim.x;
    const unsigned n0 = blockIdx.x * per, n1 = (n0 + per < N) ? n0 + per : N;
    for (unsigned n = n0 + wave; n < n1; n += 8) {
        const __amdgpu_buffer_rsrc_t wr = rsrc(W + (size_t)n * K, K);
        float acc = 0.f;
        for (unsigned c = 0; c < nch; c += 8) {
            bf16v8 wv[8];
#pragma unroll
            for (int u = 0; u < 8; u++) wv[u] = buf_ld8(wr, ((c + u) * step + lane * 8) * 2u);
#pragma unroll
            for (int u = 0; u < 8; u++) {
                unsigned k = (c + u) * step + lane * 8;
                acc = dot8(wv[u], *(const bf16v8*)(lds + ((k < K) ? k : 0)), acc);
            }
        }
        float t = wsum(acc);
        if (lane == 0) C[n] = (bf16)t;
    }
}

#define CK(e) do{hipError_t r=(e); if(r){printf("ERR %s\n", hipGetErrorString(r)); return 1;}}while(0)

template <int LDSB, int WPE>
static double run(bf16** Wb, int nbuf, bf16* x, bf16* C, unsigned N, unsigned K, int iters) {
    for (int i = 0; i < 4; i++) hipLaunchKernelGGL((gemv<LDSB, WPE>), dim3(304), dim3(512), 0, 0, C, x, Wb[i % nbuf], N, K);
    CK(hipDeviceSynchronize());
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a);
    for (int i = 0; i < iters; i++)
        hipLaunchKernelGGL((gemv<LDSB, WPE>), dim3(304), dim3(512), 0, 0, C, x, Wb[i % nbuf], N, K);
    hipEventRecord(b); CK(hipDeviceSynchronize());
    float ms; hipEventElapsedTime(&ms, a, b);
    return (double)N * K * 2.0 * iters / (ms / 1e3) / 1e9;
}

int main() {
    struct { const char* name; unsigned N, K; } S[] = {
        {"o_proj  ", 3840, 4096}, {"qkv     ", 4096, 3840},
        {"gate/up ", 15360, 3840}, {"down    ", 3840, 15360},
    };
    printf("%-9s %7s %7s | %8s %8s %8s %8s %8s\n", "shape", "N", "K", "64.5KB", "41KB", "32KB", "16KB", "8KB");
    for (auto s : S) {
        size_t rb = (size_t)s.N * s.K * 2;
        int nbuf = (int)(3ull * (256ull << 20) / rb) + 1; if (nbuf > 20) nbuf = 20; if (nbuf < 2) nbuf = 2;
        bf16 *x, *C, *Wb[20];
        CK(hipMalloc(&x, s.K * 2)); CK(hipMalloc(&C, s.N * 2));
        CK(hipMemset(x, 0x3c, s.K * 2));
        for (int i = 0; i < nbuf; i++) { CK(hipMalloc(&Wb[i], rb)); CK(hipMemset(Wb[i], 0x3c, rb)); }
        int iters = (int)(40e-3 / (rb / 2.0e12)); if (iters < 30) iters = 30;
        double a = run<64520, 2>(Wb, nbuf, x, C, s.N, s.K, iters);  // today's interp_decode_gq
        double b = run<41024, 2>(Wb, nbuf, x, C, s.N, s.K, iters);  // dropping gm[] and fa[]: fd[] sets it
        double c = run<32768, 2>(Wb, nbuf, x, C, s.N, s.K, iters);  // the 2-workgroups-per-CU threshold
        double d = run<16384, 2>(Wb, nbuf, x, C, s.N, s.K, iters);
        double e = run<8192, 2>(Wb, nbuf, x, C, s.N, s.K, iters);
        printf("%-9s %7u %7u | %8.0f %8.0f %8.0f %8.0f %8.0f\n", s.name, s.N, s.K, a, b, c, d, e);
        for (int i = 0; i < nbuf; i++) hipFree(Wb[i]);
        hipFree(x); hipFree(C);
    }
    return 0;
}
