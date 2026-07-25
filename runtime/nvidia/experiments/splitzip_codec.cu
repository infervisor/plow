// SplitZip bf16 decompressor throughput gate on RTX 5090 (sm_120a)
// Layout (SoA, as plow's compiler would emit):
//   lo[]    : 1 byte / element  = sign(bit7) | mantissa(bits6:0)
//   codes[] : 4 bits / element  = index into 16-entry exponent table
//   table   : 16 x u8 exponents, compile-time constant per tile
//   escapes : sparse (u32 pos, u16 raw) applied in a second pass
// Reconstruct: bf16 = ((lo&0x80)<<8) | (exp<<7) | (lo&0x7F)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__constant__ unsigned int c_tab[4];   // 16 exponent bytes packed into 4 u32
__constant__ unsigned char c_tab8[16];

// ---- table lookup variants, all divergence-free ----
enum { LUT_REG = 0, LUT_CONST = 1, LUT_SMEM = 2, LUT_AFFINE = 3 };

__device__ __forceinline__ unsigned int lut_reg(unsigned int c,
        unsigned int t0, unsigned int t1, unsigned int t2, unsigned int t3) {
    // select 1 of 4 words by c>>2 (two SEL), then byte by c&3 (one PRMT)
    unsigned int a = (c & 4u) ? t1 : t0;
    unsigned int b = (c & 4u) ? t3 : t2;
    unsigned int w = (c & 8u) ? b : a;
    return __byte_perm(w, 0u, c & 3u);
}

template<int LUT>
__device__ __forceinline__ unsigned int getexp(unsigned int c,
        unsigned int t0, unsigned int t1, unsigned int t2, unsigned int t3,
        const unsigned char* __restrict__ stab) {
    if (LUT == LUT_REG)    return lut_reg(c, t0, t1, t2, t3);
    if (LUT == LUT_CONST)  return c_tab8[c];
    if (LUT == LUT_SMEM)   return stab[c];
    return c + 109u;                     // measured: Qwen3-4B table is exps 109..124
}

// decode 16 elements: lo4 = 16 lo bytes, cd2 = 8 code bytes -> 16 u16 out
template<int LUT>
__device__ __forceinline__ void decode16(uint4 lo4, uint2 cd2, unsigned short* out,
        unsigned int t0, unsigned int t1, unsigned int t2, unsigned int t3,
        const unsigned char* __restrict__ stab) {
    unsigned int lw[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
    unsigned int cw[2] = {cd2.x, cd2.y};
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        unsigned int lo = (lw[e >> 2] >> ((e & 3) * 8)) & 0xFFu;
        unsigned int c  = (cw[e >> 3] >> ((e & 7) * 4)) & 0xFu;
        unsigned int ex = getexp<LUT>(c, t0, t1, t2, t3, stab);
        out[e] = (unsigned short)(((lo & 0x80u) << 8) | (ex << 7) | (lo & 0x7Fu));
    }
}

// ---------- kernel 1: pure compressed read ceiling (no decode) ----------
__global__ __launch_bounds__(512) void k_read(const uint4* __restrict__ lo,
        const uint2* __restrict__ cd, size_t nvec, unsigned int* sink) {
    unsigned int a = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nvec; i += stride) {
        uint4 l = lo[i]; uint2 c = cd[i];
        a += l.x ^ l.y ^ l.z ^ l.w ^ c.x ^ c.y;
    }
    if (a == 0xDEADBEEFu) sink[blockIdx.x] = a;
}

// ---------- kernel 2: decompress -> registers (plow mode, no HBM writeback) ----------
template<int LUT>
__global__ __launch_bounds__(512) void k_sink(const uint4* __restrict__ lo,
        const uint2* __restrict__ cd, size_t nvec, unsigned int* sink) {
    __shared__ unsigned char stab[16];
    if (threadIdx.x < 16) stab[threadIdx.x] = c_tab8[threadIdx.x];
    __syncthreads();
    unsigned int t0 = c_tab[0], t1 = c_tab[1], t2 = c_tab[2], t3 = c_tab[3];
    unsigned int a = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nvec; i += stride) {
        uint4 l = lo[i]; uint2 c = cd[i];
        unsigned short o[16];
        decode16<LUT>(l, c, o, t0, t1, t2, t3, stab);
#pragma unroll
        for (int e = 0; e < 16; ++e) a += o[e];   // stands in for the GEMV FMA
    }
    if (a == 0xDEADBEEFu) sink[blockIdx.x] = a;
}

// ---------- kernel 3: decompress -> HBM (paper mode) ----------
template<int LUT>
__global__ __launch_bounds__(512) void k_store(const uint4* __restrict__ lo,
        const uint2* __restrict__ cd, size_t nvec, uint4* __restrict__ out) {
    __shared__ unsigned char stab[16];
    if (threadIdx.x < 16) stab[threadIdx.x] = c_tab8[threadIdx.x];
    __syncthreads();
    unsigned int t0 = c_tab[0], t1 = c_tab[1], t2 = c_tab[2], t3 = c_tab[3];
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nvec; i += stride) {
        uint4 l = lo[i]; uint2 c = cd[i];
        unsigned short o[16];
        decode16<LUT>(l, c, o, t0, t1, t2, t3, stab);
        out[2 * i]     = *(uint4*)&o[0];
        out[2 * i + 1] = *(uint4*)&o[8];
    }
}

// ---------- sparse escape overwrite ----------
__global__ void k_escape(const unsigned int* __restrict__ pos,
        const unsigned short* __restrict__ val, unsigned int n,
        unsigned short* __restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[pos[i]] = val[i];
}

// ---------------- host ----------------
static double bench(void (*launch)(int, int, void**), int grid, int blk, void** args, int iters) {
    (void)launch; (void)grid; (void)blk; (void)args; (void)iters; return 0;
}

struct Cfg { int lut; int blk; int grid; };
static const char* lutname(int l) {
    return l == LUT_REG ? "reg(prmt)" : l == LUT_CONST ? "__constant__" : l == LUT_SMEM ? "smem" : "affine";
}

int main(int argc, char** argv) {
    size_t logical_mb = argc > 1 ? atoll(argv[1]) : 512;
    size_t nelem = logical_mb * 1024 * 1024 / 2;      // bf16 elements
    size_t nvec  = nelem / 16;                        // 16 elements per thread-iteration
    size_t lo_b  = nelem;                             // 1 B/elem
    size_t cd_b  = nelem / 2;                         // 0.5 B/elem
    printf("logical=%zu MB  elements=%zu  compressed(lo+codes)=%.1f MB  ratio=%.4f\n",
           logical_mb, nelem, (lo_b + cd_b) / 1048576.0, 2.0 / 1.5);

    unsigned char *d_lo, *d_cd; unsigned int* d_sink; unsigned short* d_out;
    CK(cudaMalloc(&d_lo, lo_b)); CK(cudaMalloc(&d_cd, cd_b));
    CK(cudaMalloc(&d_sink, 1 << 20)); CK(cudaMalloc(&d_out, nelem * 2));

    // fill with a plausible pattern
    unsigned char* h = (unsigned char*)malloc(lo_b);
    for (size_t i = 0; i < lo_b; ++i) h[i] = (unsigned char)(i * 1103515245u >> 13);
    CK(cudaMemcpy(d_lo, h, lo_b, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cd, h, cd_b, cudaMemcpyHostToDevice));
    free(h);

    unsigned char tab[16]; for (int i = 0; i < 16; ++i) tab[i] = 109 + i;
    CK(cudaMemcpyToSymbol(c_tab8, tab, 16));
    CK(cudaMemcpyToSymbol(c_tab, tab, 16));

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int ITER = 20;
    double comp_gb = (lo_b + cd_b) / 1e9, log_gb = nelem * 2 / 1e9;

    printf("\n%-14s %-6s %-6s %6s %12s %12s %12s\n", "kernel", "lut", "blk", "grid",
           "ms", "comp GB/s", "LOGICAL GB/s");

    auto run = [&](const char* nm, int lut, int blk, int grid, int mode) {
        for (int w = 0; w < 2; ++w) {   // warmup + timed
            if (w) CK(cudaEventRecord(e0));
            for (int it = 0; it < (w ? ITER : 3); ++it) {
                if (mode == 0) k_read<<<grid, blk>>>((uint4*)d_lo, (uint2*)d_cd, nvec, d_sink);
                else if (mode == 1) {
                    switch (lut) {
                    case 0: k_sink<0><<<grid, blk>>>((uint4*)d_lo, (uint2*)d_cd, nvec, d_sink); break;
                    case 1: k_sink<1><<<grid, blk>>>((uint4*)d_lo, (uint2*)d_cd, nvec, d_sink); break;
                    case 2: k_sink<2><<<grid, blk>>>((uint4*)d_lo, (uint2*)d_cd, nvec, d_sink); break;
                    default: k_sink<3><<<grid, blk>>>((uint4*)d_lo, (uint2*)d_cd, nvec, d_sink); break; }
                } else {
                    switch (lut) {
                    case 0: k_store<0><<<grid, blk>>>((uint4*)d_lo, (uint2*)d_cd, nvec, (uint4*)d_out); break;
                    case 1: k_store<1><<<grid, blk>>>((uint4*)d_lo, (uint2*)d_cd, nvec, (uint4*)d_out); break;
                    case 2: k_store<2><<<grid, blk>>>((uint4*)d_lo, (uint2*)d_cd, nvec, (uint4*)d_out); break;
                    default: k_store<3><<<grid, blk>>>((uint4*)d_lo, (uint2*)d_cd, nvec, (uint4*)d_out); break; }
                }
            }
            if (w) { CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); } CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
        }
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= ITER;
        printf("%-14s %-6s %-6d %6d %12.4f %12.1f %12.1f\n", nm, lutname(lut), blk, grid,
               ms, comp_gb / (ms / 1e3), log_gb / (ms / 1e3));
        return log_gb / (ms / 1e3);
    };

    // pure read ceiling for this access pattern
    for (int blk : {128, 256, 512})
        for (int g : {170, 340, 680, 1020, 2040})
            run("read-only", 0, blk, g, 0);

    printf("\n-- decompress -> registers (plow fused mode, no writeback) --\n");
    for (int lut = 0; lut < 4; ++lut)
        for (int blk : {128, 256, 512})
            for (int g : {170, 340, 680, 1020, 2040})
                run("sink", lut, blk, g, 1);

    printf("\n-- decompress -> HBM (paper mode; logical bytes also WRITTEN) --\n");
    for (int lut = 0; lut < 4; ++lut)
        for (int g : {340, 680, 1020, 2040})
            run("store", lut, 256, g, 2);
    return 0;
}
