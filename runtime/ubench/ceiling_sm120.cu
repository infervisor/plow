/* ceiling_sm120.cu — single-SM ceilings for sm_12x (GB10 = sm_121a) and the harness utilities the
 * NVIDIA path of bench_cu_roofline.c uses (fill, L2 flush, snapshot copy, output hash).
 *
 * Every ceiling kernel stamps clock64 and %globaltimer per block (CeilStamp), so the host reads
 * cycles and nanoseconds of the timed region without launch overhead. Launched with gridDim=1 it
 * is the one-SM ceiling; with gridDim=n_sm (one block per SM, forced by the dynamic smem request)
 * the host checks the same kernel against the whole-GPU number.
 *
 *   nvcc -arch=sm_121a -O3 -cubin ceiling_sm120.cu -o ceiling_sm121a.cubin
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct CeilStamp {
    unsigned long long c0, c1, t0, t1;
};

static __device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __device__ __forceinline__ void stamp_begin(CeilStamp* st) {
    __syncthreads();
    if (threadIdx.x == 0) {
        st[blockIdx.x].t0 = gtime();
        st[blockIdx.x].c0 = clock64();
    }
    __syncthreads();
}

static __device__ __forceinline__ void stamp_end(CeilStamp* st) {
    __syncthreads();
    if (threadIdx.x == 0) {
        st[blockIdx.x].c1 = clock64();
        st[blockIdx.x].t1 = gtime();
    }
}

/* ---- tensor-core issue rate ------------------------------------------------------------------
 * Each warp runs CEIL_NACC independent accumulator chains; FLOPs per instruction: bf16 m16n8k16
 * 4096, e4m3 m16n8k32 8192, NVFP4/MXFP4 m16n8k64 16384. */
#define CEIL_NACC 8

#define MMA_BF16(d, a, b)                                                                          \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                           \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                         \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                  \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
#define MMA_E4M3(d, a, b)                                                                          \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "                           \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                         \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                  \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
#define MMA_F8F6F4(d, a, b)                                                                        \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "              \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                         \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                  \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
#define MMA_MXF8(d, a, b, sa, sb)                                                                  \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X."   \
                 "f32.e4m3.e4m3.f32.ue8m0 "                                                        \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n" \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                  \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa),      \
                   "r"(sb))
#define MMA_NVFP4(d, a, b, sa, sb)                                                                 \
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."   \
                 "f32.e2m1.e2m1.f32.ue4m3 "                                                        \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n" \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                  \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa),      \
                   "r"(sb))
#define MMA_MXF4(d, a, b, sa, sb)                                                                  \
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X."       \
                 "f32.e2m1.e2m1.f32.ue8m0 "                                                        \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\n" \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                  \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa),      \
                   "r"(sb))

/* kind: 0 bf16, 1 e4m3, 2 e4m3 kind::f8f6f4, 3 MXFP8 (ue8m0 1X), 4 NVFP4 (ue4m3 4X), 5 MXFP4. */
extern "C" __global__ void __launch_bounds__(1024) ceil_mma(CeilStamp* st, float* sink, unsigned kind,
                                                            unsigned iters, unsigned seed) {
    float d[CEIL_NACC][4];
#pragma unroll
    for (int j = 0; j < CEIL_NACC; j++) d[j][0] = d[j][1] = d[j][2] = d[j][3] = 0.0f;
    /* Small finite operands in every format: 0x38/0x30 bytes are 1.0/0.5 in e4m3 and modest
     * e2m1 pairs; as bf16 halves 0x3830 is ~4e-5. */
    const unsigned v = 0x38303830u ^ (seed & 0x01010101u);
    unsigned a[4] = {v, v ^ 0x01000000u, v ^ 0x00010000u, v ^ 0x00000100u};
    unsigned b[2] = {v ^ 0x00000001u, v};
    const unsigned sa = 0x7F7F7F7Fu, sb = 0x7F7F7F7Fu; /* ue8m0 1.0 */
    const unsigned qa = 0x38383838u, qb = 0x38383838u; /* ue4m3 1.0 */
    stamp_begin(st);
    switch (kind) {
    case 0:
        for (unsigned i = 0; i < iters; i++)
#pragma unroll
            for (int j = 0; j < CEIL_NACC; j++) MMA_BF16(d[j], a, b);
        break;
    case 1:
        for (unsigned i = 0; i < iters; i++)
#pragma unroll
            for (int j = 0; j < CEIL_NACC; j++) MMA_E4M3(d[j], a, b);
        break;
    case 2:
        for (unsigned i = 0; i < iters; i++)
#pragma unroll
            for (int j = 0; j < CEIL_NACC; j++) MMA_F8F6F4(d[j], a, b);
        break;
    case 3:
        for (unsigned i = 0; i < iters; i++)
#pragma unroll
            for (int j = 0; j < CEIL_NACC; j++) MMA_MXF8(d[j], a, b, sa, sb);
        break;
    case 4:
        for (unsigned i = 0; i < iters; i++)
#pragma unroll
            for (int j = 0; j < CEIL_NACC; j++) MMA_NVFP4(d[j], a, b, qa, qb);
        break;
    default:
        for (unsigned i = 0; i < iters; i++)
#pragma unroll
            for (int j = 0; j < CEIL_NACC; j++) MMA_MXF4(d[j], a, b, sa, sb);
        break;
    }
    stamp_end(st);
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < CEIL_NACC; j++) s += d[j][0] + d[j][1] + d[j][2] + d[j][3];
    if (s == 1.2345f) sink[threadIdx.x] = s;
}

/* ---- FP32 FMA issue rate: 8 independent chains per thread, 2 FLOPs per FFMA ---------------- */
extern "C" __global__ void __launch_bounds__(1024) ceil_ffma(CeilStamp* st, float* sink, unsigned iters,
                                                             float x) {
    float c[8];
#pragma unroll
    for (int j = 0; j < 8; j++) c[j] = x * (float)(threadIdx.x + j);
    const float m = 0.999f + x * 1e-7f, k = 1e-6f * x;
    stamp_begin(st);
    for (unsigned i = 0; i < iters; i++)
#pragma unroll
        for (int r = 0; r < 16; r++)
#pragma unroll
            for (int j = 0; j < 8; j++) c[j] = fmaf(c[j], m, k);
    stamp_end(st);
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) s += c[j];
    if (s == 1.2345f) sink[threadIdx.x] = s;
}

/* ---- streaming read: one block walks [lo, hi) of buf (bytes, 16 B aligned) `passes` times ---
 * mode 0: ld.global.v4 (LDG.128)   1: ld.global.nc.L1::no_allocate.v4
 *      2: ld.global.v8.b32 (LDG.256, 32 B per lane)
 * gridDim>1 splits the buffer evenly across blocks. Unroll 8 = 128 (256) B in flight per lane. */
static __device__ __forceinline__ uint4 ld_v4(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
static __device__ __forceinline__ uint4 ld_v4_na(const uint4* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
static __device__ __forceinline__ uint4 ld_v8_fold(const uint4* p) {
    unsigned r0, r1, r2, r3, r4, r5, r6, r7;
    asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
                 : "l"(p));
    return make_uint4(r0 ^ r4, r1 ^ r5, r2 ^ r6, r3 ^ r7);
}

template <int MODE>
static __device__ __forceinline__ uint4 ld_mode(const uint4* p) {
    return MODE == 0 ? ld_v4(p) : MODE == 1 ? ld_v4_na(p) : ld_v8_fold(p);
}

template <int MODE>
static __device__ __forceinline__ uint4 stream_body(const uint4* base, unsigned long long n, unsigned passes) {
    const unsigned vec = MODE == 2 ? 2u : 1u; /* uint4 units per lane per load */
    const unsigned long long step = (unsigned long long)blockDim.x * vec;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (unsigned p = 0; p < passes; p++) {
        unsigned long long i = (unsigned long long)threadIdx.x * vec;
        for (; i + 7 * step + vec <= n; i += 8 * step) {
            uint4 r[8];
#pragma unroll
            for (int u = 0; u < 8; u++) r[u] = ld_mode<MODE>(base + i + u * step);
#pragma unroll
            for (int u = 0; u < 8; u++) {
                acc.x ^= r[u].x; acc.y ^= r[u].y; acc.z ^= r[u].z; acc.w ^= r[u].w;
            }
        }
        for (; i + vec <= n; i += step) {
            const uint4 r = ld_mode<MODE>(base + i);
            acc.x ^= r.x; acc.y ^= r.y; acc.z ^= r.z; acc.w ^= r.w;
        }
    }
    return acc;
}

extern "C" __global__ void __launch_bounds__(1024) ceil_stream(CeilStamp* st, unsigned* sink,
                                                               const uint4* buf, unsigned long long bytes,
                                                               unsigned mode, unsigned passes) {
    const unsigned long long per = (bytes / gridDim.x) & ~4095ull;
    const uint4* base = buf + (per / 16) * blockIdx.x;
    stamp_begin(st);
    const uint4 acc = mode == 0 ? stream_body<0>(base, per / 16, passes)
                      : mode == 1 ? stream_body<1>(base, per / 16, passes)
                                  : stream_body<2>(base, per / 16, passes);
    stamp_end(st);
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u) sink[threadIdx.x] = acc.x;
}

/* ---- cp.async.cg 16 B ring into shared memory (LDGSTS), CEIL_CPA_STAGES x tile bytes ---------- */
#define CEIL_CPA_STAGES 4
extern "C" __global__ void __launch_bounds__(1024) ceil_cpasync(CeilStamp* st, unsigned* sink,
                                                                const uint4* buf, unsigned long long bytes,
                                                                unsigned tile) {
    extern __shared__ __align__(16) unsigned char smem[];
    const unsigned stages = CEIL_CPA_STAGES;
    const unsigned long long per = (bytes / gridDim.x) & ~((unsigned long long)tile - 1);
    const unsigned char* base = (const unsigned char*)buf + per * blockIdx.x;
    const unsigned long long ntiles = per / tile;
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    stamp_begin(st);
    unsigned long long t = 0;
    for (unsigned s = 0; s < stages - 1 && t < ntiles; s++, t++) {
        for (unsigned o = threadIdx.x * 16; o < tile; o += blockDim.x * 16)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sbase + s * tile + o),
                         "l"(base + t * tile + o));
        asm volatile("cp.async.commit_group;\n" ::);
    }
    unsigned acc = 0;
    for (unsigned long long c = 0; c < ntiles; c++, t++) {
        if (t < ntiles) {
            const unsigned s = (unsigned)(t % stages);
            for (unsigned o = threadIdx.x * 16; o < tile; o += blockDim.x * 16)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sbase + s * tile + o),
                             "l"(base + t * tile + o));
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(CEIL_CPA_STAGES - 1));
        __syncthreads();
        acc ^= ((const unsigned*)(smem + (c % stages) * tile))[threadIdx.x & 63];
        __syncthreads();
    }
    asm volatile("cp.async.wait_all;\n" ::);
    stamp_end(st);
    if (acc == 0x9E3779B9u) sink[threadIdx.x] = acc;
}

/* ---- bulk async copy (cp.async.bulk, the TMA engine's 1-D form) + mbarrier ring -------------
 * Thread 0 keeps `stages` tile-sized bulk copies in flight; the block waits on each stage. */
static __device__ __forceinline__ void mbar_init(unsigned a, unsigned n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(n));
}
static __device__ __forceinline__ void mbar_expect(unsigned a, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes));
}
static __device__ __forceinline__ void mbar_wait(unsigned a, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 " @!p bra W;\n}\n" ::"r"(a), "r"(parity));
}
static __device__ __forceinline__ void bulk_g2s(unsigned dst, const void* src, unsigned bytes, unsigned bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

extern "C" __global__ void __launch_bounds__(1024) ceil_bulk(CeilStamp* st, unsigned* sink,
                                                             const uint4* buf, unsigned long long bytes,
                                                             unsigned tile, unsigned stages) {
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ __align__(8) unsigned long long bar[16];
    const unsigned long long per = (bytes / gridDim.x) & ~((unsigned long long)tile - 1);
    const unsigned char* base = (const unsigned char*)buf + per * blockIdx.x;
    const unsigned long long ntiles = per / tile;
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned bbase = (unsigned)__cvta_generic_to_shared(bar);
    if (threadIdx.x == 0) {
        for (unsigned s = 0; s < stages; s++) mbar_init(bbase + 8 * s, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
    }
    __syncthreads();
    stamp_begin(st);
    if (threadIdx.x == 0)
        for (unsigned s = 0; s < stages && s < ntiles; s++) {
            mbar_expect(bbase + 8 * s, tile);
            bulk_g2s(sbase + s * tile, base + (unsigned long long)s * tile, tile, bbase + 8 * s);
        }
    unsigned acc = 0;
    for (unsigned long long c = 0; c < ntiles; c++) {
        const unsigned s = (unsigned)(c % stages);
        mbar_wait(bbase + 8 * s, (unsigned)((c / stages) & 1));
        acc ^= ((const unsigned*)(smem + s * tile))[threadIdx.x & 63];
        __syncthreads();
        if (threadIdx.x == 0 && c + stages < ntiles) {
            mbar_expect(bbase + 8 * s, tile);
            bulk_g2s(sbase + s * tile, base + (c + stages) * tile, tile, bbase + 8 * s);
        }
    }
    stamp_end(st);
    if (acc == 0x9E3779B9u) sink[threadIdx.x] = acc;
}

/* ---- shared-memory read bandwidth: conflict-free ld.shared.v4 ------------------------------- */
extern "C" __global__ void __launch_bounds__(1024) ceil_smem(CeilStamp* st, unsigned* sink, unsigned iters) {
    extern __shared__ __align__(16) uint4 sm4[];
    const unsigned n4 = 2048; /* 32 KiB */
    for (unsigned i = threadIdx.x; i < n4; i += blockDim.x) sm4[i] = make_uint4(i, i * 3u, i * 5u, i * 7u);
    uint4 acc = make_uint4(0, 0, 0, 0);
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(sm4);
    stamp_begin(st);
    for (unsigned it = 0; it < iters; it++)
#pragma unroll
        for (unsigned u = 0; u < 16; u++) {
            const unsigned idx = (threadIdx.x + u * 128u + it) & (n4 - 1);
            uint4 r;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(sbase + idx * 16u));
            acc.x ^= r.x; acc.y ^= r.y; acc.z ^= r.z; acc.w ^= r.w;
        }
    stamp_end(st);
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u) sink[threadIdx.x] = acc.x;
}

/* ===================================================================================== *
 * Harness utilities (bench_cu_roofline.c NVIDIA path)
 * ===================================================================================== */

/* Fill kinds. Values are finite and in-range for the dtype so quantizers, softplus and top-k see
 * ordinary inputs; timing does not depend on them. */
enum {
    UB_FILL_ZERO = 0,
    UB_FILL_U8 = 1,      /* any byte (packed e2m1)                                  */
    UB_FILL_E4M3 = 2,    /* e4m3 in [-2, 2], no NaN                                 */
    UB_FILL_UE4M3 = 3,   /* positive e4m3 scale bytes in [0.5, 1.75]                */
    UB_FILL_BF16 = 4,    /* uniform [b - a, b + a]                                  */
    UB_FILL_F16 = 5,
    UB_FILL_F32 = 6,
    UB_FILL_I32 = 7,     /* constant a                                              */
    UB_FILL_I32_DIV = 8, /* i / b (iota at b = 1)                                   */
    UB_FILL_I32_MOD = 9, /* hash mod b                                              */
};

static __device__ __forceinline__ unsigned ub_hash(unsigned long long i, unsigned seed) {
    unsigned long long x = i * 0x9E3779B97F4A7C15ull + seed * 0xBF58476D1CE4E5B9ull + 0x94D049BB133111EBull;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29; x *= 0x94D049BB133111EBull; x ^= x >> 32;
    return (unsigned)x;
}
static __device__ __forceinline__ float ub_unif(unsigned long long i, unsigned seed) {
    return (float)(ub_hash(i, seed) >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

extern "C" __global__ void ub_fill(void* p, unsigned long long bytes, unsigned kind, unsigned seed,
                                   float a, float b) {
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long t0 = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    switch (kind) {
    case UB_FILL_ZERO:
    case UB_FILL_U8:
    case UB_FILL_E4M3:
    case UB_FILL_UE4M3: {
        unsigned char* q = (unsigned char*)p;
        for (unsigned long long i = t0; i < bytes; i += stride) {
            const unsigned h = ub_hash(i, seed);
            unsigned char v = 0;
            if (kind == UB_FILL_U8) v = (unsigned char)h;
            /* exponent field 4..8 (0.03..2), any mantissa, either sign */
            else if (kind == UB_FILL_E4M3) v = (unsigned char)((h & 0x80u) | ((4u + (h >> 8) % 5u) << 3) | (h & 7u));
            else if (kind == UB_FILL_UE4M3) v = (unsigned char)(0x30u + (h >> 4) % 14u);
            q[i] = v;
        }
        break;
    }
    case UB_FILL_BF16: {
        __nv_bfloat16* q = (__nv_bfloat16*)p;
        for (unsigned long long i = t0; i < bytes / 2; i += stride) q[i] = __float2bfloat16(b + a * ub_unif(i, seed));
        break;
    }
    case UB_FILL_F16: {
        __half* q = (__half*)p;
        for (unsigned long long i = t0; i < bytes / 2; i += stride) q[i] = __float2half(b + a * ub_unif(i, seed));
        break;
    }
    case UB_FILL_F32: {
        float* q = (float*)p;
        for (unsigned long long i = t0; i < bytes / 4; i += stride) q[i] = b + a * ub_unif(i, seed);
        break;
    }
    default: {
        int* q = (int*)p;
        const unsigned m = b > 0.0f ? (unsigned)b : 1u;
        for (unsigned long long i = t0; i < bytes / 4; i += stride)
            q[i] = kind == UB_FILL_I32 ? (int)a : kind == UB_FILL_I32_DIV ? (int)(i / m)
                                                                          : (int)(ub_hash(i, seed) % m);
        break;
    }
    }
}

/* Evict L2: read-modify-write a buffer several times its size. */
extern "C" __global__ void ub_flush(uint4* buf, unsigned long long n16) {
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        uint4 v = buf[i];
        v.x += 1u;
        buf[i] = v;
    }
}

extern "C" __global__ void ub_copy(unsigned char* dst, const unsigned char* src, unsigned long long bytes) {
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long t0 = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long n16 = bytes / 16;
    for (unsigned long long i = t0; i < n16; i += stride) ((uint4*)dst)[i] = ((const uint4*)src)[i];
    for (unsigned long long i = n16 * 16 + t0; i < bytes; i += stride) dst[i] = src[i];
}

/* Order-independent 64-bit digest of a byte range (sum of position-mixed words). */
extern "C" __global__ void ub_digest(unsigned long long* out, const unsigned char* p, unsigned long long bytes) {
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long s = 0;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 4; i += stride)
        s += (unsigned long long)ub_hash(((const unsigned*)p)[i] ^ (i << 32), 7u) * (2ull * i + 1ull);
    for (unsigned long long i = bytes / 4 * 4 + threadIdx.x + blockIdx.x * blockDim.x; i < bytes; i += stride)
        s += (unsigned long long)p[i] * (i + 1ull);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    if ((threadIdx.x & 31u) == 0) atomicAdd(out, s);
}

/* Bytes that differ between two ranges. */
extern "C" __global__ void ub_diff(unsigned long long* out, const unsigned char* a, const unsigned char* b,
                                   unsigned long long bytes) {
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long n = 0;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < bytes; i += stride)
        n += a[i] != b[i];
    for (int o = 16; o; o >>= 1) n += __shfl_xor_sync(0xFFFFFFFFu, n, o);
    if ((threadIdx.x & 31u) == 0) atomicAdd(out, n);
}
