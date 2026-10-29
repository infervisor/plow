// Page-gather dequantizer in front of a library attention kernel: plowrt's FP8 KV (e4m3, one f32
// scale per (position, KV head) row, head-major [slot][kv head][kv_stride][hd]) -> a bf16 page
// pool [kv head][pages][64][hd] (bf16, or fp16 for an fp16 library) the library reads through a
// page table. One CTA per (page, KV
// head, K|V); rows past the request's kv_len are zero (the library loads whole pages and masks
// their scores, and 0 * Inf would poison P.V). The source loads stop at kv_len: under live KV the
// slot rows past it may be unmapped.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

extern "C" __device__ unsigned plow_kv_dequant_abi = 2;

// Mirrored by plowrt exec/gpu/lib_attention.rs `DequantArgs`.
struct PlowKvDequant {
    const uint8_t* k8;
    const float* ks;
    const uint8_t* v8;
    const float* vs;
    void* ko;
    void* vo;
    const int* pages;  // [n_pages][3]: slot, first source row, valid rows (1..64)
    unsigned n_pages, kv_heads, kv_stride, pool_pages;  // pool_pages: one KV head's page count
    unsigned f16;                                       // pool element: 0 bf16, 1 fp16
};

constexpr int PAGE = 64;
constexpr int HD = 512;
constexpr int THREADS = 256;

__device__ __forceinline__ float e4m3(uint8_t b) {
    __nv_fp8_storage_t s = b;
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3)));
}

extern "C" __global__ __launch_bounds__(THREADS) void plow_kv_dequant_pages(
    const __grid_constant__ PlowKvDequant a) {
    const unsigned page = blockIdx.x, head = blockIdx.y, which = blockIdx.z;
    if (page >= a.n_pages || head >= a.kv_heads) return;
    const int slot = a.pages[3 * page], row0 = a.pages[3 * page + 1], valid = a.pages[3 * page + 2];
    const uint8_t* src = which ? a.v8 : a.k8;
    const float* scl = which ? a.vs : a.ks;
    const size_t dst_off = ((size_t)head * a.pool_pages + page) * PAGE * HD;
    uint16_t* dst = static_cast<uint16_t*>(which ? a.vo : a.ko) + dst_off;
    const size_t base_row = ((size_t)slot * a.kv_heads + head) * a.kv_stride + row0;
    // 16 B (16 e4m3) per thread step: a row is 32 steps, a page 2048.
    for (unsigned i = threadIdx.x; i < PAGE * HD / 16; i += THREADS) {
        const unsigned r = i / (HD / 16), c = (i % (HD / 16)) * 16;
        __align__(16) uint16_t out[16] = {};
        if ((int)r < valid) {
            const uint4 raw = *reinterpret_cast<const uint4*>(src + (base_row + r) * HD + c);
            const float s = scl[base_row + r];
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
            if (a.f16) {
#pragma unroll
                for (int j = 0; j < 16; ++j) out[j] = __half_as_ushort(__float2half_rn(e4m3(b[j]) * s));
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j) out[j] = __bfloat16_as_ushort(__float2bfloat16(e4m3(b[j]) * s));
            }
        }
        uint4* d = reinterpret_cast<uint4*>(dst + (size_t)r * HD + c);
        d[0] = reinterpret_cast<const uint4*>(out)[0];
        d[1] = reinterpret_cast<const uint4*>(out)[1];
    }
}

// In-place bf16 <-> fp16 over `n8` groups of 8 elements (16 B aligned): an fp16 library's Q in
// and O out. bf16 -> fp16 -> bf16 is exact for |x| < 65504 at fp16's normal range.
extern "C" __global__ __launch_bounds__(THREADS) void plow_cvt_rows(uint4* p, unsigned long long n8,
                                                                   unsigned to_f16) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)THREADS + threadIdx.x; i < n8;
         i += (unsigned long long)gridDim.x * THREADS) {
        uint4 v = p[i];
        uint16_t* h = reinterpret_cast<uint16_t*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            h[j] = to_f16 ? __half_as_ushort(__float2half_rn(__bfloat162float(__ushort_as_bfloat16(h[j]))))
                          : __bfloat16_as_ushort(__float2bfloat16(__half2float(__ushort_as_half(h[j]))));
        p[i] = v;
    }
}

// Mirrored by plowrt exec/gpu/lib_attention.rs `ZeroTailArgs`: a BF16 cache read in place.
struct PlowKvTail {
    uint16_t* k;
    uint16_t* v;
    const int* list;  // [n][3]: slot, kv_len, unused
    unsigned n, kv_heads, kv_stride, pad;
};

// Rows [kv_len, round_up(kv_len, 64)) of each request's slot, every KV head, K and V: zero.
// They sit in the block holding row kv_len - 1 (64 divides the live-KV block), the slot's next
// rows; the library loads whole 64-row pages and masks only the scores.
extern "C" __global__ __launch_bounds__(THREADS) void plow_kv_zero_tail(const __grid_constant__ PlowKvTail a) {
    const unsigned r = blockIdx.x, head = blockIdx.y;
    if (r >= a.n || head >= a.kv_heads) return;
    const unsigned slot = a.list[3 * r], len = a.list[3 * r + 1];
    const unsigned end = (len + PAGE - 1) / PAGE * PAGE;
    if (end > a.kv_stride || end == len) return;
    uint16_t* base = (blockIdx.z ? a.v : a.k) + (((size_t)slot * a.kv_heads + head) * a.kv_stride + len) * HD;
    const unsigned n16 = (end - len) * HD / 8;
    for (unsigned i = threadIdx.x; i < n16; i += THREADS) reinterpret_cast<uint4*>(base)[i] = make_uint4(0, 0, 0, 0);
}

#ifdef PLOW_FA4_HOST
// Host launchers, linked into libplow_fa4_hd512.so (scripts/build_sm90a_fa4_hd512.py): the
// adapters ship inside the library they serve. Each launches on the caller's stream and returns
// the launch status; no allocation, no sync.
extern "C" int plow_fa4_kv_gather(const PlowKvDequant* a, void* stream) {
    if (!a || !a->n_pages || !a->kv_heads) return 0;
    plow_kv_dequant_pages<<<dim3(a->n_pages, a->kv_heads, 2), THREADS, 0, (cudaStream_t)stream>>>(*a);
    return (int)cudaGetLastError();
}
extern "C" int plow_fa4_kv_zero_tail(const PlowKvTail* a, void* stream) {
    if (!a || !a->n || !a->kv_heads) return 0;
    plow_kv_zero_tail<<<dim3(a->n, a->kv_heads, 2), THREADS, 0, (cudaStream_t)stream>>>(*a);
    return (int)cudaGetLastError();
}
extern "C" int plow_fa4_convert(void* p, unsigned long long n8, unsigned to_f16, void* stream) {
    if (!n8) return 0;
    const unsigned long long blocks = (n8 + THREADS - 1) / THREADS;
    plow_cvt_rows<<<(unsigned)(blocks < 1056 ? blocks : 1056), THREADS, 0, (cudaStream_t)stream>>>(
        (uint4*)p, n8, to_f16);
    return (int)cudaGetLastError();
}
#endif
