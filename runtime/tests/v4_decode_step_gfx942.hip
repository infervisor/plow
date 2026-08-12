/* v4_decode_step_gfx942.hip — a WHOLE DeepSeek-V4 decode step, measured.
 *
 * Runs every op a token passes through at T=1, ctx 16k, TP1, at the shipped
 * shapes, and reports ms/token and tok/s against the 200 tok/s target.
 *
 * METHODOLOGY, and its one substitution, stated up front because it is what
 * makes the number believable or not:
 *
 *   Every projection is issued as a bf16 GEMV with `N` HALVED (or quartered) so
 *   the WEIGHT BYTES MOVED EQUAL the real fp8/fp4 tensor's. At M=1 all of these
 *   are bandwidth bound, not compute bound (a GEMV is ~2 flop/byte against this
 *   part's 19.6 ridge), so a bf16 GEMV moving 33.5 MB takes the time an fp8
 *   GEMV moving 33.5 MB takes.
 *
 *   THE GEMV HERE IS THIS FILE'S OWN, NOT `op_gemm.h`'s TUNED ARM. That header
 *   is device-only by construction — its fetch macros reinterpret_cast
 *   address-space-1 pointers, so it cannot be compiled in a host+device
 *   translation unit, which is why every existing test of it uses a two-step
 *   --genco build with a separate HSA host driver. The kernel below is a
 *   straightforward wave-per-row streaming GEMV: coalesced bf16x8 loads and a
 *   `dot8` reduction. So this number is a LOWER BOUND on what the emitter would
 *   get by routing to the tuned arm, not an upper one, and the gap between them
 *   is unmeasured here.
 *
 *   Not measured either: fp4 dequant overhead in the expert path.
 *
 *   So this is a COMPUTE step, not a served token: no sampler, no tokenizer, no
 *   HTTP, no batching, no KV paging. It is the decode arithmetic and memory
 *   traffic of one token, which is the floor a served number sits on top of.
 *
 * Weights are synthetic. Byte counts are what govern a bandwidth-bound decode,
 * and they are exact; values are not.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_decode_step_gfx942.hip -o /tmp/v4step && /tmp/v4step
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amd_common.h"
#include "op_deepseek_v4.h"

#define HC_CHECK(x)                                                                               \
    do {                                                                                          \
        hipError_t e_ = (x);                                                                      \
        if (e_) {                                                                                 \
            printf("hip %s @%d\n", hipGetErrorString(e_), __LINE__);                              \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

enum { NCU = 304, L = 43, HID = 4096, HC = 4, MIX = (2 + HC) * HC };
static const double BW = 4164e9;

/* M=1 GEMV: one WAVE per output row, lanes striding K in bf16x8 chunks so the
 * weight stream is fully coalesced, then one wave reduction. `dot8` is four
 * v_dot2c_f32_bf16 per 8 elements with no conversions. */
__global__ __launch_bounds__(PLOW_THREADS) void k_gemv(bf16* C, const bf16* x, const bf16* W,
                                                       unsigned N, unsigned K) {
    const unsigned lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const unsigned nwave = PLOW_THREADS >> 6;
    const unsigned NV = K >> 3; /* bf16x8 chunks per row */
    for (unsigned n = blockIdx.x * nwave + wave; n < N; n += gridDim.x * nwave) {
        const bf16* w = W + (size_t)n * K;
        float acc = 0.0f;
        for (unsigned c = lane; c < NV; c += 64) acc = dot8(ld_glob8(w + c * 8), ld_glob8(x + c * 8), acc);
        for (int o = 32; o; o >>= 1) acc += __shfl_xor(acc, o);
        if (lane == 0) st_act1(&C[n], f2bf(acc));
    }
}
__global__ __launch_bounds__(PLOW_THREADS) void k_hcdot(float* p, const bf16* x, const float* fn,
                                                        unsigned D) {
    __shared__ __align__(16) char raw[32768];
    d_v4_hc_dot(p, x, fn, 1, D, HC, blockIdx.x, gridDim.x, (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_hcmix(bf16* o, float* m, const bf16* x,
                                                        const float* p, const float* sc,
                                                        const float* ba, unsigned D) {
    __shared__ __align__(16) char raw[32768];
    d_v4_hc_mix(o, m, x, p, sc, ba, 1, D, HC, 20, 1e-6f, 1e-6f, blockIdx.x, gridDim.x,
                (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_hcexp(bf16* o, const bf16* b, const bf16* r,
                                                        const float* m, unsigned D) {
    d_hc_expand(o, b, r, m, 1, D, HC, blockIdx.x, gridDim.x);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_attn(bf16* o, const bf16* q, const bf16* kv,
                                                       const int* i, const float* s, unsigned H,
                                                       unsigned D, unsigned TOPK) {
    __shared__ __align__(16) char raw[40960];
    d_v4_sparse_attn(o, q, kv, i, s, 1, H, D, TOPK, 0.0442f, blockIdx.x, gridDim.x, (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_attn_sp(float* op, float* mlp, const bf16* q,
                                                          const bf16* kv, const int* i, unsigned H,
                                                          unsigned D, unsigned TOPK, unsigned SP) {
    __shared__ __align__(16) char raw[40960];
    d_v4_sparse_attn_split(op, mlp, q, kv, i, 1, H, D, TOPK, SP, 0.0442f, blockIdx.x, gridDim.x,
                           (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_attn_mg(bf16* o, const float* op,
                                                          const float* mlp, const float* s,
                                                          unsigned H, unsigned D, unsigned SP) {
    d_v4_sparse_attn_merge(o, op, mlp, s, 1, H, D, SP, blockIdx.x, gridDim.x);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_idx(float* sc, const bf16* q, const bf16* ckv,
                                                      const bf16* w, unsigned H, unsigned HD,
                                                      unsigned NC) {
    d_v4_index_score(sc, q, ckv, w, 1, H, HD, NC, blockIdx.x, gridDim.x);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_topk(int* idx, const float* sc, unsigned NC,
                                                       unsigned K) {
    __shared__ __align__(16) char raw[40960];
    d_v4_index_topk(idx, sc, 1, NC, K, 0, 0, blockIdx.x, gridDim.x, (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_cstep(bf16* o, float* kvs, float* scs,
                                                        const float* kvp, const float* scp,
                                                        const float* ape, const bf16* nw,
                                                        unsigned D, unsigned ratio, unsigned ov) {
    __shared__ __align__(16) char raw[16384];
    d_v4_kv_compress_step(o, kvs, scs, kvp, scp, ape, nw, D, ratio, ov, ratio - 1, 1e-6f,
                          blockIdx.x, gridDim.x, (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_glu(bf16* o, const bf16* g, const bf16* u,
                                                      unsigned n) {
    d_v4_clamped_swiglu(o, g, u, n, 10.0f);
}

template <typename F>
static double time_us(F f, int reps = 11) {
    for (int i = 0; i < 3; i++) f();
    HC_CHECK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; i++) {
        hipEvent_t a, b;
        hipEventCreate(&a);
        hipEventCreate(&b);
        hipEventRecord(a);
        f();
        hipEventRecord(b);
        HC_CHECK(hipEventSynchronize(b));
        float ms = 0;
        hipEventElapsedTime(&ms, a, b);
        t.push_back(ms * 1000.0);
        hipEventDestroy(a);
        hipEventDestroy(b);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

__global__ void k_null(void) {}

static bf16* dev_bf16(size_t n) {
    bf16* p;
    HC_CHECK(hipMalloc(&p, n * 2));
    HC_CHECK(hipMemset(p, 0x3c, n * 2));
    return p;
}
static float* dev_f32(size_t n) {
    float* p;
    HC_CHECK(hipMalloc(&p, n * 4));
    HC_CHECK(hipMemset(p, 0, n * 4));
    return p;
}
/* Distinct values, because a zero-filled score buffer makes EVERY entry tie at
 * the top-k threshold — a degenerate input that measures the tie fix-up rather
 * than the selector. It found a real O(need*nt) defect, but it is not the
 * decode step's shape. */
static float* dev_f32_varied(size_t n) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; i++) h[i] = (float)drand48() * 2.0f - 1.0f;
    float* p;
    HC_CHECK(hipMalloc(&p, n * 4));
    HC_CHECK(hipMemcpy(p, h.data(), n * 4, hipMemcpyHostToDevice));
    return p;
}

int main(void) {
    printf("DeepSeek-V4-Flash — one decode step, TP1, ctx 16384, MI325X\n");
    printf("target 200 tok/s = 5.00 ms/token   roofline 2.72 ms/token = 367 tok/s\n\n");

    const unsigned CTX = 16384;
    const unsigned NC4 = CTX / 4;      /* compressed entries, ratio-4 layers  */
    const unsigned TOPK = 640;         /* 128 window + 512 indexed            */
    const unsigned KVROWS = 128 + NC4; /* window ring + compressed history    */

    /* One shared activation/weight arena, reused by every op: the weights are
     * synthetic so only their SIZE matters, and one big buffer keeps the test
     * from allocating 11 GB. Each GEMV reads a distinct byte range so the L2
     * cannot serve the whole model. */
    /* Must exceed the LARGEST single weight, which is the lm_head at 1.06 GB —
     * a smaller arena makes the wrap below hand back a buffer shorter than the
     * read, which faults rather than silently reading garbage. */
    const size_t ARENA = (size_t)2 << 30; /* 2 GiB of bf16 weight bytes */
    bf16* Warena = dev_bf16(ARENA / 2);
    bf16* xbuf = dev_bf16(1 << 20);
    bf16* obuf = dev_bf16(1 << 22);
    float* fbuf = dev_f32_varied(1 << 22);
    int* ibuf;
    HC_CHECK(hipMalloc(&ibuf, (size_t)(1 << 20) * 4));
    HC_CHECK(hipMemset(ibuf, 0, (size_t)(1 << 20) * 4));

    size_t off = 0;
    auto W = [&](size_t bytes) {
        if (off + bytes > ARENA) off = 0;
        bf16* p = Warena + off / 2;
        off += bytes;
        return p;
    };

    /* The per-launch floor, measured with an empty kernel at the same grid.
     * ~1900 packets in a V4 step means this is not a rounding error, and the
     * persistent interpreter does not pay it: packets are dispatched by a
     * counter DAG inside one long-running kernel. Both numbers are reported —
     * as-measured, and with the floor removed — because neither alone is
     * honest. The second is an ESTIMATE of the interpreter-resident cost, not
     * a measurement of one. */
    const double floor_us = time_us([&] { k_null<<<NCU, PLOW_THREADS>>>(); });

    struct Row {
        const char* name;
        double us;
        double bytes;
        int count;
    };
    std::vector<Row> rows;
    auto gemv = [&](const char* nm, unsigned N, unsigned K, int count) {
        const double bytes = (double)N * K * 2;
        bf16* w = W((size_t)N * K * 2);
        const double us = time_us([&] { k_gemv<<<NCU, PLOW_THREADS>>>(obuf, xbuf, w, N, K); });
        rows.push_back({nm, us, bytes, count});
    };

    /* ---- per-layer projections. N is halved/quartered against the real
     * tensor so the BYTES match the fp8 (or fp4) weight the model ships. ---- */
    gemv("attn wq_a   (fp8 1024x4096)", 512, 4096, L);
    gemv("attn wq_b   (fp8 32768x1024)", 16384, 1024, L);
    gemv("attn wkv    (fp8 512x4096)", 256, 4096, L);
    gemv("attn wo_a   (fp8 8192x4096)", 4096, 4096, L);
    gemv("attn wo_b   (fp8 4096x8192)", 2048, 8192, L);
    gemv("moe gate    (bf16 256x4096)", 256, 4096, L);
    gemv("shared w1/w3(fp8 2048x4096)", 1024, 4096, 2 * L);
    gemv("shared w2   (fp8 4096x2048)", 2048, 2048, L);
    /* 6 of 256 routed experts per layer, fp4 => quarter the bf16 N. */
    gemv("routed w1/w3(fp4 2048x4096)", 256, 4096, 2 * 6 * L);
    gemv("routed w2   (fp4 4096x2048)", 512, 2048, 6 * L);
    /* compressors: 41 layers have one; 21 of those also carry an indexer. */
    gemv("compressor wkv/wgate", 1024, 4096, 2 * 41);
    gemv("indexer wq_b (fp8 8192x1024)", 4096, 1024, 21);
    gemv("indexer weights_proj", 64, 4096, 21);
    gemv("indexer compressor", 256, 4096, 2 * 21);
    /* the head, once per token */
    gemv("lm_head    (bf16 129280x4096)", 129280, 4096, 1);

    /* ---- V4-specific ops -------------------------------------------------- */
    {
        float* fn = dev_f32((size_t)MIX * HC * HID);
        float* part = dev_f32(1 + MIX);
        float* mixo = dev_f32(HC + HC * HC);
        const double us = time_us([&] {
            k_hcdot<<<NCU, PLOW_THREADS>>>(part, xbuf, fn, HID);
            k_hcmix<<<NCU, PLOW_THREADS>>>(obuf, mixo, xbuf, part, fbuf, fbuf, HID);
        });
        rows.push_back({"hc reduce (dot+mix)", us, (double)MIX * HC * HID * 4, 2 * L});
        const double ue =
            time_us([&] { k_hcexp<<<NCU, PLOW_THREADS>>>(obuf, xbuf, xbuf, mixo, HID); });
        rows.push_back({"hc expand", ue, (double)HC * HID * 2 * 3, 2 * L});
    }
    {
        bf16* kv = dev_bf16((size_t)KVROWS * 512);
        /* SPLIT-K. The (token, head) work index gives 64 blocks at T=1 on a
         * 304-CU part; v4_attn_scale_gfx942.hip shows the total is FLAT from 64
         * to 256 blocks, i.e. the op is CU-starved rather than slow. Four key
         * chunks fill the machine. */
#ifndef V4_ATTN_SPLIT
#define V4_ATTN_SPLIT 4u
#endif
        const unsigned SP = V4_ATTN_SPLIT;
        float* op = dev_f32((size_t)64 * SP * 512);
        float* mlp = dev_f32((size_t)64 * SP * 2);
        const double us =
            (SP <= 1u)
                ? time_us([&] {
                      k_attn<<<64, PLOW_THREADS>>>(obuf, xbuf, kv, ibuf, fbuf, 64, 512, TOPK);
                  })
                : time_us([&] {
                      k_attn_sp<<<64 * SP, PLOW_THREADS>>>(op, mlp, xbuf, kv, ibuf, 64, 512, TOPK,
                                                           SP);
                      k_attn_mg<<<64, PLOW_THREADS>>>(obuf, op, mlp, fbuf, 64, 512, SP);
                  });
        rows.push_back({"sparse attention (split-K)", us, (double)TOPK * 512 * 2, L});
    }
    {
        bf16* ckv = dev_bf16((size_t)NC4 * 128);
        const double us =
            time_us([&] { k_idx<<<NCU, PLOW_THREADS>>>(fbuf, xbuf, ckv, xbuf, 64, 128, NC4); });
        rows.push_back({"indexer score", us, (double)NC4 * 128 * 2, 21});
        const double ut = time_us([&] { k_topk<<<1, PLOW_THREADS>>>(ibuf, fbuf, NC4, 512); });
        rows.push_back({"indexer top-k", ut, (double)NC4 * 4, 21});
    }
    {
        float* kvs = dev_f32((size_t)2 * 4 * 1024);
        float* scs = dev_f32((size_t)2 * 4 * 1024);
        float* ape = dev_f32((size_t)4 * 1024);
        const double us = time_us(
            [&] { k_cstep<<<1, PLOW_THREADS>>>(obuf, kvs, scs, fbuf, fbuf, ape, xbuf, 512, 4, 1); });
        rows.push_back({"kv compress step", us, 0.0, 41});
    }
    {
        const double us = time_us([&] { k_glu<<<NCU, PLOW_THREADS>>>(obuf, xbuf, xbuf, 2048); });
        rows.push_back({"clamped swiglu", us, 0.0, 7 * L});
    }

    /* ---- report ----------------------------------------------------------- */
    double tot_us = 0, tot_bytes = 0;
    printf("  %-32s %5s %9s %10s %9s\n", "op", "count", "us each", "ms total", "GB/s");
    for (const auto& r : rows) {
        const double ms = r.us * r.count / 1000.0;
        tot_us += r.us * r.count;
        tot_bytes += r.bytes * r.count;
        printf("  %-32s %5d %9.2f %10.3f %9.1f\n", r.name, r.count, r.us, ms,
               r.bytes > 0 ? r.bytes / (r.us * 1e-6) / 1e9 : 0.0);
    }
    const double ms = tot_us / 1000.0;
    int nlaunch = 0;
    for (const auto& r : rows) nlaunch += r.count;
    const double floor_ms = floor_us * nlaunch / 1000.0;
    const double net_ms = ms - floor_ms;
    printf("\n  %-32s %5d %9s %10.3f %9.1f\n", "TOTAL", nlaunch, "", ms,
           tot_bytes / (tot_us * 1e-6) / 1e9);
    printf("\n  as measured        %.3f ms/token  =>  %6.1f tok/s   %s\n", ms, 1000.0 / ms,
           ms <= 5.0 ? "MET" : "NOT MET");
    printf("  launch floor       %.3f ms  (%d kernels x %.2f us) — harness only\n", floor_ms,
           nlaunch, floor_us);
    printf("  floor removed      %.3f ms/token  =>  %6.1f tok/s   %s   <- interpreter estimate\n",
           net_ms, 1000.0 / net_ms, net_ms <= 5.0 ? "MET" : "NOT MET");
    printf("  weight+KV bytes moved: %.2f GB/token; roofline at 4164 GB/s = %.3f ms\n",
           tot_bytes / 1e9, tot_bytes / BW * 1e3);
    printf("  achieved %.1f%% of measured peak bandwidth\n",
           100.0 * (tot_bytes / (tot_us * 1e-6)) / BW);
    return 0;
}
