/* nvcc -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common -I runtime/nvidia
 * runtime/tests/token_batch_dense_sm90_test.cu -o token_batch_dense_sm90_test */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define PLOW_NV_TOKEN_BATCH_BF16 1
#include "sm120_common.cuh"
#include "op_norm.cuh"
#if PLOW_MIXED_STEP
#define TB_NO_MIXED nullptr,
#else
#define TB_NO_MIXED
#endif
using bf16 = __nv_bfloat16;
#define CK(x) do { auto e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(2); } } while (0)
constexpr unsigned HD = 128, QH = 16, KH = 8, STRIDE = 2048;
static size_t cache_row(unsigned slot, unsigned head, unsigned position) {
#if defined(PLOW_NV_KV_ILV) && PLOW_NV_KV_ILV
    return (((size_t)slot * STRIDE + (position & (STRIDE - 1))) * KH + head) * HD;
#else
    return (((size_t)slot * KH + head) * STRIDE + (position & (STRIDE - 1))) * HD;
#endif
}
static unsigned rng = 193;
static float random_float() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return ((int)(rng % 2001) - 1000) / 1000.f; }
template<class T> static T* upload(const std::vector<T>& x) {
    T* p; CK(cudaMalloc(&p, x.size() * sizeof(T))); CK(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
template<class T> static std::vector<T> read(T* p, size_t n) {
    std::vector<T> x(n); CK(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x;
}
__global__ void normalize(bf16* q, bf16* k, bf16* v, const bf16* qg, const bf16* kg, const bf16* vg,
                          const bf16* gamma, const float* cs, const float* sn, const int* pos,
                          unsigned rows, const PlowTokenBatch* batch) {
    d_headnorm_rope<HD>(q, qg, gamma, cs, sn, pos, rows, QH, 1e-6f, 0, 0, STRIDE - 1,
        0, blockIdx.x, gridDim.x, 0, nullptr, TB_NO_MIXED nullptr, nullptr, 0, batch);
    d_headnorm_rope<HD>(k, kg, gamma, cs, sn, pos, rows, KH, 1e-6f, 0, STRIDE, STRIDE - 1,
        0, blockIdx.x, gridDim.x, 1, nullptr, TB_NO_MIXED v, vg, 0, batch);
}
__global__ void normalize_query(bf16* q, const bf16* x, const bf16* gamma, const float* cs,
                                const float* sn, const int* pos, unsigned rows) {
    d_headnorm_rope<HD>(q, x, gamma, cs, sn, pos, rows, QH, 1e-6f, 0, 0, STRIDE - 1,
        0, blockIdx.x, gridDim.x);
}
__global__ void normalize_one(bf16* k, bf16* v, const bf16* kg, const bf16* vg, const bf16* gamma,
                              const float* cs, const float* sn, const int* pos) {
#if defined(PLOW_NV_KV_ILV) && PLOW_NV_KV_ILV
    d_headnorm_rope<HD>(k, kg, gamma, cs, sn, pos, 1, KH, 1e-6f, 0, 0, STRIDE - 1,
#else
    d_headnorm_rope<HD>(k, kg, gamma, cs, sn, pos, 1, KH, 1e-6f, 0, STRIDE, STRIDE - 1,
#endif
        0, blockIdx.x, gridDim.x, 1, nullptr, TB_NO_MIXED v, vg);
}
__global__ void normalize_packed(bf16* k, bf16* v, const bf16* kg, const bf16* vg,
                                 const bf16* gamma, const float* cs, const float* sn,
                                 const int* pos, unsigned rows) {
    d_headnorm_rope<HD>(k, kg, gamma, cs, sn, pos, rows, KH, 1e-6f, 0, 0, STRIDE - 1,
        0, blockIdx.x, gridDim.x, 0, nullptr, TB_NO_MIXED v, vg);
}
__global__ void normalize_cache_mode(bf16* k, bf16* v, const bf16* kg, const bf16* vg,
                                     const bf16* gamma, const float* cs, const float* sn,
                                     const int* pos, const int* slots, unsigned rows,
                                     unsigned row0, unsigned mode) {
    d_headnorm_rope<HD>(k, kg, gamma, cs, sn, pos, rows, KH, 1e-6f, row0, STRIDE, STRIDE - 1,
        0, blockIdx.x, gridDim.x, mode == 1, mode == 0 ? slots : nullptr, TB_NO_MIXED v, vg);
}
#if PLOW_MIXED_STEP
__global__ void normalize_mixed(bf16* k, bf16* v, const bf16* kg, const bf16* vg,
                                const bf16* gamma, const float* cs, const float* sn,
                                const int* pos, const int* slots, unsigned rows,
                                const PlowProgram* mixed) {
    d_headnorm_rope<HD>(k, kg, gamma, cs, sn, pos, rows, KH, 1e-6f, 0, STRIDE, STRIDE - 1,
        0, blockIdx.x, gridDim.x, 0, slots, mixed, v, vg);
}
#endif
__global__ void attention(float* partial, float* ml, const bf16* q, const bf16* k, const bf16* v,
                          const int* lens, const int* slots, unsigned rows, unsigned window,
                          unsigned splits, unsigned cache_rows, const PlowTokenBatch* batch) {
    extern __shared__ float arena[];
    d_flash_decode_slots<HD, 2>(partial, ml, q, k, v, lens, rows, QH, KH, STRIDE, window,
        1.f / sqrtf((float)HD), splits, STRIDE - 1, blockIdx.x, gridDim.x, arena, cache_rows,
        slots, nullptr, nullptr, batch);
}
__global__ void merge(bf16* out, const float* partial, const float* ml, unsigned rows, unsigned splits) {
    d_flash_merge<HD>(out, partial, ml, rows, QH, splits, blockIdx.x, gridDim.x);
}
static void run(unsigned cap, unsigned decode, unsigned arrivals, unsigned tokens, unsigned ctx,
                unsigned window, unsigned splits, bool timing) {
    const unsigned live = decode + arrivals * tokens, slots = decode + arrivals + 3;
    std::vector<PlowPrefillSpan> spans;
    std::vector<unsigned> positions(cap), active(cap), ids(cap), physical(cap);
    unsigned row = 0;
    for (unsigned s = 0; s < decode + arrivals; s++) {
        const bool dec = s < decode;
        const unsigned count = dec ? 1 : tokens, base = dec ? ctx - 1 : ctx / 2;
        const unsigned slot = slots - 1 - s;
        spans.push_back({row, count, slot, dec ? PLOW_PREFILL_SPAN_DECODE : 0u,
            base, base + count, slot, 0});
        for (unsigned j = 0; j < count; j++, row++) { positions[row] = base + j; active[row] = 1; physical[row] = slot; }
    }
    if (live > cap || ctx / 2 + tokens > STRIDE) exit(3);
    auto* ds = upload(spans); auto* dp = upload(positions); auto* da = upload(active); auto* di = upload(ids);
    PlowTokenBatch batch{};
    batch.version = 1; batch.row_capacity = cap; batch.real_rows = live; batch.n_spans = spans.size();
    batch.spans = ds; batch.positions = dp; batch.active = da; batch.input_ids = di;
    auto* db = upload(std::vector<PlowTokenBatch>{batch});
    std::vector<bf16> qg(cap * QH * HD), kg(cap * KH * HD), vg(kg.size()), gamma(HD);
    for (auto& x : qg) x = __float2bfloat16(random_float());
    for (auto& x : kg) x = __float2bfloat16(random_float());
    for (auto& x : vg) x = __float2bfloat16(random_float());
    for (auto& x : gamma) x = __float2bfloat16(1.f + .1f * random_float());
    std::vector<float> cs(STRIDE * HD / 2), sn(cs.size());
    for (unsigned t = 0; t < STRIDE; t++) for (unsigned j = 0; j < HD / 2; j++) {
        const float angle = t * powf(10000.f, -2.f * j / HD);
        cs[t * HD / 2 + j] = cosf(angle); sn[t * HD / 2 + j] = sinf(angle);
    }
    auto *dqg = upload(qg), *dkg = upload(kg), *dvg = upload(vg), *dg = upload(gamma);
    auto *dcs = upload(cs), *dsn = upload(sn);
    std::vector<bf16> cache((size_t)slots * KH * STRIDE * HD);
    for (auto& x : cache) x = __float2bfloat16(.2f * random_float());
    auto *dk = upload(cache), *drk = upload(cache);
    for (auto& x : cache) x = __float2bfloat16(.2f * random_float());
    auto *dv = upload(cache), *drv = upload(cache);
    std::vector<bf16> sentinel(qg.size(), __float2bfloat16(-13.f));
    auto *dq = upload(sentinel), *drq = upload(sentinel), *out = upload(sentinel), *ref = upload(sentinel);
    normalize<<<132, 256>>>(dq, dk, dv, dqg, dkg, dvg, dg, dcs, dsn, (const int*)dp, cap, db);
    // Interleaved references scatter packed outputs, bypassing the cache-addressing arm.
    for (unsigned r = 0; r < live; r++) {
#if defined(PLOW_NV_KV_ILV) && PLOW_NV_KV_ILV
        const size_t off = cache_row(physical[r], 0, positions[r]);
#else
        const size_t off = (size_t)physical[r] * KH * STRIDE * HD;
#endif
        normalize_one<<<1, 256>>>(drk + off, drv + off, dkg + (size_t)r * KH * HD,
            dvg + (size_t)r * KH * HD, dg, dcs, dsn, (const int*)dp + r);
    }
    normalize_query<<<132, 256>>>(drq, dqg, dg, dcs, dsn, (const int*)dp, cap);
    CK(cudaDeviceSynchronize());
    auto k = read(dk, cache.size()), rk = read(drk, cache.size());
    auto v = read(dv, cache.size()), rv = read(drv, cache.size());
    if (memcmp(k.data(), rk.data(), k.size() * sizeof(bf16)) || memcmp(v.data(), rv.data(), v.size() * sizeof(bf16))) {
        fprintf(stderr, "FAIL mapped K/V writes cap%u ctx%u\n", cap, ctx); exit(1);
    }
    auto q = read(dq, sentinel.size());
    const auto qr = read(drq, sentinel.size());
    if (memcmp(q.data(), qr.data(), (size_t)live * QH * HD * sizeof(bf16))) { fprintf(stderr, "FAIL mapped Q\n"); exit(1); }
    for (size_t e = (size_t)live * QH * HD; e < q.size(); e++)
        if (memcmp(&q[e], &sentinel[e], sizeof(bf16))) { fprintf(stderr, "FAIL padding Q store\n"); exit(1); }
#if PLOW_MIXED_STEP
    if (decode && arrivals) {
        std::vector<PlowPrefillSpan> arrival_spans(spans.begin() + decode, spans.end());
        std::vector<unsigned> parked(cap);
        for (unsigned r = live; r < cap; r++) parked[r] = 1;
        auto* mixed_spans = upload(arrival_spans);
        auto* mixed_parked = upload(parked);
        auto* mixed_slots = upload(std::vector<int>(physical.begin(), physical.end()));
        PlowProgram program{};
        program.prefill_spans = mixed_spans;
        program.prefill_parked = mixed_parked;
        program.n_prefill_spans = arrivals;
        program.n_prefill_rows = cap;
        auto* mixed = upload(std::vector<PlowProgram>{program});
        for (unsigned r = 0; r < live; r++) for (unsigned h = 0; h < KH; h++) {
            const size_t index = cache_row(physical[r], h, positions[r]);
            std::fill_n(k.begin() + index, HD, __float2bfloat16(-13.f));
            std::fill_n(v.begin() + index, HD, __float2bfloat16(-13.f));
        }
        CK(cudaMemcpy(dk, k.data(), k.size() * sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * sizeof(bf16), cudaMemcpyHostToDevice));
        normalize_mixed<<<132,256>>>(dk,dv,dkg,dvg,dg,dcs,dsn,(const int*)dp,mixed_slots,cap,mixed);
        const auto mk = read(dk, k.size()), mv = read(dv, v.size());
        if (memcmp(mk.data(), rk.data(), mk.size() * sizeof(bf16)) ||
            memcmp(mv.data(), rv.data(), mv.size() * sizeof(bf16))) {
            fprintf(stderr, "FAIL mixed cache writer cap%u ctx%u\n",cap,ctx); exit(1);
        }
        k = rk; v = rv;
        CK(cudaFree(mixed)); CK(cudaFree(mixed_spans));
        CK(cudaFree(mixed_parked)); CK(cudaFree(mixed_slots));
    }
#endif
    if (live == cap && live == decode && cap <= 16) {
        const size_t packed_elements = (size_t)cap * KH * HD;
        bf16 *packed_k, *packed_v;
        CK(cudaMalloc(&packed_k, packed_elements * sizeof(bf16)));
        CK(cudaMalloc(&packed_v, packed_elements * sizeof(bf16)));
        auto* physical_slots = upload(std::vector<int>(physical.begin(), physical.end()));
        normalize_packed<<<132,256>>>(packed_k,packed_v,dkg,dvg,dg,dcs,dsn,(const int*)dp,cap);
        const auto pk = read(packed_k, packed_elements), pv = read(packed_v, packed_elements);
        for (unsigned mode : {0u,1u,2u}) {
            CK(cudaMemset(dk, 0x5a, k.size() * sizeof(bf16)));
            CK(cudaMemset(dv, 0x5a, v.size() * sizeof(bf16)));
            memset(rk.data(), 0x5a, rk.size() * sizeof(bf16));
            memset(rv.data(), 0x5a, rv.size() * sizeof(bf16));
            for (unsigned r = 0; r < cap; r++) for (unsigned h = 0; h < KH; h++) {
                const unsigned slot = mode == 0 ? physical[r] : mode == 1 ? r : 0;
                const unsigned position = mode == 2 ? ctx - 1 + r : positions[r];
                const size_t dst = cache_row(slot, h, position), src = ((size_t)r * KH + h) * HD;
                memcpy(rk.data() + dst, pk.data() + src, HD * sizeof(bf16));
                memcpy(rv.data() + dst, pv.data() + src, HD * sizeof(bf16));
            }
            normalize_cache_mode<<<132,256>>>(dk,dv,dkg,dvg,dg,dcs,dsn,(const int*)dp,
                physical_slots,cap,ctx-1,mode);
            const auto mk = read(dk, k.size()), mv = read(dv, v.size());
            if (memcmp(mk.data(), rk.data(), mk.size() * sizeof(bf16)) ||
                memcmp(mv.data(), rv.data(), mv.size() * sizeof(bf16))) {
                fprintf(stderr, "FAIL cache writer mode%u cap%u ctx%u\n", mode,cap,ctx); exit(1);
            }
        }
        CK(cudaMemcpy(dk, k.data(), k.size() * sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaFree(packed_k)); CK(cudaFree(packed_v)); CK(cudaFree(physical_slots));
    }
    std::vector<int> lens(cap), poison(cap, 0x10000000), slotmap(cap);
    for (unsigned r = 0; r < live; r++) { lens[r] = positions[r] + 1; slotmap[r] = physical[r]; }
    auto *dl = upload(lens), *dpoison = upload(poison), *dslot = upload(slotmap);
    const size_t partial_size = (size_t)cap * QH * splits * HD, ml_size = (size_t)cap * QH * splits * 2;
    float *part, *ml; CK(cudaMalloc(&part, partial_size * 4)); CK(cudaMalloc(&ml, ml_size * 4));
    const unsigned shared = FA_DEC_SMEM_FLOATS(HD, 2) * 4;
    attention<<<132, 256, shared>>>(part, ml, dq, dk, dv, dpoison, dslot, cap, window, splits,
        slots * KH * STRIDE, db);
    merge<<<132, 256>>>(out, part, ml, cap, splits);
    attention<<<132, 256, shared>>>(part, ml, drq, drk, drv, dl, dslot, cap, window, splits,
        slots * KH * STRIDE, nullptr);
    merge<<<132, 256>>>(ref, part, ml, cap, splits);
    CK(cudaDeviceSynchronize());
    const auto got = read(out, sentinel.size()), expected = read(ref, sentinel.size());
    if (memcmp(got.data(), expected.data(), got.size() * sizeof(bf16))) {
        fprintf(stderr, "FAIL mapped attention cap%u ctx%u split%u\n", cap, ctx, splits); exit(1);
    }
    double err = 0, den = 0;
    for (unsigned r = 0; r < std::min(live, 3u); r++) for (unsigned h = 0; h < QH; h++) {
        const unsigned end = positions[r] + 1, begin = window && end > window ? end - window : 0;
        std::vector<double> scores(end - begin); double maximum = -INFINITY;
        const size_t qo = ((size_t)r * QH + h) * HD;
        for (unsigned t = begin; t < end; t++) {
            const size_t ko = cache_row(physical[r], h / 2, t);
            double dot = 0; for (unsigned d = 0; d < HD; d++) dot += (double)__bfloat162float(q[qo+d]) * __bfloat162float(k[ko+d]);
            maximum = std::max(maximum, scores[t-begin] = dot / sqrt((double)HD));
        }
        double sum = 0; for (auto& x : scores) sum += x = exp(x - maximum);
        for (unsigned d = 0; d < HD; d++) {
            double y = 0;
            for (unsigned t = begin; t < end; t++) {
                const size_t vo = cache_row(physical[r], h / 2, t);
                y += scores[t-begin] * __bfloat162float(v[vo+d]);
            }
            y /= sum; const double diff = __bfloat162float(got[qo+d]) - y;
            err += diff * diff; den += y * y;
        }
    }
    const double rel = sqrt(err / den);
    if (!std::isfinite(rel) || rel > .004) { fprintf(stderr, "FAIL CPU attention relL2 %g\n", rel); exit(1); }
    if (timing) {
        cudaEvent_t start, end; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&end));
        for (const auto* descriptor : {static_cast<const PlowTokenBatch*>(nullptr), static_cast<const PlowTokenBatch*>(db)}) {
            for (unsigned rep = 0; rep < 2; rep++) {
                CK(cudaEventRecord(start));
                for (unsigned i = 0; i < 40; i++) attention<<<132,256,shared>>>(part,ml,dq,dk,dv,dl,dslot,cap,window,splits,slots*KH*STRIDE,descriptor);
                CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end)); float ms; CK(cudaEventElapsedTime(&ms,start,end));
                printf("TIMING cap=%u ctx=%u mapped=%u rep=%u us=%.3f floor_hbm_us=%.3f\n", cap,ctx,descriptor!=nullptr,rep,ms*1000/40,
                    (double)cap * KH * ctx * HD * 4 / 3049.7e3);
            }
        }
        CK(cudaEventDestroy(start)); CK(cudaEventDestroy(end));
    }
    printf("PASS cap=%u decode=%u arrivals=%u x%u ctx=%u window=%u splits=%u relL2=%g\n",cap,decode,arrivals,tokens,ctx,window,splits,rel);
    for (void* p : {static_cast<void*>(ds),static_cast<void*>(dp),static_cast<void*>(da),static_cast<void*>(di),static_cast<void*>(db),
        static_cast<void*>(dqg),static_cast<void*>(dkg),static_cast<void*>(dvg),static_cast<void*>(dg),static_cast<void*>(dcs),static_cast<void*>(dsn),
        static_cast<void*>(dk),static_cast<void*>(drk),static_cast<void*>(dv),static_cast<void*>(drv),static_cast<void*>(dq),static_cast<void*>(drq),
        static_cast<void*>(out),static_cast<void*>(ref),static_cast<void*>(dl),static_cast<void*>(dpoison),static_cast<void*>(dslot),static_cast<void*>(part),static_cast<void*>(ml)}) CK(cudaFree(p));
}
int main(int argc, char** argv) {
    if (argc == 4 && !strcmp(argv[1], "--case")) {
        char *batch_end, *context_end;
        const auto batch = strtoul(argv[2], &batch_end, 10);
        const auto context = strtoul(argv[3], &context_end, 10);
        if (argv[2][0] < '0' || argv[2][0] > '9' || argv[3][0] < '0' || argv[3][0] > '9' ||
            *batch_end || *context_end || !batch || batch > 320 || !context || context > STRIDE) return 3;
        run((unsigned)batch, (unsigned)batch, 0, 0, (unsigned)context, 0, 1, true);
        return 0;
    }
    const bool sanitize = argc == 2 && !strcmp(argv[1], "--sanitize");
    if (argc != 1 && !sanitize) return 3;
    run(8,8,0,0,1025,0,1,false);
    for (unsigned split : {1u,3u}) for (unsigned window : {0u,32u}) run(16,2,2,3,64,window,split,false);
#if defined(PLOW_NV_KV_ILV) && PLOW_NV_KV_ILV
    for (unsigned context : {1023u,1024u,1025u}) run(16,2,2,3,context,0,1,false);
#endif
    if (!sanitize) {
        for (unsigned ctx : {64u,256u}) for (unsigned b : {32u,64u,128u,200u,224u}) run(b,b,0,0,ctx,0,1,true);
        run(192,128,2,24,256,0,1,true);
    }
}
