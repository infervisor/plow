/* nvcc -gencode arch=compute_90a,code=sm_90a -O3 -I runtime/common -I runtime/nvidia
 * runtime/tests/token_batch_dense_sm90_test.cu -o token_batch_dense_sm90_test */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define PLOW_NV_TOKEN_BATCH_BF16 1
#include "sm120_common.cuh"
#include "op_norm.cuh"
using bf16 = __nv_bfloat16;
#define CK(x) do { auto e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(2); } } while (0)
constexpr unsigned HD = 128, QH = 16, KH = 8, STRIDE = 512;
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
        0, blockIdx.x, gridDim.x, 0, nullptr, nullptr, nullptr, 0, batch);
    d_headnorm_rope<HD>(k, kg, gamma, cs, sn, pos, rows, KH, 1e-6f, 0, STRIDE, STRIDE - 1,
        0, blockIdx.x, gridDim.x, 1, nullptr, v, vg, 0, batch);
}
__global__ void normalize_query(bf16* q, const bf16* x, const bf16* gamma, const float* cs,
                                const float* sn, const int* pos, unsigned rows) {
    d_headnorm_rope<HD>(q, x, gamma, cs, sn, pos, rows, QH, 1e-6f, 0, 0, STRIDE - 1,
        0, blockIdx.x, gridDim.x);
}
__global__ void normalize_one(bf16* k, bf16* v, const bf16* kg, const bf16* vg, const bf16* gamma,
                              const float* cs, const float* sn, const int* pos) {
    d_headnorm_rope<HD>(k, kg, gamma, cs, sn, pos, 1, KH, 1e-6f, 0, STRIDE, STRIDE - 1,
        0, blockIdx.x, gridDim.x, 1, nullptr, v, vg);
}
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
    // Reference uses the unchanged one-row cache writer for each physical slot/position.
    for (unsigned r = 0; r < live; r++) {
        const size_t off = (size_t)physical[r] * KH * STRIDE * HD;
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
            const size_t ko = (((size_t)physical[r] * KH + h / 2) * STRIDE + (t & (STRIDE-1))) * HD;
            double dot = 0; for (unsigned d = 0; d < HD; d++) dot += (double)__bfloat162float(q[qo+d]) * __bfloat162float(k[ko+d]);
            maximum = std::max(maximum, scores[t-begin] = dot / sqrt((double)HD));
        }
        double sum = 0; for (auto& x : scores) sum += x = exp(x - maximum);
        for (unsigned d = 0; d < HD; d++) {
            double y = 0;
            for (unsigned t = begin; t < end; t++) {
                const size_t vo = (((size_t)physical[r] * KH + h / 2) * STRIDE + (t & (STRIDE-1))) * HD;
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
    const bool sanitize = argc == 2 && !strcmp(argv[1], "--sanitize");
    for (unsigned split : {1u,3u}) for (unsigned window : {0u,32u}) run(16,2,2,3,64,window,split,false);
    if (!sanitize) {
        for (unsigned ctx : {64u,256u}) for (unsigned b : {32u,64u,128u,200u,224u}) run(b,b,0,0,ctx,0,1,true);
        run(192,128,2,24,256,0,1,true);
    }
}
