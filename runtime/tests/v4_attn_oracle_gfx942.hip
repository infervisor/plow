/* v4_attn_oracle_gfx942.hip — `d_v4_sparse_attn` against a HOST TRANSCRIPTION
 * of `inference/kernel.py::sparse_attn_kernel`.
 *
 * WHY THIS EXISTS. V4 has ONE attention kernel. The sliding window and the
 * compressed history are the same call with a different index list, so this op
 * carries every attention site in the model — and it carries a learned sink,
 * which no other attention in this tree has.
 *
 * THE REFERENCE, for one (token, head):
 *
 *     s_j    = (q . kv[idx_j]) * scale        , or -inf where idx_j == -1
 *     m      = max_j s_j                      # REAL KEYS ONLY
 *     o      = sum_j exp(s_j - m) * kv[idx_j]
 *     l      = sum_j exp(s_j - m) + exp(sink[h] - m)
 *     out    = o / l
 *
 * WHAT THIS PINS, each of which is a way to be plausibly wrong:
 *
 *   * the sink is in the DENOMINATOR ONLY — it has no value row. Dropping it
 *     entirely is NEGATIVE-CONTROLLED, and the last case below exists because
 *     that control is only marginal at a large key count (one term among 640)
 *     and needs the sequence-start regime to bite properly.
 *   * whether the sink joins the running max is NOT a correctness question —
 *     softmax is shift invariant and the folded form measures identical here.
 *     Recorded because it looks like a trap and is not one.
 *   * `idx == -1` means a -inf score AND a zeroed KV row, not "read row 0".
 *     NEGATIVE-CONTROLLED, and loud (9.7 max rel).
 *   * K and V are the SAME rows: V4 projects one KV vector per token.
 *
 * SWEPT: the shipped geometry (H=64, D=512), a masked tail (the window index
 * builder emits -1 for positions before the start of the sequence), T>1 across
 * several workgroups, and the indexer's D=128 width.
 *
 * TOLERANCE 2e-2 on max elementwise difference relative to the row RMS — the
 * bf16 floor, as in the K3 rung gates. The host reduces in double.
 *
 *   hipcc --offload-arch=gfx942 -O3 -w -I runtime/amd -I runtime/common \
 *       runtime/tests/v4_attn_oracle_gfx942.hip -o /tmp/v4attn && /tmp/v4attn
 */
#include <hip/hip_runtime.h>

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

enum { ATTN_LDS = 40960 };

__global__ __launch_bounds__(PLOW_THREADS) void k_attn(bf16* out, const bf16* q, const bf16* kv,
                                                       const int* idx, const float* sink,
                                                       unsigned T, unsigned H, unsigned D,
                                                       unsigned TOPK, float scale) {
    __shared__ __align__(16) char raw[ATTN_LDS];
    d_v4_sparse_attn(out, q, kv, idx, sink, T, H, D, TOPK, scale, blockIdx.x, gridDim.x,
                     (float*)raw);
}

__global__ __launch_bounds__(PLOW_THREADS) void k_split(float* op, float* mlp, const bf16* q,
                                                        const bf16* kv, const int* idx, unsigned T,
                                                        unsigned H, unsigned D, unsigned TOPK,
                                                        unsigned SP, float scale) {
    __shared__ __align__(16) char raw[ATTN_LDS];
    d_v4_sparse_attn_split(op, mlp, q, kv, idx, nullptr, T, H, D, TOPK, SP, 0, 0, 0, scale, blockIdx.x,
                           gridDim.x, (float*)raw);
}
__global__ __launch_bounds__(PLOW_THREADS) void k_merge(bf16* out, const float* op,
                                                        const float* mlp, const float* sink,
                                                        unsigned T, unsigned H, unsigned D,
                                                        unsigned SP) {
    d_v4_sparse_attn_merge(out, op, mlp, sink, T, H, D, SP, blockIdx.x, gridDim.x);
}

static bf16 f2bf_host(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    const unsigned r = (u >> 16) & 1u;
    return (bf16)((u + 0x7fffu + r) >> 16);
}
static float bf2f_host(bf16 b) {
    const unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float frand(void) { return (float)drand48() * 2.0f - 1.0f; }

static double rms_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return sqrt(s / (double)v.size()) + 1e-9;
}

static int run_case(unsigned T, unsigned H, unsigned D, unsigned NKV, unsigned TOPK,
                    unsigned n_masked) {
    const float scale = 1.0f / sqrtf((float)D);
    const double TOL = 2e-2;
    printf("T=%u H=%u D=%u NKV=%u TOPK=%u masked=%u\n", T, H, D, NKV, TOPK, n_masked);

    std::vector<bf16> q((size_t)T * H * D), kv((size_t)NKV * D);
    std::vector<int> idx((size_t)T * TOPK);
    std::vector<float> sink(H);
    for (auto& v : q) v = f2bf_host(frand());
    for (auto& v : kv) v = f2bf_host(frand());
    /* A real sink is O(1) against scores that are also O(1) here, so it changes
     * the answer materially — a sink far below the scores would make this gate
     * blind to the whole feature. */
    for (auto& v : sink) v = frand();
    for (unsigned t = 0; t < T; t++)
        for (unsigned j = 0; j < TOPK; j++)
            idx[(size_t)t * TOPK + j] =
                (j < n_masked) ? -1 : (int)(lrand48() % NKV); /* -1 tail as the window builder emits */

    bf16 *d_q, *d_kv, *d_out;
    int* d_idx;
    float* d_sink;
    HC_CHECK(hipMalloc(&d_q, q.size() * 2));
    HC_CHECK(hipMalloc(&d_kv, kv.size() * 2));
    HC_CHECK(hipMalloc(&d_out, q.size() * 2));
    HC_CHECK(hipMalloc(&d_idx, idx.size() * 4));
    HC_CHECK(hipMalloc(&d_sink, sink.size() * 4));
    HC_CHECK(hipMemcpy(d_q, q.data(), q.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_kv, kv.data(), kv.size() * 2, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_idx, idx.data(), idx.size() * 4, hipMemcpyHostToDevice));
    HC_CHECK(hipMemcpy(d_sink, sink.data(), sink.size() * 4, hipMemcpyHostToDevice));

    /* One workgroup per (token, head). Deliberately fewer than T*H on one case
     * so the `for (w = slice; w < T*H; w += nblk)` wrap is exercised — a block
     * that handles two work items must reset its LDS partials between them. */
    const unsigned blocks = (T * H > 96u) ? 96u : T * H;
    k_attn<<<blocks, PLOW_THREADS>>>(d_out, d_q, d_kv, d_idx, d_sink, T, H, D, TOPK, scale);
    HC_CHECK(hipDeviceSynchronize());
    std::vector<bf16> got(q.size());
    HC_CHECK(hipMemcpy(got.data(), d_out, got.size() * 2, hipMemcpyDeviceToHost));

    std::vector<double> ref(q.size());
    for (unsigned t = 0; t < T; t++)
        for (unsigned h = 0; h < H; h++) {
            const bf16* qh = &q[((size_t)t * H + h) * D];
            std::vector<double> s(TOPK);
            double m = -INFINITY;
            for (unsigned j = 0; j < TOPK; j++) {
                const int p = idx[(size_t)t * TOPK + j];
                if (p < 0) {
                    s[j] = -INFINITY;
                    continue;
                }
                double d = 0.0;
                for (unsigned e = 0; e < D; e++)
                    d += bf2f_host(qh[e]) * bf2f_host(kv[(size_t)p * D + e]);
                s[j] = d * (double)scale;
                m = std::max(m, s[j]); /* real keys only */
            }
            std::vector<double> o(D, 0.0);
            double l = 0.0;
            for (unsigned j = 0; j < TOPK; j++) {
                const int p = idx[(size_t)t * TOPK + j];
                if (p < 0) continue;
                const double e = exp(s[j] - m);
                l += e;
                for (unsigned d = 0; d < D; d++) o[d] += e * bf2f_host(kv[(size_t)p * D + d]);
            }
            l += exp((double)sink[h] - m); /* denominator only */
            for (unsigned d = 0; d < D; d++) ref[((size_t)t * H + h) * D + d] = o[d] / l;
        }

    /* SPLIT-K must equal the single-kernel path — same op, more blocks. SP=1
     * is batched decode's shared-KV head-group path; SP=4 is B1 split-K. */
    for (unsigned SP : {1u, 4u}) {
        float *d_op, *d_mlp;
        bf16* d_out2;
        HC_CHECK(hipMalloc(&d_op, (size_t)T * H * SP * D * 4));
        HC_CHECK(hipMalloc(&d_mlp, (size_t)T * H * SP * 2 * 4));
        HC_CHECK(hipMalloc(&d_out2, q.size() * 2));
        k_split<<<T * H * SP, PLOW_THREADS>>>(d_op, d_mlp, d_q, d_kv, d_idx, T, H, D, TOPK, SP,
                                              scale);
        HC_CHECK(hipDeviceSynchronize());
        k_merge<<<T * H, PLOW_THREADS>>>(d_out2, d_op, d_mlp, d_sink, T, H, D, SP);
        HC_CHECK(hipDeviceSynchronize());
        std::vector<bf16> g2(q.size());
        HC_CHECK(hipMemcpy(g2.data(), d_out2, g2.size() * 2, hipMemcpyDeviceToHost));
        double wsp = 0.0;
        for (size_t i = 0; i < g2.size(); i++)
            wsp = std::max(wsp, fabs(bf2f_host(got[i]) - bf2f_host(g2[i])));
        printf("  split-%u vs single             max abs %.3e  %s\n", SP, wsp,
               wsp <= 5e-2 ? "ok" : "FAIL");
        hipFree(d_op);
        hipFree(d_mlp);
        hipFree(d_out2);
    }

    double worst = 0.0;
    const double r = rms_of(ref);
    for (size_t i = 0; i < ref.size(); i++)
        worst = std::max(worst, fabs(ref[i] - bf2f_host(got[i])) / r);
    printf("  %-28s max rel %.3e  %s\n", "sparse_attn+sink", worst, worst <= TOL ? "ok" : "FAIL");

    hipFree(d_q);
    hipFree(d_kv);
    hipFree(d_out);
    hipFree(d_idx);
    hipFree(d_sink);
    return worst <= TOL ? 0 : 1;
}

int main(void) {
    srand48(7);
    int bad = 0;
    /* The shipped attention geometry: 64 heads, one KV head, head_dim 512.
     * TOPK 640 = the 128-wide window plus the indexer's 512 compressed entries. */
    bad |= run_case(1, 64, 512, 2048, 640, 0);
    /* A masked prefix, as the window index builder emits at the start of a
     * sequence, plus T>1 across several workgroups. */
    bad |= run_case(5, 64, 512, 2048, 640, 96);
    /* The indexer's own width. */
    bad |= run_case(3, 8, 128, 512, 128, 16);
    /* SEQUENCE START, where the sink actually bites. With 640 keys the sink is
     * one term in a 640-term denominator and dropping it moves the output by
     * only ~2e-2 — barely over this gate's tolerance. Near position 0 the
     * window holds a handful of valid entries and the same omission is an
     * order of magnitude louder, so this row is what gives the gate real power
     * over the sink rather than a marginal pass/fail. */
    bad |= run_case(2, 64, 512, 64, 8, 4);
    printf("%s\n", bad ? "FAIL" : "PASS — d_v4_sparse_attn matches sparse_attn_kernel");
    return bad;
}
