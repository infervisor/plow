/* attention_gfx950_test.c — FlashAttention correctness against a CPU reference.
 *
 * Covers both Gemma 4 geometries and both phases:
 *   prefill hd=256, GQA 2:1, sliding window 1024 (inclusive)
 *   prefill hd=256, GQA 2:1, full causal
 *   prefill hd=512, GQA 8:1, full causal          <- the global layers
 *   decode  hd=256 / hd=512, split-KV + merge
 *
 * The reference is written from HF `modeling_gemma4.py` semantics, not from the
 * kernel: scale = 1.0 (there is NO 1/sqrt(head_dim)), causal is kv <= q, and the
 * sliding window is INCLUSIVE of the current token (0 <= q-kv <= window-1).
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

static int fails = 0;

/* O[q][h][d] = softmax_kv( scale * Q[q][h].K[hkv][kv] , masked ) . V[hkv][kv]
 *
 * K and V are HEAD-MAJOR: [n_kv_head][n_kv][hd]. See dev_isa.h -- flash reads one head at a
 * time, so head-major makes its rows contiguous. The buffers here are random, so the layout is
 * a free choice; it just has to match the kernel. */
static void ref_attn(float* O, const bf16* Q, const bf16* K, const bf16* V, unsigned n_q,
                     unsigned n_kv, unsigned n_head, unsigned n_kv_head, unsigned hd,
                     unsigned q_pos0, unsigned window, float scale) {
    const unsigned gqa = n_head / n_kv_head;
    double* s = malloc(sizeof(double) * n_kv);
    for (unsigned q = 0; q < n_q; q++)
        for (unsigned h = 0; h < n_head; h++) {
            const unsigned hkv = h / gqa;
            const unsigned qg = q_pos0 + q;
            double mx = -1e300;
            for (unsigned kv = 0; kv < n_kv; kv++) {
                const int keep = (kv <= qg) && (!window || (qg - kv) < window);
                if (!keep) { s[kv] = -1e300; continue; }
                double d = 0.0;
                for (unsigned i = 0; i < hd; i++)
                    d += (double)bf2f(Q[((size_t)q * n_head + h) * hd + i]) *
                         bf2f(K[((size_t)hkv * n_kv + kv) * hd + i]);
                s[kv] = d * scale;
                if (s[kv] > mx) mx = s[kv];
            }
            double sum = 0.0;
            for (unsigned kv = 0; kv < n_kv; kv++) {
                s[kv] = (s[kv] <= -1e299) ? 0.0 : exp(s[kv] - mx);
                sum += s[kv];
            }
            const double inv = (sum > 0) ? 1.0 / sum : 0.0;
            for (unsigned d = 0; d < hd; d++) {
                double acc = 0.0;
                for (unsigned kv = 0; kv < n_kv; kv++)
                    if (s[kv] != 0.0)
                        acc += s[kv] * bf2f(V[((size_t)hkv * n_kv + kv) * hd + d]);
                O[((size_t)q * n_head + h) * hd + d] = (float)(acc * inv);
            }
        }
    free(s);
}

/* Error is normalised by the OUTPUT's own scale, not per-element.
 *
 * O is a softmax-weighted average of V, so individual elements pass through zero
 * by coincidence; dividing by |want| there reports a huge "relative error" for an
 * absolute error that is entirely negligible. The meaningful question is the
 * error against the magnitude of the signal, so we report
 *   max|got - want| / max|want|   and   rms(got - want) / rms(want).
 *
 * Expect ~1e-3, NOT ~1e-7: like every FlashAttention, the prefill kernel rounds
 * the softmax probabilities P to bf16 before feeding them to the matrix core, so
 * each of the ~n_kv accumulated terms carries a ~0.4% quantisation. The decode
 * kernel keeps P in f32 (it uses no MFMA) and is correspondingly tighter. */
static void check(const char* what, const bf16* got, const float* want, size_t n) {
    double max_w = 0.0, max_d = 0.0, se = 0.0, sw = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs(bf2f(got[i]) - want[i]);
        max_w = fmax(max_w, fabs(want[i]));
        max_d = fmax(max_d, d);
        se += d * d;
        sw += (double)want[i] * want[i];
    }
    const double rel_max = max_d / (max_w + 1e-12);
    const double rel_rms = sqrt(se / n) / (sqrt(sw / n) + 1e-12);
    const int ok = rel_max < 2e-2 && rel_rms < 5e-3;
    printf("  %-38s %s  (max %.4f  rms %.5f  of |O|max=%.3f)\n", what, ok ? "PASS" : "FAIL",
           rel_max, rel_rms, max_w);
    if (!ok) fails++;
}

static plow_hsa* H;
static void* dev(size_t b) { return plow_hsa_alloc(H, 0, b); }

static void prefill_case(plow_hsa_kernel* k, plow_hsa_kernel* mk, unsigned NCU,
                         const char* label, unsigned hd, unsigned n_q, unsigned n_kv,
                         unsigned n_head, unsigned n_kv_head, unsigned window,
                         unsigned nsplit) {
    const float SCALE = 1.0f; /* Gemma: no 1/sqrt(d) */
    const size_t nq = (size_t)n_q * n_head * hd;
    const size_t nkv = (size_t)n_kv * n_kv_head * hd;

    bf16* hQ = plow_hsa_alloc_host(H, nq * 2);
    bf16* hK = plow_hsa_alloc_host(H, nkv * 2);
    bf16* hV = plow_hsa_alloc_host(H, nkv * 2);
    bf16* hO = plow_hsa_alloc_host(H, nq * 2);
    /* Small magnitudes: scale=1.0 over hd=512 means logits of ~sqrt(512)*var, and
     * we want to exercise the softmax, not overflow it. */
    for (size_t i = 0; i < nq; i++) hQ[i] = f2bf(frand() * 0.15f);
    for (size_t i = 0; i < nkv; i++) { hK[i] = f2bf(frand() * 0.15f); hV[i] = f2bf(frand()); }

    void *dQ = dev(nq * 2), *dK = dev(nkv * 2), *dV = dev(nkv * 2), *dO = dev(nq * 2);
    /* Prefill now emits UNNORMALIZED split partials; the merge folds them. */
    void* dOp = dev((size_t)n_q * n_head * nsplit * hd * sizeof(float));
    void* dMl = dev((size_t)n_q * n_head * nsplit * 2 * sizeof(float));
    plow_hsa_copy_h2d(H, 0, dQ, hQ, nq * 2);
    plow_hsa_copy_h2d(H, 0, dK, hK, nkv * 2);
    plow_hsa_copy_h2d(H, 0, dV, hV, nkv * 2);

    struct __attribute__((packed)) {
        void *op, *ml; const void *q, *k, *v;
        unsigned n_q, n_kv, n_head, n_kv_head, q_pos0, window; float scale; unsigned nsplit;
    } a = {dOp, dMl, dQ, dK, dV, n_q, n_kv, n_head, n_kv_head, 0, window, SCALE, nsplit};

    if (plow_hsa_launch(H, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a)) != 0) {
        fprintf(stderr, "launch: %s\n", plow_hsa_last_error()); fails++; return;
    }
    plow_hsa_wait(H, 0);

    struct __attribute__((packed)) {
        void* o; const void *op, *ml; unsigned n_batch, n_head, nsplit;
    } m = {dO, dOp, dMl, n_q, n_head, nsplit};
    if (plow_hsa_launch(H, 0, mk, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &m, sizeof(m)) != 0) {
        fprintf(stderr, "merge launch: %s\n", plow_hsa_last_error()); fails++; return;
    }
    plow_hsa_wait(H, 0);
    plow_hsa_copy_d2h(H, 0, hO, dO, nq * 2);

    float* want = malloc(nq * sizeof(float));
    ref_attn(want, hQ, hK, hV, n_q, n_kv, n_head, n_kv_head, hd, 0, window, SCALE);
    check(label, hO, want, nq);
    free(want);
    plow_hsa_free(H, dQ); plow_hsa_free(H, dK); plow_hsa_free(H, dV); plow_hsa_free(H, dO);
}

static void decode_case(plow_hsa_kernel* kd, plow_hsa_kernel* km, unsigned NCU,
                        const char* label, unsigned hd, unsigned n_kv, unsigned n_head,
                        unsigned n_kv_head, unsigned window, unsigned nsplit) {
    const float SCALE = 1.0f;
    const unsigned B = 2;
    const size_t nq = (size_t)B * n_head * hd;
    const size_t nkv = (size_t)B * n_kv * n_kv_head * hd;

    bf16* hQ = plow_hsa_alloc_host(H, nq * 2);
    bf16* hK = plow_hsa_alloc_host(H, nkv * 2);
    bf16* hV = plow_hsa_alloc_host(H, nkv * 2);
    bf16* hO = plow_hsa_alloc_host(H, nq * 2);
    int* hLen = plow_hsa_alloc_host(H, B * 4);
    for (size_t i = 0; i < nq; i++) hQ[i] = f2bf(frand() * 0.15f);
    for (size_t i = 0; i < nkv; i++) { hK[i] = f2bf(frand() * 0.15f); hV[i] = f2bf(frand()); }
    for (unsigned b = 0; b < B; b++) hLen[b] = (int)n_kv;

    void *dQ = dev(nq * 2), *dK = dev(nkv * 2), *dV = dev(nkv * 2), *dO = dev(nq * 2);
    void* dLen = dev(B * 4);
    void* dOp = dev((size_t)B * n_head * nsplit * hd * 4);
    void* dMl = dev((size_t)B * n_head * nsplit * 2 * 4);
    plow_hsa_copy_h2d(H, 0, dQ, hQ, nq * 2);
    plow_hsa_copy_h2d(H, 0, dK, hK, nkv * 2);
    plow_hsa_copy_h2d(H, 0, dV, hV, nkv * 2);
    plow_hsa_copy_h2d(H, 0, dLen, hLen, B * 4);

    struct __attribute__((packed)) {
        void* op; void* ml; const void* q; const void* k; const void* v; const void* len;
        unsigned n_batch, n_head, n_kv_head, kv_stride, window; float scale; unsigned nsplit;
    } a = {dOp, dMl, dQ, dK, dV, dLen, B, n_head, n_kv_head, n_kv, window, SCALE, nsplit};
    plow_hsa_launch(H, 0, kd, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));

    struct __attribute__((packed)) {
        void* o; const void* op; const void* ml; unsigned n_batch, n_head, nsplit;
    } m = {dO, dOp, dMl, B, n_head, nsplit};
    plow_hsa_launch(H, 0, km, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &m, sizeof(m));
    plow_hsa_wait(H, 0);
    plow_hsa_copy_d2h(H, 0, hO, dO, nq * 2);

    /* The decode query sits at position len-1. Reference one batch at a time. */
    float* want = malloc(nq * sizeof(float));
    for (unsigned b = 0; b < B; b++)
        ref_attn(want + (size_t)b * n_head * hd, hQ + (size_t)b * n_head * hd,
                 hK + (size_t)b * n_kv * n_kv_head * hd, hV + (size_t)b * n_kv * n_kv_head * hd,
                 1, n_kv, n_head, n_kv_head, hd, n_kv - 1, window, SCALE);
    check(label, hO, want, nq);
    free(want);
}

int main(void) {
    H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &lds);
    printf("dev0: %s  CUs=%u  LDS=%u B\n\n", nm, cus, lds);

    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(H, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }

    plow_hsa_kernel p128, m128;
    if (plow_hsa_get_kernel(H, 0, "gemma_flash_prefill_128", &p128) ||
        plow_hsa_get_kernel(H, 0, "gemma_flash_merge_128", &m128)) {
        fprintf(stderr, "sym128: %s\n", plow_hsa_last_error()); return 1;
    }

    plow_hsa_kernel p256, p512, d256, d512, m256, m512;
    if (plow_hsa_get_kernel(H, 0, "gemma_flash_prefill_256", &p256) ||
        plow_hsa_get_kernel(H, 0, "gemma_flash_prefill_512", &p512) ||
        plow_hsa_get_kernel(H, 0, "gemma_flash_decode_256", &d256) ||
        plow_hsa_get_kernel(H, 0, "gemma_flash_decode_512", &d512) ||
        plow_hsa_get_kernel(H, 0, "gemma_flash_merge_256", &m256) ||
        plow_hsa_get_kernel(H, 0, "gemma_flash_merge_512", &m512)) {
        fprintf(stderr, "sym: %s\n", plow_hsa_last_error()); return 1;
    }
    printf("prefill_256 LDS=%uB  prefill_512 LDS=%uB\n\n", p256.group_segment_size,
           p512.group_segment_size);
    srand(13);

    /* D=128 (Llama/Qwen), full causal. Previously UNCOVERED — the golden suite only had
     * Gemma's hd=256/512. n_q spans several 128-row q-tiles and n_kv crosses KV-tile
     * boundaries (and non-multiples of the block) so masking + split-KV carving are
     * exercised. Validates whichever FA_BKV_D128 the object was built with (32 shipped,
     * 64 the experimental two-subtile softmax). */
    printf("Prefill (Llama/Qwen hd=128):\n");
    prefill_case(&p128, &m128, cus, "hd=128 causal  n_q=300 nsplit=1", 128, 300, 300, 8, 2, 0, 1);
    prefill_case(&p128, &m128, cus, "hd=128 causal  n_q=300 nsplit=4", 128, 300, 300, 8, 2, 0, 4);
    prefill_case(&p128, &m128, cus, "hd=128 causal  n_q=300 nsplit=8", 128, 300, 300, 8, 2, 0, 8);
    prefill_case(&p128, &m128, cus, "hd=128 causal  n_q=520 nsplit=1", 128, 520, 520, 32, 8, 0, 1);
    prefill_case(&p128, &m128, cus, "hd=128 causal  n_q=97  nsplit=3", 128, 97, 97, 8, 2, 0, 3);
    prefill_case(&p128, &m128, cus, "hd=128 GQA4:1  n_q=640 nsplit=8", 128, 640, 640, 32, 8, 0, 8);

    printf("\nPrefill (Gemma 4 31B geometries):\n");
    /* sliding layers: 32 q heads / 16 kv heads, hd 256, window 1024 */
    /* Sweep nsplit: a split-KV prefill must give the SAME answer for every nsplit, or the
     * online-softmax merge is wrong. nsplit=1 also has to work, because the program always
     * emits a merge. */
    prefill_case(&p256, &m256, cus, "hd=256 causal  nsplit=1", 256, 300, 300, 8, 4, 0, 1);
    prefill_case(&p256, &m256, cus, "hd=256 causal  nsplit=4", 256, 300, 300, 8, 4, 0, 4);
    prefill_case(&p256, &m256, cus, "hd=256 causal  nsplit=8", 256, 300, 300, 8, 4, 0, 8);
    prefill_case(&p256, &m256, cus, "hd=256 slide64 nsplit=1", 256, 300, 300, 8, 4, 64, 1);
    prefill_case(&p256, &m256, cus, "hd=256 slide64 nsplit=4", 256, 300, 300, 8, 4, 64, 4);
    prefill_case(&p256, &m256, cus, "hd=256 slide128 n_q=520 nsplit=8", 256, 520, 520, 8, 4, 128, 8);
    /* global layers: 32 q heads / 4 kv heads, hd 512, full causal */
    prefill_case(&p512, &m512, cus, "hd=512 causal  nsplit=1", 512, 300, 300, 8, 1, 0, 1);
    prefill_case(&p512, &m512, cus, "hd=512 causal  nsplit=4", 512, 300, 300, 8, 1, 0, 4);
    prefill_case(&p512, &m512, cus, "hd=512 MQA n_q=140 nsplit=8", 512, 140, 140, 8, 1, 0, 8);

    printf("\nDecode (split-KV + merge):\n");
    decode_case(&d256, &m256, cus, "hd=256 GQA 2:1 causal, nsplit=4", 256, 500, 8, 4, 0, 4);
    decode_case(&d256, &m256, cus, "hd=256 GQA 2:1 sliding w=64, nsplit=4", 256, 500, 8, 4, 64, 4);
    decode_case(&d512, &m512, cus, "hd=512 MQA causal, nsplit=8", 512, 700, 8, 1, 0, 8);

    printf("\n%s (%d failure%s)\n", fails ? "ATTENTION FAILED" : "ATTENTION CORRECT", fails,
           fails == 1 ? "" : "s");
    plow_hsa_shutdown(H);
    return fails ? 1 : 0;
}
