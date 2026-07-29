/* mla_gfx950_test.c — DeepSeek MLA decode correctness on device vs the Rust oracle. [DEEPSEEK-MLA]
 *
 * Drives the three MLA decode ops in isolation (like attention_gfx950_test.c):
 *     mla_flash_decode_512   latent flash: q_abs.C_kv + q_rope.K_rope, PV on the latent
 *     gemma_flash_merge_512  the split-KV LSE merge, at D=kv_lora_rank=512 (REUSED)
 *     mla_o_uv_fold_512      the W_uv fold of the merged latent -> v_head_dim
 * and checks the final o[n_head][v_head_dim] against the golden produced by the Rust
 * reference (runtime/tests/mla_ref.rs) — device MLA output == CPU absorbed-MLA reference.
 *
 * The fixture (fixture.bin, "MLA1") carries, per case, the bf16 inputs and the f32
 * golden; the Rust side owns all the MLA math, this harness only moves bytes + launches.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned short bf16;
static float bf2f(bf16 b) {
    unsigned u = (unsigned)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static int fails = 0;

/* Same output-scaled error metric as attention_gfx950_test.c: O is a softmax-weighted
 * average folded by W_uv, so elements pass through zero and a per-element relative error
 * is meaningless. Report max|got-want|/max|want| and rms(got-want)/rms(want). Expect
 * ~1e-3: flash keeps P in f32 (no MFMA) but the latent/query/W_uv are bf16 and the merged
 * latent is rounded to bf16 before the fold. */
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
    printf("  %-42s %s  (max %.4f  rms %.5f  of |O|max=%.3f)\n", what, ok ? "PASS" : "FAIL",
           rel_max, rel_rms, max_w);
    if (!ok) fails++;
}

static plow_hsa* H;
static void* dev(size_t b) { return plow_hsa_alloc(H, 0, b); }

/* Little-endian readers over the fixture byte stream. */
static const uint8_t* P;
static uint32_t rd_u32(void) { uint32_t v; memcpy(&v, P, 4); P += 4; return v; }
static float rd_f32(void) { float v; memcpy(&v, P, 4); P += 4; return v; }

int main(int argc, char** argv) {
    const char* fx = argc > 1 ? argv[1] : "fixture.bin";
    FILE* ff = fopen(fx, "rb");
    if (!ff) { perror(fx); return 1; }
    fseek(ff, 0, SEEK_END); long fn = ftell(ff); fseek(ff, 0, SEEK_SET);
    uint8_t* fixture = malloc(fn);
    if (fread(fixture, 1, fn, ff) != (size_t)fn) return 1;
    fclose(ff);
    P = fixture;
    if (rd_u32() != 0x4d4c4131u) { fprintf(stderr, "bad fixture magic\n"); return 1; }
    const uint32_t n_cases = rd_u32();

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

    /* MLA_MFMA=1 exercises the head-packed MFMA decode instead of the scalar GF kernel. */
    const int use_mfma = getenv("MLA_MFMA") && atoi(getenv("MLA_MFMA"));
    const char* dsym = use_mfma ? "mla_flash_decode_mfma_512" : "mla_flash_decode_512";
    const char* gsym = use_mfma ? "mla_gather_decode_mfma_512" : "mla_gather_decode_512";
    plow_hsa_kernel kd, kg, km, kf;
    if (plow_hsa_get_kernel(H, 0, dsym, &kd) ||
        plow_hsa_get_kernel(H, 0, gsym, &kg) ||
        plow_hsa_get_kernel(H, 0, "gemma_flash_merge_512", &km) ||
        plow_hsa_get_kernel(H, 0, "mla_o_uv_fold_512", &kf)) {
        fprintf(stderr, "sym: %s\n", plow_hsa_last_error()); return 1;
    }
    printf("kernel: %s\n", dsym);
    printf("mla_flash_decode_512 LDS=%uB\n\n", kd.group_segment_size);

    printf("DeepSeek MLA decode (device vs Rust absorbed-MLA oracle):\n");
    for (uint32_t ci = 0; ci < n_cases; ci++) {
        const uint32_t n_head = rd_u32(), DK = rd_u32(), DR = rd_u32(), Vd = rd_u32();
        const uint32_t ctx = rd_u32(), nsplit = rd_u32(), top_k = rd_u32();
        const float scale = rd_f32();
        const unsigned B = 1;
        const int32_t* hIdx = NULL;
        if (top_k > 0) { hIdx = (const int32_t*)P; P += (size_t)top_k * 4; }

        const size_t nckv = (size_t)ctx * DK, nkr = (size_t)ctx * DR;
        const size_t nqa = (size_t)n_head * DK, nqr = (size_t)n_head * DR;
        const size_t nwuv = (size_t)n_head * DK * Vd, no = (size_t)n_head * Vd;

        const bf16* hCkv = (const bf16*)P; P += nckv * 2;
        const bf16* hKr = (const bf16*)P; P += nkr * 2;
        const bf16* hQa = (const bf16*)P; P += nqa * 2;
        const bf16* hQr = (const bf16*)P; P += nqr * 2;
        const bf16* hWuv = (const bf16*)P; P += nwuv * 2;
        const float* golden = (const float*)P; P += no * 4;

        /* host-pinned staging (h2d requires alloc_host memory) */
        void* pCkv = plow_hsa_alloc_host(H, nckv * 2); memcpy(pCkv, hCkv, nckv * 2);
        void* pKr = plow_hsa_alloc_host(H, nkr * 2); memcpy(pKr, hKr, nkr * 2);
        void* pQa = plow_hsa_alloc_host(H, nqa * 2); memcpy(pQa, hQa, nqa * 2);
        void* pQr = plow_hsa_alloc_host(H, nqr * 2); memcpy(pQr, hQr, nqr * 2);
        void* pWuv = plow_hsa_alloc_host(H, nwuv * 2); memcpy(pWuv, hWuv, nwuv * 2);
        int* pLen = plow_hsa_alloc_host(H, B * 4); pLen[0] = (int)ctx;
        bf16* hO = plow_hsa_alloc_host(H, no * 2);

        void* dCkv = dev(nckv * 2); plow_hsa_copy_h2d(H, 0, dCkv, pCkv, nckv * 2);
        void* dKr = dev(nkr * 2); plow_hsa_copy_h2d(H, 0, dKr, pKr, nkr * 2);
        void* dQa = dev(nqa * 2); plow_hsa_copy_h2d(H, 0, dQa, pQa, nqa * 2);
        void* dQr = dev(nqr * 2); plow_hsa_copy_h2d(H, 0, dQr, pQr, nqr * 2);
        void* dWuv = dev(nwuv * 2); plow_hsa_copy_h2d(H, 0, dWuv, pWuv, nwuv * 2);
        void* dLen = dev(B * 4); plow_hsa_copy_h2d(H, 0, dLen, pLen, B * 4);

        void* dOp = dev((size_t)B * n_head * nsplit * DK * 4);
        void* dMl = dev((size_t)B * n_head * nsplit * 2 * 4);
        void* dOlat = dev((size_t)B * n_head * DK * 2);
        void* dO = dev(no * 2);

        if (top_k == 0) {
            struct __attribute__((packed)) {
                void *op, *ml; const void *qa, *qr, *ckv, *kr, *len;
                unsigned n_batch, n_head, kv_stride, window; float scale; unsigned nsplit;
            } a = {dOp, dMl, dQa, dQr, dCkv, dKr, dLen, B, n_head, ctx, 0, scale, nsplit};
            if (plow_hsa_launch(H, 0, &kd, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a,
                                sizeof(a)) != 0) {
                fprintf(stderr, "decode launch: %s\n", plow_hsa_last_error()); fails++;
            }
        } else {
            void* pIdx = plow_hsa_alloc_host(H, (size_t)top_k * 4);
            memcpy(pIdx, hIdx, (size_t)top_k * 4);
            void* dIdx = dev((size_t)top_k * 4);
            plow_hsa_copy_h2d(H, 0, dIdx, pIdx, (size_t)top_k * 4);
            struct __attribute__((packed)) {
                void *op, *ml; const void *qa, *qr, *ckv, *kr, *len, *idx;
                unsigned top_k, n_batch, n_head, kv_stride; float scale; unsigned nsplit;
            } a = {dOp, dMl, dQa, dQr, dCkv, dKr, dLen, dIdx, top_k, B, n_head, ctx, scale, nsplit};
            if (plow_hsa_launch(H, 0, &kg, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a,
                                sizeof(a)) != 0) {
                fprintf(stderr, "gather launch: %s\n", plow_hsa_last_error()); fails++;
            }
            /* dIdx is read by the async kernel; leave it live until process exit (bounded). */
        }

        struct __attribute__((packed)) {
            void* o; const void *op, *ml; unsigned n_batch, n_head, nsplit;
        } m = {dOlat, dOp, dMl, B, n_head, nsplit};
        plow_hsa_launch(H, 0, &km, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &m,
                        sizeof(m));

        struct __attribute__((packed)) {
            void* o; const void *olat, *wuv; unsigned n_batch, n_head, v;
        } fld = {dO, dOlat, dWuv, B, n_head, Vd};
        plow_hsa_launch(H, 0, &kf, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &fld,
                        sizeof(fld));
        plow_hsa_wait(H, 0);
        plow_hsa_copy_d2h(H, 0, hO, dO, no * 2);

        char label[96];
        if (top_k == 0)
            snprintf(label, sizeof(label), "dense   n_head=%u ctx=%u nsplit=%u", n_head, ctx, nsplit);
        else
            snprintf(label, sizeof(label), "gather  n_head=%u ctx=%u nsplit=%u top_k=%u", n_head,
                     ctx, nsplit, top_k);
        check(label, hO, golden, no);
        if (getenv("MLA_DBG") && ci == 0) {
            printf("    got:  "); for (int z=0; z<6; z++) printf("%.5f ", bf2f(hO[z])); printf("\n");
            printf("    want: "); for (int z=0; z<6; z++) printf("%.5f ", golden[z]); printf("\n");
        }

        plow_hsa_free(H, dCkv); plow_hsa_free(H, dKr); plow_hsa_free(H, dQa);
        plow_hsa_free(H, dQr); plow_hsa_free(H, dWuv); plow_hsa_free(H, dLen);
        plow_hsa_free(H, dOp); plow_hsa_free(H, dMl); plow_hsa_free(H, dOlat); plow_hsa_free(H, dO);
    }

    /* ====================================================================
     * PHASE 2 — MLA PREFILL (ops 51 / 55), the Kimi K2.7 / DeepSeek / GLM-5.2
     * long-context path. Until now these arms compiled and shipped (they are two
     * of the seventeen register-gated code objects) and had NEVER been run: the
     * fixture above is decode-only, and there is no CPU oracle for a multi-token
     * MLA prefill.
     *
     * There does not need to be one. The prefill kernel IS the decode kernel with
     * n_tok > 1 (op_attention.h: "these are wrappers, not kernels"), and its causal
     * bound is exact and analytic: with kv_len = L and n_tok = T, query row t sits at
     * qpos = L-T+t and attends rows [0, L-T+t+1). That is, ELEMENT FOR ELEMENT, what
     * the decode kernel computes when handed kv_len = L-T+t+1 and nsplit = 1 — same
     * body, same GF, same tile loop from kv0 = 0, so the f32 accumulation order is
     * identical too. So the already-validated decode kernel is a BIT-EXACT oracle for
     * every row of a prefill, and the check below is memcmp, not a tolerance.
     *
     * This is the check the kernel's own "ORACLE NOTE for hardware day" asks for
     * (test_kernels.hip, above mla_flash_prefill_512), including its T=1 base case.
     *
     * Second pass over the same fixture bytes: the stream is a fixed layout, so
     * re-reading it from the top is cheaper than retaining every case.
     * ==================================================================== */
    plow_hsa_kernel kpf, kgpf;
    if (plow_hsa_get_kernel(H, 0, "mla_flash_prefill_512", &kpf) ||
        plow_hsa_get_kernel(H, 0, "mla_gather_prefill_512", &kgpf)) {
        fprintf(stderr, "prefill sym: %s\n", plow_hsa_last_error());
        return 1;
    }
    /* T=1 first (the base case: prefill MUST degenerate to the validated decode),
     * then T=4 (the causal bound actually doing work). */
    static const unsigned TOKS[] = {1, 4};
    printf("\nMLA PREFILL (device prefill vs the validated device decode, bit-exact):\n");
    P = fixture + 8; /* past magic + n_cases */
    for (uint32_t ci = 0; ci < n_cases; ci++) {
        const uint32_t n_head = rd_u32(), DK = rd_u32(), DR = rd_u32(), Vd = rd_u32();
        const uint32_t ctx = rd_u32(), nsplit_fx = rd_u32(), top_k = rd_u32();
        const float scale = rd_f32();
        (void)nsplit_fx; /* prefill is nsplit=1 by construction; the oracle run is too */
        const unsigned B = 1;
        const int32_t* hIdx = NULL;
        if (top_k > 0) { hIdx = (const int32_t*)P; P += (size_t)top_k * 4; }

        const size_t nckv = (size_t)ctx * DK, nkr = (size_t)ctx * DR;
        const size_t nqa = (size_t)n_head * DK, nqr = (size_t)n_head * DR;
        const size_t nwuv = (size_t)n_head * DK * Vd;
        const bf16* hCkv = (const bf16*)P; P += nckv * 2;
        const bf16* hKr = (const bf16*)P; P += nkr * 2;
        const bf16* hQa = (const bf16*)P; P += nqa * 2;
        const bf16* hQr = (const bf16*)P; P += nqr * 2;
        P += nwuv * 2;                       /* W_uv: the fold is phase 1's business */
        P += (size_t)n_head * Vd * 4;        /* golden o                             */

        void* pCkv = plow_hsa_alloc_host(H, nckv * 2); memcpy(pCkv, hCkv, nckv * 2);
        void* pKr = plow_hsa_alloc_host(H, nkr * 2); memcpy(pKr, hKr, nkr * 2);
        void* dCkv = dev(nckv * 2); plow_hsa_copy_h2d(H, 0, dCkv, pCkv, nckv * 2);
        void* dKr = dev(nkr * 2); plow_hsa_copy_h2d(H, 0, dKr, pKr, nkr * 2);
        void* dIdx1 = NULL;
        if (top_k > 0) {
            void* p = plow_hsa_alloc_host(H, (size_t)top_k * 4);
            memcpy(p, hIdx, (size_t)top_k * 4);
            dIdx1 = dev((size_t)top_k * 4);
            plow_hsa_copy_h2d(H, 0, dIdx1, p, (size_t)top_k * 4);
        }

        for (unsigned ti = 0; ti < sizeof(TOKS) / sizeof(TOKS[0]); ti++) {
            const unsigned T = TOKS[ti];
            if (ctx < T) continue;
            /* Q replicated across the T rows. Identical queries make every row's EXPECTED
             * answer differ only by its causal bound, which is precisely the thing under
             * test — a per-row random Q would test the same code and make the oracle run
             * harder to set up for no extra coverage. */
            const size_t nqa_t = nqa * T, nqr_t = nqr * T;
            bf16* pQa = plow_hsa_alloc_host(H, nqa_t * 2);
            bf16* pQr = plow_hsa_alloc_host(H, nqr_t * 2);
            for (unsigned t = 0; t < T; t++) {
                memcpy(pQa + (size_t)t * nqa, hQa, nqa * 2);
                memcpy(pQr + (size_t)t * nqr, hQr, nqr * 2);
            }
            void* dQa = dev(nqa_t * 2); plow_hsa_copy_h2d(H, 0, dQa, pQa, nqa_t * 2);
            void* dQr = dev(nqr_t * 2); plow_hsa_copy_h2d(H, 0, dQr, pQr, nqr_t * 2);
            int* pLen = plow_hsa_alloc_host(H, 4); pLen[0] = (int)ctx;
            void* dLen = dev(4); plow_hsa_copy_h2d(H, 0, dLen, pLen, 4);

            const size_t nop_t = (size_t)T * n_head * DK, nml_t = (size_t)T * n_head * 2;
            void* dOp = dev(nop_t * 4);
            void* dMl = dev(nml_t * 4);
            /* Selected-index table is [b][t][top_k] for a gathered prefill — replicate the
             * decode's single row T times so every query row selects the same set and the
             * decode oracle applies unchanged. */
            void* dIdxT = NULL;
            if (top_k > 0) {
                int32_t* p = plow_hsa_alloc_host(H, (size_t)T * top_k * 4);
                for (unsigned t = 0; t < T; t++) memcpy(p + (size_t)t * top_k, hIdx, (size_t)top_k * 4);
                dIdxT = dev((size_t)T * top_k * 4);
                plow_hsa_copy_h2d(H, 0, dIdxT, p, (size_t)T * top_k * 4);
            }

            if (top_k == 0) {
                struct __attribute__((packed)) {
                    void *op, *ml; const void *qa, *qr, *ckv, *kr, *len;
                    unsigned n_batch, n_tok, n_head, kv_stride, window; float scale;
                } a = {dOp, dMl, dQa, dQr, dCkv, dKr, dLen, B, T, n_head, ctx, 0, scale};
                if (plow_hsa_launch(H, 0, &kpf, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                    0, &a, sizeof(a)) != 0) {
                    fprintf(stderr, "prefill launch: %s\n", plow_hsa_last_error()); fails++;
                }
            } else {
                struct __attribute__((packed)) {
                    void *op, *ml; const void *qa, *qr, *ckv, *kr, *len, *idx;
                    unsigned top_k, n_batch, n_tok, n_head, kv_stride; float scale;
                } a = {dOp, dMl, dQa, dQr, dCkv, dKr, dLen, dIdxT, top_k, B, T, n_head, ctx, scale};
                if (plow_hsa_launch(H, 0, &kgpf, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                    0, &a, sizeof(a)) != 0) {
                    fprintf(stderr, "gather prefill launch: %s\n", plow_hsa_last_error()); fails++;
                }
            }
            plow_hsa_wait(H, 0);
            float* hOp = plow_hsa_alloc_host(H, nop_t * 4);
            float* hMl = plow_hsa_alloc_host(H, nml_t * 4);
            plow_hsa_copy_d2h(H, 0, hOp, dOp, nop_t * 4);
            plow_hsa_copy_d2h(H, 0, hMl, dMl, nml_t * 4);

            /* ORACLE: one decode launch per query row, at that row's causal context. */
            const size_t nop1 = (size_t)n_head * DK, nml1 = (size_t)n_head * 2;
            void* dOp1 = dev(nop1 * 4);
            void* dMl1 = dev(nml1 * 4);
            float* hOp1 = plow_hsa_alloc_host(H, nop1 * 4);
            float* hMl1 = plow_hsa_alloc_host(H, nml1 * 4);
            int* pLen1 = plow_hsa_alloc_host(H, 4);
            void* dLen1 = dev(4);
            unsigned bad = 0, checked = 0;
            for (unsigned t = 0; t < T; t++) {
                pLen1[0] = (int)(ctx - T + 1 + t); /* == qpos+1: this row's causal end */
                plow_hsa_copy_h2d(H, 0, dLen1, pLen1, 4);
                if (top_k == 0) {
                    struct __attribute__((packed)) {
                        void *op, *ml; const void *qa, *qr, *ckv, *kr, *len;
                        unsigned n_batch, n_head, kv_stride, window; float scale; unsigned nsplit;
                    } a = {dOp1, dMl1, dQa, dQr, dCkv, dKr, dLen1, B, n_head, ctx, 0, scale, 1};
                    plow_hsa_launch(H, 0, &kd, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0,
                                    &a, sizeof(a));
                } else {
                    /* The gathered set is causal by construction, so a gathered row does NOT
                     * depend on kv_len — every row must equal the SAME decode result. */
                    struct __attribute__((packed)) {
                        void *op, *ml; const void *qa, *qr, *ckv, *kr, *len, *idx;
                        unsigned top_k, n_batch, n_head, kv_stride; float scale; unsigned nsplit;
                    } a = {dOp1, dMl1, dQa, dQr, dCkv, dKr, dLen1, dIdx1, top_k, B, n_head, ctx,
                           scale, 1};
                    plow_hsa_launch(H, 0, &kg, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0,
                                    &a, sizeof(a));
                }
                plow_hsa_wait(H, 0);
                plow_hsa_copy_d2h(H, 0, hOp1, dOp1, nop1 * 4);
                plow_hsa_copy_d2h(H, 0, hMl1, dMl1, nml1 * 4);
                /* Opart is [b][t][head][nsplit=1][DK]; row t is a contiguous n_head*DK block. */
                if (memcmp(hOp + (size_t)t * nop1, hOp1, nop1 * 4)) bad++;
                if (memcmp(hMl + (size_t)t * nml1, hMl1, nml1 * 4)) bad++;
                checked++;
            }
            char label[96];
            snprintf(label, sizeof(label), "%s n_head=%u ctx=%u n_tok=%u%s",
                     top_k ? "gather " : "dense  ", n_head, ctx, T, top_k ? " (gathered)" : "");
            printf("  %-42s %s  (%u/%u rows bit-exact vs decode)\n", label, bad ? "FAIL" : "PASS",
                   checked * 2 - bad, checked * 2);
            if (bad) fails++;

            plow_hsa_free(H, dQa); plow_hsa_free(H, dQr); plow_hsa_free(H, dLen);
            plow_hsa_free(H, dOp); plow_hsa_free(H, dMl);
            plow_hsa_free(H, dOp1); plow_hsa_free(H, dMl1); plow_hsa_free(H, dLen1);
        }
        plow_hsa_free(H, dCkv); plow_hsa_free(H, dKr);
    }

    printf("\n%s (%d failure%s)\n", fails ? "MLA FAILED" : "MLA CORRECT", fails,
           fails == 1 ? "" : "s");
    plow_hsa_shutdown(H);
    return fails ? 1 : 0;
}
