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
static uint32_t rnd_u32(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

/* Little-endian readers over the fixture byte stream. */
static const uint8_t* P;
static uint32_t rd_u32(void) { uint32_t v; memcpy(&v, P, 4); P += 4; return v; }
static float rd_f32(void) { float v; memcpy(&v, P, 4); P += 4; return v; }

int main(int argc, char** argv) {
    const char* fx = argc > 1 ? argv[1] : "fixture.bin";
    const char* object = argc > 2 ? argv[2] : "test_kernels.elf";
    FILE* ff = fopen(fx, "rb");
    if (!ff) { perror(fx); return 1; }
    fseek(ff, 0, SEEK_END); long fn = ftell(ff); fseek(ff, 0, SEEK_SET);
    uint8_t* fixture = malloc(fn);
    if (fread(fixture, 1, fn, ff) != (size_t)fn) return 1;
    fclose(ff);
    P = fixture;
    /* "MLA2": the case header grew an 8th word (`nope`). Bumped with the layout so a
     * stale fixture is refused here instead of mis-slicing every array after it. */
    if (rd_u32() != 0x4d4c4132u) { fprintf(stderr, "bad fixture magic (want MLA2)\n"); return 1; }
    const uint32_t n_cases = rd_u32();

    H = plow_hsa_init();
    if (!H) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    char nm[64]; uint32_t cus = 0, lds = 0;
    plow_hsa_device_info(H, 0, nm, &cus, &lds);
    printf("dev0: %s  CUs=%u  LDS=%u B\n\n", nm, cus, lds);

    FILE* f = fopen(object, "rb");
    if (!f) { perror(object); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(H, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }

    /* THE OBJECT AND THIS BINARY MUST AGREE ON THE WORKGROUP WIDTH. See the comment on
     * `plow_probe_wg_threads` in test_kernels.hip: an 8-wave object launched with 256
     * threads is a LEGAL dispatch that leaves half of every per-wave LDS array unwritten,
     * and d_flash_mla_decode's output fold sums them anyway. It reads as a kernel defect
     * -- 1e24..1e35 outputs and `rms nan` -- and it was diagnosed as one for a while.
     * Nothing else in this harness can detect it, so check before running anything. */
    {
        plow_hsa_kernel kp;
        if (plow_hsa_get_kernel(H, 0, "plow_probe_wg_threads", &kp)) {
            fprintf(stderr, "sym plow_probe_wg_threads: %s\n"
                            "  (rebuild test_kernels.elf from this tree)\n",
                    plow_hsa_last_error());
            return 1;
        }
        void* dW = dev(4);
        struct __attribute__((packed)) { void* out; } pa = {dW};
        if (plow_hsa_launch(H, 0, &kp, 1, 1, 1, 1, 1, 1, 0, &pa, sizeof(pa)) != 0) {
            fprintf(stderr, "probe launch: %s\n", plow_hsa_last_error()); return 1;
        }
        plow_hsa_wait(H, 0);
        unsigned* hW = plow_hsa_alloc_host(H, 4);
        plow_hsa_copy_d2h(H, 0, hW, dW, 4);
        if (*hW != PLOW_WG_THREADS) {
            fprintf(stderr,
                    "FATAL: test_kernels.elf was built for %u threads/workgroup, this harness "
                    "launches %u.\n"
                    "  Rebuild BOTH with the same -DPLOW_WG_WAVES. A mismatch in this direction "
                    "is a legal\n"
                    "  dispatch, not a launch error, and it corrupts MLA output in a way that "
                    "looks exactly\n"
                    "  like a kernel bug (nsplit>1 cases at 1e24..1e35, rms nan).\n",
                    *hW, (unsigned)PLOW_WG_THREADS);
            return 1;
        }
        printf("workgroup width: %u threads (object and harness agree)\n", *hW);
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
        const uint32_t nope = rd_u32();
        const float scale = rd_f32();
        (void)nope; /* the decode phase runs the roped arm on zeroed tables either way */
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
    plow_hsa_kernel kpf, kgpf, kpfm, kpfv, kpfvn;
    if (plow_hsa_get_kernel(H, 0, "mla_flash_prefill_512", &kpf) ||
        plow_hsa_get_kernel(H, 0, "mla_gather_prefill_512", &kgpf) ||
        plow_hsa_get_kernel(H, 0, "mla_flash_prefill_v2_512", &kpfv) ||
        plow_hsa_get_kernel(H, 0, "mla_flash_prefill_v2_nope_512", &kpfvn) ||
        plow_hsa_get_kernel(H, 0, "mla_flash_prefill_mfma_512", &kpfm)) {
        fprintf(stderr, "prefill sym: %s\n", plow_hsa_last_error());
        return 1;
    }
    /* The TILED kernel (PLOW_MLA_PF_MFMA, the shipped path) is checked against the SAME
     * decode oracle, and it is the reason this phase grew a tolerance. It cannot be
     * memcmp'd: MFMA contracts DK in a different order, so the scores differ in the low
     * bits, and through the online softmax's running max that moves every unnormalized
     * partial. What IS comparable is the quantity the merge actually produces — Opart/l —
     * because it is invariant to the max the two kernels chose. Reported relative to the
     * row's own magnitude so a near-zero component cannot manufacture a large ratio. */
#define MLA_PF_MFMA_TOL 2e-2f
    /* T=1 first (the base case: prefill MUST degenerate to the validated decode),
     * then T=4 (the causal bound actually doing work). */
    /* 1 and 4 are the original base cases (prefill degenerating to decode, then a real
     * causal bound). The rest exist for the TILED kernel and are chosen against its q-tile
     * of FA_MLA_PF_BQ=64: 33 crosses an M-tile inside one q-tile, 64 is the exact tile,
     * 65 is the first ragged second tile, and 130 is several tiles with a ragged tail —
     * the sizes where a q-tile bound or a mask edge is wrong but a single tile hides it. */
    static const unsigned TOKS[] = {1, 4, 33, 64, 65, 130};
    printf("\nMLA PREFILL (device prefill vs the validated device decode, bit-exact):\n");
    P = fixture + 8; /* past magic + n_cases */
    for (uint32_t ci = 0; ci < n_cases; ci++) {
        const uint32_t n_head = rd_u32(), DK = rd_u32(), DR = rd_u32(), Vd = rd_u32();
        const uint32_t ctx = rd_u32(), nsplit_fx = rd_u32(), top_k = rd_u32();
        const uint32_t nope = rd_u32();
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
            /* The tiled kernel, same operands, its own buffers. Dense only — the gathered
             * arm keeps the scalar body because its top_k set is per query row. */
            void *dOpM = NULL, *dMlM = NULL;
            float *hOpM = NULL, *hMlM = NULL;
            /* The V2 (full-column-wave) body on the SAME operands — the shipped MLA prefill
             * for GLM/DeepSeek, and until now the only prefill kernel here with no bf16
             * coverage at all. Its (m, l) are NOT comparable elementwise: under
             * FA_MLA_PF2_DEFER `m` is an exponent frame rather than the running max, so the
             * check below is on the NORMALIZED output, the only thing that is frame-free.
             * Four waves, hence the explicit 256 against PLOW_WG_THREADS. */
            void *dOpV = NULL, *dMlV = NULL;
            float *hOpV = NULL, *hMlV = NULL;
            /* NoPE (zero-rope) V2. Only on a `nope` case, where the fixture's rope tables are
             * identically zero -- so the SAME f64 golden that validates every other case is the
             * golden for this one, and the roped V2 above becomes a bit-exact control. */
            void *dOpVN = NULL, *dMlVN = NULL;
            float *hOpVN = NULL, *hMlVN = NULL;
            if (top_k == 0) {
                dOpM = dev(nop_t * 4);
                dMlM = dev(nml_t * 4);
                dOpV = dev(nop_t * 4);
                dMlV = dev(nml_t * 4);
                struct __attribute__((packed)) {
                    void *op, *ml; const void *qa, *qr, *ckv, *kr, *len;
                    unsigned n_batch, n_tok, n_head, kv_stride, window; float scale;
                } a = {dOpM, dMlM, dQa, dQr, dCkv, dKr, dLen, B, T, n_head, ctx, 0, scale},
                  v = {dOpV, dMlV, dQa, dQr, dCkv, dKr, dLen, B, T, n_head, ctx, 0, scale};
                if (plow_hsa_launch(H, 0, &kpfm, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1,
                                    1, 0, &a, sizeof(a)) != 0) {
                    fprintf(stderr, "mfma prefill launch: %s\n", plow_hsa_last_error()); fails++;
                }
                if (plow_hsa_launch(H, 0, &kpfv, cus * 256u, 1, 1, 256u, 1, 1, 0, &v,
                                    sizeof(v)) != 0) {
                    fprintf(stderr, "v2 prefill launch: %s\n", plow_hsa_last_error()); fails++;
                }
                if (nope) {
                    dOpVN = dev(nop_t * 4);
                    dMlVN = dev(nml_t * 4);
                    /* NULL rope pointers, deliberately: the DR=0 body must not touch them. A
                     * staging path that still read the rope half would fault here rather than
                     * quietly load zeros from a live buffer and pass the numeric check. */
                    struct __attribute__((packed)) {
                        void *op, *ml; const void *qa, *ckv, *len;
                        unsigned n_batch, n_tok, n_head, kv_stride, window; float scale;
                    } vn = {dOpVN, dMlVN, dQa, dCkv, dLen, B, T, n_head, ctx, 0, scale};
                    if (plow_hsa_launch(H, 0, &kpfvn, cus * 256u, 1, 1, 256u, 1, 1, 0, &vn,
                                        sizeof(vn)) != 0) {
                        fprintf(stderr, "v2 nope prefill launch: %s\n", plow_hsa_last_error());
                        fails++;
                    }
                }
            }
            plow_hsa_wait(H, 0);
            float* hOp = plow_hsa_alloc_host(H, nop_t * 4);
            float* hMl = plow_hsa_alloc_host(H, nml_t * 4);
            plow_hsa_copy_d2h(H, 0, hOp, dOp, nop_t * 4);
            plow_hsa_copy_d2h(H, 0, hMl, dMl, nml_t * 4);
            if (dOpM) {
                hOpM = plow_hsa_alloc_host(H, nop_t * 4);
                hMlM = plow_hsa_alloc_host(H, nml_t * 4);
                plow_hsa_copy_d2h(H, 0, hOpM, dOpM, nop_t * 4);
                plow_hsa_copy_d2h(H, 0, hMlM, dMlM, nml_t * 4);
                hOpV = plow_hsa_alloc_host(H, nop_t * 4);
                hMlV = plow_hsa_alloc_host(H, nml_t * 4);
                plow_hsa_copy_d2h(H, 0, hOpV, dOpV, nop_t * 4);
                plow_hsa_copy_d2h(H, 0, hMlV, dMlV, nml_t * 4);
                if (dOpVN) {
                    hOpVN = plow_hsa_alloc_host(H, nop_t * 4);
                    hMlVN = plow_hsa_alloc_host(H, nml_t * 4);
                    plow_hsa_copy_d2h(H, 0, hOpVN, dOpVN, nop_t * 4);
                    plow_hsa_copy_d2h(H, 0, hMlVN, dMlVN, nml_t * 4);
                }
            }
            float mfma_err = 0.0f, v2_err = 0.0f, nope_err = 0.0f;
            unsigned long nope_ne = 0; /* bits where the zero-rope body != the roped one */

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
                /* Tiled kernel vs the same oracle row, per head, on the NORMALIZED output. */
                if (hOpM) {
                    for (unsigned hh = 0; hh < n_head; hh++) {
                        const float lr = hMl1[hh * 2 + 1];
                        const float lm = hMlM[((size_t)t * n_head + hh) * 2 + 1];
                        const float* orow = hOp1 + (size_t)hh * DK;
                        const float* mrow = hOpM + ((size_t)t * n_head + hh) * DK;
                        float mag = 0.0f, dif = 0.0f;
                        for (unsigned d = 0; d < DK; d++) {
                            const float rv = (lr > 0.0f) ? orow[d] / lr : 0.0f;
                            const float mv = (lm > 0.0f) ? mrow[d] / lm : 0.0f;
                            const float av = rv < 0 ? -rv : rv;
                            const float dv = (rv - mv) < 0 ? (mv - rv) : (rv - mv);
                            if (av > mag) mag = av;
                            if (dv > dif) dif = dv;
                        }
                        const float e = (mag > 0.0f) ? dif / mag : dif;
                        if (e > mfma_err) mfma_err = e;
                    }
                    for (unsigned hh = 0; hh < n_head; hh++) {
                        const float lr = hMl1[hh * 2 + 1];
                        const float lv = hMlV[((size_t)t * n_head + hh) * 2 + 1];
                        const float* orow = hOp1 + (size_t)hh * DK;
                        const float* vrow = hOpV + ((size_t)t * n_head + hh) * DK;
                        float mag = 0.0f, dif = 0.0f;
                        for (unsigned d = 0; d < DK; d++) {
                            const float rv = (lr > 0.0f) ? orow[d] / lr : 0.0f;
                            const float vv = (lv > 0.0f) ? vrow[d] / lv : 0.0f;
                            const float av = rv < 0 ? -rv : rv;
                            const float dv = (rv - vv) < 0 ? (vv - rv) : (rv - vv);
                            if (av > mag) mag = av;
                            if (dv > dif) dif = dv;
                        }
                        const float e = (mag > 0.0f) ? dif / mag : dif;
                        if (e > v2_err) v2_err = e;
                        /* ---- the ZERO-ROPE body, on the same row ----
                         * Two independent claims, both required:
                         *   (a) vs the DECODE ORACLE, normalized -- the same chain that ties
                         *       every other case back to the f64 CPU golden. This is the claim
                         *       that the DR=0 body computes MLA, not merely that it agrees with
                         *       something.
                         *   (b) vs the ROPED body, BIT-EXACT. On a nope case the rope tables are
                         *       identically zero, so the two extra k-tiles the <512,64> body
                         *       contracts contribute 0*k = 0 into an f32 MFMA accumulator, which
                         *       is exact. Any bit of difference means an `if constexpr (DR > 0)`
                         *       guard removed work that was not dead, or the DR=0 slab addressed
                         *       a row differently. memcmp, not a tolerance. */
                        if (hOpVN) {
                            const float lvn = hMlVN[((size_t)t * n_head + hh) * 2 + 1];
                            const float* nrow = hOpVN + ((size_t)t * n_head + hh) * DK;
                            float nmag = 0.0f, ndif = 0.0f;
                            for (unsigned d = 0; d < DK; d++) {
                                const float rv = (lr > 0.0f) ? orow[d] / lr : 0.0f;
                                const float nv = (lvn > 0.0f) ? nrow[d] / lvn : 0.0f;
                                const float av = rv < 0 ? -rv : rv;
                                const float dv = (rv - nv) < 0 ? (nv - rv) : (rv - nv);
                                if (av > nmag) nmag = av;
                                if (dv > ndif) ndif = dv;
                                if (nrow[d] != vrow[d]) nope_ne++;
                            }
                            if (lvn != lv) nope_ne++;
                            const float ne = (nmag > 0.0f) ? ndif / nmag : ndif;
                            if (ne > nope_err) nope_err = ne;
                        }
                    }
                }
            }
            char label[96];
            snprintf(label, sizeof(label), "%s n_head=%u ctx=%u n_tok=%u%s",
                     top_k ? "gather " : "dense  ", n_head, ctx, T, top_k ? " (gathered)" : "");
            printf("  %-42s %s  (%u/%u rows bit-exact vs decode)\n", label, bad ? "FAIL" : "PASS",
                   checked * 2 - bad, checked * 2);
            if (bad) fails++;
            if (hOpM) {
                const int mbad = !(mfma_err <= MLA_PF_MFMA_TOL);
                printf("  %-42s %s  (max rel err %.2e vs decode, tol %.0e)\n",
                       "  \\_ tiled MFMA", mbad ? "FAIL" : "PASS", mfma_err,
                       (double)MLA_PF_MFMA_TOL);
                if (mbad) fails++;
                const int vbad = !(v2_err <= MLA_PF_MFMA_TOL);
                printf("  %-42s %s  (max rel err %.2e vs decode, tol %.0e)\n",
                       "  \\_ V2", vbad ? "FAIL" : "PASS", v2_err, (double)MLA_PF_MFMA_TOL);
                if (vbad) fails++;
                if (hOpVN) {
                    const int nbad = !(nope_err <= MLA_PF_MFMA_TOL) || nope_ne != 0;
                    printf("  %-42s %s  (max rel err %.2e vs decode, tol %.0e; "
                           "%lu float(s) differ from the roped body, want 0)\n",
                           "  \\_ V2 NoPE <512,0>", nbad ? "FAIL" : "PASS", nope_err,
                           (double)MLA_PF_MFMA_TOL, nope_ne);
                    if (nbad) fails++;
                }
            }

            plow_hsa_free(H, dQa); plow_hsa_free(H, dQr); plow_hsa_free(H, dLen);
            plow_hsa_free(H, dOp); plow_hsa_free(H, dMl);
            if (dOpM) { plow_hsa_free(H, dOpM); plow_hsa_free(H, dMlM); }
            if (dOpV) { plow_hsa_free(H, dOpV); plow_hsa_free(H, dMlV); }
            if (dOpVN) { plow_hsa_free(H, dOpVN); plow_hsa_free(H, dMlVN); }
            plow_hsa_free(H, dOp1); plow_hsa_free(H, dMl1); plow_hsa_free(H, dLen1);
        }
        plow_hsa_free(H, dCkv); plow_hsa_free(H, dKr);
    }

    /* ====================================================================
     * PHASE 3 — FP8-LATENT MLA PREFILL (op 110), which is the arm Kimi-K3 actually
     * dispatches and the one phase 2 does not reach: the fixture is bf16 throughout.
     *
     * There is no CPU oracle for this and the bf16 trick does not transfer — the decode
     * kernel is only an oracle for a body it shares, and the two fp8 bodies deliberately
     * do NOT share one. So the claim here is the narrower, and correct, one: the tiled
     * kernel agrees with the SHIPPED scalar fp8 kernel on the same bytes. Phase 2 is what
     * ties the tiling itself to a real reference; this ties the fp8 handling to the
     * implementation it replaces.
     *
     * Inputs are synthesized, not quantized from the fixture: both kernels read the same
     * e4m3 bytes and the same scales, so what a "good" quantizer would have produced is
     * irrelevant to whether they agree. Exponents are held to [4,10] to stay clear of
     * e4m3's NaN encoding (exp=15, mantissa=7) and of subnormals.
     *
     * krot_fp8 = 0: K3 keeps the 64-wide rope cache in bf16 (op 110 carries i6=0), so
     * that is the shipped form and the one gated here.
     * ==================================================================== */
    plow_hsa_kernel kpf8, kpfm8, kpfv8;
    if (plow_hsa_get_kernel(H, 0, "mla_flash_prefill_fp8_512", &kpf8) ||
        plow_hsa_get_kernel(H, 0, "mla_flash_prefill_mfma_fp8_512", &kpfm8)) {
        fprintf(stderr, "fp8 prefill sym: %s\n", plow_hsa_last_error());
        return 1;
    }
    if (plow_hsa_get_kernel(H, 0, "mla_flash_prefill_v2_fp8_512", &kpfv8)) {
        fprintf(stderr, "fp8 v2 sym: %s\n", plow_hsa_last_error());
        return 1;
    }
    printf("\nMLA PREFILL, FP8 LATENT (tiled vs the shipped scalar fp8 body):\n");
    {
        static const unsigned NH[] = {12, 12, 12, 64};
        static const unsigned CX[] = {2048, 2048, 512, 1024};
        static const unsigned TK[] = {130, 64, 512, 65};
        uint32_t rs = 99991u;
#define RND rnd_u32(&rs)
        for (unsigned c = 0; c < sizeof(NH) / sizeof(NH[0]); c++) {
            const unsigned nh = NH[c], ctx = CX[c], T = TK[c], DK = 512, DR = 64;
            const size_t nckv = (size_t)ctx * DK, nkr = (size_t)ctx * DR;
            const size_t nqa = (size_t)T * nh * DK, nqr = (size_t)T * nh * DR;
            const size_t nop = (size_t)T * nh * DK, nml = (size_t)T * nh * 2;

            unsigned char* pC = plow_hsa_alloc_host(H, nckv);
            for (size_t i = 0; i < nckv; i++) {
                const uint32_t r = RND;
                pC[i] = (unsigned char)(((r >> 31) << 7) | ((4u + ((r >> 8) % 7u)) << 3) |
                                        ((r >> 16) & 7u));
            }
            bf16* pR = plow_hsa_alloc_host(H, nkr * 2);
            bf16* pQa = plow_hsa_alloc_host(H, nqa * 2);
            bf16* pQr = plow_hsa_alloc_host(H, nqr * 2);
            for (size_t i = 0; i < nkr; i++)
                pR[i] = (bf16)((((RND) >> 31) << 15) | 0x3d00u | ((RND >> 16) & 0x7fu));
            for (size_t i = 0; i < nqa; i++)
                pQa[i] = (bf16)((((RND) >> 31) << 15) | 0x3c00u | ((RND >> 16) & 0x7fu));
            for (size_t i = 0; i < nqr; i++)
                pQr[i] = (bf16)((((RND) >> 31) << 15) | 0x3c00u | ((RND >> 16) & 0x7fu));
            /* Two strips of kv_stride each: ckv scales, then krot scales (unread at
             * krot_fp8=0, still allocated because the kernel forms the pointer). */
            float* pS = plow_hsa_alloc_host(H, (size_t)2 * ctx * 4);
            for (size_t i = 0; i < (size_t)2 * ctx; i++)
                pS[i] = 0.25f + (float)((RND >> 12) & 0xffu) / 256.0f;
            int* pL = plow_hsa_alloc_host(H, 4); pL[0] = (int)ctx;

            void* dC = dev(nckv); plow_hsa_copy_h2d(H, 0, dC, pC, nckv);
            void* dR = dev(nkr * 2); plow_hsa_copy_h2d(H, 0, dR, pR, nkr * 2);
            void* dQa = dev(nqa * 2); plow_hsa_copy_h2d(H, 0, dQa, pQa, nqa * 2);
            void* dQr = dev(nqr * 2); plow_hsa_copy_h2d(H, 0, dQr, pQr, nqr * 2);
            void* dS = dev((size_t)2 * ctx * 4);
            plow_hsa_copy_h2d(H, 0, dS, pS, (size_t)2 * ctx * 4);
            void* dL = dev(4); plow_hsa_copy_h2d(H, 0, dL, pL, 4);
            void* dO1 = dev(nop * 4); void* dM1 = dev(nml * 4);
            void* dO2 = dev(nop * 4); void* dM2 = dev(nml * 4);
            void* dO3 = dev(nop * 4); void* dM3 = dev(nml * 4);

            struct __attribute__((packed)) {
                void *op, *ml; const void *qa, *qr, *ckv, *kr, *len;
                unsigned n_batch, n_tok, n_head, kv_stride, window; float scale;
                const void* kvs; unsigned krot;
            } a1 = {dO1, dM1, dQa, dQr, dC, dR, dL, 1, T, nh, ctx, 0, 0.0883883f, dS, 0},
              a2 = {dO2, dM2, dQa, dQr, dC, dR, dL, 1, T, nh, ctx, 0, 0.0883883f, dS, 0};
            if (plow_hsa_launch(H, 0, &kpf8, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                0, &a1, sizeof(a1)) != 0 ||
                plow_hsa_launch(H, 0, &kpfm8, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1,
                                0, &a2, sizeof(a2)) != 0) {
                fprintf(stderr, "fp8 launch: %s\n", plow_hsa_last_error()); fails++;
            }
            {
                typeof(a1) a3 = {dO3, dM3, dQa, dQr, dC, dR, dL, 1, T, nh, ctx, 0,
                                 0.0883883f, dS, 0};
                if (plow_hsa_launch(H, 0, &kpfv8, cus * 256u, 1, 1, 256u, 1, 1, 0, &a3,
                                    sizeof(a3)) != 0) {
                    fprintf(stderr, "fp8 v2 launch: %s\n", plow_hsa_last_error()); fails++;
                }
            }
            plow_hsa_wait(H, 0);
            float* hO1 = plow_hsa_alloc_host(H, nop * 4);
            float* hM1 = plow_hsa_alloc_host(H, nml * 4);
            float* hO2 = plow_hsa_alloc_host(H, nop * 4);
            float* hM2 = plow_hsa_alloc_host(H, nml * 4);
            float* hO3 = plow_hsa_alloc_host(H, nop * 4);
            float* hM3 = plow_hsa_alloc_host(H, nml * 4);
            plow_hsa_copy_d2h(H, 0, hO1, dO1, nop * 4);
            plow_hsa_copy_d2h(H, 0, hM1, dM1, nml * 4);
            plow_hsa_copy_d2h(H, 0, hO2, dO2, nop * 4);
            plow_hsa_copy_d2h(H, 0, hM2, dM2, nml * 4);
            plow_hsa_copy_d2h(H, 0, hO3, dO3, nop * 4);
            plow_hsa_copy_d2h(H, 0, hM3, dM3, nml * 4);

            float err = 0.0f;
            for (size_t row = 0; row < (size_t)T * nh; row++) {
                const float l1 = hM1[row * 2 + 1], l2 = hM2[row * 2 + 1];
                const float* o1 = hO1 + row * DK;
                const float* o2 = hO2 + row * DK;
                float mag = 0.0f, dif = 0.0f;
                for (unsigned d = 0; d < DK; d++) {
                    const float v1 = (l1 > 0.0f) ? o1[d] / l1 : 0.0f;
                    const float v2 = (l2 > 0.0f) ? o2[d] / l2 : 0.0f;
                    const float av = v1 < 0 ? -v1 : v1;
                    const float dv = (v1 - v2) < 0 ? (v2 - v1) : (v1 - v2);
                    if (av > mag) mag = av;
                    if (dv > dif) dif = dv;
                }
                const float e = (mag > 0.0f) ? dif / mag : dif;
                if (e > err) err = e;
            }
            char label[96];
            snprintf(label, sizeof(label), "fp8    n_head=%u ctx=%u n_tok=%u", nh, ctx, T);
            const int bad = !(err <= MLA_PF_MFMA_TOL);
            printf("  %-42s %s  (max rel err %.2e, tol %.0e)\n", label, bad ? "FAIL" : "PASS",
                   err, (double)MLA_PF_MFMA_TOL);
            if (bad) fails++;
            {
                float v2_err = 0.0f;
                for (size_t row = 0; row < (size_t)T * nh; row++) {
                    const float l1 = hM1[row * 2 + 1], l3 = hM3[row * 2 + 1];
                    float mag = 0.0f, dif = 0.0f;
                    for (unsigned d = 0; d < DK; d++) {
                        const float v1 = l1 > 0.0f ? hO1[row * DK + d] / l1 : 0.0f;
                        const float v3 = l3 > 0.0f ? hO3[row * DK + d] / l3 : 0.0f;
                        const float av = v1 < 0 ? -v1 : v1;
                        const float dv = (v1 - v3) < 0 ? (v3 - v1) : (v1 - v3);
                        if (av > mag) mag = av;
                        if (dv > dif) dif = dv;
                    }
                    const float e = mag > 0.0f ? dif / mag : dif;
                    if (e > v2_err) v2_err = e;
                }
                const int v2_bad = !(v2_err <= MLA_PF_MFMA_TOL);
                printf("  %-42s %s  (max rel err %.2e, tol %.0e)\n", "  \\_ fp8 V2",
                       v2_bad ? "FAIL" : "PASS", v2_err, (double)MLA_PF_MFMA_TOL);
                if (v2_bad) fails++;
            }

            plow_hsa_free(H, dC); plow_hsa_free(H, dR); plow_hsa_free(H, dQa);
            plow_hsa_free(H, dQr); plow_hsa_free(H, dS); plow_hsa_free(H, dL);
            plow_hsa_free(H, dO1); plow_hsa_free(H, dM1);
            plow_hsa_free(H, dO2); plow_hsa_free(H, dM2);
            plow_hsa_free(H, dO3); plow_hsa_free(H, dM3);
        }
#undef RND
    }

    /* ====================================================================
     * MLA MERGE+FOLD (d_mla_merge_fold) vs a DOUBLE-PRECISION CPU golden.
     *
     * This body had a BENCH (runtime/tests/decode_bench_gfx942.c) and NO ORACLE, which is
     * how a 7.7x scalar fallback survived at V<128 without anyone noticing: nothing here
     * ever checked what it computed, only how fast it computed it.
     *
     * It is also the largest single line in a GLM decode token, and it fuses two steps that
     * used to be separate ops -- the nsplit online-softmax merge and the W_uv fold -- so a
     * fault in either half is invisible to any test of the flash kernels alone.
     *
     * THE SHAPES ARE THE DISPATCH TABLE, not a sample. `exec_mla_merge_fold` picks VT from V
     * and the workgroup budget, and each row below is one arm of that pick:
     *   V=32,64,96  -> <512,32>, the arm the V<128 branch now routes to (this is the new
     *                  coverage; before, these V fell into <512,256>'s scalar `else`)
     *   V=128       -> <512,128>, Kimi-K3's v_head_dim
     *   V=256       -> <512,256>, GLM-5.2's
     * and V=128 is ALSO run through <512,32> to pin the property the V<128 branch relies on:
     * a V that is a whole multiple of VT gets full tiles and the same answer.
     *
     * nsplit is swept 1 and 8 because the merge half is only exercised above 1, and the
     * kernel has a separate MS=8 blocked path that only a multiple of 8 reaches.
     *
     * The golden accumulates in DOUBLE while the kernel accumulates in f32 over 512 terms,
     * so this is a tolerance and not a memcmp -- but a LOOSE tolerance would pass the scalar
     * fallback too (it is slow, not wrong), so the point of this phase is the SHAPE coverage:
     * a routing bug shows up as a wrong tile count or an unwritten output column, which is a
     * gross error, not a rounding one. Both are caught here. ==================== */
    printf("\nMLA MERGE+FOLD (device vs f64 CPU golden, over the VT dispatch table):\n");
    {
        plow_hsa_kernel kf32, kf128, kf256;
        if (plow_hsa_get_kernel(H, 0, "mla_merge_fold_512_v32", &kf32) ||
            plow_hsa_get_kernel(H, 0, "mla_merge_fold_512_v128", &kf128) ||
            plow_hsa_get_kernel(H, 0, "mla_merge_fold_512_v256", &kf256)) {
            fprintf(stderr, "merge_fold sym: %s\n", plow_hsa_last_error());
            return 1;
        }
        static const unsigned FV[]  = { 32,  64,  96, 128, 128, 256 };
        static const char* FVT[]    = {"32","32","32","32","128","256"};
        static const unsigned FNS[] = {  1,   8,   4,   8,   8,    8  };
        const unsigned DKf = 512, nb = 2, nh = 8;
        uint32_t rs = 1234567u;
        for (unsigned c = 0; c < sizeof(FV) / sizeof(FV[0]); c++) {
            const unsigned V = FV[c], ns = FNS[c];
            const size_t nrow = (size_t)nb * nh;
            const size_t nop = nrow * ns * DKf, nml = nrow * ns * 2;
            const size_t nw = (size_t)nh * DKf * V, no = nrow * V;

            float* hOp = plow_hsa_alloc_host(H, nop * 4);
            float* hMl = plow_hsa_alloc_host(H, nml * 4);
            bf16* hW = plow_hsa_alloc_host(H, nw * 2);
            /* Partials as a real flash epilogue leaves them: UNNORMALIZED o, with (m, l) per
             * split. Some splits are dead (m = -inf, l = 0) because the causal bound makes
             * them so, and the merge must weigh those exactly 0 rather than divide by them. */
            for (size_t i = 0; i < nop; i++)
                hOp[i] = ((float)(int)((rnd_u32(&rs) >> 8) & 0xffffu) - 32768.0f) / 8192.0f;
            for (size_t r = 0; r < nrow; r++) {
                for (unsigned sp = 0; sp < ns; sp++) {
                    const int dead = (ns > 1) && (sp == ns - 1);
                    hMl[(r * ns + sp) * 2 + 0] =
                        dead ? -INFINITY
                             : ((float)(int)((rnd_u32(&rs) >> 12) & 0x7ffu) - 1024.0f) / 256.0f;
                    hMl[(r * ns + sp) * 2 + 1] =
                        dead ? 0.0f : 1.0f + (float)((rnd_u32(&rs) >> 16) & 0xffu) / 64.0f;
                }
            }
            for (size_t i = 0; i < nw; i++) {
                const float w = ((float)(int)((rnd_u32(&rs) >> 8) & 0xffffu) - 32768.0f) / 65536.0f;
                uint32_t u; memcpy(&u, &w, 4);
                hW[i] = (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
            }

            void* dOp = dev(nop * 4); plow_hsa_copy_h2d(H, 0, dOp, hOp, nop * 4);
            void* dMl = dev(nml * 4); plow_hsa_copy_h2d(H, 0, dMl, hMl, nml * 4);
            void* dW = dev(nw * 2);   plow_hsa_copy_h2d(H, 0, dW, hW, nw * 2);
            void* dO = dev(no * 2);

            struct __attribute__((packed)) {
                void* o; const void *op, *ml, *w; unsigned n_batch, n_head, V, nsplit;
            } a = {dO, dOp, dMl, dW, nb, nh, V, ns};
            plow_hsa_kernel* k = (V == 256) ? &kf256
                               : (c == 4)   ? &kf128   /* the V=128 row that uses <512,128> */
                                            : &kf32;
            if (plow_hsa_launch(H, 0, k, cus * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0,
                                &a, sizeof(a)) != 0) {
                fprintf(stderr, "merge_fold launch: %s\n", plow_hsa_last_error()); fails++;
            }
            plow_hsa_wait(H, 0);
            bf16* hO = plow_hsa_alloc_host(H, no * 2);
            plow_hsa_copy_d2h(H, 0, hO, dO, no * 2);

            /* f64 golden: the same two steps in the same order, at double width. */
            double err = 0.0, mag = 0.0;
            for (size_t r = 0; r < nrow; r++) {
                const unsigned h = (unsigned)(r % nh);
                double gm = -INFINITY;
                for (unsigned sp = 0; sp < ns; sp++) {
                    const double m = hMl[(r * ns + sp) * 2];
                    if (hMl[(r * ns + sp) * 2 + 1] > 0.0f && m > gm) gm = m;
                }
                double gl = 0.0;
                for (unsigned sp = 0; sp < ns; sp++) {
                    const double m = hMl[(r * ns + sp) * 2], l = hMl[(r * ns + sp) * 2 + 1];
                    /* exp2, NOT exp: FA_EXP is __builtin_amdgcn_exp2f, so every flash
                     * epilogue in this tree stores `m` already in log2 space. The two agree
                     * exactly at nsplit==1 (the weight is exp(0)=exp2(0)=1), which is why a
                     * golden that gets this wrong passes the unsplit case and fails every
                     * split one -- the shape of the first run of this phase. */
                    if (l > 0.0) gl += l * exp2(m - gm);
                }
                const double inv = gl > 0.0 ? 1.0 / gl : 0.0;
                double* olat = (double*)malloc(DKf * sizeof(double));
                for (unsigned d = 0; d < DKf; d++) {
                    double acc = 0.0;
                    for (unsigned sp = 0; sp < ns; sp++) {
                        const double m = hMl[(r * ns + sp) * 2];
                        const double w = (m == -INFINITY) ? 0.0 : exp2(m - gm);
                        acc += (double)hOp[(r * ns + sp) * DKf + d] * w;
                    }
                    olat[d] = acc * inv;
                }
                for (unsigned v = 0; v < V; v++) {
                    double acc = 0.0;
                    for (unsigned d = 0; d < DKf; d++) {
                        const uint32_t wb = (uint32_t)hW[(size_t)(h * DKf + d) * V + v] << 16;
                        float wf; memcpy(&wf, &wb, 4);
                        acc += olat[d] * (double)wf;
                    }
                    const uint32_t ob = (uint32_t)hO[r * V + v] << 16;
                    float of; memcpy(&of, &ob, 4);
                    const double d0 = fabs(acc - (double)of);
                    if (fabs(acc) > mag) mag = fabs(acc);
                    if (d0 > err) err = d0;
                }
                free(olat);
            }
            const double rel = mag > 0.0 ? err / mag : err;
            /* bf16 output, f32 accumulation over 512 terms against an f64 golden. */
            const int bad = !(rel <= 2e-2);
            printf("  V=%-4u nsplit=%-2u VT=%-4s               %s  (max rel err %.2e, tol 2e-02)\n",
                   V, ns, FVT[c], bad ? "FAIL" : "PASS", rel);
            if (bad) fails++;

            plow_hsa_free(H, dOp); plow_hsa_free(H, dMl);
            plow_hsa_free(H, dW); plow_hsa_free(H, dO);
        }
    }

    printf("\n%s (%d failure%s)\n", fails ? "MLA FAILED" : "MLA CORRECT", fails,
           fails == 1 ? "" : "s");
    plow_hsa_shutdown(H);
    return fails ? 1 : 0;
}
