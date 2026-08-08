/* gemma_gfx950_test.c — validates the Gemma 4 BF16 gfx950 kernels against an
 * independent f32 CPU reference, driving them through the ROCr/HSA backend.
 *
 * The reference here is written from HF `modeling_gemma4.py` semantics directly
 *, NOT from the kernels — otherwise a
 * shared misreading of the model would pass silently.
 *
 * Needs a gfx950 device and `test_kernels.elf` next to the binary.
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
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}

static int fails = 0;
/* bf16 carries ~3 decimal digits; 2e-2 relative is the honest tolerance. */
static void check(const char* what, const bf16* got, const float* want, size_t n) {
    double worst = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        const double g = bf2f(got[i]), w = want[i];
        const double d = fabs(g - w) / (fabs(w) + 1e-3);
        if (d > worst) { worst = d; at = i; }
    }
    const int ok = worst < 2e-2;
    printf("  %-28s %s  (worst rel %.4f at %zu)\n", what, ok ? "PASS" : "FAIL", worst, at);
    if (!ok) {
        printf("      got %.5f want %.5f\n", bf2f(got[at]), want[at]);
        fails++;
    }
}

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }
static float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

int main(void) {
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "plow_hsa_init: %s\n", plow_hsa_last_error()); return 1; }
    printf("HSA up: %d GPU agent(s)\n", plow_hsa_device_count(h));

    char nm[64]; uint32_t cus = 0, lds = 0;
    if (plow_hsa_device_info(h, 0, nm, &cus, &lds) != 0) {
        fprintf(stderr, "device_info: %s\n", plow_hsa_last_error()); return 1;
    }
    printf("dev0: %s  CUs=%u  LDS=%u B\n\n", nm, cus, lds);

    /* --- load the code object -------------------------------------------- */
    FILE* f = fopen("test_kernels.elf", "rb");
    if (!f) { perror("test_kernels.elf"); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(co_n);
    if (fread(co, 1, co_n, f) != (size_t)co_n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, co_n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }

    plow_hsa_kernel k_rms, k_hnr, k_res, k_glu, k_emb, k_cap;
    if (plow_hsa_get_kernel(h, 0, "gemma_rmsnorm_bf16", &k_rms) ||
        plow_hsa_get_kernel(h, 0, "gemma_headnorm_rope_bf16", &k_hnr) ||
        plow_hsa_get_kernel(h, 0, "gemma_residual_add_bf16", &k_res) ||
        plow_hsa_get_kernel(h, 0, "gemma_geglu_bf16", &k_glu) ||
        plow_hsa_get_kernel(h, 0, "gemma_embed_scale_bf16", &k_emb) ||
        plow_hsa_get_kernel(h, 0, "gemma_logit_softcap_bf16", &k_cap)) {
        fprintf(stderr, "symbol: %s\n", plow_hsa_last_error()); return 1;
    }
    printf("kernels resolved\n\n");

    srand(7);
    const float EPS = 1e-6f;

    /* ================= A2: RMSNorm over hidden (31B: 5376) ================= */
    {
        const unsigned rows = 8, feat = 5376;
        const size_t n = (size_t)rows * feat;
        bf16* hx = plow_hsa_alloc_host(h, n * 2);
        bf16* hg = plow_hsa_alloc_host(h, feat * 2);
        bf16* ho = plow_hsa_alloc_host(h, n * 2);
        for (size_t i = 0; i < n; i++) hx[i] = f2bf(frand());
        for (unsigned i = 0; i < feat; i++) hg[i] = f2bf(1.0f + 0.1f * frand());

        void *dx = plow_hsa_alloc(h, 0, n * 2), *dg = plow_hsa_alloc(h, 0, feat * 2),
             *dof = plow_hsa_alloc(h, 0, n * 2);
        plow_hsa_copy_h2d(h, 0, dx, hx, n * 2);
        plow_hsa_copy_h2d(h, 0, dg, hg, feat * 2);

        struct __attribute__((packed)) {
            void* out; const void* x; const void* g; unsigned rows, feat; float eps;
        } a = {dof, dx, dg, rows, feat, EPS};
        plow_hsa_launch(h, 0, &k_rms, rows * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
        plow_hsa_wait(h, 0);
        plow_hsa_copy_d2h(h, 0, ho, dof, n * 2);

        float* want = malloc(n * sizeof(float));
        for (unsigned r = 0; r < rows; r++) {
            double ss = 0.0;
            for (unsigned i = 0; i < feat; i++) {
                const double v = bf2f(hx[(size_t)r * feat + i]);
                ss += v * v;
            }
            /* NOTE: * w, not * (1 + w). eps inside the rsqrt. */
            const double inv = pow(ss / feat + EPS, -0.5);
            for (unsigned i = 0; i < feat; i++)
                want[(size_t)r * feat + i] =
                    (float)(bf2f(hx[(size_t)r * feat + i]) * inv * bf2f(hg[i]));
        }
        check("rmsnorm (hidden=5376)", ho, want, n);
        free(want);
    }

    /* ===== A5: per-head norm + RoPE, both layer geometries ================ *
     * hd=256 sliding (full rotary) and hd=512 global (partial: only the first
     * 64 freqs are non-zero, so dims [64,256) and [320,512) must come back
     * BIT-UNCHANGED through the cos=1/sin=0 path).                           */
    for (int cfg = 0; cfg < 2; cfg++) {
        const unsigned hd = cfg ? 512 : 256;
        const unsigned n_tok = 5, n_head = 4, H2 = hd / 2;
        const unsigned rope_angles = cfg ? 64 : 128; /* int(prf * hd / 2) */
        const float theta = cfg ? 1e6f : 1e4f;
        const size_t n = (size_t)n_tok * n_head * hd;
        const unsigned MAXP = 64;

        bf16* hx = plow_hsa_alloc_host(h, n * 2);
        bf16* hg = plow_hsa_alloc_host(h, hd * 2);
        bf16* ho = plow_hsa_alloc_host(h, n * 2);
        float* hc = plow_hsa_alloc_host(h, (size_t)MAXP * H2 * 4);
        float* hs = plow_hsa_alloc_host(h, (size_t)MAXP * H2 * 4);
        int* hp = plow_hsa_alloc_host(h, n_tok * 4);
        for (size_t i = 0; i < n; i++) hx[i] = f2bf(frand());
        for (unsigned i = 0; i < hd; i++) hg[i] = f2bf(1.0f + 0.1f * frand());
        for (unsigned t = 0; t < n_tok; t++) hp[t] = (int)(t + 3);

        /* "proportional" rope: inv_freq[j] = theta^(-2j/hd) for j < rope_angles,
         * and exactly 0.0 beyond -> cos=1, sin=0 -> identity on the NoPE dims. */
        for (unsigned p = 0; p < MAXP; p++)
            for (unsigned j = 0; j < H2; j++) {
                const double inv_freq =
                    (j < rope_angles) ? pow((double)theta, -2.0 * (double)j / (double)hd) : 0.0;
                const double ang = (double)p * inv_freq;
                hc[(size_t)p * H2 + j] = (float)cos(ang);
                hs[(size_t)p * H2 + j] = (float)sin(ang);
            }

        void* dx = plow_hsa_alloc(h, 0, n * 2);
        void* dg = plow_hsa_alloc(h, 0, hd * 2);
        void* dof = plow_hsa_alloc(h, 0, n * 2);
        void* dc = plow_hsa_alloc(h, 0, (size_t)MAXP * H2 * 4);
        void* ds = plow_hsa_alloc(h, 0, (size_t)MAXP * H2 * 4);
        void* dp = plow_hsa_alloc(h, 0, n_tok * 4);
        plow_hsa_copy_h2d(h, 0, dx, hx, n * 2);
        plow_hsa_copy_h2d(h, 0, dg, hg, hd * 2);
        plow_hsa_copy_h2d(h, 0, dc, hc, (size_t)MAXP * H2 * 4);
        plow_hsa_copy_h2d(h, 0, ds, hs, (size_t)MAXP * H2 * 4);
        plow_hsa_copy_h2d(h, 0, dp, hp, n_tok * 4);

        struct __attribute__((packed)) {
            void* out; const void* x; const void* g; const void* c; const void* s;
            const void* pos; unsigned n_tok, n_head, hd; float eps;
        } a = {dof, dx, dg, dc, ds, dp, n_tok, n_head, hd, EPS};

        const unsigned waves = n_tok * n_head;
        plow_hsa_launch(h, 0, &k_hnr, ((waves * 64 + PLOW_WG_THREADS - 1) / PLOW_WG_THREADS) * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &a, sizeof(a));
        plow_hsa_wait(h, 0);
        plow_hsa_copy_d2h(h, 0, ho, dof, n * 2);

        float* want = malloc(n * sizeof(float));
        for (unsigned t = 0; t < n_tok; t++)
            for (unsigned hh = 0; hh < n_head; hh++) {
                const size_t base = ((size_t)t * n_head + hh) * hd;
                double ss = 0.0;
                for (unsigned i = 0; i < hd; i++) {
                    const double v = bf2f(hx[base + i]);
                    ss += v * v;
                }
                const double inv = pow(ss / hd + EPS, -0.5);
                float nv[512];
                for (unsigned i = 0; i < hd; i++)
                    nv[i] = (float)(bf2f(hx[base + i]) * inv * bf2f(hg[i]));
                /* half-split rotate: pairs (i, i+H2) */
                for (unsigned i = 0; i < H2; i++) {
                    const float c = hc[(size_t)hp[t] * H2 + i];
                    const float s = hs[(size_t)hp[t] * H2 + i];
                    want[base + i]      = nv[i] * c - nv[i + H2] * s;
                    want[base + i + H2] = nv[i + H2] * c + nv[i] * s;
                }
            }
        char lbl[64];
        snprintf(lbl, sizeof(lbl), "headnorm+rope (hd=%u,%s)", hd, cfg ? "partial" : "full");
        check(lbl, ho, want, n);

        /* The NoPE dims must be pure identity — cos=1, sin=0 means `v*1 - v2*0`,
         * which is exact in f32, so the kernel's bf16 output must be BIT-equal to
         * the rounded un-rotated value. Compare in bf16 bits: comparing against
         * the f32 reference would only be testing bf16 rounding. */
        if (cfg) {
            int drift = 0;
            for (unsigned t = 0; t < n_tok; t++)
                for (unsigned hh = 0; hh < n_head; hh++) {
                    const size_t base = ((size_t)t * n_head + hh) * hd;
                    for (unsigned i = rope_angles; i < H2; i++) {
                        if (ho[base + i] != f2bf(want[base + i])) drift++;
                        if (ho[base + i + H2] != f2bf(want[base + i + H2])) drift++;
                    }
                }
            printf("  %-28s %s  (NoPE dims pass through bit-exact)\n",
                   "partial-rope identity", drift ? "FAIL" : "PASS");
            if (drift) fails++;
        }
        free(want);
    }

    /* ================= A3 / A4 / A7: elementwise ========================== */
    {
        const unsigned n = 21504 * 4; /* 31B intermediate, 4 rows */
        bf16 *ha = plow_hsa_alloc_host(h, n * 2), *hb = plow_hsa_alloc_host(h, n * 2),
             *ho = plow_hsa_alloc_host(h, n * 2);
        for (unsigned i = 0; i < n; i++) { ha[i] = f2bf(frand()); hb[i] = f2bf(frand()); }
        void *da = plow_hsa_alloc(h, 0, n * 2), *db = plow_hsa_alloc(h, 0, n * 2),
             *dof = plow_hsa_alloc(h, 0, n * 2);
        plow_hsa_copy_h2d(h, 0, da, ha, n * 2);
        plow_hsa_copy_h2d(h, 0, db, hb, n * 2);
        float* want = malloc(n * sizeof(float));
        const unsigned grid = (n + PLOW_WG_THREADS - 1) / PLOW_WG_THREADS * PLOW_WG_THREADS;

        const float LS = 1.37f; /* layer_scalar */
        struct __attribute__((packed)) {
            void* o; const void* a; const void* b; unsigned n; float s;
        } ra = {dof, da, db, n, LS};
        plow_hsa_launch(h, 0, &k_res, grid, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &ra, sizeof(ra));
        plow_hsa_wait(h, 0);
        plow_hsa_copy_d2h(h, 0, ho, dof, n * 2);
        for (unsigned i = 0; i < n; i++) want[i] = (bf2f(ha[i]) + bf2f(hb[i])) * LS;
        check("residual add * layer_scalar", ho, want, n);

        struct __attribute__((packed)) {
            void* o; const void* g; const void* u; unsigned n;
        } ga = {dof, da, db, n};
        plow_hsa_launch(h, 0, &k_glu, grid, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &ga, sizeof(ga));
        plow_hsa_wait(h, 0);
        plow_hsa_copy_d2h(h, 0, ho, dof, n * 2);
        for (unsigned i = 0; i < n; i++) want[i] = gelu_tanh(bf2f(ha[i])) * bf2f(hb[i]);
        check("geglu (gelu_tanh, not silu)", ho, want, n);

        const float CAP = 30.0f;
        struct __attribute__((packed)) { void* o; const void* x; unsigned n; float c; } ca =
            {dof, da, n, CAP};
        plow_hsa_launch(h, 0, &k_cap, grid, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &ca, sizeof(ca));
        plow_hsa_wait(h, 0);
        plow_hsa_copy_d2h(h, 0, ho, dof, n * 2);
        for (unsigned i = 0; i < n; i++) want[i] = CAP * tanhf(bf2f(ha[i]) / CAP);
        check("logit softcap (cap=30)", ho, want, n);
        free(want);
    }

    /* ================= A1: embedding gather + scale ======================= */
    {
        const unsigned vocab = 4096, hidden = 5376, n_tok = 6;
        const float SCALE = 73.5f; /* bf16(sqrt(5376)) — NOT 73.3212 */
        bf16* ht = plow_hsa_alloc_host(h, (size_t)vocab * hidden * 2);
        bf16* ho = plow_hsa_alloc_host(h, (size_t)n_tok * hidden * 2);
        int* hi = plow_hsa_alloc_host(h, n_tok * 4);
        for (size_t i = 0; i < (size_t)vocab * hidden; i++) ht[i] = f2bf(frand());
        for (unsigned t = 0; t < n_tok; t++) hi[t] = (int)(rand() % vocab);

        void* dt = plow_hsa_alloc(h, 0, (size_t)vocab * hidden * 2);
        void* di = plow_hsa_alloc(h, 0, n_tok * 4);
        void* dof = plow_hsa_alloc(h, 0, (size_t)n_tok * hidden * 2);
        plow_hsa_copy_h2d(h, 0, dt, ht, (size_t)vocab * hidden * 2);
        plow_hsa_copy_h2d(h, 0, di, hi, n_tok * 4);

        struct __attribute__((packed)) {
            void* o; const void* tbl; const void* ids; unsigned n_tok, hidden; float s;
        } ea = {dof, dt, di, n_tok, hidden, SCALE};
        plow_hsa_launch(h, 0, &k_emb, n_tok * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &ea, sizeof(ea));
        plow_hsa_wait(h, 0);
        plow_hsa_copy_d2h(h, 0, ho, dof, (size_t)n_tok * hidden * 2);

        const size_t n = (size_t)n_tok * hidden;
        float* want = malloc(n * sizeof(float));
        for (unsigned t = 0; t < n_tok; t++)
            for (unsigned i = 0; i < hidden; i++)
                want[(size_t)t * hidden + i] = bf2f(ht[(size_t)hi[t] * hidden + i]) * SCALE;
        check("embed gather * bf16(sqrt(H))", ho, want, n);
        free(want);
    }

    printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL TIER-A KERNELS PASS",
           fails, fails == 1 ? "" : "s");
    plow_hsa_shutdown(h);
    return fails ? 1 : 0;
}
