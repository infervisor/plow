/* rope_interleave_glm_test.c — GLM-5.2 INTERLEAVED partial RoPE (decode) on device vs a CPU ref.
 *
 * GLM-5.2 MLA applies GPT-J-style INTERLEAVED RoPE (rotate adjacent pairs (x[2i], x[2i+1])) to the
 * 64-dim rope slice of q (per head) and the shared k_rope, with rope_theta=8e6, dynamically per decode
 * step (position index per token). plow's existing d_headnorm_rope is HALF-SPLIT (NeoX); this drives
 * the new INTERLEAVE=true template path (glm_rope_interleave_bf16 wrapper, HD=64, no norm) and checks
 * it against a CPU interleaved reference == HF transformers apply_rotary_pos_emb_interleave:
 *   for i in [0,32): out[2i]   = q[2i]*cos_i - q[2i+1]*sin_i
 *                    out[2i+1] = q[2i+1]*cos_i + q[2i]*sin_i,  cos_i=cos(pos*inv_freq[i]).
 *
 * Build with scripts/build_block_fp8.sh (shares test_kernels.elf); run under `sg render` on one GPU.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned short bf16;
static float bf2f(bf16 b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
static bf16 f2bf(float f) {
    unsigned u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (bf16)(u >> 16);
}

#define HD 64
#define H2 32
static int fails = 0;

/* One case: ntok tokens, nhead heads, 64-dim rope slice, at the given positions. */
static void run(plow_hsa* h, plow_hsa_kernel* k, unsigned NCU, const char* label, unsigned ntok,
                unsigned nhead, const int* positions, double theta) {
    const size_t nX = (size_t)ntok * nhead * HD;
    /* inv_freq[i] = theta^(-2i/HD); cos/sin table sized to max position + 1. */
    int maxpos = 0;
    for (unsigned t = 0; t < ntok; t++) if (positions[t] > maxpos) maxpos = positions[t];
    const unsigned NP = (unsigned)maxpos + 1;
    double invf[H2];
    for (int i = 0; i < H2; i++) invf[i] = pow(theta, -(2.0 * i) / (double)HD);

    bf16* hx = plow_hsa_alloc_host(h, nX * 2);
    float* hcos = plow_hsa_alloc_host(h, (size_t)NP * H2 * 4);
    float* hsin = plow_hsa_alloc_host(h, (size_t)NP * H2 * 4);
    int* hpos = plow_hsa_alloc_host(h, (size_t)ntok * 4);
    bf16* hout = plow_hsa_alloc_host(h, nX * 2);
    for (size_t i = 0; i < nX; i++) hx[i] = f2bf(((float)(rand() % 4001) - 2000.0f) / 1000.0f);
    for (unsigned pp = 0; pp < NP; pp++)
        for (int i = 0; i < H2; i++) {
            hcos[(size_t)pp * H2 + i] = (float)cos((double)pp * invf[i]);
            hsin[(size_t)pp * H2 + i] = (float)sin((double)pp * invf[i]);
        }
    for (unsigned t = 0; t < ntok; t++) hpos[t] = positions[t];

    void* dx = plow_hsa_alloc(h, 0, nX * 2); plow_hsa_copy_h2d(h, 0, dx, hx, nX * 2);
    void* dcos = plow_hsa_alloc(h, 0, (size_t)NP * H2 * 4); plow_hsa_copy_h2d(h, 0, dcos, hcos, (size_t)NP * H2 * 4);
    void* dsin = plow_hsa_alloc(h, 0, (size_t)NP * H2 * 4); plow_hsa_copy_h2d(h, 0, dsin, hsin, (size_t)NP * H2 * 4);
    void* dpos = plow_hsa_alloc(h, 0, (size_t)ntok * 4); plow_hsa_copy_h2d(h, 0, dpos, hpos, (size_t)ntok * 4);
    void* dout = plow_hsa_alloc(h, 0, nX * 2);

    struct __attribute__((packed)) {
        void* out; const void* x; const void* cos; const void* sin; const void* pos;
        unsigned ntok, nhead;
    } args = {dout, dx, dcos, dsin, dpos, ntok, nhead};
    plow_hsa_launch(h, 0, k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &args, sizeof(args));
    plow_hsa_wait(h, 0);
    plow_hsa_copy_d2h(h, 0, hout, dout, nX * 2);

    double worst = 0.0;
    for (unsigned t = 0; t < ntok; t++)
        for (unsigned hh = 0; hh < nhead; hh++) {
            const size_t b = ((size_t)t * nhead + hh) * HD;
            for (int i = 0; i < H2; i++) {
                const double c = cos((double)positions[t] * invf[i]);
                const double s = sin((double)positions[t] * invf[i]);
                const double e = bf2f(hx[b + 2 * i]), o = bf2f(hx[b + 2 * i + 1]);
                const double we = e * c - o * s;      /* interleaved even */
                const double wo = o * c + e * s;      /* interleaved odd  */
                const double ge = bf2f(hout[b + 2 * i]), go = bf2f(hout[b + 2 * i + 1]);
                double r0 = fabs(ge - we) / (fabs(we) + 1e-2);
                double r1 = fabs(go - wo) / (fabs(wo) + 1e-2);
                if (r0 > worst) worst = r0;
                if (r1 > worst) worst = r1;
            }
        }
    const int ok = worst < 1e-2;
    printf("  %-26s ntok=%u nhead=%u pos0=%d  %s (worst rel %.5f)\n", label, ntok, nhead,
           positions[0], ok ? "PASS" : "FAIL", worst);
    if (!ok) fails++;

    plow_hsa_free(h, dx); plow_hsa_free(h, dcos); plow_hsa_free(h, dsin);
    plow_hsa_free(h, dpos); plow_hsa_free(h, dout);
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "test_kernels.elf";
    setbuf(stdout, NULL);
    srand(4321);
    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "%s\n", plow_hsa_last_error()); return 1; }
    FILE* f = fopen(elf, "rb");
    if (!f) { perror(elf); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc(n);
    if (fread(co, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, n) != 0) {
        fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 1;
    }
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, "glm_rope_interleave_bf16", &k) != 0) {
        fprintf(stderr, "no kernel glm_rope_interleave_bf16: %s\n", plow_hsa_last_error()); return 1;
    }

    const unsigned NCU = 8;
    const double THETA = 8e6; /* GLM-5.2 rope_theta */
    printf("GLM-5.2 interleaved partial RoPE (HD=64, theta=8e6):\n");
    int p_q[1] = {37};
    run(h, &k, NCU, "q_rope [nhead=64] @pos37",   1, 64, p_q, THETA);
    int p_k[1] = {37};
    run(h, &k, NCU, "k_rope [nhead=1]  @pos37",    1,  1, p_k, THETA);
    int p_0[1] = {0};
    run(h, &k, NCU, "q_rope @pos0 (identity cos)", 1, 64, p_0, THETA);
    int p_big[1] = {100000};
    run(h, &k, NCU, "q_rope @pos100000 (long ctx)",1, 64, p_big, THETA);
    int p_multi[3] = {5, 128, 4096};
    run(h, &k, NCU, "multi-token pos 5/128/4096",  3, 64, p_multi, THETA);

    printf(fails ? "FAIL (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
