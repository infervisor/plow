/* interp_gemm_bench.c — measure the GEMM *through the persistent interpreter*.
 *
 * This is the only number that matters. A standalone GEMM kernel and the same GEMM
 * inlined into the interpreter get DIFFERENT register allocations, and on CDNA that
 * difference is worth 3-4x: if the accumulator lands in AccVGPRs while the MFMA
 * operands sit in arch VGPRs, the compiler brackets every MFMA with
 * v_accvgpr_read/write moves. Measured standalone: 256x256 (agpr=128) 212 TF/s vs
 * 256x128 (agpr=0) 770 TF/s -- same math, same tile family, 3.6x apart.
 *
 * So we benchmark the serving path itself: build a one-instruction program, hand it
 * to the interpreter, and time the whole dispatch.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static unsigned short f2bf(float f) {
    union { float f; unsigned u; } c;
    c.f = f;
    unsigned r = c.u + 0x7fff + ((c.u >> 16) & 1);
    return (unsigned short)(r >> 16);
}

struct Shape { const char* name; unsigned M, N, K; };

int main(void) {
    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; unsigned cus = 0, lds = 0;
    plow_hsa_device_info(h, 0, gfx, &cus, &lds);
    printf("dev0: %s  CUs=%u  LDS=%u B\n", gfx, cus, lds);

    FILE* f = fopen("interp.elf", "rb");
    if (!f) { printf("interp.elf missing\n"); return 1; }
    fseek(f, 0, SEEK_END); long co_n = ftell(f); fseek(f, 0, SEEK_SET);
    void* co = malloc((size_t)co_n);
    if (fread(co, 1, (size_t)co_n, f) != (size_t)co_n) return 1;
    fclose(f);
    if (plow_hsa_load_code_object(h, 0, co, (size_t)co_n)) { printf("load failed\n"); return 1; }
    plow_hsa_kernel k;
    if (plow_hsa_get_kernel(h, 0, "plow_interp_gfx950", &k)) { printf("no kernel\n"); return 1; }
    printf("interpreter: LDS=%u B, kernarg=%u B\n\n", k.group_segment_size, k.kernarg_size);

    struct Shape shapes[] = {
        {"q_proj  (sliding)", 4096, 8192, 5376},
        {"kv_proj (sliding)", 4096, 4096, 5376},
        {"q_proj  (global) ", 4096, 16384, 5376},
        {"o_proj  (sliding)", 4096, 5376, 8192},
        {"gate/up_proj     ", 4096, 21504, 5376},
        {"down_proj        ", 4096, 5376, 21504},
    };
    const int NS = 6;
    const unsigned NCU = cus;

    printf("GEMM through the persistent interpreter (%u WGs x %u thr):\n", NCU, PLOW_WG_THREADS);
    printf("  peak on this machine: 1660 TF/s sustained bf16 MFMA\n\n");

    for (int s = 0; s < NS; s++) {
        const unsigned M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
        void *dA, *dB, *dC;
        dA = plow_hsa_alloc(h, 0, (size_t)M * K * 2);
        dB = plow_hsa_alloc(h, 0, (size_t)N * K * 2);
        dC = plow_hsa_alloc(h, 0, (size_t)M * N * 2);

        unsigned short* ha = malloc((size_t)M * K * 2);
        unsigned short* hb = malloc((size_t)N * K * 2);
        srand(5);
        for (size_t i = 0; i < (size_t)M * K; i++) ha[i] = f2bf((float)(rand() % 200 - 100) / 400.0f);
        for (size_t i = 0; i < (size_t)N * K; i++) hb[i] = f2bf((float)(rand() % 200 - 100) / 400.0f);
        plow_hsa_upload(h, 0, dA, ha, (size_t)M * K * 2);
        plow_hsa_upload(h, 0, dB, hb, (size_t)N * K * 2);

        /* One-instruction program: C = A . B^T */
        PlowDevInst inst;
        memset(&inst, 0, sizeof(inst));
        inst.op = PLOW_DOP_GEMM;
        inst.blocks = (unsigned short)NCU;
        inst.t[0] = 0; inst.t[1] = 1; inst.t[2] = 2;
        for (int i = 3; i < 8; i++) inst.t[i] = PLOW_TENSOR_NONE;
        inst.i[0] = M; inst.i[1] = N; inst.i[2] = K;

        /* calloc, NOT malloc: a stream entry carries `flags`, and a garbage PLOW_SE_FINE bit
         * sends the interpreter to a wait list that does not exist. Zero == the coarse path. */
        PlowStreamEnt* stream = calloc(NCU, sizeof(PlowStreamEnt));
        unsigned* sofs = malloc(sizeof(unsigned) * NCU);
        unsigned* slen = malloc(sizeof(unsigned) * NCU);
        for (unsigned i = 0; i < NCU; i++) {
            stream[i].inst = 0; stream[i].slice = i;
            sofs[i] = i; slen[i] = 1;
        }
        void* tensors_h[3] = {dC, dA, dB};

        void *d_inst, *d_stream, *d_sofs, *d_slen, *d_ctr, *d_tensors;
        d_inst = plow_hsa_alloc(h, 0, sizeof(PlowDevInst));
        d_stream = plow_hsa_alloc(h, 0, sizeof(PlowStreamEnt) * NCU);
        d_sofs = plow_hsa_alloc(h, 0, sizeof(unsigned) * NCU);
        d_slen = plow_hsa_alloc(h, 0, sizeof(unsigned) * NCU);
        d_ctr = plow_hsa_alloc(h, 0, sizeof(unsigned) * 8);
        d_tensors = plow_hsa_alloc(h, 0, sizeof(void*) * 3);
        plow_hsa_upload(h, 0, d_inst, &inst, sizeof(inst));
        plow_hsa_upload(h, 0, d_stream, stream, sizeof(PlowStreamEnt) * NCU);
        plow_hsa_upload(h, 0, d_sofs, sofs, sizeof(unsigned) * NCU);
        plow_hsa_upload(h, 0, d_slen, slen, sizeof(unsigned) * NCU);
        plow_hsa_upload(h, 0, d_tensors, tensors_h, sizeof(void*) * 3);
        unsigned zero[8] = {0};
        plow_hsa_upload(h, 0, d_ctr, zero, sizeof(zero));

        PlowProgram prog;
        memset(&prog, 0, sizeof(prog));
        prog.insts = d_inst; prog.stream = d_stream; prog.stream_ofs = d_sofs;
        prog.stream_len = d_slen; prog.waits = NULL; prog.succs = NULL;
        prog.counters = d_ctr; prog.tensors = d_tensors;

        plow_hsa_launch(h, 0, &k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &prog,
                        sizeof(prog));
        plow_hsa_wait(h, 0);

        /* Do NOT reset the counters inside the timed loop. plow_hsa_upload() pins host
         * memory (hsa_amd_memory_lock) and issues an SDMA copy -- that is a syscall-
         * class operation, and at these kernel times it DOMINATED the measurement: it
         * made a GEMM whose registers are provably identical to the fast standalone
         * kernel (256 arch / 0 agpr / 32 B scratch) look 2x slower, and sent me hunting
         * a register-allocation bug that did not exist.
         *
         * Nothing waits on the counters in a one-instruction program (wait_len == 0), so
         * they can just accumulate across reps. */
        const int reps = 5;
        const double t0 = now();
        for (int r = 0; r < reps; r++) {
            plow_hsa_launch(h, 0, &k, NCU * PLOW_WG_THREADS, 1, 1, PLOW_WG_THREADS, 1, 1, 0, &prog,
                            sizeof(prog));
            plow_hsa_wait(h, 0);
        }
        const double dt = (now() - t0) / reps;
        const double tf = 2.0 * M * N * K / dt / 1e12;

        /* spot-check one output element against a CPU dot product */
        unsigned short* hc = malloc((size_t)M * N * 2);
        plow_hsa_download(h, 0, hc, dC, (size_t)M * N * 2);
        int bad = 0;
        for (int t = 0; t < 8; t++) {
            const unsigned m = (unsigned)(rand() % (int)M), n = (unsigned)(rand() % (int)N);
            double acc = 0;
            for (unsigned kk = 0; kk < K; kk++) {
                union { unsigned u; float f; } x, y;
                x.u = (unsigned)ha[(size_t)m * K + kk] << 16;
                y.u = (unsigned)hb[(size_t)n * K + kk] << 16;
                acc += (double)x.f * y.f;
            }
            union { unsigned u; float f; } g;
            g.u = (unsigned)hc[(size_t)m * N + n] << 16;
            const double rel = fabs(g.f - acc) / (fabs(acc) + 1e-3);
            if (rel > 0.02) bad++;
        }
        printf("  %s %5u x %6u x %5u  %7.3f ms  %7.1f TF/s  (%4.1f%%)  %s\n", shapes[s].name, M, N,
               K, dt * 1e3, tf, 100.0 * tf / 1660.0, bad ? "MISMATCH" : "ok");

        free(ha); free(hb); free(hc); free(stream); free(sofs); free(slen);
        plow_hsa_free(h, dA); plow_hsa_free(h, dB); plow_hsa_free(h, dC);
        plow_hsa_free(h, d_inst); plow_hsa_free(h, d_stream); plow_hsa_free(h, d_sofs);
        plow_hsa_free(h, d_slen); plow_hsa_free(h, d_ctr); plow_hsa_free(h, d_tensors);
    }
    plow_hsa_shutdown(h);
    return 0;
}
