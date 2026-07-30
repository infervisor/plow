/* moe_mxfp4_decode_gfx950_test.hip — the DECODE expert bodies against a CPU reference, at every
 * value of the runtime encoding field they claim to support, and at the values they do not.
 *                                                                              [MXFP4-DECODE]
 * WHY THIS EXISTS. Ops 40/43/48/49 pick their weight decode from `i[6]` at RUNTIME, and until
 * this test the only thing standing behind that field was reading the code. Three separate arms
 * read it (`wave_dot_enc`, the flattened group body, and — new — the MFMA group body) and one of
 * them used to DROP it. A dropped encoding does not fault and does not produce zeros: it produces
 * finite numbers from the wrong bytes. So every arm is checked against the same f64 reference,
 * and the encodings with no body are checked for a NaN rather than for a number.
 *
 * WHAT IT PINS:
 *   1. fdot2 (`d_moe_expert_glu_fp8_blk`) at enc=1 and enc=2 vs the CPU oracle.
 *   2. The MFMA group body (`PLOW_MOE_MFMA=1`) at enc=1 and — the arm this test was written for —
 *      enc=2, vs the SAME oracle. Before, enc=2 on that body read fp4 nibbles as block-fp8.
 *   3. The DOWN twin at enc=2.
 *   4. REFUSAL: enc=0 (bf16, which is also what an unwritten `i[6]` reads as) and enc=3 must
 *      produce NaN, not a plausible number.
 *
 * GEOMETRY is Kimi-K3-shaped where it is cheap to be: the routed experts run at the LATENT width
 * K=3584, not hidden 7168, so the K-tail behaviour of both bodies is exercised at a K that is
 * NOT a multiple of the 2048-wide fp4 wave chunk.
 *
 *   nix develop --command bash -c '/opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 -w \
 *       -I runtime/amd -I runtime/common runtime/tests/moe_mxfp4_decode_gfx950_test.hip \
 *       -o /tmp/moe_mxfp4_dec'
 *   sg render -c '/tmp/moe_mxfp4_dec'
 */
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define PLOW_MOE_MFMA 1 /* so d_moe_group_glu_mfma is compiled and callable directly */
#include "amd_common.h"
#include "op_moe.h"

extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void t_glu_fdot(bf16* fu, const bf16* x, const unsigned char* tab, const unsigned long long* wtab,
                const unsigned long long* stab, unsigned k, unsigned I, unsigned H, unsigned n_exp,
                unsigned act, unsigned enc) {
    for (unsigned s = 0; s < k; s++)
        d_moe_expert_glu_fp8_blk(fu, x, tab, wtab, stab, s, I, H, n_exp, act, blockIdx.x, gridDim.x,
                                 enc, 0.0f, 0.0f);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void t_glu_mfma(bf16* fu, const bf16* x, const unsigned char* tab, const unsigned long long* wtab,
                const unsigned long long* stab, unsigned k, unsigned I, unsigned H, unsigned n_exp,
                unsigned act, unsigned enc) {
    d_moe_group_glu_mfma(fu, x, tab, wtab, stab, k, I, H, n_exp, act, blockIdx.x, gridDim.x, enc,
                         0.0f, 0.0f);
}
extern "C" __global__ __launch_bounds__(PLOW_THREADS, PLOW_WAVES / 4)
void t_down(float* part, const bf16* fu, const unsigned char* tab, const unsigned long long* wtab,
            const unsigned long long* stab, unsigned k, unsigned H, unsigned I, unsigned n_exp,
            unsigned enc) {
    for (unsigned s = 0; s < k; s++)
        d_moe_expert_down_fp8_blk(part, fu, tab, wtab, stab, s, H, I, n_exp, blockIdx.x, gridDim.x,
                                  enc);
}

/* ---- host-side format decoders. Independent of the device helpers ON PURPOSE. ---- */
static double fp4d(unsigned n) {
    const double T[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    return (n & 8) ? -T[n & 7] : T[n & 7];
}
static double e8(unsigned char b) { return ldexp(1.0, (int)b - 127); }
static double e4m3d(unsigned char b) { /* OCP e4m3fn: 1.4.3, bias 7, no inf */
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    double v = e ? ldexp(1.0 + m / 8.0, e - 7) : ldexp(m / 8.0, -6);
    return s ? -v : v;
}
static double bf(unsigned short h) {
    unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return (double)f;
}
static unsigned short f2b(double d) {
    float f = (float)d; unsigned u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u); return (unsigned short)(u >> 16);
}
#define HC(x) do{hipError_t e_=(x); if(e_){printf("hip err %s @%d\n",hipGetErrorString(e_),__LINE__);exit(1);} }while(0)

static unsigned rs = 12345u;
static unsigned rnd() { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

int main() {
    /* K3-shaped but small enough to reference in f64: K = latent-like, I = a few tiles. */
    const unsigned H = 3584, I = 128, n_exp = 4, k = 2, act = PLOW_MOE_ACT_SILU;
    const unsigned KB = (H + 127) >> 7;          /* block-fp8 K blocks = 28 */
    const unsigned NB = (I + 127) >> 7;          /* block-fp8 N blocks = 1  */
    const unsigned slots[2] = {2, 0};            /* the two routed experts  */

    std::vector<unsigned char> tab(k * 8, 0);
    for (unsigned s = 0; s < k; s++) {
        *(unsigned*)&tab[s * 8] = slots[s];
        *(float*)&tab[s * 8 + 4] = 0.5f + 0.25f * (float)s;
    }
    std::vector<unsigned short> x(H);
    for (unsigned i = 0; i < H; i++) x[i] = f2b(((double)(rnd() % 2001) - 1000.0) * 0.001);

    /* Per expert: MXFP4 gate/up/down (+E8M0 rows) AND block-fp8 gate/up/down (+f32 grids). The
     * two encodings carry DIFFERENT numbers; each is referenced against its own oracle. */
    std::vector<std::vector<unsigned char>> Q(n_exp * 3), QS(n_exp * 3), F(n_exp * 3);
    std::vector<std::vector<float>> FS(n_exp * 3);
    for (unsigned e = 0; e < n_exp; e++)
        for (int p = 0; p < 3; p++) {
            const unsigned N = (p == 2) ? H : I, Kd = (p == 2) ? I : H; /* down is [H][I] */
            Q[e * 3 + p].resize((size_t)N * Kd / 2);
            QS[e * 3 + p].resize((size_t)N * Kd / 32);
            F[e * 3 + p].resize((size_t)N * Kd);
            FS[e * 3 + p].resize((size_t)((N + 127) / 128) * ((Kd + 127) / 128));
            for (auto& b : Q[e * 3 + p]) b = (unsigned char)(rnd() & 0xFFu);
            for (auto& b : QS[e * 3 + p]) b = (unsigned char)(124 + (rnd() % 7));
            for (auto& b : F[e * 3 + p]) { unsigned char v = (unsigned char)(rnd() & 0xFFu);
                                          if ((v & 0x7F) == 0x7F) v ^= 1; b = v; } /* no NaN */
            for (auto& v : FS[e * 3 + p]) v = 0.004f + 0.001f * (float)(rnd() % 17);
        }

    /* device copies */
    bf16* dx; unsigned char* dtab; bf16* dfu; float* dpart;
    unsigned long long *dwt4, *dst4, *dwt8, *dst8;
    HC(hipMalloc(&dx, H * 2)); HC(hipMemcpy(dx, x.data(), H * 2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dtab, tab.size())); HC(hipMemcpy(dtab, tab.data(), tab.size(), hipMemcpyHostToDevice));
    HC(hipMalloc(&dfu, (size_t)k * I * 2));
    HC(hipMalloc(&dpart, (size_t)k * H * 4));
    std::vector<unsigned long long> wt4(n_exp * 3), st4(n_exp * 3), wt8(n_exp * 3), st8(n_exp * 3);
    for (unsigned i = 0; i < n_exp * 3; i++) {
        void *a, *b, *c, *d;
        HC(hipMalloc(&a, Q[i].size()));  HC(hipMemcpy(a, Q[i].data(), Q[i].size(), hipMemcpyHostToDevice));
        HC(hipMalloc(&b, QS[i].size())); HC(hipMemcpy(b, QS[i].data(), QS[i].size(), hipMemcpyHostToDevice));
        HC(hipMalloc(&c, F[i].size()));  HC(hipMemcpy(c, F[i].data(), F[i].size(), hipMemcpyHostToDevice));
        HC(hipMalloc(&d, FS[i].size() * 4)); HC(hipMemcpy(d, FS[i].data(), FS[i].size() * 4, hipMemcpyHostToDevice));
        wt4[i] = (unsigned long long)a; st4[i] = (unsigned long long)b;
        wt8[i] = (unsigned long long)c; st8[i] = (unsigned long long)d;
    }
    HC(hipMalloc(&dwt4, n_exp * 24)); HC(hipMemcpy(dwt4, wt4.data(), n_exp * 24, hipMemcpyHostToDevice));
    HC(hipMalloc(&dst4, n_exp * 24)); HC(hipMemcpy(dst4, st4.data(), n_exp * 24, hipMemcpyHostToDevice));
    HC(hipMalloc(&dwt8, n_exp * 24)); HC(hipMemcpy(dwt8, wt8.data(), n_exp * 24, hipMemcpyHostToDevice));
    HC(hipMalloc(&dst8, n_exp * 24)); HC(hipMemcpy(dst8, st8.data(), n_exp * 24, hipMemcpyHostToDevice));

    /* ---- f64 oracle: gate/up for both encodings ---- */
    auto oracle_glu = [&](unsigned enc, std::vector<double>& out) {
        out.assign((size_t)k * I, 0.0);
        for (unsigned s = 0; s < k; s++) {
            const unsigned e = slots[s];
            for (unsigned n = 0; n < I; n++) {
                double g = 0, u = 0;
                for (unsigned kk = 0; kk < H; kk++) {
                    const double a = bf(x[kk]);
                    double wg, wu;
                    if (enc == 2) {
                        wg = fp4d((Q[e*3+0][(size_t)n*H/2 + kk/2] >> ((kk&1)*4)) & 0xF)
                             * e8(QS[e*3+0][(size_t)n*H/32 + kk/32]);
                        wu = fp4d((Q[e*3+1][(size_t)n*H/2 + kk/2] >> ((kk&1)*4)) & 0xF)
                             * e8(QS[e*3+1][(size_t)n*H/32 + kk/32]);
                    } else {
                        wg = e4m3d(F[e*3+0][(size_t)n*H + kk]) * FS[e*3+0][(size_t)(n>>7)*KB + (kk>>7)];
                        wu = e4m3d(F[e*3+1][(size_t)n*H + kk]) * FS[e*3+1][(size_t)(n>>7)*KB + (kk>>7)];
                    }
                    g += a * wg; u += a * wu;
                }
                out[(size_t)s * I + n] = (g / (1.0 + exp(-g))) * u; /* silu gate * up */
            }
        }
        (void)NB;
    };

    int fail = 0;
    auto run_glu = [&](const char* name, bool mfma, unsigned enc, double tol) {
        std::vector<unsigned short> poison((size_t)k * I, 0x7FC0u); /* bf16 NaN */
        HC(hipMemcpy(dfu, poison.data(), poison.size() * 2, hipMemcpyHostToDevice));
        auto* kern = mfma ? t_glu_mfma : t_glu_fdot;
        hipLaunchKernelGGL(kern, dim3(64), dim3(PLOW_THREADS), 0, 0, dfu, dx, dtab,
                           enc == 2 ? dwt4 : dwt8, enc == 2 ? dst4 : dst8, k, I, H, n_exp, act, enc);
        HC(hipDeviceSynchronize());
        std::vector<unsigned short> got((size_t)k * I);
        HC(hipMemcpy(got.data(), dfu, got.size() * 2, hipMemcpyDeviceToHost));
        std::vector<double> want; oracle_glu(enc, want);
        double worst = 0, scale = 0;
        for (size_t i = 0; i < want.size(); i++) scale = fmax(scale, fabs(want[i]));
        for (size_t i = 0; i < want.size(); i++) worst = fmax(worst, fabs(bf(got[i]) - want[i]) / scale);
        const bool ok = worst < tol;
        printf("  %-34s enc=%u  worst rel(vs max) %.3e   %s\n", name, enc, worst, ok ? "PASS" : "FAIL");
        if (!ok) fail = 1;
    };
    auto run_refuse = [&](const char* name, bool mfma, unsigned enc) {
        std::vector<unsigned short> zero((size_t)k * I, 0);
        HC(hipMemcpy(dfu, zero.data(), zero.size() * 2, hipMemcpyHostToDevice));
        auto* kern = mfma ? t_glu_mfma : t_glu_fdot;
        hipLaunchKernelGGL(kern, dim3(64), dim3(PLOW_THREADS), 0, 0, dfu, dx, dtab, dwt8, dst8, k, I,
                           H, n_exp, act, enc);
        HC(hipDeviceSynchronize());
        std::vector<unsigned short> got((size_t)k * I);
        HC(hipMemcpy(got.data(), dfu, got.size() * 2, hipMemcpyDeviceToHost));
        size_t nan = 0;
        for (size_t i = 0; i < got.size(); i++) if (std::isnan((float)bf(got[i]))) nan++;
        const bool ok = nan == got.size();
        printf("  %-34s enc=%u  NaN %zu/%zu                 %s\n", name, enc, nan, got.size(),
               ok ? "PASS" : "FAIL");
        if (!ok) fail = 1;
    };

    printf("MoE DECODE expert bodies, K3 latent geometry H=%u I_moe=%u top-%u of %u:\n", H, I, k, n_exp);
    /* fdot2 is the reference implementation of the encoding field; MFMA is bit-CLOSE by design
     * (its weights round through bf16 after the fp8 block-scale fold, and its dot order is the
     * matrix core's), so it gets a looser tolerance than fdot2 — but a DROPPED encoding is an
     * O(1) relative error, three orders of magnitude outside either. */
    run_glu("fdot2  d_moe_expert_glu_fp8_blk", false, PLOW_MOE_ENC_FP8BLK, 5e-3);
    run_glu("fdot2  d_moe_expert_glu_fp8_blk", false, PLOW_MOE_ENC_MXFP4, 5e-3);
    run_glu("mfma   d_moe_group_glu_mfma", true, PLOW_MOE_ENC_FP8BLK, 2e-2);
    run_glu("mfma   d_moe_group_glu_mfma", true, PLOW_MOE_ENC_MXFP4, 5e-3);
    run_refuse("fdot2 refuses", false, PLOW_MOE_ENC_BF16);
    run_refuse("fdot2 refuses", false, 3u);
    run_refuse("mfma  refuses", true, PLOW_MOE_ENC_BF16);
    run_refuse("mfma  refuses", true, 3u);

    /* ---- DOWN at enc=2, fed the fdot2 gate/up output so the whole fp4 chain is covered ---- */
    {
        run_glu("fdot2  (feed for DOWN)", false, PLOW_MOE_ENC_MXFP4, 5e-3);
        std::vector<unsigned short> fu((size_t)k * I);
        HC(hipMemcpy(fu.data(), dfu, fu.size() * 2, hipMemcpyDeviceToHost));
        std::vector<float> poison((size_t)k * H, -12345.0f);
        HC(hipMemcpy(dpart, poison.data(), poison.size() * 4, hipMemcpyHostToDevice));
        hipLaunchKernelGGL(t_down, dim3(64), dim3(PLOW_THREADS), 0, 0, dpart, dfu, dtab, dwt4, dst4,
                           k, H, I, n_exp, PLOW_MOE_ENC_MXFP4);
        HC(hipDeviceSynchronize());
        std::vector<float> part((size_t)k * H);
        HC(hipMemcpy(part.data(), dpart, part.size() * 4, hipMemcpyDeviceToHost));
        double worst = 0, scale = 0; unsigned untouched = 0;
        for (unsigned s = 0; s < k; s++) {
            const unsigned e = slots[s];
            const double gate = *(float*)&tab[s * 8 + 4];
            for (unsigned h = 0; h < H; h++) {
                double acc = 0;
                for (unsigned kk = 0; kk < I; kk++)
                    acc += bf(fu[(size_t)s * I + kk])
                           * fp4d((Q[e*3+2][(size_t)h*I/2 + kk/2] >> ((kk&1)*4)) & 0xF)
                           * e8(QS[e*3+2][(size_t)h*I/32 + kk/32]);
                const double want = acc * gate, got = part[(size_t)s * H + h];
                if (got == -12345.0f) { untouched++; continue; }
                scale = fmax(scale, fabs(want));
                worst = fmax(worst, fabs(got - want));
            }
        }
        const double rel = worst / fmax(scale, 1e-30);
        const bool ok = untouched == 0 && rel < 2e-3;
        printf("  %-34s enc=%u  worst rel(vs max) %.3e   %s  (%u unwritten)\n",
               "fdot2  d_moe_expert_down_fp8_blk", PLOW_MOE_ENC_MXFP4, rel, ok ? "PASS" : "FAIL",
               untouched);
        if (!ok) fail = 1;
    }
    printf("%s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}
