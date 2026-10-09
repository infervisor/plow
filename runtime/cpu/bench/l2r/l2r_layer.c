/* l2r_layer.c <refdir> <steps>: one real Gemma-4 text decoder layer (BF16 weights) as a weight-stationary stage.
 *
 * Input: a ref_layer.py dump (BF16 weights, the decode step's inputs, the layer's KV cache, FP32 boundaries).
 * One worker per cpu of L2R_CPUS (default the 90 isolated inference cores). Every GEMV is output-row
 * partitioned; each worker keeps its row slices of all matrices in one node-local, THP-backed arena that it
 * touches first. KV rows are split by position across workers (node-local); attention is split-K with per-head
 * (max, sum, partial out) partials combined by element slices. Norms, RoPE and residuals are recomputed by every
 * worker from the shared vectors, so a decode step has 8 barriers:
 *   qkv | attention | combine | o | gate,up,gelu | down | ple gate | ple proj
 * Numerics as ref_layer.py's BF16 mode: BF16 weights, GEMV inputs rounded to BF16, BF16 KV, FP32 accumulation.
 * GEMV kernel: L2R_GEMV=avx (VDPBF16PS, default) or amx (TDPBF16PS, weights packed in 16-row VNNI tiles).
 * L2R_CTX_REPEAT=n tiles the dumped KV n times (long-context timing; numerics are then not checked).
 * Every step recomputes the same token (the new KV row overwrites itself), so weights stay hot and outputs repeat.
 * L2R_RESIDENT_KIB=n (AVX only): keep n KiB of each worker's slice L2-resident; the FFN rows beyond it stream from L3
 * with PREFETCHNTA (the E4B split: one E4B layer is ~2 MiB per worker).
 * L2R_LOCK=1 (with L2R_RESIDENT_KIB): lock the resident part in L2 through /dev/pseudo_lock (level 2, owner = the worker's
 *   cpu); the rest streams normally, confined by CAT to the unlocked ways. Prints the minimum held fraction.
 * L2R_NOBCAST=1: after step 0 every worker reads private snapshots of the shared vectors (identical values, since
 * the step repeats), which removes the activation all-gather from the timing; writes are unchanged.
 * Prints one JSON line: per-boundary error vs the FP32 reference (first and last step), step p50/p95/p99,
 * per-phase critical path (max over workers of phase compute) and barrier time. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <x86intrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAXW 192
#define NPH 8
typedef uint16_t bf16;
typedef struct { _Alignas(64) volatile uint64_t v; char pad[56]; } line_t;

static const char* DIR;
static int NW, STEPS, AMX, REP, NOBCAST;
static size_t RESKIB;
static int PLFD = -1;
struct pl_lock_req { uint64_t addr, len; int32_t cpu; uint32_t level, id, pad; };
struct pl_measure { uint32_t id, pad; uint64_t lines, l1_l2, l3, dram, p50, cal_l2, cal_l3, cal_dram; };
#define PL_IOC_LOCK _IOWR('P', 11, struct pl_lock_req)
#define PL_IOC_MEASURE _IOWR('P', 13, struct pl_measure)
static int cpus[MAXW];
static double tsc_ghz;

/* ---- model ---- */
static int H, NH, KVH, HD, I, PLE, WIN, CL0, CL; /* CL0: dumped cache rows, CL: rows attended before the new token */
static float EPS, SCALAR;
static float *x_in, *pli, *cosv, *sinv;
static bf16 *kc0, *vc0;
static const char* WN[9] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "per_layer_input_gate", "per_layer_projection"};
static bf16* Wfull[9];
static int Wn[9], Wk[9];
static float *w_in, *w_pa, *w_pf, *w_pff, *w_pn, *w_qn, *w_kn;

/* ---- shared vectors ---- */
static float *q, *k, *v, *o, *gate, *up, *down, *pg, *pp, *outv;
static bf16 *attn_b, *act_b, *pact_b;
static float *pm, *pl, *po; /* partials [NW][NH], [NW][NH], [NW][NH][HD] */
static float *chk_qn, *chk_h1, *chk_xn2, *chk_h2, *chk_attn, *chk_act, *chk_pact;

/* ---- per worker ---- */
typedef struct {
    int id, node;
    int r0[9], r1[9];
    int nres[9];          /* rows of the slice kept L2-resident; the rest stream with PREFETCHNTA */
    bf16* ws[9];
    bf16* w[9];
    int p0, p1;           /* KV positions [p0, p1) of 0..CL-1; the last worker also owns the new row */
    bf16 *kc, *vc;        /* [KVH][p1-p0 (+1)][HD] */
    int e0, e1;           /* attention-combine element slice of NH*HD */
    uint32_t lock_id; double held0, held1;
    uint64_t ph[NPH], wt[NPH], pro[NPH]; /* pro: the phase's redundant prologue (reads the shared vectors) */
    uint64_t* steps_t;
} worker_t;
static worker_t WK[MAXW];
static line_t dis[MAXW][8];
static pthread_barrier_t pbar;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double calib(void) { double t0 = now_s(); uint64_t c0 = __rdtsc(); while (now_s() - t0 < 0.2) ; return (__rdtsc() - c0) / (now_s() - t0) / 1e9; }

static inline float bf2f(bf16 b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline bf16 f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (bf16)(u >> 16); }

static void* load(const char* name, size_t* n_out, char* dt_out) {
    char path[512], line[512];
    snprintf(path, sizeof path, "%s/manifest.txt", DIR);
    FILE* m = fopen(path, "r");
    if (!m) { perror(path); exit(1); }
    while (fgets(line, sizeof line, m)) {
        char nm[256], dt[8]; int off;
        if (sscanf(line, "%255s %7s %n", nm, dt, &off) < 2 || strcmp(nm, name)) continue;
        size_t n = 1; long d; char* p = line + off;
        while (sscanf(p, "%ld%n", &d, &off) == 1) { n *= (size_t)d; p += off; }
        fclose(m);
        const size_t es = strcmp(dt, "bf16") ? 4 : 2;
        snprintf(path, sizeof path, "%s/%s.%s", DIR, name, dt);
        FILE* f = fopen(path, "rb");
        if (!f) { perror(path); exit(1); }
        void* b = aligned_alloc(64, (n * es + 63) / 64 * 64);
        if (fread(b, es, n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
        fclose(f);
        if (n_out) *n_out = n;
        if (dt_out) *dt_out = es == 2 ? 'b' : 'f';
        return b;
    }
    fprintf(stderr, "%s not in manifest\n", name); exit(1);
}

static float* loadbf_as_f(const char* name) {
    size_t n; char dt; void* p = load(name, &n, &dt);
    if (dt == 'f') return p;
    float* f = malloc(n * 4);
    for (size_t i = 0; i < n; i++) f[i] = bf2f(((bf16*)p)[i]);
    free(p);
    return f;
}

static int meta_int(const char* js, const char* key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\": ", key);
    const char* p = strstr(js, pat);
    if (!p) { fprintf(stderr, "meta: %s\n", key); exit(1); }
    p += strlen(pat);
    return strncmp(p, "null", 4) ? atoi(p) : 0;
}

static double meta_f(const char* js, const char* key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\": ", key);
    const char* p = strstr(js, pat);
    return p ? atof(p + strlen(pat)) : 0;
}

/* ---- barrier: dissemination, epoch-tagged flags, one line each ---- */
static inline void barrier(int id, uint64_t e) {
    for (int r = 0, d = 1; d < NW; r++, d <<= 1) {
        dis[(id + d) % NW][r].v = e;
        while (dis[id][r].v < e) _mm_pause();
    }
}

/* ---- GEMV kernels: y[r] = sum_k W[r][k] * x[k], W rows bf16 [n][K], x bf16 [K] ---- */
static void gemv_avx(const bf16* W, int n, int K, const bf16* x, float* y) {
    int r = 0;
    for (; r + 4 <= n; r += 4) {
        const bf16 *w0 = W + (size_t)r * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int c = 0; c < K; c += 32) {
            const __m512bh xv = (__m512bh)_mm512_loadu_si512(x + c);
            a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_loadu_si512(w0 + c), xv);
            a1 = _mm512_dpbf16_ps(a1, (__m512bh)_mm512_loadu_si512(w1 + c), xv);
            a2 = _mm512_dpbf16_ps(a2, (__m512bh)_mm512_loadu_si512(w2 + c), xv);
            a3 = _mm512_dpbf16_ps(a3, (__m512bh)_mm512_loadu_si512(w3 + c), xv);
        }
        y[r] = _mm512_reduce_add_ps(a0); y[r + 1] = _mm512_reduce_add_ps(a1);
        y[r + 2] = _mm512_reduce_add_ps(a2); y[r + 3] = _mm512_reduce_add_ps(a3);
    }
    for (; r < n; r++) {
        const bf16* w0 = W + (size_t)r * K;
        __m512 a0 = _mm512_setzero_ps();
        for (int c = 0; c < K; c += 32)
            a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_loadu_si512(w0 + c), (__m512bh)_mm512_loadu_si512(x + c));
        y[r] = _mm512_reduce_add_ps(a0);
    }
}

/* AMX: rows packed per 16-row group g, per 32-wide K chunk c, as one 1 KiB B tile [16 k-pairs][16 rows][2].
 * A = x chunk in tile row 0 (1 row); C[0][0..15] = the group's 16 outputs. n must be a multiple of 16. */
typedef struct { uint8_t palette, start; uint8_t res[14]; uint16_t colsb[16]; uint8_t rows[16]; } tilecfg_t;
static void pack_amx(const bf16* src, int n, int K, bf16* dst) {
    for (int g = 0; g < n / 16; g++)
        for (int c = 0; c < K / 32; c++) {
            bf16* t = dst + ((size_t)g * (K / 32) + c) * 512;
            for (int kp = 0; kp < 16; kp++)
                for (int r = 0; r < 16; r++)
                    for (int e = 0; e < 2; e++)
                        t[kp * 32 + r * 2 + e] = src[(size_t)(g * 16 + r) * K + c * 32 + kp * 2 + e];
        }
}
static void gemv_amx(const bf16* Wp, int n, int K, const bf16* x, float* y) {
    _Alignas(64) float cbuf[16 * 16];
    const int nc = K / 32;
    int g = 0;
    for (; g + 4 <= n / 16; g += 4) {
        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
        const bf16* b = Wp + (size_t)g * nc * 512;
        for (int c = 0; c < nc; c++) {
            _tile_loadd(4, x + c * 32, 64);
            _tile_loadd(5, b + (size_t)c * 512, 64); _tile_dpbf16ps(0, 4, 5);
            _tile_loadd(6, b + ((size_t)nc + c) * 512, 64); _tile_dpbf16ps(1, 4, 6);
            _tile_loadd(7, b + ((size_t)2 * nc + c) * 512, 64); _tile_dpbf16ps(2, 4, 7);
            _tile_loadd(5, b + ((size_t)3 * nc + c) * 512, 64); _tile_dpbf16ps(3, 4, 5);
        }
        _tile_stored(0, cbuf, 64); memcpy(y + (g + 0) * 16, cbuf, 64);
        _tile_stored(1, cbuf, 64); memcpy(y + (g + 1) * 16, cbuf, 64);
        _tile_stored(2, cbuf, 64); memcpy(y + (g + 2) * 16, cbuf, 64);
        _tile_stored(3, cbuf, 64); memcpy(y + (g + 3) * 16, cbuf, 64);
    }
    for (; g < n / 16; g++) {
        _tile_zero(0);
        const bf16* b = Wp + (size_t)g * nc * 512;
        for (int c = 0; c < nc; c++) { _tile_loadd(4, x + c * 32, 64); _tile_loadd(5, b + (size_t)c * 512, 64); _tile_dpbf16ps(0, 4, 5); }
        _tile_stored(0, cbuf, 64); memcpy(y + g * 16, cbuf, 64);
    }
}
/* Streamed rows: same math as gemv_avx; the next 4-row block is prefetched non-temporally (into L1, not L2) while the
 * current one is computed, so streaming rows from L3 does not evict the L2-resident part of the slice. */
static void gemv_nta(const bf16* W, int n, int K, const bf16* x, float* y) {
    int r = 0;
    for (; r + 4 <= n; r += 4) {
        const bf16 *w0 = W + (size_t)r * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K, *nx = w0 + (size_t)4 * K;
        const int pf = r + 8 <= n;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int c = 0; c < K; c += 32) {
            if (pf) {
                _mm_prefetch((const char*)(nx + c), _MM_HINT_NTA); _mm_prefetch((const char*)(nx + K + c), _MM_HINT_NTA);
                _mm_prefetch((const char*)(nx + 2 * K + c), _MM_HINT_NTA); _mm_prefetch((const char*)(nx + 3 * K + c), _MM_HINT_NTA);
            }
            const __m512bh xv = (__m512bh)_mm512_loadu_si512(x + c);
            a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_loadu_si512(w0 + c), xv);
            a1 = _mm512_dpbf16_ps(a1, (__m512bh)_mm512_loadu_si512(w1 + c), xv);
            a2 = _mm512_dpbf16_ps(a2, (__m512bh)_mm512_loadu_si512(w2 + c), xv);
            a3 = _mm512_dpbf16_ps(a3, (__m512bh)_mm512_loadu_si512(w3 + c), xv);
        }
        y[r] = _mm512_reduce_add_ps(a0); y[r + 1] = _mm512_reduce_add_ps(a1);
        y[r + 2] = _mm512_reduce_add_ps(a2); y[r + 3] = _mm512_reduce_add_ps(a3);
    }
    if (r < n) gemv_avx(W + (size_t)r * K, n - r, K, x, y + r);
}

static inline void gemv(const bf16* W, int n, int K, const bf16* x, float* y) {
    if (n <= 0) return;
    if (AMX) gemv_amx(W, n, K, x, y); else gemv_avx(W, n, K, x, y);
}

/* ---- small ops ---- */
/* n is a multiple of 16 (hidden, head dim) */
static void rms(const float* x, const float* w, int n, float* y) {
    __m512 a = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) { const __m512 t = _mm512_loadu_ps(x + i); a = _mm512_fmadd_ps(t, t, a); }
    const __m512 r = _mm512_set1_ps(powf(_mm512_reduce_add_ps(a) / n + EPS, -0.5f));
    for (int i = 0; i < n; i += 16) {
        const __m512 t = _mm512_mul_ps(_mm512_loadu_ps(x + i), r);
        _mm512_storeu_ps(y + i, w ? _mm512_mul_ps(t, _mm512_loadu_ps(w + i)) : t);
    }
}
/* round-to-nearest-even, n a multiple of 32 */
static void tobf(const float* x, int n, bf16* y) {
    for (int i = 0; i < n; i += 32)
        _mm512_storeu_si512(y + i, (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(x + i)));
}
static void rope(float* x) {
    const int h = HD / 2;
    float t[1024];
    for (int i = 0; i < HD; i++) t[i] = i < h ? -x[i + h] : x[i - h];
    for (int i = 0; i < HD; i++) x[i] = x[i] * cosv[i] + t[i] * sinv[i];
}
static inline float gelu(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

/* exp for x <= 0 (softmax): x = n ln2 + r, |r| <= ln2/2, degree-7 Taylor (< 1 ulp), 2^n by scalef */
static inline __m512 exp512(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 y = _mm512_set1_ps(1.f / 5040);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f / 720));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f / 120));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f / 24));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f / 6));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(0.5f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.f));
    return _mm512_scalef_ps(y, n);
}

/* Attention partials of one KV head for its G query heads over np rows: scores in sc[p][G]; per head the max m, the
 * sum l of exp(s - m) and the unnormalised output o = sum exp(s - m) * V. Each K / V row is widened once for all G
 * heads; G is a compile-time constant at every call so the accumulators stay in registers. */
static inline __attribute__((always_inline)) void attn_group(const int G, const float* q, const bf16* K, const bf16* V,
                                                              int np, float* sc, float* m, float* l, float* o) {
    for (int j = 0; j < G; j++) m[j] = -INFINITY, l[j] = 0;
    for (int p = 0; p < np; p++) {
        __m512 a[G];
        for (int j = 0; j < G; j++) a[j] = _mm512_setzero_ps();
        const bf16* kr = K + (size_t)p * HD;
        for (int d = 0; d < HD; d += 16) {
            const __m512 kf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(kr + d))), 16));
            for (int j = 0; j < G; j++) a[j] = _mm512_fmadd_ps(kf, _mm512_loadu_ps(q + j * HD + d), a[j]);
        }
        for (int j = 0; j < G; j++) {
            const float x = _mm512_reduce_add_ps(a[j]);
            sc[(size_t)p * G + j] = x;
            if (x > m[j]) m[j] = x;
        }
    }
    {
        float mp[16];
        for (int i = 0; i < 16; i++) mp[i] = m[i % G];
        const __m512 mv = _mm512_loadu_ps(mp);
        __m512 lv = _mm512_setzero_ps();
        const int n = np * G;
        for (int i = 0; i < n; i += 16) {
            const __mmask16 k = n - i >= 16 ? 0xffff : (__mmask16)((1u << (n - i)) - 1);
            const __m512 w = exp512(_mm512_sub_ps(_mm512_maskz_loadu_ps(k, sc + i), mv));
            _mm512_mask_storeu_ps(sc + i, k, w);
            lv = _mm512_add_ps(lv, _mm512_maskz_mov_ps(k, w));
        }
        float lp[16];
        _mm512_storeu_ps(lp, lv);
        for (int i = 0; i < 16; i++) l[i % G] += lp[i];
    }
    for (int d = 0; d < HD; d += 16) {
        __m512 acc[G];
        for (int j = 0; j < G; j++) acc[j] = _mm512_setzero_ps();
        for (int p = 0; p < np; p++) {
            const __m512 vf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(V + (size_t)p * HD + d))), 16));
            for (int j = 0; j < G; j++) acc[j] = _mm512_fmadd_ps(_mm512_set1_ps(sc[(size_t)p * G + j]), vf, acc[j]);
        }
        for (int j = 0; j < G; j++) _mm512_storeu_ps(o + (size_t)j * HD + d, acc[j]);
    }
    if (!np) for (int j = 0; j < G; j++) m[j] = -INFINITY;
}

/* ---- the decode step ---- */
static void split(int n, int unit, int w, int* a, int* b) {
    const int u = n / unit;
    *a = (int)((long)u * w / NW) * unit;
    *b = (int)((long)u * (w + 1) / NW) * unit;
}

static void* run(void* arg) {
    worker_t* me = arg;
    const int id = me->id;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus[id], &s); sched_setaffinity(0, sizeof s, &s);
    if (AMX) {
        if (syscall(SYS_arch_prctl, 0x1023, 18)) { perror("amx"); exit(1); }
        tilecfg_t c; memset(&c, 0, sizeof c); c.palette = 1;
        for (int t = 0; t < 4; t++) { c.colsb[t] = 64; c.rows[t] = 1; }
        c.colsb[4] = 64; c.rows[4] = 1;
        for (int t = 5; t < 8; t++) { c.colsb[t] = 64; c.rows[t] = 16; }
        _tile_loadconfig(&c);
    }
    /* weights: one arena, 2 MiB aligned, THP, first touch here. With L2R_RESIDENT_KIB, FFN rows beyond the budget
     * (gate/up paired, down proportional) go after all resident rows and stream through gemv_nta. */
    size_t bytes = 0;
    for (int m = 0; m < 9; m++) {
        split(Wn[m], AMX ? 16 : 4, id, &me->r0[m], &me->r1[m]);
        me->nres[m] = me->r1[m] - me->r0[m];
        bytes += (size_t)(me->r1[m] - me->r0[m]) * Wk[m] * 2;
    }
    if (RESKIB && !AMX && bytes > RESKIB << 10) {
        const size_t ffn = (size_t)(me->nres[4] + me->nres[5]) * H * 2 + (size_t)me->nres[6] * I * 2;
        const double f = (double)(bytes - (RESKIB << 10)) / ffn;
        const int sgu = (int)(me->nres[4] * f + 3) / 4 * 4, sd = (int)(me->nres[6] * f + 3) / 4 * 4;
        me->nres[4] -= sgu < me->nres[4] ? sgu : me->nres[4]; me->nres[5] = me->nres[4];
        me->nres[6] -= sd < me->nres[6] ? sd : me->nres[6];
    }
    const size_t HUGE = 2u << 20, al = (bytes + HUGE - 1) / HUGE * HUGE;
    uint8_t* raw = mmap(NULL, al + HUGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint8_t* arena = (uint8_t*)(((uintptr_t)raw + HUGE - 1) & ~(HUGE - 1));
    madvise(arena, al, MADV_HUGEPAGE);
    size_t off = 0;
    for (int m = 0; m < 9; m++) {
        const int n = me->nres[m];
        me->w[m] = (bf16*)(arena + off);
        if (AMX) pack_amx(Wfull[m] + (size_t)me->r0[m] * Wk[m], n, Wk[m], me->w[m]);
        else memcpy(me->w[m], Wfull[m] + (size_t)me->r0[m] * Wk[m], (size_t)n * Wk[m] * 2);
        off += (size_t)n * Wk[m] * 2;
    }
    const size_t resident = off;
    for (int m = 0; m < 9; m++) {
        const int n = me->r1[m] - me->r0[m] - me->nres[m];
        me->ws[m] = (bf16*)(arena + off);
        memcpy(me->ws[m], Wfull[m] + (size_t)(me->r0[m] + me->nres[m]) * Wk[m], (size_t)n * Wk[m] * 2);
        off += (size_t)n * Wk[m] * 2;
    }
    if (PLFD >= 0) {
        struct pl_lock_req rq = {(uintptr_t)arena, (resident + 4095) / 4096 * 4096, cpus[id], 2, 0, 0};
        if (ioctl(PLFD, PL_IOC_LOCK, &rq)) { fprintf(stderr, "cpu %d LOCK %zu B: %m\n", cpus[id], (size_t)rq.len); exit(1); }
        me->lock_id = rq.id;
        struct pl_measure ms = {rq.id, 0};
        ioctl(PLFD, PL_IOC_MEASURE, &ms); me->held0 = ms.lines ? (double)ms.l1_l2 / ms.lines : -1;
    }
    /* KV slice, node-local */
    me->p0 = (int)((long)CL * id / NW); me->p1 = (int)((long)CL * (id + 1) / NW);
    const int last = id == NW - 1, np = me->p1 - me->p0 + last;
    me->kc = aligned_alloc(64, (size_t)KVH * (np + 1) * HD * 2);
    me->vc = aligned_alloc(64, (size_t)KVH * (np + 1) * HD * 2);
    for (int h = 0; h < KVH; h++)
        for (int p = me->p0; p < me->p1; p++) {
            memcpy(me->kc + ((size_t)h * np + p - me->p0) * HD, kc0 + ((size_t)h * CL0 + p % CL0) * HD, HD * 2);
            memcpy(me->vc + ((size_t)h * np + p - me->p0) * HD, vc0 + ((size_t)h * CL0 + p % CL0) * HD, HD * 2);
        }
    split(NH * HD, 1, id, &me->e0, &me->e1);

    float* xn = aligned_alloc(64, (size_t)H * 4); bf16* xb = aligned_alloc(64, (size_t)H * 2 + 64);
    float* qn = aligned_alloc(64, (size_t)NH * HD * 4);
    float* h1 = aligned_alloc(64, (size_t)H * 4); float* h2 = aligned_alloc(64, (size_t)H * 4);
    float* tmp = aligned_alloc(64, (size_t)(H > I ? H : I) * 4);
    bf16* x2b = aligned_alloc(64, (size_t)H * 2 + 64); bf16* h2b = aligned_alloc(64, (size_t)H * 2 + 64);
    float* sc = aligned_alloc(64, (size_t)(np + 1) * 4 * NH);
    float fw[MAXW];
    /* read side of every shared vector; L2R_NOBCAST=1 points them at private snapshots after step 0 */
    const float *rq = q, *rk = k, *rv = v, *rpm = pm, *rpl = pl, *rpo = po, *ro = o, *rdown = down;
    const bf16 *rattn = attn_b, *ract = act_b, *rpact = pact_b;
    const int g = NH / KVH;
    me->steps_t = calloc(STEPS, 8);

    pthread_barrier_wait(&pbar);
    uint64_t e = 0;
    for (int st = 0; st < STEPS; st++) {
        uint64_t t = __rdtsc(), t0 = t, u;
#define PHASE(i) do { u = __rdtsc(); me->ph[i] += u - t; barrier(id, ++e); t = __rdtsc(); me->wt[i] += t - u; } while (0)
        /* 0: input norm (redundant), qkv rows */
        rms(x_in, w_in, H, xn); tobf(xn, H, xb);
        gemv(me->w[0], me->r1[0] - me->r0[0], H, xb, q + me->r0[0]);
        gemv(me->w[1], me->r1[1] - me->r0[1], H, xb, k + me->r0[1]);
        gemv(me->w[2], me->r1[2] - me->r0[2], H, xb, v + me->r0[2]);
        PHASE(0);
        /* 1: q/k norm + rope, v norm (redundant); attention over this worker's KV positions */
        for (int h = 0; h < NH; h++) { rms(rq + h * HD, w_qn, HD, qn + h * HD); rope(qn + h * HD); }
        me->pro[1] += __rdtsc() - t;
        if (last) {
            float kn[1024], vn[1024];
            for (int h = 0; h < KVH; h++) {
                rms(rk + h * HD, w_kn, HD, kn); rope(kn); rms(rv + h * HD, NULL, HD, vn);
                tobf(kn, HD, me->kc + ((size_t)h * np + np - 1) * HD);
                tobf(vn, HD, me->vc + ((size_t)h * np + np - 1) * HD);
            }
        }
        for (int kh = 0; kh < KVH; kh++) {
            float* m = pm + id * NH + kh * g; float* l = pl + id * NH + kh * g;
            float* oh = po + ((size_t)id * NH + kh * g) * HD;
            const bf16 *kr = me->kc + (size_t)kh * np * HD, *vr = me->vc + (size_t)kh * np * HD;
            if (g == 8) attn_group(8, qn + kh * g * HD, kr, vr, np, sc, m, l, oh);
            else if (g == 4) attn_group(4, qn + kh * g * HD, kr, vr, np, sc, m, l, oh);
            else if (g == 2) attn_group(2, qn + kh * g * HD, kr, vr, np, sc, m, l, oh);
            else attn_group(1, qn + kh * g * HD, kr, vr, np, sc, m, l, oh);
        }
        PHASE(1);
        /* 2: combine partials for this worker's slice of NH*HD: one scale per (worker, head) */
        for (int h = me->e0 / HD; me->e1 > me->e0 && h <= (me->e1 - 1) / HD; h++) {
            float M = -INFINITY, den = 0;
            for (int w = 0; w < NW; w++) if (rpm[w * NH + h] > M) M = rpm[w * NH + h];
            for (int w = 0; w < NW; w++) { fw[w] = rpm[w * NH + h] == -INFINITY ? 0.f : expf(rpm[w * NH + h] - M); den += fw[w] * rpl[w * NH + h]; }
            const int a = h * HD > me->e0 ? h * HD : me->e0, b = (h + 1) * HD < me->e1 ? (h + 1) * HD : me->e1;
            for (int i = a; i < b; i++) {
                const int d = i - h * HD;
                float num = 0;
                for (int w = 0; w < NW; w++) num += fw[w] * rpo[((size_t)w * NH + h) * HD + d];
                attn_b[i] = f2bf(num / den);
                if (st == 0) chk_attn[i] = num / den;
            }
        }
        PHASE(2);
        /* 3: o rows */
        gemv(me->w[3], me->r1[3] - me->r0[3], NH * HD, rattn, o + me->r0[3]);
        PHASE(3);
        /* 4: residual + norms (redundant); gate/up rows, gelu * up */
        rms(ro, w_pa, H, tmp); for (int i = 0; i < H; i++) h1[i] = x_in[i] + tmp[i];
        rms(h1, w_pf, H, tmp); tobf(tmp, H, x2b);
        me->pro[4] += __rdtsc() - t;
        if (st == 0 && id == 0) { memcpy(chk_h1, h1, H * 4); memcpy(chk_xn2, tmp, H * 4); memcpy(chk_qn, qn, NH * HD * 4); }
        gemv(me->w[4], me->nres[4], H, x2b, gate + me->r0[4]);
        gemv(me->w[5], me->nres[5], H, x2b, up + me->r0[5]);
        (PLFD >= 0 ? gemv : gemv_nta)(me->ws[4], me->r1[4] - me->r0[4] - me->nres[4], H, x2b, gate + me->r0[4] + me->nres[4]);
        (PLFD >= 0 ? gemv : gemv_nta)(me->ws[5], me->r1[5] - me->r0[5] - me->nres[5], H, x2b, up + me->r0[5] + me->nres[5]);
        for (int r = me->r0[4]; r < me->r1[4]; r++) {
            const float a = gelu(gate[r]) * up[r];
            act_b[r] = f2bf(a);
            if (st == 0) chk_act[r] = a;
        }
        PHASE(4);
        /* 5: down rows */
        gemv(me->w[6], me->nres[6], I, ract, down + me->r0[6]);
        (PLFD >= 0 ? gemv : gemv_nta)(me->ws[6], me->r1[6] - me->r0[6] - me->nres[6], I, ract, down + me->r0[6] + me->nres[6]);
        PHASE(5);
        /* 6: residual (redundant); ple gate rows, gelu * per-layer input */
        rms(rdown, w_pff, H, tmp); for (int i = 0; i < H; i++) h2[i] = h1[i] + tmp[i];
        tobf(h2, H, h2b);
        me->pro[6] += __rdtsc() - t;
        if (st == 0 && id == 0) memcpy(chk_h2, h2, H * 4);
        gemv(me->w[7], me->r1[7] - me->r0[7], H, h2b, pg + me->r0[7]);
        for (int r = me->r0[7]; r < me->r1[7]; r++) {
            const float a = gelu(pg[r]) * pli[r];
            pact_b[r] = f2bf(a);
            if (st == 0) chk_pact[r] = a;
        }
        PHASE(6);
        /* 7: ple projection rows */
        gemv(me->w[8], me->r1[8] - me->r0[8], PLE, rpact, pp + me->r0[8]);
        PHASE(7);
        if (id == 0) {
            rms(pp, w_pn, H, tmp);
            for (int i = 0; i < H; i++) outv[(size_t)(st == 0 ? 0 : 1) * H + i] = (h2[i] + tmp[i]) * SCALAR;
        }
        me->steps_t[st] = __rdtsc() - t0;
        if (st == 0 && NOBCAST) {
#define SNAP(dst, src, bytes) do { void* c_ = aligned_alloc(64, ((bytes) + 63) / 64 * 64); memcpy(c_, src, bytes); dst = c_; } while (0)
            SNAP(rq, q, (size_t)NH * HD * 4); SNAP(rk, k, (size_t)KVH * HD * 4); SNAP(rv, v, (size_t)KVH * HD * 4);
            SNAP(rpm, pm, (size_t)NW * NH * 4); SNAP(rpl, pl, (size_t)NW * NH * 4); SNAP(rpo, po, (size_t)NW * NH * HD * 4);
            SNAP(ro, o, (size_t)H * 4); SNAP(rdown, down, (size_t)H * 4);
            SNAP(rattn, attn_b, (size_t)NH * HD * 2); SNAP(ract, act_b, (size_t)I * 2); SNAP(rpact, pact_b, (size_t)PLE * 2);
        }
    }
    if (PLFD >= 0) { struct pl_measure ms = {me->lock_id, 0}; ioctl(PLFD, PL_IOC_MEASURE, &ms); me->held1 = ms.lines ? (double)ms.l1_l2 / ms.lines : -1; }
    if (AMX) _tile_release();
    return NULL;
}

/* ---- checks ---- */
typedef struct { double rel_rms, cos, max_abs; } err_t;
static err_t err(const float* a, const float* r, int n) {
    double d2 = 0, r2 = 0, a2 = 0, ar = 0, mx = 0;
    for (int i = 0; i < n; i++) {
        const double d = (double)a[i] - r[i];
        d2 += d * d; r2 += (double)r[i] * r[i]; a2 += (double)a[i] * a[i]; ar += (double)a[i] * r[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    return (err_t){sqrt(d2 / (r2 ? r2 : 1)), ar / sqrt(a2 * r2 + 1e-300), mx};
}
static int cmpu64(const void* a, const void* b) { uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: l2r_layer <refdir> <steps>\n"); return 1; }
    DIR = argv[1]; STEPS = atoi(argv[2]);
    const char* g = getenv("L2R_GEMV"); AMX = g && !strcmp(g, "amx");
    NOBCAST = getenv("L2R_NOBCAST") && atoi(getenv("L2R_NOBCAST"));
    RESKIB = getenv("L2R_RESIDENT_KIB") ? strtoull(getenv("L2R_RESIDENT_KIB"), 0, 10) : 0;
    if (getenv("L2R_LOCK") && atoi(getenv("L2R_LOCK"))) {
        if (!RESKIB || AMX) { fprintf(stderr, "L2R_LOCK needs L2R_RESIDENT_KIB and the AVX GEMV\n"); return 1; }
        if ((PLFD = open("/dev/pseudo_lock", O_RDWR)) < 0) { perror("/dev/pseudo_lock"); return 1; }
    }
    REP = getenv("L2R_CTX_REPEAT") ? atoi(getenv("L2R_CTX_REPEAT")) : 1;
    const char* cl = getenv("L2R_CPUS") ? getenv("L2R_CPUS") : "2-31,34-63,66-95";
    for (const char* p = cl; *p;) {
        int a = (int)strtol(p, (char**)&p, 10), b = a;
        if (*p == '-') b = (int)strtol(p + 1, (char**)&p, 10);
        for (int c = a; c <= b && NW < MAXW; c++) cpus[NW++] = c;
        if (*p == ',') p++;
    }
    char path[512]; snprintf(path, sizeof path, "%s/meta.json", DIR);
    FILE* f = fopen(path, "r"); if (!f) { perror(path); return 1; }
    static char js[1 << 20]; js[fread(js, 1, sizeof js - 1, f)] = 0; fclose(f);
    H = meta_int(js, "hidden"); NH = meta_int(js, "heads"); KVH = meta_int(js, "kv_heads"); HD = meta_int(js, "head_dim");
    I = meta_int(js, "inter"); PLE = meta_int(js, "ple"); WIN = meta_int(js, "window"); CL0 = meta_int(js, "cache_len");
    EPS = (float)meta_f(js, "eps");
    CL = CL0 * REP;
    for (int m = 0; m < 9; m++) {
        char nm[128]; snprintf(nm, sizeof nm, "w.%s.weight", WN[m]);
        size_t n; Wfull[m] = load(nm, &n, NULL);
    }
    Wn[0] = NH * HD; Wk[0] = H; Wn[1] = Wn[2] = KVH * HD; Wk[1] = Wk[2] = H; Wn[3] = H; Wk[3] = NH * HD;
    Wn[4] = Wn[5] = I; Wk[4] = Wk[5] = H; Wn[6] = H; Wk[6] = I; Wn[7] = PLE; Wk[7] = H; Wn[8] = H; Wk[8] = PLE;
    w_in = loadbf_as_f("w.input_layernorm.weight"); w_pa = loadbf_as_f("w.post_attention_layernorm.weight");
    w_pf = loadbf_as_f("w.pre_feedforward_layernorm.weight"); w_pff = loadbf_as_f("w.post_feedforward_layernorm.weight");
    w_pn = loadbf_as_f("w.post_per_layer_input_norm.weight");
    w_qn = loadbf_as_f("w.self_attn.q_norm.weight"); w_kn = loadbf_as_f("w.self_attn.k_norm.weight");
    float* ls = load("layer_scalar", NULL, NULL); SCALAR = ls[0];
    x_in = load("x_in", NULL, NULL); pli = load("per_layer_input", NULL, NULL);
    cosv = load("cos", NULL, NULL); sinv = load("sin", NULL, NULL);
    kc0 = load("kcache", NULL, NULL); vc0 = load("vcache", NULL, NULL);
    const int A = NH * HD;
#define ZA(p, n) p = aligned_alloc(64, ((size_t)(n) * 4 + 63) / 64 * 64), memset(p, 0, (size_t)(n) * 4)
    ZA(q, A); ZA(k, KVH * HD); ZA(v, KVH * HD); ZA(o, H); ZA(gate, I); ZA(up, I); ZA(down, H); ZA(pg, PLE); ZA(pp, H); ZA(outv, 2 * H);
    ZA(pm, NW * NH); ZA(pl, NW * NH); ZA(po, (size_t)NW * NH * HD);
    ZA(chk_qn, A); ZA(chk_h1, H); ZA(chk_xn2, H); ZA(chk_h2, H); ZA(chk_attn, A); ZA(chk_act, I); ZA(chk_pact, PLE);
    attn_b = aligned_alloc(64, A * 2 + 64); act_b = aligned_alloc(64, I * 2 + 64); pact_b = aligned_alloc(64, PLE * 2 + 64);
    tsc_ghz = calib();
    pthread_barrier_init(&pbar, NULL, NW);
    pthread_t th[MAXW];
    for (int i = 0; i < NW; i++) { WK[i].id = i; pthread_create(&th[i], NULL, run, &WK[i]); }
    for (int i = 0; i < NW; i++) pthread_join(th[i], NULL);

    /* numerics (step 0 boundaries; out of step 0 and of the last step) */
    struct { const char* n; float* a; int len; } B[] = {
        {"q", q, A}, {"k", k, KVH * HD}, {"v", v, KVH * HD}, {"qn", chk_qn, A}, {"attn", chk_attn, A}, {"o", o, H},
        {"h1", chk_h1, H}, {"xn2", chk_xn2, H}, {"gate", gate, I}, {"up", up, I}, {"act", chk_act, I},
        {"down", down, H}, {"h2", chk_h2, H}, {"pg", pg, PLE}, {"pact", chk_pact, PLE}, {"pp", pp, H},
        {"out", outv, H}, {"out_last", outv + H, H}};
    printf("{\"ref\":\"%s\",\"gemv\":\"%s\",\"nobcast\":%d,\"resident_kib\":%zu,\"workers\":%d,\"steps\":%d,\"ctx_rows\":%d,\"tsc_ghz\":%.3f,\"err\":{", DIR, AMX ? "amx" : "avx", NOBCAST, RESKIB, NW, STEPS, CL + 1, tsc_ghz);
    for (size_t b = 0; b < sizeof B / sizeof B[0]; b++) {
        char nm[64]; snprintf(nm, sizeof nm, "ref.%s", !strcmp(B[b].n, "out_last") ? "out" : B[b].n);
        float* r = REP == 1 ? load(nm, NULL, NULL) : NULL;
        if (r) {
            err_t e = err(B[b].a, r, B[b].len);
            printf("%s\"%s\":[%.3e,%.8f,%.3e]", b ? "," : "", B[b].n, e.rel_rms, e.cos, e.max_abs);
            free(r);
        } else printf("%s\"%s\":null", b ? "," : "", B[b].n);
    }
    /* timing: step 0..skip warm-up excluded */
    const int skip = STEPS / 10;
    uint64_t* st = malloc(STEPS * 8);
    int n = 0;
    for (int s = skip; s < STEPS; s++) { uint64_t m = 0; for (int w = 0; w < NW; w++) if (WK[w].steps_t[s] > m) m = WK[w].steps_t[s]; st[n++] = m; }
    qsort(st, n, 8, cmpu64);
    const double us = 1e-3 / tsc_ghz;
    double mean = 0; for (int i = 0; i < n; i++) mean += st[i]; mean /= n;
    printf("},\"step_us\":{\"mean\":%.2f,\"p50\":%.2f,\"p95\":%.2f,\"p99\":%.2f,\"max\":%.2f},\"phase_us\":[", mean * us,
           st[n / 2] * us, st[n * 95 / 100] * us, st[n * 99 / 100] * us, st[n - 1] * us);
    for (int p = 0; p < NPH; p++) {
        double mx = 0, avg = 0, wmx = 0, pr = 0;
        for (int w = 0; w < NW; w++) { double c = (double)WK[w].ph[p] / STEPS; avg += c; if (c > mx) mx = c; wmx += (double)WK[w].wt[p] / STEPS; pr += (double)WK[w].pro[p] / STEPS; }
        printf("%s{\"compute_max\":%.3f,\"compute_mean\":%.3f,\"prologue_mean\":%.3f,\"barrier_mean\":%.3f}", p ? "," : "", mx * us, avg / NW * us, pr / NW * us, wmx / NW * us);
    }
    size_t wb = 0; for (int m = 0; m < 9; m++) wb += (size_t)Wn[m] * Wk[m] * 2;
    double h0 = 1, h1 = 1;
    for (int w = 0; w < NW; w++) { if (WK[w].held0 < h0) h0 = WK[w].held0; if (WK[w].held1 < h1) h1 = WK[w].held1; }
    printf("]");
    if (PLFD >= 0) printf(",\"lock\":{\"held_l2_before_min\":%.4f,\"held_l2_after_min\":%.4f}", h0, h1);
    printf(",\"weight_bytes\":%zu,\"weight_bytes_per_worker\":%.0f}\n", wb, (double)wb / NW);
    return 0;
}
