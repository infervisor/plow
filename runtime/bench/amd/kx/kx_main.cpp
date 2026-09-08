/* kx driver — loads one experiment's code objects, times its arms, gates them, prints the result.
 *
 * Generic: it knows nothing about any experiment beyond the `kx_experiment()` descriptor it is
 * linked against. Every discipline the benches in this directory had to re-earn by hand lives
 * here instead — A/A control, palindromic interleave, the device-side correctness gate, the two
 * labelled geometries, and the refusals.
 *
 * Built and driven by scripts/kx.sh; see runtime/bench/amd/kx/README.md.
 */
#define KX_HOST 1
#include "kx.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            fprintf(stderr, "kx: HIP FAIL %s @%d: %s\n", #x, __LINE__, hipGetErrorString(e_));     \
            exit(70);                                                                              \
        }                                                                                          \
    } while (0)

enum { GEOM_1WG = 0, GEOM_MODEL = 1 };
static const char* geom_name[] = {"1wg", "model"};

struct Row {
    int geom, shape;
    std::vector<double> us; /* per arm, per rep */
    double aa;
    bool aa_ok;
};

struct Chk {
    int arm, shape;
    double maxe, rms;
    double exact;
    unsigned n;
    int bad; /* non-finite outputs in the reference */
    int verdict; /* 0 ok, 1 wrong */
};

/* Reps per launch: a fixed count when the experiment names one, otherwise as many fresh slabs as
 * the arena holds -- so the number is a STREAM and not an L2 hit. */
static unsigned kx_reps(const kx_shape& S, size_t arena) {
    if (S.reps) return S.reps;
    if (S.slab_bytes <= 0) return 1;
    return (unsigned)std::max(1.0, std::min<double>((double)arena / S.slab_bytes, 256.0));
}

static double median_us(hipFunction_t f, unsigned grid, unsigned threads, void** a, int it) {
    for (int i = 0; i < 3; i++)
        CK(hipModuleLaunchKernel(f, grid, 1, 1, threads, 1, 1, 0, 0, a, nullptr));
    CK(hipDeviceSynchronize());
    std::vector<double> v;
    v.reserve(it);
    hipEvent_t s, e;
    CK(hipEventCreate(&s));
    CK(hipEventCreate(&e));
    for (int i = 0; i < it; i++) {
        CK(hipEventRecord(s, 0));
        CK(hipModuleLaunchKernel(f, grid, 1, 1, threads, 1, 1, 0, 0, a, nullptr));
        CK(hipEventRecord(e, 0));
        CK(hipEventSynchronize(e));
        float ms = 0;
        CK(hipEventElapsedTime(&ms, s, e));
        v.push_back((double)ms * 1000.0);
    }
    CK(hipEventDestroy(s));
    CK(hipEventDestroy(e));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* THE ENVIRONMENT REFUSALS. Each of these has cost this tree a whole result at least once, so
 * they are hard failures and not warnings. */
static void gate_environment(void) {
    const char* hv = getenv("HIP_VISIBLE_DEVICES");
    const char* cv = getenv("CUDA_VISIBLE_DEVICES");
    if (hv && *hv) {
        fprintf(stderr,
                "kx: REFUSING — HIP_VISIBLE_DEVICES=%s is set.\n"
                "    It COMPOSES with ROCR_VISIBLE_DEVICES; setting both makes a correctly\n"
                "    targeted card report \"no ROCm-capable device is detected\". Pin with\n"
                "    ROCR_VISIBLE_DEVICES only.\n",
                hv);
        exit(64);
    }
    if (cv && *cv) {
        fprintf(stderr, "kx: REFUSING — CUDA_VISIBLE_DEVICES=%s is set alongside ROCm.\n", cv);
        exit(64);
    }
    const char* rv = getenv("ROCR_VISIBLE_DEVICES");
    if (!rv || !*rv) {
        fprintf(stderr,
                "kx: REFUSING — ROCR_VISIBLE_DEVICES is unset, so this run is not pinned to an\n"
                "    idle card. An unpinned run on a contended box has read A/A 0.27..4.57 on\n"
                "    this machine. Run through scripts/kx.sh, which takes the lease.\n");
        exit(64);
    }
    int nd = 0;
    CK(hipGetDeviceCount(&nd));
    if (nd != 1) {
        fprintf(stderr,
                "kx: REFUSING — %d devices visible with ROCR_VISIBLE_DEVICES=%s. The lease must\n"
                "    narrow this to exactly one card.\n",
                nd, rv);
        exit(64);
    }
}

int main(int argc, char** argv) {
    const kx_exp* E = kx_experiment();

    /* --plan: the build plan, read by scripts/kx.sh BEFORE any device object exists. Keeping it
     * here rather than in the shell is what makes an experiment ONE file: the variants and their
     * defines live next to the arms that need them. */
    if (argc > 1 && !strcmp(argv[1], "--plan")) {
        printf("name\t%s\n", E->name);
        printf("source\t%s\n", E->source);
        printf("arch\t%s\n", E->arch);
        printf("defines\t%s\n", E->defines);
        printf("threads\t%u\n", E->threads);
        for (int i = 0; i < E->nvar; i++) printf("variant\t%s\t%s\n", E->var[i].label, E->var[i].defines);
        return 0;
    }

    int it = 41, want_geom = -1, arena_mb = 3072;
    double aa_lo = E->aa_lo, aa_hi = E->aa_hi;
    const char* only_shape = nullptr;
    std::vector<const char*> cos;
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (!strncmp(a, "--it=", 5)) it = atoi(a + 5);
        else if (!strncmp(a, "--arena-mb=", 11)) arena_mb = atoi(a + 11);
        else if (!strncmp(a, "--shape=", 8)) only_shape = a + 8;
        /* Tightening the A/A gate is how you check the gate itself still bites. */
        else if (!strncmp(a, "--aa-tol=", 9)) sscanf(a + 9, "%lf:%lf", &aa_lo, &aa_hi);
        else if (!strcmp(a, "--grid=1")) want_geom = GEOM_1WG;
        else if (!strcmp(a, "--grid=model")) want_geom = GEOM_MODEL;
        else if (!strcmp(a, "--grid=both")) want_geom = -1;
        else cos.push_back(a);
    }
    if ((int)cos.size() != E->nvar) {
        fprintf(stderr, "kx: expected %d code object(s), got %d\n", E->nvar, (int)cos.size());
        return 65;
    }

    gate_environment();
    CK(hipInit(0));
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));

    std::vector<hipModule_t> mod(E->nvar);
    for (int i = 0; i < E->nvar; i++) CK(hipModuleLoad(&mod[i], cos[i]));
    std::vector<hipFunction_t> fn(E->narm);
    for (int i = 0; i < E->narm; i++) {
        hipError_t e = hipModuleGetFunction(&fn[i], mod[E->arm[i].variant], E->arm[i].sym);
        if (e != hipSuccess) {
            fprintf(stderr, "kx: arm '%s' not found in variant '%s' (%s)\n", E->arm[i].sym,
                    E->var[E->arm[i].variant].label, cos[E->arm[i].variant]);
            return 66;
        }
    }
    /* The gate kernels come from variant 0's object; they are identical in all of them. */
    hipFunction_t f_cmp, f_fill;
    CK(hipModuleGetFunction(&f_cmp, mod[0], E->dtype == KX_DT_F32 ? "kx_cmp_f32" : "kx_cmp_bf16"));
    CK(hipModuleGetFunction(&f_fill, mod[0], "kx_fill_u16"));

    /* Display name. Two variants of one experiment routinely export the SAME symbol -- that is
     * the point of a variant axis -- so a bare symbol makes the report ambiguous. */
    std::vector<std::string> dn(E->narm);
    for (int i = 0; i < E->narm; i++)
        dn[i] = std::string(E->var[E->arm[i].variant].label) + ":" + E->arm[i].sym;

    int i_base = -1, i_aa = -1, i_ref = -1;
    for (int i = 0; i < E->narm; i++) {
        if (E->arm[i].flags & KX_BASE) i_base = i;
        if (E->arm[i].flags & KX_AA) i_aa = i;
        if (E->arm[i].flags & KX_REF) i_ref = i;
    }
    if (i_base < 0 || i_aa < 0) {
        fprintf(stderr, "kx: REFUSING — experiment declares no %s arm. Every experiment needs a\n"
                        "    baseline and an A/A control that is byte-identical device code.\n",
                i_base < 0 ? "KX_BASE" : "KX_AA");
        return 67;
    }
    if (i_ref < 0) i_ref = i_base;

    const size_t ARENA = (size_t)arena_mb << 20;
    /* REFUSE rather than fault. A shape whose slab does not fit the arena cannot be walked, and
     * silently clamping nrep to 1 makes the kernel read past the allocation. */
    for (int s = 0; s < E->nshp; s++) {
        if (E->shp[s].slab_bytes > (double)ARENA) {
            fprintf(stderr,
                    "kx: REFUSING — shape '%s' walks a %.2f GiB slab and the arena is %d MiB.\n"
                    "    Pass --arena-mb=%.0f or drop the shape; clamping would read out of bounds.\n",
                    E->shp[s].name, E->shp[s].slab_bytes / 1073741824.0, arena_mb,
                    E->shp[s].slab_bytes / 1048576.0 + 64);
            return 68;
        }
    }
    const size_t SIDE = 64u << 20; /* out/x/aux; every ported shape's output fits with room */
    unsigned short *W, *X, *AUX, *OA, *OB;
    float* ACC;
    CK(hipMalloc(&W, ARENA));
    CK(hipMalloc(&X, SIDE));
    CK(hipMalloc(&AUX, SIDE));
    CK(hipMalloc(&OA, SIDE));
    CK(hipMalloc(&OB, SIDE));
    CK(hipMalloc(&ACC, 4 * sizeof(float)));
    {
        unsigned long long n;
        unsigned seed;
        void* a[] = {nullptr, &n, &seed};
        struct {
            unsigned short* p;
            size_t bytes;
            unsigned seed;
        } fills[] = {{W, ARENA, 1u}, {X, SIDE, 2u}, {AUX, SIDE, 3u}};
        for (auto& f : fills) {
            n = f.bytes / 2;
            seed = f.seed;
            a[0] = &f.p;
            CK(hipModuleLaunchKernel(f_fill, 4096, 1, 1, 256, 1, 1, 0, 0, a, nullptr));
        }
        CK(hipDeviceSynchronize());
    }

    printf("\nkx %s — %s\n", E->name, E->question);
    printf("    source   %s\n", E->source);
    printf("    device   %s (%s)  ROCR_VISIBLE_DEVICES=%s\n", prop.name, prop.gcnArchName,
           getenv("ROCR_VISIBLE_DEVICES"));
    printf("    launch   blockDim=%u  median-of-%d, palindromic interleave, all arms in ONE process\n",
           E->threads, it);
    for (int i = 0; i < E->nvar; i++)
        printf("    variant  %-8s %s\n", E->var[i].label,
               E->var[i].defines[0] ? E->var[i].defines : "(shipped defines only)");
    printf("    arms\n");
    for (int i = 0; i < E->narm; i++)
        printf("      %-18s %s%s\n", dn[i].c_str(), E->arm[i].note,
               (E->arm[i].flags & KX_BASE)   ? "  [BASE]"
               : (E->arm[i].flags & KX_AA)   ? "  [A/A control]"
               : (E->arm[i].flags & KX_SLOWOK) ? "  [expected loss]"
                                             : "");
    if (E->expect) printf("    known    %s\n", E->expect);

    /* ------------------------------------------------------------------ TIMING */
    std::vector<Row> rows;
    int g0 = (want_geom < 0) ? GEOM_1WG : want_geom;
    int g1 = (want_geom < 0) ? GEOM_MODEL : want_geom;
    for (int g = g0; g <= g1; g++) {
        for (int s = 0; s < E->nshp; s++) {
            const kx_shape& S = E->shp[s];
            if (only_shape && strcmp(only_shape, S.name)) continue;
            const unsigned grid = (g == GEOM_1WG) ? 1u : S.grid;
            unsigned nrep = kx_reps(S, ARENA);
            void* a[] = {&OA, &W, &X, &AUX, &nrep, (void*)&S.grid, (void*)&S.d0, (void*)&S.d1, (void*)&S.d2};
            Row r;
            r.geom = g;
            r.shape = s;
            r.us.assign(E->narm, 1e30);
            /* PALINDROMIC: forward then reverse, min per arm. A monotone drift then biases both
             * directions equally instead of favouring whichever arm ran first. */
            for (int i = 0; i < E->narm; i++)
                r.us[i] = std::min(r.us[i], median_us(fn[i], grid, E->threads, a, it) / nrep);
            for (int i = E->narm - 1; i >= 0; i--)
                r.us[i] = std::min(r.us[i], median_us(fn[i], grid, E->threads, a, it) / nrep);
            r.aa = r.us[i_aa] / r.us[i_base];
            r.aa_ok = (r.aa >= aa_lo && r.aa <= aa_hi);
            rows.push_back(r);
        }
    }

    /* ----------------------------------------------------------- CORRECTNESS, ON DEVICE.
     * Always at the MODEL grid: at 1wg a body that partitions by blockIdx writes only its own
     * slice, so a 1wg comparison would silently pass on untouched memory. */
    std::vector<Chk> chks;
    for (int s = 0; s < E->nshp; s++) {
        const kx_shape& S = E->shp[s];
        if (only_shape && strcmp(only_shape, S.name)) continue;
        if (!S.out_elems) continue;
        unsigned one = 1;
        void* aref[] = {&OB, &W, &X, &AUX, &one, (void*)&S.grid, (void*)&S.d0, (void*)&S.d1, (void*)&S.d2};
        CK(hipMemset(OB, 0, SIDE));
        CK(hipModuleLaunchKernel(fn[i_ref], S.grid, 1, 1, E->threads, 1, 1, 0, 0, aref, nullptr));
        CK(hipDeviceSynchronize());
        for (int i = 0; i < E->narm; i++) {
            if (i == i_ref || (E->arm[i].flags & KX_NOCHECK)) continue;
            void* ac[] = {&OA, &W, &X, &AUX, &one, (void*)&S.grid, (void*)&S.d0, (void*)&S.d1, (void*)&S.d2};
            CK(hipMemset(OA, 0, SIDE));
            CK(hipMemset(ACC, 0, 4 * sizeof(float)));
            CK(hipModuleLaunchKernel(fn[i], S.grid, 1, 1, E->threads, 1, 1, 0, 0, ac, nullptr));
            void* cargs[] = {&OA, &OB, (void*)&S.out_elems, &ACC};
            CK(hipModuleLaunchKernel(f_cmp, 256, 1, 1, 256, 1, 1, 0, 0, cargs, nullptr));
            CK(hipDeviceSynchronize());
            float h[4];
            CK(hipMemcpy(h, ACC, sizeof(h), hipMemcpyDeviceToHost));
            Chk c;
            c.arm = i;
            c.shape = s;
            c.maxe = h[0];
            c.rms = std::sqrt((double)h[1] / (double)S.out_elems);
            c.exact = h[2];
            c.n = S.out_elems;
            c.bad = (int)h[3];
            const bool exact_ok = (h[2] >= (float)S.out_elems);
            c.verdict = (E->arm[i].flags & KX_EXACT) ? (exact_ok ? 0 : 1)
                                                     : ((c.maxe <= E->tol_max) ? 0 : 1);
            chks.push_back(c);
        }
    }

    /* ------------------------------------------------------------------- REPORT */
    bool aa_failed = false, any_wrong = false;
    double aa_lo_seen = 1e30, aa_hi_seen = -1e30;
    for (auto& r : rows) {
        aa_lo_seen = std::min(aa_lo_seen, r.aa);
        aa_hi_seen = std::max(aa_hi_seen, r.aa);
        if (!r.aa_ok) aa_failed = true;
    }
    for (auto& c : chks)
        if (c.verdict) any_wrong = true;

    for (int g = g0; g <= g1; g++) {
        printf("\nGEOM %s%s\n", geom_name[g],
               g == GEOM_1WG
                   ? "   ONE workgroup, doing exactly the share the model deals it (slice 0 of"
                     " kx_nblk). The body on its own, NOT the model."
                   : "  grid = the emitter's own workgroup count for the shape.");
        printf("%-16s %7s %7s %5s %5s | %10s", "shape", "d0", "d1", "grid", "nrep", "base_us");
        for (int i = 0; i < E->narm; i++) {
            if (i == i_base) continue;
            printf(" %14.14s", (dn[i] + "/base").c_str());
        }
        printf("   (>1 = faster than base)\n");
        for (auto& r : rows) {
            if (r.geom != g) continue;
            const kx_shape& S = E->shp[r.shape];
            const unsigned nrep = kx_reps(S, ARENA);
            printf("%-16s %7u %7u %5u %5u | %10.3f", S.name, S.d0, S.d1,
                   g == GEOM_1WG ? 1u : S.grid, nrep, r.us[i_base]);
            for (int i = 0; i < E->narm; i++) {
                if (i == i_base) continue;
                if (!r.aa_ok) printf(" %14s", "REFUSED");
                else printf(" %14.3f", r.us[i_base] / r.us[i]);
            }
            printf("%s\n", r.aa_ok ? "" : "   <- A/A out of tolerance");
        }
        /* Per-token totals only where they mean anything: the model geometry. */
        if (g == GEOM_MODEL && !aa_failed) {
            bool any_inst = false;
            for (int s = 0; s < E->nshp; s++) any_inst |= (E->shp[s].inst != 0);
            if (any_inst) {
                printf("  per-token totals over the tabulated instance counts (ms):");
                for (int i = 0; i < E->narm; i++) {
                    if (E->arm[i].flags & (KX_AA | KX_NOCHECK)) continue;
                    double t = 0;
                    for (auto& r : rows)
                        if (r.geom == g) t += r.us[i] * E->shp[r.shape].inst / 1e3;
                    printf("  %s %.3f", dn[i].c_str(), t);
                }
                printf("\n");
            }
        }
    }

    printf("\nCORRECTNESS (device-side elementwise, vs %s at the model grid)\n", dn[i_ref].c_str());
    printf("%-18s %-16s %11s %11s %14s  %s\n", "arm", "shape", "max_err", "rms_err", "exact/N",
           "verdict");
    for (auto& c : chks) {
        char ex[40];
        snprintf(ex, sizeof ex, "%.0f/%u", c.exact, c.n);
        printf("%-18s %-16s %11.3e %11.3e %14s  %s\n", dn[c.arm].c_str(), E->shp[c.shape].name,
               c.maxe, c.rms, ex,
               c.verdict ? ((E->arm[c.arm].flags & KX_EXACT) ? "WRONG — not bit-identical"
                                                             : "WRONG — outside tolerance")
                         : ((c.exact >= (float)c.n) ? "exact" : "ok"));
    }
    if (chks.empty()) printf("  (no shape declares out_elems — nothing was checked)\n");

    printf("\nA/A control (%s / %s): %.4f .. %.4f   tolerance %.3f .. %.3f   %s\n",
           dn[i_aa].c_str(), dn[i_base].c_str(), aa_lo_seen, aa_hi_seen, aa_lo, aa_hi,
           aa_failed ? "FAIL" : "PASS");

    int rc = 0;
    if (aa_failed) {
        printf("\nREFUSED: the A/A control is outside tolerance, so every ratio above is\n"
               "  instrument, not kernel. Nothing here may be reported. Re-run pinned to an idle\n"
               "  card under the lease, or widen --it.\n");
        rc = 3;
    }
    if (any_wrong) {
        printf("\nREFUSED AS A WIN: an arm above is WRONG. A candidate that is faster and wrong\n"
               "  is not a result. Its speed column is printed only so the trap is visible.\n");
        rc = rc ? rc : 4;
    }
    if (!rc) printf("\nOK: A/A in tolerance and every arm matches the reference.\n");
    return rc;
}
