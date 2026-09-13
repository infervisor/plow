/* mq_probe_gemmlt_bench.c — multi-queue AQL probe, STAGE 1: the real GLM-5.3 op pair.
 *
 * Same mechanism question as mq_probe_bench.c (does a second, barrier-free AQL queue let two
 * kernels co-reside on gfx942?), with the REAL production hipBLASLt-derived `GemmLtPf` kernel
 * standing in for the synthetic FMA proxy. `GemmLtPf` is not a live hipBLASLt library call at
 * runtime — `crates/plowrt/src/exec/amd_gemm_lt.rs` loads a plain raw ELF
 * (`runtime/amd/glm_lt_gfx942.elf`, Tensile-generated assembly, dispatched through the exact
 * same `HsaBackend::launch` path as everything else) and picks a PINNED kernel + 160-byte
 * kernarg (`Args`) per shape. This harness reproduces that exactly, for the decode shape
 * rows=20 n=6144 k=2048 (`decode_choice(20,6144,2048)` -> combined index 10 =
 * `glm_lt_decode_gfx942.json[6]`, MT32x32).
 *
 * Two differences from mq_probe_bench.c's synthetic proxy that matter here:
 *   - GemmLtPf is OPAQUE Tensile assembly: no source to instrument with s_memrealtime writes.
 *     Timing instead uses `hsa_amd_profiling_get_dispatch_time` on each dispatch's own
 *     completion signal, in the HSA SYSTEM clock domain — works on any kernel, vendor or not,
 *     and (unlike mq_probe_bench.c's device-side timestamps) puts the collective and the GEMM
 *     on the SAME clock so their windows are directly comparable with no refclock conversion.
 *     Precondition for a clean read: query a signal only once, right after waiting for the ONE
 *     dispatch that just used it — true by construction in every arm below.
 *   - Its natural grid (192 workgroups at MT32x32x256) is not calibrated to fill a CU count; it
 *     is whatever the pinned kernel's tile size gives this shape, exactly as production
 *     dispatches it — "not as polite as a synthetic 272-CU kernel."
 *
 * GemmLtPf's object is loaded as a SECOND, independent `hsa_executable_t` on rank 0's agent via
 * raw HSA calls below (`plow_hsa_agent_raw`) — `hsa_backend.c`'s `plow_dev_t` carries only one
 * executable slot by design, so a second one is built here rather than by extending that
 * struct. It is then dispatched exactly like any other `plow_hsa_kernel`: through
 * `plow_hsa_launch` (default queue, arm `1q`) or through the same `q2_launch` mq_probe_bench.c
 * uses (second queue, arms `2q`/`2qm`).
 *
 * env: MQ_ARM=solo_gemmlt|1q|2q|2qm (required)  MQ_ELF (mq_probe_kernels.elf)
 *      MQ_GEMMLT_ELF (glm_lt_gfx942.elf)  MQ_XR_NWG (32)  MQ_REPS (9)
 * usage: mq_probe_gemmlt_bench gpu0 ... gpu7
 */
#include "../amd/hsa_backend.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 8
#define REGION_BYTES (127u * 1024u * 1024u)
#define XCTR_OFF 132120576u
#define MQ_KARG_SLOT 512u
#define MQ_Q2_SIZE 64u

#define GLT_ARGS_BYTES 160u
#define GLT_MAX_N 16u

/* Two real, pinned GemmLtPf shapes (crates/plowrt/src/exec/amd_gemm_lt.rs). `decode_choice`
 * covers DECODE; `kernel_choice`'s generic fallback (n,k)=(4096,2048) — one of the three
 * literal prefill tuples `routes()` accepts for mode i[3]==0 — covers the PREFILL shape the
 * coordinator named: q_absorb / idx q_b, 8192x4096x2048, ~275 us/instance, 99/chunk
 * (attribution.md §4a). Decode is ~14-18 us next to a ~670 us collective (a small hiding
 * opportunity by construction); prefill is the shape where hiding independent work actually
 * matters. Each pinned descriptor ships with KERNARG_SIZE zeroed at `kernarg_offset` — patched
 * to GLT_ARGS_BYTES before loading, exactly as production's `load_kernels()` does. */
typedef struct {
    uint32_t m, n, k, mt_i, mt_j, info1, kernarg_offset;
    const char* kernel_name;
} GltShape;

static const GltShape GLT_DECODE = {
    20u, 6144u, 2048u, 32u, 32u, 524289u, 2977032u,
    "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x256_MI16x16x1_SN_LDSB1_AFC1_AG0_AGGSUA0_"
    "AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_"
    "DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_HPLR0_"
    "ISA942_ICIW0_IU1_K1_LDSTI0_LBSPPA1024_LBSPPB1024_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA16_LPB16_"
    "LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT2_2_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_"
    "NTC0_NTD4_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_"
    "NVM0_NVWS0_NEPBS16_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_SPO0_SRVW0_"
    "SSO0_SVW2_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_THn1_THA0_"
    "THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO1_USL1_USLMX0_UIOFGRO0_"
    "UPLRP0_USFGROn1_USI0_VSn1_VWA2_VWB2_WSGRA0_WSGRB0_WS64_WG16_4_4"};

/* glm_lt_gfx942.json[0]: prefill (n,k)=(4096,2048), MT256x224x64, kernel_choice()'s generic
 * fallback for that pair (index 0, info1=(8<<16)|6=524294). */
static const GltShape GLT_PREFILL = {
    8192u, 4096u, 2048u, 256u, 224u, 524294u, 2979144u,
    "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x224x64_MI16x16x1_SN_LDSB0_AFC1_AG0_"
    "AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA1_DTVB0_"
    "DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMBSK_GLS0_"
    "HPLR0_ISA942_ICIW0_IU1_K1_LDSTI0_LBSPPA0_LBSPPB256_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_"
    "LPB16_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_14_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_"
    "NTA0_NTB0_NTC0_NTD4_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_"
    "NVMXSB0_NVM0_NVWS0_NEPBS16_NLCA2_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA3_SS1_"
    "SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_TDMIM0_TDMS0_TIN0_"
    "THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSM1_ULSGRO0_USL1_"
    "USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA4_VWB2_WSGRA0_WSGRB0_WS64_WG64_4_1"};

typedef uint16_t bf16;

/* Mirrors crates/plowrt/src/exec/amd_gemm_lt.rs's `Args` (`#[repr(C)]`, asserted 160 bytes). */
typedef struct {
    uint32_t dims[8];
    uint64_t pointers[4];
    uint32_t strides[8];
    float alpha;
    float beta;
    uint32_t epilogue[14];
} GltArgs;

static uint32_t envu(const char* k, uint32_t d) { const char* v = getenv(k); return v ? (uint32_t)strtoul(v, 0, 10) : d; }
static const char* envs(const char* k, const char* d) { const char* v = getenv(k); return v ? v : d; }
static void* slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f); *len = (size_t)n; return p;
}
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }
static double med(double* a, uint32_t n) { qsort(a, n, sizeof(double), cmpd); return a[n / 2]; }

/* --- a caller-owned second AQL queue on one agent (identical to mq_probe_bench.c's q2_t). --- */
typedef struct {
    hsa_agent_t agent;
    hsa_queue_t* q;
    uint8_t* karg_ring;
    hsa_signal_t sig;
} q2_t;

static int q2_init(q2_t* c, hsa_agent_t agent, hsa_amd_memory_pool_t kpool) {
    c->agent = agent;
    if (hsa_queue_create(agent, MQ_Q2_SIZE, HSA_QUEUE_TYPE_SINGLE, NULL, NULL,
                         UINT32_MAX, UINT32_MAX, &c->q) != HSA_STATUS_SUCCESS)
        return -1;
    if (hsa_amd_memory_pool_allocate(kpool, (size_t)MQ_Q2_SIZE * MQ_KARG_SLOT, 0,
                                     (void**)&c->karg_ring) != HSA_STATUS_SUCCESS)
        return -1;
    hsa_amd_agents_allow_access(1, &agent, NULL, c->karg_ring);
    return hsa_signal_create(0, 0, NULL, &c->sig) == HSA_STATUS_SUCCESS ? 0 : -1;
}

static int q2_launch(q2_t* c, const plow_hsa_kernel* k, uint32_t grid_x, uint16_t wg_x,
                     const void* args, size_t args_size, int barrier) {
    hsa_queue_t* q = c->q;
    uint64_t idx = hsa_queue_add_write_index_screlease(q, 1);
    while (idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {}
    const uint32_t slot = (uint32_t)(idx & (q->size - 1));
    uint8_t* karg = c->karg_ring + (size_t)slot * MQ_KARG_SLOT;
    memcpy(karg, args, args_size);
    memset(karg + args_size, 0, k->kernarg_size - args_size);
    /* No COv5 hidden block: GemmLtPf's kernarg_size equals args_size exactly (no implicit
     * args), so hoff == kernarg_size and the generic hidden-args write is a no-op — same
     * invariant plow_hsa_launch relies on for this kernel family. */
    const size_t hoff = (args_size + 7u) & ~(size_t)7u;
    if (k->kernarg_size > hoff) {
        uint8_t* hid = karg + hoff;
        const size_t avail = k->kernarg_size - hoff;
#define PUT32(off, val) if (avail >= (off) + 4) *(uint32_t*)(hid + (off)) = (uint32_t)(val)
#define PUT16(off, val) if (avail >= (off) + 2) *(uint16_t*)(hid + (off)) = (uint16_t)(val)
        PUT32(0, (grid_x + wg_x - 1) / wg_x);
        PUT16(12, wg_x);
        PUT16(18, grid_x % wg_x);
        PUT16(64, 1);
#undef PUT32
#undef PUT16
    }
    hsa_kernel_dispatch_packet_t* p = (hsa_kernel_dispatch_packet_t*)q->base_address + slot;
    memset((uint8_t*)p + 4, 0, sizeof(*p) - 4);
    p->workgroup_size_x = wg_x; p->workgroup_size_y = 1; p->workgroup_size_z = 1;
    p->grid_size_x = grid_x; p->grid_size_y = 1; p->grid_size_z = 1;
    p->kernel_object = k->kernel_object;
    p->kernarg_address = karg;
    p->group_segment_size = k->group_segment_size;
    p->private_segment_size = k->private_segment_size;
    p->completion_signal = c->sig;
    hsa_signal_add_screlease(c->sig, 1);
    uint16_t header = (uint16_t)((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
                    | ((barrier ? 1u : 0u) << HSA_PACKET_HEADER_BARRIER)
                    | (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
                    | (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
    uint16_t setup = (uint16_t)(1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
    __atomic_store_n((uint32_t*)p, ((uint32_t)setup << 16) | header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)idx);
    return 0;
}

static void q2_wait(q2_t* c) {
    while (hsa_signal_wait_scacquire(c->sig, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                     HSA_WAIT_STATE_BLOCKED) != 0) {}
}

/* One-shot dispatch to an ARBITRARY queue with its OWN dedicated (non-counting) completion
 * signal and kernarg buffer — needed for the `1q` arm, which must fire the collective and the
 * GEMM at the SAME queue back-to-back with NO host wait between them (the "fire-ahead" pattern
 * docs/arch/06-runtime.md's segmented dispatch actually uses: "the host does not wait between
 * them"). plow_hsa_launch can't be reused for this: its shared per-device counting signal
 * cannot isolate two back-to-back dispatches' individual hsa_amd_profiling_get_dispatch_time
 * records, only a SEPARATE signal per dispatch can. */
static void raw_launch(hsa_queue_t* q, uint8_t* karg, hsa_signal_t sig,
                       const plow_hsa_kernel* k, uint32_t grid_x, uint16_t wg_x,
                       const void* args, size_t args_size, int barrier) {
    uint64_t idx = hsa_queue_add_write_index_screlease(q, 1);
    while (idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {}
    memcpy(karg, args, args_size);
    memset(karg + args_size, 0, k->kernarg_size - args_size);
    /* COv5 hidden block: mq_xreduce_twoshot_ts is a normal HIP kernel and reads gridDim.x /
     * blockIdx.x-derived nblk from HERE (d_xreduce_twoshot_mega's rendezvous partitions its
     * work by it) — omitting this, unlike q2_launch, silently zeroes nblk and hangs the
     * collective's own arrival count. GemmLtPf's kernarg_size == args_size exactly, so this is
     * a no-op for it (hoff == kernarg_size), matching q2_launch's invariant for that kernel. */
    const size_t hoff = (args_size + 7u) & ~(size_t)7u;
    if (k->kernarg_size > hoff) {
        uint8_t* hid = karg + hoff;
        const size_t avail = k->kernarg_size - hoff;
#define PUT32(off, val) if (avail >= (off) + 4) *(uint32_t*)(hid + (off)) = (uint32_t)(val)
#define PUT16(off, val) if (avail >= (off) + 2) *(uint16_t*)(hid + (off)) = (uint16_t)(val)
        PUT32(0, (grid_x + wg_x - 1) / wg_x);
        PUT16(12, wg_x);
        PUT16(18, grid_x % wg_x);
        PUT16(64, 1);
#undef PUT32
#undef PUT16
    }
    hsa_kernel_dispatch_packet_t* p =
        (hsa_kernel_dispatch_packet_t*)q->base_address + (idx & (q->size - 1));
    memset((uint8_t*)p + 4, 0, sizeof(*p) - 4);
    p->workgroup_size_x = wg_x; p->workgroup_size_y = 1; p->workgroup_size_z = 1;
    p->grid_size_x = grid_x; p->grid_size_y = 1; p->grid_size_z = 1;
    p->kernel_object = k->kernel_object;
    p->kernarg_address = karg;
    p->group_segment_size = k->group_segment_size;
    p->private_segment_size = k->private_segment_size;
    p->completion_signal = sig;
    hsa_signal_store_relaxed(sig, 1);
    uint16_t header = (uint16_t)((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
                    | ((barrier ? 1u : 0u) << HSA_PACKET_HEADER_BARRIER)
                    | (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
                    | (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
    uint16_t setup = (uint16_t)(1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
    __atomic_store_n((uint32_t*)p, ((uint32_t)setup << 16) | header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)idx);
}

static void raw_wait(hsa_signal_t sig) {
    while (hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                     HSA_WAIT_STATE_BLOCKED) != 0) {}
}

/* Manually load GemmLtPf's real object as a SECOND executable on `agent`, bypassing
 * plow_hsa_load_code_object (one-executable-per-device by design) entirely — resolves straight
 * to a plow_hsa_kernel the SAME shape plow_hsa_get_kernel would produce. */
static int load_gemmlt(hsa_agent_t agent, void* elf, size_t elf_len, const char* name,
                       uint32_t kernarg_offset, plow_hsa_kernel* out) {
    if ((size_t)kernarg_offset + 4u > elf_len) return -1;
    uint8_t* field = (uint8_t*)elf + kernarg_offset;
    if (field[0] || field[1] || field[2] || field[3]) {
        fprintf(stderr, "GemmLtPf descriptor at %u is not the expected zeroed KERNARG_SIZE "
                        "(object drifted from the pinned spec?)\n", kernarg_offset);
        return -1;
    }
    const uint32_t args_bytes_le = GLT_ARGS_BYTES;
    memcpy(field, &args_bytes_le, 4);
    hsa_code_object_reader_t rdr;
    if (hsa_code_object_reader_create_from_memory(elf, elf_len, &rdr) != HSA_STATUS_SUCCESS)
        return -1;
    hsa_executable_t exe;
    if (hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
                                  &exe) != HSA_STATUS_SUCCESS)
        return -1;
    if (hsa_executable_load_agent_code_object(exe, agent, rdr, NULL, NULL) != HSA_STATUS_SUCCESS)
        return -1;
    if (hsa_executable_freeze(exe, NULL) != HSA_STATUS_SUCCESS) return -1;
    hsa_code_object_reader_destroy(rdr);
    const size_t name_len = strlen(name);
    char* sym_name = malloc(name_len + 4);
    memcpy(sym_name, name, name_len);
    memcpy(sym_name + name_len, ".kd", 4);
    hsa_executable_symbol_t sym;
    const hsa_status_t lookup = hsa_executable_get_symbol_by_name(exe, sym_name, &agent, &sym);
    free(sym_name);
    if (lookup != HSA_STATUS_SUCCESS) return -1;
    if (hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &out->kernel_object) != HSA_STATUS_SUCCESS)
        return -1;
    if (hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                       &out->kernarg_size) != HSA_STATUS_SUCCESS)
        return -1;
    if (hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                       &out->group_segment_size) != HSA_STATUS_SUCCESS)
        return -1;
    if (hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                       &out->private_segment_size) != HSA_STATUS_SUCCESS)
        return -1;
    out->kernarg_explicit = 0;
    return 0;
}

typedef struct {
    void* out; const void* peers; uint32_t nranks, rank, n, slot_bytes;
    uint64_t xctr_byte_off; uint64_t deadline; void* ts; void* status;
} arg_xr;
typedef struct { void* part; uint32_t n; uint32_t rank; } arg_fill;

int main(int argc, char** argv) {
    if (argc != NR + 1) { fprintf(stderr, "usage: %s gpu0 ... gpu7\n", argv[0]); return 2; }
    const char* arm = envs("MQ_ARM", "");
    if (strcmp(arm, "solo_gemmlt") && strcmp(arm, "1q") && strcmp(arm, "2q") && strcmp(arm, "2qm")) {
        fprintf(stderr, "MQ_ARM must be solo_gemmlt|1q|2q|2qm\n"); return 2;
    }
    const int masked = !strcmp(arm, "2qm");
    const int prefill_shape = !strcmp(envs("MQ_GLT_SHAPE", "decode"), "prefill");
    const GltShape* shp = prefill_shape ? &GLT_PREFILL : &GLT_DECODE;
    /* MQ_GLT_N: independent GEMM copies queued back-to-back on q2 in 2q/2qm, each with its own
     * output buffer + dedicated signal — "how many us of independent work can hide inside one
     * collective", not just "does one small kernel hide" (only meaningful for 2q/2qm; 1q/solo
     * always use exactly one dispatch). */
    const uint32_t glt_n_fill = envu("MQ_GLT_N", 1);
    if (glt_n_fill < 1 || glt_n_fill > GLT_MAX_N) {
        fprintf(stderr, "MQ_GLT_N must be 1..%u\n", GLT_MAX_N); return 2;
    }
    const uint32_t rows = 8192, hidden = 6144; /* the collective's own shape, as in mq_probe_bench.c */
    const uint32_t xr_nwg = envu("MQ_XR_NWG", 32), reps = envu("MQ_REPS", 9);
    const uint64_t n = (uint64_t)rows * hidden;
    int dev[NR]; for (int r = 0; r < NR; r++) dev[r] = atoi(argv[r + 1]);
    /* Defensive: print which arm/shape is about to run BEFORE any HSA call, and flush — if
     * something below faults or hangs, this line still reaches the log. */
    fprintf(stderr, "mq_probe_gemmlt_bench: arm=%s shape=%s glt_n=%u starting\n",
            arm, prefill_shape ? "prefill" : "decode", glt_n_fill);
    fflush(stderr);

    plow_hsa* h = plow_hsa_init();
    if (!h) { fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error()); return 2; }
    uint64_t freq = 0; hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    const uint64_t deadline = freq ? freq : 1000000000ull;
    size_t elf_len = 0;
    void* elf = slurp(envs("MQ_ELF", "mq_probe_kernels.elf"), &elf_len);
    if (!elf) { fprintf(stderr, "no collective ELF\n"); return 2; }
    size_t glt_len = 0;
    void* glt_elf = slurp(envs("MQ_GEMMLT_ELF", "glm_lt_gfx942.elf"), &glt_len);
    if (!glt_elf) { fprintf(stderr, "no GemmLtPf ELF\n"); return 2; }

    plow_hsa_kernel kfill[NR], kxr[NR], kgemmlt;
    void *scratch[NR], *table[NR], *out[NR], *xrts[NR];
    uint32_t* status[NR];
    for (int r = 0; r < NR; r++) {
        if (plow_hsa_load_code_object(h, dev[r], elf, elf_len) ||
            plow_hsa_get_kernel(h, dev[r], "mq_fill_partial", &kfill[r]) ||
            plow_hsa_get_kernel(h, dev[r], "mq_xreduce_twoshot_ts", &kxr[r])) {
            fprintf(stderr, "load: %s\n", plow_hsa_last_error()); return 2;
        }
        scratch[r] = plow_hsa_alloc_peer(h, dev[r], REGION_BYTES);
        table[r] = plow_hsa_alloc(h, dev[r], NR * sizeof(void*));
        out[r] = plow_hsa_alloc(h, dev[r], (size_t)n * 2u);
        xrts[r] = plow_hsa_alloc(h, dev[r], 16);
        status[r] = (uint32_t*)plow_hsa_alloc(h, dev[r], 4);
        if (!scratch[r] || !table[r] || !out[r] || !xrts[r] || !status[r]) {
            fprintf(stderr, "alloc: %s\n", plow_hsa_last_error()); return 2;
        }
    }
    for (int r = 0; r < NR; r++)
        plow_hsa_upload(h, dev[r], table[r], scratch, sizeof scratch);

    hsa_agent_t agent0 = { .handle = plow_hsa_agent_raw(h, dev[0]) };
    if (load_gemmlt(agent0, glt_elf, glt_len, shp->kernel_name, shp->kernarg_offset, &kgemmlt) != 0) {
        fprintf(stderr, "load GemmLtPf: %s\n", plow_hsa_last_error()); return 2;
    }
    if (kgemmlt.kernarg_size != GLT_ARGS_BYTES) {
        fprintf(stderr, "GemmLtPf kernarg %u != expected %u (object drifted?)\n",
                kgemmlt.kernarg_size, GLT_ARGS_BYTES);
        return 2;
    }

    q2_t q2;
    hsa_amd_memory_pool_t kpool = { .handle = plow_hsa_kernarg_pool_raw(h) };
    if (q2_init(&q2, agent0, kpool) != 0) { fprintf(stderr, "q2_init failed\n"); return 2; }
    hsa_queue_t* q0 = (hsa_queue_t*)(uintptr_t)plow_hsa_queue_raw(h, dev[0]);
    hsa_amd_profiling_set_profiler_enabled(q0, 1);
    hsa_amd_profiling_set_profiler_enabled(q2.q, 1);

    /* Dedicated kernarg buffers + signals for the `1q` arm's fire-ahead pair on q0: HOST-visible
     * kernarg-pool memory (same pool q2 uses), NOT device VRAM — raw_launch memcpy's into these
     * from host code, which plow_hsa_alloc's coarse-grained VRAM would not permit. */
    uint8_t *xr_karg0 = NULL, *gemm_karg0 = NULL;
    hsa_signal_t xr_sig0 = {0}, gemm_sig0 = {0};
    if (hsa_amd_memory_pool_allocate(kpool, MQ_KARG_SLOT, 0, (void**)&xr_karg0) != HSA_STATUS_SUCCESS ||
        hsa_amd_memory_pool_allocate(kpool, MQ_KARG_SLOT, 0, (void**)&gemm_karg0) != HSA_STATUS_SUCCESS ||
        hsa_signal_create(0, 0, NULL, &xr_sig0) != HSA_STATUS_SUCCESS ||
        hsa_signal_create(0, 0, NULL, &gemm_sig0) != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "1q dedicated signal/kernarg setup failed\n"); return 2;
    }
    hsa_amd_agents_allow_access(1, &agent0, NULL, xr_karg0);
    hsa_amd_agents_allow_access(1, &agent0, NULL, gemm_karg0);
    if (masked) {
        uint32_t mask0[10] = {0}, mask1[10] = {0};
        for (uint32_t c = 0; c < 304; c++)
            (c < 32 ? mask0 : mask1)[c / 32] |= 1u << (c % 32);
        if (hsa_amd_queue_cu_set_mask(q0, 320, mask0) != HSA_STATUS_SUCCESS ||
            hsa_amd_queue_cu_set_mask(q2.q, 320, mask1) != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "cu_set_mask failed\n"); return 2;
        }
    }

    /* GemmLtPf operands: x [m,k], weight [n,k] shared by every copy; one out [m,n] PER copy so
     * MQ_GLT_N concurrent dispatches don't race on the same write target. All bf16, filled once
     * with the same hash pattern mq_fill_partial uses elsewhere in this probe family —
     * arbitrary, nonzero, deterministic. */
    const uint32_t glt_grid = ((shp->n + shp->mt_i - 1) / shp->mt_i) * ((shp->m + shp->mt_j - 1) / shp->mt_j);
    const size_t glt_out_bytes = (size_t)shp->m * shp->n * 2u;
    void* glt_x = plow_hsa_alloc(h, dev[0], (size_t)shp->m * shp->k * 2u);
    void* glt_w = plow_hsa_alloc(h, dev[0], (size_t)shp->n * shp->k * 2u);
    if (!glt_x || !glt_w) { fprintf(stderr, "gemmlt x/w alloc failed\n"); return 2; }
    {
        arg_fill fx = {glt_x, shp->m * shp->k, 11u}, fw = {glt_w, shp->n * shp->k, 13u};
        plow_hsa_launch(h, dev[0], &kfill[0], 4096, 1, 1, 256, 1, 1, 0, &fx, sizeof fx);
        plow_hsa_launch(h, dev[0], &kfill[0], 4096, 1, 1, 256, 1, 1, 0, &fw, sizeof fw);
        plow_hsa_wait(h, dev[0]);
    }
    _Static_assert(sizeof(GltArgs) == GLT_ARGS_BYTES, "Args layout drifted from amd_gemm_lt.rs");

    void** glt_out_n = calloc(glt_n_fill, sizeof(void*));
    uint8_t** glt_karg_n = calloc(glt_n_fill, sizeof(uint8_t*));
    hsa_signal_t* glt_sig_n = calloc(glt_n_fill, sizeof(hsa_signal_t));
    GltArgs* gargs_n = calloc(glt_n_fill, sizeof(GltArgs));
    hsa_amd_profiling_dispatch_time_t* tgm_n =
        calloc(glt_n_fill, sizeof(hsa_amd_profiling_dispatch_time_t));
    if (!glt_out_n || !glt_karg_n || !glt_sig_n || !gargs_n || !tgm_n) {
        fprintf(stderr, "MQ_GLT_N bookkeeping alloc failed\n"); return 2;
    }
    for (uint32_t i = 0; i < glt_n_fill; i++) {
        glt_out_n[i] = plow_hsa_alloc(h, dev[0], glt_out_bytes);
        if (!glt_out_n[i] ||
            hsa_amd_memory_pool_allocate(kpool, MQ_KARG_SLOT, 0, (void**)&glt_karg_n[i]) != HSA_STATUS_SUCCESS ||
            hsa_signal_create(0, 0, NULL, &glt_sig_n[i]) != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "gemmlt copy %u setup failed: %s\n", i, plow_hsa_last_error());
            return 2;
        }
        hsa_amd_agents_allow_access(1, &agent0, NULL, glt_karg_n[i]);
        const uint64_t out_u = (uint64_t)(uintptr_t)glt_out_n[i];
        GltArgs* g = &gargs_n[i];
        g->dims[0] = 1; g->dims[1] = 1; g->dims[2] = shp->info1; g->dims[3] = glt_grid;
        g->dims[4] = shp->n; g->dims[5] = shp->m; g->dims[6] = 1; g->dims[7] = shp->k;
        g->pointers[0] = out_u; g->pointers[1] = out_u;
        g->pointers[2] = (uint64_t)(uintptr_t)glt_w; g->pointers[3] = (uint64_t)(uintptr_t)glt_x;
        g->strides[0] = shp->n; g->strides[1] = shp->n * shp->m; g->strides[2] = shp->n;
        g->strides[3] = shp->n * shp->m; g->strides[4] = shp->k; g->strides[5] = shp->k * shp->n;
        g->strides[6] = shp->k; g->strides[7] = shp->k * shp->m;
        g->alpha = 1.0f; g->beta = 0.0f;
        g->epilogue[9] = (uint32_t)out_u; g->epilogue[10] = (uint32_t)(out_u >> 32);
    }

    /* Reference: copy 0 run alone (q2, untimed), before any concurrent arm runs, for the
     * bit-exact corruption check below. */
    bf16* ref_out = malloc(glt_out_bytes);
    bf16* rep_out = malloc(glt_out_bytes);
    raw_launch(q2.q, glt_karg_n[0], glt_sig_n[0], &kgemmlt, glt_grid * 256u, 256, &gargs_n[0],
              sizeof(GltArgs), 0);
    raw_wait(glt_sig_n[0]);
    plow_hsa_download(h, dev[0], ref_out, glt_out_n[0], glt_out_bytes);

    double* wall = calloc(reps, sizeof(double));
    double* xr_dur = calloc(reps, sizeof(double));
    double* gemm_dur = calloc(reps, sizeof(double));
    double* overlap = calloc(reps, sizeof(double));
    double* fill_busy = calloc(reps, sizeof(double));
    double* fill_cover = calloc(reps, sizeof(double));
    int any_timeout = 0;
    const int run_xr = strcmp(arm, "solo_gemmlt") != 0;

    for (uint32_t rep = 0; rep < reps; rep++) {
        uint8_t zero[8192] = {0};
        for (int r = 0; r < NR; r++) {
            if (!run_xr) break;
            plow_hsa_upload(h, dev[r], (char*)scratch[r] + XCTR_OFF, zero, sizeof zero);
            plow_hsa_upload(h, dev[r], status[r], zero, 4);
            arg_fill fa = {scratch[r], (uint32_t)n, (uint32_t)r};
            plow_hsa_launch(h, dev[r], &kfill[r], 4096, 1, 1, 256, 1, 1, 0, &fa, sizeof fa);
            plow_hsa_wait(h, dev[r]);
        }
        arg_xr axr = {out[0], table[0], NR, 0u, (uint32_t)n, 0u, XCTR_OFF, deadline, xrts[0], status[0]};

        hsa_amd_profiling_dispatch_time_t txr = {0, 0}, tgm = {0, 0};
        if (!strcmp(arm, "solo_gemmlt")) {
            raw_launch(q2.q, glt_karg_n[0], glt_sig_n[0], &kgemmlt, glt_grid * 256u, 256,
                      &gargs_n[0], sizeof(GltArgs), 0);
            raw_wait(glt_sig_n[0]);
            hsa_amd_profiling_get_dispatch_time(agent0, glt_sig_n[0], &tgm);
        } else if (!strcmp(arm, "1q")) {
            /* Fire-ahead: both packets enqueued on q0 back-to-back, ordered by the barrier bit,
             * with NO host wait in between — the actual production pattern
             * (docs/arch/06-runtime.md: "the host does not wait between them"), not the
             * dispatch+host-wait+dispatch round trip plow_hsa_launch+plow_hsa_wait would give if
             * called twice in a row (that would ADD a host round trip today's runtime never
             * pays, inflating this arm against production). Each dispatch gets its OWN
             * dedicated signal so both are independently profilable afterward. Always exactly
             * one GEMM copy — MQ_GLT_N only applies to 2q/2qm below. */
            raw_launch(q0, xr_karg0, xr_sig0, &kxr[0], xr_nwg * 512u, 512, &axr, sizeof axr, 1);
            raw_launch(q0, gemm_karg0, gemm_sig0, &kgemmlt, glt_grid * 256u, 256, &gargs_n[0],
                      sizeof(GltArgs), 1);
            for (int r = 1; r < NR; r++) {
                arg_xr a = {out[r], table[r], NR, (uint32_t)r, (uint32_t)n, 0u, XCTR_OFF, deadline, NULL, status[r]};
                plow_hsa_launch(h, dev[r], &kxr[r], xr_nwg * 512u, 1, 1, 512, 1, 1, 0, &a, sizeof a);
            }
            raw_wait(gemm_sig0);
            raw_wait(xr_sig0);
            for (int r = 1; r < NR; r++) plow_hsa_wait(h, dev[r]);
            hsa_amd_profiling_get_dispatch_time(agent0, xr_sig0, &txr);
            hsa_amd_profiling_get_dispatch_time(agent0, gemm_sig0, &tgm);
        } else { /* 2q, 2qm: the collective on q0, glt_n_fill independent GEMM copies on q2,
                  * all fired ahead before any wait — "how much independent work fits". */
            plow_hsa_launch(h, dev[0], &kxr[0], xr_nwg * 512u, 1, 1, 512, 1, 1, 0, &axr, sizeof axr);
            for (uint32_t i = 0; i < glt_n_fill; i++)
                raw_launch(q2.q, glt_karg_n[i], glt_sig_n[i], &kgemmlt, glt_grid * 256u, 256,
                          &gargs_n[i], sizeof(GltArgs), 0 /* no barrier */);
            for (int r = 1; r < NR; r++) {
                arg_xr a = {out[r], table[r], NR, (uint32_t)r, (uint32_t)n, 0u, XCTR_OFF, deadline, NULL, status[r]};
                plow_hsa_launch(h, dev[r], &kxr[r], xr_nwg * 512u, 1, 1, 512, 1, 1, 0, &a, sizeof a);
            }
            for (int r = 0; r < NR; r++) plow_hsa_wait(h, dev[r]);
            for (uint32_t i = 0; i < glt_n_fill; i++) raw_wait(glt_sig_n[i]);
            hsa_amd_profiling_get_dispatch_time(agent0, (hsa_signal_t){plow_hsa_done_signal_raw(h, dev[0])}, &txr);
            for (uint32_t i = 0; i < glt_n_fill; i++)
                hsa_amd_profiling_get_dispatch_time(agent0, glt_sig_n[i], &tgm_n[i]);
            tgm = tgm_n[0];
        }

        if (run_xr) { uint32_t st = 0; plow_hsa_download(h, dev[0], &st, status[0], 4); any_timeout |= st != 0; }

        /* HSA system clock ticks -> us (freq is HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, in Hz). */
        const double tick_us = freq ? 1e6 / (double)freq : 0.0;
        const double xr0 = (double)txr.start * tick_us, xr1 = (double)txr.end * tick_us;
        xr_dur[rep] = run_xr ? xr1 - xr0 : 0.0;

        if (glt_n_fill > 1 && (!strcmp(arm, "2q") || !strcmp(arm, "2qm"))) {
            /* Aggregate over all N copies: total busy time (sum of each's own duration — the
             * "how much independent work" number), the union window they occupy, and how much
             * of THAT window falls inside the collective's window (the "how much hides" number,
             * as a fraction of the collective's own duration). */
            double busy = 0.0, wlo = 1e30, whi = -1e30, ov = 0.0;
            for (uint32_t i = 0; i < glt_n_fill; i++) {
                const double s = (double)tgm_n[i].start * tick_us, e = (double)tgm_n[i].end * tick_us;
                busy += e - s;
                if (s < wlo) wlo = s;
                if (e > whi) whi = e;
                const double lo = xr0 > s ? xr0 : s, hi = xr1 < e ? xr1 : e;
                if (hi > lo) ov += hi - lo; /* sum, not union — copies rarely overlap each other much */
            }
            gemm_dur[rep] = busy / glt_n_fill; /* median-ish: mean per-copy, N is usually small */
            overlap[rep] = ov;
            const double glo = xr0 < wlo ? xr0 : wlo, ghi = xr1 > whi ? xr1 : whi;
            wall[rep] = ghi - glo;
            fill_busy[rep] = busy;
            fill_cover[rep] = run_xr && xr_dur[rep] > 0 ? 100.0 * ov / xr_dur[rep] : 0.0;
        } else {
            const double g0 = (double)tgm.start * tick_us, g1 = (double)tgm.end * tick_us;
            gemm_dur[rep] = g1 - g0;
            if (run_xr) {
                const double lo = xr0 > g0 ? xr0 : g0, hi = xr1 < g1 ? xr1 : g1;
                overlap[rep] = hi > lo ? hi - lo : 0.0;
                const double wlo = xr0 < g0 ? xr0 : g0, whi = xr1 > g1 ? xr1 : g1;
                wall[rep] = whi - wlo;
            } else {
                wall[rep] = gemm_dur[rep];
            }
            fill_busy[rep] = gemm_dur[rep];
            fill_cover[rep] = run_xr && xr_dur[rep] > 0 ? 100.0 * overlap[rep] / xr_dur[rep] : 0.0;
        }
    }

    plow_hsa_download(h, dev[0], rep_out, glt_out_n[0], glt_out_bytes);
    const int gemm_bad = memcmp(ref_out, rep_out, glt_out_bytes) != 0;

    int xr_bad = 0;
    if (run_xr) {
        bf16* host = malloc((size_t)n * 2u);
        bf16* want = malloc((size_t)n * 2u);
        for (uint64_t e = 0; e < n; e++) {
            float sum = 0.0f;
            for (int r = 0; r < NR; r++) sum += (float)(r + 1) * (1.0f + (float)(e & 7u) * 0.125f);
            uint32_t u; memcpy(&u, &sum, 4);
            u += 0x7fffu + ((u >> 16) & 1u);
            want[e] = (bf16)(u >> 16);
        }
        for (int r = 0; r < NR; r++) {
            plow_hsa_download(h, dev[r], host, out[r], (size_t)n * 2u);
            for (uint64_t e = 0; e < n; e++)
                if (host[e] != want[e]) { xr_bad = 1; break; }
            if (xr_bad) break;
        }
        free(host); free(want);
    }

    printf("arm=%s(gemmlt-real) shape=%s rows=%u hidden=%u xr_nwg=%u glt_m=%u glt_n=%u glt_k=%u "
           "glt_grid=%u glt_n_fill=%u reps=%u timeout=%s\n",
           arm, prefill_shape ? "prefill" : "decode", rows, hidden, xr_nwg, shp->m, shp->n, shp->k,
           glt_grid, glt_n_fill, reps, any_timeout ? "YES" : "no");
    printf("  xr_us=%.2f gemm_us=%.2f(per-copy) overlap_us=%.2f wall_us=%.2f "
           "fill_busy_us=%.2f fill_cover_pct=%.1f\n",
           med(xr_dur, reps), med(gemm_dur, reps), med(overlap, reps), med(wall, reps),
           med(fill_busy, reps), med(fill_cover, reps));
    printf("  xr_parity=%s gemmlt_parity=%s (bit-exact vs the solo reference, copy 0)\n",
           xr_bad ? "FAIL" : (run_xr ? "PASS" : "n/a"), gemm_bad ? "FAIL" : "PASS");
    plow_hsa_shutdown(h);
    return (any_timeout || xr_bad || gemm_bad) ? 1 : 0;
}
