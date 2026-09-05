/* isa_detect.c — cpuid tiering, AMX permission, process/thread init. */
/* syscall() is a GNU extension: the cc build compiles -std=c11, not gnu11. */
#define _GNU_SOURCE
#include "cpu_dev_internal.h"
#include <errno.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#define PLOW_X86 1
#else
#define PLOW_X86 0
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

/* Linux: request XTILEDATA before any AMX instruction, or the first one SIGILLs. */
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

static int g_isa = -1;

#if PLOW_X86
static uint64_t xgetbv0(void) {
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}
#endif

static int detect_isa(void) {
#if PLOW_X86
    uint32_t a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return PLOW_CPU_ISA_SCALAR;
    const int osxsave = (c >> 27) & 1;
    if (!osxsave) return PLOW_CPU_ISA_SCALAR;
    const uint64_t xcr0 = xgetbv0();
    /* AVX-512 state: opmask (5), ZMM_Hi256 (6), Hi16_ZMM (7). */
    if ((xcr0 & 0xE0u) != 0xE0u) return PLOW_CPU_ISA_SCALAR;
    if (__get_cpuid_max(0, NULL) < 7) return PLOW_CPU_ISA_SCALAR;
    __cpuid_count(7, 0, a, b, c, d);
    const int f = (b >> 16) & 1, bw = (b >> 30) & 1, vl = (b >> 31) & 1;
    const int amx_bf16 = (d >> 22) & 1, amx_tile = (d >> 24) & 1;
    uint32_t a1, b1, c1, d1;
    __cpuid_count(7, 1, a1, b1, c1, d1);
    const int bf16 = (a1 >> 5) & 1;
    if (!(f && bw && vl && bf16)) return PLOW_CPU_ISA_SCALAR;
    if (!(amx_tile && amx_bf16)) return PLOW_CPU_ISA_AVX512;
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0)
        return PLOW_CPU_ISA_AVX512;
#endif
    /* XTILECFG (17) and XTILEDATA (18) must be enabled in XCR0 after the request. */
    if ((xgetbv0() & (3ull << 17)) != (3ull << 17)) return PLOW_CPU_ISA_AVX512;
    return PLOW_CPU_ISA_AMX;
#else
    return PLOW_CPU_ISA_SCALAR;
#endif
}

int plow_cpu_init(int isa_cap) {
    if (isa_cap < PLOW_CPU_ISA_SCALAR) return -EINVAL;
    if (g_isa >= 0) return g_isa;
    int isa = detect_isa();
    if (isa > isa_cap) isa = isa_cap;
    plow_cpu_table_reset();
    plow_cpu_register_golden(plow_cpu_table());
    /* AVX-512 / AMX registrars override entries here once those tiers exist. */
    g_isa = isa;
    return g_isa;
}

int plow_cpu_isa(void) { return g_isa; }

int plow_cpu_thread_init(PlowCpuCtx* ctx) {
    if (!ctx) return -EINVAL;
    if (g_isa < 0) return -EAGAIN;
    ctx->isa = (uint32_t)g_isa;
    return 0;
}

uint32_t plow_cpu_scratch_bytes(void) { return PLOW_CPU_SCRATCH_BYTES; }
