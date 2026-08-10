/* Exhaustive value-identity gate for the branchless f2bf: all 2^32 float bit patterns.
 * Pure integer arithmetic, so the CPU is the decisive check on the ALGORITHM. */
#include <stdio.h>
#include <string.h>
typedef unsigned short bf16; typedef unsigned u32;
static inline bf16 shipped(u32 u) {
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
static inline bf16 sel(u32 u) {
    const u32 rne = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    const u32 qnan = (u >> 16) | 0x0040u;
    return (bf16)(((u & 0x7fffffffu) > 0x7f800000u) ? qnan : rne);
}
int main(void) {
    unsigned long long bad = 0, n = 0; u32 first = 0;
    for (unsigned long long v = 0; v <= 0xffffffffull; v++) {
        u32 u = (u32)v; bf16 a = shipped(u), b = sel(u);
        if (a != b) { if (!bad) first = u; bad++; }
        n++;
    }
    printf("checked %llu patterns, %llu mismatches", n, bad);
    if (bad) printf(" (first at 0x%08x: shipped=0x%04x sel=0x%04x)", first, shipped(first), sel(first));
    printf(" -> %s\n", bad ? "FAIL" : "VALUE-IDENTICAL");
    /* falsification control: a DELIBERATELY wrong variant must be caught */
    unsigned long long wrongbad = 0;
    for (unsigned long long v = 0; v <= 0xffffffffull; v += 7) {
        u32 u = (u32)v;
        bf16 wrong = (bf16)((u + 0x8000u) >> 16);  /* plain round-half-up, no RNE, no NaN guard */
        if (wrong != shipped(u)) wrongbad++;
    }
    printf("control (round-half-up, sampled 1/7): %llu mismatches -> gate %s\n",
           wrongbad, wrongbad ? "CAN FAIL" : "IS VACUOUS");
    return bad != 0;
}
