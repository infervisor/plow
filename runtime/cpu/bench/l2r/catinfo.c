/* catinfo.c: CPUID leaf 0x10 (Intel RDT allocation): L3 / L2 CAT mask width, shareable bits, classes. Read-only. */
#include <cpuid.h>
#include <stdio.h>
int main(void) {
    unsigned a, b, c, d;
    __cpuid_count(0x10, 0, a, b, c, d);
    printf("RDT-A resources: L3 %u, L2 %u, MBA %u\n", (b >> 1) & 1, (b >> 2) & 1, (b >> 3) & 1);
    for (unsigned sub = 1; sub <= 2; sub++) {
        if (!((b >> sub) & 1)) continue;
        unsigned a2, b2, c2, d2;
        __cpuid_count(0x10, sub, a2, b2, c2, d2);
        printf("%s CAT: mask length %u bits, shareable-with-other-agents bits 0x%x, classes (CLOS) %u, CDP %s\n",
               sub == 1 ? "L3" : "L2", (a2 & 0x1f) + 1, b2, (d2 & 0xffff) + 1, (c2 >> 2) & 1 ? "yes" : "no");
    }
    __cpuid_count(4, 2, a, b, c, d); /* deterministic cache params, index 2 = L2 on Intel */
    printf("L2: %u ways, %u sets, line %u B -> %u KiB per way\n", ((b >> 22) & 0x3ff) + 1, c + 1, (b & 0xfff) + 1,
           (((b & 0xfff) + 1) * (c + 1)) / 1024);
    return 0;
}
