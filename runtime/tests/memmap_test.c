/* memmap_test — the runtime rebase resolves logical slots to arena_base+offset,
 * and rejects out-of-range slots / offsets. */
#include "memmap.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fail = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); g_fail = 1; } } while (0)

int main(void) {
    /* Mirror of a compiler map: a weight, a scratch buffer, and a growable KV. */
    PlowMemEntry entries[3] = {
        { /*slot*/0, PLOW_BUF_PERSISTENT, 0, {0,0}, /*off*/0,     /*reserved*/256 },
        { /*slot*/1, PLOW_BUF_SCRATCH,    0, {0,0}, /*off*/256,   /*reserved*/512 },
        { /*slot*/2, PLOW_BUF_GROWABLE,   1, {0,0}, /*off*/768,   /*reserved*/1024 },
    };
    PlowMemMap map = { entries, 3, /*arena_bytes*/768 + 1024, /*growable_base*/768 };

    CHECK(plow_memmap_n_slots(&map) == 3, "n_slots");

    uint32_t n = plow_memmap_n_slots(&map);
    void** tensors = (void**)calloc(n, sizeof(void*));
    /* A fake arena; we only compare pointer arithmetic, never dereference. */
    uint8_t* arena = (uint8_t*)0x100000;

    uint32_t w = plow_memmap_resolve(&map, arena, tensors, n);
    CHECK(w == 3, "resolved all slots");
    CHECK(tensors[0] == arena + 0,   "slot 0 = base+0");
    CHECK(tensors[1] == arena + 256, "slot 1 = base+256");
    CHECK(tensors[2] == arena + 768, "slot 2 = base+768 (growable_base)");

    /* Out-of-range slot capacity → reject. */
    void* small[1] = { 0 };
    CHECK(plow_memmap_resolve(&map, arena, small, 1) == 0, "reject undersized tensors[]");

    /* Offset past arena → reject. */
    PlowMemEntry bad[1] = { { 0, PLOW_BUF_SCRATCH, 0, {0,0}, /*off*/2000, /*reserved*/100 } };
    PlowMemMap badmap = { bad, 1, /*arena_bytes*/1000, 0 };
    void* one[1] = { 0 };
    CHECK(plow_memmap_resolve(&badmap, arena, one, 1) == 0, "reject offset past arena");

    /* Operand views: a contiguous sub-view = base + byte_off (no copy). */
    PlowOperand whole = { /*slot*/1, /*byte_off*/0, 0, 0, {0,0}, {0} };
    CHECK(plow_operand_ptr(&whole, tensors, n) == (uint8_t*)tensors[1], "operand whole-buffer");
    PlowOperand view = { /*slot*/1, /*byte_off*/128, 0, 0, {0,0}, {0} };
    CHECK(plow_operand_ptr(&view, tensors, n) == (uint8_t*)tensors[1] + 128, "operand offset view");
    PlowOperand oob = { /*slot*/99, 0, 0, 0, {0,0}, {0} };
    CHECK(plow_operand_ptr(&oob, tensors, n) == NULL, "operand rejects out-of-range slot");

    free(tensors);
    printf("memmap_test: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail;
}
