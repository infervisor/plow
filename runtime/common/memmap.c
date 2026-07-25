/* memmap.c — resolve the compiler address map onto an allocated arena. */
#include "memmap.h"

uint32_t plow_memmap_n_slots(const PlowMemMap* map) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < map->n_entries; i++)
        if (map->entries[i].slot + 1 > n) n = map->entries[i].slot + 1;
    return n;
}

uint32_t plow_memmap_resolve(const PlowMemMap* map, void* arena_base,
                             void** tensors, uint32_t n_tensors) {
    uint8_t* base = (uint8_t*)arena_base;
    uint32_t written = 0;
    for (uint32_t i = 0; i < map->n_entries; i++) {
        const PlowMemEntry* e = &map->entries[i];
        if (e->slot >= n_tensors) return 0;
        if (e->offset + e->reserved > map->arena_bytes) return 0;
        tensors[e->slot] = base + e->offset;
        written++;
    }
    return written;
}

void* plow_operand_ptr(const PlowOperand* op, void* const* tensors, uint32_t n_tensors) {
    if (op->slot >= n_tensors) return NULL;
    uint8_t* base = (uint8_t*)tensors[op->slot];
    if (!base) return NULL;
    return base + op->byte_off;
}
