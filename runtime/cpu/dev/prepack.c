/* prepack.c — load-time weight relayouts. */
#include "cpu_dev.h"

size_t plow_cpu_prepack_bf16_b_bytes(uint32_t n, uint32_t k) { return (size_t)n * k * 2u; }

/* dst[nb][kb][kp][nn][p] = W[nb*16 + nn][kb*32 + 2*kp + p]: one AMX B tile per (nb, kb) —
 * 16 rows of K-pairs, each row 16 columns x 2 halves = 64 bytes, as TDPBF16PS consumes it. */
int plow_cpu_prepack_bf16_b(void* dst, const void* src, uint32_t n, uint32_t k) {
    if (!dst || !src || (n & 15u) || (k & 31u)) return -1;
    const plow_bf16* w = (const plow_bf16*)src;
    plow_bf16* o = (plow_bf16*)dst;
    const uint32_t nb_n = n / 16, nb_k = k / 32;
    for (uint32_t nb = 0; nb < nb_n; nb++)
        for (uint32_t kb = 0; kb < nb_k; kb++) {
            plow_bf16* tile = o + ((size_t)nb * nb_k + kb) * 512u;
            for (uint32_t kp = 0; kp < 16; kp++)
                for (uint32_t nn = 0; nn < 16; nn++) {
                    const plow_bf16* wr = w + (size_t)(nb * 16 + nn) * k + kb * 32 + 2 * kp;
                    tile[kp * 32 + nn * 2 + 0] = wr[0];
                    tile[kp * 32 + nn * 2 + 1] = wr[1];
                }
        }
    return 0;
}
