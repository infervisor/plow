/* Concurrent peer-SDMA transport ceiling for a routed-MoE EP boundary. */
#define _POSIX_C_SOURCE 200809L
#include "../../../amd/hsa_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1.0e6;
}

static size_t slot(unsigned rank, unsigned peer, unsigned nranks) {
    (void)nranks;
    return peer < rank ? peer : peer - 1;
}

static int cmp_double(const void* a, const void* b) {
    const double x = *(const double*)a;
    const double y = *(const double*)b;
    return (x > y) - (x < y);
}

int main(int argc, char** argv) {
    const size_t chunk = argc > 1 ? strtoull(argv[1], NULL, 0) : 6473415u;
    plow_hsa* h = plow_hsa_init();
    if (!h) {
        fprintf(stderr, "hsa init: %s\n", plow_hsa_last_error());
        return 2;
    }
    const unsigned nranks = (unsigned)plow_hsa_device_count(h);
    if (nranks < 2 || nranks > 16) {
        fprintf(stderr, "need 2..16 GPUs, found %u\n", nranks);
        plow_hsa_shutdown(h);
        return 2;
    }
    const size_t slab_bytes = (nranks - 1) * chunk;
    void* send[16] = {0};
    void* recv[16] = {0};
    void* stage = plow_hsa_alloc_host(h, slab_bytes);
    plow_hsa_p2p_copy copies[16 * 15];
    if (!stage) {
        fprintf(stderr, "host slab: %s\n", plow_hsa_last_error());
        plow_hsa_shutdown(h);
        return 2;
    }
    for (unsigned r = 0; r < nranks; r++) {
        send[r] = plow_hsa_alloc_peer(h, (int)r, slab_bytes);
        recv[r] = plow_hsa_alloc_peer(h, (int)r, slab_bytes);
        if (!send[r] || !recv[r]) {
            fprintf(stderr, "rank %u slab: %s\n", r, plow_hsa_last_error());
            return 2;
        }
        memset(stage, (int)(r + 1), slab_bytes);
        if (plow_hsa_copy_h2d(h, (int)r, send[r], stage, slab_bytes) != 0) {
            fprintf(stderr, "rank %u upload: %s\n", r, plow_hsa_last_error());
            return 2;
        }
    }
    size_t ncopy = 0;
    for (unsigned src = 0; src < nranks; src++) {
        for (unsigned dst = 0; dst < nranks; dst++) {
            if (src == dst) continue;
            copies[ncopy++] = (plow_hsa_p2p_copy){
                .dst_dev = (int)dst,
                .dst = (char*)recv[dst] + slot(dst, src, nranks) * chunk,
                .src_dev = (int)src,
                .src = (const char*)send[src] + slot(src, dst, nranks) * chunk,
                .bytes = chunk,
            };
        }
    }
    for (unsigned i = 0; i < 3; i++) {
        if (plow_hsa_copy_p2p_batch(h, copies, ncopy) != 0) {
            fprintf(stderr, "warmup: %s\n", plow_hsa_last_error());
            return 2;
        }
    }
    double samples[12];
    for (unsigned i = 0; i < 12; i++) {
        const double t0 = now_ms();
        if (plow_hsa_copy_p2p_batch(h, copies, ncopy) != 0) {
            fprintf(stderr, "sample: %s\n", plow_hsa_last_error());
            return 2;
        }
        samples[i] = now_ms() - t0;
    }
    qsort(samples, 12, sizeof(samples[0]), cmp_double);
    unsigned errors = 0;
    for (unsigned dst = 0; dst < nranks; dst++) {
        for (unsigned src = 0; src < nranks; src++) {
            if (src == dst) continue;
            unsigned char got = 0;
            const void* at = (const char*)recv[dst] + slot(dst, src, nranks) * chunk;
            if (plow_hsa_download(h, (int)dst, &got, at, 1) != 0 || got != src + 1) errors++;
        }
    }
    const double median = (samples[5] + samples[6]) * 0.5;
    const double per_rank_gbs = (double)slab_bytes / (median * 1.0e6);
    printf("ranks=%u copies=%zu chunk=%zu bytes_per_rank=%zu median_ms=%.6f "
           "send_GBps_per_rank=%.3f errors=%u\n",
           nranks, ncopy, chunk, slab_bytes, median, per_rank_gbs, errors);
    for (unsigned r = 0; r < nranks; r++) {
        plow_hsa_free(h, send[r]);
        plow_hsa_free(h, recv[r]);
    }
    plow_hsa_free(h, stage);
    plow_hsa_shutdown(h);
    return errors ? 1 : 0;
}
