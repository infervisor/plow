/* safetensors.h — mmap a checkpoint's shards and resolve a tensor's byte range.
 *
 * Extracted VERBATIM from gemma4_chat.c so the CUDA and HSA harnesses share one
 * loader, with the two documented defects fixed here (once) rather than in each
 * copy:
 *
 *   1. MAX_SHARD was 8. Gemma-31B ships more than 8 shards, and the probe loop
 *      simply found no candidate `-of-000NN` and returned "no safetensors" --
 *      a checkpoint-not-found error for a checkpoint that is right there. 64.
 *
 *   2. The single-file `model.safetensors` fallback was DOCUMENTED at the old
 *      :142-143 ("A single-file checkpoint is handled as the last fallback")
 *      and never implemented at :144-163. Qwen3-4B has 3 shards so it worked by
 *      accident; Gemma-12B is a single file and did not load at all. Implemented
 *      below as the genuine last resort, after sharded probing fails.
 */
#ifndef PLOW_SAFETENSORS_H
#define PLOW_SAFETENSORS_H

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> /* strtoull — the header must stand alone, not inherit it from
                     * its includer the way gemma4_chat.c happened to provide it */
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Gemma-31B exceeds 8. This is a probe bound, not a per-file cost. */
#define MAX_SHARD 64

typedef struct {
    int n;
    uint8_t* base[MAX_SHARD];
    char* hdr[MAX_SHARD];
    size_t hdr_len[MAX_SHARD];
    uint64_t data0[MAX_SHARD];
} Safet;

/* mmap one .safetensors file and record its JSON header span. */
static int st_map_one(Safet* s, const char* path) {
    if (s->n >= MAX_SHARD) return 1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 1;
    struct stat st;
    if (fstat(fd, &st) || (size_t)st.st_size < 8) { close(fd); return 1; }
    uint8_t* m = (uint8_t*)mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return 1;
    uint64_t hn;
    memcpy(&hn, m, 8);
    /* A truncated/garbage header would send st_find scanning off the mapping. */
    if (hn + 8 > (uint64_t)st.st_size) { munmap(m, (size_t)st.st_size); return 1; }
    s->base[s->n] = m;
    s->hdr[s->n] = (char*)(m + 8);
    s->hdr_len[s->n] = (size_t)hn;
    s->data0[s->n] = 8 + hn;
    s->n++;
    return 0;
}

static int st_open(Safet* s, const char* dir) {
    s->n = 0;
    /* The shard count is embedded in every filename ("-of-000NN") and it VARIES:
     * Gemma-31B has many, Llama-3.1-8B has 4, Qwen3-4B has 3. Probe the total by
     * trying each candidate for shard 1; the first that exists gives the count. */
    int total = 0;
    for (int cand = 1; cand <= MAX_SHARD; cand++) {
        char p[512];
        snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, 1, cand);
        if (access(p, R_OK) == 0) { total = cand; break; }
    }
    for (int i = 1; total && i <= total; i++) {
        char p[512];
        snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, i, total);
        if (st_map_one(s, p)) break;
    }
    /* FIX 2: the documented-but-absent single-file fallback. Only reached when
     * sharded probing found nothing, so a sharded checkpoint is unaffected. */
    if (!s->n) {
        char p[512];
        snprintf(p, sizeof(p), "%s/model.safetensors", dir);
        if (access(p, R_OK) == 0) st_map_one(s, p);
    }
    return s->n ? 0 : 1;
}

static const uint8_t* st_find(Safet* s, const char* name, uint64_t* nb) {
    char key[256];
    int kl = snprintf(key, sizeof(key), "\"%s\":", name);
    for (int i = 0; i < s->n; i++) {
        const char* h = s->hdr[i];
        const char* end = h + s->hdr_len[i];
        const char* p = NULL;
        for (const char* c = h; c + kl <= end; c++)
            if (!memcmp(c, key, (size_t)kl)) { p = c + kl; break; }
        if (!p) continue;
        const char* d = strstr(p, "\"data_offsets\":[");
        if (!d || d > end) continue;
        d += strlen("\"data_offsets\":[");
        unsigned long long a = strtoull(d, (char**)&d, 10);
        d++;
        unsigned long long e = strtoull(d, (char**)&d, 10);
        *nb = (uint64_t)(e - a);
        return s->base[i] + s->data0[i] + a;
    }
    return NULL;
}

#endif /* PLOW_SAFETENSORS_H */
