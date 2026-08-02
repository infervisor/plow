// Measures the host side of plowrt's checkpoint upload path on a real
// safetensors file, with no GPU involved.
//
//   mode A ("serial"): one thread, 64 MiB chunks, mmap -> memcpy into a staging
//                      buffer. This is exactly what UploadPipe::push does minus
//                      the cuMemcpyHtoDAsync (gpu.rs:548).
//   mode B ("pool"):   N threads issuing MADV_POPULATE_READ over disjoint 64 MiB
//                      spans, then one serial memcpy pass over now-resident
//                      pages. This is what asset/checkpoint.rs::Prefetcher does
//                      on the AMD path.
//
// usage: loadbench <file> <serial|pool> [threads]

#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

#define STAGE (64UL << 20)
/// Size of the worker array below; `argv[3]` is clamped to it.
#define MAX_THREADS 256

static char *g_map;
static size_t g_len;
static int g_nthreads;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

// Each worker takes every nthreads'th 64 MiB span, mirroring the bounded
// round-robin the Prefetcher pool ends up with.
static void *worker(void *arg) {
  long id = (long)arg;
  for (size_t off = (size_t)id * STAGE; off < g_len; off += STAGE * g_nthreads) {
    size_t n = g_len - off < STAGE ? g_len - off : STAGE;
    madvise(g_map + off, n, MADV_POPULATE_READ);
  }
  return NULL;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <file> <serial|pool> [threads]\n", argv[0]);
    return 2;
  }
  const char *path = argv[1];
  int pool = strcmp(argv[2], "pool") == 0;
  g_nthreads = argc > 3 ? atoi(argv[3]) : 16;
  // atoi gives back whatever was typed, including junk and values past the
  // worker array — clamp before it indexes anything.
  if (g_nthreads < 1) g_nthreads = 1;
  if (g_nthreads > MAX_THREADS) {
    fprintf(stderr, "clamping threads %d -> %d\n", g_nthreads, MAX_THREADS);
    g_nthreads = MAX_THREADS;
  }

  int fd = open(path, O_RDONLY);
  if (fd < 0) { perror("open"); return 1; }
  struct stat st;
  if (fstat(fd, &st)) { perror("fstat"); return 1; }
  g_len = st.st_size;
  g_map = mmap(NULL, g_len, PROT_READ, MAP_PRIVATE, fd, 0);
  if (g_map == MAP_FAILED) { perror("mmap"); return 1; }

  // Page-locked staging is irrelevant to the read side; a plain buffer keeps
  // this benchmark GPU-free while preserving the memcpy cost.
  char *stage = aligned_alloc(4096, STAGE);
  if (!stage) { perror("aligned_alloc"); return 1; }

  double t0 = now();

  if (pool) {
    pthread_t th[MAX_THREADS];
    for (long i = 0; i < g_nthreads; i++) pthread_create(&th[i], NULL, worker, (void *)i);
    for (long i = 0; i < g_nthreads; i++) pthread_join(th[i], NULL);
  }

  // The serial copy pass happens in both modes: in "pool" the pages are already
  // resident so it measures memcpy alone; in "serial" it also pays every fault.
  double t_pop = now();
  // `sink` is volatile so the compiler cannot prove the staging buffer is dead
  // and delete the memcpy — which is the very thing being measured, and which
  // it does at -O2 without this.
  static volatile unsigned char sink;
  for (size_t off = 0; off < g_len; off += STAGE) {
    size_t n = g_len - off < STAGE ? g_len - off : STAGE;
    memcpy(stage, g_map + off, n);
    sink = stage[n - 1];
  }
  double t1 = now();

  double gib = g_len / (double)(1UL << 30);
  printf("mode=%s threads=%d bytes=%.2f GiB populate=%.2f s copy=%.2f s total=%.2f s "
         "=> %.2f GiB/s\n",
         pool ? "pool" : "serial", pool ? g_nthreads : 1, gib, t_pop - t0, t1 - t_pop,
         t1 - t0, gib / (t1 - t0));
  return 0;
}
