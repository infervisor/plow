#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static const char* PATH_ = "/workspace/models/gemma-4-12B-it/model.safetensors";
static size_t NBYTES = 8ull<<30;   // 8 GiB
static size_t CHUNK  = 16ull<<20;  // 16 MiB

// drop this file's clean page cache (works unprivileged)
static void evict(const char* p){
  int fd = open(p, O_RDONLY);
  if (fd<0) return;
  struct stat s; fstat(fd,&s);
  fdatasync(fd);
  posix_fadvise(fd, 0, s.st_size, POSIX_FADV_DONTNEED);
  close(fd);
}

typedef struct { int fd; size_t off, len, chunk; void* buf; } job_t;
static void* worker(void* a){
  job_t* j = (job_t*)a;
  size_t done=0;
  while(done < j->len){
    size_t n = j->len-done < j->chunk ? j->len-done : j->chunk;
    ssize_t r = pread(j->fd, j->buf, n, j->off+done);
    if (r<=0){ fprintf(stderr,"pread fail off=%zu n=%zu: %s\n", j->off+done,n,strerror(errno)); break; }
    done += r;
  }
  j->len = done;
  return NULL;
}

// O_DIRECT sequential read with T threads, QD via chunk size
static double odirect_read(int T, size_t chunk, size_t total, int do_evict){
  if (do_evict) evict(PATH_);
  int fd = open(PATH_, O_RDONLY|O_DIRECT);
  if (fd<0){ fprintf(stderr,"O_DIRECT open failed: %s\n", strerror(errno)); return -1; }
  pthread_t th[64]; job_t j[64];
  size_t per = (total/T) & ~(size_t)4095;
  for(int i=0;i<T;i++){
    j[i].fd=fd; j[i].off=i*per; j[i].len=per; j[i].chunk=chunk;
    if (posix_memalign(&j[i].buf, 4096, chunk)) { fprintf(stderr,"memalign\n"); return -1; }
  }
  double t0=now();
  for(int i=0;i<T;i++) pthread_create(&th[i],NULL,worker,&j[i]);
  size_t got=0;
  for(int i=0;i<T;i++){ pthread_join(th[i],NULL); got+=j[i].len; }
  double dt=now()-t0;
  for(int i=0;i<T;i++) free(j[i].buf);
  close(fd);
  printf("  O_DIRECT T=%-2d chunk=%4zuMiB  %6.2f GiB in %6.3fs = %7.2f GB/s\n",
    T, chunk>>20, (double)got/(1<<30), dt, (double)got/dt/1e9);
  return (double)got/dt/1e9;
}

// buffered pread (page cache path)
static double buffered_read(int T, size_t chunk, size_t total, int do_evict, const char* tag){
  if (do_evict) evict(PATH_);
  int fd = open(PATH_, O_RDONLY);
  if (fd<0) return -1;
  pthread_t th[64]; job_t j[64];
  size_t per = total/T;
  for(int i=0;i<T;i++){
    j[i].fd=fd; j[i].off=i*per; j[i].len=per; j[i].chunk=chunk;
    j[i].buf = malloc(chunk);
  }
  double t0=now();
  for(int i=0;i<T;i++) pthread_create(&th[i],NULL,worker,&j[i]);
  size_t got=0;
  for(int i=0;i<T;i++){ pthread_join(th[i],NULL); got+=j[i].len; }
  double dt=now()-t0;
  for(int i=0;i<T;i++) free(j[i].buf);
  close(fd);
  printf("  buffered  T=%-2d chunk=%4zuMiB  %-6s %6.2f GiB in %6.3fs = %7.2f GB/s\n",
    T, chunk>>20, tag, (double)got/(1<<30), dt, (double)got/dt/1e9);
  return (double)got/dt/1e9;
}

// mmap + memcpy
static double mmap_read(size_t total, int do_evict, const char* tag){
  if (do_evict) evict(PATH_);
  int fd = open(PATH_, O_RDONLY);
  struct stat s; fstat(fd,&s);
  size_t n = total < (size_t)s.st_size ? total : (size_t)s.st_size;
  void* m = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
  if (m==MAP_FAILED){ fprintf(stderr,"mmap: %s\n",strerror(errno)); return -1; }
  volatile char* dst = malloc(CHUNK);
  unsigned long long sink=0;
  double t0=now();
  for(size_t off=0; off<n; off+=CHUNK){
    size_t c = n-off<CHUNK?n-off:CHUNK;
    memcpy((void*)dst, (char*)m+off, c);
    sink += dst[0] + dst[c-1];
  }
  double dt=now()-t0;
  if(sink==0xdeadbeef) printf("");
  munmap(m,n); free((void*)dst); close(fd);
  printf("  mmap+memcpy            %-6s %6.2f GiB in %6.3fs = %7.2f GB/s\n",
    tag, (double)n/(1<<30), dt, (double)n/dt/1e9);
  return (double)n/dt/1e9;
}

int main(int argc,char**argv){
  if (argc>1) NBYTES = strtoull(argv[1],0,10)<<30;
  printf("file=%s  total=%zu GiB\n", PATH_, NBYTES>>30);
  printf("\n[NVMe sequential read ceiling — O_DIRECT, page cache BYPASSED (true disk)]\n");
  odirect_read(1,  16<<20, NBYTES, 1);
  odirect_read(2,  16<<20, NBYTES, 1);
  odirect_read(4,  16<<20, NBYTES, 1);
  odirect_read(8,  16<<20, NBYTES, 1);
  odirect_read(16, 16<<20, NBYTES, 1);
  odirect_read(8,   4<<20, NBYTES, 1);
  odirect_read(8,  64<<20, NBYTES, 1);

  printf("\n[buffered pread — COLD (fadvise DONTNEED first) = disk + page-cache overhead]\n");
  buffered_read(1, 16<<20, NBYTES, 1, "cold");
  buffered_read(4, 16<<20, NBYTES, 1, "cold");
  buffered_read(8, 16<<20, NBYTES, 1, "cold");

  printf("\n[buffered pread — WARM (same range re-read, in page cache) = RAM, NOT disk]\n");
  buffered_read(1, 16<<20, NBYTES, 0, "warm");
  buffered_read(8, 16<<20, NBYTES, 0, "warm");

  printf("\n[mmap+memcpy]\n");
  mmap_read(NBYTES, 1, "cold");
  mmap_read(NBYTES, 0, "warm");
  return 0;
}
