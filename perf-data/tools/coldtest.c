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
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static const char* P="/workspace/models/gemma-4-12B-it/model.safetensors";

// fraction of [off,off+len) resident in page cache
static double resident(size_t off,size_t len){
  int fd=open(P,O_RDONLY); struct stat s; fstat(fd,&s);
  size_t pg=4096; size_t aoff=off&~(pg-1);
  void* m=mmap(NULL,len,PROT_READ,MAP_PRIVATE,fd,aoff);
  if(m==MAP_FAILED){close(fd);return -1;}
  size_t np=(len+pg-1)/pg;
  unsigned char* v=malloc(np);
  mincore(m,len,v);
  size_t in=0; for(size_t i=0;i<np;i++) in+=v[i]&1;
  munmap(m,len); free(v); close(fd);
  return (double)in/np;
}
static void evict(void){
  int fd=open(P,O_RDONLY); struct stat s; fstat(fd,&s);
  posix_fadvise(fd,0,s.st_size,POSIX_FADV_DONTNEED); close(fd);
}
int main(void){
  size_t LEN=7ull<<30;
  // three disjoint 8GiB-ish windows in the 23GB file
  size_t offs[3]={0, 7ull<<30, 14ull<<30};
  for(int k=0;k<3;k++){
    size_t off=offs[k];
    evict();
    double r0=resident(off,LEN);
    int fd=open(P,O_RDONLY);
    void* m=mmap(NULL,LEN,PROT_READ,MAP_PRIVATE,fd,off);
    volatile char* dst=malloc(16<<20); unsigned long long sink=0;
    double t0=now();
    for(size_t o=0;o<LEN;o+=(16<<20)){ memcpy((void*)dst,(char*)m+o,16<<20); sink+=dst[0]; }
    double dt=now()-t0;
    double r1=resident(off,LEN);
    printf("window@%2zuGiB  resident_before=%5.1f%%  mmap+memcpy 7GiB in %6.3fs = %6.2f GB/s  resident_after=%5.1f%% (sink=%llu)\n",
      off>>30, r0*100, dt, (double)LEN/dt/1e9, r1*100, sink&1);
    munmap(m,LEN); free((void*)dst); close(fd);
  }
  printf("\n-- now WARM: re-read window@0 without evicting --\n");
  { size_t off=0; double r0=resident(off,LEN);
    int fd=open(P,O_RDONLY);
    void* m=mmap(NULL,LEN,PROT_READ,MAP_PRIVATE,fd,off);
    volatile char* dst=malloc(16<<20); unsigned long long sink=0;
    // prefault
    for(size_t o=0;o<LEN;o+=(16<<20)) { memcpy((void*)dst,(char*)m+o,16<<20); sink+=dst[0]; }
    double rr=resident(off,LEN);
    double t0=now();
    for(size_t o=0;o<LEN;o+=(16<<20)){ memcpy((void*)dst,(char*)m+o,16<<20); sink+=dst[0]; }
    double dt=now()-t0;
    printf("window@ 0GiB  resident_before=%5.1f%% (after prefault %5.1f%%)  = %6.2f GB/s (sink=%llu)\n",
      r0*100, rr*100, (double)LEN/dt/1e9, sink&1);
  }
  return 0;
}
