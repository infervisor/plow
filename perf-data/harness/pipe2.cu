#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static const char* P="/workspace/models/gemma-4-12B-it/model.safetensors";
static void evict(void){int fd=open(P,O_RDONLY);struct stat s;fstat(fd,&s);posix_fadvise(fd,0,s.st_size,POSIX_FADV_DONTNEED);close(fd);}
static void warmup(size_t off,size_t len){int fd=open(P,O_RDONLY);void*b=malloc(16<<20);
  for(size_t o=0;o<len;o+=(16<<20)) if(pread(fd,b,16<<20,off+o)<0)break; free(b);close(fd);}
static double resident(size_t off,size_t len){
  int fd=open(P,O_RDONLY);void*m=mmap(NULL,len,PROT_READ,MAP_PRIVATE,fd,off);
  if(m==MAP_FAILED){close(fd);return -1;}
  size_t np=(len+4095)/4096;unsigned char*v=(unsigned char*)malloc(np);
  mincore(m,len,v);size_t in=0;for(size_t i=0;i<np;i++)in+=v[i]&1;
  munmap(m,len);free(v);close(fd);return (double)in/np;}

#define MAXSLOT 16
// per-slot handshake: slot i holds block (b) with b%NSLOT==i, consumed in order
static void* g_h[MAXSLOT]; static size_t g_n[MAXSLOT];
static sem_t s_empty[MAXSLOT], s_full[MAXSLOT];
static int NSLOT;
static int g_fd; static size_t g_off,g_len,g_ch; static size_t g_nblk;
static size_t g_next; static pthread_mutex_t g_mu=PTHREAD_MUTEX_INITIALIZER;

static void* reader(void*arg){
  (void)arg;
  for(;;){
    pthread_mutex_lock(&g_mu);
    size_t b=g_next; if(b>=g_nblk){pthread_mutex_unlock(&g_mu);break;} g_next++;
    pthread_mutex_unlock(&g_mu);
    size_t o=b*g_ch, c=g_len-o<g_ch?g_len-o:g_ch;
    int s=b%NSLOT;
    sem_wait(&s_empty[s]);                 // wait for consumer to free THIS slot
    ssize_t n=pread(g_fd,g_h[s],c,g_off+o);
    g_n[s]= n>0?n:0;
    sem_post(&s_full[s]);
  }
  return NULL;
}

static void run(size_t off,size_t len,size_t ch,int nthr,int nstream,int nslot,int cold,const char* tag){
  if(cold) evict(); else warmup(off,len);
  double r=resident(off,len);
  NSLOT=nslot;
  g_fd=open(P,O_RDONLY); g_off=off; g_len=len; g_ch=ch; g_next=0;
  g_nblk=(len+ch-1)/ch;
  for(int i=0;i<NSLOT;i++){ CK(cudaMallocHost(&g_h[i],ch)); sem_init(&s_empty[i],0,1); sem_init(&s_full[i],0,0); }
  void* d; CK(cudaMalloc(&d,ch*(size_t)NSLOT));
  cudaStream_t st[8]; cudaEvent_t ev[MAXSLOT];
  for(int i=0;i<nstream;i++) CK(cudaStreamCreate(&st[i]));
  for(int i=0;i<NSLOT;i++) CK(cudaEventCreateWithFlags(&ev[i],cudaEventDisableTiming|cudaEventBlockingSync));
  pthread_t th[32];
  double t0=now();
  for(int i=0;i<nthr;i++) pthread_create(&th[i],NULL,reader,NULL);
  size_t got=0;
  for(size_t b=0;b<g_nblk;b++){
    int s=b%NSLOT;
    if(b>=(size_t)NSLOT){                  // reclaim slot used NSLOT blocks ago
      CK(cudaEventSynchronize(ev[s]));
      sem_post(&s_empty[s]);
    }
    sem_wait(&s_full[s]);
    CK(cudaMemcpyAsync((char*)d+(size_t)s*ch, g_h[s], g_n[s], cudaMemcpyHostToDevice, st[b%nstream]));
    CK(cudaEventRecord(ev[s], st[b%nstream]));
    got+=g_n[s];
  }
  CK(cudaDeviceSynchronize());
  for(size_t b=g_nblk>=(size_t)NSLOT?g_nblk-NSLOT:0;b<g_nblk;b++) sem_post(&s_empty[b%NSLOT]);
  for(int i=0;i<nthr;i++) pthread_join(th[i],NULL);
  double dt=now()-t0;
  printf("  %-5s thr=%-2d str=%d slot=%-2d ch=%2zuMiB res=%5.1f%%  %.2f GiB %6.3fs = %6.2f GB/s\n",
    tag,nthr,nstream,nslot,ch>>20,r*100,(double)got/(1<<30),dt,(double)got/dt/1e9);
  for(int i=0;i<NSLOT;i++){cudaFreeHost(g_h[i]);cudaEventDestroy(ev[i]);sem_destroy(&s_empty[i]);sem_destroy(&s_full[i]);}
  cudaFree(d); for(int i=0;i<nstream;i++) cudaStreamDestroy(st[i]);
  close(g_fd);
}
int main(void){
  CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)); CK(cudaSetDevice(0));
  size_t LEN=4ull<<30;
  printf("[pipelined: pread(N thr) -> pinned ring(slots) -> async H2D, overlap via events]\n");
  printf("-- WARM cache (page-cache source) : isolates PCIe+memcpy --\n");
  run(0,LEN,32<<20,1,2,8,0,"warm");
  run(0,LEN,32<<20,2,2,8,0,"warm");
  run(0,LEN,32<<20,4,2,8,0,"warm");
  run(0,LEN,32<<20,8,4,8,0,"warm");
  run(0,LEN,64<<20,4,2,8,0,"warm");
  printf("-- COLD cache (real disk) : isolates NVMe --\n");
  run(0ull<<30,LEN,32<<20,1,2,8,1,"cold");
  run(4ull<<30,LEN,32<<20,4,2,8,1,"cold");
  run(8ull<<30,LEN,32<<20,8,4,8,1,"cold");
  run(12ull<<30,LEN,32<<20,16,4,16,1,"cold");
  return 0;}
