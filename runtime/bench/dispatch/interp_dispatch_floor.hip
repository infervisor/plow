// interp_dispatch_floor.hip — DECOMPOSE plow's per-op dispatch floor (the ~4.6us 1-CU op wall
// the GLM-5.2-FP8 decode autopsy isolated: 164us/layer dead-air ~= 37 ops x 4.6us).
//
// Faithfully reproduces interp.hip's per-op dispatch structure for a CHAIN of near-empty ops:
//   WIDE op   = 256 workgroups (a GEMV-shaped producer), each signals counter[k] (+1), threshold 256
//   NARROW op = 1 workgroup (WG0), the RmsNorm/Residual-shaped 1-CU consumer that sets the floor.
// The narrow op is gated on the wide producer (cross-XCD, threshold 256) exactly like decode's
// norm/residual gated on the preceding all-CU GEMV. It uses interp.hip's EXACT primitive lowering
// (relaxed-agent ctr_poll + one agent-scope acquire fence + release-agent ctr_signal; PLOW_CTR_STRIDE
// counters; 512 threads; 1 block/CU forced via 90KB LDS so all 8 XCDs light up). Each participating
// (WG,op) records t_arrive/t_ready/t_end via s_memrealtime (the clock the autopsy used, ~100 MHz).
//
//   Per-op WALL = t_ready -> t_end     (body + syncthreads + succ-signal)
//   Per-op GAP  = t_arrive -> t_ready  (gate poll cross-XCD + acquire buffer_inv)
//
// MEASURED DECOMPOSITION (MI350X gfx950, GPUs 4/5/6, stable run-to-run, ticks/us=100):
//   baseline period 6.5-6.9us = WALL ~2.85 + GAP ~3.8. Attributed by the toggles below:
//     ~3.0us  cross-XCD 256-way producer convergence (straggler + fabric) -- SOLO collapses GAP 3.7->0.5
//     ~1.9us  the op's own HBM round-trip (body 12KB r/w; +0.6 cold DRAM)  -- NO_BODYMEM removes it
//     ~0.8us  barriers/loop bookkeeping (3x syncthreads over 512 thr)      -- SOLO_NOBODY skeleton 1.3us
//     ~0.4us  succ-signal release (buffer_wbl2)                            -- NO_REL removes it
//     ~0.3us  acquire (buffer_inv)                                         -- NO_ACQ removes it
//     ~0.0us  interp loop-advance / TEN staging / LDS re-zero (const-addr scalar loads; no re-zero)
//   VERDICT: the floor is STRUCTURAL (cross-XCD convergence + HBM round-trip dominate). The
//   in-interp-loop reducible surface is ~0.7us of coherence fences that E8/sentinel proved
//   irreducible in scope, and ~0 of loop bookkeeping. PREFETCH (occ-1) yields no win (>= noise).
//
// Compile-time toggles (default = faithful baseline, all on):
//   -DNO_ACQ     drop the agent-scope acquire fence (buffer_inv)
//   -DNO_REL     signal RELAXED not RELEASE (drop buffer_wbl2)
//   -DNO_BODYMEM body does no HBM read/write (pure dispatch skeleton)
//   -DBODY_BYTES=<n> HBM bytes the narrow body reads+writes (default 12288 = 6144 bf16, GLM hidden)
//   -DCOLD       narrow body reads a rotating >L2 slice (cold DRAM round-trip)
//   -DSOLO       WG0 chains to itself (threshold 1) -- isolates pure dispatch, no cross-XCD convergence
//   -DPREFETCH   touch the next body's input line during the gate spin (the in-scope reduction test)
//
// BUILD/RUN (needs the render group for GPU access; pin an idle GPU with ROCR_VISIBLE_DEVICES):
//   hipcc --offload-arch=gfx950 -O3 -std=c++17 -w interp_dispatch_floor.hip -o floor
//   sg render -c 'LD_LIBRARY_PATH=/opt/rocm/lib ROCR_VISIBLE_DEVICES=4 ./floor 600'
//
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
namespace cg = cooperative_groups;

#ifndef BODY_BYTES
#define BODY_BYTES 12288          // 6144 bf16 = GLM hidden; a 1-CU RmsNorm reads ~this
#endif
#define THREADS 512               // 8 waves, matches PLOW_THREADS (faithful syncthreads cost)
#define CTR_STRIDE 32u            // PLOW_CTR_STRIDE (128 B / counter, one cache line)
#ifndef NSTEP
#define NSTEP 400                 // wide+narrow pairs in the chain
#endif

__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p){
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void ctr_signal(uint32_t* p){
#ifdef NO_REL
  __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);   // buffer_wbl2
#endif
}
__device__ __forceinline__ void ctr_acquire(){
#ifndef NO_ACQ
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");                            // buffer_inv
#endif
}
__device__ __forceinline__ uint64_t now(){ return __builtin_amdgcn_s_memrealtime(); }

#ifdef COLD
#define ABYTES (256u*1024u*1024u)   // >> L2: each op reads a cold DRAM slice
#else
#define ABYTES BODY_BYTES
#endif
struct Args {
  uint32_t* counters;   // NSTEP*2 counters, strided CTR_STRIDE
  uint32_t* hbmA;        // ABYTES/4 input word buffer (narrow body reads)
  uint32_t* hbmB;        // BODY_BYTES/4 output (narrow body writes)
  uint32_t* count;       // NX registration
  uint64_t* tr_arr;      // narrow-op t_arrive [NSTEP]
  uint64_t* tr_rdy;      // narrow-op t_ready  [NSTEP]
  uint64_t* tr_end;      // narrow-op t_end    [NSTEP]
  uint64_t* wtr_end;     // wide-op max t_end  [NSTEP] (producer tail)
  uint32_t* sink;
  int NS;
};

#define NX 8
#define CTR(base,id) ((base) + (size_t)(id)*CTR_STRIDE)

// minimal RmsNorm-shaped body: read BODY_BYTES, LDS block-reduce (2 syncthreads), write BODY_BYTES.
__device__ __forceinline__ float narrow_body(const uint32_t* in, uint32_t* out, uint32_t* sink, uint32_t off){
  __shared__ float red[THREADS/64];
  const int tid = threadIdx.x;
  const int NW = BODY_BYTES/4;
  in += off; // COLD: rotate to a fresh DRAM slice each op
  float acc = 0.f;
#ifndef NO_BODYMEM
  for (int i=tid;i<NW;i+=THREADS){ uint32_t v=in[i]; acc += (float)(v & 0xffff) + (float)(v>>16); }
#endif
  // wave reduce + LDS cross-wave reduce (mirrors sm->part block_sum)
  for (int o=32;o>0;o>>=1) acc += __shfl_down(acc,o);
  if ((tid&63)==0) red[tid>>6]=acc;
  __syncthreads();
  float tot=0.f; if (tid<THREADS/64) tot=red[tid];
  for (int o=(THREADS/64)/2;o>0;o>>=1) tot += __shfl_down(tot,o);
  float rms = 1.f/(1e-6f + tot/ (float)(BODY_BYTES/2));
  __syncthreads();
#ifndef NO_BODYMEM
  for (int i=tid;i<NW;i+=THREADS){ uint32_t v=in[i];
    float a=(float)(v&0xffff)*rms, b=(float)(v>>16)*rms;
    out[i] = ((uint32_t)b<<16)|((uint32_t)a & 0xffff); }
#else
  if (tid==0) *sink ^= (uint32_t)rms;
#endif
  return rms;
}

__global__ void __launch_bounds__(THREADS)
chain(Args a){
  __shared__ char pad[90*1024];   // force 1 block/CU => 256 blocks over all 8 XCDs
  ((volatile char*)pad)[threadIdx.x & 1023] = 0;
  const int bid = blockIdx.x, tid = threadIdx.x;
  cg::grid_group grid = cg::this_grid();
  grid.sync();
  const int NS = a.NS;

  // Each step s: WIDE op (all WGs, counter 2s) then NARROW op (WG0 only, counter 2s+1).
  // SOLO: WG0 chains to itself (narrow gates on prev narrow, threshold 1) — isolates pure dispatch
  //       with NO 256-way cross-XCD producer convergence.
#ifdef SOLO
  const uint32_t NARROW_THRESH = 1u;
#else
  const uint32_t NARROW_THRESH = 256u;
#endif
  uint32_t coff = 0;
  for (int s=0; s<NS; s++){
#ifndef SOLO
    // ---------- WIDE op (all 256 WGs): gate on prev narrow (2s-1, threshold 1) ----------
    if (s>0){
      if (tid==0){ while (ctr_poll(CTR(a.counters,2*s-1)) < 1u) __builtin_amdgcn_s_sleep(2); }
      __syncthreads();
      if (tid==0) ctr_acquire();
      __syncthreads();
    }
    // tiny wide body (a few HBM touches so it is not degenerate) — not the measured op
    { uint32_t acc=0; for (int i=tid;i<256;i+=THREADS) acc^=a.hbmA[i]; if(tid==0) a.sink[bid]^=acc; }
    __syncthreads();
    uint64_t we = now();
    if (tid==0) ctr_signal(CTR(a.counters,2*s));           // each WG +1 -> threshold 256
    if (tid==0) atomicMax((unsigned long long*)&a.wtr_end[s],(unsigned long long)we);
#else
    // SOLO producer: WG0 signals counter 2s to threshold 1 itself (no cross-XCD)
    if (bid==0 && tid==0){ a.wtr_end[s]=now(); ctr_signal(CTR(a.counters,2*s)); }
#endif

    // ---------- NARROW op (WG0 only): the 4.6us-floor op, gated on the wide producer ----------
    if (bid==0){
      uint64_t t_arr = now();
#ifdef PREFETCH
      // in-scope reduction under test: touch the next body's input line during the spin
      volatile uint32_t pf=0;
#endif
      if (tid==0){ while (ctr_poll(CTR(a.counters,2*s)) < NARROW_THRESH){
#ifdef PREFETCH
          pf ^= a.hbmA[0];
#endif
          __builtin_amdgcn_s_sleep(2);
        } }
      __syncthreads();
      if (tid==0) ctr_acquire();
      __syncthreads();
      uint64_t t_rdy = now();
      narrow_body(a.hbmA, a.hbmB, &a.sink[0], coff);
#ifdef COLD
      coff += (BODY_BYTES/4);
      if (coff + BODY_BYTES/4 > ABYTES/4) coff = 0;   // rotate over the >L2 buffer
#endif
      __syncthreads();
      uint64_t t_end;
      if (tid==0){ ctr_signal(CTR(a.counters,2*s+1)); t_end = now();
        a.tr_arr[s]=t_arr; a.tr_rdy[s]=t_rdy; a.tr_end[s]=t_end; }
    }
  }
  grid.sync();
}

#define CK(x) do{ hipError_t e=(x); if(e){ printf("HIP ERR %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(2);} }while(0)

static int cmp64(const void*a,const void*b){ uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b; return x<y?-1:x>y?1:0; }

int main(int argc,char**argv){
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p,0));
  const int GD = p.multiProcessorCount;
  int NS = (argc>1)? atoi(argv[1]) : NSTEP;
  const char* cfg =
#if defined(NO_ACQ)
    "NO_ACQ (drop acquire buffer_inv)";
#elif defined(NO_REL)
    "NO_REL (relaxed signal, no buffer_wbl2)";
#elif defined(NO_BODYMEM)
    "NO_BODYMEM (no HBM r/w in body)";
#elif defined(PREFETCH)
    "PREFETCH (touch next body input in spin)";
#else
    "BASELINE (faithful interp dispatch)";
#endif
  printf("dev=%s CU=%d THREADS=%d BODY_BYTES=%d NSTEP=%d cfg=%s\n",
    p.gcnArchName,GD,THREADS,BODY_BYTES,NS,cfg);

  Args a; a.NS=NS;
  size_t nc = (size_t)NS*2*CTR_STRIDE;
  CK(hipMalloc(&a.counters,nc*4)); CK(hipMemset(a.counters,0,nc*4));
  CK(hipMalloc(&a.hbmA,ABYTES)); CK(hipMemset(a.hbmA,1,ABYTES));
  CK(hipMalloc(&a.hbmB,BODY_BYTES)); CK(hipMemset(a.hbmB,0,BODY_BYTES));
  CK(hipMalloc(&a.count,NX*4)); CK(hipMemset(a.count,0,NX*4));
  CK(hipMalloc(&a.tr_arr,NS*8)); CK(hipMemset(a.tr_arr,0,NS*8));
  CK(hipMalloc(&a.tr_rdy,NS*8)); CK(hipMemset(a.tr_rdy,0,NS*8));
  CK(hipMalloc(&a.tr_end,NS*8)); CK(hipMemset(a.tr_end,0,NS*8));
  CK(hipMalloc(&a.wtr_end,NS*8)); CK(hipMemset(a.wtr_end,0,NS*8));
  CK(hipMalloc(&a.sink,(size_t)GD*4)); CK(hipMemset(a.sink,0,(size_t)GD*4));

  int mab=0; CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&mab,(void*)chain,THREADS,0));
  if (mab<1){ printf("FATAL mab=%d\n",mab); return 3; }

  void* kargs[]={&a};
  hipEvent_t e0,e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  CK(hipEventRecord(e0));
  CK(hipLaunchCooperativeKernel((void*)chain,dim3(GD),dim3(THREADS),kargs,0,0));
  CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
  float ms=0; CK(hipEventElapsedTime(&ms,e0,e1)); CK(hipDeviceSynchronize());

  uint64_t *arr=(uint64_t*)malloc(NS*8),*rdy=(uint64_t*)malloc(NS*8),*end=(uint64_t*)malloc(NS*8),*we=(uint64_t*)malloc(NS*8);
  CK(hipMemcpy(arr,a.tr_arr,NS*8,hipMemcpyDeviceToHost));
  CK(hipMemcpy(rdy,a.tr_rdy,NS*8,hipMemcpyDeviceToHost));
  CK(hipMemcpy(end,a.tr_end,NS*8,hipMemcpyDeviceToHost));
  CK(hipMemcpy(we, a.wtr_end,NS*8,hipMemcpyDeviceToHost));

  // ticks->us : s_memrealtime is ~100 MHz (10 ns/tick) per dev_isa.h & autopsy calibration.
  const double TPUS = 100.0; // ticks per us
  int lo=NS/10, hi=NS-2;     // skip warmup + last
  int n=hi-lo;
  uint64_t *wall=(uint64_t*)malloc(n*8), *gap=(uint64_t*)malloc(n*8), *g2=(uint64_t*)malloc(n*8);
  double swall=0,sgap=0,sg2=0;
  for (int i=lo;i<hi;i++){
    wall[i-lo]= end[i]-rdy[i];             // t_ready->t_end : the "4.6us floor" WALL
    gap[i-lo] = rdy[i]-arr[i];             // t_arrive->t_ready : poll + acquire
    g2[i-lo]  = (arr[i]>we[i])? arr[i]-we[i] : 0; // producer-tail->narrow-arrive slack
    swall+=wall[i-lo]; sgap+=gap[i-lo]; sg2+=g2[i-lo];
  }
  qsort(wall,n,8,cmp64); qsort(gap,n,8,cmp64); qsort(g2,n,8,cmp64);
  auto med=[&](uint64_t*x){return x[n/2];};
  printf("---- n=%d ops (median over steps %d..%d) ----\n",n,lo,hi);
  printf(" WALL  t_ready->t_end   : median %.3f us  mean %.3f us  (min %.3f p90 %.3f)\n",
         med(wall)/TPUS, (swall/n)/TPUS, wall[0]/TPUS, wall[(int)(n*0.9)]/TPUS);
  printf(" GAP   t_arrive->t_ready: median %.3f us  mean %.3f us  (min %.3f p90 %.3f)\n",
         med(gap)/TPUS,(sgap/n)/TPUS, gap[0]/TPUS, gap[(int)(n*0.9)]/TPUS);
  printf(" PROD-tail slack        : median %.3f us  mean %.3f us\n", med(g2)/TPUS,(sg2/n)/TPUS);
  printf(" per-op period (wall+gap) median: %.3f us\n", (med(wall)+med(gap))/TPUS);
  printf("RESULTLINE cfg=[%s] wall_med=%.3f gap_med=%.3f period=%.3f\n",
         cfg, med(wall)/TPUS, med(gap)/TPUS, (med(wall)+med(gap))/TPUS);
  return 0;
}
