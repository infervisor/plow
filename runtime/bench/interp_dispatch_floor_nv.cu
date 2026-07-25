// interp_dispatch_floor_nv.cu — measure plow's per-op dispatch floor on NVIDIA.
//
// WHY THIS EXISTS
//
// `costmodel::cost::DECODE_DISPATCH_FLOOR_US = 4.6` is the per-op counter-gate
// "dead air" every M=1 op pays at each hand-off. It was measured on MI350X
// gfx950 and it is applied to EVERY GPU in the registry, scaled only by
// `clock_boost`. That constant dominates the cost of a decode-sized op, so it
// decides how much the model values fusing ops — on hardware where nobody has
// ever measured it.
//
// This is the NVIDIA counterpart of `interp_dispatch_floor.hip`. It reproduces
// the same structure that bench isolates, using the primitive lowering
// `runtime/nvidia/interp_sm120.cu` actually compiles:
//
//   WIDE op   = every block signals counter[k] (+1); threshold = gridDim.
//   NARROW op = block 0 alone, gated on that counter — the RmsNorm/Residual
//               shaped 1-SM consumer whose gate wait IS the floor.
//
// Gating uses `ld.acquire.gpu` / `red.release.gpu`, which is what
// PLOW_NV_PTXSYNC=1 (the default, interp_sm120.cu:96) emits instead of
// membar+atomicAdd. Counters are `PLOW_CTR_STRIDE`-strided so each lands on its
// own cache line, as in dev_isa.h:843. Timing is `%globaltimer` (nanoseconds).
//
// The kernel is launched cooperatively at exactly the resident grid, because
// co-residency is the interpreter's safety condition (dev_isa.h:15) — a larger
// grid deadlocks on the gate rather than running slowly.
//
//   Per-op GAP  = t_arrive -> t_ready  (the gate wait: this is the floor)
//   Per-op WALL = t_ready  -> t_end    (body + syncthreads + successor signal)
//
// BUILD/RUN
//   nvcc -arch=sm_90a -O3 -std=c++17 -w interp_dispatch_floor_nv.cu -o floor_nv
//   perf-data/harness/gpulease floor ./floor_nv [nstep]
//
// gpulease exits 76 if the GPU was contended; a contended run is discarded, not
// reported, because a contended timing is indistinguishable from a real one.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <vector>
namespace cg = cooperative_groups;

#ifndef BODY_BYTES
#define BODY_BYTES 12288          // 6144 bf16 — a 1-SM RmsNorm reads about this
#endif
#define THREADS 256               // interp_sm120.cu launches __launch_bounds__(256, ...)
#define CTR_STRIDE 32u            // PLOW_CTR_STRIDE — one cache line per counter
#ifndef NSTEP
#define NSTEP 400
#endif

#define CUDA_OK(call)                                                                    \
  do {                                                                                   \
    cudaError_t _e = (call);                                                              \
    if (_e != cudaSuccess) {                                                              \
      fprintf(stderr, "FATAL %s:%d %s: %s\n", __FILE__, __LINE__, #call,                  \
              cudaGetErrorString(_e));                                                    \
      return 2;                                                                           \
    }                                                                                     \
  } while (0)

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// The gate poll. Relaxed would be enough to observe the value; the ACQUIRE is
// what orders the producer's writes before the consumer's reads, and it is the
// half that costs. interp_sm120.cu uses ld.acquire.gpu here.
__device__ __forceinline__ unsigned ctr_poll_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void ctr_signal_release(unsigned* p) {
  asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}

struct Rec {
  unsigned long long t_arrive, t_ready, t_end;
};

// One persistent kernel walking a chain of (wide, narrow) op pairs, exactly the
// shape a decode step has: an all-SM GEMV followed by a 1-SM norm gated on it.
__global__ __launch_bounds__(THREADS, 1) void chain(unsigned* ctrs, float* body, Rec* recs,
                                                    int nstep, int nblk) {
  cg::grid_group grid = cg::this_grid();
  const int b = blockIdx.x;
  const int t = threadIdx.x;

  for (int s = 0; s < nstep; ++s) {
    unsigned* wide_ctr = ctrs + (size_t)(2 * s) * CTR_STRIDE;
    unsigned* narrow_ctr = ctrs + (size_t)(2 * s + 1) * CTR_STRIDE;

    // ---- WIDE op: every block participates, then signals. ----
    unsigned long long a0 = now_ns();
    // A trivial body so the wide op is not free but is not the thing measured.
    if (t == 0) body[b] = body[b] * 1.000001f + 1.0f;
    __syncthreads();
    unsigned long long r0 = now_ns();
    if (t == 0) ctr_signal_release(wide_ctr);
    unsigned long long e0 = now_ns();

    if (b == 0 && t == 0) {
      recs[2 * s].t_arrive = a0;
      recs[2 * s].t_ready = r0;
      recs[2 * s].t_end = e0;
    }

    // ---- NARROW op: block 0 only, gated on all nblk wide signals. ----
    if (b == 0) {
      unsigned long long a1 = now_ns();
      if (t == 0) {
        // The gate. This spin is the dispatch floor.
        while (ctr_poll_acquire(wide_ctr) < (unsigned)nblk) {
          __nanosleep(64);
        }
      }
      __syncthreads();
      unsigned long long r1 = now_ns();

#ifndef NO_BODYMEM
      // A 1-SM HBM round trip of BODY_BYTES, like a decode norm.
      const int n = BODY_BYTES / 4;
      float acc = 0.f;
      for (int i = t; i < n; i += THREADS) acc += body[i];
      __syncthreads();
      if (t == 0) body[n] = acc;
#endif
      __syncthreads();
      if (t == 0) ctr_signal_release(narrow_ctr);
      unsigned long long e1 = now_ns();

      if (t == 0) {
        recs[2 * s + 1].t_arrive = a1;
        recs[2 * s + 1].t_ready = r1;
        recs[2 * s + 1].t_end = e1;
      }
    }

    // Every block waits for the narrow op before the next pair, which is what
    // makes this a chain rather than nblk independent streams.
    if (t == 0) {
      while (ctr_poll_acquire(narrow_ctr) < 1u) {
        __nanosleep(64);
      }
    }
    __syncthreads();
    grid.sync();
  }
}

static double median(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static double pct(std::vector<double>& sorted_v, double q) {
  if (sorted_v.empty()) return 0.0;
  size_t i = (size_t)((sorted_v.size() - 1) * q + 0.5);
  return sorted_v[std::min(i, sorted_v.size() - 1)];
}

int main(int argc, char** argv) {
  int nstep = (argc > 1) ? atoi(argv[1]) : NSTEP;
  if (nstep < 8) nstep = 8;

  cudaDeviceProp prop;
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));

  // Co-residency is the safety condition: launch exactly the resident grid.
  int per_sm = 0;
  CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, (const void*)chain, THREADS, 0));
  if (per_sm < 1) {
    fprintf(stderr, "FATAL: kernel is not resident at %d threads\n", THREADS);
    return 2;
  }
  int nblk = per_sm * prop.multiProcessorCount;

  printf("device      : %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  printf("grid        : %d blocks (%d/SM x %d SMs), %d threads\n", nblk, per_sm,
         prop.multiProcessorCount, THREADS);
  printf("chain       : %d wide+narrow pairs, body %d B\n", nstep, BODY_BYTES);

  unsigned* ctrs = nullptr;
  float* body = nullptr;
  Rec* recs = nullptr;
  size_t nctr = (size_t)2 * nstep * CTR_STRIDE;
  CUDA_OK(cudaMalloc(&ctrs, nctr * sizeof(unsigned)));
  CUDA_OK(cudaMemset(ctrs, 0, nctr * sizeof(unsigned)));
  CUDA_OK(cudaMalloc(&body, (BODY_BYTES + 4096) * sizeof(float)));
  CUDA_OK(cudaMemset(body, 0, (BODY_BYTES + 4096) * sizeof(float)));
  CUDA_OK(cudaMalloc(&recs, (size_t)2 * nstep * sizeof(Rec)));
  CUDA_OK(cudaMemset(recs, 0, (size_t)2 * nstep * sizeof(Rec)));

  void* args[] = {&ctrs, &body, &recs, &nstep, &nblk};
  cudaEvent_t t0, t1;
  CUDA_OK(cudaEventCreate(&t0));
  CUDA_OK(cudaEventCreate(&t1));

  CUDA_OK(cudaEventRecord(t0));
  CUDA_OK(cudaLaunchCooperativeKernel((void*)chain, dim3(nblk), dim3(THREADS), args, 0, 0));
  CUDA_OK(cudaEventRecord(t1));
  CUDA_OK(cudaDeviceSynchronize());

  float ms = 0.f;
  CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));

  std::vector<Rec> h((size_t)2 * nstep);
  CUDA_OK(cudaMemcpy(h.data(), recs, h.size() * sizeof(Rec), cudaMemcpyDeviceToHost));

  // Drop the first 10% as warm-up: the first passes pay cold instruction and
  // DRAM residency that a steady-state decode step does not.
  int skip = nstep / 10 + 1;
  std::vector<double> gap, wall, period;
  for (int s = skip; s < nstep; ++s) {
    const Rec& nr = h[2 * s + 1];
    if (nr.t_ready < nr.t_arrive || nr.t_end < nr.t_ready) continue;
    gap.push_back((double)(nr.t_ready - nr.t_arrive) / 1000.0);
    wall.push_back((double)(nr.t_end - nr.t_ready) / 1000.0);
    if (s > skip) {
      const Rec& pv = h[2 * (s - 1) + 1];
      if (nr.t_end > pv.t_end) period.push_back((double)(nr.t_end - pv.t_end) / 1000.0);
    }
  }
  if (gap.empty()) {
    fprintf(stderr, "FATAL: no usable samples\n");
    return 2;
  }

  double g = median(gap), w = median(wall), p = median(period);
  std::sort(gap.begin(), gap.end());
  std::sort(period.begin(), period.end());

  printf("samples     : %zu (skipped %d warm-up)\n", gap.size(), skip);
  printf("kernel wall : %.3f ms total\n", ms);
  printf("\n");
  printf("  GAP   (gate wait, THE FLOOR)  median %8.3f us   p10 %8.3f  p90 %8.3f\n", g,
         pct(gap, 0.10), pct(gap, 0.90));
  printf("  WALL  (body + signal)         median %8.3f us\n", w);
  printf("  PERIOD(narrow op to narrow op) median %8.3f us   p10 %8.3f  p90 %8.3f\n", p,
         pct(period, 0.10), pct(period, 0.90));
  printf("\n");
  printf("floor_us_median %.4f\n", g + w);
  printf("gap_us_median %.4f\n", g);
  printf("wall_us_median %.4f\n", w);
  printf("period_us_median %.4f\n", p);
  printf("samples %zu\n", gap.size());

  cudaFree(ctrs);
  cudaFree(body);
  cudaFree(recs);
  return 0;
}
