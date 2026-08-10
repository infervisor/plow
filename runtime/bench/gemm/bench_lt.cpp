// hipBLASLt tuned (best-algo) benchmark for Qwen3-4B GEMM/GEMV shapes.
// TN gemm: C[M,N] = A[M,K] * B[N,K]^T  (opA=T, opB=N, K-major storage)
// FLOPs = 2*M*N*K.  Reports best-over-all-algos TFLOP/s and effective TB/s.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CHK(x) do{ auto e=(x); if(e!=hipSuccess){printf("HIP err %d @%d\n",(int)e,__LINE__);exit(1);} }while(0)
#define LCHK(x) do{ auto s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){printf("LT err %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)

struct Shape{ const char* name; int64_t N, K; };

static size_t dbytes(hipDataType t){
  switch(t){ case HIP_R_16BF: return 2; case HIP_R_8F_E4M3: return 1; case HIP_R_32F: return 4; default: return 2; }
}

int main(int argc, char** argv){
  // dtype: 0 = bf16, 1 = fp8 e4m3
  int mode = argc>1 ? atoi(argv[1]) : 0;
  hipDataType tin = mode==0 ? HIP_R_16BF : HIP_R_8F_E4M3;
  hipDataType tout = HIP_R_16BF;               // accumulate/output bf16
  hipblasComputeType_t tcomp = HIPBLAS_COMPUTE_32F;
  const char* dname = mode==0 ? "bf16" : "fp8e4m3";

  std::vector<int64_t> Ms = {1, 512, 4096};
  std::vector<Shape> shapes = {
    {"q_proj",   4096, 2560},
    {"kv_proj",  1024, 2560},
    {"o_proj",   2560, 4096},
    {"gate/up",  9728, 2560},
    {"down",     2560, 9728},
  };

  /* SHAPE ON ARGV, so the same tuned-library reference can be pointed at another model's
   * demand without a recompile per shape. `bench_lt <mode>` alone is unchanged: the Qwen
   * table above. `bench_lt <mode> M N K [name]` replaces the whole cross product with the
   * one (M,N,K), which is what a per-shape A/B against plow's own tile ladder needs. */
  if(argc>4){
    Ms.assign(1, (int64_t)atoll(argv[2]));
    shapes.assign(1, Shape{ argc>5 ? argv[5] : "shape", (int64_t)atoll(argv[3]), (int64_t)atoll(argv[4]) });
  }

  hipblasLtHandle_t handle; LCHK(hipblasLtCreate(&handle));
  size_t wsBytes = size_t(512)*1024*1024;
  void* ws; CHK(hipMalloc(&ws,wsBytes));
  hipStream_t stream; CHK(hipStreamCreate(&stream));
  hipEvent_t ev0,ev1; CHK(hipEventCreate(&ev0)); CHK(hipEventCreate(&ev1));

  float alpha=1.f, beta=0.f;
  // per-tensor fp8 scales = 1.0 on device
  float one=1.f; void* dscale=nullptr; CHK(hipMalloc(&dscale,sizeof(float)));
  CHK(hipMemcpy(dscale,&one,sizeof(float),hipMemcpyHostToDevice));

  printf("# hipBLASLt  dtype=%s  (TN gemm, out=bf16, compute=f32)\n", dname);
  printf("%-9s %6s %6s %6s | %10s %8s | %9s %8s\n",
         "shape","M","N","K","best_us","TFLOPs","effTB/s","GB");

  for(int64_t M : Ms){
   for(auto& sh : shapes){
    int64_t N=sh.N, K=sh.K;
    // allocate device buffers (K-major)
    void *dA,*dB,*dC;
    size_t szA=(size_t)M*K*dbytes(tin), szB=(size_t)N*K*dbytes(tin), szC=(size_t)M*N*dbytes(tout);
    CHK(hipMalloc(&dA,szA)); CHK(hipMalloc(&dB,szB)); CHK(hipMalloc(&dC,szC));
    CHK(hipMemset(dA,1,szA)); CHK(hipMemset(dB,1,szB)); CHK(hipMemset(dC,0,szC));

    hipblaslt_ext::Gemm gemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, tin, tin, tout, tout, tcomp);
    hipblaslt_ext::GemmEpilogue epi;
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(dA); inputs.setB(dB); inputs.setC(dC); inputs.setD(dC);
    inputs.setAlpha(&alpha); inputs.setBeta(&beta);
    if(mode==1){ inputs.setScaleA(dscale); inputs.setScaleB(dscale); }
    LCHK(gemm.setProblem(M,N,K,1,epi,inputs));

    std::vector<hipblasLtMatmulHeuristicResult_t> algos;
    hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        HIPBLAS_OP_T, HIPBLAS_OP_N, tin, tin, tout, tout, tcomp, algos);

    double bestMs = 1e30; int bestIdx=-1; int supported=0;
    int screen = (M==1)?60:15;
    for(size_t i=0;i<algos.size();++i){
      size_t wsNeed=0;
      if(gemm.isAlgoSupported(algos[i].algo, wsNeed)!=HIPBLAS_STATUS_SUCCESS) continue;
      if(wsNeed>wsBytes) continue;
      if(gemm.initialize(algos[i].algo, ws)!=HIPBLAS_STATUS_SUCCESS) continue;
      supported++;
      // warmup
      if(gemm.run(stream)!=HIPBLAS_STATUS_SUCCESS) continue;
      CHK(hipStreamSynchronize(stream));
      CHK(hipEventRecord(ev0,stream));
      for(int r=0;r<screen;++r) gemm.run(stream);
      CHK(hipEventRecord(ev1,stream)); CHK(hipEventSynchronize(ev1));
      float ms; CHK(hipEventElapsedTime(&ms,ev0,ev1)); ms/=screen;
      if(ms<bestMs){ bestMs=ms; bestIdx=(int)i; }
    }
    if(bestIdx<0){ printf("%-9s %6ld %6ld %6ld | NO SUPPORTED ALGO (of %zu)\n",sh.name,M,N,K,algos.size());
      hipFree(dA);hipFree(dB);hipFree(dC); continue; }

    // final precise measure of best algo
    LCHK(gemm.initialize(algos[bestIdx].algo, ws));
    for(int r=0;r<10;++r) gemm.run(stream); CHK(hipStreamSynchronize(stream));
    int iters = (M==1)?300:80;
    CHK(hipEventRecord(ev0,stream));
    for(int r=0;r<iters;++r) gemm.run(stream);
    CHK(hipEventRecord(ev1,stream)); CHK(hipEventSynchronize(ev1));
    float ms; CHK(hipEventElapsedTime(&ms,ev0,ev1)); double us=(ms/iters)*1000.0;

    double flops=2.0*M*N*K;
    double tflops = flops/ (us*1e-6) /1e12;
    double bytes = (double)szA + szB + szC;
    double tbs = bytes/(us*1e-6)/1e12;
    printf("%-9s %6ld %6ld %6ld | %10.2f %8.1f | %9.3f %8.4f  [%d/%zu algos]\n",
           sh.name,M,N,K,us,tflops,tbs,bytes/1e9,supported,algos.size());
    fflush(stdout);
    hipFree(dA);hipFree(dB);hipFree(dC);
   }
  }
  return 0;
}
