// Focused hipBLASLt prof harness: for ONE shape, screen algos, pick best, then run ONLY the
// best algo N times so a rocprofv3 --pmc pass yields clean uniform rows (one Kernel_Name).
// Also prints the selected kernel name so we can disassemble exactly what the library chose.
//   usage: bench_lt_prof <mode 0=bf16 1=fp8> <shapeIdx 0..4> [M] [runs]
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

#define CHK(x) do{ auto e=(x); if(e!=hipSuccess){printf("HIP err %d @%d\n",(int)e,__LINE__);exit(1);} }while(0)
#define LCHK(x) do{ auto s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){printf("LT err %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)

struct Shape{ const char* name; int64_t N, K; };
static size_t dbytes(hipDataType t){ switch(t){ case HIP_R_16BF: return 2; case HIP_R_8F_E4M3: return 1; default: return 2; } }

int main(int argc, char** argv){
  int mode = argc>1 ? atoi(argv[1]) : 0;
  int si   = argc>2 ? atoi(argv[2]) : 0;
  int64_t M = argc>3 ? atoll(argv[3]) : 4096;
  int runs  = argc>4 ? atoi(argv[4]) : 100;
  hipDataType tin = mode==0 ? HIP_R_16BF : HIP_R_8F_E4M3;
  hipDataType tout = HIP_R_16BF;
  hipblasComputeType_t tcomp = HIPBLAS_COMPUTE_32F;

  std::vector<Shape> shapes = {
    {"q_proj",4096,2560},{"kv_proj",1024,2560},{"o_proj",2560,4096},{"gate_up",9728,2560},{"down",2560,9728},
  };
  Shape sh = shapes[si];
  int64_t N=sh.N, K=sh.K;

  hipblasLtHandle_t handle; LCHK(hipblasLtCreate(&handle));
  size_t wsBytes=size_t(512)*1024*1024; void* ws; CHK(hipMalloc(&ws,wsBytes));
  hipStream_t stream; CHK(hipStreamCreate(&stream));
  hipEvent_t ev0,ev1; CHK(hipEventCreate(&ev0)); CHK(hipEventCreate(&ev1));
  float alpha=1.f,beta=0.f;
  float one=1.f; void* dscale=nullptr; CHK(hipMalloc(&dscale,sizeof(float))); CHK(hipMemcpy(dscale,&one,sizeof(float),hipMemcpyHostToDevice));

  void *dA,*dB,*dC;
  size_t szA=(size_t)M*K*dbytes(tin), szB=(size_t)N*K*dbytes(tin), szC=(size_t)M*N*dbytes(tout);
  CHK(hipMalloc(&dA,szA)); CHK(hipMalloc(&dB,szB)); CHK(hipMalloc(&dC,szC));
  CHK(hipMemset(dA,1,szA)); CHK(hipMemset(dB,1,szB)); CHK(hipMemset(dC,0,szC));

  hipblaslt_ext::Gemm gemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, tin, tin, tout, tout, tcomp);
  hipblaslt_ext::GemmEpilogue epi; hipblaslt_ext::GemmInputs inputs;
  inputs.setA(dA); inputs.setB(dB); inputs.setC(dC); inputs.setD(dC);
  inputs.setAlpha(&alpha); inputs.setBeta(&beta);
  if(mode==1){ inputs.setScaleA(dscale); inputs.setScaleB(dscale); }
  LCHK(gemm.setProblem(M,N,K,1,epi,inputs));

  std::vector<hipblasLtMatmulHeuristicResult_t> algos;
  hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
      HIPBLAS_OP_T, HIPBLAS_OP_N, tin, tin, tout, tout, tcomp, algos);

  double bestMs=1e30; int bestIdx=-1;
  for(size_t i=0;i<algos.size();++i){
    size_t wsNeed=0;
    if(gemm.isAlgoSupported(algos[i].algo, wsNeed)!=HIPBLAS_STATUS_SUCCESS) continue;
    if(wsNeed>wsBytes) continue;
    if(gemm.initialize(algos[i].algo, ws)!=HIPBLAS_STATUS_SUCCESS) continue;
    if(gemm.run(stream)!=HIPBLAS_STATUS_SUCCESS) continue;
    CHK(hipStreamSynchronize(stream));
    CHK(hipEventRecord(ev0,stream));
    for(int r=0;r<15;++r) gemm.run(stream);
    CHK(hipEventRecord(ev1,stream)); CHK(hipEventSynchronize(ev1));
    float ms; CHK(hipEventElapsedTime(&ms,ev0,ev1)); ms/=15;
    if(ms<bestMs){ bestMs=ms; bestIdx=(int)i; }
  }
  if(bestIdx<0){ printf("no algo\n"); return 1; }
  int algoIdx = hipblaslt_ext::getIndexFromAlgo(algos[bestIdx].algo);
  double us=bestMs*1000.0; double tflops=2.0*M*N*K/(us*1e-6)/1e12;
  printf("# shape=%s M=%ld N=%ld K=%ld  best_algo_index=%d  %.2f us  %.1f TF/s\n",
         sh.name,M,N,K,algoIdx,us,tflops);
  fflush(stdout);

  // Run ONLY the best algo, many times, for rocprof.
  LCHK(gemm.initialize(algos[bestIdx].algo, ws));
  for(int r=0;r<10;++r) gemm.run(stream);
  CHK(hipStreamSynchronize(stream));
  for(int r=0;r<runs;++r) gemm.run(stream);
  CHK(hipStreamSynchronize(stream));
  return 0;
}
