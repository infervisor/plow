// rocBLAS bf16 cross-check (gemm_ex, tuned default). TN: C[M,N]=A[M,K]*B[N,K]^T
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#define CHK(x) do{ auto e=(x); if(e!=hipSuccess){printf("HIP err %d @%d\n",(int)e,__LINE__);exit(1);} }while(0)
#define RCHK(x) do{ auto s=(x); if(s!=rocblas_status_success){printf("ROC err %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)
struct Shape{ const char* name; int N,K; };
int main(){
  std::vector<int> Ms={1,512,4096};
  std::vector<Shape> shapes={{"q_proj",4096,2560},{"kv_proj",1024,2560},{"o_proj",2560,4096},{"gate/up",9728,2560},{"down",2560,9728}};
  rocblas_handle h; RCHK(rocblas_create_handle(&h));
  hipStream_t strm; CHK(hipStreamCreate(&strm)); RCHK(rocblas_set_stream(h,strm));
  hipEvent_t e0,e1; CHK(hipEventCreate(&e0)); CHK(hipEventCreate(&e1));
  float alpha=1.f,beta=0.f;
  printf("# rocBLAS bf16 gemm_ex (TN, out=bf16, compute=f32)\n");
  printf("%-9s %6s %6s %6s | %10s %8s | %9s\n","shape","M","N","K","best_us","TFLOPs","effTB/s");
  for(int M:Ms) for(auto&sh:shapes){
    int N=sh.N,K=sh.K;
    void *dA,*dB,*dC; size_t szA=(size_t)M*K*2,szB=(size_t)N*K*2,szC=(size_t)M*N*2;
    CHK(hipMalloc(&dA,szA));CHK(hipMalloc(&dB,szB));CHK(hipMalloc(&dC,szC));
    CHK(hipMemset(dA,1,szA));CHK(hipMemset(dB,1,szB));CHK(hipMemset(dC,0,szC));
    // TN: transA=T lda=K, transB=N ldb=K, ldc=M
    auto call=[&](){ return rocblas_gemm_ex(h,rocblas_operation_transpose,rocblas_operation_none,
      M,N,K,&alpha,dA,rocblas_datatype_bf16_r,K,dB,rocblas_datatype_bf16_r,K,&beta,
      dC,rocblas_datatype_bf16_r,M,dC,rocblas_datatype_bf16_r,M,
      rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0); };
    rocblas_status st=call();
    if(st!=rocblas_status_success){ printf("%-9s %6d %6d %6d | ERR %d\n",sh.name,M,N,K,(int)st); hipFree(dA);hipFree(dB);hipFree(dC); continue; }
    for(int r=0;r<10;r++) call(); CHK(hipStreamSynchronize(strm));
    int it=(M==1)?300:80;
    CHK(hipEventRecord(e0,strm)); for(int r=0;r<it;r++) call(); CHK(hipEventRecord(e1,strm)); CHK(hipEventSynchronize(e1));
    float ms; CHK(hipEventElapsedTime(&ms,e0,e1)); double us=(ms/it)*1000.0;
    double tf=2.0*M*N*K/(us*1e-6)/1e12; double tb=((double)szA+szB+szC)/(us*1e-6)/1e12;
    printf("%-9s %6d %6d %6d | %10.2f %8.1f | %9.3f\n",sh.name,M,N,K,us,tf,tb); fflush(stdout);
    hipFree(dA);hipFree(dB);hipFree(dC);
  }
  return 0;
}
