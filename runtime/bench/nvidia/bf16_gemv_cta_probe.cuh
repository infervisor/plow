#pragma once
#include <cstring>

extern "C" const void* gemv_cta_kernel_256(unsigned);
extern "C" const void* gemv_cta_kernel_512(unsigned);

struct CtaShape { const char* name; unsigned op, nq, nk, nv, k; };
static constexpr CtaShape CTA_SHAPES[] = {
    {"q_local",0,4096,0,0,3840}, {"q_full",0,8192,0,0,3840},
    {"k_full",0,512,0,0,3840}, {"o_local",0,3840,0,0,4096},
    {"o_full",0,3840,0,0,8192}, {"down",0,3840,0,0,15360},
    {"qkv_local",1,4096,2048,2048,3840}, {"qkv_full",1,8192,512,512,3840},
    {"glu",2,15360,0,0,3840}, {"tail",0,1031,0,0,648},
    {"qkv_tail",1,513,257,261,648}, {"glu_tail",2,1031,0,0,648},
};
static constexpr unsigned cta_slices = 132, cta_arena = 32768, cta_shared = cta_arena + 128;

static void cta_check_layout() {
    for (const auto& s : CTA_SHAPES) {
        const unsigned n = s.nq + s.nk + s.nv, per = (n + cta_slices - 1) / cta_slices;
        if (!s.k || s.k % 8 || s.k * sizeof(bf16) > cta_arena) exit(3);
        for (unsigned warps : {8u,16u}) {
            std::vector<unsigned> counts(n);
            for (unsigned slice=0; slice<cta_slices; ++slice)
                for (unsigned warp=0; warp<warps; ++warp)
                    for (unsigned row=slice*per+warp; row<std::min((slice+1)*per,n); row+=warps) {
                        if (row / per != slice) exit(3);
                        ++counts[row];
                    }
            for (auto count : counts) if (count != 1) exit(3);
        }
    }
    printf("layout PASS: 12 shapes, 132 slices, 8/16 warps, K/arena bounds\n");
}

__global__ void cta_reference(bf16* out, const float* a, const float* b, unsigned n, bool glu) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = glu ? gemma_glu_epilogue(a[i], b[i], 0) : __float2bfloat16(a[i]);
}

static int bench_gemv_cta(int argc, char** argv) {
    cta_check_layout();
    if (argc > 1 && !std::strcmp(argv[1], "--check-layout")) return 0;
    const unsigned iters = argc > 1 ? (unsigned)std::atoi(argv[1]) : 20;
    if (!iters || iters > 1000) { printf("usage: probe [iterations|--check-layout]\n"); return 2; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    if (prop.major != 9 || prop.multiProcessorCount != cta_slices || prop.sharedMemPerBlock < cta_shared) {
        printf("requires Hopper with 132 SMs and >=%u shared bytes/block\n",cta_shared); return 2;
    }
    cublasHandle_t blas; LTK(cublasCreate(&blas));
    unsigned* errors; CK(cudaMalloc(&errors,sizeof(unsigned))); CK(cudaMemset(errors,0,sizeof(unsigned)));
    cudaEvent_t start,stop; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
    for (const auto& s : CTA_SHAPES) {
        const unsigned n=s.nq+s.nk+s.nv;
        bf16* x=dev_bf16(s.k);
        bf16* w0=dev_bf16((size_t)s.nq*s.k);
        bf16* w1=(s.op==1 || s.op==2)?dev_bf16((size_t)(s.op==2?s.nq:s.nk)*s.k):nullptr;
        bf16* w2=s.op==1?dev_bf16((size_t)s.nv*s.k):nullptr;
        float *r0,*r1; CK(cudaMalloc(&r0,n*sizeof(float))); CK(cudaMalloc(&r1,n*sizeof(float)));
        bf16* ref; CK(cudaMalloc(&ref,n*sizeof(bf16)));
        const float alpha=1, beta=0;
        auto project=[&](const bf16* w,unsigned rows,float* out) {
            LTK(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,rows,1,s.k,
                &alpha,w,CUDA_R_16BF,s.k,x,CUDA_R_16BF,s.k,&beta,out,CUDA_R_32F,rows,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        };
        project(w0,s.nq,r0);
        if (s.op==1) { project(w1,s.nk,r0+s.nq); project(w2,s.nv,r0+s.nq+s.nk); }
        if (s.op==2) project(w1,s.nq,r1);
        cta_reference<<<(n+255)/256,256>>>(ref,r0,r1,n,s.op==2);
        bf16* allocation[2]; bf16* output[2];
        const void* kernels[]={gemv_cta_kernel_256(s.op),gemv_cta_kernel_512(s.op)};
        unsigned nq=s.nq,nk=s.nk,nv=s.nv,k=s.k;
        auto launch=[&](unsigned arm) {
            void* args[]={&output[arm],&x,&w0,&w1,&w2,&nq,&nk,&nv,&k,&errors};
            CK(cudaLaunchKernel(kernels[arm],dim3(cta_slices),dim3(arm?512:256),args,cta_shared,0));
        };
        for (unsigned arm=0;arm<2;++arm) {
            CK(cudaMalloc(&allocation[arm],(n+16)*sizeof(bf16))); output[arm]=allocation[arm]+8;
            CK(cudaMemset(allocation[arm],0xa5,(n+16)*sizeof(bf16)));
            CK(cudaMemset(output[arm],0xff,n*sizeof(bf16)));
            cudaFuncAttributes attr; CK(cudaFuncGetAttributes(&attr,kernels[arm]));
            int occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,kernels[arm],arm?512:256,cta_shared));
            if (!occ || (arm && attr.numRegs>128)) { printf("invalid512 resource contract\n"); return 3; }
            printf("resources %s threads=%u regs=%d local=%zu static_shared=%zu dynamic_shared=%u capacity_CTA_SM=%d\n",
                s.name,arm?512:256,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,cta_shared,occ);
            launch(arm);
        }
        CK(cudaDeviceSynchronize());
        std::vector<bf16> expected(n),got[2];
        CK(cudaMemcpy(expected.data(),ref,n*sizeof(bf16),cudaMemcpyDeviceToHost));
        for (unsigned arm=0;arm<2;++arm) {
            got[arm].resize(n); CK(cudaMemcpy(got[arm].data(),output[arm],n*sizeof(bf16),cudaMemcpyDeviceToHost));
            double err2=0,ref2=0,maxerr=0,maxref=0;
            for (unsigned i=0;i<n;++i) {
                double a=__bfloat162float(expected[i]),b=__bfloat162float(got[arm][i]);
                if (!std::isfinite(a)||!std::isfinite(b)) { printf("nonfinite output\n"); return 3; }
                err2+=(a-b)*(a-b); ref2+=a*a; maxerr=std::max(maxerr,std::abs(a-b)); maxref=std::max(maxref,std::abs(a));
            }
            const double rel=std::sqrt(err2/std::max(ref2,1e-30));
            printf("oracle %s threads=%u relL2=%.6g max_abs=%.6g\n",s.name,arm?512:256,rel,maxerr);
            if (rel>0.006 || maxerr>0.05+0.02*maxref) return 3;
        }
        if (std::memcmp(got[0].data(),got[1].data(),n*sizeof(bf16))) { printf("256/512 exact parity FAILED\n"); return 3; }
        for (unsigned i=0;i<5;++i) { launch(0); launch(1); }
        std::vector<float> times[2];
        for (unsigned i=0;i<iters;++i) for (unsigned j=0;j<2;++j) {
            const unsigned arm=(i+j)%2;
            cold_flush(); CK(cudaEventRecord(start)); launch(arm); CK(cudaEventRecord(stop));
            CK(cudaEventSynchronize(stop)); float ms; CK(cudaEventElapsedTime(&ms,start,stop));
            times[arm].push_back(ms);
        }
        const double bytes=2.0*s.k*(s.op==2?2*s.nq:n);
        float median[2];
        for (unsigned arm=0;arm<2;++arm) {
            std::sort(times[arm].begin(),times[arm].end()); median[arm]=times[arm][times[arm].size()/2];
            uint16_t guards[16]; CK(cudaMemcpy(guards,allocation[arm],16,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(guards+8,output[arm]+n,16,cudaMemcpyDeviceToHost));
            for (auto guard:guards) if (guard!=0xa5a5) { printf("output guard FAILED\n"); return 3; }
            printf("timing %s threads=%u slices=132 median_us=%.3f weight_GBs=%.1f\n",s.name,arm?512:256,median[arm]*1000,bytes/(median[arm]*1e6));
            CK(cudaFree(allocation[arm]));
        }
        unsigned guard_errors; CK(cudaMemcpy(&guard_errors,errors,sizeof(unsigned),cudaMemcpyDeviceToHost));
        if (guard_errors) { printf("shared arena guards FAILED=%u\n",guard_errors); return 3; }
        printf("PASS %s exact finite guards speedup512=%.4f\n",s.name,median[0]/median[1]);
        CK(cudaFree(x)); CK(cudaFree(w0)); if(w1) CK(cudaFree(w1)); if(w2) CK(cudaFree(w2));
        CK(cudaFree(r0)); CK(cudaFree(r1)); CK(cudaFree(ref));
    }
    CK(cudaFree(errors)); CK(cudaEventDestroy(start)); CK(cudaEventDestroy(stop)); LTK(cublasDestroy(blas));
    return 0;
}
