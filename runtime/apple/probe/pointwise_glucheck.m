#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "golden/golden.h"

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 2);
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSError* error = nil;
        NSString* src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe; opts.languageVersion = MTLLanguageVersion3_2;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&error];
        if (!lib) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"plow_single"] error:&error];
        assert(pso && pso.maxTotalThreadsPerThreadgroup >= 1024);
        unsigned sizes[] = {1, 33, 1027, 4097}, partitions[] = {1, 16, 64};
        unsigned checks = 0, max_ulp = 0, gelu_tails = 0;
        for (unsigned shape = 0; shape < 4; shape++) for (unsigned part = 0; part < 3; part++)
        for (unsigned act = 0; act < 4; act++) { @autoreleasepool {
            unsigned n = sizes[shape];
            NSMutableArray<id<MTLBuffer>>* bs = [NSMutableArray array];
            uint64_t tab[3];
            for (unsigned i = 0; i < 3; i++) {
                id<MTLBuffer> b = [dev newBufferWithLength:(n+8)*2 options:MTLResourceStorageModeShared];
                assert(b); memset(b.contents, 0x5a, b.length);
                [bs addObject:b]; tab[i] = b.gpuAddress;
            }
            plow_bf16* gate = bs[1].contents; plow_bf16* up = bs[2].contents;
            for (unsigned i = 0; i < n; i++) {
                gate[i] = plow_f2bf(((int)(i % 257) - 128) / 16.0f);
                up[i] = plow_f2bf(((int)(i % 193) - 96) / 8.0f);
            }
            PlowDevInst d = {.op=PLOW_DOP_GLU, .blocks=partitions[part]};
            d.t[0]=0; d.t[1]=1; d.t[2]=2; d.i[0]=n; d.i[1]=act;
            d.fj[0].f=1.702f; d.fj[1].f=7;
            id<MTLBuffer> fault = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
            *(unsigned*)fault.contents=0;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            unsigned zero=0;
            [enc setComputePipelineState:pso]; [enc setBytes:&d length:sizeof(d) atIndex:0];
            [enc setBytes:tab length:sizeof(tab) atIndex:7]; [enc setBytes:&zero length:4 atIndex:8];
            [enc setBuffer:fault offset:0 atIndex:9];
            for (id<MTLBuffer> b in bs) [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [enc dispatchThreadgroups:MTLSizeMake(d.blocks,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            assert(cb.status == MTLCommandBufferStatusCompleted && *(unsigned*)fault.contents == 0);
            plow_bf16* out = bs[0].contents;
            for (unsigned i = 0; i < n; i++) {
                plow_bf16 ref=plow_f2bf(g_glu_pair(plow_bf2f(gate[i]), plow_bf2f(up[i]), act, 1.702f, 7));
                unsigned ulp=abs((int)out[i]-(int)ref);
                if (plow_bf2f(out[i]) == plow_bf2f(ref)) ulp=0;
                // Metal's shared tanh helper cancels tiny negative GELU tails in FP32.
                if (act == 0 && ulp > 1 && fabsf(plow_bf2f(out[i])-plow_bf2f(ref)) <= 1e-5f) {
                    gelu_tails++; continue;
                }
                if (ulp > max_ulp) max_ulp=ulp;
                if (ulp > 1) fprintf(stderr, "act=%u i=%u gate=%g up=%g expected=%04x actual=%04x\n",
                    act,i,plow_bf2f(gate[i]),plow_bf2f(up[i]),ref,out[i]);
                assert(ulp <= 1);
            }
            for (unsigned i=n; i<n+8; i++) assert(out[i] == 0x5a5a);
            checks++;
        } }
        printf("checks=%u guards=1 max_bf16_ulp=%u gelu_tail_abs_1e5=%u\n",checks,max_ulp,gelu_tails);
    }
    return 0;
}
