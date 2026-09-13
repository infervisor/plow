#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "dev_isa.h"

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLBuffer> buffer(size_t n) {
    id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    assert(b); memset(b.contents, 0x5a, n); return b;
}
static void fill(id<MTLBuffer> b, unsigned seed) {
    uint16_t* p = b.contents;
    for (size_t i = 0; i < b.length / 2; i++) {
        seed = seed * 1664525u + 1013904223u;
        float f = ((int)(seed >> 8) - 8388608) / 8388608.0f;
        uint32_t u; memcpy(&u, &f, 4);
        p[i] = (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    }
}
static double run(PlowDevInst d, NSArray<id<MTLBuffer>>* bs, id<MTLComputePipelineState> pso) {
    uint64_t tab[12] = {0};
    for (unsigned i = 0; i < bs.count; i++) tab[i] = bs[i].gpuAddress;
    id<MTLBuffer> fault = buffer(4); *(unsigned*)fault.contents = 0;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    unsigned zero = 0;
    [enc setComputePipelineState:pso];
    [enc setBytes:&d length:sizeof(d) atIndex:0];
    [enc setBytes:tab length:sizeof(tab) atIndex:7];
    [enc setBytes:&zero length:4 atIndex:8];
    [enc setBuffer:fault offset:0 atIndex:9];
    for (id<MTLBuffer> b in bs) [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [enc dispatchThreadgroups:MTLSizeMake(d.blocks,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    assert(cb.status == MTLCommandBufferStatusCompleted && *(unsigned*)fault.contents == 0);
    return (cb.GPUEndTime-cb.GPUStartTime) * 1e6;
}
int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 2);
        dev = MTLCreateSystemDefaultDevice(); queue = [dev newCommandQueue];
        NSError* error = nil;
        NSString* src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
        assert(src);
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe; opts.languageVersion = MTLLanguageVersion3_2;
        id<MTLComputePipelineState> psos[2];
        for (unsigned v=0;v<2;v++) {
            NSString* code=v?[@"#define PLOW_GLU_PAIR 1\n" stringByAppendingString:src]:src;
            id<MTLLibrary> lib=[dev newLibraryWithSource:code options:opts error:&error];
            if (!lib) { fprintf(stderr,"%s\n",error.description.UTF8String); return 1; }
            psos[v]=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"plow_single"] error:&error];
            assert(psos[v] && psos[v].maxTotalThreadsPerThreadgroup>=1024);
        }
        const unsigned shapes[][3]={{1,1,4},{2,7,36},{1,129,2880},{1,6144,2048},{2,33,24}};
        const unsigned partitions[]={1,8,16,20,64};
        unsigned checks=0;
        for(unsigned c=0;c<5;c++) for(unsigned part=0;part<5;part++)
        for(unsigned bias=0;bias<4;bias++) for(unsigned act=0;act<4;act++) {
            unsigned m=shapes[c][0],n=shapes[c][1],k=shapes[c][2];
            NSMutableArray<id<MTLBuffer>>* bs=[NSMutableArray array];
            size_t sizes[]={m*n*2+64,m*k*2,n*k*2,4,4,n*k*2,n*2,n*2};
            for(unsigned i=0;i<8;i++) { [bs addObject:buffer(sizes[i])]; if(i) fill(bs[i],17+i); }
            PlowDevInst d={.op=PLOW_DOP_GEMV_GLU,.blocks=partitions[part]};
            for(unsigned i=0;i<8;i++) d.t[i]=i;
            d.t[6]=(bias&1)?6:PLOW_TENSOR_NONE; d.t[7]=(bias&2)?7:PLOW_TENSOR_NONE;
            d.i[0]=m;d.i[1]=n;d.i[2]=k;d.i[5]=act;d.fj[0].f=1.702f;d.fj[1].f=7.0f;
            NSData* expected=nil;
            for(unsigned v=0;v<2;v++) {
                memset(bs[0].contents,0x5a,bs[0].length);
                run(d,bs,psos[v]);
                if(!v) expected=[NSData dataWithBytes:bs[0].contents length:bs[0].length];
                else assert(memcmp(expected.bytes,bs[0].contents,bs[0].length)==0);
                for(size_t i=0;i<m*n;i++) assert(((uint16_t*)bs[0].contents)[i]!=0x5a5a);
                for(size_t i=m*n*2;i<bs[0].length;i++) assert(((unsigned char*)bs[0].contents)[i]==0x5a);
            }
            checks++;
            if(c==3 && !bias && act==1 && partitions[part]==16) {
                for(unsigned repeat=0;repeat<14;repeat++) {
                    double times[2];
                    for(unsigned order=0;order<2;order++) {
                        unsigned v=(repeat+order)%2;
                        times[v]=run(d,bs,psos[v]);
                    }
                    if(repeat>=2) printf("repeat=%u scalar_us=%.3f pair_us=%.3f\n",repeat-2,times[0],times[1]);
                }
            }
        }
        printf("checks=%u exact=1 guards=1\n",checks);
    }
    return 0;
}
