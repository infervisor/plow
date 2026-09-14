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
        for (unsigned variant = 0; variant < 2; variant++) {
            NSString* code = variant ? [@"#define PLOW_QKV_DOT4 1\n" stringByAppendingString:src] : src;
            id<MTLLibrary> lib = [dev newLibraryWithSource:code options:opts error:&error];
            if (!lib) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
            psos[variant] = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"plow_single"] error:&error];
            assert(psos[variant] && psos[variant].maxTotalThreadsPerThreadgroup >= 1024);
        }
        const unsigned cases[][5] = {{1,1,1,1,8},{2,7,13,5,24},{1,129,65,33,2880},{1,2048,1024,1024,2048},{2,33,17,9,36}};
        unsigned checks = 0;
        const unsigned partitions[] = {1,8,16,20,64};
        for (unsigned c = 0; c < 5; c++) for (unsigned part = 0; part < 5; part++) for (unsigned bias = 0; bias < 2; bias++) for (unsigned norm = 0; norm < 2; norm++) {
            unsigned blocks = partitions[part];
            unsigned m=cases[c][0], nq=cases[c][1], nk=cases[c][2], nv=cases[c][3], k=cases[c][4];
            NSMutableArray<id<MTLBuffer>>* bs = [NSMutableArray array];
            size_t sizes[] = {m*nq*2+64,m*k*2,nq*k*2,m*nk*2+64,nk*k*2,m*nv*2+64,nv*k*2,k*2,nq*2,nk*2,nv*2};
            for (unsigned i=0;i<11;i++) { [bs addObject:buffer(sizes[i])]; if (i!=0 && i!=3 && i!=5) fill(bs[i],17+i); }
            PlowDevInst d = {.op=PLOW_DOP_GEMV_QKV,.blocks=blocks};
            for (unsigned i=0;i<8;i++) d.t[i]=i;
            d.t[7]=norm?7:PLOW_TENSOR_NONE;
            d.i[0]=m;d.i[1]=nq;d.i[2]=k;d.i[3]=nk;d.i[4]=nv;
            d.i[5]=bias?8:0;d.i[6]=bias?9:0;d.i[7]=bias?10:0;
            d.fj[0].f=1e-6f;
            NSData* expected[3]; unsigned outputs[]={0,3,5};
            double times[2]={0};
            for (unsigned v=0;v<2;v++) {
                for (unsigned j=0;j<3;j++) memset(bs[outputs[j]].contents,0x5a,bs[outputs[j]].length);
                times[v]=run(d,bs,psos[v]);
                for (unsigned j=0;j<3;j++) {
                    id<MTLBuffer> b=bs[outputs[j]];
                    if (!v) expected[j]=[NSData dataWithBytes:b.contents length:b.length];
                    else assert(memcmp(expected[j].bytes,b.contents,b.length)==0);
                    for (size_t i=0;i<(b.length-64)/2;i++) assert(((uint16_t*)b.contents)[i]!=0x5a5a);
                    for (size_t i=b.length-64;i<b.length;i++) assert(((unsigned char*)b.contents)[i]==0x5a);
                }
            }
            printf("case=%u blocks=%u bias=%u norm=%u scalar_us=%.3f dot4_us=%.3f exact=1\n",c,blocks,bias,norm,times[0],times[1]);
            if (c == 3 && !norm && !bias) {
                for (unsigned repeat=0;repeat<10;repeat++) {
                    double paired[2];
                    for (unsigned order=0;order<2;order++) {
                        unsigned variant=(repeat+order)%2;
                        paired[variant]=run(d,bs,psos[variant]);
                    }
                    printf("bench blocks=%u repeat=%u scalar_us=%.3f dot4_us=%.3f\n",blocks,repeat,paired[0],paired[1]);
                }
            }
            checks++;
        }
        printf("%u QKV cases passed\n",checks);
    }
}
