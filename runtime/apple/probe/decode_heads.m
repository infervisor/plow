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
            NSString* code = variant ? [@"#define PLOW_DECODE_HEADS 1\n" stringByAppendingString:src] : src;
            id<MTLLibrary> lib = [dev newLibraryWithSource:code options:opts error:&error];
            if (!lib) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
            psos[variant] = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"plow_single"] error:&error];
            assert(psos[variant] && psos[variant].maxTotalThreadsPerThreadgroup >= 1024);
        }
        unsigned checks = 0;
        for (unsigned width=1;width<=2;width++) for (unsigned D=128;D<=256;D*=2)
        for (unsigned gqa=1;gqa<=4;gqa++) for (unsigned split=1;split<=3;split+=2)
        for (unsigned blocks=1;blocks<=16;blocks*=4) {
            unsigned B=2,HK=3,H=HK*gqa,stride=128;
            size_t output=B*H*split*D*4, ml=B*H*split*2*4;
            NSMutableArray<id<MTLBuffer>>* bs=[NSMutableArray array];
            size_t sizes[]={output+64,ml+64,B*H*D*2,B*HK*stride*D*width,B*HK*stride*D*width,B*4,B*HK*stride*4,B*HK*stride*4};
            for (unsigned i=0;i<8;i++) [bs addObject:buffer(sizes[i])];
            fill(bs[2],11);
            for (unsigned i=3;i<=4;i++) {
                if (width==2) fill(bs[i],11+i);
                else for (size_t j=0;j<bs[i].length;j++) ((uint8_t*)bs[i].contents)[j]=(j*13+i)%120;
            }
            for (unsigned i=6;i<8;i++) for (size_t j=0;j<bs[i].length/4;j++) ((float*)bs[i].contents)[j]=0.015625f*(1+j%3);
            ((int*)bs[5].contents)[0]=17;((int*)bs[5].contents)[1]=179;
            PlowDevInst d={.op=width==1?PLOW_DOP_FLASH_DECODE_FP8:PLOW_DOP_FLASH_DECODE,.blocks=blocks};
            for (unsigned i=0;i<8;i++) d.t[i]=i;
            if (width==2) d.t[6]=d.t[7]=PLOW_TENSOR_NONE;
            d.i[0]=B;d.i[1]=H;d.i[2]=HK;d.i[3]=stride;d.i[4]=29;
            d.i[5]=split;d.i[6]=D;d.i[7]=stride-1;d.fj[0].f=1.0f/sqrtf(D);
            NSData* expected[2];
            for (unsigned v=0;v<2;v++) {
                memset(bs[0].contents,0x5a,bs[0].length);memset(bs[1].contents,0x5a,bs[1].length);
                run(d,bs,psos[v]);
                for (unsigned j=0;j<2;j++) {
                    id<MTLBuffer> buf=bs[j];
                    if (!v) expected[j]=[NSData dataWithBytes:buf.contents length:buf.length];
                    else assert(memcmp(expected[j].bytes,buf.contents,buf.length)==0);
                    for (size_t i=buf.length-64;i<buf.length;i++) assert(((uint8_t*)buf.contents)[i]==0x5a);
                    for (size_t i=0;i<(buf.length-64)/4;i++) assert(((uint32_t*)buf.contents)[i]!=0x5a5a5a5a);
                }
            }
            checks++;
            printf("kv_bytes=%u D=%u gqa=%u split=%u blocks=%u exact=1\n",width,D,gqa,split,blocks);
        }
        printf("%u decode head cases passed\n",checks);
    }
}
