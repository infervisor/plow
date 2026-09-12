#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "dev_isa.h"
#include "golden/fp8.h"

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
        id<MTLLibrary> lib=[dev newLibraryWithSource:src options:opts error:&error];
        if (!lib) { fprintf(stderr,"%s\n",error.description.UTF8String); return 1; }
        id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"plow_single"] error:&error];
        assert(pso && pso.maxTotalThreadsPerThreadgroup>=1024);
        const unsigned cases[][5]={{1,1,1,1,4},{2,7,13,5,24},{1,129,65,33,2884},
            {1,2048,1024,1024,2048},{2,33,17,9,36},{1,17,9,0,2048}};
        const unsigned partitions[]={1,8,16,20,64};
        unsigned checks=0;
        for(unsigned c=0;c<6;c++) for(unsigned part=0;part<5;part++) { @autoreleasepool {
            unsigned m=cases[c][0],k=cases[c][4],ns[]={cases[c][1],cases[c][2],cases[c][3]};
            unsigned outputs[]={0,3,5},weights[]={2,4,6};
            NSMutableArray<id<MTLBuffer>>* bs=[NSMutableArray array];
            size_t sizes[]={m*ns[0]*2+64,m*k*2,ns[0]*k,m*ns[1]*2+64,ns[1]*k,m*ns[2]*2+64,ns[2]*k,4,ns[0]*4,ns[1]*4,ns[2]*4};
            for(unsigned i=0;i<11;i++) [bs addObject:buffer(MAX(sizes[i],4))];
            fill(bs[1],18);
            for(unsigned j=0;j<3;j++) {
                uint8_t* w=bs[weights[j]].contents;
                for(unsigned i=0;i<ns[j]*k;i++) { unsigned v=(i*71u+j*13u)%256u; w[i]=(v&127u)==127u?0:v; }
                float* scale=bs[8+j].contents;
                for(unsigned i=0;i<ns[j];i++) scale[i]=0.001f*(1+i%7);
            }
            PlowDevInst fused={.op=PLOW_DOP_GEMV_QKV_FP8,.blocks=partitions[part]};
            for(unsigned i=0;i<8;i++) fused.t[i]=i;
            fused.t[7]=PLOW_TENSOR_NONE;
            fused.i[0]=m;fused.i[1]=ns[0];fused.i[2]=k;fused.i[3]=ns[1];fused.i[4]=ns[2];
            fused.i[5]=8;fused.i[6]=9;fused.i[7]=10;
            PlowDevInst split[3];
            for(unsigned j=0;j<3;j++) {
                split[j]=(PlowDevInst){.op=PLOW_DOP_GEMV_FP8,.blocks=partitions[part]};
                for(unsigned i=0;i<8;i++) split[j].t[i]=PLOW_TENSOR_NONE;
                split[j].t[0]=outputs[j];split[j].t[1]=1;split[j].t[2]=weights[j];split[j].t[5]=8+j;
                split[j].i[0]=m;split[j].i[1]=ns[j];split[j].i[2]=k;
                if(ns[j]) run(split[j],bs,pso);
            }
            void* golden[11];
            for(unsigned i=0;i<11;i++) {
                golden[i]=malloc(bs[i].length);
                memcpy(golden[i],bs[i].contents,bs[i].length);
            }
            for(unsigned j=0;j<3;j++) if(ns[j])
                for(unsigned slice=0;slice<split[j].blocks;slice++)
                    g_gemv_fp8(&split[j],slice,split[j].blocks,golden,NULL);
            NSData* cpu_expected[3];
            for(unsigned j=0;j<3;j++) {
                cpu_expected[j]=[NSData dataWithBytes:golden[outputs[j]] length:bs[outputs[j]].length];
                memset(golden[outputs[j]],0x5a,bs[outputs[j]].length);
            }
            for(unsigned slice=0;slice<fused.blocks;slice++)
                g_gemv_qkv_fp8(&fused,slice,fused.blocks,golden,NULL);
            for(unsigned j=0;j<3;j++)
                assert(memcmp(cpu_expected[j].bytes,golden[outputs[j]],bs[outputs[j]].length)==0);
            for(unsigned i=0;i<11;i++) free(golden[i]);
            NSData* expected[3];
            for(unsigned j=0;j<3;j++) {
                id<MTLBuffer> b=bs[outputs[j]];
                expected[j]=[NSData dataWithBytes:b.contents length:b.length];
                memset(b.contents,0x5a,b.length);
            }
            run(fused,bs,pso);
            for(unsigned j=0;j<3;j++) {
                id<MTLBuffer> b=bs[outputs[j]];
                assert(memcmp(expected[j].bytes,b.contents,b.length)==0);
                for(size_t i=0;i<m*ns[j];i++) assert(((uint16_t*)b.contents)[i]!=0x5a5a);
                for(size_t i=m*ns[j]*2;i<b.length;i++) assert(((uint8_t*)b.contents)[i]==0x5a);
            }
            checks++;
            if(c==3) for(unsigned repeat=0;repeat<14;repeat++) {
                double times[2]={0};
                for(unsigned order=0;order<2;order++) {
                    unsigned v=(repeat+order)%2;
                    if(v) times[v]=run(fused,bs,pso);
                    else for(unsigned j=0;j<3;j++) times[v]+=run(split[j],bs,pso);
                }
                if(repeat>=2) printf("blocks=%u repeat=%u split_us=%.3f fused_us=%.3f\n",partitions[part],repeat-2,times[0],times[1]);
            }
        }}
        printf("checks=%u exact=1 cpu_exact=1 guards=1\n",checks);
    }
    return 0;
}
