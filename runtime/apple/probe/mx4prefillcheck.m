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
    [enc dispatchThreadgroups:MTLSizeMake(d.blocks+1,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
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
            NSString* code=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
            id<MTLLibrary> lib=[dev newLibraryWithSource:code options:opts error:&error];
            if (!lib) { fprintf(stderr,"%s\n",error.description.UTF8String); return 1; }
            psos[v]=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:v?@"plow_mx4_prefill":@"plow_single"] error:&error];
            assert(psos[v] && psos[v].maxTotalThreadsPerThreadgroup>=1024);
        }
        const unsigned shapes[][3]={{1,8,32},{31,33,64},{32,2048,2048},{128,6144,2048},{129,1024,2048},{208,2048,6144},{256,2048,2048}};
        const unsigned partitions[]={1,5,16,20,64};
        unsigned checks=0;
        for(unsigned c=0;c<7;c++) for(unsigned part=0;part<5;part++)
        for(unsigned op_index=0;op_index<4;op_index++) { @autoreleasepool {

            unsigned m=shapes[c][0],n=shapes[c][1],k=shapes[c][2];
            NSMutableArray<id<MTLBuffer>>* bs=[NSMutableArray array];
            size_t sizes[]={(m+2)*n*2+64,(m+1)*k*2,n*k/2,n*((k+31)/32)};
            for(unsigned i=0;i<4;i++) [bs addObject:buffer(sizes[i])];
            fill(bs[1],18);
            uint8_t* wg=bs[2].contents;
            for(unsigned i=0;i<n*k/2;i++) {
                wg[i]=(i*71u%255u);
                if((wg[i]&127u)==127u) wg[i]=0;
            }
            {
                uint8_t* scales=bs[3].contents;
                for(unsigned i=0;i<n*((k+31)/32);i++) scales[i]=120+i%7;
            }
            const unsigned ops[]={93,96,97,98};
            PlowDevInst d={.op=ops[op_index],.blocks=partitions[part]};
            for(unsigned i=0;i<8;i++) d.t[i]=PLOW_TENSOR_NONE;
            d.t[0]=0;d.t[1]=1;d.t[2]=2;d.t[3]=3;

            d.i[4]=1;d.i[0]=m;d.i[1]=n;d.i[2]=k;d.i[5]=1;
            NSData* expected=nil;
            for(unsigned v=0;v<2;v++) {
                memset(bs[0].contents,0x5a,bs[0].length);
                run(d,bs,psos[v]);
                if(!v) expected=[NSData dataWithBytes:bs[0].contents length:bs[0].length];
                else if(memcmp(expected.bytes,bs[0].contents,bs[0].length)!=0) {
                    fprintf(stderr,"parity_fail shape=%u M=%u N=%u K=%u blocks=%u op_index=%u\n",c,m,n,k,d.blocks,op_index);
                    const uint16_t* a=expected.bytes;const uint16_t* b=bs[0].contents;
                    for(unsigned z=0;z<m*n;z++) if(a[z]!=b[z]) fprintf(stderr,"index=%u expected=%04x actual=%04x\n",z,a[z],b[z]);
                    return 2;
                }
                for(size_t i=n;i<(m+1)*n;i++) assert(((uint16_t*)bs[0].contents)[i]!=0x5a5a);
                for(size_t i=0;i<n*2;i++) assert(((unsigned char*)bs[0].contents)[i]==0x5a);
                for(size_t i=(m+1)*n*2;i<bs[0].length;i++) assert(((unsigned char*)bs[0].contents)[i]==0x5a);
            }
            checks++;
            if(c>=2 && partitions[part]==16) {
                for(unsigned repeat=0;repeat<22;repeat++) {
                    double times[2];
                    for(unsigned order=0;order<2;order++) {
                        unsigned v=(repeat+order)%2;
                        times[v]=run(d,bs,psos[v]);
                    }
                    if(repeat>=2) printf("shape=%u op_index=%u repeat=%u baseline_us=%.3f candidate_us=%.3f\n",c,op_index,repeat-2,times[0],times[1]);
                }
            }
        }
        }
        printf("checks=%u exact=1 guards=1\n",checks);
    }
    return 0;
}
