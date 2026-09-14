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
static double run(PlowDevInst d, NSArray<id<MTLBuffer>>* bs, id<MTLComputePipelineState> pso, unsigned variant) {
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
    unsigned width=d.op==PLOW_DOP_GEMV_GLU_MXFP4?2:8;
    [enc dispatchThreadgroups:MTLSizeMake(variant?(d.i[1]+width-1)/width:d.blocks,1,1) threadsPerThreadgroup:MTLSizeMake(variant?64:1024,1,1)];
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
            NSString* code=src;
            id<MTLLibrary> lib=[dev newLibraryWithSource:code options:opts error:&error];
            if (!lib) { fprintf(stderr,"%s\n",error.description.UTF8String); return 1; }
            psos[v]=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:v?@"plow_mx4_dedicated":@"plow_single"] error:&error];
            assert(psos[v] && psos[v].maxTotalThreadsPerThreadgroup>=(v?64:1024));
        }
        const unsigned shapes[][3]={{1,8,32},{2,16,64},{1,128,2048},{1,6144,2048},{2,32,1024},{1,2048,6144},{1,151936,2048}};
        const unsigned partitions[]={1,8,16,20,64};
        unsigned checks=0;
        for(unsigned c=0;c<7;c++) for(unsigned part=0;part<5;part++)
        for(unsigned glu=0;glu<2;glu++) { @autoreleasepool {
            if(c==6 && glu) continue;
            unsigned m=shapes[c][0],n=shapes[c][1],k=shapes[c][2];
            NSMutableArray<id<MTLBuffer>>* bs=[NSMutableArray array];
            size_t sizes[]={m*n*2+64,m*k*2,n*k/2,n*((k+31)/32),n*((k+31)/32),n*k/2,n*((k+31)/32),4};
            for(unsigned i=0;i<8;i++) [bs addObject:buffer(sizes[i])];
            fill(bs[1],18);
            uint8_t* wg=bs[2].contents; uint8_t* wu=bs[5].contents;
            for(unsigned i=0;i<n*k/2;i++) {
                wg[i]=(i*71u%255u);
                wu[i]=(i*37u%255u);
                if((wg[i]&127u)==127u) wg[i]=0;
                if((wu[i]&127u)==127u) wu[i]=0;
            }
            for(unsigned slot=3;slot<=6;slot++) if(slot!=5) {
                uint8_t* scales=bs[slot].contents;
                for(unsigned i=0;i<n*((k+31)/32);i++) scales[i]=120+i%7;
            }
            PlowDevInst d={.op=glu?PLOW_DOP_GEMV_GLU_MXFP4:PLOW_DOP_GEMV_MXFP4,.blocks=partitions[part]};
            for(unsigned i=0;i<8;i++) d.t[i]=PLOW_TENSOR_NONE;
            d.t[0]=0;d.t[1]=1;d.t[2]=2;d.t[5]=glu?5:6;d.t[3]=3;
            if(glu) {d.t[3]=3;d.t[4]=4;}
            d.i[0]=m;d.i[1]=n;d.i[2]=k;d.i[5]=1;
            NSData* expected=nil;
            for(unsigned v=0;v<2;v++) {
                memset(bs[0].contents,0x5a,bs[0].length);
                run(d,bs,psos[v],v);
                if(!v) expected=[NSData dataWithBytes:bs[0].contents length:bs[0].length];
                else if(memcmp(expected.bytes,bs[0].contents,bs[0].length)!=0) {
                    fprintf(stderr,"parity_fail shape=%u M=%u N=%u K=%u blocks=%u glu=%u\n",c,m,n,k,d.blocks,glu);
                    const uint16_t* a=expected.bytes;const uint16_t* b=bs[0].contents;
                    for(unsigned z=0;z<m*n;z++) if(a[z]!=b[z]) fprintf(stderr,"index=%u expected=%04x actual=%04x\n",z,a[z],b[z]);
                    return 2;
                }
                for(size_t i=0;i<m*n;i++) assert(((uint16_t*)bs[0].contents)[i]!=0x5a5a);
                for(size_t i=m*n*2;i<bs[0].length;i++) assert(((unsigned char*)bs[0].contents)[i]==0x5a);
            }
            checks++;
            if(c>=3 && c!=4 && partitions[part]==16) {
                for(unsigned repeat=0;repeat<22;repeat++) {
                    double times[2];
                    for(unsigned order=0;order<2;order++) {
                        unsigned v=(repeat+order)%2;
                        times[v]=run(d,bs,psos[v],v);
                    }
                    if(repeat>=2) printf("shape=%u glu=%u repeat=%u baseline_us=%.3f candidate_us=%.3f\n",c,glu,repeat-2,times[0],times[1]);
                }
            }
        }
        }
        printf("checks=%u exact=1 guards=1\n",checks);
    }
    return 0;
}
