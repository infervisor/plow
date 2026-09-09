/* attn_offload.m — is attention worth giving to the ANE? Builds a CoreML attention graph at a
 * chosen shape, reports the compiler's per-layer placement and times it, so the row- vs
 * channel-split question is answered by measurement rather than by the datasheet TOPS.
 *
 * It #includes dtypecheck.m, so it links the CPU golden tier too (build it first: cmake the
 * runtime into build-cpu).
 *
 *   clang -fobjc-arc -O2 -I../../cpu/dev -I../../common \
 *     -framework Metal -framework CoreML -framework Foundation attn_offload.m \
 *     -o attn_offload -L../../../build-cpu -lplow_cpu_dev
 *   ./attn_offload <model-dir> [M=128] [L=512] [groups=2] [rows|channels] [reps=11]
 * Placement is the compiler's preference, not a hardware-counter trace.
 */
/* Experimental head split; link exactly like dtypecheck.m, plus CoreML.
 * Build with -O3. Usage: attn_offload interp.metal [M=128] [L=512] [ANE KV groups=2] [rows|channels] [reps=11]
 * No production dispatch or calibration is changed. */
#define main dtypecheck_main
#include "dtypecheck.m"
#undef main
#import <CoreML/CoreML.h>
#include <time.h>

static double now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static void varint(NSMutableData* d, uint64_t x) {
    do { uint8_t b = x & 127; x >>= 7; if (x) b |= 128; [d appendBytes:&b length:1]; } while (x);
}
static void integer(NSMutableData* d, unsigned f, uint64_t x) { varint(d, f*8); varint(d,x); }
static void message(NSMutableData* d, unsigned f, NSData* x) {
    varint(d, f*8+2); varint(d,x.length); [d appendData:x];
}
static void string(NSMutableData* d, unsigned f, NSString* x) { message(d,f,[x dataUsingEncoding:NSUTF8StringEncoding]); }
static NSData* feature(NSString* name, NSArray<NSNumber*>* shape) {
    NSMutableData *dims=[NSMutableData data], *arr=[NSMutableData data], *type=[NSMutableData data], *f=[NSMutableData data];
    for (NSNumber* v in shape) varint(dims,v.unsignedLongLongValue);
    message(arr,1,dims); integer(arr,2,65568);
    message(type,5,arr); string(f,1,name); message(f,3,type); return f;
}
static void layer(NSMutableData* nn, NSString* name, NSArray<NSString*>* inputs, NSString* output, unsigned tag, NSData* params) {
    NSMutableData* d=[NSMutableData data]; string(d,1,name);
    for (NSString* i in inputs) string(d,2,i);
    string(d,3,output); message(d,tag,params); message(nn,1,d);
}
static NSData* graph(NSDictionary<NSString*,NSArray*>* shapes, bool channels) {
    NSMutableData *desc=[NSMutableData data], *nn=[NSMutableData data], *model=[NSMutableData data];
    for (NSString* name in @[@"q",@"k",@"v",@"mask"]) message(desc,1,feature(name,shapes[name]));
    message(desc,10,feature(@"y",shapes[@"q"]));
    NSMutableData* mm=[NSMutableData data]; integer(mm,channels ? 1 : 2,1);
    layer(nn,@"qk",channels ? @[@"k",@"q"] : @[@"q",@"k"],@"scores",1045,mm);
    layer(nn,@"mask",@[@"scores",@"mask"],@"masked",880,[NSData data]);
    NSMutableData* sm=[NSMutableData data]; integer(sm,1,channels ? (uint64_t)-2 : (uint64_t)-1);
    layer(nn,@"softmax",@[@"masked"],@"p",950,sm);
    layer(nn,@"pv",channels ? @[@"v",@"p"] : @[@"p",@"v"],@"y",1045,[NSData data]);
    integer(nn,5,1); integer(model,1,5); message(model,2,desc); message(model,500,nn); return model;
}
static MLComputePlan* placement(NSURL* url, MLModelConfiguration* cfg) {
    dispatch_semaphore_t done=dispatch_semaphore_create(0);
    __block MLComputePlan* plan=nil;
    [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* p, NSError* e) {
        plan=p;
        if (e) fprintf(stderr,"placement unavailable: %s\n",e.description.UTF8String);
        dispatch_semaphore_signal(done);
    }];
    if (dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,60*NSEC_PER_SEC))) {
        fprintf(stderr,"placement timeout\n"); return nil;
    }
    unsigned ane_layers=0, total_layers=0;
    for (MLModelStructureNeuralNetworkLayer* l in plan.modelStructure.neuralNetwork.layers) {
        MLComputePlanDeviceUsage* u=[plan computeDeviceUsageForNeuralNetworkLayer:l];
        total_layers++;
        ane_layers += [(id)u.preferredComputeDevice isKindOfClass:[MLNeuralEngineComputeDevice class]];
        printf("placement %s (%s): preferred=%s supported=%s\n",l.name.UTF8String,l.type.UTF8String,
            NSStringFromClass([(id)u.preferredComputeDevice class]).UTF8String,u.supportedComputeDevices.description.UTF8String);
    }
    printf("compiler placement: %u/%u layers prefer ANE (not a hardware-counter trace)\n",ane_layers,total_layers);
    return plan;
}
static id<MTLCommandBuffer> metal_async(PlowDevInst d, NSArray<id<MTLBuffer>>* bs, id<MTLBuffer> fault) {
    uint64_t addresses[8]={0};
    for (unsigned i=0;i<bs.count;i++) addresses[i]=bs[i].gpuAddress;
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    unsigned zero=0; *(unsigned*)fault.contents=0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&d length:sizeof(d) atIndex:0];
    [enc setBytes:addresses length:sizeof(addresses) atIndex:7];
    [enc setBytes:&zero length:4 atIndex:8]; [enc setBuffer:fault offset:0 atIndex:9];
    for (id<MTLBuffer> b in bs) [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [enc dispatchThreadgroups:MTLSizeMake(d.blocks,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
    [enc endEncoding]; [cb commit]; return cb;
}
static void joined(id<MTLCommandBuffer> cb, id<MTLBuffer> fault) {
    [cb waitUntilCompleted];
    assert(cb.status==MTLCommandBufferStatusCompleted && *(unsigned*)fault.contents==0);
}
static double median(double* x, unsigned n) {
    for (unsigned i=0;i<n;i++) for (unsigned j=i+1;j<n;j++) if (x[j]<x[i]) { double t=x[i]; x[i]=x[j]; x[j]=t; }
    return x[n/2];
}
int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc>=2 && argc<=7); setbuf(stdout,NULL);
        const unsigned H=24, HK=8, D=128, GQA=3;
        unsigned M=argc>2 ? atoi(argv[2]) : 128, L=argc>3 ? atoi(argv[3]) : 512;
        unsigned groups=argc>4 ? atoi(argv[4]) : 2, reps=argc>6 ? atoi(argv[6]) : 11;
        bool channels=argc>5 && strcmp(argv[5],"channels")==0;
        assert(argc<=5 || strcmp(argv[5],"rows")==0 || strcmp(argv[5],"channels")==0);
        assert(M && L>=M && (L&(L-1))==0 && groups>0 && groups<HK && reps>0);
        unsigned ah=groups*GQA, gh=H-ah;
        printf("M=%u L=%u D=%u Metal heads=%u CoreML heads=%u layout=%s reps=%u\n",M,L,D,gh,ah,channels?"channels":"rows",reps);
        dev=MTLCreateSystemDefaultDevice(); queue=[dev newCommandQueue]; NSError* error=nil;
        NSString* source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error]; assert(source);
        // Probe-only head range: preserve physical Q/output strides and restrict work to low heads.
        NSArray* from=@[@"uint work = ((nq + 31u) / 32u) * nh * splits;",
            @"uint sp = w % splits, h = (w / splits) % nh, qb = w / (splits * nh) * 32u;"];
        NSArray* to=@[@"uint active = in.i[2] >> 16; nh &= 65535u; gqa = nh / nkh; if (!active) active = nh; uint work = ((nq + 31u) / 32u) * active * splits;",
            @"uint sp = w % splits, h = (w / splits) % active, qb = w / (splits * active) * 32u;"];
        for (unsigned i=0;i<from.count;i++) { assert([source containsString:from[i]]); source=[source stringByReplacingOccurrencesOfString:from[i] withString:to[i]]; }
        MTLCompileOptions* opts=[MTLCompileOptions new]; opts.mathMode=MTLMathModeSafe; opts.languageVersion=MTLLanguageVersion3_2;
        id<MTLLibrary> lib=[dev newLibraryWithSource:source options:opts error:&error];
        if (!lib) { fprintf(stderr,"Metal: %s\n",error.description.UTF8String); return 1; }
        pipeline=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"plow_single"] error:&error]; assert(pipeline);
        NSArray* bs=@[buffer(M*H*D*4),buffer(M*H*8),buffer(M*H*D*2),buffer(HK*L*D*2),buffer(HK*L*D*2),buffer(M*H*D*2)];
        fill(bs[2],2,17); fill(bs[3],2,31); fill(bs[4],2,49);
        PlowDevInst full=inst(11,16); for (unsigned i=0;i<6;i++) full.t[i]=i;
        full.i[0]=M; full.i[1]=L; full.i[2]=H; full.i[3]=HK; full.i[4]=L-M; full.i[6]=D; full.i[7]=1;
        full.fj[0].f=1.0f/sqrtf(D); full.fj[1].u=L; full.fj[2].u=L-1;
        id<MTLBuffer> fault=buffer(4); joined(metal_async(full,bs,fault),fault);
        NSData* reference=[NSData dataWithBytes:[bs[5] contents] length:[bs[5] length]];
        NSDictionary* shapes=@{@"q":channels?@[@(groups),@3,@(D),@(M)]:@[@(groups),@3,@(M),@(D)],
            @"k":channels?@[@(groups),@1,@(D),@(L)]:@[@(groups),@1,@(L),@(D)],
            @"v":channels?@[@(groups),@1,@(D),@(L)]:@[@(groups),@1,@(L),@(D)],
            @"mask":channels?@[@1,@1,@(L),@(M)]:@[@1,@1,@(M),@(L)]};
        NSString* dir=[NSTemporaryDirectory() stringByAppendingPathComponent:[@"plow-attn-offload-" stringByAppendingString:NSUUID.UUID.UUIDString]];
        assert([[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:NO attributes:nil error:&error]);
        NSURL* src=[NSURL fileURLWithPath:[dir stringByAppendingPathComponent:[dir.lastPathComponent stringByAppendingPathExtension:@"mlmodel"]]];
        assert([graph(shapes,channels) writeToURL:src options:NSDataWritingAtomic error:&error]);
        double compile_start=now_ms(); NSURL* url=[MLModel compileModelAtURL:src error:&error];
        if (!url) { fprintf(stderr,"CoreML compile: %s\n",error.description.UTF8String); return 1; }
        MLModelConfiguration* cfg=[MLModelConfiguration new]; cfg.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
        MLModel* model=[MLModel modelWithContentsOfURL:url configuration:cfg error:&error];
        if (!model) { fprintf(stderr,"CoreML load: %s\n",error.description.UTF8String); return 1; }
        printf("compile/load %.1f ms model=%s\n",now_ms()-compile_start,url.path.UTF8String);
        placement(url,cfg);
        NSMutableDictionary* arrays=[NSMutableDictionary dictionary];
        for (NSString* name in shapes) { arrays[name]=[[MLMultiArray alloc] initWithShape:shapes[name] dataType:MLMultiArrayDataTypeFloat32 error:&error]; assert(arrays[name]); }
        MLDictionaryFeatureProvider* provider=[[MLDictionaryFeatureProvider alloc] initWithDictionary:arrays error:&error]; assert(provider);
        MLMultiArray* backing=[[MLMultiArray alloc] initWithShape:shapes[@"q"] dataType:MLMultiArrayDataTypeFloat32 error:&error]; assert(backing);
        MLPredictionOptions* pred=[MLPredictionOptions new]; pred.outputBackings=@{@"y":backing};
        float *aq=[arrays[@"q"] dataPointer], *ak=[arrays[@"k"] dataPointer], *av=[arrays[@"v"] dataPointer], *am=[arrays[@"mask"] dataPointer];
        for (unsigned q=0;q<M;q++) for (unsigned k=0;k<L;k++) am[channels?k*M+q:q*L+k]=k<=L-M+q ? 0.0f : -10000.0f;
        void (^pack)(void)=^{
            const uint16_t *q=[bs[2] contents], *k=[bs[3] contents], *v=[bs[4] contents];
            for (unsigned h=0;h<ah;h++) for (unsigned r=0;r<M;r++) for (unsigned d=0;d<D;d++)
                aq[h*M*D+(channels?d*M+r:r*D+d)]=plow_bf2f(q[(r*H+gh+h)*D+d])/sqrtf(D);
            for (unsigned h=0;h<groups;h++) for (unsigned r=0;r<L;r++) for (unsigned d=0;d<D;d++) {
                unsigned out=h*L*D+(channels?d*L+r:r*D+d), in=((gh/GQA+h)*L+r)*D+d;
                ak[out]=plow_bf2f(k[in]); av[out]=plow_bf2f(v[in]);
            }
        };
        void (^scatter)(MLMultiArray*)=^(MLMultiArray* y) {
            assert(y.dataType==MLMultiArrayDataTypeFloat32 && y.shape.count==4);
            const float* out=y.dataPointer; uint16_t* dst=[bs[5] contents];
            size_t s[4]; for (unsigned i=0;i<4;i++) s[i]=[y.strides[i] unsignedLongLongValue];
            for (unsigned h=0;h<ah;h++) for (unsigned r=0;r<M;r++) for (unsigned d=0;d<D;d++) {
                size_t ix=(h/GQA)*s[0]+(h%GQA)*s[1]+(channels?d*s[2]+r*s[3]:r*s[2]+d*s[3]);
                dst[(r*H+gh+h)*D+d]=plow_f2bf(out[ix]);
            }
        };
        PlowDevInst split=full; split.i[2]=H|(gh<<16);
        double* baseline=calloc(reps,sizeof(double)), *mixed=calloc(reps,sizeof(double)), *ane=calloc(reps,sizeof(double));
        double* packs=calloc(reps,sizeof(double)), *scatters=calloc(reps,sizeof(double)), *gpus=calloc(reps,sizeof(double));
        bool same_backing=false;
        const unsigned warmup=30;
        for (unsigned r=0;r<reps+warmup;r++) @autoreleasepool {
            if (r==warmup+reps/2) fill(bs[2],2,73);
            double t=now_ms(); joined(metal_async(full,bs,fault),fault); double base_ms=now_ms()-t;
            reference=[NSData dataWithBytes:[bs[5] contents] length:[bs[5] length]];
            memset([bs[5] contents],0,[bs[5] length]);
            t=now_ms(); id<MTLCommandBuffer> cb=metal_async(split,bs,fault);
            double p0=now_ms(); pack(); double pack_ms=now_ms()-p0;
            double a0=now_ms(); id<MLFeatureProvider> result=[model predictionFromFeatures:provider options:pred error:&error];
            if (!result) { fprintf(stderr,"CoreML predict: %s\n",error.description.UTF8String); return 1; }
            double ane_ms=now_ms()-a0;
            MLMultiArray* y=[result featureValueForName:@"y"].multiArrayValue;
            same_backing=y.dataPointer==backing.dataPointer;
            double s0=now_ms(); scatter(y); double scatter_ms=now_ms()-s0;
            joined(cb,fault); double total_ms=now_ms()-t;
            double diff=0,norm=0; float worst=0; unsigned gpu_bad=0;
            const uint16_t* ref=reference.bytes; const uint16_t* actual=[bs[5] contents];
            for (unsigned row=0;row<M;row++) for (unsigned h=0;h<H;h++) for (unsigned d=0;d<D;d++) {
                unsigned i=(row*H+h)*D+d; float x=plow_bf2f(actual[i]), y=plow_bf2f(ref[i]); assert(isfinite(x)&&isfinite(y));
                if (h<gh) gpu_bad+=actual[i]!=ref[i];
                else { diff+=(double)(x-y)*(x-y); norm+=(double)y*y; worst=fmaxf(worst,fabsf(x-y)); }
            }
            double rel=sqrt(diff/fmax(norm,1e-30)); assert(gpu_bad==0);
            if (!(rel<0.03 && worst<0.03)) { fprintf(stderr,"parity failed rel_l2=%g max_abs=%g\n",rel,worst); return 1; }
            if (r>=warmup) { unsigned j=r-warmup; baseline[j]=base_ms; mixed[j]=total_ms; ane[j]=ane_ms; packs[j]=pack_ms; scatters[j]=scatter_ms; gpus[j]=(cb.GPUEndTime-cb.GPUStartTime)*1e3; }
            if (r==reps+warmup-1) printf("parity GPU heads exact; CoreML heads rel_l2=%.6f max_abs=%.6f backing=%s\n",rel,worst,same_backing?"used":"not used");
        }
        double base_ms=median(baseline,reps), split_ms=median(mixed,reps);
        printf("median: Metal-only %.4f ms; split %.4f ms; speedup %.3fx; CoreML %.4f ms; pack %.4f ms; scatter %.4f ms; concurrent GPU device %.4f ms\n",base_ms,split_ms,base_ms/split_ms,median(ane,reps),median(packs,reps),median(scatters,reps),median(gpus,reps));
        cfg.computeUnits=MLComputeUnitsCPUOnly;
        MLModel* cpu=[MLModel modelWithContentsOfURL:url configuration:cfg error:&error]; assert(cpu);
        for (unsigned r=0;r<reps+3;r++) { double t=now_ms(); assert([cpu predictionFromFeatures:provider options:pred error:&error]); if (r>=3) ane[r-3]=now_ms()-t; }
        printf("CPU-only same graph median %.4f ms (prediction only)\n",median(ane,reps));
        free(baseline); free(mixed); free(ane); free(packs); free(scatters); free(gpus);
    }
    return 0;
}
