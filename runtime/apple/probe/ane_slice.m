/* ane_slice.m — compile one hand-written NeuralNetwork graph, run it, and report both where
 * CoreML placed it and how far its fp16 output drifts from an f32 reference. Used to qualify a
 * candidate ANE sub-graph (the column slice the channel MLP relies on) before wiring it up.
 *
 *   clang -fobjc-arc -O2 -framework CoreML -framework Foundation ane_slice.m -o ane_slice
 *   ./ane_slice <model.mlmodel>
 * Prints per-layer placement then `placement=<ane>/<total> rel_l2=… max_abs=…`.
 */
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <math.h>

static void vi(NSMutableData* d, uint64_t x) {
    do { uint8_t b=x&127; x>>=7; if (x) b|=128; [d appendBytes:&b length:1]; } while (x);
}
static void integer(NSMutableData* d, unsigned f, uint64_t x) { vi(d,f*8); vi(d,x); }
static void message(NSMutableData* d, unsigned f, NSData* x) { vi(d,f*8+2); vi(d,x.length); [d appendData:x]; }
static void string(NSMutableData* d, unsigned f, NSString* x) { message(d,f,[x dataUsingEncoding:NSUTF8StringEncoding]); }
static void ints(NSMutableData* d, unsigned f, NSArray<NSNumber*>* xs) {
    NSMutableData* p=[NSMutableData data]; for (NSNumber* x in xs) vi(p,x.longLongValue); message(d,f,p);
}
static NSData* feature(NSString* name, unsigned m, unsigned n) {
    NSMutableData *a=[NSMutableData data],*t=[NSMutableData data],*f=[NSMutableData data];
    ints(a,1,@[@(m),@(n)]); integer(a,2,65568); message(t,5,a); string(f,1,name); message(f,3,t); return f;
}
static void layer(NSMutableData* nn, NSString* name, NSArray* inputs, unsigned tag, NSData* p) {
    NSMutableData* d=[NSMutableData data]; string(d,1,name); for (NSString* i in inputs) string(d,2,i);
    string(d,3,name); message(d,tag,p); message(nn,1,d);
}
int main(void) {
    @autoreleasepool {
        const unsigned M=64,K=4096,N=1024,C=2048;
        NSMutableData *desc=[NSMutableData data],*nn=[NSMutableData data],*model=[NSMutableData data];
        message(desc,1,feature(@"x",M,K)); message(desc,10,feature(@"y",M,N));
        for (unsigned b=0;b<2;b++) {
            NSMutableData* s=[NSMutableData data];
            ints(s,1,@[@0,@(b*C)]); ints(s,2,@[@1,@0]); ints(s,3,@[@0,@((b+1)*C)]);
            ints(s,4,@[@1,@0]); ints(s,5,@[@1,@1]);
            NSString* slice=[NSString stringWithFormat:@"s%u",b]; layer(nn,slice,@[@"x"],995,s);
            NSMutableData* w=[NSMutableData dataWithLength:N*C*2];
            for (unsigned n=0;n<N;n++) for (unsigned k=0;k<C;k++)
                ((_Float16*)w.mutableBytes)[n*C+k]=(_Float16)(((int)((n*13+k+b*C)%31)-15)*0.001f);
            NSMutableData *wp=[NSMutableData data],*ip=[NSMutableData data]; message(wp,2,w);
            integer(ip,1,C); integer(ip,2,N); message(ip,20,wp);
            layer(nn,[NSString stringWithFormat:@"y%u",b],@[slice],140,ip);
        }
        layer(nn,@"y",@[@"y0",@"y1"],880,[NSData data]); integer(nn,5,1);
        integer(model,1,5); message(model,2,desc); message(model,500,nn);
        NSString* root=[NSTemporaryDirectory() stringByAppendingPathComponent:[@"plow-slice-" stringByAppendingString:NSUUID.UUID.UUIDString]];
        [[NSFileManager defaultManager] createDirectoryAtPath:root withIntermediateDirectories:NO attributes:nil error:nil];
        NSURL* source=[NSURL fileURLWithPath:[root stringByAppendingPathComponent:@"slice.mlmodel"]];
        assert([model writeToURL:source atomically:YES]); NSError* e=nil;
        NSURL* compiled=[MLModel compileModelAtURL:source error:&e]; if (!compiled) { NSLog(@"%@",e); return 1; }
        MLModelConfiguration* cfg=[MLModelConfiguration new]; cfg.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
        dispatch_semaphore_t done=dispatch_semaphore_create(0); __block MLComputePlan* plan=nil;
        [MLComputePlan loadContentsOfURL:compiled configuration:cfg completionHandler:^(MLComputePlan* p,NSError* error) {
            plan=p; if (error) NSLog(@"%@",error); dispatch_semaphore_signal(done);
        }];
        if (dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC)) || !plan) return 1;
        unsigned ane=0,total=0;
        for (MLModelStructureNeuralNetworkLayer* l in plan.modelStructure.neuralNetwork.layers) {
            id d=[plan computeDeviceUsageForNeuralNetworkLayer:l].preferredComputeDevice;
            printf("%s %s preferred=%s\n",l.name.UTF8String,l.type.UTF8String,NSStringFromClass([d class]).UTF8String);
            total++; ane += [d isKindOfClass:MLNeuralEngineComputeDevice.class];
        }
        MLModel* net=[MLModel modelWithContentsOfURL:compiled configuration:cfg error:&e]; assert(net);
        MLMultiArray* x=[[MLMultiArray alloc] initWithShape:@[@(M),@(K)] dataType:MLMultiArrayDataTypeFloat32 error:&e]; assert(x);
        float* xp=x.dataPointer;
        for (unsigned j=0;j<M*K;j++) xp[j]=(int)(j%23)/11.0f-1.0f;
        MLDictionaryFeatureProvider* in=[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x":[MLFeatureValue featureValueWithMultiArray:x]} error:&e]; assert(in);
        id<MLFeatureProvider> out=[net predictionFromFeatures:in error:&e]; if (!out) { NSLog(@"%@",e); return 1; }
        MLMultiArray* y=[out featureValueForName:@"y"].multiArrayValue; assert(y);
        double err=0,norm=0; float maxerr=0;
        for (unsigned n=0;n<N;n++) {
            float ref=0; for (unsigned k=0;k<K;k++) ref += xp[k]*(float)(_Float16)(((int)((n*13+k)%31)-15)*0.001f);
            float actual=[y objectAtIndexedSubscript:n].floatValue;
            assert(isfinite(actual)); err+=(actual-ref)*(actual-ref); norm+=ref*ref; maxerr=fmaxf(maxerr,fabsf(actual-ref));
        }
        printf("placement=%u/%u rel_l2=%g max_abs=%g source=%s compiled=%s\n",ane,total,sqrt(err/norm),maxerr,root.UTF8String,compiled.path.UTF8String);
        return ane==total && total==5 && sqrt(err/norm)<0.03 ? 0 : 2;
    }
}
