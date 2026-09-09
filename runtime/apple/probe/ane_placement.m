/* ane_placement.m — does CoreML actually place a compiled model's layers on the ANE?
 * The channel-MLP path (`plowrt --ane-mlp-placement <this binary>`) refuses to offload unless
 * this reports every layer preferring the ANE, because a graph the compiler silently moves to
 * the CPU costs a copy and wins nothing.
 *
 *   clang -fobjc-arc -O2 -framework CoreML -framework Foundation ane_placement.m -o ane_placement
 *   ./ane_placement <model.mlmodelc> [more.mlmodelc ...]
 * Prints one `<path> <layer> (<type>) preferred=<device>` line per layer.
 */
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

int main(int argc, const char** argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: ane_placement model.mlmodelc [...]\n"); return 2; }
        for (int i = 1; i < argc; i++) {
            MLModelConfiguration* cfg = [MLModelConfiguration new];
            cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[i]]];
            dispatch_semaphore_t done = dispatch_semaphore_create(0);
            __block MLComputePlan* plan = nil;
            [MLComputePlan loadContentsOfURL:url configuration:cfg completionHandler:^(MLComputePlan* p, NSError* e) {
                plan = p;
                if (e) fprintf(stderr, "%s\n", e.description.UTF8String);
                dispatch_semaphore_signal(done);
            }];
            if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) || !plan) return 1;
            for (MLModelStructureNeuralNetworkLayer* layer in plan.modelStructure.neuralNetwork.layers) {
                MLComputePlanDeviceUsage* usage = [plan computeDeviceUsageForNeuralNetworkLayer:layer];
                printf("%s %s (%s) preferred=%s\n", argv[i], layer.name.UTF8String, layer.type.UTF8String,
                    NSStringFromClass([(id)usage.preferredComputeDevice class]).UTF8String);
            }
        }
    }
    return 0;
}
