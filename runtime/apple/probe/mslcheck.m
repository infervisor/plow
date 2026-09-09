/* mslcheck.m — compile a .metal file with the system runtime compiler and report errors.
 *   clang -fobjc-arc -framework Metal -framework Foundation mslcheck.m -o mslcheck && ./mslcheck file.metal [fn] */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: mslcheck file.metal [function]\n"); return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "read: %s\n", err.localizedDescription.UTF8String); return 2; }
        MTLCompileOptions *opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 1; }
        if (err) fprintf(stderr, "warnings: %s\n", err.localizedDescription.UTF8String);
        NSString *fn = argc > 2 ? @(argv[2]) : @"plow_interp";
        id<MTLFunction> f = [lib newFunctionWithName:fn];
        if (!f) { fprintf(stderr, "no function %s\n", fn.UTF8String); return 1; }
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
        if (!p) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
        printf("ok: %s maxThreads=%lu tgmem=%lu\n", fn.UTF8String, (unsigned long)p.maxTotalThreadsPerThreadgroup,
               (unsigned long)p.staticThreadgroupMemoryLength);
    }
    return 0;
}
