
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
static double run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso,
                  NSArray<id<MTLBuffer>> *bs, unsigned m, unsigned h,
                  bool staged) {
  unsigned params[5] = {m, h, 0, 0, 0};
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  for (unsigned i = 0; i < 4; i++)
    [e setBuffer:bs[i] offset:16 atIndex:i];
  [e setBytes:params length:sizeof(params) atIndex:4];
  [e dispatchThreadgroups:MTLSizeMake(staged ? m : (m + 31) / 32, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  assert(cb.status == MTLCommandBufferStatusCompleted);
  return (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
}
int main(int argc, char **argv) {
  @autoreleasepool {
    assert(argc == 2);
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [d newCommandQueue];
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:@(argv[1])
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.mathMode = MTLMathModeSafe;
    o.languageVersion = MTLLanguageVersion3_2;
    id<MTLLibrary> lib = [d newLibraryWithSource:src options:o error:&err];
    if (!lib) {
      fprintf(stderr, "%s\n", err.description.UTF8String);
      return 1;
    }
    id<MTLComputePipelineState> a = [d
        newComputePipelineStateWithFunction:[lib
                                                newFunctionWithName:@"asr_norm"]
                                      error:&err];
    id<MTLComputePipelineState> b =
        [d newComputePipelineStateWithFunction:
                [lib newFunctionWithName:@"asr_norm_staged"]
                                         error:&err];
    assert(a && b);
    unsigned ms[] = {64, 128, 129, 208, 255, 256, 257, 320, 391},
             hs[] = {1, 31, 33, 1024};
    unsigned cases = 0;
    for (unsigned mi = 0; mi < 9; mi++)
      for (unsigned hi = 0; hi < 4; hi++)
        for (unsigned fixture = 0; fixture < 3; fixture++) {
          @autoreleasepool {
            unsigned m = ms[mi], h = hs[hi];
            NSMutableArray *bs = [NSMutableArray array];
            for (unsigned i = 0; i < 4; i++) {
              size_t n = i == 1 || i == 2 ? h : m * h;
              id<MTLBuffer> v =
                  [d newBufferWithLength:(n + 8) * sizeof(float)
                                 options:MTLResourceStorageModeShared];
              memset(v.contents, 0x5a, v.length);
              [bs addObject:v];
            }
            for (unsigned i = 0; i < 3; i++) {
              id<MTLBuffer> v = bs[i];
              float *f = (float *)v.contents + 4;
              unsigned n = i ? h : m * h;
              for (unsigned j = 0; j < n; j++)
                f[j] = i == 1         ? 1.0f + ((int)(j % 7) - 3) / 16.0f
                       : i == 2       ? ((int)(j % 11) - 5) / 8.0f
                       : fixture == 0 ? ((int)(j % 251) - 125) / 32.0f
                       : fixture == 1 ? 64.0f
                                      : 64.0f + ((int)(j % 5) - 2) * 0.5f;
            }
            run(q, a, bs, m, h, false);
            id<MTLBuffer> y = bs[3];
            NSData *reference = [NSData dataWithBytes:y.contents
                                               length:y.length];
            memset(y.contents, 0x5a, y.length);
            run(q, b, bs, m, h, true);
            if (memcmp(reference.bytes, y.contents, y.length)) {
              fprintf(stderr, "mismatch m=%u h=%u fixture=%u\n", m, h, fixture);
              return 2;
            }
            for (unsigned j = 0; j < 4; j++)
              assert(((unsigned *)y.contents)[j] == 0x5a5a5a5a &&
                     ((unsigned *)y.contents)[m * h + 4 + j] == 0x5a5a5a5a);
            cases++;
            if (h == 1024 && fixture == 0 && m >= 64)
              for (unsigned r = 0; r < 22; r++) {
                double ta, tb;
                if (r % 2) {
                  tb = run(q, b, bs, m, h, true);
                  ta = run(q, a, bs, m, h, false);
                } else {
                  ta = run(q, a, bs, m, h, false);
                  tb = run(q, b, bs, m, h, true);
                }
                if (r >= 2)
                  printf("m=%u baseline_us=%.3f staged_us=%.3f\n", m, ta, tb);
              }
          }
        }
    printf("exact cases=%u\n", cases);
  }
}
