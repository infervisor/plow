
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
static void enc(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso,
                id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> b,
                id<MTLBuffer> y, unsigned *p, unsigned groups, unsigned threads,
                size_t xo, size_t yo) {
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:x offset:16 + xo atIndex:0];
  [e setBuffer:w offset:16 atIndex:1];
  [e setBuffer:b offset:16 atIndex:2];
  [e setBuffer:y offset:16 + yo atIndex:3];
  [e setBytes:p length:20 atIndex:4];
  [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [e endEncoding];
}
static double run(id<MTLCommandQueue> q, NSArray *ps, NSArray *bs, unsigned ci,
                  unsigned n, unsigned f, unsigned t, unsigned batches,
                  bool candidate) {
  unsigned m = ((f + 1) / 2) * ((t + 1) / 2), k = ci * 9,
           p[5] = {ci, n, f, t, batches};
  id<MTLCommandBuffer> cb = [q commandBuffer];
  if (candidate)
    enc(cb, ps[0], bs[0], bs[1], bs[2], bs[3], p,
        batches * ((m + 31) / 32) * ((n + 63) / 64) + 1, 256, 0, 0);
  else if (ci == 1)
    enc(cb, ps[1], bs[0], bs[1], bs[2], bs[3], p,
        batches * ((m + 7) / 8) * ((n + 7) / 8), 32, 0, 0);
  else
    for (unsigned batch = 0; batch < batches; batch++) {
      unsigned pp[5] = {ci, n, f, t, 1}, mp[5] = {m, n, k, 1, 1};
      enc(cb, ps[2], bs[0], bs[0], bs[0], bs[4], pp, (m * k + 31) / 32, 32,
          batch * ci * f * t * 4, 0);
      enc(cb, ps[3], bs[4], bs[1], bs[2], bs[5], mp,
          ((m + 31) / 32) * ((n + 63) / 64), 256, 0, 0);
      enc(cb, ps[4], bs[5], bs[5], bs[5], bs[3], pp, (m * n + 31) / 32, 32, 0,
          batch * m * n * 4);
    }
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
    NSString *s = [NSString stringWithContentsOfFile:@(argv[1])
                                            encoding:NSUTF8StringEncoding
                                               error:&err];
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.mathMode = MTLMathModeSafe;
    o.languageVersion = MTLLanguageVersion3_2;
    id<MTLLibrary> lib = [d newLibraryWithSource:s options:o error:&err];
    if (!lib) {
      fprintf(stderr, "%s\n", err.description.UTF8String);
      return 1;
    }
    NSMutableArray *ps = [NSMutableArray array];
    for (NSString *name in @[
           @"asr_conv_implicit", @"asr_conv_tiled", @"asr_unfold",
           @"asr_linear_direct", @"asr_unpack_conv"
         ]) {
      id<MTLComputePipelineState> p =
          [d newComputePipelineStateWithFunction:[lib newFunctionWithName:name]
                                           error:&err];
      assert(p);
      [ps addObject:p];
    }
    unsigned shapes[][5] = {{1, 7, 7, 9, 2},        {3, 65, 13, 17, 2},
                            {480, 480, 64, 50, 1},  {480, 480, 32, 25, 1},
                            {1, 480, 128, 100, 16}, {480, 480, 64, 50, 16},
                            {480, 480, 32, 25, 16}};
    for (unsigned shape = 0; shape < 7; shape++) {
      @autoreleasepool {
        unsigned ci = shapes[shape][0], n = shapes[shape][1],
                 f = shapes[shape][2], t = shapes[shape][3],
                 batches = shapes[shape][4];
        unsigned m = ((f + 1) / 2) * ((t + 1) / 2), k = ci * 9;
        size_t sizes[] = {batches * ci * f * t, n * k, n,
                          batches * m * n,      m * k, m * n};
        NSMutableArray *bs = [NSMutableArray array];
        for (unsigned i = 0; i < 6; i++) {
          id<MTLBuffer> b =
              [d newBufferWithLength:(sizes[i] + 8) * 4
                             options:MTLResourceStorageModeShared];
          assert(b);
          memset(b.contents, 0x5a, b.length);
          [bs addObject:b];
        }
        for (unsigned i = 0; i < 3; i++) {
          id<MTLBuffer> b = bs[i];
          float *a = (float *)b.contents + 4;
          for (unsigned j = 0; j < sizes[i]; j++)
            a[j] = i == 0   ? ((int)(j % 61) - 30) / 16.0f
                   : i == 1 ? ((int)(j % 43) - 21) / 256.0f
                            : ((int)(j % 7) - 3) / 16.0f;
        }
        run(q, ps, bs, ci, n, f, t, batches, false);
        id<MTLBuffer> y = bs[3];
        NSData *ref = [NSData dataWithBytes:y.contents length:y.length];
        memset(y.contents, 0x5a, y.length);
        run(q, ps, bs, ci, n, f, t, batches, true);
        if (memcmp(ref.bytes, y.contents, y.length)) {
          unsigned diff = 0;
          float max = 0;
          const float *a = (const float *)ref.bytes;
          float *b = y.contents;
          for (unsigned j = 0; j < y.length / 4; j++)
            if (memcmp(a + j, b + j, 4)) {
              diff++;
              max = fmaxf(max, fabsf(a[j] - b[j]));
            }
          fprintf(stderr, "shape=%u differences=%u max=%g\n", shape, diff, max);
          return 2;
        }
        for (unsigned j = 0; j < 4; j++)
          assert(((unsigned *)y.contents)[j] == 0x5a5a5a5a &&
                 ((unsigned *)y.contents)[sizes[3] + 4 + j] == 0x5a5a5a5a);
        printf("shape=%u exact\n", shape);
        fflush(stdout);
        if (shape >= 2)
          for (unsigned r = 0; r < 12; r++) {
            double a, b;
            if (r % 2) {
              b = run(q, ps, bs, ci, n, f, t, batches, true);
              a = run(q, ps, bs, ci, n, f, t, batches, false);
            } else {
              a = run(q, ps, bs, ci, n, f, t, batches, false);
              b = run(q, ps, bs, ci, n, f, t, batches, true);
            }
            if (r >= 2)
              printf("shape=%u baseline_us=%.3f implicit_us=%.3f\n", shape, a,
                     b);
          }
      }
    }
  }
}
