/* dtypecheck.m — golden check for the Metal interpreter's dtype arms: runs each op on the GPU
 * and against the CPU golden tier on the same operands, so a new fp8/mxfp4/KV encoding cannot
 * pass on shape alone. `--bench-attn` also times the flash arms.
 *
 *   clang -fobjc-arc -O2 -I../../cpu/dev -I../../common \
 *     -framework Metal -framework Foundation dtypecheck.m \
 *     -o dtypecheck -L../../../build-cpu -lplow_cpu_dev
 *   ./dtypecheck ../interp.metal [--bench-attn]
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "cpu_dev.h"
#include "fp8_common.h"

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pipeline;
static id<MTLComputePipelineState> scalar_pipeline;
static PlowCpuCtx ctx;
static unsigned checks;

static id<MTLBuffer> buffer(size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    assert(b);
    memset(b.contents, 0, bytes);
    return b;
}
static void fill(id<MTLBuffer> b, unsigned bytes, unsigned seed) {
    for (size_t i = 0; i < b.length / bytes; i++) {
        seed = seed * 1664525u + 1013904223u;
        float v = (int)(seed >> 24) / 128.0f - 1.0f;
        if (bytes == 1) ((uint8_t*)b.contents)[i] = plow_f32_to_e4m3(v);
        else if (bytes == 2) ((plow_bf16*)b.contents)[i] = plow_f2bf(v);
        else ((float*)b.contents)[i] = 0.01f + fabsf(v);
    }
}
static PlowDevInst inst(unsigned op, unsigned blocks) {
    PlowDevInst d = {.op = op, .blocks = blocks};
    for (unsigned i = 0; i < 8; i++) d.t[i] = PLOW_TENSOR_NONE;
    return d;
}
static double dispatch(PlowDevInst d, NSArray<id<MTLBuffer>>* bs, id<MTLComputePipelineState> pso) {
    uint64_t addresses[8] = {0};
    for (unsigned i = 0; i < bs.count; i++) addresses[i] = bs[i].gpuAddress;
    id<MTLBuffer> fault = buffer(4);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    unsigned zero = 0;
    [enc setComputePipelineState:pso];
    [enc setBytes:&d length:sizeof(d) atIndex:0];
    [enc setBytes:addresses length:sizeof(addresses) atIndex:7];
    [enc setBytes:&zero length:4 atIndex:8];
    [enc setBuffer:fault offset:0 atIndex:9];
    for (id<MTLBuffer> b in bs) [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [enc dispatchThreadgroups:MTLSizeMake(d.blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    assert(cb.status == MTLCommandBufferStatusCompleted && *(unsigned*)fault.contents == 0);
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}
/* Output widths: 1 = raw bytes, 2 = bf16, 4 = f32. Non-outputs have width zero. */
static void check(PlowDevInst d, NSArray<id<MTLBuffer>>* bs, const unsigned widths[8]) {
    uint64_t addresses[8] = {0};
    void* golden[8] = {0};
    for (unsigned i = 0; i < bs.count; i++) {
        addresses[i] = bs[i].gpuAddress;
        golden[i] = malloc(bs[i].length);
        memcpy(golden[i], bs[i].contents, bs[i].length);
    }
    for (unsigned s = 0; s < d.blocks; s++) assert(plow_cpu_exec(&d, s, d.blocks, golden, &ctx) == 0);
    void* scalar[8] = {0};
    if (d.op == PLOW_DOP_FLASH_PREFILL || d.op == PLOW_DOP_FLASH_PREFILL_FP8) {
        dispatch(d, bs, scalar_pipeline);
        for (unsigned i = 0; i < bs.count; i++) if (widths[i]) {
            scalar[i] = malloc(bs[i].length);
            memcpy(scalar[i], bs[i].contents, bs[i].length);
        }
    }
    id<MTLBuffer> fault = buffer(4);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    unsigned zero = 0;
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&d length:sizeof(d) atIndex:0];
    [enc setBytes:addresses length:sizeof(addresses) atIndex:7];
    [enc setBytes:&zero length:4 atIndex:8];
    [enc setBuffer:fault offset:0 atIndex:9];
    for (id<MTLBuffer> b in bs) [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [enc dispatchThreadgroups:MTLSizeMake(d.blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    assert(cb.status == MTLCommandBufferStatusCompleted && *(unsigned*)fault.contents == 0);
    for (unsigned i = 0; i < bs.count; i++) {
        unsigned width = widths[i];
        if (width) for (size_t e = 0; e < bs[i].length / width; e++) {
            float a, b;
            if (width == 1) {
                a = ((uint8_t*)bs[i].contents)[e]; b = ((uint8_t*)golden[i])[e];
            } else if (width == 2) {
                a = plow_bf2f(((plow_bf16*)bs[i].contents)[e]); b = plow_bf2f(((plow_bf16*)golden[i])[e]);
            } else {
                a = ((float*)bs[i].contents)[e]; b = ((float*)golden[i])[e];
            }
            float tol = width == 1 ? 0.0f : 0.002f + 0.02f * fabsf(b);
            if (a == -INFINITY && b == -INFINITY) continue;
            if (scalar[i]) {
                float s = width == 2 ? plow_bf2f(((plow_bf16*)scalar[i])[e]) : ((float*)scalar[i])[e];
                /* BF16 exp rounding can already differ between scalar Metal and libm near
                 * ties. Bound the new path against BOTH the original error and scalar Metal. */
                float scalar_tol = 0.0002f + 0.002f * fabsf(s);
                if (width == 2) scalar_tol = fmaxf(scalar_tol, fabsf(plow_bf2f(((plow_bf16*)scalar[i])[e] + 1) - s));
                if (!isfinite(s) || fabsf(a - s) > scalar_tol) {
                    fprintf(stderr, "op %u D=%u element %zu: tiled %g scalar Metal %g golden %g\n", d.op, d.i[6], e, a, s, b);
                    exit(1);
                }
                tol = fmaxf(tol, fabsf(s - b) + 0.0002f);
            }
            if (!isfinite(a) || !isfinite(b) || fabsf(a - b) > tol) {
                fprintf(stderr, "op %u blocks %u tensor %u element %zu: GPU %g golden %g\n", d.op, d.blocks, i, e, a, b);
                exit(1);
            }
        }
        free(golden[i]);
        free(scalar[i]);
    }
    checks++;
}

static void quant(unsigned blocks) {
    unsigned M = 5, K = 1031;
    NSArray* bs = @[buffer(M*K), buffer(M*K*2), buffer(M*4), buffer(M*K*2), buffer(M*K*2)];
    fill(bs[1], 2, 1); fill(bs[3], 2, 2); fill(bs[4], 2, 3);
    memset([bs[1] contents], 0, K*2);
    PlowDevInst d = inst(PLOW_DOP_QUANT_FP8, blocks);
    d.t[0] = 0; d.t[1] = 1; d.t[2] = 2; d.i[0] = M; d.i[1] = K;
    check(d, bs, (unsigned[8]){1,0,4});
    d.t[3] = 3; d.t[4] = 4; d.i[2] = 1;
    check(d, bs, (unsigned[8]){1,2,4});
    NSArray* norm = @[buffer(M*K*2), bs[1], buffer(K*2), bs[0], bs[2]];
    fill(norm[2], 2, 5);
    d = inst(PLOW_DOP_RMSNORM, blocks);
    for (unsigned i = 0; i < 5; i++) d.t[i] = i;
    d.i[0] = M; d.i[1] = K; d.fj[0].f = 1e-6f;
    check(d, norm, (unsigned[8]){2,0,0,1,4});
}
static void gemm(unsigned blocks) {
    unsigned M = 137, N = 71, K = 301;
    NSArray* bs = @[buffer((M+2)*N*2), buffer((M+3)*K), buffer(N*K), buffer((M+3)*4), buffer(N*4), buffer(N*K), buffer(N*4)];
    fill(bs[1],1,1); fill(bs[2],1,2); fill(bs[3],4,3); fill(bs[4],4,4); fill(bs[5],1,5); fill(bs[6],4,6);
    unsigned ops[] = {33,34,35,100,101,36};
    for (unsigned j = 0; j < sizeof(ops)/sizeof(*ops); j++) {
        PlowDevInst d = inst(ops[j], blocks);
        for (unsigned i = 0; i < 7; i++) d.t[i] = i;
        d.i[0] = M; d.i[1] = N; d.i[2] = K;
        if (d.op == 36) d.i[5] = 1;
        else { d.i[4] = 3; d.i[5] = 2; }
        check(d, bs, (unsigned[8]){2});
    }
}
static void attention(unsigned D, unsigned blocks, unsigned M, unsigned width) {
    unsigned B = 2, H = 6, HK = 2, stride = 64, split = 3;
    NSArray* bs = @[buffer(B*H*split*D*4), buffer(B*H*split*2*4), buffer(B*H*D*2), buffer(B*HK*stride*D*width), buffer(B*HK*stride*D*width), buffer(B*4), buffer(B*HK*stride*4), buffer(B*HK*stride*4)];
    fill(bs[2],2,1); fill(bs[3],width,2); fill(bs[4],width,3); fill(bs[6],4,4); fill(bs[7],4,5);
    ((int*)[bs[5] contents])[0] = 1;
    ((int*)[bs[5] contents])[1] = 91;
    PlowDevInst d = inst(width == 1 ? PLOW_DOP_FLASH_DECODE_FP8 : PLOW_DOP_FLASH_DECODE, blocks);
    for (unsigned i = 0; i < 8; i++) d.t[i] = i;
    d.i[0] = B; d.i[1] = H; d.i[2] = HK; d.i[3] = stride; d.i[4] = 29;
    d.i[5] = split; d.i[6] = D; d.i[7] = stride-1; d.fj[0].f = 1.0f/sqrtf(D);
    check(d, bs, (unsigned[8]){4,4});
    NSArray* pf = @[buffer(M*H*split*D*4), buffer(M*H*split*2*4), buffer(M*H*D*2), bs[3], bs[4], buffer(M*H*D*2), bs[6], bs[7]];
    fill(pf[2],2,6);
    d = inst(width == 1 ? PLOW_DOP_FLASH_PREFILL_FP8 : PLOW_DOP_FLASH_PREFILL, blocks);
    for (unsigned i = 0; i < 8; i++) d.t[i] = i;
    d.i[0] = M; d.i[1] = 82+M; d.i[2] = H; d.i[3] = HK; d.i[4] = 82; d.i[5] = 29;
    d.i[6] = D; d.i[7] = split; d.fj[0].f = 1.0f/sqrtf(D); d.fj[1].u = stride; d.fj[2].u = stride-1;
    check(d, pf, (unsigned[8]){4,4,0,0,0,2});
    d.i[7] = 1;
    check(d, pf, (unsigned[8]){4,4,0,0,0,2});
    if (width != 1) return;
    NSArray* rope = @[bs[3], buffer(B*HK*D*2), buffer(D*2), buffer(128*D/2*4), buffer(128*D/2*4), bs[5], bs[6]];
    fill(rope[1],2,7); fill(rope[2],2,8); fill(rope[3],4,9); fill(rope[4],4,10);
    for (unsigned form = 0; form < 3; form++) {
        d = inst(PLOW_DOP_HEADNORM_ROPE_FP8, blocks);
        for (unsigned i = 0; i < 7; i++) d.t[i] = i;
        d.i[0] = B; d.i[1] = HK; d.i[2] = D; d.i[5] = form; d.i[6] = B;
        d.fj[0].f = 1e-6f; d.fj[1].u = stride; d.fj[2].u = stride-1;
        check(d, rope, (unsigned[8]){1,0,0,0,0,0,4});
    }
    memset([rope[1] contents], 0, [rope[1] length]);
    d.i[4] = 1; d.t[2] = d.t[3] = d.t[4] = PLOW_TENSOR_NONE;
    check(d, rope, (unsigned[8]){1,0,0,0,0,0,4});
}
static void bench_attention(void) {
    for (unsigned width = 1; width <= 2; width++) {
        for (unsigned D = 128; D <= 256; D *= 2) {
            unsigned H = 24, HK = 8, M = 128;
            for (unsigned L = 128; L <= 2048; L *= 4) {
                NSArray* bs = @[buffer(M*H*D*4), buffer(M*H*2*4), buffer(M*H*D*2), buffer(HK*L*D*width), buffer(HK*L*D*width), buffer(M*H*D*2), buffer(HK*L*4), buffer(HK*L*4)];
                fill(bs[2],2,1); fill(bs[3],width,2); fill(bs[4],width,3); fill(bs[6],4,4); fill(bs[7],4,5);
                for (unsigned decode = 0; decode <= 1; decode++) {
                    PlowDevInst d = inst(decode ? (width == 1 ? 38 : 12) : (width == 1 ? 39 : 11), 16);
                    for (unsigned i = 0; i < 8; i++) d.t[i] = i;
                    d.fj[0].f = 1.0f/sqrtf(D);
                    if (decode) {
                        ((int*)[bs[5] contents])[0] = L;
                        d.i[0] = 1; d.i[1] = H; d.i[2] = HK; d.i[3] = L;
                        d.i[5] = 1; d.i[6] = D; d.i[7] = L-1;
                    } else {
                        d.i[0] = M; d.i[1] = L; d.i[2] = H; d.i[3] = HK; d.i[4] = L-M;
                        d.i[6] = D; d.i[7] = 1; d.fj[1].u = L; d.fj[2].u = L-1;
                    }
                    double best = INFINITY;
                    for (unsigned r = 0; r < 11; r++) {
                        double ms = dispatch(d, bs, pipeline);
                        if (r) best = fmin(best, ms);
                    }
                    double pairs = decode ? L : (double)M * (2*L-M+1)/2;
                    double flops = 4.0 * H * D * pairs;
                    double bytes = 2.0*HK*L*D*width + (width == 1 ? 8.0*HK*L : 0) + 4.0*(decode ? 1 : M)*H*D;
                    printf("%s kv=%s D=%u H=%u HK=%u M=%u L=%u device_ms=%.4f GFLOPS=%.1f AI_minbytes=%.2f ideal_GBs=%.2f\n",
                        decode ? "decode" : "prefill", width == 1 ? "fp8" : "bf16", D,H,HK,decode ? 1 : M,L,best,flops/best/1e6,flops/bytes,bytes/best/1e6);
                }
            }
        }
    }
}
int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 2 || (argc == 3 && strcmp(argv[2], "--bench-attn") == 0));
        dev = MTLCreateSystemDefaultDevice(); assert(dev);
        queue = [dev newCommandQueue];
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe; options.languageVersion = MTLLanguageVersion3_2;
        id<MTLLibrary> library = [dev newLibraryWithSource:source options:options error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
        pipeline = [dev newComputePipelineStateWithFunction:[library newFunctionWithName:@"plow_single"] error:&error];
        assert(pipeline);
        if (argc == 3) { bench_attention(); return 0; }
        NSString* scalar_source = [source stringByReplacingOccurrencesOfString:@"switch (in.i[6]) {" withString:@"switch (0u) {"];
        assert(![source isEqualToString:scalar_source]);
        id<MTLLibrary> scalar_library = [dev newLibraryWithSource:scalar_source options:options error:&error];
        assert(scalar_library);
        scalar_pipeline = [dev newComputePipelineStateWithFunction:[scalar_library newFunctionWithName:@"plow_single"] error:&error];
        assert(scalar_pipeline);
        assert(plow_cpu_init(PLOW_CPU_ISA_SCALAR) >= 0);
        assert(plow_cpu_thread_init(&ctx) == 0);
        for (unsigned blocks = 1; blocks <= 3; blocks += 2) {
            quant(blocks); gemm(blocks);
            for (unsigned D = 64; D <= 512; D *= 2) {
                for (unsigned width = 1; width <= 2; width++) {
                    attention(D, blocks, 9, width); attention(D, blocks, 137, width);
                }
            }
        }
        printf("Metal dtype checks: %u passed\n", checks);
    }
    return 0;
}
