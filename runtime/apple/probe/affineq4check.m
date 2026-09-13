#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define main affine_cpu_cases
#define g_gemv_affine_q4 gpu_gemv_affine_q4
#define g_gemm_affine_q4 gpu_gemm_affine_q4
#include "../../tests/affine_q4_cpu_test.c"
#undef main
#undef g_gemv_affine_q4
#undef g_gemm_affine_q4

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pso;

static void dispatch(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* T, int gemm) {
    PlowDevInst d = *in;
    d.op = gemm ? PLOW_DOP_GEMM_AFFINE_Q4 : PLOW_DOP_GEMV_AFFINE_Q4;
    d.blocks = nblk;
    unsigned M = in->i[0], N = in->i[1], K = in->i[2];
    size_t sizes[] = {(size_t)(M + in->i[5] + 1) * N * 2,
        (size_t)(M + in->i[4]) * K * 2, (size_t)N * K / 8 * 4,
        (size_t)N * K / 64 * 2, (size_t)N * K / 64 * 2};
    // The scalar rounding fixture has a single output element and no guard row.
    if (!in->i[5]) sizes[0] = (size_t)M * N * 2;
    NSMutableArray<id<MTLBuffer>>* bs = [NSMutableArray array];
    uint64_t tab[5];
    for (unsigned i = 0; i < 5; i++) {
        id<MTLBuffer> b = [dev newBufferWithLength:sizes[i] ? sizes[i] : 4 options:MTLResourceStorageModeShared];
        assert(b); if (sizes[i]) memcpy(b.contents, T[i], sizes[i]);
        [bs addObject:b]; tab[i] = b.gpuAddress;
    }
    id<MTLBuffer> fault = [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    *(unsigned*)fault.contents = 0;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    unsigned zero = 0;
    [enc setComputePipelineState:pso];
    [enc setBytes:&d length:sizeof(d) atIndex:0];
    [enc setBytes:tab length:sizeof(tab) atIndex:7];
    [enc setBytes:&zero length:4 atIndex:8];
    [enc setBuffer:fault offset:0 atIndex:9];
    for (id<MTLBuffer> b in bs) [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [enc setBytes:&slice length:4 atIndex:10];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    assert(cb.status == MTLCommandBufferStatusCompleted && *(unsigned*)fault.contents == 0);
    memcpy(T[0], bs[0].contents, sizes[0]);
}

G_K(gpu_gemv_affine_q4) { (void)ctx; @autoreleasepool { dispatch(in, slice, nblk, T, 0); } }
G_K(gpu_gemm_affine_q4) { (void)ctx; @autoreleasepool { dispatch(in, slice, nblk, T, 1); } }

static void real_weights(const char* path) {
    FILE* f = fopen(path, "rb"); assert(f);
    uint32_t h[4]; unsigned checks = 0;
    while (fread(h, sizeof(h), 1, f) == 1) { @autoreleasepool {
        unsigned M = h[0], N = h[1], K = h[2], gemm = h[3];
        size_t count = (size_t)M * N;
        size_t bytes[] = {(size_t)M * K * 2, (size_t)N * K / 2,
            (size_t)N * K / 64 * 2, (size_t)N * K / 64 * 2, count * 2};
        void* data[5];
        for (unsigned i = 0; i < 5; i++) {
            data[i] = malloc(bytes[i]); assert(data[i]);
            assert(fread(data[i], bytes[i], 1, f) == 1);
        }
        plow_bf16* out = calloc(count, 2); assert(out);
        void* tab[] = {out, data[0], data[1], data[2], data[3]};
        PlowDevInst in = {0};
        for (unsigned i = 0; i < 5; i++) in.t[i] = i;
        in.i[0] = M; in.i[1] = N; in.i[2] = K;
        dispatch(&in, 0, 1, tab, gemm);
        double error = 0, norm = 0; size_t different = 0;
        for (size_t i = 0; i < count; i++) {
            float a = plow_bf2f(((plow_bf16*)data[4])[i]), b = plow_bf2f(out[i]);
            assert(isfinite(a) && isfinite(b));
            error += (double)(a-b)*(a-b); norm += (double)a*a;
            different += a != b;
        }
        double rel = sqrt(error / norm);
        printf("real M=%u N=%u K=%u gemm=%u different=%zu relative_l2=%.9g\n", M,N,K,gemm,different,rel);
        assert(gemm ? rel < 0.001 : different == 0);
        free(out); for (unsigned i = 0; i < 5; i++) free(data[i]);
        checks++;
    } }
    assert(feof(f) && checks); fclose(f);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 2 || argc == 3);
        dev = MTLCreateSystemDefaultDevice(); queue = [dev newCommandQueue];
        NSError* error = nil;
        NSString* src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
        assert(src);
        // Select one partition so the shared CPU cases can verify exclusive ownership.
        src = [src stringByAppendingString:@"\nkernel void affine_check(device const Inst* ins [[buffer(0)]], device const ulong* tab [[buffer(7)]], device uint* fault [[buffer(9)]], constant uint& slice [[buffer(10)]], uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { threadgroup float tile[TILE_FLOATS]; Inst in=ins[0]; op_affine_q4(in,tab,slice,in.blocks,in.op==157,tile,lid,sg,lane); }\n"];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe; opts.languageVersion = MTLLanguageVersion3_2;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&error];
        if (!lib) { fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
        pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"affine_check"] error:&error];
        assert(pso && pso.maxTotalThreadsPerThreadgroup >= 1024);
        affine_cpu_cases();
        if (argc == 3) real_weights(argv[2]);
    }
    return 0;
}
