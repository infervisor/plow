/* probe.m — rung 0 probes for the Apple Silicon backend (plans/apple-silicon-backend.md §5.1).
 *
 *   clang -fobjc-arc -O2 -framework Metal -framework Foundation probe.m -o probe
 *   ./probe coresident   (b) largest co-resident threadgroup grid, per threadgroup-memory size
 *   ./probe counters     (c) in-kernel device-scope counter handoff latency (ping-pong, chain)
 *   ./probe live         (a) CPU<->GPU flag visibility while a kernel runs, round-trip latency
 *   ./probe events       (d) command-buffer cost and MTLSharedEvent GPU->CPU->GPU boundary cost
 *   ./probe gemv         (e) bf16 GEMV weight-streaming bandwidth: GPU alone, CPU alone, both
 *
 * MSL is compiled at runtime from the string below (no Xcode metal toolchain needed). */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <arm_neon.h>

static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/* (b) every threadgroup arrives, then spins until all G have arrived. */
kernel void coresident(device atomic_uint* arrive [[buffer(0)]], device uint* result [[buffer(1)]],
                       constant uint& G [[buffer(2)]], constant uint& spin_max [[buffer(3)]],
                       threadgroup uint* tgm [[threadgroup(0)]],
                       uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    if (lid == 0) {
        tgm[0] = tg;
        atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed);
        uint it = 0;
        while (atomic_load_explicit(arrive, memory_order_relaxed) < G && it < spin_max) it++;
        result[tg] = it < spin_max ? it : 0xFFFFFFFFu;
    }
}

/* (c) two threadgroups ping-pong a device counter N rounds; the rest of the grid idles so
 * the grid still matches the resident count. Chain: threadgroup g waits for ctr == g, bumps. */
kernel void pingpong(device atomic_uint* ctr [[buffer(0)]], device uint* result [[buffer(1)]],
                     constant uint& N [[buffer(2)]], constant uint& spin_max [[buffer(3)]],
                     uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    if (lid != 0 || tg > 1) return;
    uint fails = 0;
    for (uint i = 0; i < N; i++) {
        const uint want = 2u * i + tg, next = want + 1u;
        uint it = 0;
        while (atomic_load_explicit(ctr, memory_order_relaxed) != want && it < spin_max) it++;
        if (it >= spin_max) fails++;
        atomic_store_explicit(ctr, next, memory_order_relaxed);
    }
    result[tg] = fails;
}
kernel void chain(device atomic_uint* ctr [[buffer(0)]], device uint* result [[buffer(1)]],
                  constant uint& G [[buffer(2)]], constant uint& spin_max [[buffer(3)]],
                  uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    if (lid != 0) return;
    uint it = 0;
    while (atomic_load_explicit(ctr, memory_order_relaxed) != tg && it < spin_max) it++;
    threadgroup_barrier(mem_flags::mem_device);
    atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
    result[tg] = it < spin_max ? it : 0xFFFFFFFFu;
}

/* (a) GPU writes flag[0] = 2i+1 and waits for the CPU to answer flag[1] = 2i+1. */
kernel void live(device atomic_uint* flag [[buffer(0)]], device uint* result [[buffer(1)]],
                 constant uint& N [[buffer(2)]], constant uint& spin_max [[buffer(3)]],
                 uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    if (lid != 0 || tg != 0) return;
    uint fails = 0, total_it = 0;
    for (uint i = 0; i < N; i++) {
        const uint v = 2u * i + 1u;
        atomic_store_explicit(flag, v, memory_order_relaxed);
        uint it = 0;
        while (atomic_load_explicit(flag + 1, memory_order_relaxed) != v && it < spin_max) it++;
        if (it >= spin_max) fails++;
        total_it += it;
    }
    result[0] = fails;
    result[1] = total_it;
}

/* (f) cross-threadgroup data visibility inside one dispatch. tg1 optionally pre-reads X (warms
 * its L1), then spins on flag; tg0 writes X (plain stores), fences, sets flag; tg1 re-reads X
 * plainly and through a volatile pointer and reports what it saw. N rounds with a fresh value. */
kernel void visibility(device uint* X [[buffer(0)]], device atomic_uint* flag [[buffer(1)]],
                       device uint* result [[buffer(2)]], constant uint& N [[buffer(3)]],
                       constant uint& prewarm [[buffer(4)]],
                       uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    if (tg > 1) return;
    uint stale_plain = 0, stale_volatile = 0, spins = 0;
    for (uint r = 1; r <= N; r++) {
        if (tg == 0) {
            if (lid == 0) {
                /* wait for tg1 to have (pre)read round r-1 */
                while (atomic_load_explicit(flag + 1, memory_order_relaxed) != r - 1) {}
            }
            threadgroup_barrier(mem_flags::mem_device);
            if (prewarm == 2u) { for (uint i = lid; i < 4096; i += 256) atomic_store_explicit((device atomic_uint*)X + i, r, memory_order_relaxed); }
            else { for (uint i = lid; i < 4096; i += 256) X[i] = r; }
            threadgroup_barrier(mem_flags::mem_device);
#ifdef PLOW_TRY_FENCE
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
            if (lid == 0) atomic_store_explicit(flag, r, memory_order_relaxed);
        } else {
            uint sum_pre = 0;
            if (prewarm) for (uint i = lid; i < 4096; i += 256) sum_pre += X[i];
            threadgroup_barrier(mem_flags::mem_device);
            if (lid == 0) atomic_store_explicit(flag + 1, r - 1, memory_order_relaxed);
            if (lid == 0) { uint it = 0; while (atomic_load_explicit(flag, memory_order_relaxed) != r && it < 100000000u) it++; spins += it; }
            threadgroup_barrier(mem_flags::mem_device);
            uint bad_p = 0, bad_v = 0;
            for (uint i = lid; i < 4096; i += 256) {
                if (X[i] != r) bad_p++;
                if (atomic_load_explicit((device atomic_uint*)X + i, memory_order_relaxed) != r) bad_v++;
            }
            stale_plain += bad_p;
            stale_volatile += bad_v;
            result[3] += sum_pre;
        }
    }
    if (tg == 1 && lid == 0) { result[0] = stale_plain; result[1] = stale_volatile; result[2] = spins; }
    if (tg == 1) { /* fold per-thread counts */ }
}

kernel void tiny(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0) out[0] += 1u;
}

/* (e) y[n] = W[n][:] . x, bf16 weights as ushort, one SIMD group per row, 8 bf16 per lane. */
kernel void gemv_bf16(device const ushort* W [[buffer(0)]], device const ushort* x [[buffer(1)]],
                      device float* y [[buffer(2)]], constant uint& N [[buffer(3)]],
                      constant uint& K [[buffer(4)]],
                      uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tid / 32u;
    if (row >= N) return;
    device const ushort4* w = (device const ushort4*)(W + (size_t)row * K);
    device const ushort4* xv = (device const ushort4*)x;
    float acc = 0.0f;
    const uint n4 = K / 4u;
    for (uint i = lane; i < n4; i += 32u) {
        const ushort4 a = w[i], b = xv[i];
        const float4 af = as_type<float4>(uint4(a) << 16u);
        const float4 bf = as_type<float4>(uint4(b) << 16u);
        acc += dot(af, bf);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}
)MSL";

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> lib;

static double now_s(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

static id<MTLComputePipelineState> pso(NSString *name) {
    NSError *err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:name];
    if (!f) { fprintf(stderr, "no function %s\n", name.UTF8String); exit(1); }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) { fprintf(stderr, "pso %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
    return p;
}

static id<MTLBuffer> buf(size_t bytes) {
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

/* --- (b) ------------------------------------------------------------------------------ */
static bool coresident_ok(id<MTLComputePipelineState> p, uint32_t G, uint32_t tgmem, uint32_t T) {
    id<MTLBuffer> arrive = buf(64), res = buf(G * 4);
    memset(arrive.contents, 0, 64);
    uint32_t spin = 2000000u;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:arrive offset:0 atIndex:0];
    [e setBuffer:res offset:0 atIndex:1];
    [e setBytes:&G length:4 atIndex:2];
    [e setBytes:&spin length:4 atIndex:3];
    [e setThreadgroupMemoryLength:tgmem ? tgmem : 16 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(G, 1, 1) threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const uint32_t *r = res.contents;
    for (uint32_t i = 0; i < G; i++) if (r[i] == 0xFFFFFFFFu) return false;
    return true;
}

static void probe_coresident(void) {
    id<MTLComputePipelineState> p = pso(@"coresident");
    printf("maxTotalThreadsPerThreadgroup=%lu threadExecutionWidth=%lu maxThreadgroupMemory=%lu\n",
           (unsigned long)p.maxTotalThreadsPerThreadgroup, (unsigned long)p.threadExecutionWidth,
           (unsigned long)dev.maxThreadgroupMemoryLength);
    const uint32_t tgmems[] = {0, 16384, 32768};
    const uint32_t Ts[] = {256, 512, 1024};
    for (int ti = 0; ti < 3; ti++)
        for (int mi = 0; mi < 3; mi++) {
            uint32_t lo = 1, hi = 512; /* find largest G that passes 3 times */
            while (lo < hi) {
                uint32_t mid = (lo + hi + 1) / 2;
                bool ok = true;
                for (int k = 0; k < 3 && ok; k++) ok = coresident_ok(p, mid, tgmems[mi], Ts[ti]);
                if (ok) lo = mid; else hi = mid - 1;
            }
            printf("  T=%-4u tgmem=%-5u  max co-resident threadgroups = %u\n", Ts[ti], tgmems[mi], lo);
        }
}

/* --- (c) ------------------------------------------------------------------------------ */
static void probe_counters(void) {
    id<MTLComputePipelineState> pp = pso(@"pingpong"), pc = pso(@"chain");
    const uint32_t N = 100000, spin = 50000000u, G = 16;
    id<MTLBuffer> ctr = buf(64), res = buf(1024);
    memset(ctr.contents, 0, 64);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pp];
    [e setBuffer:ctr offset:0 atIndex:0];
    [e setBuffer:res offset:0 atIndex:1];
    [e setBytes:&N length:4 atIndex:2];
    [e setBytes:&spin length:4 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(G, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const uint32_t *r = res.contents;
    double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
    printf("pingpong: %u rounds, fails tg0=%u tg1=%u, %.1f us per handoff (GPU time %.3f s)\n", N, r[0], r[1],
           gpu_s / (2.0 * N) * 1e6, gpu_s);
    for (uint32_t Gc = 16; Gc <= 256; Gc *= 4) {
        memset(ctr.contents, 0, 64);
        id<MTLBuffer> rc = buf(Gc * 4);
        cb = [queue commandBuffer];
        e = [cb computeCommandEncoder];
        [e setComputePipelineState:pc];
        [e setBuffer:ctr offset:0 atIndex:0];
        [e setBuffer:rc offset:0 atIndex:1];
        [e setBytes:&Gc length:4 atIndex:2];
        [e setBytes:&spin length:4 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(Gc, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const uint32_t *q = rc.contents;
        uint32_t fails = 0;
        for (uint32_t i = 0; i < Gc; i++) fails += q[i] == 0xFFFFFFFFu;
        gpu_s = cb.GPUEndTime - cb.GPUStartTime;
        printf("chain G=%-3u: fails=%u  %.1f us per hop\n", Gc, fails, gpu_s / Gc * 1e6);
    }
}

/* --- (a) ------------------------------------------------------------------------------ */
static void probe_live(void) {
    id<MTLComputePipelineState> p = pso(@"live");
    const uint32_t N = 4, spin = 20000000u;
    id<MTLBuffer> flag = buf(256), res = buf(64);
    memset(flag.contents, 0, 256);
    _Atomic uint32_t *f = (_Atomic uint32_t *)flag.contents;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:flag offset:0 atIndex:0];
    [e setBuffer:res offset:0 atIndex:1];
    [e setBytes:&N length:4 atIndex:2];
    [e setBytes:&spin length:4 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [cb commit];
    /* CPU side: answer each GPU flag as soon as it is visible. */
    uint32_t cpu_timeouts = 0;
    double t0 = now_s(), t_first = 0;
    for (uint32_t i = 0; i < N; i++) {
        const uint32_t v = 2u * i + 1u;
        double td = now_s();
        while (atomic_load_explicit(&f[0], memory_order_relaxed) != v) {
            if (now_s() - td > 2.0) { cpu_timeouts++; break; }
        }
        if (i == 0) t_first = now_s() - t0;
        atomic_store_explicit(&f[1], v, memory_order_relaxed);
    }
    double t1 = now_s();
    [cb waitUntilCompleted];
    const uint32_t *r = res.contents;
    printf("live: GPU->CPU->GPU rounds=%u  gpu_timeouts=%u cpu_timeouts=%u  first flag seen after %.1f ms  "
           "%.1f us per round trip (cpu wall)  status=%ld\n",
           N, r[0], cpu_timeouts, t_first * 1e3, (t1 - t0 - t_first) / (N - 1) * 1e6, (long)cb.status);
    printf("      verdict: %s\n", (r[0] == 0 && cpu_timeouts == 0)
                                  ? "LiveCounter — both directions visible mid-kernel"
                                  : "Event mode — mid-kernel visibility failed in at least one direction");
}

/* --- (d) ------------------------------------------------------------------------------ */
static void probe_events(void) {
    id<MTLComputePipelineState> p = pso(@"tiny");
    id<MTLBuffer> out = buf(64);
    memset(out.contents, 0, 64);
    const int N = 500;
    /* 1. back-to-back command buffers, one tiny dispatch each. */
    double t0 = now_s();
    id<MTLCommandBuffer> last = nil;
    for (int i = 0; i < N; i++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:out offset:0 atIndex:0];
        [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
        last = cb;
    }
    [last waitUntilCompleted];
    double t1 = now_s();
    printf("command buffers: %d x (1 tiny dispatch): %.1f us each, pipelined\n", N, (t1 - t0) / N * 1e6);
    /* 2. one command buffer per step, wait for completion each (a token step's sync). */
    t0 = now_s();
    for (int i = 0; i < N; i++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:out offset:0 atIndex:0];
        [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    t1 = now_s();
    printf("command buffer + waitUntilCompleted: %.1f us each (host sync per step)\n", (t1 - t0) / N * 1e6);
    /* 3. many encoders in one command buffer (segment boundaries without events). */
    t0 = now_s();
    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (int i = 0; i < N; i++) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBuffer:out offset:0 atIndex:0];
            [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
        t1 = now_s();
        printf("encoder boundaries in one command buffer: %.1f us each (wall), %.1f us each (GPU time)\n",
               (t1 - t0) / N * 1e6, (cb.GPUEndTime - cb.GPUStartTime) / N * 1e6);
    }
    /* 4. GPU -> CPU -> GPU through MTLSharedEvent: signal after a dispatch, CPU listener answers,
     * next dispatch waits for the answer. */
    id<MTLSharedEvent> ev = [dev newSharedEvent];
    MTLSharedEventListener *lis = [[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_queue_create("ev", DISPATCH_QUEUE_SERIAL)];
    t0 = now_s();
    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (int i = 0; i < N; i++) {
            const uint64_t gpu_v = 2 * (uint64_t)i + 1, cpu_v = gpu_v + 1;
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBuffer:out offset:0 atIndex:0];
            [e dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e endEncoding];
            [cb encodeSignalEvent:ev value:gpu_v];
            [ev notifyListener:lis atValue:gpu_v block:^(id<MTLSharedEvent> ev2, uint64_t v) { ev2.signaledValue = cpu_v; }];
            [cb encodeWaitForEvent:ev value:cpu_v];
        }
        [cb commit];
        [cb waitUntilCompleted];
        t1 = now_s();
        printf("MTLSharedEvent GPU->CPU->GPU boundary: %.1f us each (status %ld)\n", (t1 - t0) / N * 1e6, (long)cb.status);
    }
}

/* --- (e) ------------------------------------------------------------------------------ */
typedef struct { const uint16_t *p; size_t n; double bytes; int rounds; uint64_t sink; } CpuArg;
static void *cpu_stream(void *a) {
    CpuArg *c = a;
    uint64x2_t acc = vdupq_n_u64(0);
    for (int r = 0; r < c->rounds; r++)
        for (size_t i = 0; i + 8 <= c->n; i += 8) acc = veorq_u64(acc, vld1q_u64((const uint64_t *)(c->p + i)));
    c->sink = vgetq_lane_u64(acc, 0);
    return NULL;
}

static double gpu_gemv_once(id<MTLComputePipelineState> p, id<MTLBuffer> W, id<MTLBuffer> x, id<MTLBuffer> y,
                            uint32_t N, uint32_t K) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:W offset:0 atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:y offset:0 atIndex:2];
    [e setBytes:&N length:4 atIndex:3];
    [e setBytes:&K length:4 atIndex:4];
    [e dispatchThreads:MTLSizeMake((size_t)N * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb.GPUEndTime - cb.GPUStartTime;
}

static void probe_gemv(void) {
    id<MTLComputePipelineState> p = pso(@"gemv_bf16");
    const uint32_t K = 4096, N = 262144; /* 2 GiB of bf16 */
    const double bytes = (double)N * K * 2;
    id<MTLBuffer> W = buf((size_t)N * K * 2), x = buf(K * 2), y = buf(N * 4);
    uint16_t *w = W.contents;
    for (size_t i = 0; i < (size_t)N * K; i++) w[i] = 0x3f80 + (i & 7);
    uint16_t *xv = x.contents;
    for (uint32_t i = 0; i < K; i++) xv[i] = 0x3f80;
    gpu_gemv_once(p, W, x, y, N, K);
    double best = 1e9;
    for (int r = 0; r < 5; r++) { double t = gpu_gemv_once(p, W, x, y, N, K); if (t < best) best = t; }
    printf("GPU gemv bf16 2 GiB: %.1f ms  %.1f GB/s (GPU time)\n", best * 1e3, bytes / best / 1e9);
    /* CPU alone: 8 threads streaming a second 2 GiB buffer. */
    const size_t cn = (size_t)N * K;
    uint16_t *c2 = malloc(cn * 2);
    memcpy(c2, w, cn * 2);
    const int T = 8;
    pthread_t th[8];
    CpuArg args[8];
    for (int rep = 0; rep < 2; rep++) {
        double t0 = now_s();
        for (int t = 0; t < T; t++) {
            args[t] = (CpuArg){c2 + (cn / T) * t, cn / T, 0, 3, 0};
            pthread_create(&th[t], NULL, cpu_stream, &args[t]);
        }
        for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
        double dt = now_s() - t0;
        if (rep) printf("CPU stream 8 threads: %.1f GB/s\n", 3 * bytes / dt / 1e9);
    }
    /* Both at once: GPU gemv repeated while the CPU streams. */
    double t0 = now_s();
    for (int t = 0; t < T; t++) {
        args[t] = (CpuArg){c2 + (cn / T) * t, cn / T, 0, 6, 0};
        pthread_create(&th[t], NULL, cpu_stream, &args[t]);
    }
    int gpu_rounds = 0;
    double gsum = 0;
    while (now_s() - t0 < 0.8) { gsum += gpu_gemv_once(p, W, x, y, N, K); gpu_rounds++; }
    double tg = now_s() - t0;
    for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
    double tc = now_s() - t0;
    printf("concurrent: GPU %.1f GB/s over %d rounds (%.2fs), CPU %.1f GB/s (%.2fs wall) -> combined ~%.1f GB/s\n",
           gpu_rounds * bytes / tg / 1e9, gpu_rounds, tg, 6 * bytes / tc / 1e9, tc,
           gpu_rounds * bytes / tg / 1e9 + 6 * bytes / tc / 1e9);
    free(c2);
}

static void probe_visibility(void) {
    id<MTLComputePipelineState> p = pso(@"visibility");
    for (uint32_t pre = 0; pre < 3; pre++) {
        id<MTLBuffer> X = buf(4096 * 4), flag = buf(64), res = buf(64);
        memset(X.contents, 0, 4096 * 4); memset(flag.contents, 0, 64); memset(res.contents, 0, 64);
        uint32_t N = 2000;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:X offset:0 atIndex:0];
        [e setBuffer:flag offset:0 atIndex:1];
        [e setBuffer:res offset:0 atIndex:2];
        [e setBytes:&N length:4 atIndex:3];
        [e setBytes:&pre length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const uint32_t *r = res.contents;
        printf("visibility mode=%u (0 plain,1 prewarm,2 atomic-store): stale plain loads = %u, atomic loads = %u, spins = %u, status=%ld\n",
               pre, r[0], r[1], r[2], (long)cb.status);
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        NSError *err = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new];
        lib = [dev newLibraryWithSource:kSrc options:opt error:&err];
        if (!lib) { fprintf(stderr, "MSL compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        printf("device: %s  unified=%d  maxBufferLength=%.1f GiB\n", dev.name.UTF8String, dev.hasUnifiedMemory,
               dev.maxBufferLength / 1073741824.0);
        const char *what = argc > 1 ? argv[1] : "all";
        if (!strcmp(what, "coresident") || !strcmp(what, "all")) probe_coresident();
        if (!strcmp(what, "counters") || !strcmp(what, "all")) probe_counters();
        if (!strcmp(what, "live") || !strcmp(what, "all")) probe_live();
        if (!strcmp(what, "events") || !strcmp(what, "all")) probe_events();
        if (!strcmp(what, "gemv") || !strcmp(what, "all")) probe_gemv();
        if (!strcmp(what, "visibility")) probe_visibility();
    }
    return 0;
}
