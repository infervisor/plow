/* interp_sm120_poc.cu — proof-of-concept: the plow persistent on-device
 * interpreter, ported to NVIDIA consumer Blackwell (GB202, sm_120a).
 *
 * WHAT THIS PROVES (milestone M1, launch-model half):
 *   1. A single COOPERATIVE persistent grid (grid sized to occupancy, so every
 *      block is co-resident) can stay resident and walk a per-block packet
 *      stream — the CUDA analogue of interp.hip's "grid == CU count" model.
 *   2. The counter-gated DAG works: instruction 1 waits on two counters that
 *      designated threads in every instruction-0 block bump, so a cross-block producer->consumer
 *      dependency resolves with NO grid-wide barrier primitive (no grid.sync()).
 *   3. The relaxed-poll + __threadfence() coherence recipe is sound on sm_120:
 *      consumer block C reads bytes written by a DIFFERENT producer block P.
 *
 * It reuses the REAL device ABI verbatim (runtime/common/dev_isa.h:
 * PlowDevInst / PlowStreamEnt / PlowProgram / PlowWait / counters), so the
 * host-side stream builder here is the same shape the emitter will produce.
 * The two op bodies are deliberately trivial PoC ops (add-1, mul-2), NOT the
 * real gemma_sm120 kernels — wiring those in as __device__ op bodies is the
 * SECOND half of M1. This file is about the resident-launch machinery only.
 *
 * Build (on the sm_120 box):
 *   nvcc -arch=sm_120a -I ../common interp_sm120_poc.cu -o /tmp/interp_poc
 * Run:
 *   /tmp/interp_poc
 */
/* dev_isa.h has two raw C11 `_Static_assert`s (lines 672/675) that bypass its
 * own __cplusplus PLOW_SASSERT macro. clang/hipcc accepts them as an extension;
 * nvcc's C++ front-end does not. Map to C++ static_assert without touching the
 * ABI-locked shared header. */
#ifdef __cplusplus
#define _Static_assert static_assert
#endif
#include "dev_isa.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// PoC opcodes — local, not real PLOW_DOP_* bodies. Chosen out of the way of the
// device ISA enum so the switch reads like the real interpreter's.
enum { POC_OP_ADD1 = 1000, POC_OP_MUL2 = 1001 };

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Device: counter protocol (mirrors interp.hip ctr_poll / ctr_acquire /
// ctr_signal, retargeted to the NVIDIA memory model).
// ---------------------------------------------------------------------------

// Relaxed poll: `volatile` forces each read past L1 to L2 (the single coherence
// point on a GB202 — one L2, unlike CDNA's per-XCD L2, so device scope suffices).
__device__ __forceinline__ uint32_t ctr_poll(const uint32_t* p) {
    return *(const volatile uint32_t*)p;
}

// Release before signal / acquire after the gate — one device-scope fence,
// exactly where interp.hip puts its agent-scope buffer_wbl2 / buffer_inv.
__device__ __forceinline__ void ctr_signal(uint32_t* p) {
    atomicAdd(p, 1u);
}

// ---------------------------------------------------------------------------
// Device: the two PoC op bodies. Each is sliced by the packet's (slice, blocks)
// exactly like a real op would be — grid-stride over n by (slice, blocks),
// NOT by (blockIdx, gridDim): inside a megakernel the block is the "CU" and its
// share is carried in the stream entry, because 188 blocks are not at one PC.
// ---------------------------------------------------------------------------
__device__ void poc_exec(const PlowDevInst* in, unsigned slice,
                         void* const* tensors) {
    const unsigned n = in->i[0];
    const unsigned blocks = in->blocks;
    const unsigned base = slice * blockDim.x + threadIdx.x;
    const unsigned step = blocks * blockDim.x;

    switch (in->op) {
    case POC_OP_ADD1: {
        float* out = (float*)tensors[in->t[0]];
        const float* x = (const float*)tensors[in->t[1]];
        for (unsigned i = base; i < n; i += step) out[i] = x[i] + 1.0f;
        break;
    }
    case POC_OP_MUL2: {
        float* out = (float*)tensors[in->t[0]];
        const float* x = (const float*)tensors[in->t[1]]; // == inst0's output
        for (unsigned i = base; i < n; i += step) out[i] = x[i] * 2.0f;
        break;
    }
    default:
        break;
    }
}

// ---------------------------------------------------------------------------
// The persistent interpreter kernel. One launch, grid == co-resident capacity.
// Structure mirrors interp.hip:926-1096 (coarse path only; no fine/xctr/trace).
// ---------------------------------------------------------------------------
__global__ void interp_sm120(PlowProgram prog) {
    const unsigned cu = blockIdx.x;
    const unsigned n = prog.stream_len[cu];
    const PlowStreamEnt* my = prog.stream + prog.stream_ofs[cu];

    for (unsigned ix = 0; ix < n; ix++) {
        const PlowStreamEnt e = my[ix];
        const PlowDevInst* in = prog.insts + e.inst;

        // Coarse gates live on the instruction (PoC never sets PLOW_SE_FINE).
        const unsigned wait_len = in->wait_len;
        const unsigned wait_ofs = in->wait_ofs;
        const unsigned succ_len = in->succ_len;
        const unsigned succ_ofs = in->succ_ofs;

        // Gate: one thread per counter, polled concurrently (interp.hip:978).
        for (unsigned w = threadIdx.x; w < wait_len; w += blockDim.x) {
            const PlowWait pw = prog.waits[wait_ofs + w];
            while (ctr_poll(PLOW_CTR(prog.counters, pw.id)) < pw.threshold)
                __nanosleep(64);
        }
        __syncthreads(); // every counter in the list is now satisfied

        // ONE acquire for the whole block, only after the gate clears: makes the
        // producers' released writes visible before any thread reads an operand.
        if (threadIdx.x == 0 && wait_len) __threadfence();
        __syncthreads();

        poc_exec(in, e.slice, prog.tensors);

        __syncthreads(); // retire this block's stores before the release

        // Publish, then assign each successor counter to a designated writer.
        if (succ_len) {
            if (threadIdx.x == 0) __threadfence(); // release stores before relaxed bumps
            __syncthreads();
        }
        for (unsigned s = threadIdx.x; s < succ_len; s += blockDim.x)
            ctr_signal(PLOW_CTR(prog.counters, prog.succs[succ_ofs + s]));
    }
}

// ---------------------------------------------------------------------------
// Host: build a 2-instruction program with a cross-block dependency and verify.
//   inst0 (ADD1): out0[i] = x[i] + 1        sliced across all G blocks; succ=ctr0
//   inst1 (MUL2): out1[i] = out0[i] * 2     waits ctr0>=G;             succ=ctr1
// Correct result requires inst1's block to read out0 bytes written by OTHER
// blocks in inst0 — i.e. the fence/counter protocol must actually be coherent.
// ---------------------------------------------------------------------------
int main() {
    int dev = 0;
    CUDA_OK(cudaSetDevice(dev));
    cudaDeviceProp p;
    CUDA_OK(cudaGetDeviceProperties(&p, dev));

    int coop = 0;
    CUDA_OK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
    printf("device: %s  cc %d.%d  SMs=%d  coopLaunch=%d\n", p.name, p.major,
           p.minor, p.multiProcessorCount, coop);
    if (!coop) {
        fprintf(stderr, "cooperative launch unsupported — cannot host a "
                        "resident interpreter this way\n");
        return 2;
    }

    const int TPB = 256;
    // Grid = co-resident capacity (occupancy x SMs). Larger would deadlock the
    // spin (interp.hip's "grid == CU count" invariant); cooperative launch also
    // refuses to launch a grid that would not fit, turning the AMD silent-
    // deadlock risk into a launch-time error.
    int blocksPerSM = 0;
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, interp_sm120, TPB, 0));
    const int G = blocksPerSM * p.multiProcessorCount;
    printf("blocksPerSM=%d  grid=%d  threads/block=%d\n", blocksPerSM, G, TPB);

    const unsigned N = 4u * 1024u * 1024u; // 4M floats

    // Device tensors: 0=x, 1=out0 (inst0 out / inst1 in), 2=out1.
    float *d_x, *d_out0, *d_out1;
    CUDA_OK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_out0, N * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_out1, N * sizeof(float)));
    float* h_x = (float*)malloc(N * sizeof(float));
    for (unsigned i = 0; i < N; i++) h_x[i] = (float)(i % 97);
    CUDA_OK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    void* h_tensors[3] = {d_x, d_out0, d_out1};
    void** d_tensors;
    CUDA_OK(cudaMalloc(&d_tensors, sizeof(h_tensors)));
    CUDA_OK(cudaMemcpy(d_tensors, h_tensors, sizeof(h_tensors),
                       cudaMemcpyHostToDevice));

    // Instructions.
    PlowDevInst insts[2];
    memset(insts, 0, sizeof(insts));
    insts[0].op = POC_OP_ADD1;
    insts[0].blocks = (uint16_t)G;
    insts[0].t[0] = 1; insts[0].t[1] = 0; // out0 = x + 1
    insts[0].i[0] = N;
    insts[0].wait_len = 0; insts[0].wait_ofs = 0;
    insts[0].succ_len = 2; insts[0].succ_ofs = 0; // -> ctr 0, ctr 2

    insts[1].op = POC_OP_MUL2;
    insts[1].blocks = (uint16_t)G;
    insts[1].t[0] = 2; insts[1].t[1] = 1; // out1 = out0 * 2
    insts[1].i[0] = N;
    insts[1].wait_len = 2; insts[1].wait_ofs = 0; // wait[0], wait[1]
    insts[1].succ_len = 1; insts[1].succ_ofs = 2; // -> ctr 1

    PlowWait waits[2] = {{/*id*/ 0, /*threshold*/ (uint32_t)G},
                         {/*id*/ 2, /*threshold*/ (uint32_t)G}};
    uint32_t succs[3] = {0u, 2u, 1u};

    // Per-block streams: every block runs both instructions, slice = blockIdx.
    const int NENT = 2 * G;
    PlowStreamEnt* h_stream = (PlowStreamEnt*)calloc(NENT, sizeof(PlowStreamEnt));
    uint32_t* h_ofs = (uint32_t*)malloc(G * sizeof(uint32_t));
    uint32_t* h_len = (uint32_t*)malloc(G * sizeof(uint32_t));
    for (int b = 0; b < G; b++) {
        h_ofs[b] = (uint32_t)(2 * b);
        h_len[b] = 2;
        h_stream[2 * b + 0].inst = 0; h_stream[2 * b + 0].slice = (uint32_t)b;
        h_stream[2 * b + 1].inst = 1; h_stream[2 * b + 1].slice = (uint32_t)b;
    }

    // Upload the program tables.
    PlowDevInst* d_insts; PlowStreamEnt* d_stream;
    uint32_t *d_ofs, *d_len, *d_succs, *d_counters;
    PlowWait* d_waits;
    const int NCTR = 3; // ctr0..ctr2 — each PLOW_CTR_STRIDE apart
    CUDA_OK(cudaMalloc(&d_insts, sizeof(insts)));
    CUDA_OK(cudaMalloc(&d_stream, NENT * sizeof(PlowStreamEnt)));
    CUDA_OK(cudaMalloc(&d_ofs, G * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_len, G * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_waits, sizeof(waits)));
    CUDA_OK(cudaMalloc(&d_succs, sizeof(succs)));
    CUDA_OK(cudaMalloc(&d_counters, NCTR * PLOW_CTR_STRIDE * sizeof(uint32_t)));
    CUDA_OK(cudaMemcpy(d_insts, insts, sizeof(insts), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_stream, h_stream, NENT * sizeof(PlowStreamEnt),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ofs, h_ofs, G * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_len, h_len, G * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_waits, waits, sizeof(waits), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_succs, succs, sizeof(succs), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_counters, 0,
                       NCTR * PLOW_CTR_STRIDE * sizeof(uint32_t))); // zeroed per run

    PlowProgram prog;
    memset(&prog, 0, sizeof(prog));
    prog.insts = d_insts;
    prog.stream = d_stream;
    prog.stream_ofs = d_ofs;
    prog.stream_len = d_len;
    prog.waits = d_waits;
    prog.succs = d_succs;
    prog.counters = d_counters;
    prog.tensors = d_tensors;
    prog.trace = nullptr;
    prog.cur_seg = 0;

    // Cooperative launch — the resident interpreter.
    void* args[] = {&prog};
    cudaEvent_t t0, t1;
    CUDA_OK(cudaEventCreate(&t0));
    CUDA_OK(cudaEventCreate(&t1));
    CUDA_OK(cudaEventRecord(t0));
    CUDA_OK(cudaLaunchCooperativeKernel((void*)interp_sm120, dim3(G), dim3(TPB),
                                        args, 0, 0));
    CUDA_OK(cudaEventRecord(t1));
    CUDA_OK(cudaDeviceSynchronize());
    float ms = 0;
    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));

    // Verify out1 == (x + 1) * 2 for every element.
    float* h_out1 = (float*)malloc(N * sizeof(float));
    CUDA_OK(cudaMemcpy(h_out1, d_out1, N * sizeof(float), cudaMemcpyDeviceToHost));
    unsigned bad = 0;
    double maxerr = 0;
    for (unsigned i = 0; i < N; i++) {
        float want = (h_x[i] + 1.0f) * 2.0f;
        double err = fabs((double)h_out1[i] - want);
        if (err > 1e-4) { bad++; if (err > maxerr) maxerr = err; }
    }
    printf("verify: N=%u  mismatches=%u  maxerr=%g  kernel=%.3f ms\n", N, bad,
           maxerr, ms);
    if (bad == 0)
        printf("RESULT: PASS — resident 2-op DAG, cross-block dependency, "
               "coherent on sm_120\n");
    else
        printf("RESULT: FAIL\n");

    return bad == 0 ? 0 : 1;
}
