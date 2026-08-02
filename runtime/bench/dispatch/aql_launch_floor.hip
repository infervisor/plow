// aql_launch_floor.hip — device side of aql_launch_floor.c (see that file for what is measured).
//
// Three trivial kernels, all with a 4-byte kernarg (a u32* sink) so the kernarg path is exercised
// exactly as a real dispatch does. Compiled with `hipcc --genco --offload-arch=gfx950` into a raw
// code object the host driver loads through hsa_code_object_reader_create_from_memory.
//
//   d_nop     : returns immediately. Isolates packet-processor + workgroup launch/teardown.
//   d_store   : one thread writes one word. Forces a real memory op so the release fence at the
//               end of the dispatch has something to write back (a truly empty kernel could let
//               the CP skip work).
//   d_spin    : burns `sink[1]` iterations of s_memrealtime, so a packet can be given a known
//               body length and the launch overhead read off as (period - body).
#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(1, 512)))
void d_nop(uint32_t* sink) { (void)sink; }

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(1, 512)))
void d_store(uint32_t* sink) {
    if (threadIdx.x == 0) sink[blockIdx.x] = blockIdx.x + 1u;
}

// --------------------------------------------------------------------------------------------
// DEVICE-SIDE AQL ENQUEUE. ROCm's device library exports the HSA queue/signal ABI to device code
// (`/opt/rocm/amdgcn/bitcode/ockl.bc`), and on gfx950 `__ockl_hsa_signal_store` on a DOORBELL-kind
// signal compiles to `buffer_wbl2 sc0 sc1; s_waitcnt vmcnt(0); global_store_dwordx2 ... sc0 sc1`
// straight at `amd_signal_t.hardware_doorbell_ptr`. So a kernel CAN dispatch its own successor.
// `d_parent` does exactly the sequence `__amd_scheduler_rocm` does, and times it:
//   t_ring  = s_memrealtime() immediately before the doorbell store  (in the PARENT)
//   t_child = s_memrealtime() at the child's first instruction       (in the CHILD)
// so t_child - t_ring is the doorbell -> first-wave-executing latency, on one shared 100 MHz clock.
extern "C" __device__ uint64_t __ockl_hsa_queue_add_write_index(const void*, uint64_t, int);
extern "C" __device__ uint64_t __ockl_hsa_queue_load_read_index(const void*, int);
extern "C" __device__ void     __ockl_hsa_signal_store(uint64_t, int64_t, int);

struct ChildArgs { uint64_t* t_child; uint32_t* flag; uint32_t* seq; };
struct ParentArgs {
    const void* queue;      // hsa_queue_t* of the CHILD queue (must not be the parent's own)
    uint64_t doorbell;      // that queue's doorbell_signal handle
    uint64_t qbase;         // that queue's base_address
    uint32_t qsize;         // power of two
    uint64_t child_kobj, child_karg;
    uint32_t* flag;         // child sets flag[i]; parent spins on it
    uint64_t* t_ring;
    uint32_t* seq;
    int n;
};

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(1, 64)))
void d_child(ChildArgs a) {
    if (threadIdx.x == 0) {
        uint64_t t = __builtin_amdgcn_s_memrealtime();
        uint32_t i = __hip_atomic_fetch_add(a.seq, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&a.t_child[i], t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __builtin_amdgcn_s_waitcnt(0x0070);
        __hip_atomic_store(&a.flag[i], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(1, 64)))
void d_parent(ParentArgs a) {
    if (threadIdx.x != 0) return;
    /* Packet stores go to the queue ring, which lives in fine-grained system memory the command
     * processor reads, so they are SYSTEM scope. __ockl_hsa_signal_store's own
     * `buffer_wbl2 sc0 sc1` then flushes them ahead of the doorbell. */
    /* EVERY spin below is bounded. This box is shared: a parent kernel that spins forever because
     * the command processor ignored a device-written packet does not just fail the measurement, it
     * wedges the GPU until someone resets it. 2 s at 100 MHz = 2e8 ticks. */
    const uint64_t DEADLINE = 200000000ull;
    for (int i = 0; i < a.n; i++) {
        uint64_t idx = __ockl_hsa_queue_add_write_index(a.queue, 1, __ATOMIC_RELAXED);
        uint64_t tw = __builtin_amdgcn_s_memrealtime();
        while (idx - __ockl_hsa_queue_load_read_index(a.queue, __ATOMIC_ACQUIRE) >= a.qsize)
            if (__builtin_amdgcn_s_memrealtime() - tw > DEADLINE) return;
        uint32_t* p = (uint32_t*)(a.qbase + (idx & (a.qsize - 1)) * 64);
        uint64_t* q = (uint64_t*)p;
#define SS32(o, v) __hip_atomic_store(&p[o], (uint32_t)(v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
#define SS64(o, v) __hip_atomic_store(&q[o], (uint64_t)(v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
        SS32(1, 1u | (1u << 16));            /* workgroup_size_x=1... overwritten below */
        SS32(1, 64u | (1u << 16));           /* wg_x=64, wg_y=1 */
        SS32(2, 1u);                         /* wg_z=1, reserved0=0 */
        SS32(3, 64u);                        /* grid_size_x */
        SS32(4, 1u); SS32(5, 1u);            /* grid_size_y, grid_size_z */
        SS32(6, 0u); SS32(7, 0u);            /* private_segment_size, group_segment_size */
        SS64(4, a.child_kobj);
        SS64(5, a.child_karg);
        SS64(6, 0ull);                       /* reserved2 */
        SS64(7, 0ull);                       /* completion_signal = none */
        a.t_ring[i] = __builtin_amdgcn_s_memrealtime();
        /* publish: 32-bit release store of header|setup, exactly as the host path does */
        uint32_t hdr = (2u /*KERNEL_DISPATCH*/ << 0) | (1u << 8 /*BARRIER*/)
                     | (1u << 9 /*ACQ agent*/) | (1u << 11 /*REL agent*/);
        __hip_atomic_store(&p[0], hdr | (1u << 16), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        __ockl_hsa_signal_store(a.doorbell, (int64_t)idx, __ATOMIC_RELEASE);
        uint64_t td = __builtin_amdgcn_s_memrealtime();
        while (__hip_atomic_load(&a.flag[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0u)
            if (__builtin_amdgcn_s_memrealtime() - td > DEADLINE) return;  /* CP never ran it */
#undef SS32
#undef SS64
    }
}

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(1, 512)))
void d_spin(uint32_t* sink) {
    uint64_t n = sink[1];
    uint64_t t0 = __builtin_amdgcn_s_memrealtime();
    while (__builtin_amdgcn_s_memrealtime() - t0 < n) { __builtin_amdgcn_s_sleep(1); }
    if (threadIdx.x == 0) sink[2 + blockIdx.x] = 1u;
}

// --------------------------------------------------------------------------------------------
// DOES THE AQL BARRIER BIT ACTUALLY ORDER MEMORY ACROSS XCDs? Speed is not soundness. These two
// kernels alternate in a barrier-bit-chained queue and use PLAIN (unmarked) accesses, so the only
// thing that can make the handoff visible is the packet header's agent-scope release fence after
// `d_pub` and agent-scope acquire fence before `d_chk`. Any stale word means the barrier bit is
// not a substitute for plow's counter protocol; zero means the command processor's per-packet
// maintenance replaces 256 workgroups' worth of it.
//   ver[0] is the version stamp, bumped by d_chk so the pair is self-clocking.
//   ver[1] accumulates stale words; ver[2] counts words checked.
/* DIMENSIONS COME FROM ctl[], NOT blockDim/gridDim. Those live in the COv5 implicit kernarg block,
 * which this bench does not fill (it writes only the two explicit pointers), so they read ZERO --
 * `i += blockDim.x` then spins forever and the queue never drains. blockIdx/threadIdx are fine;
 * they come from hardware. ctl[2]=payload words, ctl[3]=blocks, ctl[4]=threads. */
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(1, 512)))
void d_pub(const uint32_t* ctl, uint32_t* out) {
    const uint32_t pw = ctl[2], nthr = ctl[4];
    const uint32_t v = out[0];                       /* plain load; CP acquire made it visible */
    uint32_t* slice = out + 64 + (size_t)blockIdx.x * 1024;
    for (uint32_t i = threadIdx.x; i < pw; i += nthr) slice[i] = v;   /* plain stores */
}
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(1, 512)))
void d_chk(const uint32_t* ctl, uint32_t* out) {
    const uint32_t pw = ctl[2], nblk = ctl[3], nthr = ctl[4];
    const uint32_t v = out[0];
    const uint32_t peer = (blockIdx.x + nblk / 2) % nblk;   /* 128 away => a different XCD */
    const uint32_t* slice = out + 64 + (size_t)peer * 1024;
    uint32_t bad = 0;
    for (uint32_t i = threadIdx.x; i < pw; i += nthr) if (slice[i] != v) bad++;
    if (bad) atomicAdd(&out[1], bad);
}
/* The version bump gets its OWN 1-workgroup packet. Doing it inside d_chk (block 0, after its own
 * __syncthreads) races: sibling blocks read out[0] at their start, and nothing orders that against
 * block 0's write, so a late block could read v+1 and compare it against a slice holding v. That
 * flaw can only INVENT staleness, never hide it -- but a clean run is worth more than a noisy one,
 * so the cycle is pub(256) -> chk(256) -> bump(1), chained by the barrier bit. */
extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(1, 512)))
void d_bump(const uint32_t* ctl, uint32_t* out) {
    if (threadIdx.x == 0) { out[2] += ctl[2] * ctl[3]; out[0] = out[0] + 1u; }
}
