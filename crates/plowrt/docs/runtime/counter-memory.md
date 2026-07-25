# Counter memory placement (host ↔ SM/CU)

> Are the counters memory-mapped and in unified memory, since the GPU SM/CU
> access them?

**Short answer:** the counter cells are placed in a **memory-mapped region that
both the host and the device address**, but the *kind* of mapping is chosen per
counter **scope**, and it is **not** CUDA managed/unified memory
(`cuMemAllocManaged`) — that would thrash under hot atomic traffic. Implemented
in `src/exec/counters.rs`.

## Placement by scope

| Scope | Storage | Atomic scope | Host access |
|-------|---------|--------------|-------------|
| `IntraSm` | SM shared memory | block-scope | none (device-only) |
| `IntraGpu` | device-global HBM (`cuMemAlloc`) | device-scope | read milestones via mapping/DtoH |
| `CrossUnit` / cross-device | **host-pinned, device-mapped** (`cuMemHostAlloc` + `DEVICEMAP` → `cuMemHostGetDevicePointer`) | **system-scope** | direct — same cells |

The cross-device tier is the memory-mapped, coherent region both sides share: a
single allocation exposes a **host pointer** *and* a **device pointer**. The SM/CU
does system-scope atomics through the device pointer; the host coordinator and
the counter-space monitor poll through the host pointer — same physical cells,
no copy.

## Why not unified (managed) memory

`cuMemAllocManaged` gives one pointer valid on both sides, but the pages migrate
on demand. Counters are tiny, contended, and updated constantly; migration would
ping-pong pages between host and device and destroy atomic throughput. So we use
**pinned + device-mapped** memory (zero-copy, no migration) for the counters the
host must see, and plain **device-global** memory for intra-GPU counters the host
only samples occasionally.

## How the Rust side maps it

`CounterPool` holds a raw `*const AtomicU64` base into the mapped region plus a
64-byte stride (one cache line per counter, matching the device
`struct counter { u64 value; u8 pad[56]; }` ABI). Two constructors:

- `CounterPool::from_counters` — host box backing (CPU backend, and the model for
  host-pinned cross-device counters).
- `unsafe CounterPool::over_mapped(dev_mem, counters)` — builds over a
  backend-allocated device-mapped `DeviceMem` (its `base` is the host pointer;
  the same region's device pointer is handed to the kernel).

`Backend::alloc_counter_region(n)` allocates the region; GPU backends override it
with `cuMemHostAlloc(DEVICEMAP)` / `hipHostMalloc(Mapped)`, the default is host
memory (CPU backend). Hot ops (`load`/`add`) index `base + id*64` and do a single
atomic with `Acquire`/`Release` ordering — no branch on scope, no allocation.
