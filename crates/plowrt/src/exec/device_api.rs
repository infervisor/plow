//! `EngineDevice` — the device surface the persistent-interpreter engine needs.
//!
//! `exec::gpu` is written against `CudaBackend` and its concrete
//! `CudaStream`/`CudaEvent`/`PinnedHost`/`KernelFn` types, with zero uses of the
//! object-safe [`Backend`](crate::device::Backend) trait. That is why plowrt
//! cannot serve on AMD: the engine, not the backend, is the CUDA-shaped part.
//!
//! `Backend` cannot grow these methods, because it is deliberately object-safe
//! (`Arc<dyn Backend>` holds a heterogeneous set) and everything on it is off
//! the per-token hot path. The engine's surface is the opposite on both counts:
//! it is monomorphised per device and it *is* the hot path. So this is a second,
//! generic trait rather than an extension of the first.
//!
//! ## The kernel-argument ABI is not a problem
//!
//! The obvious blocker for a CUDA/HSA abstraction is kernel arguments: CUDA
//! takes an array of pointers-to-arguments that the driver dereferences, while
//! an AQL dispatch takes one flat kernarg block. Reconciling those in general
//! needs per-argument size information that the CUDA form does not carry.
//!
//! It does not arise here. The engine launches the interpreter with exactly ONE
//! argument, a POD struct:
//!
//! ```ignore
//! let mut params = [&mut arg as *mut DevProgram as *mut c_void];
//! ```
//!
//! and `HsaBackend::dispatch` already takes `(args: *const c_void, args_size)`,
//! mirroring `plow_hsa_launch` in `runtime/amd/hsa_backend.h` 1:1. So
//! [`EngineDevice::launch_cooperative`] takes the argument as a byte slice: the
//! CUDA impl points its one-element array at it, the HSA impl memcpy's it into
//! the kernarg ring. Nothing is lost and no reflection is needed.

use crate::device::{DeviceMem, Module};
use crate::error::Result;

/// Page-locked host staging memory.
///
/// The engine fills these slabs on the host and hands them to an async H2D
/// copy, so it needs slice access and nothing else.
pub trait PinnedBuf: Send {
    fn as_slice(&self) -> &[u8];
    fn as_mut_slice(&mut self) -> &mut [u8];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The device surface `exec::gpu` drives.
///
/// Implemented by `CudaBackend` (forwarding — the NVIDIA path must stay
/// byte-identical) and `HsaBackend`. Ordering semantics are the CUDA ones,
/// because that is what the engine was written to; see the `device::hsa`
/// module notes for how an AQL queue is mapped onto them (one ordered queue,
/// so a "stream" is a handle onto it rather than a new queue).
pub trait EngineDevice: Send + Sync + 'static {
    /// An ordered command stream. The engine uses exactly one by design:
    /// decode and prefill share mutable run state, so overlapping streams would
    /// race until every in-flight command owns separate storage.
    type Stream: Send;
    /// A point in stream order. Used for two different jobs — gating pinned
    /// buffer reuse (ordering only) and step timing (reported, not
    /// load-bearing).
    type Event: Send;
    /// Page-locked host staging.
    type Pinned: PinnedBuf;
    /// A resolved kernel entry point.
    ///
    /// `Copy` because the engine copies one out of `&mut self` before a launch
    /// precisely so the launch does not hold a borrow of the struct that owns
    /// it (`let (sf, sdp, ses) = (smp.f, ...)`). Both `KernelFn` (a `usize`) and
    /// `HsaKernel` (four PODs) are trivially `Copy`, so this costs nothing and
    /// avoids rewriting five borrow sites into something borrowck rejects.
    type Function: Copy;

    // --- identity / geometry ---------------------------------------------

    fn device_name(&self) -> &str;

    /// The ISA key that selects a code object and its kernel symbols.
    ///
    /// NOT a compute capability: `gfx950` does not decompose into a
    /// major/minor pair, and the engine's profile table keys on the string
    /// anyway (`"sm120"`, `"sm90a"`, `"gfx950"`). A `(u32, u32)` here would
    /// force the AMD path to invent a fake version pair and then map it back.
    fn arch(&self) -> String;

    /// Executor count — SMs on NVIDIA, CUs on AMD. The engine sizes its
    /// cooperative grid from this, so a wrong value is a wrong launch, not a
    /// wrong number in a log.
    fn sm_count(&self) -> u32;

    // --- memory ------------------------------------------------------------

    fn alloc(&self, bytes: u64) -> Result<DeviceMem>;
    fn upload(&self, dst: &DeviceMem, off: u64, src: &[u8]) -> Result<()>;
    fn download(&self, src: &DeviceMem, off: u64, dst: &mut [u8]) -> Result<()>;
    fn memcpy_htod(&self, dptr: u64, src: &[u8]) -> Result<()>;
    fn memcpy_dtod(&self, dst: u64, src: u64, bytes: u64) -> Result<()>;
    /// Many device-to-device copies under ONE completion wait.
    ///
    /// `memcpy_dtod` is create-signal / copy / BLOCKED-wait / destroy per call, which is the
    /// right shape for one copy and the wrong shape for hundreds: a prefix-state snapshot is 276
    /// separate tensors, so the per-copy synchronisation dominates the bytes by orders of
    /// magnitude. Backends that can issue N copies against a single signal should override this.
    /// `pairs` is `(dst, src, bytes)`.
    fn memcpy_dtod_batch(&self, pairs: &[(u64, u64, u64)]) -> Result<()> {
        for &(dst, src, bytes) in pairs {
            self.memcpy_dtod(dst, src, bytes)?;
        }
        Ok(())
    }
    fn host_alloc_pinned(&self, bytes: usize) -> Result<Self::Pinned>;

    /// Fill `n` bytes at `dptr` with `value`.
    fn memset_d8(&self, dptr: u64, value: u8, n: usize) -> Result<()>;

    /// Same, enqueued on `stream`.
    ///
    /// **The two backends differ in kind here, not in name.** `cuMemsetD8Async`
    /// is queue-ordered; HSA has no queue-ordered fill, so the AMD impl is a
    /// blocking SDMA copy from a preallocated fill buffer. This is on the
    /// per-token path (counter re-arm), so the AMD engine should prefer to
    /// batch its zeroing rather than assume this is free — see
    /// `plans/tp-design.md`'s counter-reset numbers for what a copy-engine
    /// round-trip actually costs.
    fn memset_d8_async(&self, dptr: u64, value: u8, n: usize, stream: &Self::Stream) -> Result<()>;

    /// Enqueue a host→device copy on `stream`.
    ///
    /// # Safety
    /// `src` must stay live and unmodified until the copy retires; the caller
    /// gates that with an event.
    unsafe fn memcpy_htod_async(&self, dptr: u64, src: &[u8], stream: &Self::Stream) -> Result<()>;

    /// Enqueue a device→host copy on `stream`.
    ///
    /// # Safety
    /// `dst` must stay live until the copy retires.
    unsafe fn memcpy_dtoh_async(
        &self,
        dst: &mut [u8],
        dptr: u64,
        stream: &Self::Stream,
    ) -> Result<()>;

    // --- ordering ----------------------------------------------------------

    fn stream_create(&self) -> Result<Self::Stream>;
    fn stream_synchronize(&self, stream: &Self::Stream) -> Result<()>;
    fn synchronize(&self) -> Result<()>;

    /// `timing == false` is the cheap ordering-only event used to gate buffer
    /// reuse; `event_elapsed_ms` over such a pair need not be meaningful.
    fn event_create(&self, timing: bool) -> Result<Self::Event>;
    fn event_record(&self, event: &Self::Event, stream: &Self::Stream) -> Result<()>;
    fn event_synchronize(&self, event: &Self::Event) -> Result<()>;
    fn event_elapsed_ms(&self, start: &Self::Event, end: &Self::Event) -> Result<f32>;

    // --- modules -----------------------------------------------------------

    fn module_load(&self, image: &[u8]) -> Result<Module>;
    fn module_unload(&self, module: &Module) -> Result<()>;
    fn get_function(&self, module: &Module, name: &str) -> Result<Self::Function>;

    /// Zero `n` bytes of a named module global. `false` when the symbol is
    /// absent — the engine probes for optional globals, so a miss is not an
    /// error.
    fn module_global_zero(&self, module: &Module, name: &str, n: usize) -> Result<bool>;

    /// Read a `u32` module global; `None` when the symbol is absent.
    ///
    /// The engine reads build-time facts out of the code object this way —
    /// `plow_arena_bytes` (the dynamic-LDS budget the object was compiled for)
    /// and `plow_packet_hash_lo/_hi` (which packet a specialised object was
    /// built against, a mismatch being fatal). Absence is a legitimate answer:
    /// an unspecialised object simply does not carry them.
    fn module_global_u32(&self, module: &Module, name: &str) -> Result<Option<u32>>;

    /// Read up to `max` bytes of a named module global into `out`; `false` when
    /// absent. Used for the device-side trace buffers.
    fn module_global_bytes(
        &self,
        module: &Module,
        name: &str,
        max: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool>;

    // --- launch ------------------------------------------------------------

    /// Opt a function into a dynamic-shared-memory budget above the default
    /// cap. No-op where the platform carries the size in the dispatch packet
    /// (HSA) rather than as a function attribute (CUDA).
    fn set_max_dynamic_smem(&self, f: Self::Function, bytes: u32) -> Result<()>;

    /// Resident blocks per executor at this block size and smem budget — the
    /// occupancy gate a cooperative launch must satisfy.
    ///
    /// **Only NVIDIA can answer this from the driver.**
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor` has no HSA counterpart, and
    /// the difference is not cosmetic: `cuLaunchCooperativeKernel` *fails* when
    /// the grid cannot be co-resident, which turns a would-be counter deadlock
    /// into a loud launch-time error. An AQL dispatch has no such check — it
    /// oversubscribes happily and the interpreter then deadlocks on a counter
    /// no resident workgroup will ever signal. See the AMD impl for what it
    /// returns instead and why.
    fn occupancy_blocks_per_sm(&self, f: Self::Function, block: u32, smem: usize) -> Result<u32>;

    /// Cooperative (co-resident, grid-synchronising) launch: the persistent
    /// interpreter. `args` is the packed kernel-argument block — see the module
    /// note on why one byte slice suffices.
    fn launch_cooperative(
        &self,
        f: Self::Function,
        grid: u32,
        block: u32,
        smem_bytes: u32,
        args: &[u8],
        stream: Option<&Self::Stream>,
    ) -> Result<()>;

    /// Ordinary launch, for the standalone device-sampler epilogue: a plain
    /// grid run after the interpreter on the same ordered stream, so it needs
    /// neither co-residency nor the occupancy gate.
    ///
    /// `args` is a packed kernel-argument block, same as
    /// [`EngineDevice::launch_cooperative`]. NOTE that the CUDA engine's two
    /// epilogues (`plow_sample`, `plow_advance`) are launched today with 10-
    /// and 8-element `kernelParams` arrays of mixed `u64`/`u32`, NOT one POD
    /// struct — so routing them through this method means packing the block on
    /// the engine side, where the argument types are known, rather than
    /// reflecting over them here. The AMD path has no sampler object yet, so
    /// nothing packs one today.
    fn launch_kernel(
        &self,
        f: Self::Function,
        grid: u32,
        block: u32,
        smem_bytes: u32,
        args: &[u8],
        stream: Option<&Self::Stream>,
    ) -> Result<()>;
}
