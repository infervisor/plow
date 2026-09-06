//! CUDA driver-API backend (feature `cuda`).
//!
//! ## Static binary + no link-time GPU dependency
//!
//! The driver library is **`dlopen`ed at runtime** (via `libloading`), not
//! linked. So:
//! * the binary carries **no link-time `-lcuda`** — everything else can be
//!   statically linked (musl for a fully static CPU-only build; glibc when GPU
//!   features are on, since `dlopen` needs the dynamic loader), and
//! * NVIDIA and AMD backends **coexist in one process** — `libcuda`'s `cu*`
//!   symbols and `libhsa-runtime64`'s `hsa_*` symbols share no names, so a
//!   heterogeneous NV+AMD instance just `dlopen`s both.
//!
//! ## What runs through here
//!
//! The trait surface ([`Backend`]) covers alloc/upload/download/module-load —
//! the vendor-neutral seam. The persistent sm_120 interpreter additionally
//! needs the cooperative-launch path the standalone harnesses proved out
//! (`runtime/tests/gemma4_sm120_chat.cu`): `cuModuleLoadData` on a prebuilt
//! cubin, `cuOccupancyMaxActiveBlocksPerMultiprocessor` × SM count for the
//! grid, `cuLaunchCooperativeKernel` with the `PlowProgram` kernarg.
//! Those live on [`CudaBackend`]'s inherent methods, consumed by
//! `crate::exec::gpu`.
//!
//! ## Contexts are per-thread
//!
//! The CUDA driver binds a context to the calling OS thread. plowrt's callers
//! run on tokio blocking-pool threads (a different one each tick), so **every
//! public entry point re-binds the primary context** with `cuCtxSetCurrent` —
//! a few tens of nanoseconds, nothing against a multi-millisecond token step.

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::device::{Backend, DeviceMem, ExecutorClass, ExecutorTarget, LaunchCfg, Module};
use crate::{DeviceErrorInfo, Result, RuntimeError};

pub(crate) mod lt;
pub(crate) mod qwen_gdn;

// Driver ABI types (bindgen-equivalent, transcribed from cuda.h).
type CUresult = i32;
type CUdevice = i32;
type CUdeviceptr = u64;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;

/// `CUDA_ERROR_NOT_READY` — the query result meaning "still running".
const ERROR_NOT_READY: CUresult = 600;

/// Does this `CUresult` permanently poison the context? These are the codes
/// the driver documents as leaving the context unusable (a trapped kernel, a
/// dead context, uncorrectable ECC) — after one, every further call on the
/// context fails and only process/context teardown recovers. Deliberately NOT
/// here: 2 `OUT_OF_MEMORY` and 701 `LAUNCH_OUT_OF_RESOURCES` (transient,
/// retryable) and 720 `COOPERATIVE_LAUNCH_TOO_LARGE` (a launch-shape
/// rejection, nothing ran).
fn is_cuda_fatal(rc: CUresult) -> bool {
    matches!(
        rc,
        4        // CUDA_ERROR_DEINITIALIZED — driver shutting down under us
        | 214    // CUDA_ERROR_ECC_UNCORRECTABLE
        | 700    // CUDA_ERROR_ILLEGAL_ADDRESS
        | 702    // CUDA_ERROR_LAUNCH_TIMEOUT
        | 709    // CUDA_ERROR_CONTEXT_IS_DESTROYED
        | 710    // CUDA_ERROR_ASSERT — device-side assert fired
        | 714    // CUDA_ERROR_HARDWARE_STACK_ERROR
        | 715    // CUDA_ERROR_ILLEGAL_INSTRUCTION
        | 716    // CUDA_ERROR_MISALIGNED_ADDRESS
        | 717    // CUDA_ERROR_INVALID_ADDRESS_SPACE
        | 718    // CUDA_ERROR_INVALID_PC
        | 719 // CUDA_ERROR_LAUNCH_FAILED
    )
}
/// `CU_STREAM_NON_BLOCKING`: the engine stream never implicitly synchronizes
/// with the legacy default stream (module loads, weight uploads) — every
/// cross-stream ordering the engine needs is a host-side stream synchronize.
const STREAM_NON_BLOCKING: u32 = 1;
/// `CU_EVENT_DEFAULT` (timing-capable) / `CU_EVENT_DISABLE_TIMING` (sync-only,
/// the cheaper record path).
const EVENT_DEFAULT: u32 = 0;
const EVENT_DISABLE_TIMING: u32 = 2;

// CUdevice_attribute values used here (cuda.h).
const ATTR_WARP_SIZE: i32 = 10;
const ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
const ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;
const ATTR_COOPERATIVE_LAUNCH: i32 = 95;
const ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: i32 = 97;
/// `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`: 1 only
/// on hardware-coherent platforms (Grace-Hopper ATS), where the DMA engines
/// read pageable host memory — an mmap'd checkpoint included — at full link
/// speed with no staging. Deliberately NOT the weaker attribute 88
/// (`PAGEABLE_MEMORY_ACCESS`), which HMM also sets on PCIe boxes where the
/// same copy degrades to fault-driven migration.
const ATTR_PAGEABLE_USES_HOST_PAGE_TABLES: i32 = 100;
// CUfunction_attribute.
const FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
// cuMemHostAlloc flags.
const MEMHOSTALLOC_PORTABLE: u32 = 1;
// VMM enums (cuda.h): allocation type / location / access flags / granularity.
const MEM_ALLOCATION_TYPE_PINNED: i32 = 1;
const MEM_LOCATION_TYPE_DEVICE: i32 = 1;
const MEM_ACCESS_FLAGS_PROT_READWRITE: i32 = 3;
const MEM_ALLOC_GRANULARITY_RECOMMENDED: i32 = 1;

/// `CUmemLocation` (cuda.h) — `{ CUmemLocationType type; int id; }`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CUmemLocation {
    type_: i32,
    id: i32,
}

/// `CUmemAllocationProp` (cuda.h). The allocFlags tail is
/// `{ u8 compressionType; u8 gpuDirectRDMACapable; u16 usage; u8 reserved[4] }`
/// — transcribed as raw bytes (all zero here).
#[repr(C)]
struct CUmemAllocationProp {
    type_: i32,
    requested_handle_types: i32,
    location: CUmemLocation,
    win32_handle_meta_data: *mut c_void,
    alloc_flags: [u8; 8],
}

/// `CUmemAccessDesc` (cuda.h) — `{ CUmemLocation location; CUmemAccess_flags flags; }`.
#[repr(C)]
struct CUmemAccessDesc {
    location: CUmemLocation,
    flags: i32,
}

macro_rules! driver_api {
    ($( $name:ident : fn( $($arg:ty),* ) -> CUresult ),+ $(,)?) => {
        /// The resolved driver entry points. Raw fn pointers copied out of the
        /// mapped library once at `open()`; `_lib` keeps the mapping alive.
        #[allow(non_snake_case)]
        struct Api {
            _lib: libloading::Library,
            $( $name: unsafe extern "C" fn($($arg),*) -> CUresult, )+
        }

        impl Api {
            #[allow(non_snake_case)]
            fn resolve(lib: libloading::Library) -> Result<Api> {
                $(
                    let $name = *unsafe {
                        lib.get::<unsafe extern "C" fn($($arg),*) -> CUresult>(
                            concat!(stringify!($name), "\0").as_bytes(),
                        )
                    }
                    .map_err(|e| RuntimeError::Device(format!(
                        "resolve {}: {e}", stringify!($name)
                    )))?;
                )+
                Ok(Api { _lib: lib, $( $name, )+ })
            }
        }
    };
}

driver_api! {
    cuInit: fn(u32) -> CUresult,
    cuDriverGetVersion: fn(*mut i32) -> CUresult,
    cuGetErrorName: fn(CUresult, *mut *const c_char) -> CUresult,
    cuDeviceGet: fn(*mut CUdevice, i32) -> CUresult,
    cuDeviceGetName: fn(*mut c_char, i32, CUdevice) -> CUresult,
    cuDeviceGetAttribute: fn(*mut i32, i32, CUdevice) -> CUresult,
    cuDevicePrimaryCtxRetain: fn(*mut CUcontext, CUdevice) -> CUresult,
    cuDevicePrimaryCtxRelease_v2: fn(CUdevice) -> CUresult,
    cuCtxSetCurrent: fn(CUcontext) -> CUresult,
    cuCtxSynchronize: fn() -> CUresult,
    cuModuleLoadData: fn(*mut CUmodule, *const c_void) -> CUresult,
    cuModuleUnload: fn(CUmodule) -> CUresult,
    cuModuleGetFunction: fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    cuModuleGetGlobal_v2: fn(*mut CUdeviceptr, *mut usize, CUmodule, *const c_char) -> CUresult,
    cuFuncSetAttribute: fn(CUfunction, i32, i32) -> CUresult,
    cuOccupancyMaxActiveBlocksPerMultiprocessor:
        fn(*mut i32, CUfunction, i32, usize) -> CUresult,
    cuMemAlloc_v2: fn(*mut CUdeviceptr, usize) -> CUresult,
    cuMemFree_v2: fn(CUdeviceptr) -> CUresult,
    cuMemGetInfo_v2: fn(*mut usize, *mut usize) -> CUresult,
    cuMemcpyHtoD_v2: fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    cuMemcpyDtoH_v2: fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cuMemsetD8_v2: fn(CUdeviceptr, u8, usize) -> CUresult,
    cuMemHostAlloc: fn(*mut *mut c_void, usize, u32) -> CUresult,
    cuMemFreeHost: fn(*mut c_void) -> CUresult,
    cuLaunchCooperativeKernel:
        fn(CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream, *mut *mut c_void) -> CUresult,
    cuLaunchKernel:
        fn(CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream, *mut *mut c_void, *mut *mut c_void) -> CUresult,
    // VMM surface (probed by
    // runtime/nvidia/experiments/vmm_probe.cu on this driver line). Present
    // since CUDA 10.2 — safe to resolve unconditionally.
    cuMemGetAllocationGranularity:
        fn(*mut usize, *const CUmemAllocationProp, i32) -> CUresult,
    cuMemAddressReserve: fn(*mut CUdeviceptr, usize, usize, CUdeviceptr, u64) -> CUresult,
    cuMemAddressFree: fn(CUdeviceptr, usize) -> CUresult,
    cuMemCreate: fn(*mut u64, usize, *const CUmemAllocationProp, u64) -> CUresult,
    cuMemRelease: fn(u64) -> CUresult,
    cuMemMap: fn(CUdeviceptr, usize, usize, u64, u64) -> CUresult,
    cuMemUnmap: fn(CUdeviceptr, usize) -> CUresult,
    cuMemSetAccess: fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize) -> CUresult,
    // D2D copy: VMM sliding-window restore + the prefix-attach copy baseline.
    cuMemcpyDtoD_v2: fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,
    // Async submission surface: one ordered stream per engine, async copies/memsets,
    // events. All present since CUDA 4.0 — safe to resolve unconditionally.
    cuStreamCreate: fn(*mut CUstream, u32) -> CUresult,
    cuStreamDestroy_v2: fn(CUstream) -> CUresult,
    cuStreamSynchronize: fn(CUstream) -> CUresult,
    cuMemcpyHtoDAsync_v2: fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    cuMemcpyDtoHAsync_v2: fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
    cuMemsetD8Async: fn(CUdeviceptr, u8, usize, CUstream) -> CUresult,
    cuEventCreate: fn(*mut CUevent, u32) -> CUresult,
    cuEventDestroy_v2: fn(CUevent) -> CUresult,
    cuEventRecord: fn(CUevent, CUstream) -> CUresult,
    cuEventQuery: fn(CUevent) -> CUresult,
    cuEventSynchronize: fn(CUevent) -> CUresult,
    cuEventElapsedTime: fn(*mut f32, CUevent, CUevent) -> CUresult,
    // T35: CUDA graphs — batch the ~480 per-chunk segment launches into ONE submit.
    // All present since CUDA 10; kernel nodes take the same CUDA_KERNEL_NODE_PARAMS
    // as cuLaunchKernel.
    cuStreamBeginCapture: fn(CUstream, i32) -> CUresult,
    cuStreamEndCapture: fn(CUstream, *mut CUgraph) -> CUresult,
    cuGraphCreate: fn(*mut CUgraph, u32) -> CUresult,
    cuGraphAddKernelNode_v2: fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CudaKernelNodeParams) -> CUresult,
    cuGraphInstantiateWithFlags: fn(*mut CUgraphExec, CUgraph, u64) -> CUresult,
    cuGraphLaunch: fn(CUgraphExec, CUstream) -> CUresult,
    cuGraphDestroy: fn(CUgraph) -> CUresult,
    cuGraphExecDestroy: fn(CUgraphExec) -> CUresult,
}

type CUgraph = *mut c_void;
type CUgraphNode = *mut c_void;
type CUgraphExec = *mut c_void;

/// The driver's name for `rc` (`cuGetErrorName`), or `"CUresult {rc}"` when
/// the driver doesn't know the code.
fn cu_error_name(api: &Api, rc: CUresult) -> String {
    let mut p: *const c_char = std::ptr::null();
    // SAFETY: driver call querying an error string.
    unsafe {
        if (api.cuGetErrorName)(rc, &mut p) == 0 && !p.is_null() {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        } else {
            format!("CUresult {rc}")
        }
    }
}

/// Build the typed fault for a failed driver call (no poisoning — that is
/// [`CudaBackend::check`]'s job, and bring-up in `new()` has no backend yet).
fn cu_fault(api: &Api, rc: CUresult, what: &str) -> RuntimeError {
    RuntimeError::DeviceFault {
        info: DeviceErrorInfo {
            operation: what.to_string(),
            code: rc,
            name: cu_error_name(api, rc),
            fatal: is_cuda_fatal(rc),
        },
    }
}

/// `CUDA_KERNEL_NODE_PARAMS_v2` (cuda.h). `kern` + grid/block dims + smem + params;
/// `extra`/`kernel_ctx`/`func` layout per the v2 struct.
#[repr(C)]
pub struct CudaKernelNodeParams {
    func: *mut c_void,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared_mem_bytes: u32,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
    kern: *mut c_void,
    ctx: *mut c_void,
}

/// An instantiated segment-chain graph (T35). Freed on drop.
pub struct GraphExec {
    exec: CUgraphExec,
    api: *const c_void, // not used for drop safety; freed via CudaBackend::graph_destroy
}
unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

/// A loaded-kernel handle (a `CUfunction`, valid for the module's lifetime).
#[derive(Clone, Copy)]
pub struct KernelFn(usize);

/// The freer every owned [`DeviceMem`] carries ([`crate::device::DeviceFree`]).
/// It holds the primary-context **retain**: the context is released only when
/// the backend AND every owned allocation are gone, so a late `DeviceMem` drop
/// can never call `cuMemFree` against a destroyed context. `api` keeps the
/// dlopen'd library mapped for the same reason.
struct CudaFreer {
    api: Arc<Api>,
    ctx: usize,
    dev: CUdevice,
    /// The backend's poison latch, shared so late frees on a dead context
    /// (which all fail with the same sticky status) report at debug, not one
    /// warn per allocation — a poisoned engine teardown frees hundreds.
    poisoned: Arc<std::sync::OnceLock<DeviceErrorInfo>>,
}

// SAFETY: as CudaBackend — driver calls are thread-safe, handles are
// process-global driver objects.
unsafe impl Send for CudaFreer {}
unsafe impl Sync for CudaFreer {}

impl crate::device::DeviceFree for CudaFreer {
    fn free(&self, base: u64, len: u64) {
        // SAFETY: rebind the retained context (threads vary); `base` came from
        // cuMemAlloc and DeviceMem's ownership rule frees it exactly once.
        let rc = unsafe {
            (self.api.cuCtxSetCurrent)(self.ctx as CUcontext);
            (self.api.cuMemFree_v2)(base)
        };
        if rc != 0 {
            if self.poisoned.get().is_some() {
                tracing::debug!(rc, base, len, "cuMemFree failed (context poisoned)");
            } else {
                tracing::warn!(rc, base, len, "cuMemFree failed");
            }
        }
    }
}

impl Drop for CudaFreer {
    fn drop(&mut self) {
        // SAFETY: pairs the single cuDevicePrimaryCtxRetain in `new()`.
        let rc = unsafe { (self.api.cuDevicePrimaryCtxRelease_v2)(self.dev) };
        if rc != 0 {
            if self.poisoned.get().is_some() {
                tracing::debug!(rc, "cuDevicePrimaryCtxRelease (context poisoned)");
            } else {
                tracing::warn!(rc, "cuDevicePrimaryCtxRelease");
            }
        }
    }
}

/// An owned CUDA stream — the engine's single ordered device queue. Holds the
/// primary-context retain (via `CudaFreer`) so a late drop never destroys a
/// stream against a dead context. Created `CU_STREAM_NON_BLOCKING`: work here
/// never implicitly orders against the legacy default stream.
pub struct CudaStream {
    raw: usize,
    keep: Arc<CudaFreer>,
}

// SAFETY: driver stream handles are process-global; every use re-binds the
// retained context (threads vary).
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // SAFETY: handle from cuStreamCreate, destroyed exactly once.
        let rc = unsafe {
            let _ = (self.keep.api.cuCtxSetCurrent)(self.keep.ctx as CUcontext);
            (self.keep.api.cuStreamDestroy_v2)(self.raw as CUstream)
        };
        if rc != 0 {
            if self.keep.poisoned.get().is_some() {
                tracing::debug!(rc, "cuStreamDestroy failed (context poisoned)");
            } else {
                tracing::warn!(rc, "cuStreamDestroy failed");
            }
        }
    }
}

/// An owned CUDA event (same keepalive rule as [`CudaStream`]). Timing-capable
/// events support [`CudaBackend::event_elapsed_ms`]; sync-only events record
/// cheaper.
pub struct CudaEvent {
    raw: usize,
    keep: Arc<CudaFreer>,
}

// SAFETY: as CudaStream.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        // SAFETY: handle from cuEventCreate, destroyed exactly once.
        let rc = unsafe {
            let _ = (self.keep.api.cuCtxSetCurrent)(self.keep.ctx as CUcontext);
            (self.keep.api.cuEventDestroy_v2)(self.raw as CUevent)
        };
        if rc != 0 {
            if self.keep.poisoned.get().is_some() {
                tracing::debug!(rc, "cuEventDestroy failed (context poisoned)");
            } else {
                tracing::warn!(rc, "cuEventDestroy failed");
            }
        }
    }
}

/// An owned pinned (page-locked) host allocation. True async H2D/D2H requires
/// pinned memory — pageable sources degrade to a staged, partially blocking
/// copy. Freed on drop (pinned pages are a scarce OS resource); holds the
/// primary-context retain like every other owned handle.
pub struct PinnedHost {
    ptr: *mut u8,
    len: usize,
    keep: Arc<CudaFreer>,
}

// SAFETY: pinned host memory is ordinary process memory; the driver handle is
// process-global and freed through the retained context.
unsafe impl Send for PinnedHost {}
unsafe impl Sync for PinnedHost {}

impl PinnedHost {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is a live allocation of `len` bytes until Drop.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, with `&mut self` guaranteeing uniqueness.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        // SAFETY: ptr from cuMemHostAlloc, freed exactly once.
        let rc = unsafe {
            let _ = (self.keep.api.cuCtxSetCurrent)(self.keep.ctx as CUcontext);
            (self.keep.api.cuMemFreeHost)(self.ptr as *mut c_void)
        };
        if rc != 0 {
            if self.keep.poisoned.get().is_some() {
                tracing::debug!(
                    rc,
                    len = self.len,
                    "cuMemFreeHost failed (context poisoned)"
                );
            } else {
                tracing::warn!(rc, len = self.len, "cuMemFreeHost failed");
            }
        }
    }
}

/// `cuTensorMapEncodeTiled` (CUDA 12.0+): out map (128 B, 128-aligned), dtype, rank,
/// globalAddress, globalDim[rank], globalStrides[rank-1] (bytes), boxDim[rank],
/// elementStrides[rank], interleave, swizzle, l2Promotion, oobFill.
type TmapEncodeFn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    *mut c_void,
    *const u64,
    *const u64,
    *const u32,
    *const u32,
    u32,
    u32,
    u32,
    u32,
) -> CUresult;

pub struct CudaBackend {
    api: Arc<Api>,
    /// See the resolve site in [`CudaBackend::open`] — `None` on pre-12.0 drivers.
    tmap_encode: Option<TmapEncodeFn>,
    /// Primary context, retained once; re-bound per call (threads vary).
    ctx: usize,
    /// Handed to every owned `DeviceMem`; its Drop releases the primary ctx.
    freer: Arc<CudaFreer>,
    pub device_ordinal: u8,
    /// `cuDriverGetVersion` (e.g. 12080 = CUDA 12.8) — the load-time ceiling
    /// on cubin toolkit versions; surfaced in the `module_load` error.
    driver_version: i32,
    name: String,
    sm_count: u32,
    warp_size: u32,
    compute_capability: (u32, u32),
    smem_optin: u32,
    /// Hardware-coherent pageable access (attr 100) — see
    /// [`Backend::coherent_host_dma`].
    coherent_host_dma: bool,
    /// Real loaded modules by placeholder-exclusive id (id 0 = "no module",
    /// handed out for an empty image so `ExecutorSet::bringup` works before
    /// any real cubin exists — the engine loads its module explicitly).
    modules: Mutex<FxHashMap<u64, usize>>,
    next_module: AtomicU64,
    /// Physical slab chunks kept across loads (`PLOW_SLAB_KEEP=1`,
    /// `VmmOps::pool_put`/`pool_take`). Device-local by construction (one
    /// pool per backend, one backend per device); leftovers released in Drop.
    slab_pool: Mutex<Vec<(u64, u64)>>,
    /// Set once, on the first fatal driver status ([`is_cuda_fatal`]): the
    /// fault that killed the context. When set, [`Self::bind`] short-circuits
    /// with a clone BEFORE touching the driver, so a poisoned context stops
    /// generating one driver error (and one log line) per queued dispatch.
    /// Teardown paths (Drop impls, `DeviceFree`) bypass `bind` and still
    /// attempt their driver calls — shared with `CudaFreer` so those sites
    /// can downgrade their expected failures to debug.
    poisoned: Arc<std::sync::OnceLock<DeviceErrorInfo>>,
}

// SAFETY: the driver API is thread-safe; the raw context/module handles are
// process-global driver objects, and every entry point re-binds the context.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// `dlopen` the driver, `cuInit`, retain the device's primary context, and
    /// read the device geometry. Fails cleanly on a host without a CUDA driver
    /// — the runtime then falls back to another backend.
    pub fn new(device_ordinal: u8) -> Result<Self> {
        // SONAME first, then the dev symlink, then the usual absolute homes —
        // a nix-glibc binary's dlopen does not consult the host ld cache, so
        // the bare SONAME alone misses a perfectly present driver.
        // `--libcuda` / `PLOW_LIBCUDA` overrides everything.
        let mut candidates: Vec<String> = Vec::new();
        if let Some(p) = crate::config::RuntimeConfig::get().nv.libcuda.clone() {
            tracing::debug!(path = %p, "PLOW_LIBCUDA override set");
            candidates.push(p);
        }
        for p in [
            "libcuda.so.1",
            "libcuda.so",
            // CUDA forward-compat driver, BEFORE the distro path. On a box whose
            // toolkit is newer than its kernel driver, the distro libcuda cannot
            // load the toolkit's cubins at all: nvcc 13.0 emits ELF ABI 8 and a
            // 570.x (CUDA 12.8) driver rejects it with CUDA_ERROR_INVALID_IMAGE.
            // ldconfig resolves the SONAME here correctly, but the bare-SONAME
            // candidates above miss for a nix-glibc binary (see the comment on
            // this list), so without an explicit entry we silently fall through
            // to the older driver and fail at the first cuModuleLoadData.
            "/usr/local/cuda/compat/libcuda.so.1",
            // Both multiarch homes: a Grace-Hopper / Grace-Blackwell host (sbsa)
            // keeps the driver under `aarch64-linux-gnu`, and `/lib` is not
            // always symlinked to `/usr/lib` there. Listing the x86_64 dir alone
            // sent every nix-built binary on a GH200 to the CPU fallback.
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
            "/lib/aarch64-linux-gnu/libcuda.so.1",
            "/usr/local/nvidia/lib64/libcuda.so.1",
            "/usr/lib64/libcuda.so.1",
        ] {
            candidates.push(p.to_string());
        }
        let mut lib = None;
        let mut last_err = String::new();
        for c in &candidates {
            tracing::trace!(candidate = %c, "trying dlopen");
            // SAFETY: loading a system shared library; no init runs beyond
            // the library's own constructors.
            match unsafe { libloading::Library::new(c) } {
                Ok(l) => {
                    tracing::debug!(path = %c, "dlopen libcuda succeeded");
                    lib = Some(l);
                    break;
                }
                Err(e) => {
                    tracing::trace!(path = %c, error = %e, "dlopen candidate failed");
                    last_err = format!("{c}: {e}");
                }
            }
        }
        let lib = lib.ok_or_else(|| RuntimeError::Device(format!("dlopen libcuda: {last_err}")))?;
        tracing::debug!("resolving CUDA driver API symbols...");
        // Resolved OPTIONALLY, outside driver_api!: cuTensorMapEncodeTiled is CUDA 12.0+,
        // and an eager resolve would make plowrt refuse to open ANY device on an older
        // driver. Absent symbol => `encode_tmap_bf16` errors, everything else unaffected.
        // SAFETY: signature transcribed from cuda.h (asserted working by
        // runtime/nvidia/experiments/tma_ws_gemm_bf16.cu on this driver line).
        let tmap_encode: Option<TmapEncodeFn> = unsafe {
            lib.get::<TmapEncodeFn>(b"cuTensorMapEncodeTiled\0")
                .ok()
                .map(|s| *s)
        };
        let api = Api::resolve(lib)?;

        let check = |rc: CUresult, what: &str| -> Result<()> {
            if rc == 0 {
                return Ok(());
            }
            Err(cu_fault(&api, rc, what))
        };

        // SAFETY below: signatures match the CUDA driver ABI (transcribed
        // from cuda.h; sizes/argument order asserted by the working vector-add
        // GPU test).
        unsafe {
            check((api.cuInit)(0), "cuInit")?;
            // The CUDA version this driver supports (e.g. 12080 = 12.8) — the
            // ceiling on what cubins it will load. Kept for the module_load
            // error path: a cubin from a newer nvcc fails there with the
            // opaque CUDA_ERROR_INVALID_IMAGE, and the version is the fact
            // that explains it.
            let mut driver_version = 0i32;
            check(
                (api.cuDriverGetVersion)(&mut driver_version),
                "cuDriverGetVersion",
            )?;
            let mut dev: CUdevice = 0;
            check(
                (api.cuDeviceGet)(&mut dev, device_ordinal as i32),
                "cuDeviceGet",
            )?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            check(
                (api.cuDevicePrimaryCtxRetain)(&mut ctx, dev),
                "cuDevicePrimaryCtxRetain",
            )?;
            check((api.cuCtxSetCurrent)(ctx), "cuCtxSetCurrent")?;

            let mut buf = [0 as c_char; 128];
            check(
                (api.cuDeviceGetName)(buf.as_mut_ptr(), 128, dev),
                "cuDeviceGetName",
            )?;
            let name = std::ffi::CStr::from_ptr(buf.as_ptr())
                .to_string_lossy()
                .into_owned();

            let attr = |a: i32, what: &str| -> Result<i32> {
                let mut v = 0i32;
                check((api.cuDeviceGetAttribute)(&mut v, a, dev), what)?;
                Ok(v)
            };
            let sm_count = attr(ATTR_MULTIPROCESSOR_COUNT, "attr sm_count")? as u32;
            let warp_size = attr(ATTR_WARP_SIZE, "attr warp_size")? as u32;
            let cc_major = attr(ATTR_COMPUTE_CAPABILITY_MAJOR, "attr cc major")? as u32;
            let cc_minor = attr(ATTR_COMPUTE_CAPABILITY_MINOR, "attr cc minor")? as u32;
            let compute_capability = (cc_major, cc_minor);
            let smem_optin = attr(ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, "attr smem")? as u32;
            // Unknown attribute (older driver) reads as "no coherent access" —
            // the staged upload path is always correct.
            let coherent_host_dma = {
                let mut v = 0i32;
                // SAFETY: attribute query; failure leaves v = 0.
                let rc =
                    (api.cuDeviceGetAttribute)(&mut v, ATTR_PAGEABLE_USES_HOST_PAGE_TABLES, dev);
                rc == 0 && v == 1
            };
            let coop = attr(ATTR_COOPERATIVE_LAUNCH, "attr cooperative")?;
            if coop == 0 {
                return Err(RuntimeError::Device(format!(
                    "{name}: no cooperative-launch support (required for the persistent interp)"
                )));
            }

            tracing::info!(
                %name,
                sm_count,
                smem_optin,
                cc_major,
                cc_minor,
                coherent_host_dma,
                driver = %format_args!("{}.{}", driver_version / 1000, (driver_version % 1000) / 10),
                "CUDA device ready"
            );
            let api = Arc::new(api);
            let poisoned = Arc::new(std::sync::OnceLock::new());
            let freer = Arc::new(CudaFreer {
                api: Arc::clone(&api),
                ctx: ctx as usize,
                dev,
                poisoned: Arc::clone(&poisoned),
            });
            Ok(CudaBackend {
                api,
                tmap_encode,
                ctx: ctx as usize,
                freer,
                device_ordinal,
                driver_version,
                name,
                sm_count,
                warp_size,
                compute_capability,
                smem_optin,
                coherent_host_dma,
                modules: Mutex::new(FxHashMap::default()),
                next_module: AtomicU64::new(1),
                slab_pool: Mutex::new(Vec::new()),
                poisoned,
            })
        }
    }

    /// Encode the `tma_ws_gemm_bf16.cu` tensor-map recipe over a device tensor: rank-2
    /// `[rows][k]` bf16 K-major, box `{64, box_rows}`, 128 B swizzle, L2 128 B promotion,
    /// OOB zero-fill. Returns the 128 descriptor bytes for the loader to upload into the
    /// map tensor's buffer (`GEN_TMAP_BF16`). Errors on drivers older than CUDA 12.0.
    pub fn encode_tmap_bf16(
        &self,
        base: u64,
        rows: u32,
        k: u32,
        box_rows: u32,
    ) -> Result<[u8; 128]> {
        self.encode_tmap(base, rows, k, box_rows, false)
    }

    /// e4m3 twin: same 128 B swizzle, dtype UINT8, inner box 128 elems (= 128 B).
    pub fn encode_tmap_e4m3(
        &self,
        base: u64,
        rows: u32,
        k: u32,
        box_rows: u32,
    ) -> Result<[u8; 128]> {
        self.encode_tmap(base, rows, k, box_rows, true)
    }

    /// Rank-3 bf16 map over a `[n_kv_head][ring_rows][hd]` KV cache tensor: globalDim
    /// {hd, ring_rows, n_kv_head}, box {64, box_rows, 1}, 128 B swizzle — the flash-prefill
    /// TMA stager's recipe (`GEN_TMAP_KV_PAIR`, one such map per K and per V).
    pub fn encode_tmap_kv3(
        &self,
        base: u64,
        ring_rows: u32,
        hd: u32,
        n_kv_head: u32,
        box_rows: u32,
    ) -> Result<[u8; 128]> {
        let f = self.tmap_encode.ok_or_else(|| {
            RuntimeError::Device(
                "cuTensorMapEncodeTiled unresolved (driver < CUDA 12.0) but this packet \
                 carries GEN_TMAP tensors"
                    .into(),
            )
        })?;
        assert!(
            hd % 64 == 0 && hd > 0,
            "GEN_TMAP_KV_PAIR needs hd%64==0, got {hd}"
        );
        #[repr(C, align(128))]
        struct Buf([u8; 128]);
        let mut m = Buf([0u8; 128]);
        let gd = [hd as u64, ring_rows as u64, n_kv_head as u64];
        let gs = [hd as u64 * 2, ring_rows as u64 * hd as u64 * 2];
        let bd = [64u32, box_rows, 1u32];
        let es = [1u32, 1u32, 1u32];
        self.check(
            unsafe { (self.api.cuCtxSetCurrent)(self.ctx as CUcontext) },
            "cuCtxSetCurrent",
        )?;
        // SAFETY: as encode_tmap; rank 3.
        let rc = unsafe {
            f(
                &mut m as *mut Buf as *mut c_void,
                9,
                3,
                base as *mut c_void,
                gd.as_ptr(),
                gs.as_ptr(),
                bd.as_ptr(),
                es.as_ptr(),
                0,
                3,
                2,
                0,
            )
        };
        self.check(rc, "cuTensorMapEncodeTiled(kv3)")?;
        Ok(m.0)
    }

    fn encode_tmap(
        &self,
        base: u64,
        rows: u32,
        k: u32,
        box_rows: u32,
        e4m3: bool,
    ) -> Result<[u8; 128]> {
        let f = self.tmap_encode.ok_or_else(|| {
            RuntimeError::Device(
                "cuTensorMapEncodeTiled unresolved (driver < CUDA 12.0) but this packet \
                 carries GEN_TMAP tensors"
                    .into(),
            )
        })?;
        // globalStrides entries must be 16-byte multiples; devgen only mints maps for
        // K%16B==0 (the kernel traps otherwise), so this is a build bug, not input.
        assert!(
            k > 0 && (k as u64 * if e4m3 { 1 } else { 2 }) % 16 == 0,
            "GEN_TMAP K misaligned: {k}"
        );
        #[repr(C, align(128))]
        struct Buf([u8; 128]);
        let mut m = Buf([0u8; 128]);
        let gd = [k as u64, rows as u64];
        let gs = [k as u64 * if e4m3 { 1 } else { 2 }];
        let bd = [if e4m3 { 128u32 } else { 64u32 }, box_rows];
        let es = [1u32, 1u32];
        self.check(
            unsafe { (self.api.cuCtxSetCurrent)(self.ctx as CUcontext) },
            "cuCtxSetCurrent",
        )?;
        // Constants transcribed from cuda.h, byte-for-byte the probe's make_map():
        // BFLOAT16=9 / UINT8=0, rank=2, INTERLEAVE_NONE=0, SWIZZLE_128B=3,
        // L2_PROMOTION_L2_128B=2, OOB_FILL_NONE=0.
        // SAFETY: driver call; buffers outlive it; out is 128 B and 128-aligned.
        let rc = unsafe {
            f(
                &mut m as *mut Buf as *mut c_void,
                if e4m3 { 0 } else { 9 },
                2,
                base as *mut c_void,
                gd.as_ptr(),
                gs.as_ptr(),
                bd.as_ptr(),
                es.as_ptr(),
                0,
                3,
                2,
                0,
            )
        };
        self.check(rc, "cuTensorMapEncodeTiled")?;
        Ok(m.0)
    }

    /// Map a nonzero CUresult to a typed error, resolving the driver's name
    /// and classifying it ([`is_cuda_fatal`]). A fatal status poisons the
    /// backend: `bind` then refuses further work up front.
    fn check(&self, rc: CUresult, what: &str) -> Result<()> {
        if rc == 0 {
            return Ok(());
        }
        let err = cu_fault(&self.api, rc, what);
        if let Some(info) = err.device_fault() {
            if info.fatal {
                self.mark_poisoned(info);
            }
        }
        Err(err)
    }

    /// Has a fatal driver status permanently poisoned this context?
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.get().is_some()
    }

    /// Record the fault that killed the context. Logs `error!` exactly once —
    /// every later dispatch is short-circuited by `bind`, not re-logged.
    fn mark_poisoned(&self, info: &DeviceErrorInfo) {
        if self.poisoned.set(info.clone()).is_ok() {
            tracing::error!(
                error_op = %info.operation,
                error_code = info.code,
                error_name = %info.name,
                device = self.device_ordinal,
                "CUDA context poisoned — rejecting further work on this device"
            );
        }
    }

    /// Re-bind the primary context on the calling thread (see module docs).
    /// Elides the driver call when this thread already has `self.ctx` bound.
    /// On a poisoned context this short-circuits with the recorded fault
    /// BEFORE calling the driver — every public entry point crosses here, so
    /// this is what stops a dead context from flooding the log.
    fn bind(&self) -> Result<()> {
        if let Some(info) = self.poisoned.get() {
            return Err(RuntimeError::DeviceFault { info: info.clone() });
        }
        thread_local! {
            static LAST_CTX: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        }
        let ctx = self.ctx;
        if LAST_CTX.with(|c| c.get()) == ctx {
            return Ok(());
        }
        // SAFETY: rebinding a retained context.
        self.check(
            unsafe { (self.api.cuCtxSetCurrent)(ctx as CUcontext) },
            "cuCtxSetCurrent",
        )?;
        LAST_CTX.with(|c| c.set(ctx));
        Ok(())
    }

    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }

    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Resolve a kernel by (possibly mangled) name in a loaded module.
    pub fn get_function(&self, module: &Module, name: &str) -> Result<KernelFn> {
        self.bind()?;
        let raw = *self.modules.lock().get(&module.id).ok_or_else(|| {
            RuntimeError::Device(format!("get_function: module {} not loaded", module.id))
        })?;
        let cname = std::ffi::CString::new(name)
            .map_err(|_| RuntimeError::Device("kernel name contains NUL".into()))?;
        let mut f: CUfunction = std::ptr::null_mut();
        // SAFETY: module handle from cuModuleLoadData; name is NUL-terminated.
        self.check(
            unsafe { (self.api.cuModuleGetFunction)(&mut f, raw as CUmodule, cname.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        Ok(KernelFn(f as usize))
    }

    /// Read a `u32` module-scope global (`__constant__`/`__device__`) by
    /// name. `Ok(None)` when the module has no such symbol — older cubins
    /// predate the embedded metadata, and the caller keeps its fallback.
    pub fn module_global_u32(&self, module: &Module, name: &str) -> Result<Option<u32>> {
        self.bind()?;
        let raw = *self.modules.lock().get(&module.id).ok_or_else(|| {
            RuntimeError::Device(format!(
                "module_global_u32: module {} not loaded",
                module.id
            ))
        })?;
        let cname = std::ffi::CString::new(name)
            .map_err(|_| RuntimeError::Device("global name contains NUL".into()))?;
        let mut ptr: CUdeviceptr = 0;
        let mut bytes: usize = 0;
        // SAFETY: module handle from cuModuleLoadData; name is NUL-terminated.
        let rc = unsafe {
            (self.api.cuModuleGetGlobal_v2)(&mut ptr, &mut bytes, raw as CUmodule, cname.as_ptr())
        };
        if rc != 0 {
            return Ok(None); // CUDA_ERROR_NOT_FOUND: symbol absent from this object
        }
        if bytes != 4 {
            return Err(RuntimeError::Device(format!(
                "module global {name} is {bytes} B, want 4"
            )));
        }
        let mut v = 0u32;
        // SAFETY: 4-byte device-to-host copy from the symbol's device address.
        self.check(
            unsafe { (self.api.cuMemcpyDtoH_v2)(&mut v as *mut u32 as *mut c_void, ptr, 4) },
            &format!("cuMemcpyDtoH({name})"),
        )?;
        Ok(Some(v))
    }

    /// Read up to `max` bytes of a module-scope global (`__device__`/
    /// `__constant__`) by name into `out` (cleared first). `Ok(false)` when
    /// the symbol is absent — the caller keeps its fallback. Copies
    /// `min(max, symbol size)` bytes. Used by the PLOW_NV_TRACE readback
    /// (stage-7 profiling); never on the token critical path.
    pub fn module_global_bytes(
        &self,
        module: &Module,
        name: &str,
        max: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool> {
        self.bind()?;
        let raw = *self.modules.lock().get(&module.id).ok_or_else(|| {
            RuntimeError::Device(format!(
                "module_global_bytes: module {} not loaded",
                module.id
            ))
        })?;
        let cname = std::ffi::CString::new(name)
            .map_err(|_| RuntimeError::Device("global name contains NUL".into()))?;
        let mut ptr: CUdeviceptr = 0;
        let mut bytes: usize = 0;
        // SAFETY: module handle from cuModuleLoadData; name is NUL-terminated.
        let rc = unsafe {
            (self.api.cuModuleGetGlobal_v2)(&mut ptr, &mut bytes, raw as CUmodule, cname.as_ptr())
        };
        if rc != 0 {
            return Ok(false); // symbol absent from this object
        }
        let n = bytes.min(max);
        out.clear();
        out.resize(n, 0);
        // SAFETY: n <= symbol size; host buffer holds n bytes.
        self.check(
            unsafe { (self.api.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, ptr, n) },
            &format!("cuMemcpyDtoH({name})"),
        )?;
        Ok(true)
    }

    /// Zero the first `n` bytes of a module global by name. `Ok(false)` when
    /// the symbol is absent. Used to reset the PLOW_NV_TRACE counter between
    /// warmup and the measured window (stage-7 profiling); off the hot path.
    pub fn module_global_zero(&self, module: &Module, name: &str, n: usize) -> Result<bool> {
        self.bind()?;
        let raw = *self.modules.lock().get(&module.id).ok_or_else(|| {
            RuntimeError::Device(format!(
                "module_global_zero: module {} not loaded",
                module.id
            ))
        })?;
        let cname = std::ffi::CString::new(name)
            .map_err(|_| RuntimeError::Device("global name contains NUL".into()))?;
        let mut ptr: CUdeviceptr = 0;
        let mut bytes: usize = 0;
        // SAFETY: module handle from cuModuleLoadData; name is NUL-terminated.
        let rc = unsafe {
            (self.api.cuModuleGetGlobal_v2)(&mut ptr, &mut bytes, raw as CUmodule, cname.as_ptr())
        };
        if rc != 0 {
            return Ok(false);
        }
        self.memset_d8(ptr, 0, n.min(bytes))?;
        Ok(true)
    }

    /// Opt a kernel into > 48 KiB dynamic shared memory.
    pub fn set_max_dynamic_smem(&self, f: KernelFn, bytes: u32) -> Result<()> {
        self.bind()?;
        if bytes > self.smem_optin {
            return Err(RuntimeError::Device(format!(
                "dynamic smem {bytes} B exceeds device opt-in limit {}",
                self.smem_optin
            )));
        }
        // SAFETY: valid function handle.
        self.check(
            unsafe {
                (self.api.cuFuncSetAttribute)(
                    f.0 as CUfunction,
                    FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
            },
            "cuFuncSetAttribute(max dynamic smem)",
        )
    }

    /// Occupancy-derived resident blocks per SM for `(block, smem)` — the
    /// harness's grid rule is `blocks_per_sm * sm_count`, and co-residency is a
    /// correctness condition for the counter protocol, not a tuning knob.
    pub fn occupancy_blocks_per_sm(&self, f: KernelFn, block: u32, smem: usize) -> Result<u32> {
        self.bind()?;
        let mut n = 0i32;
        // SAFETY: valid function handle.
        self.check(
            unsafe {
                (self.api.cuOccupancyMaxActiveBlocksPerMultiprocessor)(
                    &mut n,
                    f.0 as CUfunction,
                    block as i32,
                    smem,
                )
            },
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )?;
        Ok(n as u32)
    }

    /// Cooperative launch (all blocks co-resident or the launch fails — which
    /// turns a would-be counter deadlock into a loud launch-time error).
    /// `params` is the driver's kernelParams array: one pointer per kernel
    /// parameter, each pointing at the host copy of that argument. `stream`
    /// orders the launch on an engine stream; `None` = the legacy default
    /// stream (harness-identical behavior).
    pub fn launch_cooperative(
        &self,
        f: KernelFn,
        grid: u32,
        block: u32,
        smem_bytes: u32,
        params: &mut [*mut c_void],
        stream: Option<&CudaStream>,
    ) -> Result<()> {
        self.bind()?;
        // SAFETY: function/param pointers valid for the duration of the call —
        // the driver copies kernel arguments out before returning.
        self.check(
            unsafe {
                (self.api.cuLaunchCooperativeKernel)(
                    f.0 as CUfunction,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    smem_bytes,
                    stream.map_or(std::ptr::null_mut(), |s| s.raw as CUstream),
                    params.as_mut_ptr(),
                )
            },
            "cuLaunchCooperativeKernel",
        )
    }

    /// Ordinary (non-cooperative) kernel launch on `stream` (`None` = default
    /// stream). Used by the standalone device sampler epilogue (stage 4),
    /// which is a plain grid launched after the cooperative decode kernel on
    /// the same ordered stream — not co-resident, so it needs no cooperative
    /// launch and no occupancy gate. `params` is the driver kernelParams
    /// array (one pointer per argument).
    pub fn launch_kernel(
        &self,
        f: KernelFn,
        grid: u32,
        block: u32,
        smem_bytes: u32,
        params: &mut [*mut c_void],
        stream: Option<&CudaStream>,
    ) -> Result<()> {
        self.bind()?;
        // SAFETY: param pointers valid for the call; the driver copies kernel
        // arguments out before returning.
        self.check(
            unsafe {
                (self.api.cuLaunchKernel)(
                    f.0 as CUfunction,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    smem_bytes,
                    stream.map_or(std::ptr::null_mut(), |s| s.raw as CUstream),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// T35: build + instantiate a serial kernel-node chain (each node depends on its
    /// predecessor). `params_blobs[i]` is the by-value kernarg for node i (copied out by
    /// the driver during node creation). One `graph_launch` then replaces N submits.
    pub fn graph_build_chain(
        &self,
        nodes: &[(KernelFn, u32, u32, u32)],
        params_blobs: &mut [*mut c_void],
    ) -> Result<GraphExec> {
        self.bind()?;
        let mut graph: CUgraph = std::ptr::null_mut();
        self.check(
            unsafe { (self.api.cuGraphCreate)(&mut graph, 0) },
            "cuGraphCreate",
        )?;
        let mut prev: CUgraphNode = std::ptr::null_mut();
        for (i, &(f, grid, block, smem)) in nodes.iter().enumerate() {
            let mut kp = [params_blobs[i]];
            let np = CudaKernelNodeParams {
                func: f.0 as *mut c_void,
                grid_x: grid,
                grid_y: 1,
                grid_z: 1,
                block_x: block,
                block_y: 1,
                block_z: 1,
                shared_mem_bytes: smem,
                kernel_params: kp.as_mut_ptr(),
                extra: std::ptr::null_mut(),
                kern: std::ptr::null_mut(),
                ctx: std::ptr::null_mut(),
            };
            let deps = [prev];
            let ndeps = if prev.is_null() { 0 } else { 1 };
            let mut node: CUgraphNode = std::ptr::null_mut();
            let r = unsafe {
                (self.api.cuGraphAddKernelNode_v2)(
                    &mut node,
                    graph,
                    if ndeps == 0 {
                        std::ptr::null()
                    } else {
                        deps.as_ptr()
                    },
                    ndeps,
                    &np,
                )
            };
            if r != 0 {
                unsafe { (self.api.cuGraphDestroy)(graph) };
                self.check(r, "cuGraphAddKernelNode_v2")?;
            }
            prev = node;
        }
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let r = unsafe { (self.api.cuGraphInstantiateWithFlags)(&mut exec, graph, 0) };
        unsafe { (self.api.cuGraphDestroy)(graph) };
        self.check(r, "cuGraphInstantiateWithFlags")?;
        Ok(GraphExec {
            exec,
            api: std::ptr::null(),
        })
    }

    pub(crate) fn graph_capture(
        &self,
        stream: &CudaStream,
        enqueue: impl FnOnce() -> Result<()>,
    ) -> Result<GraphExec> {
        self.bind()?;
        // Thread-local capture; callers enqueue only immutable same-stream operations.
        self.check(
            unsafe { (self.api.cuStreamBeginCapture)(stream.raw as CUstream, 1) },
            "cuStreamBeginCapture",
        )?;
        let result = enqueue();
        let mut graph = std::ptr::null_mut();
        let ended = unsafe { (self.api.cuStreamEndCapture)(stream.raw as CUstream, &mut graph) };
        if let Err(e) = result {
            if !graph.is_null() {
                unsafe {
                    (self.api.cuGraphDestroy)(graph);
                }
            }
            return Err(e);
        }
        self.check(ended, "cuStreamEndCapture")?;
        let mut exec = std::ptr::null_mut();
        let status = unsafe { (self.api.cuGraphInstantiateWithFlags)(&mut exec, graph, 0) };
        unsafe {
            (self.api.cuGraphDestroy)(graph);
        }
        self.check(status, "cuGraphInstantiateWithFlags")?;
        Ok(GraphExec {
            exec,
            api: self.api.as_ref() as *const _ as *const c_void,
        })
    }

    pub fn graph_launch(&self, g: &GraphExec, stream: &CudaStream) -> Result<()> {
        self.bind()?;
        self.check(
            unsafe { (self.api.cuGraphLaunch)(g.exec, stream.raw as CUstream) },
            "cuGraphLaunch",
        )
    }

    pub fn graph_destroy(&self, g: GraphExec) {
        unsafe {
            let _ = (self.api.cuGraphExecDestroy)(g.exec);
        }
        std::mem::forget(g);
    }

    /// Block until every queued launch/copy on the context has retired.
    /// Steady-state serving must never call this (the plan's gate) — it is the
    /// load/unload and error-path quiesce.
    pub fn synchronize(&self) -> Result<()> {
        self.bind()?;
        // SAFETY: no arguments.
        self.check(unsafe { (self.api.cuCtxSynchronize)() }, "cuCtxSynchronize")
    }

    /// One ordered execution stream (`CU_STREAM_NON_BLOCKING`) — the engine's
    /// device queue for every decode/prefill copy, memset, and launch.
    pub fn stream_create(&self) -> Result<CudaStream> {
        self.bind()?;
        let mut s: CUstream = std::ptr::null_mut();
        // SAFETY: out-pointer to a driver stream handle.
        self.check(
            unsafe { (self.api.cuStreamCreate)(&mut s, STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        )?;
        Ok(CudaStream {
            raw: s as usize,
            keep: self.freer.clone(),
        })
    }

    /// Block until every operation queued on `stream` has retired. The
    /// steady-state serving sync point — narrower than [`Self::synchronize`].
    pub fn stream_synchronize(&self, stream: &CudaStream) -> Result<()> {
        self.bind()?;
        // SAFETY: live stream handle.
        self.check(
            unsafe { (self.api.cuStreamSynchronize)(stream.raw as CUstream) },
            "cuStreamSynchronize",
        )
    }

    /// Async H2D on `stream`.
    ///
    /// # Safety
    /// The copy retires when the stream reaches it, not when this returns:
    /// `src` must stay valid (unmoved, undropped) until the stream is
    /// synchronized past this op, and `dptr..dptr+src.len()` must be a live
    /// device range. Pinned `src` ([`Self::host_alloc_pinned`]) makes the copy
    /// truly asynchronous; pageable degrades to a staged copy.
    pub unsafe fn memcpy_htod_async(
        &self,
        dptr: u64,
        src: &[u8],
        stream: &CudaStream,
    ) -> Result<()> {
        self.bind()?;
        self.check(
            // SAFETY: caller upholds the lifetime/range contract above.
            unsafe {
                (self.api.cuMemcpyHtoDAsync_v2)(
                    dptr,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    stream.raw as CUstream,
                )
            },
            "cuMemcpyHtoDAsync",
        )
    }

    /// Async D2H on `stream`.
    ///
    /// # Safety
    /// As [`Self::memcpy_htod_async`], reversed: `dst` is written when the
    /// stream reaches the op and must stay valid until the synchronize.
    pub unsafe fn memcpy_dtoh_async(
        &self,
        dst: &mut [u8],
        dptr: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        self.bind()?;
        self.check(
            // SAFETY: caller upholds the lifetime/range contract above.
            unsafe {
                (self.api.cuMemcpyDtoHAsync_v2)(
                    dst.as_mut_ptr() as *mut c_void,
                    dptr,
                    dst.len(),
                    stream.raw as CUstream,
                )
            },
            "cuMemcpyDtoHAsync",
        )
    }

    /// Async fill of `n` bytes on `stream` (counter/cursor re-arm without a
    /// synchronous submission stall). `dptr..dptr+n` inside a live allocation
    /// (caller contract, as [`Self::memset_d8`]).
    pub fn memset_d8_async(
        &self,
        dptr: u64,
        value: u8,
        n: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.bind()?;
        // SAFETY: device range contract above; no host memory involved.
        self.check(
            unsafe { (self.api.cuMemsetD8Async)(dptr, value, n, stream.raw as CUstream) },
            "cuMemsetD8Async",
        )
    }

    /// A CUDA event. `timing = true` supports [`Self::event_elapsed_ms`]
    /// (device-timeline instrumentation); `false` records cheaper (sync-only).
    pub fn event_create(&self, timing: bool) -> Result<CudaEvent> {
        self.bind()?;
        let flags = if timing {
            EVENT_DEFAULT
        } else {
            EVENT_DISABLE_TIMING
        };
        let mut e: CUevent = std::ptr::null_mut();
        // SAFETY: out-pointer to a driver event handle.
        self.check(
            unsafe { (self.api.cuEventCreate)(&mut e, flags) },
            "cuEventCreate",
        )?;
        Ok(CudaEvent {
            raw: e as usize,
            keep: self.freer.clone(),
        })
    }

    /// Record `event` at the stream's current tail.
    pub fn event_record(&self, event: &CudaEvent, stream: &CudaStream) -> Result<()> {
        self.bind()?;
        // SAFETY: live event/stream handles.
        self.check(
            unsafe { (self.api.cuEventRecord)(event.raw as CUevent, stream.raw as CUstream) },
            "cuEventRecord",
        )
    }

    /// Non-blocking completion poll: `Ok(true)` once every op preceding the
    /// event's record has retired.
    pub fn event_query(&self, event: &CudaEvent) -> Result<bool> {
        self.bind()?;
        // SAFETY: live event handle.
        match unsafe { (self.api.cuEventQuery)(event.raw as CUevent) } {
            0 => Ok(true),
            ERROR_NOT_READY => Ok(false),
            rc => self.check(rc, "cuEventQuery").map(|_| false),
        }
    }

    /// Block until the event's recorded point has retired.
    pub fn event_synchronize(&self, event: &CudaEvent) -> Result<()> {
        self.bind()?;
        // SAFETY: live event handle.
        self.check(
            unsafe { (self.api.cuEventSynchronize)(event.raw as CUevent) },
            "cuEventSynchronize",
        )
    }

    /// Device-timeline milliseconds between two retired timing events — the
    /// plan's instrumentation rule: CUDA events, not CPU timestamps around
    /// synchronous calls.
    pub fn event_elapsed_ms(&self, start: &CudaEvent, end: &CudaEvent) -> Result<f32> {
        self.bind()?;
        let mut ms = 0f32;
        // SAFETY: both events recorded with timing enabled (caller contract).
        self.check(
            unsafe {
                (self.api.cuEventElapsedTime)(&mut ms, start.raw as CUevent, end.raw as CUevent)
            },
            "cuEventElapsedTime",
        )?;
        Ok(ms)
    }

    /// Owned pinned host memory (portable) — staging for true-async copies
    /// and the checkpoint upload. RAII: freed on drop (pinned pages are a
    /// scarce OS resource, not just process memory).
    pub fn host_alloc_pinned(&self, bytes: usize) -> Result<PinnedHost> {
        let ptr = self.host_alloc(bytes)?;
        Ok(PinnedHost {
            ptr,
            len: bytes,
            keep: self.freer.clone(),
        })
    }

    /// H2D copy to a raw device pointer (the engine patches sub-ranges of its
    /// own allocations; `Backend::upload` wraps this for `DeviceMem`).
    pub fn memcpy_htod(&self, dptr: u64, src: &[u8]) -> Result<()> {
        self.bind()?;
        // SAFETY: dptr from cuMemAlloc with at least src.len() bytes (caller
        // contract); src outlives the synchronous copy.
        self.check(
            unsafe { (self.api.cuMemcpyHtoD_v2)(dptr, src.as_ptr() as *const c_void, src.len()) },
            "cuMemcpyHtoD",
        )
    }

    /// D2H copy from a raw device pointer.
    pub fn memcpy_dtoh(&self, dst: &mut [u8], dptr: u64) -> Result<()> {
        self.bind()?;
        // SAFETY: as memcpy_htod, reversed.
        self.check(
            unsafe { (self.api.cuMemcpyDtoH_v2)(dst.as_mut_ptr() as *mut c_void, dptr, dst.len()) },
            "cuMemcpyDtoH",
        )
    }

    /// Fill `n` bytes at a raw device pointer (counter zeroing each step).
    pub fn memset_d8(&self, dptr: u64, value: u8, n: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: dptr..dptr+n inside a live allocation (caller contract).
        self.check(
            unsafe { (self.api.cuMemsetD8_v2)(dptr, value, n) },
            "cuMemsetD8",
        )
    }

    /// Raw pinned host memory (portable) — [`Self::host_alloc_pinned`]'s
    /// backing call; freed by `PinnedHost::drop`.
    fn host_alloc(&self, bytes: usize) -> Result<*mut u8> {
        self.bind()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-pointer to a driver allocation.
        self.check(
            unsafe { (self.api.cuMemHostAlloc)(&mut p, bytes, MEMHOSTALLOC_PORTABLE) },
            "cuMemHostAlloc",
        )?;
        Ok(p as *mut u8)
    }

    /// Unload a module loaded via [`Backend::module_load`], releasing its
    /// device-side storage (code, globals). Idempotent: the placeholder
    /// module (id 0) and an already-unloaded module are no-ops, so an
    /// explicit unload followed by the engine's Drop never double-unloads.
    /// Any `KernelFn` resolved from the module is invalid afterwards.
    pub fn module_unload(&self, module: &Module) -> Result<()> {
        if module.id == 0 {
            return Ok(());
        }
        let Some(raw) = self.modules.lock().remove(&module.id) else {
            return Ok(()); // already unloaded
        };
        self.bind()?;
        // SAFETY: handle came from cuModuleLoadData and was removed from the
        // map above, so it is unloaded exactly once.
        self.check(
            unsafe { (self.api.cuModuleUnload)(raw as CUmodule) },
            "cuModuleUnload",
        )
    }

    /// `(free, total)` device memory in bytes (`cuMemGetInfo`) — the VRAM
    /// ledger the unload/reload lifecycle tests assert against.
    pub fn mem_info(&self) -> Result<(u64, u64)> {
        self.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two out-pointers.
        self.check(
            unsafe { (self.api.cuMemGetInfo_v2)(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok((free as u64, total as u64))
    }

    /// This device's `CUmemAllocationProp` for pinned device-local VMM
    /// physical allocations (the shape both probes used).
    fn vmm_prop(&self) -> CUmemAllocationProp {
        CUmemAllocationProp {
            type_: MEM_ALLOCATION_TYPE_PINNED,
            requested_handle_types: 0,
            location: CUmemLocation {
                type_: MEM_LOCATION_TYPE_DEVICE,
                id: self.device_ordinal as i32,
            },
            win32_handle_meta_data: std::ptr::null_mut(),
            alloc_flags: [0; 8],
        }
    }

    pub fn memcpy_dtod(&self, dst: u64, src: u64, bytes: u64) -> Result<()> {
        self.bind()?;
        // SAFETY: both ranges inside live (mapped) allocations — caller contract.
        self.check(
            unsafe { (self.api.cuMemcpyDtoD_v2)(dst, src, bytes as usize) },
            "cuMemcpyDtoD",
        )
    }
}

/// The VMM driver surface as `crate::memory::vmm` consumes it. Thin, checked
/// pass-throughs; every entry re-binds the primary context (threads vary —
/// the pool's pre-mapper thread calls in from its own thread).
impl crate::memory::vmm::VmmOps for CudaBackend {
    fn granularity(&self) -> Result<u64> {
        self.bind()?;
        let prop = self.vmm_prop();
        let mut g = 0usize;
        // SAFETY: out-pointer + prop transcribed from cuda.h.
        self.check(
            unsafe {
                (self.api.cuMemGetAllocationGranularity)(
                    &mut g,
                    &prop,
                    MEM_ALLOC_GRANULARITY_RECOMMENDED,
                )
            },
            "cuMemGetAllocationGranularity",
        )?;
        Ok(g as u64)
    }

    fn reserve(&self, bytes: u64) -> Result<u64> {
        self.bind()?;
        let mut va: CUdeviceptr = 0;
        // SAFETY: out-pointer; no alignment/fixed-address request.
        self.check(
            unsafe { (self.api.cuMemAddressReserve)(&mut va, bytes as usize, 0, 0, 0) },
            "cuMemAddressReserve",
        )?;
        Ok(va)
    }

    fn address_free(&self, va: u64, bytes: u64) {
        // SAFETY: va/bytes from a prior reserve; freed exactly once (pool
        // contract). Infallible teardown path — log, don't propagate.
        let rc = unsafe {
            let _ = (self.api.cuCtxSetCurrent)(self.ctx as CUcontext);
            (self.api.cuMemAddressFree)(va, bytes as usize)
        };
        if rc != 0 {
            if self.is_poisoned() {
                tracing::debug!(rc, va, bytes, "cuMemAddressFree failed (context poisoned)");
            } else {
                tracing::warn!(rc, va, bytes, "cuMemAddressFree failed");
            }
        }
    }

    fn create(&self, bytes: u64) -> Result<u64> {
        self.bind()?;
        let prop = self.vmm_prop();
        let mut h = 0u64;
        // SAFETY: out-pointer; bytes is a granularity multiple (pool contract).
        self.check(
            unsafe { (self.api.cuMemCreate)(&mut h, bytes as usize, &prop, 0) },
            "cuMemCreate",
        )?;
        Ok(h)
    }

    fn release(&self, handle: u64) {
        // SAFETY: handle from create, released exactly once (pool refcount).
        let rc = unsafe {
            let _ = (self.api.cuCtxSetCurrent)(self.ctx as CUcontext);
            (self.api.cuMemRelease)(handle)
        };
        if rc != 0 {
            if self.is_poisoned() {
                tracing::debug!(rc, handle, "cuMemRelease failed (context poisoned)");
            } else {
                tracing::warn!(rc, handle, "cuMemRelease failed");
            }
        }
    }

    fn map(&self, va: u64, bytes: u64, handle: u64) -> Result<()> {
        self.bind()?;
        // SAFETY: va range inside a reservation, handle live, offset 0 —
        // multi-map of one handle into several ranges is legal (probe [2]).
        self.check(
            unsafe { (self.api.cuMemMap)(va, bytes as usize, 0, handle, 0) },
            "cuMemMap",
        )
        .map_err(|e| {
            RuntimeError::Device(format!("{e} (va={va:#x} bytes={bytes} handle={handle:#x})"))
        })
    }

    fn unmap(&self, va: u64, bytes: u64) {
        // SAFETY: exactly the mapped range (pool contract).
        let rc = unsafe {
            let _ = (self.api.cuCtxSetCurrent)(self.ctx as CUcontext);
            (self.api.cuMemUnmap)(va, bytes as usize)
        };
        if rc != 0 {
            if self.is_poisoned() {
                tracing::debug!(rc, va, bytes, "cuMemUnmap failed (context poisoned)");
            } else {
                tracing::warn!(rc, va, bytes, "cuMemUnmap failed");
            }
        }
    }

    fn set_access(&self, va: u64, bytes: u64) -> Result<()> {
        self.bind()?;
        let desc = CUmemAccessDesc {
            location: CUmemLocation {
                type_: MEM_LOCATION_TYPE_DEVICE,
                id: self.device_ordinal as i32,
            },
            flags: MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: range fully mapped (pool maps before granting access).
        self.check(
            unsafe { (self.api.cuMemSetAccess)(va, bytes as usize, &desc, 1) },
            "cuMemSetAccess",
        )
    }

    fn alloc(&self, bytes: u64) -> Result<u64> {
        self.bind()?;
        let mut dptr: CUdeviceptr = 0;
        // SAFETY: out-pointer to a device allocation (snapshot buffers).
        self.check(
            unsafe { (self.api.cuMemAlloc_v2)(&mut dptr, bytes as usize) },
            "cuMemAlloc(vmm snapshot)",
        )?;
        Ok(dptr)
    }

    fn free(&self, va: u64) {
        // SAFETY: va from VmmOps::alloc, freed exactly once (pool contract).
        let rc = unsafe {
            let _ = (self.api.cuCtxSetCurrent)(self.ctx as CUcontext);
            (self.api.cuMemFree_v2)(va)
        };
        if rc != 0 {
            if self.is_poisoned() {
                tracing::debug!(rc, va, "cuMemFree(vmm snapshot) failed (context poisoned)");
            } else {
                tracing::warn!(rc, va, "cuMemFree(vmm snapshot) failed");
            }
        }
    }

    fn copy_dtod(&self, dst: u64, src: u64, bytes: u64) -> Result<()> {
        self.memcpy_dtod(dst, src, bytes)
    }

    fn pool_take(&self) -> Vec<(u64, u64)> {
        std::mem::take(&mut *self.slab_pool.lock())
    }

    fn pool_put(&self, mut chunks: Vec<(u64, u64)>) {
        self.slab_pool.lock().append(&mut chunks);
    }

    fn pool_bytes(&self) -> u64 {
        self.slab_pool.lock().iter().map(|&(_, b)| b).sum()
    }

    fn pool_trim(&self, keep_bytes: u64) -> u64 {
        // Collect victims under the lock, release outside it — cuMemRelease
        // re-binds the context and can take ms-class time per chunk.
        let victims = {
            let mut pool = self.slab_pool.lock();
            let mut held: u64 = pool.iter().map(|&(_, b)| b).sum();
            let mut victims = Vec::new();
            while held > keep_bytes {
                let Some((h, b)) = pool.pop() else { break };
                held -= b;
                victims.push((h, b));
            }
            victims
        };
        let mut released = 0u64;
        for &(h, b) in &victims {
            crate::memory::vmm::VmmOps::release(self, h);
            released += b;
        }
        released
    }
}

impl Drop for CudaBackend {
    /// Unload any modules still registered. The primary-context release lives
    /// in `CudaFreer::drop` — it must wait for the last owned `DeviceMem`,
    /// which may outlive the backend. Errors are logged, not propagated.
    fn drop(&mut self) {
        // SAFETY: handles from cuModuleLoadData; the map is drained so each
        // is unloaded exactly once. Pooled slab chunks (PLOW_SLAB_KEEP) are
        // released here — the pool's whole point is to outlive engines, so
        // the backend is its terminal owner.
        let poisoned = self.is_poisoned();
        unsafe {
            (self.api.cuCtxSetCurrent)(self.ctx as CUcontext);
            for (id, raw) in self.modules.lock().drain() {
                let rc = (self.api.cuModuleUnload)(raw as CUmodule);
                if rc != 0 {
                    if poisoned {
                        tracing::debug!(
                            id,
                            rc,
                            "cuModuleUnload at backend drop (context poisoned)"
                        );
                    } else {
                        tracing::warn!(id, rc, "cuModuleUnload at backend drop");
                    }
                }
            }
            for (h, bytes) in self.slab_pool.lock().drain(..) {
                let rc = (self.api.cuMemRelease)(h);
                if rc != 0 {
                    if poisoned {
                        tracing::debug!(rc, handle = h, bytes, "cuMemRelease at backend drop");
                    } else {
                        tracing::warn!(rc, handle = h, bytes, "cuMemRelease at backend drop");
                    }
                }
            }
        }
    }
}

impl Backend for CudaBackend {
    fn class(&self) -> ExecutorClass {
        ExecutorClass::SmNv
    }
    fn coherent_host_dma(&self) -> bool {
        self.coherent_host_dma
    }
    fn vendor(&self) -> Option<hwspec::Vendor> {
        Some(hwspec::Vendor::Nvidia)
    }
    fn enumerate(&self) -> Vec<ExecutorTarget> {
        // One executor per SM: the persistent interpreter runs one 256-thread
        // block per SM (8 worker warps), and the compiler partitions packets
        // across exactly this many streams.
        (0..self.sm_count)
            .map(|i| ExecutorTarget {
                class: ExecutorClass::SmNv,
                instance_id: i,
                wave_width: self.warp_size,
                worker_count: 256 / self.warp_size,
                shmem_bytes: self.smem_optin,
                opcode_mask: !0,
            })
            .collect()
    }
    fn alloc(&self, _device: u8, bytes: u64) -> Result<DeviceMem> {
        self.bind()?;
        let mut dptr: CUdeviceptr = 0;
        // SAFETY: out-pointer to a device allocation.
        self.check(
            unsafe { (self.api.cuMemAlloc_v2)(&mut dptr, bytes as usize) },
            "cuMemAlloc",
        )?;
        // Owned: freed on Drop through the freer (which pins ctx + library).
        Ok(DeviceMem::owned(dptr, bytes, self.freer.clone()))
    }
    fn upload(&self, dst: &DeviceMem, off: u64, src: &[u8]) -> Result<()> {
        if off + src.len() as u64 > dst.len {
            return Err(RuntimeError::Device(format!(
                "upload out of range: off {off} + {} > {}",
                src.len(),
                dst.len
            )));
        }
        self.memcpy_htod(dst.base + off, src)
    }
    fn download(&self, src: &DeviceMem, off: u64, dst: &mut [u8]) -> Result<()> {
        if off + dst.len() as u64 > src.len {
            return Err(RuntimeError::Device(format!(
                "download out of range: off {off} + {} > {}",
                dst.len(),
                src.len
            )));
        }
        self.memcpy_dtoh(dst, src.base + off)
    }
    fn module_load(&self, image: &[u8]) -> Result<Module> {
        // Empty image: the placeholder module `ExecutorSet::bringup` asks for
        // before any real cubin exists. Real modules are loaded by the engine.
        if image.is_empty() {
            return Ok(Module { id: 0 });
        }
        self.bind()?;
        let mut m: CUmodule = std::ptr::null_mut();
        // SAFETY: image is a complete cubin/fatbin byte image.
        self.check(
            unsafe { (self.api.cuModuleLoadData)(&mut m, image.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )
        .map_err(|e| {
            // The classic version trap reads as an opaque INVALID_IMAGE: state
            // the driver's ceiling so it diagnoses itself (the sm90a build
            // script's header documents exactly this failure).
            RuntimeError::Device(format!(
                "{e} — this driver supports CUDA {}.{} at most; a cubin built by a \
                 newer nvcc fails here with CUDA_ERROR_INVALID_IMAGE. Rebuild the \
                 cubin with a toolkit the driver accepts (PLOW_NVCC), or upgrade \
                 the driver.",
                self.driver_version / 1000,
                (self.driver_version % 1000) / 10
            ))
        })?;
        let id = self.next_module.fetch_add(1, Ordering::Relaxed);
        self.modules.lock().insert(id, m as usize);
        Ok(Module { id })
    }
    fn launch_persistent(&self, module: &Module, _cfg: LaunchCfg) -> Result<()> {
        // The persistent interpreter is launched per token step through
        // `launch_cooperative` (the harness-proven sequence); the bringup-time
        // placeholder module has nothing to launch.
        if module.id == 0 {
            return Ok(());
        }
        Err(RuntimeError::Device(
            "launch_persistent: use launch_cooperative for real modules".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::is_cuda_fatal;

    #[test]
    fn fatal_classification_matches_the_driver_contract() {
        // Context-poisoning statuses.
        for rc in [4, 214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719] {
            assert!(is_cuda_fatal(rc), "CUresult {rc} must be fatal");
        }
        // Transient / rejection statuses that must NOT kill the engine.
        for rc in [2, 701, 720, 600, 1, 100] {
            assert!(!is_cuda_fatal(rc), "CUresult {rc} must not be fatal");
        }
    }
}
