//! HSA driver-API backend (feature `hsa`).
//!
//! Pure-Rust FFI to `libhsa-runtime64.so` (ROCr). No HIP runtime, no
//! `libamdhip64` — the binary talks directly to the HSA/ROCr layer that sits on
//! `amdkfd`. This mirrors the design in `runtime/amd/hsa_backend.c` but is
//! self-contained in Rust with `dlopen` via `libloading`.
//!
//! ## Dispatch path
//!
//! A kernel launch is: memcpy the kernarg slot, ~16 stores into the AQL ring,
//! one release-store of the packet header, one doorbell store. No driver
//! round-trip, no HIP launch overhead (~200 ns vs ~5-15 µs through HIP).
//!
//! ## Static binary + no link-time GPU dependency
//!
//! Same discipline as [`super::cuda`]: `libhsa-runtime64.so` is `dlopen`ed at
//! runtime. A build with `--features hsa` runs on any box — if the driver is
//! absent, probe fails gracefully and the CPU reference backend takes over.
//!
//! ## `unsafe` convention in this file
//!
//! Roughly 110 `unsafe` blocks here are one thing: **a call through a
//! `dlsym`-resolved ROCr function pointer**. Their safety argument is identical
//! and is made once, here, rather than restated per site:
//!
//! * every pointer is resolved once in [`Driver::load`] against the signature
//!   transcribed from `hsa.h` / `hsa_ext_amd.h`, and a missing symbol fails the
//!   load rather than yielding a null to call;
//! * the handles passed (agent, queue, pool, signal, executable) are opaque
//!   `u64`/pointer values the runtime itself produced and this backend owns for
//!   its lifetime, and every out-parameter is a live local;
//! * ROCr reports failure in the returned `hsa_status_t`, which each call site
//!   checks — none of them rely on the call being infallible.
//!
//! A per-site `// SAFETY:` comment is therefore reserved for the blocks that do
//! something OTHER than an FFI call — raw-pointer arithmetic into the AQL ring
//! and the kernarg slab (`dispatch`), `write_bytes`/`write_unaligned` into
//! device-visible memory, and the `Send`/`Sync` impls — because those carry an
//! argument that is not shared with the rest of the file.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::device::{
    Backend, DeviceFree, DeviceMem, ExecutorClass, ExecutorTarget, LaunchCfg, Module, PeerMemory,
};
use crate::{DeviceErrorInfo, Result, RuntimeError};

// ─── HSA ABI constants ───────────────────────────────────────────────────────

const HSA_STATUS_SUCCESS: i32 = 0;

/// The `hsa_status_t` name for `rc` (hsa.h / hsa_ext_amd.h). A static table
/// rather than `hsa_status_string`: fault construction must work even when the
/// runtime itself is wedged, and the enum values are stable ABI.
fn hsa_status_name(rc: i32) -> &'static str {
    match rc {
        0x0 => "HSA_STATUS_SUCCESS",
        0x1 => "HSA_STATUS_INFO_BREAK",
        0x1000 => "HSA_STATUS_ERROR",
        0x1001 => "HSA_STATUS_ERROR_INVALID_ARGUMENT",
        0x1002 => "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION",
        0x1003 => "HSA_STATUS_ERROR_INVALID_ALLOCATION",
        0x1004 => "HSA_STATUS_ERROR_INVALID_AGENT",
        0x1005 => "HSA_STATUS_ERROR_INVALID_REGION",
        0x1006 => "HSA_STATUS_ERROR_INVALID_SIGNAL",
        0x1007 => "HSA_STATUS_ERROR_INVALID_QUEUE",
        0x1008 => "HSA_STATUS_ERROR_OUT_OF_RESOURCES",
        0x1009 => "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT",
        0x100A => "HSA_STATUS_ERROR_RESOURCE_FREE",
        0x100B => "HSA_STATUS_ERROR_NOT_INITIALIZED",
        0x100C => "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW",
        0x100D => "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS",
        0x100E => "HSA_STATUS_ERROR_INVALID_INDEX",
        0x100F => "HSA_STATUS_ERROR_INVALID_ISA",
        0x1010 => "HSA_STATUS_ERROR_INVALID_CODE_OBJECT",
        0x1011 => "HSA_STATUS_ERROR_INVALID_EXECUTABLE",
        0x1012 => "HSA_STATUS_ERROR_FROZEN_EXECUTABLE",
        0x1013 => "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME",
        0x1014 => "HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED",
        0x1015 => "HSA_STATUS_ERROR_VARIABLE_UNDEFINED",
        0x1016 => "HSA_STATUS_ERROR_EXCEPTION",
        0x1017 => "HSA_STATUS_ERROR_INVALID_ISA_NAME",
        0x1018 => "HSA_STATUS_ERROR_INVALID_CODE_SYMBOL",
        0x1019 => "HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL",
        0x101A => "HSA_STATUS_ERROR_INVALID_FILE",
        0x101B => "HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER",
        0x101C => "HSA_STATUS_ERROR_INVALID_CACHE",
        0x101D => "HSA_STATUS_ERROR_INVALID_WAVEFRONT",
        0x101E => "HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP",
        0x101F => "HSA_STATUS_ERROR_INVALID_RUNTIME_STATE",
        0x1020 => "HSA_STATUS_ERROR_FATAL",
        // AMD extension (hsa_ext_amd.h).
        40 => "HSA_STATUS_ERROR_INVALID_MEMORY_POOL",
        41 => "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION",
        42 => "HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION",
        43 => "HSA_STATUS_ERROR_MEMORY_FAULT",
        44 => "HSA_STATUS_CU_MASK_REDUCED",
        45 => "HSA_STATUS_ERROR_OUT_OF_REGISTERS",
        _ => "HSA_STATUS_UNKNOWN",
    }
}

/// Does this `hsa_status_t` mean the agent/queue is permanently poisoned?
/// The AMD-extension trap statuses (a kernel faulted — the queue is dead) plus
/// the core EXCEPTION/FATAL pair. Deliberately NOT here: OUT_OF_RESOURCES
/// (0x1008, transient alloc failure — the CUDA-OOM analogue) and every
/// INVALID_* argument/handle status (a bad call, not a dead device).
fn is_hsa_fatal(rc: i32) -> bool {
    matches!(
        rc,
        41       // MEMORY_APERTURE_VIOLATION
        | 42     // ILLEGAL_INSTRUCTION
        | 43     // MEMORY_FAULT
        | 0x1016 // EXCEPTION — a trapped kernel
        | 0x1020 // FATAL
    )
}

/// Build the typed fault for a failed ROCr call. Free-standing for bring-up
/// sites (`new()`, `find_pool`, the upload ring) that have no backend yet;
/// [`HsaBackend::fault`] wraps it with poison marking.
fn hsa_fault(rc: i32, what: &str) -> RuntimeError {
    RuntimeError::DeviceFault {
        info: DeviceErrorInfo {
            operation: what.to_string(),
            code: rc,
            name: hsa_status_name(rc).to_string(),
            fatal: is_hsa_fatal(rc),
        },
    }
}

// hsa_device_type_t
const HSA_DEVICE_TYPE_CPU: u32 = 0;
const HSA_DEVICE_TYPE_GPU: u32 = 1;

// hsa_agent_info_t
const HSA_AGENT_INFO_NAME: u32 = 0;
const HSA_AGENT_INFO_DEVICE: u32 = 17;
// AMD extension: CU count
// hsa_amd_agent_info_t — CHIP_ID=0xA000, CACHELINE_SIZE=0xA001,
// COMPUTE_UNIT_COUNT=0xA002. This was 0xA000, so `cu_count` held the PCI CHIP
// ID: 30115 on gfx950 instead of 256. The engine sizes its cooperative grid
// from `sm_count()`, so a persistent launch would have asked for 30115 blocks.
const HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT: u32 = 0xA002;

// hsa_region_segment_t
const HSA_REGION_SEGMENT_GROUP: u32 = 2;

// hsa_region_info_t
const HSA_REGION_INFO_SEGMENT: u32 = 0;
const HSA_REGION_INFO_SIZE: u32 = 2;

// hsa_amd_segment_t
const HSA_AMD_SEGMENT_GLOBAL: u32 = 0;

// hsa_amd_memory_pool_info_t
const HSA_AMD_MEMORY_POOL_INFO_SEGMENT: u32 = 0;
const HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS: u32 = 1;
/// Recommended physical-allocation granule for `hsa_amd_vmem_handle_create`
/// (the ROCr analogue of `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED`). The *required*
/// granule is `..._RUNTIME_ALLOC_GRANULE = 6`; the recommended one is what
/// keeps internal fragmentation down, so it is what [`VmmOps::granularity`]
/// reports — same choice the CUDA path makes.
const HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE: u32 = 18;

// hsa_amd_memory_type_t — NONE = 0, PINNED = 1. Device-local physical backing.
const HSA_AMD_MEMORY_TYPE_PINNED: u32 = 1;

// hsa_access_permission_t
const HSA_ACCESS_PERMISSION_RW: u32 = 3;

// hsa_amd_memory_pool_global_flag_t
const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED: u32 = 1 << 1;
const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED: u32 = 1 << 0;
const HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT: u32 = 1 << 2;

// hsa_queue_type_t
const HSA_QUEUE_TYPE_SINGLE: u32 = 1;

// hsa_packet_type_t
const HSA_PACKET_TYPE_KERNEL_DISPATCH: u32 = 2;

// hsa_packet_header bit positions
const HSA_PACKET_HEADER_TYPE: u32 = 0;
const HSA_PACKET_HEADER_BARRIER: u32 = 8;
const HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE: u32 = 9;
const HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE: u32 = 11;

// hsa_kernel_dispatch_packet_setup bit positions
const HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS: u32 = 0;

// hsa_fence_scope_t
const HSA_FENCE_SCOPE_AGENT: u32 = 1;

// hsa_signal_condition_t
// hsa_signal_condition_t — values from hsa.h (EQ=0, NE=1, LT=2, GTE=3).
// LT was 0 here, which is EQ: every completion wait therefore meant "block
// until the signal equals 1", while a completion signal counts DOWN to 0. No
// wait could ever be satisfied, so every copy through `upload`/`download`
// hung forever. Nothing caught it because plowrt has no AMD engine yet, so
// these paths had never run on hardware.
const HSA_SIGNAL_CONDITION_LT: u32 = 2;

// hsa_wait_state_t
// hsa_wait_state_t — BLOCKED=0, ACTIVE=1. This was 1, i.e. the opposite of its
// name: every wait busy-spun a core instead of yielding it.
const HSA_WAIT_STATE_BLOCKED: u32 = 0;

// hsa_profile_t
const HSA_PROFILE_FULL: u32 = 1;

// hsa_default_float_rounding_mode_t
const HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT: u32 = 1;

// hsa_executable_symbol_info_t
const HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS: u32 = 21;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: u32 = 22;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: u32 = 11;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: u32 = 13;
const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: u32 = 14;

// Our AQL queue and kernarg ring sizing (match hsa_backend.c).
const QUEUE_SIZE: u32 = 1024;
const KARG_SLOT: usize = 512;

// ─── HSA ABI types ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaAgent {
    handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaMemoryPool {
    handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HsaSignal {
    handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaRegion {
    handle: u64,
}

/// `hsa_amd_vmem_alloc_handle_t` — an opaque physical allocation, the ROCr
/// counterpart of `CUmemGenericAllocationHandle`.
#[repr(C)]
#[derive(Clone, Copy)]
struct HsaVmemHandle {
    handle: u64,
}

/// `hsa_amd_memory_access_desc_t`. `{enum, hsa_agent_t}` — `repr(C)` pads the
/// 4-byte enum to the agent's 8-byte alignment, matching the C layout.
#[repr(C)]
#[derive(Clone, Copy)]
struct HsaAmdMemoryAccessDesc {
    permissions: u32,
    agent_handle: HsaAgent,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaCodeObjectReader {
    handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaExecutable {
    handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaExecutableSymbol {
    handle: u64,
}

/// AQL queue header (first fields of hsa_queue_t the user reads/writes).
#[repr(C)]
struct HsaQueue {
    queue_type: u32,
    features: u32,
    base_address: *mut c_void,
    doorbell_signal: HsaSignal,
    size: u32,
    // ... more fields we don't need
}

/// hsa_kernel_dispatch_packet_t — 64 bytes, naturally aligned.
#[repr(C, align(64))]
struct HsaDispatchPacket {
    header: u16,
    setup: u16,
    workgroup_size_x: u16,
    workgroup_size_y: u16,
    workgroup_size_z: u16,
    _reserved0: u16,
    grid_size_x: u32,
    grid_size_y: u32,
    grid_size_z: u32,
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: u64,
    _reserved2: u64,
    completion_signal: HsaSignal,
}

// ─── Driver function-pointer table ──────────────────────────────────────────

type HsaStatus = i32;
type AgentCb = unsafe extern "C" fn(HsaAgent, *mut c_void) -> HsaStatus;
type PoolCb = unsafe extern "C" fn(HsaMemoryPool, *mut c_void) -> HsaStatus;
type RegionCb = unsafe extern "C" fn(HsaRegion, *mut c_void) -> HsaStatus;

/// ROCr's virtual-memory API (`hsa_amd_vmem_*`), resolved as a group because it
/// is used as a group: it is what backs [`crate::memory::vmm::VmmOps`], and it
/// arrived in ROCm 5.7. `None` on an older runtime — every other path still
/// works, the VMM-backed KV pool just refuses to come up.
struct VmemFns {
    address_reserve_align:
        unsafe extern "C" fn(*mut *mut c_void, usize, u64, u64, u64) -> HsaStatus,
    address_free: unsafe extern "C" fn(*mut c_void, usize) -> HsaStatus,
    handle_create:
        unsafe extern "C" fn(HsaMemoryPool, usize, u32, u64, *mut HsaVmemHandle) -> HsaStatus,
    handle_release: unsafe extern "C" fn(HsaVmemHandle) -> HsaStatus,
    map: unsafe extern "C" fn(*mut c_void, usize, usize, HsaVmemHandle, u64) -> HsaStatus,
    unmap: unsafe extern "C" fn(*mut c_void, usize) -> HsaStatus,
    set_access:
        unsafe extern "C" fn(*mut c_void, usize, *const HsaAmdMemoryAccessDesc, usize) -> HsaStatus,
}

macro_rules! hsa_fns {
    ($($name:ident : $sig:ty),* $(,)?) => {
        struct HsaDriver {
            #[allow(dead_code)]
            lib: libloading::Library,
            $($name: $sig,)*
        }
    }
}

hsa_fns! {
    hsa_init: unsafe extern "C" fn() -> HsaStatus,
    hsa_system_get_info: unsafe extern "C" fn(u32, *mut c_void) -> HsaStatus,
    hsa_shut_down: unsafe extern "C" fn() -> HsaStatus,
    hsa_iterate_agents: unsafe extern "C" fn(AgentCb, *mut c_void) -> HsaStatus,
    hsa_agent_get_info: unsafe extern "C" fn(HsaAgent, u32, *mut c_void) -> HsaStatus,
    hsa_agent_iterate_regions: unsafe extern "C" fn(HsaAgent, RegionCb, *mut c_void) -> HsaStatus,
    hsa_region_get_info: unsafe extern "C" fn(HsaRegion, u32, *mut c_void) -> HsaStatus,
    hsa_amd_agent_iterate_memory_pools: unsafe extern "C" fn(HsaAgent, PoolCb, *mut c_void) -> HsaStatus,
    hsa_amd_memory_pool_get_info: unsafe extern "C" fn(HsaMemoryPool, u32, *mut c_void) -> HsaStatus,
    hsa_amd_memory_pool_allocate: unsafe extern "C" fn(HsaMemoryPool, usize, u32, *mut *mut c_void) -> HsaStatus,
    hsa_amd_memory_pool_free: unsafe extern "C" fn(*mut c_void) -> HsaStatus,
    hsa_amd_agents_allow_access: unsafe extern "C" fn(u32, *const HsaAgent, *const u32, *mut c_void) -> HsaStatus,
    hsa_amd_memory_lock: unsafe extern "C" fn(*mut c_void, usize, *const HsaAgent, u32, *mut *mut c_void) -> HsaStatus,
    hsa_amd_memory_unlock: unsafe extern "C" fn(*mut c_void) -> HsaStatus,
    hsa_amd_memory_async_copy: unsafe extern "C" fn(*mut c_void, HsaAgent, *const c_void, HsaAgent, usize, u32, *const HsaSignal, HsaSignal) -> HsaStatus,
    hsa_queue_create: unsafe extern "C" fn(HsaAgent, u32, u32, *const c_void, *const c_void, u32, u32, *mut *mut HsaQueue) -> HsaStatus,
    hsa_queue_destroy: unsafe extern "C" fn(*mut HsaQueue) -> HsaStatus,
    hsa_queue_add_write_index_screlease: unsafe extern "C" fn(*mut HsaQueue, u64) -> u64,
    hsa_queue_load_read_index_scacquire: unsafe extern "C" fn(*const HsaQueue) -> u64,
    hsa_signal_create: unsafe extern "C" fn(i64, u32, *const HsaAgent, *mut HsaSignal) -> HsaStatus,
    hsa_signal_destroy: unsafe extern "C" fn(HsaSignal) -> HsaStatus,
    hsa_signal_store_screlease: unsafe extern "C" fn(HsaSignal, i64),
    hsa_signal_wait_scacquire: unsafe extern "C" fn(HsaSignal, u32, i64, u64, u32) -> i64,
    hsa_signal_add_screlease: unsafe extern "C" fn(HsaSignal, i64),
    hsa_code_object_reader_create_from_memory: unsafe extern "C" fn(*const c_void, usize, *mut HsaCodeObjectReader) -> HsaStatus,
    hsa_code_object_reader_destroy: unsafe extern "C" fn(HsaCodeObjectReader) -> HsaStatus,
    hsa_executable_create_alt: unsafe extern "C" fn(u32, u32, *const c_void, *mut HsaExecutable) -> HsaStatus,
    hsa_executable_load_agent_code_object: unsafe extern "C" fn(HsaExecutable, HsaAgent, HsaCodeObjectReader, *const c_void, *mut c_void) -> HsaStatus,
    hsa_executable_freeze: unsafe extern "C" fn(HsaExecutable, *const c_void) -> HsaStatus,
    hsa_executable_destroy: unsafe extern "C" fn(HsaExecutable) -> HsaStatus,
    hsa_executable_get_symbol_by_name: unsafe extern "C" fn(HsaExecutable, *const u8, *const HsaAgent, *mut HsaExecutableSymbol) -> HsaStatus,
    hsa_executable_symbol_get_info: unsafe extern "C" fn(HsaExecutableSymbol, u32, *mut c_void) -> HsaStatus,
    // Optional group — see `VmemFns`. Declared here so it lives in the same
    // table; resolved by `open_vmem`, which tolerates absence.
    vmem: Option<VmemFns>,
}

impl HsaDriver {
    fn open() -> Result<Self> {
        let lib = unsafe {
            libloading::Library::new("libhsa-runtime64.so")
                .or_else(|_| libloading::Library::new("libhsa-runtime64.so.1"))
        }
        .map_err(|e| RuntimeError::Device(format!("dlopen libhsa-runtime64: {e}")))?;

        macro_rules! resolve {
            ($lib:expr, $name:expr) => {
                *unsafe { $lib.get($name) }.map_err(|e| {
                    RuntimeError::Device(format!(
                        "resolve {}: {e}",
                        std::str::from_utf8($name).unwrap_or("?")
                    ))
                })?
            };
        }

        let drv = HsaDriver {
            hsa_init: resolve!(lib, b"hsa_init\0"),
            hsa_system_get_info: resolve!(lib, b"hsa_system_get_info\0"),
            hsa_shut_down: resolve!(lib, b"hsa_shut_down\0"),
            hsa_iterate_agents: resolve!(lib, b"hsa_iterate_agents\0"),
            hsa_agent_get_info: resolve!(lib, b"hsa_agent_get_info\0"),
            hsa_agent_iterate_regions: resolve!(lib, b"hsa_agent_iterate_regions\0"),
            hsa_region_get_info: resolve!(lib, b"hsa_region_get_info\0"),
            hsa_amd_agent_iterate_memory_pools: resolve!(
                lib,
                b"hsa_amd_agent_iterate_memory_pools\0"
            ),
            hsa_amd_memory_pool_get_info: resolve!(lib, b"hsa_amd_memory_pool_get_info\0"),
            hsa_amd_memory_pool_allocate: resolve!(lib, b"hsa_amd_memory_pool_allocate\0"),
            hsa_amd_memory_pool_free: resolve!(lib, b"hsa_amd_memory_pool_free\0"),
            hsa_amd_agents_allow_access: resolve!(lib, b"hsa_amd_agents_allow_access\0"),
            hsa_amd_memory_lock: resolve!(lib, b"hsa_amd_memory_lock\0"),
            hsa_amd_memory_unlock: resolve!(lib, b"hsa_amd_memory_unlock\0"),
            hsa_amd_memory_async_copy: resolve!(lib, b"hsa_amd_memory_async_copy\0"),
            hsa_queue_create: resolve!(lib, b"hsa_queue_create\0"),
            hsa_queue_destroy: resolve!(lib, b"hsa_queue_destroy\0"),
            hsa_queue_add_write_index_screlease: resolve!(
                lib,
                b"hsa_queue_add_write_index_screlease\0"
            ),
            hsa_queue_load_read_index_scacquire: resolve!(
                lib,
                b"hsa_queue_load_read_index_scacquire\0"
            ),
            hsa_signal_create: resolve!(lib, b"hsa_signal_create\0"),
            hsa_signal_destroy: resolve!(lib, b"hsa_signal_destroy\0"),
            hsa_signal_store_screlease: resolve!(lib, b"hsa_signal_store_screlease\0"),
            hsa_signal_wait_scacquire: resolve!(lib, b"hsa_signal_wait_scacquire\0"),
            hsa_signal_add_screlease: resolve!(lib, b"hsa_signal_add_screlease\0"),
            hsa_code_object_reader_create_from_memory: resolve!(
                lib,
                b"hsa_code_object_reader_create_from_memory\0"
            ),
            hsa_code_object_reader_destroy: resolve!(lib, b"hsa_code_object_reader_destroy\0"),
            hsa_executable_create_alt: resolve!(lib, b"hsa_executable_create_alt\0"),
            hsa_executable_load_agent_code_object: resolve!(
                lib,
                b"hsa_executable_load_agent_code_object\0"
            ),
            hsa_executable_freeze: resolve!(lib, b"hsa_executable_freeze\0"),
            hsa_executable_destroy: resolve!(lib, b"hsa_executable_destroy\0"),
            hsa_executable_get_symbol_by_name: resolve!(
                lib,
                b"hsa_executable_get_symbol_by_name\0"
            ),
            hsa_executable_symbol_get_info: resolve!(lib, b"hsa_executable_symbol_get_info\0"),
            vmem: Self::open_vmem(&lib),
            lib,
        };
        Ok(drv)
    }

    /// Resolve the `hsa_amd_vmem_*` group, or `None` if any member is missing.
    /// All-or-nothing on purpose: a half-resolved VMM surface would fail deep
    /// inside the pool instead of at the one place that decides to use it.
    fn open_vmem(lib: &libloading::Library) -> Option<VmemFns> {
        macro_rules! sym {
            ($name:expr) => {
                *unsafe { lib.get($name) }.ok()?
            };
        }
        Some(VmemFns {
            address_reserve_align: sym!(b"hsa_amd_vmem_address_reserve_align\0"),
            address_free: sym!(b"hsa_amd_vmem_address_free\0"),
            handle_create: sym!(b"hsa_amd_vmem_handle_create\0"),
            handle_release: sym!(b"hsa_amd_vmem_handle_release\0"),
            map: sym!(b"hsa_amd_vmem_map\0"),
            unmap: sym!(b"hsa_amd_vmem_unmap\0"),
            set_access: sym!(b"hsa_amd_vmem_set_access\0"),
        })
    }
}

/// Internal resolved kernel metadata (mirrors `plow_hsa_kernel` in hsa_backend.h).
///
/// `Copy` because [`crate::exec::device_api::EngineDevice::Function`] requires
/// it: the engine copies a function handle out of `&mut self` before a launch
/// so the launch does not hold a borrow of the struct that owns it. Four PODs,
/// so this is free.
#[derive(Clone, Copy)]
pub struct HsaKernel {
    kernel_object: u64,
    kernarg_size: u32,
    group_segment_size: u32,
    private_segment_size: u32,
}

impl HsaKernel {
    /// The object's whole kernarg segment: the explicit args PLUS the 256-byte
    /// COv5 implicit tail. A caller that knows its own explicit args' size can
    /// use this to reject an object built against a different `PlowProgram`,
    /// which otherwise runs and faults — the implicit block lands at the
    /// caller's `args_size`, not the object's.
    pub fn kernarg_size(&self) -> u32 {
        self.kernarg_size
    }

    /// Per-lane scratch bytes baked into the code object.
    pub fn private_segment_size(&self) -> u32 {
        self.private_segment_size
    }
}

// ─── Trampoline-based discovery ──────────────────────────────────────────────
//
// HSA's iterate APIs take `extern "C" fn(Item, *mut c_void) -> hsa_status_t`.
// We pack both the driver fn-ptr and our accumulator into the userdata.

struct AgentAccum {
    get_info: unsafe extern "C" fn(HsaAgent, u32, *mut c_void) -> HsaStatus,
    gpus: Vec<HsaAgent>,
    cpu: Option<HsaAgent>,
}

unsafe extern "C" fn agent_trampoline(agent: HsaAgent, data: *mut c_void) -> HsaStatus {
    let acc = &mut *(data as *mut AgentAccum);
    let mut dtype: u32 = 0;
    let rc = (acc.get_info)(
        agent,
        HSA_AGENT_INFO_DEVICE,
        &mut dtype as *mut u32 as *mut c_void,
    );
    if rc != HSA_STATUS_SUCCESS {
        return HSA_STATUS_SUCCESS; // skip
    }
    if dtype == HSA_DEVICE_TYPE_GPU {
        acc.gpus.push(agent);
    } else if dtype == HSA_DEVICE_TYPE_CPU && acc.cpu.is_none() {
        acc.cpu = Some(agent);
    }
    HSA_STATUS_SUCCESS
}

struct PoolAccum {
    get_pool_info: unsafe extern "C" fn(HsaMemoryPool, u32, *mut c_void) -> HsaStatus,
    want_flag: u32,
    result: Option<HsaMemoryPool>,
}

unsafe extern "C" fn pool_trampoline(pool: HsaMemoryPool, data: *mut c_void) -> HsaStatus {
    let acc = &mut *(data as *mut PoolAccum);
    if acc.result.is_some() {
        return HSA_STATUS_SUCCESS;
    }
    let mut seg: u32 = 0;
    if (acc.get_pool_info)(
        pool,
        HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
        &mut seg as *mut _ as *mut c_void,
    ) != HSA_STATUS_SUCCESS
    {
        return HSA_STATUS_SUCCESS;
    }
    if seg != HSA_AMD_SEGMENT_GLOBAL {
        return HSA_STATUS_SUCCESS;
    }
    let mut flags: u32 = 0;
    if (acc.get_pool_info)(
        pool,
        HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
        &mut flags as *mut _ as *mut c_void,
    ) != HSA_STATUS_SUCCESS
    {
        return HSA_STATUS_SUCCESS;
    }
    if flags & acc.want_flag != 0 {
        acc.result = Some(pool);
    }
    HSA_STATUS_SUCCESS
}

struct RegionAccum {
    get_region_info: unsafe extern "C" fn(HsaRegion, u32, *mut c_void) -> HsaStatus,
    lds_bytes: u32,
}

unsafe extern "C" fn region_trampoline(region: HsaRegion, data: *mut c_void) -> HsaStatus {
    let acc = &mut *(data as *mut RegionAccum);
    let mut seg: u32 = 0;
    if (acc.get_region_info)(
        region,
        HSA_REGION_INFO_SEGMENT,
        &mut seg as *mut _ as *mut c_void,
    ) != HSA_STATUS_SUCCESS
    {
        return HSA_STATUS_SUCCESS;
    }
    if seg != HSA_REGION_SEGMENT_GROUP {
        return HSA_STATUS_SUCCESS;
    }
    let mut sz: usize = 0;
    if (acc.get_region_info)(
        region,
        HSA_REGION_INFO_SIZE,
        &mut sz as *mut _ as *mut c_void,
    ) == HSA_STATUS_SUCCESS
    {
        acc.lds_bytes = sz as u32;
    }
    HSA_STATUS_SUCCESS
}

// ─── Backend implementation ──────────────────────────────────────────────────

/// Shared driver state: `Arc`-held so `HsaFree` (called from `Drop`) can still
/// reach `hsa_amd_memory_pool_free` after the backend is gone.
struct SharedDriver {
    drv: HsaDriver,
}

// SAFETY: all HSA APIs we call are thread-safe (the runtime is process-global).
unsafe impl Send for SharedDriver {}
unsafe impl Sync for SharedDriver {}

/// Device-free implementation for HSA allocations.
struct HsaFree {
    shared: Arc<SharedDriver>,
}

impl DeviceFree for HsaFree {
    fn free(&self, base: u64, _len: u64) {
        // SAFETY: `base` was returned by hsa_amd_memory_pool_allocate and cast
        // to u64 at alloc time; we cast it back.
        unsafe {
            (self.shared.drv.hsa_amd_memory_pool_free)(base as *mut c_void);
        }
    }
}

pub struct HsaBackend {
    shared: Arc<SharedDriver>,
    pub device_ordinal: u8,
    /// ROCr's HSA interface version (major, minor); (0, 0) when the query
    /// failed. Surfaced in the `module_load` error.
    rocr_version: (u16, u16),
    agent: HsaAgent,
    /// EVERY visible GPU agent, in ordinal order — not just this backend's.
    ///
    /// `hsa_amd_agents_allow_access` REPLACES a buffer's allowed-agent list, so
    /// a peer buffer has to name all agents in ONE call; naming them one at a
    /// time silently leaves only the last GPU mapped (the footgun
    /// `plow_hsa_alloc_host` documents in `runtime/amd/hsa_backend.c`). Keeping
    /// the whole table on every backend is what makes that single call possible
    /// without a second enumeration pass, and it is what maps a peer *ordinal*
    /// to the agent an SDMA copy must name.
    agents: Vec<HsaAgent>,
    cpu_agent: HsaAgent,
    vram_pool: HsaMemoryPool,
    fine_pool: HsaMemoryPool,
    kernarg_pool: HsaMemoryPool,
    queue: *mut HsaQueue,
    done_signal: HsaSignal,
    karg_ring: *mut u8,
    device_name: String,
    cu_count: u32,
    lds_bytes: u32,
    /// 64 for CDNA (gfx9xx, gfx8xx), 32 for RDNA (gfx10xx, gfx11xx).
    wave_width: u32,
    /// Monotonically increasing module ID.
    next_module_id: AtomicU64,
    /// Reusable zero source + completion signal for [`PeerMemory::zero_peer`].
    /// See `zero_peer` for why the per-token path cannot afford to build these
    /// per call.
    zero_stage: parking_lot::Mutex<Option<ZeroStage>>,
    /// Did a peer allocation get the CPU agent onto its allow-list? See
    /// [`PeerMemory::peer_host_writable`].
    peer_host_writable: std::sync::atomic::AtomicBool,
    /// Cached fill source for [`HsaBackend::memset_d8`]. HSA has no
    /// queue-ordered fill, so a memset is a copy, and a copy needs a source.
    fill_stage: parking_lot::Mutex<Option<FillStage>>,
    /// Physical slab chunks kept across loads (`PLOW_SLAB_KEEP` /
    /// `VmmOps::pool_put`/`pool_take`) — the ROCr counterpart of
    /// `CudaBackend::slab_pool`. Device-local by construction (one backend per
    /// agent, chunks created against this agent's `vram_pool`).
    slab_pool: parking_lot::Mutex<Vec<(u64, u64)>>,
    /// Set once, on the first fatal ROCr status ([`is_hsa_fatal`]): the fault
    /// that killed the agent/queue. When set, [`HsaBackend::guard`]
    /// short-circuits every driver-touching entry point with a clone BEFORE
    /// touching the runtime — the CUDA `bind()` gate's counterpart, and here
    /// it also keeps [`HsaBackend::synchronize`]'s spin loop from hanging on
    /// a queue that will never advance. Teardown paths (Drop impls,
    /// `DeviceFree`) don't guard and still attempt their calls.
    poisoned: std::sync::OnceLock<DeviceErrorInfo>,
}

/// Preallocated fill source: `len` bytes of fine-grained memory already set to
/// `value`, reused until either changes.
struct FillStage {
    ptr: *mut c_void,
    len: usize,
    value: u8,
}

/// Preallocated zero source for counter resets: fine-grained (already
/// agent-visible, so no `hsa_amd_memory_lock` per call) plus one signal reused
/// across resets.
struct ZeroStage {
    ptr: *mut c_void,
    len: usize,
    sig: HsaSignal,
}

// SAFETY: all mutable state is either atomic or behind the AQL queue's own
// memory-order protocol. The HSA runtime is thread-safe. The `dispatch` method
// writes to the kernarg ring indexed by the queue's write-index (atomic), and
// plowrt's engine-thread model guarantees at most one dispatch in flight per
// device (the engine thread serialises ticks), so no concurrent ring writes.
unsafe impl Send for HsaBackend {}
unsafe impl Sync for HsaBackend {}

impl HsaBackend {
    /// Probe and initialise the HSA backend for `device_ordinal`-th GPU agent.
    pub fn new(device_ordinal: u8) -> Result<Self> {
        let drv = HsaDriver::open()?;

        // hsa_init
        let rc = unsafe { (drv.hsa_init)() };
        if rc != HSA_STATUS_SUCCESS {
            return Err(hsa_fault(rc, "hsa_init"));
        }

        // ROCr's HSA interface version (HSA_SYSTEM_INFO_VERSION_MAJOR/MINOR,
        // both uint16_t). Logged and kept for the module_load error path: a
        // code object built for an ISA this ROCr does not know fails there
        // with the opaque HSA_STATUS_ERROR_INVALID_ISA, and the version is
        // the fact that explains it. A query failure is not fatal — 0.0 just
        // means "unknown" in the message.
        let rocr_version = unsafe {
            let mut major: u16 = 0;
            let mut minor: u16 = 0;
            let a = (drv.hsa_system_get_info)(0, &mut major as *mut u16 as *mut c_void);
            let b = (drv.hsa_system_get_info)(1, &mut minor as *mut u16 as *mut c_void);
            if a == HSA_STATUS_SUCCESS && b == HSA_STATUS_SUCCESS {
                (major, minor)
            } else {
                (0, 0)
            }
        };
        tracing::info!(
            rocr = %format_args!("{}.{}", rocr_version.0, rocr_version.1),
            "HSA runtime initialised"
        );

        // Discover agents.
        let mut acc = AgentAccum {
            get_info: drv.hsa_agent_get_info,
            gpus: Vec::new(),
            cpu: None,
        };
        let rc = unsafe {
            (drv.hsa_iterate_agents)(agent_trampoline, &mut acc as *mut _ as *mut c_void)
        };
        if rc != HSA_STATUS_SUCCESS {
            return Err(hsa_fault(rc, "hsa_iterate_agents"));
        }
        if acc.gpus.is_empty() {
            return Err(RuntimeError::Device("no GPU agents found".into()));
        }
        let cpu_agent = acc
            .cpu
            .ok_or_else(|| RuntimeError::Device("no CPU agent found".into()))?;
        if (device_ordinal as usize) >= acc.gpus.len() {
            return Err(RuntimeError::Device(format!(
                "device ordinal {} >= {} GPU agents",
                device_ordinal,
                acc.gpus.len()
            )));
        }
        let agent = acc.gpus[device_ordinal as usize];
        let agents = acc.gpus.clone();

        // Query device name.
        let mut name_buf = [0u8; 64];
        let rc = unsafe {
            (drv.hsa_agent_get_info)(
                agent,
                HSA_AGENT_INFO_NAME,
                name_buf.as_mut_ptr() as *mut c_void,
            )
        };
        let device_name = if rc == HSA_STATUS_SUCCESS {
            let end = name_buf.iter().position(|&b| b == 0).unwrap_or(64);
            String::from_utf8_lossy(&name_buf[..end]).to_string()
        } else {
            format!("amd_gpu_{}", device_ordinal)
        };

        // Query CU count.
        let mut cu_count: u32 = 0;
        let _ = unsafe {
            (drv.hsa_agent_get_info)(
                agent,
                HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                &mut cu_count as *mut _ as *mut c_void,
            )
        };

        // Query LDS size from GROUP region.
        let mut reg_acc = RegionAccum {
            get_region_info: drv.hsa_region_get_info,
            lds_bytes: 0,
        };
        let _ = unsafe {
            (drv.hsa_agent_iterate_regions)(
                agent,
                region_trampoline,
                &mut reg_acc as *mut _ as *mut c_void,
            )
        };
        // `hsa_agent_iterate_regions` does not enumerate a GROUP region on
        // ROCm 7.2.4/gfx950 — regions are legacy, superseded by memory pools,
        // and the group segment is not exposed as one. Measured live: the
        // callback above never fires for SEGMENT_GROUP and `lds_bytes` stays 0,
        // which would size the interpreter's LDS budget to nothing and make
        // `set_max_dynamic_smem` reject every request.
        //
        // 64 KiB was wrong for gfx950 and would have rejected legitimate
        // requests. MEASURED (`tests/hsa_engine.rs`): every production object —
        // interp_prefill, interp_prefill_mla, interp_decode — declares
        // **147,464 B (144 KiB)** of STATIC group segment, and dispatches and
        // retires on this hardware at 512 threads. So a gfx950 workgroup may
        // allocate at least 144 KiB, and the CU's 160 KiB is the real bound.
        //
        // (The 64 KiB figure is a CDNA3-era per-workgroup limit. It also got
        // conflated with the CUDA engine's `SMEM_PF = 85,248 B`, which is a
        // *dynamic* budget opted into with `cuFuncSetAttribute` and has no
        // bearing here: the AMD interpreter's arena is `__shared__ plow_smem sm`,
        // static, and every dispatch passes `dynamic_lds = 0`.)
        const CDNA_WORKGROUP_LDS_MAX: u32 = 160 * 1024;
        let lds_bytes = if reg_acc.lds_bytes > 0 {
            reg_acc.lds_bytes
        } else {
            CDNA_WORKGROUP_LDS_MAX
        };

        // Find coarse-grained VRAM pool on this GPU.
        let vram_pool =
            Self::find_pool(&drv, agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)?;
        // Find fine-grained system pool on CPU agent.
        let fine_pool = Self::find_pool(
            &drv,
            cpu_agent,
            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
        )?;
        // Find kernarg pool on CPU agent.
        let kernarg_pool = Self::find_pool(
            &drv,
            cpu_agent,
            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT,
        )?;

        // Create AQL queue.
        let mut queue: *mut HsaQueue = std::ptr::null_mut();
        let rc = unsafe {
            (drv.hsa_queue_create)(
                agent,
                QUEUE_SIZE,
                HSA_QUEUE_TYPE_SINGLE,
                std::ptr::null(),
                std::ptr::null(),
                u32::MAX,
                u32::MAX,
                &mut queue,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            return Err(hsa_fault(rc, "hsa_queue_create"));
        }

        // Create completion signal.
        let mut done_signal = HsaSignal { handle: 0 };
        let rc = unsafe { (drv.hsa_signal_create)(0, 0, std::ptr::null(), &mut done_signal) };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (drv.hsa_queue_destroy)(queue);
            }
            return Err(hsa_fault(rc, "hsa_signal_create"));
        }

        // Allocate kernarg ring.
        let ring_bytes = QUEUE_SIZE as usize * KARG_SLOT;
        let mut karg_ring: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (drv.hsa_amd_memory_pool_allocate)(kernarg_pool, ring_bytes, 0, &mut karg_ring)
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (drv.hsa_signal_destroy)(done_signal);
                (drv.hsa_queue_destroy)(queue);
            }
            return Err(hsa_fault(rc, "kernarg ring alloc"));
        }
        // Allow GPU agent access to the kernarg ring.
        let rc =
            unsafe { (drv.hsa_amd_agents_allow_access)(1, &agent, std::ptr::null(), karg_ring) };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (drv.hsa_amd_memory_pool_free)(karg_ring);
                (drv.hsa_signal_destroy)(done_signal);
                (drv.hsa_queue_destroy)(queue);
            }
            return Err(hsa_fault(rc, "kernarg allow_access"));
        }

        let shared = Arc::new(SharedDriver { drv });

        // CDNA (gfx8xx, gfx9xx) is wave64; RDNA (gfx10xx, gfx11xx) is wave32.
        let wave_width = if device_name.starts_with("gfx9") || device_name.starts_with("gfx8") {
            64
        } else {
            32
        };

        Ok(HsaBackend {
            shared,
            device_ordinal,
            rocr_version,
            agent,
            agents,
            cpu_agent,
            vram_pool,
            fine_pool,
            kernarg_pool,
            queue,
            done_signal,
            karg_ring: karg_ring as *mut u8,
            device_name,
            cu_count,
            lds_bytes,
            wave_width,
            next_module_id: AtomicU64::new(1),
            zero_stage: parking_lot::Mutex::new(None),
            peer_host_writable: std::sync::atomic::AtomicBool::new(false),
            fill_stage: parking_lot::Mutex::new(None),
            slab_pool: parking_lot::Mutex::new(Vec::new()),
            poisoned: std::sync::OnceLock::new(),
        })
    }

    /// Map a non-success `hsa_status_t` to a typed error, classifying it
    /// ([`is_hsa_fatal`]) and poisoning the backend on a fatal status — the
    /// ROCr counterpart of `CudaBackend::check`.
    fn check(&self, rc: i32, what: &str) -> Result<()> {
        if rc == HSA_STATUS_SUCCESS {
            return Ok(());
        }
        Err(self.fault(rc, what))
    }

    /// [`HsaBackend::check`]'s error-building half, for sites that must run
    /// cleanup between the failing call and the `return Err`.
    fn fault(&self, rc: i32, what: &str) -> RuntimeError {
        let err = hsa_fault(rc, what);
        if let Some(info) = err.device_fault() {
            if info.fatal {
                self.mark_poisoned(info);
            }
        }
        err
    }

    /// Has a fatal ROCr status permanently poisoned this agent?
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.get().is_some()
    }

    /// Record the fault that killed the agent. Logs `error!` exactly once —
    /// every later entry point is short-circuited by `guard`, not re-logged.
    fn mark_poisoned(&self, info: &DeviceErrorInfo) {
        if self.poisoned.set(info.clone()).is_ok() {
            tracing::error!(
                error_op = %info.operation,
                error_code = info.code,
                error_name = %info.name,
                device = self.device_ordinal,
                "HSA agent poisoned — rejecting further work on this device"
            );
        }
    }

    /// Short-circuit with the recorded fault when the backend is poisoned —
    /// called at the top of every driver-touching entry point, BEFORE any
    /// runtime call (the counterpart of the CUDA `bind()` gate).
    fn guard(&self) -> Result<()> {
        match self.poisoned.get() {
            Some(info) => Err(RuntimeError::DeviceFault { info: info.clone() }),
            None => Ok(()),
        }
    }

    fn find_pool(drv: &HsaDriver, agent: HsaAgent, want_flag: u32) -> Result<HsaMemoryPool> {
        let mut acc = PoolAccum {
            get_pool_info: drv.hsa_amd_memory_pool_get_info,
            want_flag,
            result: None,
        };
        let rc = unsafe {
            (drv.hsa_amd_agent_iterate_memory_pools)(
                agent,
                pool_trampoline,
                &mut acc as *mut _ as *mut c_void,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            return Err(hsa_fault(rc, "hsa_amd_agent_iterate_memory_pools"));
        }
        acc.result.ok_or_else(|| {
            RuntimeError::Device(format!("no memory pool with flag 0x{:x}", want_flag))
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

impl Drop for HsaBackend {
    fn drop(&mut self) {
        // Pooled slab chunks (PLOW_SLAB_KEEP) are released here — the pool's
        // whole point is to outlive engines, so the backend is its terminal
        // owner (same contract as `CudaBackend::drop`).
        for (h, _) in self.slab_pool.lock().drain(..) {
            crate::memory::vmm::VmmOps::release(self, h);
        }
        if let Some(fs) = self.fill_stage.lock().take() {
            unsafe { (self.shared.drv.hsa_amd_memory_pool_free)(fs.ptr) };
        }
        if let Some(z) = self.zero_stage.lock().take() {
            unsafe {
                (self.shared.drv.hsa_amd_memory_pool_free)(z.ptr);
                (self.shared.drv.hsa_signal_destroy)(z.sig);
            }
        }
        unsafe {
            if !self.karg_ring.is_null() {
                (self.shared.drv.hsa_amd_memory_pool_free)(self.karg_ring as *mut c_void);
            }
            if !self.queue.is_null() {
                (self.shared.drv.hsa_queue_destroy)(self.queue);
            }
            (self.shared.drv.hsa_signal_destroy)(self.done_signal);
            // Note: we do NOT call hsa_shut_down here because other backends or
            // modules may still hold HSA references. The process-exit path is fine.
        }
    }
}

impl Backend for HsaBackend {
    fn class(&self) -> ExecutorClass {
        ExecutorClass::CuAmd
    }

    /// Explicitly the staged default: the gfx950 boxes this backend serves are
    /// discrete cards whose SDMA reads host memory through pinned staging
    /// (`HsaUploadRing`). An MI300A-class APU could answer `true` here once
    /// the direct-from-mmap copy is measured on one — the loaders gate on
    /// this method alone, so that is the only line to change.
    fn coherent_host_dma(&self) -> bool {
        false
    }

    fn vendor(&self) -> Option<hwspec::Vendor> {
        Some(hwspec::Vendor::Amd)
    }

    fn enumerate(&self) -> Vec<ExecutorTarget> {
        (0..self.cu_count)
            .map(|i| ExecutorTarget {
                class: ExecutorClass::CuAmd,
                instance_id: i,
                wave_width: self.wave_width,
                worker_count: 4, // persistent interpreter's wavefront budget
                shmem_bytes: self.lds_bytes,
                opcode_mask: u32::MAX,
            })
            .collect()
    }

    fn alloc(&self, _device: u8, bytes: u64) -> Result<DeviceMem> {
        self.guard()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_pool_allocate)(
                self.vram_pool,
                bytes as usize,
                0,
                &mut ptr,
            )
        };
        self.check(rc, &format!("hsa_amd_memory_pool_allocate({bytes} bytes)"))?;
        let free = Arc::new(HsaFree {
            shared: self.shared.clone(),
        });
        Ok(DeviceMem::owned(ptr as u64, bytes, free))
    }

    fn upload(&self, dst: &DeviceMem, off: u64, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        self.guard()?;
        let dst_ptr = (dst.base + off) as *mut c_void;
        // Pin the source host pages so the SDMA engine can read them.
        let mut pinned: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_lock)(
                src.as_ptr() as *mut c_void,
                src.len(),
                &self.agent,
                1,
                &mut pinned,
            )
        };
        self.check(rc, "hsa_amd_memory_lock (upload)")?;
        // Async copy with a one-shot signal for synchronization.
        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe { (self.shared.drv.hsa_signal_create)(1, 0, std::ptr::null(), &mut sig) };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (self.shared.drv.hsa_amd_memory_unlock)(src.as_ptr() as *mut c_void);
            }
            return Err(self.fault(rc, "hsa_signal_create (upload)"));
        }
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_async_copy)(
                dst_ptr,
                self.agent,
                pinned as *const c_void,
                self.cpu_agent,
                src.len(),
                0,
                std::ptr::null(),
                sig,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (self.shared.drv.hsa_signal_destroy)(sig);
                (self.shared.drv.hsa_amd_memory_unlock)(src.as_ptr() as *mut c_void);
            }
            return Err(self.fault(rc, "hsa_amd_memory_async_copy (H2D)"));
        }
        // Wait for completion.
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
            (self.shared.drv.hsa_signal_destroy)(sig);
            (self.shared.drv.hsa_amd_memory_unlock)(src.as_ptr() as *mut c_void);
        }
        Ok(())
    }

    fn download(&self, src: &DeviceMem, off: u64, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        self.guard()?;
        let src_ptr = (src.base + off) as *const c_void;
        // Pin the destination host pages.
        let mut pinned: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_lock)(
                dst.as_mut_ptr() as *mut c_void,
                dst.len(),
                &self.agent,
                1,
                &mut pinned,
            )
        };
        self.check(rc, "hsa_amd_memory_lock (download)")?;
        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe { (self.shared.drv.hsa_signal_create)(1, 0, std::ptr::null(), &mut sig) };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (self.shared.drv.hsa_amd_memory_unlock)(dst.as_mut_ptr() as *mut c_void);
            }
            return Err(self.fault(rc, "hsa_signal_create (download)"));
        }
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_async_copy)(
                pinned,
                self.cpu_agent,
                src_ptr,
                self.agent,
                dst.len(),
                0,
                std::ptr::null(),
                sig,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (self.shared.drv.hsa_signal_destroy)(sig);
                (self.shared.drv.hsa_amd_memory_unlock)(dst.as_mut_ptr() as *mut c_void);
            }
            return Err(self.fault(rc, "hsa_amd_memory_async_copy (D2H)"));
        }
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
            (self.shared.drv.hsa_signal_destroy)(sig);
            (self.shared.drv.hsa_amd_memory_unlock)(dst.as_mut_ptr() as *mut c_void);
        }
        Ok(())
    }

    fn module_load(&self, image: &[u8]) -> Result<Module> {
        self.guard()?;
        // Create a code-object reader from the in-memory ELF.
        let mut reader = HsaCodeObjectReader { handle: 0 };
        let rc = unsafe {
            (self.shared.drv.hsa_code_object_reader_create_from_memory)(
                image.as_ptr() as *const c_void,
                image.len(),
                &mut reader,
            )
        };
        self.check(rc, "hsa_code_object_reader_create")?;

        // Create executable.
        let mut exe = HsaExecutable { handle: 0 };
        let rc = unsafe {
            (self.shared.drv.hsa_executable_create_alt)(
                HSA_PROFILE_FULL,
                HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                std::ptr::null(),
                &mut exe,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (self.shared.drv.hsa_code_object_reader_destroy)(reader);
            }
            return Err(self.fault(rc, "hsa_executable_create_alt"));
        }

        // Load the code object for this agent.
        let rc = unsafe {
            (self.shared.drv.hsa_executable_load_agent_code_object)(
                exe,
                self.agent,
                reader,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (self.shared.drv.hsa_executable_destroy)(exe);
                (self.shared.drv.hsa_code_object_reader_destroy)(reader);
            }
            // INVALID_ISA here is the version trap: an object whose ISA (or
            // register budget) this ROCr does not accept. Name the runtime
            // version so the error diagnoses itself instead of reading like
            // a corrupt file.
            return Err(self.fault(
                rc,
                &format!(
                    "hsa_executable_load_agent_code_object (raw ELF expected — did you \
                     unbundle?; ROCr HSA {}.{} — an object built for a newer ISA or over \
                     this agent's register budget fails here with \
                     HSA_STATUS_ERROR_INVALID_ISA; rebuild against this ROCm or upgrade it)",
                    self.rocr_version.0, self.rocr_version.1
                ),
            ));
        }

        // Freeze.
        let rc = unsafe { (self.shared.drv.hsa_executable_freeze)(exe, std::ptr::null()) };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (self.shared.drv.hsa_executable_destroy)(exe);
                (self.shared.drv.hsa_code_object_reader_destroy)(reader);
            }
            return Err(self.fault(rc, "hsa_executable_freeze"));
        }

        unsafe {
            (self.shared.drv.hsa_code_object_reader_destroy)(reader);
        }

        // Store the executable handle as the module ID. We pack it as a u64.
        // The executable handle IS a u64.
        let id = exe.handle;
        // Also store as module id for the trait.
        let mid = self.next_module_id.fetch_add(1, Ordering::Relaxed);
        // We need to keep the executable alive. Store it in the Module's id field.
        // Since Module only has a u64 id, we use the exe.handle directly.
        let _ = mid;
        Ok(Module { id })
    }

    fn launch_persistent(&self, module: &Module, cfg: LaunchCfg) -> Result<()> {
        // Resolve the "plow_interp" kernel from the loaded executable.
        let exe = HsaExecutable { handle: module.id };
        let kernel = self.resolve_kernel(exe, "plow_interp")?;

        // Dispatch: grid = cfg.executors workgroups × cfg.workers waves each.
        let wg_x = (cfg.workers * self.wave_width) as u16; // waves × wave_width
        let grid_x = cfg.executors * wg_x as u32;

        self.dispatch(&kernel, grid_x, 1, 1, wg_x, 1, 1, 0, std::ptr::null(), 0)
    }

    fn peer(&self) -> Option<&dyn PeerMemory> {
        Some(self)
    }

    fn alloc_counter_region(&self, count: usize) -> Result<DeviceMem> {
        self.guard()?;
        // Counter region lives in fine-grained system memory (host-visible AND
        // device-accessible) so both SM/CU atomics and host polls hit the same cells.
        let bytes = (count * crate::exec::counters::CELL_STRIDE) as u64;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_pool_allocate)(
                self.fine_pool,
                bytes as usize,
                0,
                &mut ptr,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            return Err(self.fault(
                rc,
                &format!("counter region alloc (fine-grained, {bytes} bytes)"),
            ));
        }
        // Allow GPU agent access.
        let rc = unsafe {
            (self.shared.drv.hsa_amd_agents_allow_access)(1, &self.agent, std::ptr::null(), ptr)
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (self.shared.drv.hsa_amd_memory_pool_free)(ptr);
            }
            return Err(self.fault(rc, "counter region allow_access"));
        }
        // Fine-grained system memory: the host pointer IS the device pointer
        // (unified address space on MI300+). Return as an owned DeviceMem so
        // Drop frees it.
        let free = Arc::new(HsaFree {
            shared: self.shared.clone(),
        });
        Ok(DeviceMem::owned(ptr as u64, bytes, free))
    }
}

// ─── EngineDevice — the serving engine's device surface ─────────────────────

impl crate::exec::device_api::PinnedBuf for HsaPinned {
    fn as_slice(&self) -> &[u8] {
        HsaPinned::as_slice(self)
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        HsaPinned::as_mut_slice(self)
    }
    fn len(&self) -> usize {
        HsaPinned::len(self)
    }
}

impl crate::exec::device_api::EngineDevice for HsaBackend {
    type Stream = HsaStream;
    type Event = HsaEvent;
    type Pinned = HsaPinned;
    type Function = HsaKernel;

    fn device_name(&self) -> &str {
        HsaBackend::device_name(self)
    }

    /// The agent name IS the ISA key on AMD: `hsa_agent_get_info(NAME)` returns
    /// `gfx950`, which is exactly what selects a code object and its kernel
    /// symbols (`plow_interp_gfx950`).
    fn arch(&self) -> String {
        self.device_name.clone()
    }

    fn sm_count(&self) -> u32 {
        HsaBackend::sm_count(self)
    }

    fn alloc(&self, bytes: u64) -> Result<DeviceMem> {
        Backend::alloc(self, 0, bytes)
    }

    fn upload(&self, dst: &DeviceMem, off: u64, src: &[u8]) -> Result<()> {
        Backend::upload(self, dst, off, src)
    }

    fn download(&self, src: &DeviceMem, off: u64, dst: &mut [u8]) -> Result<()> {
        Backend::download(self, src, off, dst)
    }

    fn memcpy_htod(&self, dptr: u64, src: &[u8]) -> Result<()> {
        HsaBackend::memcpy_htod(self, dptr, src)
    }

    fn memcpy_dtod(&self, dst: u64, src: u64, bytes: u64) -> Result<()> {
        HsaBackend::memcpy_dtod(self, dst, src, bytes)
    }
    fn memcpy_dtod_batch(&self, pairs: &[(u64, u64, u64)]) -> Result<()> {
        HsaBackend::memcpy_dtod_batch(self, pairs)
    }

    fn host_alloc_pinned(&self, bytes: usize) -> Result<HsaPinned> {
        HsaBackend::host_alloc_pinned(self, bytes)
    }

    fn memset_d8(&self, dptr: u64, value: u8, n: usize) -> Result<()> {
        HsaBackend::memset_d8(self, dptr, value, n)
    }

    fn memset_d8_async(&self, dptr: u64, value: u8, n: usize, _s: &HsaStream) -> Result<()> {
        // Blocking, not queue-ordered — see the trait note. The AQL queue has
        // no fill packet, so "async" here would be a lie the engine would then
        // build a dependency on.
        HsaBackend::memset_d8(self, dptr, value, n)
    }

    unsafe fn memcpy_htod_async(&self, dptr: u64, src: &[u8], s: &HsaStream) -> Result<()> {
        unsafe { HsaBackend::memcpy_htod_async(self, dptr, src, s) }
    }

    unsafe fn memcpy_dtoh_async(&self, dst: &mut [u8], dptr: u64, s: &HsaStream) -> Result<()> {
        unsafe { HsaBackend::memcpy_dtoh_async(self, dst, dptr, s) }
    }

    fn stream_create(&self) -> Result<HsaStream> {
        HsaBackend::stream_create(self)
    }

    fn stream_synchronize(&self, s: &HsaStream) -> Result<()> {
        HsaBackend::stream_synchronize(self, s)
    }

    fn synchronize(&self) -> Result<()> {
        HsaBackend::synchronize(self)
    }

    fn event_create(&self, timing: bool) -> Result<HsaEvent> {
        HsaBackend::event_create(self, timing)
    }

    fn event_record(&self, e: &HsaEvent, s: &HsaStream) -> Result<()> {
        HsaBackend::event_record(self, e, s)
    }

    fn event_synchronize(&self, e: &HsaEvent) -> Result<()> {
        HsaBackend::event_synchronize(self, e)
    }

    fn event_elapsed_ms(&self, a: &HsaEvent, b: &HsaEvent) -> Result<f32> {
        HsaBackend::event_elapsed_ms(self, a, b)
    }

    fn module_load(&self, image: &[u8]) -> Result<Module> {
        Backend::module_load(self, image)
    }

    fn module_unload(&self, m: &Module) -> Result<()> {
        HsaBackend::module_unload(self, m)
    }

    fn get_function(&self, m: &Module, name: &str) -> Result<HsaKernel> {
        HsaBackend::get_function(self, m, name)
    }

    fn module_global_zero(&self, m: &Module, name: &str, n: usize) -> Result<bool> {
        HsaBackend::module_global_zero(self, m, name, n)
    }

    fn module_global_u32(&self, m: &Module, name: &str) -> Result<Option<u32>> {
        HsaBackend::module_global_u32(self, m, name)
    }

    fn module_global_bytes(
        &self,
        m: &Module,
        name: &str,
        max: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool> {
        HsaBackend::module_global_bytes(self, m, name, max, out)
    }

    fn set_max_dynamic_smem(&self, f: HsaKernel, bytes: u32) -> Result<()> {
        HsaBackend::set_max_dynamic_smem(self, &f, bytes)
    }

    /// Always 1 — and that is a statement about the design, not a stub.
    ///
    /// The persistent interpreter IS one workgroup per CU: it is launched with
    /// `grid == n_cu` and stays resident for the model's life. There is nothing
    /// for a second co-resident block to be. What NVIDIA gets from
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor` is not the number but the
    /// GATE — `cuLaunchCooperativeKernel` refuses a grid that cannot be
    /// co-resident, converting a counter deadlock into a launch error.
    ///
    /// HSA has no such refusal, so on this path co-residency is enforced at
    /// BUILD time instead: the interpreter carries
    /// `__launch_bounds__(PLOW_THREADS, PLOW_WAVES/4)` and
    /// `amdgpu_waves_per_eu`, and without them the register allocator takes 128
    /// AGPRs, arch+acc maxima sum past the limit, and the dispatch is rejected
    /// with `HSA_STATUS_ERROR_INVALID_ISA`. So a register-overcommitted object
    /// fails loudly at dispatch rather than deadlocking — a different mechanism
    /// reaching the same guarantee. If a launch here ever returns INVALID_ISA,
    /// look at the code object's register count, not at this function.
    fn occupancy_blocks_per_sm(&self, _f: HsaKernel, _block: u32, _smem: usize) -> Result<u32> {
        Ok(1)
    }

    fn launch_cooperative(
        &self,
        f: HsaKernel,
        grid: u32,
        block: u32,
        smem_bytes: u32,
        args: &[u8],
        _stream: Option<&HsaStream>,
    ) -> Result<()> {
        // There is one AQL queue and every packet carries the barrier bit, so
        // the queue IS the stream; `Some(stream)` and `None` are the same
        // ordering here. Co-residency is not requested at dispatch (see
        // `occupancy_blocks_per_sm`) — the object was built for it.
        self.launch(f, grid, block, smem_bytes, args)
    }

    fn launch_kernel(
        &self,
        f: HsaKernel,
        grid: u32,
        block: u32,
        smem_bytes: u32,
        args: &[u8],
        _stream: Option<&HsaStream>,
    ) -> Result<()> {
        self.launch(f, grid, block, smem_bytes, args)
    }
}

// ─── Peer memory (multi-GPU) ────────────────────────────────────────────────

impl PeerMemory for HsaBackend {
    fn ordinal(&self) -> u8 {
        self.device_ordinal
    }

    fn peer_agent_count(&self) -> u32 {
        self.agents.len() as u32
    }

    /// Set by the first successful `alloc_peer` that got the CPU agent onto the
    /// allow-list. Reads false before any peer allocation, which is the honest
    /// answer: nothing is host-mapped yet.
    fn peer_host_writable(&self) -> bool {
        self.peer_host_writable.load(Ordering::Relaxed)
    }

    fn alloc_peer(&self, bytes: u64, peers: &[u8]) -> Result<DeviceMem> {
        self.guard()?;
        // Resolve the allow-list BEFORE allocating: a bad ordinal here would
        // otherwise leak a VRAM allocation on the error path.
        if !peers.contains(&self.device_ordinal) {
            return Err(RuntimeError::Device(format!(
                "peer allow-list {peers:?} omits the owner (dev {}) — coarse-grained \
                 VRAM is not even self-accessible until the owner is on the list",
                self.device_ordinal
            )));
        }
        let mut allow: Vec<HsaAgent> = Vec::with_capacity(peers.len());
        for &p in peers {
            allow.push(*self.agents.get(p as usize).ok_or_else(|| {
                RuntimeError::Device(format!(
                    "peer ordinal {p} >= {} GPU agents",
                    self.agents.len()
                ))
            })?);
        }

        // COARSE-grained VRAM, exactly as an ordinary weight/activation
        // allocation: fine-grained would put the partials on the slow
        // host-coherent path, and the whole point is that a peer store runs at
        // the ~58 GB/s XGMI wire rate.
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_pool_allocate)(
                self.vram_pool,
                bytes as usize,
                0,
                &mut ptr,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            return Err(self.fault(
                rc,
                &format!("peer alloc ({bytes} bytes on dev {})", self.device_ordinal),
            ));
        }

        // ONE call naming EVERY agent in the replica. This REPLACES the
        // allowed-agent list, so a loop would leave only the last GPU mapped
        // and every other rank would fault on first peer touch. Naming only
        // the replica's agents is also what keeps two TP4 replicas on one node
        // from being able to reach into each other's partials.
        //
        // The CPU agent is named FIRST, then dropped on failure. Its presence
        // is what makes host stores into this region defined rather than a
        // large-BAR accident, and those stores are the 50×-cheaper counter
        // reset (0.32 µs vs 16.8 µs for 12 KiB, measured). A small-BAR box
        // rejects it, so the GPU-only list is retried and
        // `peer_host_writable()` stays false.
        let mut with_cpu = allow.clone();
        with_cpu.push(self.cpu_agent);
        let rc = unsafe {
            (self.shared.drv.hsa_amd_agents_allow_access)(
                with_cpu.len() as u32,
                with_cpu.as_ptr(),
                std::ptr::null(),
                ptr,
            )
        };
        let rc = if rc == HSA_STATUS_SUCCESS {
            self.peer_host_writable.store(true, Ordering::Relaxed);
            rc
        } else {
            unsafe {
                (self.shared.drv.hsa_amd_agents_allow_access)(
                    allow.len() as u32,
                    allow.as_ptr(),
                    std::ptr::null(),
                    ptr,
                )
            }
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (self.shared.drv.hsa_amd_memory_pool_free)(ptr) };
            return Err(self.fault(
                rc,
                &format!(
                    "hsa_amd_agents_allow_access({} agents, peer buffer)",
                    allow.len()
                ),
            ));
        }

        let free = Arc::new(HsaFree {
            shared: self.shared.clone(),
        });
        Ok(DeviceMem::owned(ptr as u64, bytes, free))
    }

    /// Zero device memory from a cached fine-grained source.
    ///
    /// The generic `upload` path pins the source pages with
    /// `hsa_amd_memory_lock` and creates + destroys a completion signal on
    /// EVERY call. That is fine at bring-up and far too expensive per token:
    /// measured at TP=4 with 12 KiB of counters per rank, `upload` cost
    /// **36.0 µs/token** — more than the ~29 µs that all 96 of the token's
    /// inline all-reduces cost put together (0.302 µs each, measured). Reusing
    /// a fine-grained (already agent-visible, unlockable) zero buffer and one
    /// signal removes both per-call costs.
    fn zero_peer(&self, dptr: u64, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.guard()?;
        let mut guard = self.zero_stage.lock();
        if guard.as_ref().is_none_or(|s| (s.len as u64) < bytes) {
            if let Some(old) = guard.take() {
                unsafe {
                    (self.shared.drv.hsa_amd_memory_pool_free)(old.ptr);
                    (self.shared.drv.hsa_signal_destroy)(old.sig);
                }
            }
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let rc = unsafe {
                (self.shared.drv.hsa_amd_memory_pool_allocate)(
                    self.fine_pool,
                    bytes as usize,
                    0,
                    &mut ptr,
                )
            };
            if rc != HSA_STATUS_SUCCESS {
                return Err(self.fault(rc, "zero stage alloc"));
            }
            let rc = unsafe {
                (self.shared.drv.hsa_amd_agents_allow_access)(1, &self.agent, std::ptr::null(), ptr)
            };
            if rc != HSA_STATUS_SUCCESS {
                unsafe { (self.shared.drv.hsa_amd_memory_pool_free)(ptr) };
                return Err(self.fault(rc, "zero stage access"));
            }
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes as usize) };
            let mut sig = HsaSignal { handle: 0 };
            let rc =
                unsafe { (self.shared.drv.hsa_signal_create)(0, 0, std::ptr::null(), &mut sig) };
            if rc != HSA_STATUS_SUCCESS {
                unsafe { (self.shared.drv.hsa_amd_memory_pool_free)(ptr) };
                return Err(self.fault(rc, "zero stage signal"));
            }
            *guard = Some(ZeroStage {
                ptr,
                len: bytes as usize,
                sig,
            });
        }
        let stage = guard.as_ref().expect("just populated");

        // Re-arm the reused signal. It counts DOWN to 0 on completion, so the
        // wait below is LT 1 — the constant that was wrong (EQ) and hung every
        // copy in this file until it was fixed.
        unsafe { (self.shared.drv.hsa_signal_store_screlease)(stage.sig, 1) };
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_async_copy)(
                dptr as *mut c_void,
                self.agent,
                stage.ptr as *const c_void,
                self.cpu_agent,
                bytes as usize,
                0,
                std::ptr::null(),
                stage.sig,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            return Err(self.fault(
                rc,
                &format!(
                    "zero_peer async_copy ({bytes} B on dev {})",
                    self.device_ordinal
                ),
            ));
        }
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                stage.sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
        }
        Ok(())
    }

    fn copy_peer_blocking(&self, dst_ordinal: u8, dst: u64, src: u64, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.guard()?;
        let dst_agent = *self.agents.get(dst_ordinal as usize).ok_or_else(|| {
            RuntimeError::Device(format!(
                "peer ordinal {dst_ordinal} >= {} GPU agents",
                self.agents.len()
            ))
        })?;

        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe { (self.shared.drv.hsa_signal_create)(1, 0, std::ptr::null(), &mut sig) };
        self.check(rc, "hsa_signal_create (p2p)")?;
        // The two agents name the transfer's endpoints; the SDMA engine walks
        // XGMI directly with no host bounce.
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_async_copy)(
                dst as *mut c_void,
                dst_agent,
                src as *const c_void,
                self.agent,
                bytes as usize,
                0,
                std::ptr::null(),
                sig,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (self.shared.drv.hsa_signal_destroy)(sig) };
            return Err(self.fault(
                rc,
                &format!(
                    "hsa_amd_memory_async_copy (dev {} -> dev {dst_ordinal})",
                    self.device_ordinal
                ),
            ));
        }
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
            (self.shared.drv.hsa_signal_destroy)(sig);
        }
        Ok(())
    }
}

impl HsaBackend {
    /// Number of GPU agents ROCr made visible to this process.
    ///
    /// This is the real device count, so multi-GPU bring-up enumerates instead
    /// of probing ordinals until one fails. Note that the visible set is
    /// selected by **`ROCR_VISIBLE_DEVICES`**, not `HIP_VISIBLE_DEVICES`:
    /// plowrt talks to ROCr directly and never loads the HIP runtime, so the
    /// HIP variable is ignored (measured — `HIP_VISIBLE_DEVICES=4,5,6,7` still
    /// enumerated all 8 agents).
    pub fn gpu_count(&self) -> u32 {
        self.agents.len() as u32
    }

    fn resolve_kernel(&self, exe: HsaExecutable, name: &str) -> Result<HsaKernel> {
        // The loader exposes kernels under "<name>.kd".
        let sym_name = format!("{}.kd\0", name);
        let mut sym = HsaExecutableSymbol { handle: 0 };
        let rc = unsafe {
            (self.shared.drv.hsa_executable_get_symbol_by_name)(
                exe,
                sym_name.as_ptr(),
                &self.agent,
                &mut sym,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            return Err(self.fault(rc, &format!("hsa_executable_get_symbol_by_name('{name}')")));
        }

        let mut kernel_object: u64 = 0;
        let mut kernarg_size: u32 = 0;
        let mut group_segment_size: u32 = 0;
        let mut private_segment_size: u32 = 0;

        unsafe {
            (self.shared.drv.hsa_executable_symbol_get_info)(
                sym,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                &mut kernel_object as *mut _ as *mut c_void,
            );
            (self.shared.drv.hsa_executable_symbol_get_info)(
                sym,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &mut kernarg_size as *mut _ as *mut c_void,
            );
            (self.shared.drv.hsa_executable_symbol_get_info)(
                sym,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                &mut group_segment_size as *mut _ as *mut c_void,
            );
            (self.shared.drv.hsa_executable_symbol_get_info)(
                sym,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                &mut private_segment_size as *mut _ as *mut c_void,
            );
        }

        // THE INVARIANT `dispatch` ALREADY CLAIMED. Its SAFETY comment asserted that
        // "`kernarg_size <= KARG_SLOT` is checked when the kernel is resolved" — and nothing
        // checked it. `runtime/amd/hsa_backend.c:345` does; the Rust port dropped it.
        //
        // Without it the zero-fill and the COv5 implicit block are written for `kernarg_size`
        // bytes into a `KARG_SLOT`-byte slot of a RING, so an oversized kernarg segment does not
        // fault — it silently overwrites the next slot, whose dispatch may already be enqueued.
        // The ring is synchronised against reuse of the SAME slot, not against a write that runs
        // off the end of one, so the corruption lands in another kernel's arguments.
        if kernarg_size as usize > KARG_SLOT {
            return Err(RuntimeError::Device(format!(
                "kernel kernarg segment is {kernarg_size} bytes, larger than the {KARG_SLOT}-byte \
                 kernarg ring slot it has to fit in. Writing it would run past this slot and \
                 corrupt the arguments of the next dispatch in the ring rather than faulting. \
                 Raise KARG_SLOT (crates/plowrt/src/device/hsa.rs) to at least {kernarg_size} and \
                 keep it in step with runtime/amd/hsa_backend.c."
            )));
        }

        Ok(HsaKernel {
            kernel_object,
            kernarg_size,
            group_segment_size,
            private_segment_size,
        })
    }

    fn dispatch(
        &self,
        kernel: &HsaKernel,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        wg_x: u16,
        wg_y: u16,
        wg_z: u16,
        dynamic_lds: u32,
        args: *const c_void,
        args_size: usize,
    ) -> Result<()> {
        // ARGS MUST FIT THE SEGMENT. `runtime/amd/hsa_backend.c:368` checks this; the Rust port
        // did not, and the arithmetic below is UNSIGNED: `kernarg_size - args_size` with
        // `args_size` the larger underflows to ~2^64 and `write_bytes` runs off the ring.
        //
        // `launch` is `pub` and forwards an arbitrary `args.len()` straight through, so this is
        // reachable from any caller that hands over a slice wider than the kernel declared —
        // today's in-tree callers pass the current `DevProgram` ABI against objects validated to
        // report at least that size, which is why it has never fired.
        if args_size > kernel.kernarg_size as usize {
            return Err(RuntimeError::Device(format!(
                "dispatch args are {args_size} bytes but the kernel declares a \
                 {}-byte kernarg segment. Copying them would overrun the kernarg ring slot; the \
                 tail zero-fill length would underflow to a near-2^64 count.",
                kernel.kernarg_size
            )));
        }
        self.guard()?;
        let q = self.queue;
        let idx = unsafe { (self.shared.drv.hsa_queue_add_write_index_screlease)(q, 1) };

        // Spin until ring has space.
        let size = unsafe { (*q).size } as u64;
        while idx.wrapping_sub(unsafe { (self.shared.drv.hsa_queue_load_read_index_scacquire)(q) })
            >= size
        {}

        let slot = (idx & (size - 1)) as u32;
        // SAFETY: `size` is the queue's power-of-two capacity, so `slot` is in
        // `0..size` and the kernarg ring was allocated as `size * KARG_SLOT`
        // bytes — the offset is in bounds by construction. The spin above
        // guarantees the previous user of this slot has retired.
        let karg = unsafe { self.karg_ring.add(slot as usize * KARG_SLOT) };

        // Copy explicit args.
        if args_size > 0 && !args.is_null() {
            // SAFETY: the caller's contract is that `args` points at
            // `args_size` readable bytes; `kernarg_size <= KARG_SLOT` is checked
            // in `resolve_kernel` and `args_size <= kernarg_size` at the top of
            // this function — so both the copy and the zero-fill of the tail stay
            // inside this slot and the fill length cannot underflow. The two
            // ranges are disjoint (the fill starts at `args_size`), which is what
            // `copy_nonoverlapping` requires.
            unsafe {
                std::ptr::copy_nonoverlapping(args as *const u8, karg, args_size);
                std::ptr::write_bytes(
                    karg.add(args_size),
                    0,
                    kernel.kernarg_size as usize - args_size,
                );
            }
        } else {
            // SAFETY: as above — `kernarg_size <= KARG_SLOT` bytes of this slot.
            unsafe {
                std::ptr::write_bytes(karg, 0, kernel.kernarg_size as usize);
            }
        }

        // Fill COv5 implicit block (blockDim, gridDim, remainders).
        let hoff = (args_size + 7) & !7;
        if (kernel.kernarg_size as usize) > hoff {
            // SAFETY: guarded by `kernarg_size > hoff` immediately above, so
            // the COv5 implicit block starts inside the slot.
            let hid = unsafe { karg.add(hoff) };
            let avail = kernel.kernarg_size as usize - hoff;
            let dims: u16 = if grid_z > 1 {
                3
            } else if grid_y > 1 {
                2
            } else {
                1
            };
            // SAFETY (both macros): the `avail >= off + width` guard is the
            // bounds check — a field is written only when the object's declared
            // kernarg segment actually reaches it, which is why an object built
            // without the implicit block is not corrupted here. `write_unaligned`
            // because `hoff` is only 8-byte aligned and the u16 fields are not.
            macro_rules! put32 {
                ($off:expr, $val:expr) => {
                    if avail >= $off + 4 {
                        unsafe {
                            std::ptr::write_unaligned(hid.add($off) as *mut u32, $val);
                        }
                    }
                };
            }
            macro_rules! put16 {
                ($off:expr, $val:expr) => {
                    if avail >= $off + 2 {
                        unsafe {
                            std::ptr::write_unaligned(hid.add($off) as *mut u16, $val);
                        }
                    }
                };
            }
            put32!(0, (grid_x as u32 + wg_x as u32 - 1) / wg_x as u32);
            put32!(4, (grid_y as u32 + wg_y as u32 - 1) / wg_y as u32);
            put32!(8, (grid_z as u32 + wg_z as u32 - 1) / wg_z as u32);
            put16!(12, wg_x);
            put16!(14, wg_y);
            put16!(16, wg_z);
            put16!(18, (grid_x as u16).wrapping_rem(wg_x));
            put16!(20, (grid_y as u16).wrapping_rem(wg_y));
            put16!(22, (grid_z as u16).wrapping_rem(wg_z));
            put16!(64, dims);
        }

        // Write the AQL dispatch packet (everything except the header).
        // SAFETY: `q` is the queue this backend created and holds for its
        // lifetime; its `base_address` is a ring of `size` 64-byte AQL packets,
        // and `slot < size`, so the packet is in bounds. `HsaDispatchPacket` is
        // the `#[repr(C)]` transcription of `hsa_kernel_dispatch_packet_t` and
        // the ring is 64-byte aligned, so the cast is well-aligned. The header
        // is deliberately left for the release-store below: until it is
        // written the packet processor will not read this slot.
        let pkt_base = unsafe { (*q).base_address as *mut u8 };
        let pkt = unsafe { pkt_base.add(slot as usize * 64) as *mut HsaDispatchPacket };
        unsafe {
            // Zero the packet first (memset everything after the header).
            std::ptr::write_bytes((pkt as *mut u8).add(4), 0, 60);
            (*pkt).workgroup_size_x = wg_x;
            (*pkt).workgroup_size_y = wg_y;
            (*pkt).workgroup_size_z = wg_z;
            (*pkt).grid_size_x = grid_x;
            (*pkt).grid_size_y = grid_y;
            (*pkt).grid_size_z = grid_z;
            (*pkt).kernel_object = kernel.kernel_object;
            (*pkt).kernarg_address = karg as u64;
            (*pkt).group_segment_size = kernel.group_segment_size + dynamic_lds;
            (*pkt).private_segment_size = kernel.private_segment_size;
            (*pkt).completion_signal = self.done_signal;
        }

        // Increment the counting signal (one per dispatch).
        unsafe {
            (self.shared.drv.hsa_signal_add_screlease)(self.done_signal, 1);
        }

        // Publish the packet: one release store of header|setup.
        let dims: u16 = if grid_z > 1 {
            3
        } else if grid_y > 1 {
            2
        } else {
            1
        };
        let header: u16 = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) as u16
            | (1 << HSA_PACKET_HEADER_BARRIER) as u16
            | ((HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) as u16)
            | ((HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE) as u16);
        let setup: u16 = (dims << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS as u16) as u16;
        let header_setup: u32 = (setup as u32) << 16 | header as u32;

        unsafe {
            std::sync::atomic::AtomicU32::from_ptr(pkt as *mut u32)
                .store(header_setup, Ordering::Release);
        }

        // Ring the doorbell.
        unsafe {
            (self.shared.drv.hsa_signal_store_screlease)((*q).doorbell_signal, idx as i64);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Engine primitives — the surface `exec::gpu` needs beyond the `Backend` trait.
//
// `exec::gpu` was written against `CudaBackend` concrete types (streams, events,
// pinned host memory). These are the HSA counterparts, named to line up 1:1 so
// the engine can be made generic over both without the call sites moving.
//
// Three impedance mismatches are resolved here, and each one is a deliberate
// choice rather than an oversight:
//
//   STREAMS. CUDA has N ordered streams per context; an HSA backend has ONE AQL
//   queue, and every packet on it already carries the barrier bit (see
//   `dispatch`), so the queue IS an ordered stream. `HsaStream` is therefore a
//   handle onto that single queue, not a new one. The engine's own doc comment
//   says it uses exactly one stream by design (decode and prefill share mutable
//   run state), so this loses nothing it was using.
//
//   EVENTS. A CUDA event is a device-side timestamp recorded in stream order.
//   HSA's equivalent is a completion signal, which answers "has it finished?"
//   but not "when?" without per-queue profiling enabled. The engine uses events
//   for two different jobs, and only one of them needs a clock:
//     * gating pinned-buffer reuse in `UploadPipe` — pure ordering, and the
//       signal does that exactly;
//     * step timing — reported, not load-bearing.
//   So `HsaEvent` carries a signal for ordering and a HOST timestamp taken when
//   the signal resolves. Elapsed time is therefore host-side and includes wait
//   wakeup latency. That is honest for reporting and wrong for microbenchmarks;
//   `runtime/` has dedicated harnesses for the latter.
//
//   DYNAMIC LDS. CUDA needs an explicit opt-in above 48 KiB
//   (`cuFuncSetAttribute`); HSA carries `group_segment_size` in the dispatch
//   packet itself, so the opt-in has no counterpart and `set_max_dynamic_smem`
//   is a checked no-op rather than a silent one.
// ---------------------------------------------------------------------------

/// Page-locked, device-visible host memory: the HSA counterpart of
/// `CudaBackend::host_alloc_pinned`.
///
/// Allocated from the FINE-grained global pool rather than pinned with
/// `hsa_amd_memory_lock`. Both are agent-accessible, but fine-grained pool
/// memory is coherent without an explicit release fence, which is what the
/// engine's staging slabs want: the host writes `ids/pos/kvlen` and the very
/// next packet reads them.
pub struct HsaPinned {
    ptr: *mut u8,
    len: usize,
    shared: Arc<SharedDriver>,
}

// SAFETY: fine-grained pool memory is ordinary host-addressable memory; the
// pool handle is process-global and freed through the retained driver.
unsafe impl Send for HsaPinned {}
unsafe impl Sync for HsaPinned {}

impl HsaPinned {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is a live allocation of `len` bytes owned by `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` excludes any other alias.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for HsaPinned {
    fn drop(&mut self) {
        unsafe {
            (self.shared.drv.hsa_amd_memory_pool_free)(self.ptr as *mut c_void);
        }
    }
}

/// Copy `src` into `dst`, rewriting every 0x80 byte (OCP e4m3 `-0`) to 0x00.
///
/// SWAR over u64 words — the same per-byte zero test the device-side
/// `plow_fp8_mask_neg0` uses (runtime/amd/amd_arch.h), which that header verified
/// exhaustively over all 2^32 words: XOR makes a 0x80 byte 0x00, the carry test
/// marks exactly the zero bytes, and the marker widens to 0xff. See
/// [`HsaUploadRing::push_scrub_fp8_neg0`] for who may use this and why.
fn scrub_fp8_neg0(dst: &mut [u8], src: &[u8]) {
    const H: u64 = 0x8080808080808080;
    const L: u64 = 0x7f7f7f7f7f7f7f7f;
    let n = src.len() & !7;
    for i in (0..n).step_by(8) {
        let w = u64::from_le_bytes(src[i..i + 8].try_into().unwrap());
        let t = w ^ H;
        let z = !(t & L).wrapping_add(L) & !t & H;
        let m = z | z.wrapping_sub(z >> 7);
        dst[i..i + 8].copy_from_slice(&(w & !m).to_le_bytes());
    }
    for i in n..src.len() {
        dst[i] = if src[i] == 0x80 { 0 } else { src[i] };
    }
}

/// One staging slab and the signal that says whether its copy has retired.
struct RingSlot {
    buf: HsaPinned,
    sig: HsaSignal,
    /// Is a copy out of `buf` still in flight? Only a `true` slot may be waited
    /// on — waiting on a signal no copy will ever decrement hangs forever.
    busy: bool,
}

/// Pipelined host→device staging for the weight load: N pinned slabs, N
/// reusable signals, N copies in flight.
///
/// # What this replaces, and why
///
/// [`HsaBackend::memcpy_htod_pinned`] is an ASYNCHRONOUS HSA call used
/// synchronously — it issues `hsa_amd_memory_async_copy` and immediately blocks
/// on the completion signal — and the loader wrapped it in
/// `copy_from_slice` → `memcpy_htod_pinned` over ONE slab. So the SDMA engine
/// idled while the CPU filled the slab, then the CPU idled while SDMA drained
/// it, and a signal was created and destroyed on every chunk. Measured on a warm
/// GLM-5.2 rank: 5.2 s of memcpy and 9.5 s of DMA wait, strictly serialised, for
/// 168.79 GiB across 115 200 chunks.
///
/// Here the two overlap: filling slab `k+1` runs against the copy out of slab
/// `k`, and the signals are created once and re-armed with a store rather than
/// churned per chunk.
///
/// # Why the staging memcpy is still here at all
///
/// The obvious next step is to skip it: `hsa_amd_memory_lock` the checkpoint
/// mmap range and `hsa_amd_memory_async_copy` straight out of page cache, so the
/// CPU copy leaves the critical path instead of merely being hidden. It was
/// implemented and measured on a warm TP4 GLM-5.2 load, and it lost twice over:
///
/// * **11x slower** — 294.9 s to bind against 26.3 s staged, with the upload
///   phase going from 7.6 s to 278.6 s. At this granularity (~1.4 MiB per expert
///   projection) pinning is a kernel page-walk that costs far more than the
///   60 µs memcpy it replaces. The brief's warning that lock is "itself a kernel
///   operation" is the whole story.
/// * **WRONG** — the run decoded `[0, 0, 0, 0]` where every other configuration
///   decodes `[2, 98546, 24, 12]`. Locking a read-only `MAP_SHARED` file range
///   returned success and then did not deliver those bytes. Silent corruption,
///   which is the one failure mode a weight loader must not have.
///
/// So it is not here, and this note is why nobody should spend a second
/// afternoon on it without first fixing the correctness half.
///
/// # The correctness rule
///
/// A slab may not be refilled until its copy has retired — that is what `busy`
/// and the wait at the top of [`HsaUploadRing::push`] enforce, and getting it
/// wrong is silent: the DMA would read bytes belonging to a later chunk and the
/// weight would be quietly wrong. [`HsaUploadRing::drain`] must be called before
/// any of the uploaded memory is read, and `Drop` drains as a backstop so an
/// error path cannot leave a copy running into a freed slab.
pub struct HsaUploadRing {
    slots: Vec<RingSlot>,
    next: usize,
    agent: HsaAgent,
    cpu_agent: HsaAgent,
    shared: Arc<SharedDriver>,
}

// SAFETY: the slabs are `HsaPinned` (already `Send`), and the signals are opaque
// HSA handles that the runtime documents as usable from any thread. The ring
// itself is `!Sync` by having no interior mutability — every method takes
// `&mut self`.
unsafe impl Send for HsaUploadRing {}

impl HsaUploadRing {
    fn new(be: &HsaBackend, slots: usize, bytes: usize) -> Result<HsaUploadRing> {
        let mut ring = HsaUploadRing {
            slots: Vec::with_capacity(slots.max(1)),
            next: 0,
            agent: be.agent,
            cpu_agent: be.cpu_agent,
            shared: Arc::clone(&be.shared),
        };
        for _ in 0..slots.max(1) {
            let buf = be.host_alloc_pinned(bytes)?;
            let mut sig = HsaSignal { handle: 0 };
            // Initial value 0, not 1: a fresh slot is NOT busy, and `push`
            // re-arms to 1 immediately before each copy.
            let rc =
                unsafe { (ring.shared.drv.hsa_signal_create)(0, 0, std::ptr::null(), &mut sig) };
            if rc != HSA_STATUS_SUCCESS {
                // `ring` drops here and destroys the signals already made.
                return Err(hsa_fault(rc, "hsa_signal_create (upload ring)"));
            }
            ring.slots.push(RingSlot {
                buf,
                sig,
                busy: false,
            });
        }
        Ok(ring)
    }

    /// The staging slab size — the largest `src` a single [`HsaUploadRing::push`]
    /// accepts, and therefore the chunk the caller must split by.
    pub fn chunk(&self) -> usize {
        self.slots[0].buf.len()
    }

    /// Stage `src` and SUBMIT its copy to `dptr`, returning without waiting.
    ///
    /// The bytes are safe the moment this returns because they live in the ring's
    /// own slab, not in `src`. The copy is not complete, though — nothing may
    /// read `dptr` until [`HsaUploadRing::drain`].
    pub fn push(&mut self, dptr: u64, src: &[u8]) -> Result<()> {
        self.push_inner(dptr, src, false)
    }

    /// [`HsaUploadRing::push`], but rewriting every 0x80 byte (OCP e4m3 `-0`) to 0x00 inside
    /// the pinned-slab copy — one SWAR pass over bytes that were being copied anyway.
    ///
    /// VALUE-IDENTICAL: `-0 == +0` in every product a weight byte enters. The point is the
    /// CDNA3 decoder: gfx942's `v_cvt_pk_f32_fp8` reads e4m3fnuz, where 0x80 is NaN, so the
    /// kernels guarded every decode with a ~8-VALU neg-0 mask. A scrubbed-at-rest payload
    /// lets the hot staging loop drop that mask (`mpf_fp8x4_to_bf16_h`, op_moe.h). Callers:
    /// block-fp8 expert payloads ONLY — never scales, never bf16, and never MXFP4 payloads
    /// (there 0x80 is two fp4 nibbles, and zeroing them would corrupt real values).
    pub fn push_scrub_fp8_neg0(&mut self, dptr: u64, src: &[u8]) -> Result<()> {
        self.push_inner(dptr, src, true)
    }

    fn push_inner(&mut self, dptr: u64, src: &[u8], scrub: bool) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let n = self.slots.len();
        let i = self.next % n;
        if src.len() > self.slots[i].buf.len() {
            return Err(RuntimeError::Device(format!(
                "upload ring: {} B chunk into a {} B slab",
                src.len(),
                self.slots[i].buf.len()
            )));
        }
        self.wait_slot(i);
        let slot = &mut self.slots[i];
        // Re-arm BEFORE the copy is submitted; the copy decrements to 0.
        unsafe { (self.shared.drv.hsa_signal_store_screlease)(slot.sig, 1) };
        if scrub {
            scrub_fp8_neg0(&mut slot.buf.as_mut_slice()[..src.len()], src);
        } else {
            slot.buf.as_mut_slice()[..src.len()].copy_from_slice(src);
        }
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_async_copy)(
                dptr as *mut c_void,
                self.agent,
                slot.buf.as_ptr() as *const c_void,
                self.cpu_agent,
                src.len(),
                0,
                std::ptr::null(),
                slot.sig,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            // Nothing was submitted, so the signal will never be decremented and
            // the slot must NOT be marked busy — a later `drain` would hang on
            // it. Drain what really is in flight, then report.
            unsafe { (self.shared.drv.hsa_signal_store_screlease)(slot.sig, 0) };
            self.drain()?;
            return Err(hsa_fault(rc, "hsa_amd_memory_async_copy (upload ring)"));
        }
        self.slots[i].busy = true;
        self.next += 1;
        Ok(())
    }

    fn wait_slot(&mut self, i: usize) {
        if !self.slots[i].busy {
            return;
        }
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                self.slots[i].sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
        }
        self.slots[i].busy = false;
    }

    /// Block until every submitted copy has retired.
    ///
    /// Mandatory before anything reads the uploaded bytes. A partially uploaded
    /// weight is silent garbage — there is no fault and no wrong answer until
    /// the model speaks.
    pub fn drain(&mut self) -> Result<()> {
        for i in 0..self.slots.len() {
            self.wait_slot(i);
        }
        Ok(())
    }
}

impl Drop for HsaUploadRing {
    fn drop(&mut self) {
        // A copy still reading a slab we are about to free is a use-after-free
        // in the SDMA engine, so this drain is not tidiness — it is the reason
        // an error path can unwind safely.
        let _ = self.drain();
        for s in &self.slots {
            unsafe { (self.shared.drv.hsa_signal_destroy)(s.sig) };
        }
    }
}

/// A handle onto the backend's single ordered AQL queue. See the module note:
/// this does not create a queue, because there is only ever one to order
/// against.
#[derive(Clone, Copy, Debug, Default)]
pub struct HsaStream;

/// Completion signal + the host time at which it resolved. See the module note
/// on why the timestamp is host-side.
pub struct HsaEvent {
    sig: HsaSignal,
    /// Host time captured when this event's signal was observed complete.
    /// `None` until the event has been waited on.
    ///
    /// A `Mutex`, not a `Cell`: `HsaEvent` carries `unsafe impl Sync` below, and `Cell<T>` is
    /// `!Sync` precisely because `&Cell` handed to two threads is a data race. The justification
    /// that used to sit on that impl — "only touched under `&mut`/owning access in the engine
    /// thread model" — is a claim about today's CALLERS, and `Sync` is what allows `&HsaEvent` to
    /// reach a second thread in the first place, so it could not be discharged that way. This is
    /// a cold path (one lock per `event_record` / `event_elapsed_ms`, not per op), so the lock
    /// costs nothing measurable and the type carries the guarantee instead of a comment.
    at: std::sync::Mutex<Option<std::time::Instant>>,
    /// Timing events carry a clock; sync-only events skip it (cheaper record,
    /// matching the `event_create(timing: bool)` contract on the CUDA side).
    timing: bool,
    shared: Arc<SharedDriver>,
}

// SAFETY: the only field that is not already `Send + Sync` is `sig`, an opaque HSA signal
// handle, and HSA documents signals as safe to store/wait on from any thread. The timestamp is a
// `Mutex`, so it carries its own synchronisation rather than relying on a claim about callers.
unsafe impl Send for HsaEvent {}
unsafe impl Sync for HsaEvent {}

impl HsaEvent {
    /// The raw signal, for attaching to an async copy or dispatch.
    pub fn signal(&self) -> HsaSignal {
        self.sig
    }
}

impl Drop for HsaEvent {
    fn drop(&mut self) {
        unsafe {
            (self.shared.drv.hsa_signal_destroy)(self.sig);
        }
    }
}

impl HsaBackend {
    /// Stable identity of this backend's ordered AQL queue for process-local diagnostics.
    pub fn queue_identity(&self) -> u64 {
        self.queue as usize as u64
    }

    /// Number of independently orderable AQL queues owned by this backend.
    pub fn queue_count(&self) -> usize {
        1
    }

    /// Executor (CU) count — the AMD counterpart of `sm_count`.
    pub fn sm_count(&self) -> u32 {
        self.cu_count
    }

    /// Wavefront width (64 on CDNA). The engine sizes workgroups in waves.
    pub fn wave_width(&self) -> u32 {
        self.wave_width
    }

    /// Per-CU LDS budget in bytes.
    pub fn lds_bytes(&self) -> u32 {
        self.lds_bytes
    }

    /// The single ordered queue, as a stream handle.
    pub fn stream_create(&self) -> Result<HsaStream> {
        Ok(HsaStream)
    }

    /// Drain the queue. Every packet carries the barrier bit, so waiting for
    /// the read index to catch the write index retires everything enqueued.
    pub fn stream_synchronize(&self, _stream: &HsaStream) -> Result<()> {
        self.synchronize()
    }

    /// Drain the queue (device-wide; there is one queue).
    pub fn synchronize(&self) -> Result<()> {
        // A poisoned queue never advances its read index — without this gate
        // the spin below would hang forever instead of erroring.
        self.guard()?;
        let q = self.queue;
        // The write index is the total number of packets ever enqueued; the
        // read index is how many the packet processor has retired. Equality
        // means the queue is drained. This spins rather than blocking on a
        // signal because the engine calls it once per step, after work it just
        // enqueued, so the expected wait is short and a signal round-trip
        // would cost more than the spin.
        loop {
            let w = unsafe { (self.shared.drv.hsa_queue_add_write_index_screlease)(q, 0) };
            let r = unsafe { (self.shared.drv.hsa_queue_load_read_index_scacquire)(q) };
            if r >= w {
                return Ok(());
            }
            std::hint::spin_loop();
        }
    }

    /// Create an event. `timing` selects whether the event carries a clock.
    pub fn event_create(&self, timing: bool) -> Result<HsaEvent> {
        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe { (self.shared.drv.hsa_signal_create)(1, 0, std::ptr::null(), &mut sig) };
        self.check(rc, "hsa_signal_create (event)")?;
        Ok(HsaEvent {
            sig,
            at: std::sync::Mutex::new(None),
            timing,
            shared: self.shared.clone(),
        })
    }

    /// Record `event` at the current queue tail.
    ///
    /// HSA has no "record a marker" packet — a signal is armed only by being
    /// attached to a real packet as its completion signal. That matters less
    /// than it looks here, because **this backend's copies are synchronous**:
    /// `upload`/`download` each attach their own one-shot signal and wait on it
    /// before returning, so everything the engine could have enqueued before a
    /// record has already retired by the time `event_record` is called.
    ///
    /// So the event is stored ALREADY-SATISFIED (signal value 0) plus a host
    /// timestamp. Arming it to 1 instead would deadlock: nothing would ever
    /// decrement it, and `event_synchronize` would block forever on a signal
    /// with no producer — measured, and the reason this is written down.
    ///
    /// When the async-copy path becomes genuinely asynchronous, this must
    /// change to attach `event.sig` to the copy as its completion signal; the
    /// wait in `event_synchronize` is already the correct shape for that.
    pub fn event_record(&self, event: &HsaEvent, _stream: &HsaStream) -> Result<()> {
        unsafe {
            (self.shared.drv.hsa_signal_store_screlease)(event.sig, 0);
        }
        if event.timing {
            *event.at.lock().unwrap() = Some(std::time::Instant::now());
        }
        Ok(())
    }

    /// Block until `event` resolves.
    ///
    /// Deliberately does NOT drain the AQL queue first. Every memory operation
    /// this backend performs already waits on its own completion signal before
    /// returning, so at an event there is nothing outstanding to drain — and
    /// calling `synchronize()` here HUNG on gfx950/ROCm 7.2.4 (the read index
    /// never caught the write index on a queue with no dispatched packets;
    /// reproduced with a 60 s timeout on the event-only test). The drain
    /// belongs where packets are actually dispatched, not on the event path.
    ///
    /// The timestamp is NOT re-stamped here: it belongs to the record point,
    /// and overwriting it would fold the wait itself into the measurement.
    pub fn event_synchronize(&self, event: &HsaEvent) -> Result<()> {
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                event.sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
        }
        Ok(())
    }

    /// Host-side elapsed time between two resolved timing events.
    ///
    /// Returns 0.0 rather than an error when either event has no stamp: the
    /// engine reads this only to fill a reported metric, and a missing sample
    /// must not fail a serving step.
    pub fn event_elapsed_ms(&self, start: &HsaEvent, end: &HsaEvent) -> Result<f32> {
        let (a0, b0) = (*start.at.lock().unwrap(), *end.at.lock().unwrap());
        match (a0, b0) {
            (Some(a), Some(b)) => Ok(b.saturating_duration_since(a).as_secs_f32() * 1e3),
            _ => Ok(0.0),
        }
    }

    /// Page-locked, device-visible host staging memory.
    pub fn host_alloc_pinned(&self, bytes: usize) -> Result<HsaPinned> {
        self.guard()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_pool_allocate)(self.fine_pool, bytes, 0, &mut ptr)
        };
        self.check(
            rc,
            &format!("hsa_amd_memory_pool_allocate(fine, {bytes} bytes)"),
        )?;
        // The GPU agent must be allowed to read the staging slab; without this
        // the copy engine faults on first touch.
        let rc = unsafe {
            (self.shared.drv.hsa_amd_agents_allow_access)(1, &self.agent, std::ptr::null(), ptr)
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (self.shared.drv.hsa_amd_memory_pool_free)(ptr) };
            return Err(self.fault(rc, "hsa_amd_agents_allow_access (pinned)"));
        }
        Ok(HsaPinned {
            ptr: ptr as *mut u8,
            len: bytes,
            shared: self.shared.clone(),
        })
    }

    /// A ring of pinned staging slabs with reusable signals, for the weight
    /// load. See [`HsaUploadRing`].
    pub fn upload_ring(&self, slots: usize, bytes: usize) -> Result<HsaUploadRing> {
        HsaUploadRing::new(self, slots, bytes)
    }

    /// Blocking host→device copy to a raw device pointer.
    pub fn memcpy_htod(&self, dptr: u64, src: &[u8]) -> Result<()> {
        let dst = DeviceMem::view(dptr, src.len() as u64);
        self.upload(&dst, 0, src)
    }

    /// Enqueue a host→device copy.
    ///
    /// # Safety
    /// `src` must stay live and unmodified until the copy retires — the caller
    /// gates that with an event, exactly as on the CUDA side.
    pub unsafe fn memcpy_htod_async(
        &self,
        dptr: u64,
        src: &[u8],
        _stream: &HsaStream,
    ) -> Result<()> {
        // Fine-grained staging memory is already agent-visible, so the copy can
        // be issued directly. Correctness does not depend on which engine runs
        // it; ordering against the interpreter dispatch comes from the queue.
        let dst = DeviceMem::view(dptr, src.len() as u64);
        self.upload(&dst, 0, src)
    }

    /// Enqueue a device→host copy.
    ///
    /// # Safety
    /// `dst` must stay live until the copy retires.
    pub unsafe fn memcpy_dtoh_async(
        &self,
        dst: &mut [u8],
        dptr: u64,
        _stream: &HsaStream,
    ) -> Result<()> {
        let src = DeviceMem::view(dptr, dst.len() as u64);
        self.download(&src, 0, dst)
    }

    /// Device→device copy within this agent.
    /// N device-to-device copies against ONE completion signal.
    ///
    /// `hsa_amd_memory_async_copy` decrements its completion signal on finish, so a signal armed
    /// at N and waited on once is a correct barrier for N copies — and it replaces N blocked
    /// host waits with one. A prefix snapshot is 276 tensors per rank, where the per-copy
    /// signal round trip, not the 56 MiB, is what costs.
    pub fn memcpy_dtod_batch(&self, pairs: &[(u64, u64, u64)]) -> Result<()> {
        let live: Vec<&(u64, u64, u64)> = pairs.iter().filter(|p| p.2 > 0).collect();
        if live.is_empty() {
            return Ok(());
        }
        self.guard()?;
        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe {
            (self.shared.drv.hsa_signal_create)(live.len() as i64, 0, std::ptr::null(), &mut sig)
        };
        self.check(rc, "hsa_signal_create (dtod batch)")?;
        for &&(dst, src, bytes) in &live {
            let rc = unsafe {
                (self.shared.drv.hsa_amd_memory_async_copy)(
                    dst as *mut c_void,
                    self.agent,
                    src as *const c_void,
                    self.agent,
                    bytes as usize,
                    0,
                    std::ptr::null(),
                    sig,
                )
            };
            if rc != HSA_STATUS_SUCCESS {
                // Copies already issued still hold references to `sig`; wait them out before
                // destroying it or the runtime writes into freed memory.
                unsafe {
                    (self.shared.drv.hsa_signal_wait_scacquire)(
                        sig,
                        HSA_SIGNAL_CONDITION_LT,
                        1,
                        u64::MAX,
                        HSA_WAIT_STATE_BLOCKED,
                    );
                    (self.shared.drv.hsa_signal_destroy)(sig);
                }
                return Err(self.fault(rc, "hsa_amd_memory_async_copy (dtod batch)"));
            }
        }
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
            (self.shared.drv.hsa_signal_destroy)(sig);
        }
        Ok(())
    }

    pub fn memcpy_dtod(&self, dst: u64, src: u64, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.guard()?;
        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe { (self.shared.drv.hsa_signal_create)(1, 0, std::ptr::null(), &mut sig) };
        self.check(rc, "hsa_signal_create (dtod)")?;
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_async_copy)(
                dst as *mut c_void,
                self.agent,
                src as *const c_void,
                self.agent,
                bytes as usize,
                0,
                std::ptr::null(),
                sig,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (self.shared.drv.hsa_signal_destroy)(sig) };
            return Err(self.fault(rc, "hsa_amd_memory_async_copy (dtod)"));
        }
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
            (self.shared.drv.hsa_signal_destroy)(sig);
        }
        Ok(())
    }

    /// Zero `n` bytes of a named module global. Returns `false` when the symbol
    /// is absent, matching the CUDA contract (an object without the global is
    /// not an error — the engine probes for optional ones).
    pub fn module_global_zero(&self, module: &Module, name: &str, n: usize) -> Result<bool> {
        // `module_load` stores the frozen executable handle directly as the
        // module id, so the handle round-trips without a side table.
        let exe = HsaExecutable { handle: module.id };
        let cname = std::ffi::CString::new(name)
            .map_err(|_| RuntimeError::Device(format!("bad symbol name {name:?}")))?;
        let mut sym = HsaExecutableSymbol { handle: 0 };
        let rc = unsafe {
            (self.shared.drv.hsa_executable_get_symbol_by_name)(
                exe,
                cname.as_ptr() as *const u8,
                &self.agent,
                &mut sym,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            return Ok(false);
        }
        let mut addr: u64 = 0;
        let rc = unsafe {
            (self.shared.drv.hsa_executable_symbol_get_info)(
                sym,
                HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                &mut addr as *mut _ as *mut c_void,
            )
        };
        if rc != HSA_STATUS_SUCCESS || addr == 0 {
            return Ok(false);
        }
        let zeros = vec![0u8; n];
        self.memcpy_htod(addr, &zeros)?;
        Ok(true)
    }

    /// Static group-segment (LDS) bytes the kernel was COMPILED to use.
    ///
    /// On AMD this is where the interpreter's arena actually lives: `interp.hip`
    /// declares `__shared__ plow_smem sm` statically, so the size is baked into
    /// the code object and every dispatch passes `dynamic_lds = 0`. It is NOT
    /// the counterpart of the CUDA engine's `SMEM_PF`, which is a *dynamic*
    /// budget opted into per-function with `cuFuncSetAttribute`.
    pub fn kernel_lds_bytes(k: &HsaKernel) -> u32 {
        k.group_segment_size
    }

    /// Device address of a named module global, or `None` when absent.
    ///
    /// Absence is a legitimate answer, not an error: the engine probes for
    /// optional globals (`plow_arena_bytes`, `plow_packet_hash_lo`, the trace
    /// buffers) that an unspecialised object simply does not carry.
    fn global_addr(&self, module: &Module, name: &str) -> Result<Option<u64>> {
        // `module_load` stores the frozen executable handle directly as the
        // module id, so the handle round-trips without a side table.
        let exe = HsaExecutable { handle: module.id };
        let cname = std::ffi::CString::new(name)
            .map_err(|_| RuntimeError::Device(format!("bad symbol name {name:?}")))?;
        let mut sym = HsaExecutableSymbol { handle: 0 };
        let rc = unsafe {
            (self.shared.drv.hsa_executable_get_symbol_by_name)(
                exe,
                cname.as_ptr() as *const u8,
                &self.agent,
                &mut sym,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            return Ok(None);
        }
        let mut addr: u64 = 0;
        let rc = unsafe {
            (self.shared.drv.hsa_executable_symbol_get_info)(
                sym,
                HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                &mut addr as *mut _ as *mut c_void,
            )
        };
        if rc != HSA_STATUS_SUCCESS || addr == 0 {
            return Ok(None);
        }
        Ok(Some(addr))
    }

    /// Resolve a kernel by name from a loaded module.
    pub fn get_function(&self, module: &Module, name: &str) -> Result<HsaKernel> {
        self.resolve_kernel(HsaExecutable { handle: module.id }, name)
    }

    /// Destroy a loaded executable.
    pub fn module_unload(&self, module: &Module) -> Result<()> {
        let rc = unsafe {
            (self.shared.drv.hsa_executable_destroy)(HsaExecutable { handle: module.id })
        };
        self.check(rc, "hsa_executable_destroy")
    }

    /// Fill `n` bytes at `dptr` with `value`, through the copy engine.
    ///
    /// There is no queue-ordered HSA fill — `hsa_amd_memory_fill` is a host
    /// operation on host-accessible memory, not a packet on the queue — so this
    /// is an SDMA copy from a cached staging buffer, and it BLOCKS. The
    /// staging buffer is grown and re-filled only when `value` or the size
    /// changes, because the common case by far is `value == 0` at a fixed
    /// counter-region size, once per token.
    pub fn memset_d8(&self, dptr: u64, value: u8, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.guard()?;
        let mut guard = self.fill_stage.lock();
        let need_new = match guard.as_ref() {
            Some(s) => s.len < n || s.value != value,
            None => true,
        };
        if need_new {
            if let Some(old) = guard.take() {
                unsafe { (self.shared.drv.hsa_amd_memory_pool_free)(old.ptr) };
            }
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let rc = unsafe {
                (self.shared.drv.hsa_amd_memory_pool_allocate)(self.fine_pool, n, 0, &mut ptr)
            };
            if rc != HSA_STATUS_SUCCESS {
                return Err(self.fault(rc, &format!("fill stage alloc({n})")));
            }
            let rc = unsafe {
                (self.shared.drv.hsa_amd_agents_allow_access)(1, &self.agent, std::ptr::null(), ptr)
            };
            if rc != HSA_STATUS_SUCCESS {
                unsafe { (self.shared.drv.hsa_amd_memory_pool_free)(ptr) };
                return Err(self.fault(rc, "fill stage access"));
            }
            unsafe { std::ptr::write_bytes(ptr as *mut u8, value, n) };
            *guard = Some(FillStage { ptr, len: n, value });
        }
        let src = guard.as_ref().expect("just populated").ptr;
        // SAFETY: `src` is `n` bytes of agent-visible fine-grained memory held
        // alive by the lock guard for the duration of this blocking copy.
        let slice = unsafe { std::slice::from_raw_parts(src as *const u8, n) };
        self.memcpy_htod_pinned(dptr, slice)
    }

    /// H2D copy whose source is ALREADY agent-visible (a [`HsaPinned`] slab), so
    /// it skips the `hsa_amd_memory_lock` that `upload` pays per call.
    ///
    /// Two separate reasons this exists, and both are measured:
    ///
    /// 1. `hsa_amd_memory_lock` is syscall-class. The AMD reference driver
    ///    records that pinning per step "cost more than the whole forward pass",
    ///    which is why its hot path uses `copy_h2d` over `alloc_host` memory and
    ///    reserves `upload` for load time.
    /// 2. Locking an already-device-accessible fine-grained POOL allocation is
    ///    not merely wasteful, it is INVALID — `hsa_amd_memory_lock` returns
    ///    HSA_STATUS_ERROR (4096). So `upload` cannot be used on a pinned slab
    ///    at all; the engine's first decode step failed exactly here.
    ///
    /// # Safety contract (not `unsafe`, but a contract nonetheless)
    /// `src` must live in memory the GPU agent may already read — a `HsaPinned`
    /// slab, or the fine-grained pool. A stack or `Vec` source faults the GPU
    /// with an opaque "memory access fault" that reads as a kernel bug.
    pub fn memcpy_htod_pinned(&self, dptr: u64, src: &[u8]) -> Result<()> {
        self.guard()?;
        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe { (self.shared.drv.hsa_signal_create)(1, 0, std::ptr::null(), &mut sig) };
        self.check(rc, "hsa_signal_create (fill)")?;
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_async_copy)(
                dptr as *mut c_void,
                self.agent,
                src.as_ptr() as *const c_void,
                self.cpu_agent,
                src.len(),
                0,
                std::ptr::null(),
                sig,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (self.shared.drv.hsa_signal_destroy)(sig) };
            return Err(self.fault(rc, "async_copy (fill)"));
        }
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
            (self.shared.drv.hsa_signal_destroy)(sig);
        }
        Ok(())
    }

    /// D2H copy whose destination is already agent-visible (a [`HsaPinned`]
    /// slab), so it skips the per-call `hsa_amd_memory_lock` that `download`
    /// pays. The mirror of [`HsaBackend::memcpy_htod_pinned`], and it exists
    /// for the same measured reason: the token readback happens once per decode
    /// step, and pinning a stack array to move FOUR BYTES is the pathology the
    /// reference driver records as costing "more than the whole forward pass".
    pub fn memcpy_dtoh_pinned(&self, dst: &mut [u8], dptr: u64) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        self.guard()?;
        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe { (self.shared.drv.hsa_signal_create)(1, 0, std::ptr::null(), &mut sig) };
        self.check(rc, "hsa_signal_create (d2h)")?;
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_async_copy)(
                dst.as_mut_ptr() as *mut c_void,
                self.cpu_agent,
                dptr as *const c_void,
                self.agent,
                dst.len(),
                0,
                std::ptr::null(),
                sig,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (self.shared.drv.hsa_signal_destroy)(sig) };
            return Err(self.fault(rc, "async_copy (d2h pinned)"));
        }
        unsafe {
            (self.shared.drv.hsa_signal_wait_scacquire)(
                sig,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            );
            (self.shared.drv.hsa_signal_destroy)(sig);
        }
        Ok(())
    }

    /// Read a `u32` module global.
    pub fn module_global_u32(&self, module: &Module, name: &str) -> Result<Option<u32>> {
        let Some(addr) = self.global_addr(module, name)? else {
            return Ok(None);
        };
        let mut buf = [0u8; 4];
        self.download(&DeviceMem::view(addr, 4), 0, &mut buf)?;
        Ok(Some(u32::from_le_bytes(buf)))
    }

    /// Read up to `max` bytes of a named module global.
    pub fn module_global_bytes(
        &self,
        module: &Module,
        name: &str,
        max: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool> {
        let Some(addr) = self.global_addr(module, name)? else {
            return Ok(false);
        };
        out.resize(max, 0);
        self.download(&DeviceMem::view(addr, max as u64), 0, out)?;
        Ok(true)
    }

    /// Launch `f` over `grid` workgroups of `block` threads.
    ///
    /// `grid` is in WORKGROUPS, matching the CUDA engine's call sites; an AQL
    /// dispatch counts threads, so it is multiplied here. Getting that
    /// backwards launches 256 workgroups' worth of threads as one workgroup, or
    /// 65536 workgroups — both of which the packet processor accepts.
    pub fn launch(
        &self,
        f: HsaKernel,
        grid: u32,
        block: u32,
        smem_bytes: u32,
        args: &[u8],
    ) -> Result<()> {
        if grid == 0 || block == 0 {
            return Err(RuntimeError::Device(format!(
                "degenerate launch: grid={grid} block={block}"
            )));
        }
        self.dispatch(
            &f,
            grid * block,
            1,
            1,
            block as u16,
            1,
            1,
            smem_bytes,
            args.as_ptr() as *const c_void,
            args.len(),
        )
    }

    /// No-op with a range check. HSA carries `group_segment_size` in the
    /// dispatch packet, so there is no per-function opt-in to set; the check
    /// keeps an over-budget request from silently becoming a launch failure.
    pub fn set_max_dynamic_smem(&self, _f: &HsaKernel, bytes: u32) -> Result<()> {
        if bytes > self.lds_bytes {
            return Err(RuntimeError::Device(format!(
                "dynamic LDS request {bytes} B exceeds the {} B per-CU budget",
                self.lds_bytes
            )));
        }
        Ok(())
    }

    /// The `hsa_amd_vmem_*` table, or a `Device` error naming what is missing.
    fn vmem(&self) -> Result<&VmemFns> {
        self.shared.drv.vmem.as_ref().ok_or_else(|| {
            RuntimeError::Device(
                "libhsa-runtime64 has no hsa_amd_vmem_* API (needs ROCm >= 5.7) — \
                 the VMM-backed KV pool cannot come up"
                    .into(),
            )
        })
    }

    /// Is the virtual-memory API available on this runtime? The engine asks
    /// before building a [`crate::memory::vmm::VmmKv`].
    pub fn has_vmm(&self) -> bool {
        self.shared.drv.vmem.is_some()
    }
}

// ─── VMM (hsa_amd_vmem_*) ───────────────────────────────────────────────────

/// The VMM driver surface as `crate::memory::vmm` consumes it — the ROCr
/// counterpart of the CUDA impl in [`super::cuda`]. The mapping is 1:1:
///
/// | `VmmOps`      | ROCr                                    | CUDA |
/// |---------------|-----------------------------------------|------|
/// | `granularity` | pool info `RUNTIME_ALLOC_REC_GRANULE`   | `cuMemGetAllocationGranularity` |
/// | `reserve`     | `hsa_amd_vmem_address_reserve_align`    | `cuMemAddressReserve` |
/// | `create`      | `hsa_amd_vmem_handle_create`            | `cuMemCreate` |
/// | `map`         | `hsa_amd_vmem_map`                      | `cuMemMap` |
/// | `set_access`  | `hsa_amd_vmem_set_access`               | `cuMemSetAccess` |
///
/// Two differences that matter to the caller:
///
/// * There is no context to bind — HSA is process-global, so every entry here
///   is callable from the pool's pre-mapper thread as-is (the CUDA impl has to
///   re-`bind()` per call).
/// * `hsa_amd_vmem_handle_create` names the **memory pool**, not a device
///   ordinal, so the physical block lands in this agent's coarse-grained VRAM
///   by construction; the CUDA path encodes the same intent in
///   `CUmemAllocationProp::location`.
impl crate::memory::vmm::VmmOps for HsaBackend {
    fn granularity(&self) -> Result<u64> {
        let mut g: usize = 0;
        // SAFETY: out-pointer sized for a `size_t` attribute.
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_pool_get_info)(
                self.vram_pool,
                HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
                &mut g as *mut _ as *mut c_void,
            )
        };
        self.check(rc, "hsa_amd_memory_pool_get_info(REC_GRANULE)")?;
        if g == 0 {
            return Err(RuntimeError::Device(
                "hsa_amd_memory_pool_get_info(REC_GRANULE): granule=0".into(),
            ));
        }
        Ok(g as u64)
    }

    fn reserve(&self, bytes: u64) -> Result<u64> {
        self.guard()?;
        let f = self.vmem()?;
        // Align the reservation to the physical granule. The non-`_align` entry
        // point only guarantees page alignment, and every `map` into this range
        // is a granule-sized block at a granule-multiple offset — an unaligned
        // base would turn each of those into an INVALID_ARGUMENT.
        let align = crate::memory::vmm::VmmOps::granularity(self)?;
        let mut va: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-pointer; no fixed address requested.
        let rc = unsafe { (f.address_reserve_align)(&mut va, bytes as usize, 0, align, 0) };
        self.check(
            rc,
            &format!("hsa_amd_vmem_address_reserve_align({bytes} B, align {align})"),
        )?;
        Ok(va as u64)
    }

    fn address_free(&self, va: u64, bytes: u64) {
        let Ok(f) = self.vmem() else { return };
        // SAFETY: va/bytes from a prior reserve, freed exactly once (pool
        // contract). Infallible teardown path — log, don't propagate.
        let rc = unsafe { (f.address_free)(va as *mut c_void, bytes as usize) };
        if rc != HSA_STATUS_SUCCESS {
            tracing::warn!(rc, va, bytes, "hsa_amd_vmem_address_free failed");
        }
    }

    fn create(&self, bytes: u64) -> Result<u64> {
        self.guard()?;
        let f = self.vmem()?;
        let mut h = HsaVmemHandle { handle: 0 };
        // SAFETY: out-pointer; bytes is a granule multiple (pool contract).
        let rc = unsafe {
            (f.handle_create)(
                self.vram_pool,
                bytes as usize,
                HSA_AMD_MEMORY_TYPE_PINNED,
                0,
                &mut h,
            )
        };
        self.check(rc, &format!("hsa_amd_vmem_handle_create({bytes} B)"))?;
        Ok(h.handle)
    }

    fn release(&self, handle: u64) {
        let Ok(f) = self.vmem() else { return };
        // SAFETY: handle from create, released exactly once (pool refcount).
        let rc = unsafe { (f.handle_release)(HsaVmemHandle { handle }) };
        if rc != HSA_STATUS_SUCCESS {
            tracing::warn!(rc, handle, "hsa_amd_vmem_handle_release failed");
        }
    }

    fn map(&self, va: u64, bytes: u64, handle: u64) -> Result<()> {
        self.guard()?;
        let f = self.vmem()?;
        // SAFETY: va range inside a reservation, handle live, in_offset 0 —
        // multi-map of one handle into several ranges is what prefix sharing is.
        let rc = unsafe {
            (f.map)(
                va as *mut c_void,
                bytes as usize,
                0,
                HsaVmemHandle { handle },
                0,
            )
        };
        self.check(
            rc,
            &format!("hsa_amd_vmem_map (va={va:#x} bytes={bytes} handle={handle:#x})"),
        )
    }

    fn unmap(&self, va: u64, bytes: u64) {
        let Ok(f) = self.vmem() else { return };
        // SAFETY: exactly the mapped range (pool contract).
        let rc = unsafe { (f.unmap)(va as *mut c_void, bytes as usize) };
        if rc != HSA_STATUS_SUCCESS {
            tracing::warn!(rc, va, bytes, "hsa_amd_vmem_unmap failed");
        }
    }

    fn set_access(&self, va: u64, bytes: u64) -> Result<()> {
        self.guard()?;
        let f = self.vmem()?;
        let desc = HsaAmdMemoryAccessDesc {
            permissions: HSA_ACCESS_PERMISSION_RW,
            agent_handle: self.agent,
        };
        // SAFETY: range fully mapped (pool maps before granting access).
        let rc = unsafe { (f.set_access)(va as *mut c_void, bytes as usize, &desc, 1) };
        self.check(
            rc,
            &format!("hsa_amd_vmem_set_access (va={va:#x} bytes={bytes})"),
        )
    }

    fn alloc(&self, bytes: u64) -> Result<u64> {
        self.guard()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: out-pointer to a coarse-grained VRAM allocation. Ordinary
        // pool memory, not VMM — snapshot buffers are never re-mapped.
        let rc = unsafe {
            (self.shared.drv.hsa_amd_memory_pool_allocate)(
                self.vram_pool,
                bytes as usize,
                0,
                &mut ptr,
            )
        };
        self.check(
            rc,
            &format!("hsa_amd_memory_pool_allocate(vmm snapshot, {bytes} B)"),
        )?;
        Ok(ptr as u64)
    }

    fn free(&self, va: u64) {
        // SAFETY: va from VmmOps::alloc, freed exactly once (pool contract).
        let rc = unsafe { (self.shared.drv.hsa_amd_memory_pool_free)(va as *mut c_void) };
        if rc != HSA_STATUS_SUCCESS {
            tracing::warn!(rc, va, "hsa_amd_memory_pool_free(vmm snapshot) failed");
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
        // Victims picked under the lock, released outside it (the release is
        // a runtime call that can block).
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

#[cfg(test)]
mod scrub_tests {
    use super::scrub_fp8_neg0;

    /// Every byte value, at every position within the 8-byte SWAR word and in the
    /// scalar tail: only 0x80 is rewritten, and it is rewritten to 0x00.
    #[test]
    fn scrubs_exactly_neg_zero() {
        for pos in 0..11usize {
            for b in 0..=255u8 {
                let mut src = [0xA5u8; 11]; // non-trivial background
                src[pos] = b;
                let mut dst = [0u8; 11];
                scrub_fp8_neg0(&mut dst, &src);
                for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
                    let want = if s == 0x80 { 0 } else { s };
                    assert_eq!(d, want, "pos {pos} byte {b:#x} lane {i}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{hsa_status_name, is_hsa_fatal};

    #[test]
    fn fatal_classification_matches_the_rocr_contract() {
        // Trap statuses that kill the queue/agent.
        for rc in [41, 42, 43, 0x1016, 0x1020] {
            assert!(is_hsa_fatal(rc), "hsa_status {rc:#x} must be fatal");
        }
        // Transient / bad-call statuses that must NOT kill the engine.
        for rc in [0x1008, 0x1001, 0x1007, 0x100F, 40, 45] {
            assert!(!is_hsa_fatal(rc), "hsa_status {rc:#x} must not be fatal");
        }
    }

    #[test]
    fn status_names_resolve() {
        assert_eq!(hsa_status_name(43), "HSA_STATUS_ERROR_MEMORY_FAULT");
        assert_eq!(hsa_status_name(0x1008), "HSA_STATUS_ERROR_OUT_OF_RESOURCES");
        assert_eq!(hsa_status_name(-1), "HSA_STATUS_UNKNOWN");
    }
}
