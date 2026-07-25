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

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::device::{Backend, DeviceFree, DeviceMem, ExecutorClass, ExecutorTarget, LaunchCfg, Module};
use crate::{Result, RuntimeError};

// ─── HSA ABI constants ───────────────────────────────────────────────────────

const HSA_STATUS_SUCCESS: i32 = 0;

// hsa_device_type_t
const HSA_DEVICE_TYPE_CPU: u32 = 0;
const HSA_DEVICE_TYPE_GPU: u32 = 1;

// hsa_agent_info_t
const HSA_AGENT_INFO_NAME: u32 = 0;
const HSA_AGENT_INFO_DEVICE: u32 = 17;
// AMD extension: CU count
const HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT: u32 = 0xA000;

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
const HSA_SIGNAL_CONDITION_LT: u32 = 0;

// hsa_wait_state_t
const HSA_WAIT_STATE_BLOCKED: u32 = 1;

// hsa_profile_t
const HSA_PROFILE_FULL: u32 = 1;

// hsa_default_float_rounding_mode_t
const HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT: u32 = 1;

// hsa_executable_symbol_info_t
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
struct HsaSignal {
    handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HsaRegion {
    handle: u64,
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
                *unsafe { $lib.get($name) }
                    .map_err(|e| RuntimeError::Device(format!("resolve {}: {e}", std::str::from_utf8($name).unwrap_or("?"))))?
            }
        }

        let drv = HsaDriver {
            hsa_init: resolve!(lib, b"hsa_init\0"),
            hsa_shut_down: resolve!(lib, b"hsa_shut_down\0"),
            hsa_iterate_agents: resolve!(lib, b"hsa_iterate_agents\0"),
            hsa_agent_get_info: resolve!(lib, b"hsa_agent_get_info\0"),
            hsa_agent_iterate_regions: resolve!(lib, b"hsa_agent_iterate_regions\0"),
            hsa_region_get_info: resolve!(lib, b"hsa_region_get_info\0"),
            hsa_amd_agent_iterate_memory_pools: resolve!(lib, b"hsa_amd_agent_iterate_memory_pools\0"),
            hsa_amd_memory_pool_get_info: resolve!(lib, b"hsa_amd_memory_pool_get_info\0"),
            hsa_amd_memory_pool_allocate: resolve!(lib, b"hsa_amd_memory_pool_allocate\0"),
            hsa_amd_memory_pool_free: resolve!(lib, b"hsa_amd_memory_pool_free\0"),
            hsa_amd_agents_allow_access: resolve!(lib, b"hsa_amd_agents_allow_access\0"),
            hsa_amd_memory_lock: resolve!(lib, b"hsa_amd_memory_lock\0"),
            hsa_amd_memory_unlock: resolve!(lib, b"hsa_amd_memory_unlock\0"),
            hsa_amd_memory_async_copy: resolve!(lib, b"hsa_amd_memory_async_copy\0"),
            hsa_queue_create: resolve!(lib, b"hsa_queue_create\0"),
            hsa_queue_destroy: resolve!(lib, b"hsa_queue_destroy\0"),
            hsa_queue_add_write_index_screlease: resolve!(lib, b"hsa_queue_add_write_index_screlease\0"),
            hsa_queue_load_read_index_scacquire: resolve!(lib, b"hsa_queue_load_read_index_scacquire\0"),
            hsa_signal_create: resolve!(lib, b"hsa_signal_create\0"),
            hsa_signal_destroy: resolve!(lib, b"hsa_signal_destroy\0"),
            hsa_signal_store_screlease: resolve!(lib, b"hsa_signal_store_screlease\0"),
            hsa_signal_wait_scacquire: resolve!(lib, b"hsa_signal_wait_scacquire\0"),
            hsa_signal_add_screlease: resolve!(lib, b"hsa_signal_add_screlease\0"),
            hsa_code_object_reader_create_from_memory: resolve!(lib, b"hsa_code_object_reader_create_from_memory\0"),
            hsa_code_object_reader_destroy: resolve!(lib, b"hsa_code_object_reader_destroy\0"),
            hsa_executable_create_alt: resolve!(lib, b"hsa_executable_create_alt\0"),
            hsa_executable_load_agent_code_object: resolve!(lib, b"hsa_executable_load_agent_code_object\0"),
            hsa_executable_freeze: resolve!(lib, b"hsa_executable_freeze\0"),
            hsa_executable_destroy: resolve!(lib, b"hsa_executable_destroy\0"),
            hsa_executable_get_symbol_by_name: resolve!(lib, b"hsa_executable_get_symbol_by_name\0"),
            hsa_executable_symbol_get_info: resolve!(lib, b"hsa_executable_symbol_get_info\0"),
            lib,
        };
        Ok(drv)
    }
}

/// Internal resolved kernel metadata (mirrors `plow_hsa_kernel` in hsa_backend.h).
struct HsaKernel {
    kernel_object: u64,
    kernarg_size: u32,
    group_segment_size: u32,
    private_segment_size: u32,
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
    let rc = (acc.get_info)(agent, HSA_AGENT_INFO_DEVICE, &mut dtype as *mut u32 as *mut c_void);
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
    if (acc.get_pool_info)(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &mut seg as *mut _ as *mut c_void)
        != HSA_STATUS_SUCCESS
    {
        return HSA_STATUS_SUCCESS;
    }
    if seg != HSA_AMD_SEGMENT_GLOBAL {
        return HSA_STATUS_SUCCESS;
    }
    let mut flags: u32 = 0;
    if (acc.get_pool_info)(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &mut flags as *mut _ as *mut c_void)
        != HSA_STATUS_SUCCESS
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
    if (acc.get_region_info)(region, HSA_REGION_INFO_SEGMENT, &mut seg as *mut _ as *mut c_void)
        != HSA_STATUS_SUCCESS
    {
        return HSA_STATUS_SUCCESS;
    }
    if seg != HSA_REGION_SEGMENT_GROUP {
        return HSA_STATUS_SUCCESS;
    }
    let mut sz: usize = 0;
    if (acc.get_region_info)(region, HSA_REGION_INFO_SIZE, &mut sz as *mut _ as *mut c_void)
        == HSA_STATUS_SUCCESS
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
    agent: HsaAgent,
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
            return Err(RuntimeError::Device(format!("hsa_init failed: {rc}")));
        }

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
            return Err(RuntimeError::Device(format!("hsa_iterate_agents: {rc}")));
        }
        if acc.gpus.is_empty() {
            return Err(RuntimeError::Device("no GPU agents found".into()));
        }
        let cpu_agent = acc.cpu.ok_or_else(|| {
            RuntimeError::Device("no CPU agent found".into())
        })?;
        if (device_ordinal as usize) >= acc.gpus.len() {
            return Err(RuntimeError::Device(format!(
                "device ordinal {} >= {} GPU agents",
                device_ordinal,
                acc.gpus.len()
            )));
        }
        let agent = acc.gpus[device_ordinal as usize];

        // Query device name.
        let mut name_buf = [0u8; 64];
        let rc = unsafe {
            (drv.hsa_agent_get_info)(agent, HSA_AGENT_INFO_NAME, name_buf.as_mut_ptr() as *mut c_void)
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
            (drv.hsa_agent_iterate_regions)(agent, region_trampoline, &mut reg_acc as *mut _ as *mut c_void)
        };
        let lds_bytes = reg_acc.lds_bytes;

        // Find coarse-grained VRAM pool on this GPU.
        let vram_pool = Self::find_pool(&drv, agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)?;
        // Find fine-grained system pool on CPU agent.
        let fine_pool = Self::find_pool(&drv, cpu_agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)?;
        // Find kernarg pool on CPU agent.
        let kernarg_pool = Self::find_pool(&drv, cpu_agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)?;

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
            return Err(RuntimeError::Device(format!("hsa_queue_create: {rc}")));
        }

        // Create completion signal.
        let mut done_signal = HsaSignal { handle: 0 };
        let rc = unsafe {
            (drv.hsa_signal_create)(0, 0, std::ptr::null(), &mut done_signal)
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (drv.hsa_queue_destroy)(queue); }
            return Err(RuntimeError::Device(format!("hsa_signal_create: {rc}")));
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
            return Err(RuntimeError::Device(format!("kernarg ring alloc: {rc}")));
        }
        // Allow GPU agent access to the kernarg ring.
        let rc = unsafe {
            (drv.hsa_amd_agents_allow_access)(1, &agent, std::ptr::null(), karg_ring)
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (drv.hsa_amd_memory_pool_free)(karg_ring);
                (drv.hsa_signal_destroy)(done_signal);
                (drv.hsa_queue_destroy)(queue);
            }
            return Err(RuntimeError::Device(format!("kernarg allow_access: {rc}")));
        }

        let shared = Arc::new(SharedDriver { drv });

        // CDNA (gfx8xx, gfx9xx) is wave64; RDNA (gfx10xx, gfx11xx) is wave32.
        let wave_width = if device_name.starts_with("gfx9")
            || device_name.starts_with("gfx8")
        {
            64
        } else {
            32
        };

        Ok(HsaBackend {
            shared,
            device_ordinal,
            agent,
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
        })
    }

    fn find_pool(drv: &HsaDriver, agent: HsaAgent, want_flag: u32) -> Result<HsaMemoryPool> {
        let mut acc = PoolAccum {
            get_pool_info: drv.hsa_amd_memory_pool_get_info,
            want_flag,
            result: None,
        };
        let rc = unsafe {
            (drv.hsa_amd_agent_iterate_memory_pools)(agent, pool_trampoline, &mut acc as *mut _ as *mut c_void)
        };
        if rc != HSA_STATUS_SUCCESS {
            return Err(RuntimeError::Device(format!(
                "hsa_amd_agent_iterate_memory_pools: {rc}"
            )));
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
            return Err(RuntimeError::Device(format!(
                "hsa_amd_memory_pool_allocate({} bytes): {rc}",
                bytes
            )));
        }
        let free = Arc::new(HsaFree {
            shared: self.shared.clone(),
        });
        Ok(DeviceMem::owned(ptr as u64, bytes, free))
    }

    fn upload(&self, dst: &DeviceMem, off: u64, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
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
        if rc != HSA_STATUS_SUCCESS {
            return Err(RuntimeError::Device(format!("hsa_amd_memory_lock (upload): {rc}")));
        }
        // Async copy with a one-shot signal for synchronization.
        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe { (self.shared.drv.hsa_signal_create)(1, 0, std::ptr::null(), &mut sig) };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (self.shared.drv.hsa_amd_memory_unlock)(src.as_ptr() as *mut c_void); }
            return Err(RuntimeError::Device(format!("hsa_signal_create (upload): {rc}")));
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
            return Err(RuntimeError::Device(format!("hsa_amd_memory_async_copy (H2D): {rc}")));
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
        if rc != HSA_STATUS_SUCCESS {
            return Err(RuntimeError::Device(format!("hsa_amd_memory_lock (download): {rc}")));
        }
        let mut sig = HsaSignal { handle: 0 };
        let rc = unsafe { (self.shared.drv.hsa_signal_create)(1, 0, std::ptr::null(), &mut sig) };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (self.shared.drv.hsa_amd_memory_unlock)(dst.as_mut_ptr() as *mut c_void); }
            return Err(RuntimeError::Device(format!("hsa_signal_create (download): {rc}")));
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
            return Err(RuntimeError::Device(format!("hsa_amd_memory_async_copy (D2H): {rc}")));
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
        // Create a code-object reader from the in-memory ELF.
        let mut reader = HsaCodeObjectReader { handle: 0 };
        let rc = unsafe {
            (self.shared.drv.hsa_code_object_reader_create_from_memory)(
                image.as_ptr() as *const c_void,
                image.len(),
                &mut reader,
            )
        };
        if rc != HSA_STATUS_SUCCESS {
            return Err(RuntimeError::Device(format!("hsa_code_object_reader_create: {rc}")));
        }

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
            unsafe { (self.shared.drv.hsa_code_object_reader_destroy)(reader); }
            return Err(RuntimeError::Device(format!("hsa_executable_create_alt: {rc}")));
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
            return Err(RuntimeError::Device(format!(
                "hsa_executable_load_agent_code_object: {rc} (raw ELF expected — did you unbundle?)"
            )));
        }

        // Freeze.
        let rc = unsafe { (self.shared.drv.hsa_executable_freeze)(exe, std::ptr::null()) };
        if rc != HSA_STATUS_SUCCESS {
            unsafe {
                (self.shared.drv.hsa_executable_destroy)(exe);
                (self.shared.drv.hsa_code_object_reader_destroy)(reader);
            }
            return Err(RuntimeError::Device(format!("hsa_executable_freeze: {rc}")));
        }

        unsafe { (self.shared.drv.hsa_code_object_reader_destroy)(reader); }

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

    fn alloc_counter_region(&self, count: usize) -> Result<DeviceMem> {
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
            return Err(RuntimeError::Device(format!(
                "counter region alloc (fine-grained, {} bytes): {rc}",
                bytes
            )));
        }
        // Allow GPU agent access.
        let rc = unsafe {
            (self.shared.drv.hsa_amd_agents_allow_access)(1, &self.agent, std::ptr::null(), ptr)
        };
        if rc != HSA_STATUS_SUCCESS {
            unsafe { (self.shared.drv.hsa_amd_memory_pool_free)(ptr); }
            return Err(RuntimeError::Device(format!(
                "counter region allow_access: {rc}"
            )));
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

impl HsaBackend {
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
            return Err(RuntimeError::Device(format!(
                "hsa_executable_get_symbol_by_name('{name}'): {rc}"
            )));
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
        let q = self.queue;
        let idx = unsafe { (self.shared.drv.hsa_queue_add_write_index_screlease)(q, 1) };

        // Spin until ring has space.
        let size = unsafe { (*q).size } as u64;
        while idx.wrapping_sub(unsafe { (self.shared.drv.hsa_queue_load_read_index_scacquire)(q) })
            >= size
        {}

        let slot = (idx & (size - 1)) as u32;
        let karg = unsafe { self.karg_ring.add(slot as usize * KARG_SLOT) };

        // Copy explicit args.
        if args_size > 0 && !args.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(args as *const u8, karg, args_size);
                std::ptr::write_bytes(karg.add(args_size), 0, kernel.kernarg_size as usize - args_size);
            }
        } else {
            unsafe { std::ptr::write_bytes(karg, 0, kernel.kernarg_size as usize); }
        }

        // Fill COv5 implicit block (blockDim, gridDim, remainders).
        let hoff = (args_size + 7) & !7;
        if (kernel.kernarg_size as usize) > hoff {
            let hid = unsafe { karg.add(hoff) };
            let avail = kernel.kernarg_size as usize - hoff;
            let dims: u16 = if grid_z > 1 { 3 } else if grid_y > 1 { 2 } else { 1 };
            macro_rules! put32 { ($off:expr, $val:expr) => {
                if avail >= $off + 4 { unsafe { std::ptr::write_unaligned(hid.add($off) as *mut u32, $val); } }
            }}
            macro_rules! put16 { ($off:expr, $val:expr) => {
                if avail >= $off + 2 { unsafe { std::ptr::write_unaligned(hid.add($off) as *mut u16, $val); } }
            }}
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
        unsafe { (self.shared.drv.hsa_signal_add_screlease)(self.done_signal, 1); }

        // Publish the packet: one release store of header|setup.
        let dims: u16 = if grid_z > 1 { 3 } else if grid_y > 1 { 2 } else { 1 };
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
