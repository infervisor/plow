//! Software-Defined Cache Pseudo-Locking SRAM for Intel Xeon (AMX & AVX-512).
//!
//! Provides deterministic on-chip SRAM regions by partitioning L2 and L3 caches via
//! Intel Resource Director Technology (RDT) Cache Allocation Technology (CAT).
//!
//! Exposes:
//! - [`SramDevice`]: Interfacing with `/dev/pseudo_lock_l2` and `/dev/pseudo_lock_l3`.
//! - [`SramBuffer`]: Safe memory-mapped zero-eviction SRAM buffer.
//! - [`SramPool`]: Software-defined allocator for AMX worker scratchpads, KV-cache, and hot weights.
//! - [`SramBench`]: Pure Rust latency and eviction-resistance benchmark engine.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::{Result, RuntimeError};

/// Intel RDT MSRs
pub const MSR_MISC_FEATURE_CONTROL: u32 = 0x000001a4;
pub const MSR_IA32_PQR_ASSOC: u32 = 0x00000c8f;
pub const MSR_IA32_L3_QOS_MASK_BASE: u32 = 0x00000c90;
pub const MSR_IA32_L2_QOS_MASK_BASE: u32 = 0x00000d10;

/// Prefetcher disable mask (MSR 0x1A4)
pub const PREFETCH_DISABLE_ALL: u64 = 0x0000000f;

/// Pseudo-lock IOCTL definitions
const PSEUDO_LOCK_IOC_MAGIC: u8 = b'P';

// Linux _IOR('P', 1, struct pseudo_lock_info)
// size = 32 bytes (4 + 4 + 8 + 4 + 4 + 8)
const PSEUDO_LOCK_IOC_GET_INFO: libc::c_ulong =
    (2u64 << 30) | ((PSEUDO_LOCK_IOC_MAGIC as u64) << 8) | 1 | (32u64 << 16);

// Linux _IOWR('P', 2, struct pseudo_lock_latency)
// size = 56 bytes (7 * 8)
const PSEUDO_LOCK_IOC_MEASURE: libc::c_ulong =
    (3u64 << 30) | ((PSEUDO_LOCK_IOC_MAGIC as u64) << 8) | 2 | (56u64 << 16);

// Linux _IO('P', 3)
const PSEUDO_LOCK_IOC_RELOAD: libc::c_ulong =
    (0u64 << 30) | ((PSEUDO_LOCK_IOC_MAGIC as u64) << 8) | 3 | (0u64 << 16);

/// Hardware topology and reservation parameters reported by the kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PseudoLockInfo {
    pub level: u32,
    pub cpu: u32,
    pub size: u64,
    pub cbm: u32,
    pub line_size: u32,
    pub phys_addr: u64,
}

/// Latency and residency metrics reported by the kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PseudoLockLatency {
    pub min_cycles: u64,
    pub avg_cycles: u64,
    pub max_cycles: u64,
    pub total_lines: u64,
    pub l1_l2_hits: u64,
    pub l3_hits: u64,
    pub dram_misses: u64,
}

/// A handle to an open pseudo-locked SRAM character device.
pub struct SramDevice {
    file: File,
    pub info: PseudoLockInfo,
    path: String,
}

impl SramDevice {
    /// Open an existing pseudo-lock character device (`/dev/pseudo_lock_l2` or `/dev/pseudo_lock_l3`).
    pub fn open(path: &str) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| RuntimeError::Device(format!("Failed to open {path}: {e}")))?;

        let mut info = PseudoLockInfo::default();
        let ret = unsafe { libc::ioctl(file.as_raw_fd(), PSEUDO_LOCK_IOC_GET_INFO, &mut info) };
        if ret < 0 {
            return Err(RuntimeError::Device(format!(
                "ioctl GET_INFO failed on {path}: {}",
                std::io::Error::last_os_error()
            )));
        }

        Ok(SramDevice {
            file,
            info,
            path: path.to_string(),
        })
    }

    /// Query current in-kernel latency distribution.
    pub fn measure(&self) -> Result<PseudoLockLatency> {
        let mut lat = PseudoLockLatency::default();
        let ret =
            unsafe { libc::ioctl(self.file.as_raw_fd(), PSEUDO_LOCK_IOC_MEASURE, &mut lat) };
        if ret < 0 {
            return Err(RuntimeError::Device(format!(
                "ioctl MEASURE failed on {}: {}",
                self.path,
                std::io::Error::last_os_error()
            )));
        }
        Ok(lat)
    }

    /// Trigger in-kernel cache line warming and locking.
    pub fn reload(&self) -> Result<()> {
        let ret = unsafe { libc::ioctl(self.file.as_raw_fd(), PSEUDO_LOCK_IOC_RELOAD, 0) };
        if ret < 0 {
            return Err(RuntimeError::Device(format!(
                "ioctl RELOAD failed on {}: {}",
                self.path,
                std::io::Error::last_os_error()
            )));
        }
        Ok(())
    }

    /// Memory map the full pseudo-locked SRAM buffer into userspace address space.
    pub fn mmap(&self) -> Result<SramBuffer> {
        let size = self.info.size as usize;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(RuntimeError::Oom(format!(
                "Failed to mmap {} ({} bytes): {}",
                self.path,
                size,
                std::io::Error::last_os_error()
            )));
        }

        let non_null = NonNull::new(ptr as *mut u8)
            .ok_or_else(|| RuntimeError::Oom("mmap returned null".to_string()))?;

        Ok(SramBuffer {
            ptr: non_null,
            size,
            level: self.info.level,
            fd: self.file.as_raw_fd(),
        })
    }
}

/// Memory-mapped SRAM buffer backed by hardware cache ways.
pub struct SramBuffer {
    ptr: NonNull<u8>,
    size: usize,
    level: u32,
    fd: RawFd,
}

unsafe impl Send for SramBuffer {}
unsafe impl Sync for SramBuffer {}

impl SramBuffer {
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    #[inline]
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Slice a 64-byte aligned sub-region from this SRAM buffer.
    pub fn slice(&self, offset: usize, len: usize) -> Result<*mut u8> {
        if offset + len > self.size {
            return Err(RuntimeError::Oom(format!(
                "SRAM slice out of bounds: offset={offset} len={len} capacity={}",
                self.size
            )));
        }
        if offset % 64 != 0 {
            return Err(RuntimeError::Device(format!(
                "SRAM slice offset {offset} is not 64-byte aligned"
            )));
        }
        Ok(unsafe { self.ptr.as_ptr().add(offset) })
    }

    /// Refresh and lock cache lines in this buffer.
    pub fn reload(&self) -> Result<()> {
        let ret = unsafe { libc::ioctl(self.fd, PSEUDO_LOCK_IOC_RELOAD, 0) };
        if ret < 0 {
            return Err(RuntimeError::Device(format!(
                "ioctl RELOAD failed on SRAM buffer: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(())
    }
}

impl Drop for SramBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.size);
        }
    }
}

/// Detect hardware cache sizes (L2 and L3) across any Intel CPU model via Linux sysfs.
/// Returns (l2_bytes, l3_bytes).
pub fn detect_intel_cache_sizes() -> (usize, usize) {
    let parse_cache = |index: u32| -> usize {
        let path = format!("/sys/devices/system/cpu/cpu0/cache/index{index}/size");
        if let Ok(s) = std::fs::read_to_string(path) {
            let s = s.trim();
            if let Some(kib) = s.strip_suffix('K') {
                return kib.parse::<usize>().unwrap_or(0) * 1024;
            }
            if let Some(mib) = s.strip_suffix('M') {
                return mib.parse::<usize>().unwrap_or(0) * 1024 * 1024;
            }
        }
        0
    };
    let l2 = parse_cache(2);
    let l3 = parse_cache(3);
    (l2, l3)
}

/// Software-defined SRAM pool managing both L2 and L3 on-chip SRAM partitions.
pub struct SramPool {
    #[allow(dead_code)]
    pub l2_dev: Option<SramDevice>,
    pub l2_buf: Option<SramBuffer>,
    #[allow(dead_code)]
    pub l3_dev: Option<SramDevice>,
    pub l3_buf: Option<SramBuffer>,
    l2_alloc_cursor: Mutex<usize>,
    l3_alloc_cursor: Mutex<usize>,
    pub hw_l2_cache_bytes: usize,
    pub hw_l3_cache_bytes: usize,
}

impl SramPool {
    /// Initialize the SRAM pool by detecting available pseudo-locked character devices.
    pub fn init() -> Self {
        let (hw_l2, hw_l3) = detect_intel_cache_sizes();

        let (l2_dev, l2_buf) = match SramDevice::open("/dev/pseudo_lock_l2") {
            Ok(dev) => match dev.mmap() {
                Ok(buf) => (Some(dev), Some(buf)),
                Err(_) => (Some(dev), None),
            },
            Err(_) => (None, None),
        };

        let (l3_dev, l3_buf) = match SramDevice::open("/dev/pseudo_lock_l3") {
            Ok(dev) => match dev.mmap() {
                Ok(buf) => (Some(dev), Some(buf)),
                Err(_) => (Some(dev), None),
            },
            Err(_) => (None, None),
        };

        SramPool {
            l2_dev,
            l2_buf,
            l3_dev,
            l3_buf,
            l2_alloc_cursor: Mutex::new(0),
            l3_alloc_cursor: Mutex::new(0),
            hw_l2_cache_bytes: hw_l2,
            hw_l3_cache_bytes: hw_l3,
        }
    }

    /// Check if L2 Pseudo-Lock SRAM is active.
    pub fn has_l2(&self) -> bool {
        self.l2_buf.is_some()
    }

    /// Check if L3 Pseudo-Lock SRAM is active.
    pub fn has_l3(&self) -> bool {
        self.l3_buf.is_some()
    }

    /// Total L2 SRAM capacity in bytes (e.g. 1,792 KiB to 1,920 KiB).
    pub fn l2_capacity(&self) -> usize {
        self.l2_buf.as_ref().map_or(0, |b| b.len())
    }

    /// Total L3 SRAM capacity in bytes (e.g. 30 MiB to 450 MiB).
    pub fn l3_capacity(&self) -> usize {
        self.l3_buf.as_ref().map_or(0, |b| b.len())
    }

    /// Remaining available bytes in L2 SRAM.
    pub fn l2_available(&self) -> usize {
        let cursor = *self.l2_alloc_cursor.lock().unwrap();
        self.l2_capacity().saturating_sub(cursor)
    }

    /// Remaining available bytes in L3 SRAM.
    pub fn l3_available(&self) -> usize {
        let cursor = *self.l3_alloc_cursor.lock().unwrap();
        self.l3_capacity().saturating_sub(cursor)
    }

    /// Allocate dedicated scratchpad memory for an AMX/AVX-512 worker from L2 SRAM.
    pub fn alloc_l2_scratch(&self, bytes: usize) -> Result<*mut u8> {
        let Some(buf) = self.l2_buf.as_ref() else {
            return Err(RuntimeError::Device("L2 SRAM device not active".to_string()));
        };

        let aligned_bytes = bytes.checked_next_multiple_of(64).unwrap_or(bytes);
        let mut cursor = self.l2_alloc_cursor.lock().unwrap();
        if *cursor + aligned_bytes > buf.len() {
            return Err(RuntimeError::Oom(format!(
                "L2 SRAM exhausted: requested {aligned_bytes} B, available {} B",
                buf.len().saturating_sub(*cursor)
            )));
        }

        let ptr = buf.slice(*cursor, aligned_bytes)?;
        *cursor += aligned_bytes;
        Ok(ptr)
    }

    /// Allocate tensor or KV-cache memory from L3 SRAM.
    pub fn alloc_l3(&self, bytes: usize) -> Result<*mut u8> {
        let Some(buf) = self.l3_buf.as_ref() else {
            return Err(RuntimeError::Device("L3 SRAM device not active".to_string()));
        };

        let aligned_bytes = bytes.checked_next_multiple_of(64).unwrap_or(bytes);
        let mut cursor = self.l3_alloc_cursor.lock().unwrap();
        if *cursor + aligned_bytes > buf.len() {
            return Err(RuntimeError::Oom(format!(
                "L3 SRAM exhausted: requested {aligned_bytes} B, available {} B",
                buf.len().saturating_sub(*cursor)
            )));
        }

        let ptr = buf.slice(*cursor, aligned_bytes)?;
        *cursor += aligned_bytes;
        Ok(ptr)
    }

    /// Software-defined worker scratch allocation: tries L2 SRAM first, falls back to L3 SRAM.
    pub fn alloc_scratch(&self, bytes: usize) -> Option<*mut u8> {
        if let Ok(ptr) = self.alloc_l2_scratch(bytes) {
            return Some(ptr);
        }
        if let Ok(ptr) = self.alloc_l3(bytes) {
            return Some(ptr);
        }
        None
    }

    /// Software-defined hot runtime tensor allocation (KV cache, intermediate activations).
    pub fn alloc_hot_tensor(&self, bytes: usize) -> Option<*mut u8> {
        self.alloc_l3(bytes).ok()
    }

    /// Reset SRAM allocation cursors (e.g. between independent model runs).
    pub fn reset(&self) {
        *self.l2_alloc_cursor.lock().unwrap() = 0;
        *self.l3_alloc_cursor.lock().unwrap() = 0;
    }
}

static GLOBAL_SRAM: std::sync::OnceLock<SramPool> = std::sync::OnceLock::new();

/// Global manager for Intel RDT Pseudo-Locked SRAM.
pub struct SramManager;

impl SramManager {
    /// Access the global SRAM pool singleton, dynamically detecting /dev/pseudo_lock_*
    /// and hardware cache hierarchy on any Intel CPU.
    pub fn global() -> &'static SramPool {
        GLOBAL_SRAM.get_or_init(SramPool::init)
    }
}

/// Pure Rust benchmark engine replicating in-kernel and userspace latency verification.
pub struct SramBench;

impl SramBench {
    /// Read hardware timestamp counter with memory fence serialization.
    #[inline(always)]
    pub fn rdtsc_fence() -> u64 {
        unsafe {
            core::arch::x86_64::_mm_lfence();
            let t = core::arch::x86_64::_rdtsc();
            core::arch::x86_64::_mm_lfence();
            t
        }
    }

    /// Pin the calling thread to a specific CPU.
    pub fn pin_thread(cpu: usize) -> Result<()> {
        unsafe {
            let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut cpuset);
            let ret = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
            if ret != 0 {
                return Err(RuntimeError::Device(format!(
                    "sched_setaffinity to CPU {cpu} failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
        }
        Ok(())
    }

    /// Verify 100% data read/write integrity across the SRAM buffer.
    pub fn verify_integrity(buf: &SramBuffer) -> bool {
        let words = buf.len() / std::mem::size_of::<u64>();
        let ptr64 = buf.as_mut_ptr() as *mut u64;

        // Pattern write pass
        for i in 0..words {
            unsafe {
                ptr64.add(i).write_volatile(0xcafebabe00000000 | (i as u64));
            }
        }

        // Verification read pass
        for i in 0..words {
            let val = unsafe { ptr64.add(i).read_volatile() };
            if val != (0xcafebabe00000000 | (i as u64)) {
                return false;
            }
        }

        true
    }

    /// Measure stride access latency and hit distribution across all 64-byte cache lines.
    pub fn measure_latency(buf: &SramBuffer) -> (f64, u64, u64, u32, u32, u32) {
        let num_lines = buf.len() / 64;
        let ptr = buf.as_ptr();

        let mut total_cycles: u64 = 0;
        let mut min_c: u64 = u64::MAX;
        let mut max_c: u64 = 0;
        let mut fast_hits: u32 = 0; // <= 30 cycles
        let mut l3_hits: u32 = 0;   // 31..150 cycles
        let mut misses: u32 = 0;    // > 150 cycles

        for i in 0..num_lines {
            let t0 = Self::rdtsc_fence();
            let val = unsafe { (ptr.add(i * 64) as *const u32).read_volatile() };
            let t1 = Self::rdtsc_fence();
            core::hint::black_box(val);

            let diff = if t1 > t0 { t1 - t0 } else { 1 };
            if diff < min_c {
                min_c = diff;
            }
            if diff > max_c {
                max_c = diff;
            }
            total_cycles += diff;

            if diff <= 30 {
                fast_hits += 1;
            } else if diff <= 150 {
                l3_hits += 1;
            } else {
                misses += 1;
            }
        }

        let avg = total_cycles as f64 / num_lines.max(1) as f64;
        (avg, min_c, max_c, fast_hits, l3_hits, misses)
    }

    /// Test eviction resistance under active background memory thrashing by N threads.
    pub fn measure_under_contention(
        buf: &SramBuffer,
        target_cpu: usize,
        num_stress_threads: usize,
    ) -> (f64, f64) {
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        for i in 0..num_stress_threads {
            let stop_clone = Arc::clone(&stop);
            let cpu = i + 1; // Stress on different logical cores
            let handle = thread::spawn(move || {
                let _ = Self::pin_thread(cpu);
                let sz = 64 * 1024 * 1024; // 64 MB churn buffer
                let mut chunk = vec![0x5au8; sz];
                while !stop_clone.load(Ordering::Relaxed) {
                    for j in (0..sz).step_by(64) {
                        chunk[j] = chunk[j].wrapping_add(1);
                    }
                }
            });
            handles.push(handle);
        }

        // 200ms burn-in for contention to peak
        thread::sleep(Duration::from_millis(200));

        let _ = Self::pin_thread(target_cpu);
        let (avg_cyc, _, _, fast_hits, l3_hits, _) = Self::measure_latency(buf);
        let num_lines = buf.len() / 64;
        let residency = (fast_hits + l3_hits) as f64 * 100.0 / num_lines.max(1) as f64;

        stop.store(true, Ordering::Relaxed);
        for h in handles {
            let _ = h.join();
        }

        (avg_cyc, residency)
    }
}
