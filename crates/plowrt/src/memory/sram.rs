//! Cache pseudo-locking through `/dev/pseudo_lock` (`runtime/cpu/driver/pseudo_lock_sram.c`).
//!
//! The driver partitions L2/L3 with Intel CAT at load and locks memory the caller owns: plowrt
//! allocates the buffer (node-bound, huge-page backed) and asks for it to be held in one core's
//! private L2 (level 2) or in its NUMA node's L3 slices (level 3). Locks live as long as this
//! process keeps the device open, which is the whole process lifetime.
//!
//! Only L2 locks hold for data the owner keeps using: the L3 is non-inclusive, so a read hit
//! from a core in the default CLOS moves the line to its L2 and its eviction refills the normal
//! ways. Level 3 is exposed for measurement, not placement.

use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::sync::{Mutex, OnceLock};

use crate::{Result, RuntimeError};

pub const DEVICE: &str = "/dev/pseudo_lock";

/// `struct pl_caps` (driver ABI v3).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Caps {
    pub version: u32,
    pub nodes: u32,
    pub l2_cbm_full: u32,
    pub l2_cbm_lock: u32,
    pub l3_cbm_full: u32,
    pub l3_cbm_lock: u32,
    pub l2_lock_bytes_per_core: u64,
    pub l3_lock_bytes_per_node: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct LockReq {
    addr: u64,
    len: u64,
    cpu: i32,
    level: u32,
    id: u32,
    pad: u32,
}

/// `struct pl_measure`: lines of a locked range by latency class, measured from its owner
/// against calibrated L2 / L3 / DRAM latencies (cycles).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Measure {
    pub id: u32,
    pad: u32,
    pub lines: u64,
    pub l1_l2: u64,
    pub l3: u64,
    pub dram: u64,
    pub p50: u64,
    pub cal_l2: u64,
    pub cal_l3: u64,
    pub cal_dram: u64,
}

impl Measure {
    /// Fraction of lines still in the locked level.
    pub fn held(&self, level: u32) -> f64 {
        let hit = if level == 2 { self.l1_l2 } else { self.l1_l2 + self.l3 };
        hit as f64 / self.lines.max(1) as f64
    }
}

const fn ioc(dir: u64, nr: u64, size: u64) -> libc::c_ulong {
    ((dir << 30) | (size << 16) | ((b'P' as u64) << 8) | nr) as libc::c_ulong
}
const IOC_CAPS: libc::c_ulong = ioc(2, 10, std::mem::size_of::<Caps>() as u64);
const IOC_LOCK: libc::c_ulong = ioc(3, 11, std::mem::size_of::<LockReq>() as u64);
const IOC_MEASURE: libc::c_ulong = ioc(3, 13, std::mem::size_of::<Measure>() as u64);
const IOC_WORKER: libc::c_ulong = ioc(1, 15, 8);

pub struct PseudoLock {
    file: File,
    pub caps: Caps,
    /// Serialises lock requests: each one runs with IRQs off on its owner for milliseconds.
    gate: Mutex<()>,
}

impl PseudoLock {
    pub fn open() -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(DEVICE)
            .map_err(|e| RuntimeError::Device(format!("{DEVICE}: {e}")))?;
        let mut caps = Caps::default();
        // SAFETY: IOC_CAPS writes exactly one `Caps`.
        if unsafe { libc::ioctl(file.as_raw_fd(), IOC_CAPS, &mut caps) } != 0 {
            return Err(RuntimeError::Device(format!(
                "{DEVICE} CAPS: {} (driver older than ABI v3?)",
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self { file, caps, gate: Mutex::new(()) })
    }

    /// The process-wide device, opened once; `None` when the driver is not loaded.
    pub fn global() -> Option<&'static PseudoLock> {
        static DEV: OnceLock<Option<PseudoLock>> = OnceLock::new();
        DEV.get_or_init(|| match Self::open() {
            Ok(d) => {
                tracing::info!(caps = ?d.caps, "pseudo-lock driver ready");
                Some(d)
            }
            Err(e) => {
                tracing::debug!(error = %e, "pseudo-lock driver unavailable");
                None
            }
        })
        .as_ref()
    }

    /// Hold `[ptr, ptr + len)` (page aligned, owned by this process for its lifetime) in `cpu`'s
    /// L2 (`level` 2) or its node's L3 (`level` 3). Returns the driver's measurement.
    ///
    /// # Safety
    /// The range must stay mapped until the process exits: the driver pins its pages.
    pub unsafe fn lock(&self, ptr: *mut u8, len: usize, cpu: u32, level: u32) -> Result<Measure> {
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        let mut req = LockReq { addr: ptr as u64, len: len as u64, cpu: cpu as i32, level, ..Default::default() };
        if unsafe { libc::ioctl(self.file.as_raw_fd(), IOC_LOCK, &mut req) } != 0 {
            return Err(RuntimeError::Device(format!(
                "pseudo-lock L{level} cpu {cpu} {len} B: {}",
                std::io::Error::last_os_error()
            )));
        }
        let mut m = Measure { id: req.id, ..Default::default() };
        if unsafe { libc::ioctl(self.file.as_raw_fd(), IOC_MEASURE, &mut m) } != 0 {
            return Err(RuntimeError::Device(format!(
                "pseudo-lock measure {}: {}",
                req.id,
                std::io::Error::last_os_error()
            )));
        }
        Ok(m)
    }

    /// Run `cpu` in the driver's worker CLOS (its L3 fills use the locked ways, so reading
    /// L3-locked data keeps it locked) until the process exits.
    pub fn set_worker(&self, cpu: u32, on: bool) -> Result<()> {
        let req: [u32; 2] = [cpu, on as u32];
        if unsafe { libc::ioctl(self.file.as_raw_fd(), IOC_WORKER, &req) } != 0 {
            return Err(RuntimeError::Device(format!(
                "pseudo-lock worker CLOS cpu {cpu}: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(())
    }
}
