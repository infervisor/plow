//! Page-locked host memory that devices of different vendors copy through.
//!
//! Devices of two vendors share no fabric this runtime relies on, so bytes cross between them
//! through host memory: one vendor's copy engine writes a region, the other's reads it. A
//! [`HostRegion`] is a `memfd` mapping (so a rank-worker process can map the same pages),
//! aligned to 2 MiB, registered once with every driver that copies through it. Each driver
//! hands back its own address for the pages (HSA's locked-memory agent pointer need not equal
//! the host pointer), and a copy always names the address of the driver that issues it.
//!
//! Completion never crosses vendors through device atomics on host memory: the issuing host
//! thread waits on its own driver's fence before anyone reads the bytes.

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use hwspec::Vendor;

use crate::{Result, RuntimeError};

/// The host-staging surface of one device's driver.
pub trait HostStaging: Send + Sync {
    fn staging_vendor(&self) -> Vendor;
    /// Register `[ptr, ptr + len)` for DMA by every device of this driver in the process; returns
    /// the address this driver's kernels dereference and its staging copies take for `ptr`.
    ///
    /// # Safety
    /// `ptr..ptr + len` must stay mapped until [`Self::host_unregister`].
    unsafe fn host_register(&self, ptr: *mut u8, len: usize) -> Result<u64>;
    /// # Safety
    /// `ptr` was registered by [`Self::host_register`] on this driver and no copy is in flight.
    unsafe fn host_unregister(&self, ptr: *mut u8);
    /// Queue `bytes` from device address `src` into registered host address `host`; returns a
    /// fence for [`Self::stage_wait`].
    fn stage_d2h(&self, host: u64, src: u64, bytes: u64) -> Result<u64>;
    /// Queue `bytes` from registered host address `host` into device address `dst`.
    fn stage_h2d(&self, dst: u64, host: u64, bytes: u64) -> Result<u64>;
    fn stage_wait(&self, fence: u64) -> Result<()>;
}

const ALIGN: usize = 2 << 20;

/// A shared, page-locked host region registered with one or more drivers.
pub struct HostRegion {
    _fd: OwnedFd,
    base: *mut u8,
    len: usize,
    reservation: (*mut u8, usize),
    /// One registration per vendor: (driver, its address for `base`).
    regs: Vec<(Arc<dyn HostStaging>, u64)>,
}

// SAFETY: the mapping is addressed only through raw pointers whose cross-thread use is ordered
// by the drivers' fences, which the callers wait on.
unsafe impl Send for HostRegion {}
unsafe impl Sync for HostRegion {}

fn os_err(what: &str) -> RuntimeError {
    RuntimeError::Device(format!("{what}: {}", std::io::Error::last_os_error()))
}

impl HostRegion {
    /// A zero-filled region of `len` bytes (rounded up to 2 MiB), populated and 2 MiB aligned.
    pub fn new(len: usize) -> Result<Self> {
        let len = len.max(1).div_ceil(ALIGN) * ALIGN;
        // SAFETY: plain syscalls; every result is checked and every error path releases what
        // was created before it.
        unsafe {
            let fd = libc::memfd_create(c"plow-host-region".as_ptr(), libc::MFD_CLOEXEC);
            if fd < 0 {
                return Err(os_err("memfd_create"));
            }
            let fd = OwnedFd::from_raw_fd(fd);
            if libc::ftruncate(std::os::fd::AsRawFd::as_raw_fd(&fd), len as libc::off_t) != 0 {
                return Err(os_err("ftruncate"));
            }
            let res_len = len + ALIGN;
            let res = libc::mmap(
                std::ptr::null_mut(),
                res_len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            );
            if res == libc::MAP_FAILED {
                return Err(os_err("mmap (reserve)"));
            }
            let aligned = ((res as usize).div_ceil(ALIGN) * ALIGN) as *mut libc::c_void;
            let p = libc::mmap(
                aligned,
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED | libc::MAP_POPULATE,
                std::os::fd::AsRawFd::as_raw_fd(&fd),
                0,
            );
            if p == libc::MAP_FAILED {
                libc::munmap(res, res_len);
                return Err(os_err("mmap (region)"));
            }
            Ok(HostRegion {
                _fd: fd,
                base: p as *mut u8,
                len,
                reservation: (res as *mut u8, res_len),
                regs: Vec::new(),
            })
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Register with `driver` unless a driver of the same vendor already did.
    pub fn register(&mut self, driver: Arc<dyn HostStaging>) -> Result<u64> {
        let vendor = driver.staging_vendor();
        if let Some(alias) = self.alias(vendor) {
            return Ok(alias);
        }
        // SAFETY: the region stays mapped until `Drop`, which unregisters first.
        let alias = unsafe { driver.host_register(self.base, self.len)? };
        self.regs.push((driver, alias));
        Ok(alias)
    }

    /// The address `vendor`'s copies use for offset 0, if registered.
    pub fn alias(&self, vendor: Vendor) -> Option<u64> {
        self.regs
            .iter()
            .find(|(d, _)| d.staging_vendor() == vendor)
            .map(|&(_, a)| a)
    }

    /// # Safety
    /// No device copy into or out of `off..off + len` may be in flight.
    pub unsafe fn slice(&self, off: usize, len: usize) -> &[u8] {
        assert!(off + len <= self.len);
        std::slice::from_raw_parts(self.base.add(off), len)
    }

    /// # Safety
    /// As [`Self::slice`], and no other reference to the range may be live.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut(&self, off: usize, len: usize) -> &mut [u8] {
        assert!(off + len <= self.len);
        std::slice::from_raw_parts_mut(self.base.add(off), len)
    }
}

impl Drop for HostRegion {
    fn drop(&mut self) {
        for (d, _) in self.regs.drain(..) {
            // SAFETY: registered in `register`; owners wait on their fences before dropping.
            unsafe { d.host_unregister(self.base) };
        }
        // SAFETY: the region mapping sits inside the reservation; unmapping the reservation
        // releases both.
        unsafe { libc::munmap(self.reservation.0 as *mut libc::c_void, self.reservation.1) };
    }
}

fn pattern(seed: u64, out: &mut [u8]) {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for chunk in out.chunks_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        chunk.copy_from_slice(&x.to_le_bytes()[..chunk.len()]);
    }
}

/// One device's result in [`xfer_check`].
pub struct XferResult {
    pub device: String,
    pub h2d_gbs: f64,
    pub d2h_gbs: f64,
}

/// The cross-vendor staging gate: one region registered with every driver; every device runs
/// `iters` H2D + D2H round trips of `bytes` through it concurrently and its bytes are compared;
/// then every ordered pair of devices relays a pattern device → region → device → region (the
/// prefill-to-decode handoff path) and the result is compared byte for byte.
pub fn xfer_check(
    devs: &[super::Device],
    bytes: usize,
    iters: u32,
) -> Result<(Vec<XferResult>, Vec<String>)> {
    let n = devs.len();
    if n == 0 {
        return Err(RuntimeError::Rejected("xfer check: no devices".into()));
    }
    let bytes = bytes.div_ceil(4096) * 4096;
    let mut region = HostRegion::new(bytes * (2 * n + 2))?;
    let mut aliases = Vec::with_capacity(n);
    for d in devs {
        aliases.push(region.register(d.handle.staging())?);
    }
    for i in 0..n {
        // SAFETY: no copy is in flight yet.
        pattern(i as u64 + 1, unsafe {
            region.slice_mut(2 * i * bytes, bytes)
        });
    }
    let bufs: Vec<_> = devs
        .iter()
        .map(|d| d.handle.backend().alloc(0, bytes as u64))
        .collect::<Result<_>>()?;
    let out = transfers(devs, bytes, iters, &region, &aliases, &bufs);
    if out.is_err() {
        // A failed fence does not prove the copy stopped: the region and the device buffers
        // stay mapped and allocated for the life of the process rather than be reused under DMA.
        std::mem::forget(region);
        std::mem::forget(bufs);
    }
    out
}

fn transfers(
    devs: &[super::Device],
    bytes: usize,
    iters: u32,
    region: &HostRegion,
    aliases: &[u64],
    bufs: &[super::DeviceMem],
) -> Result<(Vec<XferResult>, Vec<String>)> {
    use std::time::Instant;
    let n = devs.len();
    let start = std::sync::Barrier::new(n);
    let start = &start;
    let results: Vec<Result<XferResult>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..n)
            .map(|i| {
                let st = devs[i].handle.staging();
                let dbuf = bufs[i].base;
                let src = aliases[i] + (2 * i * bytes) as u64;
                let dst = aliases[i] + ((2 * i + 1) * bytes) as u64;
                let name = devs[i].id.to_string();
                s.spawn(move || -> Result<XferResult> {
                    // Every device's copies overlap: no thread starts before all are up.
                    start.wait();
                    let (mut h2d, mut d2h) = (0f64, 0f64);
                    for _ in 0..iters.max(1) {
                        let t = Instant::now();
                        st.stage_wait(st.stage_h2d(dbuf, src, bytes as u64)?)?;
                        h2d += t.elapsed().as_secs_f64();
                        let t = Instant::now();
                        st.stage_wait(st.stage_d2h(dst, dbuf, bytes as u64)?)?;
                        d2h += t.elapsed().as_secs_f64();
                    }
                    // SAFETY: every copy of this device was waited on above.
                    let ok = unsafe {
                        region.slice(2 * i * bytes, bytes)
                            == region.slice((2 * i + 1) * bytes, bytes)
                    };
                    if !ok {
                        return Err(RuntimeError::Device(format!(
                            "xfer check: {name} round trip corrupted bytes"
                        )));
                    }
                    let gb = bytes as f64 * iters.max(1) as f64 / 1e9;
                    Ok(XferResult {
                        device: name,
                        h2d_gbs: gb / h2d,
                        d2h_gbs: gb / d2h,
                    })
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("xfer thread panicked"))
            .collect()
    });
    let results = results.into_iter().collect::<Result<Vec<_>>>()?;

    let mid = 2 * n * bytes;
    let out = mid + bytes;
    let mut relays = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            let (sa, sb) = (devs[a].handle.staging(), devs[b].handle.staging());
            // SAFETY: every earlier copy was waited on; these are serial and waited on in turn.
            unsafe { region.slice_mut(out, bytes) }.fill(0);
            let t = std::time::Instant::now();
            sa.stage_wait(sa.stage_h2d(
                bufs[a].base,
                aliases[a] + (2 * a * bytes) as u64,
                bytes as u64,
            )?)?;
            sa.stage_wait(sa.stage_d2h(aliases[a] + mid as u64, bufs[a].base, bytes as u64)?)?;
            sb.stage_wait(sb.stage_h2d(bufs[b].base, aliases[b] + mid as u64, bytes as u64)?)?;
            sb.stage_wait(sb.stage_d2h(aliases[b] + out as u64, bufs[b].base, bytes as u64)?)?;
            let dt = t.elapsed().as_secs_f64();
            // SAFETY: all four copies were waited on.
            let ok = unsafe { region.slice(2 * a * bytes, bytes) == region.slice(out, bytes) };
            if !ok {
                return Err(RuntimeError::Device(format!(
                    "xfer check: relay {} -> {} corrupted bytes",
                    devs[a].id, devs[b].id
                )));
            }
            relays.push(format!(
                "{} -> {}: exact, 4 hops in {:.2} ms ({:.1} GB/s per hop)",
                devs[a].id,
                devs[b].id,
                dt * 1e3,
                4.0 * bytes as f64 / 1e9 / dt
            ));
        }
    }
    Ok((results, relays))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_is_aligned_rounded_and_zeroed() {
        let r = HostRegion::new(3 << 20).unwrap();
        assert_eq!(r.len(), 4 << 20);
        assert_eq!(r.base as usize % ALIGN, 0);
        let s = unsafe { r.slice(0, r.len()) };
        assert!(s.iter().all(|&b| b == 0));
        unsafe { r.slice_mut(5, 3) }.copy_from_slice(&[1, 2, 3]);
        assert_eq!(unsafe { r.slice(4, 5) }, &[0, 1, 2, 3, 0]);
    }
}
