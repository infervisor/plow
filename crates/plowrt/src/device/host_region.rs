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

pub(crate) fn unretired_copy(error: &RuntimeError) -> crate::DeviceErrorInfo {
    crate::DeviceErrorInfo {
        operation: "KV image staging".into(),
        code: error.device_code().unwrap_or(-1),
        name: format!("UNRETIRED_COPY: {error}"),
        fatal: true,
    }
}

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
    /// Queue disjoint image spans `(host, device, bytes)`. A partial submission still returns
    /// its last fence so the caller drains accepted copies before handling the error.
    fn stage_batch(&self, spans: &[(u64, u64, u64)], export: bool) -> (Option<u64>, Result<()>) {
        let mut last = None;
        for &(host, device, bytes) in spans {
            let result = if export {
                self.stage_d2h(host, device, bytes)
            } else {
                self.stage_h2d(device, host, bytes)
            };
            match result {
                Ok(fence) => last = Some(fence),
                Err(error) => return (last, Err(error)),
            }
        }
        (last, Ok(()))
    }
    /// Wait for and retire all copies queued by this thread through `fence` (the last fence).
    /// A batch must be submitted and waited on by the same thread.
    fn stage_wait(&self, fence: u64) -> Result<()>;
}

#[derive(Default)]
pub(crate) struct StageFences {
    pending: parking_lot::Mutex<std::collections::HashMap<std::thread::ThreadId, Vec<u64>>>,
}

impl StageFences {
    pub(crate) fn submit(&self, copy: impl FnOnce(Option<u64>) -> Result<u64>) -> Result<u64> {
        let mut pending = self.pending.lock();
        let fences = pending.entry(std::thread::current().id()).or_default();
        let fence = copy(fences.last().copied())?;
        fences.push(fence);
        Ok(fence)
    }

    pub(crate) fn wait(
        &self,
        fence: u64,
        wait: impl FnOnce(u64) -> Result<()>,
        mut retire: impl FnMut(u64) -> Result<bool>,
    ) -> Result<()> {
        let mut pending = self.pending.lock();
        let id = std::thread::current().id();
        if pending.get(&id).and_then(|f| f.last()) != Some(&fence) {
            return Err(RuntimeError::Device(
                "staging: not this thread's last fence".into(),
            ));
        }
        let mut fences = pending.remove(&id).expect("checked above");
        drop(pending);
        if let Err(error) = wait(fence) {
            self.pending.lock().insert(id, fences);
            return Err(error);
        }
        fences.pop();
        while let Some(&earlier) = fences.last() {
            match retire(earlier) {
                Ok(true) => {
                    fences.pop();
                }
                result => {
                    self.pending.lock().insert(id, fences);
                    return Err(result.err().unwrap_or_else(|| {
                        RuntimeError::Device("staging: unordered copy completion".into())
                    }));
                }
            }
        }
        Ok(())
    }
}

static IMAGE_REGIONS: parking_lot::Mutex<Vec<HostRegion>> = parking_lot::Mutex::new(Vec::new());

/// Pre-register two slabs before serving or timing a handoff. Registrations cover all devices
/// of each supplied driver. An unavailable slab never prevents a pageable image export.
pub fn prepare_image_pool(bytes: usize, drivers: &[Arc<dyn HostStaging>]) -> Result<()> {
    let mut regions = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut region = HostRegion::new(bytes)?;
        for driver in drivers {
            region.register(Arc::clone(driver))?;
        }
        regions.push(region);
    }
    *IMAGE_REGIONS.lock() = regions;
    Ok(())
}

pub struct RegionLease {
    region: Option<HostRegion>,
    len: usize,
    stream_started: Option<std::time::Instant>,
    pending: std::sync::atomic::AtomicUsize,
    export: parking_lot::Mutex<ImageExport>,
    changed: parking_lot::Condvar,
    #[cfg(test)]
    waiting: std::sync::atomic::AtomicUsize,
}

enum ImageExport {
    Idle,
    Active { ready: usize },
    Complete,
    Failed(String),
}

pub(crate) struct ImageExportGuard(Arc<RegionLease>);

impl ImageExportGuard {
    pub(crate) fn finish(self, result: Result<()>) -> Result<()> {
        let mut state = self.0.export.lock();
        let result = if result.is_ok()
            && !matches!(*state, ImageExport::Active { ready } if ready == self.0.len)
        {
            Err(RuntimeError::Device(
                "KV image export ended before its final range".into(),
            ))
        } else {
            result
        };
        *state = match &result {
            Ok(()) => ImageExport::Complete,
            Err(error) => ImageExport::Failed(error.to_string()),
        };
        self.0.changed.notify_all();
        drop(state);
        result
    }
}

impl Drop for ImageExportGuard {
    fn drop(&mut self) {
        let mut state = self.0.export.lock();
        if matches!(*state, ImageExport::Active { .. }) {
            *state = ImageExport::Failed("exporter left before completion".into());
            self.0.changed.notify_all();
        }
    }
}

impl RegionLease {
    pub(crate) fn take(len: usize, vendor: Vendor) -> Option<Self> {
        let mut pool = IMAGE_REGIONS.try_lock()?;
        let i = pool
            .iter()
            .position(|r| r.len() >= len && r.alias(vendor).is_some())?;
        Some(Self {
            region: Some(pool.swap_remove(i)),
            len,
            stream_started: None,
            pending: std::sync::atomic::AtomicUsize::new(0),
            export: parking_lot::Mutex::new(ImageExport::Idle),
            changed: parking_lot::Condvar::new(),
            #[cfg(test)]
            waiting: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub(crate) fn alias(&self, vendor: Vendor) -> Option<u64> {
        self.region.as_ref()?.alias(vendor)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn begin_dma(&self) {
        self.pending
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    pub(crate) fn retire_dma(&self) {
        assert!(
            self.pending
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel)
                > 0
        );
    }

    pub(crate) fn has_pending_dma(&self) -> bool {
        self.pending.load(std::sync::atomic::Ordering::Acquire) != 0
    }

    pub(crate) fn start_export(lease: &mut Arc<Self>) -> Option<ImageExportGuard> {
        let unique = Arc::get_mut(lease)?;
        unique.stream_started = Some(std::time::Instant::now());
        *unique.export.get_mut() = ImageExport::Active { ready: 0 };
        Some(ImageExportGuard(Arc::clone(lease)))
    }

    pub(crate) fn streaming(&self) -> bool {
        !matches!(*self.export.lock(), ImageExport::Idle)
    }

    pub(crate) fn stream_ms(&self) -> Option<f64> {
        self.stream_started.map(|t| t.elapsed().as_secs_f64() * 1e3)
    }

    pub(crate) fn stream_started(&self) -> Option<std::time::Instant> {
        self.stream_started
    }

    pub(crate) fn publish(&self, end: usize) -> Result<()> {
        let mut state = self.export.lock();
        match &mut *state {
            ImageExport::Active { ready } if *ready <= end && end <= self.len => *ready = end,
            _ => {
                return Err(RuntimeError::Device(
                    "KV image export range out of order".into(),
                ))
            }
        }
        self.changed.notify_all();
        Ok(())
    }

    pub(crate) fn wait_export(&self, end: usize, complete: bool) -> Result<()> {
        let mut state = self.export.lock();
        loop {
            match &*state {
                ImageExport::Idle | ImageExport::Complete => return Ok(()),
                ImageExport::Active { ready } if !complete && *ready >= end => return Ok(()),
                ImageExport::Failed(error) => {
                    return Err(RuntimeError::Unavailable(format!(
                        "KV image peer export failed: {error}"
                    )))
                }
                _ => {
                    #[cfg(test)]
                    self.waiting
                        .fetch_add(1, std::sync::atomic::Ordering::Release);
                    self.changed.wait(&mut state);
                    #[cfg(test)]
                    self.waiting
                        .fetch_sub(1, std::sync::atomic::Ordering::Release);
                }
            }
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        assert!(!self.has_pending_dma());
        assert!(matches!(
            *self.export.lock(),
            ImageExport::Idle | ImageExport::Complete
        ));
        // SAFETY: publication is final and every source and consumer DMA has retired.
        unsafe { self.region.as_ref().expect("live lease").slice(0, self.len) }
    }
}

impl Drop for RegionLease {
    fn drop(&mut self) {
        if let Some(region) = self.region.take() {
            if *self.pending.get_mut() == 0 {
                IMAGE_REGIONS.lock().push(region);
            } else {
                // A failed fence cannot prove DMA stopped. Never reuse or unmap these pages.
                std::mem::forget(region);
            }
        }
    }
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
    fn staging_chains_copies_and_retires_only_after_the_last_wait() {
        let batch = StageFences::default();
        batch
            .submit(|prev| {
                assert_eq!(prev, None);
                Ok(1)
            })
            .unwrap();
        batch
            .submit(|prev| {
                assert_eq!(prev, Some(1));
                Ok(2)
            })
            .unwrap();
        let waited = std::cell::Cell::new(false);
        batch
            .wait(
                2,
                |last| {
                    assert_eq!(last, 2);
                    waited.set(true);
                    Ok(())
                },
                |earlier| {
                    assert!(waited.get());
                    assert_eq!(earlier, 1);
                    Ok(true)
                },
            )
            .unwrap();
        batch
            .submit(|prev| {
                assert_eq!(prev, None);
                Ok(3)
            })
            .unwrap();
        // A rejected enqueue leaves the earlier work available to drain.
        let mut drain = None;
        assert!(batch
            .submit(|previous| {
                drain = previous;
                Err(RuntimeError::Device("copy refused".into()))
            })
            .is_err());
        assert_eq!(drain, Some(3));
        batch
            .wait(drain.unwrap(), |_| Ok(()), |_| unreachable!())
            .unwrap();
    }

    #[test]
    fn staging_retains_unretired_fences_on_wait_or_retire_error() {
        let batch = StageFences::default();
        batch.submit(|_| Ok(1)).unwrap();
        batch.submit(|_| Ok(2)).unwrap();
        assert!(batch
            .wait(
                2,
                |_| Err(RuntimeError::Device("wait failed".into())),
                |_| unreachable!()
            )
            .is_err());
        let id = std::thread::current().id();
        assert_eq!(batch.pending.lock()[&id], [1, 2]);
        assert!(batch.wait(2, |_| Ok(()), |_| Ok(false)).is_err());
        assert_eq!(batch.pending.lock()[&id], [1]);
        batch.wait(1, |_| Ok(()), |_| unreachable!()).unwrap();
        assert!(batch.pending.lock().is_empty());
    }

    use crate::exec::kv_image::{ImageSpan, KvBytes};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static IMAGE_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    struct StagingDriver {
        vendor: hwspec::Vendor,
        bias: u64,
        registrations: AtomicUsize,
        waits: AtomicUsize,
        copies: parking_lot::Mutex<Vec<(u64, u64, u64)>>,
        fail_enqueue: std::sync::atomic::AtomicBool,
        fail_wait: std::sync::atomic::AtomicBool,
        on_wait: parking_lot::Mutex<Option<Arc<dyn Fn(usize) + Send + Sync>>>,
    }
    impl HostStaging for StagingDriver {
        fn staging_vendor(&self) -> hwspec::Vendor {
            self.vendor
        }
        unsafe fn host_register(&self, ptr: *mut u8, _: usize) -> Result<u64> {
            self.registrations.fetch_add(1, Ordering::Relaxed);
            Ok(ptr as u64 + self.bias)
        }
        unsafe fn host_unregister(&self, _: *mut u8) {}
        fn stage_d2h(&self, host: u64, src: u64, bytes: u64) -> Result<u64> {
            let mut copies = self.copies.lock();
            if !copies.is_empty() && self.fail_enqueue.load(Ordering::Relaxed) {
                return Err(RuntimeError::Device("enqueue refused".into()));
            }
            copies.push((host - self.bias, src, bytes));
            Ok(copies.len() as u64)
        }
        fn stage_h2d(&self, dst: u64, host: u64, bytes: u64) -> Result<u64> {
            let mut copies = self.copies.lock();
            copies.push((dst, host - self.bias, bytes));
            Ok(copies.len() as u64)
        }
        fn stage_wait(&self, last: u64) -> Result<()> {
            let mut copies = self.copies.lock();
            assert_eq!(last, copies.len() as u64);
            let call = self.waits.fetch_add(1, Ordering::Relaxed) + 1;
            if self.fail_wait.load(Ordering::Relaxed) {
                return Err(RuntimeError::Device("wait failed".into()));
            }
            for (dst, src, bytes) in copies.drain(..) {
                // SAFETY: the test keeps both disjoint buffers alive until this wait.
                unsafe {
                    std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes as usize);
                }
            }
            drop(copies);
            let callback = self.on_wait.lock().clone();
            if let Some(callback) = callback {
                callback(call);
            }
            Ok(())
        }
    }
    fn driver(vendor: hwspec::Vendor, bias: u64) -> Arc<StagingDriver> {
        Arc::new(StagingDriver {
            vendor,
            bias,
            registrations: AtomicUsize::new(0),
            waits: AtomicUsize::new(0),
            copies: Default::default(),
            fail_enqueue: Default::default(),
            fail_wait: Default::default(),
            on_wait: Default::default(),
        })
    }

    #[test]
    fn staged_images_use_driver_aliases_wait_once_and_reuse_leases() {
        let _serial = IMAGE_TEST_LOCK.lock();

        let nv = driver(hwspec::Vendor::Nvidia, 0);
        let amd = driver(hwspec::Vendor::Amd, 4096);
        prepare_image_pool(16, &[nv.clone(), amd.clone()]).unwrap();
        let mut image = KvBytes::default();
        image.prepare(16, hwspec::Vendor::Nvidia);
        let mut held = KvBytes::default();
        held.prepare(16, hwspec::Vendor::Amd);
        let mut fallback = KvBytes::default();
        fallback.prepare(16, hwspec::Vendor::Nvidia);
        assert!(image.is_staged() && held.is_staged());
        assert!(!fallback.is_staged());
        let source: Vec<u8> = (0..16).collect();
        let mut dest = vec![0u8; 16];
        let spans: Vec<_> = [0, 8]
            .into_iter()
            .map(|off| ImageSpan {
                handle: 0,
                native_off: off,
                image_off: off,
                bytes: 8,
            })
            .collect();
        let src = [crate::device::DeviceMem::view(source.as_ptr() as u64, 16)];
        let dst = [crate::device::DeviceMem::view(dest.as_mut_ptr() as u64, 16)];
        assert!(image.transfer(&*nv, &spans, &src, true).unwrap());
        assert_eq!(&*image, source.as_slice());
        assert!(image.transfer(&*amd, &spans, &dst, false).unwrap());
        assert_eq!(dest, source);
        assert_eq!(nv.waits.load(Ordering::Relaxed), 1);
        assert_eq!(amd.waits.load(Ordering::Relaxed), 1);
        drop(image);
        fallback.prepare(16, hwspec::Vendor::Nvidia);
        assert!(fallback.is_staged());
        assert_eq!(nv.registrations.load(Ordering::Relaxed), 2);
        assert_eq!(amd.registrations.load(Ordering::Relaxed), 2);
        drop((fallback, held));
        nv.fail_enqueue.store(true, Ordering::Relaxed);
        image = KvBytes::default();
        image.prepare(16, hwspec::Vendor::Nvidia);
        assert!(image.transfer(&*nv, &spans, &src, true).is_err());
        assert!(!image.has_pending_dma());
        assert!(nv.copies.lock().is_empty());
        drop(image);
        assert_eq!(IMAGE_REGIONS.lock().len(), 2);
        nv.fail_enqueue.store(false, Ordering::Relaxed);
        nv.fail_wait.store(true, Ordering::Relaxed);
        image = KvBytes::default();
        image.prepare(16, hwspec::Vendor::Nvidia);
        assert!(image.transfer(&*nv, &spans, &src, true).is_err());
        assert!(image.has_pending_dma());
        struct Owner(Arc<AtomicUsize>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        impl crate::device::DeviceFree for Owner {
            fn free(&self, _: u64, _: u64) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let free = Arc::new(Owner(drops.clone()));
        let mut tensors = vec![crate::device::DeviceMem::owned(0x1000, 16, free.clone())];
        let mut slab = crate::memory::vmm::WeightSlab::Flat(crate::device::DeviceMem::owned(
            0x2000,
            16,
            free.clone(),
        ));
        let mut vmm = Some(Owner(drops.clone()));
        let backend = driver(hwspec::Vendor::Nvidia, 0);
        let backend_weak = Arc::downgrade(&backend);
        image.quarantine(&mut tensors, &mut slab, &mut vmm, backend.clone());
        assert_eq!(tensors[0].base, 0x1000);
        assert!(matches!(slab, crate::memory::vmm::WeightSlab::PerTensor));
        assert!(vmm.is_none());
        drop((tensors, slab, vmm, free, backend));
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        assert!(backend_weak.upgrade().is_some());
        assert!(image.has_pending_dma());
        assert!(unretired_copy(&RuntimeError::Device("wait failed".into())).fatal);
        drop(image);
        assert_eq!(IMAGE_REGIONS.lock().len(), 1);
        IMAGE_REGIONS.lock().clear();
    }

    fn image_spans() -> [ImageSpan; 4] {
        std::array::from_fn(|i| ImageSpan {
            handle: 0,
            native_off: i as u64 * 4,
            image_off: i as u64 * 4,
            bytes: 4,
        })
    }

    fn await_export_waiter(lease: &RegionLease) {
        let started = std::time::Instant::now();
        while lease.waiting.load(Ordering::Acquire) == 0 {
            assert!(started.elapsed() < std::time::Duration::from_secs(5));
            std::thread::yield_now();
        }
    }

    #[test]
    fn streaming_halves_overlap_but_partial_images_have_no_cpu_view() {
        let _serial = IMAGE_TEST_LOCK.lock();
        let nv = driver(Vendor::Nvidia, 0);
        let amd = driver(Vendor::Amd, 4096);
        prepare_image_pool(16, &[nv.clone(), amd.clone()]).unwrap();
        let mut image = KvBytes::default();
        image.prepare(16, Vendor::Nvidia);
        let source: Vec<u8> = (0..16).collect();
        let mut dest = vec![0u8; 16];
        let src = [crate::device::DeviceMem::view(source.as_ptr() as u64, 16)];
        let dst = [crate::device::DeviceMem::view(dest.as_mut_ptr() as u64, 16)];
        let (second_tx, second_rx) = std::sync::mpsc::channel();
        let second_rx = parking_lot::Mutex::new(second_rx);
        let (first_tx, first_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = parking_lot::Mutex::new(release_rx);
        *nv.on_wait.lock() = Some(Arc::new(move |n| {
            if n == 2 {
                second_tx.send(()).unwrap();
                release_rx
                    .lock()
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
        }));
        *amd.on_wait.lock() = Some(Arc::new(move |n| {
            if n == 1 {
                second_rx
                    .lock()
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                first_tx.send(()).unwrap();
            }
        }));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let producer = scope.spawn(|| {
                let result = image.export_with(Some(&mut |ready| tx.send(ready).unwrap()), |out| {
                    out.transfer(&*nv, &image_spans(), &src, true)
                        .map(|_| ())
                        .map_err(|e| e.error)
                });
                assert!(result.is_ok());
            });
            let ready = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
            let KvBytes::Pooled(lease) = &ready else {
                panic!("pooled image")
            };
            let lease = lease.clone();
            let consumer = scope.spawn(move || ready.transfer(&*amd, &image_spans(), &dst, false));
            first_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            assert!(lease.has_pending_dma());
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lease.bytes())).is_err()
            );
            release_tx.send(()).unwrap();
            producer.join().unwrap();
            assert!(consumer.join().unwrap().unwrap());
            assert_eq!(lease.bytes(), source);
        });
        assert_eq!(dest, source);
        drop(image);
        assert_eq!(IMAGE_REGIONS.lock().len(), 2);
        IMAGE_REGIONS.lock().clear();
    }

    #[test]
    fn a_retired_export_error_does_not_claim_the_consumers_pending_dma() {
        let _serial = IMAGE_TEST_LOCK.lock();
        let nv = driver(Vendor::Nvidia, 0);
        let amd = driver(Vendor::Amd, 4096);
        prepare_image_pool(16, &[nv.clone(), amd.clone()]).unwrap();
        let mut image = KvBytes::default();
        image.prepare(16, Vendor::Nvidia);
        let source: Vec<u8> = (0..16).collect();
        let mut dest = vec![0u8; 16];
        let src = [crate::device::DeviceMem::view(source.as_ptr() as u64, 16)];
        let dst = [crate::device::DeviceMem::view(dest.as_mut_ptr() as u64, 16)];
        let (second_tx, second_rx) = std::sync::mpsc::channel();
        let second_rx = parking_lot::Mutex::new(second_rx);
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let started_rx = parking_lot::Mutex::new(started_rx);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = parking_lot::Mutex::new(release_rx);
        let weak = Arc::downgrade(&nv);
        *nv.on_wait.lock() = Some(Arc::new(move |n| {
            if n == 1 {
                weak.upgrade()
                    .unwrap()
                    .fail_enqueue
                    .store(true, Ordering::Relaxed);
            } else if n == 2 {
                second_tx.send(()).unwrap();
                started_rx
                    .lock()
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
        }));
        *amd.on_wait.lock() = Some(Arc::new(move |n| {
            if n == 1 {
                second_rx
                    .lock()
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                started_tx.send(()).unwrap();
                entered_tx.send(()).unwrap();
                release_rx
                    .lock()
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
        }));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let producer = scope.spawn(|| {
                image.export_with(Some(&mut |ready| tx.send(ready).unwrap()), |out| {
                    let error = out.transfer(&*nv, &image_spans(), &src, true).unwrap_err();
                    assert!(!error.unretired && out.has_pending_dma());
                    Err(error.error)
                })
            });
            let ready = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
            let consumer = scope.spawn(move || ready.transfer(&*amd, &image_spans(), &dst, false));
            entered_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            assert!(producer.join().unwrap().is_err());
            release_tx.send(()).unwrap();
            let error = consumer.join().unwrap().unwrap_err();
            assert!(!error.unretired && matches!(error.error, RuntimeError::Unavailable(_)));
        });
        assert!(!image.has_pending_dma());
        drop(image);
        assert_eq!(IMAGE_REGIONS.lock().len(), 2);
        IMAGE_REGIONS.lock().clear();
    }

    #[test]
    fn source_final_success_gates_views_and_unretired_consumer_fences_retain_slabs() {
        let _serial = IMAGE_TEST_LOCK.lock();
        let nv = driver(Vendor::Nvidia, 0);
        let amd = driver(Vendor::Amd, 4096);
        prepare_image_pool(16, &[nv.clone(), amd.clone()]).unwrap();
        let source: Vec<u8> = (0..16).collect();
        let src = [crate::device::DeviceMem::view(source.as_ptr() as u64, 16)];
        let mut image = KvBytes::default();
        image.prepare(16, Vendor::Nvidia);
        let mut ready = None;
        image
            .export_with(Some(&mut |bytes| ready = Some(bytes)), |out| {
                out.transfer(&*nv, &image_spans(), &src, true)
                    .map_err(|e| e.error)?;
                assert!(!out.has_pending_dma());
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| &**out)).is_err());
                Ok(())
            })
            .unwrap();
        let ready = ready.unwrap();
        assert_eq!(&*ready, source);
        let mut dest = vec![0u8; 16];
        let dst = [crate::device::DeviceMem::view(dest.as_mut_ptr() as u64, 16)];
        amd.fail_wait.store(true, Ordering::Relaxed);
        let error = ready
            .transfer(&*amd, &image_spans(), &dst, false)
            .unwrap_err();
        assert!(error.unretired && image.has_pending_dma());
        drop((ready, image));
        assert_eq!(IMAGE_REGIONS.lock().len(), 1);
        IMAGE_REGIONS.lock().clear();
    }

    #[test]
    fn abandoned_publication_wakes_peers_without_dma_or_pool_loss() {
        let _serial = IMAGE_TEST_LOCK.lock();
        let nv = driver(Vendor::Nvidia, 0);
        let amd = driver(Vendor::Amd, 4096);
        prepare_image_pool(16, &[nv, amd.clone()]).unwrap();
        let mut image = KvBytes::default();
        image.prepare(16, Vendor::Nvidia);
        let (tx, rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let producer = scope.spawn(|| {
                assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    image.export_with(
                        Some(&mut |bytes| {
                            let KvBytes::Pooled(lease) = &bytes else {
                                panic!("pooled image")
                            };
                            let lease = lease.clone();
                            tx.send(bytes).unwrap();
                            await_export_waiter(&lease);
                            panic!("publisher stopped");
                        }),
                        |_| unreachable!(),
                    )
                }))
                .is_err());
            });
            let ready = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
            let KvBytes::Pooled(lease) = &ready else {
                panic!("pooled image")
            };
            let lease = lease.clone();
            let consumer = scope.spawn(move || {
                let error = ready
                    .transfer(&*amd, &image_spans(), &[], false)
                    .unwrap_err();
                assert!(!error.unretired && matches!(error.error, RuntimeError::Unavailable(_)));
                done_tx.send(()).unwrap();
            });
            let notified = done_rx.recv_timeout(std::time::Duration::from_secs(5));
            if notified.is_err() {
                lease.changed.notify_all();
            }
            producer.join().unwrap();
            consumer.join().unwrap();
            assert!(
                notified.is_ok(),
                "abandoned export did not wake an already-blocked consumer"
            );
        });
        assert!(!image.has_pending_dma());
        drop(image);
        assert_eq!(IMAGE_REGIONS.lock().len(), 2);
        IMAGE_REGIONS.lock().clear();
    }

    #[test]
    fn imported_halves_wait_for_final_source_success_or_error() {
        let _serial = IMAGE_TEST_LOCK.lock();
        for fail in [false, true] {
            let nv = driver(Vendor::Nvidia, 0);
            let amd = driver(Vendor::Amd, 4096);
            prepare_image_pool(16, &[nv.clone(), amd.clone()]).unwrap();
            let source: Vec<u8> = (0..16).collect();
            let mut dest = vec![0u8; 16];
            let src = [crate::device::DeviceMem::view(source.as_ptr() as u64, 16)];
            let dst = [crate::device::DeviceMem::view(dest.as_mut_ptr() as u64, 16)];
            let mut image = KvBytes::default();
            image.prepare(16, Vendor::Nvidia);
            let (copied_tx, copied_rx) = std::sync::mpsc::channel();
            *amd.on_wait.lock() = Some(Arc::new(move |n| {
                if n == 2 {
                    copied_tx.send(()).unwrap();
                }
            }));
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let release_rx = parking_lot::Mutex::new(release_rx);
            let (tx, rx) = std::sync::mpsc::channel();
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            std::thread::scope(|scope| {
                let producer = scope.spawn(|| {
                    image.export_with(Some(&mut |ready| tx.send(ready).unwrap()), |out| {
                        out.transfer(&*nv, &image_spans(), &src, true)
                            .map_err(|e| e.error)?;
                        release_rx
                            .lock()
                            .recv_timeout(std::time::Duration::from_secs(5))
                            .unwrap();
                        if fail {
                            Err(RuntimeError::Unavailable("final source failure".into()))
                        } else {
                            Ok(())
                        }
                    })
                });
                let ready = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
                let KvBytes::Pooled(lease) = &ready else {
                    panic!("pooled image")
                };
                let lease = lease.clone();
                let consumer = scope.spawn(move || {
                    let result = ready.transfer(&*amd, &image_spans(), &dst, false);
                    done_tx.send(result.is_ok()).unwrap();
                    result
                });
                copied_rx
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                await_export_waiter(&lease);
                assert!(!lease.has_pending_dma());
                assert!(done_rx.try_recv().is_err());
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lease.bytes()))
                        .is_err()
                );
                release_tx.send(()).unwrap();
                assert_eq!(producer.join().unwrap().is_err(), fail);
                let notified = done_rx.recv_timeout(std::time::Duration::from_secs(5));
                if notified.is_err() {
                    lease.changed.notify_all();
                }
                let result = consumer.join().unwrap();
                assert!(
                    notified.is_ok(),
                    "final export result did not wake the consumer"
                );
                assert_eq!(result.is_err(), fail);
                if let Err(error) = result {
                    assert!(
                        !error.unretired && matches!(error.error, RuntimeError::Unavailable(_))
                    );
                }
            });
            assert_eq!(dest, source);
            drop(image);
            assert_eq!(IMAGE_REGIONS.lock().len(), 2);
            IMAGE_REGIONS.lock().clear();
        }
    }

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
