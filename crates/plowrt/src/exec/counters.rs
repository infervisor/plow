//! §D Counter pool — the dataflow-dependency substrate.
//!
//! A counter is an atomic integer a producer bumps and a consumer gates on. The
//! compiler assigns every counter an id, threshold, and scope
//! (`packet::Counter`); the runtime materializes storage and does atomic
//! loads/adds.
//!
//! ## Where the counter cells physically live (host ↔ SM/CU)
//!
//! The cells are **not** an ordinary Rust heap `Vec` on a real device — the
//! GPU SM/CU must perform atomics on the very same cells the host coordinator
//! polls. Placement is **scope-driven**, matching the design doc (§4.4/§9.1):
//!
//! * **IntraSm** — SM shared memory (device-only; never touched by the host).
//! * **IntraGpu** — **device-global HBM** (`cuMemAlloc`), device-scope atomics
//!   between SMs; the host reads milestones via the mapping/DtoH.
//! * **CrossUnit / cross-device** — **host-pinned, device-mapped** memory
//!   (`cuMemHostAlloc` with `DEVICEMAP` → `cuMemHostGetDevicePointer`): one
//!   region, a host pointer *and* a GPU-visible device pointer, accessed with
//!   **system-scope** atomics over PCIe. This is the memory-mapped, coherent
//!   region both sides share.
//!
//! We deliberately do **not** use CUDA managed/unified memory
//! (`cuMemAllocManaged`) for hot counters: page migration would thrash the
//! atomic traffic. "Unified" here means *device-mapped pinned* memory, not UM.
//!
//! So [`CounterPool`] is built over a raw base pointer into a backend-allocated
//! mapped region (or a host box for the CPU backend). Cells are **64-byte
//! strided** (one cache line each) for isolation, matching the device
//! `struct counter { u64 value; u8 pad[56]; }` ABI. Load/add are `#[inline]`,
//! allocation-free, with `Acquire`/`Release` ordering.

use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_utils::CachePadded;

use crate::device::DeviceMem;

pub use packet::Scope;

/// One counter cell occupies a full cache line (matches the 64-byte device ABI).
pub const CELL_STRIDE: usize = 64;

/// Backing store keeping the counter region alive.
enum Backing {
    /// Host-owned, cache-line-isolated atomics (CPU backend / pinned model).
    /// Held to keep the memory `base` points into alive; accessed via `base`.
    Host(#[allow(dead_code)] Box<[CachePadded<AtomicU64>]>),
    /// A backend-allocated device-mapped region (pinned + device-mapped). The
    /// GPU reaches it via its device pointer; the host via `base` below.
    #[allow(dead_code)]
    Mapped(DeviceMem),
}

/// A pool of atomic counters over a mapped region. `base` points at cell 0's
/// `AtomicU64`; cell `id` is at `base + id * CELL_STRIDE`.
pub struct CounterPool {
    base: *const AtomicU64,
    stride: usize,
    len: usize,
    thresholds: Box<[u64]>,
    scopes: Box<[Scope]>,
    _backing: Backing,
}

// SAFETY: the cells are atomics; concurrent access from many executors + the
// host is exactly their contract. `base` stays valid for the pool's lifetime
// (kept alive by `_backing`).
unsafe impl Send for CounterPool {}
unsafe impl Sync for CounterPool {}

impl CounterPool {
    /// Build a host-backed pool for a program's counter table (CPU backend, and
    /// the model for host-pinned cross-device counters). `id` namespace is the
    /// dense `0..n` the compiler assigns.
    pub fn from_counters(counters: &[packet::Counter]) -> Self {
        let n = counters
            .iter()
            .map(|c| c.id as usize + 1)
            .max()
            .unwrap_or(0);
        let mut cells: Vec<CachePadded<AtomicU64>> = Vec::with_capacity(n);
        cells.resize_with(n, || CachePadded::new(AtomicU64::new(0)));
        let boxed = cells.into_boxed_slice();
        let base = boxed.as_ptr() as *const AtomicU64;
        let (thresholds, scopes) = meta(counters, n);
        CounterPool {
            base,
            stride: std::mem::size_of::<CachePadded<AtomicU64>>(),
            len: n,
            thresholds,
            scopes,
            _backing: Backing::Host(boxed),
        }
    }

    /// Build a pool over a backend-allocated device-mapped region. `dev.base`
    /// must be a host-usable pointer into pinned+device-mapped memory of at
    /// least `n * CELL_STRIDE` bytes; the same region's device pointer is handed
    /// to the kernel so SM/CU atomics and host polls hit the same cells.
    ///
    /// # Safety
    /// `dev` must be host-readable (pinned/mapped) and correctly aligned for
    /// `AtomicU64`, and outlive the pool.
    pub unsafe fn over_mapped(dev: DeviceMem, counters: &[packet::Counter]) -> Self {
        let n = counters
            .iter()
            .map(|c| c.id as usize + 1)
            .max()
            .unwrap_or(0);
        debug_assert!(dev.len as usize >= n * CELL_STRIDE);
        let base = dev.base as *const AtomicU64;
        let (thresholds, scopes) = meta(counters, n);
        CounterPool {
            base,
            stride: CELL_STRIDE,
            len: n,
            thresholds,
            scopes,
            _backing: Backing::Mapped(dev),
        }
    }

    #[inline]
    fn cell(&self, id: u32) -> &AtomicU64 {
        debug_assert!((id as usize) < self.len);
        // SAFETY: id < len; base+id*stride is within the region and aligned.
        unsafe { &*((self.base as *const u8).add(id as usize * self.stride) as *const AtomicU64) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Current value. `Acquire` pairs with a producer's `Release` add so the
    /// consumer sees the producer's writes to the gated buffer.
    #[inline]
    pub fn load(&self, id: u32) -> u64 {
        self.cell(id).load(Ordering::Acquire)
    }

    /// Atomically add `v`, returning the new value.
    #[inline]
    pub fn add(&self, id: u32, v: u64) -> u64 {
        self.cell(id).fetch_add(v, Ordering::AcqRel) + v
    }

    #[inline]
    pub fn reset(&self, id: u32) {
        self.cell(id).store(0, Ordering::Release);
    }

    pub fn reset_all(&self) {
        for id in 0..self.len as u32 {
            self.cell(id).store(0, Ordering::Release);
        }
    }

    #[inline]
    pub fn threshold(&self, id: u32) -> u64 {
        self.thresholds[id as usize]
    }

    #[inline]
    pub fn scope(&self, id: u32) -> Scope {
        self.scopes[id as usize]
    }

    #[inline]
    pub fn satisfied(&self, id: u32) -> bool {
        self.load(id) >= self.threshold(id)
    }
}

fn meta(counters: &[packet::Counter], n: usize) -> (Box<[u64]>, Box<[Scope]>) {
    let mut thresholds = vec![0u64; n];
    let mut scopes = vec![Scope::CrossUnit; n];
    for c in counters {
        thresholds[c.id as usize] = c.threshold as u64;
        scopes[c.id as usize] = Scope::from_u8(c.scope);
    }
    (thresholds.into_boxed_slice(), scopes.into_boxed_slice())
}
