//! §B Device backend — the cross-vendor portability boundary.
//!
//! Everything above this trait is fully portable; everything below is
//! vendor-specific. A real CPU reference backend ships in [`cpu`]; CUDA and HSA
//! are FFI backends ([`cuda`], [`hsa`]) behind the `cuda` / `hsa` features — the
//! runtime owns loading/launch/coordination, the compute kernels are external
//! prebuilt `cubin`/`hsaco` modules (design decision 3).

use std::sync::Arc;

use hwspec::Vendor;

use crate::Result;

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "hsa")]
pub mod hsa;

/// Bring up the best backend this host can actually offer.
///
/// One binary serves both CPU and GPU: the vendor drivers are `dlopen`ed, never
/// linked, so a GPU-capable build starts fine on a machine with no driver at
/// all. We try CUDA, then HSA (AMD), and fall back to the CPU reference backend —
/// which interprets the very same compiled programs, since every asset is
/// compiled for some GPU spec (there is no CPU target in [`hwspec`]). A missing
/// driver therefore costs acceleration, not availability.
/// Select a single backend (device 0 preferred). Legacy single-GPU path.
pub fn select(executors: u32) -> Arc<dyn Backend> {
    #[cfg(feature = "cuda")]
    {
        tracing::debug!("probing CUDA backend (device 0)...");
        match cuda::CudaBackend::new(0) {
            Ok(b) => {
                tracing::info!(device = %b.device_name(), "CUDA backend selected");
                return Arc::new(b);
            }
            Err(e) => tracing::warn!(%e, "CUDA probe failed"),
        }
    }

    #[cfg(feature = "hsa")]
    {
        tracing::debug!("probing HSA backend (device 0)...");
        match hsa::HsaBackend::new(0) {
            Ok(b) => {
                tracing::info!(device = %b.device_name(), "HSA backend selected");
                return Arc::new(b);
            }
            Err(e) => tracing::warn!(%e, "HSA probe failed"),
        }
    }

    tracing::warn!("all GPU probes failed — selecting CPU reference backend");
    Arc::new(cpu::CpuBackend::new(executors))
}

/// Enumerate ALL visible devices and return a backend per device.
/// Multi-GPU: returns one entry per GPU (TP/PP placement is per-device).
/// Falls back to a single CPU backend if no accelerators are found.
pub fn select_all(executors_per_device: u32) -> Vec<Arc<dyn Backend>> {
    let mut backends: Vec<Arc<dyn Backend>> = Vec::new();

    #[cfg(feature = "cuda")]
    {
        // Probe device count via the backend's enumerate path.
        // In production: cuDeviceGetCount → iterate [0..count).
        if let Ok(b) = cuda::CudaBackend::new(0) {
            backends.push(Arc::new(b));
            // Additional devices: try indices 1..8 until failure.
            for dev in 1..8u8 {
                match cuda::CudaBackend::new(dev) {
                    Ok(b) => backends.push(Arc::new(b)),
                    Err(_) => break,
                }
            }
        }
    }

    #[cfg(feature = "hsa")]
    if backends.is_empty() {
        if let Ok(b) = hsa::HsaBackend::new(0) {
            backends.push(Arc::new(b));
            for dev in 1..8u8 {
                match hsa::HsaBackend::new(dev) {
                    Ok(b) => backends.push(Arc::new(b)),
                    Err(_) => break,
                }
            }
        }
    }

    if backends.is_empty() {
        backends.push(Arc::new(cpu::CpuBackend::new(executors_per_device)));
    }

    backends
}

/// Executor substrate class (the doc's `executor_class_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutorClass {
    /// NVIDIA SM (persistent CUDA kernel).
    SmNv,
    /// AMD CU (persistent HSA kernel).
    CuAmd,
    /// CPU thread executor.
    Cpu,
    /// DPU ARM core.
    DpuArm,
}

/// An executor's published capabilities — the scheduling inputs. Mirrors the
/// design doc's `executor_target`; scheduling decisions consume this, never the
/// class alone.
#[derive(Clone, Debug)]
pub struct ExecutorTarget {
    pub class: ExecutorClass,
    pub instance_id: u32,
    /// 32 (NV / RDNA / CPU) or 64 (CDNA).
    pub wave_width: u32,
    /// Worker warps/waves the interpreter can dispatch to per executor.
    pub worker_count: u32,
    /// Shared-memory / LDS budget in bytes.
    pub shmem_bytes: u32,
    /// Bitmask of supported opcode families (bit = `Opcode::family`).
    pub opcode_mask: u32,
}

/// Frees one device allocation. Implemented by GPU backends; every **owned**
/// [`DeviceMem`] carries an `Arc` of it so Drop reaches the driver without a
/// backend reference (and the driver context provably outlives the handle —
/// the CUDA impl holds the primary-context retain alive through this `Arc`).
/// Infallible by design: Drop has no error channel, implementations log.
pub(crate) trait DeviceFree: Send + Sync {
    fn free(&self, base: u64, len: u64);
}

/// Opaque handle to a physical allocation on a device (one per address-map
/// segment; see [`crate::memory::AddressSpace`]). `base` is the device virtual
/// address slots rebase against — for the CPU backend, a real host pointer.
///
/// Ownership is fixed at construction, never inferred: handles from
/// [`Backend::alloc`] **own** their storage and free it on Drop; handles from
/// [`DeviceMem::view`] are borrowed sub-ranges of some owner's allocation and
/// never free. The owner must be kept alive for as long as any view's address
/// is used (views carry no lifetime — the holder pins the owner, e.g.
/// `GpuEngine` stores the counter block next to its two views).
pub struct DeviceMem {
    pub base: u64,
    pub len: u64,
    backing: Backing,
}

enum Backing {
    /// CPU arena — host memory that *is* the "device" memory (freed when the
    /// last `Arc` drops).
    Cpu(Arc<cpu::CpuArena>),
    /// GPU allocation owned by this handle: freed exactly once on Drop.
    Owned { free: Arc<dyn DeviceFree> },
    /// Borrowed view into another allocation (e.g. an aliased sub-range):
    /// never freed here — the owner's Drop reclaims the storage.
    View,
}

impl DeviceMem {
    /// An owning device-allocation handle (GPU backends): `base` is the device
    /// virtual address returned by the driver allocator; `free` reclaims it on
    /// Drop.
    #[cfg_attr(not(any(feature = "cuda", feature = "hsa")), allow(dead_code))]
    pub(crate) fn owned(base: u64, len: u64, free: Arc<dyn DeviceFree>) -> DeviceMem {
        DeviceMem {
            base,
            len,
            backing: Backing::Owned { free },
        }
    }

    /// A non-owning view of `[base, base+len)` inside some owner's allocation.
    /// Never freed; the caller keeps the owner alive for the view's lifetime.
    #[cfg_attr(not(any(feature = "cuda", feature = "hsa")), allow(dead_code))]
    pub(crate) fn view(base: u64, len: u64) -> DeviceMem {
        DeviceMem {
            base,
            len,
            backing: Backing::View,
        }
    }

    /// Host-visible byte slice for CPU-backed memory; `None` for real-device
    /// memory (which must be reached via [`Backend::download`]).
    ///
    /// # Safety
    /// The returned slice aliases the entire arena; the caller must guarantee no
    /// concurrent writer (e.g. `upload` or an executing schedule) for the
    /// slice's lifetime — quiescent points only. For concurrent-safe access to a
    /// subrange, use [`Backend::download`] instead.
    pub unsafe fn as_host_slice(&self) -> Option<&[u8]> {
        match &self.backing {
            Backing::Cpu(a) => Some(a.as_slice()),
            Backing::Owned { .. } | Backing::View => None,
        }
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        // Only owners free; views alias an owner's storage (a naive
        // unconditional free here is the double-free the leak audit flagged),
        // and CPU arenas free through their own `Arc`.
        if let Backing::Owned { free } = &self.backing {
            free.free(self.base, self.len);
        }
    }
}

/// A loaded compute module (prebuilt `cubin`/`hsaco`, or the CPU interpreter's
/// op table). Opaque above the backend.
pub struct Module {
    #[allow(dead_code)]
    pub(crate) id: u64,
}

/// Persistent-kernel launch config: one block per executor, `workers` warps each.
#[derive(Clone, Copy, Debug)]
pub struct LaunchCfg {
    pub executors: u32,
    pub workers: u32,
}

/// The device abstraction. Object-safe (`Arc<dyn Backend>`) so the runtime holds
/// a heterogeneous set. Everything here is off the per-token hot path (alloc,
/// upload, launch happen at bringup / on pressure); the hot path talks to the
/// counter pool and queues directly.
pub trait Backend: Send + Sync {
    fn class(&self) -> ExecutorClass;

    /// Silicon vendor this backend drives, or `None` for the CPU reference
    /// backend. Used to report whether an asset — always compiled for some
    /// [`hwspec`] GPU — is running accelerated or interpreted.
    fn vendor(&self) -> Option<Vendor> {
        None
    }

    /// Publish each executor's capability descriptor.
    fn enumerate(&self) -> Vec<ExecutorTarget>;

    /// Allocate one physical arena of `bytes` on `device` (one per segment).
    fn alloc(&self, device: u8, bytes: u64) -> Result<DeviceMem>;

    /// Copy host bytes into device memory at `off` (H2D; startup/pressure).
    fn upload(&self, dst: &DeviceMem, off: u64, src: &[u8]) -> Result<()>;

    /// Copy device bytes out to host at `off` (D2H; e.g. logits fallback).
    fn download(&self, src: &DeviceMem, off: u64, dst: &mut [u8]) -> Result<()>;

    /// Load a prebuilt kernel module image.
    fn module_load(&self, image: &[u8]) -> Result<Module>;

    /// Launch the persistent kernel once (no-op / thread-spawn on CPU).
    fn launch_persistent(&self, module: &Module, cfg: LaunchCfg) -> Result<()>;

    /// Allocate the counter region: host-pinned **device-mapped** memory whose
    /// [`DeviceMem::base`] is host-usable and whose device pointer is handed to
    /// the kernel, so SM/CU atomics and host polls hit the same cells (see
    /// [`crate::exec::counters`]). GPU backends override this with
    /// `cuMemHostAlloc(DEVICEMAP)` / HSA fine-grained system memory; the default
    /// is plain host memory (correct for the CPU backend).
    fn alloc_counter_region(&self, count: usize) -> Result<DeviceMem> {
        let bytes = (count * crate::exec::counters::CELL_STRIDE) as u64;
        self.alloc(0, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Records every free: `(count, last base)` — double-free shows as count 2.
    struct CountingFree {
        count: AtomicUsize,
        last_base: AtomicUsize,
    }

    impl DeviceFree for CountingFree {
        fn free(&self, base: u64, _len: u64) {
            self.count.fetch_add(1, Ordering::SeqCst);
            self.last_base.store(base as usize, Ordering::SeqCst);
        }
    }

    /// The audit's d_ctr/d_gq_cursor shape: one owned block, two views carved
    /// from it. Dropping the views must free nothing; dropping the owner must
    /// free exactly once, at the block base.
    #[test]
    fn owner_frees_once_views_never() {
        let f = Arc::new(CountingFree {
            count: AtomicUsize::new(0),
            last_base: AtomicUsize::new(0),
        });
        let owner = DeviceMem::owned(0x1000, 256, f.clone());
        let view_a = DeviceMem::view(0x1000, 64);
        let view_b = DeviceMem::view(0x1040, 192);
        assert!(unsafe { view_a.as_host_slice() }.is_none());

        drop(view_a);
        drop(view_b);
        assert_eq!(f.count.load(Ordering::SeqCst), 0, "views must not free");

        drop(owner);
        assert_eq!(f.count.load(Ordering::SeqCst), 1, "owner frees exactly once");
        assert_eq!(f.last_base.load(Ordering::SeqCst), 0x1000);
    }
}
