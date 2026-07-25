//! §M Host↔Device DMA plane — pinned staging + role-dedicated copy streams.
//!
//! Moves whole buffers across PCIe/NVLink between host RAM and the arena (and
//! device↔device). Intra-device HBM↔SMEM staging is the kernel's job (TMA), not
//! here. The transfer *policy* per use case (weight load, per-request marshal,
//! per-token output, KV swap) lives in the plan §M; this is the mechanism.

use crate::device::Backend;
use crate::Result;

/// Which copy stream a transfer runs on. Distinct streams (and, on real HW,
/// distinct copy engines) let H2D/D2H overlap compute and each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    /// Bulk weight load at startup — off the critical path.
    Load,
    /// Per-request inputs, host → device.
    Ingest,
    /// Per-token / logits outputs, device → host.
    Egress,
    /// Background KV swap / offload.
    Background,
}

/// A recycled page-locked staging buffer. Pinning is expensive and must not
/// happen per request; buffers are pinned once and reused. (Skeleton: a plain
/// `Vec`; the GPU backend allocates `cuMemHostAlloc` pinned pages.)
pub struct PinnedBuf {
    buf: Vec<u8>,
}

impl PinnedBuf {
    pub fn with_capacity(bytes: usize) -> Self {
        PinnedBuf {
            buf: vec![0u8; bytes],
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }
}

/// An opaque event handle representing a pending async DMA transfer.
/// On real hardware this wraps a CUDA event / HSA signal. On the CPU
/// backend the transfer is already complete at creation time.
pub struct DmaEvent {
    completed: bool,
}

impl DmaEvent {
    /// Create a pre-completed event (CPU backend / synchronous fallback).
    fn completed() -> Self {
        DmaEvent { completed: true }
    }

    /// Poll whether the transfer has landed. Non-blocking.
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    /// Block until the transfer finishes. No-op if already complete.
    pub fn wait(&mut self) {
        // On real HW: cuEventSynchronize / hsa_signal_wait_scacquire.
        self.completed = true;
    }
}

/// The DMA plane over a backend. Async on real HW (`cuMemcpyAsync` + events);
/// synchronous `memcpy` on the CPU backend.
pub struct DmaPlane<'b> {
    backend: &'b dyn Backend,
}

impl<'b> DmaPlane<'b> {
    pub fn new(backend: &'b dyn Backend) -> Self {
        DmaPlane { backend }
    }

    /// Host → device on the given stream. (Skeleton is synchronous; production
    /// enqueues on the stream and signals a completion counter/event.)
    pub fn h2d(
        &self,
        dst: &crate::device::DeviceMem,
        off: u64,
        src: &[u8],
        _stream: Stream,
    ) -> Result<()> {
        self.backend.upload(dst, off, src)
    }

    /// Device → host on the given stream.
    pub fn d2h(
        &self,
        src: &crate::device::DeviceMem,
        off: u64,
        dst: &mut [u8],
        _stream: Stream,
    ) -> Result<()> {
        self.backend.download(src, off, dst)
    }

    /// Async host → device. Returns a [`DmaEvent`] the caller polls or awaits.
    /// On the CPU backend this completes synchronously; on GPU the copy is
    /// enqueued on the stream's copy engine and the event fires on completion.
    ///
    /// Use for weight prefetch and KV offload where overlap with compute is key.
    pub fn h2d_async(
        &self,
        dst: &crate::device::DeviceMem,
        off: u64,
        src: &[u8],
        _stream: Stream,
    ) -> Result<DmaEvent> {
        // CPU backend: synchronous, return pre-completed event.
        self.backend.upload(dst, off, src)?;
        Ok(DmaEvent::completed())
    }

    /// Async device → host. Returns a [`DmaEvent`] the caller polls or awaits.
    pub fn d2h_async(
        &self,
        src: &crate::device::DeviceMem,
        off: u64,
        dst: &mut [u8],
        _stream: Stream,
    ) -> Result<DmaEvent> {
        self.backend.download(src, off, dst)?;
        Ok(DmaEvent::completed())
    }
}
