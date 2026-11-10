//! A request's KV cache as an image another engine can import (docs/arch/30 §5): the byte copies
//! between one sequence slot of a packet's declared caches and a flat host image.
//!
//! Pure functions over the packet's `live_kv` manifest, so the addressing is tested without a
//! device. The image holds, per declared cache (KV-shared readers are declared once), per tensor
//! (K, V, then the FP8 row scales), per head, the live rows `lo..n` in position order:
//! `lo = 0` for a full cache, `n - window` for a sliding one. A ring is unwrapped and every native
//! layout packs to this one order, so the image depends on neither. Rows at or past `n` are never
//! addressed: under live KV they may be unmapped.
//!
//! Two engines, of either vendor, exchange images only when their `plow_asset::kv_image` headers
//! are equal.

use plow_asset::live_kv::{Manifest, LAYOUT_HEAD_MAJOR, LAYOUT_ROW_INTERLEAVED};

use crate::{Result, RuntimeError};

pub enum KvBytes {
    #[cfg(target_os = "linux")]
    Pooled(std::sync::Arc<crate::device::host_region::RegionLease>),
    Pageable(Vec<u8>),
}

impl Default for KvBytes {
    fn default() -> Self {
        Self::Pageable(Vec::new())
    }
}

impl From<Vec<u8>> for KvBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Pageable(bytes)
    }
}

impl std::ops::Deref for KvBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            #[cfg(target_os = "linux")]
            Self::Pooled(lease) => lease.bytes(),
            Self::Pageable(bytes) => bytes,
        }
    }
}

impl KvBytes {
    pub fn len(&self) -> usize {
        match self {
            #[cfg(target_os = "linux")]
            Self::Pooled(lease) => lease.len(),
            Self::Pageable(bytes) => bytes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stream_started(&self) -> Option<std::time::Instant> {
        #[cfg(target_os = "linux")]
        if let Self::Pooled(lease) = self {
            return lease.stream_started();
        }
        None
    }

    #[cfg(all(target_os = "linux", test))]
    pub(crate) fn has_pending_dma(&self) -> bool {
        matches!(self, Self::Pooled(lease) if lease.has_pending_dma())
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn quarantine<T>(
        &self,
        devp: &mut [crate::device::DeviceMem],
        slab: &mut crate::memory::vmm::WeightSlab,
        vmm: &mut Option<T>,
        backend: std::sync::Arc<dyn crate::device::host_region::HostStaging>,
    ) {
        for mem in devp {
            mem.quarantine();
        }
        std::mem::forget(std::mem::replace(
            slab,
            crate::memory::vmm::WeightSlab::PerTensor,
        ));
        std::mem::forget(vmm.take());
        std::mem::forget(backend);
    }

    pub fn is_staged(&self) -> bool {
        match self {
            #[cfg(target_os = "linux")]
            Self::Pooled(_) => true,
            Self::Pageable(_) => false,
        }
    }

    pub(crate) fn prepare(&mut self, bytes: usize, vendor: hwspec::Vendor) {
        #[cfg(target_os = "linux")]
        if self.is_staged() {
            *self = Self::default();
        }
        #[cfg(target_os = "linux")]
        if let Some(lease) = crate::device::host_region::RegionLease::take(bytes, vendor) {
            *self = Self::Pooled(std::sync::Arc::new(lease));
            return;
        }
        let _ = vendor;
        if let Self::Pageable(out) = self {
            out.clear();
            out.resize(bytes, 0);
        }
    }

    pub(crate) fn export_with(
        &mut self,
        publish: Option<&mut dyn FnMut(Self)>,
        copy: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        let guard = publish.and_then(|publish| {
            let Self::Pooled(lease) = self else {
                return None;
            };
            let guard = crate::device::host_region::RegionLease::start_export(lease)?;
            publish(Self::Pooled(std::sync::Arc::clone(lease)));
            Some(guard)
        });
        #[cfg(not(target_os = "linux"))]
        let _ = publish;
        let result = copy(self);
        #[cfg(target_os = "linux")]
        if let Some(guard) = guard {
            return guard.finish(result);
        }
        result
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn transfer(
        &self,
        driver: &dyn crate::device::host_region::HostStaging,
        spans: &[ImageSpan],
        devp: &[crate::device::DeviceMem],
        export: bool,
    ) -> std::result::Result<bool, ImageTransferError> {
        let Self::Pooled(lease) = self else {
            return Ok(false);
        };
        let Some(alias) = lease.alias(driver.staging_vendor()) else {
            if export {
                return Err(RuntimeError::Device(
                    "KV image export driver is not registered".into(),
                )
                .into());
            }
            lease.wait_export(lease.len(), true)?;
            return Ok(false);
        };
        let timing = tracing::enabled!(tracing::Level::DEBUG).then(std::time::Instant::now);
        let streaming = lease.streaming();
        let mid = if streaming {
            lease.len() / 2
        } else {
            lease.len()
        };
        let mut submit = std::time::Duration::ZERO;
        let mut retire = std::time::Duration::ZERO;
        let mut peer_wait = std::time::Duration::ZERO;
        let mut chunks = 0;
        for (lo, hi) in [(0, mid), (mid, lease.len())] {
            if lo == hi {
                continue;
            }
            if streaming && !export {
                let start = timing.map(|_| std::time::Instant::now());
                lease.wait_export(hi, false)?;
                if let Some(start) = start {
                    peer_wait += start.elapsed();
                }
            }
            let start = timing.map(|_| std::time::Instant::now());
            let mut copies = Vec::with_capacity(spans.len());
            let mut bytes = 0;
            for &span in spans {
                if let Some(s) = image_span_range(span, lo as u64, hi as u64)? {
                    copies.push((
                        alias + s.image_off,
                        devp[s.handle].base + s.native_off,
                        s.bytes,
                    ));
                    bytes += s.bytes;
                }
            }
            if bytes != (hi - lo) as u64 {
                return Err(
                    RuntimeError::Device("KV image range does not cover its bytes".into()).into(),
                );
            }
            let dma_start = timing.and_then(|_| lease.stream_ms());
            lease.begin_dma();
            let (last, queued) = driver.stage_batch(&copies, export);
            let submitted = start.map(|t| t.elapsed());
            if let Some(fence) = last {
                if let Err(error) = driver.stage_wait(fence) {
                    return Err(ImageTransferError {
                        error,
                        unretired: true,
                    });
                }
            } else if queued.is_ok() {
                return Err(ImageTransferError {
                    error: RuntimeError::Device("KV image batch has no completion fence".into()),
                    unretired: true,
                });
            }
            lease.retire_dma();
            queued?;
            if let (Some(start), Some(submitted)) = (start, submitted) {
                submit += submitted;
                retire += start.elapsed() - submitted;
            }
            if let Some(dma_start_ms) = dma_start {
                tracing::debug!(vendor = ?driver.staging_vendor(), export, lo, hi,
                    dma_start_ms, dma_end_ms = lease.stream_ms().expect("stream clock"),
                    "KV image chunk retired");
            }
            if streaming && export {
                lease.publish(hi)?;
            }
            chunks += 1;
        }
        if streaming && !export {
            let start = timing.map(|_| std::time::Instant::now());
            lease.wait_export(lease.len(), true)?;
            if let Some(start) = start {
                peer_wait += start.elapsed();
            }
        }
        if let Some(t) = timing {
            tracing::debug!(
                vendor = ?driver.staging_vendor(), export, spans = spans.len(), bytes = lease.len(), chunks,
                submit_ms = submit.as_secs_f64() * 1e3,
                wait_retire_ms = retire.as_secs_f64() * 1e3,
                peer_wait_ms = peer_wait.as_secs_f64() * 1e3,
                transfer_ms = t.elapsed().as_secs_f64() * 1e3,
                "KV image staged transfer"
            );
        }
        Ok(true)
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
pub(crate) struct ImageTransferError {
    pub error: RuntimeError,
    pub unretired: bool,
}

#[cfg(target_os = "linux")]
impl From<RuntimeError> for ImageTransferError {
    fn from(error: RuntimeError) -> Self {
        Self {
            error,
            unretired: false,
        }
    }
}

fn image_span_range(span: ImageSpan, lo: u64, hi: u64) -> Result<Option<ImageSpan>> {
    let end = span
        .image_off
        .checked_add(span.bytes)
        .ok_or_else(|| RuntimeError::Device("KV image span overflow".into()))?;
    if lo > hi {
        return Err(RuntimeError::Device("KV image range is reversed".into()));
    }
    let start = lo.max(span.image_off);
    let end = hi.min(end);
    if start >= end {
        return Ok(None);
    }
    let native_off = span
        .native_off
        .checked_add(start - span.image_off)
        .ok_or_else(|| RuntimeError::Device("KV image native span overflow".into()))?;
    native_off
        .checked_add(end - start)
        .ok_or_else(|| RuntimeError::Device("KV image native range overflow".into()))?;
    Ok(Some(ImageSpan {
        native_off,
        image_off: start,
        bytes: end - start,
        ..span
    }))
}

/// One contiguous copy between a cache tensor and the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageSpan {
    /// Packet tensor-table handle.
    pub handle: usize,
    /// Byte offset from the tensor's base (slot included).
    pub native_off: u64,
    /// Byte offset in the image.
    pub image_off: u64,
    pub bytes: u64,
}

/// The spans moving slot `slot`'s rows `[.., n)` and the image size.
pub(crate) fn plan(m: &Manifest, slot: u32, n: u32) -> Result<(Vec<ImageSpan>, u64)> {
    if slot >= m.batch {
        return Err(RuntimeError::Device(format!(
            "KV image: slot {slot} past the packet's {} slots",
            m.batch
        )));
    }
    if n == 0 || n > m.max_ctx {
        return Err(RuntimeError::Device(format!(
            "KV image: {n} rows outside 1..={}",
            m.max_ctx
        )));
    }
    let mut spans = Vec::new();
    let mut image = 0u64;
    for c in &m.caches {
        let elem = if c.scales.is_some() { 1 } else { 2 };
        let lo = if c.window == 0 {
            0
        } else {
            n.saturating_sub(c.window)
        };
        if c.window == 0 && n > c.stride {
            return Err(RuntimeError::Device(format!(
                "KV image: {n} rows past a full cache of {}",
                c.stride
            )));
        }
        let mut tensors: Vec<(u16, u64)> =
            c.pair.iter().map(|&h| (h, (c.hd * elem) as u64)).collect();
        if let Some(s) = c.scales {
            tensors.extend(s.iter().map(|&h| (h, 4u64)));
        }
        for (handle, row) in tensors {
            let slot_bytes = c.heads as u64 * c.stride as u64 * row;
            let base = slot as u64 * slot_bytes;
            match c.layout {
                LAYOUT_ROW_INTERLEAVED => {
                    // `[slot][stride][head][hd]`, full caches only: one span per (head, row).
                    for head in 0..c.heads as u64 {
                        for p in lo as u64..n as u64 {
                            spans.push(ImageSpan {
                                handle: handle as usize,
                                native_off: base + (p * c.heads as u64 + head) * row,
                                image_off: image,
                                bytes: row,
                            });
                            image += row;
                        }
                    }
                }
                LAYOUT_HEAD_MAJOR => {
                    for head in 0..c.heads as u64 {
                        let mut p = lo;
                        while p < n {
                            let at = p & c.mask;
                            // A run ends at `n` or where the ring wraps to index 0.
                            let len = (n - p).min(c.stride - at);
                            let bytes = len as u64 * row;
                            spans.push(ImageSpan {
                                handle: handle as usize,
                                native_off: base + (head * c.stride as u64 + at as u64) * row,
                                image_off: image,
                                bytes,
                            });
                            image += bytes;
                            p += len;
                        }
                    }
                }
                other => {
                    return Err(RuntimeError::Device(format!(
                        "KV image: unknown cache layout {other}"
                    )))
                }
            }
        }
    }
    Ok((spans, image))
}

#[cfg(test)]
mod tests {
    use super::*;
    use plow_asset::live_kv::Cache;

    fn manifest(caches: Vec<Cache>) -> Manifest {
        Manifest {
            version: 1,
            n_cu: 1,
            batch: 4,
            max_ctx: 64,
            position: 0,
            kv_length: 1,
            row_slot: None,
            caches,
            maps: Vec::new(),
            programs: vec!["p".into()],
            splitk: Vec::new(),
        }
    }

    fn cache(pair: [u16; 2], heads: u32, stride: u32, window: u32) -> Cache {
        Cache {
            pair,
            scales: None,
            heads,
            hd: 64,
            stride,
            window,
            mask: if window == 0 { u32::MAX } else { stride - 1 },
            layout: LAYOUT_HEAD_MAJOR,
        }
    }

    #[test]
    fn full_head_major_cache_is_one_run_per_head_and_tensor() {
        let m = manifest(vec![cache([2, 3], 2, 64, 0)]);
        let (spans, total) = plan(&m, 1, 10).unwrap();
        let row = 128u64;
        assert_eq!(spans.len(), 4);
        assert_eq!(total, 4 * 10 * row);
        assert_eq!(
            spans[1],
            ImageSpan {
                handle: 2,
                native_off: 2 * 64 * row + 64 * row,
                image_off: 10 * row,
                bytes: 10 * row
            }
        );
        assert_eq!(spans[2].handle, 3);
    }

    #[test]
    fn a_sliding_ring_keeps_the_window_and_splits_at_the_wrap() {
        let m = manifest(vec![cache([2, 3], 1, 16, 12)]);
        // Rows 28..40 live; ring indices 12..16 then 0..8.
        let (spans, total) = plan(&m, 0, 40).unwrap();
        let row = 128u64;
        assert_eq!(total, 2 * 12 * row);
        assert_eq!(
            &spans[..2],
            &[
                ImageSpan {
                    handle: 2,
                    native_off: 12 * row,
                    image_off: 0,
                    bytes: 4 * row
                },
                ImageSpan {
                    handle: 2,
                    native_off: 0,
                    image_off: 4 * row,
                    bytes: 8 * row
                }
            ]
        );
        // Short prompts start at row 0.
        let (_, short) = plan(&m, 0, 5).unwrap();
        assert_eq!(short, 2 * 5 * row);
    }

    #[test]
    fn fp8_scales_follow_their_caches() {
        let mut c = cache([2, 3], 1, 64, 0);
        c.scales = Some([4, 5]);
        let m = manifest(vec![c]);
        let (spans, total) = plan(&m, 0, 3).unwrap();
        assert_eq!(
            spans.iter().map(|s| s.handle).collect::<Vec<_>>(),
            [2, 3, 4, 5]
        );
        assert_eq!(total, 2 * 3 * 64 + 2 * 3 * 4);
    }

    #[test]
    fn row_interleaved_packs_to_the_head_major_order() {
        let mut c = cache([2, 3], 4, 64, 0);
        let (head_major, total) = plan(&manifest(vec![c.clone()]), 2, 7).unwrap();
        c.layout = LAYOUT_ROW_INTERLEAVED;
        let (spans, same) = plan(&manifest(vec![c]), 2, 7).unwrap();
        assert_eq!(same, total);
        assert_eq!(spans.len(), 2 * 4 * 7);
        let row = 128;
        // Head 1, row 3 of slot 2.
        assert_eq!(spans[10].native_off, 2 * 4 * 64 * row + (3 * 4 + 1) * row);
        assert_eq!(spans[10].image_off, (7 + 3) * row);
        assert_eq!(head_major[1].image_off, 7 * row);
    }

    #[test]
    fn out_of_range_requests_are_refused() {
        let m = manifest(vec![cache([2, 3], 1, 64, 0)]);
        assert!(plan(&m, 4, 1).is_err());
        assert!(plan(&m, 0, 0).is_err());
        assert!(plan(&m, 0, 65).is_err());
    }

    #[test]
    fn image_halves_preserve_all_live_bytes_across_native_layouts_and_wraps() {
        let expand = |spans: &[ImageSpan]| {
            spans
                .iter()
                .flat_map(|s| {
                    (0..s.bytes).map(move |i| (s.image_off + i, s.handle, s.native_off + i))
                })
                .collect::<Vec<_>>()
        };
        for layout in [LAYOUT_HEAD_MAJOR, LAYOUT_ROW_INTERLEAVED] {
            for scales in [false, true] {
                for window in [0, 12] {
                    if layout == LAYOUT_ROW_INTERLEAVED && window != 0 {
                        continue;
                    }
                    for rows in [1, 3, 16, 40] {
                        let mut c = cache([2, 3], 3, if window == 0 { 64 } else { 16 }, window);
                        c.layout = layout;
                        c.scales = scales.then_some([4, 5]);
                        let (spans, total) =
                            plan(&manifest(vec![c, cache([6, 7], 1, 64, 0)]), 2, rows).unwrap();
                        let original = expand(&spans);
                        assert_eq!(original.len() as u64, total);
                        for cut in [1, total / 2, total - 1] {
                            let mut split = Vec::new();
                            for (lo, hi) in [(0, cut), (cut, total)] {
                                let before = split.len();
                                for &s in &spans {
                                    if let Some(piece) = image_span_range(s, lo, hi).unwrap() {
                                        assert!(
                                            piece.image_off >= lo
                                                && piece.image_off + piece.bytes <= hi
                                        );
                                        split.push(piece);
                                    }
                                }
                                assert_eq!(
                                    split[before..].iter().map(|s| s.bytes).sum::<u64>(),
                                    hi - lo
                                );
                            }
                            assert_eq!(expand(&split), original);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn image_ranges_refuse_overflow_and_do_not_copy_empty_intersections() {
        let s = ImageSpan {
            handle: 0,
            native_off: 100,
            image_off: 8,
            bytes: 8,
        };
        for (lo, hi) in [(0, 8), (16, 20), (10, 10)] {
            assert_eq!(image_span_range(s, lo, hi).unwrap(), None);
        }
        assert!(image_span_range(s, 11, 10).is_err());
        assert!(image_span_range(
            ImageSpan {
                image_off: u64::MAX,
                ..s
            },
            0,
            20
        )
        .is_err());
        assert!(image_span_range(
            ImageSpan {
                native_off: u64::MAX,
                ..s
            },
            8,
            16
        )
        .is_err());
    }
}
