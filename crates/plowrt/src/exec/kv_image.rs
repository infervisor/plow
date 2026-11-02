//! A request's KV cache as an image another engine of the same variant can import
//! (docs/arch/30 §5): the byte copies between one sequence slot of a packet's declared caches
//! and a flat host image.
//!
//! Pure functions over the packet's `live_kv` manifest, so the addressing is tested without a
//! device. The image holds, per declared cache (KV-shared readers are declared once), per tensor
//! (K, V, then the FP8 row scales), per head, the live rows `lo..n` in position order:
//! `lo = 0` for a full cache, `n - window` for a sliding one. A ring is unwrapped, so the image
//! does not depend on where the ring stood. Rows at or past `n` are never addressed: under live KV
//! they may be unmapped.
//!
//! Two engines exchange images only when [`compat_key`] matches. It covers the cache geometry and
//! the program identities, which pins the same packet; a cross-vendor image needs the semantic
//! `kv_image` section instead.

use plow_asset::live_kv::{Manifest, LAYOUT_HEAD_MAJOR, LAYOUT_ROW_INTERLEAVED};
use sha2::{Digest, Sha256};

use crate::{Result, RuntimeError};

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

/// The compatibility key two engines' images must share.
pub(crate) fn compat_key(m: &Manifest) -> String {
    let mut h = Sha256::new();
    h.update(b"plow-kv-image-v1");
    h.update(m.max_ctx.to_le_bytes());
    for c in &m.caches {
        for v in [c.heads, c.hd, c.stride, c.window, c.mask, c.layout] {
            h.update(v.to_le_bytes());
        }
        h.update([u8::from(c.scales.is_some())]);
    }
    for p in &m.programs {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p.as_bytes());
    }
    format!("{:x}", h.finalize())
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
                    // `[slot][stride][head][hd]`, full caches only: one run.
                    let bytes = (n - lo) as u64 * c.heads as u64 * row;
                    spans.push(ImageSpan {
                        handle: handle as usize,
                        native_off: base + lo as u64 * c.heads as u64 * row,
                        image_off: image,
                        bytes,
                    });
                    image += bytes;
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
    fn row_interleaved_is_one_prefix() {
        let mut c = cache([2, 3], 4, 64, 0);
        c.layout = LAYOUT_ROW_INTERLEAVED;
        let m = manifest(vec![c]);
        let (spans, _) = plan(&m, 2, 7).unwrap();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].native_off, 2 * 4 * 64 * 128);
        assert_eq!(spans[0].bytes, 7 * 4 * 128);
    }

    #[test]
    fn out_of_range_requests_are_refused() {
        let m = manifest(vec![cache([2, 3], 1, 64, 0)]);
        assert!(plan(&m, 4, 1).is_err());
        assert!(plan(&m, 0, 0).is_err());
        assert!(plan(&m, 0, 65).is_err());
        assert_ne!(
            compat_key(&m),
            compat_key(&manifest(vec![cache([2, 3], 1, 64, 8)]))
        );
    }
}
