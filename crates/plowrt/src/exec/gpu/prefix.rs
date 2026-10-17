//! CUDA KV mappings, prefix snapshots and cache attachment.

use std::sync::Arc;

use crate::asset::devblob::DevBlob;
use crate::config::RuntimeConfig;
use crate::device::cuda::CudaBackend;
use crate::exec::kv_layout::{kv_tensor_name, RingWindow};
use crate::{Result, RuntimeError};

use super::{recurrent_state_layout, GpuEngine};

pub(crate) struct VmmPrefixLayout {
    pub(crate) geo: crate::memory::vmm::VmmGeometry,
    pub(super) slide: Vec<(usize, usize, u64)>,
    pub(super) slide_scale: Vec<(usize, usize)>,
    pub(super) full_scale: Vec<(usize, usize)>,
    pub(super) ring: u64,
    pub(super) snap_row_bytes: u64,
}

/// VMM live allocation or prefix-sharing state: the pool
/// backing every FULL layer's `kv.{l}.k/v` tensor with per-sequence VA
/// windows. Live mode can retain demand-mapped whole-slot rings; prefix
/// mode snapshots the rings' last `window` rows and keeps them flat unless
/// `PLOW_VMM_LIVE_RINGS=1` commits them per admitted slot. `kv` is `None` when live full caches
/// ride `rings` slot by slot (a head window below one VMM granule).
pub(super) struct VmmServe {
    pub(super) kv: Option<crate::memory::vmm::VmmKv>,
    pub(super) rings: Option<crate::memory::vmm::VmmRings>,
    pub(super) tensor_tracks: Vec<(usize, u32, u32)>,
    pub(super) cache_tensors: Vec<usize>,
    /// Per sliding layer: (devp index of `kv.{l}.k`, of `kv.{l}.v`,
    /// per-slot byte stride). K and V share one stride.
    pub(super) slide: Vec<(usize, usize, u64)>,
    /// fp8-KV sliding layers: (devp index of `kv.{l}.k_scale`, of
    /// `kv.{l}.v_scale`) per sliding layer, aligned with `slide`. Empty when
    /// the rings are bf16. Scale rows are f32, one per KV row, same ring.
    pub(super) slide_scale: Vec<(usize, usize)>,
    /// fp8-KV full layers: (devp index of `kv.{l}.k_scale`, of
    /// `kv.{l}.v_scale`) per full layer, aligned with `geometry().full_layers`.
    /// Empty when full-layer KV is bf16/fp16. Full-layer KV rides the VMM
    /// pool but its scales live in flat cudaMalloc tensors that slot reuse
    /// overwrites — so the whole scale PREFIX `[0..p_a)` rides the boundary
    /// snapshot (appended after the rings — see `vmm_snap_copy`).
    pub(super) full_scale: Vec<(usize, usize)>,
    /// Sliding ring rows (`min(max_ctx, KV_RING)`), a power of two.
    pub(super) ring: u64,
    pub(super) snap_row_bytes: u64,
}

impl VmmServe {
    pub(super) fn prefix_kv(&self) -> Option<&crate::memory::vmm::VmmKv> {
        self.kv.as_ref().filter(|kv| kv.prefix_reuse())
    }

    /// Full-cache row mapping; slot-granular rings commit at admission instead.
    pub(super) fn ensure_rows(&self, seq: usize, rows: u32) -> Result<()> {
        match &self.kv {
            Some(kv) if kv.mapped_rows(seq) < rows => kv.ensure_rows(seq, rows),
            _ => Ok(()),
        }
    }

    pub(super) fn advise(&self, seq: usize, pos: u32) {
        if let Some(kv) = &self.kv {
            kv.advise(seq, pos);
        }
    }

    pub(super) fn begin_seq(&self, seq: usize) {
        if let Some(kv) = &self.kv {
            kv.begin_seq(seq);
        }
    }
}

impl VmmPrefixLayout {
    /// Sliding K/V ring bytes over every slot (scales excluded).
    pub(crate) fn slide_kv_bytes(&self, blob: &DevBlob) -> u64 {
        self.slide
            .iter()
            .map(|&(k, v, _)| blob.tensors[k].bytes + blob.tensors[v].bytes)
            .sum()
    }
}

impl GpuEngine {
    pub(super) fn vmm_live_bringup(
        be: &Arc<CudaBackend>,
        blob: &DevBlob,
        live_rings: bool,
        manifest: Option<&plow_asset::live_kv::Manifest>,
    ) -> Result<VmmServe> {
        let mut layout = match manifest {
            Some(m) => crate::memory::vmm::LiveKvLayout::from_manifest(blob, m)?,
            None => crate::memory::vmm::LiveKvLayout::from_blob(blob)?,
        };
        let config = RuntimeConfig::get();
        if let Some(rt_ctx) = config.rt_max_ctx {
            if rt_ctx as u32 > layout.geometry.max_ctx {
                return Err(RuntimeError::Rejected(format!(
                    "PLOW_RT_MAX_CTX {} exceeds VMM geometry max_ctx {}",
                    rt_ctx, layout.geometry.max_ctx
                )));
            }
            let row_bytes = layout.geometry.row_bytes().max(1);
            let gran = (config.vmm_block_mib() as u64) << 20;
            let rows_per_block = ((gran / row_bytes) as u32).max(1);
            let aligned_ctx = ((rt_ctx as u32).min(layout.geometry.max_ctx) + rows_per_block - 1)
                / rows_per_block
                * rows_per_block;
            layout.geometry.max_ctx = aligned_ctx.min(layout.geometry.max_ctx);
        }
        let block_hint = (config.vmm_block_mib() as u64) << 20;
        let ops = Arc::clone(be) as Arc<dyn crate::memory::vmm::VmmOps>;
        if let Some(full) = layout.slot_granular_full(blob, ops.granularity()?, block_hint) {
            // Every cache commits per admitted slot; idle rows write shared scratch.
            let tensors: Vec<_> = full.into_iter().chain(layout.ring_tensors).collect();
            let mut rings = crate::memory::vmm::VmmRings::new_idle_backed(
                ops,
                &tensors,
                layout.geometry.batch as usize,
            )?;
            rings.enable_handle_recycling(crate::memory::vmm::kv_pool_cap());
            tracing::info!(
                slot_mib = rings.slot_charge() >> 20,
                resident_mib = rings.stats().resident_bytes >> 20,
                reserved_gib = rings.stats().reserved_bytes as f64 / (1u64 << 30) as f64,
                "vmm live: full KV committed per slot (head window below one granule)"
            );
            return Ok(VmmServe {
                kv: None,
                rings: Some(rings),
                tensor_tracks: Vec::new(),
                cache_tensors: layout.cache_tensors,
                slide: Vec::new(),
                slide_scale: Vec::new(),
                full_scale: Vec::new(),
                ring: 0,
                snap_row_bytes: 0,
            });
        }
        let rings = if live_rings && !layout.ring_tensors.is_empty() {
            Some(crate::memory::vmm::VmmRings::new(
                Arc::clone(be) as Arc<dyn crate::memory::vmm::VmmOps>,
                &layout.ring_tensors,
                layout.geometry.batch as usize,
            )?)
        } else {
            None
        };
        let mut kv = crate::memory::vmm::VmmKv::new_live(
            Arc::clone(be) as Arc<dyn crate::memory::vmm::VmmOps>,
            layout.geometry,
            block_hint,
        )?;
        kv.enable_block_recycling(crate::memory::vmm::kv_pool_cap());
        // Retiring a sequence unmapped its windows inline on the engine thread: Qwen3.8-27B 4K
        // c32 paid 8-11 ms of idle GPU per retire (256 blocks), ~1 s of a 47 s run.
        if config.vmm_deferred_reclaim() {
            kv.enable_deferred_reclaim();
        }
        let tensor_tracks = layout
            .full_tensors
            .iter()
            .enumerate()
            .flat_map(|(layer, pair)| {
                pair.iter()
                    .enumerate()
                    .map(move |(which, &id)| (id, layer as u32, which as u32))
            })
            .collect();
        Ok(VmmServe {
            kv: Some(kv),
            rings,
            tensor_tracks,
            cache_tensors: layout.cache_tensors,
            slide: Vec::new(),
            slide_scale: Vec::new(),
            full_scale: Vec::new(),
            ring: 0,
            snap_row_bytes: 0,
        })
    }

    pub(crate) fn vmm_prefix_enabled(&self) -> bool {
        self.vmm_prefix().is_some()
    }

    pub(super) fn vmm_prefix(&self) -> Option<(&VmmServe, &crate::memory::vmm::VmmKv)> {
        self.vmm.as_ref().and_then(|v| Some((v, v.prefix_kv()?)))
    }

    pub(crate) fn select_vmm_prefix_layout(
        blob: &DevBlob,
        kv: Option<&plow_asset::serve_manifest::KvGeometry>,
        config: &RuntimeConfig,
        capability: (u32, u32),
        granularity: u64,
    ) -> Option<VmmPrefixLayout> {
        // Overlay rows (audio, voice conditioning) are not keyed by the prompt ids.
        if blob.tensors.iter().any(|t| t.name == "in.encoder_overlay") {
            return None;
        }
        let requested = config.nv_vmm_prefix();
        if requested == Some(false)
            || (requested.is_none()
                && (capability != (9, 0)
                    || config.nv_vmm_live()
                    || config.nv_vmm_live_rings() == Some(true)
                    || blob.tp.is_some()
                    || blob.sections.iter().any(|section| {
                        matches!(
                            section.name.as_str(),
                            plow_asset::mixed_step::SECTION
                                | plow_asset::decode_objects::SECTION
                                | plow_asset::decode_context::SECTION
                        )
                    })))
        {
            return None;
        }
        let layout = Self::vmm_prefix_layout(blob, kv)?;
        if requested.is_none() {
            // Auto-selection is an allowlist of the geometry the sliding-ring
            // snapshot path was qualified on (Gemma 4 hybrid KV), not a
            // capability probe; other layouts need an explicit PLOW_VMM_PREFIX=1.
            // FP8 KV (elem 1, per-token-head scales in the snapshot): Gemma-4 12B,
            // cached and cold multi-turn sessions give the same greedy tokens.
            let g = &layout.geo;
            let qualified = matches!(g.elem, 1 | 2)
                && matches!(g.elem_slide, 1 | 2)
                && g.hd_full == 512
                && g.hd_slide == 256
                // 512: Gemma-4 E4B (KV-shared layers read their source's rings); cached and
                // cold prompts give identical logprobs (docs/runtime/gemma4-e4b-h100.md).
                && matches!(g.window, 512 | 1024)
                && !layout.slide.is_empty()
                && recurrent_state_layout(&blob.tensors, g.batch as usize)
                    .ok()?
                    .is_none();
            tracing::info!(
                selected = qualified,
                hd_full = g.hd_full,
                hd_slide = g.hd_slide,
                window = g.window,
                kv_elem = g.elem,
                kv_elem_slide = g.elem_slide,
                "vmm prefix auto-selection (qualified Hopper hybrid BF16/FP8-KV geometry; \
                 PLOW_PREFIX_CACHE=0 or PLOW_VMM_PREFIX=0 disables, =1 forces)"
            );
            if !qualified {
                return None;
            }
        }
        layout
            .geo
            .block_bytes(granularity, u64::from(config.vmm_block_mib()) << 20)
            .ok()?;
        Some(layout)
    }

    pub(crate) fn vmm_prefix_layout(
        blob: &DevBlob,
        kv: Option<&plow_asset::serve_manifest::KvGeometry>,
    ) -> Option<VmmPrefixLayout> {
        let batch = blob.decode_prog().ok()?.t;
        let packet_max_ctx = blob
            .tensors
            .iter()
            .find(|t| t.name == "in.pos")
            .map(|t| (t.bytes / 4) as u32)?;
        let max_ctx = crate::config::RuntimeConfig::get()
            .rt_max_ctx
            .map(|c| c as u32)
            .unwrap_or(packet_max_ctx)
            .min(packet_max_ctx);
        let Some(mut geo) =
            kv.and_then(|kv| crate::memory::vmm::VmmGeometry::from_kv(kv, max_ctx, batch))
        else {
            tracing::warn!("vmm off: no usable KV geometry in the serve manifest");
            return None;
        };
        let find = |name: &str| blob.tensors.iter().position(|t| t.name == name);

        // KV dtype per layer group, resolved from the blob itself: the
        // emitter declares `kv.{l}.k_scale`/`.v_scale` iff that layer's cache
        // is fp8 e4m3 (1 B/elem + per-row f32 scales); bf16/fp16 layers have
        // no scale tensors and 2 B/elem. Presence is the discriminator —
        // byte-size inference alone is ambiguous (2× ring vs 2× elem).
        let scales_of = |l: u32| -> Option<(usize, usize)> {
            match (
                find(&format!("kv.{l}.k_scale")),
                find(&format!("kv.{l}.v_scale")),
            ) {
                (Some(ik), Some(iv)) => Some((ik, iv)),
                _ => None,
            }
        };
        let full_fp8 = geo.full_layers.first().map(|&l| scales_of(l).is_some());
        geo.elem = if full_fp8 == Some(true) { 1 } else { 2 };

        // Full layers: declared bytes must equal the batch-major shape at the
        // resolved elem, and the fp8 discriminator must be uniform — a layer
        // disagreeing with the first one is geometry drift, not a mode.
        let mut full_scale = Vec::new();
        for &l in &geo.full_layers {
            for t in ["k", "v"] {
                let Some(i) = find(&format!("kv.{l}.{t}")) else {
                    tracing::warn!(layer = l, "vmm off: missing full-layer KV tensor");
                    return None;
                };
                if blob.tensors[i].bytes != geo.full_tensor_bytes() {
                    tracing::warn!(
                        layer = l,
                        declared = blob.tensors[i].bytes,
                        expected = geo.full_tensor_bytes(),
                        "vmm off: full-layer KV bytes mismatch (geometry drift)"
                    );
                    return None;
                }
            }
            match (geo.elem, scales_of(l)) {
                (2, None) => {}
                (1, Some((ik, iv))) => {
                    let want = batch as u64 * geo.kvh_full as u64 * max_ctx as u64 * 4;
                    if blob.tensors[ik].bytes != want || blob.tensors[iv].bytes != want {
                        tracing::warn!(layer = l, "vmm off: full-layer KV scale bytes mismatch");
                        return None;
                    }
                    full_scale.push((ik, iv));
                }
                _ => {
                    tracing::warn!(layer = l, "vmm off: mixed KV dtypes across full layers");
                    return None;
                }
            }
        }

        // Sliding layers: resolve ring geometry for the boundary snapshots.
        // Ring dtype is independent of the full layers' (PLOW_FP8_KV_FULL=1
        // keeps the rings bf16 under fp8 full layers).
        let slide_fp8 = geo.slide_layers.first().map(|&l| scales_of(l).is_some());
        geo.elem_slide = if slide_fp8 == Some(true) { 1 } else { 2 };
        let hd_b = (geo.hd_slide * geo.elem_slide) as u64;
        let mut slide = Vec::with_capacity(geo.slide_layers.len());
        let mut slide_scale = Vec::new();
        let mut ring = 0u64;
        for &l in &geo.slide_layers {
            let (Some(ik), Some(iv)) = (find(&format!("kv.{l}.k")), find(&format!("kv.{l}.v")))
            else {
                tracing::warn!(layer = l, "vmm off: missing sliding KV tensor");
                return None;
            };
            let stride = blob.tensors[ik].bytes / batch as u64;
            let r = stride / (geo.kvh_slide as u64 * hd_b);
            if blob.tensors[iv].bytes != blob.tensors[ik].bytes
                || r * geo.kvh_slide as u64 * hd_b != stride
                || !r.is_power_of_two()
                || r < geo.window as u64
                || (ring != 0 && ring != r)
            {
                tracing::warn!(layer = l, "vmm off: sliding ring geometry mismatch");
                return None;
            }
            ring = r;
            slide.push((ik, iv, stride));
            match (geo.elem_slide, scales_of(l)) {
                (2, None) => {}
                (1, Some((sk, sv))) => {
                    let want = batch as u64 * geo.kvh_slide as u64 * r * 4;
                    if blob.tensors[sk].bytes != want || blob.tensors[sv].bytes != want {
                        tracing::warn!(layer = l, "vmm off: sliding KV scale bytes mismatch");
                        return None;
                    }
                    slide_scale.push((sk, sv));
                }
                _ => {
                    tracing::warn!(layer = l, "vmm off: mixed KV dtypes across sliding layers");
                    return None;
                }
            }
        }
        // Per-row snapshot cost of sliding KV and optional fp8 scales.
        let slide_rows = slide.len() as u64 * 2 * geo.kvh_slide as u64;
        let snap_row_bytes = slide_rows * hd_b
            + if geo.elem_slide == 1 {
                slide_rows * 4
            } else {
                0
            };

        tracing::info!(
            ring,
            window = geo.window,
            max_ctx = geo.max_ctx,
            batch,
            snap_row_bytes,
            slide_layers = geo.slide_layers.len(),
            full_layers = geo.full_layers.len(),
            "VMM prefix layout resolved"
        );

        Some(VmmPrefixLayout {
            geo,
            slide,
            slide_scale,
            full_scale,
            ring,
            snap_row_bytes,
        })
    }

    pub(super) fn vmm_bringup(
        be: &Arc<CudaBackend>,
        blob: &DevBlob,
        layout: Option<VmmPrefixLayout>,
        live_rings: bool,
    ) -> Result<Option<VmmServe>> {
        let Some(VmmPrefixLayout {
            geo,
            slide,
            slide_scale,
            full_scale,
            ring,
            snap_row_bytes,
        }) = layout
        else {
            return Ok(None);
        };
        // Sliding rings committed per admitted slot, idle rows on shared scratch: a slot's ring
        // costs memory only while a request owns it, so admission charges it like its KV rows.
        let rings = if live_rings && !slide.is_empty() {
            let scale_stride = geo.kvh_slide as u64 * ring * 4;
            let tensors: Vec<_> = slide
                .iter()
                .flat_map(|&(k, v, stride)| [(k, stride), (v, stride)])
                .chain(
                    slide_scale
                        .iter()
                        .flat_map(|&(k, v)| [(k, scale_stride), (v, scale_stride)]),
                )
                .map(|(tensor, slot_bytes)| crate::memory::vmm::LiveRingTensor {
                    tensor,
                    slot_bytes,
                })
                .collect();
            let mut rings = crate::memory::vmm::VmmRings::new_idle_backed(
                Arc::clone(be) as Arc<dyn crate::memory::vmm::VmmOps>,
                &tensors,
                geo.batch as usize,
            )?;
            let park = crate::config::RuntimeConfig::get().vmm_ring_park;
            if crate::config::RuntimeConfig::get().vmm_ring_recycle {
                rings.enable_handle_recycling(crate::memory::vmm::kv_pool_cap());
            } else if park > 0 {
                rings.enable_bounded_park(u64::from(park) * rings.slot_charge());
            }
            tracing::info!(
                park,
                slot_mib = rings.slot_charge() >> 20,
                resident_mib = rings.stats().resident_bytes >> 20,
                reserved_gib = rings.stats().reserved_bytes as f64 / (1u64 << 30) as f64,
                "vmm prefix: live sliding rings (idle rows on shared scratch)"
            );
            Some(rings)
        } else {
            None
        };

        // Default sharing block = the driver granularity (2 MiB measured):
        // the finest match unit VMM can map, e.g. 4096 tokens at hd256 bf16 —
        // what makes shared system prompts / multi-turn histories actually
        // hit. Attach cost stays sane because set_access is coalesced over
        // contiguous granule runs (one call per span, not per block). The
        // 128k-dedup campaign can still raise it via PLOW_VMM_BLOCK_MIB=64.
        let rt = crate::config::RuntimeConfig::get();
        let block_hint = (rt.vmm_block_mib() as u64) << 20;
        // `mem_info` total = the card's VRAM; 0 (query refused) keeps the fixed fallback.
        let device_bytes = be.mem_info().map(|(_, total)| total).unwrap_or(0);
        let cache_cap = rt.prefix_cache_cap_bytes(device_bytes);
        match crate::memory::vmm::VmmKv::new(
            Arc::clone(be) as Arc<dyn crate::memory::vmm::VmmOps>,
            geo,
            block_hint,
            cache_cap,
        ) {
            Ok(mut kv) => Ok(Some(VmmServe {
                rings,
                tensor_tracks: blob
                    .tensors
                    .iter()
                    .enumerate()
                    .filter_map(|(id, tensor)| {
                        let (layer, which) = kv_tensor_name(&tensor.name)?;
                        kv.tensor_va(layer, which).map(|_| (id, layer, which))
                    })
                    .collect(),
                cache_tensors: Vec::new(),
                kv: Some({
                    kv.enable_block_pool(crate::memory::vmm::kv_pool_cap());
                    if rt.vmm_deferred_reclaim() {
                        kv.enable_deferred_reclaim();
                        if rt.vmm_stale_reserve() {
                            kv.enable_stale_reserve();
                        }
                    }
                    if rt.vmm_publish_shared() {
                        kv.enable_shared_publish();
                    }
                    // FP8 full-layer scales snapshot over `[0, p_a)`, not just the tail.
                    if !full_scale.is_empty() && rt.vmm_strict_publish() {
                        kv.enable_strict_publish();
                    }
                    let host_cap = rt.vmm_host_cache_bytes();
                    if host_cap > 0 {
                        // The largest snapshot a publish can produce (`vmm_snap_bytes` at
                        // `max_ctx`): a full window, the fp8 full-layer scales over the whole
                        // context and the partial block's rows.
                        let g = kv.geometry();
                        let full_scale_bytes = full_scale.len() as u64
                            * 2
                            * u64::from(g.kvh_full)
                            * u64::from(g.max_ctx)
                            * 4;
                        let partial_bytes = g.full_layers.len() as u64
                            * 2
                            * u64::from(g.kvh_full)
                            * u64::from(kv.block_rows() - 1)
                            * u64::from(g.hd_full * g.elem);
                        let slot_bytes =
                            snap_row_bytes * u64::from(g.window) + full_scale_bytes + partial_bytes;
                        match kv.enable_host_tier(host_cap, slot_bytes) {
                            Ok(()) => tracing::info!(
                                cap_mib = host_cap >> 20,
                                slot_mib = slot_bytes >> 20,
                                slots = host_cap / slot_bytes,
                                "vmm prefix: host tier armed"
                            ),
                            Err(e) => tracing::warn!(error = %e, "vmm prefix: host tier off"),
                        }
                    }
                    kv
                }),
                slide,
                slide_scale,
                full_scale,
                ring,
                snap_row_bytes,
            })),
            Err(e) => {
                tracing::warn!(error = %e, "vmm off: pool bringup failed");
                Ok(None)
            }
        }
    }

    /// D2D-copy the sliding rings' last `window` rows at boundary `p_a`
    /// between slot `b`'s rings and a snapshot buffer (`to_snap` picks the
    /// direction). Layout: `[slide layer][K,V][head][window row][hd]`, rows
    /// ordered by absolute position `p_a-window..p_a`; each head is at most
    /// two runs (ring wrap).
    pub(super) fn vmm_slide_copy(&self, b: usize, p_a: u32, buf: u64, to_snap: bool) -> Result<()> {
        let (v, kv) = self
            .vmm_prefix()
            .expect("vmm_slide_copy without prefix vmm");
        if v.slide.is_empty() {
            return Ok(());
        }
        let g = kv.geometry();
        let span = RingWindow::new(p_a.into(), g.window.into(), v.ring);
        let w = span.rows;
        let hd_b = (g.hd_slide * g.elem_slide) as u64;
        let mut off = buf;
        for &(ik, iv, stride) in &v.slide {
            for idx in [ik, iv] {
                let base = self.devp[idx].base + b as u64 * stride;
                for (dev, snap, rows) in [
                    (base + span.start * hd_b, off, span.first),
                    (base, off + span.first * hd_b, w - span.first),
                ] {
                    if rows == 0 {
                        continue;
                    }
                    let ring = (dev, v.ring * hd_b);
                    let snapshot = (snap, w * hd_b);
                    let (dst, src) = if to_snap {
                        (snapshot, ring)
                    } else {
                        (ring, snapshot)
                    };
                    self.be.memcpy_dtod_pitched_async(
                        dst,
                        src,
                        rows * hd_b,
                        g.kvh_slide,
                        &self.stream,
                    )?;
                }
                off += u64::from(g.kvh_slide) * w * hd_b;
            }
        }
        Ok(())
    }

    /// fp8-KV boundary-snapshot regions past the rings. Region 2 (fp8 rings
    /// only): the ring scales' last `window` rows, same wrap logic as the
    /// rings at 4 B/row. Region 3 (fp8 full layers only): each full layer's
    /// whole scale PREFIX `[0..p_a)` — full-layer scale tensors are flat
    /// cudaMalloc `[batch][kvh][max_ctx]` f32 and slot reuse overwrites them,
    /// so the shared prefix's scales can only survive in the snapshot.
    pub(super) fn vmm_scale_copy(&self, b: usize, p_a: u32, buf: u64, to_snap: bool) -> Result<()> {
        let (v, kv) = self
            .vmm_prefix()
            .expect("vmm_scale_copy without prefix vmm");
        let g = kv.geometry();
        let ring = v.ring;
        let mut off = buf;
        for &(sk, sv) in &v.slide_scale {
            let span = RingWindow::new(p_a.into(), g.window.into(), ring);
            let w = span.rows;
            for idx in [sk, sv] {
                let base = self.devp[idx].base + b as u64 * g.kvh_slide as u64 * ring * 4;
                let heads = g.kvh_slide;
                let (dev, snap) = ((base + span.start * 4, ring * 4), (off, w * 4));
                self.vmm_copy_heads(dev, snap, span.first * 4, heads, to_snap)?;
                let (dev, snap) = ((base, ring * 4), (off + span.first * 4, w * 4));
                self.vmm_copy_heads(dev, snap, (w - span.first) * 4, heads, to_snap)?;
                off += u64::from(heads) * w * 4;
            }
        }
        let ctx = g.max_ctx as u64;
        for &(sk, sv) in &v.full_scale {
            for idx in [sk, sv] {
                let base = self.devp[idx].base + b as u64 * g.kvh_full as u64 * ctx * 4;
                let bytes = p_a as u64 * 4;
                self.vmm_copy_heads((base, ctx * 4), (off, bytes), bytes, g.kvh_full, to_snap)?;
                off += u64::from(g.kvh_full) * bytes;
            }
        }
        Ok(())
    }

    /// Copy `bytes` per KV head between a slot's cache tensor (`dev` = first head, head pitch)
    /// and a snapshot region (`snap`, same), in the `to_snap` direction: one pitched copy for all
    /// heads under `PLOW_VMM_SNAP_PITCHED`, else one copy per head.
    fn vmm_copy_heads(
        &self,
        dev: (u64, u64),
        snap: (u64, u64),
        bytes: u64,
        heads: u32,
        to_snap: bool,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let (dst, src) = if to_snap { (snap, dev) } else { (dev, snap) };
        if RuntimeConfig::get().vmm_snap_pitched {
            return self
                .be
                .memcpy_dtod_pitched_async(dst, src, bytes, heads, &self.stream);
        }
        for h in 0..u64::from(heads) {
            self.be
                .memcpy_dtod_async(dst.0 + h * dst.1, src.0 + h * src.1, bytes, &self.stream)?;
        }
        Ok(())
    }

    /// Bytes region 3 occupies for a boundary at `p_a` rows (0 unless the
    /// full layers are fp8).
    pub(super) fn vmm_full_scale_bytes(&self, p_a: u32) -> u64 {
        let Some((v, kv)) = self.vmm_prefix() else {
            return 0;
        };
        let g = kv.geometry();
        v.full_scale.len() as u64 * 2 * g.kvh_full as u64 * p_a as u64 * 4
    }

    pub(super) fn vmm_snap_bytes(&self, rows: u32) -> u64 {
        let (v, kv) = self.vmm_prefix().expect("prefix snapshot without VMM");
        let g = kv.geometry();
        let partial = u64::from(rows % kv.block_rows());
        (v.snap_row_bytes * u64::from(g.window.min(rows))
            + self.vmm_full_scale_bytes(rows)
            + v.tensor_tracks.len() as u64
                * u64::from(g.kvh_full)
                * partial
                * u64::from(g.hd_full * g.elem))
        .max(4)
    }

    pub(super) fn vmm_partial_copy(
        &self,
        b: usize,
        rows: u32,
        mut buf: u64,
        to_snap: bool,
    ) -> Result<()> {
        let (v, kv) = self.vmm_prefix().expect("partial copy without prefix vmm");
        let g = kv.geometry();
        let partial = rows % kv.block_rows();
        if partial == 0 {
            return Ok(());
        }
        let row_bytes = u64::from(g.hd_full * g.elem);
        let bytes = u64::from(partial) * row_bytes;
        let row0 = u64::from(rows - partial);
        // One copy per head: the heads' rows live in separately mapped VMM blocks, and a pitched
        // copy spanning the unmapped VA between them is rejected (CUDA_ERROR_INVALID_VALUE).
        for &(index, _, _) in &v.tensor_tracks {
            for h in 0..u64::from(g.kvh_full) {
                let dev = self.devp[index].base
                    + ((b as u64 * u64::from(g.kvh_full) + h) * u64::from(g.max_ctx) + row0)
                        * row_bytes;
                if to_snap {
                    self.be.memcpy_dtod_async(buf, dev, bytes, &self.stream)?;
                } else {
                    self.be.memcpy_dtod_async(dev, buf, bytes, &self.stream)?;
                }
                buf += bytes;
            }
        }
        Ok(())
    }

    /// Copy the whole boundary snapshot for slot `b` at boundary `p_a`:
    /// rings, fp8 ring scales, fp8 full-layer scale prefixes, partial full-KV.
    /// `to_snap` picks the direction (publish writes, attach restores).
    pub(super) fn vmm_snap_copy(&self, b: usize, p_a: u32, buf: u64, to_snap: bool) -> Result<()> {
        let t0 = RuntimeConfig::get()
            .pf_packlog
            .then(std::time::Instant::now);
        let copied = (|| {
            self.vmm_slide_copy(b, p_a, buf, to_snap)?;
            let (v, kv) = self.vmm_prefix().expect("vmm_snap_copy without prefix vmm");
            if !v.slide_scale.is_empty() || !v.full_scale.is_empty() {
                let g = kv.geometry();
                let rings = v.slide.len() as u64
                    * 2
                    * g.kvh_slide as u64
                    * g.window.min(p_a) as u64
                    * (g.hd_slide * g.elem_slide) as u64;
                self.vmm_scale_copy(b, p_a, buf + rings, to_snap)?;
            }
            let partial = buf
                + v.snap_row_bytes * u64::from(kv.geometry().window.min(p_a))
                + self.vmm_full_scale_bytes(p_a);
            self.vmm_partial_copy(b, p_a, partial, to_snap)
        })();
        let t1 = t0.map(|t| t.elapsed());
        // Drain even a partially submitted copy before VMM remapping or snapshot release. Inside a
        // launch window a whole restore is drained by the step instead (`deferred_attach`).
        let completed = if self.window_open && !to_snap && copied.is_ok() {
            Ok(())
        } else {
            self.be.stream_synchronize(&self.stream)
        };
        if let (Some(t0), Some(t1)) = (t0, t1) {
            eprintln!(
                "PACKLOG SNAP slot={b} to_snap={} rows={p_a} submit_us={} sync_us={}",
                to_snap as u8,
                t1.as_micros(),
                (t0.elapsed() - t1).as_micros()
            );
        }
        copied.and(completed)
    }

    /// Consult the prefix cache for slot `b`'s prompt and attach a published
    /// prefix: multi-map the shared full-layer blocks, restore the sliding
    /// windows and private partial block, then advance the prefill frontier.
    pub(super) fn vmm_attach(
        &mut self,
        b: usize,
        prompt: &[u32],
        key: Option<crate::memory::vmm::PrefixKey>,
    ) -> Result<()> {
        let staged = self.staged_prefix[b].take();
        let Some((_, kv)) = self.vmm_prefix() else {
            return Ok(());
        };
        let att = kv.try_attach_keyed(b, prompt, key.or(staged))?;
        let attached = att.is_some();
        let att_rows = att.as_ref().map(|a| a.rows).unwrap_or(0);
        tracing::debug!(
            slot = b,
            prompt = prompt.len(),
            attached,
            att_rows,
            "vmm_attach query"
        );
        let Some(a) = att else {
            return Ok(());
        };
        let expected_snap_bytes = self.vmm_snap_bytes(a.rows);
        let restored = (|| {
            if a.snap_bytes != expected_snap_bytes {
                return Err(RuntimeError::Rejected(format!(
                    "boundary snapshot layout drift: cached {} bytes, expected {expected_snap_bytes}",
                    a.snap_bytes
                )));
            }
            if a.rows % kv.block_rows() != 0 {
                kv.ensure_rows(b, a.rows + 1)?;
            }
            self.vmm_snap_copy(b, a.rows, a.snap_va, false)
        })();
        if let Err(error) = restored {
            kv.begin_seq(b);
            if crate::obs::pfx_trace::on() {
                let slot = b as u32;
                crate::obs::pfx_trace::emit(crate::obs::pfx_trace::Event::AttachUnwound { slot });
            }
            return Err(error);
        }
        if self.window_open {
            self.deferred_attach.push(b);
        } else {
            kv.finish_attach(b);
        }
        self.pos[b] = a.rows;
        self.vmm_attached[b] = a.rows;
        // Seed the row-token record with the attached prefix — the tail is
        // appended by prefill completion (full prompt) or per decode feed.
        self.seq_tokens[b].clear();
        self.seq_tokens[b].extend_from_slice(&prompt[..a.rows as usize]);
        tracing::debug!(
            slot = b,
            rows = a.rows,
            prompt = prompt.len(),
            "vmm: prefix attached (whole KV blocks shared, boundary restored)"
        );
        Ok(())
    }

    /// VMM prefix-sharing counters; `None` when `PLOW_VMM_PREFIX` is off.
    pub fn vmm_stats(&self) -> Option<crate::memory::vmm::VmmStats> {
        self.vmm
            .as_ref()
            .and_then(|v| v.kv.as_ref())
            .map(|kv| kv.stats())
    }

    pub fn live_ring_stats(&self) -> Option<crate::memory::vmm::LiveRingStats> {
        self.vmm
            .as_ref()
            .and_then(|v| v.rings.as_ref())
            .map(|r| r.stats())
    }

    /// Engine-lock-free stats reader for `/metrics`; `None` when
    /// `PLOW_VMM_PREFIX` is off.
    pub fn vmm_stats_handle(&self) -> Option<crate::memory::vmm::VmmStatsHandle> {
        self.vmm
            .as_ref()
            .and_then(|v| v.kv.as_ref())
            .map(|kv| kv.stats_handle())
    }

    /// Engine-lock-free attach probe for the mux's admission order; `None` without prefix reuse.
    pub fn vmm_prefix_probe(&self) -> Option<crate::memory::vmm::PrefixProbe> {
        self.vmm
            .as_ref()
            .and_then(|v| v.kv.as_ref()?.prefix_probe())
    }

    /// Rows slot `b`'s current sequence attached from the prefix cache
    /// (0 = cold start). Valid from the first prefill chunk on.
    pub fn attached_rows(&self, b: usize) -> u32 {
        self.vmm_attached.get(b).copied().unwrap_or(0)
    }

    /// Look up a prompt prefix against the radix tree and boundary cache. On
    /// a hit, maps the shared physical blocks, restores the boundary sliding
    /// ring and full scales, and sets `pos[b]` to the matched rows. Returns
    /// the new position (`> 0` on attach, `0` on miss).
    pub fn attach_prompt(&mut self, b: usize, prompt: &[u32]) -> Result<usize> {
        self.attach_prompt_keyed(b, prompt, None)
    }

    /// Whether slot `b` holds a staged key its next attach will consume.
    pub fn has_staged_prefix(&self, b: usize) -> bool {
        self.staged_prefix[b].is_some()
    }

    /// Stage `key` for slot `b`'s next attach inside [`Self::prefill_chunk`]; `None` clears it.
    pub fn stage_prefix_key(&mut self, b: usize, key: Option<crate::memory::vmm::PrefixKey>) {
        self.staged_prefix[b] = key;
    }

    /// [`Self::attach_prompt`] with the prompt's block hashes already computed.
    pub fn attach_prompt_keyed(
        &mut self,
        b: usize,
        prompt: &[u32],
        key: Option<crate::memory::vmm::PrefixKey>,
    ) -> Result<usize> {
        if self.pos[b] == 0 && self.vmm_prefix_enabled() {
            self.vmm_attach(b, prompt, key)?;
        }
        Ok(self.pos[b] as usize)
    }

    fn publish_boundary(&self, b: usize, p_a: u32, kind: crate::memory::vmm::Kind) -> bool {
        let Some((v, kv)) = self.vmm_prefix() else {
            return false;
        };
        let rows = self.pos[b];
        let toks = &self.seq_tokens[b];
        let g = kv.geometry();
        if rows == 0 || toks.len() != rows as usize || p_a == 0 {
            return false;
        }
        // An already published boundary needs no copy: refresh it before the ring-overflow
        // check, so a surviving prompt end is touched however far decode ran past it.
        if kv.touch(b, toks, p_a, kind) {
            tracing::debug!(slot = b, p_a, "vmm: boundary touched");
            return true;
        }
        if !v.slide.is_empty() && rows - p_a > v.ring as u32 - g.window {
            trace_refusal(b, p_a, Refusal::RingWrapped);
            tracing::debug!(
                slot = b,
                rows,
                p_a,
                ring = v.ring,
                window = g.window,
                slack = v.ring as u32 - g.window,
                diff = rows - p_a,
                "vmm: publish_boundary skipped: ring overflow"
            );
            return false;
        }
        if !kv.resolve_prefix_hazard(b, toks, p_a) {
            trace_refusal(b, p_a, Refusal::Hazard);
            tracing::debug!(
                slot = b,
                p_a,
                "vmm: publish_boundary skipped: orphaned blocks"
            );
            return false;
        }
        let snap_bytes = self.vmm_snap_bytes(p_a);
        if let Err(e) = kv.publish_kind(b, toks, p_a, kind, snap_bytes, |dst| {
            self.vmm_snap_copy(b, p_a, dst, true)
        }) {
            tracing::debug!(error = %e, slot = b, p_a, "vmm: publish_boundary skipped");
            trace_refusal(b, p_a, Refusal::Error);
            false
        } else {
            tracing::debug!(slot = b, p_a, snap_bytes, "vmm: published successfully");
            true
        }
    }

    /// Publish slot `b` up to a computed 32-token boundary. Prompt publication
    /// limits rows to prompt_len - 1 so an identical prompt can replay its tail.
    /// Also publishes intermediate checkpoints at every whole-block boundary below it (a
    /// block-aligned boundary has no tail, so any prompt sharing those blocks attaches to
    /// it), or every `PLOW_AMD_PREFIX_FINE_ROWS` rows when that is set.
    pub(super) fn vmm_publish(&self, b: usize, max_rows: u32) {
        use crate::memory::vmm::Kind;
        let Some((v, kv)) = self.vmm_prefix() else {
            return;
        };
        let rows = self.pos[b];
        let toks = &self.seq_tokens[b];
        let g = kv.geometry();
        let prompt = kv.prompt_rows(b);
        let prompt_end = is_prompt_end(rows, max_rows, prompt);
        // Every prompt end, not only a session's: a chat client's next turn re-renders the same
        // last rows whether or not it names a session.
        let max_rows = if prompt_end {
            session_prompt_end(prompt)
        } else {
            max_rows
        };
        let p_a = (rows.min(max_rows) / 32) * 32;
        tracing::debug!(
            slot = b,
            rows,
            toks_len = toks.len(),
            max_rows,
            p_a,
            ring = v.ring,
            window = g.window,
            "vmm_publish entered"
        );
        if rows == 0 || toks.len() != rows as usize {
            tracing::debug!(
                slot = b,
                rows,
                toks_len = toks.len(),
                "vmm_publish skipped: rows/toks mismatch"
            );
            trace_refusal(b, p_a, Refusal::NotPublishable);
            return;
        }
        if p_a == 0 {
            tracing::debug!(slot = b, p_a, "vmm_publish skipped: p_a == 0");
            trace_refusal(b, p_a, Refusal::NotPublishable);
            return;
        }
        let step = crate::config::RuntimeConfig::get()
            .amd_prefix_fine_rows()
            .map_or(kv.block_rows(), |step| step.max(32));
        // A conversation's next turn attaches to its prompt end, with or without `X-Session-Id`;
        // a whole-block checkpoint or chunk end inside the prompt only serves a request waiting
        // on this prefill (`inflight_prefix`), and a shared prefix end publishes through
        // `vmm_publish_shared`. Every other checkpoint is a private window copy
        // (`vmm_snap_bytes`, ~100 MiB at Gemma-4 26B) whose trim evicts another conversation's
        // prompt end: published unconditionally, a 6K-token chat turn wrote three such copies
        // and the cache retained a prompt end for ~20 s against think times of minutes.
        let mut p = step;
        while p < p_a {
            if kv.checkpoint_awaited(toks, p) {
                self.publish_boundary(b, p, Kind::CHECKPOINT);
            }
            p += step;
        }
        if !prompt_end && rows < prompt && !kv.checkpoint_awaited(toks, p_a) {
            return;
        }
        // A turn end (prompt + reply) is never published: the next prompt re-renders the reply
        // and diverges before it (0 of 1223 attaches on Gemma 4); the prompt end below is
        // restored or touched instead. The boundary tree retires structurally inside the publish.
        let turn_end = rows > prompt;
        if !turn_end {
            let kind = if prompt_end {
                Kind::PROMPT_END
            } else {
                Kind::CHECKPOINT
            };
            self.publish_boundary(b, p_a, kind);
        }
        // Turn end: pressure may have evicted the prompt end (the next turn's attach point)
        // while this turn decoded. The rings still hold its window, so restore it now, newest in
        // LRU just before the next request; a surviving one is only touched. A client without
        // `X-Session-Id` continues its conversation the same way.
        let end = session_prompt_end(prompt) / 32 * 32;
        if rows > prompt && end > 0 && end < p_a {
            self.publish_boundary(b, end, Kind::PROMPT_END);
        }
    }

    /// Publish slot `b`'s shared-prefix end (`VmmKv::share_rows`, e.g. a system prompt
    /// another sequence also sent) once the prefill chunk that started at `c0` passed it,
    /// while the rings still hold its window.
    pub(super) fn vmm_publish_shared(&self, b: usize, c0: u32) {
        let Some((_, kv)) = self.vmm_prefix() else {
            return;
        };
        let rows = kv.share_rows(b);
        if rows > c0 && rows <= self.pos[b] {
            self.publish_boundary(b, rows, crate::memory::vmm::Kind::SHARED_END);
        }
    }

    /// Longest prefill slice from `c0` that leaves slot `b`'s shared-prefix end publishable:
    /// the rings hold only `ring - window` rows past a boundary (`publish_boundary`), and a
    /// request slice may be wider than that.
    pub fn pf_publish_cap(&self, b: usize, c0: usize) -> usize {
        let Some((v, kv)) = self.vmm_prefix().filter(|(v, _)| !v.slide.is_empty()) else {
            return usize::MAX;
        };
        let share = kv.share_rows(b) as usize;
        if share <= c0 {
            return usize::MAX;
        }
        share + (v.ring as usize).saturating_sub(kv.geometry().window as usize) - c0
    }

    /// `PLOW_PFX_TRACE`: slot `b`'s sequence ended with these computed rows.
    pub(super) fn trace_finish(&self, b: usize) {
        if crate::obs::pfx_trace::on() && self.vmm_prefix().is_some() && self.pos[b] > 0 {
            crate::obs::pfx_trace::emit(crate::obs::pfx_trace::Event::Finish {
                t: std::time::Instant::now(),
                slot: b as u32,
                tokens: self.seq_tokens[b].clone(),
            });
        }
    }

    /// Slot `b`'s prompt is prefilled and its prompt-end publish has run.
    pub(super) fn vmm_prefill_done(&self, b: usize) {
        if let Some((_, kv)) = self.vmm_prefix() {
            kv.prefill_done(b);
        }
    }

    /// Another slot is prefilling a prompt whose whole-block checkpoint will cover more of
    /// `prompt` than the cache can attach now: `(owner slot, rows)`. A boundary is only
    /// publishable while the sliding rings still hold its window (`publish_boundary`).
    pub(super) fn vmm_inflight_prefix(&self, b: usize, prompt: &[u32]) -> Option<(usize, u32)> {
        let (v, kv) = self.vmm_prefix()?;
        let lookback = if v.slide.is_empty() {
            u32::MAX
        } else {
            v.ring as u32 - kv.geometry().window
        };
        kv.inflight_prefix(b, prompt, lookback, &self.pos)
    }
}

use crate::obs::pfx_trace::Refusal;

fn trace_refusal(b: usize, rows: u32, cause: Refusal) {
    if crate::obs::pfx_trace::on() {
        crate::obs::pfx_trace::emit(crate::obs::pfx_trace::Event::Refuse {
            t: std::time::Instant::now(),
            slot: b as u32,
            rows,
            cause,
        });
    }
}

/// Arm `PLOW_PFX_TRACE` with the prefix pool's geometry and byte costs (a no-op unset).
pub(super) fn trace_init(v: &VmmServe) {
    let Some(kv) = v.kv.as_ref().filter(|kv| kv.prefix_reuse()) else {
        return;
    };
    let g = kv.geometry();
    let mut h = crate::obs::pfx_trace::Header {
        full_layers: g.full_layers.len() as u32,
        kvh_full: g.kvh_full,
        hd_full: g.hd_full,
        elem_full: g.elem,
        full_scale: !v.full_scale.is_empty(),
        slide_layers: v.slide.len() as u32,
        kvh_slide: g.kvh_slide,
        hd_slide: g.hd_slide,
        elem_slide: g.elem_slide,
        window: g.window,
        ring: v.ring,
        max_ctx: g.max_ctx,
        batch: g.batch,
        snap_row_bytes: v.snap_row_bytes,
        host_cap: crate::config::RuntimeConfig::get().vmm_host_cache_bytes(),
        boundary_quantum: 32,
        prompt_end_backoff: 1 + SESSION_PROMPT_REWRITE,
        ..Default::default()
    };
    kv.trace_header(&mut h);
    crate::obs::pfx_trace::init(h);
}

/// Rows a prompt-end boundary may cover: the next turn re-renders the last few prompt
/// rows (Gemma 4 drops the generation prompt's empty thought channel, 4 tokens), so a boundary
/// within them never attaches — about one turn in eight missed that way.
const SESSION_PROMPT_REWRITE: u32 = 8;

/// The publish at the end of a prompt's prefill: the whole prompt with the last row held back
/// (`max_rows < rows`), or a packed pass that withheld the last prompt token (`pos` and
/// `max_rows` both `prompt - 1`). A chunk-end publish caps `max_rows` below `rows`.
fn is_prompt_end(rows: u32, max_rows: u32, prompt: u32) -> bool {
    (rows == prompt && max_rows < rows) || (rows + 1 == prompt && max_rows == rows)
}

fn session_prompt_end(prompt: u32) -> u32 {
    prompt.saturating_sub(1 + SESSION_PROMPT_REWRITE)
}

#[cfg(test)]
#[test]
fn prompt_end_covers_the_withheld_last_token() {
    assert!(
        is_prompt_end(3000, 2999, 3000),
        "whole prompt, last row held back"
    );
    assert!(
        is_prompt_end(2999, 2999, 3000),
        "packed pass withheld the last prompt token"
    );
    assert!(!is_prompt_end(2048, 2047, 3000), "chunk end");
    assert!(!is_prompt_end(2999, 2048, 3000), "chunk-capped publish");
    assert!(!is_prompt_end(3100, 3099, 3000), "turn end");
}
