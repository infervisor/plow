//! CUDA KV mappings, prefix snapshots and cache attachment.

use std::sync::Arc;

use crate::asset::devblob::DevBlob;
use crate::config::RuntimeConfig;
use crate::device::cuda::CudaBackend;
use crate::exec::kv_layout::kv_tensor_name;
use crate::{Result, RuntimeError};

use super::GpuEngine;

const SPEECH_IDENTITIES: usize = 1 << 20;

pub(super) struct SpeechIdentities {
    next: u32,
    rows: rustc_hash::FxHashMap<u64, u32>,
}

impl SpeechIdentities {
    pub(super) fn new(vocab: usize) -> Self {
        Self {
            next: u32::try_from(vocab).unwrap_or(u32::MAX),
            rows: Default::default(),
        }
    }

    fn encode(&mut self, keys: &[u64]) -> Option<Arc<[u32]>> {
        let mut ids = Vec::with_capacity(keys.len());
        for key in keys {
            let id = if let Some(&id) = self.rows.get(key) {
                id
            } else {
                if self.rows.len() == SPEECH_IDENTITIES || self.next == u32::MAX {
                    return None;
                }
                let id = self.next;
                self.next += 1;
                // Never recycle an identity while a VMM boundary or store block can name it.
                self.rows.insert(*key, id);
                id
            };
            ids.push(id);
        }
        Some(ids.into())
    }
}

pub(super) struct SlotPrefixes {
    rows: Vec<Vec<u64>>,
    backing: Vec<Option<u64>>,
    pub(super) epoch: u64,
}

impl SlotPrefixes {
    pub(super) fn new(slots: usize) -> Self {
        Self {
            rows: vec![Vec::new(); slots],
            backing: vec![None; slots],
            epoch: 0,
        }
    }

    pub(super) fn take(&mut self, slot: usize, keys: &[u64], backing: Option<u64>) -> usize {
        let old = std::mem::take(&mut self.rows[slot]);
        if !self.backing_matches(slot, backing) {
            return 0;
        }
        old.iter()
            .zip(keys)
            .take(keys.len().saturating_sub(1))
            .take_while(|(a, b)| a == b)
            .count()
    }

    pub(super) fn publish(&mut self, slot: usize, mut keys: Vec<u64>, backing: Option<u64>) {
        keys.truncate(keys.len().saturating_sub(1));
        self.rows[slot] = keys;
        self.backing[slot] = backing;
    }

    pub(super) fn backing_matches(&self, slot: usize, backing: Option<u64>) -> bool {
        self.backing[slot] == backing
    }

    pub(super) fn retained_rows(&self, slot: usize, backing: Option<u64>) -> usize {
        if self.backing_matches(slot, backing) {
            self.rows[slot].len()
        } else {
            0
        }
    }

    pub(super) fn clear(&mut self, slot: usize) {
        self.rows[slot].clear();
        self.backing[slot] = None;
    }

    pub(super) fn clear_all(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        for keys in &mut self.rows {
            keys.clear();
        }
        self.backing.fill(None);
    }
}

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
    /// (the store's `Scale` class).
    pub(super) full_scale: Vec<(usize, usize)>,
    /// Sliding ring rows (`min(max_ctx, KV_RING)`), a power of two.
    pub(super) ring: u64,
    pub(super) snap_row_bytes: u64,
    /// Boundaries live in the block store (`kv_blocks.rs`); `None` = no prefix cache.
    pub(super) blocks: Option<std::sync::Arc<super::kv_blocks::BlockServe>>,
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
    /// Live sliding rings (`VmmRings::new_idle_backed`): `(demand, idle)` = the K/V ring bytes
    /// committed per admitted slot, and the shared idle units mapped under every free slot.
    /// A ring whose per-slot stride is not a granule multiple is committed whole (resident, in
    /// neither figure), as are the fp8 ring scales.
    pub(crate) fn slide_ring_bytes(&self, blob: &DevBlob, granularity: u64) -> (u64, u64) {
        let mut idle: Vec<u64> = Vec::new();
        let mut demand = 0;
        for &(k, v, stride) in &self.slide {
            if granularity == 0 || stride % granularity != 0 {
                continue;
            }
            demand += blob.tensors[k].bytes + blob.tensors[v].bytes;
            if !idle.contains(&stride) {
                idle.push(stride);
            }
        }
        (demand, idle.iter().sum())
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
                blocks: None,
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
            blocks: None,
        })
    }

    pub(crate) fn vmm_prefix_enabled(&self) -> bool {
        self.vmm_prefix().is_some()
    }

    pub(super) fn vmm_prefix(&self) -> Option<(&VmmServe, &crate::memory::vmm::VmmKv)> {
        self.vmm.as_ref().and_then(|v| Some((v, v.prefix_kv()?)))
    }

    /// `row_interleaved`: the LIVE manifest's full-cache row layout
    /// ([`plow_asset::live_kv::Manifest::full_rows_interleaved`]); `None` = mixed, no layout.
    pub(crate) fn select_vmm_prefix_layout(
        blob: &DevBlob,
        kv: Option<&plow_asset::serve_manifest::KvGeometry>,
        config: &RuntimeConfig,
        capability: (u32, u32),
        granularity: u64,
        row_interleaved: Option<bool>,
    ) -> Option<VmmPrefixLayout> {
        let declined = |reason: &str| {
            tracing::info!(reason, "VMM prefix cache declined");
            None
        };
        // Overlay rows (audio, voice conditioning) are not keyed by the prompt ids.
        if blob.tensors.iter().any(|t| t.name == "in.encoder_overlay") {
            return declined("encoder overlay rows are not keyed by the prompt ids");
        }
        let requested = config.nv_vmm_prefix();
        // GB202 (sm_120) auto-selects the cache for every packet with a layout, rank groups
        // included. Hopper keeps the geometry allowlist it was qualified on; other targets
        // need an explicit PLOW_VMM_PREFIX=1.
        let gb202 = capability == (12, 0);
        if requested == Some(false)
            || (requested.is_none()
                && ((capability != (9, 0) && !gb202)
                    || config.nv_vmm_live()
                    || config.nv_vmm_live_rings() == Some(true)
                    || (blob.tp.is_some() && !gb202)
                    || blob.sections.iter().any(|section| {
                        matches!(
                            section.name.as_str(),
                            plow_asset::mixed_step::SECTION
                                | plow_asset::decode_objects::SECTION
                                | plow_asset::decode_context::SECTION
                        )
                    })))
        {
            return declined(if requested == Some(false) {
                "PLOW_VMM_PREFIX=0"
            } else {
                "auto mode: not Hopper/GB202, live KV/rings, TP off GB202, or a mixed-step/decode-object packet"
            });
        }
        let Some(layout) = Self::vmm_prefix_layout(blob, kv, row_interleaved) else {
            return declined("no VMM prefix KV layout for this packet");
        };
        if gb202 {
            // KV element widths the snapshot path stores (BF16 rows, FP8 rows with
            // per-token-head scales); no per-geometry allowlist on this target.
            let g = &layout.geo;
            if !matches!(g.elem, 1 | 2) || !matches!(g.elem_slide, 1 | 2) {
                return declined("KV element width outside the snapshot path (1 or 2 bytes)");
            }
            tracing::info!(
                hd_full = g.hd_full,
                hd_slide = g.hd_slide,
                window = g.window,
                kv_elem = g.elem,
                kv_elem_slide = g.elem_slide,
                mode = if requested.is_none() {
                    "auto"
                } else {
                    "explicit"
                },
                "vmm prefix layout accepted (PLOW_PREFIX_CACHE=0 or PLOW_VMM_PREFIX=0 disables)"
            );
        } else if requested.is_none() {
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
                && !blob
                    .sections
                    .iter()
                    .any(|s| s.name == plow_asset::recurrent_state::SECTION)
                && !blob
                    .tensors
                    .iter()
                    .any(|t| t.name.starts_with("state.") || t.name == "in.active");
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
                return declined("auto mode: geometry outside the qualified allowlist");
            }
        }
        if let Err(e) = layout
            .geo
            .block_bytes(granularity, u64::from(config.vmm_block_mib()) << 20)
        {
            return declined(&e.to_string());
        }
        Some(layout)
    }

    pub(crate) fn vmm_prefix_layout(
        blob: &DevBlob,
        kv: Option<&plow_asset::serve_manifest::KvGeometry>,
        row_interleaved: Option<bool>,
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
        // Head-interleaved full-layer rows (`[slot][ctx][head][hd]`, Qwen3.8): one position's
        // heads are one row, so the geometry folds the heads into the row and every track is
        // one head of `kvh * hd` elements; the mapping, the tail blocks and the partial copies
        // then follow the rows the kernels write. Interleaved fp8 scales have no layout here.
        match row_interleaved {
            Some(false) => {}
            Some(true) => {
                if geo
                    .full_layers
                    .iter()
                    .any(|&l| find(&format!("kv.{l}.k_scale")).is_some())
                {
                    tracing::warn!(
                        "vmm off: head-interleaved fp8 full-layer scales are not laid out"
                    );
                    return None;
                }
                geo.hd_full *= geo.kvh_full;
                geo.kvh_full = 1;
            }
            None => {
                tracing::warn!("vmm off: full-layer caches disagree on their row layout");
                return None;
            }
        }

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

    /// `state_tensors`: the engine's recurrent state tensors (`(devp index, per-slot bytes)`),
    /// the block store's `State` class; empty for a model without one. `rank_group`: a
    /// tensor-parallel rank, whose cache must decide every attach exactly as its peers do —
    /// no host tier (sized from this process's view of free host RAM, and refilled by a
    /// background thread).
    pub(super) fn vmm_bringup(
        be: &Arc<CudaBackend>,
        blob: &DevBlob,
        layout: Option<VmmPrefixLayout>,
        live_rings: bool,
        state_tensors: &[(usize, u64)],
        rank_group: bool,
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
            // Every released slot parks its ring units in place (bounded against the block store at
            // arm time, `bound_ring_park`): remapping a slot's ring on readmission (unmap +
            // cuMemCreate + map + set_access per unit, serial on the mux tick) cost 26B BF16 at
            // 1.8 sessions/s TTFT P99 1688 ms against 680 ms with every slot parked.
            let park = geo.batch;
            rings.enable_bounded_park(u64::from(park) * rings.slot_charge());
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
            Ok(mut kv) => {
                let mut serve = VmmServe {
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
                        kv
                    }),
                    slide,
                    slide_scale,
                    full_scale,
                    ring,
                    snap_row_bytes,
                    blocks: None,
                };
                if serve.prefix_kv().is_some() {
                    // Host tier: what the host-reserve rule allows now, unless the knob names
                    // a size (cut to the same rule by the store's growth check) or 0 (off).
                    let auto = crate::memory::vmm::host_tier_auto();
                    let host_cap =
                        crate::memory::vmm::host_tier_grant(match rt.host_kvcache_bytes() {
                            _ if rank_group => 0,
                            None => auto.map_or(0, |(cap, _)| cap),
                            Some(cap) => cap,
                        });
                    let host_on = host_cap.bytes() > 0;
                    match super::kv_blocks::BlockServe::new(
                        be,
                        &serve,
                        if cache_cap > 0 {
                            cache_cap
                        } else {
                            device_bytes / 4
                        },
                        host_cap,
                        state_tensors,
                    ) {
                        Ok(blocks) => {
                            if let Some((_, reserve)) = auto {
                                blocks.set_host_reserve(reserve);
                            }
                            let kv = serve.kv.as_mut().expect("prefix pool");
                            kv.enable_block_store(
                                std::sync::Arc::clone(&blocks)
                                    as std::sync::Arc<dyn crate::memory::vmm::SnapBlocks>,
                                true,
                            );
                            if host_on {
                                kv.spawn_spill_ahead(
                                    blocks.store(),
                                    super::kv_blocks::SPILL_AHEAD_BYTES,
                                );
                            }
                            serve.blocks = Some(blocks);
                        }
                        // The store is the boundary storage: without it nothing publishes
                        // or attaches (the live pool still serves the slots).
                        Err(e) => tracing::warn!(error = %e, "kv store: off, no prefix cache"),
                    }
                }
                Ok(Some(serve))
            }
            Err(e) => {
                tracing::warn!(error = %e, "vmm off: pool bringup failed");
                Ok(None)
            }
        }
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
        // Every prefill bucket writes all its rows: a tail left fewer rows of context than the
        // narrowest bucket cannot run after the prefix, while the cold prompt (chunked from 0)
        // can. Serve such a prompt cold.
        let room = self.max_ctx.saturating_sub(a.rows as usize);
        let narrowest = self.prefill.iter().map(|p| p.t as usize).min();
        if prompt.len() > a.rows as usize && narrowest.is_some_and(|t| t > room) {
            kv.begin_seq(b);
            tracing::debug!(
                slot = b,
                rows = a.rows,
                room,
                "vmm: prefix attach declined, tail too near the context limit"
            );
            return Ok(());
        }
        let restored = (|| {
            // The slot's own blocks hold the prefix: map them, then gather every row.
            kv.ensure_rows(b, a.rows + 1)?;
            match a.blocks {
                Some(handle) => self.blocks_gather(b, handle),
                None => Err(RuntimeError::Rejected(
                    "boundary without store blocks".into(),
                )),
            }
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
        self.dump_slot_kv(b, "attach", a.rows);
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

    /// Whether slot `b`'s next sequence may attach a cached prefix: `false` for prompts whose rows
    /// are not keyed by their ids alone (speech frame codes staged per chunk), which then publish
    /// none either until retired.
    pub fn set_prefix_attach(&mut self, b: usize, on: bool) {
        self.prefix_attach_off[b] = !on;
        if !on {
            self.staged_speech_prefix[b] = None;
        }
    }

    pub fn stage_speech_prefix(&mut self, b: usize, keys: &[u64]) -> bool {
        self.staged_speech_prefix[b] = (self.vmm_prefix_enabled() && self.packed_depth_enabled())
            .then(|| self.speech_identities.encode(keys))
            .flatten();
        let ready = self.staged_speech_prefix[b].is_some();
        self.set_prefix_attach(b, ready);
        ready
    }

    pub(super) fn record_prompt_prefix(&mut self, b: usize, prompt: &[u32], end: usize) {
        let ids = self.speech_prefix[b].as_deref().unwrap_or(prompt);
        self.seq_tokens[b].clear();
        self.seq_tokens[b].extend_from_slice(&ids[..end]);
    }

    /// [`Self::attach_prompt`] with the prompt's block hashes already computed.
    pub fn attach_prompt_keyed(
        &mut self,
        b: usize,
        prompt: &[u32],
        key: Option<crate::memory::vmm::PrefixKey>,
    ) -> Result<usize> {
        if self.pos[b] == 0 && self.vmm_prefix_enabled() && !self.prefix_attach_off[b] {
            let ids = self.speech_prefix[b].clone();
            if ids.is_some() {
                self.staged_prefix[b] = None;
            }
            let key = if ids.is_some() { None } else { key };
            self.vmm_attach(b, ids.as_deref().unwrap_or(prompt), key)?;
        }
        Ok(self.pos[b] as usize)
    }

    pub(super) fn publish_boundary(
        &self,
        b: usize,
        p_a: u32,
        kind: crate::memory::vmm::Kind,
    ) -> bool {
        if self.publish_off[b] {
            return false;
        }
        if self.speech_prefix[b]
            .as_ref()
            .is_some_and(|ids| p_a as usize >= ids.len())
        {
            return false;
        }
        let Some((v, kv)) = self.vmm_prefix().filter(|_| !self.prefix_attach_off[b]) else {
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
        // Recurrent state exists only at the slot's current row: a boundary the slot has
        // moved past has no state to capture (`state_capture_publish` publishes it in time).
        if v.blocks.as_ref().is_some_and(|bs| bs.captures_state()) && rows != p_a {
            trace_refusal(b, p_a, Refusal::NotPublishable);
            return false;
        }
        // Rejected speculative rows past `pos` were written too: the ring wrapped to the high water.
        let written = rows.max(self.kv_hw[b]);
        if !v.slide.is_empty() && written - p_a > v.ring as u32 - g.window {
            trace_refusal(b, p_a, Refusal::RingWrapped);
            tracing::debug!(
                slot = b,
                rows,
                p_a,
                ring = v.ring,
                window = g.window,
                slack = v.ring as u32 - g.window,
                diff = written - p_a,
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
        if v.blocks.is_none() {
            trace_refusal(b, p_a, Refusal::NotPublishable);
            return false;
        }
        if let Err(e) = kv.publish_blocks(b, toks, p_a, kind, || self.blocks_publish(b, p_a)) {
            tracing::debug!(error = %e, slot = b, p_a, "vmm: publish_boundary skipped");
            trace_refusal(b, p_a, Refusal::Error);
            false
        } else {
            tracing::debug!(slot = b, p_a, "vmm: published successfully");
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
        let max_rows = self.speech_prefix[b].as_ref().map_or(max_rows, |ids| {
            max_rows.min(ids.len().saturating_sub(1) as u32)
        });
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
        // (a whole window, ~100 MiB at Gemma-4 26B) whose trim evicts another conversation's
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
        host_cap: v.blocks.as_ref().map_or(0, |b| b.host_cap()),
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

/// The 32-row boundary a prompt of `prompt` rows publishes as its prompt end.
pub(super) fn prompt_end_boundary(prompt: u32) -> u32 {
    session_prompt_end(prompt) / 32 * 32
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

#[cfg(test)]
mod slot_prefix_tests {
    use super::{SlotPrefixes, SpeechIdentities};

    #[test]
    fn reclaimed_slot_prefix_is_refused_even_after_the_slot_is_mapped_again() {
        let mut cache = SlotPrefixes::new(1);
        cache.publish(0, vec![1, 2, 3], Some(10));
        assert_eq!(cache.retained_rows(0, Some(10)), 2);
        assert_eq!(cache.retained_rows(0, None), 0);
        assert_eq!(cache.take(0, &[1, 2, 3], Some(11)), 0);
        cache.publish(0, vec![1, 2, 3], Some(12));
        assert_eq!(cache.take(0, &[1, 2, 3], Some(12)), 2);
        assert!(cache.backing_matches(0, Some(12)));
        assert!(
            !cache.backing_matches(0, Some(13)),
            "a staged resume also needs its original backing"
        );
        cache.clear_all();
        assert!(!cache.backing_matches(0, Some(12)));
    }

    #[test]
    fn speech_identities_preserve_content_and_never_alias_token_ids() {
        let mut ids = SpeechIdentities::new(8194);
        let a = ids.encode(&[10, 20, 30, 20]).unwrap();
        let b = ids.encode(&[10, 21, 30, 20]).unwrap();
        assert!(a.iter().all(|&id| id >= 8194));
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
        assert_eq!(a[2..], b[2..]);
        assert_eq!(a[1], a[3]);
        assert_eq!(&*a, &*ids.encode(&[10, 20, 30, 20]).unwrap());
        ids.next = u32::MAX;
        assert!(ids.encode(&[99]).is_none());
        assert_eq!(&*a, &*ids.encode(&[10, 20, 30, 20]).unwrap());
    }

    #[test]
    fn slot_prefix_keeps_only_matching_prompt_rows_and_always_replays_the_last() {
        let mut cache = SlotPrefixes::new(2);
        cache.publish(0, vec![10, 20, 30, 40], None);
        assert_eq!(cache.take(1, &[10, 20, 30, 50], None), 0);
        assert_eq!(cache.take(0, &[10, 20, 30, 50], None), 3);
        assert_eq!(
            cache.take(0, &[10, 20, 30, 50], None),
            0,
            "lookup consumes provenance"
        );
        cache.publish(0, vec![10, 20, 30, 40], None);
        assert_eq!(
            cache.take(0, &[10, 20], None),
            1,
            "the new prompt keeps a terminal row"
        );
        cache.publish(0, vec![10, 20, 30, 40], None);
        assert_eq!(cache.take(0, &[10, 99, 30, 40], None), 1);
        cache.publish(0, vec![10, 20, 30, 40], None);
        assert_eq!(
            cache.take(0, &[10, 20, 30, 40, 50], None),
            3,
            "decode-written rows are excluded"
        );
    }

    #[test]
    fn slot_prefix_requires_explicit_successful_publication_after_reseating() {
        let mut cache = SlotPrefixes::new(2);
        for slot in 0..2 {
            cache.publish(slot, vec![1, 2, 3], None);
        }
        cache.clear(0);
        assert_eq!(cache.take(0, &[1, 2, 3], None), 0);
        assert_eq!(cache.take(1, &[1, 2, 3], None), 2);
        // Refused resume or failed/cancelled prefill publishes nothing.
        assert_eq!(cache.take(1, &[1, 2, 3], None), 0);
        for slot in 0..2 {
            cache.publish(slot, vec![1, 2, 3], None);
        }
        let epoch = cache.epoch;
        cache.clear_all();
        assert_ne!(
            cache.epoch, epoch,
            "a fault also invalidates pending publication"
        );
        for slot in 0..2 {
            assert_eq!(cache.take(slot, &[1, 2, 3], None), 0);
        }
        cache.publish(0, vec![1], None);
        assert_eq!(cache.take(0, &[1, 2], None), 0);
    }
}
