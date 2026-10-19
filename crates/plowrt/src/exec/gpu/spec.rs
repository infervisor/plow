//! Speculative-decode verify on the decode rungs ("span mode", contract plow.spec.v1).
//!
//! A packet emitted with `PLOW_EMIT_SPAN_VERIFY` declares `in.rowslot` and binds it on every
//! FP8-KV store and flash-decode site, so a decode rung's row `r` writes and reads the KV of
//! slot `rowslot[r] & !SPEC_NO_STORE` at its own `pos[r]`. A verify launch packs `k + 1` rows per
//! sequence (the pending token, then `k` drafts) at consecutive positions; flash waits on every
//! row's K/V store, so row `j` sees rows `0..j` of its own span. Plain launches keep the identity
//! map, which addresses exactly the cells the `n_batch_kv` formula does.

use std::path::Path;

use crate::device::cuda::{KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem, Module};
use crate::orch::speculative::{
    accept_greedy, SpecSpanDev, SpecVerdict, SPEC_KMAX, SPEC_NO_STORE,
};
use crate::{Result, RuntimeError};

use super::decode_rung::decode_selection;
use super::GpuEngine;
use packet::dev::{DevOp, TENSOR_NONE16};

/// One sequence's verify span: `tokens[0]` is the slot's pending token (position `pos[slot]`),
/// `tokens[1..]` the drafts at the following positions.
pub struct SpecSpan<'a> {
    pub slot: usize,
    pub tokens: &'a [u32],
}

/// One FP8-KV store site's cache geometry (cache bytes `[slot][head][stride][hd]`, f32 scales
/// `[slot][head][stride]`), for the bench-only poisoning of uncommitted rows.
#[derive(Clone, Copy, PartialEq, Eq)]
struct KvSite {
    cache: usize,
    scale: usize,
    heads: u32,
    hd: u32,
    stride: u32,
    mask: u32,
}

/// Span-mode state: present when the packet declares `in.rowslot`.
pub(super) struct SpecRt {
    t_rowslot: usize,
    /// The decode objects honour the map (`plow_row_slot_1`); without it the packet only runs
    /// plain launches on the identity map.
    object: bool,
    identity: PinnedHost,
    rows: PinnedHost,
    inputs: PinnedHost,
    d_inputs: DeviceMem,
    spans: PinnedHost,
    d_spans: DeviceMem,
    verdicts: PinnedHost,
    d_verdicts: DeviceMem,
    stop: PinnedHost,
    d_stop: DeviceMem,
    accept: Option<(KernelFn, Module)>,
    sites: Vec<KvSite>,
    /// The last verify's spans, each committable once and only against the state it ran on.
    pending: Vec<Pending>,
}

struct Pending {
    slot: usize,
    generation: u32,
    pos: u32,
    tokens: Vec<u32>,
    verdict: SpecVerdict,
}

const STOP_MAX: usize = 64;

impl SpecRt {
    pub(super) fn load(
        be: &crate::device::cuda::CudaBackend,
        t_rowslot: Option<usize>,
        object: bool,
        batch: usize,
        progs: &[&crate::asset::devblob::DevProg],
    ) -> Result<Option<SpecRt>> {
        let Some(t_rowslot) = t_rowslot else {
            return Ok(None);
        };
        let mut identity = be.host_alloc_pinned(batch * 4)?;
        for (b, v) in bytemuck::cast_slice_mut::<u8, u32>(identity.as_mut_slice())
            .iter_mut()
            .enumerate()
        {
            *v = b as u32;
        }
        let mut sites = Vec::new();
        for p in progs {
            for d in &p.insts {
                if d.op == DevOp::HeadNormRopeFp8 as u16 && d.t[7] != TENSOR_NONE16 {
                    let site = KvSite {
                        cache: d.t[0] as usize,
                        scale: d.t[6] as usize,
                        heads: d.i[1],
                        hd: d.i[2],
                        stride: d.fj[1],
                        mask: d.fj[2],
                    };
                    if !sites.contains(&site) {
                        sites.push(site);
                    }
                }
            }
        }
        let verdict_bytes = batch * std::mem::size_of::<SpecVerdict>();
        let span_bytes = batch * std::mem::size_of::<SpecSpanDev>();
        Ok(Some(SpecRt {
            t_rowslot,
            object,
            identity,
            rows: be.host_alloc_pinned(batch * 4)?,
            inputs: be.host_alloc_pinned(batch * 4)?,
            d_inputs: be.alloc(0, (batch * 4) as u64)?,
            spans: be.host_alloc_pinned(span_bytes)?,
            d_spans: be.alloc(0, span_bytes as u64)?,
            verdicts: be.host_alloc_pinned(verdict_bytes)?,
            d_verdicts: be.alloc(0, verdict_bytes as u64)?,
            stop: be.host_alloc_pinned(STOP_MAX * 4)?,
            d_stop: be.alloc(0, (STOP_MAX * 4) as u64)?,
            accept: None,
            sites,
            pending: Vec::new(),
        }))
    }
}

impl GpuEngine {
    /// Upload the identity row map. Plain launches of a span-capable packet need it: the
    /// map is bound on every KV store site.
    pub(super) fn spec_upload_identity(&self) -> Result<()> {
        if let Some(s) = &self.spec {
            // SAFETY: the pinned identity map lives as long as the engine and is never written.
            unsafe {
                self.be.memcpy_htod_async(
                    self.devp[s.t_rowslot].base,
                    s.identity.as_slice(),
                    &self.stream,
                )?;
            }
        }
        Ok(())
    }

    /// Arm span-mode verify: needs the packet's `in.rowslot`, decode objects carrying
    /// `plow_row_slot_1`, no recurrent state, no context-band rungs, and the accept kernel from
    /// the sampler cubin. Returns the largest `k` the sliding rings allow.
    pub fn spec_enable(&mut self, assets: &Path) -> Result<usize> {
        let reason = if self.recurrent.is_some() {
            Some("recurrent state cannot rewind")
        } else if self.decode_contexts.is_some() {
            Some("context-band decode rungs are not span-capable")
        } else {
            match &self.spec {
                None => Some("packet was not emitted with PLOW_EMIT_SPAN_VERIFY"),
                Some(s) if !s.object => Some("decode object lacks plow_row_slot_1"),
                Some(_) => None,
            }
        };
        if let Some(r) = reason {
            return Err(RuntimeError::Rejected(format!("speculative verify: {r}")));
        }
        let cubin = crate::config::RuntimeConfig::get()
            .nv_cubin_sample()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| assets.join("sample_sm120.cubin"));
        let image = std::fs::read(&cubin).map_err(|source| RuntimeError::Io {
            path: cubin.clone(),
            source,
        })?;
        let module = self.be.module_load(&image)?;
        let f = self.be.get_function(&module, "plow_spec_accept").map_err(|_| {
            RuntimeError::Rejected(format!(
                "speculative verify: {} lacks plow_spec_accept",
                cubin.display()
            ))
        })?;
        self.spec.as_mut().expect("checked").accept = Some((f, module));
        let ring_k = self
            .resume_ring_slack
            .map_or(SPEC_KMAX, |s| (s as usize).saturating_sub(1));
        Ok(SPEC_KMAX.min(ring_k))
    }

    /// Slot `b`'s committed position: the pending token's row.
    pub fn slot_pos(&self, b: usize) -> usize {
        self.pos[b] as usize
    }

    /// Bench-only: plain and verify launches use at least this rung (0 = the natural rung).
    pub fn set_debug_min_rung(&mut self, rows: usize) {
        self.debug_min_rows = rows;
    }

    /// Bench-only: size every launch's dense split count for this kv length instead of the
    /// launch's longest row, so a verify and the plain steps it replaces split identically.
    pub fn set_debug_nsplit_kvlen(&mut self, kvlen: Option<u32>) {
        self.debug_nsplit_kvlen = kvlen;
    }

    /// Run the target over every span in one decode-rung launch and accept greedily.
    /// Commits nothing: call [`Self::spec_commit`] per span with its verdict.
    pub fn spec_verify(
        &mut self,
        spans: &[SpecSpan<'_>],
        stop: &[u32],
        verdicts: &mut Vec<SpecVerdict>,
    ) -> Result<()> {
        let reject = |m: String| Err(RuntimeError::Rejected(format!("spec_verify: {m}")));
        let bsz = self.batch;
        match self.spec.as_mut() {
            Some(s) if s.accept.is_some() => s.pending.clear(),
            _ => return reject("not enabled (spec_enable)".into()),
        }
        if spans.is_empty() || spans.len() > bsz || stop.len() > STOP_MAX {
            return reject(format!("{} spans, {} stop ids", spans.len(), stop.len()));
        }
        let max_k = SPEC_KMAX.min(
            self.resume_ring_slack
                .map_or(SPEC_KMAX, |s| (s as usize).saturating_sub(1)),
        );
        let mut seen = vec![false; bsz];
        let mut rows = 0usize;
        for sp in spans {
            let k = sp.tokens.len().wrapping_sub(1);
            if sp.slot >= bsz || std::mem::replace(&mut seen[sp.slot], true) {
                return reject(format!("slot {} out of range or repeated", sp.slot));
            }
            if sp.tokens.is_empty() || k > max_k {
                return reject(format!("slot {}: k = {k} (max {max_k})", sp.slot));
            }
            if self.pos[sp.slot] as usize + k >= self.max_ctx {
                return reject(format!("slot {}: context exhausted", sp.slot));
            }
            rows += k + 1;
        }
        let need = rows.max(self.debug_min_rows);
        let Some(rung) = self.decode_rung(need - 1) else {
            return reject(format!("{need} rows exceed the decode ladder"));
        };
        let selection = decode_selection(Some(rung), None);
        let launch_rows = self.decode_rungs[rung].rows;

        if let Some(v) = &mut self.vmm {
            if let Some(rings) = &mut v.rings {
                let top = spans.iter().map(|sp| sp.slot).max().expect("nonempty") + 1;
                rings.ensure_prefix(top)?;
            }
            for sp in spans {
                v.ensure_rows(sp.slot, self.pos[sp.slot] + sp.tokens.len() as u32)?;
            }
        }
        let max_kvlen = spans
            .iter()
            .map(|sp| self.pos[sp.slot] + sp.tokens.len() as u32)
            .max()
            .expect("nonempty");
        self.patch_nv_nsplit(max_kvlen)?;
        // Rows up to the span ends are about to be written, whatever happens after submission.
        for sp in spans {
            let hw = self.pos[sp.slot] + sp.tokens.len() as u32;
            self.kv_hw[sp.slot] = self.kv_hw[sp.slot].max(hw);
        }

        let pad_slot = spans[0].slot as u32 | SPEC_NO_STORE;
        let s = self.spec.as_mut().expect("checked");
        {
            let (ids, pos, kvlen) = self.stage.parts_mut();
            let map: &mut [u32] = bytemuck::cast_slice_mut(s.rows.as_mut_slice());
            let inputs: &mut [i32] = bytemuck::cast_slice_mut(s.inputs.as_mut_slice());
            let dspans: &mut [SpecSpanDev] = bytemuck::cast_slice_mut(s.spans.as_mut_slice());
            let mut r = 0usize;
            for (i, sp) in spans.iter().enumerate() {
                let p0 = self.pos[sp.slot];
                dspans[i] = SpecSpanDev {
                    row0: r as u32,
                    k: sp.tokens.len() as u32 - 1,
                    req_gen: self.slot_generations[sp.slot],
                    _pad: 0,
                };
                for (j, &t) in sp.tokens.iter().enumerate() {
                    ids[r] = t as i32;
                    inputs[r] = t as i32;
                    pos[r] = (p0 + j as u32) as i32;
                    kvlen[r] = (p0 + j as u32 + 1) as i32;
                    map[r] = sp.slot as u32;
                    r += 1;
                }
            }
            for r in rows..bsz {
                ids[r] = 0;
                inputs[r] = 0;
                pos[r] = 0;
                kvlen[r] = 1;
                map[r] = pad_slot;
            }
            let stop_h: &mut [u32] = bytemuck::cast_slice_mut(s.stop.as_mut_slice());
            stop_h[..stop.len()].copy_from_slice(stop);
        }
        let n_spans = spans.len();
        if let Err(e) = self.spec_submit(selection, launch_rows, rows, n_spans, stop.len()) {
            // Plain launches must not see the span map, and staging may still be in flight.
            let _ = self.spec_upload_identity();
            let _ = self.be.stream_synchronize(&self.stream);
            return Err(e);
        }
        let verdict_bytes = n_spans * std::mem::size_of::<SpecVerdict>();
        let s = self.spec.as_ref().expect("checked");
        let dev: &[SpecVerdict] = bytemuck::cast_slice(&s.verdicts.as_slice()[..verdict_bytes]);
        let argmax: &[i32] = bytemuck::cast_slice(&self.stage.section(0)[..rows * 4]);
        let argmax: Vec<u32> = argmax.iter().map(|&t| t as u32).collect();
        verdicts.clear();
        let mut r = 0usize;
        for (sp, v) in spans.iter().zip(dev) {
            let n = sp.tokens.len();
            let gold = accept_greedy(
                sp.tokens,
                &argmax[r..r + n],
                r as u32,
                self.slot_generations[sp.slot],
                stop,
            );
            if *v != gold {
                return reject(format!(
                    "device verdict {v:?} differs from the host golden {gold:?}"
                ));
            }
            let hw = self.pos[sp.slot] + n as u32;
            self.kv_hw[sp.slot] = self.kv_hw[sp.slot].max(hw);
            verdicts.push(*v);
            r += n;
        }
        let pending: Vec<Pending> = spans
            .iter()
            .zip(verdicts.iter())
            .map(|(sp, v)| Pending {
                slot: sp.slot,
                generation: self.slot_generations[sp.slot],
                pos: self.pos[sp.slot],
                tokens: sp.tokens.to_vec(),
                verdict: *v,
            })
            .collect();
        self.spec.as_mut().expect("checked").pending = pending;
        Ok(())
    }

    /// Uploads, verify launch, identity restore, accept launch and read-backs, then the sync.
    fn spec_submit(
        &mut self,
        selection: super::decode_rung::DecodeSelection,
        launch_rows: usize,
        rows: usize,
        n_spans: usize,
        n_stop: usize,
    ) -> Result<()> {
        // SAFETY: every pinned source lives on `self` past the stream synchronize below; each
        // copy is within its [batch]-sized tensor or buffer.
        unsafe {
            for (k, t) in [(0, self.t_ids), (1, self.t_pos), (2, self.t_kvlen)] {
                self.be
                    .memcpy_htod_async(self.devp[t].base, self.stage.section(k), &self.stream)?;
            }
            let s = self.spec.as_ref().expect("checked");
            self.be
                .memcpy_htod_async(self.devp[s.t_rowslot].base, s.rows.as_slice(), &self.stream)?;
            self.be
                .memcpy_htod_async(s.d_inputs.base, s.inputs.as_slice(), &self.stream)?;
            let span_bytes = n_spans * std::mem::size_of::<SpecSpanDev>();
            self.be.memcpy_htod_async(
                s.d_spans.base,
                &s.spans.as_slice()[..span_bytes],
                &self.stream,
            )?;
            if n_stop > 0 {
                self.be.memcpy_htod_async(
                    s.d_stop.base,
                    &s.stop.as_slice()[..n_stop * 4],
                    &self.stream,
                )?;
            }
        }
        self.reset_selected_decode_counters(selection)?;
        self.launch_selected_decode(selection).map_err(|e| {
            tracing::warn!(launch_rows, rows, error = %e, "spec verify launch failed");
            e
        })?;
        // Plain launches after this one read the identity map again (stream order).
        self.spec_upload_identity()?;
        let s = self.spec.as_ref().expect("checked");
        let (f, _) = s.accept.as_ref().expect("checked");
        let mut a_ids = self.devp[self.t_ids].base;
        let mut a_in = s.d_inputs.base;
        let mut a_spans = s.d_spans.base;
        let mut a_n = n_spans as u32;
        let mut a_stop = s.d_stop.base;
        let mut a_nstop = n_stop as u32;
        let mut a_out = s.d_verdicts.base;
        let mut args = [
            &mut a_ids as *mut u64 as *mut std::ffi::c_void,
            &mut a_in as *mut u64 as *mut std::ffi::c_void,
            &mut a_spans as *mut u64 as *mut std::ffi::c_void,
            &mut a_n as *mut u32 as *mut std::ffi::c_void,
            &mut a_stop as *mut u64 as *mut std::ffi::c_void,
            &mut a_nstop as *mut u32 as *mut std::ffi::c_void,
            &mut a_out as *mut u64 as *mut std::ffi::c_void,
        ];
        self.be.launch_kernel(
            *f,
            (n_spans as u32).div_ceil(128),
            128,
            0,
            &mut args,
            Some(&self.stream),
        )?;
        let verdict_bytes = n_spans * std::mem::size_of::<SpecVerdict>();
        let s = self.spec.as_mut().expect("checked");
        // SAFETY: as above — pinned destinations outlive the synchronize.
        unsafe {
            self.be.memcpy_dtoh_async(
                &mut s.verdicts.as_mut_slice()[..verdict_bytes],
                s.d_verdicts.base,
                &self.stream,
            )?;
            self.be.memcpy_dtoh_async(
                &mut self.stage.section_mut(0)[..rows * 4],
                self.devp[self.t_ids].base,
                &self.stream,
            )?;
        }
        self.be.stream_synchronize(&self.stream)
    }

    /// Commit one verified span: its first `n_emit` input tokens now own committed KV rows; rows
    /// past them are dead (never read, overwritten by the next launch).
    pub fn spec_commit(&mut self, span: &SpecSpan<'_>, v: &SpecVerdict) -> Result<()> {
        let b = span.slot;
        let n = usize::from(v.n_emit);
        let stale = || {
            RuntimeError::Rejected(format!(
                "spec_commit: no matching verify for slot {b} (stale, replayed or altered)"
            ))
        };
        let s = self.spec.as_mut().ok_or_else(stale)?;
        let ix = s.pending.iter().position(|p| p.slot == b).ok_or_else(stale)?;
        let p = s.pending.swap_remove(ix);
        if b >= self.batch
            || p.generation != self.slot_generations[b]
            || p.pos != self.pos[b]
            || p.tokens != span.tokens
            || p.verdict != *v
            || n == 0
            || n > span.tokens.len()
        {
            return Err(stale());
        }
        self.pos[b] += n as u32;
        if self.vmm_prefix_enabled() {
            self.seq_tokens[b].extend_from_slice(&span.tokens[..n]);
        }
        if let Some(v) = &self.vmm {
            v.advise(b, self.pos[b]);
        }
        Ok(())
    }

    /// Bench-only: overwrite slot `b`'s KV bytes (e4m3 NaN) and scales (f32 NaN) at positions
    /// `from..to` on every span-mode store site, so a read of an uncommitted row poisons output.
    pub fn spec_poison_rows(&self, b: usize, from: u32, to: u32) -> Result<()> {
        let Some(s) = &self.spec else {
            return Ok(());
        };
        for site in &s.sites {
            for row in from..to {
                let cell = u64::from(row & site.mask);
                for h in 0..u64::from(site.heads) {
                    let line = (b as u64 * u64::from(site.heads) + h) * u64::from(site.stride) + cell;
                    let hd = u64::from(site.hd);
                    self.be.memset_d8_async(
                        self.devp[site.cache].base + line * hd,
                        0x7F,
                        hd as usize,
                        &self.stream,
                    )?;
                    if site.scale != TENSOR_NONE16 as usize {
                        self.be.memset_d8_async(
                            self.devp[site.scale].base + line * 4,
                            0xFF,
                            4,
                            &self.stream,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}
