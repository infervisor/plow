//! Packet draft programs for speculative decode (`plow_asset::spec_draft`).
//!
//! The packet owns the drafter: the runtime stages each sequence's input token, the verify row
//! holding the hidden that produced it, its committed length and slot, launches the packet's
//! `(rows, k)` program on the decode interpreter and reads back `k` tokens per sequence.

use std::path::Path;

use plow_asset::spec_draft::Manifest;

use crate::asset::devblob::DevBlob;
use crate::device::cuda::PinnedHost;
use crate::device::{Backend, DeviceMem};
use crate::{Result, RuntimeError};

use super::{CudaBackend, DevProgram, GpuEngine, BLOCK};

struct Loaded {
    arg: DevProgram,
    counter_bytes: usize,
    _tables: Vec<DeviceMem>,
    /// Host image of the instructions, and the full-attention flash (`i5`) and merge (`i2`)
    /// split-count sites re-pointed at the live kv_len like the decode rungs' (any split count
    /// up to the baked one is exact; more splits measured slower).
    insts: Vec<packet::dev::DevInst64>,
    sites: Vec<(usize, bool)>,
    baked: u32,
    cur: u32,
}

/// Full-attention (`hd 512`) flash-decode and merge sites, and their common baked split count.
fn split_sites(insts: &[packet::dev::DevInst64]) -> Result<(Vec<(usize, bool)>, u32)> {
    use packet::dev::DevOp;
    let mut sites = Vec::new();
    let mut baked = None;
    for (i, d) in insts.iter().enumerate() {
        let site = if d.op == DevOp::FlashDecodeFp8 as u16 && d.i[6] == 512 {
            Some((false, d.i[5]))
        } else if d.op == DevOp::FlashMerge as u16 && d.i[3] == 512 {
            Some((true, d.i[2]))
        } else {
            None
        };
        if let Some((merge, ns)) = site {
            if *baked.get_or_insert(ns) != ns {
                return Err(reject(
                    "full-attention split counts differ within a draft program",
                ));
            }
            sites.push((i, merge));
        }
    }
    Ok((sites, baked.unwrap_or(1)))
}

pub(super) struct SpecDraft {
    manifest: Manifest,
    programs: Vec<Loaded>,
    /// `[kmax + 1 tokens, hidden_row, position, kv_len, slot][rows]` i32 staging.
    stage: PinnedHost,
    rows: usize,
    selected: Option<(u32, usize, usize)>,
    /// Per slot, the hidden row its pending token came from, saved at the verify that produced
    /// it, and the `(generation, position)` it is valid for.
    hidden_bytes: u64,
    hidden_snap: DeviceMem,
    hidden_at: Vec<Option<(u32, u32)>>,
    /// Slots whose snapshot a launch is rewriting, keyed when it commits.
    staged_saves: Vec<usize>,
}

fn reject(e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Rejected(format!("spec draft: {e}"))
}

impl SpecDraft {
    fn load(e: &GpuEngine, assets: &Path) -> Result<Self> {
        let pkt = DevBlob::find_in_dir(assets)?
            .ok_or_else(|| reject(format!("no packet in {}", assets.display())))?;
        let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io {
            path: pkt.clone(),
            source,
        })?;
        let blob = DevBlob::parse(&raw)?;
        let meta = blob
            .reserved_metadata(&raw, plow_asset::spec_draft::SECTION)?
            .ok_or_else(|| reject("packet was not emitted with PLOW_EMIT_SPEC"))?;
        let manifest: Manifest = serde_json::from_slice(meta).map_err(reject)?;
        let bytes = blob
            .reserved_metadata(&raw, plow_asset::spec_draft::PROGRAMS_SECTION)?
            .ok_or_else(|| reject("draft programs section missing"))?;
        // Under PLOW_LIVE_CTX the engine re-declared the packet at its live bound; the draft
        // programs address the same full-attention caches and take the same restride.
        let ceiling = crate::exec::ctx_bound::declared_ctx(&blob).unwrap_or(0);
        let live = e.max_ctx as u32;
        let mut sized = None;
        if live != ceiling {
            if live > ceiling {
                return Err(reject(format!("widened live context {live} > {ceiling}")));
            }
            let mut narrowed = DevBlob::parse(&raw)?;
            crate::exec::ctx_bound::narrow(&mut narrowed, live)?;
            sized = Some(narrowed);
        }
        let sized_tensors = sized.as_ref().map_or(&blob.tensors, |b| &b.tensors);
        let rewritten: Vec<u16> = (0..blob.tensors.len())
            .filter(|&h| {
                blob.tensors[h].name.starts_with("kv.")
                    && sized_tensors[h].bytes != blob.tensors[h].bytes
            })
            .map(|h| h as u16)
            .collect();
        // The programs address this engine's tensor table: same packet tensors, in order.
        if blob.n_cu != e.grid
            || blob.tensors.len() != e.devp.len()
            || sized_tensors
                .iter()
                .zip(e.tensor_names.iter().zip(&e.devp))
                .any(|(t, (name, mem))| t.name != *name || mem.len < t.bytes)
        {
            return Err(reject("packet does not match the loaded engine"));
        }
        // FP8 attention honours `in.spec.slot` only in objects that export the row-slot marker.
        if !e.spec.as_ref().is_some_and(|s| s.row_slot_object()) {
            return Err(reject("decode object lacks plow_row_slot_1"));
        }
        let mut section =
            plow_asset::aux_program::parse(bytes, e.grid, e.devp.len()).map_err(reject)?;
        if live != ceiling {
            for p in &mut section.programs {
                packet::ctx_bound::restride_matching(&mut p.insts, ceiling, live, |h| {
                    rewritten.contains(&h)
                });
                if let Some((k, op, slot)) =
                    packet::ctx_bound::residual_ceiling(&p.insts, ceiling, |h| {
                        rewritten.contains(&h)
                    })
                {
                    return Err(reject(format!(
                        "instruction {k} (op {op}) keeps the emitted ctx in {slot:?}"
                    )));
                }
            }
        }
        manifest
            .validate(section.programs.len(), e.devp.len())
            .map_err(reject)?;
        let t = &manifest.tensors;
        let mut inputs: Vec<u16> = t.tokens.clone();
        inputs.extend([t.hidden_row, t.position, t.kv_len, t.slot]);
        let mut distinct = inputs.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != inputs.len() || inputs.contains(&t.hidden) {
            return Err(reject("aliased draft input tensors"));
        }
        for p in &manifest.programs {
            let prog = &section.programs[p.index as usize];
            // Step 0 gathers `rows` hidden rows (of `i1` elements) out of `hidden_rows`.
            let gather_ok = prog.insts.iter().any(|d| {
                d.op == packet::dev::DevOp::RowGather as u16
                    && d.t[1] == t.hidden
                    && d.t[2] == t.hidden_row
                    && d.i[0] == p.rows
                    && d.i[2] == t.hidden_rows
                    && e.devp[usize::from(t.hidden)].len
                        >= u64::from(t.hidden_rows) * u64::from(d.i[1]) * 2
            });
            if prog.rows != p.rows || !gather_ok {
                return Err(reject(format!("program {} geometry", p.index)));
            }
        }
        let mut hidden_bytes = None;
        for p in &section.programs {
            for d in &p.insts {
                if d.op == packet::dev::DevOp::RowGather as u16 && d.t[1] == t.hidden {
                    let b = u64::from(d.i[1]) * 2;
                    if *hidden_bytes.get_or_insert(b) != b {
                        return Err(reject("draft programs gather different hidden widths"));
                    }
                }
            }
        }
        let hidden_bytes = hidden_bytes.ok_or_else(|| reject("no hidden gather"))?;
        let rows = manifest.programs.iter().map(|p| p.rows).max().unwrap_or(0) as usize;
        for &h in &inputs {
            if e.devp[usize::from(h)].len < rows as u64 * 4 {
                return Err(reject(format!("tensor {h} holds fewer than {rows} rows")));
            }
        }
        let programs = section
            .programs
            .iter()
            .map(|p| {
                let (arg, _, counter_bytes, tables) =
                    super::mixed_step::upload_program(&e.be, p, e.d_tens.base, 0, 0)?;
                let (sites, baked) = split_sites(&p.insts)?;
                Ok(Loaded {
                    arg,
                    counter_bytes,
                    _tables: tables,
                    insts: p.insts.clone(),
                    sites,
                    baked,
                    cur: baked,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stage =
            e.be.host_alloc_pinned((manifest.kmax as usize + 5) * rows * 4)?;
        tracing::info!(
            programs = programs.len(),
            baked_splits = ?programs.iter().map(|p| p.baked).collect::<std::collections::BTreeSet<_>>(),
            kmax = manifest.kmax,
            rows,
            family = %manifest.family,
            "spec draft programs loaded"
        );
        Ok(Self {
            manifest,
            programs,
            stage,
            rows,
            selected: None,
            hidden_bytes,
            hidden_snap: e.be.alloc(0, hidden_bytes * e.batch as u64)?,
            hidden_at: vec![None; e.batch],
            staged_saves: Vec::with_capacity(e.batch),
        })
    }

    fn launch(
        &self,
        be: &CudaBackend,
        f: super::KernelFn,
        grid: u32,
        smem: u32,
        stream: &crate::device::cuda::CudaStream,
        index: usize,
    ) -> Result<()> {
        let p = &self.programs[index];
        be.memset_d8_async(p.arg.counters, 0, p.counter_bytes, stream)?;
        let mut arg = p.arg;
        let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        be.launch_cooperative(f, grid, BLOCK, smem, &mut params, Some(stream))
    }
}

impl GpuEngine {
    /// Load the packet's draft programs. Returns the deepest draft `k` they offer.
    pub fn spec_draft_enable(&mut self, assets: &Path) -> Result<u32> {
        let d = SpecDraft::load(self, assets)?;
        let kmax = d.manifest.kmax;
        self.spec_draft = Some(d);
        Ok(kmax)
    }

    /// Stage `(slot, token, hidden_row)` per sequence and select the `(rows, k)` program.
    /// `hidden_row` is the row of the packet's hidden activation that produced `token` in the
    /// last decode or verify launch; the draft reads the slot's committed KV only.
    pub fn spec_draft_stage(&mut self, rows: &[(usize, u32, u32)], k: u32) -> Result<()> {
        if rows.is_empty() {
            return Err(reject("no rows"));
        }
        // The previous launch may still read the stage and the instruction image.
        self.be.stream_synchronize(&self.stream)?;
        let vocab = self.vocab() as u32;
        let batch = self.batch;
        let pos: Vec<u32> = rows
            .iter()
            .map(|&(s, _, _)| self.pos.get(s).copied().unwrap_or(0) as u32)
            .collect();
        let d = self
            .spec_draft
            .as_mut()
            .ok_or_else(|| reject("not enabled"))?;
        let p = *d
            .manifest
            .select(rows.len(), k)
            .ok_or_else(|| reject(format!("no program for {} rows at k {k}", rows.len())))?;
        for (&(slot, token, hrow), &pos) in rows.iter().zip(&pos) {
            if slot >= batch || token >= vocab || hrow >= d.manifest.tensors.hidden_rows || pos == 0
            {
                return Err(reject(format!(
                    "row slot {slot} token {token} hidden row {hrow} pos {pos}"
                )));
            }
        }
        let n = d.rows;
        let kmax = d.manifest.kmax as usize;
        let words: &mut [i32] = bytemuck::cast_slice_mut(d.stage.as_mut_slice());
        for r in 0..p.rows as usize {
            let (slot, token, hrow) = rows[r.min(rows.len() - 1)];
            let pos = pos[r.min(rows.len() - 1)] as i32;
            words[r] = token as i32;
            words[(kmax + 1) * n + r] = hrow as i32;
            words[(kmax + 2) * n + r] = pos - 1;
            words[(kmax + 3) * n + r] = pos;
            words[(kmax + 4) * n + r] = slot as i32;
        }
        let t = d.manifest.tensors.clone();
        let bytes = d.stage.as_slice();
        for (i, h) in [
            (0, t.tokens[0]),
            (kmax + 1, t.hidden_row),
            (kmax + 2, t.position),
            (kmax + 3, t.kv_len),
            (kmax + 4, t.slot),
        ] {
            // SAFETY: the pinned stage is rewritten only after the next stage's synchronize.
            unsafe {
                self.be.memcpy_htod_async(
                    self.devp[usize::from(h)].base,
                    &bytes[i * n * 4..(i * n + p.rows as usize) * 4],
                    &self.stream,
                )?;
            }
        }
        let kv_max = pos.iter().copied().max().unwrap_or(1);
        let prog = &mut d.programs[p.index as usize];
        let want = crate::exec::kvrow::nv_dense_live_nsplit(prog.baked, kv_max);
        if want != prog.cur && !prog.sites.is_empty() {
            for &(i, merge) in &prog.sites {
                if merge {
                    prog.insts[i].i[2] = want;
                } else {
                    prog.insts[i].i[5] = want;
                }
            }
            // SAFETY: the image lives in `prog`; the next stage synchronizes before touching it.
            unsafe {
                self.be.memcpy_htod_async(
                    prog.arg.insts,
                    super::pod_bytes(&prog.insts),
                    &self.stream,
                )?;
            }
            prog.cur = want;
        }
        d.selected = Some((k, p.index as usize, rows.len()));
        Ok(())
    }

    /// Launch the staged draft and read `k` tokens per staged row into `out` (row-major).
    pub fn spec_draft_run(&mut self, out: &mut Vec<u32>) -> Result<()> {
        self.spec_draft_launch()?;
        let d = self.spec_draft.as_mut().expect("staged");
        let (k, _, live) = d.selected.expect("staged");
        let n = d.rows;
        for j in 1..=k as usize {
            let h = d.manifest.tensors.tokens[j];
            // SAFETY: as the uploads.
            unsafe {
                self.be.memcpy_dtoh_async(
                    &mut d.stage.as_mut_slice()[j * n * 4..(j * n + live) * 4],
                    self.devp[usize::from(h)].base,
                    &self.stream,
                )?;
            }
        }
        self.be.stream_synchronize(&self.stream)?;
        let words: &[u32] = bytemuck::cast_slice(d.stage.as_slice());
        out.clear();
        for r in 0..live {
            out.extend((1..=k as usize).map(|j| words[j * n + r]));
        }
        Ok(())
    }

    /// Launch the staged draft program without reading it back (bench timing).
    pub fn spec_draft_launch(&mut self) -> Result<()> {
        let d = self
            .spec_draft
            .as_ref()
            .ok_or_else(|| reject("not enabled"))?;
        let (_, index, _) = d.selected.ok_or_else(|| reject("nothing staged"))?;
        let (f, smem) = self
            .row_slot_twin
            .as_ref()
            .map_or((self.f, self.smem), |(full, _)| (full.function, full.smem));
        d.launch(&self.be, f, self.grid, smem, &self.stream, index)
    }

    /// Whether slot `b` holds the hidden its pending token came from (saved by the verify that
    /// produced it; any later step or slot reuse moves the position or generation past it).
    pub fn spec_draft_ready(&self, b: usize) -> bool {
        self.spec_draft.as_ref().is_some_and(|d| {
            d.hidden_at.get(b).copied().flatten() == Some((self.slot_generations[b], self.pos[b]))
        })
    }

    /// Copy verify row `row`'s hidden aside for slot `b`, valid once `b` sits at `pos`. On the
    /// stream right after the verify, before any launch can overwrite the row.
    pub(super) fn spec_hidden_save(&mut self, b: usize, row: u32, pos: u32) -> Result<()> {
        let Some(d) = self.spec_draft.as_mut() else {
            return Ok(());
        };
        let h = d.manifest.tensors.hidden;
        if row >= d.manifest.tensors.hidden_rows {
            return Err(reject(format!("hidden row {row}")));
        }
        let n = d.hidden_bytes;
        self.be.memcpy_dtod_async(
            d.hidden_snap.base + b as u64 * n,
            self.devp[usize::from(h)].base + u64::from(row) * n,
            n,
            &self.stream,
        )?;
        d.hidden_at[b] = Some((self.slot_generations[b], pos));
        Ok(())
    }

    /// After a plain decode launch over `feeds` (hidden row = slot): snapshot every fed slot's
    /// row in one copy, valid once the slot sits at `pos[b] + advance`. Unfed slots inside the
    /// copied range lose their snapshot.
    pub(super) fn spec_hidden_save_plain(
        &mut self,
        feeds: &[(usize, u32)],
        advance: u32,
    ) -> Result<()> {
        let Some(d) = self.spec_draft.as_mut() else {
            return Ok(());
        };
        let (Some(lo), Some(hi)) = (
            feeds.iter().map(|f| f.0).min(),
            feeds.iter().map(|f| f.0).max(),
        ) else {
            return Ok(());
        };
        if hi as u32 >= d.manifest.tensors.hidden_rows || hi >= d.hidden_at.len() {
            return Ok(());
        }
        d.hidden_at[lo..=hi].fill(None);
        let n = d.hidden_bytes;
        self.be.memcpy_dtod_async(
            d.hidden_snap.base + lo as u64 * n,
            self.devp[usize::from(d.manifest.tensors.hidden)].base + lo as u64 * n,
            (hi - lo + 1) as u64 * n,
            &self.stream,
        )?;
        for &(b, _) in feeds {
            d.hidden_at[b] = Some((self.slot_generations[b], self.pos[b] + advance));
        }
        Ok(())
    }

    /// Snapshot `(slot, row)` pairs of a `row_bytes`-pitched final-norm activation at `src`
    /// (the packed terminal's), keyed by [`Self::spec_hidden_commit`] once the launch commits.
    pub(super) fn spec_hidden_save_rows(
        &mut self,
        src: u64,
        row_bytes: u64,
        rows: impl Iterator<Item = (usize, u32)>,
    ) -> Result<()> {
        let Some(d) = self
            .spec_draft
            .as_mut()
            .filter(|d| d.hidden_bytes == row_bytes)
        else {
            return Ok(());
        };
        let n = d.hidden_bytes;
        for (b, row) in rows {
            if b >= d.hidden_at.len() {
                continue;
            }
            d.hidden_at[b] = None;
            self.be.memcpy_dtod_async(
                d.hidden_snap.base + b as u64 * n,
                src + u64::from(row) * n,
                n,
                &self.stream,
            )?;
            d.staged_saves.push(b);
        }
        Ok(())
    }

    /// The launch whose rows [`Self::spec_hidden_save_rows`] copied committed: each copied
    /// slot's snapshot is its hidden at its current position. `false` drops them.
    pub(super) fn spec_hidden_commit(&mut self, committed: bool) {
        let Some(d) = self.spec_draft.as_mut() else {
            return;
        };
        for b in d.staged_saves.drain(..) {
            if committed {
                d.hidden_at[b] = Some((self.slot_generations[b], self.pos[b]));
            }
        }
    }

    /// A rewind can revisit a keyed position with another history.
    pub(super) fn spec_hidden_forget(&mut self, b: usize) {
        if let Some(slot) = self.spec_draft.as_mut().and_then(|d| d.hidden_at.get_mut(b)) {
            *slot = None;
        }
    }

    /// Put slot `b`'s saved hidden at row `row` of the hidden tensor for the next draft.
    pub(super) fn spec_hidden_restore(&mut self, b: usize, row: u32) -> Result<()> {
        if !self.spec_draft_ready(b) {
            return Err(reject(format!(
                "slot {b}: no saved hidden for its pending token"
            )));
        }
        let d = self.spec_draft.as_ref().expect("ready");
        let h = d.manifest.tensors.hidden;
        let n = d.hidden_bytes;
        self.be.memcpy_dtod_async(
            self.devp[usize::from(h)].base + u64::from(row) * n,
            d.hidden_snap.base + b as u64 * n,
            n,
            &self.stream,
        )
    }

    /// `(deepest k, draft program row counts)` of the loaded drafter.
    pub(super) fn spec_draft_shape(&self) -> Option<(u32, Vec<usize>)> {
        let d = self.spec_draft.as_ref()?;
        let mut rows: Vec<usize> = d
            .manifest
            .programs
            .iter()
            .map(|p| p.rows as usize)
            .collect();
        rows.sort_unstable();
        rows.dedup();
        Some((d.manifest.kmax, rows))
    }

    pub fn spec_draft_sync(&self) -> Result<()> {
        self.be.stream_synchronize(&self.stream)
    }
}
