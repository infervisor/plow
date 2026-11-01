//! Packet draft programs for speculative decode (`plow_asset::spec_draft`).
//!
//! The packet owns the drafter: the runtime stages each sequence's input token and its
//! committed length and slot (a chain drafter also the verify row holding the hidden that
//! produced the token; a block drafter the block's ids, positions and store slots), launches
//! the packet's `(rows, k)` program on the decode interpreter and reads back `k` tokens per
//! sequence.

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
    /// Device base of every tensor the programs index: the target's (embedded drafter) or the
    /// drafter's own table (external drafter, whose imports alias target tensors).
    addr: Vec<u64>,
    external: Option<External>,
    /// Chain: `[kmax + 1 tokens, hidden_row, position, kv_len, slot][rows]` i32 staging.
    /// Block: `[kmax + 1 tokens][rows]` then `[ids, position, kv_len, slot, store][size * rows]`.
    stage: PinnedHost,
    rows: usize,
    selected: Option<(u32, usize, usize)>,
    /// Chain drafters: per slot, the hidden row its pending token came from, saved at the
    /// verify that produced it, and the `(generation, position)` it is valid for.
    hidden_bytes: u64,
    hidden_snap: Option<DeviceMem>,
    hidden_at: Vec<Option<(u32, u32)>>,
    /// Slots whose snapshot a launch is rewriting, keyed when it commits.
    staged_saves: Vec<usize>,
}

/// An external drafter's own device state (`plow_asset::spec_drafter`).
struct External {
    _own: Vec<DeviceMem>,
    _table: DeviceMem,
    digest: String,
}

fn reject(e: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Rejected(format!("spec draft: {e}"))
}

impl SpecDraft {
    /// The packet's own draft programs (`spec.json`), addressing the engine's tensor table.
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
        let section =
            plow_asset::aux_program::parse(bytes, e.grid, e.devp.len()).map_err(reject)?;
        let addr: Vec<(u64, u64)> = e.devp.iter().map(|m| (m.base, m.len)).collect();
        Self::finish(
            e,
            manifest,
            section,
            addr,
            (ceiling, live, &rewritten),
            e.d_tens.base,
            None,
        )
    }

    /// An external drafter directory (`plow_asset::spec_drafter`): checked against what the
    /// target proves, its own tensors allocated and bound, its imports aliased to the target's.
    fn load_external(e: &GpuEngine, assets: &Path, dir: &Path) -> Result<Self> {
        use plow_asset::spec_drafter::{self as sd, Access, Source};
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read(&path).map_err(|source| RuntimeError::Io { path, source })
        };
        let at = |e: String| reject(format!("drafter {}: {e}", dir.display()));
        let m = sd::Manifest::parse(&read(sd::FILE)?).map_err(at)?;
        if m.n_cu != e.grid {
            return Err(at(format!(
                "programs scheduled for {} CUs, the engine runs {}",
                m.n_cu, e.grid
            )));
        }
        let programs_raw = read(sd::PROGRAMS_FILE)?;
        let (view, ceiling, declared, transient) =
            e.spec_target_view(assets, &m.requires.object_symbols)?;
        m.check(&view)
            .map_err(|x| at(format!("does not fit target {}: {x}", assets.display())))?;
        plow_asset::serve_manifest::verify_pins(dir, &m.weights).map_err(at)?;
        let ckpt = m
            .tensors
            .iter()
            .any(|t| t.source == Source::Weight)
            .then(|| crate::asset::checkpoint::Checkpoint::open(dir))
            .transpose()?;
        let init = m
            .tensors
            .iter()
            .any(|t| matches!(t.source, Source::Init { .. }))
            .then(|| read(sd::INIT_FILE))
            .transpose()?;
        let live = e.max_ctx as u32;
        let (mut own, mut addr, mut rewritten) = (Vec::new(), Vec::new(), Vec::new());
        for (h, t) in m.tensors.iter().enumerate() {
            let mem = match &t.source {
                Source::Import { access } => {
                    let i = e
                        .tensor_names
                        .iter()
                        .position(|n| *n == t.name)
                        .ok_or_else(|| at(format!("import {} not in the target", t.name)))?;
                    if *access == Access::ReadWrite && !transient[i] {
                        return Err(at(format!(
                            "import {} writes a target cache, weight or table",
                            t.name
                        )));
                    }
                    let target = &e.devp[i];
                    if target.len != declared[i] {
                        // The engine narrowed this cache to its live context.
                        if !t.name.starts_with("kv.") {
                            return Err(at(format!("import {} was resized", t.name)));
                        }
                        rewritten.push(h as u16);
                    }
                    addr.push((target.base, target.len));
                    continue;
                }
                Source::Weight => {
                    let src = ckpt
                        .as_ref()
                        .and_then(|c| c.tensor(&t.name))
                        .ok_or_else(|| at(format!("MISSING WEIGHT {}", t.name)))?;
                    if src.len() as u64 != t.bytes {
                        return Err(at(format!("weight {} is {} bytes", t.name, src.len())));
                    }
                    let mem = e.be.alloc(0, t.bytes)?;
                    e.be.upload(&mem, 0, src)?;
                    mem
                }
                Source::Init { offset, len } => {
                    let src = init
                        .as_deref()
                        .and_then(|b| b.get(*offset as usize..(*offset + *len) as usize))
                        .ok_or_else(|| at(format!("init of {} out of range", t.name)))?;
                    let mem = e.be.alloc(0, t.bytes)?;
                    e.be.upload(&mem, 0, src)?;
                    mem
                }
                Source::Scratch => {
                    let mem = e.be.alloc(0, t.bytes)?;
                    e.be.memset_d8(mem.base, 0, t.bytes as usize)?;
                    mem
                }
            };
            addr.push((mem.base, mem.len));
            own.push(mem);
        }
        let ptrs: Vec<u64> = addr.iter().map(|a| a.0).collect();
        let table = e.be.alloc(0, (ptrs.len() * 8) as u64)?;
        e.be.upload(&table, 0, super::pod_bytes(&ptrs))?;
        let section =
            plow_asset::aux_program::parse(&programs_raw, e.grid, addr.len()).map_err(at)?;
        let digest = m.digest(&format!(
            "{:x}",
            <sha2::Sha256 as sha2::Digest>::digest(&programs_raw)
        ));
        tracing::info!(
            drafter = %dir.display(),
            kind = ?m.kind,
            family = %m.family,
            digest = %&digest[..16],
            imports = m.tensors.iter().filter(|t| matches!(t.source, Source::Import { .. })).count(),
            "external drafter attached"
        );
        let base = table.base;
        Self::finish(
            e,
            m.draft,
            section,
            addr,
            (ceiling, live, &rewritten),
            base,
            Some(External {
                _own: own,
                _table: table,
                digest,
            }),
        )
    }

    /// Checks and uploads common to both drafter sources. `addr[h]` = `(base, bytes)` of the
    /// tensor the programs' handle `h` names; `ctx` = `(emitted ctx, live ctx, handles of caches
    /// the engine narrowed)`.
    fn finish(
        e: &GpuEngine,
        manifest: Manifest,
        mut section: plow_asset::aux_program::Section,
        addr: Vec<(u64, u64)>,
        ctx: (u32, u32, &[u16]),
        table: u64,
        external: Option<External>,
    ) -> Result<Self> {
        let (ceiling, live, rewritten) = ctx;
        // FP8 attention honours `in.spec.slot` only in objects that export the row-slot marker.
        if !e.spec.as_ref().is_some_and(|s| s.row_slot_object()) {
            return Err(reject("decode object lacks plow_row_slot_1"));
        }
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
            .validate(section.programs.len(), addr.len())
            .map_err(reject)?;
        let len = |h: u16| addr[usize::from(h)].1;
        let t = &manifest.tensors;
        let rows = manifest.programs.iter().map(|p| p.rows).max().unwrap_or(0) as usize;
        // (input tensor, i32 entries it must hold)
        let mut inputs: Vec<(u16, usize)> = t.tokens.iter().map(|&h| (h, rows)).collect();
        let mut hidden_bytes = 0;
        match (&manifest.block, t.hidden_row, t.hidden) {
            (Some(b), None, None) => {
                let n = b.size as usize * rows;
                inputs.extend([t.position, t.kv_len, t.slot, b.ids, b.store_slot].map(|h| (h, n)));
                if manifest.kmax > b.size {
                    return Err(reject("kmax exceeds the block"));
                }
            }
            (None, Some(hidden_row), Some(hidden)) => {
                inputs.extend([hidden_row, t.position, t.kv_len, t.slot].map(|h| (h, rows)));
                if inputs.iter().any(|&(h, _)| h == hidden) {
                    return Err(reject("aliased draft input tensors"));
                }
                for p in &manifest.programs {
                    let prog = &section.programs[p.index as usize];
                    // Step 0 gathers `rows` hidden rows (of `i1` elements) out of `hidden_rows`.
                    let gather_ok = prog.insts.iter().any(|d| {
                        d.op == packet::dev::DevOp::RowGather as u16
                            && d.t[1] == hidden
                            && d.t[2] == hidden_row
                            && d.i[0] == p.rows
                            && d.i[2] == t.hidden_rows
                            && len(hidden) >= u64::from(t.hidden_rows) * u64::from(d.i[1]) * 2
                    });
                    if !gather_ok {
                        return Err(reject(format!("program {} geometry", p.index)));
                    }
                }
                let mut width = None;
                for p in &section.programs {
                    for d in &p.insts {
                        if d.op == packet::dev::DevOp::RowGather as u16 && d.t[1] == hidden {
                            let b = u64::from(d.i[1]) * 2;
                            if *width.get_or_insert(b) != b {
                                return Err(reject(
                                    "draft programs gather different hidden widths",
                                ));
                            }
                        }
                    }
                }
                hidden_bytes = width.ok_or_else(|| reject("no hidden gather"))?;
            }
            _ => return Err(reject("drafter is neither a chain nor a block drafter")),
        }
        let mut distinct: Vec<u16> = inputs.iter().map(|&(h, _)| h).collect();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != inputs.len() {
            return Err(reject("aliased draft input tensors"));
        }
        for p in &manifest.programs {
            if section.programs[p.index as usize].rows != p.rows {
                return Err(reject(format!("program {} geometry", p.index)));
            }
        }
        for &(h, n) in &inputs {
            if len(h) < n as u64 * 4 {
                return Err(reject(format!("tensor {h} holds fewer than {n} entries")));
            }
        }
        let programs = section
            .programs
            .iter()
            .map(|p| {
                let (arg, _, counter_bytes, tables) =
                    super::mixed_step::upload_program(&e.be, p, table, 0, 0)?;
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
        let words = match &manifest.block {
            Some(b) => manifest.kmax as usize + 1 + 5 * b.size as usize,
            None => manifest.kmax as usize + 5,
        };
        let stage = e.be.host_alloc_pinned(words * rows * 4)?;
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
            addr: addr.iter().map(|a| a.0).collect(),
            external,
            stage,
            rows,
            selected: None,
            hidden_bytes,
            hidden_snap: (hidden_bytes != 0)
                .then(|| e.be.alloc(0, hidden_bytes * e.batch as u64))
                .transpose()?,
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
        let d = match crate::asset::serve::drafter_dir(assets) {
            Some(dir) => SpecDraft::load_external(self, assets, &dir)?,
            None => SpecDraft::load(self, assets)?,
        };
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
        if d.manifest.block.is_some() {
            return Err(reject(
                "a block drafter stages through spec_draft_stage_block",
            ));
        }
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
            (kmax + 1, t.hidden_row.expect("chain drafter")),
            (kmax + 2, t.position),
            (kmax + 3, t.kv_len),
            (kmax + 4, t.slot),
        ] {
            // SAFETY: the pinned stage is rewritten only after the next stage's synchronize.
            unsafe {
                self.be.memcpy_htod_async(
                    d.addr[usize::from(h)],
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

    /// Block drafters: stage `(slot, pending token)` per sequence and select the `(rows, k)`
    /// program. Sequence `r`'s block rows `r * size + i` sit at positions `c + i` (`c` = its
    /// committed rows) with ids `[token, mask x (size - 1)]` and attend `c + size` rows; padding
    /// rows of a wider program attend the last sequence's KV and store nothing. The drafter
    /// layers' pages are mapped through `c + size` first.
    pub fn spec_draft_stage_block(&mut self, rows: &[(usize, u32)], k: u32) -> Result<()> {
        if rows.is_empty() {
            return Err(reject("no rows"));
        }
        self.be.stream_synchronize(&self.stream)?;
        let vocab = self.vocab() as u32;
        let (batch, max_ctx) = (self.batch, self.max_ctx as u32);
        let pos: Vec<u32> = rows
            .iter()
            .map(|&(s, _)| self.pos.get(s).copied().unwrap_or(0))
            .collect();
        let d = self
            .spec_draft
            .as_mut()
            .ok_or_else(|| reject("not enabled"))?;
        let block = d
            .manifest
            .block
            .clone()
            .ok_or_else(|| reject("not a block drafter"))?;
        let size = block.size;
        let p = *d
            .manifest
            .select(rows.len(), k)
            .ok_or_else(|| reject(format!("no program for {} rows at k {k}", rows.len())))?;
        for (i, (&(slot, token), &c)) in rows.iter().zip(&pos).enumerate() {
            if slot >= batch
                || token >= vocab
                || c == 0
                || c + size > max_ctx
                || rows[..i].iter().any(|&(s, _)| s == slot)
            {
                return Err(reject(format!(
                    "block row slot {slot} token {token} pos {c} (max_ctx {max_ctx})"
                )));
            }
        }
        let n = d.rows;
        let nb = size as usize * n;
        let kmax = d.manifest.kmax as usize;
        let words: &mut [i32] = bytemuck::cast_slice_mut(d.stage.as_mut_slice());
        let base = (kmax + 1) * n;
        for r in 0..p.rows as usize {
            let live = r < rows.len();
            let (slot, token) = rows[r.min(rows.len() - 1)];
            let c = pos[r.min(rows.len() - 1)] as i32;
            words[r] = token as i32;
            for i in 0..size as usize {
                let row = r * size as usize + i;
                words[base + row] = if i == 0 {
                    token as i32
                } else {
                    block.mask_token as i32
                };
                words[base + nb + row] = c + i as i32;
                words[base + 2 * nb + row] = c + size as i32;
                words[base + 3 * nb + row] = slot as i32;
                words[base + 4 * nb + row] = if live { slot as i32 } else { -1 };
            }
        }
        let t = d.manifest.tensors.clone();
        let used = p.rows as usize;
        let bytes = d.stage.as_slice();
        // SAFETY: the pinned stage is rewritten only after the next stage's synchronize.
        unsafe {
            self.be.memcpy_htod_async(
                d.addr[usize::from(t.tokens[0])],
                &bytes[..used * 4],
                &self.stream,
            )?;
            for (j, h) in [block.ids, t.position, t.kv_len, t.slot, block.store_slot]
                .into_iter()
                .enumerate()
            {
                let at = (base + j * nb) * 4;
                self.be.memcpy_htod_async(
                    d.addr[usize::from(h)],
                    &bytes[at..at + used * size as usize * 4],
                    &self.stream,
                )?;
            }
        }
        let kv_max = pos.iter().copied().max().unwrap_or(1) + size;
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
        if let Some(v) = &self.vmm {
            for (&(slot, _), &c) in rows.iter().zip(&pos) {
                v.ensure_rows(slot, c + size)?;
            }
        }
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
                    d.addr[usize::from(h)],
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

    /// Draft `k` tokens now for each `(slot, pending token)` row whose hidden is ready (and whose
    /// drafts stay inside the context): `out` gets `k` tokens per drafted row, in row order, and
    /// `drafted[i]` says whether row `i` was.
    pub fn spec_draft_rows(
        &mut self,
        rows: &[(usize, u32)],
        k: u32,
        drafted: &mut Vec<bool>,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        drafted.clear();
        out.clear();
        if let Some(size) = self
            .spec_draft
            .as_ref()
            .and_then(|d| d.manifest.block.as_ref())
            .map(|b| b.size)
        {
            let mut stage = Vec::with_capacity(rows.len());
            for &(slot, token) in rows {
                let c = self.pos[slot];
                let ok = c > 0 && c + size.max(k + 1) <= self.max_ctx as u32;
                if ok {
                    stage.push((slot, token));
                }
                drafted.push(ok);
            }
            if stage.is_empty() || k == 0 {
                drafted.fill(false);
                return Ok(());
            }
            self.spec_draft_stage_block(&stage, k)?;
            return self.spec_draft_run(out);
        }
        let mut stage = Vec::with_capacity(rows.len());
        for &(slot, token) in rows {
            let ok = self.spec_draft_ready(slot) && self.pos[slot] + k < self.max_ctx as u32;
            if ok {
                stage.push((slot, token, stage.len() as u32));
            }
            drafted.push(ok);
        }
        if stage.is_empty() || k == 0 {
            drafted.fill(false);
            return Ok(());
        }
        for &(slot, _, row) in &stage {
            self.spec_hidden_restore(slot, row)?;
        }
        self.spec_draft_stage(&stage, k)?;
        self.spec_draft_run(out)
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
            if d.manifest.block.is_some() {
                // Every target launch writes the drafter context of the rows it commits.
                return self.pos.get(b).is_some_and(|&p| p > 0);
            }
            d.hidden_at.get(b).copied().flatten() == Some((self.slot_generations[b], self.pos[b]))
        })
    }

    /// Copy verify row `row`'s hidden aside for slot `b`, valid once `b` sits at `pos`. On the
    /// stream right after the verify, before any launch can overwrite the row.
    pub(super) fn spec_hidden_save(&mut self, b: usize, row: u32, pos: u32) -> Result<()> {
        let Some((d, h, snap)) = self.spec_draft.as_mut().and_then(|d| {
            let h = d.manifest.tensors.hidden?;
            let snap = d.hidden_snap.as_ref()?.base;
            Some((d, h, snap))
        }) else {
            return Ok(());
        };
        if row >= d.manifest.tensors.hidden_rows {
            return Err(reject(format!("hidden row {row}")));
        }
        let n = d.hidden_bytes;
        self.be.memcpy_dtod_async(
            snap + b as u64 * n,
            d.addr[usize::from(h)] + u64::from(row) * n,
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
        let Some((d, h, snap)) = self.spec_draft.as_mut().and_then(|d| {
            let h = d.manifest.tensors.hidden?;
            let snap = d.hidden_snap.as_ref()?.base;
            Some((d, h, snap))
        }) else {
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
            snap + lo as u64 * n,
            d.addr[usize::from(h)] + lo as u64 * n,
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
        let Some((d, snap)) = self
            .spec_draft
            .as_mut()
            .filter(|d| d.hidden_bytes == row_bytes)
            .and_then(|d| {
                let snap = d.hidden_snap.as_ref()?.base;
                Some((d, snap))
            })
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
                snap + b as u64 * n,
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
        if let Some(slot) = self
            .spec_draft
            .as_mut()
            .and_then(|d| d.hidden_at.get_mut(b))
        {
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
        let (Some(h), Some(snap)) = (d.manifest.tensors.hidden, d.hidden_snap.as_ref()) else {
            return Ok(());
        };
        let n = d.hidden_bytes;
        self.be.memcpy_dtod_async(
            d.addr[usize::from(h)] + u64::from(row) * n,
            snap.base + b as u64 * n,
            n,
            &self.stream,
        )
    }

    /// What this target proves to an external drafter, with the emitted context bound and, per
    /// packet tensor, its declared (emitted) byte size and whether it is a transient activation. Only the object symbols in `symbols`
    /// are read, from the object draft programs launch on.
    pub(super) fn spec_target_view(
        &self,
        assets: &Path,
        symbols: &std::collections::BTreeMap<String, u32>,
    ) -> Result<(
        plow_asset::spec_drafter::TargetView,
        u32,
        Vec<u64>,
        Vec<bool>,
    )> {
        use plow_asset::spec_drafter::{TargetView, TokenizerFacts};
        use sha2::Digest;
        let pkt = DevBlob::find_in_dir(assets)?
            .ok_or_else(|| reject(format!("no packet in {}", assets.display())))?;
        let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io {
            path: pkt.clone(),
            source,
        })?;
        let blob = DevBlob::parse(&raw)?;
        let mut view = TargetView {
            packet_sha256: format!("{:x}", sha2::Sha256::digest(&raw)),
            tensors: blob
                .tensors
                .iter()
                .map(|t| (t.name.clone(), t.bytes))
                .collect(),
            ..Default::default()
        };
        if let Some(bytes) = blob.reserved_metadata(&raw, plow_asset::spec_target::SECTION)? {
            plow_asset::spec_target::TargetSpec::parse(bytes)
                .map_err(reject)?
                .apply(&mut view);
        }
        let checkpoint = crate::asset::serve::checkpoint_dir(assets);
        if let Some(path) = [
            assets.join("tokenizer.json"),
            checkpoint.join("tokenizer.json"),
        ]
        .into_iter()
        .find(|p| p.is_file())
        {
            let bytes = std::fs::read(&path).map_err(|source| RuntimeError::Io { path, source })?;
            view.tokenizer = Some(TokenizerFacts::from_json(&bytes).map_err(reject)?);
        }
        let module: &crate::device::Module = self
            .row_slot_twin
            .as_ref()
            .map_or(&self.module, |(full, _)| full.module());
        for sym in symbols.keys() {
            if let Some(v) = self.be.module_global_u32(module, sym)? {
                view.object_symbols.insert(sym.clone(), v);
            }
        }
        let ceiling = crate::exec::ctx_bound::declared_ctx(&blob).unwrap_or(0);
        let declared = blob.tensors.iter().map(|t| t.bytes).collect();
        let transient = (0..blob.tensors.len())
            .map(|h| crate::asset::drafter::transient(&blob, h))
            .collect();
        Ok((view, ceiling, declared, transient))
    }

    /// The attached drafter's identity: `None` for a packet's own draft programs.
    pub fn spec_drafter_digest(&self) -> Option<&str> {
        self.spec_draft
            .as_ref()
            .and_then(|d| d.external.as_ref())
            .map(|x| x.digest.as_str())
    }

    /// Whether the loaded drafter is a block drafter (`spec_draft_stage_block`).
    pub fn spec_draft_is_block(&self) -> bool {
        self.spec_draft
            .as_ref()
            .is_some_and(|d| d.manifest.block.is_some())
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
