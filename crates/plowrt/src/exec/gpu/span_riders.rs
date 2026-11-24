//! Span riders: the verify spans of a unified token batch on a Mamba packed packet (rows
//! `[row0, row0 + rows)`, planned after the prefill pack) share the launch's row-wise ops and run
//! the widest decode rung's own recurrent and attention instructions on the row-slot twin
//! between prefill segments, so their state and attention are the decode verify launch's
//! arithmetic (op_mamba_span.cuh: no state write, step records, deferred replay).
//!
//! Per Mamba layer, after its prefill scan segment (which zeroes rows outside its requests):
//! `[Mamba2Conv span, Mamba2Step span]` of the decode rung, then the bucket's own
//! `Mamba2GatedNorm` (the prefill out_proj's input). Per attention layer, after its
//! `FlashPrefill` segment: the decode rung's `[FlashDecode, FlashMerge]`. The programs read a
//! tensor table of their own: the engine's, with the row operands viewed at `row0` and the row
//! map, activity and KV lengths pointed at rider tables.
use super::*;

struct Block {
    arg: DevProgram,
    counter_bytes: usize,
    insts: Vec<DevInst64>,
    _tables: Vec<DeviceMem>,
}

/// An attention block's `[FlashDecode, FlashMerge]` (program indices `at`) as each decode rung
/// runs them (`rungs[k]`, rows `SpanRiders::rung_rows[k]`): the riders run the instructions of
/// the rung a verify launch of the same rows selects, so their split count and row extent are
/// that rung's.
struct AttnSite {
    block: usize,
    at: [usize; 2],
    rungs: Vec<[DevInst64; 2]>,
}

#[derive(Clone, Copy)]
struct View {
    handle: usize,
    pitch: u64,
}

/// A rider span: `rows` consecutive rows of `slot` whose last row's KV ends at `kv_end`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Span {
    pub slot: usize,
    pub rows: u32,
    pub kv_end: u32,
}

pub(super) struct SpanRiders {
    blocks: Vec<Block>,
    /// Per prefill bucket, `(segment, block)`: the block runs before that segment launches.
    sites: Vec<Vec<(usize, usize)>>,
    table: DeviceMem,
    host: Vec<u64>,
    views: Vec<View>,
    /// Handles of `in.rowslot`, the recurrent `in.active` and `in.kvlen`.
    remap: [usize; 3],
    own: DeviceMem,
    stage: Vec<i32>,
    twin: Arc<BoundDecodeObject>,
    attn: Vec<AttnSite>,
    rung_rows: Vec<u32>,
    /// The rung whose attention instructions (and split count) the blocks hold.
    cur: (usize, u32),
    grid: u32,
    /// Rows of the rider programs (the widest decode rung's).
    rows: u32,
    armed: bool,
}

fn reject(m: &str) -> RuntimeError {
    RuntimeError::Rejected(format!("span riders: {m}"))
}

fn unpack(d: &DevInst64) -> packet::dev::DevInst {
    let mut e = packet::dev::DevInst {
        op: d.op,
        blocks: d.blocks,
        ..Default::default()
    };
    for k in 0..8 {
        e.t[k] = if d.t[k] == TENSOR_NONE16 {
            packet::dev::TENSOR_NONE
        } else {
            u32::from(d.t[k])
        };
    }
    e.i = d.i;
    e.f = [f32::from_bits(d.fj[0]), f32::from_bits(d.fj[1])];
    e.j = [0, d.fj[2]];
    e
}

/// The rider program of `insts`, each depending on the previous one.
fn build(e: &GpuEngine, blob: &DevBlob, insts: &[DevInst64], rows: u32) -> Result<Block> {
    let mut b = packet::devbuild::Builder::new(e.grid);
    b.adopt_tensors(
        blob.tensors
            .iter()
            .map(|t| packet::devbuild::TensorDecl {
                name: t.name.clone(),
                bytes: t.bytes,
                init: None,
            })
            .collect(),
    );
    b.force_uniseg();
    let mut dep: Option<u32> = None;
    for d in insts {
        let op = DevOp::from_u16(d.op).ok_or_else(|| reject("unknown opcode"))?;
        let src = unpack(d);
        let cus: Vec<u32> = (0..u32::from(d.blocks).clamp(1, e.grid)).collect();
        let deps: Vec<u32> = dep.into_iter().collect();
        dep = Some(b.emit(op, cus, &deps, |x| {
            x.t = src.t;
            x.i = src.i;
            x.f = src.f;
            x.j = src.j;
        }));
    }
    let p = b.finish();
    let packed: Vec<DevInst64> = p.insts.iter().map(|d| d.pack()).collect();
    let view = plow_asset::program::Program {
        rows,
        role: packet::devbuild::ProgramRole::DecodeRung { rows },
        n_counter: p.n_counter,
        insts: &packed,
        stream: &p.stream,
        stream_ofs: &p.stream_ofs,
        stream_len: &p.stream_len,
        waits: &p.waits,
        succs: &p.succs,
        gq_stream: &p.gq_stream,
        gq_seg_ofs: &p.gq_seg_ofs,
        l2_domains: p.l2_domains,
    };
    let bytes =
        plow_asset::aux_program::encode(e.grid, blob.tensors.len(), &[view]).map_err(|m| reject(&m))?;
    let section =
        plow_asset::aux_program::parse(&bytes, e.grid, e.devp.len()).map_err(|m| reject(&m))?;
    let [program] = section.programs.as_slice() else {
        return Err(reject("expected one program"));
    };
    let (arg, _, counter_bytes, tables) =
        super::mixed_step::upload_program(&e.be, program, e.d_tens.base, 0, 0)?;
    Ok(Block {
        arg,
        counter_bytes,
        insts: packed,
        _tables: tables,
    })
}

impl SpanRiders {
    /// The route for `e`'s packed Mamba prefill buckets, when the decode rungs are span-bound with
    /// replay rollback and every rider program's anchor segment exists in a bucket.
    pub(super) fn load(e: &GpuEngine, blob: &DevBlob) -> Result<Option<Self>> {
        let replay = e
            .recurrent
            .as_ref()
            .and_then(|r| r.verify.as_ref())
            .is_some_and(|(v, _)| v.replay.is_some());
        let Some((twin, _)) = e.row_slot_twin.as_ref() else {
            return Ok(None);
        };
        if !e.mamba_packs() || !replay || e.seg_pf.is_none() || e.pf_batch.is_none() {
            return Ok(None);
        }
        let Some(decode) = blob.decode_progs().last().copied() else {
            return Ok(None);
        };
        let rows = decode.t;
        let none = TENSOR_NONE16;
        let mut views: Vec<View> = Vec::new();
        let mut view = |handle: u16, pitch: u64| -> Result<()> {
            if handle == none || pitch == 0 {
                return Err(reject("row operand absent"));
            }
            let handle = usize::from(handle);
            match views.iter().find(|v| v.handle == handle) {
                Some(v) if v.pitch != pitch => Err(reject("one tensor viewed at two pitches")),
                Some(_) => Ok(()),
                None => {
                    views.push(View { handle, pitch });
                    Ok(())
                }
            }
        };
        let mut remap: [Option<u16>; 3] = [None; 3];
        let mut bind = |k: usize, h: u16| -> Result<()> {
            match remap[k] {
                Some(x) if x != h => Err(reject("rider tables differ between sites")),
                _ => {
                    remap[k] = Some(h);
                    Ok(())
                }
            }
        };
        // Each layer's decode instructions and the key that finds its prefill anchor.
        enum Kind {
            Mamba { conv_weight: u16 },
            Attention { k_cache: u16 },
        }
        let mut layers: Vec<(Kind, Vec<DevInst64>)> = Vec::new();
        let insts = &decode.insts;
        for (pc, d) in insts.iter().enumerate() {
            match DevOp::from_u16(d.op) {
                Some(DevOp::Mamba2Conv) if d.i[4] == 0 => {
                    let Some(step) = insts[pc + 1..]
                        .iter()
                        .find(|s| s.op == DevOp::Mamba2Step as u16)
                    else {
                        return Err(reject("conv without a step"));
                    };
                    let span = d.t[6] != none && step.i[6] >> 16 != 0 && step.i[7] >> 16 != 0;
                    if !span || step.t[1] != d.t[0] || step.t[2] != d.t[1] || step.t[7] != d.t[5] {
                        return Ok(None);
                    }
                    let (c, source) = (u64::from(d.i[0]), u64::from(d.fj[2] & 0xFFFF));
                    view(d.t[1], 2 * if source != 0 { source } else { c })?;
                    view(d.t[0], 2 * c)?;
                    view(step.t[0], 2 * u64::from(step.i[0] * step.i[1]))?;
                    bind(0, d.t[6])?;
                    bind(1, d.t[5])?;
                    layers.push((Kind::Mamba { conv_weight: d.t[2] }, vec![*d, *step]));
                }
                Some(DevOp::FlashDecode) => {
                    let Some(merge) = insts[pc + 1..]
                        .iter()
                        .find(|m| m.op == DevOp::FlashMerge as u16)
                        .filter(|m| m.t[1] == d.t[0] && m.t[2] == d.t[1])
                    else {
                        return Ok(None);
                    };
                    view(d.t[2], 2 * u64::from(d.i[1] * d.i[6]))?;
                    view(merge.t[0], 2 * u64::from(merge.i[1] * merge.i[3]))?;
                    bind(0, d.t[6])?;
                    bind(2, d.t[5])?;
                    layers.push((Kind::Attention { k_cache: d.t[3] }, vec![*d, *merge]));
                }
                Some(DevOp::FlashDecodeFp8) => return Ok(None),
                _ => {}
            }
        }
        let [Some(rowslot), Some(active), Some(kvlen)] = remap else {
            return Ok(None);
        };
        // Each rung's `[FlashDecode, FlashMerge]` per attention layer, narrow rungs first; a rung
        // whose pair differs from the widest's beyond its scalars is not one the riders can take.
        let pair_of = |insts: &[DevInst64], k_cache: u16| -> Option<[DevInst64; 2]> {
            let pc = insts
                .iter()
                .position(|d| d.op == DevOp::FlashDecode as u16 && d.t[3] == k_cache)?;
            let merge = insts[pc + 1..]
                .iter()
                .find(|m| m.op == DevOp::FlashMerge as u16 && m.t[1] == insts[pc].t[0])?;
            Some([insts[pc], *merge])
        };
        let mut rung_rows: Vec<u32> = Vec::new();
        let mut rung_pairs: Vec<Vec<[DevInst64; 2]>> = Vec::new();
        for (rows_k, insts_k) in e
            .decode_rungs
            .iter()
            .map(|r| (r.rows as u32, r.host_insts.as_slice()))
            .chain(std::iter::once((rows, insts.as_slice())))
        {
            let pairs: Option<Vec<[DevInst64; 2]>> = layers
                .iter()
                .filter_map(|(kind, l)| match kind {
                    Kind::Attention { k_cache } => Some((*k_cache, [l[0], l[1]])),
                    Kind::Mamba { .. } => None,
                })
                .map(|(k_cache, wide)| {
                    let p = pair_of(insts_k, k_cache)?;
                    let same = |a: &DevInst64, b: &DevInst64| a.op == b.op && a.t == b.t && a.blocks == b.blocks;
                    (same(&p[0], &wide[0]) && same(&p[1], &wide[1])).then_some(p)
                })
                .collect();
            if let Some(pairs) = pairs {
                rung_rows.push(rows_k);
                rung_pairs.push(pairs);
            }
        }
        if rung_rows.last() != Some(&rows) {
            return Ok(None);
        }
        // The gated norm of each Mamba layer (prefill flavour), identical in every bucket but
        // for its row count; its rows become the riders'.
        let mut sites = Vec::with_capacity(e.prefill.len());
        let mut gnorms: Vec<Option<DevInst64>> = vec![None; layers.len()];
        for bucket in &e.prefill {
            let segment_of = |pc: usize| {
                bucket
                    .segment_sites
                    .iter()
                    .position(|s| s.iter().any(|&(i, _)| i == pc))
            };
            let h = &bucket.h_inst;
            let mut out = Vec::with_capacity(layers.len());
            let ok = (|| -> Option<()> {
                for (li, (kind, _)) in layers.iter().enumerate() {
                    let (anchor, written): (usize, Vec<u16>) = match kind {
                        Kind::Mamba { conv_weight } => {
                            let conv = h.iter().position(|d| {
                                d.op == DevOp::Mamba2Conv as u16 && d.i[4] != 0 && d.t[2] == *conv_weight
                            })?;
                            let gn = conv
                                + h[conv..]
                                    .iter()
                                    .position(|d| d.op == DevOp::Mamba2GatedNorm as u16)?;
                            let mut g = h[gn];
                            g.i[2] = 0;
                            let same = gnorms[li].is_none_or(|x| {
                                let mut x = x;
                                x.i[2] = 0;
                                x == g
                            });
                            if !same || h[gn].t[1] != layers[li].1[1].t[0] || h[gn].t[2] != layers[li].1[0].t[1] {
                                return None;
                            }
                            gnorms[li].get_or_insert(h[gn]);
                            (gn, vec![h[gn].t[0]])
                        }
                        Kind::Attention { k_cache } => {
                            let fl = h.iter().position(|d| {
                                d.op == DevOp::FlashPrefill as u16 && d.t[3] == *k_cache
                            })?;
                            (fl, vec![h[fl].t[5]])
                        }
                    };
                    let seg = segment_of(anchor)?;
                    // Nothing after the anchor in its segment touches what the riders write, and
                    // its consumer runs in a later segment.
                    let shares = bucket.segment_sites[seg]
                        .iter()
                        .any(|&(i, _)| i > anchor && h[i].t.iter().any(|t| written.contains(t)));
                    let consumer = h[anchor + 1..]
                        .iter()
                        .position(|d| d.t.iter().any(|t| written.contains(t)))
                        .and_then(|k| segment_of(anchor + 1 + k))?;
                    if shares || consumer <= seg {
                        return None;
                    }
                    out.push((seg + 1, li));
                }
                Some(())
            })();
            sites.push(if ok.is_some() { out } else { Vec::new() });
        }
        if sites.iter().all(Vec::is_empty) || gnorms.iter().zip(&layers).any(|(g, (k, _))| {
            matches!(k, Kind::Mamba { .. }) && g.is_none()
        }) {
            tracing::info!(route = "token-batch-span-riders", ready = false, "no bucket qualifies");
            return Ok(None);
        }
        let mut blocks = Vec::with_capacity(layers.len());
        let mut attn = Vec::new();
        for ((kind, mut insts), gn) in layers.into_iter().zip(gnorms) {
            let is_attn = matches!(kind, Kind::Attention { .. });
            if let (Kind::Mamba { .. }, Some(mut g)) = (kind, gn) {
                let fp8 = g.i[3] != 0;
                view(g.t[0], u64::from(g.i[0] * g.i[1]) * if fp8 { 1 } else { 2 })?;
                g.i[2] = rows;
                g.t[4] = active;
                insts.push(g);
            }
            let block = build(e, blob, &insts, rows)?;
            if is_attn {
                let find = |op: DevOp| block.insts.iter().position(|d| d.op == op as u16);
                let (Some(fd), Some(fm)) = (find(DevOp::FlashDecode), find(DevOp::FlashMerge)) else {
                    return Err(reject("attention block lost its flash pair"));
                };
                let k = attn.len();
                attn.push(AttnSite {
                    block: blocks.len(),
                    at: [fd, fm],
                    rungs: rung_pairs.iter().map(|p| p[k]).collect(),
                });
            }
            blocks.push(block);
        }
        let n = e.devp.len();
        let host: Vec<u64> = e.devp.iter().map(|m| m.base).collect();
        let table = e.be.alloc(0, (n * 8) as u64)?;
        let own = e.be.alloc(0, u64::from(rows) * 12)?;
        tracing::info!(
            route = "token-batch-span-riders",
            ready = true,
            blocks = blocks.len(),
            rows,
            rungs = ?rung_rows,
            buckets = sites.iter().filter(|s| !s.is_empty()).count(),
            "verify spans riding packed prefill launches run the decode rung's recurrent and attention sites"
        );
        Ok(Some(Self {
            blocks,
            sites,
            table,
            host,
            views,
            remap: [usize::from(rowslot), usize::from(active), usize::from(kvlen)],
            own,
            stage: vec![0; rows as usize * 3],
            twin: Arc::clone(twin),
            attn,
            cur: (rung_rows.len() - 1, 0),
            rung_rows,
            grid: e.grid,
            rows,
            armed: false,
        }))
    }

    pub(super) fn serves(&self, bucket: usize) -> bool {
        self.sites.get(bucket).is_some_and(|s| !s.is_empty())
    }

    pub(super) fn max_rows(&self) -> u32 {
        self.rows
    }

    pub(super) fn armed(&self) -> bool {
        self.armed
    }

    /// Arm the next launch of bucket `bucket` for `spans` at rows `row0..` and stage the
    /// rider table and row tables on `stream`.
    pub(super) fn arm(
        &mut self,
        be: &CudaBackend,
        devp: &[DeviceMem],
        tensors: u64,
        bucket: usize,
        row0: u32,
        spans: &[Span],
        live_nsplit: Option<(u32, u32)>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.armed = false;
        let rows: u32 = spans.iter().map(|s| s.rows).sum();
        if spans.is_empty() || rows > self.rows || !self.serves(bucket) {
            return Err(reject("riders exceed the route"));
        }
        self.select_rung(be, rows, live_nsplit, stream)?;
        for v in &self.views {
            if (u64::from(row0) + u64::from(self.rows)) * v.pitch > devp[v.handle].len {
                return Err(reject("rider rows exceed an activation"));
            }
            self.host[v.handle] = devp[v.handle].base + u64::from(row0) * v.pitch;
        }
        let r = self.rows as usize;
        let own = self.own.base;
        for (k, &h) in self.remap.iter().enumerate() {
            self.host[h] = own + (k * r * 4) as u64;
        }
        // The engine's table as it stands, then the rider entries over it.
        be.memcpy_dtod_async(self.table.base, tensors, (self.host.len() * 8) as u64, stream)?;
        let (map, rest) = self.stage.split_at_mut(r);
        let (act, kvl) = rest.split_at_mut(r);
        let pad = spans[0].slot as i32 | i32::MIN;
        map.fill(pad);
        act.fill(0);
        kvl.fill(1);
        let mut at = 0;
        for s in spans {
            if s.rows == 0 || s.kv_end < s.rows {
                return Err(reject("empty span"));
            }
            for j in 0..s.rows {
                map[at] = s.slot as i32;
                act[at] = 1;
                kvl[at] = (s.kv_end - s.rows + j + 1) as i32;
                at += 1;
            }
        }
        // SAFETY: pageable sources are staged before each call returns; both ranges lie inside
        // their allocations.
        unsafe {
            for h in self.views.iter().map(|v| v.handle).chain(self.remap) {
                be.memcpy_htod_async(
                    self.table.base + (h * 8) as u64,
                    bytemuck::bytes_of(&self.host[h]),
                    stream,
                )?;
            }
            be.memcpy_htod_async(own, bytemuck::cast_slice(&self.stage), stream)?;
        }
        for v in &self.views {
            self.host[v.handle] = devp[v.handle].base;
        }
        for &h in &self.remap {
            self.host[h] = devp[h].base;
        }
        self.armed = true;
        Ok(())
    }

    /// Hold the attention instructions of the rung a verify launch of `rows` rows runs, with the
    /// split count the engine patches into the rungs (`live_nsplit = (baked, live)`).
    fn select_rung(
        &mut self,
        be: &CudaBackend,
        rows: u32,
        live_nsplit: Option<(u32, u32)>,
        stream: &CudaStream,
    ) -> Result<()> {
        let k = self
            .rung_rows
            .iter()
            .position(|&r| r >= rows)
            .ok_or_else(|| reject("riders exceed the rungs"))?;
        let ns = |p: &[DevInst64; 2]| match live_nsplit {
            Some((baked, live)) if p[0].i[5] == baked => live,
            _ => p[0].i[5],
        };
        let want = self.attn.first().map_or(0, |a| ns(&a.rungs[k]));
        if self.cur == (k, want) {
            return Ok(());
        }
        let sz = std::mem::size_of::<DevInst64>();
        for a in &self.attn {
            let block = &mut self.blocks[a.block];
            let mut pair = a.rungs[k];
            let n = ns(&pair);
            pair[0].i[5] = n;
            pair[1].i[2] = n;
            for (j, &pc) in a.at.iter().enumerate() {
                block.insts[pc] = pair[j];
                // SAFETY: pageable source staged before the call returns; `pc` indexes the
                // program's instruction array of plain-data instructions.
                unsafe {
                    let bytes = std::slice::from_raw_parts(
                        (&block.insts[pc] as *const DevInst64).cast::<u8>(),
                        sz,
                    );
                    be.memcpy_htod_async(block.arg.insts + (pc * sz) as u64, bytes, stream)?;
                }
            }
        }
        self.cur = (k, want);
        Ok(())
    }

    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }

    /// Launch the rider programs that run before segment `seg` of bucket `bucket`.
    pub(super) fn launch_before(
        &self,
        be: &CudaBackend,
        bucket: usize,
        seg: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if !self.armed {
            return Ok(());
        }
        let Some(sites) = self.sites.get(bucket) else {
            return Ok(());
        };
        for &(_, b) in sites.iter().filter(|(s, _)| *s == seg) {
            let block = &self.blocks[b];
            be.memset_d8_async(block.arg.counters, 0, block.counter_bytes, stream)?;
            let mut arg = block.arg;
            arg.tensors = self.table.base;
            let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
            be.launch_cooperative(
                self.twin.function,
                self.grid,
                self.twin.block,
                self.twin.smem,
                &mut params,
                Some(stream),
            )?;
        }
        Ok(())
    }
}
