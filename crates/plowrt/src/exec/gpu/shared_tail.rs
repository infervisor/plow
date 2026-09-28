//! KV-SHARED TAIL on the CUDA token-batch route (`plow_asset::kv_shared_tail`).
//!
//! The packed body runs its bucket up to the tail boundary. The sampled rows' carried
//! activations are gathered into rows `0..n`, and the smallest bucket holding `n` rows runs its
//! segments from its own boundary on, each row a one-row request at its own position. The
//! terminal then samples rows `0..n`. Rows are independent past the boundary (no KV write), so
//! every sampled row computes exactly what the whole program computed for it.
use super::*;
use packet::dev::TENSOR_NONE16;

/// A decode rung's tail window: the rung's layers from the KV-shared boundary up to its lm_head,
/// with its flash reading each row's slot from the packed slot map.
struct DecodeTail {
    /// Index into `GpuEngine::decode_rungs`.
    rung: usize,
    rows: usize,
    kernarg: DevProgram,
    /// The rung's counter slab as the skipped prefix leaves it.
    image: DeviceMem,
    _tables: Vec<DeviceMem>,
}

pub(crate) struct SharedTail {
    /// First tail segment of each prefill bucket.
    boundary: Vec<usize>,
    /// Counter slab state after each bucket's prefix, restored before its tail runs.
    counters: Vec<DeviceMem>,
    /// `(tensor, row_bytes, scratch)` per carried activation.
    carried: Vec<(usize, u32, DeviceMem)>,
    capacity: usize,
    rows: DeviceMem,
    gather_insts: Vec<DevInst64>,
    gather_arg: DevProgram,
    gather_counter_bytes: usize,
    _gather_tables: Vec<DeviceMem>,
    _gather_tensors: DeviceMem,
    decode: Vec<DecodeTail>,
    host_rows: Vec<u32>,
    positions: Vec<i32>,
    slots: Vec<i32>,
    table: Vec<i32>,
}

/// The sampled rows of a packed body: `(row, slot, position)`.
pub(crate) type TailRow = (u32, u32, u32);

impl SharedTail {
    pub(crate) fn boundary_segment(&self, bucket: usize) -> usize {
        self.boundary[bucket]
    }

    pub(crate) fn load(
        e: &GpuEngine,
        blob: &DevBlob,
        manifest: Option<&plow_asset::kv_shared_tail::Manifest>,
    ) -> Result<Option<Self>> {
        let Some(manifest) = manifest else {
            return Ok(None);
        };
        let config = RuntimeConfig::get();
        let programs = blob.prefill_progs();
        if !config.nv.pf_shared_tail
            || e.packed_prefill.is_none()
            || e.pf_batch.is_none()
            || e.packed_terminal.is_none()
            || e.attention_gemm.is_some()
            || e.prefill.is_empty()
            || programs.len() != e.prefill.len()
            || e.prefill.iter().any(|b| {
                !uses_segmented_prefill(
                    e.seg_pf.is_some(),
                    false,
                    b.seg_class.len(),
                    &b.packet_segment_roles,
                )
            })
        {
            return Ok(None);
        }
        manifest
            .validate(programs.len())
            .map_err(RuntimeError::Rejected)?;
        let mut boundary = Vec::with_capacity(programs.len());
        let mut counters = Vec::with_capacity(programs.len());
        for ((program, bucket), &mark) in programs.iter().zip(&e.prefill).zip(&manifest.boundaries)
        {
            let mark = mark as usize;
            let reject = |why: &str| {
                RuntimeError::Rejected(format!("kv-shared tail, bucket {}: {why}", bucket.t))
            };
            let seg = program
                .gq_stream
                .iter()
                .find(|s| s.inst as usize == mark)
                .map(|s| s.seg as usize)
                .ok_or_else(|| reject("boundary has no stream entry"))?;
            if seg == 0
                || seg >= bucket.seg_class.len()
                || program
                    .gq_stream
                    .iter()
                    .any(|s| ((s.inst as usize) < mark) != ((s.seg as usize) < seg))
            {
                return Err(reject("boundary does not open a segment"));
            }
            if bucket.kernarg.counters != bucket.d_ctr.base
                || program.n_counter as usize * CTR_STRIDE as usize * 4 > bucket.ctr_bytes
            {
                return Err(reject("unexpected counter slab layout"));
            }
            // Every signal the prefix makes, counted: the slab exactly as the prefix leaves it.
            let mut image = vec![0u32; bucket.ctr_bytes / 4];
            for s in program.gq_stream.iter().filter(|s| (s.inst as usize) < mark) {
                for &c in &program.succs[s.succ_ofs as usize..(s.succ_ofs + u32::from(s.succ_len)) as usize] {
                    image[c as usize * CTR_STRIDE as usize] += 1;
                }
            }
            let mem = e.be.alloc(0, bucket.ctr_bytes as u64)?;
            e.be.upload(&mem, 0, pod_bytes(&image))?;
            boundary.push(seg);
            counters.push(mem);
        }

        let capacity = e.batch;
        let rows = e.be.alloc(0, capacity as u64 * 4)?;
        let mut pointers: Vec<u64> = e.devp.iter().map(|mem| mem.base).collect();
        let rows_handle = u16::try_from(pointers.len() + manifest.carried.len())
            .ok()
            .filter(|&h| h < TENSOR_NONE16)
            .ok_or_else(|| RuntimeError::Rejected("kv-shared tail: tensor handles exhausted".into()))?;
        let blocks = u16::try_from(capacity)
            .map_err(|_| RuntimeError::Rejected("kv-shared tail: capacity".into()))?;
        let mut carried = Vec::with_capacity(manifest.carried.len());
        let mut insts = Vec::with_capacity(manifest.carried.len());
        for c in &manifest.carried {
            let tensor = c.tensor as usize;
            let bytes = u64::from(c.row_bytes);
            if e.devp.get(tensor).is_none_or(|m| m.len < e.pf_max_rows() as u64 * bytes) {
                return Err(RuntimeError::Rejected(format!(
                    "kv-shared tail: carried tensor {tensor} is not [rows][{bytes}]"
                )));
            }
            let scratch = e.be.alloc(0, capacity as u64 * bytes)?;
            let mut gather = DevInst64 {
                op: DevOp::RowGather as u16,
                blocks,
                t: [TENSOR_NONE16; 8],
                ..Default::default()
            };
            gather.t[..3].copy_from_slice(&[pointers.len() as u16, c.tensor, rows_handle]);
            gather.i[1] = c.row_bytes / 2;
            pointers.push(scratch.base);
            insts.push(gather);
            carried.push((tensor, c.row_bytes, scratch));
        }
        pointers.push(rows.base);
        let program = super::packed_terminal::chain(insts, e.grid_pf, capacity as u32);
        let tensors = e.be.alloc(0, pointers.len() as u64 * 8)?;
        e.be.upload(&tensors, 0, pod_bytes(&pointers))?;
        let (gather_arg, _, gather_counter_bytes, tables) =
            super::mixed_step::upload_program(&e.be, &program, tensors.base, 0, 0)?;
        let decode = decode_tails(e, blob, manifest)?;
        tracing::info!(
            first_layer = manifest.first_layer,
            decode_rungs = decode.len(),
            carried = carried.len(),
            boundaries = ?boundary,
            "kv-shared tail: packed prefill runs the trailing KV-shared layers on sampled rows"
        );
        Ok(Some(Self {
            boundary,
            counters,
            carried,
            capacity,
            rows,
            gather_insts: program.insts,
            gather_arg,
            gather_counter_bytes,
            _gather_tables: tables,
            _gather_tensors: tensors,
            decode,
            host_rows: Vec::with_capacity(capacity),
            positions: Vec::new(),
            slots: Vec::new(),
            table: Vec::new(),
        }))
    }

    /// Whether `n` sampled rows fit the tail.
    pub(crate) fn fits(&self, n: usize) -> bool {
        n <= self.capacity
    }
}

/// Counter slab image after every stream entry of `insts < mark` has signalled.
fn prefix_image(g: &DevProg, mark: usize, bytes: usize) -> Vec<u32> {
    let mut image = vec![0u32; bytes / 4];
    for s in g.gq_stream.iter().filter(|s| (s.inst as usize) < mark) {
        for &c in &g.succs[s.succ_ofs as usize..(s.succ_ofs + u32::from(s.succ_len)) as usize] {
            image[c as usize * CTR_STRIDE as usize] += 1;
        }
    }
    image
}

fn decode_tails(
    e: &GpuEngine,
    blob: &DevBlob,
    manifest: &plow_asset::kv_shared_tail::Manifest,
) -> Result<Vec<DecodeTail>> {
    let Some(slot_handle) = e.packed_prefill.as_ref().map(|p| p.slot) else {
        return Ok(Vec::new());
    };
    let first = blob.progs.iter().position(|p| p.role.is_decode_rung());
    let mut out = Vec::new();
    for (index, rung) in e.decode_rungs.iter().enumerate() {
        if rung.library.is_some() {
            continue;
        }
        let Some((k, g)) = first.and_then(|first| {
            blob.progs[first..]
                .iter()
                .enumerate()
                .find(|(_, p)| p.role.is_decode_rung() && p.t as usize == rung.rows)
        }) else {
            continue;
        };
        let mark = manifest.decode_boundaries.get(k).copied().unwrap_or(0) as usize;
        // Through the rung's own lm_head/argmax: the ids and logits land in rows 0..n, where
        // the packed terminal would have put them.
        let end = rung.host_insts.len();
        let head = rung.host_insts.iter().any(|d| d.t[0] as usize == e.t_logits)
            && rung.host_insts.iter().any(|d| d.t[0] as usize == e.t_ids);
        if mark == 0 || mark >= end || !head {
            continue;
        }
        let mut insts = rung.host_insts.clone();
        let mut ok = true;
        for d in &mut insts[mark..end] {
            match DevOp::from_u16(d.op) {
                Some(DevOp::FlashDecode) => {
                    if d.t[6] != TENSOR_NONE16 {
                        ok = false;
                    }
                    d.t[6] = slot_handle;
                }
                Some(
                    DevOp::HeadNormRope
                    | DevOp::HeadNormRopeFp8
                    | DevOp::FlashDecodeFp8
                    | DevOp::FlashPrefill
                    | DevOp::FlashPrefillFp8,
                ) if insts_writes_kv(d, e) => ok = false,
                _ => {}
            }
        }
        if !ok {
            continue;
        }
        let stream: Vec<_> = g
            .gq_stream
            .iter()
            .filter(|s| (mark..end).contains(&(s.inst as usize)))
            .map(|s| {
                let mut s = *s;
                s.seg = 0;
                s
            })
            .collect();
        let upload = |bytes: &[u8]| -> Result<DeviceMem> {
            let mem = e.be.alloc(0, bytes.len().max(4) as u64)?;
            if !bytes.is_empty() {
                e.be.upload(&mem, 0, bytes)?;
            }
            Ok(mem)
        };
        let tables = vec![
            upload(pod_bytes(&insts))?,
            upload(pod_bytes(&stream))?,
            upload(pod_bytes(&[0u32, stream.len() as u32]))?,
        ];
        let image = e.be.alloc(0, rung.counter_bytes as u64)?;
        e.be.upload(&image, 0, pod_bytes(&prefix_image(g, mark, rung.counter_bytes)))?;
        let kernarg = DevProgram {
            insts: tables[0].base,
            gq_stream: tables[1].base,
            gq_seg_ofs: tables[2].base,
            cur_seg: 0,
            ..rung.kernarg
        };
        out.push(DecodeTail {
            rung: index,
            rows: rung.rows,
            kernarg,
            image,
            _tables: tables,
        });
    }
    Ok(out)
}

/// A KV write inside the tail window would make it unsound; the manifest says there is none.
fn insts_writes_kv(d: &DevInst64, e: &GpuEngine) -> bool {
    e.tensor_names
        .get(d.t[0] as usize)
        .is_some_and(|name| name.starts_with("kv."))
}

impl GpuEngine {
    /// After a packed body enqueued with `pf_seg_prefix`, run the KV-shared tail for `rows`
    /// (`live` = the body's real rows) and leave the sampled rows' final hidden in rows
    /// `0..rows.len()`. Stream-ordered; nothing is waited on.
    pub(super) fn shared_tail_enqueue(&mut self, rows: &[TailRow], live: usize) -> Result<()> {
        let n = rows.len();
        if n == 0 {
            return Ok(());
        }
        let Some(f_pf) = self.f_pf else {
            return Err(RuntimeError::Rejected("prefill object not loaded".into()));
        };
        let mut tail = self.shared_tail.take().expect("caller checked");
        let result = (|| -> Result<()> {
            if !tail.fits(n) || rows.iter().any(|r| r.0 as usize >= live) {
                return Err(RuntimeError::Rejected("kv-shared tail rows out of bounds".into()));
            }
            // 1. Gather the carried rows into scratch, then back to rows 0..n.
            tail.host_rows.clear();
            tail.host_rows.extend(rows.iter().map(|r| r.0));
            for inst in &mut tail.gather_insts {
                inst.i[0] = n as u32;
                inst.i[2] = live as u32;
            }
            // SAFETY: both sources live on `tail` until the stream drains (the host
            // copies from pageable memory before returning).
            unsafe {
                self.be
                    .memcpy_htod_async(tail.rows.base, pod_bytes(&tail.host_rows), &self.stream)?;
                self.be.memcpy_htod_async(
                    tail.gather_arg.insts,
                    pod_bytes(&tail.gather_insts),
                    &self.stream,
                )?;
            }
            self.be.memset_d8_async(
                tail.gather_arg.counters,
                0,
                tail.gather_counter_bytes,
                &self.stream,
            )?;
            let mut arg = tail.gather_arg;
            let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
            self.be.launch_cooperative(
                f_pf,
                self.grid_pf,
                BLOCK,
                self.smem_pf,
                &mut params,
                Some(&self.stream),
            )?;
            for (tensor, row_bytes, scratch) in &tail.carried {
                self.be.memcpy_dtod_async(
                    self.devp[*tensor].base,
                    scratch.base,
                    n as u64 * u64::from(*row_bytes),
                    &self.stream,
                )?;
            }

            // 2a. A decode rung holding n rows runs the tail as a decode step: row r reads slot
            // rows[r].1's cache at kvlen position+1. Pad rows read one row of the first slot.
            if let Some(d) = tail.decode.iter().filter(|d| d.rows >= n).min_by_key(|d| d.rows) {
                let (slot0, rows_d) = (rows[0].1 as i32, d.rows);
                tail.positions.clear();
                tail.slots.clear();
                tail.table.clear();
                for &(_, slot, pos) in rows {
                    tail.positions.push(pos as i32);
                    tail.slots.push(slot as i32);
                    tail.table.push(pos as i32 + 1);
                }
                tail.positions.resize(rows_d, 0);
                tail.slots.resize(rows_d, slot0);
                tail.table.resize(rows_d, 1);
                let pb = self.pf_batch.as_ref().expect("packed");
                // SAFETY: the sources live on `tail` until the stream drains.
                unsafe {
                    self.be.memcpy_htod_async(
                        self.devp[self.t_pos].base,
                        bytemuck::cast_slice(&tail.positions),
                        &self.stream,
                    )?;
                    self.be.memcpy_htod_async(
                        pb.d_slot.base,
                        bytemuck::cast_slice(&tail.slots),
                        &self.stream,
                    )?;
                    self.be.memcpy_htod_async(
                        self.devp[self.t_kvlen].base,
                        bytemuck::cast_slice(&tail.table),
                        &self.stream,
                    )?;
                }
                let rung = &self.decode_rungs[d.rung];
                self.be.memcpy_dtod_async(
                    rung.counters.base,
                    d.image.base,
                    rung.counter_bytes as u64,
                    &self.stream,
                )?;
                let mut arg = d.kernarg;
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                let object = rung.object.as_deref();
                if let Some(terminal) = self.packed_terminal.as_mut() {
                    terminal.skip_next();
                }
                return self.be.launch_cooperative(
                    object.map_or(self.f, |o| o.function),
                    object.map_or(self.grid, |o| o.grid),
                    object.map_or(BLOCK, |o| o.block),
                    object.map_or(
                        if rung.group_arena { self.smem } else { self.smem_narrow },
                        |o| o.smem,
                    ),
                    &mut params,
                    Some(&self.stream),
                );
            }

            // 2b. The tail bucket's request tables: row r is a one-row request at its position.
            let bt = self
                .prefill
                .iter()
                .position(|b| b.t as usize >= n)
                .ok_or_else(|| RuntimeError::Rejected("kv-shared tail bucket".into()))?;
            let tc = self.prefill[bt].t as usize;
            let masked = self.seg_pf.as_ref().is_some_and(|sp| sp.masked_padding);
            let (last_slot, last_pos) = (rows[n - 1].1 as i32, rows[n - 1].2 as i32);
            tail.positions.clear();
            tail.slots.clear();
            tail.table.clear();
            tail.table.push(n as i32);
            for (r, &(_, slot, pos)) in rows.iter().enumerate() {
                tail.positions.push(pos as i32);
                tail.slots.push(slot as i32);
                tail.table
                    .extend_from_slice(&[r as i32, 1, slot as i32, pos as i32 + 1]);
            }
            tail.positions.resize(tc, last_pos);
            tail.slots.resize(tc, if masked { -1 } else { last_slot });
            let kvlen = [rows.iter().map(|r| r.2 as i32 + 1).max().unwrap_or(0)];
            let pb = self.pf_batch.as_ref().expect("packed");
            // SAFETY: as above.
            unsafe {
                self.be.memcpy_htod_async(
                    self.devp[self.t_pos].base,
                    bytemuck::cast_slice(&tail.positions),
                    &self.stream,
                )?;
                self.be
                    .memcpy_htod_async(pb.d_slot.base, bytemuck::cast_slice(&tail.slots), &self.stream)?;
                self.be
                    .memcpy_htod_async(pb.d_req.base, bytemuck::cast_slice(&tail.table), &self.stream)?;
                self.be.memcpy_htod_async(
                    self.devp[self.t_kvlen].base,
                    bytemuck::cast_slice(&kvlen),
                    &self.stream,
                )?;
            }
            self.ensure_batch_patch(bt)?;
            self.patch_moe_rows(bt, n as u32)?;

            // 3. The tail bucket's segments from its boundary, over the prefix's counter state.
            let (ctr_base, ctr_bytes, arg) = {
                let b = &self.prefill[bt];
                (b.d_ctr.base, b.ctr_bytes, b.kernarg)
            };
            self.be.memcpy_dtod_async(
                ctr_base,
                tail.counters[bt].base,
                ctr_bytes as u64,
                &self.stream,
            )?;
            self.pf_seg_window = Some(tail.boundary[bt]..self.prefill[bt].seg_class.len());
            let chain =
                self.launch_prefill_chain(bt, arg, f_pf, rows[0].1 as usize, 0, n, tc, false);
            self.pf_seg_window = None;
            chain
        })();
        self.shared_tail = Some(tail);
        if result.is_err() {
            let _ = self.be.stream_synchronize(&self.stream);
        }
        result
    }
}

impl GpuEngine {
    /// `(row, slot, position)` of each sampled row of a packed body over `chunks`, or `None`
    /// when this engine has no KV-shared tail or the rows do not fit it.
    pub(super) fn shared_tail_rows(
        &self,
        chunks: &[PackedTokenReq<'_>],
        sample_rows: &[u32],
    ) -> Option<smallvec::SmallVec<[TailRow; 32]>> {
        let tail = self.shared_tail.as_ref()?;
        if !tail.fits(sample_rows.len()) {
            return None;
        }
        sample_rows
            .iter()
            .map(|&row| {
                let mut start = 0u32;
                chunks.iter().find_map(|chunk| {
                    let len = chunk.tokens.len() as u32;
                    let hit = (start..start + len).contains(&row).then(|| {
                        (row, chunk.slot as u32, (chunk.c0 as u32) + row - start)
                    });
                    start += len;
                    hit
                })
            })
            .collect()
    }
}
