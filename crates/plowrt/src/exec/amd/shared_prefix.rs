use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::asset::devblob::{DevBlob, DevTensor};
use crate::memory::vmm::{Attach, VmmGeometry, VmmKv, VmmOps};
use crate::{Result, RuntimeError};
use packet::dev::DevOp;

struct CacheTensor {
    id: usize,
    layer: u32,
    role: u32,
    row_bytes: u64,
    slot_bytes: u64,
    base: u64,
}

pub(super) struct Layout {
    groups: Vec<Vec<CacheTensor>>,
    scales: Vec<CacheTensor>,
    batch: usize,
    context: u32,
}

impl Layout {
    pub fn from_blob(blob: &DevBlob, batch: usize, context: u32) -> Option<Self> {
        let layout = Self::from_tensors(&blob.tensors, batch, context)?;
        let caches: BTreeMap<_, _> = layout
            .groups
            .iter()
            .flatten()
            .chain(&layout.scales)
            .map(|t| (t.id, t))
            .collect();
        if blob.progs.is_empty() {
            return None;
        }
        for program in &blob.progs {
            let mut written = BTreeSet::new();
            let mut read = BTreeSet::new();
            for d in &program.insts {
                let operands: Vec<_> =
                    d.t.iter()
                        .enumerate()
                        .filter_map(|(slot, id)| caches.get(&(*id as usize)).map(|t| (slot, *t)))
                        .collect();
                if operands.is_empty() {
                    continue;
                }
                let mut allowed = BTreeSet::new();
                match DevOp::from_u16(d.op)? {
                    DevOp::HeadNormRope | DevOp::HeadNormRopeFp8 => {
                        let dst = caches.get(&(d.t[0] as usize))?;
                        let fp8 = d.op == DevOp::HeadNormRopeFp8 as u16;
                        let elem = if fp8 { 1 } else { 2 };
                        if dst.role == 3
                            || d.i[1] != 1
                            || u64::from(d.i[2]) * elem != dst.row_bytes
                            || d.i[3] != 0
                            || d.fj[2] != u32::MAX
                            || (d.i[6] != 0 && d.fj[1] != context)
                            || (d.i[7] != 0 && d.i[7] != context)
                        {
                            return None;
                        }
                        allowed.insert(0);
                        written.insert(dst.id);
                        if fp8 {
                            let scale = caches.get(&(d.t[6] as usize))?;
                            if dst.role != 0 || scale.role != 3 || scale.layer != dst.layer {
                                return None;
                            }
                            allowed.insert(6);
                            written.insert(scale.id);
                        }
                    }
                    DevOp::RmsNorm => {
                        let dst = caches.get(&(d.t[0] as usize))?;
                        if dst.role != 0
                            || dst.row_bytes != 1024
                            || d.i[1] != 512
                            || d.i[2] != 0
                            || (d.i[7] != 0 && d.i[7] != context)
                        {
                            return None;
                        }
                        allowed.insert(0);
                        written.insert(dst.id);
                    }
                    DevOp::FlashMlaDecode
                    | DevOp::FlashMlaPrefill
                    | DevOp::FlashGatherDecode
                    | DevOp::FlashGatherPrefill
                    | DevOp::FlashMlaDecodeFp8
                    | DevOp::FlashMlaPrefillFp8 => {
                        let latent = caches.get(&(d.t[4] as usize))?;
                        let rope = caches.get(&(d.t[5] as usize))?;
                        let fp8 = matches!(
                            DevOp::from_u16(d.op)?,
                            DevOp::FlashMlaDecodeFp8 | DevOp::FlashMlaPrefillFp8
                        );
                        if latent.role != 0
                            || rope.role != 1
                            || latent.layer != rope.layer
                            || latent.row_bytes != if fp8 { 512 } else { 1024 }
                            || d.i[2] != context
                            || d.i[3] != 0
                            || d.i[5] != u32::MAX
                        {
                            return None;
                        }
                        for t in [latent, rope] {
                            read.insert(t.id);
                        }
                        allowed.extend([4, 5]);
                        if fp8 {
                            let scale = caches.get(&(d.t[7] as usize))?;
                            if scale.role != 3
                                || scale.layer != latent.layer
                                || (d.fj[1] == 0 && d.i[6] != 0)
                            {
                                return None;
                            }
                            read.insert(scale.id);
                            allowed.insert(7);
                        }
                    }
                    DevOp::IndexScore | DevOp::IndexScorePf => {
                        let index = caches.get(&(d.t[2] as usize))?;
                        if index.role != 2 || d.i[2] != context || d.i[3] != 128 {
                            return None;
                        }
                        read.insert(index.id);
                        allowed.insert(2);
                    }
                    DevOp::IndexTpPf => {
                        let index = caches.get(&(d.t[3] as usize))?;
                        if index.role != 2 || d.i[1] != context {
                            return None;
                        }
                        read.insert(index.id);
                        allowed.insert(3);
                    }
                    _ => return None,
                }
                if operands.iter().any(|(slot, _)| !allowed.contains(slot)) {
                    return None;
                }
            }
            if caches
                .iter()
                .any(|(id, t)| !written.contains(id) || (t.role != 2 && !read.contains(id)))
            {
                return None;
            }
        }
        Some(layout)
    }

    fn from_tensors(tensors: &[DevTensor], batch: usize, context: u32) -> Option<Self> {
        if batch == 0 || context == 0 {
            return None;
        }
        let mut groups: BTreeMap<u64, Vec<CacheTensor>> = BTreeMap::new();
        let mut scales = Vec::new();
        let mut layers: BTreeMap<u32, BTreeSet<&str>> = BTreeMap::new();
        let divisor = (batch as u64).checked_mul(context as u64)?;
        for (id, tensor) in tensors.iter().enumerate() {
            let Some(name) = tensor.name.strip_prefix("kv.") else {
                continue;
            };
            let (layer, kind) = name.split_once('.')?;
            let layer = layer.parse().ok()?;
            if tensor.init.is_some() || tensor.bytes == 0 || tensor.bytes % divisor != 0 {
                return None;
            }
            let row_bytes = tensor.bytes / divisor;
            let role = match (kind, row_bytes) {
                ("ckv", 512 | 1024) => 0,
                ("krot", 128) => 1,
                ("kidx", 256) => 2,
                ("scale", 4) => 3,
                _ => return None,
            };
            if !layers.entry(layer).or_default().insert(kind) {
                return None;
            }
            let cache = CacheTensor {
                id,
                layer,
                role,
                row_bytes,
                slot_bytes: tensor.bytes / batch as u64,
                base: 0,
            };
            if role == 3 {
                scales.push(cache);
            } else {
                groups.entry(row_bytes).or_default().push(cache);
            }
        }
        if layers.is_empty()
            || layers
                .values()
                .any(|k| !k.contains("ckv") || !k.contains("krot"))
        {
            return None;
        }
        for caches in groups.values() {
            for cache in caches.iter().filter(|cache| cache.role == 0) {
                if layers[&cache.layer].contains("scale") != (cache.row_bytes == 512) {
                    return None;
                }
            }
        }
        Some(Self {
            groups: groups.into_values().collect(),
            scales,
            batch,
            context,
        })
    }
}

struct Group {
    pool: VmmKv,
    tensors: Vec<CacheTensor>,
}

pub(super) struct SharedPrefix {
    groups: Vec<Group>,
    scales: Vec<CacheTensor>,
    pending: Vec<Vec<Attach>>,
    ops: Arc<dyn VmmOps>,
}

// Admission resets every rank's slot first; a clean miss already has private backing.
pub(super) fn attach_ranks<T>(
    ranks: &mut [T],
    slot: usize,
    prompt: &[u32],
    mut cache: impl for<'a> FnMut(&'a mut T) -> &'a mut SharedPrefix,
) -> Result<u32> {
    if ranks.is_empty()
        || ranks
            .iter_mut()
            .any(|rank| slot >= cache(rank).pending.len())
    {
        return Err(RuntimeError::Rejected(
            "prefix attachment requires ranks and a valid slot".into(),
        ));
    }
    let mut attempted = 0;
    let result = (|| {
        let mut common = None;
        for rank in ranks.iter_mut() {
            attempted += 1;
            let rows = cache(rank).stage_attach(slot, prompt)?;
            if rows == 0 {
                attempted -= 1;
                return Ok(0);
            }
            if common.is_some_and(|common| common != rows) {
                return Ok(0);
            }
            common = Some(rows);
        }
        let rows = common.unwrap();
        for rank in ranks.iter_mut() {
            let cache = cache(rank);
            cache.commit_attach(slot, rows)?;
            tracing::debug!(
                slot,
                rows,
                shared_mappings = cache.shared_mappings(),
                "AMD shared prefix attached"
            );
        }
        Ok(rows)
    })();
    if !matches!(result, Ok(rows) if rows > 0) {
        let mut rollback_error = None;
        for rank in &mut ranks[..attempted] {
            if let Err(error) = cache(rank).begin_slot(slot) {
                rollback_error = Some(error);
            }
        }
        if let Some(error) = rollback_error {
            return Err(error);
        }
    }
    result
}

impl SharedPrefix {
    pub fn new(
        ops: Arc<dyn VmmOps>,
        layout: Layout,
        cache_cap: u64,
        pool_cap: u64,
    ) -> Result<Self> {
        let granularity = ops.granularity()?;
        let total_row_bytes: u64 = layout.groups.iter().flatten().map(|t| t.row_bytes).sum();
        let mut groups = Vec::new();
        for tensors in layout.groups {
            let row_bytes = tensors[0].row_bytes;
            let layers: BTreeSet<_> = tensors.iter().map(|t| t.layer).collect();
            let keys: Vec<_> = tensors.iter().map(|t| (t.layer, t.role)).collect();
            let geometry = VmmGeometry {
                full_layers: layers.into_iter().collect(),
                kvh_full: 1,
                hd_full: row_bytes as u32,
                slide_layers: Vec::new(),
                kvh_slide: 0,
                hd_slide: 0,
                window: 0,
                elem: 1,
                elem_slide: 1,
                max_ctx: layout.context,
                batch: u32::try_from(layout.batch)
                    .map_err(|_| RuntimeError::Rejected("prefix batch overflow".into()))?,
            };
            let cap = if cache_cap == 0 {
                0
            } else {
                (cache_cap / total_row_bytes * row_bytes * tensors.len() as u64).max(1)
            };
            let mut pool = VmmKv::new_tensors(ops.clone(), geometry, granularity, cap, &keys)?;
            pool.enable_block_recycling(
                pool_cap / total_row_bytes * row_bytes * tensors.len() as u64,
            );
            groups.push(Group { pool, tensors });
        }
        Ok(Self {
            groups,
            scales: layout.scales,
            pending: (0..layout.batch).map(|_| Vec::new()).collect(),
            ops,
        })
    }

    pub fn tensor_va(&self, id: usize) -> Option<u64> {
        self.groups.iter().find_map(|g| {
            let t = g.tensors.iter().find(|t| t.id == id)?;
            g.pool.tensor_va(t.layer, t.role)
        })
    }

    pub fn bind(&mut self, bases: &[u64]) {
        for tensor in self
            .groups
            .iter_mut()
            .flat_map(|g| &mut g.tensors)
            .chain(&mut self.scales)
        {
            tensor.base = bases[tensor.id];
        }
    }

    pub fn ensure_rows(&self, slot: usize, rows: u32) -> Result<()> {
        for group in &self.groups {
            group.pool.ensure_rows(slot, rows)?;
        }
        Ok(())
    }

    pub fn release(&self, slot: usize) {
        for group in &self.groups {
            group.pool.release_prefix(slot);
        }
    }

    pub fn shared_mappings(&self) -> u64 {
        self.groups
            .iter()
            .map(|g| g.pool.stats().blocks_shared_mapped)
            .sum()
    }

    /// Private block mappings made by `ensure_rows` so far (one driver map each).
    pub fn mappings(&self) -> u64 {
        self.groups
            .iter()
            .map(|g| {
                let s = g.pool.stats();
                s.blocks_created + s.blocks_reused
            })
            .sum()
    }

    pub fn begin_slot(&mut self, slot: usize) -> Result<()> {
        self.pending[slot].clear();
        for group in &self.groups {
            group.pool.begin_seq(slot);
        }
        self.ensure_rows(slot, 1)
    }

    pub fn stage_attach(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
        self.pending[slot].clear();
        let result = (|| {
            let mut rows = None;
            for group in &self.groups {
                let Some(attach) = group.pool.try_attach(slot, prompt)? else {
                    return Ok(0);
                };
                if rows.is_some_and(|rows| rows != attach.rows) {
                    return Ok(0);
                }
                rows = Some(attach.rows);
                self.pending[slot].push(attach);
            }
            Ok(rows.unwrap_or(0))
        })();
        if result.is_err() || matches!(result, Ok(0)) && !self.pending[slot].is_empty() {
            self.begin_slot(slot)?;
        }
        result
    }

    fn snapshot_bytes(&self, group: usize, rows: u32) -> u64 {
        let g = &self.groups[group];
        let tail = u64::from(rows % g.pool.block_rows());
        let cache_bytes: u64 = g.tensors.iter().map(|t| tail * t.row_bytes).sum();
        let scale_bytes: u64 = if group == 0 {
            self.scales
                .iter()
                .map(|t| u64::from(rows) * t.row_bytes)
                .sum()
        } else {
            0
        };
        (cache_bytes + scale_bytes).max(4)
    }

    fn copy_snapshot(
        &self,
        group: usize,
        slot: usize,
        rows: u32,
        mut snapshot: u64,
        to_snapshot: bool,
    ) -> Result<()> {
        let g = &self.groups[group];
        let tail = rows % g.pool.block_rows();
        let mut pairs = Vec::new();
        let mut copy = |tensor: &CacheTensor, start: u32, count: u32| {
            let bytes = u64::from(count) * tensor.row_bytes;
            if bytes != 0 {
                let address = tensor.base
                    + slot as u64 * tensor.slot_bytes
                    + u64::from(start) * tensor.row_bytes;
                let (dst, src) = if to_snapshot {
                    (snapshot, address)
                } else {
                    (address, snapshot)
                };
                pairs.push((dst, src, bytes));
                snapshot += bytes;
            }
        };
        for tensor in &g.tensors {
            copy(tensor, rows - tail, tail);
        }
        if group == 0 {
            for tensor in &self.scales {
                copy(tensor, 0, rows);
            }
        }
        self.ops.copy_dtod_batch(&pairs)
    }

    pub fn commit_attach(&mut self, slot: usize, rows: u32) -> Result<()> {
        if rows == 0
            || self.pending[slot].len() != self.groups.len()
            || self.pending[slot].iter().any(|a| a.rows != rows)
        {
            return Err(RuntimeError::Device(
                "prefix attach was not staged on every cache group".into(),
            ));
        }
        self.ensure_rows(slot, rows + 1)?;
        for (group, attach) in self.pending[slot].iter().enumerate() {
            if attach.snap_bytes != self.snapshot_bytes(group, rows) {
                return Err(RuntimeError::Device(
                    "prefix snapshot layout mismatch".into(),
                ));
            }
            self.copy_snapshot(group, slot, rows, attach.snap_va, false)?;
        }
        for group in &self.groups {
            group.pool.finish_attach(slot);
        }
        self.pending[slot].clear();
        Ok(())
    }

    pub fn publish(&self, slot: usize, prompt: &[u32], frontier: u32) -> Result<()> {
        let rows = frontier.min(prompt.len().saturating_sub(1) as u32) / 32 * 32;
        if rows == 0 {
            return Ok(());
        }
        for (index, group) in self.groups.iter().enumerate() {
            group.pool.publish_at(
                slot,
                prompt,
                rows,
                self.snapshot_bytes(index, rows),
                |snapshot| self.copy_snapshot(index, slot, rows, snapshot, true),
            )?;
        }
        Ok(())
    }

    pub fn publish_completed_chunk(
        &self,
        slot: usize,
        prompt: &[u32],
        frontier: u32,
    ) -> Result<()> {
        if frontier as usize >= prompt.len()
            || self
                .groups
                .iter()
                .all(|g| frontier % g.pool.block_rows() == 0)
        {
            self.publish(slot, prompt, frontier)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    #[ignore = "requires PLOW_SHARED_PREFIX_PACKET pointing to a compiled GLM packet"]
    fn compiled_glm_cache_layout() {
        let path =
            std::env::var_os("PLOW_SHARED_PREFIX_PACKET").expect("PLOW_SHARED_PREFIX_PACKET");
        let mut blob = DevBlob::parse_l2(&std::fs::read(path).unwrap(), true).unwrap();
        let context = (blob
            .tensors
            .iter()
            .find(|t| t.name == "in.pos")
            .unwrap()
            .bytes
            / 4) as u32;
        let batch = blob.progs.last().unwrap().t as usize;
        if let Some(baseline) = std::env::var_os("PLOW_SHARED_PREFIX_BASELINE") {
            let old = DevBlob::parse_l2(&std::fs::read(baseline).unwrap(), true).unwrap();
            assert!(
                Layout::from_blob(&old, batch, context).is_none(),
                "baseline must reproduce incomplete cache writes"
            );
            assert_eq!(old.tensors.len(), blob.tensors.len());
            for (a, b) in old.tensors.iter().zip(&blob.tensors) {
                assert_eq!((&a.name, a.bytes), (&b.name, b.bytes));
            }
            for (a, b) in old
                .progs
                .iter()
                .zip(&blob.progs)
                .filter(|(a, _)| a.t <= batch as u32)
            {
                assert_eq!(a.t, b.t);
                assert_eq!(a.insts.len(), b.insts.len());
                for (a, b) in a.insts.iter().zip(&b.insts) {
                    assert_eq!(
                        (a.op, a.blocks, a.fj, a.t, a.i),
                        (b.op, b.blocks, b.fj, b.t, b.i)
                    );
                }
            }
        }
        if Layout::from_blob(&blob, batch, context).is_none() {
            let shapes: BTreeSet<_> = blob
                .tensors
                .iter()
                .filter(|t| t.name.starts_with("kv."))
                .map(|t| {
                    (
                        t.name.rsplit('.').next().unwrap(),
                        t.bytes / batch as u64 / context as u64,
                    )
                })
                .collect();
            eprintln!("batch={batch} context={context} shapes={shapes:?}");
            let inventory: BTreeSet<_> = blob
                .progs
                .iter()
                .flat_map(|p| &p.insts)
                .filter_map(|d| {
                    let operands: Vec<_> =
                        d.t.iter()
                            .enumerate()
                            .filter_map(|(i, h)| {
                                blob.tensors
                                    .get(*h as usize)
                                    .filter(|t| t.name.starts_with("kv."))
                                    .map(|t| (i, t.name.rsplit('.').next().unwrap()))
                            })
                            .collect();
                    (!operands.is_empty()).then(|| {
                        format!(
                            "{:?} {operands:?} i={:?} fj={:?}",
                            DevOp::from_u16(d.op),
                            d.i,
                            d.fj
                        )
                    })
                })
                .collect();
            for line in inventory {
                eprintln!("{line}");
            }
        }
        assert!(Layout::from_blob(&blob, batch, context).is_some());
        let tensor = blob
            .tensors
            .iter()
            .position(|t| t.name.ends_with(".ckv"))
            .unwrap();
        blob.tensors[tensor].bytes -= 1;
        assert!(Layout::from_blob(&blob, batch, context).is_none());
        blob.tensors[tensor].bytes += 1;
        for op in [
            DevOp::FlashMlaPrefillFp8,
            DevOp::FlashMlaDecodeFp8,
            DevOp::IndexScore,
            DevOp::IndexTpPf,
        ] {
            let (p, i) = blob
                .progs
                .iter()
                .enumerate()
                .find_map(|(p, program)| {
                    program
                        .insts
                        .iter()
                        .position(|d| d.op == op as u16)
                        .map(|i| (p, i))
                })
                .expect("compiled qualification packet must include the production cache routes");
            let stride = if op == DevOp::IndexTpPf { 1 } else { 2 };
            blob.progs[p].insts[i].i[stride] += 1;
            assert!(
                Layout::from_blob(&blob, batch, context).is_none(),
                "accepted stale {op:?} stride"
            );
            blob.progs[p].insts[i].i[stride] -= 1;
        }
        assert!(Layout::from_blob(&blob, batch, context).is_some());
    }

    #[derive(Default)]
    struct Memory {
        next: u64,
        blocks: BTreeMap<u64, Vec<u8>>,
        mappings: BTreeMap<u64, (u64, u64)>,
        fail_copy: usize,
    }

    impl Memory {
        fn id(&mut self) -> u64 {
            self.next += 1 << 32;
            self.next
        }

        fn location(&self, address: u64) -> (u64, usize) {
            let (&base, &(handle, size)) = self.mappings.range(..=address).next_back().unwrap();
            assert!(address - base < size, "unmapped address {address:#x}");
            (handle, (address - base) as usize)
        }

        fn read(&self, address: u64, bytes: u64) -> Vec<u8> {
            (address..address + bytes)
                .map(|a| {
                    let (handle, offset) = self.location(a);
                    self.blocks[&handle][offset]
                })
                .collect()
        }

        fn write(&mut self, address: u64, data: &[u8]) {
            for (i, value) in data.iter().enumerate() {
                let (handle, offset) = self.location(address + i as u64);
                self.blocks.get_mut(&handle).unwrap()[offset] = *value;
            }
        }
    }

    #[derive(Default)]
    struct Driver(Mutex<Memory>);

    impl VmmOps for Driver {
        fn granularity(&self) -> Result<u64> {
            Ok(16384)
        }
        fn reserve(&self, _: u64) -> Result<u64> {
            Ok(self.0.lock().unwrap().id())
        }
        fn address_free(&self, _: u64, _: u64) {}
        fn create(&self, bytes: u64) -> Result<u64> {
            let mut m = self.0.lock().unwrap();
            let id = m.id();
            m.blocks.insert(id, vec![0xa5; bytes as usize]);
            Ok(id)
        }
        fn release(&self, handle: u64) {
            assert!(self.0.lock().unwrap().blocks.remove(&handle).is_some());
        }
        fn map(&self, va: u64, bytes: u64, handle: u64) -> Result<()> {
            assert!(self
                .0
                .lock()
                .unwrap()
                .mappings
                .insert(va, (handle, bytes))
                .is_none());
            Ok(())
        }
        fn unmap(&self, va: u64, _: u64) {
            assert!(self.0.lock().unwrap().mappings.remove(&va).is_some());
        }
        fn set_access(&self, _: u64, _: u64) -> Result<()> {
            Ok(())
        }
        fn alloc(&self, bytes: u64) -> Result<u64> {
            let handle = self.create(bytes)?;
            self.map(handle, bytes, handle)?;
            Ok(handle)
        }
        fn free(&self, va: u64) {
            self.unmap(va, 0);
            self.release(va);
        }
        fn copy_dtod(&self, dst: u64, src: u64, bytes: u64) -> Result<()> {
            let mut m = self.0.lock().unwrap();
            if m.fail_copy != 0 {
                m.fail_copy -= 1;
                if m.fail_copy == 0 {
                    return Err(RuntimeError::Device(
                        "injected snapshot copy failure".into(),
                    ));
                }
            }
            let data = m.read(src, bytes);
            m.write(dst, &data);
            Ok(())
        }
    }

    fn tensors() -> Vec<DevTensor> {
        [("ckv", 512), ("krot", 128), ("kidx", 256), ("scale", 4)]
            .into_iter()
            .map(|(name, bytes)| DevTensor {
                name: format!("kv.0.{name}"),
                bytes: bytes * 256 * 3,
                init: None,
            })
            .collect()
    }

    #[test]
    fn unsupported_or_ambiguous_cache_shapes_are_refused() {
        assert!(Layout::from_tensors(&tensors(), 3, 256).is_some());
        for mutate in [
            |t: &mut Vec<DevTensor>| {
                t[0].bytes -= 1;
            },
            |t: &mut Vec<DevTensor>| {
                t[0].bytes *= 2;
            },
            |t: &mut Vec<DevTensor>| {
                t[1].bytes /= 2;
            },
            |t: &mut Vec<DevTensor>| {
                t[2].name = "kv.0.kidx_pool".into();
            },
            |t: &mut Vec<DevTensor>| {
                t[3].init = Some(0..4);
            },
            |t: &mut Vec<DevTensor>| {
                t.pop();
            },
            |t: &mut Vec<DevTensor>| {
                t[1].name = "kv.1.krot".into();
            },
            |t: &mut Vec<DevTensor>| {
                t[2].name = "kv.0.krot".into();
            },
        ] {
            let mut t = tensors();
            mutate(&mut t);
            assert!(Layout::from_tensors(&t, 3, 256).is_none());
        }
        let mut bf16 = tensors();
        bf16[0].bytes *= 2;
        bf16.pop();
        assert!(Layout::from_tensors(&bf16, 3, 256).is_some());
    }

    fn cache(ops: Arc<Driver>) -> (SharedPrefix, Vec<u64>) {
        let tensors = tensors();
        let layout = Layout::from_tensors(&tensors, 3, 256).unwrap();
        let mut cache = SharedPrefix::new(ops.clone(), layout, 0, 0).unwrap();
        let bases: Vec<_> = tensors
            .iter()
            .enumerate()
            .map(|(id, t)| {
                cache
                    .tensor_va(id)
                    .unwrap_or_else(|| ops.alloc(t.bytes).unwrap())
            })
            .collect();
        cache.bind(&bases);
        (cache, bases)
    }

    fn fill(ops: &Driver, bases: &[u64], slot: u64, rows: u64, value: u8) {
        let mut m = ops.0.lock().unwrap();
        for (&base, row_bytes) in bases.iter().zip([512, 128, 256, 4]) {
            m.write(
                base + slot * 256 * row_bytes,
                &vec![value; (rows * row_bytes) as usize],
            );
        }
    }

    #[test]
    fn cold_miss_preserves_private_writable_backing() {
        let ops = Arc::new(Driver::default());
        let (mut cache, bases) = cache(ops.clone());
        cache.begin_slot(0).unwrap();
        fill(&ops, &bases, 0, 1, 77);
        let mappings = ops.0.lock().unwrap().mappings.clone();
        assert_eq!(
            cache
                .stage_attach(0, &(0..100).collect::<Vec<_>>())
                .unwrap(),
            0
        );
        {
            let memory = ops.0.lock().unwrap();
            assert_eq!(memory.mappings, mappings);
            assert_eq!(memory.read(bases[0], 512), vec![77; 512]);
        }
        fill(&ops, &bases, 0, 1, 99);
        drop(cache);
        ops.free(bases[3]);
        assert!(ops.0.lock().unwrap().blocks.is_empty());
    }

    #[test]
    fn rank_attachment_rolls_back_hits_without_remapping_clean_misses() {
        for case in 0..5 {
            let ops: Vec<_> = (0..3).map(|_| Arc::new(Driver::default())).collect();
            let (mut ranks, bases): (Vec<_>, Vec<_>) = ops.iter().cloned().map(cache).unzip();
            let prompt: Vec<_> = (0..100).collect();
            for rank in 0..3 {
                ranks[rank].begin_slot(0).unwrap();
                ranks[rank].ensure_rows(0, 100).unwrap();
                fill(&ops[rank], &bases[rank], 0, 100, 17 + rank as u8);
                if case != 0 && !(case == 1 && rank == 1) {
                    let frontier = if case == 2 && rank == 1 { 64 } else { 100 };
                    ranks[rank].publish(0, &prompt, frontier).unwrap();
                }
                ranks[rank].begin_slot(1).unwrap();
            }
            let mappings: Vec<_> = ops
                .iter()
                .map(|op| op.0.lock().unwrap().mappings.clone())
                .collect();
            if case == 3 {
                ops[1].0.lock().unwrap().fail_copy = 2;
            }
            let result = attach_ranks(&mut ranks, 1, &prompt, |cache| cache);
            if case == 3 {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap(), if case == 4 { 96 } else { 0 });
            }
            for rank in 0..3 {
                let untouched = case == 0 || case == 1 && rank >= 1 || case == 2 && rank == 2;
                if untouched {
                    assert_eq!(ops[rank].0.lock().unwrap().mappings, mappings[rank]);
                }
                if case != 4 {
                    fill(&ops[rank], &bases[rank], 1, 1, 99);
                }
                let memory = ops[rank].0.lock().unwrap();
                assert_eq!(
                    memory.read(bases[rank][0], 96 * 512),
                    vec![17 + rank as u8; 96 * 512]
                );
                if case == 4 {
                    assert_eq!(
                        memory.read(bases[rank][0] + 256 * 512, 96 * 512),
                        vec![17 + rank as u8; 96 * 512]
                    );
                }
            }
            drop(ranks);
            for (ops, bases) in ops.iter().zip(&bases) {
                ops.free(bases[3]);
                assert!(ops.0.lock().unwrap().blocks.is_empty());
            }
        }
    }

    #[test]
    fn block_pool_grows_on_demand_and_recycles_all_groups_within_one_budget() {
        let ops = Arc::new(Driver::default());
        let layout = Layout::from_tensors(&tensors(), 3, 256).unwrap();
        let budget = (512 + 128 + 256) * 256;
        let mut cache = SharedPrefix::new(ops.clone(), layout, 0, budget).unwrap();
        assert!(cache.groups.iter().all(|g| {
            let stats = g.pool.stats();
            stats.blocks_created == 0 && stats.blocks_live == 0 && stats.blocks_pooled == 0
        }));
        cache.ensure_rows(0, 256).unwrap();
        let created: u64 = cache
            .groups
            .iter()
            .map(|g| g.pool.stats().blocks_created)
            .sum();
        for group in &cache.groups {
            group.pool.begin_seq(0);
        }
        let pooled: u64 = cache
            .groups
            .iter()
            .map(|g| g.pool.stats().blocks_pooled)
            .sum();
        assert_eq!(pooled * ops.granularity().unwrap(), budget);
        cache.begin_slot(0).unwrap();
        cache.ensure_rows(0, 256).unwrap();
        assert_eq!(
            cache
                .groups
                .iter()
                .map(|g| g.pool.stats().blocks_created)
                .sum::<u64>(),
            created
        );
        assert_eq!(
            cache
                .groups
                .iter()
                .map(|g| g.pool.stats().blocks_reused)
                .sum::<u64>(),
            created
        );
        drop(cache);
        let memory = ops.0.lock().unwrap();
        assert!(memory.blocks.is_empty());
        assert!(memory.mappings.is_empty());
    }

    #[test]
    fn full_blocks_alias_but_partial_blocks_scales_and_suffixes_are_private() {
        let ops = Arc::new(Driver::default());
        let (mut cache, bases) = cache(ops.clone());
        let prompt: Vec<_> = (0..100).collect();
        cache.begin_slot(0).unwrap();
        cache.ensure_rows(0, 100).unwrap();
        fill(&ops, &bases, 0, 100, 17);
        cache.publish(0, &prompt, 100).unwrap();
        cache.begin_slot(1).unwrap();
        assert_eq!(cache.stage_attach(1, &prompt).unwrap(), 96);
        cache.commit_attach(1, 96).unwrap();
        {
            let m = ops.0.lock().unwrap();
            for (&base, row_bytes) in bases.iter().zip([512, 128, 256, 4]) {
                assert_eq!(
                    m.read(base + 256 * row_bytes, 96 * row_bytes),
                    vec![17; (96 * row_bytes) as usize]
                );
                if row_bytes != 4 {
                    let shared = matches!(row_bytes, 512 | 256);
                    assert_eq!(
                        m.location(base).0 == m.location(base + 256 * row_bytes).0,
                        shared
                    );
                    assert_ne!(
                        m.location(base + 96 * row_bytes).0,
                        m.location(base + (256 + 96) * row_bytes).0
                    );
                }
            }
        }
        // Retiring/reusing the owner must leave the borrower's prefix intact.
        cache.begin_slot(0).unwrap();
        cache.ensure_rows(0, 100).unwrap();
        fill(&ops, &bases, 0, 100, 29);
        cache.ensure_rows(1, 100).unwrap();
        {
            let mut m = ops.0.lock().unwrap();
            for (&base, row_bytes) in bases.iter().zip([512, 128, 256, 4]) {
                let address = base + 256 * row_bytes;
                m.write(
                    address + 96 * row_bytes,
                    &vec![31; (4 * row_bytes) as usize],
                );
                assert_eq!(
                    m.read(address, 96 * row_bytes),
                    vec![17; (96 * row_bytes) as usize]
                );
            }
        }
        cache.begin_slot(2).unwrap();
        assert_eq!(cache.stage_attach(2, &prompt).unwrap(), 96);
        cache.commit_attach(2, 96).unwrap();
        {
            let m = ops.0.lock().unwrap();
            for (&base, row_bytes) in bases.iter().zip([512, 128, 256, 4]) {
                assert_eq!(
                    m.read(base + 512 * row_bytes, 96 * row_bytes),
                    vec![17; (96 * row_bytes) as usize]
                );
            }
        }
        drop(cache);
        ops.free(bases[3]);
        let m = ops.0.lock().unwrap();
        assert!(m.blocks.is_empty());
        assert!(m.mappings.is_empty());
    }

    #[test]
    fn mismatched_groups_and_failed_snapshot_restore_can_roll_back() {
        let ops = Arc::new(Driver::default());
        let (mut cache, bases) = cache(ops.clone());
        let prompt: Vec<_> = (0..100).collect();
        cache.begin_slot(0).unwrap();
        cache.ensure_rows(0, 100).unwrap();
        fill(&ops, &bases, 0, 100, 17);
        cache.publish(0, &prompt, 64).unwrap();
        let g = &cache.groups[0];
        g.pool
            .publish_at(0, &prompt, 96, cache.snapshot_bytes(0, 96), |va| {
                cache.copy_snapshot(0, 0, 96, va, true)
            })
            .unwrap();
        cache.begin_slot(1).unwrap();
        assert_eq!(cache.stage_attach(1, &prompt).unwrap(), 0);
        fill(&ops, &bases, 1, 1, 99);
        cache.publish(0, &prompt, 100).unwrap();
        cache.begin_slot(1).unwrap();
        assert_eq!(cache.stage_attach(1, &prompt).unwrap(), 96);
        ops.0.lock().unwrap().fail_copy = 2;
        assert!(cache.commit_attach(1, 96).is_err());
        cache.begin_slot(1).unwrap();
        fill(&ops, &bases, 1, 1, 99);
        cache.begin_slot(2).unwrap();
        assert_eq!(cache.stage_attach(2, &prompt).unwrap(), 96);
        cache.commit_attach(2, 96).unwrap();
        let m = ops.0.lock().unwrap();
        assert_eq!(m.read(bases[0] + 512 * 512, 96 * 512), vec![17; 96 * 512]);
        drop(m);
        drop(cache);
        ops.free(bases[3]);
        let m = ops.0.lock().unwrap();
        assert!(m.blocks.is_empty());
        assert!(m.mappings.is_empty());
    }

    #[test]
    fn completed_chunk_boundary_survives_different_request_suffixes() {
        let ops = Arc::new(Driver::default());
        let (mut cache, bases) = cache(ops.clone());
        let prompt: Vec<_> = (0..200).collect();
        cache.begin_slot(0).unwrap();
        cache.ensure_rows(0, 200).unwrap();
        fill(&ops, &bases, 0, 200, 17);
        cache.publish_completed_chunk(0, &prompt, 64).unwrap();
        cache.begin_slot(1).unwrap();
        assert_eq!(cache.stage_attach(1, &prompt).unwrap(), 0);
        cache.publish_completed_chunk(0, &prompt, 128).unwrap();
        cache.publish_completed_chunk(0, &prompt, 200).unwrap();
        let mut different = prompt.clone();
        different[128..].fill(1234);
        assert_eq!(cache.stage_attach(1, &different).unwrap(), 128);
        cache.commit_attach(1, 128).unwrap();
        assert!(cache.shared_mappings() > 0);
        cache.release(0);
        cache.begin_slot(0).unwrap();
        cache.ensure_rows(0, 200).unwrap();
        fill(&ops, &bases, 0, 200, 29);
        cache.begin_slot(2).unwrap();
        assert_eq!(cache.stage_attach(2, &different).unwrap(), 128);
        cache.commit_attach(2, 128).unwrap();
        {
            let m = ops.0.lock().unwrap();
            for (&base, row_bytes) in bases.iter().zip([512, 128, 256, 4]) {
                assert_eq!(
                    m.read(base + 512 * row_bytes, 128 * row_bytes),
                    vec![17; (128 * row_bytes) as usize]
                );
            }
        }
        drop(cache);
        ops.free(bases[3]);
        let m = ops.0.lock().unwrap();
        assert!(m.blocks.is_empty());
        assert!(m.mappings.is_empty());
    }
}
