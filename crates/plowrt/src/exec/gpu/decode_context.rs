use super::*;
use crate::asset::devblob::DevBlob;
use crate::asset::devblob::DevProg;
use plow_asset::decode_context::ContextBand;
use plow_asset::decode_context::{ContextTable, SECTION};
use plow_asset::decode_coverage::DenseBf16;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

fn reject(message: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Rejected(format!("decode context: {message}"))
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(reject(message))
    }
}

pub(super) fn metadata(blob: &DevBlob, raw: &[u8]) -> Result<Option<ContextTable>> {
    let Some(bytes) = blob.reserved_metadata(raw, SECTION)? else {
        return Ok(None);
    };
    let positions: Vec<_> = blob.tensors.iter().filter(|t| t.name == "in.pos").collect();
    require(
        positions.len() == 1 && positions[0].bytes % 4 == 0,
        "ambiguous position capacity",
    )?;
    let max_kv = u32::try_from(positions[0].bytes / 4).map_err(reject)?;
    let programs: Vec<_> = blob
        .progs
        .iter()
        .enumerate()
        .skip(blob.decode_rung_lo())
        .map(|(i, p)| (i, p.t))
        .collect();
    plow_asset::decode_context::parse_sections(
        &[bytes],
        &programs,
        blob.n_cu,
        std::mem::size_of::<DevProgram>(),
        max_kv,
    )
    .map_err(reject)
}

fn ordered(dependencies: &[BTreeSet<usize>], consumer: usize, producer: usize) -> bool {
    let mut pending = vec![consumer];
    let mut seen = BTreeSet::new();
    while let Some(pc) = pending.pop() {
        if pc == producer {
            return true;
        }
        if seen.insert(pc) {
            pending.extend(dependencies[pc].iter().copied());
        }
    }
    false
}

fn attention_lifetimes(blob: &DevBlob, g: &DevProg, deps: &[BTreeSet<usize>]) -> Result<()> {
    let mut previous_merge = BTreeMap::new();
    let mut partials = BTreeSet::new();
    let mut matched_merges = BTreeSet::new();
    for (pc, d) in g
        .insts
        .iter()
        .enumerate()
        .filter(|(_, d)| d.op == DevOp::FlashDecode as u16)
    {
        let ids = [d.t[0], d.t[1], d.t[2], d.t[3], d.t[4], d.t[5]];
        require(
            ids.into_iter().collect::<BTreeSet<_>>().len() == ids.len(),
            "attention partial/input handles alias",
        )?;
        let end = g.insts[pc + 1..]
            .iter()
            .position(|n| {
                n.op == DevOp::FlashDecode as u16
                    && (n.t[..2].contains(&d.t[0]) || n.t[..2].contains(&d.t[1]))
            })
            .map_or(g.insts.len(), |i| pc + 1 + i);
        let merges: Vec<_> = g.insts[pc + 1..end]
            .iter()
            .enumerate()
            .filter(|(_, m)| m.op == DevOp::FlashMerge as u16 && m.t[1..3] == d.t[..2])
            .collect();
        require(
            merges.len() == 1,
            "attention requires one merge before scratch reuse",
        )?;
        let merge_pc = pc + 1 + merges[0].0;
        let merge = merges[0].1;
        require(
            !ids[..2].contains(&merge.t[0]) && matched_merges.insert(merge_pc),
            "merge output aliases partial storage",
        )?;
        require(
            deps[merge_pc].contains(&pc),
            "merge lacks full producer dependency",
        )?;
        require(
            merge.i[..4] == [g.t, d.i[1], d.i[5], d.i[6]],
            "merge split/layout mismatch",
        )?;
        let rows_heads = u64::from(g.t)
            .checked_mul(u64::from(d.i[1]))
            .ok_or_else(|| reject("attention extent overflow"))?;
        let count = rows_heads
            .checked_mul(u64::from(d.i[5]))
            .ok_or_else(|| reject("partial extent overflow"))?;
        require(
            d.i[5] > 0 && count <= u64::from(u32::MAX),
            "invalid partial count",
        )?;
        for (id, elements, width) in [
            (d.t[0], count, u64::from(d.i[6]) * 4),
            (d.t[1], count, 8),
            (d.t[2], rows_heads, u64::from(d.i[6]) * 2),
            (merge.t[0], rows_heads, u64::from(d.i[6]) * 2),
        ] {
            let bytes = elements
                .checked_mul(width)
                .ok_or_else(|| reject("attention bytes overflow"))?;
            let tensor = blob
                .tensors
                .get(id as usize)
                .ok_or_else(|| reject("attention tensor outside binding"))?;
            require(
                tensor.bytes >= bytes,
                "attention tensor capacity is insufficient",
            )?;
        }
        for id in &d.t[..2] {
            let tensor = &blob.tensors[*id as usize];
            require(
                tensor.init.is_none() && !blob.gen.iter().any(|g| g.tensor == u32::from(*id)),
                "partial storage is initialized or generated",
            )?;
            if let Some(&prior) = previous_merge.get(id) {
                require(
                    ordered(deps, pc, prior),
                    "partial reuse is unordered with prior merge",
                )?;
            }
            previous_merge.insert(*id, merge_pc);
            partials.insert(*id);
        }
    }
    require(
        !matched_merges.is_empty(),
        "context program has no attention chain",
    )?;
    for (pc, d) in g.insts.iter().enumerate() {
        if d.op == DevOp::FlashMerge as u16 {
            require(matched_merges.contains(&pc), "orphan attention merge")?;
        }
        for (slot, id) in d.t.iter().enumerate() {
            if partials.contains(id) {
                require(
                    (d.op == DevOp::FlashDecode as u16 && slot < 2)
                        || (d.op == DevOp::FlashMerge as u16 && (slot == 1 || slot == 2)),
                    "partial storage used outside attention/merge",
                )?;
            }
        }
    }
    Ok(())
}

fn auxiliary_program(
    base: &DevBlob,
    aux: &DevBlob,
    band: &ContextBand,
    coverage: DenseBf16,
    splitk: Option<u32>,
) -> Result<()> {
    require(
        base.n_cu == aux.n_cu
            && base.flags == aux.flags
            && base.target == aux.target
            && base.tp == aux.tp
            && base.tp.is_none(),
        "auxiliary target/flags/TP mismatch",
    )?;
    require(
        base.tensors.len() == aux.tensors.len() && base.gen == aux.gen && base.kvrow == aux.kvrow,
        "auxiliary tensor/generated/KV patch contract differs",
    )?;
    for (a, b) in base.tensors.iter().zip(&aux.tensors) {
        require(
            a.name == b.name && a.bytes == b.bytes && a.init.is_some() == b.init.is_some(),
            "auxiliary tensor binding differs",
        )?;
        if let (Some(ar), Some(br)) = (&a.init, &b.init) {
            let a = base
                .init
                .get(ar.clone())
                .ok_or_else(|| reject("base init range invalid"))?;
            let b = aux
                .init
                .get(br.clone())
                .ok_or_else(|| reject("auxiliary init range invalid"))?;
            require(a == b, "auxiliary initialized tensor differs")?;
        }
    }
    require(
        !aux.sections.iter().any(|s| s.name == SECTION),
        "nested context metadata is unsupported",
    )?;
    require(
        validate_decode_ladder(base)? && validate_decode_ladder(aux)?,
        "context requires qualified dense decode ladders",
    )?;
    require(
        band.base_program >= base.decode_rung_lo() && band.program.index >= aux.decode_rung_lo(),
        "context alternative points into prefill",
    )?;
    let a = base
        .progs
        .get(band.base_program)
        .ok_or_else(|| reject("base program index outside packet"))?;
    let b = aux
        .progs
        .get(band.program.index)
        .ok_or_else(|| reject("auxiliary program index outside packet"))?;
    require(
        a.t == band.rows && b.t == band.rows && !a.packed_prefill_only && !b.packed_prefill_only,
        "context program physical rows differ",
    )?;
    require(
        a.n_counter == b.n_counter
            && a.insts.len() == b.insts.len()
            && a.stream == b.stream
            && a.stream_ofs == b.stream_ofs
            && a.stream_len == b.stream_len
            && a.waits == b.waits
            && a.succs == b.succs
            && a.gq_stream == b.gq_stream
            && a.gq_seg_ofs == b.gq_seg_ofs
            && a.l2_domains == b.l2_domains,
        "context program scheduling/dependency tables differ",
    )?;
    let mut full = 0;
    for (pc, (old, new)) in a.insts.iter().zip(&b.insts).enumerate() {
        let mut old = *old;
        let mut new = *new;
        if old.op == DevOp::FlashDecode as u16 && old.i[6] == 512 && old.i[4] == 0 {
            old.i[5] = 0;
            new.i[5] = 0;
            full += 1;
        } else if old.op == DevOp::FlashMerge as u16 && old.i[3] == 512 {
            // The lifetime check below links every merge to an exact producer.
            let producer = a.insts[..pc].iter().rev().find(|d| {
                d.op == DevOp::FlashDecode as u16
                    && d.t[..2] == old.t[1..3]
                    && d.i[6] == 512
                    && d.i[4] == 0
            });
            if producer.is_some() {
                old.i[2] = 0;
                new.i[2] = 0;
            }
        }
        require(
            old == new,
            "non-attention instruction or attention semantics changed",
        )?;
    }
    require(full > 0, "context alternative has no full HD512 attention")?;
    let deps = aux
        .with_packet_view(|p| -> std::result::Result<_, String> {
            coverage.program(p, band.program.index, splitk)?;
            plow_asset::splitk::dependencies(&p.programs[band.program.index])
        })
        .map_err(reject)?;
    let old_deps = base
        .with_packet_view(|p| plow_asset::splitk::dependencies(&p.programs[band.base_program]))
        .map_err(reject)?;
    require(deps == old_deps, "context dependency graph differs")?;
    attention_lifetimes(base, a, &old_deps)?;
    attention_lifetimes(aux, b, &deps)?;
    let mut old_live = base
        .with_packet_view(plow_asset::live_kv::emit)
        .map_err(reject)?;
    let mut new_live = aux
        .with_packet_view(plow_asset::live_kv::emit)
        .map_err(reject)?;
    old_live.programs.clear();
    new_live.programs.clear();
    require(
        old_live == new_live,
        "auxiliary LIVE KV/state layout differs",
    )
}

fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(|source| RuntimeError::Io {
        path: path.into(),
        source,
    })?;
    let size = file
        .metadata()
        .map_err(|source| RuntimeError::Io {
            path: path.into(),
            source,
        })?
        .len();
    let limit = 64 * 1024 * 1024u64;
    require(
        size > 0 && size <= limit,
        "context asset size outside limit",
    )?;
    let mut bytes = Vec::with_capacity(size as usize);
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| RuntimeError::Io {
            path: path.into(),
            source,
        })?;
    require(
        !bytes.is_empty() && bytes.len() as u64 <= limit,
        "context asset grew outside limit",
    )?;
    Ok(bytes)
}

struct PreparedVariant {
    blob: Arc<DevBlob>,
    image: Arc<[u8]>,
}
pub(super) struct PreparedContexts {
    table: ContextTable,
    variants: Vec<PreparedVariant>,
}

pub(super) fn prepare(
    base: &DevBlob,
    raw: &[u8],
    assets: &Path,
) -> Result<Option<PreparedContexts>> {
    let Some(table) = metadata(base, raw)? else {
        return Ok(None);
    };
    let mut packets = BTreeMap::<String, Arc<DevBlob>>::new();
    let mut images = BTreeMap::<String, Arc<[u8]>>::new();
    let mut variants = Vec::new();
    let mut retained_bytes = 0usize;
    for (index, band) in table.bands().iter().enumerate() {
        let blob = if let Some(blob) = packets.get(&band.program.file) {
            Arc::clone(blob)
        } else {
            let bytes = read_bounded(&assets.join(&band.program.file))?;
            retained_bytes += bytes.len();
            require(
                retained_bytes <= 256 * 1024 * 1024,
                "context assets exceed aggregate preparation budget",
            )?;
            table.check_program_image(index, &bytes).map_err(reject)?;
            let blob = Arc::new(DevBlob::parse(&bytes)?);
            packets.insert(band.program.file.clone(), Arc::clone(&blob));
            blob
        };
        let image = if let Some(image) = images.get(&band.object.file) {
            Arc::clone(image)
        } else {
            let image: Arc<[u8]> = read_bounded(&assets.join(&band.object.file))?.into();
            retained_bytes += image.len();
            require(
                retained_bytes <= 256 * 1024 * 1024,
                "context assets exceed aggregate preparation budget",
            )?;
            images.insert(band.object.file.clone(), Arc::clone(&image));
            image
        };
        table.check_object_image(index, &image).map_err(reject)?;
        require(
            plow_asset::cubin::global_u32(&image, "plow_packet_hash_lo").is_none()
                && plow_asset::cubin::global_u32(&image, "plow_packet_hash_hi").is_none(),
            "stamped context objects require auxiliary pairing manifests",
        )?;
        let coverage = DenseBf16::from_image(&image).map_err(reject)?;
        auxiliary_program(
            base,
            &blob,
            band,
            coverage,
            plow_asset::cubin::global_u32(&image, "plow_gemm_splitk_abi"),
        )?;
        require_dynamic_kv(
            &blob.progs[band.program.index],
            plow_asset::cubin::global_u32(&image, "plow_dyn_kvrow"),
        )?;
        variants.push(PreparedVariant { blob, image });
    }
    Ok(Some(PreparedContexts { table, variants }))
}

fn require_dynamic_kv(g: &DevProg, capability: Option<u32>) -> Result<()> {
    require(
        capability == Some(1),
        "context object requires dynamic KV ABI1",
    )?;
    for d in &g.insts {
        if d.op == DevOp::HeadNormRope as u16
            && g.insts
                .iter()
                .any(|a| a.op == DevOp::FlashDecode as u16 && a.t[3..5].contains(&d.t[0]))
        {
            require(
                d.i[6] == g.t && d.i[3] == 0,
                "context KV writer requires immutable per-slot position addressing",
            )?;
        }
    }
    Ok(())
}

pub(super) struct MaterializedContexts {
    table: ContextTable,
    rungs: Vec<Arc<DecodeRung>>,
    be: Arc<CudaBackend>,
}
impl PreparedContexts {
    pub(super) fn materialize(
        self,
        be: &Arc<CudaBackend>,
        base: DevProgram,
        assets: &Path,
    ) -> Result<MaterializedContexts> {
        let mut objects = BTreeMap::new();
        let mut programs = BTreeMap::new();
        let mut rungs = Vec::new();
        for (band, variant) in self.table.bands().iter().zip(&self.variants) {
            let object = if let Some(object) = objects.get(&band.object.file) {
                Arc::clone(object)
            } else {
                let module = DecodeModule::load(be, &variant.image)?;
                GpuEngine::check_packet_pairing(be, &module, assets)?;
                for symbol in plow_asset::decode_coverage::SYMBOLS {
                    require(
                        be.module_global_u32(&module, symbol)?
                            == plow_asset::cubin::global_u32(&variant.image, symbol),
                        "loaded dense coverage differs from image",
                    )?;
                }
                for cap in &band.capabilities {
                    require(
                        be.module_global_u32(&module, &cap.symbol)? == Some(cap.value),
                        "loaded arithmetic capability differs",
                    )?;
                }
                let object = decode_object::bind_module(&band.object, be, module)?;
                objects.insert(band.object.file.clone(), Arc::clone(&object));
                object
            };
            let key = (
                band.program.file.clone(),
                band.program.index,
                band.object.file.clone(),
            );
            let rung = if let Some(rung) = programs.get(&key) {
                Arc::clone(rung)
            } else {
                let program = &variant.blob.progs[band.program.index];
                require_dynamic_kv(
                    program,
                    plow_asset::cubin::global_u32(&variant.image, "plow_dyn_kvrow"),
                )?;
                let mut rung = DecodeRung::upload(be, program, base)?;
                rung.object = Some(object);
                let rung = Arc::new(rung);
                programs.insert(key, Arc::clone(&rung));
                rung
            };
            rungs.push(rung);
        }
        be.synchronize()?;
        Ok(MaterializedContexts {
            table: self.table,
            rungs,
            be: Arc::clone(be),
        })
    }
}
impl MaterializedContexts {
    pub(super) fn select(
        &self,
        positions: &[u32],
        slots: impl IntoIterator<Item = usize>,
    ) -> Result<Option<usize>> {
        Ok(self.table.select(positions, slots).map_err(reject)?.band)
    }

    pub(super) fn rung(&self, index: usize) -> &DecodeRung {
        self.rungs[index].as_ref()
    }
}
impl Drop for MaterializedContexts {
    fn drop(&mut self) {
        if let Err(error) = self.be.synchronize() {
            tracing::warn!(%error,"quiesce context alternatives at unload");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::devblob::DevSection;
    use packet::devbuild::{Builder, Model, TensorDecl, SECT_METADATA};
    use plow_asset::decode_context::{Capability, ContextProgram, DecodeContexts};
    use plow_asset::decode_objects::{image_sha256, DecodeObject};

    fn model(ns: u32) -> Model {
        let progs = [1, 4]
            .into_iter()
            .map(|rows| {
                let mut b = Builder::new(1);
                b.force_uniseg();
                let mut deps = vec![];
                for id in [3, 4] {
                    deps.push(b.emit(DevOp::HeadNormRope, b.all(), &[], |d| {
                        d.t[0] = id;
                        d.t[5] = 0;
                        d.i = [rows, 1, 512, 0, 0, 0, rows, 0];
                        d.j = [1024, u32::MAX];
                    }));
                }
                let fa = b.emit(DevOp::FlashDecode, b.all(), &deps, |d| {
                    d.t[..6].copy_from_slice(&[5, 6, 7, 3, 4, 1]);
                    d.i = [rows, 16, 1, 1024, 0, ns, 512, u32::MAX];
                    d.f[0] = 1.;
                    if rows > 1 {
                        d.j[0] = rows * 1024;
                    }
                });
                let merge = b.emit(DevOp::FlashMerge, b.all(), &[fa], |d| {
                    d.t[..3].copy_from_slice(&[8, 5, 6]);
                    d.i[..4].copy_from_slice(&[rows, 16, ns, 512]);
                });
                b.emit(DevOp::Residual, b.all(), &[merge], |d| {
                    d.t[..3].copy_from_slice(&[9, 8, 8]);
                    d.i[0] = rows * 16 * 512;
                });
                b.finish()
            })
            .collect();
        let tensors = [
            ("in.pos", 4096),
            ("in.kvlen", 16),
            ("in.ids", 16),
            ("key", 4 * 1024 * 512 * 2),
            ("value", 4 * 1024 * 512 * 2),
            ("partial", 4 * 16 * 8 * 512 * 4),
            ("ml", 4 * 16 * 8 * 8),
            ("query", 4 * 16 * 512 * 2),
            ("attention", 4 * 16 * 512 * 2),
            ("hidden", 4 * 16 * 512 * 2),
            ("constant", 4),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (name, bytes))| TensorDecl {
            name: name.into(),
            bytes,
            init: (i == 10).then(|| vec![1, 2, 3, 4]),
        })
        .collect();
        Model {
            n_cu: 1,
            target: 0,
            tensors,
            progs,
            kv_row_insts: vec![],
            prog_t: vec![1, 4],
            gen: vec![],
        }
    }
    fn blob(ns: u32) -> DevBlob {
        DevBlob::parse(&model(ns).to_blob()).unwrap()
    }
    fn image(qk: u32) -> Vec<u8> {
        let globals = [
            ("plow_block", 256),
            ("plow_arena_bytes", 126592),
            ("plow_dyn_kvrow", 1),
            ("plow_segment_gq_abi", 1),
            ("plow_attention_decode_abi", 1),
            ("plow_attention_qk_abi", qk),
            ("plow_attention_pv_abi", 2),
            ("plow_decode_bf16_abi", 1),
            ("plow_decode_gf256", 2),
            ("plow_decode_gf512", 16),
            ("plow_decode_staging_bytes", 16384),
            ("plow_gemv_mm_cap", 16),
        ];
        plow_asset::cubin::synthetic_elf("_Z12interp_sm90a11PlowProgram", &globals, 90)
    }
    fn band(packet: &[u8], object: &[u8]) -> ContextBand {
        ContextBand {
            base_program: 0,
            rows: 1,
            kv_min: 800,
            kv_max: 802,
            program: ContextProgram {
                file: "aux.pkt".into(),
                sha256: image_sha256(packet),
                index: 0,
            },
            object: DecodeObject {
                file: "attention.cubin".into(),
                sha256: image_sha256(object),
                profile: "sm90a".into(),
                entry: "_Z12interp_sm90a11PlowProgram".into(),
                threads: 256,
                arena_bytes: 126592,
                grid: 1,
            },
            capabilities: vec![
                Capability {
                    symbol: "plow_attention_decode_abi".into(),
                    value: 1,
                },
                Capability {
                    symbol: "plow_attention_qk_abi".into(),
                    value: 2,
                },
                Capability {
                    symbol: "plow_attention_pv_abi".into(),
                    value: 2,
                },
            ],
            qualification_sha256: "d".repeat(64),
        }
    }
    fn attach(blob: &mut DevBlob, band: ContextBand) -> Vec<u8> {
        let metadata = DecodeContexts {
            version: 1,
            kernarg_bytes: std::mem::size_of::<DevProgram>(),
            bands: vec![band],
        };
        let raw = serde_json::to_vec(&metadata).unwrap();
        blob.sections.push(DevSection {
            kind: SECT_METADATA,
            name: SECTION.into(),
            offset: 0,
            size: raw.len(),
        });
        raw
    }
    fn coverage() -> DenseBf16 {
        DenseBf16([1, 2, 16, 16384, 16, 126592])
    }

    #[test]
    fn reserved_section_kind_count_range_and_old_absence() {
        let mut base = blob(2);
        assert!(metadata(&base, &[]).unwrap().is_none());
        let raw = attach(&mut base, band(&[], &[]));
        assert!(metadata(&base, &raw).unwrap().is_some());
        base.sections[0].kind = 0;
        assert!(metadata(&base, &raw).is_err());
        base.sections[0].kind = SECT_METADATA;
        base.sections[0].size += 1;
        assert!(metadata(&base, &raw).is_err());
        base.sections[0].size -= 1;
        let s = &base.sections[0];
        let copy = DevSection {
            kind: s.kind,
            name: s.name.clone(),
            offset: s.offset,
            size: s.size,
        };
        base.sections.push(copy);
        assert!(metadata(&base, &raw).is_err());
    }
    #[test]
    fn permits_only_compatible_full_attention_split_change() {
        let base = blob(2);
        let aux = blob(4);
        let b = band(&[], &[]);
        auxiliary_program(&base, &aux, &b, coverage(), None).unwrap();
        let mut b = b;
        b.base_program = 1;
        b.program.index = 1;
        b.rows = 4;
        auxiliary_program(&base, &aux, &b, coverage(), None).unwrap();
        b.program.index = usize::MAX;
        assert!(auxiliary_program(&base, &aux, &b, coverage(), None).is_err());
    }
    #[test]
    fn rejects_tensor_generated_state_or_semantic_mutations() {
        let base = blob(2);
        let b = band(&[], &[]);
        let mutations: &[fn(&mut DevBlob)] = &[
            |b| b.tensors[7].name.push('x'),
            |b| b.tensors[7].bytes += 2,
            |b| b.init[0] ^= 1,
            |b| b.gen.push(packet::rope::GenTensor::default()),
            |b| b.kvrow.push(0),
            |b| b.flags ^= 1,
            |b| b.target ^= 1,
            |b| b.progs[0].t = 2,
            |b| b.progs[0].insts[4].i[0] += 1,
            |b| b.progs[0].insts[2].fj[0] ^= 1,
            |b| b.progs[0].insts[2].i[4] = 64,
            |b| b.progs[0].insts[3].i[2] = 3,
            |b| b.tensors[5].bytes = 16,
            |b| b.progs[0].insts[2].t[0] = 7,
            |b| b.progs[0].waits[0].threshold += 1,
            |b| b.progs[0].gq_stream.swap(0, 3),
            |b| b.progs[0].stream[0].slice = 1,
        ];
        for (case, mutate) in mutations.iter().enumerate() {
            let mut aux = blob(4);
            mutate(&mut aux);
            assert!(
                auxiliary_program(&base, &aux, &b, coverage(), None).is_err(),
                "case={case}"
            );
        }
    }
    #[test]
    fn matching_tables_cannot_hide_partial_lifetime_errors() {
        for case in 0..5 {
            let mut b = blob(2);
            let mut deps = b
                .with_packet_view(|p| plow_asset::splitk::dependencies(&p.programs[0]))
                .unwrap();
            let mut g = b.progs.remove(0);
            let fa = g.insts[2];
            let merge = g.insts[3];
            g.insts.extend([fa, merge]);
            deps.push(BTreeSet::from([4]));
            deps.push(BTreeSet::from([5]));
            match case {
                0 => {}
                1 => deps[3].clear(),
                2 => deps[5].clear(),
                3 => g.insts[4].t[0] = 5,
                4 => g.insts[6].t[1] = 7,
                _ => unreachable!(),
            }
            assert_eq!(
                attention_lifetimes(&b, &g, &deps).is_ok(),
                case == 0,
                "case={case}"
            );
        }
    }
    #[test]
    fn bounded_reads_and_absent_metadata_do_not_allocate_unbounded_files() {
        let dir = std::env::temp_dir().join(format!("plow-context-bounds-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sparse.bin");
        let file = std::fs::File::create(&path).unwrap();
        assert!(read_bounded(&path).is_err());
        file.set_len(64 * 1024 * 1024 + 1).unwrap();
        assert!(read_bounded(&path).is_err());
        assert!(prepare(&blob(2), &[], &dir.join("absent"))
            .unwrap()
            .is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn prepares_immutable_hash_bound_files_without_cuda() {
        let dir = std::env::temp_dir().join(format!("plow-context-prepare-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let packet = model(4).to_blob();
        let object = image(2);
        std::fs::write(dir.join("aux.pkt"), &packet).unwrap();
        std::fs::write(dir.join("attention.cubin"), &object).unwrap();
        let mut base = blob(2);
        let raw = attach(&mut base, band(&packet, &object));
        let prepared = prepare(&base, &raw, &dir).unwrap().unwrap();
        assert_eq!(prepared.variants.len(), 1);
        assert_eq!(
            prepared.table.select(&[799, 0, 0, 0], [0]).unwrap().band,
            Some(0)
        );
        std::fs::write(dir.join("aux.pkt"), b"changed").unwrap();
        assert!(prepare(&base, &raw, &dir).is_err());
        assert_eq!(prepared.variants[0].blob.progs[0].insts[2].i[5], 4);
        assert_eq!(prepared.variants[0].image.as_ref(), object);
        std::fs::write(dir.join("aux.pkt"), &packet).unwrap();
        let wrong = image(1);
        std::fs::write(dir.join("attention.cubin"), &wrong).unwrap();
        assert!(prepare(&base, &raw, &dir).is_err());
        let mut base = blob(2);
        let raw = attach(&mut base, band(&packet, &wrong));
        assert!(prepare(&base, &raw, &dir).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn dispatch_context_routes_and_fallbacks_use_live_physical_slots() {
        let mut base = blob(2);
        let mut first = band(&[], &[]);
        first.kv_min = 10;
        first.kv_max = 20;
        let raw = attach(&mut base, first);
        let table = metadata(&base, &raw).unwrap().unwrap();
        let route = |positions: &[u32], slots: &[usize]| {
            let selection = table.select(positions, slots.iter().copied()).unwrap();
            let narrow = decode_rung_index(std::iter::once(1), *slots.iter().max().unwrap());
            decode_selection(narrow, selection.band)
        };
        assert_eq!(
            route(&[9, 100, 100, 100], &[0]),
            DecodeSelection::Context(0)
        );
        assert_eq!(route(&[19, 0, 0, 0], &[0]), DecodeSelection::Context(0));
        for pos in [0, 8, 20, 100] {
            assert_eq!(route(&[pos, 0, 0, 0], &[0]), DecodeSelection::Base(Some(0)));
        }
        assert_eq!(route(&[9, 9, 9, 9], &[1]), DecodeSelection::Base(None));
        assert_eq!(route(&[9, 9, 9, 9], &[0, 2]), DecodeSelection::Base(None));
        assert_eq!(
            route(&[9, 9, 9, 9], &[0, 1, 2, 3]),
            DecodeSelection::Base(None)
        );
        for slots in [vec![], vec![0, 0], vec![4]] {
            assert!(table.select(&[9; 4], slots).is_err());
        }
        assert!(table.select(&[u32::MAX, 0, 0, 0], [0]).is_err());
    }

    #[test]
    fn sliding_cache_uses_position_capacity_instead_of_batch_or_ring_rows() {
        let mut base = blob(2);
        base.tensors[0].bytes = 32_768 * 4;
        for program in &mut base.progs {
            for writer in &mut program.insts[..2] {
                writer.fj[2] = 1023;
            }
            let attention = &mut program.insts[2];
            attention.i[4] = 1024;
            attention.i[7] = 1023;
        }
        let manifest = base.with_packet_view(plow_asset::live_kv::emit).unwrap();
        assert_eq!(manifest.max_ctx, 32_768);
        assert!(manifest.caches.iter().all(|cache| cache.window == 1024));
        let mut context = band(&[], &[]);
        context.kv_min = 32_000;
        context.kv_max = 32_768;
        let raw = attach(&mut base, context);
        let table = metadata(&base, &raw).unwrap().unwrap();
        assert_eq!(table.select(&[31_999, 0, 0, 0], [0]).unwrap().band, Some(0));
        assert!(table.select(&[32_768, 0, 0, 0], [0]).is_err());
    }

    #[test]
    fn full_prefix_context_selection_rechecks_mixed_lengths_and_reset() {
        let mut base = blob(2);
        let mut b = band(&[], &[]);
        b.base_program = 1;
        b.program.index = 1;
        b.rows = 4;
        b.kv_min = 10;
        b.kv_max = 20;
        let raw = attach(&mut base, b);
        let table = metadata(&base, &raw).unwrap().unwrap();
        let route = |pos: &[u32], slots: &[usize]| {
            let selected = table.select(pos, slots.iter().copied()).unwrap();
            decode_selection(None, selected.band)
        };
        assert_eq!(
            route(&[9, 12, 15, 19], &[0, 1, 2, 3]),
            DecodeSelection::Context(0)
        );
        assert_eq!(
            route(&[9, 12, 15, 20], &[0, 1, 2, 3]),
            DecodeSelection::Base(None)
        );
        assert_eq!(
            route(&[9, 12, 0, 19], &[0, 1, 2, 3]),
            DecodeSelection::Base(None)
        );
        assert_eq!(
            route(&[9, 12, 15, 19], &[0, 2, 3]),
            DecodeSelection::Base(None)
        );
        assert_eq!(
            route(&[9, 12, 15, 19], &[3, 2, 1, 0]),
            DecodeSelection::Context(0)
        );
    }

    #[test]
    fn dynamic_b1_does_not_patch_immutable_auxiliary_instructions() {
        let base = blob(2);
        let before = pod_bytes(&base.progs[0].insts).to_vec();
        require_dynamic_kv(&base.progs[0], Some(1)).unwrap();
        assert_eq!(before, pod_bytes(&base.progs[0].insts));
        for cap in [None, Some(0), Some(2)] {
            assert!(require_dynamic_kv(&base.progs[0], cap).is_err());
        }
        for (field, value) in [(6, 0), (6, 4), (3, 7)] {
            let mut changed = blob(2);
            changed.progs[0].insts[0].i[field] = value;
            assert!(require_dynamic_kv(&changed.progs[0], Some(1)).is_err());
        }
    }

    #[test]
    #[ignore = "CPU actual packet proof; set TEST_CONTEXT_BASE and TEST_CONTEXT_AUX"]
    fn actual_frozen_auxiliary_packets() {
        let base =
            DevBlob::parse(&std::fs::read(std::env::var("TEST_CONTEXT_BASE").unwrap()).unwrap())
                .unwrap();
        let aux =
            DevBlob::parse(&std::fs::read(std::env::var("TEST_CONTEXT_AUX").unwrap()).unwrap())
                .unwrap();
        for index in base.decode_rung_lo()..base.progs.len() {
            let mut b = band(&[], &[]);
            b.base_program = index;
            b.program.index = index;
            b.rows = base.progs[index].t;
            auxiliary_program(&base, &aux, &b, coverage(), None).unwrap();
        }
    }
}
