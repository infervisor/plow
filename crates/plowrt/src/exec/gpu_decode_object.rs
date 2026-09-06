use super::*;
use plow_asset::decode_coverage::{DenseBf16, SYMBOLS};
use plow_asset::decode_objects::{DecodeObject, DecodeObjects, SECTION};
use std::collections::BTreeMap;

pub(super) struct DecodeModule {
    be: Arc<CudaBackend>,
    module: Module,
}
impl DecodeModule {
    pub(super) fn load(be: &Arc<CudaBackend>, image: &[u8]) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            be: Arc::clone(be),
            module: be.module_load(image)?,
        }))
    }
}
impl std::ops::Deref for DecodeModule {
    type Target = Module;
    fn deref(&self) -> &Module {
        &self.module
    }
}
impl Drop for DecodeModule {
    fn drop(&mut self) {
        if let Err(error) = self.be.module_unload(&self.module) {
            tracing::warn!(%error, "unload owned decode object failed");
        }
    }
}
pub(super) struct BoundDecodeObject {
    pub(super) function: KernelFn,
    pub(super) grid: u32,
    pub(super) block: u32,
    pub(super) smem: u32,
    _module: Arc<DecodeModule>,
}
fn reject(message: &str) -> RuntimeError {
    RuntimeError::Rejected(format!("decode objects: {message}"))
}
pub(super) fn parse(blob: &DevBlob, raw: &[u8]) -> Result<Option<DecodeObjects>> {
    decode_context::metadata(blob, raw)?;
    let count = blob
        .sections
        .iter()
        .filter(|s| s.kind == packet::devbuild::SECT_METADATA && s.name == SECTION)
        .count();
    if count == 0 {
        return Ok(None);
    }
    if count != 1 {
        return Err(reject("duplicate metadata"));
    }
    let bytes = blob
        .section_data_named(raw, packet::devbuild::SECT_METADATA, SECTION)
        .ok_or_else(|| reject("missing metadata bytes"))?;
    let metadata: DecodeObjects =
        serde_json::from_slice(bytes).map_err(|e| reject(&e.to_string()))?;
    let start = blob.prefill_progs().len();
    let programs: Vec<_> = blob
        .progs
        .iter()
        .enumerate()
        .skip(start)
        .map(|(i, g)| (i, g.t))
        .collect();
    metadata
        .validate(&programs, blob.n_cu, std::mem::size_of::<DevProgram>())
        .map_err(|e| reject(&e))?;
    if !validate_decode_ladder(blob)? || blob.tp.is_some() {
        return Err(reject("unqualified decode ladder"));
    }
    for g in blob.decode_progs() {
        g.check_coarse_single_segment()?;
        g.check_gq_topological()?;
        if g.insts.iter().any(|d| DevOp::from_u16(d.op).is_none()) {
            return Err(reject("unknown opcode in bound program"));
        }
    }
    Ok(Some(metadata))
}
pub(super) fn check_options(
    segmented_decode: bool,
    lt: bool,
    multistep: usize,
    override_present: bool,
) -> Result<()> {
    if segmented_decode || lt || multistep > 1 || override_present {
        return Err(reject("only plain cooperative decode without object/resource overrides, Lt, decode graph or multistep is supported"));
    }
    Ok(())
}
pub(super) fn image(
    spec: &DecodeObject,
    assets: &Path,
    profile: &InterpreterProfile,
    want_sm: u32,
) -> Result<InterpImage> {
    let path = assets.join(&spec.file);
    let image =
        read_cubin_candidate(&path).ok_or_else(|| reject("missing or invalid object file"))?;
    check_image(spec, &image, profile, want_sm)?;
    DenseBf16::from_image(&image).map_err(|e| reject(&e))?;
    Ok(InterpImage {
        image,
        entry: spec.entry.clone(),
        source: format!("packet decode object {}", path.display()),
    })
}
fn check_image(
    spec: &DecodeObject,
    image: &[u8],
    profile: &InterpreterProfile,
    want_sm: u32,
) -> Result<()> {
    if spec.profile != profile.tag
        || spec.entry != profile.decode_symbol
        || !spec.matches_image(image)
    {
        return Err(reject("object hash/profile/entry mismatch"));
    }
    let info = cubin::inspect(image).ok_or_else(|| reject("invalid ELF"))?;
    if info.sm != want_sm || !info.entries.contains(&spec.entry) {
        return Err(reject("object ISA or exported entry mismatch"));
    }
    Ok(())
}
fn exports(
    spec: &DecodeObject,
    block: Option<u32>,
    arena: Option<u32>,
    dyn_kvrow: Option<u32>,
    gq: Option<u32>,
    gemv: Option<u32>,
) -> Result<()> {
    if spec.threads != BLOCK
        || block != Some(spec.threads)
        || arena != Some(spec.arena_bytes)
        || dyn_kvrow != Some(1)
        || gq != Some(1)
        || !gemv.is_some_and(|n| n > 0)
    {
        return Err(reject(
            "exported threads/arena/dynamic-row/GQ/GEMV contract mismatch",
        ));
    }
    Ok(())
}
pub(super) fn capacity(spec: &DecodeObject, sms: u32, occupancy: u32) -> Result<()> {
    if spec.grid == 0
        || occupancy == 0
        || sms
            .checked_mul(occupancy)
            .is_none_or(|capacity| spec.grid > capacity)
    {
        return Err(reject("packet grid exceeds actual cooperative capacity"));
    }
    Ok(())
}
pub(super) fn bind_module(
    spec: &DecodeObject,
    be: &Arc<CudaBackend>,
    module: Arc<DecodeModule>,
) -> Result<Arc<BoundDecodeObject>> {
    let function = be.get_function(&module, &spec.entry)?;
    check_loaded(spec, be, &module, function)?;
    Ok(Arc::new(BoundDecodeObject {
        function,
        grid: spec.grid,
        block: spec.threads,
        smem: spec.arena_bytes,
        _module: module,
    }))
}

pub(super) fn initial_grid(
    metadata: Option<&DecodeObjects>,
    packet_grid: u32,
    sms: u32,
    occupancy: u32,
) -> Result<u32> {
    if let Some(metadata) = metadata {
        let object =
            &metadata.objects[&metadata.programs.last().expect("validated coverage").object];
        capacity(object, sms, occupancy)?;
        if object.grid != packet_grid {
            return Err(reject("bound main object differs from packet grid"));
        }
        Ok(object.grid)
    } else {
        let grid = occupancy * sms;
        if grid != packet_grid {
            return Err(RuntimeError::Device(format!(
                "interpreter grid {grid} ({occupancy}/SM × {sms} SMs) != packet n_cu {packet_grid} — recompile \
                 the packet with n_cu={grid}")));
        }
        Ok(grid)
    }
}
pub(super) fn check_loaded(
    spec: &DecodeObject,
    be: &Arc<CudaBackend>,
    module: &Module,
    function: KernelFn,
) -> Result<()> {
    exports(
        spec,
        be.module_global_u32(module, "plow_block")?,
        be.module_global_u32(module, "plow_arena_bytes")?,
        be.module_global_u32(module, "plow_dyn_kvrow")?,
        be.module_global_u32(module, "plow_segment_gq_abi")?,
        be.module_global_u32(module, "plow_gemv_mm_cap")?,
    )?;
    if spec.arena_bytes > 48 * 1024 {
        be.set_max_dynamic_smem(function, spec.arena_bytes)?;
    }
    capacity(
        spec,
        be.sm_count(),
        be.occupancy_blocks_per_sm(function, spec.threads, spec.arena_bytes as usize)?,
    )
}
fn check_program_coverage(
    metadata: &DecodeObjects,
    blob: &DevBlob,
    id: u32,
    coverage: DenseBf16,
    splitk: Option<u32>,
) -> Result<()> {
    coverage.validate().map_err(|e| reject(&e))?;
    for binding in metadata.programs.iter().filter(|p| p.object == id) {
        blob.with_packet_view(|packet| coverage.program(packet, binding.index, splitk))
            .map_err(|e| reject(&e))?;
    }
    Ok(())
}
pub(super) fn bind(
    metadata: &DecodeObjects,
    blob: &DevBlob,
    assets: &Path,
    be: &Arc<CudaBackend>,
    main_module: &Arc<DecodeModule>,
    main_function: KernelFn,
    profile: &InterpreterProfile,
    want_sm: u32,
) -> Result<BTreeMap<u32, Arc<BoundDecodeObject>>> {
    let main_id = metadata.programs.last().expect("validated coverage").object;
    let mut loaded = BTreeMap::new();
    for (&id, spec) in &metadata.objects {
        let (module, function) = if id == main_id {
            (Arc::clone(main_module), main_function)
        } else {
            let resolved = image(spec, assets, profile, want_sm)?;
            let module = DecodeModule::load(be, &resolved.image)?;
            let function = be.get_function(&module, &spec.entry)?;
            GpuEngine::check_packet_pairing(be, &module, assets)?;
            (module, function)
        };
        check_loaded(spec, be, &module, function)?;
        let mut constants = [0; 6];
        for (field, symbol) in constants.iter_mut().zip(SYMBOLS) {
            *field = be
                .module_global_u32(&module, symbol)?
                .ok_or_else(|| reject("missing loaded dense BF16 capability"))?;
        }
        let coverage = DenseBf16(constants);
        let needs_splitk = metadata
            .programs
            .iter()
            .filter(|p| p.object == id)
            .any(|p| {
                blob.progs[p.index].insts.iter().any(|d| {
                    matches!(
                        DevOp::from_u16(d.op),
                        Some(DevOp::ZeroF32 | DevOp::GemmSplitK | DevOp::CastF32Bf16)
                    )
                })
            });
        let splitk = if needs_splitk {
            be.module_global_u32(&module, "plow_gemm_splitk_abi")?
        } else {
            None
        };
        check_program_coverage(metadata, blob, id, coverage, splitk)?;
        for binding in metadata.programs.iter().filter(|p| p.object == id) {
            let program = &blob.progs[binding.index];
            if program.insts.iter().any(|d| d.op == DevOp::GemmFp8 as u16) {
                check_qwen_w8a8_capability(
                    false,
                    program.t,
                    be.module_global_u32(&module, "plow_fp8_m1_arm")?,
                )?;
            }
        }
        tracing::info!(object=id,file=%spec.file,threads=spec.threads,smem=spec.arena_bytes,grid=spec.grid,"packet decode object bound");
        loaded.insert(
            id,
            Arc::new(BoundDecodeObject {
                function,
                grid: spec.grid,
                block: spec.threads,
                smem: spec.arena_bytes,
                _module: module,
            }),
        );
    }
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::devblob::DevSection;
    use packet::devbuild::SECT_METADATA;
    use plow_asset::decode_objects::{image_sha256, DecodeProgramObject};
    fn model() -> DevBlob {
        super::super::decode_rung_tests::fixture()
    }
    fn spec() -> DecodeObject {
        DecodeObject {
            file: "old.cubin".into(),
            sha256: "a".repeat(64),
            profile: "sm90a".into(),
            entry: "_Z12interp_sm90a11PlowProgram".into(),
            threads: BLOCK,
            arena_bytes: 16384,
            grid: 7,
        }
    }
    fn metadata() -> DecodeObjects {
        let mut object = spec();
        object.grid = 1;
        DecodeObjects {
            version: 1,
            kernarg_bytes: std::mem::size_of::<DevProgram>(),
            objects: BTreeMap::from([(0, object)]),
            programs: [1, 2, 4, 8, 16]
                .iter()
                .enumerate()
                .map(|(index, &rows)| DecodeProgramObject {
                    index,
                    rows,
                    object: 0,
                })
                .collect(),
        }
    }
    fn attach(mut blob: DevBlob, metadata: &DecodeObjects, duplicate: bool) -> (DevBlob, Vec<u8>) {
        let raw = serde_json::to_vec(metadata).unwrap();
        let make = || DevSection {
            kind: SECT_METADATA,
            name: SECTION.into(),
            offset: 0,
            size: raw.len(),
        };
        blob.sections = if duplicate {
            vec![make(), make()]
        } else {
            vec![make()]
        };
        (blob, raw)
    }
    #[test]
    fn absent_metadata_preserves_legacy_and_present_requires_full_coverage() {
        assert!(parse(&model(), &[]).unwrap().is_none());
        let metadata = metadata();
        let (blob, raw) = attach(model(), &metadata, false);
        assert_eq!(parse(&blob, &raw).unwrap().unwrap().programs.len(), 5);
        let (blob, raw) = attach(model(), &metadata, true);
        assert!(parse(&blob, &raw).is_err());
        for index in [0, 1] {
            let mut bad = metadata.clone();
            bad.programs.remove(index);
            let (blob, raw) = attach(model(), &bad, false);
            assert!(parse(&blob, &raw).is_err());
        }
        let mut changed = model();
        changed.progs[1].insts[0].op = u16::MAX;
        let (blob, raw) = attach(changed, &metadata, false);
        assert!(parse(&blob, &raw).is_err());
        let mut changed = model();
        changed.progs[1].stream[0].flags |= packet::dev::SE_XCTR;
        let (blob, raw) = attach(changed, &metadata, false);
        assert!(parse(&blob, &raw).is_err());
    }
    #[test]
    fn context_metadata_cannot_be_silently_ignored_before_materialization() {
        for duplicate in [false, true] {
            let (mut blob, raw) = attach(model(), &metadata(), duplicate);
            for section in &mut blob.sections {
                section.name = plow_asset::decode_context::SECTION.into();
            }
            let error = parse(&blob, &raw).unwrap_err().to_string();
            assert!(error.contains("decode context") || error.contains("unknown field"));
            blob.sections[0].kind = 0;
            assert!(parse(&blob, &raw).is_err());
        }
        assert!(parse(&model(), &[]).unwrap().is_none());
    }

    #[test]
    fn resources_and_derived_cooperative_capacity_must_match() {
        let s = spec();
        assert!(exports(&s, Some(BLOCK), Some(16384), Some(1), Some(1), Some(8)).is_ok());
        for values in [
            (None, Some(16384), Some(1), Some(1), Some(8)),
            (Some(BLOCK), Some(32768), Some(1), Some(1), Some(8)),
            (Some(BLOCK), Some(16384), None, Some(1), Some(8)),
            (Some(BLOCK), Some(16384), Some(1), Some(0), Some(8)),
            (Some(BLOCK), Some(16384), Some(1), Some(1), Some(0)),
        ] {
            assert!(exports(&s, values.0, values.1, values.2, values.3, values.4).is_err());
        }
        let mut wrong = s.clone();
        wrong.threads = 512;
        assert!(exports(&wrong, Some(512), Some(16384), Some(1), Some(1), Some(8)).is_err());
        assert!(capacity(&s, 7, 1).is_ok());
        for (sms, occ) in [(7, 0), (0, 1), (6, 1), (u32::MAX, 2)] {
            assert!(capacity(&s, sms, occ).is_err());
        }
        assert!(capacity(&s, 7, 2).is_ok());
        assert!(capacity(&s, 8, 1).is_ok());
        let mut other = s;
        other.grid = 0;
        assert!(capacity(&other, 7, 1).is_err());
        other.grid = 120;
        assert!(capacity(&other, 60, 2).is_ok());
    }
    #[test]
    fn initial_module_selection_uses_packet_grid_with_spare_capacity() {
        let object = spec();
        let metadata = DecodeObjects {
            version: 1,
            kernarg_bytes: std::mem::size_of::<DevProgram>(),
            objects: BTreeMap::from([(0, object)]),
            programs: vec![
                DecodeProgramObject {
                    index: 0,
                    rows: 1,
                    object: 0,
                },
                DecodeProgramObject {
                    index: 1,
                    rows: 2,
                    object: 0,
                },
            ],
        };
        assert_eq!(initial_grid(Some(&metadata), 7, 7, 2).unwrap(), 7);
        assert_eq!(initial_grid(Some(&metadata), 7, 7, 1).unwrap(), 7);
        assert!(initial_grid(Some(&metadata), 14, 7, 2).is_err());
        for (sms, occ) in [(6, 1), (0, 1), (7, 0), (u32::MAX, 2)] {
            assert!(initial_grid(Some(&metadata), 7, sms, occ).is_err());
        }
        assert_eq!(initial_grid(None, 7, 7, 1).unwrap(), 7);
        assert!(initial_grid(None, 7, 7, 2).is_err());
    }
    #[test]
    fn incompatible_launch_modes_are_rejected() {
        assert!(check_options(false, false, 0, false).is_ok());
        for x in [
            (true, false, 0, false),
            (false, true, 0, false),
            (false, false, 2, false),
            (false, false, 0, true),
        ] {
            assert!(check_options(x.0, x.1, x.2, x.3).is_err());
        }
    }
    #[test]
    fn validates_every_assigned_program_against_compiled_coverage() {
        let mut blob = model();
        let metadata = metadata();
        let coverage = DenseBf16([1, 2, 8, 16384, 16, 16384]);
        assert!(check_program_coverage(&metadata, &blob, 0, coverage, None).is_ok());
        blob.progs[0].insts[2].i[6] = 128;
        assert!(check_program_coverage(&metadata, &blob, 0, coverage, None).is_err());
        blob.progs[0].insts[2].i[6] = 256;
        blob.progs[0].insts[2].op = DevOp::FlashDecodeFp8 as u16;
        assert!(check_program_coverage(&metadata, &blob, 0, coverage, None).is_err());
        assert!(check_program_coverage(
            &metadata,
            &model(),
            0,
            DenseBf16([0, 2, 8, 16384, 16, 16384]),
            None
        )
        .is_err());
    }
    fn elf(sm: u32, entry: &str) -> Vec<u8> {
        let mut e = vec![0u8; 217 + entry.len() + 1];
        e[..6].copy_from_slice(b"\x7fELF\x02\x01");
        e[0x30..0x34].copy_from_slice(&(sm << 8).to_le_bytes());
        e[0x28..0x30].copy_from_slice(&64u64.to_le_bytes());
        e[0x3a..0x3c].copy_from_slice(&64u16.to_le_bytes());
        e[0x3c..0x3e].copy_from_slice(&2u16.to_le_bytes());
        e[68..72].copy_from_slice(&2u32.to_le_bytes());
        e[88..96].copy_from_slice(&192u64.to_le_bytes());
        e[96..104].copy_from_slice(&24u64.to_le_bytes());
        e[104..108].copy_from_slice(&1u32.to_le_bytes());
        e[132..136].copy_from_slice(&3u32.to_le_bytes());
        e[152..160].copy_from_slice(&216u64.to_le_bytes());
        e[160..168].copy_from_slice(&((entry.len() + 2) as u64).to_le_bytes());
        e[192..196].copy_from_slice(&1u32.to_le_bytes());
        e[196] = 0x12;
        e[217..217 + entry.len()].copy_from_slice(entry.as_bytes());
        e
    }
    #[test]
    fn image_identity_isa_and_entry_are_checked_without_external_files() {
        for cc in [(9, 0), (12, 0)] {
            let profile = interpreter_profile(cc).unwrap();
            let mut s = spec();
            s.profile = profile.tag.into();
            s.entry = profile.decode_symbol.into();
            let image = elf(cc.0 * 10 + cc.1, &s.entry);
            s.sha256 = image_sha256(&image);
            assert!(check_image(&s, &image, &profile, cc.0 * 10 + cc.1).is_ok());
            assert!(check_image(&s, &image, &profile, 80).is_err());
            let mut corrupt = image.clone();
            corrupt[0] = 0;
            assert!(check_image(&s, &corrupt, &profile, cc.0 * 10 + cc.1).is_err());
            s.sha256 = image_sha256(&corrupt);
            assert!(check_image(&s, &corrupt, &profile, cc.0 * 10 + cc.1).is_err());
            let missing = elf(cc.0 * 10 + cc.1, "other");
            s.sha256 = image_sha256(&missing);
            assert!(check_image(&s, &missing, &profile, cc.0 * 10 + cc.1).is_err());
        }
    }
}
