use packet::devbuild::{Model, SectionData, SECT_METADATA, SECT_PROGRAMS};
use plow_asset::{aux_program, mixed_step, program};

type Result<T> = std::result::Result<T, String>;

const MAX_OBJECT_BYTES: usize = 64 * 1024 * 1024;
const EM_CUDA: u16 = 190;
const EM_AMDGPU: u16 = 224;

pub struct ProgramSection<'a> {
    pub section: &'a str,
    pub version: u32,
    pub n_cu: u32,
    pub programs: &'a [program::Program<'a>],
}

pub struct ObjectSection<'a> {
    pub section: &'a str,
    pub kind: mixed_step::PayloadKind,
    pub version: u32,
    pub n_cu: u32,
    pub capability: &'a str,
    pub capability_version: u32,
    pub bytes: &'a [u8],
}

pub struct Variant<'a> {
    pub rows: u32,
    pub decode_rows: u32,
    pub program_index: u32,
    pub object_indices: &'a [usize],
}

pub struct Spec<'a> {
    pub max_active_requests: u32,
    pub physical_slot_capacity: u32,
    pub programs: ProgramSection<'a>,
    pub objects: &'a [ObjectSection<'a>],
    pub variants: &'a [Variant<'a>],
}

fn object_binding(model: &Model, object: &ObjectSection<'_>) -> Result<mixed_step::PayloadBinding> {
    if !matches!(
        object.kind,
        mixed_step::PayloadKind::Cubin | mixed_step::PayloadKind::Hsaco
    ) || object.version != mixed_step::VERSION
        || object.n_cu != model.n_cu
        || object.capability != mixed_step::OBJECT_CAPABILITY
        || object.capability_version != mixed_step::VERSION
        || object.bytes.is_empty()
        || object.bytes.len() > MAX_OBJECT_BYTES
        || plow_asset::cubin::elf_machine(object.bytes)
            != Some(match object.kind {
                mixed_step::PayloadKind::Cubin => EM_CUDA,
                mixed_step::PayloadKind::Hsaco => EM_AMDGPU,
                mixed_step::PayloadKind::Programs => unreachable!(),
            })
        || plow_asset::cubin::global_u32(object.bytes, object.capability)
            != Some(object.capability_version)
    {
        return Err("mixed step emitter: invalid object descriptor or capability".into());
    }
    if object.kind == mixed_step::PayloadKind::Cubin
        && plow_asset::cubin::inspect(object.bytes).is_none()
    {
        return Err("mixed step emitter: invalid CUBIN".into());
    }
    Ok(mixed_step::PayloadBinding {
        section: object.section.into(),
        kind: object.kind,
        version: object.version,
        sha256: mixed_step::payload_sha256(object.bytes),
        capability: mixed_step::Capability {
            name: object.capability.into(),
            version: object.capability_version,
        },
    })
}

pub fn append(model: &Model, sections: &mut Vec<SectionData>, spec: &Spec<'_>) -> Result<()> {
    if spec.programs.version != aux_program::VERSION
        || spec.programs.n_cu != model.n_cu
        || spec.programs.programs.is_empty()
        || spec.variants.is_empty()
    {
        return Err("mixed step emitter: invalid program descriptor".into());
    }

    let program_bytes = aux_program::encode(
        spec.programs.n_cu,
        model.tensors.len(),
        spec.programs.programs,
    )?;
    let parsed = aux_program::parse(&program_bytes, model.n_cu, model.tensors.len())?;
    let program_binding = mixed_step::PayloadBinding {
        section: spec.programs.section.into(),
        kind: mixed_step::PayloadKind::Programs,
        version: spec.programs.version,
        sha256: mixed_step::payload_sha256(&program_bytes),
        capability: mixed_step::Capability {
            name: aux_program::CAPABILITY.into(),
            version: aux_program::VERSION,
        },
    };

    let object_bindings = spec
        .objects
        .iter()
        .map(|object| object_binding(model, object))
        .collect::<Result<Vec<_>>>()?;
    let mut identities = std::collections::BTreeSet::new();
    identities.insert((SECT_PROGRAMS, spec.programs.section));
    identities.insert((SECT_METADATA, mixed_step::SECTION));
    for object in spec.objects {
        if !identities.insert((object.kind.section_kind(), object.section)) {
            return Err("mixed step emitter: duplicate section identity".into());
        }
    }
    if sections
        .iter()
        .any(|existing| identities.contains(&(existing.kind, existing.name.as_str())))
    {
        return Err("mixed step emitter: section identity already exists".into());
    }

    let mut variants = Vec::with_capacity(spec.variants.len());
    let mut used_programs = vec![false; parsed.programs.len()];
    let mut used_objects = vec![false; object_bindings.len()];
    for variant in spec.variants {
        let program_index = usize::try_from(variant.program_index)
            .map_err(|_| "mixed step emitter: program index overflow")?;
        if parsed
            .programs
            .get(program_index)
            .is_none_or(|program| program.rows != variant.rows)
        {
            return Err("mixed step emitter: variant program or row mismatch".into());
        }
        used_programs[program_index] = true;
        let mut objects = Vec::with_capacity(variant.object_indices.len());
        let mut selected = std::collections::BTreeSet::new();
        for &index in variant.object_indices {
            if !selected.insert(index) {
                return Err("mixed step emitter: duplicate variant object".into());
            }
            let used = used_objects
                .get_mut(index)
                .ok_or("mixed step emitter: object index out of bounds")?;
            *used = true;
            objects.push(
                object_bindings
                    .get(index)
                    .ok_or("mixed step emitter: object index out of bounds")?
                    .clone(),
            );
        }
        variants.push(mixed_step::Variant {
            rows: variant.rows,
            decode_rows: variant.decode_rows,
            program: mixed_step::ProgramBinding {
                index: variant.program_index,
                payload: program_binding.clone(),
            },
            objects,
        });
    }
    if used_programs.iter().any(|used| !used) || used_objects.iter().any(|used| !used) {
        return Err("mixed step emitter: unbound payload section".into());
    }

    let manifest = mixed_step::Manifest {
        version: mixed_step::VERSION,
        n_cu: model.n_cu,
        max_active_requests: spec.max_active_requests,
        physical_slot_capacity: spec.physical_slot_capacity,
        variants,
    };
    manifest.validate()?;
    let program_payload = mixed_step::Payload {
        section: spec.programs.section,
        kind: mixed_step::PayloadKind::Programs,
        version: spec.programs.version,
        n_cu: spec.programs.n_cu,
        bytes: &program_bytes,
    };
    for variant in &manifest.variants {
        variant.bind_program(model.n_cu, model.tensors.len(), &program_payload)?;
    }
    for (variant, input) in manifest.variants.iter().zip(spec.variants) {
        for &index in input.object_indices {
            let object = &spec.objects[index];
            let payload = mixed_step::Payload {
                section: object.section,
                kind: object.kind,
                version: object.version,
                n_cu: object.n_cu,
                bytes: object.bytes,
            };
            let capability = |name: &str| plow_asset::cubin::global_u32(object.bytes, name);
            match object.kind {
                mixed_step::PayloadKind::Cubin => {
                    variant.bind_cubin_with(model.n_cu, &payload, capability)?
                }
                mixed_step::PayloadKind::Hsaco => {
                    variant.bind_hsaco_with(model.n_cu, &payload, capability)?
                }
                mixed_step::PayloadKind::Programs => unreachable!(),
            }
        }
    }

    let metadata = serde_json::to_vec(&manifest).map_err(|error| error.to_string())?;
    let mut pending = Vec::with_capacity(spec.objects.len() + 2);
    pending.push(SectionData {
        kind: SECT_PROGRAMS,
        name: spec.programs.section.into(),
        data: program_bytes,
    });
    pending.push(SectionData {
        kind: SECT_METADATA,
        name: mixed_step::SECTION.into(),
        data: metadata,
    });
    pending.extend(spec.objects.iter().map(|object| SectionData {
        kind: object.kind.section_kind(),
        name: object.section.into(),
        data: object.bytes.to_vec(),
    }));
    sections.extend(pending);
    Ok(())
}

#[cfg(test)]
#[path = "mixed_step_emit_tests.rs"]
mod tests;
