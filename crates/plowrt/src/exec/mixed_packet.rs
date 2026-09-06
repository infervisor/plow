use packet::devbuild::SECT_METADATA;
use plow_asset::{aux_program, mixed_step};

use crate::{Result, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PacketSection<'a> {
    pub kind: u32,
    pub name: &'a str,
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectSection<'a> {
    pub kind: mixed_step::PayloadKind,
    pub name: &'a str,
    pub bytes: &'a [u8],
}

struct ProgramSection<'a> {
    name: &'a str,
    parsed: aux_program::Section,
}

#[derive(Clone, Copy)]
struct VariantEntry {
    rows: u32,
    decode_rows: u32,
    program_index: u32,
    program_section: usize,
    object_section: usize,
}

#[derive(Clone, Copy)]
pub struct SelectedVariant<'a> {
    rows: u32,
    decode_rows: u32,
    program_index: u32,
    program: &'a aux_program::Program,
    object: ObjectSection<'a>,
}

impl SelectedVariant<'_> {
    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn decode_rows(&self) -> u32 {
        self.decode_rows
    }

    pub fn program_index(&self) -> u32 {
        self.program_index
    }

    pub fn program(&self) -> &aux_program::Program {
        self.program
    }

    pub fn object(&self) -> ObjectSection<'_> {
        self.object
    }
}

pub struct LoadedMixedPacket<'a> {
    n_cu: u32,
    max_active_requests: u32,
    physical_slot_capacity: u32,
    backend: mixed_step::PayloadKind,
    programs: Vec<ProgramSection<'a>>,
    objects: Vec<ObjectSection<'a>>,
    variants: Vec<VariantEntry>,
}

impl LoadedMixedPacket<'_> {
    pub fn n_cu(&self) -> u32 {
        self.n_cu
    }

    pub fn max_active_requests(&self) -> u32 {
        self.max_active_requests
    }

    pub fn physical_slot_capacity(&self) -> u32 {
        self.physical_slot_capacity
    }

    pub fn backend(&self) -> mixed_step::PayloadKind {
        self.backend
    }

    /// Select a compiler-emitted variant. This is a linear scan over the small
    /// load-time catalog and does not allocate.
    pub fn select(&self, rows: u32, decode_rows: u32) -> Option<SelectedVariant<'_>> {
        self.variants
            .iter()
            .find(|variant| variant.rows == rows && variant.decode_rows == decode_rows)
            .map(|variant| SelectedVariant {
                rows: variant.rows,
                decode_rows: variant.decode_rows,
                program_index: variant.program_index,
                program: &self.programs[variant.program_section].parsed.programs
                    [variant.program_index as usize],
                object: self.objects[variant.object_section],
            })
    }
}

/// Bind mixed-step packet metadata to its auxiliary programs and one backend's
/// object sections. The callback must read the requested capability from the
/// actual object bytes (for example, through an ELF symbol reader).
pub fn load<'a>(
    sections: &'a [PacketSection<'a>],
    expected_n_cu: u32,
    tensor_count: usize,
    backend: mixed_step::PayloadKind,
    mut read_capability: impl FnMut(ObjectSection<'a>, &str) -> Option<u32>,
) -> Result<Option<LoadedMixedPacket<'a>>> {
    let mut metadata_sections = sections
        .iter()
        .filter(|section| section.name == mixed_step::SECTION);
    let Some(metadata) = metadata_sections.next() else {
        return Ok(None);
    };
    if metadata_sections.next().is_some() || metadata.kind != SECT_METADATA {
        return Err(reject("metadata requires exactly one metadata section"));
    }
    if !matches!(
        backend,
        mixed_step::PayloadKind::Cubin | mixed_step::PayloadKind::Hsaco
    ) {
        return Err(reject("backend payload kind"));
    }

    let manifest: mixed_step::Manifest = serde_json::from_slice(metadata.bytes)
        .map_err(|error| reject(format!("metadata: {error}")))?;
    manifest.validate().map_err(reject)?;
    if manifest.n_cu != expected_n_cu {
        return Err(reject("CU/grid mismatch"));
    }

    let mut program_catalog = Vec::<ProgramSection<'a>>::new();
    let mut object_catalog = Vec::<ObjectSection<'a>>::new();
    let mut variants = Vec::with_capacity(manifest.variants.len());
    for variant in &manifest.variants {
        let program_section = exact_section(
            sections,
            variant.program.payload.kind.section_kind(),
            &variant.program.payload.section,
        )?;
        let program_payload = payload(
            program_section,
            expected_n_cu,
            mixed_step::PayloadKind::Programs,
        );
        let program_section = if let Some(index) = program_catalog
            .iter()
            .position(|loaded| loaded.name == program_section.name)
        {
            let parsed = &program_catalog[index].parsed;
            if !parsed
                .programs
                .get(variant.program.index as usize)
                .is_some_and(|program| program.rows == variant.rows)
            {
                return Err(reject("auxiliary program index"));
            }
            index
        } else {
            let parsed = variant
                .bind_program(expected_n_cu, tensor_count, &program_payload)
                .map_err(reject)?;
            program_catalog.push(ProgramSection {
                name: program_section.name,
                parsed,
            });
            program_catalog.len() - 1
        };

        let mut object_choices = variant
            .objects
            .iter()
            .filter(|object| object.kind == backend);
        let object_binding = object_choices
            .next()
            .ok_or_else(|| reject("selected backend object is undeclared"))?;
        if object_choices.next().is_some() {
            return Err(reject("selected backend has more than one object choice"));
        }
        let object_section =
            exact_section(sections, backend.section_kind(), &object_binding.section)?;
        let object_section = if let Some(index) = object_catalog
            .iter()
            .position(|loaded| loaded.name == object_section.name && loaded.kind == backend)
        {
            index
        } else {
            let object = ObjectSection {
                kind: backend,
                name: object_section.name,
                bytes: object_section.bytes,
            };
            let object_payload = payload(object_section, expected_n_cu, backend);
            match backend {
                mixed_step::PayloadKind::Cubin => {
                    variant.bind_cubin_with(expected_n_cu, &object_payload, |name| {
                        read_capability(object, name)
                    })
                }
                mixed_step::PayloadKind::Hsaco => {
                    variant.bind_hsaco_with(expected_n_cu, &object_payload, |name| {
                        read_capability(object, name)
                    })
                }
                mixed_step::PayloadKind::Programs => unreachable!(),
            }
            .map_err(reject)?;
            object_catalog.push(object);
            object_catalog.len() - 1
        };

        variants.push(VariantEntry {
            rows: variant.rows,
            decode_rows: variant.decode_rows,
            program_index: variant.program.index,
            program_section,
            object_section,
        });
    }

    Ok(Some(LoadedMixedPacket {
        n_cu: manifest.n_cu,
        max_active_requests: manifest.max_active_requests,
        physical_slot_capacity: manifest.physical_slot_capacity,
        backend,
        programs: program_catalog,
        objects: object_catalog,
        variants,
    }))
}

fn exact_section<'a>(
    sections: &'a [PacketSection<'a>],
    kind: u32,
    name: &str,
) -> Result<PacketSection<'a>> {
    let mut named = sections.iter().filter(|section| section.name == name);
    let section = named
        .next()
        .copied()
        .ok_or_else(|| reject(format!("missing section {name}")))?;
    if named.next().is_some() {
        return Err(reject(format!("duplicate section {name}")));
    }
    if section.kind != kind {
        return Err(reject(format!("section {name} kind mismatch")));
    }
    Ok(section)
}

fn payload<'a>(
    section: PacketSection<'a>,
    n_cu: u32,
    kind: mixed_step::PayloadKind,
) -> mixed_step::Payload<'a> {
    mixed_step::Payload {
        section: section.name,
        kind,
        version: mixed_step::VERSION,
        n_cu,
        bytes: section.bytes,
    }
}

fn reject(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("mixed packet: {}", message.into()))
}

#[cfg(test)]
#[path = "mixed_packet_tests.rs"]
mod tests;
