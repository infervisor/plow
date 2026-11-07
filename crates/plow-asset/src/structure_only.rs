//! A packet compiled from header-only checkpoints: real programs, tensor table and contracts,
//! zero-filled initializers. Inspectable, never executable.

use packet::devbuild::{SectionData, SECT_METADATA};

pub const SECTION: &str = "structure_only.json";

pub fn section() -> SectionData {
    SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: br#"{"version":1,"reason":"compiled from header-only checkpoints; initializers are zero"}"#
            .to_vec(),
    }
}
