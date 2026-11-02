//! The packet's `abi.json` (`plow_asset::abi`). Appended after every other section, so the
//! sections before it keep their bytes and offsets.
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::abi::{Abi, Bindings, Target, MUST_UNDERSTAND, SECTION, VERSION};

/// `arch` and `gpu` are the emit's `--arch` and `--gpu`; an empty `arch` writes no target.
pub(crate) fn append(m: &Model, sections: &mut Vec<SectionData>, arch: &str, gpu: &str) {
    let must_understand = MUST_UNDERSTAND
        .iter()
        .filter(|n| {
            sections
                .iter()
                .any(|s| s.kind == SECT_METADATA && s.name == **n)
        })
        .map(|n| n.to_string())
        .collect();
    let contract = crate::dense_cublaslt::runtime_requires(m, sections)["plowrt_contract"]
        .as_u64()
        .and_then(|c| u32::try_from(c).ok())
        .expect("runtime_requires declares plowrt_contract");
    let names: Vec<String> = m.tensors.iter().map(|t| t.name.clone()).collect();
    let bindings = Bindings::derive(
        &names,
        m.progs
            .iter()
            .map(|p| p.insts.iter().map(|d| (d.op, d.t[0]))),
    );
    let abi = Abi {
        version: VERSION,
        contract,
        must_understand,
        target: (!arch.is_empty()).then(|| Target {
            isa: arch.to_string(),
            n_cu: m.n_cu,
            tuned_on: (!gpu.is_empty()).then(|| gpu.to_string()),
        }),
        bindings: Some(bindings),
    };
    sections.push(SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&abi).expect("serialize abi.json"),
    });
}
