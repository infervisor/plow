//! The packet's `abi.json` (`plow_asset::abi`). Appended after every other section, so the
//! sections before it keep their bytes and offsets.
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::abi::{Abi, Bindings, Target, MUST_UNDERSTAND, SECTION, VERSION};
use plow_asset::speech_flag_layout;

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
    // Regular programs and every auxiliary-program section (depth, draft, ...).
    let aux: Vec<_> = sections
        .iter()
        .filter(|s| s.kind == SECT_METADATA && speech_flag_layout::is_aux_programs(&s.data))
        .flat_map(|s| {
            speech_flag_layout::aux_insts(&s.data, m.n_cu)
                .unwrap_or_else(|e| panic!("section {}: {e}", s.name))
        })
        .collect();
    let required = speech_flag_layout::required_layout(
        m.progs
            .iter()
            .flat_map(|p| &p.insts)
            .map(|d| (d.op, d.j[1], d.i))
            .chain(aux.iter().map(|d| (d.op, d.fj[2], d.i))),
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
        speech_flag_layout: required.map(|v| v.max(speech_flag_layout::DEFAULT_LAYOUT)),
    };
    sections.push(SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&abi).expect("serialize abi.json"),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::DevOp;
    use packet::devbuild::{Builder, Model, TensorDecl};

    fn abi_json(conv_flags: u32, aux_row_view: Option<bool>) -> String {
        let model = |emit: &dyn Fn(&mut Builder)| {
            let mut b = Builder::new(4);
            b.force_uniseg();
            emit(&mut b);
            Model {
                n_cu: 4,
                target: 0,
                tensors: vec![TensorDecl {
                    name: "x".into(),
                    bytes: 8,
                    init: None,
                }],
                progs: vec![b.finish()],
                kv_row_insts: vec![],
                prog_t: vec![1],
                gen: vec![],
            }
        };
        let m = model(&|b| {
            b.emit(DevOp::Conv1dF32, b.all(), &[], |i| i.j[1] = conv_flags);
        });
        let mut sections = Vec::new();
        if let Some(view) = aux_row_view {
            let aux = model(&|b| {
                b.emit(DevOp::GemvF32Out, b.all(), &[], |i| {
                    i.t = [0; 8];
                    i.i[7] = 2048 | if view { 4096 } else { 0 };
                });
            });
            sections.push(SectionData {
                kind: SECT_METADATA,
                name: plow_asset::fish_depth::PROGRAMS_SECTION.into(),
                data: plow_asset::program::with_model(&aux, |p| {
                    plow_asset::aux_program::encode(4, 1, &[p.programs[0]]).unwrap()
                }),
            });
        }
        append(&m, &mut sections, "sm_90a", "");
        let abi = sections.last().unwrap();
        assert_eq!(abi.name, SECTION);
        String::from_utf8(abi.data.clone()).unwrap()
    }

    /// A packet on the layout-0 subset writes the `abi.json` it always wrote; a protected bit in a
    /// regular or an auxiliary program adds the declaration.
    #[test]
    fn speech_flag_layout_is_declared_only_when_a_protected_bit_is_set() {
        let key = "speech_flag_layout";
        let base = abi_json(0x6e000, None);
        // The bytes the serializer wrote for this model before the field existed.
        assert_eq!(
            base,
            r#"{"version":1,"contract":2,"must_understand":[],"target":{"isa":"sm_90a","n_cu":4},"bindings":{"roles":{},"kv_writes":[]}}"#
        );
        assert_eq!(abi_json(0x6e000, Some(false)), base);
        let declared = format!(
            "{},\"{key}\":{}}}",
            &base[..base.len() - 1],
            speech_flag_layout::DEFAULT_LAYOUT
        );
        for flags in [
            0x6e000 | 1 << 3,
            0x6e000 | 1 << 21,
            0x6e000 | 1 << 30,
            0x6e004,
        ] {
            assert_eq!(abi_json(flags, None), declared, "{flags:#x}");
        }
        assert_eq!(abi_json(0x6e000, Some(true)), declared);
        let parsed: Abi = serde_json::from_str(&declared).unwrap();
        assert_eq!(
            parsed.speech_flag_layout,
            Some(speech_flag_layout::DEFAULT_LAYOUT)
        );
        let parsed: Abi = serde_json::from_str(&base).unwrap();
        assert_eq!(parsed.speech_flag_layout, None);
    }
}
