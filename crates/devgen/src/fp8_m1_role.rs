use packet::devbuild::{Model, SectionData, SECT_METADATA};
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct Selection {
    pub program: usize,
    pub pcs: Vec<usize>,
    pub file: String,
    pub sha256: String,
    pub promote_k512: u32,
}

pub fn apply(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    selection: Option<&Selection>,
) -> Result<(), String> {
    let Some(sel) = selection else {
        return Ok(());
    };
    if sel.pcs.is_empty()
        || sel.pcs.iter().copied().collect::<BTreeSet<_>>().len() != sel.pcs.len()
        || sel.promote_k512 > 1
        || sel.sha256.len() != 64
        || !sel
            .sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        || sel.file.is_empty()
        || std::path::Path::new(&sel.file)
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("invalid FP8 M1 role selection".into());
    }
    plow_asset::program::with_model(model, |p| {
        for &pc in &sel.pcs {
            plow_asset::fp8_m1_role::validate(p, sel.program, pc)?;
        }
        Ok::<_, String>(())
    })?;
    let p = &model.progs[sel.program];
    if p.hier_base != 0
        || p.gq_seg_ofs.len() != 2
        || p.stream.iter().chain(&p.gq_stream).any(|e| e.seg != 0)
    {
        return Err("FP8 M1 role requires an unsegmented input program".into());
    }
    let mut metadata = serde_json::json!({"version":1,"objects":{},"programs":[]});
    let matches: Vec<_> = sections
        .iter()
        .enumerate()
        .filter(|(_, s)| s.name == plow_asset::segment_roles::SECTION)
        .map(|(i, _)| i)
        .collect();
    if matches.len() > 1
        || matches
            .first()
            .is_some_and(|&i| sections[i].kind != SECT_METADATA)
    {
        return Err("duplicate segment role metadata".into());
    }
    if let Some(&i) = matches.first() {
        let existing = plow_asset::segment_roles::SegmentRoles::from_bytes(&sections[i].data)?;
        for program in &existing.programs {
            let prior = model
                .progs
                .get(program.index)
                .ok_or("existing role program out of bounds")?;
            if program.roles.len() + 1 != prior.gq_seg_ofs.len() {
                return Err("existing role program window coverage".into());
            }
        }
        metadata = serde_json::to_value(existing).map_err(|e| e.to_string())?;
        if metadata["version"] != 1
            || !metadata["objects"].is_object()
            || !metadata["programs"].is_array()
            || metadata["objects"].get("4").is_some()
            || metadata["programs"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["index"] == sel.program)
        {
            return Err("conflicting FP8 M1 role metadata".into());
        }
    }
    let mut roles = Vec::new();
    let mut segs = Vec::new();
    for pc in 0..p.insts.len() {
        let role = if sel.pcs.contains(&pc) { 4 } else { 0 };
        if roles.is_empty() || role != 0 || roles.last() != Some(&0) {
            roles.push(role);
        }
        segs.push(u16::try_from(roles.len() - 1).map_err(|_| "too many FP8 role segments")?);
    }
    let p = &mut model.progs[sel.program];
    for e in p.stream.iter_mut().chain(&mut p.gq_stream) {
        e.seg = segs[e.inst as usize];
    }
    p.gq_seg_ofs = vec![0];
    for (i, pair) in p.gq_stream.windows(2).enumerate() {
        if pair[0].seg != pair[1].seg {
            p.gq_seg_ofs.push((i + 1) as u32);
        }
    }
    p.gq_seg_ofs.push(p.gq_stream.len() as u32);
    metadata["objects"]["4"] = serde_json::json!({"abi":plow_asset::fp8_m1_role::ABI,"file":sel.file,"sha256":sel.sha256,"promote_k512":sel.promote_k512});
    metadata["programs"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"index":sel.program,"roles":roles}));
    let section = SectionData {
        kind: SECT_METADATA,
        name: "segment_roles.json".into(),
        data: serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    };
    if let Some(&i) = matches.first() {
        sections[i] = section;
    } else {
        sections.push(section);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> packet::devbuild::Model {
        use packet::dev::{DevOp, TENSOR_NONE};
        use packet::devbuild::{Builder, Model};
        let mut b = Builder::new(132);
        b.force_uniseg();
        let out = b.tensor("act.out", 20480);
        let a = b.tensor("act.fp8", 5120);
        let w = b.tensor("fp8/weight", 10240 * 5120);
        let scale = b.tensor("act.scale", 4);
        let ws = b.tensor("fp8/weight_scale", 10240 * 4);
        let norm = b.tensor("act.norm", 10240);
        let input = b.tensor("act.input", 10240);
        let map = b.tensor_gen(
            "tmap.weight",
            128,
            packet::rope::GenTensor::tmap_e4m3(w, 10240, 5120, 64),
        );
        let z = b.emit(DevOp::Nop, vec![0], &[], |_| {});
        let n = b.emit(DevOp::QwenRmsNorm, vec![0], &[z], |d| {
            d.t[0] = norm;
            d.t[1] = input;
            d.i[0] = 1;
            d.i[1] = 5120;
        });
        let q = b.emit(DevOp::QuantFp8, vec![0], &[n], |d| {
            d.t = [
                a,
                norm,
                scale,
                TENSOR_NONE,
                TENSOR_NONE,
                TENSOR_NONE,
                TENSOR_NONE,
                TENSOR_NONE,
            ];
            d.i[0] = 1;
            d.i[1] = 5120;
        });
        let g = b.emit(DevOp::GemmFp8, b.all(), &[q], |d| {
            d.t[..5].copy_from_slice(&[out, a, w, scale, ws]);
            d.i = [1, 10240, 5120, 0, 0, 0, 0, map];
        });
        b.emit(DevOp::Nop, vec![0], &[g], |_| {});
        let gen = b.gen_tensors();
        let p = b.finish();
        Model {
            n_cu: 132,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen,
        }
    }

    fn selection() -> Selection {
        Selection {
            program: 0,
            pcs: vec![3],
            file: "role.cubin".into(),
            sha256: "a".repeat(64),
            promote_k512: 1,
        }
    }
    #[test]
    fn absent_selection_is_byte_identical_and_roles_preserve_work() {
        let mut model = fixture();
        let before = model.to_blob();
        let mut sections = vec![];
        apply(&mut model, &mut sections, None).unwrap();
        assert_eq!(before, model.to_blob());
        assert!(sections.is_empty());
        let old = fixture();
        apply(&mut model, &mut sections, Some(&selection())).unwrap();
        let p = &model.progs[0];
        let q = &old.progs[0];
        assert_eq!(p.insts, q.insts);
        assert_eq!(p.waits, q.waits);
        assert_eq!(p.succs, q.succs);
        assert_eq!(p.n_counter, q.n_counter);
        assert_eq!(p.stream_ofs, q.stream_ofs);
        assert_eq!(p.stream_len, q.stream_len);
        assert_eq!(model.gen, old.gen);
        for (a, b) in p
            .stream
            .iter()
            .zip(&q.stream)
            .chain(p.gq_stream.iter().zip(&q.gq_stream))
        {
            let mut a = *a;
            a.seg = 0;
            assert_eq!(&a, b);
        }
        assert_eq!(p.gq_seg_ofs, [0, 3, 135, 136]);
        let meta: serde_json::Value = serde_json::from_slice(&sections[0].data).unwrap();
        assert_eq!(meta["programs"][0]["roles"], serde_json::json!([0, 4, 0]));
        assert!(apply(&mut model, &mut sections, Some(&selection())).is_err());
    }
    #[test]
    fn invalid_selections_leave_original_packet() {
        let mutations: &[fn(&mut Selection)] = &[
            |s| s.pcs.push(3),
            |s| s.pcs = vec![2],
            |s| s.program = 1,
            |s| s.promote_k512 = 2,
            |s| s.sha256 = "bad".into(),
            |s| s.file = "../bad".into(),
        ];
        for mutate in mutations {
            let mut m = fixture();
            let bytes = m.to_blob();
            let mut s = selection();
            mutate(&mut s);
            assert!(apply(&mut m, &mut vec![], Some(&s)).is_err());
            assert_eq!(bytes, m.to_blob());
        }
    }
    #[test]
    fn existing_role_metadata_rejects_reserved_names_and_malformed_schema_before_mutation() {
        let obj = r#"{"abi":"fp8_gemm_tma128_v1","file":"role.cubin"}"#;
        let valid = format!(
            r#"{{"version":1,"objects":{{"1":{obj}}},"programs":[{{"index":0,"roles":[1]}}]}}"#
        );
        let mut cases = vec![
            valid.replace("\"version\":1", "\"version\":1,\"extra\":0"),
            valid.replace("\"file\":", "\"extra\":0,\"file\":"),
            valid.replace("\"index\":0", "\"index\":0,\"extra\":0"),
        ];
        for key in ["1", "01"] {
            cases.push(valid.replace(
                &format!(r#""1":{obj}"#),
                &format!(r#""1":{obj},"{key}":{obj}"#),
            ));
        }
        for data in cases {
            let mut model = fixture();
            let mut sections = vec![SectionData {
                kind: SECT_METADATA,
                name: "segment_roles.json".into(),
                data: data.into_bytes(),
            }];
            let before = model.to_blob_v6(&sections);
            let error = apply(&mut model, &mut sections, Some(&selection())).unwrap_err();
            assert!(!error.contains("conflicting FP8"), "{error}");
            assert_eq!(model.to_blob_v6(&sections), before);
        }
        for kinds in [
            vec![123],
            vec![SECT_METADATA, SECT_METADATA],
            vec![SECT_METADATA, 123],
        ] {
            let mut model = fixture();
            let mut sections: Vec<_> = kinds
                .into_iter()
                .map(|kind| SectionData {
                    kind,
                    name: "segment_roles.json".into(),
                    data: valid.as_bytes().to_vec(),
                })
                .collect();
            let before = model.to_blob_v6(&sections);
            assert!(apply(&mut model, &mut sections, Some(&selection()))
                .unwrap_err()
                .contains("duplicate segment role metadata"));
            assert_eq!(model.to_blob_v6(&sections), before);
        }
    }
}
