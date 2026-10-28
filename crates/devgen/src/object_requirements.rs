//! OBJECT REQUIREMENTS section (`plow_asset::object_requirements`): for every interpreter module
//! the packet executes, the exported symbols its programs need, from the emitted opcode streams.
//! The runtime checks the lists and nothing else; which arms a program needs is decided here,
//! from the same facts as the build-manifest define requirements (`manifest::backend_nvcc`).
use packet::dev::DevOp;
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::decode_objects::DecodeObjects;
use plow_asset::object_requirements::{self as req, Manifest, Requirement};

struct Facts {
    /// Runs the FP8 opcode family.
    fp8_arms: bool,
    gdn: bool,
    /// Runs the full-tile FP8 GEMM, the opcode the route-specific arms serve.
    fp8_gemm: bool,
    rows: u32,
}

fn facts<'a>(m: &Model, programs: impl Iterator<Item = &'a usize>) -> Facts {
    let mut f = Facts {
        fp8_arms: false,
        gdn: false,
        fp8_gemm: false,
        rows: 0,
    };
    for &index in programs {
        f.rows = f.rows.max(packet::devbuild::program_rows(m.prog_t[index]));
        for inst in &m.progs[index].insts {
            f.fp8_arms |= req::fp8_arm_op(inst.op);
            f.gdn |= req::gdn_op(inst.op);
            f.fp8_gemm |= inst.op == DevOp::GemmFp8 as u16;
        }
    }
    f
}

/// Packet-wide routes: W8A8 (`backend_nvcc` defines the arms for every object) and, with the
/// GDN chain, the per-token activation-scale route.
struct Routes {
    w8a8: bool,
    vllm: bool,
}

fn routes(m: &Model) -> Routes {
    let w8a8 = m
        .progs
        .iter()
        .flat_map(|p| &p.insts)
        .any(|i| req::w8a8_only_op(i.op));
    let gdn = facts(m, (0..m.progs.len()).collect::<Vec<_>>().iter()).gdn;
    Routes {
        w8a8,
        vllm: w8a8 && gdn,
    }
}

/// Prefill row counts the W8A8 route's TMA GEMM epilogue was qualified on; the route's symbol
/// attests the arm, not the bucket, so the compiler refuses any other bucket.
pub(crate) const W8A8_VLLM_PREFILL_ROWS: [u32; 4] = [128, 1024, 4096, 8192];

fn prefill_list(m: &Model, programs: &[usize], routes: &Routes) -> Vec<Requirement> {
    let f = facts(m, programs.iter());
    let mut out = Vec::new();
    if f.gdn {
        out.push(Requirement::new(req::GDN_PREFILL_ABI, 1));
    }
    if routes.w8a8 && f.fp8_arms {
        out.push(Requirement::new(req::W8A8_ARMS, 1));
    }
    if routes.vllm && f.fp8_gemm {
        for &index in programs {
            let rows = packet::devbuild::program_rows(m.prog_t[index]);
            assert!(
                W8A8_VLLM_PREFILL_ROWS.contains(&rows),
                "W8A8 prefill bucket {rows} is outside the route's qualified buckets {W8A8_VLLM_PREFILL_ROWS:?}"
            );
        }
        out.push(Requirement::new(req::W8A8_VLLM_PREFILL_ABI, 1));
    }
    out
}

fn decode_list(m: &Model, programs: &[usize], routes: &Routes) -> Vec<Requirement> {
    let f = facts(m, programs.iter());
    let mut out = Vec::new();
    if routes.w8a8 && f.fp8_arms {
        out.push(Requirement::new(req::W8A8_ARMS, 1));
    }
    if routes.vllm && f.fp8_gemm {
        out.push(Requirement::new(req::W8A8_VLLM_DECODE_ABI, 1));
        if f.rows > 1 {
            assert!(
                f.rows <= crate::qwen35::W8A8_DECODE_ROWS,
                "W8A8 decode rung {} exceeds the object's {} rows",
                f.rows,
                crate::qwen35::W8A8_DECODE_ROWS
            );
            out.push(Requirement::new(
                req::W8A8_VLLM_DECODE_ROWS,
                crate::qwen35::W8A8_DECODE_ROWS,
            ));
        }
    }
    out
}

/// The section for `m`, given the `decode_objects` metadata already emitted for it (`None`
/// when every decode rung runs on the main module).
pub(crate) fn manifest(m: &Model, decode_objects: Option<&DecodeObjects>) -> Manifest {
    let routes = routes(m);
    let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
    let prefill: Vec<usize> = (0..dec_lo).collect();
    let decode: Vec<usize> = (dec_lo..m.progs.len()).collect();
    let mut out = Manifest::new();
    out.prefill = prefill_list(m, &prefill, &routes);
    if let Some(objects) = decode_objects {
        let main = objects
            .programs
            .last()
            .expect("validated decode objects")
            .object;
        for &id in objects.objects.keys() {
            let bound: Vec<usize> = objects
                .programs
                .iter()
                .filter(|p| p.object == id)
                .map(|p| p.index)
                .collect();
            let list = decode_list(m, &bound, &routes);
            if id == main {
                out.decode = list.clone();
            }
            out.objects.insert(id, list);
        }
    } else {
        out.decode = decode_list(m, &decode, &routes);
    }
    out
}

/// Append the section; the `decode_objects` section, when present, must already be in
/// `sections`.
pub(crate) fn append(m: &Model, sections: &mut Vec<SectionData>) -> Result<(), String> {
    if sections
        .iter()
        .any(|s| s.kind == SECT_METADATA && s.name == req::SECTION)
    {
        return Err("duplicate object requirements".into());
    }
    let decode_objects = sections
        .iter()
        .find(|s| s.kind == SECT_METADATA && s.name == plow_asset::decode_objects::SECTION)
        .map(|s| serde_json::from_slice::<DecodeObjects>(&s.data).map_err(|e| e.to_string()))
        .transpose()?;
    let manifest = manifest(m, decode_objects.as_ref());
    let ids: Vec<u32> = decode_objects
        .as_ref()
        .map(|d| d.objects.keys().copied().collect())
        .unwrap_or_default();
    manifest.validate(&ids)?;
    sections.push(SectionData {
        kind: SECT_METADATA,
        name: req::SECTION.into(),
        data: serde_json::to_vec(&manifest).map_err(|e| e.to_string())?,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::Builder;

    fn model(prefill_ops: &[DevOp], decode_ops: &[DevOp], rungs: &[u32]) -> Model {
        model_with_bucket(128, prefill_ops, decode_ops, rungs)
    }

    fn model_with_bucket(
        bucket: u32,
        prefill_ops: &[DevOp],
        decode_ops: &[DevOp],
        rungs: &[u32],
    ) -> Model {
        let mut progs = Vec::new();
        let mut prog_t = Vec::new();
        for (ops, t) in [(prefill_ops, bucket)]
            .into_iter()
            .chain(rungs.iter().map(|&r| (decode_ops, r)))
        {
            let mut b = Builder::new(2);
            b.force_uniseg();
            let mut prev = None;
            for &op in ops {
                let deps: Vec<u32> = prev.into_iter().collect();
                prev = Some(b.emit(op, b.all(), &deps, |d| {
                    d.i[0] = 1;
                    d.i[1] = 1;
                    d.i[2] = 1;
                }));
            }
            if prev.is_none() {
                b.emit(DevOp::Nop, b.all(), &[], |_| {});
            }
            progs.push(b.finish());
            prog_t.push(t);
        }
        Model {
            n_cu: 2,
            target: 0,
            tensors: vec![],
            progs,
            kv_row_insts: vec![],
            prog_t,
            gen: vec![],
        }
    }

    #[test]
    fn dense_bf16_packet_requires_nothing() {
        let m = model(&[DevOp::Gemm], &[DevOp::Gemv], &[1, 2]);
        let r = manifest(&m, None);
        assert!(r.prefill.is_empty() && r.decode.is_empty() && r.objects.is_empty());
        let mut sections = Vec::new();
        append(&m, &mut sections).unwrap();
        assert_eq!(sections.len(), 1);
        assert!(append(&m, &mut sections).unwrap_err().contains("duplicate"));
    }

    #[test]
    fn w8a16_packet_requires_nothing() {
        let m = model(
            &[DevOp::GemmFp8],
            &[DevOp::GemvFp8, DevOp::GemmFp8],
            &[1, 4],
        );
        let r = manifest(&m, None);
        assert!(r.prefill.is_empty() && r.decode.is_empty());
    }

    #[test]
    fn dense_w8a8_packet_requires_the_arms_where_they_run() {
        let m = model(
            &[DevOp::QuantFp8, DevOp::GemmFp8],
            &[DevOp::GemvFp8, DevOp::GemmFp8],
            &[1, 4],
        );
        let r = manifest(&m, None);
        assert_eq!(r.prefill, vec![Requirement::new(req::W8A8_ARMS, 1)]);
        assert_eq!(r.decode, vec![Requirement::new(req::W8A8_ARMS, 1)]);
        let gemv_only = model(&[DevOp::QuantFp8, DevOp::GemmFp8], &[DevOp::GemvFp8], &[1]);
        assert!(manifest(&gemv_only, None).decode.is_empty());
    }

    #[test]
    fn gdn_w8a8_packet_requires_the_route_and_its_rows() {
        let m = model(
            &[
                DevOp::QwenRmsNorm,
                DevOp::QuantFp8,
                DevOp::GemmFp8,
                DevOp::QwenGdnPrefill,
            ],
            &[
                DevOp::QwenRmsNorm,
                DevOp::QuantFp8,
                DevOp::GemmFp8,
                DevOp::QwenGdnStep,
            ],
            &[1, 8],
        );
        let r = manifest(&m, None);
        assert_eq!(
            r.prefill,
            vec![
                Requirement::new(req::GDN_PREFILL_ABI, 1),
                Requirement::new(req::W8A8_ARMS, 1),
                Requirement::new(req::W8A8_VLLM_PREFILL_ABI, 1),
            ]
        );
        assert_eq!(
            r.decode,
            vec![
                Requirement::new(req::W8A8_ARMS, 1),
                Requirement::new(req::W8A8_VLLM_DECODE_ABI, 1),
                Requirement::new(req::W8A8_VLLM_DECODE_ROWS, 128),
            ]
        );
        // A single-row ladder needs no rows ceiling.
        let one = model(
            &[DevOp::QwenRmsNorm, DevOp::QuantFp8, DevOp::GemmFp8],
            &[
                DevOp::QwenRmsNorm,
                DevOp::QuantFp8,
                DevOp::GemmFp8,
                DevOp::QwenGdnStep,
            ],
            &[1],
        );
        assert_eq!(manifest(&one, None).decode.len(), 2);
    }

    #[test]
    #[should_panic(expected = "outside the route's qualified buckets")]
    fn w8a8_route_refuses_an_unqualified_prefill_bucket() {
        let m = model_with_bucket(
            256,
            &[DevOp::QwenRmsNorm, DevOp::QuantFp8, DevOp::GemmFp8],
            &[DevOp::QwenRmsNorm, DevOp::QuantFp8, DevOp::GemmFp8],
            &[1],
        );
        manifest(&m, None);
    }

    #[test]
    fn gdn_bf16_packet_requires_only_the_prefill_arms() {
        let m = model(
            &[DevOp::QwenRmsNorm, DevOp::Gemm, DevOp::QwenGdnPrefill],
            &[DevOp::QwenRmsNorm, DevOp::Gemv, DevOp::QwenGdnStep],
            &[1, 2],
        );
        let r = manifest(&m, None);
        assert_eq!(r.prefill, vec![Requirement::new(req::GDN_PREFILL_ABI, 1)]);
        assert!(r.decode.is_empty());
    }

    #[test]
    fn ladder_objects_get_their_own_lists() {
        use plow_asset::decode_objects::{DecodeObject, DecodeProgramObject};
        let m = model(
            &[DevOp::QwenRmsNorm, DevOp::QuantFp8, DevOp::GemmFp8],
            &[
                DevOp::QwenRmsNorm,
                DevOp::QuantFp8,
                DevOp::GemmFp8,
                DevOp::QwenGdnStep,
            ],
            &[1, 8],
        );
        let object = |file: &str| DecodeObject {
            file: file.into(),
            sha256: String::new(),
            profile: "sm90a".into(),
            entry: "_Z12interp_sm90a11PlowProgram".into(),
            threads: 256,
            arena_bytes: 1,
            grid: 2,
        };
        let objects = DecodeObjects {
            version: 1,
            kernarg_bytes: 0,
            objects: [(0, object("main.cubin")), (1, object("wide.cubin"))].into(),
            programs: vec![
                DecodeProgramObject {
                    index: 1,
                    rows: 1,
                    object: 1,
                },
                DecodeProgramObject {
                    index: 2,
                    rows: 8,
                    object: 0,
                },
            ],
        };
        let r = manifest(&m, Some(&objects));
        assert_eq!(r.objects[&1].len(), 2, "rung 1 alone needs no rows ceiling");
        assert_eq!(r.objects[&0].len(), 3);
        assert_eq!(
            r.decode, r.objects[&0],
            "the main module is the last binding's object"
        );
        assert!(r.validate(&[0, 1]).is_ok());
    }
}
