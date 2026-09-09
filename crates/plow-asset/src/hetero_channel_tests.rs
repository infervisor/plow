use super::*;
use packet::devbuild::{Builder, Model};

fn fixture() -> (ChannelPlan, Model) {
    let (h, i, a) = (64, 96, 32);
    let mut b = Builder::new(1);
    let x = b.tensor("x", 128 * h * 2);
    let r = b.tensor("residual", 128 * h * 2);
    let z = b.tensor("z", 128 * i * 2);
    let y = b.tensor("y", 128 * h * 2);
    let gamma = b.tensor("gamma", h * 2);
    let wg = b.tensor("gate", i * h * 2);
    let wu = b.tensor("up", i * h * 2);
    let wd = b.tensor("down", h * i * 2);
    let norm = b.emit(DevOp::RmsNorm, vec![0], &[], |d| {
        d.t[..3].copy_from_slice(&[x, r, gamma]);
        d.i[..2].copy_from_slice(&[128, h as u32]);
        d.f[0] = 1e-6;
    });
    let gate = b.emit(DevOp::GemmGlu, vec![0], &[norm], |d| {
        d.t[..3].copy_from_slice(&[z, x, wg]);
        d.t[5] = wu;
        d.i[..3].copy_from_slice(&[128, i as u32, h as u32]);
        d.i[5] = 1;
    });
    let down = b.emit(DevOp::GemmMed, vec![0], &[gate], |d| {
        d.t[..3].copy_from_slice(&[y, z, wd]);
        d.i[..3].copy_from_slice(&[128, h as u32, i as u32]);
    });
    b.emit(DevOp::Residual, vec![0], &[down], |d| {
        d.t[..3].copy_from_slice(&[r, r, y]);
        d.i[0] = 128 * h as u32;
        d.f[0] = 1.0;
    });
    let tensors = b.tensors();
    let model = Model {
        n_cu: 1,
        target: 0,
        tensors,
        progs: vec![b.finish(), Builder::new(1).finish()],
        prog_t: vec![128, 1],
        kv_row_insts: vec![],
        gen: vec![],
    };
    let weight = |name: &str, down| Weight {
        tensor: name.into(),
        scale: None,
        rows: if down { h } else { i } as u32,
        cols: if down { i } else { h } as u32,
        ane: if down {
            Slice {
                rows: [0, h as u32],
                cols: [0, a],
            }
        } else {
            Slice {
                rows: [0, a],
                cols: [0, h as u32],
            }
        },
        gpu: if down {
            Slice {
                rows: [0, h as u32],
                cols: [a, i as u32],
            }
        } else {
            Slice {
                rows: [a, i as u32],
                cols: [0, h as u32],
            }
        },
    };
    let mut plan = ChannelPlan {
        schema: SCHEMA.into(),
        mode: Mode::ChannelMlp,
        arch: "metal3".into(),
        hidden: h as u32,
        inter: i as u32,
        ane_channels: a,
        weight_encoding: WeightEncoding::Bf16,
        layers: vec![Layer {
            layer: 0,
            gate: weight("gate", false),
            up: weight("up", false),
            down: weight("down", true),
        }],
        programs: vec![ProgPlan {
            prog: 0,
            rows: 128,
            min_rows: 64,
            max_rows: 128,
            call_rows: 128,
            original_sha256: String::new(),
            partials: Partials {
                gpu: "gpu_partial".into(),
                ane: "ane_partial".into(),
                rows: 128,
                cols: h as u32,
                dtype: PartialDtype::F32,
            },
            spans: vec![Span {
                layer: 0,
                insts: [1, 4],
                input: "x".into(),
                residual: "residual".into(),
                intermediate: "z".into(),
                down_output: "y".into(),
            }],
        }],
    };
    crate::program::with_model(&model, |p| {
        plan.programs[0].original_sha256 = crate::live_kv::program_digest(&p.programs[0]);
        plan.validate(p).unwrap();
    });
    (plan, model)
}

#[test]
fn typed_schema_preserves_row_versions_and_rejects_ambiguity() {
    let (plan, _) = fixture();
    let mut v = serde_json::to_value(&plan).unwrap();
    assert!(matches!(
        crate::hetero::parse(&serde_json::to_vec(&v).unwrap()),
        Ok(crate::hetero::Plan::Channel(_))
    ));
    for (field, value) in [
        ("mode", serde_json::json!("row")),
        ("schema", serde_json::json!("plow-hetero-v4")),
        ("rows_ane", serde_json::json!(64)),
    ] {
        let mut invalid = v.clone();
        invalid[field] = value;
        assert!(crate::hetero::parse(&serde_json::to_vec(&invalid).unwrap()).is_err());
    }
    v["programs"][0]["partials"]["dtype"] = "bf16".into();
    assert!(crate::hetero::parse(&serde_json::to_vec(&v).unwrap()).is_err());
    for schema in ["plow-hetero-v1", "plow-hetero-v2"] {
        let mut v = serde_json::to_value(crate::hetero::HeteroPlan::default()).unwrap();
        v["schema"] = schema.into();
        if schema.ends_with("v1") {
            v.as_object_mut().unwrap().remove("weight_encoding");
        }
        assert!(matches!(
            crate::hetero::parse(&serde_json::to_vec(&v).unwrap()),
            Ok(crate::hetero::Plan::Row(_))
        ));
    }
}

#[test]
fn refuses_bad_coverage_scales_aliases_rows_and_spans() {
    let (good, model) = fixture();
    let mutations: &[fn(&mut ChannelPlan)] = &[
        |p| p.ane_channels = 0,
        |p| p.ane_channels = p.inter,
        |p| p.ane_channels = 31,
        |p| p.layers[0].down.gpu.cols[0] += 32,
        |p| p.layers[0].gate.ane.rows[1] += 32,
        |p| p.layers[0].gate.scale = Some("gamma".into()),
        |p| p.layers[0].down.cols -= 32,
        |p| p.layers[0].gate.tensor = "missing".into(),
        |p| p.programs[0].partials.gpu = "x".into(),
        |p| p.programs[0].partials.cols += 1,
        |p| p.programs[0].min_rows = 0,
        |p| p.programs[0].max_rows = 129,
        |p| p.programs[0].call_rows = 64,
        |p| p.programs[0].prog = 1,
        |p| p.programs[0].spans[0].insts = [1, 99],
        |p| p.programs[0].spans[0].insts = [1, 1],
        |p| p.programs[0].spans[0].residual = "x".into(),
        |p| p.programs[0].spans[0].layer = 1,
        |p| p.programs.push(p.programs[0].clone()),
        |p| p.layers.push(p.layers[0].clone()),
        |p| p.programs[0].original_sha256 = "a".repeat(64),
    ];
    for (i, mutate) in mutations.iter().enumerate() {
        let mut bad = good.clone();
        mutate(&mut bad);
        assert!(
            crate::program::with_model(&model, |p| bad.validate(p)).is_err(),
            "mutation {i}"
        );
    }
}

#[test]
fn rejects_packet_drift_even_with_refreshed_digest() {
    let (mut plan, mut model) = fixture();
    model.progs[0].insts[3].f[0] = 2.0;
    assert!(crate::program::with_model(&model, |p| plan.validate(p)).is_err());
    crate::program::with_model(&model, |p| {
        plan.programs[0].original_sha256 = crate::live_kv::program_digest(&p.programs[0]);
        assert!(plan.validate(p).unwrap_err().contains("single residual"));
    });
}
