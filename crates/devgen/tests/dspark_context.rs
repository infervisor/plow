//! DSpark context tail (`PLOW_EMIT_SPEC`, `PLOW_SPEC_KIND=gemma4_dspark`) on a miniature
//! Gemma-4 target: the knob off (or the kind alone) leaves the packet byte-identical; on, every
//! target program keeps its own instructions exactly and gains only the tap copies, the tail and
//! the drafter-layer KV writers, which clone the program's last full-layer writers.
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use clap::Parser;
use packet::dev::{DevInst, DevOp, TENSOR_NONE};

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

static EMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const LAYERS: u32 = 4;
const TAPS: [u32; 2] = [1, 2];
const DRAFT_LAYERS: u32 = 2;
const HIDDEN: u64 = 512;
const VOCAB: u64 = 4096;

fn target_config(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        r#"{
        "model_type":"gemma4_text", "hidden_size":512, "intermediate_size":1024,
        "num_hidden_layers":4, "num_attention_heads":8, "head_dim":64,
        "global_head_dim":512, "num_key_value_heads":2, "num_global_key_value_heads":1,
        "attention_k_eq_v":true, "sliding_window":512, "rms_norm_eps":1e-6,
        "vocab_size":4096, "final_logit_softcapping":30.0, "tie_word_embeddings":true,
        "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
        "rope_parameters":{"sliding_attention":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
            "full_attention":{"rope_theta":1000000.0,"partial_rotary_factor":0.25,
                "rope_type":"proportional"}}
    }"#,
    )
    .unwrap();
}

fn draft_config(dir: &Path, vocab: u64) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        format!(
            r#"{{
        "architectures":["Gemma4DSparkModel"], "model_type":"gemma4_text",
        "hidden_size":512, "intermediate_size":1024, "num_hidden_layers":2,
        "num_attention_heads":8, "global_head_dim":512, "num_global_key_value_heads":1,
        "attention_k_eq_v":true, "hidden_size_per_layer_input":0, "vocab_size":{vocab},
        "num_target_layers":4, "target_layer_ids":[1,2], "block_size":3, "mask_token_id":4,
        "markov_head_type":"vanilla", "markov_rank":16, "enable_confidence_head":false,
        "rms_norm_eps":1e-6, "layer_types":["full_attention","full_attention"],
        "rope_parameters":{{"full_attention":{{"partial_rotary_factor":0.25,
            "rope_theta":1000000.0,"rope_type":"proportional"}}}}
    }}"#
        ),
    )
    .unwrap();
}

/// A safetensors file with this header and zero (sparse) data.
fn shard(path: &Path, meta: serde_json::Value, entries: &[(String, &str, Vec<u64>)]) {
    let mut h = serde_json::Map::new();
    h.insert("__metadata__".into(), meta);
    let mut ofs = 0u64;
    for (name, dtype, shape) in entries {
        let elt = match *dtype {
            "BF16" => 2,
            "F32" => 4,
            _ => 1,
        };
        let size = elt * shape.iter().product::<u64>();
        h.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [ofs, ofs + size]}),
        );
        ofs += size;
    }
    let mut hb = serde_json::to_vec(&h).unwrap();
    hb.resize(hb.len().div_ceil(8) * 8, b' ');
    let mut bytes = (hb.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&hb);
    std::fs::write(path, &bytes).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(bytes.len() as u64 + ofs)
        .unwrap();
}

/// `model-dspark.safetensors` (+ the FP8 twins) as `scripts/gemma4_dspark_rekey.py` and
/// `gemma4_dspark_fp8.py` lay them out; `skip` leaves one tensor out.
fn shards(dir: &Path, skip: Option<&str>) {
    let (h, v, hd, r, i) = (HIDDEN, VOCAB, 512u64, 16u64, 1024u64);
    let mut e: Vec<(String, &str, Vec<u64>)> = vec![
        ("dspark.embed_tokens.weight".into(), "BF16", vec![v, h]),
        ("dspark.hidden_norm.weight".into(), "BF16", vec![h]),
        ("dspark.lm_head.weight".into(), "BF16", vec![v, h]),
        (
            "dspark.markov_head.markov_w1.weight".into(),
            "BF16",
            vec![v, r],
        ),
        (
            "dspark.markov_head.markov_w2.weight".into(),
            "BF16",
            vec![v, r],
        ),
        ("dspark.norm.weight".into(), "BF16", vec![h]),
    ];
    for j in 0..TAPS.len() {
        e.push((format!("dspark.fc.{j}.weight"), "BF16", vec![h, h]));
    }
    for l in 0..DRAFT_LAYERS {
        for (s, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("layer_scalar", vec![1]),
            ("mlp.down_proj.weight", vec![h, i]),
            ("mlp.gate_proj.weight", vec![i, h]),
            ("mlp.up_proj.weight", vec![i, h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("post_feedforward_layernorm.weight", vec![h]),
            ("pre_feedforward_layernorm.weight", vec![h]),
            ("self_attn.k_norm.weight", vec![hd]),
            ("self_attn.k_proj.weight", vec![hd, h]),
            ("self_attn.o_proj.weight", vec![h, 8 * hd]),
            ("self_attn.q_norm.weight", vec![hd]),
            ("self_attn.q_proj.weight", vec![8 * hd, h]),
        ] {
            e.push((format!("dspark.layers.{l}.{s}"), "BF16", shape));
        }
    }
    let twins: Vec<(String, &str, Vec<u64>)> = e
        .iter()
        .filter(|(n, _, s)| s.len() == 2 && !n.contains("embed") && !n.contains("markov"))
        .flat_map(|(n, _, s)| {
            [
                (format!("fp8/{n}"), "F8_E4M3", s.clone()),
                (format!("fp8/{n}_scale"), "F32", vec![s[0]]),
            ]
        })
        .collect();
    e.retain(|(n, _, _)| Some(n.as_str()) != skip);
    let bf16 = dir.join("model-dspark.safetensors");
    let config =
        plow_asset::knob::sha256_hex(&std::fs::read(dir.join("draft/config.json")).unwrap());
    shard(
        &bf16,
        serde_json::json!({ "config_sha256": config, "fc_split": TAPS.len().to_string() }),
        &e,
    );
    let sha = plow_asset::knob::sha256_hex(&std::fs::read(&bf16).unwrap());
    shard(
        &dir.join("model-dspark-fp8.safetensors"),
        serde_json::json!({ "source_shard_sha256": sha }),
        &twins,
    );
}

#[derive(Clone)]
struct Snapshot {
    tensors: Vec<(String, u64)>,
    progs: Vec<(u32, Vec<DevInst>)>,
}

fn emit(dir: &Path, f: impl FnOnce(&mut devgen::emit_config::EmitConfig)) -> (Vec<u8>, Snapshot) {
    let mut cfg = Args::try_parse_from(["test"]).unwrap().emit;
    cfg.fp8_kv = true;
    cfg.max_chunk = Some(1024);
    cfg.decode_ladder = Some("1,2,4".into());
    cfg.skip_coverage = true;
    f(&mut cfg);
    let out = dir.join("model.pkt");
    let snap = Rc::new(RefCell::new(None));
    let s = snap.clone();
    devgen::run_verified(
        devgen::EmitArgs {
            dir: dir.to_path_buf(),
            ctx: 2048,
            out: out.display().to_string(),
            n_cu: 132,
            tp: 1,
            block_spec: None,
            embed_cubin: None,
            embed_hsaco: None,
            rope_gen: true,
            l2_layout: None,
            gpu: String::new(),
            arch: "sm_90a".into(),
            emit_cfg: Some(cfg),
            whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
        },
        Some(Box::new(move |m| {
            *s.borrow_mut() = Some(Snapshot {
                tensors: m
                    .tensors
                    .iter()
                    .map(|t| (t.name.clone(), t.bytes))
                    .collect(),
                progs: m
                    .prog_t
                    .iter()
                    .copied()
                    .zip(m.progs.iter().map(|p| p.insts.clone()))
                    .collect(),
            });
            Ok(devgen::LeanReport {
                verified: false,
                oracle: false,
                reason: Some("structural contract test".into()),
                ..Default::default()
            })
        })),
    );
    let snap = snap.borrow().clone().unwrap();
    (std::fs::read(out).unwrap(), snap)
}

fn root(name: &str) -> PathBuf {
    let r = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&r);
    std::fs::create_dir_all(&r).unwrap();
    target_config(&r);
    // The drafter shard makes this a checkpoint dir; the target shard only needs the layer
    // scalars (the structural emit binds no weight bytes).
    let scalars: Vec<(String, &str, Vec<u64>)> = (0..LAYERS)
        .map(|l| (format!("model.layers.{l}.layer_scalar"), "BF16", vec![1]))
        .collect();
    shard(
        &r.join("model.safetensors"),
        serde_json::json!({}),
        &scalars,
    );
    r
}

fn on(draft: &Path) -> impl FnOnce(&mut devgen::emit_config::EmitConfig) + '_ {
    move |c| {
        c.emit_spec = true;
        c.spec_kind = Some("gemma4_dspark".into());
        c.spec_draft_dir = Some(draft.display().to_string());
        c.span_verify = true;
    }
}

/// The JSON object at the first occurrence of `prefix` in the blob.
fn json_at(blob: &[u8], prefix: &str) -> serde_json::Value {
    let at = blob
        .windows(prefix.len())
        .position(|w| w == prefix.as_bytes())
        .unwrap_or_else(|| panic!("no {prefix} in the blob"));
    serde_json::Deserializer::from_slice(&blob[at..])
        .into_iter::<serde_json::Value>()
        .next()
        .unwrap()
        .unwrap()
}

fn kv_writer(d: &DevInst) -> bool {
    d.op == DevOp::HeadNormRopeFp8 as u16
}

#[test]
fn the_context_tail_adds_ops_and_changes_no_target_instruction() {
    let _g = EMIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = root("dspark_context");
    let draft = dir.join("draft");
    draft_config(&draft, VOCAB);
    shards(&dir, None);

    let (off, base) = emit(&dir, |_| {});
    let (kind_only, _) = emit(&dir, |c| c.spec_kind = Some("gemma4_dspark".into()));
    assert!(off == kind_only, "PLOW_SPEC_KIND alone changed the packet");
    let (span_off, span_base) = emit(&dir, |c| c.span_verify = true);
    let (blob, ds) = emit(&dir, on(&draft));
    assert_ne!(blob, span_off);

    // Target tensors keep their handles; the drafter's follow them.
    assert_eq!(ds.tensors[..span_base.tensors.len()], span_base.tensors[..]);
    let name = |h: u32| -> &str {
        if h == TENSOR_NONE {
            ""
        } else {
            &ds.tensors[h as usize].0
        }
    };
    let bytes = |n: &str| ds.tensors.iter().find(|t| t.0 == n).unwrap().1;
    for l in LAYERS..LAYERS + DRAFT_LAYERS {
        for s in ["k", "v", "k_scale", "v_scale"] {
            assert_eq!(bytes(&format!("kv.{l}.{s}")), bytes(&format!("kv.3.{s}")));
        }
    }
    let drafter = |d: &DevInst| {
        d.t.iter().any(|&h| {
            let n = name(h);
            n.starts_with("act.dspark.")
                || n.contains("dspark.")
                || n.strip_prefix("kv.")
                    .and_then(|r| r.split('.').next())
                    .and_then(|l| l.parse::<u32>().ok())
                    .is_some_and(|l| l >= LAYERS)
        })
    };
    let key = |d: &DevInst| (d.op, d.blocks, d.t, d.i, d.f.map(f32::to_bits), d.j);
    assert_eq!(ds.progs.len(), span_base.progs.len());
    for ((rows, insts), (base_rows, base_insts)) in ds.progs.iter().zip(&span_base.progs) {
        assert_eq!(rows, base_rows);
        let kept: Vec<_> = insts.iter().filter(|d| !drafter(d)).map(key).collect();
        let want: Vec<_> = base_insts.iter().map(key).collect();
        assert!(kept == want, "rows {rows}: a target instruction changed");

        // One copy per tap, between the tapped layer's KV write and the next layer's.
        let writer = |l: u32| {
            insts
                .iter()
                .position(|d| kv_writer(d) && name(d.t[0]) == format!("kv.{l}.k"))
                .unwrap()
        };
        for (j, &l) in TAPS.iter().enumerate() {
            let copies: Vec<usize> = insts
                .iter()
                .enumerate()
                .filter(|(_, d)| name(d.t[0]) == format!("act.dspark.tap{j}"))
                .map(|(pc, _)| pc)
                .collect();
            assert_eq!(copies.len(), 1, "rows {rows} tap {j}");
            let c = &insts[copies[0]];
            assert_eq!(c.op, DevOp::Residual as u16);
            assert_eq!(
                (name(c.t[1]), name(c.t[2]), c.f[0]),
                ("act.x", "act.x", 0.5)
            );
            assert_eq!(
                c.i[0],
                packet::devbuild::program_rows(*rows) * HIDDEN as u32
            );
            assert!(writer(l) < copies[0] && copies[0] < writer(l + 1));
        }

        // Drafter writers clone the last full layer's K/V writers.
        let (kw, vw) = (&insts[writer(3)], {
            let pc = insts
                .iter()
                .position(|d| kv_writer(d) && name(d.t[0]) == "kv.3.v")
                .unwrap();
            &insts[pc]
        });
        for i in 0..DRAFT_LAYERS {
            let l = LAYERS + i;
            for (src, half, gamma) in [(kw, "k", true), (vw, "v", false)] {
                let w: Vec<_> = insts
                    .iter()
                    .filter(|d| kv_writer(d) && name(d.t[0]) == format!("kv.{l}.{half}"))
                    .collect();
                assert_eq!(w.len(), 1, "rows {rows} kv.{l}.{half}");
                let w = w[0];
                assert_eq!(name(w.t[1]), format!("act.dspark.k{i}"));
                assert_eq!(name(w.t[6]), format!("kv.{l}.{half}_scale"));
                if gamma {
                    assert_eq!(
                        name(w.t[2]),
                        format!("dspark.layers.{i}.self_attn.k_norm.weight")
                    );
                } else {
                    assert_eq!(w.t[2], src.t[2]);
                }
                for s in [3, 4, 5, 7] {
                    assert_eq!(w.t[s], src.t[s], "rows {rows} kv.{l}.{half} t{s}");
                }
                assert_eq!((w.i, w.j, w.blocks), (src.i, src.j, src.blocks));
            }
        }
    }

    let ctx = json_at(&blob, r#"{"version":1,"family":"gemma4_dspark""#);
    let m: plow_asset::spec_context::Manifest = serde_json::from_value(ctx).unwrap();
    m.validate(LAYERS).unwrap();
    assert_eq!((m.taps, m.kv_layers), (TAPS.to_vec(), vec![4, 5]));
    assert_eq!(
        (m.projection.as_str(), m.twins_sha256.as_str()),
        ("bf16", "")
    );
    let kv = json_at(&blob, r#"{"full_layers":"#);
    assert_eq!(kv["full_layers"], serde_json::json!([1, 3, 4, 5]));
    assert_eq!(
        json_at(&off, r#"{"full_layers":"#)["full_layers"],
        serde_json::json!([1, 3])
    );

    // FP8 W8A8: the tail binds the twins and quantizes its activations for the GEMMs.
    let (_, fp8) = emit(&dir, |c| {
        on(&draft)(c);
        c.fp8 = true;
        c.w8a8 = true;
    });
    let names: Vec<&str> = fp8.tensors.iter().map(|t| t.0.as_str()).collect();
    assert!(names.contains(&"fp8/dspark.fc.1.weight"));
    assert!(!names.contains(&"dspark.fc.1.weight"));
    let prefill = &fp8.progs[0].1;
    let fname = |h: u32| fp8.tensors[h as usize].0.as_str();
    assert!(prefill
        .iter()
        .any(|d| d.op == DevOp::QuantFp8 as u16 && fname(d.t[0]) == "act.dspark.xq1"));
    // Each projection phase runs back to back (one cuBLASLt run per phase on library rungs).
    for prefix in ["fp8/dspark.fc.", "fp8/dspark.layers."] {
        let at: Vec<usize> = prefill
            .iter()
            .enumerate()
            .filter(|(_, d)| d.t[2] != TENSOR_NONE && fname(d.t[2]).starts_with(prefix))
            .map(|(i, _)| i)
            .collect();
        assert!(at.len() >= 2 && at.windows(2).all(|w| w[1] == w[0] + 1), "{prefix}: {at:?}");
    }
    let _ = std::fs::remove_dir_all(dir);
}

fn refusal(f: impl FnOnce(&Path, &Path) + std::panic::UnwindSafe) -> String {
    let _g = EMIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = root("dspark_refusal");
    let draft = dir.join("draft");
    let err = std::panic::catch_unwind(move || {
        f(&dir, &draft);
        emit(&dir, on(&draft));
    })
    .expect_err("emit should refuse");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn a_mismatched_drafter_is_refused_by_name() {
    let e = refusal(|dir, draft| {
        draft_config(draft, 8192);
        shards(dir, None);
    });
    assert!(e.contains("vocab_size != target vocab"), "{e}");
    let e = refusal(|dir, draft| {
        draft_config(draft, VOCAB);
        shards(dir, Some("dspark.layers.1.self_attn.k_norm.weight"));
    });
    assert!(
        e.contains("lacks dspark.layers.1.self_attn.k_norm.weight"),
        "{e}"
    );
    let e = refusal(|dir, draft| {
        draft_config(draft, VOCAB);
        shards(dir, None);
        let cfg = std::fs::read_to_string(draft.join("config.json")).unwrap();
        let moved = cfg.replace(r#""target_layer_ids":[1,2]"#, r#""target_layer_ids":[0,2]"#);
        std::fs::write(draft.join("config.json"), moved).unwrap();
    });
    assert!(e.contains("was not re-keyed from"), "{e}");
}
