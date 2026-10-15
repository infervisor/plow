//! `PLOW_EMIT_RUNG_ENV`: per-rung overrides of rung-eligible emit knobs, end to end through
//! `run_verified` on a two-layer Gemma-4 BF16 sm_90a config (whose production defaults turn
//! `no_glu_fuse` on packet-wide).
//!
//! Each emit runs in a child process: the emitter's installed config and knob record are
//! process-global.

use clap::Parser;
use packet::dev::DevOp;
use std::path::{Path, PathBuf};

const CONFIG: &str = r#"{
        "model_type":"gemma4_text", "hidden_size":3840, "intermediate_size":15360,
        "num_hidden_layers":2, "num_attention_heads":16, "head_dim":256,
        "global_head_dim":512, "num_key_value_heads":8, "num_global_key_value_heads":1,
        "attention_k_eq_v":true, "sliding_window":1024, "rms_norm_eps":1e-6,
        "vocab_size":262144, "final_logit_softcapping":30.0, "tie_word_embeddings":true,
        "layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{"sliding_attention":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
            "full_attention":{"rope_theta":1000000.0,"partial_rotary_factor":0.25}}
    }"#;

const BUCKETS: [u32; 3] = [128, 512, 1024];

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    emit: devgen::emit_config::EmitConfig,
}

struct Emitted {
    pkt: Vec<u8>,
    build: serde_json::Value,
    /// Per prefill program: (rows, instruction ops).
    progs: Vec<(u32, Vec<u16>)>,
}

/// One emit in a child process of this test binary: the emitter's knob record (production-default
/// notes included) is process-global, so emits that compare packet-wide settings must not share one.
fn emit(
    root: &Path,
    name: &str,
    env: &[(&str, &str)],
    rung_env: Option<&str>,
) -> Result<Emitted, String> {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "--exact",
        "child_emit",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ])
    .env("RUNG_ENV_CHILD_DIR", &dir)
    .envs(env.iter().copied());
    if let Some(r) = rung_env {
        cmd.env("RUNG_ENV_CHILD_SPEC", r);
    }
    let out = cmd.output().unwrap();
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr));
    }
    let progs: Vec<(u32, Vec<u16>)> =
        serde_json::from_slice(&std::fs::read(dir.join("progs.json")).unwrap()).unwrap();
    Ok(Emitted {
        pkt: std::fs::read(dir.join("model.pkt")).unwrap(),
        build: serde_json::from_slice(&std::fs::read(dir.join("build.json")).unwrap()).unwrap(),
        progs,
    })
}

#[test]
#[ignore = "run by rung_overrides_resolve_per_program_and_nowhere_else in a child process"]
fn child_emit() {
    let Some(dir) = std::env::var_os("RUNG_ENV_CHILD_DIR").map(PathBuf::from) else {
        return;
    };
    let mut cfg = Args::try_parse_from(["test"]).unwrap().emit;
    cfg.max_chunk = Some(1024);
    cfg.emit_packed_prefill = Some(false);
    cfg.rung_env = std::env::var("RUNG_ENV_CHILD_SPEC").ok();
    let sink = dir.join("progs.json");
    devgen::run_verified(
        devgen::EmitArgs {
            dir: dir.clone(),
            ctx: 4096,
            out: dir.join("model.pkt").display().to_string(),
            n_cu: 132,
            tp: 1,
            block_spec: None,
            embed_cubin: None,
            embed_hsaco: None,
            rope_gen: true,
            l2_layout: None,
            gpu: "h100".into(),
            arch: "sm_90a".into(),
            emit_cfg: Some(cfg),
            whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
        },
        Some(Box::new(move |model| {
            let lo = packet::devbuild::decode_rung_lo(&model.prog_t);
            let progs: Vec<(u32, Vec<u16>)> = (0..lo)
                .map(|i| {
                    let ops = model.progs[i].insts.iter().map(|d| d.op).collect();
                    (packet::devbuild::program_rows(model.prog_t[i]), ops)
                })
                .collect();
            std::fs::write(&sink, serde_json::to_vec(&progs).unwrap()).unwrap();
            Ok(devgen::LeanReport::skipped("rung override contract test"))
        })),
    );
}

fn has(ops: &[u16], op: DevOp) -> bool {
    ops.contains(&(op as u16))
}

fn lt_rows(build: &serde_json::Value) -> Vec<u64> {
    let mut rows: Vec<u64> = build["cublaslt"]["shapes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|s| s["phase"] == "prefill")
        .filter_map(|s| s["rows"].as_u64())
        .collect();
    rows.sort();
    rows.dedup();
    rows
}

fn resolved(build: &serde_json::Value, rows: u32, knob: &str) -> (String, String) {
    let p = build["rung_knobs"]["programs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["phase"] == "prefill" && p["rows"] == rows)
        .unwrap_or_else(|| panic!("no prefill:{rows} in rung_knobs"));
    let k = &p["knobs"][knob];
    (
        k["value"].as_str().unwrap().into(),
        k["source"].as_str().unwrap().into(),
    )
}

fn all_rungs(env: &str, value: &str) -> String {
    BUCKETS
        .iter()
        .map(|r| format!("prefill:{r}:{env}={value}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn refused(root: &Path, name: &str, rung_env: &str) -> String {
    match emit(root, name, &[], Some(rung_env)) {
        Ok(_) => panic!("emit accepted {rung_env}"),
        Err(e) => e,
    }
}

#[test]
fn rung_overrides_resolve_per_program_and_nowhere_else() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rung_env");
    let _ = std::fs::remove_dir_all(&root);

    // Packet-wide: the production defaults (unfused GLU, Lt prefill projections on every rung).
    let base = emit(&root, "base", &[], None).unwrap();
    assert_eq!(base.progs.iter().map(|p| p.0).collect::<Vec<_>>(), BUCKETS);
    assert!(base.progs.iter().all(|(_, ops)| !has(ops, DevOp::GemmGlu)));
    assert_eq!(lt_rows(&base.build), [128, 512, 1024]);
    assert_eq!(
        resolved(&base.build, 128, "emit.no_glu_fuse"),
        ("true".into(), "packet".into())
    );

    // Mixed: rung 128 fused; the other rungs' programs are byte-for-byte the base's.
    let mixed = emit(&root, "mixed", &[], Some("prefill:128:PLOW_NO_GLU_FUSE=0")).unwrap();
    let (rows, ops) = &mixed.progs[0];
    assert_eq!(*rows, 128);
    assert!(
        has(ops, DevOp::GemmGlu),
        "rung 128 must lower the fused GLU"
    );
    assert_eq!(
        &mixed.progs[1..],
        &base.progs[1..],
        "the 512/1024 programs must not move"
    );
    assert_ne!(mixed.progs[0], base.progs[0]);
    assert_eq!(
        resolved(&mixed.build, 128, "emit.no_glu_fuse"),
        ("false".into(), "rung".into())
    );
    assert_eq!(
        resolved(&mixed.build, 512, "emit.no_glu_fuse"),
        ("true".into(), "packet".into())
    );

    // Opposite-base uniform equivalence: every rung overridden == the packet-wide value.
    let fused = emit(&root, "fused", &[("PLOW_NO_GLU_FUSE", "0")], None).unwrap();
    assert!(has(&fused.progs[0].1, DevOp::GemmGlu));
    let fused_by_rung = emit(
        &root,
        "fused-rung",
        &[],
        Some(&all_rungs("PLOW_NO_GLU_FUSE", "0")),
    )
    .unwrap();
    assert_eq!(fused_by_rung.progs, fused.progs);
    assert!(
        fused_by_rung.pkt == fused.pkt,
        "model.pkt must be byte-identical"
    );
    let unfused_by_rung = emit(
        &root,
        "unfused-rung",
        &[("PLOW_NO_GLU_FUSE", "0")],
        Some(&all_rungs("PLOW_NO_GLU_FUSE", "1")),
    )
    .unwrap();
    assert!(
        unfused_by_rung.pkt == base.pkt,
        "model.pkt must be byte-identical"
    );

    // Refusals: a width the packet does not emit, a packet-wide knob, an unknown knob.
    let e = refused(&root, "stray", "prefill:96:PLOW_NO_GLU_FUSE=0");
    assert!(e.contains("match no rung"), "{e}");
    let e = refused(
        &root,
        "packetwide",
        "prefill:128:PLOW_EMIT_PREFILL_CUBLASLT=0",
    );
    assert!(e.contains("packet-wide"), "{e}");
    let e = refused(&root, "unknown", "prefill:128:PLOW_NOT_A_KNOB=1");
    assert!(e.contains("not a registered emit knob"), "{e}");

    std::fs::remove_dir_all(root).unwrap();
}
