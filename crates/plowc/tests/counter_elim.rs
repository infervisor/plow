//! End-to-end: compile every example with counter-elimination on and confirm
//! the run succeeds. When built with `--features lean-verify`, we also enable
//! the Lean verifier so the reduced schedule gets cross-checked per bucket.

use std::path::PathBuf;

use plowc::net::NetConfig;
use plowc::{compile, Options, Parallel, Source};
use schedule::Phase;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

fn elim_opts(out: PathBuf) -> Options {
    Options {
        no_tuning: false,
        tuning_db: None,
        gpu: "H100 SXM5".into(),
        num_gpus: 1,
        parallel: Parallel::Tp,
        batches: vec![1],
        seqs: vec![128],
        phases: vec![Phase::Prefill],
        page_kib: 16,
        out,
        // With the feature on, cross-check the reduced schedule against the
        // Lean side (this test is `#[ignore]`d in that case; see below).
        #[cfg(feature = "lean-verify")]
        lean_verify: true,
        #[cfg(not(feature = "lean-verify"))]
        lean_verify: false,
        counter_elim: true,
        scope_narrow: false,
        prefetch: false,
        sram_fit: false,
        lean_oracle: false,
        emit_sample: false,
        emit_tokenize: false,
        emit_trace: false,        kv: Default::default(),
        weight_dtype_override: None,
    }
}

/// Pure-Rust path (no Lean binary required): every example compiles with
/// counter elimination on. Guarded so the Lean-feature build routes to the
/// ignored end-to-end variant below.
#[cfg(not(feature = "lean-verify"))]
#[test]
fn every_example_compiles_with_counter_elim() {
    run_all();
}

/// With the Lean feature on, we also verify — needs the binary, so `#[ignore]`.
#[cfg(feature = "lean-verify")]
#[test]
#[ignore = "requires plow_verify binary (run `lake build` in lean-plow/)"]
fn every_example_compiles_with_counter_elim_and_verifier() {
    run_all();
}

fn run_all() {
    let dir = examples_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read examples dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no example networks in {dir:?}");

    for path in files {
        let json = std::fs::read_to_string(&path).expect("read example");
        let net: NetConfig = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("{}: parse failed: {e}", path.display()));

        let out = std::env::temp_dir()
            .join(format!("plowc-elim-{}-{}", std::process::id(), net.name));
        let report = compile(&Source::Net(net), &elim_opts(out.clone()))
            .unwrap_or_else(|e| panic!("{}: compile failed: {e}", path.display()));
        assert!(!report.buckets.is_empty(), "{}: no buckets", path.display());
        std::fs::remove_dir_all(&out).ok();
    }
}
