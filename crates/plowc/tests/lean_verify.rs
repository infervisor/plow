//! End-to-end: run `plowc` with `lean_verify: true` and confirm every bucket
//! is accepted by the Lean CLI (`plow_verify`).
//!
//! Compiled only under `--features lean-verify`; `#[ignore]`d because it also
//! requires the Lean binary to exist:
//!
//!   cd lean-plow && lake build
//!   cargo test -p plowc --features lean-verify --tests -- --ignored

#![cfg(feature = "lean-verify")]

use std::path::PathBuf;

use plowc::net::NetConfig;
use plowc::{compile, Options, Parallel, Source};
use schedule::Phase;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

fn verify_opts(out: PathBuf) -> Options {
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
        lean_verify: true,
        counter_elim: false,
        scope_narrow: false,
        prefetch: false,
        sram_fit: false,
        lean_oracle: false,
        emit_sample: false,
        emit_tokenize: false,
        emit_trace: false,
        kv: Default::default(),
        weight_dtype_override: None,
    }
}

#[test]
#[ignore = "requires plow_verify binary (run `lake build` in lean-plow/)"]
fn every_example_bucket_is_accepted_by_lean() {
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

        let out =
            std::env::temp_dir().join(format!("plowc-lean-{}-{}", std::process::id(), net.name));
        let report = compile(&Source::Net(net), &verify_opts(out.clone()))
            .unwrap_or_else(|e| panic!("{}: lean-verified compile failed: {e}", path.display()));
        assert!(!report.buckets.is_empty(), "{}: no buckets", path.display());
        std::fs::remove_dir_all(&out).ok();
    }
}
