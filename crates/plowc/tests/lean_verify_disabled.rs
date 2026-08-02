//! When plowc is built *without* `--features lean-verify`, setting
//! `Options.lean_verify = true` must return a specific error rather than
//! silently skipping verification.
#![cfg(not(feature = "lean-verify"))]

use std::path::PathBuf;

use plowc::net::NetConfig;
use plowc::{compile, Options, Parallel, PlowcError, Source};
use schedule::Phase;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

#[test]
fn setting_lean_verify_without_feature_fails_with_disabled_error() {
    let dir = examples_dir();
    let first = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().is_some_and(|x| x == "json"))
        .expect("at least one example");

    let json = std::fs::read_to_string(&first).unwrap();
    let net: NetConfig = serde_json::from_str(&json).unwrap();
    let out = std::env::temp_dir().join(format!("plowc-noleandisabled-{}", std::process::id()));
    let opts = Options {
        no_tuning: false,
        tuning_db: None,
        gpu: "H100 SXM5".into(),
        num_gpus: 1,
        parallel: Parallel::Tp,
        batches: vec![1],
        seqs: vec![128],
        phases: vec![Phase::Prefill],
        page_kib: 16,
        out: out.clone(),
        lean_verify: true, // opt-in without the compile-time feature
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
    };
    match compile(&Source::Net(net), &opts) {
        Err(PlowcError::LeanVerifyDisabled) => {}
        other => panic!("expected LeanVerifyDisabled, got {other:?}"),
    }
    std::fs::remove_dir_all(&out).ok();
}
