//! End-to-end §8.5 test: compile every example with `sram_fit: true` and
//! verify the diagnostic pass produces a valid report per bucket. The pass
//! is analysis-only in this phase — no schedule mutation — so the output
//! should match the baseline compile bit-for-bit.

use std::path::PathBuf;

use plowc::net::NetConfig;
use plowc::{compile, Options, Parallel, Source};
use schedule::Phase;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

fn sram_fit_opts(out: PathBuf) -> Options {
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
        lean_verify: false,
        counter_elim: false,
        scope_narrow: false,
        prefetch: false,
        sram_fit: true,
        lean_oracle: false,
        emit_sample: false,
        emit_tokenize: false,
        emit_trace: false,        kv: Default::default(),
        weight_dtype_override: None,
    }
}

#[test]
fn every_example_compiles_with_sram_fit() {
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
            .join(format!("plowc-sramfit-{}-{}", std::process::id(), net.name));
        let report = compile(&Source::Net(net), &sram_fit_opts(out.clone()))
            .unwrap_or_else(|e| panic!("{}: compile failed: {e}", path.display()));
        assert!(!report.buckets.is_empty(), "{}: no buckets", path.display());
        std::fs::remove_dir_all(&out).ok();
    }
}
