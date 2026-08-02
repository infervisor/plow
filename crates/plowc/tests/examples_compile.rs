//! Every shipped `examples/*.json` network parses and compiles end-to-end to a
//! non-empty packet stream on a single GPU. Guards the example dims against drift
//! in the net schema or the compiler pipeline.

use plowc::{compile, net::NetConfig, Options, Parallel, Source};
use schedule::Phase;
use std::path::PathBuf;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

/// A small, fast bucket — realistic dims, tiny token count, single GPU (so the
/// known cross-unit clustering bug is not in scope here).
fn opts(out: PathBuf) -> Options {
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
fn all_examples_compile_to_packets() {
    let dir = examples_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read examples dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no example networks found in {dir:?}");

    for path in files {
        let json = std::fs::read_to_string(&path).expect("read example");
        let net: NetConfig = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("{}: parse failed: {e}", path.display()));

        let out =
            std::env::temp_dir().join(format!("plowc-ex-{}-{}", std::process::id(), net.name));
        let report = compile(&Source::Net(net), &opts(out.clone()))
            .unwrap_or_else(|e| panic!("{}: compile failed: {e}", path.display()));

        assert_eq!(
            report.buckets.len(),
            1,
            "{}: expected one bucket",
            path.display()
        );
        let b = &report.buckets[0];
        assert!(
            b.packet_bytes > 0,
            "{}: empty packet stream",
            path.display()
        );
        assert!(b.instructions > 0, "{}: no instructions", path.display());
        assert!(b.makespan > 0, "{}: zero makespan estimate", path.display());

        std::fs::remove_dir_all(&out).ok();
    }
}
