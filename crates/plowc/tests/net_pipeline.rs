//! End-to-end: a plow-native network JSON compiles, across a bucket ladder, to
//! per-bucket packet streams + a shared weight layout, with sane estimates.

use plowc::{compile, net::NetConfig, Options, Parallel, Source};
use schedule::Phase;

fn net() -> NetConfig {
    let json = include_str!("../examples/mlp_block.json");
    serde_json::from_str(json).expect("parse example net")
}

fn opts(out: std::path::PathBuf) -> Options {
    Options {
        no_tuning: false,
        tuning_db: None,
        gpu: "H100 SXM5".into(),
        num_gpus: 1,
        parallel: Parallel::Tp,
        // Realistic shapes: fine per-tile counters (ClusterMode::Fine, now the
        // default) emit per-consumer-tile counters that match the fine schedule,
        // eliminating the coarse-counter deadlock at multi-tile scales.
        batches: vec![1, 8],
        seqs: vec![128, 2048],
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
fn net_compiles_to_packet_streams() {
    let dir = std::env::temp_dir().join(format!("plowc-test-{}", std::process::id()));
    let report = compile(&Source::Net(net()), &opts(dir.clone())).expect("compile");

    // 2 batches × 2 seqs × 1 phase = 4 buckets, each a real packet stream.
    assert_eq!(report.buckets.len(), 4);
    for b in &report.buckets {
        assert!(b.packet_bytes > 0, "empty packet stream");
        assert!(b.instructions > 0, "no instructions emitted");
        assert!(b.makespan > 0, "zero makespan estimate");
        assert!(b.ideal_makespan > 0, "zero ideal-makespan estimate");
        assert!(
            dir.join(&b.packet_file).exists(),
            "{} not written",
            b.packet_file
        );
    }

    // One shared weight layout across all buckets — a flip moves no weights.
    assert!(report.weight_shared);
    assert!(report.weight.is_some());
    // The network has attention ⇒ a KV layout was determined.
    assert!(report.kv.is_some());
    // Manifest written.
    assert!(dir.join("weights.json").exists());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn rejects_unimplemented_parallelism() {
    let dir = std::env::temp_dir().join(format!("plowc-pp-{}", std::process::id()));
    let mut o = opts(dir);
    o.num_gpus = 2;
    o.parallel = Parallel::Pp;
    assert!(compile(&Source::Net(net()), &o).is_err());
}

#[test]
fn rejects_degenerate_dimensions() {
    let dir = std::env::temp_dir().join(format!("plowc-bad-{}", std::process::id()));

    // Zero batch in a bucket — clean error, no panic.
    let mut o = opts(dir.clone());
    o.batches = vec![0];
    assert!(
        compile(&Source::Net(net()), &o).is_err(),
        "batch=0 must be rejected"
    );

    // hidden = 0 in the network config.
    let mut bad = net();
    bad.hidden = 0;
    assert!(
        compile(&Source::Net(bad), &opts(dir.clone())).is_err(),
        "hidden=0 must be rejected"
    );

    // A GEMM with n = 0.
    let bad: NetConfig =
        serde_json::from_str(r#"{"name":"z","hidden":16,"ops":[{"op":"gemm","n":0}]}"#).unwrap();
    assert!(
        compile(&Source::Net(bad), &opts(dir)).is_err(),
        "gemm n=0 must be rejected"
    );
}
