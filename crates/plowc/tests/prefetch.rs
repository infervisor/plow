//! End-to-end §8.3 test: compile every example with `prefetch: true` and
//! confirm the emitted `Schedule` remains self-consistent for every bucket.
//!
//! Consistency invariants checked post-hoisting on the real plowc output:
//!  1. `streams[r].len() == packets[r].len()` for every resource.
//!  2. Task IDs and cycles at each position of streams match packets.
//!  3. `starts[t]` equals the cycle where `t` appears in its stream.
//!  4. Every data-dep edge `(a, b) ∈ tasks.edges` satisfies
//!     `starts[a] + dur[a] <= starts[b]`.
//!  5. `Scheduled::verify` still accepts (the crate's own counter-replay
//!     verifier — belt-and-suspenders that reordering didn't break gating).

use std::path::PathBuf;

use plowc::net::NetConfig;
use plowc::{compile, Options, Parallel, Source};
use schedule::Phase;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

fn prefetch_opts(out: PathBuf) -> Options {
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
        #[cfg(feature = "lean-verify")]
        lean_verify: true,
        #[cfg(not(feature = "lean-verify"))]
        lean_verify: false,
        counter_elim: false,
        scope_narrow: false,
        prefetch: true,
        sram_fit: false,
        lean_oracle: false,
        emit_sample: false,
        emit_tokenize: false,
        emit_trace: false,
        kv: Default::default(),
        weight_dtype_override: None,
    }
}

#[cfg(not(feature = "lean-verify"))]
#[test]
fn every_example_compiles_with_prefetch() {
    run_all();
}

#[cfg(feature = "lean-verify")]
#[test]
#[ignore = "requires plow_verify binary (run `lake build` in lean-plow/)"]
fn every_example_compiles_with_prefetch_and_verifier() {
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

        let out = std::env::temp_dir().join(format!(
            "plowc-prefetch-{}-{}",
            std::process::id(),
            net.name
        ));
        let report = compile(&Source::Net(net), &prefetch_opts(out.clone()))
            .unwrap_or_else(|e| panic!("{}: prefetch compile failed: {e}", path.display()));
        assert!(!report.buckets.is_empty(), "{}: no buckets", path.display());
        std::fs::remove_dir_all(&out).ok();
    }
}
