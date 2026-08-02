//! End-to-end: compile a real network JSON through `plowc`, then run the
//! resulting packet stream through `plowrt`'s simulator AND the full runtime
//! execution path (ModelBundle::load → Registry → AppState::generate). Proves
//! the compiler → binary → runtime interpreter path produces a valid,
//! deadlock-free schedule for an actual transformer block.

use std::sync::Arc;

use plowc::{compile, net::NetConfig, Options, Parallel, Source};
use plowrt::asset::ModelBundle;
use plowrt::device::cpu::CpuBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::AppState;
use plowrt::sim::{MathMode, SimReport, Simulator};
use schedule::Phase;

/// Parse one of the checked-in example networks.
fn load_net(json: &str) -> NetConfig {
    serde_json::from_str(json).expect("parse example net")
}

/// Standard compiler options for E2E simulation tests.
fn opts(out: std::path::PathBuf) -> Options {
    Options {
        no_tuning: false,
        tuning_db: None,
        gpu: "H100 SXM5".into(),
        num_gpus: 1,
        parallel: Parallel::Tp,
        batches: vec![1, 4],
        seqs: vec![128, 512],
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

/// Compile, decode every emitted `.pkt`, simulate each, and return all reports.
fn compile_and_simulate(net: NetConfig) -> Vec<(String, SimReport)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowc-e2e-sim-{}-{}", std::process::id(), id));
    let report = compile(&Source::Net(net), &opts(dir.clone())).expect("compile succeeded");

    let mut sim_reports = Vec::new();
    for bucket in &report.buckets {
        let pkt_path = dir.join(&bucket.packet_file);
        let bytes =
            std::fs::read(&pkt_path).unwrap_or_else(|e| panic!("read {}: {e}", bucket.packet_file));
        let program = packet::Program::decode(&bytes)
            .unwrap_or_else(|e| panic!("decode {}: {e}", bucket.packet_file));

        let sim = Simulator::new(MathMode::DryRun);
        let sim_report = sim.run(&program);
        sim_reports.push((bucket.packet_file.clone(), sim_report));
    }

    std::fs::remove_dir_all(&dir).ok();
    sim_reports
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Full Llama-3-8B transformer block: compile → simulate → assert no deadlock,
/// positive makespan, all packets fire.
#[test]
fn llama3_8b_compiles_and_simulates_deadlock_free() {
    let net = load_net(include_str!("../examples/transformer_block_llama3_8b.json"));
    let results = compile_and_simulate(net);

    assert!(!results.is_empty(), "at least one bucket compiled");
    for (file, report) in &results {
        assert!(
            report.stats.completed,
            "{file}: simulator did not complete — deadlock!\n{}",
            report.summary()
        );
        assert!(
            report.unsatisfiable.is_empty(),
            "{file}: unsatisfiable counters detected:\n{}",
            report.summary()
        );
        assert!(report.stats.makespan > 0, "{file}: zero makespan",);
        assert_eq!(
            report.events.len(),
            report.stats.executed,
            "{file}: event count mismatch",
        );
        assert_eq!(
            report.stats.executed,
            report.stats.total,
            "{file}: not all packets fired ({} / {})\n{}",
            report.stats.executed,
            report.stats.total,
            report.summary()
        );
        // Every event must have a valid timing window.
        for e in &report.events {
            assert!(
                e.t_end >= e.t_start,
                "{file}: event #{} has inverted timing",
                e.seq
            );
        }
    }
}

/// §P `--emit-sample` injects a host SAMPLE packet at the decode tail, gated on
/// the output-stage counter. Verify the compiled decode `.pkt` contains it,
/// that it's genuinely gated on real producers, and that the schedule still
/// simulates deadlock-free with the SAMPLE firing after the compute.
#[test]
fn emit_sample_injects_gated_host_packet() {
    use packet::{Body, Opcode, ResourceKind};
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: AtomicU64 = AtomicU64::new(0);
    let id = C.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowc-emit-sample-{}-{id}", std::process::id()));

    let net = load_net(include_str!("../examples/mlp_block.json"));
    let mut o = opts(dir.clone());
    o.phases = vec![Phase::Decode];
    o.batches = vec![1];
    o.seqs = vec![128];
    o.emit_sample = true;
    let report = compile(&Source::Net(net), &o).expect("compile");

    assert!(!report.buckets.is_empty());
    for bucket in &report.buckets {
        let bytes = std::fs::read(dir.join(&bucket.packet_file)).unwrap();
        let prog = packet::Program::decode(&bytes).unwrap();

        let sample = prog
            .insts
            .iter()
            .find(|i| matches!(i.body, Body::Token { kind, .. } if kind == Opcode::TOKEN_SAMPLE_GREEDY))
            .expect("decode bucket contains a SAMPLE host packet");
        assert!(
            matches!(sample.resource, ResourceKind::Host),
            "SAMPLE runs on Host"
        );
        assert!(!sample.wait.is_empty(), "SAMPLE is counter-gated");

        // The gate is a real dependency: some compute packet increments it.
        let wc = sample.wait[0];
        let producers = prog.insts.iter().filter(|i| i.succ.contains(&wc)).count();
        assert!(
            producers > 0,
            "SAMPLE's wait counter has {producers} producers"
        );

        // Simulate: completes, and a SAMPLE event fires after the producers.
        let sim = Simulator::new(MathMode::DryRun).run(&prog);
        assert!(
            sim.stats.completed,
            "sample-tailed schedule deadlocked:\n{}",
            sim.summary()
        );
        let sample_ev = sim
            .events
            .iter()
            .find(|e| e.name == "SAMPLE")
            .expect("SAMPLE event");
        let last_producer_end = sim
            .events
            .iter()
            .filter(|e| e.succs.contains(&wc))
            .map(|e| e.t_end)
            .max()
            .unwrap_or(0);
        assert!(
            sample_ev.t_start >= last_producer_end,
            "SAMPLE starts after its producers"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The small MLP block exercises the same pipeline with a simpler network.
#[test]
fn mlp_block_compiles_and_simulates_deadlock_free() {
    let net = load_net(include_str!("../examples/mlp_block.json"));
    let results = compile_and_simulate(net);

    assert!(!results.is_empty());
    for (file, report) in &results {
        assert!(
            report.stats.completed,
            "{file}: deadlock!\n{}",
            report.summary()
        );
        assert!(
            report.unsatisfiable.is_empty(),
            "{file}: unsatisfiable counters:\n{}",
            report.summary()
        );
        assert!(report.stats.makespan > 0);
        assert_eq!(report.stats.executed, report.stats.total);
    }
}

/// SwiGLU MLP (Llama-3 style) — tests the non-attention path at scale.
#[test]
fn swiglu_mlp_compiles_and_simulates_deadlock_free() {
    let net = load_net(include_str!("../examples/mlp_swiglu_llama3_8b.json"));
    let results = compile_and_simulate(net);

    assert!(!results.is_empty());
    for (file, report) in &results {
        assert!(
            report.stats.completed,
            "{file}: deadlock!\n{}",
            report.summary()
        );
        assert!(report.unsatisfiable.is_empty());
        assert_eq!(report.stats.executed, report.stats.total);
    }
}

/// Dependencies must be respected: every consumer starts after its producer ends.
#[test]
fn timing_respects_counter_dependencies() {
    let net = load_net(include_str!("../examples/mlp_block.json"));
    let results = compile_and_simulate(net);

    for (file, report) in &results {
        // For each event that waits on counters, find the producer events for
        // those counters and verify the consumer starts no earlier.
        for consumer in &report.events {
            for &counter_id in &consumer.waits {
                // Find all producers that increment this counter.
                let producers: Vec<_> = report
                    .events
                    .iter()
                    .filter(|e| e.succs.contains(&counter_id))
                    .collect();
                for producer in &producers {
                    assert!(
                        consumer.t_start >= producer.t_end,
                        "{file}: event #{} (wait c{counter_id}) starts at {} but producer #{} \
                         ends at {} — dependency violated!",
                        consumer.seq,
                        consumer.t_start,
                        producer.seq,
                        producer.t_end,
                    );
                }
            }
        }
    }
}

/// Chrome trace output from the simulation must be valid JSON and contain the
/// right number of events (one per packet fired).
#[test]
fn chrome_trace_from_real_network_is_well_formed() {
    let net = load_net(include_str!("../examples/mlp_block.json"));
    let results = compile_and_simulate(net);

    for (file, report) in &results {
        let json = report.to_chrome_json();
        let v: serde_json::Value = serde_json::from_str(&json).expect("chrome JSON must parse");
        let events = v["traceEvents"]
            .as_array()
            .expect("traceEvents must be array");
        assert_eq!(
            events.len(),
            report.events.len(),
            "{file}: chrome trace event count mismatch"
        );
        // Each span is a complete-duration event.
        for ev in events {
            assert_eq!(ev["ph"], "X", "{file}: unexpected event phase");
            assert!(ev["dur"].as_u64().is_some(), "{file}: missing dur");
        }
    }
}

// ─── Full runtime execution path ─────────────────────────────────────────────
//
// These tests exercise the real runtime stack: compile → write to disk →
// ModelBundle::load → Registry → AppState::generate, proving the compiled
// assets are loadable and executable by the CPU reference interpreter.

/// Compile a real network, write assets to disk, load via the runtime's
/// ModelBundle, register in the Registry, and run a request through the full
/// CPU backend interpreter path.
#[test]
fn llama3_full_runtime_load_and_execute() {
    let net = load_net(include_str!("../examples/transformer_block_llama3_8b.json"));
    let dir = std::env::temp_dir().join(format!("plowc-rt-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // Compile with a decode bucket so the runtime's generate() path can find one.
    let o = Options {
        no_tuning: false,
        tuning_db: None,
        gpu: "H100 SXM5".into(),
        num_gpus: 1,
        parallel: Parallel::Tp,
        batches: vec![1],
        seqs: vec![128],
        phases: vec![Phase::Prefill, Phase::Decode],
        page_kib: 16,
        out: dir.clone(),
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
    };
    let report = compile(&Source::Net(net), &o).expect("compile");
    assert!(!report.buckets.is_empty());

    // --- Runtime path ---
    let bundle = ModelBundle::load(&dir).expect("ModelBundle::load");
    assert_eq!(bundle.network(), "transformer-block-llama3-8b");
    assert_eq!(bundle.bucket_count(), report.buckets.len());

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());

    let mut registry = Registry::new();
    registry.load(&dir, None).unwrap();
    assert!(registry.slugs().any(|s| s == "transformer-block-llama3-8b"));

    let state = AppState::new(registry, execset);
    let gen = plowrt::serve::GenParams {
        max_tokens: 8,
        ..Default::default()
    };
    let (text, executed) = state
        .generate("transformer-block-llama3-8b", "hello world", &gen)
        .expect("generate must succeed — no deadlock");

    // The CPU backend ran the full schedule to completion, producing tokens.
    assert!(executed > 0, "no instructions executed");
    assert!(!text.is_empty(), "produced detokenized output");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Same full-runtime path with the MLP block — smaller, faster, validates the
/// pipeline works across different network shapes.
#[test]
fn mlp_block_full_runtime_load_and_execute() {
    let net = load_net(include_str!("../examples/mlp_block.json"));
    let dir = std::env::temp_dir().join(format!("plowc-rt-mlp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let o = Options {
        no_tuning: false,
        tuning_db: None,
        gpu: "H100 SXM5".into(),
        num_gpus: 1,
        parallel: Parallel::Tp,
        batches: vec![1],
        seqs: vec![64],
        phases: vec![Phase::Decode],
        page_kib: 16,
        out: dir.clone(),
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
    };
    let report = compile(&Source::Net(net), &o).expect("compile");
    assert_eq!(report.buckets.len(), 1);

    // Load + run through the full runtime.
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let mut registry = Registry::new();
    registry.load(&dir, None).unwrap();

    let state = AppState::new(registry, execset);
    let gen = plowrt::serve::GenParams {
        max_tokens: 8,
        ..Default::default()
    };
    let (_, executed) = state
        .generate("mlp-block", "test prompt", &gen)
        .expect("generate must not deadlock");
    assert!(executed > 0);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify that the runtime can load the compiled assets and run every bucket
/// individually through the Scheduler (not just the generate path which picks
/// one bucket).
#[test]
fn all_buckets_run_through_scheduler() {
    use plowrt::sched::Scheduler;

    let net = load_net(include_str!("../examples/transformer_block_llama3_8b.json"));
    let dir = std::env::temp_dir().join(format!("plowc-sched-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let o = Options {
        no_tuning: false,
        tuning_db: None,
        gpu: "H100 SXM5".into(),
        num_gpus: 1,
        parallel: Parallel::Tp,
        batches: vec![1, 4],
        seqs: vec![128, 512],
        phases: vec![Phase::Prefill],
        page_kib: 16,
        out: dir.clone(),
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
    };
    compile(&Source::Net(net), &o).expect("compile");

    let bundle = ModelBundle::load(&dir).expect("load");
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let scheduler = Scheduler::new(&execset);

    // Run every compiled bucket individually.
    let mut ran = 0;
    for key in bundle.bucket_keys() {
        let bucket = bundle.bucket(key).unwrap();
        let outcome = scheduler
            .run_bucket(bucket)
            .unwrap_or_else(|e| panic!("bucket {:?} deadlocked: {e}", key));
        assert!(outcome.completed);
        assert!(outcome.executed > 0);
        ran += 1;
    }
    assert_eq!(ran, 4, "expected 4 buckets (2 batches × 2 seqs × 1 phase)");

    let _ = std::fs::remove_dir_all(&dir);
}

/// GLM-5.2 MoE decoder block (MLA + router + routed expert + shared expert):
/// compile → simulate → assert no deadlock, positive makespan, all packets fire.
#[test]
fn glm5_transformer_block_compiles_and_simulates_deadlock_free() {
    let net = load_net(include_str!("../examples/transformer_block_glm5.json"));
    let results = compile_and_simulate(net);

    assert!(!results.is_empty(), "at least one bucket compiled");
    for (file, report) in &results {
        assert!(
            report.stats.completed,
            "{file}: simulator did not complete — deadlock!\n{}",
            report.summary()
        );
        assert!(
            report.unsatisfiable.is_empty(),
            "{file}: unsatisfiable counters detected:\n{}",
            report.summary()
        );
        assert!(report.stats.makespan > 0, "{file}: zero makespan",);
        assert_eq!(
            report.events.len(),
            report.stats.executed,
            "{file}: event count mismatch",
        );
        assert_eq!(
            report.stats.executed,
            report.stats.total,
            "{file}: not all packets fired ({} / {})\n{}",
            report.stats.executed,
            report.stats.total,
            report.summary()
        );
        // Every event must have a valid timing window.
        for e in &report.events {
            assert!(
                e.t_end >= e.t_start,
                "{file}: event #{} has inverted timing",
                e.seq
            );
        }
    }
}
