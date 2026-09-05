//! Per-op / per-worker profile of one decode step (and optionally one prefill):
//! where the wall time goes — kernel busy time per op family, per-worker idle,
//! and the critical-path span of each op. Uses the interpreter's opt-in trace.
//!
//! `cargo run --release --features cpu --example cpu_profile -- <model.pkt> <ckpt> [--threads T] [--prompt-tokens N] [--prefill]`

#[cfg(feature = "cpu")]
fn main() {
    use packet::dev::DevOp;
    use plowrt::exec::cpu::engine::{CpuEngine, CpuEngineOpts};
    use plowrt::exec::cpu::interp::{trace_begin, trace_take, TraceEv};
    use plowrt::text::tokenizer::load_tokenizer;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::Instant;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let mut opts = CpuEngineOpts::default();
    let mut n_prompt = 64usize;
    let mut do_prefill = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => opts.threads = args.next().unwrap().parse().unwrap(),
            "--spin-us" => opts.spin_us = args.next().unwrap().parse().unwrap(),
            "--prompt-tokens" => n_prompt = args.next().unwrap().parse().unwrap(),
            "--prefill" => do_prefill = true,
            other => panic!("unknown arg {other}"),
        }
    }
    let tok = load_tokenizer(&ckpt);
    let base = tok.encode_with_special_tokens(
        "The mill's ledgers record more than flour: the weather on every delivery day, the price of candles, and the names of children hired to pick stones. ",
        false,
    );
    let mut ids = vec![2u32];
    while ids.len() < n_prompt {
        ids.extend_from_slice(&base);
    }
    ids.truncate(n_prompt);

    let mut eng = CpuEngine::load(&blob, &ckpt, &opts).expect("load");
    println!("engine: isa={:?} threads={} n_cu={}", eng.isa, eng.threads, eng.model().blob.n_cu);

    let report = |title: &str, evs: &[TraceEv], insts: &[packet::dev::DevInst64], wall_ms: f64, threads: usize| {
        // per op: count, busy sum, span (first start .. last end)
        let mut per_op: BTreeMap<u16, (usize, u64, u64, u64)> = BTreeMap::new();
        let mut per_worker = vec![0u64; threads.max(1)];
        let (mut t_min, mut t_max) = (u64::MAX, 0u64);
        for e in evs {
            let op = insts[e.inst as usize].op;
            let ent = per_op.entry(op).or_insert((0, 0, u64::MAX, 0));
            ent.0 += 1;
            ent.1 += e.t1_ns - e.t0_ns;
            ent.2 = ent.2.min(e.t0_ns);
            ent.3 = ent.3.max(e.t1_ns);
            if (e.worker as usize) < per_worker.len() {
                per_worker[e.worker as usize] += e.t1_ns - e.t0_ns;
            }
            t_min = t_min.min(e.t0_ns);
            t_max = t_max.max(e.t1_ns);
        }
        let traced_ms = (t_max.saturating_sub(t_min)) as f64 / 1e6;
        println!("\n== {title}: wall {wall_ms:.1} ms, traced span {traced_ms:.1} ms, {} packets", evs.len());
        println!("{:<28} {:>7} {:>10} {:>10} {:>9}", "op", "packets", "busy ms", "busy/thr", "span ms");
        let mut rows: Vec<_> = per_op.iter().collect();
        rows.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        for (op, (n, busy, t0, t1)) in rows {
            println!(
                "{:<28} {:>7} {:>10.2} {:>10.2} {:>9.2}",
                DevOp::from_u16(*op).map(|o| o.c_name()).unwrap_or("?"),
                n,
                *busy as f64 / 1e6,
                *busy as f64 / 1e6 / threads as f64,
                (*t1 - *t0) as f64 / 1e6
            );
        }
        let total_busy: u64 = per_worker.iter().sum();
        println!(
            "workers: busy mean {:.1} ms  min {:.1}  max {:.1}  => idle {:.0}% of wall",
            total_busy as f64 / 1e6 / threads as f64,
            *per_worker.iter().min().unwrap_or(&0) as f64 / 1e6,
            *per_worker.iter().max().unwrap_or(&0) as f64 / 1e6,
            100.0 * (1.0 - total_busy as f64 / 1e6 / threads as f64 / wall_ms.max(1e-9))
        );
    };

    if do_prefill {
        let _ = eng.prefill(&ids).expect("prefill warm");
        trace_begin();
        let t = Instant::now();
        let _ = eng.prefill(&ids).expect("prefill");
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let evs = trace_take();
        // Prefill may span several programs (chunks); attribute by the first prefill program.
        let insts = eng.model().prefill_progs()[0].insts.clone();
        report("prefill", &evs, &insts, wall, eng.threads);
    }
    let first = eng.prefill(&ids).expect("prefill");
    let mut pos = ids.len() as u32;
    let _ = eng.decode_step(pos, pos + 1).expect("warm decode");
    pos += 1;
    trace_begin();
    let t = Instant::now();
    let next = eng.decode_step(pos, pos + 1).expect("decode");
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let evs = trace_take();
    let insts = eng.model().decode_prog().insts.clone();
    report("decode step", &evs, &insts, wall, eng.threads);
    println!("tokens: first {first} next {next} {:?}", tok.decode(&[first, next]));
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}
