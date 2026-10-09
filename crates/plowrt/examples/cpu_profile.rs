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
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let mut opts = CpuEngineOpts::default();
    let mut n_prompt = 64usize;
    let mut do_prefill = false;
    let mut batch = 1usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--batch" => batch = args.next().unwrap().parse().unwrap(),
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
    println!(
        "engine: isa={:?} threads={} n_cu={}",
        eng.isa,
        eng.threads,
        eng.model().blob.n_cu
    );

    let report = |title: &str,
                  evs: &[TraceEv],
                  insts: &[packet::dev::DevInst64],
                  wall_ms: f64,
                  threads: usize| {
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
        // PROF_DUMP=<path>: every packet as CSV (inst, op, slice, worker, t0_ns, t1_ns, t2_ns = after the successor bumps, i0, i1, i2).
        if let Ok(p) = std::env::var("PROF_DUMP") {
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{p}.{}", title.replace(' ', "_"))).unwrap());
            writeln!(f, "inst,op,slice,worker,t0_ns,t1_ns,t2_ns,i0,i1,i2").unwrap();
            for e in evs {
                let d = &insts[e.inst as usize];
                let op = DevOp::from_u16(d.op).map(|o| o.c_name()).unwrap_or("?");
                writeln!(f, "{},{op},{},{},{},{},{},{},{},{}", e.inst, e.slice, e.worker, e.t0_ns, e.t1_ns, e.t2_ns, d.i[0], d.i[1], d.i[2]).unwrap();
            }
        }
        let traced_ms = (t_max.saturating_sub(t_min)) as f64 / 1e6;
        println!(
            "\n== {title}: wall {wall_ms:.1} ms, traced span {traced_ms:.1} ms, {} packets",
            evs.len()
        );
        println!(
            "{:<28} {:>7} {:>10} {:>10} {:>9}",
            "op", "packets", "busy ms", "busy/thr", "span ms"
        );
        // PROF_TOP=n: the n instructions with the most busy time (index, op, packets, busy, max packet).
        if let Some(n) = std::env::var("PROF_TOP").ok().and_then(|v| v.parse::<usize>().ok()) {
            let mut per_inst: BTreeMap<u32, (usize, u64, u64, u64, u64)> = BTreeMap::new();
            for e in evs {
                let d = e.t1_ns - e.t0_ns;
                let ent = per_inst.entry(e.inst).or_insert((0, 0, 0, u64::MAX, 0));
                ent.0 += 1;
                ent.1 += d;
                ent.2 = ent.2.max(d);
                ent.3 = ent.3.min(e.t0_ns);
                ent.4 = ent.4.max(e.t1_ns);
            }
            let mut top: Vec<_> = per_inst.iter().collect();
            top.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
            let t_first = evs.iter().map(|e| e.t0_ns).min().unwrap_or(0);
            for (i, (cnt, busy, mx, a, z)) in top.into_iter().take(n) {
                let op = insts[*i as usize].op;
                let d = &insts[*i as usize];
                println!(
                    "  #{i:<5} {:<26} {cnt:>4} pk  busy {:>9.2} ms  max pk {:>7.3} ms  window {:>8.2}..{:>8.2} ms  i {}x{}x{}",
                    DevOp::from_u16(op).map(|o| o.c_name()).unwrap_or("?"),
                    *busy as f64 / 1e6,
                    *mx as f64 / 1e6,
                    (*a - t_first) as f64 / 1e6,
                    (*z - t_first) as f64 / 1e6,
                    d.i[0],
                    d.i[1],
                    d.i[2]
                );
            }
        }
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
        // Attribute by the bucket a whole-prompt prefill runs: the narrowest covering the prompt,
        // else the widest (chunked). Instruction indices differ per bucket, so any other program
        // mislabels every event.
        let progs = eng.model().prefill_progs();
        let p = progs
            .iter()
            .filter(|p| p.t as usize >= ids.len())
            .min_by_key(|p| p.t)
            .or_else(|| progs.iter().max_by_key(|p| p.t))
            .expect("prefill program");
        let insts = p.insts.clone();
        report("prefill", &evs, &insts, wall, eng.threads);
    }
    let first = eng.prefill(&ids).expect("prefill");
    let mut pos = ids.len() as u32;
    if batch > 1 {
        // Timing-only: every slot is stepped at the same position with the same token (only
        // slot 0 holds a real KV block; the rest attend over whatever their block contains).
        let b = eng.batch();
        let bb = batch.min(b);
        let mut pos_v = vec![0u32; b];
        let mut kv_v = vec![1u32; b];
        let id_v = vec![first; b];
        for s in 0..bb {
            pos_v[s] = pos;
            kv_v[s] = pos + 1;
        }
        let dp = eng.model().decode_prog_for(bb);
        let _ = eng
            .decode_step_batched_at(&pos_v, &kv_v, &id_v, dp)
            .expect("warm decode");
        for s in 0..bb {
            pos_v[s] += 1;
            kv_v[s] += 1;
        }
        trace_begin();
        let t = Instant::now();
        let out = eng
            .decode_step_batched_at(&pos_v, &kv_v, &id_v, dp)
            .expect("decode");
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let evs = trace_take();
        let insts = eng.model().blob.progs[dp].insts.clone();
        report(
            &format!("decode step B={bb} (rung program {dp})"),
            &evs,
            &insts,
            wall,
            eng.threads,
        );
        println!("tokens: first {first} next {:?}", &out[..bb]);
        return;
    }
    // PROF_WAITS=i,j,..: the wait list (counter id, threshold) of slice 0/1 of those decode instructions, and
    // which instructions' entries bump each counter (entry count).
    if let Ok(list) = std::env::var("PROF_WAITS") {
        let p = eng.model().decode_prog();
        for want in list.split(',').filter_map(|s| s.parse::<u32>().ok()) {
            let op = DevOp::from_u16(p.insts[want as usize].op).map(|o| o.c_name()).unwrap_or("?");
            for e in p.stream.iter().filter(|e| e.inst == want && e.slice < 2) {
                let ew = &p.waits[e.wait_ofs as usize..e.wait_ofs as usize + e.wait_len as usize];
                println!("inst {want} {op} slice {} flags {:#x} waits {:?}", e.slice, e.flags, ew.iter().map(|w| (w.id, w.threshold)).collect::<Vec<_>>());
                for w in ew {
                    let mut by: BTreeMap<u32, usize> = BTreeMap::new();
                    for f in p.stream.iter() {
                        if p.succs[f.succ_ofs as usize..f.succ_ofs as usize + f.succ_len as usize].contains(&w.id) {
                            *by.entry(f.inst).or_default() += 1;
                        }
                    }
                    println!("  counter {} thr {} bumped by inst:entries {:?}", w.id, w.threshold, by);
                }
            }
        }
    }
    let _ = eng.decode_step(pos, pos + 1).expect("warm decode");
    pos += 1;
    trace_begin();
    let t = Instant::now();
    let next = eng.decode_step(pos, pos + 1).expect("decode");
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let evs = trace_take();
    let insts = eng.model().decode_prog().insts.clone();
    report("decode step", &evs, &insts, wall, eng.threads);
    println!(
        "tokens: first {first} next {next} {:?}",
        tok.decode(&[first, next])
    );
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}
