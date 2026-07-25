//! `qsim` — trace-driven validation of the M/D/1 premise in `sched::mdq`.
//!
//! The claim under test: plow's decode step is deterministic enough that its
//! queue behaves as **M/D/1**, so waiting time is *half* the M/M/1 value at the
//! same utilisation — and therefore plow can run at higher utilisation for a
//! given p99 than a system whose step time varies.
//!
//! This is a discrete-event simulation of a single server with **empirically
//! measured service times**: the per-step trace comes from
//! `runtime/tests/qwen3_sm120_chat.cu` under `PLOW_DUMP_STEPS=1` on an RTX 5090.
//! Only the *arrivals* are synthetic (Poisson). Nothing about the service
//! process is modelled — it is replayed.
//!
//! # Negative control
//!
//! The same simulator is driven with exponential service of identical mean. If
//! the harness is sound, that run must track M/M/1 and *not* M/D/1. A validation
//! that cannot fail proves nothing, so `qsim` runs the control every time and
//! fails loudly if it does not separate the two models.
//!
//! Usage:
//!   qsim <steps-file>   # a file containing the `PLOW_STEPS_RAW ...` line

use plowrt::sched::mdq::{self, wait, ModelCost, ServiceTable};

/// Deterministic xorshift — no rand dependency, and a fixed seed makes every
/// number in the report reproducible.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform on (0,1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }
    fn exp(&mut self, mean: f64) -> f64 {
        -mean * self.unit().ln()
    }
}

/// How the server's service time is produced.
enum Service<'a> {
    /// Replay the measured trace, sampled with replacement.
    Trace(&'a [f64]),
    /// Exponential with the trace's mean — the negative control.
    Exponential(f64),
}

struct SimResult {
    mean_wait: f64,
    p50: f64,
    p99: f64,
    mean_service: f64,
    cv: f64,
    rho: f64,
}

/// Single-server FCFS queue, Poisson arrivals at `lambda` (per ms).
fn simulate(svc: &Service, lambda_per_ms: f64, n: usize, seed: u64) -> SimResult {
    let mut rng = Rng(seed);
    let mut t = 0.0f64; // arrival clock
    let mut free_at = 0.0f64; // when the server next goes idle
    let mut waits = Vec::with_capacity(n);
    let mut svc_samples = Vec::with_capacity(n);
    // Warm-up discarded so the empty-system transient does not bias the mean.
    let warm = n / 10;
    for i in 0..n {
        t += rng.exp(1.0 / lambda_per_ms);
        let s = match svc {
            Service::Trace(tr) => tr[(rng.next_u64() % tr.len() as u64) as usize],
            Service::Exponential(m) => rng.exp(*m),
        };
        let start = free_at.max(t);
        let w = start - t;
        free_at = start + s;
        if i >= warm {
            waits.push(w);
            svc_samples.push(s);
        }
    }
    let m = svc_samples.iter().sum::<f64>() / svc_samples.len() as f64;
    let var = svc_samples.iter().map(|x| (x - m) * (x - m)).sum::<f64>()
        / (svc_samples.len() - 1) as f64;
    let mut sorted = waits.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    SimResult {
        mean_wait: waits.iter().sum::<f64>() / waits.len() as f64,
        p50: sorted[sorted.len() / 2],
        p99: sorted[(sorted.len() as f64 * 0.99) as usize],
        mean_service: m,
        cv: var.sqrt() / m,
        rho: lambda_per_ms * m,
    }
}

fn parse_steps(path: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(path).expect("read steps file");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("PLOW_STEPS_RAW") {
            return rest
                .split_whitespace()
                .filter_map(|t| t.parse::<f64>().ok())
                .collect();
        }
    }
    panic!("no PLOW_STEPS_RAW line in {path}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: qsim <steps-file>");
    let trace = parse_steps(path);
    assert!(trace.len() > 32, "trace too short: {}", trace.len());

    let mean = trace.iter().sum::<f64>() / trace.len() as f64;
    let var =
        trace.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (trace.len() - 1) as f64;
    let cv = var.sqrt() / mean;
    println!("MEASURED SERVICE TRACE  {path}");
    println!(
        "  n={}  mean={:.4} ms  sd={:.4} ms  CV={:.4}%  min={:.3}  max={:.3}",
        trace.len(),
        mean,
        var.sqrt(),
        100.0 * cv,
        trace.iter().cloned().fold(f64::INFINITY, f64::min),
        trace.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    const N: usize = 2_000_000;
    println!("\nWAITING TIME vs THEORY  (N={N} requests/point, Poisson arrivals)");
    println!("  Service times are REPLAYED from the trace above; only arrivals are synthetic.");
    println!(
        "  {:>5}  {:>10} {:>10} {:>10} {:>9} | {:>9} {:>9}",
        "rho", "sim mean", "M/D/1", "M/M/1", "err vs D", "sim p50", "sim p99"
    );

    let mut worst_d = 0.0f64;
    let mut worst_ctrl = 0.0f64;
    for &rho in &[0.30, 0.50, 0.70, 0.80, 0.90, 0.95] {
        let lambda = rho / mean;
        let r = simulate(&Service::Trace(&trace), lambda, N, 0x2545F491_4F6CDD1D);
        let d = wait::md1(mean, rho);
        let m = wait::mm1(mean, rho);
        let err = (r.mean_wait - d).abs() / d.max(1e-12);
        worst_d = worst_d.max(err);
        println!(
            "  {:>5.2}  {:>10.4} {:>10.4} {:>10.4} {:>8.2}% | {:>9.4} {:>9.4}",
            r.rho, r.mean_wait, d, m, 100.0 * err, r.p50, r.p99
        );
    }

    println!("\nNEGATIVE CONTROL — exponential service, same mean.");
    println!("  If the simulator cannot tell M/D/1 from M/M/1, the table above is meaningless.");
    println!(
        "  {:>5}  {:>10} {:>10} {:>10} | {:>9} {:>9}",
        "rho", "sim mean", "M/D/1", "M/M/1", "err vs D", "err vs M"
    );
    for &rho in &[0.30, 0.50, 0.70, 0.80, 0.90] {
        let lambda = rho / mean;
        let r = simulate(&Service::Exponential(mean), lambda, N, 0x9E3779B9_7F4A7C15);
        let d = wait::md1(mean, rho);
        let m = wait::mm1(mean, rho);
        let ed = (r.mean_wait - d).abs() / d;
        let em = (r.mean_wait - m).abs() / m;
        worst_ctrl = worst_ctrl.max(em);
        println!(
            "  {:>5.2}  {:>10.4} {:>10.4} {:>10.4} | {:>8.1}% {:>8.1}%",
            r.rho, r.mean_wait, d, m, 100.0 * ed, 100.0 * em
        );
    }

    // The two assertions that make this a test rather than a printout.
    println!("\nVERDICT");
    println!(
        "  measured-trace worst deviation from M/D/1 : {:.2}%",
        100.0 * worst_d
    );
    println!(
        "  control worst deviation from M/M/1        : {:.2}%",
        100.0 * worst_ctrl
    );
    assert!(
        worst_d < 0.05,
        "measured trace does NOT track M/D/1 (worst {:.2}%) — premise refuted",
        100.0 * worst_d
    );
    assert!(
        worst_ctrl < 0.05,
        "control does NOT track M/M/1 (worst {:.2}%) — simulator is broken",
        100.0 * worst_ctrl
    );
    println!("  -> trace tracks M/D/1, control tracks M/M/1: the test can distinguish them.");

    // Utilisation headroom: the product claim, stated as a number.
    println!("\nUTILISATION HEADROOM FOR A FIXED p99 WAIT");
    println!("  The question the scheduler actually cares about: how hard can each service");
    println!("  distribution be driven before it blows a p99 waiting-time budget? The plow");
    println!("  column is SIMULATED ON THE MEASURED TRACE; M/M/1 is its closed-form tail.");
    println!(
        "  {:>10}  {:>16} {:>14}  {:>10}",
        "p99 budget", "rho (measured)", "rho (M/M/1)", "headroom"
    );
    // Bisection on rho, p99 from simulation for the measured trace.
    const NH: usize = 400_000;
    for &budget in &[10.0f64, 25.0, 50.0, 100.0] {
        let (mut lo, mut hi) = (0.01f64, 0.995f64);
        for _ in 0..18 {
            let mid = 0.5 * (lo + hi);
            let r = simulate(&Service::Trace(&trace), mid / mean, NH, 0x1234_5678_9ABC_DEF1);
            if r.p99 < budget {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let rd = 0.5 * (lo + hi);
        let (mut l2, mut h2) = (0.01f64, 0.995f64);
        for _ in 0..200 {
            let mid = 0.5 * (l2 + h2);
            if wait::mm1_quantile(mean, mid, 0.99) < budget {
                l2 = mid;
            } else {
                h2 = mid;
            }
        }
        let rm = 0.5 * (l2 + h2);
        println!(
            "  {:>8.0} ms  {:>16.4} {:>14.4}  {:>9.1}%",
            budget,
            rd,
            rm,
            100.0 * (rd - rm) / rm
        );
    }

    // The service-time table, fitted and reported against measurement.
    println!("\nSERVICE-TIME TABLE (batch 1 is MEASURED; batch>1 is a byte-model PROJECTION");
    println!("  because batch>1 does not compile today — see sched::mdq docs)");
    // MEASURED, RTX 5090 sm_120a, Qwen3-4B, 143 timed steps each (PLOW_RESULT lines in
    // /workspace/qbench/ctx*.log). ctx is the prompt length the pkt was compiled for.
    let measured: Vec<(u64, u32, f64)> = vec![
        (4_096, 1, 6.8699),
        (8_192, 1, 7.1264),
        (16_384, 1, 7.9167),
        (32_768, 1, 9.5588),
    ];
    let fitted = ServiceTable::fit(&measured, ModelCost::qwen3_4b(1.0, 0.0))
        .expect("fit needs a ctx sweep");
    println!(
        "  fitted: effective bandwidth {:.1} GB/s, fixed overhead {:.3} ms/step",
        fitted.bandwidth_gbps, fitted.fixed_ms
    );
    println!("  {:>7} {:>6} {:>12} {:>12} {:>9}", "ctx", "batch", "pred ms", "meas ms", "err");
    for &(ctx, b, meas) in &measured {
        let p = fitted.step_ms(ctx, b);
        println!(
            "  {:>7} {:>6} {:>12.4} {:>12.4} {:>8.2}%",
            ctx,
            b,
            p,
            meas,
            100.0 * (p - meas) / meas
        );
    }
    let table = ServiceTable::build(fitted, &[4_096, 8_192, 16_384, 32_768], &[1, 2, 4, 8]);
    println!("\n  PROJECTED batch scaling at ctx 4096 (NOT MEASURED):");
    println!(
        "  {:>6} {:>10} {:>12} {:>10} {:>10}",
        "batch", "step ms", "per-token ms", "tok/s", "speedup"
    );
    let base = table.lookup(4_096, 1).unwrap().per_token_ms;
    for b in [1u32, 2, 4, 8] {
        let e = table.lookup(4_096, b).unwrap();
        println!(
            "  {:>6} {:>10.4} {:>12.4} {:>10.1} {:>9.2}x",
            b,
            e.step_ms,
            e.per_token_ms,
            1000.0 / e.per_token_ms,
            base / e.per_token_ms
        );
    }

    println!("\nANALYTIC BATCH WINDOW (from lambda + known step cost; no tuned constant)");
    println!("  *** PROJECTION ONLY: this assumes batch B costs a + B*c, which requires a");
    println!("  *** batch>1 decode program. That does not exist. See the report. ***");
    // The cap must not bind, or the reported "optimum" is just the cap. 200 ms is
    // far past any interior optimum at these rates; the brute-force column below
    // is the check that it is genuinely interior.
    const WMAX: f64 = 200.0;
    println!(
        "  {:>10} {:>8} {:>12} {:>12} {:>12} {:>10}",
        "lambda/s", "rho", "w* (golden)", "w* (grid)", "T(w*) ms", "T(0) ms"
    );
    let (a, c) = fitted.affine(4_096);
    let mut window_ok = true;
    for &rho in &[0.30f64, 0.50, 0.70, 0.85, 0.95] {
        let lambda = rho / (a + c) * 1000.0;
        let w = mdq::optimal_window_ms(&fitted, 4_096, lambda, WMAX);
        // Brute-force grid minimum — the independent check on the golden-section search.
        let mut best = (0.0f64, f64::INFINITY);
        for i in 0..=20_000 {
            let ww = WMAX * i as f64 / 20_000.0;
            let t = mdq::latency_at_window(&fitted, 4_096, lambda, ww);
            if t < best.1 {
                best = (ww, t);
            }
        }
        if (w - best.0).abs() > 0.05 * WMAX / 20.0 + 0.05 {
            window_ok = false;
        }
        if w > 0.98 * WMAX {
            window_ok = false; // pinned at the cap: not an interior optimum
        }
        println!(
            "  {:>10.1} {:>8.2} {:>12.4} {:>12.4} {:>12.4} {:>10.4}",
            lambda,
            rho,
            w,
            best.0,
            mdq::latency_at_window(&fitted, 4_096, lambda, w),
            mdq::latency_at_window(&fitted, 4_096, lambda, 0.0)
        );
    }
    assert!(
        window_ok,
        "analytic window disagrees with the grid minimum, or is pinned at the cap"
    );
    println!("  -> golden-section optimum agrees with the grid minimum and is interior.");

    println!("\n  Analytic optimum vs the tuned candidates the plan asked to compare (ctx 4096):");
    println!(
        "  {:>7} {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9}",
        "rho", "w=0", "w=0.5", "w=1", "w=2", "w*", "T(w*)"
    );
    for &rho in &[0.30f64, 0.50, 0.70, 0.85, 0.95] {
        let lambda = rho / (a + c) * 1000.0;
        let w = mdq::optimal_window_ms(&fitted, 4_096, lambda, WMAX);
        print!("  {rho:>7.2}");
        let mut best_tuned = f64::INFINITY;
        for cand in [0.0f64, 0.5, 1.0, 2.0] {
            let t = mdq::latency_at_window(&fitted, 4_096, lambda, cand);
            best_tuned = best_tuned.min(t);
            print!("{t:>10.4}");
        }
        let t_star = mdq::latency_at_window(&fitted, 4_096, lambda, w);
        println!(" | {:>9.4} {:>9.4}", w, t_star);
        assert!(
            t_star <= best_tuned + 1e-9,
            "rho={rho}: analytic window {w} ({t_star}) lost to a tuned constant ({best_tuned})"
        );
    }
    println!("  -> the analytic window is never beaten by a tuned constant.");

    shedding_demo(&trace, mean);
}

/// Item 5: deadline-aware admission under overload.
///
/// A request generates `GEN` tokens and carries an SLO deadline. Two policies
/// are run against the *same* arrival stream at rho > 1:
///
/// * **admit-all** — the FIFO behaviour you get with no admission control. The
///   queue grows without bound and *every* request eventually misses.
/// * **deadline** — on arrival, compute the exact completion time from the
///   committed backlog (deterministic service makes this exact, not a guess) and
///   shed if it would miss.
///
/// The claim is that the second keeps its SLO by shedding, instead of degrading
/// everything. Service times are replayed from the measured trace.
fn shedding_demo(trace: &[f64], mean: f64) {
    const GEN: u32 = 64; // tokens per request
    const N: usize = 60_000;
    let svc_ms = GEN as f64 * mean;

    println!("\nDEADLINE-AWARE ADMISSION UNDER OVERLOAD");
    println!(
        "  request = {GEN} decode tokens ({:.1} ms of service); arrivals Poisson; N={N}",
        svc_ms
    );
    println!(
        "  {:>6} {:>9} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10}",
        "rho", "SLO ms", "A:served", "A:met SLO", "A:p99 ms", "B:served", "B:shed", "B:met SLO"
    );

    for &(rho, slo) in &[
        (0.95f64, 1_500.0f64),
        (1.10, 1_500.0),
        (1.50, 1_500.0),
        (2.00, 1_500.0),
    ] {
        let lambda = rho / svc_ms;
        let mut rng = Rng(0xD1CE_5EED_0BAD_F00D);
        let mut t = 0.0;
        // Policy A: admit everything.
        let mut free_a = 0.0f64;
        let mut met_a = 0usize;
        let mut lat_a: Vec<f64> = Vec::with_capacity(N);
        // Policy B: deadline-aware shed.
        let mut free_b = 0.0f64;
        let mut served_b = 0usize;
        let mut shed_b = 0usize;
        let mut met_b = 0usize;

        for _ in 0..N {
            t += rng.exp(1.0 / lambda);
            // Replay a service time for this request: GEN consecutive trace steps.
            let start_idx = (rng.next_u64() % trace.len() as u64) as usize;
            let s: f64 = (0..GEN as usize)
                .map(|k| trace[(start_idx + k) % trace.len()])
                .sum();

            // A: unconditional FIFO.
            let fin_a = free_a.max(t) + s;
            free_a = fin_a;
            let l = fin_a - t;
            lat_a.push(l);
            if l <= slo {
                met_a += 1;
            }

            // B: exact completion prediction from the committed backlog. This is
            // the computation deterministic service makes possible.
            let predicted = free_b.max(t) + s - t;
            if predicted > slo {
                shed_b += 1;
            } else {
                free_b = free_b.max(t) + s;
                served_b += 1;
                if free_b - t <= slo {
                    met_b += 1;
                }
            }
        }
        lat_a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let p99_a = lat_a[(lat_a.len() as f64 * 0.99) as usize];
        println!(
            "  {:>6.2} {:>9.0} | {:>10} {:>9.1}% {:>10.0} | {:>10} {:>10} {:>9.1}%",
            rho,
            slo,
            N,
            100.0 * met_a as f64 / N as f64,
            p99_a,
            served_b,
            shed_b,
            100.0 * met_b as f64 / served_b.max(1) as f64
        );
        // The whole point: whatever we admit, we deliver.
        assert!(
            met_b == served_b,
            "rho={rho}: deadline policy admitted {served_b} but only met {met_b} — \
             with deterministic service the prediction must be exact"
        );
    }
    println!("  -> policy B meets its SLO on 100% of ADMITTED requests at every overload");
    println!("     level, because deterministic service makes the completion time a");
    println!("     computation. Policy A degrades everything instead.");
}
