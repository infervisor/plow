//! Oracle A/B comparison: schedule Gemma 4 family transformer blocks with and
//! without the lean oracle pipeline (Phases 2–5) and compare makespans statically.
//!
//! This test does NOT require the Lean binary — the oracle's Rust fallback is
//! exercised (same arithmetic, no certificate). The test proves that:
//! 1. The oracle pipeline compiles cleanly on real model shapes.
//! 2. Bubble-fill never increases makespan (monotone improvement).
//! 3. The lower bound is always ≤ achieved makespan.
//! 4. Prefetch depth recommendations are in [1, max_depth].

use costmodel::{hwspec, Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use nn_graph::{infer_shapes, ActKind, DType, Nn};
use rewrite::{assemble, plan_from_block, LayerPlan};
use schedule::{schedule, schedule_with_oracle, Config, OracleReport};

fn mi350x() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("MI350X").unwrap()
}

// ─── Gemma 4 12B dimensions ─────────────────────────────────────────────────
mod gemma12b {
    pub const H: i64 = 3840;
    pub const NH: i64 = 16;
    pub const NKV: i64 = 8;
    pub const HD: i64 = 256;
    pub const QD: i64 = NH * HD; // 4096
    pub const KVD: i64 = NKV * HD; // 2048
    pub const IM: i64 = 21504; // SwiGLU intermediate
}

// ─── Gemma 4 27B (MoE) simplified as dense for scheduling ───────────────────
mod gemma27b {
    pub const H: i64 = 4608;
    pub const NH: i64 = 32;
    pub const NKV: i64 = 16;
    pub const HD: i64 = 128;
    pub const QD: i64 = NH * HD; // 4096
    pub const KVD: i64 = NKV * HD; // 2048
    pub const IM: i64 = 36864;
}

// ─── Gemma 4 31B (dense) ────────────────────────────────────────────────────
mod gemma31b {
    pub const H: i64 = 5120;
    pub const NH: i64 = 32;
    pub const NKV: i64 = 16;
    pub const HD: i64 = 128;
    pub const QD: i64 = NH * HD; // 4096
    pub const KVD: i64 = NKV * HD; // 2048
    pub const IM: i64 = 40960;
}

/// Build a Gemma-style decoder block (pre-norm, RoPE, GQA, SwiGLU).
fn gemma_block(
    h: i64,
    nh: i64,
    nkv: i64,
    hd: i64,
    qd: i64,
    kvd: i64,
    im: i64,
    t: i64,
) -> LayerPlan {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let x = nn.input("x", nn.shape([t.into(), h.into()]), DType::BF16);
    nn.begin_block("layers.0");

    let h1 = nn.rmsnorm("input_norm", x, h, 1e-6);
    let q = nn.linear("q_proj", h1, h, qd, false);
    let k = nn.linear("k_proj", h1, h, kvd, false);
    let v = nn.linear("v_proj", h1, h, kvd, false);
    let qh = nn.reshape(q, [t.into(), nh.into(), hd.into()]);
    let kh = nn.reshape(k, [t.into(), nkv.into(), hd.into()]);
    let vh = nn.reshape(v, [t.into(), nkv.into(), hd.into()]);
    let qn = nn.rmsnorm("q_norm", qh, hd, 1e-6);
    let kn = nn.rmsnorm("k_norm", kh, hd, 1e-6);
    let qr = nn.rope(qn, hd as u32, 1e6);
    let kr = nn.rope(kn, hd as u32, 1e6);
    let attn = nn.attention(
        qr, kr, vh, nh as u32, nkv as u32, hd as u32, true, None, None,
    );
    let ao = nn.reshape(attn, [t.into(), qd.into()]);
    let o = nn.linear("o_proj", ao, qd, h, false);
    let r1 = nn.add(x, o);

    let h2 = nn.rmsnorm("post_norm", r1, h, 1e-6);
    let gate = nn.linear("gate_proj", h2, h, im, false);
    let up = nn.linear("up_proj", h2, h, im, false);
    let ga = nn.act(ActKind::Silu, gate);
    let gu = nn.mul(ga, up);
    let down = nn.linear("down_proj", gu, im, h, false);
    let out = nn.add(r1, down);

    nn.end_block();
    nn.mark_output(out);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("infer");
    plan_from_block(&g, 0).expect("plan")
}

/// Run the A/B comparison: schedule with default config (no oracle) then with
/// oracle. Return (baseline_makespan, oracle_makespan, report).
fn compare(plan: &LayerPlan) -> (u64, u64, OracleReport) {
    let soc = Soc::single(mi350x(), DEFAULT_PAGE_BYTES);
    let (tile_g, cons) = assemble(&soc, plan, SramPolicy::Stream, None).expect("assemble");

    let baseline = schedule(&soc, &tile_g, &cons, &Config::default());
    let with_oracle = schedule_with_oracle(&soc, &tile_g, &cons, &Config::default());

    let baseline_ms = baseline.schedule.makespan;
    let oracle_ms = with_oracle.schedule.makespan;
    let report = with_oracle.oracle_report.expect("oracle report present");

    (baseline_ms, oracle_ms, report)
}

fn validate_report(baseline_ms: u64, oracle_ms: u64, report: &OracleReport, model: &str) {
    // 1. Oracle never increases makespan (bubble-fill is monotone).
    assert!(
        oracle_ms <= baseline_ms,
        "{model}: oracle increased makespan! baseline={baseline_ms}, oracle={oracle_ms}"
    );

    // 2. Lower bound is always ≤ achieved makespan.
    assert!(
        report.lower_bound.bound <= oracle_ms,
        "{model}: lower bound ({}) exceeds achieved makespan ({oracle_ms})",
        report.lower_bound.bound
    );

    // 3. Prefetch depth recommendations are in [1, 8].
    for (i, pd) in report.prefetch_depths.iter().enumerate() {
        assert!(
            pd.depth >= 1 && pd.depth <= 8,
            "{model}: unit {i} prefetch depth {} out of range [1,8]",
            pd.depth
        );
    }

    // 4. Optimality gap is non-negative.
    assert!(
        report.optimality_gap >= 0.0,
        "{model}: negative optimality gap: {}",
        report.optimality_gap
    );

    // Print results for static comparison.
    let improvement_pct = if baseline_ms > 0 {
        (baseline_ms as f64 - oracle_ms as f64) / baseline_ms as f64 * 100.0
    } else {
        0.0
    };
    eprintln!("  {model}:");
    eprintln!("    baseline makespan:  {baseline_ms} cycles");
    eprintln!("    oracle makespan:    {oracle_ms} cycles");
    eprintln!("    improvement:        {improvement_pct:.2}%");
    eprintln!(
        "    lower bound:        {} cycles",
        report.lower_bound.bound
    );
    eprintln!("    binding constraint: {:?}", report.lower_bound.binding);
    eprintln!(
        "    optimality gap:     {:.2}%",
        report.optimality_gap * 100.0
    );
    eprintln!(
        "    bubbles filled:     {}",
        report.bubble_fill.bubbles_filled
    );
    eprintln!(
        "    cycles recovered:   {}",
        report.bubble_fill.cycles_recovered
    );
    eprintln!(
        "    prefetch depths:    {:?}",
        report
            .prefetch_depths
            .iter()
            .map(|p| p.depth)
            .collect::<Vec<_>>()
    );
}

// ─── Tests ──────────────────────────────────────────────────────────────────

/// Decode shape: batch=1, seq=1.
#[test]
fn gemma_12b_decode_b1() {
    let plan = gemma_block(
        gemma12b::H,
        gemma12b::NH,
        gemma12b::NKV,
        gemma12b::HD,
        gemma12b::QD,
        gemma12b::KVD,
        gemma12b::IM,
        1,
    );
    let (baseline, oracle, report) = compare(&plan);
    validate_report(baseline, oracle, &report, "gemma4-12b decode b1");
}

/// Prefill shape: batch=1, seq=512.
#[test]
fn gemma_12b_prefill_512() {
    let plan = gemma_block(
        gemma12b::H,
        gemma12b::NH,
        gemma12b::NKV,
        gemma12b::HD,
        gemma12b::QD,
        gemma12b::KVD,
        gemma12b::IM,
        512,
    );
    let (baseline, oracle, report) = compare(&plan);
    validate_report(baseline, oracle, &report, "gemma4-12b prefill s512");
}

/// Decode batch=8.
#[test]
fn gemma_12b_decode_b8() {
    let plan = gemma_block(
        gemma12b::H,
        gemma12b::NH,
        gemma12b::NKV,
        gemma12b::HD,
        gemma12b::QD,
        gemma12b::KVD,
        gemma12b::IM,
        8,
    );
    let (baseline, oracle, report) = compare(&plan);
    validate_report(baseline, oracle, &report, "gemma4-12b decode b8");
}

/// Gemma 27B (MoE-sized dense) decode.
#[test]
fn gemma_27b_decode_b1() {
    let plan = gemma_block(
        gemma27b::H,
        gemma27b::NH,
        gemma27b::NKV,
        gemma27b::HD,
        gemma27b::QD,
        gemma27b::KVD,
        gemma27b::IM,
        1,
    );
    let (baseline, oracle, report) = compare(&plan);
    validate_report(baseline, oracle, &report, "gemma4-27b decode b1");
}

/// Gemma 27B prefill.
#[test]
fn gemma_27b_prefill_512() {
    let plan = gemma_block(
        gemma27b::H,
        gemma27b::NH,
        gemma27b::NKV,
        gemma27b::HD,
        gemma27b::QD,
        gemma27b::KVD,
        gemma27b::IM,
        512,
    );
    let (baseline, oracle, report) = compare(&plan);
    validate_report(baseline, oracle, &report, "gemma4-27b prefill s512");
}

/// Gemma 31B decode.
#[test]
fn gemma_31b_decode_b1() {
    let plan = gemma_block(
        gemma31b::H,
        gemma31b::NH,
        gemma31b::NKV,
        gemma31b::HD,
        gemma31b::QD,
        gemma31b::KVD,
        gemma31b::IM,
        1,
    );
    let (baseline, oracle, report) = compare(&plan);
    validate_report(baseline, oracle, &report, "gemma4-31b decode b1");
}

/// Gemma 31B prefill.
#[test]
fn gemma_31b_prefill_512() {
    let plan = gemma_block(
        gemma31b::H,
        gemma31b::NH,
        gemma31b::NKV,
        gemma31b::HD,
        gemma31b::QD,
        gemma31b::KVD,
        gemma31b::IM,
        512,
    );
    let (baseline, oracle, report) = compare(&plan);
    validate_report(baseline, oracle, &report, "gemma4-31b prefill s512");
}

/// Long prefill: Gemma 12B at seq=2048.
#[test]
fn gemma_12b_prefill_2048() {
    let plan = gemma_block(
        gemma12b::H,
        gemma12b::NH,
        gemma12b::NKV,
        gemma12b::HD,
        gemma12b::QD,
        gemma12b::KVD,
        gemma12b::IM,
        2048,
    );
    let (baseline, oracle, report) = compare(&plan);
    validate_report(baseline, oracle, &report, "gemma4-12b prefill s2048");
}

/// Batch 32 decode: Gemma 12B (high-throughput scenario).
#[test]
fn gemma_12b_decode_b32() {
    let plan = gemma_block(
        gemma12b::H,
        gemma12b::NH,
        gemma12b::NKV,
        gemma12b::HD,
        gemma12b::QD,
        gemma12b::KVD,
        gemma12b::IM,
        32,
    );
    let (baseline, oracle, report) = compare(&plan);
    validate_report(baseline, oracle, &report, "gemma4-12b decode b32");
}
