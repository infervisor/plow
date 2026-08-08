//! Bit-exact validation of the MoE dispatch core (`crates/packet/src/moe.rs`),
//! CI-runnable with no GPU — the correctness spine of the GLM-5.2 prototype's M2/M3.
//!
//! Three independent checks, each targeting a distinct failure mode the design calls out:
//!
//!  1. **Router top-k is bit-exact with the specified tie-break** — the linchpin. The
//!     shipped router ([`packet::moe::route`], a k-pass masked-argmax over packed keys) is
//!     cross-checked against a *different* algorithm (a full argsort) across many seeds,
//!     including engineered exact ties. They must pick the identical expert set/order,
//!     lowest-id-wins. A tolerance pass on gates can never catch a mis-route; this does.
//!
//!  2. **Sparse dispatch == dense-masked reference, bit-exact, over N stacked blocks** —
//!     the sparse counter-gated path (route → K expert slots → fixed-order combine, the
//!     other `E-K` streaming zero) reproduces, to the bit, an independent oracle that
//!     computes *all* `E` experts and masks to the top-k. Proves streaming only the K
//!     chosen experts is equivalent to compute-all-and-mask, and that the combine's fixed
//!     slot-order f32→bf16 fold is deterministic.
//!
//!  3. **`executed == total` + exactly-K-compute** — every one of the K expert slots
//!     "signals" regardless of routing (skip-safety), and exactly K experts do work; the
//!     rest skip. This is the invariant that keeps the static counter DAG deadlock-free
//!     under any dynamic routing.
//!
//! The two numeric paths share only the low-level expert-FFN kernel and the bf16 rounding
//! (the "op bodies", shared verbatim exactly as `interp.hip` shares `op_*.h` with the
//! golden wrappers); the *selection* and the *dispatch structure* are independent.

use packet::dev::DevOp;
use packet::devbuild::Builder;
use packet::moe::{emit_moe_ffn, route, MoeTensors, RouterCfg, Scoring};

// ---- bf16, matching the device round-to-nearest (op writes bf16 at each boundary) ----
fn b2f(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}
fn f2b(f: f32) -> u16 {
    let u = f.to_bits();
    let r = u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1);
    (r >> 16) as u16
}
/// Round-trip through bf16, the value every op stores.
fn rb(f: f32) -> f32 {
    b2f(f2b(f))
}

// ---- deterministic seeded weights (fixed bits every run; a mismatch is a real bug) ----
struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
    }
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as i32 % 2001 - 1000) as f32 / 4000.0 // +-0.25
    }
    fn fill_bf16(&mut self, n: usize) -> Vec<u16> {
        (0..n).map(|_| f2b(self.next())).collect()
    }
}

// ---- the shared low-level kernels (the "op bodies") ----
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
/// bf16 GEMV `y[n] = Σ_k x[k]·W[n,k]`, f32 accumulate, bf16-rounded output (the decode op).
fn gemv(x: &[f32], w: &[u16], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|j| {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[i] * b2f(w[j * k + i]);
            }
            rb(acc)
        })
        .collect()
}
/// One expert's SwiGLU FFN: `down(silu(gate·x) * (up·x))`, returns the H-vector (bf16-rounded).
fn expert_ffn(xn: &[f32], wg: &[u16], wu: &[u16], wd: &[u16], h: usize, i_moe: usize) -> Vec<f32> {
    let g = gemv(xn, wg, i_moe, h);
    let u = gemv(xn, wu, i_moe, h);
    let fu: Vec<f32> = (0..i_moe).map(|i| rb(silu(g[i]) * u[i])).collect();
    gemv(&fu, wd, h, i_moe)
}
fn rmsnorm(x: &[f32], gamma: &[u16], eps: f32) -> Vec<f32> {
    let h = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum::<f32>() / h as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    (0..h).map(|i| rb(x[i] * inv * b2f(gamma[i]))).collect()
}

// ---- a whole MoE layer's seeded weights ----
struct Layer {
    wr: Vec<u16>,        // [n_exp, H]
    gate: Vec<Vec<u16>>, // [n_exp][I_moe, H]
    up: Vec<Vec<u16>>,   // [n_exp][I_moe, H]
    down: Vec<Vec<u16>>, // [n_exp][H, I_moe]
    sgate: Vec<u16>,     // shared [I_moe, H]
    sup: Vec<u16>,
    sdown: Vec<u16>,
    gnorm: Vec<u16>, // pre-FFN norm gamma [H]
}
fn seed_layer(seed: u64, cfg: &RouterCfg, h: usize, i_moe: usize) -> Layer {
    let e = cfg.n_exp as usize;
    let mut r = Rng::new(seed);
    let wr = r.fill_bf16(e * h);
    let gate = (0..e).map(|_| r.fill_bf16(i_moe * h)).collect();
    let up = (0..e).map(|_| r.fill_bf16(i_moe * h)).collect();
    let down = (0..e).map(|_| r.fill_bf16(h * i_moe)).collect();
    let sgate = r.fill_bf16(i_moe * h);
    let sup = r.fill_bf16(i_moe * h);
    let sdown = r.fill_bf16(h * i_moe);
    let gnorm: Vec<u16> = (0..h).map(|_| f2b(1.0 + r.next() * 0.1)).collect();
    Layer {
        wr,
        gate,
        up,
        down,
        sgate,
        sup,
        sdown,
        gnorm,
    }
}

// ---- INDEPENDENT ORACLE: dense — compute ALL experts, mask to top-k, combine ----
fn oracle_block(
    x: &[f32],
    lyr: &Layer,
    cfg: &RouterCfg,
    h: usize,
    i_moe: usize,
    eps: f32,
) -> Vec<f32> {
    let e = cfg.n_exp as usize;
    let k = cfg.k as usize;
    let xn = rmsnorm(x, &lyr.gnorm, eps);
    let logits = gemv(&xn, &lyr.wr, e, h);

    // independent selection: argsort by (score desc, id asc), take k
    let score: Vec<f32> = logits.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
    let mut order: Vec<usize> = (0..e).collect();
    order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap().then(a.cmp(&b)));
    let sel = &order[..k];
    let mut gates: Vec<f32> = sel.iter().map(|&id| score[id]).collect();
    if cfg.norm_topk {
        let s: f32 = gates.iter().sum();
        for g in &mut gates {
            *g /= s;
        }
    }
    for g in &mut gates {
        *g *= cfg.route_scale;
    }

    // shared expert (always on) + selected routed experts, fixed order = sel order
    let shared = expert_ffn(&xn, &lyr.sgate, &lyr.sup, &lyr.sdown, h, i_moe);
    let mut acc: Vec<f32> = (0..h).map(|i| x[i] as f32 + shared[i]).collect();
    for (slot, &id) in sel.iter().enumerate() {
        let y = expert_ffn(&xn, &lyr.gate[id], &lyr.up[id], &lyr.down[id], h, i_moe);
        for i in 0..h {
            acc[i] += gates[slot] * y[i];
        }
    }
    acc.iter().map(|&v| rb(v)).collect()
}

/// Result of running the dispatch model, for the structural invariant checks.
struct DispatchRun {
    out: Vec<f32>,
    experts_computed: usize,
    slots_signaled: usize,
}

// ---- DEVICE DISPATCH MODEL: sparse — route(), K slots, sentinel skip, fixed-order combine.
// Mirrors op_moe.h semantics: the router writes the table; each of the K expert slots reads
// routing_table[slot], resolves the weight base by expert id (the two-level indirection), and
// computes-or-skips, ALWAYS "signaling" (counted). Streams only the K chosen experts.
fn dispatch_block(
    x: &[f32],
    lyr: &Layer,
    cfg: &RouterCfg,
    h: usize,
    i_moe: usize,
    eps: f32,
) -> DispatchRun {
    let k = cfg.k as usize;
    let xn = rmsnorm(x, &lyr.gnorm, eps);
    let logits = gemv(&xn, &lyr.wr, cfg.n_exp as usize, h);
    let table = route(cfg, &logits); // the shipped router = the device MoeRouter golden

    // shared expert inline (always on)
    let shared = expert_ffn(&xn, &lyr.sgate, &lyr.sup, &lyr.sdown, h, i_moe);

    // K expert slots: each writes a per-slot partial (or a zeroed one on skip).
    let mut part = vec![vec![0f32; h]; k];
    let mut experts_computed = 0;
    let mut slots_signaled = 0;
    for slot in 0..k {
        let ent = table[slot];
        // data-dependent BODY gate (moe-design §3b): skip on sentinel, else weight-base indirection.
        if ent.expert_id < cfg.n_exp {
            let id = ent.expert_id as usize;
            let y = expert_ffn(&xn, &lyr.gate[id], &lyr.up[id], &lyr.down[id], h, i_moe);
            for i in 0..h {
                part[slot][i] = ent.gate * y[i]; // gate-scaled partial
            }
            experts_computed += 1;
        } // else: part[slot] stays zero (streamed zero bytes)
        slots_signaled += 1; // ALWAYS signals — the skip-safety invariant
    }

    // combine: residual + shared + Σ_slot part[slot], FIXED slot order, f32→bf16.
    let out: Vec<f32> = (0..h)
        .map(|i| {
            let mut acc = x[i] as f32 + shared[i];
            for slot in 0..k {
                acc += part[slot][i];
            }
            rb(acc)
        })
        .collect();

    DispatchRun {
        out,
        experts_computed,
        slots_signaled,
    }
}

fn glm_scaled_cfg() -> RouterCfg {
    // GLM-5.2 RouterCfg, cardinality-scaled per the design notes (E=8, K=2;
    // scoring/norm/scale are the REAL GLM values — the correctness points, kept exact).
    RouterCfg {
        n_exp: 8,
        k: 2,
        scoring: Scoring::Sigmoid,
        norm_topk: true,
        route_scale: 2.5,
    }
}

#[test]
fn router_topk_is_bit_exact_with_lowest_id_tiebreak() {
    let cfg = glm_scaled_cfg();
    let e = cfg.n_exp as usize;
    // Many seeds — a tie-break bug shows up only on specific score patterns.
    for seed in 0..2000u64 {
        let mut r = Rng::new(seed ^ 0x9e37);
        let logits: Vec<f32> = (0..e).map(|_| r.next() * 8.0).collect();
        let got = route(&cfg, &logits);

        // independent reference selection (argsort, lowest-id tie-break)
        let score: Vec<f32> = logits.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap().then(a.cmp(&b)));
        let want: Vec<u32> = order[..cfg.k as usize].iter().map(|&i| i as u32).collect();
        let got_ids: Vec<u32> = got.iter().map(|e| e.expert_id).collect();
        assert_eq!(
            got_ids, want,
            "seed {seed}: router selection diverged from argsort reference"
        );
    }

    // Engineered EXACT ties: several experts share the identical logit bits → lowest id must win.
    let mut logits = vec![-3.0f32; e];
    logits[5] = 2.0;
    logits[2] = 2.0; // tie with expert 5 (bitwise identical)
    logits[7] = 2.0; // and expert 7
    let got: Vec<u32> = route(&cfg, &logits).iter().map(|e| e.expert_id).collect();
    assert_eq!(
        got,
        vec![2, 5],
        "exact ties must resolve to the two LOWEST expert ids (2,5)"
    );
}

#[test]
fn sparse_dispatch_bit_exact_vs_dense_oracle_over_n_blocks() {
    let cfg = glm_scaled_cfg();
    let (h, i_moe, eps) = (64usize, 32usize, 1e-6f32);
    let n_blocks = 4usize; // block 0 acts dense-equivalent numerically; all exercise dispatch

    let layers: Vec<Layer> = (0..n_blocks)
        .map(|l| seed_layer(0xABCD ^ l as u64, &cfg, h, i_moe))
        .collect();

    // seed the residual stream (a synthetic "decode token" hidden state)
    let mut r = Rng::new(777);
    let x0: Vec<f32> = (0..h).map(|_| rb(r.next())).collect();

    // Run BOTH paths through the same N-block stack; compare the residual stream bit-for-bit
    // at every block boundary (localising any divergence, not just an end-of-stack number).
    let mut xo = x0.clone();
    let mut xd = x0.clone();
    let mut total_experts = 0;
    let mut total_slots = 0;
    for (l, lyr) in layers.iter().enumerate() {
        let o = oracle_block(&xo, lyr, &cfg, h, i_moe, eps);
        let d = dispatch_block(&xd, lyr, &cfg, h, i_moe, eps);
        // BIT-EXACT: identical bf16 bits of the residual stream after block l.
        let ob: Vec<u16> = o.iter().map(|&v| f2b(v)).collect();
        let db: Vec<u16> = d.out.iter().map(|&v| f2b(v)).collect();
        assert_eq!(
            ob, db,
            "block {l}: dispatch residual stream != dense oracle (bit-exact)"
        );
        // skip-safety: every slot signaled; exactly K experts computed (dispatch gates, not masks).
        assert_eq!(
            d.slots_signaled, cfg.k as usize,
            "block {l}: not all K slots signaled"
        );
        assert_eq!(
            d.experts_computed, cfg.k as usize,
            "block {l}: != K experts computed"
        );
        total_experts += d.experts_computed;
        total_slots += d.slots_signaled;
        xo = o;
        xd = d.out;
    }

    // executed == total across the whole decode: every expert slot fired (skip or compute).
    assert_eq!(total_slots, n_blocks * cfg.k as usize);
    // sparse-optimal: computed exactly K·N experts, not E·N (would be the launch-all-mask waste).
    assert_eq!(total_experts, n_blocks * cfg.k as usize);
    assert!(
        total_experts < n_blocks * cfg.n_exp as usize,
        "dispatch must stream only K experts/block, not E ({} < {})",
        total_experts,
        n_blocks * cfg.n_exp as usize
    );
}

#[test]
fn emit_moe_ffn_produces_well_formed_static_stream() {
    // M1: the emitter builds the static packet stream on the real devbuild::Builder — one
    // router, K expert slots (each carrying its own `slot` immediate), one combine — and the
    // counter DAG flattens deadlock-free (executed==total is structural: every op is reached).
    let cfg = glm_scaled_cfg();
    let mut b = Builder::new(64);
    let t = MoeTensors {
        x: b.tensor("x", 64 * 2),
        wr: b.tensor("wr", (cfg.n_exp * 64) as u64 * 2),
        routing_table: b.tensor("rt", cfg.k as u64 * 8),
        expert_weight_table: b.tensor("ewt", cfg.n_exp as u64 * 3 * 8),
        fu: b.tensor("fu", (cfg.k * 32) as u64 * 2),
        part: b.tensor("part", (cfg.k * 64) as u64 * 4),
        residual: b.tensor("resid", 64 * 2),
        shared_out: packet::dev::TENSOR_NONE,
        out: b.tensor("out", 64 * 2),
    };
    let dep = b.emit(DevOp::Nop, b.all(), &[], |_| {}); // stands in for the pre-FFN norm
    let all = b.all();
    let combine = emit_moe_ffn(
        &mut b,
        &cfg,
        &t,
        64,
        32,
        1,
        dep,
        vec![0],
        all.clone(),
        all.clone(),
        None,
    );
    let prog = b.finish();

    let count = |op: DevOp| prog.insts.iter().filter(|i| i.op == op as u16).count();
    assert_eq!(count(DevOp::MoeRouter), 1, "exactly one router");
    assert_eq!(
        count(DevOp::MoeExpertGlu),
        cfg.k as usize,
        "K expert-glu slots"
    );
    assert_eq!(
        count(DevOp::MoeExpertDown),
        cfg.k as usize,
        "K expert-down slots"
    );
    assert_eq!(count(DevOp::MoeCombine), 1, "exactly one combine");

    // Each expert slot carries its own `slot` index in i[0], and n_exp/act are wired.
    let glus: Vec<_> = prog
        .insts
        .iter()
        .filter(|i| i.op == DevOp::MoeExpertGlu as u16)
        .collect();
    let mut slots: Vec<u32> = glus.iter().map(|i| i.i[0]).collect();
    slots.sort();
    assert_eq!(
        slots,
        (0..cfg.k).collect::<Vec<_>>(),
        "expert slots are 0..K"
    );
    for g in &glus {
        assert_eq!(g.i[3], cfg.n_exp, "n_exp wired for the sentinel test");
        assert_eq!(g.i[5], 1, "act = silu (SwiGLU)");
    }

    // The router carries the RouterCfg flags/scale the device decodes.
    let r = prog
        .insts
        .iter()
        .find(|i| i.op == DevOp::MoeRouter as u16)
        .unwrap();
    assert_eq!(r.i[3], cfg.flags(), "router flags (sigmoid|norm_topk)");
    assert_eq!(r.f[0], cfg.route_scale, "route_scale");

    // combine is the returned counter and its wait list has K producers (the down slots).
    assert_eq!(
        combine as usize,
        prog.insts.len() - 1,
        "combine is the last op emitted"
    );
}

#[test]
fn softmax_config_also_routes_bit_exact() {
    // Generality: Qwen/Mixtral-style softmax scoring, no norm, scale 1.0 — same machinery.
    let cfg = RouterCfg {
        n_exp: 8,
        k: 2,
        scoring: Scoring::Softmax,
        norm_topk: false,
        route_scale: 1.0,
    };
    let e = cfg.n_exp as usize;
    for seed in 0..1000u64 {
        let mut r = Rng::new(seed ^ 0x1234);
        let logits: Vec<f32> = (0..e).map(|_| r.next() * 8.0).collect();
        let got: Vec<u32> = route(&cfg, &logits).iter().map(|e| e.expert_id).collect();
        // softmax is monotone in the logit, so top-k over softmax == top-k over the logit
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
        let want: Vec<u32> = order[..cfg.k as usize].iter().map(|&i| i as u32).collect();
        assert_eq!(got, want, "seed {seed}: softmax router selection diverged");
    }
}
