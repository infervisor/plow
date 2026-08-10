use super::*;

// ---- reference Mamba-2 SSD math (f32) ------------------------------------------------------

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    // numerically-stable log(1+e^x)
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Deterministic pseudo-random stream in [-amp, amp] (reproducible, no rand dep).
struct Lcg(u64);
impl Lcg {
    fn f(&mut self, amp: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 33) as f32) / ((1u64 << 31) as f32); // [0,1)
        (u * 2.0 - 1.0) * amp
    }
}

struct Dims {
    t: usize,
    n_head: usize,
    head_dim: usize,
    d_state: usize,
    n_groups: usize,
}
impl Dims {
    fn d_inner(&self) -> usize {
        self.n_head * self.head_dim
    }
    fn hpg(&self) -> usize {
        self.n_head / self.n_groups
    }
}

/// SSD selective scan, STATEFUL recurrence form (what the device kernel / block emit mirror).
/// `x` [T, d_inner], `b`/`cc` [T, n_groups*d_state], `dt_eff` [T, n_head] (already softplus'd),
/// `a` [n_head] (= -exp(A_log)), `dd` [n_head] (the D skip). `ssm` [n_head*head_dim*d_state] is
/// read as the initial state and OVERWRITTEN with the final state. Returns yscan [T, d_inner].
fn scan_recurrence(
    d: &Dims,
    x: &[f32],
    b: &[f32],
    cc: &[f32],
    dt_eff: &[f32],
    a: &[f32],
    dd: &[f32],
    ssm: &mut [f32],
) -> Vec<f32> {
    let (nh, hd, ds, ng) = (d.n_head, d.head_dim, d.d_state, d.n_groups);
    let di = d.d_inner();
    let hpg = d.hpg();
    let mut y = vec![0.0f32; d.t * di];
    for t in 0..d.t {
        for h in 0..nh {
            let dtv = dt_eff[t * nh + h];
            let da = (dtv * a[h]).exp();
            let g = h / hpg;
            for p in 0..hd {
                let xv = x[t * di + h * hd + p];
                let mut acc = 0.0f32;
                for n in 0..ds {
                    let bn = b[t * ng * ds + g * ds + n];
                    let cn = cc[t * ng * ds + g * ds + n];
                    let si = h * hd * ds + p * ds + n;
                    ssm[si] = da * ssm[si] + dtv * xv * bn;
                    acc += cn * ssm[si];
                }
                y[t * di + h * hd + p] = acc + dd[h] * xv;
            }
        }
    }
    y
}

/// SSD selective scan, INDEPENDENT closed-form dual: h_t = exp(cum_t)·h_init +
/// Σ_{s≤t} exp(cum_t − cum_s)·dt_s·x_s⊗B_s, y_t = Σ_n C_t·h_t + D·x_t. Materializes the decay
/// per (t,s) and sums — a structurally different computation (different float order) than the
/// stateful recurrence, so agreement to tolerance validates the recurrence. `ssm_init` is the
/// carried-in state; does NOT mutate it.
fn scan_dual(
    d: &Dims,
    x: &[f32],
    b: &[f32],
    cc: &[f32],
    dt_eff: &[f32],
    a: &[f32],
    dd: &[f32],
    ssm_init: &[f32],
) -> Vec<f32> {
    let (nh, hd, ds, ng) = (d.n_head, d.head_dim, d.d_state, d.n_groups);
    let di = d.d_inner();
    let hpg = d.hpg();
    let mut y = vec![0.0f32; d.t * di];
    for h in 0..nh {
        // cumulative log-decay per t: cum[t] = Σ_{r=0}^{t} dt_r·A_h
        let mut cum = vec![0.0f32; d.t];
        let mut run = 0.0f32;
        for t in 0..d.t {
            run += dt_eff[t * nh + h] * a[h];
            cum[t] = run;
        }
        let g = h / hpg;
        for t in 0..d.t {
            for p in 0..hd {
                let mut acc = dd[h] * x[t * di + h * hd + p];
                for n in 0..ds {
                    let cn = cc[t * ng * ds + g * ds + n];
                    // initial-state contribution
                    let mut hval = cum[t].exp() * ssm_init[h * hd * ds + p * ds + n];
                    // input contributions from all s ≤ t
                    for s in 0..=t {
                        let decay = (cum[t] - cum[s]).exp();
                        let xs = x[s * di + h * hd + p];
                        let bs = b[s * ng * ds + g * ds + n];
                        hval += decay * dt_eff[s * nh + h] * xs * bs;
                    }
                    acc += cn * hval;
                }
                y[t * di + h * hd + p] = acc;
            }
        }
    }
    y
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The NEW SSM math: the stateful recurrence (kernel/emit form) equals the independent
/// closed-form dual to f32 tolerance — with a NON-ZERO carried-in ssm_state, so the initial
/// state term is exercised. Reports the max-abs error vs the golden.
#[test]
fn mamba2_scan_matches_independent_recurrence() {
    let d = Dims {
        t: 6,
        n_head: 4,
        head_dim: 5,
        d_state: 3,
        n_groups: 2,
    };
    let di = d.d_inner();
    let gd = d.n_groups * d.d_state;
    let mut r = Lcg(0x1234_5678_9abc_def0);
    let x: Vec<f32> = (0..d.t * di).map(|_| r.f(0.5)).collect();
    let b: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
    let cc: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
    // dt already softplus'd (positive); A = -exp(a_log) (negative) => stable decay in (0,1).
    let dt_eff: Vec<f32> = (0..d.t * d.n_head).map(|_| softplus(r.f(1.0))).collect();
    let a: Vec<f32> = (0..d.n_head).map(|_| -(r.f(0.5) + 0.7).exp()).collect();
    let dd: Vec<f32> = (0..d.n_head).map(|_| r.f(0.5)).collect();
    let ssm_init: Vec<f32> = (0..d.n_head * d.head_dim * d.d_state)
        .map(|_| r.f(0.3))
        .collect();

    let mut ssm = ssm_init.clone();
    let y_rec = scan_recurrence(&d, &x, &b, &cc, &dt_eff, &a, &dd, &mut ssm);
    let y_dual = scan_dual(&d, &x, &b, &cc, &dt_eff, &a, &dd, &ssm_init);
    let err = max_abs(&y_rec, &y_dual);
    eprintln!("mamba2 SSM scan: max-abs err (recurrence vs independent dual) = {err:e}");
    assert!(
        err < 1e-4,
        "SSM scan diverges from independent golden: max-abs {err:e}"
    );
}

/// Prefill/decode equivalence: running the scan as ONE T-step prefill leaves the same
/// ssm_state, and yields the same last-token output, as feeding the tokens one at a time
/// through single-step decode calls (each carrying the state forward). This is the
/// state-carry contract the harness relies on (§6, §7).
#[test]
fn mamba2_decode_equals_prefill() {
    let d = Dims {
        t: 5,
        n_head: 3,
        head_dim: 4,
        d_state: 3,
        n_groups: 1,
    };
    let di = d.d_inner();
    let gd = d.n_groups * d.d_state;
    let mut r = Lcg(0xdead_beef_cafe_1234);
    let x: Vec<f32> = (0..d.t * di).map(|_| r.f(0.5)).collect();
    let b: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
    let cc: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
    let dt_eff: Vec<f32> = (0..d.t * d.n_head).map(|_| softplus(r.f(1.0))).collect();
    let a: Vec<f32> = (0..d.n_head).map(|_| -(r.f(0.3) + 0.7).exp()).collect();
    let dd: Vec<f32> = (0..d.n_head).map(|_| r.f(0.5)).collect();

    // Full prefill scan.
    let mut ssm_pf = vec![0.0f32; d.n_head * d.head_dim * d.d_state];
    let y_pf = scan_recurrence(&d, &x, &b, &cc, &dt_eff, &a, &dd, &mut ssm_pf);

    // Token-at-a-time decode, carrying ssm_state forward.
    let mut ssm_dec = vec![0.0f32; d.n_head * d.head_dim * d.d_state];
    let mut y_last = vec![0.0f32; di];
    for t in 0..d.t {
        let d1 = Dims {
            t: 1,
            ..copy_dims(&d)
        };
        let xr = &x[t * di..(t + 1) * di];
        let br = &b[t * gd..(t + 1) * gd];
        let cr = &cc[t * gd..(t + 1) * gd];
        let dtr = &dt_eff[t * d.n_head..(t + 1) * d.n_head];
        y_last = scan_recurrence(&d1, xr, br, cr, dtr, &a, &dd, &mut ssm_dec);
    }
    let err_state = max_abs(&ssm_pf, &ssm_dec);
    let err_y = max_abs(&y_pf[(d.t - 1) * di..], &y_last);
    eprintln!("mamba2 prefill-vs-decode: ssm_state err={err_state:e} last-token err={err_y:e}");
    assert!(
        err_state < 1e-5,
        "decode state != prefill state: {err_state:e}"
    );
    assert!(err_y < 1e-5, "decode last-token != prefill: {err_y:e}");
}

fn copy_dims(d: &Dims) -> Dims {
    Dims {
        t: d.t,
        n_head: d.n_head,
        head_dim: d.head_dim,
        d_state: d.d_state,
        n_groups: d.n_groups,
    }
}

// ---- emit op-sequence + descriptor ---------------------------------------------------------

/// Synthetic small Nemotron-3 hybrid cfg (structurally faithful: mamba mixer + GQA attn + MoE).
/// Layer 0 = mamba, 1 = attn, 2 = moe (a minimal one-of-each pattern the block extraction walks).
fn nemo_ref_cfg() -> NemoCfg {
    NemoCfg {
        layers: 3,
        hidden: 64,
        d_inner: 128,
        n_head: 8,
        head_dim: 16, // d_inner / n_head
        d_state: 16,
        d_conv: 4,
        n_groups: 2,
        attn_heads: 8,
        attn_kv_heads: 2,
        attn_head_dim: 16,
        n_exp: 16,
        top_k: 4,
        shared_exp: 1,
        moe_inter: 96,
        eps: 1e-5,
        kinds: vec![NemoKind::Mamba, NemoKind::Attn, NemoKind::Moe],
    }
}

fn block_ops(c: &NemoCfg, block: std::ops::Range<usize>) -> Vec<u16> {
    let (m, _d) = nemotron_build_block(c, 512, 256, block, "nemotron-ref");
    m.progs[0].insts.iter().map(|d| d.op).collect()
}

/// Mamba mixer block: input RMSNorm, 3 in_proj GEMVs (z/xBC/dt), the NEW Mamba2Scan, out_proj
/// GEMV, residual — `act.x` in and out, no embed/tail.
#[test]
fn nemotron_mamba_block_sequence() {
    use DevOp::*;
    let c = nemo_ref_cfg();
    assert_eq!(
        block_ops(&c, 0..1),
        vec![RmsNorm, Gemv, Gemv, Gemv, Mamba2Scan, Gemv, Residual]
            .into_iter()
            .map(|o| o as u16)
            .collect::<Vec<_>>(),
        "mamba mixer block sequence"
    );
}

/// GQA attention block reuses the existing attn DevOps.
#[test]
fn nemotron_attn_block_sequence() {
    use DevOp::*;
    let c = nemo_ref_cfg();
    assert_eq!(
        block_ops(&c, 1..2),
        vec![
            RmsNorm,
            GemvQkv,
            HeadNormRope,
            FlashDecode,
            FlashMerge,
            Gemv,
            Residual
        ]
        .into_iter()
        .map(|o| o as u16)
        .collect::<Vec<_>>(),
        "gqa attention block sequence"
    );
}

/// MoE block reuses the existing MoE DevOps (router split + shared expert + top_k experts +
/// combine), matching the kimi MoE structure.
#[test]
fn nemotron_moe_block_sequence() {
    use DevOp::*;
    let c = nemo_ref_cfg();
    let mut want = vec![RmsNorm, Gemv, MoeRouterTopk, GemvGlu, Gemv];
    for _ in 0..c.top_k {
        want.push(MoeExpertGlu);
        want.push(MoeExpertDown);
    }
    want.push(MoeCombine);
    assert_eq!(
        block_ops(&c, 2..3),
        want.into_iter().map(|o| o as u16).collect::<Vec<_>>(),
        "moe block sequence"
    );
}

/// Mamba block descriptor: arch nemotron_h, kind ["mamba2"], Mamba-2 dims, conv+ssm carried
/// state (NO kv), no attn/MoE dims.
#[test]
fn nemotron_mamba_descriptor() {
    let c = nemo_ref_cfg();
    let (_, d) = nemotron_build_block(&c, 512, 256, 0..1, "Nemotron-3");
    assert_eq!(d.arch, "nemotron_h");
    assert_eq!(d.kind, vec!["mamba2"]);
    assert_eq!(d.layer, 0);
    assert_eq!(d.dims.d_inner, Some(128));
    assert_eq!(d.dims.n_head, Some(8));
    assert_eq!(d.dims.head_dim, Some(16));
    assert_eq!(d.dims.d_state, Some(16));
    assert_eq!(d.dims.d_conv, Some(4));
    assert_eq!(d.dims.n_groups, Some(2));
    assert_eq!(d.dims.heads, None, "mamba block has no attn dims");
    assert_eq!(d.dims.n_exp, None, "mamba block has no MoE dims");
    assert_eq!(d.carried_state.len(), 2);
    assert_eq!(d.carried_state[0].role, "conv");
    assert_eq!(d.carried_state[0].layout, "conv");
    assert_eq!(d.carried_state[0].tensors, vec!["mamba.0.conv_state"]);
    assert_eq!(d.carried_state[1].role, "ssm");
    assert_eq!(d.carried_state[1].layout, "ssm_head_major");
    assert_eq!(d.carried_state[1].tensors, vec!["mamba.0.ssm_state"]);
    assert_eq!(d.weights.prefix, "backbone.layers.0.");
    assert!(d.programs.prefill_buckets.is_empty());
    assert_eq!(
        d.outputs[0].name, "act.xnext",
        "one (odd) layer -> act.xnext"
    );
}

/// Attention block descriptor: kind ["gqa_attn"], GQA dims, kv carried state.
#[test]
fn nemotron_attn_descriptor() {
    let c = nemo_ref_cfg();
    let (_, d) = nemotron_build_block(&c, 512, 256, 1..2, "Nemotron-3");
    assert_eq!(d.kind, vec!["gqa_attn"]);
    assert_eq!(d.dims.heads, Some(8));
    assert_eq!(d.dims.kv_heads, Some(2));
    assert_eq!(d.dims.head_dim, Some(16));
    assert_eq!(d.dims.d_inner, None, "attn block has no mamba dims");
    assert_eq!(d.dims.n_exp, None);
    assert_eq!(d.carried_state.len(), 1);
    assert_eq!(d.carried_state[0].role, "kv");
    assert_eq!(d.carried_state[0].tensors, vec!["kv.1.k", "kv.1.v"]);
}

/// MoE block descriptor: kind ["moe_ffn"], MoE dims, NO carried state.
#[test]
fn nemotron_moe_descriptor() {
    let c = nemo_ref_cfg();
    let (_, d) = nemotron_build_block(&c, 512, 256, 2..3, "Nemotron-3");
    assert_eq!(d.kind, vec!["moe_ffn"]);
    assert_eq!(d.dims.n_exp, Some(16));
    assert_eq!(d.dims.top_k, Some(4));
    assert_eq!(d.dims.shared_exp, Some(1));
    assert_eq!(d.dims.moe_inter, Some(96));
    assert_eq!(d.dims.d_inner, None);
    assert_eq!(d.dims.heads, None);
    assert!(d.carried_state.is_empty(), "MoE block carries no state");
}

/// A multi-layer block chains all three layer kinds; kind lists each, carried_state unions the
/// mamba (conv+ssm) and attn (kv) entries, and the residual ping-pong lands the output in
/// `act.xnext` after 3 (odd) layers.
#[test]
fn nemotron_multi_layer_chains() {
    use DevOp::*;
    let c = nemo_ref_cfg();
    let ops = block_ops(&c, 0..3);
    // mamba(7) + attn(7) + moe(5 + 2*top_k + 1)
    assert_eq!(ops[0], RmsNorm as u16);
    assert_eq!(ops[4], Mamba2Scan as u16, "mamba mixer first");
    assert!(ops.contains(&(FlashDecode as u16)), "attn layer present");
    assert!(ops.contains(&(MoeCombine as u16)), "moe layer present");
    let (_, d) = nemotron_build_block(&c, 512, 256, 0..3, "Nemotron-3");
    assert_eq!(d.kind, vec!["mamba2", "gqa_attn", "moe_ffn"]);
    assert_eq!(d.layer, 0);
    assert_eq!(
        d.outputs[0].name, "act.xnext",
        "3 layers (odd) -> act.xnext"
    );
    // conv + ssm (mamba L0) + kv (attn L1); moe contributes none.
    let roles: Vec<&str> = d.carried_state.iter().map(|s| s.role.as_str()).collect();
    assert_eq!(roles, vec!["conv", "ssm", "kv"]);
    // all Mamba-2 dims and attn dims and MoE dims populated.
    assert_eq!(d.dims.d_inner, Some(128));
    assert_eq!(d.dims.kv_heads, Some(2));
    assert_eq!(d.dims.n_exp, Some(16));
}
