//! MoE dispatch core — the reusable, model-general machinery for the
//! data-dependent counter-gate.
//!
//! Two halves live here, both model-invariant:
//!
//! 1. **The router** ([`RouterCfg`] + [`route`]): the config-driven top-k selection
//!    that runs as a single device packet ([`crate::dev::DevOp::MoeRouter`]) and, on
//!    the host, is the CPU golden/oracle for it. Its **top-k is bit-exact**: ties break
//!    by lowest expert id, baked into a packed-key masked argmax that mirrors the
//!    on-device `amax_pack` algorithm (`moe-ep-kernels.md §2c`). This is *the*
//!    bit-exactness linchpin — the on-device and reference selection must agree, or a
//!    silent mis-route diverges the whole layer.
//!
//! 2. **The FFN emitter** ([`emit_moe_ffn`]): turns a [`RouterCfg`] + tensor handles
//!    into the static packet stream — one router, `k` expert-body slots (each carrying
//!    its `slot` index; the SM resolves the weight base from the routing table it reads),
//!    an always-run shared expert, and a fixed-order combine. The counter DAG is static
//!    (deadlock-free, `executed == total`); the *conditionality* is in each expert
//!    packet's body, which always signals whether it computed or skipped.
//!
//! The cardinality (`k` slots, one router, one combine) is fixed at compile time, so the
//! stream is deterministic even though the *choices* are dynamic — this is what keeps the
//! whole design inside plow's existing counter-gate model with no interpreter change.

use crate::dev::{DevOp, EXPERT_UNUSED, TENSOR_NONE};
use crate::devbuild::Builder;

/// Router scoring function (`moe-plow-design.md §5a`). GLM-5.2 / DeepSeek-V3 use
/// `Sigmoid` (per-expert, independent); Qwen3-MoE / Mixtral use `Softmax` (over all
/// experts). Shape-identical; only the score transform differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scoring {
    Softmax,
    Sigmoid,
}

/// The config that makes the router model-general (`moe-plow-design.md §5a`,
/// `moe-ep-kernels.md §1`). GLM-5.2: `{ n_exp: 256, k: 8, Sigmoid, norm_topk: true,
/// route_scale: 2.5 }` (`group`/`bias` are DeepSeek-only and out of scope here).
#[derive(Clone, Copy, Debug)]
pub struct RouterCfg {
    /// Number of routed experts (E).
    pub n_exp: u32,
    /// Experts per token (top-k, K).
    pub k: u32,
    /// Score transform.
    pub scoring: Scoring,
    /// Renormalise the k selected gates to sum 1, *after* selection (GLM/DS: true).
    pub norm_topk: bool,
    /// `routed_scaling_factor`: the k gates are multiplied by this, last (GLM/DS: 2.5).
    pub route_scale: f32,
}

impl RouterCfg {
    /// The `i3` flags word carried on the [`DevOp::MoeRouter`] instruction, mirroring the
    /// `flags` field decoded in `op_moe.h` (bit0 = sigmoid, bit1 = norm_topk).
    pub fn flags(&self) -> u32 {
        let mut f = 0u32;
        if self.scoring == Scoring::Sigmoid {
            f |= 1;
        }
        if self.norm_topk {
            f |= 2;
        }
        f
    }
}

/// One routing-table entry: the expert chosen for a slot and its (normalised, scaled)
/// gate. `expert_id == EXPERT_UNUSED` marks an unused slot (never happens for ungrouped
/// top-k where `k <= n_exp`, but the sentinel is part of the ABI the expert body checks).
///
/// Byte layout (`u32 expert_id`, `f32 gate`) is exactly the 8-byte routing-table entry the
/// device writes and reads (`moe-ep-kernels.md §3a`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct RouteEntry {
    pub expert_id: u32,
    pub gate: f32,
}

/// Monotonic bijection f32→u32 so an unsigned compare orders floats correctly (ties in
/// the raw score then fall to the id tie-break). Mirrors the device `amax_pack` ordering.
#[inline]
fn ordered_bits(f: f32) -> u32 {
    let b = f.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// The router body (`moe-ep-kernels.md §2b`), as the CPU reference / golden. Given the
/// `n_exp` **logits** `x·Wr`, produce the `k` `(expert_id, gate)` entries:
///
/// `score = scoring(logit)` → **top-k via k-pass masked argmax, lowest-id tie-break** →
/// `if norm_topk: gate /= Σ gate` → `gate *= route_scale`.
///
/// The tie-break is baked into a packed key `(ordered_score << 20) | (n_exp-1 - id)` so a
/// plain unsigned max selects the highest score and, among equal scores, the **lowest id**
/// — deterministic and reproducible, matching the device (`moe-ep-kernels.md §2c`). The
/// gate is the *unbiased* score of the winner (GLM has no routing bias).
pub fn route(cfg: &RouterCfg, logits: &[f32]) -> Vec<RouteEntry> {
    let n = cfg.n_exp as usize;
    assert_eq!(logits.len(), n, "router logits must be n_exp wide");
    let k = cfg.k as usize;
    assert!(k <= n, "top-k k must not exceed n_exp");

    // 1. score(logit)
    let score: Vec<f32> = match cfg.scoring {
        Scoring::Sigmoid => logits.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect(),
        Scoring::Softmax => {
            // numerically-stable softmax over all experts
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&z| (z - m).exp()).collect();
            let s: f32 = exps.iter().sum();
            exps.iter().map(|&e| e / s).collect()
        }
    };

    // 2. k-pass masked argmax over packed keys (lowest-id tie-break).
    let mut keys: Vec<u64> = (0..n)
        .map(|id| {
            ((ordered_bits(score[id]) as u64) << 20) | ((n as u64 - 1 - id as u64) & 0xF_FFFF)
        })
        .collect();
    let mut winners: Vec<usize> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut best = 0u64;
        for &key in keys.iter() {
            if key > best {
                best = key;
            }
        }
        let id = (n - 1) - (best & 0xF_FFFF) as usize;
        winners.push(id);
        keys[id] = 0; // mark dead
    }

    // 3. gates = unbiased scores of the winners; optional renorm; then scale.
    let mut gates: Vec<f32> = winners.iter().map(|&id| score[id]).collect();
    if cfg.norm_topk {
        let s: f32 = gates.iter().sum();
        let inv = if s != 0.0 { 1.0 / s } else { 0.0 };
        for g in &mut gates {
            *g *= inv;
        }
    }
    for g in &mut gates {
        *g *= cfg.route_scale;
    }

    winners
        .into_iter()
        .zip(gates)
        .map(|(id, g)| RouteEntry {
            expert_id: id as u32,
            gate: g,
        })
        .collect()
}

/// The tensor handles an MoE FFN layer refers to. All are declared by the caller (the
/// compiler / harness); the emitter only wires the ops. Per-slot scratch (`fu`, `part`)
/// is `[k, ·]` so the K expert slots never clobber each other.
#[derive(Clone, Copy, Debug)]
pub struct MoeTensors {
    /// `x`: the normed residual fed to the router + experts (bf16, `[H]`).
    pub x: u32,
    /// Router weight `Wr` (bf16, `[n_exp, H]`).
    pub wr: u32,
    /// Routing table the router writes and the experts read (`[k]` × 8 bytes).
    pub routing_table: u32,
    /// Table of device pointers to each expert's `{gate, up, down}` weight bases
    /// (`Persistent`, filled at load by `orch/moe.rs::resolve_expert_tables`).
    pub expert_weight_table: u32,
    /// Per-slot gate/up scratch (bf16, `[k, I_moe]`).
    pub fu: u32,
    /// Per-slot gate-scaled down partials (bf16, `[k, H]`).
    pub part: u32,
    /// The residual stream to add into the combine (bf16, `[H]`).
    pub residual: u32,
    /// Shared-expert output, or [`TENSOR_NONE`] for a 0-shared-expert config (bf16, `[H]`).
    pub shared_out: u32,
    /// Combine output (bf16, `[H]`).
    pub out: u32,
}

/// Emit one MoE FFN layer's static packet stream onto `b`, gated behind `dep` (the
/// producer of `x` — typically the pre-FFN norm). Returns the combine's counter, which the
/// next sublayer depends on.
///
/// The structure (`moe-plow-design.md §3`, `moe-ep-kernels.md §2-§3): **router →
/// k×(expert_glu → expert_down) → combine**, with the shared expert emitted by the caller
/// (a plain dense FFN feeding `shared_out`). Every expert slot signals its counter
/// unconditionally, so the combine's threshold is met for *any* routing — the skip-safety
/// invariant. `dispatch = KSlot` (decode bs=1): exactly `k` expert packets, weight base
/// remapped per slot from the table the router wrote.
///
/// `H`, `i_moe`, `act` are the layer geometry; `act` is the GLU activation (0 = gelu_tanh,
/// 1 = silu — GLM is SwiGLU/silu). The GEMV path swaps to `GEMV_FP8` cleanly once the fp8
/// merge lands: only the two expert-op opcodes change, not this structure.
#[allow(clippy::too_many_arguments)]
pub fn emit_moe_ffn(
    b: &mut Builder,
    cfg: &RouterCfg,
    t: &MoeTensors,
    hidden: u32,
    i_moe: u32,
    act: u32,
    dep: u32,
    router_cus: Vec<u32>,
    expert_cus: Vec<u32>,
    combine_cus: Vec<u32>,
    shared_dep: Option<u32>,
) -> u32 {
    // 1. Router: one packet, writes the routing table. Its counter is the gate the K
    //    expert slots wait on (moe-ep-kernels §2d).
    let c_router = b.emit(DevOp::MoeRouter, router_cus, &[dep], |d| {
        d.t[0] = t.routing_table;
        d.t[1] = t.x;
        d.t[2] = t.wr;
        d.i[0] = hidden;
        d.i[1] = cfg.n_exp;
        d.i[2] = cfg.k;
        d.i[3] = cfg.flags();
        d.f[0] = cfg.route_scale;
    });

    // 2. K expert slots. Each is (gate/up GEMV) -> (down GEMV, gate-scale). The slot reads
    //    routing_table[slot] itself; the emitted stream is identical for every slot bar the
    //    `slot` immediate — one common template, K instances (moe-design §4a).
    let mut down_ctrs: Vec<u32> = Vec::with_capacity(cfg.k as usize);
    for slot in 0..cfg.k {
        let c_glu = b.emit(DevOp::MoeExpertGlu, expert_cus.clone(), &[c_router], |d| {
            d.t[0] = t.fu;
            d.t[1] = t.x;
            d.t[2] = t.routing_table;
            d.t[3] = t.expert_weight_table;
            d.i[0] = slot;
            d.i[1] = i_moe;
            d.i[2] = hidden;
            d.i[3] = cfg.n_exp;
            d.i[5] = act;
        });
        let c_down = b.emit(DevOp::MoeExpertDown, expert_cus.clone(), &[c_glu], |d| {
            d.t[0] = t.part;
            d.t[1] = t.fu;
            d.t[2] = t.routing_table;
            d.t[3] = t.expert_weight_table;
            d.i[0] = slot;
            d.i[1] = hidden;
            d.i[2] = i_moe;
            d.i[3] = cfg.n_exp;
        });
        down_ctrs.push(c_down);
    }

    // 3. Combine: wait on all K down slots (+ the shared expert), fixed-order f32 fold.
    let mut deps = down_ctrs;
    if let Some(sd) = shared_dep {
        deps.push(sd);
    }
    b.emit(DevOp::MoeCombine, combine_cus, &deps, |d| {
        d.t[0] = t.out;
        d.t[1] = t.residual;
        d.t[2] = t.shared_out; // TENSOR_NONE ok
        d.t[3] = t.part;
        d.i[0] = hidden;
        d.i[1] = cfg.k;
    })
}

/// The sentinel, re-exported for callers building routing tables by hand.
pub const UNUSED: u32 = EXPERT_UNUSED;
/// Re-export so callers don't need to import from two modules.
pub const NONE_TENSOR: u32 = TENSOR_NONE;
