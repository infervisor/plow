//! NVFP4 MoE on the sm_120 interpreter (router ops 232/233, experts 234-238, activation quant
//! 239, combine 87): sigmoid top-k with the correction bias and the routed scale folded into the
//! gates, routed experts (optionally in a latent space between bf16 `fc1_latent_proj` /
//! `fc2_latent_proj`), one dense shared expert. Decode W4A16 (bf16 activations); prefill W4A4
//! (the peer's NVFP4 activations, op 239 and the W4A4 modes of 237/238, op_moe_nvfp4_pf.cuh).
//!
//! Routed-expert tensors `<module>.experts.<x>.<proj>.*` bind concatenated over the expert axis
//! (`derived_weights` `cat`): W u8 `[E][n][k/2]`, S e4m3 `[E][n][k/16]`, gs f32 `[E]`.
use kernelcaps::arms::{Arms, MissingArm};
use packet::dev::{DevOp, TENSOR_NONE};
use packet::typed_abi::{
    MoeAlignNvfp4Args, MoeCombinePfArgs, MoeDownNvfp4Args, MoeDownNvfp4PfArgs, MoeQuantNvfp4Args,
    MoeRouterLogitsF32Args, MoeRouterSigmoidTopkArgs, MoeUpNvfp4Args, MoeUpNvfp4PfArgs,
    ResidualArgs,
};

use super::{bf16_proj, Program};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub hidden: u32,
    pub experts: u32,
    pub top_k: u32,
    pub inter: u32,
    pub shared_inter: u32,
    /// The routed experts' latent width (`fc1_latent_proj` / `fc2_latent_proj`); `None` routes
    /// the hidden rows directly.
    pub latent: Option<u32>,
    pub routed_scale: f32,
    pub norm_topk: bool,
}

/// What the checkpoint's expert MLP is, checked against the expert arms.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpertMlp<'a> {
    pub act: &'a str,
    pub bias: bool,
    pub n_group: u32,
    pub topk_group: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct Nvfp4Moe {
    g: Geometry,
    arms: Arms,
}

impl Nvfp4Moe {
    pub fn new(g: Geometry, mlp: ExpertMlp, arms: &Arms) -> Result<Self, MissingArm> {
        arms.require_named(DevOp::MoeUpNvfp4, "activation", mlp.act)?;
        arms.require_flag(DevOp::MoeUpNvfp4, "bias", mlp.bias)?;
        arms.require(DevOp::MoeRouterSigmoidTopk, "groups", mlp.n_group)?;
        arms.require(DevOp::MoeRouterSigmoidTopk, "groups", mlp.topk_group)?;
        arms.require(DevOp::MoeRouterSigmoidTopk, "experts", g.experts)?;
        arms.require(DevOp::MoeRouterSigmoidTopk, "top-k", g.top_k)?;
        Ok(Self { g, arms: *arms })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.g
    }

    /// The MoE of `module` (`<...>.mixer`): `mixed = bf16(routed(hn) + shared(hn))`.
    pub fn emit(&self, p: &mut Program, module: &str, hn: u32, mixed: u32, dep: u32) -> u32 {
        let g = &self.g;
        let (hidden, e, k) = (g.hidden, g.experts, g.top_k);
        let (inter, sinter) = (g.inter, g.shared_inter);
        // The routed experts' input width.
        let kdim = g.latent.unwrap_or(hidden);
        let (rows, prefill) = (p.rows, p.prefill);
        if !prefill {
            self.arms
                .require(DevOp::MoeUpNvfp4, "decode rows", rows)
                .unwrap_or_else(|err| panic!("{module}: {err}"));
        }
        let tk = u64::from(rows) * u64::from(k);
        // Router logits: K-split partials for the decode GEMV, one tiled GEMM pass in prefill.
        let ks: u32 = if prefill { 1 } else { 16 };
        let logits = p.b.tensor(
            "act.moe.logits",
            u64::from(ks) * u64::from(rows) * u64::from(e) * 4,
        );
        let table = p.b.tensor("act.moe.table", tk * 8);
        let gate = p.b.tensor(
            &format!("{module}.gate.weight"),
            u64::from(e) * u64::from(hidden) * 2,
        );
        let bias = p.b.tensor(
            &format!("{module}.gate.e_score_correction_bias"),
            u64::from(e) * 4,
        );
        let d0 = p.emit(
            MoeRouterLogitsF32Args {
                logits,
                x: hn,
                W: gate,
                T: rows,
                E: e,
                H: hidden,
                KS: ks,
            },
            &[dep],
        );
        let d1 = p.emit(
            MoeRouterSigmoidTopkArgs {
                table,
                logits,
                bias,
                // Decode: an inactive slot routes to no expert (op 233 writes the unused
                // sentinel). Prefill tables must stay sentinel-free (op 238).
                active: if prefill { TENSOR_NONE } else { p.active },
                T: rows,
                E: e,
                k,
                KS: ks,
                flags: u32::from(g.norm_topk),
                scale: g.routed_scale,
            },
            &[d0],
        );
        // Decode: fc1 reads only hn, so it does not wait on the one-block-per-token top-k (which
        // then runs under fc1 on the other blocks); the routed up waits on both. The global queue
        // ranked the top-k one step after fc1 and the shared up, so blocks claimed it only after
        // finishing those and the routed up waited on its tail: queue it right behind the logits.
        if !prefill {
            p.b.gq_hoist(d1);
        }
        let fc1_dep = if prefill { d1 } else { dep };
        let (xl, d2) = match g.latent {
            Some(latent) => {
                let xl = p.act("moe.xl", latent);
                let fc1 = format!("{module}.fc1_latent_proj.weight");
                (xl, bf16_proj(p, xl, hn, &fc1, latent, hidden, fc1_dep))
            }
            None => (hn, fc1_dep),
        };
        // Prefill runs the W4A4 arm (NVFP4 q tensors, no bf16 intermediate).
        let fu = if prefill {
            TENSOR_NONE
        } else {
            p.b.tensor("act.moe.fu", tk * u64::from(inter) * 2)
        };
        // Prefill: bf16 ungated expert outputs (down mode 2), gated in the combine.
        let part = p.b.tensor(
            "act.moe.part",
            tk * u64::from(kdim) * if prefill { 2 } else { 4 },
        );
        let up = expert_weights(p, module, "up_proj", e, inter, kdim);
        let down = expert_weights(p, module, "down_proj", e, kdim, inter);
        // Shared expert: dense (E = 1, no table) on hn.
        let sp = format!("{module}.shared_experts");
        let sup = shared_weights(p, &format!("{sp}.up_proj"), sinter, hidden);
        let sdown = shared_weights(p, &format!("{sp}.down_proj"), hidden, sinter);
        let sout = p.act("moe.sout", hidden);
        let none = TENSOR_NONE;
        let mut shared_down = None;
        let d4 = if prefill {
            let meta = p.b.tensor("act.moe.meta", (3 * u64::from(e) + 1) * 4);
            let row_token = p.b.tensor("act.moe.row_token", tk * 4);
            let row_part = p.b.tensor("act.moe.row_partidx", tk * 4);
            let row_gate = p.b.tensor("act.moe.row_gate", tk * 4);
            let da = p.emit(
                MoeAlignNvfp4Args {
                    meta,
                    table,
                    row_token,
                    row_partidx: row_part,
                    row_gate,
                    T: rows,
                    E: e,
                    k,
                },
                &[d2],
            );
            // W4A4 (the peer's NVFP4 activations): xl -> q tensor; up writes the q tensor of
            // relu^2 for down. Input scales: the experts' per-layer input_scale vectors (max).
            let [in_up, in_dn] = ["up_proj", "down_proj"].map(|proj| {
                let name = crate::derived::cat(
                    (0..e)
                        .map(|x| format!("{module}.experts.{x}.{proj}.input_scale"))
                        .collect(),
                );
                p.b.tensor(&name, u64::from(e) * 4)
            });
            let xq =
                p.b.tensor("act.moe.xq", nvfp4_q_bytes(u64::from(rows), kdim));
            let fq = p.b.tensor("act.moe.fq", nvfp4_q_bytes(tk, inter));
            let dq = p.emit(
                MoeQuantNvfp4Args {
                    q: xq,
                    x: xl,
                    in_scale: in_up,
                    R: rows,
                    K: kdim,
                    n: e,
                },
                &[d2],
            );
            let du = p.emit(
                MoeUpNvfp4PfArgs {
                    fu_g: fq,
                    x: xq,
                    meta,
                    W: up[0],
                    S: up[1],
                    gs: up[2],
                    row_token,
                    s2: in_dn,
                    T: rows,
                    k,
                    I: inter,
                    K: kdim,
                    E: e,
                    w4a4: 1,
                    n2: e,
                },
                &[da, dq],
            );
            p.emit(
                MoeDownNvfp4PfArgs {
                    out: part,
                    fu_g: fq,
                    meta,
                    W: down[0],
                    S: down[1],
                    gs: down[2],
                    row_partidx: row_part,
                    row_gate,
                    T: rows,
                    k,
                    H: kdim,
                    I: inter,
                    E: e,
                    mode: 2,
                    w4a4: 1,
                },
                &[du],
            )
        } else {
            // The shared expert reads hn only: its up streams while the top-k and fc1 finish
            // and its down runs between the routed up and down, so the blocks fill each routed
            // op's dependency wait with independent work instead of idling.
            let sfu = p.act("moe.sfu", sinter);
            let su = p.emit(
                MoeUpNvfp4Args {
                    fu: sfu,
                    x: hn,
                    table: none,
                    W: sup[0],
                    S: sup[1],
                    gs: sup[2],
                    T: rows,
                    k: 1,
                    I: sinter,
                    K: hidden,
                    E: 1,
                },
                &[dep],
            );
            let du = p.emit(
                MoeUpNvfp4Args {
                    fu,
                    x: xl,
                    table,
                    W: up[0],
                    S: up[1],
                    gs: up[2],
                    T: rows,
                    k,
                    I: inter,
                    K: kdim,
                    E: e,
                },
                &[d1, d2],
            );
            shared_down = Some(p.emit(
                MoeDownNvfp4Args {
                    out: sout,
                    fu: sfu,
                    table: none,
                    W: sdown[0],
                    S: sdown[1],
                    gs: sdown[2],
                    T: rows,
                    k: 1,
                    H: hidden,
                    I: sinter,
                    E: 1,
                    mode: 1,
                },
                &[su],
            ));
            p.emit(
                MoeDownNvfp4Args {
                    out: part,
                    fu,
                    table,
                    W: down[0],
                    S: down[1],
                    gs: down[2],
                    T: rows,
                    k,
                    H: kdim,
                    I: inter,
                    E: e,
                    mode: 0,
                },
                &[du],
            )
        };
        let combined = p.act(
            if g.latent.is_some() {
                "moe.lat"
            } else {
                "moe.rout"
            },
            kdim,
        );
        let d5 = p.emit(
            MoeCombinePfArgs {
                out: combined,
                part,
                table: if prefill { table } else { none },
                H: kdim,
                k,
                T: rows,
                bf16: prefill as u32,
                ..Default::default()
            },
            &[d4],
        );
        let (rout, d6) = match g.latent {
            Some(latent) => {
                let rout = p.act("moe.rout", hidden);
                let fc2 = format!("{module}.fc2_latent_proj.weight");
                (rout, bf16_proj(p, rout, combined, &fc2, hidden, latent, d5))
            }
            None => (combined, d5),
        };
        let d8 = if prefill {
            // W4A4 shared expert: hn and relu^2 as NVFP4 q tensors with their input_scales.
            let [in_su, in_sd] = ["up_proj", "down_proj"]
                .map(|proj| p.b.tensor(&format!("{sp}.{proj}.input_scale"), 4));
            let hq =
                p.b.tensor("act.moe.hq", nvfp4_q_bytes(u64::from(rows), hidden));
            let sfq =
                p.b.tensor("act.moe.sfq", nvfp4_q_bytes(u64::from(rows), sinter));
            let dq = p.emit(
                MoeQuantNvfp4Args {
                    q: hq,
                    x: hn,
                    in_scale: in_su,
                    R: rows,
                    K: hidden,
                    n: 1,
                },
                &[d6],
            );
            let d7 = p.emit(
                MoeUpNvfp4PfArgs {
                    fu_g: sfq,
                    x: hq,
                    meta: none,
                    W: sup[0],
                    S: sup[1],
                    gs: sup[2],
                    row_token: none,
                    s2: in_sd,
                    T: rows,
                    k: 1,
                    I: sinter,
                    K: hidden,
                    E: 1,
                    w4a4: 1,
                    n2: 1,
                },
                &[dq],
            );
            // Mode 3 folds `mixed = bf16(rout + bf16(shared))` into the shared down's epilogue
            // (rout in the row_partidx slot), replacing the Residual pass below.
            let radd = shared_radd();
            let d8 = p.emit(
                MoeDownNvfp4PfArgs {
                    out: if radd { mixed } else { sout },
                    fu_g: sfq,
                    meta: none,
                    W: sdown[0],
                    S: sdown[1],
                    gs: sdown[2],
                    row_partidx: if radd { rout } else { none },
                    row_gate: none,
                    T: rows,
                    k: 1,
                    H: hidden,
                    I: sinter,
                    E: 1,
                    mode: if radd { 3 } else { 1 },
                    w4a4: 1,
                },
                &[d7],
            );
            if radd {
                return d8;
            }
            d8
        } else {
            shared_down.expect("decode shared expert")
        };
        // mixed = bf16(routed + shared); the block residual is deferred onto the next norm.
        let deps: &[u32] = if prefill { &[d8] } else { &[d6, d8] };
        p.emit(
            ResidualArgs {
                out: mixed,
                a: rout,
                b: sout,
                pre: none,
                n: rows * hidden,
                scale: 1.0,
            },
            deps,
        )
    }
}

/// `PLOW_NVFP4_SHARED_RADD` (default on; 0 = the separate Residual pass, for A/B): prefill shared
/// down writes `mixed` itself (op 238 mode 3).
fn shared_radd() -> bool {
    std::env::var("PLOW_NVFP4_SHARED_RADD").map_or(true, |v| v != "0")
}

/// The shared expert's NVFP4 tensors of `module` (`[n, k]`), bound directly (E = 1):
/// W u8 `[n][k/2]`, S e4m3 `[n][k/16]`, gs f32 rank-0.
fn shared_weights(p: &mut Program, module: &str, n: u32, k: u32) -> [u32; 3] {
    let (n, k) = (u64::from(n), u64::from(k));
    let parts = [
        ("weight", n * k / 2),
        ("weight_scale", n * k / 16),
        ("weight_scale_2", 4),
    ];
    parts.map(|(suffix, bytes)| p.b.tensor(&format!("{module}.{suffix}"), bytes))
}

/// Routed-expert NVFP4 tensors of `<module>.experts.<x>.<proj>` concatenated over the experts.
fn expert_weights(p: &mut Program, module: &str, proj: &str, e: u32, n: u32, k: u32) -> [u32; 3] {
    let (n, k, e64) = (u64::from(n), u64::from(k), u64::from(e));
    let parts = [
        ("weight", n * k / 2),
        ("weight_scale", n * k / 16),
        ("weight_scale_2", 4),
    ];
    parts.map(|(suffix, bytes)| {
        let name = crate::derived::cat(
            (0..e)
                .map(|x| format!("{module}.experts.{x}.{proj}.{suffix}"))
                .collect(),
        );
        p.b.tensor(&name, e64 * bytes)
    })
}

/// Bytes of an NVFP4 q tensor of `rows` x `k` (op_moe_nvfp4_pf.cuh): e2m1 `[rows][k/2]`, e4m3
/// `[rows][k/16]`, one f32 alpha (16 B slot).
pub(crate) fn nvfp4_q_bytes(rows: u64, k: u32) -> u64 {
    rows * u64::from(k) / 2 + rows * u64::from(k) / 16 + 16
}
