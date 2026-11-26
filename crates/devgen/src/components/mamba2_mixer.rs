//! The Mamba-2 mixer: in_proj -> causal conv + SiLU (`Mamba2Conv`) -> SSD (`Mamba2Step` decode;
//! `Mamba2ChunkState/Pass/Out` prefill, continuing from the slot state) -> grouped gated RMSNorm
//! (`Mamba2GatedNorm`) -> out_proj.
//!
//! in_proj's output columns are `z [0, d_inner) | xBC [d_inner, d_inner + conv_dim) | dt`, read
//! in place through the ops' `pitch | column << 16` sources. Slot state: conv history
//! `state.<family>.<layer>.conv` (bf16 `[slots][conv_dim][kernel - 1]`) and SSM state
//! `state.<family>.<layer>.ssm` (`[slots][heads][head_dim][state]` of `state_bytes` each).
use kernelcaps::arms::{Arms, MissingArm};
use packet::dev::{DevOp, TENSOR_NONE};
use packet::typed_abi::{
    Mamba2ChunkOutArgs, Mamba2ChunkPassArgs, Mamba2ChunkStateArgs, Mamba2ConvArgs,
    Mamba2GatedNormArgs, Mamba2StepArgs,
};

use super::{fp8_proj, Program, Proj};

/// The chunk length of the prefill scan.
pub(crate) const CHUNK: u32 = 128;

/// Model values the mixer is built from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub state: u32,
    pub groups: u32,
    pub conv_kernel: u32,
    pub conv_bias: bool,
    pub chunk: u32,
    pub dt_limit: (f32, f32),
    pub eps: f32,
    /// Bytes per SSM state element (2 = fp16).
    pub state_bytes: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct Mamba2Mixer {
    g: Geometry,
    /// State buffer names: `state.<family>.<layer>.{conv,ssm}`.
    family: &'static str,
    /// out_proj is the FP8 per-tensor linear: on its static prefill route the gated norm writes
    /// out_proj's e4m3 input itself (the same arithmetic as the QuantFp8 pass it replaces).
    out_proj_fp8: bool,
}

impl Geometry {
    pub fn d_inner(&self) -> u32 {
        self.heads * self.head_dim
    }
    /// x, B and C: the conv's channels.
    pub fn conv_dim(&self) -> u32 {
        self.d_inner() + 2 * self.groups * self.state
    }
    pub fn in_proj_width(&self) -> u32 {
        self.d_inner() + self.conv_dim() + self.heads
    }
}

impl Mamba2Mixer {
    pub fn new(
        g: Geometry,
        family: &'static str,
        out_proj_fp8: bool,
        arms: &Arms,
    ) -> Result<Self, MissingArm> {
        arms.require(DevOp::Mamba2Step, "head_dim", g.head_dim)?;
        arms.require(DevOp::Mamba2Step, "state", g.state)?;
        arms.require(DevOp::Mamba2ChunkState, "chunk", g.chunk)?;
        arms.require(DevOp::Mamba2Conv, "kernel", g.conv_kernel)?;
        arms.require_flag(DevOp::Mamba2Conv, "bias", g.conv_bias)?;
        arms.require(DevOp::Mamba2Conv, "source pitch", g.in_proj_width())?;
        arms.require(DevOp::Mamba2GatedNorm, "group", g.d_inner() / g.groups)?;
        Ok(Self {
            g,
            family,
            out_proj_fp8,
        })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.g
    }

    /// The mixer of `module` (`<...>.mixer`) at `layer`: `mixed = mixer(hn)`.
    #[allow(clippy::too_many_arguments)]
    pub fn emit(
        &self,
        p: &mut Program,
        layer: usize,
        module: &str,
        hn: u32,
        mixed: u32,
        dep: u32,
        proj: &mut Proj,
    ) -> u32 {
        let g = &self.g;
        let (hidden, heads, groups) = (g.hidden, g.heads, g.groups);
        let (hd, ns) = (g.head_dim, g.state);
        let d_inner = g.d_inner();
        let conv_dim = g.conv_dim();
        let width = g.in_proj_width();
        let kw = g.conv_kernel;
        let (dt_lo, dt_hi) = g.dt_limit;
        let slots = u64::from(p.slots);
        let proj_t = p.act(&format!("{}.proj", self.family), width);
        let xbc = p.act(&format!("{}.xbc", self.family), conv_dim);
        let y = p.act(&format!("{}.y", self.family), d_inner);
        let gated = p.act(&format!("{}.gated", self.family), d_inner);
        let history = p.b.tensor(
            &format!("state.{}.{layer}.conv", self.family),
            slots * u64::from(conv_dim) * u64::from(kw - 1) * 2,
        );
        let state = p.b.tensor(
            &format!("state.{}.{layer}.ssm", self.family),
            slots * u64::from(heads) * u64::from(hd) * u64::from(ns) * u64::from(g.state_bytes),
        );
        let w = |p: &mut Program, name: &str, elems: u32| {
            p.b.tensor(&format!("{module}.{name}"), u64::from(elems) * 2)
        };
        let cw = w(p, "conv1d.weight", conv_dim * kw);
        let cb = w(p, "conv1d.bias", conv_dim);
        let alog = w(p, "A_log", heads);
        let dskip = w(p, "D", heads);
        let dtb = w(p, "dt_bias", heads);
        let gamma = w(p, "norm.weight", d_inner);
        let dp = proj(
            p,
            proj_t,
            hn,
            &format!("{module}.in_proj"),
            width,
            hidden,
            dep,
        );
        let xbc_src = width | (d_inner << 16);
        let dt_src = width | ((d_inner + conv_dim) << 16);
        let z_src = width;
        let (rows, prefill, active) = (p.rows, p.prefill, p.active);
        let dc = p.emit(
            Mamba2ConvArgs {
                out: xbc,
                x: proj_t,
                weight: cw,
                bias: cb,
                history,
                active: if prefill { TENSOR_NONE } else { active },
                C: conv_dim,
                W: kw,
                rows,
                T: if prefill { rows } else { 0 },
                prefill: prefill as u32,
                source: xbc_src,
                ..Default::default()
            },
            &[dp],
        );
        let ds = if prefill {
            // Packed: each request's rows form their own chunks (op_mamba_pf.cuh), at most one
            // partial chunk per request beyond the bucket's.
            let extra = if p.packed { u64::from(p.slots) } else { 0 };
            let chunks = u64::from(rows / g.chunk) + extra;
            let cstate = p.b.tensor(
                &format!("act.{}.cstate", self.family),
                chunks * u64::from(heads) * u64::from(hd) * u64::from(ns) * 4,
            );
            let scan_rows = u64::from(rows) + extra * u64::from(g.chunk);
            let dtv = p.b.tensor(
                &format!("act.{}.dt", self.family),
                u64::from(heads) * scan_rows * 4,
            );
            let cum = p.b.tensor(
                &format!("act.{}.cum", self.family),
                u64::from(heads) * scan_rows * 4,
            );
            let d1 = p.emit(
                Mamba2ChunkStateArgs {
                    cstate,
                    xbc,
                    proj: proj_t,
                    A_log: alog,
                    dt_bias: dtb,
                    dtv,
                    cum,
                    H: heads,
                    P: hd,
                    R: rows,
                    T: rows,
                    N: ns,
                    G: groups,
                    dt_lo,
                    dt_hi,
                    dt_source: dt_src,
                },
                &[dc],
            );
            let d2 = p.emit(
                Mamba2ChunkPassArgs {
                    cstate,
                    cum,
                    state,
                    H: heads,
                    P: hd,
                    R: rows,
                    T: rows,
                    N: ns,
                    state_bytes: g.state_bytes,
                },
                &[d1],
            );
            p.emit(
                Mamba2ChunkOutArgs {
                    y,
                    xbc,
                    cstate,
                    cum,
                    dtv,
                    D: dskip,
                    H: heads,
                    P: hd,
                    R: rows,
                    T: rows,
                    N: ns,
                    G: groups,
                },
                &[d2],
            )
        } else {
            // The stochastic-rounding seed is `in.active` word `slots` (runtime-written per step).
            p.emit(
                Mamba2StepArgs {
                    y,
                    xbc,
                    proj: proj_t,
                    A_log: alog,
                    D: dskip,
                    dt_bias: dtb,
                    state,
                    active,
                    H: heads,
                    P: hd,
                    N: ns,
                    G: groups,
                    B: rows,
                    state_bytes: g.state_bytes,
                    seed_word: p.slots,
                    layer: layer as u32,
                    dt_lo,
                    dt_hi,
                    dt_source: dt_src,
                },
                &[dc],
            )
        };
        let fp8_out =
            self.out_proj_fp8 && prefill && fp8_proj::static_route(p.isa, rows, hidden, d_inner);
        let (gout, gscale) = if fp8_out {
            let xq = p.b.tensor(
                &format!("act.fp8.x{d_inner}"),
                u64::from(rows) * u64::from(d_inner),
            );
            (xq, p.b.tensor(&format!("{module}.out_proj.input_scale"), 4))
        } else {
            (gated, TENSOR_NONE)
        };
        let dn = p.emit(
            Mamba2GatedNormArgs {
                out: gout,
                y,
                z: proj_t,
                gamma,
                active: if prefill { TENSOR_NONE } else { active },
                in_scale: gscale,
                groups,
                group: d_inner / groups,
                rows,
                fp8_out: fp8_out as u32,
                eps: g.eps,
                source: z_src,
            },
            &[ds],
        );
        if fp8_out {
            p.quant = Some((gated, d_inner, gscale));
        }
        proj(
            p,
            mixed,
            gated,
            &format!("{module}.out_proj"),
            hidden,
            d_inner,
            dn,
        )
    }
}
