use crate::hetero::WeightEncoding;
use crate::program::{Packet, Program};
use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "plow-hetero-v3";
#[cfg(test)]
#[path = "hetero_channel_tests.rs"]
mod tests;
type Result<T> = std::result::Result<T, String>;

fn need(ok: bool, why: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("channel MLP: {why}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    ChannelMlp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialDtype {
    F32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Slice {
    pub rows: [u32; 2],
    pub cols: [u32; 2],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Weight {
    pub tensor: String,
    pub scale: Option<String>,
    pub rows: u32,
    pub cols: u32,
    pub gpu: Slice,
    pub ane: Slice,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub layer: u32,
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub layer: u32,
    /// Half-open instruction range of the original, complete GPU MLP plus residual.
    pub insts: [u32; 2],
    pub input: String,
    pub residual: String,
    pub intermediate: String,
    pub down_output: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Partials {
    pub gpu: String,
    pub ane: String,
    pub rows: u32,
    pub cols: u32,
    pub dtype: PartialDtype,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgPlan {
    pub prog: u32,
    pub rows: u32,
    pub min_rows: u32,
    pub max_rows: u32,
    pub call_rows: u32,
    pub original_sha256: String,
    pub partials: Partials,
    pub spans: Vec<Span>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelPlan {
    pub schema: String,
    pub mode: Mode,
    pub arch: String,
    pub hidden: u32,
    pub inter: u32,
    pub ane_channels: u32,
    pub weight_encoding: WeightEncoding,
    pub layers: Vec<Layer>,
    pub programs: Vec<ProgPlan>,
}

fn tensor(p: &Packet<'_>, name: &str, bytes: u64, exact: bool) -> Result<u16> {
    let mut found = p.tensors.iter().enumerate().filter(|(_, t)| t.name == name);
    let (h, t) = found
        .next()
        .ok_or_else(|| format!("channel MLP: missing tensor {name}"))?;
    need(
        found.next().is_none() && h < TENSOR_NONE16 as usize,
        "ambiguous tensor",
    )?;
    need(
        !name.is_empty()
            && if exact {
                t.bytes == bytes
            } else {
                t.bytes >= bytes
            },
        "tensor byte extent",
    )?;
    Ok(h as u16)
}

impl ChannelPlan {
    pub fn validate(&self, p: &Packet<'_>) -> Result<()> {
        need(
            self.schema == SCHEMA && self.arch == "metal3" && !p.tp,
            "schema/target/topology",
        )?;
        need(
            self.hidden > 0
                && self.inter > 0
                && self.ane_channels > 0
                && self.ane_channels < self.inter,
            "nonempty complementary channels required",
        )?;
        need(
            self.hidden % 32 == 0 && self.inter % 32 == 0 && self.ane_channels % 32 == 0,
            "32-column packing alignment",
        )?;
        need(
            !self.layers.is_empty() && !self.programs.is_empty(),
            "empty plan",
        )?;
        let mut last_layer = None;
        for layer in &self.layers {
            need(
                last_layer.is_none_or(|l| l < layer.layer),
                "duplicate/unordered layers",
            )?;
            last_layer = Some(layer.layer);
            for (w, down) in [
                (&layer.gate, false),
                (&layer.up, false),
                (&layer.down, true),
            ] {
                let (h, i, a) = (self.hidden, self.inter, self.ane_channels);
                let (rows, cols, ane, gpu) = if down {
                    (
                        h,
                        i,
                        Slice {
                            rows: [0, h],
                            cols: [0, a],
                        },
                        Slice {
                            rows: [0, h],
                            cols: [a, i],
                        },
                    )
                } else {
                    (
                        i,
                        h,
                        Slice {
                            rows: [0, a],
                            cols: [0, h],
                        },
                        Slice {
                            rows: [a, i],
                            cols: [0, h],
                        },
                    )
                };
                need(
                    w.rows == rows && w.cols == cols && w.ane == ane && w.gpu == gpu,
                    "weight partition coverage",
                )?;
                self.weight_handles(p, w)?;
            }
        }
        let mut last_prog = None;
        for pp in &self.programs {
            need(
                last_prog.is_none_or(|g| g < pp.prog),
                "duplicate/unordered programs",
            )?;
            last_prog = Some(pp.prog);
            need(
                (pp.prog as usize) < p.prefill_count,
                "decode cannot be offloaded",
            )?;
            let prog = p
                .programs
                .get(pp.prog as usize)
                .ok_or("channel MLP: missing program")?;
            need(
                !prog.role.is_packed_sibling()
                    && prog.rows == pp.rows
                    && pp.rows == 128
                    && pp.min_rows == 64
                    && pp.max_rows == 128
                    && pp.call_rows == 128,
                "initial fixed-call row contract",
            )?;
            need(
                pp.original_sha256 == crate::live_kv::program_digest(prog),
                "stale original program",
            )?;
            let partial = &pp.partials;
            need(
                partial.rows == pp.rows
                    && partial.cols == self.hidden
                    && !partial.gpu.is_empty()
                    && !partial.ane.is_empty()
                    && partial.gpu != partial.ane
                    && !p
                        .tensors
                        .iter()
                        .any(|t| t.name == partial.gpu || t.name == partial.ane),
                "FP32 scratch must not alias packet tensors",
            )?;
            need(pp.spans.len() == self.layers.len(), "layer span coverage")?;
            let mut end = 0;
            for (span, layer) in pp.spans.iter().zip(&self.layers) {
                need(
                    span.layer == layer.layer
                        && span.insts[0] >= end
                        && span.insts[0] < span.insts[1],
                    "overlapping/empty layer spans",
                )?;
                self.validate_span(p, prog, span, layer)?;
                end = span.insts[1];
            }
        }
        Ok(())
    }

    fn weight_handles(&self, p: &Packet<'_>, w: &Weight) -> Result<(u16, u16)> {
        let elements = u64::from(w.rows) * u64::from(w.cols);
        let bytes = match self.weight_encoding {
            WeightEncoding::Bf16 => elements
                .checked_mul(2)
                .ok_or("channel MLP: weight size overflow")?,
            WeightEncoding::Fp8 => elements,
            WeightEncoding::Mxfp4 => elements / 2,
        };
        let wh = tensor(p, &w.tensor, bytes, true)?;
        let sh = match (self.weight_encoding, &w.scale) {
            (WeightEncoding::Bf16, None) => TENSOR_NONE16,
            (WeightEncoding::Fp8, Some(s)) => tensor(p, s, u64::from(w.rows) * 4, true)?,
            (WeightEncoding::Mxfp4, Some(s)) => tensor(p, s, elements / 32, true)?,
            _ => return Err("channel MLP: weight scale encoding".into()),
        };
        Ok((wh, sh))
    }

    fn validate_span(
        &self,
        p: &Packet<'_>,
        prog: &Program<'_>,
        s: &Span,
        layer: &Layer,
    ) -> Result<()> {
        let insts = prog
            .insts
            .get(s.insts[0] as usize..s.insts[1] as usize)
            .ok_or("channel MLP: span bounds")?;
        let hbytes = u64::from(prog.rows) * u64::from(self.hidden) * 2;
        let x = tensor(p, &s.input, hbytes, false)?;
        let residual = tensor(p, &s.residual, hbytes, false)?;
        let z = tensor(
            p,
            &s.intermediate,
            u64::from(prog.rows) * u64::from(self.inter) * 2,
            false,
        )?;
        let y = tensor(p, &s.down_output, hbytes, false)?;
        let handles = [x, residual, z, y];
        need(
            handles
                .iter()
                .enumerate()
                .all(|(i, h)| !handles[..i].contains(h)),
            "preserved input/residual alias scratch",
        )?;
        let norm = s.insts[0]
            .checked_sub(1)
            .and_then(|i| prog.insts.get(i as usize))
            .ok_or("channel MLP: missing input producer")?;
        need(
            norm.op == DevOp::RmsNorm as u16
                && norm.t[..2] == [x, residual]
                && norm.i[..2] == [prog.rows, self.hidden],
            "preserved normalized input producer",
        )?;
        let (wg, sg) = self.weight_handles(p, &layer.gate)?;
        let (wu, su) = self.weight_handles(p, &layer.up)?;
        let (wd, sd) = self.weight_handles(p, &layer.down)?;
        // Only these complete, unfurled SwiGLU shapes are replaceable. No runtime graph search.
        let fused = match self.weight_encoding {
            WeightEncoding::Bf16 => DevOp::GemmGlu,
            WeightEncoding::Fp8 => DevOp::GemmGluFp8,
            WeightEncoding::Mxfp4 => DevOp::GemmGluMxfp4,
        };
        need(
            insts.len() == 3 || insts.len() == 5,
            "unsupported fused MLP span",
        )?;
        if insts.len() == 3 {
            let d = &insts[0];
            let scales = match self.weight_encoding {
                WeightEncoding::Bf16 => [
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    wu,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
                WeightEncoding::Fp8 => [TENSOR_NONE16, sg, wu, su, TENSOR_NONE16],
                WeightEncoding::Mxfp4 => [sg, su, wu, TENSOR_NONE16, TENSOR_NONE16],
            };
            need(
                d.op == fused as u16
                    && d.t[..3] == [z, x, wg]
                    && d.t[3..] == scales
                    && d.i[..3] == [prog.rows, self.inter, self.hidden]
                    && d.i[3..] == [0, 0, 1, 0, 0]
                    && d.fj == [0; 3],
                "gate/up SwiGLU binding",
            )?;
        } else {
            let (g, u, act) = (&insts[0], &insts[1], &insts[2]);
            self.validate_projection(g, [prog.rows, self.inter, self.hidden], x, wg, sg)?;
            self.validate_projection(u, [prog.rows, self.inter, self.hidden], x, wu, su)?;
            need(
                g.t[0] != u.t[0] && !handles.contains(&g.t[0]) && !handles.contains(&u.t[0]),
                "gate/up scratch alias",
            )?;
            for h in [g.t[0], u.t[0]] {
                tensor(
                    p,
                    p.tensors
                        .get(h as usize)
                        .ok_or("channel MLP: gate/up scratch")?
                        .name,
                    u64::from(prog.rows) * u64::from(self.inter) * 2,
                    false,
                )?;
            }
            need(
                act.op == DevOp::Glu as u16
                    && act.t[..3] == [z, g.t[0], u.t[0]]
                    && act.t[3..] == [TENSOR_NONE16; 5]
                    && act.i
                        == [
                            prog.rows
                                .checked_mul(self.inter)
                                .ok_or("channel MLP: GLU extent overflow")?,
                            1,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    && act.fj == [0; 3],
                "SwiGLU binding",
            )?;
        }
        let down = &insts[insts.len() - 2];
        self.validate_projection(down, [prog.rows, self.hidden, self.inter], z, wd, sd)?;
        need(down.t[0] == y, "down output binding")?;
        let r = &insts[insts.len() - 1];
        need(
            r.op == DevOp::Residual as u16
                && r.t[..3] == [residual, residual, y]
                && r.t[3..] == [TENSOR_NONE16; 5]
                && r.i
                    == [
                        prog.rows
                            .checked_mul(self.hidden)
                            .ok_or("channel MLP: residual extent overflow")?,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                && r.fj == [1f32.to_bits(), 0, 0],
            "single residual addition required",
        )?;
        Ok(())
    }

    fn validate_projection(
        &self,
        d: &DevInst64,
        dims: [u32; 3],
        input: u16,
        weight: u16,
        scale: u16,
    ) -> Result<()> {
        let op = DevOp::from_u16(d.op);
        let (accepted, rest) = match self.weight_encoding {
            WeightEncoding::Bf16 => (
                matches!(
                    op,
                    Some(
                        DevOp::Gemm
                            | DevOp::GemmSmall
                            | DevOp::GemmMed
                            | DevOp::GemmWide
                            | DevOp::GemmC5
                    )
                ),
                [TENSOR_NONE16; 5],
            ),
            WeightEncoding::Fp8 => (
                matches!(
                    op,
                    Some(
                        DevOp::GemmFp8
                            | DevOp::GemmSmallFp8
                            | DevOp::GemmMedFp8
                            | DevOp::GemmWideFp8
                            | DevOp::GemmC5Fp8
                    )
                ),
                [
                    TENSOR_NONE16,
                    scale,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
            ),
            WeightEncoding::Mxfp4 => (
                matches!(
                    op,
                    Some(
                        DevOp::GemmMxfp4
                            | DevOp::GemmSmallMxfp4
                            | DevOp::GemmMedMxfp4
                            | DevOp::GemmWideMxfp4
                            | DevOp::GemmC5Mxfp4
                    )
                ),
                [
                    scale,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                    TENSOR_NONE16,
                ],
            ),
        };
        // Plain BF16 emission carries eps in f0 even without a folded norm (i3=0).
        let floats = d.fj[1..] == [0, 0]
            && if self.weight_encoding == WeightEncoding::Bf16 {
                f32::from_bits(d.fj[0]).is_finite()
            } else {
                d.fj[0] == 0
            };
        need(
            accepted
                && d.t[1..3] == [input, weight]
                && d.t[3..] == rest
                && d.i[..3] == dims
                && d.i[3..] == [0; 5]
                && floats,
            "projection binding/encoding",
        )
    }
}
