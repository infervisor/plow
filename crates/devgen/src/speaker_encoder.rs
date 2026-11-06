//! ECAPA-TDNN speaker embeddings, lowered through generic FP32 packet ops with explicit BF16
//! tensor boundaries. The input is an unnormalized F32 log-mel sequence; frontend tables travel
//! with the packet. No batch normalization or output L2 normalization is applied.

use std::collections::BTreeMap;
use std::path::Path;

use crate::fish_codec::Seq;
use crate::pipeline::{
    Activation, BinaryF32Stage, BinaryOp, Conv1dF32Stage, CopyColsF32Stage, GatherRowsF32Stage,
    PacketPrefix, PadMode, ReduceRowsF32Stage, ReduceRowsOp, TensorRef, UnaryF32Stage,
};

pub const PACKET: &str = "speaker_encoder.pkt";
pub const DRIVER: &str = "speaker.encode.v1";
const CAPACITIES: &[u32] = &[128, 256, 512, 1024, 2048];

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Cfg {
    model_type: String,
    version: u32,
    channels: Vec<u32>,
    kernel_sizes: Vec<u32>,
    dilations: Vec<u32>,
    res2net_scale: u32,
    se_channels: u32,
    attention_channels: u32,
    embedding_dim: u32,
    batch_norm: bool,
    l2_normalize: bool,
    statistics_eps: f32,
    compute_dtype: String,
    conv_bias_round: String,
    weight_layout: String,
    frontend: serde_json::Value,
}

impl Cfg {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let c: Self = serde_json::from_slice(bytes).map_err(|e| format!("speaker encoder: {e}"))?;
        let n = c.channels.len();
        let frontend = serde_json::json!({
            "compute_dtype": "float32", "sample_rate": 24000, "n_mels": 128,
            "n_fft": 1024, "hop": 256, "win_length": 1024, "pad": 384,
            "pad_mode": "reflect", "center": false, "magnitude_eps": 1e-9,
            "mel_norm": "slaney", "mel_htk": false, "fmin": 0, "fmax": 12000,
            "log_floor": 1e-5, "sentence_normalize": false
        });
        if c.model_type != "ecapa_tdnn"
            || c.version != 1
            || n < 3
            || c.kernel_sizes.len() != n
            || c.dilations.len() != n
            || c.channels
                .iter()
                .chain(&c.kernel_sizes)
                .chain(&c.dilations)
                .chain([
                    &c.res2net_scale,
                    &c.se_channels,
                    &c.attention_channels,
                    &c.embedding_dim,
                ])
                .any(|&x| x == 0)
            || c.res2net_scale < 2
            || c.kernel_sizes.iter().any(|&k| k % 2 == 0)
            || c.channels[..n - 1]
                .iter()
                .any(|&x| x != c.channels[0] || x % c.res2net_scale != 0)
            || c.channels[1..n - 1]
                .iter()
                .try_fold(0u32, |a, &b| a.checked_add(b))
                != Some(c.channels[n - 1])
            || c.channels[n - 1].checked_mul(3).is_none()
            || c.batch_norm
            || c.l2_normalize
            || c.compute_dtype != "bfloat16"
            || c.conv_bias_round != "before_add"
            || c.weight_layout != "out_tap_in"
            || c.statistics_eps != 1e-12
            || c.frontend != frontend
        {
            return Err("unsupported speaker encoder configuration".into());
        }
        c.min_rows()?;
        Ok(c)
    }

    fn min_rows(&self) -> Result<u32, String> {
        self.kernel_sizes
            .iter()
            .zip(&self.dilations)
            .try_fold(1, |m, (&k, &d)| {
                let pad = (k - 1).checked_mul(d).ok_or("speaker padding overflows")? / 2;
                Ok(m.max(pad.checked_add(1).ok_or("speaker padding overflows")?))
            })
    }

    fn width(&self) -> u32 {
        *self.channels.last().unwrap()
    }
}

pub fn lower_speaker_encoder(
    dir: &Path,
    n_cu: u32,
    target: u32,
) -> Result<(packet::devbuild::Model, packet::devbuild::SectionData), String> {
    let cfg = Cfg::parse(&std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?)?;
    let max = *CAPACITIES.last().unwrap();
    if cfg.min_rows()? > CAPACITIES[0] {
        return Err("speaker reflection exceeds smallest capacity".into());
    }
    let mut b = packet::devbuild::Builder::new(n_cu);
    b.set_tensor_dedup(true);
    let mel = b.tensor("in.spk.mel", u64::from(max) * 128 * 4);
    let length = b.tensor("in.spk.length", 4);
    let output = b.tensor("act.spk.embedding", u64::from(cfg.embedding_dim) * 4);
    let reciprocal = b.tensor("c.spk.reciprocal", u64::from(max + 1) * 4);
    b.tensor("w.frontend.mel", 128 * 513 * 4);
    b.tensor("w.frontend.window", 1024 * 4);
    let mut prefix = PacketPrefix {
        model: packet::devbuild::Model {
            n_cu,
            target,
            tensors: b.tensors(),
            progs: Vec::new(),
            kv_row_insts: Vec::new(),
            prog_t: Vec::new(),
            gen: Vec::new(),
        },
        programs: Vec::new(),
        input: mel,
        output,
        input_shape: vec![u64::from(max), 128],
    };
    let mut roles = BTreeMap::new();
    for &rows in CAPACITIES {
        let index = prefix.model.progs.len() as u32;
        prefix = program(prefix, &cfg, rows, length, reciprocal)?;
        roles.insert(format!("encode.r{rows}"), index);
    }
    let reader = crate::checkpoint::TensorReader::open(dir)?;
    for (h, tensor) in prefix.model.tensors.iter_mut().enumerate() {
        if h == reciprocal as usize {
            tensor.init = Some(
                (0..=max)
                    .flat_map(|n| {
                        let r = if n == 0 {
                            0.0
                        } else {
                            crate::config::bf16_round(1.0 / crate::config::bf16_round(n as f32))
                        };
                        r.to_le_bytes()
                    })
                    .collect(),
            );
        } else if let Some(name) = tensor.name.strip_prefix("w.") {
            let (dtype, bytes) = reader.read(name)?;
            if dtype != "F32"
                || bytes.len() as u64 != tensor.bytes
                || bytes.chunks_exact(4).any(|b| {
                    let value = f32::from_le_bytes(b.try_into().unwrap());
                    !value.is_finite()
                        || (!name.starts_with("frontend.") && value.to_bits() & 0xffff != 0)
                })
            {
                return Err(format!(
                    "speaker tensor {name}: expected finite F32 (BF16 weights), {} bytes",
                    tensor.bytes
                ));
            }
            tensor.init = Some(bytes);
        }
    }
    let section = section(&prefix, &cfg, roles, length, max)?;
    Ok((prefix.model, section))
}

struct Encoder {
    seq: Seq,
    rows: u32,
    length: u32,
}

impl Encoder {
    fn unary(&mut self, x: u32, rows: u32, width: u32, kind: Activation) -> Result<u32, String> {
        self.seq.unary(x, kind, rows, width)
    }

    fn round(&mut self, x: u32, rows: u32, width: u32) -> Result<u32, String> {
        self.unary(x, rows, width, Activation::RoundBf16)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy(
        &mut self,
        x: u32,
        out: &str,
        rows: u32,
        cols: u32,
        input_stride: u32,
        input_offset: u32,
        output_stride: u32,
        output_offset: u32,
    ) -> Result<u32, String> {
        let dep = self.seq.last;
        let e = self.seq.p().copy_cols_f32(
            x,
            &[dep],
            CopyColsF32Stage {
                output: TensorRef::Named(out),
                items: 1,
                rows,
                cols,
                in_item_stride: 0,
                in_stride: input_stride,
                in_offset: input_offset,
                out_item_stride: 0,
                out_stride: output_stride,
                out_offset: output_offset,
            },
            None,
        )?;
        Ok(self.seq.done(e))
    }

    #[allow(clippy::too_many_arguments)]
    fn binary(
        &mut self,
        a: u32,
        b: u32,
        out: &str,
        rows: u32,
        width: u32,
        op: BinaryOp,
        b_row: u32,
        b_col: u32,
        round: bool,
    ) -> Result<u32, String> {
        let dep = self.seq.last;
        let e = self.seq.p().binary_f32(
            a,
            b,
            &[dep],
            BinaryF32Stage {
                output: TensorRef::Named(out),
                op,
                items: 1,
                rows,
                width,
                b_item_stride: 0,
                b_row_stride: b_row,
                b_col_stride: b_col,
                scale: None,
            },
        )?;
        let out = self.seq.done(e);
        if round {
            self.round(out, rows, width)?;
        }
        Ok(out)
    }

    fn reduce(
        &mut self,
        x: u32,
        out: &str,
        width: u32,
        op: ReduceRowsOp,
        round: bool,
    ) -> Result<u32, String> {
        let dep = self.seq.last;
        let rows = self.rows;
        let length = self.length;
        let e = self.seq.p().reduce_rows_f32(
            x,
            &[dep],
            ReduceRowsF32Stage {
                output: TensorRef::Named(out),
                lengths: Some(length),
                items: 1,
                rows,
                width,
                op,
                seg_rows: 0,
                stride: 0,
                col0: 0,
            },
        )?;
        let out = self.seq.done(e);
        if round {
            self.round(out, 1, width)?;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        x: u32,
        out_name: &str,
        weight: &str,
        rows: u32,
        cin: u32,
        cout: u32,
        kernel: u32,
        dilation: u32,
        relu: bool,
    ) -> Result<u32, String> {
        let dep = self.seq.last;
        let weight_name = format!("w.{weight}.weight");
        let bias_name = format!("w.{weight}.bias");
        let length = (rows != 1).then_some(self.length);
        let pad = (kernel - 1) * dilation / 2;
        let e = self.seq.p().conv1d_f32(
            x,
            false,
            &[dep],
            Conv1dF32Stage {
                output: TensorRef::Named(out_name),
                weight: TensorRef::Named(&weight_name),
                bias: None,
                alpha: None,
                residual: None,
                lengths: length,
                batch: 1,
                in_rows: rows,
                in_channels: cin,
                out_channels: cout,
                kernel,
                stride: 1,
                dilation_or_output_padding: dilation,
                groups: 1,
                pad_before: pad,
                pad_after: pad,
                pad_mode: PadMode::Reflect,
                input_activation: Activation::None,
                output_activation: Activation::None,
                slope: 0.0,
                weight_f16: false,
                split_bf16: false,
                weight_tap_major: true,
                wgmma: false,
                weight_split: false,
                dual_alpha: false,
            },
        )?;
        let out = self.seq.done(e);
        self.round(out, rows, cout)?;
        let bias = self.seq.p().declare(&bias_name, u64::from(cout) * 4);
        self.binary(out, bias, out_name, rows, cout, BinaryOp::Add, 0, 1, true)?;
        if relu {
            self.unary(out, rows, cout, Activation::Relu)?;
        }
        Ok(out)
    }

    fn statistics(
        &mut self,
        x: u32,
        weights: u32,
        width: u32,
        scalar: bool,
        eps: f32,
    ) -> Result<(u32, u32), String> {
        let (br, bc) = if scalar { (0, 0) } else { (width, 1) };
        let weighted = self.binary(
            x,
            weights,
            "act.spk.weighted",
            self.rows,
            width,
            BinaryOp::Mul,
            br,
            bc,
            true,
        )?;
        let mean = self.reduce(weighted, "act.spk.mean", width, ReduceRowsOp::Sum, true)?;
        let centered = self.binary(
            x,
            mean,
            "act.spk.centered",
            self.rows,
            width,
            BinaryOp::Sub,
            0,
            1,
            true,
        )?;
        self.binary(
            centered,
            centered,
            "act.spk.centered",
            self.rows,
            width,
            BinaryOp::Mul,
            width,
            1,
            true,
        )?;
        self.binary(
            centered,
            weights,
            "act.spk.centered",
            self.rows,
            width,
            BinaryOp::Mul,
            br,
            bc,
            true,
        )?;
        let std = self.reduce(centered, "act.spk.std", width, ReduceRowsOp::Sum, true)?;
        let dep = self.seq.last;
        let e = self.seq.p().unary_f32(
            std,
            &[dep],
            UnaryF32Stage {
                output: TensorRef::Handle(std),
                param: None,
                rows: 1,
                width,
                stride: 0,
                col0: 0,
                kind: Activation::Clamp,
                p0: crate::config::bf16_round(eps),
                p1: f32::MAX,
            },
        )?;
        self.seq.done(e);
        self.unary(std, 1, width, Activation::Sqrt)?;
        self.round(std, 1, width)?;
        Ok((mean, std))
    }
}

fn program(
    prefix: PacketPrefix,
    cfg: &Cfg,
    rows: u32,
    length: u32,
    reciprocal: u32,
) -> Result<PacketPrefix, String> {
    let input = prefix.input;
    let mut p = prefix.program();
    let e = p.unary_f32(
        input,
        &[],
        UnaryF32Stage {
            output: TensorRef::Named("act.spk.input"),
            param: None,
            rows,
            width: 128,
            stride: 0,
            col0: 0,
            kind: Activation::RoundBf16,
            p0: 0.0,
            p1: 0.0,
        },
    )?;
    let input = e.output;
    let mut enc = Encoder {
        seq: Seq {
            p: Some(p),
            last: e.done,
        },
        rows,
        length,
    };
    let c = cfg.channels[0];
    let mut h = enc.conv(
        input,
        "act.spk.block0",
        "blocks.0.conv",
        rows,
        128,
        c,
        cfg.kernel_sizes[0],
        cfg.dilations[0],
        true,
    )?;
    let mut blocks = Vec::new();
    for b in 1..cfg.channels.len() - 1 {
        let a = enc.conv(
            h,
            "act.spk.a",
            &format!("blocks.{b}.tdnn1.conv"),
            rows,
            c,
            c,
            1,
            1,
            true,
        )?;
        let part = c / cfg.res2net_scale;
        let cat = enc.copy(a, "act.spk.res2", rows, part, c, 0, c, 0)?;
        let mut previous = None;
        for k in 1..cfg.res2net_scale {
            let x = enc.copy(a, "act.spk.part", rows, part, c, k * part, part, 0)?;
            if let Some(previous) = previous {
                enc.binary(
                    x,
                    previous,
                    "act.spk.part",
                    rows,
                    part,
                    BinaryOp::Add,
                    part,
                    1,
                    true,
                )?;
            }
            let y = enc.conv(
                x,
                "act.spk.previous",
                &format!("blocks.{b}.res2net.{}.conv", k - 1),
                rows,
                part,
                part,
                cfg.kernel_sizes[b],
                cfg.dilations[b],
                true,
            )?;
            enc.copy(y, "act.spk.res2", rows, part, part, 0, c, k * part)?;
            previous = Some(y);
        }
        let a = enc.conv(
            cat,
            "act.spk.a",
            &format!("blocks.{b}.tdnn2.conv"),
            rows,
            c,
            c,
            1,
            1,
            true,
        )?;
        let mean = enc.reduce(a, "act.spk.se.mean", c, ReduceRowsOp::Mean, true)?;
        let se = enc.conv(
            mean,
            "act.spk.se.hidden",
            &format!("blocks.{b}.se.conv1"),
            1,
            c,
            cfg.se_channels,
            1,
            1,
            true,
        )?;
        let gate = enc.conv(
            se,
            "act.spk.se.gate",
            &format!("blocks.{b}.se.conv2"),
            1,
            cfg.se_channels,
            c,
            1,
            1,
            false,
        )?;
        enc.unary(gate, 1, c, Activation::Sigmoid)?;
        enc.round(gate, 1, c)?;
        enc.binary(a, gate, "act.spk.a", rows, c, BinaryOp::Mul, 0, 1, true)?;
        h = enc.binary(
            a,
            h,
            &format!("act.spk.block{b}"),
            rows,
            c,
            BinaryOp::Add,
            c,
            1,
            true,
        )?;
        blocks.push(h);
    }
    let m = cfg.width();
    let mut cat = 0;
    for (i, &block) in blocks.iter().enumerate() {
        cat = enc.copy(block, "act.spk.cat", rows, c, c, 0, m, i as u32 * c)?;
    }
    let x = enc.conv(
        cat,
        "act.spk.mfa",
        "mfa.conv",
        rows,
        m,
        m,
        *cfg.kernel_sizes.last().unwrap(),
        *cfg.dilations.last().unwrap(),
        true,
    )?;
    let dep = enc.seq.last;
    let e = enc.seq.p().gather_rows_f32(
        &[dep],
        GatherRowsF32Stage {
            output: TensorRef::Named("act.spk.uniform"),
            table: TensorRef::Handle(reciprocal),
            table_rows: CAPACITIES.last().unwrap() + 1,
            index: Some(length),
            rows: 1,
            width: 1,
            vocab: CAPACITIES.last().unwrap() + 1,
            rows_per_item: 1,
            repeat: 1,
            index_item_stride: 0,
            table_item_stride: 0,
            table_f16: false,
            accumulate: false,
            out_stride: 0,
            out_col0: 0,
        },
    )?;
    let uniform = enc.seq.done(e);
    let (mean, std) = enc.statistics(x, uniform, m, true, cfg.statistics_eps)?;
    let attention = enc.copy(x, "act.spk.attention_input", rows, m, m, 0, 3 * m, 0)?;
    enc.copy(mean, "act.spk.attention_input", rows, m, 0, 0, 3 * m, m)?;
    enc.copy(std, "act.spk.attention_input", rows, m, 0, 0, 3 * m, 2 * m)?;
    let attention = enc.conv(
        attention,
        "act.spk.attention_hidden",
        "asp.tdnn.conv",
        rows,
        3 * m,
        cfg.attention_channels,
        1,
        1,
        true,
    )?;
    enc.unary(attention, rows, cfg.attention_channels, Activation::Tanh)?;
    enc.round(attention, rows, cfg.attention_channels)?;
    let attention = enc.conv(
        attention,
        "act.spk.attention",
        "asp.conv",
        rows,
        cfg.attention_channels,
        m,
        1,
        1,
        false,
    )?;
    let max = enc.reduce(
        attention,
        "act.spk.attention_max",
        m,
        ReduceRowsOp::Max,
        false,
    )?;
    enc.binary(
        attention,
        max,
        "act.spk.attention",
        rows,
        m,
        BinaryOp::Sub,
        0,
        1,
        false,
    )?;
    enc.unary(attention, rows, m, Activation::Exp)?;
    let sum = enc.reduce(
        attention,
        "act.spk.attention_sum",
        m,
        ReduceRowsOp::Sum,
        false,
    )?;
    enc.binary(
        attention,
        sum,
        "act.spk.attention",
        rows,
        m,
        BinaryOp::Div,
        0,
        1,
        true,
    )?;
    let (mean, std) = enc.statistics(x, attention, m, false, cfg.statistics_eps)?;
    let pooled = enc.copy(mean, "act.spk.pooled", 1, m, m, 0, 2 * m, 0)?;
    enc.copy(std, "act.spk.pooled", 1, m, m, 0, 2 * m, m)?;
    enc.conv(
        pooled,
        "act.spk.embedding",
        "fc",
        1,
        2 * m,
        cfg.embedding_dim,
        1,
        1,
        false,
    )?;
    Ok(enc.seq.p.take().unwrap().finish(rows))
}

fn section(
    prefix: &PacketPrefix,
    cfg: &Cfg,
    roles: BTreeMap<String, u32>,
    length: u32,
    max: u32,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{
        PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION,
    };
    let tensor = |h: u32, dtype, shape| PipelineTensor {
        name: prefix.model.tensors[h as usize].name.clone(),
        dtype,
        shape,
    };
    let named = |name: &str, shape| PipelineTensor {
        name: name.into(),
        dtype: PipelineDType::F32,
        shape,
    };
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            name: "speaker.encode".into(),
            driver: DRIVER.into(),
            programs: roles,
            tensors: BTreeMap::from([
                (
                    "mel".into(),
                    tensor(prefix.input, PipelineDType::F32, vec![u64::from(max), 128]),
                ),
                ("length".into(), tensor(length, PipelineDType::U32, vec![1])),
                (
                    "embedding".into(),
                    tensor(
                        prefix.output,
                        PipelineDType::F32,
                        vec![u64::from(cfg.embedding_dim)],
                    ),
                ),
                (
                    "audio.frontend.filterbank".into(),
                    named("w.frontend.mel", vec![128, 513]),
                ),
                (
                    "audio.frontend.window_values".into(),
                    named("w.frontend.window", vec![1024]),
                ),
            ]),
            parameters: BTreeMap::from([
                ("input.min_rows".into(), u64::from(cfg.min_rows()?)),
                ("input.max_rows".into(), u64::from(max)),
                ("input.width".into(), 128),
                ("output.width".into(), u64::from(cfg.embedding_dim)),
                ("audio.frontend.kind".into(), 2),
                ("audio.sample_rate".into(), 24_000),
                ("audio.frontend.fft".into(), 1024),
                ("audio.frontend.window".into(), 1024),
                ("audio.frontend.hop".into(), 256),
                ("audio.frontend.bins".into(), 128),
                ("audio.frontend.preemphasis_f32".into(), 0),
                ("audio.frontend.center_window".into(), 0),
                ("audio.frontend.periodic_hann".into(), 1),
                ("audio.frontend.normalize_per_feature".into(), 0),
                ("audio.frontend.mask_invalid_frames".into(), 0),
                (
                    "audio.frontend.log_guard_f32".into(),
                    u64::from(1e-5f32.to_bits()),
                ),
                ("audio.frontend.pad_reflect".into(), 1),
                ("audio.frontend.drop_last_frame".into(), 0),
                ("audio.frontend.log10".into(), 0),
                ("audio.frontend.log_floor".into(), 1),
                ("audio.frontend.dynamic_range_f32".into(), 0),
                ("audio.frontend.scale_f32".into(), u64::from(1f32.to_bits())),
                ("audio.frontend.shift_f32".into(), 0),
                ("audio.frontend.pad_samples".into(), 384),
                (
                    "audio.frontend.magnitude_guard_f32".into(),
                    u64::from(1e-9f32.to_bits()),
                ),
                (
                    "audio.min_samples".into(),
                    (u64::from(cfg.min_rows()?) * 256).max(385),
                ),
                ("audio.max_samples".into(), (u64::from(max) + 1) * 256 - 1),
            ]),
            strings: BTreeMap::from([("audio.frontend".into(), cfg.frontend.to_string())]),
        }],
    };
    metadata.validate(prefix.model.progs.len(), |name| {
        prefix
            .model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.bytes)
    })?;
    Ok(packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "ecapa_tdnn", "version": 1, "channels": [8,8,8,16],
            "kernel_sizes": [5,3,3,1], "dilations": [1,2,3,1], "res2net_scale": 2,
            "se_channels": 3, "attention_channels": 4, "embedding_dim": 6,
            "batch_norm": false, "l2_normalize": false, "statistics_eps": 1e-12,
            "compute_dtype": "bfloat16", "conv_bias_round": "before_add", "weight_layout": "out_tap_in",
            "frontend": {"compute_dtype": "float32", "sample_rate": 24000, "n_mels": 128,
                "n_fft": 1024, "hop": 256, "win_length": 1024, "pad": 384,
                "pad_mode": "reflect", "center": false, "magnitude_eps": 1e-9,
                "mel_norm": "slaney", "mel_htk": false, "fmin": 0, "fmax": 12000,
                "log_floor": 1e-5, "sentence_normalize": false}
        })
    }

    #[test]
    fn configuration_rejects_changed_precision_normalization_and_geometry() {
        let parse = |v: &serde_json::Value| Cfg::parse(&serde_json::to_vec(v).unwrap());
        assert_eq!(parse(&config()).unwrap().min_rows().unwrap(), 4);
        for (key, value) in [
            ("compute_dtype", serde_json::json!("float32")),
            ("conv_bias_round", serde_json::json!("fused")),
            ("batch_norm", serde_json::json!(true)),
            ("l2_normalize", serde_json::json!(true)),
            ("statistics_eps", serde_json::json!(1e-5)),
            ("channels", serde_json::json!([8, 8, 8, 24])),
            ("channels", serde_json::json!([])),
            ("res2net_scale", serde_json::json!(3)),
            ("dilations", serde_json::json!([1, 2])),
            ("kernel_sizes", serde_json::json!([4, 3, 3, 1])),
            ("unknown", serde_json::json!(1)),
        ] {
            let mut v = config();
            v[key] = value;
            assert!(parse(&v).is_err(), "{key}");
        }
        let mut v = config();
        v["frontend"]["sentence_normalize"] = true.into();
        assert!(parse(&v).is_err());
    }

    #[test]
    fn emitted_encoder_keeps_lengths_rounding_and_effects() {
        use packet::dev::DevOp;
        let cfg = Cfg::parse(&serde_json::to_vec(&config()).unwrap()).unwrap();
        let mut b = packet::devbuild::Builder::new(1);
        b.set_tensor_dedup(true);
        let input = b.tensor("in.spk.mel", 128 * 128 * 4);
        let length = b.tensor("in.spk.length", 4);
        let reciprocal = b.tensor("c.spk.reciprocal", 2049 * 4);
        b.tensor("w.frontend.mel", 128 * 513 * 4);
        b.tensor("w.frontend.window", 1024 * 4);
        let prefix = PacketPrefix {
            model: packet::devbuild::Model {
                n_cu: 1,
                target: 90,
                tensors: b.tensors(),
                progs: Vec::new(),
                kv_row_insts: Vec::new(),
                prog_t: Vec::new(),
                gen: Vec::new(),
            },
            programs: Vec::new(),
            input,
            output: input,
            input_shape: vec![128, 128],
        };
        let prefix = program(prefix, &cfg, 128, length, reciprocal).unwrap();
        plow_asset::program::with_model(&prefix.model, |packet| {
            plow_asset::logical_effects::obligation(packet, 0).unwrap();
        });
        let insts = &prefix.model.progs[0].insts;
        let convolutions: Vec<_> = insts
            .iter()
            .filter(|i| i.op == DevOp::Conv1dF32 as u16)
            .collect();
        assert_eq!(convolutions.len(), 15);
        for conv in convolutions {
            assert_eq!(
                conv.t[6],
                if conv.i[1] == 1 {
                    packet::dev::TENSOR_NONE
                } else {
                    length
                }
            );
            assert_eq!(conv.j[1] & 3, 1);
            assert_eq!(conv.t[3], packet::dev::TENSOR_NONE);
            let at = insts.iter().position(|i| std::ptr::eq(i, conv)).unwrap();
            let next = &insts[at + 1];
            assert_eq!(next.op, DevOp::UnaryF32 as u16);
            assert_eq!(next.i[2], packet::dev::ACT_ROUND_BF16);
            assert_eq!(next.t[0], conv.t[0]);
            let bias = &insts[at + 2];
            assert_eq!(bias.op, DevOp::BinaryF32 as u16);
            assert_eq!(bias.i[3], 0);
            assert_eq!(bias.t[1], conv.t[0]);
            assert_eq!(bias.t[0], conv.t[0]);
        }
        let reductions: Vec<_> = insts
            .iter()
            .filter(|i| i.op == DevOp::ReduceRowsF32 as u16)
            .collect();
        assert!(!reductions.is_empty());
        assert!(reductions.iter().all(|i| i.t[2] == length && i.i[3] <= 2));
        assert_eq!(
            prefix.model.tensors[prefix.output as usize].name,
            "act.spk.embedding"
        );
        let metadata = section(
            &prefix,
            &cfg,
            BTreeMap::from([("encode.r128".into(), 0)]),
            length,
            128,
        )
        .unwrap();
        let metadata: plow_asset::packet_pipeline::PacketPipelines =
            serde_json::from_slice(&metadata.data).unwrap();
        let pipeline = &metadata.pipelines[0];
        assert_eq!(pipeline.parameters["audio.frontend.kind"], 2);
        assert_eq!(pipeline.parameters["audio.frontend.pad_samples"], 384);
        assert_eq!(pipeline.parameters["audio.min_samples"], 1024);
        assert_eq!(pipeline.parameters["audio.max_samples"], 129 * 256 - 1);
        assert_eq!(
            pipeline.tensors["audio.frontend.filterbank"].shape,
            [128, 513]
        );
        assert_eq!(
            pipeline.tensors["audio.frontend.window_values"].shape,
            [1024]
        );
    }
}
