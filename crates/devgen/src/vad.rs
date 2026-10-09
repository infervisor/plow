//! Voice activity detectors lowered to `vad.v1` packets (`vad.pkt`): one 16 kHz frame (with its
//! left context) and the recurrent state of each stream in, the frame's speech probability and
//! the next state out.
//!
//! Silero VAD v5 (scripts/asr/silero_vad_export.py): a reflect-padded STFT as a strided
//! convolution, magnitude, four conv + ReLU blocks, an LSTM cell carrying `(h, c)` across frames
//! and a ReLU + 1x1 conv + sigmoid head. Every batch capacity is one program; rows are independent
//! streams, so a stream's probabilities do not depend on what it is batched with.

use std::collections::BTreeMap;

use crate::pipeline::{
    Activation, BinaryF32Stage, BinaryOp, Conv1dF32Stage, CopyColsF32Stage, DenseActivation,
    DenseF32ProgramStage, DenseWeight, PacketPrefix, PadMode, TensorRef, UnaryF32Stage,
};

pub const PACKET: &str = "vad.pkt";
pub const PIPELINE: &str = "vad.silero";
pub const DRIVER: &str = "vad.v1";

/// Streams per launch. One frame per stream per launch, so capacities follow the session count.
pub const CAPACITIES: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];

struct EncoderBlock {
    cin: u32,
    cout: u32,
    kernel: u32,
    stride: u32,
    padding: u32,
}

struct SileroConfig {
    sampling_rate: u32,
    frame_samples: u32,
    context_samples: u32,
    filter_length: u32,
    hop_length: u32,
    pad_after: u32,
    encoder: Vec<EncoderBlock>,
    lstm_width: u32,
}

impl SileroConfig {
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let u = |v: &serde_json::Value, k: &str| -> Result<u32, String> {
            v[k].as_u64().and_then(|x| u32::try_from(x).ok()).ok_or(format!("silero_vad config: {k}"))
        };
        if v["model_type"].as_str() != Some("silero_vad") {
            return Err("not a silero_vad export".into());
        }
        let encoder = v["encoder"]
            .as_array()
            .ok_or("silero_vad config: encoder")?
            .iter()
            .map(|b| {
                Ok(EncoderBlock {
                    cin: u(b, "cin")?,
                    cout: u(b, "cout")?,
                    kernel: u(b, "kernel")?,
                    stride: u(b, "stride")?,
                    padding: u(b, "padding")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            sampling_rate: u(v, "sampling_rate")?,
            frame_samples: u(v, "frame_samples")?,
            context_samples: u(v, "context_samples")?,
            filter_length: u(v, "filter_length")?,
            hop_length: u(v, "hop_length")?,
            pad_after: u(v, "pad_after")?,
            encoder,
            lstm_width: u(v, "lstm_width")?,
        })
    }

    fn window(&self) -> u32 {
        self.context_samples + self.frame_samples
    }

    fn bins(&self) -> u32 {
        self.filter_length / 2 + 1
    }

    fn stft_rows(&self) -> u32 {
        (self.window() + self.pad_after - self.filter_length) / self.hop_length + 1
    }
}

/// `vad.pkt` (model + pipeline section) for the Silero VAD export in `dir`.
pub fn lower_silero(dir: &std::path::Path, n_cu: u32, target: u32) -> Result<(packet::devbuild::Model, packet::devbuild::SectionData), String> {
    let cfg = SileroConfig::parse(
        &serde_json::from_slice(&std::fs::read(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.display()))?)
            .map_err(|e| format!("silero_vad config: {e}"))?,
    )?;
    let bins = cfg.bins();
    if cfg.encoder.first().map(|b| b.cin) != Some(bins)
        || cfg.encoder.last().map(|b| b.cout) != Some(cfg.lstm_width)
        || cfg.pad_after >= cfg.window()
    {
        return Err("unsupported silero_vad geometry".into());
    }
    let mut rows = cfg.stft_rows();
    let mut widest = rows * bins.max(cfg.encoder[0].cout);
    for b in &cfg.encoder {
        rows = (rows + 2 * b.padding - b.kernel) / b.stride + 1;
        widest = widest.max(rows * b.cout);
    }
    if rows != 1 {
        return Err(format!("silero_vad encoder leaves {rows} frames, expected 1"));
    }

    let bmax = *CAPACITIES.last().expect("capacities");
    let width = cfg.lstm_width;
    let model = packet::devbuild::Model {
        n_cu,
        target,
        tensors: Vec::new(),
        progs: Vec::new(),
        kv_row_insts: Vec::new(),
        prog_t: Vec::new(),
        gen: Vec::new(),
    };
    // Largest batch first: tensors dedup by name and keep their first size.
    let mut builder = packet::devbuild::Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let f32s = |n: u32| u64::from(bmax) * u64::from(n) * 4;
    let audio = builder.tensor("in.vad.audio", f32s(cfg.window()));
    let state = [builder.tensor("in.vad.h", f32s(width)), builder.tensor("in.vad.c", f32s(width))];
    let state_out = [builder.tensor("act.vad.h_out", f32s(width)), builder.tensor("act.vad.c_out", f32s(width))];
    let prob = builder.tensor("act.vad.prob", f32s(1));
    builder.tensor("act.vad.stft", f32s(cfg.stft_rows() * 2 * bins));
    builder.tensor("act.vad.mag", f32s(cfg.stft_rows() * bins));
    builder.tensor("act.vad.a", f32s(widest));
    builder.tensor("act.vad.b", f32s(widest));
    builder.tensor("act.vad.gi", f32s(4 * width));
    builder.tensor("act.vad.gh", f32s(4 * width));
    let mut model = model;
    model.tensors = builder.tensors();
    let mut prefix = PacketPrefix { model, programs: Vec::new(), input: audio, output: prob, input_shape: vec![u64::from(bmax), u64::from(cfg.window())] };

    let mut roles = BTreeMap::new();
    for &batch in CAPACITIES {
        let program = prefix.model.progs.len();
        prefix = silero_program(prefix, &cfg, batch, audio, state, state_out)?;
        roles.insert(format!("step.b{batch}"), program as u32);
    }
    let reader = crate::checkpoint::TensorReader::open(dir)?;
    for tensor in &mut prefix.model.tensors {
        if let Some(name) = tensor.name.strip_prefix("w.") {
            let (dtype, bytes) = reader.read(name)?;
            if dtype != "F32" || bytes.len() as u64 != tensor.bytes {
                return Err(format!("{name}: {dtype} {} bytes, expected F32 {}", bytes.len(), tensor.bytes));
            }
            tensor.init = Some(bytes);
        }
    }
    let section = pipeline_section(&prefix, &cfg, roles, bmax, state, state_out)?;
    Ok((prefix.model, section))
}

#[allow(clippy::too_many_arguments)]
fn conv<'a>(
    output: &'a str,
    weight: &'a str,
    bias: Option<&'a str>,
    batch: u32,
    in_rows: u32,
    cin: u32,
    cout: u32,
    kernel: u32,
) -> Conv1dF32Stage<'a> {
    Conv1dF32Stage {
        output: TensorRef::Named(output),
        weight: TensorRef::Named(weight),
        bias: bias.map(TensorRef::Named),
        alpha: None,
        residual: None,
        lengths: None,
        batch,
        in_rows,
        in_channels: cin,
        out_channels: cout,
        kernel,
        stride: 1,
        dilation_or_output_padding: 1,
        groups: 1,
        pad_before: 0,
        pad_after: 0,
        pad_mode: PadMode::Zero,
        input_activation: Activation::None,
        output_activation: Activation::None,
        slope: 0.0,
        weight_f16: false,
        split_bf16: false,
        weight_tap_major: false,
        wgmma: false,
        weight_split: false,
    }
}

fn silero_program(
    prefix: PacketPrefix,
    cfg: &SileroConfig,
    batch: u32,
    audio: u32,
    state: [u32; 2],
    state_out: [u32; 2],
) -> Result<PacketPrefix, String> {
    let mut p = prefix.program();
    let bins = cfg.bins();
    let frames = cfg.stft_rows();
    let width = cfg.lstm_width;
    // STFT: [context | frame] reflect-padded on the right, one strided conv into [re | im] bins.
    let mut c = conv("act.vad.stft", "w.stft.basis", None, batch, cfg.window(), 1, 2 * bins, cfg.filter_length);
    c.stride = cfg.hop_length;
    c.pad_after = cfg.pad_after;
    c.pad_mode = PadMode::Reflect;
    let stft = p.conv1d_f32(audio, false, &[], c)?;
    let squared = p.binary_f32(stft.output, stft.output, &[stft.done], BinaryF32Stage {
        output: TensorRef::Handle(stft.output),
        op: BinaryOp::Mul,
        items: 1,
        rows: batch * frames,
        width: 2 * bins,
        b_item_stride: 0,
        b_row_stride: 2 * bins,
        b_col_stride: 1,
        scale: None,
    })?;
    let imag = p.copy_cols_f32(stft.output, &[squared.done], CopyColsF32Stage {
        output: TensorRef::Named("act.vad.mag"),
        items: 1,
        rows: batch * frames,
        cols: bins,
        in_item_stride: 0,
        in_stride: 2 * bins,
        in_offset: bins,
        out_item_stride: 0,
        out_stride: bins,
        out_offset: 0,
    })?;
    let power = p.binary_f32(imag.output, stft.output, &[imag.done], BinaryF32Stage {
        output: TensorRef::Handle(imag.output),
        op: BinaryOp::Add,
        items: 1,
        rows: batch * frames,
        width: bins,
        b_item_stride: 0,
        b_row_stride: 2 * bins,
        b_col_stride: 1,
        scale: None,
    })?;
    let mut x = p.unary_f32(power.output, &[power.done], UnaryF32Stage {
        output: TensorRef::Handle(power.output),
        param: None,
        rows: batch * frames,
        width: bins,
        stride: 0,
        col0: 0,
        kind: Activation::Sqrt,
        p0: 0.0,
        p1: 0.0,
    })?;
    let mut rows = frames;
    let weights: Vec<(String, String)> = (0..cfg.encoder.len()).map(|i| (format!("w.enc{i}.w"), format!("w.enc{i}.b"))).collect();
    for (i, b) in cfg.encoder.iter().enumerate() {
        let target = if i % 2 == 0 { "act.vad.a" } else { "act.vad.b" };
        let mut c = conv(target, &weights[i].0, Some(&weights[i].1), batch, rows, b.cin, b.cout, b.kernel);
        c.stride = b.stride;
        c.pad_before = b.padding;
        c.pad_after = b.padding;
        c.output_activation = Activation::Relu;
        rows = c.out_rows(false).ok_or("silero_vad encoder output is empty")?;
        x = p.conv1d_f32(x.output, false, &[x.done], c)?;
    }
    // LSTM cell: gates = x Wih^T + bih + (h Whh^T + bhh), the reference's summation order.
    let dense = |output, weight, bias| DenseF32ProgramStage {
        output: TensorRef::Named(output),
        weight: TensorRef::Named(weight),
        bias: Some(TensorRef::Named(bias)),
        rows: batch,
        input_width: width,
        output_width: 4 * width,
        activation: DenseActivation::None,
        round_bf16: false,
        weight_type: DenseWeight::F32,
        operands_bf16_exact: false,
        tf32x3: false,
        layer_norm: None,
    };
    let gi = p.dense_f32(x.output, &[x.done], dense("act.vad.gi", "w.lstm.wih", "w.lstm.bih"))?;
    let gh = p.dense_f32(state[0], &[], dense("act.vad.gh", "w.lstm.whh", "w.lstm.bhh"))?;
    let gates = p.binary_f32(gi.output, gh.output, &[gi.done, gh.done], BinaryF32Stage {
        output: TensorRef::Handle(gi.output),
        op: BinaryOp::Add,
        items: 1,
        rows: batch,
        width: 4 * width,
        b_item_stride: 0,
        b_row_stride: 4 * width,
        b_col_stride: 1,
        scale: None,
    })?;
    let h = p.lstm_cell_f32(gates.output, state[1], &[gates.done], TensorRef::Handle(state_out[0]), TensorRef::Handle(state_out[1]), batch, width)?;
    let mut c = conv("act.vad.prob", "w.head.w", Some("w.head.b"), batch, 1, width, 1, 1);
    c.input_activation = Activation::Relu;
    c.output_activation = Activation::Sigmoid;
    p.conv1d_f32(h.output, false, &[h.done], c)?;
    Ok(p.finish(batch))
}

fn pipeline_section(
    prefix: &PacketPrefix,
    cfg: &SileroConfig,
    roles: BTreeMap<String, u32>,
    bmax: u32,
    state: [u32; 2],
    state_out: [u32; 2],
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION};
    let t = |h: u32| prefix.model.tensors[h as usize].name.clone();
    let f32s = |h: u32, per_row: u32| PipelineTensor { name: t(h), dtype: PipelineDType::F32, shape: vec![u64::from(bmax), u64::from(per_row)] };
    let mut tensors = BTreeMap::from([
        ("audio".into(), f32s(prefix.input, cfg.window())),
        ("prob".into(), f32s(prefix.output, 1)),
    ]);
    for k in 0..2 {
        tensors.insert(format!("state.{k}"), f32s(state[k], cfg.lstm_width));
        tensors.insert(format!("state_out.{k}"), f32s(state_out[k], cfg.lstm_width));
    }
    let parameters = BTreeMap::from([
        ("audio.sample_rate".into(), u64::from(cfg.sampling_rate)),
        ("vad.frame_samples".into(), u64::from(cfg.frame_samples)),
        ("vad.context_samples".into(), u64::from(cfg.context_samples)),
        ("state.count".into(), 2),
    ]);
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            name: PIPELINE.into(),
            driver: DRIVER.into(),
            programs: roles,
            tensors,
            parameters,
            strings: Default::default(),
        }],
    };
    metadata.validate(prefix.model.progs.len(), |name| prefix.model.tensors.iter().find(|x| x.name == name).map(|x| x.bytes))?;
    Ok(packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Silero-shaped export with zero weights: config.json + a hand-written safetensors file.
    fn export(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        let block = |cin, cout, stride| serde_json::json!({"cin": cin, "cout": cout, "kernel": 3, "stride": stride, "padding": 1});
        let config = serde_json::json!({"model_type": "silero_vad", "sampling_rate": 16000, "frame_samples": 512,
            "context_samples": 64, "filter_length": 256, "hop_length": 128, "pad_after": 64, "lstm_width": 128,
            "encoder": [block(129, 128, 1), block(128, 64, 2), block(64, 64, 2), block(64, 128, 1)]});
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let mut shapes: Vec<(String, Vec<usize>)> = vec![("stft.basis".into(), vec![258, 1, 256])];
        for (i, (cin, cout)) in [(129, 128), (128, 64), (64, 64), (64, 128)].into_iter().enumerate() {
            shapes.push((format!("enc{i}.w"), vec![cout, cin, 3]));
            shapes.push((format!("enc{i}.b"), vec![cout]));
        }
        for (name, shape) in [("lstm.wih", vec![512, 128]), ("lstm.whh", vec![512, 128]), ("lstm.bih", vec![512]),
            ("lstm.bhh", vec![512]), ("head.w", vec![1, 128, 1]), ("head.b", vec![1])] {
            shapes.push((name.into(), shape));
        }
        let (mut header, mut at) = (serde_json::Map::new(), 0usize);
        for (name, shape) in &shapes {
            let bytes = shape.iter().product::<usize>() * 4;
            header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [at, at + bytes]}));
            at += bytes;
        }
        let header = serde_json::Value::Object(header).to_string().into_bytes();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.resize(file.len() + at, 0);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
    }

    #[test]
    fn silero_lowers_to_one_program_per_capacity_with_the_vad_contract() {
        let dir = std::env::temp_dir().join(format!("plow-vad-export-{}", std::process::id()));
        export(&dir);
        let (model, section) = lower_silero(&dir, 8, 0).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(model.progs.len(), CAPACITIES.len());
        let meta: serde_json::Value = serde_json::from_slice(&section.data).unwrap();
        let pipeline = &meta["pipelines"][0];
        assert_eq!(pipeline["driver"], DRIVER);
        assert_eq!(pipeline["programs"]["step.b256"], CAPACITIES.len() - 1);
        assert_eq!(pipeline["parameters"]["state.count"], 2);
        assert_eq!(pipeline["tensors"]["audio"]["shape"], serde_json::json!([256, 576]));
        // Every weight comes from the export; nothing is left for a checkpoint.
        assert!(model.tensors.iter().filter(|t| t.name.starts_with("w.")).all(|t| t.init.is_some()));
        assert!(model.tensors.iter().all(|t| t.name.starts_with("w.") || packet::names::is_runtime_tensor(&t.name)));
        // STFT, magnitude (mul, copy, add, sqrt), 4 convs, 2 dense, add, LSTM, head.
        let lstm_rows = |p: usize| model.progs[p].insts.iter().find(|i| i.op == packet::dev::DevOp::LstmCellF32 as u16).map(|i| i.i[1]);
        assert_eq!(model.progs[0].insts.len(), 14);
        assert_eq!((lstm_rows(0), lstm_rows(CAPACITIES.len() - 1)), (Some(1), Some(256)));
    }
}
