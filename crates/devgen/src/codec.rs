//! Audio codec decoders lowered to `forward.v1` packets (`codec.pkt`): codes in, PCM out.
//!
//! SNAC (the 24 kHz multi-scale neural codec Veena/Orpheus emit tokens for): per-level codebook
//! tables (already projected by the export), a depthwise + pointwise stem, four decoder blocks
//! (snake, transposed convolution, noise injection, three dilated residual units) and a snake +
//! convolution + tanh head. Every (batch, frames) capacity is one program; per-item valid lengths
//! at each time resolution and per-item noise seeds make every item decode exactly as it would
//! alone.
//!
//! Fusions (each convolution's input activation, output activation and residual) come from the
//! rewrite's fused sites over the SNAC graph (`nn_graph` `snac`, `rewrite` conv1d rules), subject
//! to [`ConvFusions`]' cost model; without sites they are the hand choices below.

use std::collections::BTreeMap;

use crate::pipeline::{
    Activation, BinaryF32Stage, BinaryOp, Conv1dF32Stage, CopyColsF32Stage, Emitted, GatherRowsF32Stage,
    PacketPrefix, PadMode, RandCoord, RandF32Stage, StageProgram, TensorRef, UnaryF32Stage,
};
use crate::rewrite_lower::ConvFusions;
use crate::RewriteSites;

pub const PIPELINE: &str = "codec.decode";

struct SnacBlock {
    stride: u32,
    kernel: u32,
    padding: u32,
    output_padding: u32,
    cin: u32,
    cout: u32,
    dilations: Vec<u32>,
}

struct SnacConfig {
    model_type: String,
    sampling_rate: u32,
    codebook_size: u32,
    vq_strides: Vec<u32>,
    latent_dim: u32,
    blocks: Vec<SnacBlock>,
    frame_codes: u32,
    frame_samples: u32,
    frame_layout: Vec<(u32, u32)>,
}

impl SnacConfig {
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let u = |v: &serde_json::Value, k: &str| -> Result<u32, String> {
            v[k].as_u64().and_then(|x| u32::try_from(x).ok()).ok_or(format!("snac config: {k}"))
        };
        let list = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect()).unwrap_or_default()
        };
        let blocks = v["blocks"]
            .as_array()
            .ok_or("snac config: blocks")?
            .iter()
            .map(|b| {
                Ok(SnacBlock {
                    stride: u(b, "stride")?,
                    kernel: u(b, "kernel")?,
                    padding: u(b, "padding")?,
                    output_padding: u(b, "output_padding")?,
                    cin: u(b, "cin")?,
                    cout: u(b, "cout")?,
                    dilations: list(&b["dilations"]),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let frame_layout = v["frame_layout"]
            .as_array()
            .ok_or("snac config: frame_layout")?
            .iter()
            .map(|p| {
                let p = list(p);
                (p.first().copied().unwrap_or(u32::MAX), p.get(1).copied().unwrap_or(0))
            })
            .collect();
        Ok(Self {
            model_type: v["model_type"].as_str().unwrap_or_default().to_string(),
            sampling_rate: u(v, "sampling_rate")?,
            codebook_size: u(v, "codebook_size")?,
            vq_strides: list(&v["vq_strides"]),
            latent_dim: u(v, "latent_dim")?,
            blocks,
            frame_codes: u(v, "frame_codes")?,
            frame_samples: u(v, "frame_samples")?,
            frame_layout,
        })
    }
}

/// (batch, frames) capacities: streaming windows are short and batched across streams, a whole
/// utterance is long and alone. B * F stays <= 512 (activations are [B][2048 F][64] f32).
pub const SNAC_CAPACITIES: &[(u32, u32)] = &[
    (1, 8), (2, 8), (4, 8), (8, 8), (16, 8), (32, 8), (64, 8),
    (1, 16), (4, 16), (16, 16), (32, 16),
    (1, 32), (4, 32), (16, 32),
    (1, 64), (4, 64), (8, 64),
    (1, 128), (4, 128),
];

/// `codec.pkt` for the exported SNAC decoder in `dir` (scripts/tts/snac_export.py), with the
/// rewrite's fused sites of that export's graph when plowc extracted them.
pub fn lower_snac(dir: &std::path::Path, n_cu: u32, target: u32, sites: Option<&RewriteSites>) -> Result<Vec<u8>, String> {
    let cfg = SnacConfig::parse(
        &serde_json::from_slice(&std::fs::read(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.display()))?)
            .map_err(|e| format!("snac config: {e}"))?,
    )?;
    if cfg.model_type != "snac" || cfg.frame_layout.len() != cfg.frame_codes as usize || cfg.vq_strides.len() != 3 {
        return Err("unsupported SNAC export".into());
    }
    let levels = cfg.vq_strides.len() as u32;
    // Latent rows per frame: the finest level has one code per latent row.
    let latent_per_frame = cfg.vq_strides[0];
    let mut per_level = vec![0u32; levels as usize];
    for &(level, _) in &cfg.frame_layout {
        per_level[level as usize] += 1;
    }
    let upsample: u32 = cfg.blocks.iter().map(|b| b.stride).product();
    if latent_per_frame * upsample != cfg.frame_samples {
        return Err("SNAC strides do not produce the frame size".into());
    }

    let bmax = SNAC_CAPACITIES.iter().map(|&(b, _)| b).max().unwrap_or(1);
    let max_rows = SNAC_CAPACITIES.iter().map(|&(b, f)| b * f).max().unwrap_or(1);
    let mut model = packet::devbuild::Model {
        n_cu,
        target,
        tensors: Vec::new(),
        progs: Vec::new(),
        kv_row_insts: Vec::new(),
        prog_t: Vec::new(),
        gen: Vec::new(),
    };
    // Largest geometry first: tensors dedup by name and keep their first size.
    let mut builder = packet::devbuild::Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let codes = builder.tensor("in.codec.codes", u64::from(max_rows * cfg.frame_codes) * 4);
    let seed = builder.tensor("in.codec.seed", u64::from(bmax) * 8);
    let stages = cfg.blocks.len() + 1;
    let lengths: Vec<u32> = (0..stages).map(|k| builder.tensor(&format!("in.codec.lengths.{k}"), u64::from(bmax) * 4)).collect();
    let mut resolution = cfg.latent_dim.max(1024);
    let mut rows = max_rows * latent_per_frame;
    let mut widest = u64::from(rows) * u64::from(resolution);
    for b in &cfg.blocks {
        rows *= b.stride;
        resolution = b.cout;
        widest = widest.max(u64::from(rows) * u64::from(b.cin.max(b.cout)));
    }
    for name in ["act.codec.a", "act.codec.b", "act.codec.c", "act.codec.noise"] {
        builder.tensor(name, widest * 4);
    }
    let output = builder.tensor("act.codec.pcm", u64::from(max_rows * cfg.frame_samples) * 4);
    for l in 0..levels {
        builder.tensor(&format!("act.codec.level{l}"), u64::from(max_rows * per_level[l as usize]) * 4);
    }
    model.tensors = builder.tensors();
    let mut prefix = PacketPrefix { model, programs: Vec::new(), input: codes, output, input_shape: vec![u64::from(max_rows), u64::from(cfg.frame_codes)] };

    let mut roles = BTreeMap::new();
    for &(batch, frames) in SNAC_CAPACITIES {
        let program = prefix.model.progs.len();
        prefix = snac_program(prefix, &cfg, ConvFusions(sites), batch, frames, codes, seed, &lengths, &per_level)?;
        roles.insert(format!("decode.b{batch}.f{frames}"), program as u32);
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
    let section = pipeline_section(&prefix, &cfg, roles, bmax, max_rows, &lengths, seed)?;
    Ok(prefix.model.to_blob_v6(&[section]))
}

#[allow(clippy::too_many_arguments)]
fn snac_program(
    prefix: PacketPrefix,
    cfg: &SnacConfig,
    fuse: ConvFusions<'_>,
    batch: u32,
    frames: u32,
    codes: u32,
    seed: u32,
    lengths: &[u32],
    per_level: &[u32],
) -> Result<PacketPrefix, String> {
    let latent = cfg.latent_dim;
    let t0 = frames * cfg.vq_strides[0];
    let mut p = prefix.program();
    let fc = cfg.frame_codes;

    // Code demux: code position c of frame f -> level L row f * per_level[L] + slot.
    let mut deps = Vec::new();
    for (c, &(level, slot)) in cfg.frame_layout.iter().enumerate() {
        let n = per_level[level as usize];
        deps.push(p.copy_cols_f32(codes, &[], CopyColsF32Stage {
            output: TensorRef::Named(&format!("act.codec.level{level}")),
            items: batch,
            rows: frames,
            cols: 1,
            in_item_stride: frames * fc,
            in_stride: fc,
            in_offset: c as u32,
            out_item_stride: frames * n,
            out_stride: n,
            out_offset: slot,
        })?.done);
    }
    // Latent: sum over levels (coarse first, the reference order) of the projected codebooks,
    // each code repeated over the latent rows it covers.
    let z = TensorRef::Named("act.codec.a");
    let mut last = None;
    for level in 0..per_level.len() {
        let repeat = cfg.vq_strides[0] / per_level[level];
        let level_tensor = p_handle(&p, &format!("act.codec.level{level}"));
        let e = p.gather_rows_f32(&last.map_or(deps.clone(), |d| vec![d]), GatherRowsF32Stage {
            output: z,
            table: TensorRef::Named(&format!("w.q.P{level}")),
            table_rows: cfg.codebook_size,
            index: Some(level_tensor),
            rows: batch * t0,
            width: latent,
            vocab: cfg.codebook_size,
            rows_per_item: t0,
            repeat,
            index_item_stride: frames * per_level[level],
            table_item_stride: 0,
            table_f16: false,
            accumulate: level > 0,
            out_stride: 0,
            out_col0: 0,
        })?;
        last = Some(e.done);
    }
    let conv = |output: &'static str, weight: &str, bias: Option<String>, in_rows: u32, cin: u32, cout: u32, kernel: u32| Conv {
        output,
        weight: weight.to_string(),
        bias,
        alpha: None,
        batch,
        in_rows,
        cin,
        cout,
        kernel,
        stride: 1,
        dilation: 1,
        groups: 1,
        pad: 0,
        len: lengths[0],
        act_in: Activation::None,
        act_out: Activation::None,
        residual: None,
        transpose: false,
        input_is_temp: false,
    };
    let z_h = p_handle(&p, "act.codec.a");
    let a_handle = z_h;
    let mut c = conv("act.codec.b", "w.pre.dw.w", Some("w.pre.dw.b".into()), t0, latent, latent, 7);
    c.groups = latent;
    c.pad = 3;
    let e = fused_conv(&mut p, fuse, z_h, last.unwrap(), c)?;
    let pre_out = cfg.blocks.first().map_or(latent, |b| b.cin);
    let mut x = fused_conv(&mut p, fuse, e.output, e.done, conv("act.codec.a", "w.pre.pw.w", Some("w.pre.pw.b".into()), t0, latent, pre_out, 1))?;
    let mut rows = t0;
    for (i, b) in cfg.blocks.iter().enumerate() {
        // Transposed convolution (input snake), crop `padding` on both sides.
        let mut c = conv("act.codec.b", &format!("w.blk{i}.up.w"), Some(format!("w.blk{i}.up.b")), rows, b.cin, b.cout, b.kernel);
        c.alpha = Some(format!("w.blk{i}.snake"));
        c.act_in = Activation::Snake;
        c.transpose = true;
        c.stride = b.stride;
        c.dilation = b.output_padding;
        c.pad = b.padding;
        c.len = lengths[i];
        let up = fused_conv(&mut p, fuse, x.output, x.done, c)?;
        rows = (rows - 1) * b.stride + b.kernel + b.output_padding - 2 * b.padding;
        let len = lengths[i + 1];
        // Noise injection: x + n[b, t] * (x @ Wn^T), n ~ N(0, 1) keyed (seed, block, item, row).
        // After the previous block consumed the shared noise buffer (write-after-read). The
        // rewrite extracts it as `FusedGatedResidual`, which no op executes.
        let noise = p.rand_f32(seed, &[x.done], RandF32Stage {
            output: TensorRef::Named("act.codec.noise"),
            items: batch,
            rows,
            width: 1,
            stream: i as u32,
            stream_shift: 58,
            // Per-item seed keyed by (block, row) only, so batching never changes a request's
            // noise; the key equals the reference decoder's at batch index 0.
            a: RandCoord::Column,
            b: RandCoord::Row,
            a_offset: 0,
            b_offset: 0,
            normal: true,
            shared_seed: false,
            scale: 1.0,
            offset: 0.0,
        })?;
        let mut c = conv("act.codec.c", &format!("w.blk{i}.noise.w"), None, rows, b.cout, b.cout, 1);
        c.len = len;
        let proj = fused_conv(&mut p, fuse, up.output, up.done, c)?;
        let scaled = p.binary_f32(proj.output, noise.output, &[proj.done, noise.done], BinaryF32Stage {
            output: TensorRef::Handle(proj.output),
            op: BinaryOp::Mul,
            items: batch,
            rows,
            width: b.cout,
            b_item_stride: rows,
            b_row_stride: 1,
            b_col_stride: 0,
            scale: None,
        })?;
        x = p.binary_f32(up.output, scaled.output, &[scaled.done], BinaryF32Stage {
            output: TensorRef::Handle(up.output),
            op: BinaryOp::Add,
            items: batch,
            rows,
            width: b.cout,
            b_item_stride: rows * b.cout,
            b_row_stride: b.cout,
            b_col_stride: 1,
            scale: None,
        })?;
        for (j, &d) in b.dilations.iter().enumerate() {
            let pre = format!("w.blk{i}.ru{j}");
            let mut c = conv("act.codec.c", &format!("{pre}.dw.w"), Some(format!("{pre}.dw.b")), rows, b.cout, b.cout, 7);
            c.alpha = Some(format!("{pre}.a1"));
            c.act_in = Activation::Snake;
            c.dilation = d;
            c.groups = b.cout;
            c.pad = 3 * d;
            c.len = len;
            let h = fused_conv(&mut p, fuse, x.output, x.done, c)?;
            let target = if x.output == a_handle { "act.codec.b" } else { "act.codec.a" };
            let mut c = conv(target, &format!("{pre}.pw.w"), Some(format!("{pre}.pw.b")), rows, b.cout, b.cout, 1);
            c.alpha = Some(format!("{pre}.a2"));
            c.act_in = Activation::Snake;
            c.residual = Some(x.output);
            c.len = len;
            c.input_is_temp = true;
            x = fused_conv(&mut p, fuse, h.output, h.done, c)?;
        }
    }
    let last_c = cfg.blocks.last().map_or(latent, |b| b.cout);
    let mut c = conv("act.codec.pcm", "w.out.w", Some("w.out.b".into()), rows, last_c, 1, 7);
    c.alpha = Some("w.out.snake".into());
    c.act_in = Activation::Snake;
    c.act_out = Activation::Tanh;
    c.pad = 3;
    c.len = lengths[cfg.blocks.len()];
    let out = fused_conv(&mut p, fuse, x.output, x.done, c)?;
    debug_assert_eq!(out.output, p_handle(&p, "act.codec.pcm"));
    Ok(p.finish(batch * frames))
}

/// One convolution of the decoder in its unfused meaning: `residual + act_out(conv(act_in(x)))`.
struct Conv {
    output: &'static str,
    weight: String,
    bias: Option<String>,
    /// Snake alpha of `act_in`.
    alpha: Option<String>,
    batch: u32,
    in_rows: u32,
    cin: u32,
    cout: u32,
    kernel: u32,
    stride: u32,
    /// Output padding for a transposed convolution.
    dilation: u32,
    groups: u32,
    pad: u32,
    len: u32,
    act_in: Activation,
    act_out: Activation,
    residual: Option<u32>,
    transpose: bool,
    /// `x` is dead after this conv, so an unfused input activation may overwrite it.
    input_is_temp: bool,
}

/// `c` with the fusions `fuse` decides; an unfused piece runs as its own `UnaryF32` /
/// `BinaryF32` pass (an input activation in place when `x` is a temporary, else into
/// `act.codec.s`).
fn fused_conv(p: &mut StageProgram, fuse: ConvFusions<'_>, x: u32, dep: u32, c: Conv) -> Result<Emitted, String> {
    let act_in = c.act_in != Activation::None;
    let fuse_in = act_in && fuse.input_act(&c.weight, c.kernel, true);
    let fuse_out = c.act_out != Activation::None && fuse.output_act(&c.weight, true);
    let fuse_res = c.residual.is_some() && fuse.residual(&c.weight, true);
    let (mut x, mut dep) = (x, dep);
    if act_in && !fuse_in {
        let output = if c.input_is_temp { TensorRef::Handle(x) } else { TensorRef::Named("act.codec.s") };
        let e = p.unary_f32(x, &[dep], UnaryF32Stage {
            output,
            param: c.alpha.as_deref().map(TensorRef::Named),
            rows: c.batch * c.in_rows,
            width: c.cin,
            stride: 0,
            col0: 0,
            kind: c.act_in,
            p0: 0.0,
            p1: 0.0,
        })?;
        (x, dep) = (e.output, e.done);
    }
    let e = p.conv1d_f32(x, c.transpose, &[dep], Conv1dF32Stage {
        output: TensorRef::Named(c.output),
        weight: TensorRef::Named(&c.weight),
        bias: c.bias.as_deref().map(TensorRef::Named),
        alpha: if fuse_in { c.alpha.as_deref().map(TensorRef::Named) } else { None },
        residual: if fuse_res { c.residual } else { None },
        lengths: Some(c.len),
        batch: c.batch,
        in_rows: c.in_rows,
        in_channels: c.cin,
        out_channels: c.cout,
        kernel: c.kernel,
        stride: c.stride,
        dilation_or_output_padding: c.dilation,
        groups: c.groups,
        pad_before: c.pad,
        pad_after: c.pad,
        pad_mode: PadMode::Zero,
        input_activation: if fuse_in { c.act_in } else { Activation::None },
        output_activation: if fuse_out { c.act_out } else { Activation::None },
        slope: 0.0,
        weight_f16: false,
    })?;
    let out_rows = if c.transpose {
        (c.in_rows - 1) * c.stride + c.kernel + c.dilation - 2 * c.pad
    } else {
        c.in_rows + 2 * c.pad - c.dilation * (c.kernel - 1)
    };
    let mut e = e;
    if c.act_out != Activation::None && !fuse_out {
        e = p.unary_f32(e.output, &[e.done], UnaryF32Stage {
            output: TensorRef::Handle(e.output),
            param: None,
            rows: c.batch * out_rows,
            width: c.cout,
            stride: 0,
            col0: 0,
            kind: c.act_out,
            p0: 0.0,
            p1: 0.0,
        })?;
    }
    if let Some(r) = c.residual.filter(|_| !fuse_res) {
        e = p.binary_f32(e.output, r, &[e.done], BinaryF32Stage {
            output: TensorRef::Handle(e.output),
            op: BinaryOp::Add,
            items: 1,
            rows: c.batch * out_rows,
            width: c.cout,
            b_item_stride: 0,
            b_row_stride: c.cout,
            b_col_stride: 1,
            scale: None,
        })?;
    }
    Ok(e)
}

fn p_handle(p: &crate::pipeline::StageProgram, name: &str) -> u32 {
    p.handle_of(name).unwrap_or_else(|| panic!("codec tensor {name} is not declared"))
}

fn pipeline_section(
    prefix: &PacketPrefix,
    cfg: &SnacConfig,
    roles: BTreeMap<String, u32>,
    bmax: u32,
    max_rows: u32,
    lengths: &[u32],
    seed: u32,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION};
    let t = |h: u32| prefix.model.tensors[h as usize].name.clone();
    let mut tensors = BTreeMap::from([
        ("codes".into(), PipelineTensor { name: t(prefix.input), dtype: PipelineDType::U32, shape: vec![u64::from(max_rows * cfg.frame_codes)] }),
        ("seed".into(), PipelineTensor { name: t(seed), dtype: PipelineDType::U32, shape: vec![u64::from(bmax) * 2] }),
        ("pcm".into(), PipelineTensor { name: t(prefix.output), dtype: PipelineDType::F32, shape: vec![u64::from(max_rows * cfg.frame_samples)] }),
    ]);
    let mut parameters = BTreeMap::from([
        ("codec.frame_codes".into(), u64::from(cfg.frame_codes)),
        ("codec.frame_samples".into(), u64::from(cfg.frame_samples)),
        ("codec.codebook".into(), u64::from(cfg.codebook_size)),
        ("audio.sample_rate".into(), u64::from(cfg.sampling_rate)),
        ("lengths.count".into(), lengths.len() as u64),
        // Streaming decode: frames of left context per window and right context before a frame
        // is final (the decoder's receptive field, measured against whole-utterance decodes).
        ("stream.window_frames".into(), 6),
        ("stream.lookahead_frames".into(), 2),
    ]);
    // Valid rows per frame at each resolution: the host writes frames * scale per item.
    let mut scale = u64::from(cfg.vq_strides[0]);
    for (k, &h) in lengths.iter().enumerate() {
        tensors.insert(format!("lengths.{k}"), PipelineTensor { name: t(h), dtype: PipelineDType::U32, shape: vec![u64::from(bmax)] });
        parameters.insert(format!("lengths.{k}.rows_per_frame"), scale);
        if let Some(b) = cfg.blocks.get(k) {
            scale *= u64::from(b.stride);
        }
    }
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            name: PIPELINE.into(),
            driver: "codec.v1".into(),
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
