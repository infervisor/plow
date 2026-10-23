//! Breeze TTS 2's reference-audio encoder (Qwen3-TTS-Tokenizer-12Hz encoder = HF Mimi,
//! `scripts/tts/mimi_encoder_export.py`) lowered to a `codec.encode.v1` packet: 24 kHz PCM in,
//! `semantic + acoustic` codes per 1920-sample frame out.
//!
//! causal conv k7 -> 4 SEANet stages (ELU residual unit, ELU strided causal conv) -> ELU, conv k3
//! -> 8-layer causal transformer (LayerNorm, half-split RoPE, GELU MLP, LayerScale folded into
//! wo / fc2; the reference's eager attention ignores the 250 window) -> causal conv k4 s2
//! (replicate pad) -> split RVQ: each codebook's nearest centroid is the argmax of
//! `x . e - |e|^2 / 2`. The PCM is zero-padded to whole frames, so every full frame equals the
//! reference (`scripts/tts/mimi_encoder_ref.py`); a trailing partial frame may differ, as the
//! reference pads each layer instead.

use std::collections::BTreeMap;

use packet::dev::DevOp;

use crate::codec::{fused_conv, p_handle, Conv};
use crate::fish_codec::Seq;
use crate::pipeline::{
    Activation, AttentionF32Stage, BinaryOp, Conv1dF32Stage, CopyColsF32Stage, GatherRowsF32Stage,
    LayerNormRowsF32Stage, PacketPrefix, PadMode, TensorRef,
};
use crate::rewrite_lower::ConvFusions;

pub const PACKET: &str = "ref_encoder.pkt";
pub const DRIVER: &str = "codec.encode.v1";
pub const PIPELINE: &str = "codec.encode";

/// Frame capacities (12.5 Hz): one reference clip per launch, at most 20.48 s.
pub const CAPACITIES: &[u32] = &[32, 64, 128, 256];

struct Cfg {
    sampling_rate: u32,
    frame_samples: u32,
    filters: u32,
    ratios: Vec<u32>,
    kernel: u32,
    last_kernel: u32,
    residual_kernel: u32,
    compress: u32,
    hidden: u32,
    layers: u32,
    heads: u32,
    head_dim: u32,
    inter: u32,
    eps: f32,
    theta: f32,
    down_kernel: u32,
    down_stride: u32,
    cb_dim: u32,
    cb_size: u32,
    semantic: u32,
    acoustic: u32,
}

impl Cfg {
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let u = |v: &serde_json::Value, k: &str| -> Result<u32, String> {
            v[k].as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .ok_or(format!("mimi encoder config: {k}"))
        };
        let f = |k: &str| -> Result<f32, String> {
            v[k].as_f64()
                .map(|x| x as f32)
                .ok_or(format!("mimi encoder config: {k}"))
        };
        let cfg = Cfg {
            sampling_rate: u(v, "sampling_rate")?,
            frame_samples: u(v, "frame_samples")?,
            filters: u(v, "filters")?,
            ratios: v["ratios"]
                .as_array()
                .ok_or("mimi encoder config: ratios")?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|x| x as u32)
                        .ok_or("mimi encoder config: ratio")
                })
                .collect::<Result<_, _>>()?,
            kernel: u(v, "kernel")?,
            last_kernel: u(v, "last_kernel")?,
            residual_kernel: u(v, "residual_kernel")?,
            compress: u(v, "compress")?,
            hidden: u(v, "hidden")?,
            layers: u(v, "layers")?,
            heads: u(v, "heads")?,
            head_dim: u(v, "head_dim")?,
            inter: u(v, "intermediate")?,
            eps: f("eps")?,
            theta: f("rope_theta")?,
            down_kernel: u(&v["downsample"], "kernel")?,
            down_stride: u(&v["downsample"], "stride")?,
            cb_dim: u(v, "codebook_dim")?,
            cb_size: u(v, "codebook_size")?,
            semantic: u(v, "semantic")?,
            acoustic: u(v, "acoustic")?,
        };
        let down: u32 = cfg.ratios.iter().product::<u32>() * cfg.down_stride;
        if v["model_type"] != "mimi_encoder"
            || down != cfg.frame_samples
            || cfg.heads * cfg.head_dim != cfg.hidden
            || cfg.semantic == 0
            || cfg.compress == 0
            || cfg.filters % cfg.compress != 0
        {
            return Err("unsupported mimi encoder export".into());
        }
        Ok(cfg)
    }

    fn codebooks(&self) -> u32 {
        self.semantic + self.acoustic
    }

    /// Transformer rows per frame (the rate before the downsample).
    fn rows_per_frame(&self) -> u32 {
        self.down_stride
    }
}

pub fn lower_mimi_encoder(
    dir: &std::path::Path,
    n_cu: u32,
    target: u32,
) -> Result<(packet::devbuild::Model, packet::devbuild::SectionData), String> {
    let cfg = Cfg::parse(
        &serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .map_err(|e| format!("{}: {e}", dir.display()))?,
        )
        .map_err(|e| format!("mimi encoder config: {e}"))?,
    )?;
    let fmax = CAPACITIES.iter().copied().max().unwrap_or(1);
    let samples = fmax * cfg.frame_samples;
    let mut builder = packet::devbuild::Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let pcm = builder.tensor("in.enc.pcm", u64::from(samples) * 4);
    // The widest activation: the first stage's residual input (`filters` at the sample rate), or
    // the transformer / RVQ logits rows.
    let rows = fmax * cfg.rows_per_frame();
    let widest = (u64::from(samples) * u64::from(cfg.filters))
        .max(u64::from(rows) * u64::from(3 * cfg.hidden).max(u64::from(cfg.inter)))
        .max(u64::from(fmax) * u64::from(cfg.cb_size));
    for name in ["act.codec.a", "act.codec.b", "act.codec.c", "act.codec.s"] {
        builder.tensor(name, widest * 4);
    }
    builder.tensor("act.enc.ids", u64::from(fmax) * 4);
    let codes = builder.tensor("act.enc.codes", u64::from(fmax * cfg.codebooks()) * 4);
    let model = packet::devbuild::Model {
        n_cu,
        target,
        tensors: builder.tensors(),
        progs: Vec::new(),
        kv_row_insts: Vec::new(),
        prog_t: Vec::new(),
        gen: Vec::new(),
    };
    let mut prefix = PacketPrefix {
        model,
        programs: Vec::new(),
        input: pcm,
        output: codes,
        input_shape: vec![u64::from(samples)],
    };
    let mut roles = BTreeMap::new();
    let mut positions = Vec::new();
    for &frames in CAPACITIES {
        let program = prefix.model.progs.len();
        let (p, pos) = program_for(prefix, &cfg, frames, pcm)?;
        prefix = p;
        positions.push((pos, frames * cfg.rows_per_frame()));
        roles.insert(format!("encode.f{frames}"), program as u32);
    }
    prefix.output = codes;
    let reader = crate::checkpoint::TensorReader::open(dir)?;
    for (h, tensor) in prefix.model.tensors.iter_mut().enumerate() {
        if let Some(&(_, rows)) = positions.iter().find(|(p, _)| *p as usize == h) {
            tensor.init = Some((0..rows).flat_map(u32::to_le_bytes).collect());
            continue;
        }
        let Some(name) = tensor.name.strip_prefix("w.") else {
            continue;
        };
        let (dtype, bytes) = reader.read(name)?;
        if dtype != "F32" || bytes.len() as u64 != tensor.bytes {
            return Err(format!(
                "{name}: {dtype} {} bytes, expected F32 {}",
                bytes.len(),
                tensor.bytes
            ));
        }
        tensor.init = Some(bytes);
    }
    let section = section(&prefix, &cfg, roles, fmax)?;
    Ok((prefix.model, section))
}

fn program_for(
    prefix: PacketPrefix,
    cfg: &Cfg,
    frames: u32,
    pcm: u32,
) -> Result<(PacketPrefix, u32), String> {
    let mut p = prefix.program();
    let t = frames * cfg.rows_per_frame();
    let pos = p.declare(&format!("c.enc.pos.f{frames}"), u64::from(t) * 4);
    let fuse = ConvFusions(None);
    let conv = |output: &'static str,
                weight: String,
                in_rows: u32,
                cin: u32,
                cout: u32,
                kernel: u32,
                stride: u32| Conv {
        output,
        bias: Some(weight.replace(".w", ".b")),
        weight,
        alpha: None,
        batch: 1,
        in_rows,
        cin,
        cout,
        kernel,
        stride,
        dilation: 1,
        groups: 1,
        // Mimi's causal pad: kernel - stride rows on the left; whole frames need none on the right.
        pad: kernel - stride,
        pad_after: Some(0),
        len: packet::dev::TENSOR_NONE,
        act_in: Activation::Elu,
        act_out: Activation::None,
        residual: None,
        transpose: false,
        input_is_temp: true,
        row_scale: None,
        alpha_out: None,
        taps_wgmma: false,
    };
    let x = p_handle(&p, "act.codec.a");
    let other = |e: u32| if e == x { "act.codec.b" } else { "act.codec.a" };
    let mut rows = frames * cfg.frame_samples;
    let mut cur = p.conv1d_f32(
        pcm,
        false,
        &[],
        Conv1dF32Stage {
            output: TensorRef::Handle(x),
            weight: TensorRef::Named("w.enc.c0.w"),
            bias: Some(TensorRef::Named("w.enc.c0.b")),
            alpha: None,
            residual: None,
            lengths: None,
            batch: 1,
            in_rows: rows,
            in_channels: 1,
            out_channels: cfg.filters,
            kernel: cfg.kernel,
            stride: 1,
            dilation_or_output_padding: 1,
            groups: 1,
            pad_before: cfg.kernel - 1,
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
            dual_alpha: false,
        },
    )?;
    let mut d = cfg.filters;
    for (i, &r) in cfg.ratios.iter().enumerate() {
        let w = |s: &str| format!("w.enc.s{i}.{s}");
        let mut c = conv(
            "act.codec.c",
            w("r1.w"),
            rows,
            d,
            d / cfg.compress,
            cfg.residual_kernel,
            1,
        );
        c.input_is_temp = false;
        let h = fused_conv(&mut p, fuse, None, cur.output, cur.done, c)?;
        let mut c = conv(
            other(cur.output),
            w("r2.w"),
            rows,
            d / cfg.compress,
            d,
            1,
            1,
        );
        c.residual = Some(cur.output);
        cur = fused_conv(&mut p, fuse, None, h.output, h.done, c)?;
        let c = conv(other(cur.output), w("down.w"), rows, d, 2 * d, 2 * r, r);
        cur = fused_conv(&mut p, fuse, None, cur.output, cur.done, c)?;
        rows /= r;
        d *= 2;
    }
    let c = conv(
        other(cur.output),
        "w.enc.out.w".into(),
        rows,
        d,
        cfg.hidden,
        cfg.last_kernel,
        1,
    );
    cur = fused_conv(&mut p, fuse, None, cur.output, cur.done, c)?;
    debug_assert_eq!(rows, t);

    let mut s = Seq {
        p: Some(p),
        last: cur.done,
    };
    let x = cur.output;
    let h = p_handle(s.p(), other(x));
    let (dim, hd, heads) = (cfg.hidden, cfg.head_dim, cfg.heads);
    let norm = |s: &mut Seq, out: u32, w: String| -> Result<u32, String> {
        let dep = s.last;
        let e = s.p().layer_norm_f32(
            x,
            &[dep],
            LayerNormRowsF32Stage {
                output: TensorRef::Handle(out),
                gamma: Some(TensorRef::Named(&format!("{w}.w"))),
                beta: Some(TensorRef::Named(&format!("{w}.b"))),
                rows: t,
                width: dim,
                epsilon: cfg.eps,
                activation: Activation::None,
                add_row: None,
            },
        )?;
        Ok(s.done(e))
    };
    for l in 0..cfg.layers {
        let w = |s: &str| format!("w.xf.{l}.{s}");
        norm(&mut s, h, w("ln1"))?;
        let qkv = s.dense(None, h, "act.codec.c", t, dim, 3 * dim, &w("wqkv"), None)?;
        s.raw(
            DevOp::RopeAxialF32,
            u64::from(t * 2 * heads * hd / 2).div_ceil(2048),
            qkv,
            |d| {
                d.t[..2].copy_from_slice(&[qkv, pos]);
                d.i[..7].copy_from_slice(&[t, 2 * heads, hd, 3 * dim, 1, 0, 2]);
                d.f[0] = cfg.theta;
            },
        )?;
        let dep = s.last;
        let e = s.p().attention_f32(
            qkv,
            qkv,
            qkv,
            &[dep],
            AttentionF32Stage {
                output: TensorRef::Named("act.codec.s"),
                key_lengths: None,
                bias: None,
                batch: 1,
                q_rows: t,
                kv_rows: t,
                heads,
                head_width: hd,
                in_stride: 3 * dim,
                k_col0: dim,
                v_col0: 2 * dim,
                causal: true,
                scale: 1.0 / (hd as f32).sqrt(),
                bias_head_stride: 0,
                relative: false,
                key_length_heads: 0,
                prefix: None,
                window: 0,
                stream: None,
            },
        )?;
        let at = s.done(e);
        let o = s.dense(None, at, other(x), t, dim, dim, &w("wo_s"), None)?;
        s.binary(x, o, BinaryOp::Add, t, dim)?;
        norm(&mut s, h, w("ln2"))?;
        let g = s.dense(None, h, "act.codec.c", t, dim, cfg.inter, &w("fc1"), None)?;
        s.unary(g, Activation::GeluErf, t, cfg.inter)?;
        let o = s.dense(None, g, "act.codec.s", t, cfg.inter, dim, &w("fc2_s"), None)?;
        s.binary(x, o, BinaryOp::Add, t, dim)?;
    }

    // Causal conv k4 s2, replicate pad, no bias: transformer rows -> frames.
    let dep = s.last;
    let e = s.p().conv1d_f32(
        x,
        false,
        &[dep],
        Conv1dF32Stage {
            output: TensorRef::Handle(h),
            weight: TensorRef::Named("w.down.w"),
            bias: None,
            alpha: None,
            residual: None,
            lengths: None,
            batch: 1,
            in_rows: t,
            in_channels: dim,
            out_channels: dim,
            kernel: cfg.down_kernel,
            stride: cfg.down_stride,
            dilation_or_output_padding: 1,
            groups: 1,
            pad_before: cfg.down_kernel - cfg.down_stride,
            pad_after: 0,
            pad_mode: PadMode::Replicate,
            input_activation: Activation::None,
            output_activation: Activation::None,
            slope: 0.0,
            weight_f16: false,
            split_bf16: false,
            weight_tap_major: false,
            wgmma: false,
            weight_split: false,
            dual_alpha: false,
        },
    )?;
    let down = s.done(e);

    // Split RVQ over the frames: semantic codebooks, then the acoustic residual chain.
    let ids = p_handle(s.p(), "act.enc.ids");
    let codes = p_handle(s.p(), "act.enc.codes");
    let fc = cfg.codebooks();
    let mut col = 0;
    for (g, n) in [("sem", cfg.semantic), ("ac", cfg.acoustic)] {
        let r = s.dense(
            None,
            down,
            "act.codec.c",
            frames,
            dim,
            cfg.cb_dim,
            &format!("w.{g}.proj"),
            None,
        )?;
        for k in 0..n {
            let logits = s.dense(
                None,
                r,
                "act.codec.s",
                frames,
                cfg.cb_dim,
                cfg.cb_size,
                &format!("w.{g}.E{k}"),
                Some(&format!("w.{g}.nb{k}")),
            )?;
            s.raw(DevOp::ArgmaxF32, u64::from(frames), ids, |d| {
                d.t[..2].copy_from_slice(&[ids, logits]);
                d.i[..2].copy_from_slice(&[frames, cfg.cb_size]);
            })?;
            let dep = s.last;
            let e = s.p().copy_cols_f32(
                ids,
                &[dep],
                CopyColsF32Stage {
                    output: TensorRef::Handle(codes),
                    items: 1,
                    rows: frames,
                    cols: 1,
                    in_item_stride: frames,
                    in_stride: 1,
                    in_offset: 0,
                    out_item_stride: frames * fc,
                    out_stride: fc,
                    out_offset: col,
                },
                None,
            )?;
            s.done(e);
            col += 1;
            if k + 1 == n {
                break;
            }
            let dep = s.last;
            let e = s.p().gather_rows_f32(
                &[dep],
                GatherRowsF32Stage {
                    output: TensorRef::Named("act.codec.s"),
                    table: TensorRef::Named(&format!("w.{g}.E{k}")),
                    table_rows: cfg.cb_size,
                    index: Some(ids),
                    rows: frames,
                    width: cfg.cb_dim,
                    vocab: cfg.cb_size,
                    rows_per_item: frames,
                    repeat: 1,
                    index_item_stride: frames,
                    table_item_stride: 0,
                    table_f16: false,
                    accumulate: false,
                    out_stride: 0,
                    out_col0: 0,
                },
            )?;
            let q = s.done(e);
            s.binary(r, q, BinaryOp::Sub, frames, cfg.cb_dim)?;
        }
    }
    let p = s.p.take().unwrap();
    Ok((p.finish(frames), pos))
}

fn section(
    prefix: &PacketPrefix,
    cfg: &Cfg,
    roles: BTreeMap<String, u32>,
    fmax: u32,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{
        PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION,
    };
    let t = |h: u32| prefix.model.tensors[h as usize].name.clone();
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            name: PIPELINE.into(),
            driver: DRIVER.into(),
            programs: roles,
            tensors: BTreeMap::from([
                (
                    "pcm".into(),
                    PipelineTensor {
                        name: t(prefix.input),
                        dtype: PipelineDType::F32,
                        shape: vec![u64::from(fmax * cfg.frame_samples)],
                    },
                ),
                (
                    "codes".into(),
                    PipelineTensor {
                        name: t(prefix.output),
                        dtype: PipelineDType::U32,
                        shape: vec![u64::from(fmax), u64::from(cfg.codebooks())],
                    },
                ),
            ]),
            parameters: BTreeMap::from([
                ("audio.sample_rate".into(), u64::from(cfg.sampling_rate)),
                ("codec.frame_samples".into(), u64::from(cfg.frame_samples)),
                ("codec.frame_codes".into(), u64::from(cfg.codebooks())),
                ("codec.codebook".into(), u64::from(cfg.cb_size)),
            ]),
            strings: Default::default(),
        }],
    };
    metadata.validate(prefix.model.progs.len(), |name| {
        prefix
            .model
            .tensors
            .iter()
            .find(|x| x.name == name)
            .map(|x| x.bytes)
    })?;
    Ok(packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    })
}
