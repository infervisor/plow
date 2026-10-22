//! Fish S2 Pro's reference-audio encoder (the modded-DAC codec's encoder + quantizer,
//! `scripts/tts/fish_encoder_export.py`) lowered to a `codec.encode.v1` packet: 44.1 kHz PCM in,
//! 10 codes per 2048-sample frame out.
//!
//! causal conv k7 -> 4 stages (3 Snake residual units, Snake + strided causal conv) -> window-512
//! transformer -> Snake, conv k3 -> 2 x (conv k2 s2, ConvNeXt) -> window-128 transformer -> 10
//! cosine VQ stages over the 1024-wide residual (semantic, then 9 residual codebooks). Every layer
//! is causal and the PCM is zero-padded to whole frames as the reference does, so every frame
//! equals the reference; all math is plain FP32 (bf16 weights flip codes: near-tie argmaxes).

use std::collections::BTreeMap;

use packet::dev::DevOp;

use crate::codec::{fused_conv, p_handle, Conv};
use crate::fish_codec::Seq;
use crate::pipeline::{
    Activation, AttentionF32Stage, BinaryOp, Conv1dF32Stage, CopyColsF32Stage, Emitted,
    GatherRowsF32Stage, LayerNormRowsF32Stage, PacketPrefix, PadMode, TensorRef,
};
use crate::rewrite_lower::ConvFusions;

pub use crate::mimi_encoder::{DRIVER, PACKET, PIPELINE};

/// Frame capacities (21.5 Hz): one reference clip per launch, at most 23.8 s.
pub const CAPACITIES: &[u32] = &[32, 64, 128, 256, 512];

struct Xf {
    layers: u32,
    window: u32,
}

struct Cfg {
    sampling_rate: u32,
    frame_samples: u32,
    channels: u32,
    strides: Vec<u32>,
    kernel: u32,
    dilations: Vec<u32>,
    out_kernel: u32,
    hidden: u32,
    heads: u32,
    head_dim: u32,
    inter: u32,
    eps: f32,
    theta: f32,
    xf: Xf,
    downs: u32,
    cnx_kernel: u32,
    ln_eps: f32,
    pre: Xf,
    cb_dim: u32,
    sizes: Vec<u32>,
}

impl Cfg {
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let err = |k: &str| format!("fish encoder config: {k}");
        let u = |v: &serde_json::Value, k: &str| -> Result<u32, String> {
            v[k].as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .ok_or(err(k))
        };
        let f = |v: &serde_json::Value, k: &str| -> Result<f32, String> {
            v[k].as_f64().map(|x| x as f32).ok_or(err(k))
        };
        let list = |k: &str| -> Result<Vec<u32>, String> {
            v[k].as_array()
                .ok_or(err(k))?
                .iter()
                .map(|x| x.as_u64().map(|x| x as u32).ok_or(err(k)))
                .collect()
        };
        let xf = |k: &str| -> Result<Xf, String> {
            Ok(Xf {
                layers: u(&v[k], "layers")?,
                window: u(&v[k], "window")?,
            })
        };
        let cfg = Cfg {
            sampling_rate: u(v, "sampling_rate")?,
            frame_samples: u(v, "frame_samples")?,
            channels: u(v, "channels")?,
            strides: list("strides")?,
            kernel: u(v, "kernel")?,
            dilations: list("dilations")?,
            out_kernel: u(v, "out_kernel")?,
            hidden: u(v, "hidden")?,
            heads: u(v, "heads")?,
            head_dim: u(v, "head_dim")?,
            inter: u(v, "intermediate")?,
            eps: f(v, "eps")?,
            theta: f(v, "rope_theta")?,
            xf: xf("transformer")?,
            downs: u(&v["downsample"], "count")?,
            cnx_kernel: u(&v["downsample"], "convnext_kernel")?,
            ln_eps: f(&v["downsample"], "ln_eps")?,
            pre: xf("pre")?,
            cb_dim: u(v, "codebook_dim")?,
            sizes: list("codebook_sizes")?,
        };
        let hop: u32 = cfg.strides.iter().product();
        if v["model_type"] != "fish_dac_encoder"
            || v["downsample"]["stride"] != 2
            || hop << cfg.downs != cfg.frame_samples
            || cfg.channels << cfg.strides.len() != cfg.hidden
            || cfg.heads * cfg.head_dim != cfg.hidden
            || cfg.sizes.is_empty()
        {
            return Err("unsupported fish encoder export".into());
        }
        Ok(cfg)
    }

    /// Rows per frame of the first transformer (the rate before the downsamples).
    fn rows_per_frame(&self) -> u32 {
        1 << self.downs
    }

    fn codebook(&self) -> u32 {
        self.sizes.iter().copied().max().unwrap_or(0)
    }
}

pub fn lower_fish_encoder(
    dir: &std::path::Path,
    n_cu: u32,
    target: u32,
) -> Result<(packet::devbuild::Model, packet::devbuild::SectionData), String> {
    let cfg = Cfg::parse(
        &serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .map_err(|e| format!("{}: {e}", dir.display()))?,
        )
        .map_err(|e| format!("fish encoder config: {e}"))?,
    )?;
    let fmax = CAPACITIES.iter().copied().max().unwrap_or(1);
    let samples = u64::from(fmax * cfg.frame_samples);
    let mut builder = packet::devbuild::Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let pcm = builder.tensor("in.enc.pcm", samples * 4);
    // The widest activation: the first stage's residual input (`channels` at the sample rate).
    let rows = u64::from(fmax * cfg.rows_per_frame());
    let widest = (samples * u64::from(cfg.channels))
        .max(rows * u64::from(3 * cfg.hidden).max(u64::from(cfg.inter)))
        .max(rows / 2 * u64::from(4 * cfg.hidden))
        .max(u64::from(fmax) * u64::from(cfg.codebook()));
    for name in ["act.codec.a", "act.codec.b", "act.codec.c", "act.codec.s"] {
        builder.tensor(name, widest * 4);
    }
    builder.tensor("act.enc.ids", u64::from(fmax) * 4);
    let codes = builder.tensor(
        "act.enc.codes",
        u64::from(fmax) * cfg.sizes.len() as u64 * 4,
    );
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
        input_shape: vec![samples],
    };
    let mut roles = BTreeMap::new();
    let mut positions = Vec::new();
    for &frames in CAPACITIES {
        let program = prefix.model.progs.len();
        let (p, pos) = program_for(prefix, &cfg, frames, pcm)?;
        prefix = p;
        positions.extend(pos);
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

/// Causal RoPE transformer over `t` rows of `x` (in place, final RMSNorm included); `h` is the
/// other ping-pong buffer.
#[allow(clippy::too_many_arguments)]
fn transformer(
    s: &mut Seq,
    cfg: &Cfg,
    x: u32,
    h: (u32, &str),
    pos: u32,
    t: u32,
    xf: &Xf,
    pre: &str,
) -> Result<(), String> {
    let (dim, hd, heads) = (cfg.hidden, cfg.head_dim, cfg.heads);
    for l in 0..xf.layers {
        let w = |s: &str| format!("w.{pre}.{l}.{s}");
        s.rms(x, h.0, t, dim, &w("attn_norm"), cfg.eps)?;
        let qkv = s.dense(None, h.0, "act.codec.c", t, dim, 3 * dim, &w("wqkv"), None)?;
        // flags 2: the reference's bf16 cos / sin table (fp32 angles flip ~1% of codes).
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
                window: xf.window,
                stream: None,
            },
        )?;
        let at = s.done(e);
        s.dense(None, at, h.1, t, dim, dim, &w("wo_s"), None)?;
        s.binary(x, h.0, BinaryOp::Add, t, dim)?;
        s.rms(x, h.0, t, dim, &w("ffn_norm"), cfg.eps)?;
        let g = s.dense(None, h.0, "act.codec.c", t, dim, cfg.inter, &w("w1"), None)?;
        let u = s.dense(None, h.0, "act.codec.s", t, dim, cfg.inter, &w("w3"), None)?;
        s.unary(g, Activation::Silu, t, cfg.inter)?;
        s.binary(g, u, BinaryOp::Mul, t, cfg.inter)?;
        s.dense(None, g, h.1, t, cfg.inter, dim, &w("w2_s"), None)?;
        s.binary(x, h.0, BinaryOp::Add, t, dim)?;
    }
    s.rms(x, x, t, dim, &format!("w.{pre}.norm"), cfg.eps)?;
    Ok(())
}

/// One `encode.f{frames}` program, and its two position tensors (handle, rows).
fn program_for(
    prefix: PacketPrefix,
    cfg: &Cfg,
    frames: u32,
    pcm: u32,
) -> Result<(PacketPrefix, [(u32, u32); 2]), String> {
    let mut p = prefix.program();
    let t = frames * cfg.rows_per_frame();
    let pos_xf = p.declare(&format!("c.enc.pos.f{frames}.xf"), u64::from(t) * 4);
    let pos_pre = p.declare(&format!("c.enc.pos.f{frames}.pre"), u64::from(frames) * 4);
    let fuse = ConvFusions(None);
    let conv = |output: &'static str, weight: String, in_rows: u32, cin: u32, cout: u32| Conv {
        output,
        bias: Some(weight.replace(".w", ".b")),
        weight,
        alpha: None,
        batch: 1,
        in_rows,
        cin,
        cout,
        kernel: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
        pad: 0,
        pad_after: Some(0),
        len: packet::dev::TENSOR_NONE,
        act_in: Activation::None,
        act_out: Activation::None,
        residual: None,
        transpose: false,
        input_is_temp: true,
        row_scale: None,
        alpha_out: None,
        taps_wgmma: false,
    };
    // Causal: `(k - 1) * d + 1 - s` zero rows on the left, none on the right at whole frames.
    let causal = |mut c: Conv, kernel: u32, stride: u32, dilation: u32| {
        c.kernel = kernel;
        c.stride = stride;
        c.dilation = dilation;
        c.pad = (kernel - 1) * dilation + 1 - stride;
        c
    };
    let snake = |mut c: Conv, alpha: String| {
        c.act_in = Activation::Snake;
        c.alpha = Some(alpha);
        c
    };
    let a = p_handle(&p, "act.codec.a");
    let other = |e: u32| if e == a { "act.codec.b" } else { "act.codec.a" };
    let mut rows = frames * cfg.frame_samples;
    let mut cur = p.conv1d_f32(
        pcm,
        false,
        &[],
        Conv1dF32Stage {
            output: TensorRef::Handle(a),
            weight: TensorRef::Named("w.enc.c0.w"),
            bias: Some(TensorRef::Named("w.enc.c0.b")),
            alpha: None,
            residual: None,
            lengths: None,
            batch: 1,
            in_rows: rows,
            in_channels: 1,
            out_channels: cfg.channels,
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
    let mut d = cfg.channels;
    for (i, &stride) in cfg.strides.iter().enumerate() {
        let w = |s: &str| format!("w.enc.b{i}.{s}");
        for (j, &dil) in cfg.dilations.iter().enumerate() {
            let ru = |s: &str| w(&format!("ru{j}.{s}"));
            let mut c = causal(
                snake(conv("act.codec.c", ru("c1.w"), rows, d, d), ru("a1")),
                cfg.kernel,
                1,
                dil,
            );
            c.input_is_temp = false;
            let h = fused_conv(&mut p, fuse, None, cur.output, cur.done, c)?;
            let mut c = snake(conv(other(cur.output), ru("c2.w"), rows, d, d), ru("a2"));
            c.residual = Some(cur.output);
            cur = fused_conv(&mut p, fuse, None, h.output, h.done, c)?;
        }
        let c = causal(
            snake(
                conv(other(cur.output), w("down.w"), rows, d, 2 * d),
                w("snake"),
            ),
            2 * stride,
            stride,
            1,
        );
        cur = fused_conv(&mut p, fuse, None, cur.output, cur.done, c)?;
        rows /= stride;
        d *= 2;
    }
    debug_assert_eq!(rows, t);
    let dim = cfg.hidden;

    let ping = |x: u32, p: &crate::pipeline::StageProgram| {
        let name = other(x);
        (p_handle(p, name), name)
    };
    let mut s = Seq {
        p: Some(p),
        last: cur.done,
    };
    let x = cur.output;
    let h = ping(x, s.p());
    transformer(&mut s, cfg, x, h, pos_xf, t, &cfg.xf, "xf")?;
    let mut p = s.p.take().unwrap();
    let c = causal(
        snake(
            conv(other(x), "w.enc.out.w".into(), rows, dim, dim),
            "w.enc.snake".into(),
        ),
        cfg.out_kernel,
        1,
        1,
    );
    cur = fused_conv(&mut p, fuse, None, x, s.last, c)?;
    for u in 0..cfg.downs {
        let w = |s: &str| format!("w.dn{u}.{s}");
        let c = causal(
            conv(other(cur.output), w("conv.w"), rows, dim, dim),
            2,
            2,
            1,
        );
        let down = fused_conv(&mut p, fuse, None, cur.output, cur.done, c)?;
        rows /= 2;
        let mut c = causal(
            conv("act.codec.c", w("cnx.dw.w"), rows, dim, dim),
            cfg.cnx_kernel,
            1,
            1,
        );
        c.groups = dim;
        c.input_is_temp = false;
        let dw = fused_conv(&mut p, fuse, None, down.output, down.done, c)?;
        let e = p.layer_norm_f32(
            dw.output,
            &[dw.done],
            LayerNormRowsF32Stage {
                output: TensorRef::Handle(dw.output),
                gamma: Some(TensorRef::Named(&w("cnx.ln.w"))),
                beta: Some(TensorRef::Named(&w("cnx.ln.b"))),
                rows,
                width: dim,
                epsilon: cfg.ln_eps,
                activation: Activation::None,
                add_row: None,
            },
        )?;
        let mut s = Seq {
            p: Some(p),
            last: e.done,
        };
        let y = s.dense(
            None,
            dw.output,
            "act.codec.s",
            rows,
            dim,
            4 * dim,
            &w("cnx.1.w"),
            Some(&w("cnx.1.b")),
        )?;
        s.unary(y, Activation::GeluErf, rows, 4 * dim)?;
        let z = s.dense(
            None,
            y,
            "act.codec.c",
            rows,
            4 * dim,
            dim,
            &w("cnx.2.w_s"),
            Some(&w("cnx.2.b_s")),
        )?;
        s.binary(down.output, z, BinaryOp::Add, rows, dim)?;
        cur = Emitted {
            output: down.output,
            done: s.last,
        };
        p = s.p.take().unwrap();
    }
    debug_assert_eq!(rows, frames);

    let r = cur.output;
    let h = ping(r, &p);
    let mut s = Seq {
        p: Some(p),
        last: cur.done,
    };
    transformer(&mut s, cfg, r, h, pos_pre, frames, &cfg.pre, "pre")?;

    // Cosine VQ over the 1024-wide residual: argmax of in_proj(r) . normalized codebook (its
    // positive row scale does not move the argmax), then r -= out_proj(codebook[code]).
    let ids = p_handle(s.p(), "act.enc.ids");
    let codes = p_handle(s.p(), "act.enc.codes");
    let fc = cfg.sizes.len() as u32;
    for (k, &size) in cfg.sizes.iter().enumerate() {
        let w = |s: &str| format!("w.vq{k}.{s}");
        let ze = s.dense(
            None,
            r,
            "act.codec.c",
            frames,
            dim,
            cfg.cb_dim,
            &w("win"),
            Some(&w("bin")),
        )?;
        let logits = s.dense(
            None,
            ze,
            "act.codec.s",
            frames,
            cfg.cb_dim,
            size,
            &w("C"),
            None,
        )?;
        s.raw(DevOp::ArgmaxF32, u64::from(frames), ids, |d| {
            d.t[..2].copy_from_slice(&[ids, logits]);
            d.i[..2].copy_from_slice(&[frames, size]);
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
                out_offset: k as u32,
            },
        )?;
        s.done(e);
        if k + 1 == cfg.sizes.len() {
            break;
        }
        let dep = s.last;
        let e = s.p().gather_rows_f32(
            &[dep],
            GatherRowsF32Stage {
                output: TensorRef::Named("act.codec.c"),
                table: TensorRef::Named(&w("T")),
                table_rows: size,
                index: Some(ids),
                rows: frames,
                width: dim,
                vocab: size,
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
        s.binary(r, q, BinaryOp::Sub, frames, dim)?;
    }
    let p = s.p.take().unwrap();
    Ok((p.finish(frames), [(pos_xf, t), (pos_pre, frames)]))
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
    let fc = cfg.sizes.len() as u64;
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
                        shape: vec![u64::from(fmax), fc],
                    },
                ),
            ]),
            parameters: BTreeMap::from([
                ("audio.sample_rate".into(), u64::from(cfg.sampling_rate)),
                ("codec.frame_samples".into(), u64::from(cfg.frame_samples)),
                ("codec.frame_codes".into(), fc),
                ("codec.codebook".into(), u64::from(cfg.codebook())),
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
