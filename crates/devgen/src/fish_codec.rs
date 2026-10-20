//! Fish S2 Pro's codec decoder (modded DAC, `scripts/tts/fish_codec_export.py`) lowered to a
//! `codec.v1` packet: 10 codes per frame in, 2048 PCM samples per frame out.
//!
//! codebook tables summed -> 8-layer causal transformer (sliding window, RoPE, LayerScale folded
//! into wo / w2) -> 2 x (ConvTranspose k2 s2, ConvNeXt) -> causal conv k7 -> 4 snake decoder blocks
//! (ConvTranspose k=2s cropped right, three residual units) -> snake, conv k7, tanh. Every conv is
//! causal, so a batched item decodes exactly as it would alone; `scripts/tts/fish_codec_ref.py` is
//! the reference.
//!
//! Breeze TTS 2's Qwen3-TTS-Tokenizer-12Hz decoder (`model_type` `qwen3_tts_12hz`,
//! `scripts/tts/breeze_codec_export.py`) is the same skeleton with: projection-folded tables of
//! width `vq_dim`, a causal pre-conv to `latent`, the transformer between an input and an output
//! projection (`dim` residual, `heads * head_dim` attention width), SnakeBeta activations as their
//! own passes (`.sb` = `[exp(alpha) | 1 / (exp(beta) + 1e-9)]`), and a final clamp.

use std::collections::BTreeMap;

use packet::dev::DevOp;

use crate::codec::{fused_conv, p_handle, Conv};
use crate::pipeline::{
    Activation, AttentionF32Stage, BinaryF32Stage, BinaryOp, Conv1dF32Stage, CopyColsF32Stage,
    Emitted, GatherRowsF32Stage, LayerNormRowsF32Stage, PacketPrefix, PadMode, StageProgram,
    TensorRef, UnaryF32Stage,
};
use crate::rewrite_lower::ConvFusions;

type Splits<'a> = Option<&'a std::cell::RefCell<BTreeMap<String, (u32, u32, u32)>>>;

/// (batch, frames): streaming windows batched across streams, long utterances alone. The smallest
/// frame count sets the stream chunk (`frames + 1 - stream.window_frames` = 17 frames): a capacity
/// under the 48-frame window would re-decode the whole window for every frame.
pub const CAPACITIES: &[(u32, u32)] = &[
    (1, 64),
    (2, 64),
    (4, 64),
    (8, 64),
    (16, 64),
    (1, 256),
    (2, 256),
    (1, 512),
];

struct Block {
    cin: u32,
    cout: u32,
    kernel: u32,
    stride: u32,
    dilations: Vec<u32>,
}

struct Cfg {
    breeze: bool,
    sampling_rate: u32,
    frame_samples: u32,
    sizes: Vec<u32>,
    /// Table width (the summed frame embedding) and the conv stack's width after the transformer.
    vq_dim: u32,
    latent: u32,
    pre_kernel: u32,
    dim: u32,
    layers: u32,
    heads: u32,
    head_dim: u32,
    inter: u32,
    eps: f32,
    window: u32,
    theta: f32,
    ups: Vec<(u32, f32)>,
    channels: u32,
    blocks: Vec<Block>,
}

impl Cfg {
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let u = |v: &serde_json::Value, k: &str| -> Result<u32, String> {
            v[k].as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .ok_or(format!("fish codec config: {k}"))
        };
        let f = |v: &serde_json::Value, k: &str| -> Result<f32, String> {
            v[k].as_f64()
                .map(|x| x as f32)
                .ok_or(format!("fish codec config: {k}"))
        };
        let t = &v["transformer"];
        let d = &v["decoder"];
        let list = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64().map(|x| x as u32))
                        .collect()
                })
                .unwrap_or_default()
        };
        let breeze = v["model_type"] == "qwen3_tts_12hz";
        let dim = u(t, "dim")?;
        let cfg = Cfg {
            breeze,
            vq_dim: if breeze { u(v, "vq_dim")? } else { dim },
            latent: if breeze { u(v, "latent_dim")? } else { dim },
            pre_kernel: if breeze { u(v, "pre_kernel")? } else { 0 },
            sampling_rate: u(v, "sampling_rate")?,
            frame_samples: u(v, "frame_samples")?,
            sizes: list(&v["codebook_sizes"]),
            dim: u(t, "dim")?,
            layers: u(t, "layers")?,
            heads: u(t, "heads")?,
            head_dim: u(t, "head_dim")?,
            inter: u(t, "intermediate")?,
            eps: f(t, "eps")?,
            window: u(t, "window")?,
            theta: f(t, "rope_theta")?,
            ups: v["upsample"]
                .as_array()
                .ok_or("fish codec config: upsample")?
                .iter()
                .map(|x| Ok((u(x, "stride")?, f(x, "ln_eps")?)))
                .collect::<Result<_, String>>()?,
            channels: u(d, "channels")?,
            blocks: d["blocks"]
                .as_array()
                .ok_or("fish codec config: decoder.blocks")?
                .iter()
                .map(|b| {
                    Ok(Block {
                        cin: u(b, "cin")?,
                        cout: u(b, "cout")?,
                        kernel: u(b, "kernel")?,
                        stride: u(b, "stride")?,
                        dilations: list(&b["dilations"]),
                    })
                })
                .collect::<Result<_, String>>()?,
        };
        let up: u32 = cfg.ups.iter().map(|u| u.0).product::<u32>()
            * cfg.blocks.iter().map(|b| b.stride).product::<u32>();
        if (v["model_type"] != "fish_dac" && !breeze)
            || cfg.sizes.is_empty()
            || (!breeze && cfg.heads * cfg.head_dim != cfg.dim)
            || up != cfg.frame_samples
            || cfg.blocks.first().map(|b| b.cin) != Some(cfg.channels)
            || cfg.blocks.iter().any(|b| b.kernel != 2 * b.stride)
        {
            return Err("unsupported fish_dac export".into());
        }
        Ok(cfg)
    }

    /// Attention width: q, k, v and the output projection's input.
    fn attn(&self) -> u32 {
        self.heads * self.head_dim
    }

    /// Rows per frame after each resolution change: frames, each upsample, each decoder block.
    fn scales(&self) -> Vec<u32> {
        let mut s = vec![1u32];
        for &(stride, _) in &self.ups {
            s.push(s.last().unwrap() * stride);
        }
        for b in &self.blocks {
            s.push(s.last().unwrap() * b.stride);
        }
        s
    }
}

pub fn lower_fish_dac(
    dir: &std::path::Path,
    n_cu: u32,
    target: u32,
    split_bf16: bool,
) -> Result<(packet::devbuild::Model, packet::devbuild::SectionData), String> {
    let cfg = Cfg::parse(
        &serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .map_err(|e| format!("{}: {e}", dir.display()))?,
        )
        .map_err(|e| format!("fish codec config: {e}"))?,
    )?;
    let fc = cfg.sizes.len() as u32;
    let scales = cfg.scales();
    let bmax = CAPACITIES.iter().map(|&(b, _)| b).max().unwrap_or(1);
    let max_rows = CAPACITIES.iter().map(|&(b, f)| b * f).max().unwrap_or(1);
    let mut builder = packet::devbuild::Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let codes = builder.tensor("in.codec.codes", u64::from(max_rows * fc) * 4);
    let seed = builder.tensor("in.codec.seed", u64::from(bmax) * 8);
    let lengths: Vec<u32> = (0..scales.len())
        .map(|k| builder.tensor(&format!("in.codec.lengths.{k}"), u64::from(bmax) * 4))
        .collect();
    let mut widest = u64::from(max_rows)
        * u64::from(
            cfg.dim
                .max(2 * cfg.inter)
                .max(3 * cfg.attn())
                .max(cfg.latent)
                .max(cfg.vq_dim),
        );
    let mut rows = max_rows;
    for &(stride, _) in &cfg.ups {
        rows *= stride;
        widest = widest.max(u64::from(rows) * u64::from(4 * cfg.latent));
    }
    widest = widest.max(u64::from(rows) * u64::from(cfg.channels));
    for b in &cfg.blocks {
        rows *= b.stride;
        widest = widest.max(u64::from(rows) * u64::from(b.cin.max(b.cout)));
    }
    for name in ["act.codec.a", "act.codec.b", "act.codec.c", "act.codec.s"] {
        builder.tensor(name, widest * 4);
    }
    for k in 0..fc {
        builder.tensor(&format!("act.codec.level{k}"), u64::from(max_rows) * 4);
    }
    let output = builder.tensor("act.codec.pcm", u64::from(max_rows * cfg.frame_samples) * 4);
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
        input: codes,
        output,
        input_shape: vec![u64::from(max_rows), u64::from(fc)],
    };
    let splits = split_bf16.then(|| std::cell::RefCell::new(BTreeMap::new()));
    let mut roles = BTreeMap::new();
    let mut positions = Vec::new();
    for &(batch, frames) in CAPACITIES {
        let program = prefix.model.progs.len();
        let (p, pos) = program_for(
            prefix,
            &cfg,
            batch,
            frames,
            codes,
            &lengths,
            splits.as_ref(),
        )?;
        prefix = p;
        positions.push((pos, batch, frames));
        roles.insert(format!("decode.b{batch}.f{frames}"), program as u32);
    }
    let reader = crate::checkpoint::TensorReader::open(dir)?;
    for (h, tensor) in prefix.model.tensors.iter_mut().enumerate() {
        if let Some(&(_, batch, frames)) = positions.iter().find(|(p, _, _)| *p as usize == h) {
            let pos: Vec<u8> = (0..batch * frames)
                .flat_map(|r| (r % frames).to_le_bytes())
                .collect();
            tensor.init = Some(pos);
            continue;
        }
        let Some(name) = tensor.name.strip_prefix("w.") else {
            continue;
        };
        let (dtype, mut bytes) = match name.strip_suffix(".a12") {
            Some(unit) => {
                let (d1, mut a) = reader.read(&format!("{unit}.a1"))?;
                let (_, b) = reader.read(&format!("{unit}.a2"))?;
                a.extend_from_slice(&b);
                (d1, a)
            }
            None => reader.read(name)?,
        };
        // Every ConvT whose Conv takes phase-major weights (`codec::phase_major`): the decoder
        // blocks' upsamplers and the post-transformer ConvNeXt upsamplers (kernel = stride).
        let convt = cfg
            .blocks
            .iter()
            .enumerate()
            .find(|(i, _)| name == format!("dec.blk{i}.up.w"))
            .map(|(_, b)| (b.cin, b.cout, b.kernel, b.stride))
            .or_else(|| {
                cfg.ups
                    .iter()
                    .enumerate()
                    .find(|(u, _)| name == format!("up{u}.convt.w"))
                    .map(|(_, &(stride, _))| (cfg.latent, cfg.latent, stride, stride))
            });
        if let Some((cin, cout, kernel, stride)) = convt {
            if cin % 4 == 0 && cout >= 16 {
                bytes = crate::codec::to_phase_major(&bytes, cin, cout, kernel, stride);
            }
        }
        if let Some((n, k, taps)) = splits.as_ref().and_then(|s| s.borrow().get(name).copied()) {
            if taps > 1 {
                bytes = crate::s3gen::to_tap_major(&bytes, n, k / taps, taps);
            }
            bytes = crate::pipeline::wgmma_weight_split(&bytes, n, k);
        }
        if dtype != "F32" || bytes.len() as u64 != tensor.bytes {
            return Err(format!(
                "{name}: {dtype} {} bytes, expected F32 {}",
                bytes.len(),
                tensor.bytes
            ));
        }
        tensor.init = Some(bytes);
    }
    let section = section(
        &prefix, &cfg, roles, bmax, max_rows, &lengths, &scales, seed,
    )?;
    Ok((prefix.model, section))
}

/// A transformer / ConvNeXt step chain inside one program: each op depends on the previous.
struct Seq {
    p: Option<StageProgram>,
    last: u32,
}

impl Seq {
    fn p(&mut self) -> &mut StageProgram {
        self.p.as_mut().unwrap()
    }

    fn done(&mut self, e: Emitted) -> u32 {
        self.last = e.done;
        e.output
    }

    /// `x @ w^T + b` as a pointwise Conv1dF32 (weights `[n][k]`), on split-bf16 wgmma with `splits`.
    #[allow(clippy::too_many_arguments)]
    fn dense(
        &mut self,
        splits: Splits<'_>,
        x: u32,
        out: &str,
        rows: u32,
        k: u32,
        n: u32,
        w: &str,
        b: Option<&str>,
    ) -> Result<u32, String> {
        let split = splits.is_some_and(|s| {
            s.borrow_mut()
                .insert(w.trim_start_matches("w.").to_string(), (n, k, 1));
            true
        });
        let dep = self.last;
        let e = self.p().conv1d_f32(
            x,
            false,
            &[dep],
            Conv1dF32Stage {
                output: TensorRef::Named(out),
                weight: TensorRef::Named(w),
                bias: b.map(TensorRef::Named),
                alpha: None,
                residual: None,
                lengths: None,
                batch: 1,
                in_rows: rows,
                in_channels: k,
                out_channels: n,
                kernel: 1,
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
                split_bf16: split,
                weight_tap_major: false,
                wgmma: split,
                weight_split: split,
                dual_alpha: false,
            },
        )?;
        Ok(self.done(e))
    }

    fn raw(
        &mut self,
        op: DevOp,
        units: u64,
        out: u32,
        fill: impl FnOnce(&mut packet::dev::DevInst),
    ) -> Result<u32, String> {
        let dep = self.last;
        let e = self.p().raw(op, units, &[dep], out, fill)?;
        Ok(self.done(e))
    }

    fn rms(
        &mut self,
        x: u32,
        out: u32,
        rows: u32,
        width: u32,
        gamma: &str,
        eps: f32,
    ) -> Result<u32, String> {
        let g = self.p().declare(gamma, u64::from(width) * 4);
        self.raw(DevOp::RmsNormF32, u64::from(rows).div_ceil(8), out, |d| {
            d.t[..3].copy_from_slice(&[out, x, g]);
            d.i[..6].copy_from_slice(&[rows, 1, width, 0, 1, 0]);
            d.f[0] = eps;
        })
    }

    fn binary(
        &mut self,
        a: u32,
        b: u32,
        op: BinaryOp,
        rows: u32,
        width: u32,
    ) -> Result<u32, String> {
        let dep = self.last;
        let e = self.p().binary_f32(
            a,
            b,
            &[dep],
            BinaryF32Stage {
                output: TensorRef::Handle(a),
                op,
                items: 1,
                rows,
                width,
                b_item_stride: 0,
                b_row_stride: width,
                b_col_stride: 1,
                scale: None,
            },
        )?;
        Ok(self.done(e))
    }

    fn unary(&mut self, x: u32, kind: Activation, rows: u32, width: u32) -> Result<u32, String> {
        let dep = self.last;
        let e = self.p().unary_f32(
            x,
            &[dep],
            UnaryF32Stage {
                output: TensorRef::Handle(x),
                param: None,
                rows,
                width,
                stride: 0,
                col0: 0,
                kind,
                p0: 0.0,
                p1: 0.0,
            },
        )?;
        Ok(self.done(e))
    }
}

#[allow(clippy::too_many_arguments)]
fn program_for(
    prefix: PacketPrefix,
    cfg: &Cfg,
    batch: u32,
    frames: u32,
    codes: u32,
    lengths: &[u32],
    splits: Option<&std::cell::RefCell<BTreeMap<String, (u32, u32, u32)>>>,
) -> Result<(PacketPrefix, u32), String> {
    let fc = cfg.sizes.len() as u32;
    let n = batch * frames;
    let dim = cfg.dim;
    let mut p = prefix.program();
    let pos = p.declare(&format!("c.codec.pos.b{batch}.f{frames}"), u64::from(n) * 4);

    // Codebook k's codes -> level k, then z = sum_k table_k[code_k].
    let mut deps = Vec::new();
    for k in 0..fc {
        deps.push(
            p.copy_cols_f32(
                codes,
                &[],
                CopyColsF32Stage {
                    output: TensorRef::Named(&format!("act.codec.level{k}")),
                    items: batch,
                    rows: frames,
                    cols: 1,
                    in_item_stride: frames * fc,
                    in_stride: fc,
                    in_offset: k,
                    out_item_stride: frames,
                    out_stride: 1,
                    out_offset: 0,
                },
            )?
            .done,
        );
    }
    let mut last = None;
    for k in 0..fc {
        let level = p_handle(&p, &format!("act.codec.level{k}"));
        let e = p.gather_rows_f32(
            &last.map_or(deps.clone(), |d| vec![d]),
            GatherRowsF32Stage {
                output: TensorRef::Named(if cfg.breeze {
                    "act.codec.c"
                } else {
                    "act.codec.a"
                }),
                table: TensorRef::Named(&format!("w.vq.T{k}")),
                table_rows: cfg.sizes[k as usize],
                index: Some(level),
                rows: n,
                width: cfg.vq_dim,
                vocab: cfg.sizes[k as usize],
                rows_per_item: frames,
                repeat: 1,
                index_item_stride: frames,
                table_item_stride: 0,
                table_f16: false,
                accumulate: k > 0,
                out_stride: 0,
                out_col0: 0,
            },
        )?;
        last = Some(e.done);
    }
    let mut last = last.unwrap();
    if cfg.breeze {
        // Causal pre-conv (vq_dim -> latent), then the transformer's input projection.
        let c = p_handle(&p, "act.codec.c");
        let k = cfg.pre_kernel;
        let pre = Conv {
            output: "act.codec.b",
            weight: "w.pre.w".into(),
            bias: Some("w.pre.b".into()),
            alpha: None,
            batch,
            in_rows: frames,
            cin: cfg.vq_dim,
            cout: cfg.latent,
            kernel: k,
            stride: 1,
            dilation: 1,
            groups: 1,
            pad: k - 1,
            pad_after: Some(0),
            len: lengths[0],
            act_in: Activation::None,
            act_out: Activation::None,
            residual: None,
            transpose: false,
            input_is_temp: false,
            row_scale: None,
            alpha_out: None,
            taps_wgmma: false,
        };
        last = fused_conv(&mut p, ConvFusions(None), splits, c, last, pre)?.done;
    }
    let mut s = Seq { p: Some(p), last };
    let x = p_handle(s.p(), "act.codec.a");
    let h = p_handle(s.p(), "act.codec.b");
    if cfg.breeze {
        s.dense(
            splits,
            h,
            "act.codec.a",
            n,
            cfg.latent,
            dim,
            "w.post.in.w",
            Some("w.post.in.b"),
        )?;
    }
    let (hd, heads, aw) = (cfg.head_dim, cfg.heads, cfg.attn());
    for l in 0..cfg.layers {
        let w = |s: &str| format!("w.post.layers.{l}.{s}");
        s.rms(x, h, n, dim, &w("attn_norm"), cfg.eps)?;
        let qkv = s.dense(splits, h, "act.codec.c", n, dim, 3 * aw, &w("wqkv"), None)?;
        // q and k are adjacent: one rotation over 2 * heads heads of the 3 * aw columns.
        s.raw(
            DevOp::RopeAxialF32,
            u64::from(n * 2 * heads * hd / 2).div_ceil(2048),
            qkv,
            |d| {
                d.t[..2].copy_from_slice(&[qkv, pos]);
                d.i[..7].copy_from_slice(&[n, 2 * heads, hd, 3 * aw, 1, 0, 2]);
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
                key_lengths: Some(lengths[0]),
                bias: None,
                batch,
                q_rows: frames,
                kv_rows: frames,
                heads,
                head_width: hd,
                in_stride: 3 * aw,
                k_col0: aw,
                v_col0: 2 * aw,
                causal: true,
                scale: 1.0 / (hd as f32).sqrt(),
                bias_head_stride: 0,
                relative: false,
                key_length_heads: 0,
                prefix: None,
                window: cfg.window,
            },
        )?;
        let at = s.done(e);
        s.dense(splits, at, "act.codec.b", n, aw, dim, &w("wo_s"), None)?;
        s.binary(x, h, BinaryOp::Add, n, dim)?;
        s.rms(x, h, n, dim, &w("ffn_norm"), cfg.eps)?;
        let g = s.dense(splits, h, "act.codec.c", n, dim, cfg.inter, &w("w1"), None)?;
        let u = s.dense(splits, h, "act.codec.s", n, dim, cfg.inter, &w("w3"), None)?;
        s.unary(g, Activation::Silu, n, cfg.inter)?;
        s.binary(g, u, BinaryOp::Mul, n, cfg.inter)?;
        s.dense(
            splits,
            g,
            "act.codec.b",
            n,
            cfg.inter,
            dim,
            &w("w2_s"),
            None,
        )?;
        s.binary(x, h, BinaryOp::Add, n, dim)?;
    }
    s.rms(x, x, n, dim, "w.post.norm", cfg.eps)?;
    let stream = if cfg.breeze {
        s.dense(
            splits,
            x,
            "act.codec.b",
            n,
            dim,
            cfg.latent,
            "w.post.out.w",
            Some("w.post.out.b"),
        )?
    } else {
        x
    };
    let dim = cfg.latent;

    let fuse = ConvFusions(None);
    let conv = |output: &'static str,
                weight: String,
                bias: Option<String>,
                in_rows: u32,
                cin: u32,
                cout: u32,
                kernel: u32| Conv {
        output,
        weight,
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
        pad: kernel - 1,
        pad_after: Some(0),
        len: lengths[0],
        act_in: Activation::None,
        act_out: Activation::None,
        residual: None,
        transpose: false,
        input_is_temp: false,
        row_scale: None,
        alpha_out: None,
        taps_wgmma: false,
    };
    let other = |e: u32| if e == x { "act.codec.b" } else { "act.codec.a" };
    let snake = if cfg.breeze {
        Activation::SnakeBeta
    } else {
        Activation::Snake
    };
    let mut cur = Emitted {
        output: stream,
        done: s.last,
    };
    let mut rows = frames;
    for (u, &(stride, ln_eps)) in cfg.ups.iter().enumerate() {
        let w = |s: &str| format!("w.up{u}.{s}");
        let mut c = conv(
            other(cur.output),
            w("convt.w"),
            Some(w("convt.b")),
            rows,
            dim,
            dim,
            stride,
        );
        c.transpose = true;
        c.stride = stride;
        c.dilation = 0;
        c.pad = 0;
        c.len = lengths[u];
        let mut p = s.p.take().unwrap();
        let up = fused_conv(&mut p, fuse, splits, cur.output, cur.done, c)?;
        rows *= stride;
        let mut c = conv(
            "act.codec.c",
            w("cnx.dw.w"),
            Some(w("cnx.dw.b")),
            rows,
            dim,
            dim,
            7,
        );
        c.groups = dim;
        c.len = lengths[u + 1];
        let dw = fused_conv(&mut p, fuse, splits, up.output, up.done, c)?;
        let e = p.layer_norm_f32(
            dw.output,
            &[dw.done],
            LayerNormRowsF32Stage {
                output: TensorRef::Handle(dw.output),
                gamma: Some(TensorRef::Named(&w("cnx.ln.w"))),
                beta: Some(TensorRef::Named(&w("cnx.ln.b"))),
                rows: batch * rows,
                width: dim,
                epsilon: ln_eps,
                activation: Activation::None,
                add_row: None,
            },
        )?;
        s = Seq {
            p: Some(p),
            last: e.done,
        };
        let y = s.dense(
            splits,
            dw.output,
            "act.codec.s",
            batch * rows,
            dim,
            4 * dim,
            &w("cnx.1.w"),
            Some(&w("cnx.1.b")),
        )?;
        s.unary(y, Activation::GeluErf, batch * rows, 4 * dim)?;
        let z = s.dense(
            splits,
            y,
            "act.codec.c",
            batch * rows,
            4 * dim,
            dim,
            &w("cnx.2.w_s"),
            Some(&w("cnx.2.b_s")),
        )?;
        s.binary(up.output, z, BinaryOp::Add, batch * rows, dim)?;
        cur = Emitted {
            output: up.output,
            done: s.last,
        };
    }
    let mut p = s.p.take().unwrap();
    let c = conv(
        other(cur.output),
        "w.dec.in.w".into(),
        Some("w.dec.in.b".into()),
        rows,
        dim,
        cfg.channels,
        7,
    );
    let mut c = c;
    c.len = lengths[cfg.ups.len()];
    cur = fused_conv(&mut p, fuse, splits, cur.output, cur.done, c)?;
    for (i, b) in cfg.blocks.iter().enumerate() {
        let w = |s: &str| format!("w.dec.blk{i}.{s}");
        let len = lengths[cfg.ups.len() + 1 + i];
        let len_in = lengths[cfg.ups.len() + i];
        // Causal transposed conv: (rows - 1) * s + k outputs, the last k - s cropped.
        let mut c = conv(
            other(cur.output),
            w("up.w"),
            Some(w("up.b")),
            rows,
            b.cin,
            b.cout,
            b.kernel,
        );
        c.alpha = Some(w("snake"));
        c.act_in = snake;
        c.transpose = true;
        c.stride = b.stride;
        c.dilation = 0;
        c.pad = 0;
        c.pad_after = Some(b.kernel - b.stride);
        c.len = len_in;
        c.input_is_temp = true;
        cur = fused_conv(&mut p, fuse, splits, cur.output, cur.done, c)?;
        rows *= b.stride;
        for (j, &d) in b.dilations.iter().enumerate() {
            let pre = w(&format!("ru{j}"));
            let mut c = conv(
                "act.codec.c",
                format!("{pre}.c1.w"),
                Some(format!("{pre}.c1.b")),
                rows,
                b.cout,
                b.cout,
                7,
            );
            c.taps_wgmma = true;
            c.act_in = snake;
            c.dilation = d;
            c.pad = 6 * d;
            c.len = len;
            if cfg.breeze {
                c.alpha = Some(format!("{pre}.a1"));
            } else {
                c.alpha_out = Some(format!("{pre}.a12"));
                c.act_out = Activation::Snake;
            }
            let mut h = fused_conv(&mut p, fuse, splits, cur.output, cur.done, c)?;
            if cfg.breeze {
                h = snake_beta(&mut p, h, &format!("{pre}.a2"), batch * rows, b.cout)?;
            }
            let mut c = conv(
                other(cur.output),
                format!("{pre}.c2.w"),
                Some(format!("{pre}.c2.b")),
                rows,
                b.cout,
                b.cout,
                1,
            );
            c.residual = Some(cur.output);
            c.len = len;
            c.input_is_temp = true;
            cur = fused_conv(&mut p, fuse, splits, h.output, h.done, c)?;
        }
    }
    let last_c = cfg.blocks.last().map_or(cfg.channels, |b| b.cout);
    let mut c = conv(
        "act.codec.pcm",
        "w.dec.out.w".into(),
        Some("w.dec.out.b".into()),
        rows,
        last_c,
        1,
        7,
    );
    c.alpha = Some("w.dec.out.snake".into());
    c.act_in = snake;
    c.act_out = if cfg.breeze {
        Activation::None
    } else {
        Activation::Tanh
    };
    c.len = *lengths.last().unwrap();
    let pcm = fused_conv(&mut p, fuse, splits, cur.output, cur.done, c)?;
    if cfg.breeze {
        p.unary_f32(
            pcm.output,
            &[pcm.done],
            UnaryF32Stage {
                output: TensorRef::Handle(pcm.output),
                param: None,
                rows: batch * rows,
                width: 1,
                stride: 0,
                col0: 0,
                kind: Activation::Clamp,
                p0: -1.0,
                p1: 1.0,
            },
        )?;
    }
    Ok((p.finish(n), pos))
}

/// In-place SnakeBeta over `rows x width` of `x` (parameter `[a | r]`).
fn snake_beta(
    p: &mut StageProgram,
    x: Emitted,
    param: &str,
    rows: u32,
    width: u32,
) -> Result<Emitted, String> {
    p.unary_f32(
        x.output,
        &[x.done],
        UnaryF32Stage {
            output: TensorRef::Handle(x.output),
            param: Some(TensorRef::Named(param)),
            rows,
            width,
            stride: 0,
            col0: 0,
            kind: Activation::SnakeBeta,
            p0: 0.0,
            p1: 0.0,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn section(
    prefix: &PacketPrefix,
    cfg: &Cfg,
    roles: BTreeMap<String, u32>,
    bmax: u32,
    max_rows: u32,
    lengths: &[u32],
    scales: &[u32],
    seed: u32,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{
        PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION,
    };
    let t = |h: u32| prefix.model.tensors[h as usize].name.clone();
    let fc = cfg.sizes.len() as u64;
    let mut tensors = BTreeMap::from([
        (
            "codes".into(),
            PipelineTensor {
                name: t(prefix.input),
                dtype: PipelineDType::U32,
                shape: vec![u64::from(max_rows) * fc],
            },
        ),
        (
            "seed".into(),
            PipelineTensor {
                name: t(seed),
                dtype: PipelineDType::U32,
                shape: vec![u64::from(bmax) * 2],
            },
        ),
        (
            "pcm".into(),
            PipelineTensor {
                name: t(prefix.output),
                dtype: PipelineDType::F32,
                shape: vec![u64::from(max_rows * cfg.frame_samples)],
            },
        ),
    ]);
    let mut parameters = BTreeMap::from([
        ("codec.frame_codes".into(), fc),
        ("codec.frame_samples".into(), u64::from(cfg.frame_samples)),
        ("codec.codebook".into(), u64::from(cfg.sizes[0])),
        ("audio.sample_rate".into(), u64::from(cfg.sampling_rate)),
        ("lengths.count".into(), lengths.len() as u64),
        // Causal decoder: no right context; the left window bounds the replay error (the exact
        // receptive field spans every attention window).
        ("stream.window_frames".into(), 48),
        ("codec.breeze".into(), u64::from(cfg.breeze)),
        ("stream.lookahead_frames".into(), 0),
    ]);
    for (k, (&h, &scale)) in lengths.iter().zip(scales).enumerate() {
        tensors.insert(
            format!("lengths.{k}"),
            PipelineTensor {
                name: t(h),
                dtype: PipelineDType::U32,
                shape: vec![u64::from(bmax)],
            },
        );
        parameters.insert(format!("lengths.{k}.rows_per_frame"), u64::from(scale));
    }
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            name: crate::codec::PIPELINE.into(),
            driver: "codec.v1".into(),
            programs: roles,
            tensors,
            parameters,
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
