//! Maya 2 Native's DepthV2 (`scripts/tts/maya2_prep.py` weights) lowered to a standalone
//! `maya2.depth.v1` packet: per stream the backbone's final hidden `h` and cb0's embedding row in,
//! codebooks 1..31 out (plus each level's logits, for the parity gates).
//!
//! Level `L` (1..=31) runs its weight group's 4 blocks over the prefix `P[0..=L]`: `P[0] =
//! in_proj(h) + pos0`, `P[k] = emb[eidx_{k-1}]` (the folded `code_up(code_lo) + level_emb` row of the
//! code sampled at level `k - 1`). A block is pre-LayerNorm (bias) attention without RoPE, pre-LayerNorm
//! GELU-erf MLP, then the one-token cross-attention's constant. Where the group changes (levels 1..=9,
//! 17, 25) the whole prefix runs as rows with causal attention; a level sharing the previous level's
//! group runs one new row against the K | V the group cached (`act.depth.cache.{l}`, `[B][32]` rows of
//! the packed qkv). The final LayerNorm of the last position feeds head `L - 1`, sampled by
//! [`DevOp::SampleSmallF32`] (whose `eidx` is the next level's embedding row). Matrices are fp16 and
//! activations fp32, as the official engine serves the depth stage.

use std::collections::BTreeMap;

use packet::dev::DevOp;

use crate::codec::p_handle;
use crate::pipeline::{
    Activation, AttentionF32Stage, BinaryF32Stage, BinaryOp, Conv1dF32Stage, CopyColsF32Stage,
    GatherRowsF32Stage, LayerNormRowsF32Stage, PacketPrefix, PadMode, StageProgram, TensorRef,
};

pub const PACKET: &str = "depth.pkt";
pub const DRIVER: &str = "maya2.depth.v1";
pub const PIPELINE: &str = "maya2.depth";
/// Streams per launch.
pub const CAPACITIES: &[u32] = &[1, 2, 4, 8, 16];

struct Geo {
    hidden_in: u32,
    dim: u32,
    heads: u32,
    head_dim: u32,
    inter: u32,
    layers: u32,
    levels: u32,
    codes: u32,
    eps: f32,
    group_of_level: Vec<u32>,
}

impl Geo {
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let m = &v["maya2"];
        let d = &m["depth"];
        let u = |v: &serde_json::Value, k: &str| {
            v[k].as_u64()
                .map(|x| x as u32)
                .ok_or(format!("maya2 depth config: {k}"))
        };
        let g = Geo {
            hidden_in: u(v, "hidden_size")?,
            dim: u(d, "hidden")?,
            heads: u(d, "heads")?,
            head_dim: u(d, "head_dim")?,
            inter: u(d, "intermediate")?,
            layers: u(d, "layers")?,
            levels: u(d, "levels")?,
            codes: u(m, "codebook_size")?,
            eps: d["eps"].as_f64().ok_or("maya2 depth config: eps")? as f32,
            group_of_level: d["group_of_level"]
                .as_array()
                .ok_or("maya2 depth config: group_of_level")?
                .iter()
                .map(|x| x.as_u64().map(|x| x as u32).ok_or("group_of_level"))
                .collect::<Result<_, _>>()?,
        };
        if g.heads * g.head_dim != g.dim
            || g.group_of_level.len() as u32 != g.levels
            || u(m, "codebooks")? != g.levels + 1
        {
            return Err("unsupported maya2 depth config".into());
        }
        Ok(g)
    }

    /// Positions per stream: `h`, then one per level's input code.
    fn positions(&self) -> u32 {
        self.levels + 1
    }
}

/// The depth packet for a `scripts/tts/maya2_prep.py` checkpoint directory.
pub fn lower_maya2_depth(
    dir: &std::path::Path,
    n_cu: u32,
    target: u32,
) -> Result<(packet::devbuild::Model, packet::devbuild::SectionData), String> {
    let geo = Geo::parse(
        &serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .map_err(|e| format!("{}: {e}", dir.display()))?,
        )
        .map_err(|e| format!("maya2 depth config: {e}"))?,
    )?;
    let bmax = CAPACITIES.iter().copied().max().unwrap_or(1);
    let (b64, d, np) = (u64::from(bmax), u64::from(geo.dim), u64::from(geo.positions()));
    let cb = u64::from(geo.levels + 1);
    let mut builder = packet::devbuild::Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let h = builder.tensor("in.depth.h", b64 * u64::from(geo.hidden_in) * 4);
    for k in 0..geo.positions() {
        builder.tensor(&format!("act.depth.eidx.{k}"), b64 * 4);
    }
    builder.tensor_init(
        "in.depth.force",
        vec![0xFF; (b64 * cb * 4) as usize],
    );
    builder.tensor("in.depth.params", b64 * 4 * 4);
    builder.tensor("in.depth.ctl", b64 * 2 * 4);
    builder.tensor("in.depth.seed", b64 * 8);
    let codes = builder.tensor("out.depth.codes", b64 * cb * 4);
    builder.tensor(
        "out.depth.logits",
        b64 * u64::from(geo.levels * geo.codes) * 4,
    );
    for (name, width) in [
        ("act.depth.p0", d),
        ("act.depth.y", d),
        ("act.depth.yn", d),
        ("act.depth.logits", u64::from(geo.codes)),
    ] {
        builder.tensor(name, b64 * width * 4);
    }
    let qkv_w = 3 * d;
    for (name, width) in [
        ("act.depth.x", d),
        ("act.depth.h1", d),
        ("act.depth.at", d),
        ("act.depth.qkv", qkv_w),
        ("act.depth.f", u64::from(geo.inter)),
    ] {
        builder.tensor(name, b64 * np * width * 4);
    }
    for l in 0..geo.layers {
        builder.tensor(&format!("act.depth.cache.{l}"), b64 * np * qkv_w * 4);
    }
    for level in 1..=geo.levels {
        builder.tensor_init(
            &format!("c.depth.len.{level}"),
            (0..bmax).flat_map(|_| (level + 1).to_le_bytes()).collect(),
        );
    }
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
        input: h,
        output: codes,
        input_shape: vec![b64, u64::from(geo.hidden_in)],
    };
    let mut roles = BTreeMap::new();
    for &batch in CAPACITIES {
        let program = prefix.model.progs.len();
        prefix = program_for(prefix, &geo, batch)?;
        roles.insert(format!("depth.b{batch}.0"), program as u32);
    }
    prefix.output = codes;
    let reader = crate::checkpoint::TensorReader::open(dir)?;
    for tensor in prefix.model.tensors.iter_mut() {
        let Some(name) = tensor.name.strip_prefix("w.") else {
            continue;
        };
        let (dtype, bytes) = reader.read(name)?;
        if !matches!(dtype, "F16" | "F32") || bytes.len() as u64 != tensor.bytes {
            return Err(format!(
                "{name}: {dtype} {} bytes, expected {}",
                bytes.len(),
                tensor.bytes
            ));
        }
        tensor.init = Some(bytes);
    }
    let section = section(&prefix, &geo, roles, bmax)?;
    Ok((prefix.model, section))
}

/// Projections over at most this many rows stream their weights through `GemvF32Out` (f32 x, f16 W);
/// the tiled Conv1dF32 wastes most of its tile on fewer rows (132 us per 6 MB projection at B=1).
const GEMV_ROWS: u32 = 256;
/// The GEMV's LayerNorm prologue normalises a staged row of at most this many columns.
const GEMV_LN_K: u32 = 1024;

/// What a projection adds after the matmul: `act(acc + bias + residual)`.
#[derive(Clone, Copy)]
struct Epilogue<'a> {
    bias: Option<&'a str>,
    residual: Option<u32>,
    gelu: bool,
    /// LayerNorm (`{ln}.weight`, `{ln}.bias`, eps) of each x row before the matmul: fused into
    /// the GEMV's staged rows, a separate pass before the conv.
    ln: Option<(&'a str, f32)>,
}

impl Epilogue<'_> {
    const NONE: Epilogue<'static> = Epilogue {
        bias: None,
        residual: None,
        gelu: false,
        ln: None,
    };
}

/// An op chain inside one program: each op depends on the previous.
struct Chain {
    p: StageProgram,
    last: Option<u32>,
    n_cu: u32,
}

impl Chain {
    fn deps(&self) -> Vec<u32> {
        self.last.into_iter().collect()
    }

    fn done(&mut self, e: crate::pipeline::Emitted) -> u32 {
        self.last = Some(e.done);
        e.output
    }

    /// `x @ w^T (+ b)` over `rows` rows, fp16 weights `[n][k]`, fp32 bias.
    #[allow(clippy::too_many_arguments)]
    fn dense(
        &mut self,
        x: u32,
        out: &str,
        rows: u32,
        k: u32,
        n: u32,
        w: &str,
        b: Option<&str>,
    ) -> Result<u32, String> {
        self.dense_ep(x, out, rows, k, n, w, Epilogue { bias: b, ..Epilogue::NONE })
    }

    /// `act(x @ w^T + bias + residual)`: GEMV-shaped (weight streaming) up to [`GEMV_ROWS`] rows,
    /// the tiled conv above.
    #[allow(clippy::too_many_arguments)]
    fn dense_ep(
        &mut self,
        x: u32,
        out: &str,
        rows: u32,
        k: u32,
        n: u32,
        w: &str,
        mut ep: Epilogue<'_>,
    ) -> Result<u32, String> {
        let mut x = x;
        if rows <= GEMV_ROWS && (ep.ln.is_none() || k <= GEMV_LN_K) {
            let deps = self.deps();
            let o = p_handle(&self.p, out);
            let wh = self.p.declare(w, u64::from(n) * u64::from(k) * 2);
            let bh = ep.bias.map(|b| self.p.declare(b, u64::from(n) * 4));
            let ln = ep.ln.map(|(name, eps)| {
                let mut wd = |s: &str| self.p.declare(&format!("{name}.{s}"), u64::from(k) * 4);
                (wd("weight"), wd("bias"), eps)
            });
            let flags = 1 | 2 | if bh.is_some() { 4 } else { 0 } | if ep.residual.is_some() { 8 } else { 0 }
                | if ep.gelu { 1 << 4 } else { 0 }
                | if ln.is_some() { 1 << 8 } else { 0 };
            let none = packet::dev::TENSOR_NONE;
            let e = self.p.raw(DevOp::GemvF32Out, u64::from(self.n_cu.min(n / 8).max(1)), &deps, o, |i| {
                i.t[..7].copy_from_slice(&[
                    o,
                    x,
                    wh,
                    bh.unwrap_or(none),
                    ep.residual.unwrap_or(none),
                    ln.map_or(none, |l| l.0),
                    ln.map_or(none, |l| l.1),
                ]);
                i.i[..3].copy_from_slice(&[rows, n, k]);
                i.i[7] = flags;
                i.f[0] = ln.map_or(0.0, |l| l.2);
            })?;
            return Ok(self.done(e));
        }
        if let Some((name, eps)) = ep.ln.take() {
            x = self.layer_norm(x, "act.depth.h1", rows, k, name, eps)?;
        }
        let deps = self.deps();
        let e = self.p.conv1d_f32(
            x,
            false,
            &deps,
            Conv1dF32Stage {
                output: TensorRef::Named(out),
                weight: TensorRef::Named(w),
                bias: ep.bias.map(TensorRef::Named),
                alpha: None,
                residual: ep.residual,
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
                output_activation: if ep.gelu {
                    Activation::GeluErf
                } else {
                    Activation::None
                },
                slope: 0.0,
                weight_f16: true,
                split_bf16: false,
                weight_tap_major: false,
                wgmma: false,
                weight_split: false,
                dual_alpha: false,
            },
        )?;
        Ok(self.done(e))
    }

    fn layer_norm(&mut self, x: u32, out: &str, rows: u32, width: u32, w: &str, eps: f32) -> Result<u32, String> {
        let deps = self.deps();
        let e = self.p.layer_norm_f32(
            x,
            &deps,
            LayerNormRowsF32Stage {
                output: TensorRef::Named(out),
                gamma: Some(TensorRef::Named(&format!("{w}.weight"))),
                beta: Some(TensorRef::Named(&format!("{w}.bias"))),
                rows,
                width,
                epsilon: eps,
                activation: Activation::None,
                add_row: None,
            },
        )?;
        Ok(self.done(e))
    }

    /// `a += b` over `rows x width`; `b_row_stride` 0 broadcasts one `[width]` row.
    fn add(&mut self, a: u32, b: u32, rows: u32, width: u32, b_row_stride: u32) -> Result<(), String> {
        let deps = self.deps();
        let e = self.p.binary_f32(
            a,
            b,
            &deps,
            BinaryF32Stage {
                output: TensorRef::Handle(a),
                op: BinaryOp::Add,
                items: 1,
                rows,
                width,
                b_item_stride: 0,
                b_row_stride,
                b_col_stride: 1,
                scale: None,
            },
        )?;
        self.done(e);
        Ok(())
    }

    fn copy(&mut self, x: u32, out: &str, s: CopyColsF32Stage<'_>) -> Result<u32, String> {
        let deps = self.deps();
        let e = self.p.copy_cols_f32(
            x,
            &deps,
            CopyColsF32Stage {
                output: TensorRef::Named(out),
                ..s
            },
            None,
        )?;
        Ok(self.done(e))
    }

    /// `out` row `b * out_stride + col0` = `emb[index[b]]` (fp16 table), for `rows` streams.
    fn gather(&mut self, index: u32, out: &str, rows: u32, width: u32, vocab: u32, out_stride: u32, col0: u32) -> Result<u32, String> {
        let deps = self.deps();
        let e = self.p.gather_rows_f32(
            &deps,
            GatherRowsF32Stage {
                output: TensorRef::Named(out),
                table: TensorRef::Named("w.maya2.depth.emb.weight"),
                table_rows: vocab,
                index: Some(index),
                rows,
                width,
                vocab,
                rows_per_item: 1,
                repeat: 1,
                index_item_stride: 1,
                table_item_stride: 0,
                table_f16: true,
                accumulate: false,
                out_stride,
                out_col0: col0,
            },
        )?;
        Ok(self.done(e))
    }
}

fn program_for(prefix: PacketPrefix, geo: &Geo, batch: u32) -> Result<PacketPrefix, String> {
    let (dim, qkv_w, np) = (geo.dim, 3 * geo.dim, geo.positions());
    let n_cu = prefix.model.n_cu;
    let mut c = Chain {
        p: prefix.program(),
        last: None,
        n_cu,
    };
    let h = p_handle(&c.p, "in.depth.h");
    let x = p_handle(&c.p, "act.depth.x");
    let codes = p_handle(&c.p, "out.depth.codes");
    let p0 = c.dense(
        h,
        "act.depth.p0",
        batch,
        geo.hidden_in,
        dim,
        "w.maya2.depth.in_proj.weight",
        Some("w.maya2.depth.pos0"),
    )?;
    let emb_rows = geo.levels * geo.codes;
    for level in 1..=geo.levels {
        let g = geo.group_of_level[level as usize - 1];
        let prev = (level > 1).then(|| geo.group_of_level[level as usize - 2]);
        let next = geo.group_of_level.get(level as usize).copied();
        let full = prev != Some(g);
        let keep = next == Some(g);
        let p = if full { level + 1 } else { 1 };
        let rows = batch * p;
        if full {
            c.copy(
                p0,
                "act.depth.x",
                CopyColsF32Stage {
                    output: TensorRef::Named("act.depth.x"),
                    items: batch,
                    rows: 1,
                    cols: dim,
                    in_item_stride: dim,
                    in_stride: dim,
                    in_offset: 0,
                    out_item_stride: p * dim,
                    out_stride: dim,
                    out_offset: 0,
                },
            )?;
            for k in 1..=level {
                let idx = p_handle(&c.p, &format!("act.depth.eidx.{}", k - 1));
                c.gather(idx, "act.depth.x", batch, dim, emb_rows, p * dim, k * dim)?;
            }
        } else {
            let idx = p_handle(&c.p, &format!("act.depth.eidx.{}", level - 1));
            c.gather(idx, "act.depth.x", batch, dim, emb_rows, dim, 0)?;
        }
        for l in 0..geo.layers {
            let w = |s: &str| format!("w.maya2.depth.groups.{g}.{l}.{s}");
            let cache = format!("act.depth.cache.{l}");
            let n1 = w("n1");
            let pre = Epilogue {
                ln: Some((&n1, geo.eps)),
                ..Epilogue::NONE
            };
            let qkv = c.dense_ep(x, "act.depth.qkv", rows, dim, qkv_w, &w("qkv.weight"), pre)?;
            let (k, kv_rows, key_lengths) = if full {
                if keep {
                    c.copy(
                        qkv,
                        &cache,
                        CopyColsF32Stage {
                            output: TensorRef::Named(&cache),
                            items: batch,
                            rows: p,
                            cols: qkv_w,
                            in_item_stride: p * qkv_w,
                            in_stride: qkv_w,
                            in_offset: 0,
                            out_item_stride: np * qkv_w,
                            out_stride: qkv_w,
                            out_offset: 0,
                        },
                    )?;
                }
                (qkv, p, None)
            } else {
                let kc = c.copy(
                    qkv,
                    &cache,
                    CopyColsF32Stage {
                        output: TensorRef::Named(&cache),
                        items: batch,
                        rows: 1,
                        cols: qkv_w,
                        in_item_stride: qkv_w,
                        in_stride: qkv_w,
                        in_offset: 0,
                        out_item_stride: np * qkv_w,
                        out_stride: qkv_w,
                        out_offset: level * qkv_w,
                    },
                )?;
                let len = p_handle(&c.p, &format!("c.depth.len.{level}"));
                (kc, np, Some(len))
            };
            let deps = c.deps();
            let e = c.p.attention_f32(
                qkv,
                k,
                k,
                &deps,
                AttentionF32Stage {
                    output: TensorRef::Named("act.depth.at"),
                    key_lengths,
                    bias: None,
                    batch,
                    q_rows: p,
                    kv_rows,
                    heads: geo.heads,
                    head_width: geo.head_dim,
                    in_stride: qkv_w,
                    k_col0: dim,
                    v_col0: 2 * dim,
                    causal: full,
                    scale: 1.0 / (geo.head_dim as f32).sqrt(),
                    bias_head_stride: 0,
                    relative: false,
                    key_length_heads: 0,
                    prefix: None,
                    window: 0,
                    stream: None,
                },
            )?;
            let at = c.done(e);
            // x += o(attn); x += f2(gelu(f1(LN(x)))) + cross_l (the one-token cross-attention).
            let res = Epilogue {
                residual: Some(x),
                ..Epilogue::NONE
            };
            c.dense_ep(at, "act.depth.x", rows, dim, dim, &w("o.weight"), res)?;
            let n2 = w("n2");
            let gelu = Epilogue {
                gelu: true,
                ln: Some((&n2, geo.eps)),
                ..Epilogue::NONE
            };
            let f = c.dense_ep(x, "act.depth.f", rows, dim, geo.inter, &w("f1.weight"), gelu)?;
            let cross = format!("w.maya2.depth.cross.{l}");
            let out = Epilogue {
                bias: Some(&cross),
                ..res
            };
            c.dense_ep(f, "act.depth.x", rows, geo.inter, dim, &w("f2.weight"), out)?;
        }
        let last = if p > 1 {
            c.copy(
                x,
                "act.depth.y",
                CopyColsF32Stage {
                    output: TensorRef::Named("act.depth.y"),
                    items: batch,
                    rows: 1,
                    cols: dim,
                    in_item_stride: p * dim,
                    in_stride: dim,
                    in_offset: (p - 1) * dim,
                    out_item_stride: dim,
                    out_stride: dim,
                    out_offset: 0,
                },
            )?
        } else {
            x
        };
        let final_norm = Epilogue {
            ln: Some(("w.maya2.depth.norm", geo.eps)),
            ..Epilogue::NONE
        };
        let logits = c.dense_ep(
            last,
            "act.depth.logits",
            batch,
            dim,
            geo.codes,
            &format!("w.maya2.depth.heads.{}.weight", level - 1),
            final_norm,
        )?;
        c.copy(
            logits,
            "out.depth.logits",
            CopyColsF32Stage {
                output: TensorRef::Named("out.depth.logits"),
                items: batch,
                rows: 1,
                cols: geo.codes,
                in_item_stride: geo.codes,
                in_stride: geo.codes,
                in_offset: 0,
                out_item_stride: geo.levels * geo.codes,
                out_stride: geo.codes,
                out_offset: (level - 1) * geo.codes,
            },
        )?;
        let handles = [
            "in.depth.params",
            "in.depth.ctl",
            "in.depth.seed",
            "in.depth.force",
        ]
        .map(|n| p_handle(&c.p, n));
        let eidx = p_handle(&c.p, &format!("act.depth.eidx.{level}"));
        let deps = c.deps();
        let e = c.p.raw(DevOp::SampleSmallF32, u64::from(batch), &deps, codes, |i| {
            i.t = [
                codes,
                logits,
                handles[0],
                handles[1],
                handles[2],
                packet::dev::TENSOR_NONE,
                handles[3],
                eidx,
            ];
            i.i = [batch, geo.codes, geo.levels + 1, level, level, 0, 0, 0];
            i.j[0] = level * geo.codes;
        })?;
        c.done(e);
    }
    Ok(c.p.finish(batch))
}

fn section(
    prefix: &PacketPrefix,
    geo: &Geo,
    roles: BTreeMap<String, u32>,
    bmax: u32,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{
        PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION,
    };
    let b = u64::from(bmax);
    let cb = u64::from(geo.levels + 1);
    let tensor = |name: &str, dtype, shape: Vec<u64>| {
        (
            name.to_string(),
            PipelineTensor {
                name: name.to_string(),
                dtype,
                shape,
            },
        )
    };
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            name: PIPELINE.into(),
            driver: DRIVER.into(),
            programs: roles,
            tensors: BTreeMap::from([
                tensor("in.depth.h", PipelineDType::F32, vec![b, u64::from(geo.hidden_in)]),
                tensor("act.depth.eidx.0", PipelineDType::U32, vec![b]),
                tensor("in.depth.force", PipelineDType::U32, vec![b, cb]),
                tensor("in.depth.params", PipelineDType::F32, vec![b, 4]),
                tensor("in.depth.ctl", PipelineDType::U32, vec![b, 2]),
                tensor("in.depth.seed", PipelineDType::U32, vec![b * 2]),
                tensor("out.depth.codes", PipelineDType::U32, vec![b, cb]),
                tensor(
                    "out.depth.logits",
                    PipelineDType::F32,
                    vec![b, u64::from(geo.levels), u64::from(geo.codes)],
                ),
            ]),
            parameters: BTreeMap::from([
                ("depth.levels".into(), u64::from(geo.levels)),
                ("codec.frame_codes".into(), cb),
                ("codec.codebook".into(), u64::from(geo.codes)),
                ("backbone.hidden".into(), u64::from(geo.hidden_in)),
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
