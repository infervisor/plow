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

pub(crate) struct Geo {
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
    /// An LM packet's DepthV2 (config `maya2` section, read into [`crate::config::FrameEmbed`]).
    pub(crate) fn of(c: &crate::config::Cfg) -> Self {
        let f = c.frame.as_ref().expect("maya2 frame embedding");
        let d = f.depth.as_ref().expect("maya2 depth");
        let m = d.maya2.as_ref().expect("maya2 depth");
        Geo {
            hidden_in: c.hidden,
            dim: d.hidden,
            heads: d.heads,
            head_dim: d.head_dim,
            inter: d.inter,
            layers: d.layers,
            levels: m.levels,
            codes: f.cb_size,
            eps: d.eps,
            group_of_level: m.group_of_level.clone(),
        }
    }

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
    let b64 = u64::from(bmax);
    let cb = u64::from(geo.levels + 1);
    let mut builder = packet::devbuild::Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let h = builder.tensor("in.depth.h", b64 * u64::from(geo.hidden_in) * 4);
    declare_io(&mut builder, &geo, bmax);
    let codes = builder.tensor("out.depth.codes", b64 * cb * 4);
    builder.tensor(
        "out.depth.logits",
        b64 * u64::from(geo.levels * geo.codes) * 4,
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
        input: h,
        output: codes,
        input_shape: vec![b64, u64::from(geo.hidden_in)],
    };
    let mut roles = BTreeMap::new();
    for &batch in CAPACITIES {
        let program = prefix.model.progs.len();
        prefix = program_for(prefix, &geo, batch, &Io::Standalone)?;
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
/// Projections over at least this many rows run on tensor cores (`GemvF32Out` bit 10: f16 x, W
/// streamed once per 64 rows); the f32-x GEMV re-streams W every 8 rows.
const MMA_ROWS: u32 = 1;

/// What a projection adds after the matmul: `act(acc + bias + residual)`.
#[derive(Clone, Copy)]
struct Epilogue<'a> {
    bias: Option<&'a str>,
    residual: Option<u32>,
    gelu: bool,
    /// LayerNorm (`{ln}.weight`, `{ln}.bias`, eps) of each x row before the matmul: fused into
    /// the GEMV's staged rows, a separate pass before the conv.
    ln: Option<(&'a str, f32)>,
    /// x is bf16 (the LM's gathered final hidden), not f32.
    x_bf16: bool,
    /// The bias has a row per this many rows (per stream: the slot's depth memory); 0 broadcasts.
    bias_group: u32,
}

impl Epilogue<'_> {
    const NONE: Epilogue<'static> = Epilogue {
        bias: None,
        residual: None,
        gelu: false,
        ln: None,
        x_bf16: false,
        bias_group: 0,
    };
}

/// An op chain inside one program: each op depends on the previous.
struct Chain {
    p: StageProgram,
    last: Option<u32>,
    n_cu: u32,
    /// Weight-name prefix.
    w: &'static str,
}

/// Where a depth program reads `h` and writes its codes.
pub(crate) enum Io {
    /// `in.depth.h` f32 rows in, `out.depth.codes` and every level's logits out (the gates).
    Standalone,
    /// Inside an LM packet: row `r` gathers its bf16 final-norm hidden from `hn` (of `hidden_rows`
    /// rows) at `in.depth.hrow[r]`, takes the host-sampled cb0 from `act.depth.eidx.0`, and writes
    /// all codes (cb0 in column 0) to `codes[r]` (the next decode step's frame input).
    Lm { hn: u32, hidden_rows: u32, codes: u32 },
}

impl Io {
    fn weights(&self) -> &'static str {
        match self {
            Io::Standalone => "w.maya2.depth",
            Io::Lm { .. } => "maya2.depth",
        }
    }
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
        if ep.x_bf16 || ep.bias_group > 0 || (rows <= GEMV_ROWS && (ep.ln.is_none() || k <= GEMV_LN_K)) {
            let deps = self.deps();
            let o = p_handle(&self.p, out);
            let wh = self.p.declare(w, u64::from(n) * u64::from(k) * 2);
            let bh = ep.bias.map(|b| self.p.declare(b, u64::from(n) * 4));
            let ln = ep.ln.map(|(name, eps)| {
                let mut wd = |s: &str| self.p.declare(&format!("{name}.{s}"), u64::from(k) * 4);
                (wd("weight"), wd("bias"), eps)
            });
            let flags = u32::from(!ep.x_bf16) | 2 | if bh.is_some() { 4 } else { 0 } | if ep.residual.is_some() { 8 } else { 0 }
                | if ep.gelu { 1 << 4 } else { 0 }
                | if ln.is_some() { 1 << 8 } else { 0 }
                | if ep.bias_group > 0 { 1 << 9 } else { 0 }
                | if !ep.x_bf16 && rows >= MMA_ROWS && k % 64 == 0 { 1 << 10 } else { 0 };
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
                i.i[..4].copy_from_slice(&[rows, n, k, ep.bias_group]);
                i.i[7] = flags;
                i.f[0] = ln.map_or(0.0, |l| l.2);
            })?;
            return Ok(self.done(e));
        }
        assert!(ep.bias_group == 0, "per-stream bias needs the GEMV path");
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
                table: TensorRef::Named(&format!("{}.emb.weight", self.w)),
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

fn program_for(prefix: PacketPrefix, geo: &Geo, batch: u32, io: &Io) -> Result<PacketPrefix, String> {
    let (dim, qkv_w, np) = (geo.dim, 3 * geo.dim, geo.positions());
    let n_cu = prefix.model.n_cu;
    let wp = io.weights();
    let mut c = Chain {
        p: prefix.program(),
        last: None,
        n_cu,
        w: wp,
    };
    let x = p_handle(&c.p, "act.depth.x");
    let (h, codes, x_bf16) = match *io {
        Io::Standalone => (p_handle(&c.p, "in.depth.h"), p_handle(&c.p, "out.depth.codes"), false),
        Io::Lm { hn, hidden_rows, codes } => {
            let hx = c.p.declare("act.depth.hx", u64::from(batch * geo.hidden_in) * 2);
            let hrow = p_handle(&c.p, "in.depth.hrow");
            let e = c.p.raw(DevOp::RowGather, u64::from(batch.min(n_cu)), &[], hx, |i| {
                i.t[..3].copy_from_slice(&[hx, hn, hrow]);
                i.i[..3].copy_from_slice(&[batch, geo.hidden_in, hidden_rows]);
            })?;
            c.done(e);
            let cb0 = p_handle(&c.p, "act.depth.eidx.0");
            let deps = c.deps();
            let e = c.p.copy_cols_f32(
                cb0,
                &deps,
                CopyColsF32Stage {
                    output: TensorRef::Handle(codes),
                    items: batch,
                    rows: 1,
                    cols: 1,
                    in_item_stride: 1,
                    in_stride: 1,
                    in_offset: 0,
                    out_item_stride: geo.levels + 1,
                    out_stride: geo.levels + 1,
                    out_offset: 0,
                },
                None,
            )?;
            c.done(e);
            (hx, codes, true)
        }
    };
    let pos0 = format!("{wp}.pos0");
    let p0 = c.dense_ep(
        h,
        "act.depth.p0",
        batch,
        geo.hidden_in,
        dim,
        &format!("{wp}.in_proj.weight"),
        Epilogue {
            bias: Some(&pos0),
            x_bf16,
            ..Epilogue::NONE
        },
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
            let w = |s: &str| format!("{wp}.groups.{g}.{l}.{s}");
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
            // Inside an LM packet each slot has its own depth memory (the speaker's, or the null one).
            let (cross, bias_group) = match io {
                Io::Standalone => (format!("{wp}.cross.{l}"), 0),
                Io::Lm { .. } => (format!("in.depth.cross.{l}"), p),
            };
            let out = Epilogue {
                bias: Some(&cross),
                bias_group,
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
        let norm = format!("{wp}.norm");
        let final_norm = Epilogue {
            ln: Some((&norm, geo.eps)),
            ..Epilogue::NONE
        };
        let logits = c.dense_ep(
            last,
            "act.depth.logits",
            batch,
            dim,
            geo.codes,
            &format!("{wp}.heads.{}.weight", level - 1),
            final_norm,
        )?;
        if matches!(io, Io::Standalone) {
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
        }
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

/// Host-staged per-row tensors and scratch of the depth programs, sized for `bmax` rows.
fn declare_io(b: &mut packet::devbuild::Builder, geo: &Geo, bmax: u32) {
    let (b64, d, np) = (u64::from(bmax), u64::from(geo.dim), u64::from(geo.positions()));
    let cb = u64::from(geo.levels + 1);
    for k in 0..geo.positions() {
        b.tensor(&format!("act.depth.eidx.{k}"), b64 * 4);
    }
    b.tensor_init("in.depth.force", vec![0xFF; (b64 * cb * 4) as usize]);
    b.tensor("in.depth.params", b64 * 4 * 4);
    b.tensor("in.depth.ctl", b64 * 2 * 4);
    b.tensor("in.depth.seed", b64 * 8);
    for (name, width) in [
        ("act.depth.p0", d),
        ("act.depth.y", d),
        ("act.depth.yn", d),
        ("act.depth.logits", u64::from(geo.codes)),
    ] {
        b.tensor(name, b64 * width * 4);
    }
    let qkv_w = 3 * d;
    for (name, width) in [
        ("act.depth.x", d),
        ("act.depth.h1", d),
        ("act.depth.at", d),
        ("act.depth.qkv", qkv_w),
        ("act.depth.f", u64::from(geo.inter)),
    ] {
        b.tensor(name, b64 * np * width * 4);
    }
    for l in 0..geo.layers {
        b.tensor(&format!("act.depth.cache.{l}"), b64 * np * qkv_w * 4);
    }
    for level in 1..=geo.levels {
        b.tensor_init(
            &format!("c.depth.len.{level}"),
            (0..bmax).flat_map(|_| (level + 1).to_le_bytes()).collect(),
        );
    }
}

/// DepthV2 as an LM packet's depth stage: one program per decode rung (row `r` = slot `r`), appended
/// to `m`'s tensors, as the `fish_depth` sections of [`plow_asset::fish_depth::Sampler::Maya2`].
pub(crate) fn lm_sections(
    m: &mut packet::devbuild::Model,
    c: &crate::config::Cfg,
    hn: u32,
    codes: u32,
    prompt_codes: u32,
    rungs: &[u32],
) -> (Vec<packet::devbuild::SectionData>, u64) {
    use plow_asset::fish_depth as manifest;
    let geo = Geo::of(c);
    let d = c
        .frame
        .as_ref()
        .and_then(|f| f.depth.as_ref())
        .and_then(|d| d.maya2.as_ref())
        .unwrap();
    let bmax = rungs.iter().copied().max().expect("decode rungs");
    let hidden_rows = (m.tensors[hn as usize].bytes / (u64::from(c.hidden) * 2)) as u32;
    let mut b = packet::devbuild::Builder::new(m.n_cu);
    b.set_tensor_dedup(true);
    b.adopt_tensors(std::mem::take(&mut m.tensors));
    declare_io(&mut b, &geo, bmax);
    b.tensor("in.depth.hrow", u64::from(bmax) * 4);
    for l in 0..geo.layers {
        b.tensor(&format!("in.depth.cross.{l}"), u64::from(bmax * geo.dim) * 4);
        // The null speaker's memory: the runtime copies it into a slot's row at bind.
        b.tensor(&format!("maya2.depth.cross.{l}"), u64::from(geo.dim) * 4);
    }
    let mut prefix = PacketPrefix {
        model: packet::devbuild::Model {
            n_cu: m.n_cu,
            target: m.target,
            tensors: b.tensors(),
            progs: Vec::new(),
            kv_row_insts: Vec::new(),
            prog_t: Vec::new(),
            gen: Vec::new(),
        },
        programs: Vec::new(),
        input: hn,
        output: codes,
        input_shape: Vec::new(),
    };
    let io = Io::Lm {
        hn,
        hidden_rows,
        codes,
    };
    for &rows in rungs {
        prefix =
            program_for(prefix, &geo, rows, &io).unwrap_or_else(|e| panic!("maya2 depth: {e}"));
    }
    m.tensors = std::mem::take(&mut prefix.model.tensors);
    let handle = |name: &str| {
        let h = m.tensors.iter().position(|t| t.name == name).expect(name);
        u16::try_from(h).expect("maya2 depth: tensor handle")
    };
    let ctl = handle("in.depth.ctl");
    let man = manifest::Manifest {
        version: manifest::VERSION,
        rows: rungs.to_vec(),
        codebooks: geo.levels + 1,
        semantic_base: c.frame.as_ref().unwrap().lo,
        semantic_tokens: geo.codes,
        stop_class: geo.codes,
        ras_window: 0,
        sampler: manifest::Sampler::Maya2,
        pair_cfg: false,
        depth_sampling: Some([d.temperature.to_bits(), d.top_k]),
        slot_memory: (0..geo.layers)
            .map(|l| handle(&format!("in.depth.cross.{l}")))
            .collect(),
        tensors: manifest::Tensors {
            params: handle("in.depth.params"),
            top_k: ctl,
            seed: handle("in.depth.seed"),
            step: ctl,
            history: handle("act.depth.eidx.0"),
            hidden_row: handle("in.depth.hrow"),
            frame_codes: u16::try_from(codes).unwrap(),
            prompt_codes: u16::try_from(prompt_codes).unwrap(),
            force: Some(handle("in.depth.force")),
        },
    };
    let bytes = crate::fish_depth::encode(m.n_cu, m.tensors.len(), &prefix.model.progs, rungs);
    eprintln!("  maya2 depth: {} programs (rows {rungs:?})", rungs.len());
    (
        crate::fish_depth::manifest_sections(&man, rungs.len(), m.tensors.len(), bytes),
        crate::manifest::speech_ops(&prefix.model),
    )
}
