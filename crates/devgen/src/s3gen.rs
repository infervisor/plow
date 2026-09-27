//! Chatterbox S3Gen (speech tokens -> 24 kHz PCM) lowered to `s3gen.pkt` from generic ops.
//!
//! Token embedding + UpsampleConformerEncoder (ESPnet relative-position attention) + encoder_proj;
//! CausalConditionalCFM (Euler on the cosine schedule, classifier-free guidance as a doubled
//! batch) with the ConditionalDecoder estimator; HiFTGenerator (F0 predictor, NSF sine source,
//! STFT, transposed convolutions with snake ResBlocks, iSTFT) and the trim fade. Per (batch,
//! tokens) capacity the synthesis is a program sequence: encoder, one program per Euler step,
//! vocoder. Voices (prompt tokens, prompt mel, speaker vector) are packet tensors selected per item
//! on device; per-item valid lengths at every time resolution make each item synthesize exactly
//! as it would alone.
//!
//! Layout notes. The conformer runs time-major (`[t][item][c]`) so relative-position attention is
//! one batch-1 AttentionF32 with the items folded into the heads: its additive bias
//! `[item*heads][t][t]` is `(q + u)·P[t - j] / 8 + (v - u)·P[t - j] / 8` (a grouped 1x1
//! convolution against the capacity's projected position table, then a diagonal copy), plus a
//! per-item key mask.
//!
//! Fusions (each convolution's and linear's input activation, output activation and residual)
//! come from the rewrite's fused sites over the S3Gen graph (`nn_graph` `chatterbox_s3gen`),
//! subject to [`ConvFusions`]' cost model; without sites they are the hand choices below.

use std::collections::{BTreeMap, HashMap};

use crate::pipeline::{
    Activation, AttentionF32Stage, BinaryF32Stage, BinaryOp, Conv1dF32Stage, CopyColsF32Stage, CumSumF64Stage,
    Emitted, GatherRowsF32Stage, LayerNormRowsF32Stage, PacketPrefix, PadMode, RandCoord, RandF32Stage,
    StageProgram, TensorRef, UnaryF32Stage,
};
use crate::rewrite_lower::{ActAt, ConvFusions};
use crate::RewriteSites;

pub const PIPELINE: &str = "vocoder.synth";
pub const PACKET: &str = "s3gen.pkt";

/// (batch, speech tokens) capacities. The encoder's attention bias is `[B*8][T][T]` f32 with
/// T = 2 (prompt + tokens), so large batches get short buckets.
pub const S3GEN_CAPACITIES: &[(u32, u32)] = &[
    (1, 64), (1, 128), (1, 256), (1, 512), (1, 1000),
    (2, 128), (2, 256), (2, 512),
    (4, 128), (4, 256),
    (8, 64), (8, 128), (8, 256),
];

const D_ENC: u32 = 512;
const D_CFM: u32 = 256;
const HEADS: u32 = 8;
const MEL: u32 = 80;
const ENC_LAYERS: (u32, u32) = (6, 4);
const TBLOCKS: u32 = 4;
const MID_BLOCKS: u32 = 12;
const F0_CONVS: u32 = 5;

struct Config {
    sampling_rate: u32,
    vocab: u32,
    voices: Vec<String>,
    prompt: u32,
    max_tokens: u32,
    relpos_rows: [u32; 2],
    t_span: Vec<f32>,
    cfg_rate: f32,
    upsample: Vec<[u32; 3]>,
    source_downs: Vec<[u32; 3]>,
    source_resblocks: Vec<(u32, Vec<u32>)>,
    resblocks: Vec<(u32, Vec<u32>)>,
    n_fft: u32,
    hop: u32,
    harmonics: u32,
    sine_amp: f32,
    noise_std: f32,
    voiced_threshold: f32,
    trim: u32,
    audio_limit: f32,
}

impl Config {
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let u = |v: &serde_json::Value, k: &str| -> Result<u32, String> {
            v[k].as_u64().and_then(|x| u32::try_from(x).ok()).ok_or(format!("s3gen config: {k}"))
        };
        let f = |k: &str| -> Result<f32, String> { v[k].as_f64().map(|x| x as f32).ok_or(format!("s3gen config: {k}")) };
        let list = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect()).unwrap_or_default()
        };
        let triples = |k: &str| -> Result<Vec<[u32; 3]>, String> {
            v[k].as_array()
                .ok_or(format!("s3gen config: {k}"))?
                .iter()
                .map(|x| <[u32; 3]>::try_from(list(x)).map_err(|_| format!("s3gen config: {k}")))
                .collect()
        };
        let blocks = |k: &str| -> Result<Vec<(u32, Vec<u32>)>, String> {
            v[k].as_array()
                .ok_or(format!("s3gen config: {k}"))?
                .iter()
                .map(|x| Ok((x[0].as_u64().ok_or(format!("s3gen config: {k}"))? as u32, list(&x[1]))))
                .collect()
        };
        let rows = list(&v["relpos_rows"]);
        Ok(Self {
            sampling_rate: u(v, "sampling_rate")?,
            vocab: u(v, "vocab")?,
            voices: v["voices"].as_array().ok_or("s3gen config: voices")?.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
            prompt: u(v, "prompt_tokens")?,
            max_tokens: u(v, "max_tokens")?,
            relpos_rows: <[u32; 2]>::try_from(rows).map_err(|_| "s3gen config: relpos_rows")?,
            t_span: v["t_span"].as_array().ok_or("s3gen config: t_span")?.iter().filter_map(|x| x.as_f64().map(|x| x as f32)).collect(),
            cfg_rate: f("cfg_rate")?,
            upsample: triples("upsample")?,
            source_downs: triples("source_downs")?,
            source_resblocks: blocks("source_resblocks")?,
            resblocks: blocks("resblocks")?,
            n_fft: u(v, "n_fft")?,
            hop: u(v, "hop")?,
            harmonics: u(v, "harmonics")?,
            sine_amp: f("sine_amp")?,
            noise_std: f("noise_std")?,
            voiced_threshold: f("voiced_threshold")?,
            trim: u(v, "trim")?,
            audio_limit: f("audio_limit")?,
        })
    }

    /// Output samples per mel frame.
    fn frame_samples(&self) -> u32 {
        self.upsample.iter().map(|u| u[0]).product::<u32>() * self.hop
    }
}

/// Per-item valid rows at each time resolution (`tokens * rows_per_token + offset`), derived on
/// device from the host's token counts.
#[derive(Clone, Copy)]
struct Lengths {
    tok: u32,
    mel: u32,
    gen: u32,
    /// HiFT stage outputs (the last is the STFT frame count).
    stage: [u32; 3],
    wav: u32,
}

impl Lengths {
    fn all(&self) -> [u32; 7] {
        [self.tok, self.mel, self.gen, self.stage[0], self.stage[1], self.stage[2], self.wav]
    }
}

/// `(rows_per_token, offset)` of [`Lengths::all`].
fn length_scales(cfg: &Config) -> [(u32, u32); 7] {
    let p = cfg.prompt;
    let (u0, u1, u2) = (cfg.upsample[0][0], cfg.upsample[1][0], cfg.upsample[2][0]);
    // Two mel frames per token; the last HiFT stage is reflection-padded by one frame.
    [(1, p), (2, 2 * p), (2, 0), (2 * u0, 0), (2 * u0 * u1, 0), (2 * u0 * u1 * u2, 1), (2 * cfg.frame_samples(), 0)]
}

/// `s3gen.pkt` for the S3Gen export in `dir` (scripts/tts/s3gen_export.py), with the rewrite's
/// fused sites of that export's graph when plowc extracted them.
pub fn lower_s3gen(dir: &std::path::Path, n_cu: u32, target: u32, sites: Option<&RewriteSites>) -> Result<Vec<u8>, String> {
    let cfg = Config::parse(
        &serde_json::from_slice(&std::fs::read(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.display()))?)
            .map_err(|e| format!("s3gen config: {e}"))?,
    )?;
    if cfg.upsample.len() != 3
        || cfg.source_downs.len() != 3
        || cfg.source_resblocks.len() != 3
        || cfg.resblocks.len() != 9
        || cfg.t_span.len() < 2
        || cfg.voices.is_empty()
        || cfg.n_fft != 4 * cfg.hop
        || cfg.harmonics == 0
    {
        return Err("unsupported S3Gen export".into());
    }
    let caps: Vec<(u32, u32)> = S3GEN_CAPACITIES.iter().copied().filter(|&(_, n)| n <= cfg.max_tokens).collect();
    let bmax = caps.iter().map(|c| c.0).max().ok_or("no S3Gen capacity fits max_tokens")?;
    let nmax = caps.iter().map(|c| c.1).max().unwrap_or(1);
    let mut model = packet::devbuild::Model { n_cu, target, tensors: Vec::new(), progs: Vec::new(), kv_row_insts: Vec::new(), prog_t: Vec::new(), gen: Vec::new() };
    let mut builder = packet::devbuild::Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let inputs = Inputs {
        tokens: builder.tensor("in.s3gen.tokens", u64::from(bmax * nmax) * 4),
        voice: builder.tensor("in.s3gen.voice", u64::from(bmax) * 4),
        seed: builder.tensor("in.s3gen.seed", u64::from(bmax) * 8),
        count: builder.tensor("in.s3gen.lengths.0", u64::from(bmax) * 4),
        nmax,
        lengths: {
            let h: Vec<u32> = (0..7).map(|k| builder.tensor(&format!("act.s3gen.len.{k}"), u64::from(bmax) * 4)).collect();
            Lengths { tok: h[0], mel: h[1], gen: h[2], stage: [h[3], h[4], h[5]], wav: h[6] }
        },
    };
    let samples_per_token = length_scales(&cfg)[6].0;
    builder.tensor("act.s3gen.pcm", u64::from(bmax * nmax * samples_per_token) * 4);
    model.tensors = builder.tensors();
    let mut prefix = PacketPrefix { model, programs: Vec::new(), input: inputs.tokens, output: inputs.tokens, input_shape: vec![u64::from(bmax * nmax)] };
    let mut names = HashMap::new();
    let mut roles = BTreeMap::new();
    for &(batch, tokens) in &caps {
        let lo = Lowering { cfg: &cfg, inputs, b: batch, n: tokens };
        let mut stage = 0;
        let mut run = |prefix: PacketPrefix,
                       names: &mut HashMap<String, u32>,
                       body: &dyn Fn(&mut Ops) -> Result<(), String>|
         -> Result<PacketPrefix, String> {
            let program = prefix.model.progs.len() as u32;
            let mut ops = Ops::new(prefix.program(), std::mem::take(names), ConvFusions(sites));
            body(&mut ops)?;
            let (prefix, n) = ops.finish(batch * tokens);
            *names = n;
            roles.insert(format!("synth.b{batch}.t{tokens}.{stage}"), program);
            stage += 1;
            Ok(prefix)
        };
        prefix = run(prefix, &mut names, &|o| lo.encoder(o))?;
        for step in 0..cfg.t_span.len() - 1 {
            prefix = run(prefix, &mut names, &|o| lo.cfm_step(o, step))?;
        }
        prefix = run(prefix, &mut names, &|o| lo.vocoder(o))?;
    }
    let reader = crate::checkpoint::TensorReader::open(dir)?;
    let mut derived = Derived { reader: &reader, cfg: &cfg, nmax, cache: HashMap::new() };
    for tensor in &mut prefix.model.tensors {
        if let Some(name) = tensor.name.strip_prefix("w.") {
            let bytes = derived.bytes(name)?;
            if bytes.len() as u64 != tensor.bytes {
                return Err(format!("{name}: {} bytes, expected {}", bytes.len(), tensor.bytes));
            }
            tensor.init = Some(bytes);
        }
    }
    let section = pipeline_section(&prefix, &cfg, roles, &inputs, bmax, nmax, &names)?;
    Ok(prefix.model.to_blob_v6(&[section]))
}

#[derive(Clone, Copy)]
struct Inputs {
    tokens: u32,
    voice: u32,
    seed: u32,
    /// Speech tokens per item (host).
    count: u32,
    nmax: u32,
    lengths: Lengths,
}

/// A [`StageProgram`] with read/write hazard tracking: every op depends on the last writer of
/// each tensor it reads and on the last writer and every later reader of the tensor it writes.
struct Ops<'s> {
    p: StageProgram,
    names: HashMap<String, u32>,
    writer: HashMap<u32, u32>,
    readers: HashMap<u32, Vec<u32>>,
    fuse: ConvFusions<'s>,
}

impl<'s> Ops<'s> {
    fn new(p: StageProgram, names: HashMap<String, u32>, fuse: ConvFusions<'s>) -> Self {
        Self { p, names, writer: HashMap::new(), readers: HashMap::new(), fuse }
    }

    fn finish(self, tag: u32) -> (PacketPrefix, HashMap<String, u32>) {
        (self.p.finish(tag), self.names)
    }

    fn op(
        &mut self,
        reads: &[u32],
        out: &str,
        emit: impl FnOnce(&mut StageProgram, TensorRef<'_>, &[u32]) -> Result<Emitted, String>,
    ) -> Result<u32, String> {
        let target = self.names.get(out).copied();
        let mut deps: Vec<u32> = reads.iter().filter_map(|r| self.writer.get(r).copied()).collect();
        if let Some(t) = target {
            deps.extend(self.writer.get(&t).copied());
            deps.extend(self.readers.get(&t).into_iter().flatten().copied());
        }
        deps.sort_unstable();
        deps.dedup();
        let e = emit(&mut self.p, TensorRef::Named(out), &deps).map_err(|err| format!("{out}: {err}"))?;
        self.names.insert(out.to_string(), e.output);
        for r in reads {
            self.readers.entry(*r).or_default().push(e.done);
        }
        self.writer.insert(e.output, e.done);
        self.readers.insert(e.output, Vec::new());
        Ok(e.output)
    }

    fn name_of(&self, h: u32) -> String {
        self.names.iter().find(|(_, &v)| v == h).map(|(k, _)| k.clone()).expect("tensor declared by name")
    }

    /// `c` with the fusions [`Self::fuse`] decides; an unfused piece runs as its own `UnaryF32` /
    /// `BinaryF32` pass (an input activation in place when `c.input_is_temp`, else into
    /// `act.s3gen.pre`).
    fn conv(&mut self, x: u32, out: &str, c: Conv<'_>) -> Result<u32, String> {
        let weight = format!("w.{}.w", c.w);
        let act_in = c.act_in != Activation::None;
        let fuse_in = act_in && self.fuse.input_act(&weight, c.k, true);
        let fuse_out = c.act_out != Activation::None && self.fuse.output_act(&weight, true);
        let fuse_res = c.residual.is_some() && self.fuse.residual(&weight, true);
        let mut raw = c;
        let mut x = x;
        if act_in && !fuse_in {
            let (rows, param) = (c.batch * c.rows, c.alpha.map(|a| format!("w.{a}")));
            let target = if c.input_is_temp { self.name_of(x) } else { "act.s3gen.pre".to_string() };
            x = self.op(&[x], &target, |p, output, deps| {
                p.unary_f32(x, deps, UnaryF32Stage {
                    output,
                    param: param.as_deref().map(TensorRef::Named),
                    rows,
                    width: c.cin,
                    stride: 0,
                    col0: 0,
                    kind: c.act_in,
                    p0: c.slope,
                    p1: 0.0,
                })
            })?;
            (raw.act_in, raw.alpha) = (Activation::None, None);
        }
        if !fuse_out {
            raw.act_out = Activation::None;
        }
        if !fuse_res {
            raw.residual = None;
        }
        let y = self.conv_op(x, out, raw)?;
        let out_rows = if c.transpose {
            (c.rows - 1) * c.stride + c.k + c.dil - c.pad.0 - c.pad.1
        } else {
            (c.rows + c.pad.0 + c.pad.1 - c.dil * (c.k - 1) - 1) / c.stride + 1
        };
        if c.act_out != Activation::None && !fuse_out {
            self.unary(y, c.batch * out_rows, c.cout, 0, 0, c.act_out, c.slope, 0.0)?;
        }
        if let Some(r) = c.residual.filter(|_| !fuse_res) {
            self.binary(y, r, BinaryOp::Add, (1, c.batch * out_rows, c.cout), (0, c.cout, 1), None)?;
        }
        Ok(y)
    }

    fn conv_op(&mut self, x: u32, out: &str, c: Conv<'_>) -> Result<u32, String> {
        let mut reads = vec![x];
        reads.extend(c.residual);
        reads.extend(c.lengths);
        let weight = format!("w.{}.w", c.w);
        let bias = c.bias.then(|| c.bias_name.map_or_else(|| format!("w.{}.b", c.w), |b| format!("w.{b}")));
        let alpha = c.alpha.map(|a| format!("w.{a}"));
        self.op(&reads, out, |p, output, deps| {
            p.conv1d_f32(x, c.transpose, deps, Conv1dF32Stage {
                output,
                weight: TensorRef::Named(&weight),
                bias: bias.as_deref().map(TensorRef::Named),
                alpha: alpha.as_deref().map(TensorRef::Named),
                residual: c.residual,
                lengths: c.lengths,
                batch: c.batch,
                in_rows: c.rows,
                in_channels: c.cin,
                out_channels: c.cout,
                kernel: c.k,
                stride: c.stride,
                dilation_or_output_padding: c.dil,
                groups: c.groups,
                pad_before: c.pad.0,
                pad_after: c.pad.1,
                pad_mode: c.mode,
                input_activation: c.act_in,
                output_activation: c.act_out,
                slope: c.slope,
                weight_f16: false,
            })
        })
    }

    /// Row-wise linear layer `out = x W^T (+ b) (+ residual)` over `rows` rows.
    fn linear(&mut self, x: u32, out: &str, w: &str, rows: u32, cin: u32, cout: u32, bias: bool, act: Activation, residual: Option<u32>) -> Result<u32, String> {
        let mut c = Conv::new(w, 1, rows, cin, cout, 1);
        c.bias = bias;
        c.act_out = act;
        c.residual = residual;
        self.conv(x, out, c)
    }

    fn layer_norm(&mut self, x: u32, out: &str, w: &str, rows: u32, width: u32, epsilon: f32) -> Result<u32, String> {
        let (g, b) = (format!("w.{w}.g"), format!("w.{w}.b"));
        self.op(&[x], out, |p, output, deps| {
            p.layer_norm_f32(x, deps, LayerNormRowsF32Stage {
                output,
                gamma: Some(TensorRef::Named(&g)),
                beta: Some(TensorRef::Named(&b)),
                rows,
                width,
                epsilon,
            })
        })
    }

    /// In-place `x = f(x)` on `rows x width` at row stride `stride` from `col0`.
    fn unary(&mut self, x: u32, rows: u32, width: u32, stride: u32, col0: u32, kind: Activation, p0: f32, p1: f32) -> Result<u32, String> {
        let out = self.name_of(x);
        self.unary_to(x, &out, rows, width, stride, col0, kind, p0, p1)
    }

    fn unary_to(&mut self, x: u32, out: &str, rows: u32, width: u32, stride: u32, col0: u32, kind: Activation, p0: f32, p1: f32) -> Result<u32, String> {
        self.op(&[x], out, |p, output, deps| {
            p.unary_f32(x, deps, UnaryF32Stage { output, param: None, rows, width, stride, col0, kind, p0, p1 })
        })
    }

    /// In-place `a = scale * (a op b[i*bs.0 + r*bs.1 + c*bs.2])` over `[items][rows][width]`.
    fn binary(&mut self, a: u32, b: u32, op: BinaryOp, dims: (u32, u32, u32), bs: (u32, u32, u32), scale: Option<f32>) -> Result<u32, String> {
        let out = self.name_of(a);
        self.op(&[a, b], &out, |p, output, deps| {
            p.binary_f32(a, b, deps, BinaryF32Stage {
                output,
                op,
                items: dims.0,
                rows: dims.1,
                width: dims.2,
                b_item_stride: bs.0,
                b_row_stride: bs.1,
                b_col_stride: bs.2,
                scale,
            })
        })
    }

    /// `out[i][r][c] (at out strides) = x[i][r][c] (at in strides)`; strides `(item, row, offset)`.
    fn copy(&mut self, x: u32, out: &str, dims: (u32, u32, u32), src: (u32, u32, u32), dst: (u32, u32, u32)) -> Result<u32, String> {
        self.op(&[x], out, |p, output, deps| {
            p.copy_cols_f32(x, deps, CopyColsF32Stage {
                output,
                items: dims.0,
                rows: dims.1,
                cols: dims.2,
                in_item_stride: src.0,
                in_stride: src.1,
                in_offset: src.2,
                out_item_stride: dst.0,
                out_stride: dst.1,
                out_offset: dst.2,
            })
        })
    }

    fn gather(&mut self, table: Table<'_>, index: Option<u32>, out: &str, g: Gather) -> Result<u32, String> {
        let mut reads: Vec<u32> = index.into_iter().collect();
        let named;
        let (table, table_rows) = match table {
            Table::Weight(name, rows) => {
                named = format!("w.{name}");
                (TensorRef::Named(&named), rows)
            }
            Table::Tensor(h, rows) => {
                reads.push(h);
                (TensorRef::Handle(h), rows)
            }
        };
        self.op(&reads, out, |p, output, deps| {
            p.gather_rows_f32(deps, GatherRowsF32Stage {
                output,
                table,
                table_rows,
                index,
                rows: g.rows,
                width: g.width,
                vocab: g.vocab,
                rows_per_item: g.per_item,
                repeat: g.repeat,
                index_item_stride: g.index_stride,
                table_item_stride: g.table_stride,
                table_f16: false,
                accumulate: g.accumulate,
                out_stride: g.out_stride,
                out_col0: g.out_col0,
            })
        })
    }

    fn rand(&mut self, seed: u32, out: &str, dims: (u32, u32, u32), stream: u32, normal: bool, scale: f32, offset: f32) -> Result<u32, String> {
        self.op(&[seed], out, |p, output, deps| {
            p.rand_f32(seed, deps, RandF32Stage {
                output,
                items: dims.0,
                rows: dims.1,
                width: dims.2,
                stream,
                stream_shift: 56,
                a: if stream == 1 { RandCoord::Row } else { RandCoord::Column },
                b: if stream == 1 { RandCoord::Column } else { RandCoord::Row },
                a_offset: 0,
                b_offset: 0,
                normal,
                shared_seed: false,
                scale,
                offset,
            })
        })
    }
}

#[derive(Clone, Copy)]
struct Conv<'a> {
    /// Weight stem: `w.{w}.w`, bias `w.{w}.b`.
    w: &'a str,
    bias: bool,
    /// Bias name (after `w.`) other than `{w}.b`.
    bias_name: Option<&'a str>,
    alpha: Option<&'a str>,
    batch: u32,
    rows: u32,
    cin: u32,
    cout: u32,
    k: u32,
    stride: u32,
    dil: u32,
    groups: u32,
    pad: (u32, u32),
    mode: PadMode,
    act_in: Activation,
    act_out: Activation,
    slope: f32,
    lengths: Option<u32>,
    residual: Option<u32>,
    transpose: bool,
    /// `x` is dead after this conv, so an unfused input activation may overwrite it.
    input_is_temp: bool,
}

impl<'a> Conv<'a> {
    fn new(w: &'a str, batch: u32, rows: u32, cin: u32, cout: u32, k: u32) -> Self {
        Self {
            w,
            bias: true,
            bias_name: None,
            alpha: None,
            batch,
            rows,
            cin,
            cout,
            k,
            stride: 1,
            dil: 1,
            groups: 1,
            pad: (0, 0),
            mode: PadMode::Zero,
            act_in: Activation::None,
            act_out: Activation::None,
            slope: 0.0,
            lengths: None,
            residual: None,
            transpose: false,
            input_is_temp: false,
        }
    }
}

enum Table<'a> {
    /// `w.{name}` with this many rows.
    Weight(&'a str, u32),
    Tensor(u32, u32),
}

#[derive(Clone, Copy, Default)]
struct Gather {
    rows: u32,
    width: u32,
    vocab: u32,
    per_item: u32,
    repeat: u32,
    index_stride: u32,
    table_stride: u32,
    out_stride: u32,
    out_col0: u32,
    accumulate: bool,
}

struct Lowering<'a> {
    cfg: &'a Config,
    inputs: Inputs,
    b: u32,
    n: u32,
}

impl Lowering<'_> {
    fn t0(&self) -> u32 {
        self.cfg.prompt + self.n
    }

    fn t1(&self) -> u32 {
        2 * self.t0()
    }

    fn gen(&self) -> u32 {
        2 * self.n
    }

    /// Rows at each HiFT stage output (the last is the STFT frame count).
    fn stage_rows(&self) -> [u32; 3] {
        let s = length_scales(self.cfg);
        [s[3].0 * self.n, s[4].0 * self.n, s[5].0 * self.n + 1]
    }

    fn wav(&self) -> u32 {
        self.gen() * self.cfg.frame_samples()
    }

    /// Additive key mask `[b][t]`: 0 for rows below the item's length, -1e30 past it.
    fn key_mask(&self, o: &mut Ops, t: u32, lengths: u32) -> Result<u32, String> {
        let b = self.b;
        let ones = o.gather(Table::Weight("const.one", 1), None, "act.s3gen.ones", Gather { rows: b * t, width: 1, vocab: 1, per_item: t, repeat: t, ..Default::default() })?;
        let mut c = Conv::new("const", b, t, 1, 1, 1);
        c.bias = false;
        c.lengths = Some(lengths);
        let mask = o.conv(ones, "act.s3gen.mask", c)?;
        o.unary(mask, b * t, 1, 0, 0, Activation::ScaleShift, 1e30, -1e30)
    }

    /// One conformer layer on the time-major `x` (`[t][b][512]`).
    fn conformer(&self, o: &mut Ops, i: u32, x: u32, t: u32, mask: u32) -> Result<(), String> {
        let (b, r) = (self.b, t * self.b);
        let p = format!("enc.L{i}");
        let a = o.layer_norm(x, "act.s3gen.a", &format!("{p}.ln_mha"), r, D_ENC, 1e-12)?;
        let (qw, qb) = (format!("{p}.q"), format!("{p}.q_u.b"));
        let mut c = Conv::new(&qw, 1, r, D_ENC, D_ENC, 1);
        c.bias_name = Some(&qb);
        let q = o.conv(a, "act.s3gen.q", c)?;
        let k = o.linear(a, "act.s3gen.k", &format!("{p}.k"), r, D_ENC, D_ENC, true, Activation::None, None)?;
        let v = o.linear(a, "act.s3gen.v", &format!("{p}.v"), r, D_ENC, D_ENC, true, Activation::None, None)?;
        let m = 2 * t - 1;
        let rel = format!("{p}.relpos.t{t}");
        let mut c = Conv::new(&rel, 1, r, D_ENC, HEADS * m, 1);
        c.groups = HEADS;
        let bd = o.conv(q, "act.s3gen.bd", c)?;
        // bias[b*8 + h][t][j] = bd[t][b*8 + h][t - 1 - t + j]: relative position t - j.
        let bias = o.copy(bd, "act.s3gen.bias", (b * HEADS, t, t), (m, b * HEADS * m - 1, t - 1), (t * t, t, 0))?;
        o.binary(bias, mask, BinaryOp::Add, (b, HEADS * t, t), (t, 0, 1), None)?;
        let att = o.op(&[q, k, v, bias], "act.s3gen.att", |pp, output, deps| {
            pp.attention_f32(q, k, v, deps, AttentionF32Stage {
                output,
                key_lengths: None,
                bias: Some(TensorRef::Handle(bias)),
                batch: 1,
                q_rows: t,
                kv_rows: t,
                heads: HEADS * b,
                head_width: D_ENC / HEADS,
                in_stride: 0,
                k_col0: 0,
                v_col0: 0,
                causal: false,
                scale: 1.0 / ((D_ENC / HEADS) as f32).sqrt(),
                bias_head_stride: t * t,
            })
        })?;
        let xn = o.name_of(x);
        o.linear(att, &xn, &format!("{p}.out"), r, D_ENC, D_ENC, true, Activation::None, Some(x))?;
        let a = o.layer_norm(x, "act.s3gen.a", &format!("{p}.ln_ff"), r, D_ENC, 1e-12)?;
        let h = o.linear(a, "act.s3gen.ff", &format!("{p}.ff1"), r, D_ENC, 4 * D_ENC, true, Activation::Silu, None)?;
        o.linear(h, &xn, &format!("{p}.ff2"), r, 4 * D_ENC, D_ENC, true, Activation::None, Some(x))?;
        Ok(())
    }

    /// `[b][t][c]` <-> `[t][b][c]`.
    fn to_time_major(&self, o: &mut Ops, x: u32, out: &str, t: u32, c: u32) -> Result<u32, String> {
        o.copy(x, out, (self.b, t, c), (t * c, c, 0), (c, self.b * c, 0))
    }

    fn to_item_major(&self, o: &mut Ops, x: u32, out: &str, t: u32, c: u32) -> Result<u32, String> {
        o.copy(x, out, (self.b, t, c), (c, self.b * c, 0), (t * c, c, 0))
    }

    fn encoder(&self, o: &mut Ops) -> Result<(), String> {
        let (cfg, inp, b, n) = (self.cfg, self.inputs, self.b, self.n);
        let (t0, t1, p) = (self.t0(), self.t1(), cfg.prompt);
        let nv = cfg.voices.len() as u32;
        for k in 0..inp.lengths.all().len() {
            let table = format!("s3gen.len.{k}");
            let g = Gather { rows: b, width: 1, vocab: inp.nmax + 1, per_item: 1, repeat: 1, index_stride: 1, ..Default::default() };
            o.gather(Table::Weight(&table, inp.nmax + 1), Some(inp.count), &format!("act.s3gen.len.{k}"), g)?;
        }
        // Token ids [b][prompt | tokens] (u32 bit patterns moved by the f32 copy / gather).
        let idx = o.copy(inp.tokens, "act.s3gen.idx", (b, 1, n), (n, n, 0), (t0, t0, p))?;
        o.gather(Table::Weight("voice.prompt_token", nv), Some(inp.voice), "act.s3gen.idx", Gather { rows: b, width: p, vocab: nv, per_item: 1, repeat: 1, index_stride: 1, out_stride: t0, ..Default::default() })?;
        let e0 = o.gather(Table::Weight("emb", cfg.vocab), Some(idx), "act.s3gen.e0", Gather { rows: b * t0, width: D_ENC, vocab: cfg.vocab, per_item: t0, repeat: 1, index_stride: t0, ..Default::default() })?;
        let e1 = o.linear(e0, "act.s3gen.e1", "enc.embed", b * t0, D_ENC, D_ENC, true, Activation::None, None)?;
        let e0 = o.layer_norm(e1, "act.s3gen.e0", "enc.embed.ln", b * t0, D_ENC, 1e-5)?;
        // leaky(0.01) between the two pre-lookahead convs.
        let leaky = o.fuse.between("w.enc.la1.w", "w.enc.la2.w", 3, ActAt::ProducerOutput);
        let mut c = Conv::new("enc.la1", b, t0, D_ENC, D_ENC, 4);
        c.pad = (0, 3);
        if leaky != ActAt::ConsumerInput {
            c.act_out = Activation::LeakyRelu;
        }
        c.slope = 0.01;
        c.lengths = Some(inp.lengths.tok);
        let e1 = o.conv(e0, "act.s3gen.e1", c)?;
        let mut c = Conv::new("enc.la2", b, t0, D_ENC, D_ENC, 3);
        c.pad = (2, 0);
        if leaky == ActAt::ConsumerInput {
            c.act_in = Activation::LeakyRelu;
            c.slope = 0.01;
        }
        c.residual = Some(e0);
        c.lengths = Some(inp.lengths.tok);
        let e2 = o.conv(e1, "act.s3gen.e2", c)?;
        let x = self.to_time_major(o, e2, "act.s3gen.x", t0, D_ENC)?;
        let mask = self.key_mask(o, t0, inp.lengths.tok)?;
        for i in 0..ENC_LAYERS.0 {
            self.conformer(o, i, x, t0, mask)?;
        }
        let e2 = self.to_item_major(o, x, "act.s3gen.e2", t0, D_ENC)?;
        // Nearest x2, then a causal k5 convolution.
        let e1 = o.gather(Table::Tensor(e2, b * t0), None, "act.s3gen.e1", Gather { rows: b * t1, width: D_ENC, vocab: t0, per_item: t1, repeat: 2, table_stride: t0, ..Default::default() })?;
        let mut c = Conv::new("enc.up", b, t1, D_ENC, D_ENC, 5);
        c.pad = (4, 0);
        c.lengths = Some(inp.lengths.mel);
        let e0 = o.conv(e1, "act.s3gen.e0", c)?;
        let e1 = o.linear(e0, "act.s3gen.e1", "enc.up_embed", b * t1, D_ENC, D_ENC, true, Activation::None, None)?;
        let e0 = o.layer_norm(e1, "act.s3gen.e0", "enc.up_embed.ln", b * t1, D_ENC, 1e-5)?;
        let x = self.to_time_major(o, e0, "act.s3gen.x", t1, D_ENC)?;
        let mask = self.key_mask(o, t1, inp.lengths.mel)?;
        for i in 0..ENC_LAYERS.1 {
            self.conformer(o, ENC_LAYERS.0 + i, x, t1, mask)?;
        }
        let a = o.layer_norm(x, "act.s3gen.a", "enc.after_ln", t1 * b, D_ENC, 1e-5)?;
        let mu = o.linear(a, "act.s3gen.mu", "enc.proj", t1 * b, D_ENC, MEL, true, Activation::None, None)?;
        // Estimator input [2b][t1][x | mu | spks | cond]; the unconditional half's context is 0.
        let w = 4 * MEL;
        o.copy(mu, "act.s3gen.x320", (b, t1, MEL), (MEL, b * MEL, 0), (t1 * w, w, MEL))?;
        o.gather(Table::Weight("voice.spks", nv), Some(inp.voice), "act.s3gen.x320", Gather { rows: b * t1, width: MEL, vocab: nv, per_item: t1, repeat: t1, index_stride: 1, out_stride: w, out_col0: 2 * MEL, ..Default::default() })?;
        let pf = 2 * p;
        let cond = o.gather(Table::Weight("voice.prompt_feat", nv), Some(inp.voice), "act.s3gen.cond", Gather { rows: b, width: pf * MEL, vocab: nv, per_item: 1, repeat: 1, index_stride: 1, out_stride: t1 * MEL, ..Default::default() })?;
        let zeros = o.gather(Table::Weight("const.zeros", 1), None, "act.s3gen.zeros", Gather { rows: 1, width: w, vocab: 1, per_item: 1, repeat: 1, ..Default::default() })?;
        o.copy(zeros, "act.s3gen.cond", (b, t1 - pf, MEL), (0, 0, 0), (t1 * MEL, MEL, pf * MEL))?;
        o.copy(cond, "act.s3gen.x320", (b, t1, MEL), (t1 * MEL, MEL, 0), (t1 * w, w, 3 * MEL))?;
        o.copy(zeros, "act.s3gen.x320", (1, b * t1, w), (0, 0, 0), (0, w, b * t1 * w))?;
        // CFM state x = z ~ N(0, 1) (stream 1, keyed by mel frame and bin).
        o.rand(inp.seed, "act.s3gen.xt", (b, t1, MEL), 1, true, 1.0, 0.0)?;
        // Guidance lengths: the unconditional half repeats the conditional one.
        o.copy(inp.lengths.mel, "act.s3gen.len2", (1, 1, b), (0, 0, 0), (0, 0, 0))?;
        o.copy(inp.lengths.mel, "act.s3gen.len2", (1, 1, b), (0, 0, 0), (0, 0, b))?;
        Ok(())
    }


    fn len2(o: &Ops) -> u32 {
        o.names["act.s3gen.len2"]
    }

    /// CausalResnetBlock1D `j` of Euler step `step` on `x` (`cin` channels) into `out`.
    fn resnet(&self, o: &mut Ops, step: usize, j: u32, x: u32, cin: u32, out: &str) -> Result<u32, String> {
        let (b2, t1) = (2 * self.b, self.t1());
        let (r, len2) = (b2 * t1, Self::len2(o));
        let p = format!("cfm.r{j}");
        let (c1, c2, res) = (format!("{p}.c1"), format!("{p}.c2"), format!("{p}.res"));
        let mut c = Conv::new(&c1, b2, t1, cin, D_CFM, 3);
        c.pad = (2, 0);
        c.lengths = Some(len2);
        let y1 = o.conv(x, "act.s3gen.y1", c)?;
        o.layer_norm(y1, "act.s3gen.y1", &format!("{p}.ln1"), r, D_CFM, 1e-5)?;
        o.unary(y1, r, D_CFM, 0, 0, Activation::Mish, 0.0, 0.0)?;
        let tv = format!("cfm.tvec.s{step}.r{j}");
        o.gather(Table::Weight(&tv, 1), None, "act.s3gen.y1", Gather { rows: r, width: D_CFM, vocab: 1, per_item: r, repeat: r, accumulate: true, ..Default::default() })?;
        let mut c = Conv::new(&c2, b2, t1, D_CFM, D_CFM, 3);
        c.pad = (2, 0);
        c.lengths = Some(len2);
        let y2 = o.conv(y1, "act.s3gen.y2", c)?;
        o.layer_norm(y2, "act.s3gen.y2", &format!("{p}.ln2"), r, D_CFM, 1e-5)?;
        o.unary(y2, r, D_CFM, 0, 0, Activation::Mish, 0.0, 0.0)?;
        let mut c = Conv::new(&res, b2, t1, cin, D_CFM, 1);
        c.lengths = Some(len2);
        c.residual = Some(y2);
        o.conv(x, out, c)
    }

    /// BasicTransformerBlock `k` on `h` (`[2b][t1][256]`), in place.
    fn tblock(&self, o: &mut Ops, k: u32, h: u32) -> Result<(), String> {
        let (b2, t1) = (2 * self.b, self.t1());
        let (r, len2) = (b2 * t1, Self::len2(o));
        let inner = HEADS * 64;
        let p = format!("cfm.tb{k}");
        let hn = o.name_of(h);
        let a = o.layer_norm(h, "act.s3gen.a", &format!("{p}.ln1"), r, D_CFM, 1e-5)?;
        let qkv = o.linear(a, "act.s3gen.qkv", &format!("{p}.qkv"), r, D_CFM, 3 * inner, false, Activation::None, None)?;
        let att = o.op(&[qkv, len2], "act.s3gen.att", |pp, output, deps| {
            pp.attention_f32(qkv, qkv, qkv, deps, AttentionF32Stage {
                output,
                key_lengths: Some(len2),
                bias: None,
                batch: b2,
                q_rows: t1,
                kv_rows: t1,
                heads: HEADS,
                head_width: 64,
                in_stride: 3 * inner,
                k_col0: inner,
                v_col0: 2 * inner,
                causal: false,
                scale: 0.125,
                bias_head_stride: 0,
            })
        })?;
        o.linear(att, &hn, &format!("{p}.out"), r, inner, D_CFM, true, Activation::None, Some(h))?;
        let a = o.layer_norm(h, "act.s3gen.a", &format!("{p}.ln3"), r, D_CFM, 1e-5)?;
        let f = o.linear(a, "act.s3gen.ff", &format!("{p}.ff1"), r, D_CFM, 4 * D_CFM, true, Activation::GeluErf, None)?;
        o.linear(f, &hn, &format!("{p}.ff2"), r, 4 * D_CFM, D_CFM, true, Activation::None, Some(h))?;
        Ok(())
    }

    /// One Euler step of the guided flow: `x += dt * ((1 + cfg) v_cond - cfg v_uncond)`.
    fn cfm_step(&self, o: &mut Ops, step: usize) -> Result<(), String> {
        let (b, t1) = (self.b, self.t1());
        let (b2, r, w) = (2 * b, 2 * b * t1, 4 * MEL);
        let len2 = Self::len2(o);
        let xt = o.names["act.s3gen.xt"];
        o.copy(xt, "act.s3gen.x320", (b, t1, MEL), (t1 * MEL, MEL, 0), (t1 * w, w, 0))?;
        let x320 = o.copy(xt, "act.s3gen.x320", (b, t1, MEL), (t1 * MEL, MEL, 0), (t1 * w, w, b * t1 * w))?;
        let other = |o: &Ops, h: u32| if o.name_of(h) == "act.s3gen.h0" { "act.s3gen.h1" } else { "act.s3gen.h0" };
        let causal = |w: &'static str, x: u32, o: &mut Ops, out: &str| -> Result<u32, String> {
            let mut c = Conv::new(w, b2, t1, D_CFM, D_CFM, 3);
            c.pad = (2, 0);
            c.lengths = Some(len2);
            o.conv(x, out, c)
        };
        let mut h = self.resnet(o, step, 0, x320, w, "act.s3gen.h0")?;
        for i in 0..TBLOCKS {
            self.tblock(o, i, h)?;
        }
        let cat = o.copy(h, "act.s3gen.cat", (1, r, D_CFM), (0, D_CFM, 0), (0, 2 * D_CFM, D_CFM))?;
        h = causal("cfm.down", h, o, "act.s3gen.h1")?;
        for j in 1..=MID_BLOCKS {
            let out = other(o, h);
            h = self.resnet(o, step, j, h, D_CFM, out)?;
            for i in 0..TBLOCKS {
                self.tblock(o, TBLOCKS * j + i, h)?;
            }
        }
        o.copy(h, "act.s3gen.cat", (1, r, D_CFM), (0, D_CFM, 0), (0, 2 * D_CFM, 0))?;
        let out = other(o, h);
        h = self.resnet(o, step, MID_BLOCKS + 1, cat, 2 * D_CFM, out)?;
        for i in 0..TBLOCKS {
            self.tblock(o, TBLOCKS * (MID_BLOCKS + 1) + i, h)?;
        }
        let out = other(o, h);
        h = causal("cfm.upc", h, o, out)?;
        let y = causal("cfm.fin", h, o, "act.s3gen.y1")?;
        o.layer_norm(y, "act.s3gen.y1", "cfm.fin.ln", r, D_CFM, 1e-5)?;
        let mut c = Conv::new("cfm.proj", b2, t1, D_CFM, MEL, 1);
        c.act_in = Activation::Mish;
        c.input_is_temp = true;
        c.lengths = Some(len2);
        let d = o.conv(y, "act.s3gen.d", c)?;
        let rows = b * t1;
        let cfg = self.cfg.cfg_rate;
        let du = o.copy(d, "act.s3gen.du", (1, rows, MEL), (0, MEL, rows * MEL), (0, MEL, 0))?;
        o.unary(du, rows, MEL, 0, 0, Activation::ScaleShift, cfg as f32, 0.0)?;
        o.unary(d, rows, MEL, 0, 0, Activation::ScaleShift, (1.0 + cfg) as f32, 0.0)?;
        o.binary(d, du, BinaryOp::Sub, (1, rows, MEL), (0, MEL, 1), None)?;
        let dt = self.cfg.t_span[step + 1] - self.cfg.t_span[step];
        o.unary(d, rows, MEL, 0, 0, Activation::ScaleShift, dt, 0.0)?;
        o.binary(xt, d, BinaryOp::Add, (1, rows, MEL), (0, MEL, 1), None)?;
        Ok(())
    }

    /// HiFT ResBlock: for each dilation `x += conv2(snake(conv1(snake(x))))`; the result lands in `out`.
    #[allow(clippy::too_many_arguments)]
    fn resblock(&self, o: &mut Ops, pre: &str, x: u32, out: &str, k: u32, dils: &[u32], ch: u32, rows: u32, len: u32) -> Result<u32, String> {
        let mut cur = x;
        for (j, &d) in dils.iter().enumerate() {
            let (c1, a1, c2, a2) = (format!("{pre}.d{j}.c1"), format!("{pre}.d{j}.a1"), format!("{pre}.d{j}.c2"), format!("{pre}.d{j}.a2"));
            let mut c = Conv::new(&c1, self.b, rows, ch, ch, k);
            c.dil = d;
            c.pad = (d * (k - 1) / 2, d * (k - 1) / 2);
            c.alpha = Some(&a1);
            c.act_in = Activation::Snake;
            c.lengths = Some(len);
            let t = o.conv(cur, "act.s3gen.rbt", c)?;
            let next = if j + 1 == dils.len() {
                out
            } else if o.name_of(cur) == "act.s3gen.rbp" {
                "act.s3gen.rbq"
            } else {
                "act.s3gen.rbp"
            };
            let mut c = Conv::new(&c2, self.b, rows, ch, ch, k);
            c.pad = ((k - 1) / 2, (k - 1) / 2);
            c.alpha = Some(&a2);
            c.act_in = Activation::Snake;
            c.residual = Some(cur);
            c.lengths = Some(len);
            cur = o.conv(t, next, c)?;
        }
        Ok(cur)
    }

    fn vocoder(&self, o: &mut Ops) -> Result<(), String> {
        let (cfg, inp, b) = (self.cfg, self.inputs, self.b);
        let (t1, g, wav) = (self.t1(), self.gen(), self.wav());
        let rows_out = self.stage_rows();
        let f = rows_out[2];
        let (lg, lf, lw) = (inp.lengths.gen, inp.lengths.stage[2], inp.lengths.wav);
        let pf = 2 * cfg.prompt;
        let xt = o.names["act.s3gen.xt"];
        let mel = o.copy(xt, "act.s3gen.mel", (b, g, MEL), (t1 * MEL, MEL, pf * MEL), (g * MEL, MEL, 0))?;
        // F0 predictor.
        let mut v = mel;
        let mut cin = MEL;
        let mut elu_in = false;
        for i in 0..F0_CONVS {
            let w = format!("hift.f0.c{i}");
            // ELU after each conv; the last one's only candidate host is this conv.
            let elu = if i + 1 < F0_CONVS {
                o.fuse.between(&format!("w.{w}.w"), &format!("w.hift.f0.c{}.w", i + 1), 3, ActAt::ProducerOutput)
            } else {
                ActAt::ProducerOutput
            };
            let mut c = Conv::new(&w, b, g, cin, 512, 3);
            c.pad = (1, 1);
            if elu_in {
                c.act_in = Activation::Elu;
            }
            if elu != ActAt::ConsumerInput {
                c.act_out = Activation::Elu;
            }
            c.lengths = Some(lg);
            v = o.conv(v, if i % 2 == 0 { "act.s3gen.va" } else { "act.s3gen.vb" }, c)?;
            elu_in = elu == ActAt::ConsumerInput;
            cin = 512;
        }
        let mut c = Conv::new("hift.f0.cls", b, g, 512, 1, 1);
        c.act_out = Activation::Abs;
        c.lengths = Some(lg);
        let f0 = o.conv(v, "act.s3gen.f0", c)?;
        // NSF source: 9 harmonics of the upsampled F0, phase integral in fp64 (wrapped), random
        // initial phases (stream 2, harmonic 0 at 0) and noise (stream 3); voiced where F0 > 10.
        let fs = cfg.frame_samples();
        let nh = cfg.harmonics;
        let f0up = o.gather(Table::Tensor(f0, b * g), None, "act.s3gen.f0up", Gather { rows: b * wav, width: 1, vocab: g, per_item: wav, repeat: fs, table_stride: g, ..Default::default() })?;
        let uv = o.unary_to(f0, "act.s3gen.uv", b * g, 1, 0, 0, Activation::ScaleShift, 1.0, -cfg.voiced_threshold)?;
        o.unary(uv, b * g, 1, 0, 0, Activation::ScaleShift, 1e30, 0.0)?;
        o.unary(uv, b * g, 1, 0, 0, Activation::Clamp, 0.0, 1.0)?;
        let third = cfg.sine_amp / 3.0;
        let amp = o.unary_to(uv, "act.s3gen.amp", b * g, 1, 0, 0, Activation::ScaleShift, cfg.noise_std - third, third)?;
        let (_, per_sample) = harmonic_scale(cfg.sampling_rate);
        let sine = o.op(&[f0up, lw], "act.s3gen.sine", |pp, output, deps| {
            pp.cumsum_f64(f0up, deps, CumSumF64Stage {
                output,
                column_scale: Some(TensorRef::Named("w.s3gen.harm")),
                lengths: Some(lw),
                items: b,
                rows: wav,
                width: nh,
                x_width: 1,
                exclusive: false,
                wrap: true,
                f64_output: false,
                scale: per_sample,
                post_scale: std::f32::consts::TAU,
            })
        })?;
        let pv = o.rand(inp.seed, "act.s3gen.pv", (b, 1, nh), 2, false, std::f32::consts::TAU, -0.5)?;
        o.unary(pv, b, 1, nh, 0, Activation::ScaleShift, 0.0, 0.0)?;
        o.binary(sine, pv, BinaryOp::Add, (b, wav, nh), (nh, 0, 1), None)?;
        o.unary(sine, b * wav, nh, 0, 0, Activation::Sin, 0.0, 0.0)?;
        o.binary(sine, uv, BinaryOp::Mul, (b * g, fs, nh), (1, 0, 0), Some(cfg.sine_amp))?;
        let noise = o.rand(inp.seed, "act.s3gen.noise", (b, wav, nh), 3, true, 1.0, 0.0)?;
        o.binary(noise, amp, BinaryOp::Mul, (b * g, fs, nh), (1, 0, 0), None)?;
        o.binary(sine, noise, BinaryOp::Add, (1, b * wav, nh), (0, nh, 1), None)?;
        let mut c = Conv::new("hift.src", b, wav, nh, 1, 1);
        c.act_out = Activation::Tanh;
        c.lengths = Some(lw);
        let src = o.conv(sine, "act.s3gen.src", c)?;
        let (nfft, hop) = (cfg.n_fft, cfg.hop);
        let spec = 2 * (nfft / 2 + 1);
        let mut c = Conv::new("hift.stft", b, wav, 1, spec, nfft);
        c.bias = false;
        c.stride = hop;
        c.pad = (nfft / 2, nfft / 2);
        c.mode = PadMode::Reflect;
        c.lengths = Some(lw);
        let stft = o.conv(src, "act.s3gen.stft", c)?;
        // Mel -> waveform.
        // leaky(0.1) between the pre conv and the first upsampling.
        let leaky0 = o.fuse.between("w.hift.pre.w", "w.hift.up0.w", cfg.upsample[0][1], ActAt::ConsumerInput);
        let mut c = Conv::new("hift.pre", b, g, MEL, 512, 7);
        c.pad = (3, 3);
        if leaky0 != ActAt::ConsumerInput {
            c.act_out = Activation::LeakyRelu;
            c.slope = 0.1;
        }
        c.lengths = Some(lg);
        let mut x = o.conv(mel, "act.s3gen.xa", c)?;
        let rows_in = [g, rows_out[0], rows_out[1]];
        let len_in = [lg, inp.lengths.stage[0], inp.lengths.stage[1]];
        let mut cin = 512;
        for i in 0..3 {
            let co = cin / 2;
            let [u, k, pad] = cfg.upsample[i];
            let [sk, ss, sp] = cfg.source_downs[i];
            let (rows, len) = (rows_out[i], inp.lengths.stage[i]);
            let sdw = format!("hift.sd{i}");
            let mut c = Conv::new(&sdw, b, f, spec, co, sk);
            c.stride = ss;
            c.pad = (sp, sp);
            c.lengths = Some(lf);
            let sd = o.conv(stft, "act.s3gen.sd", c)?;
            let (rk, dils) = &cfg.source_resblocks[i];
            let si = self.resblock(o, &format!("hift.sr{i}"), sd, "act.s3gen.si", *rk, dils, co, rows, len)?;
            // leaky-ReLU 0.1 -> ConvTranspose; the last stage is reflection-padded by one frame on
            // the left: crop one frame less there and copy frame 2 over frame 0.
            let last = i == 2;
            let upw = format!("hift.up{i}");
            let mut c = Conv::new(&upw, b, rows_in[i], cin, co, k);
            c.transpose = true;
            c.stride = u;
            c.dil = 0;
            c.pad = if last { (pad - 1, pad) } else { (pad, pad) };
            if i > 0 || leaky0 == ActAt::ConsumerInput {
                c.act_in = Activation::LeakyRelu;
            }
            c.slope = 0.1;
            c.lengths = Some(len_in[i]);
            c.residual = (!last).then_some(si);
            let xu = o.conv(x, "act.s3gen.xu", c)?;
            if last {
                o.copy(xu, "act.s3gen.xu", (b, 1, co), (rows * co, co, 2 * co), (rows * co, co, 0))?;
                o.binary(xu, si, BinaryOp::Add, (1, b * rows, co), (0, co, 1), None)?;
            }
            let mut acc = 0;
            for j in 0..3 {
                let (rk, dils) = &cfg.resblocks[3 * i + j];
                let out = if j == 0 { "act.s3gen.acc" } else { "act.s3gen.rbo" };
                let y = self.resblock(o, &format!("hift.rb{}", 3 * i + j), xu, out, *rk, dils, co, rows, len)?;
                if j == 0 {
                    acc = y;
                } else {
                    o.binary(acc, y, BinaryOp::Add, (1, b * rows, co), (0, co, 1), (j == 2).then_some(1.0 / 3.0))?;
                }
            }
            x = acc;
            cin = co;
        }
        let mut c = Conv::new("hift.post", b, f, cin, spec, 7);
        c.pad = (3, 3);
        c.act_in = Activation::LeakyRelu;
        c.slope = 0.01;
        c.lengths = Some(lf);
        let post = o.conv(x, "act.s3gen.post", c)?;
        // iSTFT: magnitude exp (clipped at 100), phase sin; Re/Im through a transposed
        // convolution (inverse real DFT times the window), divided by the window envelope.
        let nb = nfft / 2 + 1;
        o.unary(post, b * f, nb, spec, 0, Activation::Exp, 0.0, 0.0)?;
        o.unary(post, b * f, nb, spec, 0, Activation::Clamp, f32::MIN, 100.0)?;
        o.unary(post, b * f, nb, spec, nb, Activation::Sin, 0.0, 0.0)?;
        let ri = o.copy(post, "act.s3gen.ri", (b * f, 2, nb), (spec, 0, nb), (spec, nb, 0))?;
        o.unary(ri, b * f, nb, spec, 0, Activation::Cos, 0.0, 0.0)?;
        o.unary(ri, b * f, nb, spec, nb, Activation::Sin, 0.0, 0.0)?;
        o.binary(ri, post, BinaryOp::Mul, (b * f, 2, nb), (spec, 0, 1), None)?;
        let synth = |w: &'static str, cin: u32| {
            let mut c = Conv::new(w, b, f, cin, 1, nfft);
            c.bias = false;
            c.transpose = true;
            c.stride = hop;
            c.dil = 0;
            c.pad = (nfft / 2, nfft / 2);
            c.lengths = Some(lf);
            c
        };
        let pcm = o.conv(ri, "act.s3gen.pcm", synth("hift.istft", spec))?;
        let ones = o.gather(Table::Weight("const.one", 1), None, "act.s3gen.ones", Gather { rows: b * f, width: 1, vocab: 1, per_item: f, repeat: f, ..Default::default() })?;
        let env = o.conv(ones, "act.s3gen.env", synth("hift.env", 1))?;
        o.unary(env, b * wav, 1, 0, 0, Activation::Clamp, 1e-11, f32::MAX)?;
        o.binary(pcm, env, BinaryOp::Div, (1, b * wav, 1), (0, 1, 0), None)?;
        o.unary(pcm, b * wav, 1, 0, 0, Activation::Clamp, -cfg.audio_limit, cfg.audio_limit)?;
        let trim = cfg.trim;
        let fade = o.copy(pcm, "act.s3gen.fade", (b, 1, trim), (wav, 0, 0), (trim, 0, 0))?;
        let window = o.gather(Table::Weight("hift.fade", 1), None, "act.s3gen.fadew", Gather { rows: 1, width: trim, vocab: 1, per_item: 1, repeat: 1, ..Default::default() })?;
        o.binary(fade, window, BinaryOp::Mul, (b, 1, trim), (0, 0, 1), None)?;
        o.copy(fade, "act.s3gen.pcm", (b, 1, trim), (trim, 0, 0), (wav, 0, 0))?;
        Ok(())
    }
}

/// `(a, s)` with `a * s` (exact in f64) nearest `1 / rate` and `a`'s low four mantissa bits zero,
/// so the per-harmonic column scale `(i + 1) * a` (i < 16) is exact: the f64 prefix sum then turns
/// into phase cycles at f64 accuracy although the instruction carries f32 constants.
fn harmonic_scale(rate: u32) -> (f32, f32) {
    let target = 1.0 / f64::from(rate);
    let mut best = (1.0f32, target as f32, f64::MAX);
    for m in (0u32..1 << 23).step_by(16) {
        let a = f32::from_bits(0x3f80_0000 | m);
        let s = (target / f64::from(a)) as f32;
        let err = (f64::from(a) * f64::from(s) - target).abs();
        if err < best.2 {
            best = (a, s, err);
        }
    }
    (best.0, best.1)
}

/// Packet constants: export tensors as stored, plus the ones derived from them per capacity.
struct Derived<'a> {
    reader: &'a crate::checkpoint::TensorReader,
    cfg: &'a Config,
    nmax: u32,
    cache: HashMap<String, Vec<f32>>,
}

fn le_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Derived<'_> {
    fn f32s(&mut self, name: &str) -> Result<&Vec<f32>, String> {
        if !self.cache.contains_key(name) {
            let (dtype, bytes) = self.reader.read(name)?;
            if dtype != "F32" {
                return Err(format!("{name}: {dtype}, expected F32"));
            }
            let v = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            self.cache.insert(name.to_string(), v);
        }
        Ok(&self.cache[name])
    }

    fn bytes(&mut self, name: &str) -> Result<Vec<u8>, String> {
        match name {
            "const.one" | "const.w" => return Ok(le_bytes(&[1.0])),
            "const.zeros" => return Ok(le_bytes(&[0.0; 4 * MEL as usize])),
            "s3gen.harm" => {
                let (a, _) = harmonic_scale(self.cfg.sampling_rate);
                return Ok(le_bytes(&(1..=self.cfg.harmonics).map(|i| i as f32 * a).collect::<Vec<_>>()));
            }
            "voice.prompt_token" => {
                let (dtype, bytes) = self.reader.read(name)?;
                return if dtype == "I32" { Ok(bytes) } else { Err(format!("{name}: {dtype}, expected I32")) };
            }
            _ => {}
        }
        if let Some(k) = name.strip_prefix("s3gen.len.").and_then(|k| k.parse::<usize>().ok()) {
            let (per, offset) = *length_scales(self.cfg).get(k).ok_or(format!("{name}: no such resolution"))?;
            return Ok((0..=self.nmax).flat_map(|n| (n * per + offset).to_le_bytes()).collect());
        }
        if let Some((step, block)) = name.strip_prefix("cfm.tvec.s").and_then(|r| r.split_once(".r")) {
            let (step, block): (usize, usize) = (step.parse().map_err(|_| name.to_string())?, block.parse().map_err(|_| name.to_string())?);
            let width = D_CFM as usize;
            let tv = self.f32s("cfm.tvec")?;
            let at = (step * (MID_BLOCKS as usize + 2) + block) * width;
            return tv.get(at..at + width).map(le_bytes).ok_or(format!("{name}: out of range"));
        }
        if let Some((layer, rest)) = name.strip_prefix("enc.L").and_then(|r| r.split_once(".relpos.t")) {
            let (t, part) = rest.split_once('.').ok_or(name.to_string())?;
            let (layer, t): (u32, u32) = (layer.parse().map_err(|_| name.to_string())?, t.parse().map_err(|_| name.to_string())?);
            return self.relpos(layer, t, part == "b").map(|v| le_bytes(&v));
        }
        let (dtype, bytes) = self.reader.read(name)?;
        if dtype != "F32" {
            return Err(format!("{name}: {dtype}, expected F32"));
        }
        Ok(bytes)
    }

    /// Grouped 1x1 weight `[head][2t-1][64]` (row m <-> relative position t-1-m, already / 8) of
    /// layer `layer`'s projected position table for `t` rows, or its bias `(v - u)_head · row`.
    fn relpos(&mut self, layer: u32, t: u32, bias: bool) -> Result<Vec<f32>, String> {
        let rows = self.cfg.relpos_rows[usize::from(layer >= ENC_LAYERS.0)];
        if t > rows {
            return Err(format!("layer {layer}: {t} rows exceed the exported position table ({rows})"));
        }
        let vu = self.f32s(&format!("enc.L{layer}.pos_vu"))?.clone();
        let table = self.f32s(&format!("enc.L{layer}.relpos"))?;
        let (m, dk, width) = (2 * t as usize - 1, (D_ENC / HEADS) as usize, D_ENC as usize);
        let first = (rows - t) as usize;
        let mut out = Vec::with_capacity(HEADS as usize * m * if bias { 1 } else { dk });
        for h in 0..HEADS as usize {
            for r in 0..m {
                let row = &table[(first + r) * width + h * dk..][..dk];
                if bias {
                    out.push(row.iter().zip(&vu[h * dk..]).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum::<f64>() as f32);
                } else {
                    out.extend_from_slice(row);
                }
            }
        }
        Ok(out)
    }
}

fn pipeline_section(
    prefix: &PacketPrefix,
    cfg: &Config,
    roles: BTreeMap<String, u32>,
    inputs: &Inputs,
    bmax: u32,
    nmax: u32,
    names: &HashMap<String, u32>,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION};
    let t = |h: u32| prefix.model.tensors[h as usize].name.clone();
    let u32s = |h: u32, n: u64| PipelineTensor { name: t(h), dtype: PipelineDType::U32, shape: vec![n] };
    let pcm = *names.get("act.s3gen.pcm").ok_or("act.s3gen.pcm is not declared")?;
    let samples = length_scales(cfg)[6].0;
    let tensors = BTreeMap::from([
        ("codes".into(), u32s(inputs.tokens, u64::from(bmax * nmax))),
        ("voice".into(), u32s(inputs.voice, u64::from(bmax))),
        ("seed".into(), u32s(inputs.seed, u64::from(bmax) * 2)),
        ("lengths.0".into(), u32s(inputs.count, u64::from(bmax))),
        ("pcm".into(), PipelineTensor { name: t(pcm), dtype: PipelineDType::F32, shape: vec![u64::from(bmax * nmax * samples)] }),
    ]);
    // The host writes each item's token count; every other resolution is derived on device.
    let parameters = BTreeMap::from([
        ("codec.frame_codes".into(), 1),
        ("codec.frame_samples".into(), u64::from(samples)),
        ("audio.sample_rate".into(), u64::from(cfg.sampling_rate)),
        ("lengths.count".into(), 1),
        ("lengths.0.rows_per_frame".into(), 1),
        ("stream.window_frames".into(), 0),
        ("stream.lookahead_frames".into(), 0),
        ("stream.first_tokens".into(), 20),
        ("stream.chunk_tokens".into(), 25),
        ("stream.hold_tokens".into(), 3),
        ("stream.fade_samples".into(), 480),
    ]);
    let strings = BTreeMap::from([("voices".into(), cfg.voices.join("\n"))]);
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline { name: PIPELINE.into(), driver: "codec.v1".into(), programs: roles, tensors, parameters, strings }],
    };
    metadata.validate(prefix.model.progs.len(), |name| prefix.model.tensors.iter().find(|x| x.name == name).map(|x| x.bytes))?;
    Ok(packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    })
}
