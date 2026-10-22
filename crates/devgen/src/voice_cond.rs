//! Chatterbox voice conditioning (`prepare_conditionals`, `scripts/tts/chatterbox_voice_export.py`)
//! lowered to a `voice.encode.v1` packet: one 24 kHz reference clip in, the T3 conditioning rows
//! and an s3gen voice slot (prompt tokens, prompt mels, speaker vector) out.

use packet::dev::DevOp;

use crate::fish_codec::Seq;
use crate::pipeline::{
    Activation, AttentionF32Stage, BinaryOp, Conv1dF32Stage, CopyColsF32Stage,
    LayerNormRowsF32Stage, LengthAffineU32Stage, PadMode, ReduceRowsF32Stage, ReduceRowsOp,
    TensorRef, UnaryF32Stage,
};

const TOK_DIM: u32 = 1280;
const TOK_HEADS: u32 = 20;
const TOK_MELS: u32 = 128;
const TOK_FSMN: u32 = 31;
const FSQ_DIM: u32 = 8;
const FSQ_CODES: u32 = 6561;

/// A channels-last FP32 conv over one item: `in_rows` rows of `cin` channels.
fn conv<'a>(
    output: &'a str,
    weight: &'a str,
    in_rows: u32,
    cin: u32,
    cout: u32,
) -> Conv1dF32Stage<'a> {
    Conv1dF32Stage {
        output: TensorRef::Named(output),
        weight: TensorRef::Named(weight),
        bias: None,
        alpha: None,
        residual: None,
        lengths: None,
        batch: 1,
        in_rows,
        in_channels: cin,
        out_channels: cout,
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
        split_bf16: false,
        weight_tap_major: false,
        wgmma: false,
        weight_split: false,
        dual_alpha: false,
    }
}

impl Seq {
    fn conv(&mut self, x: u32, c: Conv1dF32Stage<'_>) -> Result<u32, String> {
        let dep = self.last;
        let e = self.p().conv1d_f32(x, false, &[dep], c)?;
        Ok(self.done(e))
    }

    fn layer_norm(
        &mut self,
        x: u32,
        out: &str,
        rows: u32,
        width: u32,
        w: &str,
    ) -> Result<u32, String> {
        let (g, b) = (format!("{w}.w"), format!("{w}.b"));
        let dep = self.last;
        let e = self.p().layer_norm_f32(
            x,
            &[dep],
            LayerNormRowsF32Stage {
                output: TensorRef::Named(out),
                gamma: Some(TensorRef::Named(&g)),
                beta: Some(TensorRef::Named(&b)),
                rows,
                width,
                epsilon: 1e-5,
                activation: Activation::None,
                add_row: None,
            },
        )?;
        Ok(self.done(e))
    }

    fn scale_shift(
        &mut self,
        x: u32,
        rows: u32,
        width: u32,
        a: f32,
        b: f32,
    ) -> Result<u32, String> {
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
                kind: Activation::ScaleShift,
                p0: a,
                p1: b,
            },
        )?;
        Ok(self.done(e))
    }
}

/// Rows after a k3 s2 p1 conv.
pub(crate) fn halve(rows: u32) -> u32 {
    rows.div_ceil(2)
}

/// The S3Tokenizer V2 encoder + FSQ over `mel` (`[t][128]`, valid rows `lens[0]`; `lens[1]`,
/// `lens[2]` the valid rows after each stride-2 conv) -> u32 tokens in `tokens` (`[t/4]`, rows
/// past `lens[2]` unspecified). `pos` holds the rows' RoPE positions `0..`.
pub(crate) fn s3tok(
    s: &mut Seq,
    mel: u32,
    t: u32,
    lens: [u32; 3],
    pos: u32,
    tokens: u32,
) -> Result<u32, String> {
    let d = TOK_DIM;
    let t1 = halve(t);
    let t2 = halve(t1);
    let mut c = conv("act.vc.b", "w.tok.conv1.w", t, TOK_MELS, d);
    (c.bias, c.kernel, c.stride, c.pad_before, c.pad_after) =
        (Some(TensorRef::Named("w.tok.conv1.b")), 3, 2, 1, 1);
    (c.lengths, c.output_activation) = (Some(lens[0]), Activation::GeluErf);
    let x1 = s.conv(mel, c)?;
    let mut c = conv("act.vc.a", "w.tok.conv2.w", t1, d, d);
    (c.bias, c.kernel, c.stride, c.pad_before, c.pad_after) =
        (Some(TensorRef::Named("w.tok.conv2.b")), 3, 2, 1, 1);
    (c.lengths, c.output_activation) = (Some(lens[1]), Activation::GeluErf);
    let x = s.conv(x1, c)?;
    let hd = d / TOK_HEADS;
    for k in 0..6 {
        let w = |n: &str| format!("w.tok.b{k}.{n}");
        let a = s.layer_norm(x, "act.vc.b", t2, d, &w("ln1"))?;
        let (qw, qb) = (w("qkv.w"), w("qkv.b"));
        let mut c = conv("act.vc.qkv", &qw, t2, d, 3 * d);
        c.bias = Some(TensorRef::Named(&qb));
        let qkv = s.conv(a, c)?;
        // Rotate-half RoPE on the q and k heads (theta 1e4, positions 0..t2).
        s.raw(
            DevOp::RopeAxialF32,
            u64::from(t2 * 2 * d / 2).div_ceil(2048),
            qkv,
            |di| {
                di.t[..2].copy_from_slice(&[qkv, pos]);
                di.i[..7].copy_from_slice(&[t2, 2 * TOK_HEADS, hd, 3 * d, 1, 0, 0]);
                di.f[0] = 10000.0;
            },
        )?;
        let dep = s.last;
        let e = s.p().copy_cols_f32(
            qkv,
            &[dep],
            CopyColsF32Stage {
                output: TensorRef::Named("act.vc.v"),
                items: 1,
                rows: t2,
                cols: d,
                in_item_stride: 0,
                in_stride: 3 * d,
                in_offset: 2 * d,
                out_item_stride: 0,
                out_stride: d,
                out_offset: 0,
            },
        )?;
        let v = s.done(e);
        // q and k are each scaled by hd^-1/4 in the reference: one 1/8 on the scores.
        let dep = s.last;
        let e = s.p().attention_f32(
            qkv,
            qkv,
            qkv,
            &[dep],
            AttentionF32Stage {
                output: TensorRef::Named("act.vc.att"),
                key_lengths: Some(lens[2]),
                bias: None,
                batch: 1,
                q_rows: t2,
                kv_rows: t2,
                heads: TOK_HEADS,
                head_width: hd,
                in_stride: 3 * d,
                k_col0: d,
                v_col0: 2 * d,
                causal: false,
                scale: 1.0 / (hd as f32).sqrt(),
                bias_head_stride: 0,
                relative: false,
                key_length_heads: 0,
                prefix: None,
                window: 0,
                stream: None,
            },
        )?;
        let att = s.done(e);
        // FSMN memory: v + depthwise k31 over the valid rows (zero past them).
        let fw = w("fsmn.w");
        let mut c = conv("act.vc.f", &fw, t2, d, d);
        (c.kernel, c.groups, c.pad_before, c.pad_after) = (TOK_FSMN, d, TOK_FSMN / 2, TOK_FSMN / 2);
        (c.residual, c.lengths) = (Some(v), Some(lens[2]));
        let f = s.conv(v, c)?;
        let (ow, ob) = (w("out.w"), w("out.b"));
        let mut c = conv("act.vc.b", &ow, t2, d, d);
        (c.bias, c.residual) = (Some(TensorRef::Named(&ob)), Some(f));
        let o = s.conv(att, c)?;
        s.binary(x, o, BinaryOp::Add, t2, d)?;
        let b = s.layer_norm(x, "act.vc.b", t2, d, &w("ln2"))?;
        let (w1, b1, w2, b2) = (w("mlp1.w"), w("mlp1.b"), w("mlp2.w"), w("mlp2.b"));
        let mut c = conv("act.vc.h", &w1, t2, d, 4 * d);
        (c.bias, c.output_activation) = (Some(TensorRef::Named(&b1)), Activation::GeluErf);
        let h = s.conv(b, c)?;
        let mut c = conv("act.vc.a", &w2, t2, 4 * d, d);
        (c.bias, c.residual) = (Some(TensorRef::Named(&b2)), Some(x));
        s.conv(h, c)?;
    }
    // FSQ: 0.999 tanh(project_down(x)), then the per-dimension round as a product-codebook argmax.
    let mut c = conv("act.vc.f", "w.tok.fsq.w", t2, d, FSQ_DIM);
    (c.bias, c.output_activation) = (Some(TensorRef::Named("w.tok.fsq.b")), Activation::Tanh);
    let z = s.conv(x, c)?;
    s.scale_shift(z, t2, FSQ_DIM, 0.999_000_013, 0.0)?;
    let mut c = conv("act.vc.h", "w.tok.fsq.E", t2, FSQ_DIM, FSQ_CODES);
    c.bias = Some(TensorRef::Named("w.tok.fsq.Eb"));
    let logits = s.conv(z, c)?;
    s.raw(DevOp::ArgmaxF32, u64::from(t2), tokens, |di| {
        di.t[..2].copy_from_slice(&[tokens, logits]);
        di.i[..2].copy_from_slice(&[t2, FSQ_CODES]);
    })?;
    Ok(t2)
}

const CAMP_SEG: u32 = 100;
const CAMP_GROWTH: u32 = 32;
const CAMP_BN: u32 = 128;

impl Seq {
    /// `op` over each 100-row segment (`seg_rows` 0 = all rows) of the valid rows of `x`.
    #[allow(clippy::too_many_arguments)]
    fn reduce_rows(
        &mut self,
        x: u32,
        out: u32,
        lengths: u32,
        rows: u32,
        width: u32,
        op: ReduceRowsOp,
        seg_rows: u32,
        stride: u32,
    ) -> Result<u32, String> {
        let dep = self.last;
        let e = self.p().reduce_rows_f32(
            x,
            &[dep],
            ReduceRowsF32Stage {
                output: TensorRef::Handle(out),
                lengths: Some(lengths),
                items: 1,
                rows,
                width,
                op,
                seg_rows,
                stride,
                col0: 0,
            },
        )?;
        Ok(self.done(e))
    }

    /// `min(clamp(floor((x * mul + add) / div), lo, hi), min_with)` into `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn len_affine(
        &mut self,
        x: u32,
        out: &str,
        (mul, add, div): (u32, i32, u32),
        (lo, hi): (u32, u32),
        min_with: Option<u32>,
    ) -> Result<u32, String> {
        let dep = self.last;
        let e = self.p().length_affine_u32(
            x,
            &[dep],
            LengthAffineU32Stage {
                output: TensorRef::Named(out),
                min_with,
                items: 1,
                mul,
                add,
                div,
                lo,
                hi,
            },
        )?;
        Ok(self.done(e))
    }

    fn copy_cols(
        &mut self,
        x: u32,
        out: &str,
        rows: u32,
        cols: u32,
        (in_stride, in_offset): (u32, u32),
        (out_stride, out_offset): (u32, u32),
    ) -> Result<u32, String> {
        let dep = self.last;
        let e = self.p().copy_cols_f32(
            x,
            &[dep],
            CopyColsF32Stage {
                output: TensorRef::Named(out),
                items: 1,
                rows,
                cols,
                in_item_stride: 0,
                in_stride,
                in_offset,
                out_item_stride: 0,
                out_stride,
                out_offset,
            },
        )?;
        Ok(self.done(e))
    }
}

/// CAMPPlus after the FCM: `fcm` (`[t][320]`, valid rows `len_t`) -> TDNN (stride 2, valid rows
/// `len_h`) -> 3 CAM dense blocks with transits -> stats pool -> the 192-wide x-vector in
/// `act.vc.xvec`. `zeros` is a zero tensor of at least one concat buffer.
pub(crate) fn camp_xvector(
    s: &mut Seq,
    fcm: u32,
    t: u32,
    len_t: u32,
    len_h: u32,
    layers: &[u32; 3],
    dils: &[u32; 3],
) -> Result<u32, String> {
    let h = halve(t);
    let mut c = conv("act.vc.y", "w.camp.tdnn.w", t, 320, CAMP_BN);
    (c.bias, c.kernel, c.stride, c.pad_before, c.pad_after) =
        (Some(TensorRef::Named("w.camp.tdnn.b")), 5, 2, 2, 2);
    (c.lengths, c.output_activation) = (Some(len_t), Activation::Relu);
    let mut x = s.conv(fcm, c)?;
    let mut cin = CAMP_BN;
    for (blk, (&n, &dil)) in layers.iter().zip(dils).enumerate() {
        let blk = blk + 1;
        let width = cin + n * CAMP_GROWTH;
        // The block's concat buffer [h][width]: the input, then each layer's 32 columns (zero
        // until written; the zero-padded layer params ignore the columns past their input).
        s.copy_cols(x, "act.vc.cat", h, cin, (cin, 0), (width, 0))?;
        let zero = s.p().declare("c.vc.zero", 4096);
        let dep = s.last;
        let e = s.p().copy_cols_f32(
            zero,
            &[dep],
            CopyColsF32Stage {
                output: TensorRef::Named("act.vc.cat"),
                items: 1,
                rows: h,
                cols: width - cin,
                in_item_stride: 0,
                in_stride: 0,
                in_offset: 0,
                out_item_stride: 0,
                out_stride: width,
                out_offset: cin,
            },
        )?;
        let cat = s.done(e);
        for i in 0..n {
            let w = |k: &str| format!("w.camp.d{blk}.{i}.{k}");
            let (bs, bt) = (w("bn1.s"), w("bn1.t"));
            // Pre-activation BN + ReLU: a depthwise k1 conv (weight = scale, bias = shift).
            let mut c = conv("act.vc.h", &bs, h, width, width);
            (c.bias, c.groups, c.output_activation) =
                (Some(TensorRef::Named(&bt)), width, Activation::Relu);
            c.lengths = Some(len_h);
            let a = s.conv(cat, c)?;
            let (lw, lb) = (w("l1.w"), w("l1.b"));
            let mut c = conv("act.vc.b", &lw, h, width, CAMP_BN);
            (c.bias, c.output_activation, c.lengths) =
                (Some(TensorRef::Named(&lb)), Activation::Relu, Some(len_h));
            let b = s.conv(a, c)?;
            let lo = w("local.w");
            let mut c = conv("act.vc.y", &lo, h, CAMP_BN, CAMP_GROWTH);
            (
                c.kernel,
                c.dilation_or_output_padding,
                c.pad_before,
                c.pad_after,
            ) = (3, dil, dil, dil);
            c.lengths = Some(len_h);
            let y = s.conv(b, c)?;
            // CAM context per 100-row segment: mean over the valid rows + the segment's mean.
            let segs = h.div_ceil(CAMP_SEG);
            let ctx = s.p().declare("act.vc.ctx", u64::from(segs * CAMP_BN) * 4);
            let g = s.p().declare("act.vc.gm", u64::from(CAMP_BN) * 4);
            s.reduce_rows(b, ctx, len_h, h, CAMP_BN, ReduceRowsOp::Mean, CAMP_SEG, 0)?;
            s.reduce_rows(b, g, len_h, h, CAMP_BN, ReduceRowsOp::Mean, 0, 0)?;
            let dep = s.last;
            let e = s.p().binary_f32(
                ctx,
                g,
                &[dep],
                crate::pipeline::BinaryF32Stage {
                    output: TensorRef::Handle(ctx),
                    op: BinaryOp::Add,
                    items: 1,
                    rows: segs,
                    width: CAMP_BN,
                    b_item_stride: 0,
                    b_row_stride: 0,
                    b_col_stride: 1,
                    scale: None,
                },
            )?;
            s.done(e);
            let (c1w, c1b, c2w, c2b) = (w("cam1.w"), w("cam1.b"), w("cam2.w"), w("cam2.b"));
            let mut c = conv("act.vc.m1", &c1w, segs, CAMP_BN, CAMP_BN / 2);
            (c.bias, c.output_activation) = (Some(TensorRef::Named(&c1b)), Activation::Relu);
            let m1 = s.conv(ctx, c)?;
            let mut c = conv("act.vc.m2", &c2w, segs, CAMP_BN / 2, CAMP_GROWTH);
            (c.bias, c.output_activation) = (Some(TensorRef::Named(&c2b)), Activation::Sigmoid);
            let m2 = s.conv(m1, c)?;
            let dep = s.last;
            let e = s.p().gather_rows_f32(
                &[dep],
                crate::pipeline::GatherRowsF32Stage {
                    output: TensorRef::Named("act.vc.m"),
                    table: TensorRef::Handle(m2),
                    table_rows: segs,
                    index: None,
                    rows: h,
                    width: CAMP_GROWTH,
                    vocab: segs,
                    rows_per_item: h,
                    repeat: CAMP_SEG,
                    index_item_stride: 0,
                    table_item_stride: 0,
                    table_f16: false,
                    accumulate: false,
                    out_stride: 0,
                    out_col0: 0,
                },
            )?;
            let m = s.done(e);
            s.binary(y, m, BinaryOp::Mul, h, CAMP_GROWTH)?;
            s.copy_cols(
                y,
                "act.vc.cat",
                h,
                CAMP_GROWTH,
                (CAMP_GROWTH, 0),
                (width, cin + i * CAMP_GROWTH),
            )?;
        }
        // Transit: BN + ReLU, 1x1 (no bias) to half the channels.
        let (bs, bt, tw) = (
            format!("w.camp.t{blk}.bn.s"),
            format!("w.camp.t{blk}.bn.t"),
            format!("w.camp.t{blk}.w"),
        );
        let mut c = conv("act.vc.h", &bs, h, width, width);
        (c.bias, c.groups, c.output_activation) =
            (Some(TensorRef::Named(&bt)), width, Activation::Relu);
        c.lengths = Some(len_h);
        let a = s.conv(cat, c)?;
        let mut c = conv("act.vc.y", &tw, h, width, width / 2);
        c.lengths = Some(len_h);
        x = s.conv(a, c)?;
        cin = width / 2;
    }
    let mut c = conv("act.vc.h", "w.camp.out.bn.s", h, cin, cin);
    (c.bias, c.groups, c.output_activation) = (
        Some(TensorRef::Named("w.camp.out.bn.t")),
        cin,
        Activation::Relu,
    );
    c.lengths = Some(len_h);
    let a = s.conv(x, c)?;
    let stats = s.p().declare("act.vc.stats", u64::from(2 * cin) * 4);
    s.reduce_rows(a, stats, len_h, h, cin, ReduceRowsOp::MeanStd, 0, 0)?;
    let mut c = conv("act.vc.xvec", "w.camp.dense.w", 1, 2 * cin, 192);
    c.bias = Some(TensorRef::Named("w.camp.dense.b"));
    s.conv(stats, c)
}

const VE_MELS: u32 = 40;
const VE_HIDDEN: u32 = 256;
const VE_FRAMES: u32 = 160;

/// The VoiceEncoder over `parts` (`[b][160][40]`, the clip's partials; `n_parts` u32 valid count)
/// -> the L2-normalized mean of the L2-normalized partial embeddings in `out` (`[256]`).
/// Each LSTM layer runs over a time-major `[161][b][in + 256]` buffer of `[x_t | h_{t-1}]`: one
/// gate GEMM and one cell per step, the cell writing `h_t` into the next step's row block.
pub(crate) fn ve(
    s: &mut Seq,
    parts: u32,
    b: u32,
    n_parts: u32,
    layers: u32,
    out: u32,
) -> Result<(), String> {
    let hdim = VE_HIDDEN;
    let zero = s.p().declare("c.vc.zero", 4096);
    // Layer input: the partials (item-major), then the previous layer's buffer, whose h_t sits
    // in row block t + 1 at column `prev_cin`.
    let (mut src, mut prev_cin) = (parts, None::<u32>);
    let mut cin = VE_MELS;
    for l in 0..layers {
        let w = cin + hdim;
        let name = if l % 2 == 0 { "act.vc.z0" } else { "act.vc.z1" };
        let z = s.p().declare(name, u64::from((VE_FRAMES + 1) * b * w) * 4);
        let (in_item, in_row, in_off) = match prev_cin {
            None => (VE_FRAMES * cin, cin, 0),
            Some(pc) => (pc + hdim, b * (pc + hdim), b * (pc + hdim) + pc),
        };
        let dep = s.last;
        let e = s.p().copy_cols_f32(
            src,
            &[dep],
            CopyColsF32Stage {
                output: TensorRef::Handle(z),
                items: b,
                rows: VE_FRAMES,
                cols: cin,
                in_item_stride: in_item,
                in_stride: in_row,
                in_offset: in_off,
                out_item_stride: w,
                out_stride: b * w,
                out_offset: 0,
            },
        )?;
        s.done(e);
        // h_{-1} = 0, c = 0.
        let zcopy = |s: &mut Seq, out: u32, stride: u32, offset: u32| -> Result<u32, String> {
            let dep = s.last;
            let e = s.p().copy_cols_f32(
                zero,
                &[dep],
                CopyColsF32Stage {
                    output: TensorRef::Handle(out),
                    items: 1,
                    rows: b,
                    cols: hdim,
                    in_item_stride: 0,
                    in_stride: 0,
                    in_offset: 0,
                    out_item_stride: 0,
                    out_stride: stride,
                    out_offset: offset,
                },
            )?;
            Ok(s.done(e))
        };
        zcopy(s, z, w, cin)?;
        let c = s.p().declare("act.vc.c", u64::from(b * hdim) * 4);
        zcopy(s, c, hdim, 0)?;
        let gates = s.p().declare("act.vc.gates", u64::from(b * 4 * hdim) * 4);
        let wt = s
            .p()
            .declare(&format!("w.ve.l{l}.w"), u64::from(4 * hdim * w) * 4);
        let bias = s
            .p()
            .declare(&format!("w.ve.l{l}.b"), u64::from(4 * hdim) * 4);
        for t in 0..VE_FRAMES {
            let units = u64::from(b.div_ceil(64) * (4 * hdim).div_ceil(64));
            s.raw(DevOp::DenseGemmF32, units, gates, |d| {
                d.t[..4].copy_from_slice(&[gates, z, wt, bias]);
                d.i[..8].copy_from_slice(&[b, 4 * hdim, w, 0, t * b, 0, 0, 1 << 4]);
            })?;
            // c in place; h_t into row block t + 1 at column cin.
            s.raw(DevOp::LstmCellF32, u64::from(b), z, |d| {
                d.t[..4].copy_from_slice(&[z, c, gates, c]);
                d.i[..4].copy_from_slice(&[hdim, b, w, (t + 1) * b * w + cin]);
            })?;
        }
        (src, prev_cin, cin) = (z, Some(cin), hdim);
    }
    let pc = prev_cin.unwrap_or(VE_MELS);
    let w = pc + hdim;
    // The last step's h of each partial: proj + ReLU, L2 norm, masked mean, L2 norm.
    let hl = s.copy_cols(
        src,
        "act.vc.hl",
        b,
        hdim,
        (w, VE_FRAMES * b * w + pc),
        (hdim, 0),
    )?;
    let mut c = conv("act.vc.pe", "w.ve.proj.w", b, hdim, hdim);
    (c.bias, c.output_activation) = (Some(TensorRef::Named("w.ve.proj.b")), Activation::Relu);
    let pe = s.conv(hl, c)?;
    let inv = s.p().declare("c.vc.inv16", u64::from(hdim) * 4);
    for (x, rows) in [(pe, b), (out, 1)] {
        if x == out {
            s.reduce_rows(pe, out, n_parts, b, hdim, ReduceRowsOp::Mean, 0, 0)?;
        }
        s.raw(DevOp::RmsNormF32, u64::from(rows).div_ceil(8), x, |d| {
            d.t[..3].copy_from_slice(&[x, x, inv]);
            d.i[..6].copy_from_slice(&[rows, 1, hdim, 0, 1, 0]);
            d.f[0] = 0.0;
        })?;
    }
    Ok(())
}

const T3_DIM: u32 = 1024;
const COND_QUERIES: u32 = 32;
const COND_HEADS: u32 = 4;

/// T3CondEnc: `spk` (`[256]`), the prompt `tokens` (`[cap]`, `n_tok` valid) and `exaggeration`
/// (`[1]`) -> `rows` (`[34][1024]`: speaker, 32 perceiver rows, emotion).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cond(
    s: &mut Seq,
    spk: u32,
    tokens: u32,
    cap: u32,
    n_tok: u32,
    vocab: u32,
    exaggeration: u32,
    rows: u32,
) -> Result<(), String> {
    let d = T3_DIM;
    let dep = s.last;
    let e = s.p().gather_rows_f32(
        &[dep],
        crate::pipeline::GatherRowsF32Stage {
            output: TensorRef::Named("act.vc.e"),
            table: TensorRef::Named("w.cond.speech_emb"),
            table_rows: vocab,
            index: Some(tokens),
            rows: cap,
            width: d,
            vocab,
            rows_per_item: cap,
            repeat: 1,
            index_item_stride: cap,
            table_item_stride: 0,
            table_f16: false,
            accumulate: false,
            out_stride: 0,
            out_col0: 0,
        },
    )?;
    s.done(e);
    let dep = s.last;
    let e = s.p().gather_rows_f32(
        &[dep],
        crate::pipeline::GatherRowsF32Stage {
            output: TensorRef::Named("act.vc.e"),
            table: TensorRef::Named("w.cond.speech_pos"),
            table_rows: cap,
            index: None,
            rows: cap,
            width: d,
            vocab: cap,
            rows_per_item: cap,
            repeat: 1,
            index_item_stride: 0,
            table_item_stride: 0,
            table_f16: false,
            accumulate: true,
            out_stride: 0,
            out_col0: 0,
        },
    )?;
    let emb = s.done(e);
    // The perceiver's one attention block, as cross-attention (32 queries over the prompt) then
    // self-attention: x1 + proj(attn(q(ln x1), k(ln x2), v(ln x2))).
    let query = s
        .p()
        .declare("w.cond.query", u64::from(COND_QUERIES * d) * 4);
    let mut x1 = query;
    let (mut x2, mut kv_rows, mut klen) = (emb, cap, Some(n_tok));
    for pass in 0..2 {
        let a1 = s.layer_norm(x1, "act.vc.a1", COND_QUERIES, d, "w.cond.ln")?;
        let mut c = conv("act.vc.q", "w.cond.qkv.w", COND_QUERIES, d, 3 * d);
        c.bias = Some(TensorRef::Named("w.cond.qkv.b"));
        let q = s.conv(a1, c)?;
        let kv = if pass == 0 {
            let a2 = s.layer_norm(x2, "act.vc.a2", kv_rows, d, "w.cond.ln")?;
            let mut c = conv("act.vc.kv", "w.cond.qkv.w", kv_rows, d, 3 * d);
            c.bias = Some(TensorRef::Named("w.cond.qkv.b"));
            s.conv(a2, c)?
        } else {
            q
        };
        let dep = s.last;
        let e = s.p().attention_f32(
            q,
            kv,
            kv,
            &[dep],
            AttentionF32Stage {
                output: TensorRef::Named("act.vc.att"),
                key_lengths: klen,
                bias: None,
                batch: 1,
                q_rows: COND_QUERIES,
                kv_rows,
                heads: COND_HEADS,
                head_width: d / COND_HEADS,
                in_stride: 3 * d,
                k_col0: d,
                v_col0: 2 * d,
                causal: false,
                scale: 1.0 / ((d / COND_HEADS) as f32).sqrt(),
                bias_head_stride: 0,
                relative: false,
                key_length_heads: 0,
                prefix: None,
                window: 0,
                stream: None,
            },
        )?;
        let att = s.done(e);
        let out = if pass == 0 {
            "act.vc.pre"
        } else {
            "act.vc.post"
        };
        let mut c = conv(out, "w.cond.o.w", COND_QUERIES, d, d);
        (c.bias, c.residual) = (Some(TensorRef::Named("w.cond.o.b")), Some(x1));
        x1 = s.conv(att, c)?;
        (x2, kv_rows, klen) = (x1, COND_QUERIES, None);
    }
    let mut c = conv("act.vc.spkr", "w.cond.spkr.w", 1, VE_HIDDEN, d);
    c.bias = Some(TensorRef::Named("w.cond.spkr.b"));
    let sp = s.conv(spk, c)?;
    let n = COND_QUERIES + 2;
    let dep = s.last;
    let e = s.p().copy_cols_f32(
        sp,
        &[dep],
        CopyColsF32Stage {
            output: TensorRef::Handle(rows),
            items: 1,
            rows: 1,
            cols: d,
            in_item_stride: 0,
            in_stride: d,
            in_offset: 0,
            out_item_stride: 0,
            out_stride: d,
            out_offset: 0,
        },
    )?;
    s.done(e);
    let dep = s.last;
    let e = s.p().copy_cols_f32(
        x1,
        &[dep],
        CopyColsF32Stage {
            output: TensorRef::Handle(rows),
            items: 1,
            rows: COND_QUERIES,
            cols: d,
            in_item_stride: 0,
            in_stride: d,
            in_offset: 0,
            out_item_stride: 0,
            out_stride: d,
            out_offset: d,
        },
    )?;
    s.done(e);
    // Emotion row = exaggeration * w (the input scalar broadcast).
    let emo = s.p().declare("w.cond.emo.w", u64::from(d) * 4);
    let r = s.copy_cols(emo, "act.vc.emo", 1, d, (d, 0), (d, 0))?;
    let dep = s.last;
    let e = s.p().binary_f32(
        r,
        exaggeration,
        &[dep],
        crate::pipeline::BinaryF32Stage {
            output: TensorRef::Handle(r),
            op: BinaryOp::Mul,
            items: 1,
            rows: 1,
            width: d,
            b_item_stride: 0,
            b_row_stride: 0,
            b_col_stride: 0,
            scale: None,
        },
    )?;
    s.done(e);
    let dep = s.last;
    let e = s.p().copy_cols_f32(
        r,
        &[dep],
        CopyColsF32Stage {
            output: TensorRef::Handle(rows),
            items: 1,
            rows: 1,
            cols: d,
            in_item_stride: 0,
            in_stride: d,
            in_offset: 0,
            out_item_stride: 0,
            out_stride: d,
            out_offset: (n - 1) * d,
        },
    )?;
    s.done(e);
    Ok(())
}

impl Seq {
    fn unary_p(
        &mut self,
        x: u32,
        rows: u32,
        width: u32,
        kind: Activation,
        p: (f32, f32),
    ) -> Result<u32, String> {
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
                p0: p.0,
                p1: p.1,
            },
        )?;
        Ok(self.done(e))
    }

    /// `|STFT|^2` of `x` (valid samples `len`) through `basis` (`[2 bins][1][n_fft]`, re | im):
    /// `[frames][2 bins]`, squared in place.
    #[allow(clippy::too_many_arguments)]
    fn power_stft(
        &mut self,
        x: u32,
        out: &str,
        samples: u32,
        len: u32,
        basis: &str,
        (n_fft, hop, pad): (u32, u32, u32),
        bins2: u32,
        mode: PadMode,
    ) -> Result<(u32, u32), String> {
        let mut c = conv(out, basis, samples, 1, bins2);
        (
            c.kernel,
            c.stride,
            c.pad_before,
            c.pad_after,
            c.pad_mode,
            c.lengths,
        ) = (n_fft, hop, pad, pad, mode, Some(len));
        let spec = self.conv(x, c)?;
        let frames = (samples + 2 * pad - n_fft) / hop + 1;
        let dep = self.last;
        let e = self.p().binary_f32(
            spec,
            spec,
            &[dep],
            crate::pipeline::BinaryF32Stage {
                output: TensorRef::Handle(spec),
                op: BinaryOp::Mul,
                items: 1,
                rows: frames,
                width: bins2,
                b_item_stride: 0,
                b_row_stride: bins2,
                b_col_stride: 1,
                scale: None,
            },
        )?;
        self.done(e);
        Ok((spec, frames))
    }

    /// `x = max(x, max(x over its valid rows and all columns) - 8)`, then `(x + 4) / 4`.
    fn whisper_shape(&mut self, x: u32, rows: u32, len: u32, width: u32) -> Result<u32, String> {
        let m = self.p().declare("act.vc.gmax", u64::from(width) * 4);
        self.reduce_rows(x, m, len, rows, width, ReduceRowsOp::Max, 0, 0)?;
        let g = self.p().declare("act.vc.gmax1", 4);
        let dep = self.last;
        let e = self.p().reduce_rows_f32(
            m,
            &[dep],
            ReduceRowsF32Stage {
                output: TensorRef::Handle(g),
                lengths: None,
                items: 1,
                rows: width,
                width: 1,
                op: ReduceRowsOp::Max,
                seg_rows: 0,
                stride: 1,
                col0: 0,
            },
        )?;
        self.done(e);
        self.unary_p(g, 1, 1, Activation::ScaleShift, (1.0, -8.0))?;
        let dep = self.last;
        let e = self.p().binary_f32(
            x,
            g,
            &[dep],
            crate::pipeline::BinaryF32Stage {
                output: TensorRef::Handle(x),
                op: BinaryOp::Max,
                items: 1,
                rows,
                width,
                b_item_stride: 0,
                b_row_stride: 0,
                b_col_stride: 0,
                scale: None,
            },
        )?;
        self.done(e);
        self.unary_p(x, rows, width, Activation::ScaleShift, (0.25, 1.0))
    }

    /// S3Tokenizer log-mel `[frames][128]` of 16 kHz `x` (valid `len` samples).
    fn s3tok_mel(
        &mut self,
        x: u32,
        out: &str,
        samples: u32,
        len: u32,
        frames_len: u32,
    ) -> Result<(u32, u32), String> {
        let (p, frames) = self.power_stft(
            x,
            "act.vc.spec",
            samples,
            len,
            "w.fe.stft400.w",
            (400, 160, 200),
            402,
            PadMode::Reflect,
        )?;
        let mut c = conv(out, "w.fe.mel128x2", frames, 402, TOK_MELS);
        c.lengths = Some(frames_len);
        let m = self.conv(p, c)?;
        self.unary_p(m, frames, TOK_MELS, Activation::Clamp, (1e-10, f32::MAX))?;
        self.unary_p(m, frames, TOK_MELS, Activation::Log, (0.0, 0.0))?;
        self.unary_p(
            m,
            frames,
            TOK_MELS,
            Activation::ScaleShift,
            (std::f32::consts::LOG10_E, 0.0),
        )?;
        self.whisper_shape(m, frames, frames_len, TOK_MELS)?;
        Ok((m, frames))
    }
}

impl Seq {
    /// s3gen prompt mels `[frames][80]` of 24 kHz `x` (valid `len`): ln(clamp(mel(sqrt(|X|^2 +
    /// 1e-9)), 1e-5)), Matcha's n_fft 1920 / hop 480 with a 720-sample reflect pad.
    fn prompt_mel(
        &mut self,
        x: u32,
        samples: u32,
        len: u32,
        frames_len: u32,
    ) -> Result<(u32, u32), String> {
        let (p, frames) = self.power_stft(
            x,
            "act.vc.spec",
            samples,
            len,
            "w.fe.stft1920.w",
            (1920, 480, 720),
            1922,
            PadMode::Reflect,
        )?;
        let mut c = conv("act.vc.mag", "w.fe.pair961", frames, 1922, 961);
        c.lengths = Some(frames_len);
        let m = self.conv(p, c)?;
        self.unary_p(m, frames, 961, Activation::ScaleShift, (1.0, 1e-9))?;
        self.unary_p(m, frames, 961, Activation::Log, (0.0, 0.0))?;
        self.unary_p(m, frames, 961, Activation::ScaleShift, (0.5, 0.0))?;
        self.unary_p(m, frames, 961, Activation::Exp, (0.0, 0.0))?;
        let mut c = conv("act.vc.pmel", "w.fe.mel80", frames, 961, 80);
        c.lengths = Some(frames_len);
        let mel = self.conv(m, c)?;
        self.unary_p(mel, frames, 80, Activation::Clamp, (1e-5, f32::MAX))?;
        self.unary_p(mel, frames, 80, Activation::Log, (0.0, 0.0))?;
        Ok((mel, frames))
    }

    /// Kaldi fbank (80 bins, mean-normalized over the valid frames, zero past them) of 16 kHz `x`.
    fn fbank(
        &mut self,
        x: u32,
        samples: u32,
        len: u32,
        frames_len: u32,
    ) -> Result<(u32, u32), String> {
        let (p, frames) = self.power_stft(
            x,
            "act.vc.spec",
            samples,
            len,
            "w.fe.kaldi.w",
            (400, 160, 0),
            514,
            PadMode::Zero,
        )?;
        let mut c = conv("act.vc.fb", "w.fe.kaldimelx2", frames, 514, 80);
        c.lengths = Some(frames_len);
        let fb = self.conv(p, c)?;
        self.unary_p(fb, frames, 80, Activation::Clamp, (f32::EPSILON, f32::MAX))?;
        self.unary_p(fb, frames, 80, Activation::Log, (0.0, 0.0))?;
        let m = self.p().declare("act.vc.fbm", 80 * 4);
        self.reduce_rows(fb, m, frames_len, frames, 80, ReduceRowsOp::Mean, 0, 0)?;
        let dep = self.last;
        let e = self.p().binary_f32(
            fb,
            m,
            &[dep],
            crate::pipeline::BinaryF32Stage {
                output: TensorRef::Handle(fb),
                op: BinaryOp::Sub,
                items: 1,
                rows: frames,
                width: 80,
                b_item_stride: 0,
                b_row_stride: 0,
                b_col_stride: 1,
                scale: None,
            },
        )?;
        self.done(e);
        let out = self.mask_rows(fb, "act.vc.fbank", frames, 80, frames_len)?;
        Ok((out, frames))
    }

    /// `x` with the rows past `len` zeroed (a depthwise identity conv over the valid rows).
    fn mask_rows(
        &mut self,
        x: u32,
        out: &str,
        rows: u32,
        width: u32,
        len: u32,
    ) -> Result<u32, String> {
        let mut c = conv(out, "c.vc.ones", rows, width, width);
        (c.groups, c.lengths) = (width, Some(len));
        self.conv(x, c)
    }
}

pub const PACKET: &str = "voice_cond.pkt";
pub const PIPELINE: &str = "voice.cond";
pub const DRIVER: &str = "voice.encode.v1";

const SR24: u32 = 24000;
/// Clip capacities (seconds at 24 kHz): s3gen sees the first 10 s, T3 the first 6 s (16 kHz),
/// the VoiceEncoder the whole clip.
pub const CAPACITIES: &[u32] = &[5, 10, 20];
const GEN_SECONDS: u32 = 10;
const T3_SAMPLES16: u32 = 96000;
const T3_TOKENS: u32 = 150;

struct Cfg {
    rs16_width: u32,
    rs16s_width: u32,
    camp_layers: [u32; 3],
    camp_dilations: [u32; 3],
    ve_layers: u32,
    speech_vocab: u32,
}

impl Cfg {
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let err = |k: &str| format!("voice cond config: {k}");
        let u = |k: &str| -> Result<u32, String> {
            v[k].as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .ok_or(err(k))
        };
        let three = |k: &str| -> Result<[u32; 3], String> {
            let l: Vec<u32> = v[k]
                .as_array()
                .ok_or(err(k))?
                .iter()
                .filter_map(|x| x.as_u64().map(|x| x as u32))
                .collect();
            <[u32; 3]>::try_from(l).map_err(|_| err(k))
        };
        if v["model_type"] != "chatterbox_voice_cond"
            || v["tok_dim"] != TOK_DIM
            || v["tok_heads"] != TOK_HEADS
            || v["tok_blocks"] != 6
            || v["cond_prompt_tokens"] != T3_TOKENS
        {
            return Err("unsupported Chatterbox voice-conditioning export".into());
        }
        Ok(Cfg {
            rs16_width: u("rs16_width")?,
            rs16s_width: u("rs16s_width")?,
            camp_layers: three("camp_layers")?,
            camp_dilations: three("camp_dilations")?,
            ve_layers: u("ve_layers")?,
            speech_vocab: u("speech_vocab")?,
        })
    }
}

/// Per-capacity row counts.
#[derive(Clone, Copy)]
struct Dims {
    s24: u32,
    s24d: u32,
    /// 16 kHz samples of the torchaudio (first 10 s) and soxr-like (whole clip) streams.
    s16: u32,
    s16s: u32,
    t3_16: u32,
    t3_mel: u32,
    gen_mel: u32,
    pmel: u32,
    fbank: u32,
}

impl Dims {
    fn of(seconds: u32) -> Self {
        let s24 = seconds * SR24;
        let s24d = s24.min(GEN_SECONDS * SR24);
        let (s16, s16s) = (2 * (s24d / 3 + 1), 2 * (s24 / 3 + 1));
        let t3_16 = s16s.min(T3_SAMPLES16);
        Dims {
            s24,
            s24d,
            s16,
            s16s,
            t3_16,
            t3_mel: (t3_16 / 160 + 1).min(4 * T3_TOKENS),
            gen_mel: s16 / 160 + 1,
            pmel: (s24d + 2 * 720 - 1920) / 480 + 1,
            fbank: (s16 - 400) / 160 + 1,
        }
    }
}

/// The input and output tensors, shared by every capacity.
#[derive(Clone, Copy)]
struct Io {
    pcm: u32,
    n: u32,
    exaggeration: u32,
    t3_tokens: u32,
    t3_count: u32,
    gen_tokens: u32,
    prompt_len: u32,
    prompt_mel: u32,
    spks: u32,
    rows: u32,
}

pub fn lower_voice_cond(
    dir: &std::path::Path,
    n_cu: u32,
    target: u32,
) -> Result<(packet::devbuild::Model, packet::devbuild::SectionData), String> {
    let cfg = Cfg::parse(
        &serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .map_err(|e| format!("{}: {e}", dir.display()))?,
        )
        .map_err(|e| format!("voice cond config: {e}"))?,
    )?;
    let big = Dims::of(*CAPACITIES.last().unwrap());
    let mut builder = packet::devbuild::Builder::new(n_cu);
    builder.set_tensor_dedup(true);
    let io = Io {
        pcm: builder.tensor("in.vc.pcm", u64::from(big.s24) * 4),
        n: builder.tensor("in.vc.samples", 4),
        exaggeration: builder.tensor("in.vc.exaggeration", 4),
        t3_tokens: builder.tensor("out.vc.t3_tokens", u64::from(T3_TOKENS) * 4),
        t3_count: builder.tensor("out.vc.t3_count", 4),
        gen_tokens: builder.tensor(
            "out.vc.gen_tokens",
            u64::from(halve(halve(big.gen_mel))) * 4,
        ),
        prompt_len: builder.tensor("out.vc.prompt_len", 4),
        prompt_mel: builder.tensor("out.vc.prompt_mel", u64::from(big.pmel * 80) * 4),
        spks: builder.tensor("out.vc.spks", 80 * 4),
        rows: builder.tensor("out.vc.rows", u64::from((COND_QUERIES + 2) * T3_DIM) * 4),
    };
    let model = packet::devbuild::Model {
        n_cu,
        target,
        tensors: builder.tensors(),
        progs: Vec::new(),
        kv_row_insts: Vec::new(),
        prog_t: Vec::new(),
        gen: Vec::new(),
    };
    let mut prefix = crate::pipeline::PacketPrefix {
        model,
        programs: Vec::new(),
        input: io.pcm,
        output: io.rows,
        input_shape: vec![u64::from(big.s24)],
    };
    let mut roles = std::collections::BTreeMap::new();
    for &secs in CAPACITIES {
        let d = Dims::of(secs);
        let stages: [(&str, &dyn Fn(&mut Seq) -> Result<(), String>); 3] = [
            ("front", &|s| front(s, &cfg, d, io)),
            ("tok", &|s| tokens(s, d, io)),
            ("camp", &|s| speaker(s, &cfg, d, io)),
        ];
        for (k, (name, body)) in stages.iter().enumerate() {
            let program = prefix.model.progs.len() as u32;
            let mut p = prefix.program();
            // A dependency-free first instruction to chain the stage from.
            let e = p.length_affine_u32(
                io.n,
                &[],
                LengthAffineU32Stage {
                    output: TensorRef::Named("act.vc.start"),
                    min_with: None,
                    items: 1,
                    mul: 1,
                    add: 0,
                    div: 1,
                    lo: 0,
                    hi: u32::MAX,
                },
            )?;
            let mut s = Seq {
                p: Some(p),
                last: e.done,
            };
            body(&mut s)?;
            prefix = s.p.take().unwrap().finish(secs);
            roles.insert(format!("encode.c{secs}.{k}"), program);
            roles.insert(format!("stage.{name}.c{secs}.0"), program);
        }
    }
    let reader = crate::checkpoint::TensorReader::open(dir)?;
    for tensor in prefix.model.tensors.iter_mut() {
        if let Some(c) = constant(&tensor.name, tensor.bytes) {
            tensor.init = Some(c);
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
    let section = section(&prefix, io, big, roles)?;
    Ok((prefix.model, section))
}

/// Packet constants: zeros, ones, RoPE positions, L2-norm gammas.
fn constant(name: &str, bytes: u64) -> Option<Vec<u8>> {
    let n = (bytes / 4) as u32;
    let f = |v: f32| (0..n).flat_map(|_| v.to_le_bytes()).collect();
    match name {
        "c.vc.zero" => Some(vec![0; bytes as usize]),
        "c.vc.ones" => Some(f(1.0)),
        "c.vc.inv16" => Some(f(1.0 / 16.0)),
        "c.vc.inv192" => Some(f(1.0 / 192f32.sqrt())),
        "c.vc.pos" => Some((0..n).flat_map(u32::to_le_bytes).collect()),
        _ => None,
    }
}

/// Valid lengths (u32 scalars) of the front end's streams.
fn front(s: &mut Seq, cfg: &Cfg, d: Dims, io: Io) -> Result<(), String> {
    let max = u32::MAX;
    let l24d = s.len_affine(io.n, "act.vc.l24d", (1, 0, 1), (0, d.s24d), None)?;
    let l16 = s.len_affine(l24d, "act.vc.l16", (2, 2, 3), (0, max), None)?;
    let l16s = s.len_affine(io.n, "act.vc.l16s", (2, 2, 3), (0, max), None)?;
    let l16t3 = s.len_affine(l16s, "act.vc.l16t3", (1, 0, 1), (0, T3_SAMPLES16), None)?;
    let tm3 = s.len_affine(l16t3, "act.vc.tm3", (1, 0, 160), (0, 4 * T3_TOKENS), None)?;
    let tmg = s.len_affine(l16, "act.vc.tmg", (1, 0, 160), (0, max), None)?;
    let tmat = s.len_affine(l24d, "act.vc.tmat", (1, 0, 480), (0, max), None)?;
    s.len_affine(l16, "act.vc.tfb", (1, -240, 160), (0, max), None)?;
    // 24 -> 16 kHz: one stride-3 conv with 2 output channels per stream ([rows][2] interleaved).
    for (w, out, rows, len, width) in [
        ("w.fe.rs16.w", "act.vc.w16", d.s24d, l24d, cfg.rs16_width),
        ("w.fe.rs16s.w", "act.vc.w16s", d.s24, io.n, cfg.rs16s_width),
    ] {
        let mut c = conv(out, w, rows, 1, 2);
        (c.kernel, c.stride, c.pad_before, c.pad_after, c.lengths) =
            (2 * width + 3, 3, width, width + 3, Some(len));
        s.conv(io.pcm, c)?;
    }
    let w16 = s.p().declare("act.vc.w16", u64::from(d.s16) * 4);
    let w16s = s.p().declare("act.vc.w16s", u64::from(d.s16s) * 4);
    let (m3, _) = s.s3tok_mel(w16s, "act.vc.mel_t3", d.t3_16, l16t3, tm3)?;
    let _ = m3;
    s.s3tok_mel(w16, "act.vc.mel_gen", d.s16, l16, tmg)?;
    let (pm, _) = s.prompt_mel(io.pcm, d.s24d, l24d, tmat)?;
    s.copy_cols(pm, "out.vc.prompt_mel", d.pmel, 80, (80, 0), (80, 0))?;
    let tfb = s.p().declare("act.vc.tfb", 4);
    s.fbank(w16, d.s16, l16, tfb)?;
    Ok(())
}

/// Both S3Tokenizer passes: the T3 prompt tokens (first 6 s) and the s3gen prompt tokens
/// (first 10 s), with `P = min(tokens, prompt mel rows / 2)`.
fn tokens(s: &mut Seq, d: Dims, io: Io) -> Result<(), String> {
    let pos = s.p().declare("c.vc.pos", 4096 * 4);
    let max = u32::MAX;
    for (mel, rows, len, out) in [
        ("act.vc.mel_t3", d.t3_mel, "act.vc.tm3", io.t3_tokens),
        ("act.vc.mel_gen", d.gen_mel, "act.vc.tmg", io.gen_tokens),
    ] {
        let mel = s.p().declare(mel, u64::from(rows * TOK_MELS) * 4);
        let l0 = s.p().declare(len, 4);
        let l1 = s.len_affine(l0, "act.vc.tl1", (1, 1, 2), (0, max), None)?;
        let l2 = s.len_affine(l1, "act.vc.tl2", (1, 1, 2), (0, max), None)?;
        s3tok(s, mel, rows, [l0, l1, l2], pos, out)?;
        if out == io.t3_tokens {
            s.len_affine(l2, "out.vc.t3_count", (1, 0, 1), (0, T3_TOKENS), None)?;
        } else {
            let tmat = s.p().declare("act.vc.tmat", 4);
            s.len_affine(tmat, "out.vc.prompt_len", (1, 0, 2), (0, max), Some(l2))?;
        }
    }
    Ok(())
}

/// CAMPPlus x-vector -> the s3gen speaker vector `spk_embed_affine(normalize(x))`.
fn speaker(s: &mut Seq, cfg: &Cfg, d: Dims, io: Io) -> Result<(), String> {
    let max = u32::MAX;
    let tfb = s.p().declare("act.vc.tfb", 4);
    let th = s.len_affine(tfb, "act.vc.th", (1, 1, 2), (0, max), None)?;
    let fcm = s.p().declare("act.vc.fcm", u64::from(d.fbank * 320) * 4);
    let x = camp_xvector(
        s,
        fcm,
        d.fbank,
        tfb,
        th,
        &cfg.camp_layers,
        &cfg.camp_dilations,
    )?;
    let g = s.p().declare("c.vc.inv192", 192 * 4);
    s.raw(DevOp::RmsNormF32, 1, x, |di| {
        di.t[..3].copy_from_slice(&[x, x, g]);
        di.i[..6].copy_from_slice(&[1, 1, 192, 0, 1, 0]);
        di.f[0] = 0.0;
    })?;
    let mut c = conv("out.vc.spks", "w.spks.w", 1, 192, 80);
    c.bias = Some(TensorRef::Named("w.spks.b"));
    let _ = (io, cfg.ve_layers, cfg.speech_vocab);
    s.conv(x, c)?;
    Ok(())
}

fn section(
    prefix: &crate::pipeline::PacketPrefix,
    io: Io,
    big: Dims,
    roles: std::collections::BTreeMap<String, u32>,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::{
        PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION,
    };
    let t = |h: u32| prefix.model.tensors[h as usize].name.clone();
    let tensor = |h: u32, dtype: PipelineDType, n: u32| PipelineTensor {
        name: t(h),
        dtype,
        shape: vec![u64::from(n)],
    };
    let (f, u) = (PipelineDType::F32, PipelineDType::U32);
    let metadata = PacketPipelines {
        version: VERSION,
        pipelines: vec![PacketPipeline {
            name: PIPELINE.into(),
            driver: DRIVER.into(),
            programs: roles,
            tensors: std::collections::BTreeMap::from([
                ("pcm".into(), tensor(io.pcm, f, big.s24)),
                ("samples".into(), tensor(io.n, u, 1)),
                ("exaggeration".into(), tensor(io.exaggeration, f, 1)),
                ("t3.tokens".into(), tensor(io.t3_tokens, u, T3_TOKENS)),
                ("t3.count".into(), tensor(io.t3_count, u, 1)),
                (
                    "t3.rows".into(),
                    tensor(io.rows, f, (COND_QUERIES + 2) * T3_DIM),
                ),
                (
                    "gen.tokens".into(),
                    tensor(io.gen_tokens, u, halve(halve(big.gen_mel))),
                ),
                ("gen.prompt_len".into(), tensor(io.prompt_len, u, 1)),
                (
                    "gen.prompt_mel".into(),
                    tensor(io.prompt_mel, f, big.pmel * 80),
                ),
                ("gen.spks".into(), tensor(io.spks, f, 80)),
            ]),
            parameters: std::collections::BTreeMap::from([
                ("audio.sample_rate".into(), u64::from(SR24)),
                ("t3.rows".into(), u64::from(COND_QUERIES + 2)),
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
