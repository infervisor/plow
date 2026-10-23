//! `UnaryF32` `ACT_LOG`, `ReduceRowsF32`, `LengthAffineU32`, `TrimSilenceF32` and the multi-row
//! `LstmCellF32` / width-strided `Conv2dF32` / head_width 256 `AttentionF32` extensions (golden
//! CPU tier) against their `packet::dev` contracts: masked ceil-mode segments, column windows,
//! every reduction, floor division with negative offsets, clamps and an elementwise bound,
//! independent LSTM / convolution / softmax-attention references and librosa trim indices.
#![cfg(feature = "cpu")]

use std::ffi::c_void;

use packet::dev::{DevOp, ACT_LOG, TENSOR_NONE16};
use plowrt::exec::cpu::ffi::{self, DevInst64, Isa, PlowCpuCtx};

fn op(op: DevOp, t: &[u16], i: &[u32]) -> DevInst64 {
    let mut d = DevInst64 {
        op: op as u16,
        blocks: 3,
        ..Default::default()
    };
    d.t = [TENSOR_NONE16; 8];
    d.t[..t.len()].copy_from_slice(t);
    d.i[..i.len()].copy_from_slice(i);
    d
}

fn run(d: &DevInst64, tensors: &[*mut c_void]) {
    ffi::init(Isa::Scalar).unwrap();
    assert!(ffi::has(d.op));
    let mut ctx = PlowCpuCtx::new(0, 0);
    ffi::thread_init(&mut ctx).unwrap();
    let mut scratch = vec![0u8; (ffi::scratch_bytes() as usize).max(64)];
    ctx.scratch = scratch.as_mut_ptr() as *mut c_void;
    ctx.scratch_bytes = scratch.len() as u32;
    for slice in 0..3 {
        // SAFETY: every handle the op names is a live buffer sized for the geometry.
        unsafe { ffi::exec(d, slice, 3, tensors, &mut ctx) }.unwrap();
    }
}

fn ptr<T>(v: &mut [T]) -> *mut c_void {
    v.as_mut_ptr() as *mut c_void
}

fn data(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as i32) as f32 / 2_147_483_648.0 * 3.0 + 0.5
        })
        .collect()
}

#[test]
fn unary_log_window() {
    let (rows, width, stride, col0) = (5usize, 7usize, 10usize, 2usize);
    let x: Vec<f32> = data(rows * stride, 7)
        .iter()
        .map(|v| v.abs() * 10.0 + 1e-6)
        .collect();
    let mut x = x;
    x[col0] = 1e-10;
    let mut out = vec![-7.0f32; rows * stride];
    let d = op(
        DevOp::UnaryF32,
        &[0, 1],
        &[rows, width, ACT_LOG as usize, stride, col0].map(|v| v as u32),
    );
    run(&d, &[ptr(&mut out), ptr(&mut x)]);
    for r in 0..rows {
        for c in 0..stride {
            let i = r * stride + c;
            if c < col0 || c >= col0 + width {
                assert_eq!(out[i], -7.0);
            } else {
                let want = (x[i] as f64).ln();
                assert!(
                    (out[i] as f64 - want).abs() <= want.abs() * 2e-7 + 1e-7,
                    "[{r}][{c}] {} vs {want}",
                    out[i]
                );
            }
        }
    }
}

struct Reduce {
    items: usize,
    rows: usize,
    width: usize,
    seg_rows: usize,
    stride: usize,
    col0: usize,
}

/// `[items][segs][W or 2W]` in f64, from the op's contract.
fn reduce_ref(g: &Reduce, x: &[f32], lengths: Option<&[u32]>, op: u32) -> Vec<f64> {
    let seg = if g.seg_rows == 0 { g.rows } else { g.seg_rows };
    let stride = if g.stride == 0 { g.width } else { g.stride };
    let segs = g.rows.div_ceil(seg);
    let ow = if op == 4 { 2 * g.width } else { g.width };
    let mut out = vec![0.0; g.items * segs * ow];
    for b in 0..g.items {
        let len = lengths.map_or(g.rows, |l| (l[b] as usize).min(g.rows));
        for s in 0..segs {
            let rows = s * seg..((s + 1) * seg).min(len).max(s * seg);
            let n = rows.len() as f64;
            for c in 0..g.width {
                let v: Vec<f64> = rows
                    .clone()
                    .map(|r| x[(b * g.rows + r) * stride + g.col0 + c] as f64)
                    .collect();
                let sum: f64 = v.iter().sum();
                let mean = if v.is_empty() { 0.0 } else { sum / n };
                let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let std = if v.len() < 2 {
                    0.0
                } else {
                    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)).sqrt()
                };
                let o = &mut out[(b * segs + s) * ow..];
                match op {
                    0 => o[c] = sum,
                    1 => o[c] = mean,
                    2 => o[c] = if v.is_empty() { 0.0 } else { max },
                    3 => o[c] = std,
                    _ => {
                        o[c] = mean;
                        o[g.width + c] = std;
                    }
                }
            }
        }
    }
    out
}

fn check_reduce(g: &Reduce, lengths: Option<Vec<u32>>) {
    let stride = if g.stride == 0 { g.width } else { g.stride };
    let mut x = data(g.items * g.rows * stride, 11);
    let mut lengths = lengths;
    let seg = if g.seg_rows == 0 { g.rows } else { g.seg_rows };
    let segs = g.rows.div_ceil(seg);
    for op_code in 0..5u32 {
        let ow = if op_code == 4 { 2 * g.width } else { g.width };
        let mut out = vec![f32::NAN; g.items * segs * ow];
        let mut t = vec![ptr(&mut out), ptr(&mut x)];
        let mut handles = vec![0u16, 1];
        if let Some(l) = lengths.as_mut() {
            t.push(ptr(l));
            handles.push(2);
        }
        let d = op(
            DevOp::ReduceRowsF32,
            &handles,
            &[
                g.items,
                g.rows,
                g.width,
                op_code as usize,
                g.seg_rows,
                g.stride,
                g.col0,
            ]
            .map(|v| v as u32),
        );
        run(&d, &t);
        let want = reduce_ref(g, &x, lengths.as_deref(), op_code);
        for (i, (&got, &want)) in out.iter().zip(&want).enumerate() {
            assert!(
                (got as f64 - want).abs() <= want.abs() * 1e-6 + 1e-6,
                "op {op_code} [{i}] {got} vs {want}"
            );
        }
    }
}

#[test]
fn reduce_rows_ceil_segments_with_lengths_and_window() {
    // L = 250 (short last segment), 173 (n = 73, then empty), 201 (n = 1: std 0), > rows.
    check_reduce(
        &Reduce {
            items: 4,
            rows: 250,
            width: 6,
            seg_rows: 100,
            stride: 11,
            col0: 3,
        },
        Some(vec![250, 173, 201, 1000]),
    );
}

#[test]
fn reduce_rows_global_with_and_without_lengths() {
    let g = Reduce {
        items: 3,
        rows: 37,
        width: 9,
        seg_rows: 0,
        stride: 0,
        col0: 0,
    };
    check_reduce(&g, Some(vec![37, 0, 5]));
    check_reduce(&g, None);
}

#[test]
fn reduce_rows_invalid_op_writes_nothing() {
    let mut x = data(4 * 3, 3);
    let mut out = vec![-1.0f32; 3];
    let d = op(DevOp::ReduceRowsF32, &[0, 1], &[1, 4, 3, 5]);
    run(&d, &[ptr(&mut out), ptr(&mut x)]);
    assert_eq!(out, [-1.0; 3]);
}

fn affine_ref(x: u32, mul: u32, add: i32, div: u32, lo: u32, hi: u32, bound: Option<u32>) -> u32 {
    let v = i64::from(x) * i64::from(mul) + i64::from(add);
    let q = v
        .div_euclid(i64::from(div))
        .clamp(i64::from(lo), i64::from(hi)) as u32;
    bound.map_or(q, |b| q.min(b))
}

#[test]
fn length_affine_floor_clamp_and_bound() {
    let mut x = vec![0u32, 50, 51, 128, 353, 100_000, 4_000_000_000];
    let mut bound = vec![5u32, 5, 0, 1, 100, 7, u32::MAX];
    let items = x.len();
    // VE partial windows: max(1, floor((T - 51) / 77)), bounded.
    for (mul, add, div, lo, hi, with_bound) in [
        (1u32, -51i32, 77u32, 1u32, u32::MAX, true),
        (1, -51, 77, 1, 1000, false),
        (2, 1, 4, 0, u32::MAX, false),
        (1, -400, 160, 0, 50, true),
        (3, i32::MIN, 1, 0, u32::MAX, false),
    ] {
        let mut out = vec![0xDEADu32; items];
        let mut t = vec![ptr(&mut out), ptr(&mut x)];
        let mut handles = vec![0u16, 1];
        if with_bound {
            t.push(ptr(&mut bound));
            handles.push(2);
        }
        let d = op(
            DevOp::LengthAffineU32,
            &handles,
            &[items as u32, mul, add as u32, div, lo, hi],
        );
        run(&d, &t);
        for b in 0..items {
            let want = affine_ref(x[b], mul, add, div, lo, hi, with_bound.then_some(bound[b]));
            assert_eq!(out[b], want, "item {b} mul {mul} add {add} div {div}");
        }
    }
    assert_eq!(
        affine_ref(353, 1, -51, 77, 1, u32::MAX, None),
        3,
        "T=353 gives 3 VE windows"
    );
    let mut out = vec![0xDEADu32; items];
    let d = op(
        DevOp::LengthAffineU32,
        &[0, 1],
        &[items as u32, 1, 0, 0, 0, u32::MAX],
    );
    run(&d, &[ptr(&mut out), ptr(&mut x)]);
    assert!(out.iter().all(|&v| v == 0xDEAD), "div 0 writes nothing");
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn lstm_rows_strided_h_and_in_place_c() {
    let (width, rows, h_stride, h_offset) = (7usize, 5usize, 20usize, 13usize);
    let gates = data(rows * 4 * width, 21);
    let c_prev = data(rows * width, 22);
    let z0 = data(h_offset + rows * h_stride, 23);
    let (mut z, mut c, mut g) = (z0.clone(), c_prev.clone(), gates.clone());
    let d = op(
        DevOp::LstmCellF32,
        &[0, 1, 2, 1],
        &[width, rows, h_stride, h_offset].map(|v| v as u32),
    );
    run(&d, &[ptr(&mut z), ptr(&mut c), ptr(&mut g)]);
    let mut written = vec![false; z.len()];
    for r in 0..rows {
        for i in 0..width {
            let gate = |k: usize| gates[r * 4 * width + k * width + i] as f64;
            let cell =
                sigmoid(gate(1)) * c_prev[r * width + i] as f64 + sigmoid(gate(0)) * gate(2).tanh();
            let h = sigmoid(gate(3)) * cell.tanh();
            let at = h_offset + r * h_stride + i;
            written[at] = true;
            assert!(
                (c[r * width + i] as f64 - cell).abs() <= 1e-5,
                "c[{r}][{i}]"
            );
            assert!((z[at] as f64 - h).abs() <= 1e-5, "h[{r}][{i}]");
        }
    }
    for (at, &w) in written.iter().enumerate() {
        if !w {
            assert_eq!(z[at], z0[at], "untouched [{at}]");
        }
    }
    // Legacy encoding: one row, h at the base.
    let (mut h, mut c, mut g) = (vec![0.0f32; width], c_prev.clone(), gates.clone());
    let mut cp = c_prev.clone();
    let d = op(DevOp::LstmCellF32, &[0, 1, 2, 3], &[width as u32]);
    run(&d, &[ptr(&mut h), ptr(&mut c), ptr(&mut g), ptr(&mut cp)]);
    for i in 0..width {
        assert!((h[i] - z[h_offset + i]).abs() <= 1e-6, "row 0 [{i}]");
    }
}

#[test]
fn conv2d_width_stride_against_direct_reference() {
    // NHWC [frames][width][ic] in, NHWC out, f32 weights (flag 64); (kernel, pad) 3/1 and 1/0.
    let (batches, frames, width, ic, oc) = (2usize, 6usize, 9usize, 3usize, 4usize);
    for (kernel, pad, stride, ws) in [(3usize, 1usize, 1usize, 2usize), (1, 0, 1, 2), (3, 1, 2, 3)]
    {
        let of = (frames + 2 * pad - kernel) / stride + 1;
        let ow = (width + 2 * pad - kernel) / ws + 1;
        let mut x = data(batches * frames * width * ic, 31);
        let mut w = data(oc * ic * kernel * kernel, 32);
        let mut bias = data(oc, 33);
        let mut out = vec![f32::NAN; batches * of * ow * oc];
        let mut d = op(
            DevOp::Conv2dF32,
            &[0, 1, 2, 3],
            &[frames, width, ic, oc, kernel, stride, pad, pad].map(|v| v as u32),
        );
        d.fj[1] = 64 | (ws as u32) << 16;
        d.fj[2] = batches as u32;
        run(
            &d,
            &[ptr(&mut out), ptr(&mut x), ptr(&mut w), ptr(&mut bias)],
        );
        for b in 0..batches {
            for oy in 0..of {
                for ox in 0..ow {
                    for o in 0..oc {
                        let mut want = bias[o] as f64;
                        for ky in 0..kernel {
                            for kx in 0..kernel {
                                let iy = (oy * stride + ky) as isize - pad as isize;
                                let ix = (ox * ws + kx) as isize - pad as isize;
                                if iy < 0 || ix < 0 || iy >= frames as isize || ix >= width as isize
                                {
                                    continue;
                                }
                                for c in 0..ic {
                                    let xi =
                                        ((b * frames + iy as usize) * width + ix as usize) * ic + c;
                                    let wi = ((o * ic + c) * kernel + ky) * kernel + kx;
                                    want += x[xi] as f64 * w[wi] as f64;
                                }
                            }
                        }
                        let got = out[((b * of + oy) * ow + ox) * oc + o] as f64;
                        assert!(
                            (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
                            "k{kernel} s({stride},{ws}) [{b}][{oy}][{ox}][{o}] {got} vs {want}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn attention_head_width_256_against_reference() {
    let (batch, q_rows, kv_rows, heads, hw) = (2usize, 5usize, 9usize, 2usize, 256usize);
    let width = heads * hw;
    let (stride, k_col0, v_col0) = (2 * width + 3, 1usize, width + 2);
    let mut q = data(batch * q_rows * stride, 41);
    let mut kv = data(batch * kv_rows * stride, 42);
    let mut lengths = vec![9u32, 4];
    let scale = 1.0f32 / 16.0;
    let mut out = vec![f32::NAN; batch * q_rows * width];
    let mut d = op(
        DevOp::AttentionF32,
        &[0, 1, 2, 2, 3],
        &[batch, q_rows, kv_rows, heads, hw, stride, 0, 0].map(|v| v as u32),
    );
    d.fj = [scale.to_bits(), k_col0 as u32, v_col0 as u32];
    run(
        &d,
        &[ptr(&mut out), ptr(&mut q), ptr(&mut kv), ptr(&mut lengths)],
    );
    for b in 0..batch {
        let visible = lengths[b] as usize;
        for h in 0..heads {
            for r in 0..q_rows {
                let qr = &q[(b * q_rows + r) * stride + h * hw..][..hw];
                let row =
                    |j: usize, col0: usize| &kv[(b * kv_rows + j) * stride + col0 + h * hw..][..hw];
                let s: Vec<f64> = (0..visible)
                    .map(|j| {
                        qr.iter()
                            .zip(row(j, k_col0))
                            .map(|(&a, &k)| a as f64 * k as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
                let den: f64 = e.iter().sum();
                for c in 0..hw {
                    let want = (0..visible)
                        .map(|j| e[j] * row(j, v_col0)[c] as f64)
                        .sum::<f64>()
                        / den;
                    let got = out[(b * q_rows + r) * width + h * hw + c] as f64;
                    assert!(
                        (got - want).abs() <= 1e-5,
                        "[{b}][{h}][{r}][{c}] {got} vs {want}"
                    );
                }
            }
        }
    }
}

struct TrimVec {
    seed: u32,
    len_in: u32,
    segs: Vec<(u32, f32)>,
    /// librosa 0.11 `effects.trim(frame_length 2048, hop 512)` `[start, end)` at top_db 20 / 60.
    want20: (u32, u32),
    want60: (u32, u32),
}

fn trim_vectors() -> Vec<TrimVec> {
    let mut ramp: Vec<(u32, f32)> = (0..16)
        .map(|k| (1500 * (k + 1), 2f32.powi(-(k as i32))))
        .collect();
    ramp.push((u32::MAX, 0.0));
    let v = |seed, len_in, segs: &[(u32, f32)], want20, want60| TrimVec {
        seed,
        len_in,
        segs: segs.to_vec(),
        want20,
        want60,
    };
    vec![
        v(
            1,
            20000,
            &[(5000, 0.0), (14000, 0.5), (u32::MAX, 0.001)],
            (4096, 15360),
            (4096, 20000),
        ),
        // All zero: every frame is 0 dB below the (floored) reference, so nothing is trimmed.
        v(2, 7000, &[(u32::MAX, 0.0)], (0, 7000), (0, 7000)),
        v(3, 12345, &[(u32::MAX, 0.25)], (0, 12345), (0, 12345)),
        v(4, 300, &[(u32::MAX, 0.5)], (0, 300), (0, 300)),
        // len_in > cap: clamped to cap.
        v(
            5,
            30000,
            &[(3000, 3e-3), (9000, 0.5), (u32::MAX, 0.0)],
            (2048, 10240),
            (0, 10240),
        ),
        v(6, 24000, &ramp, (0, 6656), (0, 16384)),
        v(
            7,
            22000,
            &[
                (2000, 0.0),
                (4000, 0.3),
                (15000, 0.0),
                (16000, 0.3),
                (u32::MAX, 0.0),
            ],
            (1024, 17408),
            (1024, 17408),
        ),
    ]
}

fn trim_signal(v: &TrimVec, cap: usize) -> Vec<f32> {
    let len = (v.len_in as usize).min(cap);
    let mut st = v.seed;
    let mut si = 0;
    (0..cap)
        .map(|i| {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = ((st >> 8) as i32 - 8388608) as f32 / 8388608.0;
            while i as u32 >= v.segs[si].0 {
                si += 1;
            }
            if i < len {
                u * v.segs[si].1
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn trim_silence_matches_librosa_indices() {
    let cap = 24000usize;
    let vs = trim_vectors();
    let items = vs.len();
    let mut x: Vec<f32> = vs.iter().flat_map(|v| trim_signal(v, cap)).collect();
    let mut len_in: Vec<u32> = vs.iter().map(|v| v.len_in).collect();
    for top_db in [20.0f32, 60.0, 0.0] {
        let mut out = vec![f32::NAN; items * cap];
        let mut len_out = vec![0xDEADu32; items];
        let mut d = op(
            DevOp::TrimSilenceF32,
            &[0, 1, 2, 3],
            &[items as u32, cap as u32, 2048, 512],
        );
        d.fj[0] = top_db.to_bits();
        run(
            &d,
            &[
                ptr(&mut out),
                ptr(&mut x),
                ptr(&mut len_in),
                ptr(&mut len_out),
            ],
        );
        for (b, v) in vs.iter().enumerate() {
            let (start, end) = match top_db as u32 {
                20 => v.want20,
                60 => v.want60,
                _ => (0, 0),
            };
            let n = (end - start) as usize;
            assert_eq!(len_out[b] as usize, n, "item {b} top_db {top_db}");
            let row = &out[b * cap..][..cap];
            let src = &x[b * cap + start as usize..][..n];
            assert_eq!(&row[..n], src, "item {b} top_db {top_db}");
            assert!(row[n..].iter().all(|&s| s == 0.0), "item {b} tail");
        }
    }
}

struct PrefixCase {
    batch: usize,
    q_rows: usize,
    kv_rows: usize,
    heads: usize,
    hw: usize,
    /// i7: rows per prefix block (the block stride).
    pre: usize,
    blocks: usize,
    index: Vec<u32>,
    prefix_rows: Option<Vec<u32>>,
    lengths: Option<Vec<u32>>,
}

/// Runs the golden AttentionF32 with a key prefix; rows nobody may read are NaN.
fn attention_prefix_golden(c: &PrefixCase) -> (Vec<f32>, Vec<f64>) {
    let width = c.heads * c.hw;
    let pre_b = |b: usize| {
        c.prefix_rows
            .as_ref()
            .map_or(c.pre, |r| (r[b] as usize).min(c.pre))
    };
    let klen = |b: usize| {
        let all = pre_b(b) + c.kv_rows;
        c.lengths.as_ref().map_or(all, |l| (l[b] as usize).min(all))
    };
    let mut q = data(c.batch * c.q_rows * width, 51);
    let mut k = data(c.batch * c.kv_rows * width, 52);
    let mut v = data(c.batch * c.kv_rows * width, 53);
    let mut prefix = data(c.blocks * c.pre * 2 * width, 54);
    for blk in 0..c.blocks {
        let used = (0..c.batch)
            .filter(|&b| c.index[b] as usize == blk)
            .map(|b| pre_b(b).min(klen(b)))
            .max()
            .unwrap_or(0);
        for r in used..c.pre {
            prefix[(blk * c.pre + r) * 2 * width..][..2 * width].fill(f32::NAN);
        }
    }
    for b in 0..c.batch {
        let own = klen(b).saturating_sub(pre_b(b));
        for r in own..c.kv_rows {
            k[(b * c.kv_rows + r) * width..][..width].fill(f32::NAN);
            v[(b * c.kv_rows + r) * width..][..width].fill(1e30);
        }
    }
    let mut index = c.index.clone();
    let mut rows = c.prefix_rows.clone().unwrap_or_default();
    let mut lengths = c.lengths.clone().unwrap_or_default();
    let mut out = vec![f32::NAN; c.batch * c.q_rows * width];
    let scale = 0.125f32;
    let flags = if c.prefix_rows.is_some() { 32 } else { 0 };
    let mut d = op(
        DevOp::AttentionF32,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[c.batch, c.q_rows, c.kv_rows, c.heads, c.hw, 0, flags, c.pre].map(|v| v as u32),
    );
    if c.lengths.is_none() {
        d.t[4] = TENSOR_NONE16;
    }
    if c.prefix_rows.is_none() {
        d.t[5] = TENSOR_NONE16;
    }
    d.fj = [scale.to_bits(), 0, 0];
    run(
        &d,
        &[
            ptr(&mut out),
            ptr(&mut q),
            ptr(&mut k),
            ptr(&mut v),
            ptr(&mut lengths),
            ptr(&mut rows),
            ptr(&mut prefix),
            ptr(&mut index),
        ],
    );
    let mut want = vec![0.0f64; out.len()];
    for b in 0..c.batch {
        let (pb, kl) = (pre_b(b), klen(b));
        let blk = c.index[b] as usize;
        for h in 0..c.heads {
            let key = |j: usize, value: bool| -> &[f32] {
                if j < pb {
                    let o = (blk * c.pre + j) * 2 * width + usize::from(value) * width + h * c.hw;
                    &prefix[o..][..c.hw]
                } else {
                    let o = (b * c.kv_rows + j - pb) * width + h * c.hw;
                    &(if value { &v } else { &k })[o..][..c.hw]
                }
            };
            for r in 0..c.q_rows {
                let qr = &q[(b * c.q_rows + r) * width + h * c.hw..][..c.hw];
                let s: Vec<f64> = (0..kl)
                    .map(|j| {
                        qr.iter()
                            .zip(key(j, false))
                            .map(|(&a, &x)| a as f64 * x as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let den: f64 = e.iter().sum();
                for col in 0..c.hw {
                    want[(b * c.q_rows + r) * width + h * c.hw + col] = if kl == 0 {
                        0.0
                    } else {
                        (0..kl)
                            .map(|j| e[j] * key(j, true)[col] as f64)
                            .sum::<f64>()
                            / den
                    };
                }
            }
        }
    }
    (out, want)
}

fn check_prefix(c: &PrefixCase) {
    let (got, want) = attention_prefix_golden(c);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g as f64 - w).abs() <= 1e-5,
            "pre={} rows={:?} [{i}] {g} vs {w}",
            c.pre,
            c.prefix_rows
        );
    }
}

#[test]
fn attention_prefix_rows_per_item_against_reference() {
    // i7 = 37 (not a multiple of 16); blocks reused by items with different pre_b.
    for rows in [
        vec![0u32, 1, 15, 16, 17, 36, 37, 1000],
        vec![37; 8],
        vec![0; 8],
        vec![17, 0, 36, 1, 37, 16, 15, 2],
    ] {
        check_prefix(&PrefixCase {
            batch: 8,
            q_rows: 5,
            kv_rows: 9,
            heads: 2,
            hw: 64,
            pre: 37,
            blocks: 3,
            index: vec![0, 1, 0, 2, 1, 0, 2, 0],
            prefix_rows: Some(rows.clone()),
            lengths: None,
        });
        // Lengths cutting inside the own region (and one inside the prefix).
        let lengths: Vec<u32> = rows
            .iter()
            .enumerate()
            .map(|(b, &r)| r.min(37) + [3, 0, 8, 1, 5, 2, 9, 4][b])
            .collect();
        let mut short = lengths.clone();
        short[2] = rows[2].min(37) / 2;
        check_prefix(&PrefixCase {
            batch: 8,
            q_rows: 5,
            kv_rows: 9,
            heads: 2,
            hw: 72,
            pre: 37,
            blocks: 3,
            index: vec![2, 2, 0, 1, 1, 0, 2, 0],
            prefix_rows: Some(rows),
            lengths: Some(short),
        });
    }
    check_prefix(&PrefixCase {
        batch: 3,
        q_rows: 4,
        kv_rows: 6,
        heads: 2,
        hw: 64,
        pre: 48,
        blocks: 2,
        index: vec![1, 0, 1],
        prefix_rows: None,
        lengths: Some(vec![54, 50, 20]),
    });
}

#[test]
fn attention_prefix_rows_all_full_is_bitwise_flag_off() {
    let base = PrefixCase {
        batch: 4,
        q_rows: 6,
        kv_rows: 7,
        heads: 2,
        hw: 64,
        pre: 29,
        blocks: 2,
        index: vec![0, 1, 1, 0],
        prefix_rows: None,
        lengths: Some(vec![36, 30, 33, 29]),
    };
    let (off, _) = attention_prefix_golden(&base);
    let (on, _) = attention_prefix_golden(&PrefixCase {
        prefix_rows: Some(vec![29, 29, 400, 29]),
        index: base.index.clone(),
        lengths: base.lengths.clone(),
        ..base
    });
    assert_eq!(
        off.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        on.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

/// CopyColsF32 flag bit 1: `[items][rows][cols]` into `out` item `g * items + b`, capacity in
/// flag bits 8..31.
fn copy_group(out: &mut [f32], x: &mut [f32], items: u32, rows: u32, g: u32, capacity: u32) {
    let (cols, in_stride, out_stride) = (5u32, 7u32, 6u32);
    let mut group = vec![g];
    let mut d = op(
        DevOp::CopyColsF32,
        &[0, 1, 2],
        &[
            items,
            rows,
            cols,
            in_stride,
            1,
            out_stride,
            1,
            2 | capacity << 8,
        ],
    );
    d.fj = [0, rows * in_stride, 40 * out_stride];
    run(&d, &[ptr(out), ptr(x), ptr(&mut group)]);
}

#[test]
fn copy_cols_group_offsets_guard_and_reuse() {
    let (items, capacity) = (2u32, 6u32);
    let item = 40 * 6usize;
    let x0 = data(items as usize * 40 * 7, 61);
    let mut x = x0.clone();
    let fresh = |rows: u32, g: u32| {
        let mut out = vec![-1.0f32; capacity as usize * item];
        let mut x = x0.clone();
        copy_group(&mut out, &mut x, items, rows, g, capacity);
        out
    };
    for g in [0u32, 2] {
        let out = fresh(40, g);
        for (i, &o) in out.iter().enumerate() {
            let (it, r, c) = (i / item, i % item / 6, i % 6);
            let want = if (it as u32) / items == g && c >= 1 {
                let b = it - (g * items) as usize;
                x0[b * 40 * 7 + r * 7 + 1 + c - 1]
            } else {
                -1.0
            };
            assert_eq!(o, want, "g={g} [{it}][{r}][{c}]");
        }
    }
    // (g + 1) * items > capacity: nothing written.
    assert!(fresh(40, 3).iter().all(|&o| o == -1.0));
    assert!(fresh(40, u32::MAX).iter().all(|&o| o == -1.0));
    // Long then short into group 2: rows < short as a fresh short write, the rest the long one.
    let mut out = vec![-1.0f32; capacity as usize * item];
    copy_group(&mut out, &mut x, items, 40, 2, capacity);
    let long = out.clone();
    copy_group(&mut out, &mut x, items, 11, 2, capacity);
    let short = fresh(11, 2);
    for (i, &o) in out.iter().enumerate() {
        let r = i % item / 6;
        assert_eq!(o, if r < 11 { short[i] } else { long[i] }, "[{i}]");
    }
}
