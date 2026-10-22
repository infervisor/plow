//! `UnaryF32` `ACT_LOG`, `ReduceRowsF32` and `LengthAffineU32` (golden CPU tier) against their
//! `packet::dev` contracts: masked ceil-mode segments, column windows, every reduction, and
//! floor division with negative offsets, clamps and an elementwise bound.
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
