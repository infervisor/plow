//! `StreamTailF32` (golden CPU tier) against its `packet::dev` contract: fresh streams, `n` = 0,
//! `n < L`, `n >= L`, mixed items, column windows and an out-of-range slot.
#![cfg(feature = "cpu")]

use std::ffi::c_void;

use packet::dev::{DevOp, TENSOR_NONE16};
use plowrt::exec::cpu::ffi::{self, DevInst64, Isa, PlowCpuCtx};

struct Geo {
    items: usize,
    slots: usize,
    tail: usize,
    rows: usize,
    cols: usize,
    z_cols: usize,
    z_col0: usize,
    src_stride: usize,
}

fn inst(g: &Geo, mode: u32) -> DevInst64 {
    let mut d = DevInst64 {
        op: DevOp::StreamTailF32 as u16,
        blocks: 3,
        ..Default::default()
    };
    d.t = [0, 1, 2, 3, 4, 5, TENSOR_NONE16, TENSOR_NONE16];
    if mode == 1 {
        d.t[0] = TENSOR_NONE16;
        d.t[5] = TENSOR_NONE16;
    }
    let src_stride = if mode == 0 { g.src_stride } else { 0 };
    d.i = [
        g.items,
        g.tail,
        g.rows,
        g.cols,
        g.z_cols,
        g.z_col0,
        src_stride,
        mode as usize,
    ]
    .map(|v| v as u32);
    d.fj[1] = g.slots as u32;
    d
}

/// `t0`/`t1` = z/src for gather, unused/z for store.
fn run(d: &DevInst64, t0: &mut [f32], t1: &mut [f32], state: &mut [f32], ints: &mut [Vec<u32>; 3]) {
    let mut ctx = PlowCpuCtx::new(0, 0);
    ffi::thread_init(&mut ctx).unwrap();
    let mut scratch = vec![0u8; (ffi::scratch_bytes() as usize).max(64)];
    ctx.scratch = scratch.as_mut_ptr() as *mut c_void;
    ctx.scratch_bytes = scratch.len() as u32;
    let [slot, n, pos] = ints;
    let tensors = [
        t0.as_mut_ptr() as *mut c_void,
        t1.as_mut_ptr() as *mut c_void,
        state.as_mut_ptr() as *mut c_void,
        slot.as_mut_ptr() as *mut c_void,
        n.as_mut_ptr() as *mut c_void,
        pos.as_mut_ptr() as *mut c_void,
    ];
    for slice in 0..3 {
        // SAFETY: every handle the op names is a live buffer sized for the geometry.
        unsafe { ffi::exec(d, slice, 3, &tensors, &mut ctx) }.unwrap();
    }
}

fn gather_ref(g: &Geo, src: &[f32], state: &[f32], slot: &[u32], pos: &[u32]) -> Vec<f32> {
    let zr = g.tail + g.rows;
    let mut z = vec![0.0; g.items * zr * g.z_cols];
    for b in 0..g.items {
        for r in 0..g.tail {
            for c in 0..g.cols {
                if pos[b] != 0 {
                    z[(b * zr + r) * g.z_cols + g.z_col0 + c] =
                        state[(slot[b] as usize * g.tail + r) * g.cols + c];
                }
            }
        }
        for r in 0..g.rows {
            for c in 0..g.z_cols {
                z[(b * zr + g.tail + r) * g.z_cols + c] = src[(b * g.rows + r) * g.src_stride + c];
            }
        }
    }
    z
}

fn store_ref(g: &Geo, z: &[f32], state: &mut [f32], slot: &[u32], n: &[u32]) {
    let zr = g.tail + g.rows;
    for b in 0..g.items {
        for i in 0..g.tail {
            for c in 0..g.cols {
                state[(slot[b] as usize * g.tail + i) * g.cols + c] =
                    z[(b * zr + n[b] as usize + i) * g.z_cols + g.z_col0 + c];
            }
        }
    }
}

fn fill(len: usize, base: f32) -> Vec<f32> {
    (0..len).map(|i| base + i as f32).collect()
}

/// Gather then store one chunk; returns `(z, state before, state after)`.
fn chunk(g: &Geo, slot: &[u32], n: &[u32], pos: &[u32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    ffi::init(Isa::Scalar).unwrap();
    assert!(ffi::has(DevOp::StreamTailF32 as u16));
    let zr = g.tail + g.rows;
    let mut src = fill((g.items * g.rows - 1) * g.src_stride + g.z_cols, 1000.0);
    let before = fill(g.slots * g.tail * g.cols, 0.5);
    let mut state = before.clone();
    let mut z = vec![f32::NAN; g.items * zr * g.z_cols];
    let mut ints = [slot.to_vec(), n.to_vec(), pos.to_vec()];
    run(&inst(g, 0), &mut z, &mut src, &mut state, &mut ints);
    assert_eq!(z, gather_ref(g, &src, &before, slot, pos));
    assert_eq!(state, before, "gather must not write state");
    run(&inst(g, 1), &mut [], &mut z, &mut state, &mut ints);
    let mut expect = before.clone();
    store_ref(g, &z, &mut expect, slot, n);
    assert_eq!(state, expect);
    (z, before, state)
}

#[test]
fn stream_tail_gather_and_store_window() {
    let g = Geo {
        items: 4,
        slots: 5,
        tail: 3,
        rows: 4,
        cols: 5,
        z_cols: 8,
        z_col0: 2,
        src_stride: 10,
    };
    // Item 0: fresh, n = 0. Item 1: n < L. Item 2: n > L (= N). Item 3: fresh, n < L.
    let (slot, n, pos) = ([2, 0, 4, 1], [0, 2, 4, 1], [0, 7, 3, 0]);
    let (z, before, state) = chunk(&g, &slot, &n, &pos);
    let span = g.tail * g.cols;
    assert!(state[2 * span..3 * span].iter().all(|&v| v == 0.0));
    assert_eq!(state[3 * span..4 * span], before[3 * span..4 * span]);
    // Fresh item 3 with one new row: two zero tail rows, then src row 0's window.
    assert!(state[span..span + 2 * g.cols].iter().all(|&v| v == 0.0));
    let zr = g.tail + g.rows;
    assert_eq!(
        state[span + 2 * g.cols..2 * span],
        z[(3 * zr + g.tail) * g.z_cols + g.z_col0..][..g.cols]
    );
    // Columns outside the window of a carried tail row are zero.
    assert!(z[zr * g.z_cols..][..g.z_col0].iter().all(|&v| v == 0.0));
}

#[test]
fn stream_tail_full_width_tail_longer_than_chunk() {
    let g = Geo {
        items: 3,
        slots: 3,
        tail: 6,
        rows: 2,
        cols: 4,
        z_cols: 4,
        z_col0: 0,
        src_stride: 4,
    };
    chunk(&g, &[1, 2, 0], &[2, 0, 1], &[5, 1, 0]);
}

#[test]
fn stream_tail_out_of_range_slot_is_not_written() {
    let g = Geo {
        items: 2,
        slots: 2,
        tail: 2,
        rows: 2,
        cols: 2,
        z_cols: 2,
        z_col0: 0,
        src_stride: 2,
    };
    ffi::init(Isa::Scalar).unwrap();
    let mut z = fill(2 * 4 * 2, 100.0);
    let before = fill(2 * 2 * 2, 0.5);
    let mut state = before.clone();
    let mut ints = [vec![1, 2], vec![0, 0], vec![1, 1]];
    run(&inst(&g, 1), &mut [], &mut z, &mut state, &mut ints);
    assert_eq!(state[..4], before[..4]);
    assert_eq!(state[4..], z[..4]);
}
