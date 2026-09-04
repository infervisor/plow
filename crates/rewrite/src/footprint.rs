//! Per-tile footprints and tile-coordinate domains.
//!
//! Turns an op's `(kind, chosen tile, I/O)` into the rectangular tensor slices
//! each of its tiles reads and writes. This is what the cross-op *tile*
//! dependency derivation in [`crate::tilegraph`] consumes: a consumer tile
//! depends on exactly the producer tiles whose write-slice overlaps its
//! read-slice on the shared tensor.
//!
//! Footprints are a tiling concern, so they live here in `rewrite` and key off
//! [`OpKind`] — itself an abstraction of `nn_graph::Op`'s access class
//! (row-wise norm/act, contraction GEMM, flash attention, layout) — so
//! `nn-graph` stays a lean shape/dataflow IR.
//!
//! Simplification (deliberate): rectangular footprints coupled on the token/row
//! axis (tensor axis 0). General affine / polyhedral access (transpose remaps,
//! conv halos, attention masks) is deferred.

use crate::tilegraph::{Compute, OpKind};
use std::ops::Range;

/// A rectangular region of a tensor: a half-open interval per axis. An empty
/// `ranges` means the whole tensor (layout passthrough).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSlice {
    pub tensor: String,
    pub ranges: Vec<Range<i64>>,
}

/// The tile-coordinate grid of one op, after its tile is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileDomain {
    /// Row-wise op: `r ∈ 0..⌈rows/br⌉`. Coord `[r]`.
    Row { rows: i64, br: i64 },
    /// GEMM: `(i,j) ∈ ⌈m/bm⌉ × ⌈n/bn⌉`. Coord `[i, j]`.
    Gemm { m: i64, n: i64, bm: i64, bn: i64 },
    /// Flash attention: `(h, q) ∈ heads × ⌈seq_q/bq⌉`. Coord `[h, q]`.
    Flash { heads: i64, seq_q: i64, bq: i64 },
    /// Layout / join: a single coarse coord `[0]`.
    Layout,
}

fn n_tiles(extent: i64, block: i64) -> i64 {
    if block <= 0 {
        return 1;
    }
    (extent.max(0) as u64).div_ceil(block as u64) as i64
}

impl TileDomain {
    /// All tile coordinates of this op's grid.
    pub fn coords(&self) -> Vec<Vec<i64>> {
        match *self {
            TileDomain::Row { rows, br } => (0..n_tiles(rows, br)).map(|r| vec![r]).collect(),
            TileDomain::Gemm { m, n, bm, bn } => {
                let (ti, tj) = (n_tiles(m, bm), n_tiles(n, bn));
                let mut v = Vec::with_capacity((ti * tj) as usize);
                for i in 0..ti {
                    for j in 0..tj {
                        v.push(vec![i, j]);
                    }
                }
                v
            }
            TileDomain::Flash { heads, seq_q, bq } => {
                let tq = n_tiles(seq_q, bq);
                let mut v = Vec::with_capacity((heads * tq) as usize);
                for h in 0..heads {
                    for q in 0..tq {
                        v.push(vec![h, q]);
                    }
                }
                v
            }
            TileDomain::Layout => vec![vec![0]],
        }
    }

    /// The grid axis that indexes the tensor's token/row axis (tensor axis 0),
    /// paired with that axis's block size. `None` for layout (untiled).
    pub fn row_axis(&self) -> Option<(usize, i64)> {
        match *self {
            TileDomain::Row { br, .. } => Some((0, br)),
            TileDomain::Gemm { bm, .. } => Some((0, bm)),
            TileDomain::Flash { bq, .. } => Some((1, bq)),
            TileDomain::Layout => None,
        }
    }
}

/// What a single tile reads and writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Footprint {
    pub write: TensorSlice,
    pub reads: Vec<TensorSlice>,
}

/// An op's tensor names (convention: `inputs[0]` is the activation).
#[derive(Clone, Copy, Debug)]
pub struct OpIo<'a> {
    pub inputs: &'a [String],
    pub output: &'a str,
}

/// The read/write tensor slices of one tile `coord` of an op.
pub fn footprints(kind: &OpKind, tile: &Compute, io: &OpIo, coord: &[i64]) -> Footprint {
    match (kind, tile) {
        (OpKind::Gemm(g), Compute::Gemm(t)) => {
            let (i, j) = (coord[0], coord[1]);
            let m0 = i * t.bm;
            let m1 = (m0 + t.bm).min(g.m);
            let n0 = j * t.bn;
            let n1 = (n0 + t.bn).min(g.n);
            Footprint {
                write: TensorSlice {
                    tensor: io.output.into(),
                    ranges: vec![m0..m1, n0..n1],
                },
                reads: vec![
                    // activation rows [m0:m1] × all K
                    TensorSlice {
                        tensor: io.inputs[0].clone(),
                        ranges: vec![m0..m1, 0..g.k],
                    },
                    // weight rows [n0:n1] × all K  (weight is [N, K])
                    TensorSlice {
                        tensor: io.inputs[1].clone(),
                        ranges: vec![n0..n1, 0..g.k],
                    },
                ],
            }
        }
        (OpKind::Row(r), Compute::Row(t)) => {
            let r0 = coord[0] * t.br;
            let r1 = (r0 + t.br).min(r.rows);
            let row = |name: &str| TensorSlice {
                tensor: name.into(),
                ranges: vec![r0..r1, 0..r.feat],
            };
            Footprint {
                write: row(io.output),
                reads: io.inputs.iter().map(|n| row(n)).collect(),
            }
        }
        (OpKind::Model(m), Compute::Row(t)) => {
            let r0 = coord[0] * t.br;
            let r1 = (r0 + t.br).min(m.rows);
            let row = |name: &str| TensorSlice {
                tensor: name.into(),
                ranges: vec![r0..r1, 0..m.feat],
            };
            if m.kind == crate::ModelOpKind::Embedding {
                return Footprint {
                    write: row(io.output),
                    reads: vec![
                        TensorSlice {
                            tensor: io.inputs[0].clone(),
                            ranges: vec![r0..r1],
                        },
                        // Runtime gathers arbitrary vocabulary rows, but the
                        // transfer volume is one hidden row per token tile.
                        TensorSlice {
                            tensor: io.inputs[1].clone(),
                            ranges: vec![r0..r1, 0..m.feat],
                        },
                    ],
                };
            }
            let row_aligned = matches!(
                m.kind,
                crate::ModelOpKind::Silu
                    | crate::ModelOpKind::Sigmoid
                    | crate::ModelOpKind::Add
                    | crate::ModelOpKind::Sub
                    | crate::ModelOpKind::Mul
                    | crate::ModelOpKind::Div
            );
            Footprint {
                write: row(io.output),
                reads: io
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(idx, n)| {
                        let norm_activation = idx == 0
                            && matches!(
                                m.kind,
                                crate::ModelOpKind::RmsNorm
                                    | crate::ModelOpKind::RmsNormZeroCentered
                            );
                        if row_aligned || norm_activation {
                            row(n)
                        } else {
                            TensorSlice {
                                tensor: n.clone(),
                                ranges: vec![],
                            }
                        }
                    })
                    .collect(),
            }
        }
        (OpKind::Flash(a), Compute::Flash(t)) => {
            let q0 = coord[1] * t.bq;
            let q1 = (q0 + t.bq).min(a.seq_q);
            // Q-block reads its own rows; K/V are read in full for that head.
            let mut reads = vec![TensorSlice {
                tensor: io.inputs[0].clone(),
                ranges: vec![q0..q1, 0..a.head_dim],
            }];
            for kv in io.inputs.iter().skip(1) {
                reads.push(TensorSlice {
                    tensor: kv.clone(),
                    ranges: vec![0..a.seq_kv, 0..a.head_dim],
                });
            }
            Footprint {
                write: TensorSlice {
                    tensor: io.output.into(),
                    ranges: vec![q0..q1, 0..a.head_dim],
                },
                reads,
            }
        }
        // Layout / Join: whole-tensor passthrough (empty ranges).
        _ => Footprint {
            write: TensorSlice {
                tensor: io.output.into(),
                ranges: vec![],
            },
            reads: io
                .inputs
                .iter()
                .map(|n| TensorSlice {
                    tensor: n.clone(),
                    ranges: vec![],
                })
                .collect(),
        },
    }
}
