//! Shape buckets end-to-end: compile one packet stream per `(batch, seq)`
//! bucket sharing a single weight + KV layout, verify a flip moves no
//! weights/KV (unified `(BN,BK)`, only `BM` varies), and the cost-model chooser.

use costmodel::{hwspec, Soc, DEFAULT_PAGE_BYTES};
use nn_graph::{infer_shapes, ActKind, DType, Nn};
use rewrite::{plan_from_block, Compute, GraphNode, LayerPlan, TileGraph};
use schedule::{
    choose_buckets, compile_buckets, BucketStream, Config, Phase, Request, ShapeBucket,
};
use std::collections::HashSet;

const H: i64 = 256;
const NH: i64 = 4;
const NKV: i64 = 2;
const HD: i64 = 64;
const QD: i64 = NH * HD;
const KVD: i64 = NKV * HD;
const IM: i64 = 512;

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}

/// A tiny transformer block whose token count is the bucket's GEMM `M`.
fn block_plan_at(bucket: &ShapeBucket) -> LayerPlan {
    let t = bucket.rows().max(1);
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let x = nn.input("x", nn.shape([t.into(), H.into()]), DType::BF16);
    nn.begin_block("layers.0");
    let h1 = nn.rmsnorm("input_norm", x, H, 1e-6);
    let q = nn.linear("q_proj", h1, H, QD, false);
    let k = nn.linear("k_proj", h1, H, KVD, false);
    let v = nn.linear("v_proj", h1, H, KVD, false);
    let qh = nn.reshape(q, [t.into(), NH.into(), HD.into()]);
    let kh = nn.reshape(k, [t.into(), NKV.into(), HD.into()]);
    let vh = nn.reshape(v, [t.into(), NKV.into(), HD.into()]);
    let qn = nn.rmsnorm("q_norm", qh, HD, 1e-6);
    let kn = nn.rmsnorm("k_norm", kh, HD, 1e-6);
    let qr = nn.rope(qn, HD as u32, 1e6);
    let kr = nn.rope(kn, HD as u32, 1e6);
    let attn = nn.attention(
        qr, kr, vh, NH as u32, NKV as u32, HD as u32, true, None, None,
    );
    let ao = nn.reshape(attn, [t.into(), QD.into()]);
    let o = nn.linear("o_proj", ao, QD, H, false);
    let r1 = nn.add(x, o);
    let h2 = nn.rmsnorm("post_norm", r1, H, 1e-6);
    let gate = nn.linear("gate_proj", h2, H, IM, false);
    let up = nn.linear("up_proj", h2, H, IM, false);
    let ga = nn.act(ActKind::Silu, gate);
    let gu = nn.mul(ga, up);
    let down = nn.linear("down_proj", gu, IM, H, false);
    let out = nn.add(r1, down);
    nn.end_block();
    nn.mark_output(out);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("infer");
    plan_from_block(&g, 0).expect("plan")
}

/// The GEMM tiles `(BM, BN, BK)` chosen in one bucket's tile graph.
fn gemm_tiles(g: &TileGraph) -> Vec<(i64, i64, i64)> {
    g.nodes
        .iter()
        .filter_map(|n| match n {
            GraphNode::Compute {
                kind: Compute::Gemm(t),
                ..
            } => Some((t.bm, t.bn, t.bk)),
            _ => None,
        })
        .collect()
}

fn buckets() -> Vec<ShapeBucket> {
    vec![
        ShapeBucket {
            batch: 1,
            seq: 1,
            phase: Phase::Decode,
        }, // M = 1  (skinny)
        ShapeBucket {
            batch: 1,
            seq: 256,
            phase: Phase::Prefill,
        }, // M = 256
        ShapeBucket {
            batch: 1,
            seq: 1024,
            phase: Phase::Prefill,
        }, // M = 1024 (square)
    ]
}

#[test]
fn flip_shares_one_weight_and_kv_layout() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let c = compile_buckets(&soc, &Config::default(), &buckets(), block_plan_at);

    // A single weight layout (BN, BK) was chosen and KV layout derived.
    let w = c.weight.expect("weight layout chosen");
    let _kv = c.kv.expect("kv layout derived");
    assert_eq!(c.streams.len(), 3);

    // The hard invariant: every stream's GEMM tiles use the SAME (BN, BK) — so
    // the weight HBM layout is identical and a flip moves no weight bytes.
    let mut bnbk: HashSet<(i64, i64)> = HashSet::new();
    for s in &c.streams {
        for (_, bn, bk) in gemm_tiles(&s.graph) {
            bnbk.insert((bn, bk));
        }
    }
    assert_eq!(
        bnbk.len(),
        1,
        "weight tiling not unified across buckets: {bnbk:?}"
    );
    assert_eq!(bnbk.into_iter().next().unwrap(), (w.bn, w.bk));

    // KV layout is one shared value (nothing per-bucket to flip).
    // (Compiled.kv is a single Option, so this is structural.)
}

#[test]
fn tiling_varies_by_bucket() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let c = compile_buckets(&soc, &Config::default(), &buckets(), block_plan_at);
    let bm = |b: &BucketStream| {
        gemm_tiles(&b.graph)
            .into_iter()
            .map(|(bm, _, _)| bm)
            .collect::<HashSet<_>>()
    };

    let decode = c
        .streams
        .iter()
        .find(|s| s.bucket.phase == Phase::Decode)
        .unwrap();
    let prefill = c.streams.iter().find(|s| s.bucket.rows() == 1024).unwrap();
    // Decode (M=1) tiles skinny; prefill (M=1024) keeps a square BM=128.
    assert!(bm(decode).contains(&64), "decode should use skinny BM=64");
    assert!(
        bm(prefill).contains(&128),
        "prefill should use square BM=128"
    );
}

#[test]
fn select_rounds_up_and_flips() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let c = compile_buckets(&soc, &Config::default(), &buckets(), block_plan_at);

    // A 700-token prefill rounds up to the 1024 bucket.
    let s = c
        .select(&Request {
            batch: 1,
            seq: 700,
            phase: Phase::Prefill,
        })
        .expect("covered");
    assert_eq!(s.bucket.seq, 1024);
    // A 200-token prefill rounds up to 256 — a *different* precompiled stream (flip).
    let s2 = c
        .select(&Request {
            batch: 1,
            seq: 200,
            phase: Phase::Prefill,
        })
        .expect("covered");
    assert_eq!(s2.bucket.seq, 256);
    // Exceeding every bucket ⇒ no stream (caller must chunk / recompile).
    assert!(c
        .select(&Request {
            batch: 1,
            seq: 5000,
            phase: Phase::Prefill
        })
        .is_none());
    // Decode dispatches to the decode stream.
    assert_eq!(
        c.select(&Request {
            batch: 1,
            seq: 1,
            phase: Phase::Decode
        })
        .unwrap()
        .bucket
        .phase,
        Phase::Decode
    );
}

#[test]
fn chooser_bounds_buckets_and_covers() {
    // Workload: a spike at seq=128 plus a few longer prompts.
    let load = vec![
        (
            Request {
                batch: 1,
                seq: 128,
                phase: Phase::Prefill,
            },
            100,
        ),
        (
            Request {
                batch: 1,
                seq: 130,
                phase: Phase::Prefill,
            },
            5,
        ),
        (
            Request {
                batch: 1,
                seq: 900,
                phase: Phase::Prefill,
            },
            3,
        ),
        (
            Request {
                batch: 1,
                seq: 1000,
                phase: Phase::Prefill,
            },
            2,
        ),
    ];
    let chosen = choose_buckets(&load, 1, 2, |rows| rows as u64);
    // batch ≤ 1 bucket, seq ≤ 2 buckets ⇒ ≤ 2 buckets total.
    assert!(chosen.len() <= 2 && !chosen.is_empty());
    // Every request rounds up to some chosen bucket (full coverage).
    for (r, _) in &load {
        assert!(
            chosen
                .iter()
                .any(|b| b.phase == r.phase && b.batch >= r.batch && b.seq >= r.seq),
            "request {r:?} not covered by {chosen:?}"
        );
    }
    // The dominant cluster gets a tight bucket (≤ 130, not rounded to 1000).
    assert!(chosen.iter().any(|b| b.seq <= 130));
}

#[test]
fn kv_layout_growth() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let c = compile_buckets(&soc, &Config::default(), &buckets(), block_plan_at);
    let kv = c.kv.expect("kv layout");
    assert_eq!(kv.head_dim, HD);
    // Prefill of 1000 tokens at block_seq=256 → 4 KV blocks; layout drives growth.
    assert_eq!(kv.blocks_for_prefill(1000), 1000_i64.div_euclid(256) + 1);
    assert_eq!(kv.blocks_for_prefill(256), 1);
    // Decode appends a block on step 1 and every block_seq-th step after.
    assert!(kv.appends_block_at(1));
    assert!(!kv.appends_block_at(2));
    assert!(kv.block_bytes() > 0);
}
