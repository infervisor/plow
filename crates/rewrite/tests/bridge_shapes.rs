use nn_graph::{DType, Dim, Nn};
use rewrite::{plan_from_all_blocks, plan_from_fused, OpKind};

fn attention(batch: Option<i64>, seq_q: i64, seq_kv: i64) -> nn_graph::Graph {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let shape = |seq, heads| {
        let mut dims = vec![seq, heads, 16];
        if let Some(b) = batch {
            dims.insert(0, b);
        }
        nn_graph::Shape::new(dims.into_iter().map(Dim::stat))
    };
    let q = nn.input("q", shape(seq_q, 4), DType::BF16);
    let k = nn.input("k", shape(seq_kv, 2), DType::BF16);
    let v = nn.input("v", shape(seq_kv, 2), DType::BF16);
    nn.begin_block("attention");
    let out = nn.attention(q, k, v, 4, 2, 16, true, None, None);
    nn.mark_output(out);
    let mut graph = nn.finish();
    nn_graph::infer_shapes(&mut graph).unwrap();
    graph
}

#[test]
fn attention_preserves_sequence_axis_in_both_bridges() {
    for batch in [None, Some(1)] {
        for (q, kv) in [(1, 128), (128, 256)] {
            let graph = attention(batch, q, kv);
            let (fused, _) = rewrite::rewrite_graph(&graph).unwrap();
            for plan in [
                plan_from_all_blocks(&graph).unwrap(),
                plan_from_fused(&fused, &graph).unwrap(),
            ] {
                let flash = plan
                    .ops
                    .iter()
                    .find_map(|op| match op.kind {
                        OpKind::Flash(a) => Some(a),
                        _ => None,
                    })
                    .unwrap();
                assert_eq!(
                    (flash.seq_q, flash.seq_kv, flash.heads, flash.kv_heads),
                    (q, kv, 4, 2)
                );
            }
        }
    }
}

#[test]
fn attention_rejects_batches_the_flash_shape_cannot_represent() {
    let graph = attention(Some(2), 128, 128);
    assert!(matches!(
        plan_from_all_blocks(&graph),
        Err(rewrite::BridgeError::AttentionBatch { batch: 2, .. })
    ));
    let (fused, _) = rewrite::rewrite_graph(&graph).unwrap();
    assert!(matches!(
        plan_from_fused(&fused, &graph),
        Err(rewrite::BridgeError::AttentionBatch { batch: 2, .. })
    ));
}

#[test]
fn broadcast_alignment_uses_shapes_not_equal_byte_counts() {
    let mut nn = Nn::new(DType::BF16, DType::F32);
    let x = nn.input(
        "x",
        nn_graph::Shape::new([Dim::stat(2), Dim::stat(64)]),
        DType::BF16,
    );
    let scale = nn.param("channel_scale", [Dim::stat(64)]);
    nn.begin_block("broadcast");
    let out = nn.mul(x, scale);
    nn.mark_output(out);
    let mut graph = nn.finish();
    nn_graph::infer_shapes(&mut graph).unwrap();
    let plan = plan_from_all_blocks(&graph).unwrap();
    let op = plan
        .ops
        .iter()
        .find(|op| matches!(op.kind, OpKind::Model(_)))
        .unwrap();
    let OpKind::Model(m) = op.kind else {
        unreachable!()
    };
    assert_eq!(m.input_bytes[0], m.input_bytes[1]);
    assert_eq!(&m.input_row_aligned[..2], &[true, false]);
    let io = rewrite::OpIo {
        inputs: &op.inputs,
        output: &op.output,
    };
    let fp = rewrite::footprints(
        &op.kind,
        &rewrite::Compute::Row(costmodel::RowTile { br: 1 }),
        &io,
        &[1],
    );
    assert_eq!(fp.reads[0].ranges, vec![1..2, 0..64]);
    assert!(fp.reads[1].ranges.is_empty());
}
