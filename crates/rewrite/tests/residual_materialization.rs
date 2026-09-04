use nn_graph::{DType, Dim, Nn};

fn residual_consumer(nested: bool) -> nn_graph::Graph {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let shape = nn.shape([Dim::stat(1), Dim::stat(64)]);
    let pre = nn.input("pre", shape.clone(), DType::BF16);
    let a = nn.input("a", shape.clone(), DType::BF16);
    let b = nn.input("b", shape, DType::BF16);
    let inner = nn.add(a, b);
    let residual = if nested { nn.add(pre, inner) } else { inner };
    let out = nn.block_residual("mix", residual, &[], 64, 1);
    nn.mark_output(out);
    nn.finish()
}

#[test]
fn fuses_two_input_residual_and_preserves_a_materialized_result() {
    let (fused, _) = rewrite::rewrite_graph(&residual_consumer(false)).expect("rewrite");
    assert!(fused.contains("FusedMaterializedResidualBlock"));
    assert!(!fused.contains("FusedMaterializedResidual3Block"));
}

#[test]
fn nested_residual_keeps_the_inner_rounding_point() {
    let (fused, _) = rewrite::rewrite_graph(&residual_consumer(true)).expect("rewrite");
    assert!(fused.contains("FusedMaterializedResidual3Block"));
    let node = fused
        .nodes
        .iter()
        .find(|n| n.op == "FusedMaterializedResidual3Block")
        .unwrap();
    assert_eq!(
        node.args
            .iter()
            .filter(|a| matches!(a, rewrite::Arg::Node(_)))
            .count(),
        6
    );
}
