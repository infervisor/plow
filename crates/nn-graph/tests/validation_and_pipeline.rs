//! Shape-inference validation (symbolic reshape/concat/slice/attention
//! checks) and the multi-network `PipelineConfig` build path.

use nn_graph::models::{build_encoder_graph, ModelConfig, PipelineConfig, ShapeBucket};
use nn_graph::{infer_shapes, DType, Dim, Graph, LinearAttnKind, Nn, Origin};

fn weight_names(g: &Graph) -> Vec<String> {
    g.tensors
        .iter()
        .filter(|t| matches!(t.origin, Origin::Weight))
        .filter_map(|t| t.name.clone())
        .collect()
}

// ---- symbolic validation ----------------------------------------------

#[test]
fn reshape_rejects_provably_unequal_symbolic_counts() {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let b = nn.sym("B");
    let x = nn.input("x", nn.shape([b.clone(), Dim::stat(256)]), DType::BF16);
    // [B, 256] -> [B, 512]: B*256 != B*512 for every B >= 1.
    let y = nn.reshape(x, [b.clone(), Dim::stat(512)]);
    nn.mark_output(y);
    let mut g = nn.finish();
    let err = infer_shapes(&mut g).expect_err("reshape must be rejected");
    assert!(err.to_string().contains("element count"), "got: {err}");
}

#[test]
fn kimi_linear_attention_rejects_mismatched_gate_heads() {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let b = nn.sym("B");
    let s = nn.sym("S");
    let qkv_shape = nn.shape([b.clone(), s.clone(), Dim::stat(4), Dim::stat(32)]);
    let q = nn.input("q", qkv_shape.clone(), DType::BF16);
    let k = nn.input("k", qkv_shape.clone(), DType::BF16);
    let v = nn.input("v", qkv_shape, DType::BF16);
    let gate = nn.input(
        "gate",
        nn.shape([b.clone(), s.clone(), Dim::stat(2), Dim::stat(32)]),
        DType::BF16,
    );
    let beta = nn.input("beta", nn.shape([b, s, Dim::stat(4)]), DType::BF16);
    let a_log = nn.param("A_log", [Dim::stat(4)]);
    let dt_bias = nn.param("dt_bias", [Dim::stat(128)]);
    let out = nn.linear_attention(
        LinearAttnKind::KimiDelta,
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        4,
        32,
    );
    nn.mark_output(out);
    let mut g = nn.finish();
    let err = infer_shapes(&mut g).expect_err("mismatched Kimi gate must be rejected");
    assert!(err.to_string().contains("gate axis 2"), "got: {err}");
}

#[test]
fn reshape_allows_unprovable_and_equal_symbolic_counts() {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let b = nn.sym("B");
    let s = nn.sym("S");
    let x = nn.input(
        "x",
        nn.shape([b.clone(), s.clone(), Dim::stat(64)]),
        DType::BF16,
    );
    // Equal canonical polynomials: [B, S, 64] -> [B*S, 64].
    let y = nn.reshape(x, [b.mul(&s), Dim::stat(64)]);
    // Unprovable: [B*S, 64] -> [4096, 64] (equal when B*S = 4096) — trusted.
    let z = nn.reshape(y, [Dim::stat(4096), Dim::stat(64)]);
    nn.mark_output(z);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("both reshapes are legal");
}

#[test]
fn slice_rejects_static_out_of_bounds() {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let x = nn.input("x", nn.shape([Dim::stat(4), Dim::stat(100)]), DType::BF16);
    let y = nn.slice_dim(x, 1, Dim::stat(64), Dim::stat(64)); // [64, 128) of 100
    nn.mark_output(y);
    let mut g = nn.finish();
    let err = infer_shapes(&mut g).expect_err("slice must be rejected");
    assert!(err.to_string().contains("out of bounds"), "got: {err}");
}

#[test]
fn concat_rejects_provably_mismatched_axes() {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let b = nn.sym("B");
    let a = nn.input(
        "a",
        nn.shape([b.clone(), Dim::stat(8), Dim::stat(256)]),
        DType::BF16,
    );
    let c = nn.input(
        "c",
        nn.shape([b.clone(), Dim::stat(4), Dim::stat(512)]),
        DType::BF16,
    );
    // Concat on axis 1, but axis 2 differs (256 vs 512).
    let y = nn.concat(1, vec![a, c]);
    nn.mark_output(y);
    let mut g = nn.finish();
    let err = infer_shapes(&mut g).expect_err("concat must be rejected");
    assert!(err.to_string().contains("mismatch"), "got: {err}");
}

// ---- qwen3 architecture fixes ------------------------------------------

const QWEN3_4B_STYLE: &str = r#"{
    "model_type": "qwen3",
    "vocab_size": 1000,
    "hidden_size": 2560,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": true
}"#;

#[test]
fn qwen3_explicit_head_dim_and_qk_norm() {
    let g = nn_graph::models::build_from_config_json(QWEN3_4B_STYLE).expect("build qwen3");
    let ws = weight_names(&g);
    // Explicit head_dim 128 (not 2560/32 = 80): q_proj is [32*128, 2560].
    let q = g
        .tensors
        .iter()
        .find(|t| t.name.as_deref() == Some("model.layers.0.self_attn.q_proj.weight"))
        .expect("q_proj weight");
    let dims: Vec<i64> = q
        .shape
        .as_ref()
        .unwrap()
        .dims()
        .iter()
        .map(|d| d.as_static().unwrap())
        .collect();
    assert_eq!(dims, vec![4096, 2560], "q_proj must use explicit head_dim");
    // Per-head qk-norm weights are consumed.
    assert!(ws
        .iter()
        .any(|w| w == "model.layers.0.self_attn.q_norm.weight"));
    assert!(ws
        .iter()
        .any(|w| w == "model.layers.1.self_attn.k_norm.weight"));
}

// ---- encoder taps + pipeline -------------------------------------------

const LLAMA_SMALL: &str = r#"{
    "model_type": "llama",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": false
}"#;

#[test]
fn encoder_build_taps_hidden_states_and_drops_lm_head() {
    let cfg = ModelConfig::from_json(LLAMA_SMALL).expect("parse");
    // FLUX.2-style: tap layers 1 and 2, plus the final hidden states.
    let g = build_encoder_graph(&cfg, &[1, 2]).expect("encoder build");
    assert_eq!(g.outputs.len(), 3, "2 taps + final hidden states");
    for &out in &g.outputs {
        let shape = g.tensor(out).shape.as_ref().expect("inferred");
        assert_eq!(shape.display_with(&g.syms), "[B, S, 256]");
    }
    assert!(
        !weight_names(&g).iter().any(|w| w == "lm_head.weight"),
        "encoder build must not declare lm_head"
    );
}

#[test]
fn pipeline_builds_named_subnetwork_graphs() {
    let mut pipe = PipelineConfig::from_json_parts([("text_encoder", QWEN3_4B_STYLE)])
        .expect("parse pipeline");
    pipe.networks[0].encoder_taps = Some(vec![]);
    let graphs = pipe.build_all(&ShapeBucket::default()).expect("build all");
    assert_eq!(graphs.len(), 1);
    let (name, g) = &graphs[0];
    assert_eq!(name, "text_encoder");
    // Encoder output is hidden states, not logits.
    let out = *g.outputs.last().unwrap();
    assert_eq!(
        g.tensor(out).shape.as_ref().unwrap().display_with(&g.syms),
        "[B, S, 2560]"
    );
}
