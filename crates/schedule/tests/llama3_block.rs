//! End-to-end: build a LLaMA 3.1 8B transformer block at real dimensions,
//! assemble its tile graph, schedule it on an H100, and verify feasibility
//! invariants. This proves the full pipeline handles the exact shapes, GQA
//! ratio, and BF16 dtype of the production model.

use costmodel::{hwspec, Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use nn_graph::{infer_shapes, ActKind, DType, Nn};
use rewrite::{assemble, plan_from_block, LayerPlan};
use schedule::{schedule, Config};

// Real LLaMA 3.1 8B dimensions.
const H: i64 = 4096;
const NH: i64 = 32; // query heads
const NKV: i64 = 8; // KV heads (GQA 4:1)
const HD: i64 = 128; // head_dim
const QD: i64 = NH * HD; // 4096
const KVD: i64 = NKV * HD; // 1024
const IM: i64 = 14336; // SwiGLU intermediate
                       // Sequence length for the compile bucket.
const T: i64 = 512;

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}

/// Build a LayerPlan for one LLaMA 3.1 8B decoder block at the real dimensions.
fn llama3_block_plan() -> LayerPlan {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let x = nn.input("x", nn.shape([T.into(), H.into()]), DType::BF16);
    nn.begin_block("layers.0");

    // Pre-norm residual: attention.
    let h1 = nn.rmsnorm("input_layernorm", x, H, 1e-5);
    let q = nn.linear("self_attn.q_proj", h1, H, QD, false);
    let k = nn.linear("self_attn.k_proj", h1, H, KVD, false);
    let v = nn.linear("self_attn.v_proj", h1, H, KVD, false);
    let qh = nn.reshape(q, [T.into(), NH.into(), HD.into()]);
    let kh = nn.reshape(k, [T.into(), NKV.into(), HD.into()]);
    let vh = nn.reshape(v, [T.into(), NKV.into(), HD.into()]);
    // LLaMA: NO qk_norm. RoPE directly on projections.
    let qr = nn.rope(qh, HD as u32, 500000.0);
    let kr = nn.rope(kh, HD as u32, 500000.0);
    let attn = nn.attention(
        qr, kr, vh, NH as u32, NKV as u32, HD as u32, true, None, None,
    );
    let ao = nn.reshape(attn, [T.into(), QD.into()]);
    let o = nn.linear("self_attn.o_proj", ao, QD, H, false);
    let r1 = nn.add(x, o);

    // Pre-norm residual: MLP (SwiGLU).
    let h2 = nn.rmsnorm("post_attention_layernorm", r1, H, 1e-5);
    let gate = nn.linear("mlp.gate_proj", h2, H, IM, false);
    let up = nn.linear("mlp.up_proj", h2, H, IM, false);
    let ga = nn.act(ActKind::Silu, gate);
    let gu = nn.mul(ga, up);
    let down = nn.linear("mlp.down_proj", gu, IM, H, false);
    let out = nn.add(r1, down);

    nn.end_block();
    nn.mark_output(out);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("infer");
    plan_from_block(&g, 0).expect("plan")
}

#[test]
fn llama3_8b_block_plan_has_correct_bf16_dtypes() {
    let plan = llama3_block_plan();
    // Verify the plan has the expected ops (at least: norm, q, k, v, attn, o, norm, gate, up, down).
    assert!(
        plan.ops.len() >= 8,
        "LLaMA 3.1 8B block should have at least 8 tile ops, got {}",
        plan.ops.len()
    );
    // All ops must be BF16.
    for op in &plan.ops {
        assert_eq!(
            op.weight_dtype,
            nn_graph::DType::BF16,
            "op {} weight_dtype is not BF16",
            op.name
        );
        assert_eq!(
            op.compute_dtype,
            nn_graph::DType::BF16,
            "op {} compute_dtype is not BF16",
            op.name
        );
    }
}

#[test]
fn llama3_8b_block_schedules_on_h100() {
    let plan = llama3_block_plan();
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();

    // The tile graph should have a non-trivial number of nodes.
    assert!(
        g.nodes.len() > 10,
        "tile graph should have > 10 nodes for a LLaMA 3.1 8B block, got {}",
        g.nodes.len()
    );

    let s = schedule(&soc, &g, &cons, &Config::default());

    // The schedule should have non-zero makespan.
    assert!(
        s.schedule.makespan > 0,
        "schedule should have non-zero makespan"
    );
    // Counters should have positive thresholds.
    assert!(
        s.schedule.counters.iter().all(|c| c.threshold >= 1),
        "all counters should have threshold >= 1"
    );

    eprintln!(
        "llama3_8b_block: {} tiles, {} tasks, makespan {} cycles, {} counters",
        g.nodes.len(),
        s.tasks.tasks.len(),
        s.schedule.makespan,
        s.schedule.counters.len()
    );
}
