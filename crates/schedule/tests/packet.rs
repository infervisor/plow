//! The runtime packet ABI: each scheduled task lowers to a variable-length
//! `packet::Inst` (kernel opcode + only-needed parameters + counter windows),
//! and the program round-trips through the compact wire format.

use costmodel::{hwspec, GemmShape, RowShape, Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use nn_graph::Bindings;
use rewrite::{assemble, plan_from_all_blocks, LayerPlan, ModelOpKind, OpKind, OpSpec};
use schedule::packet::{Body, Opcode, Program};
use schedule::{emit_program, schedule, Config};

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}

fn plan() -> LayerPlan {
    LayerPlan {
        ops: vec![
            OpSpec {
                name: "norm".into(),
                inputs: vec!["x".into(), "nw".into()],
                output: "h".into(),
                kind: OpKind::Row(RowShape {
                    rows: 512,
                    feat: 512,
                    operands: 2,
                    reduce: true,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
            OpSpec {
                name: "proj".into(),
                inputs: vec!["h".into(), "w".into()],
                output: "y".into(),
                kind: OpKind::Gemm(GemmShape {
                    m: 512,
                    n: 512,
                    k: 512,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
        ],
    }
}

fn emit() -> Program {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &Config::default());
    emit_program(&g, &cons, &s.tasks, &s.schedule)
}

#[test]
fn lowers_each_op_to_its_body() {
    let prog = emit();

    // The GEMM body carries full problem + tile dims; a DMA only bytes/slot.
    let gemm = prog.insts.iter().find_map(|i| match i.body {
        Body::Gemm {
            m,
            n,
            k,
            bm,
            bn,
            bk,
            ..
        } => Some((m, n, k, bm, bn, bk)),
        _ => None,
    });
    let (m, n, k, bm, bn, bk) = gemm.expect("a GEMM instruction");
    assert!(m > 0 && n > 0 && k > 0 && bm > 0 && bn > 0 && bk > 0);
    assert!(prog
        .insts
        .iter()
        .any(|i| matches!(i.body, Body::Dma { load: true, bytes, .. } if bytes > 0)));
    assert!(prog
        .insts
        .iter()
        .any(|i| matches!(i.body, Body::Row { reduce: true, .. })));

    // Counter windows reference real counters.
    let n_ctr = prog.counters.len() as u32;
    for i in &prog.insts {
        assert!(i.wait.iter().chain(&i.succ).all(|&c| c < n_ctr));
    }
    assert!(prog.counters.iter().all(|c| c.threshold >= 1));
}

#[test]
fn stream_round_trips() {
    let prog = emit();
    let bytes = prog.to_bytes();
    assert_eq!(Program::decode(&bytes).unwrap(), prog);
}

#[test]
fn variable_length_beats_fixed_width() {
    // The compact stream is far smaller than a fixed 96-byte-per-op struct array.
    let prog = emit();
    let bytes = prog.to_bytes().len();
    let fixed = prog.insts.len() * 96;
    assert!(
        bytes < fixed,
        "variable stream {bytes} not smaller than fixed {fixed}"
    );
}

#[test]
fn opcode_is_u16_structured() {
    let prog = emit();
    // All GEMM instructions carry family=GEMM with variant=BF16 (default dtype)
    let expected = Opcode::new(0, Opcode::FAMILY_GEMM, Opcode::VARIANT_BF16);
    for i in &prog.insts {
        if matches!(i.body, Body::Gemm { .. }) {
            assert_eq!(i.body.opcode(), expected);
            assert_eq!(i.body.opcode().family(), Opcode::FAMILY_GEMM);
        }
    }
}

#[test]
fn stream_header_carries_metadata() {
    let prog = emit();
    // Default emit sets bucket_id=0, plan_gen=0
    assert_eq!(prog.bucket_id, 0);
    assert_eq!(prog.plan_gen, 0);

    // Round-trip preserves metadata
    let mut p2 = prog.clone();
    p2.bucket_id = 123;
    p2.plan_gen = 42;
    let decoded = Program::decode(&p2.to_bytes()).unwrap();
    assert_eq!(decoded.bucket_id, 123);
    assert_eq!(decoded.plan_gen, 42);
}

fn qwen_plan() -> LayerPlan {
    let json = r#"{
      "model_type":"qwen3_5",
      "dtype":"bfloat16",
      "text_config": {
        "model_type":"qwen3_5_text",
        "vocab_size":256,
        "hidden_size":64,
        "intermediate_size":128,
        "num_hidden_layers":4,
        "num_attention_heads":24,
        "num_key_value_heads":4,
        "head_dim":16,
        "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
        "linear_conv_kernel_dim":4,
        "linear_key_head_dim":8,
        "linear_num_key_heads":2,
        "linear_num_value_heads":4,
        "linear_value_head_dim":8,
        "rms_norm_eps":0.000001,
        "rope_parameters":{"rope_theta":10000000.0,"partial_rotary_factor":0.25,"rope_type":"default","mrope_interleaved":true},
        "attention_bias":false,
        "attn_output_gate":true,
        "hidden_act":"silu",
        "mamba_ssm_dtype":"float32",
        "output_gate_type":"swish",
        "tie_word_embeddings":false
      }
    }"#;
    let mut graph = nn_graph::models::build_text_generation_from_config_json_at(
        json,
        &nn_graph::models::ShapeBucket::default(),
    )
    .unwrap();
    graph.bind(&Bindings::new().set("B", 1).set("S", 8));
    plan_from_all_blocks(&graph).unwrap()
}

fn gemma4_plan() -> LayerPlan {
    let json = r#"{
      "model_type":"gemma4",
      "dtype":"bfloat16",
      "text_config": {
        "model_type":"gemma4_text",
        "vocab_size":256,
        "hidden_size":64,
        "intermediate_size":128,
        "num_hidden_layers":2,
        "num_attention_heads":4,
        "num_key_value_heads":2,
        "num_global_key_value_heads":1,
        "head_dim":16,
        "global_head_dim":32,
        "attention_k_eq_v":true,
        "tie_word_embeddings":true,
        "final_logit_softcapping":30.0,
        "use_qk_norm":true,
        "sliding_window":32,
        "layer_types":["sliding_attention","full_attention"],
        "rope_parameters": {
          "full_attention": {
            "rope_theta":1000000.0,
            "partial_rotary_factor":0.25,
            "rope_type":"proportional"
          },
          "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}
        }
      }
    }"#;
    let mut graph = nn_graph::models::build_text_generation_from_config_json_at(
        json,
        &nn_graph::models::ShapeBucket::default(),
    )
    .unwrap();
    graph.bind(&Bindings::new().set("B", 1).set("S", 1));
    plan_from_all_blocks(&graph).unwrap()
}

#[test]
fn gemma4_plan_and_packets_preserve_numeric_semantics() {
    let plan = gemma4_plan();
    let model_ops = plan
        .ops
        .iter()
        .filter_map(|op| match op.kind {
            OpKind::Model(model) => Some(model),
            _ => None,
        })
        .collect::<Vec<_>>();

    let scales = model_ops
        .iter()
        .filter(|model| model.kind == ModelOpKind::Scale)
        .map(|model| f32::from_bits(model.args[0]))
        .collect::<Vec<_>>();
    assert_eq!(scales, vec![8.0, 1.0 / 30.0, 30.0]);
    assert_eq!(
        model_ops
            .iter()
            .filter(|model| model.kind == ModelOpKind::Tanh)
            .count(),
        1
    );
    assert_eq!(
        model_ops
            .iter()
            .filter(|model| model.kind == ModelOpKind::RmsNorm && model.operands == 1)
            .count(),
        2
    );
    assert!(model_ops.iter().any(|model| {
        model.kind == ModelOpKind::Rope && model.args[0] == 8 && model.args[3] == 32
    }));

    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (graph, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let scheduled = schedule(&soc, &graph, &cons, &Config::default());
    let program = emit_program(&graph, &cons, &scheduled.tasks, &scheduled.schedule);
    let decoded = Program::decode(&program.to_bytes()).unwrap();
    let variants = decoded
        .insts
        .iter()
        .filter_map(|inst| match inst.body {
            Body::Row { variant, args, .. } => Some((variant, args)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(variants
        .iter()
        .any(|(variant, _)| *variant == Opcode::VARIANT_MODEL_SCALE));
    assert!(variants
        .iter()
        .any(|(variant, _)| *variant == Opcode::VARIANT_MODEL_TANH));
    assert!(variants.iter().any(|(variant, args)| {
        *variant == Opcode::VARIANT_MODEL_ROPE && args[0] == 8 && args[3] == 32
    }));
}

#[test]
fn qwen_plan_is_complete_and_emits_semantic_packets_for_both_nvidia_targets() {
    let plan = qwen_plan();
    assert!(matches!(
        plan.ops.first().map(|o| o.kind),
        Some(OpKind::Model(m)) if m.kind == ModelOpKind::Embedding
            && m.input_bytes[1] == 256 * 64 * 2
    ));
    assert!(matches!(
        plan.ops.get(plan.ops.len() - 2).map(|o| o.kind),
        Some(OpKind::Model(m)) if m.kind == ModelOpKind::RmsNormZeroCentered
    ));
    assert!(matches!(
        plan.ops.last().map(|o| o.kind),
        Some(OpKind::Gemm(g)) if g.n == 256 && g.k == 64
    ));
    let attn = plan
        .ops
        .iter()
        .find_map(|o| match o.kind {
            OpKind::Flash(a) => Some(a),
            _ => None,
        })
        .unwrap();
    assert_eq!((attn.heads, attn.kv_heads), (24, 4));
    assert!(attn.causal);
    assert_eq!(attn.sliding_window, 0);

    for target in ["H100 SXM5", "rtx6000"] {
        let spec = hwspec::registry::lookup(target).unwrap();
        let soc = Soc::single(spec, DEFAULT_PAGE_BYTES);
        let (graph, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
        let scheduled = schedule(&soc, &graph, &cons, &Config::default());
        let embedding_load = scheduled
            .tasks
            .tasks
            .iter()
            .find(|task| task.tensor.as_deref() == Some("model.language_model.embed_tokens.weight"))
            .unwrap();
        assert_eq!(embedding_load.tensor_bytes, 256 * 64 * 2);
        assert_eq!(embedding_load.bytes, 8 * 64 * 2);
        let program = emit_program(&graph, &cons, &scheduled.tasks, &scheduled.schedule);
        let decoded = Program::decode(&program.to_bytes()).unwrap();

        let variants: Vec<_> = decoded
            .insts
            .iter()
            .filter_map(|i| match i.body {
                Body::Row { variant, args, .. } => Some((variant, args)),
                _ => None,
            })
            .collect();
        assert!(
            variants.iter().any(|(v, args)| {
                *v == Opcode::VARIANT_MODEL_QWEN_GATED_DELTA && *args == [4, 8, 1, 0]
            }),
            "{target}: missing Qwen GDN packet"
        );
        assert!(
            variants.iter().any(|(v, args)| {
                *v == Opcode::VARIANT_MODEL_CAUSAL_DEPTHWISE_CONV1D && args[0] == 4
            }),
            "{target}: missing causal depthwise-conv packet"
        );
        for required in [
            Opcode::VARIANT_MODEL_EMBEDDING,
            Opcode::VARIANT_MODEL_RMSNORM,
            Opcode::VARIANT_MODEL_RMSNORM_ZERO_CENTERED,
            Opcode::VARIANT_MODEL_ROPE,
            Opcode::VARIANT_MODEL_SILU,
            Opcode::VARIANT_MODEL_SIGMOID,
            Opcode::VARIANT_MODEL_ADD,
            Opcode::VARIANT_MODEL_MUL,
        ] {
            assert!(
                variants.iter().any(|(variant, _)| *variant == required),
                "{target}: missing semantic row variant {required:#04x}"
            );
        }
        let flash = decoded.insts.iter().find_map(|i| match i.body {
            Body::Flash {
                heads,
                kv_heads,
                variant,
                ..
            } => Some((heads, kv_heads, variant)),
            _ => None,
        });
        assert_eq!(flash, Some((24, 4, Opcode::VARIANT_FLASH_CAUSAL_BF16)));
        assert!(decoded.insts.iter().all(|i| i.unit < spec.sm_count as u8));
    }
}
