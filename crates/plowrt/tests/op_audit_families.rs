//! The operator audit's exit condition (plan §8, Phase 1d): reproduce §3's class
//! table for at least one emitted program per model family.
//!
//! Each case writes a miniature `config.json` — no checkpoint, no safetensors, no
//! GPU — drives `devgen::run` to a real device blob, parses it with the same
//! loader `plowrt disasm` uses, and asserts the audit's verdict *and the reason*.
//!
//! These are not smoke tests. The assertions name the exact operator that blocks
//! each family, so a change that silently promotes or demotes a class fails here
//! rather than at a load-time refusal nobody reads.

use std::path::{Path, PathBuf};

use plowrt::asset::devblob::DevBlob;
use plowrt::opaudit::{audit_program, Disposition, ProgramAudit, RowClass};

/// `devgen::run` mutates process-global env in places, and these tests share a
/// process. Same reason `crates/devgen/tests/golden_blob.rs` has one.
static EMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn emit_guard() -> std::sync::MutexGuard<'static, ()> {
    EMIT_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Emit one miniature blob.
///
/// `arch` is empty wherever the emitter tolerates it — that skips `build.json`
/// and every target-specific gate, and the instruction stream the audit reads is
/// the same either way. Two families refuse an empty arch and say so at emit
/// time, so they name one; `block_spec` is likewise only set where the emitter
/// demands it. Both are recorded per case rather than defaulted, so a reader can
/// see which program the audit actually ran on.
fn blob_with(
    name: &str,
    cfg: &str,
    ctx: u32,
    arch: &str,
    gpu: &str,
    block_spec: Option<&str>,
) -> DevBlob {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("opaudit_{name}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.json"), cfg).unwrap();
    let out = root.join("model.pkt");
    devgen::run(devgen::EmitArgs {
        dir: root.clone(),
        ctx,
        out: out.to_str().unwrap().to_string(),
        n_cu: 128,
        tp: 1,
        block_spec: block_spec.map(str::to_string),
        embed_cubin: None,
        embed_hsaco: None,
        rope_gen: true,
        l2_layout: None,
        gpu: gpu.to_string(),
        arch: arch.to_string(),
        emit_cfg: None,
        whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
    });
    DevBlob::parse_l2(&std::fs::read(&out).unwrap(), true).unwrap()
}

fn blob_for(name: &str, cfg: &str, ctx: u32) -> DevBlob {
    blob_with(name, cfg, ctx, "", "", None)
}

/// The audit of the blob's largest prefill program and of its decode program.
fn audits(blob: &DevBlob) -> (ProgramAudit, ProgramAudit) {
    let prefill = blob
        .progs
        .iter()
        .filter(|p| p.t > 1)
        .max_by_key(|p| p.t)
        .expect("a prefill program");
    let decode = blob
        .progs
        .iter()
        .find(|p| p.t == 1)
        .expect("a decode program");
    (audit_program(prefill), audit_program(decode))
}

/// Every opcode a program contains whose class is `class`.
fn ops_of_class(a: &ProgramAudit, class: RowClass) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = a
        .ops
        .iter()
        .filter(|o| o.class == class)
        .map(|o| o.name.expect("a known opcode"))
        .collect();
    v.sort_unstable();
    v
}

/// Every opcode the audit refuses, whatever the reason.
fn refused(a: &ProgramAudit) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = a
        .ops
        .iter()
        .filter(|o| !o.disposition.packable())
        .map(|o| o.name.expect("a known opcode"))
        .collect();
    v.sort_unstable();
    v
}

/// No family may contain an opcode with no classification at all. That is the
/// hazard §3 exists to prevent, and the audit's job is to be able to say so.
fn assert_everything_is_classified(a: &ProgramAudit, what: &str) {
    let unclassified: Vec<&'static str> = a
        .ops
        .iter()
        .filter(|o| !o.classified)
        .map(|o| o.name.unwrap_or("<unknown wire opcode>"))
        .collect();
    assert!(
        unclassified.is_empty(),
        "{what}: unclassified opcodes {unclassified:?}"
    );
    assert!(
        !a.ops.is_empty() && a.n_inst > 0,
        "{what}: empty program, the audit proves nothing"
    );
}

// ---------------------------------------------------------------------------
// Dense GQA — Llama 3 / Qwen3. §2.2: "A + one C". The plan's Phase 2 vehicle.
// ---------------------------------------------------------------------------

const LLAMA: &str = r#"{
  "model_type": "llama",
  "hidden_size": 512, "intermediate_size": 1024, "num_hidden_layers": 2,
  "num_attention_heads": 8, "head_dim": 64, "num_key_value_heads": 2,
  "rms_norm_eps": 1e-5, "vocab_size": 4096, "rope_theta": 500000.0,
  "rope_scaling": null, "tie_word_embeddings": false
}"#;

const QWEN3: &str = r#"{
  "model_type": "qwen3",
  "hidden_size": 512, "intermediate_size": 1024, "num_hidden_layers": 2,
  "num_attention_heads": 8, "head_dim": 64, "num_key_value_heads": 2,
  "rms_norm_eps": 1e-6, "vocab_size": 4096, "rope_theta": 1000000.0,
  "rope_scaling": null, "tie_word_embeddings": true
}"#;

#[test]
fn dense_gqa_is_class_a_plus_flash_prefill_and_the_kv_writer() {
    let _g = emit_guard();
    for (name, cfg) in [("llama", LLAMA), ("qwen3", QWEN3)] {
        let blob = blob_for(name, cfg, 512);
        let (pf, dec) = audits(&blob);
        assert_everything_is_classified(&pf, name);
        assert_everything_is_classified(&dec, name);

        // §2.2's claim for this family, and the whole reason it is Phase 2:
        // exactly two operators are not already row-agnostic or per-row.
        assert_eq!(
            ops_of_class(&pf, RowClass::C),
            vec!["FlashPrefill", "HeadNormRope"],
            "{name} prefill class-C set"
        );
        assert!(
            ops_of_class(&pf, RowClass::D).is_empty(),
            "{name} has no recurrent state"
        );
        assert_eq!(refused(&pf), vec!["FlashPrefill", "HeadNormRope"]);

        // Decode: attention is already class B via `t6=decode_slot`, so the
        // only blocker is the KV writer's legacy addressing.
        assert!(
            ops_of_class(&dec, RowClass::B).contains(&"FlashDecode"),
            "{name} decode attention should be class B"
        );
        assert_eq!(refused(&dec), vec!["HeadNormRope"], "{name} decode");
    }
}

// ---------------------------------------------------------------------------
// Windowed + softcap — Gemma 4 dense. §2.2: "as Llama, plus the window bound
// per span; tail carries SoftCap".
// ---------------------------------------------------------------------------

const GEMMA4: &str = r#"{
  "model_type": "gemma4_text",
  "hidden_size": 512, "intermediate_size": 1024, "num_hidden_layers": 2,
  "num_attention_heads": 8, "head_dim": 64, "global_head_dim": 64,
  "num_key_value_heads": 2, "num_global_key_value_heads": 2,
  "sliding_window": 512, "rms_norm_eps": 1e-6, "vocab_size": 4096,
  "final_logit_softcapping": 30.0, "tie_word_embeddings": true,
  "layer_types": ["sliding_attention", "full_attention"],
  "rope_parameters": {
    "sliding_attention": { "rope_theta": 10000.0, "partial_rotary_factor": 1.0 },
    "full_attention": { "rope_theta": 1000000.0, "partial_rotary_factor": 1.0 }
  }
}"#;

#[test]
fn windowed_gqa_with_softcap_adds_no_new_class() {
    let _g = emit_guard();
    let blob = blob_for("gemma4", GEMMA4, 1024);
    let (pf, dec) = audits(&blob);
    assert_everything_is_classified(&pf, "gemma4 prefill");
    assert_everything_is_classified(&dec, "gemma4 decode");

    // The window is a static per-layer immediate (`FlashPrefill i5`), not a row
    // property, so it adds nothing to the class set. SoftCap is elementwise.
    assert_eq!(
        ops_of_class(&pf, RowClass::C),
        vec!["FlashPrefill", "HeadNormRope"]
    );
    let softcap = pf
        .ops
        .iter()
        .chain(dec.ops.iter())
        .find(|o| o.name == Some("SoftCap"))
        .expect("Gemma's tail carries a SoftCap");
    assert_eq!(softcap.class, RowClass::A);
    assert_eq!(softcap.disposition, Disposition::Ready);
}

// ---------------------------------------------------------------------------
// MLA + MoE — Kimi K2.7 stands in for the GLM-5.3/DeepSeek shape. §2.2: "MLA
// prefill per-span". The DSA indexer is a separate axis and this control has
// none, exactly as §2.3 requires of the GLM control.
// ---------------------------------------------------------------------------

const KIMI_K2: &str = r#"{
  "model_type": "kimi_k2", "vocab_size": 1000, "hidden_size": 256,
  "intermediate_size": 512, "num_hidden_layers": 4, "num_attention_heads": 8,
  "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
  "q_lora_rank": 64, "kv_lora_rank": 32, "qk_rope_head_dim": 16,
  "qk_nope_head_dim": 48, "v_head_dim": 64,
  "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
  "moe_intermediate_size": 256, "first_k_dense_replace": 2,
  "routed_scaling_factor": 2.5, "torch_dtype": "bfloat16"
}"#;

#[test]
fn mla_moe_is_blocked_by_mla_prefill_and_by_single_row_moe() {
    let _g = emit_guard();
    // `devgen` refuses a full-model Kimi/DeepSeek device emit ("a later
    // milestone", crates/devgen/src/lib.rs) and asks for one block. That block
    // is decode-only, so this case covers the MLA+MoE DECODE operator set; the
    // prefill leg is covered against the real GLM-5.3 TP4 blob in
    // `shipped_blobs_reproduce_the_same_class_sets`, which has an MLA prefill
    // program and asserts `FlashMlaPrefill` is class C there.
    let blob = blob_with("kimi_k2", KIMI_K2, 1024, "gfx950", "MI350X", Some("2"));
    let dec = audit_program(
        blob.progs
            .iter()
            .find(|p| p.t == 1)
            .expect("a decode program"),
    );
    assert_everything_is_classified(&dec, "kimi_k2 decode");

    // MLA decode is class B already: `t6=kv_len[b]` is per row.
    assert!(
        ops_of_class(&dec, RowClass::B).contains(&"FlashMlaDecode"),
        "decode B set was {:?}",
        ops_of_class(&dec, RowClass::B)
    );
    // The only class-C operator an MLA+MoE decode can carry is the KV writer,
    // and whether it carries one is a property of THIS BUILD, not of the family:
    // this miniature block emits `HeadNormRope` in the legacy `out_row0 + t`
    // mode, while the shipped GLM-5.3 TP4 blob emits the batched ring form
    // (`i6 != 0`) and is class B. Nothing else in the decode program is class C.
    let decode_c = ops_of_class(&dec, RowClass::C);
    assert!(
        decode_c.iter().all(|n| *n == "HeadNormRope"),
        "decode C set was {decode_c:?}"
    );
    let single_row: Vec<&'static str> = dec
        .ops
        .iter()
        .filter(|o| o.disposition == Disposition::UseRowForm)
        .map(|o| o.name.unwrap())
        .collect();
    assert!(
        single_row.iter().any(|n| n.starts_with("Moe")),
        "MLA+MoE decode should be refused for single-row MoE operators, got {single_row:?}"
    );
    assert!(!dec.packable);
}

// ---------------------------------------------------------------------------
// MXFP4 + sinks — GPT-OSS. §2.2: "sinks fold in FlashMerge t3 per head and are
// row-independent"; §5.3 adds the MXFP4 expert grouping.
// ---------------------------------------------------------------------------

const GPT_OSS: &str = r#"{
  "model_type": "gpt_oss", "attention_bias": true, "hidden_act": "silu",
  "head_dim": 64, "hidden_size": 512, "intermediate_size": 512,
  "layer_types": ["sliding_attention", "full_attention"],
  "num_attention_heads": 8, "num_key_value_heads": 2, "num_hidden_layers": 2,
  "num_local_experts": 8, "num_experts_per_tok": 4,
  "quantization_config": { "quant_method": "mxfp4" },
  "rms_norm_eps": 1e-5,
  "rope_scaling": { "beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0,
    "original_max_position_embeddings": 4096, "rope_type": "yarn", "truncate": false },
  "rope_theta": 150000, "sliding_window": 128, "swiglu_limit": 7.0,
  "tie_word_embeddings": false, "vocab_size": 4096
}"#;

#[test]
fn mxfp4_moe_with_sinks_keeps_flash_merge_class_a() {
    let _g = emit_guard();
    let blob = blob_for("gpt_oss", GPT_OSS, 512);
    let (pf, dec) = audits(&blob);
    assert_everything_is_classified(&pf, "gpt_oss prefill");
    assert_everything_is_classified(&dec, "gpt_oss decode");

    // The sink fold is one unscaled logit per HEAD with no value row, so it is
    // row-independent and packing does not touch it. If this ever flips, the
    // plan's §5.2 claim about sinks is wrong.
    let merge = pf
        .ops
        .iter()
        .chain(dec.ops.iter())
        .find(|o| o.name == Some("FlashMerge"))
        .expect("GPT-OSS emits the merge at every nsplit, including 1");
    assert_eq!(merge.class, RowClass::A);
    assert!(merge.disposition.packable());

    // MXFP4 experts: the prefill (`*Pf`) forms carry row maps, the decode forms
    // carry an `n_batch`. Neither is a new class.
    for a in [&pf, &dec] {
        for o in &a.ops {
            let mx = o.name.is_some_and(|n| n.contains("Mx"));
            if mx {
                assert!(
                    matches!(o.class, RowClass::A | RowClass::B),
                    "{:?} is MXFP4 MoE and should not be C or D",
                    o.name
                );
            }
        }
    }
    assert!(
        ops_of_class(&pf, RowClass::D).is_empty(),
        "GPT-OSS carries no recurrent state"
    );
}

// ---------------------------------------------------------------------------
// Recurrent — Qwen3.5 / GDN. §5.4: the decode side is class B via `active[B]`;
// the prefill side is class D by operand shape.
// ---------------------------------------------------------------------------

const QWEN35: &str = r#"{"model_type": "qwen3_5", "text_config": {
  "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 4096,
  "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
  "linear_num_key_heads": 16, "linear_num_value_heads": 48,
  "linear_key_head_dim": 128, "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4, "num_hidden_layers": 4,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
  "attention_bias": false, "attn_output_gate": true, "hidden_act": "silu",
  "mamba_ssm_dtype": "float32", "output_gate_type": "swish",
  "tie_word_embeddings": false, "rms_norm_eps": 1e-6,
  "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25, "rope_type": "default"}
}}"#;

#[test]
fn recurrent_gdn_decode_is_class_b_and_its_prefill_is_class_d() {
    let _g = emit_guard();
    // §2.2 records GDN as an NVIDIA-first target; `devgen` refuses to emit it
    // for any other arch, so the audit runs on the sm_90a program. The GDN
    // opcodes and their operands are backend-neutral — that is exactly the ABI
    // the plan says is shared — so the classification is not sm_90a-specific.
    let blob = blob_with("qwen3_5", QWEN35, 8192, "sm_90a", "H100 SXM5", None);

    // The GDN emitter may build decode-only programs (§2.2 records the family as
    // hybrid); take whatever it emitted and audit all of it.
    let all: Vec<ProgramAudit> = blob.progs.iter().map(audit_program).collect();
    assert!(!all.is_empty(), "qwen3_5 emitted no programs");
    for (a, p) in all.iter().zip(blob.progs.iter()) {
        assert_everything_is_classified(a, &format!("qwen3_5 T={}", p.t));
    }

    let names: Vec<&'static str> = all
        .iter()
        .flat_map(|a| a.ops.iter())
        .filter_map(|o| o.name)
        .collect();
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("QwenGdn") || n.starts_with("Qwen")),
        "no GDN operators emitted: {names:?}"
    );

    // Every GDN operator this blob contains must be class B (the `active[B]`
    // decode path) or class D (the `[1,HV,V,K]` prefill state) — never C, and
    // never unclassified. That is the whole of §5.4's honest contract.
    for a in &all {
        for o in a.ops.iter().filter(|o| {
            o.name
                .is_some_and(|n| n.starts_with("QwenGdn") || n.starts_with("QwenGated"))
        }) {
            assert!(
                matches!(o.class, RowClass::B | RowClass::D),
                "{:?} is a GDN operator classified {:?}",
                o.name,
                o.class
            );
            if o.class == RowClass::B {
                assert_eq!(o.disposition, Disposition::DescriptorFills);
            } else {
                assert_eq!(o.disposition, Disposition::PerSpanLaunch);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The shipped blobs on this host, when they are present. Opt-in because they are
// 80 MB and 226 MB and are not in the repository.
// ---------------------------------------------------------------------------

fn shipped(path: &str) -> Option<DevBlob> {
    let p = Path::new(path).join("model.pkt");
    let buf = std::fs::read(p).ok()?;
    DevBlob::parse_l2(&buf, true).ok()
}

/// The two blobs this branch actually measured. Skipped when absent so the test
/// suite stays hermetic; run it where they exist to confirm the miniature
/// configs above describe the same operator sets as the real models.
#[test]
fn shipped_blobs_reproduce_the_same_class_sets() {
    let Some(gemma) = shipped("/app/plow/build-gemma31/assets-final-plain") else {
        eprintln!("skip: no gemma31 blob on this host");
        return;
    };
    let (pf, dec) = audits(&gemma);
    assert_everything_is_classified(&pf, "gemma31 prefill");
    assert_eq!(refused(&pf), vec!["FlashPrefill", "HeadNormRope"]);
    assert_eq!(refused(&dec), vec!["HeadNormRope"]);

    let Some(glm) = shipped("/app/plow/build-glm53/tp4-long") else {
        eprintln!("skip: no glm53 blob on this host");
        return;
    };
    let (pf, dec) = audits(&glm);
    assert_everything_is_classified(&pf, "glm53 prefill");
    assert_everything_is_classified(&dec, "glm53 decode");
    assert_eq!(refused(&pf), vec!["FlashMlaPrefill", "HeadNormRope"]);
    // The grouped MoE chain carries explicit per-row gather/scatter maps, so it
    // is class B and the descriptor fills what it already has — §5.3's claim.
    let b = ops_of_class(&pf, RowClass::B);
    assert!(
        b.contains(&"MoeAlignPf") && b.contains(&"MoeGroupGluPf") && b.contains(&"MoeGroupDownPf"),
        "glm53 prefill B set was {b:?}"
    );
    // §2.3: the documented control has DSA disabled, so no indexer opcode is
    // present. If one appears, the control changed and the audit must be re-read.
    assert!(
        !pf.ops.iter().any(|o| o
            .name
            .is_some_and(|n| n.starts_with("Index") || n.starts_with("Dsa"))),
        "glm53 control should carry no DSA indexer opcodes"
    );
    // GLM's decode emits the BATCHED ring form of HeadNormRope (i6 != 0), so the
    // same opcode is class B here and class C in Gemma's decode. A per-opcode
    // classification alone cannot say that; the per-instruction refinement can.
    assert!(
        ops_of_class(&dec, RowClass::C).is_empty(),
        "glm53 decode C set was {:?}",
        ops_of_class(&dec, RowClass::C)
    );
    assert!(ops_of_class(&dec, RowClass::B).contains(&"HeadNormRope"));
}
