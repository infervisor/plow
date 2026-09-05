//! End-to-end `--hf-dir` compile: synthesize a miniature Gemma-4-style
//! checkpoint directory (config.json + a real safetensors file with
//! checkpoint-exact tensor names), compile it with `Source::HfDir`, and check:
//!
//! - The full-model unrolled plan compiles to non-empty packet streams.
//! - Checkpoint validation passes when every tensor is present.
//! - Checkpoint validation HARD-FAILS when a tensor the plan needs is missing,
//!   and when the checkpoint ships a tensor the plan does not cover — the two
//!   silent-wrong-model failure modes the `gemma4` binary guards against.

use plowc::{compile, Options, Parallel, PlowcError, Source};
use schedule::Phase;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Miniature 2-layer Gemma-4 (1 sliding + 1 full): nested text_config,
/// per-layer geometry (full layer: global_head_dim, 1 kv head, k_eq_v → no
/// v_proj), tied embeddings.
const CONFIG: &str = r#"{
    "model_type": "gemma4_unified",
    "text_config": {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "head_dim": 64,
        "global_head_dim": 128,
        "intermediate_size": 512,
        "attention_k_eq_v": true,
        "tie_word_embeddings": true,
        "num_hidden_layers": 2,
        "vocab_size": 1024,
        "layer_types": ["sliding_attention", "full_attention"]
    }
}"#;

/// Every tensor of the miniature checkpoint: (name, shape), matching what the
/// real gemma-4 checkpoints ship per layer type.
fn tensors() -> Vec<(String, Vec<usize>)> {
    let p = "model.language_model.";
    let mut t: Vec<(String, Vec<usize>)> = vec![
        (format!("{p}embed_tokens.weight"), vec![1024, 256]),
        (format!("{p}norm.weight"), vec![256]),
    ];
    for (l, (hd, kvh, has_v)) in [(64usize, 2usize, true), (128, 1, false)]
        .into_iter()
        .enumerate()
    {
        let lp = format!("{p}layers.{l}.");
        t.push((format!("{lp}input_layernorm.weight"), vec![256]));
        t.push((format!("{lp}self_attn.q_proj.weight"), vec![4 * hd, 256]));
        t.push((format!("{lp}self_attn.k_proj.weight"), vec![kvh * hd, 256]));
        if has_v {
            t.push((format!("{lp}self_attn.v_proj.weight"), vec![kvh * hd, 256]));
        }
        t.push((format!("{lp}self_attn.o_proj.weight"), vec![256, 4 * hd]));
        t.push((format!("{lp}self_attn.q_norm.weight"), vec![hd]));
        t.push((format!("{lp}self_attn.k_norm.weight"), vec![hd]));
        t.push((format!("{lp}post_attention_layernorm.weight"), vec![256]));
        t.push((format!("{lp}pre_feedforward_layernorm.weight"), vec![256]));
        t.push((format!("{lp}post_feedforward_layernorm.weight"), vec![256]));
        t.push((format!("{lp}layer_scalar"), vec![1]));
        t.push((format!("{lp}mlp.gate_proj.weight"), vec![512, 256]));
        t.push((format!("{lp}mlp.up_proj.weight"), vec![512, 256]));
        t.push((format!("{lp}mlp.down_proj.weight"), vec![256, 512]));
    }
    t
}

/// Write a minimal valid safetensors file (bf16 zeros) with the given tensors.
fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>)]) {
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (i, (name, shape)) in tensors.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let bytes = numel * 2;
        if i > 0 {
            header.push(',');
        }
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{dims}],\"data_offsets\":[{offset},{}]}}",
            offset + bytes
        ));
        offset += bytes;
    }
    header.push('}');
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&vec![0u8; offset]).unwrap();
}

fn make_dir(tag: &str, tensors: &[(String, Vec<usize>)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("plowc-hf-dir-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    write_safetensors(&dir.join("model.safetensors"), tensors);
    dir
}

fn opts(out: PathBuf) -> Options {
    Options {
        no_tuning: false,
        tuning_db: None,
        gpu: "H100 SXM5".into(),
        num_gpus: 1,
        parallel: Parallel::Tp,
        batches: vec![1],
        seqs: vec![128],
        phases: vec![Phase::Decode, Phase::Prefill],
        page_kib: 16,
        out,
        lean_verify: false,
        counter_elim: false,
        scope_narrow: false,
        prefetch: false,
        sram_fit: false,
        lean_oracle: false,
        emit_sample: true,
        emit_tokenize: false,
        emit_trace: false,
        kv: Default::default(),
        weight_dtype_override: None,
    }
}

#[test]
fn hf_dir_compiles_end_to_end() {
    let dir = make_dir("ok", &tensors());
    let out = dir.join("out");
    let report = compile(&Source::HfDir(dir.clone()), &opts(out.clone()))
        .expect("hf-dir compile should succeed");
    assert_eq!(report.buckets.len(), 2, "decode + prefill buckets");
    for b in &report.buckets {
        assert!(
            b.packet_bytes > 0,
            "bucket {} emitted no packets",
            b.packet_file
        );
        assert!(out.join(&b.packet_file).exists());
    }
    assert!(out.join("weights.json").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hf_dir_fails_on_missing_checkpoint_tensor() {
    // Drop layer 1's k_norm — the plan references it, so this must hard-fail.
    let t: Vec<_> = tensors()
        .into_iter()
        .filter(|(n, _)| n != "model.language_model.layers.1.self_attn.k_norm.weight")
        .collect();
    let dir = make_dir("missing", &t);
    let err = compile(&Source::HfDir(dir.clone()), &opts(dir.join("out")))
        .expect_err("missing tensor must fail the compile");
    match err {
        PlowcError::Hub(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("1 missing") && msg.contains("k_norm"),
                "unexpected message: {msg}"
            )
        }
        other => panic!("expected checkpoint metadata error, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hf_dir_fails_on_uncovered_checkpoint_tensor() {
    // Ship an extra per-layer tensor the plan does not reference — silently
    // dropping part of the model must hard-fail.
    let mut t = tensors();
    t.push((
        "model.language_model.layers.0.mlp.secret_extra_proj.weight".into(),
        vec![512, 256],
    ));
    let dir = make_dir("uncovered", &t);
    let err = compile(&Source::HfDir(dir.clone()), &opts(dir.join("out")))
        .expect_err("uncovered tensor must fail the compile");
    match err {
        PlowcError::Hub(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("1 unexpected") && msg.contains("secret_extra_proj"),
                "unexpected message: {msg}"
            )
        }
        other => panic!("expected checkpoint metadata error, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}
