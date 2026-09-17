//! `deepseek_v41` must be CLAIMED by the emitter, not left to fall through.
//!
//! V4.1 nests its geometry under `text_config`, and `cfg_from`
//! (crates/devgen/src/config.rs) treats any config with that key as "Gemma-4
//! multimodal" before it looks at `model_type` at all. So an unclaimed V4.1
//! checkpoint does not report "unsupported model" — it dies unwrapping Gemma's
//! `layer_types`, naming the wrong field of the wrong architecture. That is the
//! exact failure `kimi_k3` and `kimi_k25` are claimed early to avoid, and this
//! pins it for V4.1 so a later refactor of the claim order cannot reopen it.
//!
//! The assertion is on WHICH error, not merely that one happens: a panic is the
//! correct outcome today (there is no device emit yet — see
//! docs/amd/deepseek-v41-flash-mi300x.md section 5.6), so "it panicked" alone
//! would pass just as well before the claim as after it.

use std::path::PathBuf;

fn write_min_config(dir: &PathBuf, model_type: &str) {
    // The shape that matters: `text_config` present, so the Gemma probe would
    // claim it. Nothing else needs to be real — the claim fires before any
    // geometry is read, which is itself part of the contract.
    let cfg = format!(
        r#"{{"model_type":"{model_type}",
            "architectures":["DeepseekV41ForCausalLM"],
            "text_config":{{"model_type":"deepseek_v41_text",
                            "num_hidden_layers":40,
                            "hidden_size":5120,
                            "hc_mult":4}}}}"#
    );
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), cfg).unwrap();
}

fn emit(dir: PathBuf) {
    devgen::run_verified(
        devgen::EmitArgs {
            dir: dir.clone(),
            ctx: 2048,
            out: dir.join("model.pkt").display().to_string(),
            n_cu: 304,
            tp: 1,
            block_spec: None,
            embed_cubin: None,
            embed_hsaco: None,
            rope_gen: true,
            l2_layout: None,
            gpu: String::new(),
            arch: "gfx942".into(),
            emit_cfg: None,
            whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
        },
        None,
    );
}

/// `tag` keeps each test's scratch directory distinct: two tests drive the same
/// `model_type`, and cargo runs them in parallel, so a shared path let one read
/// the config while the other was still writing it -- which parses as nothing,
/// leaves `model_type` empty, and falls through to the very Gemma probe this
/// file exists to prove we do not reach.
fn claim_message(model_type: &str, tag: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "dsv41-claim-{model_type}-{tag}-{}",
        std::process::id()
    ));
    write_min_config(&dir, model_type);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let err = std::panic::catch_unwind(move || emit(dir)).unwrap_err();
    std::panic::set_hook(hook);
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn the_wrapper_model_type_is_claimed_and_names_itself() {
    let msg = claim_message("deepseek_v41", "names-itself");
    assert!(
        msg.contains("deepseek_v41"),
        "the refusal must name the architecture it is refusing; got: {msg}"
    );
    assert!(
        !msg.contains("layer_types"),
        "fell through to the Gemma-4 probe — the claim is not early enough. got: {msg}"
    );
}

#[test]
fn the_text_tower_model_type_is_claimed_too() {
    // The `-text` re-export carries the same tower under a flat config, and the
    // `kimi`/`deepseek_v3` arm would happily parse its MLA keys (same spelling)
    // and emit a blob missing CSA2, Engram and the two-level indexer.
    let msg = claim_message("deepseek_v41_text", "text-tower");
    assert!(
        msg.contains("deepseek_v41"),
        "the text tower must be claimed as well; got: {msg}"
    );
}

#[test]
fn the_refusal_says_what_is_missing_and_where_it_is_tracked() {
    let msg = claim_message("deepseek_v41", "what-is-missing");
    for needle in ["CSA2", "Engram", "indexer", "5.6"] {
        assert!(
            msg.contains(needle),
            "the refusal should point at {needle:?} so a reader knows what to build; got: {msg}"
        );
    }
}

/// With the REAL checkpoint present, the refusal must carry its validated geometry -- not a
/// hardcoded copy of it. This is what makes the `devgen -> nn-graph` edge worth having: the
/// config is parsed and validated in exactly one place, and devgen reports what that place
/// found. Skips when the shards are not on this machine.
#[test]
fn the_refusal_carries_the_real_checkpoints_validated_geometry() {
    let hf = std::path::PathBuf::from("/workspace/models/DeepSeek-V4.1-Flash");
    if !hf.join("config.json").exists() {
        eprintln!("skipping: DeepSeek-V4.1-Flash not present");
        return;
    }
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let err = std::panic::catch_unwind(move || emit(hf)).unwrap_err();
    std::panic::set_hook(hook);
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();

    assert!(
        msg.contains("VALIDATES"),
        "the real config should parse AND validate through nn-graph; got: {msg}"
    );
    // Read off the released checkpoint: 40 layers, kv_source [2, 8, 14, 20], engram [1, 14].
    for needle in ["40 layers", "4 kv_source layers", "[2, 8, 14, 20]", "[1, 14]"] {
        assert!(
            msg.contains(needle),
            "the refusal should report {needle:?} from the checkpoint; got: {msg}"
        );
    }
}

/// Helper: the refusal text for the REAL checkpoint, or `None` when the shards are absent.
fn real_checkpoint_message() -> Option<String> {
    let hf = std::path::PathBuf::from("/workspace/models/DeepSeek-V4.1-Flash");
    if !hf.join("config.json").exists() {
        eprintln!("skipping: DeepSeek-V4.1-Flash not present");
        return None;
    }
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let err = std::panic::catch_unwind(move || emit(hf)).unwrap_err();
    std::panic::set_hook(hook);
    Some(
        err.downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    )
}

/// The attention shape, which is the thing the config does NOT say and the shards do.
///
/// V4.1's MLA is fully absorbed: `wkv` is `[512, 5120]` and there is no `kv_b` tensor anywhere in
/// the 48 shards. So the 512 is one latent per token shared by all 64 heads
/// (`num_key_value_heads = 1`) acting as both K and V -- NOT a per-head width, and not something
/// to shard under TP. Reading it as V4's split latent is the misread this whole claim exists to
/// prevent, and it would produce a blob that loads and runs.
#[test]
fn the_refusal_states_the_absorbed_mla_and_the_grouped_output_lora() {
    let Some(msg) = real_checkpoint_message() else {
        return;
    };
    for needle in ["FULLY ABSORBED", "512-wide latent", "no kv_b tensor", "64 heads"] {
        assert!(
            msg.contains(needle),
            "the refusal should state {needle:?} -- it is read from the shards, not the config; \
             got: {msg}"
        );
    }
    // 8 x 1024, the output LoRA, whose block-diagonal structure is why a plain Linear is wrong.
    assert!(
        msg.contains("grouped output LoRA (8 x 1024)"),
        "the refusal should report o_groups x o_lora_rank; got: {msg}"
    );
}

/// The shard cross-check must find NOTHING to complain about on the released checkpoint.
///
/// This is the half of the contract that can fail loudly: `dsv41_shard_check` compares every
/// layer-0 attention tensor against what the config implies, and the refusal prepends a WARNING
/// when they disagree. On the real checkpoint they agree, so the warning must be absent -- and if
/// this test ever starts failing, the config and the tensors have diverged and the geometry in the
/// refusal is not this checkpoint's.
#[test]
fn the_shards_do_not_contradict_the_config() {
    let Some(msg) = real_checkpoint_message() else {
        return;
    };
    assert!(
        !msg.contains("shards CONTRADICT"),
        "the released checkpoint's tensors should match its config; got: {msg}"
    );
}

/// The grouped output LoRA is NOT a blocker for a prefill emit, and the refusal has to say so.
///
/// Section 5.2 lists it under "new kernel work", which is true for DECODE (T=1 turns it into 8
/// tiny GEMVs that want one fused dispatch). It is NOT true for prefill: block-diagonal over 8
/// groups is 8 ordinary GEMMs of [T, 4096] x [4096, 1024], and at T=8192 one of those alone fills
/// 304 CUs. Left unstated, the item reads as gating item 3 of the critical path when it does not.
#[test]
fn the_output_lora_is_marked_not_a_prefill_blocker() {
    let Some(msg) = real_checkpoint_message() else {
        return;
    };
    assert!(
        msg.contains("NOT a blocker for a PREFILL emit"),
        "the refusal should say the grouped output LoRA does not gate a prefill emit; got: {msg}"
    );
    assert!(
        msg.contains("DECODE optimisation"),
        "...and should say where the fused kernel actually pays; got: {msg}"
    );
}
