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
