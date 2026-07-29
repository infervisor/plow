//! What a declared blob tensor's bytes come FROM — the one rule, in one place.
//!
//! # Why this is not four `starts_with("model.")`s
//!
//! Every `TensorDecl` in a PLOWDEV blob is filled from exactly one of four sources: the
//! HuggingFace checkpoint, the blob's own `init` section, a `gen` recipe (RoPE tables), or a
//! host-computed pointer table. Everything else is scratch and is zeroed. Deciding which is a
//! *classification of the name*, and until this module existed the classification was written out
//! by hand at five sites that did not agree with each other:
//!
//! | site | `model.` | `fp8/` | `lm_head` | excludes pointer tables |
//! |---|---|---|---|---|
//! | `plowrt/exec/gpu.rs` (CUDA bind) | yes | yes | **no** | **no** |
//! | `plowrt/exec/amd.rs` (HSA bind) | yes | yes | yes | yes |
//! | `plowrt/serve/manager.rs` (VRAM plan) | yes | yes | **no** | **no** |
//! | `plowc/bin/llama3.rs` (byte report) | yes | **no** | yes | **no** |
//! | `devgen/lib.rs` (byte report) | yes | yes | **no** | **no** |
//!
//! Two live consequences of the disagreement, both silent:
//!
//! * **Untied `lm_head` is zeroed on CUDA.** `devgen` declares a top-level `lm_head.weight` for
//!   models that do not tie the embedding (Llama, some Qwen). It does not start with `model.`, so
//!   `gpu.rs` classified it as scratch, `memset` it to zero and uploaded nothing. The AMD loader
//!   has an `lm_head` arm and does not have the bug; the CUDA one did.
//! * **A wrapper prefix zeroes the WHOLE model.** All 497 052 Kimi-K3 language-tower tensors are
//!   spelled `language_model.model.…` and *none* starts with `model.`. Under a prefix allowlist
//!   the loader allocates every weight, uploads none of them, reports "0 tensors, 0.00 GiB", and
//!   decodes fluent garbage. (Gemma-4 multimodal survives only by accident of spelling —
//!   `model.language_model.…` happens to start with `model.`.)
//!
//! # The rule
//!
//! The allowlist is inverted. The COMPILER's own namespaces are a small closed set it mints
//! itself ([`RUNTIME_PREFIXES`]); the CHECKPOINT's namespace is open and belongs to whoever
//! trained the model. So "weight" is defined by exclusion:
//!
//! > a declared tensor is checkpoint-bound unless devgen minted the name itself.
//!
//! That makes an unrecognised name a **hard `MISSING WEIGHT: <name>` error** naming the tensor,
//! instead of a silently zeroed buffer — which is the only difference that matters, because a
//! zeroed weight produces fluent wrong output and a missing one produces a stack trace.
//!
//! Adding a new compiler-owned tensor family therefore means adding its prefix HERE. Forgetting
//! to is loud (the loader demands it from the checkpoint); the old polarity made forgetting the
//! *checkpoint* side silent.

/// Namespaces `devgen` mints itself. Nothing under these is ever looked up in a checkpoint.
///
/// * `act.` — activations and scratch (`act.x`, `act.logits`, …)
/// * `in.`  — engine inputs and compiler-materialised tables (`in.ids`, `in.pos`, `in.cos`, …)
/// * `kv.`  — the KV / MLA-latent ring (`kv.{l}.ckv`, `kv.{l}.krot`, …)
/// * `moe.` — Gemma's fused-expert pointer tables (`moe.ewt.{l}`, `moe.est.{l}`), filled by the
///   host from the addresses of tensors that ARE weights
///
/// Kept as data, not as a chain of `||`, so a test can assert over the constant rather than
/// re-spelling the literals it is supposed to be checking.
pub const RUNTIME_PREFIXES: &[&str] = &["act.", "in.", "kv.", "moe."];

/// True for a name in one of the [`RUNTIME_PREFIXES`] namespaces.
pub fn is_runtime_tensor(name: &str) -> bool {
    RUNTIME_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// GLM/DeepSeek-style expert POINTER tables: named like weights, filled by the host.
///
/// These live under the model prefix (`model.layers.{l}.mlp.expert_weight_table`) because they are
/// declared next to the layer they belong to, but they hold DEVICE ADDRESSES the loader computes
/// after packing the experts — no checkpoint contains them. `bind_packed_experts` /
/// `bind_dense_ffn_tables` fill them once packing is done.
///
/// Suffix-matched, not prefix-matched, precisely because the model prefix in front of them is
/// whatever the checkpoint uses.
pub fn is_host_filled_table(name: &str) -> bool {
    name.ends_with("mlp.expert_weight_table")
        || name.ends_with("mlp.expert_scale_table")
        || name.ends_with("mlp.dense_weight_table")
        || name.ends_with("mlp.dense_scale_table")
}

/// True when this declared tensor's bytes must come from the checkpoint.
///
/// Note what is deliberately NOT here: any mention of `model.`. The weight namespace is the
/// checkpoint's to choose — `model.` (GLM, Llama, Qwen), `model.language_model.` (Gemma-4
/// multimodal), `language_model.model.` (Kimi-K3), `lm_head.weight` (untied heads) and the
/// `fp8/`-prefixed twins all land here without an entry, and so does the next one.
///
/// A tensor this returns `true` for and the checkpoint does not have is an error, not a zero fill.
pub fn is_checkpoint_weight(name: &str) -> bool {
    !is_runtime_tensor(name) && !is_host_filled_table(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two silent-corruption cases this module exists for. Written against real spellings
    /// taken from the checkpoints (`language_model.model.…` is verbatim from Kimi-K3's
    /// `model.safetensors.index.json`, 497 052 of its 497 220 entries).
    #[test]
    fn a_wrapper_prefix_and_an_untied_head_are_weights() {
        // Kimi-K3: ZERO tensors start with `model.`.
        assert!(is_checkpoint_weight(
            "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight"
        ));
        assert!(is_checkpoint_weight("language_model.lm_head.weight"));
        assert!(is_checkpoint_weight("vision_tower.encoder.blocks.0.wqkv.weight"));
        // Untied head, as `devgen::declare` spells it (crates/devgen/src/lib.rs).
        assert!(is_checkpoint_weight("lm_head.weight"));
        // And everything that already worked keeps working.
        assert!(is_checkpoint_weight("model.layers.0.mlp.down_proj.weight"));
        assert!(is_checkpoint_weight("model.language_model.layers.0.mlp.down_proj.weight"));
        assert!(is_checkpoint_weight("fp8/model.layers.0.mlp.down_proj.weight"));
        assert!(is_checkpoint_weight("fp8/model.layers.0.mlp.down_proj.weight_scale"));
    }

    /// Compiler-owned namespaces are never looked up in a checkpoint. Asserted over
    /// [`RUNTIME_PREFIXES`] itself so adding a prefix cannot leave this test asserting the old set.
    #[test]
    fn compiler_owned_namespaces_are_not_weights() {
        for p in RUNTIME_PREFIXES {
            let n = format!("{p}whatever.0");
            assert!(is_runtime_tensor(&n), "{n}");
            assert!(!is_checkpoint_weight(&n), "{n} must not be demanded of a checkpoint");
        }
        // The real spellings, so a prefix that stops matching them is caught too.
        for n in ["act.x", "act.logits", "in.ids", "in.pos", "in.cos", "kv.3.krot", "moe.ewt.7"] {
            assert!(!is_checkpoint_weight(n), "{n}");
        }
    }

    /// Host-filled pointer tables sit UNDER the model prefix, so exclusion-by-prefix alone would
    /// demand them from the checkpoint and fail every GLM bind. They are excluded by suffix, which
    /// is prefix-agnostic on purpose.
    #[test]
    fn expert_pointer_tables_are_not_checkpoint_weights() {
        for pfx in ["model.", "model.language_model.", "language_model.model."] {
            for suf in [
                "mlp.expert_weight_table",
                "mlp.expert_scale_table",
                "mlp.dense_weight_table",
                "mlp.dense_scale_table",
            ] {
                let n = format!("{pfx}layers.3.{suf}");
                assert!(is_host_filled_table(&n), "{n}");
                assert!(!is_checkpoint_weight(&n), "{n}");
            }
        }
        // A real weight whose name merely CONTAINS "table" is unaffected.
        assert!(is_checkpoint_weight("model.layers.3.mlp.experts.0.down_proj.weight"));
    }
}
