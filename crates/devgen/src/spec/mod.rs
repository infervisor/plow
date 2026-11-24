//! Speculative decode scaffolding shared by the checkpoint-MTP drafter families: the verify
//! depth, the chain drafter's inputs / programs / `spec.json`, and the span-verify binder.
pub(crate) mod chain_drafter;
pub(crate) mod span_verify;

/// Deepest verify span the recurrent rollback holds (`PLOW_SPEC_KMAX`, default 3): every
/// recurrent buffer keeps `depth` snapshots per slot (Qwen3.8-27B GDN: 147 MiB per slot per
/// depth row), and the drafters chain at most this many steps.
pub(crate) fn verify_depth() -> u32 {
    let k = crate::emit_config::active().spec_kmax.unwrap_or(3);
    assert!(
        (1..=15).contains(&k),
        "PLOW_SPEC_KMAX={k}: 1..=15 (the verify span holds at most 16 rows)"
    );
    k
}
