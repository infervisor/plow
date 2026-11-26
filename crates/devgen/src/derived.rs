//! `cat/` tensors: checkpoint tensors a packet binds through a load-time derivation
//! (`derived_weights.json`): axis-0 concatenation, a column block, a tiled scalar.
//!
//! Every emitter that declares one records its spec here, by label; [`section`] writes the ones
//! a model binds and [`checkpoint_sources`] maps a declared tensor back to what it reads.
use plow_asset::derived_weights::DerivedTensor;

use crate::Model;

thread_local! {
    /// The `cat/` tensors this thread's emits declared, by label.
    static DERIVED: std::cell::RefCell<std::collections::BTreeMap<String, DerivedTensor>> =
        Default::default();
}

fn record(spec: DerivedTensor) -> String {
    let name = spec.name.clone();
    DERIVED.with(|d| {
        let prev = d.borrow_mut().insert(name.clone(), spec.clone());
        assert!(prev.is_none_or(|p| p == spec), "{name}: two derivations");
    });
    name
}

fn spec(name: String, sources: Vec<String>) -> DerivedTensor {
    DerivedTensor {
        name,
        sources,
        widen_f32: false,
        cols: None,
        repeat: None,
    }
}

/// Checkpoint tensors `sources` concatenated along axis 0 (widened to f32 when `widen_f32`),
/// labelled by the first source.
pub(crate) fn concat<S: AsRef<str>>(sources: &[S], widen_f32: bool) -> String {
    let sources: Vec<String> = sources.iter().map(|s| s.as_ref().to_string()).collect();
    let name = packet::names::concat_label(&sources[0]);
    record(DerivedTensor {
        widen_f32,
        ..spec(name, sources)
    })
}

/// Checkpoint tensors `sources` concatenated along axis 0, labelled by the first source and the
/// count (routed experts' tensors; their rank-0 scales become `[E]`).
pub(crate) fn cat(sources: Vec<String>) -> String {
    let name = packet::names::concat_label(&format!("{}[cat{}]", sources[0], sources.len()));
    record(spec(name, sources))
}

/// Columns `lo..hi` of the 2-D checkpoint tensor `source`.
pub(crate) fn cols(source: &str, lo: u64, hi: u64) -> String {
    let name = packet::names::concat_label(&format!("{source}[:,{lo}:{hi}]"));
    record(DerivedTensor {
        cols: Some([lo, hi]),
        ..spec(name, vec![source.into()])
    })
}

/// Checkpoint tensor `source` tiled `n` times (a per-tensor scale as a per-channel vector).
pub(crate) fn repeat(source: &str, n: u32) -> String {
    let name = packet::names::concat_label(&format!("{source}[x{n}]"));
    record(DerivedTensor {
        repeat: Some(u64::from(n)),
        ..spec(name, vec![source.into()])
    })
}

/// `derived_weights.json` for the `cat/` tensors `m` declares.
pub(crate) fn section(m: &Model) -> Option<packet::devbuild::SectionData> {
    use plow_asset::derived_weights::{DerivedWeights, SECTION, VERSION};
    let tensors: Vec<_> = DERIVED.with(|d| {
        let d = d.borrow();
        m.tensors
            .iter()
            .filter(|t| t.name.starts_with(packet::names::CONCAT_PREFIX))
            .map(|t| d.get(&t.name).cloned().expect("undeclared cat/ tensor"))
            .collect()
    });
    (!tensors.is_empty()).then(|| packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&DerivedWeights {
            version: VERSION,
            tensors,
        })
        .expect("derived weights json"),
    })
}

/// The checkpoint tensors a declared tensor reads: a `cat/` tensor's sources, an `f32/` view's
/// source, otherwise itself.
pub(crate) fn checkpoint_sources(name: &str) -> Vec<String> {
    if name.starts_with(packet::names::CONCAT_PREFIX) {
        return DERIVED.with(|d| d.borrow()[name].sources.clone());
    }
    vec![name.strip_prefix("f32/").unwrap_or(name).to_string()]
}
