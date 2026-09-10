//! The KV-geometry contract for an emitted model.
//!
//! A CPU twin and its device packet are separate emits for different targets.
//! A prompt head prefilled against the twin reaches the device as a byte copy
//! of a row range out of each per-slot KV cache, which is sound only if both
//! lay a sequence slot out identically. This derives the digest that says so,
//! at emit, from the tensor declarations that were actually serialized.
//!
//! The runtime derives the same digest from the loaded blob
//! (`plowrt::exec::kvrow::kv_slot_tensors`), over `DevTensor { name, bytes }` —
//! the same pair this reads off `TensorDecl` — so the two cannot disagree about
//! what they are hashing.

use packet::devbuild::Model;
use plow_asset::kv_contract::{self, CacheTensor};

/// Sequence slots the packet's caches are cut into.
///
/// The `in.kvlen` width, which is the rule both engines already use to recover
/// the batch (`AmdEngine` and `CpuModel` both take `bytes / 4`). Deriving it the
/// same way here is what keeps the emit-time and load-time contracts over the
/// same partition; taking it from an emit flag instead would let the two drift
/// whenever a flag stopped reaching the declaration.
fn slots(m: &Model) -> u32 {
    m.tensors
        .iter()
        .find(|t| t.name == "in.kvlen")
        .map(|t| (t.bytes / 4).max(1) as u32)
        .unwrap_or(1)
}

/// The transferable caches and their per-slot extent, sorted by name.
pub fn cache_tensors(m: &Model) -> Vec<CacheTensor> {
    let batch = u64::from(slots(m));
    let mut out: Vec<CacheTensor> = m
        .tensors
        .iter()
        .filter(|t| kv_contract::is_cache_tensor(&t.name))
        .map(|t| CacheTensor {
            name: t.name.clone(),
            per_slot_bytes: t.bytes / batch.max(1),
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The digest a device packet and its CPU twin must agree on.
pub fn digest(m: &Model) -> String {
    kv_contract::digest(&cache_tensors(m))
}

static LAST: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Record the finished model's contract, from `apply_verify_gate` — the one
/// point every emit path reaches with the model in hand.
pub(crate) fn record(m: &Model) {
    if let Ok(mut slot) = LAST.lock() {
        *slot = Some(digest(m));
    }
}

/// The contract of the emit that most recently completed in this process.
///
/// `None` before the first emit. A caller comparing a packet against its twin
/// reads this after each [`crate::run_verified`], in order — which is what
/// makes a mismatch a compile error rather than a serving-time surprise.
pub fn last() -> Option<String> {
    LAST.lock().ok().and_then(|s| s.clone())
}
