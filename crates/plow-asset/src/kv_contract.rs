//! The KV-geometry contract binding a device packet to its CPU twin.
//!
//! A prompt head prefilled on the CPU reaches the device as a **byte copy of a
//! row range** out of each per-slot KV cache. That is sound only if both packets
//! lay a sequence slot out identically, and nothing else in the bundle says so:
//! the two are separate emits, the twin is compiled for a different target, and
//! a mismatch produces no fault and no missing weight — just a slot whose rows
//! `[0, n)` hold another geometry's bytes, which reads as fluent wrong output.
//! So the pair carries a digest and is refused when they disagree.
//!
//! **Per-slot bytes is the right invariant, and it has to be.** The caches are
//! head-major — a `(kv, head)` slot holds `max_seq_len × head_dim × elem` bytes
//! and a token's row sits inside it — so a twin compiled at a different
//! `--max-ctx` keeps the row stride and moves every head-slot boundary. Rows
//! `[0, n)` of head 1 then land at the wrong offset while head 0 still looks
//! right. Comparing the per-slot extent catches that; comparing a row stride
//! would not.

use sha2::{Digest, Sha256};

/// One append-only KV cache tensor and the bytes one sequence slot occupies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheTensor {
    pub name: String,
    pub per_slot_bytes: u64,
}

/// Is `name` an append-only per-slot KV cache — the set a row range may be
/// copied out of?
///
/// Two exclusions from the `kv.` namespace, both load-bearing and both already
/// stated at their other call sites in `exec::amd`:
///
/// * `kv.blkres` — K3's AttnRes snapshot ring, `[t][nb_cap][hidden]`, sized at
///   the widest prefill bucket rather than the slot count. It has no sequence
///   axis, so it has no per-slot stride to agree about.
/// * the KDA carried state (`kv.{l}.state`, `kv.{l}.conv_state.*`) — read
///   before it is written, so a row range does not reconstruct it. A head that
///   copied one would hand the device a recurrence that skipped its own update.
///
/// Substring rather than suffix, for the reason `is_carried_state` gives: the
/// three `conv_state.{q,k,v}` tensors are one idea, and a future
/// `kv.{l}.state.v` must not slip past a match written against today's
/// spellings.
pub fn is_cache_tensor(name: &str) -> bool {
    name.starts_with("kv.") && !name.contains("blkres") && !name.contains("state")
}

/// Digest over the per-slot KV geometry. Equal digests mean a row range copied
/// out of one packet's slot lands correctly in the other's.
pub fn digest(tensors: &[CacheTensor]) -> String {
    let mut h = Sha256::new();
    h.update(b"plow-kv-contract-v1");
    h.update((tensors.len() as u64).to_le_bytes());
    for t in tensors {
        h.update((t.name.len() as u64).to_le_bytes());
        h.update(t.name.as_bytes());
        h.update(t.per_slot_bytes.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ct(name: &str, per_slot_bytes: u64) -> CacheTensor {
        CacheTensor {
            name: name.to_string(),
            per_slot_bytes,
        }
    }

    #[test]
    fn the_cache_set_is_the_append_only_kv_tensors() {
        assert!(is_cache_tensor("kv.0.k"));
        assert!(is_cache_tensor("kv.7.ckv"));
        assert!(is_cache_tensor("kv.7.krot"));
        assert!(is_cache_tensor("kv.3.k_scale"));
        // Not per-sequence, or not reconstructible from a row range.
        assert!(!is_cache_tensor("kv.blkres"));
        assert!(!is_cache_tensor("kv.4.state"));
        assert!(!is_cache_tensor("kv.4.conv_state.q"));
        assert!(!is_cache_tensor("kv.4.state.v"));
        // Not the KV namespace at all.
        assert!(!is_cache_tensor("model.layers.0.mlp.gate"));
        assert!(!is_cache_tensor("in.kvlen"));
    }

    #[test]
    fn the_digest_is_over_a_sorted_set_not_a_declaration_order() {
        // Callers sort; this asserts the digest actually distinguishes the two
        // orders, so an unsorted caller is a bug the pair check will catch
        // rather than one it will silently tolerate.
        let sorted = vec![ct("kv.0.k", 32), ct("kv.1.k", 64)];
        let unsorted = vec![ct("kv.1.k", 64), ct("kv.0.k", 32)];
        assert_ne!(digest(&sorted), digest(&unsorted));
    }

    #[test]
    fn a_different_slot_extent_is_a_different_contract() {
        // The twin compiled at another `--max-ctx`: same tensors, same row
        // stride, every head-slot boundary moved.
        let dev = vec![ct("kv.0.k", 8192)];
        let twin = vec![ct("kv.0.k", 4096)];
        assert_ne!(digest(&dev), digest(&twin));
    }

    #[test]
    fn a_missing_cache_is_a_different_contract() {
        // The GLM-5.3 hazard: a twin that writes the latent but not the DSA
        // pooled indexer cache would leave the indexer scoring the head's rows
        // against uninitialised keys.
        let dev = vec![ct("kv.0.ckv", 4096), ct("kv.0.idx_pool", 512)];
        let twin = vec![ct("kv.0.ckv", 4096)];
        assert_ne!(digest(&dev), digest(&twin));
    }

    #[test]
    fn names_are_length_prefixed_so_concatenation_cannot_collide() {
        let a = vec![ct("kv.0.ka", 1), ct("kv.0.b", 1)];
        let b = vec![ct("kv.0.k", 1), ct("kv.0.ab", 1)];
        assert_ne!(digest(&a), digest(&b));
    }

}
