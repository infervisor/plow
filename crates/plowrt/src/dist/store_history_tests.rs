//! A pin's generations: `upgrade` leaves a trail, `rollback` walks it back,
//! `prune` ends it.

use super::*;

const REF: &str = "pdist.infervisor.ai/infervisor/kimi-k3";

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "plow-history-{}-{}-{name}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&p);
    p
}

/// A materialized bundle of `name` as variant `id`, with its manifest.
fn bundle(s: &Store, id: &str, name: &str) {
    let dir = s.root().join("bundles").join(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(".plow-blobs"), "").unwrap();
    let b = serde_json::json!({
        "schema": "plow.dist.bundle.v1", "namespace": "infervisor", "name": name,
        "label": "l", "generation": 1, "variant_id": id, "plow_git": "abc", "network": name,
        "target": {"vendor": "amd", "isa": "gfx942", "sku": "MI325X", "units": 304, "mem_bytes": 1},
        "parallel": {"mode": "tp", "n": 8}, "max_ctx": 32768,
        "files": [{"name": "model.pkt", "role": "packet", "bytes": 1, "sha256": "0".repeat(64)}],
        "tokenizer": {"file": "tokenizer.json", "source": "checkpoint"},
        "checkpoint": {"source": "hf:o/m", "revision": "r", "shards": 1, "layout": "x", "bytes": 1}
    });
    std::fs::write(dir.join(BUNDLE_MANIFEST), serde_json::to_vec(&b).unwrap()).unwrap();
}

#[test]
fn a_pin_remembers_what_it_moved_away_from_and_rolls_back_to_it() {
    let root = tmp("rollback");
    let s = Store::open(&root).unwrap();
    for id in ["a", "b", "c", "d"] {
        bundle(&s, id, "kimi-k3");
    }
    s.pin(REF, "a").unwrap();
    assert!(s.history(REF).is_empty());
    s.pin(REF, "b").unwrap();
    assert_eq!(s.history(REF), ["a"]);
    // Re-pinning the current generation is not a move.
    s.pin(REF, "b").unwrap();
    assert_eq!(s.history(REF), ["a"]);
    s.pin(REF, "c").unwrap();
    s.pin(REF, "d").unwrap();
    assert_eq!(s.history(REF), ["c", "b"], "{HISTORY} deep");
    assert_eq!(
        s.roots().into_iter().collect::<Vec<_>>(),
        ["b", "c", "d"],
        "the current and remembered generations"
    );

    assert_eq!(s.rollback(REF).unwrap(), ("d".to_string(), "c".to_string()));
    assert_eq!(s.pinned(REF).as_deref(), Some("c"));
    assert_eq!(
        s.history(REF),
        ["d", "b"],
        "the generation left is the next rollback"
    );
    assert_eq!(s.rollback(REF).unwrap(), ("c".to_string(), "d".to_string()));
    assert_eq!(s.pinned(REF).as_deref(), Some("d"));
    assert_eq!(s.list()[0].previous, ["c", "b"]);

    // A generation `gc` already took cannot be rolled back to.
    std::fs::remove_dir_all(s.root().join("bundles/c")).unwrap();
    let e = s.rollback(REF).unwrap_err().to_string();
    assert!(e.contains("no longer in the store"), "{e}");
    assert_eq!(s.pinned(REF).as_deref(), Some("d"), "nothing moved");

    s.unpin(REF).unwrap();
    let e = s.rollback(REF).unwrap_err().to_string();
    assert!(e.contains("not pinned"), "{e}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn prune_takes_only_the_superseded_generations_of_pinned_models() {
    let root = tmp("prune");
    let s = Store::open(&root).unwrap();
    for id in ["g1", "g2", "g3"] {
        bundle(&s, id, "kimi-k3");
    }
    // Another model nobody pins: a hand-placed bundle, not a generation.
    bundle(&s, "elsewhere", "other");
    // A directory with no manifest at all is opaque and left alone.
    std::fs::create_dir_all(s.root().join("bundles/hand-made")).unwrap();
    std::fs::write(s.root().join("bundles/hand-made/.plow-blobs"), "").unwrap();

    s.pin(REF, "g1").unwrap();
    s.pin(REF, "g2").unwrap();
    s.pin(REF, "g3").unwrap();
    assert_eq!(s.history(REF), ["g2", "g1"]);
    assert_eq!(s.prune().unwrap(), 0, "everything is current or remembered");

    // One more upgrade pushes g1 out of memory; prune takes it, and only it.
    bundle(&s, "g4", "kimi-k3");
    s.pin(REF, "g4").unwrap();
    assert_eq!(s.prune().unwrap(), 1);
    assert!(!s.root().join("bundles/g1").exists());
    for kept in ["g2", "g3", "g4", "elsewhere", "hand-made"] {
        assert!(
            s.root().join("bundles").join(kept).is_dir(),
            "{kept} survives"
        );
    }
    let _ = std::fs::remove_dir_all(root);
}
