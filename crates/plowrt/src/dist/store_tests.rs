use super::*;

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "plow-store-{}-{}-{name}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&p);
    p
}

fn store(name: &str) -> (Store, PathBuf) {
    let root = tmp(name);
    (Store::open(&root).unwrap(), root)
}

#[test]
fn a_blob_is_stored_under_its_own_digest_and_read_back() {
    let (s, root) = store("roundtrip");
    let bytes = b"the packet".to_vec();
    let d = Digest::of(&bytes);
    assert!(!s.has(&d));
    assert!(s.put(&d, &bytes).unwrap(), "first put fetches");
    assert!(s.has(&d));
    assert_eq!(s.get(&d).unwrap(), bytes);
    // Re-putting is a no-op, which is what makes a re-pull free.
    assert!(!s.put(&d, &bytes).unwrap(), "second put is already present");
    let _ = std::fs::remove_dir_all(root);
}

// A corrupted download must not be stored at all: a partially-written blob
// under a name promising full content is the failure the store exists to stop.
#[test]
fn content_that_does_not_match_its_digest_is_refused_and_not_stored() {
    let (s, root) = store("corrupt");
    let want = Digest::of(b"good");
    let err = s.put(&want, b"evil").unwrap_err().to_string();
    assert!(err.contains("digest mismatch"), "{err}");
    assert!(!s.has(&want), "nothing was stored");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn on_disk_corruption_is_caught_on_read() {
    let (s, root) = store("bitrot");
    let bytes = b"objects".to_vec();
    let d = Digest::of(&bytes);
    s.put(&d, &bytes).unwrap();
    std::fs::write(s.blob_path(&d), b"tampered").unwrap();
    let err = s.get(&d).unwrap_err().to_string();
    assert!(err.contains("store is corrupt"), "{err}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_digest_must_be_64_lowercase_hex() {
    assert!(Digest::parse(&"a".repeat(64)).is_ok());
    for bad in ["", "abc", &"A".repeat(64), &"g".repeat(64), &"a".repeat(63)] {
        assert!(Digest::parse(bad).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn materialize_builds_the_directory_the_runtime_expects() {
    let (s, root) = store("materialize");
    let pkt = b"PLOWDEV\x0b....".to_vec();
    let build = b"{\"schema\":1}".to_vec();
    let (dp, db) = (Digest::of(&pkt), Digest::of(&build));
    s.put(&dp, &pkt).unwrap();
    s.put(&db, &build).unwrap();

    let dir = s
        .materialize(
            "v1",
            &[
                ("model.pkt".into(), dp.clone()),
                ("build.json".into(), db.clone()),
            ],
        )
        .unwrap();
    assert_eq!(std::fs::read(dir.join("model.pkt")).unwrap(), pkt);
    assert_eq!(std::fs::read(dir.join("build.json")).unwrap(), build);

    // Materializing twice is idempotent, so `load` after `load` is safe.
    let again = s
        .materialize("v1", &[("model.pkt".into(), dp.clone())])
        .unwrap();
    assert_eq!(again, dir);
    assert_eq!(std::fs::read(dir.join("model.pkt")).unwrap(), pkt);
    let _ = std::fs::remove_dir_all(root);
}

// The property the whole layout is for: a generation that changes only the
// objset shares its packet with its predecessor, so an upgrade moves objects
// and nothing else.
#[test]
fn two_variants_sharing_a_packet_store_it_once() {
    let (s, root) = store("share");
    let pkt = vec![7u8; 4096];
    let obj_g2 = b"objset generation 2".to_vec();
    let obj_g3 = b"objset generation 3".to_vec();
    let (dp, d2, d3) = (Digest::of(&pkt), Digest::of(&obj_g2), Digest::of(&obj_g3));

    assert!(s.put(&dp, &pkt).unwrap());
    assert!(s.put(&d2, &obj_g2).unwrap());
    s.materialize(
        "g2",
        &[("model.pkt".into(), dp.clone()), ("i.elf".into(), d2)],
    )
    .unwrap();

    // Upgrading to g3: the packet is already present, so only the object is new.
    assert!(!s.put(&dp, &pkt).unwrap(), "packet already stored");
    assert!(s.put(&d3, &obj_g3).unwrap(), "only the objset is fetched");
    s.materialize(
        "g3",
        &[("model.pkt".into(), dp.clone()), ("i.elf".into(), d3)],
    )
    .unwrap();

    // One stored copy of the packet backing two bundles.
    let meta = std::fs::metadata(s.blob_path(&dp)).unwrap();
    assert_eq!(meta.len(), 4096);
    let blobs: Vec<_> = std::fs::read_dir(s.root().join("blobs/sha256"))
        .unwrap()
        .flatten()
        .collect();
    assert_eq!(blobs.len(), 3, "packet + two objsets, not four files");
    let _ = std::fs::remove_dir_all(root);
}

// Manifest file names arrive over the network, so they are untrusted input.
// Objects legitimately live under `hsaco/`, so nesting is allowed — downward only.
#[test]
fn a_bundle_file_name_may_nest_but_never_escape() {
    let (s, root) = store("nesting");
    let obj = b"an elf".to_vec();
    let d = Digest::of(&obj);
    s.put(&d, &obj).unwrap();

    let dir = s
        .materialize("v", &[("hsaco/interp_decode.elf".into(), d.clone())])
        .unwrap();
    assert_eq!(
        std::fs::read(dir.join("hsaco/interp_decode.elf")).unwrap(),
        obj
    );

    for bad in ["../escape", "/etc/passwd", "a/../../b", ""] {
        assert!(
            s.materialize("v", &[(bad.into(), d.clone())]).is_err(),
            "accepted {bad:?}"
        );
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn materializing_a_missing_blob_names_the_fix() {
    let (s, root) = store("missing");
    let d = Digest::of(b"absent");
    let err = s
        .materialize("v", &[("model.pkt".into(), d)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("plowrt pull"), "{err}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn pins_are_recorded_listed_and_removed() {
    let (s, root) = store("pins");
    assert_eq!(s.pinned("dist.infervisor.ai/infervisor/kimi-k3"), None);
    s.pin("dist.infervisor.ai/infervisor/kimi-k3", "variant-a")
        .unwrap();
    s.pin("dist.infervisor.ai/acme/other", "variant-b").unwrap();
    assert_eq!(
        s.pinned("dist.infervisor.ai/infervisor/kimi-k3").as_deref(),
        Some("variant-a")
    );
    assert_eq!(
        s.pins(),
        vec![
            (
                "dist.infervisor.ai/acme/other".to_string(),
                "variant-b".to_string()
            ),
            (
                "dist.infervisor.ai/infervisor/kimi-k3".to_string(),
                "variant-a".to_string()
            ),
        ]
    );
    // Re-pinning moves it; that is what `upgrade` does.
    s.pin("dist.infervisor.ai/infervisor/kimi-k3", "variant-c")
        .unwrap();
    assert_eq!(
        s.pinned("dist.infervisor.ai/infervisor/kimi-k3").as_deref(),
        Some("variant-c")
    );
    s.unpin("dist.infervisor.ai/infervisor/kimi-k3").unwrap();
    assert_eq!(s.pinned("dist.infervisor.ai/infervisor/kimi-k3"), None);
    // Unpinning what is not pinned is not an error.
    assert!(s.unpin("dist.infervisor.ai/infervisor/kimi-k3").is_ok());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn gc_removes_exactly_what_no_bundle_references() {
    let (s, root) = store("gc");
    let keep = b"still referenced".to_vec();
    let drop = vec![0u8; 1024];
    let (dk, dd) = (Digest::of(&keep), Digest::of(&drop));
    s.put(&dk, &keep).unwrap();
    s.put(&dd, &drop).unwrap();
    s.materialize("live", &[("model.pkt".into(), dk.clone())])
        .unwrap();

    let (n, freed) = s.gc().unwrap();
    assert_eq!(n, 1, "only the unreferenced blob");
    assert_eq!(freed, 1024);
    assert!(s.has(&dk), "a referenced blob survives");
    assert!(!s.has(&dd));

    // A second gc has nothing to do.
    assert_eq!(s.gc().unwrap(), (0, 0));
    let _ = std::fs::remove_dir_all(root);
}

// Objects live under `bundles/<id>/hsaco/`, so reachability that only looks at
// the bundle's top level marks every object dead while a bundle still uses it.
#[test]
fn gc_sees_objects_nested_under_hsaco() {
    let (s, root) = store("gc-nested");
    let pkt = b"packet".to_vec();
    let obj = b"an object under hsaco".to_vec();
    let (dp, dobj) = (Digest::of(&pkt), Digest::of(&obj));
    s.put(&dp, &pkt).unwrap();
    s.put(&dobj, &obj).unwrap();
    s.materialize(
        "v",
        &[
            ("model.pkt".into(), dp.clone()),
            ("hsaco/interp_decode.elf".into(), dobj.clone()),
        ],
    )
    .unwrap();

    let (n, _) = s.gc().unwrap();
    assert_eq!(n, 0, "both blobs are referenced");
    assert!(s.has(&dp));
    assert!(s.has(&dobj), "a nested object must not be collected");
    let _ = std::fs::remove_dir_all(root);
}

// A bundle whose record is missing is opaque. Collecting anyway could delete a
// blob it still needs, so refusing is the only safe answer.
#[test]
fn gc_refuses_rather_than_guess_at_an_unrecorded_bundle() {
    let (s, root) = store("gc-opaque");
    let bytes = b"orphan".to_vec();
    let d = Digest::of(&bytes);
    s.put(&d, &bytes).unwrap();
    std::fs::create_dir_all(s.root().join("bundles/hand-made")).unwrap();

    let err = s.gc().unwrap_err().to_string();
    assert!(err.contains("nothing can be collected safely"), "{err}");
    assert!(s.has(&d), "nothing was collected");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn open_is_idempotent_and_creates_the_layout() {
    let root = tmp("layout");
    let s = Store::open(&root).unwrap();
    for sub in ["blobs/sha256", "manifests", "refs", "bundles"] {
        assert!(root.join(sub).is_dir(), "{sub} missing");
    }
    // Opening an existing store keeps its contents.
    let bytes = b"x".to_vec();
    let d = Digest::of(&bytes);
    s.put(&d, &bytes).unwrap();
    let s2 = Store::open(&root).unwrap();
    assert!(s2.has(&d));
    let _ = std::fs::remove_dir_all(root);
}
