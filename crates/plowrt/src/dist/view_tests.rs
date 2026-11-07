use super::*;

fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "plow-view-{}-{}-{tag}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// A safetensors file; tensors are written in the order given.
pub(crate) fn write_st(path: &Path, tensors: &[(&str, &[u8])]) {
    let mut header = serde_json::Map::new();
    let mut at = 0usize;
    for (name, bytes) in tensors {
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": "U8", "shape": [bytes.len()], "data_offsets": [at, at + bytes.len()]}),
        );
        at += bytes.len();
    }
    let mut text = serde_json::to_vec(&header).unwrap();
    while text.len() % 8 != 0 {
        text.push(b' ');
    }
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend(&text);
    for (_, bytes) in tensors {
        out.extend(*bytes);
    }
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).unwrap();
    }
    std::fs::write(path, out).unwrap();
}

fn bundle() -> Bundle {
    let mut b: Bundle = serde_json::from_value(serde_json::json!({
        "schema": plow_asset::dist::BUNDLE_SCHEMA_VIEWS, "namespace": "infervisor", "name": "tts",
        "label": "l", "generation": 1, "variant_id": "v", "plow_git": "g", "network": "tts",
        "target": {"vendor": "nvidia", "isa": "sm_90a", "sku": "H100 SXM5", "units": 132,
                   "mem_bytes": 1},
        "parallel": {"mode": "tp", "n": 1}, "max_ctx": 2048,
        "files": [{"role": "packet", "name": "model.pkt", "sha256": "a".repeat(64), "bytes": 1}],
        "tokenizer": {"source": "checkpoint", "file": "tokenizer.json"},
        "checkpoint": {"source": "hf:org/repo", "revision": "c".repeat(40), "shards": 1,
                       "layout": "hf-safetensors", "bytes": 0}
    }))
    .unwrap();
    b.checkpoint.views.clear();
    b
}

/// A snapshot and a checkpoint prep built from it: a renamed tensor, a same-name tensor, a
/// derived tensor, a derived config, a file moved from a subdirectory, a derived subdirectory.
fn fixture() -> (PathBuf, PathBuf) {
    let root = tmp("fx");
    let snap = root.join("snap");
    write_st(
        &snap.join("model-00001-of-00001.safetensors"),
        &[("src.a", b"AAAAAAAA"), ("b", b"BBBB"), ("unused", b"zz")],
    );
    std::fs::write(snap.join("config.json"), b"{\"hf\":1}").unwrap();
    std::fs::create_dir_all(snap.join("backbone")).unwrap();
    std::fs::write(snap.join("backbone/tokenizer.json"), b"{\"tok\":1}").unwrap();
    let prep = root.join("prep");
    write_st(
        &prep.join("model.safetensors"),
        &[
            ("model.a", b"AAAAAAAA"),
            ("b", b"BBBB"),
            ("new", b"derived!"),
        ],
    );
    std::fs::write(prep.join("config.json"), b"{\"plow\":1}").unwrap();
    std::fs::write(prep.join("tokenizer.json"), b"{\"tok\":1}").unwrap();
    write_st(&prep.join("ecapa/model.safetensors"), &[("w", b"ecapa")]);
    (snap, prep)
}

/// The store a pull would leave: every view blob present.
fn store_with(view: &CheckpointView, prep: &Path) -> (super::super::Store, PathBuf) {
    let root = tmp("store");
    let store = super::super::Store::open(&root).unwrap();
    let scratch = tmp("blobs");
    for (d, _) in blobs(view) {
        let out = scratch.join(&d);
        write_blob(view, prep, &d, &out).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        store
            .put(&super::super::Digest::parse(&d).unwrap(), &bytes)
            .unwrap();
    }
    (store, root)
}

#[test]
fn a_prepared_checkpoint_ships_only_what_prep_derived_and_rebuilds_byte_for_byte() {
    let (snap, prep) = fixture();
    let view = compute("checkpoint", &prep, &snap).unwrap();
    let by_name = |n: &str| view.files.iter().find(|f| f.name == n).unwrap();
    assert_eq!(
        by_name("tokenizer.json").hf.as_deref(),
        Some("backbone/tokenizer.json")
    );
    assert!(by_name("config.json").hf.is_none() && by_name("config.json").assembled.is_none());
    let a = by_name("model.safetensors").assembled.as_ref().unwrap();
    let kinds: Vec<_> = a
        .parts
        .iter()
        .map(|p| (p.tensor.as_deref(), p.derived))
        .collect();
    assert_eq!(
        kinds,
        [(Some("src.a"), None), (Some("b"), None), (None, Some(8))],
        "renamed and same-name tensors come from the snapshot, the new one is derived"
    );
    // Shipped: the derived tensor bytes, the derived config, the derived subdirectory's file.
    // The derived subdirectory's safetensors ships its tensor bytes; its header rides the view.
    let shipped: u64 = blobs(&view).iter().map(|(_, n)| n).sum();
    assert_eq!(shipped, 8 + 10 + 5);

    let (store, _) = store_with(&view, &prep);
    let b = bundle();
    let dir = materialize(&store, &b, &view, &snap).unwrap();
    for f in &view.files {
        assert_eq!(
            std::fs::read(dir.join(&f.name)).unwrap(),
            std::fs::read(prep.join(&f.name)).unwrap(),
            "{}",
            f.name
        );
    }
    assert_eq!(
        materialize(&store, &b, &view, &snap).unwrap(),
        dir,
        "reused"
    );
}

#[test]
fn a_snapshot_that_is_not_the_pinned_one_is_refused_and_leaves_nothing() {
    let (snap, prep) = fixture();
    let view = compute("checkpoint", &prep, &snap).unwrap();
    let (store, _) = store_with(&view, &prep);
    write_st(
        &snap.join("model-00001-of-00001.safetensors"),
        &[("src.a", b"AAAAAAAX"), ("b", b"BBBB"), ("unused", b"zz")],
    );
    let b = bundle();
    let e = materialize(&store, &b, &view, &snap)
        .unwrap_err()
        .to_string();
    assert!(e.contains("model.safetensors"), "{e}");
    assert!(!view_path(store.root(), &b, &view).unwrap().exists());
}

#[test]
fn two_views_of_one_snapshot_never_share_a_directory() {
    let (snap, prep) = fixture();
    let one = compute("checkpoint", &prep, &snap).unwrap();
    std::fs::write(prep.join("config.json"), b"{\"plow\":2}").unwrap();
    let two = compute("checkpoint", &prep, &snap).unwrap();
    let b = bundle();
    let root = Path::new("/s");
    assert_ne!(
        view_path(root, &b, &one).unwrap(),
        view_path(root, &b, &two).unwrap()
    );
}

#[test]
fn only_repository_files_of_a_hub_snapshot_are_sources() {
    let root = tmp("hub");
    let blobs_dir = root.join("models--org--repo/blobs");
    let snap = root
        .join("models--org--repo/snapshots")
        .join("c".repeat(40));
    std::fs::create_dir_all(&blobs_dir).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(blobs_dir.join("x"), b"{\"tok\":1}").unwrap();
    std::os::unix::fs::symlink("../../blobs/x", snap.join("tokenizer.json")).unwrap();
    // Not in the repository: a nested cache a tool wrote into the snapshot.
    std::fs::write(snap.join("stray.json"), b"{\"plow\":1}").unwrap();
    write_st(&snap.join("model.safetensors"), &[("w", b"1234")]);
    std::fs::rename(snap.join("model.safetensors"), blobs_dir.join("m")).unwrap();
    std::os::unix::fs::symlink("../../blobs/m", snap.join("model.safetensors")).unwrap();
    let prep = root.join("prep");
    std::fs::create_dir_all(&prep).unwrap();
    std::fs::write(prep.join("tokenizer.json"), b"{\"tok\":1}").unwrap();
    std::fs::write(prep.join("config.json"), b"{\"plow\":1}").unwrap();
    let view = compute("checkpoint", &prep, &snap).unwrap();
    let f = |n: &str| view.files.iter().find(|f| f.name == n).unwrap().hf.clone();
    assert_eq!(f("tokenizer.json").as_deref(), Some("tokenizer.json"));
    assert_eq!(
        f("config.json"),
        None,
        "stray.json is not a repository file"
    );
}

/// A prepared safetensors file that is not header + tensors would ship whole, snapshot tensors
/// and all: refused instead.
#[test]
fn a_safetensors_file_that_does_not_tile_is_refused() {
    let (snap, prep) = fixture();
    let p = prep.join("model.safetensors");
    let mut bytes = std::fs::read(&p).unwrap();
    bytes.extend(b"trailing");
    std::fs::write(&p, bytes).unwrap();
    let e = compute("checkpoint", &prep, &snap).unwrap_err().to_string();
    assert!(e.contains("do not tile"), "{e}");
}

/// A prepared directory may link a subdirectory in; its files are part of the view.
#[test]
fn a_linked_subdirectory_of_the_prep_is_walked() {
    let (snap, prep) = fixture();
    let other = tmp("linked");
    write_st(&other.join("model.safetensors"), &[("w", b"speaker")]);
    std::os::unix::fs::symlink(&other, prep.join("speaker")).unwrap();
    let view = compute("checkpoint", &prep, &snap).unwrap();
    assert!(view
        .files
        .iter()
        .any(|f| f.name == "speaker/model.safetensors"));
}
