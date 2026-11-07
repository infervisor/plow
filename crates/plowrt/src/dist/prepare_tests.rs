use super::*;
use plow_asset::dist::*;
use std::collections::BTreeMap;

fn tmp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "plow-prep-{}-{}-{tag}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn bundle(tokenizer: TokenizerRef, files: Vec<FileRef>) -> Bundle {
    Bundle {
        schema: BUNDLE_SCHEMA.into(),
        namespace: "infervisor".into(),
        name: "kimi-k3".into(),
        label: "gfx942-mi325x-tp8".into(),
        generation: 1,
        variant_id: "v1".into(),
        plow_git: "468674e9".into(),
        network: "kimi-k3".into(),
        target: Target {
            vendor: "amd".into(),
            isa: "gfx942".into(),
            sku: "MI325X".into(),
            units: 304,
            mem_bytes: 256 << 30,
        },
        parallel: Parallel {
            mode: "tp".into(),
            n: 8,
        },
        max_ctx: 32768,
        files,
        tokenizer,
        checkpoint: BundleCheckpoint {
            source: "hf:moonshotai/Kimi-K3".into(),
            revision: "abc123".into(),
            shards: 2,
            layout: "native-mxfp4".into(),
            bytes: 1_590_000_000_000,
            files: vec![],
            views: vec![],
        },
        objset: Some(BundleObjset {
            objset_id: "o".into(),
            manifest: "v1/objsets/o.json".into(),
            sha256: "b".repeat(64),
            lowrung: vec![],
        }),
        pairing_hash: None,
        runtime_env: BTreeMap::new(),
        recipe: None,
        executor: None,
    }
}

fn packet_file() -> FileRef {
    FileRef {
        role: "packet".into(),
        name: "model.pkt".into(),
        sha256: "a".repeat(64),
        bytes: 10,
    }
}

fn from_checkpoint() -> TokenizerRef {
    TokenizerRef {
        source: Provenance::Checkpoint,
        file: "tokenizer.json".into(),
        sha256: None,
        generator: None,
        verified: false,
    }
}

/// A snapshot with `n` shards and whatever extra files are named.
fn snapshot(tag: &str, n: usize, extra: &[&str]) -> PathBuf {
    let p = tmp(tag);
    for i in 1..=n {
        std::fs::write(
            p.join(format!("model-{i:05}-of-{n:05}.safetensors")),
            b"shard",
        )
        .unwrap();
    }
    for f in extra {
        std::fs::write(p.join(f), b"{}").unwrap();
    }
    p
}

#[test]
fn the_farm_links_shards_sidecars_and_the_tokenizer() {
    let store = tmp("store");
    let snap = snapshot(
        "snap",
        2,
        &["tokenizer.json", "config.json", "tokenizer_config.json"],
    );
    let bdir = tmp("bundle");
    std::fs::write(bdir.join("model.pkt"), b"pkt").unwrap();

    let b = bundle(from_checkpoint(), vec![packet_file()]);
    let farm = build(&store, &b, &bdir, &snap).unwrap();

    assert_eq!(farm.shards, 2);
    assert!(farm.dir.ends_with("moonshotai--Kimi-K3@abc123"));
    assert!(farm.dir.join("model-00001-of-00002.safetensors").exists());
    assert!(farm.dir.join("tokenizer.json").exists());
    assert!(farm.dir.join("config.json").exists());
    // The runtime reads the tokenizer from the ASSETS dir, so it must be there too.
    assert!(bdir.join("tokenizer.json").exists());
    // And it resolves weights as `<assets>/checkpoint` unless --rt-checkpoint
    // overrides. Without this link `load` succeeds and `serve` then dies with a
    // bare NotFound on a path the operator never chose.
    let ckpt = bdir.join("checkpoint");
    assert!(
        ckpt.exists(),
        "the bundle must reach the farm as `checkpoint`"
    );
    assert_eq!(
        std::fs::canonicalize(&ckpt).unwrap(),
        std::fs::canonicalize(&farm.dir).unwrap()
    );
    assert!(ckpt.join("model-00001-of-00002.safetensors").exists());

    for p in [store, snap, bdir] {
        let _ = std::fs::remove_dir_all(p);
    }
}

// The case the tokenizer-provenance split exists for: K3 ships tiktoken.model
// and no tokenizer.json, so a bundle claiming the checkpoint owns it is wrong,
// and must say so at prepare time rather than at first token.
#[test]
fn a_checkpoint_tokenizer_that_the_snapshot_lacks_is_refused_with_the_fix() {
    let store = tmp("store2");
    let snap = snapshot("snap2", 2, &["tiktoken.model"]);
    let bdir = tmp("bundle2");

    let b = bundle(from_checkpoint(), vec![packet_file()]);
    let err = build(&store, &b, &bdir, &snap).unwrap_err().to_string();
    assert!(err.contains("tiktoken.model"), "{err}");
    assert!(err.contains("derived"), "{err}");

    for p in [store, snap, bdir] {
        let _ = std::fs::remove_dir_all(p);
    }
}

#[test]
fn a_derived_tokenizer_comes_from_the_bundle() {
    let store = tmp("store3");
    let snap = snapshot("snap3", 2, &["tiktoken.model"]);
    let bdir = tmp("bundle3");
    std::fs::write(bdir.join("tokenizer.json"), b"{\"real\":1}").unwrap();

    let tok = TokenizerRef {
        source: Provenance::Derived,
        file: "tokenizer.json".into(),
        sha256: Some("c".repeat(64)),
        generator: Some("scripts/kimi_k3_tokenizer.py".into()),
        verified: true,
    };
    let b = bundle(tok, vec![packet_file()]);
    let farm = build(&store, &b, &bdir, &snap).unwrap();
    assert!(farm.dir.join("tokenizer.json").exists());

    for p in [store, snap, bdir] {
        let _ = std::fs::remove_dir_all(p);
    }
}

// The sidecar's override only works because it sorts after the base shards —
// `'i' > '0'`. A name that does not is a silent no-op, so it is refused.
#[test]
fn a_derived_sidecar_must_sort_after_every_base_shard() {
    let store = tmp("store4");
    let snap = snapshot("snap4", 2, &["tokenizer.json"]);
    let bdir = tmp("bundle4");

    for (name, ok) in [
        ("model-idx-derived-00001.safetensors", true),
        ("aaa-derived.safetensors", false),
    ] {
        std::fs::write(bdir.join(name), b"derived").unwrap();
        let b = bundle(
            from_checkpoint(),
            vec![
                packet_file(),
                FileRef {
                    role: "derived_shard".into(),
                    name: name.into(),
                    sha256: "d".repeat(64),
                    bytes: 7,
                },
            ],
        );
        let got = build(&store, &b, &bdir, &snap);
        assert_eq!(got.is_ok(), ok, "{name}: {got:?}");
        if !ok {
            assert!(got.unwrap_err().to_string().contains("sort after"));
        }
    }

    for p in [store, snap, bdir] {
        let _ = std::fs::remove_dir_all(p);
    }
}

// A partial snapshot loads and then produces wrong output, which is exactly the
// failure mode worth refusing early.
#[test]
fn a_short_snapshot_is_refused_naming_both_counts() {
    let store = tmp("store5");
    let snap = snapshot("snap5", 1, &["tokenizer.json"]);
    let bdir = tmp("bundle5");
    let b = bundle(from_checkpoint(), vec![packet_file()]);
    let err = build(&store, &b, &bdir, &snap).unwrap_err().to_string();
    assert!(err.contains("found 1 shards"), "{err}");
    assert!(err.contains('2'), "{err}");

    for p in [store, snap, bdir] {
        let _ = std::fs::remove_dir_all(p);
    }
}

#[test]
fn a_directory_that_is_not_a_checkpoint_says_what_is_needed() {
    let store = tmp("store6");
    let empty = tmp("empty");
    let bdir = tmp("bundle6");
    let b = bundle(from_checkpoint(), vec![packet_file()]);

    let err = build(&store, &b, &bdir, &empty).unwrap_err().to_string();
    assert!(err.contains("no *.safetensors"), "{err}");

    let err = build(&store, &b, &bdir, &empty.join("nope"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("hf:moonshotai/Kimi-K3"), "{err}");
    assert!(err.contains("--fetch-weights"), "{err}");

    for p in [store, empty, bdir] {
        let _ = std::fs::remove_dir_all(p);
    }
}

/// A `dist` revision is a label a publisher may reuse with new content, so
/// a rebuilt farm holds the new snapshot's shards and nothing from before.
#[test]
fn a_rebuilt_farm_keeps_no_shard_from_the_previous_snapshot() {
    let store = tmp("store-rebuild");
    let bdir = tmp("bundle-rebuild");
    std::fs::write(bdir.join("model.pkt"), b"pkt").unwrap();
    let b = bundle(from_checkpoint(), vec![packet_file()]);

    let old = snapshot("snap-old", 3, &["tokenizer.json"]);
    let farm = build(&store, &b, &bdir, &old).unwrap();
    assert!(farm.dir.join("model-00003-of-00003.safetensors").exists());

    let new = snapshot("snap-new", 2, &["tokenizer.json"]);
    let farm = build(&store, &b, &bdir, &new).unwrap();
    let shards = std::fs::read_dir(&farm.dir)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
        .count();
    assert_eq!(shards, 2, "a stale shard would load beside the new ones");
    assert!(
        old.join("model-00003-of-00003.safetensors").exists(),
        "targets untouched"
    );
}

#[test]
fn a_refused_snapshot_leaves_the_farm_as_it_was() {
    let store = tmp("store-refused");
    let bdir = tmp("bundle-refused");
    std::fs::write(bdir.join("model.pkt"), b"pkt").unwrap();
    let b = bundle(from_checkpoint(), vec![packet_file()]);
    let good = snapshot("snap-good", 3, &["tokenizer.json"]);
    let farm = build(&store, &b, &bdir, &good).unwrap();

    let empty = tmp("snap-empty");
    assert!(build(&store, &b, &bdir, &empty).is_err());
    let shards = std::fs::read_dir(&farm.dir)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
        .count();
    assert_eq!(shards, 3, "a refused rebuild removed the working shards");
}

#[test]
fn a_qwen2_checkpoint_without_tokenizer_json_links_vocab_and_merges() {
    let store = tmp("store-qwen2");
    let snap = snapshot(
        "snap-qwen2",
        2,
        &["vocab.json", "merges.txt", "tokenizer_config.json"],
    );
    let bdir = tmp("bundle-qwen2");
    std::fs::write(bdir.join("model.pkt"), b"PLOWDEV").unwrap();

    let b = bundle(from_checkpoint(), vec![packet_file()]);
    let farm = build(&store, &b, &bdir, &snap).expect("a vocab/merges checkpoint prepares");
    for f in ["vocab.json", "merges.txt", "tokenizer_config.json"] {
        assert!(farm.dir.join(f).is_file(), "farm lacks {f}");
        assert!(bdir.join(f).is_file(), "bundle lacks {f}");
    }
    assert!(!bdir.join("tokenizer.json").exists());

    for p in [store, snap, bdir] {
        let _ = std::fs::remove_dir_all(p);
    }
}

/// Two variants on one checkpoint, prepared in sequence: the second finds the farm and passes it
/// as its snapshot. Rebuilt from itself it linked every shard to itself; both bundles must still
/// reach real shard files.
#[test]
fn a_second_variant_on_the_farm_keeps_every_link_resolving() {
    second_variant_on_the_farm(false);
}

/// The same with a hub-cache snapshot, whose shards are links into `blobs/`: the farm's source
/// is the snapshot dir, not where its links resolve.
#[test]
fn a_second_variant_on_a_hub_snapshot_farm_keeps_every_link_resolving() {
    second_variant_on_the_farm(true);
}

fn second_variant_on_the_farm(hub: bool) {
    let store = tmp("store");
    let snap = snapshot("snap", 2, &["tokenizer.json", "config.json"]);
    let blobs = tmp("blobs");
    if hub {
        for (i, e) in std::fs::read_dir(&snap).unwrap().flatten().enumerate() {
            let blob = blobs.join(format!("{i:064x}"));
            std::fs::rename(e.path(), &blob).unwrap();
            std::os::unix::fs::symlink(&blob, e.path()).unwrap();
        }
    }
    let (first, second) = (tmp("bundle-a"), tmp("bundle-b"));
    for d in [&first, &second] {
        std::fs::write(d.join("model.pkt"), b"pkt").unwrap();
    }
    let b = bundle(from_checkpoint(), vec![packet_file()]);
    let farm = build(&store, &b, &first, &snap).unwrap();
    let again = build(&store, &b, &second, &farm.dir).unwrap();
    assert_eq!(again.dir, farm.dir);
    for d in [&first, &second] {
        for f in [
            "checkpoint/model-00001-of-00002.safetensors",
            "checkpoint/model-00002-of-00002.safetensors",
            "checkpoint/config.json",
            "tokenizer.json",
        ] {
            let p = d.join(f);
            assert!(
                std::fs::metadata(&p).is_ok_and(|m| m.is_file()),
                "{} does not resolve to a file",
                p.display()
            );
        }
    }
    for e in std::fs::read_dir(&farm.dir).unwrap().flatten() {
        let target = std::fs::canonicalize(e.path()).unwrap();
        assert!(
            !target.starts_with(&farm.dir),
            "{} links into the farm",
            e.path().display()
        );
    }
    for p in [store, snap, blobs, first, second] {
        let _ = std::fs::remove_dir_all(p);
    }
}

/// The serve contract reads more of the snapshot than shards and tokenizer: a chat template,
/// a feature extractor's config. The farm is the whole repository.
#[test]
fn the_farm_holds_every_snapshot_file() {
    let store = tmp("store-all");
    let snap = snapshot(
        "snap-all",
        1,
        &[
            "tokenizer.json",
            "chat_template.json",
            "preprocessor_config.json",
            "merges.txt",
        ],
    );
    std::fs::write(snap.join(".gitattributes"), b"").unwrap();
    let bdir = tmp("bundle-all");
    let mut b = bundle(from_checkpoint(), vec![packet_file()]);
    b.checkpoint.shards = 1;
    let farm = build(&store, &b, &bdir, &snap).unwrap();
    for f in [
        "chat_template.json",
        "preprocessor_config.json",
        "merges.txt",
    ] {
        assert!(farm.dir.join(f).is_file(), "{f}");
    }
    assert!(!farm.dir.join(".gitattributes").exists());
}

/// A bundle whose packets bound prep-built checkpoints reaches each as `<dir>/checkpoint`,
/// rebuilt byte for byte, and never writes into the immutable view.
#[cfg(feature = "dist")]
#[test]
fn a_bundle_with_views_reaches_each_rebuilt_checkpoint() {
    use super::super::view;
    let write_st = view::tests::write_st;
    let snap = tmp("snap-view");
    write_st(
        &snap.join("model.safetensors"),
        &[("hf.a", b"AAAAAAAA"), ("hf.b", b"BBBB")],
    );
    std::fs::write(snap.join("tokenizer.json"), b"{\"tok\":1}").unwrap();
    let lm = tmp("prep-lm");
    write_st(
        &lm.join("model.safetensors"),
        &[("a", b"AAAAAAAA"), ("d", b"derived")],
    );
    std::fs::write(lm.join("tokenizer.json"), b"{\"tok\":1}").unwrap();
    let enc = tmp("prep-enc");
    write_st(&enc.join("model.safetensors"), &[("b", b"BBBB")]);
    std::fs::write(enc.join("config.json"), b"{\"enc\":1}").unwrap();

    let views = [
        view::compute("checkpoint", &lm, &snap).unwrap(),
        view::compute("text_encoder/checkpoint", &enc, &snap).unwrap(),
    ];
    let store_root = tmp("store-view");
    let store = super::super::Store::open(&store_root).unwrap();
    let scratch = tmp("blobs-view");
    for (v, prep) in views.iter().zip([&lm, &enc]) {
        for (d, _) in view::blobs(v) {
            let out = scratch.join(&d);
            view::write_blob(v, prep, &d, &out).unwrap();
            store
                .put(
                    &super::super::Digest::parse(&d).unwrap(),
                    &std::fs::read(&out).unwrap(),
                )
                .unwrap();
        }
    }
    let mut b = bundle(from_checkpoint(), vec![packet_file()]);
    b.target.vendor = "nvidia".into();
    b.checkpoint.source = "hf:org/repo".into();
    b.checkpoint.views = views.to_vec();
    let bdir = tmp("bundle-view");
    build(&store_root, &b, &bdir, &snap).unwrap();
    for (dir, prep) in [("checkpoint", &lm), ("text_encoder/checkpoint", &enc)] {
        for f in ["model.safetensors", "tokenizer.json", "config.json"] {
            let want = prep.join(f);
            if want.exists() {
                assert_eq!(
                    std::fs::read(bdir.join(dir).join(f)).unwrap(),
                    std::fs::read(&want).unwrap(),
                    "{dir}/{f}"
                );
            }
        }
    }
    assert!(
        bdir.join("tokenizer.json").is_file(),
        "the runtime reads it from the bundle"
    );
    let names = |d: &Path| {
        let mut n: Vec<_> = std::fs::read_dir(d)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        n.sort();
        n
    };
    assert_eq!(
        names(&bdir.join("checkpoint")),
        ["model.safetensors", "tokenizer.json"],
        "the view holds exactly what prep wrote"
    );
}
