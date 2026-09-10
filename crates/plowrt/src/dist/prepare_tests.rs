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
        },
        objset: BundleObjset {
            objset_id: "o".into(),
            manifest: "v1/objsets/o.json".into(),
            sha256: "b".repeat(64),
            lowrung: vec![],
        },
        pairing_hash: None,
        runtime_env: BTreeMap::new(),
        recipe: None,
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
