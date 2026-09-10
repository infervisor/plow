//! A whole registry, built in a temp directory, exercised end to end over the
//! `file://` transport. No network, no hardware.

use super::*;
use crate::dist::{reference, transport};
use plow_asset::dist::*;
use std::collections::BTreeMap;
use std::path::PathBuf;

const MI325X: u64 = 256 << 30;

struct Fixture {
    root: PathBuf,
    store_root: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
        let _ = std::fs::remove_dir_all(&self.store_root);
    }
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let uniq = format!(
            "{}-{}-{tag}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(format!("plow-reg-{uniq}"));
        let store_root = std::env::temp_dir().join(format!("plow-st-{uniq}"));
        std::fs::create_dir_all(root.join("v1/blobs/sha256")).unwrap();
        std::fs::create_dir_all(root.join("v1/infervisor/kimi-k3/manifests")).unwrap();
        std::fs::create_dir_all(root.join("v1/objsets")).unwrap();
        Fixture { root, store_root }
    }

    fn store(&self) -> Store {
        Store::open(&self.store_root).unwrap()
    }

    fn fetch(&self) -> Box<dyn Fetch> {
        transport(&format!("file://{}", self.root.display())).unwrap()
    }

    /// Publish a blob and return `(digest, len)`.
    fn blob(&self, bytes: &[u8]) -> (String, u64) {
        let d = Digest::of(bytes);
        std::fs::write(self.root.join(format!("v1/blobs/sha256/{d}")), bytes).unwrap();
        (d.as_str().to_string(), bytes.len() as u64)
    }

    fn write(&self, path: &str, bytes: &[u8]) -> String {
        let p = self.root.join(path);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, bytes).unwrap();
        Digest::of(bytes).as_str().to_string()
    }
}

fn target() -> Target {
    Target {
        vendor: "amd".into(),
        isa: "gfx942".into(),
        sku: "MI325X".into(),
        units: 304,
        mem_bytes: MI325X,
    }
}

fn live() -> LiveTarget {
    LiveTarget {
        vendor: "amd".into(),
        isa: "gfx942".into(),
        sku: Some("MI325X".into()),
        units: 304,
        mem_bytes: MI325X,
        gpus: 8,
        toolchain: None,
    }
}

fn objset(id: &str, obj_sha: &str, obj_bytes: u64, pairing: Option<&str>) -> ObjSet {
    ObjSet {
        schema: OBJSET_SCHEMA.into(),
        objset_id: id.into(),
        target: target(),
        toolchain: "rocm-7.14.0-nix".into(),
        plow_git: "468674e985adf62c32b2178fbde29f0c5325c02e".into(),
        script: "scripts/build_gfx942.sh".into(),
        env: BTreeMap::new(),
        defines: BTreeMap::new(),
        objects: vec![ObjSetObject {
            name: "interp_decode_fp8kv_k3.elf".into(),
            sha256: obj_sha.into(),
            bytes: obj_bytes,
            arms: vec!["plow_k3_arms_1".into()],
            packet_hash: pairing.map(str::to_string),
        }],
    }
}

fn bundle(label: &str, gen: u32, pkt_sha: &str, pkt_bytes: u64, objset_id: &str) -> Bundle {
    Bundle {
        schema: BUNDLE_SCHEMA.into(),
        namespace: "infervisor".into(),
        name: "kimi-k3".into(),
        label: label.into(),
        generation: gen,
        variant_id: format!("{label}-g{gen}"),
        plow_git: "468674e985adf62c32b2178fbde29f0c5325c02e".into(),
        network: "kimi-k3".into(),
        target: target(),
        parallel: Parallel {
            mode: "tp".into(),
            n: 8,
        },
        max_ctx: 32768,
        files: vec![FileRef {
            role: "packet".into(),
            name: "model.pkt".into(),
            sha256: pkt_sha.into(),
            bytes: pkt_bytes,
        }],
        tokenizer: TokenizerRef {
            source: Provenance::Checkpoint,
            file: "tokenizer.json".into(),
            sha256: None,
            generator: None,
            verified: false,
        },
        checkpoint: BundleCheckpoint {
            source: "hf:moonshotai/Kimi-K3".into(),
            revision: "abc123".into(),
            shards: 96,
            layout: "native-mxfp4".into(),
            bytes: 1_590_000_000_000,
        },
        objset: BundleObjset {
            objset_id: objset_id.into(),
            manifest: format!("v1/objsets/{objset_id}.json"),
            sha256: String::new(), // filled by the publisher below
            lowrung: vec![],
        },
        pairing_hash: Some("0x9fd0e880fb6fbf09".into()),
        runtime_env: BTreeMap::from([("PLOW_CTR_DBUF".into(), "1".into())]),
        recipe: None,
    }
}

fn variant(label: &str, gen: u32, manifest_sha: &str, objset_id: &str) -> Variant {
    Variant {
        variant_id: format!("{label}-g{gen}"),
        label: label.into(),
        generation: gen,
        status: Status::Validated,
        target: target(),
        parallel: Parallel {
            mode: "tp".into(),
            n: 8,
        },
        max_ctx: 32768,
        features: BTreeMap::new(),
        build: Build {
            plow_git: "468674e985adf62c32b2178fbde29f0c5325c02e".into(),
            plowc: "0.2.0".into(),
            objset_id: objset_id.into(),
            pairing_hash: Some("0x9fd0e880fb6fbf09".into()),
        },
        manifest: format!("manifests/{label}@g{gen}"),
        sha256: manifest_sha.into(),
        released: None,
        supersedes: None,
        delta: None,
        measured: Some(Measured {
            tok_s: 131.162,
            p50_tpot_ms: None,
            concurrency: None,
        }),
        refusal: None,
    }
}

/// Publish one generation; returns its `Variant` row.
fn publish(f: &Fixture, label: &str, gen: u32, pkt: &[u8], obj: &[u8]) -> Variant {
    let (pkt_sha, pkt_len) = f.blob(pkt);
    let (obj_sha, obj_len) = f.blob(obj);
    let objset_id = format!("obj-g{gen}");

    let set = objset(&objset_id, &obj_sha, obj_len, Some("0x9fd0e880fb6fbf09"));
    let set_sha = f.write(
        &format!("v1/objsets/{objset_id}.json"),
        &serde_json::to_vec(&set).unwrap(),
    );

    let mut b = bundle(label, gen, &pkt_sha, pkt_len, &objset_id);
    b.objset.sha256 = set_sha;
    let manifest_sha = f.write(
        &format!("v1/infervisor/kimi-k3/manifests/{label}@g{gen}"),
        &serde_json::to_vec(&b).unwrap(),
    );
    variant(label, gen, &manifest_sha, &objset_id)
}

fn write_index(f: &Fixture, variants: Vec<Variant>) {
    let idx = ModelIndex {
        schema: INDEX_SCHEMA.into(),
        namespace: "infervisor".into(),
        name: "kimi-k3".into(),
        hf: "moonshotai/Kimi-K3".into(),
        revision: "abc123".into(),
        network: "kimi-k3".into(),
        aliases: vec!["k3".into()],
        checkpoint: CheckpointRef {
            shards: 96,
            layout: "native-mxfp4".into(),
            bytes: 1_590_000_000_000,
        },
        variants,
    };
    f.write(
        "v1/infervisor/kimi-k3/index.json",
        &serde_json::to_vec(&idx).unwrap(),
    );
}

const LABEL: &str = "gfx942-mi325x-tp8-32k-fp8kv-mxfp4";

#[test]
fn a_full_pull_materializes_a_servable_directory() {
    let f = Fixture::new("full");
    let pkt = vec![7u8; 4096];
    let obj = b"objset g1".to_vec();
    let v = publish(&f, LABEL, 1, &pkt, &obj);
    write_index(&f, vec![v]);

    let store = f.store();
    let fetch = f.fetch();
    let r = reference::parse("kimi-k3").unwrap();
    let resolved = resolve(&*fetch, &r, &live(), Constraints::default()).unwrap();
    assert_eq!(resolved.variant.generation, 1);

    let plan = resolved.plan(&store).unwrap();
    assert_eq!(plan.fetched_blobs, 2, "nothing is local yet");
    assert_eq!(plan.reused_blobs, 0);

    let (dir, t) = pull(&store, &*fetch, &resolved).unwrap();
    assert_eq!(t.fetched_blobs, 2);
    assert_eq!(t.fetched_bytes, pkt.len() as u64 + obj.len() as u64);

    // The layout `serve --assets` expects: the packet at the top, objects under
    // `hsaco/` where the AMD loader looks.
    assert_eq!(std::fs::read(dir.join("model.pkt")).unwrap(), pkt);
    assert_eq!(
        std::fs::read(dir.join("hsaco/interp_decode_fp8kv_k3.elf")).unwrap(),
        obj
    );

    // A second pull moves nothing.
    let (_, t2) = pull(&store, &*fetch, &resolved).unwrap();
    assert_eq!(t2.fetched_blobs, 0);
    assert_eq!(t2.reused_blobs, 2);
}

/// The property the whole layout exists for: a generation that changes only the
/// objset transfers only the objset, and the number reported comes from digests
/// rather than from what kind of change it was.
#[test]
fn an_objset_only_upgrade_transfers_only_the_objset() {
    let f = Fixture::new("upgrade");
    let pkt = vec![7u8; 400_000]; // the big, unchanged artifact
    let obj_g1 = vec![1u8; 4_000];
    let obj_g2 = vec![2u8; 4_000]; // same packet, new objects

    let v1 = publish(&f, LABEL, 1, &pkt, &obj_g1);
    let v2 = publish(&f, LABEL, 2, &pkt, &obj_g2);
    write_index(&f, vec![v1, v2]);

    let store = f.store();
    let fetch = f.fetch();
    let r = reference::parse("kimi-k3").unwrap();

    // Take g1 first.
    let g1 = resolve(
        &*fetch,
        &reference::parse(&format!("kimi-k3:{LABEL}@g1")).unwrap(),
        &live(),
        Constraints::default(),
    )
    .unwrap();
    let (_, t1) = pull(&store, &*fetch, &g1).unwrap();
    assert_eq!(t1.fetched_bytes, 404_000);

    // Unpinned now resolves to the newest generation.
    let g2 = resolve(&*fetch, &r, &live(), Constraints::default()).unwrap();
    assert_eq!(g2.variant.generation, 2, "generation ranks first");

    let plan = g2.plan(&store).unwrap();
    assert_eq!(plan.fetched_bytes, 4_000, "only the objset is missing");
    assert_eq!(plan.reused_bytes, 400_000, "the packet is shared");

    let (dir, t2) = pull(&store, &*fetch, &g2).unwrap();
    assert_eq!(t2.fetched_bytes, 4_000);
    assert_eq!(
        std::fs::read(dir.join("hsaco/interp_decode_fp8kv_k3.elf")).unwrap(),
        obj_g2
    );
    // Rolling back needs no network: g1's blobs are still in the store.
    assert_eq!(g1.plan(&store).unwrap().fetched_blobs, 0);
}

#[test]
fn a_tampered_manifest_is_refused_against_the_index_digest() {
    let f = Fixture::new("tamper");
    let v = publish(&f, LABEL, 1, b"pkt", b"obj");
    write_index(&f, vec![v]);

    // Rewrite the manifest after the index recorded its digest.
    let p = f
        .root
        .join(format!("v1/infervisor/kimi-k3/manifests/{LABEL}@g1"));
    let mut b: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    b["max_ctx"] = serde_json::json!(4096);
    std::fs::write(&p, serde_json::to_vec(&b).unwrap()).unwrap();

    let err = resolve(
        &*f.fetch(),
        &reference::parse("kimi-k3").unwrap(),
        &live(),
        Constraints::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("hashes to"), "{err}");
}

#[test]
fn a_corrupted_blob_is_refused_and_the_bundle_is_not_written() {
    let f = Fixture::new("corrupt");
    let pkt = vec![9u8; 1024];
    let v = publish(&f, LABEL, 1, &pkt, b"obj");
    write_index(&f, vec![v]);

    // Corrupt the published packet, keeping its name.
    let d = Digest::of(&pkt);
    std::fs::write(
        f.root.join(format!("v1/blobs/sha256/{d}")),
        b"not the packet",
    )
    .unwrap();

    let store = f.store();
    let fetch = f.fetch();
    let resolved = resolve(
        &*fetch,
        &reference::parse("kimi-k3").unwrap(),
        &live(),
        Constraints::default(),
    )
    .unwrap();
    let err = pull(&store, &*fetch, &resolved).unwrap_err().to_string();
    assert!(err.contains("digest mismatch"), "{err}");
    assert!(!store.has(&d), "a bad blob is never stored");
}

/// A specialised object pairs only with the packet that produced it. Refusing at
/// resolve time means no object byte moves before the mismatch is reported.
#[test]
fn an_objset_stamped_for_another_packet_is_refused_before_any_blob_moves() {
    let f = Fixture::new("pairing");
    let (obj_sha, obj_len) = f.blob(b"obj");
    let (pkt_sha, pkt_len) = f.blob(b"pkt");

    let set = objset("obj-x", &obj_sha, obj_len, Some("0xdeadbeef"));
    let set_sha = f.write("v1/objsets/obj-x.json", &serde_json::to_vec(&set).unwrap());
    let mut b = bundle(LABEL, 1, &pkt_sha, pkt_len, "obj-x");
    b.objset.sha256 = set_sha;
    let manifest_sha = f.write(
        &format!("v1/infervisor/kimi-k3/manifests/{LABEL}@g1"),
        &serde_json::to_vec(&b).unwrap(),
    );
    write_index(&f, vec![variant(LABEL, 1, &manifest_sha, "obj-x")]);

    let err = resolve(
        &*f.fetch(),
        &reference::parse("kimi-k3").unwrap(),
        &live(),
        Constraints::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("stamped for packet"), "{err}");
}

#[test]
fn a_machine_with_no_matching_variant_gets_the_refusal_table() {
    let f = Fixture::new("nofit");
    let v = publish(&f, LABEL, 1, b"pkt", b"obj");
    write_index(&f, vec![v]);

    // Four GPUs cannot host a tp8 build.
    let mut small = live();
    small.gpus = 4;
    let err = resolve(
        &*f.fetch(),
        &reference::parse("kimi-k3").unwrap(),
        &small,
        Constraints::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains(LABEL), "{err}");
    assert!(err.contains("8 GPUs") || err.contains("needs 8"), "{err}");
}

#[test]
fn an_unpublished_model_names_the_url_it_tried() {
    let f = Fixture::new("404");
    let err = resolve(
        &*f.fetch(),
        &reference::parse("acme/absent").unwrap(),
        &live(),
        Constraints::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("v1/acme/absent/index.json"), "{err}");
}

#[test]
fn transfer_sizes_render_readably() {
    assert_eq!(human(512), "512 B");
    assert_eq!(human(41 << 20), "41.0 MiB");
    assert_eq!(human(400 << 20), "400.0 MiB");
}
