//! `push` into a directory, then `pull` from it: the registry a push writes
//! is the one a client reads, with no service in between. No network.

use super::*;
use crate::dist::manifest::{generate, BundleArgs, Overrides};
use crate::dist::sign::{SecretKey, Trust};
use crate::dist::sink::FileSink;
use crate::dist::{pull, reference, transport, Store};
use plow_asset::dist::LiveTarget;
use std::path::PathBuf;

const H100: u64 = 85899345920;

const RECIPE: &str = r#"
schema = "plow.recipe.v1"
namespace = "infervisor"
name = "veena"
label = "sm90a-h100-tp1"
plow_git = "1972fa37ac7a6f632cd5ece0b32fe37ab76eec75"
network = "llama"

[target]
vendor = "nvidia"
isa = "sm_90a"
sku = "H100 SXM5"
units = 132
mem_bytes = 85899345920
num_gpus = 1
parallel = "tp"
toolchain = "cuda-13.4+gcc-13"

[checkpoint]
source = "hf:maya-research/Veena"
revision = "8b770f9e69e6b35ef320d4cd70a99a4ab6dd022f"
shards = 1

[cell]
max_ctx = 2048

[serve]
env = { PLOW_VMM_PREFIX = "1" }
"#;

struct Fixture {
    root: PathBuf,
    assets: PathBuf,
    recipe: PathBuf,
    registry: PathBuf,
    store_root: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "plow-push-{}-{}-{tag}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let assets = root.join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        for (file, body) in [
            ("model.pkt", "PLOWDEV".as_bytes()),
            ("plow_config.h", b"#define X 1"),
            ("interp_sm90a.cubin", b"ELF-cubin-v1"),
            ("weights.json", br#"{"network":"veena"}"#),
            (
                "build.json",
                br#"{"pairing":{"hash":"0x541009da5f9890c1"},
                    "features":{"fp8_kv":false,"mxfp4_weights":false,"w8a8":false}}"#,
            ),
        ] {
            std::fs::write(assets.join(file), body).unwrap();
        }
        let recipe = root.join("r.toml");
        std::fs::write(&recipe, RECIPE).unwrap();
        let registry = root.join("registry");
        std::fs::create_dir_all(&registry).unwrap();
        let store_root = root.join("store");
        Fixture {
            root,
            assets,
            recipe,
            registry,
            store_root,
        }
    }

    fn sink(&self) -> FileSink {
        FileSink::new(&self.registry)
    }

    fn generated(&self) -> crate::dist::manifest::Generated {
        generate(
            &self.recipe,
            &self.assets,
            None,
            Overrides::default(),
            &BundleArgs::default(),
        )
        .unwrap()
    }

    fn push(&self, key: Option<&SecretKey>, opts: &Options<'_>) -> Outcome {
        let g = self.generated();
        let src = Sources {
            assets: &self.assets,
            objects: &self.assets,
            checkpoint: None,
        };
        push(&self.sink(), g.bundle, g.objset, &src, key, opts).unwrap()
    }

    fn has(&self, key: &str) -> bool {
        self.registry.join(key).is_file()
    }

    fn index(&self) -> plow_asset::dist::ModelIndex {
        serde_json::from_slice(
            &std::fs::read(self.registry.join("v1/infervisor/veena/index.json")).unwrap(),
        )
        .unwrap()
    }

    fn resolve(&self, trust: &Trust) -> crate::Result<pull::Resolved> {
        let f = transport(&format!("file://{}", self.registry.display())).unwrap();
        let r = reference::parse("veena").unwrap();
        pull::resolve_trusted(&*f, &r, &live(), Default::default(), trust)
    }
}

fn live() -> LiveTarget {
    LiveTarget {
        vendor: "nvidia".into(),
        isa: "sm_90a".into(),
        sku: Some("H100 SXM5".into()),
        units: 132,
        mem_bytes: H100,
        gpus: 1,
        toolchain: None,
    }
}

#[test]
fn a_push_writes_the_registry_a_pull_reads() {
    let f = Fixture::new("roundtrip");
    let out = f.push(None, &Options::default());
    assert_eq!(out.generation, 1);
    assert!(!out.unchanged);
    assert_eq!(out.reference, "infervisor/veena:sm90a-h100-tp1@g1");
    // Four bundle files and one object, each a blob; the objset manifest is
    // not a blob.
    assert_eq!(out.uploaded_blobs, 5, "{out:?}");
    assert!(f.has("v1/infervisor/veena/index.json"));
    assert!(f.has("v1/infervisor/veena/manifests/sm90a-h100-tp1@g1"));
    assert!(f.has("catalog.json"));
    assert!(
        !f.has("v1/infervisor/veena/index.json.sig"),
        "no key, no signature"
    );
    let idx = f.index();
    assert_eq!(idx.hf, "maya-research/Veena");
    assert_eq!(idx.variants.len(), 1);
    assert_eq!(idx.variants[0].variant_id, out.variant_id);
    assert!(f.has(&format!(
        "v1/objsets/{}.json",
        idx.variants[0].build.objset_id
    )));
    let catalog: serde_json::Value =
        serde_json::from_slice(&std::fs::read(f.registry.join("catalog.json")).unwrap()).unwrap();
    assert_eq!(catalog["models"][0]["name"], "veena");

    // The client side, over the same directory.
    let resolved = f.resolve(&Trust::none()).unwrap();
    assert_eq!(resolved.variant.generation, 1);
    let store = Store::open(&f.store_root).unwrap();
    let fetch = transport(&format!("file://{}", f.registry.display())).unwrap();
    let (dir, t) = pull::pull(&store, &*fetch, &resolved).unwrap();
    assert_eq!(t.fetched_blobs, 5);
    assert_eq!(std::fs::read(dir.join("model.pkt")).unwrap(), b"PLOWDEV");
    assert_eq!(
        std::fs::read(dir.join("hsaco/interp_sm90a.cubin")).unwrap(),
        b"ELF-cubin-v1"
    );
}

#[test]
fn the_same_build_again_writes_nothing() {
    let f = Fixture::new("idempotent");
    let first = f.push(None, &Options::default());
    let again = f.push(None, &Options::default());
    assert!(again.unchanged, "{again:?}");
    assert_eq!(again.generation, 1);
    assert_eq!(again.variant_id, first.variant_id);
    assert_eq!(f.index().variants.len(), 1);
    // A new status for the same build is a new generation: a published one
    // is immutable.
    let validated = f.push(
        None,
        &Options {
            status: Some(plow_asset::dist::Status::Validated),
            tok_s: Some(131.2),
            ..Default::default()
        },
    );
    assert_eq!(validated.generation, 2);
    assert_eq!(validated.uploaded_blobs, 0, "every blob was already there");
    assert_eq!(
        f.index().variants[1].measured.as_ref().unwrap().tok_s,
        131.2
    );
}

#[test]
fn a_changed_object_moves_only_itself_and_records_the_delta() {
    let f = Fixture::new("delta");
    let g1 = f.push(None, &Options::default());
    std::fs::write(f.assets.join("interp_sm90a.cubin"), b"ELF-cubin-v2").unwrap();
    let g2 = f.push(None, &Options::default());
    assert_eq!(g2.generation, 2);
    assert_eq!(g2.uploaded_blobs, 1, "only the object changed: {g2:?}");
    assert_eq!(g2.supersedes.as_deref(), Some(g1.variant_id.as_str()));
    assert_eq!(g2.delta, Some(plow_asset::dist::Delta::Objset));
    std::fs::write(f.assets.join("model.pkt"), b"PLOWDEV-v2").unwrap();
    let g3 = f.push(None, &Options::default());
    assert_eq!(g3.generation, 3);
    assert_eq!(g3.delta, Some(plow_asset::dist::Delta::Packet));
    assert_eq!(g3.supersedes.as_deref(), Some(g2.variant_id.as_str()));
    // A client takes the newest generation and the manifests still verify.
    let resolved = f.resolve(&Trust::none()).unwrap();
    assert_eq!(resolved.variant.generation, 3);
    assert_eq!(
        resolved.variant.supersedes.as_deref(),
        Some(g2.variant_id.as_str())
    );
}

#[test]
fn a_dry_run_reports_the_transfer_and_writes_nothing() {
    let f = Fixture::new("dry");
    let out = f.push(
        None,
        &Options {
            dry_run: true,
            ..Default::default()
        },
    );
    assert_eq!(out.uploaded_blobs, 5);
    assert!(!f.has("v1/infervisor/veena/index.json"));
    assert!(
        std::fs::read_dir(&f.registry).unwrap().next().is_none(),
        "nothing written"
    );
}

#[test]
fn a_signed_index_is_accepted_only_from_a_trusted_key_and_only_untampered() {
    let f = Fixture::new("signed");
    let key = SecretKey::generate("infervisor-1").unwrap();
    let other = SecretKey::generate("someone-else").unwrap();
    f.push(Some(&key), &Options::default());
    assert!(f.has("v1/infervisor/veena/index.json.sig"));

    f.resolve(&Trust(vec![key.public()])).unwrap();
    f.resolve(&Trust(vec![other.public(), key.public()]))
        .unwrap();
    let e = f
        .resolve(&Trust(vec![other.public()]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a trusted key"), "{e}");

    // Tampering with the index after signing: refused.
    let p = f.registry.join("v1/infervisor/veena/index.json");
    let mut idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    idx["aliases"] = serde_json::json!(["evil"]);
    std::fs::write(&p, serde_json::to_vec(&idx).unwrap()).unwrap();
    let e = f
        .resolve(&Trust(vec![key.public()]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("does not verify"), "{e}");
    // A client with no trusted key checks nothing, as before.
    f.resolve(&Trust::none()).unwrap();

    // An unsigned registry, with trust configured: refused.
    let g = Fixture::new("unsigned");
    g.push(None, &Options::default());
    let e = g
        .resolve(&Trust(vec![key.public()]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("unsigned"), "{e}");
}

#[test]
fn validated_needs_a_measurement_and_a_dist_checkpoint_needs_hf() {
    let f = Fixture::new("args");
    let g = f.generated();
    let src = Sources {
        assets: &f.assets,
        objects: &f.assets,
        checkpoint: None,
    };
    let e = push(
        &f.sink(),
        g.bundle.clone(),
        None,
        &src,
        None,
        &Options {
            status: Some(plow_asset::dist::Status::Validated),
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("pass --tok-s"), "{e}");
    let mut b = g.bundle;
    b.checkpoint.source = plow_asset::dist::DIST_SOURCE.into();
    let e = push(&f.sink(), b, None, &src, None, &Options::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("pass --hf"), "{e}");
    assert!(
        !f.has("v1/infervisor/veena/index.json"),
        "refused before any write"
    );
}
