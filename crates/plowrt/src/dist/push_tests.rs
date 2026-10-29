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
        self.try_push(key, &Trust::none(), opts).unwrap()
    }

    fn try_push(
        &self,
        key: Option<&SecretKey>,
        trust: &Trust,
        opts: &Options<'_>,
    ) -> crate::Result<Outcome> {
        let g = self.generated();
        let src = Sources {
            assets: &self.assets,
            objects: &self.assets,
            checkpoint: None,
        };
        push(&self.sink(), g.bundle, g.objset, &src, key, trust, opts)
    }

    fn read(&self, key: &str) -> Vec<u8> {
        std::fs::read(self.registry.join(key)).unwrap()
    }

    fn write(&self, key: &str, bytes: &[u8]) {
        let p = self.registry.join(key);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    fn remove(&self, key: &str) {
        std::fs::remove_file(self.registry.join(key)).unwrap();
    }

    fn sig_key(&self) -> String {
        format!(
            "v1/infervisor/veena/sigs/{}.sig",
            crate::dist::Digest::of(&self.read("v1/infervisor/veena/index.json")).as_str()
        )
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
        !f.registry.join("v1/infervisor/veena/sigs").exists(),
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
    assert!(
        f.has(&f.sig_key()),
        "the signature is keyed by the index digest"
    );

    f.resolve(&Trust(vec![key.public()])).unwrap();
    f.resolve(&Trust(vec![other.public(), key.public()]))
        .unwrap();
    let e = f
        .resolve(&Trust(vec![other.public()]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a trusted key"), "{e}");

    // Tampering with the index after signing: the bytes read have no
    // signature (it is keyed by their digest), so they are refused.
    let p = f.registry.join("v1/infervisor/veena/index.json");
    let signed = std::fs::read(&p).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&signed).unwrap();
    idx["aliases"] = serde_json::json!(["evil"]);
    std::fs::write(&p, serde_json::to_vec(&idx).unwrap()).unwrap();
    let e = f
        .resolve(&Trust(vec![key.public()]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("unsigned"), "{e}");
    // A client with no trusted key checks nothing, as before.
    f.resolve(&Trust::none()).unwrap();
    std::fs::write(&p, &signed).unwrap();

    // A signature file holding someone else's signature over these bytes,
    // and one carrying the trusted name over a wrong signature: refused.
    let sig = f.sig_key();
    f.write(&sig, other.sign(&signed).to_string().as_bytes());
    let e = f
        .resolve(&Trust(vec![key.public()]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a trusted key"), "{e}");
    let forged = format!(
        "infervisor-1:{}",
        other.sign(&signed).to_string().split_once(':').unwrap().1
    );
    f.write(&sig, forged.as_bytes());
    let e = f
        .resolve(&Trust(vec![key.public()]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("does not verify"), "{e}");

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
        &Trust::none(),
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
    let e = push(
        &f.sink(),
        b,
        None,
        &src,
        None,
        &Trust::none(),
        &Options::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("pass --hf"), "{e}");
    assert!(
        !f.has("v1/infervisor/veena/index.json"),
        "refused before any write"
    );
}

// --- what a push that died, or raced, leaves behind -------------------------

#[test]
fn a_manifest_left_by_a_push_that_died_before_its_index_is_adopted() {
    let f = Fixture::new("orphan");
    let g1 = f.push(None, &Options::default());
    let index_g1 = f.read("v1/infervisor/veena/index.json");
    // A second generation lands in full, then the index is put back to how
    // the first run left it: exactly the state of a push that wrote
    // manifests/<label>@g2 and died before the index write.
    std::fs::write(f.assets.join("interp_sm90a.cubin"), b"ELF-cubin-v2").unwrap();
    let g2 = f.push(None, &Options::default());
    assert_eq!(g2.generation, 2);
    f.write("v1/infervisor/veena/index.json", &index_g1);

    let again = f.push(None, &Options::default());
    assert_eq!(
        again.generation, 2,
        "the orphan is this build: adopted, not skipped"
    );
    assert_eq!(again.variant_id, g2.variant_id);
    assert!(!again.unchanged);
    assert_eq!(f.index().variants.len(), 2);
    assert_eq!(
        f.index().variants[1].supersedes.as_deref(),
        Some(g1.variant_id.as_str())
    );
    f.resolve(&Trust::none()).unwrap();
}

#[test]
fn a_generation_name_taken_by_another_build_is_skipped_never_overwritten() {
    let f = Fixture::new("occupied");
    f.push(None, &Options::default());
    // Another publisher reserved g2 for a different build and has not (yet)
    // written the index.
    f.write(
        "v1/infervisor/veena/manifests/sm90a-h100-tp1@g2",
        b"{\"theirs\": true}",
    );
    std::fs::write(f.assets.join("model.pkt"), b"PLOWDEV-v2").unwrap();
    let out = f.push(None, &Options::default());
    assert_eq!(out.generation, 3, "{out:?}");
    assert_eq!(
        f.read("v1/infervisor/veena/manifests/sm90a-h100-tp1@g2"),
        b"{\"theirs\": true}",
        "the other build's reservation is untouched"
    );
    let resolved = f.resolve(&Trust::none()).unwrap();
    assert_eq!(resolved.variant.generation, 3);
}

#[test]
fn a_rerun_finishes_a_push_that_died_after_the_index() {
    let f = Fixture::new("repair");
    let key = SecretKey::generate("k").unwrap();
    f.push(Some(&key), &Options::default());
    let sig = f.sig_key();
    assert!(f.has(&sig));
    assert!(f.has("catalog.json"));

    // Died after the index: no signature, no catalog row.
    f.remove(&sig);
    f.remove("catalog.json");
    let e = f
        .resolve(&Trust(vec![key.public()]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("unsigned"), "{e}");

    let out = f.push(Some(&key), &Options::default());
    assert!(out.unchanged, "the build is published: {out:?}");
    assert!(f.has(&sig), "the signature is written on the re-run");
    assert!(f.has("catalog.json"), "and so is the catalog row");
    f.resolve(&Trust(vec![key.public()])).unwrap();

    // A later description lands on the row without a new generation.
    f.push(
        Some(&key),
        &Options {
            description: Some("a TTS model"),
            ..Default::default()
        },
    );
    let catalog: serde_json::Value = serde_json::from_slice(&f.read("catalog.json")).unwrap();
    assert_eq!(catalog["models"][0]["description"], "a TTS model");
}

#[test]
fn two_signed_publishers_never_sign_each_others_index() {
    // The signature is keyed by the index digest, so each version of the
    // index has exactly its own publisher's signature: B's push of a second
    // label leaves an index signed by B, and A's earlier signature stays
    // with the earlier bytes, verifying nothing else.
    let f = Fixture::new("two-keys");
    let a = SecretKey::generate("a").unwrap();
    let b = SecretKey::generate("b").unwrap();
    let both = Trust(vec![a.public(), b.public()]);
    f.push(Some(&a), &Options::default());
    let sig_a = f.sig_key();
    std::fs::write(
        &f.recipe,
        RECIPE.replace(
            "label = \"sm90a-h100-tp1\"",
            "label = \"sm90a-h100-tp1-other\"",
        ),
    )
    .unwrap();
    f.try_push(Some(&b), &both, &Options::default()).unwrap();
    let sig_b = f.sig_key();
    assert_ne!(sig_a, sig_b);
    assert!(f.has(&sig_a) && f.has(&sig_b));
    let fetch = transport(&format!("file://{}", f.registry.display())).unwrap();
    let r = reference::parse("veena").unwrap();
    pull::index_trusted(&*fetch, &r, &Trust(vec![b.public()])).unwrap();
    let e = pull::index_trusted(&*fetch, &r, &Trust(vec![a.public()]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a trusted key"), "{e}");
}

#[test]
fn a_publisher_with_trusted_keys_refuses_to_carry_an_unverified_index() {
    let f = Fixture::new("carry");
    let key = SecretKey::generate("k").unwrap();
    let other = SecretKey::generate("other").unwrap();
    f.push(None, &Options::default());
    std::fs::write(f.assets.join("model.pkt"), b"PLOWDEV-v2").unwrap();
    // The rows already there would be re-signed: with trust configured they
    // are held to it, exactly as a client would hold them.
    let e = f
        .try_push(Some(&key), &Trust(vec![key.public()]), &Options::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("unsigned"), "{e}");
    assert_eq!(f.index().variants.len(), 1, "nothing was published");
    // Without trust configured the rows are carried (the CLI warns).
    f.try_push(Some(&key), &Trust::none(), &Options::default())
        .unwrap();
    // Signed by `key`; a publisher trusting only `other` refuses it.
    std::fs::write(f.assets.join("model.pkt"), b"PLOWDEV-v3").unwrap();
    let e = f
        .try_push(
            Some(&other),
            &Trust(vec![other.public()]),
            &Options::default(),
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a trusted key"), "{e}");
    f.try_push(
        Some(&other),
        &Trust(vec![key.public()]),
        &Options::default(),
    )
    .unwrap();
}

#[test]
fn an_objset_name_holding_other_bytes_is_refused_before_the_bundle() {
    let f = Fixture::new("objset-clash");
    f.push(None, &Options::default());
    let id = f.index().variants[0].build.objset_id.clone();
    let key = format!("v1/objsets/{id}.json");
    let mut bytes = f.read(&key);
    bytes.push(b'\n');
    f.write(&key, &bytes);
    std::fs::write(f.assets.join("model.pkt"), b"PLOWDEV-v2").unwrap();
    let e = f
        .try_push(None, &Trust::none(), &Options::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("different bytes"), "{e}");
    assert!(
        !f.has("v1/infervisor/veena/manifests/sm90a-h100-tp1@g2"),
        "refused before the manifest"
    );
}

#[test]
fn a_rung_override_needs_its_objset_published_first() {
    let f = Fixture::new("lowrung");
    let g = f.generated();
    let src = Sources {
        assets: &f.assets,
        objects: &f.assets,
        checkpoint: None,
    };
    let mut bundle = g.bundle;
    bundle
        .objset
        .as_mut()
        .unwrap()
        .lowrung
        .push(plow_asset::dist::LowRung {
            max: 512,
            objset_id: "0123456789ab".into(),
            sha256: "0".repeat(64),
        });
    let e = push(
        &f.sink(),
        bundle,
        g.objset,
        &src,
        None,
        &Trust::none(),
        &Options::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("is not published"), "{e}");
    assert!(!f.has("v1/infervisor/veena/index.json"));
}

#[test]
fn a_rung_override_must_name_the_digest_of_the_bytes_the_registry_serves() {
    let f = Fixture::new("lowrung-digest");
    let g = f.generated();
    let src = Sources {
        assets: &f.assets,
        objects: &f.assets,
        checkpoint: None,
    };
    let served = br#"{"objset_id":"0123456789ab","newer_field":1}"#;
    let path = f.registry.join("v1/objsets/0123456789ab.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, served).unwrap();
    let rung = |sha256: String| plow_asset::dist::LowRung {
        max: 512,
        objset_id: "0123456789ab".into(),
        sha256,
    };
    let with = |l| {
        let mut b = g.bundle.clone();
        b.objset.as_mut().unwrap().lowrung.push(l);
        b
    };
    let e = push(
        &f.sink(),
        with(rung("0".repeat(64))),
        g.objset.clone(),
        &src,
        None,
        &Trust::none(),
        &Options::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("the registry serves"), "{e}");
    assert!(!f.has("v1/infervisor/veena/index.json"));

    push(
        &f.sink(),
        with(rung(Digest::of(served).as_str().to_string())),
        g.objset,
        &src,
        None,
        &Trust::none(),
        &Options::default(),
    )
    .unwrap();
    assert!(f.has("v1/infervisor/veena/index.json"));
}

#[test]
fn a_recipe_that_names_a_codec_needs_the_codec_packet() {
    let f = Fixture::new("codec");
    std::fs::write(
        &f.recipe,
        RECIPE.replace(
            "shards = 1\n",
            "shards = 1\ncodec = \"hf:hubertsiuzdak/snac_24khz\"\ncodec_revision = \"d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec\"\n",
        ),
    )
    .unwrap();
    let e = generate(
        &f.recipe,
        &f.assets,
        None,
        Overrides::default(),
        &BundleArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("no codec.pkt"), "{e}");
    std::fs::write(f.assets.join("codec.pkt"), b"PLOWDEV-codec").unwrap();
    let g = f.generated();
    assert!(g.bundle.files.iter().any(|x| x.role == "codec_packet"));
}

// --- listing ----------------------------------------------------------------

fn catalog_of(f: &Fixture) -> serde_json::Value {
    serde_json::from_slice(&f.read("catalog.json")).unwrap()
}

#[test]
fn a_listed_row_omits_the_field_and_an_old_catalog_reads_as_listed() {
    let f = Fixture::new("listed-default");
    f.push(None, &Options::default());
    let row = &catalog_of(&f)["models"][0];
    assert!(
        row.get("listed").is_none(),
        "listed is the default and not written: {row}"
    );
    let c = super::catalog(&f.sink()).unwrap();
    assert!(c.models[0].listed);

    // A catalog written before the field existed.
    let old =
        br#"{"models":[{"namespace":"infervisor","name":"veena","hf":"maya-research/Veena"}]}"#;
    let parsed: super::Catalog = serde_json::from_slice(old).unwrap();
    assert!(parsed.models[0].listed);
}

#[test]
fn an_unlisted_push_stays_unlisted_until_relisted() {
    let f = Fixture::new("unlisted");
    let out = f.push(
        None,
        &Options {
            listed: Some(false),
            ..Default::default()
        },
    );
    assert_eq!(out.generation, 1);
    assert_eq!(catalog_of(&f)["models"][0]["listed"], false);
    // Published all the same: a client resolves it by name.
    f.resolve(&Trust::none()).unwrap();

    // A later push without the flag (a new generation here) keeps the row.
    std::fs::write(f.assets.join("interp_sm90a.cubin"), b"ELF-cubin-v2").unwrap();
    let g2 = f.push(None, &Options::default());
    assert_eq!(g2.generation, 2);
    assert_eq!(catalog_of(&f)["models"][0]["listed"], false);

    // And an explicit relist on push, or `catalog list`, lifts it.
    f.push(
        None,
        &Options {
            listed: Some(true),
            ..Default::default()
        },
    );
    assert!(catalog_of(&f)["models"][0].get("listed").is_none());
}

#[test]
fn catalog_list_and_unlist_write_only_the_row() {
    use super::{set_listed, Listing};
    let f = Fixture::new("catalog-set");
    let key = SecretKey::generate("k").unwrap();
    f.push(Some(&key), &Options::default());
    let index_before = f.read("v1/infervisor/veena/index.json");
    let sig = f.sig_key();

    let sink = f.sink();
    assert_eq!(
        set_listed(&sink, "infervisor", "veena", false).unwrap(),
        Listing::Changed
    );
    assert_eq!(catalog_of(&f)["models"][0]["listed"], false);
    assert_eq!(
        set_listed(&sink, "infervisor", "veena", false).unwrap(),
        Listing::Unchanged
    );
    assert_eq!(
        set_listed(&sink, "infervisor", "veena", true).unwrap(),
        Listing::Changed
    );
    assert!(catalog_of(&f)["models"][0].get("listed").is_none());

    // Nothing under v1/ moved: the signed index is byte-identical and its
    // signature still verifies.
    assert_eq!(f.read("v1/infervisor/veena/index.json"), index_before);
    assert!(f.has(&sig));
    f.resolve(&Trust(vec![key.public()])).unwrap();

    // A name that was never published is refused, not reserved.
    let e = set_listed(&sink, "infervisor", "nope", false)
        .unwrap_err()
        .to_string();
    assert!(e.contains("nothing published"), "{e}");
    assert_eq!(catalog_of(&f)["models"].as_array().unwrap().len(), 1);
}

#[test]
fn unlisting_survives_the_catalog_being_rewritten_by_a_push() {
    // A second model's push rewrites catalog.json; the first model's row,
    // unlisted, is carried as is.
    use super::set_listed;
    let f = Fixture::new("catalog-carry");
    f.push(None, &Options::default());
    set_listed(&f.sink(), "infervisor", "veena", false).unwrap();
    // Same build pushed again: the catalog row is "present and current"
    // only if `listed` is part of what is compared.
    let again = f.push(None, &Options::default());
    assert!(again.unchanged);
    assert_eq!(catalog_of(&f)["models"][0]["listed"], false);
}

/// An index or catalog row written by a newer publisher may carry fields this
/// one does not know. A client reads past them; a publisher must not rewrite
/// the document without them, so it refuses and names the field.
#[test]
fn a_document_with_a_field_this_publisher_does_not_know_is_not_rewritten() {
    let f = Fixture::new("newer-fields");
    f.push(None, &Options::default());
    let index_key = "v1/infervisor/veena/index.json";
    let mut index: serde_json::Value = serde_json::from_slice(&f.read(index_key)).unwrap();
    index["variants"][0]["placement_hint"] = serde_json::json!("numa0");
    std::fs::write(
        f.registry.join(index_key),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    std::fs::write(f.assets.join("interp_sm90a.cubin"), b"ELF-cubin-v2").unwrap();
    let src = Sources {
        assets: &f.assets,
        objects: &f.assets,
        checkpoint: None,
    };
    let g = f.generated();
    let e = push(
        &f.sink(),
        g.bundle,
        g.objset,
        &src,
        None,
        &Trust::none(),
        &Options::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("variants[0].placement_hint"), "{e}");

    // Values the writer itself omits as defaults are not "unknown".
    let was =
        serde_json::json!({"models": [{"name": "a", "listed": true, "aliases": [], "x": null}]});
    let now = serde_json::json!({"models": [{"name": "a"}]});
    assert_eq!(super::dropped(&was, &now, ""), None);
    let was = serde_json::json!({"models": [{"name": "a", "pinned": true}]});
    assert_eq!(
        super::dropped(&was, &now, "").as_deref(),
        Some("models[0].pinned")
    );
}
