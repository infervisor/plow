use super::*;

fn target(isa: &str, sku: &str, units: u32, mem: u64) -> Target {
    Target {
        vendor: if isa.starts_with("gfx") {
            "amd"
        } else {
            "nvidia"
        }
        .into(),
        isa: isa.into(),
        sku: sku.into(),
        units,
        mem_bytes: mem,
    }
}

const MI300X: u64 = 192 << 30;
const MI325X: u64 = 256 << 30;

fn variant(label: &str, gen: u32, t: Target, mode: &str, n: u32) -> Variant {
    Variant {
        variant_id: format!("{label}-g{gen}"),
        label: label.into(),
        generation: gen,
        status: Status::Validated,
        target: t,
        parallel: Parallel {
            mode: mode.into(),
            n,
        },
        max_ctx: 32768,
        features: BTreeMap::new(),
        build: Build {
            plow_git: "468674e9".into(),
            plowc: "0.2.0".into(),
            objset_id: "obj0".into(),
            pairing_hash: None,
        },
        manifest: format!("manifests/{label}@g{gen}"),
        sha256: "a".repeat(64),
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

fn live(isa: &str, sku: &str, units: u32, mem: u64, gpus: u32) -> LiveTarget {
    LiveTarget {
        vendor: if isa.starts_with("gfx") {
            "amd"
        } else {
            "nvidia"
        }
        .into(),
        isa: isa.into(),
        sku: Some(sku.into()),
        units,
        mem_bytes: mem,
        gpus,
        toolchain: None,
    }
}

// MI300X and MI325X are the SAME die: gfx942, 304 CUs, same LDS. Only capacity
// separates them, which is why the HSA VRAM probe is load-bearing.
#[test]
fn memory_is_what_separates_mi300x_from_mi325x() {
    let vs = vec![
        variant(
            "gfx942-mi300x-tp8",
            1,
            target("gfx942", "MI300X", 304, MI300X),
            "tp",
            8,
        ),
        variant(
            "gfx942-mi325x-tp8",
            1,
            target("gfx942", "MI325X", 304, MI325X),
            "tp",
            8,
        ),
    ];
    let picked = select(
        &vs,
        &live("gfx942", "MI300X", 304, MI300X, 8),
        &Constraints::default(),
    );
    assert_eq!(picked.unwrap().target.sku, "MI300X");

    // The larger box can run both; the exact-SKU tie-break decides.
    let picked = select(
        &vs,
        &live("gfx942", "MI325X", 304, MI325X, 8),
        &Constraints::default(),
    );
    assert_eq!(picked.unwrap().target.sku, "MI325X");
}

// A variant records the datasheet capacity; a probe reports the usable pool.
// These are the numbers measured on a live MI300X — 16 MiB apart — and a strict
// `>=` between them rejects the part for being itself.
#[test]
fn a_probe_under_the_datasheet_capacity_still_matches_its_own_part() {
    const SPEC: u64 = 206_158_430_208; // 192 GiB, MI300X datasheet
    const PROBED: u64 = 206_141_652_992; // what ROCr reports as the usable pool
    assert!(PROBED < SPEC);

    let vs = vec![variant(
        "gfx942-mi300x-tp1",
        1,
        target("gfx942", "MI300X", 304, SPEC),
        "tp",
        1,
    )];
    let mut l = live("gfx942", "MI300X", 304, PROBED, 8);
    assert!(select(&vs, &l, &Constraints::default()).is_ok());

    // A genuinely smaller part is still refused: the classes are ~25% apart, far
    // outside the tolerance.
    l.sku = Some("MI210".into());
    l.mem_bytes = 64 << 30;
    assert!(select(&vs, &l, &Constraints::default()).is_err());
}

#[test]
fn refuses_a_narrower_part_naming_the_unit_counts() {
    let vs = vec![variant(
        "gfx942-mi325x-tp8",
        1,
        target("gfx942", "MI325X", 304, MI325X),
        "tp",
        8,
    )];
    let err = select(
        &vs,
        &live("gfx942", "MI308X", 256, MI300X, 8),
        &Constraints::default(),
    )
    .unwrap_err();
    assert!(err.contains("304"), "{err}");
    assert!(err.contains("256"), "{err}");
}

// `exec/amd.rs` permits an exact multiple under PLOW_OVERSUB and nothing else.
#[test]
fn oversub_permits_an_exact_multiple_only() {
    let vs = vec![variant(
        "gfx942-tp8",
        1,
        target("gfx942", "MI325X", 304, MI325X),
        "tp",
        8,
    )];
    let c = Constraints {
        oversub: true,
        ..Default::default()
    };
    // 304 = 2 x 152: an exact multiple, so oversubscription is legal.
    assert!(select(&vs, &live("gfx942", "X", 152, MI325X, 8), &c).is_ok());
    // 300 does not divide 304.
    assert!(select(&vs, &live("gfx942", "X", 300, MI325X, 8), &c).is_err());
    // And it is opt-in.
    assert!(select(
        &vs,
        &live("gfx942", "X", 152, MI325X, 8),
        &Constraints::default()
    )
    .is_err());
}

#[test]
fn gpu_count_picks_the_widest_that_fits() {
    let vs = vec![
        variant(
            "gfx942-mi300x-tp4",
            1,
            target("gfx942", "MI300X", 304, MI300X),
            "tp",
            4,
        ),
        variant(
            "gfx942-mi300x-tp8",
            1,
            target("gfx942", "MI300X", 304, MI300X),
            "tp",
            8,
        ),
    ];
    let four = live("gfx942", "MI300X", 304, MI300X, 4);
    assert_eq!(
        select(&vs, &four, &Constraints::default())
            .unwrap()
            .parallel
            .n,
        4
    );
    let eight = live("gfx942", "MI300X", 304, MI300X, 8);
    assert_eq!(
        select(&vs, &eight, &Constraints::default())
            .unwrap()
            .parallel
            .n,
        8
    );
    // An explicit --tp pins it.
    let c = Constraints {
        parallel_n: Some(4),
        ..Default::default()
    };
    assert_eq!(select(&vs, &eight, &c).unwrap().parallel.n, 4);
}

// Generation leads the ranking: a newer build of one label is the same model on
// the same hardware, only built better.
#[test]
fn newer_generation_wins_over_a_faster_recorded_older_one() {
    let t = target("gfx942", "MI325X", 304, MI325X);
    let mut g2 = variant("gfx942-mi325x-tp8", 2, t.clone(), "tp", 8);
    g2.measured = Some(Measured {
        tok_s: 999.0,
        p50_tpot_ms: None,
        concurrency: None,
    });
    let g3 = variant("gfx942-mi325x-tp8", 3, t, "tp", 8);
    let vs = vec![g2, g3];
    let picked = select(
        &vs,
        &live("gfx942", "MI325X", 304, MI325X, 8),
        &Constraints::default(),
    );
    assert_eq!(picked.unwrap().generation, 3);
}

#[test]
fn validated_outranks_emits_at_equal_generation() {
    let t = target("gfx942", "MI300X", 304, MI300X);
    let mut emits = variant("gfx942-mi300x-tp8", 1, t.clone(), "tp", 8);
    emits.status = Status::Emits;
    emits.measured = None;
    let vs = vec![emits, variant("gfx942-mi300x-tp4", 1, t, "tp", 4)];
    // tp8 is wider, but only tp4 is validated.
    let picked = select(
        &vs,
        &live("gfx942", "MI300X", 304, MI300X, 8),
        &Constraints::default(),
    );
    let picked = picked.unwrap();
    assert_eq!(picked.status, Status::Validated);
    assert_eq!(picked.parallel.n, 4);
}

// plowc parses --parallel pp and then refuses it. A published row must never be
// selected on a build that cannot run it.
#[test]
fn an_unimplemented_parallel_mode_is_never_selected() {
    let vs = vec![variant(
        "gfx942-mi325x-tp4pp2",
        1,
        target("gfx942", "MI325X", 304, MI325X),
        "pp",
        8,
    )];
    let err = select(
        &vs,
        &live("gfx942", "MI325X", 304, MI325X, 8),
        &Constraints::default(),
    )
    .unwrap_err();
    assert!(err.contains("pp"), "{err}");
}

#[test]
fn a_refused_variant_is_never_selected_but_still_explains_itself() {
    let mut v = variant(
        "gfx942-mi300x-tp8",
        1,
        target("gfx942", "MI300X", 304, MI300X),
        "tp",
        8,
    );
    v.status = Status::Refused;
    v.measured = None;
    v.refusal = Some("crates/nn-graph/src/models/config/deepseek_v4.rs:261".into());
    let vs = vec![v];
    assert!(select(
        &vs,
        &live("gfx942", "MI300X", 304, MI300X, 8),
        &Constraints::default()
    )
    .is_err());
    let table = assess(
        &vs,
        &live("gfx942", "MI300X", 304, MI300X, 8),
        &Constraints::default(),
    );
    assert_eq!(table[0].reject, Some(Reject::Status));
}

// A gfx942 packet is rejected by the CPU backend, so a mac must be told at
// selection time rather than at first token.
#[test]
fn a_cpu_box_refuses_a_gpu_variant() {
    let vs = vec![variant(
        "gfx942-mi325x-tp8",
        1,
        target("gfx942", "MI325X", 304, MI325X),
        "tp",
        8,
    )];
    let mac = LiveTarget {
        vendor: "cpu".into(),
        isa: "cpu".into(),
        sku: None,
        units: 12,
        mem_bytes: 64 << 30,
        gpus: 1,
        toolchain: None,
    };
    let err = select(&vs, &mac, &Constraints::default()).unwrap_err();
    assert!(err.contains("amd") && err.contains("cpu"), "{err}");
}

#[test]
fn a_tie_is_an_error_not_a_coin_flip() {
    let t = target("gfx942", "MI325X", 304, MI325X);
    let mut a = variant("label-a", 1, t.clone(), "tp", 8);
    let mut b = variant("label-b", 1, t, "tp", 8);
    a.measured = None;
    b.measured = None;
    a.status = Status::Emits;
    b.status = Status::Emits;
    let vs = vec![a, b];
    let err = select(
        &vs,
        &live("gfx942", "MI325X", 304, MI325X, 8),
        &Constraints::default(),
    )
    .unwrap_err();
    assert!(err.contains("ambiguous"), "{err}");
}

#[test]
fn explicit_label_and_generation_pin_the_choice() {
    let t = target("gfx942", "MI325X", 304, MI325X);
    let vs = vec![
        variant("gfx942-mi325x-tp8", 2, t.clone(), "tp", 8),
        variant("gfx942-mi325x-tp8", 3, t.clone(), "tp", 8),
        variant("gfx942-mi325x-tp4", 3, t, "tp", 4),
    ];
    let l = live("gfx942", "MI325X", 304, MI325X, 8);
    let c = Constraints {
        label: Some("gfx942-mi325x-tp8".into()),
        generation: Some(2),
        ..Default::default()
    };
    let picked = select(&vs, &l, &c).unwrap();
    assert_eq!(picked.generation, 2);
    assert_eq!(picked.parallel.n, 8);
    // Pinning still runs the compatibility rule.
    let c = Constraints {
        label: Some("gfx942-mi325x-tp8".into()),
        ..Default::default()
    };
    assert!(select(&vs, &live("gfx942", "MI325X", 304, MI325X, 4), &c).is_err());
}

#[test]
fn max_ctx_and_features_narrow() {
    let t = target("gfx942", "MI325X", 304, MI325X);
    let mut v = variant("gfx942-mi325x-tp8", 1, t, "tp", 8);
    v.max_ctx = 16384;
    v.features.insert("fp8_kv".into(), true);
    v.features.insert("mxfp4_weights".into(), false);
    let vs = vec![v];
    let l = live("gfx942", "MI325X", 304, MI325X, 8);
    assert!(select(
        &vs,
        &l,
        &Constraints {
            max_ctx: Some(32768),
            ..Default::default()
        }
    )
    .is_err());
    assert!(select(
        &vs,
        &l,
        &Constraints {
            max_ctx: Some(8192),
            features: vec!["fp8_kv".into()],
            ..Default::default()
        }
    )
    .is_ok());
    assert!(select(
        &vs,
        &l,
        &Constraints {
            features: vec!["mxfp4_weights".into()],
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn a_different_sku_at_the_same_isa_and_units_warns_rather_than_refuses() {
    let vs = vec![variant(
        "gfx942-mi325x-tp8",
        1,
        target("gfx942", "MI325X", 304, MI325X),
        "tp",
        8,
    )];
    // An MI300X cannot host it — 192 GiB < 256 GiB.
    assert!(select(
        &vs,
        &live("gfx942", "MI300X", 304, MI300X, 8),
        &Constraints::default()
    )
    .is_err());
    // A same-capacity part with a different name loads, with a warning.
    let l = live("gfx942", "MI325X-OAM", 304, MI325X, 8);
    let table = assess(&vs, &l, &Constraints::default());
    assert!(table[0].ok());
    assert!(
        table[0].warns.iter().any(|w| w.contains("SKU")),
        "{:?}",
        table[0].warns
    );
}

#[test]
fn the_refusal_table_names_every_candidate() {
    let vs = vec![
        variant(
            "gfx942-mi325x-tp8",
            1,
            target("gfx942", "MI325X", 304, MI325X),
            "tp",
            8,
        ),
        variant(
            "gfx950-mi355x-tp8",
            1,
            target("gfx950", "MI355X", 256, MI325X),
            "tp",
            8,
        ),
    ];
    let err = select(
        &vs,
        &live("gfx942", "MI300X", 304, MI300X, 2),
        &Constraints::default(),
    )
    .unwrap_err();
    assert!(err.contains("gfx942-mi325x-tp8"), "{err}");
    assert!(err.contains("gfx950-mi355x-tp8"), "{err}");
    assert!(err.contains("gfx950"), "{err}");
}

#[test]
fn an_empty_variant_list_says_so() {
    let err = select(
        &[],
        &live("gfx942", "MI300X", 304, MI300X, 8),
        &Constraints::default(),
    )
    .unwrap_err();
    assert!(err.contains("publishes no variants"), "{err}");
}

// --- schema validation -------------------------------------------------------

fn bundle() -> Bundle {
    Bundle {
        schema: BUNDLE_SCHEMA.into(),
        namespace: "infervisor".into(),
        name: "kimi-k3".into(),
        label: "gfx942-mi325x-tp8-32k-fp8kv-mxfp4".into(),
        generation: 3,
        variant_id: "v0".into(),
        plow_git: "468674e9".into(),
        network: "kimi-k3".into(),
        target: target("gfx942", "MI325X", 304, MI325X),
        parallel: Parallel {
            mode: "tp".into(),
            n: 8,
        },
        max_ctx: 32768,
        files: vec![FileRef {
            role: "packet".into(),
            name: "model.pkt".into(),
            sha256: "f".repeat(64),
            bytes: 380901800,
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
            revision: "abc".into(),
            shards: 96,
            layout: "native-mxfp4".into(),
            bytes: 1_590_000_000_000,
            files: vec![],
        },
        objset: Some(BundleObjset {
            objset_id: "obj0".into(),
            manifest: "v1/objsets/amd/gfx942/mi325x/obj0.json".into(),
            sha256: "b".repeat(64),
            lowrung: vec![
                LowRung {
                    max: 1,
                    objset_id: "obj1".into(),
                    sha256: "1".repeat(64),
                },
                LowRung {
                    max: 2,
                    objset_id: "obj2".into(),
                    sha256: "2".repeat(64),
                },
            ],
        }),
        pairing_hash: Some("0x9fd0e880fb6fbf09".into()),
        runtime_env: BTreeMap::from([("PLOW_CTR_DBUF".into(), "1".into())]),
        recipe: None,
    }
}

// K3 ships tiktoken.model and NO tokenizer.json, so plow reconstructs one — the
// single case where a tokenizer is a shipped artifact.
/// Apple's Metal shaders compile on the serving machine from the packet, so
/// an Apple bundle ships no objset; any other vendor without one is a
/// broken publication.
#[test]
fn a_runtime_compiled_target_ships_no_objset_and_nothing_else_may() {
    let mut b = bundle();
    b.objset = None;
    b.pairing_hash = None;
    let err = b.validate().unwrap_err();
    assert!(err.contains("no objset"), "{err}");

    b.target = Target {
        vendor: "apple".into(),
        isa: "metal3".into(),
        sku: "Apple M4 Pro".into(),
        units: 16,
        mem_bytes: 18 << 30,
    };
    b.validate().unwrap();
    assert!(runtime_compiled("apple") && !runtime_compiled("amd"));

    b.pairing_hash = Some("0x1".into());
    assert!(b.validate().is_err(), "a pairing hash with nothing to pair");

    let mut v = variant("metal3-apple-m4-pro-tp1-2k", 1, b.target.clone(), "tp", 1);
    v.build.objset_id = String::new();
    v.build.pairing_hash = None;
    v.validate().unwrap();
    v.target = target("gfx942", "MI325X", 304, MI325X);
    assert!(v.validate().is_err(), "an AMD variant names its objset");
}

#[test]
fn a_derived_tokenizer_must_be_pinned_and_shipped() {
    let mut b = bundle();
    b.tokenizer = TokenizerRef {
        source: Provenance::Derived,
        file: "tokenizer.json".into(),
        sha256: Some("c".repeat(64)),
        generator: Some("scripts/kimi_k3_tokenizer.py".into()),
        verified: true,
    };
    // Declared derived but not in `files` — the file would never arrive.
    assert!(b.validate().is_err());
    b.files.push(FileRef {
        role: "tokenizer".into(),
        name: "tokenizer.json".into(),
        sha256: "c".repeat(64),
        bytes: 20217442,
    });
    assert!(b.validate().is_ok());
    // Derived without a generator is unreproducible.
    b.tokenizer.generator = None;
    assert!(b.validate().is_err());
}

#[test]
fn a_checkpoint_tokenizer_must_not_be_shipped() {
    let mut b = bundle();
    assert!(b.validate().is_ok());
    b.files.push(FileRef {
        role: "tokenizer".into(),
        name: "tokenizer.json".into(),
        sha256: "c".repeat(64),
        bytes: 20217442,
    });
    let err = b.validate().unwrap_err();
    assert!(err.contains("must not be shipped"), "{err}");
}

/// A `dist` checkpoint carries every file it needs as a blob; an `hf:` one
/// is fetched whole from its repository and lists none.
#[test]
fn a_dist_checkpoint_lists_safe_files_matching_its_shard_count() {
    let file = |name: &str| CheckpointFile {
        name: name.into(),
        sha256: "d".repeat(64),
        bytes: 1,
    };
    let mut dist = bundle();
    dist.checkpoint.source = DIST_SOURCE.into();
    dist.checkpoint.revision = "2026-10-12".into();
    dist.checkpoint.shards = 2;
    dist.checkpoint.files = vec![
        file("model-00001-of-00002.safetensors"),
        file("model-00002-of-00002.safetensors"),
        file("config.json"),
        // Nested: not a shard `prepare` would glob.
        file("onnx/model.safetensors"),
    ];
    dist.validate().unwrap();

    let mut b = dist.clone();
    b.checkpoint.files.clear();
    let err = b.validate().unwrap_err();
    assert!(err.contains("lists the files"), "{err}");

    let mut b = dist.clone();
    b.checkpoint.shards = 3;
    let err = b.validate().unwrap_err();
    assert!(err.contains("2 top-level safetensors shards"), "{err}");

    for name in ["", "../x", "/etc/passwd", "a//b", "a/./b", "a\\b"] {
        let mut b = dist.clone();
        b.checkpoint.files[2].name = name.into();
        assert!(b.validate().is_err(), "accepted path {name:?}");
    }
    let mut b = dist.clone();
    b.checkpoint.files[2].sha256 = "D".repeat(64);
    assert!(b.validate().is_err(), "an uppercase digest");
    let mut b = dist.clone();
    b.checkpoint.files.push(file("config.json"));
    assert!(b.validate().is_err(), "a duplicate file");
    let mut b = dist.clone();
    b.checkpoint.files.push(file("config.json/extra"));
    let err = b.validate().unwrap_err();
    assert!(err.contains("also a file"), "{err}");
    for rev in ["", "..", "a/b"] {
        let mut b = dist.clone();
        b.checkpoint.revision = rev.into();
        assert!(b.validate().is_err(), "accepted revision {rev:?}");
    }

    for (ns, name) in [("a--b", "c"), ("a", "b--c"), ("a", "..")] {
        let mut b = dist.clone();
        (b.namespace, b.name) = (ns.into(), name.into());
        assert!(b.validate().is_err(), "accepted {ns}/{name}");
    }

    let mut hf = bundle();
    hf.checkpoint.files = vec![file("model.safetensors")];
    let err = hf.validate().unwrap_err();
    assert!(err.contains("lists no"), "{err}");

    let mut other = bundle();
    other.checkpoint.source = "s3://bucket/k3".into();
    let err = other.validate().unwrap_err();
    assert!(err.contains("hf:") && err.contains("`dist`"), "{err}");

    // An `hf:` manifest is unchanged on the wire.
    let json = serde_json::to_value(bundle()).unwrap();
    assert!(json["checkpoint"].get("files").is_none());
}

#[test]
fn bundle_rejects_structural_damage() {
    let base = bundle();
    assert!(base.validate().is_ok());

    let mut b = base.clone();
    b.schema = "plow.dist.bundle.v2".into();
    assert!(b.validate().is_err());

    let mut b = base.clone();
    b.files.clear();
    assert!(b.validate().is_err());

    let mut b = base.clone();
    b.files[0].role = "asset_manifest".into();
    assert!(
        b.validate().is_err(),
        "a bundle with no packet is not a bundle"
    );

    for name in [
        "",
        "../model.pkt",
        "/model.pkt",
        "dir/model.pkt",
        "bundle.json",
    ] {
        let mut b = base.clone();
        b.files[0].name = name.into();
        assert!(b.validate().is_err(), "accepted path {name}");
    }

    let mut b = base.clone();
    b.files[0].sha256 = "short".into();
    assert!(b.validate().is_err());

    let mut b = base.clone();
    b.checkpoint.source = "/local/path".into();
    assert!(
        b.validate().is_err(),
        "the checkpoint is always an hf: reference"
    );

    let mut b = base.clone();
    b.objset.as_mut().unwrap().lowrung[1].max = 1;
    assert!(b.validate().is_err(), "duplicate lowrung width");

    let mut b = base.clone();
    b.objset.as_mut().unwrap().lowrung[0].max = 0;
    assert!(b.validate().is_err());

    // A rung override with no manifest digest is fetched unverified, and its
    // objects are checked against digests that manifest itself names — so an
    // unpinned rung can serve arbitrary decode objects.
    let mut b = base;
    b.objset.as_mut().unwrap().lowrung[0].sha256 = String::new();
    let err = b.validate().unwrap_err();
    assert!(err.contains("unverified"), "{err}");
}

#[test]
fn unknown_fields_are_ignored_across_every_schema() {
    // A field added for a newer client must not make this build refuse the
    // whole document: a registry is read by every plowrt ever released.
    let mut raw = serde_json::to_value(bundle()).unwrap();
    raw["surprise"] = serde_json::json!(1);
    assert_eq!(serde_json::from_value::<Bundle>(raw).unwrap(), bundle());

    let mut raw = serde_json::to_value(objset()).unwrap();
    raw["objects"][0]["surprise"] = serde_json::json!(1);
    assert_eq!(serde_json::from_value::<ObjSet>(raw).unwrap(), objset());

    let mut raw = serde_json::to_value(index()).unwrap();
    raw["variants"][0]["surprise"] = serde_json::json!(1);
    raw["variants"][0]["target"]["surprise"] = serde_json::json!({"nested": true});
    assert_eq!(serde_json::from_value::<ModelIndex>(raw).unwrap(), index());
}

#[test]
fn every_schema_round_trips() {
    for (name, v) in [
        ("bundle", serde_json::to_value(bundle()).unwrap()),
        ("objset", serde_json::to_value(objset()).unwrap()),
        ("index", serde_json::to_value(index()).unwrap()),
    ] {
        let s = serde_json::to_string(&v).unwrap();
        match name {
            "bundle" => {
                assert_eq!(serde_json::from_str::<Bundle>(&s).unwrap(), bundle())
            }
            "objset" => {
                assert_eq!(serde_json::from_str::<ObjSet>(&s).unwrap(), objset())
            }
            _ => assert_eq!(serde_json::from_str::<ModelIndex>(&s).unwrap(), index()),
        }
    }
}

fn objset() -> ObjSet {
    ObjSet {
        schema: OBJSET_SCHEMA.into(),
        objset_id: "obj0".into(),
        target: target("gfx942", "MI325X", 304, MI325X),
        toolchain: "rocm-7.14.0-nix".into(),
        plow_git: "468674e9".into(),
        script: "scripts/build_gfx942.sh".into(),
        env: BTreeMap::from([("PLOW_DECODE_BATCH".into(), "32".into())]),
        defines: BTreeMap::from([(
            "interp_decode_fp8kv_k3".into(),
            "-DPLOW_ARCH_SUFFIX=gfx942 -DPLOW_K3=1".into(),
        )]),
        objects: vec![ObjSetObject {
            name: "interp_decode_fp8kv_k3.elf".into(),
            sha256: "4c0d2ef95a2bef839965c977d53c873b74c1a1c50c92e0655e132e1bcfa16393".into(),
            bytes: 1234567,
            arms: vec!["plow_k3_arms_1".into(), "plow_gemv_walk_1".into()],
            packet_hash: None,
        }],
    }
}

#[test]
fn objset_rejects_structural_damage() {
    let base = objset();
    assert!(base.validate().is_ok());

    let mut o = base.clone();
    o.objects.clear();
    assert!(o.validate().is_err());

    for name in ["", "../x.elf", "/x.elf", "d/x.elf"] {
        let mut o = base.clone();
        o.objects[0].name = name.into();
        assert!(o.validate().is_err(), "accepted {name}");
    }

    let mut o = base.clone();
    let dup = o.objects[0].clone();
    o.objects.push(dup);
    assert!(o.validate().is_err());

    let mut o = base;
    o.objects[0].bytes = 0;
    assert!(o.validate().is_err());
}

// A GENERAL object carries no stamp and pairs with anything; a specialised one
// pairs only with the packet that produced it. Catching this at publish turns a
// confusing load-time refusal into an obvious one.
#[test]
fn pairing_is_checked_at_publish_not_at_load() {
    let mut o = objset();
    assert!(o.pairs_with(Some("0x9fd0e880fb6fbf09")).is_ok());
    assert!(
        o.pairs_with(None).is_ok(),
        "an unstamped objset pairs with any packet"
    );

    o.objects[0].packet_hash = Some("0x9fd0e880fb6fbf09".into());
    assert!(o.pairs_with(Some("0x9fd0e880fb6fbf09")).is_ok());
    let err = o.pairs_with(Some("0xdeadbeef")).unwrap_err();
    assert!(err.contains("interp_decode_fp8kv_k3.elf"), "{err}");
    assert!(
        o.pairs_with(None).is_err(),
        "a stamped object needs a stamped packet"
    );
}

fn index() -> ModelIndex {
    ModelIndex {
        schema: INDEX_SCHEMA.into(),
        namespace: "infervisor".into(),
        name: "kimi-k3".into(),
        hf: "moonshotai/Kimi-K3".into(),
        revision: "abc".into(),
        network: "kimi-k3".into(),
        aliases: vec!["k3".into()],
        checkpoint: CheckpointRef {
            shards: 96,
            layout: "native-mxfp4".into(),
            bytes: 1_590_000_000_000,
        },
        variants: vec![variant(
            "gfx942-mi325x-tp8",
            3,
            target("gfx942", "MI325X", 304, MI325X),
            "tp",
            8,
        )],
    }
}

#[test]
fn index_rejects_structural_damage() {
    let base = index();
    assert!(base.validate().is_ok());

    let mut i = base.clone();
    i.hf = "Kimi-K3".into();
    assert!(i.validate().is_err(), "hf must be <org>/<repo>");

    let mut i = base.clone();
    let dup = i.variants[0].clone();
    i.variants.push(dup);
    assert!(i.validate().is_err(), "duplicate label@generation");

    // Same label, different generation, is the whole point.
    let mut i = base.clone();
    let mut next = i.variants[0].clone();
    next.generation = 4;
    next.variant_id = "v4".into();
    i.variants.push(next);
    assert!(i.validate().is_ok());

    let mut i = base;
    i.network = String::new();
    assert!(i.validate().is_err());
}

#[test]
fn variant_status_and_supersedes_invariants() {
    let base = variant("l", 1, target("gfx942", "MI325X", 304, MI325X), "tp", 8);
    assert!(base.validate().is_ok());

    // validated without a measurement is the claim this plan exists to prevent.
    let mut v = base.clone();
    v.measured = None;
    assert!(v.validate().is_err());

    let mut v = base.clone();
    v.status = Status::Refused;
    v.measured = None;
    assert!(v.validate().is_err(), "refused needs a reason");
    v.refusal = Some("devgen/src/qwen35.rs:1071".into());
    assert!(v.validate().is_ok());

    let mut v = base.clone();
    v.supersedes = Some("v0".into());
    assert!(v.validate().is_err(), "supersedes without delta");
    v.delta = Some(Delta::Objset);
    assert!(v.validate().is_ok());

    let mut v = base;
    v.parallel.n = 0;
    assert!(v.validate().is_err());
}

fn apple(sku: &str, cores: u32, mem: u64) -> Target {
    Target {
        vendor: "apple".into(),
        isa: "metal3".into(),
        sku: sku.into(),
        units: cores,
        mem_bytes: mem,
    }
}

fn apple_live(sku: &str, cores: u32, mem: u64) -> LiveTarget {
    LiveTarget {
        vendor: "apple".into(),
        isa: "metal3".into(),
        sku: Some(sku.into()),
        units: cores,
        mem_bytes: mem,
        gpus: 1,
        toolchain: None,
    }
}

#[test]
fn an_apple_build_runs_on_any_chip_of_its_family_with_a_warning_off_its_own() {
    let gib = 1u64 << 30;
    let pro = variant(
        "metal3-apple-m4-pro-tp1-2k",
        1,
        apple("Apple M4 Pro", 16, 24 * gib),
        "tp",
        1,
    );
    // On an M4 Max: selected, with a warning about the idle cores.
    let table = assess(
        std::slice::from_ref(&pro),
        &apple_live("Apple M4 Max", 32, 36 * gib),
        &Constraints::default(),
    );
    assert!(table[0].ok(), "{:?}", table[0].reject);
    assert!(
        table[0].warns.iter().any(|w| w.contains("serial dispatch")),
        "{:?}",
        table[0].warns
    );
    // On a base M4 (10 cores, 16 GiB): the core count is no bar either, and
    // the memory check still applies to a different SKU: 24 GiB of
    // recommended working set was built for, 16 GiB is here.
    let table = assess(
        std::slice::from_ref(&pro),
        &apple_live("Apple M4", 10, 16 * gib),
        &Constraints::default(),
    );
    assert!(
        matches!(table[0].reject, Some(Reject::Mem { .. })),
        "{:?}",
        table[0].reject
    );
    let table = assess(
        std::slice::from_ref(&pro),
        &apple_live("Apple M4", 10, 24 * gib),
        &Constraints::default(),
    );
    assert!(table[0].ok(), "{:?}", table[0].reject);
    // A newer family: an M5 Pro with the same core count runs it, and is
    // only ranked by its SKU and core gap.
    let table = assess(
        std::slice::from_ref(&pro),
        &apple_live("Apple M5 Pro", 20, 32 * gib),
        &Constraints::default(),
    );
    assert!(table[0].ok(), "{:?}", table[0].reject);
    // On its own chip: no warning at all.
    let table = assess(
        std::slice::from_ref(&pro),
        &apple_live("Apple M4 Pro", 16, 24 * gib),
        &Constraints::default(),
    );
    assert!(table[0].ok() && table[0].warns.is_empty(), "{:?}", table[0]);
    // A native build beats a newer generation of another chip's build:
    // generations count builds of one label, and the Pro build would run
    // serially on a Max.
    let mut pro_g2 = pro.clone();
    pro_g2.generation = 2;
    pro_g2.variant_id = "pro-g2".into();
    pro_g2.manifest = "manifests/metal3-apple-m4-pro-tp1-2k@g2".into();
    let mut max = variant(
        "metal3-apple-m4-max-tp1-2k",
        1,
        apple("Apple M4 Max", 32, 36 * gib),
        "tp",
        1,
    );
    max.status = Status::Emits;
    max.measured = None;
    let vs = vec![pro.clone(), pro_g2.clone(), max];
    let on_max = apple_live("Apple M4 Max", 32, 36 * gib);
    let picked = select(&vs, &on_max, &Constraints::default()).unwrap();
    assert_eq!(picked.label, "metal3-apple-m4-max-tp1-2k");
    // On the Pro, its own label's newest generation.
    let on_pro = apple_live("Apple M4 Pro", 16, 24 * gib);
    let picked = select(&vs, &on_pro, &Constraints::default()).unwrap();
    assert_eq!((picked.label.as_str(), picked.generation), ("metal3-apple-m4-pro-tp1-2k", 2));
    // An NVIDIA part with more SMs than the build is still refused: the grid is baked in.
    let h100 = variant(
        "sm90a-h100-tp1",
        1,
        target("sm_90a", "H100 SXM5", 132, 80 * gib),
        "tp",
        1,
    );
    let table = assess(
        std::slice::from_ref(&h100),
        &live("sm_90a", "RTX 6000 Pro Blackwell", 188, 96 * gib, 1),
        &Constraints::default(),
    );
    assert_eq!(
        table[0].reject,
        Some(Reject::Units {
            want: 132,
            live: 188
        })
    );
}

#[test]
fn on_a_larger_chip_the_nearest_build_wins() {
    let gib = 1u64 << 30;
    let base = variant(
        "metal3-apple-m4-tp1-2k",
        1,
        apple("Apple M4", 10, 16 * gib),
        "tp",
        1,
    );
    let pro = variant(
        "metal3-apple-m4-pro-tp1-2k",
        1,
        apple("Apple M4 Pro", 16, 24 * gib),
        "tp",
        1,
    );
    let max_ = variant(
        "metal3-apple-m4-max-tp1-2k",
        1,
        apple("Apple M4 Max", 32, 36 * gib),
        "tp",
        1,
    );
    let vs = vec![base, pro, max_];
    let pick = select(
        &vs,
        &apple_live("Apple M4 Max", 32, 36 * gib),
        &Constraints::default(),
    )
    .unwrap();
    assert_eq!(pick.label, "metal3-apple-m4-max-tp1-2k");
    // An M4 Max with no build of its own takes the Pro build, not the base one.
    let pick = select(
        &vs[..2],
        &apple_live("Apple M4 Max", 32, 36 * gib),
        &Constraints::default(),
    )
    .unwrap();
    assert_eq!(pick.label, "metal3-apple-m4-pro-tp1-2k");
}
