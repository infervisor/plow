//! The sabotages the Python self-tests caught, plus the two things only a Rust
//! port can be wrong about: the canonical bytes and the published objset id.
//!
//! `objset_id` and `bundle.objset.sha256` key IMMUTABLE published state. The
//! fixtures below are the real `infervisor/veena:sm90a-h100-tp1@g1` objset, so
//! a change to the hashing or the canonicalisation fails here rather than by
//! writing a second id for content that already has one.

use super::*;

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "plow-manifest-{}-{}-{name}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Panics if asked for symbols. Every `.cubin` test uses it: a cubin's identity
/// is its digest, and reading its `.symtab` would be wrong, not merely wasteful.
struct Refuse;
impl SymbolReader for Refuse {
    fn symbols(&self, path: &Path) -> Result<Vec<String>> {
        panic!("read symbols from {} on the cubin path", path.display());
    }
}

/// Stands in for `Readelf`, whose contract is a SORTED set of markers — the
/// objset records them verbatim, so a stub that returned insertion order would
/// be testing an order the real reader never produces.
struct Stub(Vec<&'static str>);
impl SymbolReader for Stub {
    fn symbols(&self, _: &Path) -> Result<Vec<String>> {
        let mut v: Vec<String> = self.0.iter().map(|s| s.to_string()).collect();
        v.sort();
        Ok(v)
    }
}

fn amd() -> Identity {
    Identity {
        namespace: "infervisor".into(),
        name: "kimi-k3".into(),
        label: None,
        plow_git: "a".repeat(40),
        target: Target {
            vendor: "amd".into(),
            isa: "gfx942".into(),
            sku: "MI325X".into(),
            units: 304,
            mem_bytes: 274877906944,
        },
        parallel: Parallel {
            mode: "tp".into(),
            n: 8,
        },
        max_ctx: 32768,
        toolchain: "rocm-7.14.0-nix".into(),
        script: "scripts/build_gfx942.sh".into(),
        objects_env: BTreeMap::from([("PLOW_DECODE_BATCH".to_string(), "32".to_string())]),
        runtime_env: BTreeMap::new(),
        checkpoint_source: "hf:moonshotai/Kimi-K3".into(),
        checkpoint_revision: "rev1".into(),
        checkpoint_layout: "native-mxfp4".into(),
        checkpoint_shards: 96,
        checkpoint_bytes: 1590000000000,
        required: Vec::new(),
        recipe: "recipes/infervisor/kimi-k3/gfx942-mi325x-tp8-32k-fp8kv-mxfp4.toml".into(),
    }
}

// --- canonical bytes ---------------------------------------------------------

/// `json.dumps(sort_keys=True, separators=(",", ":"))` to the byte, including
/// `ensure_ascii`. The digest is CPython's over exactly this value; the
/// assertions below name which rule broke when it moves.
#[test]
fn canonical_json_is_pythons() {
    let v: serde_json::Value = serde_json::json!({
        "z": 1,
        "a": {"n": [1, 2, {"k": "v"}], "m": "tab\tnl\nq\"b\\s"},
        "u": "caf\u{e9} \u{b5} \u{1f600} \u{7f}",
        "e": {},
        "f": [],
        "b": true,
        "c": null,
    });
    let got = String::from_utf8(canonical_json(&v).unwrap()).unwrap();
    assert_eq!(
        sha256_bytes(got.as_bytes()),
        "784554271bb841c798e9b8ec944e0c8c296520e1f628c79006162af6c18d51a5",
        "not what CPython writes: {got}"
    );

    // Composed from a backslash CHAR so the expectation is the escape TEXT and
    // not the character it denotes.
    let b = '\\';
    let esc = |hex: &str| format!("{b}u{hex}");
    assert!(got.starts_with("{\"a\":{"), "sorted keys: `a` before `z`");
    assert!(!got.contains(", "), "no incidental whitespace");
    assert!(
        got.contains(&format!("tab{b}tnl{b}nq{b}\"b{b}{b}s")),
        "{got}"
    );
    assert!(
        got.contains(&esc("00e9")),
        "a non-ASCII scalar is escaped: {got}"
    );
    assert!(got.contains(&esc("00b5")), "{got}");
    assert!(
        got.contains(&format!("{}{}", esc("d83d"), esc("de00"))),
        "above the BMP is a surrogate PAIR: {got}"
    );
    assert!(
        got.contains(&esc("007f")),
        "0x7f is ascii and Python escapes it anyway: {got}"
    );
}

/// Field order in the Rust struct must not reach the bytes: a manifest's digest
/// is taken over SORTED keys, and `serde_json` emits declaration order.
#[test]
fn canonical_json_sorts_struct_fields() {
    let t = Target {
        vendor: "nvidia".into(),
        isa: "sm_90a".into(),
        sku: "H100 SXM5".into(),
        units: 132,
        mem_bytes: 85899345920,
    };
    assert_eq!(
        String::from_utf8(canonical_json(&t).unwrap()).unwrap(),
        r#"{"isa":"sm_90a","mem_bytes":85899345920,"sku":"H100 SXM5","units":132,"vendor":"nvidia"}"#
    );
}

// --- the published objset ----------------------------------------------------

/// The ten cubins of `infervisor/veena:sm90a-h100-tp1@g1`, in the order the
/// published manifest lists them.
fn veena_objects() -> Vec<ObjSetObject> {
    [
        (
            "interp_sm90a.cubin",
            "39f833e5a99946e50e976a81bfcfc50e5f2badd46770f1042c997af9091d047d",
            1789584u64,
        ),
        (
            "interp_sm90a_gw.cubin",
            "ac3c580db9559990597d4fad6a048156b5dfe0431571d472871388a291f5b825",
            1959440,
        ),
        (
            "interp_sm90a_pf.cubin",
            "67e808152e70a130c6ff61f3438770374877243cc812a4c6d5cfafa9ab775872",
            677496,
        ),
        (
            "interp_sm90a_pfgemm.cubin",
            "b73633495362e8623aa9d64d831a901a1f196e4aa4ec6520c6e214465d845515",
            409336,
        ),
        (
            "interp_sm90a_pfgemm_w8a16_m1.cubin",
            "485682dc4fb4e5e0a0315a5aa40caedd132a1ff3610b73a2af713e62b47d5318",
            220648,
        ),
        (
            "interp_sm90a_pfpackedgemm.cubin",
            "7ac952d3b08125c1bb461a6001e88e28a2555e3afed6751bd29ce61a7fb54788",
            410936,
        ),
        (
            "interp_sm90a_pfpackedseg.cubin",
            "cdc4d377fddc0d0f2b606ecc462a03aa78ed2560b0a4396826d1dfc84d1accfd",
            755896,
        ),
        (
            "interp_sm90a_pfseg.cubin",
            "a2f3bff97b7e2421b13c1cdff3b36ea64fa3bd7102707622967a1cdbd5683f38",
            677624,
        ),
        (
            "interp_sm90a_speech.cubin",
            "19103e3c2037546fdc9fc858a34a6c557263d256f489c80d9b0bdc0191fbfd33",
            8877328,
        ),
        (
            "sample_sm120.cubin",
            "4b9296d3c607933834b545670a707e901700c2c86451467af889497ec635f8c0",
            183896,
        ),
    ]
    .into_iter()
    .map(|(name, sha256, bytes)| ObjSetObject {
        name: name.into(),
        sha256: sha256.into(),
        bytes,
        arms: Vec::new(),
        packet_hash: None,
    })
    .collect()
}

fn veena() -> Identity {
    Identity {
        namespace: "infervisor".into(),
        name: "veena".into(),
        label: Some("sm90a-h100-tp1".into()),
        // The commit the bytes were BUILT at, which is not the one the recipe
        // pins; the objset id hashes it, so the fixture uses the built one.
        plow_git: "5f64e15810a219bbd4b43d6037204125e8361aad".into(),
        target: Target {
            vendor: "nvidia".into(),
            isa: "sm_90a".into(),
            sku: "H100 SXM5".into(),
            units: 132,
            mem_bytes: 85899345920,
        },
        parallel: Parallel {
            mode: "tp".into(),
            n: 1,
        },
        max_ctx: 2048,
        toolchain: "cuda-13.4+gcc-13".into(),
        script: "recipes/infervisor/veena/sm90a-h100-tp1.toml".into(),
        objects_env: BTreeMap::new(),
        runtime_env: BTreeMap::new(),
        checkpoint_source: "hf:maya-research/Veena".into(),
        checkpoint_revision: "8b770f9e69e6b35ef320d4cd70a99a4ab6dd022f".into(),
        checkpoint_layout: "hf-safetensors".into(),
        checkpoint_shards: 1,
        checkpoint_bytes: 0,
        required: Vec::new(),
        recipe: "recipes/infervisor/veena/sm90a-h100-tp1.toml".into(),
    }
}

/// The published id and the published manifest digest, from the published
/// content. If either moves, `v1/objsets/482fe256400e.json` is orphaned and a
/// re-publish writes a second id for the same ten cubins.
#[test]
fn published_veena_objset_id_and_digest() {
    let dir = tmp("veena-id");
    let objset = build_objset(&veena(), &dir, veena_objects()).unwrap();
    assert_eq!(objset.objset_id, "482fe256400e");
    assert_eq!(
        sha256_bytes(&canonical_json(&objset).unwrap()),
        "fe1dad44df06fa70d3a9035197ca875bd8c513c4746f8fe4e6b05a511906aa8a"
    );
    // And it is readable by the schema the registry enforces.
    let round: ObjSet = serde_json::from_slice(&canonical_json(&objset).unwrap()).unwrap();
    assert_eq!(round, objset);
    objset.validate().unwrap();
}

// --- identity ----------------------------------------------------------------

#[test]
fn objset_id_follows_the_defines_and_the_content() {
    let dir = tmp("ids");
    std::fs::write(dir.join("a.elf"), b"ELF-a").unwrap();
    std::fs::write(dir.join("build_defines.json"), br#"{"a":"-DX=1"}"#).unwrap();
    let syms = Stub(vec!["plow_k3_arms_1", "plow_gemv_walk_1"]);
    let id = amd();
    let pack = |d: &Path| build_objset(&id, d, discover_objects(d, &syms).unwrap()).unwrap();

    let one = pack(&dir);
    assert_eq!(one.schema, OBJSET_SCHEMA);
    assert_eq!(one.objects[0].arms, ["plow_gemv_walk_1", "plow_k3_arms_1"]);
    assert!(
        one.objects[0].packet_hash.is_none(),
        "no stamp means general"
    );
    assert_eq!(one.env["PLOW_DECODE_BATCH"], "32");
    assert_eq!(
        pack(&dir).objset_id,
        one.objset_id,
        "identity must be stable"
    );

    // The id must move when any -D moves.
    std::fs::write(dir.join("build_defines.json"), br#"{"a":"-DX=2"}"#).unwrap();
    assert_ne!(
        pack(&dir).objset_id,
        one.objset_id,
        "a changed -D must change the id"
    );

    // Two directories with the SAME inputs but DIFFERENT objects must not share
    // an id: they would overwrite each other at `v1/objsets/<id>.json`. This is
    // the `PLOW_ROWS_ONLY` shape — a rung override compiled from the same
    // defines as the full set.
    std::fs::write(dir.join("build_defines.json"), br#"{"a":"-DX=1"}"#).unwrap();
    std::fs::write(dir.join("a.elf"), b"ELF-different").unwrap();
    assert_ne!(
        pack(&dir).objset_id,
        one.objset_id,
        "different objects must not collide on one id"
    );
    std::fs::write(dir.join("a.elf"), b"ELF-a").unwrap();
    assert_eq!(
        pack(&dir).objset_id,
        one.objset_id,
        "identity must be stable"
    );
}

/// Order is fixed: `.elf`, then `.co`, then `.cubin`. `content_digest` hashes
/// `objects` in order, so a directory that gains a cubin must not move the
/// objects already in the set. `a.cubin` sorts BEFORE `fmoe.co`, so this fails
/// if the three globs are ever merged into one sorted pass.
#[test]
fn object_order_is_part_of_the_identity() {
    let dir = tmp("order");
    std::fs::write(dir.join("a.elf"), b"ELF-a").unwrap();
    std::fs::write(dir.join("fmoe.co"), b"ELF-vendor").unwrap();
    let syms = Stub(vec!["plow_k3_arms_1"]);
    let id = amd();
    let pack = |d: &Path| build_objset(&id, d, discover_objects(d, &syms).unwrap()).unwrap();

    let with_co = pack(&dir);
    // A pinned vendor code object is part of the set, not debris: a set without
    // the 64-row fmoe object serves the 32x256 MoE tile.
    let vendor = &with_co.objects[1];
    assert_eq!(vendor.name, "fmoe.co");
    assert!(vendor.arms.is_empty() && vendor.packet_hash.is_none());

    std::fs::write(dir.join("a.cubin"), b"ELF-cubin").unwrap();
    let with_cubin = pack(&dir);
    assert_eq!(
        with_cubin
            .objects
            .iter()
            .map(|o| o.name.as_str())
            .collect::<Vec<_>>(),
        ["a.elf", "fmoe.co", "a.cubin"]
    );
    std::fs::remove_file(dir.join("a.cubin")).unwrap();
    assert_eq!(
        pack(&dir).objset_id,
        with_co.objset_id,
        "removing the cubin must restore the id"
    );
}

#[test]
fn a_cubin_never_reaches_the_symbol_reader() {
    let dir = tmp("cuda");
    // The packet and its manifests share the directory and are not objects.
    std::fs::write(dir.join("model.pkt"), b"PLOWDEV").unwrap();
    std::fs::write(dir.join("plow_config.h"), b"#define X 1").unwrap();
    std::fs::write(dir.join("interp_sm90a.cubin"), b"ELF-cubin-a").unwrap();
    std::fs::write(dir.join("interp_sm90a_gw.cubin"), b"ELF-cubin-b").unwrap();

    let objects = discover_objects(&dir, &Refuse).unwrap();
    assert_eq!(
        objects.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
        ["interp_sm90a.cubin", "interp_sm90a_gw.cubin"]
    );
    for o in &objects {
        assert!(o.arms.is_empty() && o.packet_hash.is_none(), "{o:?}");
    }
    // No `build_defines.json` on the CUDA path, which is why `content_digest`
    // exists: the defines alone would be `{}` for every such directory.
    let objset = build_objset(&veena(), &dir, objects).unwrap();
    assert!(objset.defines.is_empty());

    // A cubin is part of the set, not debris: dropping one must move the id.
    std::fs::remove_file(dir.join("interp_sm90a_gw.cubin")).unwrap();
    let fewer = build_objset(&veena(), &dir, discover_objects(&dir, &Refuse).unwrap()).unwrap();
    assert_ne!(
        fewer.objset_id, objset.objset_id,
        "dropping a cubin must change the id"
    );
}

#[test]
fn an_empty_directory_is_an_error() {
    let dir = tmp("empty");
    std::fs::write(dir.join("model.pkt"), b"PLOWDEV").unwrap();
    let e = discover_objects(&dir, &Refuse).unwrap_err().to_string();
    assert!(e.contains("no *.elf, *.cubin or *.co found"), "{e}");
}

#[test]
fn half_a_packet_stamp_is_a_broken_build() {
    let dir = tmp("halfstamp");
    std::fs::write(dir.join("a.elf"), b"ELF-a").unwrap();
    let e = discover_objects(&dir, &Stub(vec!["plow_packet_hash_lo_0000dead"]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("half a packet-hash stamp"), "{e}");
}

#[test]
fn both_stamp_halves_compose_into_the_pairing_hash() {
    let dir = tmp("stamp");
    std::fs::write(dir.join("a.elf"), b"ELF-a").unwrap();
    let objects = discover_objects(
        &dir,
        &Stub(vec![
            "plow_packet_hash_lo_fb6fbf09",
            "plow_packet_hash_hi_9fd0e880",
        ]),
    )
    .unwrap();
    assert_eq!(
        objects[0].packet_hash.as_deref(),
        Some("0x9fd0e880fb6fbf09")
    );
    // The stamp markers are not arms: recording them would put the pairing
    // hash in two places that can disagree.
    assert!(objects[0].arms.is_empty(), "{:?}", objects[0].arms);
}

#[test]
fn objects_stamped_for_different_packets_are_a_mixed_build() {
    struct PerFile;
    impl SymbolReader for PerFile {
        fn symbols(&self, path: &Path) -> Result<Vec<String>> {
            let hi = if path.ends_with("a.elf") {
                "9fd0e880"
            } else {
                "11112222"
            };
            Ok(vec![
                "plow_packet_hash_lo_fb6fbf09".into(),
                format!("plow_packet_hash_hi_{hi}"),
            ])
        }
    }
    let dir = tmp("mixed");
    std::fs::write(dir.join("a.elf"), b"ELF-a").unwrap();
    std::fs::write(dir.join("b.elf"), b"ELF-b").unwrap();
    let e = discover_objects(&dir, &PerFile).unwrap_err().to_string();
    assert!(e.contains("stamped for different packets"), "{e}");
}

#[test]
fn markers_are_the_plow_symbols_only() {
    // `_Z9not_plow_1` contains `plow_` but not at a word boundary, and `@@VER`
    // is not a word character, so the run stops before it.
    let out = concat!(
        "    1: 0000 0 FUNC GLOBAL DEFAULT 1 plow_k3_arms_1\n",
        "    2: 0000 0 FUNC GLOBAL DEFAULT 1 _Z9not_plow_1\n",
        "    3: 0000 0 NOTYPE GLOBAL DEFAULT 1 plow_gemv_walk_1@@VER\n",
    );
    assert_eq!(markers(out), ["plow_gemv_walk_1", "plow_k3_arms_1"]);
    assert!(markers("plow_").is_empty(), "a bare prefix is not a marker");
}

/// Every other test here hands [`discover_objects`] a stubbed reader — and so
/// did the Python packer's self-test, which stubbed `elf_symbols` out with the
/// note that "the symbol table is the runtime's business". So the one path that
/// matters on AMD, reading `.symtab` with a real `readelf`, was never covered
/// on either side, and the cubins of the one published objset carry no
/// `plow_*` symbol and no stamp, so the real-hardware gate could not reach it
/// either.
///
/// A compiler on this host closes that without a GPU: a device object is an
/// ELF container like any other, and both producers only ever shelled out to
/// `readelf -sW` — nothing vendor-specific. Skips rather than fails where
/// there is no compiler or no readelf, because neither is this crate's
/// dependency.
#[test]
fn readelf_reads_arms_and_a_stamp_from_real_objects() {
    let dir = tmp("readelf-real");
    let objs = dir.join("objs");
    std::fs::create_dir_all(&objs).unwrap();
    let sources = [
        (
            "a.elf",
            "void plow_k3_arms_1(void) {}\nvoid plow_gemv_walk_1(void) {}\n",
        ),
        // Composed as `0x{hi<<32|lo:016x}`, so this pair must read back as
        // 0x9fd0e880fb6fbf09.
        (
            "b.elf",
            "void plow_mla_qh8_1(void) {}\n\
             void plow_packet_hash_lo_fb6fbf09(void) {}\n\
             void plow_packet_hash_hi_9fd0e880(void) {}\n",
        ),
        // A pinned vendor code object. It carries a `plow_*` symbol here on
        // purpose: identity is the digest alone, and reading this one would put
        // an arm in the manifest that no loader consults.
        ("v.co", "void plow_vendor_fmoe_64row(void) {}\n"),
    ];
    for (name, body) in sources {
        let c = dir.join(name).with_extension("c");
        std::fs::write(&c, body).unwrap();
        let mut built = false;
        for cc in ["cc", "gcc", "clang"] {
            match std::process::Command::new(cc)
                .arg("-c")
                .arg("-o")
                .arg(objs.join(name))
                .arg(&c)
                .output()
            {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => panic!("{cc}: {e}"),
                Ok(out) if !out.status.success() => {
                    panic!("{cc} {name}: {}", String::from_utf8_lossy(&out.stderr))
                }
                Ok(_) => {
                    built = true;
                    break;
                }
            }
        }
        if !built {
            eprintln!("skipped: no C compiler on this host");
            return;
        }
    }

    let objects = match discover_objects(&objs, &Readelf) {
        Err(e) if e.to_string().contains("no readelf available") => {
            eprintln!("skipped: {e}");
            return;
        }
        other => other.unwrap(),
    };

    // `.elf` first, then `.co`: a directory that gains an object must not
    // reorder one that is already published.
    let names: Vec<&str> = objects.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["a.elf", "b.elf", "v.co"]);
    // Sorted, because `readelf` lists `.symtab` in its own order and the
    // manifest is content-addressed.
    assert_eq!(objects[0].arms, ["plow_gemv_walk_1", "plow_k3_arms_1"]);
    assert!(objects[0].packet_hash.is_none(), "no stamp means general");
    assert_eq!(
        objects[1].arms,
        ["plow_mla_qh8_1"],
        "the stamp is not an arm"
    );
    assert_eq!(
        objects[1].packet_hash.as_deref(),
        Some("0x9fd0e880fb6fbf09")
    );
    assert!(
        objects[2].arms.is_empty() && objects[2].packet_hash.is_none(),
        "a vendor .co is identified by its digest: {:?}",
        objects[2]
    );
}

// --- the bundle --------------------------------------------------------------

/// A speech asset directory: the four base roles plus the four a TTS build
/// adds.
fn speech_assets(name: &str) -> PathBuf {
    let dir = tmp(name);
    for (file, body) in [
        ("model.pkt", "PLOWDEV"),
        ("plow_config.h", "#define X 1"),
        ("codec.pkt", "PLOWDEV-codec"),
        ("cublaslt_algos.jsonl", "{}"),
        ("lean-checks.json", "[]"),
        ("codec.lean-checks.json", "[]"),
    ] {
        std::fs::write(dir.join(file), body).unwrap();
    }
    std::fs::write(dir.join("weights.json"), br#"{"network":"veena"}"#).unwrap();
    std::fs::write(
        dir.join("build.json"),
        br#"{"pairing":{"hash":"0x541009da5f9890c1"},
             "features":{"fp8_kv":true,"mxfp4_weights":true,"w8a8":false}}"#,
    )
    .unwrap();
    dir
}

fn objset_for(dir: &Path) -> ObjSet {
    std::fs::write(dir.join("interp_sm90a.cubin"), b"ELF-cubin").unwrap();
    build_objset(&veena(), dir, discover_objects(dir, &Refuse).unwrap()).unwrap()
}

#[test]
fn every_speech_role_reaches_the_bundle() {
    let dir = speech_assets("roles");
    let objset = objset_for(&dir);
    let b = build_bundle(&dir, &veena(), Some(&objset), &BundleArgs::default()).unwrap();

    let mut got: Vec<(&str, &str)> = b
        .files
        .iter()
        .map(|f| (f.role.as_str(), f.name.as_str()))
        .collect();
    got.sort();
    assert_eq!(
        got,
        [
            ("algo_table", "cublaslt_algos.jsonl"),
            ("asset_manifest", "weights.json"),
            ("build_manifest", "build.json"),
            ("codec_packet", "codec.pkt"),
            ("config_header", "plow_config.h"),
            ("packet", "model.pkt"),
            ("packet_checks", "lean-checks.json"),
            ("sidecar_checks", "codec.lean-checks.json"),
        ]
    );
    // The objset's objects share the directory and are the objset's, not the
    // bundle's — they travel under `objects/` in the archive.
    assert!(!b.files.iter().any(|f| f.name.ends_with(".cubin")));
    // `weights.json`'s `network`, not the recipe's architecture family.
    assert_eq!(b.network, "veena");
    assert_eq!(b.pairing_hash.as_deref(), Some("0x541009da5f9890c1"));
    assert_eq!(b.objset.as_ref().unwrap().objset_id, objset.objset_id);
    assert_eq!(
        b.objset.as_ref().unwrap().manifest,
        format!("v1/objsets/{}.json", objset.objset_id)
    );
    assert_eq!(b.generation, 0, "assigned at publish");
    assert_eq!(b.variant_id, "", "assigned at publish");
    b.validate().unwrap();

    // Round-trip through the schema the registry enforces. Free, because the
    // manifest IS that struct — which is the whole point of the port.
    let bytes = canonical_json(&b).unwrap();
    assert_eq!(serde_json::from_slice::<Bundle>(&bytes).unwrap(), b);
}

/// A file no role claims is REPORTED, never skipped: a silently dropped asset
/// publishes an immutable generation that loads and serves wrongly.
#[test]
fn an_unclaimed_file_is_left_out_and_named() {
    let dir = speech_assets("unclaimed");
    let objset = objset_for(&dir);
    std::fs::write(dir.join("libplow_ext.so"), b"stray").unwrap();
    // A tokenizer the checkpoint owns is skipped QUIETLY — it is classified by
    // provenance, not by name.
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let b = build_bundle(&dir, &veena(), Some(&objset), &BundleArgs::default()).unwrap();
    assert!(!b.files.iter().any(|f| f.name == "libplow_ext.so"));
    assert!(!b.files.iter().any(|f| f.name == "tokenizer.json"));
    assert_eq!(b.tokenizer.source, Provenance::Checkpoint);
    assert!(
        b.tokenizer.sha256.is_none(),
        "a checkpoint tokenizer is not pinned"
    );
}

#[test]
fn a_mismatched_packet_stamp_is_refused_at_pack_time() {
    let dir = speech_assets("pairing");
    let mut objset = objset_for(&dir);
    objset.objects[0].packet_hash = Some("0xdeadbeefdeadbeef".into());
    let e = build_bundle(&dir, &veena(), Some(&objset), &BundleArgs::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("this packet is 0x541009da5f9890c1"), "{e}");

    // And a stamped object with no pairing hash to compare against: the same
    // refusal, twenty frames earlier than the loader's.
    std::fs::write(dir.join("build.json"), b"{}").unwrap();
    let e = build_bundle(&dir, &veena(), Some(&objset), &BundleArgs::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("records no pairing hash"), "{e}");
}

#[test]
fn a_directory_without_a_packet_is_not_an_asset_directory() {
    let dir = speech_assets("nopacket");
    let objset = objset_for(&dir);
    std::fs::remove_file(dir.join("model.pkt")).unwrap();
    let e = build_bundle(&dir, &veena(), Some(&objset), &BundleArgs::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("no model.pkt"), "{e}");
}

/// A `checkpoint`-sourced `tokenizer.json` must NOT be shipped — `plowc`
/// symlinks the checkpoint's own file, and shipping it would redistribute HF
/// content and risk it diverging from the checkpoint it must match.
#[test]
fn a_derived_tokenizer_must_be_a_real_file() {
    let dir = speech_assets("tokenizer");
    let objset = objset_for(&dir);
    let args = BundleArgs {
        tokenizer: Tokenizer::Derived {
            generator: "scripts/kimi_k3_tokenizer.py".into(),
            verified: true,
        },
        ..Default::default()
    };

    let real = tmp("tok-src").join("tokenizer.json");
    std::fs::write(&real, br#"{"model":{}}"#).unwrap();
    std::os::unix::fs::symlink(&real, dir.join("tokenizer.json")).unwrap();
    let e = build_bundle(&dir, &veena(), Some(&objset), &args)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("a symlink means it came from the checkpoint"),
        "{e}"
    );

    std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
    std::fs::copy(&real, dir.join("tokenizer.json")).unwrap();
    let b = build_bundle(&dir, &veena(), Some(&objset), &args).unwrap();
    assert_eq!(b.tokenizer.source, Provenance::Derived);
    assert!(b.tokenizer.sha256.is_some() && b.tokenizer.verified);
    let shipped = b.files.iter().find(|f| f.name == "tokenizer.json").unwrap();
    assert_eq!(shipped.role, "tokenizer");
    assert_eq!(Some(&shipped.sha256), b.tokenizer.sha256.as_ref());
}

#[test]
fn a_build_host_path_never_reaches_a_published_manifest() {
    let e = no_build_host_paths(br#"{"recipe":"/home/me/plow/r.toml"}"#, "bundle.json")
        .unwrap_err()
        .to_string();
    assert!(e.contains("build-host path (/home/)"), "{e}");
    no_build_host_paths(
        br#"{"recipe":"recipes/infervisor/veena/x.toml"}"#,
        "bundle.json",
    )
    .unwrap();
}

#[test]
fn a_label_is_derived_from_the_target_when_the_recipe_omits_one() {
    let features = BTreeMap::from([
        ("fp8_kv".to_string(), true),
        ("mxfp4_weights".to_string(), true),
        ("w8a8".to_string(), false),
    ]);
    let id = amd();
    assert_eq!(
        default_label(&id.target, &id.parallel, id.max_ctx, &features),
        "gfx942-mi325x-tp8-32k-fp8_kv-mxfp4_weights"
    );
    // A context that is not a whole number of KiB is spelled out.
    assert_eq!(
        default_label(&id.target, &id.parallel, 18000, &features),
        "gfx942-mi325x-tp8-18000-fp8_kv-mxfp4_weights"
    );
    // A multi-word SKU becomes one hyphenated token.
    let v = veena();
    assert_eq!(
        default_label(&v.target, &v.parallel, v.max_ctx, &BTreeMap::new()),
        "sm_90a-h100-sxm5-tp1-2k"
    );
}

// --- the recipe --------------------------------------------------------------

fn write_recipe(dir: &Path, body: &str) -> PathBuf {
    let p = dir.join("r.toml");
    std::fs::write(&p, body).unwrap();
    p
}

const MINIMAL: &str = r#"
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

#[test]
fn a_recipe_carries_the_whole_identity() {
    let dir = tmp("recipe");
    let p = write_recipe(&dir, MINIMAL);
    let id = read_recipe(&p, Overrides::default()).unwrap();
    assert_eq!(id.namespace, "infervisor");
    assert_eq!(id.name, "veena");
    assert_eq!(id.label.as_deref(), Some("sm90a-h100-tp1"));
    assert_eq!(id.plow_git, "1972fa37ac7a6f632cd5ece0b32fe37ab76eec75");
    assert_eq!(id.target, veena().target);
    assert_eq!(id.parallel, veena().parallel);
    assert_eq!(id.max_ctx, 2048);
    assert_eq!(id.toolchain, "cuda-13.4+gcc-13");
    assert_eq!(id.checkpoint_source, "hf:maya-research/Veena");
    assert_eq!(id.checkpoint_shards, 1);
    assert_eq!(id.checkpoint_layout, "hf-safetensors", "the default layout");
    assert_eq!(id.runtime_env["PLOW_VMM_PREFIX"], "1");
    // With no `[objects]` step the recipe itself drove the emit, so it IS the
    // script the objset records.
    assert_eq!(id.script, p.to_string_lossy());
    // A recipe's `network` names the ARCHITECTURE family; the bundle's names
    // the registry key, and comes from `weights.json`. Nothing reads it here.
    assert!(!format!("{id:?}").contains("llama"));
}

#[test]
fn an_override_reproduces_a_manifest_whose_recipe_moved() {
    let dir = tmp("override");
    let p = write_recipe(&dir, MINIMAL);
    let id = read_recipe(
        &p,
        Overrides {
            plow_git: Some("5f64e15810a219bbd4b43d6037204125e8361aad"),
            toolchain: Some("cuda-12.9-nix"),
        },
    )
    .unwrap();
    assert_eq!(id.plow_git, "5f64e15810a219bbd4b43d6037204125e8361aad");
    assert_eq!(id.toolchain, "cuda-12.9-nix");
}

#[test]
fn a_recipe_missing_an_identity_field_is_refused_by_name() {
    let dir = tmp("missing");
    for (drop, want) in [
        (
            "toolchain = \"cuda-13.4+gcc-13\"\n",
            "[target] has no `toolchain`",
        ),
        ("max_ctx = 2048\n", "declares `max_ctx`"),
        (
            "source = \"hf:maya-research/Veena\"\n",
            "[checkpoint] has no `source`",
        ),
        ("name = \"veena\"\n", "no `name`"),
        (
            "plow_git = \"1972fa37ac7a6f632cd5ece0b32fe37ab76eec75\"\n",
            "no `plow_git`",
        ),
    ] {
        let p = write_recipe(&dir, &MINIMAL.replace(drop, ""));
        let e = read_recipe(&p, Overrides::default())
            .unwrap_err()
            .to_string();
        assert!(
            e.contains(want),
            "dropping {drop:?} gave {e:?}, wanted {want:?}"
        );
    }
}

/// `mem_bytes` was a `--target` flag on the Python packer, so no recipe carried
/// it — and 40 of the 41 still do not. `hwspec` already records it per part,
/// and reading it from there is what lets `--recipe` replace the flag for every
/// model instead of only the one whose recipe was hand-edited to suit.
#[test]
fn mem_bytes_comes_from_the_part_unless_the_recipe_states_one() {
    let dir = tmp("mem-bytes");

    // A known part: dropping the field derives the SKU's memory.
    let p = write_recipe(&dir, &MINIMAL.replace("mem_bytes = 85899345920\n", ""));
    let id = read_recipe(&p, Overrides::default()).unwrap();
    assert_eq!(id.target.mem_bytes, 85899345920, "H100 SXM5 is 80 GiB");

    // A stated figure wins: `hwspec` records the SMALLEST shipping
    // configuration of a unified-memory part, which is a floor rather than the
    // memory of the target a variant is published for.
    let p = write_recipe(&dir, &MINIMAL.replace("85899345920", "25769803776"));
    let id = read_recipe(&p, Overrides::default()).unwrap();
    assert_eq!(id.target.mem_bytes, 25769803776);

    // Neither the recipe nor the registry has it: refused, naming both remedies.
    let p = write_recipe(
        &dir,
        &MINIMAL
            .replace("mem_bytes = 85899345920\n", "")
            .replace("sku = \"H100 SXM5\"", "sku = \"H100 SXM9\""),
    );
    let e = read_recipe(&p, Overrides::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a part `hwspec` describes"), "{e}");

    // Enabled SMs are fixed by the part, so a disagreement is a typo.
    let p = write_recipe(&dir, &MINIMAL.replace("units = 132", "units = 131"));
    let e = read_recipe(&p, Overrides::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("units = 131 but H100 SXM5 has 132"), "{e}");
}

/// The Python packer derived the vendor from the isa prefix while the recipe
/// declares it. Preferring the recipe means a disagreement must be refused,
/// not published under a vendor nothing will ever match.
#[test]
fn a_vendor_that_disagrees_with_the_isa_is_refused() {
    let dir = tmp("vendor");
    let p = write_recipe(
        &dir,
        &MINIMAL.replace("vendor = \"nvidia\"", "vendor = \"amd\""),
    );
    let e = read_recipe(&p, Overrides::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("is a nvidia isa"), "{e}");
}

#[test]
fn a_short_plow_git_is_not_provenance() {
    let dir = tmp("shortgit");
    let p = write_recipe(
        &dir,
        &MINIMAL.replace("1972fa37ac7a6f632cd5ece0b32fe37ab76eec75", "1972fa37"),
    );
    let e = read_recipe(&p, Overrides::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("full 40-hex commit"), "{e}");
}

// --- end to end --------------------------------------------------------------

#[test]
fn generate_produces_both_manifests_from_a_recipe_and_a_directory() {
    let assets = speech_assets("generate");
    std::fs::write(assets.join("interp_sm90a.cubin"), b"ELF-cubin").unwrap();
    let p = write_recipe(&tmp("generate-recipe"), MINIMAL);

    let g = generate(
        &p,
        &assets,
        Some(&assets),
        Overrides::default(),
        &BundleArgs::default(),
    )
    .unwrap();
    let (objset, objset_bytes) = g.objset.as_ref().unwrap();

    assert_eq!(g.bundle.label, "sm90a-h100-tp1");
    assert_eq!(g.bundle.files.len(), 8);
    assert_eq!(objset.objects.len(), 1);
    // The bytes in the archive are the canonical ones, and the bundle's digest
    // of the objset is taken over exactly those.
    assert_eq!(&canonical_json(&g.bundle).unwrap(), &g.bundle_bytes);
    assert_eq!(
        g.bundle.objset.as_ref().unwrap().sha256,
        sha256_bytes(objset_bytes)
    );
    // Both deserialize into the `deny_unknown_fields` structs the registry uses.
    serde_json::from_slice::<Bundle>(&g.bundle_bytes).unwrap();
    serde_json::from_slice::<ObjSet>(objset_bytes).unwrap();
}

/// One command: a CUDA recipe emits its cubins into the asset directory, so
/// naming the objects separately is not something an operator should have to
/// know. Omitting it must find exactly what naming it finds.
#[test]
fn the_object_directory_defaults_to_the_asset_directory() {
    let assets = speech_assets("noobjects");
    std::fs::write(assets.join("interp_sm90a.cubin"), b"ELF-cubin").unwrap();
    let p = write_recipe(&tmp("noobjects-recipe"), MINIMAL);
    let implied = generate(
        &p,
        &assets,
        None,
        Overrides::default(),
        &BundleArgs::default(),
    )
    .unwrap();
    let named = generate(
        &p,
        &assets,
        Some(&assets),
        Overrides::default(),
        &BundleArgs::default(),
    )
    .unwrap();
    assert_eq!(implied.bundle_bytes, named.bundle_bytes);
    assert_eq!(
        implied.objset.unwrap().0.objset_id,
        named.objset.unwrap().0.objset_id
    );
}

/// An asset directory with no objects in it is the shape of a forgotten
/// `--objects` on a recipe whose build script writes them elsewhere.
#[test]
fn a_target_that_ships_objects_needs_them_somewhere() {
    let assets = speech_assets("objectless");
    let p = write_recipe(&tmp("objectless-recipe"), MINIMAL);
    let e = generate(
        &p,
        &assets,
        None,
        Overrides::default(),
        &BundleArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("no *.elf, *.cubin or *.co found"), "{e}");
}

/// Nothing records a compiler for a runtime-compiled target, so its recipe
/// may omit `toolchain`; a CUDA recipe without one is still refused (see
/// `a_recipe_missing_an_identity_field_is_refused_by_name`).
#[test]
fn a_runtime_compiled_recipe_needs_no_toolchain() {
    let assets = speech_assets("apple-no-toolchain");
    let recipe = MINIMAL
        .replace("vendor = \"nvidia\"", "vendor = \"apple\"")
        .replace("isa = \"sm_90a\"", "isa = \"m4\"")
        .replace("sku = \"H100 SXM5\"", "sku = \"M4 Max\"")
        .replace("toolchain = \"cuda-13.4+gcc-13\"\n", "");
    assert!(!recipe.contains("toolchain"), "{recipe}");
    let p = write_recipe(&tmp("apple-no-toolchain"), &recipe);
    let g = generate(&p, &assets, None, Overrides::default(), &BundleArgs::default()).unwrap();
    assert!(g.objset.is_none());
}

/// Apple's Metal shaders compile on the serving machine from the packet, so its
/// bundle carries no objset and must not be handed an object directory.
#[test]
fn a_runtime_compiled_target_has_no_objset() {
    let assets = speech_assets("apple");
    let recipe = MINIMAL
        .replace("vendor = \"nvidia\"", "vendor = \"apple\"")
        .replace("isa = \"sm_90a\"", "isa = \"m4\"")
        .replace("sku = \"H100 SXM5\"", "sku = \"M4 Max\"");
    let p = write_recipe(&tmp("apple-recipe"), &recipe);

    let g = generate(
        &p,
        &assets,
        None,
        Overrides::default(),
        &BundleArgs::default(),
    )
    .unwrap();
    assert!(g.objset.is_none() && g.bundle.objset.is_none());
    assert!(
        g.bundle.pairing_hash.is_none(),
        "no objset, nothing to pair"
    );

    let e = generate(
        &p,
        &assets,
        Some(&assets),
        Overrides::default(),
        &BundleArgs::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("compiles its kernels at run time"), "{e}");
}

/// A private or air-gapped checkpoint travels in the distribution as blobs.
#[test]
fn a_carried_checkpoint_becomes_blobs() {
    let assets = speech_assets("dist-ckpt");
    let objset = objset_for(&assets);
    let ckpt = tmp("ckpt");
    std::fs::write(ckpt.join("model-00001.safetensors"), b"shard").unwrap();
    std::fs::write(ckpt.join("config.json"), b"{}").unwrap();
    std::fs::create_dir_all(ckpt.join(".cache/huggingface")).unwrap();
    std::fs::write(ckpt.join(".cache/huggingface/x"), b"tool state").unwrap();
    std::fs::create_dir_all(ckpt.join("sub")).unwrap();
    std::fs::write(ckpt.join("sub/extra.bin"), b"nested").unwrap();

    let b = build_bundle(
        &assets,
        &veena(),
        Some(&objset),
        &BundleArgs {
            checkpoint_dir: Some(&ckpt),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(b.checkpoint.source, plow_asset::dist::DIST_SOURCE);
    assert_eq!(b.checkpoint.revision, veena().checkpoint_revision);
    // `prepare` globs `*.safetensors` in the snapshot's top level only.
    assert_eq!(b.checkpoint.shards, 1);
    assert_eq!(
        b.checkpoint
            .files
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["config.json", "model-00001.safetensors", "sub/extra.bin"],
        "hidden tool state is not the checkpoint"
    );
    assert_eq!(b.checkpoint.bytes, 5 + 2 + 6);
}

#[test]
fn a_directory_with_no_shards_is_not_a_checkpoint() {
    let assets = speech_assets("badckpt");
    let objset = objset_for(&assets);
    let ckpt = tmp("badckpt-src");
    std::fs::write(ckpt.join("config.json"), b"{}").unwrap();
    let e = build_bundle(
        &assets,
        &veena(),
        Some(&objset),
        &BundleArgs {
            checkpoint_dir: Some(&ckpt),
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("no top-level *.safetensors"), "{e}");
}

/// Each rung is pinned by its manifest DIGEST, not just its id: the objects
/// inside a rung manifest are verified against digests that manifest itself
/// names, so an unverified one could serve arbitrary decode objects — on the
/// decode hot path.
#[test]
fn rung_overrides_are_pinned_by_digest_and_sorted() {
    let assets = speech_assets("rungs");
    let objset = objset_for(&assets);
    let dir = tmp("rung-manifests");
    let mut paths = Vec::new();
    for (max, body) in [(8u32, b"ELF-r8".as_slice()), (1, b"ELF-r1")] {
        let d = dir.join(format!("r{max}"));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("interp.elf"), body).unwrap();
        let rung = build_objset(
            &amd(),
            &d,
            discover_objects(&d, &Stub(vec!["plow_arms_1"])).unwrap(),
        )
        .unwrap();
        let p = d.join("objset.json");
        std::fs::write(&p, canonical_json(&rung).unwrap()).unwrap();
        paths.push((max, p, rung));
    }

    let b = build_bundle(
        &assets,
        &veena(),
        Some(&objset),
        &BundleArgs {
            lowrung: paths.iter().map(|(m, p, _)| (*m, p.clone())).collect(),
            ..Default::default()
        },
    )
    .unwrap();
    let got = &b.objset.as_ref().unwrap().lowrung;
    assert_eq!(
        got.iter().map(|r| r.max).collect::<Vec<_>>(),
        [1, 8],
        "sorted by width"
    );
    for r in got {
        let (_, _, rung) = paths
            .iter()
            .find(|(_, _, o)| o.objset_id == r.objset_id)
            .unwrap();
        assert_eq!(r.sha256, sha256_bytes(&canonical_json(rung).unwrap()));
    }

    // A duplicate width is a mistake, not a last-writer-wins override.
    let e = build_bundle(
        &assets,
        &veena(),
        Some(&objset),
        &BundleArgs {
            lowrung: vec![(1, paths[0].1.clone()), (1, paths[1].1.clone())],
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("duplicate rung width"), "{e}");
}

/// The one producer has to read every recipe in the tree. Deleting
/// `--bundle`/`--objset` is what made this load-bearing: a recipe this refuses
/// is now a model nothing can pack at all, where before there was a second
/// path to fall back to.
///
/// Two declarations are the publisher's rather than the recipe's. 24 of the 41
/// do not record `[target] toolchain`, a build fact with no registry to come
/// from, so `--toolchain` stands in here exactly as a publisher would pass it;
/// and 7 declare no `max_ctx` — 3 `refused` builds and 4 ASR/VAD packets, where
/// a context length is not a property of the model at all. Those must declare
/// one before they can be published. NOTHING ELSE may refuse: any other
/// message means a recipe the only producer cannot pack.
///
/// Written because the sweep caught two: `[target] mem_bytes` was recorded by
/// exactly one recipe — the one hand-edited to suit the generator — and the
/// five Xeon recipes were refused outright for declaring `vendor = "intel"`
/// beside `isa = "amx"`, which is the correct declaration for a CPU part.
#[test]
fn every_recipe_in_the_tree_is_readable() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../recipes");
    if !root.is_dir() {
        eprintln!("skipped: {} is not present", root.display());
        return;
    }
    let mut files = Vec::new();
    for ns in std::fs::read_dir(&root).unwrap() {
        for name in std::fs::read_dir(ns.unwrap().path()).unwrap() {
            for f in std::fs::read_dir(name.unwrap().path()).unwrap() {
                let p = f.unwrap().path();
                if p.extension().is_some_and(|e| e == "toml") {
                    files.push(p);
                }
            }
        }
    }
    files.sort();
    assert!(files.len() > 30, "only {} recipes found", files.len());

    let over = Overrides {
        plow_git: None,
        toolchain: Some("rocm-7.14.0-nix"),
    };
    let refused: Vec<String> = files
        .iter()
        .filter_map(|p| match read_recipe(p, over) {
            Ok(_) => None,
            Err(e) if e.to_string().contains("declares `max_ctx`") => None,
            // A checkpoint that is neither a Hugging Face repository nor
            // carried by the distribution (`local:`, `pip:`, a re-keyed
            // local file set) was never packable; the rule only names it now.
            Err(e) if e.to_string().contains("is not `hf:<org>/<repo>` or `dist`") => None,
            Err(e) => Some(format!(
                "{}: {e}",
                p.strip_prefix(&root).unwrap_or(p).display()
            )),
        })
        .collect();
    assert!(
        refused.is_empty(),
        "{} of {} recipes cannot be packed:\n{}",
        refused.len(),
        files.len(),
        refused.join("\n")
    );
}

#[test]
fn a_sidecar_packet_and_its_receipts_are_bundled_by_suffix() {
    assert_eq!(super::classify("model.pkt"), Some("packet"));
    assert_eq!(super::classify("codec.pkt"), Some("codec_packet"));
    assert_eq!(super::classify("encoder.pkt"), Some("sidecar_packet"));
    assert_eq!(super::classify("s3gen.pkt"), Some("sidecar_packet"));
    assert_eq!(
        super::classify("encoder.lean-checks.json"),
        Some("sidecar_checks")
    );
    assert_eq!(super::classify("lean-checks.json"), Some("packet_checks"));
    assert_eq!(super::classify("notes.txt"), None);
}
