//! The container half of phase 2: a synthetic parent packet and extension, written with the
//! real emitter and read back with the real loader.
//!
//! The rules themselves are tested in `plow_asset::extension`; what is tested here is that the
//! two containers are told apart, that a parent's facts come out of the file the contract
//! expects, and that discovery and the phase-3 object pinning behave.

use std::path::{Path, PathBuf};

use packet::dev::DevInst;
use packet::devbuild::{Model, Program, TensorDecl};
use plow_asset::extension::{self, KvRing, ProgramRole};

use super::*;

// --- fixtures ----------------------------------------------------------------

const N_CU: u32 = 4;

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "plow-ext-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `unwrap_err().to_string()` needs `Debug` on the Ok type, and neither `DevBlob` nor
/// `LoadedExtension` carries one — deriving it on a struct holding every instruction stream
/// would be a debug print nobody wants.
fn err<T>(r: Result<T>) -> String {
    match r {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e.to_string(),
    }
}

fn prog(n_counter: u32) -> Program {
    Program {
        hier_base: 0,
        n_cu: N_CU,
        n_counter,
        insts: vec![DevInst {
            op: packet::dev::DevOp::Gemm as u16,
            blocks: 1,
            ..Default::default()
        }],
        stream: vec![],
        stream_ofs: vec![0; N_CU as usize],
        stream_len: vec![0; N_CU as usize],
        waits: vec![],
        succs: vec![],
        tensors: vec![],
        gq_stream: vec![],
        gq_seg_ofs: vec![0, 0],
        l2_sms: 0,
        l2_domains: 0,
    }
}

fn tensor_decls() -> Vec<TensorDecl> {
    vec![
        TensorDecl {
            name: "tok_emb".into(),
            bytes: 4096,
            init: None,
        },
        TensorDecl {
            name: "rope_cos".into(),
            bytes: 8,
            init: Some(vec![1u8; 8]),
        },
    ]
}

/// A parent: three prefill buckets and two decode rungs over one tensor table.
fn parent_model() -> Model {
    Model {
        n_cu: N_CU,
        target: 0,
        tensors: tensor_decls(),
        progs: (0..5).map(|i| prog(10 + i)).collect(),
        kv_row_insts: vec![],
        prog_t: vec![512, 2048, 8192, 1, 8],
        gen: vec![],
    }
}

/// A decode-rung `prog_t`. An extension STATES its roles; an unmarked program is a prefill
/// bucket. See `packet::devbuild::DECODE_RUNG_PROG`.
fn rung(rows: u32) -> u32 {
    packet::devbuild::decode_rung_program_t(rows)
}

/// An extension: programs only, tensor table replaced by a reference to the parent's.
fn ext_model(progs: usize, prog_t: Vec<u32>) -> Model {
    Model {
        n_cu: N_CU,
        target: 0,
        tensors: vec![],
        progs: (0..progs).map(|i| prog(20 + i as u32)).collect(),
        kv_row_insts: vec![],
        prog_t,
        gen: vec![],
    }
}

const CONFIG: &str = "\
#define PLOW_PACKET_HASH 0x00000000deadbeefull
#define PLOW_PACKET_GQA 8
#define PLOW_PACKET_DECODE_BATCH 8
#define PLOW_PACKET_LINEAR_BIAS 0
#define PLOW_PACKET_ROPE_HALF_HD64 0
#define PLOW_PACKET_ATTENTION_SINKS 0
#define PLOW_PACKET_HAS_FLASH_DECODE_FP8 1
#define PLOW_PACKET_HAS_FLASH_MLA_PREFILL_FP8 1
#ifndef PLOW_HAS_FLASH_HD64
#define PLOW_HAS_FLASH_HD64 0
#endif
#ifndef PLOW_HAS_FLASH_HD128
#define PLOW_HAS_FLASH_HD128 1
#endif
#ifndef PLOW_HAS_FLASH_HD256
#define PLOW_HAS_FLASH_HD256 0
#endif
#ifndef PLOW_HAS_FLASH_HD512
#define PLOW_HAS_FLASH_HD512 1
#endif
";

const EXT_CONFIG_HASH: u64 = 0x0000_0000_cafe_f00d;

fn ext_config() -> String {
    CONFIG.replace(
        "0x00000000deadbeefull",
        &format!("0x{EXT_CONFIG_HASH:016x}ull"),
    )
}

fn requires_json(hash: u64) -> String {
    serde_json::to_string_pretty(&extension::Requires {
        version: 1,
        arch: "gfx942".into(),
        pairing_hash: format!("0x{hash:016x}"),
        objects: vec![extension::RequiredObject {
            stem: "interp_prefill".into(),
            sha256: "a".repeat(64),
            rows: Some(4096),
        }],
    })
    .unwrap()
}

/// Write a parent serving directory and return `(dir, image)`.
fn write_parent(tag: &str, kv: Option<(u64, u64)>) -> (PathBuf, Vec<u8>) {
    let dir = tmpdir(tag);
    let image = parent_model().to_blob_v6(&[]);
    std::fs::write(dir.join("model.pkt"), &image).unwrap();
    std::fs::write(dir.join("plow_config.h"), CONFIG).unwrap();
    let mut shapes = serde_json::json!({ "max_chunk": 8192 });
    if let Some((window, ring)) = kv {
        shapes["kv_window"] = window.into();
        shapes["kv_ring_rows"] = ring.into();
    }
    std::fs::write(
        dir.join("build.json"),
        serde_json::json!({
            "pairing": { "hash": "0x00000000deadbeef" },
            "shapes": shapes,
        })
        .to_string(),
    )
    .unwrap();
    (dir, image)
}

/// Write `<parent>.ext/<name>/` and return its directory.
fn write_ext(parent: &Path, name: &str, model: &Model, parent_image: &[u8]) -> PathBuf {
    write_ext_with(parent, name, model, parent_image, None, &ext_config(), None)
}

#[allow(clippy::too_many_arguments)]
fn write_ext_with(
    parent: &Path,
    name: &str,
    model: &Model,
    parent_image: &[u8],
    tensor_digest: Option<[u8; 32]>,
    config: &str,
    requires_hash: Option<u64>,
) -> PathBuf {
    let blob = DevBlob::parse(parent_image).unwrap();
    let mut root = parent.as_os_str().to_os_string();
    root.push(EXT_DIR_SUFFIX);
    let dir = PathBuf::from(root).join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let pref = packet::ext::BlobParentRef::new(
        extension::packet_hash(parent_image),
        tensor_digest.unwrap_or_else(|| blob.tensor_table_digest()),
        blob.tensors.len() as u32,
    );
    std::fs::write(
        dir.join(extension::EXTENSION_FILE),
        model.to_ext_blob(&pref, &[]),
    )
    .unwrap();
    std::fs::write(dir.join(extension::CONFIG_FILE), config).unwrap();
    std::fs::write(
        dir.join(extension::REQUIRES_FILE),
        requires_json(requires_hash.unwrap_or(EXT_CONFIG_HASH)),
    )
    .unwrap();
    dir
}

// --- the container -----------------------------------------------------------

#[test]
fn an_extension_container_round_trips_with_no_tensor_table() {
    let parent_image = parent_model().to_blob_v6(&[]);
    let parent = DevBlob::parse(&parent_image).unwrap();
    assert!(parent.parent.is_none(), "a model packet has no parent");

    let pref = packet::ext::BlobParentRef::new(
        extension::packet_hash(&parent_image),
        parent.tensor_table_digest(),
        parent.tensors.len() as u32,
    );
    let image = ext_model(1, vec![4096]).to_ext_blob(&pref, &[]);
    assert_eq!(&image[..8], packet::ext::EXT_MAGIC);

    let ext = DevBlob::parse_extension(&image, false).unwrap();
    assert!(ext.tensors.is_empty(), "an extension declares no tensors");
    assert!(ext.init.is_empty());
    assert_eq!(ext.progs.len(), 1);
    assert_eq!(ext.progs[0].t, 4096);
    let got = ext.parent.unwrap();
    assert_eq!(got.parent_hash, extension::packet_hash(&parent_image));
    assert_eq!(got.tensor_digest, parent.tensor_table_digest());
    assert_eq!(got.tensor_count, 2);
}

#[test]
fn the_two_containers_are_never_read_as_each_other() {
    let parent_image = parent_model().to_blob_v6(&[]);
    let parent = DevBlob::parse(&parent_image).unwrap();
    let pref = packet::ext::BlobParentRef::new(
        extension::packet_hash(&parent_image),
        parent.tensor_table_digest(),
        2,
    );
    let ext_image = ext_model(1, vec![4096]).to_ext_blob(&pref, &[]);

    let e = err(DevBlob::parse(&ext_image));
    assert!(e.contains("this is an extension.pkt"), "{e}");
    let e = err(DevBlob::parse_extension(&parent_image, false));
    assert!(e.contains("this is a model.pkt"), "{e}");

    // ...and `find_in_dir`, which only tests the PLOWDEV magic, must not see an extension as a
    // second model. That was the whole reason for a separate magic.
    assert!(!packet::devbuild::is_blob_magic(
        &ext_image[..8].try_into().unwrap()
    ));
    let dir = tmpdir("find");
    std::fs::write(dir.join("model.pkt"), &parent_image).unwrap();
    std::fs::write(dir.join("extension.pkt"), &ext_image).unwrap();
    assert_eq!(
        DevBlob::find_in_dir(&dir).unwrap().unwrap().file_name(),
        Some(std::ffi::OsStr::new("model.pkt"))
    );
}

#[test]
fn a_parent_packet_may_not_carry_a_parent_ref_section() {
    let mut m = parent_model();
    m.tensors = tensor_decls();
    let pref = packet::ext::BlobParentRef::new([0u8; 32], [0u8; 32], 2);
    let image = m.to_blob_v6(&[packet::devbuild::SectionData {
        kind: packet::ext::SECT_PARENT_REF,
        name: "parent".into(),
        data: pref.to_bytes(),
    }]);
    let e = err(DevBlob::parse(&image));
    assert!(e.contains("cannot extend another packet"), "{e}");
}

#[test]
fn an_extension_with_no_parent_ref_is_refused() {
    // Hand-build the container an emitter must never produce: extension magic, no reference.
    let mut image = ext_model(1, vec![4096]).to_blob_v6(&[]);
    image[..8].copy_from_slice(packet::ext::EXT_MAGIC);
    let e = err(DevBlob::parse_extension(&image, false));
    assert!(e.contains("no parent-ref section"), "{e}");
}

#[test]
#[should_panic(expected = "an extension declares no tensors of its own")]
fn the_emitter_refuses_to_put_a_tensor_table_in_an_extension() {
    let mut m = ext_model(1, vec![4096]);
    m.tensors = tensor_decls();
    let pref = packet::ext::BlobParentRef::new([0u8; 32], [0u8; 32], 2);
    let _ = m.to_ext_blob(&pref, &[]);
}

#[test]
fn a_v6_parent_blob_is_byte_identical_after_the_container_refactor() {
    // The extension writer shares `Model::container` with `to_blob_v6`. A parent must be
    // untouched by that — every shipped packet is read by this code path.
    let m = parent_model();
    let a = m.to_blob_v6(&[]);
    assert_eq!(&a[..8], packet::devbuild::BLOB_MAGIC_V6);
    let b = m.to_blob_v6(&[]);
    assert_eq!(a, b);
    let parsed = DevBlob::parse(&a).unwrap();
    assert_eq!(parsed.tensors.len(), 2);
    assert_eq!(parsed.progs.len(), 5);
    assert!(parsed.parent.is_none());
}

#[test]
fn an_extension_states_its_roles_because_it_has_no_position_to_imply_them() {
    let parent_image = parent_model().to_blob_v6(&[]);
    let parent = DevBlob::parse(&parent_image).unwrap();
    let pref = packet::ext::BlobParentRef::new(
        extension::packet_hash(&parent_image),
        parent.tensor_table_digest(),
        2,
    );
    // A lone program is `prog_t.len() - 1`, so the positional rule would call EVERY extension's
    // single program a decode rung. Unmarked means prefill bucket; the rung says so.
    let bucket = DevBlob::parse_extension(
        &ext_model(1, vec![4096]).to_ext_blob(&pref, &[]),
        false,
    )
    .unwrap();
    assert!(!bucket.progs[0].role.is_decode_rung());
    assert_eq!(
        bucket.program_roles(),
        vec![ProgramRole::PrefillBucket { rows: 4096 }]
    );

    let r = DevBlob::parse_extension(
        &ext_model(1, vec![rung(12)]).to_ext_blob(&pref, &[]),
        false,
    )
    .unwrap();
    assert!(r.progs[0].role.is_decode_rung());
    assert_eq!(r.progs[0].t, 12, "the role bit is masked out of the row count");
    assert_eq!(r.program_roles(), vec![ProgramRole::DecodeRung { rows: 12 }]);

    // A parent never sets the bit, and its ladder is still the positional one: the same
    // `prog_t` read as an extension would call every program a bucket.
    assert_eq!(
        parent.program_roles(),
        vec![
            ProgramRole::PrefillBucket { rows: 512 },
            ProgramRole::PrefillBucket { rows: 2048 },
            ProgramRole::PrefillBucket { rows: 8192 },
            ProgramRole::DecodeRung { rows: 1 },
            ProgramRole::DecodeRung { rows: 8 },
        ]
    );
    assert!(packet::devbuild::derive_roles(
        &[512, 2048, 8192, 1, 8],
        packet::devbuild::RoleSource::Stated,
        |_| 0
    )
    .iter()
    .all(|r| r.is_prefill_bucket()));
}

// --- the facts ---------------------------------------------------------------

#[test]
fn parent_facts_come_out_of_the_files_the_contract_names() {
    let (dir, image) = write_parent("facts", None);
    let blob = DevBlob::parse(&image).unwrap();
    let f = parent_facts(&dir, &image, &blob).unwrap();

    assert_eq!(f.hash, extension::packet_hash(&image));
    assert_eq!(f.tensor_count, 2);
    assert_eq!(f.tensor_digest, blob.tensor_table_digest());
    assert_eq!(f.config.get("PLOW_PACKET_GQA"), Some(8));
    assert_eq!(f.config.get(extension::BLOB_AXIS_N_CU), Some(N_CU as i64));
    assert_eq!(f.config.get(extension::BLOB_AXIS_TP_DEGREE), Some(1));
    assert_eq!(f.decode_batch, 8);
    assert_eq!(f.pairing_hash, Some(0x0000_0000_dead_beef));
    // No KV geometry in this build.json, so rule 4 is armed only for a wider bucket.
    assert_eq!(f.kv, KvRing::Unstated);
    assert_eq!(
        f.programs,
        vec![
            ProgramRole::PrefillBucket { rows: 512 },
            ProgramRole::PrefillBucket { rows: 2048 },
            ProgramRole::PrefillBucket { rows: 8192 },
            ProgramRole::DecodeRung { rows: 1 },
            ProgramRole::DecodeRung { rows: 8 },
        ]
    );
    assert!(f.arenas.growable);
    assert_eq!(f.arenas.segment_ceiling, 2048);
    assert_eq!(f.arenas.reserved.counters, 14, "the widest program's counters");
}

#[test]
fn a_stated_kv_window_reaches_rule_4() {
    let (dir, image) = write_parent("kv", Some((1024, 16_384)));
    let blob = DevBlob::parse(&image).unwrap();
    assert_eq!(
        parent_facts(&dir, &image, &blob).unwrap().kv,
        KvRing::Windowed {
            window: 1024,
            ring_rows: 16_384
        }
    );

    let (dir, image) = write_parent("kv-global", Some((0, 0)));
    let blob = DevBlob::parse(&image).unwrap();
    assert_eq!(
        parent_facts(&dir, &image, &blob).unwrap().kv,
        KvRing::FullCausal,
        "window 0 is an all-global model"
    );
}

// --- discovery ---------------------------------------------------------------

#[test]
fn discovery_finds_nothing_beside_a_plain_serving_directory() {
    let (dir, _) = write_parent("bare", None);
    assert!(discover_in(&dir).unwrap().is_empty());
}

#[test]
fn discovery_is_sorted_and_skips_directories_without_a_packet() {
    let (dir, image) = write_parent("discover", None);
    write_ext(&dir, "rung-12", &ext_model(1, vec![rung(12)]), &image);
    write_ext(&dir, "bucket-4096", &ext_model(1, vec![4096]), &image);
    let mut empty = dir.as_os_str().to_os_string();
    empty.push(EXT_DIR_SUFFIX);
    std::fs::create_dir_all(PathBuf::from(empty).join("notes")).unwrap();

    let found = discover_in(&dir).unwrap();
    assert_eq!(found.len(), 2);
    assert!(found[0].ends_with("bucket-4096"), "{found:?}");
    assert!(found[1].ends_with("rung-12"), "{found:?}");
}

// --- the merge, end to end ---------------------------------------------------

#[test]
fn a_serving_directory_with_no_extensions_loads_exactly_as_before() {
    let (dir, image) = write_parent("noext", None);
    let blob = DevBlob::parse(&image).unwrap();
    let (merged, loaded) = load(&dir, &image, &blob, false).unwrap();
    assert!(loaded.is_empty());
    assert!(merged.growth.grew.is_empty());
    assert_eq!(merged.prefill_widths(), vec![512, 2048, 8192]);
    assert_eq!(merged.decode_rungs(), vec![1, 8]);
    assert_eq!(
        merged.programs.iter().map(|(_, r)| *r).collect::<Vec<_>>(),
        blob.program_roles()
    );
}

#[test]
fn a_well_formed_extension_merges_end_to_end() {
    let (dir, image) = write_parent("merge", None);
    write_ext(&dir, "bucket-4096", &ext_model(1, vec![4096]), &image);
    let blob = DevBlob::parse(&image).unwrap();
    let (merged, loaded) = load(&dir, &image, &blob, false).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].requires.rows(), vec![4096]);
    assert_eq!(merged.prefill_widths(), vec![512, 2048, 4096, 8192]);
}

#[test]
fn an_extension_for_a_different_parent_is_refused_by_name_and_rule() {
    let (dir, image) = write_parent("wrongparent", None);
    write_ext_with(
        &dir,
        "bucket-4096",
        &ext_model(1, vec![4096]),
        &image,
        Some([0u8; 32]),
        &ext_config(),
        None,
    );
    let blob = DevBlob::parse(&image).unwrap();
    let e = err(load(&dir, &image, &blob, false));
    assert!(e.contains("bucket-4096"), "{e}");
    assert!(e.contains("rule 2 (tensor-table digest)"), "{e}");
}

#[test]
fn a_wider_bucket_on_a_parent_with_no_stated_ring_is_refused() {
    let (dir, image) = write_parent("wider", None);
    write_ext(&dir, "bucket-16384", &ext_model(1, vec![16_384]), &image);
    let blob = DevBlob::parse(&image).unwrap();
    let e = err(load(&dir, &image, &blob, false));
    assert!(e.contains("rule 4 (KV ring invariant)"), "{e}");

    // Stating the geometry makes the same extension loadable.
    let (dir, image) = write_parent("wider-ok", Some((1024, 16_384)));
    write_ext(&dir, "bucket-8192b", &ext_model(1, vec![16_384]), &image);
    let blob = DevBlob::parse(&image).unwrap();
    let e = err(load(&dir, &image, &blob, false));
    // ring 16384 < window 1024 + chunk 16384 - 1, so it is still refused — but now on the
    // arithmetic rather than on ignorance, and the message says the number.
    assert!(e.contains("needs a KV ring of 32768 rows"), "{e}");
}

#[test]
fn an_explicit_extension_list_overrides_discovery() {
    let (dir, image) = write_parent("explicit", None);
    let a = write_ext(&dir, "bucket-4096", &ext_model(1, vec![4096]), &image);
    write_ext(&dir, "rung-4", &ext_model(1, vec![rung(4)]), &image);
    let blob = DevBlob::parse(&image).unwrap();

    // `PLOW_EXTENSIONS` is parsed, not read, so this does not race the rest of the suite.
    assert_eq!(explicit(&a.display().to_string()), vec![a]);
    assert!(
        explicit("").is_empty(),
        "an empty list means serve the parent alone"
    );

    // Discovery, unset, still finds both.
    assert_eq!(discover_in(&dir).unwrap().len(), 2);
    let (merged, _) = load(&dir, &image, &blob, false).unwrap();
    assert_eq!(merged.prefill_widths(), vec![512, 2048, 4096, 8192]);
    assert_eq!(merged.decode_rungs(), vec![1, 4, 8]);
}

// --- phase 3 -----------------------------------------------------------------

#[test]
fn requires_json_must_pin_the_extensions_own_packet_hash() {
    let (dir, image) = write_parent("pin", None);
    // The common mistake: objects built against the PARENT's plow_config.h.
    let d = write_ext_with(
        &dir,
        "bucket-4096",
        &ext_model(1, vec![4096]),
        &image,
        None,
        &ext_config(),
        Some(0x0000_0000_dead_beef),
    );
    let e = err(load_one(&d, false));
    assert!(e.contains("requires.json pins packet 0x00000000deadbeef"), "{e}");
    assert!(e.contains("0x00000000cafef00d"), "{e}");
}

#[test]
fn the_extension_config_must_carry_a_packet_hash() {
    let (dir, image) = write_parent("nohash", None);
    let d = write_ext_with(
        &dir,
        "bucket-4096",
        &ext_model(1, vec![4096]),
        &image,
        None,
        "#define PLOW_PACKET_GQA 8\n",
        None,
    );
    let e = err(load_one(&d, false));
    assert!(e.contains("no PLOW_PACKET_HASH"), "{e}");
}

#[test]
fn config_pairing_hash_reads_the_generated_header() {
    assert_eq!(config_pairing_hash(CONFIG), Some(0x0000_0000_dead_beef));
    assert_eq!(config_pairing_hash("#define PLOW_X 1\n"), None);
}
