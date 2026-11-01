use super::*;
use crate::spec_draft::{DraftProgram, DraftTensors};

const PKT: &str = "1111111111111111111111111111111111111111111111111111111111111111";

fn tokenizer_json(extra: &str) -> Vec<u8> {
    format!(
        r#"{{"model":{{"type":"BPE","vocab":{{"a":0,"b":1}}}},
            "added_tokens":[{{"id":2,"content":"<bos>"}}{extra}]}}"#
    )
    .into_bytes()
}

fn manifest() -> Manifest {
    let names = [
        ("in.spec.tokens.0", Source::Scratch),
        ("in.spec.tokens.1", Source::Scratch),
        ("in.spec.hidden_row", Source::Scratch),
        ("in.spec.position", Source::Scratch),
        ("in.spec.kv_len", Source::Scratch),
        ("in.spec.slot", Source::Scratch),
        (
            "n.hn",
            Source::Import {
                access: Access::Read,
            },
        ),
        ("mtp.w", Source::Weight),
    ];
    let facts = TokenizerFacts::from_json(&tokenizer_json("")).unwrap();
    Manifest {
        contract: CONTRACT.into(),
        kind: Kind::ChainSharedKv,
        family: "gemma4_assistant".into(),
        n_cu: 132,
        requires: Requires {
            target_packet_sha256: Some(PKT.into()),
            tokenizer: Some(TokenizerReq {
                vocab_sha256: facts.vocab_sha256,
                added_tokens: vec![(2, "<bos>".into())],
            }),
            object_symbols: [("plow_row_slot_1".to_string(), 1)].into(),
            ..Default::default()
        },
        tensors: names
            .iter()
            .map(|(n, s)| Tensor {
                name: n.to_string(),
                bytes: 4096,
                source: s.clone(),
            })
            .collect(),
        draft: spec_draft::Manifest {
            version: spec_draft::VERSION,
            family: "gemma4_assistant".into(),
            kmax: 1,
            programs: vec![DraftProgram {
                rows: 1,
                k: 1,
                index: 0,
            }],
            tensors: DraftTensors {
                tokens: vec![0, 1],
                hidden_row: Some(2),
                position: 3,
                kv_len: 4,
                slot: 5,
                hidden: Some(6),
                hidden_rows: 16,
            },
            block: None,
            shard: "model-mtp.safetensors".into(),
            shard_sha256: String::new(),
        },
        weights: vec![],
    }
}

fn target() -> TargetView {
    TargetView {
        packet_sha256: PKT.into(),
        tensors: vec![("n.hn".into(), 4096), ("kv.0.k".into(), 1 << 20)],
        tokenizer: Some(
            TokenizerFacts::from_json(&tokenizer_json(r#",{"id":3,"content":"<img>"}"#)).unwrap(),
        ),
        object_symbols: [("plow_row_slot_1".to_string(), 1)].into(),
        ..Default::default()
    }
}

fn refusal(m: &Manifest, t: &TargetView) -> String {
    m.check(t).expect_err("must refuse")
}

#[test]
fn a_matching_drafter_attaches_and_round_trips() {
    let m = manifest();
    m.validate().unwrap();
    m.check(&target()).unwrap();
    let json = serde_json::to_vec(&m).unwrap();
    assert_eq!(Manifest::parse(&json).unwrap(), m);
}

#[test]
fn each_mismatch_is_refused_by_name() {
    let m = manifest();
    let mut t = target();
    t.packet_sha256 = "2".repeat(64);
    assert!(refusal(&m, &t).contains("target_packet_sha256"));

    let mut t = target();
    t.tensors[0].1 = 2048;
    assert!(refusal(&m, &t).contains("import n.hn"));
    let mut t = target();
    t.tensors.remove(0);
    assert!(refusal(&m, &t).contains("import n.hn"));

    let mut t = target();
    t.object_symbols.clear();
    assert!(refusal(&m, &t).contains("object symbol plow_row_slot_1"));

    let mut t = target();
    t.tokenizer = Some(
        TokenizerFacts::from_json(br#"{"model":{"type":"BPE","vocab":{"a":0,"c":1}},"added_tokens":[{"id":2,"content":"<bos>"}]}"#)
            .unwrap(),
    );
    assert!(refusal(&m, &t).contains("tokenizer.vocab_sha256"));
    let mut t = target();
    t.tokenizer = None;
    assert!(refusal(&m, &t).contains("tokenizer"));

    let mut g = manifest();
    g.requires
        .geometry
        .insert("rope_theta_full".into(), "1000000".into());
    assert!(refusal(&g, &target()).contains("geometry.rope_theta_full"));
    let mut t = target();
    t.geometry
        .insert("rope_theta_full".into(), "1000000".into());
    g.check(&t).unwrap();

    let mut tap = manifest();
    tap.requires.taps = vec![5, 17];
    assert!(refusal(&tap, &target()).contains("taps"));
}

#[test]
fn the_target_may_add_tokens_but_not_change_the_drafters() {
    let m = manifest();
    let mut t = target();
    t.tokenizer
        .as_mut()
        .unwrap()
        .added_tokens
        .insert(2, "<eos>".into());
    assert!(refusal(&m, &t).contains("tokenizer.added_tokens[2]"));
}

#[test]
fn identity_fields_need_a_target_that_proves_them() {
    let mut m = manifest();
    m.requires.target_packet_sha256 = None;
    assert!(refusal(&m, &target()).contains("spec.target.json"));
    m.requires.model = Some("google/gemma-4-26b-a4b-it".into());
    m.requires.revision = Some("a".repeat(40));
    let mut t = target();
    t.model = Some("google/gemma-4-26b-a4b-it".into());
    assert!(refusal(&m, &t).contains("revision"));
    t.revision = Some("a".repeat(40));
    m.check(&t).unwrap();
}

#[test]
fn validate_refuses_malformed_manifests() {
    let mut m = manifest();
    m.contract = "plow.spec.drafter.v0".into();
    assert!(m.validate().unwrap_err().contains("contract"));
    let mut m = manifest();
    m.tensors[1].name = m.tensors[0].name.clone();
    assert!(m.validate().unwrap_err().contains("duplicate"));
    let mut m = manifest();
    m.kind = Kind::BlockOwnKv;
    assert!(m.validate().unwrap_err().contains("kind"));
    let mut m = manifest();
    m.requires.target_packet_sha256 = Some("ABC".into());
    assert!(m.validate().is_err());
    let mut m = manifest();
    m.requires.tokenizer = None;
    assert!(m.validate().unwrap_err().contains("tokenizer"));
    let mut m = manifest();
    m.draft.tensors.slot = 99;
    assert!(
        m.validate().is_err(),
        "draft handles index the drafter table"
    );
}

#[test]
fn vocab_digest_ignores_key_order() {
    let a: Value =
        serde_json::from_str(r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1}}}"#).unwrap();
    let b: Value =
        serde_json::from_str(r#"{"model":{"vocab":{"b":1,"a":0},"type":"BPE"}}"#).unwrap();
    assert_eq!(vocab_digest(&a).unwrap(), vocab_digest(&b).unwrap());
    let c: Value =
        serde_json::from_str(r#"{"model":{"type":"BPE","vocab":{"a":1,"b":0}}}"#).unwrap();
    assert_ne!(vocab_digest(&a).unwrap(), vocab_digest(&c).unwrap());
}
