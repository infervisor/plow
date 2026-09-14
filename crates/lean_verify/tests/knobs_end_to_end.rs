//! Checkpoint K against `plow_verify`, over the generated registry subset
//! (`plow_asset::knob_gen`). Needs the binary, like `end_to_end.rs`.

use lean_verify::checkpoints::knobs::{check_knobs, verdicts};
use plow_asset::knob::{
    lookup, registry_json, resolve, sources_json, well_formed, Constraint, Domain, KnobSpec,
    Source, Target, Val,
};
use plow_asset::knob_gen::{holds, CONSTRAINTS, KNOBS, TARGETS};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;
use serde_json::{json, Value};

fn registry() -> Value {
    let specs: Vec<&KnobSpec> = KNOBS.iter().collect();
    let constraints: Vec<&Constraint> = CONSTRAINTS.iter().collect();
    registry_json(&specs, &constraints, TARGETS)
}

fn request(target: &Target, sources: &[(&'static str, Val<'static>)]) -> Value {
    let mut p = registry();
    p["target"] = target.to_json();
    p["sources"] = sources_json(
        sources
            .iter()
            .filter(|(id, _)| KNOBS.iter().any(|k| k.id == *id))
            .map(|(id, v)| {
                (
                    *id,
                    Source {
                        cli: None,
                        env: Some(*v),
                    },
                )
            }),
    );
    p
}

fn glm_with(extra: &[(&'static str, Val<'static>)]) -> Vec<(&'static str, Val<'static>)> {
    let mut r = extra.to_vec();
    r.extend(TARGETS[0].recipe.iter().copied());
    r
}

#[test]
#[ignore = "requires plow_verify binary"]
fn production_recipe_and_registry_are_accepted() {
    let cert = check_knobs(&request(&TARGETS[0].target(), &glm_with(&[]))).unwrap();
    assert!(cert.ok, "{cert:?}");
    assert!(
        cert.notes.unwrap().contains("declared targets consistent"),
        "registry consistency did not run"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn negative_fixtures_are_rejected() {
    const TRUE: Val = Val::Bool(true);
    let glm = TARGETS[0].target();
    let cases: &[(&[(&str, Val)], &str)] = &[
        (
            &[
                ("emit.glm_seq_par", TRUE),
                ("emit.glm_xr_band", Val::Nat(2)),
            ],
            "seq_par_excludes_two_shot_seams",
        ),
        (
            &[("emit.glm_ofold", TRUE)],
            "ofold_excludes_dsa_pf_and_fp8_kv",
        ),
        (
            &[("emit.token_batch_tp", TRUE)],
            "token_batch_tp_excludes_seq_par",
        ),
    ];
    for (extra, id) in cases {
        let cert = check_knobs(&request(&glm, &glm_with(extra))).unwrap();
        assert!(
            !cert.ok && cert.reason.as_deref().unwrap_or("").contains(id),
            "{id}: {cert:?}"
        );
    }
    let mut pooled = glm.clone();
    pooled.caps.push("indexer_pooled".into());
    let cert = check_knobs(&request(
        &pooled,
        &glm_with(&[("emit.packed_sparse_pf", TRUE)]),
    ))
    .unwrap();
    assert!(
        !cert.ok && cert.reason.unwrap().contains("packed_sparse_pf_contract"),
        "pooled indexer"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn stale_registry_defaults_are_rejected() {
    let mut p = request(&TARGETS[0].target(), &glm_with(&[]));
    p["recorded"] = json!([{"id": "emit.glm_seq_par", "value": false}]);
    let cert = check_knobs(&p).unwrap();
    assert!(!cert.ok && cert.reason.unwrap().contains("resolves differently"));
}

#[test]
#[ignore = "requires plow_verify binary"]
fn contradictory_constraints_are_rejected() {
    let spec = |id: &str| {
        json!({"id": id, "layer": "emit", "domain": {"kind": "bool"},
               "default": {"static": false}, "status": {"kind": "opt_in"}})
    };
    let atom = |id: &str| json!(["atom", id, "eq", true]);
    let t = json!({"name": "t", "arch": "x", "tp": 1, "n_cu": 1, "model": "m", "caps": [], "recipe": []});
    let p = json!({
        "registry": [spec("emit.k"), spec("emit.a")],
        "constraints": [
            {"id": "k_needs_a", "formula": ["implies", atom("emit.k"), atom("emit.a")]},
            {"id": "k_forbids_a", "formula": ["implies", atom("emit.k"), ["not", atom("emit.a")]]},
        ],
        "targets": [t],
        "target": t,
        "sources": [],
    });
    let cert = check_knobs(&p).unwrap();
    assert!(!cert.ok && cert.reason.unwrap().contains("emit.k=true contradicts"));
}

const STRS: &[&str] = &[
    "1",
    "0",
    "",
    "true",
    "x",
    "o_proj,band,shared",
    "1,2,4,8",
    "1,2,4,8,16,20",
    "full:128,512,2048,8192",
];

fn value(k: &KnobSpec) -> BoxedStrategy<Val<'static>> {
    let typed = match k.domain {
        Domain::Bool => any::<bool>().prop_map(Val::Bool).boxed(),
        Domain::Nat { .. } => prop_oneof![0u64..=20, Just(304u64), Just(u32::MAX as u64)]
            .prop_map(Val::Nat)
            .boxed(),
        _ => proptest::sample::select(STRS.to_vec())
            .prop_map(Val::Str)
            .boxed(),
    };
    let odd = prop_oneof![
        Just(Val::Str("x")),
        Just(Val::Nat(1 << 40)),
        Just(Val::Unset),
        Just(Val::Bool(true))
    ];
    prop_oneof![19 => typed, 1 => odd].boxed()
}

fn source(k: &KnobSpec) -> BoxedStrategy<Source<'static>> {
    (
        proptest::option::weighted(0.15, value(k)),
        proptest::option::weighted(0.3, value(k)),
    )
        .prop_map(|(cli, env)| Source { cli, env })
        .boxed()
}

fn targets() -> Vec<Target> {
    let mut ts: Vec<Target> = TARGETS.iter().map(|t| t.target()).collect();
    for t in TARGETS {
        let mut extra = t.target();
        extra
            .caps
            .extend(["indexer_pooled", "segmented", "bundled_segment_pair"].map(String::from));
        ts.push(extra);
    }
    ts
}

fn rust_verdict(t: &Target, sources: &[Source<'static>]) -> String {
    let specs: Vec<&KnobSpec> = KNOBS.iter().collect();
    let src = |id: &str| {
        KNOBS
            .iter()
            .position(|k| k.id == id)
            .map_or(Source::default(), |i| sources[i])
    };
    if !well_formed(&specs, &src) {
        return "wf".into();
    }
    let config = resolve(&specs, &src, t);
    let vals: Vec<Val> = KNOBS.iter().map(|k| lookup(&config, k.id)).collect();
    (0..CONSTRAINTS.len())
        .find(|&i| !holds(i, &vals, t))
        .map_or("ok".into(), |i| CONSTRAINTS[i].id.into())
}

/// The generated Rust evaluator and `plow_verify` agree on 10k random sources × targets.
#[test]
#[ignore = "requires plow_verify binary"]
fn generated_evaluator_agrees_with_lean_on_random_configs() {
    let ts = targets();
    let strategy = (0..ts.len(), KNOBS.iter().map(source).collect::<Vec<_>>());
    let mut runner = TestRunner::deterministic();
    let cases: Vec<(usize, Vec<Source<'static>>)> = (0..10_000)
        .map(|_| strategy.new_tree(&mut runner).unwrap().current())
        .collect();
    let mut outcomes = std::collections::BTreeMap::<String, usize>::new();
    for chunk in cases.chunks(1000) {
        let mut p = registry();
        p["cases"] = Value::Array(
            chunk
                .iter()
                .map(|(t, srcs)| {
                    json!({
                        "target": ts[*t].to_json(),
                        "sources": sources_json(KNOBS.iter().map(|k| k.id).zip(srcs.iter().copied())),
                    })
                })
                .collect(),
        );
        let lean = verdicts(&p).unwrap();
        assert_eq!(lean.len(), chunk.len());
        for ((t, srcs), l) in chunk.iter().zip(&lean) {
            let r = rust_verdict(&ts[*t], srcs);
            assert_eq!(&r, l, "target {} sources {srcs:?}", ts[*t].name);
            *outcomes.entry(r).or_default() += 1;
        }
    }
    eprintln!("differential outcomes: {outcomes:?}");
    assert!(
        outcomes.len() > 3,
        "the random cases exercised too few outcomes: {outcomes:?}"
    );
}
