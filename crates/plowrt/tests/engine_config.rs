//! Per-engine resolution of packet serve defaults (`config::resolve_engine`): each co-served
//! bundle gets its own engine knobs, an explicit setting wins for every engine, and the result
//! equals the legacy single-model resolution (defaults exported into the environment).

use std::collections::BTreeMap;
use std::sync::Mutex;

use clap::{Args, FromArgMatches};
use plowrt::config::{resolve_engine, RuntimeConfig, Startup};

/// The tests set process environment variables: one at a time.
static ENV: Mutex<()> = Mutex::new(());

/// The legacy resolution: `env` exported into the environment, then the process parse.
fn legacy(env: &[(&str, &str)], argv: &[&str]) -> RuntimeConfig {
    for (k, v) in env {
        std::env::set_var(k, v);
    }
    let cmd = RuntimeConfig::augment_args(clap::Command::new("plowrt"));
    let m = cmd.try_get_matches_from(argv).unwrap();
    let cfg = RuntimeConfig::from_arg_matches(&m).unwrap();
    for (k, _) in env {
        std::env::remove_var(k);
    }
    cfg
}

fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
    v.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn exported(v: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs(v).into_iter().collect()
}

/// The serve env every production recipe carries, as `(key, value)`.
fn recipe_serve_env() -> Vec<(String, String)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../recipes/infervisor");
    let mut out = Vec::new();
    for model in std::fs::read_dir(root).unwrap().flatten() {
        for f in std::fs::read_dir(model.path()).unwrap().flatten() {
            let p = f.path();
            if p.extension().is_none_or(|e| e != "toml") {
                continue;
            }
            // `[serve.env]` lines `KEY = "value"` / `KEY = 123` (no TOML parser in dev deps).
            let text = std::fs::read_to_string(&p).unwrap();
            let mut inside = false;
            for line in text.lines().map(str::trim) {
                if line.starts_with('[') {
                    inside = line == "[serve.env]";
                    continue;
                }
                let Some((k, v)) = line.split_once('=').filter(|_| inside) else {
                    continue;
                };
                let v = v.split('#').next().unwrap().trim().trim_matches('"');
                out.push((k.trim().to_string(), v.to_string()));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Single-model equivalence, knob by knob: every engine-scoped value a production recipe
/// carries resolves through the per-engine path to exactly the config the legacy export gave.
#[test]
fn every_production_engine_default_resolves_as_the_legacy_export() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let st = Startup::new(Vec::new(), BTreeMap::new());
    let engine: Vec<(String, String)> = recipe_serve_env()
        .into_iter()
        .filter(|(k, _)| {
            plowrt::knob_spec::knob_scope(k) == Some(plowrt::knob_spec::KnobScope::Engine)
        })
        .collect();
    assert!(engine.len() > 20, "{engine:?}");
    for (k, v) in &engine {
        let want = format!("{:?}", legacy(&[(k, v)], &["plowrt"]));
        let got = format!(
            "{:?}",
            resolve_engine(&st, &[(k.clone(), v.clone())]).unwrap()
        );
        assert_eq!(got, want, "{k}={v}");
    }
    // And all of one recipe's at once, as startup exports them, through the exported path.
    let e4b = [("PLOW_VMM_PREFIX", "1"), ("PLOW_LT_RUNG_ALGOS", "true")];
    let g12 = [
        ("PLOW_PREFIX_CACHE", "1"),
        ("PLOW_VMM_PREFIX", "1"),
        ("PLOW_PF_SEG_PURE", "fp8"),
        ("PLOW_PF_SEG_FA512", "1"),
        ("PLOW_PF_INTERLEAVE", "2048"),
        ("PLOW_VMM_LIVE_RINGS", "0"),
        ("PLOW_MULTISTEP", "0"),
        ("PLOW_DECODE_MAX_RUNG", "128"),
        ("PLOW_PF_NORM_ROW_BLOCKS", "1152"),
        ("PLOW_HOST_KVCACHE_SIZE", "64GB"),
    ];
    for set in [&e4b[..], &g12[..]] {
        let want = format!("{:?}", legacy(set, &["plowrt"]));
        for (k, v) in set {
            std::env::set_var(k, v);
        }
        let st = Startup::new(Vec::new(), exported(set));
        let got = format!("{:?}", resolve_engine(&st, &pairs(set)).unwrap());
        for (k, _) in set {
            std::env::remove_var(k);
        }
        assert_eq!(got, want);
    }
}

/// Two co-served bundles with conflicting defaults each get their own; the knob one of them
/// does not carry is the built-in default for it, not the other's exported value.
#[test]
fn co_served_bundles_resolve_their_own_engine_defaults() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let a = [("PLOW_DECODE_MAX_RUNG", "128"), ("PLOW_PF_SEG_PURE", "fp8")];
    let b = [("PLOW_DECODE_MAX_RUNG", "64")];
    // Startup exported A's values (the first bundle wins the environment).
    for (k, v) in &a {
        std::env::set_var(k, v);
    }
    let st = Startup::new(Vec::new(), exported(&a));
    let ca = resolve_engine(&st, &pairs(&a)).unwrap();
    let cb = resolve_engine(&st, &pairs(&b)).unwrap();
    let dynamic = resolve_engine(&st, &[]).unwrap();
    for (k, _) in &a {
        std::env::remove_var(k);
    }
    assert_eq!(ca.decode_max_rung, Some(128));
    assert_eq!(ca.nv.pf_seg_pure.as_deref(), Some("fp8"));
    assert_eq!(cb.decode_max_rung, Some(64));
    assert_eq!(cb.nv.pf_seg_pure, None, "B does not run A's fp8 classing");
    assert_eq!(dynamic.decode_max_rung, None);
}

/// An explicit setting wins for every engine: the operator's environment and the command line.
#[test]
fn an_explicit_setting_overrides_every_bundle() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("PLOW_DECODE_MAX_RUNG", "32");
    let st = Startup::new(Vec::new(), BTreeMap::new());
    let a = resolve_engine(&st, &pairs(&[("PLOW_DECODE_MAX_RUNG", "128")])).unwrap();
    let b = resolve_engine(&st, &pairs(&[("PLOW_DECODE_MAX_RUNG", "64")])).unwrap();
    std::env::remove_var("PLOW_DECODE_MAX_RUNG");
    assert_eq!((a.decode_max_rung, b.decode_max_rung), (Some(32), Some(32)));

    let cli = vec![("decode_max_rung".to_string(), vec!["16".into()])];
    let st = Startup::new(cli, BTreeMap::new());
    let a = resolve_engine(&st, &pairs(&[("PLOW_DECODE_MAX_RUNG", "128")])).unwrap();
    assert_eq!(a.decode_max_rung, Some(16));
    // The command line also beats an environment value, as in the process parse.
    std::env::set_var("PLOW_DECODE_MAX_RUNG", "32");
    let a = resolve_engine(&st, &[]).unwrap();
    std::env::remove_var("PLOW_DECODE_MAX_RUNG");
    assert_eq!(a.decode_max_rung, Some(16));
}

/// Process-scoped knobs are not per engine: a bundle default for one never changes an engine
/// config (the process value stands).
#[test]
fn process_knobs_are_not_resolved_per_engine() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let st = Startup::new(Vec::new(), BTreeMap::new());
    let base = format!("{:?}", resolve_engine(&st, &[]).unwrap());
    let with = resolve_engine(&st, &pairs(&[("PLOW_MAX_QUEUED_REQUESTS", "7")])).unwrap();
    assert_eq!(format!("{with:?}"), base);
}

/// Command-line settings carried into an engine config resolve as the process parse did:
/// valued flags, a bare optional-value bool and a list.
#[test]
fn command_line_settings_resolve_as_the_process_parse() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let argv = [
        "plowrt",
        "--decode-max-rung=8",
        "--pf-seg-pure",
        "fp8",
        "--amd-dense-native",
        "--models-root",
        "/a:/b",
        "--live-ctx=4096",
    ];
    let cmd = RuntimeConfig::augment_args(clap::Command::new("plowrt"));
    let m = cmd.try_get_matches_from(argv).unwrap();
    let want = format!("{:?}", RuntimeConfig::from_arg_matches(&m).unwrap());
    let st = Startup::from_matches(&m, BTreeMap::new());
    let got = format!("{:?}", resolve_engine(&st, &[]).unwrap());
    assert_eq!(got, want);
}

/// Once engine configs diverge, engine code reading the config with no engine scope is a
/// wrong-model read: it panics in a debug build; under a scope (or one its thread inherited) it
/// reads that scope.
#[cfg(feature = "hsa")]
#[test]
fn an_unscoped_engine_read_is_caught_once_engines_diverge() {
    use plowrt::config::{check_engine_scope, inherit, scoped};
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let plan = || plowrt::exec::amd::plan_chunks(&[512, 1024], 1500).map(|v| v.len());
    let before = plan().unwrap();
    check_engine_scope();
    let unscoped = std::panic::catch_unwind(plan);
    let msg = *unscoped.unwrap_err().downcast::<String>().unwrap();
    assert!(
        msg.contains("no engine scope at") && msg.contains("exec/amd.rs"),
        "{msg}"
    );
    let cfg: &'static RuntimeConfig = Box::leak(Box::new(
        resolve_engine(&Startup::new(Vec::new(), BTreeMap::new()), &[]).unwrap(),
    ));
    assert_eq!(scoped(Some(cfg), plan).unwrap(), before);
    let helper = scoped(Some(cfg), || std::thread::spawn(inherit(plan)));
    assert_eq!(helper.join().unwrap().unwrap(), before);
}

/// A minimal device packet whose serve manifest carries `defaults`.
fn packet_with_defaults(name: &str, defaults: &[(&str, &str)]) -> std::path::PathBuf {
    use packet::devbuild::{Model, SectionData};
    let dir = std::env::temp_dir().join(format!("plowrt-engcfg-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut m = plow_asset::serve_manifest::ServeManifest {
        version: 1,
        ..Default::default()
    };
    for (k, v) in defaults {
        m.serve_defaults.insert(k.to_string(), v.to_string());
    }
    let sections = vec![SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: plow_asset::serve_manifest::SECTION.into(),
        data: m.to_json(),
    }];
    let model = Model {
        n_cu: 1,
        target: 0,
        tensors: Vec::new(),
        progs: Vec::new(),
        kv_row_insts: Vec::new(),
        prog_t: Vec::new(),
        gen: Vec::new(),
    };
    std::fs::write(dir.join("model.pkt"), model.to_blob_v6(&sections)).unwrap();
    dir
}

/// Startup engines outside the CUDA manager (AMD, CPU) resolve and record their own bundle's
/// engine config: two bundles asking different engine knobs no longer run the first one's.
#[test]
fn startup_engines_resolve_their_own_bundles_engine_config() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    plowrt::config::record_startup(Startup::new(Vec::new(), BTreeMap::new()));
    let a = packet_with_defaults("a", &[("PLOW_DECODE_MAX_RUNG", "128")]);
    let b = packet_with_defaults("b", &[("PLOW_DECODE_MAX_RUNG", "64")]);
    let backend: std::sync::Arc<dyn plowrt::device::Backend> =
        std::sync::Arc::new(plowrt::device::cpu::CpuBackend::new(1));
    let execset = std::sync::Arc::new(plowrt::exec::ExecutorSet::bringup(backend).unwrap());
    let state = plowrt::serve::AppState::new(plowrt::orch::Registry::new(), execset);
    let ca = state.bundle_engine_config("a", &a).unwrap();
    let cb = state.bundle_engine_config("b", &b).unwrap();
    assert_eq!(
        (ca.decode_max_rung, cb.decode_max_rung),
        (Some(128), Some(64))
    );
    assert!(std::ptr::eq(state.engine_config("b").unwrap(), cb));
    let under_b = plowrt::config::scoped(Some(cb), || RuntimeConfig::get().decode_max_rung);
    assert_eq!(under_b, Some(64), "b's engine load reads b's knobs");
    for d in [a, b] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// A packet emitted before the host KV tier knob's rename carries its qualified MiB value over
/// to `PLOW_HOST_KVCACHE_SIZE`, exactly.
#[test]
fn packet_default_under_the_old_host_cache_name_carries_over() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let dir = packet_with_defaults("renamed", &[("PLOW_VMM_HOST_CACHE_MIB", "65536")]);
    let got =
        plowrt::asset::serve::asset_env_defaults(std::slice::from_ref(&dir), |_| None).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(
        (got[0].0.as_str(), got[0].1.as_str()),
        ("PLOW_HOST_KVCACHE_SIZE", "65536MB")
    );
    let st = Startup::new(Vec::new(), BTreeMap::new());
    let cfg = resolve_engine(&st, &[(got[0].0.clone(), got[0].1.clone())]).unwrap();
    assert_eq!(
        cfg.host_kvcache_size,
        Some(plowrt::config::HostKvcacheSize::Bytes(64 << 30))
    );
    let bad = packet_with_defaults("renamed-bad", &[("PLOW_VMM_HOST_CACHE_MIB", "64GB")]);
    assert!(plowrt::asset::serve::asset_env_defaults(&[bad.clone()], |_| None).is_err());
    let both = packet_with_defaults(
        "renamed-both",
        &[
            ("PLOW_VMM_HOST_CACHE_MIB", "65536"),
            ("PLOW_HOST_KVCACHE_SIZE", "0"),
        ],
    );
    let got = plowrt::asset::serve::asset_env_defaults(&[both.clone()], |_| None).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].1, "0", "the new spelling wins");
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(bad);
    let _ = std::fs::remove_dir_all(both);
}
