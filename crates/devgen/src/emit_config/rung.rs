//! Per-rung overrides of emit knobs (`PLOW_EMIT_RUNG_ENV`).
//!
//! A knob the registry marks [`KnobSpec::per_rung`](plow_asset::knob::KnobSpec::per_rung) may take
//! a different value on one prefill bucket or decode rung than packet-wide. The emitter lowers each
//! rung inside [`with_rung`], so every `emit_config::active()` read there resolves to that rung's
//! config; post-passes that walk finished programs ask [`at`]. Everything else stays packet-wide.
//!
//! Eligible knobs only pick the op or library route of the program being lowered: no tensor,
//! object, program set or checkpoint-K constraint reads them (enforced by the registry tests), so
//! a rung's program depends only on its own resolved values.

use super::EmitConfig;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(not(test))]
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Prefill,
    Decode,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
        }
    }

    fn parse(s: &str) -> Option<Phase> {
        match s {
            "prefill" => Some(Phase::Prefill),
            "decode" => Some(Phase::Decode),
            _ => None,
        }
    }
}

/// A rung by logical rows: a packed AMD sibling and its ordinary program share one key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rung {
    pub phase: Phase,
    pub rows: u32,
}

impl Rung {
    pub fn prefill(rows: u32) -> Rung {
        Rung {
            phase: Phase::Prefill,
            rows,
        }
    }

    pub fn decode(rows: u32) -> Rung {
        Rung {
            phase: Phase::Decode,
            rows,
        }
    }
}

impl std::fmt::Display for Rung {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.phase.name(), self.rows)
    }
}

/// Rung-eligible `EmitConfig` fields. `tests::setter_list_matches_the_registry` holds this list to
/// the registry's `per_rung` entries, so a knob is eligible exactly when knob_spec says so.
macro_rules! rung_fields {
    ($($field:ident: $ty:ty),* $(,)?) => {
        pub const FIELDS: &[&str] = &[$(stringify!($field)),*];

        fn set(cfg: &mut EmitConfig, id: &str, raw: &str) -> Result<(), String> {
            match id {
                $(stringify!($field) => cfg.$field = parse_value::<$ty>(id, raw)?,)*
                _ => return Err(format!("{id} has no per-rung setter")),
            }
            Ok(())
        }

        fn check(id: &str, raw: &str) -> Result<(), String> {
            match id {
                $(stringify!($field) => parse_value::<$ty>(id, raw).map(drop),)*
                _ => Err(format!("{id} has no per-rung setter")),
            }
        }

        fn get(cfg: &EmitConfig, id: &str) -> String {
            match id {
                $(stringify!($field) => cfg.$field.to_string(),)*
                _ => unreachable!("{id} is not a rung field"),
            }
        }
    };
}

rung_fields! {
    no_glu_fuse: bool,
}

/// `raw` through the field's own clap value parser, on an isolated copy of its arg: the same
/// spellings mean the same thing packet-wide and per rung, and no unrelated env var is re-read.
fn parse_value<T: Clone + Send + Sync + 'static>(id: &str, raw: &str) -> Result<T, String> {
    use clap::Args;
    let cmd = EmitConfig::augment_args(clap::Command::new("plowc"));
    let arg = cmd
        .get_arguments()
        .find(|a| a.get_id() == id)
        .ok_or_else(|| format!("{id} is not an EmitConfig arg"))?
        .clone();
    let long = arg
        .get_long()
        .ok_or_else(|| format!("{id} has no --flag"))?
        .to_string();
    let arg = arg.env(None).default_value(None);
    clap::Command::new("rung")
        .arg(arg)
        .try_get_matches_from(["rung".to_string(), format!("--{long}={raw}")])
        .map_err(|e| format!("{id}={raw:?}: {e}"))?
        .get_one::<T>(id)
        .cloned()
        .ok_or_else(|| format!("{id}={raw:?}: no value"))
}

/// Parsed overrides: rung -> EmitConfig field id -> raw value.
pub type Overrides = BTreeMap<Rung, BTreeMap<&'static str, String>>;

/// Parse `phase:rows:ENV=value` entries joined by `;`. Refuses an unknown or packet-wide knob,
/// a phase the knob is not declared for, rows 0, malformed entries and duplicates.
pub fn parse(spec: &str) -> Result<Overrides, String> {
    let mut out = Overrides::new();
    for entry in spec.split(';').map(str::trim).filter(|e| !e.is_empty()) {
        let bad = || format!("rung override {entry:?}: expected phase:rows:ENV=value");
        let (phase, rest) = entry.split_once(':').ok_or_else(bad)?;
        let (rows, assign) = rest.split_once(':').ok_or_else(bad)?;
        let (env, value) = assign.split_once('=').ok_or_else(bad)?;
        let phase = Phase::parse(phase.trim())
            .ok_or_else(|| format!("rung override {entry:?}: phase must be prefill or decode"))?;
        let rows: u32 =
            rows.trim().parse().ok().filter(|&r| r > 0).ok_or_else(|| {
                format!("rung override {entry:?}: rows must be a positive integer")
            })?;
        let env = env.trim();
        let spec = crate::knob_spec::EMIT
            .iter()
            .find(|k| k.env == Some(env))
            .ok_or_else(|| {
                format!("rung override {entry:?}: {env} is not a registered emit knob")
            })?;
        if !spec.rungs.contains(&phase.name()) {
            return Err(format!(
                "rung override {entry:?}: {} ({env}) is packet-wide; knob_spec declares no per-rung \
                 {} override for it (it may shape tensors, objects, the program set or a checked \
                 constraint)",
                spec.id,
                phase.name()
            ));
        }
        let id = spec.name();
        let value = value.trim().to_string();
        // Validate the spelling now, not at the first lowering that reads it.
        check(id, &value)?;
        let rung = Rung { phase, rows };
        if out.entry(rung).or_default().insert(id, value).is_some() {
            return Err(format!(
                "rung override {entry:?}: {env} given twice for {rung}"
            ));
        }
    }
    Ok(out)
}

struct State {
    overrides: Overrides,
    cfgs: BTreeMap<Rung, &'static EmitConfig>,
    consumed: BTreeSet<Rung>,
}

#[cfg(not(test))]
static STATE: Mutex<Option<State>> = Mutex::new(None);
#[cfg(test)]
thread_local! {
    static TEST_STATE: std::cell::RefCell<Option<State>> = const { std::cell::RefCell::new(None) };
}

thread_local! {
    static CURRENT: Cell<Option<&'static EmitConfig>> = const { Cell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut Option<State>) -> R) -> R {
    #[cfg(not(test))]
    return f(&mut STATE.lock().unwrap_or_else(|e| e.into_inner()));
    #[cfg(test)]
    return TEST_STATE.with_borrow_mut(f);
}

/// Build every overridden rung's config from the installed packet-wide one. Called by
/// [`super::install`]; replaces any previous emit's state.
pub(super) fn install(base: &EmitConfig) {
    let overrides = match base.rung_env.as_deref().map(parse).transpose() {
        Ok(o) => o.unwrap_or_default(),
        Err(e) => panic!("PLOW_EMIT_RUNG_ENV: {e}"),
    };
    let mut cfgs = BTreeMap::new();
    for (rung, knobs) in &overrides {
        let mut cfg = base.clone();
        for (id, raw) in knobs {
            set(&mut cfg, id, raw).unwrap_or_else(|e| panic!("PLOW_EMIT_RUNG_ENV {rung}: {e}"));
        }
        cfg.validate();
        let cfg: &'static EmitConfig = Box::leak(Box::new(cfg));
        cfgs.insert(*rung, cfg);
    }
    with_state(|s| {
        *s = Some(State {
            overrides,
            cfgs,
            consumed: BTreeSet::new(),
        })
    });
    CURRENT.with(|c| c.set(None));
}

/// The config of the rung being lowered on this thread, if any.
pub(super) fn current() -> Option<&'static EmitConfig> {
    CURRENT.with(Cell::get)
}

/// The resolved config for `rung`: its override, else the packet-wide config.
pub fn at(rung: Rung) -> &'static EmitConfig {
    with_state(|s| s.as_ref().and_then(|s| s.cfgs.get(&rung).copied()))
        .unwrap_or_else(super::packet_wide)
}

/// Restores the enclosing scope on drop, also on unwind.
struct Scope(Option<&'static EmitConfig>);

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.0));
    }
}

/// Lower `rung` under its resolved config and record that its overrides were consumed.
pub fn with_rung<R>(rung: Rung, f: impl FnOnce() -> R) -> R {
    let cfg = with_state(|s| {
        let s = s.as_mut()?;
        let cfg = s.cfgs.get(&rung).copied()?;
        s.consumed.insert(rung);
        Some(cfg)
    })
    .unwrap_or_else(super::packet_wide);
    let _scope = Scope(CURRENT.with(|c| c.replace(Some(cfg))));
    f()
}

/// Every override that named a rung no emitter lowered: a typo'd width, or a family whose
/// lowering never enters [`with_rung`]. Checked before the blob is written.
pub fn unconsumed() -> Vec<Rung> {
    with_state(|s| {
        s.as_ref().map_or_else(Vec::new, |s| {
            s.overrides
                .keys()
                .filter(|r| !s.consumed.contains(r))
                .copied()
                .collect()
        })
    })
}

/// `build.json` `rung_knobs`: every emitted program's resolved value of every rung-eligible knob,
/// and whether the value is the packet-wide one or a rung override.
pub fn manifest_section(prog_t: &[u32]) -> serde_json::Value {
    use serde_json::json;
    let lo = packet::devbuild::decode_rung_lo(prog_t);
    let overrides = with_state(|s| s.as_ref().map(|s| s.overrides.clone()).unwrap_or_default());
    let base = super::packet_wide();
    let programs: Vec<_> = prog_t
        .iter()
        .enumerate()
        .map(|(index, &t)| {
            let rows = packet::devbuild::program_rows(t);
            let rung = if index < lo {
                Rung::prefill(rows)
            } else {
                Rung::decode(rows)
            };
            let cfg = at(rung);
            let mine = overrides.get(&rung);
            let knobs: serde_json::Map<_, _> = FIELDS
                .iter()
                .map(|id| {
                    let source = if mine.is_some_and(|m| m.contains_key(id)) {
                        "rung"
                    } else {
                        "packet"
                    };
                    (
                        format!("emit.{id}"),
                        json!({"value": get(cfg, id), "packet": get(base, id), "source": source}),
                    )
                })
                .collect();
            json!({
                "program": index,
                "phase": rung.phase.name(),
                "rows": rows,
                "packed": packet::devbuild::is_packed_prefill_program(t),
                "knobs": knobs,
            })
        })
        .collect();
    json!({
        "schema": 1,
        "note": "per-program resolution of the rung-eligible emit knobs (PLOW_EMIT_RUNG_ENV over the \
                 packet-wide value). Checkpoint K verified the packet-wide values; rung-eligible knobs \
                 appear in no registered constraint, so an override cannot bypass one.",
        "eligible": FIELDS.iter().map(|id| format!("emit.{id}")).collect::<Vec<_>>(),
        "overrides": overrides
            .iter()
            .map(|(r, m)| json!({"phase": r.phase.name(), "rows": r.rows, "knobs": m}))
            .collect::<Vec<_>>(),
        "programs": programs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_refuses() {
        let o =
            parse("prefill:128:PLOW_NO_GLU_FUSE=0; prefill:4096:PLOW_NO_GLU_FUSE=true").unwrap();
        assert_eq!(o[&Rung::prefill(128)]["no_glu_fuse"], "0");
        assert_eq!(o[&Rung::prefill(4096)]["no_glu_fuse"], "true");
        assert!(parse("").unwrap().is_empty());
        for (bad, why) in [
            ("prefill:128:PLOW_FP8_KV=1", "packet-wide"),
            ("prefill:128:PLOW_QNORM_FUSE=1", "packet-wide"),
            ("decode:8:PLOW_NO_GLU_FUSE=1", "packet-wide"),
            ("prefill:128:PLOW_NOT_A_KNOB=1", "not a registered"),
            ("prefill:0:PLOW_NO_GLU_FUSE=1", "positive"),
            ("prefill:x:PLOW_NO_GLU_FUSE=1", "positive"),
            ("tail:128:PLOW_NO_GLU_FUSE=1", "phase"),
            ("prefill:128:PLOW_NO_GLU_FUSE", "expected"),
            ("prefill:128:PLOW_NO_GLU_FUSE=maybe", "no_glu_fuse"),
            (
                "prefill:128:PLOW_NO_GLU_FUSE=1;prefill:128:PLOW_NO_GLU_FUSE=0",
                "twice",
            ),
        ] {
            let e = parse(bad).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
    }

    #[test]
    fn values_parse_like_the_packet_wide_flag() {
        for (raw, want) in [
            ("1", true),
            ("true", true),
            ("on", true),
            ("0", false),
            ("false", false),
            ("off", false),
        ] {
            assert_eq!(
                parse_value::<bool>("no_glu_fuse", raw).unwrap(),
                want,
                "{raw}"
            );
        }
    }

    #[test]
    fn setter_list_matches_the_registry() {
        let declared: BTreeSet<&str> = crate::knob_spec::EMIT
            .iter()
            .filter(|k| !k.rungs.is_empty())
            .map(|k| k.name())
            .collect();
        assert_eq!(declared, FIELDS.iter().copied().collect());
        for k in crate::knob_spec::EMIT
            .iter()
            .filter(|k| !k.rungs.is_empty())
        {
            assert!(
                k.rungs.iter().all(|p| Phase::parse(p).is_some()),
                "{}: {:?}",
                k.id,
                k.rungs
            );
        }
    }

    /// Checkpoint K resolves packet-wide values only. A per-rung knob that a constraint or another
    /// knob's production default read could take a value K never checked.
    #[test]
    fn rung_knobs_are_read_by_no_constraint_or_default() {
        let per_rung: Vec<&str> = crate::knob_spec::EMIT
            .iter()
            .filter(|k| !k.rungs.is_empty())
            .map(|k| k.id)
            .collect();
        let all = crate::knob_spec::EMIT
            .iter()
            .chain(crate::knob_spec::RAW_ENV)
            .chain(crate::knob_spec::OBJECT_DEFINES);
        let mut constraint_vars = Vec::new();
        for k in all.clone() {
            for c in k.constraints {
                c.formula.vars(&mut constraint_vars);
            }
        }
        for k in plow_asset::knob_gen::CONSTRAINTS {
            k.formula.vars(&mut constraint_vars);
        }
        for id in &per_rung {
            assert!(
                !constraint_vars.contains(id),
                "{id} is per-rung but a constraint reads it"
            );
        }
        for k in all {
            let vars = k.default.vars();
            for id in &per_rung {
                assert!(
                    *id == k.id || !vars.contains(id),
                    "{id} is per-rung but {}'s production default reads it",
                    k.id
                );
            }
        }
    }

    /// Consecutive installs in one process: the second clears the first's overrides and
    /// consumption, and a rung scope resolves, nests and unwinds back to the packet-wide config.
    #[test]
    fn install_resets_and_scopes_nest() {
        let mut cfg = EmitConfig::from_env();
        cfg.no_glu_fuse = true;
        cfg.rung_env = Some("prefill:128:PLOW_NO_GLU_FUSE=0".into());
        install(&cfg);
        assert_eq!(unconsumed(), [Rung::prefill(128)]);
        assert!(!at(Rung::prefill(128)).no_glu_fuse);
        with_rung(Rung::prefill(128), || {
            assert!(!current().unwrap().no_glu_fuse);
            with_rung(Rung::prefill(512), || assert!(current().is_some()));
            assert!(!current().unwrap().no_glu_fuse);
            let r = std::panic::catch_unwind(|| {
                with_rung(Rung::prefill(512), || panic!("lowering failed"))
            });
            assert!(r.is_err());
            assert!(!current().unwrap().no_glu_fuse);
        });
        assert!(current().is_none());
        assert!(unconsumed().is_empty());
        cfg.rung_env = None;
        install(&cfg);
        assert!(unconsumed().is_empty());
        assert!(with_state(|s| s.as_ref().unwrap().cfgs.is_empty()));
    }

    #[test]
    fn scope_restores_on_unwind() {
        let cfg: &'static EmitConfig = Box::leak(Box::new(EmitConfig::from_env()));
        CURRENT.with(|c| c.set(None));
        let r = std::panic::catch_unwind(|| {
            let _scope = Scope(CURRENT.with(|c| c.replace(Some(cfg))));
            panic!("lowering failed");
        });
        assert!(r.is_err());
        assert!(current().is_none());
    }
}
