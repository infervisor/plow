//! The knob registry: one `KnobSpec` per emit knob, runtime knob, object define and raw env read,
//! and the cross-knob constraints over them.
//!
//! This is the Rust mirror of `lean-plow/Plow/Knobs`. `plowc` sends [`payload`] to `plow_verify`
//! checkpoint K at emit; `plowrt` evaluates the same constraints at load through the generated
//! [`crate::knob_gen::violation`]. [`resolve`] and [`Formula::eval`] follow the Lean definitions
//! clause for clause so a differential test can hold them to each other.
//!
//! Ids are layer-qualified — `emit.fp8`, `rt.mla_pf_v2`, `env.PLOW_BLOCK`, `def.PLOW_GLM_OFOLD` —
//! so one registry spans every layer and one env var read by two layers stays two knobs.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Emit,
    Runtime,
    ObjectDefine,
    RawEnv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Val<'a> {
    Unset,
    Bool(bool),
    Nat(u64),
    Str(&'a str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Bool,
    Nat { min: u64, max: u64 },
    Enum(&'static [&'static str]),
    List(&'static [&'static str]),
    Str,
}

pub const U32: Domain = Domain::Nat {
    min: 0,
    max: u32::MAX as u64,
};
pub const USIZE: Domain = Domain::Nat {
    min: 0,
    max: u64::MAX,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetAtom {
    Arch(&'static str),
    Tp(u32),
    NCu(u32),
    Model(&'static str),
    Cap(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Formula {
    True,
    Atom(&'static str, Cmp, Val<'static>),
    Target(TargetAtom),
    Not(&'static Formula),
    And(&'static [Formula]),
    Or(&'static [Formula]),
    Implies(&'static Formula, &'static Formula),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefaultCase {
    pub when: Formula,
    pub value: Val<'static>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Default {
    Static(Val<'static>),
    /// The first case whose `when` holds, else `otherwise`. A `when` may read only knobs declared
    /// earlier in the registry.
    Production {
        cases: &'static [DefaultCase],
        otherwise: Val<'static>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Qualified {
        evidence: &'static [&'static str],
    },
    OptIn,
    /// Opt-in, measured, and awaiting the gate that would flip its default.
    Candidate {
        evidence: &'static [&'static str],
    },
    Parked {
        reason: &'static str,
        evidence: &'static [&'static str],
    },
    Diagnostic,
    Removed,
}

/// Where a constraint is enforced outside `plowc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    /// Every var is known from `build.json` and the runtime config: plowrt's load evaluator.
    Load,
    /// It reads a fact only the asserting site has (a bundled object pair); the assert at `site`
    /// enforces it, and checkpoint K still checks it for consistency.
    Site,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub id: &'static str,
    pub formula: Formula,
    /// The assert this encodes, as `file: message`.
    pub site: &'static str,
    pub check: Check,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnobSpec {
    pub id: &'static str,
    pub env: Option<&'static str>,
    pub layer: Layer,
    pub domain: Domain,
    pub default: Default,
    pub status: Status,
    pub constraints: &'static [Constraint],
}

impl KnobSpec {
    pub const fn new(
        id: &'static str,
        env: Option<&'static str>,
        layer: Layer,
        domain: Domain,
        default: Default,
        status: Status,
    ) -> KnobSpec {
        KnobSpec {
            id,
            env,
            layer,
            domain,
            default,
            status,
            constraints: &[],
        }
    }

    pub const fn with(self, constraints: &'static [Constraint]) -> KnobSpec {
        KnobSpec {
            constraints,
            ..self
        }
    }

    /// The id without its layer prefix: the clap id, env var or define name.
    pub fn name(&self) -> &'static str {
        self.id.split_once('.').map_or(self.id, |(_, n)| n)
    }
}

/// A declared target and the recipe it is qualified with.
#[derive(Clone, Copy, Debug)]
pub struct TargetSpec {
    pub name: &'static str,
    pub arch: &'static str,
    pub tp: u32,
    pub n_cu: u32,
    pub model: &'static str,
    pub caps: &'static [&'static str],
    pub recipe: &'static [(&'static str, Val<'static>)],
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Target {
    pub name: String,
    pub arch: String,
    pub tp: u32,
    pub n_cu: u32,
    pub model: String,
    pub caps: Vec<String>,
}

impl TargetSpec {
    pub fn target(&self) -> Target {
        Target {
            name: self.name.into(),
            arch: self.arch.into(),
            tp: self.tp,
            n_cu: self.n_cu,
            model: self.model.into(),
            caps: self.caps.iter().map(|c| c.to_string()).collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Source<'a> {
    pub cli: Option<Val<'a>>,
    pub env: Option<Val<'a>>,
}

impl Domain {
    pub fn admits(&self, v: Val) -> bool {
        match (self, v) {
            (_, Val::Unset) => true,
            (Domain::Bool, Val::Bool(_)) => true,
            (Domain::Nat { min, max }, Val::Nat(n)) => *min <= n && n <= *max,
            (Domain::Enum(vs), Val::Str(s)) => vs.contains(&s),
            (Domain::List(vs), Val::Str(s)) => s.split(',').all(|x| vs.contains(&x)),
            (Domain::Str, Val::Str(_)) => true,
            _ => false,
        }
    }

    /// Type a recorded or environment string. `None` for text this domain cannot carry, which
    /// the caller reports as out of domain.
    pub fn parse<'a>(&self, s: &'a str) -> Option<Val<'a>> {
        match self {
            Domain::Bool => match s {
                "1" | "true" | "True" | "TRUE" | "yes" | "y" | "on" | "t" => Some(Val::Bool(true)),
                "0" | "false" | "False" | "FALSE" | "no" | "n" | "off" | "f" => {
                    Some(Val::Bool(false))
                }
                _ => None,
            },
            Domain::Nat { .. } => s.parse().ok().map(Val::Nat),
            _ => Some(Val::Str(s)),
        }
    }
}

pub fn cmp(op: Cmp, a: Val, b: Val) -> bool {
    match (op, a, b) {
        (Cmp::Eq, a, b) => a == b,
        (Cmp::Ne, a, b) => a != b,
        (Cmp::Lt, Val::Nat(a), Val::Nat(b)) => a < b,
        (Cmp::Le, Val::Nat(a), Val::Nat(b)) => a <= b,
        (Cmp::Gt, Val::Nat(a), Val::Nat(b)) => a > b,
        (Cmp::Ge, Val::Nat(a), Val::Nat(b)) => a >= b,
        _ => false,
    }
}

impl TargetAtom {
    pub fn eval(&self, t: &Target) -> bool {
        match *self {
            TargetAtom::Arch(s) => t.arch == s,
            TargetAtom::Tp(n) => t.tp == n,
            TargetAtom::NCu(n) => t.n_cu == n,
            TargetAtom::Model(s) => t.model == s,
            TargetAtom::Cap(s) => t.caps.iter().any(|c| c == s),
        }
    }
}

impl Formula {
    pub fn eval<'a>(&self, get: &impl Fn(&str) -> Val<'a>, t: &Target) -> bool {
        match self {
            Formula::True => true,
            Formula::Atom(k, op, v) => cmp(*op, get(k), *v),
            Formula::Target(a) => a.eval(t),
            Formula::Not(f) => !f.eval(get, t),
            Formula::And(fs) => fs.iter().all(|f| f.eval(get, t)),
            Formula::Or(fs) => fs.iter().any(|f| f.eval(get, t)),
            Formula::Implies(a, b) => !a.eval(get, t) || b.eval(get, t),
        }
    }

    pub fn vars(&self, out: &mut Vec<&'static str>) {
        match self {
            Formula::Atom(k, _, _) => out.push(k),
            Formula::Not(f) => f.vars(out),
            Formula::And(fs) | Formula::Or(fs) => fs.iter().for_each(|f| f.vars(out)),
            Formula::Implies(a, b) => {
                a.vars(out);
                b.vars(out);
            }
            Formula::True | Formula::Target(_) => {}
        }
    }
}

impl Default {
    pub fn vars(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if let Default::Production { cases, .. } = self {
            cases.iter().for_each(|c| c.when.vars(&mut out));
        }
        out
    }
}

/// A resolved config: `(id, value)` in registry order.
pub type Config<'a> = Vec<(&'static str, Val<'a>)>;

pub fn lookup<'a>(c: &[(&'static str, Val<'a>)], id: &str) -> Val<'a> {
    c.iter()
        .find(|(k, _)| *k == id)
        .map_or(Val::Unset, |(_, v)| *v)
}

/// `Plow.Knobs.resolve`: cli > env > production default > static, left to right, each production
/// default evaluated over the knobs resolved before it.
pub fn resolve<'a>(
    specs: &[&KnobSpec],
    src: &impl Fn(&str) -> Source<'a>,
    t: &Target,
) -> Config<'a> {
    let mut c: Config<'a> = Vec::with_capacity(specs.len());
    for k in specs {
        let s = src(k.id);
        let v = match (s.cli, s.env, k.default) {
            (Some(v), _, _) => v,
            (None, Some(v), _) => v,
            (None, None, Default::Static(v)) => v,
            (None, None, Default::Production { cases, otherwise }) => cases
                .iter()
                .find(|d| d.when.eval(&|x| lookup(&c, x), t))
                .map_or(otherwise, |d| d.value),
        };
        c.push((k.id, v));
    }
    c
}

/// `Plow.Knobs.ordered ∧ defaultsInDomain ∧ sourcesInDomain`.
pub fn well_formed<'a>(specs: &[&KnobSpec], src: &impl Fn(&str) -> Source<'a>) -> bool {
    let mut seen: Vec<&str> = Vec::with_capacity(specs.len());
    for k in specs {
        if seen.contains(&k.id) || !k.default.vars().iter().all(|x| seen.contains(x)) {
            return false;
        }
        seen.push(k.id);
    }
    let defaults_ok = specs.iter().all(|k| match k.default {
        Default::Static(v) => k.domain.admits(v),
        Default::Production { cases, otherwise } => {
            k.domain.admits(otherwise) && cases.iter().all(|c| k.domain.admits(c.value))
        }
    });
    defaults_ok
        && specs.iter().all(|k| {
            let s = src(k.id);
            s.cli.is_none_or(|v| k.domain.admits(v)) && s.env.is_none_or(|v| k.domain.admits(v))
        })
}

/// `Plow.Knobs.verdict`: `"ok"`, `"wf"`, or the id of the first violated constraint.
pub fn verdict<'a>(
    specs: &[&KnobSpec],
    constraints: &[&Constraint],
    src: &impl Fn(&str) -> Source<'a>,
    t: &Target,
) -> &'static str {
    if !well_formed(specs, src) {
        return "wf";
    }
    let c = resolve(specs, src, t);
    constraints
        .iter()
        .find(|x| !x.formula.eval(&|id| lookup(&c, id), t))
        .map_or("ok", |x| x.id)
}

// ── JSON (the `plow_verify` K payload) ───────────────────────────────────────────────────────────

impl Val<'_> {
    pub fn to_json(&self) -> Value {
        match *self {
            Val::Unset => Value::Null,
            Val::Bool(b) => json!(b),
            Val::Nat(n) => json!(n),
            Val::Str(s) => json!(s),
        }
    }
}

impl Domain {
    pub fn to_json(&self) -> Value {
        match self {
            Domain::Bool => json!({"kind": "bool"}),
            Domain::Nat { min, max } => json!({"kind": "nat", "min": min, "max": max}),
            Domain::Enum(vs) => json!({"kind": "enum", "values": vs}),
            Domain::List(vs) => json!({"kind": "list", "values": vs}),
            Domain::Str => json!({"kind": "str"}),
        }
    }
}

impl Formula {
    pub fn to_json(&self) -> Value {
        let cmp = |c: &Cmp| match c {
            Cmp::Eq => "eq",
            Cmp::Ne => "ne",
            Cmp::Lt => "lt",
            Cmp::Le => "le",
            Cmp::Gt => "gt",
            Cmp::Ge => "ge",
        };
        match self {
            Formula::True => json!(["true"]),
            Formula::Atom(k, c, v) => json!(["atom", k, cmp(c), v.to_json()]),
            Formula::Target(TargetAtom::Arch(s)) => json!(["arch", s]),
            Formula::Target(TargetAtom::Tp(n)) => json!(["tp", n]),
            Formula::Target(TargetAtom::NCu(n)) => json!(["n_cu", n]),
            Formula::Target(TargetAtom::Model(s)) => json!(["model", s]),
            Formula::Target(TargetAtom::Cap(s)) => json!(["cap", s]),
            Formula::Not(f) => json!(["not", f.to_json()]),
            Formula::And(fs) => json!(["and", fs.iter().map(Formula::to_json).collect::<Vec<_>>()]),
            Formula::Or(fs) => json!(["or", fs.iter().map(Formula::to_json).collect::<Vec<_>>()]),
            Formula::Implies(a, b) => json!(["implies", a.to_json(), b.to_json()]),
        }
    }
}

impl KnobSpec {
    pub fn to_json(&self) -> Value {
        let default = match self.default {
            Default::Static(v) => json!({"static": v.to_json()}),
            Default::Production { cases, otherwise } => json!({
                "production": cases.iter().map(|c| json!({"when": c.when.to_json(), "value": c.value.to_json()})).collect::<Vec<_>>(),
                "otherwise": otherwise.to_json(),
            }),
        };
        let status = match self.status {
            Status::Qualified { evidence } => json!({"kind": "qualified", "evidence": evidence}),
            Status::OptIn => json!({"kind": "opt_in"}),
            Status::Candidate { evidence } => json!({"kind": "candidate", "evidence": evidence}),
            Status::Parked { reason, evidence } => {
                json!({"kind": "parked", "reason": reason, "evidence": evidence})
            }
            Status::Diagnostic => json!({"kind": "diagnostic"}),
            Status::Removed => json!({"kind": "removed"}),
        };
        let layer = match self.layer {
            Layer::Emit => "emit",
            Layer::Runtime => "runtime",
            Layer::ObjectDefine => "object_define",
            Layer::RawEnv => "raw_env",
        };
        json!({
            "id": self.id,
            "env": self.env,
            "layer": layer,
            "domain": self.domain.to_json(),
            "default": default,
            "status": status,
        })
    }
}

impl Constraint {
    pub fn to_json(&self) -> Value {
        json!({"id": self.id, "formula": self.formula.to_json(), "site": self.site})
    }
}

impl Target {
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "arch": self.arch,
            "tp": self.tp,
            "n_cu": self.n_cu,
            "model": self.model,
            "caps": self.caps,
        })
    }

    pub fn from_json(v: &Value) -> Option<Target> {
        Some(Target {
            name: v.get("name")?.as_str()?.into(),
            arch: v.get("arch")?.as_str()?.into(),
            tp: v.get("tp")?.as_u64()? as u32,
            n_cu: v.get("n_cu")?.as_u64()? as u32,
            model: v.get("model")?.as_str()?.into(),
            caps: v
                .get("caps")?
                .as_array()?
                .iter()
                .filter_map(|c| c.as_str().map(String::from))
                .collect(),
        })
    }
}

pub fn sources_json<'a>(sources: impl IntoIterator<Item = (&'a str, Source<'a>)>) -> Value {
    Value::Array(
        sources
            .into_iter()
            .filter(|(_, s)| s.cli.is_some() || s.env.is_some())
            .map(|(id, s)| {
                let mut o = json!({"id": id});
                if let Some(v) = s.cli {
                    o["cli"] = v.to_json();
                }
                if let Some(v) = s.env {
                    o["env"] = v.to_json();
                }
                o
            })
            .collect(),
    )
}

/// The registry half of a K payload: specs, constraints, and declared targets with their recipes.
pub fn registry_json(
    specs: &[&KnobSpec],
    constraints: &[&Constraint],
    targets: &[TargetSpec],
) -> Value {
    json!({
        "registry": specs.iter().map(|k| k.to_json()).collect::<Vec<_>>(),
        "constraints": constraints.iter().map(|c| c.to_json()).collect::<Vec<_>>(),
        "targets": targets.iter().map(|t| {
            let mut o = t.target().to_json();
            o["recipe"] = sources_json(t.recipe.iter().map(|(id, v)| (*id, Source { cli: None, env: Some(*v) })));
            o
        }).collect::<Vec<_>>(),
    })
}

/// sha256 over the canonical registry JSON: specs, constraints, targets.
pub fn registry_digest(
    specs: &[&KnobSpec],
    constraints: &[&Constraint],
    targets: &[TargetSpec],
) -> String {
    let bytes = serde_json::to_vec(&registry_json(specs, constraints, targets)).unwrap_or_default();
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Every constraint in `specs`, in registry order.
pub fn constraints_of<'s>(specs: impl IntoIterator<Item = &'s KnobSpec>) -> Vec<&'s Constraint> {
    specs
        .into_iter()
        .flat_map(|k| k.constraints.iter())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: KnobSpec = KnobSpec::new(
        "emit.a",
        None,
        Layer::Emit,
        Domain::Bool,
        Default::Static(Val::Bool(false)),
        Status::OptIn,
    );
    const B: KnobSpec = KnobSpec::new(
        "emit.b",
        None,
        Layer::Emit,
        U32,
        Default::Production {
            cases: &[DefaultCase {
                when: Formula::Atom("emit.a", Cmp::Eq, Val::Bool(true)),
                value: Val::Nat(4),
            }],
            otherwise: Val::Unset,
        },
        Status::OptIn,
    );

    #[test]
    fn production_default_reads_the_earlier_resolved_knob() {
        let t = Target::default();
        let on = |id: &str| Source {
            cli: (id == "emit.a").then_some(Val::Bool(true)),
            env: None,
        };
        assert_eq!(resolve(&[&A, &B], &on, &t)[1], ("emit.b", Val::Nat(4)));
        assert_eq!(
            resolve(&[&A, &B], &|_: &str| Source::default(), &t)[1],
            ("emit.b", Val::Unset)
        );
        assert!(!well_formed(&[&B, &A], &|_: &str| Source::default()));
    }

    #[test]
    fn unset_compares_false_and_admits_everywhere() {
        assert!(!cmp(Cmp::Gt, Val::Unset, Val::Nat(1)));
        assert!(U32.admits(Val::Unset));
        assert!(!U32.admits(Val::Str("x")));
        assert!(Domain::List(&["a", "b"]).admits(Val::Str("a,b")));
        assert!(!Domain::List(&["a", "b"]).admits(Val::Str("a,c")));
    }
}

/// Scanners and the clap cross-check shared by the devgen and plowrt registry tests.
#[doc(hidden)]
pub mod test_util {
    use super::{Default, Domain, KnobSpec, Status, Val};
    use std::path::{Path, PathBuf};

    /// One clap argument as the registry sees it.
    pub struct ArgFacts {
        pub id: String,
        pub env: Option<String>,
        pub domain: Domain,
        pub default: String,
        pub hide: bool,
    }

    /// Every argument has exactly one spec with its env, domain and default; every spec has an
    /// argument; hidden arguments are `Diagnostic`.
    pub fn check_table(args: &[ArgFacts], prefix: &str, table: &[KnobSpec]) {
        let mut problems = Vec::new();
        for a in args {
            let id = format!("{prefix}{}", a.id);
            let matching: Vec<&KnobSpec> = table.iter().filter(|k| k.id == id).collect();
            let [spec] = matching[..] else {
                problems.push(format!("{id}: {} specs", matching.len()));
                continue;
            };
            if spec.env.map(String::from) != a.env {
                problems.push(format!("{id}: env {:?} vs clap {:?}", spec.env, a.env));
            }
            if spec.domain != a.domain {
                problems.push(format!(
                    "{id}: domain {:?} vs clap {:?}",
                    spec.domain, a.domain
                ));
            }
            let clap_default = if a.default.is_empty() {
                Val::Unset
            } else {
                a.domain.parse(&a.default).unwrap_or(Val::Str(&a.default))
            };
            let spec_default = match spec.default {
                Default::Static(v) => v,
                Default::Production { otherwise, .. } => otherwise,
            };
            if spec_default != clap_default {
                problems.push(format!(
                    "{id}: default {spec_default:?} vs clap {clap_default:?}"
                ));
            }
            if a.hide && spec.status != Status::Diagnostic {
                problems.push(format!("{id}: hidden from --help but not Diagnostic"));
            }
        }
        for k in table {
            if !args.iter().any(|a| format!("{prefix}{}", a.id) == k.id) {
                problems.push(format!("{}: no clap argument", k.id));
            }
        }
        assert!(
            problems.is_empty(),
            "the knob table disagrees with clap; fix the spec line: {problems:#?}"
        );
    }

    /// `PLOW_`/`GLM_`/`K3_` names passed as a string literal to an env reader.
    pub fn env_reads(text: &str) -> Vec<&str> {
        const READERS: &[&str] = &[
            "env::var(",
            "env::var_os(",
            "env_str(",
            "env_bool(",
            "env_u32(",
            "env_usize(",
            "env_bool_opt(",
            "env_opt_out(",
            "env_bool_default_true(",
            "env_nonempty(",
            "env_parse(",
        ];
        let mut out = Vec::new();
        for r in READERS {
            for (i, _) in text.match_indices(r) {
                let rest = text[i + r.len()..].trim_start();
                let Some(rest) = rest.strip_prefix('"') else {
                    continue;
                };
                let Some(end) = rest.find('"') else { continue };
                let name = &rest[..end];
                if ["PLOW_", "GLM_", "K3_"].iter().any(|p| name.starts_with(p))
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                {
                    out.push(name);
                }
            }
        }
        out
    }

    /// Whole `PLOW_*` tokens: not inside a longer identifier, not a `PLOW_BUILD_*`-style prefix.
    pub fn plow_tokens(line: &str) -> Vec<&str> {
        let b = line.as_bytes();
        let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        let mut out = Vec::new();
        for (i, _) in line.match_indices("PLOW_") {
            if i > 0 && ident(b[i - 1]) {
                continue;
            }
            let mut j = i + 5;
            while j < b.len()
                && (b[j].is_ascii_uppercase() || b[j].is_ascii_digit() || b[j] == b'_')
            {
                j += 1;
            }
            if b[j - 1] == b'_' || (j < b.len() && ident(b[j])) {
                continue;
            }
            out.push(&line[i..j]);
        }
        out
    }

    /// `(path, text)` for every file under `dir` with extension `ext` (any when `None`), skipping
    /// vendored `third_party`/`extern` trees.
    pub fn files(dir: &Path, ext: Option<&str>, out: &mut Vec<(PathBuf, String)>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            let p = e.path();
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            if p.is_dir() {
                if name != "third_party" && name != "extern" {
                    files(&p, ext, out);
                }
            } else if ext.is_none_or(|x| p.extension().is_some_and(|e| e == x)) {
                if let Ok(t) = std::fs::read_to_string(&p) {
                    out.push((p, t));
                }
            }
        }
    }
}
