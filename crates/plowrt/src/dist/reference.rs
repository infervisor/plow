//! Parsing a model reference.
//!
//! `[<registry>/]<namespace>/<name>[:<label>][@g<n>]`
//!
//! A bare `<name>` expands to the default registry and namespace, the same
//! defaulting ollama applies with `library/`. The HuggingFace repo is recorded
//! in the model index rather than typed here, so a reference stays short and the
//! checkpoint identity stays with the artifact that declares it.

use std::fmt;

pub const DEFAULT_REGISTRY: &str = "dist.infervisor.ai";
pub const DEFAULT_NAMESPACE: &str = "infervisor";

/// A parsed reference. `label` and `generation` are absent for the common case
/// where the probe decides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference {
    pub registry: String,
    pub namespace: String,
    pub name: String,
    pub label: Option<String>,
    pub generation: Option<u32>,
}

impl Reference {
    /// Path of this model's variant index on the server, relative to the
    /// registry root.
    pub fn index_path(&self) -> String {
        format!("v1/{}/{}/index.json", self.namespace, self.name)
    }

    /// Path of one variant's manifest.
    pub fn manifest_path(&self, label: &str, generation: u32) -> String {
        format!(
            "v1/{}/{}/manifests/{label}@g{generation}",
            self.namespace, self.name
        )
    }

    /// Where this reference's pin lives in the local store.
    pub fn ref_path(&self) -> String {
        format!("{}/{}/{}", self.registry, self.namespace, self.name)
    }
}

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.registry, self.namespace, self.name)?;
        if let Some(l) = &self.label {
            write!(f, ":{l}")?;
        }
        if let Some(g) = self.generation {
            write!(f, "@g{g}")?;
        }
        Ok(())
    }
}

/// A path segment that cannot escape its directory or confuse a URL. Rejected
/// rather than sanitised: a reference is typed by a person, and silently
/// rewriting it would resolve to something they did not ask for.
fn segment_ok(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

pub fn parse(s: &str) -> Result<Reference, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty model reference".into());
    }

    // Split the trailing `@g<n>` first: a generation cannot contain `:` or `/`,
    // so peeling it off makes the rest unambiguous.
    let (rest, generation) = match s.rsplit_once("@g") {
        Some((head, gen)) => {
            let n: u32 = gen
                .parse()
                .map_err(|_| format!("`@g{gen}`: a generation is a number, e.g. `@g3`"))?;
            (head, Some(n))
        }
        None => (s, None),
    };
    if rest.contains('@') {
        return Err(format!(
            "{s}: `@` introduces a generation and is written `@g<n>`"
        ));
    }

    let (path, label) = match rest.split_once(':') {
        Some((p, l)) => {
            if !segment_ok(l) {
                return Err(format!("{s}: `{l}` is not a valid variant label"));
            }
            (p, Some(l.to_string()))
        }
        None => (rest, None),
    };
    if generation.is_some() && label.is_none() {
        return Err(format!(
            "{s}: a generation pins one variant, so it needs a label: `<name>:<label>@g<n>`"
        ));
    }

    let parts: Vec<&str> = path.split('/').collect();
    let (registry, namespace, name) = match parts.as_slice() {
        [name] => (DEFAULT_REGISTRY, DEFAULT_NAMESPACE, *name),
        [ns, name] => (DEFAULT_REGISTRY, *ns, *name),
        [reg, ns, name] => (*reg, *ns, *name),
        _ => {
            return Err(format!(
                "{s}: expected [<registry>/]<namespace>/<name>, got {} segments",
                parts.len()
            ))
        }
    };
    for (what, seg) in [
        ("registry", registry),
        ("namespace", namespace),
        ("name", name),
    ] {
        if !segment_ok(seg) {
            return Err(format!("{s}: `{seg}` is not a valid {what}"));
        }
    }
    Ok(Reference {
        registry: registry.to_string(),
        namespace: namespace.to_string(),
        name: name.to_string(),
        label,
        generation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_name_takes_the_default_registry_and_namespace() {
        let r = parse("kimi-k3").unwrap();
        assert_eq!(r.registry, DEFAULT_REGISTRY);
        assert_eq!(r.namespace, DEFAULT_NAMESPACE);
        assert_eq!(r.name, "kimi-k3");
        assert_eq!(r.label, None);
        assert_eq!(r.generation, None);
        assert_eq!(r.index_path(), "v1/infervisor/kimi-k3/index.json");
    }

    #[test]
    fn a_namespace_selects_a_publisher() {
        let r = parse("acme/kimi-k3-finetune").unwrap();
        assert_eq!(r.namespace, "acme");
        assert_eq!(r.name, "kimi-k3-finetune");
        assert_eq!(r.registry, DEFAULT_REGISTRY);
        assert_eq!(r.index_path(), "v1/acme/kimi-k3-finetune/index.json");
    }

    #[test]
    fn a_registry_can_be_named_explicitly() {
        let r = parse("mirror.internal/infervisor/kimi-k3").unwrap();
        assert_eq!(r.registry, "mirror.internal");
        assert_eq!(r.namespace, "infervisor");
        assert_eq!(r.name, "kimi-k3");
    }

    #[test]
    fn a_label_and_generation_pin_a_variant() {
        let r = parse("kimi-k3:gfx942-mi325x-tp8-32k-fp8kv-mxfp4@g3").unwrap();
        assert_eq!(
            r.label.as_deref(),
            Some("gfx942-mi325x-tp8-32k-fp8kv-mxfp4")
        );
        assert_eq!(r.generation, Some(3));
        assert_eq!(
            r.manifest_path("gfx942-mi325x-tp8-32k-fp8kv-mxfp4", 3),
            "v1/infervisor/kimi-k3/manifests/gfx942-mi325x-tp8-32k-fp8kv-mxfp4@g3"
        );
    }

    #[test]
    fn a_label_alone_means_the_newest_generation_of_it() {
        let r = parse("kimi-k3:gfx942-mi325x-tp8").unwrap();
        assert_eq!(r.label.as_deref(), Some("gfx942-mi325x-tp8"));
        assert_eq!(r.generation, None);
    }

    // A generation without a label does not identify anything: two labels can
    // both have a g3.
    #[test]
    fn a_generation_without_a_label_is_refused() {
        let err = parse("kimi-k3@g3").unwrap_err();
        assert!(err.contains("needs a label"), "{err}");
    }

    #[test]
    fn round_trips_through_display() {
        for s in [
            "dist.infervisor.ai/infervisor/kimi-k3",
            "dist.infervisor.ai/acme/x:label",
            "dist.infervisor.ai/infervisor/kimi-k3:label@g7",
        ] {
            assert_eq!(parse(s).unwrap().to_string(), s);
        }
        // The short forms expand rather than round-trip verbatim.
        assert_eq!(
            parse("kimi-k3").unwrap().to_string(),
            "dist.infervisor.ai/infervisor/kimi-k3"
        );
    }

    // A reference is typed by a person; a traversal is refused, never sanitised
    // into something they did not ask for.
    #[test]
    fn path_traversal_and_junk_are_refused() {
        for bad in [
            "",
            "   ",
            "../etc",
            "a/../b",
            "ns/..",
            "a/b/c/d",
            "kimi k3",
            "kimi/k3?x=1",
            "ns//name",
            "kimi-k3@3",
            "kimi-k3@gx",
            "kimi-k3:",
            "kimi-k3:bad label",
        ] {
            assert!(parse(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn the_pin_path_is_registry_scoped() {
        // Two registries may publish the same name; their pins must not collide.
        let a = parse("kimi-k3").unwrap();
        let b = parse("mirror.internal/infervisor/kimi-k3").unwrap();
        assert_ne!(a.ref_path(), b.ref_path());
    }
}
