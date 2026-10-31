//! `derived_weights.json`: packet tensors the loader assembles from checkpoint tensors.
//!
//! Each entry names a declared tensor (spelled `cat/...`, `packet::names::CONCAT_PREFIX`) and the
//! checkpoint tensors whose bytes it is, back to back along axis 0 (same dtype and trailing dims),
//! optionally widened BF16 -> F32 (`widen_f32`). The tensor name is only a label; the sources
//! here are the contract. Packets carrying the section require `RUNTIME_CONTRACT_CONCAT`.

use serde::{Deserialize, Serialize};

pub const SECTION: &str = "derived_weights.json";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedWeights {
    pub version: u32,
    pub tensors: Vec<DerivedTensor>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedTensor {
    pub name: String,
    pub sources: Vec<String>,
    #[serde(default)]
    pub widen_f32: bool,
}

impl DerivedWeights {
    pub fn parse(raw: &[u8]) -> Result<Self, String> {
        let s: DerivedWeights =
            serde_json::from_slice(raw).map_err(|e| format!("{SECTION}: {e}"))?;
        if s.version != VERSION {
            return Err(format!(
                "{SECTION}: version {} (this runtime reads {VERSION})",
                s.version
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for t in &s.tensors {
            if !t.name.starts_with("cat/") || !seen.insert(t.name.as_str()) {
                return Err(format!(
                    "{SECTION}: `{}` is not a unique cat/ tensor",
                    t.name
                ));
            }
            if t.sources.len() < 2
                || t.sources
                    .iter()
                    .any(|s| s.is_empty() || s.starts_with("cat/") || s.starts_with("f32/"))
            {
                return Err(format!(
                    "{SECTION}: `{}` needs two or more checkpoint sources",
                    t.name
                ));
            }
        }
        Ok(s)
    }

    pub fn get(&self, name: &str) -> Option<&DerivedTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_refuses_other_versions_duplicates_and_degenerate_sources() {
        let ok =
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a","b"],"widen_f32":true}]}"#;
        let d = DerivedWeights::parse(ok).unwrap();
        assert_eq!(d.get("cat/a").unwrap().sources, ["a", "b"]);
        assert!(d.get("cat/a").unwrap().widen_f32);
        for bad in [
            &br#"{"version":2,"tensors":[]}"#[..],
            br#"{"version":1,"tensors":[{"name":"a","sources":["a","b"]}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a"]}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a","cat/b"]}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a",""]}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a","b"]},{"name":"cat/a","sources":["a","b"]}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a","b"],"axis":1}]}"#,
        ] {
            assert!(DerivedWeights::parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
    }
}
