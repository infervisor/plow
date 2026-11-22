//! `derived_weights.json`: packet tensors the loader assembles from checkpoint tensors.
//!
//! Each entry names a declared tensor (spelled `cat/...`, `packet::names::CONCAT_PREFIX`) and the
//! checkpoint tensors whose bytes it is, back to back along axis 0 (same dtype and trailing dims),
//! optionally widened BF16 -> F32 (`widen_f32`), or, with `cols = [lo, hi]`, the columns
//! `lo..hi` of one 2-D source (every row's slice back to back: a Linear weight's input-feature
//! block), or, with `repeat = n`, one source's bytes tiled `n` times. The tensor name is only a label; the sources here are the contract. Packets carrying
//! the section require `RUNTIME_CONTRACT_CONCAT`.

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cols: Option<[u64; 2]>,
    /// The one source's bytes tiled this many times (a per-tensor FP8 `weight_scale` widened to
    /// the per-output-channel vector the FP8 GEMV/GEMM read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat: Option<u64>,
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
            if let Some(n) = t.repeat {
                if t.sources.len() != 1
                    || t.widen_f32
                    || t.cols.is_some()
                    || n < 2
                    || t.sources[0].is_empty()
                    || t.sources[0].starts_with("cat/")
                {
                    return Err(format!(
                        "{SECTION}: `{}` tiles exactly one checkpoint source {n} (>= 2) times",
                        t.name
                    ));
                }
                continue;
            }
            if let Some([lo, hi]) = t.cols {
                if t.sources.len() != 1 || t.widen_f32 || lo >= hi || t.sources[0].is_empty() {
                    return Err(format!(
                        "{SECTION}: `{}` slices columns {lo}..{hi} of exactly one source",
                        t.name
                    ));
                }
                continue;
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
        let rep = br#"{"version":1,"tensors":[{"name":"cat/r","sources":["s"],"repeat":4}]}"#;
        assert_eq!(
            DerivedWeights::parse(rep).unwrap().get("cat/r").unwrap().repeat,
            Some(4)
        );
        let cols = br#"{"version":1,"tensors":[{"name":"cat/c","sources":["w"],"cols":[2,6]}]}"#;
        assert_eq!(
            DerivedWeights::parse(cols)
                .unwrap()
                .get("cat/c")
                .unwrap()
                .cols,
            Some([2, 6])
        );
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
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a","b"],"cols":[0,4]}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a"],"cols":[4,4]}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a"],"cols":[0,4],"widen_f32":true}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a","b"],"repeat":4}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a"],"repeat":1}]}"#,
            br#"{"version":1,"tensors":[{"name":"cat/a","sources":["a"],"repeat":4,"cols":[0,2]}]}"#,
        ] {
            assert!(DerivedWeights::parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
    }
}
