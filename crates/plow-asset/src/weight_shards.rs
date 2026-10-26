//! `weight_shards.json`: the TP split axis of checkpoint weights the emitter names explicitly.
//!
//! The runtime's substring table (`plowrt::asset::shard::shard_of`) only knows generic HF
//! projection names; a tensor it does not classify binds replicated. A packet that slices such a
//! tensor declares its axis here. Whether a rank binds a slice or the whole tensor still follows
//! the declared byte size, exactly as for the name-classified tensors.

use serde::{Deserialize, Serialize};

pub const SECTION: &str = "weight_shards.json";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightShards {
    pub version: u32,
    /// Output-dim split: a contiguous row range of `[out, in]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub column: Vec<String>,
    /// Input-dim split: `in/N` contiguous elements of every row.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub row: Vec<String>,
}

impl WeightShards {
    pub fn parse(raw: &[u8]) -> Result<Self, String> {
        let s: WeightShards = serde_json::from_slice(raw).map_err(|e| format!("{SECTION}: {e}"))?;
        if s.version != VERSION {
            return Err(format!(
                "{SECTION}: version {} (this runtime reads {VERSION})",
                s.version
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for n in s.column.iter().chain(&s.row) {
            if !seen.insert(n.as_str()) {
                return Err(format!("{SECTION}: `{n}` is declared twice"));
            }
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_refuses_other_versions_and_duplicates() {
        let ok = br#"{"version":1,"row":["a.weight"],"column":["b.weight"]}"#;
        assert_eq!(
            WeightShards::parse(ok).unwrap().row,
            vec!["a.weight".to_string()]
        );
        assert!(WeightShards::parse(br#"{"version":2,"row":[]}"#).is_err());
        assert!(WeightShards::parse(br#"{"version":1,"row":["a"],"column":["a"]}"#).is_err());
        assert!(WeightShards::parse(br#"{"version":1,"rows":["a"]}"#).is_err());
    }
}
