//! What a target packet declares about itself for external drafters (packet section
//! [`SECTION`]): model identity and the geometry and taps a drafter's math may assume.
//!
//! The runtime adds the facts it derives itself (packet digest, tensor table, tokenizer, object
//! symbols) to form a [`crate::spec_drafter::TargetView`]. A packet without this section proves
//! only those derived facts.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::spec_drafter::TargetView;

pub const SECTION: &str = "spec.target.json";
pub const CONTRACT: &str = "plow.spec.target.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSpec {
    pub contract: String,
    /// HF model id the checkpoint was published under.
    pub model: String,
    /// 40-hex hub revision, or a distribution version label.
    pub revision: String,
    /// sha256 over the checkpoint's weight shard contents, in shard-name order.
    pub checkpoint_sha256: String,
    /// Geometry facts by key (`rope_theta_full`, `kv_heads_full`, ...), canonical decimal strings.
    #[serde(default)]
    pub geometry: BTreeMap<String, String>,
    /// Layers whose post-layer residual the programs can store for a drafter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub taps: Vec<u32>,
}

impl TargetSpec {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let s: Self = serde_json::from_slice(bytes).map_err(|e| format!("{SECTION}: {e}"))?;
        if s.contract != CONTRACT {
            return Err(format!(
                "{SECTION}: contract {:?}, this runtime reads {CONTRACT:?}",
                s.contract
            ));
        }
        if s.model.is_empty() || s.revision.is_empty() {
            return Err(format!("{SECTION}: empty model or revision"));
        }
        if s.checkpoint_sha256.len() != 64
            || !s.checkpoint_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(format!("{SECTION}: checkpoint_sha256 is not a sha256"));
        }
        if s.taps.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!("{SECTION}: taps must ascend"));
        }
        Ok(s)
    }

    /// Fold the declared facts into a runtime-derived view.
    pub fn apply(self, view: &mut TargetView) {
        view.model = Some(self.model);
        view.revision = Some(self.revision);
        view.checkpoint_sha256 = Some(self.checkpoint_sha256);
        view.geometry = self.geometry;
        view.taps = self.taps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_applies() {
        let json = format!(
            r#"{{"contract":"{CONTRACT}","model":"m","revision":"{}","checkpoint_sha256":"{}",
                "geometry":{{"hidden":"3840"}},"taps":[5,17]}}"#,
            "a".repeat(40),
            "b".repeat(64)
        );
        let mut v = TargetView::default();
        TargetSpec::parse(json.as_bytes()).unwrap().apply(&mut v);
        assert_eq!(v.model.as_deref(), Some("m"));
        assert_eq!(v.taps, [5, 17]);
        assert_eq!(v.geometry["hidden"], "3840");
    }

    #[test]
    fn refuses_bad_sections() {
        let ok = |c: &str, sha: &str, taps: &str| {
            TargetSpec::parse(
                format!(
                    r#"{{"contract":"{c}","model":"m","revision":"r","checkpoint_sha256":"{sha}","taps":{taps}}}"#
                )
                .as_bytes(),
            )
        };
        ok(CONTRACT, &"b".repeat(64), "[1,2]").unwrap();
        assert!(ok("plow.spec.target.v0", &"b".repeat(64), "[]").is_err());
        assert!(ok(CONTRACT, "xyz", "[]").is_err());
        assert!(ok(CONTRACT, &"b".repeat(64), "[2,2]").is_err());
    }
}
