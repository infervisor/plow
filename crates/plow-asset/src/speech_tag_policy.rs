use crate::speech_tags::{Markers, SpeechTags, VERSION};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechTagPolicy {
    pub version: u64,
    pub profile: Option<String>,
    pub map: Option<SpeechTags>,
    pub waiver: Option<String>,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub native_sweep: Option<String>,
}

impl SpeechTagPolicy {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != VERSION || self.map.is_some() == self.waiver.is_some() {
            return Err("speech_tags: requires version 1 and exactly one map or waiver".into());
        }
        if self.evidence.iter().any(|e| e.trim().is_empty()) {
            return Err("speech_tags: evidence must list measurement references".into());
        }
        if let Some(reason) = &self.waiver {
            let lower = reason.to_lowercase();
            if reason.trim().is_empty()
                || self.evidence.is_empty()
                || self.profile.is_some()
                || self.native_sweep.is_some()
                || [
                    "not measured",
                    "not yet measured",
                    "unmeasured",
                    "todo",
                    "tbd",
                ]
                .iter()
                .any(|s| lower.contains(s))
            {
                return Err("speech_tags: waiver needs a measured reason and evidence".into());
            }
        }
        if let Some(map) = &self.map {
            map.validate()?;
            if self.profile.as_ref().is_none_or(|s| s.trim().is_empty())
                || !map.syntax.bracket
                || !map.syntax.ssml
                || map.syntax.asterisk != crate::speech_tags::Asterisk::Known
                || map.syntax.markers != Markers::Strip
                || map.tags.keys().ne(SpeechTags::strip().tags.keys())
                || self
                    .native_sweep
                    .as_ref()
                    .is_some_and(|s| s.trim().is_empty())
            {
                return Err("speech_tags: declare a profile, all canonical tags and accepted syntax; strip markers".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_or_measured_waiver() {
        let mut p = SpeechTagPolicy {
            version: 1,
            profile: Some("strip-silence".into()),
            map: Some(SpeechTags::strip()),
            waiver: None,
            evidence: vec![],
            native_sweep: None,
        };
        p.validate().unwrap();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<SpeechTagPolicy>(&json).unwrap(), p);
        p.map.as_mut().unwrap().tags.remove("sigh");
        assert!(p.validate().is_err());
        p.map = None;
        p.profile = None;
        p.waiver =
            Some("CER 0.31 vs 0.01 on pinned capture; disable pending frontend correction".into());
        assert!(p.validate().is_err());
        p.evidence
            .push("qualification/tag-ab.json sha256:abc".into());
        p.validate().unwrap();
        p.map = Some(SpeechTags::strip());
        assert!(p.validate().is_err());
        assert!(serde_json::from_str::<SpeechTagPolicy>(
            &json.replace("\"version\":1", "\"extra\":1,\"version\":1")
        )
        .is_err());
    }
}
