use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const KEY: &str = "text.tags";
pub const VERSION_KEY: &str = "text.tags_version";
pub const VERSION: u64 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechTags {
    pub syntax: Syntax,
    pub unknown: Unknown,
    #[serde(default)]
    pub literal_rewrite: Vec<(String, String)>,
    pub tags: BTreeMap<String, Entry>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Syntax {
    pub bracket: bool,
    pub ssml: bool,
    pub asterisk: Asterisk,
    #[serde(default)]
    pub markers: Markers,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Markers {
    #[default]
    Neutralize,
    Strip,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Asterisk {
    Known,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unknown {
    Strip,
    Literal,
    Passthrough,
    Reject,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub silence_ms: Option<Silence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drop: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reject: Option<bool>,
    #[serde(default)]
    pub carry: bool,
    #[serde(default)]
    pub single_token: bool,
    #[serde(default)]
    pub budget_chars: Option<usize>,
    #[serde(default)]
    pub langs: Langs,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Langs {
    Only(Vec<String>),
    Text(BTreeMap<String, String>),
}

impl Default for Langs {
    fn default() -> Self {
        Self::Only(Vec::new())
    }
}

impl Langs {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Only(v) => v.is_empty(),
            Self::Text(v) => v.is_empty(),
        }
    }

    pub fn allows(&self, language: Option<&str>) -> bool {
        match self {
            Self::Only(v) => v.is_empty() || language.is_some_and(|l| v.iter().any(|v| v == l)),
            Self::Text(v) => language.is_none_or(|l| v.contains_key(l)),
        }
    }

    fn valid(&self) -> bool {
        let language = |l: &str| {
            !l.is_empty() && l.len() <= 16 && l.bytes().all(|b| b.is_ascii_lowercase() || b == b'-')
        };
        match self {
            Self::Only(v) => v.len() <= 32 && v.iter().all(|l| language(l)),
            Self::Text(v) => !v.is_empty() && v.len() <= 32 && v.keys().all(|l| language(l)),
        }
    }

    fn texts(&self) -> impl Iterator<Item = &str> {
        let values = match self {
            Self::Only(_) => None,
            Self::Text(v) => Some(v.values()),
        };
        values.into_iter().flatten().map(String::as_str)
    }
}

impl Entry {
    pub fn native_text(&self, language: Option<&str>) -> Option<&str> {
        let text = self.text.as_deref()?;
        match (&self.langs, language) {
            (Langs::Text(v), Some(l)) => v.get(l).map(String::as_str),
            _ => self.langs.allows(language).then_some(text),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Silence {
    pub default: u32,
    pub max: u32,
}

impl SpeechTags {
    pub fn strip() -> Self {
        let mut tags: BTreeMap<String, Entry> = [
            "laugh",
            "chuckle",
            "giggle",
            "sigh",
            "gasp",
            "cough",
            "clear_throat",
            "inhale",
            "exhale",
            "sniff",
            "yawn",
            "groan",
            "whisper",
            "excited",
            "happy",
            "sad",
            "angry",
            "calm",
            "serious",
            "surprised",
            "friendly",
            "casual",
            "rate",
            "pitch",
            "emphasis",
        ]
        .into_iter()
        .map(|name| {
            (
                name.into(),
                Entry {
                    drop: Some(true),
                    ..Entry::default()
                },
            )
        })
        .collect();
        tags.insert(
            "pause".into(),
            Entry {
                silence_ms: Some(Silence {
                    default: 400,
                    max: 3000,
                }),
                ..Entry::default()
            },
        );
        Self {
            syntax: Syntax {
                bracket: true,
                ssml: true,
                asterisk: Asterisk::Known,
                markers: Markers::Strip,
            },
            unknown: Unknown::Strip,
            literal_rewrite: vec![("<|".into(), "< |".into())],
            tags,
        }
    }

    pub fn parse(json: &str) -> Result<Self, String> {
        let map: Self = serde_json::from_str(json).map_err(|e| format!("text.tags: {e}"))?;
        map.validate()?;
        Ok(map)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.tags.len() > 128 || self.literal_rewrite.len() > 16 {
            return Err("text.tags: map exceeds limits".into());
        }
        if !self.tags.get("pause").is_some_and(|e| {
            e.silence_ms
                .as_ref()
                .is_some_and(|s| s.default == 400 && s.max == 3000)
        }) {
            return Err("text.tags: pause requires 400 ms default and 3000 ms maximum".into());
        }
        for (from, to) in &self.literal_rewrite {
            if from.is_empty()
                || from.len() > 64
                || to.len() > 64
                || to.contains("<|")
                || self
                    .literal_rewrite
                    .iter()
                    .any(|(source, _)| to.contains(source))
            {
                return Err("text.tags: invalid literal rewrite".into());
            }
        }
        for (name, e) in &self.tags {
            if name.is_empty()
                || name.len() > 32
                || !name.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
            {
                return Err(format!("text.tags: invalid name {name:?}"));
            }
            if !e.langs.valid() || matches!(&e.langs, Langs::Text(_)) && e.text.is_none() {
                return Err(format!("text.tags: invalid languages for {name}"));
            }
            if e.speed.is_some() && name != "rate" {
                return Err("text.tags: speed requires rate".into());
            }
            let actions = [
                e.text.is_some(),
                e.instruction.is_some(),
                e.silence_ms.is_some(),
                e.speed.is_some(),
                e.alias.is_some(),
                e.drop.is_some(),
                e.reject.is_some(),
            ];
            if actions.into_iter().filter(|v| *v).count() != 1
                || e.speed == Some(false)
                || e.drop == Some(false)
                || e.reject == Some(false)
            {
                return Err(format!("text.tags: {name} needs exactly one action"));
            }
            if e.single_token && e.text.is_none()
                || e.carry && e.text.is_none()
                || e.budget_chars.is_some() && e.text.is_none()
            {
                return Err(format!("text.tags: invalid modifiers for {name}"));
            }
            if let Some(t) = &e.text {
                if std::iter::once(t.as_str()).chain(e.langs.texts()).any(|t| {
                    t.is_empty()
                        || t.chars().count() > e.budget_chars.unwrap_or(64).min(64)
                        || t.contains("<|")
                }) {
                    return Err(format!("text.tags: invalid native text for {name}"));
                }
            }
            if let Some(i) = &e.instruction {
                if i.is_empty()
                    || i.len() > 1024
                    || i.contains("<|")
                    || i.contains("{value}") && name != "rate"
                {
                    return Err(format!("text.tags: invalid instruction for {name}"));
                }
            }
            if let Some(s) = &e.silence_ms {
                if name != "pause" || s.max > 3000 || s.default > s.max || !e.langs.is_empty() {
                    return Err("text.tags: invalid silence policy".into());
                }
            }
            if name == "pause" && e.silence_ms.is_none() {
                return Err("text.tags: pause must use runtime silence".into());
            }
            let mut seen = BTreeSet::new();
            let mut next = name.as_str();
            loop {
                if !seen.insert(next) {
                    return Err(format!("text.tags: alias cycle at {name}"));
                }
                let entry = self
                    .tags
                    .get(next)
                    .ok_or_else(|| format!("text.tags: missing alias {next}"))?;
                match entry.alias.as_deref() {
                    Some(a) => next = a,
                    None => break,
                }
            }
        }
        Ok(())
    }

    pub fn validate_binding(
        &self,
        instructions: bool,
        mut single_token: impl FnMut(&str) -> bool,
    ) -> Result<(), String> {
        self.validate()?;
        for (name, e) in &self.tags {
            if e.instruction.is_some() && !instructions {
                return Err(format!("text.tags: {name} requires instructions"));
            }
            if e.single_token
                && std::iter::once(e.text.as_deref().unwrap())
                    .chain(e.langs.texts())
                    .any(|t| !single_token(t))
            {
                return Err(format!(
                    "text.tags: {name} is not a single vocabulary token"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn map() -> SpeechTags {
        SpeechTags::parse(r#"{"syntax":{"bracket":true,"ssml":true,"asterisk":"known"},"unknown":"strip","tags":{"pause":{"silence_ms":{"default":400,"max":3000}}}}"#).unwrap()
    }
    #[test]
    fn strip_policy_has_only_runtime_silence() {
        let map = SpeechTags::strip();
        map.validate_binding(false, |_| false).unwrap();
        assert_eq!(map.unknown, Unknown::Strip);
        assert_eq!(map.syntax.markers, Markers::Strip);
        assert_eq!(self::map().syntax.markers, Markers::Neutralize);
        for (name, entry) in &map.tags {
            if name == "pause" {
                assert_eq!(
                    entry.silence_ms,
                    Some(Silence {
                        default: 400,
                        max: 3000
                    })
                );
            } else {
                assert_eq!(entry.drop, Some(true), "{name}");
            }
        }
        assert_eq!(
            SpeechTags::parse(&serde_json::to_string(&map).unwrap()).unwrap(),
            map
        );
    }

    #[test]
    fn language_forms_share_native_validation_and_binding() {
        let mut m = map();
        m.tags.insert(
            "laugh".into(),
            serde_json::from_str(
                r#"{"text":"A","langs":{"en":"A","zh":"笑"},"single_token":true,"budget_chars":3}"#,
            )
            .unwrap(),
        );
        m.validate_binding(false, |t| t.chars().count() == 1)
            .unwrap();
        assert_eq!(m.tags["laugh"].native_text(Some("zh")), Some("笑"));
        assert_eq!(m.tags["laugh"].native_text(Some("hi")), None);
        assert_eq!(m.tags["laugh"].native_text(None), Some("A"));
        for entry in [
            r#"{"text":"A","langs":{"zh":"AB"},"single_token":true}"#,
            r#"{"text":"A","langs":{"zh":"<|event|>"}}"#,
            r#"{"text":"A","langs":{"zh":""}}"#,
            r#"{"text":"A","langs":{}}"#,
            r#"{"drop":true,"langs":{"zh":"笑"}}"#,
            r#"{"instruction":"Speak {value}."}"#,
        ] {
            m.tags
                .insert("laugh".into(), serde_json::from_str(entry).unwrap());
            assert!(
                m.validate_binding(true, |t| t.chars().count() == 1)
                    .is_err(),
                "{entry}"
            );
        }
    }

    #[test]
    fn fail_closed() {
        let m = map();
        let mut json = serde_json::to_value(&m).unwrap();
        json["extra"] = true.into();
        assert!(SpeechTags::parse(&json.to_string()).is_err());
        for entry in [
            r#"{"drop":true,"text":"x"}"#,
            r#"{"drop":false}"#,
            r#"{"alias":"missing"}"#,
            r#"{"alias":"laugh"}"#,
            r#"{"text":"<|event|>"}"#,
            r#"{"text":"xx","single_token":true}"#,
        ] {
            let mut m = m.clone();
            m.tags
                .insert("laugh".into(), serde_json::from_str(entry).unwrap());
            assert!(m.validate_binding(false, |_| false).is_err(), "{entry}");
        }
        assert!(m.validate_binding(false, |_| false).is_ok());
        let mut recursive = m.clone();
        recursive.literal_rewrite = vec![("a".into(), "aa".into())];
        assert!(recursive.validate().is_err());
        let mut chained = m.clone();
        chained.literal_rewrite = vec![("a".into(), "b".into()), ("b".into(), "cc".into())];
        assert!(chained.validate().is_err());
        let mut needs_instructions = m.clone();
        needs_instructions.tags.insert(
            "calm".into(),
            Entry {
                instruction: Some("Speak calmly.".into()),
                ..Entry::default()
            },
        );
        assert!(needs_instructions
            .validate_binding(false, |_| true)
            .is_err());
        assert!(needs_instructions.validate_binding(true, |_| true).is_ok());
    }
}
