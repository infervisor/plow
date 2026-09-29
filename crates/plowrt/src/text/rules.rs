//! Packet-declared text normalization: one rule per line, tab-separated `kind\targs...`, applied
//! in order. Kinds:
//! - `default_if_empty\tTEXT`, `capitalize_first`, `collapse_whitespace`, `replace\tFROM\tTO`,
//!   `trim_end\tCHARS`, `ensure_suffix\tCHARS\tSUFFIX` (append SUFFIX unless the text ends with one
//!   of CHARS), `lowercase` (Unicode full case mapping), `nfkd`, `strip` (Unicode whitespace)
//! - `language`: the selected language's rule list (pipeline strings `text.rules.lang.<code>`)
//! - `prefix\tTEXT`: prepend TEXT; `{lang}` in any argument is the selected language code
//! - `map_chars\tTABLE`: replace each character that has a `char\treplacement` line in TABLE
//! - `dict_longest\tTABLE`: left to right, replace the longest `key\tvalue` key starting at each
//!   position (characters no key starts with are kept)
//! - `drop_chars\tHEX-HEX,...`: delete characters in the code point ranges
//! - `segment_crf\tPREFIX`: word segmentation by a linear-chain CRF over character features
//!   ([`super::segment`], tables `PREFIX.*`); words are joined with one space
//!
//! TABLEs are named blobs of the packet's `text_tables.v1` metadata section.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{Result, RuntimeError};

/// The packet's named text tables (`text_tables.v1`: repeated `u32 name_len, name, u64 len, data`).
#[derive(Default)]
pub struct Tables(HashMap<String, Vec<u8>>);

impl Tables {
    pub const SECTION: &'static str = "text_tables.v1";

    pub fn parse(mut b: &[u8]) -> Result<Self> {
        let bad = || RuntimeError::Rejected("malformed text_tables.v1 section".into());
        let mut t = HashMap::new();
        while !b.is_empty() {
            let n = u32::from_le_bytes(b.get(..4).ok_or_else(bad)?.try_into().unwrap()) as usize;
            let name = std::str::from_utf8(b.get(4..4 + n).ok_or_else(bad)?).map_err(|_| bad())?.to_string();
            b = &b[4 + n..];
            let len = u64::from_le_bytes(b.get(..8).ok_or_else(bad)?.try_into().unwrap()) as usize;
            t.insert(name, b.get(8..8 + len).ok_or_else(bad)?.to_vec());
            b = &b[8 + len..];
        }
        Ok(Tables(t))
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.0.get(name).map(Vec::as_slice).ok_or_else(|| RuntimeError::Rejected(format!("text table {name:?} is missing")))
    }

    pub fn text(&self, name: &str) -> Result<&str> {
        std::str::from_utf8(self.bytes(name)?).map_err(|_| RuntimeError::Rejected(format!("text table {name:?} is not UTF-8")))
    }

    /// `key\tvalue` lines.
    fn pairs(&self, name: &str) -> Result<impl Iterator<Item = (&str, &str)>> {
        Ok(self.text(name)?.split('\n').filter(|l| !l.is_empty()).map(|l| l.split_once('\t').unwrap_or((l, ""))))
    }
}

struct Longest {
    starts: HashSet<char>,
    map: HashMap<Box<str>, Box<str>>,
    max_chars: usize,
}

enum Rule {
    DefaultIfEmpty(String),
    CapitalizeFirst,
    CollapseWhitespace,
    Replace(String, String),
    TrimEnd(Vec<char>),
    EnsureSuffix(String, String),
    Lowercase,
    Nfkd,
    Strip,
    Language,
    Prefix(String),
    MapChars(Arc<HashMap<char, Box<str>>>),
    DictLongest(Arc<Longest>),
    DropChars(Vec<(u32, u32)>),
    Segment(Arc<super::segment::CrfSegmenter>),
}

/// Compiled rules: the main list and, per language, the list `language` runs.
#[derive(Default)]
pub struct TextRules {
    main: Vec<Rule>,
    langs: HashMap<String, Vec<Rule>>,
    default_lang: Option<String>,
}

fn compile(rules: &str, tables: &Tables, cache: &mut HashMap<String, Rule>) -> Result<Vec<Rule>> {
    let mut out = Vec::new();
    for line in rules.lines().filter(|l| !l.is_empty()) {
        let mut f = line.split('\t');
        let kind = f.next().unwrap_or_default();
        let args: Vec<&str> = f.collect();
        let arg = |i: usize| {
            args.get(i).map(|s| s.to_string()).ok_or_else(|| RuntimeError::Rejected(format!("text rule {kind:?} needs argument {i}")))
        };
        let shared = |cache: &mut HashMap<String, Rule>, build: &dyn Fn() -> Result<Rule>| -> Result<Rule> {
            let key = line.to_string();
            if !cache.contains_key(&key) {
                cache.insert(key.clone(), build()?);
            }
            Ok(match &cache[&key] {
                Rule::MapChars(m) => Rule::MapChars(Arc::clone(m)),
                Rule::DictLongest(m) => Rule::DictLongest(Arc::clone(m)),
                Rule::Segment(m) => Rule::Segment(Arc::clone(m)),
                _ => unreachable!(),
            })
        };
        out.push(match kind {
            "default_if_empty" => Rule::DefaultIfEmpty(arg(0)?),
            "capitalize_first" => Rule::CapitalizeFirst,
            "collapse_whitespace" => Rule::CollapseWhitespace,
            "replace" => Rule::Replace(arg(0)?, arg(1)?),
            "trim_end" => Rule::TrimEnd(arg(0)?.chars().collect()),
            "ensure_suffix" => Rule::EnsureSuffix(arg(0)?, arg(1)?),
            "lowercase" => Rule::Lowercase,
            "nfkd" => Rule::Nfkd,
            "strip" => Rule::Strip,
            "language" => Rule::Language,
            "prefix" => Rule::Prefix(arg(0)?),
            "map_chars" => shared(cache, &|| {
                let mut m = HashMap::new();
                for (k, v) in tables.pairs(&arg(0)?)? {
                    let mut c = k.chars();
                    match (c.next(), c.next()) {
                        (Some(ch), None) => m.insert(ch, v.into()),
                        _ => return Err(RuntimeError::Rejected(format!("map_chars key {k:?} is not one character"))),
                    };
                }
                Ok(Rule::MapChars(Arc::new(m)))
            })?,
            "dict_longest" => shared(cache, &|| {
                let (mut starts, mut map, mut max_chars) = (HashSet::new(), HashMap::new(), 0);
                for (k, v) in tables.pairs(&arg(0)?)? {
                    starts.extend(k.chars().next());
                    max_chars = max_chars.max(k.chars().count());
                    map.insert(k.into(), v.into());
                }
                Ok(Rule::DictLongest(Arc::new(Longest { starts, map, max_chars })))
            })?,
            "drop_chars" => {
                let mut ranges = Vec::new();
                for r in arg(0)?.split(',').filter(|r| !r.is_empty()) {
                    let (a, b) = r.split_once('-').unwrap_or((r, r));
                    let hex = |s: &str| u32::from_str_radix(s, 16).map_err(|_| RuntimeError::Rejected(format!("drop_chars range {r:?}")));
                    ranges.push((hex(a)?, hex(b)?));
                }
                Rule::DropChars(ranges)
            }
            "segment_crf" => shared(cache, &|| Ok(Rule::Segment(Arc::new(super::segment::CrfSegmenter::load(tables, &arg(0)?)?))))?,
            other => return Err(RuntimeError::Rejected(format!("unknown text rule {other:?}"))),
        });
    }
    Ok(out)
}

impl TextRules {
    /// `langs`: (code, rule list) pairs; empty when the text has no language selection.
    pub fn compile<'a>(
        rules: &str,
        langs: impl IntoIterator<Item = (&'a str, &'a str)>,
        default_lang: Option<&str>,
        tables: &Tables,
    ) -> Result<Self> {
        let mut cache = HashMap::new();
        let main = compile(rules, tables, &mut cache)?;
        let mut l = HashMap::new();
        for (code, r) in langs {
            l.insert(code.to_string(), compile(r, tables, &mut cache)?);
        }
        if let Some(d) = default_lang.filter(|d| !l.contains_key(*d)) {
            return Err(RuntimeError::Rejected(format!("default language {d:?} is not declared")));
        }
        let t = TextRules { main, langs: l, default_lang: default_lang.map(str::to_string) };
        t.apply("", t.default_lang.as_deref())?;
        Ok(t)
    }

    pub fn languages(&self) -> impl Iterator<Item = &str> {
        self.langs.keys().map(String::as_str)
    }

    /// The language a request selects: its own (lowercased, must be declared) or the default.
    pub fn resolve(&self, lang: Option<&str>) -> Result<Option<String>> {
        match lang.map(str::to_lowercase) {
            None => Ok(self.default_lang.clone()),
            Some(l) if self.langs.contains_key(&l) => Ok(Some(l)),
            Some(l) if self.langs.is_empty() => Err(RuntimeError::Rejected(format!("language {l:?}: this model has no language selection"))),
            Some(l) => {
                let mut known: Vec<_> = self.langs.keys().map(String::as_str).collect();
                known.sort_unstable();
                Err(RuntimeError::Rejected(format!("unsupported language {l:?} (supported: {})", known.join(", "))))
            }
        }
    }

    pub fn apply(&self, text: &str, lang: Option<&str>) -> Result<String> {
        run(&self.main, text.to_string(), lang, &self.langs)
    }
}

fn run(rules: &[Rule], mut t: String, lang: Option<&str>, langs: &HashMap<String, Vec<Rule>>) -> Result<String> {
    let expand = |s: &str| if s.contains("{lang}") { s.replace("{lang}", lang.unwrap_or_default()) } else { s.to_string() };
    for rule in rules {
        match rule {
            Rule::DefaultIfEmpty(d) => {
                if t.is_empty() {
                    t = d.clone();
                }
            }
            Rule::CapitalizeFirst => {
                let mut chars = t.chars();
                if let Some(c0) = chars.next().filter(|c| c.is_lowercase()) {
                    t = c0.to_uppercase().chain(chars).collect();
                }
            }
            Rule::CollapseWhitespace => t = t.split_whitespace().collect::<Vec<_>>().join(" "),
            Rule::Replace(a, b) => t = t.replace(a.as_str(), &expand(b)),
            Rule::TrimEnd(chars) => t = t.trim_end_matches(chars.as_slice()).to_string(),
            Rule::EnsureSuffix(ends, suffix) => {
                if !t.chars().last().is_some_and(|c| ends.contains(c)) {
                    t.push_str(suffix);
                }
            }
            Rule::Lowercase => t = t.to_lowercase(),
            Rule::Nfkd => t = nfkd(&t)?,
            Rule::Strip => t = t.trim().to_string(),
            Rule::Language => {
                let l = lang.ok_or_else(|| RuntimeError::Rejected("text rule `language` needs a language".into()))?;
                let r = langs.get(l).ok_or_else(|| RuntimeError::Rejected(format!("no text rules for language {l:?}")))?;
                t = run(r, t, lang, langs)?;
            }
            Rule::Prefix(p) => t.insert_str(0, &expand(p)),
            Rule::MapChars(m) => {
                let mut o = String::with_capacity(t.len());
                for c in t.chars() {
                    match m.get(&c) {
                        Some(r) => o.push_str(r),
                        None => o.push(c),
                    }
                }
                t = o;
            }
            Rule::DictLongest(d) => {
                let offs: Vec<usize> = t.char_indices().map(|(i, _)| i).chain(std::iter::once(t.len())).collect();
                let chars: Vec<char> = t.chars().collect();
                let mut o = String::with_capacity(t.len() * 2);
                let mut i = 0;
                while i < chars.len() {
                    let hit = d.starts.contains(&chars[i]).then(|| {
                        (1..=d.max_chars.min(chars.len() - i)).rev().find_map(|n| d.map.get(&t[offs[i]..offs[i + n]]).map(|v| (n, v)))
                    });
                    match hit.flatten() {
                        Some((n, v)) => {
                            o.push_str(v);
                            i += n;
                        }
                        None => {
                            o.push(chars[i]);
                            i += 1;
                        }
                    }
                }
                t = o;
            }
            Rule::DropChars(r) => t.retain(|c| !r.iter().any(|&(a, b)| (a..=b).contains(&(c as u32)))),
            Rule::Segment(s) => t = s.cut(&t),
        }
    }
    Ok(t)
}

#[cfg(feature = "hf-tokenizer")]
fn nfkd(t: &str) -> Result<String> {
    let mut n = tokenizers::NormalizedString::from(t);
    n.nfkd();
    Ok(n.get().to_string())
}

#[cfg(not(feature = "hf-tokenizer"))]
fn nfkd(_: &str) -> Result<String> {
    Err(RuntimeError::Rejected("text rule `nfkd` needs the hf-tokenizer feature".into()))
}

pub fn validate(rules: &str) -> Result<()> {
    apply(rules, "").map(|_| ())
}

/// Rules without tables or language selection.
pub fn apply(rules: &str, text: &str) -> Result<String> {
    let r = compile(rules, &Tables::default(), &mut HashMap::new())?;
    run(&r, text.to_string(), None, &HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PUNC_NORM: &str = "default_if_empty\tYou need to add some text for me to talk.\ncapitalize_first\ncollapse_whitespace\nreplace\t...\t, \nreplace\t:\t,\nreplace\t\u{201c}\t\"\nreplace\t\u{201d}\t\"\nreplace\t\u{2014}\t-\ntrim_end\t \nensure_suffix\t.!?-,\t.";

    #[test]
    fn rules_reproduce_reference_normalization() {
        assert_eq!(apply(PUNC_NORM, "hello world").unwrap(), "Hello world.");
        assert_eq!(apply(PUNC_NORM, "It was  a bright... day").unwrap(), "It was a bright,  day.");
        assert_eq!(apply(PUNC_NORM, "Wait: what?").unwrap(), "Wait, what?");
        assert_eq!(apply(PUNC_NORM, "").unwrap(), "You need to add some text for me to talk.");
        assert_eq!(apply(PUNC_NORM, "\u{201c}Quote\u{201d} \u{2014} dash").unwrap(), "\"Quote\" - dash.");
        assert!(apply("bogus", "x").is_err());
    }

    fn tables(entries: &[(&str, &str)]) -> Tables {
        let mut b = Vec::new();
        for (n, d) in entries {
            b.extend((n.len() as u32).to_le_bytes());
            b.extend(n.as_bytes());
            b.extend((d.len() as u64).to_le_bytes());
            b.extend(d.as_bytes());
        }
        Tables::parse(&b).unwrap()
    }

    #[test]
    fn language_rules_and_tables() {
        let t = tables(&[("m", "a\tA\nb\t[b]"), ("d", "ab\tX\nabc\tY\nb\t\nc\tC")]);
        let r = TextRules::compile(
            "lowercase\nlanguage\nprefix\t[{lang}]\nreplace\t \t[SPACE]",
            [("xx", "map_chars\tm"), ("yy", "dict_longest\td\ndrop_chars\t1F600-1F64F"), ("en", "")],
            Some("en"),
            &t,
        )
        .unwrap();
        assert_eq!(r.apply("Hi Ab", r.resolve(None).unwrap().as_deref()).unwrap(), "[en]hi[SPACE]ab");
        assert_eq!(r.apply("Hi Ab", Some("xx")).unwrap(), "[xx]hi[SPACE]A[b]");
        assert_eq!(r.apply("abcab b\u{1F600}", Some("yy")).unwrap(), "[yy]YX[SPACE]");
        assert!(r.resolve(Some("zz")).is_err());
        assert_eq!(r.resolve(Some("XX")).unwrap().as_deref(), Some("xx"));
    }

    #[cfg(feature = "hf-tokenizer")]
    #[test]
    fn nfkd_and_lowercase_match_python() {
        // Python: unicodedata.normalize("NFKD", "Ｃafé 한".lower())
        let r = TextRules::compile("lowercase\nnfkd", [], None, &Tables::default()).unwrap();
        assert_eq!(r.apply("Ｃafé 한", None).unwrap(), "cafe\u{301} \u{1112}\u{1161}\u{11ab}");
        assert_eq!(r.apply("ΟΔΟΣ", None).unwrap(), "οδος");
    }
}
