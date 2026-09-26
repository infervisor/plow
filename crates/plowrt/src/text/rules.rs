//! Packet-declared text normalization: one rule per line, tab-separated `kind\targs...`, applied
//! in order. Kinds: `default_if_empty\tTEXT`, `capitalize_first`, `collapse_whitespace`,
//! `replace\tFROM\tTO`, `trim_end\tCHARS`, `ensure_suffix\tCHARS\tSUFFIX` (append SUFFIX unless
//! the text ends with one of CHARS).

use crate::{Result, RuntimeError};

pub fn validate(rules: &str) -> Result<()> {
    apply(rules, "").map(|_| ())
}

pub fn apply(rules: &str, text: &str) -> Result<String> {
    let mut t = text.to_string();
    for line in rules.lines().filter(|l| !l.is_empty()) {
        let mut f = line.split('\t');
        let kind = f.next().unwrap_or_default();
        let args: Vec<&str> = f.collect();
        let arg = |i: usize| {
            args.get(i).copied().ok_or_else(|| RuntimeError::Rejected(format!("text rule {kind:?} needs argument {i}")))
        };
        match kind {
            "default_if_empty" => {
                if t.is_empty() {
                    t = arg(0)?.to_string();
                }
            }
            "capitalize_first" => {
                let mut chars = t.chars();
                if let Some(c0) = chars.next().filter(|c| c.is_lowercase()) {
                    t = c0.to_uppercase().chain(chars).collect();
                }
            }
            "collapse_whitespace" => t = t.split_whitespace().collect::<Vec<_>>().join(" "),
            "replace" => t = t.replace(arg(0)?, arg(1)?),
            "trim_end" => {
                let chars: Vec<char> = arg(0)?.chars().collect();
                t = t.trim_end_matches(chars.as_slice()).to_string();
            }
            "ensure_suffix" => {
                let (ends, suffix) = (arg(0)?, arg(1)?);
                if !t.chars().last().is_some_and(|c| ends.contains(c)) {
                    t.push_str(suffix);
                }
            }
            other => return Err(RuntimeError::Rejected(format!("unknown text rule {other:?}"))),
        }
    }
    Ok(t)
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
}
