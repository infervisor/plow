//! Maya 2 Native text normaliser: numbers, currency, dates, times, OTPs / phone numbers, URLs and e-mails -> spoken
//! words in the request language (as bn en gu hi kn ml mr or pa ta te).
//!
//! A Rust port of `maya3.text.normalise_text(text, lang)` (default arguments: rules on, identifier modes
//! `FR3_TN_URL_SPELL` / `FR3_TN_LONG_DIGITS` off), which wraps Maya Research's text normaliser at upstream commit
//! fe5e6ec (Apache-2.0) with caps folding, `[tag]` protection, a per-piece fallback, the targeted rules R1-R12 / P2-P3
//! and the lexicon fixes L1-L3. `tables.rs` is dumped from the live Python tables (after those patches) and `re.rs`
//! reproduces the Python `re` semantics the patterns rely on. The port is held to byte parity with the Python by
//! `tests::parity` over `tests/data/maya2_text_cases.jsonl` (written by `scripts/tts/maya2_text_cases.py`).

mod lang;
mod normalize;
mod re;
#[rustfmt::skip]
mod tables;

use lang::LangData;
use normalize::{cv, pint, re, sl, split_ws, strip, Engine, R};
use re::{alt_words, is_space, Regex};
use std::sync::OnceLock;
use tables as tb;

/// `maya3.text.normalise_text(text, lang)`.
pub fn normalise_text(text: &str, lang: &str) -> String {
    let tag = re!(r"(\[[a-z ]+\])");
    let segs = tag.split(text);
    let mut out = String::new();
    for seg in &segs {
        if seg.is_empty() || tag.is_full(seg) {
            out += seg;
            continue;
        }
        let s = caps_fold(seg);
        let s2 = piece(&s, seg, lang).unwrap_or(s);
        let lstripped = seg.trim_start_matches(is_space);
        let lead = &seg[..seg.len() - lstripped.len()];
        let trail = &seg[seg.trim_end_matches(is_space).len()..];
        out += lead;
        out += strip(&s2);
        out += trail;
    }
    strip(&re!(r"\s+").replace(&out, " ")).to_string()
}

fn piece(s: &str, seg: &str, lang: &str) -> R<String> {
    let s2 = normalize::normalize(&pre(s, lang), lang)?;
    Ok(post(s2, seg, lang))
}

fn caps_fold(t: &str) -> String {
    re!(r"\b[A-Z]{4,}\b").sub(t, |m| {
        let w = m.all();
        let n = w.len();
        let v = w.chars().filter(|c| "AEIOU".contains(*c)).count();
        if (n >= 5 && v >= 2) || (n == 4 && v >= 2) {
            let mut c = w.chars();
            let f = c.next().unwrap();
            format!("{f}{}", c.as_str().to_lowercase())
        } else {
            w
        }
    })
}

// ------------------------------------------------------------------------------------------------ rules.pre

fn pre(text: &str, lang: &str) -> String {
    let Some(eng) = normalize::table(lang) else {
        return text.to_string();
    };
    let t = cv(text);
    let mut out = String::new();
    let mut pos = 0;
    for m in re!(tb::RE_URI).find_iter(&t) {
        out += &pre_part(&sl(&t, pos, m.start()), lang, eng);
        out += &m.all();
        pos = m.end();
    }
    out += &pre_part(&sl(&t, pos, t.len()), lang, eng);
    out
}

fn tn_id_context(l: &LangData, before: &str) -> bool {
    let toks = normalize::id_toks(&before.replace('\u{200c}', "").to_lowercase());
    toks[toks.len().saturating_sub(3)..].iter().any(|t| {
        tb::ID_KEYWORDS_EN.contains(&t.as_str())
            || tb::OTP_INPUT_WORDS.contains(&t.as_str())
            || l.id_keywords.iter().any(|k| k.to_lowercase() == *t)
    })
}

fn digits(l: &LangData, s: &str) -> String {
    s.chars()
        .map(|c| l.digit(re::decimal_value(c).unwrap()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn decimal_word(l: &LangData) -> String {
    let w = l.decimal(1, "2");
    let (a, b) = (l.cardinal(1), l.digit(2));
    if w.starts_with(&a) && w.ends_with(b) && w.len() >= a.len() + b.len() {
        strip(&w[a.len()..w.len() - b.len()]).to_string()
    } else if w.starts_with(&a) && w.ends_with(b) {
        String::new()
    } else {
        "point".into()
    }
}

fn int(s: &str) -> u128 {
    pint(s).unwrap_or(0)
}

fn pre_part(text: &str, lang: &str, eng: &Engine) -> String {
    let l = eng.l;
    let mut t = text.to_string();

    // R10
    t = re!(r"(?<![\d.])(\d+(?:\.\d+)?)\s?%\s?[-–]\s?(\d+(?:\.\d+)?)\s?%")
        .sub(&t, |m| format!("{}-{}%", m.s(1), m.s(2)));

    // R11
    let per = l.sym_get("per").filter(|p| !p.is_empty()).unwrap_or("per");
    t = re!(
        r"((?:[₹$€£]|\b(?:Rs\.?|INR)\s?)\s?\d[\d,]*(?:\.\d+)?)\s?/\s?(kg|km|gm|g|ml|ltr|l|day|month|night|year|person)\b",
        i
    )
    .sub(&t, |m| {
        let mut unit = m.s(2);
        if lang != "en" && tb::SPELL_UNITS.contains(&unit.to_lowercase().as_str()) {
            unit = unit.to_uppercase().chars().map(|c| l.letter_c(c)).collect::<Vec<_>>().join(" ");
        }
        format!("{} {per} {unit}", m.s(1))
    });

    // R12
    t = re!(r"(?<![\w.])[pP]\.\s?[aA]\.?(?![\w])").sub(&t, |_| {
        if lang == "en" {
            "per annum".into()
        } else {
            format!("{} {}", l.letter("P"), l.letter("A"))
        }
    });

    // R7
    t = re!(r"(?<![\w.])[vV](\d+(?:\.\d+){1,3})(?![\w]|\.\d)").sub(&t, |m| {
        let dw = decimal_word(l);
        let parts: Vec<String> = m.s(1).split('.').map(|p| l.cardinal(int(p))).collect();
        format!("{} {}", l.letter("v"), parts.join(&format!(" {dw} ")))
    });

    // R4
    t = re!(r"(?<![\d,.])(?:\d{4,5}(?: \d{3,5}){1,3}|\d{2,4} \d{3,4} \d{4})(?![\d,]|\.\d)").sub(
        &t,
        |m| {
            let all = m.all();
            let groups = split_ws(&all);
            if groups.iter().map(|g| g.chars().count()).sum::<usize>() >= 10 {
                groups.concat()
            } else {
                all.clone()
            }
        },
    );

    // R1
    let tc = cv(&t);
    t = re!(r"(?<![\d/.\-])(\d{1,2})([/.\-])(\d{1,2})\2(\d{2})(?![\d/\-]|\.\d)").sub_chars(
        &tc,
        |m| {
            let (d, sep, mo, yy) = (m.s(1), m.s(2), m.s(3), m.s(4));
            if !((1..=31).contains(&int(&d)) && (1..=12).contains(&int(&mo))) {
                return m.all();
            }
            if sep == "-" && d.chars().count() == 1 && mo.chars().count() == 1 {
                return m.all();
            }
            let before = cv(&sl(&tc, m.start().saturating_sub(16), m.start()));
            if re!(tb::RE_NUMBER_WORD_BEFORE, i).search(&before).is_some() {
                return m.all();
            }
            let y = int(&yy);
            format!(
                "{d}{sep}{mo}{sep}{}",
                if y <= 50 { 2000 + y } else { 1900 + y }
            )
        },
    );

    // R2
    let tc = cv(&t);
    t = re!(r"(?<![\d/.\-])(0?[1-9]|1[0-2])/(\d{2}|20\d{2})(?![\d/]|\.\d)").sub_chars(&tc, |m| {
        let (mo, y) = (int(&m.s(1)), m.s(2));
        let before = sl(&tc, m.start().saturating_sub(40), m.start());
        let after = sl(&tc, m.end(), m.end() + 40);
        let strong = re!(tb::RE_EXPIRY, i).found(&before)
            || re!(r"\bcvv\b", i).found(&format!("{before} {after}"));
        let ylen = y.chars().count();
        let weak = re!(tb::RE_DATE_WORD, i).found(&before) && (ylen == 4 || int(&y) > 12);
        if !(strong || weak) {
            return m.all();
        }
        let year = if ylen == 4 { int(&y) } else { 2000 + int(&y) };
        format!("{} {}", l.month(mo as u32), l.year(year))
    });

    // R3
    t = re!(r"(?<![\d\-])((?:19|20)\d{2})[-–](\d{2})(?![\d\-])").sub(&t, |m| {
        let (y, yy) = (int(&m.s(1)), int(&m.s(2)));
        if (y + 1) % 100 != yy {
            return m.all();
        }
        format!("{} {}", l.year(y), l.cardinal(yy))
    });

    // R5
    let tc = cv(&t);
    t = re!(tb::RE_PIN, i).sub_chars(&tc, |m| {
        let s1 = m.start_of(1);
        if tn_id_context(l, &sl(&tc, s1.saturating_sub(60), s1)) {
            return m.all();
        }
        format!("{}{}", sl(&tc, m.start(), s1), digits(l, &m.s(1)))
    });

    // R6
    let tc = cv(&t);
    t = re!(r"(?<![\d,.₹$#\w])[1-9]\d{5}(?![\d,]|\.\d|\s*%)").sub_chars(&tc, |m| {
        let before = sl(&tc, m.start().saturating_sub(8), m.start());
        let sent = sl(&tc, m.start().saturating_sub(60), m.start());
        let sent = sent
            .rsplit(['.', '!', '?', ';', '।', '॥', '\n'])
            .next()
            .unwrap_or("");
        if re!(tb::RE_CURRENCY_BEFORE, i).found(&before)
            || re!(tb::RE_AMOUNT_AFTER, i).match_at(&tc, m.end()).is_some()
            || re!(r"[A-Za-z0-9]\s?-\s?$|[#÷×*/+=<>]\s*$").found(&before)
            || re!(r"\s*[÷×*/+=<>^]").match_at(&tc[m.end()..], 0).is_some()
            || tn_id_context(l, sent)
        {
            return m.all();
        }
        digits(l, &m.all())
    });

    // R8
    if lang == "ta" {
        static R8: OnceLock<Regex> = OnceLock::new();
        let rx = R8.get_or_init(|| {
            Regex::new(&format!(
                r"(?<!\w)({})\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)",
                tb::TA_MONTHS.join("|")
            ))
        });
        t = rx.sub(&t, |m| format!("{} {} {}", m.s(2), m.s(1), m.s(3)));
    }
    t
}

// ------------------------------------------------------------------------------------------------ rules.post

fn post(mut out: String, seg: &str, lang: &str) -> String {
    let sc = cv(seg);
    let uris: Vec<String> = re!(tb::RE_URI)
        .find_iter(&sc)
        .iter()
        .map(|m| m.all())
        .collect();
    if let Some(d) = lang::get(tb::DASH, lang) {
        if !uris.is_empty() {
            out = re!(r"(?<=\s)dash(?=\s)").replace(&out, d);
        }
    }
    let mut labels: Vec<String> = Vec::new();
    for u in &uris {
        let body = re!(r"^https?://").replace(u, "");
        for w in re!(r"[a-z]+").find_iter(&cv(&body)) {
            let w = w.all();
            if tb::URI_ACRONYMS.contains(&w.as_str())
                || tb::URI_ACR_PREFIX.iter().any(|a| w.starts_with(a))
                || tb::URI_ACR_SUFFIX.iter().any(|a| w.ends_with(a))
            {
                labels.push(w);
            }
        }
    }
    if labels.is_empty() {
        return out;
    }
    let Some(eng) = normalize::table(lang) else {
        return out;
    };
    let l = eng.l;
    let spell = |w: &str| {
        w.chars()
            .map(|c| l.letter(&c.to_uppercase().to_string()))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let rx = Regex::new(&format!(
        r"(?<![\w.@/])(?:{})(?![\w]|[.@/\-]\w)",
        alt_words(labels.iter())
    ));
    rx.sub(&out, |m| {
        let w = m.all();
        if tb::URI_ACRONYMS.contains(&w.as_str()) {
            return spell(&w);
        }
        for a in tb::URI_ACR_PREFIX {
            if w.starts_with(a) && w.len() - a.len() >= 3 {
                return format!("{} {}", spell(a), &w[a.len()..]);
            }
        }
        for a in tb::URI_ACR_SUFFIX {
            if w.ends_with(a) && w.len() - a.len() >= 4 {
                return format!("{} {}", &w[..w.len() - a.len()], spell(a));
            }
        }
        w
    })
}

#[cfg(test)]
mod tests {
    use super::normalise_text;

    #[test]
    fn readme_examples() {
        assert_eq!(
            normalise_text("Your OTP is 482913 [laughs]", "en"),
            "Your OTP is four, eight, two, nine, one, three [laughs]"
        );
    }

    /// Byte parity with the Python normaliser over the corpus written by scripts/tts/maya2_text_cases.py.
    #[test]
    fn parity() {
        let path = std::env::var("MAYA2_TEXT_CASES").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/maya2_text_cases.jsonl"
            )
            .into()
        });
        let data = std::fs::read_to_string(&path).expect("corpus");
        let mut per: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        let mut bad = Vec::new();
        let t0 = std::time::Instant::now();
        for line in data.lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let (lang, text, want) = (
                v["lang"].as_str().unwrap(),
                v["text"].as_str().unwrap(),
                v["expected"].as_str().unwrap(),
            );
            let got = normalise_text(text, lang);
            let e = per.entry(lang.to_string()).or_default();
            e.1 += 1;
            if got == want {
                e.0 += 1;
            } else {
                bad.push(format!(
                    "[{lang}] {text:?}\n   want {want:?}\n    got {got:?}"
                ));
            }
        }
        let el = t0.elapsed();
        for (l, (ok, n)) in &per {
            eprintln!("{l}: {ok}/{n}");
        }
        let total: usize = per.values().map(|x| x.1).sum();
        eprintln!("total {}/{total} in {el:?}", total - bad.len());
        for b in bad.iter().take(40) {
            eprintln!("{b}");
        }
        assert!(bad.is_empty(), "{} of {total} cases differ", bad.len());
    }
}
