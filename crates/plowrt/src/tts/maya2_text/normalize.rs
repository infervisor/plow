//! maya3.text.tn.normalize: validation, pre-clean, the prioritised span matchers and the final sweep.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::lang::{fmt, get, LangData};
use super::re::{
    alt_words, decimal_value, escape, is_alnum, is_alpha, is_decimal, is_digit, is_space, is_word,
    Regex,
};
use super::tables as tb;

/// A Python exception: `NormalizationError` (with its span / source when it has one) or anything else.
pub enum Exc {
    Norm {
        span: Option<(usize, usize)>,
        source: Option<Vec<char>>,
    },
    Other,
}
pub type R<T> = Result<T, Exc>;

fn nerr(span: (usize, usize)) -> Exc {
    Exc::Norm {
        span: Some(span),
        source: None,
    }
}
fn nerr_src(span: (usize, usize), t: &[char]) -> Exc {
    Exc::Norm {
        span: Some(span),
        source: Some(t.to_vec()),
    }
}
const NERR: Exc = Exc::Norm {
    span: None,
    source: None,
};

macro_rules! re {
    ($p:expr) => {{
        static RX: OnceLock<Regex> = OnceLock::new();
        RX.get_or_init(|| Regex::new($p))
    }};
    ($p:expr, i) => {{
        static RX: OnceLock<Regex> = OnceLock::new();
        RX.get_or_init(|| Regex::new_i($p))
    }};
}
pub(crate) use re;

pub fn cv(s: &str) -> Vec<char> {
    s.chars().collect()
}
pub fn st(c: &[char]) -> String {
    c.iter().collect()
}
/// Python `t[a:b]` (clamped).
pub fn sl(t: &[char], a: usize, b: usize) -> String {
    let b = b.min(t.len());
    if a >= b {
        String::new()
    } else {
        st(&t[a..b])
    }
}
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}
pub fn split_ws(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|x| !x.is_empty()).collect()
}

/// Python `int(s)` for a run of Unicode decimal digits.
pub fn pint(s: &str) -> R<u128> {
    if s.is_empty() {
        return Err(Exc::Other);
    }
    let mut v: u128 = 0;
    for c in s.chars() {
        let d = decimal_value(c).ok_or(Exc::Other)?;
        v = v
            .checked_mul(10)
            .and_then(|v| v.checked_add(d as u128))
            .ok_or(Exc::Other)?;
    }
    Ok(v)
}
fn pint_or0(s: &str) -> R<u128> {
    if s.is_empty() {
        Ok(0)
    } else {
        pint(s)
    }
}

/// `unicodedata.decimal`-based `_ascii_digits`.
pub fn ascii_digits(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii() {
                c
            } else {
                decimal_value(c).map_or(c, |d| (b'0' + d as u8) as char)
            }
        })
        .collect()
}

const SENT: &[char] = &['.', '!', '?', ';', '।', '॥', '\n'];

fn sentence_before(t: &[char], start: usize) -> String {
    let w = &t[start.saturating_sub(100)..start];
    let i = w
        .iter()
        .rposition(|c| SENT.contains(c))
        .map_or(0, |i| i + 1);
    st(&w[i..])
}
fn sentence_after(t: &[char], end: usize) -> String {
    let w = &t[end.min(t.len())..(end + 100).min(t.len())];
    let i = w.iter().position(|c| SENT.contains(c)).unwrap_or(w.len());
    st(&w[..i])
}

/// `re.findall(r"[\w#ऀ-෿']+", s)`.
pub fn id_toks(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if is_word(c) || c == '#' || c == '\'' || ('\u{900}'..='\u{dff}').contains(&c) {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}
fn word_toks(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if is_word(c) {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}
fn last_n<T>(v: &[T], n: usize) -> &[T] {
    &v[v.len().saturating_sub(n)..]
}

fn set(xs: &[&str]) -> HashSet<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

fn calendar_ok(day: u128, month: u128, year: u128) -> bool {
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) || day < 1 {
        return false;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let dim = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ][month as usize - 1];
    day <= dim
}

fn valid_grouping(number: &str) -> bool {
    let integer = number.trim_start_matches(['+', '-', '−']);
    let integer = integer.split('.').next().unwrap();
    !integer.contains(',')
        || re!(r"[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,2}(?:,[0-9]{2})*,[0-9]{3}").is_full(integer)
}

type Span = (usize, usize, String);

pub struct Engine {
    pub l: &'static LangData,
    month_lookup: HashMap<String, u32>,
    v_month_a: Regex,
    v_month_b: Regex,
    v_months: HashMap<String, u32>,
    v_rupee_paise: Regex,
    v_rupee_pct: Regex,
    time_pat: Regex,
    time_range: Regex,
    time_verbal: Regex,
    date_range: Regex,
    date_dmon: Regex,
    date_mon_d: Regex,
    cur_range: Regex,
    currency: Regex,
    amount: Regex,
    ordinal: Regex,
    label: Regex,
    number_word: Option<Regex>,
    abbrev: Vec<(Regex, &'static str)>,
    id_keywords: HashSet<String>,
    label_words: HashSet<String>,
    ratio_kws: HashSet<String>,
    roman_ctx: HashSet<String>,
    year_ctx: HashSet<String>,
    year_words2: HashSet<String>,
    id_after: HashSet<String>,
}

impl Engine {
    pub fn new(l: &'static LangData) -> Engine {
        let period_re = alt_words(l.period_detect.iter().map(|x| x.0));
        let marker_re = alt_words(l.time_markers.iter().copied());
        // dict(**_ENGLISH_MONTHS, **{native lower: i}, **{alias lower: i}); insertion order kept for the alternation
        let mut keys: Vec<String> = Vec::new();
        let mut month_lookup = HashMap::new();
        let mut put = |k: String, v: u32, keys: &mut Vec<String>| {
            if month_lookup.insert(k.clone(), v).is_none() {
                keys.push(k);
            }
        };
        for (k, v) in tb::ENGLISH_MONTHS {
            put(k.to_string(), *v, &mut keys);
        }
        for (i, m) in l.months.iter().enumerate() {
            put(m.to_lowercase(), i as u32 + 1, &mut keys);
        }
        for (k, v) in l.month_aliases {
            put(k.to_lowercase(), v.parse().unwrap(), &mut keys);
        }
        let month_re = alt_words(keys.iter());
        let mut scale: Vec<&str> = l.scale_words.iter().map(|x| x.0).collect();
        scale.extend([
            "crore", "crores", "Crore", "Crores", "Cr", "cr", "lakh", "lakhs", "Lakh", "Lakhs",
            "thousand",
        ]);
        scale.extend(tb::EN_SCALES.iter());
        let scale_re = alt_words(scale);
        let mut ords: Vec<&str> = l.ordinal_input_suffixes.to_vec();
        ords.extend(["st", "nd", "rd", "th"]);
        let ord_re = alt_words(ords);
        let native: Vec<&str> = l
            .currency_words
            .iter()
            .filter(|(w, k)| !w.is_ascii() && matches!(*k, "rupee" | "dollar" | "euro" | "pound"))
            .map(|x| x.0)
            .collect();
        let mut currency_re = r"₹|\$|€|£|Rs\.?|INR|USD|EUR|GBP".to_string();
        if !native.is_empty() {
            currency_re += &format!("|(?:{})\\.?", alt_words(native));
        }
        let mut id_keywords = set(tb::ID_KEYWORDS_EN);
        id_keywords.extend(tb::OTP_INPUT_WORDS.iter().map(|s| s.to_string()));
        id_keywords.extend(l.id_keywords.iter().map(|w| w.to_lowercase()));
        let mut label_words = set(tb::LABEL_WORDS_EN);
        label_words.extend(l.label_words.iter().map(|w| w.to_lowercase()));
        let minute_re = alt_words(
            l.minute_words
                .iter()
                .copied()
                .chain(["minute", "minutes", "min"]),
        );
        let mut ratio_kws = set(&["ratio", "odds", "proportion", "अनुपात"]);
        let tpl = l.ratio.replace("{a}", " ").replace("{b}", " ");
        ratio_kws.extend(split_ws(&tpl).iter().map(|w| w.to_lowercase()));
        let mut roman_ctx = label_words.clone();
        roman_ctx.extend(set(&[
            "war",
            "vol",
            "part",
            "act",
            "book",
            "youdh",
            "युद्ध",
            "अध्याय",
            "कक्षा",
            "भाग",
            "खंड",
        ]));
        let yw: HashSet<String> = l.year_words.iter().map(|w| w.to_lowercase()).collect();
        let mut year_ctx = yw.clone();
        year_ctx.extend(set(&["year", "in", "since", "by", "from", "साल", "वर्ष"]));
        let mut year_words2 = yw;
        year_words2.insert("year".into());
        let mut id_after = set(&["dial", "डायल"]);
        id_after.extend(l.id_after_words.iter().map(|s| s.to_string()));

        // validate_text's month tables (Tamil months first, then the locale lookup)
        let mut v_months: HashMap<String, u32> = HashMap::new();
        let mut v_keys: Vec<String> = Vec::new();
        for (i, m) in tb::TAMIL_MONTHS.iter().enumerate() {
            v_months.insert(m.to_string(), i as u32 + 1);
            v_keys.push(m.to_string());
        }
        for k in &keys {
            if v_months.insert(k.clone(), month_lookup[k]).is_none() {
                v_keys.push(k.clone());
            }
        }
        let v_month_re = alt_words(v_keys.iter());
        let mut v_ords: Vec<&str> = l.ordinal_input_suffixes.to_vec();
        v_ords.extend(["st", "nd", "rd", "th", "ஆம்", "ம்"]);
        let v_ord_re = alt_words(v_ords);
        let mut rupee_words: Vec<&str> = l
            .currency_words
            .iter()
            .filter(|(_, k)| *k == "rupee")
            .map(|x| x.0)
            .collect();
        rupee_words.extend(["INR", "Rs"]);
        let mut esc: Vec<String> = rupee_words.iter().map(|w| escape(w)).collect();
        esc.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
        let rupee_prefix = format!(r"(?:₹|(?<!\w)(?:{})\.?)", esc.join("|"));
        let mut minor: Vec<String> = tb::PAISA_INPUT_WORDS.iter().map(|w| escape(w)).collect();
        minor.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
        let minor = minor.join("|");

        let p = &period_re;
        let mk = &marker_re;
        let ampm = r"AM|PM|am|pm|a\.m\.|p\.m\.";
        let time = format!(
            r"(?:(?<!\w)({p})\s+)?(\d{{1,2}})(?=:\d{{2}}|\.\d{{2}}\s*(?:{ampm})(?![A-Za-z])|\s*(?:{ampm})(?![A-Za-z])|\s*(?:{mk})(?!\w))(?:[.:](\d{{2}}))?(?:\s*({ampm}))?(?:\s*({mk})(?!\w))?(?:\s+({p})(?!\w))?"
        );
        let lbl = {
            let mut v: Vec<&String> = label_words.iter().collect();
            v.sort();
            alt_words(v)
        };
        let mut abbrev = Vec::new();
        for (ab, word) in l.abbrev_units {
            abbrev.push((
                Regex::new_i(&format!(r"(?<![\w-]){ab}(?:\.(?=\s+\S))?(?![\w-])")),
                *word,
            ));
        }
        let mut units = String::from(
            r"rupees?|rs\.?|inr|dollars?|usd|euros?|pounds?|paise|paisa|cents?|ரூபாய்|பைசா|रुपये|रुपया|रुपए|రూ\.?|రూపాయలు|രൂപ",
        );
        units += "|";
        units += &l
            .currency_words
            .iter()
            .map(|x| escape(x.0))
            .collect::<Vec<_>>()
            .join("|");
        let sc = &scale_re;
        let mr_case = "(?:\u{93e}\u{902}|\u{902})?(?:\u{91a}\u{93e}|\u{91a}\u{940}|\u{91a}\u{947}|\u{91a}\u{94d}\u{92f}\u{93e}|\u{924}|\u{928}\u{93e}|\u{939}\u{942}\u{928}|\u{92a}\u{930}\u{94d}\u{92f}\u{902}\u{924}|\u{92e}\u{927}\u{94d}\u{92f}\u{947})";
        let marks = r"[ऀ-ःऺ-ॏঀ-ঃা-্ਁ-ਃਾ-੍ઁ-ઃા-્ଁ-ଃା-୍ஂா-்ఀ-ఄా-ౌಁ-ಃಾ-ೌഀ-ഃാ-ൌ]";
        let num = r"(\d+(?:,\d{2,3})*(?:\.\d+)?)";
        let snum = r"([+-]?\d+(?:,\d{2,3})*(?:\.\d+)?)";
        Engine {
            l,
            v_month_a: Regex::new_i(&format!(
                r"(?<!\w)([0-9]{{1,2}})(?:-?(?:{v_ord_re}))?[ -]+({v_month_re})(?!\w)(?:[ ,\-]+([0-9]{{4}}))?"
            )),
            v_month_b: Regex::new_i(&format!(
                r"(?<!\w)({v_month_re})[ -]+([0-9]{{1,2}})(?:-?(?:{v_ord_re}))?(?!\d)(?:,?\s+([0-9]{{4}}))?"
            )),
            v_months,
            v_rupee_paise: Regex::new_i(&format!(
                r"{rupee_prefix}\s*[+-]?[0-9][0-9,.]*\s*(?:{minor})(?=$|[\s.,!?;।॥-])"
            )),
            v_rupee_pct: Regex::new_i(&format!(
                r"(?:{rupee_prefix}|[$€£]|\b(?:USD|EUR|GBP))\s*[+-]?[0-9][0-9,.]*\s*%"
            )),
            time_range: Regex::new_i(&format!(r"({time})\s*[–\-]\s*({time})")),
            time_pat: Regex::new_i(&time),
            time_verbal: Regex::new(&format!(
                r"\b(\d{{1,2}})\s*(?:{mk})\s+(\d{{1,2}})\s*({minute_re})(?!\w)"
            )),
            date_range: Regex::new_i(&format!(
                r"\b(\d{{1,2}})[-–](\d{{1,2}})\s+({month_re})(?!\w)(?:\s+(\d{{4}}))?"
            )),
            date_dmon: Regex::new_i(&format!(
                r"\b(\d{{1,2}})(?:-?({ord_re}))?[\-\s]({month_re})(?!\w)(?:[\-\s,]+(\d{{4}}))?"
            )),
            date_mon_d: Regex::new_i(&format!(
                r"\b({month_re})\s+(\d{{1,2}})(?:-?({ord_re}))?(?:,?\s+(\d{{4}}))?(?!\d)"
            )),
            cur_range: Regex::new_i(&format!(
                r"(?<!\w)([+-]?)({currency_re})\s*{snum}\s*[–\-]\s*{snum}(?!\d|[.,]\d)"
            )),
            currency: Regex::new_i(&format!(
                r"(?<!\d)([+-]?)\s*(₹|\$|€|£|(?<!\w)(?:{currency_re}))\s*([+-]?){num}(?:\s*({sc})(?:({mr_case})|(?!{marks}))(?![A-Za-z0-9]))?(?:\s?/-)?"
            )),
            amount: Regex::new_i(&format!(
                r"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)(?=\s*(?:{units})(?![A-Za-z]))"
            )),
            ordinal: Regex::new_i(&format!(r"\b(\d+)-?({ord_re})(?![A-Za-z0-9])")),
            label: Regex::new_i(&format!(
                r"\b({lbl})\s+(?:(\d+)(?![\d.]|,\d)(?!-[A-Za-z]{{2,}})(?:[-\s]?([A-Za-z])\b)?|([A-Za-z])\b(?:[-\s]?(\d+))?)"
            )),
            number_word: (!l.number_word.is_empty())
                .then(|| Regex::new(r"\b[Nn][Oo]s?\.\s*(?=\d)")),
            abbrev,
            month_lookup,
            id_keywords,
            label_words,
            ratio_kws,
            roman_ctx,
            year_ctx,
            year_words2,
            id_after,
        }
    }

    fn month_index(&self, tok: &str) -> Option<u32> {
        self.month_lookup.get(&strip(tok).to_lowercase()).copied()
    }
    fn month_py(&self, i: u128) -> R<&'static str> {
        match i {
            0 => Ok(self.l.months[11]),
            1..=12 => Ok(self.l.months[i as usize - 1]),
            _ => Err(Exc::Other),
        }
    }

    // ---------------------------------------------------------------- validate_text
    pub fn validate(&self, text: &str) -> R<()> {
        let tc = cv(text);
        if tc.len() > 20000 {
            return Err(NERR);
        }
        let t = cv(&ascii_digits(text));
        if let Some(m) = re!(r"<\s*(?:/?[A-Za-z_]|[!?])|[{}]|\$\{").search(&t) {
            return Err(nerr(m.span(0)));
        }
        for (i, &c) in t.iter().enumerate() {
            if (is_cc(c) && !"\n\r\t".contains(c)) || is_co(c) {
                return Err(nerr((i, i + 1)));
            }
        }
        let masked = re!(r"https?://\S+|www\.\S+|\b[\w.%+\-]+@[\w.\-]+\.[A-Za-z]{2,}")
            .sub_chars(&t, |m| " ".repeat(m.end() - m.start()));
        let t = cv(&masked);
        if let Some(m) = re!(r"[0-9]{31,}").search(&t) {
            return Err(nerr(m.span(0)));
        }
        if let Some(m) = re!(r"\d{4}-\d{2}-\d{2}T\d", i).search(&t) {
            return Err(nerr(m.span(0)));
        }
        for m in re!(r"(?<![\w.])[+-]?[0-9][0-9,]*(?:\.[0-9]+)?").find_iter(&t) {
            let s = m.all();
            if s.chars().filter(|c| is_decimal(*c)).count() > 30 {
                return Err(nerr(m.span(0)));
            }
            if !valid_grouping(s.trim_end_matches(',')) {
                return Err(nerr(m.span(0)));
            }
        }
        for m in re!(r"(?<![\w\d])([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})(?!\d)").find_iter(&t) {
            if !calendar_ok(pint(&m.s(3))?, pint(&m.s(2))?, pint(&m.s(1))?) {
                return Err(nerr(m.span(0)));
            }
        }
        for m in
            re!(r"(?<![\w\d-])([0-9]{1,2})([/.-])([0-9]{1,2})\2([0-9]{2,4})(?!\d)").find_iter(&t)
        {
            if m.s(4).chars().count() != 4 {
                return Err(nerr(m.span(0)));
            }
            if !calendar_ok(pint(&m.s(1))?, pint(&m.s(3))?, pint(&m.s(4))?) {
                return Err(nerr(m.span(0)));
            }
        }
        for (rx, rev) in [(&self.v_month_a, false), (&self.v_month_b, true)] {
            for m in rx.find_iter(&t) {
                let (day, month) = if rev {
                    (m.s(2), m.s(1))
                } else {
                    (m.s(1), m.s(2))
                };
                let mo = *self.v_months.get(&month.to_lowercase()).ok_or(Exc::Other)?;
                let y = if m.has(3) { pint(&m.s(3))? } else { 2000 };
                if !calendar_ok(pint(&day)?, mo as u128, y) {
                    return Err(nerr(m.span(0)));
                }
            }
        }
        let tamil = re!(&format!(
            r"({})\s+(\d{{1,2}})(?:-?(?:ஆம்|ம்))?(?:\s*தேதி)?(?:\s+(\d{{4}}))?",
            tb::TAMIL_MONTHS.join("|")
        ));
        for m in tamil.find_iter(&t) {
            let mo = self.v_months[&m.s(1)];
            let y = if m.has(3) { pint(&m.s(3))? } else { 2000 };
            if !calendar_ok(pint(&m.s(2))?, mo as u128, y) {
                return Err(nerr(m.span(0)));
            }
        }
        if let Some(m) = re!(
            r"\b(?:on|by|before|after|date|due)\s+[0-9]{1,2}/[0-9]{1,2}(?![/\d])",
            i
        )
        .search(&t)
        {
            return Err(nerr(m.span(0)));
        }
        if let Some(m) = re!(r"தேதி\s+[0-9]{1,2}\.\s*[0-9]{4}").search(&t) {
            return Err(nerr(m.span(0)));
        }
        if let Some(m) = self.v_rupee_paise.search(&t) {
            return Err(nerr(m.span(0)));
        }
        if let Some(m) = self.v_rupee_pct.search(&t) {
            return Err(nerr(m.span(0)));
        }
        for m in re!(r"(?<![\w:])([0-9]{1,3}):([0-9]{2}):([0-9]{2})(?![\d:])").find_iter(&t) {
            if pint(&m.s(2))? > 59 || pint(&m.s(3))? > 59 {
                return Err(nerr(m.span(0)));
            }
        }
        let nd =
            cv(&re!(r"[0-9]+:[0-9]{2}:[0-9]{2}")
                .sub_chars(&t, |m| " ".repeat(m.end() - m.start())));
        let clock = re!(
            r"(?<![\w:])([0-9]{1,2})(?:[:.]([0-9]{2}))?\s*(a\.?m\.?|p\.?m\.?)(?!\w)|(?<![\w:])([0-9]{1,2}):([0-9]{2})(?![\d:])",
            i
        );
        for m in clock.find_iter(&nd) {
            let valid = if m.has(1) {
                let h = pint(&m.s(1))?;
                (1..=12).contains(&h) && pint_or0(&m.s(2))? < 60
            } else {
                if re!(r"\b(?:ratio|odds|proportion)\b", i).found(&sentence_before(&t, m.start())) {
                    continue;
                }
                pint(&m.s(4))? < 24 && pint(&m.s(5))? < 60
            };
            if !valid {
                return Err(nerr(m.span(0)));
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------- helpers
    fn digits_spelled(&self, digits: &str, comma: bool) -> R<String> {
        let mut w = Vec::new();
        for c in digits.chars() {
            w.push(self.l.digit(decimal_value(c).ok_or(Exc::Other)?));
        }
        Ok(w.join(if comma { ", " } else { " " }))
    }
    fn letters_spelled(&self, letters: &str) -> String {
        letters
            .chars()
            .map(|c| self.l.letter_c(c))
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn spell_alnum(&self, tok: &str, comma: bool) -> R<String> {
        let mut parts = Vec::new();
        for ch in tok.chars() {
            if is_digit(ch) {
                parts.push(
                    self.l
                        .digit(decimal_value(ch).ok_or(Exc::Other)?)
                        .to_string(),
                );
            } else if is_alpha(ch) && ch.is_ascii() {
                parts.push(self.l.letter_c(ch));
            }
        }
        Ok(parts.join(if comma { ", " } else { " " }))
    }
    fn spell_model_segment(&self, seg: &str) -> R<String> {
        let mut out = Vec::new();
        for run in re!(r"[A-Za-z]+|\d+").find_iter(&cv(seg)) {
            let run = run.all();
            if run.chars().all(is_digit) {
                let n = pint(&run)?;
                if run.chars().count() == 1 || (n % 100 == 0 && n >= 100) {
                    out.push(self.l.cardinal(n));
                } else {
                    out.push(self.digits_spelled(&run, false)?);
                }
            } else {
                out.push(self.letters_spelled(&run));
            }
        }
        Ok(out.join(" "))
    }
    pub fn number_words(&self, s: &str) -> R<String> {
        if let Some(rest) = s.strip_prefix(['-', '+']) {
            let sign = self
                .l
                .sym(if s.starts_with('-') { "minus" } else { "plus" });
            return Ok(format!("{sign} {}", self.number_words(rest)?));
        }
        let s = s.replace(',', "");
        if let Some((ip, fp)) = s.split_once('.') {
            return self.decimal(pint_or0(ip)?, fp);
        }
        Ok(self.l.cardinal(pint(&s)?))
    }
    fn decimal(&self, ip: u128, fp: &str) -> R<String> {
        if !fp.chars().all(|c| c.is_ascii_digit()) {
            return Err(Exc::Other);
        }
        Ok(self.l.decimal(ip, fp))
    }
    fn keyword_before(
        &self,
        t: &[char],
        start: usize,
        kws: &HashSet<String>,
        window: usize,
    ) -> bool {
        let before = sentence_before(t, start)
            .replace('\u{200c}', "")
            .to_lowercase();
        let toks = id_toks(&before);
        last_n(&toks, window)
            .iter()
            .any(|tok| kws.contains(tok.trim_matches(['#', '\''])))
    }
    fn time_words(&self, h: u32, m: u32, period: Option<&str>) -> String {
        let l = self.l;
        if l.code != "en" && (h == 0 || h > 12) {
            let tpl = if m != 0 { l.time_hm } else { l.time_on_hour };
            return fmt(
                &fmt(tpl, "h", &l.cardinal(h as u128)),
                "m",
                &l.cardinal(m as u128),
            );
        }
        l.time_phrase(h, m, period)
    }
    fn period_for_hour24(h: u32) -> &'static str {
        match h {
            4..=11 => "morning",
            12..=15 => "afternoon",
            16..=19 => "evening",
            _ => "night",
        }
    }
    fn resolve_time(
        &self,
        lead: Option<&str>,
        h: &str,
        mm: Option<&str>,
        ampm: Option<&str>,
        trail: Option<&str>,
    ) -> R<(u32, u32, Option<&'static str>)> {
        let l = self.l;
        let h = pint(h)? as u32;
        let m = match mm {
            Some(x) => pint(x)? as u32,
            None => 0,
        };
        let mut period: Option<&'static str> = None;
        let src = lead.or(trail);
        if let Some(sp) = src {
            period = get(l.period_detect, sp);
        }
        if l.code != "en" && ampm.is_none() && src.is_none() {
            return Ok((h, m, None));
        }
        let infer = l.ampm_words.is_empty();
        if let Some(a) = ampm {
            let a = a.to_lowercase().replace('.', "");
            let h24 = h % 12 + if a == "pm" { 12 } else { 0 };
            if infer {
                period = period.or(Some(Self::period_for_hour24(h24)));
            }
        } else if (h > 12 || h == 0) && infer {
            period = period.or(Some(Self::period_for_hour24(h)));
        }
        if !l.ampm_words.is_empty() && ampm.is_none() && (h > 12 || h == 0) {
            return Ok((h, m, period));
        }
        Ok((if h % 12 == 0 { 12 } else { h % 12 }, m, period))
    }
    fn ampm_word(&self, a: &str) -> Option<&'static str> {
        get(self.l.ampm_words, &a.to_lowercase().replace('.', ""))
    }

    fn date_words(
        &self,
        day: u128,
        month: u128,
        year: Option<u128>,
        ordinal: Option<&str>,
        month_first: bool,
    ) -> R<String> {
        let l = self.l;
        let out = if l.code == "ta" {
            let o = tb::TAMIL_ORDINALS.get(day as usize).ok_or(Exc::Other)?;
            format!("{} {o} தேதி", self.month_py(month)?)
        } else {
            let ord = ordinal.filter(|o| !o.is_empty());
            let suffix = ord
                .filter(|o| l.ordinal_input_suffixes.contains(o))
                .unwrap_or("");
            let dw = if ord.is_some() {
                l.ordinal(day, suffix)
            } else {
                l.cardinal(day)
            };
            if month_first && l.code != "en" {
                format!("{} {dw}", self.month_py(month)?)
            } else {
                format!("{dw} {}", self.month_py(month)?)
            }
        };
        Ok(match year {
            Some(y) => format!("{out} {}", l.year(y)),
            None => out,
        })
    }

    pub fn cur_key(&self, symtok: &str) -> &'static str {
        let t = strip(strip(symtok).trim_end_matches('.')).to_uppercase();
        let code = match t.as_str() {
            "₹" | "RS" | "INR" => Some("rupee"),
            "$" | "USD" => Some("dollar"),
            "€" | "EUR" => Some("euro"),
            "£" | "GBP" => Some("pound"),
            _ => None,
        };
        code.or_else(|| get(self.l.currency_words, &t))
            .or_else(|| get(self.l.currency_words, &format!("{t}.")))
            .unwrap_or("rupee")
    }

    pub fn spell_uri_part(&self, part: &str) -> R<String> {
        let s = |k| self.l.sym(k).to_string();
        let mut out: Vec<String> = Vec::new();
        for run in re!(r"[^\W\d_]+|\d+|[^\w\s]|_").find_iter(&cv(part)) {
            let run = run.all();
            out.push(if run.chars().all(is_digit) {
                self.digits_spelled(&run, false)?
            } else {
                match run.as_str() {
                    "." => s("dot"),
                    "_" => s("underscore"),
                    "@" => s("at"),
                    "/" => s("slash"),
                    "-" => "dash".into(),
                    "+" => s("plus"),
                    "?" => "question mark".into(),
                    "=" => "equals".into(),
                    "&" => "ampersand".into(),
                    "#" => "hash".into(),
                    "%" => "percent".into(),
                    ":" => "colon".into(),
                    "~" => "tilde".into(),
                    _ => {
                        if run.to_lowercase() == "www" {
                            self.letters_spelled("WWW")
                        } else if run.chars().count() == 1
                            || get(self.l.abbrev_units, &run.to_uppercase()).is_some()
                        {
                            self.letters_spelled(&run.to_uppercase())
                        } else {
                            run
                        }
                    }
                }
            });
        }
        Ok(out
            .into_iter()
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join(" "))
    }

    // ---------------------------------------------------------------- matchers
    fn m_email(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").find_iter(t) {
            v.push((m.start(), m.end(), self.spell_uri_part(&m.all())?));
        }
        Ok(v)
    }
    /// The lexicon (L3) version of `_m_url`.
    fn m_url(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        let pat = re!(
            r"https?://[^\s<>]+|www\.[^\s<>]+|\b[A-Za-z][\w\-]*(?:\.[A-Za-z0-9][\w\-]*)*\.(?:com|org|net|ai|io|co|dev|app|edu|gov|in)(?:\.(?:in|uk|us))?(?:[/?#][^\s<>]*)?(?![\w]|\.\w)"
        );
        for m in pat.find_iter(t) {
            let all = m.all();
            let token = all.trim_end_matches(['.', ',', '!', '?', ';', ')']);
            let body = re!(r"^https?://").replace(token, "");
            v.push((
                m.start(),
                m.start() + token.chars().count(),
                self.spell_uri_part(&body)?,
            ));
        }
        for m in
            re!(r"(?<![\w.@])[A-Za-z][\w\-]*\.[\w.\-]*@[A-Za-z][A-Za-z0-9]{1,15}(?![\w@]|\.\w)")
                .find_iter(t)
        {
            v.push((m.start(), m.end(), self.spell_uri_part(&m.all())?));
        }
        Ok(v)
    }
    fn m_handle(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"(?<![\w.])@([A-Za-z](?:[\w.]*\w)?)").find_iter(t) {
            v.push((
                m.start(),
                m.end(),
                format!("{} {}", self.l.sym("at"), self.spell_uri_part(&m.s(1))?),
            ));
        }
        Ok(v)
    }
    fn m_comparison(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let pat = re!(
            r"(?<![\w.])(?:(?P<left>(?:₹\s*)?[+-]?\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*%)?)\s*)?(?P<op><=|>=|≤|≥|<|>)\s*(?P<right>(?:₹\s*)?[+-]?\d+(?:,\d{2,3})*(?:\.\d+)?(?:\s*%)?)"
        );
        let words = |tok: &str| -> R<String> {
            let s = strip(tok);
            if let Some(r) = s.strip_prefix('₹') {
                return Ok(l.join_amount(&self.number_words(strip(r))?, l.sym("rupee")));
            }
            if let Some(r) = s.strip_suffix('%') {
                return Ok(format!(
                    "{} {}",
                    self.number_words(strip(r))?,
                    l.sym("percent")
                ));
            }
            self.number_words(s)
        };
        let ors: &[(&str, &str)] = &[
            ("hi", "या"),
            ("te", "లేదా"),
            ("ta", "அல்லது"),
            ("bn", "বা"),
            ("gu", "અથવા"),
            ("kn", "ಅಥವಾ"),
            ("ml", "അല്ലെങ്കിൽ"),
            ("mr", "किंवा"),
            ("or", "କିମ୍ବା"),
            ("pa", "ਜਾਂ"),
        ];
        let mut v = Vec::new();
        for m in pat.find_iter(t) {
            let op = m.named("op").unwrap();
            let mut cmp = l
                .sym(if op.starts_with('<') || op == "≤" {
                    "lt"
                } else {
                    "gt"
                })
                .to_string();
            if op.chars().count() > 1 || op == "≤" || op == "≥" {
                if l.code == "en" {
                    cmp += " or equal to";
                } else {
                    cmp += &format!(
                        " {} {}",
                        get(ors, l.code).ok_or(Exc::Other)?,
                        l.sym("equals")
                    );
                }
            }
            let left = match m.named("left") {
                Some(x) if !x.is_empty() => words(&x)? + " ",
                _ => String::new(),
            };
            let right = words(&m.named("right").unwrap())?;
            let rep = if l.code == "en" {
                format!("{left}{cmp} {right}")
            } else {
                format!("{left}{right} {cmp}")
            };
            v.push((m.start(), m.end(), rep));
        }
        Ok(v)
    }
    fn m_time_range(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in self.time_range.find_iter(t) {
            let g = |i: usize| m.get(i);
            if !m.has(3) || !m.has(10) {
                continue;
            }
            if !((m.has(5) || m.has(6) || m.has(4)) && (m.has(12) || m.has(13) || m.has(11))) {
                continue;
            }
            let (h1, m1, p1) = self.resolve_time(
                g(2).as_deref(),
                &m.s(3),
                g(4).as_deref(),
                g(5).as_deref(),
                g(7).as_deref(),
            )?;
            let (h2, m2, p2) = self.resolve_time(
                g(9).as_deref(),
                &m.s(10),
                g(11).as_deref(),
                g(12).as_deref(),
                g(14).as_deref(),
            )?;
            let mut a = self.time_words(h1, m1, p1);
            let mut b = self.time_words(h2, m2, p2);
            if !l.ampm_words.is_empty() {
                if let Some(x) = g(5) {
                    let w = self.ampm_word(&x).ok_or(Exc::Other)?;
                    a = format!(
                        "{} {w}",
                        if m1 == 0 {
                            l.two_digit(h1 as u128).to_string()
                        } else {
                            a
                        }
                    );
                }
                if let Some(x) = g(12) {
                    let w = self.ampm_word(&x).ok_or(Exc::Other)?;
                    b = format!(
                        "{} {w}",
                        if m2 == 0 {
                            l.two_digit(h2 as u128).to_string()
                        } else {
                            b
                        }
                    );
                }
            }
            let mut rep = fmt(&fmt(l.range_to, "a", &a), "b", &b);
            let tail = strip(l.range_to.rsplit("{b}").next().unwrap());
            if !tail.is_empty() {
                let rest = &t[m.end()..];
                let rx = Regex::new(&format!(r"\s*{}(?!\w)", escape(tail)));
                if rx.match_at(rest, 0).is_some() {
                    rep = rep[..rep.len() - tail.len()]
                        .trim_end_matches(is_space)
                        .to_string();
                }
            }
            v.push((m.start(), m.end(), rep));
        }
        Ok(v)
    }
    fn m_duration(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let sec = l.sym_get("second").unwrap_or("seconds");
        let mut v = Vec::new();
        for m in re!(r"\b(\d{1,3}):(\d{2}):(\d{2})\b").find_iter(t) {
            let (h, mm, ss) = (pint(&m.s(1))?, pint(&m.s(2))?, pint(&m.s(3))?);
            if mm > 59 || ss > 59 {
                continue;
            }
            let hours = get(l.abbrev_units, "HRS").unwrap_or("hours");
            let minutes = l.sym("minute");
            let parts: Vec<String> = [(h, hours), (mm, minutes), (ss, sec)]
                .iter()
                .filter(|(n, _)| *n != 0)
                .map(|&(n, u)| {
                    let u = if n != 1 || l.code != "en" {
                        u
                    } else {
                        u.trim_end_matches('s')
                    };
                    l.join_amount(&l.cardinal(n), u)
                })
                .collect();
            let rep = if parts.is_empty() {
                l.join_amount(&l.cardinal(0), sec)
            } else {
                parts.join(" ")
            };
            v.push((m.start(), m.end(), rep));
        }
        Ok(v)
    }
    fn m_time_verbalized(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in self.time_verbal.find_iter(t) {
            let (h, mm) = (pint(&m.s(1))?, pint(&m.s(2))?);
            if h > 23 || mm > 59 {
                continue;
            }
            let mut rep = self.time_words(h as u32, mm as u32, None);
            if let Some(sfx) = get(self.l.marker_suffixes, &m.s(3)) {
                rep = self.l.attach_postposition(&rep, sfx).unwrap_or(rep);
            }
            v.push((m.start(), m.end(), rep));
        }
        Ok(v)
    }
    fn m_time(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in self.time_pat.find_iter(t) {
            let (lead, h, mm, ampm, marker, trail) =
                (m.get(1), m.get(2), m.get(3), m.get(4), m.get(5), m.get(6));
            let Some(h) = h else { continue };
            let hh = pint(&h)?;
            let is_hhmm = match &mm {
                Some(x) => pint(x)? <= 59 && hh < 24,
                None => false,
            };
            let has_p = lead.is_some() || trail.is_some();
            if !(ampm.is_some() || marker.is_some() || (has_p && (is_hhmm || mm.is_none())))
                && !(is_hhmm && mm.is_some() && hh > 12)
                && !(is_hhmm && !self.keyword_before(t, m.start(), &self.ratio_kws, 4))
            {
                continue;
            }
            if mm.is_none() && !(ampm.is_some() || marker.is_some()) {
                continue;
            }
            if let Some(x) = &mm {
                if pint(x)? > 59 {
                    continue;
                }
            }
            if hh > 23 {
                continue;
            }
            let (h12, mnt, period) = self.resolve_time(
                lead.as_deref(),
                &h,
                mm.as_deref(),
                ampm.as_deref(),
                trail.as_deref(),
            )?;
            let mut rep = match lead.as_ref().or(trail.as_ref()) {
                Some(sp) => format!("{sp} {}", l.time_phrase(h12, mnt, None)),
                None => self.time_words(h12, mnt, period),
            };
            if let Some(a) = &ampm {
                if !l.ampm_words.is_empty() {
                    if let Some(w) = self.ampm_word(a) {
                        if mnt == 0 {
                            rep = l.two_digit(h12 as u128).to_string();
                        }
                        rep = format!("{rep} {w}");
                    }
                }
            }
            if let Some(sfx) = get(l.marker_suffixes, marker.as_deref().unwrap_or("")) {
                rep = l.attach_postposition(&rep, sfx).unwrap_or(rep);
            }
            v.push((m.start(), m.end(), rep));
        }
        Ok(v)
    }
    fn m_tamil_date(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        if l.code != "en" && l.code != "ta" {
            return Ok(v);
        }
        let pat = re!(&format!(
            r"(?<!\w)({})\s+(\d{{1,2}})(?:-?(?:ஆம்|ம்))?(?![\dA-Za-z])(?:\s*தேதி)?",
            tb::TAMIL_MONTHS.join("|")
        ));
        for m in pat.find_iter(t) {
            let d = pint(&m.s(2))?;
            if (1..=31).contains(&d) {
                v.push((
                    m.start(),
                    m.end(),
                    format!("{} {} தேதி", m.s(1), tb::TAMIL_ORDINALS[d as usize]),
                ));
            }
        }
        for m in re!(r"(?<!\w)(\d+)-?(?:ஆம்|ம்)(?![\w஀-௿])").find_iter(t) {
            let d = pint(&m.s(1))?;
            if !(1..=31).contains(&d) {
                if l.code == "en" {
                    return Err(nerr_src(m.span(0), t));
                }
                v.push((m.start(), m.end(), l.ordinal(d, "ஆம்")));
            } else {
                v.push((
                    m.start(),
                    m.end(),
                    tb::TAMIL_ORDINALS[d as usize].to_string(),
                ));
            }
        }
        Ok(v)
    }
    fn m_date(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in self.date_range.find_iter(t) {
            let mo = self.month_index(&m.s(3)).ok_or(Exc::Other)?;
            let y = if m.has(4) { pint(&m.s(4))? } else { 2000 };
            for d in [m.s(1), m.s(2)] {
                if !calendar_ok(pint(&d)?, mo as u128, y) {
                    return Err(NERR);
                }
            }
            let days = fmt(
                &fmt(l.range, "a", &l.cardinal(pint(&m.s(1))?)),
                "b",
                &l.cardinal(pint(&m.s(2))?),
            );
            let mut out = format!("{days} {}", self.month_py(mo as u128)?);
            if m.has(4) {
                out += &format!(" {}", l.year(y));
            }
            v.push((m.start(), m.end(), out));
        }
        for m in re!(r"(?<![\w-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])").find_iter(t) {
            v.push((
                m.start(),
                m.end(),
                self.date_words(
                    pint(&m.s(3))?,
                    pint(&m.s(2))?,
                    Some(pint(&m.s(1))?),
                    None,
                    false,
                )?,
            ));
        }
        for m in re!(r"\b(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})\b(?![/A-Za-z-])").find_iter(t) {
            let (d, mo, y) = (pint(&m.s(1))?, pint(&m.s(3))?, pint(&m.s(4))?);
            if !((1..=31).contains(&d) && (1..=12).contains(&mo)) {
                continue;
            }
            if m.s(4).chars().count() == 2 {
                return Err(Exc::Other); // `y += None`: two_digit_year_base is unset
            }
            v.push((
                m.start(),
                m.end(),
                self.date_words(d, mo, Some(y), None, false)?,
            ));
        }
        for m in self.date_dmon.find_iter(t) {
            let d = pint(&m.s(1))?;
            let Some(mo) = self.month_index(&m.s(3)) else {
                continue;
            };
            if mo == 0 || !(1..=31).contains(&d) {
                continue;
            }
            let y = if m.has(4) { Some(pint(&m.s(4))?) } else { None };
            v.push((
                m.start(),
                m.end(),
                self.date_words(d, mo as u128, y, m.get(2).as_deref(), false)?,
            ));
        }
        for m in self.date_mon_d.find_iter(t) {
            let Some(mo) = self.month_index(&m.s(1)) else {
                continue;
            };
            let d = pint(&m.s(2))?;
            if mo == 0 || !(1..=31).contains(&d) {
                continue;
            }
            let y = if m.has(4) { Some(pint(&m.s(4))?) } else { None };
            v.push((
                m.start(),
                m.end(),
                self.date_words(d, mo as u128, y, m.get(3).as_deref(), true)?,
            ));
        }
        Ok(v)
    }
    fn m_currency_range(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in self.cur_range.find_iter(t) {
            if !m.s(1).is_empty() && m.s(3).starts_with(['-', '+']) {
                return Err(nerr_src(m.span(0), t));
            }
            let a = self.number_words(&(m.s(1) + &m.s(3)))?;
            let b = self.number_words(&m.s(4))?;
            let cw = l.sym(self.cur_key(&m.s(2)));
            v.push((
                m.start(),
                m.end(),
                fmt(&fmt(l.range, "a", &a), "b", &l.join_amount(&b, cw)),
            ));
        }
        Ok(v)
    }
    fn m_bare_rupee_suffix(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in re!(r"(?<![\w₹.])(\d+(?:,\d{2,3})*(?:\.\d+)?)\s?/-").find_iter(t) {
            let words = self.number_words(&m.s(1).replace(',', ""))?;
            v.push((m.start(), m.end(), l.join_amount(&words, l.sym("rupee"))));
        }
        Ok(v)
    }
    fn m_currency(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in self.currency.find_iter(t) {
            if !m.s(1).is_empty() && !m.s(3).is_empty() {
                return Err(nerr_src(m.span(0), t));
            }
            let key = self.cur_key(&m.s(2));
            let num = m.s(4).replace(',', "");
            let scale = m.get(5).filter(|s| !s.is_empty());
            let cur = l.sym(key);
            let two_dec = num.split_once('.').filter(|(_, f)| f.chars().count() == 2);
            let mut out = if let (Some((ip, fp)), None) = (two_dec, &scale) {
                let main = l.cardinal(pint_or0(ip)?);
                let sub = if key == "rupee" && l.sym_get("paisa").is_some_and(|p| !p.is_empty()) {
                    Some(l.sym("paisa"))
                } else if key != "rupee" {
                    Some(l.sym("cent"))
                } else {
                    None
                };
                if fp.trim_matches('0').is_empty() {
                    l.join_amount(&main, cur)
                } else if let Some(sw) = sub.filter(|s| !s.is_empty()) {
                    format!(
                        "{} {}",
                        l.join_amount(&main, cur),
                        l.join_amount(&l.cardinal(pint(fp)?), sw)
                    )
                } else {
                    l.join_amount(&self.decimal(pint_or0(ip)?, fp)?, cur)
                }
            } else {
                let words = self.number_words(&num)?;
                match &scale {
                    Some(sc) => {
                        let sl = sc.to_lowercase();
                        let canon = get(l.scale_words, sc);
                        let scale_word = if canon.is_none() && tb::EN_SCALES.contains(&sl.as_str())
                        {
                            sc.clone()
                        } else {
                            let canon = canon.unwrap_or(if sl == "cr" || sl.contains("crore") {
                                "crore"
                            } else if sl.contains("lakh") {
                                "lakh"
                            } else {
                                "thousand"
                            });
                            l.scale_noun(canon, &num)
                        };
                        let mut out = l.join_amount(&l.join_amount(&words, &scale_word), cur);
                        if let Some(case) = m.get(6).filter(|c| !c.is_empty()) {
                            out = l.attach_postposition(&out, &case).unwrap_or(out);
                        }
                        out
                    }
                    None => l.join_amount(&words, cur),
                }
            };
            let sign = if !m.s(1).is_empty() { m.s(1) } else { m.s(3) };
            if !sign.is_empty() {
                out = format!(
                    "{} {out}",
                    l.sym(if sign == "-" { "minus" } else { "plus" })
                );
            }
            let all = m.all();
            let space = all.chars().next().is_some_and(is_space)
                || (m.start() > 0 && is_alnum(t[m.start() - 1]));
            v.push((
                m.start(),
                m.end(),
                if space { format!(" {out}") } else { out },
            ));
        }
        Ok(v)
    }
    fn m_amount_context(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in self.amount.find_iter(t) {
            v.push((m.start(), m.end(), self.number_words(&m.s(1))?));
        }
        const LABELS: &[&str] = &[
            "balance",
            "amount",
            "payable",
            "outstanding",
            "payment",
            "price",
            "fee",
            "pay",
            "emi",
            "தவணை",
        ];
        for m in re!(r"(?<![\w.,+/-])\d+(?:,\d{2,3})*(?!\w|[.,/-]\d)").find_iter(t) {
            let before = sentence_before(t, m.start()).to_lowercase();
            let toks = word_toks(&before);
            let toks = last_n(&toks, 3);
            if toks.iter().any(|x| LABELS.contains(&x.as_str()))
                && !toks
                    .iter()
                    .any(|x| ["number", "id", "code", "otp", "pin"].contains(&x.as_str()))
            {
                v.push((m.start(), m.end(), self.number_words(&m.all())?));
            }
        }
        Ok(v)
    }
    fn m_percent_range(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in re!(r"\b(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)\s*%").find_iter(t) {
            let (a, b) = (self.number_words(&m.s(1))?, self.number_words(&m.s(2))?);
            v.push((
                m.start(),
                m.end(),
                format!(
                    "{} {}",
                    fmt(&fmt(l.range, "a", &a), "b", &b),
                    l.sym("percent")
                ),
            ));
        }
        Ok(v)
    }
    fn m_percent(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in re!(r"(-?)(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*%").find_iter(t) {
            let mut w = self.number_words(&m.s(2))?;
            if !m.s(1).is_empty() {
                w = format!("{} {w}", l.sym("minus"));
            }
            v.push((m.start(), m.end(), format!("{w} {}", l.sym("percent"))));
        }
        Ok(v)
    }
    fn m_negative(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"(?<![\w.,-])-(\d+(?:\.\d+)?)").find_iter(t) {
            v.push((
                m.start(),
                m.end(),
                format!("{} {}", self.l.sym("minus"), self.number_words(&m.s(1))?),
            ));
        }
        Ok(v)
    }
    fn m_designation(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"\b([A-Z][a-z]{2,})-([A-Za-z0-9]+)\b").find_iter(t) {
            let seg = m.s(2);
            if !seg.chars().any(is_decimal) {
                continue;
            }
            if self.month_index(&seg).is_some_and(|x| x != 0)
                || self.month_index(&m.s(1)).is_some_and(|x| x != 0)
            {
                continue;
            }
            v.push((
                m.start(),
                m.end(),
                format!("{} {}", m.s(1), self.spell_model_segment(&seg)?),
            ));
        }
        for m in re!(r"\b([A-Z][a-z]{2,})\s+([A-Z]{1,4}-[A-Za-z0-9]+)\b").find_iter(t) {
            let brand = m.s(1).to_lowercase();
            if self.id_keywords.contains(&brand) || self.label_words.contains(&brand) {
                continue;
            }
            let seg = m.s(2);
            if !seg.chars().any(is_decimal) {
                continue;
            }
            v.push((m.start_of(2), m.end_of(2), self.spell_model_segment(&seg)?));
        }
        Ok(v)
    }
    fn m_vehicle_plate(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"\b([A-Z]{2})[ -](\d{1,2})[ -]([A-Z]{1,3})[ -](\d{3,4})\b").find_iter(t) {
            let mut out = Vec::new();
            for g in 1..=4 {
                for ch in m.s(g).chars() {
                    out.push(if is_digit(ch) {
                        self.l
                            .digit(decimal_value(ch).ok_or(Exc::Other)?)
                            .to_string()
                    } else {
                        self.l.letter_c(ch)
                    });
                }
            }
            v.push((m.start(), m.end(), out.join(", ")));
        }
        Ok(v)
    }
    fn m_ordinal(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in self.ordinal.find_iter(t) {
            let n = pint(&m.s(1))?;
            let suf = m.s(2).to_lowercase();
            let rep = if ["st", "nd", "rd", "th"].contains(&suf.as_str()) {
                self.l.ordinal(n, "")
            } else {
                self.l.ordinal(n, &suf)
            };
            v.push((m.start(), m.end(), rep));
        }
        Ok(v)
    }
    fn m_code(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        let pat = re!(
            r"#?\b(?=[A-Za-z0-9/-]*\d)(?=[A-Za-z0-9/-]*[A-Za-z])[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+(?![A-Za-z0-9])"
        );
        for m in pat.find_iter(t) {
            let all = m.all();
            let tok = all.trim_start_matches('#');
            let parts: Vec<&str> = tok.split(['-', '/']).collect();
            let mi = |p: &str| self.month_index(p).is_some_and(|x| x != 0);
            let isdig = |p: &str| !p.is_empty() && p.chars().all(is_digit);
            let isalph = |p: &str| !p.is_empty() && p.chars().all(is_alpha);
            if parts.iter().any(|p| mi(p)) && parts.iter().filter(|p| !mi(p)).all(|p| isdig(p)) {
                continue;
            }
            if parts
                .iter()
                .any(|p| re!(r"[a-z][a-z]+|[A-Z][a-z]{2,}").is_full(p))
            {
                continue;
            }
            if parts.len() == 2
                && isdig(parts[0])
                && isalph(parts[1])
                && parts[1].chars().count() >= 3
            {
                continue;
            }
            if parts.len() == 2
                && isdig(parts[0])
                && tb::UNIT_INITIALISMS.contains(&parts[1].to_uppercase().as_str())
            {
                continue;
            }
            if re!(r"\d{1,3}-?[A-Za-z]").is_full(tok)
                && self.keyword_before(t, m.start(), &self.label_words, 2)
            {
                continue;
            }
            if tok.chars().filter(|c| is_alnum(*c)).count() <= 2 {
                continue;
            }
            v.push((m.start(), m.end(), self.spell_alnum(tok, true)?));
        }
        for m in
            re!(r"\b(?=(?:[A-Za-z]*\d){2})(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{5,}\b").find_iter(t)
        {
            let tok = m.all();
            if re!(r"\d+s").is_full(&tok) || re!(r"\d+[x×]\d+").is_full(&tok) {
                continue;
            }
            let tc = cv(&tok);
            if let Some(m2) = re!(r"\d+([A-Za-z]{1,4})").fullmatch(&tc) {
                if tb::UNIT_INITIALISMS.contains(&m2.s(1).to_uppercase().as_str()) {
                    continue;
                }
            }
            if re!(r"(?:[A-Z][a-z]?\d*)+").is_full(&tok) && chem_ok(&tok) {
                continue;
            }
            v.push((m.start(), m.end(), self.spell_alnum(&tok, true)?));
        }
        Ok(v)
    }
    fn m_id_digits(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        let pat = re!(
            r"(\+?)(?:#?\b\d+(?:[-\s]\d+)+\b|#\s?\d{3,}\b|(?<![\d.])(?<!\d,)\d{7,}(?![.,]?\d))"
        );
        for m in pat.find_iter(t) {
            let has_plus = m.s(1) == "+";
            let all = m.all();
            let tok = all.trim_start_matches(['+', '#', ' ']);
            let groups: Vec<&str> = tok.split(|c: char| c == '-' || is_space(c)).collect();
            let digits: String = groups.concat();
            let nd = digits.chars().count();
            let after_kw = || {
                let a = id_toks(&sentence_after(t, m.end()).to_lowercase());
                a.iter().take(4).any(|x| self.id_keywords.contains(x))
            };
            let is_id = (has_plus && nd >= 8)
                || all.starts_with('#')
                || (nd >= 7 && !tok.contains(' '))
                || (groups.iter().any(|g| g.chars().count() >= 4)
                    && groups.len() >= 2
                    && tok.contains('-'))
                || (groups.len() >= 3 && tok.contains('-'))
                || self.keyword_before(t, m.start(), &self.id_keywords, 3)
                || after_kw();
            if !is_id {
                continue;
            }
            if tok.contains('-')
                && groups.len() == 2
                && groups.iter().all(|g| g.chars().count() <= 3)
                && pint(groups[0])? < pint(groups[1])?
                && !self.keyword_before(t, m.start(), &self.id_keywords, 3)
                && !after_kw()
            {
                continue;
            }
            let mut rep = self.digits_spelled(&digits, true)?;
            if has_plus {
                rep = format!("{} {rep}", l.sym("plus"));
            }
            v.push((m.start(), m.end(), rep));
        }
        for m in re!(r"(?<![\d.])(?<!\d,)\d{3,6}(?![.,]?\d)(?!s\b)").find_iter(t) {
            let after = id_toks(&sentence_after(t, m.end()).to_lowercase());
            let after = &after[..after.len().min(6)];
            if after.iter().any(|x| self.id_after.contains(x)) {
                v.push((m.start(), m.end(), self.digits_spelled(&m.all(), true)?));
                continue;
            }
            if after.iter().take(4).any(|x| self.id_keywords.contains(x))
                && !self.keyword_before(t, m.start(), &self.label_words, 2)
                && !self.keyword_before(t, m.start(), &self.year_words2, 2)
            {
                v.push((m.start(), m.end(), self.digits_spelled(&m.all(), true)?));
                continue;
            }
            if self.keyword_before(t, m.start(), &self.id_keywords, 3) {
                if self.keyword_before(t, m.start(), &self.label_words, 2)
                    || self.keyword_before(t, m.start(), &self.year_words2, 2)
                {
                    continue;
                }
                v.push((m.start(), m.end(), self.digits_spelled(&m.all(), true)?));
            }
        }
        Ok(v)
    }
    fn m_chem(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"\b(?:[A-Z][a-z]?\d*){1,8}\b").find_iter(t) {
            let tok = m.all();
            if tok.chars().count() < 2
                || tb::ACRONYM_WHITELIST.contains(&tok.to_uppercase().as_str())
            {
                continue;
            }
            if !chem_ok(&tok) {
                continue;
            }
            let mut out = Vec::new();
            for p in re!(r"[A-Z][a-z]?|\d+").find_iter(&cv(&tok)) {
                let p = p.all();
                if p.chars().all(is_digit) {
                    out.push(self.l.cardinal(pint(&p)?));
                } else {
                    out.push(self.letters_spelled(&p.to_uppercase()));
                }
            }
            v.push((m.start(), m.end(), out.join(" ")));
        }
        Ok(v)
    }
    fn m_math(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        let word = |x: &str| -> R<String> {
            if x.chars().next().is_some_and(is_digit) {
                self.number_words(x)
            } else {
                Ok(l.letter(x))
            }
        };
        for m in re!(r"(\d+(?:\.\d+)?|[A-Za-z])\s*\^\s*(\d+|[A-Za-z])").find_iter(t) {
            let (b, e) = (word(&m.s(1))?, word(&m.s(2))?);
            v.push((
                m.start(),
                m.end(),
                fmt(&fmt(l.power, "base", &b), "exp", &e),
            ));
        }
        for m in re!(r"(\d+(?:\.\d+)?|[A-Za-z])²").find_iter(t) {
            v.push((
                m.start(),
                m.end(),
                format!("{} {}", word(&m.s(1))?, l.sym("squared")),
            ));
        }
        for m in re!(r"√\s*(\d+(?:\.\d+)?)").find_iter(t) {
            v.push((
                m.start(),
                m.end(),
                format!("{} {}", l.sym("sqrt"), self.number_words(&m.s(1))?),
            ));
        }
        let eq = re!(
            r"(?<![\w.])((?:[A-Za-z]{1,2}|\d+(?:\.\d+)?)(?:\s*[=+\-−×*÷/^]\s*(?:[A-Za-z]{1,2}|\d+(?:\.\d+)?))+)"
        );
        for m in eq.find_iter(t) {
            let span = m.s(1);
            if !span.contains(['=', '×', '÷', '^', '√']) {
                continue;
            }
            v.push((m.start_of(1), m.end_of(1), self.render_math_expr(&span)?));
        }
        for m in
            re!(r"\b([A-Za-z])\b(?=\s+(?:squared|cubed|plus|minus|times|equals|to the power)\b)")
                .find_iter(t)
        {
            v.push((m.start(), m.end(), l.letter(&m.s(1))));
        }
        Ok(v)
    }
    fn render_math_expr(&self, span: &str) -> R<String> {
        let l = self.l;
        let toks: Vec<String> = re!(r"\d+(?:\.\d+)?|[A-Za-z]+|[=+\-−×*÷/^]")
            .find_iter(&cv(span))
            .iter()
            .map(|m| m.all())
            .collect();
        let op = |x: &str| -> Option<&'static str> {
            Some(l.sym(match x {
                "=" => "equals",
                "+" => "plus",
                "-" | "−" => "minus",
                "×" | "*" => "times",
                "÷" | "/" => "divide",
                _ => return None,
            }))
        };
        let mut out: Vec<String> = Vec::new();
        let mut i = 0;
        while i < toks.len() {
            let tok = &toks[i];
            if tok == "^" {
                let exp = toks.get(i + 1).map(String::as_str).unwrap_or("");
                let ew = if exp.chars().next().is_some_and(is_digit) {
                    self.number_words(exp)?
                } else {
                    self.letters_spelled(&exp.to_uppercase())
                };
                let base = out.pop().unwrap_or_default();
                out.push(fmt(&fmt(l.power, "base", &base), "exp", &ew));
                i += 2;
                continue;
            }
            if let Some(w) = op(tok) {
                out.push(w.to_string());
            } else if tok.chars().next().is_some_and(is_digit) {
                out.push(self.number_words(tok)?);
            } else {
                out.push(self.letters_spelled(&tok.to_uppercase()));
            }
            i += 1;
        }
        Ok(out.join(" "))
    }
    fn m_num_range(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        let rng = |a: &str, b: &str| fmt(&fmt(l.range, "a", a), "b", b);
        for m in re!(r"(\d+(?:\.\d+)?)\s*[–]\s*(\d+(?:\.\d+)?)").find_iter(t) {
            v.push((
                m.start(),
                m.end(),
                rng(&self.number_words(&m.s(1))?, &self.number_words(&m.s(2))?),
            ));
        }
        for m in re!(r"\b(\d{1,3})-(\d{1,3})\b").find_iter(t) {
            if self.keyword_before(t, m.start(), &self.id_keywords, 3) {
                continue;
            }
            let (a, b) = (pint(&m.s(1))?, pint(&m.s(2))?);
            let (ca, cb) = (l.cardinal(a), l.cardinal(b));
            if a > b
                && re!(r"\s+(?:days?|months?|years?|hours?|minutes?)\b", i)
                    .match_at(&t[m.end()..], 0)
                    .is_none()
                || a == b
            {
                v.push((m.start(), m.end(), format!("{ca} {cb}")));
            } else {
                v.push((m.start(), m.end(), rng(&ca, &cb)));
            }
        }
        Ok(v)
    }
    fn m_dimensions(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in re!(r"\b(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\b").find_iter(t) {
            if m.s(1) == "0" {
                continue;
            }
            let rep = format!(
                "{} {} {}",
                self.number_words(&m.s(1))?,
                l.sym("times"),
                self.number_words(&m.s(2))?
            );
            v.push((m.start(), m.end(), rep));
        }
        for m in re!(r"(?<=\s)[x×](?=\s\d)").find_iter(t) {
            v.push((m.start(), m.end(), l.sym("times").to_string()));
        }
        Ok(v)
    }
    fn m_roman(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        let pat = re!(
            r"\b(?=[IVXLCDM]{2,7}\b)M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b"
        );
        for m in pat.find_iter(t) {
            let tok = m.all();
            if tok.is_empty() || !self.keyword_before(t, m.start(), &self.roman_ctx, 2) {
                continue;
            }
            let val = |c: char| -> i64 {
                match c {
                    'I' => 1,
                    'V' => 5,
                    'X' => 10,
                    'L' => 50,
                    'C' => 100,
                    'D' => 500,
                    _ => 1000,
                }
            };
            let cs: Vec<char> = tok.chars().collect();
            let mut total = 0i64;
            for (i, &c) in cs.iter().enumerate() {
                if i + 1 < cs.len() && val(cs[i + 1]) > val(c) {
                    total -= val(c);
                } else {
                    total += val(c);
                }
            }
            v.push((m.start(), m.end(), self.l.cardinal_i(total as i128)));
        }
        Ok(v)
    }
    fn m_unit_initialism(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        let pat = re!(
            &{
                let mut u: Vec<&str> = tb::UNIT_INITIALISMS.to_vec();
                u.sort_by_key(|w| std::cmp::Reverse(w.len()));
                format!(
                    r"(?<![\w.])(-?)\b(\d+(?:,\d{{2,3}})*(?:\.\d+)?)[\s\-]?({})(?:\s?/\s?([A-Za-z]{{1,4}}))?\b",
                    u.join("|")
                )
            },
            i
        );
        for m in pat.find_iter(t) {
            let unit = m.s(3);
            if !tb::UNIT_INITIALISMS.contains(&unit.to_uppercase().as_str()) {
                continue;
            }
            if unit.chars().count() == 1 && !unit.chars().all(|c| c.is_uppercase()) {
                continue;
            }
            let mut num = self.number_words(&m.s(2))?;
            if !m.s(1).is_empty() {
                num = format!("{} {num}", l.sym("minus"));
            }
            let uk: String = unit
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_uppercase();
            let word = get(l.unit_words, &uk)
                .map(str::to_string)
                .unwrap_or_else(|| self.letters_spelled(&uk));
            let mut rep = l.join_amount(&num, &word);
            if let Some(u2) = m.get(4) {
                let u2 = u2.to_uppercase();
                let w2 = get(l.unit_words, &u2)
                    .map(str::to_string)
                    .unwrap_or_else(|| self.letters_spelled(&u2));
                rep += &format!(" {} {w2}", l.sym_get("per").unwrap_or("per"));
            }
            v.push((m.start(), m.end(), rep));
        }
        Ok(v)
    }
    fn m_kmb_suffix(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"(?<![\d.A-Za-z])(\d+(?:\.\d+)?)([KMB])\b").find_iter(t) {
            v.push((
                m.start(),
                m.end(),
                format!("{} {}", self.number_words(&m.s(1))?, self.l.letter(&m.s(2))),
            ));
        }
        Ok(v)
    }
    fn m_fraction(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in re!(r"\b(\d{1,4}(?:\.\d+)?)/(\d{1,4}(?:\.\d+)?)\b").find_iter(t) {
            let rep = match (m.s(1).as_str(), m.s(2).as_str()) {
                ("1", "2") => l.sym("half").to_string(),
                ("1", "4") => l.sym("quarter").to_string(),
                ("3", "4") => l.sym("three_quarters").to_string(),
                (a, b) => fmt(
                    &fmt(l.fraction, "num", &self.number_words(a)?),
                    "den",
                    &self.number_words(b)?,
                ),
            };
            v.push((m.start(), m.end(), rep));
        }
        Ok(v)
    }
    fn m_ratio(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in re!(r"\b(\d{1,3}):(\d{1,2})\b").find_iter(t) {
            let mm = m.s(2);
            if mm.chars().count() == 2
                && pint(&mm)? <= 59
                && !self.keyword_before(t, m.start(), &self.ratio_kws, 4)
            {
                continue;
            }
            let (a, b) = (l.cardinal(pint(&m.s(1))?), l.cardinal(pint(&mm)?));
            let after = sl(t, m.end(), m.end() + 20);
            let after = strip(&after);
            if self
                .ratio_kws
                .iter()
                .filter(|w| !w.is_ascii())
                .any(|w| after.starts_with(w.as_str()))
            {
                v.push((m.start(), m.end(), format!("{a} {b}")));
                continue;
            }
            v.push((m.start(), m.end(), fmt(&fmt(l.ratio, "a", &a), "b", &b)));
        }
        Ok(v)
    }
    fn m_label(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in self.label.find_iter(t) {
            let mut out = vec![m.s(1)];
            if !m.s(2).is_empty() {
                out.push(l.cardinal(pint(&m.s(2))?));
                if !m.s(3).is_empty() {
                    out.push(l.letter(&m.s(3)));
                }
            } else if !m.s(4).is_empty() {
                let g4 = m.s(4);
                out.push(if g4.chars().count() == 1 {
                    l.letter(&g4)
                } else {
                    g4
                });
                if !m.s(5).is_empty() {
                    out.push(l.cardinal(pint(&m.s(5))?));
                }
            }
            v.push((m.start(), m.end(), out.join(" ")));
        }
        Ok(v)
    }
    fn m_acronym_model(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"\b([A-Z]{2,4})\s+(\d{1,3})(?![\d.])").find_iter(t) {
            let tok = m.s(1);
            if tb::ACRONYM_WHITELIST.contains(&tok.as_str())
                || tb::UNIT_INITIALISMS.contains(&tok.to_uppercase().as_str())
            {
                continue;
            }
            let lw = tok.to_lowercase();
            if self.id_keywords.contains(&lw) || self.label_words.contains(&lw) {
                continue;
            }
            v.push((
                m.start(),
                m.end(),
                format!(
                    "{} {}",
                    self.letters_spelled(&tok),
                    self.l.cardinal(pint(&m.s(2))?)
                ),
            ));
        }
        Ok(v)
    }
    fn m_num_hyphen_word(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"\b(\d+(?:,\d{2,3})*(?:\.\d+)?)-([A-Za-zऀ-ൿ]{2,})\b").find_iter(t) {
            let word = m.s(2);
            if self.month_index(&word).is_some_and(|x| x != 0)
                || tb::UNIT_INITIALISMS.contains(&word.to_uppercase().as_str())
            {
                continue;
            }
            v.push((
                m.start(),
                m.end(),
                format!("{} {word}", self.number_words(&m.s(1))?),
            ));
        }
        Ok(v)
    }
    fn m_decade(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in re!(r"\b(19|20)(\d)0s\b").find_iter(t) {
            let d = pint(&m.s(2))?;
            if m.s(1) == "19" {
                v.push((m.start(), m.end(), l.decade(d * 10, true)));
            } else {
                let y = pint(&m.all().chars().take(4).collect::<String>())?;
                v.push((m.start(), m.end(), fmt(l.decade_full, "tens", &l.year(y))));
            }
        }
        for m in re!(r"(?<![\w])'?(\d)0s\b").find_iter(t) {
            v.push((m.start(), m.end(), l.decade(pint(&m.s(1))? * 10, false)));
        }
        Ok(v)
    }
    fn m_year(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in re!(r"\b(1[5-9]\d{2}|20\d{2})\b").find_iter(t) {
            let y = pint(&m.all())?;
            let ctx = self.keyword_before(t, m.start(), &self.year_ctx, 2);
            let after = sl(t, m.end(), m.end() + 20);
            let suffixed = re!(r"^-?(?![।॥])[ऀ-෿]").search(&cv(&after)).is_some();
            let a = strip(&after);
            let next_tok = split_ws(a).first().copied().unwrap_or("");
            let follower = ["में", "से", "तक", "માં", "मध्ये", "ਵਿੱਚ", "ਤੋਂ"].contains(&next_tok)
                || l.year_words
                    .iter()
                    .any(|w| !w.is_empty() && next_tok.starts_with(w));
            if ctx || suffixed || follower {
                v.push((m.start(), m.end(), l.year(y)));
            }
        }
        Ok(v)
    }
    fn m_quoted_letter(&self, t: &[char]) -> R<Vec<Span>> {
        Ok(re!(r#"['"‘“]([A-Za-z])['"’”]"#)
            .find_iter(t)
            .iter()
            .map(|m| (m.start(), m.end(), self.l.letter(&m.s(1))))
            .collect())
    }
    fn m_short_mixed_token(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        let pat = re!(
            r"(?<![\d.])\b(?=[A-Za-z0-9]{2,5}\b)(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]+\b"
        );
        for m in pat.find_iter(t) {
            let tok = m.all();
            if re!(r"\d+(?:st|nd|rd|th|s)").is_full(&tok) {
                continue;
            }
            let mut out = Vec::new();
            for r in re!(r"[A-Za-z]+|\d+").find_iter(&cv(&tok)) {
                let r = r.all();
                if r.chars().all(is_digit) {
                    out.push(self.l.cardinal(pint(&r)?));
                } else {
                    out.push(self.letters_spelled(&r.to_uppercase()));
                }
            }
            v.push((m.start(), m.end(), out.join(" ")));
        }
        Ok(v)
    }
    fn m_decimal(&self, t: &[char]) -> R<Vec<Span>> {
        let mut v = Vec::new();
        for m in re!(r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+|\d+)\.(\d+)(?!\.?\d)").find_iter(t) {
            v.push((
                m.start(),
                m.end(),
                self.decimal(pint(&m.s(1).replace(',', ""))?, &m.s(2))?,
            ));
        }
        Ok(v)
    }
    fn m_int(&self, t: &[char]) -> R<Vec<Span>> {
        let l = self.l;
        let mut v = Vec::new();
        for m in re!(r"\b\d{1,3}(?:,\d{2,3})+\b|\b\d+\b").find_iter(t) {
            let digits = m.all().replace(',', "");
            let after = sl(t, m.end(), m.end() + 30);
            let after = strip(&after).to_lowercase();
            let fw = split_ws(&after);
            let street = fw
                .iter()
                .take(2)
                .any(|w| tb::STREET_WORDS.contains(&w.trim_matches(['.', ','])));
            if digits.chars().count() == 4 && !digits.starts_with('0') && street {
                v.push((m.start(), m.end(), l.year(pint(&digits)?)));
                continue;
            }
            if digits.starts_with('0') && digits.chars().count() > 1 {
                v.push((m.start(), m.end(), self.digits_spelled(&digits, true)?));
                continue;
            }
            v.push((m.start(), m.end(), l.cardinal(pint(&digits)?)));
        }
        Ok(v)
    }

    fn gather(&self, t: &[char]) -> R<Vec<(usize, usize, usize, String)>> {
        type F = fn(&Engine, &[char]) -> R<Vec<Span>>;
        let fs: [F; 40] = [
            Self::m_email,
            Self::m_url,
            Self::m_handle,
            Self::m_comparison,
            Self::m_time_range,
            Self::m_duration,
            Self::m_time_verbalized,
            Self::m_time,
            Self::m_tamil_date,
            Self::m_date,
            Self::m_currency_range,
            Self::m_currency,
            Self::m_bare_rupee_suffix,
            Self::m_amount_context,
            Self::m_percent_range,
            Self::m_percent,
            Self::m_negative,
            Self::m_designation,
            Self::m_vehicle_plate,
            Self::m_ordinal,
            Self::m_code,
            Self::m_id_digits,
            Self::m_chem,
            Self::m_math,
            Self::m_num_range,
            Self::m_dimensions,
            Self::m_roman,
            Self::m_unit_initialism,
            Self::m_kmb_suffix,
            Self::m_fraction,
            Self::m_ratio,
            Self::m_label,
            Self::m_acronym_model,
            Self::m_num_hyphen_word,
            Self::m_decade,
            Self::m_year,
            Self::m_quoted_letter,
            Self::m_short_mixed_token,
            Self::m_decimal,
            Self::m_int,
        ];
        let mut out = Vec::new();
        // _m_long_digit_runs (off) sat at priority 13; the priority values only order, so the gap is harmless
        for (prio, f) in fs.iter().enumerate() {
            for (s, e, r) in f(self, t)? {
                out.push((prio, s, e, r));
            }
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- pre-clean / sweep
    fn preclean(&self, text: &str) -> String {
        let l = self.l;
        let mut t = ascii_digits(text).replace('−', "-");
        t.retain(|c| !matches!(c, '\u{200b}' | '\u{200e}' | '\u{200f}' | '\u{feff}'));
        t = re!(r#"["“”„]+"#).replace(&t, "");
        t = re!(r"(?<=[\s(।,—–-])['‘’](?![A-Za-z]['‘’])(?=\S)").replace(&t, "");
        t = re!(r"^['‘’](?![A-Za-z]['‘’])(?=\S)").replace(&t, "");
        t = re!(r"(?<=\S)(?<!['‘’][A-Za-z])['‘’](?=[\s,।.!?;:)]|$)").replace(&t, "");
        t = re!(r"#{2,}").replace(&t, " ");
        t = re!(r"\s*(?:…|‥|\.{2,})\s*").replace(&t, ", ");
        t = re!(r"\s*[()]\s*").replace(&t, ", ");
        t = re!(r"(?<!:)/{2,}").replace(&t, " ");
        if let Some(rx) = &self.number_word {
            t = rx.replace(&t, &format!("{} ", l.number_word));
        }
        for (rx, word) in &self.abbrev {
            t = rx.replace(&t, word);
        }
        t = re!(r"^[,\s]+").replace(&t, "");
        t = re!(tb::RE_EMOJI).replace(&t, " ");
        t = re!(r"(?<=[\d])\s*[–—]\s*(?=[\d])").replace(&t, "–");
        t = re!(r"\s+[–—]\s+(?=\d)").replace(&t, " – ");
        t = t.replace('—', " ");
        t = re!(r"(?<![\d\s])[–](?![\d\s])").replace(&t, " ");
        re!(r"[ \t]{2,}").replace(&t, " ")
    }

    fn final_sweep(&self, text: &str) -> String {
        let s = |k| self.l.sym(k);
        let mut t = re!(r"<=|≤").replace(text, " less than or equal to ");
        t = re!(r">=|≥").replace(&t, " greater than or equal to ");
        t = re!(r"°\s*C\b").replace(&t, &format!(" {} ", s("celsius")));
        t = re!(r"°\s*F\b").replace(&t, &format!(" {} ", s("fahrenheit")));
        t = t.replace('°', &format!(" {} ", s("degree")));
        for (sym, key) in [
            ('%', "percent"),
            ('@', "at"),
            ('&', "amp"),
            ('=', "equals"),
            ('+', "plus"),
            ('×', "times"),
            ('÷', "divide"),
            ('±', "plus_minus"),
            ('√', "sqrt"),
            ('_', "underscore"),
            ('#', "hash"),
        ] {
            t = t.replace(sym, &format!(" {} ", s(key)));
        }
        t = t
            .replace('<', &format!(" {} ", s("lt")))
            .replace('>', &format!(" {} ", s("gt")));
        t = re!(r"[\^*]").replace(&t, " ");
        t = re!(r"\s*/\s*").replace(&t, " ");
        t = re!(r"\s*:\s*").replace(&t, ", ");
        t = re!(r"[₹$€£](?!\s?\d)").replace(&t, " ");
        t.replace(['–', '—'], " ")
    }

    fn digit_fallback(&self, text: &str) -> R<String> {
        let mut err = false;
        let out = re!(r"\d+").sub(text, |m| {
            let d = m.all();
            let r = if d.chars().count() >= 7 || d.starts_with('0') {
                self.digits_spelled(&d, true)
            } else {
                pint(&d).map(|n| self.l.cardinal(n))
            };
            r.unwrap_or_else(|_| {
                err = true;
                String::new()
            })
        });
        if err {
            Err(Exc::Other)
        } else {
            Ok(out)
        }
    }

    /// `Normalizer.normalize`.
    pub fn normalize(&self, text: &str) -> R<String> {
        self.validate(text)?;
        let raw = cv(&ascii_digits(text));
        let mut protected: Vec<(char, String)> = Vec::new();
        let mut originals: Vec<(char, String)> = Vec::new();
        let mut spans: Vec<(usize, usize, char)> = Vec::new();
        for f in [
            Self::m_email as fn(&Engine, &[char]) -> R<Vec<Span>>,
            Self::m_url,
        ] {
            for (s, e, spoken) in f(self, &raw)? {
                if !spans.iter().any(|&(a, b, _)| s < b && e > a) {
                    let key = char::from_u32(0xF0000 + protected.len() as u32).unwrap();
                    protected.push((key, spoken));
                    originals.push((key, sl(&raw, s, e)));
                    spans.push((s, e, key));
                }
            }
        }
        let mut sorted = spans.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        let mut raw = raw;
        for (s, e, key) in sorted {
            raw.splice(s..e, [key]);
        }
        let text = cv(&self.preclean(&st(&raw)));
        let mut matches = match self.gather(&text) {
            Ok(m) => m,
            Err(Exc::Norm {
                span: Some((a, b)), ..
            }) => {
                let mut pieces = [sl(&text, 0, a), sl(&text, a, b), sl(&text, b, text.len())];
                for (key, orig) in &originals {
                    for p in pieces.iter_mut() {
                        *p = p.replace(*key, orig);
                    }
                }
                let n0 = pieces[0].chars().count();
                let n1 = pieces[1].chars().count();
                return Err(Exc::Norm {
                    span: Some((n0, n0 + n1)),
                    source: Some(cv(&pieces.concat())),
                });
            }
            Err(e) => return Err(e),
        };
        matches.sort_by_key(|m| (m.0, m.1, std::cmp::Reverse(m.2 - m.1)));
        let mut taken: Vec<(usize, usize)> = Vec::new();
        let mut accepted: Vec<(usize, usize, String)> = Vec::new();
        for (_, s, e, rep) in matches {
            if taken.iter().any(|&(ts, te)| !(e <= ts || s >= te)) {
                continue;
            }
            taken.push((s, e));
            accepted.push((s, e, rep));
        }
        accepted.sort_by_key(|a| a.0);
        let mut out = String::new();
        let mut pos = 0usize;
        let pp = re!(r"[ \t]*-?([^\s\d,।.!?;:()\-—–]{1,14})");
        for (s, e, mut rep) in accepted {
            out += &sl(&text, pos, s);
            pos = e;
            loop {
                let rest = if pos < text.len() {
                    &text[pos..]
                } else {
                    &[][..]
                };
                let Some(nx) = pp.match_at(rest, 0) else {
                    break;
                };
                if sl(&text, pos + nx.end(), pos + nx.end() + 1) == "." {
                    break;
                }
                let Some(merged) = self.l.attach_postposition(&rep, &nx.s(1)) else {
                    break;
                };
                rep = merged;
                pos += nx.end();
            }
            out += &rep;
        }
        out += &sl(&text, pos, text.len());
        let mut r = self.final_sweep(&out);
        r = self.digit_fallback(&r)?;
        for (key, spoken) in &protected {
            r = r.replace(*key, spoken);
        }
        r = re!(r"(?<=[A-Za-z])(?=[ऀ-෿])").replace(&r, " ");
        r = re!(r"\s+([,।.!?;])").sub(&r, |m| m.s(1));
        r = re!(r",\s*([।.!?;,])").sub(&r, |m| m.s(1));
        r = re!(r",(?=\S)").replace(&r, ", ");
        r = re!(r"\s{2,}").replace(&r, " ");
        r = re!(r"\s+’").replace(&r, "’");
        if r.trim_end_matches(is_space).ends_with(',') {
            r = re!(r"[,\s]+$").replace(&r, ".");
        }
        if self.l.code == "hi" {
            r = r
                .chars()
                .filter(|&c| c != '\u{93c}')
                .map(|c| match c {
                    '\u{929}' => '\u{928}',
                    '\u{931}' => '\u{930}',
                    '\u{934}' => '\u{933}',
                    '\u{958}' => '\u{915}',
                    '\u{959}' => '\u{916}',
                    '\u{95a}' => '\u{917}',
                    '\u{95b}' => '\u{91c}',
                    '\u{95c}' => '\u{921}',
                    '\u{95d}' => '\u{922}',
                    '\u{95e}' => '\u{92b}',
                    '\u{95f}' => '\u{92f}',
                    c => c,
                })
                .collect();
        }
        Ok(strip(&r).to_string())
    }
}

fn chem_ok(tok: &str) -> bool {
    if tb::CHEM_WHITELIST.contains(&tok) {
        return true;
    }
    let pieces: Vec<String> = re!(r"[A-Z][a-z]?|\d+")
        .find_iter(&cv(tok))
        .iter()
        .map(|m| m.all())
        .collect();
    let isd = |p: &String| p.chars().all(is_digit);
    let letters: Vec<&String> = pieces.iter().filter(|p| !isd(p)).collect();
    let has_digits = pieces.iter().any(isd);
    if letters.is_empty() || letters.iter().any(|p| !tb::ELEMENTS.contains(&p.as_str())) {
        return false;
    }
    has_digits && letters.len() >= 2
}

fn is_cc(c: char) -> bool {
    (c as u32) < 0x20 || (0x7f..=0x9f).contains(&(c as u32))
}
fn is_co(c: char) -> bool {
    let u = c as u32;
    (0xE000..=0xF8FF).contains(&u)
        || (0xF0000..=0xFFFFD).contains(&u)
        || (0x100000..=0x10FFFD).contains(&u)
}

fn engine(code: &str) -> Option<&'static Engine> {
    static ENGINES: [OnceLock<Engine>; 12] = [const { OnceLock::new() }; 12];
    let i = tb::LANGS.iter().position(|l| l.code == code)?;
    Some(ENGINES[i].get_or_init(|| Engine::new(tb::LANGS[i])))
}

/// `resolve_language`: the request code (or a documented alias) -> table code.
pub fn resolve(lang: &str) -> Option<&'static str> {
    let l = lang.to_lowercase();
    let code = match l.as_str() {
        "en" | "en-in" | "english" | "indian english" | "en-us" | "en-gb" | "indian-english" => {
            "en"
        }
        "oriya" | "od" => "or",
        _ => {
            let names = [
                ("hi", "hindi"),
                ("te", "telugu"),
                ("ta", "tamil"),
                ("bn", "bengali"),
                ("as", "assamese"),
                ("gu", "gujarati"),
                ("kn", "kannada"),
                ("ml", "malayalam"),
                ("mr", "marathi"),
                ("or", "odia"),
                ("pa", "punjabi"),
            ];
            names
                .iter()
                .find(|(c, n)| l == *c || l == *n || l == format!("{c}-in"))
                .map(|x| x.0)?
        }
    };
    Some(code)
}

pub fn table(lang: &str) -> Option<&'static Engine> {
    engine(resolve(lang)?)
}

/// Module-level `normalize(text, language)` with the span fallback: an offending token is re-read character by
/// character and the text normalised again.
pub fn normalize(text: &str, lang: &str) -> R<String> {
    let eng = table(lang).ok_or(NERR)?;
    let mut text: Vec<char> = cv(text);
    loop {
        let r = eng
            .validate(&st(&text))
            .and_then(|_| eng.normalize(&st(&text)));
        match r {
            Ok(s) => return Ok(s),
            Err(Exc::Norm {
                span: Some((a, b)),
                source,
            }) => {
                let src = source.unwrap_or_else(|| text.clone());
                let token = &src[a..b];
                let mut rep = String::new();
                for &c in token {
                    if let Some(d) = decimal_value(c) {
                        rep += &format!(" {} ", eng.l.ones[d as usize]);
                    } else if let Some(k) = match c {
                        '/' => Some("slash"),
                        '.' => Some("dot"),
                        '-' => Some("minus"),
                        '+' => Some("plus"),
                        '@' => Some("at"),
                        '_' => Some("underscore"),
                        '₹' => Some("rupee"),
                        '$' => Some("dollar"),
                        '%' => Some("percent"),
                        _ => None,
                    } {
                        rep += &format!(" {} ", eng.l.sym(k));
                    } else if ",:;<>{}".contains(c) || is_cc(c) || is_co(c) {
                        rep.push(' ');
                    } else {
                        rep.push(c);
                    }
                }
                if rep == st(token) {
                    return Err(Exc::Other);
                }
                let mut nt = src[..a].to_vec();
                nt.push(' ');
                nt.extend(rep.chars());
                nt.push(' ');
                nt.extend_from_slice(&src[b..]);
                text = nt;
            }
            Err(e) => return Err(e),
        }
    }
}
