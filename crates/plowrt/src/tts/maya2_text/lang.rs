//! Per-language number / date / unit composition (maya3.text.tn LangBase and its subclasses).

pub type Map = &'static [(&'static str, &'static str)];
pub type List = &'static [&'static str];

pub struct LangData {
    pub code: &'static str,
    pub ones: List,
    pub hundred: &'static str,
    pub thousand: &'static str,
    pub lakh: &'static str,
    pub crore: &'static str,
    pub time_on_hour: &'static str,
    pub time_hm: &'static str,
    pub range: &'static str,
    pub range_to: &'static str,
    pub ratio: &'static str,
    pub power: &'static str,
    pub fraction: &'static str,
    pub decade_short: &'static str,
    pub decade_full: &'static str,
    pub ordinal_suffix: &'static str,
    pub number_word: &'static str,
    pub letters: List,
    pub months: List,
    pub month_aliases: Map,
    pub sym: Map,
    pub periods: Map,
    pub time_markers: List,
    pub year_words: List,
    pub id_keywords: List,
    pub id_after_words: List,
    pub label_words: List,
    pub minute_words: List,
    pub ordinal_input_suffixes: List,
    pub marker_suffixes: Map,
    pub period_detect: Map,
    pub scale_words: Map,
    pub currency_words: Map,
    pub unit_words: Map,
    pub abbrev_units: Map,
    pub ampm_words: Map,
    pub singular_nouns: Map,
    pub ordinal_exceptions: &'static [(u32, &'static str)],
    pub count_one: Option<&'static str>,
    pub extra_lists: &'static [(&'static str, List)],
    pub extra_maps: &'static [(&'static str, Map)],
}

pub fn get(m: Map, k: &str) -> Option<&'static str> {
    m.iter().find(|(a, _)| *a == k).map(|x| x.1)
}

/// Python `s[:-n]` / `s[n:]` by code point.
pub fn drop_last(s: &str, n: usize) -> &str {
    let mut it = s.char_indices().rev();
    let mut cut = s.len();
    for _ in 0..n {
        match it.next() {
            Some((i, _)) => cut = i,
            None => return "",
        }
    }
    &s[..cut]
}
pub fn drop_first(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[i..],
        None => "",
    }
}
fn last_char(s: &str) -> Option<char> {
    s.chars().last()
}
fn last_word(s: &str) -> &str {
    s.rsplit(' ').next().unwrap_or("")
}
fn ends_any(s: &str, xs: &[&str]) -> bool {
    xs.iter().any(|x| s.ends_with(x))
}

pub fn fmt(t: &str, k: &str, v: &str) -> String {
    t.replace(&format!("{{{k}}}"), v)
}

impl LangData {
    pub fn sym(&self, k: &str) -> &'static str {
        get(self.sym, k).unwrap_or_else(|| panic!("SYM[{k}] missing"))
    }
    pub fn sym_get(&self, k: &str) -> Option<&'static str> {
        get(self.sym, k)
    }
    fn xl(&self, k: &str) -> List {
        self.extra_lists
            .iter()
            .find(|(a, _)| *a == k)
            .map(|x| x.1)
            .unwrap_or(&[])
    }
    fn xm(&self, k: &str) -> Map {
        self.extra_maps
            .iter()
            .find(|(a, _)| *a == k)
            .map(|x| x.1)
            .unwrap_or(&[])
    }
    fn xs(&self, k: &str) -> &'static str {
        self.xl(k)[0]
    }
    fn xmi(&self, k: &str, i: u128) -> Option<&'static str> {
        get(self.xm(k), &i.to_string())
    }
    fn exc(&self, n: u128) -> Option<&'static str> {
        self.ordinal_exceptions
            .iter()
            .find(|(a, _)| *a as u128 == n)
            .map(|x| x.1)
    }

    pub fn digit(&self, d: u32) -> &'static str {
        self.ones[d as usize]
    }
    pub fn two_digit(&self, n: u128) -> &'static str {
        self.ones[n as usize]
    }

    fn hundreds(&self, h: u128, rest: u128) -> String {
        let tail = |mut s: String| {
            if rest > 0 {
                s.push(' ');
                s.push_str(self.two_digit(rest));
            }
            s
        };
        match self.code {
            "bn" | "mr" => {
                if self.code == "mr" && h == 1 && rest == 0 {
                    return "शंभर".into();
                }
                tail(format!("{}{}", self.ones[h as usize], self.hundred))
            }
            "as" => tail(self.xl("HUNDREDS")[h as usize].to_string()),
            "gu" => tail(self.xmi("HUNDREDS_FUSED", h).unwrap().to_string()),
            "kn" | "ta" => {
                if rest > 0 {
                    let parts = [
                        self.xl("HUNDREDS_COMB")[h as usize].to_string(),
                        self.two_digit(rest).to_string(),
                    ];
                    if self.code == "ta" {
                        self.sandhi_join(&parts)
                    } else {
                        parts.join(" ")
                    }
                } else {
                    self.xl("HUNDREDS")[h as usize].to_string()
                }
            }
            "ml" => {
                if rest > 0 {
                    format!(
                        "{} {}",
                        self.xmi("_H_COMB", h).unwrap(),
                        self.two_digit(rest)
                    )
                } else {
                    self.xmi("_H_STAND", h).unwrap().to_string()
                }
            }
            "te" => match (h == 1, rest > 0) {
                (true, true) => format!("నూట {}", self.ones[rest as usize]),
                (true, false) => "వంద".into(),
                (false, true) => format!(
                    "{} వందల {}",
                    self.ones[h as usize], self.ones[rest as usize]
                ),
                (false, false) => format!("{} వందలు", self.ones[h as usize]),
            },
            _ => tail(format!("{} {}", self.ones[h as usize], self.hundred)),
        }
    }

    fn under_thousand(&self, n: u128) -> String {
        if n < 100 {
            self.two_digit(n).to_string()
        } else {
            self.hundreds(n / 100, n % 100)
        }
    }

    fn sandhi_join(&self, parts: &[String]) -> String {
        let mut out: Vec<String> = parts.iter().filter(|p| !p.is_empty()).cloned().collect();
        for i in 0..out.len().saturating_sub(1) {
            if ends_any(&out[i], &["த்து", "ற்று"]) {
                let f = out[i + 1].chars().next().unwrap().to_string();
                if let Some(d) = get(self.xm("_DOUBLE"), &f) {
                    out[i].push_str(d);
                }
            }
        }
        out.join(" ")
    }

    pub fn cardinal_i(&self, n: i128) -> String {
        if n < 0 {
            format!("{} {}", self.sym("minus"), self.cardinal_i(-n))
        } else {
            self.cardinal(n as u128)
        }
    }

    pub fn cardinal(&self, n: u128) -> String {
        let crore = n / 10_000_000;
        let lakh = n % 10_000_000 / 100_000;
        let thousand = n % 100_000 / 1000;
        let low = n % 1000;
        let mut parts: Vec<String> = Vec::new();
        match self.code {
            "kn" => {
                if n < 1000 {
                    return self.under_thousand(n);
                }
                if crore > 0 {
                    parts.push(format!("{} {}", self.cardinal(crore), self.crore));
                }
                if lakh > 0 {
                    let w = if thousand > 0 || low > 0 {
                        self.xs("LAKH_COMB")
                    } else {
                        self.lakh
                    };
                    parts.push(format!("{} {}", self.two_digit(lakh), w));
                }
                if thousand > 0 {
                    let w = if low > 0 {
                        self.xs("THOUSAND_COMB")
                    } else {
                        self.thousand
                    };
                    if thousand == 1 && crore == 0 && lakh == 0 {
                        parts.push(w.to_string());
                    } else {
                        parts.push(format!("{} {}", self.under_thousand(thousand), w));
                    }
                }
                if low > 0 {
                    parts.push(self.under_thousand(low));
                }
                parts.join(" ")
            }
            "ml" => {
                if n < 100 {
                    return self.ones[n as usize].to_string();
                }
                if n < 1000 {
                    return self.hundreds(n / 100, n % 100);
                }
                let one = self.xs("_ONE_MULT");
                if crore > 0 {
                    let mult = if crore == 1 {
                        one.to_string()
                    } else {
                        self.cardinal(crore)
                    };
                    parts.push(format!("{mult} {}", self.crore));
                }
                if lakh > 0 {
                    let w = if thousand > 0 || low > 0 {
                        format!("{}ത്തി", drop_last(self.lakh, 1))
                    } else {
                        self.lakh.to_string()
                    };
                    let mult = if lakh == 1 { one } else { self.two_digit(lakh) };
                    parts.push(format!("{mult} {w}"));
                }
                if thousand > 0 {
                    parts.push(self.ml_thousand(thousand, low > 0));
                }
                if low > 0 {
                    parts.push(self.under_thousand(low));
                }
                parts.join(" ")
            }
            "ta" => {
                if n == 0 {
                    return self.ones[0].to_string();
                }
                if crore > 0 {
                    let mult = if crore == 1 {
                        "ஒரு".to_string()
                    } else {
                        self.cardinal(crore)
                    };
                    let w = if lakh > 0 || thousand > 0 || low > 0 {
                        "கோடியே"
                    } else {
                        "கோடி"
                    };
                    parts.push(format!("{mult} {w}"));
                }
                if lakh > 0 {
                    let mult = if lakh == 1 {
                        "ஒரு"
                    } else {
                        self.two_digit(lakh)
                    };
                    let w = if thousand > 0 || low > 0 {
                        "லட்சத்து"
                    } else {
                        "லட்சம்"
                    };
                    parts.push(format!("{mult} {w}"));
                }
                if thousand > 0 {
                    let base = if low > 0 {
                        "ஆயிரத்து"
                    } else {
                        "ஆயிரம்"
                    };
                    parts.push(if thousand == 1 {
                        base.to_string()
                    } else {
                        format!(
                            "{}ா{}",
                            drop_last(self.ones[thousand as usize], 1),
                            drop_first(base, 1)
                        )
                    });
                }
                if low > 0 {
                    parts.push(self.under_thousand(low));
                }
                self.sandhi_join(&parts)
            }
            "te" => {
                if n < 1000 {
                    return self.under_thousand(n);
                }
                if crore > 0 {
                    let rest = lakh > 0 || thousand > 0 || low > 0;
                    parts.push(if crore == 1 {
                        (if rest { "కోటీ" } else { "కోటి" }).to_string()
                    } else {
                        format!(
                            "{} {}",
                            self.cardinal(crore),
                            if rest {
                                "కోట్ల"
                            } else {
                                "కోట్లు"
                            }
                        )
                    });
                }
                if lakh > 0 {
                    let rest = thousand > 0 || low > 0;
                    parts.push(if lakh == 1 {
                        (if rest {
                            "లక్షా"
                        } else {
                            "లక్ష"
                        })
                        .to_string()
                    } else {
                        format!(
                            "{} {}",
                            self.ones[lakh as usize],
                            if rest {
                                "లక్షల"
                            } else {
                                "లక్షలు"
                            }
                        )
                    });
                }
                if thousand > 0 {
                    parts.push(if thousand == 1 {
                        "వెయ్యి".to_string()
                    } else {
                        format!(
                            "{} {}",
                            self.ones[thousand as usize],
                            if low > 0 { "వేల" } else { "వేలు" }
                        )
                    });
                }
                if low > 0 {
                    parts.push(self.under_thousand(low));
                }
                parts.join(" ")
            }
            _ => {
                if n < 1000 {
                    return self.under_thousand(n);
                }
                if crore > 0 {
                    parts.push(format!("{} {}", self.cardinal(crore), self.crore));
                }
                if lakh > 0 {
                    parts.push(format!("{} {}", self.two_digit(lakh), self.lakh));
                }
                if thousand > 0 {
                    parts.push(format!("{} {}", self.two_digit(thousand), self.thousand));
                }
                if low > 0 {
                    parts.push(self.under_thousand(low));
                }
                parts.join(" ")
            }
        }
    }

    fn ml_thousand(&self, m: u128, combining: bool) -> String {
        let word = match self.xmi("_TH_STAND", m) {
            Some(w) => w.to_string(),
            None => {
                let base = self.ones[m as usize];
                if base.ends_with("ത്") {
                    format!("{}ിനായിരം", drop_last(base, 1))
                } else if base.ends_with('്') {
                    format!("{}ായിരം", drop_last(base, 1))
                } else {
                    format!("{base} {}", self.thousand)
                }
            }
        };
        if combining {
            format!("{}ത്തി", drop_last(&word, 1))
        } else {
            word
        }
    }

    pub fn year(&self, y: u128) -> String {
        if !(1100..=1999).contains(&y) || matches!(self.code, "as" | "kn" | "ml" | "ta") {
            return self.cardinal(y);
        }
        let (hi, lo) = (y / 100, y % 100);
        let head = match self.code {
            "bn" | "mr" => format!("{}{}", self.two_digit(hi), self.hundred),
            "gu" => {
                let w = self.two_digit(hi);
                if w.ends_with('સ') {
                    format!("{}{}", drop_last(w, 1), self.hundred)
                } else {
                    format!("{w}{}", self.hundred)
                }
            }
            "te" => {
                return if lo > 0 {
                    format!("{} వందల {}", self.ones[hi as usize], self.ones[lo as usize])
                } else {
                    format!("{} వందలు", self.ones[hi as usize])
                };
            }
            _ => format!("{} {}", self.two_digit(hi), self.hundred),
        };
        if lo > 0 {
            format!("{head} {}", self.two_digit(lo))
        } else {
            head
        }
    }

    /// `LETTERS.get(c.upper(), c)`.
    pub fn letter(&self, c: &str) -> String {
        let up = c.to_uppercase();
        let mut it = up.chars();
        if let (Some(u), None) = (it.next(), it.next()) {
            if u.is_ascii_uppercase() {
                return self.letters[(u as u8 - b'A') as usize].to_string();
            }
        }
        c.to_string()
    }
    pub fn letter_c(&self, c: char) -> String {
        self.letter(c.encode_utf8(&mut [0; 4]))
    }

    pub fn month(&self, i: u32) -> &'static str {
        self.months[(i as usize + 11) % 12]
    }

    pub fn time_phrase(&self, h: u32, m: u32, period: Option<&str>) -> String {
        match self.code {
            "en" => {
                let core = if h == 0 {
                    format!(
                        "zero {}",
                        if m == 0 {
                            "hundred"
                        } else {
                            self.two_digit(m as u128)
                        }
                    )
                } else {
                    self.time_phrase_base(h, m, None)
                };
                match period {
                    Some(p) => format!("{core} {}", get(self.periods, p).unwrap()),
                    None => core,
                }
            }
            "te" => {
                let h12 = if h % 12 == 0 { 12 } else { h % 12 };
                let hw = if h12 == 1 {
                    "ఒక"
                } else {
                    self.two_digit(h12 as u128)
                };
                let hnoun = if h12 == 1 {
                    "గంట"
                } else if m == 0 {
                    "గంటలు"
                } else {
                    "గంటల"
                };
                let core = match m {
                    0 => format!("{hw} {hnoun}"),
                    1 => format!("{hw} {hnoun} ఒక నిమిషం"),
                    _ => format!("{hw} {hnoun} {} నిమిషాలు", self.two_digit(m as u128)),
                };
                match period {
                    Some(p) => format!("{} {core}", get(self.periods, p).unwrap()),
                    None => core,
                }
            }
            _ => self.time_phrase_base(h, m, period),
        }
    }

    fn time_phrase_base(&self, h: u32, m: u32, period: Option<&str>) -> String {
        let h12 = if !self.ampm_words.is_empty() && h > 12 {
            h
        } else if h % 12 == 0 {
            12
        } else {
            h % 12
        };
        let core = if m == 0 {
            fmt(self.time_on_hour, "h", self.two_digit(h12 as u128))
        } else {
            fmt(
                &fmt(self.time_hm, "h", self.two_digit(h12 as u128)),
                "m",
                self.two_digit(m as u128),
            )
        };
        match period {
            Some(p) => format!("{} {core}", get(self.periods, p).unwrap()),
            None => core,
        }
    }

    pub fn ordinal(&self, n: u128, src: &str) -> String {
        if src.is_empty() {
            if let Some(e) = self.exc(n) {
                return e.to_string();
            }
        }
        self.ordinal_compose(n, src)
    }

    fn ordinal_compose(&self, n: u128, src: &str) -> String {
        let card = || self.cardinal(n);
        match self.code {
            "en" => {
                if n < 100 {
                    return if n == 0 {
                        "zeroth".into()
                    } else {
                        self.exc(n).unwrap().to_string()
                    };
                }
                for (scale, word) in [
                    (10_000_000u128, self.crore),
                    (100_000, self.lakh),
                    (1000, self.thousand),
                    (100, self.hundred),
                ] {
                    if n >= scale {
                        let (count, rest) = (n / scale, n % scale);
                        let head = format!("{} {word}", self.cardinal(count));
                        return if rest > 0 {
                            format!("{head} {}", self.ordinal(rest, ""))
                        } else {
                            head + "th"
                        };
                    }
                }
                unreachable!()
            }
            "hi" => {
                if !src.is_empty() {
                    if let Some(e) = self.exc(n) {
                        let ending = match src {
                            "वीं" => "ी",
                            "वें" | "वे" => "े",
                            _ => "ा",
                        };
                        return format!("{}{ending}", drop_last(e, 1));
                    }
                    return format!("{}{src}", card());
                }
                self.exc(n)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("{}{}", card(), self.ordinal_suffix))
            }
            "bn" | "as" => {
                if matches!(src, "লা" | "রা" | "ঠা") {
                    if let Some(d) = self.xmi("_DATE_ORDINALS", n) {
                        return d.to_string();
                    }
                }
                match src {
                    "ই" => format!("{}ই", card()),
                    "শে" => format!("{}ে", card()),
                    "তম" => format!("{}তম", card()),
                    _ => self
                        .exc(n)
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("{}{}", card(), self.ordinal_suffix)),
                }
            }
            "gu" => {
                if let Some(stem) = self.xmi("ORDINAL_STEMS", n) {
                    if src.chars().count() > 1 {
                        return format!("{stem}{}", drop_first(src, 1));
                    }
                    return self.exc(n).unwrap().to_string();
                }
                if src.starts_with('મ') {
                    return format!("{}{src}", card());
                }
                format!("{}{}", card(), self.ordinal_suffix)
            }
            "kn" | "ml" => {
                let c = card();
                let strip: &[char] = if self.code == "kn" {
                    &['ು']
                } else {
                    &['്', 'ം']
                };
                let base = if last_char(&c).is_some_and(|l| strip.contains(&l)) {
                    drop_last(&c, 1)
                } else {
                    &c
                };
                let suffix = if self.ordinal_input_suffixes.contains(&src) {
                    src
                } else {
                    self.ordinal_suffix
                };
                format!("{base}{suffix}")
            }
            "mr" | "pa" => {
                if !src.is_empty() {
                    return format!("{}{src}", card());
                }
                self.exc(n)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("{}{}", card(), self.ordinal_suffix))
            }
            "ta" => {
                let suffix = if matches!(src, "ஆம்" | "ம்") {
                    "ஆம்"
                } else {
                    "ஆவது"
                };
                if n == 1 {
                    return if suffix == "ஆம்" {
                        "முதலாம்".into()
                    } else {
                        "முதலாவது".into()
                    };
                }
                let base = card();
                match last_char(&base) {
                    Some('ு' | '்') => {
                        format!("{}ா{}", drop_last(&base, 1), drop_first(suffix, 1))
                    }
                    Some('ி' | 'ீ' | 'ை') => format!("{base}யா{}", drop_first(suffix, 1)),
                    _ => base + suffix,
                }
            }
            "te" => {
                if src.is_empty() {
                    if let Some(e) = self.exc(n) {
                        return e.to_string();
                    }
                }
                let w = card();
                let stem = if w.ends_with('ు') || w.ends_with('ి') {
                    drop_last(&w, 1).to_string()
                } else if w.ends_with('ై') {
                    format!("{}య్య", drop_last(&w, 1))
                } else {
                    w
                };
                format!("{stem}{}", self.ordinal_suffix)
            }
            _ => {
                if src.is_empty() {
                    if let Some(e) = self.exc(n) {
                        return e.to_string();
                    }
                }
                let suffix = if self.ordinal_input_suffixes.contains(&src) {
                    src
                } else {
                    self.ordinal_suffix
                };
                format!("{}{suffix}", card())
            }
        }
    }

    /// `decimal(int_part, frac_digits)`; `frac` holds ASCII digits.
    pub fn decimal(&self, ip: u128, frac: &str) -> String {
        let f: Vec<&str> = frac
            .chars()
            .map(|d| self.digit(d.to_digit(10).unwrap()))
            .collect();
        let c = self.cardinal(ip);
        let c = if self.code == "te" {
            oblique_last(&c).to_string()
        } else {
            c
        };
        format!("{c} {} {}", self.sym("point"), f.join(" "))
    }

    pub fn decade(&self, tens: u128, full: bool) -> String {
        let w = self.two_digit(tens);
        let stem = match self.code {
            "kn" if w.ends_with('ು') => drop_last(w, 1).to_string(),
            "ml" if w.ends_with('്') => format!("{}ു", drop_last(w, 1)),
            _ => w.to_string(),
        };
        fmt(
            if full {
                self.decade_full
            } else {
                self.decade_short
            },
            "tens",
            &stem,
        )
    }

    pub fn join_amount(&self, words: &str, noun: &str) -> String {
        if self.code == "te" {
            if words == "ఒకటి" {
                return format!(
                    "ఒక {}",
                    get(self.xm("_NOUN_SINGULAR"), noun).unwrap_or(noun)
                );
            }
            return format!("{} {noun}", oblique_last(words));
        }
        if words == self.ones[1] {
            return format!(
                "{} {}",
                self.count_one.unwrap_or(words),
                get(self.singular_nouns, noun).unwrap_or(noun)
            );
        }
        format!("{words} {noun}")
    }

    pub fn scale_noun(&self, canon: &str, numstr: &str) -> String {
        if self.code == "te" {
            let one = numstr.replace(',', "") == "1";
            return match canon {
                "crore" => {
                    if one {
                        "కోటి"
                    } else {
                        "కోట్లు"
                    }
                }
                "lakh" => {
                    if one {
                        "లక్ష"
                    } else {
                        "లక్షలు"
                    }
                }
                "thousand" => {
                    if one {
                        "వెయ్యి"
                    } else {
                        "వేలు"
                    }
                }
                _ => canon,
            }
            .to_string();
        }
        match canon {
            "crore" => self.crore,
            "lakh" => self.lakh,
            "thousand" => self.thousand,
            _ => canon,
        }
        .to_string()
    }

    pub fn attach_postposition(&self, rep: &str, tok: &str) -> Option<String> {
        let t = tok.trim_start_matches('-');
        let last = last_word(rep);
        let base = &rep[..rep.len() - last.len()];
        let in_list = |k: &str, w: &str| self.xl(k).contains(&w);
        match self.code {
            "as" => (super::tables::AS_ATTACH.contains(&last) && super::tables::AS_PP.contains(&t))
                .then(|| format!("{rep}{t}")),
            "bn" => {
                if !in_list("_ATTACH_NOUNS", last) {
                    return None;
                }
                if t == last {
                    return Some(rep.into());
                }
                let vow = ends_any(last, &["া", "ো", "ে"]);
                match t {
                    "এ" | "তে" | "য়" => {
                        Some(format!("{rep}{}", if vow { "য়" } else { "ে" }))
                    }
                    "এর" | "র" => Some(format!("{rep}{}", if vow { "র" } else { "ের" })),
                    "মধ্যে" | "নাগাদ" | "থেকে" | "পর্যন্ত" | "দিয়ে" => {
                        Some(format!("{rep}{} {t}", if vow { "র" } else { "ের" }))
                    }
                    _ => None,
                }
            }
            "gu" | "or" => {
                let noun = in_list("_ATTACH_NOUNS", last);
                if t == last && noun {
                    return Some(rep.into());
                }
                (in_list("_PP", t) && noun).then(|| format!("{rep}{t}"))
            }
            "kn" => {
                if !last_char(last).is_some_and(|c| ('\u{c80}'..='\u{cff}').contains(&c)) {
                    return None;
                }
                if t == last {
                    return Some(rep.into());
                }
                if !in_list("_PP", t) {
                    return None;
                }
                let dative = matches!(t, "ಗೆ" | "ಕ್ಕೆ");
                let ablative = matches!(t, "ಇಂದ" | "ದಿಂದ" | "ರಿಂದ");
                let locative = matches!(t, "ನಲ್ಲಿ" | "ಅಲ್ಲಿ" | "ರಲ್ಲಿ");
                let pick = |d: &'static str, a: &'static str, l: &'static str| {
                    if dative {
                        d
                    } else if ablative {
                        a
                    } else if locative {
                        l
                    } else {
                        t
                    }
                };
                if last.ends_with('\u{cc1}') {
                    return Some(format!(
                        "{base}{}{}",
                        drop_last(last, 1),
                        pick("ಕ್ಕೆ", "ರಿಂದ", "ರಲ್ಲಿ")
                    ));
                }
                if ends_any(last, &["\u{cc6}", "\u{cbf}", "\u{cc0}"]) {
                    return Some(format!("{base}{last}{}", pick("ಗೆ", "ಯಿಂದ", "ಯಲ್ಲಿ")));
                }
                Some(format!("{base}{last}{}", pick("ಕ್ಕೆ", "ದಿಂದ", "ದಲ್ಲಿ")))
            }
            "ml" => {
                if !last_char(last).is_some_and(|c| ('\u{d00}'..='\u{d7f}').contains(&c)) {
                    return None;
                }
                let virama = last.ends_with('്');
                let anusvara = last.ends_with('ം');
                let ivow = matches!(last_char(last), Some('ി' | 'ീ'));
                let stem = drop_last(last, 1);
                match t {
                    "ന്റെ" | "ുടെ" | "യുടെ" => Some(if virama {
                        format!("{base}{stem}ിന്റെ")
                    } else if anusvara {
                        format!("{base}{stem}ത്തിന്റെ")
                    } else {
                        format!("{base}{last}യുടെ")
                    }),
                    "ൽ" | "ിൽ" | "ലെ" | "ിലെ" => {
                        let end = if matches!(t, "ലെ" | "ിലെ") {
                            "ിലെ"
                        } else {
                            "ിൽ"
                        };
                        Some(if virama {
                            format!("{base}{stem}{end}")
                        } else if anusvara {
                            format!("{base}{stem}ത്ത{end}")
                        } else {
                            format!("{base}{last}യ{end}")
                        })
                    }
                    "ലേക്ക്" | "ിലേക്ക്" => Some(if virama {
                        format!("{base}{stem}ിലേക്ക്")
                    } else if anusvara {
                        format!("{base}{stem}ത്തിലേക്ക്")
                    } else {
                        format!("{base}{last}യിലേക്ക്")
                    }),
                    "ന്" | "ിന്" | "ക്ക്" | "ിനാണ്" => {
                        let tail = if t == "ിനാണ്" {
                            "ിനാണ്"
                        } else {
                            "ിന്"
                        };
                        Some(if virama {
                            format!("{base}{stem}{tail}")
                        } else if anusvara {
                            format!("{base}{stem}ത്ത{tail}")
                        } else if ivow {
                            format!("{base}{last}ക്ക്")
                        } else {
                            format!("{base}{last}യ്ക്ക്")
                        })
                    }
                    _ => None,
                }
            }
            "mr" => {
                let t = t.trim_start_matches(['ा', 'ं']);
                let case = get(self.xm("_CASE"), t);
                if !in_list("_ATTACH_NOUNS", last) || case.is_none() {
                    return None;
                }
                let obl = if last.ends_with('े') {
                    format!("{}ां", drop_last(last, 1))
                } else {
                    last.to_string()
                };
                Some(format!("{base}{obl}{}", case.unwrap()))
            }
            "ta" => {
                const NUM_OBL: &[(&str, &str)] = &[
                    ("ஒன்று", "ஒன்ற"),
                    ("இரண்டு", "இரண்ட"),
                    ("மூன்று", "மூன்ற"),
                    ("நான்கு", "நான்க"),
                    ("ஐந்து", "ஐந்த"),
                    ("ஆறு", "ஆற"),
                    ("ஏழு", "ஏழ"),
                    ("எட்டு", "எட்ட"),
                    ("ஒன்பது", "ஒன்பத"),
                    ("பத்து", "பத்த"),
                    ("பூஜ்ஜியம்", "பூஜ்ஜியத்த"),
                ];
                if let Some(o) = get(NUM_OBL, last) {
                    return match t {
                        "இல்" | "ல்" => Some(format!("{base}{o}ில்")),
                        "க்கு" | "கு" => Some(format!("{base}{o}ிற்கு")),
                        "இலிருந்து" | "ல்இருந்து" => {
                            Some(format!("{base}{o}ிலிருந்து"))
                        }
                        _ => None,
                    };
                }
                if !in_list("_ATTACH_NOUNS", last) {
                    return None;
                }
                match t {
                    "க்கு" | "கு" => Some(match last {
                        "நிமிடம்" => format!("{base}நிமிடத்திற்கு"),
                        "சதவீதம்" => format!("{base}சதவீதத்திற்கு"),
                        _ => format!("{base}{last}க்கு"),
                    }),
                    "இல்" | "ல்" | "இலிருந்து" | "ல்இருந்து" => {
                        Some(if last.ends_with("ம்") {
                            format!("{base}{}த்தில்", drop_last(last, 2))
                        } else {
                            format!("{base}{last}இல்")
                        })
                    }
                    _ => None,
                }
            }
            "te" => self.te_attach(rep, tok),
            _ => None,
        }
    }

    fn te_attach(&self, rep: &str, tok: &str) -> Option<String> {
        let tok = get(self.xm("_PP_ROMAN"), &tok.to_lowercase()).unwrap_or(tok);
        let last = last_word(rep);
        let first = tok.chars().next();
        if rep == "ఒకటి" && first.is_some_and(|c| c.is_ascii_alphabetic()) {
            return Some(format!("ఒక {tok}"));
        }
        if !last_char(last).is_some_and(|c| ('\u{c00}'..='\u{c7f}').contains(&c)) {
            return None;
        }
        if self.letters.contains(&last) {
            return None;
        }
        if tok == last {
            return Some(rep.into());
        }
        let has = |k: &str, w: &str| self.xl(k).contains(&w);
        let plural = last.ends_with("లు");
        if has("_PP_NOMINATIVE", tok) {
            return Some(format!("{rep}{tok}"));
        }
        let mut base_tok = tok.replace('\u{200c}', "");
        for end in ["ను", "ని", "కు", "కి", "లో", "గా", "తో", "పై"]
        {
            if base_tok.ends_with(end) {
                let stem = drop_last(&base_tok, end.chars().count());
                if has("_COUNT_NOUNS", stem) {
                    base_tok = stem.to_string();
                    break;
                }
            }
        }
        let rep1 = drop_last(rep, 1);
        if plural && has("_COUNT_NOUNS", &base_tok) {
            return Some(format!("{rep1} {tok}"));
        }
        if rep == "ఒకటి" && has("_COUNT_NOUNS", tok) {
            return Some(format!(
                "ఒక {}",
                get(self.xm("_NOUN_SINGULAR"), tok).unwrap_or(tok)
            ));
        }
        if has("_PP_SPACED", tok) {
            return Some(format!("{} {tok}", if plural { rep1 } else { rep }));
        }
        if !has("_PP_GLUE", tok) {
            return None;
        }
        if last == "శాతం" && matches!(tok, "కి" | "కు") {
            return Some(format!("{rep1}ానికి"));
        }
        if plural {
            if tok == "ని" {
                return Some(format!("{rep1}ను"));
            }
            return Some(format!("{rep1}{tok}"));
        }
        Some(format!("{rep}{tok}"))
    }
}

/// Telugu `_oblique_last`: వందలు -> వందల.
fn oblique_last(words: &str) -> &str {
    if last_word(words).ends_with("లు") {
        drop_last(words, 1)
    } else {
        words
    }
}
