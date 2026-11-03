//! The subset of Python `re` the normaliser uses: a backtracking matcher over `char` slices (Python indexes
//! strings by code point) with Python's Unicode `\w`/`\d`/`\s`, ordered alternation, fixed-width lookbehind,
//! backreferences, IGNORECASE, and `finditer`/`sub`/`split`'s empty-match rule (an empty match may not end where
//! the previous match ended when that match was empty).

use super::tables;

fn in_ranges(c: char, r: &[(u32, u32)]) -> bool {
    let c = c as u32;
    r.binary_search_by(|&(a, b)| {
        if b < c {
            std::cmp::Ordering::Less
        } else if a > c {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

pub fn is_alnum(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric();
    }
    in_ranges(c, tables::ALNUM)
}
pub fn is_word(c: char) -> bool {
    c == '_' || is_alnum(c)
}
pub fn is_alpha(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    in_ranges(c, tables::ALPHA)
}
pub fn is_decimal(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_ranges(c, tables::DECIMAL)
}
pub fn is_digit(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    in_ranges(c, tables::DIGIT)
}
pub fn is_space(c: char) -> bool {
    if c.is_ascii() {
        return matches!(
            c,
            ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r' | '\x1c'..='\x1f'
        );
    }
    in_ranges(c, tables::SPACE)
}
/// `unicodedata.decimal(c)`; decimal digits come in runs that start at zero.
pub fn decimal_value(c: char) -> Option<u32> {
    let u = c as u32;
    let r = tables::DECIMAL;
    let i = r.partition_point(|&(_, b)| b < u);
    (i < r.len() && r[i].0 <= u).then(|| (u - r[i].0) % 10)
}

pub fn lower_char(c: char) -> char {
    let mut l = c.to_lowercase();
    match (l.next(), l.next()) {
        (Some(a), None) => a,
        _ => c,
    }
}
fn upper_char(c: char) -> char {
    let mut l = c.to_uppercase();
    match (l.next(), l.next()) {
        (Some(a), None) => a,
        _ => c,
    }
}

#[derive(Debug)]
enum Item {
    Ch(char),
    Range(char, char),
    Word(bool),
    Digit(bool),
    Space(bool),
}

#[derive(Debug)]
struct Set {
    neg: bool,
    items: Vec<Item>,
}

impl Set {
    fn has(&self, c: char) -> bool {
        self.items.iter().any(|it| match *it {
            Item::Ch(x) => x == c,
            Item::Range(a, b) => a <= c && c <= b,
            Item::Word(p) => is_word(c) == p,
            Item::Digit(p) => is_decimal(c) == p,
            Item::Space(p) => is_space(c) == p,
        })
    }
    fn matches(&self, c: char, icase: bool) -> bool {
        let hit = self.has(c) || (icase && (self.has(lower_char(c)) || self.has(upper_char(c))));
        hit != self.neg
    }
}

#[derive(Debug)]
enum Node {
    Lit(char),
    Any,
    Set(Box<Set>),
    Bol,
    Eol,
    WordB(bool),
    Group(usize, Box<Node>),
    Cat(Vec<Node>),
    Alt(Vec<Node>),
    Rep(Box<Node>, u32, u32, bool),
    Look(Box<Node>, bool, bool, usize),
    Backref(usize),
}

const INF: u32 = u32::MAX;

pub struct Regex {
    root: Node,
    ngroups: usize,
    names: Vec<(String, usize)>,
    icase: bool,
}

struct Parser<'a> {
    p: &'a [char],
    i: usize,
    ngroups: usize,
    names: Vec<(String, usize)>,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.p.get(self.i).copied()
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn alt(&mut self) -> Node {
        let mut alts = vec![self.cat()];
        while self.eat('|') {
            alts.push(self.cat());
        }
        if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Node::Alt(alts)
        }
    }
    fn cat(&mut self) -> Node {
        let mut v = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom();
            let atom = self.quant(atom);
            v.push(atom);
        }
        if v.len() == 1 {
            v.pop().unwrap()
        } else {
            Node::Cat(v)
        }
    }
    fn num(&mut self) -> Option<u32> {
        let s = self.i;
        while matches!(self.peek(), Some('0'..='9')) {
            self.i += 1;
        }
        (self.i > s).then(|| {
            self.p[s..self.i]
                .iter()
                .collect::<String>()
                .parse()
                .unwrap()
        })
    }
    fn quant(&mut self, atom: Node) -> Node {
        let (min, max) = match self.peek() {
            Some('*') => {
                self.i += 1;
                (0, INF)
            }
            Some('+') => {
                self.i += 1;
                (1, INF)
            }
            Some('?') => {
                self.i += 1;
                (0, 1)
            }
            Some('{') => {
                let save = self.i;
                self.i += 1;
                let a = self.num();
                let r = if self.eat(',') {
                    let b = self.num();
                    if self.eat('}') {
                        Some((a.unwrap_or(0), b.unwrap_or(INF)))
                    } else {
                        None
                    }
                } else if a.is_some() && self.eat('}') {
                    Some((a.unwrap(), a.unwrap()))
                } else {
                    None
                };
                match r {
                    Some(r) => r,
                    None => {
                        self.i = save;
                        return atom;
                    }
                }
            }
            _ => return atom,
        };
        let greedy = !self.eat('?');
        self.quant(Node::Rep(Box::new(atom), min, max, greedy))
    }
    fn hex(&mut self, n: usize) -> char {
        let s: String = self.p[self.i..self.i + n].iter().collect();
        self.i += n;
        char::from_u32(u32::from_str_radix(&s, 16).unwrap()).unwrap()
    }
    /// An escape after `\`; `None` for a class escape returned via `cls`.
    fn escape(&mut self, in_class: bool) -> Result<char, Item> {
        let c = self.p[self.i];
        self.i += 1;
        match c {
            'w' => Err(Item::Word(true)),
            'W' => Err(Item::Word(false)),
            'd' => Err(Item::Digit(true)),
            'D' => Err(Item::Digit(false)),
            's' => Err(Item::Space(true)),
            'S' => Err(Item::Space(false)),
            'n' => Ok('\n'),
            't' => Ok('\t'),
            'r' => Ok('\r'),
            'f' => Ok('\x0c'),
            'v' => Ok('\x0b'),
            'u' => Ok(self.hex(4)),
            'U' => Ok(self.hex(8)),
            'x' => Ok(self.hex(2)),
            c if c.is_ascii_alphanumeric() && !(in_class && c == 'b') => {
                panic!("unsupported escape \\{c}")
            }
            c => Ok(c),
        }
    }
    fn atom(&mut self) -> Node {
        let c = self.p[self.i];
        self.i += 1;
        match c {
            '.' => Node::Any,
            '^' => Node::Bol,
            '$' => Node::Eol,
            '[' => self.class(),
            '(' => {
                if self.eat('?') {
                    if self.eat(':') {
                        let n = self.alt();
                        assert!(self.eat(')'));
                        n
                    } else if self.eat('=') || self.peek() == Some('!') {
                        let neg = self.eat('!');
                        let n = self.alt();
                        assert!(self.eat(')'));
                        Node::Look(Box::new(n), true, neg, 0)
                    } else if self.eat('<') {
                        let neg = if self.eat('!') {
                            true
                        } else {
                            assert!(self.eat('='));
                            false
                        };
                        let n = self.alt();
                        assert!(self.eat(')'));
                        let w = width(&n);
                        Node::Look(Box::new(n), false, neg, w)
                    } else if self.eat('P') {
                        assert!(self.eat('<'));
                        let s = self.i;
                        while self.p[self.i] != '>' {
                            self.i += 1;
                        }
                        let name: String = self.p[s..self.i].iter().collect();
                        self.i += 1;
                        self.ngroups += 1;
                        let g = self.ngroups;
                        self.names.push((name, g));
                        let n = self.alt();
                        assert!(self.eat(')'));
                        Node::Group(g, Box::new(n))
                    } else {
                        panic!("unsupported group syntax")
                    }
                } else {
                    self.ngroups += 1;
                    let g = self.ngroups;
                    let n = self.alt();
                    assert!(self.eat(')'));
                    Node::Group(g, Box::new(n))
                }
            }
            '\\' => {
                let e = self.p[self.i];
                match e {
                    'b' => {
                        self.i += 1;
                        Node::WordB(true)
                    }
                    'B' => {
                        self.i += 1;
                        Node::WordB(false)
                    }
                    'A' => {
                        self.i += 1;
                        Node::Bol
                    }
                    '1'..='9' => {
                        self.i += 1;
                        Node::Backref(e as usize - '0' as usize)
                    }
                    _ => match self.escape(false) {
                        Ok(c) => Node::Lit(c),
                        Err(it) => Node::Set(Box::new(Set {
                            neg: false,
                            items: vec![it],
                        })),
                    },
                }
            }
            c => Node::Lit(c),
        }
    }
    fn class_char(&mut self) -> Result<char, Item> {
        let c = self.p[self.i];
        self.i += 1;
        if c == '\\' {
            self.escape(true)
        } else {
            Ok(c)
        }
    }
    fn class(&mut self) -> Node {
        let neg = self.eat('^');
        let mut items = Vec::new();
        let mut first = true;
        loop {
            let c = self.p[self.i];
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            match self.class_char() {
                Err(it) => items.push(it),
                Ok(a) => {
                    if self.peek() == Some('-') && self.p.get(self.i + 1).is_some_and(|&n| n != ']')
                    {
                        self.i += 1;
                        match self.class_char() {
                            Ok(b) => items.push(Item::Range(a, b)),
                            Err(_) => panic!("bad class range"),
                        }
                    } else {
                        items.push(Item::Ch(a));
                    }
                }
            }
        }
        Node::Set(Box::new(Set { neg, items }))
    }
}

fn width(n: &Node) -> usize {
    match n {
        Node::Lit(_) | Node::Any | Node::Set(_) => 1,
        Node::Bol | Node::Eol | Node::WordB(_) | Node::Look(..) => 0,
        Node::Group(_, n) => width(n),
        Node::Cat(v) => v.iter().map(width).sum(),
        Node::Alt(v) => {
            let w = width(&v[0]);
            assert!(v.iter().all(|x| width(x) == w), "variable-width lookbehind");
            w
        }
        Node::Rep(n, a, b, _) => {
            assert_eq!(a, b, "variable-width lookbehind");
            width(n) * *a as usize
        }
        Node::Backref(_) => panic!("backref in lookbehind"),
    }
}

struct St<'t> {
    t: &'t [char],
    caps: Vec<(usize, usize)>,
    icase: bool,
}

const UNSET: (usize, usize) = (usize::MAX, usize::MAX);

impl St<'_> {
    fn eq(&self, a: char, b: char) -> bool {
        a == b || (self.icase && lower_char(a) == lower_char(b))
    }
    fn one(&self, n: &Node, i: usize) -> bool {
        let Some(&c) = self.t.get(i) else {
            return false;
        };
        match n {
            Node::Lit(x) => self.eq(c, *x),
            Node::Any => c != '\n',
            Node::Set(s) => s.matches(c, self.icase),
            _ => unreachable!(),
        }
    }
    fn boundary(&self, i: usize) -> bool {
        let a = i > 0 && is_word(self.t[i - 1]);
        let b = i < self.t.len() && is_word(self.t[i]);
        a != b
    }
}

type K<'k, 't> = &'k mut dyn FnMut(&mut St<'t>, usize) -> bool;

fn m<'t>(st: &mut St<'t>, n: &Node, i: usize, k: K<'_, 't>) -> bool {
    match n {
        Node::Lit(_) | Node::Any | Node::Set(_) => st.one(n, i) && k(st, i + 1),
        Node::Bol => i == 0 && k(st, i),
        Node::Eol => {
            let l = st.t.len();
            (i == l || (i + 1 == l && st.t[i] == '\n')) && k(st, i)
        }
        Node::WordB(b) => st.boundary(i) == *b && k(st, i),
        Node::Group(g, inner) => {
            let g = *g;
            m(st, inner, i, &mut |st: &mut St<'t>, j| {
                let old = st.caps[g];
                st.caps[g] = (i, j);
                if k(st, j) {
                    true
                } else {
                    st.caps[g] = old;
                    false
                }
            })
        }
        Node::Cat(v) => seq(st, v, i, k),
        Node::Alt(v) => v.iter().any(|a| m(st, a, i, k)),
        Node::Rep(inner, min, max, greedy) => {
            if matches!(**inner, Node::Lit(_) | Node::Any | Node::Set(_)) {
                let (min, max) = (*min as usize, *max as usize);
                let mut n = 0;
                while n < max && st.one(inner, i + n) {
                    n += 1;
                }
                if n < min {
                    return false;
                }
                if *greedy {
                    (min..=n).rev().any(|c| k(st, i + c))
                } else {
                    (min..=n).any(|c| k(st, i + c))
                }
            } else {
                rep(st, inner, *min, *max, *greedy, 0, i, k)
            }
        }
        Node::Look(inner, ahead, neg, w) => {
            let saved = st.caps.clone();
            let ok = if *ahead {
                m(st, inner, i, &mut |_, _| true)
            } else {
                i >= *w && m(st, inner, i - *w, &mut |_, j| j == i)
            };
            if *neg || !ok {
                st.caps = saved;
            }
            if ok != *neg {
                k(st, i)
            } else {
                false
            }
        }
        Node::Backref(g) => {
            let (a, b) = st.caps[*g];
            if a == usize::MAX {
                return false;
            }
            let len = b - a;
            if i + len > st.t.len() {
                return false;
            }
            (0..len).all(|x| st.eq(st.t[i + x], st.t[a + x])) && k(st, i + len)
        }
    }
}

fn seq<'t>(st: &mut St<'t>, v: &[Node], i: usize, k: K<'_, 't>) -> bool {
    match v.split_first() {
        None => k(st, i),
        Some((h, rest)) => m(st, h, i, &mut |st: &mut St<'t>, j| seq(st, rest, j, k)),
    }
}

#[allow(clippy::too_many_arguments)]
fn rep<'t>(
    st: &mut St<'t>,
    n: &Node,
    min: u32,
    max: u32,
    greedy: bool,
    count: u32,
    i: usize,
    k: K<'_, 't>,
) -> bool {
    let more = |st: &mut St<'t>, k: K<'_, 't>| {
        count < max
            && m(st, n, i, &mut |st: &mut St<'t>, j| {
                if j == i && count >= min {
                    return false;
                }
                rep(st, n, min, max, greedy, count + 1, j, k)
            })
    };
    if greedy {
        more(st, k) || (count >= min && k(st, i))
    } else {
        (count >= min && k(st, i)) || more(st, k)
    }
}

/// One match: group spans over the searched text.
pub struct M<'t> {
    pub t: &'t [char],
    sp: Vec<(usize, usize)>,
    names: &'t [(String, usize)],
}

impl<'t> M<'t> {
    pub fn start(&self) -> usize {
        self.sp[0].0
    }
    pub fn end(&self) -> usize {
        self.sp[0].1
    }
    pub fn has(&self, g: usize) -> bool {
        self.sp[g].0 != usize::MAX
    }
    pub fn span(&self, g: usize) -> (usize, usize) {
        self.sp[g]
    }
    pub fn start_of(&self, g: usize) -> usize {
        self.sp[g].0
    }
    pub fn end_of(&self, g: usize) -> usize {
        self.sp[g].1
    }
    pub fn get(&self, g: usize) -> Option<String> {
        self.has(g)
            .then(|| self.t[self.sp[g].0..self.sp[g].1].iter().collect())
    }
    /// Group text, or "" when the group did not participate.
    pub fn s(&self, g: usize) -> String {
        self.get(g).unwrap_or_default()
    }
    pub fn all(&self) -> String {
        self.s(0)
    }
    pub fn named(&self, name: &str) -> Option<String> {
        let g = self.names.iter().find(|(n, _)| n == name).map(|x| x.1)?;
        self.get(g)
    }
    pub fn ngroups(&self) -> usize {
        self.sp.len() - 1
    }
}

impl Regex {
    pub fn new(p: &str) -> Regex {
        Self::build(p, false)
    }
    pub fn new_i(p: &str) -> Regex {
        Self::build(p, true)
    }
    fn build(p: &str, icase: bool) -> Regex {
        let pc: Vec<char> = p.chars().collect();
        let mut ps = Parser {
            p: &pc,
            i: 0,
            ngroups: 0,
            names: Vec::new(),
        };
        let root = ps.alt();
        assert!(ps.i == pc.len(), "regex parse stopped early: {p}");
        Regex {
            root,
            ngroups: ps.ngroups,
            names: ps.names,
            icase,
        }
    }

    /// Match starting exactly at `start`; `must_advance` rejects an empty match there; `full` requires the end of text.
    fn at<'t>(
        &'t self,
        t: &'t [char],
        start: usize,
        reject_end: Option<usize>,
        full: bool,
    ) -> Option<M<'t>> {
        let mut st = St {
            t,
            caps: vec![UNSET; self.ngroups + 1],
            icase: self.icase,
        };
        let mut end = usize::MAX;
        let mut caps = Vec::new();
        let ok = m(&mut st, &self.root, start, &mut |st: &mut St<'t>, j| {
            if Some(j) == reject_end || (full && j != st.t.len()) {
                return false;
            }
            end = j;
            caps = st.caps.clone();
            true
        });
        ok.then(|| {
            caps[0] = (start, end);
            M {
                t,
                sp: caps,
                names: &self.names,
            }
        })
    }

    pub fn search_from<'t>(
        &'t self,
        t: &'t [char],
        from: usize,
        must_advance: bool,
    ) -> Option<M<'t>> {
        (from..=t.len())
            .find_map(|s| self.at(t, s, (must_advance && s == from).then_some(from), false))
    }
    pub fn search<'t>(&'t self, t: &'t [char]) -> Option<M<'t>> {
        self.search_from(t, 0, false)
    }
    /// `pattern.match(text, pos)`: anchored at `pos`, lookbehind still sees the text before it.
    pub fn match_at<'t>(&'t self, t: &'t [char], pos: usize) -> Option<M<'t>> {
        self.at(t, pos, None, false)
    }
    pub fn fullmatch<'t>(&'t self, t: &'t [char]) -> Option<M<'t>> {
        self.at(t, 0, None, true)
    }
    pub fn is_full(&self, s: &str) -> bool {
        let t: Vec<char> = s.chars().collect();
        self.fullmatch(&t).is_some()
    }
    pub fn found(&self, s: &str) -> bool {
        let t: Vec<char> = s.chars().collect();
        self.search(&t).is_some()
    }
    pub fn find_iter<'t>(&'t self, t: &'t [char]) -> Vec<M<'t>> {
        let mut out = Vec::new();
        let (mut pos, mut adv) = (0, false);
        while pos <= t.len() {
            let Some(mm) = self.search_from(t, pos, adv) else {
                break;
            };
            adv = mm.start() == mm.end();
            pos = mm.end();
            out.push(mm);
        }
        out
    }
    pub fn sub_chars(&self, t: &[char], mut f: impl FnMut(&M) -> String) -> String {
        let mut out = String::new();
        let mut last = 0;
        for mm in self.find_iter(t) {
            out.extend(&t[last..mm.start()]);
            out.push_str(&f(&mm));
            last = mm.end();
        }
        out.extend(&t[last..]);
        out
    }
    pub fn sub(&self, s: &str, f: impl FnMut(&M) -> String) -> String {
        let t: Vec<char> = s.chars().collect();
        self.sub_chars(&t, f)
    }
    pub fn replace(&self, s: &str, rep: &str) -> String {
        self.sub(s, |_| rep.to_string())
    }
    /// `re.split`: pieces, interleaved with every group (participating or not, as "").
    pub fn split(&self, s: &str) -> Vec<String> {
        let t: Vec<char> = s.chars().collect();
        let mut out = Vec::new();
        let mut last = 0;
        for mm in self.find_iter(&t) {
            out.push(t[last..mm.start()].iter().collect());
            for g in 1..=mm.ngroups() {
                out.push(mm.s(g));
            }
            last = mm.end();
        }
        out.push(t[last..].iter().collect());
        out
    }
}

/// `re.escape`.
pub fn escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        if "()[]{}?*+-|^$\\.&~# \t\n\r\x0b\x0c".contains(c) {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

/// `"|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True)) or r"(?!x)x"`.
pub fn alt_words<S: AsRef<str>>(words: impl IntoIterator<Item = S>) -> String {
    let mut v: Vec<String> = Vec::new();
    for w in words {
        let w = w.as_ref();
        if !v.iter().any(|x| x == w) {
            v.push(w.to_string());
        }
    }
    v.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
    if v.is_empty() {
        return "(?!x)x".into();
    }
    v.iter().map(|w| escape(w)).collect::<Vec<_>>().join("|")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fa(p: &str, s: &str) -> Vec<String> {
        let t: Vec<char> = s.chars().collect();
        Regex::new(p)
            .find_iter(&t)
            .iter()
            .map(|m| m.all())
            .collect()
    }
    #[test]
    fn basics() {
        assert_eq!(fa(r"\d+", "a12 b3"), ["12", "3"]);
        assert_eq!(fa(r"(?<![\d.])\d{2}(?!\d)", "12 3.45 678 90"), ["12", "90"]);
        assert_eq!(fa(r"a*", "baa"), ["", "aa", ""]);
        assert_eq!(fa(r"(\d)([/.-])\d\2\d", "1/2/3 1/2-3"), ["1/2/3"]);
        assert_eq!(
            Regex::new(r"(\[[a-z ]+\])").split("a [x] b"),
            ["a ", "[x]", " b"]
        );
        assert_eq!(fa(r"\bab|abc", "abc"), ["ab"]);
        assert!(Regex::new_i(r"pm").is_full("PM"));
        assert_eq!(fa(r"x{2,}?", "xxxx"), ["xx", "xx"]);
    }
}
