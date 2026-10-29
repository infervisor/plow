//! Maya 2 Native's host-side frame control (`maya3` engine.py `_cb0` / `_plan` / `_tick`,
//! guards.py): the cb0 controller (presence penalty, top-k with the stop class competing,
//! controlled stop, stop-excluded draw), the duration caps, the loop guard, and the per-language
//! priors that size them. Constants come from the packet's `maya2.controller`
//! (`scripts/tts/maya2_prep.py`).

use std::collections::BTreeMap;

use serde::Deserialize;

/// The stop class (`<aeos>`): the last of the cb0 head's 2049 rows.
pub const STOP: usize = 2048;

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Controller {
    pub cb0: Cb0,
    pub guard: Guard,
    pub frame_s: f32,
    pub chunk_s: f32,
    pub gap_s: f32,
    /// Per language: (a, b, resid_p99) of `frames = a + b * chars`.
    pub frames_prior: BTreeMap<String, [f32; 3]>,
    /// Per language: seconds = c . [1, non-latin letters, latin letters, pauses, digits].
    pub duration_prior: BTreeMap<String, [f32; 5]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Cb0 {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub penalty_window: u32,
    pub stop_p: f32,
    pub stop_hazard: f32,
    pub min_frames_floor: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Guard {
    pub dur_k: f32,
    pub dur_margin_s: f32,
    pub stall: u32,
    pub period_win: u32,
    pub period_thr: f32,
    pub pmin: u32,
    pub pmax: u32,
    pub start: u32,
    pub every: u32,
    pub boost: f32,
    pub backstop: u32,
    pub hist: u32,
    pub hard_cap: u32,
}

/// `s2_serve_controls.PRIOR_DEFAULT`: a language the prior lacks.
const FRAMES_DEFAULT: [f32; 3] = [2.0, 0.72, 90.0];
/// Unicode blocks -> language (`server.py` SCRIPTS), for a request without a language.
const SCRIPTS: [(u32, u32, &str); 9] = [
    (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
];
/// Loop-guard suspension windows a plan keeps (later ones are dropped).
pub const MAX_WINDOWS: usize = 16;

/// The language a text's letters mostly belong to (`server.py` detect_lang; ties: the first
/// counted).
pub fn detect_lang(text: &str) -> &'static str {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    for ch in text.chars() {
        let o = ch as u32;
        let lang = if o < 0x80 {
            if !ch.is_ascii_alphabetic() {
                continue;
            }
            "en"
        } else {
            match SCRIPTS.iter().find(|&&(lo, hi, _)| (lo..=hi).contains(&o)) {
                Some(&(_, _, l)) => l,
                None => continue,
            }
        };
        match counts.iter_mut().find(|(l, _)| *l == lang) {
            Some((_, n)) => *n += 1,
            None => counts.push((lang, 1)),
        }
    }
    let mut best: Option<(&'static str, usize)> = None;
    for &(l, n) in &counts {
        if best.is_none_or(|(_, b)| n > b) {
            best = Some((l, n));
        }
    }
    best.map_or("en", |(l, _)| l)
}

fn is_pause(ch: char) -> bool {
    matches!(ch, ',' | ';' | ':' | '.' | '!' | '?' | '।' | '॥' | '…' | '-')
}

/// Python `str.isalnum() or "_"` (the `re` `\w` class).
fn is_word(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

impl Controller {
    pub fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("maya2.controller: {e}"))
    }

    /// (expected frames, their p99 residual) of `chars` characters.
    pub fn expected_frames(&self, lang: &str, chars: usize) -> (f32, f32) {
        let [a, b, r] = self.frames_prior.get(lang).copied().unwrap_or(FRAMES_DEFAULT);
        (a + b * chars as f32, r)
    }

    fn duration_coef(&self, lang: &str) -> [f32; 5] {
        if let Some(c) = self.duration_prior.get(lang) {
            return *c;
        }
        let n = self.duration_prior.len().max(1) as f32;
        let mut m = [0.0; 5];
        for c in self.duration_prior.values() {
            for (m, c) in m.iter_mut().zip(c) {
                *m += c / n;
            }
        }
        m
    }

    /// `DurationPrior.expected_s`.
    pub fn expected_s(&self, text: &str, lang: &str) -> f32 {
        let (mut letters, mut latin, mut pauses, mut digits) = (0u32, 0u32, 0u32, 0u32);
        for ch in text.chars() {
            let o = ch as u32;
            if ch.is_alphabetic() || (0x900..=0xDFF).contains(&o) {
                letters += 1;
            }
            if ch.is_ascii_alphabetic() {
                latin += 1;
            }
            if is_pause(ch) {
                pauses += 1;
            }
            if ch.is_numeric() {
                digits += 1;
            }
        }
        let f = [
            1.0,
            (letters - latin) as f32,
            latin as f32,
            pauses as f32,
            digits as f32,
        ];
        let c = self.duration_coef(lang);
        f.iter().zip(&c).map(|(f, c)| f * c).sum::<f32>().max(0.35)
    }

    fn cap_frames(&self, text: &str, lang: &str) -> u32 {
        let g = &self.guard;
        ((g.dur_k * self.expected_s(text, lang) + g.dur_margin_s) / self.frame_s) as u32
    }

    /// The segment length that keeps a piece near `chunk_s` seconds, in frames.
    pub fn chunk_frames(&self) -> u32 {
        (self.chunk_s / self.frame_s) as u32
    }

    /// One text piece's frame budget and guard plan (`engine._plan`); `max_frames` is the
    /// request's own limit.
    pub fn plan(&self, text: &str, lang: &str, max_frames: Option<usize>, cb0: Cb0, seed: u64) -> Plan {
        let g = &self.guard;
        let (ef, er) = self.expected_frames(lang, text.chars().count());
        let loose = g.hard_cap.min((2.0 * (ef + er) + 40.0) as u32);
        let mut cap = loose.min((((ef * 0.5) as u32) + 8).max(self.cap_frames(text, lang)));
        if let Some(m) = max_frames {
            cap = cap.min((m as u32).max(1));
        }
        let mut plan = Plan {
            cb0,
            guard: *g,
            min_frames: cb0.min_frames_floor.max((0.5 * ef) as u32),
            cap,
            loose,
            windows: [(0, 0); MAX_WINDOWS],
            n_windows: 0,
            seed,
        };
        for (a, b) in self.suspend_windows(text, lang) {
            if usize::from(plan.n_windows) < MAX_WINDOWS {
                plan.windows[usize::from(plan.n_windows)] = (a, b);
                plan.n_windows += 1;
            }
        }
        plan
    }

    /// `guards.suspend_windows_v2` (v2b): frames around each `[tag]`, filler and dialogue speaker
    /// switch where the loop guard takes no action.
    fn suspend_windows(&self, text: &str, lang: &str) -> Vec<(u32, u32)> {
        let chars: Vec<char> = text.chars().collect();
        let mut ev: Vec<(usize, f32)> = Vec::new();
        // `\[([a-z _]+)\]`: style and event tags both hold 4 s in v2b.
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '[' {
                let n = chars[i + 1..]
                    .iter()
                    .take_while(|&&c| c.is_ascii_lowercase() || c == ' ' || c == '_')
                    .count();
                if n > 0 && chars.get(i + 1 + n) == Some(&']') {
                    ev.push((i, 4.0));
                    i += n + 2;
                    continue;
                }
            }
            i += 1;
        }
        for pos in fillers(&chars) {
            ev.push((pos, 2.5));
        }
        for pos in speaker_switches(&chars) {
            ev.push((pos, 2.0));
        }
        ev.sort_by_key(|&(p, _)| p);
        ev.into_iter()
            .map(|(pos, hold)| {
                let t = if pos > 0 {
                    self.expected_s(&chars[..pos].iter().collect::<String>(), lang)
                } else {
                    0.0
                };
                (
                    ((0.8 * t - 0.5) / self.frame_s).max(0.0) as u32,
                    ((1.5 * t + hold) / self.frame_s) as u32,
                )
            })
            .collect()
    }
}

/// `FILLER_RE` matches: a filler word not inside a longer word.
fn fillers(chars: &[char]) -> Vec<usize> {
    const FIXED: &[&str] = &[
        "uhh", "erm", "um", "ह्म्म", "ह्म", "हम्म", "उम्म", "उम", "ಹ್ಮ್", "ಉಮ್", "ஹ்ம்", "ம்ம்", "உம்", "উমম", "উম",
        "হুম", "ଉମ୍", "ହମ୍", "ਹਮ", "ਉਮ", "હમ્મ", "ઉમ", "ఉమ్", "హ్మ్", "ഉം", "ഹ്മ്",
    ];
    let lower: Vec<char> = chars.iter().map(|c| c.to_ascii_lowercase()).collect();
    let mut out = Vec::new();
    for i in 0..chars.len() {
        if i > 0 && is_word(chars[i - 1]) {
            continue;
        }
        // hmm+ | hm+ | umm+ | um | uh+ | uhh | erm, then the fixed words; the regex takes the first
        // alternative that ends at a non-word character.
        let run = |from: usize, c: char| lower[from..].iter().take_while(|&&x| x == c).count();
        let mut ends: Vec<usize> = Vec::new();
        if lower[i] == 'h' && run(i + 1, 'm') >= 1 {
            ends.push(i + 1 + run(i + 1, 'm'));
        }
        if lower[i] == 'u' {
            if run(i + 1, 'm') >= 1 {
                ends.push(i + 1 + run(i + 1, 'm'));
            }
            if run(i + 1, 'h') >= 1 {
                ends.push(i + 1 + run(i + 1, 'h'));
            }
        }
        for w in FIXED {
            let w: Vec<char> = w.chars().collect();
            if lower[i..].starts_with(&w) {
                ends.push(i + w.len());
            }
        }
        if ends
            .iter()
            .any(|&e| chars.get(e).is_none_or(|&c| !is_word(c)))
        {
            out.push(i);
        }
    }
    out
}

/// `SPEAKER_RE`: a `Name: ` (1-24 characters without whitespace or `:,.!?।`) after a sentence end
/// and whitespace, or after a newline; the position of the name.
fn speaker_switches(chars: &[char]) -> Vec<usize> {
    let name_char = |c: char| !c.is_whitespace() && !matches!(c, ':' | ',' | '.' | '!' | '?' | '।');
    let mut out = Vec::new();
    for i in 1..chars.len() {
        let after_ws = chars[i - 1].is_whitespace()
            && (chars[i - 1] == '\n' || {
                let mut j = i - 1;
                while j > 0 && chars[j - 1].is_whitespace() {
                    j -= 1;
                }
                j > 0 && matches!(chars[j - 1], '.' | '!' | '?' | '।' | '॥')
                    || chars[j..i].contains(&'\n')
            });
        if !after_ws {
            continue;
        }
        let n = chars[i..].iter().take_while(|&&c| name_char(c)).count();
        if (1..=24).contains(&n)
            && chars.get(i + n) == Some(&':')
            && chars.get(i + n + 1).is_some_and(|c| c.is_whitespace())
        {
            out.push(i);
        }
    }
    out
}

/// Everything the engine needs to control one piece on a slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    /// The request's cb0 sampling (defaults with its overrides).
    pub cb0: Cb0,
    pub guard: Guard,
    pub min_frames: u32,
    /// Frames after which the piece ends (`cap`; `loose` the cap without the duration guard).
    pub cap: u32,
    pub loose: u32,
    pub windows: [(u32, u32); MAX_WINDOWS],
    pub n_windows: u8,
    pub seed: u64,
}

/// One slot's controller state for its current piece.
#[derive(Clone, Debug)]
pub struct Slot {
    plan: Plan,
    /// The last `penalty_window` cb0 codes (-1 = empty), oldest first.
    recent: Vec<i32>,
    /// The loop guard's tail (the last `guard.hist` emitted codes), oldest first.
    tail: Vec<i32>,
    hazard: f32,
    /// Ticks so far (the controller's `frames`, stop tick included).
    ticks: u32,
    /// Frames emitted so far.
    frames: u32,
    aeos_bias: f32,
    boost_at: Option<u32>,
    /// The loop backstop fired: the next tick stops.
    backstop: bool,
    /// The code the depth stage last read (the slot's frame column 0).
    pub last: u32,
}

/// One tick's outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tick {
    Stop,
    Code(u32),
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// splitmix64.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform in (0, 1) per (seed, tick, class).
fn uniform(seed: u64, tick: u32, class: usize) -> f32 {
    let h = mix(mix(seed ^ (u64::from(tick) << 32)) ^ class as u64);
    ((h >> 40) as f32 + 0.5) / (1u64 << 24) as f32
}

impl Slot {
    pub fn new(plan: Plan) -> Self {
        Slot {
            plan,
            recent: vec![-1; plan.cb0.penalty_window as usize],
            tail: vec![-1; plan.guard.hist as usize],
            hazard: 0.0,
            ticks: 0,
            frames: 0,
            aeos_bias: 0.0,
            boost_at: None,
            backstop: false,
            last: 0,
        }
    }

    /// `engine._cb0` over one row of bf16 cb0 logits (2049 classes); `l` is scratch.
    pub fn tick(&mut self, logits: &[u16], l: &mut Vec<f32>) -> Tick {
        let c = self.plan.cb0;
        let n = logits.len();
        l.clear();
        l.extend(logits.iter().map(|&b| bf16_to_f32(b) / c.temperature));
        l[STOP] += self.aeos_bias;
        if c.presence_penalty != 0.0 {
            let mut seen = [false; STOP];
            for &h in &self.recent {
                if h >= 0 && (h as usize) < STOP && !seen[h as usize] {
                    seen[h as usize] = true;
                    l[h as usize] -= c.presence_penalty;
                }
            }
        }
        let k = (c.top_k as usize).clamp(1, n);
        if k < n {
            if n <= STOP + 1 {
                let mut s = [0.0f32; STOP + 1];
                s[..n].copy_from_slice(l);
                s[..n].select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
                let kth = s[k - 1];
                for v in l.iter_mut() {
                    if *v < kth {
                        *v = f32::NEG_INFINITY;
                    }
                }
            } else {
                let mut s = l.clone();
                s.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
                let kth = s[k - 1];
                for v in l.iter_mut() {
                    if *v < kth {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
        }
        if c.top_p < 1.0 {
            let mut p_stack = [0.0f32; STOP + 1];
            let mut p_heap = Vec::new();
            let p: &mut [f32] = if n <= p_stack.len() {
                &mut p_stack[..n]
            } else {
                p_heap.resize(n, 0.0);
                &mut p_heap[..]
            };
            softmax_into(l, p);
            let mut order: Vec<usize> = (0..n).filter(|&i| p[i] > 0.0).collect();
            order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
            let mut cum = 0.0f32;
            for &i in &order {
                if cum > c.top_p {
                    l[i] = f32::NEG_INFINITY;
                }
                cum += p[i];
            }
        }
        let mut p_stack = [0.0f32; STOP + 1];
        let mut p_heap = Vec::new();
        let p: &mut [f32] = if n <= p_stack.len() {
            &mut p_stack[..n]
        } else {
            p_heap.resize(n, 0.0);
            &mut p_heap[..]
        };
        softmax_into(l, p);
        let pa = p[STOP];
        let am = (0..n).fold(0, |b, i| if p[i] > p[b] { i } else { b });
        let can = self.ticks >= self.plan.min_frames;
        if can {
            self.hazard = 1.0 - (1.0 - self.hazard) * (1.0 - pa);
        }
        let stop = self.backstop
            || (can && (am == STOP || pa >= c.stop_p || self.hazard >= c.stop_hazard));
        p[STOP] = 0.0;
        // Exponential race over the stop-excluded distribution (multinomial; no renormalisation needed).
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, &q) in p.iter().enumerate().take(STOP) {
            if q > 0.0 {
                let e = -uniform(self.plan.seed, self.ticks, i).ln();
                let s = q.ln() - e.ln();
                if s > best.1 {
                    best = (i, s);
                }
            }
        }
        let code = best.0.min(STOP - 1) as u32;
        self.ticks += 1;
        if !self.recent.is_empty() {
            self.recent.rotate_left(1);
            *self.recent.last_mut().unwrap() = code as i32;
        }
        if stop {
            Tick::Stop
        } else {
            self.last = code;
            Tick::Code(code)
        }
    }

    /// After an emitted frame of `code`: the loop guard (`engine._tick`): from `start`, every
    /// `every` frames outside the suspension windows, a stall or period boosts the stop class and,
    /// if it persists `backstop` frames, ends the piece.
    pub fn emitted(&mut self, code: u32) {
        let g = self.plan.guard;
        self.frames += 1;
        if !self.tail.is_empty() {
            self.tail.rotate_left(1);
            *self.tail.last_mut().unwrap() = code as i32;
        }
        let f = self.frames;
        let suspended = self.plan.windows[..usize::from(self.plan.n_windows)]
            .iter()
            .any(|&(a, b)| (a..=b).contains(&f));
        if f >= self.plan.cap || f < g.start || g.every == 0 || f % g.every != 0 || suspended {
            return;
        }
        if self.loop_hit() {
            if self.boost_at.is_none() {
                self.boost_at = Some(f);
                self.aeos_bias = g.boost;
            }
            if self.boost_at.is_some_and(|b| f - b >= g.backstop) {
                self.backstop = true;
            }
        } else if self.boost_at.is_some() {
            self.aeos_bias = 0.0;
            self.boost_at = None;
        }
    }

    /// `guards.loop_detect_batch` for this slot.
    fn loop_hit(&self) -> bool {
        let g = self.plan.guard;
        let (st, pw, n) = (g.stall as usize, g.period_win as usize, self.frames as usize);
        let w = &self.tail;
        if n < 12.max(st) || st == 0 || st > w.len() || pw > w.len() {
            return false;
        }
        let t = &w[w.len() - st..];
        if t.iter().all(|&x| x == t[0]) {
            return true;
        }
        if n < 12.max(st).max(pw) {
            return false;
        }
        let win = &w[w.len() - pw..];
        (g.pmin as usize..=g.pmax as usize).any(|p| {
            p < pw && {
                let same = (0..pw - p).filter(|&i| win[i + p] == win[i]).count();
                same as f32 / (pw - p) as f32 >= g.period_thr
            }
        })
    }
}

/// A cloned speaker's conditioning (`spk` / `icl_spk` modes) from its ECAPA embedding: the anchor
/// added at each `<|audio|>` sentinel, and the depth stage's per-layer memory constants. Tables from
/// `checkpoint/speaker/model.safetensors` (`scripts/tts/maya2_prep.py`).
pub struct Speaker {
    fc1: (Vec<f32>, Vec<f32>),
    fc2: (Vec<f32>, Vec<f32>),
    audio_row: Vec<f32>,
    memory: Vec<Vec<f32>>,
}

/// ECAPA embedding width.
pub const ECAPA_DIM: usize = 192;

fn round_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// f32 -> IEEE binary16 -> f32, round to nearest even (normal range; the memory constants are O(1)).
fn round_f16(x: f32) -> f32 {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127;
    if !(-14..=15).contains(&e) {
        return x;
    }
    let drop = 13;
    let half = 1u32 << (drop - 1);
    let r = b + half - 1 + ((b >> drop) & 1);
    f32::from_bits(r & !((1u32 << drop) - 1))
}

fn matvec(w: &[f32], x: &[f32], b: Option<&[f32]>) -> Vec<f32> {
    let k = x.len();
    w.chunks_exact(k)
        .enumerate()
        .map(|(n, row)| row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + b.map_or(0.0, |b| b[n]))
        .collect()
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm_erf(x / std::f32::consts::SQRT_2))
}

/// erf through Numerical Recipes' `erfcc` Chebyshev fit in f64 (|error| < 1.2e-7).
fn libm_erf(x: f32) -> f32 {
    let x = f64::from(x);
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

impl Speaker {
    #[cfg(feature = "cuda")]
    pub fn load(dir: &std::path::Path) -> Result<Self, String> {
        let path = dir.join("model.safetensors");
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let get = |name: &str| -> Result<Vec<f32>, String> {
            let t = st.tensor(name).map_err(|e| format!("{}: {name}: {e}", path.display()))?;
            if t.dtype() != safetensors::Dtype::F32 {
                return Err(format!("{}: {name} is not f32", path.display()));
            }
            Ok(t.data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let mut memory = Vec::new();
        while let Ok(m) = get(&format!("memory.{}", memory.len())) {
            memory.push(m);
        }
        let s = Speaker {
            fc1: (get("anchor.fc1.weight")?, get("anchor.fc1.bias")?),
            fc2: (get("anchor.fc2.weight")?, get("anchor.fc2.bias")?),
            audio_row: get("audio_row")?,
            memory,
        };
        let h = s.audio_row.len();
        if s.memory.is_empty()
            || s.fc1.0.len() != s.fc1.1.len() * ECAPA_DIM
            || s.fc2.0.len() != s.fc1.1.len() * h
            || s.fc2.1.len() != h
            || s.memory.iter().any(|m| m.len() % ECAPA_DIM != 0)
        {
            return Err(format!("{}: speaker table shapes disagree", path.display()));
        }
        Ok(s)
    }

    /// The `<|audio|>` sentinel's input row for this speaker: `bf16(embed + MLP(e))`.
    pub fn anchor_row(&self, e: &[f32; ECAPA_DIM]) -> Vec<f32> {
        let h1: Vec<f32> = matvec(&self.fc1.0, e, Some(&self.fc1.1)).into_iter().map(gelu).collect();
        let a = matvec(&self.fc2.0, &h1, Some(&self.fc2.1));
        a.iter()
            .zip(&self.audio_row)
            .map(|(a, r)| round_bf16(r + a))
            .collect()
    }

    /// The depth memory constant of every layer, `[layers][hidden]` (f16-rounded, as the engine's).
    pub fn memory(&self, e: &[f32; ECAPA_DIM]) -> Vec<f32> {
        self.memory
            .iter()
            .flat_map(|m| matvec(m, e, None).into_iter().map(round_f16))
            .collect()
    }
}

fn softmax_into(l: &[f32], out: &mut [f32]) {
    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (dst, &src) in out.iter_mut().zip(l) {
        let v = (src - m).exp();
        *dst = v;
        sum += v;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for dst in out.iter_mut() {
        *dst *= inv;
    }
}

fn softmax(l: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; l.len()];
    softmax_into(l, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> Controller {
        Controller::parse(
            r#"{"cb0":{"temperature":0.9,"top_k":64,"top_p":1.0,"presence_penalty":0.8,
            "penalty_window":12,"stop_p":0.35,"stop_hazard":0.6,"min_frames_floor":4},
            "guard":{"dur_k":2.2,"dur_margin_s":1.2,"stall":32,"period_win":64,"period_thr":0.7,
            "pmin":2,"pmax":16,"start":32,"every":2,"boost":4.0,"backstop":24,"hist":96,"hard_cap":1500},
            "frame_s":0.08,"chunk_s":20.0,"gap_s":0.15,
            "frames_prior":{"en":[0.6,0.7416,75.0]},
            "duration_prior":{"en":[2.87006,0.01,0.04601,0.40336,0.06541]}}"#,
        )
        .unwrap()
    }

    #[test]
    fn rounding_and_gelu_match_torch() {
        assert_eq!(round_f16(1.0 + 1.0 / 4096.0), 1.0);
        assert_eq!(round_f16(1.0 + 3.0 / 4096.0), 1.0 + 1.0 / 1024.0);
        assert_eq!(round_bf16(1.0 + 1.0 / 512.0), 1.0);
        // torch.nn.functional.gelu(torch.tensor([1.0, -0.5, 3.0]))
        for (x, y) in [(1.0f32, 0.841_344_7f32), (-0.5, -0.154_268_57), (3.0, 2.995_950_4)] {
            assert!((gelu(x) - y).abs() < 1e-6, "{x}: {} vs {y}", gelu(x));
        }
    }

    #[test]
    fn languages_follow_the_script() {
        assert_eq!(detect_lang("Hello there"), "en");
        assert_eq!(detect_lang("नमस्ते दुनिया"), "hi");
        assert_eq!(detect_lang("வணக்கம்"), "ta");
        assert_eq!(detect_lang("123"), "en");
    }

    #[test]
    fn the_plan_matches_the_engine() {
        let c = controller();
        let text = "Hello there, how are you today?";
        let p = c.plan(text, "en", None, c.cb0, 1);
        // ef = 0.6 + 0.7416 * 31 = 23.59; loose = int(2 * (23.59 + 75) + 40) = 237;
        // expected_s = 2.87006 + 0.04601 * 24 + 0.40336 * 2 = 4.78; cap_frames = int((2.2 * 4.78 + 1.2) / 0.08) = 146.
        assert_eq!(p.loose, 237);
        assert_eq!(p.cap, 146);
        assert_eq!(p.min_frames, 11);
        assert_eq!(c.plan(text, "en", Some(20), c.cb0, 1).cap, 20);
    }

    #[test]
    fn fillers_and_speaker_switches_open_windows() {
        let chars: Vec<char> = "So, umm, yes. Ravi: hello hmmm".chars().collect();
        assert_eq!(fillers(&chars), vec![4, 26]);
        assert_eq!(speaker_switches(&chars), vec![14]);
        assert!(fillers(&"summer drum".chars().collect::<Vec<_>>()).is_empty());
    }

    #[test]
    fn a_confident_stop_ends_after_min_frames() {
        let c = controller();
        let mut s = Slot::new(c.plan("Hi.", "en", None, c.cb0, 3));
        let one = |hot: usize| {
            let mut v = vec![0x3f80u16; STOP + 1]; // 1.0
            v[hot] = 0x4220; // 40.0
            v
        };
        let mut l = Vec::new();
        assert_eq!(s.tick(&one(STOP), &mut l), Tick::Code(s.last));
        for _ in 1..s.plan.min_frames {
            assert!(matches!(s.tick(&one(7), &mut l), Tick::Code(7)));
        }
        assert_eq!(s.tick(&one(STOP), &mut l), Tick::Stop);
    }

    #[test]
    fn a_stall_boosts_then_backstops() {
        let c = controller();
        let mut s = Slot::new(c.plan(&"word ".repeat(200), "en", None, c.cb0, 3));
        for _ in 0..32 {
            s.emitted(5);
        }
        assert_eq!(s.aeos_bias, 4.0);
        for _ in 0..24 {
            s.emitted(5);
        }
        assert!(s.backstop);
    }
}
