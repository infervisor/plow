//! Voice turns: one user utterance → transcript → reply text → reply speech inside one call
//! (session), linked across the models of one process.
//!
//! A turn is named by the client (`X-Turn-Id`, or body `metadata.turn_id`) or inferred per
//! session: an ASR final opens turn n+1 (its arrival is the end of speech), the session's next
//! chat request joins the open turn, and the TTS request after it joins and closes it. A chat or
//! TTS request with no open turn opens one at its own arrival. Sessions stay per model for KV
//! retention; this table is process-wide.
//!
//! One lock per stage transition (join, admission, first output, done), never per tick.

use std::collections::VecDeque;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::http::{HeaderName, HeaderValue};
use parking_lot::Mutex;

use crate::obs::serving::Histogram;
use crate::obs::Metrics;
use crate::serve::session::RequestIds;

pub const TURN_HEADER: HeaderName = HeaderName::from_static("x-turn-id");
pub const TRACEPARENT: HeaderName = HeaderName::from_static("traceparent");
pub const SERVER_TIMING: HeaderName = HeaderName::from_static("server-timing");

/// Turns kept per session (oldest dropped first).
const TURNS_PER_SESSION: usize = 32;
/// Sessions tracked; past it the cache evicts one.
const MAX_SESSIONS: usize = 16_384;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TurnKey {
    pub session: Arc<str>,
    pub turn: Arc<str>,
}

#[derive(Clone, Copy, Debug)]
pub enum Stage {
    AsrFinal,
    LlmFirst,
    LlmDecode,
    TtsFirst,
    TtsStream,
}

/// What deadline computation needs from a turn.
#[derive(Clone, Copy, Debug, Default)]
pub struct TurnTimes {
    pub speech_end: Option<Instant>,
    pub budget: Duration,
    pub asr_done: Option<Instant>,
    pub llm_first: Option<Instant>,
    pub tts_first_audio: Option<Instant>,
}

/// Which endpoint a request is: the stage it opens or joins a turn as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Asr,
    Llm,
    Tts,
}

impl Kind {
    const ALL: [Kind; 3] = [Kind::Asr, Kind::Llm, Kind::Tts];

    pub fn index(self) -> usize {
        self as usize
    }

    /// The stage metric's label.
    pub fn stage(self) -> &'static str {
        match self {
            Kind::Asr => "asr_final",
            Kind::Llm => "llm_ttft",
            Kind::Tts => "tts_ttfa",
        }
    }

    /// The default target from stage start to its first output.
    pub fn target(self) -> Duration {
        let c = crate::config::RuntimeConfig::get();
        Duration::from_millis(match self {
            Kind::Asr => c.turn_asr_final_ms,
            Kind::Llm => c.turn_llm_ttft_ms,
            Kind::Tts => c.turn_tts_ttfa_ms,
        })
    }
}

/// Per-stage server timing of one request, in the turn record.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTiming {
    pub arrived: Option<Instant>,
    pub admitted: Option<Instant>,
    pub first: Option<Instant>,
    pub done: Option<Instant>,
    /// The model's own tick time (`device`) and the rest of admission → first output
    /// (`wait-turn`: co-tenant turns, host gaps), ms.
    pub device_ms: f64,
    pub wait_ms: f64,
}

/// Client playback state (`X-Playback`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Playback {
    /// Playback started at this unix time, ms.
    Started(u64),
    /// Audio the client holds unplayed, ms.
    Buffered(u64),
}

impl Playback {
    pub fn parse(value: &str) -> Option<Self> {
        let (k, v) = value.trim().split_once('=')?;
        let v: u64 = v.trim().parse().ok()?;
        match k.trim() {
            "started" => Some(Playback::Started(v)),
            "buffered_ms" => Some(Playback::Buffered(v)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Turn {
    pub id: Arc<str>,
    pub trace_id: u128,
    /// The trace id came from a client (`traceparent` / body `trace`), not generated.
    trace_from_client: bool,
    pub speech_end: Instant,
    pub budget: Duration,
    pub stages: [StageTiming; 3],
    /// Audio sent so far by the turn's TTS stream, seconds (the server's playback clock starts at
    /// the first audio).
    pub audio_s: f64,
    /// Worst playback underrun on the server's clock, seconds.
    pub underrun_s: f64,
    pub playback: Option<Playback>,
}

impl Turn {
    fn joined(&self, kind: Kind) -> bool {
        self.stages[kind.index()].arrived.is_some()
    }

    fn times(&self) -> TurnTimes {
        TurnTimes {
            speech_end: Some(self.speech_end),
            budget: self.budget,
            asr_done: self.stages[0].done,
            llm_first: self.stages[1].first,
            tts_first_audio: self.stages[2].first,
        }
    }
}

#[derive(Default)]
struct SessionTurns {
    /// Id of the inferred turn still waiting for its chat or TTS.
    open: Option<Arc<str>>,
    turns: VecDeque<Turn>,
}

impl SessionTurns {
    fn get_mut(&mut self, id: &str) -> Option<&mut Turn> {
        self.turns.iter_mut().rev().find(|t| &*t.id == id)
    }

    fn push(&mut self, turn: Turn) -> &mut Turn {
        if self.turns.len() == TURNS_PER_SESSION {
            self.turns.pop_front();
        }
        self.turns.push_back(turn);
        self.turns.back_mut().expect("just pushed")
    }
}

/// One session's turns; a clone shares them (svcache clones values out on every get).
#[derive(Clone)]
struct SessionEntry {
    session: Arc<str>,
    turns: Arc<Mutex<SessionTurns>>,
}

impl svcache::CacheKey for SessionEntry {
    type Id = Arc<str>;
    fn id(&self) -> Arc<str> {
        self.session.clone()
    }
}

/// Sessions by id, each with its last [`TURNS_PER_SESSION`] turns. The cache bounds the sessions
/// (TTL, count); its TTL runs from insert, so every stage transition re-inserts the session.
pub struct TurnTable {
    sessions: svcache::SvCache<SessionEntry>,
}

pub fn table() -> &'static TurnTable {
    static T: OnceLock<TurnTable> = OnceLock::new();
    T.get_or_init(TurnTable::new)
}

/// How long a turn outlives its last use: the session TTL (60 s when retention is off).
fn ttl() -> Duration {
    let t = crate::serve::session::retention_ttl();
    if t.is_zero() {
        Duration::from_secs(60)
    } else {
        t
    }
}

/// A request's place in a turn.
#[derive(Clone, Debug)]
pub struct Joined {
    pub key: TurnKey,
    pub trace_id: u128,
    pub speech_end: Instant,
    pub budget: Duration,
    /// The session had a turn before this request joined: it continues a session, not opens one.
    pub continuing: bool,
}

impl TurnTable {
    fn new() -> Self {
        Self { sessions: svcache::SvCache::with_ttl_and_limit(ttl(), MAX_SESSIONS) }
    }

    fn entry(&self, session: &Arc<str>) -> Option<SessionEntry> {
        self.sessions.get_by_id(session.clone())
    }

    pub fn times(&self, key: &TurnKey) -> Option<TurnTimes> {
        self.entry(&key.session)?.turns.lock().get_mut(&key.turn).map(|t| t.times())
    }

    pub fn stamp(&self, key: &TurnKey, stage: Stage, at: Instant) {
        self.update(key, |t| {
            let s = &mut t.stages;
            match stage {
                Stage::AsrFinal => s[0].done = Some(at),
                Stage::LlmFirst => s[1].first = Some(at),
                Stage::LlmDecode => s[1].done = Some(at),
                Stage::TtsFirst => s[2].first = Some(at),
                Stage::TtsStream => s[2].done = Some(at),
            }
        });
    }

    /// Apply `f` to the turn, if it is still tracked, and extend the session's life.
    pub fn update(&self, key: &TurnKey, f: impl FnOnce(&mut Turn)) {
        let Some(e) = self.entry(&key.session) else { return };
        if let Some(t) = e.turns.lock().get_mut(&key.turn) {
            f(t);
        }
        self.sessions.insert(e);
    }

    /// Whether `session` has a turn within the TTL.
    pub fn knows_session(&self, session: &str) -> bool {
        self.sessions.get_by_id(Arc::from(session)).is_some()
    }

    /// Track an admitted `session` before its first turn (a call's ASR appends precede it), so
    /// overload never sheds the call's later requests.
    pub fn note_session(&self, session: &Arc<str>) {
        let e = self.entry(session).unwrap_or_else(|| SessionEntry { session: session.clone(), turns: Default::default() });
        self.sessions.insert(e);
    }

    /// The session's turns, oldest first.
    pub fn session(&self, session: &str) -> Vec<Turn> {
        self.sessions.get_by_id(Arc::from(session)).map(|e| e.turns.lock().turns.iter().cloned().collect()).unwrap_or_default()
    }

    /// Join (or open) the turn a `kind` request of `session` serves, arriving `at`. `explicit` is
    /// the client's turn id; `trace` the client's trace id; `budget` its turn budget.
    #[allow(clippy::too_many_arguments)]
    pub fn join(
        &self,
        session: &Arc<str>,
        explicit: Option<&Arc<str>>,
        kind: Kind,
        at: Instant,
        trace: Option<u128>,
        budget: Option<Duration>,
        playback: Option<Playback>,
    ) -> Joined {
        let e = self.entry(session).unwrap_or_else(|| SessionEntry { session: session.clone(), turns: Default::default() });
        let joined = {
            let mut st = e.turns.lock();
            let continuing = !st.turns.is_empty();
            let new_turn = |id: Arc<str>, svid: i64| Turn {
                id,
                trace_id: trace.unwrap_or_else(|| minted::trace_id(svid)),
                trace_from_client: trace.is_some(),
                speech_end: at,
                budget: budget.unwrap_or_else(default_budget),
                stages: Default::default(),
                audio_s: 0.0,
                underrun_s: 0.0,
                playback: None,
            };
            let id: Arc<str> = match explicit {
                Some(id) => {
                    if st.get_mut(id).is_none() {
                        st.push(new_turn(id.clone(), minted::turn().0));
                    }
                    id.clone()
                }
                None => {
                    let open = st.open.clone().filter(|id| kind != Kind::Asr && st.get_mut(id).is_some_and(|t| !t.joined(kind)));
                    match open {
                        Some(id) => id,
                        None => {
                            let (svid, id) = minted::turn();
                            let id: Arc<str> = id.into();
                            st.push(new_turn(id.clone(), svid));
                            id
                        }
                    }
                }
            };
            if explicit.is_none() {
                // TTS is a turn's last stage: it closes the open turn.
                st.open = (kind != Kind::Tts).then(|| id.clone());
            }
            let t = st.get_mut(&id).expect("present");
            if kind == Kind::Asr && !t.joined(Kind::Llm) {
                t.speech_end = at;
            }
            if let Some(trace) = trace {
                if !t.trace_from_client {
                    t.trace_id = trace;
                    t.trace_from_client = true;
                }
            }
            if let Some(b) = budget {
                t.budget = b;
            }
            if playback.is_some() {
                t.playback = playback;
            }
            t.stages[kind.index()] = StageTiming { arrived: Some(at), ..Default::default() };
            Joined { key: TurnKey { session: session.clone(), turn: id }, trace_id: t.trace_id, speech_end: t.speech_end, budget: t.budget, continuing }
        };
        self.sessions.insert(e);
        joined
    }
}

fn default_budget() -> Duration {
    Duration::from_millis(crate::config::RuntimeConfig::get().turn_budget_ms)
}

use crate::serve::session::minted;

// ---------------------------------------------------------------------------------------------
// Process-wide turn metrics.

/// Signed latency bounds for slack, µs.
const SLACK: &[i64] = &[
    -30_000_000, -10_000_000, -5_000_000, -2_000_000, -1_000_000, -500_000, -200_000, -100_000, 0, 100_000, 200_000,
    500_000, 1_000_000,
];

#[derive(Default)]
struct SignedHistogram {
    bins: [AtomicU64; 14],
    sum: std::sync::atomic::AtomicI64,
}

impl SignedHistogram {
    fn observe(&self, us: i64) {
        self.bins[SLACK.partition_point(|&b| b < us)].fetch_add(1, Relaxed);
        self.sum.fetch_add(us, Relaxed);
    }

    fn write(&self, out: &mut String, name: &str, labels: &str) {
        let mut count = 0;
        for (i, bin) in self.bins.iter().enumerate() {
            count += bin.load(Relaxed);
            match SLACK.get(i) {
                Some(&b) => drop(writeln!(out, "{name}_bucket{{{labels},le=\"{}\"}} {count}", b as f64 / 1e6)),
                None => drop(writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {count}")),
            }
        }
        let _ = writeln!(out, "{name}_sum{{{labels}}} {}", self.sum.load(Relaxed) as f64 / 1e6);
        let _ = writeln!(out, "{name}_count{{{labels}}} {count}");
    }
}

#[derive(Default)]
struct TurnMetrics {
    response: Histogram,
    underrun: Histogram,
    slack: [SignedHistogram; 3],
    missed: [AtomicU64; 3],
}

fn metrics() -> &'static TurnMetrics {
    static M: OnceLock<TurnMetrics> = OnceLock::new();
    M.get_or_init(TurnMetrics::default)
}

/// `plowrt_turn_response_seconds`, `plowrt_tts_underrun_seconds`, `plowrt_deadline_slack_seconds`,
/// `plowrt_deadline_missed_total`.
pub fn write_metrics(out: &mut String) {
    use crate::obs::serving::family;
    let m = metrics();
    family(out, "plowrt_turn_response_seconds", "histogram", "Turn end of speech (ASR final arrival, or the turn's first request) through first TTS audio.");
    m.response.write_seconds(out, "plowrt_turn_response_seconds", "scope=\"process\"");
    family(out, "plowrt_tts_underrun_seconds", "histogram", "Worst playback underrun per TTS stream on the server's clock (playback starts at the first audio).");
    m.underrun.write_seconds(out, "plowrt_tts_underrun_seconds", "scope=\"process\"");
    family(out, "plowrt_deadline_slack_seconds", "histogram", "Stage target minus time to first output (negative = missed); targets PLOW_TURN_*_MS.");
    for k in Kind::ALL {
        m.slack[k.index()].write(out, "plowrt_deadline_slack_seconds", &format!("stage=\"{}\"", k.stage()));
    }
    family(out, "plowrt_deadline_missed_total", "counter", "Requests whose first output missed the stage target.");
    for k in Kind::ALL {
        let _ = writeln!(out, "plowrt_deadline_missed_total{{stage=\"{}\"}} {}", k.stage(), m.missed[k.index()].load(Relaxed));
    }
}

// ---------------------------------------------------------------------------------------------
// One request's stage run: joins the turn, measures queue / wait-turn / device / first output,
// stamps the turn table and the response.

/// `Server-Timing` metrics of one request, ms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ServerTiming {
    pub queue: Option<f64>,
    pub wait_turn: Option<f64>,
    pub device: Option<f64>,
    pub first: Option<f64>,
    pub total: Option<f64>,
    pub slack: Option<f64>,
    /// End of speech → this request's first output.
    pub turn: Option<f64>,
}

impl ServerTiming {
    /// `queue;dur=1.2, wait-turn;dur=0.3, ...` (W3C Server-Timing; absent metrics left out).
    pub fn header(&self) -> String {
        let mut s = String::new();
        for (name, v) in [
            ("queue", self.queue),
            ("wait-turn", self.wait_turn),
            ("device", self.device),
            ("first", self.first),
            ("total", self.total),
            ("slack", self.slack),
            ("turn", self.turn),
        ] {
            if let Some(v) = v {
                if !s.is_empty() {
                    s.push_str(", ");
                }
                let _ = write!(s, "{name};dur={v:.1}");
            }
        }
        s
    }
}

pub struct StageRun {
    pub kind: Kind,
    pub joined: Option<Joined>,
    pub trace_id: u128,
    span_id: u64,
    flags: u8,
    pub turn_id: Option<Arc<str>>,
    arrived: Instant,
    metrics: Option<Arc<Metrics>>,
    device_at_arrival: u64,
    admitted: Option<(Instant, u64)>,
    first: Option<(Instant, u64)>,
    done: Option<Instant>,
    target: Duration,
    /// Server-side playback clock (TTS): audio sent, worst underrun.
    audio_s: f64,
    underrun_s: f64,
    /// A stage of a turn (not an ASR partial): feeds the stage metrics.
    observed: bool,
    span: tracing::Span,
}

impl StageRun {
    /// Start a `kind` request arriving `arrived` for `model`; joins its turn when it has a session.
    /// `turn_stage` = false for requests that are no stage of a turn (ASR partials).
    pub fn start(ids: &RequestIds, kind: Kind, model: &str, metrics: Option<Arc<Metrics>>, arrived: Instant, turn_stage: bool) -> Self {
        let client_trace = ids.trace.map(|t| t.trace_id);
        let joined = ids.session.as_ref().filter(|_| turn_stage).map(|s| {
            table().join(s, ids.turn.as_ref(), kind, arrived, client_trace, ids.budget(), ids.playback)
        });
        let trace_id = client_trace.or(joined.as_ref().map(|j| j.trace_id)).unwrap_or_else(|| minted::trace_id(minted::turn().0));
        let turn_id = joined.as_ref().map(|j| j.key.turn.clone()).or_else(|| ids.turn.clone());
        let span = tracing::debug_span!(
            "request",
            request_id = %ids.request,
            session = ids.session.as_deref().unwrap_or(""),
            turn = turn_id.as_deref().unwrap_or(""),
            trace_id = %format_args!("{trace_id:032x}"),
            model,
            stage = kind.stage(),
        );
        let device_at_arrival = metrics.as_ref().map_or(0, |m| m.serving.device_us());
        Self {
            kind,
            joined,
            trace_id,
            span_id: minted::span(),
            flags: ids.trace.map_or(0, |t| t.flags),
            turn_id,
            arrived,
            metrics,
            device_at_arrival,
            admitted: None,
            first: None,
            done: None,
            target: kind.target(),
            audio_s: 0.0,
            underrun_s: 0.0,
            observed: turn_stage,
            span,
        }
    }

    pub fn key(&self) -> Option<TurnKey> {
        self.joined.as_ref().map(|j| j.key.clone())
    }

    pub fn continuing(&self) -> bool {
        self.joined.as_ref().is_some_and(|j| j.continuing)
    }

    fn device_now(&self) -> u64 {
        self.metrics.as_ref().map_or(0, |m| m.serving.device_us())
    }

    /// Slot admission at `at` (the session report's stamp).
    pub fn admitted(&mut self, at: Option<Instant>) {
        let Some(at) = at else { return };
        if self.admitted.is_none() {
            self.admitted = Some((at, self.device_now()));
            tracing::debug!(parent: &self.span, queue_ms = ms(at.saturating_duration_since(self.arrived)), "admitted");
        }
    }

    /// The request's first output (first token, transcript, first audio).
    pub fn first(&mut self) {
        if self.first.is_some() {
            return;
        }
        let now = Instant::now();
        self.first = Some((now, self.device_now()));
        let took = now.saturating_duration_since(self.arrived);
        let slack_us = self.target.as_micros() as i64 - took.as_micros() as i64;
        let m = metrics();
        if self.observed {
            m.slack[self.kind.index()].observe(slack_us);
            if slack_us < 0 {
                m.missed[self.kind.index()].fetch_add(1, Relaxed);
            }
            if let Some(mm) = &self.metrics {
                mm.serving.turn_stage[self.kind.index()].duration(took);
            }
        }
        let t = self.timing();
        if let Some(j) = &self.joined {
            let stage = match self.kind {
                Kind::Asr => Stage::AsrFinal,
                Kind::Llm => Stage::LlmFirst,
                Kind::Tts => Stage::TtsFirst,
            };
            if self.kind == Kind::Tts {
                m.response.duration(now.saturating_duration_since(j.speech_end));
            }
            let (kind, admitted) = (self.kind, self.admitted.map(|a| a.0));
            table().update(&j.key, |turn| {
                let s = &mut turn.stages[kind.index()];
                s.admitted = admitted;
                s.device_ms = t.device.unwrap_or(0.0);
                s.wait_ms = t.wait_turn.unwrap_or(0.0);
            });
            table().stamp(&j.key, stage, now);
            // Overload judges each stage against its own target from arrival: a client that
            // sends TTS only after the whole reply can't meet an end-to-end budget at any load.
            crate::serve::overload::observe_deadline(stage, slack_us.saturating_mul(1000));
        }
        tracing::debug!(parent: &self.span, first_ms = ms(took), slack_ms = slack_us as f64 / 1e3, "first output");
    }

    /// TTS: `samples` at `rate` just went out; advances the server's playback clock.
    pub fn audio(&mut self, samples: usize, rate: f64) {
        if let Some((first, _)) = self.first {
            let behind = Instant::now().saturating_duration_since(first).as_secs_f64() - self.audio_s;
            self.underrun_s = self.underrun_s.max(behind);
        } else {
            self.first();
        }
        self.audio_s += samples as f64 / rate;
    }

    /// The request finished.
    pub fn done(&mut self) {
        if self.done.is_some() {
            return;
        }
        let now = Instant::now();
        self.done = Some(now);
        if self.kind == Kind::Tts && self.first.is_some() && self.observed {
            metrics().underrun.duration(Duration::from_secs_f64(self.underrun_s.max(0.0)));
        }
        if let Some(j) = &self.joined {
            let (kind, audio_s, underrun_s) = (self.kind, self.audio_s, self.underrun_s);
            table().update(&j.key, |t| {
                t.stages[kind.index()].done = Some(now);
                if kind == Kind::Tts {
                    t.audio_s = audio_s;
                    t.underrun_s = underrun_s;
                }
            });
        }
        tracing::debug!(parent: &self.span, total_ms = ms(now.saturating_duration_since(self.arrived)), "done");
    }

    pub fn timing(&self) -> ServerTiming {
        let mut t = ServerTiming::default();
        if let Some((at, _)) = self.admitted {
            t.queue = Some(ms(at.saturating_duration_since(self.arrived)));
        }
        if let Some((first, dev_first)) = self.first {
            let (from, dev_from) = self.admitted.unwrap_or((self.arrived, self.device_at_arrival));
            let device = dev_first.saturating_sub(dev_from) as f64 / 1e3;
            let span = ms(first.saturating_duration_since(from));
            if self.metrics.is_some() {
                t.device = Some(device.min(span));
                t.wait_turn = Some((span - device).max(0.0));
            }
            let took = first.saturating_duration_since(self.arrived);
            t.first = Some(ms(took));
            t.slack = self.observed.then(|| ms(self.target) - ms(took));
            if let Some(j) = &self.joined {
                t.turn = Some(ms(first.saturating_duration_since(j.speech_end)));
            }
        }
        if let Some(done) = self.done {
            t.total = Some(ms(done.saturating_duration_since(self.arrived)));
        }
        t
    }

    /// `00-<trace>-<this request's span>-<flags>`.
    pub fn traceparent(&self) -> String {
        format!("00-{:032x}-{:016x}-{:02x}", self.trace_id, self.span_id, self.flags)
    }

    /// `X-Turn-Id`, `traceparent`, and `Server-Timing` (when anything is measured yet).
    pub fn stamp(&self, response: &mut axum::response::Response) {
        response.headers_mut().extend(self.headers());
    }

    pub fn headers(&self) -> axum::http::HeaderMap {
        let mut h = axum::http::HeaderMap::new();
        if let Some(v) = self.turn_id.as_deref().and_then(|t| HeaderValue::from_str(t).ok()) {
            h.insert(TURN_HEADER, v);
        }
        if let Ok(v) = HeaderValue::from_str(&self.traceparent()) {
            h.insert(TRACEPARENT, v);
        }
        let timing = self.timing().header();
        if !timing.is_empty() {
            if let Ok(v) = HeaderValue::from_str(&timing) {
                h.insert(SERVER_TIMING, v);
            }
        }
        h
    }

    /// The final SSE comment of a streamed response: `: server-timing <metrics>`.
    pub fn sse_comment(&self) -> axum::response::sse::Event {
        axum::response::sse::Event::default().comment(format!("server-timing {}", self.timing().header()))
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// `GET /v1/turns/{session}`: the call's turns, times in ms from each turn's end of speech.
pub async fn session_turns(axum::extract::Path(session): axum::extract::Path<String>) -> axum::Json<serde_json::Value> {
    let turns = table().session(&session);
    let rel = |base: Instant, t: Option<Instant>| t.map(|t| (ms(t.saturating_duration_since(base)) * 10.0).round() / 10.0);
    let rows: Vec<serde_json::Value> = turns
        .iter()
        .map(|t| {
            let mut stages = serde_json::Map::new();
            for k in Kind::ALL {
                let s = &t.stages[k.index()];
                if s.arrived.is_none() {
                    continue;
                }
                stages.insert(
                    k.stage().into(),
                    serde_json::json!({
                        "in_ms": rel(t.speech_end, s.arrived),
                        "admitted_ms": rel(t.speech_end, s.admitted),
                        "first_ms": rel(t.speech_end, s.first),
                        "done_ms": rel(t.speech_end, s.done),
                        "device_ms": (s.device_ms * 10.0).round() / 10.0,
                        "wait_turn_ms": (s.wait_ms * 10.0).round() / 10.0,
                    }),
                );
            }
            serde_json::json!({
                "turn_id": &*t.id,
                "trace_id": format!("{:032x}", t.trace_id),
                "budget_ms": t.budget.as_millis() as u64,
                "age_ms": ms(t.speech_end.elapsed()).round(),
                "stages": stages,
                "audio_s": (t.audio_s * 1e3).round() / 1e3,
                "underrun_ms": (t.underrun_s.max(0.0) * 1e4).round() / 10.0,
                "playback": t.playback.map(|p| match p {
                    Playback::Started(v) => serde_json::json!({"started": v}),
                    Playback::Buffered(v) => serde_json::json!({"buffered_ms": v}),
                }),
            })
        })
        .collect();
    axum::Json(serde_json::json!({ "session_id": session, "turns": rows }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid(s: &str) -> Arc<str> {
        s.into()
    }

    #[test]
    fn inference_links_asr_chat_and_tts_of_one_turn() {
        let t = TurnTable::new();
        let s = sid("call-a");
        let t0 = Instant::now();
        let a = t.join(&s, None, Kind::Asr, t0, None, None, None);
        let l = t.join(&s, None, Kind::Llm, t0 + Duration::from_millis(300), None, None, None);
        let v = t.join(&s, None, Kind::Tts, t0 + Duration::from_millis(900), None, None, None);
        // A minted turn id is an 11-char svid, and the trace id's high half is that svid.
        assert_eq!(a.key.turn.len(), 11);
        assert_ne!(a.trace_id >> 64, 0);
        assert_eq!(a.key, l.key);
        assert_eq!(a.key, v.key);
        assert_eq!(a.trace_id, v.trace_id);
        assert_eq!(v.speech_end, t0);
        // Next utterance: a new turn with its own trace.
        let a2 = t.join(&s, None, Kind::Asr, t0 + Duration::from_secs(5), None, None, None);
        assert_ne!(a2.key, a.key);
        assert_ne!(a2.trace_id, a.trace_id);
        let l2 = t.join(&s, None, Kind::Llm, t0 + Duration::from_secs(6), None, None, None);
        assert_eq!(l2.key, a2.key);
        assert_eq!(l2.speech_end, t0 + Duration::from_secs(5));
    }

    #[test]
    fn requests_without_an_open_turn_open_their_own() {
        let t = TurnTable::new();
        let s = sid("tts-only");
        let t0 = Instant::now();
        let v1 = t.join(&s, None, Kind::Tts, t0, None, None, None);
        let v2 = t.join(&s, None, Kind::Tts, t0 + Duration::from_millis(10), None, None, None);
        assert_ne!(v1.key, v2.key);
        assert_eq!(v2.speech_end, t0 + Duration::from_millis(10));
        // Chat-only: each chat opens a turn; a chat after a chat is a new turn.
        let s = sid("text");
        let l1 = t.join(&s, None, Kind::Llm, t0, None, None, None);
        let l2 = t.join(&s, None, Kind::Llm, t0, None, None, None);
        assert_ne!(l1.key, l2.key);
        let v = t.join(&s, None, Kind::Tts, t0, None, None, None);
        assert_eq!(v.key, l2.key);
    }

    #[test]
    fn explicit_turn_ids_and_client_traces_win() {
        let t = TurnTable::new();
        let s = sid("call-b");
        let id: Arc<str> = "t7".into();
        let t0 = Instant::now();
        let l = t.join(&s, Some(&id), Kind::Llm, t0, None, Some(Duration::from_millis(900)), None);
        let v = t.join(&s, Some(&id), Kind::Tts, t0, Some(42), None, None);
        assert_eq!(l.key, v.key);
        assert_eq!(&*v.key.turn, "t7");
        assert_eq!(v.trace_id, 42);
        assert_eq!(v.budget, Duration::from_millis(900));
        // Stamps are visible through times().
        t.stamp(&v.key, Stage::TtsFirst, t0 + Duration::from_millis(5));
        let times = t.times(&v.key).unwrap();
        assert_eq!(times.tts_first_audio, Some(t0 + Duration::from_millis(5)));
        assert_eq!(times.speech_end, Some(t0));
    }

    #[test]
    fn sessions_and_turns_are_bounded() {
        let t = TurnTable::new();
        let s = sid("long");
        let t0 = Instant::now();
        assert!(!t.knows_session("long"));
        for _ in 0..TURNS_PER_SESSION + 5 {
            t.join(&s, None, Kind::Asr, t0, None, None, None);
        }
        assert_eq!(t.session("long").len(), TURNS_PER_SESSION);
        assert!(t.knows_session("long"));
    }

    #[test]
    fn server_timing_is_valid_syntax() {
        let st = ServerTiming { queue: Some(1.25), wait_turn: Some(0.0), device: Some(30.04), first: Some(50.0), total: None, slack: Some(-12.5), turn: None };
        assert_eq!(st.header(), "queue;dur=1.2, wait-turn;dur=0.0, device;dur=30.0, first;dur=50.0, slack;dur=-12.5");
        assert_eq!(ServerTiming::default().header(), "");
    }

    #[test]
    fn a_stage_run_stamps_turn_trace_and_timing() {
        let mut ids = RequestIds::generated();
        ids.session = Some("stage-run-test".into());
        ids.trace = crate::serve::session::TraceParent::parse("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
        let mut run = StageRun::start(&ids, Kind::Llm, "m", None, Instant::now(), true);
        run.first();
        run.done();
        let h = run.headers();
        let tp = h[TRACEPARENT].to_str().unwrap();
        assert!(tp.starts_with("00-4bf92f3577b34da6a3ce929d0e0e4736-") && tp.ends_with("-01") && tp.len() == 55, "{tp}");
        assert!(!tp.contains("00f067aa0ba902b7"), "a request gets its own span id");
        let turn = h[TURN_HEADER].to_str().unwrap().to_owned();
        assert_eq!(turn.len(), 11);
        let st = h[SERVER_TIMING].to_str().unwrap();
        assert!(st.contains("first;dur=") && st.contains("total;dur=") && st.contains("slack;dur="), "{st}");
        // The TTS of the same turn shares its trace without sending one.
        let mut ids = RequestIds::generated();
        ids.session = Some("stage-run-test".into());
        let tts = StageRun::start(&ids, Kind::Tts, "m", None, Instant::now(), true);
        assert_eq!(tts.trace_id, 0x4bf92f3577b34da6a3ce929d0e0e4736);
        assert_eq!(tts.turn_id.as_deref(), Some(turn.as_str()));
        let comment = format!("{:?}", run.sse_comment());
        assert!(comment.contains("server-timing"), "{comment}");
    }

    #[test]
    fn playback_header_parses() {
        assert_eq!(Playback::parse("started=1700000000000"), Some(Playback::Started(1_700_000_000_000)));
        assert_eq!(Playback::parse(" buffered_ms = 250"), Some(Playback::Buffered(250)));
        assert_eq!(Playback::parse("bogus=1"), None);
        assert_eq!(Playback::parse("started"), None);
    }

    #[test]
    fn slack_histogram_buckets_negative_values() {
        let h = SignedHistogram::default();
        h.observe(-1_500_000);
        h.observe(50_000);
        let mut out = String::new();
        h.write(&mut out, "x", "stage=\"s\"");
        assert!(out.contains("x_bucket{stage=\"s\",le=\"-1\"} 1\n"), "{out}");
        assert!(out.contains("x_bucket{stage=\"s\",le=\"0.1\"} 2\n"), "{out}");
        assert!(out.contains("x_count{stage=\"s\"} 2\n"));
    }
}
