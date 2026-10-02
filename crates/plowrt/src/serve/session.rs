//! Request and session identity (`X-Request-Id`, `X-Session-Id`) and the per-model table of
//! retained sequences a session's next request resumes from.
//!
//! A request's KV is retained after it finishes when it carried a session id: its slot stays
//! out of admission, keyed by the session, until the session's next request takes it (the
//! longest prefix whose rows are identical is kept, the rest is prefilled), the idle TTL passes,
//! or a live request needs the slot. Rows are compared by content key ([`row_keys`]): the token
//! id and the host overlay rows (audio embeddings, voice conditioning) that replace its
//! embedding, so a reused row is always one a fresh prefill would have produced.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::{HeaderMap, HeaderName, HeaderValue};
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};

pub const REQUEST_HEADER: HeaderName = HeaderName::from_static("x-request-id");
pub const SESSION_HEADER: HeaderName = HeaderName::from_static("x-session-id");
pub const TURN_ID_HEADER: HeaderName = HeaderName::from_static("x-turn-id");
pub const TURN_BUDGET_HEADER: HeaderName = HeaderName::from_static("x-turn-budget-ms");
pub const PLAYBACK_HEADER: HeaderName = HeaderName::from_static("x-playback");
const TRACEPARENT_HEADER: HeaderName = HeaderName::from_static("traceparent");
const MAX_ID_BYTES: usize = 128;
/// OpenRouter's `session_id` limit.
const MAX_SESSION_BYTES: usize = 256;

/// A W3C trace context the client sent (`traceparent`, or body `trace`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceParent {
    pub trace_id: u128,
    /// The caller's span (0: none given).
    pub parent: u64,
    pub flags: u8,
}

impl TraceParent {
    /// `version-traceid-parentid-flags`, lowercase hex. Invalid → `None` (W3C: restart the trace).
    pub fn parse(value: &str) -> Option<Self> {
        let v = value.trim();
        let b = v.as_bytes();
        let hex = |s: &str| s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c));
        if b.len() < 55 || b[2] != b'-' || b[35] != b'-' || b[52] != b'-' || !hex(&v[..2]) || &v[..2] == "ff" {
            return None;
        }
        // Version 00 is exactly 55 bytes; a later version may append fields after a '-'.
        if (&v[..2] == "00" && b.len() != 55) || (b.len() > 55 && b[55] != b'-') {
            return None;
        }
        let (t, p, f) = (&v[3..35], &v[36..52], &v[53..55]);
        if !(hex(t) && hex(p) && hex(f)) {
            return None;
        }
        let trace_id = u128::from_str_radix(t, 16).ok().filter(|&x| x != 0)?;
        let parent = u64::from_str_radix(p, 16).ok().filter(|&x| x != 0)?;
        Some(Self { trace_id, parent, flags: u8::from_str_radix(f, 16).ok()? })
    }

    /// OpenRouter's body `trace` object: `trace_id` (32 hex) and `parent_span_id` (16 hex).
    fn from_body(trace: &serde_json::Value) -> Option<Self> {
        let field = |k: &str, len: usize| {
            trace.get(k)?.as_str().map(str::trim).filter(|s| s.len() == len && s.bytes().all(|c| c.is_ascii_hexdigit()))
        };
        let trace_id = u128::from_str_radix(field("trace_id", 32)?, 16).ok().filter(|&x| x != 0)?;
        let parent = field("parent_span_id", 16).and_then(|p| u64::from_str_radix(p, 16).ok()).unwrap_or(0);
        Some(Self { trace_id, parent, flags: 0 })
    }
}

/// Body fields that route a request (OpenRouter-compatible); flattened into the request DTOs.
#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct RouteFields {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    /// OpenAI `metadata`; `turn_id` and `turn_budget_ms` are read.
    #[serde(default)]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub trace: Option<serde_json::Value>,
}

/// One request's identity: its id (the client's `X-Request-Id`, or generated), its session, and
/// its turn and trace context.
#[derive(Clone, Debug)]
pub struct RequestIds {
    pub request: Arc<str>,
    pub session: Option<Arc<str>>,
    pub trace: Option<TraceParent>,
    /// The client's turn id (`X-Turn-Id` / `metadata.turn_id`); `None` = inferred per session.
    pub turn: Option<Arc<str>>,
    pub budget_ms: Option<u32>,
    pub playback: Option<crate::serve::turns::Playback>,
    /// The turn this request joined (`serve::turns::StageRun::start`), for the mux job.
    pub turn_key: Option<crate::serve::turns::TurnKey>,
}

/// An id of 1..=`max` visible ASCII bytes.
fn valid_id(value: &str, max: usize) -> bool {
    !value.is_empty() && value.len() <= max && value.bytes().all(|b| b.is_ascii_graphic())
}

fn budget_ms(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok().filter(|&ms| (1..=600_000).contains(&ms))
}

impl RequestIds {
    /// Read the identity headers. A request or turn id must be 1..=128 visible ASCII bytes, a
    /// session id 1..=256. An invalid `traceparent` or `X-Playback` is ignored.
    pub fn from_headers(headers: &HeaderMap) -> Result<Self, String> {
        let read = |name: &HeaderName, max: usize| -> Result<Option<Arc<str>>, String> {
            let Some(value) = headers.get(name) else { return Ok(None) };
            let value = value.to_str().map_err(|_| format!("{name} must be visible ASCII"))?.trim();
            if !valid_id(value, max) {
                return Err(format!("{name} must be 1..={max} visible ASCII characters"));
            }
            Ok(Some(value.into()))
        };
        let text = |name: &HeaderName| headers.get(name).and_then(|v| v.to_str().ok());
        let budget_ms = match text(&TURN_BUDGET_HEADER) {
            Some(v) => Some(budget_ms(v).ok_or_else(|| format!("{TURN_BUDGET_HEADER} must be 1..=600000 ms"))?),
            None => None,
        };
        Ok(Self {
            request: read(&REQUEST_HEADER, MAX_ID_BYTES)?.unwrap_or_else(|| generate_id().into()),
            session: read(&SESSION_HEADER, MAX_SESSION_BYTES)?,
            trace: text(&TRACEPARENT_HEADER).and_then(TraceParent::parse),
            turn: read(&TURN_ID_HEADER, MAX_ID_BYTES)?,
            budget_ms,
            playback: text(&PLAYBACK_HEADER).and_then(crate::serve::turns::Playback::parse),
            turn_key: None,
        })
    }

    /// Body fallbacks, OpenRouter precedence: session = body `session_id` > `X-Session-Id` > body
    /// `prompt_cache_key`; trace = `traceparent` > body `trace`; turn and budget = header > body
    /// `metadata.turn_id` / `metadata.turn_budget_ms`. An invalid body `session_id` is an error;
    /// the other body fields are hints and are ignored when invalid.
    pub fn apply_body(&mut self, body: &RouteFields) -> Result<(), String> {
        if let Some(s) = &body.session_id {
            let s = s.trim();
            if !valid_id(s, MAX_SESSION_BYTES) {
                return Err(format!("session_id must be 1..={MAX_SESSION_BYTES} visible ASCII characters"));
            }
            self.session = Some(s.into());
        } else if self.session.is_none() {
            self.session = body.prompt_cache_key.as_deref().map(str::trim).filter(|k| valid_id(k, MAX_SESSION_BYTES)).map(Into::into);
        }
        if self.trace.is_none() {
            self.trace = body.trace.as_ref().and_then(TraceParent::from_body);
        }
        let meta = |k: &str| -> Option<String> {
            match body.metadata.as_ref()?.get(k)? {
                serde_json::Value::String(s) => Some(s.trim().to_owned()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            }
        };
        if self.turn.is_none() {
            self.turn = meta("turn_id").filter(|t| valid_id(t, MAX_ID_BYTES)).map(Into::into);
        }
        if self.budget_ms.is_none() {
            self.budget_ms = meta("turn_budget_ms").as_deref().and_then(budget_ms);
        }
        Ok(())
    }

    /// The client's turn budget.
    pub fn budget(&self) -> Option<Duration> {
        self.budget_ms.map(|ms| Duration::from_millis(u64::from(ms)))
    }

    /// A generated request id and no session.
    pub fn generated() -> Self {
        Self { request: generate_id().into(), session: None, trace: None, turn: None, budget_ms: None, playback: None, turn_key: None }
    }

    /// The same identity under a fresh request id (one request of a multi-request connection).
    pub fn with_new_request(&self) -> Self {
        Self { request: generate_id().into(), turn_key: None, ..self.clone() }
    }

    /// The response headers echoing this identity.
    pub fn header_pairs(&self) -> impl Iterator<Item = (HeaderName, HeaderValue)> + '_ {
        [Some((REQUEST_HEADER, &self.request)), self.session.as_ref().map(|s| (SESSION_HEADER, s))]
            .into_iter()
            .flatten()
            .filter_map(|(name, value)| Some((name, HeaderValue::from_str(value).ok()?)))
    }

    pub fn stamp(&self, response: &mut axum::response::Response) {
        for (name, value) in self.header_pairs() {
            response.headers_mut().insert(name, value);
        }
    }

    /// Register this request as in flight in its session of `model`: `None` when the same request
    /// id is already in flight there (a duplicate). Requests without a session always pass.
    pub fn begin(&self, model: &str) -> Option<InFlight> {
        let Some(session) = self.session.clone() else {
            return Some(InFlight(None));
        };
        let key = (Arc::<str>::from(model), session);
        let mut map = in_flight().lock();
        let set = map.entry(key.clone()).or_default();
        if !set.insert(self.request.clone()) {
            return None;
        }
        Some(InFlight(Some((key, self.request.clone()))))
    }

    /// The mux ticket for a request of this session over rows keyed `keys`.
    /// The mux ticket for a request of this session over rows keyed `keys`; `report` learns what
    /// admission reused.
    pub fn ticket(&self, keys: Vec<u64>, report: Option<Report>) -> Option<Box<SessionTicket>> {
        let session = self.session.clone()?;
        (retention_ttl() > Duration::ZERO)
            .then(|| Box::new(SessionTicket { session, request: self.request.clone(), keys, report }))
    }

    /// A report channel when this request has a session.
    pub fn report(&self) -> (Option<Report>, Option<tokio::sync::oneshot::Receiver<CacheOutcome>>) {
        match self.session {
            Some(_) => {
                let (tx, rx) = tokio::sync::oneshot::channel();
                (Some(tx), Some(rx))
            }
            None => (None, None),
        }
    }
}

pub const CACHE_HEADER: HeaderName = HeaderName::from_static("x-session-cache");
pub const CACHED_ROWS_HEADER: HeaderName = HeaderName::from_static("x-session-cached-tokens");

/// What a session request found at admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheStatus {
    /// Resumed `rows` retained rows.
    Hit,
    /// Nothing retained for the session (a new session, or no row shared).
    Miss,
    /// The session's rows were retained and are gone: idle TTL, a live request needed the slot,
    /// or the retention cap.
    Evicted,
    /// The engine's VMM prefix cache serves the session (`usage.prompt_tokens_details
    /// .cached_tokens` says how much); its published prefix is pinned for the TTL.
    Prefix,
    /// This engine retains nothing (`PLOW_SESSION_TTL_MS=0`, or KV that does not stay in its slot).
    Off,
}

impl CacheStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            CacheStatus::Hit => "hit",
            CacheStatus::Miss => "miss",
            CacheStatus::Evicted => "evicted",
            CacheStatus::Prefix => "prefix-cache",
            CacheStatus::Off => "off",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheOutcome {
    pub status: CacheStatus,
    /// Prompt rows resumed instead of prefilled.
    pub rows: usize,
    /// When the request was admitted to a slot (`None`: never admitted).
    pub at: Option<Instant>,
}

impl CacheOutcome {
    pub const OFF: Self = Self { status: CacheStatus::Off, rows: 0, at: None };

    /// `X-Session-Cache` and `X-Session-Cached-Tokens`.
    pub fn stamp(&self, response: &mut axum::response::Response) {
        let headers = response.headers_mut();
        headers.insert(CACHE_HEADER, HeaderValue::from_static(self.status.as_str()));
        headers.insert(CACHED_ROWS_HEADER, HeaderValue::from(self.rows));
    }

    /// Await a request's report (a job refused before admission reports nothing: `Off`).
    pub async fn received(rx: Option<tokio::sync::oneshot::Receiver<CacheOutcome>>) -> Option<Self> {
        Some(rx?.await.unwrap_or(Self::OFF))
    }
}

pub type Report = tokio::sync::oneshot::Sender<CacheOutcome>;

/// Keep `guard` alive until `response`'s body is sent: a streamed body outlives its handler.
pub fn hold_until_sent<G: Send + Sync + 'static>(response: axum::response::Response, guard: G) -> axum::response::Response {
    use futures::StreamExt;
    let (parts, body) = response.into_parts();
    let body = axum::body::Body::from_stream(body.into_data_stream().map(move |chunk| {
        let _held = &guard;
        chunk
    }));
    axum::response::Response::from_parts(parts, body)
}

type SessionKey = (Arc<str>, Arc<str>);

fn in_flight() -> &'static Mutex<FxHashMap<SessionKey, FxHashSet<Arc<str>>>> {
    static M: std::sync::OnceLock<Mutex<FxHashMap<SessionKey, FxHashSet<Arc<str>>>>> = std::sync::OnceLock::new();
    M.get_or_init(Default::default)
}

/// Holds a request id in its session's in-flight set until dropped.
pub struct InFlight(Option<(SessionKey, Arc<str>)>);

impl Drop for InFlight {
    fn drop(&mut self) {
        let Some((key, request)) = self.0.take() else { return };
        let mut map = in_flight().lock();
        if let Some(set) = map.get_mut(&key) {
            set.remove(&request);
            if set.is_empty() {
                map.remove(&key);
            }
        }
    }
}

fn generate_id() -> String {
    minted::request()
}

/// Ids the server mints when the client sent none: svids (time-sortable 64-bit, tagged by kind,
/// server source). Strings are the fixed 11-char base58 form. Client ids are never parsed.
pub mod minted {
    use std::sync::OnceLock;

    #[derive(svid::Svid, Clone, Copy, Debug, PartialEq, Eq)]
    #[svid(registry = IdRegistry)]
    #[repr(u8)]
    pub enum IdTag {
        RequestId = 1,
        SessionId = 2,
        TurnId = 3,
        SpanId = 4,
    }

    fn registry() -> &'static IdRegistry {
        static R: OnceLock<IdRegistry> = OnceLock::new();
        R.get_or_init(|| IdRegistry::new(false))
    }

    pub fn request() -> String {
        registry().request_id.generate_id().to_str()
    }

    pub fn session() -> String {
        registry().session_id.generate_id().to_str()
    }

    /// A turn id: the svid and its string.
    pub fn turn() -> (i64, String) {
        let id = registry().turn_id.generate_id();
        (id.to_i64(), id.to_str())
    }

    /// A W3C span id: the svid's 64 bits (never zero).
    pub fn span() -> u64 {
        registry().span_id.generate_id().to_i64() as u64
    }

    /// A W3C trace id: a turn svid (sorts by time) over 64 CSPRNG bits.
    pub fn trace_id(turn: i64) -> u128 {
        (u128::from(turn as u64) << 64) | u128::from(rand::random::<u64>())
    }
}

/// `PLOW_SESSION_TTL_MS`: how long a finished session request's KV stays retained (0: never).
pub fn retention_ttl() -> Duration {
    Duration::from_millis(crate::config::RuntimeConfig::get().session_ttl_ms)
}

fn mix(x: u64) -> u64 {
    let m = u128::from(x ^ 0x9e37_79b9_7f4a_7c15) * 0xbf58_476d_1ce4_e5b9;
    (m as u64) ^ ((m >> 64) as u64)
}

fn fold(h: u64, w: u64) -> u64 {
    let m = u128::from(h ^ w) * 0x94d0_49bb_1331_11eb;
    (m as u64) ^ ((m >> 64) as u64)
}

/// The content key of a plain token row.
pub fn token_key(id: u32) -> u64 {
    mix(u64::from(id) | 1 << 40)
}

/// A content hash of one overlay row's bits.
pub fn row_hash(row: &[f32]) -> u64 {
    let mut h = row.len() as u64;
    let mut words = row.chunks_exact(2);
    for pair in &mut words {
        h = fold(h, u64::from(pair[0].to_bits()) | u64::from(pair[1].to_bits()) << 32);
    }
    if let [last] = words.remainder() {
        h = fold(h, u64::from(last.to_bits()));
    }
    mix(h)
}

/// Per-position content keys of a prompt: each row's token id and the rows of every overlay
/// (`[rows][hidden]`, one row per `overlay_pos` entry) that replace its embedding.
pub fn row_keys(prompt_ids: &[u32], overlay_pos: &[u32], overlays: &[&[f32]]) -> Vec<u64> {
    let mut keys: Vec<u64> = prompt_ids.iter().map(|&id| token_key(id)).collect();
    let rows = overlay_pos.len();
    for overlay in overlays {
        if rows == 0 || overlay.len() % rows != 0 {
            continue;
        }
        let hidden = overlay.len() / rows;
        for (&pos, row) in overlay_pos.iter().zip(overlay.chunks_exact(hidden)) {
            if let Some(key) = keys.get_mut(pos as usize) {
                *key = fold(*key, row_hash(row));
            }
        }
    }
    keys
}

/// A session request's retention contract, carried on its mux job.
#[derive(Debug)]
pub struct SessionTicket {
    pub session: Arc<str>,
    pub request: Arc<str>,
    /// One content key per prompt row ([`row_keys`]).
    pub keys: Vec<u64>,
    pub report: Option<Report>,
}

/// A finished session request's cache rows, on their way into the dispatcher's table.
#[derive(Debug)]
pub struct Retired {
    pub session: Arc<str>,
    pub slot: usize,
    /// A CFG pair: the slot and its partner (slot + 1) both hold rows keyed `keys`.
    pub pair: bool,
    pub keys: Vec<u64>,
}

pub type RetireInbox = Arc<Mutex<Vec<Retired>>>;

/// A live session request's seat: records how many of its rows are complete and, when the slot
/// is dropped, hands them to the dispatcher's retain table.
pub struct Seat {
    ticket: Box<SessionTicket>,
    slot: usize,
    pair: bool,
    /// Overlay/position-base rows: only prefill-embedded prompt rows are kept (the last prompt row
    /// and generated rows are embedded by the decode program).
    speech: bool,
    prompt_rows: usize,
    /// Rows known written: 0 until the first token.
    valid: usize,
    last: Option<u32>,
    /// `None`: the engine's prefix cache keeps the rows ([`Self::pin_ttl`]); nothing retires here.
    inbox: Option<RetireInbox>,
    pin: std::time::Duration,
}

impl Seat {
    pub fn new(ticket: Box<SessionTicket>, slot: usize, pair: bool, speech: bool, inbox: RetireInbox) -> Self {
        let prompt_rows = ticket.keys.len();
        Self { ticket, slot, pair, speech, prompt_rows, valid: 0, last: None, inbox: Some(inbox), pin: Duration::ZERO }
    }

    /// A seat whose rows the VMM prefix cache keeps, pinned for `ttl` when the sequence retires.
    fn pinned(ticket: Box<SessionTicket>, slot: usize, ttl: Duration) -> Self {
        Self { ticket, slot, pair: false, speech: false, prompt_rows: 0, valid: 0, last: None, inbox: None, pin: ttl }
    }

    /// How long the engine should pin this sequence's published prefix (`None`: slot retention).
    pub fn pin_ttl(&self) -> Option<Duration> {
        self.inbox.is_none().then_some(self.pin)
    }

    /// A token was produced: every row before it is in the cache.
    pub fn on_token(&mut self, token: u32) {
        if self.speech {
            self.valid = self.prompt_rows.saturating_sub(1);
            return;
        }
        if let Some(prev) = self.last.replace(token) {
            self.ticket.keys.push(token_key(prev));
        }
        self.valid = self.ticket.keys.len();
    }
}

impl Drop for Seat {
    fn drop(&mut self) {
        let Some(inbox) = self.inbox.take() else { return };
        if self.valid == 0 {
            return;
        }
        let mut keys = std::mem::take(&mut self.ticket.keys);
        keys.truncate(self.valid);
        inbox.lock().push(Retired { session: self.ticket.session.clone(), slot: self.slot, pair: self.pair, keys });
    }
}

struct Entry {
    session: Arc<str>,
    slot: usize,
    pair: bool,
    keys: Vec<u64>,
    used: Instant,
}

impl Entry {
    fn holds(&self, slot: usize) -> bool {
        slot == self.slot || (self.pair && slot == self.slot + 1)
    }
    fn rows(&self) -> u64 {
        self.keys.len() as u64 * (1 + self.pair as u64)
    }
}

/// A session's resumable seat: the slot (the pair's owner) and the rows to keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resume {
    pub slot: usize,
    pub rows: usize,
}

/// One dispatcher's retained sequences: at most one per session, each pinning its slot(s).
pub struct RetainTable {
    entries: Vec<Entry>,
    ttl: Duration,
    max: usize,
    /// Sessions whose rows were evicted, and when: their next request reports `evicted`.
    gone: FxHashMap<Arc<str>, Instant>,
    /// Entries evicted (TTL, pressure, cap) so far.
    pub evictions: u64,
}

/// Evicted sessions remembered at once (for `X-Session-Cache: evicted`).
const MAX_GONE: usize = 4096;

impl RetainTable {
    /// `max` = 0: bounded only by the slots.
    pub fn new(ttl: Duration, max: usize) -> Self {
        Self { entries: Vec::new(), ttl, max, gone: Default::default(), evictions: 0 }
    }

    fn evicted(&mut self, e: Entry, now: Instant) {
        self.evictions += 1;
        if self.gone.len() >= MAX_GONE {
            let ttl = self.ttl;
            self.gone.retain(|_, t| now.saturating_duration_since(*t) < ttl);
            if self.gone.len() >= MAX_GONE {
                self.gone.clear();
            }
        }
        self.gone.insert(e.session, now);
    }

    /// What admission found for `session` (`resumed` rows > 0 is a hit). The session's eviction
    /// record is consumed.
    pub fn outcome(&mut self, session: &str, resumed: usize) -> CacheOutcome {
        if !self.enabled() {
            return CacheOutcome::OFF;
        }
        let gone = self.gone.remove(session).is_some();
        let status = match (resumed > 0, gone) {
            (true, _) => CacheStatus::Hit,
            (false, true) => CacheStatus::Evicted,
            (false, false) => CacheStatus::Miss,
        };
        CacheOutcome { status, rows: resumed, at: None }
    }

    pub fn enabled(&self) -> bool {
        self.ttl > Duration::ZERO
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn holds(&self, slot: usize) -> bool {
        self.entries.iter().any(|e| e.holds(slot))
    }

    /// Retained cache rows (a pair counts both members).
    pub fn rows(&self) -> u64 {
        self.rows_except(None).sum()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Drop entries idle past the TTL.
    pub fn sweep(&mut self, now: Instant) {
        let ttl = self.ttl;
        let mut i = 0;
        while i < self.entries.len() {
            if now.saturating_duration_since(self.entries[i].used) >= ttl {
                let e = self.entries.swap_remove(i);
                self.evicted(e, now);
            } else {
                i += 1;
            }
        }
    }

    /// Retain a finished request's rows: the session's newest request replaces its older one, and
    /// past `max` the least recently used entry goes.
    pub fn insert(&mut self, retired: Retired, now: Instant) {
        if !self.enabled() || retired.keys.is_empty() {
            return;
        }
        self.entries.retain(|e| {
            e.session != retired.session && !e.holds(retired.slot) && !(retired.pair && e.holds(retired.slot + 1))
        });
        self.entries.push(Entry { session: retired.session, slot: retired.slot, pair: retired.pair, keys: retired.keys, used: now });
        while self.max > 0 && self.entries.len() > self.max {
            self.evict_lru(usize::MAX, None);
        }
    }

    /// Where `session`'s request keyed `keys` resumes: its entry when live, of the same shape
    /// (`pair`), under `limit`, and sharing at least one row. At least one prompt row is always
    /// left to prefill (the first token needs its logits).
    pub fn lookup(&self, session: &str, keys: &[u64], pair: bool, limit: usize, now: Instant) -> Option<Resume> {
        let e = self.entries.iter().find(|e| &*e.session == session)?;
        if now.saturating_duration_since(e.used) >= self.ttl || e.pair != pair || e.slot + pair as usize >= limit {
            return None;
        }
        let shared = e.keys.iter().zip(keys).take_while(|(a, b)| a == b).count();
        let rows = shared.min(keys.len().saturating_sub(1));
        (rows > 0).then_some(Resume { slot: e.slot, rows })
    }

    /// Drop `session`'s entry (its next request is seated: resumed or not, it retires anew).
    pub fn forget(&mut self, session: &str) {
        self.entries.retain(|e| &*e.session != session);
    }

    /// Each entry's retained rows, except `session`'s.
    pub fn rows_except<'a>(&'a self, session: Option<&'a str>) -> impl Iterator<Item = u64> + 'a {
        self.entries.iter().filter(move |e| Some(&*e.session) != session).map(Entry::rows)
    }

    /// Evict the least recently used entry holding a slot below `limit`, other than `keep`'s;
    /// false when none does.
    pub fn evict_lru(&mut self, limit: usize, keep: Option<&str>) -> bool {
        let lru = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.slot < limit && Some(&*e.session) != keep)
            .min_by_key(|(_, e)| e.used)
            .map(|(i, _)| i);
        match lru {
            Some(i) => {
                let e = self.entries.swap_remove(i);
                tracing::debug!(session = %e.session, slot = e.slot, rows = e.keys.len(), "session: retained KV evicted");
                self.evicted(e, Instant::now());
                true
            }
            None => false,
        }
    }
}

/// A dispatcher's retention: its table and the inbox its seats retire into.
pub struct Retention {
    pub table: RetainTable,
    pub inbox: RetireInbox,
    /// KV bytes one retained row holds (0: unknown), for the bytes gauge.
    bytes_per_row: u64,
    evictions_reported: u64,
    /// The engine's VMM prefix cache keeps finished sequences instead of their slots: session
    /// requests pin their prefix there for this TTL.
    prefix_pin: Option<Duration>,
    /// Decode launch widths, ascending (empty: the launch is the live extent).
    widths: Box<[u32]>,
    slack: usize,
}

impl Retention {
    /// Armed from `PLOW_SESSION_TTL_MS` / `PLOW_SESSION_MAX` when the engine keeps a finished
    /// sequence's rows in its slot; otherwise nothing is ever retained.
    pub fn new(supported: bool, prefix_cache: bool, bytes_per_row: u64, widths: &[u32]) -> Self {
        let ttl = if supported { retention_ttl() } else { Duration::ZERO };
        let config = crate::config::RuntimeConfig::get();
        Self {
            table: RetainTable::new(ttl, config.session_max),
            inbox: Default::default(),
            bytes_per_row,
            evictions_reported: 0,
            prefix_pin: (!supported && prefix_cache).then(retention_ttl).filter(|t| !t.is_zero()),
            widths: widths.into(),
            slack: config.session_slack,
        }
    }

    pub fn off() -> Self {
        Self::new(false, false, 0, &[])
    }

    pub fn with_table(table: RetainTable, widths: &[u32], slack: usize) -> Self {
        Self { table, inbox: Default::default(), bytes_per_row: 0, evictions_reported: 0, prefix_pin: None, widths: widths.into(), slack }
    }

    /// The lowest slot a live request may be pushed to by retained slots below it: past the decode
    /// width the live `extent` already runs, or the slack, it would widen every decode launch.
    pub fn floor(&self, extent: usize) -> usize {
        let width = self.widths.iter().map(|&w| w as usize).find(|&w| w >= extent.max(1)).unwrap_or(extent);
        width.max(self.slack)
    }

    /// Seat a live session request: report what admission found (`resumed` rows), count it, and
    /// hold the seat that retires its rows (`None` when retention is off).
    pub fn seat(
        &mut self,
        ticket: Option<Box<SessionTicket>>,
        slot: usize,
        pair: bool,
        speech: bool,
        resumed: usize,
        metrics: &crate::obs::Metrics,
    ) -> Option<Seat> {
        let mut ticket = ticket?;
        if let Some(ttl) = self.prefix_pin {
            if let Some(report) = ticket.report.take() {
                let _ = report.send(CacheOutcome { status: CacheStatus::Prefix, rows: 0, at: Some(Instant::now()) });
            }
            return Some(Seat::pinned(ticket, slot, ttl));
        }
        let outcome = self.table.outcome(&ticket.session, resumed);
        let m = &metrics.serving;
        match outcome.status {
            CacheStatus::Hit => {
                m.session_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                m.session_reused_rows.fetch_add(resumed as u64, std::sync::atomic::Ordering::Relaxed);
            }
            CacheStatus::Miss | CacheStatus::Evicted => {
                m.session_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            CacheStatus::Prefix | CacheStatus::Off => {}
        }
        if let Some(report) = ticket.report.take() {
            let _ = report.send(CacheOutcome { at: Some(Instant::now()), ..outcome });
        }
        self.table.enabled().then(|| Seat::new(ticket, slot, pair, speech, self.inbox.clone()))
    }

    /// Export the table's size and evictions.
    pub fn publish(&mut self, metrics: &crate::obs::Metrics) {
        if !self.table.enabled() {
            return;
        }
        use std::sync::atomic::Ordering::Relaxed;
        let m = &metrics.serving;
        m.session_evictions.fetch_add(self.table.evictions - self.evictions_reported, Relaxed);
        self.evictions_reported = self.table.evictions;
        let rows = self.table.rows();
        m.session_retained.store(self.table.len() as u64, Relaxed);
        m.session_retained_rows.store(rows, Relaxed);
        m.session_retained_bytes.store(rows * self.bytes_per_row, Relaxed);
    }

    /// Move retired seats into the table and drop idle entries.
    pub fn collect(&mut self, now: Instant) {
        if !self.table.enabled() {
            return;
        }
        for retired in std::mem::take(&mut *self.inbox.lock()) {
            self.table.insert(retired, now);
        }
        self.table.sweep(now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn retired(session: &str, slot: usize, keys: &[u64]) -> Retired {
        Retired { session: session.into(), slot, pair: false, keys: keys.to_vec() }
    }

    #[test]
    fn a_session_resumes_its_longest_shared_prefix_and_leaves_a_row_to_prefill() {
        let now = Instant::now();
        let mut t = RetainTable::new(Duration::from_secs(60), 0);
        t.insert(retired("a", 3, &[1, 2, 3, 4]), now);
        assert!(t.holds(3) && !t.holds(2));
        assert_eq!(t.lookup("a", &[1, 2, 9, 9, 9], false, 8, now), Some(Resume { slot: 3, rows: 2 }));
        // An identical prompt still prefills its last row (the first token needs its logits).
        assert_eq!(t.lookup("a", &[1, 2, 3, 4], false, 8, now), Some(Resume { slot: 3, rows: 3 }));
        assert_eq!(t.lookup("a", &[7, 2, 3, 4], false, 8, now), None, "no shared row: recompute");
        t.forget("a");
        assert!(t.is_empty());
    }

    #[test]
    fn sessions_are_isolated() {
        let now = Instant::now();
        let mut t = RetainTable::new(Duration::from_secs(60), 0);
        t.insert(retired("a", 0, &[1, 2, 3]), now);
        t.insert(retired("b", 1, &[1, 2, 3]), now);
        assert_eq!(t.lookup("c", &[1, 2, 3, 4], false, 8, now), None);
        assert_eq!(t.lookup("b", &[1, 2, 3, 4], false, 8, now), Some(Resume { slot: 1, rows: 3 }));
        assert_eq!(t.rows_except(Some("b")).collect::<Vec<_>>(), [3]);
        t.forget("b");
        assert!(t.holds(0) && !t.holds(1));
    }

    #[test]
    fn entries_expire_after_the_ttl() {
        let now = Instant::now();
        let mut t = RetainTable::new(Duration::from_millis(50), 0);
        t.insert(retired("a", 0, &[1, 2, 3]), now);
        t.insert(retired("b", 1, &[1, 2, 3]), now + Duration::from_millis(40));
        assert_eq!(t.lookup("a", &[1, 2, 3, 4], false, 8, now + Duration::from_millis(60)), None);
        t.sweep(now + Duration::from_millis(80));
        assert_eq!(t.len(), 1);
        t.sweep(now + Duration::from_millis(100));
        assert!(t.is_empty());
    }

    #[test]
    fn eviction_is_least_recently_used_and_bounded() {
        let now = Instant::now();
        let mut t = RetainTable::new(Duration::from_secs(60), 2);
        t.insert(retired("a", 0, &[1]), now);
        t.insert(retired("b", 1, &[1]), now + Duration::from_millis(1));
        t.insert(retired("c", 2, &[1]), now + Duration::from_millis(2));
        assert_eq!(t.len(), 2);
        assert!(!t.holds(0), "the oldest entry went past max");
        assert!(!t.evict_lru(1, None), "nothing below the limit");
        assert!(t.evict_lru(8, Some("b")), "the kept session is skipped");
        assert!(t.holds(1) && !t.holds(2));
        assert!(!t.evict_lru(8, Some("b")));
        assert!(t.evict_lru(8, None));
        assert!(t.is_empty());
    }

    #[test]
    fn a_newer_request_replaces_its_session_and_a_disabled_table_keeps_nothing() {
        let now = Instant::now();
        let mut t = RetainTable::new(Duration::from_secs(60), 0);
        t.insert(retired("a", 0, &[1, 2]), now);
        t.insert(retired("a", 5, &[1, 2, 3]), now);
        assert_eq!(t.len(), 1);
        assert!(t.holds(5) && !t.holds(0));
        let mut off = RetainTable::new(Duration::ZERO, 0);
        off.insert(retired("a", 0, &[1, 2]), now);
        assert!(off.is_empty());
    }

    #[test]
    fn pairs_hold_both_slots_and_resume_only_as_pairs() {
        let now = Instant::now();
        let mut t = RetainTable::new(Duration::from_secs(60), 0);
        t.insert(Retired { session: "a".into(), slot: 2, pair: true, keys: vec![1, 2, 3] }, now);
        assert!(t.holds(2) && t.holds(3));
        assert_eq!(t.rows(), 6);
        assert_eq!(t.lookup("a", &[1, 2, 3, 4], false, 8, now), None, "a lone request never takes a pair");
        assert_eq!(t.lookup("a", &[1, 2, 3, 4], true, 3, now), None, "partner above the admission limit");
        assert_eq!(t.lookup("a", &[1, 2, 3, 4], true, 4, now), Some(Resume { slot: 2, rows: 3 }));
    }

    #[test]
    fn overlay_rows_key_their_positions_by_content() {
        let ids = [5, 0, 0, 6];
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 2.0, 3.0, 4.5];
        let ka = row_keys(&ids, &[1, 2], &[&a]);
        let kb = row_keys(&ids, &[1, 2], &[&b]);
        assert_eq!(ka[..2], kb[..2]);
        assert_ne!(ka[2], kb[2], "same ids, different audio: a different row");
        assert_eq!(ka[3], token_key(6));
        assert_ne!(ka[1], token_key(0), "an overlay row is not its placeholder token");
        // A CFG pair keys both members' rows.
        assert_ne!(row_keys(&ids, &[1, 2], &[&a, &a])[2], row_keys(&ids, &[1, 2], &[&a, &b])[2]);
    }

    #[test]
    fn seats_retain_prompt_rows_for_speech_and_prompt_plus_output_for_text() {
        let inbox: RetireInbox = Default::default();
        let ticket = |keys: &[u64]| Box::new(SessionTicket { session: "s".into(), request: "r".into(), keys: keys.to_vec(), report: None });
        drop(Seat::new(ticket(&[1, 2, 3]), 0, false, false, inbox.clone()));
        assert!(inbox.lock().is_empty(), "no token: nothing is known written");
        let mut text = Seat::new(ticket(&[1, 2, 3]), 1, false, false, inbox.clone());
        for t in [10, 11, 12] {
            text.on_token(t);
        }
        drop(text);
        let mut speech = Seat::new(ticket(&[1, 2, 3]), 2, true, true, inbox.clone());
        speech.on_token(10);
        speech.on_token(11);
        drop(speech);
        let got = std::mem::take(&mut *inbox.lock());
        assert_eq!(got[0].keys, [1, 2, 3, token_key(10), token_key(11)]);
        assert_eq!((got[1].slot, got[1].pair, got[1].keys.as_slice()), (2, true, &[1u64, 2][..]));
    }

    #[test]
    fn duplicate_request_ids_are_refused_only_while_in_flight_in_the_same_session() {
        let ids = |r: &str, s: Option<&str>| RequestIds { request: r.into(), session: s.map(Into::into), ..RequestIds::generated() };
        let first = ids("r1", Some("dup-s")).begin("m").expect("first");
        assert!(ids("r1", Some("dup-s")).begin("m").is_none());
        assert!(ids("r1", Some("dup-s")).begin("other-model").is_some());
        assert!(ids("r1", Some("dup-t")).begin("m").is_some());
        assert!(ids("r2", Some("dup-s")).begin("m").is_some());
        assert!(ids("r1", None).begin("m").is_some() && ids("r1", None).begin("m").is_some());
        drop(first);
        assert!(ids("r1", Some("dup-s")).begin("m").is_some());
    }

    #[test]
    fn ids_are_read_validated_and_generated() {
        let mut h = HeaderMap::new();
        let got = RequestIds::from_headers(&h).unwrap();
        assert_eq!(got.request.len(), 11);
        assert!(got.session.is_none());
        assert_ne!(RequestIds::generated().request, RequestIds::generated().request);
        h.insert(REQUEST_HEADER, HeaderValue::from_static("abc"));
        h.insert(SESSION_HEADER, HeaderValue::from_static("s-1"));
        let got = RequestIds::from_headers(&h).unwrap();
        assert_eq!((&*got.request, got.session.as_deref()), ("abc", Some("s-1")));
        h.insert(SESSION_HEADER, HeaderValue::from_str(&"x".repeat(256)).unwrap());
        assert!(RequestIds::from_headers(&h).is_ok());
        h.insert(SESSION_HEADER, HeaderValue::from_str(&"x".repeat(257)).unwrap());
        assert!(RequestIds::from_headers(&h).is_err());
        h.remove(SESSION_HEADER);
        h.insert(TURN_BUDGET_HEADER, HeaderValue::from_static("soon"));
        assert!(RequestIds::from_headers(&h).is_err());
    }

    #[test]
    fn traceparent_parses_w3c_and_rejects_invalid() {
        let t = TraceParent::parse("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01").unwrap();
        assert_eq!(t.trace_id, 0x4bf92f3577b34da6a3ce929d0e0e4736);
        assert_eq!((t.parent, t.flags), (0x00f067aa0ba902b7, 1));
        for bad in [
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
            "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
            "",
        ] {
            assert_eq!(TraceParent::parse(bad), None, "{bad}");
        }
        // A later version may carry more fields.
        assert!(TraceParent::parse("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-xyz").is_some());
    }

    #[test]
    fn body_fields_follow_openrouter_precedence() {
        let body = |v: serde_json::Value| serde_json::from_value::<RouteFields>(v).unwrap();
        let mut h = HeaderMap::new();
        h.insert(SESSION_HEADER, HeaderValue::from_static("hdr"));
        h.insert(TURN_ID_HEADER, HeaderValue::from_static("t-hdr"));
        h.insert(
            HeaderName::from_static("traceparent"),
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );
        // body session_id > X-Session-Id; headers > metadata; traceparent > body trace.
        let mut ids = RequestIds::from_headers(&h).unwrap();
        ids.apply_body(&body(serde_json::json!({
            "session_id": "body", "prompt_cache_key": "pck",
            "metadata": {"turn_id": "t-body", "turn_budget_ms": "900"},
            "trace": {"trace_id": "0af7651916cd43dd8448eb211c80319c", "parent_span_id": "b7ad6b7169203331"},
        })))
        .unwrap();
        assert_eq!(ids.session.as_deref(), Some("body"));
        assert_eq!(ids.turn.as_deref(), Some("t-hdr"));
        assert_eq!(ids.trace.unwrap().trace_id, 0x4bf92f3577b34da6a3ce929d0e0e4736);
        assert_eq!(ids.budget(), Some(Duration::from_millis(900)));
        // X-Session-Id > prompt_cache_key; body trace and metadata when no headers.
        let mut h2 = HeaderMap::new();
        h2.insert(SESSION_HEADER, HeaderValue::from_static("hdr"));
        let mut ids = RequestIds::from_headers(&h2).unwrap();
        ids.apply_body(&body(serde_json::json!({
            "prompt_cache_key": "pck", "metadata": {"turn_id": 3},
            "trace": {"trace_id": "0af7651916cd43dd8448eb211c80319c", "parent_span_id": "b7ad6b7169203331"},
        })))
        .unwrap();
        assert_eq!(ids.session.as_deref(), Some("hdr"));
        assert_eq!(ids.turn.as_deref(), Some("3"));
        let t = ids.trace.unwrap();
        assert_eq!((t.trace_id, t.parent), (0x0af7651916cd43dd8448eb211c80319c, 0xb7ad6b7169203331));
        // prompt_cache_key alone is the session; an invalid body session_id is refused.
        let mut ids = RequestIds::from_headers(&HeaderMap::new()).unwrap();
        ids.apply_body(&body(serde_json::json!({"prompt_cache_key": "pck"}))).unwrap();
        assert_eq!(ids.session.as_deref(), Some("pck"));
        assert!(ids.apply_body(&body(serde_json::json!({"session_id": "has space"}))).is_err());
        assert!(ids.apply_body(&body(serde_json::json!({"session_id": "x".repeat(257)}))).is_err());
    }
}
