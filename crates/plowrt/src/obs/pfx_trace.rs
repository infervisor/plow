//! `PLOW_PFX_TRACE=<path>`: one JSONL record per request on where its prefix-cache attach
//! stopped and why, for the shadow-cache replay (`scripts/bench/pfx_shadow.py`).
//!
//! Hooks in the VMM pool and the engine build a small [`Event`] and `try_send` it on a bounded
//! channel (never blocking; a full queue drops and counts the event). One writer thread owns the
//! [`Analyzer`]: an exact token trie of everything computed so far (prompts at their attach,
//! outputs at their finish), the boundary registry (publish, spill, drop with its cause,
//! publish refusals) and the JSON output. Unset, every hook is one `OnceLock` load.
//!
//! Records (`ev`): `header` (pool geometry and byte costs), `req` (one per request, written at
//! its finish), `stats` (every 10 s: dropped events, trie size).

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;
use serde::Serialize;

const QUEUE_EVENTS: usize = 8192;
/// The trie restarts empty past this many arena tokens (4 B each); `stats` counts restarts.
const TRIE_CAP_TOKENS: usize = 256 << 20;
/// Registry entries kept before the oldest half is forgotten.
const REGISTRY_CAP: usize = 1 << 22;
const FLUSH_EVERY: Duration = Duration::from_secs(1);
const SCHEMA: u32 = 1;
/// `h16` and boundary identities: [`prefix_hashes`] (64-bit, per token).
const HASH_NAME: &str = "splitmix64-chain/token";
const STATS_EVERY: Duration = Duration::from_secs(10);
/// An arrival not attached within this many seconds is not this attach's.
const ARRIVAL_STALE_S: f64 = 120.0;

struct Tracer {
    tx: SyncSender<Event>,
}

static TRACER: OnceLock<Option<Tracer>> = OnceLock::new();
static DROPPED: AtomicU64 = AtomicU64::new(0);
static EPOCH: OnceLock<Instant> = OnceLock::new();

fn tracer() -> Option<&'static Tracer> {
    TRACER.get()?.as_ref()
}

/// Tracing is armed (`PLOW_PFX_TRACE` set, its file opened, [`init`] ran).
#[inline]
pub fn on() -> bool {
    #[cfg(test)]
    if CAPTURE.with(|c| c.borrow().is_some()) {
        return true;
    }
    tracer().is_some()
}

#[cfg(test)]
thread_local! {
    static CAPTURE: std::cell::RefCell<Option<Vec<Event>>> = const { std::cell::RefCell::new(None) };
}

/// Test hook: run `f` with tracing on for this thread, returning the events it emitted.
#[cfg(test)]
pub(crate) fn capture<R>(f: impl FnOnce() -> R) -> (R, Vec<Event>) {
    CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
    let r = f();
    (
        r,
        CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default()),
    )
}

/// Arm the trace from the resolved runtime config (once, at pool bring-up, before any request)
/// and write its header. A no-op when `PLOW_PFX_TRACE` is unset.
pub fn init(header: Header) {
    let armed = TRACER.get_or_init(|| {
        let path = crate::config::RuntimeConfig::get().pfx_trace.clone()?;
        Tracer::spawn(&path)
    });
    if armed.is_some() {
        emit(Event::Header(Box::new(header)));
    }
}

/// Write the requests still in flight and flush the file (every process exit path, through
/// [`crate::obs::logwriter::flush`]). `false` = not done within `timeout`.
pub fn flush(timeout: Duration) -> bool {
    let Some(t) = tracer() else {
        return true;
    };
    let deadline = Instant::now() + timeout;
    let (ack_tx, ack_rx) = sync_channel(1);
    let mut marker = Event::Flush(ack_tx);
    loop {
        match t.tx.try_send(marker) {
            Ok(()) => break,
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => return false,
            Err(std::sync::mpsc::TrySendError::Full(m)) => {
                if Instant::now() >= deadline {
                    return false;
                }
                marker = m;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
    ack_rx
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .unwrap_or(false)
}

/// Queue `ev` for the writer; a full queue drops it (counted in the `stats` record).
pub fn emit(ev: Event) {
    #[cfg(test)]
    let ev = {
        let mut ev = Some(ev);
        CAPTURE.with(|c| {
            if let Some(v) = c.borrow_mut().as_mut() {
                v.extend(ev.take());
            }
        });
        let Some(ev) = ev else {
            return;
        };
        ev
    };
    if let Some(t) = tracer() {
        if t.tx.try_send(ev).is_err() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Identity of a prompt for matching its [`Event::Arrive`] to its [`Event::Attach`].
pub fn fingerprint(tokens: &[u32]) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    tokens.hash(&mut h);
    h.finish()
}

impl Tracer {
    fn spawn(path: &std::path::Path) -> Option<Self> {
        let file = match std::fs::File::create(path) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "PLOW_PFX_TRACE: cannot open, trace off");
                return None;
            }
        };
        let epoch = *EPOCH.get_or_init(Instant::now);
        let (tx, rx) = sync_channel(QUEUE_EVENTS);
        let wall = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        let spawned = std::thread::Builder::new()
            .name("plowrt-pfxtrace".into())
            .spawn(move || writer(rx, std::io::BufWriter::new(file), epoch, wall));
        match spawned {
            Ok(_) => {
                tracing::info!(path = %path.display(), "PLOW_PFX_TRACE: prefix trace armed");
                Some(Tracer { tx })
            }
            Err(e) => {
                tracing::warn!(error = %e, "PLOW_PFX_TRACE: writer thread failed, trace off");
                None
            }
        }
    }
}

fn writer(rx: Receiver<Event>, mut out: impl Write, epoch: Instant, wall: f64) {
    let mut a = Analyzer::new(epoch, wall);
    let mut last_flush = Instant::now();
    let mut last_stats = Instant::now();
    loop {
        match rx.recv_timeout(FLUSH_EVERY) {
            Ok(Event::Flush(ack)) => {
                a.finish_all(&mut out);
                a.stats(&mut out, DROPPED.load(Ordering::Relaxed));
                let _ = ack.send(out.flush().is_ok());
            }
            Ok(ev) => a.handle(ev, &mut out),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if last_stats.elapsed() >= STATS_EVERY {
            a.stats(&mut out, DROPPED.load(Ordering::Relaxed));
            last_stats = Instant::now();
        }
        if last_flush.elapsed() >= FLUSH_EVERY {
            let _ = out.flush();
            last_flush = Instant::now();
        }
    }
    a.finish_all(&mut out);
    let _ = out.flush();
}

/// How a published boundary snapshot was freed ([`Event::Drop`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mechanism {
    #[default]
    Other,
    /// Structural retire: its single retained continuation was published.
    Retire,
    /// The LRU eviction unit (boundary + the blocks only it kept attachable), device or host.
    Lru,
    /// A cold (first-sighting, never attached) boundary taken first.
    Cold,
    /// A whole radix leaf taken by LRU together with its snapshots.
    NodeLru,
    /// FP8 strict publish evicting another publish's orphaned blocks.
    StrictTail,
    /// `X-Session-Id` TTL ran out.
    SessionDemote,
}

/// What made the pool free memory when a boundary was dropped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    /// No memory pressure: a publish, an attach release or a session sweep.
    #[default]
    None,
    /// The cache trim over its soft cap.
    SoftCap,
    /// The cache trim below the free-memory floor.
    MinFree,
    /// A snapshot allocation found the device out of memory.
    Oom,
    /// `VmmKv::relieve`: memory given back to an allocation outside the pool.
    Relieve,
    /// The block store's spill-ahead thread: a spill before any allocation needed the bytes.
    Ahead,
}

/// Why a boundary was not published ([`Event::Refuse`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Refusal {
    /// FP8 strict publish: blocks another request published cover this boundary.
    StrictFp8,
    /// Strict publish could not clear those blocks (a live sequence maps them).
    Hazard,
    /// The sliding rings no longer hold the boundary's window.
    RingWrapped,
    /// The publish failed (allocation, a changed sequence).
    Error,
    /// No publishable row (boundary rounds to 0, rows/tokens out of step).
    NotPublishable,
}

/// Pool gauges at an attach.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Gauges {
    pub snapshot_bytes: u64,
    pub cache_bytes: u64,
    pub host_bytes: u64,
    pub snaps_live: u64,
    /// The pool's last free-memory sample (taken by its pressure trim, no extra driver call)
    /// and that sample's age.
    pub free_bytes: Option<u64>,
    pub free_age_s: Option<f64>,
}

#[derive(Debug)]
pub struct AttachEv {
    pub t: Instant,
    pub slot: u32,
    pub prompt: Vec<u32>,
    /// Whole radix blocks the prompt matched (before picking a boundary).
    pub radix_blocks: u32,
    pub block_rows: u32,
    /// Attached rows (0 = miss) and the boundary attached.
    pub rows: u32,
    pub snap: Option<u64>,
    pub kind: u8,
    /// The boundary was host-resident (restored at this attach).
    pub host: bool,
    pub gauges: Gauges,
}

/// Pool geometry and byte costs, once at bring-up.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Header {
    pub block_rows: u32,
    pub block_bytes: u64,
    pub full_layers: u32,
    pub kvh_full: u32,
    pub hd_full: u32,
    pub elem_full: u32,
    pub full_scale: bool,
    pub slide_layers: u32,
    pub kvh_slide: u32,
    pub hd_slide: u32,
    pub elem_slide: u32,
    pub window: u32,
    pub ring: u64,
    pub max_ctx: u32,
    pub batch: u32,
    pub snap_row_bytes: u64,
    pub cache_cap: u64,
    pub min_free: u64,
    pub host_cap: u64,
    /// Publication rules the shadow replay models: boundaries are multiples of
    /// `boundary_quantum` rows, a prompt end backs off `prompt_end_backoff` rows, FP8 strict
    /// publish, and the shared-prefix end test (`recent_prompts` heads of `recent_rows` rows,
    /// `share_min_rows` past the attach).
    pub boundary_quantum: u32,
    pub prompt_end_backoff: u32,
    pub strict_publish: bool,
    pub share_min_rows: u32,
    pub recent_prompts: u32,
    pub recent_rows: u32,
}

#[derive(Debug)]
pub enum Event {
    Header(Box<Header>),
    Arrive {
        t: Instant,
        fp: u64,
        prompt_len: u32,
        session: Option<u64>,
        tenant: Option<u64>,
    },
    Attach(Box<AttachEv>),
    /// The attach of `slot` was undone (failed host restore or mapping): a miss.
    AttachUnwound {
        slot: u32,
    },
    Publish {
        t: Instant,
        slot: u32,
        id: u64,
        rows: u32,
        kind: u8,
        bytes: u64,
        cold: bool,
    },
    Refuse {
        t: Instant,
        slot: u32,
        rows: u32,
        cause: Refusal,
    },
    Drop {
        t: Instant,
        id: u64,
        rows: u32,
        mechanism: Mechanism,
        trigger: Trigger,
        host: bool,
        attaches: u32,
    },
    /// A published boundary refreshed by a touch (turn end, same-history pair).
    Touch {
        t: Instant,
        id: u64,
    },
    Spill {
        t: Instant,
        id: u64,
    },
    /// Slot `slot`'s sequence ended: every row it computed (prompt + generated).
    Finish {
        t: Instant,
        slot: u32,
        tokens: Vec<u32>,
    },
    /// Write in-flight requests, flush, ack ([`flush`]).
    Flush(SyncSender<bool>),
}

/// `vmm::Kind` bits, as `VmmStats` attributes attach kinds.
const PROMPT_END: u8 = 1;
const SHARED_END: u8 = 2;

fn kind_name(bits: u8) -> &'static str {
    if bits & PROMPT_END != 0 {
        "prompt_end"
    } else if bits & SHARED_END != 0 {
        "shared"
    } else if bits & 4 != 0 {
        "checkpoint"
    } else {
        "block"
    }
}

/// Prefix hash after each token: `h[r]` names `tokens[..r]`, so a boundary at `rows` and the
/// `h16` block chain share one identity.
fn prefix_hashes(tokens: &[u32]) -> Vec<u64> {
    let mut h = Vec::with_capacity(tokens.len() + 1);
    let mut x = 0x243F_6A88_85A3_08D3u64;
    h.push(x);
    for &t in tokens {
        x = mix(x ^ u64::from(t).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        h.push(x);
    }
    h
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Src {
    Prompt,
    Output,
}

struct TNode {
    start: u32,
    len: u32,
    kids: Vec<u32>,
    src: Src,
    rid: u64,
}

/// Compressed token trie over an append-only arena: exact longest-prefix matches against every
/// sequence inserted so far.
struct Trie {
    arena: Vec<u32>,
    nodes: Vec<TNode>,
    resets: u64,
}

impl Trie {
    fn new() -> Self {
        Trie {
            arena: Vec::new(),
            nodes: vec![TNode {
                start: 0,
                len: 0,
                kids: Vec::new(),
                src: Src::Prompt,
                rid: 0,
            }],
            resets: 0,
        }
    }

    fn kid(&self, n: usize, tok: u32) -> Option<usize> {
        self.nodes[n]
            .kids
            .iter()
            .map(|&k| k as usize)
            .find(|&k| self.arena[self.nodes[k].start as usize] == tok)
    }

    /// Longest prefix of `toks` inserted before, with the source and request of its last node.
    fn lcp(&self, toks: &[u32]) -> (usize, Option<(Src, u64)>) {
        let (mut n, mut i, mut last) = (0usize, 0usize, None);
        while i < toks.len() {
            let Some(k) = self.kid(n, toks[i]) else {
                break;
            };
            let node = &self.nodes[k];
            let edge = &self.arena[node.start as usize..(node.start + node.len) as usize];
            let m = edge
                .iter()
                .zip(&toks[i..])
                .take_while(|(a, b)| a == b)
                .count();
            i += m;
            last = Some((node.src, node.rid));
            if m < edge.len() {
                break;
            }
            n = k;
        }
        (i, last)
    }

    /// Split node `k` after `m` edge tokens; `k` keeps the head, a new child the rest.
    fn split(&mut self, k: usize, m: u32) {
        let node = &mut self.nodes[k];
        let tail = TNode {
            start: node.start + m,
            len: node.len - m,
            kids: std::mem::take(&mut node.kids),
            src: node.src,
            rid: node.rid,
        };
        node.len = m;
        let id = self.nodes.len() as u32;
        self.nodes[k].kids.push(id);
        self.nodes.push(tail);
    }

    /// Insert `toks`. A prompt marks every node it covers `Prompt`; `rid` labels them all.
    fn insert(&mut self, toks: &[u32], src: Src, rid: u64) {
        if self.arena.len() + toks.len() > TRIE_CAP_TOKENS {
            let resets = self.resets + 1;
            *self = Trie::new();
            self.resets = resets;
        }
        let (mut n, mut i) = (0usize, 0usize);
        while i < toks.len() {
            let Some(k) = self.kid(n, toks[i]) else {
                break;
            };
            let (start, len) = (self.nodes[k].start as usize, self.nodes[k].len as usize);
            let m = self.arena[start..start + len]
                .iter()
                .zip(&toks[i..])
                .take_while(|(a, b)| a == b)
                .count();
            if m < len {
                self.split(k, m as u32);
            }
            let node = &mut self.nodes[k];
            node.rid = rid;
            if src == Src::Prompt {
                node.src = Src::Prompt;
            }
            i += m;
            n = k;
            if m < len {
                break;
            }
        }
        if i == toks.len() {
            return;
        }
        let rest = &toks[i..];
        let node = &self.nodes[n];
        if n != 0
            && node.kids.is_empty()
            && node.src == src
            && (node.start + node.len) as usize == self.arena.len()
        {
            self.arena.extend_from_slice(rest);
            self.nodes[n].len += rest.len() as u32;
            return;
        }
        let start = self.arena.len() as u32;
        self.arena.extend_from_slice(rest);
        let id = self.nodes.len() as u32;
        self.nodes.push(TNode {
            start,
            len: rest.len() as u32,
            kids: Vec::new(),
            src,
            rid,
        });
        self.nodes[n].kids.push(id);
    }
}

struct Boundary {
    rows: u32,
    kind: u8,
    t_pub: f64,
    rid: u64,
    spilled: Option<f64>,
    dropped: Option<(Mechanism, Trigger, f64, bool, u32)>,
}

struct Arrival {
    t: f64,
    session: Option<u64>,
    tenant: Option<u64>,
}

/// One request in flight on a slot.
struct Pending {
    rec: Record,
    hashes: Vec<u64>,
}

#[derive(Serialize)]
struct Deeper {
    rows: u32,
    /// `live`: published and retained at the attach (yet not attached); `dropped`; `refused`.
    state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    /// `dropped`: the mechanism (`retire`, `lru`, `cold`, `node_lru`, `strict_tail`,
    /// `session_demote`); `refused`: the refusal.
    #[serde(skip_serializing_if = "Option::is_none")]
    cause: Option<String>,
    /// `dropped`: the memory pressure behind it (`none`, `soft_cap`, `min_free`, `oom`, `relieve`).
    #[serde(skip_serializing_if = "Option::is_none")]
    trigger: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    t_pub: Option<f64>,
    /// Seconds before this attach that it was dropped (or refused).
    #[serde(skip_serializing_if = "Option::is_none")]
    ago_s: Option<f64>,
    /// Its lifetime from publish to drop.
    #[serde(skip_serializing_if = "Option::is_none")]
    life_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attaches: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    publisher_rid: Option<u64>,
}

#[derive(Serialize)]
struct Record {
    ev: &'static str,
    rid: u64,
    slot: u32,
    t_arr: Option<f64>,
    t_att: f64,
    t_fin: Option<f64>,
    prompt: u32,
    out: Option<u32>,
    /// Exact token LCP of the prompt with everything computed before it.
    ideal: u32,
    /// `prompt`: an earlier prompt covered the ideal match end; `output`: only an output did.
    ideal_src: Option<&'static str>,
    /// The request whose prompt or output last covered the ideal match end (conversation hint).
    ideal_rid: Option<u64>,
    /// Token LCP with the finished sequences only: rows past it were seen only in requests still
    /// in flight at this attach (their KV may not exist yet).
    ideal_ready: u32,
    /// Rows of whole radix blocks matched.
    radix: u32,
    att: u32,
    kind: &'static str,
    lost: u32,
    lost_vs_ideal: u32,
    host_restore: bool,
    snap_id: Option<u64>,
    /// Seconds from the attached boundary's publish to this attach.
    age_s: Option<f64>,
    deeper: Option<Deeper>,
    #[serde(flatten)]
    gauges: Gauges,
    session: Option<u64>,
    tenant: Option<u64>,
    unwound: bool,
    /// Events the bounded queue had dropped when this record was written: nonzero means the
    /// reconstructed boundary registry may be incomplete from here on.
    dropped_events: u64,
    /// Prefix hash of the computed sequence at every 16th row (`h16[k]` = rows `16(k+1)`).
    h16: Vec<String>,
}

/// The writer thread's state; pure, so tests drive it with events.
pub struct Analyzer {
    epoch: Instant,
    wall: f64,
    /// Every sequence seen: prompts at their attach, prompt + output at their finish.
    trie: Trie,
    /// Finished sequences only (their KV was computed before this attach).
    ready: Trie,
    next_rid: u64,
    slots: FxHashMap<u32, Pending>,
    boundaries: FxHashMap<u64, Boundary>,
    by_chain: FxHashMap<u64, Vec<u64>>,
    refusals: FxHashMap<u64, (Refusal, f64, u32)>,
    arrivals: FxHashMap<u64, VecDeque<Arrival>>,
    arrivals_n: usize,
    records: u64,
    unresolved: u64,
}

impl Analyzer {
    pub fn new(epoch: Instant, wall: f64) -> Self {
        Analyzer {
            epoch,
            wall,
            trie: Trie::new(),
            ready: Trie::new(),
            next_rid: 1,
            slots: FxHashMap::default(),
            boundaries: FxHashMap::default(),
            by_chain: FxHashMap::default(),
            refusals: FxHashMap::default(),
            arrivals: FxHashMap::default(),
            arrivals_n: 0,
            records: 0,
            unresolved: 0,
        }
    }

    fn secs(&self, t: Instant) -> f64 {
        let s = t.saturating_duration_since(self.epoch).as_secs_f64();
        (s * 1e6).round() / 1e6
    }

    fn line(out: &mut impl Write, v: &impl Serialize) {
        if serde_json::to_writer(&mut *out, v).is_ok() {
            let _ = out.write_all(b"\n");
        }
    }

    pub fn handle(&mut self, ev: Event, out: &mut impl Write) {
        match ev {
            Event::Header(h) => {
                let mut v = serde_json::to_value(&*h).unwrap_or_default();
                if let Some(m) = v.as_object_mut() {
                    m.insert("ev".into(), "header".into());
                    m.insert("schema".into(), SCHEMA.into());
                    m.insert("hash".into(), HASH_NAME.into());
                    m.insert("unix_epoch_s".into(), self.wall.into());
                }
                Self::line(out, &v);
            }
            Event::Arrive {
                t,
                fp,
                session,
                tenant,
                ..
            } => {
                if self.arrivals_n >= REGISTRY_CAP {
                    self.arrivals.clear();
                    self.arrivals_n = 0;
                }
                let t = self.secs(t);
                self.arrivals
                    .entry(fp)
                    .or_default()
                    .push_back(Arrival { t, session, tenant });
                self.arrivals_n += 1;
            }
            Event::Attach(a) => self.attach(*a, out),
            Event::AttachUnwound { slot } => {
                if let Some(p) = self.slots.get_mut(&slot) {
                    let r = &mut p.rec;
                    r.unwound = true;
                    r.att = 0;
                    r.kind = "miss";
                    r.lost = r.prompt;
                    r.lost_vs_ideal = r.ideal;
                    r.snap_id = None;
                    r.age_s = None;
                }
            }
            Event::Publish {
                t,
                slot,
                id,
                rows,
                kind,
                bytes,
                cold,
            } => {
                let t = self.secs(t);
                let Some(p) = self.slots.get(&slot) else {
                    self.unresolved += 1;
                    return;
                };
                let Some(&h) = p.hashes.get(rows as usize) else {
                    self.unresolved += 1;
                    return;
                };
                let rid = p.rec.rid;
                Self::line(
                    out,
                    &serde_json::json!({"ev": "pub", "t": t, "id": id, "rid": rid, "rows": rows,
                        "kind": kind, "bytes": bytes, "cold": cold}),
                );
                self.forget_if_full();
                self.boundaries.insert(
                    id,
                    Boundary {
                        rows,
                        kind,
                        t_pub: t,
                        rid,
                        spilled: None,
                        dropped: None,
                    },
                );
                self.by_chain.entry(h).or_default().push(id);
            }
            Event::Refuse {
                t,
                slot,
                rows,
                cause,
            } => {
                let t = self.secs(t);
                let slot = self.slots.get(&slot);
                Self::line(
                    out,
                    &serde_json::json!({"ev": "refuse", "t": t, "rid": slot.map(|p| p.rec.rid),
                        "rows": rows, "cause": cause}),
                );
                match slot.and_then(|p| p.hashes.get(rows as usize)) {
                    Some(&h) => {
                        if self.refusals.len() >= REGISTRY_CAP {
                            self.refusals.clear();
                        }
                        self.refusals.insert(h, (cause, t, rows));
                    }
                    None => self.unresolved += 1,
                }
            }
            Event::Drop {
                t,
                id,
                rows,
                mechanism,
                trigger,
                host,
                attaches,
            } => {
                let t = self.secs(t);
                if let Some(b) = self.boundaries.get_mut(&id) {
                    b.dropped = Some((mechanism, trigger, t, host, attaches));
                }
                Self::line(
                    out,
                    &serde_json::json!({"ev": "drop", "t": t, "id": id, "rows": rows,
                        "mech": mechanism, "trigger": trigger, "host": host, "attaches": attaches}),
                );
            }
            Event::Spill { t, id } => {
                let t = self.secs(t);
                if let Some(b) = self.boundaries.get_mut(&id) {
                    b.spilled = Some(t);
                }
                Self::line(out, &serde_json::json!({"ev": "spill", "t": t, "id": id}));
            }
            Event::Touch { t, id } => {
                let t = self.secs(t);
                Self::line(out, &serde_json::json!({"ev": "touch", "t": t, "id": id}));
            }
            Event::Flush(ack) => {
                self.finish_all(out);
                let _ = ack.send(out.flush().is_ok());
            }
            Event::Finish { t, slot, tokens } => {
                let t = self.secs(t);
                let Some(mut p) = self.slots.remove(&slot) else {
                    return;
                };
                let rid = p.rec.rid;
                p.rec.t_fin = Some(t);
                p.rec.out = Some((tokens.len() as u32).saturating_sub(p.rec.prompt));
                self.trie.insert(&tokens, Src::Output, rid);
                self.ready.insert(&tokens, Src::Output, rid);
                p.rec.h16 = h16(&tokens);
                self.emit_record(p.rec, out);
            }
        }
    }

    fn forget_if_full(&mut self) {
        if self.boundaries.len() >= REGISTRY_CAP {
            let mut ids: Vec<u64> = self.boundaries.keys().copied().collect();
            ids.sort_unstable();
            for id in &ids[..ids.len() / 2] {
                self.boundaries.remove(id);
            }
            let live = &self.boundaries;
            self.by_chain.retain(|_, v| {
                v.retain(|id| live.contains_key(id));
                !v.is_empty()
            });
            self.refusals.clear();
        }
    }

    fn attach(&mut self, a: AttachEv, out: &mut impl Write) {
        // The previous request on this slot never reported its finish.
        if let Some(p) = self.slots.remove(&a.slot) {
            let mut rec = p.rec;
            rec.h16 = h16_from(&p.hashes);
            self.emit_record(rec, out);
        }
        let t_att = self.secs(a.t);
        let rid = self.next_rid;
        self.next_rid += 1;
        let prompt = &a.prompt;
        let mut arrival = None;
        let fp = fingerprint(prompt);
        if let Some(q) = self.arrivals.get_mut(&fp) {
            // An identical prompt rejected or cancelled before its attach leaves its arrival
            // behind: skip arrivals older than any queueing delay.
            while let Some(x) = q.pop_front() {
                self.arrivals_n -= 1;
                if t_att - x.t <= ARRIVAL_STALE_S {
                    arrival = Some(x);
                    break;
                }
            }
            if q.is_empty() {
                self.arrivals.remove(&fp);
            }
        }
        let (ideal, src) = self.trie.lcp(prompt);
        let ideal_ready = self.ready.lcp(prompt).0 as u32;
        self.trie.insert(prompt, Src::Prompt, rid);
        let hashes = prefix_hashes(prompt);
        let plen = prompt.len() as u32;
        let ideal = ideal as u32;
        let att = a.rows;
        let deeper = self.deeper(&hashes, att, ideal.min(plen.saturating_sub(1)), t_att);
        let age_s = a
            .snap
            .and_then(|id| self.boundaries.get(&id))
            .map(|b| round3(t_att - b.t_pub));
        let rec = Record {
            ev: "req",
            rid,
            slot: a.slot,
            t_arr: arrival.as_ref().map(|x| x.t),
            t_att,
            t_fin: None,
            prompt: plen,
            out: None,
            ideal,
            ideal_src: src.map(|(s, _)| match s {
                Src::Prompt => "prompt",
                Src::Output => "output",
            }),
            ideal_rid: src.map(|(_, r)| r),
            ideal_ready,
            radix: a.radix_blocks * a.block_rows,
            att,
            kind: if a.snap.is_some() {
                kind_name(a.kind)
            } else {
                "miss"
            },
            lost: plen - att.min(plen),
            lost_vs_ideal: ideal.saturating_sub(att),
            host_restore: a.host,
            snap_id: a.snap,
            age_s,
            deeper,
            gauges: a.gauges,
            session: arrival.as_ref().and_then(|x| x.session),
            tenant: arrival.as_ref().and_then(|x| x.tenant),
            unwound: false,
            dropped_events: 0,
            h16: Vec::new(),
        };
        self.slots.insert(a.slot, Pending { rec, hashes });
    }

    /// The deepest boundary in `(att, upto]` on this prompt's chain that was ever published,
    /// else the deepest refused one.
    fn deeper(&self, hashes: &[u64], att: u32, upto: u32, now: f64) -> Option<Deeper> {
        let rows = (att + 1..=upto).rev();
        for r in rows.clone() {
            let Some(b) = self
                .by_chain
                .get(&hashes[r as usize])
                .and_then(|ids| ids.last())
                .and_then(|id| self.boundaries.get(id))
            else {
                continue;
            };
            let (state, cause, trigger, ago, life, host, attaches) = match b.dropped {
                Some((mech, trigger, t, host, attaches)) => (
                    "dropped",
                    Some(format!("{mech:?}")),
                    Some(format!("{trigger:?}")),
                    Some(round3(now - t)),
                    Some(round3(t - b.t_pub)),
                    Some(host),
                    Some(attaches),
                ),
                None => (
                    "live",
                    None,
                    None,
                    None,
                    None,
                    b.spilled.map(|_| true),
                    None,
                ),
            };
            return Some(Deeper {
                rows: b.rows,
                state,
                kind: Some(kind_name(b.kind)),
                cause: cause.map(snake),
                trigger: trigger.map(snake),
                t_pub: Some(b.t_pub),
                ago_s: ago,
                life_s: life,
                host,
                attaches,
                publisher_rid: Some(b.rid),
            });
        }
        for r in rows {
            if let Some(&(cause, t, rows)) = self.refusals.get(&hashes[r as usize]) {
                return Some(Deeper {
                    rows,
                    state: "refused",
                    kind: None,
                    cause: Some(snake(format!("{cause:?}"))),
                    trigger: None,
                    t_pub: None,
                    ago_s: Some(round3(now - t)),
                    life_s: None,
                    host: None,
                    attaches: None,
                    publisher_rid: None,
                });
            }
        }
        None
    }

    fn emit_record(&mut self, mut rec: Record, out: &mut impl Write) {
        rec.dropped_events = DROPPED.load(Ordering::Relaxed);
        Self::line(out, &rec);
        self.records += 1;
        if self.records % 256 == 0 {
            let _ = out.flush();
        }
    }

    fn stats(&self, out: &mut impl Write, dropped: u64) {
        Self::line(
            out,
            &serde_json::json!({
                "ev": "stats",
                "t": self.secs(Instant::now()),
                "records": self.records,
                "dropped_events": dropped,
                "unresolved": self.unresolved,
                "trie_tokens": self.trie.arena.len(),
                "trie_nodes": self.trie.nodes.len(),
                "trie_resets": self.trie.resets,
                "boundaries": self.boundaries.len(),
            }),
        );
    }

    /// Write every request still in flight (shutdown).
    pub fn finish_all(&mut self, out: &mut impl Write) {
        let mut slots: Vec<u32> = self.slots.keys().copied().collect();
        slots.sort_unstable();
        for s in slots {
            if let Some(p) = self.slots.remove(&s) {
                let mut rec = p.rec;
                rec.h16 = h16_from(&p.hashes);
                self.emit_record(rec, out);
            }
        }
    }
}

fn round3(x: f64) -> f64 {
    (x * 1e3).round() / 1e3
}

fn snake(s: String) -> String {
    let mut o = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                o.push('_');
            }
            o.push(c.to_ascii_lowercase());
        } else {
            o.push(c);
        }
    }
    o
}

fn h16_from(hashes: &[u64]) -> Vec<String> {
    hashes
        .iter()
        .skip(16)
        .step_by(16)
        .map(|h| format!("{h:016x}"))
        .collect()
}

fn h16(tokens: &[u32]) -> Vec<String> {
    h16_from(&prefix_hashes(tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(a: &mut Analyzer, evs: Vec<Event>) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        for ev in evs {
            a.handle(ev, &mut out);
        }
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .filter(|v| v["ev"] == "req")
            .collect()
    }

    fn attach(t: Instant, slot: u32, prompt: Vec<u32>, rows: u32, snap: Option<u64>) -> Event {
        Event::Attach(Box::new(AttachEv {
            t,
            slot,
            prompt,
            radix_blocks: 0,
            block_rows: 4096,
            rows,
            snap,
            kind: if snap.is_some() { PROMPT_END } else { 0 },
            host: false,
            gauges: Gauges::default(),
        }))
    }

    #[test]
    fn trie_matches_exact_prefixes_of_prompts_and_outputs() {
        let mut t = Trie::new();
        assert_eq!(t.lcp(&[1, 2, 3]).0, 0);
        t.insert(&[1, 2, 3, 4, 5], Src::Prompt, 1);
        assert_eq!(t.lcp(&[1, 2, 3, 9]), (3, Some((Src::Prompt, 1))));
        t.insert(&[1, 2, 3, 4, 5, 6, 7], Src::Output, 1);
        assert_eq!(t.lcp(&[1, 2, 3, 4, 5, 6, 8]), (6, Some((Src::Output, 1))));
        t.insert(&[1, 2, 7], Src::Prompt, 2);
        assert_eq!(t.lcp(&[1, 2, 7, 7]), (3, Some((Src::Prompt, 2))));
        assert_eq!(t.lcp(&[1, 2, 3, 4, 5, 6, 7, 1]).0, 7);
        assert_eq!(t.lcp(&[1, 2]).0, 2);
        // A prompt over output-only rows marks them seen in a prompt.
        t.insert(&[1, 2, 3, 4, 5, 6], Src::Prompt, 3);
        assert_eq!(t.lcp(&[1, 2, 3, 4, 5, 6]), (6, Some((Src::Prompt, 3))));
        assert_eq!(t.lcp(&[1, 2, 3, 4, 5, 6, 7]).1, Some((Src::Output, 1)));
        assert_eq!(t.lcp(&[2]).0, 0);
    }

    #[test]
    fn prefix_hashes_name_prefixes() {
        let a = prefix_hashes(&[5, 6, 7, 8]);
        let b = prefix_hashes(&[5, 6, 9]);
        assert_eq!(a[2], b[2]);
        assert_ne!(a[3], b[3]);
        assert_eq!(a.len(), 5);
        let seq: Vec<u32> = (0..40).collect();
        let h = h16(&seq);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0], format!("{:016x}", prefix_hashes(&seq[..16])[16]));
    }

    #[test]
    fn record_attributes_the_deeper_boundary_and_its_drop() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let mut a = Analyzer::new(t0, 0.0);
        let sys: Vec<u32> = (0..64).collect();
        let mut p1 = sys.clone();
        p1.extend(100..200);
        let mut full1 = p1.clone();
        full1.extend(500..510);
        let mut p2 = full1.clone();
        p2.extend(600..650);
        let recs = run(
            &mut a,
            vec![
                Event::Arrive {
                    t: at(0),
                    fp: fingerprint(&p1),
                    prompt_len: p1.len() as u32,
                    session: Some(7),
                    tenant: None,
                },
                attach(at(1), 3, p1.clone(), 0, None),
                Event::Publish {
                    t: at(2),
                    slot: 3,
                    id: 11,
                    rows: 64,
                    kind: SHARED_END,
                    bytes: 100,
                    cold: false,
                },
                Event::Publish {
                    t: at(2),
                    slot: 3,
                    id: 12,
                    rows: 128,
                    kind: PROMPT_END,
                    bytes: 100,
                    cold: false,
                },
                Event::Refuse {
                    t: at(2),
                    slot: 3,
                    rows: 160,
                    cause: Refusal::StrictFp8,
                },
                Event::Finish {
                    t: at(3),
                    slot: 3,
                    tokens: full1.clone(),
                },
                Event::Drop {
                    t: at(5),
                    id: 12,
                    rows: 128,
                    mechanism: Mechanism::Lru,
                    trigger: Trigger::MinFree,
                    host: false,
                    attaches: 0,
                },
                attach(at(9), 4, p2.clone(), 64, Some(11)),
                Event::Finish {
                    t: at(10),
                    slot: 4,
                    tokens: p2.clone(),
                },
            ],
        );
        assert_eq!(recs.len(), 2);
        let r1 = &recs[0];
        assert_eq!(r1["t_arr"], 0.0);
        assert_eq!(r1["session"], 7);
        assert_eq!(r1["ideal"], 0);
        assert_eq!(r1["kind"], "miss");
        assert_eq!(r1["out"], 10);
        assert_eq!(r1["h16"].as_array().unwrap().len(), full1.len() / 16);
        let r2 = &recs[1];
        assert_eq!(r2["ideal"], full1.len());
        assert_eq!(r2["ideal_src"], "output");
        assert_eq!(r2["ideal_rid"], r1["rid"]);
        assert_eq!(r2["att"], 64);
        assert_eq!(r2["kind"], "prompt_end");
        assert_eq!(r2["age_s"], 7.0);
        assert_eq!(r2["lost"], p2.len() - 64);
        let d = &r2["deeper"];
        assert_eq!(d["rows"], 128);
        assert_eq!(d["state"], "dropped");
        assert_eq!(d["cause"], "lru");
        assert_eq!(d["trigger"], "min_free");
        assert_eq!(r2["ideal_ready"], full1.len());
        assert_eq!(d["ago_s"], 4.0);
        assert_eq!(d["life_s"], 3.0);
    }

    #[test]
    fn in_flight_history_is_seen_but_not_ready_and_arrivals_are_consumed() {
        let t0 = Instant::now();
        let mut a = Analyzer::new(t0, 0.0);
        let p: Vec<u32> = (0..300).collect();
        let arrive = |s: u64| Event::Arrive {
            t: t0 + Duration::from_secs(s),
            fp: fingerprint(&p),
            prompt_len: 300,
            session: None,
            tenant: Some(s),
        };
        let recs = run(
            &mut a,
            vec![
                arrive(0),
                arrive(1),
                attach(t0 + Duration::from_secs(2), 0, p.clone(), 0, None),
                attach(t0 + Duration::from_secs(3), 1, p.clone(), 0, None),
                Event::Finish {
                    t: t0 + Duration::from_secs(4),
                    slot: 1,
                    tokens: p.clone(),
                },
                Event::Finish {
                    t: t0 + Duration::from_secs(5),
                    slot: 0,
                    tokens: p.clone(),
                },
            ],
        );
        assert_eq!(recs.len(), 2);
        let (first, second) = (&recs[1], &recs[0]);
        assert_eq!(
            (first["tenant"].clone(), second["tenant"].clone()),
            (0.into(), 1.into())
        );
        assert_eq!(first["ideal"], 0);
        assert_eq!(second["ideal"], 300);
        assert_eq!(second["ideal_ready"], 0, "the twin was still in flight");
        assert!(a.arrivals.is_empty());
        assert_eq!(second["dropped_events"], 0);
    }

    #[test]
    fn refusal_reported_when_nothing_deeper_was_published() {
        let t0 = Instant::now();
        let mut a = Analyzer::new(t0, 0.0);
        let p1: Vec<u32> = (0..200).collect();
        let mut p2 = p1.clone();
        p2.extend(900..950);
        let recs = run(
            &mut a,
            vec![
                attach(t0, 0, p1.clone(), 0, None),
                Event::Refuse {
                    t: t0,
                    slot: 0,
                    rows: 160,
                    cause: Refusal::RingWrapped,
                },
                Event::Finish {
                    t: t0,
                    slot: 0,
                    tokens: p1,
                },
                attach(t0, 0, p2, 0, None),
                Event::AttachUnwound { slot: 0 },
            ],
        );
        assert_eq!(recs.len(), 1);
        let mut out = Vec::new();
        a.finish_all(&mut out);
        let r: serde_json::Value =
            serde_json::from_str(String::from_utf8(out).unwrap().trim()).unwrap();
        assert_eq!(r["ideal"], 200);
        assert_eq!(r["ideal_src"], "prompt");
        assert_eq!(r["deeper"]["state"], "refused");
        assert_eq!(r["deeper"]["cause"], "ring_wrapped");
        assert_eq!(r["unwound"], true);
        assert!(r["out"].is_null());
    }
}
