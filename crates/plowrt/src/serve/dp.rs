//! Data-parallel ranks of one model inside one serve.
//!
//! `--dp N` loads N full copies of a TP1 model, one per device group, each with its own engine,
//! dispatcher and prefix cache, registered as instance keys `slug#r`. The registry keeps one entry
//! per model; this module maps a request for the model to one rank.
//!
//! The router runs on the request path, so it reads only relaxed atomics (each rank's dispatcher
//! publishes its queue, slots, capacity and KV use in its own `Metrics`), never waits on a lock
//! (session shards are short critical sections; a prefix probe is `try_lock` and skipped when
//! contended) and hashes a prompt at most once — the chosen rank's admission and attach reuse
//! the [`PrefixKey`].
//!
//! Rule order: the session's rank (unless it is past the spill depth), then the rank whose
//! prefix cache holds the longest prefix (unless its load exceeds the least-loaded rank's by more
//! than the share of the prompt it saves, scaled by `PLOW_ROUTE_PREFIX_SLACK`), then the least
//! loaded rank with rotating ties.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use crossbeam_utils::CachePadded;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;

use crate::memory::vmm::{PrefixKey, PrefixProbe};
use crate::obs::Metrics;

/// Ranks per model; also the width of a retry's exclusion mask.
pub const MAX_DP: usize = 32;

/// The instance key of `model`'s rank `rank`.
pub fn rank_key(model: &str, rank: usize) -> String {
    format!("{model}#{rank}")
}

/// `(model, rank)` of an instance key; `None` for a plain slug.
pub fn split_key(key: &str) -> Option<(&str, usize)> {
    let (model, rank) = key.rsplit_once('#')?;
    if model.is_empty() || rank.is_empty() || !rank.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((model, rank.parse().ok()?))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteCfg {
    /// Probe ranks' prefix caches (`PLOW_ROUTE_PREFIX`).
    pub prefix: bool,
    /// Queue depth past which affinity yields (`PLOW_ROUTE_SPILL`); `None` = max(4, slots/4).
    pub spill: Option<u32>,
    /// `PLOW_ROUTE_PREFIX_SLACK`.
    pub slack: f32,
}

impl Default for RouteCfg {
    fn default() -> Self {
        RouteCfg { prefix: true, spill: None, slack: 0.5 }
    }
}

impl RouteCfg {
    pub fn from_config() -> Self {
        let c = crate::config::RuntimeConfig::get();
        RouteCfg { prefix: c.route_prefix, spill: c.route_spill, slack: c.route_prefix_slack as f32 }
    }
}

/// Why a rank was chosen (`plowrt_dp_route_decisions_total{reason}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Session = 0,
    Prefix = 1,
    LeastLoaded = 2,
    /// The session's rank was over the spill depth or down.
    Spill = 3,
    /// A resubmission after the first rank closed.
    Retry = 4,
}

impl Reason {
    pub const ALL: [Reason; 5] = [Reason::Session, Reason::Prefix, Reason::LeastLoaded, Reason::Spill, Reason::Retry];

    pub fn as_str(self) -> &'static str {
        match self {
            Reason::Session => "session",
            Reason::Prefix => "prefix",
            Reason::LeastLoaded => "least_loaded",
            Reason::Spill => "spill",
            Reason::Retry => "retry",
        }
    }
}

/// One rank as the router sees it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cand {
    pub up: bool,
    pub pending: u32,
    pub active: u32,
    pub capacity: u32,
    pub kv_milli: u32,
    /// Prompt rows its prefix cache would restore (0 = not probed or a miss).
    pub cached: u32,
    /// Sessions pinned here; set only when placing a new session, which then also balances on it.
    pub sessions: u32,
}

/// Slot occupancy, queue included, plus a steep penalty once KV is past 80% committed.
pub fn load(c: &Cand) -> f32 {
    let kv = c.kv_milli as f32 / 1000.0;
    (c.pending + c.active) as f32 / c.capacity.max(1) as f32 + (kv - 0.8).max(0.0) * 4.0
}

/// Load (fraction of slots) a session's rank must exceed the least-loaded rank by to spill.
const SPILL_MARGIN: f32 = 0.5;
/// KV commitment (per mille) past which a rank takes no new session while another has room.
const KV_FULL_MILLI: u32 = 900;

fn spill(cfg: &RouteCfg, c: &Cand) -> u32 {
    cfg.spill.unwrap_or_else(|| (c.capacity / 4).max(4))
}

/// The rank for one request, and why. Pure: `cands[i].up == false` excludes rank `i`, `sticky`
/// is the session's rank, `rr` rotates ties.
pub fn choose(
    cands: &[Cand],
    sticky: Option<usize>,
    prompt_rows: usize,
    block_rows: u32,
    cfg: &RouteCfg,
    rr: usize,
) -> Option<(usize, Reason)> {
    let n = cands.len();
    let mut least: Option<(usize, f32)> = None;
    for k in 0..n {
        let i = (rr + k) % n;
        if !cands[i].up {
            continue;
        }
        let c = &cands[i];
        let l = load(c) + c.sessions as f32 / c.capacity.max(1) as f32;
        if least.is_none_or(|(_, b)| l < b) {
            least = Some((i, l));
        }
    }
    let (least, _) = least?;
    let l_min = load(&cands[least]);
    // A session leaves its rank (and the rows retained there) only when that rank is both deep
    // and clearly busier than the least-loaded one: a burst of the session's own turns is not it.
    if let Some(c) = sticky.and_then(|i| cands.get(i)).filter(|c| c.up) {
        if c.pending <= spill(cfg, c) || load(c) <= l_min + SPILL_MARGIN {
            return Some((sticky.unwrap(), Reason::Session));
        }
    }
    if cfg.prefix && block_rows > 0 && prompt_rows >= block_rows as usize {
        let mut hit: Option<usize> = None;
        for (i, c) in cands.iter().enumerate() {
            let saved = c.cached as f32 / prompt_rows as f32;
            if !c.up || c.cached < block_rows || load(c) > l_min + SPILL_MARGIN + cfg.slack * saved {
                continue;
            }
            let better = hit.is_none_or(|h| {
                let b = &cands[h];
                c.cached > b.cached || c.cached == b.cached && load(c) < load(b)
            });
            if better {
                hit = Some(i);
            }
        }
        if let Some(h) = hit {
            return Some((h, Reason::Prefix));
        }
    }
    Some((least, if sticky.is_some() { Reason::Spill } else { Reason::LeastLoaded }))
}

const SHARDS: usize = 64;
const SHARD_CAP: usize = 4096;
/// Granularity of the router's own prompt chain (independent of the caches' block size: a
/// 2048-row KV block says nothing about a 1.5k-token conversation).
const TAIL_ROWS: usize = 128;
const TAIL_STRIDE: usize = 4;
/// Chain entries kept on the stack: prompts past 64k tokens chain their first 64k.
const TAIL_CHAIN: usize = 512;
/// Chain entries, from a prompt's end, looked up among recent prompt tails.
const TAIL_WALK: usize = 64;

/// Chained hashes of `prompt`'s whole `TAIL_ROWS` blocks into `out`; returns how many. Every
/// `TAIL_STRIDE`th token and each block's last are hashed: a collision only misplaces affinity
/// (the cache itself compares tokens), and the chain stays a small fraction of the cache's own hash.
fn tail_chain(prompt: &[u32], out: &mut [u64; TAIL_CHAIN]) -> usize {
    use std::hash::Hasher;
    let mut prev = 0x9e37_79b9_7f4a_7c15u64;
    let mut n = 0;
    for chunk in prompt.chunks_exact(TAIL_ROWS).take(TAIL_CHAIN) {
        let mut h = rustc_hash::FxHasher::default();
        h.write_u64(prev);
        for pair in chunk.chunks_exact(2 * TAIL_STRIDE) {
            h.write_u64(u64::from(pair[0]) | u64::from(pair[TAIL_STRIDE]) << 32);
        }
        h.write_u32(chunk[TAIL_ROWS - 1]);
        prev = h.finish();
        out[n] = prev;
        n += 1;
    }
    n
}
/// A session idle this long loses its rank.
const SESSION_TTL_S: u32 = 900;
/// A session's timestamp is rewritten at most this often.
const SESSION_REFRESH_S: u32 = 30;

/// Session id → rank, sharded so concurrent requests rarely meet on one lock. Expiry is swept per
/// shard, only when an insert finds that shard full.
struct Sessions {
    shards: Box<[CachePadded<Mutex<FxHashMap<u64, (u8, u32)>>>]>,
    /// Live entries per rank.
    counts: Box<[CachePadded<AtomicU32>]>,
    epoch: Instant,
}

impl Sessions {
    fn new() -> Self {
        Sessions {
            shards: (0..SHARDS).map(|_| CachePadded::new(Mutex::new(FxHashMap::default()))).collect(),
            counts: (0..MAX_DP).map(|_| CachePadded::new(AtomicU32::new(0))).collect(),
            epoch: Instant::now(),
        }
    }

    fn hash(id: &str) -> u64 {
        use std::hash::Hasher;
        let mut h = rustc_hash::FxHasher::default();
        h.write(id.as_bytes());
        h.finish()
    }

    fn now(&self) -> u32 {
        self.epoch.elapsed().as_secs() as u32
    }

    fn shard(&self, h: u64) -> &Mutex<FxHashMap<u64, (u8, u32)>> {
        &self.shards[(h >> 58) as usize % SHARDS]
    }

    fn count(&self, rank: usize) -> u32 {
        self.counts[rank].load(Relaxed)
    }

    fn forget(&self, rank: u8) {
        if let Some(c) = self.counts.get(rank as usize) {
            c.fetch_sub(1, Relaxed);
        }
    }

    fn get(&self, h: u64, now: u32) -> Option<(usize, u32)> {
        let mut s = self.shard(h).lock();
        let &(r, t) = s.get(&h)?;
        if now.wrapping_sub(t) >= SESSION_TTL_S {
            s.remove(&h);
            self.forget(r);
            return None;
        }
        Some((r as usize, t))
    }

    /// A new session's rank: fewest sessions, ties to the lower load. Queue depth does not exclude a
    /// rank: a burst of first turns skews it for a moment while the session stays for its whole
    /// life (an overloaded rank sheds sessions by spill later); only KV near full does. Claimed by
    /// CAS, so concurrent first turns cannot all read the same counts and pile onto one rank. Pass
    /// the result to [`Self::put`] as `reserved`.
    fn reserve(&self, cands: &[Cand]) -> Option<usize> {
        let roomy = cands.iter().any(|c| c.up && c.kv_milli < KV_FULL_MILLI);
        loop {
            let mut best: Option<(usize, u32, f32)> = None;
            for (i, c) in cands.iter().enumerate() {
                let l = load(c);
                if !c.up || roomy && c.kv_milli >= KV_FULL_MILLI {
                    continue;
                }
                let s = self.count(i);
                if best.is_none_or(|(_, bs, bl)| s < bs || s == bs && l < bl) {
                    best = Some((i, s, l));
                }
            }
            let (i, s, _) = best?;
            if self.counts[i].compare_exchange_weak(s, s + 1, Relaxed, Relaxed).is_ok() {
                return Some(i);
            }
        }
    }

    /// `reserved`: `rank`'s count already holds this session ([`Self::reserve`]).
    fn put(&self, h: u64, rank: usize, now: u32, reserved: bool) {
        let mut s = self.shard(h).lock();
        if s.len() >= SHARD_CAP && !s.contains_key(&h) {
            s.retain(|_, &mut (r, t)| {
                let keep = now.wrapping_sub(t) < SESSION_TTL_S;
                if !keep {
                    self.forget(r);
                }
                keep
            });
            if s.len() >= SHARD_CAP {
                for (_, (r, _)) in s.drain() {
                    self.forget(r);
                }
            }
        }
        match s.insert(h, (rank as u8, now)) {
            Some((old, _)) if old as usize == rank => {
                if reserved {
                    self.forget(old);
                }
                return;
            }
            Some((old, _)) => self.forget(old),
            None => {}
        }
        if !reserved {
            self.counts[rank].fetch_add(1, Relaxed);
        }
    }

    fn len(&self) -> usize {
        self.shards.iter().map(|s| s.lock().len()).sum()
    }

    /// [`Self::get`] for the router's tail walk: `None` also when the shard is busy, never waits.
    fn peek(&self, h: u64, now: u32) -> Option<usize> {
        let s = self.shard(h).try_lock()?;
        s.get(&h).filter(|&&(_, t)| now.wrapping_sub(t) < SESSION_TTL_S).map(|&(r, _)| r as usize)
    }

    /// A prompt tail routed to `rank`. A tail that reaches two ranks is a shared prefix (a system
    /// prompt every rank holds) and stops pointing anywhere: following it would herd unrelated
    /// conversations onto whichever rank routed it last. Uncounted.
    fn put_tail(&self, h: u64, rank: usize, now: u32) {
        let mut s = self.shard(h).lock();
        if s.len() >= SHARD_CAP && !s.contains_key(&h) {
            s.retain(|_, &mut (_, t)| now.wrapping_sub(t) < SESSION_TTL_S);
            if s.len() >= SHARD_CAP {
                s.clear();
            }
        }
        let e = s.entry(h).or_insert((rank as u8, now));
        if e.0 as usize != rank {
            e.0 = SHARED_TAIL;
        }
        e.1 = now;
    }
}

/// [`Sessions::put_tail`]'s mark for a tail seen on more than one rank.
const SHARED_TAIL: u8 = u8::MAX;

/// One rank of a DP model.
pub struct DpRank {
    pub key: String,
    pub rank: usize,
    /// First device ordinal of its group.
    pub ordinal: u32,
    pub group: usize,
    /// Dispatcher installed and residency admits.
    up: CachePadded<AtomicBool>,
    /// Requests routed here and not yet submitted ([`Pick`]): a burst routing at once sees them.
    picked: CachePadded<AtomicU32>,
    /// The rank's dispatcher metrics: the load signal.
    pub metrics: Arc<Metrics>,
    /// Its engine's prefix cache, while one is resident.
    probe: RwLock<Option<PrefixProbe>>,
    /// Host CPUs on its GPU's socket: the rank's dispatcher thread runs there (empty: unpinned).
    pub cpus: Vec<usize>,
}

impl DpRank {
    pub fn cand(&self) -> Cand {
        let m = &self.metrics;
        Cand {
            up: self.up.load(Relaxed) && !m.engine_dead.load(Relaxed),
            pending: m.queued_requests.load(Relaxed) as u32 + self.picked.load(Relaxed),
            active: m.slots_active.load(Relaxed) as u32,
            capacity: m.slots_capacity.load(Relaxed) as u32,
            kv_milli: m.kv_used_milli.load(Relaxed) as u32,
            cached: 0,
            sessions: 0,
        }
    }

    pub fn is_up(&self) -> bool {
        self.up.load(Relaxed)
    }

    pub fn set_up(&self, up: bool) {
        self.up.store(up, Relaxed);
    }
}

/// Router counters for one DP model.
#[derive(Default)]
pub struct DpStats {
    pub decisions: [AtomicU64; 5],
    pub retries: AtomicU64,
    pub probes: AtomicU64,
    pub probe_contended: AtomicU64,
    /// Route wall time, log2 ns buckets.
    pub route_ns: [AtomicU64; 32],
}

impl DpStats {
    /// Upper bound (ns) of the bucket holding quantile `q` of route times.
    pub fn route_ns_quantile(&self, q: f64) -> u64 {
        let counts: Vec<u64> = self.route_ns.iter().map(|b| b.load(Relaxed)).collect();
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return 0;
        }
        let want = (total as f64 * q).ceil() as u64;
        let mut seen = 0;
        for (i, c) in counts.iter().enumerate() {
            seen += c;
            if seen >= want {
                return 1u64 << i;
            }
        }
        u64::MAX
    }
}

/// A routing decision. Its rank counts the request as pending until the `Routed` is dropped:
/// hold it until the job is submitted.
pub struct Routed<'a> {
    pub rank: usize,
    pub reason: Reason,
    /// The prompt's prefix hashes when the router computed them: hand them to the job.
    pub key: Option<PrefixKey>,
    pub pick: Pick<'a>,
}

/// One routed request's hold on its rank's pending count.
pub struct Pick<'a>(&'a AtomicU32);

impl Drop for Pick<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Relaxed);
    }
}

thread_local! {
    static RR: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// Seeds each thread's tie rotation apart, so threads routing at once do not all break a tie
/// toward the same rank.
static RR_SEED: AtomicU32 = AtomicU32::new(0);

/// Every rank of one DP model.
pub struct DpSet {
    pub model: String,
    pub ranks: Vec<DpRank>,
    sessions: Sessions,
    /// Last full-block hash of each prompt routed with a prefix key → its rank: the prompt's next
    /// turn finds its rank here before that rank has published the rows to its prefix cache.
    tails: Sessions,
    /// Block rows of the ranks' prefix caches (0 until one is resident).
    block_rows: AtomicU32,
    pub stats: DpStats,
}

impl DpSet {
    /// `ranks[r] = (ordinal, group, metrics)`.
    pub fn new(model: &str, ranks: Vec<(u32, usize, Arc<Metrics>)>) -> Self {
        assert!(!ranks.is_empty() && ranks.len() <= MAX_DP, "DP rank count out of range");
        DpSet {
            model: model.to_string(),
            ranks: ranks
                .into_iter()
                .enumerate()
                .map(|(rank, (ordinal, group, metrics))| DpRank {
                    key: rank_key(model, rank),
                    rank,
                    ordinal,
                    group,
                    up: CachePadded::new(AtomicBool::new(false)),
                    picked: CachePadded::new(AtomicU32::new(0)),
                    metrics,
                    probe: RwLock::new(None),
                    cpus: Vec::new(),
                })
                .collect(),
            sessions: Sessions::new(),
            tails: Sessions::new(),
            block_rows: AtomicU32::new(0),
            stats: DpStats::default(),
        }
    }

    pub fn set_probe(&self, rank: usize, probe: Option<PrefixProbe>) {
        if let Some(p) = &probe {
            self.block_rows.store(p.block_rows(), Relaxed);
        }
        *self.ranks[rank].probe.write() = probe;
    }

    pub fn sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Pick a rank. `exclude` masks ranks that already refused this request (a retry);
    /// `prompt` enables the prefix probe.
    pub fn route(
        &self,
        cfg: &RouteCfg,
        session: Option<&str>,
        prompt: Option<&[u32]>,
        exclude: u32,
    ) -> Option<Routed<'_>> {
        let t0 = Instant::now();
        let n = self.ranks.len();
        let mut cands = [Cand::default(); MAX_DP];
        for (i, r) in self.ranks.iter().enumerate() {
            cands[i] = r.cand();
            if exclude >> i & 1 != 0 {
                cands[i].up = false;
            }
        }
        let retry = exclude != 0;
        let session = session.map(|s| (Sessions::hash(s), self.sessions.now()));
        let stored = session.and_then(|(h, now)| self.sessions.get(h, now));
        let sticky = stored.map(|(r, _)| r).filter(|&r| r < n);
        let sticky_ok = sticky.is_some_and(|i| cands[i].up && cands[i].pending <= spill(cfg, &cands[i]));
        if session.is_some() && !sticky_ok {
            for (i, c) in cands[..n].iter_mut().enumerate() {
                c.sessions = self.sessions.count(i);
            }
        }

        let mut key = None;
        let mut block_rows = 0;
        let mut tail = None;
        let mut followed = false;
        if cfg.prefix && n > 1 && !retry && !sticky_ok {
            if let Some(p) = prompt.filter(|p| p.len() >= TAIL_ROWS) {
                let mut chain = [0u64; TAIL_CHAIN];
                let blocks = tail_chain(p, &mut chain);
                let now = self.tails.now();
                let mut shared = false;
                for i in (0..blocks).rev().take(TAIL_WALK) {
                    if let Some(r) = self.tails.peek(chain[i], now).filter(|&r| r < n && cands[r].up) {
                        // Ending exactly on a routed tail adds nothing to it: a prefix many
                        // prompts share (a system prompt), not a conversation going on.
                        shared = i == blocks - 1;
                        if !shared {
                            cands[r].cached = ((i + 1) * TAIL_ROWS) as u32;
                            followed = true;
                        }
                        break;
                    }
                }
                tail = Some((chain[blocks - 1], now, shared));
                block_rows = TAIL_ROWS as u32;
            }
            let br = self.block_rows.load(Relaxed);
            // The caches are probed only for a prompt that follows no routed one: a prefix shared
            // across conversations (a document, a long system prompt) published there.
            if let Some(p) = prompt.filter(|p| !followed && br > 0 && p.len() >= br as usize) {
                let k = PrefixKey::new(p, br);
                for i in 0..n {
                    if !cands[i].up {
                        continue;
                    }
                    let Some(guard) = self.ranks[i].probe.try_read() else {
                        self.stats.probe_contended.fetch_add(1, Relaxed);
                        continue;
                    };
                    let Some(probe) = guard.as_ref() else { continue };
                    self.stats.probes.fetch_add(1, Relaxed);
                    match probe.try_matched_rows(&k, p) {
                        Some(rows) => {
                            cands[i].cached = cands[i].cached.max(rows);
                            // Nothing can beat a rank that holds the whole prompt.
                            if rows as usize + br as usize > p.len() {
                                break;
                            }
                        }
                        None => {
                            self.stats.probe_contended.fetch_add(1, Relaxed);
                        }
                    }
                }
                key = Some(k);
            }
        }
        let rr = RR.with(|c| {
            let v = match c.get() {
                usize::MAX => RR_SEED.fetch_add(7, Relaxed) as usize,
                v => v.wrapping_add(1),
            };
            c.set(v);
            v
        });
        let prompt_rows = prompt.map_or(0, <[u32]>::len);
        let (mut rank, reason) = choose(&cands[..n], sticky, prompt_rows, block_rows, cfg, rr)?;
        let mut reserved = false;
        if session.is_some() && sticky.is_none() && reason == Reason::LeastLoaded {
            if let Some(r) = self.sessions.reserve(&cands[..n]) {
                (rank, reserved) = (r, true);
            }
        }
        let reason = if retry { Reason::Retry } else { reason };
        if let Some((last, now, shared)) = tail {
            self.tails.put_tail(last, if shared { SHARED_TAIL as usize } else { rank }, now);
        }
        if let Some((h, now)) = session {
            let fresh = stored.is_some_and(|(r, t)| r == rank && now.wrapping_sub(t) < SESSION_REFRESH_S);
            if !fresh {
                self.sessions.put(h, rank, now, reserved);
            }
        }
        self.stats.decisions[reason as usize].fetch_add(1, Relaxed);
        if retry {
            self.stats.retries.fetch_add(1, Relaxed);
        }
        let ns = t0.elapsed().as_nanos() as u64;
        self.stats.route_ns[(64 - ns.max(1).leading_zeros() as usize).min(31)].fetch_add(1, Relaxed);
        let picked = &self.ranks[rank].picked;
        picked.fetch_add(1, Relaxed);
        Some(Routed { rank, reason, key, pick: Pick(picked) })
    }
}

/// Every DP model of this serve. Built once at startup, before any rank loads.
#[derive(Default)]
pub struct DpRouter {
    pub cfg: RouteCfg,
    sets: FxHashMap<String, Arc<DpSet>>,
    by_key: FxHashMap<String, (Arc<DpSet>, usize)>,
}

impl DpRouter {
    pub fn new(cfg: RouteCfg) -> Self {
        DpRouter { cfg, ..Default::default() }
    }

    pub fn add(&mut self, set: DpSet) {
        let set = Arc::new(set);
        for r in &set.ranks {
            self.by_key.insert(r.key.clone(), (Arc::clone(&set), r.rank));
        }
        self.sets.insert(set.model.clone(), set);
    }

    pub fn set(&self, model: &str) -> Option<&Arc<DpSet>> {
        self.sets.get(model)
    }

    /// The set and rank an instance key belongs to.
    pub fn rank(&self, key: &str) -> Option<(&Arc<DpSet>, usize)> {
        self.by_key.get(key).map(|(s, r)| (s, *r))
    }

    pub fn sets(&self) -> impl Iterator<Item = &Arc<DpSet>> {
        self.sets.values()
    }

    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(pending: u32, active: u32, capacity: u32) -> Cand {
        Cand { up: true, pending, active, capacity, ..Default::default() }
    }

    #[test]
    fn keys_round_trip_and_plain_slugs_are_not_keys() {
        assert_eq!(rank_key("gemma", 3), "gemma#3");
        assert_eq!(split_key("gemma#3"), Some(("gemma", 3)));
        assert_eq!(split_key("org/gemma#12"), Some(("org/gemma", 12)));
        for plain in ["gemma", "#3", "gemma#", "gemma#x", "a#1b"] {
            assert_eq!(split_key(plain), None, "{plain}");
        }
    }

    #[test]
    fn least_loaded_wins_and_ties_rotate() {
        let cfg = RouteCfg::default();
        let cands = [c(2, 8, 16), c(0, 3, 16), c(1, 8, 16)];
        assert_eq!(choose(&cands, None, 0, 0, &cfg, 0), Some((1, Reason::LeastLoaded)));
        let even = [c(0, 0, 16); 4];
        let picks: Vec<usize> = (0..4).map(|rr| choose(&even, None, 0, 0, &cfg, rr).unwrap().0).collect();
        assert_eq!(picks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn down_ranks_are_never_chosen() {
        let cfg = RouteCfg::default();
        let mut cands = [c(0, 0, 16), c(9, 9, 16)];
        cands[0].up = false;
        assert_eq!(choose(&cands, Some(0), 0, 0, &cfg, 0), Some((1, Reason::Spill)));
        cands[1].up = false;
        assert_eq!(choose(&cands, None, 0, 0, &cfg, 0), None);
    }

    #[test]
    fn a_session_sticks_until_its_rank_is_past_the_spill_depth() {
        let cfg = RouteCfg::default();
        let cands = [c(4, 16, 16), c(0, 0, 16)];
        assert_eq!(choose(&cands, Some(0), 0, 0, &cfg, 0), Some((0, Reason::Session)));
        let deep = [c(5, 16, 16), c(0, 0, 16)];
        assert_eq!(choose(&deep, Some(0), 0, 0, &cfg, 0), Some((1, Reason::Spill)));
        let pinned = RouteCfg { spill: Some(64), ..cfg };
        assert_eq!(choose(&deep, Some(0), 0, 0, &pinned, 0), Some((0, Reason::Session)));
    }

    #[test]
    fn a_prefix_hit_wins_within_the_slack_it_buys() {
        let cfg = RouteCfg::default();
        let mut cands = [c(0, 2, 16), c(0, 4, 16)];
        cands[1].cached = 1024;
        // 4/16 vs 2/16: 0.125 extra load, within the margin (0.5) plus the slack the hit buys (0.25).
        assert_eq!(choose(&cands, None, 2048, 32, &cfg, 0), Some((1, Reason::Prefix)));
        cands[1].active = 16;
        assert_eq!(choose(&cands, None, 2048, 32, &cfg, 0), Some((0, Reason::LeastLoaded)));
        let off = RouteCfg { prefix: false, ..cfg };
        cands[1].active = 4;
        assert_eq!(choose(&cands, None, 2048, 32, &off, 0), Some((0, Reason::LeastLoaded)));
        // Less than one block is not a hit.
        cands[1].cached = 16;
        assert_eq!(choose(&cands, None, 2048, 32, &cfg, 0), Some((0, Reason::LeastLoaded)));
    }

    #[test]
    fn the_longest_prefix_wins_unless_its_rank_is_far_busier() {
        let cfg = RouteCfg::default();
        let mut cands = [c(0, 0, 16), c(0, 0, 16), c(16, 0, 16)];
        cands[0].cached = 64;
        cands[1].cached = 512;
        cands[2].cached = 1024;
        assert_eq!(choose(&cands, None, 1100, 32, &cfg, 0), Some((1, Reason::Prefix)));
    }

    #[test]
    fn kv_pressure_past_eighty_percent_dominates_load() {
        let mut full = c(0, 1, 16);
        full.kv_milli = 950;
        assert!(load(&full) > load(&c(0, 8, 16)));
        let mut fine = c(0, 1, 16);
        fine.kv_milli = 700;
        assert_eq!(load(&fine), load(&c(0, 1, 16)));
    }

    fn set(n: usize) -> DpSet {
        DpSet::new("m", (0..n).map(|r| (r as u32, r, Arc::new(Metrics::default()))).collect())
    }

    fn up(set: &DpSet) {
        for r in &set.ranks {
            r.metrics.slots_capacity.store(16, Relaxed);
            r.set_up(true);
        }
    }

    #[test]
    fn sessions_stick_then_expire() {
        let s = set(4);
        up(&s);
        let cfg = RouteCfg::default();
        let first = s.route(&cfg, Some("abc"), None, 0).unwrap();
        s.ranks[first.rank].metrics.slots_active.store(3, Relaxed);
        for _ in 0..8 {
            let again = s.route(&cfg, Some("abc"), None, 0).unwrap();
            assert_eq!((again.rank, again.reason), (first.rank, Reason::Session));
        }
        assert_eq!(s.sessions(), 1);
        let h = Sessions::hash("abc");
        let stale = s.sessions.now() + SESSION_TTL_S;
        assert_eq!(s.sessions.get(h, stale), None, "an idle session forgets its rank");
    }

    /// New sessions spread by how many each rank already holds, not only by the instant's load:
    /// a burst of first turns would otherwise pin long-lived sessions wherever load looked lowest.
    #[test]
    fn new_sessions_balance_on_pinned_sessions() {
        let s = set(8);
        up(&s);
        let cfg = RouteCfg::default();
        s.ranks[3].metrics.slots_active.store(2, Relaxed);
        s.ranks[5].metrics.slots_active.store(12, Relaxed);
        s.ranks[6].metrics.kv_used_milli.store(950, Relaxed);
        for i in 0..64 {
            s.route(&cfg, Some(&format!("sess-{i}")), None, 0).unwrap();
        }
        let counts: Vec<u32> = (0..8).map(|r| s.sessions.count(r)).collect();
        assert_eq!(counts.iter().sum::<u32>(), 64);
        assert_eq!(counts[6], 0, "a rank with KV near full takes none: {counts:?}");
        assert!(counts[5] >= 9, "a momentary queue does not unbalance sessions: {counts:?}");
        let live = counts.iter().enumerate().filter(|&(r, _)| r != 6).map(|(_, &c)| c);
        assert!(live.clone().max().unwrap() - live.min().unwrap() <= 1, "{counts:?}");
        let h = Sessions::hash("sess-0");
        let before = counts.iter().sum::<u32>();
        assert!(s.sessions.get(h, s.sessions.now() + SESSION_TTL_S).is_none());
        assert_eq!((0..8).map(|r| s.sessions.count(r)).sum::<u32>(), before - 1, "expiry uncounts");
    }

    /// First turns placing concurrently each claim a distinct share: no rank ends more than one
    /// session ahead.
    #[test]
    fn a_concurrent_burst_of_new_sessions_spreads_evenly() {
        let s = set(8);
        up(&s);
        let cfg = RouteCfg::default();
        std::thread::scope(|t| {
            for w in 0..16 {
                let (s, cfg) = (&s, &cfg);
                t.spawn(move || {
                    for i in 0..32 {
                        let r = s.route(cfg, Some(&format!("w{w}-{i}")), None, 0).unwrap();
                        assert_eq!(r.reason, Reason::LeastLoaded);
                    }
                });
            }
        });
        let counts: Vec<u32> = (0..8).map(|r| s.sessions.count(r)).collect();
        assert_eq!(counts, vec![64; 8]);
        assert_eq!(s.sessions(), 512);
    }

    #[test]
    fn a_full_session_shard_sweeps_expired_entries_only() {
        let s = Sessions::new();
        let shard_of = |h: u64| (h >> 58) as usize % SHARDS;
        let mut same: Vec<u64> = (0u64..).map(|i| i << 58 | i).filter(|&h| shard_of(h) == 0).take(SHARD_CAP + 1).collect();
        let last = same.pop().unwrap();
        for &h in &same[..SHARD_CAP / 2] {
            s.put(h, 1, 0, false);
        }
        for &h in &same[SHARD_CAP / 2..] {
            s.put(h, 1, SESSION_TTL_S + 10, false);
        }
        s.put(last, 2, SESSION_TTL_S + 10, false);
        assert_eq!(s.shards[0].lock().len(), SHARD_CAP / 2 + 1);
        assert_eq!(s.get(last, SESSION_TTL_S + 10), Some((2, SESSION_TTL_S + 10)));
    }

    #[test]
    fn a_retry_excludes_the_rank_that_refused() {
        let s = set(2);
        up(&s);
        let cfg = RouteCfg::default();
        let r = s.route(&cfg, None, None, 0b01).unwrap();
        assert_eq!((r.rank, r.reason), (1, Reason::Retry));
        assert!(s.route(&cfg, None, None, 0b11).is_none());
        assert_eq!(s.stats.retries.load(Relaxed), 1, "a refused route is not a retry");
    }

    /// A prompt that extends one routed before goes to that prompt's rank, even before the rank's
    /// cache could show it.
    #[test]
    fn a_prompt_extending_a_routed_one_follows_it() {
        let s = set(4);
        up(&s);
        let cfg = RouteCfg::default();
        let first: Vec<u32> = (0..300).collect();
        let rank = s.route(&cfg, None, Some(&first), 0).unwrap().rank;
        s.ranks[rank].metrics.slots_active.store(4, Relaxed);
        let next: Vec<u32> = (0..700).collect();
        let r1 = s.route(&cfg, None, Some(&next), 0).unwrap();
        assert_eq!((r1.rank, r1.reason), (rank, Reason::Prefix));
        let other: Vec<u32> = (1000..1700).collect();
        assert_eq!(s.route(&cfg, None, Some(&other), 0).unwrap().reason, Reason::LeastLoaded);
    }

    /// A tail two ranks routed (a shared system prompt) attracts nothing.
    #[test]
    fn a_shared_prefix_does_not_herd() {
        let s = set(2);
        up(&s);
        let shared: Vec<u32> = (0..256).collect();
        s.tails.put_tail(tail_hash(&shared), 0, s.tails.now());
        s.tails.put_tail(tail_hash(&shared), 1, s.tails.now());
        let next: Vec<u32> = (0..300).collect();
        assert_eq!(s.route(&RouteCfg::default(), None, Some(&next), 0).unwrap().reason, Reason::LeastLoaded);
    }

    /// First turns that end inside a common system prompt do not pile onto the first one's rank.
    #[test]
    fn prompts_ending_in_a_common_prefix_spread() {
        let s = set(4);
        up(&s);
        let cfg = RouteCfg::default();
        for k in 0..8u32 {
            let mut p: Vec<u32> = (0..256).collect();
            p.extend((0..50).map(|j| 10_000 + k * 100 + j));
            let r = s.route(&cfg, None, Some(&p), 0).unwrap();
            assert_eq!(r.reason, Reason::LeastLoaded, "turn {k}");
            s.ranks[r.rank].metrics.slots_active.fetch_add(1, Relaxed);
        }
        assert!(s.ranks.iter().all(|r| r.metrics.slots_active.load(Relaxed) == 2));
    }

    fn tail_hash(p: &[u32]) -> u64 {
        let mut chain = [0u64; TAIL_CHAIN];
        let n = tail_chain(p, &mut chain);
        chain[n - 1]
    }

    #[test]
    fn no_probe_without_a_resident_cache() {
        let s = set(2);
        up(&s);
        let r = s.route(&RouteCfg::default(), None, Some(&[1; 4096]), 0).unwrap();
        assert!(r.key.is_none(), "no block size known yet: nothing hashed");
        assert_eq!(s.stats.probes.load(Relaxed), 0);
    }

    #[test]
    fn route_quantile_reads_the_log2_buckets() {
        let st = DpStats::default();
        st.route_ns[8].fetch_add(99, Relaxed);
        st.route_ns[12].fetch_add(1, Relaxed);
        assert_eq!(st.route_ns_quantile(0.5), 256);
        assert_eq!(st.route_ns_quantile(1.0), 4096);
    }
}
