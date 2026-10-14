//! Closed-loop agentic-serving simulation of the VMM prefix cache (`cargo test -p plowrt
//! --features cuda,hsa --lib --release vmm_sim -- --ignored --nocapture`): the real [`VmmKv`]
//! on a byte-accounting driver, driven the way the CUDA engine drives it (admission, prompt-end
//! publish, turn-end publish and retire). Env: `SIM_CELLS` (comma-separated prompt files of
//! `session turn prompt_tokens` lines, run back to back on one pool), `SIM_KV` (fp8|bf16),
//! `SIM_SLOTS`, `SIM_FREE_MIB` (free after load), `SIM_FLOOR_MIB`, `SIM_CHUNK` / `SIM_PF_TPS`
//! (prefill launch rows / rows per second), `SIM_TTL_S`, `SIM_GAP_MS` (client round trip between
//! turns), `SIM_MARGIN` (admission rows past prompt + output), `SIM_GROW` (map output rows as decode
//! reaches them, preempting on failure), `SIM_RING_DECODE` (ring rows a slot keeps after its
//! prefill: 1024 = paged sliding rings), and `SIM_SCALE` with `SIM_UNMAP_US` / `SIM_RELEASE_US` to
//! pace the reclaimer thread in real time instead of draining it per call. `SIM_SESSION=1` sends
//! `X-Session-Id` on every request (production sends none); `SIM_POLICY_LEGACY=1` runs the 0.1.2
//! flag/tier policy. The scripted cells below (`cell_*`) drive the same engine logic over a small
//! pool and run as ordinary tests.
use super::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex as StdMutex;

struct SimVmm {
    cap: u64,
    used: AtomicU64,
    next: AtomicU64,
    sizes: StdMutex<HashMap<u64, u64>>,
    lag_us: [u64; 2],
    gran: u64,
}

fn nap(us: u64) {
    if us > 0 {
        std::thread::sleep(std::time::Duration::from_micros(us));
    }
}

impl SimVmm {
    fn take(&self, bytes: u64) -> Result<u64> {
        let used = self.used.fetch_add(bytes, Ordering::SeqCst) + bytes;
        if used > self.cap {
            self.used.fetch_sub(bytes, Ordering::SeqCst);
            return Err(RuntimeError::Oom("sim OOM".into()));
        }
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        self.sizes.lock().unwrap().insert(id, bytes);
        Ok(id)
    }
    fn give(&self, id: u64) {
        if let Some(bytes) = self.sizes.lock().unwrap().remove(&id) {
            self.used.fetch_sub(bytes, Ordering::SeqCst);
        }
    }
}

impl VmmOps for SimVmm {
    fn granularity(&self) -> Result<u64> {
        Ok(self.gran)
    }
    fn reserve(&self, bytes: u64) -> Result<u64> {
        Ok(self
            .next
            .fetch_add(bytes.div_ceil(1 << 21) + 1, Ordering::SeqCst)
            << 21)
    }
    fn address_free(&self, _va: u64, _bytes: u64) {}
    fn create(&self, bytes: u64) -> Result<u64> {
        self.take(bytes)
    }
    fn release(&self, handle: u64) {
        nap(self.lag_us[1]);
        self.give(handle)
    }
    fn map(&self, _va: u64, _bytes: u64, _handle: u64) -> Result<()> {
        Ok(())
    }
    fn unmap(&self, _va: u64, _bytes: u64) {
        nap(self.lag_us[0]);
    }
    fn set_access(&self, _va: u64, _bytes: u64) -> Result<()> {
        Ok(())
    }
    fn alloc(&self, bytes: u64) -> Result<u64> {
        self.take(bytes)
    }
    fn free(&self, va: u64) {
        self.give(va)
    }
    fn copy_dtod(&self, _dst: u64, _src: u64, _bytes: u64) -> Result<()> {
        Ok(())
    }
    fn free_bytes(&self) -> Option<u64> {
        Some(self.cap - self.used.load(Ordering::SeqCst).min(self.cap))
    }
    // Host tier: pinned chunks are ids, copies are instant, fences retire at once.
    fn host_tier_alloc(&self, _bytes: u64) -> Result<u64> {
        Ok(self.next.fetch_add(1, Ordering::SeqCst) << 40)
    }
    fn host_tier_free(&self, _ptr: u64, _bytes: u64) {}
    fn copy_dtoh_async(&self, _host_ptr: u64, _va: u64, _bytes: u64) -> Result<u64> {
        Ok(self.next.fetch_add(1, Ordering::SeqCst))
    }
    fn fence_done(&self, _fence: u64) -> Result<bool> {
        Ok(true)
    }
    fn fence_wait(&self, _fence: u64) -> Result<()> {
        Ok(())
    }
    fn copy_htod(&self, _va: u64, _host_ptr: u64, _bytes: u64) -> Result<()> {
        Ok(())
    }
}

struct Kv {
    kv: VmmKv,
    ring: u32,
    /// Ring rows a slot keeps once its prefill is done (`SIM_RING_DECODE`): a paged slot drops to
    /// its window page, so a decode-time boundary is publishable only at its own frontier.
    ring_decode: std::cell::Cell<u32>,
    window: u32,
    snap_row: u64,
    full_row: u64,
    scale_row: u64,
    /// Turn-end re-publishes that had to copy (the prompt end was evicted while decoding) / all.
    republish: std::cell::Cell<(u64, u64)>,
    /// `SIM_SESSION=1`: every request carries `X-Session-Id` (production sends none).
    session: bool,
}

impl Kv {
    fn snap_bytes(&self, p_a: u32) -> u64 {
        let br = self.kv.block_rows();
        self.snap_row * u64::from(self.window.min(p_a))
            + self.scale_row * u64::from(p_a)
            + self.full_row * u64::from(p_a % br)
    }

    /// `exec/gpu/prefix.rs::publish_boundary`.
    fn boundary(&self, b: usize, toks: &[u32], p_a: u32, kind: Kind) -> bool {
        self.boundary_copied(b, toks, p_a, kind).0
    }

    fn boundary_copied(&self, b: usize, toks: &[u32], p_a: u32, kind: Kind) -> (bool, bool) {
        let copied = std::cell::Cell::new(false);
        let rows = toks.len() as u32;
        if rows == 0 || p_a == 0 {
            return (false, false);
        }
        if self.kv.touch(b, toks, p_a) {
            return (true, false);
        }
        let ring = if rows > self.kv.prompt_rows(b) {
            self.ring_decode.get()
        } else {
            self.ring
        };
        if rows - p_a > ring - self.window {
            return (false, false);
        }
        if !self.kv.resolve_prefix_hazard(b, toks, p_a) {
            return (false, false);
        }
        let ok = self
            .kv
            .publish_kind(b, toks, p_a, kind, self.snap_bytes(p_a), |_| {
                copied.set(true);
                Ok(())
            });
        (ok.is_ok(), copied.get())
    }

    /// `exec/gpu/prefix.rs::vmm_publish_shared`.
    fn publish_shared(&self, b: usize, toks: &[u32], c0: usize) {
        let share = self.kv.share_rows(b) as usize;
        if share > c0 && share <= toks.len() {
            self.boundary(b, toks, share as u32, Kind::SHARED_END);
        }
    }

    /// `exec/gpu/prefix.rs::vmm_publish`.
    fn publish(&self, b: usize, toks: &[u32], max_rows: u32, ttl: std::time::Duration) {
        let rows = toks.len() as u32;
        let prompt = self.kv.prompt_rows(b);
        let legacy = self.kv.policy() == Policy::Legacy;
        let session = self.session && legacy;
        let slack = env_u64("SIM_PE_SLACK", 8) as u32;
        let prompt_end = (rows == prompt && max_rows < rows) || (rows + 1 == prompt && max_rows == rows);
        let max_rows = if prompt_end {
            prompt.saturating_sub(1 + slack)
        } else {
            max_rows
        };
        let p_a = rows.min(max_rows) / 32 * 32;
        if rows == 0 || p_a == 0 {
            return;
        }
        if session {
            self.kv.note_session(b, toks);
        }
        let step = self.kv.block_rows();
        let mut p = step;
        while p < p_a {
            if self.kv.checkpoint_awaited(toks, p) {
                self.boundary(b, toks, p, Kind::CHECKPOINT);
            }
            p += step;
        }
        if !prompt_end && rows < prompt && !self.kv.checkpoint_awaited(toks, p_a) {
            return;
        }
        let turn_end = rows > prompt;
        let published = if turn_end && !session {
            false
        } else {
            let kind = if turn_end {
                Kind::TURN_END
            } else if prompt_end {
                Kind::PROMPT_END
            } else {
                Kind::CHECKPOINT
            };
            self.boundary(b, toks, p_a, kind)
        };
        if published && prompt_end && legacy {
            if session {
                self.kv
                    .pin_prefix(&toks[..p_a as usize], std::time::Instant::now() + ttl);
            }
            self.kv.retire_superseded(toks, p_a);
        }
        let end = prompt.saturating_sub(1 + slack) / 32 * 32;
        if turn_end && end > 0 && end < p_a {
            let (_, copied) = self.boundary_copied(b, toks, end, Kind::PROMPT_END);
            let (c, n) = self.republish.get();
            self.republish.set((c + u64::from(copied), n + 1));
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    Queued,
    Prefill,
    Decode,
    Done,
}

struct Sess {
    prompts: Vec<usize>,
    turn: usize,
    phase: Phase,
    slot: usize,
    prompt: Vec<u32>,
    toks: Vec<u32>,
    decoded: usize,
    admitted: u64,
    /// Output tokens kept across a preemption (`SIM_GROW`): the re-admitted prompt carries them.
    resumed: Option<usize>,
    /// The turn's own prompt length while `prompt` carries preempted output.
    base_len: Option<usize>,
    /// Sim time this turn joined the queue.
    queued_at: f64,
}

fn tokens_for(
    cell: u32,
    s: usize,
    t: usize,
    prev: &[u32],
    prev_len: usize,
    len: usize,
) -> Vec<u32> {
    let salt = cell * 130_000_000;
    let mut p: Vec<u32> = if t == 0 {
        (0..1568u32).map(|i| salt + 10 + i).collect()
    } else {
        prev[..prev_len - env_u64("SIM_RERENDER", 4) as usize].to_vec()
    };
    let base = salt + (s as u32 + 1) * 1_000_000 + t as u32 * 20_000;
    let mut i = 0;
    while p.len() < len {
        p.push(base + i);
        i += 1;
    }
    p.truncate(len);
    p
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore]
fn vmm_sim_agentic() {
    let fp8 = std::env::var("SIM_KV").map_or(true, |v| v != "bf16");
    let slots = env_u64("SIM_SLOTS", if fp8 { 128 } else { 64 }) as usize;
    let free = env_u64("SIM_FREE_MIB", if fp8 { 24844 } else { 25780 }) << 20;
    let floor = env_u64("SIM_FLOOR_MIB", 1621) << 20;
    let chunk = env_u64("SIM_CHUNK", 4096) as usize;
    let pf_tps = env_u64("SIM_PF_TPS", 30000) as f64;
    let Ok(cells) = std::env::var("SIM_CELLS") else {
        eprintln!("vmm_sim_agentic: set SIM_CELLS");
        return;
    };
    let cells: Vec<String> = cells.split(',').map(str::to_string).collect();
    let elem = if fp8 { 1 } else { 2 };
    let geo = VmmGeometry {
        full_layers: (0..8).map(|i| i * 6 + 5).collect(),
        kvh_full: 1,
        hd_full: 512,
        slide_layers: (0..48).filter(|l| l % 6 != 5).collect(),
        kvh_slide: 8,
        hd_slide: 256,
        window: 1024,
        elem,
        elem_slide: elem,
        max_ctx: 16384,
        batch: slots as u32,
    };
    let ops = Arc::new(SimVmm {
        cap: free,
        used: AtomicU64::new(0),
        next: AtomicU64::new(1),
        sizes: StdMutex::new(HashMap::new()),
        lag_us: [env_u64("SIM_UNMAP_US", 0), env_u64("SIM_RELEASE_US", 0)],
        gran: 2 << 20,
    });
    // Real-time pacing (`SIM_SCALE` > 0): the reclaimer thread runs against virtual time instead
    // of being drained after every engine call.
    let scale = std::env::var("SIM_SCALE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let sync = |kv: &VmmKv| {
        if scale == 0.0 {
            kv.sync_reclaim();
        }
    };
    let mut kv = VmmKv::new(ops.clone(), geo, 2 << 20, 0).expect("pool");
    kv.enable_block_pool(kv_pool_cap());
    kv.enable_deferred_reclaim();
    kv.enable_stale_reserve();
    kv.enable_shared_publish();
    if fp8 {
        kv.enable_strict_publish();
    }
    kv.enable_pressure_eviction(floor);
    if env_u64("SIM_POLICY_LEGACY", 0) == 1 {
        kv.set_policy(Policy::Legacy);
    }
    let k = Kv {
        kv,
        session: env_u64("SIM_SESSION", 0) == 1,
        ring: 2048,
        ring_decode: std::cell::Cell::new(env_u64("SIM_RING_DECODE", 2048) as u32),
        window: 1024,
        snap_row: 40 * 2 * 8 * 256 * elem as u64,
        full_row: 16 * 512 * elem as u64,
        scale_row: if fp8 { 64 } else { 0 },
        republish: std::cell::Cell::new((0, 0)),
    };
    let ttl = std::time::Duration::from_secs(env_u64("SIM_TTL_S", 60));
    // `GpuEngine::admission_rows` on a masked-padding packet: one decode quantum (K = 8) + 1.
    let margin = env_u64("SIM_MARGIN", 9) as usize;
    // `SIM_GROW=1`: admission maps the prompt plus `margin` only; decode maps each next row and a
    // failed map preempts that sequence (vLLM recompute: drop its rows, requeue prompt + output).
    let grow = env_u64("SIM_GROW", 0) == 1;
    let mut preempted = 0u64;
    let mut waits = 0u64;
    // `SIM_ORDER=1`: admit the oldest queued turn whose own history is cached (more than
    // `SIM_ORDER_ROWS` attachable rows) before FIFO, unless the queue head has waited `SIM_AGE_S`.
    let order = env_u64("SIM_ORDER", 0) == 1;
    let order_rows = env_u64("SIM_ORDER_ROWS", 2048) as u32;
    let age = env_u64("SIM_AGE_S", 30) as f64;
    let order_max = env_u64("SIM_ORDER_MAX", 0) == 1;
    for (cell, path) in cells.iter().enumerate() {
        let mut per: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for line in std::fs::read_to_string(path).expect("prompt file").lines() {
            let f: Vec<usize> = line
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            per.entry(f[0]).or_default().push((f[1], f[2]));
        }
        let mut ids: Vec<usize> = per.keys().copied().collect();
        ids.sort_unstable();
        let mut sess: Vec<Sess> = ids
            .iter()
            .map(|id| {
                let mut v = per[id].clone();
                v.sort_unstable();
                Sess {
                    prompts: v.into_iter().map(|(_, p)| p).collect(),
                    turn: 0,
                    phase: Phase::Queued,
                    slot: usize::MAX,
                    prompt: Vec::new(),
                    toks: Vec::new(),
                    decoded: 0,
                    admitted: 0,
                    resumed: None,
                    base_len: None,
                    queued_at: 0.0,
                }
            })
            .collect();
        for (i, s) in sess.iter_mut().enumerate() {
            s.prompt = tokens_for(cell as u32, i, 0, &[], 0, s.prompts[0]);
        }
        let mut queue: VecDeque<usize> = (0..sess.len()).collect();
        // Client round trip between a turn's end and the next turn's arrival.
        let gap = env_u64("SIM_GAP_MS", 0) as f64 / 1e3;
        let mut arriving: VecDeque<(f64, usize)> = VecDeque::new();
        let mut free_slots: Vec<usize> = (0..slots).rev().collect();
        let mut now = 0f64;
        let mut seqno = 0u64;
        let turns = sess[0].prompts.len();
        let mut hit = vec![(0u64, 0u64, 0u64); turns];
        let mut last_prefill = false;
        let stats0 = k.kv.stats();
        let mut queue_wait: Vec<f64> = Vec::new();
        loop {
            while arriving.front().is_some_and(|&(t, _)| t <= now) {
                let id = arriving.pop_front().unwrap().1;
                sess[id].queued_at = now;
                queue.push_back(id);
            }
            while let (Some(&head), Some(&b)) = (queue.front(), free_slots.last()) {
                let pick = if order && now - sess[head].queued_at < age {
                    if order_max {
                        // The waiting turn that saves the most prefill, if it beats the head by a block.
                        let rows: Vec<u32> = queue
                            .iter()
                            .map(|&i| k.kv.cached_rows(&sess[i].prompt))
                            .collect();
                        let best = (0..rows.len())
                            .max_by_key(|&i| (rows[i], std::cmp::Reverse(i)))
                            .unwrap();
                        if rows[best] >= rows[0] + k.kv.block_rows() {
                            best
                        } else {
                            0
                        }
                    } else {
                        queue
                            .iter()
                            .position(|&i| k.kv.cached_rows(&sess[i].prompt) > order_rows)
                            .unwrap_or(0)
                    }
                } else {
                    0
                };
                let id = queue.remove(pick).unwrap();
                free_slots.pop();
                let s = &mut sess[id];
                k.kv.begin_seq(b);
                let mut rows = 0;
                if let Some(a) = k.kv.try_attach(b, &s.prompt).expect("attach") {
                    rows = a.rows as usize;
                    if a.rows % k.kv.block_rows() != 0 {
                        k.kv.ensure_rows(b, a.rows + 1).expect("attach rows");
                    }
                    k.kv.finish_attach(b);
                }
                for slot in 0..slots {
                    k.kv.ensure_rows(slot, 1).expect("row 0");
                }
                let out = if grow { 0 } else { 128 };
                let total = (s.prompt.len() + out + margin).min(16384);
                // `GpuEngine::admit_packed_slot`: an admission that cannot map its rows waits
                // (backpressure) until a retirement frees pages.
                if k.kv.ensure_rows(b, total as u32).is_err() {
                    k.kv.begin_seq(b);
                    sync(&k.kv);
                    free_slots.push(b);
                    queue.push_front(id);
                    waits += 1;
                    break;
                }
                sync(&k.kv);
                if s.resumed.is_none() {
                    queue_wait.push(now - s.queued_at);
                    let h = &mut hit[s.turn];
                    h.0 += rows as u64;
                    h.1 += s.prompt.len() as u64;
                    h.2 += 1;
                }
                s.toks = s.prompt[..rows].to_vec();
                s.slot = b;
                s.phase = Phase::Prefill;
                seqno += 1;
                s.admitted = seqno;
            }
            let mut prefilling: Vec<usize> = (0..sess.len())
                .filter(|&i| sess[i].phase == Phase::Prefill)
                .collect();
            let decoding: Vec<usize> = (0..sess.len())
                .filter(|&i| sess[i].phase == Phase::Decode)
                .collect();
            if prefilling.is_empty() && decoding.is_empty() {
                match arriving.front() {
                    Some(&(t, _)) => {
                        now = now.max(t);
                        continue;
                    }
                    None => break,
                }
            }
            if !prefilling.is_empty() && (decoding.is_empty() || !last_prefill) {
                last_prefill = true;
                prefilling.sort_by_key(|&i| sess[i].admitted);
                let mut budget = chunk;
                let mut rows_done = 0;
                for &id in &prefilling {
                    if budget == 0 {
                        break;
                    }
                    let s = &mut sess[id];
                    let c0 = s.toks.len();
                    let n = (s.prompt.len() - c0).min(budget);
                    budget -= n;
                    rows_done += n;
                    s.toks.extend_from_slice(&s.prompt[c0..c0 + n]);
                    let b = s.slot;
                    k.publish_shared(b, &s.toks, c0);
                    if s.toks.len() == s.prompt.len() {
                        k.publish(b, &s.toks, s.toks.len() as u32 - 1, ttl);
                        k.kv.prefill_done(b);
                        s.phase = Phase::Decode;
                        s.decoded = s.resumed.unwrap_or(0);
                    } else {
                        k.publish(b, &s.toks, s.toks.len() as u32, ttl);
                    }
                    sync(&k.kv);
                }
                let dt = rows_done as f64 / pf_tps;
                now += dt;
                nap((dt * scale * 1e6) as u64);
            } else {
                last_prefill = false;
                let dt = 0.025 + 0.00025 * decoding.len() as f64;
                now += dt;
                nap((dt * scale * 1e6) as u64);
                for &id in &decoding {
                    let s = &mut sess[id];
                    if grow && s.decoded + 1 < 128 {
                        let need = (s.toks.len() + 1 + margin).min(16384) as u32;
                        if k.kv.ensure_rows(s.slot, need).is_err() {
                            preempted += 1;
                            k.kv.begin_seq(s.slot);
                            sync(&k.kv);
                            free_slots.push(s.slot);
                            s.resumed = Some(s.decoded);
                            s.base_len.get_or_insert(s.prompt.len());
                            s.prompt = s.toks.clone();
                            s.phase = Phase::Queued;
                            queue.push_front(id);
                            continue;
                        }
                    }
                    s.decoded += 1;
                    if s.decoded < 128 {
                        let t = (s.toks.len() as u32) ^ 0x4000_0000;
                        s.toks.push(t);
                        continue;
                    }
                    s.resumed = None;
                    let b = s.slot;
                    k.publish(b, &s.toks, s.toks.len() as u32, ttl);
                    if k.session {
                        k.kv.pin_prefix(&s.toks, std::time::Instant::now() + ttl);
                    }
                    k.kv.begin_seq(b);
                    sync(&k.kv);
                    free_slots.push(b);
                    s.turn += 1;
                    if s.turn == s.prompts.len() {
                        s.phase = Phase::Done;
                        continue;
                    }
                    let prev_len = s.base_len.take().unwrap_or(s.prompt.len());
                    s.prompt = tokens_for(
                        cell as u32,
                        id,
                        s.turn,
                        &s.prompt,
                        prev_len,
                        s.prompts[s.turn],
                    );
                    s.phase = Phase::Queued;
                    arriving.push_back((now + gap, id));
                }
            }
        }
        let st = k.kv.stats();
        let (a, p, n): (u64, u64, u64) = hit
            .iter()
            .fold((0, 0, 0), |x, h| (x.0 + h.0, x.1 + h.1, x.2 + h.2));
        println!(
            "SIM cell={} sessions={} slots={} cached={:.1}% reqs={} sim_s={:.0} out_tok_s={:.0} snaps_evicted={} units={}",
            path.rsplit('/').next().unwrap(),
            sess.len(),
            slots,
            100.0 * a as f64 / p as f64,
            n,
            now,
            (n * 128) as f64 / now,
            st.snapshots_evicted - stats0.snapshots_evicted,
            st.eviction_units - stats0.eviction_units,
        );
        let per_turn: Vec<String> = hit
            .iter()
            .map(|h| format!("{:.0}", 100.0 * h.0 as f64 / h.1.max(1) as f64))
            .collect();
        println!("SIM   per-turn cached% {}", per_turn.join(" "));
        queue_wait.sort_by(f64::total_cmp);
        let q = |f: f64| {
            queue_wait
                .get(((queue_wait.len() as f64 * f) as usize).min(queue_wait.len().max(1) - 1))
                .copied()
                .unwrap_or(0.0)
        };
        println!(
            "SIM   queue wait s p50={:.1} p99={:.1} max={:.1}",
            q(0.5),
            q(0.99),
            q(1.0)
        );
        let (c, n) = k.republish.replace((0, 0));
        println!("SIM   turn-end re-publish copied {c} of {n} preempted {preempted} admission_waits {waits}");
    }
}

// ---------------------------------------------------------------------------------------------
// Scripted cells: the engine's admission / prefill / turn-end logic over a small pool.

/// One slot's view of a request as the engine keeps it.
struct Rig {
    k: Kv,
    pos: Vec<u32>,
    lookback: u32,
    /// Prefill chunk of [`Rig::turn`]. The rig's ring is 128 rows, so a boundary more than 64
    /// rows behind a chunk end is refused: cells whose shared ends must land prefill in 32s.
    chunk: usize,
}

const RIG_TTL: std::time::Duration = std::time::Duration::from_secs(60);

impl Rig {
    /// `cap`: device bytes; `gran`: VMM granularity (block = gran / 8 rows at hd 4 x 2 B);
    /// `uniform`: no sliding layers (a stub snapshot per boundary).
    fn new(policy: Policy, uniform: bool, cap: u64, gran: u64, batch: u32) -> Rig {
        Self::new_with(policy, uniform, cap, gran, batch, true)
    }

    /// `strict`: FP8 KV's strict publish (`resolve_prefix_hazard`), the production pool's
    /// mode; `false` models a BF16-KV geometry.
    fn new_with(policy: Policy, uniform: bool, cap: u64, gran: u64, batch: u32, strict: bool) -> Rig {
        let geo = VmmGeometry {
            full_layers: vec![3],
            kvh_full: 1,
            hd_full: 4,
            slide_layers: if uniform { vec![] } else { vec![0, 1, 2] },
            kvh_slide: 2,
            hd_slide: 4,
            window: if uniform { 0 } else { 64 },
            elem: 2,
            elem_slide: 2,
            max_ctx: 2048,
            batch,
        };
        let ops = Arc::new(SimVmm {
            cap,
            used: AtomicU64::new(0),
            next: AtomicU64::new(1),
            sizes: StdMutex::new(HashMap::new()),
            lag_us: [0, 0],
            gran,
        });
        let mut kv = VmmKv::new(ops, geo, gran, 0).expect("pool");
        kv.set_policy(policy);
        kv.enable_pressure_eviction(1);
        if strict {
            kv.enable_strict_publish();
        }
        Rig {
            k: Kv {
                kv,
                session: false,
                ring: 128,
                ring_decode: std::cell::Cell::new(128),
                window: 64,
                snap_row: 1,
                full_row: 8,
                scale_row: 0,
                republish: std::cell::Cell::new((0, 0)),
            },
            pos: vec![0; batch as usize],
            lookback: 64,
            chunk: 4096,
        }
    }

    fn kv(&self) -> &VmmKv {
        &self.k.kv
    }

    fn check(&self) {
        let inner = self.k.kv.shared.inner.lock();
        if inner.policy == Policy::Tree {
            policy::check_tree(&inner);
        }
    }

    /// `GpuEngine::admit_packed_slot` without the wait: attach, map, return the attached rows.
    fn admit(&mut self, b: usize, prompt: &[u32]) -> u32 {
        self.k.kv.begin_seq(b);
        let mut rows = 0;
        if let Some(a) = self.k.kv.try_attach(b, prompt).expect("attach") {
            rows = a.rows;
            if a.rows % self.k.kv.block_rows() != 0 {
                self.k.kv.ensure_rows(b, a.rows + 1).expect("rows");
            }
            self.k.kv.finish_attach(b);
        }
        self.k
            .kv
            .ensure_rows(b, prompt.len() as u32 + 8)
            .expect("rows");
        self.pos[b] = rows;
        self.check();
        rows
    }

    /// The packed-admission wait: `(owner, rows)` while another slot's prefill will cover more.
    fn inflight(&self, b: usize, prompt: &[u32]) -> Option<(usize, u32)> {
        self.k.kv.inflight_prefix(b, prompt, self.lookback, &self.pos)
    }

    /// Prefill `prompt` in `b` from the attached rows in chunks of `chunk` rows, publishing as
    /// the token-batch route does; `toks` becomes the slot's row record.
    fn prefill(&mut self, b: usize, prompt: &[u32], chunk: usize) -> Vec<u32> {
        let mut toks = prompt[..self.pos[b] as usize].to_vec();
        while toks.len() < prompt.len() {
            let c0 = toks.len();
            let n = (prompt.len() - c0).min(chunk);
            toks.extend_from_slice(&prompt[c0..c0 + n]);
            self.pos[b] = toks.len() as u32;
            self.k.publish_shared(b, &toks, c0);
            if toks.len() == prompt.len() {
                self.k.publish(b, &toks, toks.len() as u32 - 1, RIG_TTL);
                self.k.kv.prefill_done(b);
            } else {
                self.k.publish(b, &toks, toks.len() as u32, RIG_TTL);
            }
            self.check();
        }
        toks
    }

    /// Decode `reply` rows, then retire the slot as `retire_slot` does (turn-end pass, pin,
    /// `begin_seq`). Returns the reply tokens.
    fn finish(&mut self, b: usize, toks: &mut Vec<u32>, reply: usize) -> Vec<u32> {
        let out: Vec<u32> = (0..reply)
            .map(|i| ((toks.len() + i) as u32) ^ 0x4000_0000)
            .collect();
        toks.extend_from_slice(&out);
        self.pos[b] = toks.len() as u32;
        self.k.publish(b, toks, toks.len() as u32, RIG_TTL);
        if self.k.session {
            self.k
                .kv
                .pin_prefix(toks, std::time::Instant::now() + RIG_TTL);
        }
        self.k.kv.begin_seq(b);
        self.pos[b] = 0;
        self.check();
        out
    }

    /// One whole turn in slot `b`: admit, prefill, decode `reply` rows, retire. Returns
    /// `(attached rows, reply)`.
    fn turn(&mut self, b: usize, prompt: &[u32], reply: usize) -> (u32, Vec<u32>) {
        let rows = self.admit(b, prompt);
        let mut toks = self.prefill(b, prompt, self.chunk);
        let out = self.finish(b, &mut toks, reply);
        (rows, out)
    }

    /// A conversation pair on one system prompt the way production gets a shared end under
    /// strict publish (FP8 KV): `pb` arrives while `pa` is prefilling, waits on it
    /// (`inflight_prefix` names the shared end), `pa` publishes that end from its own blocks
    /// as its prefill passes it, and `pb` attaches it. Both turns complete; returns `pb`'s
    /// attached rows.
    fn shared_pair(&mut self, pa: &[u32], pb: &[u32], reply: usize) -> u32 {
        let a_rows = self.admit(0, pa);
        let first = (a_rows as usize + 32).min(pa.len() - 1);
        self.pos[0] = first as u32;
        let wait = self.inflight(1, pb);
        assert!(wait.is_some_and(|(o, _)| o == 0), "B waits on A: {wait:?}");
        let mut a_toks = self.prefill(0, pa, 32);
        assert!(self.inflight(1, pb).is_none(), "the wait ends with A's prefill");
        let b_rows = self.admit(1, pb);
        let mut b_toks = self.prefill(1, pb, self.chunk);
        self.finish(0, &mut a_toks, reply);
        self.finish(1, &mut b_toks, reply);
        b_rows
    }

    fn live(&self) -> u64 {
        self.k.kv.stats().snapshots_live
    }

    /// Boundaries and counters for an assertion message.
    fn dump(&self) -> String {
        let s = self.k.kv.stats();
        format!(
            "snaps {:?} live {} host {} evicted {} retired {} nodes_evicted {} cache_bytes {} \
             blocks {} spilled {} restored {} host_evictions {} fallbacks {} attach pe/blk/sh/miss \
             {}/{}/{}/{} touched {}",
            self.snaps(),
            s.snapshots_live,
            s.host_snapshots,
            s.snapshots_evicted,
            s.snapshots_retired,
            s.nodes_evicted,
            s.cache_bytes,
            s.cache_blocks,
            s.snapshots_spilled,
            s.snapshots_restored,
            s.host_evictions,
            s.spill_fallbacks,
            s.attach_prompt_end,
            s.attach_block,
            s.attach_shared,
            s.attach_misses,
            s.snapshots_touched
        )
    }

    fn snaps(&self) -> Vec<(u32, Kind, usize)> {
        let inner = self.k.kv.shared.inner.lock();
        let mut v: Vec<_> = inner
            .published
            .values()
            .flatten()
            .map(|s| (s.rows, s.kind, s.children.len()))
            .collect();
        v.sort_by_key(|x| x.0);
        v
    }
}

/// The next prompt of a conversation: the previous prompt with its last `rerender` rows
/// re-rendered (the chat template drops the generation prompt), the reply, and `new` fresh rows.
fn next_prompt(prev: &[u32], reply: &[u32], rerender: usize, new: usize, salt: u32) -> Vec<u32> {
    let mut p = prev[..prev.len() - rerender].to_vec();
    p.extend(reply.iter().map(|t| t ^ 0x0100_0000));
    p.extend((0..new as u32).map(|i| salt + i));
    p
}

fn conv_prompt(c: u32, sys: usize, first: usize) -> Vec<u32> {
    (0..sys as u32)
        .map(|i| 1_000 + i)
        .chain((0..first as u32).map(|i| c * 100_000 + 10_000 + i))
        .collect()
}

/// A conversation without `X-Session-Id` keeps one boundary: each prompt end supersedes the
/// one it attached, every turn attaches the previous prompt end, and the attach kind says so.
#[test]
fn cell_no_session_conversation_keeps_one_boundary() {
    for uniform in [false, true] {
        let mut r = Rig::new(Policy::Tree, uniform, 1 << 20, 64, 2);
        let mut p = conv_prompt(1, 40, 200);
        let (rows, mut reply) = r.turn(0, &p, 30);
        assert_eq!(rows, 0);
        assert_eq!(r.live(), 1, "the prompt end");
        for t in 1..6 {
            p = next_prompt(&p, &reply, 4, 100, 50_000 * t);
            let (rows, out) = r.turn(t as usize % 2, &p, 30);
            reply = out;
            let end = (p.len() as u32 - 100 - reply.len() as u32 + 4 - 9) / 32 * 32;
            assert_eq!(rows, end, "turn {t} attaches the previous prompt end");
            assert_eq!(r.live(), 1, "turn {t}: the superseded end was retired");
        }
        let s = r.kv().stats();
        assert_eq!(s.attach_prompt_end, 5);
        assert_eq!(s.attach_block + s.attach_shared, 0);
        assert_eq!(s.snapshots_retired, 5);
        assert_eq!(s.publishes_turn_end, 0, "no output boundary");
    }
}

/// Production's loss: a conversation's prompt end is evicted by other traffic before its next
/// turn (retention below the think time), so the turn attaches a block-aligned ancestor or
/// misses. With a host tier large enough for the working set the end spills instead and the
/// next turn restores it: every turn attaches at its prompt end. The device budget here holds
/// about three private snapshots; eight conversations interleave.
#[test]
fn cell_host_tier_keeps_prompt_ends_the_device_cannot() {
    const CONVS: u32 = 8;
    let run = |host: bool| {
        // 16 KiB snapshots (64 rows x 256 B) against 128 B columns (two 64 B blocks): the
        // device holds every conversation's blocks (~80 KiB at the last turn, plus two live
        // windows) and about four prompt ends.
        let mut r = Rig::new(Policy::Tree, false, 176 << 10, 64, 2);
        r.k.snap_row = 256;
        if host {
            r.k.kv.enable_host_tier(1 << 20, 16_500).unwrap();
        }
        // No shared system prompt: the sim's 8-row blocks would share whole blocks of it
        // across conversations, which strict publish refuses (`cell_strict_publish_*`).
        let mut prompts: Vec<Vec<u32>> = (0..CONVS).map(|c| conv_prompt(c, 0, 240)).collect();
        let mut replies: Vec<Vec<u32>> = Vec::new();
        for (c, p) in prompts.iter().enumerate() {
            replies.push(r.turn(c % 2, p, 30).1);
        }
        for t in 1..4u32 {
            for c in 0..CONVS as usize {
                let p = next_prompt(&prompts[c], &replies[c], 4, 100, 70_000 * t + c as u32);
                let (_, out) = r.turn(c % 2, &p, 30);
                prompts[c] = p;
                replies[c] = out;
            }
        }
        r.check();
        r.kv().stats()
    };
    let cold = run(false);
    let warm = run(true);
    let turns = (CONVS * 3) as u64;
    assert!(
        cold.attach_prompt_end < turns,
        "device alone: {} of {turns} turns found their prompt end",
        cold.attach_prompt_end
    );
    assert_eq!(
        warm.attach_prompt_end, turns,
        "host tier: every turn attaches its prompt end (block {}, miss {}); warm {warm:?}",
        warm.attach_block, warm.attach_misses
    );
    assert!(warm.snapshots_spilled > 0 && warm.snapshots_restored > 0);
    assert_eq!(warm.spill_fallbacks, 0);
    assert_eq!(warm.spill_fence_failures, 0);
    assert!(warm.host_snapshots > 0, "the tier holds the idle prompt ends");
}

/// `X-Session-Id` changes nothing under the tree: the same cell with sessions on gives the
/// same boundaries, hit rows and retirements (no pin, no output publish).
#[test]
fn cell_session_header_is_routing_only() {
    let mut r = Rig::new(Policy::Tree, false, 1 << 20, 64, 2);
    r.k.session = true;
    let mut p = conv_prompt(2, 40, 200);
    let (_, mut reply) = r.turn(0, &p, 30);
    for t in 1..4 {
        p = next_prompt(&p, &reply, 4, 100, 60_000 * t);
        let (rows, out) = r.turn(0, &p, 30);
        reply = out;
        assert!(rows > 0);
        assert_eq!(r.live(), 1);
    }
    assert_eq!(r.kv().stats().publishes_turn_end, 0);
    assert!(r.kv().shared.inner.lock().cache.pinned().is_empty());
}

/// Two requests of one conversation on the same history at once (parallel tool calls).
/// Path 1: B waits on A's in-flight prefill (`inflight_prefix`); path 2: B is admitted while A's
/// prompt is the most recent (`note_share`). The tree publishes no shared end for the pair and
/// the conversation keeps one boundary; legacy runs the same script without the claim (its
/// residue needs the production shapes).
#[test]
fn cell_parallel_pair_on_one_history() {
    for (policy, via_inflight) in [
        (Policy::Legacy, true),
        (Policy::Legacy, false),
        (Policy::Tree, true),
        (Policy::Tree, false),
    ] {
        let mut r = Rig::new(policy, false, 1 << 20, 64, 3);
        let p1 = conv_prompt(3, 40, 150);
        let (_, reply) = r.turn(0, &p1, 30);
        let p2 = next_prompt(&p1, &reply, 4, 400, 70_000);
        // A: admitted and prefilling the first 64 rows past its attach point.
        let a_rows = r.admit(0, &p2);
        assert!(a_rows > 0);
        let mut a_toks = p2[..a_rows as usize].to_vec();
        a_toks.extend_from_slice(&p2[a_rows as usize..a_rows as usize + 64]);
        r.pos[0] = a_toks.len() as u32;
        let b_rows;
        if via_inflight {
            let wait = r.inflight(1, &p2);
            assert!(wait.is_some_and(|(o, _)| o == 0), "B waits on A ({policy:?})");
            // A finishes its prefill (publishing its shared end if one was named), B attaches.
            let c0 = a_toks.len();
            a_toks.extend_from_slice(&p2[c0..]);
            r.pos[0] = a_toks.len() as u32;
            r.k.publish_shared(0, &a_toks, c0);
            r.k.publish(0, &a_toks, a_toks.len() as u32 - 1, RIG_TTL);
            r.k.kv.prefill_done(0);
            assert!(r.inflight(1, &p2).is_none(), "the wait ends");
            b_rows = r.admit(1, &p2);
        } else {
            b_rows = r.admit(1, &p2);
            let c0 = a_toks.len();
            a_toks.extend_from_slice(&p2[c0..]);
            r.pos[0] = a_toks.len() as u32;
            r.k.publish_shared(0, &a_toks, c0);
            r.k.publish(0, &a_toks, a_toks.len() as u32 - 1, RIG_TTL);
            r.k.kv.prefill_done(0);
        }
        let mut b_toks = r.prefill(1, &p2, 4096);
        assert!(b_rows > 0);
        let mut a_toks = a_toks;
        let reply_a = r.finish(0, &mut a_toks, 30);
        r.finish(1, &mut b_toks, 30);
        let share_snaps = r.kv().stats().snapshots_share;
        // Turn 3 continues from A's reply.
        let p3 = next_prompt(&p2, &reply_a, 4, 100, 80_000);
        let (rows, _) = r.turn(2, &p3, 30);
        assert!(rows > 0, "turn 3 attaches a prompt end ({policy:?})");
        let snaps = r.snaps();
        if policy == Policy::Tree {
            assert_eq!(r.kv().stats().publishes_shared_end, 0, "no shared end: {snaps:?}");
            assert_eq!(r.live(), 1, "one boundary: {snaps:?}");
        }
        let _ = share_snaps;
    }
}

/// Parallel tool calls under strict publish: two requests on one history with different new
/// messages, the second admitted while the first prefills (no wait: its attach lands first).
/// Both attach B; A publishes C; D's own end is refused (its shared rows are A's blocks). B
/// keeps D's pending continuation and is not retired behind C, so D's next turn attaches B.
#[test]
fn cell_parallel_pair_second_request_keeps_its_attach_point_under_strict_publish() {
    let mut r = Rig::new(Policy::Tree, false, 1 << 20, 64, 3);
    let p1 = conv_prompt(7, 40, 150);
    let (_, reply) = r.turn(0, &p1, 30);
    let b_rows = (p1.len() as u32 - 9) / 32 * 32;
    let pa = next_prompt(&p1, &reply, 4, 100, 70_000);
    let pd = next_prompt(&p1, &reply, 4, 100, 71_000);
    assert_eq!(r.admit(0, &pa), b_rows);
    assert_eq!(r.admit(1, &pd), b_rows);
    let mut a_toks = r.prefill(0, &pa, 4096);
    let mut d_toks = r.prefill(1, &pd, 4096);
    assert_eq!(r.live(), 2, "B and C; D's end was refused: {}", r.dump());
    let reply_d = r.finish(1, &mut d_toks, 20);
    r.finish(0, &mut a_toks, 20);
    assert_eq!(r.live(), 2, "B stays for D: {}", r.dump());
    let (rows, _) = r.turn(2, &next_prompt(&pd, &reply_d, 4, 50, 72_000), 10);
    assert_eq!(rows, b_rows, "D's next turn attaches B: {}", r.dump());
    assert_eq!(r.kv().stats().attach_prompt_end, 3);
}

/// A retry re-sends the same prompt (dedup, no new boundary); a regenerate from an earlier turn
/// diverges below the retired end and recomputes from the boundary above it: a miss, never a
/// wrong attach, and the cache stays bounded.
#[test]
fn cell_retry_and_regenerate() {
    let mut r = Rig::new_with(Policy::Tree, false, 1 << 20, 64, 2, false);
    let p1 = conv_prompt(4, 40, 200);
    let (_, r1) = r.turn(0, &p1, 30);
    let p2 = next_prompt(&p1, &r1, 4, 100, 90_000);
    let (rows2, r2) = r.turn(0, &p2, 30);
    assert!(rows2 > 0);
    let live = r.live();
    let (rows_retry, _) = r.turn(1, &p2, 30);
    assert_eq!(rows_retry, p2.len() as u32 - 1 - 8 - (p2.len() as u32 - 9) % 32, "the retry attaches its own prompt end");
    assert_eq!(r.live(), live, "a retry adds no boundary");
    assert_eq!(
        r.kv().stats().snapshots_touched,
        1 + 3,
        "the retry's deduped publish + three turn-end touches"
    );
    // Regenerate turn 2 with a different message: P1's end is retired, so this is a miss below P2.
    let p2b = next_prompt(&p1, &r1, 4, 100, 91_000);
    let (rows_b, _) = r.turn(0, &p2b, 30);
    assert_eq!(rows_b, 0, "no boundary on the regenerate's chain survives P1's retire");
    assert_eq!(r.live(), 2, "two live branches of the conversation: {}", r.dump());
    let _ = r2;
}

/// The same regenerate under strict publish (FP8 KV): its prefix blocks are P2's chain's, so
/// its boundary is refused while that chain lives (`resolve_prefix_hazard`); P2's end stays
/// and the regenerate's next turn misses again. Pre-existing, policy-independent.
#[test]
fn cell_regenerate_under_strict_publish_gets_no_boundary() {
    let mut r = Rig::new(Policy::Tree, false, 1 << 20, 64, 2);
    let p1 = conv_prompt(4, 40, 200);
    let (_, r1) = r.turn(0, &p1, 30);
    let p2 = next_prompt(&p1, &r1, 4, 100, 90_000);
    r.turn(0, &p2, 30);
    let p2b = next_prompt(&p1, &r1, 4, 100, 91_000);
    let (rows_b, _) = r.turn(0, &p2b, 30);
    assert_eq!(rows_b, 0);
    assert_eq!(r.live(), 1, "P2's end only: {}", r.dump());
}

/// Twenty conversations share a 288-row system prompt (the sim's 8-row blocks: a system
/// prompt whose end is not 32-aligned leaves a whole shared block past the shared end, which
/// strict publish then refuses; production's 2048-row blocks cannot) under a cap of ~6
/// snapshots: the shared
/// end is attached by every new conversation (LRU refresh) and survives; under legacy it is
/// kept by the share tier.
#[test]
fn cell_shared_system_prompt_under_pressure() {
    for policy in [Policy::Tree, Policy::Legacy] {
        // The shared 300 rows (37 columns of 128 B), two live windows' private rows and
        // about six idle conversations' private rows (15 columns + a snapshot each).
        let cap = 16 << 10;
        let mut r = Rig::new(policy, false, cap, 64, 2);
        let first = r.shared_pair(&conv_prompt(10, 288, 120), &conv_prompt(11, 288, 120), 20);
        assert!(first >= 256, "{policy:?}: the pair's shared end: {first}");
        let mut hits = 1;
        for c in 2..20u32 {
            let p = conv_prompt(10 + c, 288, 120);
            let (rows, _) = r.turn((c % 2) as usize, &p, 20);
            assert!(rows >= 256, "conversation {c} ({policy:?}): attached {rows} rows");
            hits += 1;
        }
        assert_eq!(hits, 19);
        let s = r.kv().stats();
        assert_eq!(s.attach_shared, 19, "{policy:?}");
        assert!(s.snapshots_live <= 9, "{policy:?}: {} live", s.snapshots_live);
    }
}

/// A waiter's reservation survives pressure: the checkpoint it waits for is never the LRU
/// victim while the wait lasts, and is evictable once the wait ends (attach or cancel).
#[test]
fn cell_waiter_reservation_survives_pressure() {
    let mut r = Rig::new(Policy::Tree, true, 1 << 20, 64, 3);
    r.chunk = 32;
    let base: Vec<u32> = (0..200).collect();
    let (_, _) = r.turn(0, &base, 10); // boundary at (200-9)/32*32 = 160
    let _held = r.admit(0, &base);
    // An owner prefilling a fresh prompt from row 0; a waiter shares its first 128 rows.
    // The owner publishes within `lookback` (64 rows) of its end: the waiter must share a
    // whole block at row 136 or beyond.
    let long: Vec<u32> = (3_000..3_200).collect();
    assert_eq!(r.admit(1, &long), 0);
    let w: Vec<u32> = long[..144].iter().copied().chain(9_000..9_100).collect();
    assert_eq!(r.inflight(2, &w), Some((1, 144)));
    let toks = r.prefill(1, &long, 32);
    assert!(r.kv().checkpoint_awaited(&toks, 144));
    let awaited_live = r.live();
    assert!(awaited_live >= 3, "base's end, the checkpoint, long's end");
    {
        let mut inner = r.kv().shared.inner.lock();
        while evict_one(&r.kv().shared, &mut inner, false) {}
        assert!(
            inner
                .published
                .values()
                .flatten()
                .any(|s| s.rows == 144 && s.tail.is_empty()),
            "the awaited checkpoint stays"
        );
        assert!(inner.published.values().flatten().count() < awaited_live as usize);
    }
    r.kv().clear_await(2);
    {
        let mut inner = r.kv().shared.inner.lock();
        while evict_one(&r.kv().shared, &mut inner, false) {}
        assert!(
            inner
                .published
                .values()
                .flatten()
                .all(|s| s.rows != 144 || !s.tail.is_empty()),
            "evictable once the wait ended"
        );
    }
    r.kv().begin_seq(0);
    r.kv().begin_seq(1);
}

/// Late branching: B gets two continuations C and D (D's request holds B while C publishes,
/// a parallel pair) and stays as a branch point; when LRU takes C, B has one continuation left
/// and retires with it (D remains the attach point; C's chain recomputes from nothing).
#[test]
fn cell_branch_point_retires_when_one_child_is_left() {
    let mut r = Rig::new_with(Policy::Tree, true, 1 << 20, 64, 3, false);
    let p1 = conv_prompt(5, 40, 200);
    let (_, r1) = r.turn(0, &p1, 30);
    let b_rows = (p1.len() as u32 - 9) / 32 * 32;
    let pc = next_prompt(&p1, &r1, 4, 100, 110_000);
    let pd = next_prompt(&p1, &r1, 4, 100, 120_000);
    r.kv().begin_seq(1);
    let hold = r.kv().try_attach(1, &pd).unwrap().unwrap();
    assert_eq!(hold.rows, b_rows);
    r.kv().ensure_rows(1, pd.len() as u32 + 8).unwrap();
    r.pos[1] = hold.rows;
    let (rows_c, _) = r.turn(0, &pc, 10);
    assert_eq!(rows_c, b_rows);
    assert_eq!(r.live(), 2, "B held by D's request: not retired");
    let mut d_toks = r.prefill(1, &pd, 4096);
    assert_eq!(r.live(), 3, "B, C, D: {}", r.dump());
    let snaps = r.snaps();
    assert!(
        snaps.iter().any(|s| s.0 == b_rows && s.2 == 2),
        "B is a branch point: {snaps:?}"
    );
    assert_eq!(r.kv().stats().snapshots_branch, 1);
    r.finish(1, &mut d_toks, 10);
    let c_rows = (pc.len() as u32 - 9) / 32 * 32;
    let c_id = {
        let inner = r.kv().shared.inner.lock();
        inner
            .published
            .values()
            .flatten()
            .find(|s| s.rows == c_rows && pc[..c_rows as usize].ends_with(&s.tail))
            .map(|s| s.id)
            .unwrap()
    };
    {
        let mut inner = r.kv().shared.inner.lock();
        policy::evict_unit(&r.kv().shared, &mut inner, c_id);
    }
    assert_eq!(r.live(), 1, "C went, B retired with it, D remains");
    assert_eq!(r.kv().stats().snapshots_retired, 1);
    let (rows_c2, _) = r.turn(0, &pc, 10);
    assert_eq!(rows_c2, 0, "C's chain recomputes from nothing");
    let (rows_d2, _) = r.turn(1, &pd, 10);
    assert_eq!(rows_d2, (pd.len() as u32 - 9) / 32 * 32, "D is the attach point");
    r.check();
}

/// A system prompt end (`SHARED_END`) attached by several conversations is never retired by a
/// single continuation; a prompt end attached twice (a retry and the real next turn) is.
#[test]
fn cell_shared_end_is_structurally_exempt_prompt_end_is_not() {
    let mut r = Rig::new(Policy::Tree, true, 1 << 20, 64, 2);
    let a = conv_prompt(20, 288, 100);
    let b = conv_prompt(21, 288, 100);
    let rows_b = r.shared_pair(&a, &b, 10);
    assert!(rows_b >= 256, "the pair's shared end: {rows_b}");
    let c = conv_prompt(22, 288, 100);
    let (rows_c, _) = r.turn(0, &c, 10);
    assert!(rows_c >= 256, "the shared end: {rows_c}");
    let snaps = r.snaps();
    assert!(snaps.iter().any(|s| s.1.has(Kind::SHARED_END)), "{snaps:?}");
    let d = conv_prompt(23, 288, 100);
    r.turn(1, &d, 10);
    assert!(r.snaps().iter().any(|s| s.1.has(Kind::SHARED_END)), "still there: {:?}", r.snaps());
    // A prompt end attached twice and then continued is retired all the same.
    let (rows1, r1) = r.turn(0, &d, 10);
    assert!(rows1 > 0);
    let live = r.live();
    let d2 = next_prompt(&d, &r1, 4, 50, 130_000);
    let (rows2, _) = r.turn(0, &d2, 10);
    assert_eq!(rows2, rows1, "d2 attaches d's end too: {}", r.dump());
    assert_eq!(r.live(), live, "d's end retired as d2 published: {}", r.dump());
    assert_eq!(r.kv().stats().snapshots_retired, 1);
}

/// Under the tree a boundary's recency moves on consumer demand only: a republish of an
/// identical boundary (a retry's prompt end) leaves `last_used` alone; an attach and the
/// turn-end touch refresh it and its ancestors.
#[test]
fn cell_recency_moves_on_attach_and_touch_only() {
    let mut r = Rig::new(Policy::Tree, true, 1 << 20, 64, 2);
    let a = conv_prompt(30, 288, 100);
    let b = conv_prompt(31, 288, 100);
    r.shared_pair(&a, &b, 10);
    let c = conv_prompt(32, 288, 100);
    r.turn(0, &c, 10);
    let used = |r: &Rig, rows: u32| {
        r.kv()
            .shared
            .inner
            .lock()
            .published
            .values()
            .flatten()
            .find(|s| s.rows == rows)
            .map(|s| s.last_used)
            .unwrap()
    };
    let shared_rows = r.snaps().iter().find(|s| s.1.has(Kind::SHARED_END)).unwrap().0;
    let t0 = used(&r, shared_rows);
    let (rows, _) = r.turn(1, &c, 10); // a retry of c: attaches c's end, republishes it
    assert!(rows > shared_rows, "{rows} vs {shared_rows}: {}", r.dump());
    assert!(used(&r, shared_rows) > t0, "the ancestor was refreshed by the attach");
}

/// Strict publish (FP8 KV, the production pool): a conversation that arrives after its
/// system-prompt twin finished has no boundary to attach below the twin's prompt end, renders
/// its own blocks, and its shared-end publish over the twin's cached blocks is refused
/// (`resolve_prefix_hazard`): no shared end, the third conversation misses. The pair must
/// overlap in prefill (`shared_pair`) for a shared end to exist; the policy cannot change
/// this.
#[test]
fn cell_strict_publish_refuses_a_sequential_shared_end() {
    let mut r = Rig::new(Policy::Tree, true, 1 << 20, 64, 2);
    let a = conv_prompt(60, 288, 100);
    let b = conv_prompt(61, 288, 100);
    r.turn(0, &a, 10);
    let (rows_b, _) = r.turn(1, &b, 10);
    assert_eq!(rows_b, 0);
    assert_eq!(r.kv().stats().publishes_shared_end, 0, "{:?}", r.snaps());
    let (rows_c, _) = r.turn(0, &conv_prompt(62, 288, 100), 10);
    assert_eq!(rows_c, 0, "no shared end to attach");
}

/// Sliding-only geometry: the pool needs one full-attention track to share and refuses
/// explicitly; the policy itself never reads layer kinds.
#[test]
fn cell_sliding_only_geometry_is_refused_explicitly() {
    let geo = VmmGeometry {
        full_layers: vec![],
        kvh_full: 1,
        hd_full: 4,
        slide_layers: vec![0, 1, 2],
        kvh_slide: 2,
        hd_slide: 4,
        window: 64,
        elem: 2,
        elem_slide: 2,
        max_ctx: 2048,
        batch: 2,
    };
    let ops = Arc::new(SimVmm {
        cap: 1 << 20,
        used: AtomicU64::new(0),
        next: AtomicU64::new(1),
        sizes: StdMutex::new(HashMap::new()),
        lag_us: [0, 0],
        gran: 64,
    });
    let err = VmmKv::new(ops, geo, 64, 0).err().expect("refused").to_string();
    assert!(err.contains("cache tensor tracks"), "{err}");
}

/// The prompt-end publish of a conversation whose boundary was evicted during its decode
/// (pressure) restores it at the turn end; a surviving one is only touched (no copy).
#[test]
fn cell_turn_end_restores_an_evicted_prompt_end() {
    let mut r = Rig::new(Policy::Tree, false, 1 << 20, 64, 2);
    let p = conv_prompt(40, 40, 200);
    let rows = r.admit(0, &p);
    assert_eq!(rows, 0);
    let mut toks = r.prefill(0, &p, 4096);
    assert_eq!(r.live(), 1);
    {
        let mut inner = r.kv().shared.inner.lock();
        assert!(evict_one(&r.kv().shared, &mut inner, false));
        assert_eq!(inner.stats.snapshots_live, 0);
    }
    r.finish(0, &mut toks, 30);
    assert_eq!(r.live(), 1, "restored at the turn end");
    assert_eq!(r.k.republish.get(), (1, 1));
    let (rows2, _) = r.turn(1, &next_prompt(&p, &toks[p.len()..], 4, 50, 140_000), 10);
    assert!(rows2 > 0);
    assert_eq!(r.k.republish.get(), (1, 2), "the surviving end was touched, not copied");
}
