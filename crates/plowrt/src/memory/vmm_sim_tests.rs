//! Closed-loop agentic-serving simulation of the VMM prefix cache (`cargo test -p plowrt
//! --features cuda,hsa --lib --release vmm_sim -- --ignored --nocapture`): the real [`VmmKv`]
//! on a byte-accounting driver, driven the way the CUDA engine drives it (admission, prompt-end
//! publish, turn-end publish and retire). Env: `SIM_CELLS` (comma-separated prompt files of
//! `session turn prompt_tokens` lines, run back to back on one pool), `SIM_KV` (fp8|bf16),
//! `SIM_SLOTS`, `SIM_FREE_MIB` (free after load), `SIM_FLOOR_MIB`, `SIM_CHUNK` / `SIM_PF_TPS`
//! (prefill launch rows / rows per second), `SIM_TTL_S`, `SIM_GAP_MS` (client round trip between
//! turns), `SIM_MARGIN` (admission rows past prompt + output), and `SIM_SCALE` with `SIM_UNMAP_US` /
//! `SIM_RELEASE_US` to pace the reclaimer thread in real time instead of draining it per call.
use super::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex as StdMutex;

struct SimVmm {
    cap: u64,
    used: AtomicU64,
    next: AtomicU64,
    sizes: StdMutex<HashMap<u64, u64>>,
    lag_us: [u64; 2],
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
        Ok(2 << 20)
    }
    fn reserve(&self, bytes: u64) -> Result<u64> {
        Ok(self.next.fetch_add(bytes.div_ceil(1 << 21) + 1, Ordering::SeqCst) << 21)
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
}

struct Kv {
    kv: VmmKv,
    ring: u32,
    window: u32,
    snap_row: u64,
    full_row: u64,
    scale_row: u64,
    /// Turn-end re-publishes that had to copy (the prompt end was evicted while decoding) / all.
    republish: std::cell::Cell<(u64, u64)>,
}

impl Kv {
    fn snap_bytes(&self, p_a: u32) -> u64 {
        let br = self.kv.block_rows();
        self.snap_row * u64::from(self.window.min(p_a))
            + self.scale_row * u64::from(p_a)
            + self.full_row * u64::from(p_a % br)
    }

    /// `exec/gpu/prefix.rs::publish_boundary`.
    fn boundary(&self, b: usize, toks: &[u32], p_a: u32) -> bool {
        self.boundary_copied(b, toks, p_a).0
    }

    fn boundary_copied(&self, b: usize, toks: &[u32], p_a: u32) -> (bool, bool) {
        let copied = std::cell::Cell::new(false);
        let rows = toks.len() as u32;
        if rows == 0 || p_a == 0 || rows - p_a > self.ring - self.window {
            return (false, false);
        }
        if !self.kv.resolve_prefix_hazard(b, toks, p_a) {
            return (false, false);
        }
        let ok = self.kv.publish_at(b, toks, p_a, self.snap_bytes(p_a), |_| {
            copied.set(true);
            Ok(())
        });
        (ok.is_ok(), copied.get())
    }

    /// `exec/gpu/prefix.rs::vmm_publish` for a session slot.
    fn publish(&self, b: usize, toks: &[u32], max_rows: u32, ttl: std::time::Duration) {
        let rows = toks.len() as u32;
        let prompt = self.kv.prompt_rows(b);
        let slack = env_u64("SIM_PE_SLACK", 8) as u32;
        let max_rows = if rows == prompt && max_rows < rows { prompt.saturating_sub(1 + slack) } else { max_rows };
        let p_a = rows.min(max_rows) / 32 * 32;
        if rows == 0 || p_a == 0 {
            return;
        }
        self.kv.note_session(b, toks);
        let step = self.kv.block_rows();
        let mut p = step;
        while p < p_a {
            if self.kv.checkpoint_awaited(toks, p) {
                self.boundary(b, toks, p);
            }
            p += step;
        }
        if rows < prompt && !self.kv.checkpoint_awaited(toks, p_a) {
            return;
        }
        let published = self.boundary(b, toks, p_a);
        if published && rows == prompt && max_rows < rows {
            self.kv.pin_prefix(&toks[..p_a as usize], std::time::Instant::now() + ttl);
            self.kv.retire_superseded(toks, p_a);
        }
        let end = prompt.saturating_sub(1 + slack) / 32 * 32;
        if rows > prompt && end > 0 && end < p_a {
            let (_, copied) = self.boundary_copied(b, toks, end);
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
}

fn tokens_for(cell: u32, s: usize, t: usize, prev: &[u32], prev_len: usize, len: usize) -> Vec<u32> {
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
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
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
    });
    // Real-time pacing (`SIM_SCALE` > 0): the reclaimer thread runs against virtual time instead
    // of being drained after every engine call.
    let scale = std::env::var("SIM_SCALE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
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
    let k = Kv {
        kv,
        ring: 2048,
        window: 1024,
        snap_row: 40 * 2 * 8 * 256 * elem as u64,
        full_row: 16 * 512 * elem as u64,
        scale_row: if fp8 { 64 } else { 0 },
        republish: std::cell::Cell::new((0, 0)),
    };
    let ttl = std::time::Duration::from_secs(env_u64("SIM_TTL_S", 60));
    let margin = env_u64("SIM_MARGIN", 4224) as usize;
    for (cell, path) in cells.iter().enumerate() {
        let mut per: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for line in std::fs::read_to_string(path).expect("prompt file").lines() {
            let f: Vec<usize> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
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
        loop {
            while arriving.front().is_some_and(|&(t, _)| t <= now) {
                queue.push_back(arriving.pop_front().unwrap().1);
            }
            while let (Some(&id), Some(&b)) = (queue.front(), free_slots.last()) {
                queue.pop_front();
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
                let total = (s.prompt.len() + 128 + margin).min(16384);
                k.kv.ensure_rows(b, total as u32).expect("admission rows");
                sync(&k.kv);
                let h = &mut hit[s.turn];
                h.0 += rows as u64;
                h.1 += s.prompt.len() as u64;
                h.2 += 1;
                s.toks = s.prompt[..rows].to_vec();
                s.slot = b;
                s.phase = Phase::Prefill;
                seqno += 1;
                s.admitted = seqno;
            }
            let mut prefilling: Vec<usize> =
                (0..sess.len()).filter(|&i| sess[i].phase == Phase::Prefill).collect();
            let decoding: Vec<usize> = (0..sess.len()).filter(|&i| sess[i].phase == Phase::Decode).collect();
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
                    let share = k.kv.share_rows(b) as usize;
                    if share > c0 && share <= s.toks.len() {
                        k.boundary(b, &s.toks, share as u32);
                    }
                    if s.toks.len() == s.prompt.len() {
                        k.publish(b, &s.toks, s.toks.len() as u32 - 1, ttl);
                        k.kv.prefill_done(b);
                        s.phase = Phase::Decode;
                        s.decoded = 0;
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
                    s.decoded += 1;
                    if s.decoded < 128 {
                        let t = (s.toks.len() as u32) ^ 0x4000_0000;
                        s.toks.push(t);
                        continue;
                    }
                    let b = s.slot;
                    k.publish(b, &s.toks, s.toks.len() as u32, ttl);
                    k.kv.pin_prefix(&s.toks, std::time::Instant::now() + ttl);
                    k.kv.begin_seq(b);
                    sync(&k.kv);
                    free_slots.push(b);
                    s.turn += 1;
                    if s.turn == s.prompts.len() {
                        s.phase = Phase::Done;
                        continue;
                    }
                    let prev_len = s.prompt.len();
                    s.prompt = tokens_for(cell as u32, id, s.turn, &s.prompt, prev_len, s.prompts[s.turn]);
                    s.phase = Phase::Queued;
                    arriving.push_back((now + gap, id));
                }
            }
        }
        let st = k.kv.stats();
        let (a, p, n): (u64, u64, u64) = hit.iter().fold((0, 0, 0), |x, h| (x.0 + h.0, x.1 + h.1, x.2 + h.2));
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
        let per_turn: Vec<String> =
            hit.iter().map(|h| format!("{:.0}", 100.0 * h.0 as f64 / h.1.max(1) as f64)).collect();
        println!("SIM   per-turn cached% {}", per_turn.join(" "));
        let (c, n) = k.republish.replace((0, 0));
        println!("SIM   turn-end re-publish copied {c} of {n}");
    }
}
