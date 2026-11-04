//! A CUDA model served by a process rank group (`exec::proc_group`): this process runs rank 0
//! and drives one worker process per other device (`plowrt rank-worker`), each a full
//! [`GpuEngine`] bound to its rank.
//!
//! The group is a [`SeqEngine`]: every call is executed on every rank in the same order —
//! sent to the workers first, run locally, then joined. Ranks keep their own slot state; a
//! launch unit whose plan differs between ranks is refused by every rank at its arm (see
//! `exec::proc_group`), so a divergence fails the request instead of hanging the group, and a
//! failed group stays failed. Rank 0's tokens are returned; every rank's are compared.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::engine::SeqEngine;
use crate::asset::devblob::DevBlob;
use crate::device::cuda::CudaBackend;
use crate::exec::gpu::rank::RankBind;
use crate::exec::gpu::{GpuEngine, PrefillStep};
use crate::exec::proc_group::{inheritable, Link, ProcGroup};
use crate::exec::tp::{count_xgates, PeerLayout};
use crate::{Result, RuntimeError};

/// A worker answers each request within this, or the group is failed.
const RPC_TIMEOUT: Duration = Duration::from_secs(600);

#[path = "cuda_p2p.rs"]
mod p2p;

pub(super) fn write_transport_metrics(out: &mut String) {
    p2p::write_metrics(out);
}

#[derive(Serialize, Deserialize, Debug)]
enum Req {
    Begin {
        slot: usize,
        total: usize,
    },
    Prefill {
        slot: usize,
        prompt: Vec<u32>,
        cap: usize,
    },
    Step {
        feeds: Vec<(usize, u32)>,
    },
    Retire {
        slot: usize,
        cache: bool,
    },
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
enum Resp {
    Ready {
        batch: usize,
        max_ctx: usize,
    },
    Done,
    /// A prefill chunk ran: the frontier, the rows the prefix cache attached for this
    /// sequence (every rank must attach the same rows), and the first token once done.
    Chunk {
        frontier: usize,
        attached: u32,
        token: Option<u32>,
    },
    Tokens(Vec<u32>),
    Err(String),
}

impl Req {
    fn kind(&self) -> &'static str {
        match self {
            Req::Begin { .. } => "begin",
            Req::Prefill { .. } => "prefill",
            Req::Step { .. } => "step",
            Req::Retire { .. } => "retire",
        }
    }
}

/// One request on one rank's engine; a rank's audit runs after every launching request.
fn execute(e: &mut GpuEngine, req: &Req) -> Result<Resp> {
    let resp = match req {
        Req::Begin { slot, total } => e.begin_slot(*slot, *total).map(|()| Resp::Done)?,
        Req::Prefill { slot, prompt, cap } => {
            let (frontier, token) = match e.prefill_chunk(*slot, prompt, *cap)? {
                PrefillStep::Progress(n) => (n, None),
                PrefillStep::Done(t) => (prompt.len(), Some(t)),
            };
            Resp::Chunk {
                frontier,
                attached: e.attached_rows(*slot),
                token,
            }
        }
        Req::Step { feeds } => {
            let mut toks = Vec::with_capacity(feeds.len());
            e.step_slots(feeds, &mut toks)?;
            Resp::Tokens(toks)
        }
        Req::Retire { slot, cache } => {
            e.retire_slot(*slot, *cache);
            Resp::Done
        }
    };
    e.rank_audit()?;
    Ok(resp)
}

struct Worker {
    link: Link,
    child: Arc<Mutex<Child>>,
}

pub struct CudaGroupServe {
    eng: Mutex<GpuEngine>,
    workers: Vec<Worker>,
    group: Arc<ProcGroup>,
    batch: usize,
    max_ctx: usize,
    stop_ids: Arc<Vec<u32>>,
    live: Vec<bool>,
    pf_pos: Vec<usize>,
    attached: Vec<usize>,
    /// The rank-0 logits row holding each slot's latest token: a decode launch writes slot `b`
    /// to row `b`, and a finished prefill leaves its row in row 0.
    logits_at: Vec<Option<usize>>,
    prefill_turn: usize,
    _transport_metrics: p2p::Metrics,
}

/// The rank-group geometry a packet directory needs: its tensor-parallel degree and the
/// collective region each rank carries.
pub fn group_layout(assets_dir: &Path) -> Result<Option<(u32, PeerLayout)>> {
    let Some(pkt) = DevBlob::find_in_dir(assets_dir)? else {
        return Ok(None);
    };
    let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io { path: pkt, source })?;
    let blob = DevBlob::parse(&raw)?;
    let Some(tp) = blob.tp else { return Ok(None) };
    let row_bytes = u64::from(tp.hidden) * 2;
    let max_tokens = tp
        .slot_bytes
        .checked_div(row_bytes)
        .filter(|_| tp.slot_bytes % row_bytes == 0)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| {
            RuntimeError::Rejected("TP partial slot is not a whole row extent".into())
        })?;
    let layout = PeerLayout::new(tp.hidden, max_tokens, count_xgates(&blob)).ok_or_else(|| {
        RuntimeError::Rejected(format!(
            "TP packet partial slot {} B is not a whole aligned [{max_tokens}, {}] bf16 tensor",
            tp.slot_bytes, tp.hidden
        ))
    })?;
    Ok(Some((tp.n_gpu, layout)))
}

impl CudaGroupServe {
    /// Bring up the group over `devices` (one `CUDA_VISIBLE_DEVICES` value per rank; this
    /// process must already be pinned to `devices[0]`), loading the packet on every rank.
    pub fn load(
        be: Arc<CudaBackend>,
        assets_dir: &Path,
        checkpoint_dir: &Path,
        devices: &[String],
    ) -> Result<Self> {
        let (tp, layout) = group_layout(assets_dir)?.ok_or_else(|| {
            RuntimeError::Rejected(format!(
                "{} is not a tensor-parallel packet",
                assets_dir.display()
            ))
        })?;
        if devices.len() != tp as usize {
            return Err(RuntimeError::Rejected(format!(
                "tensor-parallel degree {tp} needs {tp} group devices, PLOW_GROUP_DEVICES names {}",
                devices.len()
            )));
        }
        let group = Arc::new(ProcGroup::create(tp, tp, Some(layout), 0)?);
        inheritable(group.fd())?;
        let exe = std::env::current_exe().map_err(|e| RuntimeError::Device(e.to_string()))?;
        let mut workers = Vec::with_capacity(devices.len() - 1);
        for (r, dev) in devices.iter().enumerate().skip(1) {
            workers.push(spawn_worker(
                &exe,
                &group,
                r as u32,
                dev,
                assets_dir,
                checkpoint_dir,
            )?);
        }
        let (bind, transport) = p2p::leader(be.clone(), group.clone(), assets_dir, &mut workers)
            .inspect_err(|_| group.abort())?;
        let local = GpuEngine::load_rank(be, assets_dir, checkpoint_dir, Some(bind));
        let local = match local {
            Ok(e) => e,
            Err(e) => {
                group.abort();
                return Err(e);
            }
        };
        let (batch, max_ctx) = (local.batch(), local.max_ctx());
        for (i, w) in workers.iter_mut().enumerate() {
            let ready = w.link.recv::<Resp>().inspect_err(|_| group.abort())?;
            match ready {
                Resp::Ready {
                    batch: b,
                    max_ctx: m,
                } if b == batch && m == max_ctx => {}
                Resp::Err(e) => {
                    group.abort();
                    return Err(RuntimeError::Device(format!(
                        "rank {} failed to load: {e}",
                        i + 1
                    )));
                }
                other => {
                    group.abort();
                    return Err(RuntimeError::Device(format!(
                        "rank {} loaded a different engine: {other:?} vs batch {batch} ctx {max_ctx}",
                        i + 1
                    )));
                }
            }
        }
        tracing::info!(
            ranks = tp,
            batch,
            max_ctx,
            ?devices,
            "tensor-parallel rank group up"
        );
        let transport_metrics = p2p::Metrics::new(
            transport,
            local.rank_bind().expect("rank binding").transport_tier(),
        );
        Ok(Self {
            stop_ids: local.stop_ids().clone(),
            eng: Mutex::new(local),
            workers,
            group,
            batch,
            max_ctx,
            live: vec![false; batch],
            pf_pos: vec![0; batch],
            attached: vec![0; batch],
            logits_at: vec![None; batch],
            prefill_turn: 0,
            _transport_metrics: transport_metrics,
        })
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn prefix_cache_enabled(&self) -> bool {
        self.eng.lock().vmm_prefix_enabled()
    }

    pub fn ranks(&self) -> usize {
        self.workers.len() + 1
    }

    /// Run `req` on every rank; the agreed response, or the first failure (which fails the group).
    fn all(&mut self, req: Req) -> Result<Resp> {
        if self.group.aborted() {
            return Err(RuntimeError::CollectiveBail(
                "rank group failed earlier; restart the server".into(),
            ));
        }
        let mut sent = Ok(());
        for w in &mut self.workers {
            sent = sent.and_then(|()| w.link.send(&req));
        }
        let t0 = std::time::Instant::now();
        let local = sent.and_then(|()| execute(&mut self.eng.lock(), &req));
        let local_us = t0.elapsed().as_micros() as u64;
        if local.is_err() {
            // Release workers waiting for rank 0 at a launch barrier before collecting replies.
            self.group.abort();
        }
        let mut out = local.map_err(|e| (0usize, e));
        for (i, w) in self.workers.iter_mut().enumerate() {
            let got = match w.link.recv::<Resp>() {
                Ok(Resp::Err(e)) => Err(RuntimeError::Device(e)),
                Ok(r) => Ok(r),
                Err(e) => Err(e),
            };
            out = match (out, got) {
                (Err(e), _) => Err(e),
                (Ok(_), Err(e)) => Err((i + 1, e)),
                (Ok(a), Ok(b)) if a == b => Ok(a),
                (Ok(a), Ok(b)) => Err((
                    i + 1,
                    RuntimeError::CollectiveBail(format!(
                        "ranks disagree: rank 0 {a:?}, rank {} {b:?}",
                        i + 1
                    )),
                )),
            };
        }
        tracing::debug!(
            req = req.kind(),
            local_us,
            total_us = t0.elapsed().as_micros() as u64,
            "rank group request"
        );
        out.map_err(|(rank, e)| {
            self.group.abort();
            tracing::error!(rank, error = %e, "rank group request failed; group is down");
            e
        })
    }

    fn prefill_chunk(&mut self, slot: usize, prompt: &[u32], cap: usize) -> Result<Option<u32>> {
        if slot >= self.batch {
            return Err(RuntimeError::Rejected(format!("slot {slot} out of range")));
        }
        if !self.live[slot] && self.pf_pos[slot] == 0 {
            self.all(Req::Begin {
                slot,
                total: prompt.len(),
            })?;
            self.live[slot] = true;
        }
        match self.all(Req::Prefill {
            slot,
            prompt: prompt.to_vec(),
            cap,
        })? {
            Resp::Chunk {
                frontier,
                attached,
                token,
            } => {
                tracing::debug!(
                    slot,
                    from = self.pf_pos[slot],
                    frontier,
                    "rank group prefill chunk"
                );
                self.pf_pos[slot] = frontier;
                self.attached[slot] = attached as usize;
                self.logits_at.fill(None);
                if token.is_some() {
                    self.logits_at[slot] = Some(0);
                }
                Ok(token)
            }
            other => Err(RuntimeError::Device(format!(
                "unexpected prefill reply {other:?}"
            ))),
        }
    }
}

impl Drop for CudaGroupServe {
    fn drop(&mut self) {
        // Closing the links ends every worker's request loop.
        self.workers.clear();
    }
}

fn spawn_worker(
    exe: &Path,
    group: &Arc<ProcGroup>,
    rank: u32,
    device: &str,
    assets_dir: &Path,
    checkpoint_dir: &Path,
) -> Result<Worker> {
    let (ours, theirs) =
        UnixStream::pair().map_err(|e| RuntimeError::Device(format!("socketpair: {e}")))?;
    inheritable(theirs.as_raw_fd())?;
    let mut cmd = Command::new(exe);
    cmd.arg("rank-worker")
        .arg("--rank")
        .arg(rank.to_string())
        .arg("--group-fd")
        .arg(group.fd().to_string())
        .arg("--link-fd")
        .arg(theirs.as_raw_fd().to_string())
        .arg("--assets")
        .arg(assets_dir)
        .arg("--checkpoint")
        .arg(checkpoint_dir)
        .arg("--leader-argv")
        .arg(serde_json::to_string(&std::env::args().collect::<Vec<_>>()).unwrap_or_default())
        .env("CUDA_VISIBLE_DEVICES", device)
        .env_remove("PLOW_GROUP_DEVICES");
    let child = cmd
        .spawn()
        .map_err(|e| RuntimeError::Device(format!("spawn rank {rank} worker: {e}")))?;
    drop(theirs);
    ours.set_read_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| RuntimeError::Device(e.to_string()))?;
    ours.set_write_timeout(Some(RPC_TIMEOUT))
        .map_err(|e| RuntimeError::Device(e.to_string()))?;
    let child = Arc::new(Mutex::new(child));
    // A worker that exits for any reason fails the group at once: a dead child stays a zombie
    // until reaped, so the barrier's pid probe alone would not see it.
    let (watch, g) = (child.clone(), group.clone());
    std::thread::Builder::new()
        .name(format!("rank{rank}-watch"))
        .spawn(move || loop {
            let status = watch.lock().try_wait();
            match status {
                Ok(Some(s)) => {
                    tracing::error!(rank, %s, "rank worker exited");
                    g.abort();
                    return;
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(e) => {
                    tracing::error!(rank, error = %e, "rank worker probe failed");
                    g.abort();
                    return;
                }
            }
        })
        .map_err(|e| RuntimeError::Device(e.to_string()))?;
    Ok(Worker {
        link: Link::new(ours),
        child,
    })
}

impl Drop for Worker {
    fn drop(&mut self) {
        // End the worker's request loop, then give it a moment to unload cleanly.
        self.link.close();
        for _ in 0..100 {
            if matches!(self.child.lock().try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let mut c = self.child.lock();
        let _ = c.kill();
        let _ = c.wait();
    }
}

/// `plowrt rank-worker`: one rank of a group, driven by its leader over `link_fd`.
pub fn rank_worker_main(
    rank: u32,
    group_fd: i32,
    link_fd: i32,
    assets_dir: PathBuf,
    checkpoint_dir: PathBuf,
) -> Result<()> {
    // SAFETY: both descriptors were inherited from the leader for this process to own.
    let (gfd, sock) = unsafe {
        (
            OwnedFd::from_raw_fd(group_fd),
            UnixStream::from_raw_fd(link_fd),
        )
    };
    sock.set_read_timeout(Some(RPC_TIMEOUT))
        .and_then(|()| sock.set_write_timeout(Some(RPC_TIMEOUT)))
        .map_err(|e| RuntimeError::Device(e.to_string()))?;
    let mut link = Link::new(sock);
    // Die with the leader: a parent-death signal tracks the spawning THREAD (the leader loads
    // on a short-lived thread), so watch for reparenting instead.
    let leader = unsafe { libc::getppid() };
    std::thread::Builder::new()
        .name("rank-leader-watch".into())
        .spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_millis(250));
            if unsafe { libc::getppid() } != leader {
                // SAFETY: immediate exit; nothing of this process outlives its leader.
                unsafe { libc::_exit(1) };
            }
        })
        .map_err(|e| RuntimeError::Device(format!("leader watch thread: {e}")))?;
    let loaded = (|| -> Result<GpuEngine> {
        let layout = group_layout(&assets_dir)?.map(|(_, l)| l);
        let group = Arc::new(ProcGroup::attach(gfd, rank, layout)?);
        let be = Arc::new(CudaBackend::new(0)?);
        let (bind, _) = p2p::worker(be.clone(), group.clone(), &assets_dir, &mut link)
            .inspect_err(|_| group.abort())?;
        GpuEngine::load_rank(be, &assets_dir, &checkpoint_dir, Some(bind))
    })();
    let mut eng = match loaded {
        Ok(e) => e,
        Err(e) => {
            let _ = link.send(&Resp::Err(e.to_string()));
            return Err(e);
        }
    };
    link.send(&Resp::Ready {
        batch: eng.batch(),
        max_ctx: eng.max_ctx(),
    })?;
    loop {
        let req: Req = match link.recv() {
            Ok(r) => r,
            Err(_) => return Ok(()), // leader closed the link: shut down
        };
        let resp = execute(&mut eng, &req).unwrap_or_else(|e| {
            if let Some(b) = eng.rank_bind() {
                b.group().abort();
            }
            Resp::Err(e.to_string())
        });
        link.send(&resp)?;
    }
}

impl SeqEngine for CudaGroupServe {
    fn stop_ids(&self) -> &Arc<Vec<u32>> {
        &self.stop_ids
    }
    fn batch(&self) -> usize {
        self.batch
    }
    fn release(&mut self, slot: usize) {
        if slot < self.batch && (self.live[slot] || self.pf_pos[slot] > 0) {
            if let Err(e) = self.all(Req::Retire { slot, cache: true }) {
                tracing::warn!(slot, error = %e, "rank group retire failed");
            }
        }
        if slot < self.batch {
            self.live[slot] = false;
            self.pf_pos[slot] = 0;
            self.attached[slot] = 0;
        }
    }
    fn prefill_turn(&self) -> usize {
        self.prefill_turn % self.batch.max(1)
    }
    fn advance_prefill_turn(&mut self, slot: usize) {
        self.prefill_turn = (slot + 1) % self.batch.max(1);
    }
    fn prefill_prog_t(&self, _prog: usize) -> Option<u32> {
        None
    }
    fn packable_prefill_span(
        &self,
        _slot: usize,
        _max_rows: u32,
    ) -> Option<packet::dev::PrefillSpan> {
        None
    }
    fn advance_packed_prefill(&mut self, _members: &[(usize, &[u32])]) -> Result<()> {
        Err(RuntimeError::Rejected(
            "packed prefill is not served by a rank group".into(),
        ))
    }
    fn prefill_frontier(&self, slot: usize) -> Option<usize> {
        self.pf_pos.get(slot).copied()
    }
    fn host_sampling(&self) -> bool {
        // Rank 0 draws sampled rows from its logits; the token reaches every rank as the
        // next step's feed.
        true
    }
    fn cached_rows(&self, slot: usize) -> usize {
        self.attached.get(slot).copied().unwrap_or(0)
    }
    fn logits_row(&self, slot: usize, out: &mut Vec<f32>) -> bool {
        match self.logits_at.get(slot).copied().flatten() {
            Some(row) => self.eng.lock().logits_row(row, out).is_ok(),
            None => false,
        }
    }
    fn prefill_chunked_at_most(
        &mut self,
        slot: usize,
        prompt: &[u32],
        tick_max_bucket: u32,
    ) -> Result<Option<u32>> {
        let cap = if tick_max_bucket == u32::MAX {
            usize::MAX
        } else {
            tick_max_bucket as usize
        };
        self.prefill_chunk(slot, prompt, cap)
    }
    fn multistep_quantum(&self, _feeds: &[(usize, u32)], _requested: usize) -> Option<usize> {
        None
    }
    fn multi_step(
        &mut self,
        _feeds: &[(usize, u32)],
        _q: usize,
        _out: &mut Vec<u32>,
    ) -> Result<usize> {
        Err(RuntimeError::Rejected(
            "multi-step is not served by a rank group".into(),
        ))
    }
    fn step_batch(&mut self, feeds: &[(usize, u32)]) -> Result<Vec<(usize, u32)>> {
        if feeds.is_empty() {
            return Ok(Vec::new());
        }
        match self.all(Req::Step {
            feeds: feeds.to_vec(),
        })? {
            Resp::Tokens(t) if t.len() == feeds.len() => {
                self.logits_at.fill(None);
                for &(slot, _) in feeds {
                    self.logits_at[slot] = Some(slot);
                }
                Ok(feeds.iter().map(|f| f.0).zip(t).collect())
            }
            other => Err(RuntimeError::Device(format!(
                "unexpected step reply {other:?}"
            ))),
        }
    }
}
