//! Tensor-parallel serving on CUDA: rank 0's [`GpuEngine`] leads, the other ranks' engines run on
//! follower threads and replay every state-changing call it makes, in order.
//!
//! Every launch meets its peers in the collectives, so all ranks enter it with zeroed cross-GPU
//! counters: each rank zeroes its own xctr once its last launch drained, barrier, launch (as
//! `block_run tp-check`). A launching call therefore always passes the barrier on every rank, even
//! one that is about to fail, or the others would wait for it forever.

use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;

use packet::dev::DevOp;

use super::{GpuEngine, NvTpBind, PrefillStep};
use crate::device::cuda::CudaBackend;
use crate::exec::tp::{PeerLayout, TpGroup};
use crate::{Result, RuntimeError};

enum Cmd {
    Begin(usize, usize),
    Retire(usize, bool),
    Prefill(usize, Vec<u32>, usize),
    Step(Vec<(usize, u32)>),
}

struct Follower {
    tx: Option<Sender<Cmd>>,
    done: Receiver<Result<()>>,
    thread: Option<JoinHandle<()>>,
}

pub(super) struct TpLead {
    group: Arc<TpGroup>,
    barrier: Arc<Barrier>,
    followers: Vec<Follower>,
}

impl TpLead {
    fn post(&self, cmd: impl Fn() -> Cmd) {
        for f in &self.followers {
            // A follower that has exited reports its error at the next collect.
            let _ = f.tx.as_ref().expect("follower channel").send(cmd());
        }
    }

    fn fence(&self) -> Result<()> {
        fence(&self.group, 0, &self.barrier)
    }

    fn collect(&self) -> Result<()> {
        let mut first = Ok(());
        for (r, f) in self.followers.iter().enumerate() {
            let got = f
                .done
                .recv()
                .unwrap_or_else(|_| Err(RuntimeError::Device(format!("tp rank {} exited", r + 1))));
            if first.is_ok() {
                first = got;
            }
        }
        first
    }

    /// Broadcast a launching call, fence, run rank 0's own, then wait for every follower.
    fn launch<T>(&self, cmd: impl Fn() -> Cmd, own: impl FnOnce() -> Result<T>) -> Result<T> {
        self.post(cmd);
        let fenced = self.fence();
        let out = fenced.and_then(|()| own());
        let peers = self.collect();
        let out = out?;
        peers?;
        Ok(out)
    }
}

impl Drop for TpLead {
    fn drop(&mut self) {
        for f in &mut self.followers {
            f.tx.take();
        }
        for f in &mut self.followers {
            if let Some(t) = f.thread.take() {
                let _ = t.join();
            }
        }
    }
}

fn fence(group: &TpGroup, rank: usize, barrier: &Barrier) -> Result<()> {
    let zeroed = group.zero_rank_xctr(rank);
    barrier.wait();
    zeroed
}

fn follow(
    mut e: GpuEngine,
    rank: usize,
    rx: Receiver<Cmd>,
    done: Sender<Result<()>>,
    group: Arc<TpGroup>,
    barrier: Arc<Barrier>,
) {
    // A failed non-launching call (begin) is reported at the next launch.
    let mut latched: Result<()> = Ok(());
    let mut toks = Vec::new();
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Begin(b, total) => {
                if let (Ok(()), Err(err)) = (&latched, e.begin_slot(b, total)) {
                    latched = Err(err);
                }
            }
            Cmd::Retire(b, cache) => e.retire_slot(b, cache),
            Cmd::Prefill(b, prompt, cap) => {
                let fenced = fence(&group, rank, &barrier);
                let r = std::mem::replace(&mut latched, Ok(()))
                    .and(fenced)
                    .and_then(|()| e.prefill_chunk(b, &prompt, cap).map(|_| ()));
                let _ = done.send(r);
            }
            Cmd::Step(feeds) => {
                let fenced = fence(&group, rank, &barrier);
                let r = std::mem::replace(&mut latched, Ok(()))
                    .and(fenced)
                    .and_then(|()| e.step_slots_sampled(&feeds, None, &mut toks));
                let _ = done.send(r);
            }
        }
    }
}

/// Cross-GPU counters the blob's collectives address (as `block_run tp-check` sizes them).
fn n_xctr(blob: &crate::asset::devblob::DevBlob) -> u32 {
    blob.progs
        .iter()
        .flat_map(|p| &p.insts)
        .filter_map(|d| match DevOp::from_u16(d.op) {
            Some(DevOp::XReduce) => Some(d.i[3]),
            Some(DevOp::XReduceTwoShot) => Some(d.i[3].max(d.i[4])),
            Some(DevOp::XArgmaxFin) => {
                Some(d.i[4] + packet::devbuild::xargmax_value_lines(d.i[1].max(1)).unwrap_or(1) - 1)
            }
            _ => None,
        })
        .max()
        .map_or(0, |g| g + 1)
}

impl GpuEngine {
    /// Load a tensor-parallel bundle on `bes` (rank r on `bes[r]`): the returned engine is rank 0,
    /// and drives the other ranks for every call the serve loop makes.
    pub fn load_tp_group(bes: &[Arc<CudaBackend>], assets_dir: &Path, checkpoint_dir: &Path) -> Result<GpuEngine> {
        let pkt = crate::asset::devblob::DevBlob::find_in_dir(assets_dir)?
            .ok_or_else(|| RuntimeError::Rejected(format!("no packet in {}", assets_dir.display())))?;
        let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io { path: pkt.clone(), source })?;
        let blob = crate::asset::devblob::DevBlob::parse(&raw)?;
        let tp = blob.tp.clone().ok_or_else(|| RuntimeError::Rejected("packet is not tensor-parallel".into()))?;
        if tp.n_gpu as usize != bes.len() {
            return Err(RuntimeError::Rejected(format!("packet is tp={} but {} devices were given", tp.n_gpu, bes.len())));
        }
        let max_tokens = (tp.slot_bytes / (u64::from(tp.hidden) * 2)) as u32;
        let layout = PeerLayout::new(tp.hidden, max_tokens, n_xctr(&blob))
            .ok_or_else(|| RuntimeError::Rejected("peer layout not 128 B aligned".into()))?;
        drop(raw);
        for a in bes {
            for b in bes {
                if !Arc::ptr_eq(a, b) {
                    a.enable_peer_access(b)?;
                }
            }
        }
        let group = TpGroup::bringup(bes.iter().map(|b| Arc::clone(b) as Arc<dyn crate::device::Backend>).collect(), layout)?;
        group.verify_peer_visibility()?;
        let binds: Vec<NvTpBind> = group
            .ranks()
            .iter()
            .map(|r| NvTpBind {
                rank: r.rank(),
                n_gpu: tp.n_gpu,
                peer_table: r.peer_scratch_table(),
                xctr: r.xctr(),
                scratch_base: r.scratch_base(),
                slot_b: tp.slot_bytes,
                slot_bytes: tp.slot_bytes,
            })
            .collect();
        let mut engines = std::thread::scope(|s| {
            let hs: Vec<_> = bes
                .iter()
                .zip(&binds)
                .map(|(be, bind)| {
                    let be = Arc::clone(be);
                    let bind = *bind;
                    s.spawn(move || GpuEngine::load_tp(be, assets_dir, checkpoint_dir, Some(bind)))
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err(RuntimeError::Device("tp load thread panicked".into()))))
                .collect::<Result<Vec<_>>>()
        })?;
        let barrier = Arc::new(Barrier::new(engines.len()));
        let group = Arc::new(group);
        let followers = engines
            .drain(1..)
            .enumerate()
            .map(|(i, e)| {
                let (tx, rx) = channel();
                let (dtx, done) = channel();
                let (group, barrier) = (Arc::clone(&group), Arc::clone(&barrier));
                let thread = std::thread::Builder::new()
                    .name(format!("plow-tp-rank{}", i + 1))
                    .spawn(move || follow(e, i + 1, rx, dtx, group, barrier))
                    .map_err(|e| RuntimeError::Device(format!("tp follower thread: {e}")))?;
                Ok(Follower { tx: Some(tx), done, thread: Some(thread) })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut lead = engines.pop().expect("rank 0");
        tracing::info!(n_gpu = tp.n_gpu, "tensor-parallel engine ready (rank 0 leads)");
        lead.tp_lead = Some(Box::new(TpLead { group, barrier, followers }));
        Ok(lead)
    }

    /// Whether this engine drives a tensor-parallel group (greedy tokens only: a rank's logits row
    /// is its vocab shard).
    pub fn is_tp(&self) -> bool {
        self.tp_lead.is_some()
    }

    pub(super) fn tp_post_begin(&self, b: usize, total: usize) {
        if let Some(t) = &self.tp_lead {
            t.post(|| Cmd::Begin(b, total));
        }
    }

    pub(super) fn tp_post_retire(&self, b: usize, cache: bool) {
        if let Some(t) = &self.tp_lead {
            t.post(|| Cmd::Retire(b, cache));
        }
    }

    pub fn prefill_chunk(&mut self, b: usize, prompt: &[u32], cap: usize) -> Result<PrefillStep> {
        match self.tp_lead.take() {
            None => self.prefill_chunk_local(b, prompt, cap),
            Some(t) => {
                let r = t.launch(|| Cmd::Prefill(b, prompt.to_vec(), cap), || self.prefill_chunk_local(b, prompt, cap));
                self.tp_lead = Some(t);
                r
            }
        }
    }

    pub fn step_slots_sampled(
        &mut self,
        feeds: &[(usize, u32)],
        specs: Option<&[super::DevSample]>,
        toks: &mut Vec<u32>,
    ) -> Result<()> {
        match self.tp_lead.take() {
            None => self.step_slots_sampled_local(feeds, specs, toks),
            Some(t) => {
                if feeds.is_empty() {
                    self.tp_lead = Some(t);
                    toks.clear();
                    return Ok(());
                }
                let r = t.launch(|| Cmd::Step(feeds.to_vec()), || self.step_slots_sampled_local(feeds, None, toks));
                self.tp_lead = Some(t);
                r
            }
        }
    }
}
