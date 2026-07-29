//! Tensor-parallel AMD driver — N [`AmdEngine`] ranks stepped as one.
//!
//! This is the host half of plow's inline collective. The device half already
//! exists: `interp.hip` runs `XReduce`/`XReduceTwoShot` as ordinary counter-gated
//! packets inside the persistent megakernel, so a collective costs no launch and
//! no host sync. What the host owes it is exactly two things, and getting either
//! wrong is a wrong token rather than a slow one:
//!
//! 1. **Zero every rank's `xctr` before dispatching ANY rank** (§6d). Zeroing
//!    rank-by-rank as each is launched lets an early rank signal a late rank's
//!    counter and then have that rank's own zeroing wipe the signal — the late
//!    rank waits forever.
//! 2. **Dispatch all ranks, then drain all ranks.** The ranks rendezvous on the
//!    device, through peer-mapped counters, inside their own dispatches. A host
//!    wait between two ranks' launches makes rank 0 spin on a partial that
//!    rank 1 has not been dispatched to produce.
//!
//! # Decode and prefill do NOT have the same shape
//!
//! Decode is one dispatch per rank, so "launch all, drain all" is the whole
//! story. Prefill is SEGMENTED, and there the obvious generalisation — let each
//! rank enqueue all of its segments, then drain everyone — is wrong.
//! `runtime/tests/tp_decode.c` records the failure verbatim:
//!
//! > Per-rank-all-segments let the ranks desync — a lagging rank made peers time
//! > out and bail, giving a WRONG, 100x-slow reduction at TP>=4.
//!
//! A class-8 segment holds both of a layer's all-reduces. The inline gate
//! rendezvouses in ~0.3 µs only if every rank is inside that segment at the same
//! time; if one rank is three segments behind, its peers spin to the
//! `PLOW_XCTR_DEADLINE_TICKS` deadline (1 s) and give up. So prefill goes
//! **per-segment, all-ranks, with a host barrier between segments** — see
//! [`AmdTpGroup::prefill`]. The barrier costs one drain per segment and buys the
//! rendezvous; it is not a conservatism that could be relaxed.
//!
//! # Status, measured on this node (Gemma-4 31B bf16, gfx950 x8, 2026-07-27)
//!
//! **DECODE is correct and token-identical across TP1/TP2/TP4** and to
//! `tp_decode.c --tp N --verify` on the same packet. Seeded with id 2, all three
//! degrees and the C oracle produce
//! `236773 236773 149814 237308 237323 237323 237323 237323`, and the oracle
//! additionally confirms rank-0's device argmax equals a host recompute. Since
//! TP1 binds whole tensors and TP4 binds quarters, an identical stream is
//! evidence both that the shards are right and that the collectives ran.
//!
//! **PREFILL is correct too, after a device fix.** It disagreed across ranks —
//! and so did `tp_decode.c --prefill` on the same packet, weights and objects,
//! which located the fault in the DEVICE two-shot rather than in either host.
//! The bug was in `d_xreduce_twoshot_mega`'s second rendezvous: workgroup 0
//! signalled `gate_ag` after only its OWN `__syncthreads()`, while the reduced
//! slice it was announcing is written collaboratively by all `blocks`
//! workgroups, so peers' all-gather read a half-reduced slice. Fixed by having
//! every workgroup signal, with the threshold raised to `n_gpu * blocks`.
//!
//! With that, prompt `2,106,1645` prefills to **537 on TP1, TP2 and TP4 and on
//! the C oracle at both degrees**, and the following 8 decode tokens are
//! identical across all three degrees.
//!
//! Two things that made this hard to see, worth keeping: the one-shot has no
//! such duty (it writes to LOCAL `out` that no peer reads), which is exactly why
//! decode was unaffected and prefill was not; and every gate still reached its
//! expected count, so a host audit that only checked "did everyone arrive"
//! could not see it — the arrivals were real, they were just early.
//!
//! # What this does not do — and how it is shaped for what comes next
//!
//! `plowrt serve` is out of scope here: `serve/mod.rs` gates its engine map on
//! `#[cfg(feature = "cuda")]`, so there is no AMD serve path to hang a TP group
//! off yet. The entry point is `plowrt amd-bench --tp N`, which reproduces
//! `tp_decode.c` token-for-token — a CORRECTNESS oracle. Under the benchmark law
//! (`plans/knob-contract.md` §0-BENCH) any number from it is a **bring-up**
//! number: headline comparisons come from `vllm bench serve` against the plowrt
//! endpoint, same client for both engines, and this binary is banned from a
//! table next to a vLLM number.
//!
//! That is why the API is split rather than blocking. A server must be able to
//! dispatch a token and then go do something else, so the step is
//! [`AmdTpGroup::submit_decode`] + [`AmdTpGroup::complete_decode`], with nothing
//! synchronous in between and no assumption that one caller owns the group for a
//! request's lifetime. Prefill exposes [`AmdTpGroup::plan_for`] +
//! [`AmdTpGroup::prefill_chunk`] for the same reason: chunked prefill is what a
//! continuously-batching scheduler interleaves, so the chunk is the unit.
//! [`AmdTpGroup::decode_step`] and [`AmdTpGroup::prefill`] are conveniences over
//! those, for the bench.
//!
//! # Where the margin is expected to come from
//!
//! The inline collective measures **20.6x RCCL at TP4** (0.626 µs vs 12.92 µs),
//! which is the largest single structural edge in the design — but it cashes out
//! at **long context x high concurrency x TP**, where plow's constant overhead
//! amortises and vLLM's per-layer launch and collective costs multiply.
//! Concurrency 1 at 1k context is the hardest case for plow and the least
//! representative, so it is the wrong place to judge this module.

use std::path::Path;
use std::sync::Arc;

use crate::asset::devblob::DevBlob;
use crate::device::hsa::HsaBackend;
use crate::device::Backend;
use crate::exec::amd::{AmdEngine, ChunkStep, TpBind};
use crate::exec::tp::{PeerLayout, TpGroup, XctrReset};
use crate::{Result, RuntimeError};

/// N co-resident ranks of one sharded model.
pub struct AmdTpGroup {
    /// Peer buffers, counter regions, and the launch discipline.
    group: TpGroup,
    /// One engine per rank, in rank order.
    ranks: Vec<AmdEngine>,
    reset: XctrReset,
    /// Per-program expected cross-GPU gate counts — see [`gate_expectations`].
    /// `None` for an id that is a peer-visible DATA slot rather than a counter.
    gate_expect: Vec<Vec<Option<u32>>>,
    /// Check every step that no collective timed out —
    /// [`TpGroup::audit_xctr`]. On by default: the failure it catches is a
    /// SILENTLY wrong token, and one 12 KiB readback per rank is a cheap
    /// premium against that. `PLOW_TP_NO_AUDIT=1` turns it off for a timing run.
    audit: bool,
}

impl AmdTpGroup {
    /// Bring up every rank of a sharded blob.
    ///
    /// `backends` are the group's devices, and a backend's INDEX is its rank —
    /// which is not its device ordinal when a node hosts two TP4 replicas (see
    /// [`TpGroup::split_replicas`]).
    ///
    /// The peer region is sized from the BLOB, not from a flag: `hidden` and the
    /// partial-slot offset come from the collectives `devgen` emitted, so the
    /// host cannot disagree with the program about where a partial lives. The
    /// counter count is `n_xctr` — one gate per one-shot `XReduce`, two per
    /// `XReduceTwoShot` — recovered the same way, because a region sized for
    /// one-shot and fed a two-shot program has its last collectives signalling
    /// past the end of `xctr`, into whatever the pool handed out next, with no
    /// fault.
    pub fn load(
        backends: Vec<Arc<HsaBackend>>,
        blob_path: &Path,
        hsaco_dir: &Path,
        checkpoint: Option<&Path>,
    ) -> Result<Self> {
        let n_gpu = backends.len() as u32;
        if n_gpu < 2 {
            return Err(RuntimeError::Device(format!(
                "a TP group needs at least 2 devices, got {n_gpu} — use AmdEngine::load \
                 for one"
            )));
        }

        // Parse the blob once, on the host, purely to size the peer region.
        let raw = std::fs::read(blob_path).map_err(|e| {
            RuntimeError::Device(format!("read {}: {e}", blob_path.display()))
        })?;
        let blob = DevBlob::parse(&raw)?;
        let tp = blob.tp.ok_or_else(|| {
            RuntimeError::Device(format!(
                "this packet carries no collective, so it is compiled for a single GPU, \
                 but {n_gpu} devices were requested. Recompile: plowc ... --num-gpus \
                 {n_gpu}"
            ))
        })?;
        if tp.n_gpu != n_gpu {
            return Err(RuntimeError::Device(format!(
                "packet is sharded for tp={} but {n_gpu} devices were given — the tensor \
                 slices and the peer partials only line up at tp={}",
                tp.n_gpu, tp.n_gpu
            )));
        }
        if tp.hidden == 0 {
            return Err(RuntimeError::Device(
                "packet has no one-shot XReduce, so its hidden size is unrecoverable — \
                 a decode program is required to size the peer region"
                    .into(),
            ));
        }

        // `max_tokens` is derived from the slot the PROGRAM uses, not assumed:
        // decode's partial is [1,H] but a prefill bucket's is [T,H], and devgen
        // sizes both slots to the largest chunk. Deriving it keeps the host's
        // region exactly as big as the offsets the program will address.
        let msg = tp.hidden as u64 * 2;
        if tp.slot_bytes == 0 || tp.slot_bytes % msg != 0 {
            return Err(RuntimeError::Device(format!(
                "packet's partial slot is {} B, not a multiple of hidden*2 = {msg} B",
                tp.slot_bytes
            )));
        }
        let max_tokens = (tp.slot_bytes / msg) as u32;
        let n_xctr = count_xgates(&blob);
        let gate_expect = gate_expectations(&blob, n_gpu, n_xctr);
        let layout = PeerLayout::new(tp.hidden, max_tokens, n_xctr).ok_or_else(|| {
            RuntimeError::Device(format!(
                "peer layout for hidden={} x {max_tokens} tokens is not 128 B-aligned",
                tp.hidden
            ))
        })?;
        tracing::info!(
            n_gpu, hidden = tp.hidden, max_tokens, n_xctr,
            peer_kib = layout.bytes() / 1024,
            "TP peer layout sized from the packet"
        );

        let dyn_backends: Vec<Arc<dyn Backend>> =
            backends.iter().map(|b| Arc::clone(b) as Arc<dyn Backend>).collect();
        let group = TpGroup::bringup(dyn_backends, layout)?;
        // All N*(N-1) directed pairs, byte-exact, BEFORE any weight is bound.
        // `agents_allow_access` REPLACES a buffer's allow-list, so the classic
        // failure leaves only the last-named rank mapped and every other peer
        // faulting at the first token — an hour of weight loading later.
        group.verify_peer_visibility()?;

        let mut ranks = Vec::with_capacity(backends.len());
        for (r, be) in backends.into_iter().enumerate() {
            let tr = group.rank(r as u32)?;
            let bind = TpBind {
                rank: r as u32,
                n_gpu,
                peer_table: tr.peer_scratch_table(),
                xctr: tr.xctr(),
                scratch_base: tr.scratch_base(),
                slot_b: tp.slot_bytes,
            };
            tracing::info!(rank = r, ordinal = tr.ordinal(), "binding rank");
            ranks.push(AmdEngine::load_rank(
                be, blob_path, hsaco_dir, checkpoint, Some(bind),
            )?);
        }

        Ok(AmdTpGroup {
            group,
            ranks,
            reset: XctrReset::Host,
            gate_expect,
            audit: std::env::var("PLOW_TP_NO_AUDIT").ok().as_deref() != Some("1"),
        })
    }

    pub fn n_gpu(&self) -> usize {
        self.ranks.len()
    }

    pub fn rank(&self, r: usize) -> &AmdEngine {
        &self.ranks[r]
    }

    pub fn rank_mut(&mut self, r: usize) -> &mut AmdEngine {
        &mut self.ranks[r]
    }

    pub fn max_ctx(&self) -> usize {
        self.ranks[0].max_ctx()
    }

    pub fn weights_bound(&self) -> bool {
        self.ranks.iter().all(|r| r.weights_bound())
    }

    /// One decode token across every rank. Returns each rank's sampled id.
    ///
    /// Every rank must return the SAME id: they all hold the full replicated
    /// residual stream and a full-vocab lm_head, so after the two all-reduces
    /// per layer their logits are identical. A disagreement means a collective
    /// did not happen — which is exactly the check `--verify` makes.
    ///
    /// This is [`AmdTpGroup::submit_decode`] followed by
    /// [`AmdTpGroup::complete_decode`]. A SERVER should call those two directly:
    /// the whole point of the split is that nothing between them is synchronous,
    /// so the scheduler can do its own work while N megakernels are in flight.
    pub fn decode_step(&mut self, pos: u32, kvlen: u32) -> Result<Vec<u32>> {
        self.submit_decode(pos, kvlen)?;
        self.complete_decode()
    }

    /// Dispatch one decode token on every rank and RETURN — no wait.
    ///
    /// The server-facing half. After this call N megakernels are resident and
    /// rendezvousing with each other through peer-mapped counters; the host owns
    /// nothing until [`AmdTpGroup::complete_decode`]. That is the property the
    /// design rests on, so it is expressed as two methods rather than hidden
    /// inside one blocking one: a caller that must not block cannot accidentally
    /// be given a caller that does.
    ///
    /// Ordering, all of which is load-bearing (§6d):
    /// 1. prepare + re-arm EVERY rank,
    /// 2. zero EVERY rank's `xctr`,
    /// 3. dispatch every rank, with nothing in between.
    ///
    /// Doing (2) per rank as each is launched lets an early rank signal a late
    /// rank's counter and then have that rank's own zeroing wipe the signal.
    pub fn submit_decode(&mut self, pos: u32, kvlen: u32) -> Result<()> {
        let dp = self.ranks[0].decode_prog();
        for e in &mut self.ranks {
            e.decode_prepare(pos, kvlen)?;
            e.rearm_prog(dp)?;
        }
        // `launch_token` owns zero-all-then-launch-all; the closure only says
        // what a launch IS.
        let ranks = &mut self.ranks;
        let mut i = 0usize;
        self.group.launch_token(self.reset, |_| {
            let e = &mut ranks[i];
            let k = e.decode_kernel();
            i += 1;
            e.enqueue(dp, k)
        })
    }

    /// Wait for an in-flight [`AmdTpGroup::submit_decode`] and collect the ids.
    ///
    /// Drains every rank BEFORE auditing or reading anything: a readback from
    /// rank 0 while rank 3 is still running would race the collective that rank
    /// 0's own result depends on.
    pub fn complete_decode(&mut self) -> Result<Vec<u32>> {
        for e in &self.ranks {
            e.drain()?;
        }
        if self.audit {
            let dp = self.ranks[0].decode_prog();
            self.group.audit_xctr(&self.gate_expect[dp])?;
        }
        self.ranks.iter_mut().map(|e| e.read_sampled()).collect()
    }

    /// Prefill `prompt` on every rank. Returns each rank's first sampled id.
    ///
    /// **PER-SEGMENT, ALL-RANKS, host barrier between segments** — not
    /// launch-all-segments-then-drain. From `tp_decode.c`:
    ///
    /// > Per-rank-all-segments let the ranks desync — a lagging rank made peers
    /// > time out and bail, giving a WRONG, 100x-slow reduction at TP>=4.
    ///
    /// The barrier guarantees every rank runs a class-8 segment's `XReduce`
    /// collectives CONCURRENTLY, so the inline system-scope gate rendezvouses
    /// immediately instead of spinning to its 1 s deadline. Note the failure it
    /// prevents is not a hang: the deadline expires, the op returns WITHOUT
    /// reducing, and the token is silently wrong.
    ///
    /// Counters are re-armed ONCE per chunk, never per segment: a segment's
    /// producers ran in an earlier launch, so re-zeroing between segments
    /// unsatisfies them and the next segment waits on a count that will never
    /// come again.
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(RuntimeError::Device("prefill of an empty prompt".into()));
        }
        if prompt.len() > self.max_ctx() {
            return Err(RuntimeError::Device(format!(
                "prompt of {} tokens exceeds max_ctx {}",
                prompt.len(),
                self.max_ctx()
            )));
        }
        // Every rank walks the SAME plan: the buckets are identical across
        // ranks (the shard is in the weights, not in the schedule), and a rank
        // that chunked differently would rendezvous with nobody.
        let t_plan = std::time::Instant::now();
        let steps = self.plan_for(prompt.len() as u32)?;
        crate::obs::ttft::PF_PLAN.add(t_plan.elapsed().as_nanos() as u64);
        if crate::obs::ttft::on() {
            crate::obs::ttft::set_cover(
                &self.ranks[0].plan_for(prompt.len() as u32).unwrap_or_default(),
            );
        }
        tracing::info!(tokens = prompt.len(), chunks = steps.len(), n_gpu = self.ranks.len(),
                       "TP prefill plan");

        for step in steps {
            self.prefill_chunk(prompt, step)?;
        }

        let t_read = std::time::Instant::now();
        let out = self.ranks.iter_mut().map(|e| e.read_sampled()).collect();
        crate::obs::ttft::PF_READ.add(t_read.elapsed().as_nanos() as u64);
        out
    }

    /// The chunk plan for a prompt — a server chunks prefill itself, so it needs
    /// the plan and [`AmdTpGroup::prefill_chunk`] rather than the whole loop.
    pub fn plan_for(&self, n_prompt: u32) -> Result<Vec<ChunkStep>> {
        let chunks = self.ranks[0].plan_for(n_prompt)?;
        self.ranks[0].chunk_steps(&chunks, n_prompt)
    }

    /// Run ONE prefill chunk across every rank, per-segment with a host barrier.
    ///
    /// Unlike decode this cannot be split into submit/complete, and the reason
    /// is the barrier itself: the ranks must be inside the SAME segment for
    /// their collectives to rendezvous, so the drain between segments is part of
    /// the algorithm, not a convenience. A server that wants to overlap prefill
    /// with other work overlaps whole CHUNKS, which is the granularity chunked
    /// prefill already schedules at.
    pub fn prefill_chunk(&mut self, prompt: &[u32], step: ChunkStep) -> Result<()> {
        use crate::obs::ttft;
        for e in &mut self.ranks {
            let t = std::time::Instant::now();
            e.prefill_prepare(prompt, step)?;
            ttft::PF_PREPARE.add(t.elapsed().as_nanos() as u64);
            let t = std::time::Instant::now();
            e.rearm_prog(step.prog)?;
            ttft::PF_REARM.add(t.elapsed().as_nanos() as u64);
        }
        // xctr once for the whole chunk, before any rank is dispatched.
        let t = std::time::Instant::now();
        self.group.zero_xctr()?;
        ttft::PF_XCTR.add(t.elapsed().as_nanos() as u64);

        let n_seg = self.ranks[0].prog_segments(step.prog);
        ttft::PF_SEGMENTS.tally(n_seg as u64);
        for seg in 0..n_seg {
            let t = std::time::Instant::now();
            for e in &mut self.ranks {
                e.enqueue_segment(step.prog, seg)?;
            }
            ttft::PF_ENQUEUE.add(t.elapsed().as_nanos() as u64);
            // THE BARRIER. Without it the ranks drift apart across segments and
            // the collectives inside a later segment miss each other.
            let t = std::time::Instant::now();
            for e in &self.ranks {
                e.drain()?;
            }
            let ns = t.elapsed().as_nanos() as u64;
            ttft::PF_DRAIN.add(ns);
            // Per-CHUNK, because the aggregate cannot separate a full bucket
            // from a padded one and that is the bucket-ladder question.
            if ttft::on() {
                let bucket = self.ranks[0].prog_t(step.prog);
                eprintln!(
                    "PF CHUNK T={bucket} c0={} clen={} seg={seg} drain={:.3} ms \
                     ({:.0} tok/s over the bucket, {:.0} over the real rows)",
                    step.c0,
                    step.clen,
                    ns as f64 / 1e6,
                    bucket as f64 / (ns as f64 / 1e9),
                    step.clen as f64 / (ns as f64 / 1e9),
                );
            }
        }
        if self.audit {
            self.group.audit_xctr(&self.gate_expect[step.prog])?;
        }
        Ok(())
    }

    /// Seed `in.ids` on every rank — needed once, before the first decode step.
    pub fn seed_ids(&mut self, ids: &[u32]) -> Result<()> {
        for e in &mut self.ranks {
            e.seed_ids(ids)?;
        }
        Ok(())
    }

    /// Every rank sampled the same token, or a report of who disagreed.
    ///
    /// This is the acceptance test, not a debug aid: identical streams across
    /// ranks is what proves the all-reduces actually ran. A rank that skipped
    /// its collective still produces fluent-looking ids from its own shard.
    pub fn agree(ids: &[u32]) -> Result<u32> {
        let first = *ids.first().ok_or_else(|| {
            RuntimeError::Device("no ranks sampled anything".into())
        })?;
        if let Some(r) = ids.iter().position(|&x| x != first) {
            return Err(RuntimeError::Device(format!(
                "RANKS DISAGREE: rank 0 sampled {first}, rank {r} sampled {} (all: \
                 {ids:?}) — a collective did not happen, or one rank bound the wrong \
                 shard",
                ids[r]
            )));
        }
        Ok(first)
    }
}

/// Cross-GPU gate ids a blob needs: one more than the largest gate id any
/// collective names.
///
/// Counted from the PROGRAM rather than from `layers * 2 * shots`, because
/// `devgen`'s `xgate` allocator hands out dense ids from 0 per program and the
/// arithmetic version has to know the layer count and the shot count and be
/// right about both. A region one gate short has the last collective signalling
/// past the end of `xctr`, with no fault.
///
/// `XArgmaxFin` counts too, and its omission was exactly that fault. The
/// sharded-lm_head fold (`GLM_SHARD_HEAD=1`) takes TWO ids from the same
/// per-program allocator — an arrival gate `i[3]` and an 8-byte peer-visible
/// VALUE slot `i[4]` (`op_collective.h:195`) — and they are allocated LAST, after
/// every layer collective. Counting only the reduces sized the region to the
/// reduces, so on a blob whose widest program is a prefill bucket the DECODE
/// fold landed inside the region by luck while the PREFILL fold wrote its
/// rendezvous and its published u64 past the end of `xctr` into the partial
/// slots. Measured on the GLM-5.2 TP4 stacked blob: `n_xctr = 312`, prefill fold
/// at 312/313, and the ranks sampled `[99419, 785, 99419, 785]`.
fn count_xgates(blob: &DevBlob) -> u32 {
    use packet::dev::DevOp;
    let mut top = 0u32;
    for p in &blob.progs {
        for d in &p.insts {
            if d.op == DevOp::XReduce as u16 {
                top = top.max(d.i[3] + 1); // one-shot: one gate, i3
            } else if d.op == DevOp::XReduceTwoShot as u16 {
                // two-shot: reduce-scatter (i3) and all-gather (i4)
                top = top.max(d.i[3] + 1).max(d.i[4] + 1);
            } else if d.op == DevOp::XArgmaxFin as u16 {
                // sharded-head fold: arrival gate (i3) and value slot (i4)
                top = top.max(d.i[3] + 1).max(d.i[4] + 1);
            }
        }
    }
    top
}

/// Per-program table of what each cross-GPU gate must read after a dispatch.
///
/// Gate ids are allocated densely from 0 **per program**, so the same id means
/// different things in different programs — decode's gate 1 is a one-shot
/// rendezvous, while a prefill bucket's gate 1 is a two-shot ALL-GATHER
/// rendezvous. One table cannot serve both, hence one per program.
///
/// The counts differ because the two rendezvous have different jobs:
///
/// * one-shot `XReduce`, and two-shot's reduce-scatter (`i[3]`) — **`n_gpu`**.
///   Each announces something already complete when the packet starts (the
///   producing GEMV's partial, ordered by the interpreter's local gate), so one
///   workgroup may speak for its whole rank.
/// * two-shot's all-gather (`i[4]`) — **`n_gpu * blocks`**. It announces the
///   reduced slice, which all `blocks` workgroups write collaboratively, so each
///   must signal for itself. This is the fix in `d_xreduce_twoshot_mega`; before
///   it, workgroup 0 signalled on behalf of workgroups still writing and peers
///   read a half-reduced slice.
///
/// * `XArgmaxFin`'s arrival gate (`i[3]`) — **`n_gpu`**. One thread per rank
///   signals every peer once (`op_collective.h`: `slice != 0 || threadIdx.x != 0`
///   returns), so it reads like a one-shot rendezvous.
///
/// A gate the program never uses expects `Some(0)`, which makes the audit a total
/// equality check rather than a range test — it then also catches a stale count
/// that survived a missing `zero_xctr`.
///
/// `None` means NOT A COUNTER. `XArgmaxFin`'s `i[4]` is a peer-visible u64 data
/// slot that happens to live in the counter region because a counter line is 128
/// aligned peer-visible bytes the host already zeroes; its low word is the
/// complement of the winning global vocab index, i.e. an arbitrary value. Auditing
/// it as an arrival count would fail every step on a correct run. Skipping it is
/// the honest answer, not a weakening: the value slot carries no completion
/// information, and the gate beside it does.
fn gate_expectations(blob: &DevBlob, n_gpu: u32, n_xctr: u32) -> Vec<Vec<Option<u32>>> {
    use packet::dev::DevOp;
    blob.progs
        .iter()
        .map(|p| {
            let mut e = vec![Some(0u32); n_xctr as usize];
            let mut set = |gate: u32, v: Option<u32>| {
                if let Some(slot) = e.get_mut(gate as usize) {
                    *slot = v;
                }
            };
            for d in &p.insts {
                if d.op == DevOp::XReduce as u16 {
                    set(d.i[3], Some(n_gpu));
                } else if d.op == DevOp::XReduceTwoShot as u16 {
                    set(d.i[3], Some(n_gpu));
                    set(d.i[4], Some(n_gpu * d.blocks as u32));
                } else if d.op == DevOp::XArgmaxFin as u16 {
                    set(d.i[3], Some(n_gpu));
                    set(d.i[4], None); // data, not a counter
                }
            }
            e
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rank agreement is the acceptance test, so its failure must name the rank
    /// and both ids rather than just returning false.
    #[test]
    fn disagreement_names_the_rank_and_both_ids() {
        assert_eq!(AmdTpGroup::agree(&[7, 7, 7, 7]).unwrap(), 7);
        let e = AmdTpGroup::agree(&[7, 7, 9, 7]).unwrap_err().to_string();
        assert!(e.contains("rank 2"), "{e}");
        assert!(e.contains('9') && e.contains('7'), "{e}");
        assert!(AmdTpGroup::agree(&[]).is_err());
    }

    /// The sharded-lm_head fold owns TWO xctr ids and they are allocated AFTER
    /// every layer collective, so a sizer that only counts reduces under-sizes
    /// the region by exactly the fold — and the fold then signals past the end
    /// of `xctr` with no fault. Measured: on the GLM-5.2 TP4 stacked blob the
    /// prefill fold landed at 312/313 against `n_xctr = 312` and the ranks
    /// sampled four different tokens.
    #[test]
    fn the_sharded_head_fold_is_sized_and_audited() {
        use crate::asset::devblob::DevProg;
        use packet::dev::{DevInst64, DevOp};
        let inst = |op: DevOp, i3: u32, i4: u32, blocks: u16| DevInst64 {
            op: op as u16,
            blocks,
            i: [0, 0, 0, i3, i4, 0, 0, 0],
            ..Default::default()
        };
        let prog = |insts: Vec<DevInst64>| DevProg {
            t: 1,
            n_counter: 0,
            insts,
            stream: Vec::new(),
            stream_ofs: Vec::new(),
            stream_len: Vec::new(),
            waits: Vec::new(),
            succs: Vec::new(),
            gq_stream: Vec::new(),
            gq_seg_ofs: Vec::new(),
            // 0 = not L2-domain-placed. This test is about xctr id COUNTING, not
            // placement; the field arrived with PLOW_L2_PLACE after this test was
            // written, and the merge that brought the two together was textual, so
            // only the compiler caught the gap.
            l2_domains: 0,
        };
        // A prefill bucket whose two-shot reduces top out at id 1, then the fold
        // at 2 (gate) and 3 (value slot).
        let blob = DevBlob {
            n_cu: 256,
            flags: 0,
            target: 0,
            tensors: Vec::new(),
            init: Vec::new(),
            kvrow: Vec::new(),
            progs: vec![prog(vec![
                inst(DevOp::XReduceTwoShot, 0, 1, 8),
                inst(DevOp::XArgmaxFin, 2, 3, 0),
            ])],
            sections: Vec::new(),
            gen: Vec::new(),
            tp: None,
        };
        assert_eq!(
            count_xgates(&blob),
            4,
            "the fold's value slot is the LAST id; sizing to the reduces alone \
             puts both fold ids outside the region"
        );
        let e = gate_expectations(&blob, 4, 4);
        assert_eq!(e[0][0], Some(4)); // reduce-scatter rendezvous
        assert_eq!(e[0][1], Some(32)); // all-gather: n_gpu * blocks
        assert_eq!(e[0][2], Some(4)); // fold arrival gate
        assert_eq!(e[0][3], None, "the fold's published u64 is data, not a count");
    }
}
