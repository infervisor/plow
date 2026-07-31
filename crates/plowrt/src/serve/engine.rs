//! The loaded device engine behind one served slug.
//!
//! `serve` used to hold `exec::gpu::GpuEngine` concretely, which made the whole
//! engine map — and therefore `plowrt serve` — CUDA-only. The two backends do
//! not differ in a way a trait would abstract cheaply: the CUDA engine is a
//! *slotted* engine (B independent sequences, chunked prefill, prefix sharing,
//! device sampling) while `exec::amd` is a *single-sequence* one (one KV ring,
//! one position, greedy on-device sampling). So the seam is an enum with
//! exactly the surface the mux needs from a backend BEFORE it commits to a
//! per-backend tick body: how many sequences it serves, and what its stop set
//! is. Everything richer stays inside the variant.
//!
//! What is deliberately NOT here: VMM/prefix sharing, the S1 `ModelManager`,
//! and multi-model residency. Those are CUDA-only today and stay CUDA-only —
//! an AMD serve is one model, B fixed sequence slots, no paging.

use std::sync::Arc;

/// The device engine serving one slug.
pub enum ServeEngine {
    /// The sm_120 persistent-interpreter engine (slotted, continuous batching).
    #[cfg(feature = "cuda")]
    Cuda(crate::exec::gpu::GpuEngine),
    /// The gfx950 engine (single sequence, optionally tensor-parallel).
    #[cfg(feature = "hsa")]
    Amd(AmdServe),
}

impl ServeEngine {
    /// Sequences one decode launch advances — the mux sizes its slot table to
    /// this, so mux slot `i` IS engine slot `i`.
    pub fn batch(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(e) => e.batch(),
            #[cfg(feature = "hsa")]
            ServeEngine::Amd(e) => e.batch(),
        }
    }

    /// The checkpoint's stop-token set.
    pub fn stop_ids(&self) -> &Arc<Vec<u32>> {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(e) => e.stop_ids(),
            #[cfg(feature = "hsa")]
            ServeEngine::Amd(e) => e.stop_ids(),
        }
    }

    /// The VMM prefix-cache stats reader, when this engine has one. Only the
    /// CUDA engine does.
    #[cfg(feature = "cuda")]
    pub fn vmm_stats_handle(&self) -> Option<crate::memory::vmm::VmmStatsHandle> {
        match self {
            ServeEngine::Cuda(e) => e.vmm_stats_handle(),
            #[cfg(feature = "hsa")]
            ServeEngine::Amd(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// AMD / gfx950
// ---------------------------------------------------------------------------

#[cfg(feature = "hsa")]
pub use amd_serve::AmdServe;

#[cfg(feature = "hsa")]
mod amd_serve {
    use std::path::Path;
    use std::sync::Arc;

    use crate::exec::amd::AmdEngine;
    use crate::exec::amd_tp::AmdTpGroup;
    use crate::{Result, RuntimeError};

    /// One or N ranks of a gfx950 model, driven as ONE sequence.
    ///
    /// `AmdTpGroup::load` refuses `n < 2` (the peer region and the collectives
    /// only exist in a sharded packet), so tp==1 is `AmdEngine` directly rather
    /// than a degenerate group.
    enum Ranks {
        One(AmdEngine),
        Tp(AmdTpGroup),
    }

    /// The gfx950 serving engine: `B` independent sequence slots sharing one
    /// decode dispatch.
    ///
    /// `B` is the compiled `PLOW_DECODE_BATCH` and is fixed at emit — there is
    /// no paging and no eviction under memory pressure, because the cache is
    /// one flat `[B][kv_head][ring][hd]` allocation sized at compile time.
    /// "Eviction" here means only *slot release on completion*.
    ///
    /// Serving state is per slot: `pos[s]` is the next KV row sequence `s`
    /// writes, and `live[s]` says whether the slot holds a request. Every
    /// decode dispatch advances ALL `B` rows whether or not they are live —
    /// the program's `t` is compiled, not passed — so an idle slot is fed
    /// `pos = 0, kvlen = 1, id = 0` and computes a throwaway token over KV row
    /// 0 of its own block. That is wasted work, and it is the reason a large
    /// `B` costs latency at low concurrency; it is not a correctness problem,
    /// because an idle slot's block is touched by nothing else.
    pub struct AmdServe {
        ranks: Ranks,
        stop_ids: Arc<Vec<u32>>,
        /// Sequences one decode dispatch advances (compiled `PLOW_DECODE_BATCH`).
        batch: usize,
        /// Next KV row each slot writes.
        pos: Vec<u32>,
        /// Whether each slot holds a live request.
        live: Vec<bool>,
        /// The token each slot feeds into the next dispatch. Idle slots feed 0.
        next_id: Vec<u32>,
        /// The packet declares exactly one program, so there is no prefill
        /// bucket ladder to chunk a prompt over and the prompt is walked
        /// through the decode program one token at a time. GLM-5.2 is this
        /// shape: `glm_emit_full` emits no grouped block-fp8 MoE prefill.
        decode_only: bool,
        max_ctx: usize,
        /// PREFIX CACHE (`PLOW_PREFIX_CACHE=1`, TP only, off by default).
        ///
        /// Per slot: the last prompt it prefilled, and the token offset at which its carried
        /// recurrent state is snapshotted. Invariant: the snapshot corresponds to
        /// `cached_prompt[..snap_at]`, so a new prompt matching that span may resume from it.
        prefix_cache: bool,
        cached_prompt: Vec<Vec<u32>>,
        snap_at: Vec<u32>,
        /// CHUNKED PREFILL cursor per slot. `Some` means this slot is mid-prefill: the mux has
        /// run some of its chunks and will run one more per tick, letting every other slot decode
        /// in between. `PLOW_PF_NO_CHUNK=1` restores whole-prompt-per-tick.
        pf: Vec<Option<PfCursor>>,
        chunk_prefill: bool,
    }

    /// A prompt part-way through its prefill.
    struct PfCursor {
        prompt: Vec<u32>,
        steps: Vec<crate::exec::amd::ChunkStep>,
        next: usize,
        /// Rows already written. A parked row still takes a decode dispatch's KV write, so it is
        /// pointed HERE — the row the next chunk is about to overwrite anyway.
        frontier: u32,
        /// Take a prefix snapshot after this step index. `None` on a cache hit (the snapshot it
        /// resumed from is still valid) and when the cache is off.
        snap_after: Option<usize>,
        /// Where the prefix cache resumed from, for the bookkeeping after the last chunk.
        resume: u32,
        arm: u32,
    }

    /// Shortest prefix worth caching. Below this the snapshot/restore pair costs more than the
    /// prefill it skips, and it churns the slot's cached prompt for nothing.
    const MIN_PREFIX: u32 = 128;

    fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    impl AmdServe {
        /// Bring up every rank of `blob`.
        ///
        /// The TP degree is read off the PACKET (`DevBlob::tp`), not off a host
        /// flag — a host that disagrees with the program about the shard count
        /// binds a quarter of a weight it needed all of. Backend index IS rank,
        /// so under a `gpulease` the visible ordinals `0..n` are the leased
        /// cards.
        pub fn load(
            blob_path: &Path,
            hsaco_dir: &Path,
            checkpoint: Option<&Path>,
        ) -> Result<Self> {
            let raw = std::fs::read(blob_path).map_err(|e| {
                RuntimeError::Device(format!("read {}: {e}", blob_path.display()))
            })?;
            let n_gpu = crate::asset::devblob::DevBlob::parse(&raw)?
                .tp
                .map(|t| t.n_gpu)
                .unwrap_or(1)
                .max(1);
            drop(raw);

            let stop_ids = Arc::new(
                checkpoint
                    .map(|d| {
                        let mut ids = crate::asset::checkpoint::read_eos_ids(d);
                        // A structured chat turn can close before the sequence eos; without this
                        // the framing lands in the user's text. See `chat_stop_ids`.
                        ids.extend(crate::asset::checkpoint::chat_stop_ids(d, &ids));
                        ids
                    })
                    .unwrap_or_default(),
            );

            let ranks = if n_gpu == 1 {
                let be = Arc::new(crate::device::hsa::HsaBackend::new(0)?);
                Ranks::One(AmdEngine::load(be, blob_path, hsaco_dir, checkpoint)?)
            } else {
                let mut backends = Vec::with_capacity(n_gpu as usize);
                for d in 0..n_gpu {
                    backends.push(Arc::new(crate::device::hsa::HsaBackend::new(d as u8)?));
                }
                Ranks::Tp(AmdTpGroup::load(backends, blob_path, hsaco_dir, checkpoint)?)
            };

            let (n_programs, max_ctx, bound) = match &ranks {
                Ranks::One(e) => (e.n_programs(), e.max_ctx(), e.weights_bound()),
                Ranks::Tp(g) => (g.rank(0).n_programs(), g.max_ctx(), g.weights_bound()),
            };
            // TP + BATCH IS NOW SERVED. It was refused while
            // `AmdTpGroup::submit_decode` took ONE `(pos, kvlen)`: the rendezvous
            // ordering it documents (prepare all, zero all, launch all, nothing
            // in between) had to be extended rank-wise before a per-slot array
            // could be threaded through it, and serving through the scalar path
            // would have put every rank at sequence 0's position — a wrong token
            // with no fault. `submit_decode_batched` prepares EVERY rank before
            // ANY rank launches, and `prefill_slot` rebases every rank onto the
            // slot for the duration of a collective prefill.
            //
            // The evidence, not the intention: `scripts/k3_batch_gate.sh` passes
            // at B=4 on K3 at TP8 — check A (four copies of one prompt give four
            // identical streams) and check B (four RAGGED prompts give the same
            // per-slot streams at a second batch width).
            let batch = match &ranks {
                Ranks::One(e) => e.batch(),
                Ranks::Tp(g) => g.rank(0).batch(),
            };
            if !bound {
                return Err(RuntimeError::Device(
                    "no checkpoint bound — the timings would be real and the TOKENS \
                     would not. Point PLOW_CHECKPOINT at the weights."
                        .into(),
                ));
            }
            tracing::info!(
                n_gpu,
                max_ctx,
                batch,
                decode_only = n_programs == 1,
                stop_ids = ?stop_ids,
                "AMD serve engine ready"
            );
            Ok(AmdServe {
                ranks,
                stop_ids,
                batch,
                pos: vec![0; batch],
                live: vec![false; batch],
                next_id: vec![0; batch],
                decode_only: n_programs == 1,
                max_ctx,
                prefix_cache: std::env::var("PLOW_PREFIX_CACHE").as_deref() == Ok("1"),
                cached_prompt: vec![Vec::new(); batch],
                snap_at: vec![0; batch],
                pf: (0..batch).map(|_| None).collect(),
                chunk_prefill: std::env::var("PLOW_PF_NO_CHUNK").as_deref() != Ok("1"),
            })
        }

        /// Sequence slots one decode dispatch advances. The mux sizes its slot
        /// table to this, so mux slot `i` IS engine slot `i`.
        pub fn batch(&self) -> usize {
            self.batch
        }

        pub fn stop_ids(&self) -> &Arc<Vec<u32>> {
            &self.stop_ids
        }

        pub fn max_ctx(&self) -> usize {
            self.max_ctx
        }

        /// Admit `prompt` into sequence slot `slot` and return its first
        /// generated token. Leaves `pos[slot]` at `prompt.len()`.
        ///
        /// One prefill occupies the WHOLE device — the prefill program is
        /// single-sequence and its dispatch is exclusive — so the other slots
        /// stall for its duration. That is the deliberate simplification versus
        /// the CUDA engine's chunked, interleaved prefill: a prompt's prefill is
        /// one tick, and it is why TTFT under load is bounded by the longest
        /// prompt in the batch rather than by a chunk.
        pub fn prefill(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
            if prompt.is_empty() {
                return Err(RuntimeError::Rejected("empty prompt".into()));
            }
            if prompt.len() >= self.max_ctx {
                return Err(RuntimeError::Rejected(format!(
                    "prompt is {} tokens, max_ctx is {}",
                    prompt.len(),
                    self.max_ctx
                )));
            }
            self.check_slot(slot)?;
            // Hand the slot to a new sequence. Two things happen: under
            // `PLOW_VMM_KV` the outgoing sequence's physical blocks are dropped
            // and row 0 re-mapped (the whole point of a growable cache), and —
            // on EVERY backend, always — the slot's carried KDA recurrence is
            // cleared.
            //
            // THE TP ARM USED TO BE MISSING, and it was not a small omission:
            // K3 serves at TP8, so the clear never ran at all and every request
            // after the first on a slot inherited its predecessor's recurrent
            // state across 69 of 93 layers. `AmdEngine::begin_slot` carries the
            // argument for why an append-only KV cache needs no clear and a
            // recurrence does.
            match &mut self.ranks {
                Ranks::One(e) => e.begin_slot(slot)?,
                Ranks::Tp(g) => g.begin_slot(slot)?,
            }
            self.pos[slot] = 0;
            self.live[slot] = true;
            let tok = if prompt.len() == 1 {
                // Nothing to consume: seed the single id and take one step,
                // which writes KV row 0 and samples the first token.
                self.next_id[slot] = prompt[0];
                self.dispatch(slot)?
            } else if self.decode_only {
                // No bucket ladder to chunk over — walk the prompt through the
                // decode program one token at a time. Step `p` writes KV row
                // `p` and attends over `[0, p+1)`, so nothing is read that was
                // not written. This is what `runtime/tests/glm52_decode.c`
                // does; it is O(prompt) dispatches, hence a fallback.
                //
                // At batch > 1 every OTHER slot also advances a row per
                // dispatch, so this holds their positions still and lets them
                // re-write the row they were going to write anyway. Nothing
                // reads a row before its own slot writes it, so the throwaway
                // K/V is overwritten before it can matter — see `dispatch`.
                let mut last = 0;
                for id in prompt {
                    self.next_id[slot] = *id;
                    last = self.dispatch(slot)?;
                }
                last
            } else {
                let (resume, arm) = self.plan_prefix(slot, prompt);
                let t = match &mut self.ranks {
                    Ranks::One(e) => e.prefill_slot(slot, prompt)?,
                    // `prefill_slot` and not `prefill`: the latter fills slot 0
                    // on every rank, so at batch > 1 every request would land in
                    // one slot's cache and the others would decode over rows
                    // nobody wrote. At batch 1 the two are the same call.
                    Ranks::Tp(g) => {
                        // PREFIX CACHE. `resume > 0` skips the shared prefix outright: the KV
                        // rows are already this slot's (same tokens, same positions) and the
                        // recurrence is restored from the snapshot. `arm > 0` splits this
                        // prefill so the NEXT request on the slot can resume — it prefills the
                        // same tokens either way, so a miss costs only the snapshot copy.
                        if resume > 0 || arm > 0 {
                            tracing::debug!(slot, resume, arm, n = prompt.len(), "amd: prefix cache");
                        }
                        let ids = if resume > 0 || arm > 0 {
                            g.prefill_slot_cached(slot, prompt, resume, arm)?
                        } else {
                            g.prefill_slot(slot, prompt)?
                        };
                        AmdTpGroup::agree(&ids)?
                    }
                };
                if self.prefix_cache {
                    // Invariant: the snapshot describes `cached_prompt[..snap_at]`. A hit leaves
                    // it valid (the new prompt agrees over that span); an arm replaces it;
                    // anything else makes it stale, so drop it.
                    self.cached_prompt[slot] = prompt.to_vec();
                    if arm > 0 {
                        self.snap_at[slot] = arm;
                    } else if resume == 0 {
                        self.snap_at[slot] = 0;
                    }
                }
                self.pos[slot] = prompt.len() as u32;
                t
            };
            Ok(tok)
        }

        /// Decide `(resume, arm)` for this slot's prefix cache.
        ///
        /// `resume` is a HIT: the slot's snapshot is at `snap_at`, and the incoming prompt agrees
        /// with the cached one over at least that span, so `[0, snap_at)` need not be prefilled
        /// at all. `arm` is a MISS that is worth arming: prefill splits at the common prefix so
        /// the next request can hit.
        ///
        /// Both are clamped to `len - 1`: a chunk with `clen == 0` would set the lm_head's
        /// `a_row0 = clen - 1` to `u32::MAX`, and an identical prompt must still produce a token.
        fn plan_prefix(&self, slot: usize, prompt: &[u32]) -> (u32, u32) {
            if !self.prefix_cache {
                return (0, 0);
            }
            let cap = prompt.len().saturating_sub(1) as u32;
            let lcp = (common_prefix_len(&self.cached_prompt[slot], prompt) as u32).min(cap);
            let snap = self.snap_at[slot];
            let armed = match &self.ranks {
                Ranks::One(_) => false,
                Ranks::Tp(g) => g.has_snapshot(slot),
            };
            if armed && snap > 0 && snap <= lcp {
                (snap, 0)
            } else if lcp >= MIN_PREFIX {
                (0, lcp)
            } else {
                (0, 0)
            }
        }

        /// Advance slot `slot`'s prefill by ONE CHUNK. `Ok(None)` means more chunks remain.
        ///
        /// This is what makes prefill yieldable. `AmdServe::prefill` runs a whole prompt in one
        /// call, so a 2-chunk prompt held the device for both chunks and every other slot's
        /// decode waited. Here the mux gets control back after each chunk.
        ///
        /// What makes it SAFE is the per-row parked mask. A decode dispatch between two chunks
        /// still advances all B rows, and for the mid-prefill slot two things must not happen:
        /// its recurrence must not move (it would destroy the prefix just built) and its KV must
        /// not be clobbered. The first is the mask — the slot is `live == false`, so
        /// `dispatch_all` publishes `parked = 1` for it. The second is `frontier`: the row is fed
        /// `pos = frontier`, the row the NEXT chunk overwrites anyway, which is exactly the
        /// "live slot not in `advance`" case `dispatch_all` already documents as sound.
        ///
        /// Falls back to the whole-prompt path for the shapes that have no chunk ladder to walk:
        /// single-GPU, `decode_only`, a 1-token prompt, or the prefix cache (whose split points
        /// are its own and are not the bucket plan).
        pub fn prefill_chunked(&mut self, slot: usize, prompt: &[u32]) -> Result<Option<u32>> {
            let plain_tp = matches!(self.ranks, Ranks::Tp(_))
                && self.chunk_prefill
                && !self.decode_only
                && prompt.len() > 1;
            if !plain_tp {
                return self.prefill(slot, prompt).map(Some);
            }
            if self.pf[slot].is_none() {
                if prompt.is_empty() {
                    return Err(RuntimeError::Rejected("empty prompt".into()));
                }
                if prompt.len() >= self.max_ctx {
                    return Err(RuntimeError::Rejected(format!(
                        "prompt is {} tokens, max_ctx is {}",
                        prompt.len(),
                        self.max_ctx
                    )));
                }
                self.check_slot(slot)?;
                match &mut self.ranks {
                    Ranks::One(e) => e.begin_slot(slot)?,
                    Ranks::Tp(g) => g.begin_slot(slot)?,
                }
                let n = prompt.len() as u32;
                // CHUNKING AND THE PREFIX CACHE COMPOSE. The cache decides WHICH span still has
                // to be prefilled; chunking decides how that span is broken into ticks. Building
                // the cursor from the cached plan is all it takes — they were alternatives only
                // because the first cut of this function bailed out when the cache was on.
                let (resume, arm) = self.plan_prefix(slot, prompt);
                let (steps, snap_after) = match &mut self.ranks {
                    Ranks::Tp(g) => {
                        if resume > 0 {
                            g.restore_carried(slot)?;
                            (g.plan_span(resume, n)?, None)
                        } else if arm > 0 {
                            let head = g.plan_span(0, arm)?;
                            let tail = g.plan_span(arm, n)?;
                            let cut = head.len();
                            let mut all = head;
                            all.extend(tail);
                            (all, cut.checked_sub(1))
                        } else {
                            (g.plan_span(0, n)?, None)
                        }
                    }
                    Ranks::One(_) => unreachable!("gated on Tp above"),
                };
                self.pos[slot] = 0;
                // NOT live until the last chunk lands: `live` is what the mask keys on, and a
                // half-prefilled slot must stay parked.
                self.live[slot] = false;
                self.pf[slot] = Some(PfCursor {
                    prompt: prompt.to_vec(),
                    steps,
                    next: 0,
                    frontier: resume,
                    snap_after,
                    resume,
                    arm,
                });
            }
            let Ranks::Tp(g) = &mut self.ranks else {
                unreachable!("gated on Tp above")
            };
            let cur = self.pf[slot].as_mut().expect("just built");
            let step = cur.steps[cur.next];
            // Rebase for this chunk and hand the base back before returning: the decode that runs
            // later in this same tick refuses a non-zero base.
            g.kv_rebase_all(slot)?;
            let r = g.prefill_chunk(&cur.prompt, step);
            let restore = g.kv_rebase_all(0);
            r?;
            restore?;
            // Snapshot at the arm point, which is a CHUNK BOUNDARY of the head plan — so the
            // recurrence is exactly at `arm` when this fires.
            if cur.snap_after == Some(cur.next) {
                g.snapshot_carried(slot)?;
            }
            cur.next += 1;
            cur.frontier = step.c0 + step.clen;
            if cur.next < cur.steps.len() {
                return Ok(None);
            }
            let ids = g.read_sampled_all()?;
            let tok = AmdTpGroup::agree(&ids)?;
            let n = cur.prompt.len() as u32;
            let (resume, arm) = (cur.resume, cur.arm);
            let prompt_owned = std::mem::take(&mut cur.prompt);
            self.pf[slot] = None;
            self.pos[slot] = n;
            self.live[slot] = true;
            if self.prefix_cache {
                // Same invariant as the whole-prompt path: the snapshot describes
                // `cached_prompt[..snap_at]`, so a hit keeps it, an arm replaces it, and anything
                // else makes it stale.
                self.cached_prompt[slot] = prompt_owned;
                if arm > 0 {
                    self.snap_at[slot] = arm;
                } else if resume == 0 {
                    self.snap_at[slot] = 0;
                }
            }
            Ok(Some(tok))
        }

        /// Feed `id` into slot `slot` and produce its next token.
        ///
        /// `id` is seeded explicitly rather than relying on the device argmax
        /// left in `in.ids`: for a greedy row the two are the same value, and
        /// for a host-chosen one (a resumed stream, a sampler that did not pick
        /// the argmax) only the explicit seed is right.
        ///
        /// Skipping the upload when it is a provable no-op was tried and
        /// MEASURED: GLM-5.2 TP4 through the endpoint went 36.990 -> 36.970
        /// ms/token, i.e. nothing. The ~2.4 ms serving premium over `amd-bench`
        /// is not this upload; do not re-propose it.
        pub fn step(&mut self, slot: usize, id: u32) -> Result<u32> {
            self.check_slot(slot)?;
            if self.pos[slot] as usize >= self.max_ctx {
                return Err(RuntimeError::Rejected(format!(
                    "position {} past max_ctx {}",
                    self.pos[slot], self.max_ctx
                )));
            }
            self.next_id[slot] = id;
            self.dispatch(slot)
        }

        /// Advance EVERY live slot by one token in ONE dispatch.
        ///
        /// `feeds` is `(slot, id)` for each slot that has a token to consume.
        /// Returns `(slot, sampled)` in the same order. This is the whole point
        /// of the slotted engine: N sequences amortise one read of the weights.
        pub fn step_batch(&mut self, feeds: &[(usize, u32)]) -> Result<Vec<(usize, u32)>> {
            for &(s, id) in feeds {
                self.check_slot(s)?;
                if self.pos[s] as usize >= self.max_ctx {
                    return Err(RuntimeError::Rejected(format!(
                        "slot {s} position {} past max_ctx {}",
                        self.pos[s], self.max_ctx
                    )));
                }
                self.next_id[s] = id;
            }
            let advance: Vec<usize> = feeds.iter().map(|&(s, _)| s).collect();
            let out = self.dispatch_all(&advance)?;
            Ok(feeds.iter().map(|&(s, _)| (s, out[s])).collect())
        }

        /// Free a slot. There is no cache to reclaim — the block is fixed and
        /// preallocated — so this only stops the slot being fed and lets
        /// admission reuse it. The next request rewrites every row it reads.
        pub fn release(&mut self, slot: usize) {
            if slot < self.batch {
                self.live[slot] = false;
                self.pos[slot] = 0;
                self.next_id[slot] = 0;
                // Drop any half-finished prefill: the client is gone, and a stale cursor would
                // resume someone else's prompt into this slot.
                self.pf[slot] = None;
            }
        }

        fn check_slot(&self, slot: usize) -> Result<()> {
            if slot >= self.batch {
                return Err(RuntimeError::Rejected(format!(
                    "slot {slot} past engine batch {}",
                    self.batch
                )));
            }
            Ok(())
        }

        /// One dispatch that advances only `slot`, returning its token.
        fn dispatch(&mut self, slot: usize) -> Result<u32> {
            Ok(self.dispatch_all(&[slot])?[slot])
        }

        /// ONE decode dispatch advancing all `batch` rows on the device;
        /// returns every row's sampled id, indexed by slot. Only the slots in
        /// `advance` have their host position stepped.
        ///
        /// Every row runs whether or not its slot is live — the program's `t`
        /// is compiled, not passed — so two kinds of row do throwaway work:
        ///
        /// * an IDLE slot, fed `pos = 0, kvlen = 1, id = 0`. It writes K/V into
        ///   row 0 of its own block, which is sound because an admitted request
        ///   restarts at `pos = 0` and rewrites row 0 before reading it.
        /// * a LIVE slot not in `advance` (a slot waiting while another slot's
        ///   decode-only prompt walk runs). It rewrites row `pos[s]` — the row
        ///   it is about to write for real — and nothing reads `[0, pos[s]+1)`
        ///   until it does. Its `pos` does not move, so its stream is unaffected.
        ///
        /// The cost is wasted work, and it is why a blob compiled at a large
        /// `PLOW_DECODE_BATCH` is slower at concurrency 1 than one compiled at 1.
        fn dispatch_all(&mut self, advance: &[usize]) -> Result<Vec<u32>> {
            if self.batch == 1 {
                let pos = self.pos[0];
                use crate::obs::dstep;
                let t = match &mut self.ranks {
                    Ranks::One(e) => {
                        dstep::timed(&dstep::SEED, || e.seed_ids(&self.next_id))?;
                        e.decode_step(pos, pos + 1)?
                    }
                    // The split pair, not `decode_step`, because this is the
                    // server and the split exists for it. NOTHING sits between
                    // the two calls, and that is measured rather than pending:
                    // §DSTEP puts the whole host phase at 1.4% of the token and
                    // the only part of it that is safe to move at all at 0.02%.
                    // The argument, with the numbers, is in `exec::amd_tp`'s
                    // module doc — read it before putting work here.
                    Ranks::Tp(g) => {
                        dstep::timed(&dstep::SEED, || g.seed_ids(&self.next_id))?;
                        g.submit_decode(pos, pos + 1)?;
                        let ids = g.complete_decode()?;
                        dstep::timed(&dstep::AGREE, || AmdTpGroup::agree(&ids))?
                    }
                };
                for &s in advance {
                    self.pos[s] += 1;
                }
                return Ok(vec![t]);
            }
            let (mut p, mut k) = (Vec::with_capacity(self.batch), Vec::with_capacity(self.batch));
            for s in 0..self.batch {
                // A slot MID-CHUNKED-PREFILL is not live, but it must not be fed row 0 either:
                // its rows [0, frontier) are real prefilled KV and a throwaway write at row 0
                // would clobber them. Point it at `frontier` — the row its NEXT chunk overwrites
                // anyway — which is the "live slot not in `advance`" case documented below. Its
                // recurrence is protected separately, by the parked mask.
                let (pp, kk) = match (&self.pf[s], self.live[s]) {
                    (Some(c), _) => (c.frontier, c.frontier + 1),
                    (None, true) => (self.pos[s], self.pos[s] + 1),
                    (None, false) => (0, 1),
                };
                p.push(pp);
                k.push(kk);
            }
            // PER-ROW PARKED MASK. A dispatch advances all B rows because `t` is compiled, which
            // is harmless for the append-only KV cache and NOT harmless for the KDA recurrence:
            // it reads and writes `state[row]` unconditionally, so an idle row's recurrence was
            // being advanced by a throwaway token every single step. Parking those rows is both
            // a correctness improvement and less work.
            //
            // On a blob without `in.parked` this is a no-op, so a non-batched packet is unchanged.
            let parked: Vec<u32> = (0..self.batch)
                .map(|s| u32::from(!self.live[s]))
                .collect();
            let out = match &mut self.ranks {
                Ranks::One(e) => {
                    e.upload_parked(&parked)?;
                    e.seed_ids(&self.next_id)?;
                    e.decode_step_batched(&p, &k)?
                }
                Ranks::Tp(g) => {
                    g.upload_parked(&parked)?;
                    g.seed_ids(&self.next_id)?;
                    g.decode_step_batched(&p, &k)?
                }
            };
            for &s in advance {
                self.pos[s] += 1;
            }
            Ok(out)
        }
    }
}
