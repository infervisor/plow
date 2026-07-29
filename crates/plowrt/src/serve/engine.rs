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
                    .map(crate::asset::checkpoint::read_eos_ids)
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
            // TP STAYS SINGLE-SEQUENCE. Not because the kernels object — the
            // per-sequence KV row and RoPE angle come from `pos[t]` on every
            // rank alike — but because `AmdTpGroup::submit_decode` takes ONE
            // `(pos, kvlen)` and the rendezvous ordering it documents (prepare
            // all, zero all, launch all, nothing in between) has to be extended
            // rank-wise before a per-slot array can be threaded through it.
            // Serving a batched TP packet through this scalar path would put
            // every rank at sequence 0's position, which is a wrong token with
            // no fault. Refuse the combination instead.
            let batch = match &ranks {
                Ranks::One(e) => e.batch(),
                Ranks::Tp(_) => 1,
            };
            if matches!(&ranks, Ranks::Tp(g) if g.rank(0).batch() > 1) {
                return Err(RuntimeError::Device(
                    "this packet is compiled PLOW_DECODE_BATCH > 1 AND tensor-parallel; \
                     AmdTpGroup::submit_decode is still scalar, so the batch would \
                     silently collapse onto sequence 0. Emit the TP packet at \
                     PLOW_DECODE_BATCH=1."
                        .into(),
                ));
            }
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
            // Hand the slot to a new sequence: under `PLOW_VMM_KV` this drops
            // the outgoing sequence's physical blocks (the whole point of a
            // growable cache) and re-maps row 0. A no-op otherwise.
            if let Ranks::One(e) = &mut self.ranks {
                e.begin_slot(slot)?;
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
                let t = match &mut self.ranks {
                    Ranks::One(e) => e.prefill_slot(slot, prompt)?,
                    Ranks::Tp(g) => AmdTpGroup::agree(&g.prefill(prompt)?)?,
                };
                self.pos[slot] = prompt.len() as u32;
                t
            };
            Ok(tok)
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
                let t = match &mut self.ranks {
                    Ranks::One(e) => {
                        e.seed_ids(&self.next_id)?;
                        e.decode_step(pos, pos + 1)?
                    }
                    Ranks::Tp(g) => {
                        g.seed_ids(&self.next_id)?;
                        AmdTpGroup::agree(&g.decode_step(pos, pos + 1)?)?
                    }
                };
                for &s in advance {
                    self.pos[s] += 1;
                }
                return Ok(vec![t]);
            }
            let (mut p, mut k) = (Vec::with_capacity(self.batch), Vec::with_capacity(self.batch));
            for s in 0..self.batch {
                let live = self.live[s];
                p.push(if live { self.pos[s] } else { 0 });
                k.push(if live { self.pos[s] + 1 } else { 1 });
            }
            let out = match &mut self.ranks {
                Ranks::One(e) => {
                    e.seed_ids(&self.next_id)?;
                    e.decode_step_batched(&p, &k)?
                }
                // Refused at load; unreachable, and an error beats a panic.
                Ranks::Tp(_) => {
                    return Err(RuntimeError::Device(
                        "batched decode is not wired through AmdTpGroup".into(),
                    ))
                }
            };
            for &s in advance {
                self.pos[s] += 1;
            }
            Ok(out)
        }
    }
}
