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
//! (the benchmark law) any number from it is a **bring-up**
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
//! # The host/GPU overlap that is NOT worth building — measured, 2026-07-29
//!
//! [`AmdTpGroup::submit_decode`] and [`AmdTpGroup::complete_decode`] were split
//! so a server could work between them. `serve::engine` now calls the pair
//! rather than [`AmdTpGroup::decode_step`], so the split has a caller — but
//! **nothing is placed between them, and that is the measured conclusion, not an
//! omission.** `PLOW_DSTEP_LOG=1` prints the breakdown that says so
//! ([`crate::obs::dstep`]). GLM-5.2, batch 1, warm, mean of 32-token windows —
//! TP8 under `amd-bench` on an exclusive 8-GPU lease at ctx 4096, TP4 through
//! the endpoint:
//!
//! ```text
//!                                              TP8 µs/tok       %   TP4 µs/tok
//! pre  seed_ids (H2D in.ids, x ranks)                  —        —         32.6
//! pre  decode_prepare (kvrow patch + scalars)      249.7    1.10%        126.0
//! pre  rearm_prog (local counters)                 210.3    0.93%        105.2
//! pre  zero_xctr (cross-GPU gates, all ranks)       67.1    0.30%         35.5
//! pre  enqueue (AQL launch x ranks)                  3.9    0.02%          2.0
//! GPU  drain (all ranks)                         22156     97.05%      27500
//! post audit_xctr (12 KiB D2H x ranks)              87.6    0.39%         48.7
//! post read_sampled (4 B D2H x ranks)               59.3    0.26%         30.5
//! post agree (cross-rank compare)                    0.03   0.00%          0.05
//! post detok + stop + SSE send                        —        —           5.8
//! HOST TOTAL (everything but the drain)            677      2.96%        385
//! UNATTRIBUTED (mux tick, locks, scheduler)          0.94   0.00%          1.0
//! ```
//!
//! `seed_ids` and the stream row are endpoint-only, hence blank under
//! `amd-bench`. TP8 totalled **22.729 ms/token over 128 steps, all 8 ranks
//! token-identical**.
//!
//! **The host phase is 3.0% of the token at TP8, 1.4% at TP4.** It is submission
//! overhead, so it scales with RANK COUNT and not with model size — about 65 µs
//! per rank plus ~70 µs fixed, which reproduces both columns. Perfect overlap of
//! every microsecond of it would therefore buy 3%, and three further facts cut
//! that ceiling to nothing, each stronger than "not worth it":
//!
//! 1. **Tick N+1's prepare cannot run during tick N. It is unsafe, not merely
//!    unprofitable.** The obvious hoist — `patch_kvrow(pos)` and the `pos`/
//!    `kvlen` staging do not depend on the sampled token, so run them early —
//!    overlooks *where they write*. `patch_kvrow` memcpy's into `progs[dp].
//!    d_inst`, the very instruction buffer the resident megakernel is executing
//!    from; `pos` and `kvlen` are device tensors every RoPE and attention op
//!    re-reads through the tick; and `rearm_prog` zeroes the local counters the
//!    resident tick is still signalling — the same hazard §6d rules out for
//!    `xctr`, restated one scope down. Hoisting any of the three corrupts the
//!    tick in flight, silently. It needs double-buffered `d_inst`/`in.pos`/
//!    `in.kvlen`/counters and a kernarg to select the copy — a devgen and ABI
//!    change — to buy at most the 527 µs those three rows total at TP8: 2.3%.
//! 2. **The window would be empty anyway.** With the `pre` rows excluded by (1),
//!    the only host work left to overlap is the `post` block, and its
//!    non-device-touching part — detokenise, stop check, SSE frame, channel
//!    send — is **5.8 µs**. The "serving premium over `amd-bench`" is real but
//!    it is inside the DRAIN, not around it: `UNATTRIBUTED`, which is the mux
//!    tick, the engine lock and the scheduler, is **1.0 µs**.
//! 3. **Deferring the stream to gain those 5.8 µs is net NEGATIVE.** Streaming
//!    token N during tick N+1 means the stop token is seen one tick late, so
//!    every request pays one extra full dispatch — ~27 ms — to save 5.8 µs per
//!    token. That is a loss for any request shorter than ~4,600 tokens.
//!
//! The fourth candidate — making the cross-rank agreement a sampled audit
//! instead of `N` readbacks per token — is built and argued at
//! [`AmdTpGroup::complete_decode`], and ships DISABLED for the same reason
//! quantified: it would save 7/8 of 59.3 µs, i.e. **0.23% at TP8**, against a
//! run-to-run spread between 32-token windows of 22.57–23.10 ms, i.e. 2.3%. The
//! effect is an order of magnitude below this path's own noise floor, so it
//! cannot be confirmed by measurement even in principle, and it is paid for in
//! acceptance-test coverage.
//!
//! None of this says the host phase is small in ABSOLUTE terms. 677 µs is fixed
//! per token, so on a model that decodes in ~1.5 ms rather than ~23 the same
//! work is nearly half the token and every conclusion above flips — including
//! the sign on (3), since the extra dispatch a deferred stream costs shrinks
//! with it. The instrument is checked in for exactly that day; the design
//! notes predict 1.56 ms/token for a 12B at TP8.
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

/// Tokens between full all-rank readbacks — see [`AmdTpGroup::complete_decode`]
/// for why a cadence is sound at all.
///
/// # Why the default is 1, i.e. why this ships turned OFF
///
/// The cadence exists because `N-1` of the `N` readbacks are pure audit, so it
/// looked like free host time. It was then MEASURED (`PLOW_DSTEP_LOG=1`,
/// §DSTEP): on GLM-5.2 TP8, warm, batch 1, ctx 4096, ALL EIGHT readbacks cost
/// **59.3 µs of a 22.7 ms token — 0.26%**, so the seven audit reads are 0.23%.
/// Consecutive 32-token windows of that same run ranged 22.57–23.10 ms, a spread
/// of 2.3%: the saving is an order of magnitude below the noise floor of the
/// thing it would speed up, and so cannot be confirmed even in principle.
///
/// A change no measurement can see does not buy a weakening of what this
/// module's own [`AmdTpGroup::agree`] doc calls "the acceptance test, not a
/// debug aid". The mechanism is here, argued and tested; the default leaves
/// behaviour exactly as it was. `PLOW_TP_AGREE_EVERY=16` turns it on for the
/// regime where the arithmetic changes sign: the host phase is roughly FIXED per
/// token (submission overhead, not work), so on a model that decodes in ~1.5 ms
/// rather than ~23 the same microseconds are worth cadencing. Measure first.
const DEFAULT_AGREE_EVERY: u32 = 1;

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
    audit_direct: bool,
    audit_compact: bool,
    /// Read EVERY rank's sampled id, rather than just rank 0's, once every this
    /// many decode tokens — see [`AmdTpGroup::audit_cadence`].
    agree_every: u32,
    /// Tokens since the last all-rank read. Starts at `agree_every` so the
    /// FIRST token is always audited: a rank that bound the wrong shard is wrong
    /// from token one, and a scheme that armed at the END of the first window
    /// would serve `agree_every - 1` wrong tokens before looking.
    agree_tick: u32,
    /// The decode program the LAST [`AmdTpGroup::submit_decode_batched`] launched — a rung of
    /// the decode batch ladder (`PLOW_DECODE_BATCH_LADDER`), or the sole decode program.
    ///
    /// Carried rather than re-derived because `complete_decode` audits the gate expectations of
    /// the program that actually ran, and submit/complete are separate calls by design (the
    /// split is what lets the host do nothing between launch and drain).
    cur_dp: usize,
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

        // Parse the blob once, on the host, purely to size the peer region. Metadata-only, so
        // an L2-placed blob (the gfx942 default) is fine here — the per-object dispatch-symbol
        // check in the engine is what actually guards L2 dispatch, same as the other read sites.
        let raw = std::fs::read(blob_path)
            .map_err(|e| RuntimeError::Device(format!("read {}: {e}", blob_path.display())))?;
        let blob = DevBlob::parse_l2(&raw, true)?;
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
                "packet collective has zero width, so its hidden size is unrecoverable"
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
        let audit_compact = crate::config::RuntimeConfig::get().amd.tp_audit_compact;
        if audit_compact
            && blob
                .progs
                .iter()
                .flat_map(|p| &p.stream)
                .any(|e| e.flags & packet::dev::SE_XCTR != 0)
        {
            return Err(RuntimeError::Device(
                "compact TP audit does not support SE_XCTR fine-gate programs; use the direct or copy audit"
                    .into(),
            ));
        }
        let gate_expect = gate_expectations(&blob, n_gpu, n_xctr);
        let layout = PeerLayout::new(tp.hidden, max_tokens, n_xctr).ok_or_else(|| {
            RuntimeError::Device(format!(
                "peer layout for hidden={} x {max_tokens} tokens is not 128 B-aligned",
                tp.hidden
            ))
        })?;
        tracing::info!(
            n_gpu,
            hidden = tp.hidden,
            max_tokens,
            n_xctr,
            peer_kib = layout.bytes() / 1024,
            "TP peer layout sized from the packet"
        );

        let dyn_backends: Vec<Arc<dyn Backend>> = backends
            .iter()
            .map(|b| Arc::clone(b) as Arc<dyn Backend>)
            .collect();
        let group = TpGroup::bringup(dyn_backends, layout)?;
        // All N*(N-1) directed pairs, byte-exact, BEFORE any weight is bound.
        // `agents_allow_access` REPLACES a buffer's allow-list, so the classic
        // failure leaves only the last-named rank mapped and every other peer
        // faulting at the first token — an hour of weight loading later.
        group.verify_peer_visibility()?;

        // ONE checkpoint mapping for the whole group, not one per rank.
        //
        // Each rank used to `mmap` the shards itself, and the cost of that is
        // not the mapping — it is the PAGE TABLES. A rank's own VMA starts with
        // no PTEs, so every one of the ~44 M pages it reads is a minor fault
        // even when the bytes are already in page cache, and that measured
        // **44 s per rank** on a fully warm GLM-5.2 load: 70 % of a warm rank's
        // 67 s. Sharing the mapping means rank 0 populates the PTEs and ranks
        // 1..n find them already there.
        //
        // Sound because the mapping is READ-ONLY and the ranks only read it;
        // `Checkpoint` holds `memmap2::Mmap`, which is `Sync` for exactly that
        // reason.
        //
        // `--amd-share-ckpt false` / `PLOW_SHARE_CKPT=0` restores the per-rank
        // mapping, so the difference can be measured on one binary instead of
        // asserted.
        let share = crate::config::RuntimeConfig::get().amd.share_ckpt;
        let shared_ckpt = match (share, checkpoint) {
            (true, Some(dir)) => Some(std::sync::Arc::new(
                crate::asset::checkpoint::Checkpoint::open(dir)?,
            )),
            _ => None,
        };
        // Every rank's binding, resolved before any of them loads, so the load
        // itself borrows nothing from `group`.
        let mut binds = Vec::with_capacity(backends.len());
        for (r, be) in backends.into_iter().enumerate() {
            let tr = group.rank(r as u32)?;
            binds.push((
                be,
                TpBind {
                    rank: r as u32,
                    n_gpu,
                    peer_table: tr.peer_scratch_table(),
                    xctr: tr.xctr(),
                    xstatus_id: n_xctr,
                    scratch_base: tr.scratch_base(),
                    slot_b: tp.slot_bytes,
                },
                tr.ordinal(),
            ));
        }

        // Ranks load CONCURRENTLY. They share nothing mutable: each has its own
        // agent, its own queue, its own VRAM and its own staging ring, and the
        // one thing they do share — the checkpoint mmap — is read-only. What
        // they were sharing before was the wall clock, one rank at a time.
        //
        // `thread::scope`, not detached threads, because the closures borrow
        // `blob_path`/`hsaco_dir`/`checkpoint`. Every handle is joined even
        // after the first failure: a rank still loading owns HSA queues and
        // in-flight SDMA copies, and returning out from under it is how you get
        // a fault in a thread nobody is waiting on. The first error is what the
        // caller sees; the group is dropped whole, so no rank survives
        // half-loaded.
        //
        // `--amd-tp-serial-load` / `PLOW_TP_SERIAL_LOAD=1` restores the
        // one-at-a-time loop.
        let serial = crate::config::RuntimeConfig::get().amd.tp_serial_load;
        let t_bind = std::time::Instant::now();
        let ranks: Vec<AmdEngine> = if serial {
            let mut v = Vec::with_capacity(binds.len());
            for (be, bind, ordinal) in binds {
                tracing::info!(rank = bind.rank, ordinal, "binding rank");
                v.push(AmdEngine::load_rank(
                    be,
                    blob_path,
                    hsaco_dir,
                    checkpoint,
                    Some(bind),
                    shared_ckpt.clone(),
                )?);
            }
            v
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = binds
                    .into_iter()
                    .map(|(be, bind, ordinal)| {
                        let shared = shared_ckpt.clone();
                        tracing::info!(rank = bind.rank, ordinal, "binding rank");
                        s.spawn(move || {
                            AmdEngine::load_rank(
                                be,
                                blob_path,
                                hsaco_dir,
                                checkpoint,
                                Some(bind),
                                shared,
                            )
                        })
                    })
                    .collect();
                let mut out = Vec::with_capacity(handles.len());
                let mut first: Option<RuntimeError> = None;
                for h in handles {
                    match h.join() {
                        Ok(Ok(e)) => out.push(e),
                        Ok(Err(e)) => {
                            if first.is_none() {
                                first = Some(e);
                            }
                        }
                        Err(_) => {
                            if first.is_none() {
                                first = Some(RuntimeError::Device(
                                    "a rank's weight load panicked".into(),
                                ));
                            }
                        }
                    }
                }
                match first {
                    Some(e) => Err(e),
                    None => Ok(out),
                }
            })?
        };
        tracing::info!(
            n_gpu,
            serial,
            secs = format!("{:.1}", t_bind.elapsed().as_secs_f64()).as_str(),
            "all ranks bound"
        );

        let agree_every = match crate::config::RuntimeConfig::get().amd.tp_agree_every {
            0 => DEFAULT_AGREE_EVERY,
            n => n,
        };
        Ok(AmdTpGroup {
            group,
            ranks,
            reset: XctrReset::Host,
            gate_expect,
            audit: !crate::config::RuntimeConfig::get().amd.tp_no_audit,
            audit_direct: crate::config::RuntimeConfig::get().amd.tp_audit_direct,
            audit_compact,
            agree_every,
            agree_tick: agree_every,
            // Overwritten by the first submit; the widest rung is the safe pre-first-step value.
            cur_dp: 0,
        })
    }

    /// Read every rank's sampled id once per `every` decode tokens; `1` is every
    /// token. See [`AmdTpGroup::complete_decode`] for why a cadence is sound.
    ///
    /// A CORRECTNESS ORACLE must pass `1` — `amd-bench --tp N` does, because its
    /// whole claim is "every rank emitted an identical stream" and a sampled
    /// check cannot support that sentence.
    pub fn audit_cadence(&mut self, every: u32) {
        self.agree_every = every.max(1);
        self.agree_tick = self.agree_every;
    }

    pub fn n_gpu(&self) -> usize {
        self.ranks.len()
    }

    /// Task-9 differential counter audit: rank 0's end-state counters for `dp`.
    pub fn ctr_snapshot(&mut self, dp: usize) -> Result<Vec<u32>> {
        self.ranks[0].ctr_word0_snapshot(dp)
    }

    /// Task-9 round-7 data audit: rank 0's layer-0 KV slot head + named act tensors.
    pub fn data_snapshot(
        &mut self,
        slot: usize,
        kv_bytes: usize,
        acts: &[&str],
    ) -> Result<Vec<(String, Vec<u8>)>> {
        let mut out = self.ranks[0].snapshot_kv_slot(slot, kv_bytes)?;
        for a in acts {
            out.push(((*a).to_string(), self.ranks[0].snapshot_tensor(a)?));
        }
        Ok(out)
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

    /// One batched decode step across every rank: submit, drain, return the `pos.len()` sampled
    /// ids.
    ///
    /// AGREEMENT IS CHECKED ON THE WHOLE B-VECTOR, not on slot 0. Every rank samples from its own
    /// shard of the logits, so a broken all-reduce still yields fluent ids — agreement is the only
    /// signal that distinguishes it from a working one. Checking slot 0 alone would test one of B
    /// sequences and report green while the rest diverged, which is the same mistake
    /// `amd_bench`'s B=16 batched arm made (it compared every slot against slot 0 rather than
    /// against the first slot carrying its own prompt).
    pub fn decode_step_batched(&mut self, pos: &[u32], kvlen: &[u32]) -> Result<Vec<u32>> {
        let dp = self.ranks[0].decode_prog();
        self.decode_step_batched_at(pos, kvlen, dp)
    }

    /// [`Self::decode_step_batched`] on a NAMED decode rung — see `AmdEngine::decode_prog_for`.
    pub fn decode_step_batched_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        dp: usize,
    ) -> Result<Vec<u32>> {
        self.submit_decode_batched_at(pos, kvlen, dp)?;
        for e in &self.ranks {
            e.drain()?;
        }
        // Only the rung's rows sampled; the reply still comes back `pos.len()` long so callers
        // can index it BY SLOT, with the uncovered tail zeroed.
        let b = pos.len();
        let rows = (self.ranks[0].prog_t(dp) as usize).min(b);
        let per_rank: Vec<Vec<u32>> = self
            .ranks
            .iter_mut()
            .map(|e| e.read_sampled_batched(rows))
            .collect::<Result<_>>()?;
        for (r, ids) in per_rank.iter().enumerate().skip(1) {
            if ids != &per_rank[0] {
                return Err(RuntimeError::Device(format!(
                    "TP ranks disagree on a batched decode step: rank 0 sampled {:?}, rank {r} \
                     sampled {ids:?} — the all-reduce is wrong",
                    per_rank[0]
                )));
            }
        }
        let mut ids = per_rank.into_iter().next().expect(">=1 rank");
        ids.resize(b, 0);
        Ok(ids)
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
    /// Scalar convenience over [`Self::submit_decode_batched`] — one sequence, which is every
    /// caller today.
    pub fn submit_decode(&mut self, pos: u32, kvlen: u32) -> Result<()> {
        self.submit_decode_batched(&[pos], &[kvlen])
    }

    /// Submit one decode step for `B` sequences across every rank.
    ///
    /// EVERY RANK PREPARES BEFORE ANY RANK LAUNCHES, and that ordering is the reason this cannot
    /// simply loop `decode_step_batched` per rank: `launch_token` owns zero-all-then-launch-all,
    /// so a rank that launched while another was still staging its positions would read a
    /// half-written `in.pos`. The staging itself is `decode_prepare_batched`, shared with the
    /// single-GPU path so the two cannot drift — a rank feeding one sequence a stale position is
    /// silent, not a crash.
    ///
    /// `pos` and `kvlen` are per-sequence and may be RAGGED; every rank gets the same vectors,
    /// because TP replicates the residual and every rank runs all `B` sequences.
    pub fn submit_decode_batched(&mut self, pos: &[u32], kvlen: &[u32]) -> Result<()> {
        let dp = self.ranks[0].decode_prog();
        self.submit_decode_batched_at(pos, kvlen, dp)
    }

    /// [`Self::submit_decode_batched`] on a NAMED decode rung of the ladder.
    pub fn submit_decode_batched_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        dp: usize,
    ) -> Result<()> {
        use crate::obs::dstep;
        self.cur_dp = dp;
        for e in &mut self.ranks {
            dstep::timed(&dstep::PREPARE, || e.decode_prepare_batched(pos, kvlen))?;
            dstep::timed(&dstep::REARM, || e.rearm_prog(dp))?;
        }
        // `launch_token` owns zero-all-then-launch-all; the closure only says
        // what a launch IS.
        //
        // §DSTEP splits the two phases from OUTSIDE `launch_token` rather than
        // by instrumenting it: the first call into the closure is by definition
        // the first instruction after `zero_xctr` returned, so stamping it there
        // dates the boundary exactly and leaves the ordering discipline in the
        // one place that owns it.
        let ranks = &mut self.ranks;
        let mut i = 0usize;
        let t0 = dstep::on().then(std::time::Instant::now);
        let mut launched_at: Option<std::time::Instant> = None;
        self.group.launch_token(self.reset, |_| {
            if let (Some(t0), None) = (t0, launched_at) {
                let now = std::time::Instant::now();
                dstep::XCTR.add((now - t0).as_nanos() as u64);
                launched_at = Some(now);
            }
            let e = &mut ranks[i];
            let k = e.decode_kernel_for(dp);
            i += 1;
            e.enqueue(dp, k)
        })?;
        if let Some(z) = launched_at {
            dstep::ENQUEUE.add(z.elapsed().as_nanos() as u64);
        }
        Ok(())
    }

    /// Wait for an in-flight [`AmdTpGroup::submit_decode`] and collect the ids.
    ///
    /// Drains every rank BEFORE auditing or reading anything: a readback from
    /// rank 0 while rank 3 is still running would race the collective that rank
    /// 0's own result depends on.
    ///
    /// # The returned vector is the ranks that were READ, not the ranks
    ///
    /// Every rank computes the same id — that is the invariant, and comparing
    /// them is the check that catches "a collective did not happen, or one rank
    /// bound the wrong shard". But the token only ever comes from rank 0, so on
    /// a token where the check is not being made the other `N-1` readbacks are
    /// pure audit and this returns a ONE-element vector. [`AmdTpGroup::agree`]
    /// over one id is trivially satisfied, which is the honest reading: no
    /// cross-rank claim was made this token.
    ///
    /// # Why sampling is sound, and what still runs every token
    ///
    /// The two failures split cleanly by lifetime:
    ///
    /// * **Structural** — a rank bound the wrong shard, or the packet carries no
    ///   collective at all. This is a property of the LOAD, so it is wrong on
    ///   every token from the first. Sampling finds it on token 0 (`agree_tick`
    ///   starts armed) and cannot miss it.
    /// * **Transient** — a collective hit its `PLOW_XCTR_DEADLINE_TICKS`
    ///   deadline and returned without reducing, on this token only. Sampling
    ///   *would* miss most of these — which is exactly why
    ///   [`TpGroup::audit_xctr`] stays on EVERY token. It reads the arrival
    ///   counters, so it sees a timed-out collective directly rather than
    ///   inferring it from a token that happened to differ, and it is the
    ///   stronger of the two checks for this case.
    ///
    /// So the cadence drops the check that is redundant against a permanent
    /// fault and keeps the one that catches a momentary one. `PLOW_TP_NO_AUDIT=1`
    /// removes the counter check as well, and then a timed-out collective is
    /// silent again — as it was before either check existed.
    pub fn complete_decode(&mut self) -> Result<Vec<u32>> {
        use crate::obs::dstep;
        dstep::timed(&dstep::DRAIN, || -> Result<()> {
            for e in &self.ranks {
                e.drain()?;
            }
            Ok(())
        })?;
        if self.audit {
            // The rung that actually launched, not the widest — a narrow rung emits fewer
            // collectives and auditing it against the widest program's expectations would
            // report a timeout on gates it never armed.
            let dp = self.cur_dp;
            dstep::timed(&dstep::AUDIT, || {
                if self.audit_compact {
                    for (e, r) in self.ranks.iter().zip(self.group.ranks()) {
                        e.enqueue_xaudit(
                            dp,
                            r.xctr(),
                            self.group.layout().n_xctr,
                            self.group.n_gpu(),
                            r.xstatus(),
                        )?;
                    }
                    for e in &self.ranks {
                        e.drain()?;
                    }
                    match self.group.audit_xstatus_direct() {
                        Ok(()) => Ok(()),
                        Err(status) => self
                            .group
                            .audit_xctr_direct(&self.gate_expect[dp])
                            .and(Err(status)),
                    }
                } else if self.audit_direct {
                    self.group.audit_xctr_direct(&self.gate_expect[dp])
                } else {
                    self.group.audit_xctr(&self.gate_expect[dp])
                }
            })?;
        }
        let all = self.agree_tick >= self.agree_every;
        self.agree_tick = if all { 1 } else { self.agree_tick + 1 };
        dstep::timed(&dstep::READ, || {
            if all {
                self.ranks.iter_mut().map(|e| e.read_sampled()).collect()
            } else {
                Ok(vec![self.ranks[0].read_sampled()?])
            }
        })
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
                &self.ranks[0]
                    .plan_for(prompt.len() as u32)
                    .unwrap_or_default(),
            );
        }
        tracing::info!(
            tokens = prompt.len(),
            chunks = steps.len(),
            n_gpu = self.ranks.len(),
            "TP prefill plan"
        );

        for step in steps {
            self.prefill_chunk(prompt, step)?;
        }

        let t_read = std::time::Instant::now();
        let out = self.ranks.iter_mut().map(|e| e.read_sampled()).collect();
        crate::obs::ttft::PF_READ.add(t_read.elapsed().as_nanos() as u64);
        out
    }

    /// Prefill into sequence SLOT `slot` of a batched decode program.
    ///
    /// The single-GPU twin is `AmdEngine::prefill_slot`: rebase the KV pointer table onto the
    /// slot, run the ordinary prefill, restore. The group version must rebase EVERY RANK BEFORE
    /// THE PREFILL RUNS, because prefill is collective — a rank still pointing at slot 0 would
    /// rendezvous with peers writing slot `s` and the reduce would mix two slots' partials.
    ///
    /// Restore is unconditional, on the failure path too: a half-prefilled slot is recoverable,
    /// a pointer table left rebased is not — the next decode would funnel every sequence into
    /// one slot's cache, and `decode_step_batched` refuses exactly that (`kv_slot != 0`).
    /// Hand slot `slot` to a NEW sequence on every rank: clear its carried recurrent state.
    ///
    /// This existed only on the single-GPU path, which meant that under TP — the shipped K3
    /// configuration — NOTHING cleared KDA state between requests, and every request after the
    /// first on a slot began from its predecessor's accumulated recurrence across 69 of K3's 93
    /// layers. `AmdEngine::begin_slot` documents why an append-only KV cache needs no clear and
    /// a recurrence does.
    ///
    /// EVERY RANK, and the loop is not an optimisation detail: the recurrence is sharded by head,
    /// so a rank that skipped the clear would carry stale state for its own heads only and the
    /// ranks would disagree about the sequence from its very first token.
    pub fn begin_slot(&mut self, slot: usize) -> Result<()> {
        for e in &mut self.ranks {
            e.begin_slot(slot)?;
        }
        Ok(())
    }

    pub fn prefill_slot(&mut self, slot: usize, prompt: &[u32]) -> Result<Vec<u32>> {
        for e in &mut self.ranks {
            e.kv_rebase(slot)?;
        }
        let r = self.prefill(prompt);
        let mut restore = Ok(());
        for e in &mut self.ranks {
            if let Err(err) = e.kv_rebase(0) {
                restore = Err(err);
            }
        }
        restore?;
        r
    }

    /// Snapshot / restore / probe the carried recurrent state on EVERY rank.
    ///
    /// The recurrence is sharded by head, so a snapshot is only meaningful if every rank takes
    /// one at the same point in the token stream — a rank that skipped it would resume from a
    /// state one prefix behind its peers and the group would disagree from the first token.
    pub fn snapshot_carried(&mut self, slot: usize) -> Result<()> {
        for e in &mut self.ranks {
            e.snapshot_carried(slot)?;
        }
        Ok(())
    }

    pub fn restore_carried(&mut self, slot: usize) -> Result<()> {
        for e in &mut self.ranks {
            e.restore_carried(slot)?;
        }
        Ok(())
    }

    /// Publish the per-row parked mask on every rank. See `AmdEngine::upload_parked`.
    pub fn upload_parked(&mut self, parked: &[u32]) -> Result<()> {
        for e in &mut self.ranks {
            e.upload_parked(parked)?;
        }
        Ok(())
    }

    pub fn has_snapshot(&self, slot: usize) -> bool {
        self.ranks.iter().all(|e| e.has_snapshot(slot))
    }

    /// Point every rank's KV pointer table at sequence slot `slot`.
    ///
    /// Exposed for the CHUNK-AT-A-TIME server path: `prefill_slot` holds the rebase across a
    /// whole prompt, but a scheduler that interleaves decode between chunks must hand the base
    /// back to 0 in between, because `decode_step_batched` refuses a non-zero base.
    pub fn kv_rebase_all(&mut self, slot: usize) -> Result<()> {
        for e in &mut self.ranks {
            e.kv_rebase(slot)?;
        }
        Ok(())
    }

    /// The chunk plan covering `[from, to)` — the cursor a chunked server steps through.
    pub fn plan_span(&self, from: u32, to: u32) -> Result<Vec<ChunkStep>> {
        if from >= to {
            return Ok(Vec::new());
        }
        let chunks = self.ranks[0].plan_for(to - from)?;
        self.ranks[0].chunk_steps_from(&chunks, from, to)
    }

    /// Every rank's sampled id after a prefill's final chunk.
    pub fn read_sampled_all(&mut self) -> Result<Vec<u32>> {
        self.ranks.iter_mut().map(|e| e.read_sampled()).collect()
    }

    /// Run the prefill chunks covering `[from, to)` on every rank.
    fn prefill_span(&mut self, prompt: &[u32], from: u32, to: u32) -> Result<()> {
        if from >= to {
            return Ok(());
        }
        let chunks = self.ranks[0].plan_for(to - from)?;
        let steps = self.ranks[0].chunk_steps_from(&chunks, from, to)?;
        for step in steps {
            self.prefill_chunk(prompt, step)?;
        }
        Ok(())
    }

    /// Prefill into `slot`, reusing a cached prefix where one is armed.
    ///
    /// `resume > 0` means slot `slot` holds a snapshot taken at token `resume` AND its KV rows
    /// `[0, resume)` are the same tokens at the same positions, so those tokens are skipped
    /// entirely: restore the recurrence and prefill only `[resume, len)`.
    ///
    /// `arm > 0` on a MISS means "split the prefill at `arm` and snapshot there", which costs
    /// nothing extra — the same tokens are prefilled either way — and arms the next request on
    /// this slot. `arm` needs no bucket alignment: `rebase_chunk` runs every KDA op for `clen`
    /// rows, not the padded bucket width, so a chunk ending at `arm` leaves the state at exactly
    /// `arm`.
    ///
    /// KV rebase is held across the WHOLE sequence of spans and restored unconditionally, for the
    /// same reason [`AmdTpGroup::prefill_slot`] holds it across one.
    pub fn prefill_slot_cached(
        &mut self,
        slot: usize,
        prompt: &[u32],
        resume: u32,
        arm: u32,
    ) -> Result<Vec<u32>> {
        for e in &mut self.ranks {
            e.kv_rebase(slot)?;
        }
        let r = self.prefill_cached_inner(prompt, slot, resume, arm);
        let mut restore = Ok(());
        for e in &mut self.ranks {
            if let Err(err) = e.kv_rebase(0) {
                restore = Err(err);
            }
        }
        restore?;
        r
    }

    fn prefill_cached_inner(
        &mut self,
        prompt: &[u32],
        slot: usize,
        resume: u32,
        arm: u32,
    ) -> Result<Vec<u32>> {
        let n = prompt.len() as u32;
        let from = if resume > 0 {
            self.restore_carried(slot)?;
            resume
        } else if arm > 0 {
            self.prefill_span(prompt, 0, arm)?;
            self.snapshot_carried(slot)?;
            arm
        } else {
            0
        };
        self.prefill_span(prompt, from, n)?;
        self.ranks.iter_mut().map(|e| e.read_sampled()).collect()
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
        let first = *ids
            .first()
            .ok_or_else(|| RuntimeError::Device("no ranks sampled anything".into()))?;
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
            if d.op == DevOp::XReduce as u16 || d.op == DevOp::XReduceAddNorm as u16 {
                top = top.max(d.i[3] + 1); // one-shot: one gate, i3 (fused AddNorm form included)
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
                if d.op == DevOp::XReduce as u16 || d.op == DevOp::XReduceAddNorm as u16 {
                    // One-shot rendezvous either way; the fused AddNorm form (116) signals
                    // once per rank exactly like the standalone XReduce it replaces.
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

    /// The all-rank readback ships on EVERY token: the cadence exists, but the
    /// measurement that motivated it (30.5 µs of a 27.9 ms token, 0.11%) does
    /// not justify sampling the acceptance test. A default of anything but 1 is
    /// a behaviour change and has to be argued from a fresh §DSTEP breakdown.
    #[test]
    fn cross_rank_agreement_is_not_sampled_by_default() {
        assert_eq!(DEFAULT_AGREE_EVERY, 1);
    }

    /// The cadence must fire on token ZERO. A rank that bound the wrong shard is
    /// wrong from the first token, and a scheme that armed on the *last* token
    /// of the first window would serve `every - 1` wrong tokens before looking.
    /// That is the whole reason `agree_tick` starts at `agree_every` rather
    /// than 0 — a detail with no other symptom, hence a test.
    #[test]
    fn the_cadence_is_armed_on_the_first_token() {
        // The state machine `complete_decode` runs, in isolation: it reads
        // every rank when `tick >= every`, then restarts the count at 1.
        let step = |tick: &mut u32, every: u32| {
            let all = *tick >= every;
            *tick = if all { 1 } else { *tick + 1 };
            all
        };
        let every = 4;
        let mut tick = every; // as `load` and `audit_cadence` leave it
        let fired: Vec<bool> = (0..9).map(|_| step(&mut tick, every)).collect();
        assert_eq!(
            fired,
            [true, false, false, false, true, false, false, false, true],
            "token 0 must be audited, then one in every {every}"
        );

        // `every == 1` is the oracle setting and must audit unconditionally —
        // an off-by-one here would silently downgrade `amd-bench --tp N`, whose
        // entire claim is that every rank emitted an identical stream.
        let mut tick = 1u32;
        assert!(
            (0..8).all(|_| step(&mut tick, 1)),
            "every=1 must never skip"
        );
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
        assert_eq!(
            e[0][3], None,
            "the fold's published u64 is data, not a count"
        );
    }
}
