//! Tensor-parallel device group — the host side of plow's inline collectives.
//!
//! # Why this is not "spawn N devices and call RCCL"
//!
//! In a launched runtime (vLLM+RCCL) every decode op is a kernel launch and
//! every collective is a launched RCCL kernel with a host-stream sync, so at
//! batch=1 the step is dominated by launch latency: ~96 all-reduces per token at
//! ~5–10 µs each is 0.7–1.9 ms of collective overhead alone, *regardless of the
//! GPU count*. plow's whole token is **one persistent megakernel dispatch per
//! GPU**, and a collective is more entries in that resident interpreter's
//! stream, gated by cross-GPU counters in peer-mapped VRAM. There is no
//! per-collective launch and no host synchronisation inside a token.
//!
//! That structure is what this module has to preserve, so the host's entire job
//! per token is: **reset every rank's `xctr`, then dispatch every rank once.**
//! Anything the host does *between* those two steps, or between ranks, is
//! latency the design does not have a budget for — which is why
//! [`TpGroup::launch_token`] owns the ordering instead of leaving it to a call
//! site to remember.
//!
//! And that reset is not free: at TP=4 with 96 counters it costs **31.5 µs** of
//! host time through the copy engine — MORE than the ~29 µs all 96 of the
//! token's inline all-reduces cost put together. Host stores into large-BAR
//! peer VRAM cut it to **1.97 µs**, and monotonic device-side counters would
//! cut it to zero. See [`XctrReset`]: the point of naming the three modes is
//! that "the host does nothing per token" is a claim, and a claim wants a
//! number.
//!
//! # What is on the fabric (and what must never be)
//!
//! Only the reduction partials, their result, and the cross-GPU counters are
//! peer-mapped. Weights, KV, and the replicated
//! residual stream stay in local HBM: aggregate weight bandwidth of `N × 8 TB/s`
//! is the entire win, and it evaporates the moment a weight read crosses a
//! 58 GB/s link instead. For 12B decode that is 27 KB per GPU (15 KB of
//! partials + 12 KB of counters) against 24 GB of weights — six orders of
//! magnitude apart, which is the invariant. Prefill's partial is `t·H·2`, so
//! the region grows with the CHUNK size, not the sequence length; see
//! [`PeerLayout::max_tokens`].
//!
//! # TP=4, twice — the deployment shape
//!
//! The primary degree is **4**, not 8: it keeps the expert GEMMs above their
//! efficiency knee and the collectives cheap, and an 8-GPU node then hosts
//! **two independent TP4 replicas** rather than one TP8 job. So a group is a
//! *subset* of the node's devices — see [`TpGroup::split_replicas`] — a rank's
//! index in its group is not its device ordinal, and a replica's peer buffers
//! are mapped to that replica's agents only.
//!
//! # Measured, on this node (gfx950 ×8, ROCm 7.2.4)
//!
//! `runtime/tests/tp_p2p_bench`, physical GPUs 4↔5 and 4↔7:
//! peer store 58.6 GB/s · **system-scope atomic over peer VRAM works**, ~0.06 µs
//! one-way handshake · 8 KB SDMA copy 13.95 µs. The first two are why the
//! collective can be inline; the third is why [`crate::device::PeerMemory::copy_peer_blocking`]
//! is a bulk/test primitive and never a per-token one.
//!
//! `runtime/tests/tp_allreduce_bench`, 2 ranks, 7.5 KB message:
//! **0.302 µs per inline one-shot all-reduce**, bit-exact — against ~5–10 µs
//! for a launched small-message all-reduce.
//!
//! # One-shot vs two-shot: a compile-time choice, not a runtime one
//!
//! A launched stack has to pick its collective algorithm by *batch size* at
//! run time, because it is trading a launch cost it cannot avoid against
//! bandwidth. plow pays no launch, so the trade collapses to pure fabric
//! volume, and that is a property of the PHASE: `crates/devgen` emits one-shot
//! `XReduce` for decode (`[1,H]`, latency-bound, `(N−1)·msg`) and
//! `XReduceTwoShot` for prefill (`[T,H]`, bandwidth-bound, `2(N−1)/N·msg`),
//! bit-identically. The host layer therefore never selects an algorithm; it
//! only has to size the region and the counters for whichever the program
//! carries — which is what [`PeerLayout::counters_for`] is for, since two-shot
//! takes two gates per collective and one-shot takes one.

use std::sync::Arc;

use crate::device::{Backend, DeviceMem};
use crate::{Result, RuntimeError};

/// Bytes per cross-GPU counter — `PLOW_CTR_STRIDE` (32 u32) from `dev_isa.h`.
///
/// The stride is cache-line isolation, and it matters MORE across GPUs than
/// within one: two counters sharing a line means two ranks' system-scope release
/// RMWs contend for the same line over XGMI, turning independent signals into a
/// serialised fabric round-trip each.
pub const XCTR_STRIDE: usize = 128;

/// Alignment of every sub-region inside the peer scratch.
const PEER_ALIGN: u64 = XCTR_STRIDE as u64;

/// COARSE peer-scratch layout — one peer-mapped
/// region per GPU, laid out identically on every rank.
///
/// ```text
///   [0]           partial_A   tokens·hidden·2 B   o_proj partial
///   [slot_b]      partial_B   tokens·hidden·2 B   down   partial
///   [xctr_off]    xctr        n_xctr·128 B        cross-GPU counters
///   [xstatus_off] xstatus     128 B               compact-audit result
/// ```
///
/// The layout is identical on every rank on purpose: a producer signalling peer
/// `r`'s counter `c` computes `peer_scratch[r] + xctr_off + c·128` with no
/// per-rank table lookup, which is exactly the arithmetic `dev_isa.h` documents
/// on `PlowProgram::xctr` ("the per-rank counter offset is
/// `xctr - peer_scratch[rank]`").
///
/// # Two slots, and why that is enough
///
/// `slot_b` must equal what the emitter computes — `crates/devgen` uses a raw
/// `t·h·2`, so this does too rather than rounding up: a mismatch would have
/// every rank's `down` partial land at a different offset than its peers read.
/// (`h` is a multiple of 64 on every model in the tree, so `t·h·2` is already
/// 128 B-aligned; [`PeerLayout::new`] rejects a width where it is not, instead
/// of silently giving up the counter region's line isolation.)
///
/// Reusing slot A on the next layer is safe even though no counter says "I have
/// finished reading yours". Between rank `r` reading everyone's A and rank `r`
/// overwriting its own A lies the whole FFN *and* the B collective, whose gate
/// no rank passes until every rank has published B — which each rank does only
/// after finishing its own A read. The intervening collective's rendezvous IS
/// the barrier. That argument needs consecutive collectives to alternate slots;
/// it breaks the moment two collectives in a row use the same one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerLayout {
    /// Model hidden size `H`.
    pub hidden: u32,
    /// Largest number of tokens one dispatch reduces. **1 for decode; the
    /// prefill CHUNK size for prefill** — the message is `t·H·2` bytes, so this
    /// is the difference between a 15 KB peer region and a 470 MB one.
    pub max_tokens: u32,
    /// How many cross-GPU counters this rank owns.
    pub n_xctr: u32,
    /// Bytes per partial slot, `max_tokens·H·2`.
    partial_bytes: u64,
    /// Byte offset of the counter sub-region within the peer scratch.
    xctr_off: u64,
    /// Byte offset of the compact-audit status line.
    xstatus_off: u64,
    /// Byte offset / slot size of the tagged one-shot XReduce region
    /// (`PlowProgram::xr_tag_off` / `xr_tag_slot`).
    xr_tag_off: u64,
    xr_tag_slot: u64,
    /// Total peer-mapped bytes per rank.
    total: u64,
}

/// Partial slots per rank. Slots 0/1 are the two all-reduces every TP layer has
/// (o_proj, down); see [`PeerLayout`] for why the next layer may reuse them.
///
/// SLOT 2 IS A GATHER SLOT, and it is why this is 3 and not 2. Kimi-K3's LatentMoE
/// tail has a COLUMN-parallel producer (`routed_expert_up_proj`, rank r owning hidden
/// columns `[r*H/N, (r+1)*H/N)`) whose result is ADDED to a row-parallel one (the
/// shared expert's `down_proj`). `d_xreduce` folds the gather into the reduce, so the
/// two need one packet and one rendezvous between them — but they need two slots,
/// because the gathered partial is `H/N` wide per rank and the reduced one is `H`.
///
/// Reusing slot 0 or 1 instead is not available: the one-shot's gate says every peer
/// ARRIVED, not that every peer finished READING, so a slot may only be overwritten
/// after an intervening collective (`perf-data/archive/k3/kimi-k3-tp-peer-slots.md`). The up
/// projection sits between the expert-combine reduce (slot 1) and the shared-expert
/// reduce (slot 0) with no collective of its own to hide behind.
///
/// Cost is one more `max_tokens * hidden * 2` per rank — 58.7 MB at 4096 x 7168,
/// against ~190 GiB of weights. Inert for every model that emits no `act.ug_tp`:
/// the slot is simply never addressed.
pub const PARTIAL_SLOTS: u64 = 3;

impl PeerLayout {
    /// `PLOW_XR_TAG_SLOT_BYTES` (dev_isa.h): bytes per tagged one-shot slot.
    pub const XR_TAG_SLOT: u64 = 20480;
    /// Widest `[1, hidden]` message the slot holds at three bf16 per 8-byte word.
    pub const XR_TAG_MAX_WIDTH: u32 = (Self::XR_TAG_SLOT / 8 * 3) as u32;

    /// Cross-GPU counters a program needs: **one gate per one-shot `XReduce`
    /// (decode), two per `XReduceTwoShot` (prefill — the reduce-scatter and
    /// all-gather rendezvous)**, times the two all-reduces per layer.
    ///
    /// The gate count is not cosmetic: `devgen`'s `xgate` allocator hands out
    /// dense ids from 0, so a region sized for one-shot and fed a two-shot
    /// program has its last collectives signalling past the end of `xctr` —
    /// into whatever the pool handed out next, with no fault.
    pub fn counters_for(n_layers: u32, prefill: bool) -> u32 {
        n_layers * 2 * if prefill { 2 } else { 1 }
    }

    /// Lay out one rank's peer region.
    ///
    /// # Panics / errors
    /// Returns `None` when `max_tokens·hidden·2` is not 128 B-aligned, which
    /// would put the counter region off a cache line and let two ranks'
    /// independent signals contend for one line over XGMI.
    pub fn new(hidden: u32, max_tokens: u32, n_xctr: u32) -> Option<Self> {
        let partial_bytes = max_tokens as u64 * hidden as u64 * 2;
        if partial_bytes == 0 || partial_bytes % PEER_ALIGN != 0 {
            return None;
        }
        let xctr_off = partial_bytes * PARTIAL_SLOTS;
        let xstatus_off = xctr_off + n_xctr as u64 * XCTR_STRIDE as u64;
        // Tagged one-shot region: four slots (partial parity 0/1, gather parity 0/1) of
        // 8-byte words holding three bf16 + a 16-bit tag. It follows the status line — the
        // device derives it from the packet's status id and `PLOW_XR_TAG_SLOT_BYTES`, so the
        // slot is a constant here too — and the per-token zeroing covers it in one pass.
        let xr_tag_off = xstatus_off + XCTR_STRIDE as u64;
        let xr_tag_slot = Self::XR_TAG_SLOT;
        let total = xr_tag_off + 4 * xr_tag_slot;
        Some(PeerLayout {
            hidden,
            max_tokens,
            n_xctr,
            partial_bytes,
            xctr_off,
            xstatus_off,
            xr_tag_off,
            xr_tag_slot,
            total,
        })
    }

    /// Total peer-mapped bytes per rank.
    pub fn bytes(&self) -> u64 {
        self.total
    }

    /// Byte offset of partial slot `slot` — the `i2` operand of `DevOp::XReduce`.
    pub fn partial_off(&self, slot: u32) -> Result<u64> {
        if (slot as u64) >= PARTIAL_SLOTS {
            return Err(RuntimeError::Device(format!(
                "partial slot {slot} >= {PARTIAL_SLOTS}"
            )));
        }
        Ok(self.partial_bytes * slot as u64)
    }

    /// Byte offset of the counter sub-region: `xctr - peer_scratch[rank]`.
    pub fn xctr_off(&self) -> u64 {
        self.xctr_off
    }

    /// Bytes of the counter sub-region.
    pub fn xctr_bytes(&self) -> u64 {
        self.n_xctr as u64 * XCTR_STRIDE as u64
    }

    /// Byte offset of the isolated compact-audit status line.
    pub fn xstatus_off(&self) -> u64 {
        self.xstatus_off
    }

    /// The tagged one-shot region, right after the status line (four [`Self::XR_TAG_SLOT`]
    /// slots): what `d_xreduce_tagged_mega` derives from the status id.
    pub fn xr_tag_off(&self) -> u64 {
        self.xr_tag_off
    }

    pub fn xr_tag_slot(&self) -> u64 {
        self.xr_tag_slot
    }

    /// Bytes zeroed per token from `xctr`: counters, status line and the tagged region,
    /// which are contiguous. Tags are unique per (collective, token) only because every
    /// token starts from zeroed slots.
    fn xstate_bytes(&self) -> u64 {
        self.xctr_bytes() + XCTR_STRIDE as u64 + 4 * self.xr_tag_slot
    }
}

/// One rank: its backend, its peer-mapped scratch, and its device-resident
/// table of every rank's scratch base.
pub struct TpRank {
    backend: Arc<dyn Backend>,
    /// Position in THIS group — the index into `peer_scratch[]`.
    rank: u32,
    /// Device ordinal. Distinct from `rank`: replica 1 of a 2×TP4 node has
    /// ranks 0..4 on ordinals 4..8, and conflating the two would have every
    /// peer store addressed to the wrong GPU.
    ordinal: u8,
    /// Peer-mapped reduction region owned by this rank. Held so the mapping
    /// outlives every peer that has its address in a `peer_table`.
    scratch: DeviceMem,
    /// `[n_gpu]` device pointers — `PlowProgram::peer_scratch`. Local VRAM: it
    /// is read by this rank only, and putting it on the fabric would add a peer
    /// round-trip to every collective just to find out where to write.
    peer_table: DeviceMem,
    /// `scratch.base + layout.xctr_off()`, precomputed: it is read once per
    /// dispatch and the interpreter needs it as a bare pointer.
    xctr: u64,
    /// One isolated, peer-mapped line latched by compact device audit.
    xstatus: u64,
    /// Executor count — the persistent dispatch's grid. Read from the device,
    /// never assumed: a wrong value here is a wrong launch, not a wrong log
    /// line (the CU-count agent-info enum was off by two and reported 30115).
    executors: u32,
}

impl TpRank {
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// The device this rank runs on. Equals [`TpRank::rank`] only for the
    /// replica that starts at ordinal 0.
    pub fn ordinal(&self) -> u8 {
        self.ordinal
    }

    /// `PlowProgram::peer_scratch` — the `[n_gpu]` pointer table.
    pub fn peer_scratch_table(&self) -> u64 {
        self.peer_table.base
    }

    /// `PlowProgram::xctr` — this rank's counters, inside its own peer region.
    pub fn xctr(&self) -> u64 {
        self.xctr
    }

    pub fn xstatus(&self) -> u64 {
        self.xstatus
    }

    /// This rank's peer-mapped region base (== `peer_scratch[rank]`).
    pub fn scratch_base(&self) -> u64 {
        self.scratch.base
    }

    /// Grid size for this rank's persistent dispatch: one block per executor.
    pub fn executors(&self) -> u32 {
        self.executors
    }

    /// Host→device write into this rank's peer region at byte offset `off`.
    ///
    /// Bring-up and verification only — the per-token path never touches the
    /// peer region from the host. Bounds-checked against the layout because the
    /// region is a few tens of KB sitting next to nothing: an overrun would
    /// scribble on whatever the pool handed out next, with no fault.
    pub fn write_scratch(&self, off: u64, src: &[u8]) -> Result<()> {
        let view = self.scratch_view(off, src.len() as u64)?;
        self.backend.upload(&view, 0, src)
    }

    /// Device→host read from this rank's peer region at byte offset `off`.
    pub fn read_scratch(&self, off: u64, dst: &mut [u8]) -> Result<()> {
        let view = self.scratch_view(off, dst.len() as u64)?;
        self.backend.download(&view, 0, dst)
    }

    /// Push `bytes` from this rank's peer region into `dst`'s, over the fabric.
    ///
    /// BULK/verification path: this is the copy engine, whose 8 KB floor is
    /// 13.95 µs (measured) — 96 of those per token would be 1.3 ms of pure
    /// sync. The decode collective publishes with in-kernel peer stores gated
    /// by system-scope atomics instead, which is why this exists for bring-up
    /// checks and not for the interpreter.
    pub fn publish_to(&self, dst: &TpRank, src_off: u64, dst_off: u64, bytes: u64) -> Result<()> {
        let src = self.scratch_view(src_off, bytes)?;
        let dst_view = dst.scratch_view(dst_off, bytes)?;
        let peer = self.backend.peer().ok_or_else(|| {
            RuntimeError::Device(format!("rank {} lost its peer facility", self.rank))
        })?;
        peer.copy_peer_blocking(dst.ordinal, dst_view.base, src.base, bytes)
    }

    fn scratch_view(&self, off: u64, len: u64) -> Result<DeviceMem> {
        if off + len > self.scratch.len {
            return Err(RuntimeError::Device(format!(
                "peer-region access [{off}, {}) overruns rank {}'s {} B region",
                off + len,
                self.rank,
                self.scratch.len
            )));
        }
        Ok(DeviceMem::view(self.scratch.base + off, len))
    }
}

/// N co-resident ranks, their peer buffers, and the launch discipline.
///
/// A group is **model-scoped, not request-scoped**. Everything here is
/// allocated once at bring-up and every method takes `&self`, so the muxer
/// forms a batch and calls [`TpGroup::launch_token`] per step — a request never
/// owns the group, and successive steps may carry entirely different batches.
/// Anything that made a group belong to one request for its lifetime would
/// force continuous batching to choose between fragmenting decode batches and
/// stalling prefills, which is the failure the scheduler's hold window exists
/// to avoid.
pub struct TpGroup {
    ranks: Vec<TpRank>,
    layout: PeerLayout,
}

impl TpGroup {
    /// Split a node's devices into `⌊len/tp⌋` **independent** TP groups of
    /// degree `tp`, over contiguous ordinal runs.
    ///
    /// This is the deployment shape the target benchmark actually uses: on an
    /// 8-GPU node, `2 × TP4` replicas rather than one TP8 job, because TP=4
    /// keeps the expert GEMMs above their efficiency knee and the collectives
    /// cheap. The replicas share no peer buffer, no counter region, and no
    /// device, and each replica's buffers are named to that replica's agents
    /// only — so a *shader* on a rank of replica 0 has no mapping for replica
    /// 1's partials, which is stronger than "does not happen to read them".
    ///
    /// MEASURED CAVEAT: that guarantee is about the shader's address space, and
    /// only the shader's. `hsa_amd_memory_async_copy` names its two agents
    /// explicitly and the driver programs the copy engine from that pair, so a
    /// cross-replica D2D copy still succeeds despite the allow-lists (verified
    /// on this node — a test asserting otherwise failed, correctly). The
    /// mapping-level isolation is still untested: proving it needs a shader that
    /// dereferences another replica's pointer and faults, and
    /// [`crate::exec::amd_tp::AmdTpGroup`] only ever hands a rank its OWN
    /// replica's table.
    ///
    /// Contiguous runs, not strided: every pair on this node is 1-hop XGMI with
    /// uniform weight (measured — GPU4↔5 and GPU4↔7 give the same 60 GB/s and
    /// the same ~0.06 µs handshake), so the grouping is free to be the simple
    /// one. On a node with a real fabric hierarchy this is where a topology
    /// query would go.
    pub fn split_replicas(
        backends: Vec<Arc<dyn Backend>>,
        tp: u32,
        layout: PeerLayout,
    ) -> Result<Vec<Self>> {
        if tp == 0 {
            return Err(RuntimeError::Device("TP degree 0".into()));
        }
        if backends.len() < tp as usize {
            return Err(RuntimeError::Device(format!(
                "TP degree {tp} but only {} device(s) visible",
                backends.len()
            )));
        }
        let n = backends.len() / tp as usize;
        let mut it = backends.into_iter();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let members: Vec<Arc<dyn Backend>> = it.by_ref().take(tp as usize).collect();
            out.push(Self::bringup(members, layout)?);
        }
        Ok(out)
    }

    /// Bring up ONE TP group over `backends`, whose index in the slice IS the
    /// rank. The devices need not be ordinals `0..n` — see
    /// [`TpGroup::split_replicas`].
    ///
    /// Fails rather than degrades when a backend has no peer memory: a group
    /// that quietly fell back to host-staged collectives would still produce
    /// correct tokens, at ~13.95 µs per message — a 1.3 ms/token regression
    /// hiding behind a green test.
    pub fn bringup(backends: Vec<Arc<dyn Backend>>, layout: PeerLayout) -> Result<Self> {
        if backends.is_empty() {
            return Err(RuntimeError::Device("TP bring-up with zero devices".into()));
        }
        let n_gpu = backends.len() as u32;

        // The replica's membership, as device ordinals. This is the allow-list
        // every peer allocation below is mapped to, and nothing wider.
        let mut ordinals: Vec<u8> = Vec::with_capacity(backends.len());
        for (rank, be) in backends.iter().enumerate() {
            let peer = be.peer().ok_or_else(|| {
                RuntimeError::Device(format!(
                    "rank {rank} backend ({:?}) has no peer memory — TP needs \
                     peer-mapped VRAM for the reduction partials and counters",
                    be.class()
                ))
            })?;
            if peer.peer_agent_count() < n_gpu {
                return Err(RuntimeError::Device(format!(
                    "rank {rank} maps only {} agents but the group has {n_gpu} ranks",
                    peer.peer_agent_count()
                )));
            }
            ordinals.push(peer.ordinal());
        }
        {
            // Two ranks on one device would map a buffer to itself twice and
            // then reduce that rank's partial twice — a wrong token, silently.
            let mut seen = ordinals.clone();
            seen.sort_unstable();
            seen.dedup();
            if seen.len() != ordinals.len() {
                return Err(RuntimeError::Device(format!(
                    "TP group names device ordinals {ordinals:?} — a device appears twice"
                )));
            }
        }

        // Pass 1: allocate every rank's peer region. All of them must exist
        // before any pointer table can be filled, because rank 0's table names
        // rank N-1's buffer.
        let mut scratches: Vec<DeviceMem> = Vec::with_capacity(backends.len());
        for (rank, be) in backends.iter().enumerate() {
            let peer = be.peer().expect("checked above");
            let scratch = peer.alloc_peer(layout.bytes(), &ordinals)?;
            // Counters are 128 B-strided for cross-GPU line isolation, which is
            // only isolation if the region itself starts on a line.
            if scratch.base % PEER_ALIGN != 0 {
                return Err(RuntimeError::Device(format!(
                    "peer scratch for rank {rank} at {:#x} is not {PEER_ALIGN}-byte aligned",
                    scratch.base
                )));
            }
            scratches.push(scratch);
        }

        // Pass 2: publish the pointer table. Every rank gets the SAME table —
        // one virtual address per rank, valid on every device, because
        // `agents_allow_access` maps the owner's allocation into each peer's
        // address space at that address rather than at a per-peer alias.
        let table: Vec<u8> = scratches
            .iter()
            .flat_map(|s| s.base.to_le_bytes())
            .collect();

        let mut ranks = Vec::with_capacity(backends.len());
        for (rank, (be, scratch)) in backends.into_iter().zip(scratches).enumerate() {
            let peer_table = be.alloc(0, table.len() as u64)?;
            be.upload(&peer_table, 0, &table)?;
            let executors = be.enumerate().len() as u32;
            if executors == 0 {
                return Err(RuntimeError::Device(format!(
                    "rank {rank} reports zero executors — a zero-block dispatch"
                )));
            }
            ranks.push(TpRank {
                rank: rank as u32,
                ordinal: ordinals[rank],
                xctr: scratch.base + layout.xctr_off(),
                xstatus: scratch.base + layout.xstatus_off(),
                backend: be,
                scratch,
                peer_table,
                executors,
            });
        }

        Ok(TpGroup { ranks, layout })
    }

    pub fn n_gpu(&self) -> u32 {
        self.ranks.len() as u32
    }

    pub fn layout(&self) -> &PeerLayout {
        &self.layout
    }

    pub fn ranks(&self) -> &[TpRank] {
        &self.ranks
    }

    pub fn rank(&self, r: u32) -> Result<&TpRank> {
        self.ranks
            .get(r as usize)
            .ok_or_else(|| RuntimeError::Device(format!("rank {r} >= {}", self.ranks.len())))
    }

    /// Zero EVERY rank's cross-GPU counters.
    ///
    /// The one host obligation the deadlock argument rests on
    /// (the design notes): the cross edges form a publish→consume cut
    /// with no cycle *provided* no rank starts a token seeing a counter left
    /// over from the previous one. Zeroing rank-by-rank as each is launched
    /// would break exactly that — an early rank could signal a late rank's
    /// counter and then have the zeroing wipe the signal, and the late rank
    /// waits forever. So this is all-ranks-then-launch, and
    /// [`TpGroup::launch_token`] is the only sanctioned way to get the order
    /// right.
    pub fn zero_xctr(&self) -> Result<()> {
        for r in &self.ranks {
            r.backend
                .peer()
                .ok_or_else(|| {
                    RuntimeError::Device(format!("rank {} lost its peer facility", r.rank))
                })?
                .zero_peer(r.xctr, self.layout.xstate_bytes())?;
        }
        Ok(())
    }

    /// After a drain: assert no cross-GPU gate was left PARTIALLY signalled.
    ///
    /// # The failure this exists for
    ///
    /// `interp.hip` calls the collectives with `status = nullptr` and a
    /// `PLOW_XCTR_DEADLINE_TICKS` = 1 s deadline. On timeout the op **returns
    /// without reducing** — `out` keeps the previous layer's value — and nothing
    /// is recorded anywhere. The standalone bench gets `0xDEAD|rank` because it
    /// passes a real status word; production gets silence and a wrong token.
    /// A silently wrong answer is the worst failure mode available, so it is
    /// worth a host-side check even though the device could report it better.
    ///
    /// # What the counters give us for free
    ///
    /// Every rank signals every peer's gate exactly once per collective, so a
    /// gate that ran to completion reads exactly `n_gpu`. A gate that is neither
    /// `0` (this dispatch never reached that collective — the counter region is
    /// sized for the largest program, so a decode step legitimately leaves
    /// prefill's gates at zero) nor `n_gpu` means **some rank never arrived**,
    /// which is exactly the timeout's signature.
    ///
    /// # What it does NOT catch, stated plainly
    ///
    /// A rank that bails at its deadline and whose last peer signals immediately
    /// afterwards leaves the gate reading `n_gpu` anyway — complete by the
    /// counter, unreduced in fact. Closing that hole needs the DEVICE to record
    /// the bail (a status word the collectives already accept and the
    /// interpreter passes `nullptr` for). This check is a strict improvement on
    /// silence, not a substitute for that.
    /// `expect[g]` is the count gate `g` must hold after the dispatch that just
    /// drained — `0` for a gate this program does not use. Derived from the
    /// program by the caller, because the count is NOT uniform: a one-shot
    /// `XReduce` and a two-shot's reduce-scatter rendezvous take one signal per
    /// RANK, while the all-gather rendezvous takes one per rank per WORKGROUP
    /// (the reduced slice is written collaboratively, so no single workgroup may
    /// announce it — see `d_xreduce_twoshot_mega`).
    ///
    /// `None` marks an id that is not a counter at all. The counter region is
    /// also the only block of 128 B-aligned, peer-visible, host-zeroed bytes a
    /// collective can claim without a host binding, so `XArgmaxFin` publishes its
    /// folded u64 into a spare id (`op_collective.h:195`). That word is data —
    /// auditing it as an arrival count would fail on every correct step.
    pub fn audit_xctr(&self, expect: &[Option<u32>]) -> Result<()> {
        let n = (self.layout.n_xctr as usize).min(expect.len());
        let mut buf = vec![0u8; self.layout.xctr_bytes() as usize];
        for r in &self.ranks {
            r.read_scratch(self.layout.xctr_off(), &mut buf)?;
            for gate in 0..n {
                let Some(want) = expect[gate] else { continue };
                let at = gate * XCTR_STRIDE;
                let v = u32::from_le_bytes(buf[at..at + 4].try_into().expect("4 B"));
                if v != want {
                    return Err(RuntimeError::Device(format!(
                        "cross-GPU gate {gate} on rank {} reads {v}, expected {want}. \
                         {} The reduction did not complete as compiled, so a layer's \
                         output is not the sum of the ranks' partials and the token is \
                         wrong.",
                        r.rank,
                        if v < want {
                            "Some rank never arrived — a collective hit its deadline and \
                             returned WITHOUT reducing."
                        } else {
                            "MORE arrivals than the program can produce — a stale count \
                             survived from a previous dispatch (xctr must be zeroed on \
                             every rank before any rank launches)."
                        }
                    )));
                }
            }
        }
        Ok(())
    }

    /// Audit cross-GPU gates through their host-mapped large-BAR addresses.
    pub fn audit_xctr_direct(&self, expect: &[Option<u32>]) -> Result<()> {
        let n = (self.layout.n_xctr as usize).min(expect.len());
        for r in &self.ranks {
            if !r.backend.peer().is_some_and(|p| p.peer_host_writable()) {
                return Err(RuntimeError::Device(format!(
                    "rank {} peer memory is not host-mapped — direct TP audit needs large BAR",
                    r.rank
                )));
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            for (gate, expected) in expect.iter().take(n).enumerate() {
                let Some(want) = expected else { continue };
                let ptr = (r.xctr + (gate * XCTR_STRIDE) as u64) as *const u32;
                // SAFETY: `xctr` is host-mapped (checked above), each gate is a
                // 128-byte slot within the allocation, and the dispatch drained.
                let v = unsafe { std::ptr::read_volatile(ptr) };
                if v != *want {
                    return Err(RuntimeError::Device(format!(
                        "cross-GPU gate {gate} on rank {} reads {v}, expected {want}. \
                         Direct audit found an incomplete or stale collective.",
                        r.rank
                    )));
                }
            }
        }
        Ok(())
    }

    /// Read the device-compacted audit status: one large-BAR word per rank.
    pub fn audit_xstatus_direct(&self) -> Result<()> {
        for r in &self.ranks {
            if !r.backend.peer().is_some_and(|p| p.peer_host_writable()) {
                return Err(RuntimeError::Device(format!(
                    "rank {} peer memory is not host-mapped — compact TP audit needs large BAR",
                    r.rank
                )));
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            // SAFETY: `xstatus` is an isolated line in this host-mapped allocation,
            // and all interpreter and audit dispatches have drained.
            let status = unsafe { std::ptr::read_volatile(r.xstatus as *const u32) };
            if status != 0 {
                return Err(RuntimeError::Device(format!(
                    "compact cross-GPU audit failed on rank {} (status {status:#010x})",
                    r.rank
                )));
            }
        }
        Ok(())
    }

    /// Bring-up self-check: every ordered rank pair must be able to reach the
    /// other's peer region, byte-exact.
    ///
    /// `agents_allow_access` REPLACES a buffer's allow-list, so the classic
    /// failure is a group where only the last-named rank is actually mapped and
    /// every other peer faults on first touch — silently, at the first token,
    /// far from the allocation. Checking all `N·(N-1)` directed pairs at
    /// bring-up costs microseconds and moves that failure to the call that
    /// caused it.
    ///
    /// This proves **addressability** over the fabric, using the copy engine.
    /// It does NOT prove that a device-issued system-scope atomic on the region
    /// is coherent — that needs a kernel, and is covered by
    /// `runtime/tests/tp_p2p_bench` (re-measured working on ROCm 7.2.4, ~0.06 µs
    /// one-way).
    pub fn verify_peer_visibility(&self) -> Result<()> {
        let n = self.ranks.len();
        let bytes = self.layout.partial_off(1)? as usize;
        for src in 0..n {
            // A pattern keyed on the source rank: a mapping that silently
            // aliased two ranks' buffers would otherwise pass every compare.
            let pattern: Vec<u8> = (0..bytes)
                .map(|i| ((i * 31 + src * 7) % 251) as u8)
                .collect();
            let src_rank = &self.ranks[src];
            src_rank.write_scratch(0, &pattern)?;

            for dst in 0..n {
                if dst == src {
                    continue;
                }
                let dst_rank = &self.ranks[dst];
                src_rank.publish_to(dst_rank, 0, 0, bytes as u64)?;
                let mut back = vec![0u8; bytes];
                dst_rank.read_scratch(0, &mut back)?;
                if back != pattern {
                    let at = back.iter().zip(&pattern).position(|(a, b)| a != b);
                    return Err(RuntimeError::Device(format!(
                        "peer region of rank {dst} did not receive rank {src}'s bytes \
                         (first mismatch at byte {at:?}) — is every agent on the \
                         allow-list of the SAME allow_access call?"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Step one token: reset the counters per [`XctrReset`], then dispatch
    /// every rank once.
    ///
    /// `launch` is called once per rank, in rank order, with nothing between
    /// the calls. There is deliberately no barrier and no host wait here: the
    /// ranks rendezvous *on the device*, through the counters, inside their own
    /// dispatches. A host barrier between ranks would reintroduce exactly the
    /// launched-collective latency this design exists to avoid.
    ///
    /// A rank dispatched while a peer is not yet resident does not deadlock —
    /// it spins on an `xctr` its peer will bump once it arrives. What DOES
    /// deadlock is a mid-token launch of a fresh program on one rank only,
    /// because that rank's host-side zeroing would erase signals its peers
    /// already published. One token, one call.
    pub fn launch_token<F>(&self, reset: XctrReset, mut launch: F) -> Result<()>
    where
        F: FnMut(&TpRank) -> Result<()>,
    {
        match reset {
            XctrReset::Host => self.zero_xctr()?,
            XctrReset::HostDirect => self.zero_xctr_direct()?,
            XctrReset::Program => {}
        }
        for r in &self.ranks {
            launch(r)?;
        }
        Ok(())
    }

    /// Zero every rank's counters with plain host stores instead of the copy
    /// engine. 0.32 µs per 12 KiB region against 16.8 µs — measured.
    ///
    /// # Ordering, which is the whole risk
    ///
    /// This is only correct if the previous token's dispatch has already
    /// written its dirty counter lines back (the AQL packet's release fence)
    /// AND the next dispatch's acquire fence invalidates before the shader
    /// reads them. plowrt's HSA dispatch currently carries `AGENT`-scope
    /// acquire/release, which covers the device's own L2 — but `xctr` is a
    /// SYSTEM-scope region that peers also RMW, and whether an agent-scope
    /// acquire is enough to see a host BAR store into it is exactly the kind of
    /// question that must be answered on hardware, by a kernel.
    ///
    /// It **has not been**: [`crate::exec::amd_tp::AmdTpGroup`] can now run the
    /// experiment, but running it is not the same as having run it, and the
    /// answer decides whether tokens are correct. Hence [`XctrReset::Host`] is
    /// still the default and this is opt-in — the 16 µs is the price of not
    /// guessing about a memory model.
    pub fn zero_xctr_direct(&self) -> Result<()> {
        let n = self.layout.xstate_bytes() as usize;
        for r in &self.ranks {
            if !r.backend.peer().is_some_and(|p| p.peer_host_writable()) {
                return Err(RuntimeError::Device(format!(
                    "rank {} peer memory is not host-mapped — XctrReset::HostDirect \
                     needs large-BAR device VRAM",
                    r.rank
                )));
            }
            // SAFETY: `xctr` is the base of this rank's counter sub-region,
            // `n` bytes long by construction, and the allocation is mapped into
            // the host address space (checked immediately above). No device is
            // dispatched during this call — that is `launch_token`'s ordering.
            unsafe { std::ptr::write_bytes(r.xctr as *mut u8, 0, n) };
        }
        Ok(())
    }
}

/// Who clears the cross-GPU counters between tokens.
///
/// # Why this is a choice and not an implementation detail
///
/// plow's thesis is that a token is **one dispatch per GPU and nothing else** —
/// no per-op launch, no per-collective launch, no host sync inside the step.
/// Host counter-zeroing is the one thing on this path that is none of those and
/// still costs host time, so it is worth naming and measuring rather than
/// assuming away.
///
/// # Why the obvious device fix is wrong
///
/// "Have each rank zero its own `xctr` in a device epilogue" deadlocks. Ranks
/// are only synchronised at the *last* collective of a token; after it they
/// drift. A rank that reaches the next token's first collective before a peer
/// has run its epilogue signals that peer's counter, the peer's epilogue then
/// wipes the signal, and the peer waits forever. This is the same hazard §6d
/// rules out on the host side, reintroduced on the device.
///
/// The sound hostless form is **monotonic counters**: never reset, and let
/// collective `c`'s threshold in token `t` be `(t+1)·N`. A counter that only
/// grows cannot lose a signal to a racing reset, and the host's contribution
/// falls to the token index — a scalar it already uploads each step next to
/// `pos`/`kvlen`. That needs the device gate to scale its threshold by the
/// epoch, which is device work; until it lands, [`XctrReset::Host`] is correct
/// and this enum records the cost of being correct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XctrReset {
    /// The host zeroes every rank's counters before any rank is dispatched
    /// (§6d). Correct today, and the only mode a non-monotonic device gate can
    /// use. Costs one host pass over `n_gpu · n_xctr · 128 B` per token —
    /// measure it with `plowrt devices --tp N` before assuming it is free.
    /// At TP=4 with 96 counters that is **~32 µs/token**, which is MORE than
    /// the ~29 µs all 96 of the token's inline all-reduces cost put together.
    Host,
    /// Same semantics, but zeroed with host stores into large-BAR-mapped peer
    /// VRAM rather than the copy engine: **~1.3 µs/token at TP=4** instead of
    /// ~32. Opt-in, because its cache-ordering precondition has not been
    /// validated on hardware — see [`TpGroup::zero_xctr_direct`].
    HostDirect,
    /// The program's counters are monotonic (or otherwise self-managing), so
    /// the per-token host work is exactly `n_gpu` dispatches and nothing else.
    /// [`TpGroup::zero_xctr`] is still called ONCE at bring-up.
    Program,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma-4 12B decode: H=3840, 48 layers, one token. §7a predicts a tiny
    /// peer footprint; if a layout change blows that up, the invariant that
    /// only partials cross the fabric has quietly been abandoned.
    #[test]
    fn twelve_b_decode_peer_footprint_stays_tiny() {
        let l = PeerLayout::new(3840, 1, PeerLayout::counters_for(48, false)).unwrap();
        assert_eq!(l.n_xctr, 96, "one xctr gate per one-shot XReduce");
        assert_eq!(l.partial_off(0).unwrap(), 0);
        // `devgen`'s emit_xreduce passes slot_b = h*2 for decode. If these ever
        // disagree, every rank's `down` partial lands where no peer reads it.
        assert_eq!(l.partial_off(1).unwrap(), 3840 * 2);
        // Slot 2 is the GATHER slot (`PARTIAL_SLOTS`). Gemma emits no `act.ug_tp`, so it
        // is never addressed here — it is still LAID OUT, because the counter region's
        // offset has to be the same in every rank's region and therefore in every model's.
        assert_eq!(l.partial_off(2).unwrap(), 2 * 3840 * 2);
        assert_eq!(l.xctr_off(), 3 * 3840 * 2);
        assert_eq!(l.xctr_bytes(), 96 * 128);
        assert_eq!(l.xstatus_off(), l.xctr_off() + l.xctr_bytes());
        // The tagged one-shot region: four constant slots directly after the status line
        // (the device derives that address from the packet's status id), inside the
        // per-token zeroing, and wide enough for K3's 7168 at three bf16 per word.
        assert_eq!(l.xr_tag_off(), l.xstatus_off() + 128);
        assert_eq!(l.xr_tag_slot(), 20480);
        assert_eq!(PeerLayout::XR_TAG_MAX_WIDTH, 7680);
        assert_eq!(l.bytes(), l.xr_tag_off() + 4 * 20480);
        assert_eq!(l.xstate_bytes(), l.bytes() - l.xctr_off());
        // 23 KiB of partials + 12 KiB of counters + 80 KiB of tagged slots.
        assert!(l.bytes() < 128 * 1024, "peer footprint {} B", l.bytes());
    }

    /// The device-side constant and the host's must agree: read it from dev_isa.h.
    #[test]
    fn tagged_slot_matches_dev_isa() {
        let hdr = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../runtime/common/dev_isa.h"
        ))
        .expect("dev_isa.h");
        let line = hdr
            .lines()
            .find(|l| l.starts_with("#define PLOW_XR_TAG_SLOT_BYTES"))
            .expect("PLOW_XR_TAG_SLOT_BYTES in dev_isa.h");
        let v: u64 = line
            .split_whitespace()
            .nth(2)
            .unwrap()
            .trim_end_matches('u')
            .parse()
            .unwrap();
        assert_eq!(v, PeerLayout::XR_TAG_SLOT);
    }

    /// Prefill is the case that breaks a decode-shaped layout: the message is
    /// `t·h·2`, so a 2048-token chunk is 2048× the decode partial, and two-shot
    /// takes TWO gates per collective instead of one.
    #[test]
    fn prefill_scales_with_the_chunk_and_doubles_the_gates() {
        let h = 7168; // Kimi-class hidden
        let l = PeerLayout::new(h, 2048, PeerLayout::counters_for(48, true)).unwrap();
        assert_eq!(l.n_xctr, 192, "two xctr gates per XReduceTwoShot");
        // devgen: slot_b = t*h*2.
        assert_eq!(l.partial_off(1).unwrap(), 2048 * h as u64 * 2);
        assert_eq!(
            l.bytes(),
            PARTIAL_SLOTS * 2048 * h as u64 * 2 + (192 + 1) * 128 + 4 * 20480
        );
        // ~84 MiB of peer VRAM: negligible next to weights, but three orders of
        // magnitude past the decode region — which is the whole reason
        // `max_tokens` is a parameter and not an assumption.
        assert!(l.bytes() > 50 << 20);
    }

    /// Every sub-region must start on a 128 B line: two counters sharing a line
    /// serialise two ranks' independent signals into one contended fabric
    /// round-trip. A width that cannot honour that is rejected, not rounded —
    /// rounding here would silently desynchronise from `devgen`'s raw `t·h·2`.
    #[test]
    fn unaligned_widths_are_rejected_not_rounded() {
        assert!(
            PeerLayout::new(3, 1, 4).is_none(),
            "3*2 = 6 B is not line-aligned"
        );
        assert!(PeerLayout::new(0, 1, 4).is_none());
        assert!(PeerLayout::new(64, 1, 4).is_some(), "64*2 = 128 B is");

        let l = PeerLayout::new(64, 1, 4).unwrap();
        for slot in 0..PARTIAL_SLOTS as u32 {
            assert_eq!(l.partial_off(slot).unwrap() % 128, 0);
        }
        assert!(l.partial_off(PARTIAL_SLOTS as u32).is_err());
    }
}
