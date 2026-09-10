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

#[cfg(feature = "hsa")]
use std::path::Path;

use super::bench::EngineDiagnostics;
#[cfg(feature = "hsa")]
use super::bench::{DecodeSelection, PrefillSelection, RankAgreement};

/// The device engine serving one slug.
pub enum ServeEngine {
    /// The sm_120 persistent-interpreter engine (slotted, continuous batching).
    #[cfg(feature = "cuda")]
    Cuda(crate::exec::gpu::GpuEngine),
    /// The gfx950 engine (single sequence, optionally tensor-parallel).
    #[cfg(feature = "hsa")]
    Amd(AmdServe),
    /// The CPU engine (single sequence, persistent pinned workers).
    #[cfg(feature = "cpu")]
    Cpu(CpuServe),
}

#[cfg(feature = "cpu")]
pub use super::cpu_serve::CpuServe;

/// What the mux's single-sequence tick needs from an engine: slot bookkeeping,
/// whole-or-chunked prefill, one decode step per live slot, plus the AMD-only
/// packed-prefill / deferred multistep extensions, which other engines decline
/// by returning `None`/`Err` (the tick already handles both).
#[cfg(any(feature = "hsa", feature = "cpu"))]
pub trait SeqEngine {
    fn stop_ids(&self) -> &Arc<Vec<u32>>;
    fn batch(&self) -> usize;
    fn release(&mut self, slot: usize);
    fn prefill_turn(&self) -> usize;
    fn advance_prefill_turn(&mut self, slot: usize);
    fn prefill_prog_t(&self, prog: usize) -> Option<u32>;
    /// The most request spans one packed-prefill launch of `prog` may carry, bounded by its
    /// recurrent (D-class) operators. `u32::MAX` = unbounded. See
    /// `plans/unified-token-batch.md` §5.4 and `exec::amd_packed::recurrent_span_limit`.
    /// The default is for backends with no packed-prefill route of their own.
    fn packed_prefill_span_limit(&self, _prog: usize) -> u32 {
        u32::MAX
    }
    fn packable_prefill_span(&self, slot: usize, max_rows: u32)
        -> Option<packet::dev::PrefillSpan>;
    fn advance_packed_prefill(&mut self, members: &[(usize, &[u32])]) -> crate::Result<()>;
    fn prefill_frontier(&self, slot: usize) -> Option<usize>;
    fn prefill_chunked_at_most(
        &mut self,
        slot: usize,
        prompt: &[u32],
        tick_max_bucket: u32,
    ) -> crate::Result<Option<u32>>;
    fn multistep_quantum(&self, feeds: &[(usize, u32)], requested: usize) -> Option<usize>;
    fn multi_step(
        &mut self,
        feeds: &[(usize, u32)],
        quantum: usize,
        out: &mut Vec<u32>,
    ) -> crate::Result<usize>;
    fn mixed_step_rows(&self, _decode_rows: usize, _prefill_rows: usize) -> Option<u32> {
        None
    }
    fn mixed_prefill_rows(&self, _slot: usize, _prompt: &[u32], _max_rows: u32) -> u32 {
        0
    }
    fn mixed_prefill_fits(&self, _slot: usize, _prefill_capacity: u32) -> bool {
        false
    }
    fn prepare_packed_prefill_slot(
        &mut self,
        _slot: usize,
        _prompt: &[u32],
        _max_rows: u32,
    ) -> crate::Result<()> {
        Ok(())
    }
    fn terminal_prefill_ready(&self, _slot: usize, _prompt: &[u32]) -> bool {
        false
    }
    fn finish_prefill_batch(
        &mut self,
        _feeds: &[(usize, u32)],
        _members: &[(usize, &[u32])],
    ) -> crate::Result<Vec<(usize, u32)>> {
        Err(crate::RuntimeError::Rejected(
            "batched prefill completion unavailable".into(),
        ))
    }
    fn mixed_step(
        &mut self,
        _rows: u32,
        _feeds: &[(usize, u32)],
        _members: &[(usize, &[u32], u32)],
        _output: &mut [u32],
    ) -> crate::Result<()> {
        Err(crate::RuntimeError::Rejected(
            "mixed step unavailable".into(),
        ))
    }

    // ---- unified token batch (`plans/unified-token-batch.md` §7) ----------------------------
    //
    // Deliberately NOT hung off the mixed-step surface. The two routes select different
    // buckets (the token batch needs `nsplit == 1`), admit different work (it does not need a
    // decode row, and it may complete a prompt in the same step) and commit differently, so a
    // mux arm that reused `mixed_step_rows` would silently admit a plan the other route cannot
    // execute. Every default declines, so no other backend changes.

    /// Row capacity of the bucket that would carry `leading_rows` sampled rows — decode
    /// requests plus prompts completing this step — and `prefill_rows` body rows. `None` when
    /// this engine has no token-batch route or no bucket fits.
    fn token_batch_rows(&self, _leading_rows: usize, _prefill_rows: usize) -> Option<u32> {
        None
    }
    /// Rows of `slot`'s prompt this route may take next. Unlike `mixed_prefill_rows` this does
    /// NOT hold the prompt's last token back: consuming it is the point.
    fn token_batch_prefill_rows(&self, _slot: usize, _prompt: &[u32], _max_rows: u32) -> u32 {
        0
    }
    fn token_batch_prefill_fits(&self, _slot: usize, _prefill_capacity: u32) -> bool {
        false
    }
    /// Run one token batch. `output` receives one id per leading row: the `feeds` in order,
    /// then the members whose prompt completes in this step, in order. Returns how many of
    /// those trailing ids belong to completed prompts.
    fn token_batch_step(
        &mut self,
        _rows: u32,
        _feeds: &[(usize, u32)],
        _members: &[(usize, &[u32], u32)],
        _output: &mut Vec<u32>,
    ) -> crate::Result<Vec<usize>> {
        Err(crate::RuntimeError::Rejected(
            "token batch unavailable".into(),
        ))
    }
    fn step_batch(&mut self, feeds: &[(usize, u32)]) -> crate::Result<Vec<(usize, u32)>>;
}

impl ServeEngine {
    /// The single-sequence tick surface, for the gfx950 and CPU engines.
    #[cfg(any(feature = "hsa", feature = "cpu"))]
    pub fn seq_engine_mut(&mut self) -> Option<&mut dyn SeqEngine> {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(_) => None,
            #[cfg(feature = "hsa")]
            ServeEngine::Amd(e) => Some(e),
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(e) => Some(e),
        }
    }

    #[cfg(feature = "hsa")]
    pub fn amd_overlap_capability(&self) -> Option<crate::exec::amd::AmdOverlapCapability> {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(_) => None,
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(_) => None,
            ServeEngine::Amd(e) => Some(e.overlap_capability()),
        }
    }

    #[cfg(feature = "hsa")]
    pub fn amd_overlap_evidence(&self) -> Option<Vec<crate::exec::amd::AmdOverlapRankEvidence>> {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(_) => None,
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(_) => None,
            ServeEngine::Amd(e) => Some(e.overlap_evidence()),
        }
    }

    pub fn begin_diagnostics(&mut self) {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(_) => {}
            #[cfg(feature = "hsa")]
            ServeEngine::Amd(e) => e.begin_diagnostics(),
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(_) => {}
        }
    }

    pub fn finish_diagnostics(&mut self) -> EngineDiagnostics {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(_) => EngineDiagnostics::unsupported(),
            #[cfg(feature = "hsa")]
            ServeEngine::Amd(e) => e.finish_diagnostics(),
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(_) => EngineDiagnostics::unsupported(),
        }
    }

    /// Sequences one decode launch advances — the mux sizes its slot table to
    /// this, so mux slot `i` IS engine slot `i`.
    pub fn batch(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(e) => e.batch(),
            #[cfg(feature = "hsa")]
            ServeEngine::Amd(e) => e.batch(),
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(e) => e.batch(),
        }
    }

    /// Compiled decode widths, ascending. Allocated once by the mux at model load.
    pub fn decode_rungs(&self) -> Box<[u32]> {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(e) => e.effective_decode_rungs(),
            #[cfg(feature = "hsa")]
            ServeEngine::Amd(e) => e.decode_rungs().into(),
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(e) => e.decode_rungs().into(),
        }
    }

    /// The checkpoint's stop-token set.
    pub fn stop_ids(&self) -> &Arc<Vec<u32>> {
        match self {
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(e) => e.stop_ids(),
            #[cfg(feature = "hsa")]
            ServeEngine::Amd(e) => e.stop_ids(),
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(e) => SeqEngine::stop_ids(e),
        }
    }

    /// Write the last completed raw AMD packet trace.
    ///
    /// The trace buffer is allocated only when `--trace-raw` / `PLOW_TRACE_RAW`
    /// is set before engine load. `PLOW_TRACE_ALLRANKS=1` writes TP ranks as
    /// `path.rkN`; otherwise only rank 0 is written. Callers must quiesce the
    /// model mux first.
    #[cfg(feature = "hsa")]
    pub fn write_amd_packet_trace(&self, path: &Path) -> crate::Result<()> {
        match self {
            ServeEngine::Amd(e) => e.write_packet_trace(path),
            #[cfg(feature = "cuda")]
            ServeEngine::Cuda(_) => Err(crate::RuntimeError::Device(
                "raw AMD packet traces require an AMD serving engine".into(),
            )),
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(_) => Err(crate::RuntimeError::Device(
                "raw AMD packet traces require an AMD serving engine".into(),
            )),
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
            #[cfg(feature = "cpu")]
            ServeEngine::Cpu(_) => None,
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
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use crate::exec::amd::{
        derive_overlap_capability, AmdEngine, AmdOverlapCapability, AmdOverlapRankEvidence,
    };
    use crate::exec::amd_tp::AmdTpGroup;
    use crate::{Result, RuntimeError};
    use packet::dev::PrefillSpan;

    use super::{DecodeSelection, EngineDiagnostics, PrefillSelection, RankAgreement};

    /// One or N ranks of a gfx950 model, driven as ONE sequence.
    ///
    /// `AmdTpGroup::load` refuses `n < 2` (the peer region and the collectives
    /// only exist in a sharded packet), so tp==1 is `AmdEngine` directly rather
    /// than a degenerate group.
    enum Ranks {
        One(AmdEngine),
        Tp(AmdTpGroup),
    }

    const DEFAULT_SNAPSHOT_TENSORS: &str = "act.qa,act.oat,act.attn,act.xn";
    const MAX_SNAPSHOT_TENSORS: usize = 16;
    const MAX_SNAPSHOT_TENSOR_NAME: usize = 128;
    const MAX_SNAPSHOT_TENSOR_BYTES: u64 = 64 << 20;
    const SNAPSHOT_KV_BYTES: usize = 65_536;

    struct TensorSnapshotConfig {
        dir: PathBuf,
        slot: usize,
        tensors: Vec<String>,
    }

    impl Ranks {
        fn plan_span_at_most(
            &self,
            from: u32,
            to: u32,
            cap: u32,
        ) -> Result<Vec<crate::exec::amd::ChunkStep>> {
            match self {
                Self::One(e) => {
                    let chunks = e.plan_for_at_most(to.saturating_sub(from), cap)?;
                    e.chunk_steps_from(&chunks, from, to)
                }
                Self::Tp(g) => g.plan_span_at_most(from, to, cap),
            }
        }

        fn restore_carried(&mut self, slot: usize) -> Result<()> {
            match self {
                Self::One(e) => e.restore_carried(slot),
                Self::Tp(g) => g.restore_carried(slot),
            }
        }

        fn snapshot_carried(&mut self, slot: usize) -> Result<()> {
            match self {
                Self::One(e) => e.snapshot_carried(slot),
                Self::Tp(g) => g.snapshot_carried(slot),
            }
        }

        fn kv_rebase_all(&mut self, slot: usize) -> Result<()> {
            match self {
                Self::One(e) => e.kv_rebase(slot),
                Self::Tp(g) => g.kv_rebase_all(slot),
            }
        }

        fn prefill_chunk(
            &mut self,
            prompt: &[u32],
            step: crate::exec::amd::ChunkStep,
        ) -> Result<()> {
            match self {
                Self::One(e) => e.prefill_chunk(prompt, step),
                Self::Tp(g) => g.prefill_chunk(prompt, step),
            }
        }

        fn read_prefill_token(&mut self) -> Result<u32> {
            match self {
                Self::One(e) => e.read_sampled(),
                Self::Tp(g) => AmdTpGroup::agree(&g.read_sampled_all()?),
            }
        }

        fn overlap_evidence(&self) -> Vec<AmdOverlapRankEvidence> {
            match self {
                Ranks::One(e) => vec![e.overlap_evidence(0)],
                Ranks::Tp(g) => (0..g.n_gpu())
                    .map(|rank| g.rank(rank).overlap_evidence(rank))
                    .collect(),
            }
        }

        fn rank0(&self) -> &AmdEngine {
            match self {
                Ranks::One(e) => e,
                Ranks::Tp(g) => g.rank(0),
            }
        }

        fn ctr_snapshot(&mut self, program: usize) -> Result<Vec<u32>> {
            match self {
                Ranks::One(e) => e.ctr_word0_snapshot(program),
                Ranks::Tp(g) => g.ctr_snapshot(program),
            }
        }

        fn data_snapshot(
            &mut self,
            slot: usize,
            tensors: &[String],
        ) -> Result<Vec<(String, Vec<u8>)>> {
            match self {
                Ranks::One(e) => {
                    let mut out = e.snapshot_kv_slot(slot, SNAPSHOT_KV_BYTES)?;
                    for name in tensors {
                        out.push((name.clone(), e.snapshot_tensor(name)?));
                    }
                    Ok(out)
                }
                Ranks::Tp(g) => g.data_snapshot(slot, SNAPSHOT_KV_BYTES, tensors),
            }
        }
    }

    fn parse_snapshot_tensors(spec: Option<&str>) -> Result<Vec<String>> {
        let spec = spec.unwrap_or(DEFAULT_SNAPSHOT_TENSORS);
        let tensors = spec
            .split(',')
            .map(str::trim)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if tensors.is_empty() || tensors.iter().any(String::is_empty) {
            return Err(RuntimeError::Device(
                "AMD snapshot tensor list contains an empty name".into(),
            ));
        }
        if tensors.len() > MAX_SNAPSHOT_TENSORS {
            return Err(RuntimeError::Device(format!(
                "AMD snapshot tensor list has {} entries; maximum is {MAX_SNAPSHOT_TENSORS}",
                tensors.len()
            )));
        }
        for (i, name) in tensors.iter().enumerate() {
            if name.len() > MAX_SNAPSHOT_TENSOR_NAME {
                return Err(RuntimeError::Device(format!(
                    "AMD snapshot tensor name {name:?} is longer than {MAX_SNAPSHOT_TENSOR_NAME} bytes"
                )));
            }
            if tensors[..i].contains(name) {
                return Err(RuntimeError::Device(format!(
                    "duplicate AMD snapshot tensor {name:?}"
                )));
            }
            let component = snapshot_file_component(name);
            if tensors[..i]
                .iter()
                .any(|prior| snapshot_file_component(prior) == component)
            {
                return Err(RuntimeError::Device(format!(
                    "AMD snapshot tensor {name:?} aliases another output filename"
                )));
            }
        }
        Ok(tensors)
    }

    fn validate_tensor_snapshot(engine: &AmdEngine, slot: usize, tensors: &[String]) -> Result<()> {
        if slot >= engine.batch() {
            return Err(RuntimeError::Device(format!(
                "AMD snapshot slot {slot} past engine batch {}",
                engine.batch()
            )));
        }
        let mut total = 0u64;
        for name in tensors {
            let bytes = engine.tensor_bytes(name).ok_or_else(|| {
                RuntimeError::Device(format!("AMD snapshot tensor {name:?} is not declared"))
            })?;
            total = total.checked_add(bytes).ok_or_else(|| {
                RuntimeError::Device("AMD snapshot tensor byte count overflow".into())
            })?;
        }
        if total > MAX_SNAPSHOT_TENSOR_BYTES {
            return Err(RuntimeError::Device(format!(
                "AMD snapshot tensors total {total} bytes; maximum is {MAX_SNAPSHOT_TENSOR_BYTES}"
            )));
        }
        Ok(())
    }

    fn prepare_snapshot_dir(path: &Path) -> Result<()> {
        std::fs::create_dir_all(path).map_err(|source| RuntimeError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn write_snapshot(path: PathBuf, bytes: &[u8]) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| RuntimeError::Io {
                path: path.clone(),
                source,
            })?;
        file.write_all(bytes)
            .map_err(|source| RuntimeError::Io { path, source })
    }

    fn snapshot_file_component(name: &str) -> String {
        name.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                    c
                } else {
                    '_'
                }
            })
            .collect()
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
    /// decode dispatch advances ALL of the RUNG's rows whether or not they are
    /// live — the program's `t` is compiled, not passed — so an idle slot is fed
    /// `pos = 0, kvlen = 1, id = 0` and computes a throwaway token over KV row
    /// 0 of its own block. That is wasted work, and it is the reason a large
    /// `B` costs latency at low concurrency; it is not a correctness problem,
    /// because an idle slot's block is touched by nothing else.
    ///
    /// THE DECODE BATCH LADDER is what bounds that waste. A blob emitted with
    /// `PLOW_DECODE_BATCH_LADDER=1,2,4,8,16` carries one decode program per rung,
    /// and `dispatch_all` picks the narrowest rung that covers the occupied slots
    /// — so `B` is the CEILING on wasted rows, not the price of every step. `batch`
    /// stays the widest rung, because that is what the KV cache holds and what the
    /// mux sizes its slot table to; only the program under it moves.
    pub struct AmdServe {
        ranks: Ranks,
        stop_ids: Arc<Vec<u32>>,
        decode_rungs: Box<[u32]>,
        /// Sequences one decode dispatch advances (compiled `PLOW_DECODE_BATCH`).
        batch: usize,
        /// Next KV row each slot writes.
        pos: Vec<u32>,
        /// Whether each slot holds a live request.
        live: Vec<bool>,
        /// The token each slot feeds into the next dispatch. Idle slots feed 0.
        next_id: Vec<u32>,
        /// Reused host staging for one batched dispatch. These stay at the widest
        /// rung so steady decode does not allocate.
        pos_stage: Vec<u32>,
        kvlen_stage: Vec<u32>,
        parked_stage: Vec<u32>,
        advance_stage: Vec<usize>,
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
        /// Maximum compiled prefill rung selected per tick. `u32::MAX` = packet ladder cap.
        prefill_chunk_rows: u32,
        /// Next slot considered first by opt-in cross-request prefill fairness.
        prefill_turn: usize,
        /// Rows of this slot's prompt a CPU prefill head already wrote and
        /// transferred into its KV, consumed once by the next cursor as the
        /// resume point. 0 = no head.
        ///
        /// A head is another source of RESIDENT PREFIX ROWS, which is the thing
        /// the prefix cache already produces, so it joins at the same seam
        /// rather than adding a second notion of "already prefilled".
        head_rows: Vec<u32>,
        /// Width of the decode rung the last dispatch ran, so a rung CHANGE can be logged
        /// once instead of every step. It is the only externally visible evidence that the
        /// ladder is engaging, and a measurement that cannot show that is not a measurement.
        last_rung: u32,
        diagnostics: Option<EngineDiagnostics>,
        counter_snapshot_dir: Option<PathBuf>,
        tensor_snapshot: Option<TensorSnapshotConfig>,
        counter_snapshot_tick: u64,
        tensor_snapshot_tick: u64,
    }

    /// A prompt part-way through its prefill.
    struct PfCursor {
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

    fn split_pending_prefill(cur: &mut PfCursor, steps: Vec<crate::exec::amd::ChunkStep>) {
        let added = steps.len().saturating_sub(1);
        cur.steps.splice(cur.next..=cur.next, steps);
        if let Some(snap) = cur.snap_after.as_mut() {
            if *snap >= cur.next {
                *snap += added;
            }
        }
    }

    fn commit_packed_prefill(
        cursors: &mut [Option<PfCursor>],
        completed: &[(usize, crate::exec::amd::ChunkStep)],
    ) {
        for &(slot, step) in completed {
            let cur = cursors[slot]
                .as_mut()
                .expect("packed cursor was validated before dispatch");
            debug_assert_eq!(cur.steps.get(cur.next), Some(&step));
            cur.next += 1;
            cur.frontier = step.c0 + step.clen;
        }
    }

    fn packable_prefill_step(cur: &PfCursor, max_rows: u32) -> Option<crate::exec::amd::ChunkStep> {
        let step = *cur.steps.get(cur.next)?;
        (cur.next + 1 < cur.steps.len()
            && cur.snap_after != Some(cur.next)
            && step.clen <= max_rows)
            .then_some(step)
    }

    fn mixed_cursor_rows(cur: &PfCursor, prompt_rows: u32, max_rows: u32) -> u32 {
        let Some(step) = cur.steps.get(cur.next) else {
            return 0;
        };
        if cur.snap_after == Some(cur.next) || step.c0 != cur.frontier {
            return 0;
        }
        step.clen
            .min(prompt_rows.saturating_sub(1).saturating_sub(cur.frontier))
            .min(max_rows)
    }

    /// Rows the unified token batch may take from `cur`.
    ///
    /// The only difference from [`mixed_cursor_rows`] is the missing `- 1`: mixed step v1 holds
    /// the prompt's final token back so `finish_prefill_batch` can replay it through a
    /// decode-shaped pass. This route consumes it in the same step and samples its hidden row
    /// there, which is what removes that pass (`plans/unified-token-batch.md` §1, §6.5).
    fn token_batch_cursor_rows(cur: &PfCursor, prompt_rows: u32, max_rows: u32) -> u32 {
        let Some(step) = cur.steps.get(cur.next) else {
            return 0;
        };
        if cur.snap_after == Some(cur.next) || step.c0 != cur.frontier {
            return 0;
        }
        step.clen
            .min(prompt_rows.saturating_sub(cur.frontier))
            .min(max_rows)
    }

    fn mixed_prefill_padding_fits(frontier: u32, capacity: u32, max_ctx: usize) -> bool {
        capacity > 0
            && frontier
                .checked_add(capacity)
                .is_some_and(|end| end as usize <= max_ctx)
    }

    fn mixed_prefill_continuation_fits(
        frontier: u32,
        capacity: u32,
        pending_rows: u32,
        bucket: u32,
        max_ctx: usize,
    ) -> bool {
        // A partial mixed step retains its ordinary bucket at an advanced row offset.
        let partial = capacity.min(pending_rows.saturating_sub(1));
        mixed_prefill_padding_fits(frontier, capacity, max_ctx)
            && frontier
                .checked_add(partial)
                .and_then(|row| row.checked_add(bucket))
                .is_some_and(|end| end as usize <= max_ctx)
    }

    fn commit_mixed_prefill(cur: &mut PfCursor, rows: u32) {
        let step = &mut cur.steps[cur.next];
        debug_assert!(rows > 0 && rows <= step.clen);
        cur.frontier = step.c0 + rows;
        if rows == step.clen {
            cur.next += 1;
        } else {
            step.c0 += rows;
            step.clen -= rows;
        }
    }

    fn split_terminal_prefill(cur: &mut PfCursor) {
        if cur.snap_after.is_some() {
            return;
        }
        let Some(last) = cur.steps.last_mut() else {
            return;
        };
        if last.clen > 1 {
            let mut terminal = *last;
            last.clen -= 1;
            terminal.c0 += last.clen;
            terminal.clen = 1;
            cur.steps.push(terminal);
        }
    }

    fn terminal_prefill_cursor(cur: &PfCursor, prompt_rows: usize) -> bool {
        cur.snap_after.is_none()
            && cur.next + 1 == cur.steps.len()
            && cur.steps.get(cur.next).is_some_and(|step| {
                step.clen == 1
                    && step.c0 == cur.frontier
                    && (cur.frontier as usize).checked_add(1) == Some(prompt_rows)
            })
    }

    /// Shortest prefix worth caching. Below this the snapshot/restore pair costs more than the
    /// prefill it skips, and it churns the slot's cached prompt for nothing.
    const MIN_PREFIX: u32 = 128;

    fn stage_parked(parked: &mut [u32], advance: &[usize]) {
        parked.fill(1);
        for &slot in advance {
            parked[slot] = 0;
        }
    }

    fn invalidate_prefix_metadata(
        enabled: bool,
        cached_prompt: &mut [Vec<u32>],
        snap_at: &mut [u32],
        slot: usize,
    ) {
        if enabled {
            cached_prompt[slot].clear();
            snap_at[slot] = 0;
        }
    }

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
        pub fn load(blob_path: &Path, hsaco_dir: &Path, checkpoint: Option<&Path>) -> Result<Self> {
            let raw = std::fs::read(blob_path)
                .map_err(|e| RuntimeError::Device(format!("read {}: {e}", blob_path.display())))?;
            // METADATA ONLY -- this reads the TP fan-out to decide single-GPU vs TP group and
            // then drops the blob; it never dispatches. The L2-placement guard belongs to the
            // engine (which checks the code object for `plow_l2_place_dispatch_1`), so refusing
            // here just blocked `serve` on every placed blob before the engine saw it.
            let n_gpu = crate::asset::devblob::DevBlob::parse_l2(&raw, true)?
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
                Ranks::Tp(AmdTpGroup::load(
                    backends, blob_path, hsaco_dir, checkpoint,
                )?)
            };

            // `has_prefill`, not `n_programs == 1`: with a DECODE BATCH LADDER a decode-only
            // blob has one program PER RUNG, so counting programs would call a five-rung
            // decode-only packet "has prefill" and hand a whole prompt to a decode program.
            let (has_prefill, max_ctx, bound) = match &ranks {
                Ranks::One(e) => (e.has_prefill(), e.max_ctx(), e.weights_bound()),
                Ranks::Tp(g) => (g.rank(0).has_prefill(), g.max_ctx(), g.weights_bound()),
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
            let decode_rungs = match &ranks {
                Ranks::One(e) => e.decode_rungs(),
                Ranks::Tp(g) => g.rank(0).decode_rungs(),
            }
            .into_boxed_slice();
            let snapshot_cfg = &crate::config::RuntimeConfig::get().amd;
            let counter_snapshot_dir = snapshot_cfg.ctr_snap.as_deref().map(PathBuf::from);
            let tensor_snapshot = snapshot_cfg
                .tens_snap
                .as_deref()
                .map(|dir| {
                    let tensors = parse_snapshot_tensors(snapshot_cfg.snap_tensors.as_deref())?;
                    validate_tensor_snapshot(ranks.rank0(), snapshot_cfg.snap_slot, &tensors)?;
                    Ok(TensorSnapshotConfig {
                        dir: PathBuf::from(dir),
                        slot: snapshot_cfg.snap_slot,
                        tensors,
                    })
                })
                .transpose()?;
            if let Some(dir) = counter_snapshot_dir.as_deref() {
                prepare_snapshot_dir(dir)?;
            }
            if let Some(snapshot) = tensor_snapshot.as_ref() {
                prepare_snapshot_dir(&snapshot.dir)?;
            }
            tracing::info!(
                n_gpu,
                max_ctx,
                batch,
                decode_rungs = ?decode_rungs,
                decode_only = !has_prefill,
                pf_batch = crate::config::RuntimeConfig::get().nv.pf_batch,
                pf_chunk = crate::config::RuntimeConfig::get().nv.pf_chunk,
                pf_interleave = crate::config::RuntimeConfig::get().nv.pf_interleave,
                pf_defer_decode = crate::config::RuntimeConfig::get().nv.pf_defer_decode,
                stop_ids = ?stop_ids,
                "AMD serve engine ready"
            );
            Ok(AmdServe {
                ranks,
                stop_ids,
                decode_rungs,
                batch,
                pos: vec![0; batch],
                live: vec![false; batch],
                next_id: vec![0; batch],
                pos_stage: vec![0; batch],
                kvlen_stage: vec![1; batch],
                parked_stage: vec![1; batch],
                advance_stage: Vec::with_capacity(batch),
                decode_only: !has_prefill,
                max_ctx,
                prefix_cache: crate::config::RuntimeConfig::get().nv.prefix_cache,
                cached_prompt: vec![Vec::new(); batch],
                snap_at: vec![0; batch],
                pf: (0..batch).map(|_| None).collect(),
                chunk_prefill: !crate::config::RuntimeConfig::get().nv.pf_no_chunk,
                prefill_chunk_rows: match crate::config::RuntimeConfig::get().nv.pf_chunk {
                    0 => u32::MAX,
                    rows => rows,
                },
                prefill_turn: 0,
                head_rows: vec![0; batch],
                last_rung: 0,
                diagnostics: None,
                counter_snapshot_dir,
                tensor_snapshot,
                counter_snapshot_tick: 0,
                tensor_snapshot_tick: 0,
            })
        }

        pub fn begin_diagnostics(&mut self) {
            let overlap_ranges = Some(self.overlap_evidence());
            let rank_agreement = match &self.ranks {
                Ranks::One(_) => None,
                Ranks::Tp(g) => Some(RankAgreement {
                    ranks: g.n_gpu(),
                    sampled_token_every: g.agreement_cadence(),
                    counter_audit_every_dispatch: g.counter_audit_enabled(),
                    prefill_completion_all_ranks: true,
                }),
            };
            self.diagnostics = Some(EngineDiagnostics {
                supported: true,
                complete: true,
                overflowed: false,
                scope: "warmup_and_measured",
                prefill_selections: Vec::new(),
                decode_selections: Vec::new(),
                rank_agreement,
                amd_overlap_ranges: overlap_ranges,
            });
        }

        pub fn finish_diagnostics(&mut self) -> EngineDiagnostics {
            self.diagnostics
                .take()
                .unwrap_or_else(EngineDiagnostics::unsupported)
        }

        pub fn decode_rungs(&self) -> &[u32] {
            &self.decode_rungs
        }

        pub fn overlap_evidence(&self) -> Vec<AmdOverlapRankEvidence> {
            self.ranks.overlap_evidence()
        }

        pub fn overlap_capability(&self) -> AmdOverlapCapability {
            derive_overlap_capability(&self.overlap_evidence())
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

        /// Stage a mux-formed packed-prefill descriptor without dispatching it. Every span must
        /// name the same compiled program; the rank wrappers perform the full structural checks.
        pub fn stage_packed_prefill(
            &mut self,
            spans: &[PrefillSpan],
            parked: &[u32],
        ) -> Result<()> {
            let first = spans.first().ok_or_else(|| {
                RuntimeError::Device("packed prefill requires at least one span".into())
            })?;
            if spans.iter().any(|s| s.program != first.program) {
                return Err(RuntimeError::Device(
                    "packed prefill spans do not share one compiled program".into(),
                ));
            }
            let prog = first.program as usize;
            match &mut self.ranks {
                Ranks::One(e) => e.stage_packed_prefill(prog, spans, parked),
                Ranks::Tp(g) => g.stage_packed_prefill(prog, spans, parked),
            }
        }

        pub fn clear_packed_prefill(&mut self) {
            match &mut self.ranks {
                Ranks::One(e) => e.clear_packed_prefill(),
                Ranks::Tp(g) => g.clear_packed_prefill(),
            }
        }

        pub fn prefill_prog_t(&self, prog: usize) -> Option<u32> {
            match &self.ranks {
                Ranks::One(e) => e.prefill_prog_t(prog),
                Ranks::Tp(g) => g.prefill_prog_t(prog),
            }
        }

        /// §5.4's D-class span limit for `prog`. Under TP the ranks carry the same program, so
        /// the group's answer is the minimum -- a rank that refuses the program contributes 0
        /// and no span is admitted, which is the right answer for a collective step.
        pub fn packed_prefill_span_limit(&self, prog: usize) -> u32 {
            match &self.ranks {
                Ranks::One(e) => e.packed_prefill_span_limit(prog).unwrap_or(0),
                Ranks::Tp(g) => g.packed_prefill_span_limit(prog),
            }
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
            if let Some(diagnostics) = self.diagnostics.as_mut() {
                diagnostics.complete = false;
            }
            if prompt.is_empty() {
                return Err(RuntimeError::Rejected("empty prompt".into()));
            }
            if prompt.len() >= self.max_ctx {
                return Err(RuntimeError::ContextLength(format!(
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
            crate::obs::ttft::timed(&crate::obs::ttft::PF_STATE_CLEAR, || {
                match &mut self.ranks {
                    Ranks::One(e) => e.begin_slot(slot),
                    Ranks::Tp(g) => g.begin_slot(slot),
                }
            })?;
            self.pos[slot] = 0;
            self.live[slot] = true;
            let tok = if prompt.len() == 1 {
                // Nothing to consume: seed the single id and take one step,
                // which writes KV row 0 and samples the first token. This bypasses the prefix
                // planner, so invalidate the old prefix before that write.
                self.invalidate_prefix(slot);
                self.next_id[slot] = prompt[0];
                self.dispatch(slot)?
            } else if self.decode_only {
                // No bucket ladder to chunk over — walk the prompt through the
                // decode program one token at a time. Step `p` writes KV row
                // `p` and attends over `[0, p+1)`, so nothing is read that was
                // not written. This is what `runtime/tests/glm52_decode.c`
                // does; it is O(prompt) dispatches, hence a fallback.
                //
                // At batch > 1 every other slot still executes throwaway KV
                // work at its next row, but its recurrent state is parked and
                // its host position does not move — see `dispatch_all`.
                self.invalidate_prefix(slot);
                let mut last = 0;
                for id in prompt {
                    self.next_id[slot] = *id;
                    last = self.dispatch(slot)?;
                }
                last
            } else {
                let (resume, arm) = self.plan_prefix(slot, prompt);
                if resume == 0 {
                    self.invalidate_prefix(slot);
                }
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
                            tracing::debug!(
                                slot,
                                resume,
                                arm,
                                n = prompt.len(),
                                "amd: prefix cache"
                            );
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
            if let Some(diagnostics) = self.diagnostics.as_mut() {
                if !diagnostics.overflowed {
                    diagnostics.complete = true;
                }
            }
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

        /// A miss is about to overwrite this slot's KV from row zero. Make the old snapshot
        /// ineligible before the first write so cancellation or a device error cannot later pair
        /// old recurrent state with the newly overwritten KV rows.
        fn invalidate_prefix(&mut self, slot: usize) {
            invalidate_prefix_metadata(
                self.prefix_cache,
                &mut self.cached_prompt,
                &mut self.snap_at,
                slot,
            );
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
        /// `decode_only` or a 1-token prompt. Prefix-cache split points remain chunk boundaries.
        pub fn prefill_chunked(&mut self, slot: usize, prompt: &[u32]) -> Result<Option<u32>> {
            self.prefill_chunked_at_most(slot, prompt, u32::MAX)
        }

        fn prepare_prefill_cursor(
            &mut self,
            slot: usize,
            prompt: &[u32],
            tick_max_bucket: u32,
        ) -> Result<()> {
            self.check_slot(slot)?;
            if self.pf[slot].is_none() {
                if prompt.is_empty() {
                    return Err(RuntimeError::Rejected("empty prompt".into()));
                }
                if prompt.len() >= self.max_ctx {
                    return Err(RuntimeError::ContextLength(format!(
                        "prompt is {} tokens, max_ctx is {}",
                        prompt.len(),
                        self.max_ctx
                    )));
                }
                crate::obs::ttft::timed(&crate::obs::ttft::PF_STATE_CLEAR, || {
                    match &mut self.ranks {
                        Ranks::One(e) => e.begin_slot(slot),
                        Ranks::Tp(g) => g.begin_slot(slot),
                    }
                })?;
                let n = prompt.len() as u32;
                // CHUNKING AND THE PREFIX CACHE COMPOSE. The cache decides WHICH span still has
                // to be prefilled; chunking decides how that span is broken into ticks. Building
                // the cursor from the cached plan is all it takes — they were alternatives only
                // because the first cut of this function bailed out when the cache was on.
                let (resume, arm) = self.plan_prefix(slot, prompt);
                if resume == 0 {
                    self.invalidate_prefix(slot);
                }
                // A CPU head's rows are resident like a cache hit's, and are
                // consumed exactly once. Taken only when it beats what the
                // prefix cache found, so the two never fight over the resume
                // point; `begin_slot` above has already cleared the slot, which
                // is why the head is transferred AFTER that call, not before.
                let head = std::mem::take(&mut self.head_rows[slot]).min(n.saturating_sub(1));
                let resume = resume.max(head);
                let max_bucket = self.prefill_chunk_rows.min(tick_max_bucket);
                let g = &mut self.ranks;
                let (steps, snap_after) = {
                    if head > 0 && head >= resume {
                        // NO `restore_carried`: a head writes append-only KV and
                        // there is no prefix snapshot behind it. A model with
                        // carried recurrent state cannot take a head at all —
                        // `serve::head` refuses it at load — so there is nothing
                        // to restore rather than something being skipped.
                        (g.plan_span_at_most(head, n, max_bucket)?, None)
                    } else if resume > 0 {
                        g.restore_carried(slot)?;
                        (g.plan_span_at_most(resume, n, max_bucket)?, None)
                    } else if arm > 0 {
                        let head = g.plan_span_at_most(0, arm, max_bucket)?;
                        let tail = g.plan_span_at_most(arm, n, max_bucket)?;
                        let cut = head.len();
                        let mut all = head;
                        all.extend(tail);
                        (all, cut.checked_sub(1))
                    } else {
                        (g.plan_span_at_most(0, n, max_bucket)?, None)
                    }
                };
                self.pos[slot] = 0;
                // NOT live until the last chunk lands: `live` is what the mask keys on, and a
                // half-prefilled slot must stay parked.
                self.live[slot] = false;
                self.pf[slot] = Some(PfCursor {
                    steps,
                    next: 0,
                    frontier: resume,
                    snap_after,
                    resume,
                    arm,
                });
            }
            Ok(())
        }

        /// Advance one chunk, selecting only compiled packet rungs within this tick's budget.
        pub fn prefill_chunked_at_most(
            &mut self,
            slot: usize,
            prompt: &[u32],
            tick_max_bucket: u32,
        ) -> Result<Option<u32>> {
            let chunked = self.chunk_prefill && !self.decode_only && prompt.len() > 1;
            if !chunked {
                return self.prefill(slot, prompt).map(Some);
            }
            self.prepare_prefill_cursor(slot, prompt, tick_max_bucket)?;
            let g = &mut self.ranks;
            let max_bucket = self.prefill_chunk_rows.min(tick_max_bucket);
            let pending = {
                let cur = self.pf[slot].as_ref().expect("just built");
                cur.steps[cur.next]
            };
            if pending.clen > max_bucket {
                let split =
                    g.plan_span_at_most(pending.c0, pending.c0 + pending.clen, max_bucket)?;
                let cur = self.pf[slot].as_mut().expect("just built");
                split_pending_prefill(cur, split);
            }
            let cur = self.pf[slot].as_mut().expect("just built");
            let step = cur.steps[cur.next];
            if let Some(diagnostics) = self.diagnostics.as_mut() {
                diagnostics.push_prefill(PrefillSelection {
                    slot,
                    row_start: step.c0,
                    rows: step.clen,
                    bucket: g.rank0().prog_t(step.prog),
                });
            }
            tracing::debug!(
                slot,
                c0 = step.c0,
                clen = step.clen,
                chunk = cur.next,
                frontier = cur.frontier,
                "pf chunk"
            );
            // Rebase for this chunk and hand the base back before returning: the decode that runs
            // later in this same tick refuses a non-zero base.
            g.kv_rebase_all(slot)?;
            let r = g.prefill_chunk(prompt, step);
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
            let tok = g.read_prefill_token()?;
            tracing::debug!(slot, tok, n = prompt.len(), "pf complete");
            let n = prompt.len() as u32;
            let (resume, arm) = (cur.resume, cur.arm);
            self.pf[slot] = None;
            self.pos[slot] = n;
            self.live[slot] = true;
            if self.prefix_cache {
                // Same invariant as the whole-prompt path: the snapshot describes
                // `cached_prompt[..snap_at]`, so a hit keeps it, an arm replaces it, and anything
                // else makes it stale.
                self.cached_prompt[slot].clear();
                self.cached_prompt[slot].extend_from_slice(prompt);
                if arm > 0 {
                    self.snap_at[slot] = arm;
                } else if resume == 0 {
                    self.snap_at[slot] = 0;
                }
            }
            Ok(Some(tok))
        }

        pub fn mixed_step_rows(&self, decode_rows: usize, prefill_rows: usize) -> Option<u32> {
            if !self.chunk_prefill || self.decode_only || self.prefix_cache {
                return None;
            }
            match &self.ranks {
                Ranks::One(e) => e.mixed_step_rows(decode_rows, prefill_rows),
                Ranks::Tp(_) => None,
            }
        }

        pub fn prepare_packed_prefill_slot(
            &mut self,
            slot: usize,
            prompt: &[u32],
            max_rows: u32,
        ) -> Result<()> {
            if prompt.len() <= 1
                || (self.mixed_step_rows(1, 1).is_none()
                    && !self.packed_prefill_reachable(max_rows))
            {
                return Ok(());
            }
            self.check_slot(slot)?;
            if self.live[slot] {
                return Err(RuntimeError::Rejected(
                    "packed prefill cannot initialize a live decode slot".into(),
                ));
            }
            self.prepare_prefill_cursor(slot, prompt, max_rows)?;
            // Seeding the cursor and peeling the terminal row are two different features, and
            // only the second one needs fusion.
            //
            // Co-packing can only consider a slot that ALREADY has a cursor, and the sole other
            // thing that creates one is the isolated prefill path — which the mux skips on any
            // tick where a pack ran. So without up-front seeding a burst of N fresh requests
            // bootstraps to a two-member pack and stops: slots 3..N never get an isolated tick
            // in which to acquire a cursor. Gating this on `mixed_step_rows` capped every pack
            // at two members whenever fusion was off, which is the default.
            //
            // The terminal split stays gated, because it exists only to hand the last row to
            // `finish_prefill_batch`. With fusion off `terminal_prefill_ready` is false, so a
            // peeled 1-row chunk would just be an extra isolated launch per request.
            if self.mixed_step_rows(1, 1).is_some() {
                split_terminal_prefill(self.pf[slot].as_mut().expect("prepared cursor"));
            }
            Ok(())
        }

        pub fn terminal_prefill_ready(&self, slot: usize, prompt: &[u32]) -> bool {
            self.mixed_step_rows(1, 1).is_some()
                && self.live.get(slot) == Some(&false)
                && prompt.len() < self.max_ctx
                && self
                    .pf
                    .get(slot)
                    .and_then(Option::as_ref)
                    .is_some_and(|cur| terminal_prefill_cursor(cur, prompt.len()))
        }

        pub fn finish_prefill_batch(
            &mut self,
            feeds: &[(usize, u32)],
            members: &[(usize, &[u32])],
        ) -> Result<Vec<(usize, u32)>> {
            if members.is_empty() || self.mixed_step_rows(1, 1).is_none() {
                return Err(RuntimeError::Rejected(
                    "batched prefill completion unavailable".into(),
                ));
            }
            for (index, &(slot, _)) in feeds.iter().enumerate() {
                self.check_slot(slot)?;
                if !self.live[slot]
                    || self.pf[slot].is_some()
                    || self.pos[slot] as usize >= self.max_ctx
                    || feeds[..index].iter().any(|&(prior, _)| prior == slot)
                {
                    return Err(RuntimeError::Rejected(format!(
                        "invalid prefill completion decode slot {slot}"
                    )));
                }
            }
            for (index, &(slot, prompt)) in members.iter().enumerate() {
                if !self.terminal_prefill_ready(slot, prompt)
                    || feeds.iter().any(|&(prior, _)| prior == slot)
                    || members[..index].iter().any(|&(prior, _)| prior == slot)
                {
                    return Err(RuntimeError::Rejected(format!(
                        "invalid terminal prefill slot {slot}"
                    )));
                }
            }
            let mut all = Vec::with_capacity(feeds.len() + members.len());
            all.extend_from_slice(feeds);
            all.extend(
                members
                    .iter()
                    .map(|&(slot, prompt)| (slot, prompt[prompt.len() - 1])),
            );
            // Pending cursors stage their frontier as the decode position. Commit
            // their transition to live only after the sampled rows are available.
            let output = self.step_batch(&all)?;
            for &(slot, prompt) in members {
                self.pf[slot] = None;
                self.pos[slot] = prompt.len() as u32;
                self.live[slot] = true;
            }
            Ok(output)
        }

        pub fn mixed_prefill_rows(&self, slot: usize, prompt: &[u32], max_rows: u32) -> u32 {
            if slot >= self.batch || self.live[slot] || prompt.len() >= self.max_ctx {
                return 0;
            }
            let cap = max_rows.min(self.prefill_chunk_rows);
            match &self.pf[slot] {
                Some(cur) => mixed_cursor_rows(cur, prompt.len() as u32, cap),
                None => (prompt.len().saturating_sub(1) as u32).min(cap),
            }
        }

        pub fn mixed_prefill_fits(&self, slot: usize, prefill_capacity: u32) -> bool {
            self.pf.get(slot).is_some_and(|cursor| {
                // A shorter preceding chunk can increase the final span's parked suffix.
                // Reserve the full mixed prefill capacity before touching any cursor.
                let engine = self.ranks.rank0();
                let (frontier, pending_rows, bucket) = if let Some(cursor) = cursor {
                    let Some(step) = cursor.steps.get(cursor.next) else {
                        return false;
                    };
                    (cursor.frontier, step.clen, engine.prog_t(step.prog))
                } else {
                    let bucket = (0..engine.n_programs())
                        .filter_map(|prog| engine.prefill_prog_t(prog))
                        .filter(|&rows| rows <= self.prefill_chunk_rows)
                        .max()
                        .unwrap_or(0);
                    (0, bucket, bucket)
                };
                bucket > 0
                    && mixed_prefill_continuation_fits(
                        frontier,
                        prefill_capacity,
                        pending_rows,
                        bucket,
                        self.max_ctx,
                    )
            })
        }

        pub fn mixed_step(
            &mut self,
            rows: u32,
            feeds: &[(usize, u32)],
            members: &[(usize, &[u32], u32)],
            output: &mut [u32],
        ) -> Result<()> {
            use plow_asset::mixed_step::{DecodeRequest, PrefillRequest};
            if !self.chunk_prefill
                || self.decode_only
                || self.prefix_cache
                || !matches!(self.ranks, Ranks::One(_))
                || feeds.is_empty()
                || members.is_empty()
                || output.len() != feeds.len()
                || self.mixed_step_rows(
                    feeds.len(),
                    rows.saturating_sub(feeds.len() as u32) as usize,
                ) != Some(rows)
            {
                return Err(RuntimeError::Rejected("invalid AMD mixed step".into()));
            }
            for (index, &(slot, _)) in feeds.iter().enumerate() {
                self.check_slot(slot)?;
                if !self.live[slot]
                    || self.pf[slot].is_some()
                    || self.pos[slot] as usize >= self.max_ctx
                    || feeds[..index].iter().any(|&(prior, _)| prior == slot)
                {
                    return Err(RuntimeError::Rejected(format!(
                        "invalid mixed decode slot {slot}"
                    )));
                }
            }
            let mut admitted = feeds.len() as u32;
            for (index, &(slot, prompt, take)) in members.iter().enumerate() {
                self.check_slot(slot)?;
                if take == 0
                    || !self.mixed_prefill_fits(slot, rows.saturating_sub(feeds.len() as u32))
                    || take > self.mixed_prefill_rows(slot, prompt, u32::MAX)
                    || feeds.iter().any(|&(prior, _)| prior == slot)
                    || members[..index].iter().any(|&(prior, _, _)| prior == slot)
                {
                    return Err(RuntimeError::Rejected(format!(
                        "invalid mixed prefill slot {slot}"
                    )));
                }
                admitted = admitted
                    .checked_add(take)
                    .ok_or_else(|| RuntimeError::Rejected("mixed row count overflow".into()))?;
            }
            if admitted > rows {
                return Err(RuntimeError::Rejected(
                    "mixed rows exceed packet capacity".into(),
                ));
            }
            // Cursor setup may choose a shorter first chunk than admission estimated.
            // Retain the unused capacity as padding rather than crossing a planned boundary.
            let mut completed = Vec::with_capacity(members.len());
            for &(slot, prompt, take) in members {
                self.prepare_prefill_cursor(slot, prompt, self.prefill_chunk_rows)?;
                let take = self.mixed_prefill_rows(slot, prompt, take);
                if take == 0 {
                    return Err(RuntimeError::Rejected(format!(
                        "mixed prefill slot {slot} reached a boundary"
                    )));
                }
                completed.push((slot, take));
            }
            for slot in 0..self.batch {
                self.pos_stage[slot] = self.pf[slot]
                    .as_ref()
                    .map_or(self.pos[slot], |cursor| cursor.frontier);
            }
            let decode: Vec<_> = feeds
                .iter()
                .map(|&(slot, token)| DecodeRequest {
                    slot: slot as u32,
                    state_slot: slot as u32,
                    token,
                })
                .collect();
            let prefill: Vec<_> = members
                .iter()
                .zip(&completed)
                .map(|(&(slot, prompt, _), &(_, take))| {
                    let start = self.pos_stage[slot];
                    PrefillRequest {
                        slot: slot as u32,
                        state_slot: slot as u32,
                        start,
                        tokens: &prompt[start as usize..(start + take) as usize],
                        prompt_len: prompt.len() as u32,
                    }
                })
                .collect();
            let Ranks::One(e) = &mut self.ranks else {
                unreachable!()
            };
            e.mixed_step(rows, &decode, &prefill, &mut self.pos_stage, output)?;
            for &(slot, _) in feeds {
                self.pos[slot] = self.pos_stage[slot];
            }
            for (slot, take) in completed {
                commit_mixed_prefill(self.pf[slot].as_mut().expect("staged mixed cursor"), take);
            }
            Ok(())
        }

        pub fn token_batch_rows(&self, leading_rows: usize, prefill_rows: usize) -> Option<u32> {
            if !self.chunk_prefill || self.decode_only || self.prefix_cache {
                return None;
            }
            match &self.ranks {
                Ranks::One(e) => e.token_batch_rows(leading_rows, prefill_rows),
                Ranks::Tp(_) => None,
            }
        }

        pub fn token_batch_prefill_rows(&self, slot: usize, prompt: &[u32], max_rows: u32) -> u32 {
            if slot >= self.batch || self.live[slot] || prompt.len() > self.max_ctx {
                return 0;
            }
            let cap = max_rows.min(self.prefill_chunk_rows);
            match &self.pf[slot] {
                Some(cur) => token_batch_cursor_rows(cur, prompt.len() as u32, cap),
                None => (prompt.len() as u32).min(cap),
            }
        }

        pub fn token_batch_prefill_fits(&self, slot: usize, prefill_capacity: u32) -> bool {
            self.mixed_prefill_fits(slot, prefill_capacity)
        }

        /// One unified token-batch step.
        ///
        /// Returns the slots whose prompt completed here, in the order their sampled ids
        /// follow the decode feeds in `output`. Delivery is by logical request, never by row
        /// number: packed rows, physical slots and compact output rows are three different
        /// numberings.
        pub fn token_batch_step(
            &mut self,
            rows: u32,
            feeds: &[(usize, u32)],
            members: &[(usize, &[u32], u32)],
            output: &mut Vec<u32>,
        ) -> Result<Vec<usize>> {
            use plow_asset::mixed_step::{DecodeRequest, PrefillRequest};
            if !self.chunk_prefill
                || self.decode_only
                || self.prefix_cache
                || !matches!(self.ranks, Ranks::One(_))
                || members.is_empty()
            {
                return Err(RuntimeError::Rejected("invalid AMD token batch".into()));
            }
            for (index, &(slot, _)) in feeds.iter().enumerate() {
                self.check_slot(slot)?;
                if !self.live[slot]
                    || self.pf[slot].is_some()
                    || self.pos[slot] as usize >= self.max_ctx
                    || feeds[..index].iter().any(|&(prior, _)| prior == slot)
                {
                    return Err(RuntimeError::Rejected(format!(
                        "invalid token-batch decode slot {slot}"
                    )));
                }
            }
            for (index, &(slot, prompt, take)) in members.iter().enumerate() {
                self.check_slot(slot)?;
                if take == 0
                    || !self.token_batch_prefill_fits(slot, rows.saturating_sub(feeds.len() as u32))
                    || take > self.token_batch_prefill_rows(slot, prompt, u32::MAX)
                    || feeds.iter().any(|&(prior, _)| prior == slot)
                    || members[..index].iter().any(|&(prior, _, _)| prior == slot)
                {
                    return Err(RuntimeError::Rejected(format!(
                        "invalid token-batch prefill slot {slot}"
                    )));
                }
            }
            // Cursor setup may choose a shorter first chunk than admission estimated.
            let mut completed = Vec::with_capacity(members.len());
            for &(slot, prompt, take) in members {
                self.prepare_prefill_cursor(slot, prompt, self.prefill_chunk_rows)?;
                let take = self.token_batch_prefill_rows(slot, prompt, take);
                if take == 0 {
                    return Err(RuntimeError::Rejected(format!(
                        "token-batch prefill slot {slot} reached a boundary"
                    )));
                }
                completed.push((slot, take));
            }
            for slot in 0..self.batch {
                self.pos_stage[slot] = self.pf[slot]
                    .as_ref()
                    .map_or(self.pos[slot], |cursor| cursor.frontier);
            }
            let decode: Vec<_> = feeds
                .iter()
                .map(|&(slot, token)| DecodeRequest {
                    slot: slot as u32,
                    state_slot: slot as u32,
                    token,
                })
                .collect();
            let prefill: Vec<_> = members
                .iter()
                .zip(&completed)
                .map(|(&(slot, prompt, _), &(_, take))| {
                    let start = self.pos_stage[slot];
                    PrefillRequest {
                        slot: slot as u32,
                        state_slot: slot as u32,
                        start,
                        tokens: &prompt[start as usize..(start + take) as usize],
                        prompt_len: prompt.len() as u32,
                    }
                })
                .collect();
            // Which members finish their prompt here, in span order. That is exactly the set
            // whose hidden rows the step samples, and it is decided from the plan the device
            // will see, not from a phase tag.
            let finishing: Vec<usize> = prefill
                .iter()
                .filter(|r| r.start + r.tokens.len() as u32 == r.prompt_len)
                .map(|r| r.slot as usize)
                .collect();
            let leading = feeds.len() + finishing.len();
            let live: u32 = completed.iter().map(|&(_, take)| take).sum::<u32>()
                + feeds.len() as u32;
            if leading == 0 || live > rows {
                return Err(RuntimeError::Rejected(format!(
                    "token batch admits {live} rows and {leading} sampled rows against a \
                     {rows}-row bucket"
                )));
            }
            output.clear();
            output.resize(leading, 0);
            let Ranks::One(e) = &mut self.ranks else {
                unreachable!()
            };
            e.token_batch_step(rows, &decode, &prefill, &mut self.pos_stage, output)?;
            for &(slot, _) in feeds {
                self.pos[slot] = self.pos_stage[slot];
            }
            for (slot, take) in completed {
                if finishing.contains(&slot) {
                    // The prompt is consumed and its first generated token is in `output`.
                    // Retire the cursor and make the slot live, exactly as the terminal-prefill
                    // path did — except that no second transformer pass produced the token.
                    self.pf[slot] = None;
                    self.pos[slot] = self.pos_stage[slot];
                    self.live[slot] = true;
                } else {
                    commit_mixed_prefill(
                        self.pf[slot].as_mut().expect("staged token-batch cursor"),
                        take,
                    );
                }
            }
            Ok(finishing)
        }

        /// Hand a CPU prefill head's rows to `slot`: their KV is already in the
        /// slot's caches, so the next cursor starts at `rows` instead of 0.
        ///
        /// Call AFTER writing the rows and BEFORE the slot's next prefill, and
        /// only on a slot with no cursor in flight — a head is a fresh
        /// request's prefix, never a splice into a prefill already running.
        pub fn attach_head(&mut self, slot: usize, rows: u32) -> Result<()> {
            self.check_slot(slot)?;
            if self.pf[slot].is_some() || self.live[slot] {
                return Err(RuntimeError::Rejected(format!(
                    "slot {slot} is mid-prefill or live; a head attaches to a fresh slot"
                )));
            }
            if rows as usize >= self.max_ctx {
                return Err(RuntimeError::ContextLength(format!(
                    "head of {rows} rows against a compiled context of {}",
                    self.max_ctx
                )));
            }
            self.head_rows[slot] = rows;
            Ok(())
        }

        /// Rows completed by a request whose chunked prefill is still active.
        pub fn prefill_frontier(&self, slot: usize) -> Option<usize> {
            self.pf
                .get(slot)
                .and_then(Option::as_ref)
                .map(|cursor| cursor.frontier as usize)
        }

        /// Device-independent metadata for the next already-planned chunk.
        ///
        /// A fresh request has no cursor yet and returns `None`; isolated or mixed
        /// admission must first initialize its state and prefix plan.
        /// Subsequent chunks can participate in compatible-
        /// rung pack formation without duplicating either side effect.
        pub fn prefill_span(&self, slot: usize, max_rows: u32) -> Option<packet::dev::PrefillSpan> {
            let cur = self.pf.get(slot)?.as_ref()?;
            let step = *cur.steps.get(cur.next)?;
            if step.clen > max_rows {
                return None;
            }
            Some(packet::dev::PrefillSpan {
                row0: 0,
                n_rows: step.clen,
                slot: u32::try_from(slot).ok()?,
                flags: if step.c0 == 0 {
                    packet::dev::PREFILL_SPAN_RESET_STATE
                } else {
                    0
                },
                kv_row0: step.c0,
                kv_len: step.c0.checked_add(step.clen)?,
                state_slot: u32::try_from(slot).ok()?,
                program: u32::try_from(step.prog).ok()?,
            })
        }

        /// The next span iff it can complete without producing a token or crossing a prefix-cache
        /// snapshot boundary. Fresh requests remain ineligible until admission initializes
        /// their cursor and state.
        pub fn packable_prefill_span(
            &self,
            slot: usize,
            max_rows: u32,
        ) -> Option<packet::dev::PrefillSpan> {
            let cur = self.pf.get(slot)?.as_ref()?;
            let step = packable_prefill_step(cur, max_rows)?;
            let program = self.packed_prefill_program(step.clen, max_rows, true)?;
            let mut span = self.prefill_span(slot, max_rows)?;
            span.program = u32::try_from(program).ok()?;
            Some(span)
        }

        /// Whether a pack could form at all, i.e. whether two chunks of the configured size fit
        /// one compiled prefill rung.
        ///
        /// Seeding a cursor up front is pure cost when they cannot: the slot pays its state
        /// clear and chunk planning in the arrival tick instead of spread across ticks, and no
        /// pack ever forms to repay it. Measured on Gemma 4 31B at `PLOW_PF_CHUNK=8192`, whose
        /// chunk fills the widest rung: 1.3-2.8% of short-prompt throughput, TTFT up 2-12%.
        fn packed_prefill_reachable(&self, max_rows: u32) -> bool {
            let chunk = self.prefill_chunk_rows.min(max_rows);
            chunk != 0
                && self
                    .packed_prefill_program(chunk, max_rows, true)
                    .and_then(|prog| {
                        self.ranks
                            .rank0()
                            .prefill_rungs()
                            .find(|&(index, _)| index == prog)
                            .map(|(_, width)| width)
                    })
                    .is_some_and(|rung| rung >= chunk.saturating_mul(2))
        }

        fn packed_prefill_program(&self, rows: u32, cap: u32, widest: bool) -> Option<usize> {
            self.ranks
                .rank0()
                .prefill_rungs()
                .filter(|&(program, width)| {
                    width >= rows
                        && width <= cap
                        && match &self.ranks {
                            Ranks::One(e) => e.packed_prefill_prog_capable(program),
                            Ranks::Tp(g) => g.packed_prefill_prog_capable(program),
                        }
                })
                .min_by_key(|&(_, width)| if widest { u32::MAX - width } else { width })
                .map(|(program, _)| program)
        }

        /// Advance compatible, already-initialized prefill cursors in one packed dispatch.
        /// Final chunks remain isolated because the current model prefill head exposes one
        /// sampled token, not one result per span. Snapshot boundaries remain isolated too.
        pub fn advance_packed_prefill(&mut self, members: &[(usize, &[u32])]) -> Result<()> {
            if members.len() < 2 {
                return Err(RuntimeError::Rejected(
                    "packed prefill requires at least two cursors".into(),
                ));
            }

            let mut spans = Vec::with_capacity(members.len());
            let mut slices = Vec::with_capacity(members.len());
            let mut completed = Vec::with_capacity(members.len());
            let mut row0 = 0u32;
            for &(slot, prompt) in members {
                if spans
                    .iter()
                    .any(|span: &PrefillSpan| span.slot as usize == slot)
                {
                    return Err(RuntimeError::Rejected(format!(
                        "packed prefill slot {slot} appears more than once"
                    )));
                }
                let cur = self.pf.get(slot).and_then(Option::as_ref).ok_or_else(|| {
                    RuntimeError::Rejected(format!(
                        "packed prefill slot {slot} has no initialized cursor"
                    ))
                })?;
                if self.live.get(slot).copied() != Some(false) {
                    return Err(RuntimeError::Device(format!(
                        "packed prefill slot {slot} has both a live decode row and a prefill cursor"
                    )));
                }
                let step = *cur.steps.get(cur.next).ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "packed prefill slot {slot} cursor is past its plan"
                    ))
                })?;
                if cur.next + 1 == cur.steps.len() {
                    return Err(RuntimeError::Rejected(format!(
                        "packed prefill slot {slot} is on its final chunk; use isolated prefill to collect its sampled token"
                    )));
                }
                if cur.snap_after == Some(cur.next) {
                    return Err(RuntimeError::Rejected(format!(
                        "packed prefill slot {slot} is on a prefix snapshot boundary; use isolated prefill"
                    )));
                }
                let end = (step.c0 as usize)
                    .checked_add(step.clen as usize)
                    .filter(|&end| end <= prompt.len())
                    .ok_or_else(|| {
                        RuntimeError::Rejected(format!(
                            "packed prefill slot {slot} prompt does not cover [{}, {})",
                            step.c0,
                            step.c0.saturating_add(step.clen)
                        ))
                    })?;
                let mut span = self.prefill_span(slot, u32::MAX).ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "packed prefill slot {slot} lost its planned span"
                    ))
                })?;
                span.row0 = row0;
                row0 = row0.checked_add(span.n_rows).ok_or_else(|| {
                    RuntimeError::Rejected("packed prefill row count overflows u32".into())
                })?;
                spans.push(span);
                slices.push(&prompt[step.c0 as usize..end]);
                completed.push((slot, step));
            }

            let prog = self
                .packed_prefill_program(row0, u32::MAX, false)
                .ok_or_else(|| {
                    RuntimeError::Rejected(format!("no packed prefill rung covers {row0} rows"))
                })?;
            for span in &mut spans {
                span.program = prog as u32;
            }
            let rung = self.prefill_prog_t(prog).ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "packed prefill program {prog} is not a common prefill rung"
                ))
            })?;
            if row0 > rung {
                return Err(RuntimeError::Rejected(format!(
                    "packed prefill has {row0} dense rows but program {prog} is compiled for {rung}"
                )));
            }
            let mut parked = vec![1; rung as usize];
            parked[..row0 as usize].fill(0);
            match &mut self.ranks {
                Ranks::One(e) => e.prefill_packed_chunk(&spans, &slices, &parked)?,
                Ranks::Tp(g) => g.prefill_packed_chunk(&spans, &slices, &parked)?,
            }
            commit_packed_prefill(&mut self.pf, &completed);
            if let Some(diagnostics) = self.diagnostics.as_mut() {
                for &(slot, step) in &completed {
                    diagnostics.push_prefill(PrefillSelection {
                        slot,
                        row_start: step.c0,
                        rows: step.clen,
                        bucket: rung,
                    });
                }
            }
            Ok(())
        }

        pub fn prefill_turn(&self) -> usize {
            self.prefill_turn
        }

        pub fn advance_prefill_turn(&mut self, slot: usize) {
            self.prefill_turn = (slot + 1) % self.batch.max(1);
        }

        fn packet_trace_paths(path: &Path, n_gpu: usize, all: bool) -> Vec<std::path::PathBuf> {
            (0..if all { n_gpu } else { 1 })
                .map(|rank| {
                    if all {
                        let mut out = path.as_os_str().to_owned();
                        out.push(format!(".rk{rank}"));
                        std::path::PathBuf::from(out)
                    } else {
                        path.to_owned()
                    }
                })
                .collect()
        }

        /// Write the last completed program trace. `PLOW_TRACE_ALLRANKS=1` is a
        /// diagnostic-only TP mode that appends `.rkN` to `path`.
        pub fn write_packet_trace(&self, path: &Path) -> Result<()> {
            match &self.ranks {
                Ranks::One(e) => e.trace_write(path),
                Ranks::Tp(g) => {
                    let all = crate::config::RuntimeConfig::get().amd.trace_allranks;
                    for (rank, out) in Self::packet_trace_paths(path, g.n_gpu(), all)
                        .iter()
                        .enumerate()
                    {
                        g.rank(rank).trace_write(&out)?;
                    }
                    Ok(())
                }
            }
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

        /// Largest safe deferred-read quantum for this feed set.
        pub fn multistep_quantum(&self, feeds: &[(usize, u32)], requested: usize) -> Option<usize> {
            let available = match &self.ranks {
                Ranks::One(e) => e.deferred_token_capture_available(),
                Ranks::Tp(g) => g.deferred_token_capture_available(),
            };
            if !available
                || feeds.iter().enumerate().any(|(i, &(slot, _))| {
                    self.live.get(slot).copied() != Some(true)
                        || self.pf.get(slot).is_none_or(Option::is_some)
                        || feeds[..i].iter().any(|&(previous, _)| previous == slot)
                })
                || crate::config::RuntimeConfig::get().amd.ctr_snap.is_some()
                || crate::config::RuntimeConfig::get().amd.tens_snap.is_some()
            {
                return None;
            }
            let quantum = crate::sched::multistep::decode_quantum(
                feeds.iter().map(|&(slot, _)| slot),
                &self.pos,
                self.max_ctx,
                requested,
                crate::exec::amd::DEFERRED_TOKEN_MAX_STEPS,
            )
            .ok()?;
            (quantum >= 2).then_some(quantum)
        }

        /// Run several greedy decode tokens while retaining every per-token
        /// drain/counter audit. Tokens are captured device-side and read once.
        pub fn multi_step(
            &mut self,
            feeds: &[(usize, u32)],
            quantum: usize,
            out: &mut Vec<u32>,
        ) -> Result<usize> {
            if self.multistep_quantum(feeds, quantum) != Some(quantum) {
                return Err(RuntimeError::Rejected(
                    "AMD deferred-read multi-step is unavailable for this quantum".into(),
                ));
            }
            for &(slot, id) in feeds {
                self.check_slot(slot)?;
                self.next_id[slot] = id;
            }
            let mut advance = std::mem::take(&mut self.advance_stage);
            advance.clear();
            advance.extend(feeds.iter().map(|&(slot, _)| slot));
            let rows = (0..self.batch)
                .filter(|&slot| self.live[slot] || self.pf[slot].is_some())
                .map(|slot| slot + 1)
                .max()
                .unwrap_or(1);
            if let Some(&slot) = advance.iter().find(|&&slot| slot >= rows) {
                self.advance_stage = advance;
                return Err(RuntimeError::Device(format!(
                    "decode ladder: slot {slot} is being advanced but the rung covers only {rows} rows"
                )));
            }

            let result = (|| -> Result<()> {
                let dp = self.ranks.rank0().decode_prog_for(rows);
                let width = self.ranks.rank0().prog_t(dp);
                if width != self.last_rung {
                    tracing::info!(rung = width, occupied = rows, "decode ladder rung");
                    self.last_rung = width;
                }
                stage_parked(&mut self.parked_stage, &advance);
                match &mut self.ranks {
                    Ranks::One(e) => {
                        e.upload_parked(&self.parked_stage)?;
                        e.seed_ids(&self.next_id)?;
                    }
                    Ranks::Tp(g) => {
                        g.upload_parked(&self.parked_stage)?;
                        g.seed_ids(&self.next_id)?;
                    }
                }

                let started = self.diagnostics.is_some().then(std::time::Instant::now);
                let dispatch = (|| {
                    for step in 0..quantum {
                        for slot in 0..self.batch {
                            let (pos, kvlen) = match (&self.pf[slot], self.live[slot]) {
                                (Some(cursor), _) => (cursor.frontier, cursor.frontier + 1),
                                (None, true) => (self.pos[slot], self.pos[slot] + 1),
                                (None, false) => (0, 1),
                            };
                            self.pos_stage[slot] = pos;
                            self.kvlen_stage[slot] = kvlen;
                        }
                        match &mut self.ranks {
                            Ranks::One(e) => e.decode_batched_deferred_at(
                                &self.pos_stage,
                                &self.kvlen_stage,
                                dp,
                                step,
                                quantum,
                            )?,
                            Ranks::Tp(g) => {
                                g.submit_decode_batched_at(&self.pos_stage, &self.kvlen_stage, dp)?;
                                g.complete_decode_batched_deferred(self.batch, step, quantum)?;
                            }
                        }
                        for &slot in &advance {
                            self.pos[slot] += 1;
                        }
                    }
                    match &mut self.ranks {
                        Ranks::One(e) => e.read_token_capture(width as usize, quantum, out),
                        Ranks::Tp(g) => g.read_deferred_tokens(width as usize, quantum, out),
                    }
                })();
                if let Some(started) = started {
                    self.diagnostics
                        .as_mut()
                        .expect("diagnostic timer requires diagnostics")
                        .push_decode(DecodeSelection {
                            occupied_rows: rows,
                            bucket: width,
                            elapsed_ns: u64::try_from(started.elapsed().as_nanos())
                                .unwrap_or(u64::MAX),
                            steps: quantum,
                        });
                }
                dispatch
            })();
            self.advance_stage = advance;
            result?;
            Ok(quantum)
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
            let mut advance = std::mem::take(&mut self.advance_stage);
            advance.clear();
            advance.extend(feeds.iter().map(|&(s, _)| s));
            let result = self.dispatch_all(&advance);
            self.advance_stage = advance;
            let out = result?;
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

        fn capture_decode_snapshots(&mut self, program: usize, rung: u32) -> Result<()> {
            if let Some(dir) = self.counter_snapshot_dir.as_ref() {
                let snapshot = self.ranks.ctr_snapshot(program)?;
                let bytes = snapshot
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                let path = dir.join(format!(
                    "tick_{:05}_r{rung}.bin",
                    self.counter_snapshot_tick
                ));
                write_snapshot(path, &bytes)?;
                self.counter_snapshot_tick += 1;
            }
            if let Some(snapshot) = self.tensor_snapshot.as_ref() {
                let tensors = self.ranks.data_snapshot(snapshot.slot, &snapshot.tensors)?;
                for (name, bytes) in tensors {
                    let path = snapshot.dir.join(format!(
                        "t{:05}_r{rung}_{}.bin",
                        self.tensor_snapshot_tick,
                        snapshot_file_component(&name)
                    ));
                    write_snapshot(path, &bytes)?;
                }
                self.tensor_snapshot_tick += 1;
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
        ///   decode-only prompt walk runs). Its KDA recurrence is parked. Its
        ///   append-only KV work rewrites row `pos[s]` — the row it is about to
        ///   write for real — and its host position does not move.
        ///
        /// The cost is wasted work, and it is why a blob compiled at a large
        /// `PLOW_DECODE_BATCH` is slower at concurrency 1 than one compiled at 1.
        fn dispatch_all(&mut self, advance: &[usize]) -> Result<Vec<u32>> {
            if self.batch == 1 {
                let pos = self.pos[0];
                let measure_dispatch = self.diagnostics.is_some();
                use crate::obs::dstep;
                let (token, program, rung, elapsed_ns) = match &mut self.ranks {
                    Ranks::One(e) => {
                        let program = e.decode_prog();
                        let rung = e.prog_t(program);
                        dstep::timed(&dstep::SEED, || e.seed_ids(&self.next_id))?;
                        let started = measure_dispatch.then(std::time::Instant::now);
                        let token = e.decode_step(pos, pos + 1)?;
                        (
                            token,
                            program,
                            rung,
                            started.map(|started| {
                                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
                            }),
                        )
                    }
                    // The split pair, not `decode_step`, because this is the
                    // server and the split exists for it. NOTHING sits between
                    // the two calls, and that is measured rather than pending:
                    // §DSTEP puts the whole host phase at 1.4% of the token and
                    // the only part of it that is safe to move at all at 0.02%.
                    // The argument, with the numbers, is in `exec::amd_tp`'s
                    // module doc — read it before putting work here.
                    Ranks::Tp(g) => {
                        let program = g.rank(0).decode_prog();
                        let rung = g.rank(0).prog_t(program);
                        dstep::timed(&dstep::SEED, || g.seed_ids(&self.next_id))?;
                        let started = measure_dispatch.then(std::time::Instant::now);
                        g.submit_decode(pos, pos + 1)?;
                        let ids = g.complete_decode()?;
                        (
                            dstep::timed(&dstep::AGREE, || AmdTpGroup::agree(&ids))?,
                            program,
                            rung,
                            started.map(|started| {
                                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
                            }),
                        )
                    }
                };
                if let Some(elapsed_ns) = elapsed_ns {
                    self.diagnostics
                        .as_mut()
                        .expect("diagnostic timer requires diagnostics")
                        .push_decode(DecodeSelection {
                            occupied_rows: 1,
                            bucket: rung,
                            elapsed_ns,
                            steps: 1,
                        });
                }
                self.capture_decode_snapshots(program, rung)?;
                for &s in advance {
                    self.pos[s] += 1;
                }
                return Ok(vec![token]);
            }
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
                self.pos_stage[s] = pp;
                self.kvlen_stage[s] = kk;
            }
            // PER-ROW PARKED MASK. The device executes every covered row, but only `advance`
            // owns a logical token this dispatch. Parking merely idle rows is insufficient:
            // one-token admission and decode-only prompt walking advance one slot while other
            // live slots wait. Advancing their recurrence without their host position corrupts
            // the next real token.
            //
            // On a blob without `in.parked` this is a no-op, so a non-batched packet is unchanged.
            stage_parked(&mut self.parked_stage, advance);
            // THE DECODE BATCH LADDER (`PLOW_DECODE_BATCH_LADDER` at emit). Pick the
            // NARROWEST decode program that still advances every occupied slot, and pay only
            // that program's wasted rows instead of `batch`'s.
            //
            // THE ARGUMENT IS OVER SLOTS, NOT OVER LIVE COUNT, and the distinction is the
            // whole of the correctness case. A rung of width `w` advances rows `[0, w)` only;
            // a sequence sitting in slot 5 is simply not stepped by a width-4 rung, and since
            // the host advances `pos` regardless it would decode from a KV row it never wrote.
            // So the rung is chosen from the HIGHEST OCCUPIED SLOT INDEX, and slots are handed
            // out lowest-first by the mux, which is what makes the two usually agree.
            //
            // NO COMPACTION on release. Freeing slot 0 while slot 5 is live leaves the rung
            // where it was: moving a sequence down would mean copying its whole KV block
            // (GiB), which costs far more than the rows it would save. The ladder narrows again
            // when the high slots drain — which is the common shape, since a slot is reused by
            // the next admission before higher ones are.
            //
            // A slot MID-PREFILL counts as occupied: its rows are real KV and the parked-row
            // logic above already points it at its own frontier.
            let rows = (0..self.batch)
                .filter(|&s| self.live[s] || self.pf[s].is_some())
                .map(|s| s + 1)
                .max()
                .unwrap_or(1);
            // `advance` is drawn from the live slots, so `rows` covers it by construction —
            // but the failure if it ever did not is a host position stepped past a KV row the
            // device never wrote, which is fluent wrong text and no fault. Sixteen comparisons
            // against an 11 ms token is the right price for closing that off in RELEASE, not
            // just under debug_assert.
            if let Some(&s) = advance.iter().find(|&&s| s >= rows) {
                return Err(RuntimeError::Device(format!(
                    "decode ladder: slot {s} is being advanced but the rung covers only \
                     {rows} rows — the slot table and the live set disagree"
                )));
            }
            let measure_dispatch = self.diagnostics.is_some();
            let (out, program, rung, elapsed_ns) = match &mut self.ranks {
                Ranks::One(e) => {
                    let dp = e.decode_prog_for(rows);
                    let w = e.prog_t(dp);
                    if w != self.last_rung {
                        tracing::info!(rung = w, occupied = rows, "decode ladder rung");
                        self.last_rung = w;
                    }
                    e.upload_parked(&self.parked_stage)?;
                    e.seed_ids(&self.next_id)?;
                    let started = measure_dispatch.then(std::time::Instant::now);
                    let out = e.decode_step_batched_at(&self.pos_stage, &self.kvlen_stage, dp)?;
                    (
                        out,
                        dp,
                        w,
                        started.map(|started| {
                            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
                        }),
                    )
                }
                Ranks::Tp(g) => {
                    let dp = g.rank(0).decode_prog_for(rows);
                    let w = g.rank(0).prog_t(dp);
                    if w != self.last_rung {
                        tracing::info!(rung = w, occupied = rows, "decode ladder rung");
                        self.last_rung = w;
                    }
                    tracing::debug!(
                        rung = w,
                        pos = ?&self.pos_stage[..(w as usize).min(self.pos_stage.len())],
                        kvlen = ?&self.kvlen_stage[..(w as usize).min(self.kvlen_stage.len())],
                        parked = ?&self.parked_stage[..(w as usize).min(self.parked_stage.len())],
                        ids = ?&self.next_id[..(w as usize).min(self.next_id.len())],
                        "dstep stage"
                    );
                    g.upload_parked(&self.parked_stage)?;
                    g.seed_ids(&self.next_id)?;
                    let started = measure_dispatch.then(std::time::Instant::now);
                    let out = g.decode_step_batched_at(&self.pos_stage, &self.kvlen_stage, dp)?;
                    (
                        out,
                        dp,
                        w,
                        started.map(|started| {
                            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
                        }),
                    )
                }
            };
            if let Some(elapsed_ns) = elapsed_ns {
                self.diagnostics
                    .as_mut()
                    .expect("diagnostic timer requires diagnostics")
                    .push_decode(DecodeSelection {
                        occupied_rows: rows,
                        bucket: rung,
                        elapsed_ns,
                        steps: 1,
                    });
            }
            self.capture_decode_snapshots(program, rung)?;
            for &s in advance {
                self.pos[s] += 1;
            }
            Ok(out)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{
            commit_mixed_prefill, commit_packed_prefill, invalidate_prefix_metadata,
            mixed_cursor_rows, mixed_prefill_continuation_fits, mixed_prefill_padding_fits,
            packable_prefill_step, parse_snapshot_tensors, snapshot_file_component,
            split_pending_prefill, split_terminal_prefill, stage_parked, terminal_prefill_cursor,
            AmdServe, PfCursor, DEFAULT_SNAPSHOT_TENSORS, MAX_SNAPSHOT_TENSORS,
        };
        use crate::exec::amd::ChunkStep;

        #[test]
        fn terminal_prefill_split_preserves_frontier_and_original_bucket() {
            let mut cur = PfCursor {
                steps: vec![ChunkStep {
                    prog: 7,
                    c0: 4096,
                    clen: 128,
                }],
                next: 0,
                frontier: 4096,
                snap_after: None,
                resume: 0,
                arm: 0,
            };
            split_terminal_prefill(&mut cur);
            split_terminal_prefill(&mut cur);
            assert_eq!(cur.steps.len(), 2);
            assert_eq!(cur.frontier, 4096);
            assert_eq!(
                cur.steps[0],
                ChunkStep {
                    prog: 7,
                    c0: 4096,
                    clen: 127
                }
            );
            assert_eq!(
                cur.steps[1],
                ChunkStep {
                    prog: 7,
                    c0: 4223,
                    clen: 1
                }
            );
            assert!(!terminal_prefill_cursor(&cur, 4224));
            commit_mixed_prefill(&mut cur, 127);
            assert!(terminal_prefill_cursor(&cur, 4224));
            assert!(!terminal_prefill_cursor(&cur, 4223));
            cur.snap_after = Some(1);
            assert!(!terminal_prefill_cursor(&cur, 4224));
        }

        #[test]
        fn terminal_prefill_split_preserves_snapshot_boundaries() {
            let mut cur = PfCursor {
                steps: vec![ChunkStep {
                    prog: 3,
                    c0: 0,
                    clen: 128,
                }],
                next: 0,
                frontier: 0,
                snap_after: Some(0),
                resume: 0,
                arm: 128,
            };
            split_terminal_prefill(&mut cur);
            assert_eq!(cur.steps.len(), 1);
            assert_eq!(cur.steps[0].clen, 128);
            assert_eq!(cur.snap_after, Some(0));
            assert!(!terminal_prefill_cursor(&cur, 128));
        }

        #[test]
        fn packet_trace_paths_cover_single_and_all_tp_ranks() {
            assert_eq!(
                AmdServe::packet_trace_paths(std::path::Path::new("trace"), 8, false),
                [std::path::PathBuf::from("trace")]
            );
            assert_eq!(
                AmdServe::packet_trace_paths(std::path::Path::new("trace"), 3, true),
                [
                    std::path::PathBuf::from("trace.rk0"),
                    std::path::PathBuf::from("trace.rk1"),
                    std::path::PathBuf::from("trace.rk2"),
                ]
            );
        }

        #[test]
        fn snapshot_tensor_list_is_bounded_and_uses_compatibility_default() {
            assert_eq!(
                parse_snapshot_tensors(None).unwrap().join(","),
                DEFAULT_SNAPSHOT_TENSORS
            );
            assert_eq!(
                parse_snapshot_tensors(Some(" act.logits, act.x ")).unwrap(),
                ["act.logits", "act.x"]
            );
            assert!(parse_snapshot_tensors(Some("act.x,,act.logits")).is_err());
            assert!(parse_snapshot_tensors(Some("act.x,act.x")).is_err());
            assert!(parse_snapshot_tensors(Some("act/a,act:a")).is_err());
            let too_many = (0..=MAX_SNAPSHOT_TENSORS)
                .map(|i| format!("act.{i}"))
                .collect::<Vec<_>>()
                .join(",");
            assert!(parse_snapshot_tensors(Some(&too_many)).is_err());
        }

        #[test]
        fn snapshot_tensor_names_cannot_create_subdirectories() {
            assert_eq!(snapshot_file_component("act/a:b"), "act_a_b");
        }

        #[test]
        fn snapshot_write_errors_are_not_swallowed() {
            let dir = std::env::temp_dir()
                .join(format!("plow-snapshot-write-test-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            assert!(super::write_snapshot(dir.clone(), b"snapshot").is_err());
            std::fs::remove_dir(dir).unwrap();
        }

        #[test]
        fn snapshot_writes_never_overwrite_an_existing_tick() {
            let path = std::env::temp_dir().join(format!(
                "plow-snapshot-create-new-test-{}",
                std::process::id()
            ));
            super::write_snapshot(path.clone(), b"first").unwrap();
            let err = super::write_snapshot(path.clone(), b"second").unwrap_err();
            match err {
                crate::RuntimeError::Io { source, .. } => {
                    assert_eq!(source.kind(), std::io::ErrorKind::AlreadyExists)
                }
                other => panic!("unexpected snapshot error: {other}"),
            }
            assert_eq!(std::fs::read(&path).unwrap(), b"first");
            std::fs::remove_file(path).unwrap();
        }

        #[test]
        fn shrinking_a_pending_prefill_step_preserves_the_snapshot_boundary() {
            let mut cur = PfCursor {
                steps: vec![ChunkStep {
                    prog: 3,
                    c0: 0,
                    clen: 8,
                }],
                next: 0,
                frontier: 0,
                snap_after: Some(0),
                resume: 0,
                arm: 8,
            };
            split_pending_prefill(
                &mut cur,
                vec![
                    ChunkStep {
                        prog: 1,
                        c0: 0,
                        clen: 4,
                    },
                    ChunkStep {
                        prog: 1,
                        c0: 4,
                        clen: 4,
                    },
                ],
            );
            assert_eq!(cur.steps.len(), 2);
            assert_eq!(cur.snap_after, Some(1));
        }

        #[test]
        fn packed_cursor_commit_advances_all_selected_rows_together() {
            let step0 = ChunkStep {
                prog: 2,
                c0: 0,
                clen: 3,
            };
            let step1 = ChunkStep {
                prog: 2,
                c0: 4,
                clen: 2,
            };
            let cursor = |step| {
                Some(PfCursor {
                    steps: vec![step, ChunkStep { c0: 8, ..step }],
                    next: 0,
                    frontier: step.c0,
                    snap_after: None,
                    resume: 0,
                    arm: 0,
                })
            };
            let mut cursors = vec![cursor(step0), None, cursor(step1)];
            commit_packed_prefill(&mut cursors, &[(0, step0), (2, step1)]);
            assert_eq!(cursors[0].as_ref().unwrap().next, 1);
            assert_eq!(cursors[0].as_ref().unwrap().frontier, 3);
            assert_eq!(cursors[2].as_ref().unwrap().next, 1);
            assert_eq!(cursors[2].as_ref().unwrap().frontier, 6);
            assert!(cursors[1].is_none());
        }

        #[test]
        fn packed_prefill_eligibility_excludes_final_snapshot_and_oversized_steps() {
            let step = ChunkStep {
                prog: 2,
                c0: 4,
                clen: 3,
            };
            let cursor = |next, snap_after| PfCursor {
                steps: vec![step, ChunkStep { c0: 7, ..step }],
                next,
                frontier: 4,
                snap_after,
                resume: 0,
                arm: 0,
            };
            assert_eq!(packable_prefill_step(&cursor(0, None), 3), Some(step));
            assert_eq!(packable_prefill_step(&cursor(0, None), 2), None);
            assert_eq!(packable_prefill_step(&cursor(0, Some(0)), 3), None);
            assert_eq!(packable_prefill_step(&cursor(1, None), 3), None);
        }

        #[test]
        fn mixed_prefill_padding_near_context_end_requires_isolated_fallback() {
            assert!(mixed_prefill_padding_fits(0, 1023, 8192));
            assert!(mixed_prefill_padding_fits(7900, 127, 8192));
            assert!(!mixed_prefill_padding_fits(7900, 1023, 8192));
            assert!(!mixed_prefill_padding_fits(8192, 1, 8192));
            assert!(!mixed_prefill_padding_fits(u32::MAX, 2, usize::MAX));
        }

        #[test]
        fn mixed_prefill_retained_bucket_near_context_end_requires_isolated_fallback() {
            assert!(mixed_prefill_padding_fits(7680, 127, 8192));
            assert!(!mixed_prefill_continuation_fits(7680, 127, 511, 512, 8192));
            assert!(mixed_prefill_continuation_fits(7553, 127, 511, 512, 8192));
            assert!(mixed_prefill_continuation_fits(0, 1023, 1024, 1024, 8192));
            assert!(!mixed_prefill_continuation_fits(0, 1023, 1024, 1024, 1024));
            assert!(!mixed_prefill_continuation_fits(
                u32::MAX - 1,
                1,
                2,
                2,
                usize::MAX
            ));
        }

        #[test]
        fn mixed_cursor_continuations_preserve_final_sampling_and_snapshot_boundaries() {
            let mut cur = PfCursor {
                steps: vec![ChunkStep {
                    prog: 2,
                    c0: 2048,
                    clen: 512,
                }],
                next: 0,
                frontier: 2048,
                snap_after: None,
                resume: 0,
                arm: 0,
            };
            assert_eq!(mixed_cursor_rows(&cur, 2560, 1024), 511);
            commit_mixed_prefill(&mut cur, 127);
            assert_eq!(cur.frontier, 2175);
            assert_eq!(
                cur.steps[0],
                ChunkStep {
                    prog: 2,
                    c0: 2175,
                    clen: 385
                }
            );
            assert_eq!(mixed_cursor_rows(&cur, 2560, 1024), 384);
            commit_mixed_prefill(&mut cur, 384);
            assert_eq!(cur.next, 0);
            assert_eq!(cur.steps[0].clen, 1);
            assert_eq!(mixed_cursor_rows(&cur, 2560, 1024), 0);
            assert_eq!(mixed_cursor_rows(&cur, 3000, 128), 1);
            cur.snap_after = Some(0);
            assert_eq!(mixed_cursor_rows(&cur, 3000, 128), 0);
            cur.snap_after = None;
            commit_mixed_prefill(&mut cur, 1);
            assert_eq!(cur.next, 1);
            assert_eq!(cur.frontier, 2560);
        }

        #[test]
        fn parked_rows_follow_the_advance_set_not_liveness() {
            let mut parked = vec![0; 4];
            stage_parked(&mut parked, &[2]);
            assert_eq!(parked, [1, 1, 0, 1]);

            stage_parked(&mut parked, &[0, 1, 2, 3]);
            assert_eq!(parked, [0, 0, 0, 0]);

            stage_parked(&mut parked, &[]);
            assert_eq!(parked, [1, 1, 1, 1]);
        }

        #[test]
        fn prefix_miss_is_ineligible_even_if_prefill_is_cancelled() {
            let mut cached = vec![vec![1, 2, 3], vec![4, 5, 6]];
            let mut snap_at = vec![2, 3];
            invalidate_prefix_metadata(true, &mut cached, &mut snap_at, 1);
            assert!(cached[1].is_empty());
            assert_eq!(snap_at, [2, 0]);
        }

        #[test]
        fn prefix_invalidation_is_inert_when_the_cache_is_disabled() {
            let mut cached = vec![vec![1, 2, 3]];
            let mut snap_at = vec![2];
            invalidate_prefix_metadata(false, &mut cached, &mut snap_at, 0);
            assert_eq!(cached, [vec![1, 2, 3]]);
            assert_eq!(snap_at, [2]);
        }
    }
}

#[cfg(feature = "hsa")]
impl SeqEngine for AmdServe {
    fn stop_ids(&self) -> &Arc<Vec<u32>> {
        AmdServe::stop_ids(self)
    }
    fn batch(&self) -> usize {
        AmdServe::batch(self)
    }
    fn release(&mut self, slot: usize) {
        AmdServe::release(self, slot)
    }
    fn prefill_turn(&self) -> usize {
        AmdServe::prefill_turn(self)
    }
    fn advance_prefill_turn(&mut self, slot: usize) {
        AmdServe::advance_prefill_turn(self, slot)
    }
    fn prefill_prog_t(&self, prog: usize) -> Option<u32> {
        AmdServe::prefill_prog_t(self, prog)
    }
    fn packed_prefill_span_limit(&self, prog: usize) -> u32 {
        AmdServe::packed_prefill_span_limit(self, prog)
    }
    fn packable_prefill_span(
        &self,
        slot: usize,
        max_rows: u32,
    ) -> Option<packet::dev::PrefillSpan> {
        AmdServe::packable_prefill_span(self, slot, max_rows)
    }
    fn advance_packed_prefill(&mut self, members: &[(usize, &[u32])]) -> crate::Result<()> {
        AmdServe::advance_packed_prefill(self, members)
    }
    fn prefill_frontier(&self, slot: usize) -> Option<usize> {
        AmdServe::prefill_frontier(self, slot)
    }
    fn prefill_chunked_at_most(
        &mut self,
        slot: usize,
        prompt: &[u32],
        tick_max_bucket: u32,
    ) -> crate::Result<Option<u32>> {
        AmdServe::prefill_chunked_at_most(self, slot, prompt, tick_max_bucket)
    }
    fn multistep_quantum(&self, feeds: &[(usize, u32)], requested: usize) -> Option<usize> {
        AmdServe::multistep_quantum(self, feeds, requested)
    }
    fn multi_step(
        &mut self,
        feeds: &[(usize, u32)],
        quantum: usize,
        out: &mut Vec<u32>,
    ) -> crate::Result<usize> {
        AmdServe::multi_step(self, feeds, quantum, out)
    }
    fn mixed_step_rows(&self, decode_rows: usize, prefill_rows: usize) -> Option<u32> {
        AmdServe::mixed_step_rows(self, decode_rows, prefill_rows)
    }
    fn mixed_prefill_rows(&self, slot: usize, prompt: &[u32], max_rows: u32) -> u32 {
        AmdServe::mixed_prefill_rows(self, slot, prompt, max_rows)
    }
    fn mixed_prefill_fits(&self, slot: usize, prefill_capacity: u32) -> bool {
        AmdServe::mixed_prefill_fits(self, slot, prefill_capacity)
    }
    fn prepare_packed_prefill_slot(
        &mut self,
        slot: usize,
        prompt: &[u32],
        max_rows: u32,
    ) -> crate::Result<()> {
        AmdServe::prepare_packed_prefill_slot(self, slot, prompt, max_rows)
    }
    fn terminal_prefill_ready(&self, slot: usize, prompt: &[u32]) -> bool {
        AmdServe::terminal_prefill_ready(self, slot, prompt)
    }
    fn finish_prefill_batch(
        &mut self,
        feeds: &[(usize, u32)],
        members: &[(usize, &[u32])],
    ) -> crate::Result<Vec<(usize, u32)>> {
        AmdServe::finish_prefill_batch(self, feeds, members)
    }
    fn mixed_step(
        &mut self,
        rows: u32,
        feeds: &[(usize, u32)],
        members: &[(usize, &[u32], u32)],
        output: &mut [u32],
    ) -> crate::Result<()> {
        AmdServe::mixed_step(self, rows, feeds, members, output)
    }
    fn token_batch_rows(&self, leading_rows: usize, prefill_rows: usize) -> Option<u32> {
        AmdServe::token_batch_rows(self, leading_rows, prefill_rows)
    }
    fn token_batch_prefill_rows(&self, slot: usize, prompt: &[u32], max_rows: u32) -> u32 {
        AmdServe::token_batch_prefill_rows(self, slot, prompt, max_rows)
    }
    fn token_batch_prefill_fits(&self, slot: usize, prefill_capacity: u32) -> bool {
        AmdServe::token_batch_prefill_fits(self, slot, prefill_capacity)
    }
    fn token_batch_step(
        &mut self,
        rows: u32,
        feeds: &[(usize, u32)],
        members: &[(usize, &[u32], u32)],
        output: &mut Vec<u32>,
    ) -> crate::Result<Vec<usize>> {
        AmdServe::token_batch_step(self, rows, feeds, members, output)
    }
    fn step_batch(&mut self, feeds: &[(usize, u32)]) -> crate::Result<Vec<(usize, u32)>> {
        AmdServe::step_batch(self, feeds)
    }
}
