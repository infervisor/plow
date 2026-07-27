//! The sm_120 persistent-interpreter engine (feature `cuda`).
//!
//! Rust port of the driver sequence `runtime/tests/gemma4_sm120_chat.cu`
//! proved out (that harness is HF-token-verified): parse the `PLOWDEV` blob,
//! upload the checkpoint weights and the decode program's tables, then run
//! **one cooperative launch per token step** — kv-row patch + three scalars +
//! counter zero, launch, sync, read back the device `ARGMAX_FIN` token.
//!
//! Decode-only, batch 1: the prompt is consumed BY THE DECODE PROGRAM one
//! token at a time. With causal attention that builds the same KV cache and
//! the same logits as a batched prefill — prefill is a throughput
//! optimization over exactly this loop, not a different computation (see the
//! harness header). Prefill-in-serve is the next task, deliberately not here.
//!
//! The engine always populates BOTH the static per-block stream and the
//! blob's op-major `GQ01` tables in the kernarg, so the same code drives a
//! `PLOW_NV_SCHED=0` or `=1` cubin — each build reads only its own tables.

use std::path::Path;
use std::sync::Arc;

use packet::dev::{DevInst64, DevOp, DevProgram, CTR_STRIDE, TENSOR_NONE16};

use crate::env_flag;

/// PX-1 packing measurement (RTX-12 baseline). `PLOW_PF_PACKLOG=1` emits one
/// compact stderr line per batched-prefill launch: request count R, total
/// packed rows, covering bucket size, and the per-request chunk-row list. The
/// flag is read once and cached, so when unset the hot path pays only a relaxed
/// atomic load. Post-processed into R-per-launch histograms by the bench.
env_flag!(fn pf_packlog_on, "PLOW_PF_PACKLOG");

/// `PLOW_PF_COVER=1` restores the covering bucket-pick policy (see
/// [`GpuEngine::pick_prefill_bucket`]). Read once — the per-chunk hot path
/// must not hit the environment.
env_flag!(fn pf_cover_on, "PLOW_PF_COVER");

/// Fixed cost of ONE prefill launch, expressed in padded-row equivalents —
/// the currency [`GpuEngine::pick_prefill_bucket`] minimizes. A launch is not
/// free: it re-streams every layer's weights from HBM and pays grid/counter
/// setup, so its cost is independent of how many real rows it carries.
///
/// Measured on sm_120 / gemma-4-12B by regressing TTFT over a prompt-length
/// sweep that straddles the 8192 rung (`ttft_ms = 0.112·rows + 60.1·chunks`,
/// fit within ±2.8%): **60.1 ms ≈ 537 rows**. The default rounds that to 512.
/// `PLOW_PF_CHUNK_COST=0` recovers the old pure-minimum-padding behaviour.
/// Read once — the per-chunk hot path must not hit the environment.
fn pf_chunk_cost_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("PLOW_PF_CHUNK_COST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
    })
}

use crate::asset::devblob::DevBlob;
use crate::device::cuda::{CudaBackend, CudaEvent, CudaStream, KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem, Module};
use crate::{Result, RuntimeError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InterpreterProfile {
    tag: &'static str,
    decode_file: &'static str,
    prefill_file: &'static str,
    decode_symbol: &'static str,
    prefill_symbol: &'static str,
    embedded_decode: &'static str,
}

fn interpreter_profile(cc: (u32, u32)) -> Option<InterpreterProfile> {
    match cc {
        (9, 0) => Some(InterpreterProfile {
            tag: "sm90a",
            decode_file: "interp_sm90a.cubin",
            prefill_file: "interp_sm90a_pf.cubin",
            decode_symbol: "_Z12interp_sm90a11PlowProgram",
            prefill_symbol: "_Z15interp_sm90a_pf11PlowProgram",
            embedded_decode: "interp_sm90a",
        }),
        (12, 0) => Some(InterpreterProfile {
            tag: "sm120",
            decode_file: "interp_sm120.cubin",
            prefill_file: "interp_sm120_pf.cubin",
            decode_symbol: "_Z12interp_sm12011PlowProgram",
            prefill_symbol: "_Z15interp_sm120_pf11PlowProgram",
            embedded_decode: "interp_sm120",
        }),
        _ => None,
    }
}

/// One block per executor, 8 worker warps — `dev_isa.h` workgroup geometry.
const BLOCK: u32 = 256;
/// Weight-upload staging chunk (pinned), as in the harness.
const STAGE: usize = 64 << 20;
/// Prefill dynamic-smem default: the union arena is the hd=256 flash tile,
/// 21312 floats = 83.25 KiB (`interp_sm120.cu` `PLOW_NV_PRE_A`, grown by the
/// T4 mma P·V Ps/m/l/corr staging). `PLOW_NV_SMEM_PF` overrides. Opt-in past
/// 48 KiB, so the launch always sets the attribute. Under-provisioning fails
/// the launch loudly (never silent), so a stale value here cannot corrupt.
const SMEM_PF: u32 = 21312 * 4;

/// VMM prefix-sharing state (`PLOW_VMM_PREFIX=1`, plans/rtx-09): the pool
/// backing every FULL layer's `kv.{l}.k/v` tensor with per-sequence VA
/// windows, plus the sliding-ring metadata the boundary snapshots need
/// (sliding layers stay on cudaMalloc; their last `window` rows are
/// D2D-copied to/from a snapshot buffer at publish/attach — plan §8).
struct VmmServe {
    kv: crate::memory::vmm::VmmKv,
    /// Per sliding layer: (devp index of `kv.{l}.k`, of `kv.{l}.v`,
    /// per-slot byte stride). K and V share one stride.
    slide: Vec<(usize, usize, u64)>,
    /// fp8-KV sliding layers: (devp index of `kv.{l}.k_scale`, of
    /// `kv.{l}.v_scale`) per sliding layer, aligned with `slide`. Empty when
    /// the rings are bf16. Scale rows are f32, one per KV row, same ring.
    slide_scale: Vec<(usize, usize)>,
    /// fp8-KV full layers: (devp index of `kv.{l}.k_scale`, of
    /// `kv.{l}.v_scale`) per full layer, aligned with `geometry().full_layers`.
    /// Empty when full-layer KV is bf16/fp16. Full-layer KV rides the VMM
    /// pool but its scales live in flat cudaMalloc tensors that slot reuse
    /// overwrites — so the whole scale PREFIX `[0..p_a)` rides the boundary
    /// snapshot (appended after the rings — see `vmm_snap_copy`).
    full_scale: Vec<(usize, usize)>,
    /// Sliding ring rows (`min(max_ctx, KV_RING)`), a power of two.
    ring: u64,
    /// Fixed snapshot region: rings (`n_slide × 2 × kvh_slide × window × hd ×
    /// elem_slide`) plus, for fp8 rings, their scale rows (`n_slide × 2 ×
    /// kvh_slide × window × 4`). The variable full-scale region
    /// (`vmm_full_scale_bytes`) is appended past this.
    snap_bytes: u64,
}

/// `kv.{l}.k` / `kv.{l}.v` → `(layer, 0|1)`.
fn kv_tensor_name(name: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix("kv.")?;
    let (l, t) = rest.split_once('.')?;
    let layer = l.parse().ok()?;
    match t {
        "k" => Some((layer, 0)),
        "v" => Some((layer, 1)),
        _ => None,
    }
}

use crate::asset::checkpoint::Checkpoint;

/// One uploaded prefill bucket (`t == chunk size`), run under the `_pf`
/// object. Its instruction stream is host-patched per chunk (KV write row,
/// flash `seq_kv`/`q_pos0`, lm_head `a_row0`) then the covering range is
/// re-uploaded — the port of the harness's chunked-prefill inner loop.
struct PrefillBucket {
    /// Chunk size this bucket was compiled for.
    t: u32,
    /// 128-byte kernarg (shares `tensors` + `gq_cursor` with the decode path).
    kernarg: DevProgram,
    /// Device instruction stream (patched per chunk over `[inst_lo..=inst_hi]`).
    d_inst: DeviceMem,
    /// Host copy of the instructions for the per-chunk patch.
    h_inst: Vec<DevInst64>,
    /// Contiguous instruction window covering every patch site.
    inst_lo: usize,
    inst_hi: usize,
    /// KV-writing `HeadNormRope` sites (`j[0] != 0`): patch `i[3] = c0`.
    rope_sites: Vec<usize>,
    /// `FlashPrefill` sites: patch `i[1] = c0+real`, `i[4] = c0`.
    flash_sites: Vec<usize>,
    /// lm_head GEMM sites (`M == 1`): patch `i[4] = real-1`.
    lmhead_sites: Vec<usize>,
    /// `FlashMerge` sites — neutered (`i[0] = 0`) in PX-1 batched mode, where
    /// the flash op runs the fused (`nsplit=1`, `t5=at`) epilogue per request.
    merge_sites: Vec<usize>,
    /// This bucket runs the fp8-KV opcodes (`HeadNormRopeFp8`/`FlashPrefillFp8`).
    /// PX-1 batched prefill is NOT available for them: the fp8 arm spends t6/t7
    /// on the k/v dequant scales, not on the request table, so the batched patch
    /// would overwrite a scale handle with a slot map.
    fp8_kv: bool,
    /// Whether the one-time PX-1 batched-mode patch has been applied (t6 = the
    /// slot-map / request-table handles, t5 = fused output, merge neutered).
    batch_patched: bool,
    /// This bucket's combined counter+cursor slab (its GQ cursor sits at
    /// the tail; `ctr_bytes` is the full reset size).
    d_ctr: DeviceMem,
    ctr_bytes: usize,
    /// The bucket's other tables, kept alive for the engine's lifetime.
    _tables: Vec<DeviceMem>,
}

/// The per-model GPU engine: one loaded decode program, one KV cache, one
/// live sequence. `begin()` resets the sequence; `step()` advances it.
pub struct GpuEngine {
    be: Arc<CudaBackend>,
    f: KernelFn,
    grid: u32,
    smem: u32,
    /// The engine's single ordered device queue: every decode/prefill copy,
    /// memset, and launch is enqueued here and retired by ONE
    /// `cuStreamSynchronize` per step — steady-state serving performs no
    /// `cuCtxSynchronize` (plan gate). One stream by design: decode and
    /// prefill share mutable run state (`in.*`, activations, the GQ cursor),
    /// so overlapping streams would race until every in-flight command owns
    /// separate run-state storage.
    stream: CudaStream,
    /// Keeps the interpreter module alive (id registered in the backend).
    _module: Module,

    /// The prefill object's kernel + smem, and the uploaded bucket programs.
    /// `None`/empty when no `_pf` cubin is present — the mux then falls back to
    /// decode-only prompt consumption.
    f_pf: Option<KernelFn>,
    smem_pf: u32,
    prefill: Vec<PrefillBucket>,
    _module_pf: Option<Module>,

    /// Device stochastic sampler (`PLOW_DEV_SAMPLE=1` + a sampler cubin;
    /// plan stage 4). `None` = the host path (full-logit D2H + CPU sampler),
    /// byte-identical to before. When present, `step_slots_sampled` launches
    /// `plow_sample` after the decode kernel to write each stochastic row's
    /// sampled token into `in.ids[b]` — no per-row vocab D2H.
    sampler: Option<Sampler>,
    /// Bounded device multi-step (`PLOW_MULTISTEP=K`; plan stage 5). `None` =
    /// the per-token launch model. When present, [`Self::multi_step`] runs a
    /// K-token greedy quantum with one host sync.
    multistep: Option<MultiStep>,

    /// Per-tensor device buffers, indexed by blob tensor handle.
    devp: Vec<DeviceMem>,
    d_inst: DeviceMem,
    /// Owner of the decode counter+cursor allocation. `d_ctr` and the decode
    /// GQ cursor are aliased **views** into it (never freed by their own
    /// Drop) — the owner must sit in the engine so their addresses stay valid
    /// for its whole lifetime. The cursor's address is baked into `kernarg`
    /// and re-armed as part of the `d_ctr` combined memset; the view is kept
    /// only to anchor its storage (prefill buckets carry their own cursor).
    _ctr_block: DeviceMem,
    d_ctr: DeviceMem,
    _d_gq_cursor: DeviceMem,
    /// The device tensor-pointer table (`kernarg.tensors`) — the batch-major
    /// bases decode expects. Immutable after load.
    d_tens: DeviceMem,
    /// Per-slot prefill tables (index b-1 = slot b; slot 0 is `d_tens`): the
    /// same table with every `kv.*` base shifted to that slot's ring, since
    /// the prefill programs address the cache slot-relative. A per-slot
    /// launch selects its table through the kernarg (`tens_slot_base`) —
    /// nothing is rewritten or restored. Empty at B == 1.
    d_tens_slots: Vec<DeviceMem>,
    /// The other decode tables live for the engine's lifetime and are never
    /// re-uploaded; their device pointers are baked into `kernarg`.
    _tables: Vec<DeviceMem>,

    /// The 128-byte `PlowProgram` kernarg (built once; pointers never move).
    kernarg: DevProgram,
    /// Host copy of the decode instructions for the per-step kv-row patch.
    h_inst: Vec<DevInst64>,
    kvrow: Vec<u32>,
    /// Contiguous instruction range covering every kv-row patch site.
    kvrow_lo: usize,
    kvrow_hi: usize,
    ctr_bytes: usize,

    t_ids: usize,
    t_pos: usize,
    t_kvlen: usize,
    t_logits: usize,
    /// Blob tensor names, indexed by handle (same order as `devp`). Lets the
    /// block harness resolve an activation handle by name — see
    /// [`Self::handle_of`] / [`Self::download_activation`] /
    /// [`Self::upload_activation`]. Not on any hot path.
    tensor_names: Vec<String>,
    vocab: usize,
    max_ctx: usize,

    /// Decode batch B — the last program's `t` (`PLOW_DECODE_BATCH` at
    /// compile time). The engine drives B independent sequence slots.
    batch: usize,
    /// Per-slot sequence position (== tokens consumed so far). For a slot
    /// mid-prefill (`prefill_chunk`) this is the prefill frontier — kept
    /// current per chunk so the batched-decode garbage KV write for unfed
    /// rows always lands in the row the slot's next chunk overwrites.
    pos: Vec<u32>,
    /// Per-slot rows served from the prefix cache by the current sequence's
    /// attach (0 = cold). Feeds per-request `usage.cached_tokens`.
    vmm_attached: Vec<u32>,
    /// Per-slot token ids whose KV rows the slot currently holds (prompt,
    /// then every decode-fed token) — `seq_tokens[b].len() == pos[b]` when
    /// consistent. Lets `begin_slot` publish the finished sequence's
    /// GENERATED blocks into the prefix cache, so a follow-up turn embedding
    /// this turn's output attaches instead of re-prefilling it.
    seq_tokens: Vec<Vec<u32>>,
    /// Stop-token set (the checkpoint's `eos_token_id`). `Arc` so the mux can
    /// take a per-tick handle without cloning the Vec while the engine stays
    /// mutably borrowed.
    stop_ids: std::sync::Arc<Vec<u32>>,
    /// Reusable download buffer for the bf16 logits row.
    logits_raw: Vec<u8>,
    /// Pinned per-step staging (`step_slots` is the per-token hot path — no
    /// per-step allocation, and pinned pages make the stream copies async).
    stage: StepStage,
    /// Pre-allocated prefill ids/pos buffers (sized to max prefill bucket t).
    pf_ids: Vec<i32>,
    pf_pos: Vec<i32>,
    /// Reusable f32 logits buffer for host sampling (gpu_finish_token).
    logits_f32: Vec<f32>,
    /// Env-gated (`PLOW_STEP_TIME=1`) host-op timing for the decode step.
    timing: Option<StepTiming>,
    /// VMM prefix sharing (`PLOW_VMM_PREFIX=1`); `None` = the cudaMalloc
    /// default path, byte-identical behavior to before this feature.
    vmm: Option<VmmServe>,
    /// PX-1 cross-request batched prefill (`PLOW_PF_BATCH=1`); `None` = the
    /// per-slot serialized prefill, byte-identical behavior to before.
    pf_batch: Option<PfBatch>,
}

/// Double-buffered checkpoint-upload pipeline (plan stage 9). Two pinned
/// staging buffers ping-pong on a dedicated stream so the host copy of chunk
/// N+1 (safetensors mmap → pinned) overlaps the async H2D DMA of chunk N —
/// the whole checkpoint (tens of GiB) is bandwidth-bound, and the serial
/// `copy; blocking-upload` loop left the copy and the DMA strictly sequential.
struct UploadPipe<'a> {
    be: &'a CudaBackend,
    stream: CudaStream,
    bufs: [PinnedHost; 2],
    events: [CudaEvent; 2],
    primed: [bool; 2],
    n: usize,
}

impl<'a> UploadPipe<'a> {
    fn new(be: &'a CudaBackend) -> Result<Self> {
        Ok(UploadPipe {
            stream: be.stream_create()?,
            bufs: [be.host_alloc_pinned(STAGE)?, be.host_alloc_pinned(STAGE)?],
            // Sync-only events (cheaper record) — used purely to gate buffer reuse.
            events: [be.event_create(false)?, be.event_create(false)?],
            primed: [false, false],
            n: 0,
            be,
        })
    }

    /// Stage `chunk` (≤ STAGE) into the next free pinned buffer and enqueue its
    /// async H2D to `dst` on the pipe's stream. Blocks only when that buffer's
    /// previous DMA has not yet retired (two-deep pipeline).
    fn push(&mut self, dst: u64, chunk: &[u8]) -> Result<()> {
        let slot = self.n & 1;
        if self.primed[slot] {
            self.be.event_synchronize(&self.events[slot])?;
        }
        self.bufs[slot].as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
        // SAFETY: the pinned buffer stays alive (owned by self) until finish()
        // synchronizes the stream; dst is inside a live allocation (caller).
        unsafe {
            self.be
                .memcpy_htod_async(dst, &self.bufs[slot].as_slice()[..chunk.len()], &self.stream)?;
        }
        self.be.event_record(&self.events[slot], &self.stream)?;
        self.primed[slot] = true;
        self.n += 1;
        Ok(())
    }

    /// Retire every enqueued upload (call before the pinned buffers drop).
    fn finish(&self) -> Result<()> {
        self.be.stream_synchronize(&self.stream)
    }
}

/// Pinned per-step staging slab: `[ids B][pos B][kvlen B]` i32, uploaded with
/// three async H2D copies on the engine stream. `ids` doubles as the token
/// readback destination — the same `in.ids` tensor round-trips (`ARGMAX_FIN`
/// writes the next token there), so no separate download buffer exists.
struct StepStage {
    slab: PinnedHost,
    batch: usize,
}

impl StepStage {
    fn new(be: &CudaBackend, batch: usize) -> Result<StepStage> {
        Ok(StepStage {
            slab: be.host_alloc_pinned(3 * batch * 4)?,
            batch,
        })
    }

    /// The three staging sections for the host fill.
    fn parts_mut(&mut self) -> (&mut [i32], &mut [i32], &mut [i32]) {
        let all: &mut [i32] = bytemuck::cast_slice_mut(self.slab.as_mut_slice());
        let (ids, rest) = all.split_at_mut(self.batch);
        let (pos, kvlen) = rest.split_at_mut(self.batch);
        (ids, pos, kvlen)
    }

    /// Byte view of section `k` (0 = ids, 1 = pos, 2 = kvlen) for the copies.
    fn section(&self, k: usize) -> &[u8] {
        &self.slab.as_slice()[k * self.batch * 4..(k + 1) * self.batch * 4]
    }

    /// Mutable byte view of section `k` — the async D2H destination.
    fn section_mut(&mut self, k: usize) -> &mut [u8] {
        let b = self.batch;
        &mut self.slab.as_mut_slice()[k * b * 4..(k + 1) * b * 4]
    }

    /// Token `b` read back into the ids section by the post-launch D2H.
    fn token(&self, b: usize) -> u32 {
        let ids: &[i32] = bytemuck::cast_slice(self.section(0));
        ids[b] as u32
    }
}

/// Accumulated `step_slots` timing, logged every 128 steps. Device phases
/// (upload+re-arm, interpreter, token D2H) come from CUDA events on the
/// engine stream — the plan's instrumentation rule; CPU timestamps around
/// async submission measure only enqueue cost. Host phases: `gap` is the time
/// between consecutive calls (the mux/serve layer per token), `submit` the
/// enqueue cost, `sync` the wait for the stream.
struct StepTiming {
    steps: u64,
    gap_ns: u64,
    submit_ns: u64,
    sync_ns: u64,
    upload_ms: f64,
    kernel_ms: f64,
    download_ms: f64,
    /// Stream-tail marks: start, uploads+re-arm done, interpreter done,
    /// token D2H done.
    ev: [CudaEvent; 4],
    last_end: Option<std::time::Instant>,
}

impl StepTiming {
    fn new(be: &CudaBackend) -> Result<StepTiming> {
        Ok(StepTiming {
            steps: 0,
            gap_ns: 0,
            submit_ns: 0,
            sync_ns: 0,
            upload_ms: 0.0,
            kernel_ms: 0.0,
            download_ms: 0.0,
            ev: [
                be.event_create(true)?,
                be.event_create(true)?,
                be.event_create(true)?,
                be.event_create(true)?,
            ],
            last_end: None,
        })
    }

    fn log_every(&mut self, n: u64) {
        if self.steps % n != 0 {
            return;
        }
        let us = |v: u64| v as f64 / 1000.0 / self.steps as f64;
        let ms = |v: f64| v / self.steps as f64;
        tracing::info!(
            steps = self.steps,
            gap_us = us(self.gap_ns),
            submit_us = us(self.submit_ns),
            sync_wait_ms = us(self.sync_ns) / 1000.0,
            dev_upload_us = ms(self.upload_ms) * 1000.0,
            dev_interp_ms = ms(self.kernel_ms),
            dev_download_us = ms(self.download_ms) * 1000.0,
            "step_slots means (host ns clocks + device CUDA events)"
        );
    }
}

/// PX-1 cross-request batched prefill state (`PLOW_PF_BATCH=1`). The two
/// device buffers are appended to the tensor table PAST the blob's handles:
/// the kernel sees them only where the host patches `t[6]` on prefill sites.
struct PfBatch {
    /// Per-row seq-slot map `i32[max bucket t]` — consumed by the KV-writing
    /// `HeadNormRope` sites (each packed row writes its own slot's ring).
    d_slot: DeviceMem,
    /// Request table `i32[1 + 4B]`: `[R, {q0, qlen, slot, kvlen} x R]` —
    /// consumed by the `FlashPrefill` sites (per-request-serial attention).
    d_req: DeviceMem,
    /// Tensor-table handles of the two buffers (blob handle count, +1).
    h_slot: u32,
    h_req: u32,
    /// `(t5, hd)` per flash site, harvested from a fused (`ns==1`) bucket —
    /// the per-layer `n.at` handles that turn a non-fused bucket's flash into
    /// the fused epilogue (its merge op is neutered).
    at_sites: Vec<(u32, u32)>,
    /// Host staging reused across launches (hot path — no per-launch alloc).
    slot_buf: Vec<i32>,
    req_buf: Vec<i32>,
}

/// One request's chunk inside a PX-1 batched prefill launch: rows
/// `prompt[c0..c0+len]` land at the pack's running row cursor, KV in seq-slot
/// `slot`. The mux packs chunks of N waiting requests under a token budget.
pub struct PfBatchReq<'a> {
    pub slot: usize,
    pub prompt: &'a [u32],
    pub c0: usize,
    pub len: usize,
}

/// Per-row device sampling request (plan stage 4). `temp <= 0` is greedy
/// (the device sampler writes the argmax, identical to `ARGMAX_FIN`), so a
/// spec array can carry greedy and stochastic rows together. `rng01` is the
/// request's per-step uniform draw (`seeded_unit`), so a fixed seed is
/// reproducible.
#[derive(Clone, Copy)]
pub struct DevSample {
    pub temp: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub rng01: f32,
}

impl DevSample {
    /// Greedy row (device argmax == ARGMAX_FIN).
    pub fn greedy() -> Self {
        DevSample { temp: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, rng01: 0.0 }
    }
}

/// Device sampler state: the `plow_sample` kernel + its per-slot parameter and
/// scratch buffers, all sized to the engine batch × vocab at load.
struct Sampler {
    f: KernelFn,
    _module: Module,
    /// Pinned staging for the 5 per-slot param arrays `[temp|topk|topp|minp|rng]`
    /// (each `batch` wide) — one async H2D per step.
    params: PinnedHost,
    /// Device copies of the 5 param arrays (contiguous, same layout as `params`).
    d_params: DeviceMem,
    /// `[batch][vocab]` f32 softmax-weight scratch the kernel reuses each pass.
    d_escratch: DeviceMem,
    #[allow(dead_code)]
    batch: usize,
}

/// Bounded device multi-step state (plan stage 5: `PLOW_MULTISTEP=K`). The
/// `plow_advance` kernel advances each active row's device-owned pos/kvlen and
/// appends its token to a `[batch][K]` ring BETWEEN decode launches, so a
/// K-token quantum runs with ONE host sync + ONE D2H instead of K — no
/// per-token host round trip. Requires a dynamic-kvrow decode cubin (the KV
/// write row comes from device pos) and greedy rows (device argmax).
struct MultiStep {
    f_advance: KernelFn,
    _module: Module,
    /// Token quantum K (steps per host round trip).
    quantum: usize,
    /// `[batch][K]` i32 token ring (device) + its pinned D2H staging.
    d_ring: DeviceMem,
    ring_host: PinnedHost,
    /// `[batch]` i32 active-row flags (device) + pinned staging.
    d_fed: DeviceMem,
    fed_host: PinnedHost,
    batch: usize,
}

/// Outcome of one [`GpuEngine::prefill_chunk`] call.
pub enum PrefillStep {
    /// The chunk advanced the prefill frontier to `.0` (< prompt len).
    Progress(usize),
    /// The prompt is fully consumed; `.0` is the first generated token
    /// (the exact postcondition of [`GpuEngine::prefill_slot`]).
    Done(u32),
}

impl GpuEngine {
    /// Bring the model up on the device: blob + cubin + weights + decode
    /// tables. Slow (a 12B checkpoint is ~22 GiB of H2D) — called once at
    /// server startup, never on the request path.
    pub fn load(be: Arc<CudaBackend>, assets_dir: &Path, checkpoint_dir: &Path) -> Result<Self> {
        let t0 = std::time::Instant::now();
        tracing::info!(
            assets = %assets_dir.display(),
            checkpoint = %checkpoint_dir.display(),
            device = %be.device_name(),
            "loading model onto GPU..."
        );

        // ---- blob ----
        let pkt = DevBlob::find_in_dir(assets_dir)?.ok_or_else(|| {
            RuntimeError::Device(format!("no PLOWDEV blob in {}", assets_dir.display()))
        })?;
        let raw = std::fs::read(&pkt)
            .map_err(|source| RuntimeError::Io { path: pkt.clone(), source })?;
        let blob = DevBlob::parse(&raw)?;
        tracing::info!(
            blob = %pkt.display(),
            tensors = blob.tensors.len(),
            n_cu = blob.n_cu,
            programs = blob.progs.len(),
            "parsed PLOWDEV blob"
        );
        let cc = be.compute_capability();
        let profile = interpreter_profile(cc).ok_or_else(|| {
            RuntimeError::Device(format!(
                "{} compute capability {}.{} has no persistent interpreter",
                be.device_name(), cc.0, cc.1
            ))
        })?;


        // ---- module ----
        // Priority: explicit override → matching named embedded section → filesystem.
        // Only SM120 accepts the legacy first-cubin fallback: no legacy Hopper blob exists,
        // and feeding an embedded SM120 image to a Hopper device is an opaque driver error.
        let embedded = blob
            .section_data_named(&raw, packet::devbuild::SECT_CUBIN, profile.embedded_decode)
            .or_else(|| {
                (profile.tag == "sm120")
                    .then(|| blob.section_data(&raw, packet::devbuild::SECT_CUBIN))
                    .flatten()
            });
        let image = if let Ok(p) = std::env::var("PLOW_NV_CUBIN") {
            let cubin_path = std::path::PathBuf::from(p);
            std::fs::read(&cubin_path).map_err(|source| RuntimeError::Io {
                path: cubin_path, source,
            })?
        } else if let Some(data) = embedded {
            data.to_vec()
        } else {
            let cubin_path = assets_dir.join(profile.decode_file);
            std::fs::read(&cubin_path).map_err(|source| RuntimeError::Io {
                path: cubin_path, source,
            })?
        };
        let module = be.module_load(&image)?;
        let kname = std::env::var("PLOW_NV_KERNEL").unwrap_or_else(|_| profile.decode_symbol.into());
        let f = be.get_function(&module, &kname)?;

        // ---- packet/object pairing ----
        // A SPECIALISED object carries only the arms one packet dispatches to, so
        // it is not interchangeable and must not be paired with a different packet.
        // Refusing here is the whole safety property of specialisation: without it,
        // a missing arm turns today's loud `default: __trap()` at first launch into
        // a trap MID-SERVE, on whichever bucket happens to need the dropped body.
        Self::check_packet_pairing(&be, &module, assets_dir)?;

        // Dynamic-smem arena: the cubin knows its own compile-time arena
        // (`plow_arena_bytes`, embedded by interp_sm120.cu) — a GF_FULL=4
        // flash-decode object needs 16448 B where the GF=2 default is 12352 B,
        // and launching short of the compiled claim is an illegal address on
        // the first flash op. Env override > cubin metadata > legacy default
        // (pre-metadata cubins are all GF=2 builds).
        let smem: u32 = match std::env::var("PLOW_NV_SMEM").ok().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => be
                .module_global_u32(&module, "plow_arena_bytes")?
                .unwrap_or(12352),
        };
        if smem > 48 * 1024 {
            be.set_max_dynamic_smem(f, smem)?;
        }

        // THE GRID MUST EQUAL n_cu (the harness's fatal gate): stream_ofs/len
        // are [n_cu] arrays indexed by blockIdx.x. A mismatched grid reads off
        // the tables or deadlocks the cooperative launch.
        let occ = be.occupancy_blocks_per_sm(f, BLOCK, smem as usize)?;
        let grid = occ * be.sm_count();
        if grid != blob.n_cu {
            return Err(RuntimeError::Device(format!(
                "interpreter grid {grid} ({occ}/SM × {} SMs) != packet n_cu {} — recompile \
                 the packet with n_cu={grid}",
                be.sm_count(),
                blob.n_cu
            )));
        }
        tracing::info!(
            profile = profile.tag,
            grid,
            smem,
            occ_per_sm = occ,
            cubin_bytes = image.len(),
            "interpreter module loaded"
        );

        // ---- VMM prefix sharing (PLOW_VMM_PREFIX=1; default off) ----
        // Full-layer kv.* tensors are then backed by VmmKv's VA reservations
        // (per-sequence windows, shareable blocks) instead of cudaMalloc.
        let vmm = Self::vmm_bringup(&be, &blob, checkpoint_dir);

        // ---- weights ----
        let t_weights = std::time::Instant::now();
        let ckpt = Checkpoint::open(checkpoint_dir)?;
        tracing::info!(
            checkpoint = %checkpoint_dir.display(),
            "checkpoint opened, starting weight upload to GPU..."
        );
        // Double-buffered pinned staging on a dedicated stream: the host copy
        // of the next chunk overlaps the async H2D of the current one (plan
        // stage 9). Pageable H2D runs at a fraction of pinned bandwidth and
        // this moves the whole checkpoint (tens of GiB).
        let mut pipe = UploadPipe::new(&be)?;

        let gen_by_tensor: std::collections::HashMap<u32, &packet::rope::GenTensor> =
            blob.gen.iter().map(|g| (g.tensor, g)).collect();
        let mut devp: Vec<DeviceMem> = Vec::with_capacity(blob.tensors.len());
        let (mut t_ids, mut t_pos, mut t_kvlen, mut t_logits) = (None, None, None, None);
        let (mut wb, mut kvb, mut nw) = (0u64, 0u64, 0usize);
        let mut upload_all = || -> Result<()> {
            for (i, td) in blob.tensors.iter().enumerate() {
                // Full-layer KV under VMM: the tensor base is the pool's VA
                // reservation (a view — the pool owns unmap/release); no
                // cudaMalloc and no memset (the VA is mapped lazily at the
                // per-sequence frontier; KV is always written before read).
                let vmm_va = vmm.as_ref().and_then(|v| {
                    let (l, t) = kv_tensor_name(&td.name)?;
                    v.kv.tensor_va(l, t)
                });
                let mem = match vmm_va {
                    Some(va) => DeviceMem::view(va, td.bytes),
                    None => be.alloc(0, td.bytes)?,
                };
                match td.name.as_str() {
                    "in.ids" => t_ids = Some(i),
                    "in.pos" => t_pos = Some(i),
                    "in.kvlen" => t_kvlen = Some(i),
                    "act.logits" => t_logits = Some(i),
                    _ => {}
                }
                if td.name.starts_with("kv.") {
                    kvb += td.bytes;
                }
                if td.name.starts_with("model.") || td.name.starts_with("fp8/") {
                    let src = ckpt.tensor(&td.name).ok_or_else(|| {
                        RuntimeError::Device(format!("MISSING WEIGHT: {}", td.name))
                    })?;
                    if src.len() as u64 != td.bytes {
                        return Err(RuntimeError::Device(format!(
                            "SIZE MISMATCH {} (want {} got {})",
                            td.name,
                            td.bytes,
                            src.len()
                        )));
                    }
                    tracing::debug!(
                        tensor = %td.name,
                        bytes = td.bytes,
                        mib = td.bytes / (1 << 20),
                        "uploading weight tensor"
                    );
                    // Double-buffered async H2D (overlaps host copy with DMA).
                    for (o, chunk) in src.chunks(STAGE).enumerate() {
                        pipe.push(mem.base + (o * STAGE) as u64, chunk)?;
                    }
                    wb += td.bytes;
                    nw += 1;
                } else if let Some(r) = &td.init {
                    for (o, chunk) in blob.init[r.clone()].chunks(STAGE).enumerate() {
                        pipe.push(mem.base + (o * STAGE) as u64, chunk)?;
                    }
                } else if let Some(g) = gen_by_tensor.get(&(i as u32)) {
                    // v7: the RoPE tables ride as recipes, not bytes. Materialise
                    // them here — same host-side f64 math the compiler ran, so the
                    // device sees the bytes a v5 blob would have carried. Must come
                    // before the memset arm below, which would leave cos=sin=0.
                    let data = g.generate().ok_or_else(|| {
                        RuntimeError::Device(format!(
                            "devblob: gen recipe for `{}` has unknown kind {}",
                            td.name, g.kind
                        ))
                    })?;
                    if data.len() as u64 != td.bytes {
                        return Err(RuntimeError::Device(format!(
                            "devblob: gen recipe for `{}` produced {} B, decl says {}",
                            td.name,
                            data.len(),
                            td.bytes
                        )));
                    }
                    for (o, chunk) in data.chunks(STAGE).enumerate() {
                        pipe.push(mem.base + (o * STAGE) as u64, chunk)?;
                    }
                    tracing::debug!(
                        tensor = %td.name, bytes = td.bytes, kind = g.kind,
                        "materialised generated tensor"
                    );
                } else if vmm_va.is_none() && !td.name.starts_with("kv.") {
                    // VMM windows are (partially) unmapped VA — a memset would
                    // fault. The cudaMalloc KV cache (plan stage 9: avoid
                    // zeroing proven write-before-read) needs none either:
                    // attention reads only [0, kvlen), every row of which was
                    // written by prefill/decode before it is read, and idle
                    // rows' garbage is bounded out by kvlen — so skip its
                    // memset (10.5 GiB on a B=4 ctx-8k engine). Other scratch
                    // (act.*/in.*) stays zeroed (cheap, conservative).
                    be.memset_d8(mem.base, 0, td.bytes as usize)?;
                }
                devp.push(mem);
            }
            Ok(())
        };
        let enqueued = upload_all();
        drop(upload_all); // release the &mut pipe borrow before finishing it
        // Retire every enqueued async H2D, then release the pinned buffers
        // (2×64 MiB) + load stream before serving.
        let uploaded = enqueued.and_then(|()| pipe.finish());
        drop(pipe);
        uploaded?;
        let weight_elapsed = t_weights.elapsed();
        let weight_gib = wb as f64 / (1u64 << 30) as f64;
        let throughput = if weight_elapsed.as_secs_f64() > 0.0 {
            weight_gib / weight_elapsed.as_secs_f64()
        } else {
            0.0
        };
        tracing::info!(
            tensors = nw,
            weight_gib = format!("{weight_gib:.2}").as_str(),
            kv_gib = format!("{:.2}", kvb as f64 / (1u64 << 30) as f64).as_str(),
            elapsed_s = format!("{:.1}", weight_elapsed.as_secs_f64()).as_str(),
            throughput_gib_s = format!("{throughput:.2}").as_str(),
            "checkpoint weights uploaded to GPU"
        );
        let (t_ids, t_pos, t_kvlen, t_logits) =
            match (t_ids, t_pos, t_kvlen, t_logits) {
                (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
                _ => {
                    return Err(RuntimeError::Device(
                        "blob is missing in.ids/in.pos/in.kvlen/act.logits".into(),
                    ))
                }
            };

        // ---- Gemma-4 26B-A4B fused-MoE expert pointer tables ----
        // Port of the chat harness's name-driven fill (gemma4_sm120_chat.cu §MoE):
        // for each `moe.ewt.{l}` slot, resolve the two FUSED expert tensors by name
        // suffix, derive E and the per-expert byte strides from tensor sizes alone,
        // and upload the [E][2] {gate_up, down} base table the SM indexes
        // (orch::moe::build_fused_expert_table). fp8 packets additionally fill
        // `moe.est.{l}` with the per-row scale bases. Guarded on the `moe.ewt.`
        // prefix, so dense (12B/31B/Qwen) blobs are untouched. Suffix scans keep
        // the LAST match, mirroring the harness (an fp8 twin shadows a bf16 name).
        for i in 0..blob.tensors.len() {
            let Some(layer) = blob.tensors[i].name.strip_prefix("moe.ewt.") else {
                continue;
            };
            let layer = layer.to_string();
            let find = |suf: &str| {
                blob.tensors
                    .iter()
                    .rposition(|t| t.name.ends_with(suf))
            };
            let suf_gu = format!("layers.{layer}.experts.gate_up_proj");
            let suf_dn = format!("layers.{layer}.experts.down_proj");
            let (Some(gu), Some(dn)) = (find(&suf_gu), find(&suf_dn)) else {
                return Err(RuntimeError::Device(format!(
                    "MoE: layer {layer} missing fused expert tensor(s)"
                )));
            };
            let e = blob.tensors[i].bytes / 16; // ewt = E*2 u64
            let table = crate::orch::moe::build_fused_expert_table(
                devp[gu].base,
                devp[dn].base,
                e as u32,
                blob.tensors[gu].bytes / e,
                blob.tensors[dn].bytes / e,
            );
            be.upload(&devp[i], 0, bytemuck::cast_slice(&table))?;
            if blob.tensors[gu].name.starts_with("fp8/") {
                let est_name = format!("moe.est.{layer}");
                let suf_gs = format!("layers.{layer}.experts.gate_up_proj_scale");
                let suf_ds = format!("layers.{layer}.experts.down_proj_scale");
                let est = blob.tensors.iter().position(|t| t.name == est_name);
                let (Some(est), Some(gs), Some(ds)) = (est, find(&suf_gs), find(&suf_ds))
                else {
                    return Err(RuntimeError::Device(format!(
                        "MoE fp8: layer {layer} missing expert scale tensor/table"
                    )));
                };
                let stable = crate::orch::moe::build_fused_expert_table(
                    devp[gs].base,
                    devp[ds].base,
                    e as u32,
                    blob.tensors[gs].bytes / e,
                    blob.tensors[ds].bytes / e,
                );
                be.upload(&devp[est], 0, bytemuck::cast_slice(&stable))?;
            }
        }

        // ---- PX-1 batched-prefill buffers (PLOW_PF_BATCH=1) ----
        // Allocated BEFORE the tensor table so their pointers ride in it past
        // the blob's handles (h_slot = len, h_req = len+1). Tiny (≤32 KiB), so
        // allocation is unconditional on the env flag only; the mode itself is
        // finalized after the prefill object loads (see below).
        let pf_batch_env = std::env::var("PLOW_PF_BATCH").map(|v| v == "1").unwrap_or(false);
        let pf_max_t_blob = blob.progs[..blob.progs.len().saturating_sub(1)]
            .iter()
            .map(|g| g.t as usize)
            .max()
            .unwrap_or(0);
        let dbatch_blob = blob.decode_prog().map(|g| g.t as usize).unwrap_or(1);
        let pf_bufs = if pf_batch_env && pf_max_t_blob > 0 {
            Some((
                be.alloc(0, (pf_max_t_blob * 4) as u64)?,
                be.alloc(0, ((1 + 4 * dbatch_blob) * 4) as u64)?,
            ))
        } else {
            None
        };

        let mut ptrs: Vec<u64> = devp.iter().map(|m| m.base).collect();
        let pf_handles = pf_bufs.as_ref().map(|(s, r)| {
            let h_slot = ptrs.len() as u32;
            // These runtime-appended handles are patched into u16 wire slots
            // (`DevInst64::t`), which the compiler's pack-time assert cannot see.
            assert!(h_slot + 1 < TENSOR_NONE16 as u32, "tensor table overflows u16 handles");
            ptrs.push(s.base);
            ptrs.push(r.base);
            (h_slot, h_slot + 1)
        });
        let d_tens = be.alloc(0, (ptrs.len() * 8) as u64)?;
        be.upload(&d_tens, 0, bytemuck::cast_slice(&ptrs))?;

        // ---- decode program tables ----
        let g = blob.decode_prog()?;
        // Decode batch: the compiler emits the decode program with t == B
        // (PLOW_DECODE_BATCH). Cross-check against the [B]-sized in.kvlen.
        let batch = g.t as usize;
        if blob.tensors[t_kvlen].bytes != (batch * 4) as u64 {
            return Err(RuntimeError::Device(format!(
                "in.kvlen is {} B but the decode program's batch is {batch} \
                 (want {} B) — blob/tensor mismatch",
                blob.tensors[t_kvlen].bytes,
                batch * 4
            )));
        }
        g.check_coarse_single_segment()?;
        if g.gq_stream.is_empty() || g.gq_seg_ofs.len() != 2 {
            return Err(RuntimeError::Device(format!(
                "decode program has no single-segment GQ appendix (n_seg bounds: {:?}) — \
                 recompile the packet with a GQ-capable plowc",
                g.gq_seg_ofs
            )));
        }
        g.check_gq_topological()?;

        let upload_pod = |bytes: &[u8]| -> Result<DeviceMem> {
            let mem = be.alloc(0, bytes.len().max(4) as u64)?;
            if !bytes.is_empty() {
                be.upload(&mem, 0, bytes)?;
            }
            Ok(mem)
        };
        fn pod_bytes<T: Copy>(v: &[T]) -> &[u8] {
            // SAFETY: #[repr(C)] POD mirrors, read as raw bytes for upload.
            unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
            }
        }
        // Dynamic B=1 KV row (plan stage 2): when the cubin advertises
        // `plow_dyn_kvrow`, arm the pos[t]-derived KV row on every B=1
        // KV-write site ONCE (i[6] = 1) — the decode instruction stream is
        // then immutable and `step_slots` never rewrites/re-uploads it.
        // Byte-identical rows: the dynamic formula reads in.pos[0], which the
        // step uploads with exactly the value the host used to patch i[3].
        // B>1 programs already carry i[6] = B from the compiler; old cubins
        // (no metadata symbol) keep the legacy per-token patch.
        let mut insts = g.insts.clone();
        let mut kvrow = blob.kvrow.clone();
        if batch == 1
            && !kvrow.is_empty()
            && be.module_global_u32(&module, "plow_dyn_kvrow")?.is_some()
        {
            for &ix in &kvrow {
                insts[ix as usize].i[6] = 1;
            }
            kvrow.clear();
            tracing::info!("decode: dynamic B=1 KV row — immutable instruction stream");
        }
        let d_inst = upload_pod(pod_bytes(&insts))?;
        let d_stream = upload_pod(pod_bytes(&g.stream))?;
        let d_sofs = upload_pod(pod_bytes(&g.stream_ofs))?;
        let d_slen = upload_pod(pod_bytes(&g.stream_len))?;
        let d_waits = upload_pod(pod_bytes(&g.waits))?;
        let d_succs = upload_pod(pod_bytes(&g.succs))?;
        let d_gq_stream = upload_pod(pod_bytes(&g.gq_stream))?;
        let d_gq_seg = upload_pod(pod_bytes(&g.gq_seg_ofs))?;
        // The decode counter block and the GQ cursor share a lifecycle (both
        // re-zeroed before every launch), so they share one allocation: the
        // cursor sits at the counter block's tail and ONE memset re-arms both.
        let ctr_bytes = g.n_counter as usize * CTR_STRIDE as usize * 4;
        let cursor_bytes = CTR_STRIDE as usize * 4;
        let ctr_block = be.alloc(0, (ctr_bytes.max(4) + cursor_bytes) as u64)?;
        // Two aliased views of the one owned block; `ctr_block` is stored in
        // the engine so the storage outlives both.
        let d_ctr = DeviceMem::view(ctr_block.base, ctr_bytes.max(4) as u64);
        let d_gq_cursor = DeviceMem::view(
            ctr_block.base + ctr_bytes.max(4) as u64,
            cursor_bytes as u64,
        );

        let kernarg = DevProgram {
            insts: d_inst.base,
            stream: d_stream.base,
            stream_ofs: d_sofs.base,
            stream_len: d_slen.base,
            waits: d_waits.base,
            succs: d_succs.base,
            counters: d_ctr.base,
            tensors: d_tens.base,
            trace: 0,
            cur_seg: 0,
            _segpad: 0,
            gq_stream: d_gq_stream.base,
            gq_seg_ofs: d_gq_seg.base,
            gq_cursor: d_gq_cursor.base,
            xctr: 0,
            peer_scratch: 0,
            rank: 0,
            n_gpu: 1,
        };

        // kv-row patch range: the contiguous [lo..hi] instruction window the
        // per-step upload rewrites (harness `run_step`).
        let (mut lo, mut hi) = (g.insts.len().saturating_sub(1), 0usize);
        for &ix in &blob.kvrow {
            lo = lo.min(ix as usize);
            hi = hi.max(ix as usize);
        }

        // act.logits is [B][vocab] bf16; in.pos is [ctx] i32 (shared by the
        // prefill chunk positions and the B decode positions).
        let vocab = (blob.tensors[t_logits].bytes / 2) as usize / batch;
        let max_ctx = (blob.tensors[t_pos].bytes / 4) as usize;

        // Per-slot stride of every batch-major `kv.*` tensor (slot b of
        // tensor i lives at base + b*stride). B==1: strides never used.
        let kv_slots: Vec<(usize, u64)> = blob
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, td)| td.name.starts_with("kv."))
            .map(|(i, td)| (i, td.bytes / batch as u64))
            .collect();

        // Stage 3: one immutable device tensor table PER SLOT, built at load
        // (index b-1 = slot b; slot 0 is `d_tens` itself). A per-slot prefill
        // launch selects its table through the kernarg instead of rewriting
        // and restoring the shared table around every chunk chain — the
        // tables never change after this point, and decode can never observe
        // a shifted table. A few KiB per slot.
        let mut d_tens_slots: Vec<DeviceMem> = Vec::new();
        if batch > 1 && !kv_slots.is_empty() {
            let mut shifted = ptrs.clone();
            for b in 1..batch {
                for &(i, stride) in &kv_slots {
                    shifted[i] = ptrs[i] + b as u64 * stride;
                }
                let mem = be.alloc(0, (shifted.len() * 8) as u64)?;
                be.upload(&mem, 0, bytemuck::cast_slice(&shifted))?;
                d_tens_slots.push(mem);
            }
        }

        // Every slot's row 0 must be mapped before any batched decode: unfed
        // rows write garbage KV at their own pos (mux contract), and pos
        // starts at 0.
        if let Some(v) = &vmm {
            for b in 0..batch {
                v.kv.ensure_rows(b, 1)?;
            }
        }

        // Stop set from the checkpoint's generation_config (fallback config).
        let stop_ids = read_eos_ids(checkpoint_dir);

        // ---- prefill object + bucket programs (optional) ----
        // Load the `_pf` cubin and upload every non-decode (T!=1) program so a
        // prompt is consumed in chunks by the tiled-GEMM/flash-prefill buckets
        // instead of O(n) decode launches. Absent cubin, a segmented bucket, or
        // a missing GQ appendix disables prefill (mux falls back to decode-only
        // consumption) rather than failing the whole engine.
        let pf_cubin = std::env::var("PLOW_NV_CUBIN_PF")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| assets_dir.join(profile.prefill_file));
        let (f_pf, smem_pf, module_pf, prefill) = if pf_cubin.is_file() {
            match Self::load_prefill(&be, &pf_cubin, &blob, d_tens.base, grid, profile.prefill_symbol) {
                Ok((f_pf, smem_pf, module_pf, buckets)) => {
                    tracing::info!(
                        pf_cubin = %pf_cubin.display(),
                        buckets = buckets.len(),
                        smem_pf,
                        "prefill object loaded"
                    );
                    (Some(f_pf), smem_pf, Some(module_pf), buckets)
                }
                // HARD ERROR, not a fallback. The cubin is PRESENT and failed to
                // load — a broken deployment, not a configuration. Falling back
                // to decode-only means consuming the prompt ONE TOKEN AT A TIME:
                // measured >=707 s on a 127k prompt (PX-13) against ~32 s with
                // the buckets, and the only symptom is that it is slow. The
                // usual cause is an arena over the 101376 B opt-in cap
                // (`PGM_STAGES=5`, `PGM_BN=256`), which is a build mistake the
                // operator must see. A genuinely absent cubin still falls back,
                // below — that path is documented and intentional.
                Err(e) => {
                    return Err(RuntimeError::Device(format!(
                        "prefill object {} failed to load: {e}\n\
                         Refusing to start: falling back to decode-only prompt consumption \
                         would prefill one token at a time (measured >=707 s vs ~32 s on a \
                         127k prompt) with no symptom but slowness.\n\
                         Common cause: the object's shared-memory arena exceeds this device's \
                         opt-in cap — check `-Xptxas -v` against the 101376 B limit (e.g. \
                         PGM_STAGES=5 or PGM_BN=256 overflow it). Remove the prefill cubin \
                         to serve decode-only deliberately.",
                        pf_cubin.display()
                    )));
                }
            }
        } else {
            tracing::info!(
                pf_cubin = %pf_cubin.display(),
                "no prefill cubin — decode-only prompt consumption"
            );
            (None, SMEM_PF, None, Vec::new())
        };

        // ---- PX-1 batched-prefill mode (finalized once the prefill object is up) ----
        // Requirements: env flag, prefill loaded, no VMM prefix sharing (its
        // per-slot attach/publish assumes the serialized per-slot chain), and
        // at least one FUSED (ns==1, t5=at) bucket to harvest the per-layer
        // `n.at` handles from. Anything missing → warn + serialized prefill.
        let pf_batch: Option<PfBatch> = match (pf_bufs, pf_handles) {
            (Some((d_slot, d_req)), Some((h_slot, h_req))) if f_pf.is_some() && !prefill.is_empty() => {
                if vmm.is_some() {
                    tracing::warn!("PLOW_PF_BATCH=1 ignored: incompatible with PLOW_VMM_PREFIX=1");
                    None
                } else {
                    let fused = prefill.iter().find(|b| {
                        // `!b.fp8_kv`: the fp8 flash arm reads t6/t7 as the k/v
                        // scales, so the batched patch (t6 = request table) would
                        // hand it garbage. The kernel has no fp8 mux arm either.
                        !b.fp8_kv
                            && !b.flash_sites.is_empty()
                            && b.flash_sites.iter().all(|&ix| {
                                b.h_inst[ix].i[7] == 1 && b.h_inst[ix].t[5] != TENSOR_NONE16
                            })
                    });
                    if fused.is_none() && prefill.iter().any(|b| b.fp8_kv) {
                        tracing::warn!(
                            "PLOW_PF_BATCH=1 ignored: fp8-KV packets have no batched prefill arm"
                        );
                    }
                    match fused {
                        Some(b) => {
                            let at_sites: Vec<(u32, u32)> = b
                                .flash_sites
                                .iter()
                                .map(|&ix| (b.h_inst[ix].t[5] as u32, b.h_inst[ix].i[6]))
                                .collect();
                            tracing::info!(
                                fused_bucket_t = b.t,
                                flash_sites = at_sites.len(),
                                "PX-1 cross-request batched prefill enabled (PLOW_PF_BATCH=1)"
                            );
                            Some(PfBatch {
                                d_slot,
                                d_req,
                                h_slot,
                                h_req,
                                at_sites,
                                slot_buf: vec![0; pf_max_t_blob],
                                req_buf: Vec::with_capacity(1 + 4 * dbatch_blob),
                            })
                        }
                        None => {
                            tracing::warn!(
                                "PLOW_PF_BATCH=1 ignored: no fused (nsplit==1) prefill bucket"
                            );
                            None
                        }
                    }
                }
            }
            (Some(_), _) => {
                tracing::warn!("PLOW_PF_BATCH=1 ignored: prefill object not loaded");
                None
            }
            _ => None,
        };

        tracing::info!(
            pkt = %pkt.display(),
            grid,
            weights_gib = wb as f64 / (1u64 << 30) as f64,
            kv_gib = kvb as f64 / (1u64 << 30) as f64,
            n_weights = nw,
            vocab,
            max_ctx,
            batch,
            prefill_buckets = prefill.len(),
            stop_ids = ?stop_ids,
            vmm_prefix = vmm.is_some(),
            elapsed_s = t0.elapsed().as_secs_f32(),
            // Was a hardcoded "sm_120" from the sm120-only era. On a Hopper card it
            // printed sm_120 while running the sm90a object, which reads as a
            // wrong-profile bug during exactly the kind of bring-up that needs this
            // line to be trustworthy. Report the profile actually selected.
            interp = profile.tag,
            "GPU engine loaded (decode program)"
        );

        let pf_max_t = prefill.last().map_or(0, |b| b.t as usize);

        // The engine's ordered device queue + pinned per-step staging + the
        // env-gated CUDA-event timing (plan stage 1: async submission path).
        let stream = be.stream_create()?;
        let stage = StepStage::new(&be, batch)?;
        let timing = match std::env::var("PLOW_STEP_TIME").ok().filter(|v| v == "1") {
            Some(_) => Some(StepTiming::new(&be)?),
            None => None,
        };

        // ---- device stochastic sampler (PLOW_DEV_SAMPLE=1; plan stage 4) ----
        // Loads a `plow_sample` cubin and allocates its per-slot param + [B][V]
        // scratch buffers. Absent flag/cubin → host sampling (unchanged).
        let sampler = Self::sampler_bringup(&be, assets_dir, batch, vocab)
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "device sampler disabled — host sampling");
                None
            });
        // Stage 5: bounded device multi-step. Requires the KV write row to come
        // from device `pos` (no per-step host patch — the thing multi-step
        // removes). True for B>1 (the kernel uses n_batch_kv from pos[b], and
        // step_slots never patches i[3] there) and for B==1 only when the
        // dynamic-kvrow arm fired (the local `kvrow` was cleared above). A B==1
        // legacy cubin still host-patches i[3] each step, so multi-step is off.
        let dyn_kvrow = batch > 1 || kvrow.is_empty();
        let multistep = Self::multistep_bringup(&be, assets_dir, batch, dyn_kvrow)
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "multi-step disabled");
                None
            });

        Ok(GpuEngine {
            be,
            f,
            grid,
            smem,
            stream,
            _module: module,
            f_pf,
            smem_pf,
            prefill,
            _module_pf: module_pf,
            sampler,
            multistep,
            h_inst: insts,
            kvrow,
            kvrow_lo: lo,
            kvrow_hi: hi,
            ctr_bytes,
            devp,
            d_inst,
            _ctr_block: ctr_block,
            d_ctr,
            _d_gq_cursor: d_gq_cursor,
            d_tens,
            d_tens_slots,
            _tables: vec![
                d_stream, d_sofs, d_slen, d_waits, d_succs, d_gq_stream, d_gq_seg,
            ],
            kernarg,
            t_ids,
            t_pos,
            t_kvlen,
            t_logits,
            tensor_names: blob.tensors.iter().map(|t| t.name.clone()).collect(),
            vocab,
            max_ctx,
            batch,
            pos: vec![0; batch],
            vmm_attached: vec![0; batch],
            seq_tokens: vec![Vec::new(); batch],
            stop_ids: std::sync::Arc::new(stop_ids),
            logits_raw: Vec::new(),
            stage,
            pf_ids: vec![0i32; pf_max_t],
            pf_pos: vec![0i32; pf_max_t],
            logits_f32: Vec::new(),
            timing,
            vmm,
            pf_batch,
        })
    }

    /// Refuse a specialised interpreter object paired with a different packet.
    ///
    /// The object stamps `plow_packet_hash_{lo,hi}` (interp_sm120.cu, next to
    /// `plow_arena_bytes`) ONLY when it was built from one packet's `build.json`
    /// — a general object, with every arm compiled, carries no stamp and pairs
    /// with anything, so every asset shipped today is unaffected.
    ///
    /// The expected value is read from `<assets>/build.json`, which `plowc`
    /// writes beside `model.pkt` from the very programs it serialized. No hashing
    /// happens here on purpose: recomputing it would be a second implementation of
    /// the rule and could disagree with the first, which is exactly the class of
    /// bug this check exists to end.
    ///
    /// FAILURE IS FATAL, not a warning. A stamped object is missing arms; running
    /// it against the wrong packet does not degrade, it traps mid-serve.
    fn check_packet_pairing(
        be: &Arc<CudaBackend>,
        module: &Module,
        assets_dir: &Path,
    ) -> Result<()> {
        let (lo, hi) = (
            be.module_global_u32(module, "plow_packet_hash_lo")?,
            be.module_global_u32(module, "plow_packet_hash_hi")?,
        );
        let (Some(lo), Some(hi)) = (lo, hi) else {
            // Unstamped ⇒ a general object. Nothing to check.
            return Ok(());
        };
        let stamped = ((hi as u64) << 32) | lo as u64;

        let mpath = assets_dir.join("build.json");
        let manifest = std::fs::read(&mpath).map_err(|source| RuntimeError::Io {
            path: mpath.clone(),
            source,
        })?;
        let manifest: serde_json::Value = serde_json::from_slice(&manifest).map_err(|e| {
            RuntimeError::Device(format!("{}: not valid JSON: {e}", mpath.display()))
        })?;
        let want = manifest
            .get("pairing")
            .and_then(|p| p.get("hash"))
            .and_then(|h| h.as_str())
            .and_then(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16).ok())
            .ok_or_else(|| {
                RuntimeError::Device(format!(
                    "the interpreter object is SPECIALISED (packet hash \
                     0x{stamped:016x}) but {} has no `pairing.hash` — it was written \
                     by an older plowc. Rebuild the packet, or serve a general \
                     interpreter object (one built without -DPLOW_CONFIG).",
                    mpath.display()
                ))
            })?;
        if want != stamped {
            // Name what differs, not just that something does: the arm set and the
            // rule-derived constants are the two things the hash covers.
            let arms = manifest
                .get("union")
                .and_then(|u| u.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            return Err(RuntimeError::Device(format!(
                "packet/interpreter MISMATCH: the loaded cubin was specialised for \
                 packet 0x{stamped:016x}, but the packet in {} is 0x{want:016x} \
                 ({arms} arms, tuning {}). A specialised object carries ONLY that \
                 packet's op arms, so running this pair would trap mid-serve on the \
                 first bucket needing a dropped arm. Rebuild the cubin from this \
                 packet's build.json (plowc --emit devblob+cubin), or serve a \
                 general object built without -DPLOW_CONFIG.",
                assets_dir.display(),
                manifest.get("tuning").map(|t| t.to_string()).unwrap_or_default(),
            ))
            .into());
        }
        tracing::info!(packet_hash = format!("0x{stamped:016x}"), "specialised interpreter paired");
        Ok(())
    }

    /// Bring up VMM prefix sharing when `PLOW_VMM_PREFIX=1` and the model's
    /// KV geometry (from the checkpoint's `config.json`) validates against
    /// the blob's declared tensor sizes. Any mismatch logs and falls back to
    /// the cudaMalloc path — never fails the load.
    fn vmm_bringup(
        be: &Arc<CudaBackend>,
        blob: &DevBlob,
        checkpoint_dir: &Path,
    ) -> Option<VmmServe> {
        if std::env::var("PLOW_VMM_PREFIX").as_deref() != Ok("1") {
            return None;
        }
        let batch = blob.decode_prog().ok()?.t;
        let max_ctx = blob
            .tensors
            .iter()
            .find(|t| t.name == "in.pos")
            .map(|t| (t.bytes / 4) as u32)?;
        let Some(mut geo) =
            crate::memory::vmm::VmmGeometry::from_config(checkpoint_dir, max_ctx, batch)
        else {
            tracing::warn!("vmm off: no usable KV geometry in config.json");
            return None;
        };
        let find = |name: &str| blob.tensors.iter().position(|t| t.name == name);

        // KV dtype per layer group, resolved from the blob itself: the
        // emitter declares `kv.{l}.k_scale`/`.v_scale` iff that layer's cache
        // is fp8 e4m3 (1 B/elem + per-row f32 scales); bf16/fp16 layers have
        // no scale tensors and 2 B/elem. Presence is the discriminator —
        // byte-size inference alone is ambiguous (2× ring vs 2× elem).
        let scales_of = |l: u32| -> Option<(usize, usize)> {
            match (find(&format!("kv.{l}.k_scale")), find(&format!("kv.{l}.v_scale"))) {
                (Some(ik), Some(iv)) => Some((ik, iv)),
                _ => None,
            }
        };
        let full_fp8 = geo.full_layers.first().map(|&l| scales_of(l).is_some());
        geo.elem = if full_fp8 == Some(true) { 1 } else { 2 };

        // Full layers: declared bytes must equal the batch-major shape at the
        // resolved elem, and the fp8 discriminator must be uniform — a layer
        // disagreeing with the first one is geometry drift, not a mode.
        let mut full_scale = Vec::new();
        for &l in &geo.full_layers {
            for t in ["k", "v"] {
                let Some(i) = find(&format!("kv.{l}.{t}")) else {
                    tracing::warn!(layer = l, "vmm off: missing full-layer KV tensor");
                    return None;
                };
                if blob.tensors[i].bytes != geo.full_tensor_bytes() {
                    tracing::warn!(
                        layer = l,
                        declared = blob.tensors[i].bytes,
                        expected = geo.full_tensor_bytes(),
                        "vmm off: full-layer KV bytes mismatch (geometry drift)"
                    );
                    return None;
                }
            }
            match (geo.elem, scales_of(l)) {
                (2, None) => {}
                (1, Some((ik, iv))) => {
                    let want = batch as u64 * geo.kvh_full as u64 * max_ctx as u64 * 4;
                    if blob.tensors[ik].bytes != want || blob.tensors[iv].bytes != want {
                        tracing::warn!(layer = l, "vmm off: full-layer KV scale bytes mismatch");
                        return None;
                    }
                    full_scale.push((ik, iv));
                }
                _ => {
                    tracing::warn!(layer = l, "vmm off: mixed KV dtypes across full layers");
                    return None;
                }
            }
        }

        // Sliding layers: resolve ring geometry for the boundary snapshots.
        // Ring dtype is independent of the full layers' (PLOW_FP8_KV_FULL=1
        // keeps the rings bf16 under fp8 full layers).
        let slide_fp8 = geo.slide_layers.first().map(|&l| scales_of(l).is_some());
        geo.elem_slide = if slide_fp8 == Some(true) { 1 } else { 2 };
        let hd_b = (geo.hd_slide * geo.elem_slide) as u64;
        let mut slide = Vec::with_capacity(geo.slide_layers.len());
        let mut slide_scale = Vec::new();
        let mut ring = 0u64;
        for &l in &geo.slide_layers {
            let (Some(ik), Some(iv)) =
                (find(&format!("kv.{l}.k")), find(&format!("kv.{l}.v")))
            else {
                tracing::warn!(layer = l, "vmm off: missing sliding KV tensor");
                return None;
            };
            let stride = blob.tensors[ik].bytes / batch as u64;
            let r = stride / (geo.kvh_slide as u64 * hd_b);
            if blob.tensors[iv].bytes != blob.tensors[ik].bytes
                || r * geo.kvh_slide as u64 * hd_b != stride
                || !r.is_power_of_two()
                || r < geo.window as u64
                || (ring != 0 && ring != r)
            {
                tracing::warn!(layer = l, "vmm off: sliding ring geometry mismatch");
                return None;
            }
            ring = r;
            slide.push((ik, iv, stride));
            match (geo.elem_slide, scales_of(l)) {
                (2, None) => {}
                (1, Some((sk, sv))) => {
                    let want = batch as u64 * geo.kvh_slide as u64 * r * 4;
                    if blob.tensors[sk].bytes != want || blob.tensors[sv].bytes != want {
                        tracing::warn!(layer = l, "vmm off: sliding KV scale bytes mismatch");
                        return None;
                    }
                    slide_scale.push((sk, sv));
                }
                _ => {
                    tracing::warn!(layer = l, "vmm off: mixed KV dtypes across sliding layers");
                    return None;
                }
            }
        }
        // Fixed snapshot region: ring rows, then (fp8 rings) their scale rows.
        let slide_rows = slide.len() as u64 * 2 * geo.kvh_slide as u64 * geo.window as u64;
        let snap_bytes = (slide_rows * hd_b
            + if geo.elem_slide == 1 { slide_rows * 4 } else { 0 })
        .max(4);

        let mib = |var: &str, default: u64| {
            std::env::var(var)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map(|m| m << 20)
                .unwrap_or(default)
        };
        // Default sharing block = the driver granularity (2 MiB measured):
        // the finest match unit VMM can map, e.g. 4096 tokens at hd256 bf16 —
        // what makes shared system prompts / multi-turn histories actually
        // hit. Attach cost stays sane because set_access is coalesced over
        // contiguous granule runs (one call per span, not per block). The
        // 128k-dedup campaign can still raise it via PLOW_VMM_BLOCK_MIB=64.
        let block_hint = mib("PLOW_VMM_BLOCK_MIB", 2 << 20);
        let cache_cap = mib("PLOW_VMM_CACHE_MIB", 0);
        match crate::memory::vmm::VmmKv::new(
            Arc::clone(be) as Arc<dyn crate::memory::vmm::VmmOps>,
            geo,
            block_hint,
            cache_cap,
        ) {
            Ok(kv) => Some(VmmServe {
                kv,
                slide,
                slide_scale,
                full_scale,
                ring,
                snap_bytes,
            }),
            Err(e) => {
                tracing::warn!(error = %e, "vmm off: pool bringup failed");
                None
            }
        }
    }

    /// Bring up the device sampler when `PLOW_DEV_SAMPLE=1` and a `plow_sample`
    /// cubin is found (`PLOW_NV_CUBIN_SAMPLE`, else `<assets>/sample_sm120.cubin`).
    /// Any missing piece returns `Ok(None)` (host sampling). The `[B][V]` f32
    /// scratch is the only sizeable allocation (~1 MiB per slot per 256k vocab).
    fn sampler_bringup(
        be: &Arc<CudaBackend>,
        assets_dir: &Path,
        batch: usize,
        vocab: usize,
    ) -> Result<Option<Sampler>> {
        // DEFAULT ON (`PLOW_DEV_SAMPLE=0` opts out). Host sampling costs a
        // per-token round trip that dominates the step at batch: measured on
        // Gemma-4-12B/RTX 5090 at B=16, 83.6 ms of host gap against a 41.0 ms
        // kernel. Device sampling + `PLOW_MULTISTEP` together took that blob
        // 102.74 -> 185.60 tok/s (1.74x). Every failure path below already
        // degrades to host sampling, so defaulting on cannot break a bundle
        // that lacks the cubin. See perf-data/gemma4-12b-sm120-serving.md.
        let explicit = std::env::var("PLOW_DEV_SAMPLE").ok();
        if explicit.as_deref() == Some("0") {
            return Ok(None);
        }
        let cubin = std::env::var("PLOW_NV_CUBIN_SAMPLE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| assets_dir.join("sample_sm120.cubin"));
        if !cubin.is_file() {
            // Only a warning when the operator ASKED for it; on the default
            // path a bundle without the cubin is an ordinary host-sampling
            // deployment, not a misconfiguration.
            if explicit.is_some() {
                tracing::warn!(cubin = %cubin.display(), "PLOW_DEV_SAMPLE set but no sampler cubin");
            } else {
                tracing::debug!(cubin = %cubin.display(), "no sampler cubin — host sampling");
            }
            return Ok(None);
        }
        let image = std::fs::read(&cubin)
            .map_err(|source| RuntimeError::Io { path: cubin.clone(), source })?;
        let module = be.module_load(&image)?;
        let kname = std::env::var("PLOW_NV_KERNEL_SAMPLE")
            .unwrap_or_else(|_| "plow_sample".into());
        let f = be.get_function(&module, &kname)?;
        let params = be.host_alloc_pinned(5 * batch * 4)?;
        let d_params = be.alloc(0, (5 * batch * 4) as u64)?;
        let d_escratch = be.alloc(0, (batch * vocab * 4) as u64)?;
        tracing::info!(cubin = %cubin.display(), "device sampler enabled (PLOW_DEV_SAMPLE=1)");
        Ok(Some(Sampler {
            f,
            _module: module,
            params,
            d_params,
            d_escratch,
            batch,
        }))
    }

    /// Bring up bounded device multi-step (`PLOW_MULTISTEP=K`, K in [2,64];
    /// plan stage 5). Loads the `plow_advance` kernel from the sampler cubin
    /// and allocates the `[batch][K]` token ring + `[batch]` active-flag
    /// buffers. `Ok(None)` when the flag is unset, K is out of range, the
    /// cubin is absent, or the decode cubin is not dynamic-kvrow (multi-step
    /// needs device-owned pos).
    fn multistep_bringup(
        be: &Arc<CudaBackend>,
        assets_dir: &Path,
        batch: usize,
        dyn_kvrow: bool,
    ) -> Result<Option<MultiStep>> {
        // DEFAULT ON at K=8 (`PLOW_MULTISTEP=0` or `=1` opts out). K=8 captures
        // nearly all of the win — measured 179.18 tok/s vs 185.60 at K=32, i.e.
        // the last 3.6% costs 4x the quantum. The quantum is also how far ahead
        // of the client the device runs, so a small K keeps streaming delivery
        // fine-grained and bounds work generated past a stop token; that is why
        // the default is not the throughput-optimal 32.
        const MULTISTEP_DEFAULT: usize = 8;
        let raw = std::env::var("PLOW_MULTISTEP").ok();
        let k: usize = match raw.as_deref().map(|v| v.parse::<usize>()) {
            None => MULTISTEP_DEFAULT,
            Some(Ok(0)) | Some(Ok(1)) => return Ok(None),
            Some(Ok(k)) if (2..=64).contains(&k) => k,
            _ => {
                tracing::warn!("PLOW_MULTISTEP out of range [2,64] — multi-step off");
                return Ok(None);
            }
        };
        if !dyn_kvrow {
            // Expected on a B=1 legacy cubin that host-patches the KV row, so
            // this is only noteworthy when the operator asked for multi-step.
            if raw.is_some() {
                tracing::warn!("PLOW_MULTISTEP ignored: decode cubin is not dynamic-kvrow (needs device pos)");
            }
            return Ok(None);
        }
        let cubin = std::env::var("PLOW_NV_CUBIN_SAMPLE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| assets_dir.join("sample_sm120.cubin"));
        if !cubin.is_file() {
            if raw.is_some() {
                tracing::warn!(cubin = %cubin.display(), "PLOW_MULTISTEP set but no sampler cubin (has plow_advance)");
            } else {
                tracing::debug!(cubin = %cubin.display(), "no sampler cubin — multi-step off");
            }
            return Ok(None);
        }
        let image = std::fs::read(&cubin)
            .map_err(|source| RuntimeError::Io { path: cubin.clone(), source })?;
        let module = be.module_load(&image)?;
        let f_advance = be.get_function(&module, "plow_advance")?;
        let d_ring = be.alloc(0, (batch * k * 4) as u64)?;
        let ring_host = be.host_alloc_pinned(batch * k * 4)?;
        let d_fed = be.alloc(0, (batch * 4) as u64)?;
        let fed_host = be.host_alloc_pinned(batch * 4)?;
        tracing::info!(quantum = k, "bounded device multi-step enabled (PLOW_MULTISTEP)");
        Ok(Some(MultiStep {
            f_advance,
            _module: module,
            quantum: k,
            d_ring,
            ring_host,
            d_fed,
            fed_host,
            batch,
        }))
    }

    /// D2D-copy the sliding rings' last `window` rows at boundary `p_a`
    /// between slot `b`'s rings and a snapshot buffer (`to_snap` picks the
    /// direction). Layout: `[slide layer][K,V][head][window row][hd]`, rows
    /// ordered by absolute position `p_a-window..p_a`; each head is at most
    /// two runs (ring wrap).
    fn vmm_slide_copy(&self, b: usize, p_a: u32, buf: u64, to_snap: bool) -> Result<()> {
        let v = self.vmm.as_ref().expect("vmm_slide_copy without vmm");
        let g = v.kv.geometry();
        let w = g.window as u64;
        let hd_b = (g.hd_slide * g.elem_slide) as u64;
        let ring = v.ring;
        debug_assert!(p_a as u64 >= w, "boundary below the sliding window");
        let mut off = buf;
        for &(ik, iv, stride) in &v.slide {
            for idx in [ik, iv] {
                let base = self.devp[idx].base + b as u64 * stride;
                for h in 0..g.kvh_slide as u64 {
                    let hb = base + h * ring * hd_b;
                    let start = (p_a as u64 - w) & (ring - 1);
                    let run1 = w.min(ring - start);
                    let (r1, s1) = (hb + start * hd_b, off);
                    let (r2, s2) = (hb, off + run1 * hd_b);
                    if to_snap {
                        self.be.memcpy_dtod(s1, r1, run1 * hd_b)?;
                        if run1 < w {
                            self.be.memcpy_dtod(s2, r2, (w - run1) * hd_b)?;
                        }
                    } else {
                        self.be.memcpy_dtod(r1, s1, run1 * hd_b)?;
                        if run1 < w {
                            self.be.memcpy_dtod(r2, s2, (w - run1) * hd_b)?;
                        }
                    }
                    off += w * hd_b;
                }
            }
        }
        Ok(())
    }

    /// fp8-KV boundary-snapshot regions past the rings. Region 2 (fp8 rings
    /// only): the ring scales' last `window` rows, same wrap logic as the
    /// rings at 4 B/row. Region 3 (fp8 full layers only): each full layer's
    /// whole scale PREFIX `[0..p_a)` — full-layer scale tensors are flat
    /// cudaMalloc `[batch][kvh][max_ctx]` f32 and slot reuse overwrites them,
    /// so the shared prefix's scales can only survive in the snapshot.
    fn vmm_scale_copy(&self, b: usize, p_a: u32, buf: u64, to_snap: bool) -> Result<()> {
        let v = self.vmm.as_ref().expect("vmm_scale_copy without vmm");
        let g = v.kv.geometry();
        let w = g.window as u64;
        let ring = v.ring;
        let mut off = buf;
        let mut blit = |dev: u64, snap: u64, bytes: u64| -> Result<()> {
            if to_snap {
                self.be.memcpy_dtod(snap, dev, bytes)
            } else {
                self.be.memcpy_dtod(dev, snap, bytes)
            }
        };
        for &(sk, sv) in &v.slide_scale {
            for idx in [sk, sv] {
                let base = self.devp[idx].base + b as u64 * g.kvh_slide as u64 * ring * 4;
                for h in 0..g.kvh_slide as u64 {
                    let hb = base + h * ring * 4;
                    let start = (p_a as u64 - w) & (ring - 1);
                    let run1 = w.min(ring - start);
                    blit(hb + start * 4, off, run1 * 4)?;
                    if run1 < w {
                        blit(hb, off + run1 * 4, (w - run1) * 4)?;
                    }
                    off += w * 4;
                }
            }
        }
        let ctx = g.max_ctx as u64;
        for &(sk, sv) in &v.full_scale {
            for idx in [sk, sv] {
                let base = self.devp[idx].base + b as u64 * g.kvh_full as u64 * ctx * 4;
                for h in 0..g.kvh_full as u64 {
                    blit(base + h * ctx * 4, off, p_a as u64 * 4)?;
                    off += p_a as u64 * 4;
                }
            }
        }
        Ok(())
    }

    /// Bytes region 3 occupies for a boundary at `p_a` rows (0 unless the
    /// full layers are fp8).
    fn vmm_full_scale_bytes(&self, p_a: u32) -> u64 {
        let Some(v) = &self.vmm else { return 0 };
        let g = v.kv.geometry();
        v.full_scale.len() as u64 * 2 * g.kvh_full as u64 * p_a as u64 * 4
    }

    /// Copy the whole boundary snapshot for slot `b` at boundary `p_a`:
    /// rings, then fp8 ring scales, then fp8 full-layer scale prefixes.
    /// `to_snap` picks the direction (publish writes, attach restores).
    fn vmm_snap_copy(&self, b: usize, p_a: u32, buf: u64, to_snap: bool) -> Result<()> {
        self.vmm_slide_copy(b, p_a, buf, to_snap)?;
        let v = self.vmm.as_ref().expect("vmm_snap_copy without vmm");
        if !v.slide_scale.is_empty() || !v.full_scale.is_empty() {
            let g = v.kv.geometry();
            let rings = v.slide.len() as u64
                * 2
                * g.kvh_slide as u64
                * g.window as u64
                * (g.hd_slide * g.elem_slide) as u64;
            self.vmm_scale_copy(b, p_a, buf + rings, to_snap)?;
        }
        Ok(())
    }

    /// Consult the prefix cache for slot `b`'s prompt and attach a published
    /// prefix: multi-map the shared full-layer blocks, restore the sliding
    /// windows from the boundary snapshot, and advance the prefill frontier
    /// so the tail (< one sharing block) is recomputed by normal prefill.
    fn vmm_attach(&mut self, b: usize, prompt: &[u32]) -> Result<()> {
        let Some(v) = &self.vmm else { return Ok(()) };
        let Some(a) = v.kv.try_attach(b, prompt)? else {
            return Ok(());
        };
        debug_assert_eq!(
            a.snap_bytes,
            self.vmm.as_ref().unwrap().snap_bytes + self.vmm_full_scale_bytes(a.rows),
            "boundary snapshot layout drift"
        );
        self.vmm_snap_copy(b, a.rows, a.snap_va, false)?;
        self.pos[b] = a.rows;
        self.vmm_attached[b] = a.rows;
        // Seed the row-token record with the attached prefix — the tail is
        // appended by prefill completion (full prompt) or per decode feed.
        self.seq_tokens[b].clear();
        self.seq_tokens[b].extend_from_slice(&prompt[..a.rows as usize]);
        tracing::info!(
            slot = b,
            rows = a.rows,
            prompt = prompt.len(),
            "vmm: prefix attached (full-layer KV shared, zero copy)"
        );
        Ok(())
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Decode batch B — how many independent sequence slots this engine
    /// drives (the compiled `PLOW_DECODE_BATCH`).
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// VMM prefix-sharing counters; `None` when `PLOW_VMM_PREFIX` is off.
    pub fn vmm_stats(&self) -> Option<crate::memory::vmm::VmmStats> {
        self.vmm.as_ref().map(|v| v.kv.stats())
    }

    /// Engine-lock-free stats reader for `/metrics`; `None` when
    /// `PLOW_VMM_PREFIX` is off.
    pub fn vmm_stats_handle(&self) -> Option<crate::memory::vmm::VmmStatsHandle> {
        self.vmm.as_ref().map(|v| v.kv.stats_handle())
    }

    /// Rows slot `b`'s current sequence attached from the prefix cache
    /// (0 = cold start). Valid from the first prefill chunk on.
    pub fn attached_rows(&self, b: usize) -> u32 {
        self.vmm_attached.get(b).copied().unwrap_or(0)
    }

    /// Prefix-cache attach for the DECODE-ONLY prompt path (no prefill
    /// object): consult the cache once for a fresh slot and return the row
    /// the caller should resume feeding from — the attached rows' KV is
    /// already mapped, so only the tail goes through per-token decode steps.
    /// A no-op (returns 0) with VMM off or on a warm slot.
    pub fn attach_prompt(&mut self, b: usize, prompt: &[u32]) -> Result<usize> {
        if self.pos[b] == 0 && self.vmm.is_some() {
            self.vmm_attach(b, prompt)?;
        }
        Ok(self.pos[b] as usize)
    }

    /// Shared handle to the stop-token set (checkpoint `eos_token_id`).
    pub fn stop_ids(&self) -> &std::sync::Arc<Vec<u32>> {
        &self.stop_ids
    }

    /// Take the reusable f32 logits buffer (leaves an empty Vec in its place).
    /// Caller must return it via [`Self::return_logits_buf`] after use.
    pub fn take_logits_buf(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.logits_f32)
    }

    /// Return the reusable f32 logits buffer previously taken.
    pub fn return_logits_buf(&mut self, buf: Vec<f32>) {
        self.logits_f32 = buf;
    }

    /// Start a new sequence in slot `b`: `total` tokens (prompt + generation
    /// budget) must fit the compiled context. The KV cache needs no explicit
    /// reset — `in.kvlen` bounds what the attention reads, so rewinding
    /// `pos[b]` to 0 makes the slot's old cache rows unreachable.
    pub fn begin_slot(&mut self, b: usize, total: usize) -> Result<()> {
        if b >= self.batch {
            return Err(RuntimeError::Rejected(format!(
                "slot {b} out of range (engine batch {})",
                self.batch
            )));
        }
        if total > self.max_ctx {
            return Err(RuntimeError::Rejected(format!(
                "prompt + max_tokens = {total} exceeds the compiled context {}",
                self.max_ctx
            )));
        }
        // VMM: publish the finished sequence's generated whole blocks first —
        // a follow-up turn embedding this turn's output then attaches instead
        // of re-prefilling it — then drop the previous sequence's mappings/
        // cache references and re-map row 0 (idle-row garbage writes land
        // there).
        self.vmm_tail_publish(b);
        self.pos[b] = 0;
        self.vmm_attached[b] = 0;
        if let Some(v) = &self.vmm {
            self.seq_tokens[b].clear();
            v.kv.begin_seq(b);
            v.kv.ensure_rows(b, 1)?;
        }
        Ok(())
    }

    /// Publish slot `b`'s current sequence up to its last whole block —
    /// prompt AND generated rows — into the prefix cache. Skips (never
    /// fails serving) when the row-token record is inconsistent, the
    /// sequence is shorter than a block, or the sliding rings no longer
    /// hold the boundary's window rows (`rows - p_a > ring - window`:
    /// wrapped past, unrecoverable).
    fn vmm_tail_publish(&self, b: usize) {
        let Some(v) = &self.vmm else { return };
        let rows = self.pos[b];
        let toks = &self.seq_tokens[b];
        if rows == 0 || toks.len() != rows as usize {
            return;
        }
        let g = v.kv.geometry();
        let bt = v.kv.block_rows();
        let p_a = (rows / bt) * bt;
        if p_a < g.window.max(bt) {
            return;
        }
        if !v.slide.is_empty() && rows - p_a > v.ring as u32 - g.window {
            return;
        }
        let snap_bytes = v.snap_bytes + self.vmm_full_scale_bytes(p_a);
        if let Err(e) = v.kv.publish(b, toks, snap_bytes, |dst| {
            self.vmm_snap_copy(b, p_a, dst, true)
        }) {
            tracing::debug!(error = %e, slot = b, "vmm: tail publish skipped");
        }
    }

    /// One batched decode step: feed `(slot, token)` for every live slot,
    /// advance each fed slot's KV cache by one row, and write the
    /// device-argmax next token per feed (same order) into `toks` — cleared
    /// first; caller-provided so the per-token hot path allocates nothing.
    /// Port of the harness `run_step` (kv-row patch + ids/pos/kvlen + counter
    /// zero + cooperative launch + `in.ids` readback), generalized to B rows:
    /// every row `b` carries its own `pos[b]`/`kvlen[b]`; rows not fed this
    /// step get `kvlen = 1` and their own `pos[b]` (the row a garbage KV
    /// write lands in is exactly the row the slot's next real step
    /// overwrites, so idle rows never corrupt a live sequence).
    ///
    /// Submission is asynchronous on the engine stream: patch, uploads,
    /// counter re-arm, launch, and the token D2H are enqueued back-to-back
    /// and retired by ONE `cuStreamSynchronize` — no `cuCtxSynchronize`.
    pub fn step_slots(&mut self, feeds: &[(usize, u32)], toks: &mut Vec<u32>) -> Result<()> {
        self.step_slots_sampled(feeds, None, toks)
    }

    /// [`Self::step_slots`] with optional per-slot device sampling (plan
    /// stage 4). `specs`, if given, is indexed by slot (len == batch); a row
    /// with `temp > 0` is sampled on-device (the `plow_sample` kernel launches
    /// after the decode kernel and overwrites `in.ids[b]` with the sampled
    /// token, so `toks[b]` is the final token — no vocab-row D2H). `temp <= 0`
    /// rows keep the greedy `ARGMAX_FIN` token. `None`, an all-greedy `specs`,
    /// or no loaded sampler → the greedy path, byte-identical to before.
    pub fn step_slots_sampled(
        &mut self,
        feeds: &[(usize, u32)],
        specs: Option<&[DevSample]>,
        toks: &mut Vec<u32>,
    ) -> Result<()> {
        toks.clear();
        if feeds.is_empty() {
            return Ok(());
        }
        let bsz = self.batch;
        for &(b, _) in feeds {
            if b >= bsz {
                return Err(RuntimeError::Rejected(format!(
                    "slot {b} out of range (engine batch {bsz})"
                )));
            }
            if self.pos[b] as usize >= self.max_ctx {
                return Err(RuntimeError::Rejected(format!(
                    "context exhausted at {} (compiled max {})",
                    self.pos[b], self.max_ctx
                )));
            }
        }

        // VMM backstop: every row this launch writes (fed rows at pos, idle
        // rows' garbage write at their own pos) must be mapped. The
        // pre-mapper keeps this a lock-free frontier check in steady state.
        if let Some(v) = &self.vmm {
            for b in 0..bsz {
                let need = self.pos[b] + 1;
                if v.kv.mapped_rows(b) < need {
                    v.kv.ensure_rows(b, need)?;
                }
            }
        }

        let timed = self.timing.is_some();
        let now = || std::time::Instant::now();
        let t_enter = timed.then(now);
        if let Some(t) = &self.timing {
            self.be.event_record(&t.ev[0], &self.stream)?;
        }

        // B == 1 program: the KV write row is the host-patched `i[3]` (the
        // legacy single-ring formula). B > 1 programs ignore `i[3]` — the
        // kernel derives each row from `pos[b]` (`n_batch_kv > 1`).
        if bsz == 1 && !self.kvrow.is_empty() {
            let pos = self.pos[feeds[0].0];
            for &ix in &self.kvrow {
                self.h_inst[ix as usize].i[3] = pos;
            }
            let lo = self.kvrow_lo;
            let n = self.kvrow_hi - lo + 1;
            let sz = std::mem::size_of::<DevInst64>();
            // SAFETY: DevInst64 is a #[repr(C)] POD mirror; range within
            // h_inst, which lives on `self` past the stream synchronize.
            unsafe {
                let bytes =
                    std::slice::from_raw_parts(self.h_inst[lo..].as_ptr() as *const u8, n * sz);
                self.be
                    .memcpy_htod_async(self.d_inst.base + (lo * sz) as u64, bytes, &self.stream)?;
            }
        }

        // Pinned staging (sized once at load) — no per-step allocation, and
        // the copies below are truly asynchronous.
        {
            let (ids, pos, kvlen) = self.stage.parts_mut();
            for b in 0..bsz {
                ids[b] = 0;
                kvlen[b] = 1;
                pos[b] = self.pos[b] as i32;
            }
            for &(b, token) in feeds {
                ids[b] = token as i32;
                kvlen[b] = self.pos[b] as i32 + 1;
            }
        }
        // SAFETY: the pinned slab lives on `self` past the stream synchronize;
        // each section matches its [B]-sized i32 tensor.
        unsafe {
            self.be
                .memcpy_htod_async(self.devp[self.t_ids].base, self.stage.section(0), &self.stream)?;
            self.be
                .memcpy_htod_async(self.devp[self.t_pos].base, self.stage.section(1), &self.stream)?;
            self.be.memcpy_htod_async(
                self.devp[self.t_kvlen].base,
                self.stage.section(2),
                &self.stream,
            )?;
        }
        // Counters and the GQ cursor have the same lifecycle (one launch
        // consumes them) and share one allocation — one fill re-arms both.
        self.be.memset_d8_async(
            self.d_ctr.base,
            0,
            self.ctr_bytes.max(4) + CTR_STRIDE as usize * 4,
            &self.stream,
        )?;
        if let Some(t) = &self.timing {
            self.be.event_record(&t.ev[1], &self.stream)?;
        }

        let mut arg = self.kernarg;
        let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        self.be.launch_cooperative(
            self.f,
            self.grid,
            BLOCK,
            self.smem,
            &mut params,
            Some(&self.stream),
        )?;
        if let Some(t) = &self.timing {
            self.be.event_record(&t.ev[2], &self.stream)?;
        }

        // Device sampling (plan stage 4): if any fed row wants stochastic
        // sampling and the sampler is loaded, launch `plow_sample` on the
        // stream after the decode kernel. It reads act.logits (written by the
        // decode program; PLOW_FUSE_ARGMAX must be off, the default) and
        // OVERWRITES in.ids[b] for temp>0 rows; greedy rows keep the
        // ARGMAX_FIN token (the sampler's temp<=0 branch reproduces it).
        let want_sample =
            specs.is_some_and(|s| self.sampler.is_some() && s.iter().any(|x| x.temp > 0.0));
        if want_sample {
            let specs = specs.expect("checked");
            debug_assert_eq!(specs.len(), bsz, "specs must be batch-wide");
            // Fill the pinned param slab [temp|topk|topp|minp|rng], each B.
            let smp = self.sampler.as_mut().expect("checked");
            {
                let raw = smp.params.as_mut_slice();
                let (s_temp, r) = raw.split_at_mut(bsz * 4);
                let (s_topk, r) = r.split_at_mut(bsz * 4);
                let (s_topp, r) = r.split_at_mut(bsz * 4);
                let (s_minp, s_rng) = r.split_at_mut(bsz * 4);
                let temp: &mut [f32] = bytemuck::cast_slice_mut(s_temp);
                let topk: &mut [i32] = bytemuck::cast_slice_mut(s_topk);
                let topp: &mut [f32] = bytemuck::cast_slice_mut(s_topp);
                let minp: &mut [f32] = bytemuck::cast_slice_mut(s_minp);
                let rng: &mut [f32] = bytemuck::cast_slice_mut(s_rng);
                for b in 0..bsz {
                    temp[b] = specs[b].temp;
                    topk[b] = specs[b].top_k;
                    topp[b] = specs[b].top_p;
                    minp[b] = specs[b].min_p;
                    rng[b] = specs[b].rng01;
                }
            }
            let (sf, sdp, ses) = (smp.f, smp.d_params.base, smp.d_escratch.base);
            // SAFETY: pinned slab lives on self past the synchronize.
            unsafe {
                self.be
                    .memcpy_htod_async(sdp, smp.params.as_slice(), &self.stream)?;
            }
            let dp = |k: u64| sdp + k * (bsz * 4) as u64;
            let mut a_logits = self.devp[self.t_logits].base;
            let mut a_ids = self.devp[self.t_ids].base;
            let (mut a_temp, mut a_topk, mut a_topp) = (dp(0), dp(1), dp(2));
            let (mut a_minp, mut a_rng, mut a_es) = (dp(3), dp(4), ses);
            let (mut a_v, mut a_b) = (self.vocab as u32, bsz as u32);
            let mut a = [
                &mut a_logits as *mut u64 as *mut std::ffi::c_void,
                &mut a_ids as *mut u64 as *mut std::ffi::c_void,
                &mut a_temp as *mut u64 as *mut std::ffi::c_void,
                &mut a_topk as *mut u64 as *mut std::ffi::c_void,
                &mut a_topp as *mut u64 as *mut std::ffi::c_void,
                &mut a_minp as *mut u64 as *mut std::ffi::c_void,
                &mut a_rng as *mut u64 as *mut std::ffi::c_void,
                &mut a_es as *mut u64 as *mut std::ffi::c_void,
                &mut a_v as *mut u32 as *mut std::ffi::c_void,
                &mut a_b as *mut u32 as *mut std::ffi::c_void,
            ];
            self.be
                .launch_kernel(sf, bsz as u32, 256, 0, &mut a, Some(&self.stream))?;
        }

        // Token readback: `in.ids` (rewritten by `ARGMAX_FIN`, or by the
        // device sampler above) round-trips through the pinned ids section.
        // SAFETY: as the uploads — the slab outlives the synchronize.
        unsafe {
            let base = self.devp[self.t_ids].base;
            self.be
                .memcpy_dtoh_async(self.stage.section_mut(0), base, &self.stream)?;
        }
        if let Some(t) = &self.timing {
            self.be.event_record(&t.ev[3], &self.stream)?;
        }
        let t_submit = timed.then(now);

        // The step's single sync point (the whole queue retires in order).
        self.be.stream_synchronize(&self.stream)?;
        let t_sync = timed.then(now);

        toks.reserve(feeds.len());
        for &(b, tok) in feeds {
            toks.push(self.stage.token(b));
            self.pos[b] += 1;
            self.seq_tokens[b].push(tok);
        }
        // VMM: hint the pre-mapper so the next block is mapped before the
        // frontier reaches it (map-during-decode is safe — probe [5]).
        if let Some(v) = &self.vmm {
            for &(b, _) in feeds {
                v.kv.advise(b, self.pos[b]);
            }
        }

        if let Some(t) = self.timing.as_mut() {
            let (e, s0, s1) = (
                t_enter.expect("timed"),
                t_submit.expect("timed"),
                t_sync.expect("timed"),
            );
            t.upload_ms += self.be.event_elapsed_ms(&t.ev[0], &t.ev[1])? as f64;
            t.kernel_ms += self.be.event_elapsed_ms(&t.ev[1], &t.ev[2])? as f64;
            t.download_ms += self.be.event_elapsed_ms(&t.ev[2], &t.ev[3])? as f64;
            if let Some(prev) = t.last_end {
                t.gap_ns += (e - prev).as_nanos() as u64;
            }
            t.submit_ns += (s0 - e).as_nanos() as u64;
            t.sync_ns += (s1 - s0).as_nanos() as u64;
            t.last_end = Some(std::time::Instant::now());
            t.steps += 1;
            t.log_every(128);
        }
        Ok(())
    }

    /// Bounded device multi-step decode (plan stage 5; `PLOW_MULTISTEP=K`).
    /// GREEDY only. Runs a K-token quantum for every fed row with ONE host
    /// sync: the host uploads ids/pos/kvlen ONCE, then enqueues
    /// `[memset → decode → advance] × K` on the stream — the `plow_advance`
    /// kernel advances each row's device-owned pos/kvlen and appends its token
    /// to the ring between launches, so no per-token host round trip happens.
    /// `out` is filled row-major (`feeds.len() × K`, fed row r → `out[r*K..]`)
    /// and `K` is returned. Token-identical to K [`Self::step_slots`] calls.
    /// Requires the multi-step bringup (`has_multistep`); errors otherwise so
    /// the caller falls back to per-token stepping.
    pub fn multi_step(&mut self, feeds: &[(usize, u32)], out: &mut Vec<u32>) -> Result<usize> {
        out.clear();
        let Some(ms) = self.multistep.as_ref() else {
            return Err(RuntimeError::Rejected("multi-step not enabled".into()));
        };
        let (k, f_adv) = (ms.quantum, ms.f_advance);
        if feeds.is_empty() {
            return Ok(k);
        }
        let bsz = self.batch;
        for &(b, _) in feeds {
            if b >= bsz {
                return Err(RuntimeError::Rejected(format!("slot {b} out of range")));
            }
            if self.pos[b] as usize + k > self.max_ctx {
                return Err(RuntimeError::Rejected(format!(
                    "multi-step quantum {k} from pos {} exceeds context {}",
                    self.pos[b], self.max_ctx
                )));
            }
        }
        // VMM: map every row this quantum will write (fed rows pos..pos+K).
        if let Some(v) = &self.vmm {
            for &(b, _) in feeds {
                let need = self.pos[b] + k as u32;
                if v.kv.mapped_rows(b) < need {
                    v.kv.ensure_rows(b, need)?;
                }
            }
        }

        // Active-row flags (advance only touches fed rows; idle pos must not drift).
        {
            let fed: &mut [i32] = bytemuck::cast_slice_mut(self.mst_fed_host_mut());
            fed[..bsz].fill(0);
            for &(b, _) in feeds {
                fed[b] = 1;
            }
        }
        // Step-0 run state (subsequent steps advance device-side).
        {
            let (ids, pos, kvlen) = self.stage.parts_mut();
            for b in 0..bsz {
                ids[b] = 0;
                kvlen[b] = 1;
                pos[b] = self.pos[b] as i32;
            }
            for &(b, token) in feeds {
                ids[b] = token as i32;
                kvlen[b] = self.pos[b] as i32 + 1;
            }
        }
        // Uploads (once). fed + ids/pos/kvlen.
        let (ring_base, fed_base) = {
            let ms = self.multistep.as_ref().expect("checked");
            (ms.d_ring.base, ms.d_fed.base)
        };
        // SAFETY: pinned slabs live on self past the synchronize; sections match
        // their [B]-sized i32 tensors.
        unsafe {
            let ms = self.multistep.as_ref().expect("checked");
            self.be.memcpy_htod_async(fed_base, ms.fed_host.as_slice(), &self.stream)?;
            self.be.memcpy_htod_async(self.devp[self.t_ids].base, self.stage.section(0), &self.stream)?;
            self.be.memcpy_htod_async(self.devp[self.t_pos].base, self.stage.section(1), &self.stream)?;
            self.be.memcpy_htod_async(self.devp[self.t_kvlen].base, self.stage.section(2), &self.stream)?;
        }

        // Enqueue [memset → decode → advance] × K on the stream — no sync.
        let advance_grid = (bsz as u32).div_ceil(256);
        for step in 0..k {
            self.be.memset_d8_async(
                self.d_ctr.base,
                0,
                self.ctr_bytes.max(4) + CTR_STRIDE as usize * 4,
                &self.stream,
            )?;
            let mut arg = self.kernarg;
            let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
            self.be.launch_cooperative(
                self.f,
                self.grid,
                BLOCK,
                self.smem,
                &mut params,
                Some(&self.stream),
            )?;
            let mut a_ids = self.devp[self.t_ids].base;
            let mut a_pos = self.devp[self.t_pos].base;
            let mut a_kvl = self.devp[self.t_kvlen].base;
            let mut a_ring = ring_base;
            let mut a_fed = fed_base;
            let (mut a_step, mut a_k, mut a_b) = (step as u32, k as u32, bsz as u32);
            let mut a = [
                &mut a_ids as *mut u64 as *mut std::ffi::c_void,
                &mut a_pos as *mut u64 as *mut std::ffi::c_void,
                &mut a_kvl as *mut u64 as *mut std::ffi::c_void,
                &mut a_ring as *mut u64 as *mut std::ffi::c_void,
                &mut a_fed as *mut u64 as *mut std::ffi::c_void,
                &mut a_step as *mut u32 as *mut std::ffi::c_void,
                &mut a_k as *mut u32 as *mut std::ffi::c_void,
                &mut a_b as *mut u32 as *mut std::ffi::c_void,
            ];
            self.be.launch_kernel(f_adv, advance_grid, 256, 0, &mut a, Some(&self.stream))?;
        }

        // One D2H of the whole ring, then the single sync.
        // SAFETY: ring_host lives on self past the synchronize.
        unsafe {
            let ms = self.multistep.as_mut().expect("checked");
            let ring_slice = ms.ring_host.as_mut_slice();
            self.be.memcpy_dtoh_async(ring_slice, ring_base, &self.stream)?;
        }
        self.be.stream_synchronize(&self.stream)?;

        // Extract each fed row's K tokens (row-major) and advance host pos.
        {
            let ms = self.multistep.as_ref().expect("checked");
            let ring: &[i32] = bytemuck::cast_slice(ms.ring_host.as_slice());
            out.reserve(feeds.len() * k);
            for &(b, _) in feeds {
                for step in 0..k {
                    out.push(ring[b * k + step] as u32);
                }
            }
        }
        for (ri, &(b, tok)) in feeds.iter().enumerate() {
            self.pos[b] += k as u32;
            // Rows written this quantum: the fed token, then the device's own
            // feed chain (the first k-1 produced tokens).
            self.seq_tokens[b].push(tok);
            self.seq_tokens[b].extend_from_slice(&out[ri * k..ri * k + (k - 1)]);
            if let Some(v) = &self.vmm {
                v.kv.advise(b, self.pos[b]);
            }
        }
        Ok(k)
    }

    /// Mutable pinned active-flag staging for [`Self::multi_step`].
    fn mst_fed_host_mut(&mut self) -> &mut [u8] {
        self.multistep
            .as_mut()
            .expect("multi-step enabled")
            .fed_host
            .as_mut_slice()
    }

    /// Whether bounded device multi-step is enabled, and its quantum K.
    pub fn multistep_quantum(&self) -> Option<usize> {
        self.multistep.as_ref().map(|m| m.quantum)
    }

    /// Whether the prefill object + bucket programs are loaded.
    pub fn has_prefill(&self) -> bool {
        self.f_pf.is_some() && !self.prefill.is_empty()
    }

    /// Whether the device stochastic sampler is loaded (`PLOW_DEV_SAMPLE=1` +
    /// a sampler cubin). When true the mux routes eligible `temperature>0`
    /// rows through [`Self::step_slots_sampled`] instead of the vocab-row D2H.
    pub fn dev_sample_enabled(&self) -> bool {
        self.sampler.is_some()
    }

    /// Whether PX-1 cross-request batched prefill is active (`PLOW_PF_BATCH=1`
    /// and every load-time requirement held). The mux then routes ALL prefill
    /// through [`Self::prefill_batched`] and takes each request's first token
    /// from a batched decode step of its last prompt token.
    pub fn pf_batch_enabled(&self) -> bool {
        self.pf_batch.is_some()
    }

    /// Largest prefill bucket's row count — the mux's per-launch token budget.
    pub fn pf_max_rows(&self) -> usize {
        self.prefill.last().map_or(0, |b| b.t as usize)
    }

    /// Pack budget for `avail` waiting prefill rows (PX-1 batched path), in
    /// rows. Delegates to [`Self::pick_prefill_bucket`] so the batched and
    /// serialized paths share ONE policy — the cost-aware pick that charges
    /// each launch its real weight-restream cost. It had its own copy of the
    /// minimize-padded-rows rule, which cascaded the same way: a 8190-row pack
    /// budgeted 4096 where `[8192]` is one launch with 2 rows of padding.
    ///
    /// Still keeps a cold 4.5k-row pack out of the ~45%-padded 8192 bucket:
    /// `[4096, tail]` stays cheaper than a single `[8192]` under the cost model.
    pub fn pf_pack_budget(&self, avail: usize) -> usize {
        if self.prefill.is_empty() {
            return 0;
        }
        self.prefill[self.pick_prefill_bucket(avail, usize::MAX)].t as usize
    }

    /// Bring up the `_pf` prefill object and upload every non-decode (all but
    /// the last) bucket program. Port of the harness `prep_prog` for the prefill objects:
    /// upload tables, verify the single coarse segment, and precompute the
    /// per-chunk patch sites. Returns `Err` (prefill then disabled) if the cubin
    /// grid disagrees with `n_cu`, a bucket is segmented, or the GQ appendix is
    /// missing.
    fn load_prefill(
        be: &Arc<CudaBackend>,
        cubin: &Path,
        blob: &DevBlob,
        d_tens: u64,
        grid: u32,
        default_kernel: &str,
    ) -> Result<(KernelFn, u32, Module, Vec<PrefillBucket>)> {
        let image = std::fs::read(cubin)
            .map_err(|source| RuntimeError::Io { path: cubin.to_path_buf(), source })?;
        let module = be.module_load(&image)?;
        let kname = std::env::var("PLOW_NV_KERNEL_PF").unwrap_or_else(|_| default_kernel.into());
        let f_pf = be.get_function(&module, &kname)?;

        // Same contract as decode: env override > the cubin's own
        // `plow_arena_bytes_pf` metadata > legacy default.
        let smem_pf: u32 =
            match std::env::var("PLOW_NV_SMEM_PF").ok().and_then(|s| s.parse().ok()) {
                Some(v) => v,
                None => be
                    .module_global_u32(&module, "plow_arena_bytes_pf")?
                    .unwrap_or(SMEM_PF),
            };
        be.set_max_dynamic_smem(f_pf, smem_pf)?;

        // Same fatal grid gate as decode: the prefill kernel's occupancy must
        // yield exactly n_cu blocks or the cooperative launch reads off the
        // per-block stream tables / deadlocks.
        let occ = be.occupancy_blocks_per_sm(f_pf, BLOCK, smem_pf as usize)?;
        let grid_pf = occ * be.sm_count();
        if grid_pf != grid {
            return Err(RuntimeError::Device(format!(
                "prefill grid {grid_pf} ({occ}/SM × {} SMs) != decode grid {grid} (n_cu {}) — \
                 the two objects must share the cooperative grid",
                be.sm_count(),
                blob.n_cu
            )));
        }

        let upload_pod = |bytes: &[u8]| -> Result<DeviceMem> {
            let mem = be.alloc(0, bytes.len().max(4) as u64)?;
            if !bytes.is_empty() {
                be.upload(&mem, 0, bytes)?;
            }
            Ok(mem)
        };
        fn pod_bytes<T: Copy>(v: &[T]) -> &[u8] {
            // SAFETY: #[repr(C)] POD mirrors, read as raw bytes for upload.
            unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
            }
        }

        let mut buckets = Vec::new();
        // Every program but the last (the decode program, t == B) is a
        // prefill bucket.
        for g in &blob.progs[..blob.progs.len().saturating_sub(1)] {
            g.check_coarse_single_segment()?;
            if g.gq_stream.is_empty() || g.gq_seg_ofs.len() != 2 {
                return Err(RuntimeError::Device(format!(
                    "prefill bucket T={} has no single-segment GQ appendix (n_seg bounds: {:?}) — \
                     recompile with `PLOW_UNISEG=1 plowc` (GQ-capable, single segment)",
                    g.t, g.gq_seg_ofs
                )));
            }
            g.check_gq_topological()?;

            let d_inst = upload_pod(pod_bytes(&g.insts))?;
            let d_stream = upload_pod(pod_bytes(&g.stream))?;
            let d_sofs = upload_pod(pod_bytes(&g.stream_ofs))?;
            let d_slen = upload_pod(pod_bytes(&g.stream_len))?;
            let d_waits = upload_pod(pod_bytes(&g.waits))?;
            let d_succs = upload_pod(pod_bytes(&g.succs))?;
            let d_gq_stream = upload_pod(pod_bytes(&g.gq_stream))?;
            let d_gq_seg = upload_pod(pod_bytes(&g.gq_seg_ofs))?;
            // Combined counter/cursor slab (plan: counter improvements #1):
            // the bucket's own GQ cursor sits at its counter block's tail, so
            // ONE fill re-arms both and no cursor is shared with the decode
            // program or other buckets. `ctr_bytes` is the full reset size.
            let ctr_only = g.n_counter as usize * CTR_STRIDE as usize * 4;
            let cursor_off = ctr_only.max(4);
            let ctr_bytes = cursor_off + CTR_STRIDE as usize * 4;
            let d_ctr = be.alloc(0, ctr_bytes as u64)?;

            let kernarg = DevProgram {
                insts: d_inst.base,
                stream: d_stream.base,
                stream_ofs: d_sofs.base,
                stream_len: d_slen.base,
                waits: d_waits.base,
                succs: d_succs.base,
                counters: d_ctr.base,
                tensors: d_tens,
                trace: 0,
                cur_seg: 0,
                _segpad: 0,
                gq_stream: d_gq_stream.base,
                gq_seg_ofs: d_gq_seg.base,
                gq_cursor: d_ctr.base + cursor_off as u64,
                xctr: 0,
                peer_scratch: 0,
                rank: 0,
                n_gpu: 1,
            };

            // Precompute the per-chunk patch sites (harness inner loop): KV-write
            // HeadNormRope (j[0]!=0), FlashPrefill, the M==1 lm_head GEMM, and
            // (PX-1 batched mode) the FlashMerge sites to neuter.
            let (mut rope, mut flash, mut lmhead, mut merge) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut lo, mut hi) = (g.insts.len().saturating_sub(1), 0usize);
            let mut mark = |ix: usize| {
                lo = lo.min(ix);
                hi = hi.max(ix);
            };
            // fp8-KV packets emit the Fp8 TWINS of these opcodes. They carry the
            // SAME operands at the same indices (rope: i[3]=out_row0,
            // fj[1]=out_stride; flash: i[1]=seq_kv, i[4]=q_pos0 — see the
            // HEADNORM_ROPE_FP8 / FLASH_PREFILL_FP8 arms in interp_sm120.cu), so
            // they need the IDENTICAL per-chunk patch. Matching only the bf16
            // opcodes left every fp8 chunk after the first writing its KV at
            // row 0 and reading with q_pos0=0 — a silent wrong answer for any
            // prompt long enough to need a second prefill launch (measured: a
            // single-launch prompt is correct, 8192+2048 is not), and a prefill
            // that also LOOKS 1.75x faster because the flash never grows past
            // the first chunk's keys (PX-17 measured exactly that on main).
            let mut fp8_kv = false;
            for (ix, inst) in g.insts.iter().enumerate() {
                if (inst.op == DevOp::HeadNormRope as u16
                    || inst.op == DevOp::HeadNormRopeFp8 as u16)
                    && inst.fj[1] != 0
                {
                    fp8_kv |= inst.op == DevOp::HeadNormRopeFp8 as u16;
                    rope.push(ix);
                    mark(ix);
                } else if inst.op == DevOp::FlashPrefill as u16
                    || inst.op == DevOp::FlashPrefillFp8 as u16
                {
                    fp8_kv |= inst.op == DevOp::FlashPrefillFp8 as u16;
                    flash.push(ix);
                    mark(ix);
                } else if inst.op == DevOp::FlashMerge as u16 {
                    merge.push(ix);
                    mark(ix);
                } else if (inst.op == DevOp::Gemm as u16
                    || inst.op == DevOp::GemmSmall as u16
                    || inst.op == DevOp::GemmMed as u16)
                    && inst.i[0] == 1
                {
                    lmhead.push(ix);
                    mark(ix);
                }
            }

            buckets.push(PrefillBucket {
                t: g.t,
                kernarg,
                d_inst,
                h_inst: g.insts.clone(),
                inst_lo: lo,
                inst_hi: hi,
                rope_sites: rope,
                flash_sites: flash,
                lmhead_sites: lmhead,
                merge_sites: merge,
                fp8_kv,
                batch_patched: false,
                d_ctr,
                ctr_bytes,
                _tables: vec![
                    d_stream, d_sofs, d_slen, d_waits, d_succs, d_gq_stream, d_gq_seg,
                ],
            });
        }
        if buckets.is_empty() {
            return Err(RuntimeError::Device(
                "blob has no prefill buckets (only the T=1 decode program)".into(),
            ));
        }
        buckets.sort_by_key(|b| b.t);
        Ok((f_pf, smem_pf, module, buckets))
    }

    /// Consume the whole prompt for slot `b` through the prefill bucket chain
    /// (chunked per the largest bucket), leaving slot `b`'s KV cache built to
    /// `prompt.len()` rows and `in.ids[0]` holding the first generated token —
    /// the exact postcondition of the decode-only consumption loop, so
    /// `step_slots()` continues from here unchanged. Port of
    /// `gemma4_sm120_chat.cu`'s `PLOW_PREFILL=1` path.
    ///
    /// The prefill programs address the KV cache slot-relative (the legacy
    /// single-ring headnorm/flash formulas), so for `b > 0` the `kv.*` entries
    /// of the tensor table are repointed at slot `b`'s ring for the duration
    /// of the chunk chain, then restored to the batch-major bases the decode
    /// program expects. Serialized per slot — the engine mutex already makes
    /// prefill and decode mutually exclusive.
    ///
    /// Requires `begin_slot()` to have validated the sequence budget first.
    pub fn prefill_slot(&mut self, b: usize, prompt: &[u32]) -> Result<u32> {
        loop {
            match self.prefill_chunk(b, prompt, usize::MAX)? {
                PrefillStep::Done(tok) => return Ok(tok),
                PrefillStep::Progress(_) => {}
            }
        }
    }

    /// Advance slot `b`'s prefill by ONE chunk (bucket capped at `cap` rows —
    /// the serve-layer interleave bound; `usize::MAX` = uncapped). The frontier
    /// lives in `pos[b]` (`begin_slot` resets it to 0), which also keeps the
    /// batched-decode unfed-row garbage KV write landing in exactly the row
    /// the next chunk overwrites — so decode ticks may interleave between
    /// chunks without corrupting the partially-built cache. On the final chunk
    /// the first generated token is read back (`PrefillStep::Done`), the exact
    /// postcondition of the whole-prompt [`Self::prefill_slot`].
    pub fn prefill_chunk(&mut self, b: usize, prompt: &[u32], cap: usize) -> Result<PrefillStep> {
        let Some(f_pf) = self.f_pf else {
            return Err(RuntimeError::Rejected("prefill object not loaded".into()));
        };
        if self.prefill.is_empty() {
            return Err(RuntimeError::Rejected("no prefill buckets".into()));
        }
        // Batched mode owns the bucket instruction streams (t5/t6/merge are
        // patched to the batched contract) — the serialized chunk path must
        // never launch against them.
        if self.pf_batch.is_some() {
            return Err(RuntimeError::Rejected(
                "PLOW_PF_BATCH=1: serialized prefill_chunk disabled — use prefill_batched".into(),
            ));
        }
        if b >= self.batch {
            return Err(RuntimeError::Rejected(format!(
                "slot {b} out of range (engine batch {})",
                self.batch
            )));
        }
        let n = prompt.len();
        if n == 0 {
            return Err(RuntimeError::Rejected("empty prompt".into()));
        }
        if n > self.max_ctx {
            return Err(RuntimeError::Rejected(format!(
                "prompt {n} exceeds the compiled context {}",
                self.max_ctx
            )));
        }
        // VMM: first chunk of a fresh sequence consults the prefix cache —
        // a hit shares the whole-block prefix and moves the frontier there.
        if self.pos[b] == 0 && self.vmm.is_some() {
            self.vmm_attach(b, prompt)?;
        }
        let c0 = self.pos[b] as usize;
        debug_assert!(c0 < n, "prefill_chunk past the prompt end");

        // The chunk launches against slot b's prebuilt tensor table (the
        // kernarg selects it) — the shared table is never touched, so decode
        // and other slots' sequences cannot observe a shifted table.
        let real = self.run_one_prefill_chunk(f_pf, b, prompt, c0, cap)?;
        self.pos[b] = (c0 + real) as u32;

        if c0 + real < n {
            return Ok(PrefillStep::Progress(c0 + real));
        }
        // VMM: prefill finished — publish the prompt's whole sharing blocks
        // and the boundary's sliding-window snapshot into the prefix cache.
        // The consumed prompt also seeds the slot's row-token record so the
        // sequence's GENERATED blocks can publish at the next begin_slot.
        if self.vmm.is_some() {
            self.seq_tokens[b].clear();
            self.seq_tokens[b].extend_from_slice(prompt);
        }
        if let Some(v) = &self.vmm {
            let bt = v.kv.block_rows();
            let p_a = (n as u32 / bt) * bt;
            if p_a >= v.kv.geometry().window.max(bt) {
                let snap_bytes = v.snap_bytes + self.vmm_full_scale_bytes(p_a);
                if let Err(e) = v.kv.publish(b, prompt, snap_bytes, |dst| {
                    self.vmm_snap_copy(b, p_a, dst, true)
                }) {
                    tracing::warn!(error = %e, slot = b, "vmm: publish failed (serving continues)");
                }
            }
        }
        // in.ids[0] now holds the first generated token (argmax over the last
        // real prompt row) — identical to decode-only step n_prompt-1.
        let mut out = [0u8; 4];
        self.be.download(&self.devp[self.t_ids], 0, &mut out)?;
        Ok(PrefillStep::Done(i32::from_le_bytes(out) as u32))
    }

    /// Pick the prefill bucket for `rem` remaining rows under `cap`.
    ///
    /// Default policy minimizes `padded_rows + CHUNK_COST × launches` (see
    /// [`pf_chunk_cost_rows`]). A launch is NOT free — it re-streams every
    /// layer's weights — so minimizing padded rows alone is the wrong
    /// objective: it cascades a just-under-rung tail down the ladder, e.g.
    /// 8190 rows ran `[4096, 2048, 1024, 512, 128, 128, 128, 128]` (8 launches)
    /// where the 2-row-padded `[8192]` cover is one. Measured on sm_120 /
    /// gemma-4-12B, that single case cost **+25% TTFT**, and an 8390-row prompt
    /// finished 317 ms SOONER than an 8190-row one. Charging each launch its
    /// real cost removes the whole staircase (ms/tok across the 8.2k–10.4k
    /// boundary band went 0.133–0.175 → 0.129–0.139).
    ///
    /// A 4137-row prompt runs `[4096, 128]` = 4224 padded rows; a 4500-row pack
    /// still takes `[4096, …]` rather than the ~45%-padded single `[8192]`.
    /// `PLOW_PF_COVER=1` restores the covering pick (A/B control / exact parity
    /// with harness trajectories, which chunk the covering way);
    /// `PLOW_PF_CHUNK_COST=0` recovers the pure-minimum-padding objective.
    fn pick_prefill_bucket(&self, rem: usize, cap: usize) -> usize {
        let allowed = |t: usize| t <= cap;
        // Fall back to the smallest bucket when the cap is under every rung.
        let smallest = 0usize; // buckets are sorted by t
        if pf_cover_on() {
            // Old policy: smallest allowed bucket >= rem, else largest allowed.
            let mut pick = smallest;
            for (i, bkt) in self.prefill.iter().enumerate() {
                let t = bkt.t as usize;
                if i > 0 && !allowed(t) {
                    break;
                }
                pick = i;
                if t >= rem {
                    break;
                }
            }
            return pick;
        }
        // Cost-aware pick: minimize `padded_rows + CHUNK_COST × launches`.
        //
        // Minimizing padded rows ALONE cascades a just-under-rung tail into a
        // pile of tiny launches — 8190 rows ran [4096,2048,1024,512,128×4], 8
        // launches, measured 28% SLOWER than an 8390-row prompt's [8192,128,128]
        // (1434 ms vs 1117 ms: 200 MORE tokens finished 317 ms sooner). Charging
        // each launch `pf_chunk_cost_rows()` makes the 2-row-padded 8192 cover
        // win, which is what the hardware actually prefers.
        let n_allowed = self
            .prefill
            .iter()
            .take_while(|b| allowed(b.t as usize))
            .count();
        if n_allowed == 0 {
            return smallest;
        }
        // While the largest allowed rung still FILLS, it is optimal outright:
        // minimal padding and minimal launches at the same time.
        let top = n_allowed - 1;
        if rem >= self.prefill[top].t as usize {
            return top;
        }
        // Tail. Every rung is a multiple of the smallest, so quantizing the
        // state on it bounds the table at `top_rung / smallest_rung` entries
        // (64 for the shipped 128…8192 ladder).
        let unit = (self.prefill[smallest].t as usize).max(1);
        let chunk_cost = pf_chunk_cost_rows();
        let goal = rem.div_ceil(unit);
        // best[s] = (cost of consuming s units, first rung of that plan)
        let mut best = vec![(usize::MAX, smallest); goal + 1];
        best[0] = (0, smallest);
        for s in 1..=goal {
            let left = s * unit;
            for (i, bkt) in self.prefill.iter().take(n_allowed).enumerate() {
                let t = bkt.t as usize;
                // `t >= unit` ⇒ `next < s`, so the table fills in one pass.
                let next = left.saturating_sub(t).div_ceil(unit);
                let (prev, _) = best[next];
                if prev == usize::MAX {
                    continue;
                }
                let cost = prev + t + chunk_cost;
                if cost < best[s].0 {
                    best[s] = (cost, i);
                }
            }
        }
        best[goal].1
    }

    /// Slot `b`'s immutable prefill tensor table (built once at load).
    fn tens_slot_base(&self, b: usize) -> u64 {
        if b == 0 {
            self.d_tens.base
        } else {
            self.d_tens_slots[b - 1].base
        }
    }

    /// One prefill chunk at frontier `c0` (the harness's inner-loop body) —
    /// bucket pick, instruction patch, ids/pos/kvlen upload, one cooperative
    /// launch. KV lands wherever the tensor table currently points
    /// (`bind_kv_slot`). Returns the number of real rows consumed.
    fn run_one_prefill_chunk(
        &mut self,
        f_pf: KernelFn,
        b: usize,
        prompt: &[u32],
        c0: usize,
        cap: usize,
    ) -> Result<usize> {
        let n = prompt.len();
        let rem = n - c0;
        let sz = std::mem::size_of::<DevInst64>();

        let bi = self.pick_prefill_bucket(rem, cap);
        let tc = self.prefill[bi].t as usize;
        let real = rem.min(tc);

        // VMM: the bucket writes all tc rows (pad rows write garbage past
        // `real`) — map the chunk's full row span before launching.
        if let Some(v) = &self.vmm {
            v.kv
                .ensure_rows(b, ((c0 + tc) as u32).min(self.max_ctx as u32))?;
        }

        // Patch this bucket's instruction stream for the chunk, then enqueue
        // the covering [lo..=hi] window as an async H2D on the engine stream.
        {
            let b = &mut self.prefill[bi];
            for &ix in &b.rope_sites {
                b.h_inst[ix].i[3] = c0 as u32;
            }
            for &ix in &b.flash_sites {
                b.h_inst[ix].i[1] = (c0 + real) as u32;
                b.h_inst[ix].i[4] = c0 as u32;
            }
            for &ix in &b.lmhead_sites {
                b.h_inst[ix].i[4] = (real - 1) as u32;
            }
            let lo = b.inst_lo;
            let cnt = b.inst_hi - lo + 1;
            // SAFETY: DevInst64 is a #[repr(C)] POD mirror; range within
            // h_inst which lives on self past the stream_synchronize below.
            unsafe {
                let bytes =
                    std::slice::from_raw_parts(b.h_inst[lo..].as_ptr() as *const u8, cnt * sz);
                self.be
                    .memcpy_htod_async(b.d_inst.base + (lo * sz) as u64, bytes, &self.stream)?;
            }
        }

        // ids (real tokens + zero pad) and absolute positions for the chunk.
        // Reuse pre-allocated buffers (sized to max prefill bucket t).
        self.pf_ids.resize(tc, 0);
        self.pf_pos.resize(tc, 0);
        for i in 0..tc {
            self.pf_ids[i] = if i < real { prompt[c0 + i] as i32 } else { 0 };
            self.pf_pos[i] = (c0 + i) as i32;
        }
        // SAFETY: pf_ids, pf_pos live on self past the stream_synchronize;
        // devp[t_ids/t_pos/t_kvlen] are live device ranges sized to hold the
        // uploaded data (tc ≤ max prefill bucket t ≤ tensor capacity).
        let kvlen_bytes = ((c0 + real) as i32).to_le_bytes();
        unsafe {
            self.be.memcpy_htod_async(
                self.devp[self.t_ids].base,
                bytemuck::cast_slice(&self.pf_ids[..tc]),
                &self.stream,
            )?;
            self.be.memcpy_htod_async(
                self.devp[self.t_pos].base,
                bytemuck::cast_slice(&self.pf_pos[..tc]),
                &self.stream,
            )?;
            self.be.memcpy_htod_async(
                self.devp[self.t_kvlen].base,
                &kvlen_bytes,
                &self.stream,
            )?;
        }

        let (ctr_base, ctr_bytes, mut arg) = {
            let bkt = &self.prefill[bi];
            (bkt.d_ctr.base, bkt.ctr_bytes, bkt.kernarg)
        };
        // Slot selection is a kernarg field: the launch reads slot b's
        // prebuilt tensor table (kv.* shifted to its ring); nothing uploads.
        arg.tensors = self.tens_slot_base(b);
        // One async fill re-arms the bucket's counters AND its tail GQ cursor.
        self.be.memset_d8_async(ctr_base, 0, ctr_bytes, &self.stream)?;

        // All uploads/memsets are enqueued on the engine stream — the launch
        // follows them in stream order (no context sync needed).
        let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        self.be.launch_cooperative(
            f_pf,
            self.grid,
            BLOCK,
            self.smem_pf,
            &mut params,
            Some(&self.stream),
        )?;
        self.be.stream_synchronize(&self.stream)?;

        Ok(real)
    }

    /// One-time PX-1 batched-mode patch of bucket `bi`'s instruction stream:
    /// KV-write `HeadNormRope` t6 = the per-row slot map; `FlashPrefill` t6 =
    /// the request table and (non-fused buckets) t5 = the fused `n.at` output;
    /// `FlashMerge` neutered (`i[0]=0` — the flash epilogue already normalized
    /// and wrote the bf16); lm_head `a_row0 = 0` (row 0 always holds a real
    /// row; the argmax result is unused — the first token comes from a batched
    /// decode step of the last prompt token). One covering-window upload.
    fn ensure_batch_patch(&mut self, bi: usize) -> Result<()> {
        if self.prefill[bi].batch_patched {
            return Ok(());
        }
        let (h_slot, h_req, at_sites) = {
            let pb = self.pf_batch.as_ref().expect("pf_batch checked by caller");
            (pb.h_slot, pb.h_req, pb.at_sites.clone())
        };
        let sz = std::mem::size_of::<DevInst64>();
        let b = &mut self.prefill[bi];
        if b.flash_sites.len() != at_sites.len() {
            return Err(RuntimeError::Device(format!(
                "pf-batch: bucket T={} has {} flash sites, fused bucket has {}",
                b.t,
                b.flash_sites.len(),
                at_sites.len()
            )));
        }
        for (k, &ix) in b.flash_sites.iter().enumerate() {
            let (t5, hd) = at_sites[k];
            if b.h_inst[ix].i[6] != hd {
                return Err(RuntimeError::Device(format!(
                    "pf-batch: bucket T={} flash site {k} hd {} != fused bucket's {hd}",
                    b.t, b.h_inst[ix].i[6]
                )));
            }
            if b.h_inst[ix].t[5] == TENSOR_NONE16 {
                b.h_inst[ix].t[5] = t5 as u16;
            }
            b.h_inst[ix].t[6] = h_req as u16;
        }
        for &ix in &b.rope_sites {
            b.h_inst[ix].t[6] = h_slot as u16;
        }
        for &ix in &b.merge_sites {
            b.h_inst[ix].i[0] = 0;
        }
        for &ix in &b.lmhead_sites {
            b.h_inst[ix].i[4] = 0;
        }
        let lo = b.inst_lo;
        let cnt = b.inst_hi - lo + 1;
        // SAFETY: DevInst64 is a #[repr(C)] POD mirror; range within h_inst.
        let bytes = unsafe {
            std::slice::from_raw_parts(b.h_inst[lo..].as_ptr() as *const u8, cnt * sz)
        };
        self.be.memcpy_htod(b.d_inst.base + (lo * sz) as u64, bytes)?;
        b.batch_patched = true;
        Ok(())
    }

    /// PX-1: run ONE batched prefill launch over the packed chunks of `reqs`
    /// (in order — request r's rows occupy the tile rows after request r-1's).
    /// The GEMMs see one M = Σ len matrix (weights read once, shared across
    /// requests); attention runs per-request-serial against each request's own
    /// seq-slot KV (block-diagonal by construction — see d_flash_prefill_mux);
    /// each row's KV write lands in its own slot's batch-major ring at its own
    /// absolute position. Advances `pos[slot]` per request. The mux packs to
    /// `n-1` prompt rows and takes the first generated token from a batched
    /// decode step of the last prompt token (`step_slots`), so no lm_head
    /// readback happens here.
    pub fn prefill_batched(&mut self, reqs: &[PfBatchReq]) -> Result<()> {
        let Some(f_pf) = self.f_pf else {
            return Err(RuntimeError::Rejected("prefill object not loaded".into()));
        };
        if self.pf_batch.is_none() {
            return Err(RuntimeError::Rejected("PLOW_PF_BATCH not enabled".into()));
        }
        if reqs.is_empty() {
            return Ok(());
        }
        let mut total = 0usize;
        for r in reqs {
            if r.slot >= self.batch {
                return Err(RuntimeError::Rejected(format!(
                    "slot {} out of range (engine batch {})",
                    r.slot, self.batch
                )));
            }
            if r.len == 0 || r.c0 + r.len > r.prompt.len() {
                return Err(RuntimeError::Rejected(format!(
                    "pf-batch: bad chunk c0={} len={} prompt={}",
                    r.c0,
                    r.len,
                    r.prompt.len()
                )));
            }
            if r.c0 != self.pos[r.slot] as usize {
                return Err(RuntimeError::Rejected(format!(
                    "pf-batch: slot {} frontier {} != chunk c0 {}",
                    r.slot, self.pos[r.slot], r.c0
                )));
            }
            if r.c0 + r.len > self.max_ctx {
                return Err(RuntimeError::Rejected(format!(
                    "pf-batch: chunk end {} exceeds compiled context {}",
                    r.c0 + r.len,
                    self.max_ctx
                )));
            }
            total += r.len;
        }
        // Covering bucket for the pack: smallest T >= Σ len (minimal padding).
        let bi = self
            .prefill
            .iter()
            .position(|b| b.t as usize >= total)
            .ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "pf-batch: pack of {total} rows exceeds the largest bucket"
                ))
            })?;
        self.ensure_batch_patch(bi)?;
        let tc = self.prefill[bi].t as usize;

        // Stage ids/pos/slot rows + the request table, then upload. `pf_batch`
        // is taken out for the duration so `self` stays borrowable.
        let mut pb = self.pf_batch.take().expect("checked Some");
        // Declared outside the staging closure so it outlives the closure and
        // stays valid until the stream_synchronize below — the closure enqueues
        // an async H2D of these bytes (memcpy_htod_async src-lifetime contract).
        let kvlen_bytes = reqs.iter().map(|r| (r.c0 + r.len) as i32).max().unwrap_or(0).to_le_bytes();
        let staged = (|| -> Result<()> {
            self.pf_ids.resize(tc, 0);
            self.pf_pos.resize(tc, 0);
            pb.slot_buf.resize(tc, 0);
            pb.req_buf.clear();
            pb.req_buf.push(reqs.len() as i32);
            let mut cur = 0usize;
            for r in reqs {
                for k in 0..r.len {
                    self.pf_ids[cur + k] = r.prompt[r.c0 + k] as i32;
                    self.pf_pos[cur + k] = (r.c0 + k) as i32;
                    pb.slot_buf[cur + k] = r.slot as i32;
                }
                pb.req_buf.extend_from_slice(&[
                    cur as i32,
                    r.len as i32,
                    r.slot as i32,
                    (r.c0 + r.len) as i32,
                ]);
                cur += r.len;
            }
            // Trailing pad rows continue the LAST request's positions (legacy
            // pad semantics: garbage KV past a frontier lands in rows that
            // slot's own next writes overwrite before they become readable).
            let last = reqs.last().expect("non-empty");
            let (mut p, s) = (last.c0 + last.len, last.slot as i32);
            for k in cur..tc {
                self.pf_ids[k] = 0;
                self.pf_pos[k] = p.min(self.max_ctx - 1) as i32;
                pb.slot_buf[k] = s;
                p += 1;
            }
            // SAFETY: pf_ids, pf_pos live on self past the stream_synchronize;
            // pb.slot_buf, pb.req_buf are put back into self.pf_batch (below)
            // before the sync; kvlen_bytes is declared above the closure so it
            // outlives the sync. All device destinations are live ranges.
            unsafe {
                self.be.memcpy_htod_async(
                    self.devp[self.t_ids].base,
                    bytemuck::cast_slice(&self.pf_ids[..tc]),
                    &self.stream,
                )?;
                self.be.memcpy_htod_async(
                    self.devp[self.t_pos].base,
                    bytemuck::cast_slice(&self.pf_pos[..tc]),
                    &self.stream,
                )?;
                self.be.memcpy_htod_async(
                    pb.d_slot.base,
                    bytemuck::cast_slice(&pb.slot_buf[..tc]),
                    &self.stream,
                )?;
                self.be.memcpy_htod_async(
                    pb.d_req.base,
                    bytemuck::cast_slice(&pb.req_buf),
                    &self.stream,
                )?;
                self.be.memcpy_htod_async(
                    self.devp[self.t_kvlen].base,
                    &kvlen_bytes,
                    &self.stream,
                )?;
            }
            Ok(())
        })();
        self.pf_batch = Some(pb);
        staged?;

        let (ctr_base, ctr_bytes, mut arg) = {
            let b = &self.prefill[bi];
            (b.d_ctr.base, b.ctr_bytes, b.kernarg)
        };
        // One async fill re-arms the bucket's counters AND its tail GQ cursor.
        self.be.memset_d8_async(ctr_base, 0, ctr_bytes, &self.stream)?;

        // All uploads/memsets are enqueued on the engine stream — the launch
        // follows them in stream order (no context sync needed).
        let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        self.be.launch_cooperative(
            f_pf,
            self.grid,
            BLOCK,
            self.smem_pf,
            &mut params,
            Some(&self.stream),
        )?;
        self.be.stream_synchronize(&self.stream)?;

        tracing::debug!(
            requests = reqs.len(),
            rows = total,
            bucket_t = tc,
            slots = ?reqs.iter().map(|r| r.slot).collect::<Vec<_>>(),
            "gpu: batched prefill launch"
        );
        if pf_packlog_on() {
            // One line per launch → parsed into R-per-launch histograms by the
            // RTX-12 packing bench. chunks = per-request packed rows (this tick).
            use std::fmt::Write as _;
            let mut chunks = String::new();
            for (i, r) in reqs.iter().enumerate() {
                let _ = write!(chunks, "{}{}", if i == 0 { "" } else { "," }, r.len);
            }
            eprintln!(
                "PACKLOG R={} rows={} bucket={} chunks=[{}]",
                reqs.len(),
                total,
                tc,
                chunks
            );
        }
        for r in reqs {
            self.pos[r.slot] = (r.c0 + r.len) as u32;
        }
        Ok(())
    }

    /// Download row `b` of the `[B][vocab]` logits (bf16 → f32) into `out`.
    /// Used when the request wants stochastic sampling; the greedy path
    /// consumes the device argmax without ever moving the row. Prefill writes
    /// its logits to row 0 regardless of the slot (lm_head M == 1).
    pub fn logits_row(&mut self, b: usize, out: &mut Vec<f32>) -> Result<()> {
        if b >= self.batch {
            return Err(RuntimeError::Rejected(format!(
                "slot {b} out of range (engine batch {})",
                self.batch
            )));
        }
        self.logits_raw.resize(self.vocab * 2, 0);
        self.be.download(
            &self.devp[self.t_logits],
            (b * self.vocab * 2) as u64,
            &mut self.logits_raw,
        )?;
        out.clear();
        out.resize(self.vocab, 0.0);
        bf16_to_f32_slice(&self.logits_raw, out);
        Ok(())
    }

    /// Handle (blob tensor index, = `devp` index) of the tensor named `name`,
    /// or `None`. Block-harness accessor; not a hot path.
    pub fn handle_of(&self, name: &str) -> Option<usize> {
        self.tensor_names.iter().position(|n| n == name)
    }

    /// Download an activation tensor by name as f32 (bf16 → f32, the same
    /// widening [`Self::logits_row`] does). Length = the tensor's byte size / 2
    /// (bf16). Used by the block harness to read `act.x` out. Not a hot path.
    pub fn download_activation(&self, name: &str) -> Result<Vec<f32>> {
        let i = self.handle_of(name).ok_or_else(|| {
            RuntimeError::Rejected(format!("no tensor named {name:?} in the blob"))
        })?;
        let bytes = self.devp[i].len as usize;
        let mut raw = vec![0u8; bytes];
        self.be.download(&self.devp[i], 0, &mut raw)?;
        let mut out = vec![0.0f32; bytes / 2];
        bf16_to_f32_slice(&raw, &mut out);
        Ok(out)
    }

    /// Upload an f32 slice into a bf16 activation tensor by name (f32 → bf16 by
    /// truncating the low 16 mantissa bits, `(bits >> 16) as u16`). Used by the
    /// block harness to feed `act.x` in. Not a hot path.
    pub fn upload_activation(&mut self, name: &str, data: &[f32]) -> Result<()> {
        let i = self.handle_of(name).ok_or_else(|| {
            RuntimeError::Rejected(format!("no tensor named {name:?} in the blob"))
        })?;
        let cap = self.devp[i].len as usize / 2;
        if data.len() > cap {
            return Err(RuntimeError::Rejected(format!(
                "upload_activation {name}: {} f32 > tensor capacity {cap} bf16",
                data.len()
            )));
        }
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &v in data {
            let h = (v.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        self.be.upload(&self.devp[i], 0, &bytes)?;
        Ok(())
    }

    /// Reset the PLOW_NV_TRACE packet counter so [`Self::trace_summary`]
    /// reports only launches after this call (drop prefill/warmup). No-op on
    /// a normal cubin.
    pub fn trace_reset(&self) -> Result<()> {
        self.be.module_global_zero(&self._module, "g_tr_n", 4)?;
        Ok(())
    }

    /// Stage-7 profiling (plan §Instrumentation): if the decode cubin was
    /// built `-DPLOW_NV_TRACE=1`, read back block 0's per-packet gate/body/
    /// signal cycle attribution (device globals `g_tr_*`) and aggregate it by
    /// opcode. `Ok(None)` on a normal cubin (no trace globals). The trace
    /// accumulates across launches, so this reports the whole run up to the
    /// 4096-entry cap — read the SHAPE (gate vs body vs signal per op), not
    /// the absolute total (clock64 in the recording thread over-reports). See
    /// the trace header in interp_sm120.cu.
    pub fn trace_summary(&self) -> Result<Option<String>> {
        let mut raw = Vec::new();
        if !self
            .be
            .module_global_bytes(&self._module, "g_tr_n", 4, &mut raw)?
        {
            return Ok(None);
        }
        let n = u32::from_le_bytes(raw[..4].try_into().expect("4B")) as usize;
        if n == 0 {
            return Ok(Some("trace: no packets recorded".into()));
        }
        let cap = n.min(4096);
        let read_u32 = |name: &str, out: &mut Vec<u32>| -> Result<()> {
            let mut b = Vec::new();
            self.be
                .module_global_bytes(&self._module, name, cap * 4, &mut b)?;
            *out = b
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().expect("4B")))
                .collect();
            Ok(())
        };
        let read_u64 = |name: &str, out: &mut Vec<u64>| -> Result<()> {
            let mut b = Vec::new();
            self.be
                .module_global_bytes(&self._module, name, cap * 8, &mut b)?;
            *out = b
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().expect("8B")))
                .collect();
            Ok(())
        };
        let (mut op, mut wait) = (Vec::new(), Vec::new());
        let (mut gate, mut body, mut sig) = (Vec::new(), Vec::new(), Vec::new());
        read_u32("g_tr_op", &mut op)?;
        read_u32("g_tr_wait", &mut wait)?;
        read_u64("g_tr_gate", &mut gate)?;
        read_u64("g_tr_body", &mut body)?;
        read_u64("g_tr_sig", &mut sig)?;

        // Per-opcode accumulation: (count, Σgate, Σbody, Σsig, Σwait_edges).
        let mut acc: rustc_hash::FxHashMap<u32, (u64, u64, u64, u64, u64)> =
            rustc_hash::FxHashMap::default();
        let (mut tg, mut tb, mut ts) = (0u64, 0u64, 0u64);
        for i in 0..cap {
            let e = acc.entry(op[i]).or_default();
            e.0 += 1;
            e.1 += gate[i];
            e.2 += body[i];
            e.3 += sig[i];
            e.4 += wait[i] as u64;
            tg += gate[i];
            tb += body[i];
            ts += sig[i];
        }
        let total = (tg + tb + ts).max(1) as f64;
        let mut rows: Vec<_> = acc.into_iter().collect();
        rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.1 + v.2 + v.3));
        let mut s = format!(
            "PLOW_NV_TRACE block-0 profile: {cap} packets | gate {:.1}% body {:.1}% \
             signal {:.1}% of {} kcyc\n  {:<16} {:>5} {:>10} {:>10} {:>10} {:>8}\n",
            100.0 * tg as f64 / total,
            100.0 * tb as f64 / total,
            100.0 * ts as f64 / total,
            (tg + tb + ts) / 1000,
            "op",
            "n",
            "gate_kcyc",
            "body_kcyc",
            "sig_kcyc",
            "waits",
        );
        for (op, (cnt, g, b, sg, w)) in rows {
            s += &format!(
                "  {:<16} {:>5} {:>10} {:>10} {:>10} {:>8}\n",
                devop_name(op),
                cnt,
                g / 1000,
                b / 1000,
                sg / 1000,
                w,
            );
        }
        Ok(Some(s))
    }
}

/// Convert a raw byte buffer of bf16 values into an f32 slice in-place.
/// `src` must be exactly `dst.len() * 2` bytes (one `u16` per `f32` output).
/// The inner loop is a trivial indexed pattern that auto-vectorises on x86-64
/// (LLVM emits `vpmovzxwd` + `vpslld` + `vmovups` for AVX2).
#[inline]
fn bf16_to_f32_slice(src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len() * 2);
    // Safety: &[u8] of even length → &[u16] with half the count.
    let halves: &[u16] =
        unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, dst.len()) };
    for i in 0..dst.len() {
        dst[i] = f32::from_bits((halves[i] as u32) << 16);
    }
}

/// Device opcode → short name for the trace profile (the `DevOp` values the
/// kernel records in `g_tr_op`). Unlisted opcodes print their number.
fn devop_name(op: u32) -> String {
    match op {
        1 => "RmsNorm",
        2 => "RowRms",
        3 => "HeadNormRope",
        4 => "Residual",
        5 => "Glu",
        6 => "Embed",
        7 => "SoftCap",
        8 => "Gemm",
        9 => "GemmNorm",
        10 => "Gemv",
        11 => "FlashPrefill",
        12 => "FlashDecode",
        13 => "FlashMerge",
        14 => "GemmSmall",
        15 => "GemmMed",
        16 => "NormResidual",
        17 => "Argmax",
        18 => "ArgmaxFin",
        19 => "GemvGlu",
        20 => "GemmGlu",
        21 => "AddNorm",
        22 => "GemvQkv",
        23 => "NormResidualNorm",
        30 => "GemvFp8",
        31 => "GemvGluFp8",
        32 => "QuantFp8",
        37 => "HeadNormRopeFp8",
        38 => "FlashDecodeFp8",
        44 => "GemvFp8Blk",
        80 => "GemvArgmax",
        _ => return op.to_string(),
    }
    .to_string()
}

impl Drop for GpuEngine {
    /// Model unload: quiesce the device and unload both cubin modules; every
    /// owned `DeviceMem` field (weights, KV rings, tables, counter block)
    /// then frees itself after this body, returning the engine's whole VRAM
    /// footprint to the driver. `d_ctr`/`d_gq_cursor` are views and free
    /// nothing — `_ctr_block` owns that storage. Errors are logged: Drop has
    /// no error channel, and a failed unload only pins module storage.
    fn drop(&mut self) {
        // Every public step/prefill path synchronizes before returning, but a
        // failed launch may leave work queued — never free under a running
        // cooperative kernel.
        if let Err(e) = self.be.synchronize() {
            tracing::warn!(error = %e, "synchronize at engine unload");
        }
        if let Some(m) = self._module_pf.take() {
            if let Err(e) = self.be.module_unload(&m) {
                tracing::warn!(error = %e, "unload prefill module");
            }
        }
        if let Some(s) = self.sampler.take() {
            if let Err(e) = self.be.module_unload(&s._module) {
                tracing::warn!(error = %e, "unload sampler module");
            }
        }
        if let Some(m) = self.multistep.take() {
            if let Err(e) = self.be.module_unload(&m._module) {
                tracing::warn!(error = %e, "unload multistep module");
            }
        }
        if let Err(e) = self.be.module_unload(&self._module) {
            tracing::warn!(error = %e, "unload decode module");
        }
    }
}

/// The checkpoint's stop-token set: `generation_config.json` `eos_token_id`
/// (int or list), falling back to `config.json`, falling back to empty (the
/// caller then stops on max_tokens only).
fn read_eos_ids(dir: &Path) -> Vec<u32> {
    for file in ["generation_config.json", "config.json"] {
        let Ok(bytes) = std::fs::read(dir.join(file)) else { continue };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                if let Some(id) = n.as_u64() {
                    return vec![id as u32];
                }
            }
            Some(serde_json::Value::Array(a)) => {
                let ids: Vec<u32> =
                    a.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
                if !ids.is_empty() {
                    return ids;
                }
            }
            _ => {}
        }
    }
    Vec::new()
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn selects_native_hopper_for_h100_and_h200() {
        let p = interpreter_profile((9, 0)).unwrap();
        assert_eq!(p.tag, "sm90a");
        assert_eq!(p.decode_file, "interp_sm90a.cubin");
        assert_eq!(p.decode_symbol, "_Z12interp_sm90a11PlowProgram");
    }

    #[test]
    fn preserves_sm120_profile_and_rejects_unknown_arches() {
        let p = interpreter_profile((12, 0)).unwrap();
        assert_eq!(p.prefill_file, "interp_sm120_pf.cubin");
        assert_eq!(p.prefill_symbol, "_Z15interp_sm120_pf11PlowProgram");
        assert!(interpreter_profile((8, 9)).is_none());
        assert!(interpreter_profile((10, 0)).is_none());
    }
}
