//! Shared schema types for plow's compiled artifacts.
//!
//! This crate is the **single source of truth** for types that cross the
//! compiler→runtime boundary. Both `plowc` (serialize) and `plowrt`
//! (deserialize) depend on it, eliminating duplicated struct definitions.
//!
//! `plowc` writes several JSON files alongside each `.pkt` stream:
//!
//! * `weights.json` — the per-network [`Manifest`] (weight tiling + per-bucket stats).
//! * `{stem}.map.json` — the [`MemoryMap`] (global address space + per-buffer placement).
//! * `{stem}.request_io.json` — the [`RequestIo`] sidecar (what to marshal per request).
//! * `{stem}.blocks.json` — the [`Blocks`] sidecar (per-transformer-block task ranges).
//! * `{stem}.experts.json` — the [`Experts`] sidecar (MoE routing metadata).
//! * `{stem}.decode_kv.json` — the [`DecodeKvSchema`] (KV patching for decode).

use serde::{Deserialize, Serialize};

// --- Shared domain enums -----------------------------------------------------

/// Which inference phase a bucket serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Prefill,
    Decode,
}

impl Phase {
    /// Parse from string (as serialized in `weights.json`).
    pub fn from_str_loose(s: &str) -> Phase {
        match s {
            "prefill" => Phase::Prefill,
            _ => Phase::Decode,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
        }
    }
}

// --- weights.json (per-network manifest) -------------------------------------

/// `weights.json` — one per compiled network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub network: String,
    pub gpu: String,
    pub num_gpus: usize,
    pub parallel: String,
    /// `true` iff one `(bn, bk)` tiling is legal across every bucket.
    pub weight_shared: bool,
    #[serde(default)]
    pub weight: Option<WeightLayout>,
    #[serde(default)]
    pub kv: Option<KvSummary>,
    #[serde(default)]
    pub fusion: Option<Fusion>,
    pub buckets: Vec<BucketStat>,
    /// Static-tensor manifest (compile-time constants staged from
    /// `static_tensors.bin`). Empty when no static tensors were emitted.
    #[serde(default)]
    pub static_tensors: Vec<StaticTensorEntry>,
    /// Whether `static_tensors.bin` was written next to `weights.json`.
    #[serde(default)]
    pub static_tensors_file_emitted: bool,
    /// Weight-tiling byte-layout spec — present iff `weight_shared`. See
    /// the design notes for the layout formula the runtime uses.
    #[serde(default)]
    pub weight_tiling: Option<WeightTiling>,
}

/// Shared weight tiling `(bn, bk)` reported in the manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightLayout {
    pub bn: i64,
    pub bk: i64,
}

/// Weight-tiling byte-layout spec. See the design notes.
///
/// # Layout formula (as of `block_iteration = "n_major_k_inner"`,
/// `within_block_layout = "n_outer_k_inner"`, `padding_policy = "zero_extend"`):
///
/// ```text
/// tile_grid_rows = ceil(N / BN)
/// tile_grid_cols = ceil(K / BK)
/// tile_ordinal(tr, tc) = tr * tile_grid_cols + tc
/// tile_offset(tr, tc)  = tile_ordinal * BN * BK * elem_bytes
/// byte_in_tile(n_local, k_local) = (n_local * BK + k_local) * elem_bytes
/// ```
///
/// with zero-padding at the far edge when `N % BN != 0` or `K % BK != 0`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightTiling {
    pub bn: i64,
    pub bk: i64,
    pub element_dtype: String,
    pub elem_bytes: u32,
    pub block_iteration: String,
    pub within_block_layout: String,
    pub padding_policy: String,
}

/// One entry in the `static_tensors` manifest — a byte range in
/// `static_tensors.bin` the runtime copies into an address-map slot at
/// model init. See the design notes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StaticTensorEntry {
    pub target_slot: String,
    pub offset_in_file: u64,
    pub size: u64,
    pub shape: Vec<i64>,
    pub dtype: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct KvSummary {
    pub block_seq: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Fusion {
    pub ops_before: usize,
    pub ops_after: usize,
    pub fused: usize,
}

/// One compiled `(phase, batch, seq)` bucket's stats + the artifact filenames.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BucketStat {
    pub phase: String,
    pub batch: i64,
    pub seq: i64,
    pub packet_file: String,
    pub packet_bytes: usize,
    pub instructions: usize,
    pub tile_nodes: usize,
    pub tasks: usize,
    pub makespan: u64,
    pub ideal_makespan: u64,
    pub arena_bytes: u64,
    pub memory_file: String,
}

// --- {stem}.map.json (address map) -------------------------------------------

/// Buffer lifetime class. Mirrors `schedule::BufClass`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufClass {
    /// Weights — filled once at load, stable across buckets.
    Persistent,
    /// Compile-time-computed constants (RoPE freq tables, static masks).
    /// Same lifetime as `Persistent` but bytes come from
    /// `static_tensors.bin`, not the checkpoint.
    Static,
    /// KV cache — extended in place by the runtime.
    Growable,
    /// Activations — reused via liveness; host never touches it.
    Scratch,
    /// Per-request input/output the host marshals each iteration.
    RequestIo,
}

impl BufClass {
    /// The data-type [`BufKind`] a buffer of this lifetime class holds by
    /// default. `RequestIo` can be an input or an output — the class alone can't
    /// tell, so it defaults to [`BufKind::Input`]; callers that know the
    /// direction (via the `RequestIo` sidecar) set [`MemEntry::kind`] explicitly.
    /// `Embedding` is never derived (embedding tables are `Persistent`); it is
    /// only reachable by explicit tagging.
    pub fn default_kind(self) -> BufKind {
        match self {
            BufClass::Persistent => BufKind::Weights,
            BufClass::Static => BufKind::Const,
            BufClass::Growable => BufKind::KvCache,
            BufClass::Scratch => BufKind::Activation,
            BufClass::RequestIo => BufKind::Input,
        }
    }

    /// The device-**kernel** access mode for this class: weights/consts are read
    /// only, KV and activations are read-write. `RequestIo` defaults to
    /// read-write (the region holds both kernel-read inputs and kernel-written
    /// outputs); a direction-aware caller narrows it.
    pub fn default_access(self) -> Access {
        match self {
            BufClass::Persistent | BufClass::Static => Access::Read,
            BufClass::Growable | BufClass::Scratch | BufClass::RequestIo => Access::ReadWrite,
        }
    }
}

/// What a buffer holds — the data-type taxonomy, orthogonal to the lifetime
/// [`BufClass`] and the [`Access`] mode. `#[repr(u8)]` so it can be tagged into
/// the packet ABI for on-device kernels; the discriminants are the wire values
/// (mirror `PLOW_KIND_*` in `runtime/common/memmap.h`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufKind {
    /// Checkpoint weights (GEMM/attention projection matrices).
    Weights = 0,
    /// Token/position embedding tables (gather-read).
    Embedding = 1,
    /// Compile-time constants (RoPE freq tables, static masks).
    Const = 2,
    /// Per-request input the host writes / the kernel reads (tokens, position_ids).
    Input = 3,
    /// Per-request output the kernel writes / the host reads (logits).
    Output = 4,
    /// KV cache (per-head growable pool).
    KvCache = 5,
    /// Intermediate activations (liveness-reused scratch).
    Activation = 6,
}

/// How a device kernel accesses a buffer. `#[repr(u8)]` for the packet ABI
/// (mirror `PLOW_ACCESS_*` in `runtime/common/memmap.h`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    /// Kernel reads only (weights, consts, inputs).
    Read = 0,
    /// Kernel writes only (outputs).
    Write = 1,
    /// Kernel reads and writes (KV cache, activations).
    ReadWrite = 2,
}

/// One placed buffer in the global address space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemEntry {
    pub slot: u32,
    pub name: String,
    pub class: BufClass,
    pub offset: u64,
    pub reserved: u64,
    pub growable: bool,
    pub device: u8,
    /// Source shape `[N, K]` for GEMM weight tensors (HuggingFace
    /// safetensors convention: out_features × in_features). Absent for
    /// non-weight entries. Runtime pairs this with `Manifest::weight_tiling`
    /// to arrange safetensor bytes at load time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_shape: Option<Vec<i64>>,
    /// Element dtype tag (`"bf16"`, `"fp16"`, ...). Present when
    /// `logical_shape` is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    /// Data-type tag (what the buffer holds). Absent on legacy maps — resolve
    /// via [`MemEntry::kind`], which falls back to the class-derived default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<BufKind>,
    /// Device-kernel access mode. Absent on legacy maps — resolve via
    /// [`MemEntry::access`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access: Option<Access>,
}

impl MemEntry {
    /// The buffer's [`BufKind`], falling back to the class-derived default when
    /// the map predates the explicit tag.
    pub fn kind(&self) -> BufKind {
        self.kind.unwrap_or_else(|| self.class.default_kind())
    }

    /// The buffer's kernel [`Access`] mode, falling back to the class-derived
    /// default when the map predates the explicit tag.
    pub fn access(&self) -> Access {
        self.access.unwrap_or_else(|| self.class.default_access())
    }
}

/// One device's slice of the contiguous global address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    pub device: u8,
    pub global_base: u64,
    pub size: u64,
    pub growable_base: u64,
}

/// KV paging metadata (present iff the bucket has a Flash attention op).
/// `initial_blocks` is per attention layer (`per_layer`), since a model's layers
/// each own a distinct `kv_cache_L{i}` growable buffer.
///
/// # Per-head pool geometry
///
/// The `kv_factor` / `max_seqs` / `head_slot_bytes` fields describe the
/// **per-head growable pool** the runtime allocates each layer's buffer as (see
/// `plowrt::memory::pool::GrowablePool` and `lean-plow/Plow/KvPool.lean`). A
/// layer's buffer is carved into `kv_factor × kv_heads × max_seqs` head-slots of
/// `head_slot_bytes` each, addressed kv-major → head → seq (positions inner-most
/// inside a slot). They default to `0`/`0`/`0` so older `map.json` that predate
/// the per-head layout still deserialize; a zero `head_slot_bytes` means "no
/// per-head pool geometry emitted" and the runtime falls back to the legacy
/// packed-block model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvPaging {
    pub block_tokens: i64,
    pub block_bytes: u64,
    pub kv_heads: i64,
    pub head_dim: i64,
    /// Distinct growable runs per (head, seq): `2` for separate K and V, `1` if
    /// fused. Mirrors `GrowablePool.kvFactor`.
    #[serde(default = "kv_factor_default")]
    pub kv_factor: i64,
    /// Max sequences in flight the pool reserves head-slots for. `0` ⇒ no
    /// per-head pool emitted (legacy packed-block model).
    #[serde(default)]
    pub max_seqs: i64,
    /// Bytes reserved per `(kv, head, seq)` head-slot = `max_seq_len × head_dim ×
    /// elem_bytes`. `0` ⇒ no per-head pool emitted. Mirrors
    /// `GrowablePool.headSlotBytes`.
    #[serde(default)]
    pub head_slot_bytes: u64,
    #[serde(default)]
    pub per_layer: Vec<KvLayerPaging>,
}

/// Default `kv_factor` for `map.json` predating the field: separate K and V.
fn kv_factor_default() -> i64 {
    2
}

/// One attention layer's KV paging: its growable buffer and initial reservation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvLayerPaging {
    pub layer_idx: u32,
    pub buffer_name: String,
    pub initial_blocks: i64,
}

/// `{stem}.map.json` — the address map the runtime rebases to.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryMap {
    pub arena_bytes: u64,
    pub growable_base: u64,
    pub segments: Vec<Segment>,
    pub entries: Vec<MemEntry>,
    #[serde(default)]
    pub kv_paging: Option<KvPaging>,
}

impl MemoryMap {
    /// The entry named `name` (first match). See [`MemoryMap::on_device`] for
    /// replicated (tensor-parallel) tensors.
    pub fn get(&self, name: &str) -> Option<&MemEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// The replica of `name` resident on `device`.
    pub fn on_device(&self, name: &str, device: u8) -> Option<&MemEntry> {
        self.entries
            .iter()
            .find(|e| e.name == name && e.device == device)
    }

    /// The device segment `entry` lives in.
    pub fn segment_of(&self, entry: &MemEntry) -> Option<&Segment> {
        self.segments.iter().find(|s| s.device == entry.device)
    }

    /// Rebase a global `offset` on `device` to a local byte offset within that
    /// device's physical arena: `offset − segment.global_base`.
    pub fn local_offset(&self, device: u8, offset: u64) -> Option<u64> {
        self.segments
            .iter()
            .find(|s| s.device == device)
            .map(|s| offset - s.global_base)
    }

    /// Structural sanity: every buffer fits inside the arena and slot ids are
    /// unique. Aliases overlap by design, so byte-disjointness is *not* checked.
    pub fn validate(&self) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for e in &self.entries {
            if e.offset + e.reserved > self.arena_bytes {
                return Err(format!(
                    "buffer '{}' [{}, {}) exceeds arena {}",
                    e.name,
                    e.offset,
                    e.offset + e.reserved,
                    self.arena_bytes
                ));
            }
            if !seen.insert(e.slot) {
                return Err(format!("duplicate slot id {}", e.slot));
            }
        }
        Ok(())
    }
}

// --- {stem}.request_io.json --------------------------------------------------

/// One per-request buffer the runtime marshals in (input) or out (output).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestIoField {
    /// Matches a [`MemEntry::name`] in the corresponding `map.json`.
    pub name: String,
    /// `"input"` | `"output"`.
    pub direction: String,
    /// `"tokens"` | `"logits"` | `"attention_mask"` | …
    pub semantic: String,
    pub shape: Vec<i64>,
    pub elem_bytes: u32,
}

/// `{stem}.request_io.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestIo {
    pub fields: Vec<RequestIoField>,
    pub complete: bool,
}

// --- {stem}.blocks.json ------------------------------------------------------

/// One transformer block's task-id range in the compiled program.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockRange {
    pub index: u32,
    pub label: String,
    pub first_task: usize,
    pub last_task: usize,
    pub task_count: usize,
}

/// `{stem}.blocks.json` — lets a pipeline-parallel / streaming runtime split the
/// bucket by block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Blocks {
    pub blocks: Vec<BlockRange>,
    pub complete: bool,
}

// --- {stem}.experts.json -----------------------------------------------------

/// One routed MoE layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertLayer {
    pub block: u32,
    pub layer_label: String,
    pub num_experts: u32,
    pub top_k: u32,
    pub router_op_name: String,
    #[serde(default)]
    pub routing_table_slot: String,
    #[serde(default)]
    pub expert_weight_table_slot: String,
    /// Per-expert routed weight names, `[num_experts]` of `{gate, up, down}` tensor names the
    /// runtime resolves into the flat `expert_weight_table` the SM indexes by expert id
    /// (`moe-ep-kernels.md §3a`, `orch/moe.rs::resolve_expert_tables`). Empty on a sidecar
    /// emitted before routed-expert resolution (backward-compatible).
    #[serde(default)]
    pub routed_experts: Vec<RoutedExpertWeights>,
}

/// One routed expert's three weight tensor names, in the fixed order the
/// `expert_weight_table` stores them: `{gate, up, down}` (`moe-ep-kernels.md §3a`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutedExpertWeights {
    pub gate: String,
    pub up: String,
    pub down: String,
}

/// One dense (always-run) shared expert.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedExpert {
    pub block: u32,
    pub layer_label: String,
    #[serde(default)]
    pub gate_up_weight: Option<String>,
    #[serde(default)]
    pub down_weight: Option<String>,
    pub replicated_across_gpus: bool,
}

/// `{stem}.experts.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Experts {
    pub layers: Vec<ExpertLayer>,
    pub shared: Vec<SharedExpert>,
    pub expert_unused_sentinel: u32,
    pub complete: bool,
}

// --- {stem}.decode_kv.json ---------------------------------------------------

/// One Flash-attention op in a decode packet stream. The runtime patches
/// `seq_kv` at issue time and sets the KV base address from `kv_buffer_name`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecodeFlashOp {
    /// Task.op name (matches the compiled packet stream's task attribution).
    pub op_name: String,
    /// Attention layer index (0-based).
    pub layer_idx: u32,
    /// Address-map entry name for this layer's KV cache.
    pub kv_buffer_name: String,
    /// Byte offset within the `FlashBody` POD where the `seq_kv: u32` field
    /// sits. Stable at 12 (after coord0, coord1, seq_q).
    pub seq_kv_field_offset: u32,
}

/// Decode-phase KV read address contract. Emitted as
/// `{stem}.decode_kv.json` for `phase == Decode` buckets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecodeKvSchema {
    pub flash_ops: Vec<DecodeFlashOp>,
    /// Buffer name for the runtime-owned per-request `past_len` counter.
    pub past_len_buffer: String,
    /// Tokens per KV block (matches `KvPaging.block_tokens`).
    pub block_tokens: i64,
}

// --- block.json (single-block descriptor) ------------------------------------

/// One dimension in a block-tensor shape: either a symbolic axis resolved at
/// run time (e.g. `"T"` for the token count) or a fixed extent. Serializes
/// untagged so a mixed shape like `["T", 7168]` round-trips.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dim {
    /// A named axis bound at launch (sequence / chunk length).
    Symbolic(String),
    /// A compile-time-fixed extent.
    Fixed(i64),
}

/// One block input/output activation tensor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockTensor {
    /// Handle name in the compiled program (e.g. `"act.x"`).
    pub name: String,
    pub shape: Vec<Dim>,
    pub dtype: String,
}

/// Architecture-specific dimensions. Only the keys the design notes
/// list are modeled; each is optional since which apply depends on `kind`
/// (a dense block carries `heads`; an MoE block adds `n_exp`/`top_k`/…).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BlockDims {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heads: Option<i64>,
    /// Per-head width (dense attn / Mamba). Gemma-4 varies this per layer
    /// (sliding vs full), so the descriptor records the extracted layer's value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<i64>,
    /// KV-head count (GQA). Dense-attn carried_state sizing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_heads: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_lora: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q_lora: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_exp: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_exp: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_inter: Option<i64>,
    /// DSA lightning-indexer heads (GLM-5.2 `index_n_heads`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_heads: Option<i64>,
    /// DSA index head width (GLM-5.2 `index_head_dim`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_dim: Option<i64>,
    /// DSA top-k gathered positions (GLM-5.2 `index_topk`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_topk: Option<i64>,
    /// Mamba-2 causal depthwise conv width (`d_conv`, Nemotron-3). The conv state
    /// carries `d_conv - 1` past inputs per channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d_conv: Option<i64>,
    /// Mamba-2 SSM state dimension (`d_state` / `ssm_state_size`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d_state: Option<i64>,
    /// Mamba-2 head count (`mamba_n_heads`). `head_dim` (above) is `d_inner / n_head`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_head: Option<i64>,
    /// Mamba-2 inner (expanded) width (`d_inner = expand * hidden`); the SSM runs over it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d_inner: Option<i64>,
    /// Mamba-2 group count (`n_groups`); B/C are shared across the heads in a group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_groups: Option<i64>,
}

/// One piece of state a block carries across steps (KV cache, Mamba conv/ssm
/// state, DSA indices). `role` + `tensors` is the architecture-agnostic axis:
/// the harness allocates / uploads these by name. See §4's kind→carried_state
/// table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CarriedState {
    /// `"kv"` | `"conv"` | `"ssm"` | `"dsa"` | …
    pub role: String,
    /// Handle names in the compiled program.
    pub tensors: Vec<String>,
    /// On-device layout tag (e.g. `"head_major"`).
    pub layout: String,
}

/// Where the block's layer-`l` weights come from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockWeights {
    /// `"symlink"` (bind from the checkpoint at load) or `"embed"` (captured
    /// slices in `SECT_STATIC_TENSORS`).
    pub mode: String,
    /// Checkpoint id the weights bind against.
    pub ckpt: String,
    /// Tensor-name prefix for this layer (e.g. `"model.layers.3."`).
    pub prefix: String,
}

/// The program grid one asset compiles: every prefill bucket plus the max
/// decode batch, so a whole B×T sweep runs without recompiling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockPrograms {
    pub prefill_buckets: Vec<i64>,
    pub decode_t: i64,
}

/// `block.json` — the architecture-agnostic single-block descriptor.
///
/// It is the shared driver of the block harness: an **input** to the schedule /
/// CPU-sim route (`plowc --net`) and the **`SECT_METADATA`** mirror the device
/// route emits into the PLOWDEV blob. `kind` + `carried_state` keep it
/// arch-agnostic across dense-attn, MLA, MoE, DSA, and Mamba blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockDescriptor {
    /// Model family id (e.g. `"kimi-k2.7"`).
    pub model: String,
    /// Block architecture tag (e.g. `"mla_moe"`).
    pub arch: String,
    /// Which layer of the model this block was extracted from.
    pub layer: u32,
    /// Ordered mixer/FFN kinds (e.g. `["mla_attn", "moe_ffn"]`).
    pub kind: Vec<String>,
    /// Residual-stream (hidden) width.
    pub hidden: i64,
    /// Activation dtype (e.g. `"bf16"`).
    pub dtype: String,
    pub dims: BlockDims,
    /// DSA indexer role of the extracted layer (GLM-5.2 IndexShare): `"indexer"`
    /// (owns/computes its top-k indices) or `"reuse"` (reuses the last indexer
    /// layer's indices — carried in as `dsa_indices`). Absent on architectures
    /// without a DSA indexer (dense, plain MLA/MoE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsa_role: Option<String>,
    pub inputs: Vec<BlockTensor>,
    pub outputs: Vec<BlockTensor>,
    pub carried_state: Vec<CarriedState>,
    pub weights: BlockWeights,
    pub programs: BlockPrograms,
}

pub mod decode_objects;

pub mod cubin;
pub mod decode_coverage;

pub mod program;
pub mod splitk;

pub mod live_kv;
