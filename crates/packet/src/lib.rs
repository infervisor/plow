//! Infervisor runtime packet ABI — the instruction stream the persistent-kernel
//! interpreter executes, shared byte-for-byte with the C/CUDA/ROCm runtime.
//!
//! The stream is a sequence of tagged, **`#[repr(C)]` POD records** a kernel
//! `reinterpret_cast`s directly: a fixed [`Header`] followed by an opcode-specific
//! body ([`GemmBody`], [`DmaBody`], …) carrying only the fields that op needs,
//! then its wait/successor counter ids. It is variable-length (a DMA record is
//! ~20 B, a GEMM ~44 B) yet cast-able because:
//! * every body is `#[repr(C)]` with **explicit padding** (no uninitialized
//!   bytes, identical layout to the C struct), and
//! * every record is **4-byte aligned** — the stream header and `Header` are
//!   4-byte multiples and bodies/counter-ids are ≤4-aligned, so casting a body
//!   or `u32` at any record offset is a valid aligned device load.
//!
//! ## Opcode Namespace (u16)
//!
//! ```text
//!  Bits [15:12]  Backend    [11:8]  Family       [7:0]  Variant
//!  0 = Generic   0 = Control (nop/host)   variant-specific
//!  1 = CUDA      1 = DMA                  0=tma_load, 1=tma_store
//!  2 = ROCm      2 = RDMA/Collective      0=p2p, 1=allreduce, ...
//!  3 = CPU       3 = Gemm                 dtype/epilogue combos
//!                4 = Flash                 causal/sliding/paged
//!                5 = Row                   0=reduce, 1=pointwise, ...
//!                6 = Layout                permute/pad/slice
//!                7..15 = Future            conv, scatter, custom...
//! ```
//!
//! The matching C structs are in `include/packet.h`. No dependencies — the
//! runtime links just this crate.

pub mod dev;
pub mod devbuild;
pub mod moe;
pub mod names;
pub mod rope;

use core::mem::size_of;

// --- Opcode (u16, structured) ------------------------------------------------

/// Kernel opcode — a structured u16 encoding `[backend:4][family:4][variant:8]`.
///
/// This unifies CUDA, ROCm, and CPU kernel variants in one flat namespace,
/// supporting template-generated kernel permutations (dtype, epilogue, tile-config)
/// without opcode exhaustion.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Opcode(pub u16);

impl Opcode {
    /// Construct an opcode from its three components.
    pub const fn new(backend: u8, family: u8, variant: u8) -> Self {
        Opcode(((backend as u16 & 0xF) << 12) | ((family as u16 & 0xF) << 8) | variant as u16)
    }

    /// Backend bits [15:12] — 0=Generic, 1=CUDA, 2=ROCm, 3=CPU.
    pub const fn backend(self) -> u8 {
        (self.0 >> 12) as u8
    }
    /// Family bits [11:8] — selects which body struct to cast.
    pub const fn family(self) -> u8 {
        ((self.0 >> 8) & 0xF) as u8
    }
    /// Variant bits [7:0] — kernel variant within a family (dtype, epilogue, etc.).
    pub const fn variant(self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    // --- Well-known opcodes (Generic backend) --------------------------------

    pub const NOP: Self = Self::new(0, 0, 0); // 0x0000
    pub const HOST_COORD: Self = Self::new(0, 0, 1); // 0x0001
    pub const TMA_LOAD: Self = Self::new(0, 1, 0); // 0x0100
    pub const TMA_STORE: Self = Self::new(0, 1, 1); // 0x0101
    pub const RDMA: Self = Self::new(0, 2, 0); // 0x0200
    pub const GEMM: Self = Self::new(0, 3, 0); // 0x0300
    pub const FLASH: Self = Self::new(0, 4, 0); // 0x0400
    pub const ROW_REDUCE: Self = Self::new(0, 5, 0); // 0x0500
    pub const ROW_POINTWISE: Self = Self::new(0, 5, 1); // 0x0501
    pub const LAYOUT: Self = Self::new(0, 6, 0); // 0x0600
    pub const SAMPLE: Self = Self::new(0, 7, 0); // 0x0700 — logits → token id
    pub const TOKENIZE: Self = Self::new(0, 7, 1); // 0x0701 — text/ids → tokens buffer
    pub const SAMPLE_BATCH: Self = Self::new(0, 7, 4); // 0x0704 — B×V logits → B token ids

    // --- Family constants (for body dispatch) --------------------------------

    pub const FAMILY_CONTROL: u8 = 0;
    pub const FAMILY_DMA: u8 = 1;
    pub const FAMILY_RDMA: u8 = 2;
    pub const FAMILY_GEMM: u8 = 3;
    pub const FAMILY_FLASH: u8 = 4;
    pub const FAMILY_ROW: u8 = 5;
    pub const FAMILY_LAYOUT: u8 = 6;
    /// Host-class token ops (sample/tokenize) — placeable on Host or Sm; the
    /// scheduler decides, the loader resolves the backend by `ResourceKind`.
    pub const FAMILY_TOKEN: u8 = 7;

    // Token-family variants (bits [7:0]) — the `kind` a `TokenBody` carries.
    pub const TOKEN_SAMPLE_GREEDY: u8 = 0x00;
    pub const TOKEN_SAMPLE_STOCHASTIC: u8 = 0x01;
    pub const TOKEN_TOKENIZE: u8 = 0x02;
    pub const TOKEN_DETOKENIZE: u8 = 0x03;
    /// Continuous-batching sample: `TokenBody.arg` carries the batch width B.
    /// Logits are laid out row-major `[B][vocab]` at `in_slot`; produced token
    /// ids are written row-major at `out_slot` (host-side buffer, one per row).
    /// One packet replaces B `SAMPLE_GREEDY`/`_STOCHASTIC` packets; per-row
    /// sampling params travel through the indirection table.
    pub const TOKEN_SAMPLE_BATCH: u8 = 0x04;

    // --- Backend constants ---------------------------------------------------

    pub const BACKEND_GENERIC: u8 = 0;
    pub const BACKEND_CUDA: u8 = 1;
    pub const BACKEND_ROCM: u8 = 2;
    pub const BACKEND_CPU: u8 = 3;

    // --- Variant convention (bits [7:0]) -------------------------------------
    //
    // The scheduler emits *generic* opcodes (backend 0). The runtime loader,
    // knowing the device `Arch`, resolves each to a concrete `(backend, family,
    // variant)` against the active backend's dispatch table. Per family:
    //
    // * `0x00` — golden/naive: the single-thread correctness reference for that
    //   backend (the CPU one is the oracle GPU output is checked against).
    // * `0x01..0x7F` — performant variants (dtype/epilogue/tile-class). Hopper vs
    //   Blackwell both live under `BACKEND_CUDA` and are picked by *which per-arch
    //   table the loader installs*, leaving the variant byte free for dtype.
    //
    // GEMM-family variant assignments used by the runtime kernels:
    pub const VARIANT_GOLDEN: u8 = 0x00; // naive single-thread reference
    pub const VARIANT_BF16: u8 = 0x01; // fast.cu / ThunderKittens
    pub const VARIANT_FP8: u8 = 0x02; // DeepGEMM
    pub const VARIANT_W4A8: u8 = 0x03; // LiquidGEMM
    pub const VARIANT_FP4: u8 = 0x04; // MX FP4 (native Blackwell tensor-core path)
    pub const VARIANT_GROUPED: u8 = 0x10; // DeepGEMM grouped / MoE

    // --- Gemma 4 fused-kernel variants ---------------------------------------
    //
    // Convention: the low nibble is the dtype ladder (bf16 = +1); the high nibble
    // groups a structural fusion class so the byte never exhausts:
    //   0x00..0x0F  base: 0x00 golden, 0x0N dtype ladder, 0x09 split-K bf16
    //   0x10..0x1F  GEMM grouped/MoE (existing 0x10)
    //   0x20..0x3F  GEMM RMSNorm-prologue block
    //   0x40..0x5F  Row RMSNorm→RoPE fusion block
    // The variant byte is identical across backends; only the backend nibble of
    // the full opcode differs (CUDA=1, ROCm=2). The scheduler emits generic
    // (backend 0); the loader rewrites the backend nibble per active arch.

    // GEMM (family 3)
    pub const VARIANT_BF16_SPLITK: u8 = 0x09; // plain bf16, partial-K accumulate (decode)
    pub const VARIANT_NORM_BF16: u8 = 0x21; // RMSNorm-prologue → bf16 GEMM (q/kv/gate/up)
    pub const VARIANT_NORM_SPLITK_BF16: u8 = 0x29; // norm-prologue bf16, partial-K (decode)

    // Flash (family 4)
    pub const VARIANT_FLASH_CAUSAL_BF16: u8 = 0x01; // full causal, FA-2 tiling
    pub const VARIANT_FLASH_SLIDING_BF16: u8 = 0x02; // sliding-window causal mask
    pub const VARIANT_FLASH_DECODE_BF16: u8 = 0x03; // single-query, split-KV + merge

    // Row (family 5). `variant_is_reduce` must agree with these.
    pub const VARIANT_ROW_RMS_BF16: u8 = 0x04; // bf16 RMSNorm (reduce over feat)
    pub const VARIANT_ROW_RESIDUAL_ADD_BF16: u8 = 0x06; // ew add, 2 operands
    pub const VARIANT_ROW_SWIGLU_BF16: u8 = 0x07; // silu(gate)*up, 2 operands
    pub const VARIANT_ROW_NORMROPE_BF16: u8 = 0x40; // RMSNorm→RoPE (K path)
    pub const VARIANT_ROW_NORMROPESCALE_BF16: u8 = 0x41; // RMSNorm→RoPE→scale (Q path)

    // Layout (family 6)
    pub const VARIANT_LAYOUT_COPY_BF16: u8 = 0x01; // vectorized strided copy
    pub const VARIANT_LAYOUT_GATHER_SCALE_BF16: u8 = 0x42; // gather rows by id + scale

    /// Whether a Row-family variant reads/writes a reduce-shaped body (RMSNorm/
    /// LayerNorm/softmax) vs a pointwise/fused-shaped one. The `RowBody` struct is
    /// identical either way, so this only informs the golden reference path and the
    /// `reduce` flag surfaced by [`Body`]; concrete dispatch is by full variant in
    /// the backend table. Reduce variants: `0x00` (golden) and `0x04` (bf16 RMSNorm).
    pub const fn variant_is_reduce(variant: u8) -> bool {
        matches!(variant, Self::VARIANT_GOLDEN | Self::VARIANT_ROW_RMS_BF16)
    }
}

/// Resource class a record runs on.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Sm = 0,
    Dma = 1,
    Dpu = 2,
    Host = 3,
}

impl ResourceKind {
    fn from_u8(v: u8) -> ResourceKind {
        match v {
            1 => ResourceKind::Dma,
            2 => ResourceKind::Dpu,
            3 => ResourceKind::Host,
            _ => ResourceKind::Sm,
        }
    }
}

/// Physical relationship of a counter's producers and consumers.
///
/// Shared by the compiler (`schedule::passes`) and runtime (`plowrt::exec::counters`)
/// so both sides agree on the counter scope semantics without duplicating the enum.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Within one SM/CU — shared-memory barrier.
    IntraSm = 0,
    /// Across SMs on one GPU — L2/global device-scope atomic.
    IntraGpu = 1,
    /// Across devices — system-scope atomic on host-pinned mapped memory.
    CrossUnit = 2,
}

impl Scope {
    #[inline]
    pub fn from_u8(v: u8) -> Scope {
        match v {
            0 => Scope::IntraSm,
            1 => Scope::IntraGpu,
            _ => Scope::CrossUnit,
        }
    }
}

/// "No slot" sentinel for slot-handle fields.
pub const SLOT_NONE: u16 = u16::MAX;
/// "No tensor" sentinel for the DMA tensor handle.
pub const TENSOR_NONE: u32 = u32::MAX;

// --- the C-castable record structs ------------------------------------------
// Field order is largest-first + an explicit `_pad` so size is a 4-byte multiple
// with NO implicit/uninitialized padding (safe to read as bytes; matches C).

/// Per-record fixed header (12 bytes, 4-byte aligned, 4-byte multiple).
///
/// Layout: `opcode:u16, resource:u8, unit:u8, index:u16, wait_len:u16, succ_len:u16, _pad:u16`
///
/// `wait`/`succ` counter-id arrays of `wait_len`/`succ_len` `u32`s follow the body.
/// v3 widened wait_len/succ_len from u8→u16 to support fine per-tile counters.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub opcode: u16,
    pub resource: u8,
    pub unit: u8,
    pub index: u16,
    pub wait_len: u16,
    pub succ_len: u16,
    pub _pad: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmaBody {
    pub bytes: u32,
    pub tensor: u32,
    pub slot: u16,
    /// Data-type tag of the buffer moved (`plow_asset::BufKind` as `u8`;
    /// `KIND_UNSPECIFIED` when the emitter can't resolve it). Lets a backend
    /// branch (e.g. KV write-back vs weight prefetch) without a map lookup.
    pub kind: u8,
    /// Kernel access mode (`plow_asset::Access` as `u8`): a load reads the
    /// buffer (`ACCESS_READ`), a store writes it (`ACCESS_WRITE`).
    pub access: u8,
}

/// `DmaBody::kind` sentinel: the emitter had no address-map entry for the tensor.
pub const KIND_UNSPECIFIED: u8 = 0xFF;
/// `Access::Read` / `Access::Write` wire values (mirror `plow_asset::Access`).
pub const ACCESS_READ: u8 = 0;
pub const ACCESS_WRITE: u8 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RdmaBody {
    pub bytes: u32,
    pub src_unit: u8,
    pub dst_unit: u8,
    pub _pad: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmBody {
    pub coord0: u32,
    pub coord1: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub bm: u16,
    pub bn: u16,
    pub bk: u16,
    pub out: u16,
    pub tmem: u16,
    pub _pad: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashBody {
    pub coord0: u32,
    pub coord1: u32,
    pub seq_q: u32,
    pub seq_kv: u32,
    pub head_dim: u16,
    pub bq: u16,
    pub bkv: u16,
    pub heads: u16,
    pub out: u16,
    pub tmem: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowBody {
    pub coord: u32,
    pub rows: u32,
    pub feat: u32,
    pub br: u16,
    pub out: u16,
    pub operands: u8,
    pub _pad: [u8; 3],
}

/// TOKEN body (sample / tokenize) — a host-class op the scheduler may also place
/// on an SM. `kind` selects the operation (see `TOKEN_*`); `in_slot`/`out_slot`
/// are address-map slots (e.g. logits → tokens); `vocab` is the logit width;
/// `arg` is op-specific (e.g. max tokens for tokenize). Per-request sampling
/// params travel through the indirection table, not this body.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenBody {
    pub in_slot: u16,
    pub out_slot: u16,
    pub kind: u8,
    pub _pad: u8,
    pub vocab: u32,
    pub arg: u32,
}

/// Max tensor rank a LAYOUT descriptor addresses (NCDHW + batch).
pub const LAYOUT_MAX_RANK: usize = 6;

/// LAYOUT body (v4): a strided block copy `out[out_base + Σ idxₐ·out_strideₐ] =
/// in[in_base + Σ idxₐ·in_strideₐ]` over `shape`. `kind==0` is a plain contiguous
/// copy (fast path); `kind==1` is the general strided gather/scatter that realizes
/// transpose (permuted `in_stride`), broadcast (`in_stride==0`), non-contiguous
/// slice (`in_base`+extents), and inner-axis concat (`out_stride`). Strides and
/// bases are in **elements**; `elem_size` is the per-element byte count.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutBody {
    pub kind: u8,
    pub rank: u8,
    pub elem_size: u8,
    pub _pad0: u8,
    pub out: u16,
    pub _pad1: u16,
    pub shape: [u32; LAYOUT_MAX_RANK],
    pub in_stride: [u32; LAYOUT_MAX_RANK],
    pub out_stride: [u32; LAYOUT_MAX_RANK],
    pub in_base: u32,
    pub out_base: u32,
}

/// LAYOUT body for streams at version ≤ 3 (a bare tile coordinate; the layout was
/// a flat copy whose byte count came from the host binding). Kept for decode
/// back-compat only — never emitted.
#[repr(C)]
#[derive(Clone, Copy)]
struct LayoutBodyLegacy {
    _coord: u32,
}

/// A clustered dependency counter (12 bytes, 4-aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counter {
    pub id: u32,
    pub threshold: u32,
    pub scope: u8, // 0 intra-SM, 1 intra-GPU, 2 cross-unit
    pub _pad: [u8; 3],
}

// --- ergonomic builder side --------------------------------------------------

/// The opcode-specific payload — only the fields that op needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Body {
    Dma {
        load: bool,
        bytes: u32,
        slot: u16,
        tensor: u32,
        /// Buffer data-type tag (`BufKind` as `u8`; `KIND_UNSPECIFIED` if absent).
        kind: u8,
        /// Kernel access mode (`Access` as `u8`).
        access: u8,
    },
    Rdma {
        bytes: u32,
        src_unit: u8,
        dst_unit: u8,
    },
    Gemm {
        coord: [u32; 2],
        m: u32,
        n: u32,
        k: u32,
        bm: u16,
        bn: u16,
        bk: u16,
        out: u16,
        tmem: u16,
        /// Kernel variant (dtype/epilogue). Lives in the Header opcode byte, not
        /// the body payload. `0` = golden, `VARIANT_BF16` = fast bf16, etc.
        variant: u8,
    },
    Flash {
        coord: [u32; 2],
        seq_q: u32,
        seq_kv: u32,
        head_dim: u16,
        bq: u16,
        bkv: u16,
        heads: u16,
        out: u16,
        tmem: u16,
        /// Kernel variant byte for the opcode.
        variant: u8,
    },
    Row {
        reduce: bool,
        coord: u32,
        rows: u32,
        feat: u32,
        operands: u8,
        br: u16,
        out: u16,
        /// Kernel variant byte for the opcode.
        variant: u8,
    },
    Layout {
        kind: u8,
        rank: u8,
        elem_size: u8,
        out: u16,
        shape: [u32; LAYOUT_MAX_RANK],
        in_stride: [u32; LAYOUT_MAX_RANK],
        out_stride: [u32; LAYOUT_MAX_RANK],
        in_base: u32,
        out_base: u32,
    },
    /// Sample / tokenize (host-class, may be placed on an SM). See [`TokenBody`].
    Token {
        in_slot: u16,
        out_slot: u16,
        kind: u8,
        vocab: u32,
        arg: u32,
    },
    Host,
}

impl Body {
    /// A contiguous byte copy of `bytes` bytes into slot `out` — the behavior the
    /// flat-copy LAYOUT had, expressed as a `kind==0` descriptor (`elem_size==1`,
    /// a single axis of length `bytes`, unit strides).
    pub fn layout_copy(out: u16, bytes: u32) -> Body {
        let mut shape = [0u32; LAYOUT_MAX_RANK];
        let mut in_stride = [0u32; LAYOUT_MAX_RANK];
        let mut out_stride = [0u32; LAYOUT_MAX_RANK];
        shape[0] = bytes;
        in_stride[0] = 1;
        out_stride[0] = 1;
        Body::Layout {
            kind: 0,
            rank: 1,
            elem_size: 1,
            out,
            shape,
            in_stride,
            out_stride,
            in_base: 0,
            out_base: 0,
        }
    }
}

impl Body {
    pub fn opcode(&self) -> Opcode {
        match self {
            Body::Dma { load: true, .. } => Opcode::TMA_LOAD,
            Body::Dma { load: false, .. } => Opcode::TMA_STORE,
            Body::Rdma { .. } => Opcode::RDMA,
            Body::Gemm { variant, .. } => Opcode::new(0, Opcode::FAMILY_GEMM, *variant),
            Body::Flash { variant, .. } => Opcode::new(0, Opcode::FAMILY_FLASH, *variant),
            Body::Row { variant, .. } => Opcode::new(0, Opcode::FAMILY_ROW, *variant),
            Body::Layout { .. } => Opcode::LAYOUT,
            Body::Token { kind, .. } => Opcode::new(0, Opcode::FAMILY_TOKEN, *kind),
            Body::Host => Opcode::HOST_COORD,
        }
    }
}

/// One runtime instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inst {
    pub resource: ResourceKind,
    pub unit: u8,
    pub index: u16,
    pub body: Body,
    pub wait: Vec<u32>,
    pub succ: Vec<u32>,
}

/// A full instruction stream + its counter table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub insts: Vec<Inst>,
    pub counters: Vec<Counter>,
    /// Which shape bucket this stream serves (runtime dispatch key).
    pub bucket_id: u16,
    /// Generation counter — the runtime rejects stale streams with old plan_gen.
    pub plan_gen: u16,
    /// Reserved flags (bit 0: has_crc; bits 1..15 reserved for future use).
    pub flags: u16,
}

impl Default for Program {
    fn default() -> Self {
        Program {
            insts: Vec::new(),
            counters: Vec::new(),
            bucket_id: 0,
            plan_gen: 0,
            flags: 0,
        }
    }
}

pub const MAGIC: u32 = 0x494E_5650; // "INVP"
pub const VERSION: u16 = 5;
/// Minimum version this decoder still accepts (v2 streams with u8 wait/succ lens).
pub const MIN_VERSION: u16 = 2;

/// POD struct → its bytes (safe: `#[repr(C)]`, all-integer fields, explicit pad).
fn push_pod<T: Copy>(o: &mut Vec<u8>, v: &T) {
    let p = unsafe { core::slice::from_raw_parts((v as *const T) as *const u8, size_of::<T>()) };
    o.extend_from_slice(p);
}
/// Read a POD struct from a (possibly unaligned) byte slice. Caller must ensure
/// `at + size_of::<T>() <= b.len()` (used after an explicit bounds check, or for
/// the fixed-size stream header that `decode` validates up front).
fn read_pod<T: Copy>(b: &[u8], at: usize) -> T {
    unsafe { core::ptr::read_unaligned(b[at..].as_ptr() as *const T) }
}

/// Bounds-checked POD read: `None` if `[at, at + size_of::<T>())` is out of
/// range. Used by [`Program::decode`] so a malformed/truncated stream is a clean
/// `Err`, never an out-of-bounds read.
fn try_pod<T: Copy>(b: &[u8], at: usize) -> Option<T> {
    let end = at.checked_add(size_of::<T>())?;
    if end > b.len() {
        return None;
    }
    Some(unsafe { core::ptr::read_unaligned(b[at..].as_ptr() as *const T) })
}

impl Body {
    fn push(&self, o: &mut Vec<u8>) {
        match *self {
            Body::Dma {
                bytes,
                slot,
                tensor,
                kind,
                access,
                ..
            } => push_pod(
                o,
                &DmaBody {
                    bytes,
                    tensor,
                    slot,
                    kind,
                    access,
                },
            ),
            Body::Rdma {
                bytes,
                src_unit,
                dst_unit,
            } => push_pod(
                o,
                &RdmaBody {
                    bytes,
                    src_unit,
                    dst_unit,
                    _pad: 0,
                },
            ),
            Body::Gemm {
                coord,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                out,
                tmem,
                variant: _,
            } => push_pod(
                o,
                &GemmBody {
                    coord0: coord[0],
                    coord1: coord[1],
                    m,
                    n,
                    k,
                    bm,
                    bn,
                    bk,
                    out,
                    tmem,
                    _pad: 0,
                },
            ),
            Body::Flash {
                coord,
                seq_q,
                seq_kv,
                head_dim,
                bq,
                bkv,
                heads,
                out,
                tmem,
                variant: _,
            } => push_pod(
                o,
                &FlashBody {
                    coord0: coord[0],
                    coord1: coord[1],
                    seq_q,
                    seq_kv,
                    head_dim,
                    bq,
                    bkv,
                    heads,
                    out,
                    tmem,
                },
            ),
            Body::Row {
                coord,
                rows,
                feat,
                operands,
                br,
                out,
                ..
            } => push_pod(
                o,
                &RowBody {
                    coord,
                    rows,
                    feat,
                    br,
                    out,
                    operands,
                    _pad: [0; 3],
                },
            ),
            Body::Layout {
                kind,
                rank,
                elem_size,
                out,
                shape,
                in_stride,
                out_stride,
                in_base,
                out_base,
            } => push_pod(
                o,
                &LayoutBody {
                    kind,
                    rank,
                    elem_size,
                    _pad0: 0,
                    out,
                    _pad1: 0,
                    shape,
                    in_stride,
                    out_stride,
                    in_base,
                    out_base,
                },
            ),
            Body::Token {
                in_slot,
                out_slot,
                kind,
                vocab,
                arg,
            } => push_pod(
                o,
                &TokenBody {
                    in_slot,
                    out_slot,
                    kind,
                    _pad: 0,
                    vocab,
                    arg,
                },
            ),
            Body::Host => {}
        }
    }

    /// Read a body by family at byte offset `at`, returning it + its byte size.
    /// The full `opcode` is passed to reconstruct variant info (e.g. load vs
    /// store). Bounds-checked: `Err` if the body runs past the buffer.
    fn try_read(
        op: Opcode,
        b: &[u8],
        at: usize,
        version: u16,
    ) -> Result<(Body, usize), &'static str> {
        let out = match op.family() {
            Opcode::FAMILY_DMA => {
                let r: DmaBody = try_pod(b, at).ok_or("truncated DMA body")?;
                let load = op.variant() == 0; // variant 0 = tma_load, 1 = tma_store
                (
                    Body::Dma {
                        load,
                        bytes: r.bytes,
                        slot: r.slot,
                        tensor: r.tensor,
                        kind: r.kind,
                        access: r.access,
                    },
                    size_of::<DmaBody>(),
                )
            }
            Opcode::FAMILY_RDMA => {
                let r: RdmaBody = try_pod(b, at).ok_or("truncated RDMA body")?;
                (
                    Body::Rdma {
                        bytes: r.bytes,
                        src_unit: r.src_unit,
                        dst_unit: r.dst_unit,
                    },
                    size_of::<RdmaBody>(),
                )
            }
            Opcode::FAMILY_GEMM => {
                let r: GemmBody = try_pod(b, at).ok_or("truncated GEMM body")?;
                (
                    Body::Gemm {
                        coord: [r.coord0, r.coord1],
                        m: r.m,
                        n: r.n,
                        k: r.k,
                        bm: r.bm,
                        bn: r.bn,
                        bk: r.bk,
                        out: r.out,
                        tmem: r.tmem,
                        variant: op.variant(),
                    },
                    size_of::<GemmBody>(),
                )
            }
            Opcode::FAMILY_FLASH => {
                let r: FlashBody = try_pod(b, at).ok_or("truncated FLASH body")?;
                (
                    Body::Flash {
                        coord: [r.coord0, r.coord1],
                        seq_q: r.seq_q,
                        seq_kv: r.seq_kv,
                        head_dim: r.head_dim,
                        bq: r.bq,
                        bkv: r.bkv,
                        heads: r.heads,
                        out: r.out,
                        tmem: r.tmem,
                        variant: op.variant(),
                    },
                    size_of::<FlashBody>(),
                )
            }
            Opcode::FAMILY_ROW => {
                let r: RowBody = try_pod(b, at).ok_or("truncated ROW body")?;
                let reduce = Opcode::variant_is_reduce(op.variant());
                (
                    Body::Row {
                        reduce,
                        coord: r.coord,
                        rows: r.rows,
                        feat: r.feat,
                        operands: r.operands,
                        br: r.br,
                        out: r.out,
                        variant: op.variant(),
                    },
                    size_of::<RowBody>(),
                )
            }
            Opcode::FAMILY_LAYOUT => {
                if version >= 4 {
                    let r: LayoutBody = try_pod(b, at).ok_or("truncated LAYOUT body")?;
                    (
                        Body::Layout {
                            kind: r.kind,
                            rank: r.rank,
                            elem_size: r.elem_size,
                            out: r.out,
                            shape: r.shape,
                            in_stride: r.in_stride,
                            out_stride: r.out_stride,
                            in_base: r.in_base,
                            out_base: r.out_base,
                        },
                        size_of::<LayoutBody>(),
                    )
                } else {
                    // v≤3: legacy 4-byte coord; surface as an empty copy descriptor.
                    let _r: LayoutBodyLegacy = try_pod(b, at).ok_or("truncated LAYOUT body")?;
                    (Body::layout_copy(SLOT_NONE, 0), size_of::<LayoutBodyLegacy>())
                }
            }
            Opcode::FAMILY_TOKEN => {
                let r: TokenBody = try_pod(b, at).ok_or("truncated TOKEN body")?;
                (
                    Body::Token {
                        in_slot: r.in_slot,
                        out_slot: r.out_slot,
                        kind: r.kind,
                        vocab: r.vocab,
                        arg: r.arg,
                    },
                    size_of::<TokenBody>(),
                )
            }
            _ => (Body::Host, 0), // FAMILY_CONTROL and unknowns
        };
        Ok(out)
    }
}

/// Stream header size in bytes (20 B, same across v2/v3 — record layout changes).
const STREAM_HEADER_SIZE: usize = 20;

/// Size of the v2 record header (u8 wait/succ lengths).
const HEADER_V2_SIZE: usize = 8;

impl Program {
    /// Serialize to the wire stream (20-byte header, then 4-aligned records, then
    /// the counter table). Always emits v3 (u16 wait/succ lengths).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        // Stream header (20 bytes)
        o.extend_from_slice(&MAGIC.to_le_bytes());
        o.extend_from_slice(&VERSION.to_le_bytes());
        o.extend_from_slice(&self.bucket_id.to_le_bytes());
        o.extend_from_slice(&(self.insts.len() as u32).to_le_bytes());
        o.extend_from_slice(&(self.counters.len() as u32).to_le_bytes());
        o.extend_from_slice(&self.plan_gen.to_le_bytes());
        o.extend_from_slice(&self.flags.to_le_bytes());
        debug_assert_eq!(o.len(), STREAM_HEADER_SIZE);
        // Records (v3: u16 wait/succ lengths)
        for ins in &self.insts {
            assert!(
                ins.wait.len() <= u16::MAX as usize,
                "wait_len overflow: {} > 65535",
                ins.wait.len()
            );
            assert!(
                ins.succ.len() <= u16::MAX as usize,
                "succ_len overflow: {} > 65535",
                ins.succ.len()
            );
            push_pod(
                &mut o,
                &Header {
                    opcode: ins.body.opcode().0,
                    resource: ins.resource as u8,
                    unit: ins.unit,
                    index: ins.index,
                    wait_len: ins.wait.len() as u16,
                    succ_len: ins.succ.len() as u16,
                    _pad: 0,
                },
            );
            ins.body.push(&mut o);
            for &w in &ins.wait {
                o.extend_from_slice(&w.to_le_bytes());
            }
            for &s in &ins.succ {
                o.extend_from_slice(&s.to_le_bytes());
            }
        }
        // Counter table
        for c in &self.counters {
            push_pod(&mut o, c);
        }
        o
    }

    /// Decode a stream produced by [`Program::to_bytes`]. Supports both v2 (u8
    /// wait/succ lengths, Header = 8 B) and v3 (u16 lengths, Header = 12 B).
    pub fn decode(b: &[u8]) -> Result<Program, &'static str> {
        if b.len() < STREAM_HEADER_SIZE {
            return Err("stream too short");
        }
        if read_pod::<u32>(b, 0) != MAGIC {
            return Err("bad magic");
        }
        let version = read_pod::<u16>(b, 4);
        if !(MIN_VERSION..=VERSION).contains(&version) {
            return Err("bad version");
        }
        let bucket_id = read_pod::<u16>(b, 6);
        let n_insts = read_pod::<u32>(b, 8) as usize;
        let n_counters = read_pod::<u32>(b, 12) as usize;
        let plan_gen = read_pod::<u16>(b, 16);
        let flags = read_pod::<u16>(b, 18);

        let hdr_size = if version >= 3 {
            size_of::<Header>() // 12 B (v3)
        } else {
            HEADER_V2_SIZE // 8 B (v2)
        };

        let mut i = STREAM_HEADER_SIZE;
        // Don't pre-reserve from the untrusted `n_insts`; bound it by what the
        // remaining bytes could possibly hold (each record is ≥ one header).
        let cap = n_insts.min((b.len() - STREAM_HEADER_SIZE) / hdr_size.max(1));
        let mut insts = Vec::with_capacity(cap);
        // Read a `len`-long array of u32 counter ids at `i`, advancing `i`.
        let read_ids = |b: &[u8], i: &mut usize, len: usize, what: &'static str| {
            let nbytes = len.checked_mul(4).ok_or("counter-id array overflow")?;
            let end = i.checked_add(nbytes).ok_or(what)?;
            if end > b.len() {
                return Err(what);
            }
            let ids: Vec<u32> = (0..len).map(|j| read_pod::<u32>(b, *i + j * 4)).collect();
            *i = end;
            Ok(ids)
        };
        for _ in 0..n_insts {
            // Decode header based on version.
            let (opcode, resource, unit, index, wait_len, succ_len) = if version >= 3 {
                let h: Header = try_pod(b, i).ok_or("truncated record header")?;
                i += size_of::<Header>();
                (h.opcode, h.resource, h.unit, h.index, h.wait_len as usize, h.succ_len as usize)
            } else {
                // v2: 8-byte header with u8 wait/succ lengths
                if i + HEADER_V2_SIZE > b.len() {
                    return Err("truncated record header");
                }
                let opcode = read_pod::<u16>(b, i);
                let resource = b[i + 2];
                let unit = b[i + 3];
                let index = read_pod::<u16>(b, i + 4);
                let wait_len = b[i + 6] as usize;
                let succ_len = b[i + 7] as usize;
                i += HEADER_V2_SIZE;
                (opcode, resource, unit, index, wait_len, succ_len)
            };
            let op = Opcode(opcode);
            let (body, bsz) = Body::try_read(op, b, i, version)?;
            i += bsz;
            let wait = read_ids(b, &mut i, wait_len, "truncated wait array")?;
            let succ = read_ids(b, &mut i, succ_len, "truncated succ array")?;
            insts.push(Inst {
                resource: ResourceKind::from_u8(resource),
                unit,
                index,
                body,
                wait,
                succ,
            });
        }
        // Counter table: validate the whole span before reading any row.
        let ct_bytes = n_counters
            .checked_mul(size_of::<Counter>())
            .ok_or("counter table overflow")?;
        if i.checked_add(ct_bytes).is_none_or(|end| end > b.len()) {
            return Err("truncated counter table");
        }
        let counters: Vec<Counter> = (0..n_counters)
            .map(|j| read_pod::<Counter>(b, i + j * size_of::<Counter>()))
            .collect();
        // Reject out-of-range counter ids here: the runtime sizes its atomic
        // pool from the counter table and dereferences ids unchecked on the
        // hot path, so a stale/corrupt id would be an OOB atomic write.
        let id_bound = counters.iter().map(|c| c.id as u64 + 1).max().unwrap_or(0);
        for inst in &insts {
            if inst.wait.iter().chain(&inst.succ).any(|&id| id as u64 >= id_bound) {
                return Err("counter id out of range");
            }
        }
        Ok(Program {
            insts,
            counters,
            bucket_id,
            plan_gen,
            flags,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::align_of;

    /// The C/CUDA/ROCm side casts these — the sizes are the ABI contract.
    #[test]
    fn record_layout_is_c_compatible() {
        assert_eq!(size_of::<Header>(), 12);
        assert_eq!(size_of::<DmaBody>(), 12);
        assert_eq!(size_of::<RdmaBody>(), 8);
        assert_eq!(size_of::<GemmBody>(), 32);
        assert_eq!(size_of::<FlashBody>(), 28);
        assert_eq!(size_of::<RowBody>(), 20);
        assert_eq!(size_of::<LayoutBody>(), 88);
        assert_eq!(size_of::<TokenBody>(), 16);
        assert_eq!(size_of::<Counter>(), 12);
        // Every record is ≤4-aligned and a 4-byte multiple ⇒ records + u32 fields
        // stay aligned when laid end-to-end.
        for (sz, al) in [
            (size_of::<Header>(), align_of::<Header>()),
            (size_of::<GemmBody>(), align_of::<GemmBody>()),
            (size_of::<FlashBody>(), align_of::<FlashBody>()),
            (size_of::<DmaBody>(), align_of::<DmaBody>()),
            (size_of::<RowBody>(), align_of::<RowBody>()),
            (size_of::<LayoutBody>(), align_of::<LayoutBody>()),
            (size_of::<TokenBody>(), align_of::<TokenBody>()),
            (size_of::<Counter>(), align_of::<Counter>()),
        ] {
            assert_eq!(sz % 4, 0);
            assert!(al <= 4);
        }
    }

    #[test]
    fn sample_batch_opcode_and_roundtrip() {
        // Well-known constant + structured encoding.
        assert_eq!(Opcode::SAMPLE_BATCH.0, 0x0704);
        assert_eq!(Opcode::SAMPLE_BATCH.family(), Opcode::FAMILY_TOKEN);
        assert_eq!(Opcode::SAMPLE_BATCH.variant(), Opcode::TOKEN_SAMPLE_BATCH);
        // A Body::Token with `kind = TOKEN_SAMPLE_BATCH` reports SAMPLE_BATCH.
        let body = Body::Token {
            in_slot: 5,
            out_slot: 6,
            kind: Opcode::TOKEN_SAMPLE_BATCH,
            vocab: 4096,
            arg: 8, // batch width
        };
        assert_eq!(body.opcode(), Opcode::SAMPLE_BATCH);
        // Encoded and decoded a full one-inst program round-trips the arg.
        let p = Program {
            insts: vec![Inst {
                resource: ResourceKind::Sm,
                unit: 0,
                index: 0,
                body,
                wait: vec![],
                succ: vec![],
            }],
            counters: vec![],
            bucket_id: 0,
            plan_gen: 0,
            flags: 0,
        };
        let bytes = p.to_bytes();
        let p2 = Program::decode(&bytes).unwrap();
        match &p2.insts[0].body {
            Body::Token { arg, kind, vocab, .. } => {
                assert_eq!(*arg, 8);
                assert_eq!(*kind, Opcode::TOKEN_SAMPLE_BATCH);
                assert_eq!(*vocab, 4096);
            }
            other => panic!("unexpected body: {other:?}"),
        }
    }

    #[test]
    fn opcode_structured_encoding() {
        // Generic/Gemm/variant0
        assert_eq!(Opcode::GEMM.0, 0x0300);
        assert_eq!(Opcode::GEMM.backend(), 0);
        assert_eq!(Opcode::GEMM.family(), 3);
        assert_eq!(Opcode::GEMM.variant(), 0);

        // CUDA/Gemm/fp8 (hypothetical variant 0x01)
        let cuda_gemm_fp8 = Opcode::new(1, 3, 1);
        assert_eq!(cuda_gemm_fp8.0, 0x1301);
        assert_eq!(cuda_gemm_fp8.backend(), 1);
        assert_eq!(cuda_gemm_fp8.family(), 3);
        assert_eq!(cuda_gemm_fp8.variant(), 1);

        // ROCm/Flash/paged_kv (hypothetical variant 0x02)
        let rocm_flash_paged = Opcode::new(2, 4, 2);
        assert_eq!(rocm_flash_paged.0, 0x2402);
        assert_eq!(rocm_flash_paged.backend(), 2);
        assert_eq!(rocm_flash_paged.family(), 4);
        assert_eq!(rocm_flash_paged.variant(), 2);
    }

    #[test]
    fn gemma_variant_opcodes() {
        // Norm-prologue bf16 GEMM: identical variant byte on both backends, only
        // the backend nibble differs (generic 0x0321, CUDA 0x1321, ROCm 0x2321).
        assert_eq!(Opcode::VARIANT_NORM_BF16, 0x21);
        assert_eq!(Opcode::new(0, Opcode::FAMILY_GEMM, Opcode::VARIANT_NORM_BF16).0, 0x0321);
        assert_eq!(Opcode::new(1, Opcode::FAMILY_GEMM, Opcode::VARIANT_NORM_BF16).0, 0x1321);
        assert_eq!(Opcode::new(2, Opcode::FAMILY_GEMM, Opcode::VARIANT_NORM_BF16).0, 0x2321);

        // A few more Gemma generic opcodes referenced in the plan.
        assert_eq!(Opcode::new(0, Opcode::FAMILY_ROW, Opcode::VARIANT_ROW_NORMROPESCALE_BF16).0, 0x0541);
        assert_eq!(Opcode::new(0, Opcode::FAMILY_ROW, Opcode::VARIANT_ROW_SWIGLU_BF16).0, 0x0507);
        assert_eq!(Opcode::new(0, Opcode::FAMILY_LAYOUT, Opcode::VARIANT_LAYOUT_GATHER_SCALE_BF16).0, 0x0642);
        assert_eq!(Opcode::new(0, Opcode::FAMILY_FLASH, Opcode::VARIANT_FLASH_DECODE_BF16).0, 0x0403);
    }

    #[test]
    fn variant_is_reduce_labels() {
        // Reduce-shaped Row variants.
        assert!(Opcode::variant_is_reduce(Opcode::VARIANT_GOLDEN)); // 0x00
        assert!(Opcode::variant_is_reduce(Opcode::VARIANT_ROW_RMS_BF16)); // 0x04
        // Pointwise / fused-shaped Row variants.
        assert!(!Opcode::variant_is_reduce(0x01)); // pointwise golden
        assert!(!Opcode::variant_is_reduce(Opcode::VARIANT_ROW_RESIDUAL_ADD_BF16));
        assert!(!Opcode::variant_is_reduce(Opcode::VARIANT_ROW_SWIGLU_BF16));
        assert!(!Opcode::variant_is_reduce(Opcode::VARIANT_ROW_NORMROPE_BF16));
        assert!(!Opcode::variant_is_reduce(Opcode::VARIANT_ROW_NORMROPESCALE_BF16));
    }

    fn sample() -> Program {
        Program {
            insts: vec![
                Inst {
                    resource: ResourceKind::Dma,
                    unit: 0,
                    index: 1,
                    body: Body::Dma {
                        load: true,
                        bytes: 4096,
                        slot: 2,
                        tensor: 7,
                        kind: KIND_UNSPECIFIED,
                        access: ACCESS_READ,
                    },
                    wait: vec![],
                    succ: vec![3],
                },
                Inst {
                    resource: ResourceKind::Sm,
                    unit: 0,
                    index: 5,
                    body: Body::Gemm {
                        coord: [1, 2],
                        m: 512,
                        n: 512,
                        k: 512,
                        bm: 128,
                        bn: 256,
                        bk: 64,
                        out: 0,
                        tmem: SLOT_NONE,
                        variant: Opcode::VARIANT_BF16,
                    },
                    wait: vec![3, 4],
                    succ: vec![9],
                },
                Inst {
                    resource: ResourceKind::Host,
                    unit: 0,
                    index: 0,
                    body: Body::Host,
                    wait: vec![],
                    succ: vec![],
                },
                // A host-class SAMPLE packet: gated on the logits counter (9),
                // reads slot 0 (logits) → writes slot 11 (tokens).
                Inst {
                    resource: ResourceKind::Host,
                    unit: 0,
                    index: 0,
                    body: Body::Token {
                        in_slot: 0,
                        out_slot: 11,
                        kind: Opcode::TOKEN_SAMPLE_GREEDY,
                        vocab: 128000,
                        arg: 0,
                    },
                    wait: vec![9],
                    succ: vec![],
                },
            ],
            // Every wait/succ id above must appear here — decode rejects ids
            // outside the counter table (the runtime pool sizes from it).
            counters: vec![
                Counter { id: 3, threshold: 4, scope: 1, _pad: [0; 3] },
                Counter { id: 4, threshold: 1, scope: 1, _pad: [0; 3] },
                Counter { id: 9, threshold: 1, scope: 2, _pad: [0; 3] },
            ],
            bucket_id: 42,
            plan_gen: 7,
            flags: 0,
        }
    }

    #[test]
    fn round_trips() {
        let p = sample();
        assert_eq!(Program::decode(&p.to_bytes()).unwrap(), p);
    }

    #[test]
    fn dma_kind_access_round_trip() {
        // Distinct non-trivial kind/access values must survive encode → decode
        // in the right bytes (guards the DmaBody `_pad` → kind/access change).
        let prog = Program {
            insts: vec![Inst {
                resource: ResourceKind::Dma,
                unit: 0,
                index: 0,
                body: Body::Dma {
                    load: false,
                    bytes: 128,
                    slot: 3,
                    tensor: 9,
                    kind: 5, // BufKind::KvCache
                    access: ACCESS_WRITE,
                },
                wait: vec![],
                succ: vec![],
            }],
            counters: vec![],
            bucket_id: 0,
            plan_gen: 0,
            flags: 0,
        };
        let back = Program::decode(&prog.to_bytes()).unwrap();
        assert_eq!(back, prog);
        match back.insts[0].body {
            Body::Dma { kind, access, .. } => {
                assert_eq!(kind, 5);
                assert_eq!(access, ACCESS_WRITE);
            }
            _ => panic!("expected DMA body"),
        }
    }

    /// `decode` must never panic or read out of bounds on a malformed/truncated
    /// stream — every byte prefix of a valid stream returns `Ok` or `Err`.
    #[test]
    fn decode_rejects_truncated_streams() {
        let bytes = sample().to_bytes();
        // Every prefix shorter than the full stream must fail cleanly (the full
        // length round-trips, tested above).
        for len in 0..bytes.len() {
            let r = Program::decode(&bytes[..len]);
            assert!(r.is_err(), "prefix of len {len} should be rejected");
        }
    }

    /// A header advertising a huge `n_insts` / `n_counters` must not over-read or
    /// over-allocate — it's bounded by the actual buffer length.
    #[test]
    fn decode_rejects_oversized_counts() {
        let mut bytes = sample().to_bytes();
        // n_insts at offset 8, n_counters at offset 12 (see STREAM_HEADER layout).
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Program::decode(&bytes).is_err());
        let mut bytes = sample().to_bytes();
        bytes[12..16].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert!(Program::decode(&bytes).is_err());
    }

    /// Garbage of various lengths never panics (returns `Ok` or `Err`).
    #[test]
    fn decode_handles_garbage() {
        for len in [0usize, 1, 4, 19, 20, 21, 64, 257, 1024] {
            let junk: Vec<u8> = (0..len).map(|j| (j as u8).wrapping_mul(31)).collect();
            let _ = Program::decode(&junk); // must not panic / UB
        }
        // A valid-looking header (right magic/version) but no body bytes.
        let mut hdr = Vec::new();
        hdr.extend_from_slice(&MAGIC.to_le_bytes());
        hdr.extend_from_slice(&VERSION.to_le_bytes());
        hdr.extend_from_slice(&0u16.to_le_bytes()); // bucket_id
        hdr.extend_from_slice(&100u32.to_le_bytes()); // n_insts = 100 (none present)
        hdr.extend_from_slice(&0u32.to_le_bytes()); // n_counters
        hdr.extend_from_slice(&0u16.to_le_bytes()); // plan_gen
        hdr.extend_from_slice(&0u16.to_le_bytes()); // flags
        assert!(Program::decode(&hdr).is_err());
    }

    #[test]
    fn records_stay_4_aligned_in_stream() {
        // Walk the stream the way a kernel would; every record/u32 offset is
        // 4-aligned, so casts are valid device loads.
        let bytes = sample().to_bytes();
        assert_eq!(bytes.len() % 4, 0);
        let mut i = STREAM_HEADER_SIZE;
        // Stream header is 20 B = 4-aligned ✓
        assert_eq!(i % 4, 0);
        for ins in &sample().insts {
            assert_eq!(i % 4, 0, "record not 4-aligned");
            i += size_of::<Header>();
            let mut o = Vec::new();
            ins.body.push(&mut o);
            i += o.len();
            i += (ins.wait.len() + ins.succ.len()) * 4;
        }
    }

    #[test]
    fn stream_header_is_20_bytes() {
        let p = Program::default();
        let b = p.to_bytes();
        // Empty program: just the 20-byte stream header
        assert_eq!(b.len(), STREAM_HEADER_SIZE);
    }

    #[test]
    fn bucket_id_and_plan_gen_persist() {
        let mut p = sample();
        p.bucket_id = 0xBEEF;
        p.plan_gen = 0xCAFE;
        p.flags = 0x0001;
        let decoded = Program::decode(&p.to_bytes()).unwrap();
        assert_eq!(decoded.bucket_id, 0xBEEF);
        assert_eq!(decoded.plan_gen, 0xCAFE);
        assert_eq!(decoded.flags, 0x0001);
    }
}
