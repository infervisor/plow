//! The AMD/gfx950 serving engine — ported from `runtime/tests/gemma4_chat.c`.
//!
//! This is deliberately NOT [`crate::exec::gpu`] with a type parameter. The two
//! drivers differ in kind, not in naming, and every one of those differences is
//! load-bearing:
//!
//! | | CUDA (`exec::gpu`) | AMD (here) |
//! |---|---|---|
//! | dispatch | ONE cooperative launch | **n_seg launches, ONE drain** |
//! | co-residency | `cuLaunchCooperativeKernel` refuses a bad grid | enforced at BUILD time; a bad object is `INVALID_ISA` |
//! | kernels | one interpreter | **three** — prefill / decode / flash, plus `_gq` twins |
//! | scheduler | fixed | **per-phase**: global queue or static per-CU streams |
//! | LDS | dynamic, `cuFuncSetAttribute` opt-in | **static** in the object (144 KiB), `dynamic_lds = 0` |
//!
//! A generic engine would have to abstract over all five, and `AppState` holds
//! `Arc<Mutex<GpuEngine>>` non-generically, so a type parameter there infects
//! every HTTP handler. Two modules over one shared [`crate::exec::device_api`]
//! is the smaller, safer shape.
//!
//! # The traps, carried across with their reasons
//!
//! Each of these is a silent-corruption trap in the reference driver, and the
//! reason is why the code looks the way it does:
//!
//! * **`n_seg` is DERIVED, not read.** There is no blob field. It is
//!   `max(stream[].seg) + 1`, and a segment is wave-class 4 iff any entry in it
//!   points at a flash-prefill instruction.
//! * **Counters are zeroed ONCE per dispatch group, never per segment.** A
//!   segment's producers ran in an *earlier launch*; re-zeroing between segments
//!   unsatisfies them and the next segment waits forever.
//! * **The single drain is only correct because the AQL header carries the
//!   barrier bit.** That bit is what chains segment k+1 behind segment k with no
//!   host round-trip. Drop it and segmented dispatch is silently racy.
//! * **`gq_cursor` is per-segment strided**, not one shared word — same
//!   lifecycle as the counters, for the same reason.
//! * **`in.ids` is NOT uploaded per step.** The device's argmax wrote the
//!   sampled token there at the end of the previous launch, which is exactly
//!   where this step's embed reads it. Uploading a host copy would overwrite the
//!   device's own answer with a stale one.
//!
//! # Three bugs from the reference, fixed here rather than reproduced
//!
//! 1. The wave-class test named only `FlashPrefill`, not `FlashPrefillFp8`, so
//!    an fp8-KV packet ran its flash segments on the 8-wave interpreter.
//! 2. The prefill patch loop likewise missed the fp8 opcodes, so chunked fp8-KV
//!    prefill ran unpatched.
//! 3. `PLOW_SEG_OFF` rewrote `stream[].seg` without rewriting `gq_seg_ofs`, so
//!    under the global queue it ran one segment's window and returned having
//!    done almost nothing. (Independently hit on the kernel side: a 0.9 ms run
//!    that produced no output.) Here the override refuses the combination
//!    instead of silently truncating.
//!
//! # What still stands between this and `plowrt serve` on AMD
//!
//! This engine is SINGLE-SEQUENCE: [`AmdEngine::prefill`] then a chain of
//! [`AmdEngine::decode_step`]. `serve` does not want that. `serve::mux` drives a
//! **batched, slotted, continuously-batching** engine, and the surface it calls
//! on `exec::gpu::GpuEngine` is twenty methods that are FEATURES, not forwards:
//!
//! * slots and continuous batching — `batch`, `begin_slot`, `attach_prompt`,
//!   `attached_rows`, `step_slots`, `step_slots_sampled`, `multi_step`,
//!   `multistep_quantum`;
//! * batched prefill — `prefill_batched`, `prefill_chunk`, `pf_batch_enabled`,
//!   `pf_max_rows`, `pf_pack_budget`, `has_prefill`;
//! * sampling and logits — `dev_sample_enabled`, `logits_row`,
//!   `take_logits_buf`, `return_logits_buf`, `stop_ids`.
//!
//! So the `AppState` change (an object-safe façade at
//! `serve/mod.rs`'s `RwLock<FxHashMap<String, Arc<Mutex<GpuEngine>>>>`) is
//! genuinely plumbing, but it is plumbing in FRONT of engine work, not instead
//! of it.
//!
//! And there is a second gate, in the ASSET rather than the code: the compiler
//! emits the decode program with `t == PLOW_DECODE_BATCH`, and
//! `build-amd/g31b/model.pkt` carries **`t == 1`**. Batch > 1 is not merely
//! unimplemented here, it is not expressible with that blob — `in.kvlen` is
//! `batch * 4` bytes wide and the decode stream is compiled for one row. A
//! concurrency sweep through the muxer therefore needs a blob recompiled with a
//! larger decode batch BEFORE any of the above is worth writing.
//!
//! **SUPERSEDED for K3.** The recompiled blob exists and batch > 1 ships: a
//! sequence-rows decode program carries `t == B` independent sequences (per-slot
//! KDA state, `PLOW_KDA_F_SEQ_ROWS`, `PLOW_GEMV_MM`), `serve` drives it through
//! `decode_step_batched`, and `scripts/k3_batch_gate.sh` passes at B=4 on K3 at
//! TP8. Measured 91.3 tok/s aggregate at B=16 against 34.5 at B=1. What the
//! paragraph above still describes correctly is any blob compiled at `t == 1` —
//! the refusal keyed on the carrier in `AmdEngine::load` is what tells them apart.
//! What remains genuinely absent on this backend is chunked/interleaved prefill,
//! VMM on TP, and prefix sharing; see
//! `perf-data/k3-throughput-architecture-review.md`.
//!
//! # Measured: the concurrency axis (Gemma-4 31B, one MI355X, ctx 1024)
//!
//! | config | batch 1 | batch 8 | degradation | b8 aggregate |
//! |---|---|---|---|---|
//! | bf16 | 16.63 ms / 60.1 t/s | 24.31 ms | 1.46x | 329.1 tok/s |
//! | w8a8+fp8-KV | 13.24 ms / 75.6 t/s | 18.84 ms | 1.42x | 424.7 tok/s |
//!
//! Degradation is FLAT across a 15% packet-count difference (776 packets at T=8
//! for fp8 against 676 for bf16 — the QKV-fusion loss). So the per-packet gate
//! cost does NOT scale with the batch in a way that defeats amortisation, which
//! falsifies the obvious reading of the bf16 result. Whatever the decode gap
//! is, more packets do not make batching worse.
//!
//! Against vLLM this is a lead but not a structural one: plow is slightly ahead
//! at both points (13.24 vs 13.52 ms at concurrency 1) and the two degrade at
//! essentially the same rate (vLLM's 1->8 interpolates to ~1.40x against plow's
//! 1.42x). That is unlike the long-context axis, where the lead WIDENS. Worth
//! keeping written down, because the natural assumption is that a resident
//! megakernel with counter-gated packets should also win on concurrency, and on
//! this evidence it does not.
//!
//! Batched decode is correct in both precisions — batch-8 sequence 0 reproduces
//! batch-1 token for token. The two PRECISIONS diverge from each other from the
//! first token (122826 vs 236844), which is fp8 KV quantising what attention
//! reads, not a defect.
//!
//! Keep [`AmdEngine`] and the `amd-bench` CLI working regardless of what lands
//! on top: they are what produced the matched-context measurements against the
//! reference driver with no HTTP layer in the way, and that is exactly when a
//! low-level vehicle earns its keep.

use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use packet::dev::{DevInst64, DevOp, DevProgram};
use packet::devbuild::static_seg_ofs;

use crate::asset::devblob::{DevBlob, DevProg};
use crate::device::hsa::{HsaBackend, HsaKernel, HsaPinned};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::memory::vmm::{VmmGeometry, VmmKv, VmmOps, WeightSlab};
use crate::{Result, RuntimeError};

/// `kv.{l}.k` / `kv.{l}.v` → `(layer, 0|1)`. Scales and every other `kv.*`
/// spelling return `None` and stay on the ordinary allocator.
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

/// u32 slots per counter (`PLOW_CTR_STRIDE`), i.e. one 128 B cache line.
const CTR_STRIDE_U32: usize = 32;

/// DOUBLE-BUFFER the counter/cursor banks so the per-dispatch zeroing overlaps
/// GPU execution instead of standing in front of it. `--amd-ctr-dbuf false` /
/// `PLOW_CTR_DBUF=0` reverts to the single bank cleared synchronously before
/// every launch.
///
/// The measured case for it, on this box (MI300X, Gemma-4-12B fp8, ctx 4096,
/// `PLOW_DSTEP_LOG=1`): the synchronous `rearm` is **56 µs of the 11.69 ms
/// token**, 0.48%, and it is 57% of the entire host phase (99 µs, 0.85%).
/// It costs that much because the clear is two blocking
/// `hsa_amd_memory_async_copy` round trips over `n_counter * 128 B`, and it
/// sits between "host has staged this token" and "the GPU may start".
///
/// The alternative the review priced — a kernel prologue that self-clears —
/// removes the same 56 µs from the host but ADDS ~2 µs to the GPU critical
/// path and needs a kernel/ABI change. This needs neither: the copy still
/// happens, it just happens while 304 CUs are busy, over the SDMA engine that
/// the persistent megakernel does not contend for.
fn ctr_dbuf() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| crate::config::RuntimeConfig::get().amd.ctr_dbuf)
}

/// `sizeof(PlowTraceRec)` (`runtime/common/dev_isa.h`, static-asserted at 40).
/// One record per (workgroup, packet), slotted at `stream_ofs[cu] + pc`.
const TRACE_REC_BYTES: usize = 40;

/// Wave-class 8 is `PLOW_WG_WAVES` = 8 waves of 64.
const WG_THREADS_8: u32 = 8 * 64;
/// The flash object is built 4-wave. Dispatching it at 512 threads is an
/// `INVALID_ISA`, not a slowdown.
const WG_THREADS_4: u32 = 4 * 64;

/// Host dispatch semantics for a program whose stream carries `seg` tags.
///
/// L2 placement reuses those tags for domains that drain concurrently inside
/// one launch. Only ordinary wave-class segments are separate host launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProgramDispatch {
    L2Domains(u32),
    WaveSegments(usize),
}

impl ProgramDispatch {
    pub(crate) fn classify(l2_domains: u32, n_segments: usize) -> Self {
        if l2_domains != 0 {
            Self::L2Domains(l2_domains)
        } else {
            Self::WaveSegments(n_segments)
        }
    }

    pub(crate) fn launches(self) -> usize {
        match self {
            Self::L2Domains(_) => 1,
            Self::WaveSegments(n) => n,
        }
    }
}

/// Sanity bound on `seg`, so a corrupt stream cannot make the host allocate
/// unboundedly. Was 512 — the width of the reference driver's `seg_class[512]`.
///
/// Raised to the full `u16` range because `PLOW_SEG_PER_OP` (see
/// `packet::devbuild::Builder::finish`) emits one segment per op to measure
/// host-side AQL chaining against the counter protocol, and K3's decode program
/// alone is 2459 ops. `seg` is a `u16` in `StreamEnt`, so 65536 is the real
/// representable ceiling and anything under it is arbitrary; the allocations
/// keyed off `n_seg` are `[n_cu][n_seg+1]` window bounds and `[n_seg]` wave
/// classes, i.e. bounded and linear.
const MAX_SEG: u32 = u16::MAX as u32 + 1;

/// Which scheduler a phase runs: the global work queue or static per-CU
/// streams. Bit-exact to each other — same op kernels, same tiles, same
/// registers; only the scheduling loop differs — so this is purely a
/// performance choice and safe to flip per phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sched {
    /// Static per-CU streams: each workgroup walks its own stream and skips
    /// entries whose `seg` is not the current one.
    Static,
    /// Global queue: one shared fetch-add cursor over an op-major permutation,
    /// windowed per segment by `gq_seg_ofs`.
    GlobalQueue,
}

impl Sched {
    fn suffix(self) -> &'static str {
        match self {
            Sched::Static => "",
            Sched::GlobalQueue => "_gq",
        }
    }
}

/// Which interpreter object a phase needs. Prefill and decode are separate
/// objects because **register allocation is per-kernel** — one object compiled
/// for both would be allocated for the worse of the two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
    /// The flash-prefill segments, which run at 4 waves.
    Flash,
}

impl Phase {
    fn symbol_base(self) -> &'static str {
        match self {
            Phase::Prefill => "plow_interp",
            Phase::Decode => "plow_interp_dec",
            Phase::Flash => "plow_interp_flash",
        }
    }

    fn object_stem(self) -> &'static str {
        match self {
            Phase::Prefill => "interp_prefill",
            Phase::Decode => "interp_decode",
            Phase::Flash => "interp_flash",
        }
    }
}

/// The numeric variants the objects are built for. Selected by scanning the
/// program's opcodes, because the packet is what decides which kernels must
/// exist — not a flag someone can forget to pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Variant {
    #[default]
    Bf16,
    /// fp8 weights (`GemvFp8`).
    Fp8,
    /// fp8 KV cache (`FlashDecodeFp8`). Supersedes [`Variant::Fp8`] — an fp8-KV
    /// packet is also fp8-weight — and changes BOTH objects, not just decode.
    Fp8Kv,
}

impl Variant {
    fn infix(self) -> &'static str {
        match self {
            Variant::Bf16 => "",
            Variant::Fp8 => "_fp8",
            Variant::Fp8Kv => "_fp8kv",
        }
    }

    /// Decide from the compiled programs. Scans every program, because a
    /// prefill bucket and the decode program can disagree and the union is what
    /// must be loadable.
    pub fn detect(progs: &[DevProg]) -> Variant {
        let mut v = Variant::Bf16;
        for p in progs {
            for i in &p.insts {
                if i.op == DevOp::FlashDecodeFp8 as u16
                    || i.op == DevOp::FlashMlaDecodeFp8 as u16
                    || i.op == DevOp::FlashMlaPrefillFp8 as u16
                {
                    return Variant::Fp8Kv;
                }
                if i.op == DevOp::GemvFp8 as u16 {
                    v = Variant::Fp8;
                }
            }
        }
        v
    }
}

/// Which MLA/MoE-prefill arms a PREFILL object must have been built with.
///
/// A separate axis from [`Variant`] (precision), because the shipped objects
/// never compose the two — `interp_prefill_mla{,_moe}.elf` are bf16, built by
/// `scripts/build_gfx950.sh`'s `PLOW_MLA_PREFILL`/`PLOW_MOE_PREFILL` flags, and
/// no fp8+mla object exists. Selected the same way `Variant` is: by scanning
/// the packet's own opcodes, never a flag someone could forget to pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrefillArm {
    #[default]
    None,
    /// `FlashMlaPrefill` / `FlashGatherPrefill` / `MlaMergeFold` — MLA attention,
    /// no MoE FFN.
    Mla,
    /// The `Mla` arms AND the grouped-MoE prefill ops (83-87). Supersedes
    /// [`PrefillArm::Mla`] — a whole-layer GLM/Kimi/DeepSeek prefill packet
    /// needs both, and `scripts/build_gfx950.sh`'s `PLOW_MOE_PREFILL=1` always
    /// turns MLA on with it (there is no moe-without-mla object).
    MlaMoe,
    /// **Kimi-K3.** The `PLOW_K3` arms — `AttnRes` (104), `SituGlu` (105),
    /// `MlaOutGate` (106) and the KDA mixer (99-103) — which live in NEITHER of
    /// the objects above. Supersedes both: `_hs_ax_mla_k3` composes
    /// `PLOW_MLA_PREFILL` with `PLOW_K3`, because K3's full-attention layers are
    /// MLA.
    ///
    /// It is a SEPARATE AXIS from precision and it has to be, for the reason
    /// this enum exists at all: without it a K3 blob resolves to
    /// `interp_prefill_mla_moe.elf`, which was compiled with no `PLOW_K3` and
    /// therefore has no `case` for any of those five opcodes. This
    /// interpreter's dispatch `default:` writes NOTHING, so every AttnRes mix,
    /// every `situ` GLU and the entire KDA recurrence would be skipped in
    /// silence and the model would produce fluent output from a graph missing
    /// two thirds of its layers.
    K3,
    /// [`PrefillArm::K3`] plus the grouped-MoE prefill chain, at bf16 or
    /// block-fp8 experts.
    K3Moe,
    /// [`PrefillArm::K3Moe`] with **MXFP4** experts, which need the A4W4 body.
    ///
    /// A SEPARATE ARM and not a detail of the one above, because the two are
    /// different objects: ops 85/86 select their MXFP4 body on
    /// `i[3] == PLOW_MOE_ENC_MXFP4`, and that body is compiled only under
    /// `PLOW_MOE_PF_A4W4`. Reading the encoding out of the packet is the only
    /// way to tell them apart — and collapsing them would be wrong in BOTH
    /// directions: an mxfp4 packet on the plain object takes `moe_pf_refuse`
    /// (loud, but a dead run), and a bf16-expert packet on the a4w4 object gets
    /// arms it does not use in an object 140 KB larger. The encoding is a
    /// packet field, so it is not a guess.
    K3MoeA4w4,
    /// K3 DSpark's seven-query block. It uses the grouped dense-FFN packets but replaces the
    /// ordinary dense BF16 MLA decode specialization with a non-causal multi-query body.
    DSpark,
}

impl PrefillArm {
    fn infix(self) -> &'static str {
        match self {
            PrefillArm::None => "",
            PrefillArm::Mla => "_mla",
            PrefillArm::MlaMoe => "_mla_moe",
            // `_k3` already implies the MLA prefill arms — see `_hs_ax_mla_k3`
            // in runtime/CMakeLists.txt — so it does not stack with `_mla`.
            PrefillArm::K3 => "_k3",
            PrefillArm::K3Moe => "_k3_moe",
            PrefillArm::K3MoeA4w4 => "_k3_moe_a4w4",
            PrefillArm::DSpark => "_dspark",
        }
    }

    /// Decide from the compiled programs. Scans EVERY program, not
    /// `progs.last()` (the decode program) — the MLA/MoE prefill opcodes this
    /// selects on appear ONLY in the prefill bucket programs, so scanning just
    /// the decode program would always see `None` and reproduce the bug this
    /// axis exists to fix.
    pub fn detect(progs: &[DevProg]) -> PrefillArm {
        let mut mla = false;
        let mut moe = false;
        let mut a4w4 = false;
        let mut k3 = false;
        let mut dspark = false;
        for p in progs {
            for i in &p.insts {
                let op = i.op;
                if op == DevOp::FlashMlaDecode as u16
                    && i.t[7] == packet::dev::TENSOR_NONE16
                    && i.i[6] > 1
                {
                    dspark = true;
                }
                if op == DevOp::FlashMlaPrefill as u16
                    || op == DevOp::FlashGatherPrefill as u16
                    || op == DevOp::MlaMergeFold as u16
                {
                    mla = true;
                } else if op == DevOp::MoeRouterTopkPf as u16
                    || op == DevOp::MoeAlignPf as u16
                    || op == DevOp::MoeGroupGluPf as u16
                    || op == DevOp::MoeGroupDownPf as u16
                    || op == DevOp::MoeCombinePf as u16
                {
                    moe = true;
                    // The two grouped GEMMs carry the WEIGHT ENCODING in `i[3]`
                    // (`MoeEnc::PREFILL_SLOT`; `n_exp` is `i[2]` there, so `i[3]`
                    // was free). `2` is `PLOW_MOE_ENC_MXFP4`, whose body lives
                    // behind `PLOW_MOE_PF_A4W4` — a different object, not a
                    // different immediate. The other three ops in the chain do
                    // not carry it, so only these two are asked.
                    if (op == DevOp::MoeGroupGluPf as u16 || op == DevOp::MoeGroupDownPf as u16)
                        && i.i[MOE_PF_ENC_SLOT] == MOE_ENC_MXFP4
                    {
                        a4w4 = true;
                    }
                } else if op == DevOp::AttnRes as u16
                    || op == DevOp::SituGlu as u16
                    || op == DevOp::MlaOutGate as u16
                    || op == DevOp::KdaStateStep as u16
                    || op == DevOp::KdaStateStepG as u16
                    || op == DevOp::KdaConv as u16
                    || op == DevOp::KdaConv3 as u16
                    || op == DevOp::KdaGatedNorm as u16
                {
                    // Scanned on EVERY program, decode included. K3's block ops
                    // are in BOTH buckets by construction (an AttnRes present
                    // only in decode would make the two phases compute
                    // different models), so a decode-only K3 blob still selects
                    // the K3 objects — which is what makes `interp_decode_k3`
                    // reachable at all.
                    k3 = true;
                }
            }
        }
        if dspark {
            return PrefillArm::DSpark;
        }
        match (k3, moe, a4w4, mla) {
            (true, true, true, _) => PrefillArm::K3MoeA4w4,
            (true, true, false, _) => PrefillArm::K3Moe,
            (true, false, _, _) => PrefillArm::K3,
            // The non-K3 families do NOT branch on the encoding here, and that is a
            // known gap rather than a decision: `interp_prefill_mla_moe_a4w4{,_full}`
            // are built and nothing selects them, so an mxfp4 GLM/Kimi-K2 packet takes
            // `moe_pf_refuse` today. Loud, so it is not this axis's silent failure —
            // but it is the same fix, one arm over.
            (false, true, _, _) => PrefillArm::MlaMoe,
            (false, false, _, true) => PrefillArm::Mla,
            _ => PrefillArm::None,
        }
    }
}

/// The code-object filename for a (phase, variant, prefill-arm, scheduler).
///
/// The flash object follows the PREFILL scheduler, because a flash segment IS a
/// prefill segment — pairing it with the decode choice would load an object
/// whose scheduling loop does not match the stream it is handed.
pub fn object_name(phase: Phase, variant: Variant, arm: PrefillArm, sched: Sched) -> String {
    // There is no separate fp8-weight flash object; flash only varies on KV.
    let variant = match (phase, variant) {
        (Phase::Flash, Variant::Fp8) => Variant::Bf16,
        _ => variant,
    };
    // The mla/mla_moe objects are a PREFILL-only build (`interp_prefill_mla{,_moe}{,_gq}.elf`
    // — no decode or flash twin exists), so the axis only applies there.
    //
    // K3 IS THE EXCEPTION, and it is not a special case so much as the axis behaving as it should:
    // `PLOW_K3` is a MODEL axis, not a prefill-kernel axis, and `interp_decode_k3.elf` does exist.
    // A K3 decode packet handed the plain `interp_decode.elf` has no `case` for AttnRes, situ, the
    // output gate or the KDA recurrence, and this interpreter's `default:` writes nothing.
    let arm = match (phase, arm) {
        (Phase::Decode, PrefillArm::DSpark) => PrefillArm::DSpark,
        (Phase::Prefill | Phase::Flash, PrefillArm::DSpark) => PrefillArm::None,
        (Phase::Prefill, a) => a,
        (Phase::Decode, PrefillArm::K3 | PrefillArm::K3Moe | PrefillArm::K3MoeA4w4) => {
            PrefillArm::K3
        }
        // There is no K3 flash object: K3 is NoPE MLA + KDA and emits no `FlashPrefill` at any
        // head dim, so no packet can reach this phase with a K3 arm.
        _ => PrefillArm::None,
    };
    format!(
        "{}{}{}{}.elf",
        phase.object_stem(),
        variant.infix(),
        arm.infix(),
        sched.suffix()
    )
}

/// The kernel symbol inside that object. `arch` is the ISA name (`gfx950`).
pub fn symbol_name(phase: Phase, sched: Sched, arch: &str) -> String {
    format!("{}_{}{}", phase.symbol_base(), arch, sched.suffix())
}

/// What a `build.json` `gfx950.requires` flag looks like in the SYMBOL TABLE of
/// the PREFILL code object it turns arms on in.
///
/// This is the AMD answer to the CUDA `plow_packet_hash` stamp
/// ([`super::gpu::GpuEngine::check_packet_pairing`]). There is no stamp here —
/// the objects are built by a plain `hipcc` line with no packet in scope — so the
/// only honest signal is what the object actually CONTAINS. `hipcc` leaves every
/// `__device__` function it did not fully inline in `.symtab` as a LOCAL FUNC
/// (verified on the shipped set: `interp_decode.elf` carries
/// `_Z16d_mla_merge_foldILi512ELi256EE…`, `_Z11d_o_uv_foldILi512EE…` and the whole
/// `d_moe_*` family as real symbols), and an arm compiled out by `#if` leaves
/// nothing at all.
///
/// ANY of a flag's markers is enough. Which particular helper survives inlining
/// is a compiler decision and must not be load-bearing; what is load-bearing is
/// that a `#if`-disabled block leaves NONE of them. That asymmetry is why the
/// test is "no marker at all ⇒ the arm is absent" and never "this exact symbol
/// must exist".
///
/// PREFILL only. The flash object is the same op set built at 4 waves, but MLA
/// prefill does not run there — [`derive_segments`] marks a segment class 4 only
/// for `FlashPrefill`/`FlashPrefillFp8` — so a flash object legitimately built
/// without these arms must not be refused.
/// The `i[]` slot the grouped MoE prefill ops (85/86) carry their WEIGHT ENCODING in.
///
/// Mirrors `devgen::mla::MoeEnc::PREFILL_SLOT`. It is NOT the decode ops' slot — those predate the
/// field and already use `i[3]` for `n_exp`, so they carry it in `i[6]`. Mirrored rather than
/// shared because `plowrt` does not depend on `devgen`; the two are pinned together by
/// `prefill_arm_detect_selects_the_right_variant`, which builds packets with this literal.
const MOE_PF_ENC_SLOT: usize = 3;

/// `PLOW_MOE_ENC_MXFP4` (`runtime/amd/op_moe.h`) — the encoding whose grouped body is compiled
/// only under `PLOW_MOE_PF_A4W4`, i.e. the one that selects a different OBJECT rather than a
/// different branch.
const MOE_ENC_MXFP4: u32 = 2;

const PREFILL_ARM_MARKERS: &[(&str, &[&str])] = &[
    // `#if PLOW_MLA_PREFILL` in runtime/amd/interp.hip gates ops 51/55 (via
    // `exec_flash_mla_prefill` -> `d_flash_mla_decode`) AND the latent epilogue
    // ops 53/54, which is why the fold names count as proof of the same flag.
    (
        "PLOW_MLA_PREFILL",
        &["d_flash_mla", "d_mla_merge_fold", "d_o_uv_fold"],
    ),
    // `#if PLOW_MOE_PREFILL` gates ops 83-87. The `_pf` suffix is what separates
    // them from the decode-side `d_moe_expert_*`/`d_moe_group_*_fp8_blk`, which a
    // decode object carries whether or not this flag was set.
    (
        "PLOW_MOE_PREFILL",
        &[
            "d_moe_router_topk_pf",
            "d_moe_align_pf",
            "d_moe_group_glu_pf",
            "d_moe_group_down_pf",
            "d_moe_combine_pf",
        ],
    ),
    // Runtime-flag arms carried in packet i[7] on ops 85/86/87 (unconditional markers in
    // op_moe.h — any object built since the arms landed has them). An older object given a
    // part16 packet stores f32 into a HALF-SIZED part buffer (silent heap overrun); given an
    // a8 packet it matmuls fp8 bytes as bf16. Both must refuse at load.
    ("PLOW_MOE_PF_PART16", &["plow_moe_pf_part16_arm"]),
    // Dense causal KV-split of the V2 MLA prefill (packet i6 on op 51). Older objects run
    // ns packets at the nsplit=1 partial layout while the merge reads ns — refuse.
    ("PLOW_MLA_PF_NS", &["plow_mla_pf_ns_arm"]),
    ("PLOW_MOE_PF_A8", &["plow_moe_pf_a8_arm"]),
    // T11 GLU-into-quant fold (QUANT_FP8 t3=gate t4=up i2=act). The emitter DELETES the `Glu`
    // packet when it folds, and the AMD dispatch ignored t3/t4 for its entire life — so a folded
    // packet quantized an `fu` nothing had written, the FFN output was whatever was in the buffer,
    // the KV cache was wrong, and the model answered fluently and wrongly. Unconditional arm, so
    // the marker is the whole test: no marker => the object predates the fix. Refuse.
    ("PLOW_T11_GLUQUANT", &["plow_t11_gluquant_arm"]),
    // The fused MoE decomposition (packet i[4] on op 86, t[2]/i[0] on op 83). CONDITIONALLY
    // compiled, unlike part16/a8: an object built without -DPLOW_MOE_PF_ATOMIC=1 has no atomic
    // branch at all, so it would take the `part` scatter path with `Cout` pointing at a
    // [T,H]-sized accumulator and scatter k-times past its end. Refuse.
    ("PLOW_MOE_PF_ATOMIC", &["plow_moe_pf_atomic_arm"]),
    // The DETERMINISTIC twin (packet i[5] on op 86, i[4] on op 87). Same silence without it:
    // op 86 would scatter f32 into a [T,H] f64 accumulator and op 87 would read f64 as f32.
    ("PLOW_MOE_PF_DET", &["plow_moe_pf_det_arm"]),
    // `#if PLOW_K3` (runtime/amd/interp.hip) gates ops 99-106 in BOTH buckets — the KDA mixer,
    // AttnRes, `situ` and the MLA output gate. It is the one arm flag that is not prefill-only,
    // and the one whose absence is most completely silent: a K3 packet on an object without it
    // skips every residual mix, every activation and the whole recurrence, and still produces
    // finite, fluent output.
    (
        "PLOW_K3",
        &[
            "d_attn_res",
            "d_situ_glu",
            "d_mla_out_gate",
            "d_kda_state_step",
            "d_kda_conv",
        ],
    ),
];

/// DECODE-object arms, the twin of [`PREFILL_ARM_MARKERS`] for the other object.
///
/// A separate table rather than more rows in that one, because the `requires` list is
/// BLOB-wide: it names arms of both objects, and checking a prefill flag against the decode
/// object would refuse every GLM asset in the tree. Each check therefore only looks at the flags
/// ITS table names and leaves the rest to the other phase.
const DECODE_ARM_MARKERS: &[(&str, &[&str])] = &[
    ("PLOW_KDA_CONV_STEP_DB", &["plow_kda_conv_step_db_arm"]),
    ("PLOW_DSPARK_NONCAUSAL", &["plow_dspark_noncausal_1"]),
    ("PLOW_K3_SPEC_VERIFY", &["plow_k3_spec_verify_1"]),
    // The GLM decode q-rope fold (op 50 t7 = cos, i6 = sin handle, t3 = RAW q_rope). An object
    // built before the arm stages t3 verbatim — an UNROPED query into the flash. Attention still
    // runs, nothing traps, and the model answers fluently and wrongly. This is the same silent
    // class as `PLOW_K3` and it gets the same treatment.
    ("PLOW_GLM_FUSE_ROPE", &["plow_glm_fuse_rope_arm"]),
    // The GLM decode q-norm fold (op 22 t7 = gamma, f0 = eps, t1 = the RAW pre-norm row). An
    // object built without the arm ignores t7 and projects the UNNORMED q_a row. Same silent
    // class, same treatment — and here the marker is genuinely load-bearing rather than a
    // vintage stamp, because the fold is a BUILD AXIS: an unarmed object has no fold body.
    ("PLOW_GLM_FUSE_QNORM", &["plow_glm_fuse_qnorm_arm"]),
];

/// Refuse a DECODE code object that does not carry the arms the packet needs.
///
/// See [`check_prefill_object`] for the full argument — this is that check, on the other object,
/// and it ignores any flag [`DECODE_ARM_MARKERS`] does not name (those belong to the prefill
/// object, which gets its own pass).
fn check_decode_object(syms: &[&str], path: &Path, requires: &[String]) -> Result<()> {
    if syms.is_empty() {
        tracing::warn!(
            object = %path.display(),
            "no ELF symbol table — the packet/object arm check cannot run on this file"
        );
        return Ok(());
    }
    for req in requires {
        let (flag, val) = req.split_once('=').unwrap_or((req.as_str(), "1"));
        if val == "0" {
            continue;
        }
        let Some((_, markers)) = DECODE_ARM_MARKERS.iter().find(|(f, _)| *f == flag) else {
            continue; // not a decode-object arm; the prefill pass owns it
        };
        if !markers.iter().any(|m| syms.iter().any(|s| s.contains(m))) {
            return Err(RuntimeError::Device(format!(
                "packet/object MISMATCH: this packet requires {flag}=1 but the DECODE object {} \
                 was built WITHOUT it — none of {markers:?} is in its symbol table. The arm is a \
                 runtime branch, so an older object does not trap: it reads the packet's operands \
                 under the pre-arm meaning and produces fluent, wrong tokens. Rebuild the decode \
                 object from a tree that has the arm (see `requires` in the build.json beside the \
                 packet), or emit a blob without {flag}.",
                path.display()
            )));
        }
    }
    Ok(())
}

/// `gfx950.requires` from the `build.json` sitting beside the packet, or `None`
/// when there is no manifest.
///
/// Absent is not an error: every asset shipped before `plowc`/`devgen` started
/// writing the manifest has no `build.json`, and those pairings were valid
/// before and stay valid. A manifest that exists and cannot be parsed IS an
/// error — it is the only statement of what the packet needs, and guessing past
/// a broken one is how the check would silently stop checking.
fn build_requires(blob_path: &Path) -> Result<Option<Vec<String>>> {
    let mpath = blob_path.with_file_name("build.json");
    let Ok(raw) = std::fs::read(&mpath) else {
        return Ok(None);
    };
    let man: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| RuntimeError::Device(format!("{}: not valid JSON: {e}", mpath.display())))?;
    // The AMD backend key is ARCH-NAMED by the emitter (`--arch gfx942` writes
    // `backends.gfx942`), and this lookup was hardcoded to gfx950 — which made the whole
    // packet/object arm check INERT for every gfx942 blob: a GLM prefill packet on an object
    // without its arms sailed through and completed with garbage (measured: a
    // PLOW_MOE_PF_PART16 blob ran on a pre-arm object with no complaint — the exact
    // silent-heap-overrun the check exists to refuse). A blob carries exactly one AMD key, so
    // probing both is unambiguous.
    Ok(["/backends/gfx942/requires", "/backends/gfx950/requires"]
        .iter()
        .find_map(|k| man.pointer(k))
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        }))
}

/// Every symbol NAME in an ELF64 object's symbol tables.
///
/// A deliberately minimal reader rather than a dependency: it needs the section
/// headers, the two symbol-table types and their string tables, and every bound
/// is checked so a truncated or foreign file yields an empty list instead of a
/// panic. A file this cannot parse is reported by the CALLER as "unverifiable",
/// never as "the arm is missing" — a parser bug must not become a refusal.
fn elf_symbol_names(img: &[u8]) -> Vec<&str> {
    let mut out = Vec::new();
    let u16at = |o: usize| -> Option<usize> {
        img.get(o..o + 2)
            .map(|b| u16::from_le_bytes(b.try_into().expect("2")) as usize)
    };
    let u32at = |o: usize| -> Option<usize> {
        img.get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4")) as usize)
    };
    let u64at = |o: usize| -> Option<usize> {
        img.get(o..o + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("8")) as usize)
    };
    // ELFCLASS64 (e_ident[4] == 2) and ELFDATA2LSB (e_ident[5] == 1) only. An
    // AMDGPU code object is always both; anything else is not one.
    if img.get(..4) != Some(b"\x7fELF") || img.get(4) != Some(&2) || img.get(5) != Some(&1) {
        return out;
    }
    let (Some(shoff), Some(shent), Some(shnum)) = (u64at(0x28), u16at(0x3a), u16at(0x3c)) else {
        return out;
    };
    if shent < 64 {
        return out;
    }
    let hdr = |i: usize| -> Option<usize> { (i < shnum).then(|| shoff + i * shent) };
    for i in 0..shnum {
        let Some(s) = hdr(i) else { break };
        // sh_type: SHT_SYMTAB = 2, SHT_DYNSYM = 11. Both are read — the local
        // `__device__` helpers this check keys on live in `.symtab`, and the
        // exported kernel in `.dynsym`.
        let Some(ty) = u32at(s + 4) else { break };
        if ty != 2 && ty != 11 {
            continue;
        }
        let (Some(off), Some(size), Some(link), Some(entsz)) =
            (u64at(s + 24), u64at(s + 32), u32at(s + 40), u64at(s + 56))
        else {
            continue;
        };
        // Elf64_Sym is 24 bytes with st_name a u32 at offset 0.
        if entsz < 24 {
            continue;
        }
        let Some(l) = hdr(link) else { continue };
        let (Some(stroff), Some(strsz)) = (u64at(l + 24), u64at(l + 32)) else {
            continue;
        };
        let Some(strtab) = img.get(stroff..stroff.saturating_add(strsz)) else {
            continue;
        };
        for k in 0..size / entsz {
            let Some(nm) = u32at(off + k * entsz) else {
                break;
            };
            let Some(tail) = strtab.get(nm..) else {
                continue;
            };
            let end = tail.iter().position(|&c| c == 0).unwrap_or(tail.len());
            if end > 0 {
                if let Ok(s) = std::str::from_utf8(&tail[..end]) {
                    out.push(s);
                }
            }
        }
    }
    out
}

/// Refuse a prefill code object that does not carry the arms the packet's
/// `build.json` says it needs.
///
/// WHY THIS HAS TO EXIST, and why it hard-errors. The AMD dispatch's `default:`
/// does not trap — an opcode with no `case` falls through and leaves the output
/// buffer exactly as it was. So a GLM-5.2 prefill packet paired with an object
/// built without `PLOW_MLA_PREFILL` does not fail: `FLASH_MLA_PREFILL`,
/// `MLA_MERGE_FOLD` and the grouped-MoE FFN all write nothing, every later op
/// consumes whatever was in those buffers, and the run COMPLETES with fluent
/// wrong tokens. That is the same failure mode the fp8 lm_head regression had
/// (`interp.hip`, the w8a8 prefill note: `GEMM_SMALL` fell to `default:`, logits
/// stayed zero, argmax returned token 0 for every prompt) — found only by
/// reading the disassembly. Nothing about the pairing is visible at runtime, so
/// it has to be refused at load.
///
/// A flag with no marker entry is WARNED about by name, not silently passed and
/// not faked: `PLOW_FP8`/`PLOW_FP8_KV`/`PLOW_MXFP4` select the object FILENAME
/// (see [`Variant`]), so the loader already picks by them, and their remaining
/// content is not separable in the symbol table. Saying so precisely is worth
/// more than a check that cannot fail.
fn check_prefill_object(syms: &[&str], path: &Path, requires: &[String]) -> Result<()> {
    if syms.is_empty() {
        tracing::warn!(
            object = %path.display(),
            "no ELF symbol table — the packet/object arm check cannot run on this file"
        );
        return Ok(());
    }
    let mut unverifiable = Vec::new();
    for req in requires {
        // `FLAG=0` states an arm the object must NOT have been built with. That
        // is not observable as an absence (an object simply has fewer symbols),
        // and the one such flag emitted today — `PLOW_BUCKET_DECODE=0` — is
        // already satisfied by construction: this is the PREFILL object,
        // resolved through the prefill-only `plow_interp_<arch>` symbol, which a
        // decode-bucket build does not export at all.
        let (flag, val) = req.split_once('=').unwrap_or((req.as_str(), "1"));
        if val == "0" {
            continue;
        }
        let Some((_, markers)) = PREFILL_ARM_MARKERS.iter().find(|(f, _)| *f == flag) else {
            unverifiable.push(req.as_str());
            continue;
        };
        if !markers.iter().any(|m| syms.iter().any(|s| s.contains(m))) {
            return Err(RuntimeError::Device(format!(
                "packet/object MISMATCH: this packet requires {flag}=1 but {} was built \
                 WITHOUT it — none of {markers:?} is in its symbol table. The AMD dispatch's \
                 `default:` does not trap, so those ops would write nothing and the prefill \
                 would complete with garbage instead of failing. Rebuild the prefill object \
                 with -D{flag}=1 (see `backends.gfx950.requires` in the build.json beside the \
                 packet), or serve a packet that does not need it.",
                path.display()
            )));
        }
    }
    if !unverifiable.is_empty() {
        // Loud and PRECISE: name the flags, so this reads as "these two were not
        // checked" and never as "everything checked out".
        tracing::warn!(
            object = %path.display(),
            flags = ?unverifiable,
            "NOT verified against the object: these build flags leave no distinguishing \
             symbol (they select the object filename instead). The arm-level flags were checked."
        );
    }
    Ok(())
}

/// The `extern "C" __device__` marker `runtime/amd/op_gemm.h` emits, named for
/// the `PLOW_GEMV_MM` it was compiled at (`plow_gemv_mm_cap_4` ⇒ bucket 4).
///
/// One string, spelled once. `op_gemm_h_emits_the_capacity_marker` reads
/// `op_gemm.h` and asserts the C side still concatenates onto this prefix, so
/// the two halves of the contract cannot drift apart silently — the failure
/// mode this whole check exists to end.
const GEMV_CAP_SYM_PREFIX: &str = "plow_gemv_mm_cap_";

/// The `extern "C" __device__` marker `runtime/amd/op_gemm.h` emits when it was
/// compiled with `PLOW_GEMV_WALK=1` — the single-rung outer loop over `M > MM`.
///
/// Present iff the macro is on, so absence is not ambiguous: every object built
/// before the walk existed, and every object built with it off, carries no such
/// symbol and is a hard-capacity object exactly as before.
const GEMV_WALK_SYM: &str = "plow_gemv_walk_1";

/// `PLOW_GEMV_MAXM` from `runtime/amd/op_gemm.h` — the widest bucket the GEMV
/// path can be instantiated at, and therefore the widest any object can
/// advertise. `scripts/build_gfx950.sh` and `runtime/CMakeLists.txt` both clamp
/// `PLOW_GEMV_MM` to it to satisfy the header's static assert.
///
/// A ceiling, not a pairing: see the use in [`AmdEngine::load`], and
/// [`check_gemv_capacity`] for the half that compares against the object.
const GEMV_MAXM: u32 = 16;

/// Every opcode that reaches a `<PLOW_GEMV_MM>` instantiation on gfx950, i.e.
/// every one whose rows above the object's bucket are DROPPED rather than
/// computed.
///
/// The list is the `d_gemv*` entry points in `runtime/amd/op_gemm.h` that take
/// `PLOW_GEMV_MM` as their template argument, and nothing else: the MoE expert
/// and `d_dense_glu_fp8_blk` arms live in `op_moe.h`/`op_gemm.h` with their own
/// row handling and do not read the bucket. `MoeExpertGlu`-family ops are
/// therefore deliberately ABSENT — adding them would make this refuse packets
/// the bucket cannot hurt.
const GEMV_BUCKET_OPS: &[DevOp] = &[
    DevOp::Gemv,
    DevOp::GemvGlu,
    DevOp::GemvQkv,
    DevOp::GemvFp8,
    DevOp::GemvGluFp8,
    DevOp::GemvFp8Blk,
    DevOp::GemvMxfp4,
    DevOp::GemvGluMxfp4,
    DevOp::GemvQkvMxfp4,
    DevOp::GemvQkvFp8,
];

/// The widest row count any GEMV-family instruction in `progs` asks for.
///
/// `i[0]` is M for all eight of [`GEMV_BUCKET_OPS`] — the dispatch in
/// `runtime/amd/interp.hip` passes `in->i[0]` as the row count to every one of
/// them — so this is the number the object's compiled bucket has to cover.
///
/// Read off the INSTRUCTIONS, not off `prog.t` or `in.kvlen`. Those say how many
/// sequences the program is shaped for; this says what the kernel will actually
/// be handed, and the two are not the same statement. A prefill program is `t`
/// tokens wide and still emits its lm_head GEMV at M=1.
fn required_gemv_m(progs: &[DevProg]) -> u32 {
    progs
        .iter()
        .flat_map(|p| p.insts.iter())
        .filter(|i| GEMV_BUCKET_OPS.iter().any(|&o| o as u16 == i.op))
        .map(|i| i.i[0])
        .max()
        .unwrap_or(0)
}

/// Every opcode that can produce the lm_head's logits, across all three weight encodings.
///
/// ONE LIST, because there were two hand-copied ones — in `patch_prefill` and in
/// `lm_head_operands` — and both named only `{Gemm, GemmSmall, GemmMed, Gemv}` plus the three
/// original fp8 twins. The tile-inventory campaign then added the 128x256 and 192x256 rungs in
/// each encoding (`GemmWide*`, `GemmC5*`), and the MXFP4 column already existed, so an lm_head
/// that resolved to any of those would not be RECOGNISED as a matmul.
///
/// The consequence is not an error. `patch_prefill` falls into its `(None, Some(_))` arm — a
/// `tracing::warn!` — and never runs `insts[lm].i[4] = clen - 1`, so prefill samples its logits
/// from ROW 0 of the chunk instead of the last real prompt row: a silently wrong first token with
/// a warning nobody reads. Not reachable today (Gemma emits lm_head at M=1, which picks a small
/// tile, and the GLM/MLA tail uses `Gemv`) — but it is latent, and it is exactly the drift the
/// identity-based `kv_write_row_field` refactor was introduced to end.
const LM_HEAD_MATMUL_OPS: &[DevOp] = &[
    DevOp::Gemv,
    // bf16
    DevOp::Gemm,
    DevOp::GemmMed,
    DevOp::GemmSmall,
    DevOp::GemmWide,
    DevOp::GemmC5,
    // fp8 (w8a8 / w8a16)
    DevOp::GemmFp8,
    DevOp::GemmMedFp8,
    DevOp::GemmSmallFp8,
    DevOp::GemmWideFp8,
    DevOp::GemmC5Fp8,
    // MXFP4 (w4a16)
    DevOp::GemmMxfp4,
    DevOp::GemmMedMxfp4,
    DevOp::GemmSmallMxfp4,
    DevOp::GemmWideMxfp4,
    DevOp::GemmC5Mxfp4,
];

/// Whether `op` can be the instruction that writes the logits tensor.
fn is_lm_head_matmul(op: u16) -> bool {
    LM_HEAD_MATMUL_OPS.iter().any(|&o| o as u16 == op)
}

/// The `extern "C" __device__` marker `runtime/amd/interp.hip` emits when it was compiled with
/// `PLOW_K3=1` — the seven Kimi-K3 / KDA arms.
///
/// Present iff the axis is on, so absence is not ambiguous: every object built before the axis
/// existed, and every object built with it off, carries no such symbol and has no K3 arm.
const K3_ARMS_SYM: &str = "plow_k3_arms_1";
const DSPARK_NONCAUSAL_SYM: &str = "plow_dspark_noncausal_1";
const K3_SPEC_VERIFY_SYM: &str = "plow_k3_spec_verify_1";
const MOE_PF_A4W4_SYM: &str = "plow_moe_pf_a4w4_arm";
const KDA_CONV_STEP_DB_SYM: &str = "plow_kda_conv_step_db_arm";
const KDA_CONV_STEP_DB_REPLACED_OPS: &[DevOp] = &[
    DevOp::KdaConv,
    DevOp::KdaGate,
    DevOp::KdaStateStep,
    DevOp::KdaConv3,
    DevOp::KdaStateStepG,
];

/// Every opcode that reaches an arm behind `PLOW_K3` in `runtime/amd/interp.hip`.
///
/// The KDA mixer ops — four decomposed plus the two FUSED ones the decode emitter actually uses —
/// and the three K3 block-structure ops, and nothing else. This is the Rust half of the contract
/// whose C half is the `#if PLOW_K3` region around those nine `case` labels;
/// `k3_arm_ops_match_the_interpreter` reads `interp.hip` and asserts the two agree, so a future
/// tenth arm added inside the guard cannot go unlisted here — which is exactly how `KdaConv3` and
/// `KdaStateStepG` were forced onto this list rather than remembered onto it.
const K3_ARM_OPS: &[DevOp] = &[
    DevOp::KdaConv,
    DevOp::KdaGate,
    DevOp::KdaStateStep,
    DevOp::KdaGatedNorm,
    DevOp::KdaConv3,
    DevOp::KdaStateStepG,
    DevOp::KdaConvStateStepG,
    DevOp::AttnRes,
    DevOp::SituGlu,
    DevOp::MlaOutGate,
];

/// The first K3/KDA opcode in these programs, or `None` if the packet needs no K3 arm.
fn required_k3_op(progs: &[DevProg]) -> Option<DevOp> {
    progs
        .iter()
        .flat_map(|p| p.insts.iter())
        .find_map(|i| K3_ARM_OPS.iter().copied().find(|&o| o as u16 == i.op))
}

fn requires_dspark_noncausal(progs: &[DevProg]) -> bool {
    progs.iter().flat_map(|p| p.insts.iter()).any(|i| {
        i.op == DevOp::FlashMlaDecode as u16 && i.t[7] == packet::dev::TENSOR_NONE16 && i.i[6] > 1
    })
}

fn check_dspark_noncausal(syms: &[&str], path: &Path, need: bool) -> Result<()> {
    if !need || syms.contains(&DSPARK_NONCAUSAL_SYM) {
        return Ok(());
    }
    Err(RuntimeError::Device(format!(
        "packet/object DSpark MISMATCH: this packet carries a multi-query non-causal MLA block, \
         but {} was compiled without PLOW_DSPARK_NONCAUSAL (it does not advertise \
         `{DSPARK_NONCAUSAL_SYM}`). A normal MLA decode arm would make the seven draft rows \
         attend one another or silently use only one row. Rebuild the dedicated DSpark object \
         with -DPLOW_DSPARK_NONCAUSAL=1.",
        path.display()
    )))
}

fn requires_k3_spec_verify(progs: &[DevProg]) -> bool {
    progs.iter().flat_map(|p| p.insts.iter()).any(|i| {
        (i.op == DevOp::FlashMlaDecode as u16 || i.op == DevOp::FlashMlaDecodeFp8 as u16)
            && i.fj[1] > 1
    })
}

fn check_k3_spec_verify(syms: &[&str], path: &Path, need: bool) -> Result<()> {
    if !need || syms.contains(&K3_SPEC_VERIFY_SYM) {
        return Ok(());
    }
    Err(RuntimeError::Device(format!(
        "packet/object K3 target-verifier MISMATCH: this packet carries a causal multi-token MLA \
         decode block, but {} lacks `{K3_SPEC_VERIFY_SYM}`. Rebuild the dedicated object with \
         -DPLOW_K3_SPEC_VERIFY=1; an ordinary decode object would verify every row against the \
         wrong one-token causal frontier.",
        path.display()
    )))
}

fn required_moe_pf_a4w4(progs: &[DevProg]) -> Option<DevOp> {
    progs.iter().flat_map(|p| p.insts.iter()).find_map(|i| {
        let op = DevOp::from_u16(i.op)?;
        ((op == DevOp::MoeGroupGluPf || op == DevOp::MoeGroupDownPf)
            && i.i[MOE_PF_ENC_SLOT] == MOE_ENC_MXFP4)
            .then_some(op)
    })
}

fn required_kda_conv_step_db(progs: &[DevProg]) -> Option<DevOp> {
    first_op_in(progs, &[DevOp::KdaConvStateStepG])
}

fn check_kda_conv_step_db(
    syms: &[&str],
    path: &Path,
    need: Option<DevOp>,
    legacy: Option<DevOp>,
) -> Result<()> {
    let armed = syms.contains(&KDA_CONV_STEP_DB_SYM);
    if let Some(op) = need {
        if armed {
            return Ok(());
        }
        return Err(RuntimeError::Device(format!(
            "packet/object KDA Conv3+state MISMATCH: this packet dispatches {op:?} (op {}), but \
             {} was compiled without PLOW_KDA_CONV_STEP_DB (it does not advertise \
             `{KDA_CONV_STEP_DB_SYM}`). Rebuild the K3 decode object with \
             -DPLOW_KDA_CONV_STEP_DB=1.",
            op as u16,
            path.display()
        )));
    }
    if let (true, Some(op)) = (armed, legacy) {
        return Err(RuntimeError::Device(format!(
            "packet/object KDA Conv3+state MISMATCH: {} advertises `{KDA_CONV_STEP_DB_SYM}` and \
             replaces the legacy {op:?} arm (op {}), but the packet still dispatches it. Use the \
             default K3 decode object or re-emit with PLOW_K3_KDA_CONV_STEP_DB=1.",
            path.display(),
            op as u16
        )));
    }
    Ok(())
}

fn check_moe_pf_a4w4(syms: &[&str], path: &Path, need: Option<DevOp>) -> Result<()> {
    let Some(op) = need else {
        return Ok(());
    };
    if syms.contains(&MOE_PF_A4W4_SYM) {
        return Ok(());
    }
    Err(RuntimeError::Device(format!(
        "packet/object A4W4 MISMATCH: this packet dispatches {op:?} (op {}) with MXFP4 encoding, \
         but {} was compiled without PLOW_MOE_PF_A4W4 (it does not advertise \
         `{MOE_PF_A4W4_SYM}`). The kernel refusal path writes NaNs by design. Rebuild the object \
         with -DPLOW_MOE_PF_A4W4=1.",
        op as u16,
        path.display()
    )))
}

/// Refuse a code object that has no K3/KDA arms against a packet that dispatches one.
///
/// WHY THIS HAS TO EXIST, and why it is a REFUSAL rather than a warning. AMD's dispatch
/// `default:` is `/* PLOW_DOP_NOP */` — it writes NOTHING. It does not trap, unlike sm_120's
/// `default: __trap()`. So a Kimi-K3 packet run against an object built without `PLOW_K3` would
/// not fault: every KDA mixer op and every K3 block op would leave its output buffer exactly as
/// it found it, and the run would complete fluently on uninitialised memory. That is the same
/// failure class `GFX950_DISPATCHED` was introduced for after four instances in one week, and
/// gating the arms behind a build axis is what re-opens it unless the pairing is checked.
///
/// Checked against the ELF rather than against a build flag, for the reason the GEMV capacity
/// marker states: the loader reads `.symtab` before the object is on a device, so the object
/// answers for itself and a stale `-D` on someone's shell cannot lie about it.
fn check_k3_arms(syms: &[&str], path: &Path, need: Option<DevOp>) -> Result<()> {
    let Some(op) = need else {
        return Ok(());
    };
    if syms.contains(&K3_ARMS_SYM) {
        return Ok(());
    }
    Err(RuntimeError::Device(format!(
        "packet/object K3 MISMATCH: this packet dispatches {op:?} (op {}), but {} was compiled \
         without PLOW_K3 (it does not advertise `{K3_ARMS_SYM}`). AMD's dispatch default writes \
         NOTHING rather than trapping, so this op would silently leave its output untouched and \
         the run would complete on uninitialised memory instead of failing. Rebuild the object \
         with -DPLOW_K3=1 (scripts/build_k3_*.sh and scripts/build_kda_real.sh pass it; for \
         runtime/CMakeLists.txt use -DPLOW_HSACO_K3=ON), or serve a packet that does not use the \
         Kimi-K3 block.",
        op as u16,
        path.display()
    )))
}

/// The markers `runtime/amd/interp.hip` emits for the Gemma-4 MoE axes. [GEMMA4-MOE-AMD]
///
/// Two, not one, because the decode family (ops 61-72) and the grouped-prefill family (73-77,
/// 81/82) are separate build flags: the prefill half is a second full MFMA body and a decode-only
/// object must not carry it.
const MOE_GEMMA_SYM: &str = "plow_moe_gemma_arms_1";
/// Marker for L2-DOMAIN DISPATCH (-DPLOW_L2_PLACE_DISPATCH). A placed blob repurposes the
/// global-queue `seg` as an L2 domain, so an object without this axis mis-dispatches it SILENTLY.
/// Checked instead of the operator-asserted PLOW_L2_PLACE_DISPATCH env var.
const L2_DISPATCH_SYM: &str = "plow_l2_place_dispatch_1";
const MOE_GEMMA_PF_SYM: &str = "plow_moe_gemma_pf_arms_1";

/// Every opcode behind `#if PLOW_MOE_GEMMA` in `runtime/amd/interp.hip`.
const MOE_GEMMA_OPS: &[DevOp] = &[
    DevOp::MoeRouterGemma,
    DevOp::MoeRouterGemmaScore,
    DevOp::MoeRouterGemmaScoreFast,
    DevOp::MoeRouterGemmaTopk,
    DevOp::MoeExpertGluGemma,
    DevOp::MoeExpertGluNormGemma,
    DevOp::MoeExpertDownGemma,
    DevOp::MoeExpertGluGemmaFp8,
    DevOp::MoeExpertDownGemmaFp8,
    DevOp::MoeCombineGemma,
    DevOp::MoeCombineNormGemma,
    DevOp::MoeCombineResidNormGemma,
];

/// Every opcode behind `#if PLOW_MOE_GEMMA_PF`.
const MOE_GEMMA_PF_OPS: &[DevOp] = &[
    DevOp::MoeRouterGemmaPf,
    DevOp::MoeAlignGemmaPf,
    DevOp::MoeGroupGluGemmaPf,
    DevOp::MoeGroupDownGemmaPf,
    DevOp::MoeCombineNormGemmaPf,
    DevOp::MoeGroupGluGemmaPfW8a8,
    DevOp::MoeGroupDownGemmaPfW8a8,
];

fn first_op_in(progs: &[DevProg], set: &[DevOp]) -> Option<DevOp> {
    progs
        .iter()
        .flat_map(|p| p.insts.iter())
        .find_map(|i| set.iter().copied().find(|&o| o as u16 == i.op))
}

/// Refuse a code object with no Gemma-4 MoE arms against a packet that dispatches one.
///
/// SAME ARGUMENT AS `check_k3_arms`, and the same failure it prevents. These nineteen opcodes had
/// no AMD arm at all until the port, so a Gemma-4 26B-A4B packet ran straight into the dispatch
/// `default:` — which on AMD writes NOTHING rather than trapping. Every router, expert and combine
/// would have left its buffer untouched and the model would have decoded fluently off whatever
/// was in memory. Gating the arms behind a build axis re-opens exactly that hole unless the
/// pairing is checked, so it is checked, against the ELF's `.symtab` rather than against a build
/// flag: the object answers for itself and a stale `-D` cannot lie about it.
fn check_moe_gemma_arms(
    syms: &[&str],
    path: &Path,
    need_dec: Option<DevOp>,
    need_pf: Option<DevOp>,
) -> Result<()> {
    for (need, sym, flag) in [
        (need_dec, MOE_GEMMA_SYM, "PLOW_MOE_GEMMA"),
        (need_pf, MOE_GEMMA_PF_SYM, "PLOW_MOE_GEMMA_PF"),
    ] {
        let Some(op) = need else { continue };
        if syms.contains(&sym) {
            continue;
        }
        return Err(RuntimeError::Device(format!(
            "packet/object GEMMA-MoE MISMATCH: this packet dispatches {op:?} (op {}), but {} was \
             compiled without {flag} (it does not advertise `{sym}`). AMD's dispatch default \
             writes NOTHING rather than trapping, so this op would silently leave its output \
             untouched and the run would complete on uninitialised memory instead of failing. \
             Rebuild the object with -D{flag}=1.",
            op as u16,
            path.display()
        )));
    }
    Ok(())
}

/// The marker `runtime/amd/interp.hip` emits when it was compiled with `PLOW_FP8_KV=1`.
const FP8_KV_SYM: &str = "plow_fp8_kv_1";

/// Every opcode that reaches an arm behind `#if PLOW_FP8_KV` — the fp8 half of the SWAP.
const FP8_KV_OPS: &[DevOp] = &[
    DevOp::HeadNormRopeFp8,
    DevOp::FlashDecodeFp8,
    DevOp::FlashPrefillFp8,
    DevOp::FlashMlaDecodeFp8,
    DevOp::FlashMlaPrefillFp8,
];

/// Every opcode that reaches an arm behind the `#else` — the bf16 half of the SWAP.
///
/// `HeadNormRope` is deliberately NOT here: it is unconditional in both objects (an fp8-KV
/// packet still uses it for the QUERY norm, which is not cached). Listing it would refuse every
/// fp8 packet ever emitted. The gathered MLA ops are not here either — `FlashGatherDecode` /
/// `FlashGatherPrefill` keep their bf16 arm in BOTH objects, because their `t7` is the `idx`
/// table and there is no slot left for a dequant scale.
const BF16_KV_OPS: &[DevOp] = &[
    DevOp::FlashDecode,
    DevOp::FlashPrefill,
    DevOp::FlashMlaDecode,
    DevOp::FlashMlaPrefill,
];

/// The first opcode in `progs` that reaches an arm on `side`, or `None`.
fn required_kv_op(progs: &[DevProg], side: &[DevOp]) -> Option<DevOp> {
    progs
        .iter()
        .flat_map(|p| p.insts.iter())
        .find_map(|i| side.iter().copied().find(|&o| o as u16 == i.op))
}

/// Refuse a code object whose KV ENCODING does not match the packet's — in EITHER direction.
///
/// WHY BOTH DIRECTIONS, where [`check_k3_arms`] needs only one. `PLOW_K3` is additive: an object
/// with the arms serves a packet without them perfectly. `PLOW_FP8_KV` is a **swap** —
/// `interp.hip` compiles `FLASH_DECODE_FP8` / `FLASH_MLA_DECODE_FP8` *instead of* their bf16
/// twins, deliberately, so the register budget does not carry both. Each object is therefore
/// missing an arm the other has, and AMD's dispatch `default:` writes NOTHING rather than
/// trapping.
///
/// This is not hypothetical. Running the K3 MLA gate's **bf16** packet against the **fp8**
/// object reports `all packets executed on every slice: YES` and then scores rel `1.000e+00` at
/// the attention output — a completely untouched `Opart`, read as a result, with the packet
/// graph reporting full success. The same shape as the four instances `GFX950_DISPATCHED` was
/// introduced for, reached through a build axis instead of a missing case label.
///
/// The manifest (`crates/devgen/src/manifest.rs`, `fp8_kv -> PLOW_FP8_KV=1`) is the half that
/// SELECTS the right object. This is the half that refuses a wrong pair that was selected
/// anyway — a stale `-D` on a shell, a hand-copied `.hsaco`, an object directory shared between
/// two axes. Checked against `.symtab` rather than against a build flag, for the reason the GEMV
/// capacity marker states: the object answers for itself.
///
/// AN ABSENT MARKER MEANS BF16, and unlike [`K3_ARMS_SYM`] that is a deliberate choice rather
/// than a tautology: `PLOW_FP8_KV` PREDATES this marker, so an `.hsaco` built with the axis
/// before this commit advertises nothing and will be refused for an fp8 packet. That is a FALSE
/// REFUSAL, and it is the right one — the alternative is to treat "no marker" as "might be
/// either", which is exactly the silence this check exists to remove, and the refusal names the
/// remedy (rebuild). The commit that added the marker also changed `interp.hip`, so every gfx950
/// object has to be rebuilt for it anyway; there is no deployment in which a pre-marker object is
/// still the right object.
fn check_kv_encoding(
    syms: &[&str],
    path: &Path,
    need_fp8: Option<DevOp>,
    need_bf16: Option<DevOp>,
) -> Result<()> {
    let obj_fp8 = syms.contains(&FP8_KV_SYM);
    let bad = if obj_fp8 { need_bf16 } else { need_fp8 };
    let Some(op) = bad else {
        return Ok(());
    };
    Err(RuntimeError::Device(format!(
        "packet/object KV-ENCODING MISMATCH: this packet dispatches {op:?} (op {}), which is the \
         {} half of the PLOW_FP8_KV swap, but {} was built {} (it {} `{FP8_KV_SYM}`). The axis is \
         a SWAP — that object compiles the other half INSTEAD, not as well — and AMD's dispatch \
         default writes NOTHING rather than trapping, so this op would leave its output untouched \
         and the run would complete on stale memory (measured: rel 1.000e+00 at the attention \
         output with every packet reporting success). Serve the blob against the object its \
         `build.json` `requires` names, or rebuild this object {}.",
        op as u16,
        if obj_fp8 { "bf16" } else { "fp8" },
        path.display(),
        if obj_fp8 {
            "WITH -DPLOW_FP8_KV=1"
        } else {
            "WITHOUT -DPLOW_FP8_KV=1"
        },
        if obj_fp8 {
            "advertises"
        } else {
            "does not advertise"
        },
        if obj_fp8 {
            "without -DPLOW_FP8_KV=1"
        } else {
            "with -DPLOW_FP8_KV=1"
        },
    )))
}

/// The `PLOW_GEMV_MM` an object advertises, or `None` when it advertises
/// nothing (built before the marker existed, or unparseable).
fn object_gemv_cap(syms: &[&str]) -> Option<u32> {
    syms.iter()
        .filter_map(|s| s.strip_prefix(GEMV_CAP_SYM_PREFIX))
        .filter_map(|n| n.parse::<u32>().ok())
        .max()
}

/// Refuse a code object whose compiled GEMV row bucket is narrower than the
/// packet's widest GEMV.
///
/// WHY THIS HAS TO EXIST. `gemv_rows<MM>` carries `float acc[MM]`, predicates on
/// `m < M` and writes `C[m*N + n]` — and has NO outer loop over `M > MM`. The
/// outer loop is NVIDIA's `gemv_walk` (`runtime/nvidia/op_gemm.cuh`), which makes
/// `GV_MM_MAX` a pure performance knob over there; here it was built, measured at
/// 276 registers, and removed, so the bucket is a hard CAPACITY. The packet
/// carries M as a runtime immediate and the capacity is baked into the object,
/// and until the marker existed nothing compared them: a packet asking for M=8
/// against an MM=1 object wrote row 0 and left rows 1..7 STALE. No fault, no
/// trap, no zero page — fluent output with rms error `sqrt((T-1)/T)`.
///
/// [`AmdEngine::load`] already refuses `batch > PLOW_GEMV_MAXM`, but that
/// compares the blob against a HARDCODED ceiling of 16. An M=8 blob on an MM=1
/// object passes it (8 ≤ 16) and produces one correct row out of eight. This
/// compares the blob against the OBJECT, which is the only comparison that can
/// close the gap.
///
/// AN ABSENT MARKER AT `need > 1` IS A REFUSAL, not a pass. Every gfx950 object
/// built before this marker existed compiled at some `PLOW_GEMV_MM` the loader
/// cannot see, and the overwhelmingly common value is the `op_gemm.h` default of
/// 1 — that default IS the bug. Treating silence as consent here would reproduce
/// exactly the state in which the bug shipped. `need <= 1` never refuses:
/// MM ≥ 1 always, so every object, marked or not, covers a one-row GEMV, and the
/// batch-1 path is untouched by this check.
fn check_gemv_capacity(syms: &[&str], path: &Path, need: u32) -> Result<()> {
    if need <= 1 {
        return Ok(());
    }
    // A WALKING OBJECT HAS NO CAPACITY TO EXCEED, and that is the point of the walk.
    //
    // `gemv_walk` (op_gemm.h) wraps `d_gemv`/`d_gemv_glu`/`d_gemv_qkv` in
    // `for (m0 = 0; m0 < M; m0 += MM) f(m0, min(MM, M - m0))`, so the bucket stops being a
    // capacity and becomes a per-pass WIDTH: every row is written, in ceil(M/MM) passes, and
    // the ragged tail is served by the `m < M` predicate each row body already carries. The
    // staging bound moves with it — `min(MM, M) * K`, not `M * K`.
    //
    // This is what lets an MM=8 object serve a t=16 program while keeping BOTH decode fusions
    // (`devgen::gemv_staged_rows`), which is the arm §6g-WALK's Phase B exists to test.
    // The blob-vs-`PLOW_GEMV_MAXM` ceiling in `AmdEngine::load` is deliberately NOT relaxed
    // here: raising it past 16 is a separate question about `t`-dependent workgroup counts and
    // the fine dependency map (§6g-SERVE §5), not about this kernel's row loop.
    if syms.contains(&GEMV_WALK_SYM) {
        return Ok(());
    }
    // Above `PLOW_GEMV_MAXM` there is no object to rebuild — the header's static
    // assert refuses the bucket — so do not send anyone to build one. That case
    // is also caught by the ceiling check in `load`, but this runs first (objects
    // are opened before the batch is known), so this message has to be the
    // correct one on its own.
    let rebuild = if need > GEMV_MAXM {
        format!(
            "No object can serve this: PLOW_GEMV_MAXM is {GEMV_MAXM} (runtime/amd/op_gemm.h) and \
             the bucket cannot be built wider. Re-emit the packet at PLOW_DECODE_BATCH <= \
             {GEMV_MAXM}."
        )
    } else {
        format!(
            "Rebuild it with PLOW_DECODE_BATCH={need} (scripts/build_gfx950.sh, or \
             -DPLOW_DECODE_BATCH={need} for runtime/CMakeLists.txt), or serve a packet emitted \
             at a smaller batch."
        )
    };
    match object_gemv_cap(syms) {
        Some(cap) if cap >= need => Ok(()),
        Some(cap) => Err(RuntimeError::Device(format!(
            "packet/object GEMV MISMATCH: this packet's widest GEMV asks for M={need} rows, but \
             {} was compiled PLOW_GEMV_MM={cap} (it advertises `{GEMV_CAP_SYM_PREFIX}{cap}`). \
             `gemv_rows<MM>` has no outer loop over M > MM, so rows {cap}..{need} would never be \
             written and the run would complete with STALE data in them instead of failing. \
             {rebuild}",
            path.display()
        ))),
        None => Err(RuntimeError::Device(format!(
            "packet/object GEMV MISMATCH: this packet's widest GEMV asks for M={need} rows, and \
             {} does not say what PLOW_GEMV_MM it was compiled at — it carries no \
             `{GEMV_CAP_SYM_PREFIX}<N>` symbol, so it predates the marker in \
             runtime/amd/op_gemm.h. Objects built before it compiled at the header's default of \
             1, which writes row 0 and leaves rows 1..{need} STALE with no fault anywhere. \
             {rebuild}",
            path.display()
        ))),
    }
}

/// Is the V2 MLA-prefill routing enabled (`PLOW_MLA_PF_V2=1`)?
///
/// Opt-in: it moves `FlashMlaPrefill` segments onto the 4-wave flash object, whose
/// full-column-wave kernel (`d_flash_mla_prefill_v2`) needs the 512-register budget the
/// 8-wave interpreter cannot give. The flash object must advertise
/// `plow_mla_pf_v2_arm_1` — checked at load, because the dispatch default on a pre-arm
/// object is a silent skip, not a trap.
pub fn mla_pf_v2_enabled() -> bool {
    crate::config::RuntimeConfig::get().amd.mla_pf_v2
}

/// The V2 arms' marker symbols (see `interp.hip`).
const MLA_PF_V2_SYM: &str = "plow_mla_pf_v2_arm_1";
const MLA_PF_V2_FP8_SYM: &str = "plow_mla_pf_v2_fp8_arm_1";

/// Per-segment wave class, derived from the stream.
///
/// A segment is class 4 (the flash interpreter, 256 threads) iff ANY stream
/// entry in it points at a flash-prefill instruction; everything else is class
/// 8. The fp8 twin is included here and is NOT in the reference — omitting it
/// silently ran fp8-KV flash segments on the 8-wave object.
///
/// Under `PLOW_MLA_PF_V2=1`, the bf16 and fp8-KV MLA prefill ops (51 and 110) are class 4 too — but only in
/// programs whose bucket is big enough to fill the machine at the V2 kernel's BQ=64 work
/// decomposition (`t >= 2048`: 256+ items over 304 CUs). Smaller buckets keep the 8-wave
/// kernel, whose BQ=32 fills at half the tokens.
pub fn derive_segments(prog: &DevProg) -> Result<Vec<u8>> {
    derive_segments_for(prog, mla_pf_v2_enabled() && prog.t >= 2048)
}

fn derive_segments_for(prog: &DevProg, v2: bool) -> Result<Vec<u8>> {
    let mut n_seg: u32 = 1;
    for e in &prog.stream {
        let s = e.seg as u32 + 1;
        if s > n_seg {
            n_seg = s;
        }
    }
    if n_seg > MAX_SEG {
        return Err(RuntimeError::Device(format!(
            "program declares {n_seg} segments (max {MAX_SEG}) — corrupt stream?"
        )));
    }
    let mut class = vec![8u8; n_seg as usize];
    // MLA-V2 routing needs PURE segments: the flash object's dispatch skips every op it does
    // not carry, so a segment is only sent there if EVERY entry in it is FlashMlaPrefill —
    // which is exactly what an emit under PLOW_MLA_PF_V2=1 produces. A blob emitted without
    // the split fails the purity test and stays whole on the 8-wave object; a split blob run
    // without the env falls to the t/env guard and likewise runs 8-wave. Either mismatch
    // degrades, never corrupts.
    let mut mla_pure = vec![v2; n_seg as usize];
    let mut needs_v2: Vec<(usize, u32)> = Vec::new();
    let mut mla_any = vec![false; n_seg as usize];
    for e in &prog.stream {
        let op = prog
            .insts
            .get(e.inst as usize)
            .ok_or_else(|| {
                RuntimeError::Device(format!(
                    "stream entry references instruction {} of {}",
                    e.inst,
                    prog.insts.len()
                ))
            })?
            .op;
        if op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16 {
            class[e.seg as usize] = 4;
        } else if op == DevOp::FlashMlaPrefill as u16 || op == DevOp::FlashMlaPrefillFp8 as u16 {
            // A dense causal KV-split packet (i6 = ns, no union table in t7) writes ns
            // partials per (token, head) and the merge reads ns of them. The 8-wave
            // fallback kernel writes the nsplit=1 layout — so the env/routing mismatches
            // that DEGRADE for plain V2 blobs must REFUSE here instead of corrupting.
            let d = &prog.insts[e.inst as usize];
            // DEFERRED to after the class pass, and that is the whole point. This used to
            // refuse on `!v2` -- the ENV -- which misses the case that actually corrupts:
            // PLOW_MLA_PF_V2=1 SET at serve against a blob emitted WITHOUT it. `PLOW_MLA_PF_V2`
            // is read at emit too (`packet/src/devbuild.rs:1123`) and is what splits
            // FlashMlaPrefill into its own wave-class-4 segment; without it at emit the segment
            // is IMPURE, `mla_pure` stays false, `class` never becomes 4, and the ns packet runs
            // on the 8-wave kernel anyway -- while `v2` is true so the old guard stayed silent.
            // The requirement is not "the env is set", it is "THIS packet's segment was actually
            // routed to the V2 arm", which is only known once the class pass below has run.
            if op == DevOp::FlashMlaPrefill as u16
                && d.i[6] > 1
                && d.t[7] == packet::dev::TENSOR_NONE16
            {
                needs_v2.push((e.seg as usize, d.i[6]));
            }
            mla_any[e.seg as usize] = true;
        } else {
            mla_pure[e.seg as usize] = false;
        }
    }
    for s in 0..n_seg as usize {
        if mla_any[s] && mla_pure[s] {
            class[s] = 4;
        }
    }
    for &(seg, ns) in &needs_v2 {
        if !(v2 && class[seg] == 4) {
            return Err(RuntimeError::Device(format!(
                "this packet's FlashMlaPrefill carries a causal KV-split (ns={ns}) which only \
                 the V2 flash arm honors, and V2 routing is NOT live for it: \
                 PLOW_MLA_PF_V2={} at serve, segment {seg} routed to wave class {} (needs 4). \
                 Serving it on the 8-wave kernel would write nsplit=1 partials under an ns-wide \
                 merge -- silently wrong output, not a crash. Either the serve env is unset, or \
                 the BLOB was emitted without PLOW_MLA_PF_V2=1 (it is read at emit too, \
                 packet/src/devbuild.rs, and is what puts FlashMlaPrefill in its own \
                 wave-class-4 segment) so the segment is impure. Re-emit with \
                 PLOW_MLA_PF_V2=1, or re-emit without PLOW_GLM_PF_NS.",
                if v2 { "1" } else { "unset" },
                class[seg]
            )));
        }
    }
    Ok(class)
}

/// Rows-equivalent cost charged per launch in the chunk DP. Tuned in the
/// reference; `PLOW_LAUNCH_ROWS` overrides. It is what stops the DP from
/// choosing a hundred tiny chunks that each pay a full dispatch.
///
/// **416 is about 4x low and is deliberately left alone.** Measured over five
/// plan pairs on GLM-5.2 gfx942, a launch is ~231 ms and a padded row costs
/// 0.078-0.202 ms with no trend in context, so the honest price is ~1650 rows;
/// `PLOW_LAUNCH_ROWS=1780` is -16% at 1025 tokens and -31% at 3073 and never
/// regressed at any of 14 lengths up to 71808. It is not landed because
/// [`crate::config::AmdConfig::ragged_chunk`] DOMINATES it at every one of those
/// lengths and its output-visible blast radius is a strict subset of ragged's —
/// the reprice is a partial ragged, not an alternative to it, and under ragged
/// this constant is never read (see the early return in [`plan_chunks_cfg`]).
/// Raise this only if ragged-M is ruled out for a reason other than speed.
/// `perf-data/plow-gfx942/glm52-chunk-policy.md`.
const LAUNCH_ROWS: u32 = 416;

/// Cover `n_prompt` tokens with chunks drawn from the compiled bucket ladder.
///
/// Not a fixed chunk size: buckets are a ladder and the DP mixes them, so a
/// 1500-token prompt can be 1024+512 rather than two 1024s with 548 padded rows
/// that cost full compute. Each launch is charged [`LAUNCH_ROWS`] rows so the
/// DP trades padding against dispatch count instead of minimising one alone.
///
/// Returned largest-first, which puts the ragged chunk LAST — the tail is where
/// padding lands, and a padded row writes KV nothing reads.
///
/// Under `PLOW_RAGGED_CHUNK` the trade the DP exists to make DISAPPEARS: the row
/// shrink in [`rebase_chunk_rows`] makes padded rows cost nothing, so the cover
/// is simply the fewest launches, `ceil(n / max_bucket)`. See the branch below.
pub fn plan_chunks(buckets: &[u32], n_prompt: u32) -> Result<Vec<u32>> {
    let cfg = &crate::config::RuntimeConfig::get().amd;
    plan_chunks_cfg(
        buckets,
        n_prompt,
        cfg.launch_rows.unwrap_or(LAUNCH_ROWS),
        cfg.ragged_chunk,
    )
}

/// [`plan_chunks`] with its two policy inputs passed in rather than read from the
/// process-wide [`crate::config::RuntimeConfig`], which is a `OnceLock` and so
/// cannot be toggled by a unit test. Every cover rule is argued here; the public
/// wrapper only supplies the config.
pub fn plan_chunks_cfg(
    buckets: &[u32],
    n_prompt: u32,
    launch_rows: u32,
    ragged: bool,
) -> Result<Vec<u32>> {
    // THE CAP IS THE PACKET'S OWN LADDER, and there is deliberately no second
    // constant here to disagree with it. The widest compiled prefill bucket IS
    // `shapes.max_chunk` in the manifest (`devgen::manifest` defines it as
    // `max(prefill_buckets)`), and the same emit sizes the KV ring from it — so a
    // runtime `MAX_CHUNK` could only ever be a stale copy that silently discards
    // rungs the blob was built to use. It was 8192, and it is what made a packet
    // carrying a 16384 rung serve as if the rung were absent.
    //
    // The `RING >= window + chunk - 1` invariant that constant stood for is
    // enforced where the ring is SIZED, at emit (`devgen::kv_ring`), and it is
    // vacuous for the models that can exceed 8192 today: the MLA family is
    // full-causal (`window = 0`), so `kv_ring` returns `(ctx, MASK_NONE)` and the
    // chunk does not size the cache at all. The generic (windowed) path cannot
    // reach a wider bucket in the first place — its ladder derives from
    // `max_chunk()`, which `MAX_CHUNK_MAX` caps at 8192.
    let mut bkt: Vec<u32> = buckets.iter().copied().filter(|&b| b > 0).collect();
    bkt.sort_unstable();
    bkt.dedup();
    if bkt.is_empty() {
        return Err(RuntimeError::Device(
            "no prefill bucket at or under the max chunk — is this a decode-only blob?".into(),
        ));
    }
    if n_prompt == 0 {
        return Ok(Vec::new());
    }
    // RAGGED-M: the padding is free, so the ONLY thing worth minimising is the
    // launch count, and the minimum is `ceil(n / max_bucket)`. Take the widest
    // bucket while more than one launch is left, then cover the remainder with
    // the SMALLEST bucket that holds it — `rebase_chunk_rows` runs that chunk at
    // its real row count, so the choice of rung costs nothing and only has to be
    // big enough.
    //
    // This is why repricing `LAUNCH_ROWS` is NOT an alternative to the shrink but
    // a consequence of it. Under the padded regime the DP is right to refuse a
    // wider tail: covering 4097 with one 8192 chunk really does cost ~4095 rows
    // of dead compute, which is worse than the second launch. Only once the
    // padding is free does "fewest launches" become the cheapest cover.
    if ragged {
        let max_b = *bkt.last().expect("non-empty");
        let mut out = Vec::new();
        let mut rem = n_prompt;
        while rem > max_b {
            out.push(max_b);
            rem -= max_b;
        }
        if rem > 0 {
            out.push(*bkt.iter().find(|&&b| b >= rem).expect("rem <= max bucket"));
        }
        return Ok(out);
    }
    let quant = bkt[0];
    let rows = n_prompt.div_ceil(quant) as usize;

    // cost[r] = cheapest cover of r quanta; pick[r] = the bucket that achieved it.
    let mut cost = vec![u64::MAX; rows + 1];
    let mut pick = vec![0u32; rows + 1];
    cost[0] = 0;
    for r in 1..=rows {
        for &b in &bkt {
            let step = (b / quant).max(1) as usize;
            let prev = r.saturating_sub(step);
            if cost[prev] == u64::MAX {
                continue;
            }
            let c = cost[prev] + (b + launch_rows) as u64;
            if c < cost[r] {
                cost[r] = c;
                pick[r] = b;
            }
        }
    }
    if cost[rows] == u64::MAX {
        return Err(RuntimeError::Device(format!(
            "no chunk cover for {n_prompt} tokens from buckets {bkt:?}"
        )));
    }
    let mut out = Vec::new();
    let mut r = rows;
    while r > 0 {
        let b = pick[r];
        out.push(b);
        r = r.saturating_sub((b / quant).max(1) as usize);
    }
    // Reconstruction walks backwards, so this is already smallest-last after
    // the reverse — i.e. largest first, ragged chunk at the tail.
    out.sort_unstable_by(|a, b| b.cmp(a));
    Ok(out)
}

// The packet declares the expert POINTER TABLES per layer and NO checkpoint contains them: they
// are tables of DEVICE POINTERS the host computes after the named weights land.
//
// Two families, same reason. `expert_*` are the routed experts, packed by `bind_packed_experts`.
// `dense_*` are the first `first_k_dense_replace` layers' FFN, whose PREFILL runs on the grouped
// expert arms with degenerate 1-expert routing and therefore also reaches its weights only through
// a pointer table (`bind_dense_ffn_tables`).
//
// Missing one there is not a subtle bug: the named-weight loop fails the load with
// `MISSING WEIGHT`, because no checkpoint has a tensor by these names. The predicate is
// `packet::names::is_host_filled_table`, shared with the CUDA loader and the VRAM planner —
// the AMD loader was the only one of the five sites that had it.

/// Pack a MoE model's routed experts and fill its expert POINTER TABLES.
///
/// Fill the DENSE-FFN pointer tables a PREFILL packet declares for its first
/// `first_k_dense_replace` layers (3 on GLM-5.2, 1 on Kimi).
///
/// # Why a dense layer has a pointer table at all
///
/// Its prefill runs on the GROUPED EXPERT arms (ops 85/86) with degenerate
/// 1-expert routing — see `emit_glm_dense_block_prefill` in
/// `crates/devgen/src/mla.rs` and the header of `d_moe_align_pf`. Those ops
/// reach their weights only through `wtab[e*3 + j]`, so even a single "expert"
/// needs the indirection. The reason is a real ISA gap: there is no block-fp8
/// tiled GEMM opcode, and ops 85/86 already are one.
///
/// # Why this is trivial next to [`bind_packed_experts`]
///
/// Nothing is packed and nothing is sliced. Unlike the 256 routed experts, the
/// dense `gate_proj`/`up_proj`/`down_proj` and their `weight_scale_inv` grids
/// ARE declared tensors, so they are already uploaded, already TP-sharded on the
/// right axis by the ordinary named-weight path, and already at a stable device
/// address. All this owes the packet is the three addresses, in the same
/// `{gate, up, down}` order `expert_weight_table` uses.
///
/// A decode-only blob declares no such table and this is a no-op — which is what
/// keeps every existing GLM asset loading unchanged.
fn bind_dense_ffn_tables(
    be: &HsaBackend,
    blob: &DevBlob,
    devp: &[DeviceMem],
    names: &[String],
) -> Result<usize> {
    const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
    let mut filled = 0usize;
    for (i, td) in blob.tensors.iter().enumerate() {
        // `mlp.dense_weight_table` -> weights; `mlp.dense_scale_table` -> the
        // scale twins.
        //
        // TWO scale spellings, tried in this order, because the quantisations
        // that reach this table spell theirs differently and a packet carries
        // whichever its checkpoint had:
        //
        //   `.weight_scale_inv`  block-fp8, an f32 [N/128][K/128] grid (GLM-5.2)
        //   `.weight_scale`      MX microscaling, one E8M0 byte per 32 elements
        //
        // `_inv` is probed FIRST and a block-fp8 packet therefore resolves on the
        // first candidate, to exactly the name this loop used to build — that is
        // what makes the GLM path byte-identical rather than merely equivalent.
        // A packet with neither is still a hard error naming both.
        let (pfx, suffixes): (&str, &[&str]) = match td
            .name
            .strip_suffix("dense_weight_table")
            .map(|p| (p, &[".weight"][..]))
            .or_else(|| {
                td.name
                    .strip_suffix("dense_scale_table")
                    .map(|p| (p, &[".weight_scale_inv", ".weight_scale"][..]))
            }) {
            Some(v) => v,
            None => continue,
        };
        let mut addrs = [0u64; 3];
        for (j, proj) in PROJ.iter().enumerate() {
            let cands: Vec<String> = suffixes.iter().map(|s| format!("{pfx}{proj}{s}")).collect();
            let (want, k) = cands
                .iter()
                .find_map(|w| names.iter().position(|n| n == w).map(|k| (w, k)))
                .ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "dense-FFN prefill table `{}` needs one of {cands:?}, none of which \
                         the packet declares. The table and the three projections are emitted \
                         together by declare_glm_rows; a packet with one and not the other is \
                         malformed.",
                        td.name
                    ))
                })?;
            // A zero base would be read by the kernel as the EP "not my expert"
            // sentinel and the tile would be silently skipped, so an unbound
            // weight must fail here rather than produce a layer that computes
            // nothing.
            if devp[k].base == 0 {
                return Err(RuntimeError::Device(format!(
                    "`{want}` has no device allocation; the grouped arm reads a null weight base \
                     as the 'not my expert' sentinel and would skip the whole layer"
                )));
            }
            addrs[j] = devp[k].base;
        }
        let bytes: Vec<u8> = addrs.iter().flat_map(|a| a.to_le_bytes()).collect();
        EngineDevice::upload(be, &devp[i], 0, &bytes)?;
        filled += 1;
    }
    Ok(filled)
}

/// GLM-5.2 (and DeepSeek-shaped models generally) keep every expert as its own
/// checkpoint tensor — 256 x {gate, up, down} x {weight, block scale} per layer,
/// 115k lookups a rank. `devgen` deliberately declares none of them
/// (`mla.rs`: "declaring 75*256*6 handles would bloat the tensor table for zero
/// emit benefit"): the ops only ever reach an expert through
/// `expert_weight_table[eid*3 + j]`. So the host owes the packet, per MoE layer,
/// ONE contiguous weight buffer + ONE scale buffer holding this rank's experts,
/// and the two tables of addresses into them.
///
/// Ported from `runtime/tests/glm52_decode.c` (the TP4 decode this is checked
/// against), with the per-projection slicing delegated to
/// [`crate::asset::shard`] rather than re-derived — the expert `gate`/`up`/
/// `down` shard on exactly the same axes as their dense counterparts, and the
/// classifier already keys on those names.
///
/// # TP vs EP is read off the packet, not off a flag
///
/// The gate/up op carries the `I_moe` it will stream per expert: the FULL
/// `moe_intermediate_size` under expert-parallel (whole experts, `E/N` of them
/// per rank) and `moe_intermediate_size / n_gpu` under tensor-parallel (a slice
/// of every expert). Against the checkpoint's own expert shape that decides
/// which layout to build, so a packet emitted `--ep` cannot be bound as if it
/// were TP. It matters that this is not a host-side flag: the mismatch has no
/// symptom other than wrong tokens.
///
/// The same instruction is found by OPCODE, not by the name of the tensor it
/// writes — see the note at `GLU_ARMS`.
///
/// # The expert NAME is read off the checkpoint, for the same reason
///
/// Three checkpoints reach this function and none of them spells an expert the
/// way the other two do (see [`ExpertNames`]). [`resolve_expert_names`] probes
/// for the one that is there and [`check_expert_geometry`] then makes the
/// weight and its scale agree with each other, so a checkpoint whose scale is
/// the wrong SIZE fails by name rather than by producing wrong numbers of the
/// right length.
#[allow(clippy::too_many_arguments)]
/// Where the weight-load wall clock actually goes.
///
/// There was NO load timing in this engine: it logged GiB and a tensor count and
/// never a second, so every claim about load cost came from a runbook rather
/// than from the runtime. The phases below are the four that can each be the
/// whole cost depending on the machine and the page-cache state, and which one
/// dominates decides which optimisation is worth anything:
///
/// * `fault` — first touch of the checkpoint mmap. Zero on a warm cache;
///   NVMe-bound on a genuinely cold one, in which case nothing else matters.
/// * `gather` — [`crate::asset::shard::slice_for`]. Free (a borrow) for a
///   replicated or column shard, a full row-by-row copy for a row shard.
/// * `memcpy` — page cache → pinned staging slab, on one core.
/// * `dma` — `hsa_amd_memory_async_copy` submit + signal wait.
///
/// `Cell`, not atomics: one rank's load is one thread, and the counters are
/// touched once per 64 MiB chunk, so this must not appear in any profile.
#[derive(Default)]
struct LoadProf {
    fault_ns: Cell<u64>,
    gather_ns: Cell<u64>,
    memcpy_ns: Cell<u64>,
    dma_ns: Cell<u64>,
    alloc_ns: Cell<u64>,
    memset_ns: Cell<u64>,
    /// Chunks pushed through the staging slab — the signal round-trip count.
    chunks: Cell<u64>,
}

impl LoadProf {
    fn add(c: &Cell<u64>, t: Instant) {
        c.set(c.get() + t.elapsed().as_nanos() as u64);
    }

    /// One `Instant::now()` reused as both "stop A" and "start B".
    fn split(c: &Cell<u64>, t: Instant) -> Instant {
        let now = Instant::now();
        c.set(c.get() + now.duration_since(t).as_nanos() as u64);
        now
    }

    fn ms(c: &Cell<u64>) -> f64 {
        c.get() as f64 / 1e6
    }

    fn report(&self, what: &str, wall: std::time::Duration, bytes: u64) {
        let gib = bytes as f64 / (1u64 << 30) as f64;
        let s = wall.as_secs_f64();
        tracing::info!(
            phase = what,
            wall_s = format!("{s:.2}").as_str(),
            gib = format!("{gib:.2}").as_str(),
            gib_s = format!("{:.2}", if s > 0.0 { gib / s } else { 0.0 }).as_str(),
            fault_ms = format!("{:.0}", Self::ms(&self.fault_ns)).as_str(),
            gather_ms = format!("{:.0}", Self::ms(&self.gather_ns)).as_str(),
            memcpy_ms = format!("{:.0}", Self::ms(&self.memcpy_ns)).as_str(),
            dma_ms = format!("{:.0}", Self::ms(&self.dma_ns)).as_str(),
            alloc_ms = format!("{:.0}", Self::ms(&self.alloc_ns)).as_str(),
            memset_ms = format!("{:.0}", Self::ms(&self.memset_ns)).as_str(),
            chunks = self.chunks.get(),
            "LOAD PHASES"
        );
    }
}

/// Is the mmap-fault phase being measured separately?
///
/// A page fault on an mmap'd checkpoint is charged to whoever touches the page
/// FIRST, and here that is the `copy_from_slice` into the staging slab — so on a
/// cold cache `memcpy` silently contains all of the NVMe time and the breakdown
/// says nothing. Reading one byte per page first moves that cost onto its own
/// counter. It is off by default because it is a second pass over the source
/// (cheap when warm, but not free) and this is the load path, not a benchmark.
fn profile_faults() -> bool {
    crate::config::RuntimeConfig::get().load_profile
}

/// Touch one byte of every page so the fault cost lands on `fault_ns` rather
/// than hiding inside the staging memcpy. Returns without reading anything the
/// caller can observe; `read_volatile` is what stops LLVM deleting it.
///
/// Must be handed the bytes the rank will ACTUALLY read ([`touched`]), never the
/// whole tensor. A column-parallel rank binds a contiguous 1/tp slice, so
/// prefaulting the whole thing pulls four times the bytes off the drive and the
/// profile then describes a load nobody runs — a mistake this made once, and one
/// that flattered nothing: it inflated the very phase it was there to measure.
fn prefault(src: &[u8], prof: &LoadProf) {
    let ns = prefault_ns(src);
    prof.fault_ns.set(prof.fault_ns.get() + ns);
}

/// [`prefault`] without the profile handle, for gather workers that cannot
/// touch the `Cell` counters; the caller folds the returned ns in.
fn prefault_ns(src: &[u8]) -> u64 {
    let t = Instant::now();
    let mut acc = 0u8;
    let mut i = 0usize;
    while i < src.len() {
        // SAFETY: `i < src.len()`, and a `u8` read is always aligned.
        acc ^= unsafe { std::ptr::read_volatile(src.as_ptr().add(i)) };
        i += 4096;
    }
    std::hint::black_box(acc);
    t.elapsed().as_nanos() as u64
}

use crate::asset::checkpoint::{prefetch_depth, prefetch_threads};

/// Stride between tensors carved out of the weight slab. Mirrors `exec::gpu`.
///
/// The STRIDE, not a claim about the resulting addresses: a tensor lands at
/// `slab.base + k*SLAB_ALIGN`, so its true alignment is whatever the pool gave
/// the base. ROCr reports `RUNTIME_ALLOC_GRANULE` = 4 KiB and allocates on it,
/// and for a request the size of a model's weights it hands back far more (the
/// measured rounding is to 2 MiB). Either floor already clears what the kernels
/// ask of a global address — `global_load_dwordx4` wants 16 B, the MFMA tile
/// loads no more — so the stride is chosen for padding waste, a few MiB across a
/// blob, and not to raise alignment.
const SLAB_ALIGN: u64 = 4096;

/// Bytes a tensor of `bytes` occupies in the slab, trailing pad included.
///
/// The sizing pass sums this and the carve advances by it, over the same list —
/// they must agree exactly or the carve runs past the allocation.
fn slab_pad(bytes: u64) -> u64 {
    bytes.div_ceil(SLAB_ALIGN) * SLAB_ALIGN
}

/// The bytes rank `rank` will actually TOUCH for `name`, as a queueable span.
///
/// Not the whole tensor: a column-parallel rank binds one contiguous 1/tp range
/// and faulting all of it in would read four times what that rank needs. The
/// range comes from [`crate::asset::shard::slice_for`] itself — the same
/// function that will do the real bind — so it cannot drift from the real read.
///
/// The one case that must NOT go through `slice_for` is a row-parallel gather:
/// it returns `Cow::Owned`, and doing the gather twice would cost more than the
/// prefetch saves. A row gather is strided over every row, so the bytes it
/// touches ARE the whole tensor, which is what this returns for it.
fn touched<'a>(
    ckpt: &'a crate::asset::checkpoint::Checkpoint,
    name: &str,
    want: u64,
    rank: u32,
    n_gpu: u32,
) -> Option<&'a [u8]> {
    let (src, shape) = ckpt.tensor_ex(name)?;
    let strided = n_gpu > 1
        && shape.len() == 2
        && crate::asset::shard::shard_of(name) == crate::asset::shard::Shard::Row;
    if strided {
        return Some(src);
    }
    match crate::asset::shard::slice_for(name, src, shape, want, rank, n_gpu) {
        Ok(std::borrow::Cow::Borrowed(s)) => Some(s),
        // An `Owned` here would mean the row guard above missed a case; the
        // gather has already happened, so fall back to the whole tensor rather
        // than pretend the range is known.
        _ => Some(src),
    }
}

/// [`touched`], resolved into a span the prefetch pool can carry to a thread.
fn weight_span(
    ckpt: &crate::asset::checkpoint::Checkpoint,
    name: &str,
    want: u64,
    rank: u32,
    n_gpu: u32,
) -> Option<crate::asset::checkpoint::Span> {
    let (src, _) = ckpt.tensor_ex(name)?;
    let span = touched(ckpt, name, want, rank, n_gpu)?;
    let off = span.as_ptr() as usize - src.as_ptr() as usize;
    ckpt.span(name, off, span.len())
}

/// How ONE routed expert is spelled in the checkpoint on disk.
///
/// Three spellings reach this loader and they disagree on all four axes:
///
/// | checkpoint | sub-namespace | projections | payload | scale |
/// |---|---|---|---|---|
/// | GLM-5.2 / DeepSeek block-fp8 | `…mlp.` | `gate_proj`/`up_proj`/`down_proj` | `.weight` | `.weight_scale_inv` |
/// | Kimi-K2.7-Code MXFP4 | `…mlp.` | the same three | `.weight` | `.weight_scale` |
/// | Kimi-K3 (compressed-tensors mxfp4) | `…block_sparse_moe.` | `w1`/`w3`/`w2` | `.weight_packed` | `.weight_scale` |
///
/// The middle row is the reason this is RESOLVED and not switched on a flag: a
/// K2.7 checkpoint is the standard projection names with an E8M0 scale, so
/// "mxfp4" and "Mixtral-spelled" are independent facts and no single boolean
/// carries both. A flag that disagrees with the bytes is the failure this file
/// keeps finding; the bytes are the only thing that cannot disagree with itself.
///
/// `proj` is in `expert_weight_table` slot order — gate, up, down — which is why
/// the Mixtral row reads `w1`/`w3`/`w2` and not `w1`/`w2`/`w3`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpertNames {
    /// Everything up to and including `experts.`; an expert index follows.
    ns: String,
    /// gate, up, down.
    proj: [&'static str; 3],
    /// `.weight` or `.weight_packed`.
    payload: &'static str,
    /// `.weight_scale_inv` (block-fp8 f32 grid) or `.weight_scale` (E8M0 row).
    scale: &'static str,
}

impl ExpertNames {
    fn weight_of(&self, e: u32, j: usize) -> String {
        format!("{}{e}.{}{}", self.ns, self.proj[j], self.payload)
    }

    fn scale_of(&self, e: u32, j: usize) -> String {
        format!("{}{e}.{}{}", self.ns, self.proj[j], self.scale)
    }

    /// Is the scale an MX microscaling row (one E8M0 byte per 32 elements along
    /// K) rather than a block-fp8 `[N/128][K/128]` f32 grid?
    fn microscaled(&self) -> bool {
        self.scale == ".weight_scale"
    }
}

/// Which spelling THIS checkpoint uses, decided by probing it.
///
/// `pfx` is what is left of the packet's `…expert_weight_table` after the suffix
/// is stripped, and it is not always a checkpoint prefix: the GLM emitter
/// declares the table under the model prefix (`model.layers.{l}.mlp.`), the K3
/// emitter under its own `moe.` namespace (`moe.language_model.model.layers.{l}.`)
/// because `packet::names` classifies compiler-owned tensors by that prefix. So
/// `moe.` is stripped and the MoE sub-namespace is probed rather than assumed.
///
/// ORDER IS THE COMPATIBILITY GUARANTEE. The first candidate is `{pfx}experts.0.
/// gate_proj.weight` + `.weight_scale_inv` — character for character the two
/// names this function replaced hardcoded — so a block-fp8 packet resolves on
/// probe one and every name built downstream is the name it was built before.
fn resolve_expert_names(
    ckpt: &crate::asset::checkpoint::Checkpoint,
    pfx: &str,
) -> Result<ExpertNames> {
    const TEMPLATES: [([&str; 3], &str); 2] = [
        (["gate_proj", "up_proj", "down_proj"], ".weight"),
        (["w1", "w3", "w2"], ".weight_packed"),
    ];
    const SCALES: [&str; 2] = [".weight_scale_inv", ".weight_scale"];
    let base = pfx.strip_prefix("moe.").unwrap_or(pfx);
    let mut tried: Vec<String> = Vec::new();
    for sub in ["", "mlp.", "block_sparse_moe."] {
        for (proj, payload) in TEMPLATES {
            let ns = format!("{base}{sub}experts.");
            let probe = format!("{ns}0.{}{payload}", proj[0]);
            if ckpt.tensor_ex(&probe).is_none() {
                tried.push(probe);
                continue;
            }
            // The payload is there, so this IS the layout — a missing scale is
            // now a broken checkpoint and not a wrong guess, and saying so beats
            // falling through to a spelling that cannot be right.
            for scale in SCALES {
                if ckpt
                    .tensor_ex(&format!("{ns}0.{}{scale}", proj[0]))
                    .is_some()
                {
                    return Ok(ExpertNames {
                        ns,
                        proj,
                        payload,
                        scale,
                    });
                }
            }
            return Err(RuntimeError::Device(format!(
                "MISSING EXPERT SCALE: `{probe}` is in the checkpoint but neither \
                 `{ns}0.{}{}` nor `{ns}0.{}{}` is. A quantized expert without its scale \
                 cannot be dequantized, and binding the payload alone would decode from \
                 4-bit or 8-bit mantissas read as if they were already scaled.",
                proj[0], SCALES[0], proj[0], SCALES[1]
            )));
        }
    }
    Err(RuntimeError::Device(format!(
        "MISSING EXPERT WEIGHT: the packet declares `{pfx}expert_weight_table` but the \
         checkpoint has no routed experts under any spelling this loader knows. Probed: \
         {tried:?}"
    )))
}

/// Fail unless expert 0's three scale twins are the right SIZE for the weights
/// they scale.
///
/// Every expert in a layer is the same shape, and `slice_for` re-checks each one
/// against the stride derived here — so this is the only place the WEIGHT and its
/// SCALE are compared to each other at all. Getting it wrong is silent in the
/// worst way: an E8M0 row and a block-fp8 grid can be the same number of bytes
/// for some geometries, so a size that merely "looks plausible" is exactly the
/// thing that must not be accepted.
fn check_expert_geometry(
    ckpt: &crate::asset::checkpoint::Checkpoint,
    n: &ExpertNames,
) -> Result<()> {
    let miss = |name: &str| {
        RuntimeError::Device(format!(
            "MISSING EXPERT WEIGHT: {name} (expert 0 resolved to the `{}` + `{}` layout \
             under `{}`, so every projection must be present in it)",
            n.payload, n.scale, n.ns
        ))
    };
    for j in 0..3 {
        let (wn, sn) = (n.weight_of(0, j), n.scale_of(0, j));
        let (w, ws) = ckpt.tensor_ex(&wn).ok_or_else(|| miss(&wn))?;
        let (s, ss) = ckpt.tensor_ex(&sn).ok_or_else(|| miss(&sn))?;
        let bad = |m: String| {
            Err(RuntimeError::Device(format!(
                "EXPERT SCALE GEOMETRY: `{sn}` {ss:?} ({} B) cannot be the scale of `{wn}` \
                 {ws:?} ({} B): {m}",
                s.len(),
                w.len()
            )))
        };
        if ws.len() != 2 || ss.len() != 2 {
            return bad(
                "both must be 2-D — a routed expert is a matrix and its scale is \
                        a grid or a per-group row, never a vector"
                    .into(),
            );
        }
        let (wn0, wn1, sn0, sn1) = (ws[0], ws[1], ss[0], ss[1]);
        if n.microscaled() {
            // MX: payload is [N, K/2] (two fp4 per byte), scale is [N, K/32]
            // (one E8M0 byte per group of 32 along K). Both are u8, so the byte
            // count IS the element count.
            if w.len() != wn0 * wn1 || s.len() != sn0 * sn1 {
                return bad("an mxfp4 payload and its E8M0 scale are both u8, so each \
                            must be exactly the product of its shape"
                    .into());
            }
            if sn0 != wn0 {
                return bad(format!("the output dim disagrees: {wn0} vs {sn0}"));
            }
            if wn1 * 2 != sn1 * 32 {
                return bad(format!(
                    "K disagrees: the payload packs {} elements per row, the scale covers {}",
                    wn1 * 2,
                    sn1 * 32
                ));
            }
        } else {
            // Block-fp8: payload is [N, K] e4m3 (1 B/element), scale is
            // [ceil(N/128), ceil(K/128)] f32. Verified against
            // zai-org/GLM-5.2-FP8: [2048, 6144] -> [16, 48].
            const B: usize = 128;
            if w.len() != wn0 * wn1 {
                return bad("an fp8 e4m3 payload is 1 B/element, so it must be exactly \
                            the product of its shape"
                    .into());
            }
            let (gn, gk) = (wn0.div_ceil(B), wn1.div_ceil(B));
            if (sn0, sn1) != (gn, gk) || s.len() != gn * gk * 4 {
                return bad(format!(
                    "a block-fp8 scale grid must be [{gn}, {gk}] f32 ({} B)",
                    gn * gk * 4
                ));
            }
        }
    }
    Ok(())
}

/// One expert plan entry gathered on a worker, minus the ring push.
///
/// `data` borrows the checkpoint mmap for a column/replicated shard and owns a
/// `Vec` for a row gather; either way the bytes are exactly what the old
/// sequential loop handed the ring for this destination.
struct GatheredExpert<'a> {
    dst: u64,
    want: u64,
    scrub: bool,
    data: std::borrow::Cow<'a, [u8]>,
    /// (device dst, permuted payload) when the preshuffled pf slab is declared.
    pf: Option<(u64, Vec<u8>)>,
    gather_ns: u64,
    fault_ns: u64,
}

/// Expert-gather workers per rank. Every rank of a TP group loads at once, so
/// the box runs `n_gpu *` this many gather threads; the clamp keeps one rank
/// from claiming the whole socket while still passing the ~16-thread knee of
/// [`crate::asset::checkpoint::Checkpoint::populate`]'s scaling table.
fn expert_gather_threads(n_gpu: u32) -> usize {
    let cores = std::thread::available_parallelism().map_or(16, |n| n.get());
    (cores / n_gpu.max(1) as usize).clamp(8, 16)
}

/// The host-side work for one plan entry: fault the span in, slice this rank's
/// shard, optionally preshuffle. Pure reads of the checkpoint — safe from any
/// worker — and all device interaction stays with the caller.
fn gather_expert_entry<'a>(
    ckpt: &'a crate::asset::checkpoint::Checkpoint,
    entry: &(String, u64, u64, bool, u64, u64, u64),
    shard_rank: u32,
    shard_n: u32,
    populate: bool,
    do_prefault: bool,
) -> Result<GatheredExpert<'a>> {
    let (name, dst, want, scrub, pf_dst, pf_rows, pf_k) = entry;
    let (src, shape) = ckpt
        .tensor_ex(name)
        .ok_or_else(|| RuntimeError::Device(format!("MISSING EXPERT WEIGHT: {name}")))?;
    if populate {
        if let Some(s) = weight_span(ckpt, name, *want, shard_rank, shard_n) {
            ckpt.populate(s);
        }
    }
    let mut fault_ns = 0u64;
    if do_prefault {
        if let Some(s) = touched(ckpt, name, *want, shard_rank, shard_n) {
            fault_ns = prefault_ns(s);
        }
    }
    // `tp = 1` is the EP/single-GPU case: bind the expert whole. Otherwise the
    // classifier sees the gate/up projection (`gate_proj.weight` or
    // `.w1.weight`: a contiguous output-row slice) and the down projection
    // (`down_proj.weight` or `.w2.weight`: a strided input-column gather), and
    // BOTH scale spellings ride the same substring tests onto the same axis —
    // which is exactly the C reference's hand-rolled
    // `j < 2 ? offset : gather_row` split.
    let t = Instant::now();
    let data = crate::asset::shard::slice_for(name, src, shape, *want, shard_rank, shard_n)?;
    let mut gather_ns = t.elapsed().as_nanos() as u64;
    let pf = if *pf_dst != 0 {
        // Preshuffled copy: out[((kt*R)+r)*64 + b] = in[r*K + kt*64 + b]. A pure
        // permutation of the SAME bytes — the scrub-during-push commutes with it,
        // so the pf slab sees exactly the values the row-major slab does.
        let t = Instant::now();
        let (rows, kb) = (*pf_rows as usize, *pf_k as usize);
        debug_assert_eq!(data.len(), rows * kb);
        debug_assert_eq!(kb % 64, 0);
        let mut shuffled = vec![0u8; data.len()];
        let nkt = kb / 64;
        for r in 0..rows {
            for kt in 0..nkt {
                let s = r * kb + kt * 64;
                let d = (kt * rows + r) * 64;
                shuffled[d..d + 64].copy_from_slice(&data[s..s + 64]);
            }
        }
        gather_ns += t.elapsed().as_nanos() as u64;
        Some((*pf_dst, shuffled))
    } else {
        None
    };
    Ok(GatheredExpert {
        dst: *dst,
        want: *want,
        scrub: *scrub,
        data,
        pf,
        gather_ns,
        fault_ns,
    })
}

fn bind_packed_experts(
    be: &HsaBackend,
    blob: &DevBlob,
    ckpt: &crate::asset::checkpoint::Checkpoint,
    devp: &[DeviceMem],
    names: &[String],
    ring: &mut crate::device::hsa::HsaUploadRing,
    rank: u32,
    n_gpu: u32,
    prof: &LoadProf,
    do_prefault: bool,
    populate: bool,
) -> Result<(Vec<DeviceMem>, u64)> {
    let layers: Vec<(usize, String)> = blob
        .tensors
        .iter()
        .enumerate()
        .filter_map(|(i, td)| {
            td.name
                .strip_suffix("expert_weight_table")
                .map(|pfx| (i, pfx.to_string()))
        })
        .collect();
    if layers.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let t0 = std::time::Instant::now();
    // `I_moe` is not inferred: it is read off the very instruction that will
    // stream these weights. Every gate/up arm — per-slot bf16, per-slot
    // block-fp8, and the grouped fp8 collapse — reads the layer's
    // `expert_weight_table` from `t[3]` and carries `I_moe` in `i[1]`
    // (`crates/packet/src/slots.rs`).
    //
    // The `down` arm shares `t[3]` and carries H there instead, so the two must
    // be told apart. This USED to key on `t[0] == act.fu`, the tensor the gate/up
    // arm writes — which worked only because every emitter before Kimi-K3 named
    // that tensor exactly `act.fu`. K3's does not: its MoE activations are
    // per-layer (`act.l1.moe.fu`), so the name lookup failed outright and no K3
    // packet could bind an expert. The OPCODE is what actually distinguishes the
    // two arms, it is the same fact devgen encoded when it chose the op, and it
    // cannot be renamed. On every pre-K3 packet this selects the identical
    // instruction: the ops below are exactly the ones that write `act.fu`.
    const GLU_ARMS: [DevOp; 3] = [
        DevOp::MoeExpertGlu,
        DevOp::MoeExpertGluFp8Blk,
        DevOp::MoeGroupGluFp8Blk,
    ];
    // A BATCHED decode program routes its MoE through the GROUPED PREFILL chain, not the
    // per-slot decode chain: at T rows the emitter picks `MoeGroupGluPf`, which sorts the
    // (row, expert) pairs by expert so one expert's weights cross HBM once for every row that
    // chose it. That is the correct chain for a batch — MoE has no cross-row state, and sharing
    // the weight traffic across rows is the whole point of batching a memory-bound decode.
    //
    // Its operands sit in DIFFERENT SLOTS, which is the entire reason this needs its own arm:
    // the table is `t[2]` and `I_moe` is `i[0]` (`crates/packet/src/slots.rs:196`), against
    // `t[3]` / `i[1]` on the three arms above. Matching it with the same slot indices would
    // find nothing, and the loader would report `expert_weight_table is declared but no decode
    // instruction streams experts through it` on a program that streams them perfectly well.
    let dec = blob.progs.last().expect("checked non-empty");
    let i_moe_of = |i_ewt: usize| -> Option<u64> {
        dec.insts.iter().find_map(|d| {
            if d.t[3] as usize == i_ewt && GLU_ARMS.iter().any(|&o| o as u16 == d.op) {
                Some(d.i[1] as u64)
            } else if d.t[2] as usize == i_ewt && d.op == DevOp::MoeGroupGluPf as u16 {
                Some(d.i[0] as u64)
            } else {
                None
            }
        })
    };

    let mut bufs = Vec::with_capacity(layers.len() * 2);
    let mut i_moe = 0u64;
    // Which spelling the checkpoint turned out to have, for the one log line
    // that says so. A load that binds the wrong layout is silent by nature, so
    // the resolved answer belongs in the record rather than in a debug session.
    let mut layout = String::from("none");
    let mut wbytes = 0u64;
    for (i_ewt, pfx) in &layers {
        let i_est = names
            .iter()
            .position(|x| *x == format!("{pfx}expert_scale_table"))
            .ok_or_else(|| {
                RuntimeError::Device(format!(
                    "{pfx}expert_weight_table has no matching expert_scale_table"
                ))
            })?;
        // The declared table size IS the expert count: `[E][3]` u64.
        let n_exp = (blob.tensors[*i_ewt].bytes / 24) as u32;
        if n_exp == 0 || !blob.tensors[*i_ewt].bytes.is_multiple_of(24) {
            return Err(RuntimeError::Device(format!(
                "{pfx}expert_weight_table is {} B, not a whole [E][3] u64 table",
                blob.tensors[*i_ewt].bytes
            )));
        }
        i_moe = i_moe_of(*i_ewt).ok_or_else(|| {
            RuntimeError::Device(format!(
                "{pfx}expert_weight_table is declared but no decode instruction \
                 streams experts through it — nothing to pack against"
            ))
        })?;
        // WHICH SPELLING, from the checkpoint, before anything is sized against
        // it — then the weight/scale size agreement, once, for expert 0.
        let en = resolve_expert_names(ckpt, pfx)?;
        check_expert_geometry(ckpt, &en)?;
        layout = format!("{}{{gate,up,down}}{}+{}", en.ns, en.payload, en.scale);
        // Geometry from expert 0; every expert in a layer is the same shape.
        let probe = en.weight_of(0, 0);
        let (w0, shape0) = ckpt
            .tensor_ex(&probe)
            .ok_or_else(|| RuntimeError::Device(format!("MISSING EXPERT WEIGHT: {probe}")))?;
        // `I_moe` is the gate projection's OUTPUT dim under every spelling —
        // `[I_moe, K]` for `gate_proj.weight`, `[I_moe, latent/2]` for a packed
        // `w1.weight_packed`. The packing halves K, never N, so this comparison
        // against the packet's declared `I_moe` is unaffected by it.
        let i_moe_full = *shape0.first().unwrap_or(&0) as u64;
        let (owned, whole) = if i_moe == i_moe_full {
            // EP: this rank owns a contiguous block of WHOLE experts.
            let per = n_exp / n_gpu;
            if per * n_gpu != n_exp {
                return Err(RuntimeError::Device(format!(
                    "expert-parallel packet with {n_exp} experts over {n_gpu} ranks \
                     does not divide"
                )));
            }
            (rank * per..(rank + 1) * per, true)
        } else if i_moe * n_gpu as u64 == i_moe_full {
            // TP: every rank slices every expert.
            (0..n_exp, false)
        } else {
            return Err(RuntimeError::Device(format!(
                "{pfx}: the packet streams I_moe={i_moe} per expert but the checkpoint's \
                 experts are {i_moe_full} wide, which is neither the whole expert (EP) \
                 nor a 1/{n_gpu} slice (TP)"
            )));
        };
        let n_local = owned.len() as u64;
        // Slot strides: what ONE {expert, proj} occupies in the packed buffer.
        let w_stride = w0.len() as u64 / if whole { 1 } else { n_gpu as u64 };
        let s_probe = en.scale_of(0, 0);
        let s_stride = ckpt
            .tensor_ex(&s_probe)
            .ok_or_else(|| RuntimeError::Device(format!("MISSING EXPERT SCALE: {s_probe}")))?
            .0
            .len() as u64
            / if whole { 1 } else { n_gpu as u64 };

        let t_alloc = Instant::now();
        let d_w = EngineDevice::alloc(be, (n_local * 3 * w_stride).max(1))?;
        let d_s = EngineDevice::alloc(be, (n_local * 3 * s_stride).max(1))?;
        LoadProf::add(&prof.alloc_ns, t_alloc);
        let wtab = crate::orch::moe::packed_expert_table(d_w.base, w_stride, n_exp, owned.clone());
        let stab = crate::orch::moe::packed_expert_table(d_s.base, s_stride, n_exp, owned.clone());
        // PRESHUFFLED PREFILL SLAB (PLOW_MOE_PF_SHUF): the blob declaring
        // `{pfx}expert_weight_table_pf` is the emit-time opt-in. A SECOND slab holds every
        // projection permuted to B'[K/64][R][64] so the grouped prefill GEMM's per-k-tile B
        // stream is one contiguous 16 KiB block (full 128 B lines) instead of 64 B row-slices
        // at K-stride — the aiter-asm preshuffle, done once at bind. Decode keeps streaming
        // whole rows from the row-major slab; the cost is one extra slab of HBM and one host
        // permutation pass per projection.
        let i_ewt_pf = names
            .iter()
            .position(|x| *x == format!("{pfx}expert_weight_table_pf"));
        let (d_wp, wptab) = if i_ewt_pf.is_some() {
            if en.microscaled() {
                return Err(RuntimeError::Device(format!(
                    "{pfx}expert_weight_table_pf declared for an MXFP4 expert layout — the \
                     preshuffle transform is defined for 1 B/element block-fp8 payloads only"
                )));
            }
            let t_a = Instant::now();
            let d = EngineDevice::alloc(be, (n_local * 3 * w_stride).max(1))?;
            LoadProf::add(&prof.alloc_ns, t_a);
            let t = crate::orch::moe::packed_expert_table(d.base, w_stride, n_exp, owned.clone());
            (Some(d), Some(t))
        } else {
            (None, None)
        };

        // The layer's reads, IN ORDER, before any of them happens.
        //
        // Materialised up front (~3 k `String`s per layer, the same ones a
        // build-at-use loop would allocate) so a worker pool can claim entries
        // by index; workers claim in plan order, which keeps the disk reads
        // roughly sequential.
        let (shard_rank, shard_n) = if whole { (0, 1) } else { (rank, n_gpu) };
        // Scrub 0x80 (OCP -0) out of BLOCK-FP8 expert payloads on the way through the staging
        // slab — value-identical, and it is what lets the CDNA3 grouped-GEMM staging decode
        // drop its neg-0 mask (`mpf_fp8x4_to_bf16_h`, runtime/amd/op_moe.h). Never the scales
        // (f32) and never an MXFP4 payload (0x80 there is two live fp4 nibbles).
        let scrub_w = !en.microscaled();
        // Per-entry: (name, dst, bytes, scrub, pf_dst, pf_rows, pf_kbytes). pf_dst == 0 means no
        // preshuffled copy (scale entries, or the pf table not declared). Geometry per
        // projection: gate/up are [I_moe][K] row-major shards, down is [H][I_moe] — in both
        // cases the shard is [rows][kbytes] with rows*kbytes == w_stride.
        let mut plan: Vec<(String, u64, u64, bool, u64, u64, u64)> =
            Vec::with_capacity(owned.len() * 6);
        for e in owned {
            for j in 0..3 {
                let idx = e as usize * 3 + j;
                let (pf_rows, pf_k) = if j < 2 {
                    (i_moe, w_stride / i_moe)
                } else {
                    (w_stride / i_moe, i_moe)
                };
                let pf_dst = wptab.as_ref().map_or(0, |t| t[idx]);
                plan.push((
                    en.weight_of(e, j),
                    wtab[idx],
                    w_stride,
                    scrub_w,
                    pf_dst,
                    pf_rows,
                    pf_k,
                ));
                plan.push((en.scale_of(e, j), stab[idx], s_stride, false, 0, 0, 0));
            }
        }
        // THE GATHER RUNS ON A WORKER POOL; the ring stays on this thread.
        //
        // One thread walking the plan was the whole load on a DeepSeek-shaped
        // checkpoint: the down projection and every 2-D scale are row shards, so
        // `slice_for` is thousands of sub-KiB strided copies per expert, and the
        // pf preshuffle is another full pass — page-fault- and latency-bound
        // work one core cannot saturate the page cache with (measured 57 s of
        // gather on a warm GLM-5.2 TP8 rank). Workers do the fault + slice +
        // preshuffle in parallel and each faults its own span in via
        // `Checkpoint::populate` first — the ~16-way concurrency the drive
        // needs on a cold cache comes from the pool itself, so this loop no
        // longer feeds the shared `Prefetcher`.
        //
        // Push order is COMPLETION order, not plan order: every entry carries
        // its precomputed device address, so the destination bytes are
        // identical either way. The ring keeps a single owner because pinned
        // staging is the SDMA correctness rule (see the note below), and the
        // bounded channel is the memory cap: at most `2 * workers` gathered
        // entries (~a few MiB each) exist at once.
        //
        // `gather_ns`/`fault_ns` become worker-summed parallel time — same
        // convention as `PrefetchStats`, and they can exceed wall clock.
        let workers = expert_gather_threads(n_gpu).min(plan.len().max(1));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let stop = std::sync::atomic::AtomicBool::new(false);
        let plan = &plan;
        // Through a PINNED slab, always. The copy does not pin its source, so
        // handing it a `slice_for` gather buffer (an ordinary `Vec`) faults the
        // SDMA engine — the one trap the C reference calls out by name.
        let mut push_gathered = |g: GatheredExpert| -> Result<()> {
            prof.gather_ns.set(prof.gather_ns.get() + g.gather_ns);
            prof.fault_ns.set(prof.fault_ns.get() + g.fault_ns);
            let stage_bytes = ring.chunk();
            for (o, chunk) in g.data.chunks(stage_bytes).enumerate() {
                let t = Instant::now();
                let at = g.dst + (o * stage_bytes) as u64;
                if g.scrub {
                    ring.push_scrub_fp8_neg0(at, chunk)?;
                } else {
                    ring.push(at, chunk)?;
                }
                LoadProf::add(&prof.memcpy_ns, t);
                prof.chunks.set(prof.chunks.get() + 1);
            }
            wbytes += g.want;
            if let Some((pf_dst, shuffled)) = g.pf {
                for (o, chunk) in shuffled.chunks(stage_bytes).enumerate() {
                    let t = Instant::now();
                    ring.push_scrub_fp8_neg0(pf_dst + (o * stage_bytes) as u64, chunk)?;
                    LoadProf::add(&prof.memcpy_ns, t);
                    prof.chunks.set(prof.chunks.get() + 1);
                }
                wbytes += g.want;
            }
            Ok(())
        };
        std::thread::scope(|s| -> Result<()> {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<GatheredExpert>>(workers * 2);
            for _ in 0..workers {
                let tx = tx.clone();
                let (next, stop) = (&next, &stop);
                s.spawn(move || loop {
                    if stop.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(entry) = plan.get(i) else { return };
                    let r = gather_expert_entry(
                        ckpt,
                        entry,
                        shard_rank,
                        shard_n,
                        populate,
                        do_prefault,
                    );
                    if tx.send(r).is_err() {
                        return;
                    }
                });
            }
            drop(tx);
            // On error: stop the claimers, then keep draining so no worker is
            // left blocked in `send` when the scope joins — that would hang.
            let mut first_err = None;
            while let Ok(r) = rx.recv() {
                if first_err.is_some() {
                    continue;
                }
                if let Err(e) = r.and_then(&mut push_gathered) {
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    first_err = Some(e);
                }
            }
            first_err.map_or(Ok(()), Err)
        })?;
        EngineDevice::upload(be, &devp[*i_ewt], 0, as_bytes(&wtab))?;
        EngineDevice::upload(be, &devp[i_est], 0, as_bytes(&stab))?;
        if let (Some(ip), Some(tab)) = (i_ewt_pf, &wptab) {
            EngineDevice::upload(be, &devp[ip], 0, as_bytes(tab))?;
        }
        bufs.push(d_w);
        bufs.push(d_s);
        if let Some(d) = d_wp {
            bufs.push(d);
        }
    }
    // No expert is bound until its copy has retired. The pointer tables above
    // are uploaded through the blocking path and name a DIFFERENT address, so
    // they are ordered by construction; the expert bytes are not.
    ring.drain()?;
    tracing::info!(
        layers = layers.len(),
        gib = format!("{:.2}", wbytes as f64 / (1u64 << 30) as f64).as_str(),
        i_moe,
        layout = layout.as_str(),
        secs = format!("{:.1}", t0.elapsed().as_secs_f64()).as_str(),
        "routed experts packed; expert pointer tables filled"
    );
    Ok((bufs, wbytes))
}

/// Contiguous instruction span covering every KV-row patch site, or `None` when
/// there are none.
///
/// The sites are SCATTERED in k/v pairs across all layers (Gemma-31B: `[4,664]`
/// of 676), so one contiguous slice beats a per-instruction scatter: fewer
/// bytes than the whole stream and, more importantly, ONE h2d submission
/// instead of `n_kvrow` of them. Submission overhead, not bytes, is what costs
/// here.
/// May this compiler-owned tensor skip the load-time zeroing?
///
/// The skip is a PERFORMANCE optimisation with a precondition, not a property
/// of the `kv.` namespace: it is sound only where every element is WRITTEN
/// BEFORE IT IS READ. An append-only KV cache satisfies that — attention reads
/// only `[0, kvlen)` and each row is written on the step that admits it — which
/// is why skipping it saves 11.5 GiB of memset on GLM and changes nothing.
///
/// Kimi-K3 put two things under `kv.` that do NOT satisfy it, and both were
/// silently inheriting the skip:
///
/// * the KDA RECURRENT STATE (`kv.{l}.state`, `kv.{l}.conv_state.*`). The
///   recurrence is `state = state * decay(gate) + beta * (v - state·k) ⊗ k`, so
///   it READS `state` on the very first token of the very first sequence. From
///   uninitialised HBM that is garbage folded into an accumulator over up to
///   10^6 rank-1 updates, and it never washes out.
/// * the ATTNRES SNAPSHOT RING (`kv.blkres`), which AttnRes mixes over from the
///   first layer that has a snapshot.
///
/// Neither faults and neither reports a missing weight. They are `kv.`-named
/// because `packet::names::is_checkpoint_weight` classifies by EXCLUSION, so
/// the prefix is what stops the loader demanding them of the checkpoint — it
/// was never a claim about their write-before-read discipline.
fn kv_skips_zeroing(name: &str) -> bool {
    name.starts_with("kv.") && !is_carried_state(name)
}

/// Kimi-K3's AttnRes score weight, folded from the TWO tensors the checkpoint
/// actually ships. `None` when `name` is not one of them.
///
/// `runtime/amd/op_k3.h` states the relation as the reference implementation:
///
/// ```text
/// score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
/// scores       = (k * score_weight).sum(-1)
/// ```
///
/// so the emitter declares ONE f32 `[hidden]` per site while
/// `models--moonshotai--Kimi-K3` ships two bf16 tensors — `*_res_norm.weight`
/// `[7168]` and `*_res_proj.weight` `[1, 7168]` — 93 of each, at both the
/// attention and the MLP site. Without this fold every one of those 186 handles
/// resolves to MISSING WEIGHT and no real-weight K3 run can start.
///
/// The fold is exact, not an approximation, and it is worth saying why it is
/// ALLOWED to be a plain elementwise product. The score is
/// `proj · rmsnorm(x, norm)`, and RMS normalisation scales the whole row by the
/// single scalar `1/rms(x)`. A scalar commutes out of the dot product, so
/// `proj · (x/rms · norm) == (proj ⊙ norm) · (x/rms)` — the gain can be folded
/// into the projection ahead of time and the kernel divides by the RMS itself.
/// If the norm were anything per-element-nonlinear this would not hold.
///
/// f32 because the fold is a PRODUCT OF TWO bf16 VALUES: keeping the result in
/// bf16 would round away most of what the multiply just computed, and the packet
/// declares f32 for that reason. There is no TP axis here — `[hidden]` is
/// replicated on every rank — so this runs identically at any `--num-gpus`.
fn fold_res_score(c: &crate::asset::checkpoint::Checkpoint, name: &str) -> Option<Result<Vec<u8>>> {
    let stem = name.strip_suffix("_res_score.weight")?;
    let bf16 = |n: &str| -> Option<Vec<f32>> {
        let (raw, _) = c.tensor_ex(n)?;
        Some(
            raw.chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
        )
    };
    let (nn, pn) = (
        format!("{stem}_res_norm.weight"),
        format!("{stem}_res_proj.weight"),
    );
    let (Some(g), Some(p)) = (bf16(&nn), bf16(&pn)) else {
        // Name matched the pattern but the sources are absent: report the pair
        // rather than the derived name, which is in no checkpoint by design.
        return Some(Err(RuntimeError::Device(format!(
            "MISSING WEIGHT: {name} is DERIVED and needs both `{nn}` and `{pn}`; \
             at least one is not in the checkpoint"
        ))));
    };
    if g.len() != p.len() {
        return Some(Err(RuntimeError::Device(format!(
            "{name}: norm is {} wide and proj is {} — they must agree",
            g.len(),
            p.len()
        ))));
    }
    let mut out = Vec::with_capacity(g.len() * 4);
    for (a, b) in g.iter().zip(&p) {
        out.extend_from_slice(&(a * b).to_le_bytes());
    }
    Some(Ok(out))
}

/// A `kv.`-namespace tensor that CARRIES STATE ACROSS TOKENS rather than being
/// appended to — the KDA recurrent state, its three conv windows, and the AttnRes
/// snapshot ring.
///
/// Two callers, and separating them was the bug. [`kv_skips_zeroing`] asks whether
/// LOAD may skip the memset; [`AmdEngine::begin_slot`] asks what a NEW SEQUENCE
/// must clear. Those are the same set for the same reason — these tensors are read
/// before they are written — but they were not the same code, and only the first
/// existed. So the state was correctly zeroed once at model load and then never
/// again: request 2 inherited request 1's recurrence and conv windows, and a
/// linear-attention model conditioned on the previous conversation. Nothing faults
/// and nothing reports a missing weight; the second answer is merely wrong, in a
/// way that reads as fluent.
///
/// Substring, not suffix: `conv_state.q`/`.k`/`.v` are three tensors under one
/// idea, and a future `kv.{l}.state.v` must not slip through a match written
/// against today's exact spellings.
fn is_carried_state(name: &str) -> bool {
    name.starts_with("kv.") && (name.contains("state") || name.contains("blkres"))
}

pub fn kvrow_span(kvrow: &[u32]) -> Option<(usize, usize)> {
    let lo = *kvrow.iter().min()? as usize;
    let hi = *kvrow.iter().max()? as usize;
    Some((lo, hi))
}

/// KV-append sites for a packet that DECLARED NONE — returned as
/// `(i[3] sites, i[2] sites)`.
///
/// [`DevBlob::kvrow`] is a list of instructions whose `i[3]` is the write row:
/// one field, one op family, which is everything a GQA decode needs. GLM-5.2's
/// MLA does not fit it. Its cache is a latent + a rope half written by TWO
/// different ops with the row in DIFFERENT fields — the `RmsNorm` into
/// `kv.L.ckv` carries it in `i[2]`, the `HeadNormRope` into `kv.L.krot` in
/// `i[3]` — so `devgen` declares none of them at all (`n_kvrow = 0`).
///
/// A host that then patches nothing does not fail: every token's KV lands in
/// row 0, attention reads `[0, kvlen)` of a cache that never advanced, and the
/// model emits fluent-looking ids that are wrong from the second token on. That
/// is why this is derived rather than skipped. `runtime/tests/glm52_decode.c`
/// finds the same sites by the same rule — the op, plus a `kv.` destination.
///
/// Only consulted when the packet declared no sites, so a Gemma packet keeps
/// the compiled list untouched.
fn derive_kvrow(p: &DevProg, names: &[String]) -> (Vec<u32>, Vec<u32>) {
    let (mut i3, mut i2) = (Vec::new(), Vec::new());
    for (k, d) in p.insts.iter().enumerate() {
        match kv_write_row_field(d.op, names.get(d.t[0] as usize)) {
            Some(3) => i3.push(k as u32),
            Some(2) => i2.push(k as u32),
            _ => {}
        }
    }
    (i3, i2)
}

/// Which `i[]` field of an instruction carries the KV-cache WRITE ROW, or `None`
/// when the instruction does not write the cache at all.
///
/// ONE rule, consulted by BOTH phases — [`derive_kvrow`] for the decode step's
/// per-token row and [`rebase_chunk`] for a prefill chunk's base row. They used
/// to disagree: decode found the sites by destination NAME, prefill by
/// `HeadNormRope` + `fj[1] != 0`, and the second test matches nothing on GLM's
/// MLA (its k_rope sets `j[1]`, which packs into `fj[2]`, and leaves `f[1]`/`j[0]`
/// — the two halves of `fj[1]` — at zero). So a GLM prefill chunk wrote every
/// latent row of every chunk at row 0 and only the LAST chunk's tail survived,
/// with no error anywhere. Two phases deriving "what is a KV write" by two rules
/// is the bug; this is the one rule.
///
/// The destination NAME is the discriminator, and it has to be: the field alone
/// cannot tell GLM's `kv_a_layernorm` (an `RmsNorm` whose `i[2]` is the latent
/// out-row) from the input/post-attention norms, which are the same opcode with
/// `i[2]` meaning nothing. Keying on the opcode alone would rebase those two and
/// corrupt the block input.
///
/// The `HeadNormRopeFp8` twin is included for the same reason the flash test
/// includes its own: an fp8-KV packet emits the fp8 opcode, and a bf16-only test
/// silently matches nothing there — the class of miss `derive_segments` records.
fn kv_write_row_field(op: u16, dst: Option<&String>) -> Option<usize> {
    if !dst.is_some_and(|n| n.starts_with("kv.")) {
        return None;
    }
    if op == DevOp::RmsNorm as u16 {
        // GLM/DeepSeek MLA: `kv_a_layernorm` -> `kv.{L}.ckv`, out_row0 in i[2]
        // (`runtime/amd/interp.hip`, PLOW_DOP_RMSNORM).
        Some(2)
    } else if op == DevOp::HeadNormRope as u16 || op == DevOp::HeadNormRopeFp8 as u16 {
        // Dense GQA k/v norm -> `kv.{L}.k`/`.v`, and MLA's k_rope -> `kv.{L}.krot`.
        // Both carry the write row in i[3].
        Some(3)
    } else {
        None
    }
}

/// Rebase one prefill program's instructions onto the chunk `[c0, c0+clen)`.
///
/// Split out of [`AmdEngine::patch_prefill`] so the rule can be tested without a
/// GPU: every one of these families was, at some point, patched positionally and
/// silently wrong, and a positional bug is invisible to anything but a test that
/// inspects the fields.
///
/// FOUR patch families, every one found BY IDENTITY rather than by position:
///
/// * a **KV-write site** ([`kv_write_row_field`]) → its row field = `c0`. This
///   covers dense GQA's k/v norm (`i[3]`), MLA's k_rope (`i[3]`) and MLA's latent
///   `kv_a_layernorm` (`i[2]`) with one rule, so prefill and decode cannot drift
///   about what a KV write is.
/// * `HeadNormRope`/`HeadNormRopeFp8` **with `fj[1] != 0`** → `i[3] = c0`. The
///   legacy dense-GQA test, kept because it keys on the KV RING STRIDE (`j[0]`,
///   which packs into `fj[1]`) rather than on a tensor name: a packet whose KV
///   tensors are not named `kv.*` still rebases. Its `fj[1] == 0` twin is the
///   *query* norm and must be left alone; patching it corrupts Q with a cache
///   row index. The two tests are a UNION, and on Gemma they agree site for site.
/// * `FlashPrefill`/`FlashPrefillFp8` → `i[4] = c0` (q_pos0) and
///   `i[1] = c0 + clen` (n_kv, everything written so far, not just this chunk).
/// * every **KDA op** ([`KDA_ROW_COUNT_OPS`]) → `i[0] = clen`, the REAL row count.
///
/// # Why KDA needs a row count where attention needs only a bound
///
/// The flash family above leaves its row count alone and bounds the KV *reads* at
/// `c0 + clen`: a padded row computes a garbage output that nothing ever reads,
/// because the lm_head samples row `clen - 1`. That convention is safe for
/// attention precisely because attention is STATELESS across the token axis —
/// each row's output depends only on rows behind it, so a junk row poisons only
/// itself.
///
/// KDA is not. Both of its arms CARRY STATE FORWARD along `t`, and the loop bound
/// is the baked `T`:
///
/// * `op_kda.h` conv — `for (t = 0; t < T; t++)` rolls the window left and shifts
///   `x[t]` into the newest tap. A zero pad row is not ignored; it is CONVOLVED,
///   and it evicts a real tap from a `W`-wide window. After `T - clen` pad rows a
///   `W = 4` window holds nothing but zeros.
/// * `op_kda.h` recurrence — the same `for (t = 0; t < T; t++)`, and each step
///   applies the decay `exp(a_log[h])` to the carried state. Pad rows contribute
///   no `k^T v` outer product, but they DO decay: the state handed to decode has
///   been multiplied by an extra `exp(a_log)^(T - clen)`.
///
/// So the state left after a chunk belongs to the padded BUCKET WIDTH rather than
/// to the prompt. Nothing reads the pad rows' outputs, but the next chunk and the
/// whole decode phase read the STATE, and it is wrong for every prompt that is not
/// exactly a bucket multiple — i.e. almost all of them. Setting `i[0] = clen` is
/// what makes a K3 prefill stop at the last real token.
///
/// `FlashMlaPrefill`/`FlashGatherPrefill` are deliberately NOT here. Their query
/// base is not an immediate: `d_flash_mla_decode` (the body both prefill wrappers
/// call) derives `qpos = kv_len[b] - n_tok + t` from the `in.kvlen` TENSOR, with
/// `n_tok` in `i[4]` and the causal end clamped to `qpos + 1`. So the chunk base
/// arrives through the `in.kvlen` upload in [`AmdEngine::prefill_prepare`], and
/// writing `c0` into any of `i[0..7]` here would overwrite a live operand
/// (n_batch / n_head / kv_stride / window / n_tok / kv_mask / grouping).
///
/// The fp8 twins are included in the flash test. Their absence is not a slowdown,
/// it is silence: on an fp8-KV packet the bf16-only test matches nothing, so every
/// flash window stays at whatever the compiler baked in. (Same root cause as the
/// class-4 miss in [`derive_segments`], which found 0 of 60 flash segments on such
/// a packet.) Note the two are INDEPENDENT axes: an fp8-*weight* packet keeps a
/// bf16 KV and emits plain `FlashPrefill`, so this must key on the OPCODE and
/// never on a precision flag.
/// Every KDA opcode whose `i[0]` is the token-row count `T`.
///
/// All six, not just the two that carry state. The conv and the recurrence are the
/// ones a pad row CORRUPTS, but `KdaGate` feeds the recurrence and `KdaGatedNorm`
/// consumes its output, and leaving those two at the bucket width would have them
/// read rows the shortened arms never wrote. Uniform `clen` keeps one row count
/// across the whole mixer, which is also the only version of this that can be
/// stated in one sentence.
const KDA_ROW_COUNT_OPS: &[DevOp] = &[
    DevOp::KdaConv,
    DevOp::KdaConv3,
    DevOp::KdaGate,
    DevOp::KdaStateStep,
    DevOp::KdaStateStepG,
    DevOp::KdaConvStateStepG,
    DevOp::KdaGatedNorm,
];

/// Where a prefill op carries its TOKEN-ROW COUNT, and whether that field is the
/// row count itself or a whole multiple of it.
///
/// Read by [`rebase_chunk_rows`] under `PLOW_RAGGED_CHUNK`; see its header for
/// why the multiple form exists and what the "== bucket width" guard buys.
///
/// `Rows` = the field IS `T`. `RowsTimes` = the field is `T * F` for some
/// per-instruction `F` (an element count over `[T, F]`), so it is RESCALED
/// rather than overwritten.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowField {
    Rows(usize),
    RowsTimes(usize),
}

/// The row-count field of every opcode a GLM/MLA prefill bucket emits.
///
/// DERIVED FROM THE PACKET, NOT FROM MEMORY: the op list is exactly the census of
/// `plowrt disasm --program 4096` on the shipped GLM-5.2 TP8 blob, and each field
/// index is the one `runtime/common/dev_isa.h` documents for that opcode. Four
/// opcodes in that census are deliberately ABSENT:
///
/// * `MoeGroupGluPf` / `MoeGroupDownPf` — they carry no `T` at all. Their work is
///   the sorted-row `meta` table `MoeAlignPf` builds, so shrinking align's `T`
///   shrinks them with no field of their own to patch. (This is also why a SHORT
///   chunk costs far less MoE than a padded one: with `clen` rows only
///   `clen * top_k` expert slots exist, so most of the 256 experts contribute no
///   tile and their weights are never streamed.)
/// * `Gemv` — the lm_head, `M = 1` with `a_row0` placed by
///   [`AmdEngine::patch_prefill`]. Its M is not a row count.
/// * `Argmax` / `XArgmaxFin` — vocabulary-dimensioned.
///
/// `FlashPrefill`/`FlashPrefillFp8` (dense GQA) are absent because MLA is the only
/// prefill family this axis has been measured on; leaving them out means a
/// non-MLA model under the flag runs the padded bucket — correct-just-slower
/// rather than untested.
const PREFILL_ROW_FIELDS: &[(DevOp, RowField)] = &[
    (DevOp::Embed, RowField::Rows(0)),
    (DevOp::RmsNorm, RowField::Rows(0)),
    (DevOp::HeadNormRope, RowField::Rows(0)),
    (DevOp::HeadNormRopeFp8, RowField::Rows(0)),
    (DevOp::Residual, RowField::RowsTimes(0)),
    (DevOp::Glu, RowField::RowsTimes(0)),
    (DevOp::Gemm, RowField::Rows(0)),
    (DevOp::GemmNorm, RowField::Rows(0)),
    (DevOp::GemmSmall, RowField::Rows(0)),
    (DevOp::GemmMed, RowField::Rows(0)),
    (DevOp::GemmWide, RowField::Rows(0)),
    (DevOp::GemmC5, RowField::Rows(0)),
    (DevOp::GemmGlu, RowField::Rows(0)),
    (DevOp::GemmFp8, RowField::Rows(0)),
    (DevOp::GemmMedFp8, RowField::Rows(0)),
    (DevOp::GemmSmallFp8, RowField::Rows(0)),
    (DevOp::GemmWideFp8, RowField::Rows(0)),
    (DevOp::GemmGluFp8, RowField::Rows(0)),
    (DevOp::GemmFp8Blk, RowField::Rows(0)),
    (DevOp::FlashMlaPrefill, RowField::Rows(4)),
    (DevOp::FlashMlaPrefillFp8, RowField::Rows(4)),
    (DevOp::MlaMergeFold, RowField::Rows(0)),
    (DevOp::MoeRouterTopkPf, RowField::Rows(4)),
    (DevOp::MoeAlignPf, RowField::Rows(0)),
    (DevOp::MoeCombinePf, RowField::Rows(2)),
    (DevOp::XReduce, RowField::RowsTimes(0)),
    (DevOp::XReduceTwoShot, RowField::RowsTimes(0)),
];

fn prefill_row_field(op: u16) -> Option<RowField> {
    PREFILL_ROW_FIELDS
        .iter()
        .find(|(o, _)| *o as u16 == op)
        .map(|&(_, f)| f)
}

/// [`rebase_chunk`], plus the RAGGED-M row shrink when `bucket` is `Some(T)`.
///
/// # What the shrink is for
///
/// A prefill bucket program is compiled at a fixed row count `T`, and the last
/// chunk of a prompt almost never has `T` real tokens. Without the shrink the
/// kernels compute all `T` rows and the padded ones are simply never read, so a
/// 1-token remainder costs a full `T`-row pass — and, worse, `plan_chunks` must
/// then pick the SMALLEST bucket covering the remainder to keep that waste down,
/// which is what turns a 4097-token prompt into `[4096, 128]`: two launches, the
/// second paying the whole T-invariant cost of a 78-layer pass. Measured on the
/// shipped config: 4096 -> 720.2 ms, 4097 -> 951.2 ms, for ONE more token.
///
/// With the shrink the row count is a runtime operand, so one 8192-bucket launch
/// carries 4097 real rows at the cost of ~4097 rows and the second launch
/// disappears. Every kernel this touches already tiles over `M`/`n_tok` and
/// bounds its own loads by it (`d_gemm_t`: `tm = ceil(M/BM)`, `r < M` on every A
/// fetch; `d_flash_mla_prefill`: `n_work = n_batch*n_tok*n_grp*nsplit`), so this
/// is not a new kernel contract — it is the operand finally being told the truth.
/// Workgroups whose tile index falls past the shortened range still run the
/// interpreter and still signal their successor counters, so the counter DAG is
/// untouched.
///
/// # The one operand that must move WITH it
///
/// `in.kvlen`. `d_flash_mla_decode` derives `qpos = kv_len - n_tok + t`, so
/// shrinking `n_tok` without shrinking `kv_len` shifts every query in the chunk
/// down by the padding. [`AmdEngine::prefill_prepare`] uploads `c0 + clen`
/// instead of `c0 + ch` under the same flag, and both sites read
/// [`AmdEngine::ragged_bucket`] so the pair cannot drift.
///
/// # Why the "field must equal the bucket width" guard
///
/// The mapping opcode -> field is a static table, but a field that holds the row
/// count in one packet may hold something else in another (the lm_head `Gemv`'s
/// `M = 1`; a `PLOW_GLM_XR_BAND` row-band `Gemm`'s `M = T/kb`). Rewriting one of
/// those would be silent and wrong. So a `Rows` field is patched only when it is
/// EXACTLY `T`, and a `RowsTimes` field only when it is an exact multiple of `T`
/// — anything else is left alone, which is always SAFE because computing the
/// padded row count is exactly what the engine did before this axis existed: pad
/// rows produce values nothing reads (the lm_head samples row `clen - 1`, `n_kv`
/// bounds every KV read at `c0 + clen`, and no prefill op reduces across the
/// token axis).
///
/// [`AmdEngine::refuse_unraggable`] refuses the one configuration where "left
/// alone" is NOT enough — row-banded collectives, where the band `Gemm` would be
/// skipped by the guard while its `XReduce` partner was rescaled — so the guard
/// never half-applies in silence.
fn rebase_chunk_rows(
    insts: &mut [DevInst64],
    names: &[String],
    c0: u32,
    clen: u32,
    bucket: Option<u32>,
) {
    for d in insts.iter_mut() {
        let op = d.op;
        if let Some(f) = kv_write_row_field(op, names.get(d.t[0] as usize)) {
            d.i[f] = c0;
        }
        if (op == DevOp::HeadNormRope as u16 || op == DevOp::HeadNormRopeFp8 as u16) && d.fj[1] != 0
        {
            d.i[3] = c0;
        } else if op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16 {
            d.i[4] = c0;
            d.i[1] = c0 + clen;
        } else if KDA_ROW_COUNT_OPS.iter().any(|&k| op == k as u16) {
            d.i[0] = clen;
        }
        let Some(t) = bucket.filter(|&t| t > 0 && clen < t) else {
            continue;
        };
        match prefill_row_field(op) {
            Some(RowField::Rows(f)) if d.i[f] == t => d.i[f] = clen,
            Some(RowField::RowsTimes(f)) if d.i[f] > 0 && d.i[f] % t == 0 => {
                d.i[f] = (d.i[f] / t) * clen
            }
            _ => {}
        }
    }
}

fn patch_tp_xaudit(insts: &mut [DevInst64], status_id: u32) {
    for d in insts {
        if matches!(
            DevOp::from_u16(d.op),
            Some(
                DevOp::XReduce | DevOp::XReduceTwoShot | DevOp::XReduceAddNorm | DevOp::XArgmaxFin
            )
        ) {
            d.i[7] = status_id + 1;
        }
    }
}

/// This rank's place in a TP group — everything the engine needs from
/// [`crate::exec::tp::TpGroup`], as plain device addresses.
///
/// Deliberately NOT a borrow of `TpRank`. The engine needs four numbers and a
/// base address; taking the rank itself would couple the AMD engine to the group
/// type, drag an `Arc<dyn Backend>` alongside the `Arc<HsaBackend>` it already
/// holds, and make the binding untestable without eight GPUs. Every field here
/// is one `TpRank` accessor.
#[derive(Clone, Copy, Debug)]
pub struct TpBind {
    /// This rank's index in the group — `PlowProgram::rank`.
    pub rank: u32,
    /// TP degree — `PlowProgram::n_gpu`.
    ///
    /// The interpreter's convention is that **0 means single-GPU** (`n_gpu: 0`
    /// in [`AmdEngine::kernarg`] when there is no group), so this is never 0
    /// here; a group of one is still `1`.
    pub n_gpu: u32,
    /// `[n_gpu]` device table of every rank's peer-region base —
    /// `PlowProgram::peer_scratch`. From `TpRank::peer_scratch_table`.
    pub peer_table: u64,
    /// This rank's cross-GPU counters, inside its own peer region —
    /// `PlowProgram::xctr`. From `TpRank::xctr`.
    pub xctr: u64,
    /// Reserved xctr id used by the compact device audit status line.
    pub xstatus_id: u32,
    /// This rank's peer-region base (`peer_scratch[rank]`). `act.og_tp` binds at
    /// offset 0 and `act.dg_tp` at [`TpBind::slot_b`], so the row-parallel
    /// o_proj/down write their partials where peers can read them.
    pub scratch_base: u64,
    /// Byte offset of partial slot B — `DevBlob::tp`'s `slot_bytes`, which is
    /// what `devgen` baked into every `XReduce`'s `i[2]`. Read from the blob
    /// rather than recomputed: a host that computed its own would put `dg_tp`
    /// where no peer reads it, and nothing would say so.
    pub slot_b: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct XAuditArgs {
    insts: u64,
    n_inst: u32,
    _pad: u32,
    xctr: u64,
    n_xctr: u32,
    n_gpu: u32,
    status: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct StateClearRange {
    base: u64,
    slot_stride: u64,
    words: u32,
    _pad: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct StateClearArgs {
    ranges: u64,
    n_ranges: u32,
    slot: u32,
}

const STATE_CLEAR_CHUNK: u64 = 256 * 1024;

fn state_clear_ranges(
    devp: &[DeviceMem],
    carried: &[(usize, u64)],
) -> Result<Vec<StateClearRange>> {
    let mut out = Vec::new();
    for &(i, stride) in carried {
        let m = devp.get(i).ok_or_else(|| {
            RuntimeError::Device(format!("carried-state tensor index {i} is out of range"))
        })?;
        if m.base == 0 || stride == 0 {
            continue;
        }
        if m.base % 4 != 0 || stride % 4 != 0 {
            return Err(RuntimeError::Device(format!(
                "carried-state range {i} is not u32-aligned: base={:#x} stride={stride}",
                m.base
            )));
        }
        let mut off = 0;
        while off < stride {
            let bytes = (stride - off).min(STATE_CLEAR_CHUNK);
            out.push(StateClearRange {
                base: m.base + off,
                slot_stride: stride,
                words: u32::try_from(bytes / 4).map_err(|_| {
                    RuntimeError::Device(format!("carried-state clear chunk is too large: {bytes}"))
                })?,
                _pad: 0,
            });
            off += bytes;
        }
    }
    Ok(out)
}

/// One chunk of a prefill plan: which bucket program runs it, and over which
/// absolute token range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkStep {
    /// Index of the compiled bucket program.
    pub prog: usize,
    /// Absolute position of the chunk's first row.
    pub c0: u32,
    /// REAL rows in the chunk; rows `[clen, t)` are padding.
    pub clen: u32,
}

/// Per-program local counter-bank state.
///
/// TP keeps `current` on the bank used by the last dispatch so diagnostic
/// snapshots still read that dispatch's counters. The other bank must be
/// re-armed successfully before a later TP dispatch may select it.
struct CounterBankState {
    current: Cell<u32>,
    inactive_ready: Cell<bool>,
}

impl CounterBankState {
    fn new() -> Self {
        Self {
            current: Cell::new(0),
            // Both banks are zeroed at allocation time.
            inactive_ready: Cell::new(true),
        }
    }

    fn current(&self) -> u32 {
        self.current.get()
    }

    fn inactive(&self) -> u32 {
        1 - self.current()
    }

    fn inactive_ready(&self) -> bool {
        self.inactive_ready.get()
    }

    /// Select the already-clean inactive bank for a TP dispatch.
    ///
    /// Returns `false` for the single-bank fallback, whose caller must re-arm
    /// the current bank synchronously. A failed/omitted inactive clear leaves
    /// `inactive_ready == false`, so stale local counters cannot be reused.
    fn begin_tp(&self, double_buffered: bool) -> std::result::Result<bool, ()> {
        if !double_buffered {
            return Ok(false);
        }
        if !self.inactive_ready.replace(false) {
            return Err(());
        }
        self.current.set(self.inactive());
        Ok(true)
    }

    fn mark_inactive_ready(&self) {
        self.inactive_ready.set(true);
    }

    fn select_rearmed_inactive(&self) {
        self.current.set(self.inactive());
        // The old current bank was dirtied by the dispatch that just launched.
        self.inactive_ready.set(false);
    }
}

/// One program's device-resident tables.
struct AmdProg {
    t: u32,
    n_inst: u32,
    trace_records: usize,
    n_counter: u32,
    d_inst: DeviceMem,
    d_stream: DeviceMem,
    d_sofs: DeviceMem,
    d_slen: DeviceMem,
    d_waits: DeviceMem,
    d_succs: DeviceMem,
    d_ctr: DeviceMem,
    /// `[n_cu][n_seg+1]` per-(CU, segment) window bounds into each CU's own
    /// stream slice, so a segment launch starts at its own first entry instead
    /// of rescanning the whole stream and filtering. The static path's analogue
    /// of `AmdGq::d_seg_ofs`. Derived from the uploaded stream, never read from
    /// the blob — see `PlowProgram::seg_ofs` in `runtime/common/dev_isa.h`.
    d_seg_ofs: DeviceMem,
    /// `[n_seg]` wave classes.
    seg_class: Vec<u8>,
    /// Global-queue tables; `None` when the blob carries no GQ appendix.
    gq: Option<AmdGq>,
    /// L2-domain placement (`PLOW_L2_PLACE`): domains `gq_seg_ofs` is windowed by, 0 = not
    /// placed. When non-zero the `seg` windows are L2 DOMAINS rather than wave classes, so the
    /// program is dispatched in ONE launch with every domain draining concurrently — see
    /// [`AmdEngine::run`].
    l2_domains: u32,
    /// Base counter id of the two-level maintenance scratch; 0 = off. See `DevProgram::hier_base`.
    hier_base: u32,
    /// Bytes in ONE counter bank — `n_counter * CTR_STRIDE_U32 * 4`, i.e. what
    /// the old single-bank allocation was in total.
    ctr_span: u64,
    /// Local counter/cursor bank state. See [`AmdEngine::run`] and the TP
    /// begin/post-launch methods.
    bank: CounterBankState,
}

struct AmdGq {
    d_stream: DeviceMem,
    d_seg_ofs: DeviceMem,
    /// One cursor LINE per segment, not one word. RUNSEG enqueues every
    /// segment without a host wait and zeroes state once before the loop, so a
    /// shared cursor would be corrupted by the segment that ran first.
    d_cursor: DeviceMem,
    n_seg: u32,
    /// Bytes in ONE cursor bank. The allocation holds `ctr_banks` of them.
    cur_span: u64,
}

/// The AMD serving engine.
pub struct AmdEngine {
    be: Arc<HsaBackend>,
    arch: String,
    n_cu: u32,
    progs: Vec<AmdProg>,
    /// Index of the WIDEST decode program — always last (`n_prog - 1`).
    decode: usize,
    /// Index of the FIRST decode program: the bottom of the DECODE BATCH LADDER
    /// (`PLOW_DECODE_BATCH_LADDER`). Without a ladder this equals [`Self::decode`], so
    /// `dec_lo..=decode` is a one-element range and every path is what it was.
    ///
    /// Programs `[0, dec_lo)` are the prefill bucket ladder; `[dec_lo, n_prog)` are decode
    /// rungs at ascending sequence widths, all sharing ONE tensor table sized at the widest.
    dec_lo: usize,
    devp: Vec<DeviceMem>,
    /// Owner of the one allocation the ordinarily-allocated tensors are carved
    /// out of; `devp` then holds **views** into it. Unlike the CUDA side, this
    /// wins on BOTH axes, measured on 8×MI355X loading Kimi-K3 TP8 (5408 carved
    /// tensors per rank, 22.84 GiB of named weights):
    ///
    /// | | slab | per-tensor |
    /// |---|---|---|
    /// | peak VRAM per card | 204 579 MiB | 205 904 MiB |
    /// | `alloc_ms`, named tensors | 96–266 | 6410–8802 |
    /// | wall, named tensors | 6.0–6.7 s | 13.0–16.7 s |
    ///
    /// **Memory: 1325 MiB per card, 10.35 GiB across the eight.** ROCr reports a
    /// 4 KiB granule and then ignores it — under 2 MiB it hands back the next
    /// POWER OF TWO with a 32 KiB floor, at or above 2 MiB it rounds to a 2 MiB
    /// multiple. A 1.4 MiB expert projection commits 2 MiB, 42.9% lost; a 12 KiB
    /// norm vector commits 32 KiB. One allocation pays that rounding once.
    ///
    /// **Time: ~7–8.5 s of driver time per rank, halving this phase.** This one
    /// contradicts the obvious microbenchmark, so do not re-derive it from one:
    /// 737 uniform 30 MiB allocations on an IDLE card cost 8.8 ms total, which
    /// says the call is nearly free and is why the first version of this comment
    /// claimed the slab was memory-only. The real load is not that shape — 5408
    /// unevenly sized tensors interleaved with 168 GiB of expert buffers, eight
    /// ranks against one driver — and there the per-call cost is three orders of
    /// magnitude worse. Measure this on the model, never in isolation.
    ///
    /// The packed-expert buffers are unaffected (`alloc_ms` ~1.5–2.1 s either
    /// way): `bind_packed_experts` already carves all of a layer's experts out
    /// of two allocations, which is this same trick applied earlier.
    ///
    /// Views never free, so this owner must outlive them; both live on this
    /// struct, and a view's `Drop` is a no-op, so field order cannot matter.
    ///
    /// `PerTensor` when the single allocation was refused and the loader fell
    /// back to per-tensor allocation — a fragmented card can decline one big
    /// block and still satisfy many small ones. The `Vmm` arm (lazy commit
    /// overlapped with the upload, as `exec::gpu`) is **opt-in** here
    /// (`PLOW_WEIGHT_VMM=1`) until it is measured on AMD hardware — see
    /// `asset::checkpoint::weight_vmm_amd_enabled`.
    _weight_slab: WeightSlab,
    d_tens: DeviceMem,
    tensor_names: Vec<String>,
    /// Per-MoE-layer PACKED expert buffers (weights, then block scales). Never
    /// read through here — the packet reaches them only through the addresses in
    /// `expert_weight_table`/`expert_scale_table` — but they are owning handles,
    /// so dropping them would free the memory those tables point at.
    _expert_bufs: Vec<DeviceMem>,

    /// Per-(workgroup, packet) `PlowTraceRec` buffer for the widest program,
    /// allocated only when `PLOW_TRACE_RAW` is set. The interpreter treats a
    /// null `trace` pointer as "tracing off" and then does not even read the
    /// clock, so an untraced build pays nothing for this field being here.
    ///
    d_trace: Option<DeviceMem>,
    trace_bytes: usize,
    /// Extent of the last program dispatched into `d_trace`.
    trace_write_bytes: Cell<usize>,

    k_prefill: HsaKernel,
    k_decode: HsaKernel,
    k_xaudit: Option<HsaKernel>,
    k_state_clear: Option<HsaKernel>,
    /// Task-13: low-rung decode tier ladder, ascending (max_rung, kernel).
    decode_tiers: Vec<(u32, HsaKernel)>,
    k_flash: Option<HsaKernel>,
    sched_prefill: Sched,
    sched_decode: Sched,
    _modules: Vec<Module>,

    /// Pinned copy of the decode program's instructions, patched in place so
    /// the per-step slice upload is contiguous in PINNED memory.
    h_inst: HsaPinned,
    /// Pinned scalar staging. `hsa_amd_memory_lock` is syscall-class; the
    /// reference driver measured that pinning per step "cost more than the
    /// whole forward pass", so every hot-path transfer uses pre-pinned memory.
    h_scalar: HsaPinned,
    /// Pinned zero page for the counter/cursor re-arm.
    h_zero: HsaPinned,
    /// Pinned staging for a prefill program's whole instruction array.
    h_pf_inst: HsaPinned,
    d_state_clear: Option<DeviceMem>,
    n_state_clear: u32,
    /// Pristine host copy of each program's instructions. Prefill patching
    /// rebuilds from these every chunk: `c0` changes per chunk, so patches must
    /// not accumulate.
    pf_src: Vec<Vec<DevInst64>>,

    kvrow: Vec<u32>,
    /// KV-append sites whose write row is `i[2]`, not `i[3]` — GLM-5.2's latent
    /// `RmsNorm` half. Empty for every packet that declared its own sites.
    kvrow_i2: Vec<u32>,
    kvrow_span: Option<(usize, usize)>,
    t_ids: Option<usize>,
    t_pos: Option<usize>,
    t_kvlen: Option<usize>,
    /// `in.parked`, the per-row SKIP mask (non-zero = park). Present only on a sequence-rows
    /// (batched-decode) blob; `None` means every row always participates.
    t_active: Option<usize>,
    t_logits: Option<usize>,
    max_ctx: usize,

    weights_bound: bool,
    /// Decode batch — the number of sequences one decode dispatch advances.
    ///
    /// Derived from `in.kvlen`, which the compiler sizes at `batch * 4` bytes,
    /// NOT from the decode program's `t`. The two agree on a well-formed blob
    /// and the tensor is the one the kernel actually indexes, so a disagreement
    /// should surface as a bind-time error rather than as every sequence past
    /// the first reading a length nobody wrote.
    batch: usize,
    /// Host mirror of the device tensor-pointer table, so a KV rebase is one
    /// edit + one upload instead of a read-modify-write off the device.
    tens_table: Vec<u8>,
    /// `(tensor index, per-sequence byte stride)` for every `kv.*` buffer.
    ///
    /// The cache is allocated `[batch][kv_head][ring][hd]` (`devgen`
    /// `b.tensor("kv.{l}.k", db * ...)`), so sequence `s`'s block is exactly
    /// `s * bytes/batch` in and a base-pointer shift addresses it exactly.
    /// That is what lets the SINGLE-SEQUENCE prefill program fill any slot: it
    /// writes with the legacy `hh * out_stride + row` formula, which is
    /// sequence 0's block relative to whatever base the pointer table holds.
    kv_slot_stride: Vec<(usize, u64)>,
    /// Which sequence slot the KV pointers are currently rebased onto.
    kv_slot: usize,
    /// `(tensor, per-slot bytes)` for the CARRIED recurrent state — KDA `state`
    /// and `conv_state`, per-slot strided, `blkres` excluded for the reason
    /// [`AmdEngine::begin_slot`] gives. This is what a prefix snapshot copies.
    carried_slot: Vec<(usize, u64)>,
    /// Per-slot snapshot of `carried_slot` taken at a prompt prefix boundary,
    /// for [`AmdEngine::restore_carried`]. `None` until a slot arms one.
    prefix_snap: Vec<Option<DeviceMem>>,
    /// `(alternate bank, legacy bank, bytes)` for the B1 KDA conv-window ping-pong arm.
    kda_conv_bank_pairs: Vec<(u64, u64, u64)>,
    /// Legacy prefill updates only bank 0; a set bit requires one bank0→bank1 mirror before
    /// decode selects a source by absolute-position parity.
    kda_conv_alt_stale: Vec<bool>,
    /// VMM-backed FULL-attention KV, or `None` for the flat allocation.
    ///
    /// The tensor table still holds ONE base per `kv.{l}.{k,v}` and
    /// [`AmdEngine::kv_rebase`] still shifts it by `slot * bytes/batch` — the
    /// base is simply a VA reservation instead of a `hsa_amd_memory_pool_allocate`
    /// result, and physical granules are mapped at each sequence's decode
    /// frontier. No block table, no per-block indirection, no kernel change.
    vmm: Option<VmmKv>,
    /// `(blocks, i[8])` of the last lm_head found — the fields that differ
    /// between two packets whose op and operands are identical.
    lm_detail: std::cell::RefCell<Option<(u16, [u32; 8], usize, Vec<u16>)>>,
    /// Pristine stream per program, for the scheduling diagnostic above.
    pf_stream: Vec<Vec<packet::dev::StreamEnt>>,

    /// This rank's place in its TP group, or `None` for single-GPU. Drives the
    /// four cross-GPU kernarg fields; the peer bindings it implies were made at
    /// load.
    tp: Option<TpBind>,

    /// Host-side accounting of segmented dispatch: enqueue time proves the host
    /// is NOT in the loop between segments; drain time is the GPU running every
    /// segment back to back.
    pub seg_enq_us: f64,
    pub seg_drain_us: f64,
    pub seg_launches: u64,

    /// A/B CONTROL, not a tuning knob (`PLOW_SEG_WINDOW=0`). Clears
    /// `PlowProgram::seg_ofs`, so the interpreter falls back to scanning each
    /// CU's whole stream and filtering on `seg` — the pre-window behaviour.
    ///
    /// It exists because the two arms must be compared in ONE process against
    /// ONE code object: rebuilding to compare would confound the measurement
    /// with a different build, and a 31 GB weight load per arm prices the
    /// comparison out. Both arms must produce IDENTICAL tokens; if they do not,
    /// the window is wrong and no timing from either arm means anything.
    seg_window: bool,
}

impl AmdEngine {
    /// Bring the engine up from a compiled blob and a directory of gfx950 code
    /// objects.
    ///
    /// With `checkpoint`, the model weights are bound by tensor name and the
    /// engine produces real tokens. Without it, every tensor is still allocated
    /// and the schedule still runs at full size — so the TIMING is real and the
    /// TOKENS are not. That mode exists because it isolates dispatch cost from
    /// a multi-minute weight upload, and it must never be mistaken for the
    /// other; the load log says which one you got.
    pub fn load(
        be: Arc<HsaBackend>,
        blob_path: &Path,
        hsaco_dir: &Path,
        checkpoint: Option<&Path>,
    ) -> Result<Self> {
        Self::load_rank(be, blob_path, hsaco_dir, checkpoint, None, None)
    }

    /// Bring up the VMM-backed KV pool, or `None` to keep the flat allocation.
    ///
    /// **This is growth, not paging.** One contiguous VA reservation per
    /// (full layer, K|V) spans the whole `[batch][kvh][max_ctx][hd]` tensor —
    /// exactly the shape `devgen` declares — so the tensor table keeps ONE base
    /// per tensor and [`AmdEngine::kv_rebase`]'s `base + slot·stride` is
    /// unchanged. Physical granules are mapped at each sequence's decode
    /// frontier. There is no block table and no indirection in the hot loop.
    ///
    /// What it buys: a slot's HBM follows its live context instead of
    /// `max_ctx`. On Gemma-4-31B (10 full layers, `kvh_full` 4, `hd_full` 512
    /// bf16) the head window is `max_ctx * 1024 B` and the granule is 2 MiB, so
    /// at `max_ctx` 16384 that is 16 MiB in 8 blocks of 2048 rows each.
    /// MEASURED at B=8, ~1k live context: 75.55 GiB resident vs 83.05 flat —
    /// **7.50 GiB of the 10.0 GiB full-layer reservation reclaimed**, TPOT
    /// unchanged (38.0 vs 39.0 ms). The ratio is `max_ctx / live_ctx`, so it
    /// grows with the emitted context: at 128k the same pool would hold 2.5 GiB
    /// of an 80 GiB tensor.
    ///
    /// Every failure is a warn + `None` — the flat path is always correct, so a
    /// missing `config.json` or a geometry the blob does not match must not
    /// stop the engine loading. Off by default (`--amd-vmm-kv` /
    /// `PLOW_VMM_KV=1`).
    ///
    /// Only FULL-attention `kv.{l}.k`/`.v` are backed. Sliding-window rings are
    /// bounded by `window`, not by context, so they have nothing to grow into,
    /// and fp8 scale tensors are 1/128th the size — both stay flat, which is
    /// also what the CUDA path does.
    fn vmm_bringup(
        be: &Arc<HsaBackend>,
        blob: &DevBlob,
        checkpoint: Option<&Path>,
    ) -> Option<VmmKv> {
        if !crate::config::RuntimeConfig::get().amd.vmm_kv {
            return None;
        }
        if !be.has_vmm() {
            tracing::warn!("PLOW_VMM_KV=1 but this ROCr has no hsa_amd_vmem_* — vmm off");
            return None;
        }
        let ckpt = checkpoint?;
        let find = |name: &str| blob.tensors.iter().position(|t| t.name == name);
        let bytes_of = |name: &str| find(name).map(|i| blob.tensors[i].bytes);
        let max_ctx = (bytes_of("in.pos")? / 4) as u32;
        let batch = (bytes_of("in.kvlen")? / 4).max(1) as u32;

        let mut geo = match VmmGeometry::from_config(ckpt, max_ctx, batch) {
            Some(g) => g,
            None => {
                tracing::warn!("vmm off: no usable KV geometry in config.json");
                return None;
            }
        };
        // KV dtype from the blob, not from config.json: the emitter declares
        // `kv.{l}.k_scale` iff that layer's cache is fp8 e4m3 (1 B/elem).
        // Presence is the discriminator — byte size alone is ambiguous
        // (2x ring vs 2x elem).
        geo.elem = match geo.full_layers.first() {
            Some(&l) => match find(&format!("kv.{l}.k_scale")) {
                Some(_) => 1,
                None => 2,
            },
            None => return None,
        };
        // Every full layer's declared bytes must equal the batch-major shape at
        // that elem. A mismatch is geometry drift, and backing a VA range the
        // kernel indexes differently is a silent wrong token.
        for &l in &geo.full_layers {
            for t in ["k", "v"] {
                match bytes_of(&format!("kv.{l}.{t}")) {
                    Some(b) if b == geo.full_tensor_bytes() => {}
                    other => {
                        tracing::warn!(
                            layer = l,
                            declared = ?other,
                            expected = geo.full_tensor_bytes(),
                            "vmm off: full-layer KV bytes mismatch"
                        );
                        return None;
                    }
                }
            }
        }

        // The block is the GROWTH quantum. On ROCr `hsa_amd_vmem_set_access`
        // costs ~3-4 us per CALL and is flat in size (measured, gfx950/ROCm
        // 7.2.4 — `tests/hsa_vmm.rs`), where CUDA measured `cuMemSetAccess` at
        // ~69 us per 2 MiB granule. That 20x is why the AMD default is the
        // granule itself instead of the 64 MiB-class block the CUDA feasibility
        // review settled on: the finest quantum the hardware can map costs
        // nothing extra here, and finer means less HBM held per slot.
        // Env first (tests flip PLOW_VMM_BLOCK_MIB mid-process, after the
        // config snapshot), then `--amd-vmm-block-mib`. 0 = query the device
        // granularity (2 MiB measured on gfx950 — also the config default).
        let cfg_mib = crate::config::RuntimeConfig::get().amd.vmm_block_mib as u64;
        let block_hint =
            crate::config::RuntimeConfig::env_parse_or::<u64>("PLOW_VMM_BLOCK_MIB", cfg_mib) << 20;
        let block_hint = match block_hint {
            0 => VmmOps::granularity(&**be).ok()?,
            b => b,
        };
        match VmmKv::new(Arc::clone(be) as Arc<dyn VmmOps>, geo, block_hint, 0) {
            Ok(mut kv) => {
                kv.enable_block_pool(crate::memory::vmm::kv_pool_cap_from_env());
                Some(kv)
            }
            Err(e) => {
                tracing::warn!(error = %e, "vmm off: pool bringup failed");
                None
            }
        }
    }

    /// Bring up ONE RANK of a tensor-parallel group.
    ///
    /// With `tp = None` this is [`AmdEngine::load`] and every path below is the
    /// single-GPU one, bit-for-bit. With `Some`, three things change and nothing
    /// else does:
    ///
    /// 1. weights bind this rank's **shard** (`crate::asset::shard`);
    /// 2. `act.og_tp`/`act.dg_tp` are bound into the **peer region** instead of
    ///    ordinary VRAM, so the row-parallel partials are peer-visible;
    /// 3. the kernarg carries `rank`/`n_gpu`/`xctr`/`peer_scratch`.
    ///
    /// The blob's own declared TP degree must match the group's — see the check
    /// below, which is the difference between a clear refusal and a rank
    /// silently binding a quarter of a weight it needed all of.
    pub fn load_rank(
        be: Arc<HsaBackend>,
        blob_path: &Path,
        hsaco_dir: &Path,
        checkpoint: Option<&Path>,
        tp: Option<TpBind>,
        shared_ckpt: Option<Arc<crate::asset::checkpoint::Checkpoint>>,
    ) -> Result<Self> {
        let t_rank = Instant::now();
        let raw = std::fs::read(blob_path)
            .map_err(|e| RuntimeError::Device(format!("read {}: {e}", blob_path.display())))?;
        let arch = EngineDevice::arch(&*be);
        let n_cu_dev = EngineDevice::sm_count(&*be);
        // ACCEPT L2-DOMAIN PLACEMENT AND CHECK IT AT OBJECT LOAD, rather than making the operator
        // assert it by env. scripts/build_gfx942.sh now ships -DPLOW_L2_PLACE_DISPATCH by default
        // and plowc places gfx942 blobs by default, so the env gate had become a broken default:
        // a stock build would emit a placed blob and then refuse to load it. The real guard is
        // stronger than the env var ever was -- `plow_l2_place_dispatch_1` is checked against the
        // OBJECT below, so a genuinely mismatched pairing is still refused, by inspection instead
        // of by assertion. PLOW_L2_PLACE_DISPATCH=1 still works for anyone scripting it.
        let mut blob = DevBlob::parse_l2(&raw, true)?;
        let tp_audit_compact =
            tp.is_some() && crate::config::RuntimeConfig::get().amd.tp_audit_compact;
        if tp_audit_compact {
            let status_id = tp.expect("checked").xstatus_id;
            for p in &mut blob.progs {
                patch_tp_xaudit(&mut p.insts, status_id);
            }
        }
        // WHICH PHASE is L2-placed, not "is anything placed". `Builder::finish` skips placement
        // per PROGRAM when that program is segmented, and AMD prefill always is -- so a normal
        // gfx942 blob has a PLACED DECODE program and UNPLACED prefill ones. Requiring the
        // dispatch axis on every object would then reject the stock build over its prefill
        // objects, which correctly do not carry it (the axis is scoped to the decode rows because
        // a set-wide define deadlocks -- see scripts/build_gfx942.sh).
        //
        // WHICH PROGRAMS ARE DECODE. Everything from `dec_ix` on is a decode rung of the
        // DECODE BATCH LADDER (`PLOW_DECODE_BATCH_LADDER`); everything before it is a prefill
        // bucket. Without a ladder this is `progs.len() - 1` and the split is the one every
        // caller has always used.
        let dec_ix = {
            let pt: Vec<u32> = blob.progs.iter().map(|p| p.t).collect();
            packet::devbuild::decode_rung_lo(&pt)
        };

        // THIS USED TO ASK `p.t == 1` / `p.t > 1`, WHICH IS A BUG A BATCHED BLOB ALREADY HAD.
        // A decode program emitted at `PLOW_DECODE_BATCH=16` has `t == 16`, so it counted as a
        // PREFILL program: its (correct, default-on) L2 placement was attributed to the prefill
        // object, which is not built with the axis, and every batched gfx942 blob was refused
        // at load unless it was re-emitted with PLOW_L2_PLACE=0 — giving up the -12% placement
        // win to work around a misclassification. Splitting at `dec_ix` asks each object about
        // the programs it will actually be handed.
        let decode_l2_placed = blob.progs[dec_ix..].iter().any(|p| p.l2_domains > 0);
        let prefill_l2_placed = blob.progs[..dec_ix].iter().any(|p| p.l2_domains > 0);

        // The blob's n_cu is the grid the schedule was COMPILED for. A device
        // with a different CU count cannot run it: `stream_ofs`/`stream_len` are
        // [n_cu] and workgroup w reads slot w, so a smaller grid silently drops
        // every stream above it and a larger one reads past the table.
        //
        // `--amd-oversub` / PLOW_OVERSUB=1 (expert): accept an OVERSUBSCRIBED
        // grid — blob.n_cu a
        // multiple of the device's CU count — to co-locate several workgroups
        // per CU so one workgroup's gate poll hides behind a sibling's body.
        // The launch grid follows blob.n_cu, so this is only sound when the
        // OBJECT's resource envelope actually fits that many co-resident
        // workgroups (e.g. the occ4 profile at 2/CU: 104 VGPR, 30.7 KB LDS);
        // the persistent kernel SPINS on counters, so a non-resident workgroup
        // is not "slow", it is a DEADLOCK. No occupancy oracle is consulted
        // here — that is why this is env-gated instead of a default. L2-domain
        // placement must be OFF in the blob (the wg->domain map assumes
        // grid == n_cu_dev).
        let oversub_ok = blob.n_cu > n_cu_dev
            && blob.n_cu % n_cu_dev == 0
            && crate::config::RuntimeConfig::get().amd.oversub;
        if blob.n_cu != n_cu_dev && !oversub_ok {
            return Err(RuntimeError::Device(format!(
                "blob compiled for n_cu={} but this device has {n_cu_dev} CUs — \
                 recompile the packet with --n-cu {n_cu_dev} (or, for an oversubscribed \
                 grid on a co-resident object, set PLOW_OVERSUB=1)",
                blob.n_cu
            )));
        }
        if blob.progs.is_empty() {
            return Err(RuntimeError::Device("blob carries no programs".into()));
        }

        // The blob's declared sharding and the caller's group must agree, and
        // the mismatch is refused HERE, before a byte of the checkpoint is read.
        //
        // Without this a sharded blob on the single-GPU path dies ~60 GiB later
        // at the first projection with `SIZE MISMATCH
        // model.layers.0.self_attn.q_proj.weight (blob says 5505024 B,
        // checkpoint has 22020096 B)` — which names the symptom, not the cause,
        // and reads as a corrupt packet rather than as "you asked for one GPU
        // and handed me a four-way shard". The blob knows the answer (`DevTp`,
        // recovered from its own collectives); nobody was asking it.
        //
        // The reverse mismatch is worse and equally refused: an UNSHARDED blob
        // run under a TP group has no `XReduce` at all, so the ranks would each
        // compute the whole layer, never reduce, and produce N identical
        // single-GPU tokens at N times the cost — a "working" run that has
        // silently done nothing parallel.
        let n_gpu = tp.map_or(1, |t| t.n_gpu);
        let rank = tp.map_or(0, |t| t.rank);
        match (blob.tp, tp) {
            (Some(b), _) if b.n_gpu != n_gpu => {
                return Err(RuntimeError::Device(format!(
                    "this packet is SHARDED for tp={} (hidden={}, partial slot {} B) but \
                     this engine is bringing up {n_gpu} rank(s). Every projection in it \
                     is 1/{} wide, so binding it here would fail at the first weight \
                     with a size mismatch. Run it on {} devices, or recompile: \
                     plowc ... --num-gpus {n_gpu}",
                    b.n_gpu, b.hidden, b.slot_bytes, b.n_gpu, b.n_gpu
                )));
            }
            (None, Some(_)) => {
                return Err(RuntimeError::Device(format!(
                    "this packet carries NO collective, so it is compiled for a single \
                     GPU, but a TP group of {n_gpu} was requested. Each rank would run \
                     the whole model and never all-reduce — {n_gpu} identical tokens for \
                     {n_gpu} times the hardware. Recompile: plowc ... --num-gpus {n_gpu}"
                )));
            }
            _ => {}
        }
        if let (Some(b), Some(t)) = (blob.tp, tp) {
            // `dg_tp` is bound at this offset and peers read it at the offset
            // THEY were told. devgen bakes one value into every program's
            // `XReduce.i[2]`; if the host's differs, each rank publishes its
            // `down` partial where no peer looks and the reduction silently sums
            // stale memory.
            if b.slot_bytes != t.slot_b {
                return Err(RuntimeError::Device(format!(
                    "peer layout disagrees with the packet: the host binds partial slot \
                     B at {} B but the packet's XReduce reads it at {} B. Every rank's \
                     `down` partial would land where no peer reads it.",
                    t.slot_b, b.slot_bytes
                )));
            }
            if rank >= n_gpu {
                return Err(RuntimeError::Device(format!(
                    "rank {rank} is outside a group of {n_gpu}"
                )));
            }
        }

        // --- scheduler selection -------------------------------------------
        // Default is the global queue on both phases; it is bit-exact to static
        // and measured faster (the kernel side has GQ beating static prefill by
        // 8.4% on 31B at T=1024). It is downgraded, never silently: no GQ
        // appendix in the blob, or no `_gq` object on disk, and this says so.
        let has_gq = blob.progs.iter().all(|p| !p.gq_stream.is_empty());
        let rt = &crate::config::RuntimeConfig::get().amd;
        let mut sched_prefill = Sched::GlobalQueue;
        let mut sched_decode = Sched::GlobalQueue;
        if let Some(v) = rt.global_queue.as_deref() {
            let s = if v != "0" {
                Sched::GlobalQueue
            } else {
                Sched::Static
            };
            sched_prefill = s;
            sched_decode = s;
        }
        if rt.static_both {
            sched_prefill = Sched::Static;
            sched_decode = Sched::Static;
        }
        if rt.static_prefill {
            sched_prefill = Sched::Static;
        }
        if rt.static_decode {
            sched_decode = Sched::Static;
        }
        if !has_gq && (sched_prefill == Sched::GlobalQueue || sched_decode == Sched::GlobalQueue) {
            tracing::info!("blob carries no GQ appendix — both phases fall back to static");
            sched_prefill = Sched::Static;
            sched_decode = Sched::Static;
        }

        let variant = Variant::detect(&blob.progs);
        // Which MLA/MoE-prefill object the PREFILL phase needs — scanned from
        // every program (the prefill buckets carry these opcodes, the decode
        // program never does), so a bucket-only decode packet stays `None` and
        // a whole-layer GLM/Kimi/DeepSeek prefill packet selects the object
        // that actually has the arms instead of silently falling back to
        // `interp_prefill{,_gq}.elf`, whose `default:` case does not trap.
        let prefill_arm = PrefillArm::detect(&blob.progs);

        // THE WIDEST GEMV EACH OBJECT WILL BE HANDED, split the way the objects
        // are. The decode program runs on the decode object; every prefill
        // bucket runs on the prefill object OR the flash object (a flash segment
        // is a prefill segment), so both of those must cover the prefill maximum.
        // Split rather than one global max because the buckets are compiled
        // independently: prefill and flash take `op_gemm.h`'s default MM=1 and
        // legitimately serve the M=1 lm_head GEMV, and folding a batched decode's
        // M into their requirement would refuse them for work they never do.
        // `dec_ix` (defined above) is the split, NOT `progs.len() - 1`: with a DECODE BATCH
        // LADDER the last four programs are also decode, and putting them in the prefill half
        // asks the prefill object to cover a batched decode's GEMV — which it legitimately was
        // not built for, so a laddered blob refused itself at load with "widest GEMV asks for
        // M=8 ... interp_prefill_fp8_gq.elf was compiled PLOW_GEMV_MM=1". Loudly, which is the
        // guard working; the other shape of that mistake is rows 1..7 left STALE with no fault.
        let need_m_decode = required_gemv_m(&blob.progs[dec_ix..]);
        let need_m_prefill = required_gemv_m(&blob.progs[..dec_ix]);
        // Split the same way as the GEMV bucket, and for the same reason: the K3/KDA arms are in
        // BOTH buckets (a K3 layer runs the same graph at T=1 and T>1), so each object has to be
        // asked about the phase it actually serves rather than about the blob as a whole.
        let need_k3_decode = required_k3_op(&blob.progs[dec_ix..]);
        let need_k3_prefill = required_k3_op(&blob.progs[..dec_ix]);
        let need_dspark_decode = requires_dspark_noncausal(&blob.progs[dec_ix..]);
        let need_dspark_prefill = requires_dspark_noncausal(&blob.progs[..dec_ix]);
        let need_spec_verify_decode = requires_k3_spec_verify(&blob.progs[dec_ix..]);
        let need_spec_verify_prefill = requires_k3_spec_verify(&blob.progs[..dec_ix]);
        let need_a4w4_decode = required_moe_pf_a4w4(&blob.progs[dec_ix..]);
        let need_kda_conv_step_db = required_kda_conv_step_db(&blob.progs[dec_ix..]);
        let legacy_kda_decode = first_op_in(&blob.progs[dec_ix..], KDA_CONV_STEP_DB_REPLACED_OPS);
        if let Some(p) = blob.progs[..dec_ix].iter().find(|p| {
            p.insts
                .iter()
                .any(|i| i.op == DevOp::KdaConvStateStepG as u16)
        }) {
            return Err(RuntimeError::Device(format!(
                "KdaConvStateStepG is B1 decode-only, but prefill program T={} dispatches it",
                p.t
            )));
        }
        if let Some(p) = blob.progs[dec_ix..].iter().find(|p| {
            p.t != 1
                && p.insts
                    .iter()
                    .any(|i| i.op == DevOp::KdaConvStateStepG as u16)
        }) {
            return Err(RuntimeError::Device(format!(
                "KdaConvStateStepG is B1-only, but decode program T={} dispatches it",
                p.t
            )));
        }
        // Gemma-4 MoE, both halves, per phase. Same silent-NOP argument as K3 above.
        let need_gm_decode = first_op_in(&blob.progs[dec_ix..], MOE_GEMMA_OPS);
        let need_gm_prefill = first_op_in(&blob.progs[..dec_ix], MOE_GEMMA_OPS);
        let need_gmpf_decode = first_op_in(&blob.progs[dec_ix..], MOE_GEMMA_PF_OPS);
        let need_gmpf_prefill = first_op_in(&blob.progs[..dec_ix], MOE_GEMMA_PF_OPS);
        // The KV-encoding SWAP, split per phase for the same reason: the decode object carries
        // FLASH_*_DECODE and the prefill object FLASH_*_PREFILL, so asking each about the blob as
        // a whole would refuse a decode object for a prefill opcode it never runs.
        let need_fp8kv_decode = required_kv_op(&blob.progs[dec_ix..], FP8_KV_OPS);
        let need_fp8kv_prefill = required_kv_op(&blob.progs[..dec_ix], FP8_KV_OPS);
        let need_bf16kv_decode = required_kv_op(&blob.progs[dec_ix..], BF16_KV_OPS);
        let need_bf16kv_prefill = required_kv_op(&blob.progs[..dec_ix], BF16_KV_OPS);
        // Will derive_segments route any FlashMlaPrefill segment to the flash object?
        // Then that object MUST carry the V2 arm — the dispatch default is a silent skip.
        let need_mla_v2 = mla_pf_v2_enabled()
            && blob.progs[..dec_ix].iter().any(|p| {
                p.t >= 2048
                    && p.insts
                        .iter()
                        .any(|i| i.op == DevOp::FlashMlaPrefill as u16)
            });
        let need_mla_v2_fp8 = mla_pf_v2_enabled()
            && blob.progs[..dec_ix].iter().any(|p| {
                p.t >= 2048
                    && p.insts
                        .iter()
                        .any(|i| i.op == DevOp::FlashMlaPrefillFp8 as u16)
            });

        // --- code objects ---------------------------------------------------
        // Resolve the symbol immediately after each load: the HSA backend
        // creates a fresh executable per load, so a later load makes an earlier
        // handle unreachable even though its resolved kernel object stays valid.
        //
        // The packet's `build.json` says which arms the PREFILL object must have
        // been compiled with; `check_prefill_object` refuses a pairing that does
        // not, because the AMD `default:` writes nothing rather than trapping.
        // Read once, here, so a broken manifest fails before any object is
        // loaded rather than between two of them.
        let requires = build_requires(blob_path)?;
        let mut modules = Vec::new();
        let mut k_xaudit = None;
        let mut k_state_clear = None;
        let state_clear_device = crate::config::RuntimeConfig::get().amd.state_clear_device;
        // Task-13 per-rung co-load: an optional SECOND decode object for the low
        // rungs (PLOW_HSACO_LOWRUNG=<dir>), so rung 1-2 traffic runs the tight
        // single-slot codegen while wide rungs keep the batched object. Same
        // resolution, same pairing checks, different directory.
        let lowrung_dir = crate::config::RuntimeConfig::get()
            .amd
            .hsaco_lowrung
            .clone()
            .filter(|d| !d.is_empty());
        let mut load_one_in =
            |phase: Phase, sched: Sched, dir: &Path, gemv_need: Option<u32>| -> Result<HsaKernel> {
                let name = object_name(phase, variant, prefill_arm, sched);
                let path = dir.join(&name);
                // WHICH OBJECT, BY NAME, AT INFO — and this line is not cosmetic.
                //
                // `variant` and `prefill_arm` are DETECTED from the packet's opcodes
                // ([`Variant::detect`]), so which object a run opens is a DERIVED fact, not a build
                // choice, and nothing printed it. `Variant::detect` matches `GemvFp8` and the three
                // fp8-KV flash ops; it does NOT match the block-scaled `*Fp8Blk` family, so a
                // GLM-5.2 packet — every one of whose fp8 kernels is block-scaled — detects as
                // `Bf16` and runs on `interp_decode_gq.elf`. That is correct (the `*Fp8Blk` cases in
                // interp.hip are outside `#if PLOW_FP8`, deliberately), but it is the opposite of
                // what the object names suggest, and a whole campaign of decode-kernel arms was
                // built into `interp_decode_fp8_gq.elf` and measured against a run that never
                // opened it. Its ablation — delete the kernel entirely — read as "the packet costs
                // the same", which was taken as evidence for a protocol floor that does not exist.
                // Rebuilt into the object this line names, the same ablation moves the token by
                // 11.8% (perf-data/plow-gfx942/glm52-packet-protocol-xcd.md).
                tracing::info!(object = %name, ?phase, ?variant, ?prefill_arm, ?sched,
                           "code object");
                let image = std::fs::read(&path).map_err(|e| {
                    if phase == Phase::Prefill
                        && !matches!(prefill_arm, PrefillArm::None | PrefillArm::DSpark)
                    {
                        RuntimeError::Device(format!(
                            "code object {}: {e} — this packet's prefill programs contain {} \
                         opcodes, which requires {name} in {}, and it is not there. Build it \
                         (scripts/build_gfx950.sh with {}), or serve a packet that does not \
                         need it; falling back to an object without the arms is the AMD \
                         `default:`-does-not-trap bug this check exists to prevent.",
                            path.display(),
                            match prefill_arm {
                                PrefillArm::MlaMoe => "MLA+MoE prefill",
                                PrefillArm::Mla => "MLA prefill",
                                PrefillArm::K3 => "Kimi-K3 block",
                                PrefillArm::K3Moe => "Kimi-K3 block + grouped MoE prefill",
                                PrefillArm::K3MoeA4w4 => "Kimi-K3 block + grouped A4W4 MoE prefill",
                                PrefillArm::DSpark => unreachable!(),
                                PrefillArm::None => unreachable!(),
                            },
                            hsaco_dir.display(),
                            match prefill_arm {
                                PrefillArm::MlaMoe => "PLOW_MOE_PREFILL=1",
                                PrefillArm::Mla => "PLOW_MLA_PREFILL=1",
                                PrefillArm::K3 => "PLOW_K3=1 PLOW_MLA_PREFILL=1",
                                PrefillArm::K3Moe => {
                                    "PLOW_K3=1 PLOW_MLA_PREFILL=1 PLOW_MOE_PREFILL=1"
                                }
                                PrefillArm::K3MoeA4w4 => {
                                    "PLOW_K3=1 PLOW_MLA_PREFILL=1 PLOW_MOE_PREFILL=1 \
                                 PLOW_MOE_PF_A4W4=1 PLOW_MXFP4=1"
                                }
                                PrefillArm::DSpark => unreachable!(),
                                PrefillArm::None => unreachable!(),
                            },
                        ))
                    } else {
                        RuntimeError::Device(format!("code object {}: {e}", path.display()))
                    }
                })?;
                let syms = elf_symbol_names(&image);
                if let (Phase::Prefill, Some(req)) = (phase, requires.as_ref()) {
                    check_prefill_object(&syms, &path, req)?;
                }
                if let (Phase::Decode, Some(req)) = (phase, requires.as_ref()) {
                    check_decode_object(&syms, &path, req)?;
                }
                // The W_ofold fusion's arm lives in the FLASH object (the V2 MLA-prefill arm's
                // ofold epilogue), and it additionally needs the V2 routing itself: without
                // PLOW_MLA_PF_V2=1 the MLA segments run on the 8-wave prefill kernel, which
                // ignores packet i[6] and leaves unnormalized f32 partials for the fused GEMM
                // to read as bf16 — finite, fluent, wrong. Both must hold to serve this blob.
                if let (Phase::Flash, Some(req)) = (phase, requires.as_ref()) {
                    if req.iter().any(|r| r == "PLOW_GLM_OFOLD=1") {
                        if !crate::config::RuntimeConfig::get().amd.mla_pf_v2 {
                            return Err(RuntimeError::Device(
                                "this packet fuses MlaMergeFold+o_proj (W_ofold) and REQUIRES the \
                             V2 MLA-prefill routing: serve with PLOW_MLA_PF_V2=1, or emit \
                             without PLOW_GLM_OFOLD"
                                    .into(),
                            ));
                        }
                        if !syms.iter().any(|s| s.contains("plow_glm_ofold_arm")) {
                            return Err(RuntimeError::Device(format!(
                                "packet/object MISMATCH: this packet requires PLOW_GLM_OFOLD=1 \
                             but {} lacks the ofold-aware V2 arm (plow_glm_ofold_arm) — its \
                             flash would write unnormalized f32 partials that the fused \
                             o-GEMM reads as bf16 garbage. Rebuild the flash object from a \
                             tree that carries the arm.",
                                path.display()
                            )));
                        }
                    }
                }
                // The object's compiled row bucket vs the widest GEMV it will run.
                // Every phase, not just decode: `case PLOW_DOP_GEMV` is unconditional
                // in the prefill bucket too.
                let need = match phase {
                    Phase::Decode => gemv_need.unwrap_or(need_m_decode),
                    Phase::Prefill | Phase::Flash => need_m_prefill,
                };
                check_gemv_capacity(&syms, &path, need)?;
                // Whether this object carries the PLOW_K3 arms the packet dispatches. Refused here
                // rather than tolerated, because AMD's dispatch default is a silent NOP: the run
                // would otherwise complete on untouched buffers instead of failing.
                let need_k3 = match phase {
                    Phase::Decode => need_k3_decode,
                    Phase::Prefill | Phase::Flash => need_k3_prefill,
                };
                check_k3_arms(&syms, &path, need_k3)?;
                let need_dspark = match phase {
                    Phase::Decode => need_dspark_decode,
                    Phase::Prefill | Phase::Flash => need_dspark_prefill,
                };
                check_dspark_noncausal(&syms, &path, need_dspark)?;
                let need_spec_verify = match phase {
                    Phase::Decode => need_spec_verify_decode,
                    Phase::Prefill | Phase::Flash => need_spec_verify_prefill,
                };
                check_k3_spec_verify(&syms, &path, need_spec_verify)?;
                if phase == Phase::Decode {
                    check_moe_pf_a4w4(&syms, &path, need_a4w4_decode)?;
                    check_kda_conv_step_db(&syms, &path, need_kda_conv_step_db, legacy_kda_decode)?;
                }
                let (need_gm, need_gmpf) = match phase {
                    Phase::Decode => (need_gm_decode, need_gmpf_decode),
                    Phase::Prefill | Phase::Flash => (need_gm_prefill, need_gmpf_prefill),
                };
                check_moe_gemma_arms(&syms, &path, need_gm, need_gmpf)?;
                if phase == Phase::Flash && need_mla_v2 && !syms.contains(&MLA_PF_V2_SYM) {
                    return Err(RuntimeError::Device(format!(
                        "PLOW_MLA_PF_V2=1 routes FlashMlaPrefill segments to {}, but it was \
                     compiled without the V2 arm (no `{MLA_PF_V2_SYM}`). The dispatch default \
                     writes NOTHING, so those packets would silently skip. Rebuild the flash \
                     object (scripts/build_gfx942.sh adds -DPLOW_MLA_PF_V2_ARM=1) or unset \
                     PLOW_MLA_PF_V2.",
                        path.display()
                    )));
                }
                if phase == Phase::Flash && need_mla_v2_fp8 && !syms.contains(&MLA_PF_V2_FP8_SYM) {
                    return Err(RuntimeError::Device(format!(
                        "PLOW_MLA_PF_V2=1 routes FlashMlaPrefillFp8 segments to {}, but it was \
                     compiled without the fp8 V2 arm (no `{MLA_PF_V2_FP8_SYM}`). Rebuild the \
                     fp8-KV flash object or unset PLOW_MLA_PF_V2.",
                        path.display()
                    )));
                }
                // L2-PLACED BLOB vs OBJECT. This is the guard the PLOW_L2_PLACE_DISPATCH env var used
                // to stand in for, moved to where it can be VERIFIED. A placed program's `seg` is an
                // L2 domain, not a wave class, so an object built without the axis would run every
                // packet on the wrong domain -- plausible output, inverted locality, no error.
                let phase_l2_placed = match phase {
                    Phase::Decode => decode_l2_placed,
                    Phase::Prefill | Phase::Flash => prefill_l2_placed,
                };
                if phase_l2_placed && !syms.contains(&L2_DISPATCH_SYM) {
                    return Err(RuntimeError::Device(format!(
                    "{}: blob uses L2-domain packet placement (PLOW_L2_PLACE) but this object was \
                     built WITHOUT -DPLOW_L2_PLACE_DISPATCH — its `seg` would be read as a \
                     wave class and every packet would land on the wrong domain. Rebuild the \
                     objects (scripts/build_gfx942.sh passes it by default), or recompile the \
                     model with PLOW_L2_PLACE=0.",
                    path.display()
                )));
                }
                // Whether this object's KV ENCODING matches the packet's. Both directions — the axis
                // is a swap, so each object is missing an arm the other has.
                let (need_fp8, need_bf16) = match phase {
                    Phase::Decode => (need_fp8kv_decode, need_bf16kv_decode),
                    Phase::Prefill | Phase::Flash => (need_fp8kv_prefill, need_bf16kv_prefill),
                };
                check_kv_encoding(&syms, &path, need_fp8, need_bf16)?;
                let m = EngineDevice::module_load(&*be, &image).map_err(|e| {
                    RuntimeError::Device(format!(
                        "{name}: {e} — a BUNDLED object gives exactly this; was it \
                     run through clang-offload-bundler --unbundle?"
                    ))
                })?;
                let sym = symbol_name(phase, sched, &arch);
                let k = EngineDevice::get_function(&*be, &m, &sym)
                    .map_err(|e| RuntimeError::Device(format!("{name}: no symbol {sym}: {e}")))?;
                if phase == Phase::Decode && tp_audit_compact && k_xaudit.is_none() {
                    k_xaudit = Some(
                        EngineDevice::get_function(&*be, &m, "plow_xctr_audit").map_err(|e| {
                            RuntimeError::Device(format!(
                            "{name}: compact TP audit requested but plow_xctr_audit is absent: {e}"
                        ))
                        })?,
                    );
                }
                if phase == Phase::Decode && state_clear_device && k_state_clear.is_none() {
                    k_state_clear = Some(
                        EngineDevice::get_function(&*be, &m, "plow_state_clear").map_err(|e| {
                            RuntimeError::Device(format!(
                                "{name}: device recurrent-state clear requested but \
                                 plow_state_clear is absent: {e}. Rebuild the decode object"
                            ))
                        })?,
                    );
                }
                // STALE-OBJECT REFUSAL. An object's kernarg segment is its explicit
                // args, 8-aligned, plus the COv5 implicit block — a FIXED 256 B tail
                // that hipcc emits only when the kernel uses a hidden arg (the flash
                // object does not, and reports the bare struct size; prefill and
                // decode do, and report that + 256). Those are the only two legal
                // values for a kernel whose one argument is `PlowProgram`, and both
                // are DERIVED from `size_of::<DevProgram>()` below rather than
                // written as literals — the struct has grown twice already
                // (128 -> 136 with `seg_ofs`, 136 -> 144 with `l2_domains`), and a
                // literal here goes stale exactly when it is most needed.
                //
                // This matters because the launcher writes that implicit block at
                // OUR `size_of::<DevProgram>()`. An object built against a different
                // struct loads and resolves happily, then reads its own fields, or
                // its block/grid dimensions, from the wrong offsets and faults
                // somewhere unrelated. Refuse it by name here instead.
                const IMPLICIT: u32 = 256;
                let want = (std::mem::size_of::<DevProgram>() as u32 + 7) & !7;
                let got = k.kernarg_size();
                if got != want && got != want + IMPLICIT {
                    return Err(RuntimeError::Device(format!(
                    "{name}: kernarg segment is {got} B; this build's PlowProgram needs {want} \
                     (or {} with the COv5 implicit block) — the code object is STALE. Rebuild it \
                     with scripts/build_gfx950.sh. A mismatched object does not fail to load; it \
                     faults mid-run.",
                    want + IMPLICIT
                )));
                }
                modules.push(m);
                Ok(k)
            };

        let k_prefill = load_one_in(Phase::Prefill, sched_prefill, hsaco_dir, None)?;
        let k_decode = load_one_in(Phase::Decode, sched_decode, hsaco_dir, None)?;
        // Task-13: the low-rung decode tier ladder. Each tier's pairing checks
        // run with ITS need (the widest rung it will serve), not the blob-wide
        // max — an MM=4 object legitimately serves rungs 1-2 of a B=32 blob.
        // `PLOW_HSACO_LOWRUNG` is either `<dir>` (max = PLOW_LOWRUNG_MAX,
        // default 2) or `dir:max[,dir:max]...`; selection takes the narrowest
        // tier that fits, so the list is sorted ascending here.
        let mut decode_tiers: Vec<(u32, HsaKernel)> = Vec::new();
        if let Some(spec) = &lowrung_dir {
            let mut tiers: Vec<(String, u32)> = Vec::new();
            if spec.contains(':') {
                for ent in spec.split(',').filter(|s| !s.is_empty()) {
                    let (d, m) = ent.rsplit_once(':').ok_or_else(|| {
                        RuntimeError::Device(format!(
                            "PLOW_HSACO_LOWRUNG entry `{ent}`: expected dir:max"
                        ))
                    })?;
                    let m: u32 = m.parse().map_err(|_| {
                        RuntimeError::Device(format!(
                            "PLOW_HSACO_LOWRUNG entry `{ent}`: max `{m}` is not a u32"
                        ))
                    })?;
                    tiers.push((d.to_string(), m));
                }
            } else {
                let max = crate::config::RuntimeConfig::get().amd.lowrung_max;
                tiers.push((spec.clone(), max));
            }
            tiers.sort_by_key(|&(_, m)| m);
            for (d, max) in tiers {
                let k = load_one_in(Phase::Decode, sched_decode, Path::new(&d), Some(max))?;
                decode_tiers.push((max, k));
            }
        }
        // Flash follows the PREFILL scheduler — a flash segment is a prefill
        // segment. Optional: without it every segment runs class 8, which is
        // correct and merely slower.
        let k_flash = match load_one_in(Phase::Flash, sched_prefill, hsaco_dir, None) {
            Ok(k) => Some(k),
            Err(e) => {
                tracing::info!(%e, "no flash object — flash segments run on the 8-wave interpreter");
                None
            }
        };

        // --- tensors + weights ------------------------------------------------
        // Staging is one pinned slab, filled and pushed in `STAGE` chunks. The
        // source is an mmap of the checkpoint, and `upload` would pin it per
        // call — asking the kernel to lock tens of GiB of page-cache mappings.
        // Copying through a fixed pinned buffer keeps the locked set at 64 MiB.
        const STAGE: usize = 64 << 20;
        // `Arc`, because the prefetch pool madvises these mappings from other
        // threads and must be joined before they can be unmapped. Under TP the
        // caller passes ONE checkpoint for the whole group — see `shared_ckpt`.
        let ckpt = match (shared_ckpt, checkpoint) {
            (Some(c), _) => Some(c),
            (None, Some(dir)) => Some(Arc::new(crate::asset::checkpoint::Checkpoint::open(dir)?)),
            (None, None) => None,
        };
        // The fp8 weight TWINS live in their own checkpoint, not the bf16 one:
        // they are a separate quantisation artifact, and the packet names them
        // with an `fp8/` prefix that is stripped before lookup. Without this an
        // fp8 packet fails at the first weight with "MISSING WEIGHT", which
        // reads as a broken packet rather than a missing directory.
        let fp8_ckpt = match crate::config::RuntimeConfig::get().amd.fp8_dir.as_deref() {
            Some(d) => Some(crate::asset::checkpoint::Checkpoint::open(Path::new(d))?),
            None => None,
        };
        // `--amd-upload-slots 1` / `PLOW_UPLOAD_SLOTS=1` is the pre-pipeline
        // shape exactly: one slab, one copy, waited on before the next memcpy
        // starts. Kept so the pipelining can be A/B'd on one binary rather
        // than argued about.
        let slots = (crate::config::RuntimeConfig::get().amd.upload_slots as usize).max(1);
        let mut ring = be.upload_ring(slots, STAGE)?;
        // v7 blobs carry the RoPE tables as RECIPES, not bytes. Materialising
        // them is not optional: a reader that skips this leaves cos=sin=0 and
        // serves fluent-looking garbage with no error anywhere.
        let gen_by_tensor: std::collections::HashMap<u32, &packet::rope::GenTensor> =
            blob.gen.iter().map(|g| (g.tensor, g)).collect();

        // Must precede the tensor loop: it decides whether each full-layer KV
        // tensor gets an allocation or a view onto the pool's VA reservation.
        let vmm = Self::vmm_bringup(&be, &blob, checkpoint);

        let prof = LoadProf::default();
        let do_prefault = profile_faults();
        let depth = prefetch_depth();
        let prefetch = ckpt.as_ref().and_then(|c| {
            crate::asset::checkpoint::Prefetcher::start(
                Arc::clone(c),
                prefetch_threads(),
                depth,
                None,
            )
        });
        // The pool runs `depth` WEIGHT tensors ahead of the copy, over the same
        // list in the same order, so `pf` only ever moves forward and each
        // tensor is queued exactly once. Skipping the non-weights keeps the
        // depth denominated in reads rather than in table entries — most of this
        // blob's tensors are scratch that touches no checkpoint at all.
        let mut pf = 0usize;
        let prefetch_ahead = |cur: &mut usize, budget: usize| {
            let (Some(pool), Some(c)) = (prefetch.as_ref(), ckpt.as_ref()) else {
                return;
            };
            let mut n = 0;
            while *cur < blob.tensors.len() && n < budget {
                let td = &blob.tensors[*cur];
                *cur += 1;
                // `fp8/` weights live in the twin checkpoint, which is not the
                // one the pool holds, and they are a rounding error next to the
                // experts — so they are simply not prefetched.
                if !packet::names::is_checkpoint_weight(&td.name) || td.name.starts_with("fp8/") {
                    continue;
                }
                if let Some(s) = weight_span(c, &td.name, td.bytes, rank, n_gpu) {
                    pool.push(s);
                }
                n += 1;
            }
        };
        prefetch_ahead(&mut pf, depth);

        // ---- one allocation for every tensor that would otherwise get its own
        //
        // Both passes below must agree, tensor for tensor, on which tensors are
        // carved and how much each consumes: the sizing pass decides how big the
        // slab is and the upload loop walks the cursor through it, so a filter
        // that disagreed in either direction would either overrun the end or
        // silently overlap two tensors. They are kept in step by sharing these
        // two closures rather than by two hand-copied conditions.
        //
        // The two arms that take a view instead of an allocation:
        //   * TP peer slots — storage owned by the `TpRank` peer region, which
        //     `XReduce` reads over XGMI. Carving these out of local VRAM would
        //     have every rank reduce slots its peers never wrote.
        //   * full-layer KV under VMM — the pool's VA reservation, mapped lazily
        //     at the per-sequence frontier.
        let is_peer_slot = |name: &str| {
            matches!(
                (tp.is_some(), name),
                (true, "act.og_tp") | (true, "act.dg_tp") | (true, "act.ug_tp")
            )
        };
        let is_vmm = |name: &str| {
            vmm.as_ref()
                .and_then(|v| {
                    let (l, t) = kv_tensor_name(name)?;
                    v.tensor_va(l, t)
                })
                .is_some()
        };
        // `.max(1)`, exactly as the per-tensor arm does — a zero-byte tensor
        // still needs a distinct address, and a zero-length carve would hand the
        // next tensor the same one.
        let slab_need = |bytes: u64| bytes.max(1);
        let slab_bytes: u64 = blob
            .tensors
            .iter()
            .filter(|td| !is_peer_slot(&td.name) && !is_vmm(&td.name))
            .map(|td| slab_pad(slab_need(td.bytes)))
            .sum();
        let t_slab = Instant::now();
        let weight_slab = if slab_bytes == 0 || !crate::asset::checkpoint::weight_slab_enabled() {
            WeightSlab::PerTensor
        } else {
            // Opt-in (`PLOW_WEIGHT_VMM=1`): the lazy-commit slab is unmeasured
            // on AMD hardware — see `weight_vmm_amd_enabled` for why the
            // default differs from CUDA's. Every failure falls through to the
            // flat allocation, which falls through to per-tensor.
            let vmm_slab = if crate::asset::checkpoint::weight_vmm_amd_enabled() {
                match crate::memory::vmm::VmmSlab::new(
                    Arc::clone(&be) as Arc<dyn VmmOps>,
                    slab_bytes,
                    crate::memory::vmm::WEIGHT_SLAB_CHUNK,
                ) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        tracing::warn!(
                            bytes = slab_bytes,
                            error = %e,
                            "vmm weight slab refused — falling back to flat allocation"
                        );
                        None
                    }
                }
            } else {
                None
            };
            match vmm_slab {
                Some(s) => WeightSlab::Vmm(s),
                None => match EngineDevice::alloc(&*be, slab_bytes) {
                    Ok(m) => WeightSlab::Flat(m),
                    // Not fatal: the per-tensor arm below still works and is only
                    // slower and hungrier. Better a fat load than a refused one.
                    Err(e) => {
                        tracing::warn!(
                            bytes = slab_bytes,
                            error = %e,
                            "single weight allocation refused — falling back to per-tensor alloc"
                        );
                        WeightSlab::PerTensor
                    }
                },
            }
        };
        LoadProf::add(&prof.alloc_ns, t_slab);
        let mut slab_off: u64 = 0;

        let t_tensors = Instant::now();
        let mut devp = Vec::with_capacity(blob.tensors.len());
        let mut names = Vec::with_capacity(blob.tensors.len());
        let (mut wbytes, mut nweights) = (0u64, 0usize);
        // Tensors that took a view into storage someone else owns (peer region
        // or VMM reservation) rather than a carve out of the slab — the exact
        // set the sizing pass filtered out.
        let mut n_view = 0usize;
        for (i, td) in blob.tensors.iter().enumerate() {
            // §7a: the two row-parallel partials live in the PEER region, not in
            // ordinary VRAM. `o_proj`/`down` write straight into them and the
            // peers' `XReduce` reads them over XGMI, so an ordinary local
            // allocation here would have every rank reduce three slots its peers
            // never wrote — a wrong token, with no fault and no message.
            //
            // A non-owning view: the storage belongs to the `TpRank`'s peer
            // allocation, which outlives the engine. `devgen` only declares
            // these two tensors when tp > 1.
            let peer_slot = match (tp, td.name.as_str()) {
                (Some(t), "act.og_tp") => Some(t.scratch_base),
                (Some(t), "act.dg_tp") => Some(t.scratch_base + t.slot_b),
                // Slot 2, the GATHER slot: a column-parallel partial the reduce out of
                // slot 0 folds in (`PARTIAL_SLOTS`). Only K3's LatentMoE declares it.
                (Some(t), "act.ug_tp") => Some(t.scratch_base + 2 * t.slot_b),
                _ => None,
            };
            if let Some(base) = peer_slot {
                tracing::debug!(
                    name = %td.name, base = format_args!("{base:#x}"), bytes = td.bytes,
                    "bound into the peer region"
                );
                devp.push(DeviceMem::view(base, td.bytes.max(1)));
                names.push(td.name.clone());
                n_view += 1;
                continue;
            }
            // Full-layer KV under VMM: the base is the pool's VA reservation,
            // a non-owning view (the pool owns unmap/release). No allocation
            // and no memset — the VA is mapped lazily at each sequence's
            // frontier, and KV is always written before it is read.
            let vmm_va = vmm.as_ref().and_then(|v| {
                let (l, t) = kv_tensor_name(&td.name)?;
                v.tensor_va(l, t)
            });
            let mem = match (vmm_va, &weight_slab) {
                (Some(va), _) => {
                    n_view += 1;
                    DeviceMem::view(va, td.bytes.max(1))
                }
                // Carve from the VMM reservation, in blob order, waiting for
                // the mapper to commit through this tensor's padded end BEFORE
                // handing the view out — every write path (upload ring, shard
                // gather, memset tail) starts after the carve, so this one
                // wait point covers them all. The mapper outruns the upload,
                // so the wait is ~0 after the first chunk.
                (None, WeightSlab::Vmm(slab)) => {
                    let m = DeviceMem::view(slab.base() + slab_off, slab_need(td.bytes));
                    slab_off += slab_pad(slab_need(td.bytes));
                    slab.wait_mapped(slab_off)?;
                    m
                }
                // Carve from the one allocation, in blob order. The sizing pass
                // walked this same list with the same filter and the same
                // `slab_need`, so the cursor cannot run past the end.
                (None, WeightSlab::Flat(slab)) => {
                    let m = DeviceMem::view(slab.base + slab_off, slab_need(td.bytes));
                    slab_off += slab_pad(slab_need(td.bytes));
                    m
                }
                (None, WeightSlab::PerTensor) => {
                    let t = Instant::now();
                    let m = EngineDevice::alloc(&*be, td.bytes.max(1))?;
                    LoadProf::add(&prof.alloc_ns, t);
                    m
                }
            };
            let prof = &prof;
            // `scrub` — rewrite 0x80 (OCP e4m3 `-0`) to 0x00 inside the slab copy. Set for
            // every F8_E4M3 checkpoint payload (dense-FFN projections, fp8-twin weights…),
            // value-identical everywhere; the point is the CDNA3 maskless staging decode
            // (`mpf_fp8x4_to_bf16_h`, op_moe.h), whose contract is "no 0x80 can reach me".
            // The routed-expert packing loop applies the same rule on its own path.
            let push = |ring: &mut crate::device::hsa::HsaUploadRing,
                        src: &[u8],
                        scrub: bool|
             -> Result<()> {
                for (o, chunk) in src.chunks(STAGE).enumerate() {
                    let t = Instant::now();
                    let at = mem.base + (o * STAGE) as u64;
                    if scrub {
                        ring.push_scrub_fp8_neg0(at, chunk)?;
                    } else {
                        ring.push(at, chunk)?;
                    }
                    // Staging memcpy and DMA wait are no longer separable: the
                    // point of the ring is that they overlap. One `stage_ns`
                    // counter is the honest shape; `dma_ns` stays zero.
                    LoadProf::add(&prof.memcpy_ns, t);
                    prof.chunks.set(prof.chunks.get() + 1);
                }
                Ok(())
            };

            // The two MoE pointer tables are named like weights and are not
            // weights: `expert_weight_table`/`expert_scale_table` hold DEVICE
            // ADDRESSES the host computes after packing the experts, and no
            // checkpoint contains them. Left to the branch below they resolve to
            // `MISSING WEIGHT` and GLM cannot bind at all. They fall through to
            // the zeroing tail — a zero entry is the kernel's "not my expert" —
            // and `bind_packed_experts` fills them once the packing is done.
            //
            // Classified by EXCLUSION of the compiler's own namespaces (`packet::names`),
            // not by an allowlist of weight prefixes. The allowlist here was
            // `model.` | `fp8/` | `lm_head` — three of the four arms the five sites in the
            // tree each spelled differently — and it holds for every model shipped so far
            // only because their weights happen to be under `model.`. Kimi-K3's are not:
            // all 497 052 language-tower tensors are `language_model.model.…` and NONE
            // starts with `model.`, so an allowlist binds nothing, uploads nothing, and
            // decodes from zeroed weights without a word.
            let is_weight = packet::names::is_checkpoint_weight(&td.name);
            if is_weight {
                prefetch_ahead(&mut pf, 1);
                // `fp8/` routes to the twin checkpoint with the prefix
                // stripped; everything else to the base one.
                let is_fp8 = td.name.starts_with("fp8/");
                let src_ckpt = if is_fp8 {
                    fp8_ckpt.as_ref()
                } else {
                    ckpt.as_deref()
                };
                // KIMI-K3's ATTNRES SCORE WEIGHT IS DERIVED, NOT STORED. Resolved
                // before the ordinary lookup, which would otherwise report MISSING
                // WEIGHT for 186 tensors on a 93-layer model.
                //
                // NOT a `continue`: the loop tail pushes `devp`/`names` for EVERY
                // tensor, and skipping it would shift every later tensor's index
                // against the table the packet was compiled with.
                let folded = match src_ckpt {
                    Some(c) => fold_res_score(c, &td.name).transpose()?,
                    None => None,
                };
                if let Some(folded) = folded {
                    if folded.len() as u64 != td.bytes {
                        return Err(RuntimeError::Device(format!(
                            "{}: folded score weight is {} B, blob declares {}",
                            td.name,
                            folded.len(),
                            td.bytes
                        )));
                    }
                    push(&mut ring, &folded, false)?;
                    wbytes += td.bytes;
                } else if let Some(c) = src_ckpt {
                    // BOTH spellings, because the twin checkpoints disagree
                    // with each other. `/home/lava/models/g31b-fp8w` KEEPS the
                    // `fp8/` prefix in its tensor names; the C reference strips
                    // it. Trying the packet's name first and the stripped form
                    // second costs one hash lookup and works with either
                    // convention, which is better than encoding a guess about
                    // which artifact someone hands us.
                    let stripped = td.name.strip_prefix("fp8/").unwrap_or(&td.name);
                    let (resolved, (src, shape)) = c
                        .tensor_ex(&td.name)
                        .map(|v| (td.name.as_str(), v))
                        .or_else(|| c.tensor_ex(stripped).map(|v| (stripped, v)))
                        .ok_or_else(|| {
                            RuntimeError::Device(format!(
                                "MISSING WEIGHT: {} (tried that name and the \
                                 `fp8/`-stripped form{})",
                                td.name,
                                if is_fp8 { " in PLOW_FP8_DIR" } else { "" }
                            ))
                        })?;
                    // At n_gpu == 1 this borrows the whole mmap range and the
                    // size check inside is the old `SIZE MISMATCH`. Above 1 it
                    // is the rank's shard — classified by the CHECKPOINT name
                    // (`stripped`), so an fp8 twin shards exactly like its bf16
                    // counterpart instead of falling through as replicated.
                    if do_prefault {
                        if let Some(s) = touched(c, stripped, td.bytes, rank, n_gpu) {
                            prefault(s, &prof);
                        }
                    }
                    let t = Instant::now();
                    let slice = crate::asset::shard::slice_for(
                        stripped, src, shape, td.bytes, rank, n_gpu,
                    )?;
                    LoadProf::add(&prof.gather_ns, t);
                    push(&mut ring, &slice, c.is_fp8_e4m3(resolved))?;
                    wbytes += td.bytes;
                    nweights += 1;
                } else if td.name.starts_with("fp8/") {
                    return Err(RuntimeError::Device(format!(
                        "packet declares fp8 weights ({}) but PLOW_FP8_DIR is not set",
                        td.name
                    )));
                }
            } else if let Some(r) = &td.init {
                push(&mut ring, &blob.init[r.clone()], false)?;
            } else if let Some(g) = gen_by_tensor.get(&(i as u32)) {
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
                push(&mut ring, &data, false)?;
            } else if vmm_va.is_none() && !kv_skips_zeroing(&td.name) {
                // A VMM window is (mostly) UNMAPPED VA — a memset would fault,
                // not merely waste time. The `kv.` clause below is the older
                // and independent reason to skip.
                //
                // Attention reads only [0, kvlen), every row of which is
                // written before it is read, so the KV cache needs no zeroing —
                // 11.5 GiB of memset skipped on this model. Other scratch stays
                // zeroed: cheap, and conservative where the argument is less
                // obviously airtight.
                let t = Instant::now();
                EngineDevice::memset_d8(&*be, mem.base, 0, td.bytes as usize)?;
                LoadProf::add(&prof.memset_ns, t);
            }
            devp.push(mem);
            names.push(td.name.clone());
        }
        // EVERY named tensor is bound only once its copy has retired. The ring
        // leaves copies in flight by design, so this is the line that makes
        // "uploaded" mean uploaded.
        ring.drain()?;
        // Slab tail join: the carve-site waits covered every tensor that was
        // carved, but writes AFTER this loop (`bind_packed_experts` fills the
        // expert pointer tables in place) need the WHOLE span committed —
        // VMM has no demand paging. ~0 in practice: the mapper finished while
        // the upload ran.
        if let WeightSlab::Vmm(slab) = &weight_slab {
            let t = Instant::now();
            slab.wait_mapped(slab_bytes)?;
            LoadProf::add(&prof.alloc_ns, t);
        }
        // The MoE half of the bind, and it has to be here: it needs the tensor
        // table (to find each layer's two pointer slots) and the staging ring
        // (which must outlive it — the C reference records that gathering a
        // row-parallel slice into a MALLOC'd buffer faults the SDMA engine,
        // because the copy does not pin its source).
        prof.report("named tensors", t_tensors.elapsed(), wbytes);
        if !matches!(weight_slab, WeightSlab::PerTensor) {
            // The sizing pass and the carve walked the same list with the same
            // filter, so the cursor must land exactly on the total: short wastes
            // the tail, long means two tensors were aliased onto the same bytes
            // and the weights are quietly wrong. The loop above cannot exit
            // early — an error returns from the function — so unlike `exec::gpu`
            // this needs no "did it finish" guard.
            debug_assert_eq!(
                slab_off, slab_bytes,
                "weight slab carve did not consume exactly the sized span"
            );
            // `carved` is what the pool would have been asked for per tensor;
            // the rounding it no longer pays is invisible from here (ROCr never
            // reports it), so the honest thing to log is the request, and the
            // saving is read off `MEMORY_AVAIL` by whoever is measuring.
            tracing::info!(
                slab_mib = slab_bytes / (1 << 20),
                carved = devp.len() - n_view,
                views = n_view,
                "weights carved from one allocation"
            );
        }
        let mut expert_bufs = Vec::new();
        if let Some(c) = ckpt.as_ref() {
            let eprof = LoadProf::default();
            let t_exp = Instant::now();
            let (bufs, bytes) = bind_packed_experts(
                &be,
                &blob,
                c,
                &devp,
                &names,
                &mut ring,
                rank,
                n_gpu,
                &eprof,
                do_prefault,
                // Workers populate their own spans; `prefetch_threads() == 0`
                // (the pool disabled) also turns that readahead off.
                prefetch.is_some(),
            )?;
            eprof.report("packed experts", t_exp.elapsed(), bytes);
            expert_bufs = bufs;
            wbytes += bytes;
        }
        // Dense-FFN prefill tables. Must run AFTER the named-weight upload above
        // (it is those uploads that give the projections their device addresses)
        // and is a no-op on a decode-only blob, which declares no such table.
        let n_dense_tab = bind_dense_ffn_tables(&be, &blob, &devp, &names)?;
        if n_dense_tab > 0 {
            tracing::info!(
                tables = n_dense_tab,
                "dense-FFN prefill pointer tables bound (grouped-arm 1-expert path)"
            );
        }
        drop(ring);
        if ckpt.is_some() {
            let s = t_rank.elapsed().as_secs_f64();
            let gib = wbytes as f64 / (1u64 << 30) as f64;
            tracing::info!(
                rank,
                gib = format!("{gib:.2}").as_str(),
                tensors = nweights,
                secs = format!("{s:.1}").as_str(),
                gib_s = format!("{:.2}", if s > 0.0 { gib / s } else { 0.0 }).as_str(),
                "checkpoint weights uploaded"
            );
        } else {
            tracing::warn!(
                "NO CHECKPOINT — weights are uninitialised; timings are real, tokens are not"
            );
        }
        let table: Vec<u8> = devp.iter().flat_map(|m| m.base.to_le_bytes()).collect();
        let d_tens = EngineDevice::alloc(&*be, table.len().max(1) as u64)?;
        EngineDevice::upload(&*be, &d_tens, 0, &table)?;

        // OPTIONAL, because a BLOCK asset is not a model. A block takes
        // `act.x` in and gives `act.x` out — it has no embedding, no lm_head,
        // no argmax, and therefore none of `in.ids`/`in.pos`/`in.kvlen`/
        // `act.logits`. Requiring them refused the single most useful A/B
        // vehicle in the tree (one layer, one precision difference) for want of
        // tensors that layer has no reason to own.
        let find = |n: &str| names.iter().position(|x| x == n);
        let t_ids = find("in.ids");
        let t_pos = find("in.pos");
        let t_kvlen = find("in.kvlen");
        let t_active = find("in.parked");
        let t_logits = find("act.logits");
        // The context bound is carried by in.pos, not by any prefill bucket.
        let max_ctx = t_pos.map_or(0, |t| (blob.tensors[t].bytes / 4) as usize);

        // --- per-program tables ---------------------------------------------
        let ctr_banks: u64 = if ctr_dbuf() { 2 } else { 1 };
        let mut progs = Vec::with_capacity(blob.progs.len());
        for p in &blob.progs {
            let seg_class = derive_segments(p)?;
            let up = |bytes: &[u8]| -> Result<DeviceMem> {
                let m = EngineDevice::alloc(&*be, bytes.len().max(1) as u64)?;
                if !bytes.is_empty() {
                    EngineDevice::upload(&*be, &m, 0, bytes)?;
                }
                Ok(m)
            };
            let d_inst = up(as_bytes(&p.insts))?;
            let d_stream = up(as_bytes(&p.stream))?;
            let d_sofs = up(as_bytes(&p.stream_ofs))?;
            let d_slen = up(as_bytes(&p.stream_len))?;
            let d_waits = up(as_bytes(&p.waits))?;
            let d_succs = up(as_bytes(&p.succs))?;
            // Counters are allocated, never uploaded — they are re-armed per
            // dispatch group.
            //
            // TWO BANKS when `ctr_dbuf` is on, and then they ARE zeroed here:
            // the double-buffered `run` clears the STALE bank after enqueueing,
            // so the bank a dispatch actually reads was cleared one dispatch
            // ago and the very first dispatch has no such predecessor. Without
            // this memset it would run out of whatever the allocator handed
            // back. (The single-bank path still clears synchronously in
            // `rearm`, so its memset is redundant but harmless and costs one
            // memset at load.)
            let ctr_span = (p.n_counter as usize * CTR_STRIDE_U32 * 4).max(1) as u64;
            let d_ctr = EngineDevice::alloc(&*be, ctr_span * ctr_banks)?;
            EngineDevice::memset_d8(&*be, d_ctr.base, 0, (ctr_span * ctr_banks) as usize)?;
            // The VRAM price of the second bank, per program, so it is a number
            // in the log rather than an argument in a comment.
            tracing::debug!(
                prog = progs.len(),
                n_counter = p.n_counter,
                bank_kib = ctr_span / 1024,
                banks = ctr_banks,
                "counter banks"
            );
            // Per-(CU, segment) windows. DERIVED from the stream we just
            // uploaded, so it cannot disagree with it — the standing failure of
            // a precomputed window table (`PLOW_SEG_OFF` rewrote `stream[].seg`
            // and left `gq_seg_ofs` describing a stream that no longer existed,
            // which ran one segment and reported it as the whole prefill).
            // `n_seg` comes from `derive_segments`, which is `max(seg)+1` — the
            // same count `run_segmented` launches.
            let seg_ofs = static_seg_ofs(
                &p.stream,
                &p.stream_ofs,
                &p.stream_len,
                seg_class.len() as u32,
            )
            .map_err(RuntimeError::Device)?;
            let d_seg_ofs = up(as_bytes(&seg_ofs))?;

            let gq = if p.gq_stream.is_empty() {
                None
            } else {
                let n_seg = p.gq_seg_ofs.len().saturating_sub(1) as u32;
                // DOUBLE-BUFFERED TOO. The GQ cursor is re-armed in the same
                // breath as the counters and is just as much a dispatch's live
                // state, so banking one without the other would have dispatch
                // N+1 resume from dispatch N's cursor.
                let cur_span = (n_seg.max(1) as usize * CTR_STRIDE_U32 * 4) as u64;
                let d_cursor = EngineDevice::alloc(&*be, cur_span * ctr_banks)?;
                EngineDevice::memset_d8(&*be, d_cursor.base, 0, (cur_span * ctr_banks) as usize)?;
                Some(AmdGq {
                    d_stream: up(as_bytes(&p.gq_stream))?,
                    d_seg_ofs: up(as_bytes(&p.gq_seg_ofs))?,
                    d_cursor,
                    n_seg,
                    cur_span,
                })
            };
            progs.push(AmdProg {
                t: p.t,
                n_inst: p.insts.len() as u32,
                trace_records: p.stream.len(),
                n_counter: p.n_counter,
                d_inst,
                d_stream,
                d_sofs,
                d_slen,
                d_waits,
                d_succs,
                d_ctr,
                d_seg_ofs,
                seg_class,
                gq,
                l2_domains: p.l2_domains,
                // DERIVED, not carried in the blob. The emitter appends the two-level
                // maintenance scratch to the tail of the counter region, three u32 per
                // (packet, domain), so its base is implied by fields the header already has.
                // That keeps `PlowProgHeader` at 24 bytes and the blob format unchanged.
                //
                // Zero unless the program is L2-placed: without per-domain windows there is no
                // `nper`, the emitter allocates no scratch, and the interpreter reads 0 as
                // "no hierarchy".
                hier_base: if p.l2_domains != 0 {
                    (p.n_counter).saturating_sub(3 * p.insts.len() as u32 * p.l2_domains)
                } else {
                    0
                },
                ctr_span,
                bank: CounterBankState::new(),
            });
        }
        let decode = progs.len() - 1;
        // THE DECODE BATCH LADDER: programs `[dec_lo, decode]` are decode rungs at ascending
        // widths, `[0, dec_lo)` the prefill bucket ladder. Same value the per-phase object
        // requirements were split at above — one rule, computed once.
        let dec_lo = dec_ix;
        // Packet trace (`PLOW_TRACE_RAW=<path>`). Zeroed once at allocation so
        // an entry the run never reaches reads as a zero record rather than as
        // whatever the allocator handed back; every executed slot is rewritten
        // each step, so the buffer always holds the LAST step's timeline.
        let (d_trace, trace_bytes) = if crate::config::RuntimeConfig::get().amd.trace_raw.is_some()
        {
            // Sized for the WIDEST program, not the decode one. The pointer used to be handed
            // only to `decode` (see the kernarg builder), so a prefill dispatch got a null trace
            // and recorded nothing — prefill was untraceable BY CONSTRUCTION, which is why the
            // first prefill trace ever taken came back with 0 packets. K3's prefill buckets carry
            // 2942 stream entries against decode's 2459, so sizing by `decode` alone would have
            // overflowed the buffer the moment the pointer was handed over.
            let bytes =
                blob.progs.iter().map(|g| g.stream.len()).max().unwrap_or(0) * TRACE_REC_BYTES;
            let m = EngineDevice::alloc(&*be, bytes.max(1) as u64)?;
            EngineDevice::upload(&*be, &m, 0, &vec![0u8; bytes])?;
            (Some(m), bytes)
        } else {
            (None, 0)
        };
        // `in.kvlen` is [batch] i32. Cross-check against the decode program's
        // compiled `t`, which is `PLOW_DECODE_BATCH`: they must agree, and a
        // mismatch means the blob was assembled from parts.
        let batch = t_kvlen.map_or(1, |t| (blob.tensors[t].bytes / 4).max(1) as usize);
        if t_kvlen.is_some() && batch != progs[decode].t as usize {
            return Err(RuntimeError::Device(format!(
                "in.kvlen is {batch} rows but the decode program is compiled for t={} \
                 — blob/tensor mismatch",
                progs[decode].t
            )));
        }

        // KV slot geometry, for the per-slot prefill rebase. Only meaningful at
        // batch > 1; at batch 1 the list is empty and every rebase is a no-op,
        // so a single-sequence engine is byte-identical to before.
        // THE DECODE GEMV IS A COMPILE-TIME ROW BUCKET, CAPPED AT 16.
        // `runtime/amd/op_gemm.h`: `PLOW_GEMV_MAXM 16`, and `gemv_rows<MM>`
        // carries `float acc[MM]` and loops `m < MM` — it has NO outer loop
        // over M > MM. `scripts/build_gfx950.sh` clamps `PLOW_GEMV_MM` to 16
        // to satisfy the static assert, so a blob emitted at
        // PLOW_DECODE_BATCH=32 loads against an MM=16 object and every
        // sequence from 16 up gets a ZERO logit row and samples token 0 —
        // §4's bug shape, with no fault anywhere. `plowc` does not refuse it
        // (it wrote `gv_mm_max: 32` into build.json), so refuse it here, where
        // the packet finally meets the code objects.
        //
        // THIS IS THE CEILING, NOT THE PAIRING. It compares the blob against a
        // constant no object can exceed; it says nothing about the bucket the
        // object in `hsaco_dir` was actually compiled at, so a B=8 blob on an
        // MM=1 object passes it (8 <= 16) and produces one correct row out of
        // eight. `check_gemv_capacity` is the half that compares blob against
        // OBJECT, and it runs above in `load_one`. Both are needed: this one
        // still catches a blob no build can serve, and it does so before any
        // object is opened.
        // NO LONGER A HARD CEILING: a WALKING object (`plow_gemv_walk_1`,
        // PLOW_GEMV_WALK=1) serves any M in ceil(M/MM) row-block passes, so a
        // batch above the bucket cap is servable and `check_gemv_capacity` —
        // which sees the actual object — is the gate that refuses a NON-walking
        // object at need > cap, with the correct message. What remains here is
        // a sanity bound: XArgmaxFin's two-line fold caps sequences at 32
        // (PLOW_XAMAX_MAX_BATCH), and nothing above it has ever been emitted.
        if batch > 32 {
            return Err(RuntimeError::Device(format!(
                "blob is compiled PLOW_DECODE_BATCH={batch}, past the XArgmaxFin two-line \
                 fold's 32-sequence ceiling (PLOW_XAMAX_MAX_BATCH, runtime/amd/op_collective.h). \
                 Re-emit at PLOW_DECODE_BATCH <= 32."
            )));
        }

        // The CARRIED recurrent state, per slot. Unlike `kv_slot_stride` this is built at
        // EVERY batch (a batch-1 slot has one region spanning the whole tensor), because the
        // prefix snapshot below is not a batching feature.
        //
        // `blkres` is excluded for the same reason `begin_slot` excludes it: it is sized at the
        // widest PREFILL bucket rather than at `batch`, so `bytes / batch` is not its stride —
        // and it carries nothing across passes, so a snapshot has nothing to capture.
        let carried_slot: Vec<(usize, u64)> = blob
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| is_carried_state(&t.name) && !t.name.contains("blkres"))
            .map(|(i, t)| (i, t.bytes / batch as u64))
            .collect();

        let mut kda_conv_bank_pairs = Vec::new();
        for (alt_i, alt) in blob.tensors.iter().enumerate() {
            if !alt.name.contains(".conv_state_alt.") {
                continue;
            }
            if batch != 1 {
                return Err(RuntimeError::Device(format!(
                    "{} is a double-buffered KDA conv window, but the packet batch is {batch}; \
                     KdaConvStateStepG is B1-only",
                    alt.name
                )));
            }
            let base_name = alt.name.replacen(".conv_state_alt.", ".conv_state.", 1);
            let base_i = blob
                .tensors
                .iter()
                .position(|t| t.name == base_name)
                .ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "{} has no matching legacy KDA conv window {base_name}",
                        alt.name
                    ))
                })?;
            if devp[alt_i].len != devp[base_i].len {
                return Err(RuntimeError::Device(format!(
                    "KDA conv banks disagree: {} is {} bytes, {base_name} is {} bytes",
                    alt.name, devp[alt_i].len, devp[base_i].len
                )));
            }
            kda_conv_bank_pairs.push((devp[alt_i].base, devp[base_i].base, devp[alt_i].len));
        }
        if need_kda_conv_step_db.is_some() && kda_conv_bank_pairs.is_empty() {
            return Err(RuntimeError::Device(
                "KdaConvStateStepG packet has no alternate KDA conv-window tensors".into(),
            ));
        }

        let clear_ranges = if state_clear_device {
            state_clear_ranges(&devp, &carried_slot)?
        } else {
            Vec::new()
        };
        let n_state_clear = u32::try_from(clear_ranges.len())
            .map_err(|_| RuntimeError::Device("too many recurrent-state clear ranges".into()))?;
        let d_state_clear = if clear_ranges.is_empty() {
            None
        } else {
            let bytes = std::mem::size_of_val(clear_ranges.as_slice()) as u64;
            let d = EngineDevice::alloc(&*be, bytes)?;
            EngineDevice::upload(&*be, &d, 0, as_bytes(&clear_ranges))?;
            Some(d)
        };

        if !carried_slot.is_empty() {
            tracing::info!(
                tensors = carried_slot.len(),
                per_slot_mib = carried_slot.iter().map(|&(_, n)| n).sum::<u64>() / (1024 * 1024),
                "carried recurrent state (prefix-snapshot size per slot)"
            );
        }

        let mut kv_slot_stride: Vec<(usize, u64)> = Vec::new();
        if batch > 1 {
            // A KDA recurrent state is compiler-owned per-sequence state, so it
            // lives under `kv.` like the KV cache — but it is NOT shaped
            // `[batch][...]`. It is one `[heads, head_dim, head_dim]` f32 block
            // read-modify-written in place, with no token axis and no batch
            // axis (`devgen::kda::declare_kda_state`). Dividing its bytes by
            // `batch` below would hand slot 1 a pointer 1/batch of the way into
            // slot 0's state: no fault, no missing weight, just every sequence
            // corrupting every other one's recurrence.
            //
            // A blob emitted at `RowKind::Sequences` DOES have that axis
            // (`declare_kda_state(.., slots)`), and the carrier that tells the
            // kernel to use it is `PLOW_KDA_F_SEQ_ROWS` in the state step's
            // flags word. So the question is not "is there a recurrent state"
            // but "was this state emitted per-slot" — and only the blob can
            // answer it.
            //
            // CHECK THE CARRIER, NOT THE ENV. `batch` is itself derived from
            // `in.kvlen`, so it agrees with the emitter by construction and
            // cannot discriminate. The flag cannot be faked into being: it is
            // set only where the emitter also sized the state `slots` wide, so
            // its absence at batch > 1 means the state is one block and the
            // stride below would alias every sequence onto every other's — no
            // fault, no missing weight, just fluent wrong output.
            const KDA_F_SEQ_ROWS: u32 = 2;
            let unbatched = blob.progs[dec_ix..].iter().find_map(|p| {
                p.insts
                    .iter()
                    .any(|d| {
                        (d.op == DevOp::KdaStateStep as u16 || d.op == DevOp::KdaStateStepG as u16)
                            && d.i[4] & KDA_F_SEQ_ROWS == 0
                    })
                    .then_some(p.t)
            });
            if let (Some(t), Some(rung)) = (
                blob.tensors
                    .iter()
                    .find(|t| t.name.starts_with("kv.") && t.name.contains("state")),
                unbatched,
            ) {
                return Err(RuntimeError::Device(format!(
                    "PLOW_DECODE_BATCH = {batch} with a recurrent-state tensor `{}` whose decode \
                     rung T={rung} does NOT carry PLOW_KDA_F_SEQ_ROWS. The state is one block with no \
                     slot axis, so the per-slot stride below would alias every sequence's state \
                     onto every other's. Re-emit with PLOW_DECODE_BATCH = {batch} so the emitter \
                     sizes the state per slot and sets the flag, or run at batch 1.",
                    t.name
                )));
            }
            // `kv.blkres` IS EXCLUDED, and leaving it in was an OUT-OF-BOUNDS GPU WRITE.
            //
            // It matches `kv.` but it is not a per-sequence cache. It is K3's snapshot ring,
            // `[t][nb_cap][hidden]` (`devgen::k3`, "kv.blkres"), sized at the LARGEST `t` in the
            // blob — which is the widest PREFILL bucket, not `batch`. Dividing its bytes by
            // `batch` therefore invents a stride that has nothing to do with its layout, and
            // rebasing slot `s` onto `s * bytes/batch` walks off the end:
            //
            //   T_max 8192, batch 16  ->  stride 512 rows;  slot 15 starts at row 7680,
            //   and a 1024-row prefill chunk then writes to row 8704 of an 8192-row tensor.
            //
            // MEASURED: `Memory access fault by GPU node-7` serving the B=16 packet at
            // concurrency 16 with 1038-token prompts (chunks [1024, 512]). It never fired at
            // B=4 because the stride is 2048 there and slot 3 tops out at row 7168.
            //
            // Nothing is lost by not rebasing it: prefill and decode ALTERNATE rather than
            // overlap, and layer 0 resets the ring at the head of every forward pass, so both
            // phases can use rows `[0, t)` as scratch. It carries nothing between passes —
            // which is the same property that lets `begin_slot` skip clearing it.
            kv_slot_stride = blob
                .tensors
                .iter()
                .enumerate()
                .filter(|(_, t)| t.name.starts_with("kv.") && !t.name.contains("blkres"))
                .map(|(i, t)| (i, t.bytes / batch as u64))
                .collect();
            tracing::info!(
                batch,
                kv_buffers = kv_slot_stride.len(),
                slot_bytes = kv_slot_stride.first().map(|&(_, s)| s).unwrap_or(0),
                "KV slot geometry for per-slot prefill"
            );
        }

        // --- pinned staging --------------------------------------------------
        let n_dec_inst = blob.progs[decode].insts.len();
        let mut h_inst =
            EngineDevice::host_alloc_pinned(&*be, n_dec_inst * std::mem::size_of::<DevInst64>())?;
        h_inst
            .as_mut_slice()
            .copy_from_slice(as_bytes(&blob.progs[decode].insts));
        // Prefill stages ids AND pos for a whole chunk, so this must hold
        // 2 * max_bucket_T * 4 bytes. Sizing it at a fixed 64 KiB silently
        // truncated a T=8192 chunk's position array.
        let max_t = blob.progs.iter().map(|g| g.t as usize).max().unwrap_or(1);
        let h_scalar = EngineDevice::host_alloc_pinned(&*be, (max_t * 4 * 2).max(64 * 1024))?;
        let max_pf_inst = blob.progs[..decode]
            .iter()
            .map(|g| g.insts.len())
            .max()
            .unwrap_or(0);
        let h_pf_inst = EngineDevice::host_alloc_pinned(
            &*be,
            (max_pf_inst * std::mem::size_of::<DevInst64>()).max(64),
        )?;
        let pf_src: Vec<Vec<DevInst64>> = blob.progs.iter().map(|g| g.insts.clone()).collect();
        let max_ctr = progs
            .iter()
            .map(|p| p.n_counter as usize * CTR_STRIDE_U32 * 4)
            .max()
            .unwrap_or(4)
            .max(
                progs
                    .iter()
                    .filter_map(|p| p.gq.as_ref())
                    .map(|g| g.n_seg.max(1) as usize * CTR_STRIDE_U32 * 4)
                    .max()
                    .unwrap_or(4),
            );
        let mut h_zero = EngineDevice::host_alloc_pinned(&*be, max_ctr.max(4))?;
        h_zero.as_mut_slice().fill(0);

        let (kvrow, kvrow_i2) = if blob.kvrow.is_empty() {
            derive_kvrow(&blob.progs[decode], &names)
        } else {
            (blob.kvrow.clone(), Vec::new())
        };
        let kvrow_span = kvrow_span(&kvrow.iter().chain(&kvrow_i2).copied().collect::<Vec<_>>());

        // EVERY slot's row 0 must be mapped before any batched decode runs.
        // At batch > 1 all B rows compute whether or not their slot is fed, and
        // an unfed row still writes K/V at its own `pos` — which is 0. Under
        // the flat allocation that wrote garbage into a block nobody read;
        // under VMM it would fault on unmapped VA.
        if let Some(v) = &vmm {
            for b in 0..batch {
                v.ensure_rows(b, 1)?;
            }
        }

        tracing::info!(
            arch = %arch, n_cu = blob.n_cu, progs = progs.len(),
            variant = ?variant, prefill = ?sched_prefill, decode = ?sched_decode,
            n_kvrow = kvrow.len() + kvrow_i2.len(), max_ctx,
            vmm = vmm.is_some(),
            "AMD engine ready"
        );
        // P3 (coalesce the two decode scalar H2Ds into one) is only legal when
        // `in.kvlen` sits immediately after `in.pos` IN THE DEVICE LAYOUT, so
        // log the two bases: the answer is a property of this blob's slab, not
        // of the code.
        tracing::debug!(
            pos = t_pos.map(|t| format!("{:#x}+{}", devp[t].base, devp[t].len)),
            kvlen = t_kvlen.map(|t| format!("{:#x}+{}", devp[t].base, devp[t].len)),
            "decode scalar tensors"
        );

        Ok(AmdEngine {
            be,
            arch,
            n_cu: blob.n_cu,
            progs,
            decode,
            dec_lo,
            devp,
            _weight_slab: weight_slab,
            d_tens,
            tensor_names: names,
            _expert_bufs: expert_bufs,
            d_trace,
            trace_bytes,
            trace_write_bytes: Cell::new(0),
            k_prefill,
            k_decode,
            k_xaudit,
            k_state_clear,
            decode_tiers,
            k_flash,
            sched_prefill,
            sched_decode,
            _modules: modules,
            h_inst,
            h_scalar,
            h_zero,
            h_pf_inst,
            d_state_clear,
            n_state_clear,
            pf_src,
            kvrow,
            kvrow_i2,
            kvrow_span,
            t_ids,
            t_pos,
            t_kvlen,
            t_active,
            t_logits,
            max_ctx,
            weights_bound: ckpt.is_some(),
            batch,
            tens_table: table,
            kv_slot_stride,
            kv_slot: 0,
            carried_slot,
            prefix_snap: (0..batch).map(|_| None).collect(),
            kda_conv_bank_pairs,
            kda_conv_alt_stale: vec![false; batch],
            vmm,
            lm_detail: std::cell::RefCell::new(None),
            pf_stream: blob.progs.iter().map(|g| g.stream.clone()).collect(),
            tp,
            seg_enq_us: 0.0,
            seg_drain_us: 0.0,
            seg_launches: 0,
            seg_window: crate::config::RuntimeConfig::get().amd.seg_window,
        })
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn n_programs(&self) -> usize {
        self.progs.len()
    }

    /// Device memory backing tensor `name`, for a weight loader to fill.
    pub fn tensor_slot(&self, name: &str) -> Option<&DeviceMem> {
        self.tensor_names
            .iter()
            .position(|x| x == name)
            .map(|i| &self.devp[i])
    }

    /// A model-only tensor handle, or a clear error naming what this asset is.
    fn need(&self, h: Option<usize>, what: &str) -> Result<usize> {
        h.ok_or_else(|| {
            RuntimeError::Device(format!(
                "this blob has no `{what}` — it is a BLOCK asset (act.x in, act.x out), \
                 not a model, so token-level entry points do not apply to it"
            ))
        })
    }

    /// The lm_head instruction's tensor operands, as `(handle, name)`.
    ///
    /// The diagnostic for an all-zero `act.logits` with a healthy `act.hn`: a
    /// matmul whose WEIGHT operand resolved to nothing is memset to zero, and
    /// zero times a healthy activation is exactly all-zero logits with no error
    /// anywhere. Naming the operands is the difference between "the lm_head is
    /// wrong" and "the lm_head's B operand is a tensor nobody filled".
    pub fn lm_head_operands(&self, prog: usize) -> Option<(usize, u16, Vec<(usize, String)>)> {
        let insts = self.pf_src.get(prog)?;
        let t_logits = self.t_logits?;
        for (i, d) in insts.iter().enumerate() {
            let is_matmul = is_lm_head_matmul(d.op);
            if is_matmul && d.t[0] as usize == t_logits {
                let ops =
                    d.t.iter()
                        .enumerate()
                        .filter(|(_, &h)| (h as usize) < self.tensor_names.len())
                        .map(|(k, &h)| (k, self.tensor_names[h as usize].clone()))
                        .collect();
                // How many STREAM entries reference it, and in which segments.
                // An instruction the compiler emitted but no stream entry
                // schedules never runs, and its output tensor stays exactly as
                // the loader left it — zero. That is indistinguishable from a
                // broken kernel unless you look here.
                let mut n_ent = 0usize;
                let mut segs: Vec<u16> = Vec::new();
                for e in &self.pf_stream[prog] {
                    if e.inst as usize == i {
                        n_ent += 1;
                        if !segs.contains(&e.seg) {
                            segs.push(e.seg);
                        }
                    }
                }
                segs.sort_unstable();
                *self.lm_detail.borrow_mut() = Some((d.blocks, d.i, n_ent, segs));
                return Some((i, d.op, ops));
            }
        }
        None
    }

    /// `(blocks, i[8])` of the lm_head, valid after [`AmdEngine::lm_head_operands`].
    pub fn lm_head_detail(&self) -> Option<(u16, [u32; 8], usize, Vec<u16>)> {
        self.lm_detail.borrow().clone()
    }

    /// Every tensor the blob declares, in handle order.
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Upload bytes into a named tensor (block I/O, and weight loaders).
    pub fn write_tensor(&mut self, name: &str, src: &[u8]) -> Result<()> {
        let i = self
            .tensor_names
            .iter()
            .position(|x| x == name)
            .ok_or_else(|| RuntimeError::Device(format!("no tensor {name:?}")))?;
        EngineDevice::upload(&*self.be, &self.devp[i], 0, src)
    }

    /// Dump the decode program's `PlowTraceRec[n_stream]` to `path`.
    ///
    /// A no-op unless `PLOW_TRACE_RAW` was set when the engine was built — the
    /// buffer is not allocated otherwise. The records are the LAST launch's, so
    /// call this after a steady-state step, never after the warmup.
    ///
    /// The file is the raw device buffer: `n_stream` 40-byte records, slot
    /// `stream_ofs[cu] + pc`, each carrying `(cu, pc, inst, op, slice,
    /// t_arrive, t_ready, t_end)` in `s_memrealtime` ticks (100 MHz). An entry
    /// no workgroup reached is all-zero. `scripts/k3_trace_report.py` reads it.
    pub fn trace_write(&self, path: &Path) -> Result<()> {
        let Some(m) = &self.d_trace else {
            return Err(RuntimeError::Device(
                "trace buffer was not allocated — set PLOW_TRACE_RAW before loading".into(),
            ));
        };
        let bytes = self.trace_write_bytes.get();
        if bytes == 0 || bytes > self.trace_bytes {
            return Err(RuntimeError::Device(
                "trace buffer has no completed program extent".into(),
            ));
        }
        let mut buf = vec![0u8; bytes];
        EngineDevice::download(&*self.be, m, 0, &mut buf)?;
        std::fs::write(path, &buf)
            .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))
    }

    /// Read a named tensor back.
    pub fn read_tensor(&self, name: &str, dst: &mut [u8]) -> Result<()> {
        let i = self
            .tensor_names
            .iter()
            .position(|x| x == name)
            .ok_or_else(|| RuntimeError::Device(format!("no tensor {name:?}")))?;
        EngineDevice::download(&*self.be, &self.devp[i], 0, dst)
    }

    /// Byte size the blob declares for a tensor.
    pub fn tensor_bytes(&self, name: &str) -> Option<u64> {
        self.tensor_names
            .iter()
            .position(|x| x == name)
            .map(|i| self.devp[i].len)
    }

    /// Build the kernarg block for program `p` at segment `seg`.
    fn kernarg(&self, p: usize, seg: u32) -> DevProgram {
        let g = &self.progs[p];
        DevProgram {
            insts: g.d_inst.base,
            stream: g.d_stream.base,
            stream_ofs: g.d_sofs.base,
            stream_len: g.d_slen.base,
            waits: g.d_waits.base,
            succs: g.d_succs.base,
            // BANKED. `bank` is 0 for the whole life of a single-buffered
            // program, so this is the old expression there.
            counters: g.d_ctr.base + g.bank.current() as u64 * g.ctr_span,
            tensors: self.d_tens.base,
            // EVERY program traces, not just decode. The buffer is sized for the widest one
            // (see `d_trace`'s allocation), and each dispatch writes slot `base + ix` of its own
            // stream, so a run that prefills and then decodes leaves DECODE records behind —
            // which is why `amd-bench` dumps the prefill trace before the decode loop starts.
            trace: self.d_trace.as_ref().map_or(0, |m| m.base),
            cur_seg: seg,
            l2_domains: g.l2_domains,
            // Two-level cache maintenance. Handed to the device only when the OBJECT was built
            // for it; a stale object reads the field as ordinary padding, so an old cubin on a
            // new blob is inert rather than wrong.
            hier_base: g.hier_base,
            n_seg: g.seg_class.len() as u32,
            // Static-path segment windows. Set for every program: an unsegmented
            // one has a single window covering the whole stream, so the decode
            // path does exactly what the old full scan did.
            seg_ofs: if self.seg_window { g.d_seg_ofs.base } else { 0 },
            // Set unconditionally: they are 0 without a GQ appendix, and the
            // static kernel never reads them, so one path serves both.
            gq_stream: g.gq.as_ref().map_or(0, |q| q.d_stream.base),
            gq_seg_ofs: g.gq.as_ref().map_or(0, |q| q.d_seg_ofs.base),
            gq_cursor: g.gq.as_ref().map_or(0, |q| {
                q.d_cursor.base + g.bank.current() as u64 * q.cur_span
            }),
            // Single-GPU leaves all four at zero, and `n_gpu == 0` is the
            // interpreter's "not a group" convention (`tp_decode.c:551` fills
            // them only when `n_gpu > 1`). The collective opcodes never appear
            // in an unsharded program, so nothing reads them there.
            xctr: self.tp.map_or(0, |t| t.xctr),
            peer_scratch: self.tp.map_or(0, |t| t.peer_table),
            rank: self.tp.map_or(0, |t| t.rank),
            n_gpu: self.tp.map_or(0, |t| t.n_gpu),
        }
    }

    /// Re-arm program `p`'s counters and GQ cursor.
    ///
    /// ONCE per dispatch group, NEVER per segment. A segment's producers ran in
    /// an earlier launch, so zeroing between segments unsatisfies them and the
    /// next segment waits on a count that will never come again.
    fn rearm(&self, p: usize) -> Result<()> {
        self.rearm_bank(p, self.progs[p].bank.current())
    }

    /// Zero ONE bank of program `p`'s counters and GQ cursor.
    ///
    /// Split out of [`AmdEngine::rearm`] for the double-buffered [`AmdEngine::run`],
    /// which clears the bank the PREVIOUS dispatch dirtied rather than the one
    /// the next dispatch will read.
    fn rearm_bank(&self, p: usize, bank: u32) -> Result<()> {
        let g = &self.progs[p];
        let n = g.n_counter as usize * CTR_STRIDE_U32 * 4;
        if n > 0 {
            self.be.memcpy_htod_pinned(
                g.d_ctr.base + bank as u64 * g.ctr_span,
                &self.h_zero.as_slice()[..n],
            )?;
        }
        if let Some(q) = &g.gq {
            let n = q.n_seg.max(1) as usize * CTR_STRIDE_U32 * 4;
            self.be.memcpy_htod_pinned(
                q.d_cursor.base + bank as u64 * q.cur_span,
                &self.h_zero.as_slice()[..n],
            )?;
        }
        Ok(())
    }

    /// Re-arm program `p`'s counters and cursor. Used by prefill and by TP's
    /// synchronous single-bank fallback.
    pub fn rearm_prog(&self, p: usize) -> Result<()> {
        self.rearm(p)
    }

    /// Whether TP may select this program's inactive local-counter bank.
    ///
    /// The preflight is separate so a group checks EVERY rank before changing
    /// any rank's bank selection.
    pub fn tp_counter_bank_ready(&self, p: usize) -> bool {
        !ctr_dbuf() || self.progs[p].bank.inactive_ready()
    }

    /// Whether TP should use the post-launch inactive-bank re-arm path.
    pub fn tp_counter_double_buffered(&self) -> bool {
        ctr_dbuf()
    }

    /// Select a clean counter bank for the next TP dispatch.
    ///
    /// With double-buffering disabled this performs the original synchronous
    /// current-bank re-arm. The caller must invoke it on every rank only after
    /// all rank preparation has succeeded.
    pub fn tp_begin_counter_bank(&self, p: usize) -> Result<()> {
        match self.progs[p].bank.begin_tp(ctr_dbuf()) {
            Ok(true) => Ok(()),
            Ok(false) => self.rearm(p),
            Err(()) => Err(RuntimeError::Device(format!(
                "program {p} inactive counter bank is stale: its previous post-launch re-arm \
                 did not complete; refusing to dispatch with uncleared local counters"
            ))),
        }
    }

    /// Re-arm the inactive TP counter/cursor bank after every rank has launched.
    ///
    /// The blocking SDMA copy overlaps the resident megakernels. The bank is
    /// marked reusable only after both its counter and optional GQ-cursor clears
    /// complete successfully.
    pub fn tp_rearm_inactive_counter_bank(&self, p: usize) -> Result<()> {
        if !ctr_dbuf() {
            return Ok(());
        }
        let bank = self.progs[p].bank.inactive();
        self.rearm_bank(p, bank)?;
        self.progs[p].bank.mark_inactive_ready();
        Ok(())
    }

    /// Number of segments in program `p`, and whether segment `seg` is class 4.
    pub fn segment_class(&self, p: usize, seg: usize) -> u8 {
        self.progs[p].seg_class[seg]
    }

    /// Enqueue ONE segment of program `p`. No re-arm, no drain.
    ///
    /// The building block of both the single-GPU segmented run and the TP
    /// per-segment rendezvous. Each launch memcpy's its own kernarg slot, so
    /// mutating `cur_seg` between launches is safe — every packet has already
    /// captured its own copy. For an L2-placed program the caller invokes this
    /// exactly once with `seg == 0`; the device ignores `cur_seg` and drains all
    /// domain windows concurrently.
    pub fn enqueue_segment(&mut self, p: usize, seg: usize) -> Result<()> {
        self.trace_write_bytes
            .set(self.progs[p].trace_records * TRACE_REC_BYTES);
        let use4 = self.k_flash.is_some() && self.progs[p].seg_class[seg] == 4;
        let (k, threads) = match (use4, self.k_flash) {
            (true, Some(kf)) => (kf, WG_THREADS_4),
            _ => (self.k_prefill, WG_THREADS_8),
        };
        let arg = self.kernarg(p, seg as u32);
        EngineDevice::launch_cooperative(
            &*self.be,
            k,
            self.n_cu,
            threads,
            0,
            kernarg_bytes(&arg),
            None,
        )?;
        self.seg_launches += 1;
        Ok(())
    }

    /// Enqueue the single-launch (decode) dispatch of program `p`. No drain.
    ///
    /// Split out of [`AmdEngine::run`] so a TP driver can launch EVERY rank
    /// before waiting on any: the ranks rendezvous on the device through their
    /// cross-GPU counters, inside their own dispatches, so a host wait between
    /// two ranks' launches would make rank 0 spin on a partial rank 1 has not
    /// been dispatched to produce — reintroducing exactly the launched-collective
    /// latency the inline design exists to avoid.
    pub fn enqueue(&mut self, p: usize, k: HsaKernel) -> Result<()> {
        self.trace_write_bytes
            .set(self.progs[p].trace_records * TRACE_REC_BYTES);
        let arg = self.kernarg(p, 0);
        EngineDevice::launch_cooperative(
            &*self.be,
            k,
            self.n_cu,
            WG_THREADS_8,
            0,
            kernarg_bytes(&arg),
            None,
        )?;
        self.seg_launches += 1;
        Ok(())
    }

    /// Wait for everything this rank has enqueued.
    pub fn drain(&self) -> Result<()> {
        EngineDevice::synchronize(&*self.be)
    }

    /// Enqueue the post-drain exact xctr scan used by compact TP audit.
    pub fn enqueue_xaudit(
        &self,
        p: usize,
        xctr: u64,
        n_xctr: u32,
        n_gpu: u32,
        status: u64,
    ) -> Result<()> {
        let k = self
            .k_xaudit
            .ok_or_else(|| RuntimeError::Device("compact TP audit kernel was not loaded".into()))?;
        let prog = &self.progs[p];
        let arg = XAuditArgs {
            insts: prog.d_inst.base,
            n_inst: prog.n_inst,
            _pad: 0,
            xctr,
            n_xctr,
            n_gpu,
            status,
        };
        EngineDevice::launch_kernel(
            &*self.be,
            k,
            1,
            256,
            0,
            as_bytes(std::slice::from_ref(&arg)),
            None,
        )
    }

    /// Run every segment of program `p`, then drain ONCE.
    ///
    /// The single drain is correct only because each dispatch carries the AQL
    /// barrier bit, which chains segment k+1 behind segment k on the packet
    /// processor with no host round-trip.
    ///
    /// # This shape is WRONG under TP, and that is not a performance opinion
    ///
    /// `tp_decode.c` records the failure: enqueueing all of one rank's segments
    /// and only then moving to the next rank let the ranks **desync — a lagging
    /// rank made peers time out and bail, giving a WRONG, 100x-slow reduction at
    /// TP>=4.** A class-8 segment holds both of a layer's all-reduces, and the
    /// inline system-scope gate only rendezvouses cheaply if every rank is
    /// inside that segment at the same time. So a TP prefill goes PER-SEGMENT,
    /// ALL-RANKS, with a host barrier between segments — see
    /// [`crate::exec::amd_tp::AmdTpGroup::prefill`]. This method stays as it is
    /// because on ONE GPU there is no peer to desync from.
    ///
    /// # Never valid on an L2-PLACED program
    ///
    /// Placement makes `seg` an L2 DOMAIN, so `seg_class` is 8 domains that all read as
    /// wave-class 8 and this loop would issue 8 launches over windows that are meant to run
    /// CONCURRENTLY — the opposite of the point. It would still produce correct tokens (the
    /// first launch drains every domain and the other 7 find their cursors past `hi` and exit),
    /// which is exactly why it has to be refused rather than left to be noticed: a silent 8x
    /// launch overhead on a change whose whole purpose is latency. Placed programs go through
    /// [`AmdEngine::run`], which is already single-launch.
    pub fn run_segmented(&mut self, p: usize) -> Result<()> {
        if self.progs[p].l2_domains != 0 {
            return Err(RuntimeError::Device(format!(
                "program {p} is L2-domain placed ({} domains): its `seg` windows are L2 domains \
                 meant to drain concurrently in ONE launch, not wave-class segments to relaunch \
                 over. Use `run` (the single-launch path).",
                self.progs[p].l2_domains
            )));
        }
        self.rearm(p)?;
        let n_seg = self.progs[p].seg_class.len();
        let t0 = std::time::Instant::now();
        for seg in 0..n_seg {
            self.enqueue_segment(p, seg)?;
        }
        let t1 = std::time::Instant::now();
        if let Err(e) = self.drain() {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                program = p,
                segments = n_seg,
                grid = self.n_cu,
                "segmented run failed at drain"
            );
            return Err(e);
        }
        let t2 = std::time::Instant::now();
        self.seg_enq_us += (t1 - t0).as_secs_f64() * 1e6;
        self.seg_drain_us += (t2 - t1).as_secs_f64() * 1e6;
        crate::obs::ttft::PF_SEGMENTS.tally(n_seg as u64);
        crate::obs::ttft::PF_ENQUEUE.add((t1 - t0).as_nanos() as u64);
        crate::obs::ttft::PF_DRAIN.add((t2 - t1).as_nanos() as u64);
        Ok(())
    }

    /// Single-launch run — the decode path, which is not segmented.
    /// # The re-arm is BEHIND the enqueue, not in front of it
    ///
    /// With [`ctr_dbuf`] on, the bank this dispatch reads was already zeroed —
    /// by the PREVIOUS dispatch, while the GPU was busy. So the order is
    /// enqueue, then clear the stale bank, then drain: the clear's two blocking
    /// SDMA round trips (56 µs measured, §`ctr_dbuf`) overlap 11.6 ms of
    /// megakernel instead of delaying its start. Flipping `bank` after the
    /// enqueue is safe because `enqueue` memcpy'd its own kernarg slot, so the
    /// launch already captured the address it will use.
    ///
    /// Correctness rests on one thing: the bank being cleared is not the one
    /// the in-flight dispatch is reading. That holds because `run` drains
    /// before it returns, so dispatch N-1 (which dirtied `stale`) has retired
    /// before dispatch N is even staged.
    pub fn run(&mut self, p: usize, k: HsaKernel) -> Result<()> {
        use crate::obs::dstep;
        if !ctr_dbuf() {
            dstep::timed(&dstep::REARM, || self.rearm(p))?;
        }
        let t0 = std::time::Instant::now();
        if let Err(e) = dstep::timed(&dstep::ENQUEUE, || self.enqueue(p, k)) {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                program = p,
                grid = self.n_cu,
                block = WG_THREADS_8,
                "program dispatch failed at enqueue"
            );
            return Err(e);
        }
        if ctr_dbuf() {
            let cur = self.progs[p].bank.current();
            dstep::timed(&dstep::REARM, || self.rearm_bank(p, 1 - cur))?;
            self.progs[p].bank.select_rearmed_inactive();
        }
        // The drain is where an async kernel trap surfaces — capture the
        // dispatch shape at the site before propagating.
        if let Err(e) = dstep::timed(&dstep::DRAIN, || self.drain()) {
            tracing::warn!(
                error = %e,
                error_code = ?e.device_code(),
                fatal = e.is_fatal(),
                program = p,
                grid = self.n_cu,
                block = WG_THREADS_8,
                "program dispatch failed at drain"
            );
            return Err(e);
        }
        self.seg_drain_us += t0.elapsed().as_secs_f64() * 1e6;
        Ok(())
    }

    /// Patch the KV-append row into every `kvrow` site and push ONE contiguous
    /// slice of the instruction stream.
    ///
    /// The sites are scattered in k/v pairs across all layers (Gemma-31B:
    /// `[4,664]` of 676), so one contiguous slice beats a per-site scatter:
    /// fewer bytes than the whole stream and, more importantly, ONE h2d
    /// submission instead of `n_kvrow` of them. Submission overhead, not bytes,
    /// is what costs here. Patched in the PINNED copy so the slice is
    /// contiguous in pinned memory and needs no per-call page pin.
    fn patch_kvrow(&mut self, dp: usize, pos: u32) -> Result<()> {
        let Some((lo, hi)) = self.kvrow_span else {
            return Ok(());
        };
        let sz = std::mem::size_of::<DevInst64>();
        {
            let slab = self.h_inst.as_mut_slice();
            let n = slab.len() / sz;
            // SAFETY: the slab was allocated as `n_inst * size_of::<DevInst64>()`
            // and seeded from a `&[DevInst64]`, so it is exactly `n` live,
            // aligned, initialised records. `DevInst64` is `#[repr(C)]` POD.
            let insts: &mut [DevInst64] =
                unsafe { std::slice::from_raw_parts_mut(slab.as_mut_ptr() as *mut DevInst64, n) };
            for (sites, field) in [(&self.kvrow, 3usize), (&self.kvrow_i2, 2)] {
                for &idx in sites {
                    let i = idx as usize;
                    if i >= n {
                        return Err(RuntimeError::Device(format!(
                            "kvrow site {i} past the decode program's {n} instructions"
                        )));
                    }
                    insts[i].i[field] = pos;
                }
            }
        }
        let src = &self.h_inst.as_slice()[lo * sz..(hi + 1) * sz];
        // NOT `upload`: that pins its source, and pinning an already
        // device-visible pinned slab is invalid (HSA 4096) as well as
        // syscall-class. This is the whole reason `h_inst` is pinned.
        self.be
            .memcpy_htod_pinned(self.progs[dp].d_inst.base + (lo * sz) as u64, src)
    }

    /// One decode step at absolute position `pos`, with `kvlen` valid KV rows
    /// after it. Returns the token id the DEVICE sampled.
    ///
    /// Per-step host work is deliberately tiny: patch the KV-append row into the
    /// instructions, push one contiguous slice of them, push two 4-byte scalars,
    /// launch, wait, read 4 bytes back.
    ///
    /// `in.ids` is NOT uploaded. The device's argmax wrote the sampled token
    /// there at the end of the previous launch, which is exactly where this
    /// step's embed reads it; writing a host copy over it would feed the model
    /// last step's token twice.
    pub fn decode_step(&mut self, pos: u32, kvlen: u32) -> Result<u32> {
        use crate::obs::dstep;
        self.vmm_ensure(self.kv_slot, pos + 1)?;
        dstep::timed(&dstep::PREPARE, || self.decode_prepare(pos, kvlen))?;
        self.run(self.decode, self.decode_kernel_for(self.decode))?;
        let id = dstep::timed(&dstep::READ, || self.read_sampled())?;
        if let Some(v) = &self.vmm {
            v.advise(self.kv_slot, pos + 1);
        }
        Ok(id)
    }

    /// Everything a decode step does BEFORE the dispatch: patch the KV-append
    /// row and push the two scalars. No launch, no wait.
    ///
    /// Separate because a TP group must prepare and re-arm every rank before
    /// dispatching any of them — see [`AmdEngine::enqueue`].
    pub fn decode_prepare(&mut self, pos: u32, kvlen: u32) -> Result<()> {
        if pos as usize >= self.max_ctx {
            return Err(RuntimeError::Device(format!(
                "position {pos} past max_ctx {}",
                self.max_ctx
            )));
        }
        self.sync_kda_conv_alt(self.kv_slot)?;
        let dp = self.decode;
        self.patch_kvrow(dp, pos)?;

        // Stage both scalars in pinned memory for the same reason.
        {
            let s = self.h_scalar.as_mut_slice();
            s[..4].copy_from_slice(&pos.to_le_bytes());
            s[4..8].copy_from_slice(&kvlen.to_le_bytes());
        }
        let ptr_pos = self.devp[self.need(self.t_pos, "in.pos")?].base;
        let ptr_kvlen = self.devp[self.need(self.t_kvlen, "in.kvlen")?].base;
        self.be
            .memcpy_htod_pinned(ptr_pos, &self.h_scalar.as_slice()[..4])?;
        self.be
            .memcpy_htod_pinned(ptr_kvlen, &self.h_scalar.as_slice()[4..8])?;
        Ok(())
    }

    /// The token the DEVICE sampled into `in.ids`, read after a drain.
    ///
    /// 4 bytes, not the logit row — and read back through the PINNED slab.
    /// `download` pins its destination per call, so a 4-byte readback paid a
    /// page-lock syscall every single decode step.
    pub fn read_sampled(&mut self) -> Result<u32> {
        let src = self.devp[self.need(self.t_ids, "in.ids")?].base;
        let slab = self.h_scalar.as_mut_slice();
        self.be.memcpy_dtoh_pinned(&mut slab[..4], src)?;
        Ok(u32::from_le_bytes(
            self.h_scalar.as_slice()[..4].try_into().expect("4 bytes"),
        ))
    }

    /// The bucket width to shrink prefill row counts against, or `None` when
    /// `PLOW_RAGGED_CHUNK` is off (or this is the decode program).
    ///
    /// ONE function, read by both `patch_prefill` and `prefill_prepare`, because
    /// the instruction shrink and the `in.kvlen` upload are the same decision
    /// taken in two places — see `rebase_chunk_rows` for what happens when they
    /// disagree.
    fn ragged_bucket(&self, prog: usize) -> Option<u32> {
        (crate::config::RuntimeConfig::get().amd.ragged_chunk && prog != self.decode)
            .then(|| self.progs[prog].t)
    }

    /// Refuse `PLOW_RAGGED_CHUNK` on a packet whose prefill is ROW-BANDED.
    ///
    /// `PLOW_GLM_XR_BAND=K` splits each `[T, hidden]` collective into K row bands,
    /// giving the producing `Gemm` `M = T/K` at `a_row0 = i*T/K` and the
    /// `XReduce` `n = (T/K)*hidden` at its own offset. The row shrink's guard
    /// then declines the `Gemm` (its M is not `T`) but ACCEPTS the `XReduce` (its
    /// n IS a multiple of `T`), which would reduce the wrong element range —
    /// exactly the silent half-application the guard exists to prevent.
    ///
    /// Detected by the signature banding leaves and nothing else does: a non-lm_head
    /// matmul with `a_row0 != 0`, or a `MoeCombinePf` with `t_row0 != 0`. The
    /// shipped GLM-5.2 blob is unbanded (the axis is emit-time, default OFF and
    /// measured net-negative), so this is a guard, not a limitation in practice.
    fn refuse_unraggable(&self) -> Result<()> {
        if !crate::config::RuntimeConfig::get().amd.ragged_chunk {
            return Ok(());
        }
        // Say so ONCE, at the first prefill. An A/B whose two arms differ only by
        // an environment variable needs a positive signal in the log that the
        // variable reached the process; "the number moved" is not that signal.
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::info!(
                buckets = ?(0..self.decode).map(|p| self.progs[p].t).collect::<Vec<_>>(),
                "PLOW_RAGGED_CHUNK: fewest-launch cover, last chunk runs at its real row count"
            )
        });
        for p in 0..self.decode {
            for (i, d) in self.pf_src[p].iter().enumerate() {
                let banded_gemm = is_lm_head_matmul(d.op)
                    && d.i[4] != 0
                    && Some(d.t[0] as usize) != self.t_logits;
                let banded_combine = d.op == DevOp::MoeCombinePf as u16 && d.i[3] != 0;
                if banded_gemm || banded_combine {
                    return Err(RuntimeError::Device(format!(
                        "PLOW_RAGGED_CHUNK cannot serve this packet: prefill bucket T={} \
                         instruction #{i} (op {}) carries a non-zero row-band offset, which is \
                         the PLOW_GLM_XR_BAND layout. The ragged row shrink would rescale the \
                         banded collective without shrinking the band's GEMM. Re-emit without \
                         PLOW_GLM_XR_BAND, or serve without PLOW_RAGGED_CHUNK.",
                        self.progs[p].t, d.op
                    )));
                }
            }
        }
        Ok(())
    }

    /// Patch a prefill program's instructions for the chunk at `[c0, c0+clen)`.
    ///
    /// The row/window families live in [`rebase_chunk`] (which is where their
    /// identities are argued and where the unit test drives them). What stays
    /// here is the one site that needs the ENGINE's state:
    ///
    /// * lm_head — the FIRST matmul writing `act.logits` → `i[4] = clen - 1`,
    ///   the chunk's last REAL row, so the sampled logits come from the last
    ///   prompt token and not from a padded one.
    fn patch_prefill(&mut self, prog: usize, c0: u32, clen: u32) -> Result<()> {
        let sz = std::mem::size_of::<DevInst64>();
        let n = self.pf_src[prog].len();
        // Rebuild from the pristine copy every chunk: patches must not
        // accumulate across chunks, and c0 changes every time.
        self.h_pf_inst.as_mut_slice()[..n * sz].copy_from_slice(as_bytes(&self.pf_src[prog]));
        // SAFETY: the slab holds exactly `n` live `DevInst64` records, just
        // written from a `&[DevInst64]`. `DevInst64` is `#[repr(C)]` POD.
        let insts: &mut [DevInst64] = unsafe {
            std::slice::from_raw_parts_mut(self.h_pf_inst.as_mut_ptr() as *mut DevInst64, n)
        };

        rebase_chunk_rows(
            insts,
            &self.tensor_names,
            c0,
            clen,
            self.ragged_bucket(prog),
        );

        let mut lm = None;
        for (i, d) in insts.iter().enumerate() {
            let is_matmul = is_lm_head_matmul(d.op);
            if Some(d.t[0] as usize) == self.t_logits && is_matmul {
                lm = Some(i);
                break;
            }
        }
        // A BLOCK has no lm_head; there is no a_row0 to place and that is not
        // an error. A MODEL without one is, because the sampled logits would
        // then come from whatever row the compiler baked in.
        match (lm, self.t_logits) {
            // DIAGNOSTIC: `--amd-lm-row0` / `PLOW_LM_ROW0=1` leaves a_row0 at
            // 0 instead of the
            // chunk's last real row. It samples the WRONG row, so it is not a
            // serving mode — it exists to answer one question. The lm_head is
            // the ONLY op whose a_row0 the host patches to a non-zero value at
            // runtime, so a bug in the a_row0 path is invisible to any check
            // that inspects the packet statically (where all fp8 GEMMs carry
            // a_row0 == 0) and shows up only here.
            (Some(lm), _) if crate::config::RuntimeConfig::get().amd.lm_row0 => {
                tracing::warn!(
                    lm,
                    "PLOW_LM_ROW0=1: a_row0 left at 0 — DIAGNOSTIC, wrong row"
                );
                insts[lm].i[4] = 0;
            }
            (Some(lm), _) => insts[lm].i[4] = clen - 1,
            (None, Some(_)) => {
                // A BLOCK can DECLARE act.logits and never write it — the
                // tensor table is emitted from the model's vocabulary of names,
                // not from what this program actually produces. Refusing here
                // rejected the layer-0 A/B asset outright. Warn, because on a
                // real model the same shape means the sampled logits come from
                // whatever row the compiler baked in.
                tracing::warn!(
                    prog,
                    "act.logits is declared but no matmul writes it — no a_row0 to place \
                     (expected for a block asset, WRONG for a model)"
                );
            }
            (None, None) => {}
        }

        self.be.memcpy_htod_pinned(
            self.progs[prog].d_inst.base,
            &self.h_pf_inst.as_slice()[..n * sz],
        )
    }

    /// Resolve a chunk plan to the programs and ranges that run it.
    ///
    /// Shared by the single-GPU prefill and the TP one, so both walk the prompt
    /// identically — a TP prefill that chunked differently from tp=1 would not
    /// be comparable token-for-token, which is the whole acceptance test.
    pub fn chunk_steps(&self, chunks: &[u32], n_prompt: u32) -> Result<Vec<ChunkStep>> {
        let mut out = Vec::with_capacity(chunks.len());
        let mut c0 = 0u32;
        for &ch in chunks {
            // The DP's cover is >= the prompt (buckets are a ladder, so the
            // last chunk usually overshoots), and it can overshoot by a whole
            // bucket. A chunk starting past the end has clen == 0, and
            // `a_row0 = clen - 1` would then wrap to u32::MAX and index the
            // logits off a row that does not exist. Stop instead.
            if c0 >= n_prompt {
                break;
            }
            let prog = (0..self.dec_lo)
                .find(|&p| self.progs[p].t == ch)
                .ok_or_else(|| {
                    RuntimeError::Device(format!("no compiled bucket for chunk T={ch}"))
                })?;
            out.push(ChunkStep {
                prog,
                c0,
                clen: (n_prompt - c0).min(ch),
            });
            c0 += ch;
        }
        Ok(out)
    }

    /// Upload one chunk's `ids`/`pos`/`kvlen` and patch its bucket program. No
    /// dispatch.
    pub fn prefill_prepare(&mut self, prompt: &[u32], step: ChunkStep) -> Result<()> {
        if !self.kda_conv_bank_pairs.is_empty() {
            self.kda_conv_alt_stale[self.kv_slot] = true;
        }
        let ch = self.progs[step.prog].t;
        // in.kvlen FIRST, so it can borrow the head of the staging slab before
        // ids/pos fill it — the slab is sized for exactly `ids + pos` at the
        // widest bucket and has no spare word past them.
        //
        // This is the MLA prefill's QUERY BASE, and the only place it comes
        // from. `d_flash_mla_decode` (the body `d_flash_mla_prefill` wraps)
        // computes `qpos = kv_len[b] - n_tok + t` with `n_tok = i[4]`, which
        // devgen bakes at the BUCKET width — so `kv_len` must be `c0 + ch`, not
        // `c0 + clen`, or every query in the chunk shifts down by the padding.
        // The pad rows really are part of the cache: they write `ckv`/`krot` at
        // rows `c0+clen .. c0+ch`, and a real row `i` is causally bounded at
        // `c0+i`, so it never reads one.
        //
        // Nothing wrote this during prefill before. The dense-GQA path does not
        // need it (`FlashPrefill` takes n_kv as the `i[1]` immediate that
        // [`rebase_chunk`] patches), so the omission was invisible until MLA
        // prefill landed — at which point `qpos` came out of an uninitialised
        // device word. The CUDA engine has uploaded it since it gained MLA
        // prefill ([`super::gpu`], `run_one_prefill_chunk`).
        // Under RAGGED-M the flash's `n_tok` was shrunk to `clen`, so the query
        // base `qpos = kv_len - n_tok + t` needs `kv_len = c0 + clen` to land on
        // the same absolute positions. The two are one decision, taken once by
        // `ragged_bucket`; see `rebase_chunk_rows`.
        let kv_rows = if self.ragged_bucket(step.prog).is_some() {
            step.c0 + step.clen
        } else {
            step.c0 + ch
        };
        if let Some(t) = self.t_kvlen {
            let d_kvlen = self.devp[t].base;
            self.h_scalar.as_mut_slice()[..4].copy_from_slice(&kv_rows.to_le_bytes());
            self.be
                .memcpy_htod_pinned(d_kvlen, &self.h_scalar.as_slice()[..4])?;
        }
        // ids: the chunk's tokens, ZERO-PADDED past clen. Padded rows write
        // KV nothing reads — `n_kv` bounds every later read at c0+clen.
        // pos: ABSOLUTE positions, so RoPE and the KV row agree with what
        // the decode steps will later assume.
        {
            let s = self.h_scalar.as_mut_slice();
            for i in 0..ch as usize {
                let id = if (i as u32) < step.clen {
                    prompt[(step.c0 + i as u32) as usize]
                } else {
                    0
                };
                s[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
            let off = ch as usize * 4;
            for i in 0..ch as usize {
                let p = step.c0 + i as u32;
                s[off + i * 4..off + i * 4 + 4].copy_from_slice(&p.to_le_bytes());
            }
        }
        let (d_ids, d_pos) = (
            self.devp[self.need(self.t_ids, "in.ids")?].base,
            self.devp[self.need(self.t_pos, "in.pos")?].base,
        );
        let nb = ch as usize * 4;
        self.be
            .memcpy_htod_pinned(d_ids, &self.h_scalar.as_slice()[..nb])?;
        self.be
            .memcpy_htod_pinned(d_pos, &self.h_scalar.as_slice()[nb..nb * 2])?;
        self.patch_prefill(step.prog, step.c0, step.clen)
    }

    /// The chunk plan for a prompt, from the compiled bucket ladder.
    pub fn plan_for(&self, n_prompt: u32) -> Result<Vec<u32>> {
        // Here rather than at load: it is the ONE call every prefill path goes
        // through (`prefill`, `prefill_span`, `AmdTpGroup::plan_for`), so a
        // banded packet cannot reach the row shrink by some other door.
        self.refuse_unraggable()?;
        // `dec_lo`, not `decode`: with a DECODE BATCH LADDER the trailing programs are decode
        // rungs, and offering one to `plan_chunks` as a prefill bucket would cover a prompt
        // with a decode program. Without a ladder the two are the same index.
        let buckets: Vec<u32> = (0..self.dec_lo).map(|p| self.progs[p].t).collect();
        // Same reason `refuse_unraggable` announces itself: an A/B whose arms
        // differ only by `PLOW_LAUNCH_ROWS` needs a positive signal that the
        // variable reached the process. "The number moved" is not that signal,
        // and neither is "the number did not move".
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            let cfg = &crate::config::RuntimeConfig::get().amd;
            tracing::info!(
                launch_rows = cfg.launch_rows.unwrap_or(LAUNCH_ROWS),
                overridden = cfg.launch_rows.is_some(),
                ragged = cfg.ragged_chunk,
                // The packet's own cap, not a constant: this line is the only
                // positive signal that a blob's widest rung reached the planner.
                max_chunk = buckets.iter().copied().max().unwrap_or(0),
                ?buckets,
                "prefill chunk policy"
            )
        });
        plan_chunks(&buckets, n_prompt)
    }

    /// Prefill `prompt`, leaving the KV cache populated for `[0, prompt.len())`
    /// and the first sampled token in `in.ids`.
    ///
    /// Returns the token the device sampled from the last real prompt row.
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        if prompt.is_empty() {
            return Err(RuntimeError::Device("prefill of an empty prompt".into()));
        }
        if prompt.len() > self.max_ctx {
            return Err(RuntimeError::Device(format!(
                "prompt of {} tokens exceeds max_ctx {}",
                prompt.len(),
                self.max_ctx
            )));
        }
        // The padded cover, not the prompt length, is what the kernels write —
        // refuse before allocating rather than clamping and writing past it.
        self.refuse_overlong_cover(prompt.len() as u32)?;
        // Back the rows this prefill writes, in whichever slot the KV base is
        // rebased onto. Here rather than in `prefill_slot` so the direct
        // single-sequence path (`amd-bench`, `AmdServe` at batch 1) is covered
        // by the same line.
        self.vmm_ensure(self.kv_slot, self.prefill_rows(prompt.len() as u32))?;

        let t_plan = std::time::Instant::now();
        let chunks = self.plan_for(prompt.len() as u32)?;
        let steps = self.chunk_steps(&chunks, prompt.len() as u32)?;
        crate::obs::ttft::PF_PLAN.add(t_plan.elapsed().as_nanos() as u64);
        crate::obs::ttft::set_cover(&chunks);
        tracing::info!(
            tokens = prompt.len(),
            chunks = ?chunks,
            "prefill plan"
        );

        for step in steps {
            let t = std::time::Instant::now();
            self.prefill_prepare(prompt, step)?;
            crate::obs::ttft::PF_PREPARE.add(t.elapsed().as_nanos() as u64);
            self.run_segmented(step.prog)?;
        }

        // The device sampled into in.ids itself; the first decode step will
        // embed it from there without the host touching it.
        let t_read = std::time::Instant::now();
        let src = self.devp[self.need(self.t_ids, "in.ids")?].base;
        let slab = self.h_scalar.as_mut_slice();
        self.be.memcpy_dtoh_pinned(&mut slab[..4], src)?;
        crate::obs::ttft::PF_READ.add(t_read.elapsed().as_nanos() as u64);
        Ok(u32::from_le_bytes(
            self.h_scalar.as_slice()[..4].try_into().expect("4 bytes"),
        ))
    }

    /// Sequences one decode dispatch advances.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Point every `kv.*` tensor at sequence `slot`'s block of the cache.
    ///
    /// THE PREFILL PROGRAM IS SINGLE-SEQUENCE and always will be: its
    /// `HeadNormRope` runs with `n_batch_kv == 0`, so it writes at
    /// `hh * out_stride + row` — the *first* sequence's block relative to
    /// whatever base the pointer table hands it. Rebasing the pointer is
    /// therefore the whole of "prefill into slot s": exact, one 8-byte edit per
    /// KV buffer, and it needs no second prefill program and no kernel change.
    ///
    /// The decode program must always run at slot 0 (it derives each
    /// sequence's block itself from `n_batch_kv`), so every prefill restores
    /// the base before returning. A stale rebase would put ALL sequences'
    /// decode KV inside one slot's block — hence the invariant is enforced in
    /// [`AmdEngine::decode_step_batched`] rather than left to callers.
    pub fn kv_rebase(&mut self, slot: usize) -> Result<()> {
        if self.kv_slot == slot || self.kv_slot_stride.is_empty() {
            return Ok(());
        }
        if slot >= self.batch {
            return Err(RuntimeError::Device(format!(
                "kv_rebase to slot {slot} past batch {}",
                self.batch
            )));
        }
        for &(i, stride) in &self.kv_slot_stride {
            let base = self.devp[i].base + stride * slot as u64;
            self.tens_table[i * 8..i * 8 + 8].copy_from_slice(&base.to_le_bytes());
        }
        // One upload of the whole table (a few KiB) beats one per KV buffer:
        // there are 2-4 per layer and the submission, not the bytes, is the
        // cost. This is off the per-token path — it happens once per prefill.
        EngineDevice::upload(&*self.be, &self.d_tens, 0, &self.tens_table)?;
        self.kv_slot = slot;
        if tracing::enabled!(tracing::Level::DEBUG) {
            let (i, stride) = self.kv_slot_stride[0];
            let mut back = [0u8; 8];
            EngineDevice::download(&*self.be, &self.d_tens, i as u64 * 8, &mut back)?;
            tracing::debug!(
                slot,
                tensor = %self.tensor_names[i],
                want = format_args!("{:#x}", self.devp[i].base + stride * slot as u64),
                got = format_args!("{:#x}", u64::from_le_bytes(back)),
                "kv rebase readback"
            );
        }
        Ok(())
    }

    /// Prefill `prompt` into sequence slot `slot`, restoring the decode base.
    pub fn prefill_slot(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
        self.kv_rebase(slot)?;
        let r = self.prefill(prompt);
        // Restore even on failure: a half-prefilled slot is recoverable, a
        // pointer table left pointing at slot s is not.
        self.kv_rebase(0)?;
        r
    }

    /// Rows a prefill of `n_prompt` tokens WRITES, which is the padded bucket
    /// cover, not `n_prompt`. `prefill_prepare` zero-pads the last chunk out to
    /// its bucket width and those pad rows write KV (nothing reads them —
    /// `n_kv` bounds every later read at `c0 + clen`), so the backing has to
    /// cover them or the pad write faults.
    ///
    /// NOT CLAMPED TO `max_ctx`, and it used to be. `.min(self.max_ctx)` made this
    /// function contradict its own doc comment: the cover is exactly the count of
    /// rows the kernels WILL write, so clamping it returns a number that is not
    /// that, and the caller then backs fewer rows than the hardware touches. See
    /// [`AmdEngine::refuse_overlong_cover`] for what the caller does about it.
    fn prefill_rows(&self, n_prompt: u32) -> u32 {
        let cover: u32 = self
            .plan_for(n_prompt)
            .map(|c| c.iter().sum())
            .unwrap_or(n_prompt);
        cover.max(n_prompt)
    }

    /// Refuse a prompt whose PADDED cover runs past the compiled context.
    ///
    /// `plan_chunks` covers a prompt with compiled bucket widths, so the last
    /// chunk is rounded UP: at `max_ctx = 1500` a 1499-token prompt plans as
    /// `1024 + 512 = 1536`. Every admission check upstream tests `n_prompt`
    /// (`prefill` here, `EngineServe::prefill` at `>= max_ctx`), and 1499 passes
    /// all of them — but the kernels execute the full bucket and write padded KV
    /// rows through 1535, into a cache whose geometry is `max_ctx` rows.
    ///
    /// The old code hid this by clamping [`AmdEngine::prefill_rows`] to `max_ctx`,
    /// which does not make the writes stop — it only stops the ROWS FROM BEING
    /// BACKED. Under `PLOW_VMM_KV` that is a fault on unmapped VA; without VMM it
    /// is a silent write past the end of the KV tensor into whatever the allocator
    /// placed next. The second is the one worth refusing for: it corrupts a
    /// neighbour and reports nothing.
    ///
    /// A refusal rather than a bigger allocation, because the allocation is not
    /// this layer's to grow: `max_ctx` is read out of the compiled `in.pos` tensor
    /// and `VmmPool::ensure_rows` clamps to `geo.max_ctx` independently, so both
    /// the reservation and the mapping are sized by the PACKET. Making the padded
    /// cover fit is an emitter-side decision (pad the KV geometry to the worst
    /// bucket overshoot, or compile a terminal context-sized bucket); until it is
    /// made, this is the boundary that says so instead of writing past it.
    ///
    /// The refused band is narrow — only prompts within one bucket-rounding of
    /// `max_ctx` — and the message names all four numbers so the fix is obvious.
    fn refuse_overlong_cover(&self, n_prompt: u32) -> Result<()> {
        let cover = self.prefill_rows(n_prompt);
        if cover as usize > self.max_ctx {
            return Err(RuntimeError::Rejected(format!(
                "prompt of {n_prompt} tokens plans as {:?} = {cover} padded rows, past max_ctx \
                 {}. The kernels write every row of the last bucket, so this would write KV rows \
                 [{}, {cover}) outside the cache. Shorten the prompt, or recompile with a \
                 prefill bucket that lands on {} without overshooting.",
                self.plan_for(n_prompt).unwrap_or_default(),
                self.max_ctx,
                self.max_ctx,
                self.max_ctx,
            )));
        }
        Ok(())
    }

    /// Map physical backing for `seq` out to `rows`. No-op without VMM.
    fn vmm_ensure(&self, seq: usize, rows: u32) -> Result<()> {
        match &self.vmm {
            Some(v) if v.mapped_rows(seq) < rows => v.ensure_rows(seq, rows),
            _ => Ok(()),
        }
    }

    /// Release slot `seq`'s physical backing, remap its row 0, and CLEAR any
    /// carried recurrent state.
    ///
    /// Called when a slot is handed to a NEW sequence: the outgoing sequence's
    /// blocks are what a growable pool exists to reclaim. Row 0 goes straight
    /// back because an idle row still writes KV at `pos = 0`.
    ///
    /// # The clear, and why the KV cache does not need one but KDA does
    ///
    /// An append-only KV cache carries nothing between sequences: `kvlen` returns
    /// to 0, every row the new sequence reads is a row it wrote, and the stale
    /// bytes underneath are unreachable. That is the whole argument behind
    /// [`kv_skips_zeroing`], and it is why handing over a slot costs a pointer
    /// remap and no memset.
    ///
    /// [`is_carried_state`] tensors break that argument: the KDA recurrence READS
    /// `state` on its very first token, and the conv arms read a window that is
    /// supposed to hold the `W - 1` tokens before the sequence began. With no
    /// clear, "the tokens before this sequence began" were the previous REQUEST's
    /// — so a second prompt started from the first one's accumulated state.
    ///
    /// THE CLEAR IS WHOLE-TENSOR, AND AT `batch > 1` THAT IS NOW WRONG. This comment
    /// used to say the state was not per-slot; it is, since `declare_kda_state` gained
    /// `slots` and `k3.rs` passes `slots = t` for a sequence-rows program
    /// (`crates/devgen/src/k3.rs`, `RowKind::Sequences`). So `kv.{layer}.state` and the
    /// conv states hold B INDEPENDENT recurrences, and the `memset` below zeroes ALL of
    /// them — admitting into slot 2 would wipe slots 0/1/3 mid-stream.
    ///
    /// That is latent rather than live only because this function used to be called ONLY on
    /// the single-GPU path, and the shipped K3 config is TP8 — so on TP nothing cleared
    /// carried state at all and every request after the first on a slot inherited the
    /// previous one's recurrence across 69 of K3's 93 layers.
    ///
    /// # BOTH HALVES ARE FIXED HERE
    ///
    /// The clear is now PER SLOT, and [`AmdTpGroup::begin_slot`] calls it on every rank.
    ///
    /// The stride is `len / batch`, which is right because these tensors are SLOT-MAJOR by
    /// construction: `declare_kda_state` sizes `state` as `state_elems * 4 * slots` and
    /// `conv_state` as `proj * conv_w * 4 * slots`, and the kernels index them as
    /// `st_h + t*bstride` with `bstride = H*D*D` (state) and `C*W` (conv) — the same
    /// `[slot][...]` layout the memset now assumes. `slots` is `t` for a sequence-rows
    /// program and 1 otherwise, so at `batch == 1` this is byte-identical to the old
    /// whole-tensor clear.
    ///
    /// `kv.blkres` is EXCLUDED at `batch > 1`, and that is deliberate rather than an
    /// oversight. It is `[T][nb_cap][hidden]` sized at `max(T_max, B)` rows — T_max being
    /// the widest PREFILL bucket — so `len / batch` is NOT its row stride and a per-slot
    /// memset would clear the wrong bytes. It also carries nothing across steps: layer 0
    /// is a snapshot layer that resets the ring every forward pass, so each pass
    /// re-establishes it. Clearing it was always belt-and-braces; skipping it at `batch > 1`
    /// is strictly safer than clearing every live slot's rows.
    pub fn begin_slot(&mut self, seq: usize) -> Result<()> {
        if self.k_state_clear.is_some() {
            self.prepare_device_state_clear(seq)?;
            self.enqueue_state_clear(seq)?;
            return self.drain();
        }
        self.clear_state_serial(seq)
    }

    pub fn device_state_clear_enabled(&self) -> bool {
        self.k_state_clear.is_some()
    }

    pub fn prepare_device_state_clear(&mut self, seq: usize) -> Result<()> {
        if seq >= self.batch {
            return Err(RuntimeError::Device(format!(
                "prepare_device_state_clear {seq} past batch {}",
                self.batch
            )));
        }
        if let Some(v) = &self.vmm {
            v.begin_seq(seq);
            v.ensure_rows(seq, 1)?;
        }
        self.kda_conv_alt_stale[seq] = false;
        Ok(())
    }

    pub fn enqueue_state_clear(&self, seq: usize) -> Result<()> {
        if seq >= self.batch {
            return Err(RuntimeError::Device(format!(
                "enqueue_state_clear {seq} past batch {}",
                self.batch
            )));
        }
        let Some(k) = self.k_state_clear else {
            return Err(RuntimeError::Device(
                "device recurrent-state clear kernel was not loaded".into(),
            ));
        };
        let Some(ranges) = &self.d_state_clear else {
            return Ok(());
        };
        let arg = StateClearArgs {
            ranges: ranges.base,
            n_ranges: self.n_state_clear,
            slot: seq as u32,
        };
        EngineDevice::launch_kernel(
            &*self.be,
            k,
            self.n_state_clear,
            256,
            0,
            as_bytes(std::slice::from_ref(&arg)),
            None,
        )
    }

    fn clear_state_serial(&mut self, seq: usize) -> Result<()> {
        if seq >= self.batch {
            return Err(RuntimeError::Device(format!(
                "begin_slot {seq} past batch {}",
                self.batch
            )));
        }
        if let Some(v) = &self.vmm {
            v.begin_seq(seq);
            v.ensure_rows(seq, 1)?;
        }
        for (i, name) in self.tensor_names.iter().enumerate() {
            if !is_carried_state(name) {
                continue;
            }
            let m = &self.devp[i];
            if m.base == 0 || m.len == 0 {
                continue;
            }
            if self.batch == 1 {
                EngineDevice::memset_d8(&*self.be, m.base, 0, m.len as usize)?;
                continue;
            }
            // See the doc above: only the slot-major carried tensors can be strided.
            if name.contains("blkres") {
                continue;
            }
            let b = self.batch as u64;
            if m.len % b != 0 {
                return Err(RuntimeError::Device(format!(
                    "carried-state tensor {name} is {} bytes, not divisible by batch {b} — \
                     its slot stride is unknown, so clearing it would corrupt live slots",
                    m.len
                )));
            }
            let stride = m.len / b;
            EngineDevice::memset_d8(&*self.be, m.base + stride * seq as u64, 0, stride as usize)?;
        }
        self.kda_conv_alt_stale[seq] = false;
        Ok(())
    }

    fn sync_kda_conv_alt(&mut self, slot: usize) -> Result<()> {
        if !self.kda_conv_alt_stale.get(slot).copied().unwrap_or(false) {
            return Ok(());
        }
        self.be.memcpy_dtod_batch(&self.kda_conv_bank_pairs)?;
        self.kda_conv_alt_stale[slot] = false;
        Ok(())
    }

    /// Bytes one slot's carried recurrent state occupies — the size of a prefix snapshot.
    pub fn carried_bytes(&self) -> u64 {
        self.carried_slot.iter().map(|&(_, n)| n).sum()
    }

    /// Has slot `slot` got a prefix snapshot armed?
    pub fn has_snapshot(&self, slot: usize) -> bool {
        self.prefix_snap
            .get(slot)
            .map(|s| s.is_some())
            .unwrap_or(false)
    }

    /// Capture slot `slot`'s carried recurrent state, so a later prompt sharing the prefix that
    /// produced it can resume from here instead of re-prefilling those tokens.
    ///
    /// This is the half of prefix caching that a KV cache alone cannot provide. Reusing KV rows
    /// `[0, P)` is positional and free — identical tokens at identical positions give identical
    /// K/V. The KDA recurrence is not positional: resuming at `P` requires the STATE at `P`, and
    /// there is no way to rewind it. So the state at `P` is copied out and copied back.
    ///
    /// The snapshot is exact rather than approximate because `rebase_chunk` sets every KDA op's
    /// row count to `clen`, not to the padded bucket width — so a chunk with `clen == P` leaves
    /// the recurrence at exactly `P` and the split point needs no bucket alignment.
    pub fn snapshot_carried(&mut self, slot: usize) -> Result<()> {
        if slot >= self.batch {
            return Err(RuntimeError::Device(format!(
                "snapshot_carried {slot} past batch {}",
                self.batch
            )));
        }
        self.sync_kda_conv_alt(slot)?;
        let total = self.carried_bytes();
        if total == 0 {
            return Ok(());
        }
        if self.prefix_snap[slot].is_none() {
            self.prefix_snap[slot] = Some(EngineDevice::alloc(&*self.be, total)?);
        }
        let dst_base = self.prefix_snap[slot]
            .as_ref()
            .expect("just allocated")
            .base;
        let mut off = 0u64;
        let mut pairs = Vec::with_capacity(self.carried_slot.len());
        for &(i, stride) in &self.carried_slot {
            pairs.push((
                dst_base + off,
                self.devp[i].base + stride * slot as u64,
                stride,
            ));
            off += stride;
        }
        // ONE completion wait for all 276 tensors. Per-copy `memcpy_dtod` blocks the host on its
        // own signal, and at this count that synchronisation — not the 56 MiB — is the cost.
        let t = std::time::Instant::now();
        self.be.memcpy_dtod_batch(&pairs)?;
        crate::obs::pfx::SNAP.add(t.elapsed().as_nanos() as u64);
        Ok(())
    }

    /// Put slot `slot`'s carried state back to its snapshot. The inverse of
    /// [`AmdEngine::snapshot_carried`]; a no-op refusal if nothing is armed.
    pub fn restore_carried(&mut self, slot: usize) -> Result<()> {
        if !self.has_snapshot(slot) {
            return Err(RuntimeError::Device(format!(
                "restore_carried: slot {slot} has no snapshot"
            )));
        }
        let src_base = self.prefix_snap[slot].as_ref().expect("checked").base;
        let mut off = 0u64;
        let mut pairs = Vec::with_capacity(self.carried_slot.len());
        for &(i, stride) in &self.carried_slot {
            pairs.push((
                self.devp[i].base + stride * slot as u64,
                src_base + off,
                stride,
            ));
            off += stride;
        }
        let t = std::time::Instant::now();
        self.be.memcpy_dtod_batch(&pairs)?;
        self.kda_conv_alt_stale[slot] = false;
        crate::obs::pfx::RESTORE.add(t.elapsed().as_nanos() as u64);
        Ok(())
    }

    /// [`AmdEngine::chunk_steps`] starting at an arbitrary token offset.
    ///
    /// `from > 0` is a prefix-cache resume: the KV for `[0, from)` is already resident and this
    /// covers `[from, n_prompt)`. Nothing about the chunk itself is special — an ordinary second
    /// chunk is already in exactly this position, attending over KV it did not write.
    pub fn chunk_steps_from(
        &self,
        chunks: &[u32],
        from: u32,
        n_prompt: u32,
    ) -> Result<Vec<ChunkStep>> {
        let mut out = Vec::with_capacity(chunks.len());
        let mut c0 = from;
        for &ch in chunks {
            if c0 >= n_prompt {
                break;
            }
            let prog = (0..self.dec_lo)
                .find(|&p| self.progs[p].t == ch)
                .ok_or_else(|| {
                    RuntimeError::Device(format!("no compiled bucket for chunk T={ch}"))
                })?;
            out.push(ChunkStep {
                prog,
                c0,
                clen: (n_prompt - c0).min(ch),
            });
            c0 += ch;
        }
        Ok(out)
    }

    /// Pool counters (`blocks_live` is the HBM the KV cache actually holds).
    pub fn vmm_stats(&self) -> Option<crate::memory::vmm::VmmStats> {
        self.vmm.as_ref().map(|v| v.stats())
    }

    /// Stage the per-sequence `pos` and `kvlen` for a batched decode step.
    ///
    /// Factored out of [`Self::decode_step_batched`] because the TENSOR-PARALLEL path needs the
    /// same staging without the launch: `AmdTpGroup::submit_decode` owns
    /// zero-all-then-launch-all across ranks, so it must prepare every rank first and launch them
    /// together. Duplicating this there is how the two would drift — and the failure would be a
    /// rank feeding one sequence a stale position, which is silent.
    ///
    /// `patch_kvrow` runs only at `batch == 1`. Above it, `i[3]` is dead: `devgen` arms
    /// `i[6] = n_batch_kv` on the decode `HeadNormRope` and the kernel takes BOTH the write row
    /// and the RoPE angle from `pos[t]`, so the host must not patch a single write row it no
    /// longer owns.
    pub fn decode_prepare_batched(&mut self, pos: &[u32], kvlen: &[u32]) -> Result<()> {
        let b = self.batch;
        // The bound lives HERE, not only in `decode_step_batched`: the TP path
        // (`amd_tp::submit_decode_batched`) calls this directly, so a guard one
        // level up left tensor-parallel decode with no refusal at all — an
        // over-long `pos` walked past the KV geometry. Every other decode entry
        // (B=1 `decode_prepare`, batched, both CUDA paths) checks it.
        if let Some(&p) = pos.iter().find(|&&p| p as usize >= self.max_ctx) {
            return Err(RuntimeError::Device(format!(
                "position {p} past max_ctx {}",
                self.max_ctx
            )));
        }
        self.sync_kda_conv_alt(self.kv_slot)?;
        if b == 1 {
            self.patch_kvrow(self.decode, pos[0])?;
        }
        {
            let s = self.h_scalar.as_mut_slice();
            for (i, p) in pos.iter().enumerate() {
                s[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
            }
            for (i, k) in kvlen.iter().enumerate() {
                s[(b + i) * 4..(b + i) * 4 + 4].copy_from_slice(&k.to_le_bytes());
            }
        }
        let d_pos = self.devp[self.need(self.t_pos, "in.pos")?].base;
        let d_kvlen = self.devp[self.need(self.t_kvlen, "in.kvlen")?].base;
        self.be
            .memcpy_htod_pinned(d_pos, &self.h_scalar.as_slice()[..b * 4])?;
        self.be
            .memcpy_htod_pinned(d_kvlen, &self.h_scalar.as_slice()[b * 4..b * 8])?;
        Ok(())
    }

    /// One decode step for ALL `batch` sequences, returning each one's sampled
    /// token.
    ///
    /// `pos` and `kvlen` are per-sequence and may be RAGGED at `batch > 1`.
    ///
    /// Ragged used to be refused here on the grounds that the KV write row is
    /// one host-patched immediate (`i[3]`). That is only true of a `batch == 1`
    /// program. `devgen` arms `i[6] = n_batch_kv` on every decode `HeadNormRope`
    /// when `t > 1`, and the kernel then takes BOTH the write row and the RoPE
    /// angle from `pos[t]` — `op_norm.h`:
    ///   `obase = (t*nhead + hh) * out_stride + (pos[t] & kv_mask)`, `p = pos[t] * H2`
    /// — while `flash_decode` reads `kv_len[b]` and bases K/V/Q at
    /// `b * n_kv_head`. So every position-dependent term is already
    /// per-sequence; nothing about a common `pos` was load-bearing. `i[3]` is
    /// dead on this arm, and `patch_kvrow` is skipped rather than fed a lie.
    ///
    /// At `batch == 1` the legacy single-ring formula still applies and `i[3]`
    /// is still the write row, so that path is unchanged.
    pub fn decode_step_batched(&mut self, pos: &[u32], kvlen: &[u32]) -> Result<Vec<u32>> {
        self.decode_step_batched_at(pos, kvlen, self.decode)
    }

    /// [`Self::decode_step_batched`] on a NAMED decode rung (`decode_prog_for`).
    ///
    /// `pos`/`kvlen` still carry all `batch` slots — the tensors are sized at the widest rung
    /// and a narrow rung simply reads the prefix it advances. Rows the rung does not cover are
    /// NOT stepped on the device, which is exactly the wasted work the ladder exists to skip;
    /// the caller guarantees they hold no live sequence.
    pub fn decode_step_batched_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        dp: usize,
    ) -> Result<Vec<u32>> {
        let b = self.batch;
        if pos.len() != b || kvlen.len() != b {
            return Err(RuntimeError::Device(format!(
                "decode_step_batched wants {b} positions and {b} kvlens, got {} and {}",
                pos.len(),
                kvlen.len()
            )));
        }
        if let Some(&p) = pos.iter().find(|&&p| p as usize >= self.max_ctx) {
            return Err(RuntimeError::Device(format!(
                "position {p} past max_ctx {}",
                self.max_ctx
            )));
        }
        // Decode derives each sequence's own block; a base left rebased by a
        // prefill would funnel all B into one slot's cache.
        if self.kv_slot != 0 {
            return Err(RuntimeError::Device(format!(
                "decode with the KV base rebased onto slot {} — prefill_slot must \
                 restore it",
                self.kv_slot
            )));
        }
        // Backing must cover the row this step WRITES, for EVERY slot — not
        // just the fed ones. An idle row writes K/V at its own `pos` too.
        // `mapped_rows` is a lock-free atomic read, so the common case (nothing
        // to map) costs one load per slot.
        if self.vmm.is_some() {
            for (b, &p) in pos.iter().enumerate() {
                self.vmm_ensure(b, p + 1)?;
            }
        }

        self.decode_prepare_batched(pos, kvlen)?;

        self.run(dp, self.decode_kernel_for(dp))?;

        // Only the rung's own rows sampled a token this step. The returned vector stays
        // `batch` long — every caller indexes it BY SLOT — with the uncovered tail zeroed
        // rather than carrying a stale id that would read as a real token.
        let rows = (self.progs[dp].t as usize).min(b);
        let mut ids = self.read_sampled_batched(rows)?;
        ids.resize(b, 0);

        // Hand the pre-mapper the new frontier so the next block is mapped
        // BEFORE a step needs it. Never blocks; `vmm_ensure` above is the
        // correctness backstop if it falls behind.
        if let Some(v) = &self.vmm {
            for (i, &p) in pos.iter().enumerate() {
                v.advise(i, p + 1);
            }
        }
        Ok(ids)
    }

    /// The `b` tokens the DEVICE sampled into `in.ids`, one per sequence slot.
    ///
    /// The batched twin of [`AmdEngine::read_sampled`], and factored out of `decode_step_batched`
    /// so the TP group can read per-slot ids too: `AmdTpGroup::complete_decode` returns one id per
    /// RANK (it is an agreement check across shards), which at B>1 collapses B sequences to one.
    pub fn read_sampled_batched(&mut self, b: usize) -> Result<Vec<u32>> {
        let src = self.devp[self.need(self.t_ids, "in.ids")?].base;
        let slab = self.h_scalar.as_mut_slice();
        self.be.memcpy_dtoh_pinned(&mut slab[..b * 4], src)?;
        Ok(self.h_scalar.as_slice()[..b * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4")))
            .collect())
    }

    /// Seed `in.ids` with a starting token.
    ///
    /// Needed exactly once, before the first decode step. After that the device
    /// writes its own sampled token there and the host must NOT touch it — see
    /// the module note on why `in.ids` is absent from the per-step uploads.
    ///
    /// Decoding from position 0 with a seeded id is a genuinely self-contained
    /// forward pass: the step writes KV row 0 and attends over exactly `[0,1)`,
    /// so nothing is read that was not written. Starting mid-context without a
    /// prefill would attend over KV rows nobody ever wrote, which is how a run
    /// samples the same id every step and looks like a working decoder.
    pub fn seed_ids(&mut self, ids: &[u32]) -> Result<()> {
        let n = ids.len().min(self.h_scalar.len() / 4);
        {
            let s = self.h_scalar.as_mut_slice();
            for (i, id) in ids[..n].iter().enumerate() {
                s[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
            }
        }
        let dst = self.devp[self.need(self.t_ids, "in.ids")?].base;
        self.be
            .memcpy_htod_pinned(dst, &self.h_scalar.as_slice()[..n * 4])
    }

    /// Publish the per-row participation mask for the next decode dispatch.
    ///
    /// `parked[s] != 0` parks row `s`: its KDA recurrence and conv window are left ALONE for that
    /// dispatch. The sense is inverted deliberately — an all-zero (or never-written) mask means
    /// every row participates, so a caller that does not know about the mask cannot break the
    /// model by omitting it. `amd-bench` is exactly such a caller. Everything else about the row still runs — `t` is compiled, so the GEMVs and the
    /// KV write happen regardless — and that is fine, because those are the parts an idle or
    /// mid-prefill row can safely redo. The recurrence is the part it cannot.
    ///
    /// A blob without `in.parked` (anything not emitted at `RowKind::Sequences`) ignores this.
    pub fn upload_parked(&mut self, parked: &[u32]) -> Result<()> {
        let Some(t) = self.t_active else {
            return Ok(());
        };
        let n = parked.len().min(self.batch);
        {
            let s = self.h_scalar.as_mut_slice();
            for (i, a) in parked[..n].iter().enumerate() {
                s[i * 4..i * 4 + 4].copy_from_slice(&a.to_le_bytes());
            }
        }
        let dst = self.devp[t].base;
        self.be
            .memcpy_htod_pinned(dst, &self.h_scalar.as_slice()[..n * 4])
    }

    /// Whether model weights were bound at load. A `false` here means the
    /// timings are real and the tokens are not.
    pub fn weights_bound(&self) -> bool {
        self.weights_bound
    }

    /// The decode program's index, for callers that want [`AmdEngine::run`]. This is the
    /// WIDEST rung, which is the one whose `t` matches `in.kvlen` and therefore the only
    /// safe answer for a caller that does not know the ladder exists (`amd-bench`, the TP
    /// audit, `patch_kvrow`).
    pub fn decode_prog(&self) -> usize {
        self.decode
    }

    /// Does this blob carry a prefill bucket ladder? `false` means the prompt has to be
    /// walked through a decode program one token at a time.
    ///
    /// Was `n_programs() == 1` at the call site, which a DECODE LADDER breaks: five rungs
    /// and no prefill is five programs and still decode-only.
    pub fn has_prefill(&self) -> bool {
        self.dec_lo > 0
    }

    /// The decode rung widths, ascending. One entry without a ladder.
    pub fn decode_rungs(&self) -> Vec<u32> {
        (self.dec_lo..=self.decode)
            .map(|p| self.progs[p].t)
            .collect()
    }

    /// THE LADDER SELECTION: the program index of the NARROWEST decode rung that advances
    /// at least `rows` sequence slots.
    ///
    /// `rows` is a SLOT COUNT, not a live-request count, and the difference is the whole
    /// correctness argument: slot `s` is only advanced by a rung whose width exceeds `s`, so
    /// the caller must pass `highest_live_slot + 1`. A sequence parked in slot 5 while rung 4
    /// runs would have its position stepped on the host and never on the device.
    ///
    /// Saturates at the widest rung, so an out-of-range `rows` degrades to today's behaviour
    /// rather than refusing — the slot itself is bounded by `batch` elsewhere.
    pub fn decode_prog_for(&self, rows: usize) -> usize {
        (self.dec_lo..=self.decode)
            .find(|&p| self.progs[p].t as usize >= rows)
            .unwrap_or(self.decode)
    }

    /// The decode kernel handle.
    pub fn decode_kernel(&self) -> HsaKernel {
        self.k_decode
    }

    /// Task-13 per-rung object selection: the tier ladder from
    /// PLOW_HSACO_LOWRUNG (each entry pairing-checked at its own max) serves a
    /// rung on the NARROWEST object that fits it — the dead-lane cost of a
    /// GEMV bucket is paid per compiled MM, not per live row (r14/r15: rung 1
    /// on MM16 measured +35% TPOT over MM1). Everything wider than the last
    /// tier runs the primary object. Without tiers this is `decode_kernel()`.
    pub fn decode_kernel_for(&self, dp: usize) -> HsaKernel {
        let t = self.progs[dp].t;
        for &(max, k) in &self.decode_tiers {
            if t <= max {
                return k;
            }
        }
        self.k_decode
    }

    /// Task-9 round-7 data audit: the head of sequence slot `slot`'s region of
    /// every layer-0 KV ring buffer (via the same `kv_slot_stride` table the
    /// rebase uses, so the addressing under test is the addressing measured),
    /// plus any tensor by name via [`AmdEngine::snapshot_tensor`]. Debug
    /// instrument, only reachable under PLOW_TENS_SNAP.
    pub fn snapshot_kv_slot(
        &mut self,
        slot: usize,
        bytes: usize,
    ) -> Result<Vec<(String, Vec<u8>)>> {
        let mut out = Vec::new();
        let picks: Vec<(usize, u64)> = self
            .kv_slot_stride
            .iter()
            .filter(|&&(i, _)| self.tensor_names[i].starts_with("kv.0."))
            .copied()
            .collect();
        for (i, stride) in picks {
            let len = bytes.min(stride as usize);
            let mut buf = vec![0u8; len];
            let name = self.tensor_names[i].clone();
            EngineDevice::download(&*self.be, &self.devp[i], stride * slot as u64, &mut buf)?;
            out.push((name, buf));
        }
        Ok(out)
    }

    /// Full download of a named tensor (small act.* buffers only — caller's
    /// responsibility). Task-9 round-7 instrument.
    pub fn snapshot_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let i = self.need(self.tensor_names.iter().position(|x| x == name), name)?;
        let len = self.devp[i].len as usize;
        let mut buf = vec![0u8; len];
        EngineDevice::download(&*self.be, &self.devp[i], 0, &mut buf)?;
        Ok(buf)
    }

    /// Word 0 of every counter line of program `p`'s current bank — the task-9
    /// differential audit. A deterministic program must leave the IDENTICAL
    /// counter end-state every step; a per-tick diff on the same rung names the
    /// corrupted counter and therefore the packet. Debug instrument, off the
    /// hot path (only called under PLOW_CTR_SNAP).
    pub fn ctr_word0_snapshot(&mut self, p: usize) -> Result<Vec<u32>> {
        let g = &self.progs[p];
        let span = g.n_counter as usize * CTR_STRIDE_U32 * 4;
        let mut buf = vec![0u8; span];
        EngineDevice::download(
            &*self.be,
            &g.d_ctr,
            g.bank.current() as u64 * g.ctr_span,
            &mut buf,
        )?;
        Ok(buf
            .chunks_exact(CTR_STRIDE_U32 * 4)
            .map(|c| u32::from_le_bytes(c[..4].try_into().expect("4")))
            .collect())
    }

    /// Per-program compiled `T` (decode is 1).
    pub fn prog_t(&self, p: usize) -> u32 {
        self.progs[p].t
    }

    /// Segment count for program `p`.
    pub fn prog_segments(&self, p: usize) -> usize {
        self.progs[p].seg_class.len()
    }

    /// Whether `seg` denotes concurrently-drained L2 domains or sequential
    /// wave-class launches for program `p`.
    pub(crate) fn prog_dispatch(&self, p: usize) -> ProgramDispatch {
        let g = &self.progs[p];
        ProgramDispatch::classify(g.l2_domains, g.seg_class.len())
    }

    pub fn schedulers(&self) -> (Sched, Sched) {
        (self.sched_prefill, self.sched_decode)
    }
}

/// Reinterpret a POD slice as bytes. The blob's tables are `#[repr(C)]` mirrors
/// of the C structs, so their in-memory form IS the wire form.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is a `#[repr(C)]` POD mirror (`DevInst64`, `StreamEnt`,
    // `Wait`, `u32`) whose every bit pattern is valid, and the slice is live.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// The kernarg block is the `DevProgram` struct's own bytes — which is what the
/// kernarg-ring memcpy copies. `dev_isa.h` static-asserts the size and
/// `packet::dev_abi` pins the Rust mirror against the C header.
fn kernarg_bytes(p: &DevProgram) -> &[u8] {
    // `size_of`, NEVER a literal. This was hard-coded `128` and the struct grew
    // to 136 when `seg_ofs` was appended: the launcher then copied 128 bytes and
    // wrote the COv5 implicit block at `(args_size+7)&!7 == 128` — i.e. ON TOP of
    // `seg_ofs`, which the interpreter read as a device pointer. Every static
    // prefill died with `Memory access fault ... Reason: Unknown`, in BOTH arms of
    // the window A/B, because the implicit block's first word (grid dim) is
    // non-zero so the NULL fallback never triggered either.
    //
    // SAFETY: `DevProgram` is `repr(C)` and POD (u64/u32 only).
    unsafe {
        std::slice::from_raw_parts(
            p as *const DevProgram as *const u8,
            std::mem::size_of::<DevProgram>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segmented_prog(ops: &[DevOp], segs: &[u16]) -> DevProg {
        assert_eq!(ops.len(), segs.len());
        DevProg {
            t: 2048,
            n_counter: 0,
            insts: ops
                .iter()
                .map(|&op| DevInst64 {
                    op: op as u16,
                    ..Default::default()
                })
                .collect(),
            stream: segs
                .iter()
                .enumerate()
                .map(|(inst, &seg)| packet::dev::StreamEnt {
                    inst: inst as u32,
                    seg,
                    ..Default::default()
                })
                .collect(),
            stream_ofs: Vec::new(),
            stream_len: Vec::new(),
            waits: Vec::new(),
            succs: Vec::new(),
            gq_stream: Vec::new(),
            gq_seg_ofs: Vec::new(),
            l2_domains: 0,
        }
    }

    #[test]
    fn fp8_mla_v2_routes_only_a_pure_segment_to_four_waves() {
        let pure = segmented_prog(&[DevOp::FlashMlaPrefillFp8], &[0]);
        assert_eq!(derive_segments_for(&pure, true).unwrap(), [4]);
        assert_eq!(derive_segments_for(&pure, false).unwrap(), [8]);

        let mixed = segmented_prog(&[DevOp::FlashMlaPrefillFp8, DevOp::Gemv], &[0, 0]);
        assert_eq!(derive_segments_for(&mixed, true).unwrap(), [8]);
    }

    #[test]
    fn state_clear_ranges_cover_each_slot_stride_once() {
        let devp = vec![
            DeviceMem::view(0x1000, 3 * STATE_CLEAR_CHUNK),
            DeviceMem::view(0x20_0000, 24 * 1024),
        ];
        let ranges =
            state_clear_ranges(&devp, &[(0, 3 * STATE_CLEAR_CHUNK), (1, 24 * 1024)]).unwrap();
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0].base, 0x1000);
        assert_eq!(ranges[2].base, 0x1000 + 2 * STATE_CLEAR_CHUNK);
        assert!(ranges[..3]
            .iter()
            .all(|r| r.slot_stride == 3 * STATE_CLEAR_CHUNK));
        assert_eq!(
            ranges.iter().map(|r| r.words as u64 * 4).sum::<u64>(),
            3 * STATE_CLEAR_CHUNK + 24 * 1024
        );
    }

    #[test]
    fn tp_counter_banks_are_clean_on_first_use_and_alternate() {
        let state = CounterBankState::new();
        assert_eq!(state.current(), 0);
        assert!(state.inactive_ready());

        let mut used = Vec::new();
        for _ in 0..3 {
            assert_eq!(state.begin_tp(true), Ok(true));
            used.push(state.current());
            assert!(!state.inactive_ready());
            let executed = state.current();
            state.mark_inactive_ready();
            assert_eq!(state.current(), executed, "re-arm must not move snapshots");
        }
        assert_eq!(used, [1, 0, 1]);
    }

    #[test]
    fn tp_counter_bank_state_is_per_program() {
        let rung8 = CounterBankState::new();
        let rung32 = CounterBankState::new();

        assert_eq!(rung8.begin_tp(true), Ok(true));
        rung8.mark_inactive_ready();
        assert_eq!(rung8.current(), 1);
        assert_eq!(rung32.current(), 0);

        assert_eq!(rung32.begin_tp(true), Ok(true));
        assert_eq!(rung32.current(), 1);
        assert_eq!(rung8.current(), 1);
    }

    #[test]
    fn tp_counter_single_bank_mode_requires_synchronous_rearm() {
        let state = CounterBankState::new();
        assert_eq!(state.begin_tp(false), Ok(false));
        assert_eq!(state.current(), 0);
        assert!(state.inactive_ready());
    }

    #[test]
    fn tp_counter_bank_refuses_stale_inactive_bank() {
        let state = CounterBankState::new();
        assert_eq!(state.begin_tp(true), Ok(true));
        assert_eq!(state.begin_tp(true), Err(()));
        assert_eq!(state.current(), 1, "failed selection must not change banks");

        state.mark_inactive_ready();
        assert_eq!(state.begin_tp(true), Ok(true));
        assert_eq!(state.current(), 0);
    }

    /// The kernarg slice must cover the WHOLE struct.
    ///
    /// It was a literal `128`, and appending `seg_ofs` made the struct 136: the
    /// launcher copied 128 bytes and then wrote the COv5 implicit block at
    /// `(args_size + 7) & !7 == 128`, ON TOP of the missing field. The
    /// interpreter dereferenced the grid-dimension word as a device pointer and
    /// every static-scheduler prefill died with `Memory access fault`. Nothing
    /// caught it at compile time because the literal is still a valid length.
    #[test]
    fn the_kernarg_slice_is_the_whole_struct() {
        // SAFETY: `DevProgram` is a `#[repr(C)]` POD of integers and raw
        // pointers, so all-zeroes is a valid value (null pointers included —
        // this instance is only ever measured, never dispatched).
        let p: DevProgram = unsafe { std::mem::zeroed() };
        assert_eq!(kernarg_bytes(&p).len(), std::mem::size_of::<DevProgram>());
    }

    #[test]
    fn compact_audit_patches_only_tp_collectives() {
        let mut insts = [
            DevInst64 {
                op: DevOp::XReduce as u16,
                ..Default::default()
            },
            DevInst64 {
                op: DevOp::XArgmaxFin as u16,
                ..Default::default()
            },
            DevInst64 {
                op: DevOp::Gemm as u16,
                ..Default::default()
            },
        ];
        patch_tp_xaudit(&mut insts, 464);
        assert_eq!(insts[0].i[7], 465);
        assert_eq!(insts[1].i[7], 465);
        assert_eq!(insts[2].i[7], 0);
    }

    /// The flash object follows the PREFILL scheduler: a flash segment IS a
    /// prefill segment. Pairing it with the decode choice loads an object whose
    /// scheduling loop does not match the stream it is handed.
    #[test]
    fn object_names_match_the_shipped_set() {
        assert_eq!(
            object_name(
                Phase::Prefill,
                Variant::Bf16,
                PrefillArm::None,
                Sched::GlobalQueue
            ),
            "interp_prefill_gq.elf"
        );
        assert_eq!(
            object_name(
                Phase::Decode,
                Variant::Bf16,
                PrefillArm::None,
                Sched::Static
            ),
            "interp_decode.elf"
        );
        assert_eq!(
            object_name(
                Phase::Decode,
                Variant::Fp8Kv,
                PrefillArm::None,
                Sched::GlobalQueue
            ),
            "interp_decode_fp8kv_gq.elf"
        );
        assert_eq!(
            object_name(Phase::Decode, Variant::Fp8, PrefillArm::None, Sched::Static),
            "interp_decode_fp8.elf"
        );
        // There is no fp8-WEIGHT flash object — flash only varies on KV — so an
        // fp8 packet must fall back to the bf16 flash object rather than ask
        // for a file that was never built.
        assert_eq!(
            object_name(Phase::Flash, Variant::Fp8, PrefillArm::None, Sched::Static),
            "interp_flash.elf"
        );
        assert_eq!(
            object_name(
                Phase::Flash,
                Variant::Fp8Kv,
                PrefillArm::None,
                Sched::GlobalQueue
            ),
            "interp_flash_fp8kv_gq.elf"
        );
        // The MLA/MoE-prefill axis is PREFILL-only — no decode or flash twin
        // exists (`scripts/build_gfx950.sh` never builds one) — so a non-None
        // arm on those phases must not leak into the filename.
        assert_eq!(
            object_name(
                Phase::Prefill,
                Variant::Bf16,
                PrefillArm::Mla,
                Sched::GlobalQueue
            ),
            "interp_prefill_mla_gq.elf"
        );
        assert_eq!(
            object_name(
                Phase::Prefill,
                Variant::Bf16,
                PrefillArm::Mla,
                Sched::Static
            ),
            "interp_prefill_mla.elf"
        );
        assert_eq!(
            object_name(
                Phase::Prefill,
                Variant::Bf16,
                PrefillArm::MlaMoe,
                Sched::GlobalQueue
            ),
            "interp_prefill_mla_moe_gq.elf"
        );
        assert_eq!(
            object_name(
                Phase::Prefill,
                Variant::Bf16,
                PrefillArm::MlaMoe,
                Sched::Static
            ),
            "interp_prefill_mla_moe.elf"
        );
        assert_eq!(
            object_name(
                Phase::Decode,
                Variant::Bf16,
                PrefillArm::MlaMoe,
                Sched::Static
            ),
            "interp_decode.elf"
        );
        assert_eq!(
            object_name(
                Phase::Flash,
                Variant::Bf16,
                PrefillArm::MlaMoe,
                Sched::Static
            ),
            "interp_flash.elf"
        );
        // KIMI-K3 is the exception, and deliberately: `PLOW_K3` is a MODEL axis, not a
        // prefill-kernel one, and `interp_decode_k3.elf` is a real row in
        // `runtime/CMakeLists.txt`. A K3 decode packet handed the plain `interp_decode.elf` has no
        // `case` for AttnRes (104), SituGlu (105), MlaOutGate (106) or the KDA mixer, and this
        // interpreter's dispatch `default:` writes NOTHING.
        assert_eq!(
            object_name(Phase::Prefill, Variant::Bf16, PrefillArm::K3, Sched::Static),
            "interp_prefill_k3.elf"
        );
        assert_eq!(
            object_name(
                Phase::Prefill,
                Variant::Bf16,
                PrefillArm::K3Moe,
                Sched::Static
            ),
            "interp_prefill_k3_moe.elf"
        );
        assert_eq!(
            object_name(
                Phase::Prefill,
                Variant::Bf16,
                PrefillArm::K3MoeA4w4,
                Sched::GlobalQueue
            ),
            "interp_prefill_k3_moe_a4w4_gq.elf"
        );
        // Decode collapses ALL THREE K3 arms onto one object: the grouped-MoE ops are
        // prefill-only, so there is nothing for a `_k3_moe` decode object to contain that `_k3`
        // does not, and no such row exists in runtime/CMakeLists.txt.
        for a in [PrefillArm::K3, PrefillArm::K3Moe, PrefillArm::K3MoeA4w4] {
            assert_eq!(
                object_name(Phase::Decode, Variant::Bf16, a, Sched::Static),
                "interp_decode_k3.elf"
            );
            // No K3 flash object, and no packet can ask for one: K3 is NoPE MLA + KDA and emits
            // no `FlashPrefill` at any head dim.
            assert_eq!(
                object_name(Phase::Flash, Variant::Bf16, a, Sched::Static),
                "interp_flash.elf"
            );
        }
        assert_eq!(
            object_name(
                Phase::Decode,
                Variant::Fp8Kv,
                PrefillArm::K3,
                Sched::GlobalQueue
            ),
            "interp_decode_fp8kv_k3_gq.elf"
        );
        assert_eq!(
            object_name(
                Phase::Prefill,
                Variant::Fp8Kv,
                PrefillArm::K3MoeA4w4,
                Sched::GlobalQueue,
            ),
            "interp_prefill_fp8kv_k3_moe_a4w4_gq.elf"
        );
        assert_eq!(
            object_name(
                Phase::Decode,
                Variant::Bf16,
                PrefillArm::DSpark,
                Sched::GlobalQueue,
            ),
            "interp_decode_dspark_gq.elf"
        );
        assert_eq!(
            object_name(
                Phase::Prefill,
                Variant::Bf16,
                PrefillArm::DSpark,
                Sched::Static,
            ),
            "interp_prefill.elf"
        );
    }

    /// Synthetic packets exercising `PrefillArm::detect` — the axis whose
    /// absence let a GLM-5.2 prefill packet load `interp_prefill_gq.elf`
    /// (no MLA/MoE arms) and silently produce all-zero activations.
    #[test]
    fn prefill_arm_detect_selects_the_right_variant() {
        fn prog_with_ops(ops: &[DevOp]) -> DevProg {
            let insts = ops
                .iter()
                .map(|&op| DevInst64 {
                    op: op as u16,
                    ..Default::default()
                })
                .collect();
            DevProg {
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
                l2_domains: 0,
            }
        }

        // MoE-prefill opcodes present (even alongside MLA ones) => MlaMoe.
        let moe_progs = vec![prog_with_ops(&[
            DevOp::FlashMlaPrefill,
            DevOp::MoeRouterTopkPf,
            DevOp::MoeAlignPf,
            DevOp::MoeGroupGluPf,
            DevOp::MoeGroupDownPf,
            DevOp::MoeCombinePf,
        ])];
        assert_eq!(PrefillArm::detect(&moe_progs), PrefillArm::MlaMoe);

        // Only MLA-prefill opcodes => Mla, not MlaMoe.
        let mla_progs = vec![prog_with_ops(&[
            DevOp::FlashMlaPrefill,
            DevOp::FlashGatherPrefill,
            DevOp::MlaMergeFold,
        ])];
        assert_eq!(PrefillArm::detect(&mla_progs), PrefillArm::Mla);

        // Decode-only packet (no prefill opcodes at all) => None, unchanged.
        let decode_only = vec![prog_with_ops(&[DevOp::Embed, DevOp::Gemv, DevOp::RmsNorm])];
        assert_eq!(PrefillArm::detect(&decode_only), PrefillArm::None);

        // The opcodes live ONLY in an earlier program (a prefill bucket ahead
        // of the decode program) — detect must scan every program, not just
        // `progs.last()`, or this regresses to the exact bug being fixed.
        let bucket_then_decode = vec![
            prog_with_ops(&[
                DevOp::FlashMlaPrefill,
                DevOp::MoeRouterTopkPf,
                DevOp::MoeAlignPf,
                DevOp::MoeGroupGluPf,
                DevOp::MoeGroupDownPf,
                DevOp::MoeCombinePf,
            ]),
            prog_with_ops(&[DevOp::Embed, DevOp::Gemv, DevOp::RmsNorm]),
        ];
        assert_eq!(PrefillArm::detect(&bucket_then_decode), PrefillArm::MlaMoe);

        // KIMI-K3. The block ops SUPERSEDE both — `_hs_ax_mla_k3` composes PLOW_MLA_PREFILL with
        // PLOW_K3, because K3's full-attention layers are MLA — and without this axis a K3 blob
        // resolves to `interp_prefill_mla_moe.elf`, which has no `case` for any of them.
        //
        // The grouped GEMMs carry the expert ENCODING, and it selects an OBJECT: MXFP4 needs the
        // A4W4 body, which is compiled only under `PLOW_MOE_PF_A4W4`. Wrong in both directions —
        // an mxfp4 packet on the plain object takes `moe_pf_refuse`, a bf16 packet on the a4w4
        // object gets 140 KB of arms it never runs — so the field is read, not assumed.
        let k3_moe = |enc: u32| {
            let grouped = |op: DevOp| {
                let mut i = [0u32; 8];
                i[MOE_PF_ENC_SLOT] = enc;
                DevInst64 {
                    op: op as u16,
                    i,
                    ..Default::default()
                }
            };
            let mut first = prog_with_ops(&[
                DevOp::AttnRes,
                DevOp::SituGlu,
                DevOp::KdaStateStepG,
                DevOp::FlashMlaPrefill,
            ]);
            first.insts.push(grouped(DevOp::MoeGroupGluPf));
            first.insts.push(grouped(DevOp::MoeGroupDownPf));
            vec![
                first,
                prog_with_ops(&[DevOp::AttnRes, DevOp::SituGlu, DevOp::Gemv]),
            ]
        };
        assert_eq!(
            PrefillArm::detect(&k3_moe(2)),
            PrefillArm::K3MoeA4w4,
            "PLOW_MOE_ENC_MXFP4"
        );
        assert_eq!(
            PrefillArm::detect(&k3_moe(0)),
            PrefillArm::K3Moe,
            "bf16 experts"
        );
        assert_eq!(
            PrefillArm::detect(&k3_moe(1)),
            PrefillArm::K3Moe,
            "block-fp8 experts"
        );

        // A DECODE-ONLY K3 blob still selects the K3 objects. This is what makes
        // `interp_decode_k3` reachable at all, and it is not a corner: K3 emitted decode-only for
        // its whole bring-up, and `K3_PREFILL=0` still does.
        let k3_decode = vec![prog_with_ops(&[
            DevOp::Embed,
            DevOp::AttnRes,
            DevOp::KdaConv3,
            DevOp::MlaOutGate,
            DevOp::Gemv,
        ])];
        assert_eq!(PrefillArm::detect(&k3_decode), PrefillArm::K3);

        let mut dspark = prog_with_ops(&[
            DevOp::FlashMlaDecode,
            DevOp::MlaMergeFold,
            DevOp::MoeGroupGluPf,
            DevOp::MoeGroupDownPf,
        ]);
        dspark.insts[0].t[7] = packet::dev::TENSOR_NONE16;
        dspark.insts[0].i[6] = 7;
        assert_eq!(PrefillArm::detect(&[dspark]), PrefillArm::DSpark);

        // The MLA-only K3 bucket (attention emitted, FFN still on the decode ops) is `K3`, not
        // `K3MoeA4w4`: the grouped chain is what `PLOW_MOE_PREFILL` builds and it is absent here.
        let k3_attn = vec![prog_with_ops(&[
            DevOp::AttnRes,
            DevOp::FlashMlaPrefill,
            DevOp::MlaMergeFold,
        ])];
        assert_eq!(PrefillArm::detect(&k3_attn), PrefillArm::K3);
    }

    /// A prog carrying `ops`, each instruction asking for `m` GEMV rows.
    fn prog_gemv(ops: &[DevOp], m: u32) -> DevProg {
        let insts = ops
            .iter()
            .map(|&op| {
                let mut i = [0u32; 8];
                i[0] = m;
                DevInst64 {
                    op: op as u16,
                    i,
                    ..Default::default()
                }
            })
            .collect();
        DevProg {
            t: m,
            n_counter: 0,
            insts,
            stream: Vec::new(),
            stream_ofs: Vec::new(),
            stream_len: Vec::new(),
            waits: Vec::new(),
            succs: Vec::new(),
            gq_stream: Vec::new(),
            gq_seg_ofs: Vec::new(),
            l2_domains: 0,
        }
    }

    /// A packet dispatching a K3/KDA op against an object built without `PLOW_K3` is REFUSED.
    ///
    /// The gating of those seven arms (interp.hip) is what makes this necessary: AMD's dispatch
    /// `default:` writes nothing, so without this check the op would silently leave its output
    /// untouched and the run would finish fluently on uninitialised memory — the exact failure
    /// class `GFX950_DISPATCHED` was introduced to end.
    #[test]
    fn a_k3_packet_against_an_object_without_the_arms_is_refused() {
        let obj = Path::new("interp_decode.elf");
        let bare = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx950"];
        let with_k3 = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx950", K3_ARMS_SYM];

        for &op in K3_ARM_OPS {
            let pkt = vec![prog_gemv(&[op], 1)];
            assert_eq!(
                required_k3_op(&pkt),
                Some(op),
                "{op:?} must be recognised as a K3 arm"
            );

            let e = check_k3_arms(&bare, obj, required_k3_op(&pkt))
                .expect_err("a K3 op against an object with no K3 arms must be refused");
            let msg = e.to_string();
            // The refusal has to name the op, the marker, and the remedy — the object is not on a
            // device yet, so this message is the only thing the operator gets.
            assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
            assert!(
                msg.contains(K3_ARMS_SYM),
                "must name the missing marker: {msg}"
            );
            assert!(
                msg.contains("PLOW_K3"),
                "must name the flag to rebuild with: {msg}"
            );

            // The same packet against an object that advertises the arms is fine.
            assert!(check_k3_arms(&with_k3, obj, required_k3_op(&pkt)).is_ok());
        }

        // A packet with no K3 op at all is untouched on either object — gating must not refuse
        // the Gemma/GLM packets that are the whole reason the arms were taken out.
        let plain = vec![prog_gemv(&[DevOp::Gemv, DevOp::RmsNorm], 1)];
        assert_eq!(required_k3_op(&plain), None);
        assert!(check_k3_arms(&bare, obj, required_k3_op(&plain)).is_ok());
        assert!(check_k3_arms(&with_k3, obj, required_k3_op(&plain)).is_ok());
    }

    #[test]
    fn dspark_multi_query_packet_requires_the_noncausal_object() {
        let obj = Path::new("interp_decode_dspark.elf");
        let mut packet = vec![prog_gemv(&[DevOp::FlashMlaDecode], 1)];
        packet[0].insts[0].t[7] = packet::dev::TENSOR_NONE16;
        packet[0].insts[0].i[6] = 7;

        assert!(requires_dspark_noncausal(&packet));
        let err = check_dspark_noncausal(&[], obj, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains(DSPARK_NONCAUSAL_SYM));
        assert!(err.contains("PLOW_DSPARK_NONCAUSAL"));
        assert!(check_dspark_noncausal(&[DSPARK_NONCAUSAL_SYM], obj, true).is_ok());

        packet[0].insts[0].i[6] = 1;
        assert!(!requires_dspark_noncausal(&packet));
        packet[0].insts[0].i[6] = 7;
        packet[0].insts[0].t[7] = 0;
        assert!(
            !requires_dspark_noncausal(&packet),
            "op-50 t7 means q-rope fold, where i6 is a tensor handle"
        );
    }

    #[test]
    fn k3_causal_multi_token_packet_requires_the_verifier_object() {
        let obj = Path::new("interp_decode_fp8kv_k3_verify.elf");
        let mut packet = vec![prog_gemv(&[DevOp::FlashMlaDecodeFp8], 1)];
        packet[0].insts[0].fj[1] = 8;

        assert!(requires_k3_spec_verify(&packet));
        let err = check_k3_spec_verify(&[], obj, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains(K3_SPEC_VERIFY_SYM));
        assert!(err.contains("PLOW_K3_SPEC_VERIFY"));
        assert!(check_k3_spec_verify(&[K3_SPEC_VERIFY_SYM], obj, true).is_ok());

        packet[0].insts[0].fj[1] = 0;
        assert!(!requires_k3_spec_verify(&packet));
        packet[0].insts[0].fj[1] = 1;
        assert!(!requires_k3_spec_verify(&packet));
    }

    #[test]
    fn batched_mxfp4_moe_requires_a4w4_in_the_decode_object() {
        let obj = Path::new("interp_decode_k3.elf");
        let mut pkt = vec![prog_gemv(&[DevOp::MoeGroupGluPf, DevOp::MoeGroupDownPf], 4)];
        for i in &mut pkt[0].insts {
            i.i[MOE_PF_ENC_SLOT] = MOE_ENC_MXFP4;
        }
        let need = required_moe_pf_a4w4(&pkt);
        assert_eq!(need, Some(DevOp::MoeGroupGluPf));
        let bare = ["plow_k3_arms_1"];
        let armed = ["plow_k3_arms_1", MOE_PF_A4W4_SYM];
        let e = check_moe_pf_a4w4(&bare, obj, need).unwrap_err();
        assert!(e.to_string().contains("PLOW_MOE_PF_A4W4"));
        assert!(check_moe_pf_a4w4(&armed, obj, need).is_ok());

        pkt[0].insts[0].i[MOE_PF_ENC_SLOT] = 1;
        pkt[0].insts[1].i[MOE_PF_ENC_SLOT] = 0;
        assert_eq!(required_moe_pf_a4w4(&pkt), None);
        assert!(check_moe_pf_a4w4(&bare, obj, None).is_ok());
    }

    #[test]
    fn kda_conv_step_db_requires_exact_packet_object_pairing() {
        let obj = Path::new("interp_decode_fp8kv_k3.elf");
        let bare = [K3_ARMS_SYM];
        let armed = [K3_ARMS_SYM, KDA_CONV_STEP_DB_SYM];
        let fused = vec![prog_gemv(&[DevOp::KdaConvStateStepG], 1)];
        let legacy = vec![prog_gemv(&[DevOp::KdaConv3, DevOp::KdaStateStepG], 1)];

        assert!(check_kda_conv_step_db(
            &bare,
            obj,
            required_kda_conv_step_db(&fused),
            first_op_in(&fused, KDA_CONV_STEP_DB_REPLACED_OPS),
        )
        .is_err());
        assert!(check_kda_conv_step_db(
            &armed,
            obj,
            required_kda_conv_step_db(&fused),
            first_op_in(&fused, KDA_CONV_STEP_DB_REPLACED_OPS),
        )
        .is_ok());
        assert!(check_kda_conv_step_db(
            &armed,
            obj,
            required_kda_conv_step_db(&legacy),
            first_op_in(&legacy, KDA_CONV_STEP_DB_REPLACED_OPS),
        )
        .is_err());
        assert!(check_kda_conv_step_db(
            &bare,
            obj,
            required_kda_conv_step_db(&legacy),
            first_op_in(&legacy, KDA_CONV_STEP_DB_REPLACED_OPS),
        )
        .is_ok());
    }

    /// A packet dispatching a Gemma-4 MoE op against an object built without the matching axis is
    /// REFUSED — the same argument as the K3 gate, and for a family that until the AMD port had
    /// NO arm at all, so its silent-NOP failure was not hypothetical. [GEMMA4-MOE-AMD]
    #[test]
    fn a_gemma_moe_packet_against_an_object_without_the_arms_is_refused() {
        let obj = Path::new("interp_decode.elf");
        let bare = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx942"];
        let with_dec = vec![
            "plow_gemv_mm_cap_1",
            "plow_interp_dec_gfx942",
            MOE_GEMMA_SYM,
        ];
        let with_pf = vec!["plow_gemv_mm_cap_1", "plow_interp_gfx942", MOE_GEMMA_PF_SYM];

        for (set, sym, flag, good) in [
            (MOE_GEMMA_OPS, MOE_GEMMA_SYM, "PLOW_MOE_GEMMA", &with_dec),
            (
                MOE_GEMMA_PF_OPS,
                MOE_GEMMA_PF_SYM,
                "PLOW_MOE_GEMMA_PF",
                &with_pf,
            ),
        ] {
            for &op in set {
                let pkt = vec![prog_gemv(&[op], 1)];
                let need_dec = first_op_in(&pkt, MOE_GEMMA_OPS);
                let need_pf = first_op_in(&pkt, MOE_GEMMA_PF_OPS);
                assert!(
                    need_dec == Some(op) || need_pf == Some(op),
                    "{op:?} must be recognised as a Gemma MoE arm"
                );
                let e = check_moe_gemma_arms(&bare, obj, need_dec, need_pf)
                    .expect_err("a Gemma MoE op against a bare object must be refused");
                let msg = e.to_string();
                assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
                assert!(msg.contains(sym), "must name the missing marker: {msg}");
                assert!(msg.contains(flag), "must name the flag: {msg}");
                assert!(check_moe_gemma_arms(good, obj, need_dec, need_pf).is_ok());
            }
        }

        // A packet with no Gemma MoE op is untouched — the gate must not refuse every other model.
        let plain = vec![prog_gemv(&[DevOp::Gemv, DevOp::RmsNorm], 1)];
        assert_eq!(first_op_in(&plain, MOE_GEMMA_OPS), None);
        assert_eq!(first_op_in(&plain, MOE_GEMMA_PF_OPS), None);
        assert!(check_moe_gemma_arms(&bare, obj, None, None).is_ok());
    }

    /// The two Gemma-MoE opcode lists must match the `#if PLOW_MOE_GEMMA` / `#if
    /// PLOW_MOE_GEMMA_PF` regions of `interp.hip` — PARSED, not restated, for the reason
    /// `k3_arm_ops_match_the_interpreter` gives: a twentieth arm added inside the guard and not
    /// listed here would be dispatched by an object that does not advertise the axis, with no
    /// refusal and no fault.
    #[test]
    fn gemma_moe_arm_ops_match_the_interpreter() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("runtime/amd/interp.hip"));
        let Some(path) = path.filter(|p| p.exists()) else {
            eprintln!("interp.hip not found — skipping (source checkout only)");
            return;
        };
        let src = std::fs::read_to_string(&path).unwrap();
        // Walk the guard nesting and collect `case PLOW_DOP_*` inside each Gemma region.
        let mut dec: Vec<String> = Vec::new();
        let mut pf: Vec<String> = Vec::new();
        let mut stack: Vec<&'static str> = Vec::new();
        let mut depth: Vec<bool> = Vec::new(); // is this #if one of ours?
        for line in src.lines() {
            let t = line.trim();
            if t.starts_with("#if") {
                let which = if t.contains("PLOW_MOE_GEMMA_PF") {
                    Some("pf")
                } else if t.contains("PLOW_MOE_GEMMA") {
                    Some("dec")
                } else {
                    None
                };
                depth.push(which.is_some());
                if let Some(w) = which {
                    stack.push(w);
                }
                continue;
            }
            if t.starts_with("#endif") {
                if depth.pop().unwrap_or(false) {
                    stack.pop();
                }
                continue;
            }
            let Some(r) = t.strip_prefix("case PLOW_DOP_") else {
                continue;
            };
            let name: String = r
                .chars()
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            match stack.last() {
                Some(&"dec") => dec.push(format!("PLOW_DOP_{name}")),
                Some(&"pf") => pf.push(format!("PLOW_DOP_{name}")),
                _ => {}
            }
        }
        for (found, listed, what) in [
            (&dec, MOE_GEMMA_OPS, "PLOW_MOE_GEMMA"),
            (&pf, MOE_GEMMA_PF_OPS, "PLOW_MOE_GEMMA_PF"),
        ] {
            let mut want: Vec<String> = listed.iter().map(|o| o.c_name().to_string()).collect();
            want.sort();
            let mut got = found.clone();
            got.sort();
            got.dedup();
            assert_eq!(
                got, want,
                "{what}: interp.hip's guarded `case` labels disagree with the Rust list"
            );
        }
    }

    /// The KV-encoding SWAP is refused in BOTH directions, which is what makes it different from
    /// the K3 gate.
    ///
    /// The bf16 direction is the one that had no check at all and is not hypothetical: the K3 MLA
    /// gate's bf16 packet run against the fp8 object reports "all packets executed on every
    /// slice: YES" and scores rel 1.000e+00 at the attention output.
    #[test]
    fn a_kv_encoding_mismatch_is_refused_in_both_directions() {
        let obj = Path::new("interp_decode.elf");
        let bare = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx950"];
        let with_fp8 = vec!["plow_gemv_mm_cap_1", "plow_interp_dec_gfx950", FP8_KV_SYM];

        let check = |syms: &[&str], pkt: &[DevProg]| {
            check_kv_encoding(
                syms,
                obj,
                required_kv_op(pkt, FP8_KV_OPS),
                required_kv_op(pkt, BF16_KV_OPS),
            )
        };

        for &op in FP8_KV_OPS {
            let pkt = vec![prog_gemv(&[op], 1)];
            let e = check(&bare, &pkt)
                .expect_err("an fp8-KV op against a bf16-KV object must be refused");
            let msg = e.to_string();
            assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
            assert!(msg.contains(FP8_KV_SYM), "must name the marker: {msg}");
            assert!(msg.contains("PLOW_FP8_KV"), "must name the flag: {msg}");
            assert!(
                check(&with_fp8, &pkt).is_ok(),
                "{op:?} belongs on the fp8 object"
            );
        }
        for &op in BF16_KV_OPS {
            let pkt = vec![prog_gemv(&[op], 1)];
            let e = check(&with_fp8, &pkt)
                .expect_err("a bf16-KV op against an fp8-KV object must be refused");
            let msg = e.to_string();
            assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
            assert!(
                check(&bare, &pkt).is_ok(),
                "{op:?} belongs on the bf16 object"
            );
        }

        // The ops that are in BOTH objects must be refused by NEITHER. `HeadNormRope` is the one
        // that matters: an fp8-KV packet still uses it for the QUERY norm, so listing it as bf16
        // would refuse every fp8 packet ever emitted. The gathered MLA ops keep their bf16 arm in
        // both objects for want of a free tensor slot.
        for &op in &[
            DevOp::HeadNormRope,
            DevOp::FlashGatherDecode,
            DevOp::FlashGatherPrefill,
            DevOp::Gemv,
            DevOp::MlaMergeFold,
        ] {
            let pkt = vec![prog_gemv(&[op], 1)];
            assert!(
                check(&bare, &pkt).is_ok(),
                "{op:?} must not be refused on a bf16 object"
            );
            assert!(
                check(&with_fp8, &pkt).is_ok(),
                "{op:?} must not be refused on an fp8 object"
            );
        }
    }

    /// [`K3_ARM_OPS`] is exactly the set of `case` labels inside the `#if PLOW_K3` region of
    /// `runtime/amd/interp.hip`.
    ///
    /// The two halves of this contract sit in different languages in different files, and the
    /// consequence of them drifting is asymmetric: an arm added inside the guard but missing from
    /// this list is an op that silently NOPs on a non-K3 object with no refusal, which is the bug
    /// the guard was supposed to make impossible. Read out of the file rather than restated, by
    /// the same discipline `dispatched_list_matches_the_amd_interpreter` applies in devgen.
    #[test]
    fn k3_arm_ops_match_the_interpreter() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("runtime/amd/interp.hip"));
        let Some(path) = path.filter(|p| p.exists()) else {
            eprintln!("interp.hip not found — skipping (source checkout only)");
            return;
        };
        let src = std::fs::read_to_string(&path).unwrap();

        // The LAST `#if PLOW_K3` is the dispatch region; the earlier ones guard the includes and
        // the marker. Scan to its matching `#endif`.
        let mut in_region = false;
        let mut nested = 0usize;
        let mut found: Vec<String> = Vec::new();
        for line in src.lines() {
            let t = line.trim();
            if t == "#if PLOW_K3" {
                in_region = true;
                nested = 0;
                found.clear(); // a later region supersedes an earlier one
                continue;
            }
            if in_region && t.starts_with("#if") {
                nested += 1;
                continue;
            }
            if in_region && t.starts_with("#endif") {
                if nested == 0 {
                    in_region = false;
                } else {
                    nested -= 1;
                }
                continue;
            }
            if in_region {
                if let Some(r) = t.strip_prefix("case PLOW_DOP_") {
                    let n: String = r
                        .chars()
                        .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                        .collect();
                    found.push(format!("PLOW_DOP_{n}"));
                }
            }
        }
        found.sort();

        let mut want: Vec<String> = K3_ARM_OPS.iter().map(|o| o.c_name().to_string()).collect();
        want.sort();

        assert_eq!(
            found, want,
            "K3_ARM_OPS disagrees with the `#if PLOW_K3` region of interp.hip.\n  interp has: \
             {found:?}\n  K3_ARM_OPS has: {want:?}\nAn arm inside the guard but missing from the \
             list NOPs silently on an object built without PLOW_K3, with no load-time refusal."
        );
    }

    /// THE 14th SILENT-CORRUPTION BUG, CONSTRUCTED. A packet whose GEMVs ask for
    /// more rows than the object was compiled for must be REFUSED, not clamped.
    ///
    /// Every expectation is DERIVED from the constants, never written as a
    /// literal: the object's advertised capacity is built by concatenating onto
    /// [`GEMV_CAP_SYM_PREFIX`], and the packet's demand is that capacity `+ 1`.
    /// A sibling test that spelled its bound out inverted its own meaning when
    /// the bound moved underneath it; move `PLOW_GEMV_MM`'s default or the
    /// marker's spelling and this test follows rather than lying.
    #[test]
    fn a_gemv_wider_than_the_objects_bucket_is_refused() {
        let obj = Path::new("interp_decode.elf");
        for cap in [1u32, 2, 4, 8, 16] {
            let marked = format!("{GEMV_CAP_SYM_PREFIX}{cap}");
            let syms = vec![marked.as_str(), "plow_interp_dec_gfx950"];

            // Exactly at the bucket: accepted, for every op that reads it.
            let fits = vec![prog_gemv(GEMV_BUCKET_OPS, cap)];
            assert_eq!(required_gemv_m(&fits), cap);
            assert!(check_gemv_capacity(&syms, obj, required_gemv_m(&fits)).is_ok());

            // One row past it: refused. `gemv_rows<MM>` would write rows
            // 0..cap and leave the last one holding whatever was there.
            let over = vec![prog_gemv(GEMV_BUCKET_OPS, cap + 1)];
            assert_eq!(required_gemv_m(&over), cap + 1);
            let e = check_gemv_capacity(&syms, obj, required_gemv_m(&over))
                .expect_err("an M past the object's bucket must be refused");
            let msg = e.to_string();
            // The refusal must name all three: what the packet needs, what the
            // object has, and how to rebuild.
            assert!(msg.contains(&format!("M={} rows", cap + 1)), "{msg}");
            assert!(msg.contains(&format!("PLOW_GEMV_MM={cap}")), "{msg}");
            // Past `PLOW_GEMV_MAXM` there is nothing to rebuild, and the
            // remedy has to say so instead of naming an impossible bucket.
            if cap + 1 > GEMV_MAXM {
                assert!(msg.contains("No object can serve this"), "{msg}");
                assert!(
                    msg.contains(&format!("PLOW_DECODE_BATCH <= {GEMV_MAXM}")),
                    "{msg}"
                );
            } else {
                assert!(
                    msg.contains(&format!("PLOW_DECODE_BATCH={}", cap + 1)),
                    "{msg}"
                );
            }
        }
    }

    /// An object that does not advertise a bucket is refused above M=1 and
    /// accepted at M=1.
    ///
    /// Silence is not consent: every object built before the marker compiled at
    /// `op_gemm.h`'s default of 1, which is the bug. But MM >= 1 always, so an
    /// M=1 packet — the whole batch-1 world, including every TP asset, which
    /// `load` refuses to batch at all — must stay loadable against the objects
    /// already on disk.
    #[test]
    fn an_object_that_advertises_nothing_is_refused_only_above_one_row() {
        let obj = Path::new("interp_decode.elf");
        let bare = vec!["plow_interp_dec_gfx950", "__hip_cuid_3ec2926410ebcf30"];
        assert_eq!(object_gemv_cap(&bare), None);

        let one = vec![prog_gemv(GEMV_BUCKET_OPS, 1)];
        assert_eq!(required_gemv_m(&one), 1);
        assert!(check_gemv_capacity(&bare, obj, required_gemv_m(&one)).is_ok());
        // A packet with no GEMV at all is 0, which is likewise never refused.
        assert!(check_gemv_capacity(&bare, obj, 0).is_ok());

        let two = vec![prog_gemv(&[DevOp::GemvQkv], 2)];
        let e = check_gemv_capacity(&bare, obj, required_gemv_m(&two))
            .expect_err("an unmarked object must not silently serve M>1");
        let msg = e.to_string();
        assert!(msg.contains(GEMV_CAP_SYM_PREFIX), "{msg}");
        assert!(msg.contains("PLOW_DECODE_BATCH=2"), "{msg}");
    }

    /// `required_gemv_m` reads the INSTRUCTIONS, and only the ops that actually
    /// reach a `<PLOW_GEMV_MM>` instantiation.
    ///
    /// The MoE expert arms live in `op_moe.h` and do their own row handling, so
    /// counting them would refuse packets the bucket cannot hurt — the mirror
    /// image of the bug, and just as wrong.
    #[test]
    fn only_the_bucketed_ops_set_the_requirement() {
        // A wide MoE expert GEMV next to a one-row bucketed GEMV: still 1.
        let mixed = vec![prog_gemv(&[DevOp::MoeExpertGlu, DevOp::MoeExpertDown], 64)];
        assert_eq!(required_gemv_m(&mixed), 0);

        let mut p = prog_gemv(&[DevOp::MoeExpertGlu], 64);
        p.insts.extend(prog_gemv(&[DevOp::Gemv], 1).insts);
        assert_eq!(required_gemv_m(std::slice::from_ref(&p)), 1);

        // The maximum is taken over EVERY program and every instruction, not
        // the first one found.
        let progs = vec![
            prog_gemv(&[DevOp::Gemv], 1),
            prog_gemv(&[DevOp::GemvGlu], 8),
        ];
        assert_eq!(required_gemv_m(&progs), 8);
    }

    /// A REAL object from the shipped tree carries no bucket marker, and the
    /// parser says so rather than guessing.
    ///
    /// `build-amd/hsaco-abi144/interp_decode.elf` is the current ABI-144 decode
    /// object, built before this marker existed and — like every gfx950 decode
    /// object built without `PLOW_DECODE_BATCH` — compiled at the `op_gemm.h`
    /// default of 1. It is the exact input that produced the bug. Skipped when
    /// the tree is not on this machine.
    #[test]
    fn a_shipped_object_without_the_marker_reads_as_unknown() {
        let p = Path::new("/home/lava/plow/build-amd/hsaco-abi144/interp_decode.elf");
        let Ok(img) = std::fs::read(p) else { return };
        let syms = elf_symbol_names(&img);
        // The reader works on this file (sanity: the interpreter body is there),
        // so `None` below means "no marker", never "no symbol table".
        assert!(syms.iter().any(|s| s.contains("plow_exec")), "{syms:?}");
        assert_eq!(object_gemv_cap(&syms), None);
        assert!(check_gemv_capacity(&syms, p, 1).is_ok());
        assert!(check_gemv_capacity(&syms, p, 2).is_err());
    }

    /// The marker is a contract between `op_gemm.h` and this file, written in
    /// two languages. Read the C side rather than restating it.
    ///
    /// The concatenation `plow_gemv_mm_cap_##n` cannot be grepped for its
    /// expanded form, so this asserts on the token the macro pastes onto — which
    /// is what [`GEMV_CAP_SYM_PREFIX`] has to match — and on the fact that the
    /// value pasted is `PLOW_GEMV_MM` itself. If the symbol were named for
    /// anything else it could disagree with what was compiled, which is the
    /// entire failure this check exists to end.
    #[test]
    fn op_gemm_h_emits_the_capacity_marker() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/amd/op_gemm.h");
        let src = std::fs::read_to_string(&p).expect("runtime/amd/op_gemm.h");
        assert!(
            src.contains(&format!("{GEMV_CAP_SYM_PREFIX}##n")),
            "op_gemm.h no longer pastes onto `{GEMV_CAP_SYM_PREFIX}` — the loader's \
             capacity check would silently stop finding any object's bucket"
        );
        assert!(
            src.contains("PLOW_GEMV_CAP_SYM(PLOW_GEMV_MM) = PLOW_GEMV_MM"),
            "the capacity marker must be named for, and hold, PLOW_GEMV_MM itself"
        );
        // The ceiling too. [`GEMV_MAXM`] decides both the `load` refusal and
        // which remedy `check_gemv_capacity` prints, and a stale copy of it
        // would send someone to build a bucket the header's static assert
        // rejects.
        assert!(
            src.contains(&format!("#define PLOW_GEMV_MAXM {GEMV_MAXM}")),
            "op_gemm.h's PLOW_GEMV_MAXM is no longer {GEMV_MAXM}"
        );
        // The walk marker is the OTHER half of the same contract, and it fails
        // more dangerously than the capacity one: if op_gemm.h stops emitting it,
        // a walking object silently becomes a hard-capacity object to the loader
        // and every M > MM packet is REFUSED (loud, recoverable). If the loader
        // stops looking for it, an MM=8 object serving M=16 is accepted with no
        // walk and rows 8..15 come back STALE (silent, fluent, wrong).
        assert!(
            src.contains(&format!("unsigned {GEMV_WALK_SYM} = 1")),
            "op_gemm.h no longer emits `{GEMV_WALK_SYM}` under PLOW_GEMV_WALK — \
             `check_gemv_capacity` would refuse every walking object it should serve"
        );
        assert!(
            src.contains("#if PLOW_GEMV_WALK"),
            "the walk marker must be gated on PLOW_GEMV_WALK itself, or a \
             non-walking object would advertise a capacity it does not have"
        );
    }

    /// The walk marker turns the capacity check off, and nothing else does.
    ///
    /// Guards both directions of the failure above: a walking MM=8 object must
    /// serve M=16, and a NON-walking one must still refuse it. The second half
    /// is the silent-corruption case (§6g-BATCH slots 13/14/15), so it is the
    /// one worth a test.
    #[test]
    fn the_walk_marker_lifts_the_capacity_refusal() {
        let p = Path::new("/tmp/interp_decode.elf");
        let hard = ["plow_gemv_mm_cap_8"];
        let walk = ["plow_gemv_mm_cap_8", GEMV_WALK_SYM];
        assert!(check_gemv_capacity(&hard, p, 8).is_ok(), "MM=8 covers M=8");
        assert!(
            check_gemv_capacity(&hard, p, 16).is_err(),
            "a NON-walking MM=8 object must still refuse M=16 — rows 8..15 would be stale"
        );
        assert!(
            check_gemv_capacity(&walk, p, 16).is_ok(),
            "a walking MM=8 object serves M=16 in two row blocks"
        );
        // An unmarked object is still refused, walk or no walk: silence is not consent.
        assert!(check_gemv_capacity(&[], p, 2).is_err());
    }

    #[test]
    fn symbols_carry_the_isa_and_the_scheduler() {
        assert_eq!(
            symbol_name(Phase::Decode, Sched::Static, "gfx950"),
            "plow_interp_dec_gfx950"
        );
        assert_eq!(
            symbol_name(Phase::Prefill, Sched::GlobalQueue, "gfx950"),
            "plow_interp_gfx950_gq"
        );
        assert_eq!(
            symbol_name(Phase::Flash, Sched::GlobalQueue, "gfx950"),
            "plow_interp_flash_gfx950_gq"
        );
    }

    /// The ladder is mixed, not repeated: covering 1536 with 1024+512 beats two
    /// 1024s, which would compute 512 padded rows at full cost.
    #[test]
    fn chunk_plan_mixes_the_ladder_and_puts_the_ragged_chunk_last() {
        let bkt = [128, 512, 1024];
        // `plan_chunks_cfg(.., false)`, not `plan_chunks`: this test is about the
        // PADDED DP, and `plan_chunks` reads the process-wide default — which is
        // now ragged, so the wrapper would silently stop exercising the DP.
        let dp = |n| plan_chunks_cfg(&bkt, n, LAUNCH_ROWS, false).unwrap();
        assert_eq!(dp(1536), vec![1024, 512]);
        assert_eq!(dp(1024), vec![1024]);
        assert_eq!(dp(1), vec![128]);
        assert_eq!(dp(0), Vec::<u32>::new());

        // Descending, so the ragged chunk is the LAST one — padding lands in
        // the tail, where a padded row writes KV that `n_kv` bounds out.
        for n in [200u32, 700, 1300, 4000, 9000] {
            let plan = dp(n);
            assert!(
                plan.windows(2).all(|w| w[0] >= w[1]),
                "plan for {n} is not largest-first: {plan:?}"
            );
            let covered: u32 = plan.iter().sum();
            assert!(covered >= n, "plan for {n} covers only {covered}: {plan:?}");
        }
    }

    /// A blob with no prefill bucket at all is a decode-only blob, not a silent
    /// zero-chunk prefill.
    ///
    /// And the cap is the PACKET's, not a constant: a 16384 rung is USED, not
    /// filtered. This is the whole of the `MAX_CHUNK = 16384` change — the runtime
    /// used to hold its own 8192 and quietly serve a wider blob as if the rung
    /// were absent.
    #[test]
    fn the_ladder_is_its_own_cap() {
        const B16: &[u32] = &[128, 512, 1024, 2048, 4096, 8192, 16384];
        const B8: &[u32] = &[128, 512, 1024, 2048, 4096, 8192];
        let ragged16 = |n| plan_chunks_cfg(B16, n, LAUNCH_ROWS, true).unwrap();

        assert!(plan_chunks(&[], 10).is_err());
        // A 16384-rung packet plans ON the 16384 rung.
        assert_eq!(ragged16(16384), vec![16384]);
        assert_eq!(ragged16(8193), vec![16384]);
        // ...and the SAME prompt on an 8192-ladder packet still takes two chunks,
        // so the cap follows the blob rather than the binary.
        let ragged8 = |n| plan_chunks_cfg(B8, n, LAUNCH_ROWS, true).unwrap();
        assert_eq!(ragged8(8193), vec![8192, 128]);
        // Under the PADDED DP the wide rung is correctly declined below its width
        // — 8191 rows of dead compute cost more than a second launch — which is
        // why the rung is worth nothing without ragged-M.
        let padded16 = |n| plan_chunks_cfg(B16, n, LAUNCH_ROWS, false).unwrap();
        assert_eq!(padded16(8193), vec![8192, 128]);
    }

    /// One contiguous h2d, not `n_kvrow` scattered ones: the sites straddle
    /// almost the whole instruction array (Gemma-31B: [4,664] of 676), and
    /// submission overhead dominates bytes.
    #[test]
    fn kvrow_span_covers_every_site() {
        assert_eq!(kvrow_span(&[4, 300, 664, 12]), Some((4, 664)));
        assert_eq!(kvrow_span(&[7]), Some((7, 7)));
        assert_eq!(kvrow_span(&[]), None);
    }

    /// A GLM-5.2 MLA prefill chunk: the two KV-write sites move, the ordinary
    /// norms and the query rope do NOT, and the flash keeps every operand it
    /// was emitted with.
    ///
    /// This is the shape the old `HeadNormRope && fj[1] != 0` test got wrong.
    /// MLA's k_rope sets `j[1] = KV_MASK_NONE`, which packs into `fj[2]`, and
    /// leaves `f[1]`/`j[0]` — the two halves of `fj[1]` — at zero, so the test
    /// matched NOTHING on an MLA packet: every chunk's latent and rope rows were
    /// written at row 0, with no error anywhere.
    #[test]
    fn rebase_chunk_moves_only_the_kv_write_rows() {
        let names: Vec<String> = ["kv.0.ckv", "act.xn", "kv.0.krot", "act.qr", "act.opart"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let inst = |op: DevOp, dst: u16| DevInst64 {
            op: op as u16,
            t: [dst, 0, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        };
        // 0 kv_a_layernorm -> kv.0.ckv (row in i[2]); 1 the input norm, same
        // opcode, i[2] means nothing there; 2 k_rope -> kv.0.krot (row in i[3]);
        // 3 the QUERY rope, same opcode, must be left alone; 4 the flash.
        let mut insts = vec![
            inst(DevOp::RmsNorm, 0),
            inst(DevOp::RmsNorm, 1),
            inst(DevOp::HeadNormRope, 2),
            inst(DevOp::HeadNormRope, 3),
            inst(DevOp::FlashMlaPrefill, 4),
        ];
        insts[4].i = [1, 64, 8192, 0, 128, u32::MAX, 0, 7];
        let before = insts.clone();

        rebase_chunk_rows(&mut insts, &names, 512, 128, None);

        assert_eq!(insts[0].i[2], 512, "kv.0.ckv out_row0 was not rebased");
        assert_eq!(insts[2].i[3], 512, "kv.0.krot out_row was not rebased");
        // Everything else, field for field.
        assert_eq!(insts[0].i[..2], before[0].i[..2]);
        assert_eq!(insts[0].i[3..], before[0].i[3..]);
        assert_eq!(insts[1].i, before[1].i, "the ordinary RmsNorm moved");
        assert_eq!(insts[2].i[..3], before[2].i[..3]);
        assert_eq!(insts[2].i[4..], before[2].i[4..]);
        assert_eq!(insts[3].i, before[3].i, "the QUERY rope moved");
        // FlashMlaPrefill takes its query base from `in.kvlen`, not from an
        // immediate; every `i[]` here is a live operand and none may be touched.
        assert_eq!(insts[4].i, before[4].i, "FlashMlaPrefill was patched");
    }

    /// The RAGGED-M row shrink rewrites the row count of every family that
    /// carries one, RESCALES the element-count families, and leaves alone both
    /// the lm_head (whose `M` is 1, not `T`) and a row-BANDED GEMM (whose `M` is
    /// `T/kb`). The last two are the whole reason the shrink is guarded on the
    /// field already equalling the bucket width.
    #[test]
    fn ragged_rows_shrink_only_the_fields_that_hold_the_bucket_width() {
        const T: u32 = 8192;
        const CLEN: u32 = 4097;
        const H: u32 = 6144;
        let names: Vec<String> = ["act.xn", "act.logits"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut inst = |op: DevOp, i: [u32; 8]| DevInst64 {
            op: op as u16,
            t: [0, 0, 0, 0, 0, 0, 0, 0],
            i,
            ..Default::default()
        };
        let mut insts = vec![
            inst(DevOp::Embed, [T, H, 0, 0, 0, 0, 0, 0]),
            inst(DevOp::RmsNorm, [T, H, 0, 0, 0, 0, 0, 0]),
            inst(DevOp::GemmSmall, [T, 2048, H, 0, 0, 0, 0, 0]),
            inst(DevOp::FlashMlaPrefill, [1, 8, 73728, 0, T, u32::MAX, 2, 7]),
            inst(DevOp::MlaMergeFold, [T, 8, 256, 0, 2, 0, 0, 0]),
            inst(DevOp::XReduceTwoShot, [T * H, 8, 0, 0, 1, 0, 0, 0]),
            inst(DevOp::Residual, [T * H, 0, 0, 0, 0, 0, 0, 0]),
            inst(DevOp::MoeRouterTopkPf, [0, 256, 8, 0, T, 0, 1, 1]),
            inst(DevOp::MoeAlignPf, [T, 256, 8, 0, 0, 0, 0, 0]),
            inst(DevOp::MoeCombinePf, [H, 8, T, 0, 0, 0, 0, 0]),
            // The lm_head: M = 1 over `a_row0`, NOT a row count.
            inst(DevOp::Gemv, [1, 154880, H, 0, T - 1, 0, 0, 0]),
            // A PLOW_GLM_XR_BAND row-band GEMM: M = T/2 at a_row0 = T/2.
            inst(DevOp::Gemm, [T / 2, H, 2048, 0, T / 2, T / 2, 0, 0]),
        ];
        let before = insts.clone();

        rebase_chunk_rows(&mut insts, &names, 0, CLEN, Some(T));

        assert_eq!(insts[0].i[0], CLEN, "Embed ntok");
        assert_eq!(insts[0].i[1], H, "Embed hidden must not move");
        assert_eq!(insts[1].i[0], CLEN, "RmsNorm rows");
        assert_eq!(insts[2].i[0], CLEN, "GEMM M");
        assert_eq!(insts[2].i[1..3], before[2].i[1..3], "GEMM N/K moved");
        assert_eq!(insts[3].i[4], CLEN, "flash n_tok");
        assert_eq!(insts[3].i[..4], before[3].i[..4], "flash operands moved");
        assert_eq!(insts[4].i[0], CLEN, "merge token count");
        assert_eq!(insts[5].i[0], CLEN * H, "two-shot element count");
        assert_eq!(insts[6].i[0], CLEN * H, "residual element count");
        assert_eq!(insts[7].i[4], CLEN, "router T");
        assert_eq!(insts[8].i[0], CLEN, "align T");
        assert_eq!(insts[9].i[2], CLEN, "combine T");
        assert_eq!(insts[10].i, before[10].i, "the lm_head GEMV was rewritten");
        assert_eq!(insts[11].i, before[11].i, "a banded GEMM was rewritten");
    }

    /// A FULL last chunk (`clen == T`) must leave every instruction alone — the
    /// shrink is a no-op on an exactly-covered prompt, which is what keeps
    /// 1024/4096/8192/16384 byte-identical to the padded path.
    #[test]
    fn ragged_rows_are_a_no_op_when_the_chunk_is_full() {
        const T: u32 = 4096;
        let names: Vec<String> = ["act.xn"].iter().map(|s| s.to_string()).collect();
        let mut insts = vec![
            DevInst64 {
                op: DevOp::RmsNorm as u16,
                i: [T, 6144, 0, 0, 0, 0, 0, 0],
                ..Default::default()
            },
            DevInst64 {
                op: DevOp::XReduceTwoShot as u16,
                i: [T * 6144, 8, 0, 0, 1, 0, 0, 0],
                ..Default::default()
            },
        ];
        let before = insts.clone();
        rebase_chunk_rows(&mut insts, &names, 0, T, Some(T));
        assert_eq!(insts[0].i, before[0].i);
        assert_eq!(insts[1].i, before[1].i);
    }

    /// The RAGGED cover is the MINIMUM number of launches, and the tail rung is
    /// the smallest one that HOLDS the remainder — not the cheapest padded cover
    /// of it, because under the row shrink the padding is free.
    ///
    /// The paired padded plans are the shipped DP's answers, so this test is also
    /// the record of which lengths the axis changes and which it does not.
    #[test]
    fn the_ragged_cover_takes_the_fewest_launches() {
        const B: &[u32] = &[128, 512, 1024, 2048, 4096, 8192];
        let ragged = |n| plan_chunks_cfg(B, n, LAUNCH_ROWS, true).unwrap();
        let padded = |n| plan_chunks_cfg(B, n, LAUNCH_ROWS, false).unwrap();

        // Exactly on a rung, or an exact multiple of the widest one: IDENTICAL.
        for n in [128u32, 1024, 4096, 8192, 16384, 24576] {
            assert_eq!(
                ragged(n),
                padded(n),
                "the cover moved at an exact length {n}"
            );
        }
        // One token over a rung: one launch instead of two.
        assert_eq!(padded(1025), vec![1024, 128]);
        assert_eq!(ragged(1025), vec![2048]);
        assert_eq!(padded(4097), vec![4096, 128]);
        assert_eq!(ragged(4097), vec![8192]);
        // Past the widest rung the second launch is STRUCTURAL (no bucket is
        // wider than MAX_CHUNK), so the plan is the same and only the tail's row
        // count shrinks.
        assert_eq!(padded(8193), vec![8192, 128]);
        assert_eq!(ragged(8193), vec![8192, 128]);
        // A deeply ragged length: three launches become two.
        assert_eq!(ragged(12345), vec![8192, 8192]);
        assert!(
            padded(12345).len() > 2,
            "expected the DP to add a tail chunk"
        );
        // Every ragged cover is the arithmetic minimum, and covers the prompt.
        for n in [1u32, 127, 129, 1025, 4097, 8193, 12345, 16385, 65536, 73728] {
            let c = ragged(n);
            assert_eq!(
                c.len(),
                n.div_ceil(8192) as usize,
                "not the fewest launches at {n}"
            );
            assert!(c.iter().sum::<u32>() >= n, "cover short at {n}");
        }
    }

    /// The dense-GQA rules are unchanged, and the two KV tests are a UNION: a
    /// `kv.*` destination and a non-zero `fj[1]` both mark the same site.
    #[test]
    fn rebase_chunk_still_patches_the_dense_gqa_families() {
        let names: Vec<String> = ["kv.0.k", "act.q"].iter().map(|s| s.to_string()).collect();
        let mut insts = vec![
            // k norm: `kv.*` AND j[0] = ring stride, both tests fire, one field.
            DevInst64 {
                op: DevOp::HeadNormRopeFp8 as u16,
                t: [0; 8],
                fj: [0, 4096, 0],
                ..Default::default()
            },
            // q norm: neither test fires.
            DevInst64 {
                op: DevOp::HeadNormRope as u16,
                t: [1, 0, 0, 0, 0, 0, 0, 0],
                ..Default::default()
            },
            DevInst64 {
                op: DevOp::FlashPrefillFp8 as u16,
                t: [1, 0, 0, 0, 0, 0, 0, 0],
                ..Default::default()
            },
        ];
        rebase_chunk_rows(&mut insts, &names, 1024, 512, None);
        assert_eq!(insts[0].i[3], 1024);
        assert_eq!(insts[1].i, [0; 8], "the query norm was patched");
        assert_eq!(insts[2].i[4], 1024, "q_pos0");
        assert_eq!(insts[2].i[1], 1536, "n_kv is everything written so far");
    }

    /// EVERY KDA OP'S ROW COUNT BECOMES `clen`, AND THE BUG IS WHAT THIS ASSERTS.
    ///
    /// The fixture is the shape that was actually broken: a 1500-token prompt on a
    /// [1024, 512] ladder, whose last chunk is 476 real rows padded out to 512. The
    /// pre-fix `rebase_chunk` had no KDA arm at all, so `i[0]` kept the compiler's
    /// baked 512 and both stateful arms ran 36 pad rows past the prompt — the conv
    /// rolling zeros through a 4-wide window (evicting every real tap) and the
    /// recurrence applying 36 extra `exp(a_log)` decays to the state decode then
    /// starts from.
    ///
    /// Asserted as `!= t` rather than only `== clen` so that a future change which
    /// re-bakes `T` somewhere else still fails here: the property is "the KDA row
    /// count is the REAL row count", not "this field happens to hold 476".
    #[test]
    fn rebase_chunk_shortens_every_kda_arm_to_the_real_row_count() {
        let names: Vec<String> = ["kv.0.state", "act.q"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        const T: u32 = 512;
        const CLEN: u32 = 476;
        let kda = |op: DevOp| DevInst64 {
            op: op as u16,
            i: [T, 96, 128, 0, 0, 0, 0, 0],
            ..Default::default()
        };
        let mut insts: Vec<DevInst64> = KDA_ROW_COUNT_OPS.iter().map(|&o| kda(o)).collect();
        // A non-KDA neighbour that must NOT be touched by the new arm.
        insts.push(DevInst64 {
            op: DevOp::RmsNorm as u16,
            i: [T, 0, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        });
        rebase_chunk_rows(&mut insts, &names, 1024, CLEN, None);

        for (d, &op) in insts.iter().zip(KDA_ROW_COUNT_OPS) {
            assert_eq!(d.i[0], CLEN, "{op:?} still runs the padded bucket width");
            assert_ne!(d.i[0], T, "{op:?} kept the compiler's baked T");
            // The rest of the immediates are live operands (heads, head_dim) and
            // patching one of them is the positional bug this module keeps hitting.
            assert_eq!((d.i[1], d.i[2]), (96, 128), "{op:?}: a live operand moved");
        }
        assert_eq!(insts.last().unwrap().i[0], T, "a non-KDA op was shortened");
    }

    /// THE ATTNRES SCORE WEIGHT IS DERIVED, AND THE FOLD IS CHECKED AGAINST THE
    /// REAL CHECKPOINT RATHER THAN A FIXTURE.
    ///
    /// `models--moonshotai--Kimi-K3` ships `*_res_norm.weight` [7168] bf16 and
    /// `*_res_proj.weight` [1, 7168] bf16, 93 of each at both the attention and the
    /// MLP site; the packet declares one f32 [7168] per site. Without the fold all
    /// 186 resolve to MISSING WEIGHT and no real-weight K3 run can start.
    ///
    /// Skipped when the checkpoint is not on this machine — the same convention
    /// `prefill_object_without_mla_arms_is_refused` uses for its fixture.
    #[test]
    fn the_attn_res_score_weight_folds_from_the_pair_the_checkpoint_ships() {
        let Some(dir) = k3_snapshot_dir() else { return };
        let Ok(c) = crate::asset::checkpoint::Checkpoint::open(&dir) else {
            return;
        };
        const H: usize = 7168;
        for site in ["self_attention", "mlp"] {
            let base = format!("language_model.model.layers.1.{site}");
            let out = fold_res_score(&c, &format!("{base}_res_score.weight"))
                .expect("name matches the derived pattern")
                .expect("both sources present in the checkpoint");
            assert_eq!(out.len(), H * 4, "{site}: f32 [hidden]");

            // Recompute element 0 and a middle element from the two sources, so
            // this pins the RELATION and not merely the length.
            let bf = |n: &str, i: usize| -> f32 {
                let (raw, _) = c.tensor_ex(n).expect("source tensor");
                let b = &raw[i * 2..i * 2 + 2];
                f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)
            };
            for i in [0usize, H / 2, H - 1] {
                let want = bf(&format!("{base}_res_norm.weight"), i)
                    * bf(&format!("{base}_res_proj.weight"), i);
                let got = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
                assert_eq!(got, want, "{site}[{i}] is not norm * proj");
            }
        }
        // A name that is not a score weight must not be intercepted at all —
        // otherwise every ordinary weight would take the derived path.
        assert!(fold_res_score(&c, "language_model.lm_head.weight").is_none());
    }

    /// The K3 snapshot directory, or `None` on a machine without it.
    fn k3_snapshot_dir() -> Option<std::path::PathBuf> {
        let root = std::path::Path::new(
            "/home/lava/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots",
        );
        std::fs::read_dir(root)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("model.safetensors.index.json").exists())
    }

    /// The two questions about a carried-state tensor must have ONE answer.
    ///
    /// `kv_skips_zeroing` (may LOAD skip the memset?) and `begin_slot` (what must a
    /// NEW SEQUENCE clear?) are the same set for the same reason — these tensors are
    /// read before they are written. They were two separate pieces of knowledge and
    /// only the first existed, which is how the state came to be zeroed exactly once
    /// per process and never again between requests.
    #[test]
    fn carried_state_is_never_skippable_and_the_kv_cache_always_is() {
        for n in [
            "kv.0.state",
            "kv.12.conv_state.q",
            "kv.12.conv_state_alt.q",
            "kv.blkres",
            "kv.3.state.v",
        ] {
            assert!(is_carried_state(n), "{n} is carried state");
            assert!(
                !kv_skips_zeroing(n),
                "{n} would keep stale bytes across a load"
            );
        }
        // The append-only cache: skippable at load, and nothing for begin_slot to do.
        for n in ["kv.0.k", "kv.31.v", "kv.7.latent"] {
            assert!(!is_carried_state(n), "{n} is append-only, not carried");
            assert!(kv_skips_zeroing(n), "{n} lost the 11.5 GiB memset skip");
        }
        // Outside the namespace entirely: neither question applies.
        for n in ["act.x", "model.layers.0.mlp.down_proj.weight", "in.pos"] {
            assert!(!is_carried_state(n));
            assert!(!kv_skips_zeroing(n), "{n} is not a kv. tensor");
        }
    }

    /// The negative fixture is real: `/home/lava/models/glm52_objs` is a GLM-5.2
    /// object set whose prefill object was built WITHOUT `PLOW_MLA_PREFILL`, and
    /// pairing it with a GLM packet is exactly the silent-garbage run this check
    /// exists to refuse. Skipped when the fixture is not on this machine.
    #[test]
    fn prefill_object_without_mla_arms_is_refused() {
        let p = Path::new("/home/lava/models/glm52_objs/interp_prefill.elf");
        let Ok(img) = std::fs::read(p) else { return };
        let syms = elf_symbol_names(&img);
        // The reader works on this file at all (it is where the rule was
        // derived): the interpreter body and the norms are there.
        assert!(syms.iter().any(|s| s.contains("plow_exec")), "{syms:?}");
        assert!(syms.iter().any(|s| s.contains("d_rmsnorm")), "{syms:?}");

        let e = check_prefill_object(&syms, p, &["PLOW_MLA_PREFILL=1".into()])
            .expect_err("an object with no MLA-prefill symbol must be refused");
        let msg = e.to_string();
        assert!(msg.contains("PLOW_MLA_PREFILL"), "{msg}");
        assert!(msg.contains("interp_prefill.elf"), "{msg}");
        assert!(check_prefill_object(&syms, p, &["PLOW_MOE_PREFILL=1".into()]).is_err());
        // `=0` and the filename-selected flags are not refusals.
        assert!(check_prefill_object(
            &syms,
            p,
            &["PLOW_BUCKET_DECODE=0".into(), "PLOW_FP8=1".into()]
        )
        .is_ok());
    }

    /// A file that is not an ELF64 yields no symbols — and therefore no
    /// refusal. A parser miss must never be reported as a missing arm.
    #[test]
    fn elf_reader_is_bounded_on_junk() {
        assert!(elf_symbol_names(b"").is_empty());
        assert!(elf_symbol_names(b"\x7fELF\x02\x01").is_empty());
        assert!(elf_symbol_names(&[0xffu8; 4096]).is_empty());
        let junk = elf_symbol_names(&[0u8; 128]);
        assert!(
            check_prefill_object(&junk, Path::new("x.elf"), &["PLOW_MLA_PREFILL=1".into()]).is_ok()
        );
        // The GEMV capacity check reads the SAME empty list and reaches the
        // opposite conclusion, on purpose: an arm it cannot see may simply be
        // one this packet does not need, but a bucket it cannot see is a bucket
        // that is almost certainly the default 1.
        assert!(check_gemv_capacity(&junk, Path::new("x.elf"), 1).is_ok());
        assert!(check_gemv_capacity(&junk, Path::new("x.elf"), 2).is_err());
    }

    /// The load-time zeroing skip is about WRITE-BEFORE-READ, not about the
    /// `kv.` prefix. K3 names two read-modify-write things `kv.` so the loader
    /// does not demand them of the checkpoint, and they must still be zeroed:
    /// uninitialised HBM in a recurrence is garbage that never washes out, and
    /// it neither faults nor reports a missing weight.
    #[test]
    fn only_append_only_kv_caches_skip_zeroing() {
        // Append-only caches: written before read, so the skip is sound.
        for n in ["kv.0.k", "kv.0.v", "kv.3.ckv", "kv.3.krot", "kv.7.kidx"] {
            assert!(kv_skips_zeroing(n), "`{n}` is an append-only cache");
        }
        // Read-modify-write state, and the AttnRes snapshot ring.
        for n in [
            "kv.2.state",
            "kv.2.conv_state.q",
            "kv.2.conv_state.k",
            "kv.2.conv_state.v",
            "kv.blkres",
        ] {
            assert!(!kv_skips_zeroing(n), "`{n}` is READ before it is written");
        }
        // Everything outside the namespace was always zeroed and still is.
        for n in ["act.x", "in.ids", "moe.expert_weight_table"] {
            assert!(!kv_skips_zeroing(n));
        }
    }
}

/// The routed-expert NAME RESOLUTION, against synthetic checkpoints.
///
/// Three shipping spellings reach [`resolve_expert_names`], and the wrong answer
/// is SILENT: picking the block-fp8 arm for an mxfp4 checkpoint reads an E8M0
/// row as an f32 grid and gets a plausible count of plausible bytes. So each
/// spelling is pinned as the bytes a checkpoint would actually have — the tensor
/// names, the dtypes, and the shapes, all three taken from the real artifacts.
#[cfg(test)]
mod expert_name_tests {
    use super::*;
    use crate::asset::checkpoint::Checkpoint;

    /// A one-shard safetensors directory holding `(name, dtype, shape)` with
    /// zeroed data — the resolver reads names, dtypes and shapes, never payload.
    struct Fake(std::path::PathBuf);

    impl Fake {
        fn new(tag: &str, tensors: &[(&str, &str, &[usize])]) -> Fake {
            let dir =
                std::env::temp_dir().join(format!("plowrt-expert-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let width = |dt: &str| match dt {
                "F32" => 4,
                "U8" | "F8_E4M3" => 1,
                other => panic!("test helper has no width for {other}"),
            };
            let (mut hdr, mut data, mut off) = (String::from("{"), Vec::new(), 0usize);
            for (i, (n, dt, sh)) in tensors.iter().enumerate() {
                let len = sh.iter().product::<usize>() * width(dt);
                if i > 0 {
                    hdr.push(',');
                }
                hdr.push_str(&format!(
                    "{n:?}:{{\"dtype\":\"{dt}\",\"shape\":{sh:?},\"data_offsets\":[{off},{}]}}",
                    off + len
                ));
                off += len;
                data.resize(off, 0u8);
            }
            hdr.push('}');
            let mut blob = (hdr.len() as u64).to_le_bytes().to_vec();
            blob.extend_from_slice(hdr.as_bytes());
            blob.extend_from_slice(&data);
            std::fs::write(dir.join("model.safetensors"), blob).unwrap();
            Fake(dir)
        }

        fn open(&self) -> Checkpoint {
            Checkpoint::open(&self.0).unwrap()
        }
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn refs<'a>(
        t: &'a [(String, &'static str, Vec<usize>)],
    ) -> Vec<(&'a str, &'a str, &'a [usize])> {
        t.iter()
            .map(|(n, d, s)| (n.as_str(), *d, s.as_slice()))
            .collect()
    }

    /// GLM-5.2's layout, and the ONE case that must stay bit-for-bit what it
    /// was: the first candidate probed is the pair of names this resolver
    /// replaced hardcoded, so a block-fp8 checkpoint never reaches a second
    /// lookup and every name built downstream is the name it was built before.
    #[test]
    fn block_fp8_resolves_to_exactly_the_names_it_always_did() {
        // zai-org/GLM-5.2-FP8, scaled down: [N, K] fp8 + [N/128, K/128] f32.
        let mut t: Vec<(String, &str, Vec<usize>)> = Vec::new();
        for p in ["gate_proj", "up_proj", "down_proj"] {
            t.push((
                format!("model.layers.3.mlp.experts.0.{p}.weight"),
                "F8_E4M3",
                vec![256, 384],
            ));
            t.push((
                format!("model.layers.3.mlp.experts.0.{p}.weight_scale_inv"),
                "F32",
                vec![2, 3],
            ));
        }
        let f = Fake::new("blockfp8", &refs(&t));
        let c = f.open();
        let en = resolve_expert_names(&c, "model.layers.3.mlp.").unwrap();
        assert_eq!(en.ns, "model.layers.3.mlp.experts.");
        assert_eq!(en.proj, ["gate_proj", "up_proj", "down_proj"]);
        assert_eq!(en.payload, ".weight");
        assert_eq!(en.scale, ".weight_scale_inv");
        assert!(!en.microscaled());
        assert_eq!(
            en.weight_of(0, 0),
            "model.layers.3.mlp.experts.0.gate_proj.weight"
        );
        assert_eq!(
            en.scale_of(0, 2),
            "model.layers.3.mlp.experts.0.down_proj.weight_scale_inv"
        );
        check_expert_geometry(&c, &en).unwrap();
    }

    /// amd/Kimi-K2.7-Code-MXFP4: the STANDARD projection names with an E8M0
    /// scale. This is why the layout cannot be one boolean — "mxfp4" and
    /// "Mixtral-spelled" are independent facts, and this checkpoint has the
    /// first without the second.
    #[test]
    fn mxfp4_under_the_standard_projection_names_resolves_on_the_scale_alone() {
        let mut t: Vec<(String, &str, Vec<usize>)> = Vec::new();
        for p in ["gate_proj", "up_proj", "down_proj"] {
            // [N, K/2] packed + [N, K/32] E8M0, K = 96.
            t.push((
                format!("m.layers.3.mlp.experts.0.{p}.weight"),
                "U8",
                vec![64, 48],
            ));
            t.push((
                format!("m.layers.3.mlp.experts.0.{p}.weight_scale"),
                "U8",
                vec![64, 3],
            ));
        }
        let f = Fake::new("mxstd", &refs(&t));
        let c = f.open();
        let en = resolve_expert_names(&c, "m.layers.3.mlp.").unwrap();
        assert_eq!(en.proj, ["gate_proj", "up_proj", "down_proj"]);
        assert_eq!(en.payload, ".weight");
        assert_eq!(en.scale, ".weight_scale");
        assert!(en.microscaled());
        check_expert_geometry(&c, &en).unwrap();
    }

    /// Kimi-K3: `block_sparse_moe.experts.{e}.w1|w2|w3` + `weight_packed` /
    /// `weight_scale`, reached from a table declared under the compiler's own
    /// `moe.` namespace. Three things are discovered at once — the namespace,
    /// the projection names, and the payload suffix — and the slot order is
    /// gate, up, down, so `w3` must come back SECOND and `w2` LAST.
    #[test]
    fn mixtral_mxfp4_resolves_namespace_projections_and_payload_together() {
        const P: &str = "language_model.model.layers.1.block_sparse_moe.experts.0.";
        let t: Vec<(String, &str, Vec<usize>)> = vec![
            (format!("{P}w1.weight_packed"), "U8", vec![64, 48]),
            (format!("{P}w1.weight_scale"), "U8", vec![64, 3]),
            (format!("{P}w3.weight_packed"), "U8", vec![64, 48]),
            (format!("{P}w3.weight_scale"), "U8", vec![64, 3]),
            (format!("{P}w2.weight_packed"), "U8", vec![96, 32]),
            (format!("{P}w2.weight_scale"), "U8", vec![96, 2]),
        ];
        let f = Fake::new("k3", &refs(&t));
        let c = f.open();
        // The K3 emitter declares `moe.{lp}expert_weight_table` (devgen/k3.rs),
        // so this is the prefix the loader is actually handed.
        let en = resolve_expert_names(&c, "moe.language_model.model.layers.1.").unwrap();
        assert_eq!(en.ns, P.trim_end_matches("0."));
        assert_eq!(en.proj, ["w1", "w3", "w2"], "slot order is gate, up, DOWN");
        assert_eq!(en.payload, ".weight_packed");
        assert_eq!(en.scale, ".weight_scale");
        assert_eq!(en.weight_of(0, 2), format!("{P}w2.weight_packed"));
        check_expert_geometry(&c, &en).unwrap();
    }

    /// A scale that is the wrong size for its weight must be refused BY NAME.
    /// Nothing downstream would notice: the bytes are all u8, the shape is
    /// plausible, the packed buffer comes out the declared size, and every group
    /// after the first is scaled by the wrong exponent.
    #[test]
    fn a_scale_that_covers_the_wrong_k_is_refused_and_named() {
        const P: &str = "m.layers.0.mlp.experts.0.";
        let t: Vec<(String, &str, Vec<usize>)> = vec![
            (format!("{P}gate_proj.weight"), "U8", vec![64, 48]),
            // 4 groups of 32 = 128 elements, but the payload packs 96.
            (format!("{P}gate_proj.weight_scale"), "U8", vec![64, 4]),
            (format!("{P}up_proj.weight"), "U8", vec![64, 48]),
            (format!("{P}up_proj.weight_scale"), "U8", vec![64, 3]),
            (format!("{P}down_proj.weight"), "U8", vec![48, 32]),
            (format!("{P}down_proj.weight_scale"), "U8", vec![48, 2]),
        ];
        let f = Fake::new("badk", &refs(&t));
        let c = f.open();
        let en = resolve_expert_names(&c, "m.layers.0.mlp.").unwrap();
        let e = check_expert_geometry(&c, &en).unwrap_err().to_string();
        assert!(e.contains("gate_proj.weight_scale"), "{e}");
        assert!(e.contains("K disagrees"), "{e}");
    }

    /// The same for a block-fp8 grid — the arm GLM-5.2 ships on, so it is pinned
    /// that the check accepts the real geometry and rejects a near miss.
    #[test]
    fn a_block_fp8_grid_of_the_wrong_shape_is_refused() {
        const P: &str = "m.layers.0.mlp.experts.0.";
        for (tag, grid, ok) in [("good", vec![2usize, 3], true), ("bad", vec![3, 2], false)] {
            let t: Vec<(String, &str, Vec<usize>)> = ["gate_proj", "up_proj", "down_proj"]
                .iter()
                .flat_map(|p| {
                    [
                        (format!("{P}{p}.weight"), "F8_E4M3", vec![256, 384]),
                        (format!("{P}{p}.weight_scale_inv"), "F32", grid.clone()),
                    ]
                })
                .collect();
            let f = Fake::new(&format!("grid{tag}"), &refs(&t));
            let c = f.open();
            let en = resolve_expert_names(&c, "m.layers.0.mlp.").unwrap();
            assert_eq!(check_expert_geometry(&c, &en).is_ok(), ok, "{tag}");
        }
    }

    /// A payload with NO scale under either spelling is a broken checkpoint, not
    /// a spelling this loader has yet to learn — say so, and name both.
    #[test]
    fn a_payload_without_any_scale_names_both_spellings() {
        let t: Vec<(String, &str, Vec<usize>)> = vec![(
            "m.layers.0.mlp.experts.0.gate_proj.weight".into(),
            "U8",
            vec![64, 48],
        )];
        let f = Fake::new("noscale", &refs(&t));
        let e = resolve_expert_names(&f.open(), "m.layers.0.mlp.")
            .unwrap_err()
            .to_string();
        assert!(e.contains("MISSING EXPERT SCALE"), "{e}");
        assert!(
            e.contains("weight_scale_inv") && e.contains("weight_scale"),
            "{e}"
        );
    }

    /// No routed experts under ANY spelling fails loudly with what was probed.
    /// A zero-filled expert buffer is read by the kernel as real weights, so the
    /// alternative to this error is a model that loads and is nonsense.
    #[test]
    fn no_experts_at_all_reports_every_name_it_probed() {
        let t: Vec<(String, &str, Vec<usize>)> =
            vec![("m.layers.0.mlp.gate.weight".into(), "U8", vec![8, 8])];
        let f = Fake::new("noexp", &refs(&t));
        let e = resolve_expert_names(&f.open(), "m.layers.0.mlp.")
            .unwrap_err()
            .to_string();
        assert!(e.contains("MISSING EXPERT WEIGHT"), "{e}");
        assert!(e.contains("experts.0.gate_proj.weight"), "{e}");
        assert!(
            e.contains("block_sparse_moe.experts.0.w1.weight_packed"),
            "{e}"
        );
    }
}

#[cfg(test)]
mod slab_tests {
    use super::*;

    #[test]
    fn pad_rounds_up_and_leaves_exact_multiples_alone() {
        assert_eq!(slab_pad(0), 0);
        assert_eq!(slab_pad(1), SLAB_ALIGN);
        assert_eq!(slab_pad(SLAB_ALIGN - 1), SLAB_ALIGN);
        assert_eq!(slab_pad(SLAB_ALIGN), SLAB_ALIGN);
        assert_eq!(slab_pad(SLAB_ALIGN + 1), 2 * SLAB_ALIGN);
    }

    /// The property the carve depends on: sizing the allocation by summing
    /// `slab_pad` and advancing a cursor by `slab_pad` over the same list must
    /// agree, and no tensor may extend past the total. An overshoot would alias
    /// two tensors onto the same bytes — silently wrong weights rather than a
    /// crash, which is why the loader asserts it rather than trusting it.
    #[test]
    fn carve_cursor_lands_exactly_on_the_sized_total() {
        // Sub-stride, exact-stride, stride+1, zero, and a real expert
        // projection (1.4 MiB) — the size ROCr rounds worst.
        let sizes = [
            1u64,
            SLAB_ALIGN - 1,
            SLAB_ALIGN,
            SLAB_ALIGN + 1,
            0,
            1_468_006,
            1 << 20,
        ];
        let total: u64 = sizes.iter().copied().map(slab_pad).sum();

        let mut off = 0u64;
        for s in sizes {
            assert!(off + s <= total, "tensor at {off} (+{s}) runs past {total}");
            off += slab_pad(s);
        }
        assert_eq!(off, total, "cursor must consume exactly the sized span");
    }

    /// The loader carves `bytes.max(1)`, never `bytes`: a zero-byte tensor still
    /// needs an address of its own, and a zero-length carve would hand the next
    /// tensor the same one. This pins that the `.max(1)` is load-bearing.
    #[test]
    fn zero_byte_tensors_still_advance_the_cursor() {
        let need = |b: u64| b.max(1);
        let sizes = [0u64, 0, 0];
        let total: u64 = sizes.iter().copied().map(|b| slab_pad(need(b))).sum();
        assert_eq!(total, 3 * SLAB_ALIGN);

        let mut off = 0u64;
        let mut seen = Vec::new();
        for s in sizes {
            seen.push(off);
            off += slab_pad(need(s));
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 3, "every tensor must get a distinct address");
    }
}
