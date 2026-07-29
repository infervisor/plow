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

use std::path::Path;
use std::sync::Arc;

use packet::dev::{DevInst64, DevOp, DevProgram};
use packet::devbuild::static_seg_ofs;

use crate::asset::devblob::{DevBlob, DevProg};
use crate::device::hsa::{HsaBackend, HsaKernel, HsaPinned};
use crate::device::{DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::memory::vmm::{VmmGeometry, VmmKv, VmmOps};
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

/// Wave-class 8 is `PLOW_WG_WAVES` = 8 waves of 64.
const WG_THREADS_8: u32 = 8 * 64;
/// The flash object is built 4-wave. Dispatching it at 512 threads is an
/// `INVALID_ISA`, not a slowdown.
const WG_THREADS_4: u32 = 4 * 64;

/// The largest `seg` id the reference driver's `seg_class[512]` could hold.
/// Kept as a hard bound so a corrupt stream cannot make the host allocate
/// unboundedly off a `u16`.
const MAX_SEG: u32 = 512;

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
                if i.op == DevOp::FlashDecodeFp8 as u16 {
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
}

impl PrefillArm {
    fn infix(self) -> &'static str {
        match self {
            PrefillArm::None => "",
            PrefillArm::Mla => "_mla",
            PrefillArm::MlaMoe => "_mla_moe",
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
        for p in progs {
            for i in &p.insts {
                let op = i.op;
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
                }
            }
        }
        if moe {
            PrefillArm::MlaMoe
        } else if mla {
            PrefillArm::Mla
        } else {
            PrefillArm::None
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
    let arm = if phase == Phase::Prefill { arm } else { PrefillArm::None };
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
const PREFILL_ARM_MARKERS: &[(&str, &[&str])] = &[
    // `#if PLOW_MLA_PREFILL` in runtime/amd/interp.hip gates ops 51/55 (via
    // `exec_flash_mla_prefill` -> `d_flash_mla_decode`) AND the latent epilogue
    // ops 53/54, which is why the fold names count as proof of the same flag.
    ("PLOW_MLA_PREFILL", &["d_flash_mla", "d_mla_merge_fold", "d_o_uv_fold"]),
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
];

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
    let man: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| {
        RuntimeError::Device(format!("{}: not valid JSON: {e}", mpath.display()))
    })?;
    Ok(man
        .pointer("/backends/gfx950/requires")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect()))
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
        img.get(o..o + 2).map(|b| u16::from_le_bytes(b.try_into().expect("2")) as usize)
    };
    let u32at = |o: usize| -> Option<usize> {
        img.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().expect("4")) as usize)
    };
    let u64at = |o: usize| -> Option<usize> {
        img.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().expect("8")) as usize)
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
        let (Some(stroff), Some(strsz)) = (u64at(l + 24), u64at(l + 32)) else { continue };
        let Some(strtab) = img.get(stroff..stroff.saturating_add(strsz)) else { continue };
        for k in 0..size / entsz {
            let Some(nm) = u32at(off + k * entsz) else { break };
            let Some(tail) = strtab.get(nm..) else { continue };
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

/// Every opcode that reaches an arm behind `PLOW_K3` in `runtime/amd/interp.hip`.
///
/// The four KDA mixer ops and the three K3 block-structure ops, and nothing else. This is the
/// Rust half of the contract whose C half is the `#if PLOW_K3` region around those seven `case`
/// labels; `k3_arm_ops_match_the_interpreter` reads `interp.hip` and asserts the two agree, so a
/// future eighth arm added inside the guard cannot go unlisted here.
const K3_ARM_OPS: &[DevOp] = &[
    DevOp::KdaConv,
    DevOp::KdaGate,
    DevOp::KdaStateStep,
    DevOp::KdaGatedNorm,
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

/// Per-segment wave class, derived from the stream.
///
/// A segment is class 4 (the flash interpreter, 256 threads) iff ANY stream
/// entry in it points at a flash-prefill instruction; everything else is class
/// 8. The fp8 twin is included here and is NOT in the reference — omitting it
/// silently ran fp8-KV flash segments on the 8-wave object.
pub fn derive_segments(prog: &DevProg) -> Result<Vec<u8>> {
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
        }
    }
    Ok(class)
}

/// Largest prefill chunk the sliding-window KV ring can serve
/// (`PLOW_MAX_CHUNK`): the ring needs `RING >= window + MAX_CHUNK - 1`.
const MAX_CHUNK: u32 = 8192;

/// Rows-equivalent cost charged per launch in the chunk DP. Tuned in the
/// reference; `PLOW_LAUNCH_ROWS` overrides. It is what stops the DP from
/// choosing a hundred tiny chunks that each pay a full dispatch.
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
pub fn plan_chunks(buckets: &[u32], n_prompt: u32) -> Result<Vec<u32>> {
    let mut bkt: Vec<u32> = buckets.iter().copied().filter(|&b| b > 0 && b <= MAX_CHUNK).collect();
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
    let quant = bkt[0];
    let launch_rows = std::env::var("PLOW_LAUNCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(LAUNCH_ROWS);
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
        // [N/128][K/128] f32 `weight_scale_inv` grids.
        let (pfx, suffix) = match td
            .name
            .strip_suffix("dense_weight_table")
            .map(|p| (p, ".weight"))
            .or_else(|| {
                td.name
                    .strip_suffix("dense_scale_table")
                    .map(|p| (p, ".weight_scale_inv"))
            }) {
            Some(v) => v,
            None => continue,
        };
        let mut addrs = [0u64; 3];
        for (j, proj) in PROJ.iter().enumerate() {
            let want = format!("{pfx}{proj}{suffix}");
            let k = names.iter().position(|n| *n == want).ok_or_else(|| {
                RuntimeError::Device(format!(
                    "dense-FFN prefill table `{}` needs `{want}`, which the packet does not \
                     declare. The table and the three projections are emitted together by \
                     declare_glm_rows; a packet with one and not the other is malformed.",
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
#[allow(clippy::too_many_arguments)]
fn bind_packed_experts(
    be: &HsaBackend,
    blob: &DevBlob,
    ckpt: &crate::asset::checkpoint::Checkpoint,
    devp: &[DeviceMem],
    names: &[String],
    stage: &mut HsaPinned,
    stage_bytes: usize,
    rank: u32,
    n_gpu: u32,
) -> Result<(Vec<DeviceMem>, u64)> {
    const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
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
    // block-fp8, and the grouped fp8 collapse — writes `act.fu`, reads the
    // layer's `expert_weight_table` from `t[3]`, and carries `I_moe` in `i[1]`
    // (`crates/devgen/src/mla.rs`). Matching on `t[0] == act.fu` is what
    // separates it from the `down` arm, which shares `t[3]` and writes
    // `act.part`.
    let t_fu = names
        .iter()
        .position(|x| x == "act.fu")
        .ok_or_else(|| {
            RuntimeError::Device(
                "packet declares expert tables but no `act.fu` — nothing says how \
                 wide an expert this rank must bind"
                    .into(),
            )
        })? as u16;
    let dec = blob.progs.last().expect("checked non-empty");
    let i_moe_of = |i_ewt: usize| -> Option<u64> {
        dec.insts
            .iter()
            .find(|d| d.t[3] as usize == i_ewt && d.t[0] == t_fu)
            .map(|d| d.i[1] as u64)
    };

    let mut bufs = Vec::with_capacity(layers.len() * 2);
    let mut i_moe = 0u64;
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
        // Geometry from expert 0; every expert in a layer is the same shape.
        let probe = format!("{pfx}experts.0.gate_proj.weight");
        let (w0, shape0) = ckpt.tensor_ex(&probe).ok_or_else(|| {
            RuntimeError::Device(format!("MISSING EXPERT WEIGHT: {probe}"))
        })?;
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
        let s_probe = format!("{pfx}experts.0.gate_proj.weight_scale_inv");
        let s_stride = ckpt
            .tensor_ex(&s_probe)
            .ok_or_else(|| RuntimeError::Device(format!("MISSING EXPERT SCALE: {s_probe}")))?
            .0
            .len() as u64
            / if whole { 1 } else { n_gpu as u64 };

        let d_w = EngineDevice::alloc(be, (n_local * 3 * w_stride).max(1))?;
        let d_s = EngineDevice::alloc(be, (n_local * 3 * s_stride).max(1))?;
        let wtab = crate::orch::moe::packed_expert_table(
            d_w.base, w_stride, n_exp, owned.clone(),
        );
        let stab = crate::orch::moe::packed_expert_table(
            d_s.base, s_stride, n_exp, owned.clone(),
        );

        for e in owned {
            for (j, proj) in PROJ.iter().enumerate() {
                let idx = e as usize * 3 + j;
                for (name, dst, want) in [
                    (format!("{pfx}experts.{e}.{proj}.weight"), wtab[idx], w_stride),
                    (
                        format!("{pfx}experts.{e}.{proj}.weight_scale_inv"),
                        stab[idx],
                        s_stride,
                    ),
                ] {
                    let (src, shape) = ckpt.tensor_ex(&name).ok_or_else(|| {
                        RuntimeError::Device(format!("MISSING EXPERT WEIGHT: {name}"))
                    })?;
                    // `tp = 1` is the EP/single-GPU case: bind the expert whole.
                    // Otherwise the classifier sees `gate_proj.weight` /
                    // `up_proj.weight` (contiguous output-row slice) and
                    // `down_proj.weight` (strided input-column gather), and the
                    // `_scale_inv` grids ride the same substring tests onto the
                    // same axis — which is exactly the C reference's hand-rolled
                    // `j < 2 ? offset : gather_row` split.
                    let slice = crate::asset::shard::slice_for(
                        &name,
                        src,
                        shape,
                        want,
                        if whole { 0 } else { rank },
                        if whole { 1 } else { n_gpu },
                    )?;
                    // Through the PINNED slab, always. `memcpy_htod_pinned`
                    // blocks and does not pin its source, so handing it a
                    // `slice_for` gather buffer (an ordinary `Vec`) faults the
                    // SDMA engine — the one trap the C reference calls out by
                    // name.
                    for (o, chunk) in slice.chunks(stage_bytes).enumerate() {
                        stage.as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
                        be.memcpy_htod_pinned(
                            dst + (o * stage_bytes) as u64,
                            &stage.as_slice()[..chunk.len()],
                        )?;
                    }
                    wbytes += want;
                }
            }
        }
        EngineDevice::upload(be, &devp[*i_ewt], 0, as_bytes(&wtab))?;
        EngineDevice::upload(be, &devp[i_est], 0, as_bytes(&stab))?;
        bufs.push(d_w);
        bufs.push(d_s);
    }
    tracing::info!(
        layers = layers.len(),
        gib = format!("{:.2}", wbytes as f64 / (1u64 << 30) as f64).as_str(),
        i_moe,
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
/// THREE patch families, every one found BY IDENTITY rather than by position:
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
fn rebase_chunk(insts: &mut [DevInst64], names: &[String], c0: u32, clen: u32) {
    for d in insts.iter_mut() {
        let op = d.op;
        if let Some(f) = kv_write_row_field(op, names.get(d.t[0] as usize)) {
            d.i[f] = c0;
        }
        if (op == DevOp::HeadNormRope as u16 || op == DevOp::HeadNormRopeFp8 as u16)
            && d.fj[1] != 0
        {
            d.i[3] = c0;
        } else if op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16 {
            d.i[4] = c0;
            d.i[1] = c0 + clen;
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

/// One program's device-resident tables.
struct AmdProg {
    t: u32,
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
}

struct AmdGq {
    d_stream: DeviceMem,
    d_seg_ofs: DeviceMem,
    /// One cursor LINE per segment, not one word. RUNSEG enqueues every
    /// segment without a host wait and zeroes state once before the loop, so a
    /// shared cursor would be corrupted by the segment that ran first.
    d_cursor: DeviceMem,
    n_seg: u32,
}

/// The AMD serving engine.
pub struct AmdEngine {
    be: Arc<HsaBackend>,
    arch: String,
    n_cu: u32,
    progs: Vec<AmdProg>,
    /// Index of the decode program — always last (`n_prog - 1`).
    decode: usize,
    devp: Vec<DeviceMem>,
    d_tens: DeviceMem,
    tensor_names: Vec<String>,
    /// Per-MoE-layer PACKED expert buffers (weights, then block scales). Never
    /// read through here — the packet reaches them only through the addresses in
    /// `expert_weight_table`/`expert_scale_table` — but they are owning handles,
    /// so dropping them would free the memory those tables point at.
    _expert_bufs: Vec<DeviceMem>,

    k_prefill: HsaKernel,
    k_decode: HsaKernel,
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
        Self::load_rank(be, blob_path, hsaco_dir, checkpoint, None)
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
    /// stop the engine loading. Off by default (`PLOW_VMM_KV=1`).
    ///
    /// Only FULL-attention `kv.{l}.k`/`.v` are backed. Sliding-window rings are
    /// bounded by `window`, not by context, so they have nothing to grow into,
    /// and fp8 scale tensors are 1/128th the size — both stay flat, which is
    /// also what the CUDA path does.
    fn vmm_bringup(be: &Arc<HsaBackend>, blob: &DevBlob, checkpoint: Option<&Path>) -> Option<VmmKv> {
        if std::env::var("PLOW_VMM_KV").as_deref() != Ok("1") {
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
        let block_hint = std::env::var("PLOW_VMM_BLOCK_MIB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|m| m << 20)
            .unwrap_or(0);
        let block_hint = match block_hint {
            0 => VmmOps::granularity(&**be).ok()?,
            b => b,
        };
        match VmmKv::new(Arc::clone(be) as Arc<dyn VmmOps>, geo, block_hint, 0) {
            Ok(kv) => Some(kv),
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
    ) -> Result<Self> {
        let raw = std::fs::read(blob_path).map_err(|e| {
            RuntimeError::Device(format!("read {}: {e}", blob_path.display()))
        })?;
        let blob = DevBlob::parse(&raw)?;
        let arch = EngineDevice::arch(&*be);
        let n_cu_dev = EngineDevice::sm_count(&*be);

        // The blob's n_cu is the grid the schedule was COMPILED for. A device
        // with a different CU count cannot run it: `stream_ofs`/`stream_len` are
        // [n_cu] and workgroup w reads slot w, so a smaller grid silently drops
        // every stream above it and a larger one reads past the table.
        if blob.n_cu != n_cu_dev {
            return Err(RuntimeError::Device(format!(
                "blob compiled for n_cu={} but this device has {n_cu_dev} CUs — \
                 recompile the packet with --n-cu {n_cu_dev}",
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
        let env_flag = |k: &str| std::env::var(k).ok().is_some_and(|v| v != "0");
        let mut sched_prefill = Sched::GlobalQueue;
        let mut sched_decode = Sched::GlobalQueue;
        if let Ok(v) = std::env::var("PLOW_GLOBAL_QUEUE") {
            let s = if v != "0" { Sched::GlobalQueue } else { Sched::Static };
            sched_prefill = s;
            sched_decode = s;
        }
        if env_flag("PLOW_STATIC") {
            sched_prefill = Sched::Static;
            sched_decode = Sched::Static;
        }
        if env_flag("PLOW_STATIC_PREFILL") {
            sched_prefill = Sched::Static;
        }
        if env_flag("PLOW_STATIC_DECODE") {
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
        let dec_ix = blob.progs.len() - 1;
        let need_m_decode = required_gemv_m(&blob.progs[dec_ix..]);
        let need_m_prefill = required_gemv_m(&blob.progs[..dec_ix]);
        // Split the same way as the GEMV bucket, and for the same reason: the K3/KDA arms are in
        // BOTH buckets (a K3 layer runs the same graph at T=1 and T>1), so each object has to be
        // asked about the phase it actually serves rather than about the blob as a whole.
        let need_k3_decode = required_k3_op(&blob.progs[dec_ix..]);
        let need_k3_prefill = required_k3_op(&blob.progs[..dec_ix]);

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
        let mut load_one = |phase: Phase, sched: Sched| -> Result<HsaKernel> {
            let name = object_name(phase, variant, prefill_arm, sched);
            let path = hsaco_dir.join(&name);
            let image = std::fs::read(&path).map_err(|e| {
                if phase == Phase::Prefill && prefill_arm != PrefillArm::None {
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
                            PrefillArm::None => unreachable!(),
                        },
                        hsaco_dir.display(),
                        match prefill_arm {
                            PrefillArm::MlaMoe => "PLOW_MOE_PREFILL=1",
                            PrefillArm::Mla => "PLOW_MLA_PREFILL=1",
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
            // The object's compiled row bucket vs the widest GEMV it will run.
            // Every phase, not just decode: `case PLOW_DOP_GEMV` is unconditional
            // in the prefill bucket too.
            let need = match phase {
                Phase::Decode => need_m_decode,
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
            let m = EngineDevice::module_load(&*be, &image).map_err(|e| {
                RuntimeError::Device(format!(
                    "{name}: {e} — a BUNDLED object gives exactly this; was it \
                     run through clang-offload-bundler --unbundle?"
                ))
            })?;
            let sym = symbol_name(phase, sched, &arch);
            let k = EngineDevice::get_function(&*be, &m, &sym)
                .map_err(|e| RuntimeError::Device(format!("{name}: no symbol {sym}: {e}")))?;
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

        let k_prefill = load_one(Phase::Prefill, sched_prefill)?;
        let k_decode = load_one(Phase::Decode, sched_decode)?;
        // Flash follows the PREFILL scheduler — a flash segment is a prefill
        // segment. Optional: without it every segment runs class 8, which is
        // correct and merely slower.
        let k_flash = match load_one(Phase::Flash, sched_prefill) {
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
        let ckpt = match checkpoint {
            Some(dir) => Some(crate::asset::checkpoint::Checkpoint::open(dir)?),
            None => None,
        };
        // The fp8 weight TWINS live in their own checkpoint, not the bf16 one:
        // they are a separate quantisation artifact, and the packet names them
        // with an `fp8/` prefix that is stripped before lookup. Without this an
        // fp8 packet fails at the first weight with "MISSING WEIGHT", which
        // reads as a broken packet rather than a missing directory.
        let fp8_ckpt = match std::env::var("PLOW_FP8_DIR") {
            Ok(d) => Some(crate::asset::checkpoint::Checkpoint::open(Path::new(&d))?),
            Err(_) => None,
        };
        let mut stage = EngineDevice::host_alloc_pinned(&*be, STAGE)?;
        // v7 blobs carry the RoPE tables as RECIPES, not bytes. Materialising
        // them is not optional: a reader that skips this leaves cos=sin=0 and
        // serves fluent-looking garbage with no error anywhere.
        let gen_by_tensor: std::collections::HashMap<u32, &packet::rope::GenTensor> =
            blob.gen.iter().map(|g| (g.tensor, g)).collect();

        // Must precede the tensor loop: it decides whether each full-layer KV
        // tensor gets an allocation or a view onto the pool's VA reservation.
        let vmm = Self::vmm_bringup(&be, &blob, checkpoint);

        let mut devp = Vec::with_capacity(blob.tensors.len());
        let mut names = Vec::with_capacity(blob.tensors.len());
        let (mut wbytes, mut nweights) = (0u64, 0usize);
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
                _ => None,
            };
            if let Some(base) = peer_slot {
                tracing::debug!(
                    name = %td.name, base = format_args!("{base:#x}"), bytes = td.bytes,
                    "bound into the peer region"
                );
                devp.push(DeviceMem::view(base, td.bytes.max(1)));
                names.push(td.name.clone());
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
            let mem = match vmm_va {
                Some(va) => DeviceMem::view(va, td.bytes.max(1)),
                None => EngineDevice::alloc(&*be, td.bytes.max(1))?,
            };
            let mut push = |src: &[u8]| -> Result<()> {
                for (o, chunk) in src.chunks(STAGE).enumerate() {
                    stage.as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
                    be.memcpy_htod_pinned(
                        mem.base + (o * STAGE) as u64,
                        &stage.as_slice()[..chunk.len()],
                    )?;
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
                // `fp8/` routes to the twin checkpoint with the prefix
                // stripped; everything else to the base one.
                let is_fp8 = td.name.starts_with("fp8/");
                let src_ckpt = if is_fp8 { fp8_ckpt.as_ref() } else { ckpt.as_ref() };
                if let Some(c) = src_ckpt {
                    // BOTH spellings, because the twin checkpoints disagree
                    // with each other. `/home/lava/models/g31b-fp8w` KEEPS the
                    // `fp8/` prefix in its tensor names; the C reference strips
                    // it. Trying the packet's name first and the stripped form
                    // second costs one hash lookup and works with either
                    // convention, which is better than encoding a guess about
                    // which artifact someone hands us.
                    let stripped = td.name.strip_prefix("fp8/").unwrap_or(&td.name);
                    let (src, shape) = c
                        .tensor_ex(&td.name)
                        .or_else(|| c.tensor_ex(stripped))
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
                    let slice = crate::asset::shard::slice_for(
                        stripped, src, shape, td.bytes, rank, n_gpu,
                    )?;
                    push(&slice)?;
                    wbytes += td.bytes;
                    nweights += 1;
                } else if td.name.starts_with("fp8/") {
                    return Err(RuntimeError::Device(format!(
                        "packet declares fp8 weights ({}) but PLOW_FP8_DIR is not set",
                        td.name
                    )));
                }
            } else if let Some(r) = &td.init {
                push(&blob.init[r.clone()])?;
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
                push(&data)?;
            } else if vmm_va.is_none() && !td.name.starts_with("kv.") {
                // A VMM window is (mostly) UNMAPPED VA — a memset would fault,
                // not merely waste time. The `kv.` clause below is the older
                // and independent reason to skip.
                //
                // Attention reads only [0, kvlen), every row of which is
                // written before it is read, so the KV cache needs no zeroing —
                // 11.5 GiB of memset skipped on this model. Other scratch stays
                // zeroed: cheap, and conservative where the argument is less
                // obviously airtight.
                EngineDevice::memset_d8(&*be, mem.base, 0, td.bytes as usize)?;
            }
            devp.push(mem);
            names.push(td.name.clone());
        }
        // The MoE half of the bind, and it has to be here: it needs the tensor
        // table (to find each layer's two pointer slots) and the pinned stage
        // (which must outlive it — the C reference records that gathering a
        // row-parallel slice into a MALLOC'd buffer faults the SDMA engine,
        // because the blocking copy does not pin its source).
        let mut expert_bufs = Vec::new();
        if let Some(c) = ckpt.as_ref() {
            let (bufs, bytes) = bind_packed_experts(
                &be, &blob, c, &devp, &names, &mut stage, STAGE, rank, n_gpu,
            )?;
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
        drop(stage);
        if ckpt.is_some() {
            tracing::info!(
                gib = format!("{:.2}", wbytes as f64 / (1u64 << 30) as f64).as_str(),
                tensors = nweights,
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
        let t_logits = find("act.logits");
        // The context bound is carried by in.pos, not by any prefill bucket.
        let max_ctx = t_pos.map_or(0, |t| (blob.tensors[t].bytes / 4) as usize);

        // --- per-program tables ---------------------------------------------
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
            let d_ctr = EngineDevice::alloc(
                &*be,
                (p.n_counter as usize * CTR_STRIDE_U32 * 4).max(1) as u64,
            )?;
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
                Some(AmdGq {
                    d_stream: up(as_bytes(&p.gq_stream))?,
                    d_seg_ofs: up(as_bytes(&p.gq_seg_ofs))?,
                    d_cursor: EngineDevice::alloc(
                        &*be,
                        (n_seg.max(1) as usize * CTR_STRIDE_U32 * 4) as u64,
                    )?,
                    n_seg,
                })
            };
            progs.push(AmdProg {
                t: p.t,
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
            });
        }
        let decode = progs.len() - 1;
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
        if batch > GEMV_MAXM as usize {
            return Err(RuntimeError::Device(format!(
                "blob is compiled PLOW_DECODE_BATCH={batch}, but the gfx950 decode GEMV \
                 is a compile-time row bucket capped at PLOW_GEMV_MAXM={GEMV_MAXM} \
                 (runtime/amd/op_gemm.h). Sequences {GEMV_MAXM}.. would get zero logits. \
                 Re-emit at PLOW_DECODE_BATCH <= {GEMV_MAXM}."
            )));
        }

        let mut kv_slot_stride: Vec<(usize, u64)> = Vec::new();
        if batch > 1 {
            kv_slot_stride = blob
                .tensors
                .iter()
                .enumerate()
                .filter(|(_, t)| t.name.starts_with("kv."))
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
        let pf_src: Vec<Vec<DevInst64>> =
            blob.progs.iter().map(|g| g.insts.clone()).collect();
        let max_ctr = progs
            .iter()
            .map(|p| p.n_counter as usize * CTR_STRIDE_U32 * 4)
            .max()
            .unwrap_or(4)
            .max(progs.iter().filter_map(|p| p.gq.as_ref()).map(|g| {
                g.n_seg.max(1) as usize * CTR_STRIDE_U32 * 4
            }).max().unwrap_or(4));
        let mut h_zero = EngineDevice::host_alloc_pinned(&*be, max_ctr.max(4))?;
        h_zero.as_mut_slice().fill(0);

        let (kvrow, kvrow_i2) = if blob.kvrow.is_empty() {
            derive_kvrow(&blob.progs[decode], &names)
        } else {
            (blob.kvrow.clone(), Vec::new())
        };
        let kvrow_span =
            kvrow_span(&kvrow.iter().chain(&kvrow_i2).copied().collect::<Vec<_>>());

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

        Ok(AmdEngine {
            be,
            arch,
            n_cu: blob.n_cu,
            progs,
            decode,
            devp,
            d_tens,
            tensor_names: names,
            _expert_bufs: expert_bufs,
            k_prefill,
            k_decode,
            k_flash,
            sched_prefill,
            sched_decode,
            _modules: modules,
            h_inst,
            h_scalar,
            h_zero,
            h_pf_inst,
            pf_src,
            kvrow,
            kvrow_i2,
            kvrow_span,
            t_ids,
            t_pos,
            t_kvlen,
            t_logits,
            max_ctx,
            weights_bound: ckpt.is_some(),
            batch,
            tens_table: table,
            kv_slot_stride,
            kv_slot: 0,
            vmm,
            lm_detail: std::cell::RefCell::new(None),
            pf_stream: blob.progs.iter().map(|g| g.stream.clone()).collect(),
            tp,
            seg_enq_us: 0.0,
            seg_drain_us: 0.0,
            seg_launches: 0,
            seg_window: std::env::var("PLOW_SEG_WINDOW").as_deref() != Ok("0"),
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
                let ops = d
                    .t
                    .iter()
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
            counters: g.d_ctr.base,
            tensors: self.d_tens.base,
            trace: 0,
            cur_seg: seg,
            l2_domains: g.l2_domains,
            n_seg: g.seg_class.len() as u32,
            // Static-path segment windows. Set for every program: an unsegmented
            // one has a single window covering the whole stream, so the decode
            // path does exactly what the old full scan did.
            seg_ofs: if self.seg_window { g.d_seg_ofs.base } else { 0 },
            // Set unconditionally: they are 0 without a GQ appendix, and the
            // static kernel never reads them, so one path serves both.
            gq_stream: g.gq.as_ref().map_or(0, |q| q.d_stream.base),
            gq_seg_ofs: g.gq.as_ref().map_or(0, |q| q.d_seg_ofs.base),
            gq_cursor: g.gq.as_ref().map_or(0, |q| q.d_cursor.base),
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
        let g = &self.progs[p];
        let n = g.n_counter as usize * CTR_STRIDE_U32 * 4;
        if n > 0 {
            self.be
                .memcpy_htod_pinned(g.d_ctr.base, &self.h_zero.as_slice()[..n])?;
        }
        if let Some(q) = &g.gq {
            let n = q.n_seg.max(1) as usize * CTR_STRIDE_U32 * 4;
            self.be
                .memcpy_htod_pinned(q.d_cursor.base, &self.h_zero.as_slice()[..n])?;
        }
        Ok(())
    }

    /// Re-arm program `p`'s counters and cursor. Public so a TP driver can
    /// re-arm EVERY rank before dispatching ANY of them (§6d).
    pub fn rearm_prog(&self, p: usize) -> Result<()> {
        self.rearm(p)
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
    /// captured its own copy.
    pub fn enqueue_segment(&mut self, p: usize, seg: usize) -> Result<()> {
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
        self.drain()?;
        let t2 = std::time::Instant::now();
        self.seg_enq_us += (t1 - t0).as_secs_f64() * 1e6;
        self.seg_drain_us += (t2 - t1).as_secs_f64() * 1e6;
        crate::obs::ttft::PF_SEGMENTS.tally(n_seg as u64);
        crate::obs::ttft::PF_ENQUEUE.add((t1 - t0).as_nanos() as u64);
        crate::obs::ttft::PF_DRAIN.add((t2 - t1).as_nanos() as u64);
        Ok(())
    }

    /// Single-launch run — the decode path, which is not segmented.
    pub fn run(&mut self, p: usize, k: HsaKernel) -> Result<()> {
        self.rearm(p)?;
        let t0 = std::time::Instant::now();
        self.enqueue(p, k)?;
        self.drain()?;
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
            let insts: &mut [DevInst64] = unsafe {
                std::slice::from_raw_parts_mut(slab.as_mut_ptr() as *mut DevInst64, n)
            };
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
        self.vmm_ensure(self.kv_slot, pos + 1)?;
        self.decode_prepare(pos, kvlen)?;
        self.run(self.decode, self.k_decode)?;
        let id = self.read_sampled()?;
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
        self.be.memcpy_htod_pinned(ptr_pos, &self.h_scalar.as_slice()[..4])?;
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

        rebase_chunk(insts, &self.tensor_names, c0, clen);

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
            // DIAGNOSTIC: `PLOW_LM_ROW0=1` leaves a_row0 at 0 instead of the
            // chunk's last real row. It samples the WRONG row, so it is not a
            // serving mode — it exists to answer one question. The lm_head is
            // the ONLY op whose a_row0 the host patches to a non-zero value at
            // runtime, so a bug in the a_row0 path is invisible to any check
            // that inspects the packet statically (where all fp8 GEMMs carry
            // a_row0 == 0) and shows up only here.
            (Some(lm), _) if std::env::var("PLOW_LM_ROW0").as_deref() == Ok("1") => {
                tracing::warn!(lm, "PLOW_LM_ROW0=1: a_row0 left at 0 — DIAGNOSTIC, wrong row");
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
            let prog = (0..self.decode)
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
        if let Some(t) = self.t_kvlen {
            let d_kvlen = self.devp[t].base;
            self.h_scalar.as_mut_slice()[..4].copy_from_slice(&(step.c0 + ch).to_le_bytes());
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
        let buckets: Vec<u32> = (0..self.decode).map(|p| self.progs[p].t).collect();
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
    fn prefill_rows(&self, n_prompt: u32) -> u32 {
        let cover: u32 = self
            .plan_for(n_prompt)
            .map(|c| c.iter().sum())
            .unwrap_or(n_prompt);
        cover.max(n_prompt).min(self.max_ctx as u32)
    }

    /// Map physical backing for `seq` out to `rows`. No-op without VMM.
    fn vmm_ensure(&self, seq: usize, rows: u32) -> Result<()> {
        match &self.vmm {
            Some(v) if v.mapped_rows(seq) < rows => v.ensure_rows(seq, rows),
            _ => Ok(()),
        }
    }

    /// Release slot `seq`'s physical backing and remap its row 0.
    ///
    /// Called when a slot is handed to a NEW sequence: the outgoing sequence's
    /// blocks are what a growable pool exists to reclaim. Row 0 goes straight
    /// back because an idle row still writes KV at `pos = 0`.
    pub fn begin_slot(&mut self, seq: usize) -> Result<()> {
        if let Some(v) = &self.vmm {
            v.begin_seq(seq);
            v.ensure_rows(seq, 1)?;
        }
        Ok(())
    }

    /// Pool counters (`blocks_live` is the HBM the KV cache actually holds).
    pub fn vmm_stats(&self) -> Option<crate::memory::vmm::VmmStats> {
        self.vmm.as_ref().map(|v| v.stats())
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

        let dp = self.decode;
        if b == 1 {
            self.patch_kvrow(dp, pos[0])?;
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

        self.run(dp, self.k_decode)?;

        let src = self.devp[self.need(self.t_ids, "in.ids")?].base;
        let slab = self.h_scalar.as_mut_slice();
        self.be.memcpy_dtoh_pinned(&mut slab[..b * 4], src)?;

        // Hand the pre-mapper the new frontier so the next block is mapped
        // BEFORE a step needs it. Never blocks; `vmm_ensure` above is the
        // correctness backstop if it falls behind.
        if let Some(v) = &self.vmm {
            for (i, &p) in pos.iter().enumerate() {
                v.advise(i, p + 1);
            }
        }
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

    /// Whether model weights were bound at load. A `false` here means the
    /// timings are real and the tokens are not.
    pub fn weights_bound(&self) -> bool {
        self.weights_bound
    }

    /// The decode program's index, for callers that want [`AmdEngine::run`].
    pub fn decode_prog(&self) -> usize {
        self.decode
    }

    /// The decode kernel handle.
    pub fn decode_kernel(&self) -> HsaKernel {
        self.k_decode
    }

    /// Per-program compiled `T` (decode is 1).
    pub fn prog_t(&self, p: usize) -> u32 {
        self.progs[p].t
    }

    /// Segment count for program `p`.
    pub fn prog_segments(&self, p: usize) -> usize {
        self.progs[p].seg_class.len()
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

    /// The flash object follows the PREFILL scheduler: a flash segment IS a
    /// prefill segment. Pairing it with the decode choice loads an object whose
    /// scheduling loop does not match the stream it is handed.
    #[test]
    fn object_names_match_the_shipped_set() {
        assert_eq!(
            object_name(Phase::Prefill, Variant::Bf16, PrefillArm::None, Sched::GlobalQueue),
            "interp_prefill_gq.elf"
        );
        assert_eq!(
            object_name(Phase::Decode, Variant::Bf16, PrefillArm::None, Sched::Static),
            "interp_decode.elf"
        );
        assert_eq!(
            object_name(Phase::Decode, Variant::Fp8Kv, PrefillArm::None, Sched::GlobalQueue),
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
            object_name(Phase::Flash, Variant::Fp8Kv, PrefillArm::None, Sched::GlobalQueue),
            "interp_flash_fp8kv_gq.elf"
        );
        // The MLA/MoE-prefill axis is PREFILL-only — no decode or flash twin
        // exists (`scripts/build_gfx950.sh` never builds one) — so a non-None
        // arm on those phases must not leak into the filename.
        assert_eq!(
            object_name(Phase::Prefill, Variant::Bf16, PrefillArm::Mla, Sched::GlobalQueue),
            "interp_prefill_mla_gq.elf"
        );
        assert_eq!(
            object_name(Phase::Prefill, Variant::Bf16, PrefillArm::Mla, Sched::Static),
            "interp_prefill_mla.elf"
        );
        assert_eq!(
            object_name(Phase::Prefill, Variant::Bf16, PrefillArm::MlaMoe, Sched::GlobalQueue),
            "interp_prefill_mla_moe_gq.elf"
        );
        assert_eq!(
            object_name(Phase::Prefill, Variant::Bf16, PrefillArm::MlaMoe, Sched::Static),
            "interp_prefill_mla_moe.elf"
        );
        assert_eq!(
            object_name(Phase::Decode, Variant::Bf16, PrefillArm::MlaMoe, Sched::Static),
            "interp_decode.elf"
        );
        assert_eq!(
            object_name(Phase::Flash, Variant::Bf16, PrefillArm::MlaMoe, Sched::Static),
            "interp_flash.elf"
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
                .map(|&op| DevInst64 { op: op as u16, ..Default::default() })
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
    }

    /// A prog carrying `ops`, each instruction asking for `m` GEMV rows.
    fn prog_gemv(ops: &[DevOp], m: u32) -> DevProg {
        let insts = ops
            .iter()
            .map(|&op| {
                let mut i = [0u32; 8];
                i[0] = m;
                DevInst64 { op: op as u16, i, ..Default::default() }
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
            assert_eq!(required_k3_op(&pkt), Some(op), "{op:?} must be recognised as a K3 arm");

            let e = check_k3_arms(&bare, obj, required_k3_op(&pkt))
                .expect_err("a K3 op against an object with no K3 arms must be refused");
            let msg = e.to_string();
            // The refusal has to name the op, the marker, and the remedy — the object is not on a
            // device yet, so this message is the only thing the operator gets.
            assert!(msg.contains(&format!("{op:?}")), "must name the op: {msg}");
            assert!(msg.contains(K3_ARMS_SYM), "must name the missing marker: {msg}");
            assert!(msg.contains("PLOW_K3"), "must name the flag to rebuild with: {msg}");

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
        let mut found: Vec<String> = Vec::new();
        for line in src.lines() {
            let t = line.trim();
            if t == "#if PLOW_K3" {
                in_region = true;
                found.clear(); // a later region supersedes an earlier one
                continue;
            }
            if in_region && t.starts_with("#endif") {
                in_region = false;
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

        let mut want: Vec<String> =
            K3_ARM_OPS.iter().map(|o| o.c_name().to_string()).collect();
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
                assert!(msg.contains(&format!("PLOW_DECODE_BATCH <= {GEMV_MAXM}")), "{msg}");
            } else {
                assert!(msg.contains(&format!("PLOW_DECODE_BATCH={}", cap + 1)), "{msg}");
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
        let progs = vec![prog_gemv(&[DevOp::Gemv], 1), prog_gemv(&[DevOp::GemvGlu], 8)];
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
        assert_eq!(plan_chunks(&bkt, 1536).unwrap(), vec![1024, 512]);
        assert_eq!(plan_chunks(&bkt, 1024).unwrap(), vec![1024]);
        assert_eq!(plan_chunks(&bkt, 1).unwrap(), vec![128]);
        assert_eq!(plan_chunks(&bkt, 0).unwrap(), Vec::<u32>::new());

        // Descending, so the ragged chunk is the LAST one — padding lands in
        // the tail, where a padded row writes KV that `n_kv` bounds out.
        for n in [200u32, 700, 1300, 4000, 9000] {
            let plan = plan_chunks(&bkt, n).unwrap();
            assert!(
                plan.windows(2).all(|w| w[0] >= w[1]),
                "plan for {n} is not largest-first: {plan:?}"
            );
            let covered: u32 = plan.iter().sum();
            assert!(covered >= n, "plan for {n} covers only {covered}: {plan:?}");
        }
    }

    /// A bucket past the sliding-window ring's capacity cannot be used, and a
    /// blob with nothing under the cap is a decode-only blob, not a silent
    /// zero-chunk prefill.
    #[test]
    fn chunk_plan_rejects_a_ladder_it_cannot_use() {
        assert!(plan_chunks(&[], 10).is_err());
        assert!(plan_chunks(&[MAX_CHUNK * 2], 10).is_err());
        // The oversized bucket is filtered, the usable one still works.
        assert_eq!(plan_chunks(&[128, MAX_CHUNK * 2], 128).unwrap(), vec![128]);
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

        rebase_chunk(&mut insts, &names, 512, 128);

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

    /// The dense-GQA rules are unchanged, and the two KV tests are a UNION: a
    /// `kv.*` destination and a non-zero `fj[1]` both mark the same site.
    #[test]
    fn rebase_chunk_still_patches_the_dense_gqa_families() {
        let names: Vec<String> = ["kv.0.k", "act.q"].iter().map(|s| s.to_string()).collect();
        let mut insts = vec![
            // k norm: `kv.*` AND j[0] = ring stride, both tests fire, one field.
            DevInst64 { op: DevOp::HeadNormRopeFp8 as u16, t: [0; 8], fj: [0, 4096, 0], ..Default::default() },
            // q norm: neither test fires.
            DevInst64 { op: DevOp::HeadNormRope as u16, t: [1, 0, 0, 0, 0, 0, 0, 0], ..Default::default() },
            DevInst64 { op: DevOp::FlashPrefillFp8 as u16, t: [1, 0, 0, 0, 0, 0, 0, 0], ..Default::default() },
        ];
        rebase_chunk(&mut insts, &names, 1024, 512);
        assert_eq!(insts[0].i[3], 1024);
        assert_eq!(insts[1].i, [0; 8], "the query norm was patched");
        assert_eq!(insts[2].i[4], 1024, "q_pos0");
        assert_eq!(insts[2].i[1], 1536, "n_kv is everything written so far");
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
        assert!(check_prefill_object(&junk, Path::new("x.elf"), &["PLOW_MLA_PREFILL=1".into()])
            .is_ok());
        // The GEMV capacity check reads the SAME empty list and reaches the
        // opposite conclusion, on purpose: an arm it cannot see may simply be
        // one this packet does not need, but a bucket it cannot see is a bucket
        // that is almost certainly the default 1.
        assert!(check_gemv_capacity(&junk, Path::new("x.elf"), 1).is_ok());
        assert!(check_gemv_capacity(&junk, Path::new("x.elf"), 2).is_err());
    }
}
